#!/usr/bin/env python

from __future__ import annotations

import collections
import json
import os
import struct
from pathlib import Path

import argparse
import numpy as np
from PIL import Image

import viewer_assets as VA
import viewer_solid as VS

ENC_LINE = "WebP q90: inside the tissue mask the grey differs from lossless by 2.0 on average, 6 at p99, 16 at worst (out of 255). The alpha mask is pixel-identical."
GREY_LINE = "grey was scaled per section by that section's own 1st/99th percentile inside its mask (measured: p1 in 0-3, p99 in 229-248). Grey values are NOT comparable between planes. Window and gamma are a display mapping, not a correction."
def tissue_frac(vol: Path, meta: dict) -> dict:
    w, h = meta["canvas_px"]["width"], meta["canvas_px"]["height"]
    out = {}
    for b in meta["encodings"]:
        planes = meta["encodings"][b].get("planes") or []
        f = vol / b / "raw/alpha.u8"
        a = np.fromfile(f, np.uint8) if f.exists() else None
        if a is not None and planes and len(a) == len(planes) * w * h:
            out[b] = [round(float(x), 5) for x in (a.reshape(len(planes), h * w) > 0).mean(1)]
            continue


        vals = []
        for q in planes:
            im = np.asarray(Image.open(vol / q["webp_lossless"]).convert("RGBA"))[..., 3]
            vals.append(round(float((im > 0).mean()), 5))
        if vals:
            out[b] = vals
    return out

def _canvas_px(vol: Path):
    m = json.loads((vol / "metadata.json").read_text())
    return (m["canvas_px"]["width"], m["canvas_px"]["height"])


def on_canvas(name: str, paths, want) -> bool:
    paths = list(paths)
    got = collections.Counter()
    for p in paths:
        try:
            got[Image.open(p).size] += 1
        except Exception:
            got["unreadable"] += 1
    if got.get("unreadable"):


        print(f"  WARNING: {name}: {got['unreadable']} of {len(paths)} files cannot be "
              f"decoded (empty or corrupt); those planes will be blank", flush=True)


    def _ok(k):
        if k == want:
            return True
        if not isinstance(k, tuple):
            return False
        fx, fy = k[0] / want[0], k[1] / want[1]
        return fx == fy and fx == int(fx) and int(fx) >= 1
    bad = {k: v for k, v in got.items() if not _ok(k) and k != "unreadable"}
    fine = {k: v for k, v in got.items() if k != want and _ok(k)}
    if fine:
        for k, v in fine.items():
            print(f"  {name}: {v} files at {k[0]}x{k[1]}, {k[0] // want[0]}x the canvas "
                  f"pitch (same quad, finer line)", flush=True)
    if bad:
        print(f"  WARNING: {name}: {sum(bad.values())} of {len(paths)} files are not on the "
              f"volume canvas {want[0]}x{want[1]} ("
              + ", ".join(f"{k}: {v}" for k, v in bad.items())
              + "); NOT published -- rebuild that layer on the current canvas", flush=True)
        return False
    return True


def layer_rasters(vol: Path, layer, skip="/xenium_gt/") -> list:
    names = set()

    def walk(o):
        if isinstance(o, str):
            if o.endswith(".webp"):
                names.add(Path(o).name)
        elif isinstance(o, dict):
            for v in o.values():
                walk(v)
        elif isinstance(o, list):
            for v in o:
                walk(v)
    walk(layer)
    if not names:
        return []
    return [p for p in vol.rglob("*.webp")
            if p.name in names and (not skip or skip not in str(p))]


def _stamp(*paths) -> int:
    return max([int(Path(q).stat().st_mtime) for q in paths if Path(q).exists()] or [0])


def _tiles_stamp(d: Path) -> int:
    idx = list(d.glob("*/*/index.json"))
    return max([int(q.stat().st_mtime) for q in idx] or [int(d.stat().st_mtime)])


ROOT = Path(os.environ.get("THREED_ROOT", os.path.join(os.environ.get("PROJECTS_ROOT", "/data/heid"), "3d")))
OUTPUTS = ROOT / "reconstruction"
SHARED = ROOT / "reconstruction"
OUT = OUTPUTS / "viewer_server"
REL = ".."
HTML = OUT / "index.html"

RES = [("8", OUTPUTS / "volume_8um"), ("4", OUTPUTS / "volume_4um")]
SAMPLE = "HT891Z1"
TILES = ""

SHOWN = ["HT891Z1", "HT913Z1", "S22-27909", "S16-38794", "S24-4476", "HT206B1"]
REGISTERED = {"HT891Z1": "../viewer_server/index.html",
              "HT913Z1": "../viewer_server_913/index.html",
              "S22-27909": "../viewer_server_s22/index.html",
              "S16-38794": "../viewer_server_s16/index.html"}
DEFAULT_RES = "8"
SHORT = {"G_withdrawn": "G withdrawn", "G_not_withdrawn": "G not withdrawn"}


CELL_VOL = "volume_8um"


def cell_layer():
    d = OUTPUTS / CELL_VOL
    mp = d / "cells_metadata.json"
    if not mp.exists():
        print("  no cell-prediction layer (cells_metadata.json absent)")
        return None
    cm = json.loads(mp.read_text())
    want = _canvas_px(d)
    if (cm["canvas_px"]["width"], cm["canvas_px"]["height"]) != want:
        print(f"  WARNING: cell prediction layer declares canvas {cm['canvas_px']}, the volume is "
              f"{want[0]}x{want[1]}; NOT published -- rebuild the cell planes", flush=True)
        return None

    if not on_canvas("cell prediction layer",
                     (d / q["file"] for q in cm["planes"]
                      if q.get("file") and q.get("cell_class") is not None), want):
        return None
    stem, avail = {}, {}
    for q in cm["planes"]:
        b, i = q["basis"], q["index"]
        key = q["cell_class"]
        if key is None:
            continue
        st = Path(q["file"]).name.split(".")[0]
        if str(i) in stem and stem[str(i)] != st:
            raise SystemExit(f"cells: plane {i} has two stems {stem[str(i)]} / {st}")
        stem[str(i)] = st
        want = f"{b}/cells/{st}.{key}.webp"
        if q["file"] != want:
            raise SystemExit(f"cells: {q['file']} does not match the pattern {want}; "
                             f"the page builds its URLs from that pattern")
        avail.setdefault(b, {}).setdefault(key, []).append(i)
    ch = cm["checks"]
    inm = sorted(c["in_mask"] for c in ch)
    far = sorted(c["control_far_section"] for c in ch)
    worst = min(ch, key=lambda c: c["in_mask"])
    out = {
        "dir": f"{REL}/{CELL_VOL}",
        "v": int(mp.stat().st_mtime),


        "canvas_mm": cm["canvas_mm"], "canvas_px": cm["canvas_px"],
        "classes": cm["classes"],
        "default_on": cm["default_on"],
        "with": cm["indices_with"],
        "n_swap": sum(1 for s in cm["sections"] if s["route"] == "swap"),


        "xenium": [s["index"] for s in cm["sections"]
                   if s["route"] == "xenium_pointcloud"],
        "n_xenium": sum(1 for s in cm["sections"]
                        if s["route"] == "xenium_pointcloud"),
        "stem": stem, "avail": avail,
        "checks": {"in_mask_min": inm[0], "in_mask_median": inm[len(inm) // 2],
                   "control_far_median": far[len(far) // 2],
                   "worst_section": worst["section_id"] + " / " + worst["basis"]},
        "claim": cm["claim"],
    }
    n = len(cm["planes"])
    print(f"  cell prediction: {n} textures, {len(cm['classes'])} classes, "
          f"{len(cm['indices_with'])} planes, "
          f"{out['n_xenium']} of them placed by point-cloud fit, "
          f"worst in_mask {inm[0]:.3f} ({worst['section_id']})")
    return out


def xenium_gt():
    d = OUTPUTS / CELL_VOL / "xenium_gt"
    mp = d / "xenium_gt_metadata.json"
    if not mp.exists():
        print("  no Xenium annotation layer")
        return None
    g = json.loads(mp.read_text())
    want = _canvas_px(OUTPUTS / CELL_VOL)
    if (g["canvas_px"]["width"], g["canvas_px"]["height"]) != want:
        print(f"  WARNING: Xenium annotation layer declares canvas {g['canvas_px']}, the volume is "
              f"{want[0]}x{want[1]}; NOT published -- rebuild it", flush=True)
        return None
    if not on_canvas("Xenium annotation layer", d.rglob("*.webp"), want):
        return None
    files = {}
    for q in g["files"]:
        files.setdefault(q["basis"], {})[str(q["index"])] = q["file"]


    stem, avail = {}, {}
    for q in (g.get("density") or {}).get("planes", []):
        b, i = q["basis"], q["index"]
        st = Path(q["file"]).name.split(".")[0]
        if str(i) in stem and stem[str(i)] != st:
            raise SystemExit(f"xenium_gt: plane {i} has two stems")
        stem[str(i)] = st
        want = f"{b}/cells/{st}.{q['cell_class'].replace('/', '_')}.webp"
        if q["file"] != want:
            raise SystemExit(f"xenium_gt: {q['file']} does not match {want}")
        avail.setdefault(b, {}).setdefault(q["cell_class"], []).append(i)
    n = sum(q["n_cells"] for q in g["files"])
    print(f"  Xenium annotation: {g['n_planes']} planes, {len(g['classes'])} classes, "
          f"{n:,} placed cells ({len(g['xenium_only'])} classes the model has no column for)")
    tls = None
    tp = d / "xenium_tls_metadata.json"
    if tp.exists():
        t = json.loads(tp.read_text())
        tls = {"colour": t["colour"], "planes": t["planes"],
               "v": int(tp.stat().st_mtime)}
        print(f"  Xenium TLS layer: {len(t['planes'])} planes, "
              f"{sum(q['n_tls'] for q in t['planes'])} regions")
    return {"dir": f"{REL}/{CELL_VOL}/xenium_gt", "v": int(mp.stat().st_mtime), "classes": g["classes"], "tls": tls,
            "counts": g.get("counts", {}),
            "stem": stem, "avail": avail,
            "colours": g["colours"], "files": files,
            "shared_with_model": g["shared_with_model"],
            "xenium_only": g["xenium_only"],
            "sub_px": g["sub_px"], "canvas_px": g["canvas_px"],
            "planes": g["planes"], "claim": g["claim"]}


def nerve_regions():
    mp = OUTPUTS / CELL_VOL / "nerve_regions_metadata.json"
    if not mp.exists():
        print("  no denoised nerve layer (nerve_regions_metadata.json absent)")
        return None
    m = json.loads(mp.read_text())
    print(f"  denoised nerve layer: {m['n_planes']} planes")


    import numpy as np
    from PIL import Image
    from skimage.measure import find_contours, approximate_polygon
    from scipy import ndimage as ndi
    for plane in m["planes"]:
        image_path = OUTPUTS / CELL_VOL / "G_withdrawn" / "nerve" / plane["file"]
        if not image_path.exists():
            continue
        inside = np.asarray(Image.open(image_path).convert("RGBA"))[:, :, 3] > 0
        for region in plane.get("regions", []):


            if region.get("polygons_um"):
                continue
            x0, y0, x1, y1 = region["bbox_um"]
            left, top = max(0, int(x0 / 8) - 2), max(0, int(y0 / 8) - 2)
            right, bottom = min(inside.shape[1], int(x1 / 8) + 3), min(inside.shape[0], int(y1 / 8) + 3)
            sub = np.pad(inside[top:bottom, left:right], 1)


            components, count = ndi.label(sub)
            if count:
                sizes = np.bincount(components.ravel()); sizes[0] = 0
                component = components == sizes.argmax()
                y, x = np.unravel_index(ndi.distance_transform_edt(component).argmax(), sub.shape)
                region["label_anchor_um"] = [(int(x) + left - .5) * 8, (int(y) + top - .5) * 8]
            region["polygons_um"] = [
                [[round(float((x + left - .5) * 8), 2), round(float((y + top - .5) * 8), 2)]
                 for y, x in approximate_polygon(contour, tolerance=.25)]
                for contour in find_contours(sub.astype(float), .5) if len(contour) >= 4]
    return {"dir": f"{REL}/{CELL_VOL}", "colour": m.get("colour", "#00B0F0"),
            "vector_contours": True,
            "v": int(mp.stat().st_mtime),
            "planes": [{"index": r["index"], "file": r["file"], "regions": r.get("regions", [])} for r in m["planes"]]}


def nerve_lines():
    mp = OUTPUTS / CELL_VOL / "nerve_lines_metadata.json"
    if not mp.exists():
        print("  no nerve guide-line layer (nerve_lines_metadata.json absent)")
        return None
    m = json.loads(mp.read_text())
    print(f"  nerve guide lines: {m['n_planes']} planes")
    return {"dir": f"{REL}/{CELL_VOL}", "colour": m["colour"], "v": int(mp.stat().st_mtime),
            "planes": [{"index": r["index"], "file": r["file"]} for r in m["planes"]]}


def tumor_regions():
    mp = OUTPUTS / CELL_VOL / "tumor_regions_metadata.json"
    if not mp.exists():
        print("  no tumour territory layer (tumor_regions_metadata.json absent)")
        return None
    m = json.loads(mp.read_text())
    print(f"  tumour territory layer: {m['n_planes']} planes")
    return {"dir": f"{REL}/{CELL_VOL}", "colour": m["colour"], "v": int(mp.stat().st_mtime),
            "planes": [{"index": r["index"], "file": r["file"], "area_mm2": r["area_mm2"]} for r in m["planes"]]}


def duct_regions():
    mp = OUTPUTS / CELL_VOL / "duct_regions_metadata.json"
    if not mp.exists():
        print("  no duct lumen layer (duct_regions_metadata.json absent)")
        return None
    m = json.loads(mp.read_text())
    print(f"  duct lumen layer: {m['n_planes']} planes")
    return {"dir": f"{REL}/{CELL_VOL}", "colour": m["colour"], "v": int(mp.stat().st_mtime),
            "planes": [{"index": r["index"], "file": r["file"], "regions": r["regions"]} for r in m["planes"]]}


def gland_regions():
    import os
    obj = os.environ.get('HTAN3D_OBJ3D')
    root = Path(os.environ['HTAN3D_GLANDS']) if os.environ.get('HTAN3D_GLANDS') else (Path(obj) / 'S16_tumor_glands' if obj else None)
    if root is None or not (root / 'gland_regions_metadata.json').exists():
        return None
    mp = root / 'gland_regions_metadata.json'
    m = json.loads(mp.read_text())
    assert m['sample'] == SAMPLE
    paths = [root / b / 'gland' / f for b in m['bases'] for p in m['planes'] for f in p['files'].values()]
    valid = bool(paths) and all(p.is_file() for p in paths)
    if valid and m.get('raster_mode') == 'roi':
        from PIL import Image
        canvas = _canvas_px(OUTPUTS / CELL_VOL)
        valid = (m['canvas_px']['width'],m['canvas_px']['height']) == canvas
        physical = json.loads((OUTPUTS/CELL_VOL/'metadata.json').read_text())['canvas_mm']
        for plane in m['planes']:
            for oid, name in plane['files'].items():
                x0,y0,x1,y1=plane['raster_bounds_um'][oid]
                expected=(round((x1-x0)/m['raster_mpp_um']),round((y1-y0)/m['raster_mpp_um']))
                valid &= 0<=x0<x1<=physical['width']*1000 and 0<=y0<y1<=physical['height']*1000
                for basis in m['bases']:
                    with Image.open(root/basis/'gland'/name) as im: valid &= im.size == expected
    elif valid:
        valid = on_canvas('tumor glands', paths, _canvas_px(OUTPUTS / CELL_VOL))
    if not valid:
        raise SystemExit('Tumor gland regions do not match the current canvas')
    m.update(dir=REL + '/' + os.path.relpath(root, OUTPUTS), v=int(mp.stat().st_mtime))
    return m


def tls_regions():
    mp = OUTPUTS / CELL_VOL / "tls_regions_metadata.json"
    if not mp.exists():
        print("  no TLS region layer (tls_regions_metadata.json absent)")
        return None
    m = json.loads(mp.read_text())
    print(f"  TLS region layer: {m['n_planes']} planes")
    return {"dir": f"{REL}/{CELL_VOL}", "colour3d": m["colour3d"], "colour2d": m["colour2d"],
            "v": int(mp.stat().st_mtime),
            "planes": [{"index": r["index"], "file3d": r["file3d"], "file2d": r["file2d"],
                        "regions": r["regions"]} for r in m["planes"]]}


def cells_denoised3d():
    mp = OUTPUTS / CELL_VOL / "cells_denoised3d_metadata.json"
    if not mp.exists():
        print("  no 3-D denoised layer (cells_denoised3d_metadata.json absent)")
        return None
    m = json.loads(mp.read_text())
    print(f"  3-D denoised layer: {len(m['classes'])} classes")
    return {"dir": f"{REL}/{CELL_VOL}", "classes": m["classes"],
            "v": int(mp.stat().st_mtime), "files": m["files"]}


def cell_points():
    d = OUTPUTS / CELL_VOL
    mp = d / "cell_points_metadata.json"
    if not mp.exists():
        print("  no per-cell points (cell_points_metadata.json absent)")
        return None
    cm = json.loads(mp.read_text())


    want = _canvas_px(d)
    got = collections.Counter()
    for q in cm["files"]:
        with open(d / q["file"], "rb") as fh:
            h = fh.read(24)
        got[tuple(struct.unpack("<8sIHHHHI", h)[2:4])] += 1
    bad = {k: v for k, v in got.items() if k != want}
    if bad:
        print(f"  WARNING: per-cell points: {sum(bad.values())} of {len(cm['files'])} files are "
              f"packed on another canvas than the volume's {want[0]}x{want[1]} ({bad}); "
              f"NOT published -- rebuild the points", flush=True)
        return None
    files = {}
    for q in cm["files"]:
        files.setdefault(q["basis"], {})[str(q["index"])] = q["file"]
    inm = sorted(c["in_mask"] for c in cm["checks"])
    sh = sorted(c["control_shift_200px"] for c in cm["checks"])
    print(f"  per-cell points: {len(cm['files'])} files, "
          f"{cm['bytes_total']/1e6:.1f} MB, {cm['n_planes']} planes, "
          f"worst in_mask {inm[0]:.4f} (shift control median {sh[len(sh)//2]:.4f})")
    return {"dir": f"{REL}/{CELL_VOL}", "v": int(mp.stat().st_mtime), "kind": cm["kind"], "classes": cm["classes"],
            "files": files, "bytes_total": cm["bytes_total"],
            "checks": {"in_mask_min": inm[0], "in_mask_median": inm[len(inm) // 2],
                       "control_shift_median": sh[len(sh) // 2]}}


def browser():
    mp = SHARED / "sample_cells/browser_metadata.json"
    if not mp.exists():
        print("  no section browser (browser_metadata.json absent)")
        return None
    m = json.loads(mp.read_text())
    vol = [s for s in m["samples"] if s["mode"] == "volume"]
    if len(vol) != 1 or vol[0]["id"] != m["volume_sample"]:
        raise SystemExit("exactly one sample must be the registered volume; "
                         f"got {[s['id'] for s in vol]}")


    keep = set(SHOWN) | {"S22-27909-P1", "S22-27909-P2"} if SHOWN else None
    if keep:
        m["samples"] = [s for s in m["samples"] if s["id"] in keep]
        m["slides"] = {k: v for k, v in m["slides"].items() if v["sample"] in keep}
        m["n_samples"] = len(m["samples"])
        m["n_slides"] = len(m["slides"])

    parts = [x for x in m["samples"] if x["id"].startswith("S22-27909-P")]
    if parts:
        merged = {"id": "S22-27909", "mode": "volume",
                  "n_slides": sum(x["n_slides"] for x in parts),
                  "n_nerve": sum(x.get("n_nerve", 0) for x in parts),
                  "n_cells": sum(x.get("n_cells", 0) for x in parts),
                  "n_blocks_named": 0,
                  "blocks": [b for x in parts for b in x.get("blocks", [])]}
        m["samples"] = [x for x in m["samples"] if not x["id"].startswith("S22-27909-P")]
        m["samples"].append(merged)
        m["n_samples"] = len(m["samples"])
        for v in m["slides"].values():
            if str(v.get("sample", "")).startswith("S22-27909-P"):
                v["sample"] = "S22-27909"


    listed = {s["id"] for s in m["samples"]}
    for sid in list(REGISTERED) + [SAMPLE]:
        if sid in listed or (SHOWN and sid not in SHOWN):
            continue
        n = 0
        vm = OUT.parent.parent / sid / "raw" / "volume_8um" / "metadata.json"
        for cand in (vm, OUT.parent.parent / sid / "raw" / "volume_8um_x1" / "metadata.json"):
            if cand.exists():
                n = len(json.loads(cand.read_text()).get("encodings", {}).get("G_withdrawn", {}).get("planes", []))
                break
        m["samples"].append({"id": sid, "mode": "volume", "n_slides": n, "n_nerve": 0, "n_cells": 0,
                             "n_blocks_named": 0, "blocks": []})
        listed.add(sid)
    m["n_samples"] = len(m["samples"])
    for s in m["samples"]:
        if s["id"] in REGISTERED and s["id"] != SAMPLE:
            s["mode"] = "volume"
            s["page"] = REGISTERED[s["id"]]
        elif s["id"] == SAMPLE:
            s["mode"] = "volume"
            s.pop("page", None)


    m["volume_sample"] = SAMPLE


    missing = [s for s in m["slides"] if not (SHARED / "sample_cells"
                                              / m["slides"][s]["cells"]).exists()]
    if missing:
        raise SystemExit(f"cell binaries missing for {len(missing)} sections: "
                         f"{missing[:5]}")


    for s, v in m["slides"].items():
        for lv in (v.get("tiles") or {}).get("levels", []):
            if not lv.get("dir"):
                raise SystemExit(f"{s}: a tile level has no directory name; "
                                 f"the page builds its URLs from it")
    n_tiles = sum(1 for s, v in m["slides"].items() if v.get("tiles"))
    n_sec = sum(1 for s, v in m["slides"].items() if v["sample"] != m["volume_sample"])
    print(f"  section browser: {m['n_samples']} samples, {m['n_slides']} sections, "
          f"{n_tiles} with a tile pyramid ({n_sec} outside the volume), "
          f"{m['cells_bytes_total']/1e6:.0f} MB of cell binaries")
    return m


def sections():
    mp = SHARED / "sample_cells/sections_metadata.json"
    if not mp.exists():
        print("  no section index (sections_metadata.json absent)")
        return None
    m = json.loads(mp.read_text())


    for sid, sec in m["sections"].items():
        for k, L in sec["layers"].items():
            groups = ([c["levels"] for c in L["channels"]] if k == "codex"
                      else ([L["levels"]] if L.get("levels") else []))
            for lv in [x for g in groups for x in g]:
                if not lv.get("dir"):
                    raise SystemExit(f"{sid}/{k}: a tile level has no directory name")
    n_mod = collections.Counter(k for s in m["sections"].values() for k in s["layers"])
    print(f"  section index: {m['n_sections']} sections, "
          f"{m['n_multi_modal']} carrying more than one modality; "
          + ", ".join(f"{k} {v}" for k, v in sorted(n_mod.items())))
    return m


def main():
    global OUT, HTML, RES, CELL_VOL, SAMPLE, TILES, OUTPUTS, REL
    import re as _re
    ap = argparse.ArgumentParser(); ap.add_argument("--out", default="")
    ap.add_argument("--outputs-root", default="", dest="outputs_root",
                    help="directory the volume / tiles / cell-vol names are resolved under; "
                         "default reconstruction (samples: <sample>/raw)")
    ap.add_argument("--volume", default="", help="8 um/px volume dir; default volume_8um")
    ap.add_argument("--volume4", default="", help="4 um/px volume dir; '-' to omit")
    ap.add_argument("--cell-vol", default="", dest="cell_vol",
                    help="volume carrying the cell layer; '-' for a sample without one")
    ap.add_argument("--tiles", default="",
                    help="tile pyramid directory under outputs/; default volume_tiles")
    ap.add_argument("--sample", default="", help="sample name shown in the title")
    ap.add_argument("--page", action="store_true",
                    help="rebuild the framework page samples/viewer/index.html from the viewer sources. "
                         "Without it a publish writes ONLY this sample's data (sample_data.js, "
                         "objects_payload.js, solid_assets/); the page on disk is left untouched.")
    ap.add_argument('--framework-out', help='review a rebuilt framework at another HTML path in the same samples/viewer directory')
    a = ap.parse_args()
    if a.outputs_root:
        OUTPUTS = Path(a.outputs_root)
    if a.out:
        OUT = Path(a.out); HTML = OUT / HTML.name
    if a.sample:
        SAMPLE = a.sample
    import os as _os


    FRAME = OUT.parent.parent / "viewer" if OUT.parent.parent.name == "samples" else None
    REL = _os.path.relpath(OUTPUTS, FRAME if FRAME else OUT)
    if FRAME:
        FRAME.mkdir(parents=True, exist_ok=True)
        have = sorted(d.name for d in OUT.parent.parent.iterdir()
                      if (d / "viewer" / "sample_data.js").exists() or d.name == SAMPLE)
        for k in list(REGISTERED):
            if k in have:
                REGISTERED[k] = f"index.html?sample={k}"
            else:
                REGISTERED.pop(k)


        for k in have:
            REGISTERED.setdefault(k, f"index.html?sample={k}")
    if a.volume or a.volume4:
        r = []
        r.append(("8", Path(a.volume) if a.volume else OUTPUTS / "volume_8um"))
        if a.volume4 != "-":
            r.append(("4", Path(a.volume4) if a.volume4 else OUTPUTS / "volume_4um"))
        RES = r
    if a.cell_vol:
        CELL_VOL = "" if a.cell_vol == "-" else a.cell_vol
    if a.sample:
        SAMPLE = a.sample
    if a.tiles:
        TILES = a.tiles
    OUT.mkdir(parents=True, exist_ok=True)
    res, meta0 = {}, None
    for key, d in RES:
        mp = d / "metadata.json"
        if not mp.exists():
            print(f"  skipping {key} um/px: {mp} not present")
            continue
        m = json.loads(mp.read_text())
        meta0 = meta0 or m
        enc = m["encodings"]


        files = {b: [p["webp_q90_lossy"] for p in enc[b]["planes"]]
                 for b in m["bases"]}
        want_px = (m["canvas_px"]["width"], m["canvas_px"]["height"])
        if not on_canvas(f"{d.name} grey planes",
                         (d / f for fs in files.values() for f in fs), want_px):
            raise SystemExit(f"{d.name}: its grey planes are not on the canvas its own "
                             f"metadata declares; nothing can be published on it")


        he_rgb, he_bytes = None, None
        hp = d / "he_rgb_metadata.json"
        if hp.exists():
            hm = json.loads(hp.read_text())
            he_rgb = {b: {} for b in m["bases"]}
            for q in hm["planes"]:
                he_rgb[q["basis"]][str(q["index"])] = q["file"]
            he_bytes = hm["bytes_per_basis"]
            if not on_canvas(f"{d.name} H&E colour planes",
                             (d / f for fs in he_rgb.values() for f in fs.values()), want_px):
                he_rgb, he_bytes = None, None


        swap, swap_bytes = None, None
        sd = d.parent / f"{d.name}_heswap"
        smp = sd / "metadata.json"
        if smp.exists():
            sm = json.loads(smp.read_text())
            if (sm["canvas_px"] != m["canvas_px"]):
                raise SystemExit(
                    f"{sd.name} canvas {sm['canvas_px']} != {d.name} {m['canvas_px']}; "
                    f"the page draws both arms on one quad, so they must share a canvas")
            senc = sm["encodings"]


            base_idx = {q["section_id"]: q["index"] for q in m["encodings"][
                next(iter(m["bases"]))]["planes"]}
            gone = sorted({q["section_id"] for b in sm["bases"] for q in senc[b]["planes"]}
                          - set(base_idx))
            if gone:
                raise SystemExit(f"{sd.name} carries sections {gone}, which {d.name} "
                                 f"does not have; the two arms must share their planes")
            s_files = {}
            for b in sm["bases"]:
                lst = [f"../{d.name}/{f}" for f in files[b]]
                for q in senc[b]["planes"]:
                    lst[base_idx[q["section_id"]]] = q["webp_q90_lossy"]
                s_files[b] = lst
            carried = {b: {q["section_id"] for q in senc[b]["planes"]} for b in sm["bases"]}
            base_sid = {v: k for k, v in base_idx.items()}
            s_rgb = None
            shp = sd / "he_rgb_metadata.json"
            if shp.exists():
                shm = json.loads(shp.read_text())
                s_rgb = {b: {} for b in sm["bases"]}
                for q in shm["planes"]:
                    s_rgb[q["basis"]][str(q["index"])] = q["file"]
                if not on_canvas(f"{sd.name} H&E colour planes",
                                 (sd / f for fs in s_rgb.values() for f in fs.values()), want_px):
                    s_rgb = None


            if he_rgb:
                s_rgb = s_rgb or {b: {} for b in sm["bases"]}
                for b in sm["bases"]:
                    for i, f in he_rgb.get(b, {}).items():
                        sid_i = base_sid.get(int(i))
                        if sid_i is not None and sid_i not in carried[b]:
                            s_rgb[b].setdefault(i, f"../{d.name}/{f}")


            PUBLISHED_EDGE_RUNS = ["pair_fits"]
            base_runs = m.get("edge_runs") or PUBLISHED_EDGE_RUNS
            same_geom = (sm.get("edge_runs") is not None
                         and sm.get("edge_runs") == base_runs)
            if on_canvas(f"{sd.name} grey planes",
                         (sd / f for fs in s_files.values() for f in fs), want_px):
                swap = {"dir": f"{REL}/{sd.name}",
                        "v": _stamp(sd / "metadata.json", sd / "he_rgb_metadata.json"),
                        "files": s_files,
                        "he_rgb": s_rgb,
                        "same_geometry": bool(same_geom),
                        "edge_runs": sm.get("edge_runs")}
                swap_bytes = {b: senc[b]["webp_q90_LOSSY"]["bytes"] for b in sm["bases"]}
            else:
                print(f"  WARNING: the {sd.name} arm is NOT published this build", flush=True)
        res[key] = {
            "dir": f"{REL}/{d.name}",
            "v": _stamp(d / "metadata.json", d / "he_rgb_metadata.json"),
            "canvas_px": m["canvas_px"], "canvas_mm": m["canvas_mm"],
            "in_plane_um_per_px": m["in_plane_um_per_px"],
            "encoding": "WebP q90, a LOSSY compression",
            "files": files,
            "he_rgb": he_rgb,
            "he_rgb_bytes_per_basis": he_bytes,
            "swap": swap,
            "he_swap_bytes_per_basis": swap_bytes,
            "bytes_per_basis": {b: enc[b]["webp_q90_LOSSY"]["bytes"] for b in m["bases"]},
        }
        print(f"  {key} um/px: {m['canvas_px']['width']}x{m['canvas_px']['height']} px, "
              + ", ".join(f"{b} {enc[b]['webp_q90_LOSSY']['bytes']/1e6:.1f} MB"
                          for b in m["bases"]))
    if not res:
        raise SystemExit("no volume metadata found; build the volumes first")
    default = DEFAULT_RES if DEFAULT_RES in res else sorted(res)[-1]

    enc_line = ENC_LINE
    if any(v.get("he_rgb") for v in res.values()):
        enc_line += (" The H&E native-colour planes use the same q90 encoding; their "
                     "alpha was checked pixel by pixel against the grey planes' and is "
                     "identical on 50 of 50, so the two sets cover exactly the same "
                     "pixels. RGB is NOT rescaled per section, so colour differences "
                     "between H&E sections are staining differences.")
    bases = list(meta0["bases"])
    m = {"planes": [{"index": p["index"], "section_id": p["section_id"],
                     "z_um": p["z_um"], "modality": p["modality"]}
                    for p in meta0["encodings"][bases[0]]["planes"]],
         "canvas_px": meta0["canvas_px"], "canvas_mm": meta0["canvas_mm"],
         "in_plane_um_per_px": meta0["in_plane_um_per_px"],
         "true_aspect_ratio_xz": meta0["true_aspect_ratio_xz"],
         "gaps_ge_25um": meta0["gaps_ge_25um"], "anchor": meta0["anchor"],
         "encoding": "WebP q90, a LOSSY compression",
         "encoding_line": enc_line, "grey_line": GREY_LINE,
         "tissue_frac": tissue_frac(RES[0][1], meta0),
         "bases": {b: {"label": meta0["bases"][b]["label"], "short": SHORT[b]}
                   for b in bases}}


    swapmeta = OUTPUTS / CELL_VOL / "he_swap_metadata.json"
    prep = ROOT / "registration/runs/pair_fits_swap/prepare.json"
    if any(v.get("swap") for v in res.values()) and (swapmeta.exists() or prep.exists()):
        if swapmeta.exists():
            sm = json.loads(swapmeta.read_text())
            pj = {"swapped_sections": [q["section_id"] for q in sm["sections"]],
                  "cross_modal_edges_before": None, "cross_modal_edges_after": None,
                  "n_edges_rerun": None}
            if prep.exists():
                pv = json.loads(prep.read_text())
                pj.update({k: pv.get(k) for k in ("cross_modal_edges_before",
                                                  "cross_modal_edges_after", "n_edges_rerun")})
        else:
            pj = json.loads(prep.read_text())
        by_sid = {p["section_id"]: p["index"] for p in m["planes"]}
        missing = [s for s in pj["swapped_sections"] if s not in by_sid]
        if missing:
            raise SystemExit(f"swapped sections absent from the volume: {missing}")


        _mods = {}
        if swapmeta.exists():
            for q in sm["sections"]:
                if q["section_id"] in by_sid:
                    _mods[str(by_sid[q["section_id"]])] = q.get("modality", "he")
        m["he_swap"] = {
            "indices": sorted(by_sid[s] for s in pj["swapped_sections"]),
            "modalities": _mods,
            "n": len(pj["swapped_sections"]),
            "note": "these sections enter the second arm as their post-CODEX H&E, so "
                    "their registration edges are solved against H&E neighbours",
            "cross_modal_edges": [pj["cross_modal_edges_before"],
                                  pj["cross_modal_edges_after"]],
            "n_edges_resolved": pj["n_edges_rerun"],
        }


    tname = TILES if TILES else "volume_tiles"
    tiles_dir = OUTPUTS / tname
    tiles_swap = OUTPUTS / "volume_tiles_heswap"
    source = {"mode": "fetch", "res": res, "default_res": default,
              "tiles": ({"dir": f"{REL}/{tname}", "v": _tiles_stamp(tiles_dir)} if tiles_dir.is_dir() else None),
              "tiles_swap": ({"dir": f"{REL}/volume_tiles_heswap", "v": _tiles_stamp(tiles_swap)}
                             if tiles_swap.is_dir() else None),
              "cells": cell_layer(), "cell_points": cell_points(),
              "nerve_regions": nerve_regions(),
              "tls_regions": tls_regions(),
              "duct_regions": duct_regions(),
              "gland_regions": gland_regions(),
              "nerve_lines": nerve_lines(),
              "tumor_regions": tumor_regions(),
              "cells_denoised3d": cells_denoised3d(),
              "xenium_gt": xenium_gt()}


    if CELL_VOL and (OUTPUTS / CELL_VOL / "metadata.json").exists():
        want_px = _canvas_px(OUTPUTS / CELL_VOL)
        for k in ("nerve_regions", "tls_regions", "duct_regions", "nerve_lines",
                  "tumor_regions", "cells_denoised3d"):
            if source.get(k) and not on_canvas(
                    k, layer_rasters(OUTPUTS / CELL_VOL, source[k]), want_px):
                source[k] = None
    brows = browser()


    for sample in (brows or {}).get("samples", []):
        if sample["id"] == SAMPLE:
            sample["n_slides"] = len(m["planes"])
    secs = sections()


    if FRAME:


        link = ('<button id="resetBtn" title="forget everything this browser stored for the page '
                '(saved switches, colours, cached section images) and reload it clean">reset &amp; reload</button>')
    else:
        _pair = {"HT891Z1": "HT913Z1", "HT913Z1": "HT891Z1",
                 "S22-27909": "HT891Z1", "S16-38794": "S22-27909"}
        _nb = _pair.get(SAMPLE, "HT891Z1")
        other = (REGISTERED.get(_nb, "#"), _nb)
        link = ('<a id="xsample" href="' + other[0] + '" title="the other registered '
                'sample">&#8594; ' + other[1] + '</a>')
    body_html = VA.body(res_control=True).replace(
        '<span id="progress"></span>', '<span id="progress"></span>\n  ' + link, 1)
    body_html = body_html.replace(
        '<button class="railbtn" id="fsBtn"',
        '<button class="railbtn" data-g="solid" aria-pressed="false"\n'
        '      title="Solid 3D bodies" aria-label="solid three-dimensional bodies">&#10697;</button>\n'
        '    <button class="railbtn" id="fsBtn"', 1)


    body_html = body_html.replace(
        '<div class="grp" data-g="display">',
        '<div class="grp" data-g="solid"><span class="hd">Solid 3D bodies</span>'
        + VS.PANEL + '</div>\n    <div class="grp" data-g="display">', 1)

    data_js = (f"window.__SAMPLE_ID__ = {json.dumps(SAMPLE)};\n"
               f"window.__BUILD_STAMP__ = {json.dumps(__import__('datetime').datetime.now(__import__('zoneinfo').ZoneInfo('America/Chicago')).strftime('%m-%d %H:%M') + ' CT')};\n"
               f"window.__SOURCE__ = {json.dumps(source, separators=(',', ':'))};\n"
               f"window.__META__ = {json.dumps(m, separators=(',', ':'))};\n"
               f"window.__BROWSER__ = {json.dumps(brows, separators=(',', ':'))};\n"
               f"window.__SECTIONS__ = {json.dumps(secs, separators=(',', ':'))};\n"
               "window.__SOLID_STAMP__ = 0;\n")
    static_js = (f"window.__VS_QUAD__ = {json.dumps(VA.VS_QUAD)};\n"
                 f"window.__FS_QUAD__ = {json.dumps(VA.FS_QUAD)};\n"
                 f"window.__VS_RECUT__ = {json.dumps(VA.VS_RECUT)};\n"
                 f"window.__VS_FULL__ = {json.dumps(VA.VS_FULL)};\n"
                 f"window.__FS_RESOLVE__ = {json.dumps(VA.FS_RESOLVE)};")
    if FRAME:
        from build_viewer_framework import render_framework
        data_dirs = {SAMPLE: OUT.name} if a.framework_out and OUT.name != 'viewer' else None
        html = render_framework(have, data_dirs)
        HTML = FRAME / "index.html"
        if a.framework_out:
            target = Path(a.framework_out)
            if target.parent.resolve() != FRAME.resolve() or target.suffix != '.html':
                raise SystemExit('--framework-out must be an HTML file in the framework directory')
            HTML = target


        PAGE_WRITE = bool(a.page) or not HTML.exists()
        if not PAGE_WRITE:
            _m = _re.search(r"window\.__SAMPLES__ = (\[[^\]]*\]);", HTML.read_text(encoding="utf-8"))
            _reg = json.loads(_m.group(1)) if _m else []
            if SAMPLE not in _reg:
                print(f"  {SAMPLE} is not in the page's sample switch {_reg}: the page is rebuilt to add it")
                PAGE_WRITE = True
        if PAGE_WRITE:
            HTML.write_text(html, encoding="utf-8")
        else:
            print("  framework page left untouched (data-only publish; use --page / the page stage to rebuild it)")
        (OUT / "sample_data.js").write_text(data_js, encoding="utf-8")
        for stale in ("index.html",):
            (OUT / stale).unlink(missing_ok=True)


        per_sample = sorted(str(q) for q in FRAME.parent.glob("*/viewer/index.html"))
        if per_sample:
            raise SystemExit(f"a sample directory carries its own page: {per_sample}")
        for k in have:
            if not (FRAME.parent / k / "viewer" / "sample_data.js").exists() and k != SAMPLE:
                raise SystemExit(f"{k} is in the sample switch but has no sample_data.js")
    else:
        html = f"""<!doctype html>
<html lang="en" translate="no" class="notranslate"><head><meta charset="utf-8">
<meta name="google" content="notranslate">
<meta name="viewport" content="width=device-width,initial-scale=1">
<style>body,body *{{-webkit-user-select:none;user-select:none}}</style>
<script>window.__BUILD_STAMP__ = "{__import__('datetime').datetime.now(__import__('zoneinfo').ZoneInfo('America/Chicago')).strftime('%m-%d %H:%M')} CT";</script>
<link rel="icon" href="data:,">
<title>{SAMPLE} serial-section stack (served)</title>
<style>{VA.CSS}{VS.CSS}
#xsample{{margin-left:10px;color:var(--accent);text-decoration:none;font-size:12px}}
#xsample:hover{{text-decoration:underline}}</style></head>
<body>
{body_html}
<script>{data_js}
{static_js}</script>
<script>{VA.JS}</script>
<script>{VS.JS}</script>
</body></html>
"""
        HTML.write_text(html, encoding="utf-8")


    sd = VS.payload(ROOT, SAMPLE, assets=OUT / "solid_assets",
                    url_prefix=(f"../{SAMPLE}/{OUT.name if a.framework_out else 'viewer'}/" if FRAME else ""))
    if sd and FRAME:


        fz = OUT.parent / "configs" / "tls3d_frozen.tsv"
        if fz.exists():
            want = {l.split("\t")[0] for l in fz.read_text().splitlines()[1:] if l.strip()}
            have = {m["name"].replace("tls_", "", 1) for m in sd["meshes"] if m["name"].startswith("tls_")}


            if have - want:
                raise SystemExit(f"3-D TLS bodies {sorted(have - want)} are not in the frozen list {sorted(want)}; not published")
            if want - have:
                print(f"  frozen 3-D TLS ids without a body this build (retired): {sorted(want - have)}", flush=True)
    if sd:
        pj = OUT / "objects_payload.js"
        pj.write_text("window.__SOLID__ = "
                      + json.dumps(sd, separators=(",", ":")) + ";\n", encoding="utf-8")


        stamp = f"window.__SOLID_STAMP__ = {int(pj.stat().st_mtime)}"
        if FRAME:
            dj = OUT / "sample_data.js"
            dj.write_text(dj.read_text().replace("window.__SOLID_STAMP__ = 0", stamp), encoding="utf-8")
        else:
            html = html.replace("window.__SOLID_STAMP__ = 0", stamp)
            HTML.write_text(html, encoding="utf-8")
        print(f"  solid bodies: {len(sd['meshes'])} meshes, {sd['n_faces_total']:,} "
              f"faces -> {pj.name} ({pj.stat().st_size/1e6:.1f} MB, fetched on first open)")


    import shutil as _sh, subprocess as _sp
    _node = _sh.which("node") or str(Path.home() / "node/bin/node")
    if FRAME and not PAGE_WRITE:
        pass
    elif Path(_node).exists():
        _chk = ("const fs=require('fs');const h=fs.readFileSync(process.argv[1],'utf8');"
                "const m=h.match(/<script[^>]*>([\\s\\S]*?)<\\/script>/g)||[];let bad=0;"
                "for(const b of m){if(/^<script[^>]*src=/.test(b))continue;"
                "const c=b.replace(/^<script[^>]*>/,'').replace(/<\\/script>$/,'');"
                "try{new Function(c);}catch(e){bad++;console.log('SYNTAX '+e.message);}}"
                "process.exit(bad?1:0);")
        _r = _sp.run([_node, "-e", _chk, str(HTML)], capture_output=True, text=True)
        if _r.returncode != 0:
            HTML.unlink()
            raise SystemExit(f"REFUSED: the page has a script syntax error: {_r.stdout.strip()[:300]}")
        print("  page scripts parse (node)")
    else:
        print("  WARNING: node not found, page scripts not syntax-checked")
    print(f"wrote {HTML}  ({HTML.stat().st_size/1e3:.1f} kB, images fetched not embedded)")
    (OUT / "build_info.json").write_text(json.dumps(
        {"html": str(HTML), "html_bytes": HTML.stat().st_size,
         "sample_data": (str(OUT / "sample_data.js") if FRAME else None),
         "document_root": str(OUTPUTS),
         "url_path": (f"/samples/viewer/index.html?sample={SAMPLE}" if FRAME
                      else "/viewer_server/index.html"),
         "resolutions": {k: {"dir": v["dir"], "canvas_px": v["canvas_px"],
                             "bytes_per_basis": v["bytes_per_basis"],
                             "he_rgb_bytes_per_basis": v["he_rgb_bytes_per_basis"],
                             "he_rgb_planes_per_basis":
                                 (len(next(iter(v["he_rgb"].values())))
                                  if v.get("he_rgb") else 0),
                             "he_swap_bytes_per_basis": v["he_swap_bytes_per_basis"],
                             "he_swap_planes_per_basis":
                                 (len(next(iter(v["swap"]["files"].values())))
                                  if v.get("swap") else 0)}
                         for k, v in res.items()},
         "default_res_um": default,
         "cell_prediction": ({
             "textures": sum(len(v) for b in source["cells"]["avail"].values()
                             for v in b.values()),
             "classes": [c["name"] for c in source["cells"]["classes"]],
             "default_on": source["cells"]["default_on"],
             "planes_with": len(source["cells"]["with"]),
             "planes_by_pointcloud_fit": source["cells"]["n_xenium"],
             "checks": source["cells"]["checks"]}
             if source.get("cells") else None),
         "section_browser": ({
             "samples": brows["n_samples"], "sections": brows["n_slides"],
             "volume_sample": brows["volume_sample"],
             "sections_with_tiles": sum(1 for v in brows["slides"].values()
                                        if v.get("tiles")),
             "cells_bytes_total": brows["cells_bytes_total"],
             "multi_block_samples": [s["id"] for s in brows["samples"]
                                     if len(s["blocks"]) > 1]}
             if brows else None),
         "shared_with_self_contained": "scripts/viewer_assets.py",
         },
        indent=1))


if __name__ == "__main__":
    main()
