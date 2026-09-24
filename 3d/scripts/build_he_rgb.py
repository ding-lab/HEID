#!/usr/bin/env python

from __future__ import annotations
import os

import argparse
import csv
import importlib.util as _iu
import json
import time
from pathlib import Path

import cv2
import numpy as np
import tifffile
from PIL import Image
Image.MAX_IMAGE_PIXELS = None

ROOT = Path(os.environ.get("THREED_ROOT", os.path.join(os.environ.get("PROJECTS_ROOT", "/data/heid"), "3d")))
RUNS = ROOT / "registration/runs"


SEC = RUNS / "prep_sections/sections"
MANIFEST = RUNS / "prep_sections/manifest.tsv"
EDGE_RUNS = ("pair_fits",)

SRC_MPP, WG_MPP = 1.0, 2.0
RGB_MPP = 4.0
BASES = {"G_withdrawn": "R", "G_not_withdrawn": "M"}


def _load(name, path):
    s = _iu.spec_from_file_location(name, path)
    m = _iu.module_from_spec(s)
    s.loader.exec_module(m)
    return m


CHAIN = _load("placement_chain", Path(__file__).resolve().parent / "placement_chain.py")


def plane_size(sid, out_mpp):
    with Image.open(SEC / sid / f"mask_mpp{SRC_MPP:g}.png") as im:
        w, h = im.size
    f = SRC_MPP / out_mpp
    return max(1, int(round(w * f))), max(1, int(round(h * f)))


def mask_at(sid, out_mpp):
    m = cv2.imread(str(SEC / sid / f"mask_mpp{SRC_MPP:g}.png"), cv2.IMREAD_GRAYSCALE) > 127
    w, h = plane_size(sid, out_mpp)
    return cv2.resize(m.astype(np.uint8), (w, h), interpolation=cv2.INTER_AREA) > 0


def rgb_at(sid, out_mpp):
    a = tifffile.imread(str(SEC / sid / f"rgb_mpp{RGB_MPP:g}.tif"))
    if a.ndim == 2:
        a = np.dstack([a] * 3)
    a = a[..., :3]
    w, h = plane_size(sid, out_mpp)
    return cv2.resize(a, (w, h), interpolation=cv2.INTER_AREA)


def recorded_rasterization(meta, out_mpp):
    record = meta.get("rasterization")
    if record is None:
        return None
    origin = np.asarray(record["warp_origin_um"], dtype=float)
    width, height = (int(record["warp_canvas_px"][k]) for k in ("width", "height"))
    crop = record["crop_xyxy_px"]
    if len(crop) != 4 or any(int(v) != v for v in crop):
        raise ValueError("Non-integer grey rasterization crop")
    c0, r0, c1, r1 = map(int, crop)
    if origin.shape != (2,) or not np.isfinite(origin).all() or not (0 <= c0 < c1 <= width and 0 <= r0 < r1 <= height):
        raise ValueError("Invalid grey rasterization bounds/origin")
    if (c1-c0, r1-r0) != (meta["canvas_px"]["width"], meta["canvas_px"]["height"]):
        raise ValueError("Grey rasterization crop differs from its canvas size")
    if not np.allclose(origin + np.array([c0, r0]) * out_mpp, meta["canvas_origin_um"], atol=1e-7, rtol=0):
        raise ValueError("Grey rasterization crop differs from its canvas origin")
    return origin, width, height, c0, r0, c1, r1


def main():
    global SEC, MANIFEST, EDGE_RUNS
    ap = argparse.ArgumentParser()
    ap.add_argument("--mpp", type=float, default=8.0)
    ap.add_argument("--quality", type=int, default=90)
    ap.add_argument("--method", type=int, default=4, help="WebP effort; 4 encodes 6x faster than 6 at the same size")
    ap.add_argument("--plane-index", type=int, default=-1, dest="plane_index", help="one section (z order) per job: write its planes + a sidecar json")
    ap.add_argument("--finalize", action="store_true", help="collect the sidecars into he_rgb_metadata.json")
    ap.add_argument("--prep-root", default=str(SEC))
    ap.add_argument("--manifest", default=str(MANIFEST),
                    help="which sections count as H&E is read from here, so the arm "
                         "where 16 CODEX sections became H&E gets 41 colour planes")
    ap.add_argument("--edge-runs", default=",".join(EDGE_RUNS),
                    help="a re-solved run must be listed BEFORE the run it supersedes")
    ap.add_argument("--anchor", default="HT891Z1-U66",
                    help="section the chain is anchored to; must match the grey build")
    ap.add_argument("--suffix", default="",
                    help="appended to the volume directory name, e.g. _heswap")
    ap.add_argument("--out", default="", help="volume dir to write into; default reconstruction/volume_<mpp>um<suffix>")
    a = ap.parse_args()
    out_mpp = float(a.mpp)
    SEC = Path(a.prep_root)
    MANIFEST = Path(a.manifest)
    EDGE_RUNS = tuple(s.strip() for s in a.edge_runs.split(",") if s.strip())
    out = Path(a.out) if a.out else ROOT / f"reconstruction/volume_{out_mpp:g}um{a.suffix}"
    print(f"prep={SEC}\nmanifest={MANIFEST}\nedges={EDGE_RUNS}\nout={out}", flush=True)
    meta_p = out / "metadata.json"
    if not meta_p.exists():
        raise SystemExit(f"{meta_p} does not exist; build the grey volume first")
    gmeta = json.loads(meta_p.read_text())
    t0 = time.time()

    man = {r["section_id"]: r for r in
           csv.DictReader(open(MANIFEST), delimiter="\t")}
    sids = sorted(man, key=lambda s: float(man[s]["z_position_um"]))
    z = np.array([float(man[s]["z_position_um"]) for s in sids])
    he = [i for i, s in enumerate(sids) if man[s]["modality"] == "he"]
    print(f"{len(sids)} sections, {len(he)} of them H&E, target {out_mpp:g} um/px")

    edges = CHAIN.harvest(EDGE_RUNS)
    tree = CHAIN.prereg_tree(edges, sids)
    chains = {k: CHAIN.chain_on(tree, edges, b, a.anchor) for k, b in BASES.items()}


    sizes = {s: plane_size(s, out_mpp) for s in sids}
    pts = []
    for ch in chains.values():
        for s in sids:
            w, h = sizes[s]
            c = np.array([[0, 0], [w, 0], [w, h], [0, h]], float) * (out_mpp / WG_MPP)
            pts.append((c @ ch[s][:2, :2].T + ch[s][:, 2]) * WG_MPP)
    pts = np.vstack(pts)
    o = pts.min(0) - 40.0
    W = int(np.ceil((pts[:, 0].max() + 40.0 - o[0]) / out_mpp))
    H = int(np.ceil((pts[:, 1].max() + 40.0 - o[1]) / out_mpp))

    def M_for(bk, s):
        M = np.zeros((2, 3))
        M[:2, :2] = chains[bk][s][:2, :2]
        M[:, 2] = (WG_MPP / out_mpp) * chains[bk][s][:, 2] - o / out_mpp
        return M

    want = (gmeta["canvas_px"]["width"], gmeta["canvas_px"]["height"])
    if not gmeta.get("canvas_origin_um"):

        anym = np.zeros((H, W), bool)
        for bk in chains:
            for s in sids:
                m = cv2.warpAffine(mask_at(s, out_mpp).astype(np.uint8) * 255, M_for(bk, s),
                                   (W, H), flags=cv2.INTER_NEAREST, borderValue=0)
                anym |= m > 0
        ys, xs = np.nonzero(anym)
        p = int(round(80.0 / out_mpp))
        r0, r1 = max(0, ys.min() - p), min(H, ys.max() + p + 1)
        c0, c1 = max(0, xs.min() - p), min(W, xs.max() + p + 1)
        CW, CH = c1 - c0, r1 - r0
    recorded = recorded_rasterization(gmeta, out_mpp)
    if recorded is not None:
        o, W, H, c0, r0, c1, r1 = recorded
        CW, CH = c1-c0, r1-r0
        print(f"exact grey rasterization: {W} x {H}, crop {c0},{r0}:{c1},{r1}", flush=True)
    elif gmeta.get("canvas_origin_um"):

        o = np.asarray(gmeta["canvas_origin_um"], float)
        c0, r0 = 0, 0
        CW, CH = want
        W, H = want
        r1, c1 = H, W
        print(f"canvas from the volume: {CW} x {CH} at origin {o[0]:.1f}, {o[1]:.1f} um", flush=True)
    print(f"canvas rebuilt {CW} x {CH} px; grey volume says {want[0]} x {want[1]}")
    if (CW, CH) != want:
        raise SystemExit("canvas mismatch: refusing to write planes on a different grid")

    planes, checks = [], []
    for bk in BASES:
        d = out / bk / "rgb"
        d.mkdir(parents=True, exist_ok=True)
        gplanes = {q["index"]: q for q in gmeta["encodings"][bk]["planes"]}
        for i in he:
            if a.plane_index >= 0 and i != a.plane_index:
                continue
            if a.finalize:
                sc = d / f"z{i:02d}_{sids[i].split('-')[1]}.json"
                if not sc.exists():
                    raise SystemExit(f"{sc} missing: the rgb array did not finish")
                j = json.loads(sc.read_text()); planes.append(j["plane"]); checks.append(j["check"]); continue
            s = sids[i]
            M = M_for(bk, s)
            rgb = cv2.warpAffine(rgb_at(s, out_mpp), M, (W, H),
                                 flags=cv2.INTER_LINEAR, borderValue=0)[r0:r1, c0:c1]
            al = cv2.warpAffine(mask_at(s, out_mpp).astype(np.uint8) * 255, M, (W, H),
                                flags=cv2.INTER_NEAREST, borderValue=0)[r0:r1, c0:c1]
            rgba = np.dstack([rgb * (al[..., None] > 0), al])
            fn = f"z{i:02d}_{s.split('-')[1]}.rgb.q90.webp"
            fp = d / fn
            Image.fromarray(rgba).save(fp, format="WEBP", quality=a.quality,
                                       method=a.method)

            gp = out / gplanes[i]["webp_lossless"]
            ga = np.array(Image.open(gp).convert("RGBA"))[..., 3]
            same = int((ga == al).sum())
            if recorded is not None and same != al.size:
                raise ValueError(f"{s} {bk}: recorded grey rasterization differs at {al.size-same} mask pixels")
            checks.append({"index": i, "basis": bk, "section_id": s,
                           "alpha_px_equal": same, "alpha_px_total": int(al.size),
                           "alpha_identical": bool(same == al.size)})
            planes.append({"index": i, "section_id": s, "basis": bk,
                           "z_um": float(z[i]),
                           "file": f"{bk}/rgb/{fn}",
                           "bytes": int(fp.stat().st_size),
                           "raw_bytes": int(rgb.size)})
            print(f"  {bk} {s} -> {fn}  {fp.stat().st_size/1e3:7.1f} kB   "
                  f"alpha identical to grey twin: {same == al.size}")
            if a.plane_index >= 0:
                (d / f"z{i:02d}_{s.split('-')[1]}.json").write_text(json.dumps({"plane": planes[-1], "check": checks[-1]}, indent=1))
    if a.plane_index >= 0:
        print(f"plane {a.plane_index} done  ({time.time()-t0:.0f}s)"); return

    if recorded is not None and any(not c["alpha_identical"] or c["alpha_px_equal"] != c["alpha_px_total"] for c in checks):
        raise ValueError("Recorded grey rasterization has a non-identical RGB mask sidecar")
    by_b = {b: [q for q in planes if q["basis"] == b] for b in BASES}
    meta = {
        "what": "H&E planes in native RGB, on the grey volume's canvas",
        "why": "the grey volume carries haematoxylin OD after CLAHE, not H&E colour",
        "n_planes_per_basis": {b: len(v) for b, v in by_b.items()},
        "modality": "he",
        "in_plane_um_per_px": out_mpp,
        "canvas_px": {"width": int(CW), "height": int(CH)},
        "source": f"prep_sections rgb_mpp{RGB_MPP:g}.tif (raw RGB, before "
                  f"deconvolution and CLAHE), area-downsampled to {out_mpp:g} um/px",
        "normalisation": "NONE. Unlike the grey channel, RGB is not rescaled per "
                         "section, so colour differences between H&E sections are "
                         "staining differences, not a display choice.",
        "encoding": f"WebP quality {a.quality}, method {a.method}. LOSSY in colour; "
                    f"the alpha channel is stored losslessly.",
        "alpha": "the same tissue mask as the grey volume, warped with the same "
                 "matrix and NEAREST, then verified pixel by pixel against the alpha "
                 "in the grey build's lossless WebP",
        "alpha_check": {"n_planes_checked": len(checks),
                        "n_identical": int(sum(c["alpha_identical"] for c in checks)),
                        "per_plane": checks},
        "bytes_per_basis": {b: int(sum(q["bytes"] for q in v)) for b, v in by_b.items()},
        "raw_bytes_per_basis": {b: int(sum(q["raw_bytes"] for q in v))
                                for b, v in by_b.items()},
        "planes": planes,
        "elapsed_s": round(time.time() - t0, 1),
        "read_only_upstream": True,
    }
    (out / "he_rgb_metadata.json").write_text(json.dumps(meta, indent=1))
    for b, v in by_b.items():
        print(f"{b}: {len(v)} planes, {sum(q['bytes'] for q in v)/1e6:.2f} MB encoded "
              f"from {sum(q['raw_bytes'] for q in v)/1e6:.1f} MB raw")
    nid = sum(c["alpha_identical"] for c in checks)
    print(f"alpha identical to the grey twin on {nid}/{len(checks)} planes")
    print(f"wrote {out/'he_rgb_metadata.json'}  ({meta['elapsed_s']} s)")


if __name__ == "__main__":
    main()
