#!/usr/bin/env python

from __future__ import annotations

import argparse
import os
import json
import re
import struct
from pathlib import Path

import numpy as np
from scipy.ndimage import (distance_transform_edt, gaussian_filter,
                           binary_closing, binary_opening, label)
from skimage.measure import marching_cubes

ROOT = Path(os.environ.get("THREED_ROOT", os.path.join(os.environ.get("PROJECTS_ROOT", "/data/heid"), "3d")))
VOL = ROOT / "reconstruction/volume_8um"
SAMPLE = "HT891Z1"
import os as _os
_OBJ3D = _os.environ.get("HTAN3D_OBJ3D", "")
OUT = (Path(_OBJ3D) / "S11_meshes") if _OBJ3D else ROOT / "objects_3d/outputs/S11_meshes"


_exf = Path(__import__('os').environ.get("HTAN3D_EXCLUDE",
            os.environ.get("THREED_ROOT", os.path.join(os.environ.get("PROJECTS_ROOT", "/data/heid"), "3d")) + "/front/S22-27909/configs/exclude_sections_3d.json"))
EXCLUDE_3D = set()
if _exf.exists():
    import json as _json
    EXCLUDE_3D = set(_json.loads(_exf.read_text())["sections"])

BASIS = "G_withdrawn"
UM_PX = 8.0
Z_STEP = 5.0
GAP_UM = 25.0


SIGMA_ECO_UM = 30.0
DENSITY_FRAC = 0.03
OPEN_R, CLOSE_R = 3, 2
MAX_HOLE_AREA_UM2 = 50000 * 10.0 * 10.0


def read_points(path: Path):
    b = path.read_bytes()
    magic, n, cw, ch, sub, ncls = struct.unpack_from("<8sIHHHH", b, 0)
    if magic != b"CELLPT01":
        raise ValueError(f"{path}: bad magic {magic!r}")
    o = 32
    xy = np.frombuffer(b, np.uint16, 2 * n, o).reshape(n, 2).astype(np.float32) / sub
    o += 4 * n
    o += n
    cls = np.frombuffer(b, np.uint8, n, o)
    return xy, cls, (cw, ch)


def disc(r: int) -> np.ndarray:
    y, x = np.ogrid[-r:r + 1, -r:r + 1]
    return x * x + y * y <= r * r


def class_mask(xy, cls, ix, shape, frac=None):
    return tumour_mask(xy, cls, ix, shape, frac=frac)


def tumour_mask(xy, cls, tumour_ix, shape, frac=None):
    h, w = shape
    sel = cls == tumour_ix
    grid = np.zeros((h, w), np.float32)
    if sel.sum() == 0:
        return grid.astype(bool)
    xs = np.clip(xy[sel, 0].astype(int), 0, w - 1)
    ys = np.clip(xy[sel, 1].astype(int), 0, h - 1)
    np.add.at(grid, (ys, xs), 1.0)
    d = gaussian_filter(grid, SIGMA_ECO_UM / UM_PX)
    if d.max() <= 0:
        return d.astype(bool)
    m = d > (DENSITY_FRAC if frac is None else frac) * d.max()
    m = binary_opening(m, disc(OPEN_R))
    m = binary_closing(m, disc(CLOSE_R))


    lab, n = label(~m)
    if n:
        border = set(lab[0].tolist()) | set(lab[-1].tolist()) \
               | set(lab[:, 0].tolist()) | set(lab[:, -1].tolist())
        sizes = np.bincount(lab.ravel())


        for i in range(1, n + 1):
            if i not in border:
                m[lab == i] = True
    return m


def sdf(mask: np.ndarray) -> np.ndarray:
    if not mask.any():
        return np.full(mask.shape, 1e4, np.float32)
    if mask.all():
        return np.full(mask.shape, -1e4, np.float32)
    out = distance_transform_edt(~mask) - distance_transform_edt(mask)
    return (out * UM_PX).astype(np.float32)


def stack_and_fit(layer_sdf: dict, z_list, shape, down: int):
    zs = sorted(layer_sdf)
    z0, z1 = zs[0], zs[-1]
    z_grid = np.arange(z0, z1 + 0.5 * Z_STEP, Z_STEP, dtype=np.float32)
    h, w = shape
    vol = np.empty((len(z_grid), h, w), np.float32)
    unsupported = np.zeros(len(z_grid), bool)
    arr = np.stack([layer_sdf[z] for z in zs], 0)
    zs_a = np.asarray(zs, np.float32)
    for k, z in enumerate(z_grid):
        j = int(np.searchsorted(zs_a, z, "right")) - 1
        j = max(0, min(j, len(zs_a) - 2)) if len(zs_a) > 1 else 0
        za, zb = zs_a[j], zs_a[min(j + 1, len(zs_a) - 1)]
        span = float(zb - za)
        t = 0.0 if span <= 0 else float((z - za) / span)
        t = min(1.0, max(0.0, t))
        vol[k] = (1.0 - t) * arr[j] + t * arr[min(j + 1, len(arr) - 1)]
        if span > GAP_UM and 0.0 < t < 1.0:
            unsupported[k] = True
    return vol, z_grid, unsupported


def mesh_from(vol, z_grid, unsupported, down: int, smooth_vox: float = 0.0):
    if smooth_vox > 0:
        vol = gaussian_filter(vol, smooth_vox)
    if not (vol.min() < 0 < vol.max()):
        return None


    vol = np.pad(vol, 1, mode="constant", constant_values=1e4)
    verts, faces, normals, _ = marching_cubes(vol, level=0.0)
    verts = verts - 1.0

    zi = verts[:, 0]
    kz = np.clip(np.rint(zi).astype(int), 0, len(unsupported) - 1)
    flag = unsupported[kz].astype(np.uint8)
    xyz = np.empty_like(verts)
    xyz[:, 0] = verts[:, 2] * UM_PX * down
    xyz[:, 1] = verts[:, 1] * UM_PX * down
    xyz[:, 2] = z_grid[0] + zi * Z_STEP


    n = np.empty_like(normals)
    n[:, 0], n[:, 1], n[:, 2] = -normals[:, 2], -normals[:, 1], -normals[:, 0]
    ln = np.linalg.norm(n, axis=1, keepdims=True)
    n = n / np.maximum(ln, 1e-6)


    f = faces.astype(np.uint32)
    return {"verts": xyz.astype(np.float32), "normals": n.astype(np.float32),
            "faces": f, "fitted": flag}


def write_mesh(name: str, m: dict, out: Path) -> dict:
    p = out / f"{name}.bin"
    with open(p, "wb") as f:
        f.write(struct.pack("<4sIII", b"MSH1", len(m["verts"]), len(m["faces"]), 0))
        f.write(m["verts"].tobytes())
        f.write(m["normals"].tobytes())
        f.write(m["fitted"].tobytes())


        pad = (-len(m["fitted"])) % 4
        if pad:
            f.write(b"\0" * pad)
        f.write(m["faces"].tobytes())
    return {"name": name, "file": p.name, "n_verts": int(len(m["verts"])),
            "n_faces": int(len(m["faces"])),
            "frac_fitted_across_gap": round(float(m["fitted"].mean()), 4),
            "bytes": p.stat().st_size}


def u_key(name: str):
    m = re.findall(r"[Uu](\d+)", name)
    return int(m[-1]) if m else None


def tls_core_masks(sample: str, shape_d, down: int):
    import csv
    import json
    meta = json.loads((VOL / "metadata.json").read_text())
    planes = [p for p in meta["encodings"][BASIS]["planes"]
              if p["section_id"] not in EXCLUDE_3D]
    pts_dir = VOL / BASIS / "cell_points"
    _tr = _os.environ.get("HTAN3D_TLSROOT", "")
    root = Path(_tr) if _tr else ROOT / "tls_define/outputs" / sample
    classes = json.loads((VOL / "cell_points_metadata.json").read_text())["classes"]
    cidx = {c: i for i, c in enumerate(classes)}
    cache = {}
    if root.is_dir():
        for d0 in sorted(root.iterdir()):
            f0 = d0 / f"{d0.name}_tls_cells.csv"
            if not d0.is_dir() or not f0.exists():
                continue
            with open(f0) as fh:
                seq = [cidx.get(r["cell_type"], 255) for r in csv.DictReader(fh)]
            cache[d0] = np.asarray(seq, dtype=np.uint8)
    h, w = shape_d
    out, skipped, renamed = {}, [], []
    globals()["_U_BY_DIR"] = {}
    for p in planes:
        cand = sorted(pts_dir.glob(f"z{p['index']:02d}_*.cells.bin"))
        if not cand:
            continue
        u = u_key(p["section_id"])
        xy, cls, _ = read_points(cand[0])
        best, best_ag = None, 0.0
        for d0, seq in cache.items():
            if len(seq) != len(cls):
                continue
            ag = float((seq == cls).mean())
            if ag > best_ag:
                best, best_ag = d0, ag
        if best is None or best_ag < 0.99:
            skipped.append((p["section_id"], round(best_ag, 4), int(len(xy))))
            continue
        d = best
        globals()["_U_BY_DIR"][d.name] = u
        if u_key(d.name) != u_key(p["section_id"]):
            renamed.append({"plane": p["section_id"], "tls_dir": d.name,
                            "class_agreement": round(best_ag, 4)})
        with open(d / f"{d.name}_tls_cells.csv") as fh:
            rows = list(csv.DictReader(fh))
        per = {}
        for i, r in enumerate(rows):
            if str(r.get("in_tls_core", "")).lower() not in ("true", "1"):
                continue
            tag = r.get("tls_local_halo150") or r.get("tls_region_halo150", "")
            if not tag or tag == "Outside":
                continue
            per.setdefault(tag, []).append(i)
        for tag, idx in per.items():
            g = np.zeros((h, w), bool)
            xs = np.clip((xy[idx, 0] / down).astype(int), 0, w - 1)
            ys = np.clip((xy[idx, 1] / down).astype(int), 0, h - 1)
            g[ys, xs] = True


            g = binary_closing(g, disc(max(2, int(round(30.0 / (UM_PX * down))))))
            g = binary_opening(g, disc(1))
            lab, n = label(~g)
            if n:
                border = set(lab[0].tolist()) | set(lab[-1].tolist()) \
                       | set(lab[:, 0].tolist()) | set(lab[:, -1].tolist())
                for i2 in range(1, n + 1):
                    if i2 not in border:
                        g[lab == i2] = True
            out.setdefault((u, tag), []).append((float(p["z_um"]), g))
    return out, skipped, renamed


def tls_objects(shape_d, down):
    import csv
    obj_tsv = (Path(_OBJ3D) / "S9_tls_objects/objects.tsv") if _OBJ3D else ROOT / ("objects_3d/outputs/S9_tls_objects/objects.tsv"
                      if SAMPLE == "HT891Z1" else
                      f"objects_3d/outputs/S9_tls_objects_{SAMPLE}/objects.tsv")
    if not obj_tsv.exists():
        return []
    masks, skipped, renamed = tls_core_masks(SAMPLE, shape_d, down)
    if skipped:
        print(f"  TLS: {len(skipped)} planes unmatched (no directory reproduces "
              f"their class sequence)")
    if renamed:
        print(f"  TLS: {len(renamed)} directories carry another section's data; "
              f"paired by content:")
        for r in renamed:
            print(f"    {r['plane']} <- {r['tls_dir']} (agreement {r['class_agreement']})")


    globals()["_TLS_SKIPPED"] = [{"plane": s, "best_class_agreement": a, "cells": b}
                                 for s, a, b in skipped]
    globals()["_TLS_RENAMED"] = renamed
    with open(obj_tsv) as f:
        objs = list(csv.DictReader(f, delimiter="\t"))
    out = []
    for r in objs:
        layers = {}
        for tok in (r.get("member_tls") or "").split(";"):
            tok = tok.strip()
            if ":" not in tok:
                continue
            slide, tag = tok.rsplit(":", 1)
            u_plane = globals().get("_U_BY_DIR", {}).get(slide, u_key(slide))
            for z, g in masks.get((u_plane, tag), []):
                layers[z] = sdf(g) if z not in layers else np.minimum(layers[z], sdf(g))
        if len(layers) < 2:
            continue
        vol, zg, uns = stack_and_fit(layers, sorted(layers), shape_d, down)
        m = mesh_from(vol, zg, uns, down, smooth_vox=1.0)
        if m is None:


            m = mesh_from(vol, zg, uns, down, smooth_vox=0.0)
        if m is None:
            print(f"  TLS {r['object_id']}: no closed surface even unsmoothed, not meshed")
            continue
        m["_id"] = r["object_id"]; m["_p"] = float(r["p_link"]); m["_tier"] = r.get("tier", "")
        m["_n"] = int(r["n_measured_sections"])
        m["_reglim"] = (r.get("registration_limited", "").lower() in ("true", "1", "yes"))
        out.append(m)
    return out


def main():
    global VOL, SAMPLE, OUT
    ap = argparse.ArgumentParser()
    ap.add_argument("--smooth", type=float, default=1.6)
    ap.add_argument("--volume", default="",
                    help="the volume to read cell points from; default HT891Z1's")
    ap.add_argument("--sample", default="HT891Z1",
                    help="which sample: decides the TLS core source and the out dir")
    ap.add_argument("--down", type=int, default=2,
                    help="in-plane downsample before the distance transform")
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    if a.volume:
        VOL = Path(a.volume)
    SAMPLE = a.sample
    if SAMPLE != "HT891Z1" and not _OBJ3D:
        OUT = OUT.parent / f"{OUT.name}_{SAMPLE}"
    OUT.mkdir(parents=True, exist_ok=True)
    down = a.down

    meta = json.loads((VOL / "metadata.json").read_text())
    cm = json.loads((VOL / "cell_points_metadata.json").read_text())
    classes = cm["classes"]
    planes = [p for p in meta["encodings"][BASIS]["planes"]
              if p["section_id"] not in EXCLUDE_3D]
    cw, ch = meta["canvas_px"]["width"], meta["canvas_px"]["height"]
    shape_d = (ch // down, cw // down)

    pts_dir = VOL / BASIS / "cell_points"

    report = {"basis": BASIS, "in_plane_um_per_px": UM_PX * down,
              "z_step_um": Z_STEP, "n_sections": len(planes),
              "gap_flag_um": GAP_UM, "meshes": [],
              "tumour_body": "OUTER SOLID: every interior hole filled per layer, so the body is closed. Internal cavities are not drawn here.",
              "surface_smoothing_voxels": None,
              "how_layers_are_joined":
                  "per-section signed distance fields, interpolated linearly along z "
                  "onto a 5 um grid, one isosurface at zero. The surface between two "
                  "measured sections is FITTED, not measured.",
              "claim": "MODEL PREDICTION on H&E (pan-cancer Cell head, single fold). "
                       "This cohort has no spatial ground truth; nothing here was scored."}


    CARVE = {}
    _carve = os.environ.get("HTAN3D_CARVE", "")
    if _carve and Path(_carve).exists():
        _npz = np.load(_carve)
        CARVE = {k: _npz[k] for k in _npz.files}
        print(f"  carving {len(CARVE)} sections' duct lumens out of the class bodies ({_carve})")
    WANT = ["Tumor", "NonMalignant_Parenchymal", "SMC", "Fibroblast", "NK_T",
            "Endothelial", "Myeloid_NOS", "B_cell", "Plasma", "Schwann"]
    for cname in WANT:
        if cname not in classes:
            continue
        ci = classes.index(cname)
        layers = {}
        for p2 in planes:
            cand = sorted(pts_dir.glob(f"z{p2['index']:02d}_*.cells.bin"))
            if not cand:
                continue
            xy, cls, _ = read_points(cand[0])
            m2 = class_mask(xy, cls, ci, (ch, cw))
            if p2["section_id"] in CARVE:
                if CARVE[p2["section_id"]].shape == m2.shape:
                    m2 &= ~CARVE[p2["section_id"]]
                elif not CARVE.get("_warned"):
                    print(f"  carve skipped: duct planes are {CARVE[p2['section_id']].shape}, the canvas is {m2.shape}; rebuild the duct stage")
                    CARVE["_warned"] = True
            md = m2[: shape_d[0] * down: down, : shape_d[1] * down: down]
            if md.any():
                layers[float(p2["z_um"])] = sdf(md)
        if len(layers) < 2:
            print(f"  {cname}: no body (only {len(layers)} layers with a territory)")
            continue
        vol, zg, uns = stack_and_fit(layers, sorted(layers), shape_d, down)
        m = mesh_from(vol, zg, uns, down, smooth_vox=a.smooth)
        if m is None:
            print(f"  {cname}: no surface")
            continue
        nm = "tumour" if cname == "Tumor" else "cls_" + cname
        info = write_mesh(nm, m, OUT)
        info["structure"] = "cell class"
        info["cell_class"] = cname
        info["n_layers"] = len(layers)
        report["meshes"].append(info)
        print(f"  {cname}: {info['n_verts']} verts, {info['n_faces']} faces, "
              f"{len(layers)} layers, {info['frac_fitted_across_gap']:.1%} fitted")

    for m in tls_objects(shape_d, down):
        info = write_mesh(f"tls_{m['_id']}", m, OUT)


        v, f = m["verts"].astype(np.float64), m["faces"]
        vol_um3 = abs(float(np.einsum("ij,ij->i", v[f[:, 0]],
                                      np.cross(v[f[:, 1]], v[f[:, 2]])).sum()) / 6.0)
        info.update({"structure": "TLS object", "object_id": m["_id"],
                     "volume_mm3": round(vol_um3 / 1e9, 6),
                     "p_link": m["_p"], "n_measured_sections": m["_n"],


                     "above_chance": bool(m["_tier"] == "A" or (m["_n"] >= 5 and m["_p"] <= 0.05)),
                     "registration_limited": m["_reglim"]})
        report["meshes"].append(info)
    n_tls = sum(1 for x in report["meshes"] if x.get("structure") == "TLS object")
    print(f"TLS: {n_tls} objects meshed")

    report["surface_smoothing_voxels"] = a.smooth
    report["tls_planes_unmatched"] = globals().get("_TLS_SKIPPED", [])
    report["tls_dirs_paired_by_content"] = globals().get("_TLS_RENAMED", [])
    (OUT / "meshes.json").write_text(json.dumps(report, indent=1))
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
