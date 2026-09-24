#!/usr/bin/env python3
import os
import csv, json, struct
from pathlib import Path
import numpy as np
import pandas as pd
import cv2
from PIL import Image
from scipy.ndimage import label, binary_dilation, uniform_filter

P3D = Path(os.environ.get("THREED_ROOT", os.path.join(os.environ.get("PROJECTS_ROOT", "/data/heid"), "3d")))
VOL = Path(__import__('os').environ.get("HTAN3D_VOL", str(P3D / "reconstruction/volume_8um_s22M")))
S14 = Path(__import__('os').environ.get("HTAN3D_S14", str(P3D / "objects_3d/outputs/S14_nerve_3d_S22-27909")))
OUTC = Path(__import__('os').environ.get("HTAN3D_S12", str(P3D / "objects_3d/outputs/S12_cloud_S22-27909")))
PRED = Path(__import__('os').environ.get("HTAN3D_PRED", str(P3D / "inference/S22-27909/data/predictions")))
UM = 8.0


_exf = Path(__import__('os').environ.get("HTAN3D_EXCLUDE",
            os.environ.get("THREED_ROOT", os.path.join(os.environ.get("PROJECTS_ROOT", "/data/heid"), "3d")) + "/front/S22-27909/configs/exclude_sections_3d.json"))
EXCLUDE_3D = set()
if _exf.exists():
    import json as _json
    EXCLUDE_3D = set(_json.loads(_exf.read_text())["sections"])

GRID_UM = 20.0
XY_TOL_UM = 60.0
MIN_CELLS = 8
MIN_SECTIONS = 2
BIG_SINGLE = 50


NB_XY_UM = 40.0
NB_Z = 1
MIN_NEIGHBOURS = 8
SIGMA_PX, NORM_GAMMA = 1.5, 1.5

meta = json.loads((VOL / "cells_metadata.json").read_text())
pmeta = json.loads((VOL / "cell_points_metadata.json").read_text())
vmeta = json.loads((VOL / "metadata.json").read_text())
classes = pmeta["classes"]
cw, ch = meta["canvas_px"]["width"], meta["canvas_px"]["height"]
norm_by = {c["name"]: c.get("norm_density_full_alpha_by_basis") or
           {"G_withdrawn": c.get("norm_density_full_alpha", 1.0),
            "G_not_withdrawn": c.get("norm_density_full_alpha", 1.0)}
           for c in meta["classes"]}
tau = json.loads((S14 / "Schwann_nerves.json").read_text())["tau"]
import sys as _sys; _sys.path.insert(0, str(Path(__file__).resolve().parent))
from nerve_members_2d import frozen_points, keep_flags
FROZEN = frozen_points(OUTC)
v3 = {}
for r in csv.DictReader(open(__import__('os').environ.get("HTAN3D_RUNMAN", str(P3D / "inference/S22-27909/configs/s22/run_manifest.tsv"))), delimiter="\t"):
    u = int(r["section_number"])
    if u not in v3 or r["slide"].endswith("r"):
        v3[u] = r["slide"]

def read_plane(basis, p):
    f = sorted((VOL / basis / "cell_points").glob(f"z{p['index']:02d}_*.cells.bin"))
    raw = open(f[0], "rb").read()
    magic, n, _cw, _ch, sub, ncls, _ = struct.unpack("<8sIHHHHI", raw[:24])
    xy = np.frombuffer(raw[32:32 + 4 * n], dtype="<u2").reshape(n, 2).astype(np.float64) / sub
    cl = np.frombuffer(raw[32 + 5 * n:32 + 6 * n], dtype=np.uint8)
    return xy, cl


planes = sorted(vmeta["encodings"]["G_withdrawn"]["planes"], key=lambda p: p["z_um"])
gx = int(np.ceil(cw * UM / GRID_UM)); gy = int(np.ceil(ch * UM / GRID_UM))
K = len(planes)
cells = []
for p in planes:
    xy, cl = read_plane("G_withdrawn", p)
    xi = np.clip((xy[:, 0] * UM / GRID_UM).astype(int), 0, gx - 1)
    yi = np.clip((xy[:, 1] * UM / GRID_UM).astype(int), 0, gy - 1)
    cells.append((xy, cl, xi, yi))
print(f"loaded {sum(len(c[1]) for c in cells):,} cells on {K} planes", flush=True)

dil = max(1, int(round(XY_TOL_UM / GRID_UM)))
keep_mask = [np.zeros(len(c[1]), bool) for c in cells]
schwann_i = classes.index("Schwann")
excl_k = {k for k, p in enumerate(planes) if p["section_id"] in EXCLUDE_3D}
if excl_k:
    print(f"excluded planes (no support, no cells kept): {sorted(excl_k)}", flush=True)
for ci, cname in enumerate(classes):
    occ = np.zeros((K, gy, gx), bool)
    for k, (xy, cl, xi, yi) in enumerate(cells):
        if k in excl_k:
            continue
        s = cl == ci
        occ[k, yi[s], xi[s]] = True
    occ_d = np.stack([binary_dilation(occ[k], iterations=dil) for k in range(K)])
    st = np.zeros((3, 3, 3), bool); st[1] = True; st[0, 1, 1] = True; st[2, 1, 1] = True
    lab, n = label(occ_d, structure=st)
    if n == 0:
        continue

    ncells = np.zeros(n + 1, np.int64)
    zspan = [set() for _ in range(n + 1)]
    ids_per_plane = []
    for k, (xy, cl, xi, yi) in enumerate(cells):
        s = cl == ci
        ids = lab[k, yi[s], xi[s]]
        ids_per_plane.append((s, ids))
        np.add.at(ncells, ids, 1)
        for i2 in np.unique(ids):
            if i2:
                zspan[i2].add(k)
    good = np.zeros(n + 1, bool)
    for i2 in range(1, n + 1):
        ns = len(zspan[i2])
        good[i2] = (ns >= MIN_SECTIONS and ncells[i2] >= MIN_CELLS) or \
                   (ncells[i2] >= BIG_SINGLE)


    cnt = np.zeros((K, occ.shape[1], occ.shape[2]), np.float32)
    for k, (xy, cl, xi, yi) in enumerate(cells):
        if k in excl_k:
            continue
        s2 = cl == ci
        np.add.at(cnt[k], (yi[s2], xi[s2]), 1.0)
    box = int(round(NB_XY_UM / GRID_UM))
    ker = 2 * box + 1
    nb = uniform_filter(cnt, size=(2 * NB_Z + 1, ker, ker), mode="constant") \
        * ((2 * NB_Z + 1) * ker * ker)
    for k, (s, ids) in enumerate(ids_per_plane):
        if not s.any():
            continue
        xi, yi = cells[k][2], cells[k][3]
        local = nb[k, yi[s], xi[s]] >= MIN_NEIGHBOURS
        keep_mask[k][s] = good[ids] & local
    tot = sum(int((cells[k][1] == ci).sum()) for k in range(K))
    kept = sum(int(keep_mask[k][cells[k][1] == ci].sum()) for k in range(K))
    print(f"{cname:<26} kept {kept:>9,}/{tot:>9,} ({kept/max(1,tot):.1%})", flush=True)


TAU_IN_FRAC = 0.5
cls_den = [None] * K
n_boost = n_in = n_fr = 0
for k, p in enumerate(planes):
    xy, cl, xi, yi = cells[k]
    s = cl == schwann_i
    keep_mask[k][s] = False
    if p["section_id"] in EXCLUDE_3D:
        keep_mask[k][:] = False
        continue
    nf = sorted((VOL / "G_withdrawn/nerve").glob(f"z{p['index']:02d}_*.nerve.webp"))
    if not nf:
        continue
    a = np.asarray(Image.open(nf[0]).convert("RGBA"))[:, :, 3]
    u = int(p["section_id"].rsplit("_U", 1)[1])
    pred = pd.read_parquet(PRED / f"{v3[u]}.parquet", columns=["schwann_prob"])
    assert len(pred) == len(cl)
    sel = pred["schwann_prob"].to_numpy() >= tau
    cx = np.clip(np.round(xy[:, 0]).astype(int), 0, a.shape[1] - 1)
    cy = np.clip(np.round(xy[:, 1]).astype(int), 0, a.shape[0] - 1)
    if FROZEN is not None:


        inside = a[cy, cx] > 0
        frozen = keep_flags(FROZEN, p["z_um"], xy)
        boost = (~s) & inside & (pred["schwann_prob"].to_numpy() >= TAU_IN_FRAC * tau)
        keep_mask[k][frozen | (s & inside) | boost] = True
        cl2 = cl.copy(); cl2[boost | (frozen & ~s)] = schwann_i
        cls_den[k] = cl2
        n_boost += int(boost.sum()); n_in += int((s & inside & ~frozen).sum()); n_fr += int(frozen.sum())
    else:
        keep_mask[k][s & sel & (a[cy, cx] > 0)] = True
if FROZEN is not None:
    print(f"Schwann Denoised: {n_fr:,} frozen-cloud cells + {n_in:,} more Schwann-class cells inside 3-D nerve regions + {n_boost:,} other-class cells called Schwann at tau x {TAU_IN_FRAC:g}", flush=True)

from nerve_members_2d import apply_local_schwann_rescue
n_local_rescued = apply_local_schwann_rescue(VOL, planes, cells, keep_mask, classes, EXCLUDE_3D)
print(f'Local reviewed Schwann rescue: {n_local_rescued:,} cells', flush=True)


ks = int(2 * round(3 * SIGMA_PX) + 1)
files = {c: {} for c in classes}
for basis in ("G_withdrawn", "G_not_withdrawn"):
    bplanes = sorted(vmeta["encodings"][basis]["planes"], key=lambda p: p["index"])
    for p in bplanes:
        k = next(i for i, q in enumerate(planes) if q["section_id"] == p["section_id"])
        xy, cl = read_plane(basis, p)
        km = keep_mask[k]
        assert len(km) == len(cl)
        if cls_den[k] is not None:
            cl = cls_den[k]
        xi = np.clip(np.round(xy[:, 0]).astype(int), 0, cw - 1)
        yi = np.clip(np.round(xy[:, 1]).astype(int), 0, ch - 1)
        for ci, cname in enumerate(classes):
            s = (cl == ci) & km
            g = np.zeros((ch, cw), np.float32)
            if s.any():
                np.add.at(g, (yi[s], xi[s]), 1.0)
            g = cv2.GaussianBlur(g, (ks, ks), SIGMA_PX)
            al = np.clip(g / max(norm_by[cname][basis], 1e-9), 0, 1) ** NORM_GAMMA
            a8 = (al * 255.0 + 0.5).astype(np.uint8)
            rgba = np.dstack([np.full_like(a8, 255)] * 3 + [a8])
            fc = cname.replace("/", "_")
            fn = f"z{p['index']:02d}_{p['section_id'].split('-')[1]}.{fc}_denoised3d.webp"
            Image.fromarray(rgba).save(VOL / basis / "cells" / fn, format="WEBP",
                                       lossless=True, quality=100, method=4)
            if basis == "G_withdrawn":
                files[cname][p["index"]] = fn
        print(basis, p["section_id"], flush=True)


kd = VOL / "keep3d"
kd.mkdir(exist_ok=True)
for k, p2 in enumerate(planes):
    (kd / f"z{p2['index']:02d}.keep3d.bin").write_bytes(
        keep_mask[k].astype(np.uint8).tobytes())
print("keep3d masks written", flush=True)

(VOL / "cells_denoised3d_metadata.json").write_text(json.dumps({
    "what": "per-class 3-D support denoise (>=2 sections & >=5 cells, or >=30 "
            "single-section; 20 um grid, 60 um lateral tolerance). Schwann = FROZEN "
            "3-D denoise cloud + Schwann-class cells INSIDE the 3-D nerves' regions + in-region lowered-tau calls.",
    "grid_um": GRID_UM, "xy_tol_um": XY_TOL_UM, "min_cells": MIN_CELLS,
    "min_sections": MIN_SECTIONS, "big_single": BIG_SINGLE, "tau": tau,
    "classes": classes,
    "files": {c: [{"index": i, "file": f} for i, f in sorted(files[c].items())]
              for c in classes},
    "planes": [{"index": i, "file": f} for i, f in sorted(files["Schwann"].items())]},
    indent=1))


zt = [float(p["z_um"]) for p in planes]
UM_PX = 1.0
per_cls = []


_BUDGET, _rng = 200_000, np.random.default_rng(20260819)
for ci in range(len(classes)):
    rows = []
    for k, (xy, cl, xi, yi) in enumerate(cells):
        if cls_den[k] is not None:
            cl = cls_den[k]
        s = (cl == ci) & keep_mask[k]
        if s.any():
            um_xy = xy[s] * UM
            rows.append(np.column_stack([um_xy, np.full(int(s.sum()), k)]))
    q = np.vstack(rows) if rows else np.zeros((0, 3))
    if len(q) > _BUDGET:
        q = q[np.sort(_rng.choice(len(q), size=_BUDGET, replace=False))]
    per_cls.append(q)
b = bytearray(8); b[7] = len(classes)
for a2 in per_cls:
    b += struct.pack("<I", len(a2))
for a2 in per_cls:
    arr = np.empty((len(a2), 3), np.uint16)
    if len(a2):
        arr[:, 0] = np.round(a2[:, 0] / UM_PX); arr[:, 1] = np.round(a2[:, 1] / UM_PX)
        arr[:, 2] = a2[:, 2]
    b += arr.tobytes()
while len(b) % 4: b += b"\0"
b += np.asarray(zt, np.float32).tobytes()
OUTC.mkdir(parents=True, exist_ok=True)
(OUTC / "denoised_cloud.bin").write_bytes(bytes(b))
(OUTC / "denoised_cloud.json").write_text(json.dumps({
    "what": "all-class 3-D-support denoised cells (Schwann = frozen 3-D denoise cloud + cells inside the 3-D nerves' regions)",
    "classes": classes, "n": int(sum(len(a2) for a2 in per_cls)),
    "n_planes": len(zt), "um_px": UM_PX}, indent=1))
print("cloud:", sum(len(a2) for a2 in per_cls), "cells")
print("DONE")
