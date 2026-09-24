#!/usr/bin/env python3
import os
import csv, json, struct
from pathlib import Path
import numpy as np, pandas as pd, cv2
from PIL import Image

P3D = Path(os.environ.get("THREED_ROOT", os.path.join(os.environ.get("PROJECTS_ROOT", "/data/heid"), "3d")))
VOL = Path(__import__('os').environ.get("HTAN3D_VOL", str(P3D / "reconstruction/volume_8um_s22M")))
S14 = Path(__import__('os').environ.get("HTAN3D_S14", str(P3D / "objects_3d/outputs/S14_nerve_3d_S22-27909")))
S12 = Path(__import__('os').environ.get("HTAN3D_S12", str(P3D / "objects_3d/outputs/S12_cloud_S22-27909")))
PRED = Path(__import__('os').environ.get("HTAN3D_PRED", str(P3D / "inference/S22-27909/data/predictions")))
G = Path(__import__('os').environ.get("HTAN3D_GEN", str(P3D / "front/S22-27909")))
SIGMA_PX, NORM_GAMMA = 1.5, 1.5
tau = json.loads((S14 / "Schwann_nerves.json").read_text())["tau"]
import sys as _sys; _sys.path.insert(0, str(Path(__file__).resolve().parent))
from nerve_members_2d import frozen_points, keep_flags
FROZEN = frozen_points(S12)
print("Schwann denoised set:", "frozen 3-D denoise cloud + Schwann-class cells inside the 3-D nerves' regions" if FROZEN is not None else "tau cells inside accepted regions (frozen denoise not built yet)", flush=True)
v3 = {}
for r in csv.DictReader(open(__import__('os').environ.get("HTAN3D_RUNMAN", str(P3D / "inference/S22-27909/configs/s22/run_manifest.tsv"))), delimiter="\t"):
    u = int(r["section_number"])
    if u not in v3 or r["slide"].endswith("r"):
        v3[u] = r["slide"]
meta = json.loads((VOL / "cells_metadata.json").read_text())
SCHWANN_I = json.loads((VOL / "cell_points_metadata.json").read_text())["classes"].index("Schwann")
TAU_IN_FRAC = 0.5
vmeta = json.loads((VOL / "metadata.json").read_text())
cw, ch = meta["canvas_px"]["width"], meta["canvas_px"]["height"]
norm_by = {c["name"]: c.get("norm_density_full_alpha_by_basis") or
           {"G_withdrawn": c.get("norm_density_full_alpha", 1.0),
            "G_not_withdrawn": c.get("norm_density_full_alpha", 1.0)}
           for c in meta["classes"]}
NRM = norm_by["Schwann"]
ks = int(2 * round(3 * SIGMA_PX) + 1)
rows = []
for basis in ("G_withdrawn", "G_not_withdrawn"):
    planes = sorted(vmeta["encodings"][basis]["planes"], key=lambda p: p["index"])
    for p in planes:
        nf = sorted((VOL / basis / "nerve").glob(f"z{p['index']:02d}_*.nerve.webp"))
        f = sorted((VOL / basis / "cell_points").glob(f"z{p['index']:02d}_*.cells.bin"))
        if not nf or not f:
            continue
        a = np.asarray(Image.open(nf[0]).convert("RGBA"))[:, :, 3]
        raw = open(f[0], "rb").read()
        magic, n, _cw, _ch, sub, ncls, _ = struct.unpack("<8sIHHHHI", raw[:24])
        xy = np.frombuffer(raw[32:32 + 4 * n], dtype="<u2").reshape(n, 2).astype(np.float64) / sub
        u = int(p["section_id"].rsplit("_U", 1)[1])
        pred = pd.read_parquet(PRED / f"{v3[u]}.parquet", columns=["schwann_prob"])
        assert len(pred) == n
        sel = pred["schwann_prob"].to_numpy() >= tau
        xi = np.clip(np.round(xy[:, 0]).astype(int), 0, cw - 1)
        yi = np.clip(np.round(xy[:, 1]).astype(int), 0, ch - 1)
        keep = sel & (a[np.clip(yi, 0, a.shape[0]-1), np.clip(xi, 0, a.shape[1]-1)] > 0)
        if FROZEN is not None:


            cl = np.frombuffer(raw[32 + 5 * n:32 + 6 * n], dtype=np.uint8)
            inside = a[np.clip(yi, 0, a.shape[0]-1), np.clip(xi, 0, a.shape[1]-1)] > 0
            frozen = keep_flags(FROZEN, p["z_um"], xy)
            keep = frozen | (inside & ((cl == SCHWANN_I) | (pred["schwann_prob"].to_numpy() >= TAU_IN_FRAC * tau)))

        fn = None
        if basis == "G_withdrawn":
            rows.append({"index": p["index"], "section_id": p["section_id"], "n_cells": int(keep.sum())})
            kd = VOL / "keep2d"
            kd.mkdir(exist_ok=True)
            (kd / f"z{p['index']:02d}.keep2d.bin").write_bytes(
                keep.astype(np.uint8).tobytes())
        print(basis, p["section_id"], int(keep.sum()), flush=True)
(VOL / "keep2d_metadata.json").write_text(json.dumps({
    "what": ("keep2d: per-cell flag of the Schwann Denoised set = FROZEN 3-D denoise cloud + Schwann-class cells INSIDE the 3-D nerves' regions + in-region lowered-tau calls; "
             if FROZEN is not None else "Schwann density from tau-gated cells INSIDE accepted nerve regions; ")
            + "same sigma/gamma/norm as the raw planes",
    "tau": tau, "n_planes": len(rows), "planes": rows}, indent=1))
print("done", len(rows))
