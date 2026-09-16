#!/usr/bin/env python
import argparse
import json
import os
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import tifffile

UM_PER_PX = 0.2125
TILE = 1024
SEARCH = 96
BLUR_SIGMA = 1.5
MIN_CELLS = 40
MIN_MASK_FRAC = 0.02
MIN_PROM = 0.02
N_TILES = 60

PROJ = Path(os.environ.get("PROJECTS_ROOT", "/data/heid"))
QC_ROOT = Path(__file__).resolve().parents[1]
SIGN_TEST_ROOT = Path(__file__).resolve().parents[1]
XEN = PROJ / "data/X1000/Xenium/5k"
ALI = PROJ / "align/registered_he/5k"


def log(m):
    print(m, flush=True)


def zarr_l0(p):
    import zarr

    g = zarr.open(tifffile.imread(str(p), aszarr=True), mode="r")
    if hasattr(g, "shape"):
        return g
    for k in ("0", 0):
        try:
            return g[k]
        except (TypeError, KeyError):
            continue
    raise RuntimeError("no level 0")


def hematoxylin(rgb):
    a = rgb.astype(np.float32) / 255.0
    a = np.clip(a, 1e-3, 1.0)
    od = -np.log(a)
    return od[..., 2] if od.ndim == 3 else od


def match(mask, img):
    if mask.mean() < MIN_MASK_FRAC:
        return None
    m = cv2.GaussianBlur(mask.astype(np.float32), (0, 0), BLUR_SIGMA)
    i = cv2.GaussianBlur(img.astype(np.float32), (0, 0), BLUR_SIGMA)
    t = m[SEARCH:-SEARCH, SEARCH:-SEARCH]
    if t.shape[0] < 32 or t.shape[1] < 32:
        return None
    r = cv2.matchTemplate(i, t, cv2.TM_CCOEFF_NORMED)
    _, peak, _, loc = cv2.minMaxLoc(r)
    j, k = loc[1], loc[0]
    far = r.copy()
    lo_j, hi_j = max(0, j - 10), min(r.shape[0], j + 11)
    lo_k, hi_k = max(0, k - 10), min(r.shape[1], k + 11)
    far[lo_j:hi_j, lo_k:hi_k] = -1
    prom = peak - float(far.max())

    return (k - SEARCH) * UM_PER_PX, (j - SEARCH) * UM_PER_PX, prom, peak


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--field", required=True)
    ap.add_argument("--cancer", required=True)
    ap.add_argument("--n-tiles", type=int, default=N_TILES)
    ap.add_argument("--cell-keep-dir",
                    default=str(QC_ROOT / "outputs/full/cell_keep"))
    ap.add_argument("--tiles-dir", default=str(QC_ROOT / "outputs/full/offsets"))
    ap.add_argument("--aligned-dir", default=None)
    ap.add_argument("--xenium-dir", default=None)
    ap.add_argument("--out", default=str(SIGN_TEST_ROOT / "outputs/validation/sign_test"))
    a = ap.parse_args()
    log(f"host={os.uname().nodename}  {a.cancer}/{a.field}")

    adir = Path(a.aligned_dir) if a.aligned_dir else ALI / a.cancer / a.field
    xdir = Path(a.xenium_dir) if a.xenium_dir else XEN / a.cancer / a.field

    ck = pd.read_parquet(Path(a.cell_keep_dir) / f"{a.field}_cells.parquet",
                         columns=["cell_id", "x_um", "y_um", "dx_um", "dy_um",
                                  "has_model", "keep"])
    ck = ck[ck.has_model]
    if ck.empty:
        log("  no cell has a model"); return 1
    off = ck.set_index("cell_id")[["dx_um", "dy_um"]]

    nb = pd.read_parquet(xdir / "nucleus_boundaries.parquet",
                         columns=["cell_id", "vertex_x", "vertex_y"])
    nb = nb.join(off, on="cell_id", how="inner")

    t = pd.read_csv(Path(a.tiles_dir) / f"{a.field}_tiles.csv")
    good = t[(t.status == "ok") & (t.he_prominence >= MIN_PROM)
             & (~t.he_on_border.fillna(True).astype(bool))]
    good = good.nlargest(a.n_tiles, "n_nuclei")
    log(f"  {len(good)} tiles, {len(ck)} cells with a model")

    he = zarr_l0(adir / "he_aligned.ome.tif")
    H, W = he.shape[0], he.shape[1]
    arms = {"raw": (0.0, 0.0), "plus": (+1.0, +1.0), "minus": (-1.0, -1.0)}
    res = {k: [] for k in arms}

    for _, r in good.iterrows():
        x0, y0 = int(r.x0), int(r.y0)
        if x0 < 0 or y0 < 0 or y0 + TILE > H or x0 + TILE > W:
            continue
        img = hematoxylin(np.asarray(he[y0:y0 + TILE, x0:x0 + TILE]))
        sub = nb[(nb.vertex_x / UM_PER_PX >= x0 - 64)
                 & (nb.vertex_x / UM_PER_PX < x0 + TILE + 64)
                 & (nb.vertex_y / UM_PER_PX >= y0 - 64)
                 & (nb.vertex_y / UM_PER_PX < y0 + TILE + 64)]
        if sub.cell_id.nunique() < MIN_CELLS:
            continue
        for arm, (sx, sy) in arms.items():
            mask = np.zeros((TILE, TILE), np.uint8)
            for _, g in sub.groupby("cell_id", sort=False):
                px = ((g.vertex_x + sx * g.dx_um) / UM_PER_PX - x0).to_numpy()
                py = ((g.vertex_y + sy * g.dy_um) / UM_PER_PX - y0).to_numpy()
                cv2.fillPoly(mask, [np.column_stack([px, py]).round().astype(np.int32)], 1)
            m = match(mask, img)
            if m is None:
                continue
            dx, dy, prom, peak = m
            if prom >= MIN_PROM:
                res[arm].append((dx, dy))

    summ = {"field": a.field, "cancer": a.cancer}
    log(f"  {'arm':<8}{'n':>5}{'dx_um':>10}{'dy_um':>10}{'magnitude':>11}")
    for arm in ("raw", "plus", "minus"):
        v = np.array(res[arm])
        if not len(v):
            log(f"  {arm:<8}{0:>5}  no tile correlated")
            continue
        dx, dy = float(np.median(v[:, 0])), float(np.median(v[:, 1]))
        mag = float(np.hypot(dx, dy))
        summ[arm] = dict(n=len(v), dx_um=dx, dy_um=dy, magnitude_um=mag)
        log(f"  {arm:<8}{len(v):>5}{dx:>10.3f}{dy:>10.3f}{mag:>11.3f}")

    if "plus" in summ and "minus" in summ:
        p, m = summ["plus"]["magnitude_um"], summ["minus"]["magnitude_um"]
        summ["verdict"] = ("SIGN_OK" if p < m * 0.6 else
                           "SIGN_INVERTED" if m < p * 0.6 else
                           "AMBIGUOUS")
        log(f"  verdict: {summ['verdict']}  (plus {p:.3f} vs minus {m:.3f} µm)")

    outd = Path(a.out)
    outd.mkdir(parents=True, exist_ok=True)
    (outd / f"{a.field}.json").write_text(json.dumps(summ, indent=2))
    log(f"  wrote {outd / f'{a.field}.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
