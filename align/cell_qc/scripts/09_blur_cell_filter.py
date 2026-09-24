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
PAD_PX = 10
MIN_CONTRAST = 12.0
MAX_TENENGRAD = 0.055
MIN_MASK_DELTA = 6.0
MAX_RIM_GRAD = 2.0
RING_PX = 6
EDGE_PX = 2
BLOCK_PX = 4096

RING_K = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * RING_PX + 1,) * 2)
EDGE_K = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * EDGE_PX + 1,) * 2)

PROJ = Path(os.environ.get("PROJECTS_ROOT", "/data/heid"))
QC_ROOT = Path(__file__).resolve().parents[1]
XEN = PROJ / "align/xenium/5k"
ALI = PROJ / "align/registered_he/5k"
OUT = QC_ROOT / "outputs/blur/results"


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


def model_offset(b, x_um, y_um):
    if b is None or b.get("model") is None:
        return None
    if b["model"] == "constant":
        o = b["offset_um"]
        return np.full(len(x_um), o["dx"]), np.full(len(x_um), o["dy"])
    c = b["affine_um"]
    xm, ym = x_um / 1000, y_um / 1000
    return (c["dx"]["const"] + c["dx"]["d_dx_um_per_mm"] * xm + c["dx"]["d_dy_um_per_mm"] * ym,
            c["dy"]["const"] + c["dy"]["d_dx_um_per_mm"] * xm + c["dy"]["d_dy_um_per_mm"] * ym)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--field", required=True)
    ap.add_argument("--cancer", required=True)
    ap.add_argument("--max-tenengrad", type=float, default=MAX_TENENGRAD,
                    help="a cell scoring below this is called out of focus; "
                         "lower = fewer cells flagged")
    ap.add_argument("--min-contrast", type=float, default=MIN_CONTRAST)
    ap.add_argument("--min-mask-delta", type=float, default=MIN_MASK_DELTA,
                    help="the nucleus must differ from the ring around it by at "
                         "least this many grey levels")
    ap.add_argument("--max-rim-grad", type=float, default=MAX_RIM_GRAD,
                    help="a cell whose boundary is sharper than this is kept "
                         "even when its interior matches the surround -- that "
                         "is what a rim-stained nucleus looks like")
    ap.add_argument("--aligned-dir", default=None)
    ap.add_argument("--xenium-dir", default=None)
    ap.add_argument("--per-block-dir", default=str(QC_ROOT / "outputs/results/per_block"))
    ap.add_argument("--white-dir", default=None,
                    help="06_white_cell_filter.py output; cells it already "
                         "dropped as blank are not given a focus verdict")
    ap.add_argument("--out", default=str(OUT))
    a = ap.parse_args()
    log(f"host={os.uname().nodename}  {a.cancer}/{a.field}")

    adir = Path(a.aligned_dir) if a.aligned_dir else ALI / a.cancer / a.field
    xdir = Path(a.xenium_dir) if a.xenium_dir else XEN / a.cancer / a.field
    outd = Path(a.out)

    nb = pd.read_parquet(xdir / "nucleus_boundaries.parquet",
                         columns=["cell_id", "vertex_x", "vertex_y"])
    nb = nb.sort_values("cell_id", kind="stable")
    cid = nb.cell_id.to_numpy()
    vx = nb.vertex_x.to_numpy(np.float64)
    vy = nb.vertex_y.to_numpy(np.float64)
    chg = np.empty(len(cid), bool)
    chg[0] = True
    chg[1:] = cid[1:] != cid[:-1]
    starts = np.flatnonzero(chg)
    ends = np.append(starts[1:], len(cid))
    cell_ids = cid[starts]
    cx = np.add.reduceat(vx, starts) / (ends - starts)
    cy = np.add.reduceat(vy, starts) / (ends - starts)
    log(f"  {len(cell_ids)} cells")

    pb = json.loads((Path(a.per_block_dir) / f"{a.field}.json").read_text())
    info = json.loads((adir / "apply_alignment.json").read_text())
    ccs = (info.get("per_tissue_refinement") or {}).get("per_cc", [])

    blk = np.full(len(cell_ids), -1)
    for k, cc in enumerate(ccs):
        x0, y0, x1, y1 = [v * UM_PER_PX for v in cc["xen_bbox"]]
        blk[(cx >= x0) & (cx < x1) & (cy >= y0) & (cy < y1)] = k
    ddx = np.zeros(len(cell_ids))
    ddy = np.zeros(len(cell_ids))
    for k in range(len(ccs)):
        b = next((x for x in pb["blocks"] if x["block"] - 1 == k), None)
        m = blk == k
        if not m.any():
            continue
        off = model_offset(b, cx[m], cy[m])
        if off is not None:
            ddx[m], ddy[m] = off

    he = zarr_l0(adir / "he_aligned.ome.tif")
    H, W = he.shape[0], he.shape[1]
    px = (cx + ddx) / UM_PER_PX
    py = (cy + ddy) / UM_PER_PX

    n = len(cell_ids)
    contrast = np.full(n, np.nan)
    teneng = np.full(n, np.nan)
    lapstd = np.full(n, np.nan)
    mean_int = np.full(n, np.nan)
    mask_delta = np.full(n, np.nan)
    rim_grad = np.full(n, np.nan)

    done = 0
    for by in range(0, H, BLOCK_PX):
        for bx in range(0, W, BLOCK_PX):
            sel = np.flatnonzero((px >= bx) & (px < bx + BLOCK_PX) &
                                 (py >= by) & (py < by + BLOCK_PX))
            if not len(sel):
                continue
            pad = 64
            y0, y1 = max(0, by - pad), min(H, by + BLOCK_PX + pad)
            x0, x1 = max(0, bx - pad), min(W, bx + BLOCK_PX + pad)
            grey = np.asarray(he[y0:y1, x0:x1]).astype(np.float32).mean(axis=2)
            gx = cv2.Sobel(grey, cv2.CV_32F, 1, 0, ksize=3)
            gy = cv2.Sobel(grey, cv2.CV_32F, 0, 1, ksize=3)
            gmag = np.sqrt(gx * gx + gy * gy) / 4.0
            lap = cv2.Laplacian(grey, cv2.CV_32F, ksize=3)
            for i in sel:
                s, e = starts[i], ends[i]
                qx = np.round((vx[s:e] + ddx[i]) / UM_PER_PX - x0).astype(np.int32)
                qy = np.round((vy[s:e] + ddy[i]) / UM_PER_PX - y0).astype(np.int32)
                mn_x = max(0, qx.min() - PAD_PX)
                mx_x = min(x1 - x0, qx.max() + 1 + PAD_PX)
                mn_y = max(0, qy.min() - PAD_PX)
                mx_y = min(y1 - y0, qy.max() + 1 + PAD_PX)
                if mn_x >= mx_x or mn_y >= mx_y:
                    continue
                cropg = grey[mn_y:mx_y, mn_x:mx_x]
                if cropg.size < 25:
                    continue
                lo, hi = np.percentile(cropg, [2, 98])
                c = float(hi - lo)
                contrast[i] = c
                mean_int[i] = float(cropg.mean())
                if c <= 1e-6:
                    teneng[i] = 0.0
                    lapstd[i] = 0.0
                    continue
                teneng[i] = float(gmag[mn_y:mx_y, mn_x:mx_x].mean()) / c
                lapstd[i] = float(lap[mn_y:mx_y, mn_x:mx_x].std()) / c


                inner = np.zeros(cropg.shape, np.uint8)
                cv2.fillPoly(inner, [np.column_stack([qx - mn_x, qy - mn_y])], 1)
                if inner.sum() < 9:
                    continue
                grown = cv2.dilate(inner, RING_K)
                outer = grown - inner
                if outer.sum() < 9:
                    continue
                mi = float(cropg[inner > 0].mean())
                mo = float(cropg[outer > 0].mean())
                mask_delta[i] = abs(mi - mo)


                band = cv2.dilate(inner, EDGE_K) - cv2.erode(inner, EDGE_K)
                if band.sum() >= 4:
                    rim_grad[i] = float(gmag[mn_y:mx_y, mn_x:mx_x][band > 0].mean())
            done += len(sel)
        log(f"    rows to y={min(by + BLOCK_PX, H)}  cells measured {done}")

    ok = ~np.isnan(teneng)


    if a.white_dir:
        wp = Path(a.white_dir) / f"{a.field}_cells.parquet"
        if wp.exists():
            w = pd.read_parquet(wp, columns=["cell_id", "drop_white"])
            blank = set(w.loc[w.drop_white, "cell_id"])
            already = np.array([c in blank for c in cell_ids])
            log(f"  {int(already.sum())} cells already dropped as blank -- not judged")
            ok = ok & ~already


    judged = ok & (contrast >= a.min_contrast)
    flat = ok & (contrast < a.min_contrast)
    blurred = judged & (teneng < a.max_tenengrad)


    nomask = (judged & ~np.isnan(mask_delta) & (mask_delta < a.min_mask_delta)
              & (rim_grad < a.max_rim_grad))
    drop = flat | blurred | nomask

    df = pd.DataFrame({"cell_id": cell_ids, "block": blk + 1,
                       "x_um": cx, "y_um": cy, "dx_um": ddx, "dy_um": ddy,
                       "contrast": contrast, "tenengrad": teneng,
                       "lap_std": lapstd, "mean_intensity": mean_int,
                       "mask_delta": mask_delta, "rim_grad": rim_grad,
                       "judged": judged, "drop_blur": blurred,
                       "drop_flat": flat, "drop_nomask": nomask,
                       "drop_any": drop})
    outd.mkdir(parents=True, exist_ok=True)
    df.to_parquet(outd / f"{a.field}_cells.parquet", index=False)

    qs = [1, 5, 10, 25, 50, 75, 90, 99]
    summ = {"field": a.field, "cancer": a.cancer,
            "max_tenengrad": a.max_tenengrad, "min_contrast": a.min_contrast,
            "n_cells": int(n), "n_measured": int(ok.sum()),
            "n_judged": int(judged.sum()),
            "n_drop_blur": int(blurred.sum()),
            "frac_drop_blur_of_judged": float(blurred.sum() / max(1, judged.sum())),
            "n_drop_flat": int(flat.sum()),
            "n_drop_nomask": int(nomask.sum()),
            "min_mask_delta": a.min_mask_delta, "max_rim_grad": a.max_rim_grad,
            "n_drop_any": int(drop.sum()),
            "frac_drop_any_of_measured": float(drop.sum() / max(1, ok.sum())),
            "n_drop_overlap": int(int(flat.sum()) + int(blurred.sum())
                                  + int(nomask.sum()) - int(drop.sum())),
            "tenengrad_pct": {f"p{q}": float(np.percentile(teneng[judged], q))
                              for q in qs} if judged.any() else {},
            "contrast_pct": {f"p{q}": float(np.percentile(contrast[ok], q))
                             for q in qs} if ok.any() else {}}
    (outd / f"{a.field}.json").write_text(json.dumps(summ, indent=2))

    for nm, v in (("mask_delta", mask_delta), ("rim_grad", rim_grad)):
        vv = v[~np.isnan(v)]
        if vv.size:
            log(f"  {nm:<11} " + "  ".join(
                f"p{q}={np.percentile(vv, q):.2f}" for q in (1, 5, 10, 25, 50, 75, 90)))
    log(f"  contrast <  {a.min_contrast}: flat, no structure -> drop {flat.sum()}")
    log(f"  contrast >= {a.min_contrast}: {judged.sum()}/{ok.sum()} cells judged")
    log(f"  tenengrad < {a.max_tenengrad}: blurred {blurred.sum()}/{judged.sum()} "
        f"({100 * blurred.sum() / max(1, judged.sum()):.1f}%)")
    log(f"  mask_delta < {a.min_mask_delta} and rim_grad < {a.max_rim_grad}: "
        f"nucleus invisible -> drop {nomask.sum()}")
    log(f"  dropped by any rule: {drop.sum()}/{ok.sum()} "
        f"({100 * drop.sum() / max(1, ok.sum()):.1f}%)")
    if judged.any():
        log("  tenengrad percentiles: " +
            "  ".join(f"p{q}={np.percentile(teneng[judged], q):.4f}" for q in qs))
        log("  contrast  percentiles: " +
            "  ".join(f"p{q}={np.percentile(contrast[ok], q):.1f}" for q in qs))
    return 0


if __name__ == "__main__":
    sys.exit(main())
