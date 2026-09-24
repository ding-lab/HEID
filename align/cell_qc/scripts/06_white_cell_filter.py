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


WHITE_INTENSITY = 235
MAX_WHITE_FRAC = 0.60
BLOCK_PX = 4096

PROJ = Path(os.environ.get("PROJECTS_ROOT", "/data/heid"))
QC_ROOT = Path(__file__).resolve().parents[1]
XEN = PROJ / "align/xenium/5k"
ALI = PROJ / "align/registered_he/5k"
OUT = QC_ROOT / "outputs/results/white_filter"


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
    ap.add_argument("--white-intensity", type=int, default=WHITE_INTENSITY)
    ap.add_argument("--max-white-frac", type=float, default=MAX_WHITE_FRAC)
    ap.add_argument("--aligned-dir", default=None,
                    help="the registered_he field directory; give it when the "
                         "directory name carries a suffix such as -lowqc")
    ap.add_argument("--xenium-dir", default=None,
                    help="the Xenium bundle directory")
    ap.add_argument("--per-block-dir", default=str(QC_ROOT / "outputs/results/per_block"),
                    help="where 04_per_block_offsets.py wrote <field>.json")
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
    has_model = np.zeros(len(cell_ids), bool)
    for k in range(len(ccs)):
        b = next((x for x in pb["blocks"] if x["block"] - 1 == k), None)
        m = blk == k
        if not m.any():
            continue
        off = model_offset(b, cx[m], cy[m])
        if off is None:
            continue
        ddx[m], ddy[m] = off
        has_model[m] = True
    log(f"  {has_model.sum()} cells in a block with a model, "
        f"{(~has_model).sum()} without")

    he = zarr_l0(adir / "he_aligned.ome.tif")
    H, W = he.shape[0], he.shape[1]

    white_frac = np.full(len(cell_ids), np.nan)
    mean_int = np.full(len(cell_ids), np.nan)
    n_px = np.zeros(len(cell_ids), int)

    px = (cx + ddx) / UM_PER_PX
    py = (cy + ddy) / UM_PER_PX

    order = np.argsort((py // BLOCK_PX) * 10**6 + (px // BLOCK_PX))
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
            tile = np.asarray(he[y0:y1, x0:x1])
            grey = tile.astype(np.float32).mean(axis=2)
            blank = (grey > a.white_intensity).astype(np.uint8)
            for i in sel:
                s, e = starts[i], ends[i]
                qx = np.round((vx[s:e] + ddx[i]) / UM_PER_PX - x0).astype(np.int32)
                qy = np.round((vy[s:e] + ddy[i]) / UM_PER_PX - y0).astype(np.int32)
                if qx.max() < 0 or qy.max() < 0 or qx.min() >= x1 - x0 or qy.min() >= y1 - y0:
                    continue
                mn_x, mx_x = max(0, qx.min()), min(x1 - x0, qx.max() + 1)
                mn_y, mx_y = max(0, qy.min()), min(y1 - y0, qy.max() + 1)
                if mn_x >= mx_x or mn_y >= mx_y:
                    continue
                m = np.zeros((mx_y - mn_y, mx_x - mn_x), np.uint8)
                cv2.fillPoly(m, [np.column_stack([qx - mn_x, qy - mn_y])], 1)
                k = int(m.sum())
                if k == 0:
                    continue
                n_px[i] = k
                white_frac[i] = float((blank[mn_y:mx_y, mn_x:mx_x] * m).sum()) / k
                mean_int[i] = float((grey[mn_y:mx_y, mn_x:mx_x] * m).sum()) / k
            done += len(sel)
        log(f"    rows to y={min(by + BLOCK_PX, H)}  cells measured {done}")

    ok = ~np.isnan(white_frac)
    drop = ok & (white_frac >= a.max_white_frac)
    df = pd.DataFrame({"cell_id": cell_ids, "block": blk + 1,
                       "x_um": cx, "y_um": cy,
                       "dx_um": ddx, "dy_um": ddy, "has_model": has_model,
                       "nucleus_px": n_px, "white_frac": white_frac,
                       "mean_intensity": mean_int, "drop_white": drop})
    outd.mkdir(parents=True, exist_ok=True)
    df.to_parquet(outd / f"{a.field}_cells.parquet", index=False)

    summ = {"field": a.field, "cancer": a.cancer,
            "white_intensity": a.white_intensity,
            "max_white_frac": a.max_white_frac,
            "n_cells": int(len(cell_ids)),
            "n_measured": int(ok.sum()),
            "n_drop_white": int(drop.sum()),


            "frac_drop_white": float((drop & has_model).sum()
                                     / max(1, (ok & has_model).sum())),
            "frac_drop_white_all_cells": float(drop.sum() / max(1, ok.sum())),
            "n_no_model": int((~has_model).sum()),
            "per_block": {}}
    for k in range(len(ccs)):
        m = (blk == k) & ok
        if not m.any():
            continue
        summ["per_block"][f"blk{k+1}"] = {
            "n_cells": int(m.sum()),
            "n_drop_white": int((m & drop).sum()),
            "frac_drop_white": float((m & drop).sum() / m.sum()),
            "white_frac_median": float(np.median(white_frac[m])),
            "white_frac_p90": float(np.percentile(white_frac[m], 90)),
            "mean_intensity_median": float(np.median(mean_int[m])),
            "mean_intensity_p25": float(np.percentile(mean_int[m], 25))}
    (outd / f"{a.field}.json").write_text(json.dumps(summ, indent=2))

    log(f"  white_frac >= {a.max_white_frac} (pixel > {a.white_intensity}): "
        f"drop {drop.sum()}/{ok.sum()} ({100*drop.sum()/max(1,ok.sum()):.1f}%)")
    for k, v in summ["per_block"].items():
        log(f"    {k}: drop {v['n_drop_white']}/{v['n_cells']} "
            f"({100*v['frac_drop_white']:.1f}%)  "
            f"white_frac med {v['white_frac_median']:.3f} p90 {v['white_frac_p90']:.3f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
