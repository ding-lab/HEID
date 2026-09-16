#!/usr/bin/env python
import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

UM_PER_PX = 0.2125
PROJ = Path(os.environ.get("PROJECTS_ROOT", "/data/heid"))
OFFSETS_ROOT = Path(__file__).resolve().parents[1] / "outputs/offsets"
ALI = PROJ / "align/registered_he/5k"
OUT = Path(__file__).resolve().parents[1] / "outputs/results/per_block"


MAX_MAD_UM = 1.0
MAX_SE_UM = 0.30
MIN_TILES = 30
MAX_AFFINE_P90 = 1.35
MIN_GRAD_UM_PER_MM = 0.05
MAX_GRAD_UM_PER_MM = 2.0


def log(m):
    print(m, flush=True)


def mad(v):
    return float(np.median(np.abs(v - np.median(v))))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--field", required=True)
    ap.add_argument("--cancer", required=True)
    ap.add_argument("--max-mad", type=float, default=MAX_MAD_UM)
    ap.add_argument("--max-se", type=float, default=MAX_SE_UM)
    ap.add_argument("--min-tiles", type=int, default=MIN_TILES)
    ap.add_argument("--min-grad", type=float, default=MIN_GRAD_UM_PER_MM)
    ap.add_argument("--max-grad", type=float, default=MAX_GRAD_UM_PER_MM)
    ap.add_argument("--max-affine-p90", type=float, default=MAX_AFFINE_P90,
                    help="if the constant fails, keep the block when an affine "
                         "fit leaves a residual p90 below this")
    ap.add_argument("--tiles-dir", default=str(OFFSETS_ROOT),
                    help="where <field>_tiles.csv lives")
    ap.add_argument("--aligned-dir", default=None,
                    help="the registered_he field directory; give it when the "
                         "directory name carries a suffix such as -lowqc")
    ap.add_argument("--out", default=str(OUT))
    a = ap.parse_args()
    log(f"host={os.uname().nodename}  {a.cancer}/{a.field}")

    adir = Path(a.aligned_dir) if a.aligned_dir else ALI / a.cancer / a.field
    t = pd.read_csv(Path(a.tiles_dir) / f"{a.field}_tiles.csv")
    g = t[(t.status == "ok") & (t.he_prominence >= 0.02) &
          (~t.he_on_border.fillna(True).astype(bool))].copy()
    info = json.loads((adir / "apply_alignment.json").read_text())
    ccs = (info.get("per_tissue_refinement") or {}).get("per_cc", [])

    g["cc"] = -1
    for k, cc in enumerate(ccs):
        x0, y0, x1, y1 = cc["xen_bbox"]
        g.loc[(g.cx_px >= x0) & (g.cx_px < x1) &
              (g.cy_px >= y0) & (g.cy_px < y1), "cc"] = k

    rng = np.random.default_rng(0)
    blocks = []
    for k in range(len(ccs)):
        s = g[g.cc == k]
        b = {"block": k + 1, "cc_idx": ccs[k]["cc_idx"],
             "area_mm2": ccs[k]["area_mm2"], "n_tiles": int(len(s))}
        if len(s) < a.min_tiles:
            b.update(verdict="NO_MODEL", model=None,
                     reason=f"only {len(s)} usable tiles (<{a.min_tiles})",
                     offset_um=None)
            blocks.append(b)
            continue
        dx, dy = float(s.he_sx_um.median()), float(s.he_sy_um.median())
        resid = np.hypot(s.he_sx_um - dx, s.he_sy_um - dy)
        disp_mad = float(np.hypot(mad(s.he_sx_um), mad(s.he_sy_um)))
        n = len(s)
        bx = [np.median(s.he_sx_um.values[rng.integers(0, n, n)]) for _ in range(300)]
        by = [np.median(s.he_sy_um.values[rng.integers(0, n, n)]) for _ in range(300)]
        se = float(np.hypot(np.std(bx), np.std(by)))


        xm = s.cx_px.to_numpy() * UM_PER_PX / 1000
        ym = s.cy_px.to_numpy() * UM_PER_PX / 1000
        A = np.column_stack([np.ones(n), xm, ym])
        coef, res_a = {}, []
        for nm, col in (("dx", "he_sx_um"), ("dy", "he_sy_um")):
            c, *_ = np.linalg.lstsq(A, s[col].to_numpy(), rcond=None)
            coef[nm] = {"const": float(c[0]), "d_dx_um_per_mm": float(c[1]),
                        "d_dy_um_per_mm": float(c[2])}
            res_a.append(s[col].to_numpy() - A @ c)
        resid_aff = np.hypot(*res_a)
        aff_p90 = float(np.percentile(resid_aff, 90))
        aff_med = float(np.median(resid_aff))

        grad = float(max(abs(coef["dx"]["d_dx_um_per_mm"]), abs(coef["dx"]["d_dy_um_per_mm"]),
                         abs(coef["dy"]["d_dx_um_per_mm"]), abs(coef["dy"]["d_dy_um_per_mm"])))
        affine_ok = aff_p90 <= a.max_affine_p90 and grad <= a.max_grad
        const_ok = disp_mad <= a.max_mad and se <= a.max_se
        if affine_ok and grad >= a.min_grad:
            model = "affine"
        elif const_ok:
            model = "constant"
        elif affine_ok:
            model = "affine"
        else:
            model = None
        keep = model is not None

        b.update(
            model=model,
            offset_um=({"dx": dx, "dy": dy, "magnitude": float(np.hypot(dx, dy))}
                       if model == "constant" else None),
            affine_um=(coef if model == "affine" else None),
            disp_mad_um=disp_mad, se_um=se,
            residual_um={"median": float(np.median(resid)),
                         "p90": float(np.percentile(resid, 90)),
                         "p99": float(np.percentile(resid, 99))},
            residual_after_affine_um={"median": aff_med, "p90": aff_p90},
            max_gradient_um_per_mm=grad,
            ncc_peak_median=float(s.he_peak.median()),
            ncc_prominence_median=float(s.he_prominence.median()),
            verdict="MODELLED" if keep else "NO_MODEL")
        if model == "affine":
            b["reason"] = (f"gradient {grad:.3f} um/mm is real; affine residual p90 "
                           f"{aff_p90:.2f} um vs constant {np.percentile(resid, 90):.2f}")
        elif model == "constant":
            b["reason"] = f"gradient {grad:.3f} um/mm negligible; constant suffices"
        else:
            b["reason"] = (f"no model: constant MAD {disp_mad:.2f} (limit {a.max_mad}), "
                           f"affine residual p90 {aff_p90:.2f} (limit {a.max_affine_p90}), "
                           f"gradient {grad:.2f} um/mm (limit {a.max_grad})")
        if "dapi_sx_um" in s.columns and s.dapi_sx_um.notna().any():
            d = s.dropna(subset=["dapi_sx_um"])
            b["self_control_um"] = {
                "magnitude": float(np.hypot(d.dapi_sx_um.median(), d.dapi_sy_um.median())),
                "mad": float(np.hypot(mad(d.dapi_sx_um), mad(d.dapi_sy_um)))}
        blocks.append(b)

    kept = [b for b in blocks if b["verdict"] == "MODELLED"]
    res = {
        "field": a.field, "cancer": a.cancer,
        "n_xenium_samples": info.get("n_xenium_samples"),
        "n_ccs": len(ccs),
        "thresholds": {"max_disp_mad_um": a.max_mad, "max_se_um": a.max_se,
                       "min_tiles": a.min_tiles,
                       "max_affine_resid_p90_um": a.max_affine_p90,
                       "min_gradient_um_per_mm": a.min_grad,
                       "max_gradient_um_per_mm": a.max_grad,
                       "note": "provisional, set from the AL008B1 block separation"},
        "blocks": blocks,
        "n_blocks_kept": len(kept),
        "coverage": {
            "tiles_kept": int(sum(b["n_tiles"] for b in kept)),
            "tiles_total": int(len(g)),
            "area_kept_mm2": float(sum(b["area_mm2"] for b in kept)),
            "area_total_mm2": float(sum(c["area_mm2"] for c in ccs))},
        "apply_rule": "a cell takes the offset its block's model predicts at that "
                      "position; cells without a block model are excluded by the "
                      "downstream cell-keep table. Image-content exclusions are "
                      "also applied per cell.",
        "verdict": ("ALL_BLOCKS_MODELLED" if len(kept) == len(ccs)
                    else "PARTIAL" if kept else "NO_BLOCK_MODELLED"),
    }

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / f"{a.field}.json").write_text(json.dumps(res, indent=2))

    log(f"  n_xenium={res['n_xenium_samples']}  n_ccs={res['n_ccs']}  -> {res['verdict']}")
    for b in blocks:
        if b["verdict"] == "MODELLED" and b.get("model") == "constant":
            o = b["offset_um"]
            log(f"    blk{b['block']}  constant  ({o['dx']:+.3f},{o['dy']:+.3f}) um  "
                f"MAD {b['disp_mad_um']:.3f}  SE {b['se_um']:.3f}  "
                f"resid p90 {b['residual_um']['p90']:.3f}  n={b['n_tiles']}")
        elif b["verdict"] == "MODELLED":
            c = b["affine_um"]
            log(f"    blk{b['block']}  affine     const=({c['dx']['const']:+.2f},"
                f"{c['dy']['const']:+.2f})  grad dx/dy per mm "
                f"({c['dx']['d_dx_um_per_mm']:+.2f},{c['dx']['d_dy_um_per_mm']:+.2f})/"
                f"({c['dy']['d_dx_um_per_mm']:+.2f},{c['dy']['d_dy_um_per_mm']:+.2f})  "
                f"resid p90 {b['residual_after_affine_um']['p90']:.3f}  n={b['n_tiles']}")
        else:
            log(f"    blk{b['block']}  no model  {b['reason']}  n={b['n_tiles']}")
    c = res["coverage"]
    log(f"  modelled {c['tiles_kept']}/{c['tiles_total']} tiles, "
        f"{c['area_kept_mm2']:.1f}/{c['area_total_mm2']:.1f} mm²")
    return 0


if __name__ == "__main__":
    sys.exit(main())
