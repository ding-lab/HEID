#!/usr/bin/env python
import argparse
import json
import os
import sys
from pathlib import Path

import pandas as pd

UM_PER_PX = 0.2125
PROJ = Path(os.environ.get("PROJECTS_ROOT", "/data/heid"))
QC_ROOT = Path(__file__).resolve().parents[1]
WHITE = QC_ROOT / "outputs/results/white_filter"
BLUR = QC_ROOT / "outputs/blur/results"
OUT = QC_ROOT / "outputs/results/cell_keep"


def log(m):
    print(m, flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--field", required=True)
    ap.add_argument("--cancer", required=True)
    ap.add_argument("--white-dir", default=str(WHITE))
    ap.add_argument("--blur-dir", default=str(BLUR))
    ap.add_argument("--allow-missing-blur", action="store_true",
                    help="write the table with the three image-quality rules "
                         "set to False when no blur measurement exists")
    ap.add_argument("--out", default=str(OUT))
    a = ap.parse_args()
    log(f"host={os.uname().nodename}  {a.cancer}/{a.field}")

    w = pd.read_parquet(Path(a.white_dir) / f"{a.field}_cells.parquet")
    d = w[["cell_id", "block", "x_um", "y_um", "dx_um", "dy_um",
           "has_model", "white_frac", "drop_white"]].copy()
    d = d.rename(columns={"drop_white": "drop_blank"})
    d["drop_no_model"] = ~d.has_model

    bp = Path(a.blur_dir) / f"{a.field}_cells.parquet"
    if bp.exists():
        b = pd.read_parquet(bp)[["cell_id", "contrast", "tenengrad", "mask_delta",
                                 "rim_grad", "drop_blur", "drop_nomask", "drop_flat"]]
        d = d.merge(b, on="cell_id", how="left")
        d = d.rename(columns={"drop_blur": "drop_out_of_focus",
                              "drop_nomask": "drop_no_visible_nucleus",
                              "drop_flat": "drop_featureless"})
    elif a.allow_missing_blur:
        log("  no blur measurement -- image-quality rules SKIPPED (allowed)")
        for c in ("drop_out_of_focus", "drop_no_visible_nucleus", "drop_featureless"):
            d[c] = False
    else:
        log(f"  ERROR no blur measurement at {bp} -- three of the five rules "
            f"cannot be evaluated. Run 09_blur_cell_filter.py first, or pass "
            f"--allow-missing-blur if you really want a partial table.")
        return 2
    reasons = ["drop_no_model", "drop_blank", "drop_out_of_focus",
               "drop_no_visible_nucleus", "drop_featureless"]
    for c in reasons:
        d[c] = d[c].fillna(False).astype(bool)
    d["keep"] = ~d[reasons].any(axis=1)

    outd = Path(a.out)
    outd.mkdir(parents=True, exist_ok=True)
    d.to_parquet(outd / f"{a.field}_cells.parquet", index=False)


    t = d[d.keep].copy()
    t["he_px"] = (t.x_um + t.dx_um) / UM_PER_PX
    t["he_py"] = (t.y_um + t.dy_um) / UM_PER_PX
    t = t[["cell_id", "he_px", "he_py", "x_um", "y_um", "dx_um", "dy_um", "block"]]
    t.to_parquet(outd / f"{a.field}_cells_for_training.parquet", index=False)
    log(f"  {len(t)} usable cells -> {a.field}_cells_for_training.parquet")

    n = len(d)
    summ = {"field": a.field, "cancer": a.cancer, "n_cells": n,
            "n_keep": int(d.keep.sum()), "frac_keep": float(d.keep.mean()),
            "n_dropped": int(n - d.keep.sum()),
            "blur_measured": bool(bp.exists()),
            "by_reason": {c: int(d[c].sum()) for c in reasons},
            "by_reason_exclusive": {}}

    for c in reasons:
        others = d[[x for x in reasons if x != c]].any(axis=1)
        summ["by_reason_exclusive"][c] = int((d[c] & ~others).sum())
    per_blk = {}
    for k, s in d.groupby("block"):
        per_blk[f"blk{int(k)}"] = {"n_cells": int(len(s)), "n_keep": int(s.keep.sum()),
                                   "frac_keep": float(s.keep.mean()),
                                   "has_model": bool(s.has_model.any())}
    summ["per_block"] = per_blk
    (outd / f"{a.field}.json").write_text(json.dumps(summ, indent=2))

    log(f"  keep {summ['n_keep']}/{n} ({100*summ['frac_keep']:.1f}%)")
    log(f"  {'rule':<26}{'removes':>10}{'only this rule':>16}")
    for c in reasons:
        log(f"  {c:<26}{summ['by_reason'][c]:>10}"
            f"{summ['by_reason_exclusive'][c]:>16}")
    for k, v in per_blk.items():
        log(f"    {k}: keep {v['n_keep']}/{v['n_cells']} "
            f"({100*v['frac_keep']:.1f}%)  model={v['has_model']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
