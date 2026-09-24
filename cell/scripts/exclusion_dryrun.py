#!/usr/bin/env python3
from __future__ import annotations
import os
PROJECTS_ROOT = os.environ.get("PROJECTS_ROOT", "/data/heid")
from pathlib import Path
_RELEASE = Path(__file__).resolve().parents[2]
CELL_OUTPUTS = _RELEASE / "cell" / "outputs"
DEFINE_OUTPUTS = _RELEASE / "nerve" / "define" / "outputs"

import argparse
import json

import pandas as pd
import torch


def flagged_by_sample(labels_root: Path) -> dict[str, set[str]]:
    out: dict[str, set[str]] = {}
    for path in sorted(labels_root.glob("*.parquet")):
        table = pd.read_parquet(path, columns=["cell_id", "identity_excluded"])
        excluded = table.loc[table["identity_excluded"].to_numpy(dtype=bool), "cell_id"].astype(str)
        if len(excluded):
            out[path.stem] = set(excluded)
    return out


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--labels-root", type=Path, default=Path(PROJECTS_ROOT) / "cell/outputs/labels_region_schwann")
    parser.add_argument("--shared-scratch", type=Path, default=CELL_OUTPUTS / "scratch" / "shared")
    parser.add_argument("--reference-arm", default="cls_sigma3_all_schwann")
    parser.add_argument("--results-root", type=Path, default=CELL_OUTPUTS / "results")
    parser.add_argument("--folds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    parser.add_argument("--out", type=Path, default=CELL_OUTPUTS / "results" / "exclusion_dryrun.json")
    args = parser.parse_args()

    flagged = flagged_by_sample(args.labels_root)
    report = {"reference_arm": args.reference_arm, "n_slides_with_flagged_cells": len(flagged),
              "n_flagged_cells_total": int(sum(len(v) for v in flagged.values())), "per_fold": {}}
    for fold in args.folds:
        identity = torch.load(args.shared_scratch / f"fold{fold}" / "row_identity.pt", map_location="cpu", weights_only=False)
        pool = identity["train_pool_cells"]
        in_pool = 0
        labelled = 0
        per_sample: dict[str, int] = {}
        for sample, cells in pool.items():
            cells_flagged = flagged.get(sample)
            if not cells_flagged:
                continue
            labelled += len(cells_flagged)
            n = int(sum(1 for cell in cells.tolist() if cell in cells_flagged))
            if n:
                per_sample[sample] = n
            in_pool += n
        summary = json.loads((args.results_root / args.reference_arm / f"fold{fold}_summary.json").read_text())
        groups = summary["selection"]["groups"]["Schwann"]
        available = int(groups["available_in_pool"])
        selected = int(groups["selected"])
        report["per_fold"][str(fold)] = {
            "n_train_slides": len(pool),
            "n_flagged_on_train_slides_in_labels": labelled,
            "n_flagged_in_train_pool": in_pool,
            "reference_schwann_available_in_pool": available,
            "reference_schwann_selected": selected,
            "reference_schwann_uncapped": available == selected and not groups["boundary_limited"]
            and not groups["capped_source_labels"],
            "region_schwann_left_for_training": selected - in_pool,
            "n_samples_masked": len(per_sample),
        }
        del identity, pool
    bad = [k for k, v in report["per_fold"].items() if not v["reference_schwann_uncapped"]]
    report["all_folds_schwann_uncapped"] = not bad
    report["min_region_schwann_left"] = min(v["region_schwann_left_for_training"] for v in report["per_fold"].values())
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=1))
    print(json.dumps(report, indent=1))
    if bad or report["min_region_schwann_left"] <= 0:
        raise SystemExit("a fold would train with no region Schwann, or Schwann was capped")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
