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
from collections import Counter

import pandas as pd

CANCERS = ("CHOL", "CRC", "PDAC", "PRAD")


def accepted_barcodes(cells_csv: Path) -> set[str] | None:
    try:
        cells = pd.read_csv(cells_csv, dtype=str, usecols=["cell_id", "schwann_cluster"])
    except pd.errors.EmptyDataError:
        return None
    stats_csv = cells_csv.with_name(cells_csv.name.replace("_cells.csv", "_cluster_stats.csv"))
    try:
        stats = pd.read_csv(stats_csv, dtype=str)
    except (pd.errors.EmptyDataError, FileNotFoundError):
        return set()
    ok = set(stats.loc[stats["passed"].astype(str).str.lower() == "true", "cluster_id"].astype(str))
    cid = cells["schwann_cluster"].fillna("None").str.replace("Schwann-", "", regex=False)
    return set(cells.loc[(cid != "None") & cid.isin(ok), "cell_id"].astype(str))


def define_outputs(define_root: Path, cancers: tuple[str, ...]) -> dict[str, Path]:
    found: dict[str, Path] = {}
    for cancer in cancers:
        for path in sorted((define_root / cancer / "data").glob("*_cells.csv")):
            found[path.name[: -len("_cells.csv")]] = path
    return found


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--contract", type=Path, default=_RELEASE / "cell/configs/contract.json")
    parser.add_argument("--labels-root", type=Path, default=Path(PROJECTS_ROOT) / "cell/outputs/labels")
    parser.add_argument("--define-root", type=Path, default=DEFINE_OUTPUTS)
    parser.add_argument("--cancers", nargs="+", default=list(CANCERS))
    parser.add_argument("--out-root", type=Path, default=CELL_OUTPUTS / "labels" / "schwann_membership")
    parser.add_argument("--result", type=Path, default=CELL_OUTPUTS / "results" / "region_membership.json")
    args = parser.parse_args()

    args.out_root.mkdir(parents=True, exist_ok=True)
    define = define_outputs(args.define_root, tuple(args.cancers))
    contract = json.loads(args.contract.read_text())["per_sample"]
    per: dict[str, dict] = {}
    totals: Counter = Counter()
    for record in contract:
        sample, cancer = record["sample"], record["cancer"]
        table = pd.read_parquet(args.labels_root / f"{sample}.parquet", columns=["cell_id", "cell_type_refine"])
        schwann = table.loc[table["cell_type_refine"] == "Schwann", "cell_id"].astype(str)
        if schwann.empty:
            continue
        accepted = accepted_barcodes(define[sample]) if sample in define else None
        if accepted is None:
            group = pd.Series("nocov", index=schwann.index)
        else:
            group = pd.Series("scattered", index=schwann.index).where(~schwann.isin(accepted), "region")
        pd.DataFrame({"cell_id": schwann.to_numpy(), "group": group.to_numpy()}).to_parquet(
            args.out_root / f"{sample}.parquet", index=False)
        counts = {k: int(v) for k, v in group.value_counts().items()}
        per[sample] = {"cancer": cancer, "fold": record["fold_he_safe"], "define_output": sample in define, **counts}
        for key, value in counts.items():
            totals[f"{cancer}:{key}"] += value
            totals[f"ALL:{key}"] += value
    payload = {
        "define_root": str(args.define_root),
        "rule": "region iff the label is Schwann and the cell is in a Define cluster with passed == True",
        "n_slides_with_schwann": len(per),
        "totals": dict(sorted(totals.items())),
        "define_sections_not_in_contract": sorted(set(define) - {r["sample"] for r in contract}),
        "per_slide": per,
    }
    args.result.parent.mkdir(parents=True, exist_ok=True)
    args.result.write_text(json.dumps(payload, indent=1))
    print(json.dumps(payload["totals"]), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
