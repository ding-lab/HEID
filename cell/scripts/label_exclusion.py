#!/usr/bin/env python3
from __future__ import annotations
import os
PROJECTS_ROOT = os.environ.get("PROJECTS_ROOT", "/data/heid")
from pathlib import Path
_RELEASE = Path(__file__).resolve().parents[2]
CELL_OUTPUTS = _RELEASE / "cell" / "outputs"
DEFINE_OUTPUTS = _RELEASE / "nerve" / "define" / "outputs"

import argparse
import hashlib
import json
from collections import Counter
from datetime import datetime, timezone

import pandas as pd

CANCERS = ("CHOL", "CRC", "PDAC", "PRAD")
GROUPS = ("region", "scattered", "nocov")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def read_define(cells_csv: Path):
    try:
        cells = pd.read_csv(cells_csv, dtype=str, usecols=["cell_id", "schwann_cluster"])
    except pd.errors.EmptyDataError:
        return None
    stats_csv = cells_csv.with_name(cells_csv.name.replace("_cells.csv", "_cluster_stats.csv"))
    try:
        stats = pd.read_csv(stats_csv, dtype=str)
    except (pd.errors.EmptyDataError, FileNotFoundError):
        stats = pd.DataFrame(columns=["cluster_id", "passed"])
    ok = set(stats.loc[stats["passed"].astype(str).str.lower() == "true", "cluster_id"].astype(str))
    cid = cells["schwann_cluster"].fillna("None").str.replace("Schwann-", "", regex=False)
    accepted = set(cells.loc[(cid != "None") & cid.isin(ok), "cell_id"].astype(str))
    return accepted, set(cells["cell_id"].astype(str)), stats_csv


def parse_expected(text: str | None) -> dict[str, int] | None:
    if not text:
        return None
    out = {}
    for item in text.split(","):
        key, value = item.split("=", 1)
        out[key.strip()] = int(value)
    return out


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--contract", type=Path, default=_RELEASE / "cell/configs/contract.json")
    parser.add_argument("--labels-root", type=Path, default=Path(PROJECTS_ROOT) / "cell/outputs/labels")
    parser.add_argument("--define-root", type=Path, default=DEFINE_OUTPUTS)
    parser.add_argument("--cancers", nargs="+", default=list(CANCERS))
    parser.add_argument("--membership-result", type=Path,
                        default=CELL_OUTPUTS / "results" / "region_membership.json")
    parser.add_argument("--out-root", type=Path, default=Path(PROJECTS_ROOT) / "cell/outputs/labels_region_schwann")
    parser.add_argument("--manifest", type=Path,
                        default=CELL_OUTPUTS / "manifests" / "identity_exclusion_manifest.json")
    parser.add_argument("--expected-totals", default=None,
                        help="optional region=N,scattered=N,nocov=N totals the run must reproduce")
    parser.add_argument("--expected-nocov-samples", nargs="*", default=None)
    parser.add_argument("--min-id-coverage", type=float, default=0.99)
    args = parser.parse_args()

    args.out_root.mkdir(parents=True, exist_ok=True)
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    define: dict[str, Path] = {}
    for cancer in args.cancers:
        for path in sorted((args.define_root / cancer / "data").glob("*_cells.csv")):
            define[path.name[: -len("_cells.csv")]] = path
    contract = json.loads(args.contract.read_text())["per_sample"]
    membership = json.loads(args.membership_result.read_text())["per_slide"] if args.membership_result.is_file() else {}
    per: dict[str, dict] = {}
    totals: Counter = Counter()
    sources: dict[str, str] = {}
    mismatches = []
    for record in contract:
        sample = record["sample"]
        source = args.labels_root / f"{sample}.parquet"
        labels = pd.read_parquet(source)
        if list(labels.columns[:3]) != ["cell_id", "cell_type_refine", "coarse11"]:
            raise SystemExit(f"{sample}: unexpected label columns {list(labels.columns)}")
        is_schwann = (labels["cell_type_refine"] == "Schwann").to_numpy()
        group = pd.Series([""] * len(labels), dtype=object)
        coverage = None
        if is_schwann.any():
            got = read_define(define[sample]) if sample in define else None
            if got is None:
                group[is_schwann] = "nocov"
            else:
                accepted, present, stats_csv = got
                barcodes = labels["cell_id"].astype(str)
                group[is_schwann] = "scattered"
                group[is_schwann & barcodes.isin(accepted).to_numpy()] = "region"
                coverage = float(barcodes[is_schwann].isin(present).mean())
                sources[str(define[sample].relative_to(args.define_root))] = sha256(define[sample])
                if stats_csv.exists():
                    sources[str(stats_csv.relative_to(args.define_root))] = sha256(stats_csv)
        out = labels[["cell_id", "cell_type_refine", "coarse11"]].copy()
        out["schwann_group"] = group.to_numpy()
        out["identity_excluded"] = is_schwann & (group.to_numpy() != "region")
        destination = args.out_root / f"{sample}.parquet"
        temporary = destination.with_name(destination.name + f".tmp.{os.getpid()}")
        out.to_parquet(temporary, index=False)
        os.replace(temporary, destination)
        back = pd.read_parquet(destination)
        if not back[["cell_id", "cell_type_refine", "coarse11"]].equals(labels[["cell_id", "cell_type_refine", "coarse11"]]):
            raise SystemExit(f"{sample}: label columns changed on write")
        if (back["identity_excluded"] & (back["cell_type_refine"] != "Schwann")).any():
            raise SystemExit(f"{sample}: a non-Schwann cell is excluded")
        counts = {k: int(v) for k, v in pd.Series(group[is_schwann]).value_counts().items()}
        if is_schwann.any() and sample in membership:
            want = {k: v for k, v in membership[sample].items() if k in GROUPS}
            if counts != want:
                mismatches.append({"sample": sample, "labels": counts, "membership": want})
        for key, value in counts.items():
            totals[key] += value
        per[sample] = {"cancer": record["cancer"], "fold": record["fold_he_safe"], "n_rows": int(len(labels)),
                       "n_schwann": int(is_schwann.sum()), "n_excluded": int(back["identity_excluded"].sum()),
                       "groups": counts, "define_id_coverage": coverage, "source_label_sha256": sha256(source),
                       "sha256": sha256(destination)}
    expected = parse_expected(args.expected_totals)
    nocov = {s for s, r in per.items() if r["groups"].get("nocov")}
    coverages = [r["define_id_coverage"] for r in per.values() if r["define_id_coverage"] is not None]
    checks = {
        "totals_match_expected": (dict(totals) == expected) if expected is not None else None,
        "nocov_slides_match": (nocov == set(args.expected_nocov_samples)) if args.expected_nocov_samples is not None else None,
        "per_slide_counts_match_membership": not mismatches,
        "min_define_id_coverage": min(coverages) if coverages else None,
    }
    manifest = {
        "schema_version": "v12.labels.v1",
        "built_at_utc": datetime.now(timezone.utc).isoformat(),
        "rule": "identity_excluded = label Schwann and not in a Define cluster with passed == True",
        "fold_key": "fold_he_safe",
        "define_root": str(args.define_root),
        "define_sources_sha256": sources,
        "totals": dict(totals),
        "checks": checks,
        "membership_mismatches": mismatches,
        "per_sample": per,
        "script_sha256": sha256(Path(__file__)),
    }
    args.manifest.write_text(json.dumps(manifest, indent=1))
    print(json.dumps({"totals": dict(totals), "checks": checks}), flush=True)
    failed = [k for k, v in checks.items() if k != "min_define_id_coverage" and v is False]
    if failed or (checks["min_define_id_coverage"] or 0.0) < args.min_id_coverage:
        raise SystemExit(f"label checks failed: {failed or 'define id coverage'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
