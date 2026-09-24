#!/usr/bin/env python3

from __future__ import annotations
import os
PROJECTS_ROOT = os.environ.get("PROJECTS_ROOT", "/data/heid")

import argparse
import hashlib
import json
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

DATA = Path(PROJECTS_ROOT + "/cell/inputs")
ALIGN_ROOT = Path(PROJECTS_ROOT) / "align/aligned_he"
REPORT = ALIGN_ROOT / "_report"
COHORT_SUMMARY = REPORT / "cohort_summary.csv"
BLUR_MAP = REPORT / "blur_rename_map.tsv"
LOWQC = REPORT / "lowqc_samples.csv"
WEAK_GATE = REPORT / "weak_corr_gate.csv"

PC = Path(PROJECTS_ROOT + "/cell")
CELL_ROOT = PC
OUT = CELL_ROOT / "outputs/manifests"

BLUR_FRAC_KEEP_MIN = 0.50
WEAK_FRAC_MAX = 0.5


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 22), b""):
            digest.update(block)
    return digest.hexdigest()


def sha256_list(items) -> str:
    return hashlib.sha256("\0".join(items).encode()).hexdigest()


def glob_index() -> dict:
    index = {}
    for cancer_dir in sorted((ALIGN_ROOT / "5k").iterdir()):
        if not cancer_dir.is_dir():
            continue
        for slide_dir in sorted(cancer_dir.iterdir()):
            if not slide_dir.is_dir():
                continue
            dirname = slide_dir.name
            suffix = None
            field = dirname
            for suf in ("-blur", "-weak", "-lowqc", "-bad"):
                if dirname.endswith(suf):
                    suffix = suf
                    field = dirname[: -len(suf)]
                    break
            train = slide_dir / "micro" / f"{field}_cells_for_training.parquet"
            if field in index:
                raise RuntimeError(f"duplicate field name in the 5k tree: {field}")
            index[field] = {
                "cohort": "5k",
                "cancer": cancer_dir.name,
                "dirname": dirname,
                "dir_suffix": suffix,
                "dir": str(slide_dir),
                "train_parquet": str(train),
                "train_parquet_exists": train.is_file(),
            }
    return index


def patient_of(sample: str, cancer: str) -> str:

    if sample.startswith(("C3L-", "C3N-", "C3M-")):
        return "-".join(sample.split("-")[:2])
    if sample.startswith(("S15-", "S16-", "S17-", "S18-", "S19-", "HS-", "HS1")):

        parts = sample.split("-")
        if sample.startswith("HS-"):
            return "-".join(parts[:3])
        return "-".join(parts[:2])
    head = sample.split("-")[0]


    import re

    m = re.match(r"^([A-Za-z]+[0-9]+)[A-Za-z][0-9]*$", head)
    if m:
        return m.group(1)
    return head


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", default="index")
    args = ap.parse_args()

    summary = pd.read_csv(COHORT_SUMMARY)
    blur = pd.read_csv(BLUR_MAP, sep="\t")
    lowqc = pd.read_csv(LOWQC)
    weak = pd.read_csv(WEAK_GATE)

    index = glob_index()

    blur_frac = dict(zip(blur["field"].astype(str), blur["frac_keep"].astype(float), strict=True))
    lowqc_fields = set(lowqc.loc[lowqc["cohort"].astype(str) == "5k", "field"].astype(str))

    five_k = summary[summary["cohort"].astype(str) == "5k"].copy()
    five_k["field"] = five_k["field"].astype(str)
    if five_k["field"].duplicated().any():
        raise RuntimeError("duplicate field rows in cohort_summary.csv for 5k")

    kept, excluded = [], []
    for row in five_k.itertuples(index=False):
        field = row.field
        causes = []
        lowqc_flag = bool(row.lowqc)
        if lowqc_flag or field in lowqc_fields:
            causes.append("lowqc")
        weak_frac = float(row.weak_frac) if pd.notna(row.weak_frac) else None
        if weak_frac is None or weak_frac > WEAK_FRAC_MAX:
            causes.append("weak")
        n_keep = float(row.n_keep) if pd.notna(row.n_keep) else 0.0
        if not n_keep > 0:
            causes.append("zero_keep")
        fk = blur_frac.get(field)
        if fk is not None and fk < BLUR_FRAC_KEEP_MIN:
            causes.append("blur_lt_0.50")

        info = index.get(field)
        record = {
            "sample": field,
            "cancer": str(row.cancer),
            "lowqc": lowqc_flag,
            "weak_frac": weak_frac,
            "n_cells_slide": None if pd.isna(row.n_cells) else int(row.n_cells),
            "n_keep_slide": int(n_keep),
            "keep_frac_slide": None if pd.isna(row.keep_frac) else float(row.keep_frac),
            "blur": field in blur_frac,
            "blur_frac_keep": fk,
            "stage_reached": str(row.stage_reached),
            "dirname": None if info is None else info["dirname"],
            "dir_suffix": None if info is None else info["dir_suffix"],
            "train_parquet": None if info is None else info["train_parquet"],
            "train_parquet_exists": None if info is None else info["train_parquet_exists"],
        }
        if info is None:
            causes.append("no_directory_in_aligned_tree")
        elif not info["train_parquet_exists"]:
            causes.append("no_training_parquet")

        if causes:
            record["causes"] = causes
            excluded.append(record)
        else:
            kept.append(record)

    kept.sort(key=lambda r: r["sample"])
    allow = [r["sample"] for r in kept]
    by_cancer = defaultdict(int)
    for r in kept:
        by_cancer[r["cancer"]] += 1
    grouped = defaultdict(list)
    for r in excluded:
        grouped[",".join(r["causes"])].append(r["sample"])

    payload = {
        "schema_version": "v8.cohort_candidates.v1",
        "built_at_utc": datetime.now(timezone.utc).isoformat(),
        "rule": (
            "cohort=='5k' and not lowqc and weak_frac<=0.5 and n_keep>0 and "
            "(field not in blur_rename_map or frac_keep>=0.50)"
        ),
        "rule_source": str(PC / "inputs/cohort_reference/documents/COHORT.md"),
        "sources": {
            "cohort_summary": {"path": str(COHORT_SUMMARY), "sha256": sha256_file(COHORT_SUMMARY)},
            "blur_rename_map": {"path": str(BLUR_MAP), "sha256": sha256_file(BLUR_MAP)},
            "lowqc_samples": {"path": str(LOWQC), "sha256": sha256_file(LOWQC)},
            "weak_corr_gate": {"path": str(WEAK_GATE), "sha256": sha256_file(WEAK_GATE)},
        },
        "n_rows_cohort_summary_5k": int(len(five_k)),
        "n_dirs_globbed_5k": len(index),
        "n_candidates": len(kept),
        "n_keep_cells_slide_level_sum": int(sum(r["n_keep_slide"] for r in kept)),
        "per_cancer": dict(sorted(by_cancer.items())),
        "candidate_allowlist": allow,
        "candidate_allowlist_sha256": sha256_list(allow),
        "candidates": kept,
        "excluded_grouped_by_cause": {k: sorted(v) for k, v in sorted(grouped.items())},
        "excluded": sorted(excluded, key=lambda r: r["sample"]),
    }
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / "cohort_candidates.json"
    path.write_text(json.dumps(payload, indent=2) + "\n")

    print(f"5k rows in cohort_summary.csv : {len(five_k)}")
    print(f"5k directories globbed        : {len(index)}")
    print(f"candidates                    : {len(kept)}")
    print(f"slide-level n_keep sum        : {payload['n_keep_cells_slide_level_sum']:,}")
    print("per cancer                    :", dict(sorted(by_cancer.items())))
    print(f"allowlist sha256              : {payload['candidate_allowlist_sha256']}")
    for cause, names in sorted(grouped.items()):
        print(f"excluded [{cause}] : {len(names)}")
    print(f"wrote {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
