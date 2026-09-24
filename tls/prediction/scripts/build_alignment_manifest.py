#!/usr/bin/env python3

from __future__ import annotations
import os

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq

from cohort_binding import load_contract


MODULE_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = MODULE_ROOT / "configs" / "data_contract.json"
DEFAULT_OUTPUT_DIR = MODULE_ROOT / "outputs" / "results"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def strip_suffix(name: str, suffixes: tuple[str, ...]) -> tuple[str, str]:
    for suffix in suffixes:
        if name.endswith(suffix):
            return name[: -len(suffix)], suffix
    if "-bad" in name:
        return name.replace("-bad", "", 1), "-bad"
    return name, ""


def first_existing(paths: list[Path]) -> Path | None:
    return next((path for path in paths if path.exists()), None)


def as_int(value: object) -> int | None:
    if pd.isna(value):
        return None
    return int(value)


def as_float(value: object) -> float | None:
    if pd.isna(value):
        return None
    return float(value)


def verify_sources(root: Path, expected: dict[str, str]) -> dict[str, str]:
    observed = {}
    for relative, expected_digest in expected.items():
        path = root / relative
        if not path.is_file():
            raise FileNotFoundError(f"required source is missing: {path}")
        observed_digest = sha256(path)
        observed[relative] = observed_digest
        if observed_digest != expected_digest:
            raise RuntimeError(
                f"source drift for {path}: expected {expected_digest}, "
                f"observed {observed_digest}"
            )
    return observed


def build_manifest(config: dict) -> pd.DataFrame:
    root = Path(config["source_root"])
    suffixes = tuple(config["unusable_region_suffixes"])
    report = pd.read_csv(root / "_report" / "cohort_summary.csv")
    report.columns = report.columns.str.strip()
    for column in ("cohort", "cancer", "field"):
        report[column] = report[column].astype(str).str.strip()
    report_index = report.set_index(["cohort", "cancer", "field"], verify_integrity=True)

    records = []
    for cohort in config["published_cohorts"]:
        cohort_dir = root / cohort
        for cancer_dir in sorted(path for path in cohort_dir.iterdir() if path.is_dir()):
            for field_dir in sorted(path for path in cancer_dir.iterdir() if path.is_dir()):
                field, suffix = strip_suffix(field_dir.name, suffixes)
                micro = field_dir / "micro"
                training = first_existing(
                    [
                        micro / f"{field_dir.name}_cells_for_training.parquet",
                        micro / f"{field}_cells_for_training.parquet",
                    ]
                )
                cell_keep = first_existing(
                    [
                        micro / "cell_keep" / f"{field_dir.name}_cells.parquet",
                        micro / "cell_keep" / f"{field}_cells.parquet",
                    ]
                )
                n_training = (
                    pq.ParquetFile(training).metadata.num_rows if training is not None else None
                )

                key = (cohort, cancer_dir.name, field)
                if key not in report_index.index:
                    raise KeyError(f"published directory absent from cohort_summary.csv: {key}")
                source = report_index.loc[key]
                keep_fraction = as_float(source.get("keep_frac"))

                has_rows = n_training is not None and n_training > 0
                region_usable = suffix == "" and has_rows
                cell_usable = region_usable or (
                    suffix == "-blur"
                    and has_rows
                    and keep_fraction is not None
                    and keep_fraction >= config["cell_task_blur_min_keep_fraction"]
                )
                if suffix:
                    exclusion_reason = f"directory_suffix{suffix}"
                elif n_training is None:
                    exclusion_reason = "missing_training_parquet"
                elif n_training == 0:
                    exclusion_reason = "zero_training_rows"
                else:
                    exclusion_reason = ""

                records.append(
                    {
                        "cohort": cohort,
                        "cancer": cancer_dir.name,
                        "field": field,
                        "directory_name": field_dir.name,
                        "directory_suffix": suffix,
                        "n_cells": as_int(source.get("n_cells")),
                        "n_training_cells": n_training,
                        "keep_fraction": keep_fraction,
                        "training_cells_path": str(training) if training is not None else "",
                        "cell_keep_path": str(cell_keep) if cell_keep is not None else "",
                        "he_image_path": str(field_dir / "he_aligned.ome.tif"),
                        "cell_task_usable": bool(cell_usable),
                        "region_task_usable": bool(region_usable),
                        "model_scope_usable": bool(
                            region_usable and cohort in config["model_cohorts"]
                        ),
                        "region_exclusion_reason": exclusion_reason,
                    }
                )

    manifest = pd.DataFrame.from_records(records)
    return manifest.sort_values(["cohort", "cancer", "field"]).reset_index(drop=True)


def build_bridge_overlap(config: dict, manifest: pd.DataFrame) -> pd.DataFrame:
    blur = manifest.loc[manifest.directory_suffix.eq("-blur"), [
        "cohort",
        "cancer",
        "field",
        "directory_name",
        "keep_fraction",
        "region_exclusion_reason",
    ]]
    records = []
    for relative in config["v4_manifests"]:
        path = (MODULE_ROOT / relative).resolve()
        source = pd.read_csv(path, sep="\t")
        overlap = source.merge(blur, on=["cancer", "field"], how="inner", suffixes=("_v4", "_v9"))
        overlap.insert(0, "v4_manifest", path.name)
        overlap["in_reported_cancer_scope"] = overlap["cancer"].isin(
            config["reported_cancers"]
        )
        records.append(overlap)
    if not records:
        return pd.DataFrame()
    return pd.concat(records, ignore_index=True).sort_values(
        ["v4_manifest", "cancer", "field"]
    )


def summarize(config: dict, manifest: pd.DataFrame, overlap: pd.DataFrame) -> dict:
    region = manifest.loc[manifest.region_task_usable]
    by_cohort = {}
    for cohort, data in region.groupby("cohort", sort=True):
        by_cohort[cohort] = {
            "slides": int(len(data)),
            "cells": int(data.n_training_cells.sum()),
        }

    suffix_counts = {
        (suffix or "none"): int(count)
        for suffix, count in manifest.directory_suffix.value_counts(dropna=False).items()
    }
    overlap_summary = {}
    if not overlap.empty:
        for name, data in overlap.groupby("v4_manifest", sort=True):
            overlap_summary[name] = {
                "units": int(len(data)),
                "reported_cancer_units": int(data.in_reported_cancer_scope.sum()),
                "fields": data.field.tolist(),
            }

    return {
        "contract_name": config["contract_name"],
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_root": config["source_root"],
        "source_sha256": config["source_files"],
        "policy": {
            "model_cohorts": config["model_cohorts"],
            "unusable_region_suffixes": config["unusable_region_suffixes"],
            "cell_task_blur_min_keep_fraction": config[
                "cell_task_blur_min_keep_fraction"
            ],
            "tls_region_min_join_fraction": config["tls_region_min_join_fraction"],
            "tls_region_min_keep_fraction": config["tls_region_min_keep_fraction"],
            "tile_min_keep_fraction": config["tile_min_keep_fraction"],
        },
        "observed": {
            "published_directories": int(len(manifest)),
            "suffix_counts": suffix_counts,
            "zero_row_unsuffixed_directories": int(
                (
                    manifest.directory_suffix.eq("")
                    & manifest.n_training_cells.fillna(-1).eq(0)
                ).sum()
            ),
            "cell_task_usable_slides": int(manifest.cell_task_usable.sum()),
            "region_usable_slides": int(len(region)),
            "region_usable_cells": int(region.n_training_cells.sum()),
            "region_usable_by_cohort": by_cohort,
            "model_scope_usable_slides": int(manifest.model_scope_usable.sum()),
            "v4_blur_overlap": overlap_summary,
        },
    }


def strict_checks(config: dict, summary: dict) -> None:
    expected = config["expected"]
    observed = summary["observed"]
    checks = {
        "published_directories": observed["published_directories"],
        "weak_directories": observed["suffix_counts"].get("-weak", 0),
        "blur_directories": observed["suffix_counts"].get("-blur", 0),
        "zero_row_unsuffixed_directories": observed[
            "zero_row_unsuffixed_directories"
        ],
        "region_usable_slides": observed["region_usable_slides"],
        "region_usable_cells": observed["region_usable_cells"],
        "region_usable_by_cohort": observed["region_usable_by_cohort"],
    }
    mismatches = {
        key: {"expected": expected[key], "observed": value}
        for key, value in checks.items()
        if value != expected[key]
    }
    if mismatches:
        raise RuntimeError(
            "strict aligned-H&E contract checks failed:\n"
            + json.dumps(mismatches, indent=2, sort_keys=True)
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--strict",
        action="store_true",
        help="fail if the current source no longer matches the pinned aligned-cell contract",
    )
    args = parser.parse_args()

    config = load_contract(args.config)
    source_root = Path(config["source_root"])
    observed_sha = verify_sources(source_root, config["source_files"])
    manifest = build_manifest(config)
    overlap = build_bridge_overlap(config, manifest)
    summary = summarize(config, manifest, overlap)
    summary["source_sha256"] = observed_sha
    if args.strict:
        strict_checks(config, summary)
    summary["strict_checks_passed"] = bool(args.strict)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    manifest.to_csv(args.output_dir / "alignment_manifest.tsv", sep="\t", index=False)
    overlap.to_csv(args.output_dir / "bridge_overlap.tsv", sep="\t", index=False)
    (args.output_dir / "alignment_audit.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n"
    )

    print(
        json.dumps(
            {
                "published_directories": summary["observed"]["published_directories"],
                "region_usable_slides": summary["observed"]["region_usable_slides"],
                "region_usable_cells": summary["observed"]["region_usable_cells"],
                "model_scope_usable_slides": summary["observed"][
                    "model_scope_usable_slides"
                ],
                "v4_blur_overlap": {
                    key: value["units"]
                    for key, value in summary["observed"]["v4_blur_overlap"].items()
                },
                "strict_checks_passed": summary["strict_checks_passed"],
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
