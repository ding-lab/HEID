#!/usr/bin/env python3

from __future__ import annotations
import os

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

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


def resolve_bridge(config: dict) -> dict[str, Path]:
    paths = {}
    for name, source in config["gen12_bridge"].items():
        path = (MODULE_ROOT / source["path"]).resolve()
        if not path.is_file():
            raise FileNotFoundError(f"molecular bridge source is missing: {path}")
        observed = sha256(path)
        if observed != source["sha256"]:
            raise RuntimeError(
                f"molecular bridge source drift for {path}: expected {source['sha256']}, "
                f"observed {observed}"
            )
        paths[name] = path
    return paths


def load_inputs(paths: dict[str, Path], output_dir: Path) -> tuple[pd.DataFrame, ...]:
    manifest = pd.read_csv(output_dir / "alignment_manifest.tsv", sep="\t")
    per_tls = pd.read_parquet(paths["per_tls"])
    mapping = pd.read_csv(paths["sample_to_v9"], sep="\t")
    per_cell = pd.read_parquet(
        paths["per_cell"],
        columns=["sample", "cell_id_base", "tls_region_halo150"],
    )
    per_cell = per_cell.loc[per_cell.tls_region_halo150.ne("Outside")].copy()
    return manifest, per_tls, mapping, per_cell


def audit_regions(
    config: dict,
    manifest: pd.DataFrame,
    per_tls: pd.DataFrame,
    mapping: pd.DataFrame,
    per_cell: pd.DataFrame,
) -> pd.DataFrame:
    if manifest.duplicated(["field"]).any():
        duplicated = manifest.loc[
            manifest.duplicated(["field"], keep=False), "field"
        ].tolist()
        raise RuntimeError(
            f"field is not unique in aligned_HE_v9 manifest: {duplicated[:10]}"
        )
    if mapping.duplicated(["gen12_sample"]).any():
        duplicated = mapping.loc[
            mapping.duplicated(["gen12_sample"], keep=False), "gen12_sample"
        ].tolist()
        raise RuntimeError(f"molecular sample mapping is not unique: {duplicated[:10]}")

    manifest_by_field = manifest.set_index("field")
    mapping_by_sample = mapping.set_index("gen12_sample")
    rows = []

    for sample, sample_cells in per_cell.groupby("sample", sort=True):
        if sample not in mapping_by_sample.index:
            raise RuntimeError(f"molecular sample is absent from sample-to-field mapping: {sample}")
        mapped = mapping_by_sample.loc[sample]
        field = mapped.get("v9_field_base")
        has_field = isinstance(field, str) and field in manifest_by_field.index

        if has_field:
            slide = manifest_by_field.loc[field]
            cell_keep_value = slide.get("cell_keep_path")
            cell_keep_path = (
                Path(cell_keep_value) if isinstance(cell_keep_value, str) else None
            )
            slide_usable = bool(slide.region_task_usable)
            reason_value = slide.get("region_exclusion_reason")
            slide_reason = str(reason_value) if isinstance(reason_value, str) else ""
        else:
            cell_keep_path = None
            slide_usable = False
            slide_reason = "no_published_v9"

        if cell_keep_path is not None and cell_keep_path.is_file():
            cell_keep = pd.read_parquet(cell_keep_path, columns=["cell_id", "keep"])
            if cell_keep.cell_id.duplicated().any():
                raise RuntimeError(f"duplicate cell_id in aligned cell_keep table: {field}")
            joined = sample_cells.merge(
                cell_keep,
                left_on="cell_id_base",
                right_on="cell_id",
                how="left",
                validate="many_to_one",
            )
        else:
            joined = sample_cells.copy()
            joined["keep"] = pd.NA

        for tls_region, region in joined.groupby("tls_region_halo150", sort=True):
            n_members = int(len(region))
            n_joined = int(region["keep"].notna().sum())
            n_keep = int(region["keep"].eq(True).sum())
            join_fraction = n_joined / n_members if n_members else float("nan")
            keep_fraction = n_keep / n_joined if n_joined else float("nan")

            if not has_field:
                status = "no_published_v9"
            elif not slide_usable:
                status = "slide_qc_excluded"
            elif join_fraction < config["tls_region_min_join_fraction"]:
                status = "join_coverage_below_threshold"
            elif keep_fraction < config["tls_region_min_keep_fraction"]:
                status = "region_keep_below_threshold"
            else:
                status = "usable"

            rows.append(
                {
                    "sample": sample,
                    "cancer": mapped.get("gen12_cancer"),
                    "tls_region": tls_region,
                    "v9_field": field if isinstance(field, str) else "",
                    "slide_region_usable": slide_usable,
                    "slide_exclusion_reason": slide_reason,
                    "n_region_cells": n_members,
                    "n_joined_cells": n_joined,
                    "n_keep_cells": n_keep,
                    "join_fraction": join_fraction,
                    "region_keep_fraction": keep_fraction,
                    "region_qc_status": status,
                    "region_qc_usable": status == "usable",
                }
            )

    result = pd.DataFrame.from_records(rows)
    reference = per_tls[
        ["sample", "cancer", "tls_region", "n_cells_halo150", "mature", "has_maturity"]
    ].copy()
    result = reference.merge(
        result,
        on=["sample", "cancer", "tls_region"],
        how="left",
        validate="one_to_one",
    )
    if result.region_qc_status.isna().any():
        missing = result.loc[
            result.region_qc_status.isna(), ["sample", "tls_region"]
        ]
        raise RuntimeError(f"TLS regions missing from per-cell bridge:\n{missing.head(10)}")
    if not result.n_region_cells.eq(result.n_cells_halo150).all():
        bad = result.loc[
            ~result.n_region_cells.eq(result.n_cells_halo150),
            ["sample", "tls_region", "n_region_cells", "n_cells_halo150"],
        ]
        raise RuntimeError(f"molecular region cell-count mismatch:\n{bad.head(10)}")
    return result.sort_values(["cancer", "sample", "tls_region"]).reset_index(drop=True)


def summarize(config: dict, paths: dict[str, Path], regions: pd.DataFrame) -> dict:
    slide_usable = regions.slide_region_usable
    join_pass = (
        slide_usable
        & regions.join_fraction.ge(config["tls_region_min_join_fraction"])
    )
    accepted = regions.region_qc_usable
    return {
        "contract_name": config["contract_name"],
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_sha256": {name: sha256(path) for name, path in paths.items()},
        "policy": {
            "tls_region_min_join_fraction": config["tls_region_min_join_fraction"],
            "tls_region_min_keep_fraction": config["tls_region_min_keep_fraction"],
        },
        "observed": {
            "total_regions": int(len(regions)),
            "regions_on_usable_slides": int(slide_usable.sum()),
            "regions_passing_join_gate": int(join_pass.sum()),
            "accepted_regions": int(accepted.sum()),
            "status_counts": {
                str(status): int(count)
                for status, count in regions.region_qc_status.value_counts().items()
            },
            "accepted_keep_fraction": {
                "minimum": float(
                    regions.loc[accepted, "region_keep_fraction"].min()
                ),
                "median": float(
                    regions.loc[accepted, "region_keep_fraction"].median()
                ),
            },
            "accepted_join_fraction": {
                "minimum": float(regions.loc[accepted, "join_fraction"].min()),
                "median": float(regions.loc[accepted, "join_fraction"].median()),
            },
        },
    }


def strict_checks(config: dict, summary: dict) -> None:
    expected = config["expected"]["tls_region_qc"]
    observed = summary["observed"]
    mismatches = {
        key: {"expected": value, "observed": observed[key]}
        for key, value in expected.items()
        if observed[key] != value
    }
    if mismatches:
        raise RuntimeError(
            "strict molecular TLS region-QC checks failed:\n"
            + json.dumps(mismatches, indent=2, sort_keys=True)
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--strict",
        action="store_true",
        help="fail if the molecular bridge or expected region-QC counts drift",
    )
    args = parser.parse_args()

    config = load_contract(args.config)
    paths = resolve_bridge(config)
    manifest, per_tls, mapping, per_cell = load_inputs(paths, args.output_dir)
    regions = audit_regions(config, manifest, per_tls, mapping, per_cell)
    summary = summarize(config, paths, regions)
    if args.strict:
        strict_checks(config, summary)
    summary["strict_checks_passed"] = bool(args.strict)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    regions.to_csv(
        args.output_dir / "region_qc.tsv", sep="\t", index=False
    )
    (args.output_dir / "region_qc_audit.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps(summary["observed"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
