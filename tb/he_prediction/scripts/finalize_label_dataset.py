#!/usr/bin/env python3

from __future__ import annotations

import argparse
import hashlib
import json
import zipfile
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from experiment_config import load_experiment


REQUIRED_ARCHIVE_MEMBERS = {
    "tumor_mask",
    "normal_mask",
    "tissue_mask",
    "label_valid",
    "sdf_um",
    "interface_sdf_um",
    "interface_valid",
    "tumor_region_id",
    "nearest_tumor_region_id",
    "metadata_json",
}


def sha256_file(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def payload_sha256(payload: Any) -> str:
    encoded = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _optional_int(payload: dict, key: str) -> int | None:

    value = payload.get(key)
    return None if value is None else int(value)


def _optional_float(payload: dict, key: str) -> float | None:
    value = payload.get(key)
    return None if value is None else float(value)


def inspect_archive(source_row: pd.Series) -> dict[str, Any]:
    path = Path(source_row["label_path"])
    if not path.exists():
        raise FileNotFoundError(path)
    with zipfile.ZipFile(path) as archive:
        bad_member = archive.testzip()
        if bad_member is not None:
            raise ValueError(f"{path}: CRC failure in {bad_member}")

    with np.load(path, allow_pickle=False) as archive:
        missing = REQUIRED_ARCHIVE_MEMBERS.difference(archive.files)
        extra = set(archive.files).difference(REQUIRED_ARCHIVE_MEMBERS)
        if missing or extra:
            raise ValueError(
                f"{path}: archive schema missing={sorted(missing)} extra={sorted(extra)}"
            )
        metadata = json.loads(str(archive["metadata_json"]))

    sample = str(source_row["sample"])
    if metadata["sample"] != sample:
        raise ValueError(f"{path}: sample identity mismatch")
    if metadata["cancer"] != str(source_row["cancer"]):
        raise ValueError(f"{path}: cancer identity mismatch")
    if metadata["patient"] != str(source_row["patient"]):
        raise ValueError(f"{path}: patient identity mismatch")
    if int(metadata["fold"]) != int(source_row["fold"]):
        raise ValueError(f"{path}: fold identity mismatch")
    if metadata["sources"]["cell_coordinates"]["sha256"] != str(
        source_row["cell_table_sha256"]
    ):
        raise ValueError(f"{path}: cell-table source identity mismatch")
    if metadata["he"]["sha256"] != str(source_row["he_sha256"]):
        raise ValueError(f"{path}: aligned-H&E source identity mismatch")


    joined = int(metadata["n_joined_cells"])
    declared = int(source_row["n_cells"])
    if joined > declared:
        raise ValueError(f"{path}: joined {joined} exceeds the section's {declared} cells")


    audit = metadata.get("label_transfer_audit", {})
    labellable = int(audit.get("gen11_cells", joined))
    if labellable and joined < 0.5 * labellable:
        raise ValueError(
            f"{path}: only {joined}/{labellable} annotated cells kept a corrected coordinate; "
            "the label would cover a minority of the molecular annotation"
        )


    verification = metadata["reference_verification"]


    max_error = verification.get("max_abs_error_um")
    if verification["status"] not in (
        "CELL_SIDE_EXACT_SDF_WITHIN_ONE_LABEL_PIXEL",
        "NOT_REQUESTED",
    ):
        raise ValueError(f"{path}: reference replay reported {verification['status']}")


    if verification["status"] != "NOT_REQUESTED":
        if _optional_int(verification, "side_mismatch") != 0:
            raise ValueError(f"{path}: cell-side mismatch")
        if _optional_float(verification, "sdf_mismatch_fraction") > 1e-4:
            raise ValueError(f"{path}: excess SDF replay mismatch")
        if _optional_int(verification, "sdf_over_one_label_pixel_count") != 0:
            raise ValueError(f"{path}: SDF replay error exceeds one label pixel")
        one_pixel_tolerance = _optional_float(verification, "one_label_pixel_tolerance_um")
        if max_error is not None and float(max_error) > one_pixel_tolerance:
            raise ValueError(f"{path}: SDF replay error exceeds one label pixel")

    shape_y, shape_x = (int(value) for value in metadata["shape_yx"])
    return {
        "sample": sample,
        "patient": str(source_row["patient"]),
        "cancer": str(source_row["cancer"]),
        "fold": int(source_row["fold"]),
        "path": str(path.resolve()),
        "sha256": sha256_file(path),
        "size_bytes": path.stat().st_size,
        "shape_y": shape_y,
        "shape_x": shape_x,
        "x_min_um": float(metadata["x_min_um"]),
        "y_min_um": float(metadata["y_min_um"]),
        "resolution_um": float(metadata["resolution_um"]),
        "n_cells": int(metadata["n_joined_cells"]),
        "tumor_pixels": int(metadata["tumor_pixels"]),
        "normal_evidence_pixels": int(metadata["normal_evidence_pixels"]),
        "tissue_pixels": int(metadata["tissue_pixels"]),
        "valid_pixels": int(metadata["valid_pixels"]),
        "interface_pixels": int(metadata["interface_pixels"]),
        "interface_empty": bool(metadata["interface_empty"]),
        "no_tumor": bool(metadata["no_tumor"]),


        "full_cell_type_disagreement": _optional_int(
            metadata.get("label_transfer_audit", {}), "full_cell_type_disagreement"
        ),
        "tumor_seed_disagreement": _optional_int(
            metadata.get("label_transfer_audit", {}), "tumor_seed_disagreement"
        ),

        "cells_joined": _optional_int(metadata.get("label_transfer_audit", {}), "joined_cells"),
        "cells_without_v9_coordinate": _optional_int(
            metadata.get("label_transfer_audit", {}), "gen11_cells_without_v9_coordinate"
        ),
        "side_mismatch": _optional_int(verification, "side_mismatch"),
        "sdf_rounding_difference_count": _optional_int(
            verification, "sdf_rounding_difference_count"
        ),
        "sdf_mismatch": _optional_int(verification, "sdf_mismatch"),
        "sdf_over_one_label_pixel_count": _optional_int(
            verification, "sdf_over_one_label_pixel_count"
        ),
        "sdf_max_abs_error_um": max_error,
        "interface_pixel_delta_vs_gen11_log": _optional_int(
            metadata, "interface_pixel_delta_vs_gen11_log"
        ),
    }


def finalize(config_path: Path) -> tuple[pd.DataFrame, dict[str, Any]]:
    config_path = config_path.resolve()
    project_root = config_path.parent.parent
    config = load_experiment(config_path)
    source_manifest_path = project_root / config["outputs"]["source_manifest"]
    source_contract_path = project_root / config["outputs"]["data_contract"]
    source = pd.read_csv(source_manifest_path)
    if "n_samples" not in config["cohort"]:
        raise ValueError("configuration has no cohort binding; stage data/manifests/cohort_binding.json")
    if len(source) != int(config["cohort"]["n_samples"]):
        raise ValueError("source manifest sample count differs from configuration")

    records = [
        inspect_archive(row)
        for _, row in source.sort_values(["fold", "cancer", "sample"]).iterrows()
    ]
    labels = pd.DataFrame(records)
    if labels["sample"].duplicated().any():
        raise ValueError("duplicate samples in label dataset")
    patient_folds = labels.groupby("patient")["fold"].nunique()
    if int(patient_folds.max()) != 1:
        raise ValueError("patient leakage in label dataset")
    if int(labels["side_mismatch"].sum()) != 0:
        raise ValueError("full label dataset has cell-side mismatches")

    cancers = Counter(labels["cancer"])
    contract: dict[str, Any] = {
        "schema_version": "he_boundary_prediction.v1.label_dataset.v1",
        "status": "VALIDATED_COHORT_LABEL_DATASET",
        "experiment_id": config["experiment_id"],
        "sources": {
            "config": {
                "path": str(config_path),
                "sha256": sha256_file(config_path),
            },
            "source_manifest": {
                "path": str(source_manifest_path.resolve()),
                "sha256": sha256_file(source_manifest_path),
            },
            "source_contract": {
                "path": str(source_contract_path.resolve()),
                "sha256": sha256_file(source_contract_path),
            },
            "compiler": {
                "path": str(
                    (project_root / "scripts" / "export_boundary_labels.py").resolve()
                ),
                "sha256": sha256_file(
                    project_root / "scripts" / "export_boundary_labels.py"
                ),
            },
        },
        "summary": {
            "n_samples": len(labels),
            "n_patients": int(labels["patient"].nunique()),
            "n_cancers": int(labels["cancer"].nunique()),
            "n_cells": int(labels["n_cells"].sum()),
            "samples_by_cancer": dict(sorted(cancers.items())),
            "total_archive_bytes": int(labels["size_bytes"].sum()),
            "total_tumor_pixels": int(labels["tumor_pixels"].sum()),
            "total_valid_pixels": int(labels["valid_pixels"].sum()),
            "samples_without_explicit_interface": int(
                labels["interface_empty"].sum()
            ),
            "samples_without_tumor": int(labels["no_tumor"].sum()),
            "full_cell_type_disagreement": int(
                labels["full_cell_type_disagreement"].sum()
            ),
            "tumor_seed_disagreement": int(
                labels["tumor_seed_disagreement"].sum()
            ),
            "cell_side_mismatch": int(labels["side_mismatch"].sum()),
            "sdf_rounding_difference_count": int(
                labels["sdf_rounding_difference_count"].sum()
            ),
            "sdf_mismatch": int(labels["sdf_mismatch"].sum()),
            "sdf_over_one_label_pixel_count": int(
                labels["sdf_over_one_label_pixel_count"].sum()
            ),
            "maximum_sdf_error_um": float(
                labels["sdf_max_abs_error_um"].fillna(0.0).max()
            ),
        },
        "folds": {
            "patient_disjoint": True,
            "records": {
                str(int(fold)): {
                    "samples": int(len(group)),
                    "patients": int(group["patient"].nunique()),
                }
                for fold, group in labels.groupby("fold", sort=True)
            },
        },
        "manifest": {
            "path": str(
                (project_root / config["outputs"]["label_manifest"]).resolve()
            ),
            "records_sha256": payload_sha256(records),
        },
    }
    contract["contract_payload_sha256"] = payload_sha256(contract)
    return labels, contract


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "configs" / "experiment.json",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config_path = args.config.resolve()
    project_root = config_path.parent.parent
    config = load_experiment(config_path)
    labels, contract = finalize(config_path)
    manifest_path = project_root / config["outputs"]["label_manifest"]
    contract_path = project_root / config["outputs"]["label_dataset_contract"]
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_tmp = manifest_path.with_suffix(manifest_path.suffix + ".tmp")
    labels.to_csv(manifest_tmp, index=False)
    manifest_tmp.replace(manifest_path)
    contract_tmp = contract_path.with_suffix(contract_path.suffix + ".tmp")
    contract_tmp.write_text(
        json.dumps(contract, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    contract_tmp.replace(contract_path)
    print(f"{contract['status']}: {len(labels)} samples")
    print(f"manifest: {manifest_path}")
    print(f"contract: {contract_path}")
    print(f"contract_payload_sha256: {contract['contract_payload_sha256']}")


if __name__ == "__main__":
    main()
