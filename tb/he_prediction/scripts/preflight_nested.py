#!/usr/bin/env python3

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import pandas as pd

import cohort_contract

import audit_input_content as input_audit
import validate_input_content_receipt as input_receipt_validator


def sha256_file(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{path}: expected a JSON object")
    return payload


def _science(payload: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in payload.items() if key != "outputs"}


def _validate_output_contract(
    config: dict[str, Any], expected: dict[str, str], path: Path
) -> None:
    outputs = config.get("outputs")
    if not isinstance(outputs, dict):
        raise ValueError(f"{path}: missing outputs")
    for field, value in expected.items():
        if outputs.get(field) != value:
            raise ValueError(
                f"{path}: outputs.{field}={outputs.get(field)!r}, "
                f"expected {value!r}"
            )


def validate(
    project_root: Path,
    base_config_path: Path,
    screen_config_path: Path,
    final_config_path: Path,
    base_candidates_path: Path,
    nested_candidates_path: Path,
    manifest_path: Path | str = "",
) -> dict[str, Any]:
    root = project_root.resolve()
    base_config_path = base_config_path.resolve()
    screen_config_path = screen_config_path.resolve()
    final_config_path = final_config_path.resolve()
    base_candidates_path = base_candidates_path.resolve()
    nested_candidates_path = nested_candidates_path.resolve()
    base_config = read_json(base_config_path)
    screen_config = read_json(screen_config_path)
    final_config = read_json(final_config_path)
    if _science(screen_config) != _science(base_config):
        raise ValueError("Dense screen changes the base scientific contract")
    if _science(final_config) != _science(base_config):
        raise ValueError("Dense final changes the base scientific contract")
    _validate_output_contract(
        screen_config,
        {
            "models": "outputs/control/screen/models",
            "predictions": "outputs/control/screen/predictions",
            "results": "outputs/control/screen/results",
            "manifests": "outputs/control/screen/manifests",
            "logs": "logs/control/screen",
            "scratch": "scratch/control/screen",
        },
        screen_config_path,
    )
    _validate_output_contract(
        final_config,
        {
            "models": "outputs/control/models",
            "predictions": "outputs/control/predictions",
            "results": "outputs/control/results",
            "manifests": "outputs/control/manifests",
            "logs": "logs/control",
            "scratch": "scratch/control",
        },
        final_config_path,
    )

    base_candidates = read_json(base_candidates_path)
    nested_candidates = read_json(nested_candidates_path)
    for field in ("input_contract", "candidates"):
        if nested_candidates.get(field) != base_candidates.get(field):
            raise ValueError(f"Dense candidates change {field}")
    base_runtime = dict(base_candidates["training_runtime"])
    nested_runtime = dict(nested_candidates["training_runtime"])
    for field, value in base_runtime.items():
        if field == "threshold_candidates":
            continue
        if nested_runtime.get(field) != value:
            raise ValueError(f"Dense runtime changes {field}")
    expected_thresholds = [round(index * 0.05, 2) for index in range(1, 20)]
    if nested_runtime.get("threshold_candidates") != expected_thresholds:
        raise ValueError("Dense threshold grid is not 0.05..0.95 by 0.05")
    contract = nested_runtime.get("validation_contract", {})
    expected_contract = {
        "grid": "full_valid_tissue_half_stride_hann",
        "origins": "dense_origins_stride_128_with_bottom_right_tail_alignment",
        "blend": "nonzero_hann",
        "batching": "batch_size_4_flush_at_each_sample_boundary",
        "coverage": "each_valid_10um_pixel_scored_once_after_stitching",
        "probability_precision": (
            "tile_sigmoid_float32_then_float32_hann_stitch_then_"
            "canvas_float16_roundtrip"
        ),
        "aggregation": "sample_metric_then_mean_within_patient_then_patient_equal",
        "threshold_tie_break": "closest_to_0.5_then_lower",
    }
    if not isinstance(contract, dict) or any(
        contract.get(field) != value
        for field, value in expected_contract.items()
    ):
        raise ValueError("Dense validation contract differs")

    manifest_path = root / base_config["outputs"]["source_manifest"]
    table = pd.read_csv(manifest_path, keep_default_na=False)
    required = {
        "sample",
        "patient",
        "cancer",
        "fold",
        "label_path",
        "he_path",
    }
    if not required.issubset(table.columns):
        raise ValueError("source manifest is missing required columns")
    cohort_contract.validate_working_cohort(table, manifest_path)
    if (table.groupby("patient")["fold"].nunique() != 1).any():
        raise ValueError("source manifest has patients spanning folds")
    if (table.groupby("patient")["cancer"].nunique() != 1).any():
        raise ValueError("source manifest has patients spanning cancers")
    missing_labels = [
        Path(path) for path in table["label_path"] if not Path(path).is_file()
    ]
    missing_he = [
        Path(path) for path in table["he_path"] if not Path(path).is_file()
    ]
    if missing_labels or missing_he:
        raise FileNotFoundError(
            f"missing labels={len(missing_labels)}, H&E={len(missing_he)}"
        )

    label_receipt_path = (
        root / base_config["outputs"]["label_dataset_contract"]
    )
    label_manifest_path = root / base_config["outputs"]["label_manifest"]
    data_contract_path = root / base_config["outputs"]["data_contract"]
    label_receipt = read_json(label_receipt_path)
    data_contract = read_json(data_contract_path)
    input_audit.validate_payload_hash(
        data_contract,
        "contract_payload_sha256",
        data_contract_path,
    )
    input_audit.validate_payload_hash(
        label_receipt,
        "contract_payload_sha256",
        label_receipt_path,
    )
    label_summary = label_receipt.get("summary", {})
    if (
        label_receipt.get("status") != "VALIDATED_COHORT_LABEL_DATASET"
        or int(label_summary.get("n_samples", -1)) != 177
        or int(label_summary.get("n_patients", -1)) != 171
        or int(label_summary.get("n_cancers", -1)) != 9
        or int(label_summary.get("cell_side_mismatch", -1)) != 0
        or int(label_summary.get("sdf_over_one_label_pixel_count", -1)) != 0
    ):
        raise ValueError("label dataset receipt fails validation gates")
    manifest_sha256 = sha256_file(manifest_path)
    if (
        label_receipt.get("sources", {})
        .get("source_manifest", {})
        .get("sha256")
        != manifest_sha256
    ):
        raise ValueError("label receipt source-manifest hash differs")
    if (
        label_receipt.get("sources", {}).get("config", {}).get("sha256")
        != sha256_file(base_config_path)
    ):
        raise ValueError("label receipt base-config hash differs")
    if (
        label_receipt.get("sources", {})
        .get("source_contract", {})
        .get("sha256")
        != sha256_file(data_contract_path)
    ):
        raise ValueError("label receipt source-contract hash differs")

    input_receipt_path = (
        root / "data" / "manifests" / "input_content_receipt.json"
    )
    input_validation = input_receipt_validator.validate_receipt(
        root, input_receipt_path
    )
    if (
        input_validation.get("content_trees", {})
        .get("aligned_he", {})
        .get("n_files")
        != 177
        or input_validation.get("content_trees", {})
        .get("dense_labels", {})
        .get("n_files")
        != 177
    ):
        raise ValueError("Dense input-content receipt does not close full177")

    tile_index_path = root / "data" / "manifests" / "tile_index.parquet"
    tile_receipt_path = (
        root / "data" / "manifests" / "tile_index_receipt.json"
    )
    tile_receipt = read_json(tile_receipt_path)
    scope = tile_receipt.get("scope", {})
    if (
        tile_receipt.get("status") != "VALIDATED_TILE_INDEX"
        or int(scope.get("n_source_samples", -1)) != 177
        or int(scope.get("n_patients", -1)) != 171
        or not tile_receipt.get("coverage", {}).get(
            "all_valid_pixels_covered", False
        )
    ):
        raise ValueError("tile-index receipt fails validation gates")
    if (
        tile_receipt.get("artifact", {}).get("sha256")
        != sha256_file(tile_index_path)
    ):
        raise ValueError("tile-index content hash differs")
    if (
        tile_receipt.get("sources", {})
        .get("source_manifest", {})
        .get("sha256")
        != manifest_sha256
    ):
        raise ValueError("tile receipt source-manifest hash differs")
    if (
        tile_receipt.get("sources", {})
        .get("label_dataset_contract", {})
        .get("sha256")
        != sha256_file(label_receipt_path)
    ):
        raise ValueError("tile receipt label-receipt hash differs")
    finalized_labels = tile_receipt.get("sources", {}).get(
        "finalized_label_manifest", {}
    )
    if (
        finalized_labels.get("sha256") != sha256_file(label_manifest_path)
        or finalized_labels.get("records_sha256")
        != label_receipt.get("manifest", {}).get("records_sha256")
    ):
        raise ValueError("tile receipt finalized-label manifest differs")

    return {
        "status": "R4_NESTED_PREFLIGHT_PASSED",
        "cohort": {
            "n_samples": 177,
            "n_patients": 171,
            "n_cancers": 9,
            "n_folds": 5,
        },
        "hashes": {
            "base_config": sha256_file(base_config_path),
            "screen_config": sha256_file(screen_config_path),
            "final_config": sha256_file(final_config_path),
            "base_candidates": sha256_file(base_candidates_path),
            "nested_candidates": sha256_file(nested_candidates_path),
            "source_manifest": manifest_sha256,
            "data_contract": sha256_file(data_contract_path),
            "label_manifest": sha256_file(label_manifest_path),
            "label_dataset": sha256_file(label_receipt_path),
            "input_content_receipt": sha256_file(input_receipt_path),
            "input_receipt_validator": sha256_file(
                Path(input_receipt_validator.__file__).resolve()
            ),
            "input_content_receipt_payload": input_validation["receipt"][
                "payload_sha256"
            ],
            "aligned_he_tree": input_validation["content_trees"][
                "aligned_he"
            ]["receipt_tree_sha256"],
            "aligned_he_portable_tree": input_validation[
                "content_trees"
            ]["aligned_he"]["portable_tree_sha256"],
            "dense_label_tree": input_validation["content_trees"][
                "dense_labels"
            ]["receipt_tree_sha256"],
            "dense_label_portable_tree": input_validation[
                "content_trees"
            ]["dense_labels"]["portable_tree_sha256"],
            "tile_index": sha256_file(tile_index_path),
            "tile_receipt": sha256_file(tile_receipt_path),
        },
    }


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", type=Path, default=root)
    parser.add_argument(
        "--base-config",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--screen-config",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--final-config",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--base-candidates",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--nested-candidates",
        type=Path,
        required=True,
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = validate(
        args.project_root,
        args.base_config,
        args.screen_config,
        args.final_config,
        args.base_candidates,
        args.nested_candidates,
    )
    print(json.dumps(result, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
