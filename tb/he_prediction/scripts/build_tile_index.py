#!/usr/bin/env python3

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import tempfile
from collections import Counter
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

from experiment_config import load_experiment
from he_mask_dataset import sha256_file, validate_patient_fold_contract


def covering_starts(length: int, tile: int, stride: int) -> list[int]:

    if length <= 0 or tile <= 0 or stride <= 0:
        raise ValueError("length, tile, and stride must be positive")
    if stride > tile:
        raise ValueError("stride cannot exceed tile size without coverage gaps")
    if length <= tile:
        return [0]
    final = length - tile
    starts = list(range(0, final + 1, stride))
    if starts[-1] != final:
        starts.append(final)
    return starts


def integral_image(mask: np.ndarray) -> np.ndarray:
    return np.pad(
        mask.astype(np.int64, copy=False),
        ((1, 0), (1, 0)),
        mode="constant",
    ).cumsum(0).cumsum(1)


def window_sum(
    integral: np.ndarray, y0: int, x0: int, height: int, width: int
) -> int:
    array_h = integral.shape[0] - 1
    array_w = integral.shape[1] - 1
    y1 = min(y0 + height, array_h)
    x1 = min(x0 + width, array_w)
    if y0 >= y1 or x0 >= x1:
        return 0
    return int(
        integral[y1, x1]
        - integral[y0, x1]
        - integral[y1, x0]
        + integral[y0, x0]
    )


def _same_path(left: str | Path, right: str | Path) -> bool:
    def key(value: str | Path) -> str:
        normalized = os.path.abspath(os.path.normpath(str(value)))
        alias_from = os.environ.get("PATH_ALIAS_FROM", "")
        alias_to = os.environ.get("PATH_ALIAS_TO", "")
        if alias_from and (normalized == alias_from or normalized.startswith(alias_from + "/")):
            normalized = alias_to + normalized[len(alias_from):]
        return normalized

    return key(left) == key(right)


def load_label_for_index(
    source_row: Mapping[str, Any],
) -> tuple[dict[str, np.ndarray], dict[str, Any], str]:
    label_path = Path(str(source_row["label_path"]))
    label_sha256 = sha256_file(label_path)
    with np.load(label_path, allow_pickle=False) as archive:
        required = {"tumor_mask", "label_valid", "sdf_um", "metadata_json"}
        missing = required.difference(archive.files)
        if missing:
            raise ValueError(
                f"{label_path}: missing required arrays {sorted(missing)}"
            )
        arrays = {
            "tumor_mask": np.asarray(archive["tumor_mask"], dtype=np.uint8),
            "label_valid": np.asarray(archive["label_valid"], dtype=np.uint8),
            "sdf_um": np.asarray(archive["sdf_um"], dtype=np.float32),
        }
        metadata = json.loads(str(archive["metadata_json"].item()))
    shapes = {tuple(array.shape) for array in arrays.values()}
    if len(shapes) != 1:
        raise ValueError(f"{label_path}: label arrays have inconsistent shapes")
    shape = next(iter(shapes))
    if tuple(metadata.get("shape_yx", ())) != shape:
        raise ValueError(f"{label_path}: metadata shape differs from arrays")
    expected_scalars = {
        "sample": str(source_row["sample"]),
        "patient": str(source_row["patient"]),
        "cancer": str(source_row["cancer"]),
        "fold": int(source_row["fold"]),
    }
    for field, expected in expected_scalars.items():
        if metadata.get(field) != expected:
            raise ValueError(
                f"{label_path}: {field}={metadata.get(field)!r}, expected {expected!r}"
            )
    if metadata.get("axes") != "YX":
        raise ValueError(f"{label_path}: expected YX label axes")
    if metadata.get("origin_semantics") != "upper_left_pixel_edge":
        raise ValueError(f"{label_path}: unsupported label-origin semantics")
    he = metadata.get("he", {})
    if not _same_path(he.get("path", ""), source_row["he_path"]):
        raise ValueError(f"{label_path}: H&E path differs from source manifest")
    if str(he.get("sha256", "")) != str(source_row["he_sha256"]):
        raise ValueError(f"{label_path}: H&E hash differs from source manifest")
    if tuple(he.get("shape_yx", ())) != (
        int(source_row["he_size_y_px"]),
        int(source_row["he_size_x_px"]),
    ):
        raise ValueError(f"{label_path}: H&E shape differs from source manifest")
    source_cell = metadata.get("sources", {}).get("cell_coordinates", {})
    if not _same_path(source_cell.get("path", ""), source_row["cell_table_path"]):
        raise ValueError(f"{label_path}: cell-table path differs from manifest")
    if str(source_cell.get("sha256", "")) != str(source_row["cell_table_sha256"]):
        raise ValueError(f"{label_path}: cell-table hash differs from manifest")
    boundary = metadata.get("sources", {}).get(
        "gen11_cell_type_and_reference", {}
    )
    if not _same_path(
        boundary.get("path", ""), source_row["boundary_reference_csv"]
    ):
        raise ValueError(f"{label_path}: molecular reference path differs from manifest")
    if len(str(boundary.get("sha256", ""))) != 64:
        raise ValueError(f"{label_path}: molecular reference hash is absent")
    return arrays, metadata, label_sha256


def _window_stratum(
    boundary_pixels: int, tumor_pixels: int, valid_pixels: int
) -> str:
    if boundary_pixels > 0:
        return "boundary"
    if valid_pixels > 0 and tumor_pixels / valid_pixels >= 0.5:
        return "tumor_interior"
    return "normal_tissue"


def _anchor_stratum(
    tumor: np.ndarray,
    valid: np.ndarray,
    sdf_um: np.ndarray,
    y: int,
    x: int,
    boundary_candidate_um: float,
) -> str | None:

    if y < 0 or x < 0 or y >= valid.shape[0] or x >= valid.shape[1]:
        return None
    if not bool(valid[y, x]):
        return None
    distance = float(sdf_um[y, x])
    if math.isfinite(distance) and abs(distance) <= boundary_candidate_um:
        return "boundary"
    if bool(tumor[y, x]):
        return "tumor_interior"
    return "normal_tissue"


def rows_for_label(
    source_row: Mapping[str, Any],
    arrays: Mapping[str, np.ndarray],
    metadata: Mapping[str, Any],
    label_sha256: str,
    *,
    kind: str,
    tile_shape_yx: tuple[int, int],
    stride_shape_yx: tuple[int, int],
    input_shape_yx: tuple[int, int],
    fov_shape_um: tuple[float, float],
    minimum_valid_fraction: float,
    boundary_candidate_um: float,
) -> list[dict[str, Any]]:
    label_h, label_w = arrays["label_valid"].shape
    tile_h, tile_w = tile_shape_yx
    stride_h, stride_w = stride_shape_yx
    input_h, input_w = input_shape_yx
    fov_h_um, fov_w_um = fov_shape_um
    label_mpp_um = float(metadata["resolution_um"])
    if not math.isclose(tile_h * label_mpp_um, fov_h_um, abs_tol=1e-6):
        raise ValueError("label height and physical FOV disagree")
    if not math.isclose(tile_w * label_mpp_um, fov_w_um, abs_tol=1e-6):
        raise ValueError("label width and physical FOV disagree")
    input_mpp_y = fov_h_um / input_h
    input_mpp_x = fov_w_um / input_w
    if not math.isclose(input_mpp_y, input_mpp_x, abs_tol=1e-9):
        raise ValueError("anisotropic model pixels are not supported")
    if not 0 <= minimum_valid_fraction <= 1:
        raise ValueError("minimum valid fraction must be in [0, 1]")

    valid = arrays["label_valid"].astype(bool)
    tumor = arrays["tumor_mask"].astype(bool) & valid
    sdf = arrays["sdf_um"]
    boundary = valid & np.isfinite(sdf) & (
        np.abs(sdf) <= float(boundary_candidate_um)
    )
    valid_integral = integral_image(valid)
    tumor_integral = integral_image(tumor)
    boundary_integral = integral_image(boundary)
    starts_y = covering_starts(label_h, tile_h, stride_h)
    starts_x = covering_starts(label_w, tile_w, stride_w)
    denominator = tile_h * tile_w
    rows: list[dict[str, Any]] = []
    for y0 in starts_y:
        for x0 in starts_x:
            valid_pixels = window_sum(valid_integral, y0, x0, tile_h, tile_w)
            valid_fraction = valid_pixels / denominator
            if kind == "train_candidate":
                if valid_fraction < minimum_valid_fraction:
                    continue
            elif kind == "eval":
                if valid_pixels == 0:
                    continue
            else:
                raise ValueError(f"unknown tile kind: {kind}")
            tumor_pixels = window_sum(tumor_integral, y0, x0, tile_h, tile_w)
            boundary_pixels = window_sum(
                boundary_integral, y0, x0, tile_h, tile_w
            )
            anchor_y = y0 + tile_h // 2
            anchor_x = x0 + tile_w // 2
            if kind == "train_candidate":
                stratum = _anchor_stratum(
                    tumor,
                    valid,
                    sdf,
                    anchor_y,
                    anchor_x,
                    boundary_candidate_um,
                )
                if stratum is None:
                    continue
                stratum_basis = "valid_tile_centre_sdf"
            else:
                stratum = _window_stratum(
                    boundary_pixels, tumor_pixels, valid_pixels
                )
                stratum_basis = "diagnostic_window_content"
            physical_x0_um = float(metadata["x_min_um"]) + x0 * label_mpp_um
            physical_y0_um = float(metadata["y_min_um"]) + y0 * label_mpp_um
            valid_h = min(tile_h, label_h - y0)
            valid_w = min(tile_w, label_w - x0)
            row = {
                "tile_id": (
                    f"{kind}:{source_row['sample']}:y{y0:05d}:x{x0:05d}"
                ),
                "kind": kind,
                "sample": str(source_row["sample"]),
                "patient": str(source_row["patient"]),
                "cancer": str(source_row["cancer"]),
                "fold": int(source_row["fold"]),
                "stratum": stratum,
                "stratum_basis": stratum_basis,
                "anchor_y_px": anchor_y,
                "anchor_x_px": anchor_x,
                "anchor_valid": bool(
                    anchor_y < label_h
                    and anchor_x < label_w
                    and valid[anchor_y, anchor_x]
                ),
                "anchor_tumor": bool(
                    anchor_y < label_h
                    and anchor_x < label_w
                    and tumor[anchor_y, anchor_x]
                ),
                "anchor_sdf_um": (
                    float(sdf[anchor_y, anchor_x])
                    if anchor_y < label_h and anchor_x < label_w
                    else math.nan
                ),
                "valid_pixels": valid_pixels,
                "valid_fraction": valid_fraction,
                "tumor_valid_pixels": tumor_pixels,
                "tumor_valid_fraction": (
                    tumor_pixels / valid_pixels if valid_pixels else 0.0
                ),
                "boundary_valid_pixels": boundary_pixels,
                "boundary_valid_fraction": (
                    boundary_pixels / valid_pixels if valid_pixels else 0.0
                ),
                "he_path": str(source_row["he_path"]),
                "he_sha256": str(source_row["he_sha256"]),
                "he_size_y_px": int(source_row["he_size_y_px"]),
                "he_size_x_px": int(source_row["he_size_x_px"]),
                "he_mpp_um": float(source_row["he_mpp_um"]),
                "label_path": str(source_row["label_path"]),
                "label_sha256": label_sha256,
                "label_size_y_px": label_h,
                "label_size_x_px": label_w,
                "label_mpp_um": label_mpp_um,
                "label_origin_x_um": float(metadata["x_min_um"]),
                "label_origin_y_um": float(metadata["y_min_um"]),
                "label_y0_px": y0,
                "label_x0_px": x0,
                "label_valid_h_px": valid_h,
                "label_valid_w_px": valid_w,
                "label_tile_h_px": tile_h,
                "label_tile_w_px": tile_w,
                "physical_x0_um": physical_x0_um,
                "physical_y0_um": physical_y0_um,
                "input_h_px": input_h,
                "input_w_px": input_w,
                "input_mpp_um": input_mpp_y,
                "fov_h_um": fov_h_um,
                "fov_w_um": fov_w_um,
                "boundary_candidate_um": float(boundary_candidate_um),
                "source_cell_table_sha256": str(
                    source_row["cell_table_sha256"]
                ),
                "source_boundary_reference_sha256": str(
                    metadata["sources"]["gen11_cell_type_and_reference"][
                        "sha256"
                    ]
                ),
            }
            rows.append(row)
    return rows


def verify_eval_coverage(
    label_valid: np.ndarray,
    rows: Sequence[Mapping[str, Any]],
) -> dict[str, int]:
    difference = np.zeros(
        (label_valid.shape[0] + 1, label_valid.shape[1] + 1), dtype=np.int32
    )
    for row in rows:
        y0 = int(row["label_y0_px"])
        x0 = int(row["label_x0_px"])
        y1 = y0 + int(row["label_valid_h_px"])
        x1 = x0 + int(row["label_valid_w_px"])
        difference[y0, x0] += 1
        difference[y1, x0] -= 1
        difference[y0, x1] -= 1
        difference[y1, x1] += 1
    coverage = difference[:-1, :-1].cumsum(0).cumsum(1)
    valid = label_valid.astype(bool)
    if valid.any() and int(coverage[valid].min()) <= 0:
        raise ValueError("evaluation tiling leaves valid label pixels uncovered")
    return {
        "valid_pixels": int(valid.sum()),
        "minimum_valid_pixel_coverage": int(coverage[valid].min())
        if valid.any()
        else 0,
        "maximum_valid_pixel_coverage": int(coverage[valid].max())
        if valid.any()
        else 0,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    project_root = Path(__file__).resolve().parents[1]
    parser.add_argument(
        "--config",
        type=Path,
        default=project_root / "configs" / "experiment.json",
    )
    parser.add_argument(
        "--source-manifest",
        type=Path,
        default=project_root / "data" / "manifests" / "source_cohort.csv",
    )
    parser.add_argument(
        "--label-manifest",
        type=Path,
        default=project_root / "data" / "manifests" / "labels.csv",
    )
    parser.add_argument(
        "--label-dataset-contract",
        type=Path,
        default=project_root / "data" / "manifests" / "label_dataset.json",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=project_root / "data" / "manifests" / "tile_index.parquet",
    )
    parser.add_argument(
        "--receipt",
        type=Path,
        default=project_root
        / "data"
        / "manifests"
        / "tile_index_receipt.json",
    )
    parser.add_argument("--train-stride-label-px", type=int, default=64)
    parser.add_argument("--eval-stride-label-px", type=int, default=128)
    parser.add_argument("--input-px", type=int)
    parser.add_argument("--label-tile-px", type=int)
    parser.add_argument("--fov-um", type=float)
    parser.add_argument("--minimum-valid-fraction", type=float)
    parser.add_argument("--boundary-candidate-um", type=float)
    parser.add_argument(
        "--sample",
        action="append",
        help="Optional bounded pilot; may be repeated.",
    )
    return parser.parse_args()


def _atomic_parquet(table: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        dir=path.parent, prefix=path.name + ".", suffix=".tmp", delete=False
    ) as handle:
        temporary = Path(handle.name)
    try:
        table.to_parquet(temporary, index=False)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _atomic_json(payload: Mapping[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        dir=path.parent,
        prefix=path.name + ".",
        suffix=".tmp",
        mode="w",
        encoding="utf-8",
        delete=False,
    ) as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
        temporary = Path(handle.name)
    try:
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def main() -> None:
    args = parse_args()
    config_path = args.config.resolve()
    source_path = args.source_manifest.resolve()
    label_manifest_path = args.label_manifest.resolve()
    label_contract_path = args.label_dataset_contract.resolve()
    config = load_experiment(config_path)
    source = pd.read_csv(source_path)
    validate_patient_fold_contract(source)
    label_contract = json.loads(
        label_contract_path.read_text(encoding="utf-8")
    )
    if label_contract.get("status") != "VALIDATED_COHORT_LABEL_DATASET":
        raise ValueError("full label dataset contract is not validated")
    if int(label_contract.get("summary", {}).get("n_samples", -1)) != len(
        source
    ):
        raise ValueError("label contract and source manifest sample counts differ")
    if (
        label_contract.get("sources", {})
        .get("config", {})
        .get("sha256")
        != sha256_file(config_path)
    ):
        raise ValueError("label dataset was finalized against a different config")
    if (
        label_contract.get("sources", {})
        .get("source_manifest", {})
        .get("sha256")
        != sha256_file(source_path)
    ):
        raise ValueError(
            "label dataset was finalized against a different source manifest"
        )
    labels = pd.read_csv(label_manifest_path)
    if (
        len(labels) != len(source)
        or labels["sample"].duplicated().any()
        or set(labels["sample"]) != set(source["sample"])
    ):
        raise ValueError("finalized label manifest does not match source cohort")
    validate_patient_fold_contract(labels)
    label_by_sample = labels.set_index("sample", verify_integrity=True)
    if args.sample:
        requested = set(args.sample)
        source = source.loc[source["sample"].isin(requested)].copy()
        missing = requested.difference(source["sample"])
        if missing:
            raise ValueError(f"unknown samples: {sorted(missing)}")
    if source.empty:
        raise ValueError("source cohort selection is empty")

    preprocessing = config["preprocessing"]
    sampling = config["sampling"]
    input_px = int(args.input_px or preprocessing["model_input_shape_yx"][0])
    label_tile_px = int(
        args.label_tile_px or preprocessing["label_shape_yx"][0]
    )
    fov_um = float(args.fov_um or preprocessing["patch_fov_um"])
    minimum_valid_fraction = float(
        args.minimum_valid_fraction
        if args.minimum_valid_fraction is not None
        else sampling["minimum_valid_tissue_fraction"]
    )
    boundary_candidate_um = float(
        args.boundary_candidate_um
        if args.boundary_candidate_um is not None
        else sampling["boundary_candidate_um"]
    )
    if preprocessing["model_input_shape_yx"][0] != preprocessing[
        "model_input_shape_yx"
    ][1]:
        raise ValueError("current data pipeline requires square model input")
    if preprocessing["label_shape_yx"][0] != preprocessing["label_shape_yx"][1]:
        raise ValueError("current data pipeline requires square labels")

    all_rows: list[dict[str, Any]] = []
    coverage_records: dict[str, dict[str, int]] = {}
    label_digests: list[tuple[str, str]] = []
    for _, source_row in source.sort_values(
        ["fold", "cancer", "sample"]
    ).iterrows():
        arrays, metadata, label_sha = load_label_for_index(source_row)
        finalized = label_by_sample.loc[str(source_row["sample"])]
        if not _same_path(finalized["path"], source_row["label_path"]):
            raise ValueError(
                f"{source_row['sample']}: finalized label path differs"
            )
        if str(finalized["sha256"]) != label_sha:
            raise ValueError(
                f"{source_row['sample']}: finalized label hash differs"
            )
        label_digests.append((str(source_row["sample"]), label_sha))
        train_rows = rows_for_label(
            source_row,
            arrays,
            metadata,
            label_sha,
            kind="train_candidate",
            tile_shape_yx=(label_tile_px, label_tile_px),
            stride_shape_yx=(
                args.train_stride_label_px,
                args.train_stride_label_px,
            ),
            input_shape_yx=(input_px, input_px),
            fov_shape_um=(fov_um, fov_um),
            minimum_valid_fraction=minimum_valid_fraction,
            boundary_candidate_um=boundary_candidate_um,
        )
        eval_rows = rows_for_label(
            source_row,
            arrays,
            metadata,
            label_sha,
            kind="eval",
            tile_shape_yx=(label_tile_px, label_tile_px),
            stride_shape_yx=(
                args.eval_stride_label_px,
                args.eval_stride_label_px,
            ),
            input_shape_yx=(input_px, input_px),
            fov_shape_um=(fov_um, fov_um),
            minimum_valid_fraction=minimum_valid_fraction,
            boundary_candidate_um=boundary_candidate_um,
        )
        coverage_records[str(source_row["sample"])] = verify_eval_coverage(
            arrays["label_valid"], eval_rows
        )
        all_rows.extend(train_rows)
        all_rows.extend(eval_rows)
    index = pd.DataFrame(all_rows)
    if index.empty:
        raise ValueError("no eligible tiles were generated")
    if index["tile_id"].duplicated().any():
        raise ValueError("generated tile IDs are not unique")
    validate_patient_fold_contract(index)
    index = index.sort_values(
        ["kind", "fold", "cancer", "sample", "label_y0_px", "label_x0_px"]
    ).reset_index(drop=True)
    _atomic_parquet(index, args.output)
    output_sha = sha256_file(args.output)

    label_digest = hashlib.sha256()
    for sample, digest in sorted(label_digests):
        label_digest.update(f"{sample}\t{digest}\n".encode("utf-8"))
    counts = Counter(index["kind"])
    stratum_counts = {
        kind: {
            stratum: int(count)
            for stratum, count in group["stratum"].value_counts().sort_index().items()
        }
        for kind, group in index.groupby("kind", sort=True)
    }
    receipt = {
        "schema_version": "he_boundary_prediction.v1.tile_index.v1",
        "status": "VALIDATED_TILE_INDEX",
        "scope": {
            "n_source_samples": int(source["sample"].nunique()),
            "n_patients": int(source["patient"].nunique()),
            "n_rows": len(index),
            "rows_by_kind": {key: int(value) for key, value in sorted(counts.items())},
            "rows_by_kind_and_stratum": stratum_counts,
            "bounded_sample_filter": sorted(args.sample) if args.sample else None,
        },
        "geometry": {
            "coordinate_origin": "xenium_canvas_upper_left_pixel_edge",
            "target_sampling": "physical_pixel_centres",
            "label_origin": "per_sample_upper_left_pixel_edge",
            "fov_um": fov_um,
            "input_shape_yx": [input_px, input_px],
            "input_mpp_um": fov_um / input_px,
            "label_shape_yx": [label_tile_px, label_tile_px],
            "label_mpp_um": fov_um / label_tile_px,
            "train_stride_label_px": args.train_stride_label_px,
            "eval_stride_label_px": args.eval_stride_label_px,
            "eval_reconstruction": "positive_Hann_overlap_add",
        },
        "sampling": {
            "minimum_valid_fraction": minimum_valid_fraction,
            "boundary_candidate_um": boundary_candidate_um,
            "train_stratum_rule": (
                "retain only valid tile-centre anchors; boundary if centre "
                "|SDF|<=threshold, tumor interior if centre is tumor beyond "
                "threshold, otherwise normal tissue"
            ),
            "eval_stratum_role": (
                "diagnostic only; evaluation rows remain exhaustive and use "
                "window-content categories"
            ),
        },
        "coverage": {
            "all_valid_pixels_covered": True,
            "minimum_over_samples": min(
                value["minimum_valid_pixel_coverage"]
                for value in coverage_records.values()
                if value["valid_pixels"] > 0
            ),
            "maximum_over_samples": max(
                value["maximum_valid_pixel_coverage"]
                for value in coverage_records.values()
            ),
            "by_sample": coverage_records,
        },
        "sources": {
            "config": {
                "path": str(config_path),
                "sha256": sha256_file(config_path),
            },
            "source_manifest": {
                "path": str(source_path),
                "sha256": sha256_file(source_path),
            },
            "finalized_label_manifest": {
                "path": str(label_manifest_path),
                "sha256": sha256_file(label_manifest_path),
                "records_sha256": label_contract["manifest"][
                    "records_sha256"
                ],
            },
            "label_dataset_contract": {
                "path": str(label_contract_path),
                "sha256": sha256_file(label_contract_path),
                "contract_payload_sha256": label_contract[
                    "contract_payload_sha256"
                ],
            },
            "label_set_digest_sha256": label_digest.hexdigest(),
        },
        "artifact": {
            "path": str(args.output.resolve()),
            "sha256": output_sha,
        },
        "runtime_contract": {
            "rgb_storage": "lazy_OME_TIFF_pyramid_no_materialized_RGB_tiles",
            "zarr": (
                "2.x synchronous API or 3.x native asynchronous API on a "
                "private same-thread loop"
            ),
        },
    }
    _atomic_json(receipt, args.receipt)
    print(
        f"{receipt['status']}: {len(index)} rows, "
        f"{source['sample'].nunique()} samples -> {args.output}",
        flush=True,
    )


if __name__ == "__main__":
    main()
