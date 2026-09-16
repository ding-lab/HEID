#!/usr/bin/env python3

from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from cell_source import load_joined_cells as load_corrected_cells
from scipy.ndimage import (
    binary_closing,
    binary_dilation,
    binary_erosion,
    binary_fill_holes,
    binary_opening,
    distance_transform_edt,
    gaussian_filter,
    label,
)
from skimage.morphology import disk


RESOLUTION_UM = 10.0
MARGIN_UM = 100.0
SIGMA_REGION = 5.0
SIGMA_TISSUE = 10.0
CLOSING_RADIUS = 5
OPENING_RADIUS = 3
MIN_REGION_AREA = 500
MIN_NORMAL_AREA = 3000
MAX_HOLE_AREA = 50000
TISSUE_THRESHOLD = 0.015
PARAMETER_OVERRIDES = {
    "PDAC": {
        "sigma_region": 3.0,
        "closing_radius": 2,
        "minimum_region_area": 100,
        "density_threshold": 0.03,
    }
}


def sha256_file(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def rasterize(
    x_um: np.ndarray,
    y_um: np.ndarray,
    ny: int,
    nx: int,
    x_min_um: float,
    y_min_um: float,
) -> np.ndarray:
    grid = np.zeros((ny, nx), dtype=np.float32)
    ix, iy = grid_indices(x_um, y_um, ny, nx, x_min_um, y_min_um)
    np.add.at(grid, (iy, ix), 1)
    return grid


def grid_indices(
    x_um: np.ndarray,
    y_um: np.ndarray,
    ny: int,
    nx: int,
    x_min_um: float,
    y_min_um: float,
) -> tuple[np.ndarray, np.ndarray]:
    ix = np.clip(((x_um - x_min_um) / RESOLUTION_UM).astype(int), 0, nx - 1)
    iy = np.clip(((y_um - y_min_um) / RESOLUTION_UM).astype(int), 0, ny - 1)
    return ix, iy


def fill_small_holes(mask: np.ndarray, max_hole_area: int) -> np.ndarray:
    filled = binary_fill_holes(mask)
    holes = filled & ~mask
    if not holes.any():
        return filled
    hole_labels, n_holes = label(holes)
    for component_id in range(1, n_holes + 1):
        if int((hole_labels == component_id).sum()) > max_hole_area:
            filled[hole_labels == component_id] = False
    return filled


def clean_mask(
    mask: np.ndarray,
    closing_radius: int,
    minimum_region_area: int,
) -> np.ndarray:
    cleaned = binary_closing(mask, structure=disk(closing_radius))
    cleaned = fill_small_holes(cleaned, MAX_HOLE_AREA)
    cleaned = binary_opening(cleaned, structure=disk(OPENING_RADIUS))
    component_labels, n_components = label(cleaned)
    for component_id in range(1, n_components + 1):
        if int((component_labels == component_id).sum()) < minimum_region_area:
            cleaned[component_labels == component_id] = False
    return cleaned


def signed_distance_um(tumor_mask: np.ndarray) -> np.ndarray:
    if not tumor_mask.any():
        return np.full(tumor_mask.shape, np.nan, dtype=np.float32)
    return (
        (
            distance_transform_edt(~tumor_mask)
            - distance_transform_edt(tumor_mask)
        )
        * RESOLUTION_UM
    ).astype(np.float32)


def build_dense_labels(
    cells: pd.DataFrame,
    cancer: str,
    he_size_y_px: int,
    he_size_x_px: int,
    he_mpp_um: float,
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    required = {"cell_id", "x_px", "y_px", "cell_type"}
    missing = required.difference(cells.columns)
    if missing:
        raise ValueError(f"missing cell columns: {sorted(missing)}")
    if cells.empty:
        raise ValueError("no joined cells")
    if cells["cell_id"].duplicated().any():
        raise ValueError("duplicate cell_id after join")

    x_um = cells["x_px"].to_numpy(dtype=np.float64) * he_mpp_um
    y_um = cells["y_px"].to_numpy(dtype=np.float64) * he_mpp_um
    if not np.isfinite(x_um).all() or not np.isfinite(y_um).all():
        raise ValueError("non-finite coordinates")
    tumor_seed = (
        cells["cell_type"].astype(str).str.startswith("Tumor").to_numpy(dtype=bool)
    )

    x_min_um = float(x_um.min() - MARGIN_UM)
    x_max_um = float(x_um.max() + MARGIN_UM)
    y_min_um = float(y_um.min() - MARGIN_UM)
    y_max_um = float(y_um.max() + MARGIN_UM)
    nx = int(np.ceil((x_max_um - x_min_um) / RESOLUTION_UM))
    ny = int(np.ceil((y_max_um - y_min_um) / RESOLUTION_UM))

    all_raw = rasterize(x_um, y_um, ny, nx, x_min_um, y_min_um)
    tissue_density = gaussian_filter(all_raw, sigma=SIGMA_TISSUE)
    tissue_mask = tissue_density > TISSUE_THRESHOLD
    tissue_mask = binary_closing(tissue_mask, structure=disk(10))
    tissue_mask = binary_fill_holes(tissue_mask)

    override = PARAMETER_OVERRIDES.get(cancer, {})
    sigma_region = float(override.get("sigma_region", SIGMA_REGION))
    closing_radius = int(override.get("closing_radius", CLOSING_RADIUS))
    minimum_region_area = int(
        override.get("minimum_region_area", MIN_REGION_AREA)
    )
    density_threshold = float(override.get("density_threshold", 0.1))

    type_masks: dict[str, np.ndarray] = {}
    density_maxima: dict[str, float] = {}
    for type_name, selection in (
        ("Tumor", tumor_seed),
        ("Normal", ~tumor_seed),
    ):
        if int(selection.sum()) < 10:
            type_masks[type_name] = np.zeros((ny, nx), dtype=bool)
            density_maxima[type_name] = 0.0
            continue
        raw = rasterize(
            x_um[selection],
            y_um[selection],
            ny,
            nx,
            x_min_um,
            y_min_um,
        )
        density = gaussian_filter(raw, sigma=sigma_region)
        density_maxima[type_name] = float(density.max())
        if density.max() > 0:
            type_masks[type_name] = clean_mask(
                density > density_threshold * density.max(),
                closing_radius=closing_radius,
                minimum_region_area=minimum_region_area,
            )
        else:
            type_masks[type_name] = np.zeros((ny, nx), dtype=bool)

    type_masks["Tumor"] &= tissue_mask
    type_masks["Normal"] &= tissue_mask
    type_masks["Normal"] &= ~type_masks["Tumor"]
    normal_components, n_normal_components = label(type_masks["Normal"])
    for component_id in range(1, n_normal_components + 1):
        if (
            int((normal_components == component_id).sum())
            < MIN_NORMAL_AREA
        ):
            type_masks["Normal"][normal_components == component_id] = False

    region_id = np.full((ny, nx), -1, dtype=np.int32)
    regions: list[dict[str, Any]] = []
    global_id = 0
    display_id = 1
    for type_name in ("Tumor", "Normal"):
        component_labels, n_components = label(type_masks[type_name])
        components = [
            (component_id, int((component_labels == component_id).sum()))
            for component_id in range(1, n_components + 1)
        ]
        components.sort(key=lambda item: -item[1])
        for component_id, size_px in components:
            component = component_labels == component_id
            distance_inside = distance_transform_edt(component)
            max_y, max_x = np.unravel_index(
                int(distance_inside.argmax()), distance_inside.shape
            )
            region_name = f"{type_name}-{display_id}"
            region_id[component] = global_id
            regions.append(
                {
                    "region_id": global_id,
                    "name": region_name,
                    "type": type_name,
                    "area_mm2": size_px * RESOLUTION_UM**2 / 1e6,
                    "representative_x_um": max_x * RESOLUTION_UM + x_min_um,
                    "representative_y_um": max_y * RESOLUTION_UM + y_min_um,
                }
            )
            global_id += 1
            display_id += 1

    tumor_mask = type_masks["Tumor"]
    normal_mask = type_masks["Normal"]
    sdf_um = signed_distance_um(tumor_mask)
    if tumor_mask.any():
        nearest_indices = distance_transform_edt(
            ~tumor_mask, return_indices=True, return_distances=False
        )
        nearest_tumor_region_id = region_id[
            nearest_indices[0], nearest_indices[1]
        ].astype(np.int32)
    else:
        nearest_tumor_region_id = np.full((ny, nx), -1, dtype=np.int32)

    tumor_interior = binary_erosion(tumor_mask, structure=disk(1))
    tumor_boundary = tumor_mask & ~tumor_interior
    normal_outside = normal_mask & ~tumor_mask
    interface = tumor_boundary & binary_dilation(
        normal_outside, structure=disk(2)
    )
    interface_pixels = int(interface.sum())
    if interface_pixels:
        distance_from_interface = (
            distance_transform_edt(~interface) * RESOLUTION_UM
        )
        interface_sdf_um = np.where(
            tumor_mask, -distance_from_interface, distance_from_interface
        ).astype(np.float32)
    else:
        interface_sdf_um = np.full((ny, nx), np.nan, dtype=np.float32)

    x_centers_um = x_min_um + (np.arange(nx) + 0.5) * RESOLUTION_UM
    y_centers_um = y_min_um + (np.arange(ny) + 0.5) * RESOLUTION_UM
    inside_he_x = (x_centers_um >= 0.0) & (
        x_centers_um < he_size_x_px * he_mpp_um
    )
    inside_he_y = (y_centers_um >= 0.0) & (
        y_centers_um < he_size_y_px * he_mpp_um
    )
    inside_he = inside_he_y[:, None] & inside_he_x[None, :]
    label_valid = tissue_mask & inside_he
    interface_valid = label_valid & np.isfinite(interface_sdf_um)

    arrays = {
        "tumor_mask": tumor_mask.astype(np.uint8),
        "normal_mask": normal_mask.astype(np.uint8),
        "tissue_mask": tissue_mask.astype(np.uint8),
        "label_valid": label_valid.astype(np.uint8),
        "sdf_um": sdf_um.astype(np.float32),
        "interface_sdf_um": interface_sdf_um.astype(np.float32),
        "interface_valid": interface_valid.astype(np.uint8),
        "tumor_region_id": region_id,
        "nearest_tumor_region_id": nearest_tumor_region_id,
    }
    metadata = {
        "axes": "YX",
        "shape_yx": [ny, nx],
        "x_min_um": x_min_um,
        "y_min_um": y_min_um,
        "x_max_nominal_um": x_min_um + nx * RESOLUTION_UM,
        "y_max_nominal_um": y_min_um + ny * RESOLUTION_UM,
        "resolution_um": RESOLUTION_UM,
        "origin_semantics": "upper_left_pixel_edge",
        "cell_to_grid": "floor((coordinate_um-origin_um)/10)",
        "sdf_sign": "positive_outside_negative_inside",
        "n_joined_cells": len(cells),
        "n_tumor_seed_cells": int(tumor_seed.sum()),
        "n_normal_seed_cells": int((~tumor_seed).sum()),
        "tumor_pixels": int(tumor_mask.sum()),
        "normal_evidence_pixels": int(normal_mask.sum()),
        "tissue_pixels": int(tissue_mask.sum()),
        "valid_pixels": int(label_valid.sum()),
        "interface_pixels": interface_pixels,
        "interface_empty": interface_pixels == 0,
        "no_tumor": not bool(tumor_mask.any()),
        "regions": regions,
        "density_maxima": density_maxima,
        "effective_parameters": {
            "sigma_region_px": sigma_region,
            "closing_radius_px": closing_radius,
            "minimum_region_area_px": minimum_region_area,
            "relative_density_threshold": density_threshold,
            "sigma_tissue_px": SIGMA_TISSUE,
            "tissue_density_threshold": TISSUE_THRESHOLD,
            "opening_radius_px": OPENING_RADIUS,
            "minimum_normal_area_px": MIN_NORMAL_AREA,
            "maximum_hole_area_px": MAX_HOLE_AREA,
        },
    }
    return arrays, metadata


def verify_reference(
    cells: pd.DataFrame,
    arrays: dict[str, np.ndarray],
    metadata: dict[str, Any],
) -> dict[str, Any]:
    if "side" not in cells or "dist_sdf_um" not in cells:
        raise ValueError("reference verification columns were not loaded")
    ix, iy = grid_indices(
        cells["x_px"].to_numpy(dtype=np.float64) * cells.attrs["he_mpp_um"],
        cells["y_px"].to_numpy(dtype=np.float64) * cells.attrs["he_mpp_um"],
        arrays["tumor_mask"].shape[0],
        arrays["tumor_mask"].shape[1],
        metadata["x_min_um"],
        metadata["y_min_um"],
    )
    predicted_side = np.where(
        arrays["tumor_mask"][iy, ix].astype(bool), "Tumor", "Normal"
    )
    expected_side = cells["side"].astype(str).to_numpy()
    side_mismatch = int((predicted_side != expected_side).sum())

    predicted_sdf = np.round(arrays["sdf_um"][iy, ix].astype(np.float64), 1)
    expected_sdf = cells["dist_sdf_um"].to_numpy(dtype=np.float64)
    finite = np.isfinite(predicted_sdf) & np.isfinite(expected_sdf)
    same_nan = np.isnan(predicted_sdf) & np.isnan(expected_sdf)
    incompatible_nonfinite = ~(finite | same_nan)
    abs_error_um = np.abs(predicted_sdf[finite] - expected_sdf[finite])


    rounding_tolerance_um = 0.1000001
    one_label_pixel_tolerance_um = RESOLUTION_UM + 1e-6
    sdf_rounding_difference_count = int(
        incompatible_nonfinite.sum() + (abs_error_um > 0.05).sum()
    )
    sdf_mismatch = int(
        incompatible_nonfinite.sum()
        + (abs_error_um > rounding_tolerance_um).sum()
    )
    sdf_over_one_label_pixel_count = int(
        incompatible_nonfinite.sum()
        + (abs_error_um > one_label_pixel_tolerance_um).sum()
    )
    max_abs_error_um = (
        float(abs_error_um.max())
        if finite.any()
        else None
    )
    sdf_mismatch_fraction = sdf_mismatch / max(len(cells), 1)
    within_roundtrip_contract = (
        side_mismatch == 0
        and sdf_mismatch_fraction <= 1e-4
        and sdf_over_one_label_pixel_count == 0
    )
    if not within_roundtrip_contract:
        raise ValueError(
            "molecular reference reproduction mismatch: "
            f"side={side_mismatch}, sdf_gt_0.1um={sdf_mismatch}, "
            f"sdf_gt_10um={sdf_over_one_label_pixel_count}, "
            f"max_abs_error_um={max_abs_error_um}"
        )
    return {
        "status": "CELL_SIDE_EXACT_SDF_WITHIN_ONE_LABEL_PIXEL",
        "n_cells": len(cells),
        "side_mismatch": side_mismatch,
        "sdf_rounding_difference_count": sdf_rounding_difference_count,
        "sdf_mismatch": sdf_mismatch,
        "sdf_mismatch_fraction": sdf_mismatch_fraction,
        "sdf_over_one_label_pixel_count": sdf_over_one_label_pixel_count,
        "max_abs_error_um": max_abs_error_um,
        "rounding_tolerance_um": rounding_tolerance_um,
        "one_label_pixel_tolerance_um": one_label_pixel_tolerance_um,
        "roundtrip_note": (
            "reference SDF values are rounded to 0.1um; EDT replay can differ by "
            "0.1um at decimal ties. x_px was originally derived from "
            "x_um/0.2125, so multiplying back can also move coordinates exactly "
            "on a 10um grid edge by one label pixel. Cell-side identity remains "
            "the mask-equivalence gate."
        ),
    }


def load_joined_cells(
    row: pd.Series, verify: bool
) -> tuple[pd.DataFrame, dict[str, int]]:
    cell_path = Path(row["cell_table_path"])
    boundary_path = Path(row["boundary_reference_csv"])
    cell_table = pd.read_parquet(
        cell_path, columns=["cell_id", "x_px", "y_px", "cell_type"]
    )
    cell_table["cell_id"] = cell_table["cell_id"].astype(str)
    reference_columns = ["cell_id", "cell_type"]
    if verify:
        reference_columns.extend(["side", "dist_sdf_um"])
    reference = pd.read_csv(
        boundary_path,
        usecols=reference_columns,
        dtype={"cell_id": str, "cell_type": str},
        keep_default_na=False,
    )
    if cell_table["cell_id"].duplicated().any():
        raise ValueError(f"{row['sample']}: duplicate cell IDs in coordinate table")
    if reference["cell_id"].duplicated().any():
        raise ValueError(f"{row['sample']}: duplicate cell IDs in molecular reference")

    joined = cell_table.merge(
        reference,
        on="cell_id",
        how="inner",
        validate="one_to_one",
        suffixes=("_a4", ""),
        sort=False,
    )
    if len(joined) != len(cell_table) or len(joined) != len(reference):
        raise ValueError(
            f"{row['sample']}: incomplete join "
            f"coords={len(cell_table)}, molecular={len(reference)}, joined={len(joined)}"
        )
    a4_type = joined["cell_type_a4"].astype(str)
    gen11_type = joined["cell_type"].astype(str)
    transfer_audit = {
        "full_cell_type_disagreement": int((a4_type != gen11_type).sum()),
        "tumor_seed_disagreement": int(
            (
                a4_type.str.startswith("Tumor")
                != gen11_type.str.startswith("Tumor")
            ).sum()
        ),
    }
    joined.attrs["he_mpp_um"] = float(row["he_mpp_um"])
    return joined, transfer_audit


def export_one(
    row: pd.Series,
    config: dict[str, Any],
    contract: dict[str, Any],
    verify: bool,
    overwrite: bool,
    resume: bool = False,
) -> Path:
    sample = str(row["sample"])
    output_path = Path(row["label_path"])
    if output_path.exists() and resume:
        with np.load(output_path, allow_pickle=False) as archive:
            existing = json.loads(str(archive["metadata_json"]))
        expected_algorithm_sha = contract["sources"]["gen11_boundary_algorithm"][
            "sha256"
        ]
        checks = {
            "sample": existing.get("sample") == sample,
            "patient": existing.get("patient") == str(row["patient"]),
            "fold": int(existing.get("fold", -1)) == int(row["fold"]),
            "cell_source": existing.get("sources", {})
            .get("cell_coordinates", {})
            .get("sha256")
            == str(row["cell_table_sha256"]),
            "he_source": existing.get("he", {}).get("sha256")
            == str(row["he_sha256"]),
            "algorithm": existing.get("sources", {})
            .get("gen11_algorithm", {})
            .get("sha256")
            == expected_algorithm_sha,
            "reference_verification": existing.get("reference_verification", {}).get(
                "status"
            )
            == "CELL_SIDE_EXACT_SDF_WITHIN_ONE_LABEL_PIXEL",
        }
        if not all(checks.values()):
            failed = sorted(key for key, value in checks.items() if not value)
            raise ValueError(
                f"{output_path}: existing label cannot be resumed; failed {failed}"
            )
        print(f"RESUME-SKIP {sample}: {output_path}", flush=True)
        return output_path
    if output_path.exists() and not overwrite:
        raise FileExistsError(
            f"{output_path} exists; pass --resume to verify/skip or --overwrite"
        )

    cell_path = Path(row["cell_table_path"])
    boundary_path = Path(row["boundary_reference_csv"])
    observed_cell_sha = sha256_file(cell_path)
    if observed_cell_sha != row["cell_table_sha256"]:
        raise ValueError(f"{sample}: cell-table SHA-256 differs from contract")


    cells, transfer_audit = load_corrected_cells(row, verify=verify)
    arrays, metadata = build_dense_labels(
        cells=cells,
        cancer=str(row["cancer"]),
        he_size_y_px=int(row["he_size_y_px"]),
        he_size_x_px=int(row["he_size_x_px"]),
        he_mpp_um=float(row["he_mpp_um"]),
    )
    verification = (
        verify_reference(cells, arrays, metadata)
        if verify
        else {"status": "NOT_REQUESTED"}
    )


    logged = str(row.get("explicit_interface_pixels", "")).strip()
    has_logged = logged not in ("", "nan", "NaN", "None")
    metadata["gen11_log_interface_pixels"] = int(float(logged)) if has_logged else None
    metadata["interface_pixel_delta_vs_gen11_log"] = (
        int(metadata["interface_pixels"]) - int(float(logged)) if has_logged else None
    )

    metadata.update(
        {
            "schema_version": "he_boundary_prediction.v1.boundary_label.v1",
            "dataset": "5k",
            "cancer": str(row["cancer"]),
            "sample": sample,
            "patient": str(row["patient"]),
            "fold": int(row["fold"]),
            "he": {
                "path": str(row["he_path"]),
                "shape_yx": [
                    int(row["he_size_y_px"]),
                    int(row["he_size_x_px"]),
                ],
                "mpp_um": float(row["he_mpp_um"]),
                "sha256": str(row["he_sha256"]),
                "coordinate_relation": (
                    "aligned H&E already occupies the Xenium canvas; "
                    "x_edge_px=x_um/mpp,y_edge_px=y_um/mpp"
                ),
            },
            "sources": {
                "cell_coordinates": {
                    "path": str(cell_path),
                    "sha256": observed_cell_sha,
                },
                "gen11_cell_type_and_reference": {
                    "path": str(boundary_path),
                    "sha256": sha256_file(boundary_path),
                },
                "gen11_algorithm": contract["sources"][
                    "gen11_boundary_algorithm"
                ],
            },
            "join": {
                "key": "cell_id",
                "n_coordinate_rows": len(cells),
                "n_gen11_rows": len(cells),
                "n_joined_rows": len(cells),
                "duplicates": 0,
                "dropped": 0,
            },
            "label_transfer_audit": transfer_audit,
            "reference_verification": verification,
            "claim_boundary": config["task"]["claim_boundary"],
        }
    )
    metadata_json = json.dumps(
        metadata, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        dir=output_path.parent, suffix=".npz", delete=False
    ) as handle:
        temporary_path = Path(handle.name)
    try:
        np.savez_compressed(
            temporary_path,
            **arrays,
            metadata_json=np.asarray(metadata_json),
        )
        os.replace(temporary_path, output_path)
    finally:
        temporary_path.unlink(missing_ok=True)
    print(
        f"EXPORTED {sample}: shape={arrays['tumor_mask'].shape}, "
        f"tumor_px={metadata['tumor_pixels']}, "
        f"interface_px={metadata['interface_pixels']} -> {output_path}",
        flush=True,
    )
    return output_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "configs" / "experiment.json",
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--sample", action="append")
    group.add_argument("--all", action="store_true")
    parser.add_argument(
        "--verify-reference",
        action="store_true",
        help=(
            "Require exact per-cell side and bounded dist_sdf_um replay "
            "agreement with the molecular reference."
        ),
    )
    existing = parser.add_mutually_exclusive_group()
    existing.add_argument("--resume", action="store_true")
    existing.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config_path = args.config.resolve()
    project_root = config_path.parent.parent
    config = json.loads(config_path.read_text(encoding="utf-8"))
    manifest_path = project_root / config["outputs"]["source_manifest"]
    contract_path = project_root / config["outputs"]["data_contract"]
    if not manifest_path.exists() or not contract_path.exists():
        raise FileNotFoundError(
            "build the source data contract before exporting labels"
        )
    manifest = pd.read_csv(manifest_path)
    contract = json.loads(contract_path.read_text(encoding="utf-8"))
    if not str(contract.get("status", "")).startswith("VALIDATED"):
        raise ValueError("source data contract is not validated")

    if args.all:
        selected = manifest
    else:
        requested = set(args.sample)
        selected = manifest[manifest["sample"].isin(requested)]
        missing = requested.difference(selected["sample"])
        if missing:
            raise ValueError(f"unknown samples: {sorted(missing)}")
    for _, row in selected.sort_values(["fold", "cancer", "sample"]).iterrows():
        export_one(
            row=row,
            config=config,
            contract=contract,
            verify=args.verify_reference,
            overwrite=args.overwrite,
            resume=args.resume,
        )


if __name__ == "__main__":
    main()
