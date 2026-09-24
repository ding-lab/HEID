#!/usr/bin/env python3

from __future__ import annotations
import os

import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from scipy import ndimage as ndi
from skimage.morphology import disk

from cohort_binding import load_contract


MODULE_ROOT = Path(__file__).resolve().parents[1]
HE_PRED = Path(os.environ.get("TLS_HE_PRED_DATA", str(Path(os.environ.get("PROJECTS_ROOT", "/data/heid")) / "tls/prediction")))
TLS = HE_PRED.parent
BRIDGE = HE_PRED / "bridge"
FEATURES = Path(os.environ.get("TLS_FEATURES", str(HE_PRED / "features")))
BRIDGE_SLIDES = BRIDGE / "configs" / "slides_5k.tsv"
DUPLICATE_LABELS = BRIDGE / "data" / "duplicate_scan_labels"
ALIGNMENT_MANIFEST = MODULE_ROOT / "outputs" / "results" / "alignment_manifest.tsv"
TLS_QC = MODULE_ROOT / "outputs" / "results" / "region_qc.tsv"
CONFIG = MODULE_ROOT / "configs" / "data_contract.json"
OUT_CONFIG = MODULE_ROOT / "configs" / "slides_5k.tsv"
OUT_LABELS = Path(os.environ.get("TLS_LABELS", str(MODULE_ROOT / "data/labels")))
OUT_INSTANCES = MODULE_ROOT / "data" / "tls_instances.parquet"
OUT_CALIBRATION = MODULE_ROOT / "outputs" / "results" / "geometry_calibration.tsv"
OUT_SUMMARY = MODULE_ROOT / "outputs" / "results" / "training_data.json"

UM_PER_PX = 0.2125
GRID_UM = 10.0
REGION_POS = 0.20
REGION_NEG = 0.05
KEEP_MIN = 0.50
DROP_CANCERS = {"BRCA", "CHOL"}
EXPLICIT_MOLECULAR_LABELS = {"tls_positive", "tls_negative"}


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def mask_from_cells(gy: np.ndarray, gx: np.ndarray, ny: int, nx: int) -> np.ndarray:
    mask = np.zeros((ny, nx), dtype=bool)
    if len(gy):
        mask[gy, gx] = True
        mask = ndi.binary_fill_holes(
            ndi.binary_closing(mask, np.ones((3, 3), dtype=bool))
        )
    return mask


def integral_sum(arr: np.ndarray, x0: np.ndarray, y0: np.ndarray, fpx: int) -> np.ndarray:
    ny, nx = arr.shape
    cs = np.zeros((ny + 1, nx + 1), dtype=np.int64)
    cs[1:, 1:] = np.cumsum(np.cumsum(arr.astype(np.int64), axis=0), axis=1)
    gx0 = np.clip(np.floor(x0 * UM_PER_PX / GRID_UM).astype(np.int64), 0, nx)
    gx1 = np.clip(np.ceil((x0 + fpx) * UM_PER_PX / GRID_UM).astype(np.int64), 0, nx)
    gy0 = np.clip(np.floor(y0 * UM_PER_PX / GRID_UM).astype(np.int64), 0, ny)
    gy1 = np.clip(np.ceil((y0 + fpx) * UM_PER_PX / GRID_UM).astype(np.int64), 0, ny)
    return cs[gy1, gx1] - cs[gy0, gx1] - cs[gy1, gx0] + cs[gy0, gx0]


def fraction_over_tiles(
    mask: np.ndarray, x0: np.ndarray, y0: np.ndarray, fpx: int
) -> np.ndarray:
    covered = integral_sum(mask, x0, y0, fpx)
    gx0 = np.floor(x0 * UM_PER_PX / GRID_UM).astype(np.int64)
    gx1 = np.ceil((x0 + fpx) * UM_PER_PX / GRID_UM).astype(np.int64)
    gy0 = np.floor(y0 * UM_PER_PX / GRID_UM).astype(np.int64)
    gy1 = np.ceil((y0 + fpx) * UM_PER_PX / GRID_UM).astype(np.int64)
    denom = np.maximum((gx1 - gx0) * (gy1 - gy0), 1)
    return covered / denom


def parquet_sample_index(path: Path) -> tuple[pq.ParquetFile, dict[str, list[int]]]:
    pf = pq.ParquetFile(path)
    sample_col = pf.schema.names.index("sample")
    index: dict[str, list[int]] = {}
    for row_group in range(pf.num_row_groups):
        stats = pf.metadata.row_group(row_group).column(sample_col).statistics
        if stats is None or stats.min != stats.max:
            raise RuntimeError(
                f"{path}: row group {row_group} is not sample-partitioned"
            )
        index.setdefault(str(stats.min), []).append(row_group)
    return pf, index


def read_sample(
    pf: pq.ParquetFile,
    index: dict[str, list[int]],
    sample: str,
) -> pd.DataFrame:
    groups = index.get(sample, [])
    if not groups:
        return pd.DataFrame()
    columns = [
        "sample",
        "cell_id_base",
        "tls_region_halo150",
        "in_tls_core",
    ]
    out = pf.read_row_groups(groups, columns=columns).to_pandas()
    return out.loc[out["sample"].eq(sample)].copy()


def main() -> None:
    config = load_contract(CONFIG)
    bridge = {
        key: (MODULE_ROOT / item["path"]).resolve()
        for key, item in config["gen12_bridge"].items()
    }
    for key, path in bridge.items():
        observed = sha256(path)
        expected = config["gen12_bridge"][key]["sha256"]
        if observed != expected:
            raise RuntimeError(
                f"source drift for {key}: expected {expected}, observed {observed}"
            )

    v4 = pd.read_csv(BRIDGE_SLIDES, sep="\t")
    manifest = pd.read_csv(ALIGNMENT_MANIFEST, sep="\t")
    tls_qc = pd.read_csv(TLS_QC, sep="\t")
    current = manifest.loc[
        manifest["cohort"].eq("5k") & manifest["model_scope_usable"],
        ["field", "directory_name", "cell_keep_path", "he_image_path"],
    ].copy()
    if current["field"].duplicated().any():
        raise RuntimeError("aligned-cell manifest is not unique by field")

    cohort = v4.merge(current, on="field", how="inner", validate="many_to_one")
    cohort = cohort.loc[
        cohort["has_feat_uni2_b"].astype(bool)
        & ~cohort["cancer"].isin(DROP_CANCERS)
        & cohort["gen12_class"].isin(EXPLICIT_MOLECULAR_LABELS)
    ].copy()
    cohort = cohort.sort_values(["cancer", "unit_id"]).reset_index(drop=True)
    if len(cohort) == 0:
        raise RuntimeError("TLS cohort is empty")

    from group_contract import build_coarse_groups

    group_input = cohort[
        ["unit_id", "xenium_run_folder", "patient"]
    ].rename(
        columns={
            "unit_id": "sample",
            "xenium_run_folder": "xenium_run",
            "patient": "patient_stem",
        }
    )
    cohort["cv_group"] = build_coarse_groups(group_input)
    cohort["v5_label_path"] = cohort["unit_id"].map(
        lambda unit: str((OUT_LABELS / f"{unit}_tiles.parquet").resolve())
    )
    cohort.to_csv(OUT_CONFIG, sep="\t", index=False)

    per_cell_pf, per_cell_index = parquet_sample_index(bridge["per_cell"])
    per_tls = pd.read_parquet(bridge["per_tls"])
    per_tls_by_key = per_tls.set_index(["sample", "tls_region"])
    usable = {
        (str(row.sample), str(row.tls_region))
        for row in tls_qc.loc[tls_qc["region_qc_usable"]].itertuples()
    }
    rejected = {
        (str(row.sample), str(row.tls_region)): str(row.region_qc_status)
        for row in tls_qc.loc[
            tls_qc["slide_region_usable"] & ~tls_qc["region_qc_usable"]
        ].itertuples()
    }

    OUT_LABELS.mkdir(parents=True, exist_ok=True)
    summaries: list[dict] = []
    instance_rows: list[dict] = []
    source_rows: list[dict] = []
    calibration_rows: list[dict] = []

    for row in cohort.itertuples(index=False):
        unit = str(row.unit_id)
        sample = (
            str(row.gen12_sample)
            if isinstance(row.gen12_sample, str) and row.gen12_sample
            else ""
        )
        old_path = DUPLICATE_LABELS / f"{unit}_tiles.parquet"
        if not old_path.is_file():
            raise FileNotFoundError(f"audited ownership grid missing: {old_path}")
        tiles = pd.read_parquet(
            old_path,
            columns=["x_px", "y_px", "field_px", "keep_frac"],
        )
        if tiles.duplicated(["x_px", "y_px"]).any():
            raise RuntimeError(f"{unit}: duplicate tile coordinates")
        x0 = tiles["x_px"].to_numpy(dtype=np.int64)
        y0 = tiles["y_px"].to_numpy(dtype=np.int64)
        fpx_values = tiles["field_px"].dropna().unique()
        if len(fpx_values) != 1:
            raise RuntimeError(f"{unit}: non-constant field_px")
        fpx = int(fpx_values[0])

        cell_keep_path = Path(str(row.cell_keep_path))
        cell_keep = pd.read_parquet(
            cell_keep_path,
            columns=["cell_id", "x_um", "y_um", "dx_um", "dy_um", "keep"],
        )
        if cell_keep["cell_id"].duplicated().any():
            raise RuntimeError(f"{unit}: duplicate cell_id in current cell_keep")
        he_px = (
            cell_keep["x_um"].to_numpy()
            + np.nan_to_num(cell_keep["dx_um"].to_numpy())
        ) / UM_PER_PX
        he_py = (
            cell_keep["y_um"].to_numpy()
            + np.nan_to_num(cell_keep["dy_um"].to_numpy())
        ) / UM_PER_PX
        ny = int(max(float(he_py.max()), float((y0 + fpx).max())) * UM_PER_PX / GRID_UM) + 2
        nx = int(max(float(he_px.max()), float((x0 + fpx).max())) * UM_PER_PX / GRID_UM) + 2
        gx = np.clip(
            np.floor(he_px * UM_PER_PX / GRID_UM).astype(np.int64), 0, nx - 1
        )
        gy = np.clip(
            np.floor(he_py * UM_PER_PX / GRID_UM).astype(np.int64), 0, ny - 1
        )


        if str(row.split_key) == "roster":
            keep_frac = tiles["keep_frac"].to_numpy(dtype=np.float32)
            n_cells_tile = np.where(np.isfinite(keep_frac), 1, 0).astype(np.int32)
            keep_source = "v4_audited_duplicate_ownership"
        else:
            count_all = np.zeros((ny, nx), dtype=np.int32)
            count_keep = np.zeros((ny, nx), dtype=np.int32)
            np.add.at(count_all, (gy, gx), 1)
            keep_bool = cell_keep["keep"].to_numpy(dtype=bool)
            np.add.at(count_keep, (gy[keep_bool], gx[keep_bool]), 1)
            n_all = integral_sum(count_all, x0, y0, fpx)
            n_keep = integral_sum(count_keep, x0, y0, fpx)
            n_cells_tile = n_all.astype(np.int32)
            keep_frac = np.where(
                n_all > 0, n_keep / np.maximum(n_all, 1), np.nan
            ).astype(np.float32)
            keep_source = "aligned_he_v9_cell_keep"

        region_union = np.zeros((ny, nx), dtype=bool)
        rejected_union = np.zeros((ny, nx), dtype=bool)
        pc = read_sample(per_cell_pf, per_cell_index, sample) if sample else pd.DataFrame()
        if len(pc):
            joined = pc.merge(
                pd.DataFrame(
                    {
                        "cell_id": cell_keep["cell_id"].astype(str),
                        "px": he_px,
                        "py": he_py,
                        "keep": cell_keep["keep"].to_numpy(dtype=bool),
                    }
                ),
                left_on="cell_id_base",
                right_on="cell_id",
                how="left",
                validate="many_to_one",
            )


            all_raw = np.zeros((ny, nx), dtype=np.float32)
            np.add.at(all_raw, (gy, gx), 1.0)
            tissue = ndi.binary_fill_holes(
                ndi.binary_closing(
                    ndi.gaussian_filter(all_raw, sigma=10) > 0.015,
                    structure=disk(10),
                )
            )
            claimed = np.zeros((ny, nx), dtype=bool)
            region_groups = {
                str(name): cells
                for name, cells in joined.loc[
                    joined["tls_region_halo150"].ne("Outside")
                ].groupby("tls_region_halo150", sort=False)
            }
            for tls_region in sorted(
                region_groups,
                key=lambda value: int(value.rsplit("-", 1)[-1]),
            ):
                cells = region_groups[tls_region]
                key = (sample, str(tls_region))
                use = key in usable
                reject = key in rejected
                if not use and not reject:
                    continue
                located = cells["px"].notna() & cells["py"].notna()
                cx = np.clip(
                    np.floor(cells.loc[located, "px"].to_numpy() * UM_PER_PX / GRID_UM)
                    .astype(np.int64),
                    0,
                    nx - 1,
                )
                cy = np.clip(
                    np.floor(cells.loc[located, "py"].to_numpy() * UM_PER_PX / GRID_UM)
                    .astype(np.int64),
                    0,
                    ny - 1,
                )
                core_located = located & cells["in_tls_core"].astype(bool)
                core_x = np.clip(
                    np.floor(
                        cells.loc[core_located, "px"].to_numpy() * UM_PER_PX / GRID_UM
                    ).astype(np.int64),
                    0,
                    nx - 1,
                )
                core_y = np.clip(
                    np.floor(
                        cells.loc[core_located, "py"].to_numpy() * UM_PER_PX / GRID_UM
                    ).astype(np.int64),
                    0,
                    ny - 1,
                )
                target_area = float(
                    per_tls_by_key.loc[key, "region_area_mm2_halo150"]
                )
                target_pixels = max(
                    1, int(round(target_area * 1e6 / (GRID_UM * GRID_UM)))
                )
                member_mask = mask_from_cells(cy, cx, ny, nx)
                core_mask = mask_from_cells(core_y, core_x, ny, nx)
                seed_mask = core_mask | member_mask


                allowed = tissue & ~claimed
                candidate_rows = np.flatnonzero(allowed.ravel())
                distance = ndi.distance_transform_edt(~seed_mask).ravel()
                n_select = min(target_pixels, len(candidate_rows))
                if n_select == 0:
                    instance_mask = np.zeros((ny, nx), dtype=bool)
                else:
                    order = np.argpartition(distance[candidate_rows], n_select - 1)[
                        :n_select
                    ]
                    selected = candidate_rows[order]
                    instance_mask = np.zeros((ny, nx), dtype=bool)
                    instance_mask.ravel()[selected] = True
                claimed |= instance_mask
                fractions = fraction_over_tiles(instance_mask, x0, y0, fpx)
                raster_area = float(instance_mask.sum() * GRID_UM * GRID_UM / 1e6)
                calibration_rows.append(
                    {
                        "unit_id": unit,
                        "sample": sample,
                        "tls_region": str(tls_region),
                        "region_qc_status": "usable" if use else rejected[key],
                        "n_located_members": int(located.sum()),
                        "n_core_members": int(core_located.sum()),
                        "target_area_mm2": target_area,
                        "raster_area_mm2": raster_area,
                        "area_ratio": raster_area / target_area if target_area > 0 else np.nan,
                    }
                )
                if use:
                    region_union |= instance_mask
                    status = "usable"
                else:
                    rejected_union |= instance_mask
                    status = rejected[key]
                for tile_i in np.flatnonzero(fractions > 0):
                    instance_rows.append(
                        {
                            "unit_id": unit,
                            "field": str(row.field),
                            "sample": sample,
                            "cancer": str(row.cancer),
                            "tls_region": str(tls_region),
                            "region_qc_status": status,
                            "region_qc_usable": bool(use),
                            "x_px": int(x0[tile_i]),
                            "y_px": int(y0[tile_i]),
                            "region_frac": float(fractions[tile_i]),
                        }
                    )

        region_frac = fraction_over_tiles(region_union, x0, y0, fpx)
        rejected_frac = fraction_over_tiles(rejected_union, x0, y0, fpx)
        label = np.where(
            region_frac >= REGION_POS,
            1,
            np.where(region_frac <= REGION_NEG, 0, -1),
        ).astype(np.int8)
        ignore_local = ~np.isfinite(keep_frac) | (keep_frac < KEEP_MIN)
        ignore_rejected = rejected_frac > 0
        label[ignore_local | ignore_rejected] = -1

        out = pd.DataFrame(
            {
                "unit_id": unit,
                "field": str(row.field),
                "cancer": str(row.cancer),
                "cv_group": str(row.cv_group),
                "x_px": x0,
                "y_px": y0,
                "field_px": fpx,
                "region_frac": region_frac.astype(np.float32),
                "rejected_tls_frac": rejected_frac.astype(np.float32),
                "n_cells_tile": n_cells_tile,
                "keep_frac": keep_frac,
                "label": label,
                "label_source": (
                    "gen12_region_qc_v5" if sample else "not_in_gen12_negative_v5"
                ),
            }
        )
        out_path = OUT_LABELS / f"{unit}_tiles.parquet"
        out.to_parquet(out_path, index=False)
        source_rows.append(
            {
                "unit_id": unit,
                "cell_keep_path": str(cell_keep_path),
                "cell_keep_sha256": sha256(cell_keep_path),
                "ownership_grid_path": str(old_path),
                "ownership_grid_sha256": sha256(old_path),
            }
        )
        summaries.append(
            {
                "unit_id": unit,
                "field": str(row.field),
                "cancer": str(row.cancer),
                "cv_group": str(row.cv_group),
                "gen12_class": str(row.gen12_class),
                "n_tiles": int(len(out)),
                "n_positive": int((label == 1).sum()),
                "n_negative": int((label == 0).sum()),
                "n_ignore": int((label == -1).sum()),
                "n_ignore_local_qc": int(ignore_local.sum()),
                "n_ignore_rejected_tls": int(ignore_rejected.sum()),
                "keep_fraction_source": keep_source,
                "label_sha256": sha256(out_path),
            }
        )
        print(
            f"[{len(summaries):3d}/{len(cohort)}] {unit}: "
            f"pos={summaries[-1]['n_positive']} neg={summaries[-1]['n_negative']} "
            f"ignore={summaries[-1]['n_ignore']}",
            flush=True,
        )

    summary_df = pd.DataFrame(summaries)
    instance_df = pd.DataFrame(instance_rows)
    OUT_INSTANCES.parent.mkdir(parents=True, exist_ok=True)
    instance_df.to_parquet(OUT_INSTANCES, index=False)
    calibration_df = pd.DataFrame(calibration_rows)
    calibration_df.to_csv(OUT_CALIBRATION, sep="\t", index=False)
    source_df = pd.DataFrame(source_rows)
    source_path = MODULE_ROOT / "outputs" / "results" / "training_sources.tsv"
    source_df.to_csv(source_path, sep="\t", index=False)
    tile_summary_path = MODULE_ROOT / "outputs" / "results" / "tile_label_summary.tsv"
    summary_df.to_csv(tile_summary_path, sep="\t", index=False)

    result = {
        "contract_name": config["contract_name"],
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "selection": {
            "cohort": "5k",
            "requires_model_scope_usable": True,
            "requires_canonical_uni2_feature": True,
            "dropped_cancers": sorted(DROP_CANCERS),
        },
        "label_rule": {
            "region_positive_min": REGION_POS,
            "region_negative_max": REGION_NEG,
            "tile_keep_min": KEEP_MIN,
            "rejected_tls_overlap": "ignore_if_overlap_gt_0",
        },
        "observed": {
            "units": int(len(summary_df)),
            "cv_groups": int(summary_df["cv_group"].nunique()),
            "tiles": int(summary_df["n_tiles"].sum()),
            "positive_tiles": int(summary_df["n_positive"].sum()),
            "negative_tiles": int(summary_df["n_negative"].sum()),
            "ignore_tiles": int(summary_df["n_ignore"].sum()),
            "tiles_ignored_by_rejected_tls": int(
                summary_df["n_ignore_rejected_tls"].sum()
            ),
            "tls_instance_tile_rows": int(len(instance_df)),
            "usable_tls_instances_represented": int(
                instance_df.loc[instance_df["region_qc_usable"], ["sample", "tls_region"]]
                .drop_duplicates()
                .shape[0]
            )
            if len(instance_df)
            else 0,
            "usable_tls_geometry_area_ratio": {
                "minimum": float(
                    calibration_df.loc[
                        calibration_df["region_qc_status"].eq("usable"), "area_ratio"
                    ].min()
                ),
                "median": float(
                    calibration_df.loc[
                        calibration_df["region_qc_status"].eq("usable"), "area_ratio"
                    ].median()
                ),
                "mean": float(
                    calibration_df.loc[
                        calibration_df["region_qc_status"].eq("usable"), "area_ratio"
                    ].mean()
                ),
                "below_0_8": int(
                    (
                        calibration_df.loc[
                            calibration_df["region_qc_status"].eq("usable"), "area_ratio"
                        ]
                        < 0.8
                    ).sum()
                ),
            },
        },
        "sources": {
            "slides_v4_5k_sha256": sha256(BRIDGE_SLIDES),
            "aligned_he_v9_manifest_sha256": sha256(ALIGNMENT_MANIFEST),
            "gen12_tls_region_qc_sha256": sha256(TLS_QC),
            "gen12_per_cell_sha256": sha256(bridge["per_cell"]),
            "feature_root": str(FEATURES),
        },
        "artifacts": {
            "slides_v5_5k": str(OUT_CONFIG),
            "labels": str(OUT_LABELS),
            "tls_instances_v5": str(OUT_INSTANCES),
            "tls_geometry_calibration": str(OUT_CALIBRATION),
            "source_manifest": str(source_path),
            "tile_summary": str(tile_summary_path),
        },
    }
    OUT_SUMMARY.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result["observed"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
