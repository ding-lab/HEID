from __future__ import annotations
import os
PROJECTS_ROOT = os.environ.get("PROJECTS_ROOT", "/data/heid")

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.ndimage import (
    binary_closing,
    binary_fill_holes,
    binary_opening,
    gaussian_filter,
    label,
)

REPO_ROOT = Path(PROJECTS_ROOT) / "tb" / "boundary"
INPUTS_ROOT = REPO_ROOT / "inputs"
DATA_ROOT = INPUTS_ROOT / "xenium"
CELLATLAS_477 = INPUTS_ROOT / "cell_atlas" / "477" / "allsolidtumor.cells.csv"
CELLATLAS_5K = INPUTS_ROOT / "cell_atlas" / "5k" / "allsolidtumor5k.cells.csv"
K10_CACHE = REPO_ROOT / "cache" / "celltype_k10"

NORMAL_MAP = {
    "BRCA": ["Fibroblast"],
    "CHOL": ["Fibroblast", "Hepatocyte"],
    "CRC": ["Fibroblast", "SMC"],
    "GBM": ["Astrocyte", "Neuron", "Oligodendrocyte", "vSMC"],
    "HNSC": ["Fibroblast", "Basal_keratinocyte", "Squamous_cell", "Skeletal_muscle", "vSMC"],
    "PDAC": ["CAF", "SMC"],
    "PRAD": ["SMC", "Smooth_muscle"],
    "RCC": ["Fibroblast", "SMC"],
    "SKCM": ["Fibroblast", "Epidermal", "Secretory_Epithelium", "vSMC"],
    "LUNG": ["Fibroblast", "SMC"],
}

RESOLUTION = 10
SIGMA_REGION = 5
SIGMA_ECO = 10
CLOSING_R = 5
OPENING_R = 3
MIN_REGION_AREA = 500
MIN_NORMAL_AREA = 3000
MAX_HOLE_AREA = 50000

PARAMS_OVERRIDE = {
    "PDAC": {"SIGMA_REGION": 3, "CLOSING_R": 2, "MIN_REGION_AREA": 100, "DENSITY_THRESH": 0.03},
}

PX_MM2 = RESOLUTION ** 2 / 1e6
SAMPLE_LIST = Path(__file__).resolve().parent.parent / "samples_5k_all.tsv"
OUT_ROOT = REPO_ROOT / "outputs" / "5k"


def disk(radius: int) -> np.ndarray:
    L = np.arange(-radius, radius + 1)
    X, Y = np.meshgrid(L, L)
    return (X ** 2 + Y ** 2 <= radius ** 2).astype(np.uint8)


def rasterize(sub: pd.DataFrame, ny: int, nx: int, x_min: float, y_min: float) -> np.ndarray:
    grid = np.zeros((ny, nx), dtype=np.float32)
    ix = np.clip(((sub.x_centroid.values - x_min) / RESOLUTION).astype(int), 0, nx - 1)
    iy = np.clip(((sub.y_centroid.values - y_min) / RESOLUTION).astype(int), 0, ny - 1)
    np.add.at(grid, (iy, ix), 1)
    return grid


def fill_small_holes(mask: np.ndarray, max_hole_area: int) -> np.ndarray:
    filled = binary_fill_holes(mask)
    holes = filled & ~mask
    if not holes.any():
        return filled
    lb_h, n_h = label(holes)
    for idx in range(1, n_h + 1):
        if (lb_h == idx).sum() > max_hole_area:
            filled[lb_h == idx] = False
    return filled


def clean_mask(mask: np.ndarray, closing_r: int = CLOSING_R,
               min_region_area: int = MIN_REGION_AREA) -> np.ndarray:
    mask = binary_closing(mask, structure=disk(closing_r))
    mask = fill_small_holes(mask, MAX_HOLE_AREA)
    mask = binary_opening(mask, structure=disk(OPENING_R))
    lb, n = label(mask)
    for idx in range(1, n + 1):
        if (lb == idx).sum() < min_region_area:
            mask[lb == idx] = False
    return mask


def load_coords(dataset: str, cancer_type: str, sample: str) -> pd.DataFrame:
    xendir = DATA_ROOT / dataset / cancer_type / sample
    for path in [xendir / "cells.parquet", xendir / "cells.csv.gz", xendir / "cells.csv"]:
        if path.exists():
            if path.suffix == ".parquet":
                coords = pd.read_parquet(path, columns=["cell_id", "x_centroid", "y_centroid"])
            else:
                coords = pd.read_csv(path, usecols=["cell_id", "x_centroid", "y_centroid"])
            coords["cell_id"] = coords["cell_id"].astype(str)
            return coords
    raise FileNotFoundError(f"No cells file in {xendir}")


def load_cell_types(sample: str) -> pd.DataFrame:
    k10 = K10_CACHE / f"{sample}.csv"
    if k10.exists():
        return pd.read_csv(k10, usecols=["cell_id", "cell_type"], dtype=str,
                           keep_default_na=False)
    cache_5k = REPO_ROOT / "cache" / "cell_types_by_sample" / f"{sample}.csv"
    if cache_5k.exists():
        return pd.read_csv(cache_5k, usecols=["cell_id", "cell_type"], dtype=str,
                           keep_default_na=False)
    for atlas in (CELLATLAS_477, CELLATLAS_5K):
        if atlas.exists():
            df = pd.read_csv(atlas, usecols=["barcode", "cell_type", "dataset"],
                             dtype=str, keep_default_na=False)
            sub = df[df["dataset"] == sample].copy()
            if not sub.empty:
                sub["cell_id"] = sub["barcode"].str[:-len(f"_{sample}")]
                return sub[["cell_id", "cell_type"]].copy()
    raise FileNotFoundError(f"No cell type data for {sample}")


def tumor_mask_for(dataset: str, cancer_type: str, sample: str):
    coords = load_coords(dataset, cancer_type, sample)
    types_sub = load_cell_types(sample)
    df = coords.merge(types_sub, on="cell_id", how="inner")
    if df.empty:
        raise RuntimeError("no cells after merge")

    df["binary_type"] = "Normal"
    df.loc[df["cell_type"].astype(str).str.startswith("Tumor"), "binary_type"] = "Tumor"
    n_tumor = int((df.binary_type == "Tumor").sum())

    x_min, x_max = df.x_centroid.min() - 100, df.x_centroid.max() + 100
    y_min, y_max = df.y_centroid.min() - 100, df.y_centroid.max() + 100
    nx = int(np.ceil((x_max - x_min) / RESOLUTION))
    ny = int(np.ceil((y_max - y_min) / RESOLUTION))


    all_raw = rasterize(df, ny, nx, x_min, y_min)
    tissue_d = gaussian_filter(all_raw, sigma=SIGMA_ECO)
    tissue_mask = tissue_d > 0.015
    tissue_mask = binary_closing(tissue_mask, structure=disk(10))
    tissue_mask = binary_fill_holes(tissue_mask)


    ovr = PARAMS_OVERRIDE.get(cancer_type, {})
    sigma_r = ovr.get("SIGMA_REGION", SIGMA_REGION)
    closing_r = ovr.get("CLOSING_R", CLOSING_R)
    min_area = ovr.get("MIN_REGION_AREA", MIN_REGION_AREA)
    density_thresh = ovr.get("DENSITY_THRESH", 0.1)

    sub = df[df.binary_type == "Tumor"]
    if len(sub) < 10:
        tumor_mask = np.zeros((ny, nx), dtype=bool)
    else:
        raw = rasterize(sub, ny, nx, x_min, y_min)
        density = gaussian_filter(raw, sigma=sigma_r)
        if density.max() > 0:
            tumor_mask = clean_mask(density > density_thresh * density.max(),
                                    closing_r=closing_r, min_region_area=min_area)
        else:
            tumor_mask = np.zeros((ny, nx), dtype=bool)
    tumor_mask &= tissue_mask
    return tumor_mask, tissue_mask, n_tumor, int(len(df))


def sample_area(dataset: str, cancer_type: str, sample: str) -> dict:
    tumor_mask, tissue_mask, n_tumor, n_cells = tumor_mask_for(dataset, cancer_type, sample)
    lb_t, n_t = label(tumor_mask)
    region_px = [int((lb_t == di).sum()) for di in range(1, n_t + 1)]
    total_px = int(tumor_mask.sum())
    largest_px = max(region_px) if region_px else 0
    tissue_px = int(tissue_mask.sum())
    total_mm2 = total_px * PX_MM2
    tissue_mm2 = tissue_px * PX_MM2
    return {
        "sample": sample,
        "n_tumor_regions": n_t,
        "total_tumor_area_mm2": round(total_mm2, 6),
        "total_tumor_area_um2": round(total_px * float(RESOLUTION ** 2), 1),
        "largest_tumor_region_mm2": round(largest_px * PX_MM2, 6),
        "tissue_area_mm2": round(tissue_mm2, 6),
        "tumor_pct_of_tissue": round(100.0 * total_mm2 / tissue_mm2, 4) if tissue_mm2 > 0 else 0.0,
        "n_tumor_cells": n_tumor,
        "n_cells": n_cells,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Tumor area per section for one cancer type.")
    parser.add_argument("cancer_type", choices=sorted(NORMAL_MAP))
    parser.add_argument("--samples", type=Path, default=SAMPLE_LIST,
                        help="tab-separated roster without header: dataset, cancer_type, sample")
    args = parser.parse_args()
    cancer_type = args.cancer_type

    samples = pd.read_csv(args.samples, sep="\t", header=None,
                          names=["dataset", "cancer_type", "sample"], dtype=str)
    samples = samples[samples.cancer_type == cancer_type].reset_index(drop=True)
    print(f"{cancer_type}: {len(samples)} samples", flush=True)

    rows, failed = [], []
    for _, r in samples.iterrows():
        try:
            row = sample_area(r.dataset, cancer_type, r["sample"])
            rows.append(row)
            print(f"  {r['sample']}: {row['n_tumor_regions']} regions, "
                  f"{row['total_tumor_area_mm2']:.4f} mm^2 "
                  f"({row['tumor_pct_of_tissue']}% of tissue)", flush=True)
        except Exception as exc:
            failed.append((r["sample"], str(exc)))
            print(f"  WARN {r['sample']}: FAILED — {exc}", flush=True)

    out_dir = OUT_ROOT / cancer_type
    out_dir.mkdir(parents=True, exist_ok=True)
    out_csv = out_dir / f"{cancer_type}_tumor_area.csv"
    df = pd.DataFrame(rows).sort_values("total_tumor_area_mm2", ascending=False)
    df.to_csv(out_csv, index=False)
    print(f"\nWrote {out_csv} ({len(df)} samples; {len(failed)} failed)", flush=True)
    for s, e in failed:
        print(f"  FAILED {s}: {e}", flush=True)


if __name__ == "__main__":
    main()
