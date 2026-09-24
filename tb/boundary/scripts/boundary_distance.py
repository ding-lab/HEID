
from __future__ import annotations
import os
PROJECTS_ROOT = os.environ.get("PROJECTS_ROOT", "/data/heid")

import argparse
import time
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
from scipy.ndimage import (
    binary_closing,
    binary_dilation,
    binary_fill_holes,
    binary_opening,
    distance_transform_edt,
    gaussian_filter,
    label,
)
from skimage.measure import find_contours
from skimage.morphology import disk

matplotlib.use("Agg")
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.ticker import MultipleLocator
from mpl_toolkits.axes_grid1.anchored_artists import AnchoredSizeBar

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


GRAD_MAX_UM = 150.0
BAND_COLORS_OUT = ["#ff7f0e", "#ffbb78", "#ffe0b2"]
BAND_COLORS_IN  = ["#c0392b", "#e74c3c", "#f1948a"]

COLORS = {
    "Tumor": "#e74c3c",
    "Normal": "#2ecc71",
    "Other": "#d5d5d5",
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset", choices=["5k", "477"])
    parser.add_argument("cancer_type", choices=sorted(NORMAL_MAP))
    parser.add_argument("sample")
    return parser.parse_args()


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


def main() -> None:
    args = parse_args()
    dataset = args.dataset
    cancer_type = args.cancer_type
    sample = args.sample


    normal_types = NORMAL_MAP[cancer_type]
    normal_label = "All non-tumor"
    t0 = time.time()

    out_root = REPO_ROOT / "outputs" / dataset
    out_dir = out_root / cancer_type / sample
    out_dir.mkdir(parents=True, exist_ok=True)


    print(f"[1] Loading {dataset}/{cancer_type}/{sample}...", flush=True)
    coords = load_coords(dataset, cancer_type, sample)
    types_sub = load_cell_types(sample)
    df = coords.merge(types_sub, on="cell_id", how="inner")
    print(f"  {len(df)} cells", flush=True)
    if df.empty:
        raise RuntimeError(f"No cells after merge for {sample}")


    df["binary_type"] = "Normal"
    df.loc[df["cell_type"].astype(str).str.startswith("Tumor"), "binary_type"] = "Tumor"

    n_tumor = int((df.binary_type == "Tumor").sum())
    n_normal = int((df.binary_type == "Normal").sum())
    n_other = 0
    print(f"    Tumor: {n_tumor}, Normal: {n_normal}, Other: {n_other}", flush=True)

    x_min, x_max = df.x_centroid.min() - 100, df.x_centroid.max() + 100
    y_min, y_max = df.y_centroid.min() - 100, df.y_centroid.max() + 100
    nx = int(np.ceil((x_max - x_min) / RESOLUTION))
    ny = int(np.ceil((y_max - y_min) / RESOLUTION))


    print("[2] Masks...", flush=True)
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

    type_masks = {}
    for t in ["Tumor", "Normal"]:
        sub = df[df.binary_type == t]
        if len(sub) < 10:
            type_masks[t] = np.zeros((ny, nx), dtype=bool)
            continue
        raw = rasterize(sub, ny, nx, x_min, y_min)
        density = gaussian_filter(raw, sigma=sigma_r)
        if density.max() > 0:
            type_masks[t] = clean_mask(density > density_thresh * density.max(),
                                       closing_r=closing_r, min_region_area=min_area)
        else:
            type_masks[t] = np.zeros((ny, nx), dtype=bool)

    type_masks["Tumor"] &= tissue_mask
    type_masks["Normal"] &= tissue_mask
    type_masks["Normal"] &= ~type_masks["Tumor"]

    lb_n, n_n = label(type_masks["Normal"])
    for idx in range(1, n_n + 1):
        if (lb_n == idx).sum() < MIN_NORMAL_AREA:
            type_masks["Normal"][lb_n == idx] = False


    print("[3] Labeling regions...", flush=True)
    label_grid = np.full((ny, nx), -1, dtype=int)
    regions = []
    gid = 0
    rid = 1

    for t in ["Tumor", "Normal"]:
        if not type_masks[t].any():
            continue
        lb_t, n_t = label(type_masks[t])
        sizes = [(di, int((lb_t == di).sum())) for di in range(1, n_t + 1)]
        sizes.sort(key=lambda x: -x[1])
        for di, sz in sizes:
            comp = lb_t == di
            area_mm2 = sz * RESOLUTION**2 / 1e6
            dt = distance_transform_edt(comp)
            max_iy, max_ix = np.unravel_index(dt.argmax(), dt.shape)
            cx = max_ix * RESOLUTION + x_min
            cy = max_iy * RESOLUTION + y_min
            ls = f"{t}-{rid}"
            label_grid[comp] = gid
            regions.append((gid, t, ls, rid, comp, area_mm2, cx, cy))
            gid += 1
            rid += 1

    print(f"  {len(regions)} regions", flush=True)


    print("[4] Distance fields...", flush=True)
    tumor_mask = type_masks["Tumor"]
    normal_mask = type_masks["Normal"]
    normal_outside = normal_mask & ~tumor_mask


    if tumor_mask.any():
        sdf = (distance_transform_edt(~tumor_mask)
               - distance_transform_edt(tumor_mask)) * RESOLUTION


        _ti = distance_transform_edt(~tumor_mask, return_indices=True, return_distances=False)
        nearest_tumor_gid = label_grid[_ti[0], _ti[1]]
    else:
        sdf = np.full((ny, nx), np.nan, dtype=float)
        nearest_tumor_gid = np.full((ny, nx), -1, dtype=int)


    from scipy.ndimage import binary_erosion as _be
    tumor_interior = _be(tumor_mask, structure=disk(1))
    tumor_boundary = tumor_mask & ~tumor_interior
    interface = tumor_boundary & binary_dilation(normal_outside, structure=disk(2))
    n_interface = int(interface.sum())
    print(f"  Interface pixels: {n_interface}; Normal-outside: {int(normal_outside.sum())} px",
          flush=True)
    if n_interface > 0:
        dist_from_interface = distance_transform_edt(~interface) * RESOLUTION
        iface_signed = np.where(tumor_mask, -dist_from_interface, dist_from_interface)
    else:
        dist_from_interface = np.full((ny, nx), np.inf)
        iface_signed = np.full((ny, nx), np.nan, dtype=float)


    print("[5] Mapping cells...", flush=True)
    cell_ix = np.clip(((df.x_centroid.values - x_min) / RESOLUTION).astype(int), 0, nx - 1)
    cell_iy = np.clip(((df.y_centroid.values - y_min) / RESOLUTION).astype(int), 0, ny - 1)

    id_to_label = {r[0]: r[2] for r in regions}
    df["tumor_region"] = [
        id_to_label.get(label_grid[cell_iy[i], cell_ix[i]], "Unassigned")
        for i in range(len(df))
    ]
    in_tumor = tumor_mask[cell_iy, cell_ix]
    df["side"] = np.where(in_tumor, "Tumor", "Normal")
    df["dist_sdf_um"] = np.round(sdf[cell_iy, cell_ix], 1)


    df["dist_sdf_region"] = [
        id_to_label.get(int(nearest_tumor_gid[cell_iy[i], cell_ix[i]]), "")
        for i in range(len(df))
    ]
    df["dist_iface_um"] = np.round(iface_signed[cell_iy, cell_ix], 1)


    sdf_v = df["dist_sdf_um"].to_numpy(dtype=float)
    ifc_v = df["dist_iface_um"].to_numpy(dtype=float)
    n_cells = len(df)
    n_ifc_nan = int(np.isnan(ifc_v).sum())
    both_near = (np.abs(sdf_v) <= 100) & ~np.isnan(ifc_v)
    if both_near.sum() > 2:
        corr = float(np.corrcoef(sdf_v[both_near], ifc_v[both_near])[0, 1])
    else:
        corr = float("nan")
    print("  Distance measurement:", flush=True)
    if np.isfinite(sdf_v).any():
        print(f"    dist_sdf_um   range [{np.nanmin(sdf_v):.0f}, {np.nanmax(sdf_v):.0f}] "
              f"median {np.nanmedian(sdf_v):.0f}", flush=True)
    else:
        print("    dist_sdf_um   undefined (no tumor mask)", flush=True)
    print(f"    dist_iface_um undefined for {n_ifc_nan}/{n_cells} cells "
          f"({100*n_ifc_nan/max(n_cells,1):.1f}%)", flush=True)
    print(f"    corr(sdf, iface) near edge (|sdf|<=100um): {corr:.3f}", flush=True)
    print(f"    cells inside tumor: {int(in_tumor.sum())}, outside: {int((~in_tumor).sum())}",
          flush=True)


    csv_path = out_dir / f"{sample}_boundary_distance.csv"
    df[["cell_id", "tumor_region", "side", "dist_sdf_um", "dist_sdf_region",
        "dist_iface_um", "cell_type"]].to_csv(csv_path, index=False)
    print(f"  Saved: {csv_path} ({len(df)} cells, all — no +-150 filter)", flush=True)


    print("[6] Figure...", flush=True)
    from matplotlib.colors import LinearSegmentedColormap
    cmap_out = LinearSegmentedColormap.from_list("bd_out", BAND_COLORS_OUT)
    cmap_in = LinearSegmentedColormap.from_list("bd_in", BAND_COLORS_IN)

    fig, ax = plt.subplots(figsize=(14, 14))


    other = df[df.binary_type == "Other"]
    if len(other):
        ax.scatter(other.x_centroid, other.y_centroid, c='#d5d5d5', s=0.1, alpha=0.1,
                   rasterized=True, zorder=0)


    for t in ["Tumor", "Normal"]:
        sub = df[df.binary_type == t]
        if len(sub):
            ax.scatter(sub.x_centroid, sub.y_centroid, c=COLORS[t], s=0.8, alpha=0.6,
                       rasterized=True, zorder=2)


    overlay = np.zeros((ny, nx, 4), dtype=np.float32)
    for _, t, _, _, comp, _, _, _ in regions:
        overlay[comp] = mcolors.to_rgba(COLORS[t], alpha=0.25)


    out_zone = tissue_mask & ~tumor_mask & (dist_from_interface <= GRAD_MAX_UM)
    if out_zone.any():
        cols = cmap_out(np.clip(dist_from_interface[out_zone] / GRAD_MAX_UM, 0, 1))
        cols[:, 3] = 0.35
        overlay[out_zone] = cols
    in_zone = tumor_mask & (dist_from_interface <= GRAD_MAX_UM)
    if in_zone.any():
        cols = cmap_in(np.clip(dist_from_interface[in_zone] / GRAD_MAX_UM, 0, 1))
        cols[:, 3] = 0.35
        overlay[in_zone] = cols

    ax.imshow(overlay, extent=[x_min, x_max, y_min, y_max], origin='lower',
              aspect='auto', zorder=3, interpolation='nearest')


    LABEL_OFFSET_UM = 250
    for _, t, _, rid_num, comp, area, cx, cy in sorted(regions, key=lambda r: (r[1] == "Tumor")):
        color = COLORS[t]
        zc = 7 if t == "Tumor" else 5
        for c in find_contours(comp.astype(float), 0.5):
            xs = c[:, 1] * RESOLUTION + x_min
            ys = c[:, 0] * RESOLUTION + y_min
            ax.plot(xs, ys, color=color, lw=1.5, alpha=0.8, zorder=zc)

        dt = distance_transform_edt(comp)
        max_radius_um = dt.max() * RESOLUTION
        img_cx = (x_min + x_max) / 2
        img_cy = (y_min + y_max) / 2
        dx = cx - img_cx
        dy = cy - img_cy
        dist = max(np.sqrt(dx**2 + dy**2), 1)
        offset = max(max_radius_um + 200, LABEL_OFFSET_UM)
        ox = cx + offset * dx / dist
        oy = cy + offset * dy / dist
        ax.annotate(f"{t}-{rid_num}", xy=(cx, cy), xytext=(ox, oy),
                    fontsize=8, fontweight='bold', ha='center', va='center',
                    color='black', zorder=10,
                    bbox=dict(boxstyle='round,pad=0.2', fc='white', alpha=0.9,
                              ec='black', lw=0.8),
                    arrowprops=dict(arrowstyle='-', color='black', lw=0.8))


    ISO_LEVELS = [50, 100, 150]
    for bd in ISO_LEVELS:
        region_out = tumor_mask | (tissue_mask & ~tumor_mask & (dist_from_interface <= bd))
        for c in find_contours(region_out.astype(float), 0.5):
            ax.plot(c[:, 1] * RESOLUTION + x_min, c[:, 0] * RESOLUTION + y_min,
                    color="#b35900", lw=0.9, alpha=0.9, ls="--", zorder=6)
        region_in = tumor_mask & (dist_from_interface >= bd)
        for c in find_contours(region_in.astype(float), 0.5):
            ax.plot(c[:, 1] * RESOLUTION + x_min, c[:, 0] * RESOLUTION + y_min,
                    color="#7b241c", lw=0.9, alpha=0.9, ls="--", zorder=6)

    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_max, y_min)
    ax.set_aspect('equal')
    ax.set_xlabel("X (um)", fontsize=12)
    ax.set_ylabel("Y (um)", fontsize=12)
    ax.xaxis.set_major_locator(MultipleLocator(1000))
    ax.yaxis.set_major_locator(MultipleLocator(1000))


    gmax = int(GRAD_MAX_UM)
    normal_regions = [r for r in regions if r[1] == "Normal"]
    legend_items = [
        Patch(fc=COLORS["Tumor"], alpha=0.7, label=f'Tumor core (>{gmax}um)'),
        Patch(fc=BAND_COLORS_IN[1], alpha=0.6, label=f'Inward 0-{gmax}um (gradient)'),
        Patch(fc=BAND_COLORS_OUT[0], alpha=0.6, label=f'Outward 0-{gmax}um (gradient)'),
        plt.Line2D([0], [0], color="#555555", ls="--", lw=1.0, label='iso 50/100/150um'),
        Patch(fc=COLORS["Normal"], alpha=0.7,
              label=f'Normal ({len(normal_regions)} islands)\n[{normal_label}]'),
    ]
    ax.legend(handles=legend_items, loc='upper left', bbox_to_anchor=(1.01, 1),
              fontsize=9, framealpha=0.9, borderaxespad=0)
    ax.set_title(f"{sample} ({cancer_type}): Tumor Boundary",
                 fontsize=14, fontweight='bold')

    fig_path = out_dir / f"{sample}_boundary_distance.png"
    fig.savefig(fig_path, dpi=250, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: {fig_path}", flush=True)
    print(f"Done. {time.time() - t0:.1f}s", flush=True)


if __name__ == "__main__":
    main()
