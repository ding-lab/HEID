#!/usr/bin/env python -u
from __future__ import annotations
import os
PROJECTS_ROOT = os.environ.get("PROJECTS_ROOT", "/data/heid")
from pathlib import Path
_RELEASE = Path(__file__).resolve().parents[3]
import argparse
import json
import os
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tifffile
from matplotlib.lines import Line2D
from scipy.ndimage import (
    binary_closing, binary_dilation, binary_fill_holes, binary_opening,
    distance_transform_edt, gaussian_filter, label as ndimage_label,
)
from skimage.measure import find_contours

INFERENCE_ROOT = Path(PROJECTS_ROOT + "/cell/inference")
COHORT = os.environ.get("CELL_COHORT", "tcga_pancancer")
ROOT = INFERENCE_ROOT / COHORT
PRED = ROOT / "data" / "predictions"
OUT = ROOT / "outputs"
def tau_file(project: str):
    from out_paths import summary_dir
    return summary_dir(project) / "operating_point.json"

NERVE_ROOT = _RELEASE / "nerve/prediction"
sys.path.insert(0, str(_RELEASE / "nerve/prediction/scripts"))
import region_operator as OP

sys.path.insert(0, str(_RELEASE / "cell/inference/scripts"))
from region_order import serpentine_order
from region_ops import regions_for
from out_paths import slide_dir, summary_path, all_projects

TARGET_POS_FRAC = 0.003
TAU_ABS = 0.60


MIN_CONFIDENCE = os.environ.get("NERVE_CONFIDENCE", "high")
PANEL = "5k"
SCHWANN_CLASS = "Schwann"
TUMOR_CLASS = "Tumor"

PALETTE = {
    "Tumor": "#D62728", "Fibroblast": "#8C564B", "Myeloid_NOS": "#FF7F0E", "NK_T": "#1F77B4",
    "B_cell": "#08306B", "Plasma": "#9467BD", "NonMalignant_Parenchymal": "#E377C2",
    "Endothelial": "#1B7A1B", "SMC": "#7F7F7F", "Schwann": "#00B0F0", "Others": "#C7C7C7",
}


C_TERRITORY = "#7A1416"
C_SCHWANN = "#00B0F0"
UM_PER_INCH_WIDE = 1400.0
UM_PER_INCH = 1000.0


RESOLUTION = 10
SIGMA_REGION = 3
SIGMA_ECO = 10
CLOSING_R = 2
OPENING_R = 3
MIN_REGION_AREA = 100
MAX_HOLE_AREA = 50000
DENSITY_THRESH = 0.03
TISSUE_DENSITY_THRESH = 0.015
PX_MM2 = RESOLUTION ** 2 / 1e6


def schwann_tag(n_regions: int) -> str:
    return f" [Schwann x{n_regions}]" if n_regions else " [Schwann: none]"


def thumbnail(svs: str, native_mpp: float, target_um_px: float = 4.0):
    with tifffile.TiffFile(svs) as handle:
        series = handle.series[0]
        level0_h = series.levels[0].shape[0]
        pick, pick_mpp = 0, native_mpp
        for index, level in enumerate(series.levels):
            mpp = native_mpp * level0_h / level.shape[0]
            if mpp <= target_um_px * (1 + 1e-6):
                pick, pick_mpp = index, mpp
        step = max(1, int(target_um_px / pick_mpp))
        if step > 1:


            import zarr
            store = zarr.open(series.levels[pick].aszarr(), mode="r")
            image = np.asarray(store[::step, ::step])
            pick_mpp *= step
        else:
            image = series.levels[pick].asarray()
    if image.ndim == 2:
        image = np.stack([image] * 3, -1)
    return np.ascontiguousarray(image[..., :3]), pick_mpp


def disk(radius: int) -> np.ndarray:
    line = np.arange(-radius, radius + 1)
    xs, ys = np.meshgrid(line, line)
    return (xs ** 2 + ys ** 2 <= radius ** 2).astype(np.uint8)


def rasterize(x_um, y_um, shape) -> np.ndarray:
    grid = np.zeros(shape, np.float32)
    ix = np.clip((x_um / RESOLUTION).astype(int), 0, shape[1] - 1)
    iy = np.clip((y_um / RESOLUTION).astype(int), 0, shape[0] - 1)
    np.add.at(grid, (iy, ix), 1)
    return grid


def clean_mask(mask: np.ndarray) -> np.ndarray:
    mask = binary_closing(mask, structure=disk(CLOSING_R))
    filled = binary_fill_holes(mask)
    holes = filled & ~mask
    if holes.any():
        labelled, count = ndimage_label(holes)
        for index in range(1, count + 1):
            if (labelled == index).sum() > MAX_HOLE_AREA:
                filled[labelled == index] = False
    mask = binary_opening(filled, structure=disk(OPENING_R))
    labelled, count = ndimage_label(mask)
    for index in range(1, count + 1):
        if (labelled == index).sum() < MIN_REGION_AREA:
            mask[labelled == index] = False
    return mask


def tumour_territory(pred: pd.DataFrame):
    span_x = float(pred.x_um.max()) + 200.0
    span_y = float(pred.y_um.max()) + 200.0
    shape = (int(np.ceil(span_y / RESOLUTION)), int(np.ceil(span_x / RESOLUTION)))
    all_raw = rasterize(pred.x_um.to_numpy(), pred.y_um.to_numpy(), shape)
    tissue = binary_fill_holes(binary_closing(
        gaussian_filter(all_raw, sigma=SIGMA_ECO) > TISSUE_DENSITY_THRESH, structure=disk(10)))
    tumour = pred[pred.pred_class_name == TUMOR_CLASS]
    if len(tumour) < 10:
        return np.zeros(shape, bool), tissue, tumour
    density = gaussian_filter(
        rasterize(tumour.x_um.to_numpy(), tumour.y_um.to_numpy(), shape), sigma=SIGMA_REGION)
    mask = (clean_mask(density > DENSITY_THRESH * density.max())
            if density.max() > 0 else np.zeros(shape, bool))
    return mask & tissue, tissue, tumour


def tau_label(tau: float) -> str:
    if os.environ.get("CELL_TAU_FORCE"):
        return f"forced tau {tau:.3f}"
    return (f"absolute tau {tau:.3f}" if os.environ.get("CELL_TAU_ABS_MODE")
            else f"posfrac {TARGET_POS_FRAC} (tau {tau:.4f})")


def head_identity_of(slides: list[str]) -> str | None:
    seen = set()
    for slide in slides:
        meta_path = PRED / f"{slide}.meta.json"
        if not meta_path.exists():
            continue
        identity = json.loads(meta_path.read_text()).get("head_identity")
        if identity and identity.get("experiment_id"):
            seen.add(str(identity["experiment_id"]))
    if len(seen) > 1:
        raise SystemExit(f"FATAL: this group mixes predictions from different Cell heads: {sorted(seen)}")
    return next(iter(seen)) if seen else None


def calibrate_tau(slides: list[str], project: str) -> dict:
    head_id = head_identity_of(slides)

    available = sum(1 for s in slides if (PRED / f"{s}.parquet").exists())
    partial_ok = bool(os.environ.get("CELL_TAU_PARTIAL_OK"))


    forced = os.environ.get("CELL_TAU_FORCE")
    if forced:
        tau = float(forced)
        print(f"[tau] forcing {tau:.5f} (experimental; bypasses the threshold cache; used for this run's standard output files)",
              flush=True)
        return {"mode": "forced", "tau": tau, "rule": f"forced by CELL_TAU_FORCE={forced}",
                "project_id": project, "n_slides_pooled": None}

    requested_mode = "absolute" if os.environ.get("CELL_TAU_ABS_MODE") else "quantile"
    TAU_FILE = tau_file(project)
    if TAU_FILE.exists():
        cached = json.loads(TAU_FILE.read_text())
        cached_mode = cached.get("mode")
        if cached_mode is None:
            absolute_rule = cached.get("rule") in (
                "explicit absolute probability threshold",
            )
            if "target_pos_frac" in cached and not absolute_rule:
                cached_mode = "quantile"
            elif "target_pos_frac" not in cached and absolute_rule:
                cached_mode = "absolute"
        if cached_mode not in ("quantile", "absolute"):
            raise SystemExit(f"FATAL: {TAU_FILE} does not identify an unambiguous threshold mode.")
        if cached_mode != requested_mode:
            raise SystemExit(
                f"FATAL: cached threshold mode is {cached_mode}, but this run requests {requested_mode}. "
                "Use a separate output workspace or explicitly recalibrate its threshold cache."
            )
        if requested_mode == "absolute":
            requested_tau = float(os.environ.get("CELL_TAU_ABS", TAU_ABS))
            if float(cached["tau"]) != requested_tau:
                raise SystemExit(
                    f"FATAL: cached absolute threshold is {cached['tau']}, but this run requests {requested_tau}. "
                    "Use a separate output workspace or explicitly recalibrate its threshold cache."
                )
        cached_head = cached.get("head_experiment_id")
        if cached_head is not None and head_id is not None and cached_head != head_id:
            raise SystemExit(
                f"FATAL: cached threshold was calibrated on predictions of Cell head {cached_head}, "
                f"but this group's predictions come from {head_id}. The operating point is head-specific; "
                "use a separate output workspace or explicitly recalibrate its threshold cache."
            )
        was = int(cached.get("n_slides_pooled", 0))
        if available > was and not partial_ok:
            raise SystemExit(
                f"FATAL: cached threshold was calibrated on {was} slides, but this group now has {available} available.\n"
                f"  The group gained more slides, so the old threshold is no longer its quantile.\n"
                f"  Delete {TAU_FILE} and recalibrate, or set CELL_TAU_PARTIAL_OK=1 to explicitly accept it (its outputs are not deliverable).")
        return cached
    if available < len(slides) and not partial_ok:
        raise SystemExit(
            f"FATAL: threshold should be calibrated on {len(slides)} slides, but only {available} have inference results.\n"
            f"  Wait for this group's inference to finish before calibrating, or set CELL_TAU_PARTIAL_OK=1 to explicitly accept it (its outputs are not deliverable).")
    if os.environ.get("CELL_TAU_ABS_MODE"):
        tau = float(os.environ.get("CELL_TAU_ABS", TAU_ABS))
        pool_n = 0
        n_pos = 0
        n_slides = 0
        for slide in slides:
            path = PRED / f"{slide}.parquet"
            if not path.exists():
                continue
            frame = pd.read_parquet(path, columns=["scored", "schwann_prob"])
            values = frame.loc[frame.scored, "schwann_prob"].to_numpy(np.float32)
            pool_n += values.size
            n_slides += 1
            n_pos += int((values > tau).sum())
        record = {
            "mode": "absolute", "rule": "explicit absolute probability threshold",
            "tau": tau, "tau_source": "CELL_TAU_ABS or the default absolute runtime setting",
            "achieved_pos_frac": (n_pos / pool_n) if pool_n else 0.0,
            "n_cells_pooled": pool_n, "n_slides_pooled": n_slides,
            "n_slides_requested": len(slides), "head_experiment_id": head_id,
        }
        TAU_FILE.write_text(json.dumps(record, indent=2))
        print(f"[tau] absolute {tau} -> {n_pos:,}/{pool_n:,} cells "
              f"({record['achieved_pos_frac']*100:.4f}%)", flush=True)
        return record
    pool = []
    for slide in slides:
        path = PRED / f"{slide}.parquet"
        if not path.exists():
            continue
        frame = pd.read_parquet(path, columns=["scored", "schwann_prob"])
        pool.append(frame.loc[frame.scored, "schwann_prob"].to_numpy(np.float32))
    if not pool:
        raise SystemExit("no predictions to calibrate on")
    allp = np.concatenate(pool)
    tau = float(np.quantile(allp, 1.0 - TARGET_POS_FRAC))
    record = {
        "mode": "quantile", "target_pos_frac": TARGET_POS_FRAC, "tau": tau,
        "achieved_pos_frac": float((allp > tau).mean()),
        "n_cells_pooled": int(allp.size), "n_slides_pooled": len(pool),
        "n_slides_requested": len(slides),
        "rule": "transferred positive fraction; tau is the pooled quantile of this cancer group",
        "project_id": project, "head_experiment_id": head_id,
    }
    TAU_FILE.write_text(json.dumps(record, indent=2))
    print(f"[tau] {tau:.6f} from {allp.size:,} cells over {len(pool)} slides "
          f"(achieved posfrac {record['achieved_pos_frac']:.5f})", flush=True)
    return record


def place_region_labels(ax, cluster_grid, clusters, meta, span_x, span_y):
    if not clusters:
        return
    occupied = binary_dilation(cluster_grid > 0, disk(2))
    n_rows, n_cols = occupied.shape
    centroids = []
    for cluster in clusters:
        component = cluster_grid == cluster["cluster_id"]
        rows, cols = np.nonzero(component)
        centroids.append((cols.mean() * OP.RESOLUTION + meta["x_min"],
                          rows.mean() * OP.RESOLUTION + meta["y_min"],
                          max(rows.ptp(), cols.ptp()) * OP.RESOLUTION / 2 + OP.RESOLUTION))
    centroids = np.asarray(centroids, float)
    ranks = serpentine_order(centroids[:, 0], centroids[:, 1])

    def free(px, py, radius):
        if not (0 <= px <= span_x and 0 <= py <= span_y):
            return False
        gx = int((px - meta["x_min"]) / OP.RESOLUTION)
        gy = int((py - meta["y_min"]) / OP.RESOLUTION)
        if 0 <= gy < n_rows and 0 <= gx < n_cols and occupied[gy, gx]:
            return False
        return all((px - qx) ** 2 + (py - qy) ** 2 > (radius + qr) ** 2 for qx, qy, qr in placed)

    chip_r = 0.011 * max(span_x, span_y)
    angles = np.deg2rad(np.arange(0, 360, 30))
    placed = []
    for index in np.argsort(ranks):
        cx, cy, region_r = centroids[index]
        spot = None
        for step in (1.0, 1.6, 2.4, 3.4, 4.8, 6.6):
            distance = region_r + chip_r * step
            for angle in angles:
                px, py = cx + distance * np.cos(angle), cy + distance * np.sin(angle)
                if free(px, py, chip_r):
                    spot = (px, py)
                    break
            if spot:
                break
        if spot is None:
            spot = (cx + region_r + chip_r, cy)
        px, py = spot
        placed.append((px, py, chip_r))
        ax.plot([cx, px], [cy, py], color="#333333", lw=0.45, alpha=0.75,
                zorder=11, solid_capstyle="butt")
        ax.text(px, py, str(int(ranks[index])), fontsize=4.6, fontweight="bold",
                ha="center", va="center", color="#111111", zorder=13, clip_on=False,
                bbox=dict(boxstyle="round,pad=0.16", fc="white", ec="#555555",
                          lw=0.35, alpha=0.88))


def build(slide: str, row: pd.Series, tau: float) -> dict:
    pred = pd.read_parquet(PRED / f"{slide}.parquet")
    pred = pred[pred.scored].reset_index(drop=True)
    x = pred.x_um.to_numpy(float)
    y = pred.y_um.to_numpy(float)
    is_schwann = pred.schwann_prob.to_numpy(float) > tau

    clusters, member, cluster_grid, meta = regions_for(x, y, is_schwann, prob=pred.schwann_prob.to_numpy(float))
    if meta.get("fill_mode"):
        print(f"[fill] {slide}: {meta['fill_mode']} -> {len(clusters)} regions", flush=True)
    if meta["n_dropped_low_confidence"]:
        print(f"[conf] {slide}: dropped {meta['n_dropped_low_confidence']} low-confidence "
              f"regions, {len(clusters)} kept", flush=True)

    mask, tissue, tumour = tumour_territory(pred)
    to_tumour = distance_transform_edt(~mask) * RESOLUTION if mask.any() else None

    rows = []
    for cluster in clusters:
        members = member == cluster["cluster_id"]
        cx = float(x[members].mean()) if members.any() else np.nan
        cy = float(y[members].mean()) if members.any() else np.nan
        status = "-tumor"
        if to_tumour is not None and np.isfinite(cx):
            gx = int(np.clip(cx / RESOLUTION, 0, mask.shape[1] - 1))
            gy = int(np.clip(cy / RESOLUTION, 0, mask.shape[0] - 1))
            distance = float(to_tumour[gy, gx])
            status = ("in tumor core" if distance <= 0 else
                      "in 50um halo zone" if distance <= 50 else
                      "100um halo" if distance <= 100 else "-tumor")
        rows.append({
            "cluster_id": cluster["cluster_id"], "status": status,
            "shape_class": cluster["shape_class"], "confidence": cluster["confidence"],
            "n_schwann": cluster["n_schwann"], "area_mm2": round(cluster["area_mm2"], 4),
            "areal_density": round(cluster["areal_density"], 1),
            "linear_density": round(cluster["linear_density"], 1),
            "aspect_ratio": round(cluster["aspect_ratio"], 3),
            "solidity": round(cluster["solidity"], 3),
            "median_nn_um": round(cluster["median_nn_um"], 2),
            "centroid_x_um": round(cx, 1), "centroid_y_um": round(cy, 1),
            "mean_schwann_prob": round(float(pred.schwann_prob.to_numpy()[members].mean()), 4)
            if members.any() else np.nan,
        })
    stats = pd.DataFrame(rows, columns=[
        "cluster_id", "status", "shape_class", "confidence", "n_schwann", "area_mm2",
        "areal_density", "linear_density", "aspect_ratio", "solidity", "median_nn_um",
        "centroid_x_um", "centroid_y_um", "mean_schwann_prob"])

    directory = slide_dir(slide)
    directory.mkdir(parents=True, exist_ok=True)
    stats.to_csv(directory / f"{slide}_cluster_stats.csv", index=False)
    status_of = dict(zip(stats.cluster_id, stats.status)) if len(stats) else {}
    pd.DataFrame({
        "cell_id": pred.cell_id,
        "schwann_cluster": np.where(member > 0, member, pd.NA),
        "schwann_tumor_status": [status_of.get(int(r)) if r > 0 else pd.NA for r in member],
        "cell_type": pred.pred_class_name,
        "schwann_prob": pred.schwann_prob.round(4),
    }).to_csv(directory / f"{slide}_cells.csv", index=False)

    if os.environ.get("CELL_TABLES_ONLY"):


        stats = pd.read_csv(directory / f"{slide}_cluster_stats.csv") if \
            (directory / f"{slide}_cluster_stats.csv").exists() else pd.DataFrame()

        mask, tissue, _ = tumour_territory(pred)
        tumour_mm2 = float(mask.sum() * PX_MM2)
        tissue_mm2 = float(tissue.sum() * PX_MM2)
        return {"slide": slide, "has_schwann": int(len(stats) > 0), "n_cells": int(len(pred)),
                "n_schwann_cells": int(is_schwann.sum()),
                "n_regions": int(len(stats)),
                "region_area_mm2": round(float(stats.area_mm2.sum()), 4) if len(stats) else 0.0,
                "n_tumor_cells": int((pred.pred_class_name == TUMOR_CLASS).sum()),
                "tumor_area_mm2": tumour_mm2, "tissue_area_mm2": tissue_mm2,
                "tumor_fraction_of_tissue": round(tumour_mm2 / tissue_mm2, 4) if tissue_mm2 else 0.0,
                "status_counts": stats.status.value_counts().to_dict() if len(stats) else {}}

    image, thumb_mpp = thumbnail(str(row.svs), float(row.native_mpp))
    span_x = image.shape[1] * thumb_mpp
    span_y = image.shape[0] * thumb_mpp


    figure, ax = plt.subplots(figsize=(span_x / UM_PER_INCH + 3.0,
                                       max(span_y / UM_PER_INCH, 3.0)))
    ax.imshow(image, extent=[0, span_x, span_y, 0], zorder=1)
    for cluster in clusters:
        component = cluster_grid == cluster["cluster_id"]
        for contour in find_contours(component.astype(float), 0.5):
            ax.plot(contour[:, 1] * OP.RESOLUTION + meta["x_min"],
                    contour[:, 0] * OP.RESOLUTION + meta["y_min"],
                    color=C_SCHWANN, lw=1.3, alpha=0.9, zorder=4)
    ax.scatter(x[is_schwann], y[is_schwann], s=2.0, c=C_SCHWANN, alpha=0.5,
               linewidths=0, rasterized=True, zorder=3)
    place_region_labels(ax, cluster_grid, clusters, meta, span_x, span_y)
    ax.set_xlim(0, span_x); ax.set_ylim(span_y, 0); ax.set_aspect("equal")
    ax.set_xlabel("X (µm)"); ax.set_ylabel("Y (µm)")
    ax.set_title(f"{slide}{schwann_tag(len(stats))}  {len(stats)} predicted nerve regions from "
                 f"{int(is_schwann.sum()):,} Schwann cells\n"
                 f"pan-cancer Cell probability + nerve operator, {tau_label(tau)}"
                 f"  |  MODEL PREDICTION", fontsize=10, fontweight="bold")
    ax.legend(handles=[
        Line2D([], [], marker="o", linestyle="none", markersize=5, markerfacecolor=C_SCHWANN,
               markeredgecolor="none", label=f"Schwann cell (prob > {tau:.3f})"),
        Line2D([], [], color=C_SCHWANN, lw=1.3, label="called nerve region")],
        loc="upper left", bbox_to_anchor=(1.01, 1.0), fontsize=9, borderaxespad=0.0)
    plt.tight_layout()
    for suffix in ("png",):
        figure.savefig(directory / f"{slide}.{suffix}", dpi=200, bbox_inches="tight")
    plt.close(figure)


    counts = pred.pred_class_name.value_counts()
    figure, ax = plt.subplots(figsize=(span_x / UM_PER_INCH_WIDE + 3.2,
                                       max(span_y / UM_PER_INCH_WIDE, 4.0)))
    ax.imshow(image, extent=[0, span_x, span_y, 0], zorder=1)
    handles = []
    for rank, (name, count) in enumerate(counts.sort_values(ascending=False).items()):
        subset = pred[pred.pred_class_name == name]
        colour = PALETTE.get(name, "#000000")
        size = float(np.clip(2.0e3 / max(count, 1), 0.5, 4.0))
        alpha = float(np.clip(0.30 + 900.0 / max(count, 1), 0.35, 0.90))
        ax.scatter(subset.x_um, subset.y_um, s=size, c=colour, alpha=alpha,
                   linewidths=0, rasterized=True, zorder=3 + rank)
        handles.append((count, Line2D([], [], marker="o", linestyle="none", markersize=6,
                                      markerfacecolor=colour, markeredgecolor="none",
                                      label=f"{name} ({count:,})")))
    handles.sort(key=lambda item: -item[0])
    ax.set_xlim(0, span_x); ax.set_ylim(span_y, 0); ax.set_aspect("equal")
    ax.set_xlabel("X (µm)"); ax.set_ylabel("Y (µm)")
    ax.set_title(f"{slide}  {len(pred):,} typed cells, {len(counts)} of 11 classes present\n"
                 f"colour = predicted class · dot size grows for rarer classes  ·  "
                 f"pan-cancer Cell  |  MODEL PREDICTION", fontsize=10, fontweight="bold")
    ax.legend(handles=[h for _, h in handles], loc="upper left", bbox_to_anchor=(1.01, 1.0),
              fontsize=8, borderaxespad=0.0, title="predicted class (cells)", title_fontsize=8)
    plt.tight_layout()

    for suffix in ("png",):
        figure.savefig(directory / f"{slide}_celltype_overlay.{suffix}", dpi=200,
                       bbox_inches="tight")
    plt.close(figure)


    tumour_mm2 = float(mask.sum() * PX_MM2)
    tissue_mm2 = float(tissue.sum() * PX_MM2)
    width_in = span_x / UM_PER_INCH_WIDE
    figure, axes = plt.subplots(1, 2, figsize=(2 * width_in + 3.5,
                                               max(span_y / UM_PER_INCH_WIDE, 4.0)))
    point_size = float(np.clip(1.2e4 / max(len(tumour), 1), 0.35, 1.0))
    for ax, show in zip(axes, (False, True)):
        ax.imshow(image, extent=[0, span_x, span_y, 0], zorder=1)
        if show and mask.any():
            rgba = np.zeros((*mask.shape, 4), np.float32)
            rgba[mask] = list(matplotlib.colors.to_rgba(C_TERRITORY)[:3]) + [0.28]
            ax.imshow(rgba, extent=[0, mask.shape[1] * RESOLUTION, mask.shape[0] * RESOLUTION, 0],
                      interpolation="nearest", zorder=2)
            for contour in find_contours(mask.astype(float), 0.5):
                ax.plot(contour[:, 1] * RESOLUTION, contour[:, 0] * RESOLUTION,
                        color=C_TERRITORY, lw=0.8, alpha=0.9, zorder=3)
        ax.scatter(tumour.x_um, tumour.y_um, s=point_size, c=PALETTE["Tumor"], alpha=0.55,
                   linewidths=0, rasterized=True, zorder=4)
        if len(stats):
            ax.scatter(stats.centroid_x_um, stats.centroid_y_um, s=48, facecolors="none",
                       edgecolors=C_SCHWANN, linewidths=1.4, zorder=5)
        ax.set_xlim(0, span_x); ax.set_ylim(span_y, 0); ax.set_aspect("equal")
        ax.set_xlabel("X (µm)")
        ax.set_title("predicted tumour cells" if not show
                     else "+ tumour territory called from them", fontsize=11)
    axes[0].set_ylabel("Y (µm)")
    figure.suptitle(f"{slide}{schwann_tag(len(stats))}   {len(tumour):,} predicted tumour cells "
                    f"of {len(pred):,}  ·  "
                    f"territory {tumour_mm2:.1f} mm² "
                    f"({tumour_mm2 / tissue_mm2 if tissue_mm2 else 0:.0%} of tissue)  ·  "
                    f"{len(stats)} predicted nerve regions   |   MODEL PREDICTION",
                    fontsize=11, fontweight="bold")
    axes[1].legend(handles=[
        Line2D([], [], marker="o", linestyle="none", markersize=6,
               markerfacecolor=PALETTE["Tumor"], markeredgecolor="none",
               label="predicted tumour cell"),
        Line2D([], [], color=C_TERRITORY, lw=1.2, label="tumour territory"),
        Line2D([], [], marker="o", linestyle="none", markersize=8, markerfacecolor="none",
               markeredgecolor=C_SCHWANN, label="predicted nerve region")],
        loc="upper left", bbox_to_anchor=(1.01, 1.0), fontsize=9, borderaxespad=0.0)
    plt.tight_layout()

    for suffix in ("png",):
        figure.savefig(directory / f"{slide}_tumor_region.{suffix}", dpi=200,
                       bbox_inches="tight")
    plt.close(figure)

    record = {"slide": slide, "has_schwann": int(len(stats) > 0), "n_cells": int(len(pred)),
              "n_schwann_cells": int(is_schwann.sum()), "n_regions": len(stats),
              "region_area_mm2": round(float(stats.area_mm2.sum()) if len(stats) else 0.0, 4),
              "n_tumor_cells": int(len(tumour)), "tumor_area_mm2": round(tumour_mm2, 4),
              "tissue_area_mm2": round(tissue_mm2, 4),
              "tumor_fraction_of_tissue": round(tumour_mm2 / tissue_mm2, 4) if tissue_mm2 else 0.0,
              "status_counts": stats.status.value_counts().to_dict() if len(stats) else {}}
    print(f"[{slide}] {record['n_schwann_cells']:,} Schwann -> {record['n_regions']} regions · "
          f"{record['n_tumor_cells']:,} tumour cells -> {record['tumor_area_mm2']:.1f} mm²",
          flush=True)
    return record


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest",
                        default=str(INFERENCE_ROOT / "configs" / COHORT / "run_manifest.tsv"))
    parser.add_argument("--slides", nargs="+")
    parser.add_argument("--shard", type=int, default=0)
    parser.add_argument("--n-shards", type=int, default=1)
    parser.add_argument("--calibrate-only", action="store_true")
    parser.add_argument("--project", help="process only this one cancer type (deliverables are organized by cancer type)")
    args = parser.parse_args()

    table = pd.read_csv(args.manifest, sep="\t").set_index("slide")
    if args.project:
        table = table[table.project_id == args.project]
        if not len(table):
            sys.exit(f"FATAL: cancer type {args.project} not in the manifest")


    slides = args.slides if args.slides else list(table.index)[args.shard::args.n_shards]
    by_project = {}
    for slide in slides:
        by_project.setdefault(table.loc[slide, "project_id"], []).append(slide)

    tau_of = {}
    for project in sorted(by_project):

        every = list(table[table.project_id == project].index)
        point = calibrate_tau(every, project)
        tau_of[project] = float(point["tau"])
        if point.get("mode") == "forced":
            print(f"[tau] {project}: {tau_of[project]:.5f} (forced for this run)", flush=True)
        else:
            print(f"[tau] {project}: {tau_of[project]:.5f} "
                  f"(calibrated on {point.get('n_slides_pooled')} slides of this cancer type)", flush=True)
    if args.calibrate_only:
        return

    rows = []
    for slide in slides:
        tau = tau_of[table.loc[slide, "project_id"]]
        if not (PRED / f"{slide}.parquet").exists():
            print(f"[SKIP] {slide}: no predictions", flush=True)
            continue
        try:
            rows.append(build(slide, table.loc[slide], tau))
        except Exception as exc:
            print(f"[FAIL] {slide}: {type(exc).__name__}: {exc}", flush=True)
    if rows:
        summary = pd.DataFrame(rows)


        out = summary_path(args.project or COHORT, args.shard)
        out.parent.mkdir(parents=True, exist_ok=True)
        summary.to_csv(out, index=False)
        print(summary.to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
