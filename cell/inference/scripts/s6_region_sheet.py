#!/usr/bin/env python -u
from __future__ import annotations
import os
PROJECTS_ROOT = os.environ.get("PROJECTS_ROOT", "/data/heid")
from pathlib import Path
_RELEASE = Path(__file__).resolve().parents[3]

import argparse
import json
import math
import sys
from pathlib import Path

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from skimage.measure import find_contours

INFERENCE_ROOT = Path(PROJECTS_ROOT + "/cell/inference")
COHORT = os.environ.get("CELL_COHORT", "tcga_pancancer")
ROOT = INFERENCE_ROOT / COHORT
PRED = ROOT / "data" / "predictions"
OUT = ROOT / "outputs"
def _tau_file(project):
    from out_paths import summary_dir
    return summary_dir(project) / "operating_point.json"

sys.path.insert(0, str(_RELEASE / "nerve/prediction/scripts"))
import region_operator as OP

sys.path.insert(0, str(_RELEASE / "cell/inference/scripts"))
from region_order import serpentine_order
from region_ops import regions_for
from out_paths import slide_dir, summary_path, all_projects

sys.path.insert(0, str(_RELEASE / "cell/inference/scripts"))
import slide_canvas as L

PANEL = "5k"
MIN_CONFIDENCE = os.environ.get("NERVE_CONFIDENCE", "all")
C_SCHWANN = "#00B0F0"
C_REGION = os.environ.get("CELL_OUTLINE_COLOR", "#00B0F0")

WINDOW_UM = 400.0
TILE_PX = 400
NCOL = 6
TILE_IN = 2.0
DPI = 200
MAX_ROWS_PER_SHEET = 70


PREFIX = "sheet_"
def _clear_stale(directory, slide):
    import glob as _glob
    import os as _os
    for path in _glob.glob(str(directory / f"{PREFIX}{slide}*")):
        _os.remove(path)


def log(message: str) -> None:
    print(message, flush=True)


def region_contours_um(cluster_grid, meta, cluster_id):
    component = (cluster_grid == cluster_id).astype(float)
    return [np.column_stack([c[:, 1] * OP.RESOLUTION + meta["x_min"],
                             c[:, 0] * OP.RESOLUTION + meta["y_min"]])
            for c in find_contours(component, 0.5)]


def tumour_status(stats: pd.DataFrame) -> dict:
    if "status" not in stats.columns:
        return {}
    return dict(zip(stats.cluster_id.astype(int), stats.status))


def do_slide(slide: str, svs: str, native_mpp: float, tau: float) -> int:
    pred = pd.read_parquet(PRED / f"{slide}.parquet")
    pred = pred[pred.scored].reset_index(drop=True)
    x = pred.x_um.to_numpy(float)
    y = pred.y_um.to_numpy(float)
    is_schwann = pred.schwann_prob.to_numpy(float) > tau

    clusters, member, cluster_grid, meta = regions_for(x, y, is_schwann, prob=pred.schwann_prob.to_numpy(float))


    only_at = os.environ.get("CELL_ONLY_REGIONS_AT")
    if only_at and clusters:
        want = pd.read_csv(only_at)
        wx = want.centroid_x_um.to_numpy(float); wy = want.centroid_y_um.to_numpy(float)
        picked = []
        for c in clusters:


            mem = member == c["cluster_id"]
            if not mem.any():
                continue
            d = np.hypot(wx - float(x[mem].mean()), wy - float(y[mem].mean()))
            if len(d) and d.min() <= 50.0:
                picked.append(c)
        log(f"[{slide[:26]}] drawing only the requested {len(picked)}/{len(clusters)} regions")
        clusters = picked

    if not clusters:
        _clear_stale(slide_dir(slide), slide)
        log(f"[{slide[:26]}] no shippable region -> cleared stale figures")
        return 0


    directory = slide_dir(slide)
    directory.mkdir(parents=True, exist_ok=True)

    _clear_stale(directory, slide)
    stats_path = directory / f"{slide}_cluster_stats.csv"
    status_of = tumour_status(pd.read_csv(stats_path)) if stats_path.exists() else {}


    centroids = []
    for cluster in clusters:
        cts = region_contours_um(cluster_grid, meta, cluster["cluster_id"])
        pts = np.vstack(cts) if cts else np.zeros((1, 2))
        centroids.append(((pts[:, 0].min() + pts[:, 0].max()) / 2,
                          (pts[:, 1].min() + pts[:, 1].max()) / 2))
    centroids = np.asarray(centroids, float)
    ranks = serpentine_order(centroids[:, 0], centroids[:, 1])
    order = [clusters[i] for i in np.argsort(ranks)]
    rank_of = {clusters[i]["cluster_id"]: int(ranks[i]) for i in range(len(clusters))}
    half_px = int(round(WINDOW_UM / 2 / native_mpp))

    canvas = L.Canvas(svs)
    height0, width0 = canvas.H, canvas.W
    tiles = []
    try:
        for cluster in order:
            cid = cluster["cluster_id"]
            rank = rank_of[cid]
            contours = region_contours_um(cluster_grid, meta, cid)
            if not contours:
                continue
            points = np.vstack(contours)
            cxu = (points[:, 0].min() + points[:, 0].max()) / 2
            cyu = (points[:, 1].min() + points[:, 1].max()) / 2
            span = max(points[:, 0].max() - points[:, 0].min(),
                       points[:, 1].max() - points[:, 1].min())

            cxp, cyp = cxu / native_mpp, cyu / native_mpp
            px0 = int(round(cxp - half_px)); py0 = int(round(cyp - half_px))
            px1, py1 = px0 + 2 * half_px, py0 + 2 * half_px


            sx = min(max(0, px0), max(0, width0 - 2 * half_px))
            sy = min(max(0, py0), max(0, height0 - 2 * half_px))
            px0, py0 = sx, sy
            px1, py1 = min(width0, sx + 2 * half_px), min(height0, sy + 2 * half_px)

            crop = canvas.read(py0, px0, py1, px1)
            if crop.size == 0:
                continue
            scale = TILE_PX / (2 * half_px)
            small = cv2.resize(crop, (TILE_PX, TILE_PX), interpolation=cv2.INTER_AREA)
            members = (member == cid) & is_schwann
            tiles.append({
                "rank": rank, "cid": int(cid), "image": small,
                "contours": [np.column_stack([(c[:, 0] / native_mpp - px0) * scale,
                                              (c[:, 1] / native_mpp - py0) * scale])
                             for c in contours],
                "cells": np.column_stack([(x[members] / native_mpp - px0) * scale,
                                          (y[members] / native_mpp - py0) * scale]),
                "area_um2": cluster["area_mm2"] * 1e6,
                "n_schwann": int(cluster["n_schwann"]),
                "shape": cluster["shape_class"],
                "status": status_of.get(int(cid), ""),
                "clipped": span > WINDOW_UM,
            })
            if len(tiles) % 100 == 0:
                log(f"[{slide[:26]}] cut {rank}/{len(order)} tiles")
    finally:
        canvas.close()

    if not tiles:
        log(f"[{slide[:26]}] no usable tile")
        return 0

    per_sheet = MAX_ROWS_PER_SHEET * NCOL
    parts = [tiles[i:i + per_sheet] for i in range(0, len(tiles), per_sheet)]
    n_clipped = sum(t["clipped"] for t in tiles)
    for part_index, part in enumerate(parts, 1):
        nrow = math.ceil(len(part) / NCOL)
        figure, axes = plt.subplots(nrow, NCOL, figsize=(NCOL * TILE_IN,
                                                         nrow * TILE_IN + 1.1), dpi=DPI)
        axes = np.atleast_2d(axes).reshape(nrow, NCOL)
        for ax in axes.ravel():
            ax.axis("off")
        for position, tile in enumerate(part):
            ax = axes[position // NCOL, position % NCOL]
            ax.imshow(tile["image"])
            for contour in tile["contours"]:
                ax.plot(contour[:, 0], contour[:, 1], color=C_REGION, lw=0.9)
            ax.set_xlim(0, TILE_PX); ax.set_ylim(TILE_PX, 0)
            ax.set_title(f"R{tile['rank']} id{tile['cid']}  {tile['area_um2']:.0f}um2  "
                         f"{tile['n_schwann']}c{'  CLIPPED' if tile['clipped'] else ''}",
                         fontsize=5.2, pad=1.4)
        part_tag = f"  part {part_index}/{len(parts)}" if len(parts) > 1 else ""
        figure.suptitle(
            f"{slide}   all {len(tiles)} predicted nerve regions, numbered by position{part_tag}\n"
            f"every tile is the same {WINDOW_UM:.0f} um window at 1.0 um/px, so tiles compare "
            f"directly\nblue = region outline   |  "
            f"MODEL PREDICTION on H&E, no spatial ground truth",
            fontsize=7.5, y=1 - 0.18 / (nrow * TILE_IN + 1.1))
        figure.subplots_adjust(left=0.004, right=0.996, top=1 - 1.0 / (nrow * TILE_IN + 1.1),
                               bottom=0.004, wspace=0.02, hspace=0.16)
        suffix = f"_part{part_index}" if len(parts) > 1 else ""
        base = directory / f"sheet_{slide}{suffix}"
        for ext in ("png",):
            figure.savefig(f"{base}.{ext}")
        plt.close(figure)
        log(f"[{slide[:26]}] sheet{suffix}: {len(part)} tiles, {nrow} rows -> {base}.png")

    log(f"[{slide[:26]}] {len(tiles)} regions, {len(parts)} sheet(s), {n_clipped} clipped")
    return len(tiles)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", default=str(INFERENCE_ROOT / "configs" / COHORT / "run_manifest.tsv"))
    parser.add_argument("--slide")
    parser.add_argument("--slides", nargs="+", help="multiple slides; same argument name as s4 so the three steps can be chained")
    parser.add_argument("--shard", type=int, default=0)
    parser.add_argument("--n-shards", type=int, default=1)
    parser.add_argument("--project", help="process only this one cancer type (outputs are organized by cancer type)")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    _proj = getattr(args, "project", None) or getattr(a, "project", None)
    if not _proj:
        sys.exit("FATAL: --project is required -- thresholds are stored per cancer type, without it we don't know which file to read")

    _forced = os.environ.get("CELL_TAU_FORCE")
    if _forced:
        tau = float(_forced)
        print(f"[tau] forcing {tau:.5f} (experimental)", flush=True)
    else:
      _tf = _tau_file(_proj)
      if not _tf.exists():
          sys.exit(f"FATAL: {_tf} not found; run s4_outputs.py --project {_proj} to calibrate the threshold first")
      tau = float(json.loads(_tf.read_text())["tau"])
    table = pd.read_csv(args.manifest, sep="\t").set_index("slide")
    if getattr(args, 'project', None) or getattr(a, 'project', None):
        _p = getattr(args, 'project', None) or getattr(a, 'project', None)
        table = table[table.project_id == _p] if 'project_id' in table.columns else table

    slides = (args.slides or ([args.slide] if args.slide else list(table.index)[args.shard::args.n_shards]))
    log(f"cohort={COHORT} tau={tau:.6f} slides={len(slides)}")

    for slide in slides:
        directory = slide_dir(slide)
        done = (directory / f"sheet_{slide}.png").exists() or \
               (directory / f"sheet_{slide}_part1.png").exists()
        if done and not args.overwrite:
            log(f"[skip] {slide}")
            continue
        if not (PRED / f"{slide}.parquet").exists():
            log(f"[wait] {slide}: no prediction yet")
            continue
        row = table.loc[slide]
        try:
            do_slide(slide, str(row.svs), float(row.native_mpp), tau)
        except Exception as error:
            log(f"[FAIL] {slide}: {type(error).__name__}: {error}")


if __name__ == "__main__":
    main()
