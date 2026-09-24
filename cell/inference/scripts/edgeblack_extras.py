#!/usr/bin/env python
from __future__ import annotations
import os
PROJECTS_ROOT = os.environ.get("PROJECTS_ROOT", "/data/heid")
from pathlib import Path
_RELEASE = Path(__file__).resolve().parents[3]

import argparse
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd

INFERENCE_ROOT = Path(PROJECTS_ROOT + "/cell/inference")
WINDOW_UM, TILE_PX, NCOL, TILE_IN, DPI = 400.0, 400, 6, 2.0, 200
C_DROP = "#D62728"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--slide", required=True)
    ap.add_argument("--src-project", required=True)
    ap.add_argument("--dst-project", required=True)
    ap.add_argument("--manifest", default=str(INFERENCE_ROOT / "configs/tcga_pancancer/run_manifest.tsv"))
    a = ap.parse_args()
    S = a.slide

    os.environ.pop("CELL_COHORT", None)
    sys.path.insert(0, str(_RELEASE / "cell/inference/scripts"))
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    from matplotlib.lines import Line2D
    from skimage.measure import find_contours
    import cv2
    import s4_outputs as s4
    import s6_region_sheet as s6m
    import edgeblack_filter as ef

    row = pd.read_csv(a.manifest, sep="\t").set_index("slide").loc[S]
    tau = float(pd.read_json(INFERENCE_ROOT / a.src_project / "summary" / "operating_point.json", typ="series")["tau"])
    ledger = pd.read_csv(INFERENCE_ROOT / a.dst_project / "ledger" / f"{S}_regions.csv")
    dropped_ids = set(ledger.loc[ledger.dropped.astype(bool), "cluster_id"].astype(int))
    outdir = INFERENCE_ROOT / a.dst_project / "outputs" / S
    outdir.mkdir(parents=True, exist_ok=True)

    p = ef.Params()
    img, mpp = ef.thumbnail_from_svs(str(row.svs), float(row.native_mpp), int(row.level0_W), p)
    m = ef.build_masks(img, mpp, p)
    span_x, span_y = img.shape[1] * mpp, img.shape[0] * mpp

    pred = pd.read_parquet(INFERENCE_ROOT / "tcga_pancancer" / "data" / "predictions" / f"{S}.parquet")
    pred = pred[pred.scored].reset_index(drop=True)
    x = pred.x_um.to_numpy(float); y = pred.y_um.to_numpy(float)
    prob = pred.schwann_prob.to_numpy(float)
    is_schwann = prob > tau
    clusters, member, cluster_grid, meta = s4.regions_for(x, y, is_schwann, prob=prob)
    is_drop = [int(c["cluster_id"]) in dropped_ids for c in clusters]


    img_ext = [0, span_x, span_y, 0]
    band_rgba = np.zeros((*m.band.shape, 4), np.float32); band_rgba[m.band & m.tissue] = (0.12, 0.45, 0.90, 0.18)
    kill_px = m.band & (m.zone | (m.frac > p.frac_max) | m.touch_px)
    kill_rgba = np.zeros((*kill_px.shape, 4), np.float32); kill_rgba[kill_px] = (0.95, 0.55, 0.05, 0.60)
    figure, ax = plt.subplots(figsize=(span_x / s4.UM_PER_INCH + 3.0, max(span_y / s4.UM_PER_INCH, 3.0)))
    ax.imshow(img, extent=img_ext, zorder=1)
    ax.imshow(band_rgba, extent=img_ext, interpolation="nearest", zorder=2)
    ax.imshow(kill_rgba, extent=img_ext, interpolation="nearest", zorder=2)
    for c, dr in zip(clusters, is_drop):
        comp = cluster_grid == c["cluster_id"]
        for ct in find_contours(comp.astype(float), 0.5):
            ax.plot(ct[:, 1] * s4.OP.RESOLUTION + meta["x_min"], ct[:, 0] * s4.OP.RESOLUTION + meta["y_min"],
                    color=C_DROP if dr else s4.C_SCHWANN, lw=1.3, alpha=0.9, zorder=4)
    s4.place_region_labels(ax, cluster_grid, clusters, meta, span_x, span_y)
    ax.set_xlim(0, span_x); ax.set_ylim(span_y, 0); ax.set_aspect("equal")
    ax.set_xlabel("X (µm)"); ax.set_ylabel("Y (µm)")
    ax.set_title(f"{S}{s4.schwann_tag(len(clusters))}  {len(clusters)} predicted nerve regions from "
                 f"{int(is_schwann.sum()):,} Schwann cells; edge-artifact filter dropped {sum(is_drop)}\n"
                 f"pan-cancer Cell probability + nerve operator, {s4.tau_label(tau)}  |  MODEL PREDICTION",
                 fontsize=10, fontweight="bold")
    band_lbl = f"edge band ({p.edge_margin_um:.0f} µm rim + {p.border_near_um:.0f} µm of any border)"
    ax.legend(handles=[Line2D([], [], color=s4.C_SCHWANN, lw=1.3, label="called nerve region (kept)"),
                       Line2D([], [], color=C_DROP, lw=1.3, label="nerve region dropped by filter"),
                       mpatches.Patch(color=(0.12, 0.45, 0.90, 0.18), label=band_lbl),
                       mpatches.Patch(color=(0.95, 0.55, 0.05, 0.60), label="band ∩ artifact kill zone")],
              loc="upper left", bbox_to_anchor=(1.01, 1.0), fontsize=9, borderaxespad=0.0)
    plt.tight_layout()
    figure.savefig(outdir / f"{S}_step2_edge_black.png", dpi=200, bbox_inches="tight")
    plt.close(figure)


    sheet_path = outdir / f"sheet_{S}_DROPPED_vs_original.png"
    if not dropped_ids:
        if sheet_path.exists():
            sheet_path.unlink()
        print(f"[{S}] step2 written; 0 dropped regions -> no DROPPED sheet", flush=True)
        return
    native_mpp = float(row.native_mpp)
    centroids = []
    for c in clusters:
        cts = s6m.region_contours_um(cluster_grid, meta, c["cluster_id"])
        pts = np.vstack(cts) if cts else np.zeros((1, 2))
        centroids.append(((pts[:, 0].min() + pts[:, 0].max()) / 2, (pts[:, 1].min() + pts[:, 1].max()) / 2))
    centroids = np.asarray(centroids, float)
    ranks = s6m.serpentine_order(centroids[:, 0], centroids[:, 1])
    order = [i for i in np.argsort(ranks) if is_drop[i]]
    half_px = int(round(WINDOW_UM / 2 / native_mpp))
    canvas = s6m.L.Canvas(str(row.svs))
    H0, W0 = canvas.H, canvas.W
    tiles = []
    try:
        for i in order:
            c = clusters[i]; cid = c["cluster_id"]
            contours = s6m.region_contours_um(cluster_grid, meta, cid)
            if not contours:
                continue
            pts = np.vstack(contours)
            cxu = (pts[:, 0].min() + pts[:, 0].max()) / 2; cyu = (pts[:, 1].min() + pts[:, 1].max()) / 2
            span = max(pts[:, 0].max() - pts[:, 0].min(), pts[:, 1].max() - pts[:, 1].min())
            sx = min(max(0, int(round(cxu / native_mpp - half_px))), max(0, W0 - 2 * half_px))
            sy = min(max(0, int(round(cyu / native_mpp - half_px))), max(0, H0 - 2 * half_px))
            crop = canvas.read(sy, sx, min(H0, sy + 2 * half_px), min(W0, sx + 2 * half_px))
            if crop.size == 0:
                continue
            scale = TILE_PX / (2 * half_px)
            tiles.append(dict(rank=int(ranks[i]), cid=int(cid),
                              image=cv2.resize(crop, (TILE_PX, TILE_PX), interpolation=cv2.INTER_AREA),
                              contours=[np.column_stack([(ct[:, 0] / native_mpp - sx) * scale,
                                                         (ct[:, 1] / native_mpp - sy) * scale]) for ct in contours],
                              area_um2=c["area_mm2"] * 1e6, n_schwann=int(c["n_schwann"]), clipped=span > WINDOW_UM))
    finally:
        canvas.close()
    nrow = max(math.ceil(len(tiles) / NCOL), 1)
    figure, axes = plt.subplots(nrow, NCOL, figsize=(NCOL * TILE_IN, nrow * TILE_IN + 1.1), dpi=DPI)
    axes = np.atleast_2d(axes).reshape(nrow, NCOL)
    for ax in axes.ravel():
        ax.axis("off")
    for pos, t in enumerate(tiles):
        ax = axes[pos // NCOL, pos % NCOL]
        ax.imshow(t["image"])
        for ct in t["contours"]:
            ax.plot(ct[:, 0], ct[:, 1], color=C_DROP, lw=0.9)
        ax.set_xlim(0, TILE_PX); ax.set_ylim(TILE_PX, 0)
        ax.set_title(f"R{t['rank']} id{t['cid']}  {t['area_um2']:.0f}um2  {t['n_schwann']}c"
                     f"{'  CLIPPED' if t['clipped'] else ''}", fontsize=5.2, pad=1.4)
    figure.suptitle(f"{S}   DROPPED by edge-artifact filter: {len(tiles)} of {len(clusters)} predicted nerve regions, "
                    f"numbered by position\nevery tile is the same 400 um window at 1.0 um/px, so tiles compare directly\n"
                    f"red = region outline   |  MODEL PREDICTION on H&E, no spatial ground truth",
                    fontsize=8.5, fontweight="bold")
    figure.savefig(sheet_path, bbox_inches="tight")
    plt.close(figure)
    print(f"[{S}] step2 + DROPPED sheet written ({len(tiles)}/{len(clusters)} dropped)", flush=True)


if __name__ == "__main__":
    main()
