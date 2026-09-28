#!/usr/bin/env python -u
from __future__ import annotations
import os
PROJECTS_ROOT = os.environ.get("PROJECTS_ROOT", "/data/heid")
from pathlib import Path
_RELEASE = Path(__file__).resolve().parents[3]

import argparse
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Rectangle
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
C_REGION = "#FF2D2D"
N_ROI = 8
ROI_EXPAND = 1.8
MIN_CTX_UM = 320.0


PREFIX = "roi_"
def _clear_stale(directory, slide):
    import glob as _glob
    import os as _os
    for path in _glob.glob(str(directory / f"{PREFIX}{slide}*")):
        _os.remove(path)


def log(message: str) -> None:
    print(message, flush=True)


def scalebar(ax, um_per_px, length_um, label, loc=(0.04, 0.05), color="k"):
    xa, ya = ax.get_xlim(), ax.get_ylim()
    w, h = abs(xa[1] - xa[0]), abs(ya[0] - ya[1])
    x0 = min(xa) + loc[0] * w
    y0 = max(ya) - loc[1] * h if ya[0] > ya[1] else min(ya) + loc[1] * h
    bar = length_um / um_per_px
    ax.add_patch(Rectangle((x0, y0), bar, max(h * 0.006, 1), color=color, zorder=6))
    ax.text(x0 + bar / 2, y0 - h * 0.012, label, ha="center", va="bottom", fontsize=8,
            color=color, zorder=6, bbox=dict(fc="white", ec="none", alpha=.7, pad=1))


def region_contours_um(cluster_grid, meta, cluster_id):
    component = (cluster_grid == cluster_id).astype(float)
    out = []
    for contour in find_contours(component, 0.5):
        out.append(np.column_stack([contour[:, 1] * OP.RESOLUTION + meta["x_min"],
                                    contour[:, 0] * OP.RESOLUTION + meta["y_min"]]))
    return out


def do_slide(slide: str, svs: str, native_mpp: float, tau: float,
             n_roi: int, expand: float, min_ctx: float) -> list[dict]:
    pred = pd.read_parquet(PRED / f"{slide}.parquet")
    pred = pred[pred.scored].reset_index(drop=True)
    x = pred.x_um.to_numpy(float)
    y = pred.y_um.to_numpy(float)
    is_schwann = pred.schwann_prob.to_numpy(float) > tau


    clusters, member, cluster_grid, meta = regions_for(x, y, is_schwann, prob=pred.schwann_prob.to_numpy(float))
    if not clusters:
        _clear_stale(slide_dir(slide), slide)
        log(f"[{slide[:26]}] no shippable region -> cleared stale figures")
        return []


    _clear_stale(slide_dir(slide), slide)


    centroids = []
    for cluster in clusters:
        cts = region_contours_um(cluster_grid, meta, cluster["cluster_id"])
        pts = np.vstack(cts) if cts else np.zeros((1, 2))
        centroids.append(((pts[:, 0].min() + pts[:, 0].max()) / 2,
                          (pts[:, 1].min() + pts[:, 1].max()) / 2))
    centroids = np.asarray(centroids, float)
    ranks = serpentine_order(centroids[:, 0], centroids[:, 1])
    rank_of = {clusters[i]["cluster_id"]: int(ranks[i]) for i in range(len(clusters))}
    order = sorted(clusters, key=lambda c: -c["area_mm2"])[:n_roi]
    directory = slide_dir(slide)
    directory.mkdir(parents=True, exist_ok=True)

    canvas = L.Canvas(svs)
    height0, width0 = canvas.H, canvas.W
    records = []
    try:
        for cluster in order:
            cid = cluster["cluster_id"]
            k = rank_of[cid]
            contours = region_contours_um(cluster_grid, meta, cid)
            if not contours:
                log(f"[{slide[:26]}] R{k} cid={cid} no contour, skip")
                continue
            points = np.vstack(contours)
            x0u, x1u = points[:, 0].min(), points[:, 0].max()
            y0u, y1u = points[:, 1].min(), points[:, 1].max()
            cxu, cyu = (x0u + x1u) / 2, (y0u + y1u) / 2
            wu = max((x1u - x0u) * expand, min_ctx)
            hu = max((y1u - y0u) * expand, min_ctx)
            px0 = int(max(0, round((cxu - wu / 2) / native_mpp)))
            py0 = int(max(0, round((cyu - hu / 2) / native_mpp)))
            px1 = int(min(width0, round((cxu + wu / 2) / native_mpp)))
            py1 = int(min(height0, round((cyu + hu / 2) / native_mpp)))

            crop = canvas.read(py0, px0, py1, px1)
            if crop.size == 0:
                log(f"[{slide[:26]}] R{k} empty crop, skip")
                continue
            blank_frac = float(np.mean(crop > 240))
            if blank_frac > 0.995:


                log(f"[{slide[:26]}] R{k} WARN crop is {100*blank_frac:.1f}% white "
                    f"(bbox y{py0}:{py1} x{px0}:{px1})")


            members = (member == cid) & is_schwann
            in_footprint = int((member == cid).sum())


            aspect = crop.shape[0] / crop.shape[1]
            panel_w = 7.6
            figure, axes = plt.subplots(1, 2, figsize=(2 * panel_w + 0.6,
                                                       min(max(panel_w * aspect, 4.0), 13.0)),
                                        dpi=170)
            for ax, overlay in zip(axes, (False, True)):
                ax.imshow(crop)
                if overlay:
                    for contour in contours:
                        ax.plot(contour[:, 0] / native_mpp - px0,
                                contour[:, 1] / native_mpp - py0, color=C_REGION, lw=2.0)
                    ax.scatter(x[members] / native_mpp - px0, y[members] / native_mpp - py0,
                               s=13, facecolors="none", edgecolors=C_SCHWANN, linewidths=0.9)
                ax.set_xlim(0, crop.shape[1])
                ax.set_ylim(crop.shape[0], 0)
                ax.axis("off")
                ax.set_title("H&E (clean)" if not overlay else
                             f"+ region outline & its {int(members.sum())} predicted Schwann cells",
                             fontsize=10)
                scalebar(ax, native_mpp, 100, "100 um")
            figure.suptitle(
                f"{slide[:34]}  ROI R{k} (region_id={cid})   "
                f"area={cluster['area_mm2']*1e6:.0f} um^2   "
                f"{cluster['n_schwann']} predicted Schwann cells "
                f"({in_footprint} cells of any type inside the outline)   "
                f"shape={cluster['shape_class']} aspect={cluster['aspect_ratio']:.2f}   "
                f"conf={cluster['confidence']}\n"
                f"native {native_mpp} um/px, crop {crop.shape[1]}x{crop.shape[0]} px = "
                f"{crop.shape[1]*native_mpp:.0f}x{crop.shape[0]*native_mpp:.0f} um   |  "
                f"MODEL PREDICTION on H&E, no spatial ground truth", fontsize=10)
            figure.tight_layout(rect=(0, 0, 1, 0.94))
            base = directory / f"roi_{slide}_r{k}"
            for ext in ("png",):
                figure.savefig(f"{base}.{ext}", bbox_inches="tight")
            plt.close(figure)

            records.append({
                "roi": f"R{k}", "slide": slide, "region_id": int(cid),
                "area_um2": round(cluster["area_mm2"] * 1e6, 1),
                "n_schwann": int(cluster["n_schwann"]),
                "n_schwann_plotted": int(members.sum()),
                "n_cells_any_type_in_region": in_footprint,
                "shape_class": cluster["shape_class"],
                "aspect_ratio": round(cluster["aspect_ratio"], 3),
                "solidity": round(cluster["solidity"], 3),
                "confidence": cluster["confidence"],
                "centroid_x_um": round(float(cxu), 1), "centroid_y_um": round(float(cyu), 1),
                "crop_x0_px": px0, "crop_y0_px": py0, "crop_x1_px": px1, "crop_y1_px": py1,
                "crop_w_um": round((px1 - px0) * native_mpp, 1),
                "crop_h_um": round((py1 - py0) * native_mpp, 1),
                "native_mpp": native_mpp, "blank_frac": round(blank_frac, 4),
            })
            log(f"[{slide[:26]}] R{k} cid={cid} area={cluster['area_mm2']*1e6:.0f}um2 "
                f"cells={cluster['n_schwann']} crop={crop.shape[1]}x{crop.shape[0]}px")
    finally:
        canvas.close()

    if records:
        pd.DataFrame(records).to_csv(directory / f"roi_{slide}_index.csv", index=False)
    return records


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", default=str(INFERENCE_ROOT / "configs" / COHORT / "run_manifest.tsv"))
    parser.add_argument("--slide")
    parser.add_argument("--slides", nargs="+", help="multiple slides; same argument name as s4 so the three steps can be chained")
    parser.add_argument("--shard", type=int, default=0)
    parser.add_argument("--n-shards", type=int, default=1)
    parser.add_argument("--project", help="process only this one cancer type (outputs are organized by cancer type)")
    parser.add_argument("--n-roi", type=int, default=N_ROI)
    parser.add_argument("--roi-expand", type=float, default=ROI_EXPAND)
    parser.add_argument("--min-ctx-um", type=float, default=MIN_CTX_UM)
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
    log(f"cohort={COHORT} tau={tau:.6f} slides={len(slides)} n_roi={args.n_roi}")

    for slide in slides:
        directory = slide_dir(slide)
        if not args.overwrite and (directory / f"roi_{slide}_index.csv").exists():
            log(f"[skip] {slide}")
            continue
        if not (PRED / f"{slide}.parquet").exists():
            log(f"[wait] {slide}: no prediction yet")
            continue
        row = table.loc[slide]
        try:
            do_slide(slide, str(row.svs), float(row.native_mpp), tau,
                     args.n_roi, args.roi_expand, args.min_ctx_um)
        except Exception as error:
            log(f"[FAIL] {slide}: {type(error).__name__}: {error}")


if __name__ == "__main__":
    main()
