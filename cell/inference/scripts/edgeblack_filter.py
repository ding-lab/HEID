#!/usr/bin/env python
from __future__ import annotations
import os
PROJECTS_ROOT = os.environ.get("PROJECTS_ROOT", "/data/heid")

import argparse
from dataclasses import dataclass, asdict
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.ndimage import (binary_closing, binary_dilation, binary_erosion,
                           binary_fill_holes, distance_transform_edt, label,
                           uniform_filter)

INFERENCE_ROOT = Path(PROJECTS_ROOT + "/cell/inference")


@dataclass(frozen=True)
class Params:

    thumb_mpp: float = 8.0
    tissue_white: float = 235.0
    speckle_um2: float = 500.0
    max_hole_um2: float = 2e5
    min_tissue_um2: float = 5e5

    edge_margin_um: float = 400.0
    border_near_um: float = 60.0

    black_max: float = 90.0
    black_sat_max: float = 40.0
    very_dark_max: float = 50.0
    blue_mean_max: float = 170.0
    blue_sat_min: float = 25.0
    blue_min: float = 15.0
    blue_offset: float = 35.0
    grey_mean_max: float = 180.0
    grey_sat_max: float = 20.0

    min_blob_um2: float = 300.0
    dilate_um: float = 100.0
    win_um: float = 48.0
    frac_max: float = 0.15
    touch_um: float = 15.0
    touch_anywhere: bool = False

    region_kill_frac: float = 0.8
    region_zone_frac: float = 0.35
    region_touch_frac: float = 0.5


@dataclass
class Masks:
    mpp: float
    tissue: np.ndarray
    raw: np.ndarray
    band: np.ndarray
    rim: np.ndarray
    black: np.ndarray
    blue: np.ndarray
    grey: np.ndarray
    zone: np.ndarray
    frac: np.ndarray
    touch_px: np.ndarray
    blue_thr: float


@dataclass
class FilterResult:
    kill: np.ndarray
    in_band: np.ndarray
    hit_zone: np.ndarray
    hit_frac: np.ndarray
    hit_touch: np.ndarray
    drop: dict
    masks: Masks
    diag: dict


def disk(r: int) -> np.ndarray:
    n = np.arange(-r, r + 1)
    xs, ys = np.meshgrid(n, n)
    return (xs**2 + ys**2) <= r**2


def thumbnail_from_svs(svs: str, native_mpp: float, level0_w: int, p: Params = Params()):
    import tifffile
    import cv2
    target_w = max(int(round(int(level0_w) * float(native_mpp) / p.thumb_mpp)), 256)
    with tifffile.TiffFile(svs) as tf:
        series = tf.series[0]
        levels = list(getattr(series, "levels", [series]))
        finer = [lv for lv in levels if lv.shape[1] >= target_w]
        level = min(finer, key=lambda lv: lv.shape[1]) if finer else max(levels, key=lambda lv: lv.shape[1])
        img = level.asarray()
    if img.ndim == 2:
        img = np.stack([img] * 3, -1)
    img = img[..., :3].astype(np.uint8)
    if img.shape[1] != target_w:
        h = int(round(img.shape[0] * target_w / img.shape[1]))
        img = cv2.resize(img, (target_w, h), interpolation=cv2.INTER_AREA)
    return img, float(native_mpp) * float(level0_w) / img.shape[1]


def _um(px_area: np.ndarray, mpp: float) -> np.ndarray:
    return px_area * mpp * mpp


def build_masks(img: np.ndarray, mpp: float, p: Params = Params()) -> Masks:
    f = img.astype(np.float32)
    mean = f.mean(axis=2)
    sat = f.max(axis=2) - f.min(axis=2)
    br = f[..., 2] - f[..., 0]
    r_touch = max(int(round(p.touch_um / mpp)), 1)


    raw = mean < p.tissue_white
    raw_f = raw.copy()
    rh = binary_fill_holes(raw) & ~raw
    lb, n = label(rh)
    if n:
        sz = np.bincount(lb.ravel()); sz[0] = 0
        raw_f |= np.isin(lb, np.flatnonzero((sz > 0) & (_um(sz, mpp) < p.speckle_um2)))
    near_border = raw_f & (distance_transform_edt(raw_f) * mpp < p.border_near_um)


    tissue = binary_closing(raw, structure=disk(2))
    holes = binary_fill_holes(tissue) & ~tissue
    lb, n = label(holes)
    if n:
        sz = np.bincount(lb.ravel()); sz[0] = 0
        tissue |= np.isin(lb, np.flatnonzero((sz > 0) & (_um(sz, mpp) < p.max_hole_um2)))
    lb, n = label(tissue)
    if n:
        sz = np.bincount(lb.ravel()); sz[0] = 0
        tissue = np.isin(lb, np.flatnonzero(_um(sz, mpp) >= p.min_tissue_um2))
    rim = tissue & ~binary_erosion(tissue, structure=disk(max(int(round(p.edge_margin_um / mpp)), 1)))
    band = rim | near_border | ~tissue


    black = ((mean < p.black_max) & (sat < p.black_sat_max)) | (mean < p.very_dark_max)
    dark_chrom = (mean < 110.0) & (sat > 40.0)
    blue_thr = max(p.blue_min, float(np.median(br[dark_chrom])) + p.blue_offset) if dark_chrom.any() else p.blue_min
    blue = (mean < p.blue_mean_max) & (sat > p.blue_sat_min) & (br > blue_thr)
    grey = (mean < p.grey_mean_max) & (sat < p.grey_sat_max)
    artifact = black | blue | grey


    lb, n = label(black)
    if n:
        sz = np.bincount(lb.ravel()); sz[0] = 0
        blob = np.isin(lb, np.flatnonzero(_um(sz, mpp) >= p.min_blob_um2))
    else:
        blob = black
    zone = binary_dilation(blob, structure=disk(max(int(round(p.dilate_um / mpp)), 1)))
    frac = uniform_filter(black.astype(np.float32), size=max(int(round(p.win_um / mpp)), 1))
    touch_px = binary_dilation(artifact, structure=disk(r_touch))
    return Masks(mpp=mpp, tissue=tissue, raw=raw_f, band=band, rim=rim, black=black, blue=blue,
                 grey=grey, zone=zone, frac=frac, touch_px=touch_px, blue_thr=blue_thr)


def cell_index(x_um: np.ndarray, y_um: np.ndarray, m: Masks):
    ix = np.clip((x_um / m.mpp).astype(int), 0, m.tissue.shape[1] - 1)
    iy = np.clip((y_um / m.mpp).astype(int), 0, m.tissue.shape[0] - 1)
    return iy, ix


def filter_slide(pred: pd.DataFrame, img: np.ndarray, mpp: float, member: np.ndarray,
                 cluster_ids, p: Params = Params(), tau: float | None = None) -> FilterResult:
    m = build_masks(img, mpp, p)
    x = pred.x_um.to_numpy(float); y = pred.y_um.to_numpy(float)
    iy, ix = cell_index(x, y, m)
    in_band = m.band[iy, ix]
    hit_zone = m.zone[iy, ix]
    hit_frac = m.frac[iy, ix] > p.frac_max
    hit_touch = m.touch_px[iy, ix]
    kill = in_band & (hit_zone | hit_frac | hit_touch)
    if p.touch_anywhere:
        kill = kill | hit_touch

    zone_b = hit_zone & in_band
    touch_b = hit_touch if p.touch_anywhere else (hit_touch & in_band)
    drop = {}
    for cid in cluster_ids:
        mm = member == cid
        if not mm.any():
            drop[int(cid)] = True
            continue
        drop[int(cid)] = bool(kill[mm].mean() >= p.region_kill_frac
                              or zone_b[mm].mean() >= p.region_zone_frac
                              or touch_b[mm].mean() >= p.region_touch_frac)
    diag = {"n_scored": int(len(pred)), "n_killed": int(kill.sum()),
            "n_regions": int(len(cluster_ids)), "n_dropped": int(sum(drop.values())),
            "blue_thr": round(m.blue_thr, 1)}
    if tau is not None:
        pos = pred.schwann_prob.to_numpy(float) > tau
        diag.update({"n_pos": int(pos.sum()), "n_pos_killed": int((kill & pos).sum())})
    return FilterResult(kill=kill, in_band=in_band, hit_zone=hit_zone, hit_frac=hit_frac,
                        hit_touch=hit_touch, drop=drop, masks=m, diag=diag)


def apply_to_predictions(full: pd.DataFrame, scored_index: np.ndarray, kill: np.ndarray,
                         member: np.ndarray, drop: dict) -> pd.DataFrame:
    out = full.copy()
    dropped_ids = np.array([cid for cid, d in drop.items() if d], dtype=member.dtype)
    zero = kill | np.isin(member, dropped_ids)
    out.loc[scored_index[zero], "schwann_prob"] = 0.0
    return out


def production_membership(project: str, slide: str, pred_scored: pd.DataFrame):
    d = INFERENCE_ROOT / project / "outputs" / slide
    cells = pd.read_csv(d / f"{slide}_cells.csv", usecols=["cell_id", "schwann_cluster"])
    stats = pd.read_csv(d / f"{slide}_cluster_stats.csv")
    mem = pred_scored[["cell_id"]].merge(cells, on="cell_id", how="left").schwann_cluster
    return mem.fillna(0).astype(int).to_numpy(), stats


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--slide", required=True)
    ap.add_argument("--project", required=True, help="production project dir, e.g. TCGA-SKCM")
    ap.add_argument("--cohort", default="tcga_pancancer", help="predictions cohort under cell/inference/")
    ap.add_argument("--manifest", default=str(INFERENCE_ROOT / "configs/tcga_pancancer/run_manifest.tsv"))
    ap.add_argument("--thumb-npz", default=None, help="cached thumbnail (img, mpp); else read the SVS")
    ap.add_argument("--out-pred", required=True, help="filtered predictions parquet")
    ap.add_argument("--out-ledger", required=True, help="per-region ledger csv (original table + dropped flag)")
    ap.add_argument("--tau", type=float, default=None, help="only for the diagnostics line")
    a = ap.parse_args()
    p = Params()

    full = pd.read_parquet(INFERENCE_ROOT / a.cohort / "data" / "predictions" / f"{a.slide}.parquet")
    scored_index = np.flatnonzero(full.scored.to_numpy(bool))
    pred = full.iloc[scored_index].reset_index(drop=True)
    member, stats = production_membership(a.project, a.slide, pred)
    if a.thumb_npz:
        z = np.load(a.thumb_npz); img, mpp = z["img"], float(z["mpp"])
    else:
        row = pd.read_csv(a.manifest, sep="\t").set_index("slide").loc[a.slide]
        img, mpp = thumbnail_from_svs(str(row.svs), float(row.native_mpp), int(row.level0_W), p)
    res = filter_slide(pred, img, mpp, member, stats.cluster_id.astype(int).tolist(), p, tau=a.tau)

    Path(a.out_pred).parent.mkdir(parents=True, exist_ok=True)
    apply_to_predictions(full, scored_index, res.kill, member, res.drop).to_parquet(a.out_pred)
    ledger = stats.copy()
    ledger["dropped"] = ledger.cluster_id.astype(int).map(res.drop).astype(bool)
    Path(a.out_ledger).parent.mkdir(parents=True, exist_ok=True)
    ledger.to_csv(a.out_ledger, index=False)
    d = res.diag
    print(f"[{a.slide}] killed {d['n_killed']}/{d['n_scored']} cells"
          + (f" (pos {d['n_pos_killed']}/{d['n_pos']})" if "n_pos" in d else "")
          + f" | regions {d['n_regions']} = kept {d['n_regions'] - d['n_dropped']} + dropped {d['n_dropped']}"
          + f" | blue_thr {d['blue_thr']}", flush=True)


if __name__ == "__main__":
    main()
