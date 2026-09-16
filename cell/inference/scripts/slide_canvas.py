#!/usr/bin/env python3
import os
import json
from functools import lru_cache
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

import det_config as C

PXSIZE_UM = C.PXSIZE_UM
UM_PER_PX = PXSIZE_UM
PX_PER_UM = 1.0 / PXSIZE_UM


def _target():
    cancer = os.environ.get("DET_CANCER", "pancancer")
    return (None if cancer.lower() in ("pancancer", "pan", "all", "") else cancer.upper())


def _include_flagged():
    return os.environ.get("DET_INCLUDE_FLAGGED", "0") == "1"


@lru_cache(maxsize=1)
def _sample_index():
    rows = C.load_manifest(cancer=None, clean_only=False, include_flagged=True)
    return {r["sample"]: r for r in rows}


def usable_samples():
    m = C.load_manifest(cancer=_target(), clean_only=True, include_flagged=_include_flagged())
    return [{"sample": r["sample"], "ncc": r["ncc"], "test_fold": r["fold"]} for r in m]


def find_he(sample):
    r = _sample_index().get(sample)
    if r is None:
        return None, None
    return C.resolve_he(r["cohort"], r["cancer"], sample)


def _cancer_of(sample):
    r = _sample_index().get(sample)
    return r["cancer"] if r else None


class Canvas:

    def __init__(self, path):
        import tifffile
        import zarr
        self.path = path
        self._tif = tifffile.TiffFile(path)
        store = self._tif.series[0].aszarr()
        zg = zarr.open(store, mode="r")
        self.z = zg["0"] if isinstance(zg, zarr.hierarchy.Group) else zg
        self.H, self.W = int(self.z.shape[0]), int(self.z.shape[1])

    def read(self, y0, x0, y1, x1):
        y0 = max(0, int(y0)); x0 = max(0, int(x0))
        y1 = min(self.H, int(y1)); x1 = min(self.W, int(x1))
        if y1 <= y0 or x1 <= x0:
            return np.zeros((0, 0, 3), np.uint8)
        win = np.asarray(self.z[y0:y1, x0:x1])
        if win.ndim == 2:
            win = np.repeat(win[:, :, None], 3, axis=2)
        return win[:, :, :3].astype(np.uint8, copy=False)

    def close(self):
        try:
            self._tif.close()
        except Exception:
            pass


def _read_parquet_rg(path, columns):
    pf = pq.ParquetFile(path)
    tabs = [pf.read_row_group(i, columns=list(columns)) for i in range(pf.num_row_groups)]
    return pa.concat_tables(tabs).to_pandas()


def read_gt_centroids(sample):
    path = C.xcache_path(_cancer_of(sample), sample, "cells")
    df = _read_parquet_rg(path, ["cell_id", "x_centroid", "y_centroid"])
    df["x_px"] = df["x_centroid"].to_numpy() * PX_PER_UM
    df["y_px"] = df["y_centroid"].to_numpy() * PX_PER_UM
    return df[["cell_id", "x_px", "y_px"]]


def read_cell_types(sample):
    import pandas as pd
    path = os.path.join(C.HEID, "data", "cell_tables", f"{sample}.parquet")
    if not os.path.exists(path):
        return None
    return pd.read_parquet(path)


def greedy_match(gt_xy, pred_xy, tau_px, k=5):
    from scipy.spatial import cKDTree
    n_pred = len(pred_xy)
    n_gt = len(gt_xy)
    pred_to_gt = np.full(n_pred, -1, dtype=np.int64)
    pred_dist = np.full(n_pred, np.inf, dtype=np.float64)
    if n_pred == 0 or n_gt == 0:
        return {"pred_to_gt": pred_to_gt, "pred_dist": pred_dist}
    tree = cKDTree(gt_xy)
    kk = min(k, n_gt)
    dists, idxs = tree.query(pred_xy, k=kk, distance_upper_bound=tau_px)
    if kk == 1:
        dists = dists[:, None]; idxs = idxs[:, None]
    finite = np.isfinite(dists)
    pr, rk = np.nonzero(finite)
    if pr.size == 0:
        return {"pred_to_gt": pred_to_gt, "pred_dist": pred_dist}
    cand_d = dists[pr, rk]
    cand_g = idxs[pr, rk]
    order = np.argsort(cand_d, kind="stable")
    pr = pr[order]; cand_g = cand_g[order]; cand_d = cand_d[order]
    gt_taken = np.zeros(n_gt, dtype=bool)
    for p, g, d in zip(pr, cand_g, cand_d):
        if pred_to_gt[p] != -1:
            continue
        if gt_taken[g]:
            continue
        pred_to_gt[p] = g
        pred_dist[p] = d
        gt_taken[g] = True
    return {"pred_to_gt": pred_to_gt, "pred_dist": pred_dist}


def tissue_fraction(rgb, white_thresh=220):
    if rgb.size == 0:
        return 0.0
    g = rgb.reshape(-1, rgb.shape[-1]).mean(axis=1)
    return float((g < white_thresh).mean())
