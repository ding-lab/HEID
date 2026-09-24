import os
from pathlib import Path
_RELEASE = Path(__file__).resolve().parents[3]
import sys

import numpy as np

NERVE_ROOT = str(_RELEASE / "nerve/prediction")
sys.path.insert(0, str(_RELEASE / "nerve/prediction/scripts"))
import region_operator as OP

PANEL = "5k"
MIN_CONFIDENCE = os.environ.get("NERVE_CONFIDENCE", "high")


FILL_ALL = os.environ.get("NERVE_FILL_ALL", "") == "1"
FILL_TOP = 8
FILL_DISK_UM = 30.0


SPLIT_SOLIDITY = float(os.environ["NERVE_SPLIT_SOLIDITY"]) if os.environ.get("NERVE_SPLIT_SOLIDITY") else None


SPLIT_MAX_MM2 = float(os.environ.get("NERVE_SPLIT_MAX_MM2", "0.5"))


def split_low_solidity(grid, clusters, x_um, y_um, is_schwann, meta, solidity_max):
    from scipy.ndimage import distance_transform_edt
    from scipy.spatial import cKDTree
    from skimage.feature import peak_local_max
    from skimage.measure import regionprops
    from skimage.segmentation import watershed
    R = OP.RESOLUTION
    ny, nx = grid.shape
    xs = np.asarray(x_um, float)[np.asarray(is_schwann, bool)]
    ys = np.asarray(y_um, float)[np.asarray(is_schwann, bool)]
    s_ix = np.clip(((xs - meta["x_min"]) / R).astype(int), 0, nx - 1)
    s_iy = np.clip(((ys - meta["y_min"]) / R).astype(int), 0, ny - 1)
    out, n_split = [], 0
    for c in clusters:
        cid = int(c["cluster_id"]); comp = grid == cid
        sol = float(c.get("solidity", 1.0))
        if sol >= solidity_max or not comp.any() or float(c.get("area_mm2", 0.0)) > SPLIT_MAX_MM2:
            out.append((comp, dict(c))); continue
        dist = distance_transform_edt(comp)
        peaks = peak_local_max(dist, min_distance=4, labels=comp.astype(int), exclude_border=False)
        if len(peaks) < 2:
            out.append((comp, dict(c))); continue
        markers = np.zeros_like(grid, dtype=np.int32)
        for k, (py, px) in enumerate(peaks, 1):
            markers[py, px] = k
        lab = watershed(-dist, markers, mask=comp)
        parts = []
        for part_id in range(1, len(peaks) + 1):
            pm = lab == part_id
            if not pm.any():
                continue
            in_part = pm[s_iy, s_ix]
            n_in = int(in_part.sum())
            if n_in < OP.MIN_SCHWANN_CELLS:
                continue
            props = regionprops(pm.astype(int))
            if not props:
                continue
            r = props[0]
            area_mm2 = int(pm.sum()) * R ** 2 / 1e6
            areal_density = n_in / area_mm2 if area_mm2 > 0 else 0.0
            major_um = float(r.axis_major_length) * R; minor_um = float(r.axis_minor_length) * R
            aspect = (major_um / minor_um) if minor_um > 0 else (999.0 if major_um > 0 else 1.0)
            solidity = float(r.solidity)
            linear_density = n_in / (major_um / 1000.0) if major_um > 0 else 0.0
            memb = np.column_stack([xs[in_part], ys[in_part]])
            median_nn = float(np.median(cKDTree(memb).query(memb, k=2)[0][:, 1])) if len(memb) >= 2 else float("nan")
            elongated = aspect >= OP.MIN_ASPECT
            not_loose = median_nn <= OP.MAX_MEDIAN_NN_UM
            dense_enough = areal_density >= OP.required_areal_density(area_mm2)
            if elongated:
                passed = (minor_um >= OP.MIN_MINOR_UM and major_um >= OP.MIN_MAJOR_UM
                          and linear_density >= OP.MIN_LINEAR_DENSITY and dense_enough
                          and area_mm2 >= OP.MIN_AREA_ELONG and not_loose)
            else:
                passed = (dense_enough and solidity >= OP.MIN_SOLIDITY
                          and area_mm2 >= OP.MIN_AREA_COMPACT and not_loose)
            if not passed:
                continue
            parts.append((pm, dict(n_schwann=n_in, area_mm2=area_mm2, areal_density=areal_density,
                                   aspect_ratio=aspect, solidity=solidity, linear_density=linear_density,
                                   median_nn_um=median_nn, shape_class="elongated" if elongated else "compact",
                                   confidence="high" if n_in >= OP.HIGH_CONF_CELLS else "low",
                                   split_from=cid)))
        n_split += 1
        out.extend(parts)
    new_grid = np.zeros_like(grid)
    new_clusters = []
    for i, (m, c) in enumerate(out, 1):
        c["cluster_id"] = i
        new_grid[m] = i
        new_clusters.append(c)
    return new_grid, new_clusters, n_split


def regions_for(x_um, y_um, is_schwann, panel=PANEL, min_confidence=None, prob=None):
    conf = MIN_CONFIDENCE if min_confidence is None else min_confidence
    grid, clusters, meta = OP.run_operator(x_um, y_um, is_schwann, panel=panel)
    if SPLIT_SOLIDITY is not None and clusters:
        grid, clusters, n_split = split_low_solidity(grid, clusters, x_um, y_um, is_schwann, meta, SPLIT_SOLIDITY)
        meta["n_split"] = n_split
    member = (OP.membership(grid, x_um, y_um, meta["x_min"], meta["y_min"])
              if clusters else np.zeros(len(x_um), np.int32))
    grid0, clusters0 = grid, clusters
    n_dropped = 0
    if conf == "high" and clusters:
        keep = {int(c["cluster_id"]) for c in clusters if c["confidence"] == "high"}
        n_dropped = len(clusters) - len(keep)
        clusters = [c for c in clusters if int(c["cluster_id"]) in keep]
        keep_arr = np.array(sorted(keep), dtype=member.dtype)
        member = np.where(np.isin(member, keep_arr), member, 0)
    if conf == "high" and clusters is not None:

        keep_ids = {int(c["cluster_id"]) for c in clusters}
        grid = np.where(np.isin(grid, np.array(sorted(keep_ids) or [0], dtype=grid.dtype)), grid, 0)
    fill_mode = None
    if FILL_ALL and not clusters:
        clusters, member, grid, fill_mode = _fill_regions(
            x_um, y_um, is_schwann, prob, panel, grid0, clusters0, meta)
        if fill_mode:
            n_dropped = 0
    clusters, member, grid = _renumber_by_position(clusters, member, grid, meta)
    return clusters, member, grid, {"n_dropped_low_confidence": n_dropped,
                                    "confidence_gate": conf, "fill_mode": fill_mode, **meta}


def _renumber_by_position(clusters, member, grid, meta):
    if not clusters:
        return clusters, member, grid
    from skimage.measure import find_contours
    from region_order import serpentine_order
    cents = []
    for c in clusters:
        pts = [np.column_stack([ct[:, 1] * OP.RESOLUTION + meta["x_min"], ct[:, 0] * OP.RESOLUTION + meta["y_min"]])
               for ct in find_contours((grid == int(c["cluster_id"])).astype(float), 0.5)]
        pts = np.vstack(pts) if pts else np.zeros((1, 2))
        cents.append(((pts[:, 0].min() + pts[:, 0].max()) / 2, (pts[:, 1].min() + pts[:, 1].max()) / 2))
    cents = np.asarray(cents, float)
    ranks = serpentine_order(cents[:, 0], cents[:, 1])
    old = np.array([int(c["cluster_id"]) for c in clusters])
    lut = np.zeros(int(max(grid.max(), old.max())) + 1, dtype=grid.dtype)
    lut[old] = ranks
    new_grid = lut[grid]
    new_member = lut[np.clip(member, 0, len(lut) - 1)]
    out = []
    for c, r in zip(clusters, ranks):
        c = dict(c); c["cluster_id"] = int(r); out.append(c)
    out.sort(key=lambda c: c["cluster_id"])
    return out, new_member, new_grid


def _rank_and_keep(clusters, grid, x_um, y_um, prob, meta, top):
    member = OP.membership(grid, x_um, y_um, meta["x_min"], meta["y_min"])
    def score(c):
        m = member == int(c["cluster_id"])
        if prob is not None and m.any():
            return float(np.asarray(prob, float)[m].mean())
        return float(c["n_schwann"])
    kept = sorted(clusters, key=score, reverse=True)[:top]
    ids = np.array(sorted(int(c["cluster_id"]) for c in kept), dtype=grid.dtype)
    grid = np.where(np.isin(grid, ids), grid, 0)
    member = np.where(np.isin(member, ids), member, 0)
    return kept, member, grid


def _fill_regions(x_um, y_um, is_schwann, prob, panel, grid0, clusters0, meta):

    if clusters0:
        member = OP.membership(grid0, x_um, y_um, meta["x_min"], meta["y_min"])
        return clusters0, member, grid0, "all_confidence"

    saved = (OP.MIN_SCHWANN_CELLS, OP.COUNT_MIN_CELLS)
    try:
        OP.MIN_SCHWANN_CELLS, OP.COUNT_MIN_CELLS = 1, 1
        grid_b, clusters_b, meta_b = OP.run_operator(x_um, y_um, is_schwann, panel=panel)
    finally:
        OP.MIN_SCHWANN_CELLS, OP.COUNT_MIN_CELLS = saved
    if clusters_b:
        kept, member, grid_b = _rank_and_keep(clusters_b, grid_b, x_um, y_um, prob, meta_b, FILL_TOP)
        meta.update({k: meta_b[k] for k in ("x_min", "y_min", "nx", "ny")})
        return kept, member, grid_b, "relaxed_top%d" % FILL_TOP

    x = np.asarray(x_um, float); y = np.asarray(y_um, float)
    if prob is not None:
        order = np.argsort(-np.asarray(prob, float))
    else:
        order = np.flatnonzero(np.asarray(is_schwann, bool))
    order = order[:FILL_TOP * 6]
    if not len(order):
        return [], np.zeros(len(x), np.int32), np.zeros_like(grid0), None
    ny, nx = int(meta["ny"]), int(meta["nx"])
    grid_c = np.zeros((ny, nx), dtype=np.int32)
    r_px = max(int(round(FILL_DISK_UM / OP.RESOLUTION)), 1)
    yy, xx = np.ogrid[-r_px:r_px + 1, -r_px:r_px + 1]
    disk = (xx ** 2 + yy ** 2) <= r_px ** 2
    clusters = []
    area_mm2 = float(np.pi * FILL_DISK_UM ** 2 / 1e6)
    k = 0
    for i in order:
        if len(clusters) >= FILL_TOP:
            break
        cx = int(np.clip((x[i] - meta["x_min"]) / OP.RESOLUTION, 0, nx - 1))
        cy = int(np.clip((y[i] - meta["y_min"]) / OP.RESOLUTION, 0, ny - 1))
        y0, y1 = max(0, cy - r_px), min(ny, cy + r_px + 1)
        x0, x1 = max(0, cx - r_px), min(nx, cx + r_px + 1)
        sub = disk[(y0 - (cy - r_px)):(y1 - (cy - r_px)), (x0 - (cx - r_px)):(x1 - (cx - r_px))]
        patch = grid_c[y0:y1, x0:x1]
        paint = (patch == 0) & sub
        if not paint.any():
            continue
        k += 1
        patch[paint] = k
        comp = grid_c == k
        n_in = int(np.sum(OP.membership(grid_c, x, y, meta["x_min"], meta["y_min"]) == k))
        clusters.append(dict(cluster_id=k, n_schwann=max(n_in, 1), area_mm2=area_mm2,
                             areal_density=max(n_in, 1) / area_mm2, aspect_ratio=1.0,
                             solidity=1.0, linear_density=0.0, median_nn_um=0.0,
                             shape_class="compact", confidence="low", comp=comp))
    member = OP.membership(grid_c, x, y, meta["x_min"], meta["y_min"])
    return clusters, member, grid_c, "pseudo_top%d" % FILL_TOP
