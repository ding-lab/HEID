#!/usr/bin/env python3

from __future__ import annotations

import numpy as np
from scipy.ndimage import (binary_closing, binary_dilation, binary_opening, convolve,
                           gaussian_filter, label)
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components, minimum_spanning_tree
from skimage.draw import line as skline
from skimage.feature import peak_local_max
from skimage.measure import regionprops
from skimage.morphology import disk
from skimage.segmentation import watershed


RESOLUTION = 10
PAD_UM = 200
SIGMA_PX = 4
MIN_SCHWANN_CELLS = 5
HIGH_CONF_CELLS = 15
CORE_THRESH_ABS_BY_PANEL = {"5k": 0.086, "477": 0.096}
COUNT_RADIUS_UM = 35
COUNT_MIN_CELLS = 5
OPEN_RADIUS_UM = 10
SMOOTH_CLOSE_UM = 20
GRAB_UM = 30
ADAPT_REF_DENSITY = 90.0
ADAPT_LINK_POWER = 0.165
ADAPT_LINK_MAX = 1.8
ADAPT_GRAB_POWER = 0.24
ADAPT_GRAB_MAX = 2.2
ADAPT_SCALE_MIN = 0.9
PEAK_MIN_DIST_PX = 4
GAP_SPLIT_UM = 50.0
SPLIT_MIN_CELLS = 6
AREA_CUT_MM2 = 0.10
AREAL_FLOOR_SMALL = 250.0
AREAL_FLOOR_LARGE = 500.0
MIN_SOLIDITY = 0.50
MIN_AREA_COMPACT = 0.0
MIN_ASPECT = 3.0
MIN_MINOR_UM = 25.0
MIN_MAJOR_UM = 150.0
MIN_LINEAR_DENSITY = 30.0
MIN_AREA_ELONG = 0.0
MAX_MEDIAN_NN_UM = 30.0
TRACT_LINK_UM = 50.0
TRACT_MIN_CELLS = 5
TRACT_MIN_MAJOR_UM = 105.0
TRACT_MIN_ASPECT = 2.2
TRACT_MIN_LIN = 10.0
TRACT_DILATE_PX = 2
TRACT_SUBSUME_FRAC = 0.6
TRACT_CONT_UM = 90.0
TRACT_CONT_FRAC = 0.6
LOOSE_INCL_DENS_HI = 25.0
LOOSE_INCL_DENS_LO = 3.0
LOOSE_ALL_DENS = 8.0
LOOSE_ADD_MIN = 2


def required_areal_density(area_mm2: float) -> float:
    return AREAL_FLOOR_SMALL if area_mm2 <= AREA_CUT_MM2 else AREAL_FLOOR_LARGE


def _linkage_labels(pts, eps):
    n = len(pts)
    if n <= 1:
        return n, np.zeros(n, dtype=int)
    from scipy.spatial import cKDTree
    pairs = cKDTree(pts).query_pairs(eps, output_type="ndarray")
    if len(pairs) == 0:
        return n, np.arange(n)
    m = csr_matrix((np.ones(len(pairs)), (pairs[:, 0], pairs[:, 1])), shape=(n, n))
    return connected_components(m, directed=False)


def refine_components(lb, n_comps, density, thresh, s_iy, s_ix):
    from scipy.spatial import cKDTree
    new = np.zeros_like(lb)
    nxt = 1
    cell_comp = lb[s_iy, s_ix]
    gap_px = max(GAP_SPLIT_UM / RESOLUTION, 1.0)
    for i in range(1, n_comps + 1):
        comp = lb == i
        if not comp.any():
            continue
        rp = regionprops(comp.astype(np.uint8))
        major = float(rp[0].axis_major_length) if rp else 0.0
        minor = float(rp[0].axis_minor_length) if rp else 0.0
        ar = (major / minor) if minor > 0 else (999.0 if major > 0 else 1.0)
        if ar >= MIN_ASPECT:
            sel = np.where(cell_comp == i)[0]
            cpx = np.column_stack([s_iy[sel], s_ix[sel]]).astype(float)
            ng, gl = _linkage_labels(cpx, gap_px)
            sig = [g for g in range(ng) if int(np.sum(gl == g)) >= SPLIT_MIN_CELLS]
            if len(sig) <= 1:
                new[comp] = nxt
                nxt += 1
                continue
            keep = np.isin(gl, sig)
            sig_px = cpx[keep]
            remap = {g: j for j, g in enumerate(sig)}
            sig_g = np.array([remap[g] for g in gl[keep]])
            pix = np.column_stack(np.where(comp))
            _, idx = cKDTree(sig_px).query(pix)
            for j in range(len(sig)):
                segp = pix[sig_g[idx] == j]
                if len(segp):
                    new[segp[:, 0], segp[:, 1]] = nxt
                    nxt += 1
            continue
        peaks = peak_local_max(density, min_distance=PEAK_MIN_DIST_PX, threshold_abs=thresh,
                               labels=comp, exclude_border=False)
        if len(peaks) <= 1:
            new[comp] = nxt
            nxt += 1
            continue
        markers = np.zeros(lb.shape, dtype=np.int32)
        for j, (yy, xx) in enumerate(peaks, start=1):
            markers[yy, xx] = j
        ws = watershed(-density, markers, mask=comp)
        for j in range(1, len(peaks) + 1):
            seg = ws == j
            if seg.any():
                new[seg] = nxt
                nxt += 1
    return new, nxt - 1


def detect_tracts(sch_xy, x_min, y_min, nx, ny, link_um):
    from scipy.spatial import cKDTree
    out = []
    n = len(sch_xy)
    if n < TRACT_MIN_CELLS:
        return out
    pairs = cKDTree(sch_xy).query_pairs(link_um, output_type="ndarray")
    if len(pairs) == 0:
        return out
    m = csr_matrix((np.ones(len(pairs)), (pairs[:, 0], pairs[:, 1])), shape=(n, n))
    ncomp, lab = connected_components(m, directed=False)
    for g in range(ncomp):
        idx = np.where(lab == g)[0]
        if len(idx) < TRACT_MIN_CELLS:
            continue
        gxy = sch_xy[idx]
        gy = np.clip(((gxy[:, 1] - y_min) / RESOLUTION).astype(int), 0, ny - 1)
        gx = np.clip(((gxy[:, 0] - x_min) / RESOLUTION).astype(int), 0, nx - 1)
        comp = np.zeros((ny, nx), bool)
        comp[gy, gx] = True
        sub = cKDTree(gxy).query_pairs(link_um, output_type="ndarray")
        if len(sub):
            d = np.linalg.norm(gxy[sub[:, 0]] - gxy[sub[:, 1]], axis=1)
            mst = minimum_spanning_tree(
                csr_matrix((d, (sub[:, 0], sub[:, 1])), shape=(len(gxy), len(gxy)))).tocoo()
            for a, b in zip(mst.row, mst.col):
                rr, cc = skline(int(gy[a]), int(gx[a]), int(gy[b]), int(gx[b]))
                comp[rr, cc] = True
        comp = binary_dilation(comp, structure=disk(TRACT_DILATE_PX))
        props = regionprops(comp.astype(int))
        if not props:
            continue
        r = props[0]
        major = float(r.axis_major_length) * RESOLUTION
        minor = max(float(r.axis_minor_length) * RESOLUTION, 1.0)
        aspect = major / minor
        lin = len(idx) / (major / 1000.0) if major > 0 else 0.0
        nn = float(np.median(cKDTree(gxy).query(gxy, k=2)[0][:, 1])) if len(gxy) >= 2 else 99.0
        if not (major >= TRACT_MIN_MAJOR_UM and aspect >= TRACT_MIN_ASPECT
                and lin >= TRACT_MIN_LIN and nn <= MAX_MEDIAN_NN_UM):
            continue
        cp = cKDTree(gxy).query_pairs(TRACT_CONT_UM, output_type="ndarray")
        if len(cp):
            _, sub_lab = connected_components(
                csr_matrix((np.ones(len(cp)), (cp[:, 0], cp[:, 1])), shape=(len(gxy), len(gxy))),
                directed=False)
            largest = int(np.bincount(sub_lab).max())
        else:
            largest = 1
        if largest < TRACT_CONT_FRAC * len(idx):
            continue

        area = int(comp.sum()) * RESOLUTION ** 2 / 1e6
        out.append(dict(n_schwann=len(idx), area_mm2=area,
                        areal_density=len(idx) / area if area else 0.0,
                        linear_density=lin, aspect_ratio=aspect, minor_axis_um=minor,
                        major_axis_um=major, solidity=float(r.solidity),
                        median_nn_um=nn, shape_class="elongated",
                        confidence="high" if len(idx) >= HIGH_CONF_CELLS else "low",
                        comp=comp))
    return out


def run_operator(all_x_um, all_y_um, is_schwann, panel="5k", grid=None):
    from scipy.spatial import cKDTree
    ax = np.asarray(all_x_um, dtype=np.float64)
    ay = np.asarray(all_y_um, dtype=np.float64)
    sel = np.asarray(is_schwann, dtype=bool)
    thresh_abs = CORE_THRESH_ABS_BY_PANEL[panel]

    if grid is None:
        x_min, x_max = ax.min() - PAD_UM, ax.max() + PAD_UM
        y_min, y_max = ay.min() - PAD_UM, ay.max() + PAD_UM
        nx = int(np.ceil((x_max - x_min) / RESOLUTION))
        ny = int(np.ceil((y_max - y_min) / RESOLUTION))
    else:
        x_min, y_min = float(grid["x_min"]), float(grid["y_min"])
        nx, ny = int(grid["nx"]), int(grid["ny"])
        x_max, y_max = x_min + nx * RESOLUTION, y_min + ny * RESOLUTION

    empty = np.zeros((ny, nx), dtype=np.int32)
    meta = {"n_schwann_pred": int(sel.sum()), "nx": nx, "ny": ny,
            "x_min": x_min, "y_min": y_min}
    if int(sel.sum()) < MIN_SCHWANN_CELLS:
        meta["stop"] = "TOO_FEW_PREDICTED_SCHWANN"
        return empty, [], meta

    sch_xy = np.column_stack([ax[sel], ay[sel]])
    s_ix = np.clip(((sch_xy[:, 0] - x_min) / RESOLUTION).astype(int), 0, nx - 1)
    s_iy = np.clip(((sch_xy[:, 1] - y_min) / RESOLUTION).astype(int), 0, ny - 1)

    raw = np.zeros((ny, nx), dtype=np.float32)
    np.add.at(raw, (s_iy, s_ix), 1.0)
    density = gaussian_filter(raw, sigma=SIGMA_PX)
    if float(density.max()) <= 0:
        meta["stop"] = "NO_DENSITY"
        return empty, [], meta

    count_r = max(int(round(COUNT_RADIUS_UM / RESOLUTION)), 1)
    local_count = convolve(raw, disk(count_r).astype(np.float32), mode="constant", cval=0.0)
    core_mask = local_count >= COUNT_MIN_CELLS
    open_r = max(int(round(OPEN_RADIUS_UM / RESOLUTION)), 1)
    mask = binary_opening(core_mask, structure=disk(open_r))
    lb, n_comps = label(mask)
    lb, n_comps = refine_components(lb, n_comps, density, thresh_abs, s_iy, s_ix)
    if n_comps:
        close_r = max(int(round(SMOOTH_CLOSE_UM / RESOLUTION)), 1)
        cr = disk(close_r)
        sm = np.zeros_like(lb)
        for i in range(1, n_comps + 1):
            ci = binary_closing(lb == i, structure=cr)
            sm[ci & (sm == 0)] = i
        lb = sm

    s_labels = lb[s_iy, s_ix]
    counts = np.bincount(s_labels, minlength=n_comps + 1)
    props = {r.label: r for r in regionprops(lb)}

    cluster_grid = np.zeros((ny, nx), dtype=np.int32)
    clusters = []
    cid = 1
    n_gate_rejected = 0
    for i in range(1, n_comps + 1):
        n_in = int(counts[i])
        if n_in < MIN_SCHWANN_CELLS:
            continue
        r = props.get(i)
        comp = lb == i
        area_px = int(comp.sum())
        area_mm2 = area_px * RESOLUTION ** 2 / 1e6
        areal_density = n_in / area_mm2 if area_mm2 > 0 else 0.0
        major_um = float(r.axis_major_length) * RESOLUTION if r else 0.0
        minor_um = float(r.axis_minor_length) * RESOLUTION if r else 0.0
        aspect = (major_um / minor_um) if minor_um > 0 else (999.0 if major_um > 0 else 1.0)
        solidity = float(r.solidity) if r else 0.0
        linear_density = n_in / (major_um / 1000.0) if major_um > 0 else 0.0
        memb = sch_xy[s_labels == i]
        if len(memb) >= 2:
            d, _ = cKDTree(memb).query(memb, k=2)
            median_nn = float(np.median(d[:, 1]))
        else:
            median_nn = float("nan")
        elongated_shape = aspect >= MIN_ASPECT
        not_loose = median_nn <= MAX_MEDIAN_NN_UM
        dense_enough = areal_density >= required_areal_density(area_mm2)
        if elongated_shape:
            passed = (minor_um >= MIN_MINOR_UM and major_um >= MIN_MAJOR_UM
                      and linear_density >= MIN_LINEAR_DENSITY and dense_enough
                      and area_mm2 >= MIN_AREA_ELONG and not_loose)
        else:
            passed = (dense_enough and solidity >= MIN_SOLIDITY
                      and area_mm2 >= MIN_AREA_COMPACT and not_loose)

        if not passed:
            n_gate_rejected += 1
            continue
        cluster_grid[comp] = cid
        clusters.append(dict(cluster_id=cid, n_schwann=n_in, area_mm2=area_mm2,
                             areal_density=areal_density, aspect_ratio=aspect,
                             solidity=solidity, linear_density=linear_density,
                             median_nn_um=median_nn,
                             shape_class="elongated" if elongated_shape else "compact",
                             confidence="high" if n_in >= HIGH_CONF_CELLS else "low",
                             comp=comp))
        cid += 1


    sch_density = int(sel.sum()) / (((x_max - x_min) * (y_max - y_min)) / 1e6)
    ratio = ADAPT_REF_DENSITY / max(sch_density, 1e-6)
    link_um = TRACT_LINK_UM * float(np.clip(ratio ** ADAPT_LINK_POWER,
                                            ADAPT_SCALE_MIN, ADAPT_LINK_MAX))
    grab_um = GRAB_UM * float(np.clip(ratio ** ADAPT_GRAB_POWER,
                                      ADAPT_SCALE_MIN, ADAPT_GRAB_MAX))


    body_mask = cluster_grid[s_iy, s_ix] > 0
    loose_halo = link_um * float(np.clip((LOOSE_INCL_DENS_HI - sch_density)
                                         / (LOOSE_INCL_DENS_HI - LOOSE_INCL_DENS_LO), 0.0, 1.0))
    if sch_density <= LOOSE_ALL_DENS:
        elig = np.ones(len(sch_xy), dtype=bool)
    else:
        elig = body_mask.copy()
        if loose_halo > 0 and body_mask.any() and (~body_mask).any():
            d2b = cKDTree(sch_xy[body_mask]).query(sch_xy, k=1)[0]
            elig |= (~body_mask) & (d2b <= loose_halo)
    tracts = detect_tracts(sch_xy[elig], x_min, y_min, nx, ny, link_um)
    n_tracts_kept = 0
    if tracts:
        kept, to_remove = [], set()
        for t in tracts:
            covered = [i for i, c in enumerate(clusters)
                       if (c["comp"] & t["comp"]).sum() / max(int(c["comp"].sum()), 1)
                       > TRACT_SUBSUME_FRAC]
            added = t["n_schwann"] - sum(clusters[i]["n_schwann"] for i in covered)
            if len(covered) >= 2 or (len(covered) <= 1 and added >= LOOSE_ADD_MIN):
                kept.append(t)
                to_remove.update(covered)
        clusters = [c for i, c in enumerate(clusters) if i not in to_remove]
        clusters += kept
        n_tracts_kept = len(kept)


    cluster_grid = np.zeros((ny, nx), dtype=np.int32)
    for i, c in enumerate(clusters, 1):
        c["cluster_id"] = i
        cluster_grid[c["comp"]] = i
        c.pop("comp", None)

    meta.update({"n_components": int(n_comps), "n_gate_rejected": n_gate_rejected,
                 "n_tracts_kept": n_tracts_kept, "n_clusters": len(clusters),
                 "sch_density_per_mm2": round(float(sch_density), 4),
                 "link_um": round(link_um, 3), "grab_um": round(grab_um, 3),
                 "gene_blind": True, "rescued_cores_possible": False})
    return cluster_grid, clusters, meta


def membership(cluster_grid, all_x_um, all_y_um, x_min, y_min):
    ny, nx = cluster_grid.shape
    ix = np.clip(((np.asarray(all_x_um, float) - x_min) / RESOLUTION).astype(int), 0, nx - 1)
    iy = np.clip(((np.asarray(all_y_um, float) - y_min) / RESOLUTION).astype(int), 0, ny - 1)
    return cluster_grid[iy, ix]
