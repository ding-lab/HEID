#!/usr/bin/env python3

import argparse
import json
import os
import sys

import numpy as np
import pandas as pd
from scipy.ndimage import (binary_closing, binary_dilation, binary_fill_holes,
                           binary_opening, distance_transform_edt, gaussian_filter, label)
from scipy.spatial import cKDTree
from skimage.feature import peak_local_max
from skimage.measure import find_contours, regionprops
from skimage.morphology import disk
from skimage.segmentation import watershed

import matplotlib
matplotlib.use("Agg")
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

P3D = os.environ.get("THREED_ROOT", os.path.join(os.environ.get("PROJECTS_ROOT", "/data/heid"), "3d"))
PRED = os.environ.get("TLS_PRED_DIR",
                      os.path.join(P3D, "inference/cohort/data/predictions"))
DEFAULT_OUT = os.path.join(P3D, "tls_define")

OUT_ROOT = os.environ.get("TLS_OUT_ROOT", "")


RESCUE_JSON = os.environ.get("TLS_RESCUE_JSON", "")
RESCUE_GATE_FRAC = 0.5
RESCUE_OVERLAP_MIN = 0.15


RESOLUTION = 10
SIGMA_PX = 4
DENSITY_THRESH_REL = 0.08
CLOSING_R_PX = 10
OPENING_R_PX = 3
MAX_HOLE_AREA_PX = 50000
MIN_B_CELLS = 50
MIN_B_DENSITY_CELLS_MM2 = 8000
MIN_B_CORE_AREA_UM2 = 10000
TLS_PEAK_MIN_SEP_UM = 250.0
HALO_RADIUS_UM = 150
MIN_T_IN_REGION = 60
MIN_TLS_AREA_UM2 = 80000
MIN_LYMPHOID_DENSITY_CELLS_MM2 = 3000.0
MIN_B_DENSITY_REGION_MM2 = 920
MIN_REGION_CIRCULARITY = 0.65
MIN_REGION_AXIS_RATIO = 0.50
CORE_SIGMA_PX = 1.5
CORE_THRESH_REL = 0.30
CORE_CLOSING_PX = 3
CORE_OPENING_PX = 2
CORE_NECK_RADIUS_PX = 3.0
CORE_HOLE_FILL_MAX_PX = 30
CORE_HOLE_CONNECT_FRAC = 0.5
SPLIT_MAX_CORE_AREA_UM2 = 100000.0
SPLIT_PEAK_SEP_UM = 150.0
SPLIT_MERGE_FRAC = 0.8
SPLIT_CORE_THRESH = 0.45


VALLEY_PEAK_MIN_REL = 0.5
VALLEY_SADDLE_MAX = 0.5
CORE_SIGMA_SMALL_PX = 1.5
CORE_SIGMA_BIG_PX = 4.0
BIG_COMP_AREA_UM2 = 100000.0
MIN_B_DENSITY_BIG_MM2 = 4000.0
TISSUE_SIGMA = 10
TISSUE_THRESH = 0.015
MERGE_SADDLE_FRAC = 0.5


AGG_MIN_CELLS = 50
AGG_MIN_CORE_AREA_UM2 = 10000
AGG_CORE_DENS_MM2 = 8000.0
AGG_BROAD_T_AREA_UM2 = 300000.0
AGG_BROAD_T_DENS_MM2 = 7000.0

HALOS_UM = [50.0, 100.0, 150.0]
PLASMA_FRAC_CUTS = (0.08, 0.20)


B_TYPES = ["B_cell", "NK_T"]
T_TYPES = ["NK_T"]
P_TYPES = ["Plasma"]
MYELOID_TYPES = ["Myeloid_NOS"]
TUMOR_TYPE = "Tumor"


CAVEAT = ("MODEL PREDICTION on H&E (pan-cancer Cell head, single fold); no spatial ground "
          "truth in this cohort, so nothing here is scored. No molecular grading: "
          "this cohort has no expression and the Cell model has no DC class.")


def fill_small_holes(mask, max_hole_area):
    filled = binary_fill_holes(mask)
    holes = filled & ~mask
    if not holes.any():
        return filled
    lb_h, n_h = label(holes)
    for idx in range(1, n_h + 1):
        if (lb_h == idx).sum() > max_hole_area:
            filled[lb_h == idx] = False
    return filled


def b_density_mask(raw):
    d = gaussian_filter(raw, sigma=SIGMA_PX)
    if d.max() <= 0:
        return np.zeros_like(raw, dtype=bool)
    m = d > DENSITY_THRESH_REL * d.max()
    m = binary_closing(m, structure=disk(CLOSING_R_PX))
    m = fill_small_holes(m, MAX_HOLE_AREA_PX)
    m = binary_opening(m, structure=disk(OPENING_R_PX))
    return m


def merge_shallow_basins(ws, dens, frac):
    nlab = int(ws.max())
    if nlab <= 1 or frac <= 0:
        return ws
    peak_d = np.zeros(nlab + 1)
    for j in range(1, nlab + 1):
        m = ws == j
        if m.any():
            peak_d[j] = float(dens[m].max())
    saddle = {}
    for s_lo, s_hi in (((slice(None), slice(0, -1)), (slice(None), slice(1, None))),
                       ((slice(0, -1), slice(None)), (slice(1, None), slice(None)))):
        a, b = ws[s_lo], ws[s_hi]
        diff = (a != b) & (a > 0) & (b > 0)
        if not diff.any():
            continue
        au, bu = a[diff], b[diff]
        col = np.minimum(dens[s_lo][diff], dens[s_hi][diff])
        for u, v, c in zip(au, bu, col):
            k = (u, v) if u < v else (v, u)
            if c > saddle.get(k, 0.0):
                saddle[k] = float(c)
    parent = list(range(nlab + 1))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for (u, v), col in saddle.items():
        if col >= frac * min(peak_d[u], peak_d[v]):
            parent[find(u)] = find(v)
    out = np.zeros_like(ws)
    remap, nxt = {}, 1
    for j in range(1, nlab + 1):
        r = find(j)
        if r not in remap:
            remap[r] = nxt
            nxt += 1
        out[ws == j] = remap[r]
    return out


def clean_core(core, dens_c, thr):
    core = core.copy()
    filled = binary_fill_holes(core)
    holes = filled & ~core
    if holes.any():
        lb_h, n_h = label(holes)
        for h in range(1, n_h + 1):
            hole = lb_h == h
            ha = int(hole.sum())
            hole_dens = float(dens_c[hole].mean()) if ha else 0.0
            if ha <= CORE_HOLE_FILL_MAX_PX or hole_dens >= CORE_HOLE_CONNECT_FRAC * thr:
                core |= hole
    edt = distance_transform_edt(core)
    if edt.max() < CORE_NECK_RADIUS_PX:
        return []
    seeds_mask = edt >= CORE_NECK_RADIUS_PX
    seeds, n = label(seeds_mask)
    if n <= 1:
        return [binary_fill_holes(core)]
    ws = watershed(-edt, seeds, mask=core, watershed_line=True)
    pieces = []
    for b in range(1, n + 1):
        piece = ws == b
        if piece.any():
            pieces.append(binary_fill_holes(piece))
    return pieces


def split_oversized_core(core, b_density, b_iy, b_ix):
    sep_px = max(int(SPLIT_PEAK_SEP_UM / RESOLUTION), 3)
    peaks = peak_local_max(b_density, min_distance=sep_px, labels=core, exclude_border=False)
    if len(peaks) <= 1:
        basins = [core]
    else:
        markers = np.zeros_like(core, dtype=np.int32)
        for i, (yy, xx) in enumerate(peaks, 1):
            markers[yy, xx] = i
        ws = watershed(-b_density, markers, mask=core)
        ws = merge_shallow_basins(ws, b_density, SPLIT_MERGE_FRAC)
        basins = [ws == j for j in range(1, int(ws.max()) + 1) if (ws == j).any()]
    out = []
    for sb in basins:
        in_sb = sb[b_iy, b_ix]
        if not in_sb.any():
            continue
        raw_c = np.zeros_like(b_density, dtype=np.float32)
        np.add.at(raw_c, (b_iy[in_sb], b_ix[in_sb]), 1)
        dens_c = gaussian_filter(raw_c, sigma=CORE_SIGMA_PX)
        if dens_c.max() <= 0:
            continue
        thr = SPLIT_CORE_THRESH * dens_c.max()
        tight = (dens_c > thr) & sb
        if CORE_CLOSING_PX > 0:
            tight = binary_closing(tight, structure=disk(CORE_CLOSING_PX))
        if CORE_OPENING_PX > 0:
            tight = binary_opening(tight, structure=disk(CORE_OPENING_PX))
        tight &= sb
        lb_p, n_p = label(tight)
        for pid in range(1, n_p + 1):
            out.extend(clean_core(lb_p == pid, dens_c, thr))
    return out if out else [core]


def valley_split(core, b_density):
    sep_px = max(int(SPLIT_PEAK_SEP_UM / RESOLUTION), 3)
    peaks = peak_local_max(b_density, min_distance=sep_px, labels=core.astype(np.int32), exclude_border=False)
    if len(peaks) <= 1:
        return [core]
    vals = b_density[peaks[:, 0], peaks[:, 1]]
    peaks = peaks[vals >= VALLEY_PEAK_MIN_REL * vals.max()]
    if len(peaks) <= 1:
        return [core]
    markers = np.zeros_like(core, dtype=np.int32)
    for i, (yy, xx) in enumerate(peaks, 1):
        markers[yy, xx] = i
    ws = watershed(-b_density, markers, mask=core)
    ws = merge_shallow_basins(ws, b_density, VALLEY_SADDLE_MAX)
    pieces = [ws == j for j in range(1, int(ws.max()) + 1) if (ws == j).any()]
    if len(pieces) <= 1:
        return [core]
    min_px = MIN_B_CORE_AREA_UM2 / (RESOLUTION ** 2)
    if any(int(p.sum()) < min_px for p in pieces):
        return [core]
    return pieces


def region_shape_stats(region, core):
    lbl, n = label(region)
    if n == 0:
        return None, None
    if n == 1:
        comp = region
    else:
        core_lbls = lbl[core]
        core_lbls = core_lbls[core_lbls > 0]
        if len(core_lbls):
            vals, cnts = np.unique(core_lbls, return_counts=True)
            pick = int(vals[cnts.argmax()])
        else:
            sizes = np.bincount(lbl.ravel())
            sizes[0] = 0
            pick = int(sizes.argmax())
        comp = lbl == pick
    return regionprops(comp.astype(np.uint8))[0], comp


def make_region(core, tissue, ny, nx, halo_px):
    base = binary_dilation(core, structure=disk(halo_px)) & tissue
    region = base
    props, _ = region_shape_stats(region, core)
    if props is None:
        return region, False
    M_full, m_full = props.major_axis_length, props.minor_axis_length
    ar = (m_full / M_full) if M_full > 0 else 1.0
    peri = props.perimeter
    circ = (4 * np.pi * props.area / (peri ** 2)) if peri > 0 else 0.0
    reshaped = False
    if (ar < MIN_REGION_AXIS_RATIO or circ < MIN_REGION_CIRCULARITY) and M_full > 0:
        ys_r, xs_r = np.where(region)
        if len(ys_r) >= 5:
            pts = np.column_stack([ys_r, xs_r]).astype(np.float64)
            ctr = pts.mean(axis=0)
            pts_c = pts - ctr
            cov = (pts_c.T @ pts_c) / len(pts_c)
            eigvals, eigvecs = np.linalg.eigh(cov)
            major_vec = eigvecs[:, int(np.argmax(eigvals))]
            proj = pts_c @ major_vec
            cyi, cxi = np.where(core)
            core_center = (float(np.median((np.column_stack([cyi, cxi]).astype(np.float64) - ctr)
                                           @ major_vec)) if len(cyi) else 0.0)
            half = min((proj.max() - proj.min()) / 2, m_full / MIN_REGION_AXIS_RATIO / 2)
            keep = np.abs(proj - core_center) <= half
            cand = np.zeros((ny, nx), dtype=bool)
            cand[ys_r[keep], xs_r[keep]] = True
            cand &= tissue
            if (cand & core).any():
                region = cand | base
                reshaped = True
    return region, reshaped


def build_cores(ny, nx, tissue, a_ix, a_iy, sigma_small, sigma_big, big_area_um2):
    raw = np.zeros((ny, nx), np.float32)
    np.add.at(raw, (a_iy, a_ix), 1)
    density = gaussian_filter(raw, sigma=SIGMA_PX)
    lb, n_cc = label(b_density_mask(raw) & tissue)

    if n_cc > 0:
        sep_px = max(int(TLS_PEAK_MIN_SEP_UM / RESOLUTION), 3)
        new_lb = np.zeros_like(lb, dtype=np.int32)
        nxt = 1
        for cid in range(1, n_cc + 1):
            comp = lb == cid
            peaks = peak_local_max(density, min_distance=sep_px, labels=comp,
                                   exclude_border=False)
            if len(peaks) <= 1:
                new_lb[comp] = nxt
                nxt += 1
                continue
            markers = np.zeros_like(comp, dtype=np.int32)
            for i, (yy, xx) in enumerate(peaks, 1):
                markers[yy, xx] = i
            ws = merge_shallow_basins(watershed(-density, markers, mask=comp),
                                      density, MERGE_SADDLE_FRAC)
            for j in range(1, ws.max() + 1):
                sm = ws == j
                if sm.any():
                    new_lb[sm] = nxt
                    nxt += 1
        lb, n_cc = new_lb, nxt - 1

    cores = []
    for cid in range(1, n_cc + 1):
        comp = lb == cid
        in_comp = comp[a_iy, a_ix]
        if not in_comp.any():
            continue
        is_big = int(comp.sum()) * (RESOLUTION ** 2) > big_area_um2
        raw_c = np.zeros((ny, nx), dtype=np.float32)
        np.add.at(raw_c, (a_iy[in_comp], a_ix[in_comp]), 1)
        dens_c = gaussian_filter(raw_c, sigma=sigma_big if is_big else sigma_small)
        if dens_c.max() <= 0:
            continue
        thr = CORE_THRESH_REL * dens_c.max()
        tight = (dens_c > thr) & comp
        if CORE_CLOSING_PX > 0:
            tight = binary_closing(tight, structure=disk(CORE_CLOSING_PX))
        if CORE_OPENING_PX > 0:
            tight = binary_opening(tight, structure=disk(CORE_OPENING_PX))
        tight &= comp
        lb_p, n_p = label(tight)
        for pid in range(1, n_p + 1):
            for cc in clean_core(lb_p == pid, dens_c, thr):
                cores.append((cc, is_big))

    split_max_px = SPLIT_MAX_CORE_AREA_UM2 / (RESOLUTION ** 2)
    if cores:
        resplit = []
        for core, is_big in cores:
            if (not is_big) and int(core.sum()) > split_max_px:
                resplit.extend((p, is_big) for p in
                               split_oversized_core(core, density, a_iy, a_ix))
            elif is_big and int(core.sum()) > split_max_px:
                resplit.extend((p, is_big) for p in valley_split(core, density))
            else:
                resplit.append((core, is_big))
        cores = resplit
    return cores, density


def _fp_overlap(core, t, x_min, y_min):
    rows = t.get("fp_rows")
    if not rows:
        return 0.0
    fx0, fy0 = float(t["fp_x0"]), float(t["fp_y0"])
    fr = float(t.get("fp_res", RESOLUTION))
    if abs(fr - RESOLUTION) > 1e-6:
        return 0.0
    ys, xs = np.nonzero(core)
    if not len(ys):
        return 0.0
    gx = ((xs * RESOLUTION + x_min - fx0) / fr).astype(int)
    gy = ((ys * RESOLUTION + y_min - fy0) / fr).astype(int)
    ny_f, nx_f = len(rows), len(rows[0])
    ok = (gx >= 0) & (gx < nx_f) & (gy >= 0) & (gy < ny_f)
    if not ok.any():
        return 0.0
    hit = sum(1 for x, y in zip(gx[ok], gy[ok]) if rows[y][x] == "1")
    fp_px = sum(r.count("1") for r in rows)
    return hit / max(1, min(len(ys), fp_px))


def main(sample, slide, outdir):
    pq = os.path.join(PRED, slide + ".parquet")
    if not os.path.exists(pq):
        sys.exit(f"no prediction parquet: {pq}")
    df = pd.read_parquet(pq, columns=["x_um", "y_um", "pred_class_name"])
    df = df.rename(columns={"pred_class_name": "cell_type"})
    ct = df["cell_type"].values

    cls = np.full(len(df), "Other", dtype=object)
    cls[np.isin(ct, B_TYPES)] = "B_cell"
    cls[np.isin(ct, P_TYPES)] = "Plasma"
    df["class"] = cls

    x_min, y_min = df.x_um.min() - 200, df.y_um.min() - 200
    x_max, y_max = df.x_um.max() + 200, df.y_um.max() + 200
    nx = int(np.ceil((x_max - x_min) / RESOLUTION))
    ny = int(np.ceil((y_max - y_min) / RESOLUTION))
    ix = np.clip(((df.x_um.values - x_min) / RESOLUTION).astype(int), 0, nx - 1)
    iy = np.clip(((df.y_um.values - y_min) / RESOLUTION).astype(int), 0, ny - 1)

    all_raw = np.zeros((ny, nx), np.float32)
    np.add.at(all_raw, (iy, ix), 1)
    tissue = binary_fill_holes(binary_closing(
        gaussian_filter(all_raw, sigma=TISSUE_SIGMA) > TISSUE_THRESH, structure=disk(10)))
    area_of = lambda m: float(int(m.sum()) * (RESOLUTION ** 2) / 1e6)

    sel_b = (df["class"] == "B_cell").values
    sel_t = np.isin(ct, T_TYPES)
    sel_p = (df["class"] == "Plasma").values
    sel_m = np.isin(ct, MYELOID_TYPES)
    sel_tu = ct == TUMOR_TYPE
    b_ix, b_iy = ix[sel_b], iy[sel_b]
    t_ix, t_iy = ix[sel_t], iy[sel_t]
    p_ix, p_iy = ix[sel_p], iy[sel_p]

    os.makedirs(outdir, exist_ok=True)
    halo_px = int(round(HALO_RADIUS_UM / RESOLUTION))
    min_core_area_px = int(MIN_B_CORE_AREA_UM2 / (RESOLUTION ** 2))
    min_tls_area_px = int(MIN_TLS_AREA_UM2 / (RESOLUTION ** 2))

    rejected = {"size": 0, "density": 0, "tls_area": 0, "reshape_too_small": 0,
                "halo_T": 0, "lymph_density": 0, "region_b_density": 0}
    accepted = []
    rescue = []
    if RESCUE_JSON and os.path.exists(RESCUE_JSON):
        rescue = json.load(open(RESCUE_JSON)).get(slide, [])
    if sel_b.sum() >= 2:
        cand, b_density = build_cores(ny, nx, tissue, b_ix, b_iy,
                                      CORE_SIGMA_SMALL_PX, CORE_SIGMA_BIG_PX, BIG_COMP_AREA_UM2)
    else:
        cand, b_density = [], np.zeros((ny, nx), np.float32)

    for core, is_big in cand:
        n_b_in = int(core[b_iy, b_ix].sum()) if len(b_ix) else 0
        core_area_px = int(core.sum())
        core_area_mm2 = core_area_px * (RESOLUTION ** 2) / 1e6


        ys_c, xs_c = np.where(core)
        cxn, cyn = xs_c.mean() * RESOLUTION + x_min, ys_c.mean() * RESOLUTION + y_min
        near = any(np.hypot(cxn - t["cx_um"], cyn - t["cy_um"]) <= t["tol_um"]
                   or _fp_overlap(core, t, x_min, y_min) >= RESCUE_OVERLAP_MIN
                   for t in rescue)
        f = RESCUE_GATE_FRAC if near else 1.0
        if n_b_in < MIN_B_CELLS * f or core_area_px < min_core_area_px * f:
            rejected["size"] += 1
            continue
        b_dens = n_b_in / core_area_mm2 if core_area_mm2 > 0 else 0
        gate = (MIN_B_DENSITY_BIG_MM2 if is_big else MIN_B_DENSITY_CELLS_MM2) * f
        if b_dens < gate:
            rejected["density"] += 1
            continue
        region, reshaped = make_region(core, tissue, ny, nx, halo_px)
        if int(region.sum()) < min_tls_area_px * f:
            rejected["reshape_too_small" if reshaped else "tls_area"] += 1
            continue
        area_mm2 = area_of(region)
        n_t = int(region[t_iy, t_ix].sum()) if len(t_ix) else 0
        if n_t < MIN_T_IN_REGION * f:
            rejected["halo_T"] += 1
            continue
        n_b_r = int(region[b_iy, b_ix].sum()) if len(b_ix) else 0
        n_p_r = int(region[p_iy, p_ix].sum()) if len(p_ix) else 0
        if (n_b_r + n_p_r) / area_mm2 < MIN_LYMPHOID_DENSITY_CELLS_MM2 * f:
            rejected["lymph_density"] += 1
            continue
        if n_b_r / area_mm2 < MIN_B_DENSITY_REGION_MM2 * f:
            rejected["region_b_density"] += 1
            continue
        accepted.append({"core": core, "is_big": is_big, "n_b_core": n_b_in,
                         "core_area_mm2": round(core_area_mm2, 4),
                         "core_b_density_mm2": round(b_dens, 1), "reshaped": reshaped,
                         "rescued": bool(near)})


    owned = {}
    for h in HALOS_UM:
        hp = int(round(h / RESOLUTION))
        claimed = np.zeros((ny, nx), bool)
        per = []
        for it in accepted:
            reg, _ = make_region(it["core"], tissue, ny, nx, hp)
            own = reg & ~claimed
            claimed |= own
            per.append(own)
        owned[h] = per

    tumor_xy = df.loc[sel_tu, ["x_um", "y_um"]].to_numpy()
    tumor_tree = cKDTree(tumor_xy) if len(tumor_xy) else None

    rows = []
    for k, it in enumerate(accepted, 1):
        ys, xs = np.where(it["core"])
        row = {"sample": sample, "slide": slide, "tls_id": f"TLS-{k}",
               "n_b_core": it["n_b_core"], "core_area_mm2": it["core_area_mm2"],
               "core_b_density_mm2": it["core_b_density_mm2"],
               "core_class": "broad" if it["is_big"] else "compact",
               "region_reshaped": it["reshaped"], "rescued": it["rescued"],
               "cx_um": float(xs.mean() * RESOLUTION + x_min),
               "cy_um": float(ys.mean() * RESOLUTION + y_min)}
        for h in HALOS_UM:
            own = owned[h][k - 1]
            tag = f"_halo{int(h)}"
            sel = own[iy, ix]
            n = int(sel.sum())
            row[f"n_cells{tag}"] = n
            row[f"region_area_mm2{tag}"] = round(area_of(own), 4)
            nb = int(own[b_iy, b_ix].sum()) if len(b_ix) else 0
            nt = int(own[t_iy, t_ix].sum()) if len(t_ix) else 0
            npl = int(own[p_iy, p_ix].sum()) if len(p_ix) else 0
            row[f"n_B{tag}"], row[f"n_T_NK{tag}"], row[f"n_Plasma{tag}"] = nb, nt, npl
            lymph = nb + npl
            row[f"plasma_fraction{tag}"] = round(npl / lymph, 4) if lymph else None
            if lymph:
                pf = npl / lymph
                row[f"tls_composition{tag}"] = ("b_dominant" if pf < PLASMA_FRAC_CUTS[0]
                                                else "mixed" if pf < PLASMA_FRAC_CUTS[1]
                                                else "plasma_dominant")
            else:
                row[f"tls_composition{tag}"] = None
            if n:
                for c, cnt in pd.Series(ct[sel]).value_counts().items():
                    row[f"frac_{c}{tag}"] = round(int(cnt) / n, 4)

        if tumor_tree is not None:
            d, _ = tumor_tree.query([[row["cx_um"], row["cy_um"]]], k=1)
            row["dist_core_centroid_to_nearest_tumor_cell_um"] = round(float(d[0]), 1)
        else:
            row["dist_core_centroid_to_nearest_tumor_cell_um"] = None
        row["molecular_grading"] = "NOT AVAILABLE (no expression; the Cell model has no DC class)"
        rows.append(row)
    tls_df = pd.DataFrame(rows)
    tls_df.to_csv(os.path.join(outdir, f"{slide}_tls.csv"), index=False)


    agg_out, agg_reject = {}, {}
    for tag, sel_a, broad_area, broad_dens in (
            ("tagg", sel_t, AGG_BROAD_T_AREA_UM2, AGG_BROAD_T_DENS_MM2),
            ("magg", sel_m, np.inf, AGG_CORE_DENS_MM2)):
        items = []
        agg_rej = {"n_candidates": 0, "size": 0, "density": 0, "overlaps_tls": 0}
        if sel_a.sum() >= 2:
            a_ix, a_iy = ix[sel_a], iy[sel_a]
            cand_a, _ = build_cores(ny, nx, tissue, a_ix, a_iy,
                                    CORE_SIGMA_SMALL_PX, CORE_SIGMA_BIG_PX, broad_area)
            union_tls = np.zeros((ny, nx), bool)
            for own in owned[150.0]:
                union_tls |= own
            agg_rej["n_candidates"] = len(cand_a)
            for core, is_big in cand_a:
                n_in = int(core[a_iy, a_ix].sum())
                a_px = int(core.sum())
                a_mm2 = a_px * (RESOLUTION ** 2) / 1e6
                if n_in < AGG_MIN_CELLS or a_px < int(AGG_MIN_CORE_AREA_UM2 / RESOLUTION ** 2):
                    agg_rej["size"] += 1
                    continue
                dens = n_in / a_mm2 if a_mm2 else 0
                if dens < (broad_dens if is_big else AGG_CORE_DENS_MM2):
                    agg_rej["density"] += 1
                    continue
                if (core & union_tls).any():
                    agg_rej["overlaps_tls"] += 1
                    continue
                items.append({"core": core, "n_cells_core": n_in,
                              "core_area_mm2": round(a_mm2, 4),
                              "core_density_mm2": round(dens, 1)})
        agg_out[tag] = items
        agg_reject[tag] = agg_rej
        arows = []
        for j, it in enumerate(items, 1):
            ys, xs = np.where(it["core"])
            r = {"sample": sample, "slide": slide, "agg_id": f"{'T' if tag == 'tagg' else 'M'}{j}",
                 "anchor": "NK_T (merged T/NK)" if tag == "tagg" else "Myeloid_NOS (not macrophage-specific)",
                 "n_cells_core": it["n_cells_core"], "core_area_mm2": it["core_area_mm2"],
                 "core_density_mm2": it["core_density_mm2"],
                 "cx_um": float(xs.mean() * RESOLUTION + x_min),
                 "cy_um": float(ys.mean() * RESOLUTION + y_min)}
            for h in HALOS_UM:
                reg = binary_dilation(it["core"], structure=disk(int(round(h / RESOLUTION)))) & tissue
                r[f"region_area_mm2_halo{int(h)}"] = round(area_of(reg), 4)
                r[f"n_cells_halo{int(h)}"] = int(reg[iy, ix].sum())
            arows.append(r)
        if arows:
            pd.DataFrame(arows).to_csv(os.path.join(outdir, f"{slide}_{tag}.csv"), index=False)


    percell = {"cell_type": ct}
    core_flag = np.zeros(len(df), bool)
    for it in accepted:
        core_flag |= it["core"][iy, ix]
    for h in HALOS_UM:
        lab_h = np.full(len(df), "Outside", dtype=object)
        for k, own in enumerate(owned[h], 1):
            lab_h[own[iy, ix]] = f"TLS-{k}"
        percell[f"tls_region_halo{int(h)}"] = lab_h
    percell["in_tls_core"] = core_flag
    pd.DataFrame(percell).to_csv(os.path.join(outdir, f"{slide}_tls_cells.csv"), index=False)

    meta = {"sample": sample, "slide": slide, "caveat": CAVEAT,
            "n_cells": int(len(df)),
            "n_B": int(sel_b.sum()), "n_T_NK": int(sel_t.sum()),
            "n_Plasma": int(sel_p.sum()), "n_Myeloid": int(sel_m.sum()),
            "b_fraction": round(float(sel_b.sum() / len(df)), 6),
            "tissue_area_mm2": round(area_of(tissue), 4),
            "n_candidate_cores": len(cand), "n_tls_accepted": len(accepted),
            "n_tls_rescued": int(sum(it["rescued"] for it in accepted)),
            "rescue_targets": rescue,
            "rejected_by": rejected,
            "n_tagg": len(agg_out["tagg"]), "n_magg": len(agg_out["magg"]),
            "aggregate_rejected_by": agg_reject,
            "canvas_px": [int(nx), int(ny)], "resolution_um": RESOLUTION,
            "ported_from": "TLS Define detector and reporter geometry",
            "dropped": ["TLS_State", "GC rung", "tls_development_5k",
                        "molecular descriptors", "Canonical LN",
                        "Lymphatic_Endothelial mature/immature flag",
                        "tumor zone class (no boundary_zone annotation)"],
            "merged": {"B_cell": "B_cell + NK_T (inseparable on H&E); n_T_NK is the NK_T subset"},
            "degraded": {"T": "NK_T only (source had 5 T classes)",
                         "myeloid": "Myeloid_NOS only (source had 3 macrophage classes)"}}
    with open(os.path.join(outdir, f"{slide}_meta.json"), "w") as fh:
        json.dump(meta, fh, indent=2)


    np.savez_compressed(
        os.path.join(outdir, f"{slide}_masks.npz"),
        cores=(np.stack([it["core"] for it in accepted]) if accepted else np.zeros((0, ny, nx), bool)),
        tagg=(np.stack([it["core"] for it in agg_out["tagg"]]) if agg_out["tagg"] else np.zeros((0, ny, nx), bool)),
        magg=(np.stack([it["core"] for it in agg_out["magg"]]) if agg_out["magg"] else np.zeros((0, ny, nx), bool)),
        x_min=x_min, y_min=y_min, x_max=x_max, y_max=y_max, resolution=RESOLUTION,


        cands=(np.stack([c for c, _ in cand]) if cand else np.zeros((0, ny, nx), bool)),
        cand_n_b=np.array([int(c[b_iy, b_ix].sum()) if len(b_ix) else 0 for c, _ in cand]),
        cand_area_mm2=np.array([float(c.sum()) * (RESOLUTION ** 2) / 1e6 for c, _ in cand]))
    if accepted:
        render_figure(slide, outdir, df, sel_b, sel_t, sel_p,
                      [it["core"] for it in accepted],
                      [it["core"] for it in agg_out["tagg"]],
                      [it["core"] for it in agg_out["magg"]],
                      x_min, x_max, y_min, y_max, ny, nx)

    print(f"{slide}: cells {len(df)}  B {int(sel_b.sum())} ({100*sel_b.sum()/len(df):.2f}%)  "
          f"cand {len(cand)} -> TLS {len(accepted)}  Tagg {len(agg_out['tagg'])}  "
          f"Magg {len(agg_out['magg'])}  rejected {rejected}  agg_rej {agg_reject}", flush=True)
    return 0


def render_figure(slide, outdir, df, sel_b, sel_t, sel_p, cores, tagg, magg,
                  x_min, x_max, y_min, y_max, ny, nx, id_map=None):
    fig, ax = plt.subplots(figsize=(14, 12))
    other = df[df["class"] == "Other"]
    ax.scatter(other.x_um, other.y_um, c="#d5d5d5", s=0.6, alpha=0.08,
               rasterized=True, zorder=0, edgecolors="none")
    overlay = np.zeros((ny, nx, 4), np.float32)
    def kind0(k):
        v = (id_map or {}).get(f"TLS-{k}", "")
        return "2d" if str(v).startswith("2d-") else "3d"
    for k, core in enumerate(cores, 1):
        rgba = list(mcolors.to_rgba("#2ca02c" if kind0(k) == "2d" else "#d62728")); rgba[3] = 0.20
        overlay[core] = rgba
    for masks, col in ((tagg, "#1f77b4"), (magg, "#d81b60")):
        rgba = list(mcolors.to_rgba(col)); rgba[3] = 0.20
        for core in masks:
            overlay[core] = rgba
    ax.imshow(overlay, extent=[x_min, x_max, y_min, y_max], origin="lower",
              aspect="auto", zorder=2, interpolation="nearest")
    sel_t = np.asarray(sel_t, bool)
    is_b_only = (df["class"] == "B_cell").values & ~sel_t
    for sel, col, al, z in (((df["class"] == "Plasma").values, "#e67e22", 0.8, 3),
                            (sel_t, "#3498db", 0.7, 4),
                            (is_b_only, "#8e44ad", 0.9, 5)):
        sub = df[sel]
        if len(sub):
            ax.scatter(sub.x_um, sub.y_um, c=col, s=0.8, alpha=al, rasterized=True,
                       zorder=z, edgecolors="none")

    def outline(mask, col):
        for c in find_contours(mask.astype(float), 0.5):
            ax.plot(c[:, 1] * RESOLUTION + x_min, c[:, 0] * RESOLUTION + y_min,
                    color=col, lw=1.2, alpha=0.9, zorder=9)


    COL_3D, COL_2D = "#d62728", "#2ca02c"
    def kind(k):
        v = (id_map or {}).get(f"TLS-{k}", "")
        return "3d" if str(v).startswith("3d-") else ("2d" if str(v).startswith("2d-") else "")
    def disp(v):
        v = str(v)
        return v.replace("3d-tls-", "3D-").replace("2d-tls-", "2D-") if v else "unlinked"
    core_col = [COL_2D if kind(k) == "2d" else COL_3D for k in range(1, len(cores) + 1)]
    for core, col in zip(cores, core_col):
        outline(core, col)
    for masks, col in ((tagg, "#1f77b4"), (magg, "#d81b60")):
        for core in masks:
            outline(core, col)
    if id_map is not None and cores:


        cx = np.array([np.nonzero(c)[1].mean() * RESOLUTION + x_min for c in cores])
        cy = np.array([np.nonzero(c)[0].mean() * RESOLUTION + y_min for c in cores])
        rad = np.array([np.sqrt(c.sum() * RESOLUTION ** 2 / np.pi) for c in cores])
        X0, X1, Y0, Y1 = df.x_um.min(), df.x_um.max(), df.y_um.min(), df.y_um.max()
        diag = float(np.hypot(X1 - X0, Y1 - Y0)); sep, pad = 0.05 * diag, 0.03 * diag
        d0x, d0y = cx - (X0 + X1) / 2, cy - (Y0 + Y1) / 2
        dn = np.hypot(d0x, d0y); dn[dn < 1e-6] = 1.0
        lx = cx + d0x / dn * (rad + pad); ly = cy + d0y / dn * (rad + pad)
        for _ in range(400):
            dx = lx[:, None] - lx[None, :]; dy = ly[:, None] - ly[None, :]
            dist = np.hypot(dx, dy); np.fill_diagonal(dist, 1e9)
            f = np.where(dist < sep, (sep - dist) / np.maximum(dist, 1e-6) * 0.5, 0.0)
            mvx = (dx * f).sum(1); mvy = (dy * f).sum(1)
            ex = lx[:, None] - cx[None, :]; ey = ly[:, None] - cy[None, :]
            ed = np.hypot(ex, ey); need = (rad[None, :] + pad) - ed
            g = np.where(need > 0, need / np.maximum(ed, 1e-6), 0.0)
            mvx += (ex * g).sum(1); mvy += (ey * g).sum(1)
            lx += np.clip(mvx, -sep, sep) * 0.35; ly += np.clip(mvy, -sep, sep) * 0.35
            lx = np.clip(lx, X0 - 0.10 * (X1 - X0), X1 + 0.10 * (X1 - X0))
            ly = np.clip(ly, Y0 - 0.10 * (Y1 - Y0), Y1 + 0.10 * (Y1 - Y0))
        for k, _core in enumerate(cores, 1):
            col = core_col[k - 1]
            ax.annotate(disp(id_map.get(f"TLS-{k}", "")), xy=(cx[k - 1], cy[k - 1]),
                        xytext=(lx[k - 1], ly[k - 1]), fontsize=8, fontweight="bold", color="#111",
                        ha="center", va="center", zorder=13, annotation_clip=False,
                        arrowprops=dict(arrowstyle="-", color=col, lw=0.8, alpha=0.7,
                                        shrinkA=0, shrinkB=0),
                        bbox=dict(boxstyle="round,pad=0.25", fc="white", ec=col, lw=0.8, alpha=0.95))
    ax.set_xlim(df.x_um.min(), df.x_um.max())
    ax.set_ylim(df.y_um.max(), df.y_um.min())
    ax.set_aspect("equal")
    ax.set_xlabel("X (µm)"); ax.set_ylabel("Y (µm)")
    handles = [Line2D([], [], marker="o", linestyle="none", markersize=7, markeredgecolor="none",
                      markerfacecolor=c, label=f"{n} ({int(v)})")
               for c, n, v in (("#8e44ad", "B_cell", is_b_only.sum()),
                               ("#3498db", "NK_T", sel_t.sum()),
                               ("#e67e22", "Plasma", sel_p.sum()))]
    handles += [Line2D([], [], linestyle="none", label="TLS gates pool B_cell + NK_T as B")]
    n3 = sum(1 for k in range(1, len(cores) + 1) if kind(k) == "3d")
    n2 = sum(1 for k in range(1, len(cores) + 1) if kind(k) == "2d")
    handles += ([Line2D([], [], color=COL_3D, lw=2, label=f"3-D TLS core, label 3D-NN ({n3})"),
                 Line2D([], [], color=COL_2D, lw=2, label=f"2-D-only TLS core, label 2D-NN ({n2})")]
                if id_map is not None and (n3 + n2) else
                [Line2D([], [], color=COL_3D, lw=2, label=f"TLS core ({len(cores)})" + (", labelled by id" if id_map is not None else ""))])
    handles += [
                Line2D([], [], color="#1f77b4", lw=2, label=f"T/NK aggregate ({len(tagg)})"),
                Line2D([], [], color="#d81b60", lw=2, label=f"Myeloid aggregate ({len(magg)})")]
    ax.legend(handles=handles, loc="upper left", bbox_to_anchor=(1.01, 1.0), fontsize=9)
    ax.set_title(f"{slide} — {len(cores)} TLS", fontsize=12, fontweight="bold")
    fig.text(0.01, 0.005, CAVEAT, fontsize=7.5, color="#444", wrap=True)
    fig.savefig(os.path.join(outdir, f"{slide}_tls.png"), dpi=180, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("sample")
    ap.add_argument("slide")
    ap.add_argument("--outdir", default=None)
    a = ap.parse_args()
    sys.exit(main(a.sample, a.slide, a.outdir or (os.path.join(OUT_ROOT, a.slide) if OUT_ROOT
                                                  else os.path.join(DEFAULT_OUT, a.sample, a.slide))))
