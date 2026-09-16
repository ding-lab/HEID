import paths as P
import os
PROJECTS_ROOT = os.environ.get("PROJECTS_ROOT", "/data/heid")
import json
import os
import sys

import numpy as np
import pandas as pd
from scipy.ndimage import (binary_closing, binary_dilation, binary_fill_holes,
                           binary_opening, distance_transform_edt,
                           gaussian_filter, label)
from skimage.feature import peak_local_max
from skimage.measure import find_contours
from skimage.morphology import disk
from skimage.segmentation import watershed

import matplotlib
matplotlib.use("Agg")
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator

REPO_ROOT = str(P.PROJECTS)
BASE = str(P.WORKSPACE)
H5AD_DIR = str(P.CELLS)


WORK_ROOT = str(P.ACCEPTED / "work")


ANCHORS = {
    "T":   {"label": "T_cell",     "tag": "tagg",
            "types": ["T_cell", "NK_T", "CD4_T", "CD8_T", "T_reg", "Treg"]},
    "MAC": {"label": "Macrophage", "tag": "magg",
            "types": ["Macrophage", "Macrophage_Monocyte", "Monocyte"]},
}

AGG_COL = {"T": "#1f77b4", "MAC": "#d81b60"}
CELL_POINT_SIZE = 1.0


RESOLUTION          = 10
SIGMA_PX            = 4
DENSITY_THRESH_REL  = 0.08
CLOSING_R_PX        = 10
MAX_HOLE_AREA_PX    = 50000
TISSUE_SIGMA        = 10
TISSUE_THRESH       = 0.015
PEAK_MIN_SEP_UM     = float(os.environ.get("AGG_PEAK_MIN_SEP_UM", "250"))
MERGE_SADDLE_FRAC   = float(os.environ.get("AGG_MERGE_SADDLE_FRAC", "0.5"))
CORE_SIGMA_PX       = float(os.environ.get("AGG_CORE_SIGMA_PX", "1.5"))
CORE_THRESH_REL     = float(os.environ.get("AGG_CORE_THRESH_REL", "0.30"))
CORE_CLOSING_PX     = int(os.environ.get("AGG_CORE_CLOSING_PX", "3"))
CORE_OPENING_PX     = int(os.environ.get("AGG_CORE_OPENING_PX", "2"))
CORE_NECK_RADIUS_PX = float(os.environ.get("AGG_CORE_NECK_RADIUS_PX", "3"))
CORE_HOLE_FILL_MAX_PX  = int(os.environ.get("AGG_CORE_HOLE_FILL_MAX_PX", "30"))
CORE_HOLE_CONNECT_FRAC = float(os.environ.get("AGG_CORE_HOLE_CONNECT_FRAC", "0.5"))
SPLIT_MAX_CORE_AREA_UM2 = float(os.environ.get("AGG_SPLIT_MAX_CORE_AREA_UM2", "100000"))
SPLIT_PEAK_SEP_UM       = float(os.environ.get("AGG_SPLIT_PEAK_SEP_UM", "150"))
SPLIT_MERGE_FRAC        = float(os.environ.get("AGG_SPLIT_MERGE_FRAC", "0.8"))
SPLIT_CORE_THRESH       = float(os.environ.get("AGG_SPLIT_CORE_THRESH", "0.45"))


TAGG_CORE_SIGMA_BIG_PX   = float(os.environ.get("TAGG_CORE_SIGMA_BIG_PX", "4.0"))
TAGG_BIG_COMP_AREA_UM2   = float(os.environ.get("TAGG_BIG_COMP_AREA_UM2", "300000"))
TAGG_MIN_DENSITY_BIG_MM2 = float(os.environ.get("TAGG_MIN_DENSITY_BIG_MM2", "7000"))


MIN_CELLS         = int(os.environ.get("AGG_MIN_CELLS", "50"))
MIN_CORE_AREA_UM2 = float(os.environ.get("AGG_MIN_CORE_AREA_UM2", "10000"))
MIN_DENSITY_MM2   = float(os.environ.get("AGG_MIN_DENSITY_CELLS_MM2", "8000"))


HALO_RADIUS_UM    = float(os.environ.get("AGG_HALO_RADIUS_UM", "150"))


def fill_small_holes(mask, max_area):
    filled = binary_fill_holes(mask)
    holes = filled & ~mask
    if not holes.any():
        return mask
    lb, n = label(holes)
    out = mask.copy()
    for h in range(1, n + 1):
        hm = lb == h
        if int(hm.sum()) <= max_area:
            out |= hm
    return out


def density_mask(raw):
    d = gaussian_filter(raw, sigma=SIGMA_PX)
    if d.max() <= 0:
        return np.zeros_like(raw, dtype=bool)
    m = d > DENSITY_THRESH_REL * d.max()
    m = binary_closing(m, structure=disk(CLOSING_R_PX))
    return fill_small_holes(m, MAX_HOLE_AREA_PX)


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
        col = np.minimum(dens[s_lo][diff], dens[s_hi][diff])
        for u, v, c in zip(a[diff], b[diff], col):
            k = (u, v) if u < v else (v, u)
            if c > saddle.get(k, 0.0):
                saddle[k] = float(c)
    parent = list(range(nlab + 1))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]; x = parent[x]
        return x

    for (u, v), col in saddle.items():
        if col >= frac * min(peak_d[u], peak_d[v]):
            parent[find(u)] = find(v)
    out = np.zeros_like(ws); remap = {}; nxt = 1
    for j in range(1, nlab + 1):
        r = find(j)
        if r not in remap:
            remap[r] = nxt; nxt += 1
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
            hd = float(dens_c[hole].mean()) if ha else 0.0
            if ha <= CORE_HOLE_FILL_MAX_PX or hd >= CORE_HOLE_CONNECT_FRAC * thr:
                core |= hole
    edt = distance_transform_edt(core)
    if edt.max() < CORE_NECK_RADIUS_PX:
        return []
    seeds, n = label(edt >= CORE_NECK_RADIUS_PX)
    if n <= 1:
        return [binary_fill_holes(core)]
    ws = watershed(-edt, seeds, mask=core, watershed_line=True)
    return [binary_fill_holes(ws == b) for b in range(1, n + 1) if (ws == b).any()]


def split_oversized_core(core, dens, iy, ix):
    sep_px = max(int(SPLIT_PEAK_SEP_UM / RESOLUTION), 3)
    peaks = peak_local_max(dens, min_distance=sep_px, labels=core, exclude_border=False)
    if len(peaks) <= 1:
        return [core]
    markers = np.zeros_like(core, dtype=np.int32)
    for i, (yy, xx) in enumerate(peaks, 1):
        markers[yy, xx] = i
    ws = merge_shallow_basins(watershed(-dens, markers, mask=core), dens, SPLIT_MERGE_FRAC)
    out = []
    for j in range(1, int(ws.max()) + 1):
        sub = ws == j
        if not sub.any():
            continue
        in_sub = sub[iy, ix]
        raw_c = np.zeros_like(dens, dtype=np.float32)
        if in_sub.any():
            np.add.at(raw_c, (iy[in_sub], ix[in_sub]), 1)
        dens_c = gaussian_filter(raw_c, sigma=CORE_SIGMA_PX)
        if dens_c.max() <= 0:
            continue
        thr = SPLIT_CORE_THRESH * dens_c.max()
        tight = (dens_c > thr) & sub
        if CORE_CLOSING_PX > 0:
            tight = binary_closing(tight, structure=disk(CORE_CLOSING_PX)) & sub
        lb_p, n_p = label(tight)
        for pid in range(1, n_p + 1):
            out.extend(clean_core(lb_p == pid, dens_c, thr))
    return out if out else [core]


def main(dataset, cancer, sample, anchor_key):
    A = ANCHORS[anchor_key]
    tag = A["tag"]
    is_tagg = anchor_key == "T"
    p = f"{H5AD_DIR}/{sample}.parquet"
    if not os.path.exists(p):
        sys.exit(f"missing input {p}")
    df = pd.read_parquet(p, columns=["cell_type", "x_centroid", "y_centroid"])
    print(f"[1] {dataset}/{cancer}/{sample}  {len(df)} cells  anchor={anchor_key}", flush=True)

    sub = df[df.cell_type.isin(A["types"])]
    n_anchor = len(sub)
    present = sorted(set(df.cell_type) & set(A["types"]))
    print(f"    anchor cells ({A['label']}): {n_anchor}  from types {present}", flush=True)

    outdir = f"{WORK_ROOT}/{dataset}/{cancer}/{sample}"
    os.makedirs(outdir, exist_ok=True)
    meta = {"sample": sample, "dataset": dataset, "cancer_type": cancer,
            "anchor": anchor_key, "anchor_label": A["label"],
            "anchor_types_present": present,
            "n_cells_total": int(len(df)), "n_anchor_total": int(n_anchor),
            "gates": {"min_cells": MIN_CELLS, "min_core_area_um2": MIN_CORE_AREA_UM2,
                      "min_density_cells_mm2": MIN_DENSITY_MM2},
            "n_agg": 0, "agg": []}

    if n_anchor < MIN_CELLS:
        print(f"    too few anchor cells ({n_anchor} < {MIN_CELLS}); no aggregate possible.", flush=True)
        np.savez_compressed(f"{outdir}/{sample}_{tag}_grid.npz",
                            grid=np.zeros((1, 1), np.int16), empty=True)
        return

    x_min, x_max = df.x_centroid.min() - 200, df.x_centroid.max() + 200
    y_min, y_max = df.y_centroid.min() - 200, df.y_centroid.max() + 200
    nx = int(np.ceil((x_max - x_min) / RESOLUTION))
    ny = int(np.ceil((y_max - y_min) / RESOLUTION))

    def to_px(s):
        ix = np.clip(((s.x_centroid.values - x_min) / RESOLUTION).astype(int), 0, nx - 1)
        iy = np.clip(((s.y_centroid.values - y_min) / RESOLUTION).astype(int), 0, ny - 1)
        return ix, iy

    all_ix, all_iy = to_px(df)
    all_raw = np.zeros((ny, nx), np.float32); np.add.at(all_raw, (all_iy, all_ix), 1)
    tissue = gaussian_filter(all_raw, sigma=TISSUE_SIGMA) > TISSUE_THRESH
    tissue = binary_closing(tissue, structure=disk(10))

    ix, iy = to_px(sub)
    raw = np.zeros((ny, nx), np.float32); np.add.at(raw, (iy, ix), 1)
    dens = gaussian_filter(raw, sigma=SIGMA_PX)
    mask = density_mask(raw) & tissue
    lb, n_cc = label(mask)
    print(f"[2] {A['label']} mask: {n_cc} raw components", flush=True)


    cores = []
    if n_cc > 0:
        sep_px = max(int(PEAK_MIN_SEP_UM / RESOLUTION), 3)
        new_lb = np.zeros_like(lb); nxt = 1
        for cid in range(1, n_cc + 1):
            comp = lb == cid
            peaks = peak_local_max(dens, min_distance=sep_px, labels=comp, exclude_border=False)
            if len(peaks) <= 1:
                new_lb[comp] = nxt; nxt += 1; continue
            markers = np.zeros_like(comp, dtype=np.int32)
            for i, (yy, xx) in enumerate(peaks, 1):
                markers[yy, xx] = i
            ws = merge_shallow_basins(watershed(-dens, markers, mask=comp), dens, MERGE_SADDLE_FRAC)
            for j in range(1, int(ws.max()) + 1):
                m = ws == j
                if m.any():
                    new_lb[m] = nxt; nxt += 1
        lb = new_lb; n_cc = nxt - 1
        print(f"    density-peak split -> {n_cc} sub-cores", flush=True)


        n_big = 0
        for cid in range(1, n_cc + 1):
            comp = lb == cid
            in_c = comp[iy, ix]
            if not in_c.any():
                continue
            is_big = is_tagg and int(comp.sum()) * (RESOLUTION ** 2) > TAGG_BIG_COMP_AREA_UM2
            n_big += int(is_big)
            core_sigma = TAGG_CORE_SIGMA_BIG_PX if is_big else CORE_SIGMA_PX
            raw_c = np.zeros((ny, nx), np.float32)
            np.add.at(raw_c, (iy[in_c], ix[in_c]), 1)
            dens_c = gaussian_filter(raw_c, sigma=core_sigma)
            if dens_c.max() <= 0:
                continue
            thr = CORE_THRESH_REL * dens_c.max()
            tight = (dens_c > thr) & comp
            if CORE_CLOSING_PX > 0:
                tight = binary_closing(tight, structure=disk(CORE_CLOSING_PX)) & comp
            if CORE_OPENING_PX > 0:
                tight = binary_opening(tight, structure=disk(CORE_OPENING_PX)) & comp
            lb_p, n_p = label(tight)
            for pid in range(1, n_p + 1):
                cores.extend((c, is_big) for c in clean_core(lb_p == pid, dens_c, thr))
        if is_tagg:
            print(f"    tight-core re-split -> {len(cores)} dense cores "
                  f"(adaptive σ small={CORE_SIGMA_PX}/big={TAGG_CORE_SIGMA_BIG_PX}px, "
                  f"{n_big} big components)", flush=True)
        else:
            print(f"    tight-core re-split -> {len(cores)} dense cores", flush=True)


        max_px = SPLIT_MAX_CORE_AREA_UM2 / (RESOLUTION ** 2)
        expanded = []
        for c, is_big in cores:
            if (not is_big) and c.sum() > max_px:
                expanded.extend((piece, is_big)
                                for piece in split_oversized_core(c, dens, iy, ix))
            else:
                expanded.append((c, is_big))
        cores = expanded
        print(f"    size-conditional split -> {len(cores)} cores total", flush=True)


    min_area_px = int(MIN_CORE_AREA_UM2 / (RESOLUTION ** 2))
    rej = {"size": 0, "density": 0}
    grid = np.zeros((ny, nx), np.int32)
    kept = []
    for c, is_big in cores:
        n_in = int(c[iy, ix].sum())
        area_px = int(c.sum())
        area_mm2 = area_px * (RESOLUTION ** 2) / 1e6
        if n_in < MIN_CELLS or area_px < min_area_px:
            rej["size"] += 1; continue
        d = n_in / area_mm2 if area_mm2 > 0 else 0
        density_gate = TAGG_MIN_DENSITY_BIG_MM2 if is_big else MIN_DENSITY_MM2
        if d < density_gate:
            rej["density"] += 1; continue
        kept.append((c, n_in, area_mm2, d, is_big))

    halo_r_px = int(round(HALO_RADIUS_UM / RESOLUTION))
    region_grid = np.zeros((ny, nx), np.int32)
    clusters = []
    for i, (c, n_in, area_mm2, d, is_big) in enumerate(kept, 1):
        grid[c] = i
        reg = binary_dilation(c, structure=disk(halo_r_px)) & tissue
        region_grid[reg] = i
        ys, xs = np.where(c)
        rys, rxs = np.where(reg)
        cx = float(xs.mean() * RESOLUTION + x_min)
        cy = float(ys.mean() * RESOLUTION + y_min)
        max_r_um = float(np.max(np.hypot(rxs - rxs.mean(), rys - rys.mean())) * RESOLUTION) if len(rxs) else 1.0
        reg_area_mm2 = int(reg.sum()) * (RESOLUTION ** 2) / 1e6
        n_in_region = int(reg[iy, ix].sum())
        rec = {
            "id": i,
            "n_cells": int(n_in),
            "area_mm2": round(float(area_mm2), 4),
            "density_cells_mm2": round(float(d), 1),
            "n_cells_region": n_in_region,
            "region_area_mm2": round(float(reg_area_mm2), 4),
            "cx": cx, "cy": cy, "max_r_um": round(max_r_um, 1),
        }
        meta["agg"].append(rec)
        clusters.append({**rec, "mask": c, "region": reg})
    meta["halo_radius_um"] = HALO_RADIUS_UM
    meta["n_agg"] = len(kept)
    meta["rejected"] = rej
    print(f"[3] Rejected: {rej}", flush=True)
    print(f"    {A['label']} aggregates: {len(kept)}", flush=True)


    np.savez_compressed(
        f"{outdir}/{sample}_{tag}_grid.npz",
        grid=grid.astype(np.int16),
        ids=np.array([c["id"] for c in clusters], np.int32),
        n_cells=np.array([c["n_cells"] for c in clusters], np.int32),
        core_area_mm2=np.array([c["area_mm2"] for c in clusters], np.float32),
        density_cells_mm2=np.array([c["density_cells_mm2"] for c in clusters], np.float32),
        cx=np.array([c["cx"] for c in clusters], np.float64),
        cy=np.array([c["cy"] for c in clusters], np.float64),
        x_min=x_min, y_min=y_min, resolution=RESOLUTION, halo_radius_um=HALO_RADIUS_UM)


    print(f"[4] wrote {outdir}/{sample}_{tag}_*", flush=True)


if __name__ == "__main__":
    if len(sys.argv) != 5:
        sys.exit("usage: agg_cluster.py <dataset> <cancer_type> <sample> <T|MAC>")
    main(sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4].upper())
