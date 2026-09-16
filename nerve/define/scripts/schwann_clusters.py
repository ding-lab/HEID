
from __future__ import annotations
import os
PROJECTS_ROOT = os.environ.get("PROJECTS_ROOT", "/data/heid")

import argparse
import time
from pathlib import Path

import h5py
import matplotlib
import numpy as np
import pandas as pd
from scipy.ndimage import (
    binary_closing,
    binary_dilation,
    binary_opening,
    convolve,
    distance_transform_edt,
    gaussian_filter,
    gaussian_filter1d,
    label,
)
from scipy.spatial import cKDTree
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components, minimum_spanning_tree
from skimage.draw import line as skline
from skimage.feature import peak_local_max
from skimage.measure import find_contours, regionprops
from skimage.morphology import disk
from skimage.segmentation import watershed

matplotlib.use("Agg")
import matplotlib.colors as mcolors
import matplotlib.patheffects as pe
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.ticker import MultipleLocator


PROJ_ROOT = Path(__file__).resolve().parents[1]
INPUT_ROOT = Path(PROJECTS_ROOT) / "nerve/define"
XENIUM_ROOT = INPUT_ROOT / "xenium"
CACHE_477 = INPUT_ROOT / "annotations/477"
REANNOT_CACHE = INPUT_ROOT / "annotations/5k"
OUT_ROOT = PROJ_ROOT / "outputs"


CANCERS = ["PRAD", "PDAC", "MCRC", "CRC", "CHOL"]
PANELS = ["5k", "477"]
CANCER_DIR = {"PRAD": "PRAD", "PDAC": "PDAC", "MCRC": "CRC", "CRC": "CRC", "CHOL": "CHOL"}
TUMOR_PREFIX = "Tumor"


RESOLUTION = 10
SIGMA_PX = 4
MIN_SCHWANN_CELLS = 5
HIGH_CONF_CELLS = 15


CORE_THRESH_REL = 0.30


CORE_THRESH_ABS_BY_PANEL = {"5k": 0.086, "477": 0.096}


COUNT_RADIUS_UM = 35

COUNT_MIN_CELLS = 5
OPEN_RADIUS_UM = 10
SMOOTH_CLOSE_UM = 20


FOOT_RADIUS_UM = 20


GRAB_UM = 30


HALO_SINGLE_UM = 4.0
HALO_MULTI_UM = 3.0
HALO_WEIGHT_FULL = 5.0
EXPR_WEIGHT = 2.0


ADAPT_REF_DENSITY = 90.0
ADAPT_LINK_POWER = 0.165
ADAPT_LINK_MAX = 1.8
ADAPT_GRAB_POWER = 0.24
ADAPT_GRAB_MAX = 2.2
ADAPT_SCALE_MIN = 0.9
REPORT_CORE_REL = 0.40


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


def required_areal_density(area_mm2: float) -> float:
    return AREAL_FLOOR_SMALL if area_mm2 <= AREA_CUT_MM2 else AREAL_FLOOR_LARGE


def _linkage_labels(pts_px, eps_px):
    n = len(pts_px)
    if n <= 1:
        return n, np.zeros(n, dtype=int)
    pairs = cKDTree(pts_px).query_pairs(eps_px, output_type="ndarray")
    if len(pairs) == 0:
        return n, np.arange(n)
    m = csr_matrix((np.ones(len(pairs)), (pairs[:, 0], pairs[:, 1])), shape=(n, n))
    return connected_components(m, directed=False)


def refine_components(lb, n_comps, density, thresh, s_iy, s_ix):
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
                new[comp] = nxt; nxt += 1
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
                    new[segp[:, 0], segp[:, 1]] = nxt; nxt += 1
            continue

        peaks = peak_local_max(density, min_distance=PEAK_MIN_DIST_PX,
                               threshold_abs=thresh,
                               labels=comp, exclude_border=False)
        if len(peaks) <= 1:
            new[comp] = nxt; nxt += 1
            continue
        markers = np.zeros(lb.shape, dtype=np.int32)
        for j, (yy, xx) in enumerate(peaks, start=1):
            markers[yy, xx] = j
        ws = watershed(-density, markers, mask=comp)
        for j in range(1, len(peaks) + 1):
            seg = ws == j
            if seg.any():
                new[seg] = nxt; nxt += 1
    return new, nxt - 1


MAX_MEDIAN_NN_UM = 30.0


TUMOR_LEGACY_RADIUS = 50
PROX_BANDS_UM = [27, 100, 1000]


SOX10_GENE = "SOX10"
SOX10_MIN_COUNT = 1
MIN_SOX10_CELLS = 3


NGFR_GENE = "NGFR"
NGFR_MIN_COUNT = 1
MIN_NGFR_CELLS = 3


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


RESCUE_MIN_CELLS = 3
RESCUE_MAX_NN_UM = 15.0
RESCUE_LINK_UM = 20.0


def schwann_label(panel: str) -> str:
    return "Glial" if panel == "477" else "Schwann"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("panel", choices=PANELS)
    p.add_argument("cancer", choices=CANCERS)
    p.add_argument("sample")
    return p.parse_args()


def load_coords(panel: str, cancer: str, sample: str) -> pd.DataFrame:
    xendir = XENIUM_ROOT / panel / CANCER_DIR[cancer] / sample
    for fname in ["cells.parquet", "cells.csv.gz", "cells.csv"]:
        p = xendir / fname
        if p.exists():
            if p.suffix == ".parquet":
                c = pd.read_parquet(p, columns=["cell_id", "x_centroid", "y_centroid"])
            else:
                c = pd.read_csv(p, usecols=["cell_id", "x_centroid", "y_centroid"])
            c["cell_id"] = c["cell_id"].astype(str)
            return c
    raise FileNotFoundError(f"No cells file under {xendir}")


def load_cell_types(panel: str, cancer: str, sample: str) -> pd.DataFrame:


    if panel == "477":
        f = CACHE_477 / cancer / f"{sample}.csv"
    else:
        f = REANNOT_CACHE / f"{sample}.csv"
    if not f.exists():
        raise FileNotFoundError(f"No cell-type cache: {f}")
    return pd.read_csv(f, usecols=["cell_id", "cell_type"], dtype=str, keep_default_na=False)


def load_genes(panel: str, cancer: str, sample: str, genes: dict):
    h5 = XENIUM_ROOT / panel / CANCER_DIR[cancer] / sample / "cell_feature_matrix.h5"
    if not h5.exists():
        raise FileNotFoundError(f"No cell_feature_matrix.h5 under {h5.parent}")
    out = {}
    with h5py.File(h5, "r") as fh:
        m = fh["matrix"]
        names = np.array(m["features"]["name"]).astype(str)
        idx = {g: np.where(names == g)[0] for g in genes}
        if all(len(v) == 0 for v in idx.values()):
            return {g: None for g in genes}
        barcodes = np.array(m["barcodes"]).astype(str)
        data = np.array(m["data"]); indices = np.array(m["indices"]); indptr = np.array(m["indptr"])
    for g, min_count in genes.items():
        gi = idx[g]
        if len(gi) == 0:
            out[g] = None
            continue
        pos = np.nonzero(indices == int(gi[0]))[0]
        cells = np.searchsorted(indptr, pos, side="right") - 1
        out[g] = {barcodes[c]: int(v) for c, v in zip(cells, data[pos]) if int(v) >= min_count}
    return out


STAT_COLS = ["cluster_id", "status", "shape_class", "confidence", "n_schwann",
             "area_mm2", "areal_density", "linear_density", "aspect_ratio",
             "minor_axis_um", "major_axis_um", "solidity", "eccentricity",
             "core_frac", "median_nn_um", "n_sox10", "n_ngfr", "passA", "passB", "passed",
             "reject_reason", "n_tumor_inside", "n_tumor_le27", "n_tumor_le100",
             "n_tumor_le1000", "n_tumor_le50"]


def _layout_labels(half_sizes, pref, anchors, rads, xlim, ylim, iters=600):
    pos = pref.copy().astype(float)
    n = len(pos)
    if n == 0:
        return pos
    x0, x1 = xlim
    y0, y1 = ylim
    mx, my = (x1 - x0) * 0.01, (y1 - y0) * 0.01
    half_diag = np.hypot(half_sizes[:, 0], half_sizes[:, 1])
    pad = (x1 - x0 + y1 - y0) * 0.0015
    for _ in range(iters):
        disp = np.zeros((n, 2))
        for i in range(n):
            for j in range(i + 1, n):
                ddx = pos[i, 0] - pos[j, 0]
                ddy = pos[i, 1] - pos[j, 1]
                ox = (half_sizes[i, 0] + half_sizes[j, 0]) - abs(ddx)
                oy = (half_sizes[i, 1] + half_sizes[j, 1]) - abs(ddy)
                if ox > 0 and oy > 0:
                    if ox <= oy:
                        s = ox * 0.5 * (1.0 if ddx >= 0 else -1.0)
                        disp[i, 0] += s; disp[j, 0] -= s
                    else:
                        s = oy * 0.5 * (1.0 if ddy >= 0 else -1.0)
                        disp[i, 1] += s; disp[j, 1] -= s
        for i in range(n):
            dx = pos[i, 0] - anchors[:, 0]
            dy = pos[i, 1] - anchors[:, 1]
            dist = np.hypot(dx, dy); dist[dist < 1e-6] = 1e-6
            over = (rads + half_diag[i] + pad) - dist
            hit = over > 0
            if np.any(hit):
                disp[i, 0] += float(np.sum(over[hit] / dist[hit] * dx[hit])) * 0.5
                disp[i, 1] += float(np.sum(over[hit] / dist[hit] * dy[hit])) * 0.5
        disp += (pref - pos) * 0.03
        pos += disp
        np.clip(pos[:, 0], x0 + mx, x1 - mx, out=pos[:, 0])
        np.clip(pos[:, 1], y0 + my, y1 - my, out=pos[:, 1])
    return pos


def detect_tracts(sch_xy, sox_pos, ngfr_pos, sox_avail, ngfr_avail,
                  x_min, y_min, nx, ny, next_cid, link_um):
    out = []
    n = len(sch_xy)
    if n < TRACT_MIN_CELLS:
        return out, next_cid
    pairs = cKDTree(sch_xy).query_pairs(link_um, output_type="ndarray")
    if len(pairs) == 0:
        return out, next_cid
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
        nsox = int(sox_pos[idx].sum()) if sox_avail else -1
        nngfr = int(ngfr_pos[idx].sum()) if ngfr_avail else -1
        sox_ok = nsox >= MIN_SOX10_CELLS
        ngfr_resc = (not sox_ok) and nsox >= 1 and ngfr_avail and nngfr >= MIN_NGFR_CELLS
        if sox_avail and not (sox_ok or ngfr_resc):
            continue
        area = int(comp.sum()) * RESOLUTION ** 2 / 1e6
        out.append(dict(
            n_schwann=len(idx), area_mm2=area, areal_density=len(idx) / area if area else 0.0,
            linear_density=lin, aspect_ratio=aspect, minor_axis_um=minor, major_axis_um=major,
            solidity=float(r.solidity), eccentricity=float(r.eccentricity), core_frac=0.0,
            median_nn_um=nn, n_sox10=nsox, n_ngfr=nngfr, passA=False, passB=True, passed=True,
            reject_reason="", shape_class="elongated",
            confidence="high" if len(idx) >= HIGH_CONF_CELLS else "low",
            comp=comp, cluster_id=next_cid))
        next_cid += 1
    return out, next_cid


def main() -> None:
    args = parse_args()
    panel, cancer, sample = args.panel, args.cancer, args.sample
    SCHWANN = schwann_label(panel)
    out_group = cancer if panel == "5k" else f"{cancer}_477"
    CORE_THRESH_ABS = CORE_THRESH_ABS_BY_PANEL[panel]
    fig_dir = OUT_ROOT / out_group / "figures"
    data_dir = OUT_ROOT / out_group / "data"
    fig_dir.mkdir(parents=True, exist_ok=True)
    data_dir.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    print(f"[1] Loading {panel}/{cancer}/{sample} (Schwann='{SCHWANN}')...", flush=True)
    coords = load_coords(panel, cancer, sample)
    types = load_cell_types(panel, cancer, sample)
    df = coords.merge(types, on="cell_id", how="inner")
    if len(df) == 0:
        raise RuntimeError(f"0 cells after merge (coords={len(coords)}, types={len(types)})")

    schwann_cells = df[df.cell_type == SCHWANN]
    tumor_cells = df[df.cell_type.astype(str).str.startswith(TUMOR_PREFIX)]
    n_schwann, n_tumor = len(schwann_cells), len(tumor_cells)
    print(f"  {len(df)} merged cells, {n_schwann} {SCHWANN}, {n_tumor} Tumor", flush=True)

    if n_schwann < MIN_SCHWANN_CELLS:
        print(f"  Too few Schwann ({n_schwann}); skip.", flush=True)
        pd.DataFrame(columns=STAT_COLS).to_csv(data_dir / f"{sample}_cluster_stats.csv", index=False)
        return


    gene_counts = load_genes(panel, cancer, sample,
                             {SOX10_GENE: SOX10_MIN_COUNT, NGFR_GENE: NGFR_MIN_COUNT})
    sox10, ngfr = gene_counts[SOX10_GENE], gene_counts[NGFR_GENE]
    sox_avail = sox10 is not None
    ngfr_avail = ngfr is not None
    if sox_avail:
        sids = schwann_cells["cell_id"].to_numpy()
        sox_pos = np.fromiter((s in sox10 for s in sids), dtype=bool, count=len(sids))
        sox_cnt = np.fromiter((sox10.get(s, 0) for s in sids), dtype=int, count=len(sids))
        print(f"  SOX10 gate ON: {int(sox_pos.sum())}/{n_schwann} Schwann SOX10+ "
              f"(need >= {MIN_SOX10_CELLS}/region to keep)", flush=True)
    else:
        sox_pos = None
        sox_cnt = None
        print(f"  SOX10 not in {panel} panel -> gene gate SKIPPED (n_sox10=-1)", flush=True)
    if ngfr_avail:
        ngfr_pos = np.fromiter((s in ngfr for s in schwann_cells["cell_id"].to_numpy()),
                               dtype=bool, count=n_schwann)
        print(f"  NGFR rescue ON: {int(ngfr_pos.sum())}/{n_schwann} Schwann NGFR+ "
              f"(a region with 1-{MIN_SOX10_CELLS - 1} SOX10+ is kept if it has "
              f">= {MIN_NGFR_CELLS} NGFR+)", flush=True)
    else:
        ngfr_pos = None
        print(f"  NGFR not in {panel} panel -> rescue not applicable (n_ngfr=-1)", flush=True)


    x_min, x_max = df.x_centroid.min() - 200, df.x_centroid.max() + 200
    y_min, y_max = df.y_centroid.min() - 200, df.y_centroid.max() + 200
    nx = int(np.ceil((x_max - x_min) / RESOLUTION))
    ny = int(np.ceil((y_max - y_min) / RESOLUTION))

    def to_px(sub):
        ix = np.clip(((sub.x_centroid.values - x_min) / RESOLUTION).astype(int), 0, nx - 1)
        iy = np.clip(((sub.y_centroid.values - y_min) / RESOLUTION).astype(int), 0, ny - 1)
        return ix, iy


    print("[2] KDE + mask + components...", flush=True)
    s_ix, s_iy = to_px(schwann_cells)
    raw = np.zeros((ny, nx), dtype=np.float32)
    np.add.at(raw, (s_iy, s_ix), 1.0)
    density = gaussian_filter(raw, sigma=SIGMA_PX)
    gmax = float(density.max())
    if gmax <= 0:
        print("  No density; abort.", flush=True)
        pd.DataFrame(columns=STAT_COLS).to_csv(data_dir / f"{sample}_cluster_stats.csv", index=False)
        return


    count_r = max(int(round(COUNT_RADIUS_UM / RESOLUTION)), 1)
    local_count = convolve(raw, disk(count_r).astype(np.float32), mode="constant", cval=0.0)
    core_mask = local_count >= COUNT_MIN_CELLS
    open_r = max(int(round(OPEN_RADIUS_UM / RESOLUTION)), 1)
    mask = binary_opening(core_mask, structure=disk(open_r))
    report_core = density > CORE_THRESH_ABS * (REPORT_CORE_REL / CORE_THRESH_REL)
    lb, n_comps = label(mask)


    lb, n_comps = refine_components(lb, n_comps, density, CORE_THRESH_ABS, s_iy, s_ix)


    if n_comps:
        close_r = max(int(round(SMOOTH_CLOSE_UM / RESOLUTION)), 1)
        cr = disk(close_r)
        sm = np.zeros_like(lb)
        for i in range(1, n_comps + 1):
            ci = binary_closing(lb == i, structure=cr)
            sm[ci & (sm == 0)] = i
        lb = sm
    print(f"  Components after shape-aware split: {n_comps}", flush=True)

    s_labels = lb[s_iy, s_ix]
    counts = np.bincount(s_labels, minlength=n_comps + 1)
    sch_xy = schwann_cells[["x_centroid", "y_centroid"]].values


    props = {r.label: r for r in regionprops(lb)}


    print("[3] Geometry + shape-aware dual-gate...", flush=True)
    cluster_grid = np.full((ny, nx), -1, dtype=int)
    clusters, scattered = [], []
    cid = 1
    for i in range(1, n_comps + 1):
        n_in = int(counts[i])
        if n_in < MIN_SCHWANN_CELLS:
            continue
        r = props.get(i)
        comp = lb == i
        area_px = int(comp.sum())
        area_mm2 = area_px * RESOLUTION ** 2 / 1e6
        areal_density = n_in / area_mm2 if area_mm2 > 0 else 0.0
        core_frac = float((comp & report_core).sum()) / area_px if area_px else 0.0
        major_um = float(r.axis_major_length) * RESOLUTION if r else 0.0
        minor_um = float(r.axis_minor_length) * RESOLUTION if r else 0.0
        aspect = (major_um / minor_um) if minor_um > 0 else (999.0 if major_um > 0 else 1.0)
        solidity = float(r.solidity) if r else 0.0
        ecc = float(r.eccentricity) if r else 0.0
        linear_density = n_in / (major_um / 1000.0) if major_um > 0 else 0.0


        memb = sch_xy[s_labels == i]
        if len(memb) >= 2:
            d, _ = cKDTree(memb).query(memb, k=2)
            median_nn = float(np.median(d[:, 1]))
        else:
            median_nn = float("nan")


        if sox_avail:
            n_sox10 = int(sox_pos[s_labels == i].sum())
            n_ngfr = int(ngfr_pos[s_labels == i].sum()) if ngfr_avail else -1
            sox_ok = n_sox10 >= MIN_SOX10_CELLS
            ngfr_rescue = (not sox_ok) and n_sox10 >= 1 and ngfr_avail and n_ngfr >= MIN_NGFR_CELLS
            gene_ok = sox_ok or ngfr_rescue
        else:
            n_sox10 = -1
            n_ngfr = -1
            gene_ok = True


        elongated_shape = aspect >= MIN_ASPECT
        not_loose = median_nn <= MAX_MEDIAN_NN_UM
        req_areal = required_areal_density(area_mm2)
        dense_enough = areal_density >= req_areal
        if elongated_shape:
            passA = False
            passB = (minor_um >= MIN_MINOR_UM and major_um >= MIN_MAJOR_UM
                     and linear_density >= MIN_LINEAR_DENSITY
                     and dense_enough
                     and area_mm2 >= MIN_AREA_ELONG
                     and not_loose)
        else:
            passB = False
            passA = (dense_enough and solidity >= MIN_SOLIDITY
                     and area_mm2 >= MIN_AREA_COMPACT
                     and not_loose)
        passed = (passA or passB) and gene_ok

        reasons = []
        if not passed:
            if (passA or passB) and not gene_ok:


                reasons.append("no-sox10" if n_sox10 == 0 else "low-sox10-no-ngfr-rescue")
            if not not_loose:
                reasons.append("loose")
            if not dense_enough:

                reasons.append("low-density-for-area" if area_mm2 > AREA_CUT_MM2 else "sparse")
            if elongated_shape:
                if linear_density < MIN_LINEAR_DENSITY:
                    reasons.append("long-sparse")
                if minor_um < MIN_MINOR_UM:
                    reasons.append("too-thin")
                if major_um < MIN_MAJOR_UM:
                    reasons.append("too-short")
                if area_mm2 < MIN_AREA_ELONG:
                    reasons.append("too-small")
            else:
                if solidity < MIN_SOLIDITY:
                    reasons.append("diffuse")
                if area_mm2 < MIN_AREA_COMPACT:
                    reasons.append("too-small")
            if not reasons:
                reasons.append("fails-gate")
        shape_class = ("elongated" if elongated_shape else "compact") if passed else ""

        rec = dict(n_schwann=n_in, area_mm2=area_mm2, areal_density=areal_density,
                   linear_density=linear_density, aspect_ratio=aspect, minor_axis_um=minor_um,
                   major_axis_um=major_um, solidity=solidity, eccentricity=ecc,
                   core_frac=core_frac, median_nn_um=median_nn, n_sox10=n_sox10, n_ngfr=n_ngfr,
                   passA=passA, passB=passB,
                   passed=passed, reject_reason=";".join(reasons),
                   shape_class=shape_class,
                   confidence=("high" if n_in >= HIGH_CONF_CELLS else "low") if passed else "",
                   comp=comp)
        if passed:
            cluster_grid[comp] = cid
            rec["cluster_id"] = cid
            clusters.append(rec)
            cid += 1
        else:
            rec["cluster_id"] = 0
            scattered.append(rec)


    sch_density = n_schwann / (((x_max - x_min) * (y_max - y_min)) / 1e6)
    r = ADAPT_REF_DENSITY / max(sch_density, 1e-6)
    link_um = TRACT_LINK_UM * float(np.clip(r ** ADAPT_LINK_POWER, ADAPT_SCALE_MIN, ADAPT_LINK_MAX))
    grab_um = GRAB_UM * float(np.clip(r ** ADAPT_GRAB_POWER, ADAPT_SCALE_MIN, ADAPT_GRAB_MAX))
    print(f"  adaptive: {sch_density:.1f} Schwann/mm2 -> tract-link {link_um:.0f}um, "
          f"grab {grab_um:.0f}um", flush=True)


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
    tracts, _ = detect_tracts(sch_xy[elig],
                              sox_pos[elig] if sox_avail else None,
                              ngfr_pos[elig] if ngfr_avail else None,
                              sox_avail, ngfr_avail, x_min, y_min, nx, ny, 1, link_um)
    if tracts:


        kept_tracts, to_remove = [], set()
        for t in tracts:
            covered = [i for i, c in enumerate(clusters)
                       if (c["comp"] & t["comp"]).sum() / max(int(c["comp"].sum()), 1) > TRACT_SUBSUME_FRAC]
            added = t["n_schwann"] - sum(clusters[i]["n_schwann"] for i in covered)
            if len(covered) >= 2 or (len(covered) <= 1 and added >= LOOSE_ADD_MIN):
                kept_tracts.append(t)
                to_remove.update(covered)
        clusters = [c for i, c in enumerate(clusters) if i not in to_remove]
        clusters += kept_tracts
        print(f"  + {len(kept_tracts)} merge-tracts ({len(to_remove)} bodies merged); "
              f"{len(tracts) - len(kept_tracts)} duplicate tracts dropped", flush=True)


    if sox_avail:
        in_cl = np.zeros(n_schwann, dtype=bool)
        for c in clusters:
            in_cl |= c["comp"][s_iy, s_ix]
        loose_r = np.where(~in_cl)[0]
        foot_r0 = max(int(round(FOOT_RADIUS_UM / RESOLUTION)), 1)
        n_resc_core = 0
        if len(loose_r) >= RESCUE_MIN_CELLS:
            rp = cKDTree(sch_xy[loose_r]).query_pairs(RESCUE_LINK_UM, output_type="ndarray")
            rlab = connected_components(
                csr_matrix((np.ones(len(rp)), (rp[:, 0], rp[:, 1])), shape=(len(loose_r), len(loose_r)))
                if len(rp) else csr_matrix((len(loose_r), len(loose_r))), directed=False)[1]
            for rg in range(rlab.max() + 1 if len(rlab) else 0):
                gi = loose_r[rlab == rg]
                if len(gi) < RESCUE_MIN_CELLS or int(sox_pos[gi].sum()) < MIN_SOX10_CELLS:
                    continue
                gg = sch_xy[gi]
                nn = float(np.median(cKDTree(gg).query(gg, k=2)[0][:, 1]))
                if nn > RESCUE_MAX_NN_UM:
                    continue
                gy = np.clip(((gg[:, 1] - y_min) / RESOLUTION).astype(int), 0, ny - 1)
                gx = np.clip(((gg[:, 0] - x_min) / RESOLUTION).astype(int), 0, nx - 1)
                fr = np.zeros((ny, nx), dtype=bool); fr[gy, gx] = True
                comp_r = binary_dilation(fr, structure=disk(foot_r0))
                pr2 = regionprops(comp_r.astype(int))[0]
                major = float(pr2.axis_major_length) * RESOLUTION
                minor = max(float(pr2.axis_minor_length) * RESOLUTION, 1.0)
                area = int(comp_r.sum()) * RESOLUTION ** 2 / 1e6
                clusters.append(dict(
                    n_schwann=len(gi), area_mm2=area, areal_density=len(gi) / area if area else 0.0,
                    linear_density=len(gi) / (major / 1000.0) if major else 0.0, aspect_ratio=major / minor,
                    minor_axis_um=minor, major_axis_um=major, solidity=float(pr2.solidity),
                    eccentricity=float(pr2.eccentricity), core_frac=0.0, median_nn_um=nn,
                    n_sox10=int(sox_pos[gi].sum()), n_ngfr=int(ngfr_pos[gi].sum()) if ngfr_avail else -1,
                    passA=False, passB=False, passed=True, reject_reason="rescued-core",
                    shape_class="compact", confidence="low", comp=comp_r, cluster_id=0))
                n_resc_core += 1
        if n_resc_core:
            print(f"  + {n_resc_core} rescued tiny SOX10+ cores", flush=True)


    cluster_grid = np.full((ny, nx), -1, dtype=int)
    for i, c in enumerate(clusters, 1):
        c["cluster_id"] = i
        cluster_grid[c["comp"]] = i

    n_elong = sum(1 for c in clusters if c["shape_class"] == "elongated")
    n_compact = len(clusters) - n_elong
    n_resc = sum(1 for c in clusters if 1 <= c["n_sox10"] < MIN_SOX10_CELLS)
    print(f"  Kept {len(clusters)} (compact={n_compact}, elongated={n_elong}; "
          f"{n_resc} NGFR-rescued); scattered {len(scattered)}", flush=True)


    print("[4] Tumor proximity (signed EDT)...", flush=True)
    mask_all = cluster_grid > 0
    if n_tumor and mask_all.any():
        t_ix, t_iy = to_px(tumor_cells)
        inside_edt = distance_transform_edt(mask_all, sampling=RESOLUTION)
        outside_edt, (iy_idx, ix_idx) = distance_transform_edt(
            ~mask_all, sampling=RESOLUTION, return_indices=True)
        for c in clusters:
            c.update(n_tumor_inside=0, n_tumor_le27=0, n_tumor_le100=0,
                     n_tumor_le1000=0, n_tumor_le50=0)
        for k in range(len(t_ix)):
            yy, xx = t_iy[k], t_ix[k]
            if mask_all[yy, xx]:
                rid = cluster_grid[yy, xx]
                dist = 0.0
            else:
                rid = cluster_grid[iy_idx[yy, xx], ix_idx[yy, xx]]
                dist = float(outside_edt[yy, xx])
            if rid <= 0:
                continue
            c = clusters[rid - 1]
            if dist == 0.0:
                c["n_tumor_inside"] += 1
            if dist <= PROX_BANDS_UM[0]:
                c["n_tumor_le27"] += 1
            if dist <= PROX_BANDS_UM[1]:
                c["n_tumor_le100"] += 1
            if dist <= PROX_BANDS_UM[2]:
                c["n_tumor_le1000"] += 1
            if dist <= TUMOR_LEGACY_RADIUS:
                c["n_tumor_le50"] += 1
    else:
        for c in clusters:
            c.update(n_tumor_inside=0, n_tumor_le27=0, n_tumor_le100=0,
                     n_tumor_le1000=0, n_tumor_le50=0)
    for c in clusters:
        c["status"] = "Tumor+" if c["n_tumor_le50"] > 0 else "Tumor-"
    for c in scattered:
        for kk in ["n_tumor_inside", "n_tumor_le27", "n_tumor_le100", "n_tumor_le1000",
                   "n_tumor_le50"]:
            c[kk] = 0
        c["status"] = "Scattered"

    for c in clusters:
        print(f"    C{c['cluster_id']} [{c['shape_class']},{c['confidence']}] {c['status']}: "
              f"{c['n_schwann']} Schwann, AR={c['aspect_ratio']:.1f}, "
              f"rho={c['areal_density']:.0f}, lin={c['linear_density']:.0f}/mm, "
              f"sol={c['solidity']:.2f}, {c['area_mm2']:.3f}mm2, SOX10+={c['n_sox10']}, "
              f"NGFR+={c['n_ngfr']}"
              f"{'  <-- NGFR-rescued' if 1 <= c['n_sox10'] < MIN_SOX10_CELLS else ''}", flush=True)


    pd.DataFrame([{k: c.get(k) for k in STAT_COLS} for c in clusters + scattered]).to_csv(
        data_dir / f"{sample}_cluster_stats.csv", index=False)


    print("[5] Mapping cells...", flush=True)
    c_ix, c_iy = to_px(df)
    cid_arr = cluster_grid[c_iy, c_ix]
    status_map = {c["cluster_id"]: c["status"] for c in clusters}
    shape_map = {c["cluster_id"]: c["shape_class"] for c in clusters}
    df["schwann_cluster"] = [f"Schwann-{v}" if v > 0 else "None" for v in cid_arr]
    df["schwann_shape_class"] = [shape_map.get(v, "") for v in cid_arr]
    df["schwann_tumor_status"] = [status_map.get(v, "") for v in cid_arr]
    df[["cell_id", "schwann_cluster", "schwann_shape_class", "schwann_tumor_status",
        "cell_type"]].to_csv(data_dir / f"{sample}_cells.csv", index=False)


    foot_r = max(int(round(FOOT_RADIUS_UM / RESOLUTION)), 1)
    fdisk = disk(foot_r)
    final_s_lab = cluster_grid[s_iy, s_ix]
    all_um = schwann_cells[["x_centroid", "y_centroid"]].to_numpy()


    loose = np.where(final_s_lab <= 0)[0]
    clump_of = np.full(len(all_um), -1, dtype=int)
    if len(loose):
        lp = cKDTree(all_um[loose]).query_pairs(grab_um, output_type="ndarray")
        g = (csr_matrix((np.ones(len(lp)), (lp[:, 0], lp[:, 1])), shape=(len(loose), len(loose)))
             if len(lp) else csr_matrix((len(loose), len(loose))))
        clump_of[loose] = connected_components(g, directed=False)[1]


    clump_members, clump_thresh = {}, {}
    for cl in np.unique(clump_of[clump_of >= 0]):
        m = np.where(clump_of == cl)[0]
        clump_members[cl] = m
        nsox = int(sox_pos[m].sum()) if sox_avail else 0
        w = (len(m) - 1) + EXPR_WEIGHT * nsox
        clump_thresh[cl] = (grab_um - HALO_SINGLE_UM
                            + (HALO_SINGLE_UM + HALO_MULTI_UM) * min(w / HALO_WEIGHT_FULL, 1.0))
    for c in clusters:
        if c["shape_class"] == "elongated":
            continue
        mp = final_s_lab == c["cluster_id"]
        if not mp.any():
            continue
        dmin = cKDTree(all_um[mp]).query(all_um, k=1)[0]
        grabbed = [cl for cl, m in clump_members.items() if dmin[m].min() <= clump_thresh[cl]]
        sel = mp | np.isin(clump_of, grabbed)
        si = np.where(sel)[0]
        fr = np.zeros((ny, nx), dtype=bool)
        fr[s_iy[si], s_ix[si]] = True
        if len(si) >= 2:


            sp = cKDTree(all_um[si]).query_pairs(grab_um, output_type="ndarray")
            if len(sp):
                dd = np.linalg.norm(all_um[si][sp[:, 0]] - all_um[si][sp[:, 1]], axis=1)
                mst = minimum_spanning_tree(
                    csr_matrix((dd, (sp[:, 0], sp[:, 1])), shape=(len(si), len(si)))).tocoo()
                for a, b in zip(mst.row, mst.col):
                    rr, cc = skline(int(s_iy[si[a]]), int(s_ix[si[a]]),
                                    int(s_iy[si[b]]), int(s_ix[si[b]]))
                    fr[rr, cc] = True
        c["comp"] = binary_dilation(fr, structure=fdisk)


    print("[6] Figure...", flush=True)
    fig, ax = plt.subplots(figsize=(14, 14))
    ax.scatter(df.x_centroid, df.y_centroid, c="#e0e0e0", s=0.1, alpha=0.1, rasterized=True, zorder=0)
    if n_tumor:
        ax.scatter(tumor_cells.x_centroid, tumor_cells.y_centroid, c="#c0c0c0", s=0.2,
                   alpha=0.15, rasterized=True, zorder=1)
    if n_schwann:
        ax.scatter(schwann_cells.x_centroid, schwann_cells.y_centroid, c="#3498db", s=1.0,
                   alpha=0.8, rasterized=True, zorder=4)

    POS, NEG, SCAT, RESC = "#e74c3c", "#2ecc71", "#9aa0a6", "#8e44ad"

    def _smooth(ctr, sigma=2.0):


        if len(ctr) < 8:
            return ctr
        p = ctr[:-1] if np.allclose(ctr[0], ctr[-1]) else ctr
        s = np.column_stack([gaussian_filter1d(p[:, 0], sigma, mode="wrap"),
                             gaussian_filter1d(p[:, 1], sigma, mode="wrap")])
        return np.vstack([s, s[:1]])

    for c in scattered:
        for ctr in find_contours(c["comp"].astype(float), 0.5):
            ctr = _smooth(ctr)
            xs_c = (ctr[:, 1] + 0.5) * RESOLUTION + x_min
            ys_c = (ctr[:, 0] + 0.5) * RESOLUTION + y_min
            ax.fill(xs_c, ys_c, color=SCAT, alpha=0.10, lw=0, zorder=2.5)
            ax.plot(xs_c, ys_c, color=SCAT, lw=0.8, alpha=0.7, ls=(0, (1, 2)), zorder=5)
    for c in clusters:
        color = POS if c["status"] == "Tumor+" else NEG
        ls = (0, (1, 1.5)) if c["shape_class"] == "elongated" else "solid"
        lw = 1.0 if c["shape_class"] == "elongated" else 0.8
        rescued = 1 <= c["n_sox10"] < MIN_SOX10_CELLS
        for ctr in find_contours(c["comp"].astype(float), 0.5):
            ctr = _smooth(ctr)
            xs_c = (ctr[:, 1] + 0.5) * RESOLUTION + x_min
            ys_c = (ctr[:, 0] + 0.5) * RESOLUTION + y_min
            ax.fill(xs_c, ys_c, color=color, alpha=0.28, lw=0, zorder=2.5)
            if rescued:
                ax.plot(xs_c, ys_c, color=RESC, lw=lw + 0.8, alpha=0.9, ls="solid", zorder=5.5,
                        solid_capstyle="round")
            ax.plot(xs_c, ys_c, color=color, lw=lw, alpha=0.95, ls=ls, zorder=6)


    if clusters:
        comps_yx = [np.where(c["comp"]) for c in clusters]
        anchors = np.array([[(xs.mean() + 0.5) * RESOLUTION + x_min, (ys.mean() + 0.5) * RESOLUTION + y_min]
                            for ys, xs in comps_yx])
        rads = np.array([max(xs.ptp(), ys.ptp()) / 2 * RESOLUTION for ys, xs in comps_yx])
        tags = [f"{c['cluster_id']}{'E' if c['shape_class']=='elongated' else 'C'}" for c in clusters]
        cols = [POS if c["status"] == "Tumor+" else NEG for c in clusters]
        span = max(x_max - x_min, y_max - y_min)
        half = np.array([[max(len(t), 2) * span * 0.0028 + span * 0.002, span * 0.006] for t in tags])
        half_diag = np.hypot(half[:, 0], half[:, 1])
        pad = (x_max - x_min + y_max - y_min) * 0.0015
        img_c = np.array([(x_min + x_max) / 2, (y_min + y_max) / 2])
        d = anchors - img_c
        dn = np.hypot(d[:, 0], d[:, 1]); dn[dn < 1] = 1
        pref = anchors + (rads + half_diag + pad)[:, None] * d / dn[:, None]
        pos = _layout_labels(half, pref, anchors, rads, (x_min, x_max), (y_min, y_max))
        for (ax_um, ay_um), (px, py), tag, color in zip(anchors, pos, tags, cols):
            ax.annotate(tag, xy=(ax_um, ay_um), xytext=(px, py), fontsize=7, fontweight="bold",
                        ha="center", va="center", color=color, zorder=11,
                        path_effects=[pe.withStroke(linewidth=2.2, foreground="white")],
                        arrowprops=dict(arrowstyle="-", color=color, lw=0.5, alpha=0.5,
                                        shrinkA=0, shrinkB=1))

    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_max, y_min)
    ax.set_aspect("equal")
    ax.set_xlabel("X (um)", fontsize=12)
    ax.set_ylabel("Y (um)", fontsize=12)
    ax.xaxis.set_major_locator(MultipleLocator(1000))
    ax.yaxis.set_major_locator(MultipleLocator(1000))
    npos = sum(c["status"] == "Tumor+" for c in clusters)
    nneg = sum(c["status"] == "Tumor-" for c in clusters)
    legend = [
        Patch(fc="#3498db", alpha=0.8, label=f"{SCHWANN} cells ({n_schwann})"),
        Patch(fc=POS, alpha=0.5, label=f"Cluster + Tumor ({npos})"),
        Patch(fc=NEG, alpha=0.5, label=f"Cluster - Tumor ({nneg})"),
        Line2D([0], [0], color="#555", lw=1.6, ls="solid", label=f"Compact ({n_compact})"),
        Line2D([0], [0], color="#555", lw=1.0, ls=(0, (1, 1.5)), label=f"Elongated ({n_elong})"),
        Line2D([0], [0], color=SCAT, lw=1.0, ls=(0, (1, 2)), label=f"Scattered ({len(scattered)})"),
    ]
    n_resc_fig = sum(1 for c in clusters if 1 <= c["n_sox10"] < MIN_SOX10_CELLS)
    if n_resc_fig:
        legend.append(Line2D([0], [0], color=RESC, lw=2.0, ls="solid",
                             label=f"NGFR-rescued ({n_resc_fig})  [SOX10 1-2, NGFR+ >= {MIN_NGFR_CELLS}]"))
    ax.legend(handles=legend, loc="upper left", bbox_to_anchor=(1.01, 1), fontsize=9,
              framealpha=0.9, borderaxespad=0)
    ax.set_title(f"{sample} ({cancer}/{panel}): Schwann clusters "
                 f"[{n_compact} compact + {n_elong} elongated, {len(scattered)} scattered]",
                 fontsize=13, fontweight="bold")
    fig.savefig(fig_dir / f"{sample}.png", dpi=250, bbox_inches="tight")
    fig.savefig(fig_dir / f"{sample}.pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved -> figures/{sample}.png + .pdf, data/{sample}_*.csv\nDone. {time.time() - t0:.1f}s",
          flush=True)


if __name__ == "__main__":
    main()
