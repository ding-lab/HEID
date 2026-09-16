
from __future__ import annotations
import paths as P
import os
PROJECTS_ROOT = os.environ.get("PROJECTS_ROOT", "/data/heid")

import argparse
import json
import os
import time
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
from scipy.ndimage import (
    binary_closing,
    binary_dilation,
    binary_fill_holes,
    binary_opening,
    distance_transform_edt,
    gaussian_filter,
    label,
)
from skimage.measure import find_contours, regionprops
from skimage.morphology import disk
from skimage.segmentation import watershed
from skimage.feature import peak_local_max

matplotlib.use("Agg")
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.ticker import MultipleLocator

import h5py
from scipy.sparse import csc_matrix
from scipy.spatial import cKDTree


DATA_ROOT = P.WORKSPACE / "raw_xenium"
CELL_ATLAS = P.WORKSPACE / "cell_atlas"
REPO_ROOT = P.PROJECTS
OUT_ROOT_BASE = P.ACCEPTED
PROGRAM_SCORE_DIR = P.PROGRAMS
CELL_TYPE_CACHE = P.CELL_TYPES / "5k"
TUMOR_ZONE_ROOT = P.ZONE_ROOT


MAT_THRESH_B  = -0.12854
MAT_THRESH_DC = -0.10229
MAT_DC_TYPES  = ["pDC", "cDC", "cDC1", "cDC2", "mregDC", "DC"]
MAT_STATES    = ["Mature TLS", "FDC-deficient TLS", "Premature TLS", "Early TLS"]
MAT_LN_LABEL  = "Canonical LN"


MAT_COL = {
    "Mature TLS":        "#d62728",
    "FDC-deficient TLS": "#ffd92f",
    "Premature TLS":     "#9467bd",
    "Early TLS":         "#ff7f0e",
    "Canonical LN":      "#2ca02c",
}
MAT_STATE_REFERENCE_CSV = P.STATE_REFERENCE


def classify_tls_state(b, dc):
    if b >= MAT_THRESH_B and dc >= MAT_THRESH_DC:  return "Mature TLS"
    if b >= MAT_THRESH_B and dc <  MAT_THRESH_DC:  return "FDC-deficient TLS"
    if b <  MAT_THRESH_B and dc >= MAT_THRESH_DC:  return "Premature TLS"
    return "Early TLS"


def _load_reference_labels():
    try:
        y = pd.read_csv(MAT_STATE_REFERENCE_CSV, index_col=0)
        return y["TLS_State"].to_dict()
    except Exception as e:
        print(f"  [maturation] state reference unavailable ({e}); using the molecular rule only", flush=True)
        return {}


H5AD_DIR = str(P.CELLS)
NORM_MODE = bool(H5AD_DIR)


GC_DIR = str(P.GC)


def _load_gc_tau():
    p = os.environ.get("TLS_GC_THRESH", str(Path(__file__).resolve().parents[1] / "configs/gc_thresholds.json"))
    if p and Path(p).exists():
        with open(p) as fh:
            d = json.load(fh)
        return d.get("TAU"), d.get("TAU_LOW")
    return None, None
TAU_GC, TAU_GC_LOW = _load_gc_tau()
_PS_CACHE: dict = {}

def _persample(sample: str) -> pd.DataFrame:
    if sample not in _PS_CACHE:
        p = Path(H5AD_DIR) / f"{sample}.parquet"
        if not p.exists():
            raise FileNotFoundError(f"TLS_H5AD_DIR set but missing {p}")
        d = pd.read_parquet(p)
        d["cell_id"] = d["cell_id"].astype(str)
        d["cell_type"] = d["cell_type"].astype(str)
        if GC_DIR:
            gp = Path(GC_DIR) / f"{sample}.parquet"
            if gp.exists():
                g = pd.read_parquet(gp)[["cell_id", "gc_score"]]
                g["cell_id"] = g["cell_id"].astype(str)
                d = d.merge(g, on="cell_id", how="left")
            else:
                d["gc_score"] = np.nan
        _PS_CACHE[sample] = d
    return _PS_CACHE[sample]


B_TYPES = ["B_cell"]


T_TYPES = ["T_cell", "NK_T", "CD4_T", "CD8_T", "T_reg"]
P_TYPES = ["Plasma"]
LE_TYPES = ["Lymphatic_Endothelial"]


INTRA_ZONES = {"In_0-50um", "In_50-100um", "In_100-150um", "In_>150um"}
PERI_ZONES = {"Out_0-50um", "Out_50-100um", "Out_100-150um"}
DISTANT_ZONES = {"Outside"}


INTRA_FRAC_MIN = 0.50
PERI_FRAC_MIN = 0.50
DISTANT_FRAC_MIN = 0.80


COLORS = {
    "B_cell": "#8e44ad",
    "T_cell": "#3498db",
    "Plasma": "#e67e22",
    "Lymphatic_Endothelial": "#27ae60",
    "Other": "#d5d5d5",
    "BCL6_high": "#f06292",
    "TLS_mature_fill":   "#f1c40f",
    "TLS_mature_line":   "#b7950b",
    "TLS_immature_fill": "#ec7063",
    "TLS_immature_line": "#922b21",

    "intra_tumor":   "#c0392b",
    "peri_tumor":    "#e67e22",
    "tumor_distant": "#7f8c8d",
    "mixed":         "#8e44ad",
}


CT_BAR_COL = {
    "B_cell": "#8e24aa", "Plasma": "#ef6c00", "NK_T": "#42a5f5", "CD4_T": "#42a5f5",
    "CD8_T": "#1565c0", "T_reg": "#26a69a", "cDC": "#7e57c2", "cDC1": "#5e35b1",
    "pDC": "#9575cd", "mregDC": "#ba68c8", "Macrophage": "#8d6e63", "Mast": "#ffb300",
    "Tumor": "#bdbdbd", "Fibroblast": "#cfd8dc", "Endothelial": "#90a4ae",
    "Lymphatic_Endothelial": "#80deea", "Stromal": "#d7ccc8", "Langerhans_cell": "#aed581",
}


RESOLUTION = 10
SIGMA_PX = 4
DENSITY_THRESH_REL = 0.08
CLOSING_R_PX = 10
OPENING_R_PX = 3
MAX_HOLE_AREA_PX = 50000

MIN_B_CELLS = 50


MIN_B_DENSITY_CELLS_MM2 = 8000


MIN_B_CORE_AREA_UM2 = 10000
TLS_PEAK_MIN_SEP_UM = float(os.environ.get("TLS_PEAK_MIN_SEP_UM", "250"))
HALO_RADIUS_UM = 150
MIN_T_IN_REGION = 60
MIN_TLS_AREA_UM2 = 80000
MIN_LYMPHOID_DENSITY_CELLS_MM2 = float(os.environ.get("TLS_MIN_LYMPHOID_DENSITY_CELLS_MM2", "3000"))


MIN_B_DENSITY_REGION_MM2 = 920
MIN_REGION_CIRCULARITY = 0.65
MIN_REGION_AXIS_RATIO = 0.50


CORE_SIGMA_PX   = float(os.environ.get("TLS_CORE_SIGMA_PX", "1.5"))
CORE_THRESH_REL = float(os.environ.get("TLS_CORE_THRESH_REL", "0.30"))
CORE_CLOSING_PX = int(os.environ.get("TLS_CORE_CLOSING_PX", "3"))
CORE_OPENING_PX = int(os.environ.get("TLS_CORE_OPENING_PX", "2"))


CORE_NECK_RADIUS_PX  = float(os.environ.get("TLS_CORE_NECK_RADIUS_PX", "3"))
CORE_HOLE_FILL_MAX_PX = int(os.environ.get("TLS_CORE_HOLE_FILL_MAX_PX", "30"))
CORE_HOLE_CONNECT_FRAC = float(os.environ.get("TLS_CORE_HOLE_CONNECT_FRAC", "0.5"))


SPLIT_MAX_CORE_AREA_UM2 = float(os.environ.get("TLS_SPLIT_MAX_CORE_AREA_UM2", "100000"))
SPLIT_PEAK_SEP_UM       = float(os.environ.get("TLS_SPLIT_PEAK_SEP_UM", "150"))
SPLIT_MERGE_FRAC        = float(os.environ.get("TLS_SPLIT_MERGE_FRAC", "0.8"))
SPLIT_CORE_THRESH       = float(os.environ.get("TLS_SPLIT_CORE_THRESH", "0.45"))


CORE_SIGMA_SMALL_PX   = float(os.environ.get("TLS_CORE_SIGMA_SMALL_PX", "1.5"))
CORE_SIGMA_BIG_PX     = float(os.environ.get("TLS_CORE_SIGMA_BIG_PX", "4.0"))
BIG_COMP_AREA_UM2     = float(os.environ.get("TLS_BIG_COMP_AREA_UM2", "100000"))
MIN_B_DENSITY_BIG_MM2 = float(os.environ.get("TLS_MIN_B_DENSITY_BIG_MM2", "4000"))
_SWEEP_TAG = os.environ.get("TLS_SWEEP_TAG", "")

BCL6_GENE = "BCL6"
BCL6_MIN_PCT_FOR_HIGH = 90


FDC_GENES = ["CR2", "CR1", "FCER2"]
MATURITY_GENES = ["BCL6", "AICDA", "MKI67", "CXCL13", "CD3E", "SELP"] + FDC_GENES


GC_PCTL_GENES = ["BCL6", "AICDA", "MKI67"]
GC_PCTL = 90


W_GC, W_TFH, W_PLASMA, W_STRUCT = 0.45, 0.20, 0.15, 0.20


SAT_AICDA = 0.024
SAT_BCL6  = 0.05
SAT_MKI67 = 0.03
SAT_TFH   = 0.10
SAT_PLASMA_FRAC = 0.20
SAT_COMPACT = 0.5

RUNG_AICDA_FLOOR = 0.004
RUNG_AICDA_ACTIVE = 0.013
RUNG_BCL6_COPRIMARY = 0.07
RUNG_TFH_COPRIMARY = 0.05


AICDA_FLOOR_N = 4
MKI67_CORROB_CUT = 0.013


COMP_PLASMA_DOM = 0.20
COMP_B_DOM      = 0.08

CELL_POINT_SIZE = 1.0
BCL6_BIN_UM = 100
BCL6_BASE_SIZE = 2
BCL6_SCALE = 8
BCL6_POWER = 0.85
CARD_BIN_UM = 30

TISSUE_SIGMA = 10
TISSUE_THRESH = 0.015


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("dataset", choices=["5k"])
    p.add_argument("cancer_type")
    p.add_argument("sample")
    return p.parse_args()


def rasterize(sub: pd.DataFrame, ny: int, nx: int, x_min: float, y_min: float) -> np.ndarray:
    grid = np.zeros((ny, nx), dtype=np.float32)
    if len(sub) == 0:
        return grid
    ix = np.clip(((sub.x_centroid.values - x_min) / RESOLUTION).astype(int), 0, nx - 1)
    iy = np.clip(((sub.y_centroid.values - y_min) / RESOLUTION).astype(int), 0, ny - 1)
    np.add.at(grid, (iy, ix), 1)
    return grid


def fill_small_holes(mask: np.ndarray, max_hole_area: int) -> np.ndarray:
    filled = binary_fill_holes(mask)
    holes = filled & ~mask
    if not holes.any():
        return filled
    lb_h, n_h = label(holes)
    for idx in range(1, n_h + 1):
        if (lb_h == idx).sum() > max_hole_area:
            filled[lb_h == idx] = False
    return filled


def b_density_mask(raw: np.ndarray) -> np.ndarray:
    d = gaussian_filter(raw, sigma=SIGMA_PX)
    if d.max() <= 0:
        return np.zeros_like(raw, dtype=bool)
    m = d > DENSITY_THRESH_REL * d.max()
    m = binary_closing(m, structure=disk(CLOSING_R_PX))
    m = fill_small_holes(m, MAX_HOLE_AREA_PX)
    m = binary_opening(m, structure=disk(OPENING_R_PX))
    return m


MERGE_SADDLE_FRAC = float(os.environ.get("TLS_MERGE_SADDLE_FRAC", "0.5"))

def merge_shallow_basins(ws: np.ndarray, dens: np.ndarray, frac: float) -> np.ndarray:
    nlab = int(ws.max())
    if nlab <= 1 or frac <= 0:
        return ws
    peak_d = np.zeros(nlab + 1)
    for j in range(1, nlab + 1):
        m = ws == j
        if m.any():
            peak_d[j] = float(dens[m].max())

    saddle: dict = {}
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
            parent[x] = parent[parent[x]]; x = parent[x]
        return x
    for (u, v), col in saddle.items():
        if col >= frac * min(peak_d[u], peak_d[v]):
            parent[find(u)] = find(v)
    out = np.zeros_like(ws)
    remap: dict = {}; nxt = 1
    for j in range(1, nlab + 1):
        r = find(j)
        if r not in remap:
            remap[r] = nxt; nxt += 1
        out[ws == j] = remap[r]
    return out


def clean_core(core: np.ndarray, dens_c: np.ndarray, thr: float) -> list:
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


def split_oversized_core(core: np.ndarray, b_density: np.ndarray,
                         b_iy: np.ndarray, b_ix: np.ndarray) -> list:
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
        tight = dens_c > thr
        tight &= sb
        if CORE_CLOSING_PX > 0:
            tight = binary_closing(tight, structure=disk(CORE_CLOSING_PX))
        if CORE_OPENING_PX > 0:
            tight = binary_opening(tight, structure=disk(CORE_OPENING_PX))
        tight &= sb
        lb_p, n_p = label(tight)
        for pid in range(1, n_p + 1):
            out.extend(clean_core(lb_p == pid, dens_c, thr))
    return out if out else [core]


def region_shape_stats(region: np.ndarray, core: np.ndarray):
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


def load_coords(dataset: str, cancer_type: str, sample: str) -> pd.DataFrame:
    if NORM_MODE:
        return _persample(sample)[["cell_id", "x_centroid", "y_centroid"]].copy()
    xendir = DATA_ROOT / dataset / cancer_type / sample
    for path in [xendir / "cells.parquet", xendir / "cells.csv.gz", xendir / "cells.csv"]:
        if path.exists():
            if path.suffix == ".parquet":
                c = pd.read_parquet(path, columns=["cell_id", "x_centroid", "y_centroid"])
            else:
                c = pd.read_csv(path, usecols=["cell_id", "x_centroid", "y_centroid"])
            c["cell_id"] = c["cell_id"].astype(str)
            return c
    raise FileNotFoundError(f"No cells file in {xendir}")


def _from_big_csv(csv_path: Path, sample: str) -> pd.DataFrame | None:
    if not csv_path.exists():
        return None
    df = pd.read_csv(csv_path, dtype=str, keep_default_na=False)
    sub = df[df["dataset"] == sample].copy()
    if sub.empty:
        return None
    sub["cell_id"] = sub["barcode"].str[: -len(f"_{sample}")]
    return sub[["cell_id", "cell_type"]].copy()


def load_genes(dataset: str, cancer_type: str, sample: str,
               genes: list) -> tuple:
    if NORM_MODE:
        d = _persample(sample)
        cols = [g for g in genes if g in d.columns]
        extra = ["gc_score"] if "gc_score" in d.columns else []
        out = d[["cell_id"] + cols + extra].set_index("cell_id")
        return out, set(cols)
    h5 = DATA_ROOT / dataset / cancer_type / sample / "cell_feature_matrix.h5"
    if not h5.exists():
        return pd.DataFrame(columns=["cell_id"]).set_index("cell_id"), set()
    with h5py.File(h5, "r") as f:
        names = [n.decode() for n in f["matrix/features/name"][:]]
        barcodes = [b.decode() for b in f["matrix/barcodes"][:]]
        data = f["matrix/data"][:]
        indices = f["matrix/indices"][:]
        indptr = f["matrix/indptr"][:]
        shape = tuple(f["matrix/shape"][:])
    name_idx = {n: i for i, n in enumerate(names)}
    mat = csc_matrix((data, indices, indptr), shape=shape)
    on_panel = set()
    out = {}
    for g in genes:
        if g in name_idx:
            on_panel.add(g)
            out[g] = np.asarray(mat[name_idx[g], :].todense()).flatten().astype(np.int32)
    df = pd.DataFrame(out, index=barcodes)
    df.index.name = "cell_id"
    return df, on_panel


def pctl_positive(counts: np.ndarray, b_mask: np.ndarray, pctl: int = GC_PCTL) -> np.ndarray:
    expr = counts[b_mask & (counts > 0)]
    if expr.size == 0:
        return np.zeros_like(counts, dtype=bool)
    if NORM_MODE:
        thr = float(np.percentile(expr, pctl))
        return b_mask & (counts >= thr) & (counts > 0)
    thr = max(1, int(np.ceil(np.percentile(expr, pctl))))
    return b_mask & (counts >= thr)


def _median_nn(xy: np.ndarray) -> float:
    if len(xy) < 2:
        return float("nan")
    d, _ = cKDTree(xy).query(xy, k=2)
    return float(np.median(d[:, 1]))


def aicda_clust_ratio(aicda_xy: np.ndarray, region_b_xy: np.ndarray,
                      rng: np.random.Generator, n_rep: int = 20,
                      min_pts: int = 5) -> float | None:
    k = len(aicda_xy)
    if k < min_pts or len(region_b_xy) < k:
        return None
    nn_aicda = _median_nn(aicda_xy)
    if not np.isfinite(nn_aicda) or nn_aicda <= 0:
        return None
    samp = [_median_nn(region_b_xy[rng.choice(len(region_b_xy), k, replace=False)])
            for _ in range(n_rep)]
    nn_rand = float(np.nanmedian(samp))
    if not np.isfinite(nn_rand) or nn_rand <= 0:
        return None
    return float(nn_aicda / nn_rand)


def bivariate_morans_i(grid_a: np.ndarray, grid_b: np.ndarray, mask: np.ndarray) -> float:
    if mask.sum() < 4:
        return float("nan")
    a = grid_a[mask].astype(np.float64)
    b = grid_b[mask].astype(np.float64)
    if a.std() == 0 or b.std() == 0:
        return float("nan")
    pad_a = np.zeros_like(grid_a, dtype=np.float64)
    pad_b = np.zeros_like(grid_b, dtype=np.float64)
    pad_a[mask] = (grid_a[mask] - a.mean()) / a.std()
    pad_b[mask] = (grid_b[mask] - b.mean()) / b.std()
    nb_b = np.zeros_like(pad_b)
    nb_n = np.zeros_like(pad_b)
    mf = mask.astype(np.float64)
    for dy in (-1, 0, 1):
        for dx in (-1, 0, 1):
            if dy == 0 and dx == 0:
                continue
            nb_b += np.roll(np.roll(pad_b, dy, axis=0), dx, axis=1) * np.roll(np.roll(mf, dy, axis=0), dx, axis=1)
            nb_n += np.roll(np.roll(mf, dy, axis=0), dx, axis=1)
    valid = mask & (nb_n > 0)
    if not valid.any():
        return float("nan")
    return float(np.mean(pad_a[valid] * nb_b[valid] / nb_n[valid]))


def plasma_shell_index(region: np.ndarray, p_iy: np.ndarray, p_ix: np.ndarray,
                       ny: int, nx: int) -> float:
    if len(p_ix) == 0 or region.sum() == 0:
        return float("nan")
    shell_r = max(1, int(round(50 / RESOLUTION)))
    rim = binary_dilation(region, structure=disk(shell_r)) & ~region
    if rim.sum() == 0:
        return float("nan")
    n_in = int(region[p_iy, p_ix].sum())
    n_out = int(rim[p_iy, p_ix].sum())
    a_in = float(region.sum()); a_out = float(rim.sum())
    if n_in == 0:
        return float("inf") if n_out > 0 else float("nan")
    return float((n_out / a_out) / (n_in / a_in))


def _isbad(x) -> bool:
    return x is None or (isinstance(x, float) and not np.isfinite(x))


def _clip01(x: float) -> float:
    return float(min(max(x, 0.0), 1.0))


def _mean_avail(terms: list):
    vals = [t for t in terms if not _isbad(t)]
    return float(sum(vals) / len(vals)) if vals else None


def compute_maturity(f: dict, norms: dict) -> dict:


    _gc_terms = [(1.0, _clip01(f["gc_aid_frac"] / SAT_AICDA)),
                 (1.0, _clip01(f["bcl6_b_frac"] / SAT_BCL6)),
                 (0.5, _clip01(f["mki67_b_frac"] / SAT_MKI67))]
    _gc_num = sum(w * t for w, t in _gc_terms if not _isbad(t))
    _gc_den = sum(w for w, t in _gc_terms if not _isbad(t))
    gc_axis = float(_gc_num / _gc_den) if _gc_den > 0 else None

    tfh_terms = [_clip01(f["tfh_frac"] / SAT_TFH)]
    if norms.get("lamp3_avail") and norms.get("lamp3_std", 0) > 0 and not _isbad(f.get("lamp3_density")):
        z = (f["lamp3_density"] - norms["lamp3_mean"]) / norms["lamp3_std"]
        tfh_terms.append(_clip01((z + 2) / 4))
    if norms.get("hev_p90", 0) > 0 and not _isbad(f.get("hev_density")):
        tfh_terms.append(_clip01(f["hev_density"] / norms["hev_p90"]))
    tfh_axis = _mean_avail(tfh_terms)

    pl_terms = [_clip01(f["plasma_frac"] / SAT_PLASMA_FRAC)]
    if not _isbad(f.get("plasma_shell_index")):
        pl_terms.append(_clip01((f["plasma_shell_index"] - 0.5) / 1.5))
    plasma_axis = _mean_avail(pl_terms)

    st_terms = []
    if not _isbad(f.get("moran_i")):
        st_terms.append(_clip01(0.5 - f["moran_i"]))
    st_terms.append(_clip01(f["region_circularity"]))
    if norms.get("area_p90", 0) > 0:
        st_terms.append(_clip01(f["area_mm2"] / norms["area_p90"]))


    struct_axis = _mean_avail(st_terms)

    axes = [(W_GC, gc_axis), (W_TFH, tfh_axis), (W_PLASMA, plasma_axis), (W_STRUCT, struct_axis)]
    num = sum(w * a for w, a in axes if not _isbad(a))
    den = sum(w for w, a in axes if not _isbad(a))
    score = float(num / den) if den > 0 else None


    aid, gcb, tfhf = f["gc_aid_frac"], f["gc_b_frac"], f["tfh_frac"]
    n_aicda_in = int(f.get("n_aicda_b_in", 0) or 0)
    S = f.get("gc_score_mean")
    n_rb = int(f.get("n_region_b", 0) or 0)
    have_score = (S is not None) and (TAU_GC is not None)
    if have_score:

        path1 = (S >= TAU_GC) and (n_aicda_in >= AICDA_FLOOR_N) and (n_rb >= MIN_B_CELLS)
    else:

        path1 = (aid >= RUNG_AICDA_ACTIVE and n_aicda_in >= AICDA_FLOOR_N)

    path2 = (gcb >= RUNG_BCL6_COPRIMARY and tfhf >= RUNG_TFH_COPRIMARY and aid >= RUNG_AICDA_FLOOR)
    if path1 or path2:
        rung = 2
    elif (have_score and TAU_GC_LOW is not None and S >= TAU_GC_LOW) or (aid >= RUNG_AICDA_FLOOR):
        rung = 1
    else:
        rung = 0
    rung_label = ["gc_quiet", "low_GC", "active_GC"][rung]
    mature = (rung == 2)
    gc_present = "mature" if mature else "immature"
    gc_call_path = ("score" if (have_score and path1) else
                    ("frac" if (path1 and not have_score) else
                     ("path2" if path2 else "none")))

    mki67_b = float(f.get("mki67_b_frac", 0.0) or 0.0)
    bcl6_b = float(f.get("bcl6_b_frac", 0.0) or 0.0)
    mki67_corroborated = bool(mki67_b >= MKI67_CORROB_CUT)


    soft_witness = (mki67_b >= MKI67_CORROB_CUT) or (bcl6_b >= 0.02) or (tfhf >= 0.05)
    confidence = "LOW" if (mature and path1 and 4 <= n_aicda_in <= 6 and not soft_witness) else "HIGH"


    pf = f["plasma_frac"]
    if pf >= COMP_PLASMA_DOM:
        tls_composition = "plasma_dominant"
    elif pf < COMP_B_DOM:
        tls_composition = "b_dominant"
    else:
        tls_composition = "mixed"
    return {
        "tls_development_5k": score,
        "gc_axis": gc_axis, "tfh_axis": tfh_axis,
        "plasma_axis": plasma_axis, "struct_axis": struct_axis,
        "gc_rung": rung, "gc_rung_label": rung_label,
        "gc_present": gc_present, "mature": bool(mature),
        "tls_composition": tls_composition,
        "mki67_corroborated": mki67_corroborated, "confidence": confidence,


        "gc_call_path": gc_call_path,
    }


def load_cell_types(dataset: str, sample: str) -> pd.DataFrame:
    if NORM_MODE:
        return _persample(sample)[["cell_id", "cell_type"]].copy()


    override = os.environ.get("TLS_CT_CACHE_DIR", "")
    if override:
        ov = Path(override) / f"{sample}.csv"
        if ov.exists():
            return pd.read_csv(ov, usecols=["cell_id", "cell_type"], dtype=str,
                               keep_default_na=False)
        raise FileNotFoundError(f"TLS_CT_CACHE_DIR set but no {ov}")


    cache = P.CELL_TYPES / dataset / f"{sample}.csv"
    if cache.exists():
        return pd.read_csv(cache, usecols=["cell_id", "cell_type"], dtype=str,
                           keep_default_na=False)
    primary = (CELL_ATLAS / "5k" / "allsolidtumor5k.cells.csv" if dataset == "5k"
               else CELL_ATLAS / "477" / "allsolidtumor.cells.csv")
    r = _from_big_csv(primary, sample)
    if r is not None:
        return r
    other = (CELL_ATLAS / "477" / "allsolidtumor.cells.csv" if dataset == "5k"
             else CELL_ATLAS / "5k" / "allsolidtumor5k.cells.csv")
    r = _from_big_csv(other, sample)
    if r is not None:
        return r
    raise FileNotFoundError(f"No cell type data for {sample}")


def load_boundary_zones(dataset: str, cancer_type: str, sample: str) -> pd.DataFrame | None:
    p = TUMOR_ZONE_ROOT / dataset / "outputs" / cancer_type / sample / f"{sample}_boundary_zones.csv"
    if not p.exists():
        return None
    df = pd.read_csv(p, usecols=["cell_id", "tumor_region", "boundary_zone"], dtype=str,
                     keep_default_na=False)
    return df


def classify_tls_tumor(zone_counts: dict, n_cells_in_region: int) -> tuple[str, dict]:
    if n_cells_in_region == 0:
        return "mixed", {"intra": 0.0, "peri": 0.0, "distant": 0.0, "unmapped": 1.0}
    n_intra = sum(zone_counts.get(z, 0) for z in INTRA_ZONES)
    n_peri = sum(zone_counts.get(z, 0) for z in PERI_ZONES)
    n_distant = sum(zone_counts.get(z, 0) for z in DISTANT_ZONES)
    n_mapped = n_intra + n_peri + n_distant
    n_unmapped = n_cells_in_region - n_mapped

    f_intra = n_intra / n_cells_in_region
    f_peri = n_peri / n_cells_in_region
    f_distant = n_distant / n_cells_in_region
    f_unmapped = n_unmapped / n_cells_in_region

    fracs = {"intra": float(f_intra), "peri": float(f_peri),
             "distant": float(f_distant), "unmapped": float(f_unmapped)}

    if f_intra >= INTRA_FRAC_MIN:
        return "intra_tumor", fracs
    if f_peri >= PERI_FRAC_MIN:
        return "peri_tumor", fracs
    if f_distant >= DISTANT_FRAC_MIN:
        return "tumor_distant", fracs

    if f_intra + f_peri >= 0.70 and f_intra < INTRA_FRAC_MIN and f_peri < PERI_FRAC_MIN:
        return "peri_tumor", fracs
    return "mixed", fracs


def _label_fs(txt: str) -> float:
    return {1: 8.5, 2: 8.5, 3: 6.3}.get(len(txt), 5.3)


def main() -> None:
    args = parse_args()
    dataset, cancer_type, sample = args.dataset, args.cancer_type, args.sample
    t0 = time.time()

    out_base = OUT_ROOT_BASE / "sweep" / _SWEEP_TAG if _SWEEP_TAG else OUT_ROOT_BASE
    out_dir = out_base / "outputs" / dataset / cancer_type / sample


    print(f"[1] Loading {dataset}/{cancer_type}/{sample}...", flush=True)
    coords = load_coords(dataset, cancer_type, sample)
    types = load_cell_types(dataset, sample)
    df = coords.merge(types, on="cell_id", how="inner")
    print(f"  {len(df)} cells", flush=True)
    if df.empty:
        raise RuntimeError(f"No cells after merge for {sample}")

    df["class"] = "Other"
    df.loc[df["cell_type"].isin(B_TYPES), "class"] = "B_cell"
    df.loc[df["cell_type"].isin(T_TYPES), "class"] = "T_cell"
    df.loc[df["cell_type"].isin(P_TYPES), "class"] = "Plasma"
    df.loc[df["cell_type"].isin(LE_TYPES), "class"] = "Lymphatic_Endothelial"


    print(f"[1b] Loading tumor boundary zones for tumor classification...", flush=True)
    bz_df = load_boundary_zones(dataset, cancer_type, sample)
    if bz_df is None:
        print(f"  WARN: tumor-zone CSV not found at {TUMOR_ZONE_ROOT}/{dataset}/outputs/"
              f"{cancer_type}/{sample}/. All TLS will be 'unmapped'.", flush=True)
        df["boundary_zone"] = ""
        df["gen7_tumor_region"] = ""
        zones_available = False
    else:
        df = df.merge(bz_df.rename(columns={"tumor_region": "gen7_tumor_region"}),
                      on="cell_id", how="left")
        df["boundary_zone"] = df["boundary_zone"].fillna("")
        df["gen7_tumor_region"] = df["gen7_tumor_region"].fillna("")
        n_mapped = int((df["boundary_zone"] != "").sum())
        print(f"  Cells with boundary_zone: {n_mapped}/{len(df)} "
              f"({100 * n_mapped / len(df):.1f}%)", flush=True)
        zones_available = True

        zone_summary = df["boundary_zone"].value_counts()
        for z, n in zone_summary.items():
            if z:
                print(f"    {z}: {n}", flush=True)


    gene_df, genes_on_panel = load_genes(dataset, cancer_type, sample, MATURITY_GENES)
    if not gene_df.empty:
        df = df.merge(gene_df.reset_index(), on="cell_id", how="left")
    for g in MATURITY_GENES:
        if g in df.columns:

            df[g] = df[g].fillna(0).astype(np.float32 if NORM_MODE else np.int32)
        else:
            df[g] = np.float32(0) if NORM_MODE else np.int32(0)
    print(f"  Maturity genes on panel: {sorted(genes_on_panel)} "
          f"(missing: {sorted(set(MATURITY_GENES) - genes_on_panel)})", flush=True)

    b_mask_df = df["class"] == "B_cell"
    b_mask = b_mask_df.values

    df["AICDA_pos"] = pctl_positive(df["AICDA"].values, b_mask)
    df["BCL6_pos"] = pctl_positive(df["BCL6"].values, b_mask)
    df["MKI67_pos"] = pctl_positive(df["MKI67"].values, b_mask)
    df["gcb_pos"] = df["BCL6_pos"].values | df["AICDA_pos"].values


    t_mask = (df["class"] == "T_cell").values
    df["tfh_pos"] = t_mask & (df["CD3E"].values > 0) & (df["CXCL13"].values > 0)

    df["hev_pos"] = (df["cell_type"] == "Endothelial").values & (df["SELP"].values > 0)
    df["mregdc"] = (df["cell_type"] == "mregDC").values

    df["bcl6_count"] = df["BCL6"]
    df["bcl6_high"] = df["BCL6_pos"]
    n_bcl6_high = int(df["bcl6_high"].sum())
    lamp3_avail = bool(df["mregdc"].any())


    cand_mask = (~df["class"].isin(["B_cell", "Plasma"])).values
    fdc_pos = np.zeros(len(df), dtype=bool)
    for g in FDC_GENES:
        if g in genes_on_panel:
            fdc_pos |= pctl_positive(df[g].values, cand_mask)
    df["fdc_pos"] = fdc_pos

    n_b = int(b_mask_df.sum())
    n_t = int((df["class"] == "T_cell").sum())
    n_p = int((df["class"] == "Plasma").sum())
    n_le = int((df["class"] == "Lymphatic_Endothelial").sum())


    n_t_all, n_p_all = n_t, n_p
    n_tumor = int(df["cell_type"].astype(str).str.startswith("Tumor").sum())
    print(f"    B_cell: {n_b}, T(+NK_T): {n_t}, Plasma: {n_p}, LE: {n_le}, "
          f"Tumor: {n_tumor}, BCL6+ B (de-noised): {n_bcl6_high}, "
          f"AICDA+ B: {int(df['AICDA_pos'].sum())}, Tfh: {int(df['tfh_pos'].sum())}, "
          f"mregDC_avail: {lamp3_avail}", flush=True)

    if n_b < MIN_B_CELLS:
        print(f"  Too few B cells ({n_b} < {MIN_B_CELLS}), no TLS possible. Skipping output.",
              flush=True)
        return

    x_min, x_max = df.x_centroid.min() - 200, df.x_centroid.max() + 200
    y_min, y_max = df.y_centroid.min() - 200, df.y_centroid.max() + 200
    nx = int(np.ceil((x_max - x_min) / RESOLUTION))
    ny = int(np.ceil((y_max - y_min) / RESOLUTION))

    def to_px(sub):
        ix = np.clip(((sub.x_centroid.values - x_min) / RESOLUTION).astype(int), 0, nx - 1)
        iy = np.clip(((sub.y_centroid.values - y_min) / RESOLUTION).astype(int), 0, ny - 1)
        return ix, iy


    all_raw = rasterize(df, ny, nx, x_min, y_min)
    tissue = gaussian_filter(all_raw, sigma=TISSUE_SIGMA) > TISSUE_THRESH
    tissue = binary_closing(tissue, structure=disk(10))
    tissue = binary_fill_holes(tissue)


    print("[2] B-cell mask + components...", flush=True)
    b_sub = df[df["class"] == "B_cell"]
    b_raw = rasterize(b_sub, ny, nx, x_min, y_min)
    b_density = gaussian_filter(b_raw, sigma=SIGMA_PX)
    b_mask = b_density_mask(b_raw) & tissue
    lb_b, n_b_cc = label(b_mask)
    print(f"  Raw B components: {n_b_cc}", flush=True)

    if n_b_cc > 0:
        peak_min_sep_px = max(int(TLS_PEAK_MIN_SEP_UM / RESOLUTION), 3)
        new_lb = np.zeros_like(lb_b, dtype=np.int32)
        next_lid = 1
        n_split = 0
        n_merged = 0
        for cid in range(1, n_b_cc + 1):
            comp = lb_b == cid
            peaks = peak_local_max(b_density, min_distance=peak_min_sep_px,
                                   labels=comp, exclude_border=False)
            if len(peaks) <= 1:
                new_lb[comp] = next_lid
                next_lid += 1
                continue
            markers = np.zeros_like(comp, dtype=np.int32)
            for i, (yy, xx) in enumerate(peaks, 1):
                markers[yy, xx] = i
            ws = watershed(-b_density, markers, mask=comp)
            n_pre = int(ws.max())

            ws = merge_shallow_basins(ws, b_density, MERGE_SADDLE_FRAC)
            n_post = int(ws.max())
            if n_post < n_pre:
                n_merged += (n_pre - n_post)
            for j in range(1, ws.max() + 1):
                sub_mask = ws == j
                if sub_mask.any():
                    new_lb[sub_mask] = next_lid
                    next_lid += 1
            n_split += 1
        if n_split > 0:
            print(f"  Density-peak split: {n_split} components → {next_lid - 1} sub-cores "
                  f"(saddle-merge frac={MERGE_SADDLE_FRAC} re-merged {n_merged} spurious cuts)",
                  flush=True)
        lb_b = new_lb
        n_b_cc = next_lid - 1

    if n_b_cc == 0:
        print("  No B components.", flush=True)

    b_ix, b_iy = to_px(b_sub)


    final_cores = []
    n_big = 0
    for cid in range(1, n_b_cc + 1):
        comp = lb_b == cid
        in_comp = comp[b_iy, b_ix]
        if not in_comp.any():
            continue
        is_big = int(comp.sum()) * (RESOLUTION ** 2) > BIG_COMP_AREA_UM2
        n_big += int(is_big)
        sig = CORE_SIGMA_BIG_PX if is_big else CORE_SIGMA_SMALL_PX
        raw_c = np.zeros((ny, nx), dtype=np.float32)
        np.add.at(raw_c, (b_iy[in_comp], b_ix[in_comp]), 1)
        dens_c = gaussian_filter(raw_c, sigma=sig)
        if dens_c.max() <= 0:
            continue
        thr = CORE_THRESH_REL * dens_c.max()
        tight = dens_c > thr
        tight &= comp
        if CORE_CLOSING_PX > 0:
            tight = binary_closing(tight, structure=disk(CORE_CLOSING_PX))
        if CORE_OPENING_PX > 0:
            tight = binary_opening(tight, structure=disk(CORE_OPENING_PX))
        tight &= comp
        lb_p, n_p = label(tight)
        for pid in range(1, n_p + 1):
            piece = lb_p == pid

            for cc in clean_core(piece, dens_c, thr):
                final_cores.append((cc, is_big))
    print(f"  Tight-core re-split: {n_b_cc} sub-cores → {len(final_cores)} dense cores "
          f"(adaptive σ small={CORE_SIGMA_SMALL_PX}/big={CORE_SIGMA_BIG_PX}px, "
          f"{n_big} big components; thr={CORE_THRESH_REL} close={CORE_CLOSING_PX} "
          f"open={CORE_OPENING_PX} neck={CORE_NECK_RADIUS_PX})",
          flush=True)


    split_max_px = SPLIT_MAX_CORE_AREA_UM2 / (RESOLUTION ** 2)
    if final_cores:
        resplit, n_over, n_gained = [], 0, 0
        for core, is_big in final_cores:


            if (not is_big) and int(core.sum()) > split_max_px:
                pieces = split_oversized_core(core, b_density, b_iy, b_ix)
                n_over += 1
                n_gained += max(0, len(pieces) - 1)
                resplit.extend((p, is_big) for p in pieces)
            else:
                resplit.append((core, is_big))
        final_cores = resplit
        print(f"  Size-conditional split: {n_over} oversized cores "
              f"(>{SPLIT_MAX_CORE_AREA_UM2 / 1e6:.2f}mm²) → +{n_gained} sub-cores "
              f"(peak={SPLIT_PEAK_SEP_UM} frac={SPLIT_MERGE_FRAC} thr={SPLIT_CORE_THRESH}); "
              f"{len(final_cores)} cores total", flush=True)

    min_core_area_px = int(MIN_B_CORE_AREA_UM2 / (RESOLUTION ** 2))

    t_sub = df[df["class"] == "T_cell"]
    t_ix, t_iy = to_px(t_sub)
    p_sub = df[df["class"] == "Plasma"]
    p_ix, p_iy = to_px(p_sub)
    le_sub = df[df["class"] == "Lymphatic_Endothelial"]
    le_ix, le_iy = to_px(le_sub)

    bcl6_sub = df[df["bcl6_high"]]
    bc_ix, bc_iy = to_px(bcl6_sub)


    aicda_ix, aicda_iy = to_px(df[df["AICDA_pos"]])
    bcl6b_ix, bcl6b_iy = to_px(df[df["BCL6_pos"]])
    mki67_ix, mki67_iy = to_px(df[df["MKI67_pos"]])
    gcb_ix, gcb_iy = to_px(df[df["gcb_pos"]])
    tfh_ix, tfh_iy = to_px(df[df["tfh_pos"]])
    hev_ix, hev_iy = to_px(df[df["hev_pos"]])
    mregdc_ix, mregdc_iy = to_px(df[df["mregdc"]])
    fdc_ix, fdc_iy = to_px(df[df["fdc_pos"]])

    _aicda_sub = df[df["AICDA_pos"]]
    aicda_xy_all = _aicda_sub[["x_centroid", "y_centroid"]].values
    aicda_pix_ix, aicda_pix_iy = to_px(_aicda_sub)
    b_xy_all = b_sub[["x_centroid", "y_centroid"]].values
    _clust_rng = np.random.default_rng(0)


    all_ix, all_iy = to_px(df)

    halo_r_px = int(round(HALO_RADIUS_UM / RESOLUTION))
    min_tls_area_px = int(MIN_TLS_AREA_UM2 / (RESOLUTION ** 2))


    print("[3] Verifying TLS structure (B core + halo + tumor zone)...", flush=True)
    tls_grid = np.zeros((ny, nx), dtype=np.int32)
    tls_clusters = []
    next_id = 1

    rejected_count = {"size": 0, "density": 0, "halo": 0,
                      "tls_area": 0, "lymph_density": 0,
                      "region_b_density": 0,
                      "reshape_too_small": 0}
    reshape_count = 0
    for core, is_big in final_cores:
        n_b_in = int(core[b_iy, b_ix].sum()) if len(b_ix) else 0
        core_area_px = int(core.sum())
        core_area_mm2 = core_area_px * (RESOLUTION ** 2) / 1e6
        if n_b_in < MIN_B_CELLS or core_area_px < min_core_area_px:
            rejected_count["size"] += 1
            continue

        b_dens = n_b_in / core_area_mm2 if core_area_mm2 > 0 else 0


        core_dens_gate = MIN_B_DENSITY_BIG_MM2 if is_big else MIN_B_DENSITY_CELLS_MM2
        if b_dens < core_dens_gate:
            rejected_count["density"] += 1
            continue


        tls_region = binary_dilation(core, structure=disk(halo_r_px)) & tissue


        props_r_chk, _comp_chk = region_shape_stats(tls_region, core)
        if props_r_chk is None:
            rejected_count["tls_area"] += 1
            continue
        M_full = props_r_chk.major_axis_length
        m_full = props_r_chk.minor_axis_length
        ar_chk = (m_full / M_full) if M_full > 0 else 1.0
        peri_chk = props_r_chk.perimeter
        circ_chk = (4 * np.pi * props_r_chk.area / (peri_chk ** 2)) if peri_chk > 0 else 0.0
        reshaped = False
        if (ar_chk < MIN_REGION_AXIS_RATIO or circ_chk < MIN_REGION_CIRCULARITY) and M_full > 0:
            ys_r, xs_r = np.where(tls_region)
            if len(ys_r) >= 5:
                pts = np.column_stack([ys_r, xs_r]).astype(np.float64)
                ctr = pts.mean(axis=0)
                pts_c = pts - ctr
                cov = (pts_c.T @ pts_c) / len(pts_c)
                eigvals, eigvecs = np.linalg.eigh(cov)
                major_idx = int(np.argmax(eigvals))
                major_vec = eigvecs[:, major_idx]

                proj = pts_c @ major_vec

                cyi, cxi = np.where(core)
                if len(cyi):
                    core_proj = (np.column_stack([cyi, cxi]).astype(np.float64) - ctr) @ major_vec
                    core_center = float(np.median(core_proj))
                else:
                    core_center = 0.0
                M_actual = proj.max() - proj.min()
                M_target_half = min(M_actual / 2, m_full / MIN_REGION_AXIS_RATIO / 2)
                keep = np.abs(proj - core_center) <= M_target_half
                cand = np.zeros((ny, nx), dtype=bool)
                cand[ys_r[keep], xs_r[keep]] = True
                cand &= tissue


                core_halo = binary_dilation(core, structure=disk(halo_r_px)) & tissue
                if (cand & core).any():
                    tls_region = cand | core_halo
                    reshape_count += 1
                    reshaped = True


        tls_area_px = int(tls_region.sum())
        if tls_area_px < min_tls_area_px:
            if reshaped:
                rejected_count["reshape_too_small"] += 1
            else:
                rejected_count["tls_area"] += 1
            continue
        tls_area_mm2 = tls_area_px * (RESOLUTION ** 2) / 1e6

        if len(t_ix):
            n_t_region = int(tls_region[t_iy, t_ix].sum())
        else:
            n_t_region = 0
        if n_t_region < MIN_T_IN_REGION:
            rejected_count["halo"] += 1
            continue

        n_p_region = int(tls_region[p_iy, p_ix].sum()) if len(p_ix) else 0
        n_b_region = int(tls_region[b_iy, b_ix].sum()) if len(b_ix) else 0

        lymphoid_density = (n_b_region + n_t_region + n_p_region) / tls_area_mm2
        if lymphoid_density < MIN_LYMPHOID_DENSITY_CELLS_MM2:
            rejected_count["lymph_density"] += 1
            continue


        region_b_density_check = n_b_region / tls_area_mm2 if tls_area_mm2 > 0 else 0.0
        if region_b_density_check < MIN_B_DENSITY_REGION_MM2:
            rejected_count["region_b_density"] += 1
            continue

        n_le_inside = int(tls_region[le_iy, le_ix].sum()) if len(le_ix) else 0
        status = "mature" if n_le_inside > 0 else "immature"


        cells_in_region_mask = tls_region[all_iy, all_ix].astype(bool)
        n_cells_in_region = int(cells_in_region_mask.sum())
        if zones_available and n_cells_in_region > 0:
            zone_counts_series = df.loc[cells_in_region_mask, "boundary_zone"].value_counts()
            zone_counts = zone_counts_series.to_dict()
        else:
            zone_counts = {}
        tumor_class, fracs = classify_tls_tumor(zone_counts, n_cells_in_region)


        if zones_available and n_cells_in_region > 0:
            tr_counts = df.loc[cells_in_region_mask, "gen7_tumor_region"].value_counts()
            tr_top = tr_counts.index[0] if len(tr_counts) else ""
        else:
            tr_top = ""

        if len(bc_ix):
            n_bcl6_inside = int(tls_region[bc_iy, bc_ix].sum())
        else:
            n_bcl6_inside = 0

        dt = distance_transform_edt(tls_region)
        max_iy, max_ix = np.unravel_index(dt.argmax(), dt.shape)
        cx = max_ix * RESOLUTION + x_min
        cy = max_iy * RESOLUTION + y_min
        max_r_um = dt.max() * RESOLUTION


        props_r, _comp_r = region_shape_stats(tls_region, core)
        region_b_density = n_b_region / tls_area_mm2 if tls_area_mm2 > 0 else 0.0
        if props_r is not None:
            peri_px = props_r.perimeter
            region_solidity = float(props_r.solidity)
            region_circularity = (4 * np.pi * props_r.area / (peri_px ** 2)) if peri_px > 0 else 0.0
            region_axis_ratio = (props_r.minor_axis_length / props_r.major_axis_length
                                 if props_r.major_axis_length > 0 else 1.0)
        else:
            region_solidity = region_circularity = region_axis_ratio = 0.0
        core_b_density = (n_b_in / core_area_mm2) if core_area_mm2 > 0 else 0.0
        compactness = (core_area_mm2 / tls_area_mm2) if tls_area_mm2 > 0 else 0.0

        tls_grid[tls_region & (tls_grid == 0)] = next_id
        tls_clusters.append({
            "id": next_id,
            "status": status,
            "tumor_class": tumor_class,
            "intratumor_frac": fracs["intra"],
            "peritumor_frac": fracs["peri"],
            "distant_frac": fracs["distant"],
            "unmapped_frac": fracs["unmapped"],
            "gen7_tumor_region_modal": tr_top,
            "zone_counts": {k: int(v) for k, v in zone_counts.items() if k},
            "n_cells_in_region": n_cells_in_region,
            "n_b": n_b_in,
            "n_t_region": n_t_region,
            "n_p_region": n_p_region,
            "n_le": n_le_inside,
            "n_bcl6_inside": n_bcl6_inside,
            "area_mm2": float(tls_region.sum() * RESOLUTION ** 2 / 1e6),
            "core_area_mm2": float(core_area_mm2),
            "region_b_density": float(region_b_density),
            "region_solidity": region_solidity,
            "region_circularity": float(region_circularity),
            "region_axis_ratio": float(region_axis_ratio),
            "core_b_density": float(core_b_density),
            "compactness": float(compactness),
            "reshaped_to_disk": bool(reshaped),


            "core_outside_region_px": int((core & ~tls_region).sum()),
            "core": core,
            "region": tls_region,
            "cx": float(cx), "cy": float(cy), "max_r_um": float(max_r_um),
        })
        next_id += 1


    for c in tls_clusters:
        owned = (tls_grid == c["id"])
        area_px = int(owned.sum())
        area_mm2 = area_px * (RESOLUTION ** 2) / 1e6
        n_t = int(owned[t_iy, t_ix].sum()) if len(t_ix) else 0
        n_p = int(owned[p_iy, p_ix].sum()) if len(p_ix) else 0
        n_b_reg = int(owned[b_iy, b_ix].sum()) if len(b_ix) else 0
        n_le_ = int(owned[le_iy, le_ix].sum()) if len(le_ix) else 0
        n_bc_ = int(owned[bc_iy, bc_ix].sum()) if len(bc_ix) else 0
        cmask = owned[all_iy, all_ix].astype(bool)
        n_cells = int(cmask.sum())
        if zones_available and n_cells > 0:
            zc = df.loc[cmask, "boundary_zone"].value_counts().to_dict()
            trc = df.loc[cmask, "gen7_tumor_region"].value_counts()
            tr_top = trc.index[0] if len(trc) else ""
        else:
            zc, tr_top = {}, ""
        tclass, fr = classify_tls_tumor(zc, n_cells)


        n_b_core = c["n_b"]
        def _cnt(iy, ix):
            return int(owned[iy, ix].sum()) if len(ix) else 0
        n_aicda = _cnt(aicda_iy, aicda_ix)
        n_bcl6b = _cnt(bcl6b_iy, bcl6b_ix)
        n_mki67 = _cnt(mki67_iy, mki67_ix)
        n_gcb = _cnt(gcb_iy, gcb_ix)
        n_tfh = _cnt(tfh_iy, tfh_ix)
        n_hev = _cnt(hev_iy, hev_ix)
        n_mregdc = _cnt(mregdc_iy, mregdc_ix)
        gc_aid_frac = (n_aicda / n_b_core) if n_b_core else 0.0
        bcl6_b_frac = (n_bcl6b / n_b_core) if n_b_core else 0.0
        gc_b_frac = (n_gcb / n_b_core) if n_b_core else 0.0
        mki67_b_frac = (n_mki67 / n_b_core) if n_b_core else 0.0
        tfh_frac = (n_tfh / n_t) if n_t else 0.0
        denom_lymph = n_b_reg + n_t + n_p
        plasma_frac = (n_p / denom_lymph) if denom_lymph else 0.0
        psi = plasma_shell_index(owned, p_iy, p_ix, ny, nx)
        moran = bivariate_morans_i(
            rasterize(b_sub, ny, nx, x_min, y_min),
            rasterize(t_sub, ny, nx, x_min, y_min), owned)
        lamp3_density = (n_mregdc / area_mm2) if area_mm2 > 0 else 0.0
        hev_density = (n_hev / area_mm2) if area_mm2 > 0 else 0.0


        core_mask = c["core"]
        fdc_focus = int(core_mask[fdc_iy, fdc_ix].sum()) if len(fdc_ix) else 0

        if len(aicda_pix_ix):
            _ain = owned[aicda_pix_iy, aicda_pix_ix].astype(bool)
            aicda_xy_in = aicda_xy_all[_ain]
        else:
            aicda_xy_in = aicda_xy_all[:0]
        _bin = owned[b_iy, b_ix].astype(bool) if len(b_ix) else np.zeros(0, dtype=bool)
        region_b_xy = b_xy_all[_bin] if len(b_ix) else b_xy_all[:0]
        clust_ratio = aicda_clust_ratio(aicda_xy_in, region_b_xy, _clust_rng)


        if ("gc_score" in b_sub.columns) and len(b_ix) and _bin.any():
            _bgc = b_sub["gc_score"].values[_bin]
            _fin = np.isfinite(_bgc)
            n_region_b = int(_fin.sum())
            gc_score_mean = float(np.nanmean(_bgc)) if n_region_b > 0 else None
        else:
            gc_score_mean = None
            n_region_b = 0

        c.update({
            "area_mm2": float(area_mm2),
            "n_t_region": n_t, "n_p_region": n_p, "n_b_region": n_b_reg,
            "n_le": n_le_, "n_bcl6_inside": n_bc_,
            "n_cells_in_region": n_cells,
            "zone_counts": {k: int(v) for k, v in zc.items() if k},
            "gen7_tumor_region_modal": tr_top,
            "tumor_class": tclass,
            "intratumor_frac": fr["intra"], "peritumor_frac": fr["peri"],
            "distant_frac": fr["distant"], "unmapped_frac": fr["unmapped"],
            "region_b_density": float(n_b_reg / area_mm2) if area_mm2 > 0 else 0.0,
            "compactness": float(c["core_area_mm2"] / area_mm2) if area_mm2 > 0 else 0.0,
            "status": "mature" if n_le_ > 0 else "immature",

            "gc_aid_frac": float(gc_aid_frac), "bcl6_b_frac": float(bcl6_b_frac),
            "gc_b_frac": float(gc_b_frac), "mki67_b_frac": float(mki67_b_frac),
            "tfh_frac": float(tfh_frac), "plasma_frac": float(plasma_frac),
            "plasma_frac_of_lymphoid": float(plasma_frac),
            "plasma_shell_index": (None if _isbad(psi) else float(psi)),
            "plasma_shell_available": bool(not _isbad(psi)),
            "lamp3_density": float(lamp3_density), "hev_density": float(hev_density),
            "moran_i": (None if _isbad(moran) else float(moran)),
            "b_t_segregation_moran_i": (None if _isbad(moran) else float(moran)),
            "n_aicda_b_in": n_aicda, "n_tfh_in": n_tfh,

            "fdc_focus": fdc_focus,
            "aicda_clust_ratio": (None if clust_ratio is None else float(clust_ratio)),

            "gc_score_mean": gc_score_mean,
            "n_region_b": n_region_b,
        })


    lamp3_vals = np.array([c["lamp3_density"] for c in tls_clusters], dtype=float)
    hev_vals = np.array([c["hev_density"] for c in tls_clusters], dtype=float)
    area_vals = np.array([c["area_mm2"] for c in tls_clusters], dtype=float)
    norms = {
        "lamp3_avail": lamp3_avail,
        "lamp3_mean": float(lamp3_vals.mean()) if lamp3_vals.size else 0.0,
        "lamp3_std": float(lamp3_vals.std()) if lamp3_vals.size > 1 else 0.0,
        "hev_p90": float(np.percentile(hev_vals, 90)) if hev_vals.size else 0.0,
        "area_p90": float(np.percentile(area_vals, 90)) if area_vals.size else 0.0,
    }
    for c in tls_clusters:
        c.update(compute_maturity(c, norms))

    n_tls = len(tls_clusters)
    n_mat = sum(1 for c in tls_clusters if c["status"] == "mature")
    n_imm = n_tls - n_mat
    class_counts = {}
    for c in tls_clusters:
        class_counts[c["tumor_class"]] = class_counts.get(c["tumor_class"], 0) + 1
    rung_counts = {0: 0, 1: 0, 2: 0}
    for c in tls_clusters:
        rung_counts[c["gc_rung"]] += 1
    n_gcpos = rung_counts[2]
    print(f"  Rejected: {rejected_count}", flush=True)
    print(f"  Reshaped (PCA-axis clip): {reshape_count}", flush=True)
    print(f"  TLS regions: {n_tls} (LE mature {n_mat})", flush=True)
    n_mature = sum(1 for c in tls_clusters if c["mature"])
    n_mat_lowconf = sum(1 for c in tls_clusters if c["mature"] and c["confidence"] == "LOW")
    n_fdc = sum(1 for c in tls_clusters if c.get("fdc_focus", 0) > 0)
    print(f"  Maturity gc_rung: gc_quiet={rung_counts[0]} low_GC={rung_counts[1]} "
          f"active_GC={rung_counts[2]} | MATURE(GC+ LOCKED)={n_mature} "
          f"(low_conf={n_mat_lowconf}) | TLS w/ fdc_focus>0={n_fdc}", flush=True)
    print(f"  Tumor classes: {class_counts}", flush=True)
    if n_tls == 0:
        print("  No TLS detected; skipping output.", flush=True)
        return
    for c in tls_clusters:
        rs = "R" if c["reshaped_to_disk"] else " "
        sc = c["tls_development_5k"]
        print(f"    TLS-{c['id']} [{rs}] dev={sc:.2f} {c['gc_rung_label']}/{c['tls_composition']} "
              f"[{c['tumor_class']}] B={c['n_b']} "
              f"AICDAf={c['gc_aid_frac']:.3f} bcl6f={c['bcl6_b_frac']:.3f} "
              f"tfhf={c['tfh_frac']:.2f} plasf={c['plasma_frac']:.2f} "
              f"gc_ax={c['gc_axis']} struct={c['struct_axis']:.2f} "
              f"zone={c['gen7_tumor_region_modal']}", flush=True)


    print("[4] Mapping cells...", flush=True)
    out_dir.mkdir(parents=True, exist_ok=True)
    c_ix, c_iy = to_px(df)
    rid_arr = tls_grid[c_iy, c_ix]
    tumor_class_map = {c["id"]: c["tumor_class"] for c in tls_clusters}
    intra_map = {c["id"]: c["intratumor_frac"] for c in tls_clusters}
    peri_map = {c["id"]: c["peritumor_frac"] for c in tls_clusters}
    dist_map = {c["id"]: c["distant_frac"] for c in tls_clusters}
    unmap_map = {c["id"]: c["unmapped_frac"] for c in tls_clusters}

    mat_map = {c["id"]: c["tls_development_5k"] for c in tls_clusters}
    rung_map = {c["id"]: c["gc_rung_label"] for c in tls_clusters}
    gcp_map = {c["id"]: c["gc_present"] for c in tls_clusters}
    comp_map = {c["id"]: c["tls_composition"] for c in tls_clusters}
    mature_map = {c["id"]: bool(c["mature"]) for c in tls_clusters}
    conf_map = {c["id"]: c["confidence"] for c in tls_clusters}
    df["tls_region"] = [f"TLS-{rid}" if rid > 0 else "Outside" for rid in rid_arr]


    _core_union = np.zeros((ny, nx), dtype=bool)
    for _cl in tls_clusters:
        _core_union |= _cl["core"]
    df["in_tls_core"] = _core_union[all_iy, all_ix]


    _GRP = {
        "B_cell": ("score_B",  ["B_cell"]),
        "NK_T":   ("score_T",  ["T_cell", "CD4_T", "CD8_T", "T_reg", "NK_T", "NK"]),
        "DC":     ("score_DC", ["DC", "cDC", "cDC1", "cDC2", "pDC", "mregDC", "Langerhans_cell"]),
    }
    _T2G = {t: g for g, (_, ts) in _GRP.items() for t in ts}
    df["g7grp"] = df["cell_type"].map(_T2G)
    df["gen7_score"] = np.nan
    _score_path = PROGRAM_SCORE_DIR / f"{sample}.parquet"
    if _score_path.exists():
        _g7 = pd.read_parquet(_score_path); _g7["cell_id"] = _g7["cell_id"].astype(str)
        _g7 = _g7.set_index("cell_id")
        _cid = df["cell_id"].astype(str)
        for _grp, (_col, _ts) in _GRP.items():
            df[_col] = _cid.map(_g7[_col])
            df.loc[df["g7grp"] == _grp, "gen7_score"] = df.loc[df["g7grp"] == _grp, _col]


    _reference = _load_reference_labels()
    _has_sB  = "score_B"  in df.columns
    _has_sDC = "score_DC" in df.columns
    for c in tls_clusters:
        _rcells = df[df["tls_region"] == f"TLS-{c['id']}"]
        if _has_sB:
            _bv = _rcells.loc[_rcells["cell_type"] == "B_cell", "score_B"]
            _ab = float(_bv.mean()) if len(_bv) and not np.isnan(_bv.mean()) else 0.0
        else:
            _ab = 0.0
        if _has_sDC:
            _dv = _rcells.loc[_rcells["cell_type"].isin(MAT_DC_TYPES), "score_DC"]
            _adc = float(_dv.mean()) if len(_dv) and not np.isnan(_dv.mean()) else 0.0
        else:
            _adc = 0.0
        c["avg_score_B"]  = _ab
        c["avg_score_DC"] = _adc
        if cancer_type == "Lymph_nodes":


            c["TLS_State"] = MAT_LN_LABEL
            c["TLS_State_source"] = "reference"
        else:
            _ported = classify_tls_state(_ab, _adc)
            _key = f"{cancer_type}_{sample}_{c['id']}"
            _his = _reference.get(_key)
            c["TLS_State"] = _his if _his is not None else _ported
            c["TLS_State_source"] = "reference_override" if _his is not None else "ported"
    state_map = {c["id"]: c["TLS_State"] for c in tls_clusters}

    df["tls_status"] = [gcp_map.get(rid, "") for rid in rid_arr]
    df["tls_development_5k"] = [mat_map.get(rid) if rid > 0 else None for rid in rid_arr]

    df["TLS_State"] = [state_map.get(rid, "") for rid in rid_arr]
    df["gc_rung_label"] = [rung_map.get(rid, "") for rid in rid_arr]
    df["gc_present"] = [gcp_map.get(rid, "") for rid in rid_arr]
    df["mature"] = [mature_map.get(rid) if rid > 0 else None for rid in rid_arr]
    df["confidence"] = [conf_map.get(rid, "") for rid in rid_arr]
    df["tls_composition"] = [comp_map.get(rid, "") for rid in rid_arr]
    df["tls_tumor_class"] = [tumor_class_map.get(rid, "") for rid in rid_arr]
    df["tls_intratumor_frac"] = [intra_map.get(rid) if rid > 0 else None for rid in rid_arr]
    df["tls_peritumor_frac"] = [peri_map.get(rid) if rid > 0 else None for rid in rid_arr]
    df["tls_distant_frac"] = [dist_map.get(rid) if rid > 0 else None for rid in rid_arr]
    df["tls_unmapped_frac"] = [unmap_map.get(rid) if rid > 0 else None for rid in rid_arr]

    csv_path = out_dir / f"{sample}_tls.csv"
    df[["cell_id", "tls_region", "in_tls_core", "TLS_State", "tls_status",
        "tls_development_5k", "gc_rung_label", "gc_present", "mature", "confidence",
        "tls_composition",
        "tls_tumor_class", "tls_intratumor_frac", "tls_peritumor_frac",
        "tls_distant_frac", "tls_unmapped_frac",
        "boundary_zone", "gen7_tumor_region",
        "bcl6_count", "bcl6_high", "cell_type"]].to_csv(csv_path, index=False)
    print(f"  Saved: {csv_path}", flush=True)


    meta_path = out_dir / f"{sample}_tls_meta.json"
    meta = {
        "sample": sample, "dataset": dataset, "cancer_type": cancer_type,
        "n_cells_total": int(len(df)),
        "n_b_total": n_b, "n_t_total": n_t, "n_p_total": n_p, "n_le_total": n_le,
        "n_tls": n_tls, "n_tls_mature": n_mat, "n_tls_immature": n_imm,
        "gen7_available": bool(zones_available),
        "tumor_class_counts": class_counts,
        "maturity": {
            "panel_mode": "molecular_5k",
            "scheme": "TLS_State — two-axis B x DC signature-score quadrant",
            "definition": "TLS_State classifies each TLS on two per-TLS axes: avg_score_B = mean(score_B over its cell_type=='B_cell' cells) and avg_score_DC = mean(score_DC over its DC-type cells, DC_TYPES=['pDC','cDC','cDC1','cDC2','mregDC','DC']), NaN->0; score_B/score_DC are per-cell decoupler-ULM signature scores (B-cell GCB set / DC follicular-DC set). The frozen thresholds tB=-0.12854, tDC=-0.10229 split the plane into 4 quadrants: B>=tB & DC>=tDC -> Mature TLS; B>=tB & DC<tDC -> FDC-deficient TLS; B<tB & DC>=tDC -> Premature TLS; B<tB & DC<tDC -> Early TLS. A matching label in the supplied authoritative per-TLS reference CSV overrides the quadrant rule; reference lymph-node regions use Canonical LN. avg_score_B/avg_score_DC/TLS_State/TLS_State_source are stored per TLS below. This run does not compute rule/reference agreement.",
            "ref": "molecular state reference table",
            "thresholds_BxDC": {"THRESH_B": MAT_THRESH_B, "THRESH_DC": MAT_THRESH_DC, "DC_TYPES": MAT_DC_TYPES},
            "TLS_State_counts": {st: sum(1 for c in tls_clusters if c["TLS_State"] == st)
                                 for st in MAT_STATES + [MAT_LN_LABEL]},
            "n_state_from_override": sum(1 for c in tls_clusters if c.get("TLS_State_source") == "reference_override"),
            "n_state_from_rule": sum(1 for c in tls_clusters if c.get("TLS_State_source") == "ported"),
            "n_state_reference": sum(1 for c in tls_clusters if c.get("TLS_State_source") == "reference"),
            "validated": None,
            "validation_status": "NOT_EVALUATED_THIS_RUN",
            "validation_note": "Rule/reference agreement is not computed in this run. Matching reference labels are used directly; other tumor TLS use the molecular rule. The frozen thresholds are not independently pathology calibrated.",

            "gc_rung_counts": {lab: sum(1 for c in tls_clusters if c["gc_rung_label"] == lab)
                               for lab in ("gc_quiet", "low_GC", "active_GC")},
            "n_mature": sum(1 for c in tls_clusters if c["mature"]),
            "n_gc_present": sum(1 for c in tls_clusters if c["gc_present"] == "mature"),
            "n_mki67_corroborated": sum(1 for c in tls_clusters if c["mki67_corroborated"]),
            "composition_counts": {lab: sum(1 for c in tls_clusters if c["tls_composition"] == lab)
                                   for lab in ("b_dominant", "mixed", "plasma_dominant")},
            "weights": {"GC": W_GC, "Tfh": W_TFH, "Plasma": W_PLASMA, "Struct": W_STRUCT},
        },
        "thresholds": {
            "INTRA_FRAC_MIN": INTRA_FRAC_MIN,
            "PERI_FRAC_MIN": PERI_FRAC_MIN,
            "DISTANT_FRAC_MIN": DISTANT_FRAC_MIN,
            "SAT_AICDA": SAT_AICDA, "SAT_BCL6": SAT_BCL6, "SAT_MKI67": SAT_MKI67,
            "SAT_TFH": SAT_TFH, "SAT_PLASMA_FRAC": SAT_PLASMA_FRAC,
            "RUNG_AICDA_FLOOR": RUNG_AICDA_FLOOR, "RUNG_AICDA_ACTIVE": RUNG_AICDA_ACTIVE,
            "RUNG_BCL6_COPRIMARY": RUNG_BCL6_COPRIMARY, "RUNG_TFH_COPRIMARY": RUNG_TFH_COPRIMARY,
            "AICDA_FLOOR_N": AICDA_FLOOR_N, "MKI67_CORROB_CUT": MKI67_CORROB_CUT,
            "TAU_GC": TAU_GC, "TAU_GC_LOW": TAU_GC_LOW, "MIN_B_CELLS": MIN_B_CELLS,
            "COMP_PLASMA_DOM": COMP_PLASMA_DOM, "COMP_B_DOM": COMP_B_DOM,
            "_provenance": "Frozen algorithm settings and uncalibrated cohort-percentile priors; n_aicda>=4 is a fixed minimum-count setting. These thresholds are not independently pathology calibrated.",
        },
        "tls": [{k: v for k, v in c.items() if k not in ("core", "region")}
                for c in tls_clusters],
    }
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2, default=lambda x: None if x is None else x)
    print(f"  Saved: {meta_path}", flush=True)


    print("[5] Figure...", flush=True)
    from matplotlib.lines import Line2D
    score_cmap = plt.get_cmap("YlOrRd")


    MK = {
        "b":      "#8e24aa",
        "mki67":  "#fbc02d",
        "bcl6":   "#e040fb",
        "aicda":  "#d32f2f",
        "t":      "#64b5f6",
        "tfh":    "#2e7d32",
        "plasma": "#ef6c00",
        "hev":    "#00bcd4",
        "other":  "#ececec",
    }
    axis_col = {"GC": MK["aicda"], "Tfh": MK["tfh"], "Pl": MK["plasma"], "St": "#455a64"}
    zt_map = {"intra_tumor": "INT", "peri_tumor": "PER", "tumor_distant": "DST", "mixed": "MIX"}


    slide_w = max(x_max - x_min, 1.0)
    slide_h = max(y_max - y_min, 1.0)
    ov_w = 16.0
    ov_h = float(np.clip(ov_w * (slide_h / slide_w), 4.0, 16.0))


    _upi = max(slide_w / ov_w, slide_h / ov_h)
    cps = float(np.clip((72.0 * 13.5 / _upi) ** 2, CELL_POINT_SIZE, 9.0))
    fig, ax = plt.subplots(figsize=(ov_w, ov_h))
    other = df[df["class"] == "Other"]
    if len(other):
        ax.scatter(other.x_centroid, other.y_centroid, c=COLORS["Other"],
                   s=cps, alpha=0.08, rasterized=True, zorder=0,
                   edgecolors='none')


    overlay = np.zeros((ny, nx, 4), dtype=np.float32)
    for c in tls_clusters:
        rgba = list(mcolors.to_rgba(MAT_COL.get(c["TLS_State"], "#bdbdbd")))
        rgba[3] = 0.20
        overlay[c["region"]] = rgba
    ax.imshow(overlay, extent=[x_min, x_max, y_min, y_max], origin='lower',
              aspect='auto', zorder=2, interpolation='nearest')
    cell_alpha = {"T_cell": 0.7, "Plasma": 0.8, "B_cell": 0.9, "Lymphatic_Endothelial": 0.95}
    cell_z = {"Plasma": 3, "T_cell": 4, "B_cell": 5, "Lymphatic_Endothelial": 6}
    for cls in ("T_cell", "Plasma", "B_cell", "Lymphatic_Endothelial"):
        sub = df[df["class"] == cls]
        if len(sub):
            ax.scatter(sub.x_centroid, sub.y_centroid, c=COLORS[cls],
                       s=cps, alpha=cell_alpha[cls], rasterized=True,
                       zorder=cell_z[cls], edgecolors='none')
    for c in tls_clusters:
        line_col = MAT_COL.get(c["TLS_State"], "#555555")
        line_ls = "-"
        for ctr in find_contours(c["region"].astype(float), 0.5):
            ax.plot(ctr[:, 1] * RESOLUTION + x_min, ctr[:, 0] * RESOLUTION + y_min,
                    color=line_col, lw=1.2, alpha=0.9, linestyle=line_ls, zorder=8)
        if c["core"].any():
            for ctr in find_contours(c["core"].astype(float), 0.5):
                ax.plot(ctr[:, 1] * RESOLUTION + x_min, ctr[:, 0] * RESOLUTION + y_min,
                        color=line_col, lw=0.9, alpha=0.85, linestyle='-', zorder=9)


    AGG_OVERLAY = [("tagg", "#1f77b4", "T-cell aggregate"),
                   ("magg", "#d81b60", "Macrophage aggregate")]
    tls_halo_union = np.zeros((ny, nx), dtype=bool)
    for c in tls_clusters:
        tls_halo_union |= c["region"]
    agg_counts: dict = {}
    agg_summary: dict = {}
    agg_label_items: list = []

    agg_overlay_rgba = np.zeros((ny, nx, 4), dtype=np.float32)
    for _tag, _col, _lab in AGG_OVERLAY:
        _p = (Path(str(out_base)) / "work" / dataset / cancer_type / sample
              / f"{sample}_{_tag}_grid.npz")
        if not _p.exists():
            continue
        _z = np.load(_p)
        _g = _z["grid"]
        if _g.shape != (ny, nx):
            print(f"  [agg overlay] {_tag}: grid shape {_g.shape} != {(ny, nx)}; skipped", flush=True)
            continue
        _n_raw = int(_g.max())
        _keep = []
        for _i in range(1, _n_raw + 1):
            _m = (_g == _i)
            if not _m.any():
                continue


            _rm = binary_dilation(_m, structure=disk(halo_r_px)) & tissue


            if (_m & tls_halo_union).any():
                continue
            _keep.append((_i, _m, _rm))
        _g_f = np.zeros_like(_g)
        for _k, (_orig, _m, _rm) in enumerate(_keep, 1):
            _g_f[_m] = _k
        _n = len(_keep)
        agg_counts[_lab] = _n
        print(f"  [agg overlay] {_tag}: {_n_raw} raw -> {_n} kept "
              f"({_n_raw - _n} dropped for overlapping a TLS halo)", flush=True)


        _stat = {int(_id): _r for _id, _r in zip(
            _z["ids"], zip(_z["n_cells"], _z["core_area_mm2"], _z["density_cells_mm2"],
                           _z["cx"], _z["cy"]))} if "ids" in _z.files else {}
        _recs = []
        for _k, (_orig, _m, _rm) in enumerate(_keep, 1):
            _nc, _ca, _dn, _cx, _cy = _stat.get(_orig, (0, 0.0, 0.0, 0.0, 0.0))


            _in_reg = _rm[all_iy, all_ix]
            _n_reg = int(_in_reg.sum())
            _rec = {
                "id": _k, "id_before_tls_dedup": _orig,
                "n_cells_core": int(_nc), "n_cells_region": _n_reg,
                "core_area_mm2": round(float(_ca), 4),
                "region_area_mm2": round(float(int(_rm.sum()) * (RESOLUTION ** 2) / 1e6), 4),
                "core_density_cells_mm2": round(float(_dn), 1),
                "cx": float(_cx), "cy": float(_cy),
            }
            if _n_reg > 0:
                _vc = pd.Series(df["cell_type"].values[_in_reg]).value_counts()
                for _ct, _cnt in _vc.items():
                    _rec[f"frac_{_ct}"] = round(int(_cnt) / _n_reg, 4)
            _recs.append(_rec)
        _cp = out_dir / f"{sample}_{_tag}.csv"
        if _recs:
            pd.DataFrame(_recs).to_csv(_cp, index=False)
        elif _cp.exists():
            _cp.unlink()


        _pc = out_dir / f"{sample}_{_tag}_cells.csv"
        if _keep:
            _pfx_c = "T" if _tag == "tagg" else "M"
            _agg_reg = np.full(len(df), "Outside", dtype=object)
            _agg_core = np.zeros(len(df), dtype=bool)
            for _k, (_orig, _m, _rm) in enumerate(_keep, 1):
                _sel = _rm[all_iy, all_ix]
                _agg_reg[_sel] = f"{_pfx_c}{_k}"
                _agg_core |= _m[all_iy, all_ix]
            pd.DataFrame({
                "cell_id": df["cell_id"].values,
                f"{_pfx_c}agg_region": _agg_reg,
                f"in_{_pfx_c}agg_core": _agg_core,
                "cell_type": df["cell_type"].values,
            }).to_csv(_pc, index=False)
        elif _pc.exists():
            _pc.unlink()
        agg_summary[_tag] = {
            "label": _lab, "n_raw": _n_raw, "n_kept": _n,
            "n_dropped_tls_overlap": _n_raw - _n,
            "halo_radius_um": HALO_RADIUS_UM,
            "gates": {"min_cells": 50, "min_core_area_um2": 10000,
                      "min_core_density_cells_mm2": 8000},
            "tls_priority_rule": (f"aggregate dropped iff its CORE overlaps any TLS region "
                                  f"(TLS core + {HALO_RADIUS_UM}µm halo); halo-vs-halo does NOT count"),
            "table": _cp.name if _recs else None,
        }
        if _n == 0:
            continue


        _pfx = "T" if _tag == "tagg" else "M"
        for _k, (_orig, _m, _rm) in enumerate(_keep, 1):
            _ys, _xs = np.where(_m)
            _rys, _rxs = np.where(_rm)
            agg_label_items.append({
                "cx": float(_xs.mean() * RESOLUTION + x_min),
                "cy": float(_ys.mean() * RESOLUTION + y_min),
                "max_r_um": (float(np.max(np.hypot(_rxs - _rxs.mean(), _rys - _rys.mean())) * RESOLUTION)
                             if len(_rxs) else 1.0),
                "label": f"{_pfx}{_k}", "col": _col,
            })


        _rgba = list(mcolors.to_rgba(_col)); _rgba[3] = 0.20
        for _k, (_orig, _m, _rm) in enumerate(_keep, 1):
            agg_overlay_rgba[_rm] = _rgba
            for ctr in find_contours(_rm.astype(float), 0.5):
                ax.plot(ctr[:, 1] * RESOLUTION + x_min, ctr[:, 0] * RESOLUTION + y_min,
                        color=_col, lw=1.2, alpha=0.9, linestyle='-', zorder=8)
            for ctr in find_contours(_m.astype(float), 0.5):
                ax.plot(ctr[:, 1] * RESOLUTION + x_min, ctr[:, 0] * RESOLUTION + y_min,
                        color=_col, lw=0.9, alpha=0.85, linestyle='-', zorder=9)
    if agg_counts:
        ax.imshow(agg_overlay_rgba, extent=[x_min, x_max, y_min, y_max], origin='lower',
                  aspect='auto', zorder=2, interpolation='nearest')
    if agg_summary:
        _tm = out_dir / f"{sample}_tls_meta.json"
        if _tm.exists():
            _d = json.loads(_tm.read_text())
            _d["immune_aggregates"] = agg_summary
            _tm.write_text(json.dumps(_d, indent=1))
    if agg_counts:
        print(f"  [agg overlay] {agg_counts}", flush=True)


    gx0, gx1 = float(df.x_centroid.min()), float(df.x_centroid.max())
    gy0, gy1 = float(df.y_centroid.min()), float(df.y_centroid.max())
    mx = 0.015 * (gx1 - gx0); my = 0.015 * (gy1 - gy0)
    ax.set_xlim(gx0 - mx, gx1 + mx); ax.set_ylim(gy1 + my, gy0 - my); ax.set_aspect('equal')
    ax.set_xlabel("X (µm)", fontsize=11); ax.set_ylabel("Y (µm)", fontsize=11)
    ax.xaxis.set_major_locator(MultipleLocator(1000))
    ax.yaxis.set_major_locator(MultipleLocator(1000))


    label_items = [{"cx": c["cx"], "cy": c["cy"], "max_r_um": max(c["max_r_um"], 1.0),
                    "label": str(c["id"]), "col": MAT_COL.get(c["TLS_State"], "#555555")}
                   for c in tls_clusters] + agg_label_items
    cxs = np.array([it["cx"] for it in label_items], float)
    cys = np.array([it["cy"] for it in label_items], float)
    rr = np.array([max(it["max_r_um"], 1.0) for it in label_items], float)
    pw, ph = (gx1 - gx0), (gy1 - gy0)
    diag = float(np.hypot(pw, ph))
    sep = 0.050 * diag
    pad = 0.030 * diag
    tcx, tcy = float(cxs.mean()), float(cys.mean())
    dirx, diry = cxs - tcx, cys - tcy


    deg = np.hypot(dirx, diry) < 1e-6
    dirx = np.where(deg, 1.0, dirx); diry = np.where(deg, -1.0, diry)
    dn = np.hypot(dirx, diry); dn[dn < 1e-6] = 1.0
    lx = cxs + dirx / dn * (rr + pad)
    ly = cys + diry / dn * (rr + pad)
    lo_x, hi_x = gx0 - 0.10 * pw, gx1 + 0.10 * pw
    lo_y, hi_y = gy0 - 0.10 * ph, gy1 + 0.10 * ph
    n = len(tls_clusters)
    for _ in range(400):
        dx = lx[:, None] - lx[None, :]; dy = ly[:, None] - ly[None, :]
        dist = np.hypot(dx, dy); np.fill_diagonal(dist, 1e9)
        with np.errstate(invalid="ignore", divide="ignore"):
            f = np.where(dist < sep, (sep - dist) / np.maximum(dist, 1e-6) * 0.5, 0.0)
        mvx = (dx * f).sum(1); mvy = (dy * f).sum(1)

        ex = lx[:, None] - cxs[None, :]; ey = ly[:, None] - cys[None, :]
        ed = np.hypot(ex, ey); need = (rr[None, :] + pad) - ed
        with np.errstate(invalid="ignore", divide="ignore"):
            g = np.where(need > 0, need / np.maximum(ed, 1e-6), 0.0)
        mvx += (ex * g).sum(1); mvy += (ey * g).sum(1)
        lx += np.clip(mvx, -sep, sep) * 0.6
        ly += np.clip(mvy, -sep, sep) * 0.6
        lx = np.clip(lx, lo_x, hi_x); ly = np.clip(ly, lo_y, hi_y)
    for i, it in enumerate(label_items):
        line_col = it["col"]
        ax.plot([lx[i], it["cx"]], [ly[i], it["cy"]], color=line_col, lw=0.8,
                alpha=0.65, zorder=11, clip_on=False)
        ax.scatter([lx[i]], [ly[i]], s=210, marker='o', facecolor='white',
                   edgecolor=line_col, linewidth=1.8, alpha=0.98, zorder=12, clip_on=False)
        ax.text(lx[i], ly[i], it["label"], ha='center', va='center',
                fontsize=_label_fs(it["label"]),
                fontweight='bold', color=line_col, zorder=13, clip_on=False)


    disp_states = list(MAT_STATES) + ([MAT_LN_LABEL] if any(c["TLS_State"] == MAT_LN_LABEL for c in tls_clusters) else [])
    state_counts = {st: sum(1 for c in tls_clusters if c["TLS_State"] == st) for st in disp_states}
    if cancer_type == "Lymph_nodes":
        _hdr = f"Canonical LN={state_counts.get(MAT_LN_LABEL, 0)} (reference)"
    else:
        _hdr = f"Mature={state_counts['Mature TLS']}"
    cls_str = ", ".join(f"{k}={v}" for k, v in sorted(class_counts.items()))
    ax.set_title(f"{sample} ({cancer_type}, 5k): {n_tls} TLS — TLS_State (B×DC quadrant) "
                 f"[{_hdr}]  |  {cls_str}"
                 f"{'  [no tumor zones]' if not zones_available else ''}",
                 fontsize=12, fontweight='bold')
    ov_handles = [
        Patch(fc=COLORS["B_cell"], label=f"B_cell ({n_b})"),
        Patch(fc=COLORS["T_cell"], label=f"T_cell+NK_T ({n_t_all})"),
        Patch(fc=COLORS["Plasma"], label=f"Plasma ({n_p_all})"),
        Patch(fc=COLORS["Lymphatic_Endothelial"], label=f"Lymph. Endothelial ({n_le})"),
        Patch(fc="none", ec="none", label="— fill + outline = TLS_State —"),
    ] + [
        Patch(fc=MAT_COL[st], alpha=0.5, ec=MAT_COL[st], label=f"{st} ({state_counts[st]})")
        for st in disp_states
    ] + [
        Line2D([0], [0], color="#555", lw=0.9, linestyle="-", label="inner = high-density B core"),
    ] + ([
        Patch(fc="none", ec="none", label="— independent aggregate lines —"),
    ] + [
        Patch(fc=_c, alpha=0.5, ec=_c, label=f"{_l} ({agg_counts[_l]})")
        for _t, _c, _l in AGG_OVERLAY if _l in agg_counts
    ] if agg_counts else [])
    ax.legend(handles=ov_handles, loc='upper left', bbox_to_anchor=(1.01, 1),
              fontsize=9, framealpha=0.9, borderaxespad=0)
    fig.savefig(out_dir / f"{sample}_tls.png", dpi=200, bbox_inches='tight')
    plt.close(fig)


    CARD_MAX = int(os.environ.get("TLS_CARD_MAX", str(len(tls_clusters))))
    cards = sorted(tls_clusters,
                   key=lambda c: (c["tls_development_5k"] if c["tls_development_5k"] is not None else -1.0),
                   reverse=True)[:CARD_MAX]
    n_cards = len(cards)
    n_more = n_tls - n_cards
    NCOL = 4
    card_rows = max(1, int(np.ceil(n_cards / NCOL)))
    last_off = 0
    last_row_count = n_cards - (card_rows - 1) * NCOL


    fig_w = min(20.0, max(12.0, NCOL * 5.0))
    row_h = (fig_w / NCOL) * 1.20 + 0.5
    legend_in = 1.6
    fig_h_total = card_rows * row_h + legend_in
    fig2 = plt.figure(figsize=(fig_w, fig_h_total))

    gs = fig2.add_gridspec(2, 1, height_ratios=[card_rows * row_h, legend_in],
                           hspace=0.06, top=0.965, bottom=0.012)
    gs_cards = gs[0].subgridspec(card_rows, NCOL, hspace=0.30, wspace=0.16)
    ax_leg = fig2.add_subplot(gs[1]); ax_leg.axis("off")
    used_cells = set()
    fig2.suptitle(f"{sample} ({cancer_type}, 5k): per-TLS marker genes — "
                  f"{'all' if n_more <= 0 else 'top'} {n_cards} of {n_tls} TLS "
                  f"(card outline/header = TLS_State, B×DC quadrant)",
                  fontsize=13, fontweight='bold', y=0.990)


    def _sc(cax, sub, mask, color, s, z, ec='none', lw=0.0, marker='o'):
        if mask.any():
            cax.scatter(sub.x_centroid.values[mask], sub.y_centroid.values[mask],
                        c=color, s=s, alpha=0.95, zorder=z, edgecolors=ec,
                        linewidth=lw, marker=marker, rasterized=True)
    for k, c in enumerate(cards):
        r = k // NCOL
        col = (k % NCOL) + (last_off if r == card_rows - 1 else 0)
        used_cells.add((r, col))


        slot = gs_cards[r, col].subgridspec(2, 1, height_ratios=[5.6, 1.0], hspace=0.06)
        cax = fig2.add_subplot(slot[0])
        line_col = MAT_COL.get(c["TLS_State"], "#555555")
        ys_i, xs_i = np.where(c["region"])
        rx0 = x_min + xs_i.min() * RESOLUTION; rx1 = x_min + (xs_i.max() + 1) * RESOLUTION
        ry0 = y_min + ys_i.min() * RESOLUTION; ry1 = y_min + (ys_i.max() + 1) * RESOLUTION
        cxc = (rx0 + rx1) / 2; cyc = (ry0 + ry1) / 2
        half = max(rx1 - rx0, ry1 - ry0) / 2 + 60


        bp = cax.get_position()
        box_asp = (bp.width * fig_w) / max(bp.height * fig_h_total, 1e-6)
        hx = half * box_asp if box_asp >= 1.0 else half
        hy = half if box_asp >= 1.0 else half / box_asp
        m = ((df.x_centroid >= cxc - hx) & (df.x_centroid <= cxc + hx) &
             (df.y_centroid >= cyc - hy) & (df.y_centroid <= cyc + hy))
        sub = df[m]
        isB = (sub["class"] == "B_cell").values
        isT = (sub["class"] == "T_cell").values
        tfh = isT & sub["tfh_pos"].values
        tpl = isT & ~tfh
        plasma = (sub["class"] == "Plasma").values
        hev = sub["hev_pos"].values
        oth = ~(isB | isT | plasma | hev)

        _sc(cax, sub, oth, MK["other"], 3, 1)
        _sc(cax, sub, tpl, MK["t"], 5, 2)
        _sc(cax, sub, plasma, MK["plasma"], 9, 3, ec='#9a4a00', lw=0.2)
        _sc(cax, sub, hev, MK["hev"], 11, 4, ec='#006978', lw=0.4, marker='D')
        _sc(cax, sub, tfh, MK["tfh"], 12, 5, ec='#1b5e20', lw=0.3)

        _sc(cax, sub, isB, MK["b"], 7, 6)


        subB = sub[isB]
        def _disks(gene, color, ec, base, scale, z, p=0.7, alpha=0.8):
            e = subB[subB[gene].values > 0]
            if len(e) == 0:
                return
            gbx = (e.x_centroid.values // CARD_BIN_UM).astype(int)
            gby = (e.y_centroid.values // CARD_BIN_UM).astype(int)
            ag = (e.assign(_bx=gbx, _by=gby).groupby(["_bx", "_by"])
                  .agg(total=(gene, "sum"), x=("x_centroid", "mean"), y=("y_centroid", "mean"))
                  .reset_index().sort_values("total", ascending=False))
            cax.scatter(ag["x"], ag["y"], s=base + scale * np.power(ag["total"].values, p),
                        c=color, alpha=alpha, edgecolors=ec, linewidth=0.5,
                        zorder=z, rasterized=True)
        _disks("BCL6",  MK["bcl6"],  '#6a1b9a', 12, 11, 7)
        _disks("MKI67", MK["mki67"], '#a06800', 12, 11, 8)
        _disks("AICDA", MK["aicda"], '#7f0000', 14, 13, 9)
        for ctr in find_contours(c["region"].astype(float), 0.5):
            cax.plot(ctr[:, 1] * RESOLUTION + x_min, ctr[:, 0] * RESOLUTION + y_min,
                     color=line_col, lw=1.4, alpha=0.9,
                     linestyle="-", zorder=10)
        if c["core"].any():
            for ctr in find_contours(c["core"].astype(float), 0.5):
                cax.plot(ctr[:, 1] * RESOLUTION + x_min, ctr[:, 0] * RESOLUTION + y_min,
                         color=line_col, lw=3.0, alpha=1.0, zorder=10)
        cax.set_xlim(cxc - hx, cxc + hx); cax.set_ylim(cyc + hy, cyc - hy)
        cax.set_aspect('equal'); cax.set_xticks([]); cax.set_yticks([])
        for sp in cax.spines.values():
            sp.set_edgecolor(line_col); sp.set_linewidth(1.8)
        status = c["TLS_State"]
        comp_short = {"plasma_dominant": "plasma-dom", "b_dominant": "B-dom",
                      "mixed": "mixed"}[c["tls_composition"]]
        cax.set_title(f"#{c['id']}  {status}  B={c['avg_score_B']:.2f} DC={c['avg_score_DC']:.2f}\n"
                      f"[{zt_map.get(c['tumor_class'],'?')}] · {comp_short} · B={c['n_b']}",
                      fontsize=8, fontweight='bold', color=line_col, pad=3)


        iax = fig2.add_subplot(slot[1])
        rcells = df[(df["tls_region"] == f"TLS-{c['id']}") & df["g7grp"].notna()]
        sm = rcells.groupby("g7grp")["gen7_score"].mean()
        GRP_COL = {"B_cell": "#8e24aa", "NK_T": "#1565c0", "DC": "#7e57c2"}
        bars = [(g, float(sm.get(g))) for g in ["DC", "NK_T", "B_cell"] if g in sm.index and pd.notna(sm.get(g))]
        yp = np.arange(len(bars))
        iax.axvline(0.0, color="#999999", lw=0.6, zorder=0)
        iax.barh(yp, [b[1] for b in bars], color=[GRP_COL[b[0]] for b in bars],
                 height=0.7, edgecolor='#333', linewidth=0.3, zorder=2)
        mxb = max(0.35, max((abs(b[1]) for b in bars), default=0.35) * 1.15)
        iax.set_xlim(-mxb, mxb); iax.set_ylim(-0.6, max(len(bars) - 0.4, 0.6))
        iax.set_yticks(yp); iax.set_yticklabels([b[0] for b in bars], fontsize=7)
        iax.tick_params(axis='y', length=0)
        iax.set_xticks([-round(mxb, 2), 0.0, round(mxb, 2)]); iax.tick_params(axis='x', labelsize=5)
        for sp in ("top", "right"):
            iax.spines[sp].set_visible(False)
    for r in range(card_rows):
        for col in range(NCOL):
            if (r, col) not in used_cells:
                fig2.add_subplot(gs_cards[r, col]).axis("off")


    def _ml(color, label, ec=None, marker='o', ms=9):
        return Line2D([0], [0], marker=marker, linestyle='None', markersize=ms,
                      markerfacecolor=color, markeredgecolor=ec or color,
                      markeredgewidth=0.6, label=label)
    marker_handles = [
        _ml(MK["b"], "B cell (purple follicle)", '#4a148c', ms=9),
        _ml(MK["bcl6"], "BCL6 (magenta) — disk size = expression (GC master TF)", '#6a1b9a', ms=10),
        _ml(MK["aicda"], "AICDA (red) — disk size = expression (GC dark-zone / SHM)", '#7f0000', ms=10),
        _ml(MK["mki67"], "MKI67 (gold) — disk size = expression (proliferation)", '#a06800', ms=9),
        _ml(MK["tfh"], "Tfh  (CD3E+ CXCL13+ T)", '#1b5e20', ms=8),
        _ml(MK["t"], "T cell  (context)", ms=6),
        _ml(MK["plasma"], "Plasma", '#9a4a00', ms=8),
        _ml(MK["hev"], "HEV  (SELP+ endothelial)", '#006978', marker='D', ms=8),
    ]
    region_handles = [
        Line2D([0], [0], color=MAT_COL[st], lw=2,
               label=f"card outline {st} ({sum(1 for c in tls_clusters if c['TLS_State']==st)})")
        for st in disp_states
    ] + [
        Patch(fc="#8e24aa", label="mini-bar = 3 groups B_cell/NK_T/DC, each = its own gene-set ULM score (x=0 centre)"),
    ]
    all_handles = marker_handles + region_handles
    leg_ncol = 4 if fig_w >= 18 else (3 if fig_w >= 14 else 2)
    ax_leg.legend(handles=all_handles, loc="upper center", bbox_to_anchor=(0.5, 1.0),
                  ncol=leg_ncol, fontsize=9, framealpha=0.9, borderaxespad=0,
                  columnspacing=1.4, handletextpad=0.5,
                  title="per-cell markers (de-noised)   ·   card outline = TLS_State (B×DC quadrant)",
                  title_fontsize=9)
    note = (f"cards = {'all' if n_more <= 0 else 'top'} {n_cards} TLS by development score"
            + (f"  (+{n_more} more not shown)" if n_more > 0 else "")
            + "   ·   marker positive = count ≥ p90 among expressing B cells (per sample)")
    ax_leg.text(0.5, 0.0, note, ha="center", fontsize=8, color="#555", transform=ax_leg.transAxes)

    markers_path = out_dir / f"{sample}_tls_markers.png"
    fig2.savefig(markers_path, dpi=200, bbox_inches='tight')
    plt.close(fig2)
    print(f"  Saved: {out_dir / f'{sample}_tls.png'}", flush=True)
    print(f"  Saved: {markers_path}", flush=True)
    print(f"Done. {time.time() - t0:.1f}s", flush=True)


if __name__ == "__main__":
    main()
