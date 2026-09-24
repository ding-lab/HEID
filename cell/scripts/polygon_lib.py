#!/usr/bin/env python3

from __future__ import annotations
import os
PROJECTS_ROOT = os.environ.get("PROJECTS_ROOT", "/data/heid")
from pathlib import Path
_RELEASE = Path(__file__).resolve().parents[2]

import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

PC = Path(PROJECTS_ROOT + "/cell")
COHORT_REFERENCE_ROOT = PC / "inputs/cohort_reference"
RAW_POLYS = COHORT_REFERENCE_ROOT / "outputs/wmaps/raw_polys"
CELL_TABLES = PC / "data/cell_tables"

PX_UM = 0.2125
CROP = 224
HALF = 112
PATCH = 14
GRID = 16
FILL_SHIFT = 4
FILL_SCALE = 1 << FILL_SHIFT

sys.path.insert(0, str(_RELEASE / "cell/scripts"))
from label_maps import COARSE11, FINE2REFINE, REFINE2COARSE11
from paths import expand_env as _expand_env


def coarse11(series: pd.Series) -> pd.Series:
    return series.map(lambda x: REFINE2COARSE11.get(FINE2REFINE.get(x)))


def load_cell_table(sample: str) -> pd.DataFrame:
    table = pd.read_parquet(CELL_TABLES / f"{sample}.parquet")
    table["cell_id"] = table["cell_id"].astype(str)
    if table["cell_id"].duplicated().any():
        raise RuntimeError(f"cell table has duplicate cell_id: {sample}")
    table["coarse11"] = coarse11(table["cell_type"])
    return table


def load_polys(sample: str, kind: str) -> pd.DataFrame:
    path = RAW_POLYS / f"{sample}__{kind}_boundaries.parquet"
    poly = pd.read_parquet(path)
    poly["cell_id"] = poly["cell_id"].astype(str)

    poly = poly.sort_values("cell_id", kind="stable").reset_index(drop=True)
    return poly


def ring_segments(poly: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    cid = poly["cell_id"].to_numpy()
    lid = poly["label_id"].to_numpy()
    new = np.r_[True, (cid[1:] != cid[:-1]) | (lid[1:] != lid[:-1])]
    starts = np.flatnonzero(new)
    counts = np.diff(np.r_[starts, len(poly)])
    return starts, counts


def assert_rings_closed(poly: pd.DataFrame, starts: np.ndarray, counts: np.ndarray) -> float:
    vx = poly["vertex_x"].to_numpy()
    vy = poly["vertex_y"].to_numpy()
    ends = starts + counts - 1
    closed = (vx[starts] == vx[ends]) & (vy[starts] == vy[ends])
    return float(closed.mean())


def cell_first_row(poly: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    cid = poly["cell_id"].to_numpy()
    new = np.r_[True, cid[1:] != cid[:-1]]
    starts = np.flatnonzero(new)
    counts = np.diff(np.r_[starts, len(poly)])
    return starts, counts


def to_local(vertex_um: np.ndarray, centre_px: np.ndarray) -> np.ndarray:
    return vertex_um / PX_UM + HALF - np.rint(centre_px)


def to_local_aligned(vertex_um: np.ndarray, centre_px: np.ndarray, offset_um: np.ndarray) -> np.ndarray:
    return (vertex_um + offset_um) / PX_UM + HALF - np.rint(centre_px)


def ring_area_centroid(x: np.ndarray, y: np.ndarray) -> tuple[float, float, float]:
    cross = x[:-1] * y[1:] - x[1:] * y[:-1]
    area = 0.5 * float(cross.sum())
    if area == 0.0:
        return float(x[:-1].mean()), float(y[:-1].mean()), 0.0
    cx = float(((x[:-1] + x[1:]) * cross).sum() / (6.0 * area))
    cy = float(((y[:-1] + y[1:]) * cross).sum() / (6.0 * area))
    return cx, cy, area


def rasterize(rings: list[tuple[np.ndarray, np.ndarray]]) -> np.ndarray:
    canvas = np.zeros((CROP, CROP), np.uint8)
    for lx, ly in rings:
        pts = np.stack([lx, ly], axis=1) * FILL_SCALE
        pts = np.rint(pts).astype(np.int32).reshape(-1, 1, 2)
        one = np.zeros((CROP, CROP), np.uint8)
        cv2.fillPoly(one, [pts], 1, lineType=cv2.LINE_8, shift=FILL_SHIFT)
        np.maximum(canvas, one, out=canvas)
    return canvas


def patch_mean(image: np.ndarray) -> np.ndarray:
    return image.reshape(GRID, PATCH, GRID, PATCH).mean(axis=(1, 3))


def blur_and_pool(mask: np.ndarray, sigma: float) -> np.ndarray:
    ksize = int(2 * round(3.0 * sigma) + 1)
    blurred = cv2.GaussianBlur(mask.astype(np.float32), (ksize, ksize), sigma,
                               borderType=cv2.BORDER_CONSTANT)
    return patch_mean(blurred).astype(np.float32)


def load_he_zarr(he_path: Path):
    import tifffile
    import zarr
    store = tifffile.imread(str(he_path), aszarr=True, level=0)
    return zarr.open(store, mode="r")


def read_crop(zarr_image, x_px: float, y_px: float) -> np.ndarray:
    height, width = zarr_image.shape[0], zarr_image.shape[1]
    y0 = int(np.rint(y_px)) - HALF
    x0 = int(np.rint(x_px)) - HALF
    crop = np.full((CROP, CROP, 3), 255, np.uint8)
    sy0, sx0 = max(0, y0), max(0, x0)
    sy1, sx1 = min(height, y0 + CROP), min(width, x0 + CROP)
    if sy1 > sy0 and sx1 > sx0:
        crop[sy0 - y0:sy1 - y0, sx0 - x0:sx1 - x0] = np.asarray(zarr_image[sy0:sy1, sx0:sx1, :])
    return crop
