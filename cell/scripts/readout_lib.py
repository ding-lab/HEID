#!/usr/bin/env python3

from __future__ import annotations
import os
PROJECTS_ROOT = os.environ.get("PROJECTS_ROOT", "/data/heid")

from pathlib import Path

import numpy as np
import pandas as pd

PC = Path(PROJECTS_ROOT + "/cell")
COHORT_REFERENCE_ROOT = PC / "inputs/cohort_reference"
RAW_POLYGONS = COHORT_REFERENCE_ROOT / "outputs/wmaps/raw_polys"
COHORT_REFERENCE_LABELS = COHORT_REFERENCE_ROOT / "outputs/labels"
CELL_TABLES = PC / "data/cell_tables"

PX_UM = 0.2125
CROP = 224
HALF = 112

TRAIN_COLUMNS = ["cell_id", "he_px", "he_py", "x_um", "y_um", "dx_um", "dy_um", "block"]


def target_local(centre_px: np.ndarray) -> np.ndarray:
    return HALF + (centre_px - np.rint(centre_px))


def place_absolute(vertex_um: np.ndarray, shift_um: np.ndarray,
                   centre_px: np.ndarray) -> np.ndarray:
    return (vertex_um + shift_um) / PX_UM + HALF - np.rint(centre_px)


def place_anchor(vertex_um: np.ndarray, anchor_um: np.ndarray,
                 centre_px: np.ndarray) -> np.ndarray:
    return (vertex_um - anchor_um) / PX_UM + HALF + (centre_px - np.rint(centre_px))


def place_old_canvas(vertex_um: np.ndarray, centre_px_old: np.ndarray) -> np.ndarray:
    return vertex_um / PX_UM + HALF - np.rint(centre_px_old)


def place_unshifted(vertex_um: np.ndarray, centre_px: np.ndarray) -> np.ndarray:
    return vertex_um / PX_UM + HALF - np.rint(centre_px)


def load_aligned_table(sample: str, index: dict) -> pd.DataFrame:
    table = pd.read_parquet(index[sample]["train_parquet"])
    if list(table.columns) != TRAIN_COLUMNS:
        raise RuntimeError(f"unexpected aligned-cell training schema for {sample}: {list(table.columns)}")
    table["cell_id"] = table["cell_id"].astype(str)
    if table["cell_id"].duplicated().any():
        raise RuntimeError(f"duplicate cell_id in the aligned-cell training table: {sample}")
    residual = max(
        float(np.abs((table["x_um"] + table["dx_um"]) / PX_UM - table["he_px"]).max()),
        float(np.abs((table["y_um"] + table["dy_um"]) / PX_UM - table["he_py"]).max()),
    )
    if residual != 0.0:
        raise RuntimeError(f"he_px/he_py is not (u_um + d_um)/{PX_UM} for {sample}: {residual}")
    return table


def load_polys(sample: str, kind: str, keep_ids: set[str]) -> pd.DataFrame:
    path = RAW_POLYGONS / f"{sample}__{kind}_boundaries.parquet"
    poly = pd.read_parquet(path)
    poly["cell_id"] = poly["cell_id"].astype(str)
    poly = poly[poly["cell_id"].isin(keep_ids)]

    poly = poly.sort_values(["cell_id", "label_id"], kind="stable").reset_index(drop=True)
    return poly


class Rings:

    def __init__(self, poly: pd.DataFrame):
        cid = poly["cell_id"].to_numpy()
        lid = poly["label_id"].to_numpy()
        self.x = poly["vertex_x"].to_numpy(np.float64)
        self.y = poly["vertex_y"].to_numpy(np.float64)

        new_ring = np.r_[True, (cid[1:] != cid[:-1]) | (lid[1:] != lid[:-1])]
        self.ring_start = np.flatnonzero(new_ring)
        self.ring_count = np.diff(np.r_[self.ring_start, len(poly)])
        if (self.ring_count < 4).any():
            raise RuntimeError("a ring has fewer than four stored vertices")

        ends = self.ring_start + self.ring_count - 1
        self.ring_closed = (self.x[self.ring_start] == self.x[ends]) & \
                           (self.y[self.ring_start] == self.y[ends])

        new_cell = np.r_[True, cid[1:] != cid[:-1]]
        self.cell_start = np.flatnonzero(new_cell)
        self.cell_count = np.diff(np.r_[self.cell_start, len(poly)])
        self.cell_id = cid[self.cell_start]
        self.n_cells = len(self.cell_start)
        self.rings_per_cell = np.add.reduceat(new_ring.astype(np.int64), self.cell_start)
        self.cell_of_ring = np.repeat(np.arange(self.n_cells), self.rings_per_cell)


        keep = np.ones(len(poly), bool)
        keep[ends] = False
        self.head = np.flatnonzero(keep)
        self.tail = self.head + 1
        self.open_count = self.ring_count - 1
        self.open_start = np.r_[0, np.cumsum(self.open_count)[:-1]]
        self.row_of_cell = np.repeat(np.arange(self.n_cells), self.cell_count)

    def ring_area_and_centroid(self, lx: np.ndarray, ly: np.ndarray):
        xh, yh = lx[self.head], ly[self.head]
        xt, yt = lx[self.tail], ly[self.tail]
        cross = xh * yt - xt * yh
        area = 0.5 * np.add.reduceat(cross, self.open_start)
        sx = np.add.reduceat((xh + xt) * cross, self.open_start)
        sy = np.add.reduceat((yh + yt) * cross, self.open_start)
        with np.errstate(divide="ignore", invalid="ignore"):
            cx = sx / (6.0 * area)
            cy = sy / (6.0 * area)
        degenerate = area == 0.0
        if degenerate.any():
            mx = np.add.reduceat(xh, self.open_start) / self.open_count
            my = np.add.reduceat(yh, self.open_start) / self.open_count
            cx = np.where(degenerate, mx, cx)
            cy = np.where(degenerate, my, cy)
        return area, cx, cy

    def cell_area_centroid(self, lx: np.ndarray, ly: np.ndarray):
        area, cx, cy = self.ring_area_and_centroid(lx, ly)
        w = np.abs(area)
        starts = np.r_[0, np.cumsum(self.rings_per_cell)[:-1]]
        num_x = np.add.reduceat(cx * w, starts)
        num_y = np.add.reduceat(cy * w, starts)
        den = np.add.reduceat(w, starts)
        vmx, vmy = self.cell_vertex_mean(lx, ly)
        bad = den == 0.0
        with np.errstate(divide="ignore", invalid="ignore"):
            ax = np.where(bad, vmx, num_x / den)
            ay = np.where(bad, vmy, num_y / den)
        return ax, ay, den

    def cell_vertex_mean(self, lx: np.ndarray, ly: np.ndarray):
        starts = np.r_[0, np.cumsum(self.rings_per_cell)[:-1]]
        ring_sum_x = np.add.reduceat(lx[self.head], self.open_start)
        ring_sum_y = np.add.reduceat(ly[self.head], self.open_start)
        sx = np.add.reduceat(ring_sum_x, starts)
        sy = np.add.reduceat(ring_sum_y, starts)
        n = np.add.reduceat(self.open_count, starts)
        return sx / n, sy / n

    def cell_vertex_mean_with_duplicates(self, lx: np.ndarray, ly: np.ndarray):
        sx = np.add.reduceat(lx, self.cell_start)
        sy = np.add.reduceat(ly, self.cell_start)
        return sx / self.cell_count, sy / self.cell_count

    def contains(self, lx: np.ndarray, ly: np.ndarray,
                 px: np.ndarray, py: np.ndarray) -> np.ndarray:
        ring_px = px[self.cell_of_ring]
        ring_py = py[self.cell_of_ring]
        edge_px = np.repeat(ring_px, self.open_count)
        edge_py = np.repeat(ring_py, self.open_count)
        y0, y1 = ly[self.head], ly[self.tail]
        x0, x1 = lx[self.head], lx[self.tail]
        straddles = (y0 > edge_py) != (y1 > edge_py)
        with np.errstate(divide="ignore", invalid="ignore"):
            xcross = (x1 - x0) * (edge_py - y0) / (y1 - y0) + x0
        hit = straddles & (edge_px < xcross)
        per_ring = np.add.reduceat(hit.astype(np.int64), self.open_start)
        starts = np.r_[0, np.cumsum(self.rings_per_cell)[:-1]]
        return np.add.reduceat(per_ring, starts) % 2 == 1


def summarise(dev: np.ndarray) -> dict:
    return {
        "n_cells": int(len(dev)),
        "p50": round(float(np.percentile(dev, 50)), 6),
        "p90": round(float(np.percentile(dev, 90)), 6),
        "p99": round(float(np.percentile(dev, 99)), 6),
        "max": round(float(dev.max()), 6),
        "frac_within_0p01px": round(float((dev <= 0.01).mean()), 6),
        "frac_within_0p05px": round(float((dev <= 0.05).mean()), 6),
        "frac_within_1px": round(float((dev <= 1.0).mean()), 6),
        "frac_within_3px": round(float((dev <= 3.0).mean()), 6),
    }
