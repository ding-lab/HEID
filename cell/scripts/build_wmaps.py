#!/usr/bin/env python3

from __future__ import annotations
import os
PROJECTS_ROOT = os.environ.get("PROJECTS_ROOT", "/data/heid")
from pathlib import Path
_RELEASE = Path(__file__).resolve().parents[2]

import argparse
import hashlib
import json
import os
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.compute as pac
import pyarrow.parquet as pq

PC = Path(PROJECTS_ROOT + "/cell")
COHORT_REFERENCE_ROOT = PC / "inputs/cohort_reference"
CELL_ROOT = PC
sys.path.insert(0, str(_RELEASE / "cell/scripts"))
import polygon_lib as R
import readout_lib as L
from paths import expand_env as _expand_env

XENIUM_INDEX = _RELEASE / "cell/configs/xenium_5k_index.json"
CONTRACT = _RELEASE / "cell/configs/contract.json"
LABELS = CELL_ROOT / "outputs/labels"
OUT_ROOT = CELL_ROOT / "outputs/wmaps/cell"
MANIFEST_DIR = CELL_ROOT / "outputs/manifests/wmaps"

SIGMAS = {"sigma10": 10.0, "sigma3": 3.0}
QUANT_SCALE = 1.0 / 255.0
POLY_COLUMNS = ["cell_id", "vertex_x", "vertex_y", "label_id"]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 22), b""):
            digest.update(block)
    return digest.hexdigest()


def ordered_sha256(values: list[str]) -> str:
    digest = hashlib.sha256(b"v7.v9_cell_order.v1\0")
    digest.update(len(values).to_bytes(8, "little"))
    for value in values:
        encoded = value.encode("utf-8")
        digest.update(len(encoded).to_bytes(8, "little"))
        digest.update(encoded)
    return digest.hexdigest()


def load_cells(sample: str, record: dict) -> tuple[pd.DataFrame, str]:
    table_path = Path(record["train_parquet"])
    table_sha = sha256_file(table_path)
    table = pd.read_parquet(table_path)
    if list(table.columns) != L.TRAIN_COLUMNS:
        raise RuntimeError(f"unexpected aligned-cell training schema: {list(table.columns)}")
    table["cell_id"] = table["cell_id"].astype(str)
    if table["cell_id"].duplicated().any():
        raise RuntimeError(f"duplicate cell_id in the aligned-cell training table: {sample}")
    if len(table) != int(record["n_v9_training_rows_in_table"]):
        raise RuntimeError(
            f"{sample}: aligned-cell table has {len(table)} rows, contract says "
            f"{record['n_v9_training_rows_in_table']}")
    residual = max(
        float(np.abs((table["x_um"] + table["dx_um"]) / L.PX_UM - table["he_px"]).max()),
        float(np.abs((table["y_um"] + table["dy_um"]) / L.PX_UM - table["he_py"]).max()),
    )
    if residual != 0.0:
        raise RuntimeError(f"he_px/he_py is not (u_um + d_um)/{L.PX_UM}: residual {residual}")

    labels = pd.read_parquet(LABELS / f"{sample}.parquet")
    labels["cell_id"] = labels["cell_id"].astype(str)
    if labels["cell_id"].duplicated().any():
        raise RuntimeError(f"duplicate cell_id in labels_v8: {sample}")
    cells = table.merge(labels, on="cell_id", how="inner")
    if len(cells) != int(record["n_keep_cells_own"]):
        raise RuntimeError(
            f"{sample}: {len(cells)} labelled aligned-cell cells, contract says "
            f"{record['n_keep_cells_own']}")
    if cells["coarse11"].isna().any():
        raise RuntimeError(f"{sample}: a labelled cell has no COARSE11 identity")
    return cells.sort_values("cell_id", kind="stable").reset_index(drop=True), table_sha


def load_polys(src: Path, keep: set[str]) -> pd.DataFrame:
    handle = pq.ParquetFile(src)
    schema = [field.name for field in handle.schema_arrow]
    if schema != POLY_COLUMNS:
        raise RuntimeError(f"unexpected boundary schema in {src}: {schema}")
    wanted = pa.array(sorted(keep))
    n_rows_file = handle.metadata.num_rows
    pieces = []
    for group in range(handle.num_row_groups):
        table = handle.read_row_group(group, columns=POLY_COLUMNS)
        table = table.filter(pac.is_in(table["cell_id"], value_set=wanted))
        if table.num_rows:
            pieces.append(table)
    if not pieces:
        raise RuntimeError(f"no boundary row matches this sample's cell_id set: {src}")
    poly = pa.concat_tables(pieces).to_pandas()
    poly["cell_id"] = poly["cell_id"].astype(str)
    poly = poly.sort_values(["cell_id", "label_id"], kind="stable").reset_index(drop=True)
    poly.attrs["n_rows_file"] = int(n_rows_file)
    return poly


def build(sample: str) -> dict:
    started = time.time()
    contract = _expand_env(json.loads(CONTRACT.read_text()))
    record = {r["sample"]: r for r in contract["per_sample"]}[sample]

    cells, table_sha = load_cells(sample, record)
    cids = cells["cell_id"].to_numpy().astype(str)
    n_cells = len(cids)
    index_of = {cid: i for i, cid in enumerate(cids.tolist())}

    index = _expand_env(json.loads(XENIUM_INDEX.read_text()))
    if sample not in index:
        raise RuntimeError(f"sample absent from the Xenium index: {sample}")
    xen_dir = Path(index[sample]["path"])
    src = xen_dir / "cell_boundaries.parquet"
    if not src.is_file():
        raise FileNotFoundError(f"unreachable boundary file (pyxis not stripped?): {src}")

    poly = load_polys(src, set(cids.tolist()))
    n_rows_file = int(poly.attrs["n_rows_file"])
    merged = poly.merge(
        cells[["cell_id", "he_px", "he_py", "dx_um", "dy_um"]], on="cell_id", how="inner")
    if len(merged) != len(poly):
        raise RuntimeError("polygon rows lost or duplicated by the cell join")
    merged = merged.sort_values(["cell_id", "label_id"], kind="stable").reset_index(drop=True)

    lx = L.place_absolute(merged["vertex_x"].to_numpy(np.float64),
                          merged["dx_um"].to_numpy(np.float64),
                          merged["he_px"].to_numpy(np.float64))
    ly = L.place_absolute(merged["vertex_y"].to_numpy(np.float64),
                          merged["dy_um"].to_numpy(np.float64),
                          merged["he_py"].to_numpy(np.float64))
    ring_start, ring_count = R.ring_segments(merged)
    frac_closed = R.assert_rings_closed(merged, ring_start, ring_count)
    if frac_closed < 0.999:
        raise RuntimeError(f"only {frac_closed:.6f} of rings are closed")
    cell_start, cell_count = R.cell_first_row(merged)
    poly_cell_ids = merged["cell_id"].to_numpy()[cell_start]
    n_poly_cells = len(cell_start)
    cell_of_row = np.repeat(np.arange(n_poly_cells), cell_count)
    rings_by_cell: dict[int, list[int]] = defaultdict(list)
    for ring, start in enumerate(ring_start):
        rings_by_cell[index_of[str(poly_cell_ids[cell_of_row[start]])]].append(ring)

    maps = {name: np.zeros((n_cells, R.GRID, R.GRID), np.uint8) for name in SIGMAS}
    n_patch_ge50 = np.zeros(n_cells, np.int16)
    area_px = np.zeros(n_cells, np.int32)
    has_poly = np.zeros(n_cells, bool)

    for cell_index, ring_ids in rings_by_cell.items():
        pieces = []
        for ring in ring_ids:
            start, count = int(ring_start[ring]), int(ring_count[ring])
            pieces.append((lx[start:start + count], ly[start:start + count]))
        mask = R.rasterize(pieces)
        area_px[cell_index] = int(mask.sum())
        has_poly[cell_index] = True
        n_patch_ge50[cell_index] = int((R.patch_mean(mask.astype(np.float32)) >= 0.5).sum())
        for name, sigma in SIGMAS.items():
            pooled = R.blur_and_pool(mask, sigma)
            maps[name][cell_index] = np.rint(
                np.clip(pooled, 0.0, 1.0) / QUANT_SCALE).astype(np.uint8)

    n_with_poly = int(has_poly.sum())
    if n_with_poly < 1:
        raise RuntimeError("not a single kept cell has a polygon")
    for name in SIGMAS:
        empty = maps[name].reshape(n_cells, -1).sum(axis=1, dtype=np.int64) == 0
        if int((empty & has_poly).sum()) != int((area_px[has_poly] == 0).sum()):
            raise RuntimeError(f"{name}: an all-zero map does not correspond to an empty raster")

    order_digest = ordered_sha256(cids.tolist())
    written = {}
    for name in SIGMAS:
        out_dir = OUT_ROOT / name
        out_dir.mkdir(parents=True, exist_ok=True)
        tmp = out_dir / f"{sample}.staging.{os.getpid()}.npz"
        np.savez_compressed(tmp, cell_id=cids.astype("U32"), w16=maps[name],
                            n_patch_ge50=n_patch_ge50, area_px=area_px, has_poly=has_poly)
        final = out_dir / f"{sample}.npz"
        tmp.replace(final)
        written[name] = {"path": str(final), "sha256": sha256_file(final),
                         "size_bytes": final.stat().st_size, "sigma_px": SIGMAS[name]}

    per_class = {}
    for name in sorted(cells["coarse11"].unique().tolist()):
        pick = (cells["coarse11"] == name).to_numpy()
        data = n_patch_ge50[pick]
        per_class[name] = {
            "n_cells": int(pick.sum()),
            "n_without_polygon": int((~has_poly[pick]).sum()),
            "p10": int(np.percentile(data, 10)),
            "p50": int(np.percentile(data, 50)),
            "p90": int(np.percentile(data, 90)),
            "frac_zero_patch": round(float((data == 0).mean()), 6),
            "median_area_um2": round(float(np.median(area_px[pick]) * L.PX_UM ** 2), 3),
        }

    manifest = {
        "schema_version": "v8.wmaps.v1",
        "sample": sample,
        "cancer": record["cancer"],
        "n_cells": n_cells,
        "n_cells_with_polygon": n_with_poly,
        "frac_cells_with_polygon": round(n_with_poly / n_cells, 6),
        "cell_id_ordered_sha256": order_digest,
        "boundary_source": str(src),
        "n_boundary_rows_in_file": n_rows_file,
        "n_boundary_rows_kept": int(len(poly)),
        "v9_training_table_sha256": table_sha,
        "labels_parquet": {"path": str(LABELS / f"{sample}.parquet"),
                           "sha256": sha256_file(LABELS / f"{sample}.parquet")},
        "contract_sha256": sha256_file(CONTRACT),
        "outputs": written,
        "per_coarse11": per_class,
        "wall_sec": round(time.time() - started, 1),
        "job_id": os.environ.get("SLURM_JOB_ID"),
        "array_task_id": os.environ.get("SLURM_ARRAY_TASK_ID"),
        "status": "PASS",
    }
    MANIFEST_DIR.mkdir(parents=True, exist_ok=True)
    out = MANIFEST_DIR / f"{sample}.json"
    tmp = out.with_name(out.name + f".tmp.{os.getpid()}")
    tmp.write_text(json.dumps(manifest, indent=2) + "\n")
    tmp.replace(out)
    print(f"[{sample}] {n_cells:,} cells, polygon coverage "
          f"{manifest['frac_cells_with_polygon']:.6f}, {manifest['wall_sec']}s", flush=True)
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sample")
    parser.add_argument("--task-id", type=int)
    parser.add_argument("--todo", default=str(CELL_ROOT / "outputs/manifests/wmaps_todo.json"))
    args = parser.parse_args()

    if args.sample:
        sample = args.sample
    else:
        todo = json.loads(Path(args.todo).read_text())["samples"]
        if args.task_id is None or not 0 <= args.task_id < len(todo):
            raise ValueError(f"task id outside 0..{len(todo) - 1}")
        sample = todo[args.task_id]

    done = MANIFEST_DIR / f"{sample}.json"
    if done.is_file() and json.loads(done.read_text()).get("status") == "PASS":
        print(f"[{sample}] already PASS, skipping", flush=True)
        return 0
    build(sample)
    return 0


if __name__ == "__main__":
    sys.exit(main())
