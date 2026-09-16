#!/usr/bin/env python3

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.compute as pac
import pyarrow.parquet as pq

FULL_COLUMNS = ["cell_id", "vertex_x", "vertex_y", "label_id"]
LEGACY_COLUMNS = ["cell_id", "vertex_x", "vertex_y"]


def boundary_schema(src: Path | str) -> list[str]:
    return [f.name for f in pq.ParquetFile(str(src)).schema_arrow]


def has_label_id(src: Path | str) -> bool:
    return "label_id" in boundary_schema(src)


def load_polys(src: Path | str, keep: set[str] | None = None) -> pd.DataFrame:
    src = Path(src)
    handle = pq.ParquetFile(str(src))
    schema = [f.name for f in handle.schema_arrow]
    label_present = "label_id" in schema
    columns = FULL_COLUMNS if label_present else LEGACY_COLUMNS
    if schema != columns:
        raise RuntimeError(
            f"unexpected boundary schema in {src}: {schema}; expected {FULL_COLUMNS} "
            f"or {LEGACY_COLUMNS}")

    pieces = []
    if keep is None:
        for group in range(handle.num_row_groups):
            pieces.append(handle.read_row_group(group, columns=columns))
    else:
        wanted = pa.array(sorted(keep))
        for group in range(handle.num_row_groups):
            tbl = handle.read_row_group(group, columns=columns)
            tbl = tbl.filter(pac.is_in(tbl["cell_id"], value_set=wanted))
            if tbl.num_rows:
                pieces.append(tbl)
    if not pieces:
        raise RuntimeError(f"no boundary row matches the requested cell_id set: {src}")

    poly = pa.concat_tables(pieces).to_pandas()
    poly["cell_id"] = poly["cell_id"].astype(str)
    if not label_present:
        poly["label_id"] = np.int32(0)
    poly = poly.sort_values(["cell_id", "label_id"], kind="stable").reset_index(drop=True)
    poly.attrs["has_label_id"] = label_present
    poly.attrs["n_rows_file"] = int(handle.metadata.num_rows)
    poly.attrs["schema"] = schema
    return poly
