
from __future__ import annotations
import os
PROJECTS_ROOT = os.environ.get("PROJECTS_ROOT", "/data/heid")

from pathlib import Path
from typing import Any

import pandas as pd

ALIGNED_HE_ROOT = Path(
    PROJECTS_ROOT + "/align/aligned_he"
)


def corrected_cell_table(sample: str, cancer: str, cohort: str = "5k") -> Path:

    base = ALIGNED_HE_ROOT / cohort / cancer
    for name in (sample, f"{sample}-weak", f"{sample}-lowqc", f"{sample}-bad-lowqc"):
        candidate = base / name / "micro" / f"{sample}_cells_for_training.parquet"
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(f"{sample}: no corrected cell table under {base}")


def load_joined_cells(
    row: pd.Series, verify: bool = False
) -> tuple[pd.DataFrame, dict[str, int]]:

    sample = str(row["sample"])
    coordinate_path = corrected_cell_table(sample, str(row["cancer"])).resolve()
    manifest_path = Path(row["cell_table_path"]).resolve()
    if coordinate_path != manifest_path:
        raise ValueError(
            f"{sample}: resolved corrected coordinate table {coordinate_path} "
            f"differs from manifest cell_table_path {manifest_path}"
        )
    cells = pd.read_parquet(
        coordinate_path, columns=["cell_id", "he_px", "he_py"]
    )
    cells["cell_id"] = cells["cell_id"].astype(str)


    cells = cells.rename(columns={"he_px": "x_px", "he_py": "y_px"})

    reference_columns = ["cell_id", "cell_type"]
    if verify:
        reference_columns.extend(["side", "dist_sdf_um"])
    reference = pd.read_csv(
        Path(row["boundary_reference_csv"]),
        usecols=reference_columns,
        dtype={"cell_id": str, "cell_type": str},
        keep_default_na=False,
    )

    if cells["cell_id"].duplicated().any():
        raise ValueError(f"{sample}: duplicate cell IDs in the corrected coordinate table")
    if reference["cell_id"].duplicated().any():
        raise ValueError(f"{sample}: duplicate cell IDs in the molecular reference")


    joined = cells.merge(
        reference, on="cell_id", how="inner", validate="one_to_one", sort=False
    )
    if joined.empty:
        raise ValueError(f"{sample}: corrected coordinates and molecular reference share no cells")

    audit = {
        "v9_cells": int(len(cells)),
        "gen11_cells": int(len(reference)),
        "joined_cells": int(len(joined)),
        "gen11_cells_without_v9_coordinate": int(len(reference) - len(joined)),
        "v9_cells_without_gen11_type": int(len(cells) - len(joined)),
    }
    joined.attrs["he_mpp_um"] = float(row["he_mpp_um"])
    joined.attrs["coordinate_source"] = "align_v9_per_block_corrected"
    joined.attrs["cell_audit"] = audit
    return joined, audit


__all__ = ["load_joined_cells", "corrected_cell_table", "ALIGNED_HE_ROOT"]
