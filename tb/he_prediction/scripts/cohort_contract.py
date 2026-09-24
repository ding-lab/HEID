
from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from experiment_config import REFERENCE, cohort_size

REFERENCE_MANIFEST = "source_cohort_reference.csv"
REQUIRED_COLUMNS = frozenset({"sample", "patient", "cancer", "fold"})


def validate_working_cohort(
    manifest: pd.DataFrame, manifest_path: Path | str
) -> dict[str, Any]:

    missing = REQUIRED_COLUMNS.difference(manifest.columns)
    if missing:
        raise ValueError(f"source manifest missing columns {sorted(missing)}")
    if len(manifest) != manifest["sample"].nunique():
        raise ValueError("source manifest contains duplicate samples")
    if set(manifest["fold"].astype(int)) != set(range(5)):
        raise ValueError("source manifest does not populate all five folds")

    straddling = (
        manifest.groupby("patient")["fold"].nunique().pipe(lambda s: s[s > 1]).index.tolist()
    )
    if straddling:
        raise ValueError(f"patients appear in more than one fold: {sorted(straddling)}")

    reference_path = Path(manifest_path).parent / REFERENCE_MANIFEST
    if not reference_path.is_file():
        raise ValueError(f"missing the reference provenance manifest: {reference_path}")
    reference = pd.read_csv(reference_path, keep_default_na=False)
    expected = cohort_size(REFERENCE)["n_samples"]
    if len(reference) != expected or reference["sample"].nunique() != expected:
        raise ValueError(f"reference provenance manifest is not {expected} unique samples")

    reference_rows = {str(row["sample"]): row for row in reference.to_dict("records")}
    outside = sorted(set(manifest["sample"].astype(str)) - set(reference_rows))
    if outside:
        raise ValueError(f"working cohort contains samples outside the reference provenance set: {outside}")
    edited = sorted(
        sample
        for row in manifest.to_dict("records")
        if (sample := str(row["sample"])) and row != reference_rows[sample]
    )
    if edited:
        raise ValueError(f"working cohort rows differ from reference provenance: {edited}")

    return {
        "n_samples": int(len(manifest)),
        "n_patients": int(manifest["patient"].nunique()),
        "n_cancers": int(manifest["cancer"].nunique()),
        "excluded_from_clean177": sorted(
            set(reference_rows) - set(manifest["sample"].astype(str))
        ),
    }


__all__ = ["validate_working_cohort", "REFERENCE_MANIFEST"]
