#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import pandas as pd

import contract as C

A4_CONTRACT = C.PC / "inputs/necrosis_annotation_contract.json"
OUT = C.MANIFEST_ROOT / "data_contract.json"
NECROSIS_LABEL = "Necrosis"


def build() -> dict[str, Any]:
    contract = C.load_contract()
    allow = sorted(C.allowlist())
    rows = C.records()
    membership = C.fold_membership()

    a4 = json.loads(A4_CONTRACT.read_text(encoding="utf-8"))
    a4_boundary = a4["necrosis_annotation_boundary"]
    rule = a4_boundary["observed_sample_rule"]
    if "at least one source-label Necrosis cell" not in rule:
        raise RuntimeError(f"the necrosis coverage rule differs from the expected rule: {rule!r}")

    records = []
    observed: list[str] = []
    unobserved: list[str] = []
    n_observed_positive_cells = 0
    n_observed_negative_cells = 0
    for sample in allow:
        rec = rows[sample]
        counts = (
            pd.read_parquet(C.labels_path(sample), columns=["cell_type_refine"])[
                "cell_type_refine"
            ]
            .astype(str)
            .value_counts()
            .to_dict()
        )
        n_necrosis = int(counts.get(NECROSIS_LABEL, 0))
        n_cells = int(rec["n_keep_cells_own"])
        if sum(int(v) for v in counts.values()) < n_cells:
            raise RuntimeError(f"{sample}: label table holds fewer rows than the joined cell set")
        is_observed = n_necrosis > 0
        (observed if is_observed else unobserved).append(sample)
        if is_observed:
            n_observed_positive_cells += n_necrosis
            n_observed_negative_cells += n_cells - n_necrosis
        records.append({
            "sample": sample,
            "patient": str(rec["patient"]),
            "cancer": str(rec["cancer"]),
            "cohort": str(rec["cohort"]),
            "fold": int(rec[C.CANONICAL_FOLD_KEY]),
            "n_cells": n_cells,
            "necrosis_annotation_observed": bool(is_observed),
            "n_source_label_necrosis_cells": n_necrosis,
        })

    patients = sorted({r["patient"] for r in records})
    cancers = sorted({r["cancer"] for r in records})
    if len(observed) + len(unobserved) != len(allow):
        raise RuntimeError("necrosis observed/unobserved lists do not cover the cohort")

    fold_records = []
    for fold in range(5):
        samples = sorted(membership[fold]["test_samples"])
        fold_patients = sorted({rows[s]["patient"] for s in samples})
        by_cancer: dict[str, int] = {}
        for name in samples:
            key = str(rows[name]["cancer"])
            by_cancer[key] = by_cancer.get(key, 0) + 1
        fold_records.append({
            "fold": fold,
            "n_samples": len(samples),
            "n_patients": len(fold_patients),
            "patients": fold_patients,
            "samples": samples,
            "by_cancer": dict(sorted(by_cancer.items())),
        })
    if sum(r["n_samples"] for r in fold_records) != len(allow):
        raise RuntimeError("fold records do not partition the Cell allowlist")
    for record in fold_records:
        for name in record["samples"]:
            if int(rows[name][C.CANONICAL_FOLD_KEY]) != record["fold"]:
                raise RuntimeError(f"{name}: fold record / sample record disagreement")

    total_cells = sum(r["n_cells"] for r in records)
    if total_cells != int(contract["labels"]["n_join_cells_total"]):
        raise RuntimeError(
            f"cell total {total_cells} differs from contract labels.n_join_cells_total "
            f"{contract['labels']['n_join_cells_total']}"
        )

    boundary = {
        **{k: v for k, v in a4_boundary.items()
           if k not in ("annotation_observed_samples", "annotation_unobserved_samples",
                        "n_annotation_observed_samples", "n_annotation_unobserved_samples",
                        "n_observed_positive_cells", "n_observed_negative_cells",
                        "cancers_with_observed_positives", "cancers_with_zero_observed_positives")},
        "annotation_observed_samples": observed,
        "annotation_unobserved_samples": unobserved,
        "n_annotation_observed_samples": len(observed),
        "n_annotation_unobserved_samples": len(unobserved),
        "n_observed_positive_cells": n_observed_positive_cells,
        "n_observed_negative_cells": n_observed_negative_cells,
        "cancers_with_observed_positives": sorted(
            {r["cancer"] for r in records if r["necrosis_annotation_observed"]}
        ),
        "cancers_with_zero_observed_positives": sorted(
            {r["cancer"] for r in records}
            - {r["cancer"] for r in records if r["necrosis_annotation_observed"]}
        ),
        "recomputed_from": "molecular cell_type_refine labels, necrosis annotation rule applied verbatim",
    }

    payload = {
        "schema_version": a4["schema_version"],
        "status": "built_for_v8",
        "scope": {
            "allowed_inputs": [
                "cell/configs/contract.json",
                "${PROJECTS_ROOT}/cell/outputs/labels/<sample>.parquet",
                "necrosis annotation contract (rule text only)",
            ],
            "explicitly_not_consumed": [
                "a1/data/cohort_clean177.csv",
                "the a2 input-generation registry",
                "features", "predictions", "checkpoints",
            ],
        },
        "provenance": {
            "derivation": (
                "built from the cohort contract and the molecular labels; the necrosis-annotation "
                "records cover 164 of the 202 slides, so they were not used as a filter"
            ),
            "canonical_fold_key": C.CANONICAL_FOLD_KEY,
            "contract_v8_path": str(C.CONTRACT),
            "contract_v8_sha256": C.contract_sha256(),
            "a4_contract_path": str(A4_CONTRACT),
            "a4_contract_sha256": C.sha256_file(A4_CONTRACT),
            "necrosis_rule_source": rule,
            "builder_sha256": C.sha256_file(Path(__file__)),
        },
        "cohort": {
            "sample_records": records,
            "included_samples": len(allow),
            "included_patients": len(patients),
            "included_cancers": len(cancers),
            "included_cells": total_cells,
            "samples_by_cancer": {
                cancer: sorted(r["sample"] for r in records if r["cancer"] == cancer)
                for cancer in cancers
            },
        },
        "folds": {
            "patient_disjoint": True,
            "group_key": "connected component of patient union shared-H&E group",
            "source_contract": str(C.CONTRACT),
            "records": fold_records,
        },
        "necrosis_annotation_boundary": boundary,
        "taxonomy": {k: v for k, v in (a4.get("taxonomy") or {}).items()
                     if k not in ("source_label_counts", "identity_supervised_counts", "n_observed_source_labels")},
        "coordinate_and_mpp_contract": {
            "cell_coordinate_space": "aligned_he level-0 pixel coordinates (he_px / he_py)",
            "mpp_um_per_px": C.PX_UM,
            "bounds_rule": "he_px = (x_um + dx_um) / 0.2125, asserted exactly per slide",
        },
        "cell_tables": {
            "source": "registered cell training table INNER JOIN molecular labels",
            "required_columns": C.TRAIN_COLUMNS,
            "n_files": len(allow),
            "n_rows": total_cells,
            "cell_id_unique_within_sample": True,
        },
    }
    return payload


def main() -> int:
    global A4_CONTRACT
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--necrosis-contract", type=Path, default=A4_CONTRACT,
                        help="necrosis-annotation contract whose necrosis_annotation_boundary is carried over")
    args = parser.parse_args()
    A4_CONTRACT = args.necrosis_contract
    payload = build()
    summary = {
        "path": str(OUT),
        "n_samples": payload["cohort"]["included_samples"],
        "n_patients": payload["cohort"]["included_patients"],
        "n_cancers": payload["cohort"]["included_cancers"],
        "n_cells": payload["cohort"]["included_cells"],
        "per_fold_samples": {
            str(r["fold"]): r["n_samples"] for r in payload["folds"]["records"]
        },
        "n_necrosis_observed": payload["necrosis_annotation_boundary"][
            "n_annotation_observed_samples"
        ],
        "n_necrosis_unobserved": payload["necrosis_annotation_boundary"][
            "n_annotation_unobserved_samples"
        ],
    }
    if not args.dry_run:
        C.atomic_json(OUT, payload)
        summary["sha256"] = C.sha256_file(OUT)
    print(json.dumps(summary, indent=2, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
