#!/usr/bin/env python3

from __future__ import annotations
import os
PROJECTS_ROOT = os.environ.get("PROJECTS_ROOT", "/data/heid")

import hashlib
import json
import shutil
from pathlib import Path

CELL_ROOT = Path(PROJECTS_ROOT + "/cell")
CONTRACT = CELL_ROOT / "outputs/manifests/contract.json"
BACKUP = CELL_ROOT / "outputs/manifests/contract.pre_he_safe_promotion.json"
AUDIT = CELL_ROOT / "outputs/manifests/adapter_leak_audit.json"


def sha256_list(items) -> str:
    return hashlib.sha256("\0".join(items).encode()).hexdigest()


def main() -> None:
    before = json.loads(CONTRACT.read_text())
    sha_before = before["cohort"]["allowlist_sha256"]
    recomputed_before = sha256_list(before["cohort"]["allowlist"])
    if recomputed_before != sha_before:
        raise RuntimeError(
            f"stored allowlist_sha256 {sha_before} does not reproduce from the stored allowlist "
            f"({recomputed_before}); refusing to edit"
        )
    if not BACKUP.exists():
        shutil.copy2(CONTRACT, BACKUP)

    prior = json.loads(AUDIT.read_text())["prior_versions_carrying_the_leak"] if AUDIT.exists() else None
    _p = prior or {}
    _g = _p.get("per_group", {})

    d = json.loads(CONTRACT.read_text())
    folds = d["folds"]
    policy_patient_only = folds["policy"]


    folds["canonical_fold_key"] = "fold_he_safe"
    folds["diagnostic_fold_key"] = "fold"
    folds["fold_key_semantics"] = {
        "fold_he_safe": {
            "role": "CANONICAL. Every model trained from this contract splits on this column.",
            "group_key": "connected component of patient union shared-H&E group",
            "policy": (
                "StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42), "
                "stratify=cancer, group=connected component of patient union shared-H&E group"
            ),
            "shared_he_groups_split_across_folds": (prior or {}).get("n_groups_split_in_v8_he_safe"),
        },
        "fold": {
            "role": (
                "DIAGNOSTIC ONLY. Retained for traceability and for comparison against the "
                "patient-grouped splits used by earlier packages. Must not be used to train or "
                "score anything, because it carries the shared-H&E leak below."
            ),
            "group_key": "patient",
            "policy": policy_patient_only,
            "leak": (
                "Two slides that share one physical H&E image (identical sha256 of the aligned-cell "
                "cells_for_training parquet) can belong to different patients, so grouping by "
                "patient alone puts the same image in a training fold and in the held-out fold."
            ),
            "shared_he_groups_split_across_folds": (prior or {}).get("n_groups_split_in_v8_patient_only"),
            "slides_on_both_sides": folds["shared_he_slides_split_across_folds_in_canonical"],
        },
    }


    folds["policy"] = folds["fold_key_semantics"]["fold_he_safe"]["policy"]
    folds["policy_of_diagnostic_fold_key"] = policy_patient_only


    folds["shared_he_slides_split_across_folds_in_diagnostic_key"] = list(
        folds["shared_he_slides_split_across_folds_in_canonical"]
    )
    folds["shared_he_slides_split_across_folds_in_canonical_DEPRECATED_NAME"] = (
        "this list belongs to the diagnostic `fold` column; see "
        "shared_he_slides_split_across_folds_in_diagnostic_key"
    )


    folds["shared_he_leak_history"] = {
        "statement": (
            "This is not a Cell-only bookkeeping note. Every earlier package that grouped folds by "
            "patient alone put at least one identical H&E image on both sides of its split, so its "
            "held-out numbers are optimistic by an unmeasured amount."
        ),
        "packages": {
            "a2 (pancancer_stainrobust_d1_v1)": {
                "split": "a1/data/split_outer5_inner5.json, group=patient, clean177",
                "n_shared_he_groups_split_across_folds": _p.get("n_groups_split_in_a2_v5"),
                "n_slides_on_both_sides": sum(
                    r["n_members_in_clean177"] for r in _g.values()
                    if r["a2_split_across_folds"]
                ),
            },
            "v5_a2clean": {
                "split": "same 177-section outer split as the feature-selection stage",
                "n_shared_he_groups_split_across_folds": _p.get("n_groups_split_in_a2_v5"),
                "n_slides_on_both_sides": sum(
                    r["n_members_in_clean177"] for r in _g.values()
                    if r["a2_split_across_folds"]
                ),
            },
            "inputs/cohort_reference": {
                "split": "inputs/cohort_reference/outputs/manifests/cohort_contract.json per_sample.fold, group=patient, 169 slides",
                "n_shared_he_groups_split_across_folds": _p.get("n_groups_split_in_v7"),
                "n_slides_on_both_sides": sum(
                    r["n_members_in_v7_169"] for r in _g.values()
                    if r["v7_split_across_folds"]
                ),
            },
        },
        "evidence": str(AUDIT),
    }
    if prior is None:
        folds.pop("shared_he_leak_history", None)


    d["cohort"]["allowlist_sha256"] = sha256_list(d["cohort"]["allowlist"])
    d["cohort"]["allowlist_sha256_unchanged_by_fold_promotion"] = (
        d["cohort"]["allowlist_sha256"] == sha_before
    )

    d["schema_version"] = "v8.contract_all5k.v2"
    d["revision"] = {
        "change": "canonical fold key switched from `fold` (group=patient) to `fold_he_safe`",
        "membership_changed": False,
        "folds_renumbered": False,
        "columns_removed": [],
        "fields_overwritten": {
            "folds.policy": "now describes fold_he_safe; prior value preserved verbatim at "
                            "folds.policy_of_diagnostic_fold_key",
        },
        "pre_edit_copy": str(BACKUP),
        "allowlist_sha256_before": sha_before,
        "allowlist_sha256_after": d["cohort"]["allowlist_sha256"],
    }

    tmp = CONTRACT.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(d, indent=1) + "\n")
    tmp.replace(CONTRACT)


    after = json.loads(CONTRACT.read_text())
    checks = {
        "allowlist_identical": after["cohort"]["allowlist"] == before["cohort"]["allowlist"],
        "allowlist_sha256_unchanged": after["cohort"]["allowlist_sha256"] == sha_before,
        "per_sample_rows_unchanged": after["per_sample"] == before["per_sample"],
        "he_safe_per_fold_unchanged": (
            after["folds"]["he_safe_alternative"]["per_fold"]
            == before["folds"]["he_safe_alternative"]["per_fold"]
        ),
        "diagnostic_per_fold_unchanged": after["folds"]["per_fold"] == before["folds"]["per_fold"],
        "canonical_key_is_he_safe": after["folds"]["canonical_fold_key"] == "fold_he_safe",
        "patient_only_policy_preserved_verbatim": (
            after["folds"]["policy_of_diagnostic_fold_key"] == before["folds"]["policy"]
        ),
        "shared_he_group_records_unchanged": (
            after["shared_he_slides"] == before["shared_he_slides"]
        ),
        "both_columns_present": all(
            ("fold" in r and "fold_he_safe" in r) for r in after["per_sample"]
        ),
    }
    print(json.dumps(checks, indent=1))
    print("allowlist_sha256 before:", sha_before)
    print("allowlist_sha256 after: ", after["cohort"]["allowlist_sha256"])
    if not all(checks.values()):
        raise RuntimeError("post-edit verification failed")


if __name__ == "__main__":
    main()
