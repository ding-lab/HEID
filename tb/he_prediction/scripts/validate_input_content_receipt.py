#!/usr/bin/env python3

from __future__ import annotations
import os
PROJECTS_ROOT = os.environ.get("PROJECTS_ROOT", "/data/heid")
import os

import argparse
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import pandas as pd

import audit_input_content as audit


SCHEMA = "he_boundary_prediction.v1.r4_input_alias_validation.v1"
PDC_PROJECT_PREFIXES = tuple(
    Path(p) for p in os.environ.get("PROJECT_PATH_ALIASES", f"{PROJECTS_ROOT}:{Path(__file__).resolve().parents[3]}").split(":")
)


def path_identity(path: Path | str) -> str:
    candidate = Path(path)
    if not candidate.is_absolute():
        raise ValueError(f"path is not absolute: {path}")
    if ".." in candidate.parts:
        raise ValueError(f"path contains parent traversal: {path}")
    for prefix in PDC_PROJECT_PREFIXES:
        try:
            relative = candidate.relative_to(prefix)
        except ValueError:
            continue
        return relative.as_posix()
    raise ValueError(f"path uses an unapproved PDC storage root: {path}")


def normalized_artifact(path: Path) -> dict[str, Any]:
    current = audit.artifact(path)
    return {
        "path_identity": path_identity(current["path"]),
        "sha256": current["sha256"],
        "size_bytes": int(current["size_bytes"]),
    }


def normalize_recorded_artifact(
    recorded: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        "path_identity": path_identity(str(recorded.get("path", ""))),
        "sha256": str(recorded.get("sha256", "")),
        "size_bytes": int(recorded.get("size_bytes", -1)),
    }


def normalize_content_entry(entry: Mapping[str, Any]) -> dict[str, Any]:
    normalized = dict(entry)
    normalized["path_identity"] = path_identity(
        str(normalized.pop("path", ""))
    )
    return normalized


def portable_content_tree(
    entries: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    normalized = sorted(
        [normalize_content_entry(entry) for entry in entries],
        key=lambda row: str(row["sample"]),
    )
    if len({str(row["sample"]) for row in normalized}) != len(normalized):
        raise ValueError("portable content tree has duplicate samples")
    return {
        "n_files": len(normalized),
        "tree_sha256": audit.canonical_sha256(normalized),
        "entries": normalized,
    }


def validate_original_tree(
    tree: Mapping[str, Any], name: str
) -> None:
    entries = tree.get("entries")
    if not isinstance(entries, list):
        raise ValueError(f"{name}: receipt entries are absent")
    if (
        int(tree.get("n_files", -1)) != len(entries)
        or str(tree.get("tree_sha256", ""))
        != audit.canonical_sha256(
            sorted(
                [dict(entry) for entry in entries],
                key=lambda row: str(row["sample"]),
            )
        )
    ):
        raise ValueError(f"{name}: original content-tree hash differs")


def validate_receipt(
    project_root: Path,
    receipt_path: Path,
    *,
    full_hash_he: bool = False,
) -> dict[str, Any]:
    root = project_root.resolve()
    receipt_path = receipt_path.resolve()
    receipt = audit.read_json(receipt_path)
    receipt_payload_sha256 = audit.validate_payload_hash(
        receipt, "receipt_payload_sha256", receipt_path
    )
    if (
        receipt.get("schema_version")
        != "he_boundary_prediction.v1.r4_input_content.v1"
        or receipt.get("status")
        != "COHORT_LABEL_AND_HE_CONTENT_VERIFIED"
    ):
        raise ValueError("Dense input receipt schema/status differs")

    (
        source,
        labels,
        data_contract,
        label_contract,
        feature_manifest,
        paths,
        excluded,
    ) = audit.load_sources(root)

    if sorted(receipt["scope"].get("excluded_from_clean177", [])) != sorted(excluded):
        raise ValueError("receipt exclusion list differs from the working cohort")
    expected_artifacts = {
        **paths,
        "auditor": root / "scripts" / "audit_input_content.py",
    }
    for name, path in expected_artifacts.items():
        recorded = receipt.get("artifacts", {}).get(name)
        if not isinstance(recorded, dict):
            raise ValueError(f"input receipt artifact {name} is absent")
        if normalize_recorded_artifact(recorded) != normalized_artifact(path):
            raise ValueError(f"input receipt artifact {name} differs")
    expected_contracts = {
        "data_contract_payload_sha256": data_contract[
            "contract_payload_sha256"
        ],
        "label_contract_payload_sha256": label_contract[
            "contract_payload_sha256"
        ],
    }
    if receipt.get("contracts") != expected_contracts:
        raise ValueError("input receipt source contract hashes differ")

    content_trees = receipt.get("content_trees")
    if not isinstance(content_trees, dict):
        raise ValueError("input receipt content trees are absent")
    for name in ("aligned_he", "dense_labels"):
        tree = content_trees.get(name)
        if not isinstance(tree, dict):
            raise ValueError(f"input receipt {name} tree is absent")
        validate_original_tree(tree, name)

    label_by_sample = {
        str(row["sample"]): row for row in labels.to_dict("records")
    }
    attestations = audit.feature_attestations(feature_manifest)
    receipt_labels = {
        str(row["sample"]): dict(row)
        for row in content_trees["dense_labels"]["entries"]
    }
    receipt_he = {
        str(row["sample"]): dict(row)
        for row in content_trees["aligned_he"]["entries"]
    }
    expected_samples = set(source["sample"].astype(str))
    if (
        set(label_by_sample) != expected_samples
        or set(receipt_labels) != expected_samples
        or set(receipt_he) != expected_samples
    ):
        raise ValueError("input receipt sample sets do not close the working cohort")


    attestations, self_attested = audit.with_self_attestation(attestations, source)
    declared = sorted(receipt["scope"].get("self_attested_he", []))
    if declared != sorted(self_attested):
        raise ValueError(
            "receipt's self-attested list disagrees with the cohort: "
            f"declared {len(declared)}, actual {len(self_attested)}"
        )

    current_labels: list[dict[str, Any]] = []
    current_he: list[dict[str, Any]] = []
    for source_row in source.sort_values("sample").to_dict("records"):
        sample = str(source_row["sample"])
        label_row = audit.inspect_label(
            source_row, label_by_sample[sample]
        )
        if normalize_content_entry(
            receipt_labels[sample]
        ) != normalize_content_entry(label_row):
            raise ValueError(f"{sample}: label receipt entry differs")
        current_labels.append(label_row)

        recorded_he = dict(receipt_he[sample])
        current_path = Path(str(source_row["he_path"])).resolve()
        replay_row = dict(recorded_he)
        replay_row["path"] = str(current_path)
        he_row = audit.inspect_he(
            source_row,
            attestations[sample],
            full_hash=full_hash_he,
            receipt_row=None if full_hash_he else replay_row,
        )
        if normalize_content_entry(
            recorded_he
        ) != normalize_content_entry(he_row):
            raise ValueError(f"{sample}: H&E receipt entry differs")
        current_he.append(he_row)

    portable_trees = {
        "aligned_he": portable_content_tree(current_he),
        "dense_labels": portable_content_tree(current_labels),
    }
    result = {
        "schema_version": SCHEMA,
        "status": "R4_INPUT_CONTENT_RECEIPT_VALIDATED",
        "verification_mode": {
            "dense_labels": "live_full_sha256_shape_and_identity",
            "aligned_he": (
                "live_full_sha256_size_mtime_and_geometry"
                if full_hash_he
                else "receipt_sha256_plus_live_size_mtime_and_geometry"
            ),
        },
        "receipt": {
            "path_identity": path_identity(receipt_path),
            "sha256": audit.sha256_file(receipt_path),
            "size_bytes": receipt_path.stat().st_size,
            "payload_sha256": receipt_payload_sha256,
        },
        "content_trees": {
            name: {
                "n_files": int(content_trees[name]["n_files"]),
                "receipt_tree_sha256": str(
                    content_trees[name]["tree_sha256"]
                ),
                "portable_tree_sha256": str(
                    portable_trees[name]["tree_sha256"]
                ),
            }
            for name in ("aligned_he", "dense_labels")
        },
        "validator": normalized_artifact(Path(__file__)),
    }


    expected_files = int(receipt["scope"]["n_samples"])
    mismatched = {
        name: int(tree["n_files"])
        for name, tree in result["content_trees"].items()
        if int(tree["n_files"]) != expected_files
    }
    if mismatched:
        raise ValueError(
            f"portable input trees do not close the declared cohort "
            f"({expected_files} samples): {mismatched}"
        )
    return result


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", type=Path, default=root)
    parser.add_argument(
        "--receipt",
        type=Path,
        default=root / "data" / "manifests" / "input_content_receipt.json",
    )
    parser.add_argument(
        "--full-hash-he",
        action="store_true",
        help="Recompute every working-cohort H&E SHA-256 digest instead of stat reuse.",
    )
    return parser.parse_args()


def main() -> None:
    arguments = parse_args()
    result = validate_receipt(
        arguments.project_root,
        arguments.receipt,
        full_hash_he=arguments.full_hash_he,
    )
    print(json.dumps(result, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
