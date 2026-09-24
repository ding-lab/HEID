#!/usr/bin/env python3

from __future__ import annotations
import os
PROJECTS_ROOT = os.environ.get("PROJECTS_ROOT", "/data/heid")

import argparse
import hashlib
import json
import tempfile
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd
import tifffile

from experiment_config import REFERENCE, cohort_size


SCHEMA = "he_boundary_prediction.v1.r4_input_content.v1"
STATUS = "COHORT_LABEL_AND_HE_CONTENT_VERIFIED"


def sha256_file(path: Path, chunk_size: int = 16 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_sha256(payload: Any) -> str:
    return hashlib.sha256(
        json.dumps(
            payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False
        ).encode("utf-8")
    ).hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{path}: expected a JSON object")
    return payload


def validate_payload_hash(
    payload: Mapping[str, Any], field: str, path: Path
) -> str:
    materialized = dict(payload)
    claimed = materialized.pop(field, None)
    observed = canonical_sha256(materialized)
    if claimed != observed:
        raise ValueError(f"{path}: {field} differs")
    return str(claimed)


def artifact(path: Path) -> dict[str, Any]:
    path = path.resolve()
    return {
        "path": str(path),
        "sha256": sha256_file(path),
        "size_bytes": path.stat().st_size,
    }


def same_file(left: Path, right: Path) -> bool:
    try:
        return os.path.samefile(left, right)
    except OSError:
        return False


def storage_alias_equivalent(left: Path, right: Path) -> bool:

    if left.exists() and right.exists():
        return same_file(left, right)
    left_parts = left.parts
    right_parts = right.parts
    if "projects" not in left_parts or "projects" not in right_parts:
        return False
    return (
        left_parts[left_parts.index("projects") :]
        == right_parts[right_parts.index("projects") :]
    )


def feature_attestations(
    feature_manifest: Mapping[str, Any],
) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    records = feature_manifest.get("records")
    if not isinstance(records, list):
        raise ValueError("feature attestation manifest lacks records")
    for record in records:
        sample = str(record.get("sample", ""))
        grids = [
            feature
            for feature in record.get("features", [])
            if feature.get("kind") == "tissue_grid"
        ]
        if not sample or len(grids) != 1:
            raise ValueError(f"{sample!r}: expected one tissue-grid feature")
        attestation = grids[0].get("full_content_attestation")
        if not isinstance(attestation, dict):
            raise ValueError(f"{sample}: missing full H&E attestation")
        if sample in result:
            raise ValueError(f"{sample}: duplicate feature attestation")
        result[sample] = dict(attestation)
    return result


def with_self_attestation(
    attestations: dict[str, dict[str, Any]], source: pd.DataFrame
) -> tuple[dict[str, dict[str, Any]], list[str]]:

    filled = dict(attestations)
    self_attested = sorted(set(source["sample"].astype(str)) - set(attestations))
    for sample in self_attested:
        row = source.loc[source["sample"].astype(str) == sample].iloc[0]
        path = Path(str(row["he_path"]))
        stat = path.stat()
        filled[sample] = {
            "he_path": str(path),
            "he_size_bytes": int(stat.st_size),
            "he_mtime_ns": int(stat.st_mtime_ns),
            "he_full_content_sha256": str(row["he_sha256"]),
            "attestation_class": "self_attested_at_cohort_build",
        }
    return filled, self_attested


def inspect_label(
    source_row: Mapping[str, Any],
    label_row: Mapping[str, Any],
) -> dict[str, Any]:
    sample = str(source_row["sample"])
    source_path = Path(str(source_row["label_path"]))
    declared_path = Path(str(label_row["path"]))
    if not source_path.is_file():
        raise FileNotFoundError(source_path)
    if not storage_alias_equivalent(source_path, declared_path):
        raise ValueError(f"{sample}: source/label manifest paths differ")
    stat = source_path.stat()
    declared_size = int(label_row["size_bytes"])
    if stat.st_size != declared_size:
        raise ValueError(f"{sample}: label size differs")
    observed_sha = sha256_file(source_path)
    declared_sha = str(label_row["sha256"])
    if observed_sha != declared_sha:
        raise ValueError(f"{sample}: label full-content hash differs")
    with np.load(source_path, allow_pickle=False) as archive:
        required = {"tumor_mask", "label_valid", "metadata_json"}
        if not required.issubset(archive.files):
            raise ValueError(f"{sample}: label archive schema differs")
        tumor = np.asarray(archive["tumor_mask"])
        valid = np.asarray(archive["label_valid"])
        metadata = json.loads(
            str(np.asarray(archive["metadata_json"]).item())
        )
    expected_shape = (int(label_row["shape_y"]), int(label_row["shape_x"]))
    if tumor.shape != expected_shape or valid.shape != expected_shape:
        raise ValueError(f"{sample}: label shape differs")
    expected_identity = {
        "sample": sample,
        "patient": str(source_row["patient"]),
        "cancer": str(source_row["cancer"]),
        "fold": int(source_row["fold"]),
    }
    for key, expected in expected_identity.items():
        observed = metadata.get(key)
        if key == "fold":
            observed = int(observed)
        if observed != expected:
            raise ValueError(f"{sample}: label metadata {key} differs")
    if list(metadata.get("shape_yx", [])) != list(expected_shape):
        raise ValueError(f"{sample}: label metadata shape differs")
    if not np.isclose(float(metadata.get("resolution_um")), 10.0):
        raise ValueError(f"{sample}: label resolution differs")
    if metadata.get("he", {}).get("sha256") != str(source_row["he_sha256"]):
        raise ValueError(f"{sample}: label H&E provenance differs")
    return {
        "sample": sample,
        "path": str(source_path.resolve()),
        "sha256": observed_sha,
        "size_bytes": stat.st_size,
        "shape_yx": list(expected_shape),
    }


def inspect_he(
    source_row: Mapping[str, Any],
    attestation: Mapping[str, Any],
    *,
    full_hash: bool,
    receipt_row: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    sample = str(source_row["sample"])
    source_path = Path(str(source_row["he_path"]))
    attested_path = Path(str(attestation.get("he_path", "")))
    if not source_path.is_file():
        raise FileNotFoundError(source_path)
    if not storage_alias_equivalent(source_path, attested_path):
        raise ValueError(f"{sample}: H&E source/attestation paths differ")
    stat = source_path.stat()
    declared_size = int(attestation["he_size_bytes"])
    declared_mtime = int(attestation["he_mtime_ns"])
    declared_sha = str(attestation["he_full_content_sha256"])
    if stat.st_size != declared_size or stat.st_mtime_ns != declared_mtime:
        raise ValueError(f"{sample}: H&E size/mtime differs from attestation")
    if str(source_row["he_sha256"]) != declared_sha:
        raise ValueError(f"{sample}: source H&E hash differs from attestation")
    if full_hash:
        observed_sha = sha256_file(source_path)
        if observed_sha != declared_sha:
            raise ValueError(f"{sample}: live H&E full-content hash differs")
    else:
        if receipt_row is None:
            raise ValueError("receipt reuse requires an H&E receipt row")
        observed_sha = str(receipt_row.get("sha256", ""))
        expected_receipt = {
            "sample": sample,
            "path": str(source_path.resolve()),
            "sha256": declared_sha,
            "size_bytes": stat.st_size,
            "mtime_ns": stat.st_mtime_ns,
            "shape_yx": [
                int(source_row["he_size_y_px"]),
                int(source_row["he_size_x_px"]),
            ],
            "mpp_um": float(source_row["he_mpp_um"]),
        }
        if dict(receipt_row) != expected_receipt:
            raise ValueError(f"{sample}: H&E receipt row differs")
    with tifffile.TiffFile(source_path) as handle:
        shape = tuple(int(value) for value in handle.series[0].shape)
    expected_yx = (
        int(source_row["he_size_y_px"]),
        int(source_row["he_size_x_px"]),
    )
    if len(shape) < 2 or shape[0:2] != expected_yx:
        raise ValueError(
            f"{sample}: live H&E geometry {shape} differs from {expected_yx}"
        )
    return {
        "sample": sample,
        "path": str(source_path.resolve()),
        "sha256": observed_sha,
        "size_bytes": stat.st_size,
        "mtime_ns": stat.st_mtime_ns,
        "shape_yx": list(expected_yx),
        "mpp_um": float(source_row["he_mpp_um"]),
    }


def content_tree(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    normalized = sorted(
        [dict(row) for row in rows], key=lambda row: str(row["sample"])
    )
    if len({str(row["sample"]) for row in normalized}) != len(normalized):
        raise ValueError("content tree has duplicate samples")
    return {
        "n_files": len(normalized),
        "tree_sha256": canonical_sha256(normalized),
        "entries": normalized,
    }


def load_sources(
    project_root: Path,
) -> tuple[
    pd.DataFrame,
    pd.DataFrame,
    dict[str, Any],
    dict[str, Any],
    dict[str, Any],
    dict[str, Path],
]:
    root = project_root.resolve()
    paths = {
        "source_manifest": root / "data" / "manifests" / "source_cohort.csv",
        "label_manifest": root / "data" / "manifests" / "labels.csv",
        "data_contract": root / "data" / "manifests" / "data_contract.json",
        "label_contract": root / "data" / "manifests" / "label_dataset.json",
    }
    data_contract = read_json(paths["data_contract"])
    label_contract = read_json(paths["label_contract"])
    validate_payload_hash(
        data_contract, "contract_payload_sha256", paths["data_contract"]
    )
    validate_payload_hash(
        label_contract, "contract_payload_sha256", paths["label_contract"]
    )
    feature_path = Path(
        str(
            data_contract["sources"]["a4_full_content_feature_manifest"][
                "path"
            ]
        )
    )
    paths["a4_feature_manifest"] = feature_path
    if (
        sha256_file(feature_path)
        != data_contract["sources"]["a4_full_content_feature_manifest"][
            "sha256"
        ]
    ):
        raise ValueError("feature attestation manifest hash differs")
    feature_manifest = read_json(feature_path)
    source = pd.read_csv(paths["source_manifest"], keep_default_na=False)
    labels = pd.read_csv(paths["label_manifest"], keep_default_na=False)


    full_path = paths["source_manifest"].parent / "source_cohort_reference.csv"
    if not full_path.is_file():
        raise ValueError(f"missing the reference provenance manifest: {full_path}")
    full = pd.read_csv(full_path, keep_default_na=False)
    expected = cohort_size(REFERENCE)["n_samples"]
    if len(full) != expected or full["sample"].nunique() != expected:
        raise ValueError(f"reference provenance manifest is not {expected} unique samples")
    if canonical_sha256(full.to_dict("records")) != data_contract["manifest"]["records_sha256"]:
        raise ValueError("reference provenance manifest records hash differs")

    if len(source) != source["sample"].nunique() or len(labels) != labels["sample"].nunique():
        raise ValueError("input manifests contain duplicate samples")
    if set(source["sample"].astype(str)) != set(labels["sample"].astype(str)):
        raise ValueError("source and label manifests cover different samples")


    outside = sorted(set(source["sample"].astype(str)) - set(full["sample"].astype(str)))

    full_rows = {str(row["sample"]): row for row in full.to_dict("records")}


    PROVENANCE = (
        "sample", "patient", "cancer", "cohort", "n_cells",
        "he_size_y_px", "he_size_x_px", "he_mpp_um", "he_sha256",
        "boundary_reference_csv",
    )
    edited = [
        sample
        for row in source.to_dict("records")
        if (sample := str(row["sample"])) in full_rows
        and any(row.get(k) != full_rows[sample].get(k) for k in PROVENANCE)
    ]
    if edited:
        raise ValueError(f"working cohort rows differ from reference provenance: {sorted(edited)}")


    ALIGNED_HE_ROOT = (
        PROJECTS_ROOT + "/align/aligned_he/"
    )
    misdeclared = sorted(
        str(row["sample"])
        for row in source.to_dict("records")
        if not (
            str(row["he_path"]).startswith(ALIGNED_HE_ROOT)
            and str(row["cell_table_path"]).startswith(ALIGNED_HE_ROOT)
        )
    )
    if misdeclared:
        raise ValueError(
            "coordinate source is declared as corrected alignment but these rows point elsewhere: "
            f"{misdeclared}"
        )


    repointed = [
        sample
        for row in source.to_dict("records")
        if (sample := str(row["sample"])) in full_rows
        and str(row["he_path"]) != str(full_rows[sample].get("he_path"))
    ]
    print(
        f"corrected coordinate source on all {len(source)} rows; "
        f"{len(repointed)} reference rows using relocated inputs, "
        "H&E content unchanged (he_sha256 compared above)",
        flush=True,
    )


    excluded = sorted(set(full["sample"].astype(str)) - set(source["sample"].astype(str)))
    print(f"cohort has {len(outside)} sections outside the reference provenance set", flush=True)
    source_records_hash = canonical_sha256(source.to_dict("records"))
    if (
        label_contract["sources"]["source_contract"]["sha256"]
        != sha256_file(paths["data_contract"])
        or label_contract["sources"]["source_manifest"]["sha256"]
        != sha256_file(paths["source_manifest"])
    ):
        raise ValueError("label contract source binding differs")
    return (
        source,
        labels,
        data_contract,
        label_contract,
        feature_manifest,
        paths,
        excluded,
    )


def build_receipt(project_root: Path) -> dict[str, Any]:
    (
        source,
        labels,
        data_contract,
        label_contract,
        feature_manifest,
        paths,
        excluded,
    ) = load_sources(project_root)
    label_by_sample = {
        str(row["sample"]): row
        for row in labels.to_dict("records")
    }
    attestations, self_attested = with_self_attestation(
        feature_attestations(feature_manifest), source
    )
    label_rows: list[dict[str, Any]] = []
    he_rows: list[dict[str, Any]] = []
    for source_row in source.sort_values("sample").to_dict("records"):
        sample = str(source_row["sample"])
        label_rows.append(
            inspect_label(source_row, label_by_sample[sample])
        )
        he_rows.append(
            inspect_he(source_row, attestations[sample], full_hash=True)
        )
        print(f"verified-input sample={sample}", flush=True)
    payload: dict[str, Any] = {
        "schema_version": SCHEMA,
        "status": STATUS,
        "scope": {
            "n_samples": int(len(source)),


            "excluded_from_clean177": excluded,

            "self_attested_he": self_attested,
            "n_patients": int(source["patient"].nunique()),
            "n_cancers": int(source["cancer"].nunique()),
            "n_folds": int(source["fold"].nunique()),
            "he_hash_policy": "full_file_sha256",
            "label_hash_policy": "full_file_sha256",
        },
        "content_trees": {
            "aligned_he": content_tree(he_rows),
            "dense_labels": content_tree(label_rows),
        },
        "contracts": {
            "data_contract_payload_sha256": data_contract[
                "contract_payload_sha256"
            ],
            "label_contract_payload_sha256": label_contract[
                "contract_payload_sha256"
            ],
        },
        "artifacts": {
            name: artifact(path)
            for name, path in {
                **paths,
                "auditor": Path(__file__),
            }.items()
        },
    }
    payload["receipt_payload_sha256"] = canonical_sha256(payload)
    return payload


def validate_existing(
    project_root: Path, receipt_path: Path
) -> dict[str, Any]:
    receipt = read_json(receipt_path)
    validate_payload_hash(
        receipt, "receipt_payload_sha256", receipt_path
    )
    if receipt.get("schema_version") != SCHEMA or receipt.get("status") != STATUS:
        raise ValueError("existing input receipt has the wrong contract")
    (
        source,
        labels,
        data_contract,
        label_contract,
        feature_manifest,
        paths,
        excluded,
    ) = load_sources(project_root)
    expected_artifacts = {
        **paths,
        "auditor": Path(__file__),
    }
    for name, path in expected_artifacts.items():
        recorded = receipt.get("artifacts", {}).get(name)
        if recorded != artifact(path):
            raise ValueError(f"existing input receipt {name} differs")
    if receipt.get("contracts") != {
        "data_contract_payload_sha256": data_contract[
            "contract_payload_sha256"
        ],
        "label_contract_payload_sha256": label_contract[
            "contract_payload_sha256"
        ],
    }:
        raise ValueError("existing input receipt contract hashes differ")
    label_by_sample = {
        str(row["sample"]): row
        for row in labels.to_dict("records")
    }
    attestations, _ = with_self_attestation(feature_attestations(feature_manifest), source)
    he_receipt_rows = {
        str(row["sample"]): row
        for row in receipt["content_trees"]["aligned_he"]["entries"]
    }
    label_rows: list[dict[str, Any]] = []
    he_rows: list[dict[str, Any]] = []
    for source_row in source.sort_values("sample").to_dict("records"):
        sample = str(source_row["sample"])
        label_rows.append(
            inspect_label(source_row, label_by_sample[sample])
        )
        he_rows.append(
            inspect_he(
                source_row,
                attestations[sample],
                full_hash=False,
                receipt_row=he_receipt_rows.get(sample),
            )
        )
    expected_trees = {
        "aligned_he": content_tree(he_rows),
        "dense_labels": content_tree(label_rows),
    }
    if receipt.get("content_trees") != expected_trees:
        raise ValueError("existing input receipt content tree differs")
    return receipt


def atomic_write_once(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if read_json(path) != dict(payload):
            raise FileExistsError(f"{path}: existing receipt differs")
        return
    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        dir=path.parent,
        prefix=".input-content-",
        suffix=".tmp",
        delete=False,
    ) as handle:
        temporary = Path(handle.name)
        json.dump(dict(payload), handle, indent=2, sort_keys=True)
        handle.write("\n")
    try:
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", type=Path, default=root)
    parser.add_argument(
        "--output",
        type=Path,
        default=root / "data" / "manifests" / "input_content_receipt.json",
    )
    return parser.parse_args()


def main() -> None:
    arguments = parse_args()
    output = arguments.output.resolve()
    if output.is_file():
        receipt = validate_existing(arguments.project_root, output)
        print(
            "INPUT_CONTENT_ALREADY_COMPLETE "
            + receipt["receipt_payload_sha256"],
            flush=True,
        )
        return
    payload = build_receipt(arguments.project_root)
    atomic_write_once(output, payload)
    validate_existing(arguments.project_root, output)
    print(
        "INPUT_CONTENT_COMPLETE " + payload["receipt_payload_sha256"],
        flush=True,
    )


if __name__ == "__main__":
    main()
