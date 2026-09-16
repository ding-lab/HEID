#!/usr/bin/env python3

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import pandas as pd

from label_contract import (
    CANONICAL_CLASSES_17,
    LABEL_MAPPING_ID,
)


A1 = Path(__file__).resolve().parents[1]
PC = A1.parent
COHORT = A1 / "data/cohort_clean177.csv"
SPLIT = A1 / "data/split_outer5_inner5.json"
CELL_TABLE_DIR = PC / "data/cell_tables"
FEATURE_ROOT = A1 / "outputs/features"
ADAPTER_ROOT = A1 / "outputs/adapters"

CLASSES = list(CANONICAL_CLASSES_17)
CT2IDX = {name: idx for idx, name in enumerate(CLASSES)}

BACKBONE_DIM = {"uni2": 1536, "phikon": 1024}


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def stable_json_sha256(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def ordered_ids_sha256(values: Iterable[str]) -> str:
    h = hashlib.sha256()
    for value in values:
        encoded = str(value).encode("utf-8")
        h.update(len(encoded).to_bytes(8, "little"))
        h.update(encoded)
    return h.hexdigest()


@dataclass(frozen=True)
class SampleContract:
    sample: str
    patient: str
    cancer: str
    own_outer_fold: int
    aligned_he_path: Path
    split_id: str


def load_contract(
    cohort_path: Path = COHORT,
    split_path: Path = SPLIT,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    cohort = pd.read_csv(cohort_path)
    split = json.loads(split_path.read_text())
    required = {"sample", "patient", "cancer", "outer_fold", "aligned_he_path", "contract_split_id"}
    missing = sorted(required - set(cohort.columns))
    if missing:
        raise ValueError(f"clean cohort missing required fields: {missing}")
    if cohort["sample"].astype(str).duplicated().any():
        raise ValueError("clean cohort contains duplicate sample IDs")
    split_id = str(split["split_id"])
    if set(cohort["contract_split_id"].astype(str)) != {split_id}:
        raise ValueError("cohort and split_id disagree")
    split_samples = set(map(str, split["sample_outer_fold"]))
    cohort_samples = set(cohort["sample"].astype(str))
    if split_samples != cohort_samples:
        raise ValueError("cohort and split sample universes differ")
    for outer in split["outer_folds"]:
        train = set(map(str, outer["train_samples"]))
        test = set(map(str, outer["test_samples"]))
        if train & test or train | test != cohort_samples:
            raise ValueError(f"outer fold {outer['outer_fold']} is not a disjoint partition")
        if set(map(str, outer["train_patients"])) & set(map(str, outer["test_patients"])):
            raise ValueError(f"outer fold {outer['outer_fold']} leaks patients")
    cohort = cohort.copy()
    cohort["sample"] = cohort["sample"].astype(str)
    cohort["patient"] = cohort["patient"].astype(str)
    return cohort, split


def sample_contract(sample: str, cohort: pd.DataFrame, split: dict[str, Any]) -> SampleContract:
    rows = cohort[cohort["sample"] == str(sample)]
    if len(rows) != 1:
        raise ValueError(f"sample {sample!r} has {len(rows)} clean-cohort records")
    row = rows.iloc[0]
    own = int(split["sample_outer_fold"][str(sample)])
    if own != int(row["outer_fold"]):
        raise ValueError(f"sample {sample} cohort/split outer fold mismatch")
    return SampleContract(
        sample=str(sample),
        patient=str(row["patient"]),
        cancer=str(row["cancer"]),
        own_outer_fold=own,
        aligned_he_path=Path(str(row["aligned_he_path"])),
        split_id=str(split["split_id"]),
    )


def outer_fold(split: dict[str, Any], fold: int) -> dict[str, Any]:
    matches = [item for item in split["outer_folds"] if int(item["outer_fold"]) == int(fold)]
    if len(matches) != 1:
        raise ValueError(f"outer fold {fold} is absent or duplicated")
    return matches[0]


def assert_role_allowed(contract: SampleContract, split: dict[str, Any], role: str, adapter_fold: int) -> None:
    if role == "oof":
        if int(adapter_fold) != contract.own_outer_fold:
            raise ValueError(
                f"OOF sample {contract.sample} requires own fold {contract.own_outer_fold}, "
                f"not adapter fold {adapter_fold}"
            )
        return
    if role != "train":
        raise ValueError(f"unsupported feature role {role!r}")
    fold = outer_fold(split, adapter_fold)
    explicit_train = set(map(str, fold["train_samples"]))
    explicit_test = set(map(str, fold["test_samples"]))
    if contract.sample in explicit_test or contract.sample not in explicit_train:
        raise ValueError(
            f"sample {contract.sample} is not an explicit outer-train sample for fold {adapter_fold}"
        )
    if contract.own_outer_fold == int(adapter_fold):
        raise ValueError("own-fold features are forbidden from the training role")


def adapter_path(
    backbone: str,
    fold: int,
    preprocessing_id: str,
    backbone_interface_id: str,
) -> Path:
    root = ADAPTER_ROOT / LABEL_MAPPING_ID / preprocessing_id / backbone_interface_id
    if backbone == "uni2":
        return root / "uni2" / f"lora_pc_fold{fold}.pt"
    if backbone == "phikon":
        return root / "phikon" / f"lora_phikon_fold{fold}.pt"
    raise ValueError(f"unsupported backbone {backbone!r}")


def feature_path(
    backbone: str,
    role: str,
    sample: str,
    fold: int,
    preprocessing_id: str,
    backbone_interface_id: str,
) -> Path:
    if backbone not in BACKBONE_DIM:
        raise ValueError(f"unsupported backbone {backbone!r}")
    if role == "oof":
        return FEATURE_ROOT / LABEL_MAPPING_ID / preprocessing_id / backbone_interface_id / backbone / "oof" / f"{sample}.pt"
    if role == "train":
        return FEATURE_ROOT / LABEL_MAPPING_ID / preprocessing_id / backbone_interface_id / backbone / "train" / f"fold{fold}" / f"{sample}.pt"
    raise ValueError(f"unsupported feature role {role!r}")


def manifest_path(
    backbone: str,
    role: str,
    sample: str,
    fold: int,
    preprocessing_id: str,
    backbone_interface_id: str,
) -> Path:
    root = FEATURE_ROOT / LABEL_MAPPING_ID / preprocessing_id / backbone_interface_id
    if role == "oof":
        return root / "manifests" / backbone / "oof" / f"{sample}.json"
    if role == "train":
        return root / "manifests" / backbone / "train" / f"fold{fold}" / f"{sample}.json"
    raise ValueError(f"unsupported feature role {role!r}")


def cell_manifest_path(
    backbone: str,
    role: str,
    sample: str,
    fold: int,
    preprocessing_id: str,
    backbone_interface_id: str,
) -> Path:
    root = FEATURE_ROOT / LABEL_MAPPING_ID / preprocessing_id / backbone_interface_id
    if role == "oof":
        return root / "cell_manifests" / backbone / "oof" / f"{sample}.npz"
    if role == "train":
        return root / "cell_manifests" / backbone / "train" / f"fold{fold}" / f"{sample}.npz"
    raise ValueError(f"unsupported feature role {role!r}")


def selection_dir(fold: int) -> Path:
    return FEATURE_ROOT / "selections" / LABEL_MAPPING_ID / f"fold{fold}"
