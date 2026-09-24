#!/usr/bin/env python3

from __future__ import annotations
from pathlib import Path
_RELEASE = Path(__file__).resolve().parents[2]

import hashlib
import json
import os
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from paths import PC, COHORT_REFERENCE_ROOT, CELL_ROOT, normalise, sha256_file, sha256_list
from paths import expand_env as _expand_env


CONTRACT = _RELEASE / "cell/configs/contract.json"
LABELS_DIR = CELL_ROOT / "outputs/labels"
WMAP_ROOTS = (CELL_ROOT / "outputs/wmaps/cell", COHORT_REFERENCE_ROOT / "outputs/wmaps/cell")

OUTPUTS = CELL_ROOT / "outputs"
CROPBANK_ROOT = OUTPUTS / "cropbanks"
ADAPTER_ROOT = OUTPUTS / "models/adapters"
FEATURE_ROOT = OUTPUTS / "features"
MANIFEST_ROOT = OUTPUTS / "manifests"
LOG_ROOT = OUTPUTS / "logs"

CONTRACT_SCHEMA = "v8.contract_all5k.v2"
CANONICAL_FOLD_KEY = "fold_he_safe"
ARM = "raw_rgb"
BACKBONES = ("uni2", "phikon")
POLY_NAMES = ("sigma10", "sigma3")
RING_NAMES = ("A0", "A1", "A2", "A3", "A4")
VARIANTS = ("cls",) + RING_NAMES + POLY_NAMES

PX_UM = 0.2125
CROP = 224
GRID = 16
TRAIN_COLUMNS = ["cell_id", "he_px", "he_py", "x_um", "y_um", "dx_um", "dy_um", "block"]


if str(_RELEASE / "cell/scripts") not in sys.path:
    sys.path.insert(0, str(_RELEASE / "cell/scripts"))
from label_contract import (
    CANONICAL_CLASSES_17,
    LABEL_MAPPING_ID,
    LABEL_MAPPING_SHA256,
)

CLASSES17 = list(CANONICAL_CLASSES_17)
N_CLASS = len(CLASSES17)


def canonical_sha256(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def order_sha256(values, domain: str = "v8.cell_order.v1") -> str:
    values = [str(v) for v in values]
    digest = hashlib.sha256(domain.encode("utf-8") + b"\0")
    digest.update(len(values).to_bytes(8, "little"))
    for value in values:
        encoded = value.encode("utf-8")
        digest.update(len(encoded).to_bytes(8, "little"))
        digest.update(encoded)
    return digest.hexdigest()


def identity_sha256(cell_ids, labels) -> str:
    if len(cell_ids) != len(labels):
        raise RuntimeError("feature identity arrays differ in length")
    digest = hashlib.sha256(b"a2.feature_identity.length_prefixed.v2\0")
    digest.update(len(cell_ids).to_bytes(8, "little"))
    for cell, label in zip(cell_ids, labels, strict=True):
        for value in (str(cell), str(label)):
            encoded = value.encode("utf-8")
            digest.update(len(encoded).to_bytes(8, "little"))
            digest.update(encoded)
    return digest.hexdigest()


def atomic_json(path: Path, value: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + f".tmp.{os.getpid()}")
    tmp.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    os.replace(tmp, path)


def atomic_npy(path: Path, array: np.ndarray) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + f".tmp.{os.getpid()}")
    with tmp.open("wb") as handle:
        np.save(handle, array, allow_pickle=False)
    os.replace(tmp, path)


_CONTRACT_CACHE: dict[str, Any] | None = None


def load_contract() -> dict[str, Any]:
    global _CONTRACT_CACHE
    if _CONTRACT_CACHE is not None:
        return _CONTRACT_CACHE
    contract = _expand_env(json.loads(CONTRACT.read_text()))
    if contract.get("schema_version") != CONTRACT_SCHEMA:
        raise RuntimeError(f"unsupported Cell contract schema: {contract.get('schema_version')}")
    if contract["folds"].get("canonical_fold_key") != CANONICAL_FOLD_KEY:
        raise RuntimeError("contract canonical fold key is not fold_he_safe")
    allow = contract["cohort"]["allowlist"]


    if sha256_list(allow) != contract["cohort"]["allowlist_sha256"]:
        raise RuntimeError("contract allowlist does not match its own recorded sha256")
    rows = contract["per_sample"]
    if len(rows) != len(allow) or {r["sample"] for r in rows} != set(allow):
        raise RuntimeError("contract per_sample does not cover the allowlist exactly")
    for row in rows:
        if CANONICAL_FOLD_KEY not in row:
            raise RuntimeError(f"{row['sample']}: contract row carries no {CANONICAL_FOLD_KEY}")
    _CONTRACT_CACHE = contract
    return contract


def contract_sha256() -> str:
    return sha256_file(CONTRACT)


def allowlist() -> list[str]:
    return list(load_contract()["cohort"]["allowlist"])


def records() -> dict[str, dict[str, Any]]:
    return {r["sample"]: r for r in load_contract()["per_sample"]}


def record(sample: str) -> dict[str, Any]:
    table = records()
    if sample not in table:
        raise RuntimeError(f"sample is absent from the Cell contract: {sample}")
    return table[sample]


def own_fold(sample: str) -> int:
    return int(record(sample)[CANONICAL_FOLD_KEY])


_MEMBERSHIP_CACHE: dict[int, dict[str, list[str]]] | None = None


def fold_membership() -> dict[int, dict[str, list[str]]]:
    global _MEMBERSHIP_CACHE
    if _MEMBERSHIP_CACHE is not None:
        return _MEMBERSHIP_CACHE
    table = records()
    order = allowlist()
    held: dict[int, list[str]] = {f: [] for f in range(5)}
    for sample in order:
        held[int(table[sample][CANONICAL_FOLD_KEY])].append(sample)
    universe = set(order)
    out: dict[int, dict[str, list[str]]] = {}
    for fold in range(5):
        test = sorted(held[fold])
        train = sorted(universe - set(test))
        if not test or set(test) & set(train) or set(test) | set(train) != universe:
            raise RuntimeError(f"fold {fold} is not an exact partition of the Cell cohort")
        out[fold] = {"test_samples": test, "train_samples": train}
    _MEMBERSHIP_CACHE = out
    return out


def assert_outside_fold(samples, fold: int, what: str) -> None:
    membership = fold_membership()[fold]
    held = set(membership["test_samples"])
    observed = {str(s) for s in samples}
    intruders = sorted(observed & held)
    if intruders:
        raise RuntimeError(
            f"{what}: {len(intruders)} fold-{fold} held-out slide(s) entered a fold-{fold} "
            f"artefact, e.g. {intruders[:5]}"
        )
    outside = sorted(observed - set(allowlist()))
    if outside:
        raise RuntimeError(f"{what}: slide(s) outside the Cell cohort: {outside[:5]}")


_BINDING_CACHE: str | None = None


def cohort_binding_payload() -> dict[str, Any]:
    contract = load_contract()
    rows = records()
    return {
        "schema": "v8.cohort_binding.v1",
        "canonical_fold_key": CANONICAL_FOLD_KEY,
        "allowlist": allowlist(),
        "fold_of_sample": {s: int(rows[s][CANONICAL_FOLD_KEY]) for s in allowlist()},
        "cells_of_sample": {s: int(rows[s]["n_keep_cells_own"]) for s in allowlist()},
        "label_table_sha256": contract["labels"]["sha256"],
        "label_column": contract["labels"]["column_used"],
        "taxonomy": contract["labels"]["taxonomy"],
    }


def cohort_binding_sha256() -> str:
    global _BINDING_CACHE
    if _BINDING_CACHE is not None:
        return _BINDING_CACHE
    _BINDING_CACHE = canonical_sha256(cohort_binding_payload())
    return _BINDING_CACHE


def fold_fingerprint(fold: int) -> dict[str, Any]:
    membership = fold_membership()[fold]
    return {
        "canonical_fold_key": CANONICAL_FOLD_KEY,
        "outer_fold": fold,
        "n_train_samples": len(membership["train_samples"]),
        "n_test_samples": len(membership["test_samples"]),
        "train_sample_sha256": sha256_list(membership["train_samples"]),
        "test_sample_sha256": sha256_list(membership["test_samples"]),
        "cohort_binding_sha256": cohort_binding_sha256(),
        "outer_test_used_for_training": False,
        "outer_test_used_for_checkpoint_selection": False,
    }


def assert_fold_binding(stored: dict[str, Any] | None, fold: int, what: str) -> None:
    if not isinstance(stored, dict):
        raise RuntimeError(f"{what}: no fold fingerprint recorded, refusing to bind")
    expected = fold_fingerprint(fold)
    for key in ("outer_fold", "train_sample_sha256", "test_sample_sha256"):
        if stored.get(key) != expected[key]:
            raise RuntimeError(
                f"{what}: fold binding differs on {key} -- stored {stored.get(key)!r}, "
                f"this cohort {expected[key]!r}"
            )


def he_path(sample: str) -> Path:
    slide_dir = Path(record(sample)["train_parquet"]).parent.parent
    path = slide_dir / "he_aligned.ome.tif"
    if not path.is_file():
        candidates = sorted(slide_dir.glob("he_aligned*.ome.tif*"))
        if len(candidates) != 1:
            raise FileNotFoundError(
                f"{sample}: {len(candidates)} H&E candidates under {slide_dir}"
            )
        path = candidates[0]
    return path


def labels_path(sample: str) -> Path:
    path = LABELS_DIR / f"{sample}.parquet"
    if not path.is_file():
        raise FileNotFoundError(path)
    return path


def wmap_path(name: str, sample: str) -> Path:
    if name not in POLY_NAMES:
        raise ValueError(name)
    candidates = [root / name / f"{sample}.npz" for root in WMAP_ROOTS]
    live = [p for p in candidates if p.is_file()]
    if len(live) != 1:
        raise FileNotFoundError(
            f"{sample}/{name}: {len(live)} weight maps across {[str(p) for p in candidates]}"
        )
    return live[0]


def load_cells(sample: str) -> pd.DataFrame:
    rec = record(sample)
    table_path = Path(rec["train_parquet"])
    table = pd.read_parquet(table_path)
    if list(table.columns) != TRAIN_COLUMNS:
        raise RuntimeError(f"{sample}: unexpected aligned-cell training schema {list(table.columns)}")
    table["cell_id"] = table["cell_id"].astype(str)
    if table["cell_id"].duplicated().any():
        raise RuntimeError(f"{sample}: duplicate cell_id in the aligned-cell training table")
    if len(table) != int(rec["n_v9_training_rows_in_table"]):
        raise RuntimeError(
            f"{sample}: aligned-cell table holds {len(table)} rows, contract says "
            f"{rec['n_v9_training_rows_in_table']}"
        )
    residual = max(
        float(np.abs((table["x_um"] + table["dx_um"]) / PX_UM - table["he_px"]).max()),
        float(np.abs((table["y_um"] + table["dy_um"]) / PX_UM - table["he_py"]).max()),
    )
    if residual != 0.0:
        raise RuntimeError(f"{sample}: he_px is not (x_um + dx_um)/{PX_UM}; residual {residual}")

    labels = pd.read_parquet(labels_path(sample))
    labels["cell_id"] = labels["cell_id"].astype(str)
    if labels["cell_id"].duplicated().any():
        raise RuntimeError(f"{sample}: duplicate cell_id in the label table")
    cells = table.merge(labels, on="cell_id", how="inner")
    if len(cells) != int(rec["n_keep_cells_own"]):
        raise RuntimeError(
            f"{sample}: {len(cells)} labelled cells, contract says {rec['n_keep_cells_own']}"
        )
    if cells["cell_type_refine"].isna().any() or cells["coarse11"].isna().any():
        raise RuntimeError(f"{sample}: a kept cell carries no label")
    unknown = sorted(set(cells["cell_type_refine"].astype(str)) - set(CLASSES17))
    if unknown:
        raise RuntimeError(f"{sample}: labels outside the frozen refine17 vocabulary: {unknown}")
    return cells.sort_values("cell_id", kind="stable").reset_index(drop=True)


def cropbank_dir(fold: int) -> Path:
    return CROPBANK_ROOT / LABEL_MAPPING_ID / "rgb_raw_v1" / f"fold{fold}"


def adapter_path(backbone: str, fold: int) -> Path:
    if backbone not in BACKBONES:
        raise ValueError(backbone)
    return ADAPTER_ROOT / ARM / backbone / f"fold{fold}.pt"


def head_selection_dir() -> Path:
    return MANIFEST_ROOT / "head_selection"


def head_selection_manifest(fold: int) -> Path:
    return head_selection_dir() / f"fold{fold}.json"


def head_selection_artifact(fold: int, sample: str) -> Path:
    return head_selection_dir() / f"fold{fold}" / f"{sample}.npy"


def uni2_feature_path(variant: str, role: str, fold: int, sample: str) -> Path:
    if variant not in VARIANTS or role not in ("oof", "train") or fold not in range(5):
        raise ValueError(f"unsupported output route {variant}/{role}/fold{fold}")
    return FEATURE_ROOT / "uni2" / variant / role / f"fold{fold}" / f"{sample}.pt"


def phikon_feature_path(role: str, fold: int, sample: str) -> Path:
    if role not in ("oof", "train") or fold not in range(5):
        raise ValueError(f"unsupported output route {role}/fold{fold}")
    return FEATURE_ROOT / "phikon" / role / f"fold{fold}" / f"{sample}.pt"


def tissue_grid_path(sample: str) -> Path:
    return FEATURE_ROOT / "tissue_grid" / f"{sample}.pt"


def environment_record() -> dict[str, Any]:
    import torch

    gpu = None
    capability = None
    arch_list: list[str] = []
    if torch.cuda.is_available():
        gpu = torch.cuda.get_device_name(0)
        major, minor = torch.cuda.get_device_capability(0)
        capability = f"sm_{major}{minor}"
        arch_list = list(torch.cuda.get_arch_list())
    return {
        "gpu": gpu,
        "capability": capability,
        "arch_list": arch_list,
        "node": os.environ.get("SLURMD_NODENAME"),
        "job_id": os.environ.get("SLURM_JOB_ID"),
        "array_task_id": os.environ.get("SLURM_ARRAY_TASK_ID"),
        "torch": torch.__version__,
    }


def require_usable_gpu(expected_gpu: str | None = None) -> dict[str, Any]:
    import torch

    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("this stage requires exactly one visible CUDA device")
    major, minor = torch.cuda.get_device_capability(0)
    capability = f"sm_{major}{minor}"
    arch_list = torch.cuda.get_arch_list()
    if capability not in arch_list:
        raise RuntimeError(f"device {capability} is absent from this torch build {arch_list}")
    probe = torch.ones((64, 64), device="cuda")
    if not bool(torch.isfinite(probe @ probe).all()):
        raise RuntimeError("a trivial CUDA matmul did not return finite values")
    name = torch.cuda.get_device_name(0)
    if expected_gpu is not None and name != expected_gpu:
        raise RuntimeError(
            f"this bank was produced on '{expected_gpu}'; this job landed on '{name}'. "
            f"Refusing so the cohort stays on one GPU architecture."
        )
    return environment_record()
