#!/usr/bin/env python3

from __future__ import annotations
import os

import ast
import hashlib
import json
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd


A1 = Path(__file__).resolve().parents[1]
LABEL_SOURCE_ROOT = os.environ.get("LABEL_SOURCE_ROOT")
PROJECT_ROOT = Path(LABEL_SOURCE_ROOT) if LABEL_SOURCE_ROOT else None
LABEL_MAPPING_ID = "k10_fine64_or_refine17_to_refine17_v1"

CANONICAL_CLASSES_17 = (
    "Tumor", "Fibroblast", "NK_T", "Macrophage", "NonMalignant_Parenchymal",
    "Endothelial", "Plasma", "Pericyte/vSMC", "SMC", "B_cell", "Others",
    "Necrosis", "DC", "Neutrophil", "Granulocyte", "Lymphatic_Endothelial", "Schwann",
)
REFINE17_CLASSES = CANONICAL_CLASSES_17
REFINE17_TO_INDEX = {name: index for index, name in enumerate(REFINE17_CLASSES)}


FINE_TO_REFINE17 = {
    "Tumor": "Tumor", "Tumor_ccRCC": "Tumor", "Tumor_pRCC": "Tumor",
    "Tumor_chRCC": "Tumor", "Neuroendocrine": "Tumor",
    "Fibroblast": "Fibroblast", "CAF": "Fibroblast",
    "NK_T": "NK_T", "T_cell": "NK_T", "NK": "NK_T",
    "Macrophage": "Macrophage", "Macrophage_Monocyte": "Macrophage",
    "Microglia": "Macrophage", "Kupffer": "Macrophage",
    "Intestinal_epithelium": "NonMalignant_Parenchymal",
    "Basal": "NonMalignant_Parenchymal",
    "Luminal": "NonMalignant_Parenchymal",
    "Hepatocyte": "NonMalignant_Parenchymal",
    "Acinar": "NonMalignant_Parenchymal",
    "Club": "NonMalignant_Parenchymal",
    "Duct_like": "NonMalignant_Parenchymal",
    "AT2": "NonMalignant_Parenchymal",
    "Epidermal": "NonMalignant_Parenchymal",
    "Dysplasia": "NonMalignant_Parenchymal",
    "Cholangiocyte": "NonMalignant_Parenchymal",
    "Basal/Myoepithelial": "NonMalignant_Parenchymal",
    "Squamous_cell": "NonMalignant_Parenchymal",
    "Islet": "NonMalignant_Parenchymal",
    "Normal_Luminal": "NonMalignant_Parenchymal",
    "Ciliated": "NonMalignant_Parenchymal",
    "Squamous_epithelium": "NonMalignant_Parenchymal",
    "AT1": "NonMalignant_Parenchymal",
    "IPMN": "NonMalignant_Parenchymal",
    "Basal_keratinocyte": "NonMalignant_Parenchymal",
    "Secretory_Epithelium": "NonMalignant_Parenchymal",
    "Astrocyte": "NonMalignant_Parenchymal",
    "NonMalignant_Parenchymal": "NonMalignant_Parenchymal",
    "Endothelial": "Endothelial",
    "Sinusoidal_endothelial": "Endothelial",
    "Plasma": "Plasma",
    "Pericyte/vSMC": "Pericyte/vSMC",
    "Pericyte": "Pericyte/vSMC",
    "vSMC": "Pericyte/vSMC",
    "Smooth_muscle": "SMC",
    "SMC": "SMC",
    "B_cell": "B_cell",
    "Unknown": "Others",
    "Oligodendrocyte": "Others",
    "Neuron": "Others",
    "Skeletal_muscle": "Others",
    "Adipocyte": "Others",
    "Others": "Others",
    "Necrosis": "Necrosis",
    "Necrotic": "Necrosis",
    "cDC2": "DC",
    "Langerhans_cell": "DC",
    "mregDC": "DC",
    "pDC": "DC",
    "cDC1": "DC",
    "cDC": "DC",
    "DC": "DC",
    "Neutrophil": "Neutrophil",
    "Mast": "Granulocyte",
    "Granulocyte": "Granulocyte",
    "Lymphatic_Endothelial": "Lymphatic_Endothelial",
    "Schwann": "Schwann",
}
RAW_TO_REFINE17 = FINE_TO_REFINE17

EVALUATOR14_MAPPING_ID = "refine17_to_evaluator14_v1"
EVALUATOR14_CLASSES = (
    "Tumor", "Fibroblast", "NK_T", "Myeloid_NOS", "NonMalignant_Parenchymal",
    "Endothelial", "Plasma", "Pericyte/vSMC", "SMC", "B_cell", "Necrosis",
    "Mast", "Schwann", "Others",
)
EVALUATOR14_TO_INDEX = {
    name: index for index, name in enumerate(EVALUATOR14_CLASSES)
}
REFINE17_TO_EVALUATOR14 = {
    "Tumor": "Tumor",
    "Fibroblast": "Fibroblast",
    "NK_T": "NK_T",
    "Macrophage": "Myeloid_NOS",
    "DC": "Myeloid_NOS",
    "Neutrophil": "Myeloid_NOS",
    "NonMalignant_Parenchymal": "NonMalignant_Parenchymal",
    "Endothelial": "Endothelial",
    "Lymphatic_Endothelial": "Endothelial",
    "Plasma": "Plasma",
    "Pericyte/vSMC": "Pericyte/vSMC",
    "SMC": "SMC",
    "B_cell": "B_cell",
    "Others": "Others",
    "Necrosis": "Necrosis",
    "Granulocyte": "Mast",
    "Schwann": "Schwann",
}
REFINE17_INDEX_TO_EVALUATOR14_INDEX = np.asarray(
    [
        EVALUATOR14_TO_INDEX[REFINE17_TO_EVALUATOR14[name]]
        for name in REFINE17_CLASSES
    ],
    dtype=np.int64,
)

LABEL_MAPPING_SPEC: dict[str, Any] = {
    "schema_version": "a1.raw_to_refine17_label_contract.v1",
    "label_mapping_id": LABEL_MAPPING_ID,
    "scope": "existing 5k clean177 k10 fine64/refine17 labels only; 477 harmonization excluded",
    "canonical_classes_17": list(CANONICAL_CLASSES_17),
    "raw_to_refine17": dict(sorted(RAW_TO_REFINE17.items())),
    "unmapped_policy": "fatal",
    "drop_policy": "none",
    "provenance": {
        "partition_evidence": {
            "path": "pan_cancer/shared/records/iter0_k10_purity.json",
            "sha256": "0a87ff7b87c57821d7485519f83b40f96c5ec845bc2113d92632fa129f9297a3",
            "finding": "64 fine classes form a strict partition of the 17 refine classes",
        },
        "canonical_shared_producer": {
            "path": "pan_cancer/shared/scripts/train_lora_mlp.py",
            "sha256": "267e4131edd17e19621ebd47bd1532b817a453939eb951417cc0f85d5ee54f12",
            "symbol": "FINE2REFINE",
        },
        "independent_identical_generations": [
            {
                "path": "pan_cancer/v1.1/pancancer/scripts/train_twohead_iter5.py",
                "sha256": "dca83f655c211f97b2f96cec553eb4a85c626463301b8ceb3bfbde23681e5bda",
                "symbol": "FINE2REFINE",
            },
            {
                "path": "pan_cancer/v2/crc/scripts/train_crc_celltype_v2.py",
                "sha256": "d8ed0694308f5a0e700b6a7e203526bd94e86ad0ca016515cdb25fe874ba4036",
                "symbol": "FINE2REFINE",
            },
        ],
    },
}


def _canonical_json(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")


LABEL_MAPPING_SHA256 = hashlib.sha256(_canonical_json(LABEL_MAPPING_SPEC)).hexdigest()

EVALUATOR14_MAPPING_SPEC: dict[str, Any] = {
    "schema_version": "a1.refine17_to_evaluator14_label_contract.v1",
    "label_mapping_id": EVALUATOR14_MAPPING_ID,
    "refine17_classes": list(REFINE17_CLASSES),
    "evaluator14_classes": list(EVALUATOR14_CLASSES),
    "refine17_to_evaluator14": dict(sorted(REFINE17_TO_EVALUATOR14.items())),
    "others_policy": "stored class at index 13; excluded only from scored macro metrics",
    "unmapped_policy": "fatal",
    "drop_policy": "none",
    "provenance": {
        "canonical_shared_producer": {
            "path": "pan_cancer/shared/scripts/train_lora_mlp.py",
            "sha256": "267e4131edd17e19621ebd47bd1532b817a453939eb951417cc0f85d5ee54f12",
            "mapping_symbol": "REFINE2COARSE",
            "class_order_symbol": "COARSE_CLASSES",
        },
        "independent_identical_generations": [
            {
                "path": "pan_cancer/v1.1/pancancer/scripts/train_twohead_iter5.py",
                "sha256": "dca83f655c211f97b2f96cec553eb4a85c626463301b8ceb3bfbde23681e5bda",
                "mapping_symbol": "REFINE2COARSE",
                "class_order_symbol": "COARSE_CLASSES",
            },
            {
                "path": "pan_cancer/v2/crc/scripts/train_crc_celltype_v2.py",
                "sha256": "d8ed0694308f5a0e700b6a7e203526bd94e86ad0ca016515cdb25fe874ba4036",
                "mapping_symbol": "REFINE2COARSE",
                "class_order_symbol": "COARSE_CLASSES",
            },
        ],
    },
}
EVALUATOR14_MAPPING_SHA256 = hashlib.sha256(
    _canonical_json(EVALUATOR14_MAPPING_SPEC)
).hexdigest()

LABEL_CONTRACT_ID = "a1_raw_refine17_evaluator14_v1"
LABEL_CONTRACT_SPEC: dict[str, Any] = {
    "schema_version": "a1.label_contract.v1",
    "label_contract_id": LABEL_CONTRACT_ID,
    "raw_to_refine17": {
        "label_mapping_id": LABEL_MAPPING_ID,
        "label_mapping_sha256": LABEL_MAPPING_SHA256,
    },
    "refine17_to_evaluator14": {
        "label_mapping_id": EVALUATOR14_MAPPING_ID,
        "label_mapping_sha256": EVALUATOR14_MAPPING_SHA256,
    },
}
LABEL_CONTRACT_SHA256 = hashlib.sha256(_canonical_json(LABEL_CONTRACT_SPEC)).hexdigest()


def validate_contract() -> None:
    canonical = set(CANONICAL_CLASSES_17)
    targets = set(RAW_TO_REFINE17.values())
    if len(CANONICAL_CLASSES_17) != 17 or len(canonical) != 17:
        raise RuntimeError("canonical refine class contract is not exactly 17 unique classes")
    if targets != canonical:
        raise RuntimeError(
            f"raw mapping targets differ from canonical refine17: "
            f"missing={sorted(canonical-targets)} extra={sorted(targets-canonical)}"
        )
    for name in CANONICAL_CLASSES_17:
        if RAW_TO_REFINE17.get(name) != name:
            raise RuntimeError(f"canonical identity mapping missing or changed for {name}")
    if len(RAW_TO_REFINE17) != 66:
        raise RuntimeError(f"inherited mapping must contain exactly 66 keys, got {len(RAW_TO_REFINE17)}")
    evaluator = set(EVALUATOR14_CLASSES)
    if len(EVALUATOR14_CLASSES) != 14 or len(evaluator) != 14:
        raise RuntimeError("evaluator class contract is not exactly 14 unique classes")
    if set(REFINE17_TO_EVALUATOR14) != canonical:
        raise RuntimeError("refine17 -> evaluator14 mapping does not cover exactly refine17")
    if set(REFINE17_TO_EVALUATOR14.values()) != evaluator:
        raise RuntimeError("refine17 -> evaluator14 targets differ from evaluator14 order")
    expected_indices = np.asarray(
        [EVALUATOR14_TO_INDEX[REFINE17_TO_EVALUATOR14[name]] for name in REFINE17_CLASSES],
        dtype=np.int64,
    )
    if not np.array_equal(REFINE17_INDEX_TO_EVALUATOR14_INDEX, expected_indices):
        raise RuntimeError("refine17 -> evaluator14 index lookup is inconsistent")


def canonicalize_raw_labels(
    values: Iterable[Any] | pd.Series | np.ndarray,
    *,
    source: str,
) -> np.ndarray:
    series = values if isinstance(values, pd.Series) else pd.Series(values, copy=False)
    raw = series.astype(str)
    observed = set(raw.unique().tolist())
    unknown = sorted(observed - set(RAW_TO_REFINE17))
    if unknown:
        raise ValueError(
            f"{source} has {len(unknown)} labels outside frozen {LABEL_MAPPING_ID}: {unknown}"
        )
    mapped = raw.map(RAW_TO_REFINE17)
    if mapped.isna().any():
        raise RuntimeError(f"{source} mapping produced missing canonical labels")
    output = mapped.to_numpy(dtype=str, copy=True)
    if set(np.unique(output)) - set(CANONICAL_CLASSES_17):
        raise RuntimeError(f"{source} mapping produced a non-refine17 target")
    return output


def refine17_indices(
    values: Iterable[Any] | pd.Series | np.ndarray,
    *,
    source: str,
) -> np.ndarray:
    labels = canonicalize_raw_labels(values, source=source)
    return np.fromiter(
        (REFINE17_TO_INDEX[label] for label in labels),
        dtype=np.int64,
        count=len(labels),
    )


def evaluator14_labels_from_refine17(
    values: Iterable[Any] | pd.Series | np.ndarray,
    *,
    source: str,
) -> np.ndarray:
    series = values if isinstance(values, pd.Series) else pd.Series(values, copy=False)
    refine = series.astype(str)
    observed = set(refine.unique().tolist())
    unknown = sorted(observed - set(REFINE17_TO_EVALUATOR14))
    if unknown:
        raise ValueError(
            f"{source} has {len(unknown)} labels outside frozen "
            f"{EVALUATOR14_MAPPING_ID} refine17 domain: {unknown}"
        )
    mapped = refine.map(REFINE17_TO_EVALUATOR14)
    if mapped.isna().any():
        raise RuntimeError(f"{source} mapping produced missing evaluator14 labels")
    output = mapped.to_numpy(dtype=str, copy=True)
    if set(np.unique(output)) - set(EVALUATOR14_CLASSES):
        raise RuntimeError(f"{source} mapping produced a non-evaluator14 target")
    return output


def evaluator14_indices_from_refine17(
    values: Iterable[Any] | pd.Series | np.ndarray,
    *,
    source: str,
) -> np.ndarray:
    labels = evaluator14_labels_from_refine17(values, source=source)
    return np.fromiter(
        (EVALUATOR14_TO_INDEX[label] for label in labels),
        dtype=np.int64,
        count=len(labels),
    )


def canonicalize_raw_to_evaluator14(
    values: Iterable[Any] | pd.Series | np.ndarray,
    *,
    source: str,
) -> np.ndarray:
    refine = canonicalize_raw_labels(values, source=source)
    return evaluator14_labels_from_refine17(refine, source=f"{source} [refine17]")


def evaluator14_indices_from_raw(
    values: Iterable[Any] | pd.Series | np.ndarray,
    *,
    source: str,
) -> np.ndarray:
    labels = canonicalize_raw_to_evaluator14(values, source=source)
    return np.fromiter(
        (EVALUATOR14_TO_INDEX[label] for label in labels),
        dtype=np.int64,
        count=len(labels),
    )


def raw_and_canonical_counts(
    values: Iterable[Any] | pd.Series | np.ndarray,
    *,
    source: str,
) -> tuple[dict[str, int], dict[str, int]]:
    series = values if isinstance(values, pd.Series) else pd.Series(values, copy=False)
    raw = series.astype(str)
    mapped = canonicalize_raw_labels(raw, source=source)
    raw_counts = {str(key): int(value) for key, value in raw.value_counts().sort_index().items()}
    canonical_counts = {
        str(key): int(value)
        for key, value in pd.Series(mapped).value_counts().sort_index().items()
    }
    return raw_counts, canonical_counts


def _literal_assignment(path: Path, symbol: str) -> Any:
    tree = ast.parse(path.read_text(), filename=str(path))
    for node in tree.body:
        if not isinstance(node, (ast.Assign, ast.AnnAssign)):
            continue
        targets = node.targets if isinstance(node, ast.Assign) else [node.target]
        if any(isinstance(target, ast.Name) and target.id == symbol for target in targets):
            return ast.literal_eval(node.value)
    raise RuntimeError(f"{path}: frozen source symbol {symbol!r} not found")


def verify_source_bindings() -> dict[str, Any]:
    if PROJECT_ROOT is None:
        return {"status": "not_checked",
                "reason": "LABEL_SOURCE_ROOT is not set; the inherited mapping sources are research files outside this package"}
    sources: dict[str, dict[str, Any]] = {}
    raw_provenance = LABEL_MAPPING_SPEC["provenance"]
    partition = raw_provenance["partition_evidence"]
    partition_path = PROJECT_ROOT / partition["path"]
    actual_partition_sha = _file_sha256(partition_path)
    if actual_partition_sha != partition["sha256"]:
        raise RuntimeError(
            f"partition evidence hash mismatch: {partition_path}: "
            f"expected {partition['sha256']}, got {actual_partition_sha}"
        )
    sources[partition["path"]] = {
        "sha256": actual_partition_sha,
        "binding": "fine64 strict-partition evidence",
    }

    raw_sources = [raw_provenance["canonical_shared_producer"]]
    raw_sources.extend(raw_provenance["independent_identical_generations"])
    evaluator_provenance = EVALUATOR14_MAPPING_SPEC["provenance"]
    evaluator_sources = [evaluator_provenance["canonical_shared_producer"]]
    evaluator_sources.extend(evaluator_provenance["independent_identical_generations"])

    for record in raw_sources:
        path = PROJECT_ROOT / record["path"]
        actual_sha = _file_sha256(path)
        if actual_sha != record["sha256"]:
            raise RuntimeError(
                f"inherited mapping source hash mismatch: {path}: "
                f"expected {record['sha256']}, got {actual_sha}"
            )
        inherited = _literal_assignment(path, record["symbol"])
        if inherited != FINE_TO_REFINE17:
            raise RuntimeError(f"{path}:{record['symbol']} differs from frozen mapping")
        sources.setdefault(record["path"], {"sha256": actual_sha, "bindings": []})
        sources[record["path"]].setdefault("bindings", []).append(record["symbol"])

    for record in evaluator_sources:
        path = PROJECT_ROOT / record["path"]
        actual_sha = _file_sha256(path)
        if actual_sha != record["sha256"]:
            raise RuntimeError(
                f"inherited evaluator source hash mismatch: {path}: "
                f"expected {record['sha256']}, got {actual_sha}"
            )
        inherited_mapping = _literal_assignment(path, record["mapping_symbol"])
        inherited_order = tuple(_literal_assignment(path, record["class_order_symbol"]))
        if inherited_mapping != REFINE17_TO_EVALUATOR14:
            raise RuntimeError(
                f"{path}:{record['mapping_symbol']} differs from frozen evaluator mapping"
            )
        if inherited_order != EVALUATOR14_CLASSES:
            raise RuntimeError(
                f"{path}:{record['class_order_symbol']} differs from evaluator class order"
            )
        sources.setdefault(record["path"], {"sha256": actual_sha, "bindings": []})
        sources[record["path"]].setdefault("bindings", []).extend(
            [record["mapping_symbol"], record["class_order_symbol"]]
        )

    return {
        "status": "verified",
        "project_relative_sources": sources,
    }


def contract_manifest_record(*, verify_sources: bool = True) -> dict[str, Any]:
    return {
        "label_mapping_id": LABEL_MAPPING_ID,
        "label_mapping_sha256": LABEL_MAPPING_SHA256,
        "contract_path": str(Path(__file__).resolve()),
        "contract_file_sha256": _file_sha256(Path(__file__)),
        "n_raw_or_refine_keys": len(RAW_TO_REFINE17),
        "n_canonical_classes": len(CANONICAL_CLASSES_17),
        "unmapped_policy": "fatal",
        "drop_policy": "none",
        "source_provenance": LABEL_MAPPING_SPEC["provenance"],
        "source_binding": verify_source_bindings() if verify_sources else "not_checked",
    }


def evaluator14_contract_manifest_record(*, verify_sources: bool = True) -> dict[str, Any]:
    return {
        "label_mapping_id": EVALUATOR14_MAPPING_ID,
        "label_mapping_sha256": EVALUATOR14_MAPPING_SHA256,
        "contract_path": str(Path(__file__).resolve()),
        "contract_file_sha256": _file_sha256(Path(__file__)),
        "n_refine17_keys": len(REFINE17_TO_EVALUATOR14),
        "n_evaluator_classes": len(EVALUATOR14_CLASSES),
        "unmapped_policy": "fatal",
        "drop_policy": "none",
        "source_provenance": EVALUATOR14_MAPPING_SPEC["provenance"],
        "source_binding": verify_source_bindings() if verify_sources else "not_checked",
    }


def label_contract_manifest_record(*, verify_sources: bool = True) -> dict[str, Any]:
    return {
        "label_contract_id": LABEL_CONTRACT_ID,
        "label_contract_sha256": LABEL_CONTRACT_SHA256,
        "raw_to_refine17": contract_manifest_record(verify_sources=verify_sources),
        "refine17_to_evaluator14": evaluator14_contract_manifest_record(
            verify_sources=verify_sources
        ),
    }


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


validate_contract()
