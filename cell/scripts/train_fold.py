
from __future__ import annotations

import argparse
import contextlib
import dataclasses
import hashlib
import json
import math
import os
import random
import tempfile
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd
import torch
from torch import Tensor, nn
from torch.nn import functional as F

VERSION_ROOT = Path(__file__).resolve().parents[1]
_RELEASE = VERSION_ROOT.parent

IDENTITY_CLASSES = (
    "Tumor",
    "Fibroblast",
    "NK_T",
    "Myeloid_NOS",
    "NonMalignant_Parenchymal",
    "Endothelial",
    "Plasma",
    "SMC",
    "B_cell",
    "Schwann",
    "Others",
)
SCORED_IDENTITY_COUNT = 10
LINEAGE_CLASSES = (
    "Epithelial",
    "Stromal",
    "Lymphoid",
    "Myeloid",
    "Endothelial",
    "Other",
)
IGNORE_INDEX = -100
NECROSIS_GROUP = len(IDENTITY_CLASSES)


OOF_SCHEMA = "a4.sample_oof.v2"
CHECKPOINT_SCHEMA = "a4.fold_checkpoint.v1"
SELECTION_SCHEMA = "a4.patient_balanced_selection.v1"
CACHE_SCHEMA = "a4.selected_feature_cache.v2"


REQUIRED_CHECKPOINT_KEYS = (
    "fold",
    "identity_classes",
    "lineage_classes",
    "model_config",
    "model_state",
    "provenance",
    "schema_version",
    "selection_sha256",
)
REQUIRED_SELECTION_KEYS = (
    "fold",
    "provenance",
    "schema_version",
    "selected",
    "selection_sha256",
)
REQUIRED_CACHE_KEYS = (
    "cache_provenance",
    "cancer",
    "cell",
    "fold",
    "grid",
    "grid_row_index",
    "identity_known_mask",
    "patient",
    "phikon",
    "sample",
    "schema_version",
    "uni2",
    "y",
)
REQUIRED_OOF_KEYS = (
    "cancer",
    "cell",
    "checkpoint_sha256",
    "config_sha256",
    "data_contract_sha256",
    "experiment_id",
    "fold",
    "identity_classes",
    "identity_known_mask",
    "patient",
    "prob",
    "sample",
    "schema_version",
    "source_sha256",
    "taxonomy_sha256",
    "y",
)

SELECTION_POLICY = "v7.post_merge_prior_repair_boundary_capped.v1"
ROW_IDENTITY_SCHEMA = "v5.shared_row_identity.v1"
FOLD_SUMMARY_SCHEMA = "v5.fold_summary.v1"
EXPERIMENT_SCHEMA = "v5.experiment.v1"
TAXONOMY_SCHEMA = "v5-taxonomy-coarse11-v1"


ARM_VOCABULARIES = ("refine17", "raw25")


def _canonical_json(value: Any) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")


def sha256_json(value: Any) -> str:
    return hashlib.sha256(_canonical_json(value)).hexdigest()


def sha256_file(path: Path, block_size: int = 8 << 20) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while True:
            block = handle.read(block_size)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def _ordered_string_sha256(values: Sequence[str]) -> str:
    digest = hashlib.sha256()
    for value in values:
        encoded = str(value).encode("utf-8")
        digest.update(len(encoded).to_bytes(8, "little"))
        digest.update(encoded)
    return digest.hexdigest()


def _fsync_parent(path: Path) -> None:
    handle = os.open(path.parent, os.O_RDONLY)
    try:
        os.fsync(handle)
    finally:
        os.close(handle)


def _atomic_write(path: Path, writer) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(dir=str(path.parent), prefix=path.name)
    os.close(descriptor)
    temporary_path = Path(temporary)
    try:
        with temporary_path.open("wb") as handle:
            writer(handle)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_path, path)
        _fsync_parent(path)
    finally:
        if temporary_path.exists():
            temporary_path.unlink()


def atomic_json(path: Path, value: Any) -> None:
    _atomic_write(
        path,
        lambda handle: handle.write(
            json.dumps(value, indent=2, sort_keys=True).encode("utf-8") + b"\n"
        ),
    )


def atomic_torch_save(path: Path, value: Any) -> None:
    _atomic_write(path, lambda handle: torch.save(value, handle))


def require_gen3_keys(payload: Mapping[str, Any], required: Sequence[str], what: str) -> None:
    missing = [key for key in required if key not in payload]
    if missing:
        raise ValueError(
            f"{what} payload lacks fields the gen3 decoder reads: {missing}"
        )


def atomic_npz(path: Path, **arrays: Any) -> None:
    _atomic_write(
        path, lambda handle: np.savez_compressed(handle, **arrays)
    )


def _ensure_inside_version_root(path: Path) -> Path:
    resolved = Path(path).resolve()
    if not resolved.is_relative_to(VERSION_ROOT):
        raise ValueError(f"refusing to write outside {VERSION_ROOT}: {resolved}")
    return resolved


def _resolve_input(path_value: str | Path, workspace: Path) -> Path:
    candidate = Path(path_value)
    return candidate if candidate.is_absolute() else (workspace / candidate)


def _resolve_output(path_value: str | Path, workspace: Path) -> Path:
    return _ensure_inside_version_root(_resolve_input(path_value, workspace))


def load_experiment(path: Path) -> tuple[dict[str, Any], Path]:
    path = Path(path).resolve()
    with path.open("r", encoding="utf-8") as handle:
        config = json.load(handle)
    if config.get("schema_version") != EXPERIMENT_SCHEMA:
        raise ValueError(f"experiment schema mismatch in {path}")
    os.environ.setdefault("RELEASE_ROOT", str(_RELEASE))
    workspace = Path(os.path.expandvars(config["workspace_root"])).resolve()
    if not workspace.is_dir():
        raise FileNotFoundError(f"workspace_root does not exist: {workspace}")
    if not (workspace / config["version_root"]).resolve() == VERSION_ROOT:
        raise ValueError("config version_root does not resolve to this script's version root")
    arm_vocabulary(config)
    return config, workspace


@dataclasses.dataclass(frozen=True)
class Vocabulary:
    name: str
    labels: tuple[str, ...]
    source_to_identity: Mapping[str, str]
    identity_supervision_mask: Mapping[str, bool]


@dataclasses.dataclass(frozen=True)
class Taxonomy:
    identity_classes: tuple[str, ...]
    lineage_classes: tuple[str, ...]
    identity_to_lineage: Mapping[str, str]
    vocabularies: Mapping[str, Vocabulary]
    necrosis_positive_source_label: str

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "Taxonomy":
        if value.get("schema_version") != TAXONOMY_SCHEMA:
            raise ValueError("taxonomy schema mismatch")
        identities = tuple(value["identity_classes"])
        if identities != IDENTITY_CLASSES:
            raise ValueError("taxonomy identity_classes must match the fixed v5 order")
        lineage_target = value["lineage_auxiliary_target"]
        lineages = tuple(lineage_target["lineage_classes"])
        if lineages != LINEAGE_CLASSES:
            raise ValueError("taxonomy lineage_classes must match the fixed v5 order")
        identity_to_lineage = dict(lineage_target["identity_to_lineage"])
        for identity in identities:
            if identity_to_lineage.get(identity) not in lineages:
                raise ValueError(f"missing or unknown lineage for {identity!r}")
        vocabularies: dict[str, Vocabulary] = {}
        for name, spec in value["source_vocabularies"].items():
            labels = tuple(spec["labels"])
            mapping = dict(spec["source_to_identity"])
            mask = {key: bool(flag) for key, flag in spec["identity_supervision_mask"].items()}
            if set(mapping) != set(labels) or set(mask) != set(labels):
                raise ValueError(f"vocabulary {name!r} tables do not close over its labels")
            for source, target in mapping.items():
                if target not in identities:
                    raise ValueError(f"{name}: unknown identity target {target!r} for {source!r}")
            necrosis = value["necrosis_annotation"]["positive_source_label"]
            if necrosis not in labels or mask[necrosis]:
                raise ValueError(f"{name}: Necrosis must be present and identity-unsupervised")
            for source, flag in mask.items():
                if flag is False and source != necrosis:
                    raise ValueError(f"{name}: unexpected unsupervised source {source!r}")
            vocabularies[str(name)] = Vocabulary(str(name), labels, mapping, mask)
        for required in ("refine17", "raw25"):
            if required not in vocabularies:
                raise ValueError(f"taxonomy lacks the {required!r} source vocabulary")
        return cls(
            identities,
            lineages,
            identity_to_lineage,
            vocabularies,
            str(value["necrosis_annotation"]["positive_source_label"]),
        )

    def identity_index(self) -> dict[str, int]:
        return {name: index for index, name in enumerate(self.identity_classes)}

    def target_codes(self, vocabulary: str) -> tuple[dict[str, int], dict[str, bool]]:
        vocab = self.vocabularies[vocabulary]
        index = self.identity_index()
        return (
            {source: index[target] for source, target in vocab.source_to_identity.items()},
            dict(vocab.identity_supervision_mask),
        )


def _lookup_per_label(
    labels: Sequence[str], table: Mapping[str, Any], *, dtype: Any, what: str
) -> np.ndarray:
    values = np.asarray(labels, dtype=object)
    unique, inverse = np.unique(values, return_inverse=True)
    missing = [str(name) for name in unique.tolist() if str(name) not in table]
    if missing:
        raise ValueError(f"unmapped {what} source label(s): {sorted(missing)[:5]}")
    resolved = np.asarray([table[str(name)] for name in unique.tolist()], dtype=dtype)
    return resolved[inverse]


def encode_labels(
    labels: Sequence[str], taxonomy: Taxonomy, vocabulary: str
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    target, supervised = taxonomy.target_codes(vocabulary)
    lineage_index = {name: index for index, name in enumerate(taxonomy.lineage_classes)}
    identity_lineage = np.asarray(
        [
            lineage_index[taxonomy.identity_to_lineage[name]]
            for name in taxonomy.identity_classes
        ],
        dtype=np.int64,
    )
    n = len(labels)
    if n == 0:
        return (
            np.zeros(0, dtype=np.int64),
            np.zeros(0, dtype=np.bool_),
            np.zeros(0, dtype=np.int64),
            np.zeros(0, dtype=np.int8),
        )
    code = _lookup_per_label(labels, target, dtype=np.int64, what=vocabulary)
    known = _lookup_per_label(labels, supervised, dtype=np.bool_, what=vocabulary)
    is_necrosis = (
        np.asarray(labels, dtype=object) == taxonomy.necrosis_positive_source_label
    )
    supervised_row = known & ~is_necrosis
    y = np.where(supervised_row, code, IGNORE_INDEX).astype(np.int64)
    identity_known = supervised_row.astype(np.bool_)
    lineage_y = np.where(
        supervised_row, identity_lineage[np.clip(code, 0, len(identity_lineage) - 1)],
        IGNORE_INDEX,
    ).astype(np.int64)
    necrosis_target = is_necrosis.astype(np.int8)
    return y, identity_known, lineage_y, necrosis_target


@dataclasses.dataclass(frozen=True)
class SampleSpec:
    sample: str
    patient: str
    cancer: str
    fold: int
    necrosis_known: bool


def load_contract_partition(
    contract: Mapping[str, Any], fold: int
) -> tuple[list[SampleSpec], list[SampleSpec], tuple[str, ...]]:
    records = list(contract["cohort"]["sample_records"])
    observed = set(
        str(name)
        for name in contract["necrosis_annotation_boundary"]["annotation_observed_samples"]
    )
    specs: dict[str, SampleSpec] = {}
    for record in records:
        sample = str(record["sample"])
        if sample in specs:
            raise ValueError(f"duplicate sample record: {sample}")
        explicit = record.get("necrosis_annotation_observed")
        if explicit is None:
            known = sample in observed
        else:
            known = bool(explicit)
            if known != (sample in observed):
                raise ValueError(f"necrosis coverage disagreement for {sample}")
        specs[sample] = SampleSpec(
            sample=sample,
            patient=str(record["patient"]),
            cancer=str(record["cancer"]),
            fold=int(record["fold"]),
            necrosis_known=known,
        )
    allowlist = tuple(sorted(specs))
    declared = contract["cohort"]["included_samples"]
    if int(declared) != len(allowlist):
        raise ValueError(f"declared included_samples={declared}, records={len(allowlist)}")


    declared_patients = contract["cohort"]["included_patients"]
    if int(declared_patients) != len({spec.patient for spec in specs.values()}):
        raise ValueError(
            f"declared included_patients={declared_patients}, records carry "
            f"{len({spec.patient for spec in specs.values()})}"
        )
    if fold not in range(5):
        raise ValueError("outer fold must be in [0, 4]")

    fold_records = {int(record["fold"]): record for record in contract["folds"]["records"]}
    if set(fold_records) != set(range(5)):
        raise ValueError("contract fold records do not cover folds 0-4")
    test_names = sorted(str(name) for name in fold_records[fold]["samples"])
    train_names = sorted(name for name in allowlist if name not in set(test_names))
    if set(train_names) & set(test_names):
        raise ValueError("outer train and test sets overlap")
    if set(train_names) | set(test_names) != set(allowlist):
        raise ValueError("outer partition does not close over the allowlist")
    for name in test_names:
        if specs[name].fold != fold:
            raise ValueError(f"fold manifest disagreement for held-out sample {name}")
    for name in train_names:
        if specs[name].fold == fold:
            raise ValueError(f"fold manifest disagreement for train sample {name}")
    train_patients = {specs[name].patient for name in train_names}
    test_patients = {specs[name].patient for name in test_names}
    if train_patients & test_patients:
        raise ValueError("outer partition is not patient-disjoint")
    return (
        [specs[name] for name in train_names],
        [specs[name] for name in test_names],
        allowlist,
    )


def _load_pt_dict(path: Path) -> dict[str, Any]:
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(path)
    value = torch.load(path, map_location="cpu", weights_only=False, mmap=True)
    if not isinstance(value, dict):
        raise ValueError(f"expected dictionary in {path}")
    return value


def _as_string_list(values: Sequence[Any]) -> list[str]:
    return [
        value.decode("utf-8") if isinstance(value, bytes) else str(value)
        for value in values
    ]


def _cell_artifact(path: Path, expected_dim: int) -> tuple[Tensor, list[str], list[str]]:
    value = _load_pt_dict(path)
    features = value.get("features")
    cell_ids = _as_string_list(value.get("cell_ids", ()))
    labels = _as_string_list(value.get("cell_type", ()))
    if not isinstance(features, Tensor) or features.ndim != 2:
        raise ValueError(f"invalid features tensor in {path}")
    if int(features.shape[1]) != int(expected_dim):
        raise ValueError(
            f"{path}: expected feature dim {expected_dim}, found {int(features.shape[1])}"
        )
    if features.shape[0] != len(cell_ids) or len(cell_ids) != len(labels):
        raise ValueError(f"row identity/label length mismatch in {path}")
    if len(set(cell_ids)) != len(cell_ids):
        raise ValueError(f"duplicate cell IDs in {path}")
    return features, cell_ids, labels


def _grid_artifact(path: Path, expected_dim: int) -> tuple[Tensor, list[str]]:
    value = _load_pt_dict(path)
    features = value.get("features")
    cell_ids = _as_string_list(value.get("cell_ids", ()))
    if not isinstance(features, Tensor) or features.ndim != 2:
        raise ValueError(f"invalid grid tensor in {path}")
    if int(features.shape[1]) != int(expected_dim):
        raise ValueError(
            f"{path}: expected grid dim {expected_dim}, found {int(features.shape[1])}"
        )
    if features.shape[0] != len(cell_ids):
        raise ValueError(f"grid row identity mismatch in {path}")
    if len(set(cell_ids)) != len(cell_ids):
        raise ValueError(f"duplicate cell IDs in {path}")
    return features, cell_ids


def _stems(directory: Path) -> set[str]:
    directory = Path(directory)
    if not directory.is_dir():
        raise FileNotFoundError(directory)
    return {path.stem for path in directory.glob("*.pt")}


def _require_file(path: Path) -> Path:
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(path)
    return path


def row_identity_roots(config: Mapping[str, Any], workspace: Path) -> dict[str, Path]:
    block = config["inputs"]["row_identity"]
    return {
        key: _resolve_input(block[key], workspace)
        for key in (
            "frame_uni2_root",
            "frame_phikon_root",
            "head_selection_root",
            "frozen_uni2_root",
            "frozen_phikon_root",
            "tissue_grid_root",
        )
    }


def role_dir(root: Path, block: Mapping[str, Any], role: str, fold: int, prefix: str) -> Path:
    if role not in ("train", "eval"):
        raise ValueError(f"unknown role {role!r}")
    return root / str(block[f"{prefix}_{role}_subdir"]).format(fold=fold)


def _root_list(value: Any, workspace: Path) -> list[Path]:
    values = value if isinstance(value, (list, tuple)) else [value]
    if not values:
        raise ValueError("an arm's feature root list is empty")
    return [_resolve_input(item, workspace) for item in values]


def arm_feature_paths(
    config: Mapping[str, Any],
    workspace: Path,
    *,
    backbone: str,
    sample: str,
    role: str,
    fold: int,
) -> list[Path]:
    block = config["inputs"]["cell_features"]
    roots = _root_list(block[f"{backbone}_root"], workspace)
    mode = str(block["mode"])
    if mode == "frozen":
        return [root / f"{sample}.pt" for root in roots]
    if mode == "fold_aware":
        return [
            role_dir(root, block, role, fold, backbone) / f"{sample}.pt"
            for root in roots
        ]
    raise ValueError(f"unknown cell feature mode {mode!r}")


def arm_part_dims(config: Mapping[str, Any], backbone: str, total_dim: int) -> list[int]:
    block = config["inputs"]["cell_features"]
    key = f"{backbone}_part_dims"
    n_roots = len(block[f"{backbone}_root"]) if isinstance(
        block[f"{backbone}_root"], (list, tuple)
    ) else 1
    if key in block:
        dims = [int(value) for value in block[key]]
    else:
        if n_roots != 1:
            raise ValueError(
                f"inputs.cell_features.{key} is required when {backbone}_root lists "
                f"{n_roots} roots"
            )
        dims = [int(total_dim)]
    if len(dims) != n_roots:
        raise ValueError(f"{key} lists {len(dims)} widths for {n_roots} roots")
    if sum(dims) != int(total_dim):
        raise ValueError(
            f"{key} sums to {sum(dims)} but model.{backbone}_dim is {total_dim}"
        )
    return dims


def _load_slot(
    paths: Sequence[Path],
    part_dims: Sequence[int],
    cells: Sequence[str],
) -> tuple[list[Tensor], np.ndarray, list[str]]:
    tensors: list[Tensor] = []
    reference_ids: list[str] | None = None
    labels: list[str] = []
    for path, dim in zip(paths, part_dims, strict=True):
        features, ids, part_labels = _cell_artifact(_require_file(path), int(dim))
        if reference_ids is None:
            reference_ids, labels = ids, part_labels
        elif ids != reference_ids:
            raise RuntimeError(
                f"{path}: row identity differs from {paths[0]}; the parts of one "
                "slot must be readouts of the same rows"
            )
        tensors.append(features)
    assert reference_ids is not None
    rows = _index_rows_by_ids(cells, reference_ids, paths[0])
    return tensors, rows, labels


def arm_vocabulary(config: Mapping[str, Any]) -> str:
    try:
        block = config["inputs"]["cell_features"]
    except KeyError as error:
        raise ValueError("config has no inputs.cell_features block") from error
    if "vocabulary" not in block:
        raise ValueError(
            "inputs.cell_features.vocabulary is required: an arm must declare the "
            f"source label set of its own feature bank, one of {list(ARM_VOCABULARIES)}"
        )
    name = str(block["vocabulary"])
    if name not in ARM_VOCABULARIES:
        raise ValueError(
            f"unknown arm feature vocabulary {name!r}; expected one of "
            f"{list(ARM_VOCABULARIES)}"
        )
    return name


def assert_frame_label_agreement(
    *,
    sample: str,
    observed: Sequence[str],
    expected: Sequence[str],
    vocabulary: str,
    taxonomy: Taxonomy,
) -> None:
    if vocabulary not in taxonomy.vocabularies:
        raise ValueError(f"taxonomy lacks the {vocabulary!r} source vocabulary")
    if len(observed) != len(expected):
        raise RuntimeError(
            f"{sample}: arm label vector has {len(observed)} rows, the shared "
            f"frame has {len(expected)}"
        )
    if not observed:
        return
    arm_target, arm_supervised = taxonomy.target_codes(vocabulary)
    frame_target, frame_supervised = taxonomy.target_codes("refine17")
    observed_code = _lookup_per_label(observed, arm_target, dtype=np.int64, what=vocabulary)
    observed_flag = _lookup_per_label(
        observed, arm_supervised, dtype=np.bool_, what=vocabulary
    )
    expected_code = _lookup_per_label(
        expected, frame_target, dtype=np.int64, what="refine17"
    )
    expected_flag = _lookup_per_label(
        expected, frame_supervised, dtype=np.bool_, what="refine17"
    )
    disagree = np.flatnonzero(
        (observed_code != expected_code) | (observed_flag != expected_flag)
    )
    if disagree.size:
        row = int(disagree[0])
        raise RuntimeError(
            f"{sample}: arm feature labels disagree with the shared frame on "
            f"{int(disagree.size)} rows; first row {row}: "
            f"{vocabulary}={observed[row]!r} -> {int(observed_code[row])}"
            f"/{bool(observed_flag[row])}, refine17={expected[row]!r} -> "
            f"{int(expected_code[row])}/{bool(expected_flag[row])}"
        )


def _shared_scratch(config: Mapping[str, Any], workspace: Path) -> Path:
    return _resolve_input(config["inputs"]["row_identity"]["shared_scratch"], workspace)


def row_identity_paths(config: Mapping[str, Any], workspace: Path, fold: int) -> tuple[Path, Path]:
    shared = _shared_scratch(config, workspace)
    return (
        shared / f"fold{fold}" / "row_identity.pt",
        _resolve_output(config["outputs"]["manifests"], workspace)
        / f"row_identity_fold{fold}.json",
    )


def _intersect_frame(
    *,
    sample: str,
    primary_ids: Sequence[str],
    primary_labels: Sequence[str],
    phikon_ids: Sequence[str],
    phikon_labels: Sequence[str],
    frozen_ids: Sequence[str],
    frozen_labels: Sequence[str],
    grid_ids: Sequence[str],
    taxonomy: Taxonomy,
) -> tuple[list[str], list[str], dict[str, int]]:
    phikon_position = {cell: row for row, cell in enumerate(phikon_ids)}
    frozen_position = {cell: row for row, cell in enumerate(frozen_ids)}
    grid_position = {cell: row for row, cell in enumerate(grid_ids)}
    primary_label_of = dict(zip(primary_ids, primary_labels, strict=True))
    kept: list[str] = []
    missing_phikon = 0
    missing_frozen = 0
    missing_grid = 0
    for cell in primary_ids:
        if cell not in phikon_position:
            missing_phikon += 1
            continue
        if cell not in frozen_position:
            missing_frozen += 1
            continue
        if cell not in grid_position:
            missing_grid += 1
            continue
        kept.append(cell)
    kept.sort()
    refine_target, refine_supervised = taxonomy.target_codes("refine17")
    raw_target, raw_supervised = taxonomy.target_codes("raw25")
    labels = [primary_label_of[cell] for cell in kept]
    counterparts = [frozen_labels[frozen_position[cell]] for cell in kept]
    if kept:
        refine_code = _lookup_per_label(labels, refine_target, dtype=np.int64, what="refine17")
        refine_flag = _lookup_per_label(
            labels, refine_supervised, dtype=np.bool_, what="refine17"
        )
        raw_code = _lookup_per_label(counterparts, raw_target, dtype=np.int64, what="raw25")
        raw_flag = _lookup_per_label(
            counterparts, raw_supervised, dtype=np.bool_, what="raw25"
        )


        disagree = np.flatnonzero((refine_code != raw_code) | (refine_flag != raw_flag))
        reannotated = int(disagree.size)
        first_reannotation = (
            {
                "cell": kept[int(disagree[0])],
                "frame_refine17": labels[int(disagree[0])],
                "frozen_raw25": counterparts[int(disagree[0])],
            }
            if disagree.size
            else None
        )
    else:
        reannotated = 0
        first_reannotation = None
    phikon_on_frame = [phikon_labels[phikon_position[cell]] for cell in kept]
    if kept:
        phikon_code = _lookup_per_label(
            phikon_on_frame, refine_target, dtype=np.int64, what="refine17"
        )
        phikon_flag = _lookup_per_label(
            phikon_on_frame, refine_supervised, dtype=np.bool_, what="refine17"
        )
        phikon_target_differences = int(
            np.flatnonzero(
                (refine_code != phikon_code) | (refine_flag != phikon_flag)
            ).size
        )
    else:
        phikon_target_differences = 0
    accounting = {
        "n_primary": int(len(primary_ids)),
        "n_phikon": int(len(phikon_ids)),
        "n_frozen": int(len(frozen_ids)),
        "n_grid": int(len(grid_ids)),
        "n_common": int(len(kept)),
        "dropped_absent_from_phikon": int(missing_phikon),
        "dropped_absent_from_frozen": int(missing_frozen),
        "dropped_absent_from_tissue_grid": int(missing_grid),
        "reannotated_vs_frozen_identity_target": reannotated,
        "reannotated_vs_phikon_identity_target": phikon_target_differences,
        "reannotated_vs_phikon_source_string": int(
            sum(
                1
                for observed, stored in zip(labels, phikon_on_frame, strict=True)
                if observed != stored
            )
        ),
    }
    if first_reannotation is not None:
        accounting["first_reannotation"] = first_reannotation
    return kept, labels, accounting


def build_row_identity(config_path: Path, fold: int, *, resume: bool = False) -> Path:
    config, workspace = load_experiment(Path(config_path))
    shared = _shared_scratch(config, workspace)
    if not shared.resolve().is_relative_to(VERSION_ROOT):
        raise ValueError(
            f"refusing to build a row identity into {shared}: it belongs to another "
            f"version root.  These arms reuse that frame read-only; copy the fold's "
            f"row_identity_fold{fold}.json into this version's outputs.manifests and "
            f"run without --prepare-row-identity."
        )
    taxonomy_path = _resolve_input(config["inputs"]["taxonomy"], workspace)
    contract_path = _resolve_input(config["inputs"]["data_contract"], workspace)
    with taxonomy_path.open("r", encoding="utf-8") as handle:
        taxonomy = Taxonomy.from_dict(json.load(handle))
    with contract_path.open("r", encoding="utf-8") as handle:
        contract = json.load(handle)
    train_samples, test_samples, allowlist = load_contract_partition(contract, fold)
    roots = row_identity_roots(config, workspace)
    block = config["inputs"]["row_identity"]
    model_cfg = config["model"]
    artifact_path, manifest_path = row_identity_paths(config, workspace, fold)
    identity_config_sha256 = sha256_json(block)

    if artifact_path.exists() or manifest_path.exists():
        if not resume:
            raise FileExistsError(
                f"{artifact_path} / {manifest_path} exist; pass --resume to reuse them"
            )
        manifest = json.loads(_require_file(manifest_path).read_text(encoding="utf-8"))
        if manifest.get("schema_version") != ROW_IDENTITY_SCHEMA:
            raise ValueError(f"row identity schema mismatch: {manifest_path}")
        if manifest.get("row_identity_config_sha256") != identity_config_sha256:
            raise ValueError(
                "row identity manifest was built from a different row_identity block"
            )
        if sha256_file(artifact_path) != manifest["artifact_sha256"]:
            raise ValueError(f"row identity artifact hash drift: {artifact_path}")
        print(json.dumps({"event": "row_identity_reused", "fold": fold}), flush=True)
        return manifest_path

    train_names = [spec.sample for spec in train_samples]
    test_names = [spec.sample for spec in test_samples]

    frame_train_uni2 = role_dir(roots["frame_uni2_root"], block, "train", fold, "frame")
    frame_train_phikon = role_dir(
        roots["frame_phikon_root"], block, "train", fold, "phikon"
    )
    frame_eval_uni2 = role_dir(roots["frame_uni2_root"], block, "eval", fold, "frame")
    frame_eval_phikon = role_dir(
        roots["frame_phikon_root"], block, "eval", fold, "phikon"
    )


    for directory, expected, name in (
        (frame_train_uni2, set(train_names), "frame/uni2/train"),
        (frame_eval_uni2, set(test_names), "frame/uni2/eval"),
    ):
        observed = _stems(directory)
        if observed != expected:
            raise RuntimeError(
                f"{name} cache set differs from the contract partition: "
                f"extra={sorted(observed - expected)[:5]}, "
                f"missing={sorted(expected - observed)[:5]}"
            )
    for directory, expected, name in (
        (frame_train_phikon, set(train_names), "phikon/train"),
        (frame_eval_phikon, set(test_names), "phikon/eval"),
        (roots["tissue_grid_root"], set(allowlist), "tissue_grid"),
    ):
        observed = _stems(directory)
        if not expected.issubset(observed):
            raise RuntimeError(
                f"{name} bank lacks required samples: "
                f"{sorted(expected - observed)[:5]}"
            )
    frozen_extra: dict[str, int] = {}
    for directory, name in (
        (roots["frozen_uni2_root"], "frozen/uni2"),
        (roots["frozen_phikon_root"], "frozen/phikon"),
    ):
        observed = _stems(directory)
        if not set(allowlist).issubset(observed):
            raise RuntimeError(
                f"{name} bank lacks allowlist samples: "
                f"{sorted(set(allowlist) - observed)[:5]}"
            )
        frozen_extra[name] = int(len(observed - set(allowlist)))

    spec_by_name = {spec.sample: spec for spec in (*train_samples, *test_samples)}
    train_pool_cells: dict[str, np.ndarray] = {}
    train_pool_labels: dict[str, np.ndarray] = {}
    eval_cells: dict[str, np.ndarray] = {}
    eval_labels: dict[str, np.ndarray] = {}
    accounting: dict[str, Any] = {}
    refine_labels = taxonomy.vocabularies["refine17"].labels
    refine_code = {name: index for index, name in enumerate(refine_labels)}
    refine_target_code, _refine_supervised_flag = taxonomy.target_codes("refine17")

    frame_uni2_dim = int(block["frame_uni2_dim"])
    for role, names in (("train", train_names), ("eval", test_names)):
        uni2_dir = frame_train_uni2 if role == "train" else frame_eval_uni2
        phikon_dir = frame_train_phikon if role == "train" else frame_eval_phikon
        for number, sample in enumerate(names, start=1):
            spec = spec_by_name[sample]
            _, primary_ids, primary_labels = _cell_artifact(
                _require_file(uni2_dir / f"{sample}.pt"), frame_uni2_dim
            )
            _, a2p_ids, a2p_labels = _cell_artifact(
                _require_file(phikon_dir / f"{sample}.pt"), int(model_cfg["phikon_dim"])
            )


            frame_coarse11 = _as_string_list(
                _load_pt_dict(uni2_dir / f"{sample}.pt").get("coarse11", ())
            )
            if len(frame_coarse11) != len(primary_ids):
                raise RuntimeError(
                    f"{sample}: frame bank stores {len(frame_coarse11)} coarse11 "
                    f"labels for {len(primary_ids)} rows"
                )
            if primary_labels:
                frame_code = _lookup_per_label(
                    primary_labels, refine_target_code, dtype=np.int64, what="refine17"
                )
                stored_identity = _lookup_per_label(
                    frame_coarse11,
                    {name: index for index, name in enumerate(IDENTITY_CLASSES)},
                    dtype=np.int64,
                    what="coarse11",
                )
                mismatch = np.flatnonzero(frame_code != stored_identity)
                if mismatch.size:
                    row = int(mismatch[0])
                    raise RuntimeError(
                        f"{sample}: the taxonomy mapping disagrees with the frame "
                        f"bank's own coarse11 on {mismatch.size} cells; first cell "
                        f"{primary_ids[row]}: {primary_labels[row]!r} -> "
                        f"{IDENTITY_CLASSES[int(frame_code[row])]!r}, stored "
                        f"{frame_coarse11[row]!r}"
                    )
            _, frozen_ids, frozen_labels = _cell_artifact(
                _require_file(roots["frozen_uni2_root"] / f"{sample}.pt"),
                int(block["frozen_uni2_dim"]),
            )
            _, frozen_p_ids, frozen_p_labels = _cell_artifact(
                _require_file(roots["frozen_phikon_root"] / f"{sample}.pt"),
                int(model_cfg["phikon_dim"]),
            )
            if frozen_p_ids != frozen_ids or frozen_p_labels != frozen_labels:
                raise RuntimeError(
                    f"{sample}: frozen UNI2/Phikon row identity or labels differ"
                )
            _, grid_ids = _grid_artifact(
                _require_file(roots["tissue_grid_root"] / f"{sample}.pt"),
                int(model_cfg["grid_dim"]),
            )
            kept, labels, entry = _intersect_frame(
                sample=sample,
                primary_ids=primary_ids,
                primary_labels=primary_labels,
                phikon_ids=a2p_ids,
                phikon_labels=a2p_labels,
                frozen_ids=frozen_ids,
                frozen_labels=frozen_labels,
                grid_ids=grid_ids,
                taxonomy=taxonomy,
            )


            observed_positive = any(
                label == taxonomy.necrosis_positive_source_label for label in frozen_labels
            )
            if observed_positive != spec.necrosis_known:
                raise RuntimeError(
                    f"{sample}: contract necrosis coverage disagrees with source labels"
                )
            entry["role"] = role
            entry["necrosis_known"] = bool(spec.necrosis_known)
            accounting[sample] = entry
            codes = (
                _lookup_per_label(labels, refine_code, dtype=np.int16, what="refine17")
                .astype(np.int8)
                if labels
                else np.zeros(0, dtype=np.int8)
            )
            width = max((len(cell) for cell in kept), default=1)
            cells = np.asarray(kept, dtype=f"<U{width}")
            if role == "train":
                train_pool_cells[sample] = cells
                train_pool_labels[sample] = codes
            else:
                eval_cells[sample] = cells
                eval_labels[sample] = codes
            print(
                json.dumps(
                    {
                        "event": "row_identity_sample",
                        "fold": fold,
                        "role": role,
                        "sample": sample,
                        "sample_index": number,
                        "sample_total": len(names),
                        **{key: entry[key] for key in ("n_primary", "n_common")},
                    },
                    sort_keys=True,
                ),
                flush=True,
            )

    train_identity = _ordered_string_sha256(
        [
            f"{sample}\x1f{cell}"
            for sample in sorted(train_pool_cells)
            for cell in train_pool_cells[sample].tolist()
        ]
    )
    eval_identity = _ordered_string_sha256(
        [
            f"{sample}\x1f{cell}"
            for sample in sorted(eval_cells)
            for cell in eval_cells[sample].tolist()
        ]
    )
    artifact = {
        "schema_version": ROW_IDENTITY_SCHEMA,
        "fold": fold,
        "row_identity_config_sha256": identity_config_sha256,
        "refine17_labels": list(refine_labels),
        "train_pool_cells": train_pool_cells,
        "train_pool_label_codes": train_pool_labels,
        "eval_cells": eval_cells,
        "eval_label_codes": eval_labels,
        "train_row_identity_sha256": train_identity,
        "eval_row_identity_sha256": eval_identity,
    }
    atomic_torch_save(artifact_path, artifact)
    manifest = {
        "schema_version": ROW_IDENTITY_SCHEMA,
        "status": "VALIDATED",
        "fold": fold,
        "row_identity_config_sha256": identity_config_sha256,
        "taxonomy_sha256": sha256_file(taxonomy_path),
        "data_contract_sha256": sha256_file(contract_path),
        "source_sha256": sha256_file(Path(__file__).resolve()),
        "artifact_path": str(artifact_path),
        "artifact_sha256": sha256_file(artifact_path),
        "train_row_identity_sha256": train_identity,
        "eval_row_identity_sha256": eval_identity,
        "n_train_samples": len(train_pool_cells),
        "n_eval_samples": len(eval_cells),
        "n_train_pool_rows": int(sum(len(v) for v in train_pool_cells.values())),
        "n_eval_rows": int(sum(len(v) for v in eval_cells.values())),
        "frozen_bank_samples_outside_allowlist": frozen_extra,
        "label_vintage": {
            "definition": (
                "the frame bank carries the reannotated cell_type_refine and is "
                "authoritative; these are the cells on which the older Phikon and "
                "frozen banks say something else -- counted, not gated, because "
                "the reannotation is the reason the aligned-cell frame exists"
            ),
            "vs_phikon_source_string": int(
                sum(
                    entry["reannotated_vs_phikon_source_string"]
                    for entry in accounting.values()
                )
            ),
            "vs_phikon_identity_target": int(
                sum(
                    entry["reannotated_vs_phikon_identity_target"]
                    for entry in accounting.values()
                )
            ),
            "vs_frozen_identity_target": int(
                sum(
                    entry["reannotated_vs_frozen_identity_target"]
                    for entry in accounting.values()
                )
            ),
            "n_samples_with_target_change": int(
                sum(
                    1
                    for entry in accounting.values()
                    if entry["reannotated_vs_frozen_identity_target"]
                )
            ),
        },
        "per_sample": accounting,
        "totals": {
            key: int(sum(entry[key] for entry in accounting.values()))
            for key in (
                "n_primary",
                "n_common",
                "dropped_absent_from_phikon",
                "dropped_absent_from_frozen",
                "dropped_absent_from_tissue_grid",
            )
        },
    }
    atomic_json(manifest_path, manifest)
    print(
        json.dumps(
            {
                "event": "row_identity_complete",
                "fold": fold,
                "manifest": str(manifest_path),
                "train_row_identity_sha256": train_identity,
                "eval_row_identity_sha256": eval_identity,
                "n_train_pool_rows": manifest["n_train_pool_rows"],
                "n_eval_rows": manifest["n_eval_rows"],
                "dropped": manifest["totals"],
            },
            sort_keys=True,
        ),
        flush=True,
    )
    return manifest_path


def load_row_identity(
    config: Mapping[str, Any],
    workspace: Path,
    fold: int,
    *,
    require_train_sha256: str | None = None,
    require_eval_sha256: str | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    artifact_path, manifest_path = row_identity_paths(config, workspace, fold)
    if not manifest_path.is_file():
        raise FileNotFoundError(
            f"missing shared row identity manifest {manifest_path}; run "
            "cell/scripts/train_fold.py --config <config> --fold <fold> --prepare-row-identity first"
        )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("schema_version") != ROW_IDENTITY_SCHEMA or manifest.get("status") != "VALIDATED":
        raise ValueError(f"row identity manifest is stale: {manifest_path}")
    if int(manifest["fold"]) != fold:
        raise ValueError("row identity manifest fold mismatch")
    expected_block = sha256_json(config["inputs"]["row_identity"])
    if manifest["row_identity_config_sha256"] != expected_block:
        raise ValueError(
            "this arm's row_identity block differs from the shared frame's; the arms "
            "would no longer be restricted to identical rows"
        )
    if sha256_file(artifact_path) != manifest["artifact_sha256"]:
        raise ValueError(f"row identity artifact hash drift: {artifact_path}")
    for label, expected, found in (
        ("train", require_train_sha256, manifest["train_row_identity_sha256"]),
        ("eval", require_eval_sha256, manifest["eval_row_identity_sha256"]),
    ):
        if expected is not None and expected != found:
            raise RuntimeError(
                f"{label} row identity mismatch: required {expected}, found {found}"
            )
    artifact = torch.load(artifact_path, map_location="cpu", weights_only=False)
    if artifact.get("row_identity_config_sha256") != expected_block:
        raise ValueError("row identity artifact/config disagreement")
    return artifact, manifest


def stable_priority(sample: str, cells: np.ndarray, seed: int) -> np.ndarray:
    frame = pd.DataFrame(
        {"sample": np.repeat(sample, len(cells)), "cell_id": np.asarray(cells).astype(str)}
    )
    values = pd.util.hash_pandas_object(frame, index=False, categorize=False).to_numpy(
        np.uint64
    )
    salt = int.from_bytes(hashlib.sha256(str(seed).encode("ascii")).digest()[:8], "little")
    return values ^ np.uint64(salt)


def load_head_selection(root: Path, fold: int, train_names: Sequence[str]) -> dict[str, Any]:
    path = _require_file(Path(root) / f"fold{fold}.json")
    manifest = json.loads(path.read_text(encoding="utf-8"))
    if manifest.get("status") != "PASS" or int(manifest.get("fold", -1)) != fold:
        raise ValueError(f"head selection manifest is not a passing fold record: {path}")


    if int(manifest["n_train_samples"]) != len(manifest["samples"]):
        raise RuntimeError("head selection manifest is internally inconsistent")
    missing = sorted(set(train_names) - set(manifest["samples"]))
    if missing:
        raise RuntimeError(
            f"head selection lacks {len(missing)} contract train samples: {missing[:5]}"
        )
    manifest["_path"] = str(path)
    manifest["_sha256"] = sha256_file(path)
    return manifest


def build_post_merge_prior_repair_selection(
    *,
    pool_cells: Mapping[str, np.ndarray],
    pool_label_codes: Mapping[str, np.ndarray],
    refine_labels: Sequence[str],
    spec_by_name: Mapping[str, SampleSpec],
    taxonomy: Taxonomy,
    selection_manifest: Mapping[str, Any],
    identity_cap: int,
    necrosis_cap: int,
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    selection_cap = int(selection_manifest["cap_per_class"])
    if identity_cap > selection_cap or necrosis_cap > selection_cap:
        raise ValueError(
            f"post-merge caps ({identity_cap}, {necrosis_cap}) exceed a2's per-source cap "
            f"{selection_cap}; the counterfactual would need rows the selection never cached"
        )
    seed = int(selection_manifest["seed"])
    target, supervised = taxonomy.target_codes("refine17")
    label_target = np.asarray([target[name] for name in refine_labels], dtype=np.int64)
    label_supervised = np.asarray([supervised[name] for name in refine_labels], dtype=np.bool_)
    necrosis_code = list(refine_labels).index(taxonomy.necrosis_positive_source_label)

    group_priority: list[list[np.ndarray]] = [[] for _ in range(NECROSIS_GROUP + 1)]
    group_sample: list[list[np.ndarray]] = [[] for _ in range(NECROSIS_GROUP + 1)]
    group_row: list[list[np.ndarray]] = [[] for _ in range(NECROSIS_GROUP + 1)]
    group_source: list[list[np.ndarray]] = [[] for _ in range(NECROSIS_GROUP + 1)]
    sample_names = sorted(pool_cells)
    sample_code = {name: index for index, name in enumerate(sample_names)}
    per_source_boundary: dict[int, int] = {}
    per_source_available: dict[int, int] = defaultdict(int)

    for name in sample_names:
        cells = pool_cells[name]
        codes = np.asarray(pool_label_codes[name], dtype=np.int64)
        if len(cells) != len(codes):
            raise ValueError(f"{name}: pool cell/label length mismatch")
        priority = stable_priority(name, cells, seed)
        for code in np.unique(codes):
            positions = np.flatnonzero(codes == code)
            per_source_available[int(code)] += int(len(positions))
            boundary = int(priority[positions].max())
            per_source_boundary[int(code)] = max(
                per_source_boundary.get(int(code), 0), boundary
            )
            group = NECROSIS_GROUP if int(code) == necrosis_code else int(label_target[code])
            if int(code) != necrosis_code and not bool(label_supervised[code]):
                raise ValueError(
                    f"{name}: refine17 label {refine_labels[int(code)]!r} is unsupervised "
                    "but is not the necrosis label"
                )
            group_priority[group].append(priority[positions])
            group_sample[group].append(
                np.full(len(positions), sample_code[name], dtype=np.int64)
            )
            group_row[group].append(positions.astype(np.int64))
            group_source[group].append(np.full(len(positions), int(code), dtype=np.int64))

    selected_rows: dict[str, list[int]] = {name: [] for name in sample_names}
    group_summaries: dict[str, Any] = {}
    source_selected: dict[str, dict[str, int]] = {}
    for group in range(NECROSIS_GROUP + 1):
        name = (
            IDENTITY_CLASSES[group] if group < NECROSIS_GROUP else "Necrosis_positive"
        )
        cap = identity_cap if group < NECROSIS_GROUP else necrosis_cap
        if not group_priority[group]:
            raise ValueError(f"training pool has zero candidate rows for {name}")
        priority = np.concatenate(group_priority[group])
        samples = np.concatenate(group_sample[group])
        rows = np.concatenate(group_row[group])
        sources = np.concatenate(group_source[group])
        available = int(len(priority))
        if len(np.unique(priority)) != available:
            raise RuntimeError(f"priority hash collision inside final class {name}")
        contributing = sorted({int(code) for code in np.unique(sources)})


        capped_sources = [
            code
            for code in contributing
            if int(selection_manifest["selected_per_class"][refine_labels[code]]) >= selection_cap
        ]
        boundary = (
            min(per_source_boundary[code] for code in capped_sources)
            if capped_sources
            else None
        )
        order_all = np.argsort(priority, kind="stable")
        available_within_boundary = available
        if boundary is not None:
            order_all = order_all[priority[order_all] <= boundary]
            available_within_boundary = int(len(order_all))
        keep = min(cap, available_within_boundary)
        if keep == 0:
            raise RuntimeError(
                f"{name}: no cached candidate row survives the source boundary"
            )
        order = order_all[:keep]
        threshold = int(priority[order].max())
        for sample_index, row in zip(samples[order].tolist(), rows[order].tolist()):
            selected_rows[sample_names[sample_index]].append(int(row))
        chosen_sources = sources[order]
        source_selected[name] = {
            refine_labels[code]: int((chosen_sources == code).sum())
            for code in contributing
        }
        group_summaries[name] = {
            "final_class_index": group if group < NECROSIS_GROUP else None,
            "contributing_source_labels": [refine_labels[code] for code in contributing],
            "available_in_pool": available,
            "available_within_source_boundary": int(available_within_boundary),
            "source_boundary_priority": None if boundary is None else int(boundary),
            "capped_source_labels": [refine_labels[code] for code in capped_sources],
            "post_merge_cap": int(cap),
            "selected": int(keep),
            "boundary_limited": bool(
                boundary is not None and available_within_boundary < min(cap, available)
            ),
            "shortfall_vs_cap": int(max(0, min(cap, available) - keep)),
            "removed_by_prior_repair": int(available - keep),
            "selected_by_source_label": source_selected[name],
            "available_by_source_label": {
                refine_labels[code]: int(per_source_available[code])
                for code in contributing
            },
            "uncapped_population_by_source_label": {
                refine_labels[code]: int(selection_manifest["population_per_class"][refine_labels[code]])
                for code in contributing
            },
            "priority_threshold": threshold,
        }

    selected_arrays = {
        name: np.asarray(sorted(set(rows)), dtype=np.int64)
        for name, rows in selected_rows.items()
    }
    for name, rows in selected_arrays.items():
        if len(rows) != len(selected_rows[name]):
            raise RuntimeError(f"{name}: duplicate row index in the selection")

    per_patient_class: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    per_sample_class: dict[str, dict[str, int]] = {}
    for name, rows in selected_arrays.items():
        codes = np.asarray(pool_label_codes[name], dtype=np.int64)[rows]
        counts: dict[str, int] = defaultdict(int)
        for code in codes.tolist():
            group_name = (
                "Necrosis_positive"
                if code == necrosis_code
                else IDENTITY_CLASSES[int(label_target[code])]
            )
            counts[group_name] += 1
        per_sample_class[name] = dict(sorted(counts.items()))
        patient = spec_by_name[name].patient
        for group_name, value in counts.items():
            per_patient_class[patient][group_name] += value

    selected_total = int(sum(len(rows) for rows in selected_arrays.values()))
    identity_total = selected_total - group_summaries["Necrosis_positive"]["selected"]
    realized_prior = {
        IDENTITY_CLASSES[index]: (
            group_summaries[IDENTITY_CLASSES[index]]["selected"] / identity_total
            if identity_total
            else 0.0
        )
        for index in range(NECROSIS_GROUP)
    }
    pool_prior_total = sum(
        group_summaries[IDENTITY_CLASSES[index]]["available_in_pool"]
        for index in range(NECROSIS_GROUP)
    )
    pool_prior = {
        IDENTITY_CLASSES[index]: (
            group_summaries[IDENTITY_CLASSES[index]]["available_in_pool"] / pool_prior_total
            if pool_prior_total
            else 0.0
        )
        for index in range(NECROSIS_GROUP)
    }
    uncapped_population: dict[str, int] = defaultdict(int)
    for label, count in selection_manifest["population_per_class"].items():
        if label == taxonomy.necrosis_positive_source_label:
            continue
        uncapped_population[IDENTITY_CLASSES[target[label]]] += int(count)
    uncapped_total = sum(uncapped_population.values())
    summary = {
        "schema_version": SELECTION_SCHEMA,
        "selection_policy": SELECTION_POLICY,
        "policy_description": (
            "final-class rows are pooled after the taxonomy merge and capped once, "
            "at the level a single post-merge cap would have produced, using a2's own "
            "per-cell priority; rows are only removed, never added; no patient "
            "balancing and no class weighting"
        ),
        "patient_balancing": "none",
        "class_weighting": "none",
        "head_selection_manifest": {
            "path": str(selection_manifest["_path"]),
            "sha256": str(selection_manifest["_sha256"]),
            "cap_per_refine17_class": selection_cap,
            "seed": seed,
            "pandas_version": str(selection_manifest["pandas_version"]),
            "label_mapping_id": str(selection_manifest["label_mapping_id"]),
        },
        "pandas_version_now": pd.__version__,
        "identity_cap_per_final_class": int(identity_cap),
        "necrosis_positive_cap": int(necrosis_cap),
        "groups": group_summaries,
        "selected_rows": selected_total,
        "selected_identity_rows": int(identity_total),
        "selected_samples": int(len(selected_arrays)),
        "samples_with_zero_selected_rows": sorted(
            name for name, rows in selected_arrays.items() if len(rows) == 0
        ),
        "class_prior_pool": pool_prior,
        "class_prior_after_repair": realized_prior,
        "class_prior_uncapped_population": {
            name: (uncapped_population[name] / uncapped_total if uncapped_total else 0.0)
            for name in IDENTITY_CLASSES
        },
        "selected_rows_by_patient_and_final_class": {
            patient: dict(sorted(counts.items()))
            for patient, counts in sorted(per_patient_class.items())
        },
        "selected_rows_by_sample_and_final_class": dict(sorted(per_sample_class.items())),
    }
    return selected_arrays, summary


def seal_shared_row_set(
    *,
    config: Mapping[str, Any],
    workspace: Path,
    fold: int,
    arm: str,
    row_set_sha256: str,
    provenance: Mapping[str, str],
) -> dict[str, Any]:
    path = (
        _resolve_output(config["outputs"]["manifests"], workspace)
        / f"train_row_set_fold{fold}.json"
    )
    payload = {
        "schema_version": "v5.shared_train_row_set.v1",
        "fold": fold,
        "train_row_set_sha256": row_set_sha256,
        "selection_policy": SELECTION_POLICY,
        "sampling_contract_sha256": provenance["sampling_contract_sha256"],
        "row_identity_sha256": provenance["row_identity_sha256"],
        "taxonomy_sha256": provenance["taxonomy_sha256"],
        "data_contract_sha256": provenance["data_contract_sha256"],
        "sealed_by_arm": arm,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(dir=str(path.parent), prefix=path.name)
    temporary_path = Path(temporary)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(json.dumps(payload, indent=2, sort_keys=True).encode("utf-8") + b"\n")
            handle.flush()
            os.fsync(handle.fileno())
        try:
            os.link(temporary_path, path)
            _fsync_parent(path)
        except FileExistsError:
            pass
    finally:
        temporary_path.unlink(missing_ok=True)
    sealed = json.loads(path.read_text(encoding="utf-8"))
    compared = ("train_row_set_sha256", "sampling_contract_sha256", "row_identity_sha256",
                "taxonomy_sha256", "data_contract_sha256", "selection_policy")
    differences = {key: (sealed.get(key), payload[key]) for key in compared
                   if sealed.get(key) != payload[key]}
    if differences:
        raise RuntimeError(
            f"arm {arm} would train on a different row set than arm "
            f"{sealed.get('sealed_by_arm')}: {differences}"
        )
    return sealed


def _selection_identity(
    selected: Mapping[str, np.ndarray], provenance: Mapping[str, str]
) -> str:
    value = {
        "provenance": dict(provenance),
        "selected": {
            sample: hashlib.sha256(rows.astype("<i8", copy=False).tobytes()).hexdigest()
            for sample, rows in sorted(selected.items())
        },
    }
    return sha256_json(value)


def _row_set_identity(
    selected: Mapping[str, np.ndarray], pool_cells: Mapping[str, np.ndarray]
) -> str:
    return _ordered_string_sha256(
        [
            f"{sample}\x1f{cell}"
            for sample in sorted(selected)
            for cell in pool_cells[sample][selected[sample]].tolist()
        ]
    )


def load_or_build_selection(
    *,
    config: Mapping[str, Any],
    workspace: Path,
    fold: int,
    taxonomy: Taxonomy,
    row_identity: Mapping[str, Any],
    spec_by_name: Mapping[str, SampleSpec],
    train_names: Sequence[str],
    provenance: Mapping[str, str],
    resume: bool,
) -> tuple[dict[str, np.ndarray], dict[str, Any], str, str]:
    scratch = _resolve_output(config["outputs"]["scratch"], workspace)
    path = scratch / f"fold{fold}" / "selection.pt"
    if path.exists():
        saved = torch.load(path, map_location="cpu", weights_only=False)
        if saved.get("schema_version") != SELECTION_SCHEMA:
            raise ValueError(f"selection cache schema mismatch: {path}")
        if saved.get("provenance") != dict(provenance):
            raise ValueError(f"selection cache provenance mismatch: {path}")
        if not resume:
            raise FileExistsError(
                f"{path} exists; use --resume only when provenance is unchanged"
            )
        selected = {
            sample: np.asarray(rows, dtype=np.int64)
            for sample, rows in saved["selected"].items()
        }
        return (
            selected,
            saved["summary"],
            str(saved["selection_sha256"]),
            str(saved["row_set_sha256"]),
        )
    sampling = config["sampling"]
    selection_manifest = load_head_selection(
        _resolve_input(config["inputs"]["row_identity"]["head_selection_root"], workspace),
        fold,
        train_names,
    )
    if int(selection_manifest["cap_per_class"]) != int(sampling["cap_per_refine17_class"]):
        raise RuntimeError(
            "head selection cap differs from the value pinned in this config"
        )
    if str(sampling["policy"]) != SELECTION_POLICY:
        raise ValueError(f"unsupported sampling policy {sampling['policy']!r}")
    if str(sampling["patient_balancing"]) != "none":
        raise ValueError("this trainer does not implement patient-balanced selection")
    selected, summary = build_post_merge_prior_repair_selection(
        pool_cells=row_identity["train_pool_cells"],
        pool_label_codes=row_identity["train_pool_label_codes"],
        refine_labels=row_identity["refine17_labels"],
        spec_by_name=spec_by_name,
        taxonomy=taxonomy,
        selection_manifest=selection_manifest,
        identity_cap=int(sampling["post_merge_cap_per_final_class"]),
        necrosis_cap=int(sampling["necrosis_positive_cap"]),
    )
    if set(selected) != set(train_names):
        raise RuntimeError("selection does not close exactly over the training samples")
    identity = _selection_identity(selected, provenance)
    row_set = _row_set_identity(selected, row_identity["train_pool_cells"])
    payload = {
            "schema_version": SELECTION_SCHEMA,
            "selection_policy": SELECTION_POLICY,
            "row_index_semantics": (
                "row index into the shared per-sample training pool held in the fold's "
                "row_identity.pt artifact, not into any single feature file"
            ),
            "fold": fold,
            "provenance": dict(provenance),
            "selection_sha256": identity,
            "row_set_sha256": row_set,
            "selected": selected,
            "selected_cells": {
                sample: row_identity["train_pool_cells"][sample][rows]
                for sample, rows in selected.items()
            },
            "summary": summary,
    }
    require_gen3_keys(payload, REQUIRED_SELECTION_KEYS, "selection")
    atomic_torch_save(path, payload)
    return selected, summary, identity, row_set


def _index_rows_by_ids(reference: Sequence[str], other: Sequence[str], path: Path) -> np.ndarray:
    position = {cell: row for row, cell in enumerate(other)}
    missing = [cell for cell in reference if cell not in position]
    if missing:
        raise ValueError(f"{path}: missing {len(missing)} required cells; first={missing[:3]}")
    return np.asarray([position[cell] for cell in reference], dtype=np.int64)


def _tensor_take(tensor: Tensor, indices: np.ndarray) -> Tensor:
    return tensor.index_select(
        0, torch.from_numpy(np.asarray(indices, dtype=np.int64))
    ).contiguous()


def load_identity_exclusion(
    config: Mapping[str, Any], workspace: Path
) -> tuple[dict[str, frozenset[str]], str] | None:
    block = config["inputs"].get("identity_exclusion")
    if block is None:
        return None
    manifest_path = _require_file(_resolve_input(block["manifest"], workspace))
    digest = sha256_file(manifest_path)
    if digest != block["manifest_sha256"]:
        raise ValueError(f"identity exclusion manifest hash drift: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    root = _resolve_input(block["labels_root"], workspace)
    excluded: dict[str, frozenset[str]] = {}
    for sample, record in manifest["per_sample"].items():
        if int(record["n_excluded"]) == 0:
            continue
        path = _require_file(root / f"{sample}.parquet")
        if sha256_file(path) != record["sha256"]:
            raise ValueError(f"identity exclusion label hash drift: {path}")
        frame = pd.read_parquet(path, columns=["cell_id", "cell_type_refine", "identity_excluded"])
        flagged = frame.loc[frame["identity_excluded"].to_numpy(dtype=bool)]
        if (flagged["cell_type_refine"] != "Schwann").any():
            raise ValueError(f"{sample}: a non-Schwann cell is flagged for identity exclusion")
        if len(flagged) != int(record["n_excluded"]):
            raise ValueError(f"{sample}: flagged count differs from the manifest")
        excluded[sample] = frozenset(flagged["cell_id"].astype(str))
    return excluded, digest


def build_sample_cache(
    *,
    spec: SampleSpec,
    cells: Sequence[str],
    label_codes: np.ndarray,
    refine_labels: Sequence[str],
    config: Mapping[str, Any],
    workspace: Path,
    fold: int,
    taxonomy: Taxonomy,
    role: str,
    excluded: frozenset[str] | None = None,
) -> dict[str, Any]:
    model_cfg = config["model"]
    vocabulary = arm_vocabulary(config)
    bundle: dict[str, Any] = {}
    for backbone, dim_key in (("uni2", "uni2_dim"), ("phikon", "phikon_dim")):
        paths = arm_feature_paths(
            config,
            workspace,
            backbone=backbone,
            sample=spec.sample,
            role=role,
            fold=fold,
        )
        part_dims = arm_part_dims(config, backbone, int(model_cfg[dim_key]))
        parts, rows, labels = _load_slot(paths, part_dims, cells)
        bundle[backbone] = (
            _tensor_take(parts[0], rows)
            if len(parts) == 1
            else torch.cat([_tensor_take(part, rows) for part in parts], dim=1)
        )
        if backbone == "uni2":
            assert_frame_label_agreement(
                sample=spec.sample,
                observed=[labels[row] for row in rows.tolist()],
                expected=[
                    refine_labels[int(code)]
                    for code in np.asarray(label_codes).tolist()
                ],
                vocabulary=vocabulary,
                taxonomy=taxonomy,
            )
        del parts
    grid_path = _require_file(
        _resolve_input(config["inputs"]["row_identity"]["tissue_grid_root"], workspace)
        / f"{spec.sample}.pt"
    )
    grid_features, grid_ids = _grid_artifact(grid_path, int(model_cfg["grid_dim"]))
    bundle["grid"] = _tensor_take(
        grid_features, _index_rows_by_ids(cells, grid_ids, grid_path)
    )
    del grid_features
    labels = [refine_labels[int(code)] for code in np.asarray(label_codes).tolist()]
    y, identity_known, lineage_y, necrosis_target = encode_labels(
        labels, taxonomy, "refine17"
    )
    masked = 0
    if excluded:
        mask = np.fromiter((cell in excluded for cell in cells), dtype=np.bool_, count=len(cells))
        masked = int(mask.sum())
        if masked:
            stray = {labels[i] for i in np.flatnonzero(mask).tolist()} - {"Schwann"}
            if stray:
                raise ValueError(
                    f"{spec.sample}: identity exclusion hits non-Schwann labels {sorted(stray)}"
                )
            y = np.where(mask, IGNORE_INDEX, y).astype(np.int64)
            identity_known = identity_known & ~mask
            lineage_y = np.where(mask, IGNORE_INDEX, lineage_y).astype(np.int64)
    if bool(necrosis_target.any()) and not spec.necrosis_known:
        raise ValueError(
            f"{spec.sample}: a Necrosis row was selected in an annotation-unknown sample"
        )
    bundle.update(
        {
            "schema_version": CACHE_SCHEMA,
            "cache_policy": "v5.selected_feature_cache.v1",
            "sample": spec.sample,
            "patient": spec.patient,
            "cancer": spec.cancer,
            "fold": fold,
            "cell": list(cells),
            "source_label": labels,
            "grid_row_index": None,
            "y": torch.from_numpy(y),
            "identity_known_mask": torch.from_numpy(identity_known),
            "lineage_y": torch.from_numpy(lineage_y),
            "necrosis_target": torch.from_numpy(necrosis_target),
            "nec_known_mask": torch.from_numpy(
                np.full(len(cells), spec.necrosis_known, dtype=np.bool_)
            ),
        }
    )
    return bundle


def build_eval_bundle(
    *,
    spec: SampleSpec,
    cells: Sequence[str],
    label_codes: np.ndarray,
    refine_labels: Sequence[str],
    config: Mapping[str, Any],
    workspace: Path,
    fold: int,
    taxonomy: Taxonomy,
) -> dict[str, Any]:
    model_cfg = config["model"]
    vocabulary = arm_vocabulary(config)
    bundle: dict[str, Any] = {}
    for backbone, dim_key in (("uni2", "uni2_dim"), ("phikon", "phikon_dim")):
        paths = arm_feature_paths(
            config,
            workspace,
            backbone=backbone,
            sample=spec.sample,
            role="eval",
            fold=fold,
        )
        part_dims = arm_part_dims(config, backbone, int(model_cfg[dim_key]))
        parts, rows, labels = _load_slot(paths, part_dims, cells)


        bundle[backbone] = parts[0] if len(parts) == 1 else list(parts)
        bundle[f"{backbone}_row_index"] = torch.from_numpy(rows)
        if backbone == "uni2":
            assert_frame_label_agreement(
                sample=spec.sample,
                observed=[labels[row] for row in rows.tolist()],
                expected=[
                    refine_labels[int(code)]
                    for code in np.asarray(label_codes).tolist()
                ],
                vocabulary=vocabulary,
                taxonomy=taxonomy,
            )
    grid_path = _require_file(
        _resolve_input(config["inputs"]["row_identity"]["tissue_grid_root"], workspace)
        / f"{spec.sample}.pt"
    )
    grid_features, grid_ids = _grid_artifact(grid_path, int(model_cfg["grid_dim"]))
    bundle["grid"] = grid_features
    bundle["grid_row_index"] = torch.from_numpy(
        _index_rows_by_ids(cells, grid_ids, grid_path)
    )
    labels = [refine_labels[int(code)] for code in np.asarray(label_codes).tolist()]
    y, identity_known, lineage_y, necrosis_target = encode_labels(
        labels, taxonomy, "refine17"
    )


    if bool(necrosis_target.any()) and not spec.necrosis_known:
        raise ValueError(
            f"{spec.sample}: a Necrosis row is present in an annotation-unknown sample"
        )
    bundle.update(
        {
            "sample": spec.sample,
            "patient": spec.patient,
            "cancer": spec.cancer,
            "fold": fold,
            "cell": list(cells),
            "source_label": labels,
            "y": torch.from_numpy(y),
            "identity_known_mask": torch.from_numpy(identity_known),
            "lineage_y": torch.from_numpy(lineage_y),
            "necrosis_target": torch.from_numpy(necrosis_target),
            "nec_known_mask": torch.from_numpy(
                np.full(len(cells), spec.necrosis_known, dtype=np.bool_)
            ),
        }
    )
    return bundle


def load_or_build_sample_cache(
    *,
    spec: SampleSpec,
    rows: np.ndarray,
    pool_cells: np.ndarray,
    pool_label_codes: np.ndarray,
    refine_labels: Sequence[str],
    config: Mapping[str, Any],
    workspace: Path,
    fold: int,
    taxonomy: Taxonomy,
    selection_sha256: str,
    provenance: Mapping[str, str],
    excluded: frozenset[str] | None = None,
) -> Path:
    scratch = _resolve_output(config["outputs"]["scratch"], workspace)
    path = scratch / f"fold{fold}" / "selected_features" / f"{spec.sample}.pt"
    expected = {
        **dict(provenance),
        "selection_sha256": selection_sha256,
        "sample_rows_sha256": hashlib.sha256(
            np.asarray(rows, dtype="<i8").tobytes()
        ).hexdigest(),
    }
    if path.exists():
        saved = _load_pt_dict(path)
        if saved.get("cache_provenance") != expected:
            raise ValueError(f"selected feature cache provenance mismatch: {path}")
        return path
    bundle = build_sample_cache(
        spec=spec,
        cells=pool_cells[rows].tolist(),
        label_codes=np.asarray(pool_label_codes)[rows],
        refine_labels=refine_labels,
        config=config,
        workspace=workspace,
        fold=fold,
        taxonomy=taxonomy,
        role="train",
        excluded=excluded,
    )
    bundle["cache_provenance"] = expected
    require_gen3_keys(bundle, REQUIRED_CACHE_KEYS, "selected feature cache")
    atomic_torch_save(path, bundle)
    return path


class GatedDualBackboneContext(nn.Module):

    def __init__(
        self,
        *,
        uni2_dim: int,
        phikon_dim: int,
        grid_dim: int,
        projection_dim: int = 256,
        hidden_dim: int = 512,
        n_identity: int = 11,
        n_lineage: int = 6,
        dropout: float = 0.2,
        cell_feature_dropout: float = 0.05,
        grid_feature_dropout: float = 0.15,
        necrosis_hidden_dim: int = 512,
        necrosis_dropout: float = 0.2,
    ) -> None:
        super().__init__()
        self.cell_feature_dropout = float(cell_feature_dropout)
        self.grid_feature_dropout = float(grid_feature_dropout)
        self.uni2 = nn.Sequential(
            nn.LayerNorm(uni2_dim), nn.Linear(uni2_dim, projection_dim), nn.GELU()
        )
        self.phikon = nn.Sequential(
            nn.LayerNorm(phikon_dim), nn.Linear(phikon_dim, projection_dim), nn.GELU()
        )
        self.gate = nn.Linear(projection_dim * 2, projection_dim)
        self.grid = nn.Sequential(
            nn.LayerNorm(grid_dim), nn.Linear(grid_dim, projection_dim), nn.GELU()
        )
        self.trunk = nn.Sequential(
            nn.Linear(projection_dim * 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.identity_head = nn.Linear(hidden_dim, n_identity)
        self.lineage_head = nn.Linear(hidden_dim, n_lineage)
        self.necrosis_head = nn.Sequential(
            nn.LayerNorm(grid_dim),
            nn.Linear(grid_dim, necrosis_hidden_dim),
            nn.GELU(),
            nn.Dropout(necrosis_dropout),
            nn.Linear(necrosis_hidden_dim, 1),
        )

    def forward(self, uni2: Tensor, phikon: Tensor, grid: Tensor) -> dict[str, Tensor]:
        u = self.uni2(uni2)
        p = self.phikon(phikon)
        u = F.dropout(u, p=self.cell_feature_dropout, training=self.training)
        p = F.dropout(p, p=self.cell_feature_dropout, training=self.training)
        gate = torch.sigmoid(self.gate(torch.cat((u, p), dim=-1)))
        cell = gate * u + (1.0 - gate) * p
        identity_grid = F.dropout(
            grid, p=self.grid_feature_dropout, training=self.training
        )
        context = self.grid(identity_grid)
        hidden = self.trunk(torch.cat((cell, context), dim=-1))
        return {
            "identity_logits": self.identity_head(hidden),
            "lineage_logits": self.lineage_head(hidden),
            "necrosis_logit": self.necrosis_head(grid.detach()).squeeze(-1),
            "gate_mean": gate.mean(),
        }


def jensen_shannon_from_logits(first: Tensor, second: Tensor) -> Tensor:
    p = F.softmax(first.float(), dim=-1)
    q = F.softmax(second.float(), dim=-1)
    midpoint = 0.5 * (p + q)
    eps = torch.finfo(midpoint.dtype).eps
    first_kl = (p * (torch.log(p.clamp_min(eps)) - torch.log(midpoint))).sum(-1)
    second_kl = (q * (torch.log(q.clamp_min(eps)) - torch.log(midpoint))).sum(-1)
    return 0.5 * (first_kl + second_kl)


def multitask_loss(
    first_pass: Mapping[str, Tensor],
    second_pass: Mapping[str, Tensor],
    *,
    y: Tensor,
    identity_known_mask: Tensor,
    lineage_y: Tensor,
    necrosis_target: Tensor,
    nec_known_mask: Tensor,
    class_weight: Tensor | None,
    necrosis_pos_weight: Tensor,
    label_smoothing: float,
    lineage_weight: float,
    js_weight: float,
    necrosis_weight: float,
) -> tuple[Tensor, dict[str, float]]:
    valid_identity = identity_known_mask.bool() & (y != IGNORE_INDEX)
    if valid_identity.any():
        identity_losses = [
            F.cross_entropy(
                output["identity_logits"][valid_identity],
                y[valid_identity],
                weight=class_weight,
                label_smoothing=label_smoothing,
            )
            for output in (first_pass, second_pass)
        ]
        identity_loss = 0.5 * (identity_losses[0] + identity_losses[1])
        lineage_loss = 0.5 * (
            F.cross_entropy(
                first_pass["lineage_logits"][valid_identity], lineage_y[valid_identity]
            )
            + F.cross_entropy(
                second_pass["lineage_logits"][valid_identity], lineage_y[valid_identity]
            )
        )
        js_loss = jensen_shannon_from_logits(
            first_pass["identity_logits"][valid_identity],
            second_pass["identity_logits"][valid_identity],
        ).mean()
    else:
        zero = first_pass["identity_logits"].sum() * 0.0
        identity_loss = lineage_loss = js_loss = zero

    valid_necrosis = nec_known_mask.bool()
    if valid_necrosis.any():
        target = necrosis_target[valid_necrosis].float()
        necrosis_loss = 0.5 * (
            F.binary_cross_entropy_with_logits(
                first_pass["necrosis_logit"][valid_necrosis],
                target,
                pos_weight=necrosis_pos_weight,
            )
            + F.binary_cross_entropy_with_logits(
                second_pass["necrosis_logit"][valid_necrosis],
                target,
                pos_weight=necrosis_pos_weight,
            )
        )
    else:
        necrosis_loss = first_pass["necrosis_logit"].sum() * 0.0
    total = (
        identity_loss
        + lineage_weight * lineage_loss
        + js_weight * js_loss
        + necrosis_weight * necrosis_loss
    )
    components = {
        "total": float(total.detach()),
        "identity": float(identity_loss.detach()),
        "lineage": float(lineage_loss.detach()),
        "js": float(js_loss.detach()),
        "necrosis": float(necrosis_loss.detach()),
    }
    return total, components


def _identity_and_necrosis_statistics(
    cache_paths: Sequence[Path], training: Mapping[str, Any]
) -> tuple[Tensor | None, Tensor, dict[str, Any]]:
    counts = np.zeros(len(IDENTITY_CLASSES), dtype=np.int64)
    nec_positive = 0
    nec_negative = 0
    for path in cache_paths:
        value = _load_pt_dict(path)
        y = value["y"].numpy()
        known = value["identity_known_mask"].numpy().astype(bool)
        counts += np.bincount(y[known], minlength=len(IDENTITY_CLASSES))
        nec_known = value["nec_known_mask"].numpy().astype(bool)
        nec_target = value["necrosis_target"].numpy()
        nec_positive += int(((nec_target == 1) & nec_known).sum())
        nec_negative += int(((nec_target == 0) & nec_known).sum())
        del value
    if np.any(counts == 0):
        missing = [IDENTITY_CLASSES[i] for i in np.flatnonzero(counts == 0)]
        raise ValueError(f"training selection has zero examples for {missing}")
    mode = str(training["class_weight_mode"])
    if mode != "uniform":
        raise ValueError(
            "this trainer deliberately runs without identity class weighting; "
            f"class_weight_mode={mode!r} is not implemented"
        )
    if nec_positive == 0 or nec_negative == 0:
        raise ValueError(
            "known-source Necrosis training rows require both positive and negative labels"
        )
    pos_weight = min(
        nec_negative / nec_positive, float(training["necrosis_pos_weight_max"])
    )
    summary = {
        "identity_counts": {
            name: int(count) for name, count in zip(IDENTITY_CLASSES, counts)
        },
        "identity_class_weighting": "uniform_none",
        "necrosis_known_positive": nec_positive,
        "necrosis_known_negative": nec_negative,
        "necrosis_pos_weight": float(pos_weight),
    }
    return None, torch.tensor(pos_weight, dtype=torch.float32), summary


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed % (2**32))
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True, warn_only=True)


def _stable_seed(*parts: Any) -> int:
    encoded = "\x1f".join(str(part) for part in parts).encode("utf-8")
    return int.from_bytes(hashlib.blake2b(encoded, digest_size=8).digest(), "little")


def _scheduler_multiplier(step: int, total_steps: int, warmup_steps: int) -> float:
    if warmup_steps and step < warmup_steps:
        return max(1e-8, (step + 1) / warmup_steps)
    progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
    return 0.5 * (1.0 + math.cos(math.pi * min(1.0, max(0.0, progress))))


def _batch_from_cache(
    value: Mapping[str, Any], rows: Tensor, device: torch.device
) -> dict[str, Tensor]:
    batch = {
        key: value[key].index_select(0, rows).to(device, non_blocking=True)
        for key in (
            "y",
            "identity_known_mask",
            "lineage_y",
            "necrosis_target",
            "nec_known_mask",
        )
    }
    for key in ("uni2", "phikon", "grid"):
        index = value.get(f"{key}_row_index")
        source_rows = rows if index is None else index.index_select(0, rows)
        stored = value[key]


        parts = stored if isinstance(stored, list) else [stored]
        taken = [part.index_select(0, source_rows) for part in parts]
        gathered = taken[0] if len(taken) == 1 else torch.cat(taken, dim=1)
        batch[key] = gathered.to(device, non_blocking=True).float()
    return batch


def _autocast_context(device: torch.device, amp_dtype: str):
    if device.type != "cuda":
        return contextlib.nullcontext()
    dtype = torch.bfloat16 if amp_dtype == "bfloat16" else torch.float16
    return torch.autocast("cuda", dtype=dtype)


def _checkpoint_provenance(
    config_path: Path, taxonomy_path: Path, contract_path: Path, config: Mapping[str, Any]
) -> dict[str, str]:
    return {
        "source_sha256": sha256_file(Path(__file__).resolve()),
        "config_sha256": sha256_file(config_path),
        "config_semantic_sha256": sha256_json(config),
        "taxonomy_sha256": sha256_file(taxonomy_path),
        "data_contract_sha256": sha256_file(contract_path),
        "sampling_contract_sha256": sha256_json(config["sampling"]),
    }


def _verify_checkpoint_provenance(
    checkpoint: Mapping[str, Any], expected: Mapping[str, str], fold: int
) -> None:
    if checkpoint.get("schema_version") != CHECKPOINT_SCHEMA:
        raise ValueError("checkpoint schema mismatch")
    if int(checkpoint.get("fold", -1)) != fold:
        raise ValueError("checkpoint fold mismatch")
    if checkpoint.get("provenance", {}) != dict(expected):
        raise ValueError("checkpoint provenance mismatch")


def _confusion_update(confusion: np.ndarray, y: np.ndarray, prediction: np.ndarray) -> None:
    valid = (y >= 0) & (y < confusion.shape[0])
    flat = y[valid] * confusion.shape[0] + prediction[valid]
    confusion += np.bincount(flat, minlength=confusion.size).reshape(confusion.shape)


def _f1_from_confusion(confusion: np.ndarray) -> np.ndarray:
    true_positive = np.diag(confusion).astype(np.float64)
    false_positive = confusion.sum(0) - true_positive
    false_negative = confusion.sum(1) - true_positive
    denominator = 2 * true_positive + false_positive + false_negative
    return np.divide(
        2 * true_positive,
        denominator,
        out=np.zeros_like(true_positive),
        where=denominator > 0,
    )


def _binary_auc(y: np.ndarray, score: np.ndarray) -> float | None:
    positives = int(y.sum())
    negatives = int(len(y) - positives)
    if positives == 0 or negatives == 0:
        return None
    order = np.argsort(score, kind="mergesort")
    sorted_score = score[order]
    ranks = np.empty(len(score), dtype=np.float64)
    start = 0
    while start < len(score):
        stop = start + 1
        while stop < len(score) and sorted_score[stop] == sorted_score[start]:
            stop += 1
        ranks[order[start:stop]] = 0.5 * (start + 1 + stop)
        start = stop
    return float(
        (ranks[y == 1].sum() - positives * (positives + 1) / 2) / (positives * negatives)
    )


def _oof_arrays(
    *,
    spec: SampleSpec,
    fold: int,
    bundle: Mapping[str, Any],
    model: nn.Module,
    device: torch.device,
    eval_batch_size: int,
    autocast_dtype: str,
    provenance: Mapping[str, str],
    checkpoint_sha256: str,
    checkpoint_relative_path: str,
    config_relative_path: str,
    source_relative_path: str,
    row_identity_sha256: str,
) -> dict[str, Any]:
    n = len(bundle["cell"])
    prob = np.empty((n, len(IDENTITY_CLASSES)), dtype=np.float32)
    lineage_prob = np.empty((n, len(LINEAGE_CLASSES)), dtype=np.float32)
    nec_prob = np.empty(n, dtype=np.float32)
    model.eval()
    with torch.inference_mode():
        for start in range(0, n, eval_batch_size):
            stop = min(n, start + eval_batch_size)
            rows = torch.arange(start, stop)
            batch = _batch_from_cache(bundle, rows, device)
            with _autocast_context(device, autocast_dtype):
                output = model(batch["uni2"], batch["phikon"], batch["grid"])
            identity_batch = F.softmax(output["identity_logits"].float(), -1)
            lineage_batch = F.softmax(output["lineage_logits"].float(), -1)
            necrosis_batch = torch.sigmoid(output["necrosis_logit"].float())
            if not (
                torch.isfinite(identity_batch).all()
                and torch.isfinite(lineage_batch).all()
                and torch.isfinite(necrosis_batch).all()
            ):
                raise FloatingPointError(
                    f"non-finite held-out output for {spec.sample}, rows {start}:{stop}"
                )
            prob[start:stop] = identity_batch.cpu().numpy()
            lineage_prob[start:stop] = lineage_batch.cpu().numpy()
            nec_prob[start:stop] = necrosis_batch.cpu().numpy()
    nec_target = bundle["necrosis_target"].numpy().astype(np.int8).copy()
    nec_known_mask = bundle["nec_known_mask"].numpy().astype(np.bool_)
    nec_target[~nec_known_mask] = -1
    arrays = {
        "schema_version": np.asarray(OOF_SCHEMA),
        "experiment_id": np.asarray(provenance["experiment_id"]),
        "arm": np.asarray(provenance["arm"]),
        "patient": np.asarray(spec.patient),
        "sample": np.asarray(spec.sample),
        "cancer": np.asarray(spec.cancer),
        "cell": np.asarray(bundle["cell"]),
        "fold": np.asarray(fold, dtype=np.int16),
        "y": bundle["y"].numpy().astype(np.int16),
        "identity_known_mask": bundle["identity_known_mask"].numpy().astype(np.bool_),
        "prob": prob,
        "lineage_y": bundle["lineage_y"].numpy().astype(np.int16),
        "lineage_prob": lineage_prob,
        "nec_prob": nec_prob,
        "nec_target": nec_target,
        "nec_known_mask": nec_known_mask,
        "source_label": np.asarray(bundle["source_label"]),
        "identity_classes": np.asarray(IDENTITY_CLASSES),
        "lineage_classes": np.asarray(LINEAGE_CLASSES),
        "checkpoint_sha256": np.asarray(checkpoint_sha256),
        "config_sha256": np.asarray(provenance["config_sha256"]),
        "source_sha256": np.asarray(provenance["source_sha256"]),
        "taxonomy_sha256": np.asarray(provenance["taxonomy_sha256"]),
        "data_contract_sha256": np.asarray(provenance["data_contract_sha256"]),
        "sampling_contract_sha256": np.asarray(provenance["sampling_contract_sha256"]),
        "eval_row_identity_sha256": np.asarray(row_identity_sha256),
        "checkpoint_path": np.asarray(checkpoint_relative_path),
        "config_path": np.asarray(config_relative_path),
        "source_path": np.asarray(source_relative_path),
    }
    require_gen3_keys(arrays, REQUIRED_OOF_KEYS, "held-out")
    return arrays


def _fold_summary(
    oof_files: Sequence[Path],
    *,
    fold: int,
    provenance: Mapping[str, str],
    checkpoint_sha256: str,
    training_history: Sequence[Mapping[str, Any]],
    selection_summary: Mapping[str, Any],
    statistics_summary: Mapping[str, Any],
    row_identity_manifest: Mapping[str, Any],
    row_set_sha256: str,
    model_config: Mapping[str, Any],
    model_parameters: int,
) -> dict[str, Any]:
    confusion = np.zeros((len(IDENTITY_CLASSES), len(IDENTITY_CLASSES)), dtype=np.int64)
    known_nec_target: list[np.ndarray] = []
    known_nec_prob: list[np.ndarray] = []
    rows = 0
    identity_rows = 0
    samples: list[str] = []
    patients: set[str] = set()
    observed_necrosis_samples = 0
    for path in oof_files:
        with np.load(path, allow_pickle=False) as value:
            if str(value["schema_version"]) != OOF_SCHEMA:
                raise ValueError(f"held-out schema mismatch: {path}")
            if str(value["checkpoint_sha256"]) != checkpoint_sha256:
                raise ValueError(f"held-out checkpoint identity mismatch: {path}")
            y = value["y"].astype(np.int64)
            _confusion_update(confusion, y, value["prob"].argmax(1))
            valid = value["identity_known_mask"].astype(bool)
            rows += len(y)
            identity_rows += int(valid.sum())
            known = value["nec_known_mask"].astype(bool)
            if known.any():
                observed_necrosis_samples += 1
                known_nec_target.append(value["nec_target"][known].astype(np.int8))
                known_nec_prob.append(value["nec_prob"][known].astype(np.float64))
            samples.append(str(value["sample"]))
            patients.add(str(value["patient"]))
    f1 = _f1_from_confusion(confusion)
    if known_nec_target:
        target = np.concatenate(known_nec_target)
        probability = np.concatenate(known_nec_prob)
        necrosis_metrics = {
            "scope": "positive_observed_samples_only",
            "n_known_cells": int(len(target)),
            "n_positive": int(target.sum()),
            "n_negative": int(len(target) - int(target.sum())),
            "n_positive_observed_samples": int(observed_necrosis_samples),
            "auroc": _binary_auc(target, probability),
            "brier": float(np.mean((probability - target) ** 2)),
        }
    else:
        necrosis_metrics = {
            "scope": "positive_observed_samples_only",
            "n_known_cells": 0,
            "status": "unavailable_no_known_samples",
        }
    return {
        "schema_version": FOLD_SUMMARY_SCHEMA,
        "fold": fold,
        "status": "complete_fixed_epoch_heldout",
        "arm": provenance["arm"],
        "experiment_id": provenance["experiment_id"],
        "provenance": dict(provenance),
        "checkpoint_sha256": checkpoint_sha256,
        "model_config": dict(model_config),
        "model_parameters": int(model_parameters),
        "train_row_set_sha256": row_set_sha256,
        "shared_row_identity": {
            "train_row_identity_sha256": row_identity_manifest["train_row_identity_sha256"],
            "eval_row_identity_sha256": row_identity_manifest["eval_row_identity_sha256"],
            "artifact_sha256": row_identity_manifest["artifact_sha256"],
        },
        "n_samples": len(samples),
        "n_patients": len(patients),
        "n_rows": rows,
        "n_identity_rows": identity_rows,
        "identity": {
            "macro_f1_10": float(f1[:SCORED_IDENTITY_COUNT].mean()),
            "macro_f1_11": float(f1.mean()),
            "per_class_f1": {
                name: float(score) for name, score in zip(IDENTITY_CLASSES, f1)
            },
            "confusion": confusion.tolist(),
        },
        "necrosis_observed_label_metrics": necrosis_metrics,
        "training_history": list(training_history),
        "selection": dict(selection_summary),
        "statistics": dict(statistics_summary),
        "outer_test_selection_use": "none",
    }


def train_fold(
    config_path: Path,
    fold: int,
    *,
    resume: bool = False,
    device_name: str | None = None,
    require_train_row_identity: str | None = None,
    require_eval_row_identity: str | None = None,
) -> Path:
    config_path = Path(config_path).resolve()
    config, workspace = load_experiment(config_path)
    taxonomy_path = _resolve_input(config["inputs"]["taxonomy"], workspace)
    contract_path = _resolve_input(config["inputs"]["data_contract"], workspace)
    with taxonomy_path.open("r", encoding="utf-8") as handle:
        taxonomy = Taxonomy.from_dict(json.load(handle))
    with contract_path.open("r", encoding="utf-8") as handle:
        contract = json.load(handle)
    train_samples, test_samples, _allowlist = load_contract_partition(contract, fold)
    train_names = [spec.sample for spec in train_samples]
    spec_by_name = {spec.sample: spec for spec in (*train_samples, *test_samples)}

    training = config["training"]
    seed = int(training["seed"]) + fold * 100_003
    _seed_everything(seed)

    models_root = _resolve_output(config["outputs"]["models"], workspace)
    oof_root = _resolve_output(config["outputs"]["oof"], workspace)
    results_root = _resolve_output(config["outputs"]["results"], workspace)
    checkpoint_path = models_root / f"fold{fold}.pt"
    summary_path = results_root / f"fold{fold}_summary.json"
    if summary_path.exists():
        raise FileExistsError(
            f"completed fold summary already exists: {summary_path}; start a new run"
        )

    row_identity, row_identity_manifest = load_row_identity(
        config,
        workspace,
        fold,
        require_train_sha256=require_train_row_identity,
        require_eval_sha256=require_eval_row_identity,
    )
    if set(row_identity["train_pool_cells"]) != set(train_names):
        raise RuntimeError("shared row identity train set differs from the contract")
    if set(row_identity["eval_cells"]) != {spec.sample for spec in test_samples}:
        raise RuntimeError("shared row identity eval set differs from the contract")

    provenance = _checkpoint_provenance(config_path, taxonomy_path, contract_path, config)
    provenance["experiment_id"] = str(config["experiment_id"])
    provenance["arm"] = str(config["arm"])
    provenance["row_identity_sha256"] = str(row_identity_manifest["artifact_sha256"])
    exclusion = load_identity_exclusion(config, workspace)
    if exclusion is not None:
        provenance["identity_exclusion_manifest_sha256"] = exclusion[1]

    selected, selection_summary, selection_sha256, row_set_sha256 = load_or_build_selection(
        config=config,
        workspace=workspace,
        fold=fold,
        taxonomy=taxonomy,
        row_identity=row_identity,
        spec_by_name=spec_by_name,
        train_names=train_names,
        provenance=provenance,
        resume=resume,
    )
    sealed_row_set = seal_shared_row_set(
        config=config,
        workspace=workspace,
        fold=fold,
        arm=str(config["arm"]),
        row_set_sha256=row_set_sha256,
        provenance=provenance,
    )
    print(
        json.dumps(
            {
                "event": "row_set_sealed",
                "fold": fold,
                "arm": config["arm"],
                "train_row_set_sha256": row_set_sha256,
                "sealed_by_arm": sealed_row_set["sealed_by_arm"],
            },
            sort_keys=True,
        ),
        flush=True,
    )
    refine_labels = list(row_identity["refine17_labels"])
    cache_paths: list[Path] = []
    for index, sample_name in enumerate(sorted(selected), start=1):
        cache_paths.append(
            load_or_build_sample_cache(
                spec=spec_by_name[sample_name],
                rows=selected[sample_name],
                pool_cells=row_identity["train_pool_cells"][sample_name],
                pool_label_codes=row_identity["train_pool_label_codes"][sample_name],
                refine_labels=refine_labels,
                config=config,
                workspace=workspace,
                fold=fold,
                taxonomy=taxonomy,
                selection_sha256=selection_sha256,
                provenance=provenance,
                excluded=None if exclusion is None else exclusion[0].get(sample_name, frozenset()),
            )
        )
        print(
            json.dumps(
                {
                    "event": "selected_cache_ready",
                    "fold": fold,
                    "arm": config["arm"],
                    "sample": sample_name,
                    "sample_index": index,
                    "sample_total": len(selected),
                },
                sort_keys=True,
            ),
            flush=True,
        )
    exclusion_summary: dict[str, Any] | None = None
    if exclusion is not None:
        per_sample_masked: dict[str, int] = {}
        n_in_pool = 0
        for sample_name in sorted(selected):
            flagged = exclusion[0].get(sample_name)
            if not flagged:
                continue
            pool = row_identity["train_pool_cells"][sample_name]
            n_in_pool += int(sum(1 for cell in pool.tolist() if cell in flagged))
            chosen = pool[selected[sample_name]]
            per_sample_masked[sample_name] = int(sum(1 for cell in chosen.tolist() if cell in flagged))
        n_labelled = int(sum(len(exclusion[0][name]) for name in train_names if name in exclusion[0]))
        n_masked = int(sum(per_sample_masked.values()))
        if n_masked != n_in_pool:
            raise ValueError(
                f"identity exclusion masked {n_masked} rows but {n_in_pool} flagged cells are in "
                f"the training pool; the cap must not drop flagged Schwann"
            )
        if n_in_pool == 0:
            raise ValueError("identity exclusion matched no training cell; check the cell id format")
        exclusion_summary = {
            "manifest_sha256": exclusion[1],
            "n_flagged_on_train_slides_in_labels": n_labelled,
            "n_flagged_in_train_pool": n_in_pool,
            "n_masked_train_rows": n_masked,
            "per_sample_masked": {k: v for k, v in per_sample_masked.items() if v},
        }
        print(json.dumps({"event": "identity_exclusion_applied", "fold": fold, "arm": config["arm"],
                          "n_flagged_on_train_slides_in_labels": n_labelled,
                          "n_flagged_in_train_pool": n_in_pool, "n_masked_train_rows": n_masked},
                         sort_keys=True), flush=True)
    class_weight, necrosis_pos_weight, statistics_summary = (
        _identity_and_necrosis_statistics(cache_paths, training)
    )

    model_cfg = dict(config["model"])
    model = GatedDualBackboneContext(**model_cfg)
    model_parameters = int(sum(p.numel() for p in model.parameters()))
    if device_name is None:
        device_name = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device_name)
    model.to(device)
    necrosis_pos_weight = necrosis_pos_weight.to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(training["learning_rate"]),
        weight_decay=float(training["weight_decay"]),
    )
    batch_size = int(training["batch_size"])
    cache_sizes: dict[Path, int] = {}
    steps_per_epoch = 0
    for path in cache_paths:
        value = _load_pt_dict(path)
        size = int(len(value["y"]))
        cache_sizes[path] = size
        steps_per_epoch += math.ceil(size / batch_size)
        del value
    epochs = int(training["epochs"])
    total_steps = max(1, epochs * steps_per_epoch)
    warmup_steps = int(round(total_steps * float(training["warmup_fraction"])))
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lambda step: _scheduler_multiplier(step, total_steps, warmup_steps)
    )
    start_epoch = 0
    history: list[dict[str, Any]] = []
    if checkpoint_path.exists():
        if not resume:
            raise FileExistsError(
                f"{checkpoint_path} exists; pass --resume only for identical provenance"
            )
        checkpoint = _load_pt_dict(checkpoint_path)
        _verify_checkpoint_provenance(checkpoint, provenance, fold)
        if checkpoint.get("selection_sha256") != selection_sha256:
            raise ValueError("checkpoint selection identity mismatch")
        model.load_state_dict(checkpoint["model_state"])
        optimizer.load_state_dict(checkpoint["optimizer_state"])
        scheduler.load_state_dict(checkpoint["scheduler_state"])
        start_epoch = int(checkpoint["completed_epochs"])
        history = list(checkpoint.get("training_history", ()))
        if start_epoch > epochs:
            raise ValueError("checkpoint has more epochs than the frozen config")

    amp_dtype = str(training["amp_dtype"])
    for epoch in range(start_epoch, epochs):
        epoch_seed = seed + epoch * 1_000_003
        _seed_everything(epoch_seed)
        rng = random.Random(epoch_seed)
        ordered_paths = list(cache_paths)
        rng.shuffle(ordered_paths)
        model.train()
        sums: dict[str, float] = defaultdict(float)
        rows_seen = 0
        batches_seen = 0
        started = time.time()
        for cache_path in ordered_paths:
            n = cache_sizes[cache_path]
            if n == 0:
                continue
            value = _load_pt_dict(cache_path)
            generator = torch.Generator().manual_seed(
                _stable_seed(epoch_seed, value["sample"])
            )
            permutation = torch.randperm(n, generator=generator)
            for start in range(0, n, batch_size):
                rows = permutation[start : start + batch_size]
                batch = _batch_from_cache(value, rows, device)
                optimizer.zero_grad(set_to_none=True)
                with _autocast_context(device, amp_dtype):
                    first_pass = model(batch["uni2"], batch["phikon"], batch["grid"])
                    second_pass = model(batch["uni2"], batch["phikon"], batch["grid"])
                    loss, components = multitask_loss(
                        first_pass,
                        second_pass,
                        y=batch["y"],
                        identity_known_mask=batch["identity_known_mask"],
                        lineage_y=batch["lineage_y"],
                        necrosis_target=batch["necrosis_target"],
                        nec_known_mask=batch["nec_known_mask"],
                        class_weight=class_weight,
                        necrosis_pos_weight=necrosis_pos_weight,
                        label_smoothing=float(training["label_smoothing"]),
                        lineage_weight=float(training["lineage_loss_weight"]),
                        js_weight=float(training["rdrop_js_weight"]),
                        necrosis_weight=float(training["necrosis_loss_weight"]),
                    )
                if not torch.isfinite(loss):
                    raise FloatingPointError(
                        f"non-finite training loss in fold {fold}, epoch {epoch + 1}, "
                        f"sample {value['sample']}"
                    )
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(), float(training["gradient_clip_norm"])
                )
                optimizer.step()
                scheduler.step()
                batch_rows = len(rows)
                rows_seen += batch_rows
                batches_seen += 1
                for key, component in components.items():
                    sums[key] += component * batch_rows
            del value
        epoch_record = {
            "epoch": epoch + 1,
            "rows": rows_seen,
            "batches": batches_seen,
            "loss": {key: value / max(1, rows_seen) for key, value in sums.items()},
            "learning_rate": float(optimizer.param_groups[0]["lr"]),
            "elapsed_seconds": time.time() - started,
            "selection_basis": "training_only",
        }
        history.append(epoch_record)
        checkpoint_payload = {
                "schema_version": CHECKPOINT_SCHEMA,
                "checkpoint_policy": "v5.fold_checkpoint.v1",
                "experiment_id": config["experiment_id"],
                "arm": config["arm"],
                "fold": fold,
                "provenance": dict(provenance),
                "source_path": str(Path(__file__).resolve().relative_to(workspace)),
                "config_path": str(config_path.relative_to(workspace)),
                "selection_sha256": selection_sha256,
                "row_set_sha256": row_set_sha256,
                "allowlist_sha256": sha256_json(list(_allowlist)),
                "completed_epochs": epoch + 1,
                "fixed_epochs": epochs,
                "model_config": model_cfg,
                "model_parameters": model_parameters,
                "identity_classes": list(IDENTITY_CLASSES),
                "lineage_classes": list(LINEAGE_CLASSES),
                "model_state": model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "scheduler_state": scheduler.state_dict(),
                "training_history": history,
                "sampling": statistics_summary,
                "outer_test_selection_use": "none",
        }
        require_gen3_keys(checkpoint_payload, REQUIRED_CHECKPOINT_KEYS, "checkpoint")
        atomic_torch_save(checkpoint_path, checkpoint_payload)
        print(json.dumps({"event": "epoch_complete", "arm": config["arm"], **epoch_record}), flush=True)

    checkpoint = _load_pt_dict(checkpoint_path)
    _verify_checkpoint_provenance(checkpoint, provenance, fold)
    if int(checkpoint["completed_epochs"]) != epochs:
        raise RuntimeError("fixed-epoch training did not complete")
    checkpoint_sha256 = sha256_file(checkpoint_path)
    model.load_state_dict(checkpoint["model_state"])
    model.to(device)

    oof_files: list[Path] = []
    for index, spec in enumerate(test_samples, start=1):
        output_path = oof_root / f"fold{fold}" / f"{spec.sample}.npz"
        if output_path.exists():
            if not resume:
                raise FileExistsError(output_path)
            with np.load(output_path, allow_pickle=False) as existing:
                if (
                    str(existing["checkpoint_sha256"]) != checkpoint_sha256
                    or str(existing["config_sha256"]) != provenance["config_sha256"]
                ):
                    raise ValueError(f"existing held-out provenance mismatch: {output_path}")
            oof_files.append(output_path)
            continue
        cells = row_identity["eval_cells"][spec.sample]
        bundle = build_eval_bundle(
            spec=spec,
            cells=cells.tolist(),
            label_codes=row_identity["eval_label_codes"][spec.sample],
            refine_labels=refine_labels,
            config=config,
            workspace=workspace,
            fold=fold,
            taxonomy=taxonomy,
        )
        arrays = _oof_arrays(
            spec=spec,
            fold=fold,
            bundle=bundle,
            model=model,
            device=device,
            eval_batch_size=int(training["eval_batch_size"]),
            autocast_dtype=amp_dtype,
            provenance=provenance,
            checkpoint_sha256=checkpoint_sha256,
            checkpoint_relative_path=str(checkpoint_path.relative_to(workspace)),
            config_relative_path=str(config_path.relative_to(workspace)),
            source_relative_path=str(Path(__file__).resolve().relative_to(workspace)),
            row_identity_sha256=str(row_identity_manifest["eval_row_identity_sha256"]),
        )
        atomic_npz(output_path, **arrays)
        oof_files.append(output_path)
        print(
            json.dumps(
                {
                    "event": "heldout_sample_complete",
                    "fold": fold,
                    "arm": config["arm"],
                    "sample": spec.sample,
                    "sample_index": index,
                    "sample_total": len(test_samples),
                    "rows": len(bundle["cell"]),
                },
                sort_keys=True,
            ),
            flush=True,
        )
        del bundle, arrays
    summary = _fold_summary(
        oof_files,
        fold=fold,
        provenance=provenance,
        checkpoint_sha256=checkpoint_sha256,
        training_history=history,
        selection_summary=selection_summary,
        statistics_summary=statistics_summary,
        row_identity_manifest=row_identity_manifest,
        row_set_sha256=row_set_sha256,
        model_config=model_cfg,
        model_parameters=model_parameters,
    )
    if exclusion_summary is not None:
        summary["identity_exclusion"] = exclusion_summary
    atomic_json(summary_path, summary)
    print(
        json.dumps(
            {
                "event": "fold_complete",
                "fold": fold,
                "arm": config["arm"],
                "summary": str(summary_path),
                "macro_f1_10_diagnostic": summary["identity"]["macro_f1_10"],
            },
            sort_keys=True,
        ),
        flush=True,
    )
    return summary_path


def run_smoke() -> dict[str, Any]:
    torch.manual_seed(0)
    config = {
        "uni2_dim": 32,
        "phikon_dim": 24,
        "grid_dim": 40,
        "projection_dim": 8,
        "hidden_dim": 16,
        "n_identity": len(IDENTITY_CLASSES),
        "n_lineage": len(LINEAGE_CLASSES),
        "dropout": 0.2,
        "cell_feature_dropout": 0.05,
        "grid_feature_dropout": 0.15,
        "necrosis_hidden_dim": 12,
        "necrosis_dropout": 0.2,
    }
    model = GatedDualBackboneContext(**config)
    rows = 32
    uni2 = torch.randn(rows, config["uni2_dim"])
    phikon = torch.randn(rows, config["phikon_dim"])
    grid = torch.randn(rows, config["grid_dim"])
    first = model(uni2, phikon, grid)
    second = model(uni2, phikon, grid)
    y = torch.randint(0, len(IDENTITY_CLASSES), (rows,))
    lineage_y = torch.randint(0, len(LINEAGE_CLASSES), (rows,))
    identity_known = torch.ones(rows, dtype=torch.bool)
    necrosis_target = torch.zeros(rows, dtype=torch.int8)
    necrosis_target[:4] = 1
    nec_known = torch.ones(rows, dtype=torch.bool)
    loss, components = multitask_loss(
        first,
        second,
        y=y,
        identity_known_mask=identity_known,
        lineage_y=lineage_y,
        necrosis_target=necrosis_target,
        nec_known_mask=nec_known,
        class_weight=None,
        necrosis_pos_weight=torch.tensor(3.0),
        label_smoothing=0.02,
        lineage_weight=0.15,
        js_weight=0.1,
        necrosis_weight=0.25,
    )
    loss.backward()


    model.zero_grad(set_to_none=True)
    output = model(uni2, phikon, grid)
    output["necrosis_logit"].sum().backward()
    identity_side = [
        name
        for name, parameter in model.named_parameters()
        if not name.startswith("necrosis_head") and parameter.grad is not None
        and bool(parameter.grad.abs().sum() > 0)
    ]
    necrosis_side = [
        name
        for name, parameter in model.named_parameters()
        if name.startswith("necrosis_head") and parameter.grad is not None
        and bool(parameter.grad.abs().sum() > 0)
    ]
    if identity_side:
        raise AssertionError(f"necrosis gradient reached the identity path: {identity_side}")
    if not necrosis_side:
        raise AssertionError("necrosis gradient did not reach the necrosis head")
    return {
        "status": "ok",
        "identity_shape": list(first["identity_logits"].shape),
        "lineage_shape": list(first["lineage_logits"].shape),
        "necrosis_shape": list(first["necrosis_logit"].shape),
        "loss": components,
        "necrosis_gradient_parameters": sorted(necrosis_side),
        "parameters": int(sum(p.numel() for p in model.parameters())),
    }


def parameter_report(config_path: Path) -> dict[str, Any]:
    config, _workspace = load_experiment(Path(config_path))
    model = GatedDualBackboneContext(**config["model"])
    return {
        "experiment_id": config["experiment_id"],
        "arm": config["arm"],
        "model_config": dict(config["model"]),
        "parameters": int(sum(p.numel() for p in model.parameters())),
        "parameters_by_module": {
            name: int(sum(p.numel() for p in module.parameters(recurse=True)))
            for name, module in model.named_children()
        },
    }


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--fold", type=int, choices=range(5))
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--prepare-row-identity", action="store_true")
    parser.add_argument("--require-train-row-identity", type=str, default=None)
    parser.add_argument("--require-eval-row-identity", type=str, default=None)
    parser.add_argument("--parameter-report", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    if args.smoke:
        print(json.dumps(run_smoke(), indent=2, sort_keys=True))
        return 0
    if args.config is None:
        raise SystemExit("--config is required")
    if args.parameter_report:
        print(json.dumps(parameter_report(args.config), indent=2, sort_keys=True))
        return 0
    if args.fold is None:
        raise SystemExit("--fold is required")
    if args.prepare_row_identity:
        build_row_identity(args.config, args.fold, resume=args.resume)
        return 0
    train_fold(
        args.config,
        args.fold,
        resume=args.resume,
        device_name=args.device,
        require_train_row_identity=args.require_train_row_identity,
        require_eval_row_identity=args.require_eval_row_identity,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
