#!/usr/bin/env python3

from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import tifffile
import zarr

from grayscale_preprocess import (
    MEMBERSHIP_FILTER_ID,
    freeze_slide_grayod_params,
    get_preprocess_spec,
    preprocess_rgb_crop,
    preprocess_spec_sha256,
)
from label_contract import (
    CANONICAL_CLASSES_17,
    LABEL_MAPPING_ID,
    LABEL_MAPPING_SHA256,
    RAW_TO_REFINE17,
    canonicalize_raw_labels,
    contract_manifest_record,
)


A1 = Path(__file__).resolve().parents[1]
PC = A1.parent
COHORT = A1 / "data/cohort_clean177.csv"
SPLIT = A1 / "data/split_outer5_inner5.json"
CELL_TABLE_DIR = PC / "data/cell_tables"
DEFAULT_CONFIG = A1 / "configs/gray_adapter_formal_registry.json"
CROPBANK_ROOT = A1 / "outputs/cropbanks"

CLASSES = list(CANONICAL_CLASSES_17)
N_CLASS = len(CLASSES)

CROP = 224
HALF = CROP // 2
CHUNK = 512
BAND_H = 4096
STRIP_PAD = HALF


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_sha256(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def atomic_json(path: Path, value: Any) -> None:
    tmp = path.with_name(path.name + f".tmp.{os.getpid()}")
    tmp.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    tmp.replace(path)


def atomic_npy(path: Path, array: np.ndarray) -> None:
    tmp = path.with_name(path.name + f".tmp.{os.getpid()}")
    with tmp.open("wb") as handle:
        np.save(handle, array, allow_pickle=False)
    tmp.replace(path)


def load_config(path: Path) -> dict[str, Any]:
    config = json.loads(path.read_text())
    if config.get("schema_version") != "a1.gray_adapter_run.v1":
        raise ValueError(f"unsupported gray adapter config schema in {path}")
    if config.get("split_id") != json.loads(SPLIT.read_text()).get("split_id"):
        raise ValueError("config split_id does not match the canonical clean177 split")
    if config.get("label_mapping_id") != LABEL_MAPPING_ID:
        raise ValueError("config label_mapping_id differs from the frozen raw-to-refine17 contract")
    if config.get("label_mapping_sha256") != LABEL_MAPPING_SHA256:
        raise ValueError("config label_mapping_sha256 differs from the frozen raw-to-refine17 contract")
    candidates = config.get("preprocessing_candidates")
    if not isinstance(candidates, list) or not candidates:
        raise ValueError("config must declare at least one preprocessing candidate")
    candidate_ids: list[str] = []
    for candidate in candidates:
        preprocessing_id = str(candidate.get("id", ""))
        spec = get_preprocess_spec(preprocessing_id)
        if spec.get("formal_launch_allowed") is not True:
            raise ValueError(f"non-formal preprocessing is forbidden in this config: {preprocessing_id}")
        expected = preprocess_spec_sha256(preprocessing_id)
        if candidate.get("sha256") != expected:
            raise ValueError(f"config preprocessing registry hash is stale for {preprocessing_id}")
        candidate_ids.append(preprocessing_id)
    if len(candidate_ids) != len(set(candidate_ids)):
        raise ValueError("config preprocessing candidate IDs are duplicated")
    return config


def configured_candidate(config: dict[str, Any], preprocessing_id: str) -> dict[str, Any]:
    matches = [
        candidate
        for candidate in config["preprocessing_candidates"]
        if str(candidate["id"]) == preprocessing_id
    ]
    if len(matches) != 1:
        raise ValueError(f"preprocessing_id {preprocessing_id!r} is not a unique configured candidate")
    return matches[0]


def cropbank_dir(
    preprocessing_id: str,
    fold: int,
    root: Path = CROPBANK_ROOT,
) -> Path:
    if get_preprocess_spec(preprocessing_id)["preprocessing_id"] != preprocessing_id:
        raise ValueError("registry returned a different preprocessing_id")
    if Path(preprocessing_id).name != preprocessing_id:
        raise ValueError("preprocessing_id must be a single path component")
    return root / LABEL_MAPPING_ID / preprocessing_id / f"fold{fold}"


def make_gray_crop(
    block: np.ndarray,
    yy: int,
    xx: int,
    bh: int,
    bw: int,
    preprocessing_id: str,
    slide_grayod_frozen: Any | None = None,
) -> np.ndarray | None:
    source = np.asarray(block)
    if source.dtype != np.uint8 or source.ndim != 3 or source.shape[2] != 3:
        raise ValueError(f"aligned H&E tile must be uint8 HxWx3 RGB, got {source.shape}/{source.dtype}")
    y0, x0 = yy - HALF, xx - HALF
    y1, x1 = y0 + CROP, x0 + CROP
    rgb = np.full((CROP, CROP, 3), 255, dtype=np.uint8)
    sy0, sx0 = max(0, y0), max(0, x0)
    sy1, sx1 = min(bh, y1), min(bw, x1)
    if sy1 > sy0 and sx1 > sx0:
        rgb[sy0 - y0:sy1 - y0, sx0 - x0:sx1 - x0] = source[sy0:sy1, sx0:sx1]
    result = preprocess_rgb_crop(
        rgb,
        preprocessing_id,
        slide_grayod_frozen=slide_grayod_frozen,
    )
    if result.is_blank:
        return None
    if result.gray is None or result.gray.dtype != np.uint8 or result.gray.shape != (CROP, CROP):
        raise RuntimeError("grayscale registry returned a non-canonical crop")
    return result.gray


def gather_gray_crops(
    he_path: Path,
    xs: np.ndarray,
    ys: np.ndarray,
    preprocessing_id: str,
    slide_grayod_frozen: Any | None = None,
) -> tuple[list[np.ndarray], np.ndarray]:
    if len(xs) != len(ys):
        raise ValueError("x/y coordinate arrays differ in length")
    if len(xs) == 0:
        return [], np.zeros(0, dtype=bool)
    store = tifffile.imread(str(he_path), aszarr=True, level=0)
    try:
        z = zarr.open(store, mode="r")
        if len(z.shape) != 3 or int(z.shape[2]) != 3 or np.dtype(z.dtype) != np.dtype(np.uint8):
            raise ValueError(f"aligned H&E must expose uint8 HxWx3 L0, got {z.shape}/{z.dtype}")
        height, width = int(z.shape[0]), int(z.shape[1])
        gathered: list[np.ndarray | None] = [None] * len(xs)
        ymin = max(0, int(ys.min()) - HALF)
        ymax = min(height, int(ys.max()) + HALF + 1)
        for sy in range(ymin, ymax, BAND_H):
            stop = min(sy + BAND_H, ymax)
            selected = np.nonzero((ys >= sy) & (ys < stop))[0]
            if len(selected) == 0:
                continue
            ry0 = (max(0, sy - STRIP_PAD) // CHUNK) * CHUNK
            ry1 = min(
                height,
                ((min(height, sy + BAND_H + STRIP_PAD) + CHUNK - 1) // CHUNK) * CHUNK,
            )
            band_x = xs[selected]
            rx0 = (max(0, int(band_x.min()) - HALF) // CHUNK) * CHUNK
            rx1 = min(
                width,
                ((int(band_x.max()) + HALF + 1 + CHUNK - 1) // CHUNK) * CHUNK,
            )
            block = np.asarray(z[ry0:ry1, rx0:rx1, :])
            bh, bw = block.shape[:2]
            for global_index in selected:
                gathered[global_index] = make_gray_crop(
                    block,
                    int(ys[global_index]) - ry0,
                    int(xs[global_index]) - rx0,
                    bh,
                    bw,
                    preprocessing_id,
                    slide_grayod_frozen,
                )
            del block
        keep = np.asarray([crop is not None for crop in gathered], dtype=bool)
        return [crop for crop in gathered if crop is not None], keep
    finally:
        close = getattr(store, "close", None)
        if callable(close):
            close()


def load_contract(fold: int) -> tuple[pd.DataFrame, dict[str, Any], dict[str, Any]]:
    cohort = pd.read_csv(COHORT)
    cohort["sample"] = cohort["sample"].astype(str)
    cohort["patient"] = cohort["patient"].astype(str)
    split = json.loads(SPLIT.read_text())
    if set(cohort["contract_split_id"].astype(str)) != {str(split["split_id"])}:
        raise ValueError("cohort and split contract IDs differ")
    if cohort["sample"].duplicated().any():
        raise ValueError("clean177 cohort contains duplicate samples")
    outer = next(
        (item for item in split["outer_folds"] if int(item["outer_fold"]) == int(fold)),
        None,
    )
    if outer is None:
        raise ValueError(f"outer fold {fold} is absent")
    train = set(map(str, outer["train_samples"]))
    test = set(map(str, outer["test_samples"]))
    universe = set(cohort["sample"])
    if train & test or train | test != universe:
        raise ValueError("outer split is not a disjoint clean177 partition")
    if set(map(str, outer["train_patients"])) & set(map(str, outer["test_patients"])):
        raise ValueError("outer train/test patients overlap")
    selected = cohort[cohort["sample"].isin(train)].copy().sort_values("sample")
    if len(selected) != int(outer["n_train_samples"]):
        raise ValueError("outer-train sample count disagrees with split metadata")
    return selected, split, outer


def scan_counts(
    train_df: pd.DataFrame,
) -> tuple[dict[str, np.ndarray], np.ndarray, dict[str, Any]]:
    per_sample: dict[str, np.ndarray] = {}
    totals = np.zeros(N_CLASS, dtype=np.int64)
    raw_totals: dict[str, int] = {}
    per_sample_raw: dict[str, dict[str, int]] = {}
    for sample in train_df["sample"].astype(str):
        path = CELL_TABLE_DIR / f"{sample}.parquet"
        if not path.exists():
            raise FileNotFoundError(path)
        raw = pd.read_parquet(path, columns=["cell_type"])["cell_type"].astype(str)
        labels = canonicalize_raw_labels(raw, source=str(path))
        counts = np.asarray([(labels == cls).sum() for cls in CLASSES], dtype=np.int64)
        raw_counts = {
            str(label): int(count)
            for label, count in raw.value_counts().sort_index().items()
        }
        if int(counts.sum()) != len(raw):
            raise RuntimeError(f"{sample}: strict label mapping changed the row denominator")
        per_sample_raw[sample] = raw_counts
        for label, count in raw_counts.items():
            raw_totals[label] = raw_totals.get(label, 0) + count
        per_sample[sample] = counts
        totals += counts
    raw_totals = dict(sorted(raw_totals.items()))
    audit = {
        "label_mapping_id": LABEL_MAPPING_ID,
        "label_mapping_sha256": LABEL_MAPPING_SHA256,
        "label_mapping_source": contract_manifest_record(),
        "n_rows": int(sum(raw_totals.values())),
        "observed_raw_label_counts": raw_totals,
        "observed_raw_labels_sha256": canonical_sha256(sorted(raw_totals)),
        "canonical_population_by_class": {
            CLASSES[index]: int(totals[index]) for index in range(N_CLASS)
        },
        "n_nonidentity_raw_cells": int(
            sum(
                count
                for label, count in raw_totals.items()
                if RAW_TO_REFINE17[label] != label
            )
        ),
        "per_sample_raw_label_counts_sha256": canonical_sha256(per_sample_raw),
        "unmapped_labels": [],
        "drop_policy": "none",
    }
    if audit["n_rows"] != int(totals.sum()):
        raise RuntimeError("strict label audit raw/canonical totals differ")
    return per_sample, totals, audit


def select_local_indices(
    per_sample: dict[str, np.ndarray],
    totals: np.ndarray,
    max_train: int,
    seed: int,
) -> tuple[dict[str, dict[int, np.ndarray]], np.ndarray]:
    if max_train < N_CLASS:
        raise ValueError(f"max_train must be at least {N_CLASS}")
    rng = np.random.RandomState(seed)
    per_class = max_train // N_CLASS
    global_picks = {
        cls: np.sort(rng.choice(int(totals[cls]), min(int(totals[cls]), per_class), replace=False))
        if totals[cls]
        else np.empty(0, dtype=np.int64)
        for cls in range(N_CLASS)
    }
    offsets = np.zeros(N_CLASS, dtype=np.int64)
    plan: dict[str, dict[int, np.ndarray]] = {}
    selected = np.zeros(N_CLASS, dtype=np.int64)
    for sample, counts in per_sample.items():
        sample_plan: dict[int, np.ndarray] = {}
        for cls in range(N_CLASS):
            low, high = int(offsets[cls]), int(offsets[cls] + counts[cls])
            picks = global_picks[cls]
            left = int(np.searchsorted(picks, low, side="left"))
            right = int(np.searchsorted(picks, high, side="left"))
            local = picks[left:right] - low
            if len(local):
                sample_plan[cls] = local.astype(np.int64)
                selected[cls] += len(local)
            offsets[cls] = high
        plan[sample] = sample_plan
    if not np.array_equal(offsets, totals):
        raise RuntimeError("selection offsets did not consume the scanned population")
    return plan, selected


def validate_single_plane_bank(crops: np.ndarray, labels: np.ndarray) -> None:
    if crops.dtype != np.uint8 or crops.ndim != 3 or tuple(crops.shape[1:]) != (CROP, CROP):
        raise RuntimeError(
            "legacy/RGB cropbank forbidden: expected uint8 [N,224,224] single-plane crops, "
            f"got {crops.dtype} {crops.shape}"
        )
    if labels.dtype != np.int64 or labels.ndim != 1 or len(crops) != len(labels):
        raise RuntimeError("cropbank labels are not aligned int64 [N]")


def membership_sha256(labels: np.ndarray, samples: np.ndarray, cells: np.ndarray) -> str:
    digest = hashlib.sha256()
    for label, sample, cell in zip(labels, samples, cells, strict=True):
        for value in (str(sample), str(cell), str(int(label))):
            encoded = value.encode("utf-8")
            digest.update(len(encoded).to_bytes(8, "little"))
            digest.update(encoded)
    return digest.hexdigest()


def load_slide_grayod_training_entries(
    manifest_path: Path | None,
    *,
    preprocessing_id: str,
    split_id: str,
    fold: int,
    train_samples: list[str],
) -> tuple[dict[str, Any], dict[str, str], dict[str, Any] | None]:
    enabled = preprocessing_id == "luster_slide_grayod_scale_v1"
    if not enabled:
        if manifest_path is not None:
            raise ValueError("slide gray-OD params manifest is forbidden for raw preprocessing")
        return {}, {}, None
    if manifest_path is None:
        raise ValueError("luster_slide_grayod_scale_v1 requires --slide-grayod-params-manifest")
    if not manifest_path.exists():
        raise FileNotFoundError(manifest_path)
    manifest = json.loads(manifest_path.read_text())
    expected = {
        "status": "PASS",
        "schema_version": "a1.slide_grayod_params_manifest.v1",
        "preprocessing_id": "luster_slide_grayod_scale_v1",
        "split_id": split_id,
    }
    mismatch = {
        key: (value, manifest.get(key))
        for key, value in expected.items()
        if manifest.get(key) != value
    }
    if mismatch:
        raise RuntimeError(f"slide gray-OD params manifest mismatch: {mismatch}")
    frozen: dict[str, Any] = {}
    entry_hashes: dict[str, str] = {}
    references: set[str] = set()
    for sample in train_samples:
        key = f"{sample}|fold{fold}|train"
        entry = manifest.get("entries", {}).get(key)
        if not isinstance(entry, dict):
            raise RuntimeError(f"slide gray-OD training entry missing: {key}")
        params = entry.get("slide_grayod_params")
        params_sha = entry.get("slide_grayod_params_sha256")
        frozen[sample] = freeze_slide_grayod_params(
            params,
            params_sha,
            expected_sample=sample,
            expected_fold=fold,
            expected_role="train",
            expected_split_id=split_id,
        )
        entry_hashes[sample] = str(params_sha)
        references.add(canonical_sha256(params["reference"]))
    if len(references) != 1:
        raise RuntimeError("slide gray-OD outer-train entries do not share one frozen reference")
    record = {
        "path": str(manifest_path.resolve()),
        "sha256": sha256_file(manifest_path),
        "entries_sha256": canonical_sha256(entry_hashes),
        "reference_sha256": next(iter(references)),
        "n_training_entries": len(entry_hashes),
    }
    return frozen, entry_hashes, record


def build(
    fold: int,
    preprocessing_id: str,
    max_train: int,
    seed_base: int,
    config_path: Path,
    slide_grayod_params_manifest: Path | None,
    output_root: Path,
    audit_only: bool,
) -> dict[str, Any]:
    start = time.time()
    config = load_config(config_path)
    candidate = configured_candidate(config, preprocessing_id)
    preprocessing_sha = preprocess_spec_sha256(preprocessing_id)
    train_df, split, outer = load_contract(fold)
    train_samples = train_df["sample"].astype(str).tolist()
    slide_frozen_by_sample, slide_entry_hashes, slide_manifest_record = (
        load_slide_grayod_training_entries(
            slide_grayod_params_manifest,
            preprocessing_id=preprocessing_id,
            split_id=str(split["split_id"]),
            fold=fold,
            train_samples=train_samples,
        )
    )
    per_sample, totals, label_audit = scan_counts(train_df)
    seed = seed_base + fold
    plan, selected = select_local_indices(per_sample, totals, max_train, seed)
    preflight = {
        "status": "PREFLIGHT_PASS",
        "schema_version": "a1.gray_cropbank.v1",
        "fold": fold,
        "split_id": split["split_id"],
        "preprocessing_id": preprocessing_id,
        "preprocessing_sha256": preprocessing_sha,
        "representation_arm": "gray",
        "representation": "single_plane_uint8_NHW",
        "membership_filter_id": MEMBERSHIP_FILTER_ID,
        "label_mapping_id": LABEL_MAPPING_ID,
        "label_mapping_sha256": LABEL_MAPPING_SHA256,
        "label_mapping_source": label_audit["label_mapping_source"],
        "label_audit": label_audit,
        "n_train_samples": len(train_df),
        "n_test_samples": int(outer["n_test_samples"]),
        "population_by_class": {CLASSES[i]: int(totals[i]) for i in range(N_CLASS)},
        "selected_before_blank_filter": {CLASSES[i]: int(selected[i]) for i in range(N_CLASS)},
        "selected_total_before_blank_filter": int(selected.sum()),
        "max_train": int(max_train),
        "seed": int(seed),
    }
    if audit_only:
        return preflight

    out_dir = cropbank_dir(preprocessing_id, fold, output_root)
    paths = {
        "crops": out_dir / "train_gray.uint8.npy",
        "labels": out_dir / "train_labels.int64.npy",
        "samples": out_dir / "train_sample_ids.npy",
        "cells": out_dir / "train_cell_ids.npy",
        "manifest": out_dir / "manifest.json",
    }
    present = {key: path.exists() for key, path in paths.items()}
    if present["manifest"]:
        prior = json.loads(paths["manifest"].read_text())
        if not all(present.values()):
            raise RuntimeError(f"incomplete existing grayscale cropbank at {out_dir}")
        expected = {
            "status": "PASS",
            "split_id": split["split_id"],
            "preprocessing_id": preprocessing_id,
            "preprocessing_sha256": preprocessing_sha,
            "label_mapping_id": LABEL_MAPPING_ID,
            "label_mapping_sha256": LABEL_MAPPING_SHA256,
            "label_mapping_source": label_audit["label_mapping_source"],
            "seed": seed,
            "max_train": max_train,
        }
        mismatch = {key: (value, prior.get(key)) for key, value in expected.items() if prior.get(key) != value}
        if mismatch:
            raise RuntimeError(f"incompatible existing grayscale cropbank: {mismatch}")
        for key in ("crops", "labels", "samples", "cells"):
            if sha256_file(paths[key]) != prior["outputs"][key]["sha256"]:
                raise RuntimeError(f"existing grayscale cropbank hash mismatch: {paths[key]}")
        crops = np.load(paths["crops"], mmap_mode="r", allow_pickle=False)
        labels = np.load(paths["labels"], mmap_mode="r", allow_pickle=False)
        validate_single_plane_bank(crops, labels)
        return {**prior, "status": "ALREADY_COMPLETE"}
    if any(present.values()):
        raise RuntimeError(f"partial grayscale cropbank exists without a manifest at {out_dir}")
    out_dir.mkdir(parents=True, exist_ok=True)

    all_crops: list[np.ndarray] = []
    all_labels: list[int] = []
    all_samples: list[str] = []
    all_cells: list[str] = []
    dropped_blank = np.zeros(N_CLASS, dtype=np.int64)
    he_of = dict(zip(train_df["sample"].astype(str), train_df["aligned_he_path"].astype(str)))
    for sample in train_df["sample"].astype(str):
        sample_plan = plan[sample]
        if not sample_plan:
            continue
        table = pd.read_parquet(
            CELL_TABLE_DIR / f"{sample}.parquet",
            columns=["cell_id", "cell_type", "x_px", "y_px"],
        )
        labels_all = canonicalize_raw_labels(
            table["cell_type"],
            source=f"{CELL_TABLE_DIR / f'{sample}.parquet'} row gather",
        )
        chosen: list[int] = []
        chosen_labels: list[int] = []
        for cls, local in sample_plan.items():
            positions = np.flatnonzero(labels_all == CLASSES[cls])
            indices = positions[local]
            chosen.extend(indices.tolist())
            chosen_labels.extend([cls] * len(indices))
        order = np.argsort(np.asarray(chosen, dtype=np.int64))
        row_indices = np.asarray(chosen, dtype=np.int64)[order]
        labels = np.asarray(chosen_labels, dtype=np.int64)[order]
        rows = table.iloc[row_indices]
        xs = np.rint(rows["x_px"].to_numpy()).astype(np.int64)
        ys = np.rint(rows["y_px"].to_numpy()).astype(np.int64)
        gray_crops, keep = gather_gray_crops(
            Path(he_of[sample]),
            xs,
            ys,
            preprocessing_id,
            slide_frozen_by_sample.get(sample),
        )
        dropped_blank += np.bincount(labels[~keep], minlength=N_CLASS)
        all_crops.extend(gray_crops)
        all_labels.extend(labels[keep].tolist())
        all_samples.extend([sample] * int(keep.sum()))
        all_cells.extend(rows.iloc[np.flatnonzero(keep)]["cell_id"].astype(str).tolist())
        print(f"[fold {fold}] {sample}: selected={len(rows)} kept={int(keep.sum())}", flush=True)

    if not all_crops:
        raise RuntimeError("no grayscale crops survived raw-gray blank filtering")
    crops_array = np.stack(all_crops).astype(np.uint8, copy=False)
    labels_array = np.asarray(all_labels, dtype=np.int64)
    samples_array = np.asarray(all_samples, dtype=f"<U{max(map(len, all_samples))}")
    cells_array = np.asarray(all_cells, dtype=f"<U{max(map(len, all_cells))}")
    validate_single_plane_bank(crops_array, labels_array)
    if not (len(crops_array) == len(samples_array) == len(cells_array)):
        raise RuntimeError("grayscale cropbank arrays are misaligned")
    observed = set(samples_array.astype(str).tolist())
    if observed & set(map(str, outer["test_samples"])):
        raise RuntimeError("grayscale cropbank contains an outer-test sample")
    if observed - set(map(str, outer["train_samples"])):
        raise RuntimeError("grayscale cropbank contains a sample outside explicit outer train")

    for key, array in (
        ("crops", crops_array),
        ("labels", labels_array),
        ("samples", samples_array),
        ("cells", cells_array),
    ):
        atomic_npy(paths[key], array)

    output_records = {
        key: {
            "path": str(paths[key].resolve()),
            "sha256": sha256_file(paths[key]),
            "dtype": str(array.dtype),
            "shape": list(array.shape),
        }
        for (key, array) in (
            ("crops", crops_array),
            ("labels", labels_array),
            ("samples", samples_array),
            ("cells", cells_array),
        )
    }
    manifest = {
        **preflight,
        "status": "PASS",
        "evidence_status": candidate["evidence_status"],
        "selection_policy": "uniform without replacement within each of 17 classes over explicit outer-train samples",
        "blank_filter": {
            "membership_filter_id": MEMBERSHIP_FILTER_ID,
            "algorithm": "Luster raw common to RGB and every grayscale arm",
            "stage": "before arm-specific preprocessing",
            "pixel_threshold_strictly_greater_than": 240,
            "drop_if_fraction_strictly_greater_than": 0.9,
        },
        "crop_convention": {
            "source_mpp": 0.2125,
            "size_px": CROP,
            "field_of_view_um": 47.6,
            "center": "Xenium cell centroid x_px/y_px",
            "padding_raw_rgb": 255,
            "durable_colour_channels": 0,
        },
        "n_kept": int(len(labels_array)),
        "kept_by_class": {CLASSES[i]: int((labels_array == i).sum()) for i in range(N_CLASS)},
        "blank_dropped_by_class": {CLASSES[i]: int(dropped_blank[i]) for i in range(N_CLASS)},
        "n_samples_represented": int(len(observed)),
        "membership_identity_sha256": membership_sha256(
            labels_array, samples_array, cells_array
        ),
        "slide_grayod": {
            "enabled": slide_manifest_record is not None,
            "params_manifest": slide_manifest_record,
            "training_entry_sha256_by_sample": slide_entry_hashes,
        },
        "outer_train_samples_sha256": canonical_sha256(sorted(map(str, outer["train_samples"]))),
        "sources": {
            "cohort": {"path": str(COHORT.resolve()), "sha256": sha256_file(COHORT)},
            "split": {"path": str(SPLIT.resolve()), "sha256": sha256_file(SPLIT)},
            "config": {"path": str(config_path.resolve()), "sha256": sha256_file(config_path)},
            "script": {"path": str(Path(__file__).resolve()), "sha256": sha256_file(Path(__file__))},
        },
        "outputs": output_records,
        "wall_sec": round(time.time() - start, 1),
        "node": os.environ.get("SLURMD_NODENAME", "login-or-local"),
        "job_id": os.environ.get("SLURM_JOB_ID"),
    }
    atomic_json(paths["manifest"], manifest)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fold", type=int, required=True, choices=range(5))
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--preprocessing-id", required=True)
    parser.add_argument("--max-train", type=int)
    parser.add_argument("--seed-base", type=int)
    parser.add_argument("--slide-grayod-params-manifest", type=Path)
    parser.add_argument("--output-root", type=Path, default=CROPBANK_ROOT)
    parser.add_argument("--audit-only", action="store_true")
    args = parser.parse_args()
    config = load_config(args.config)
    preprocessing_id = args.preprocessing_id
    configured_candidate(config, preprocessing_id)
    crop_cfg = config["cropbank"]
    max_train = args.max_train if args.max_train is not None else int(crop_cfg["max_train"])
    seed_base = args.seed_base if args.seed_base is not None else int(crop_cfg["seed_base"])
    result = build(
        args.fold,
        preprocessing_id,
        max_train,
        seed_base,
        args.config,
        args.slide_grayod_params_manifest,
        args.output_root,
        args.audit_only,
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
