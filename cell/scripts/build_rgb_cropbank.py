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

from build_gray_cropbank import (
    A1,
    BAND_H,
    CELL_TABLE_DIR,
    CHUNK,
    CLASSES,
    COHORT,
    CROP,
    HALF,
    N_CLASS,
    SPLIT,
    STRIP_PAD,
    atomic_json,
    atomic_npy,
    canonical_sha256,
    load_contract,
    scan_counts,
    select_local_indices,
    sha256_file,
)
from grayscale_preprocess import (
    MEMBERSHIP_FILTER_ID,
    backbone_interface_sha256,
    preprocess_rgb_crop,
    preprocess_spec_sha256,
    representation_arm,
    validate_formal_preprocessing_interface,
)
from label_contract import (
    LABEL_MAPPING_ID,
    LABEL_MAPPING_SHA256,
    canonicalize_raw_labels,
)


DEFAULT_CONFIG = A1 / "configs/rgb_adapter_formal_registry.json"
CROPBANK_ROOT = A1 / "outputs/cropbanks"
PREPROCESSING_ID = "rgb_raw_v1"
INTERFACE_ID = "native_rgb_imagenet_v1"


def load_config(path: Path) -> dict[str, Any]:
    config = json.loads(path.read_text())
    if config.get("schema_version") != "a1.rgb_adapter_run.v1":
        raise ValueError("unsupported RGB adapter config schema")
    split_id = json.loads(SPLIT.read_text()).get("split_id")
    if config.get("split_id") != split_id:
        raise ValueError("RGB config split_id differs from clean177")
    if config.get("label_mapping_id") != LABEL_MAPPING_ID:
        raise ValueError("RGB config label_mapping_id differs from the frozen label contract")
    if config.get("label_mapping_sha256") != LABEL_MAPPING_SHA256:
        raise ValueError("RGB config label_mapping_sha256 differs from the frozen label contract")
    preprocess, interface = validate_formal_preprocessing_interface(
        str(config.get("preprocessing_id")),
        str(config.get("backbone_interface_id")),
    )
    if preprocess["preprocessing_id"] != PREPROCESSING_ID:
        raise ValueError("RGB producer accepts only rgb_raw_v1")
    if interface["backbone_interface_id"] != INTERFACE_ID:
        raise ValueError("RGB producer accepts only native_rgb_imagenet_v1")
    if config.get("preprocessing_sha256") != preprocess_spec_sha256(PREPROCESSING_ID):
        raise ValueError("RGB config preprocessing hash is stale")
    if config.get("backbone_interface_sha256") != backbone_interface_sha256(INTERFACE_ID):
        raise ValueError("RGB config interface hash is stale")
    if representation_arm(PREPROCESSING_ID) != "rgb":
        raise ValueError("RGB preprocessing registry arm mismatch")
    return config


def cropbank_dir(fold: int, root: Path = CROPBANK_ROOT) -> Path:
    return root / LABEL_MAPPING_ID / PREPROCESSING_ID / f"fold{fold}"


def make_rgb_crop(block: np.ndarray, yy: int, xx: int, bh: int, bw: int) -> np.ndarray | None:
    source = np.asarray(block)
    if source.dtype != np.uint8 or source.ndim != 3 or source.shape[2] != 3:
        raise ValueError(f"aligned H&E tile must be uint8 HxWx3, got {source.shape}/{source.dtype}")
    y0, x0 = yy - HALF, xx - HALF
    y1, x1 = y0 + CROP, x0 + CROP
    rgb = np.full((CROP, CROP, 3), 255, dtype=np.uint8)
    sy0, sx0 = max(0, y0), max(0, x0)
    sy1, sx1 = min(bh, y1), min(bw, x1)
    if sy1 > sy0 and sx1 > sx0:
        rgb[sy0 - y0:sy1 - y0, sx0 - x0:sx1 - x0] = source[sy0:sy1, sx0:sx1]
    result = preprocess_rgb_crop(rgb, PREPROCESSING_ID)
    if result.is_blank:
        return None
    canonical = result.canonical
    if canonical is None or canonical.dtype != np.uint8 or canonical.shape != (CROP, CROP, 3):
        raise RuntimeError("RGB registry returned a non-canonical crop")
    return canonical


def gather_rgb_crops(
    he_path: Path,
    xs: np.ndarray,
    ys: np.ndarray,
) -> tuple[list[np.ndarray], np.ndarray]:
    if len(xs) != len(ys):
        raise ValueError("x/y coordinate arrays differ in length")
    if not len(xs):
        return [], np.zeros(0, dtype=bool)
    store = tifffile.imread(str(he_path), aszarr=True, level=0)
    try:
        image = zarr.open(store, mode="r")
        if len(image.shape) != 3 or int(image.shape[2]) != 3 or np.dtype(image.dtype) != np.uint8:
            raise ValueError(f"aligned H&E must expose uint8 HxWx3 L0, got {image.shape}/{image.dtype}")
        height, width = int(image.shape[0]), int(image.shape[1])
        gathered: list[np.ndarray | None] = [None] * len(xs)
        ymin = max(0, int(ys.min()) - HALF)
        ymax = min(height, int(ys.max()) + HALF + 1)
        for sy in range(ymin, ymax, BAND_H):
            stop = min(sy + BAND_H, ymax)
            selected = np.nonzero((ys >= sy) & (ys < stop))[0]
            if not len(selected):
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
            block = np.asarray(image[ry0:ry1, rx0:rx1, :])
            bh, bw = block.shape[:2]
            for index in selected:
                gathered[index] = make_rgb_crop(
                    block,
                    int(ys[index]) - ry0,
                    int(xs[index]) - rx0,
                    bh,
                    bw,
                )
            del block
        keep = np.asarray([crop is not None for crop in gathered], dtype=bool)
        return [crop for crop in gathered if crop is not None], keep
    finally:
        close = getattr(store, "close", None)
        if callable(close):
            close()


def validate_rgb_bank(
    crops: np.ndarray,
    labels: np.ndarray,
    samples: np.ndarray,
    cells: np.ndarray,
) -> None:
    if crops.dtype != np.uint8 or crops.ndim != 4 or tuple(crops.shape[1:]) != (224, 224, 3):
        raise RuntimeError(f"native RGB cropbank must be uint8 [N,224,224,3], got {crops.shape}/{crops.dtype}")
    if labels.dtype != np.int64 or labels.ndim != 1:
        raise RuntimeError("RGB labels must be int64 [N]")
    if samples.ndim != 1 or cells.ndim != 1:
        raise RuntimeError("RGB identity arrays must be one-dimensional")
    if not (len(crops) == len(labels) == len(samples) == len(cells)):
        raise RuntimeError("RGB cropbank arrays are not identity-aligned")


def membership_sha256(labels: np.ndarray, samples: np.ndarray, cells: np.ndarray) -> str:
    digest = hashlib.sha256()
    for label, sample, cell in zip(labels, samples, cells, strict=True):
        for value in (str(sample), str(cell), str(int(label))):
            encoded = value.encode("utf-8")
            digest.update(len(encoded).to_bytes(8, "little"))
            digest.update(encoded)
    return digest.hexdigest()


def build(
    fold: int,
    config_path: Path,
    output_root: Path,
    max_train_override: int | None,
    seed_base_override: int | None,
    audit_only: bool,
) -> dict[str, Any]:
    start = time.time()
    config = load_config(config_path)
    train_df, split, outer = load_contract(fold)
    per_sample, totals, label_audit = scan_counts(train_df)
    crop_cfg = config["cropbank"]
    max_train = (
        int(max_train_override)
        if max_train_override is not None
        else int(crop_cfg["max_train"])
    )
    seed_base = (
        int(seed_base_override)
        if seed_base_override is not None
        else int(crop_cfg["seed_base"])
    )
    seed = seed_base + fold
    plan, selected = select_local_indices(per_sample, totals, max_train, seed)
    preflight = {
        "status": "PREFLIGHT_PASS",
        "schema_version": "a1.rgb_cropbank.v1",
        "fold": fold,
        "split_id": split["split_id"],
        "preprocessing_id": PREPROCESSING_ID,
        "preprocessing_sha256": preprocess_spec_sha256(PREPROCESSING_ID),
        "backbone_interface_id": INTERFACE_ID,
        "backbone_interface_sha256": backbone_interface_sha256(INTERFACE_ID),
        "representation_arm": "rgb",
        "representation": "native_rgb_uint8_NHWC",
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
        "max_train": max_train,
        "seed": seed,
    }
    if audit_only:
        return preflight

    out_dir = cropbank_dir(fold, output_root)
    paths = {
        "crops": out_dir / "train_rgb.uint8.npy",
        "labels": out_dir / "train_labels.int64.npy",
        "samples": out_dir / "train_sample_ids.npy",
        "cells": out_dir / "train_cell_ids.npy",
        "manifest": out_dir / "manifest.json",
    }
    present = {key: path.exists() for key, path in paths.items()}
    if present["manifest"]:
        prior = json.loads(paths["manifest"].read_text())
        expected = {
            "status": "PASS",
            "split_id": split["split_id"],
            "preprocessing_id": PREPROCESSING_ID,
            "preprocessing_sha256": preprocess_spec_sha256(PREPROCESSING_ID),
            "label_mapping_id": LABEL_MAPPING_ID,
            "label_mapping_sha256": LABEL_MAPPING_SHA256,
            "label_mapping_source": label_audit["label_mapping_source"],
            "seed": seed,
            "max_train": max_train,
        }
        mismatch = {key: (value, prior.get(key)) for key, value in expected.items() if prior.get(key) != value}
        if mismatch or not all(present.values()):
            raise RuntimeError(f"incompatible/incomplete existing strict RGB cropbank: {mismatch}")
        for key in ("crops", "labels", "samples", "cells"):
            if sha256_file(paths[key]) != prior["outputs"][key]["sha256"]:
                raise RuntimeError(f"strict RGB cropbank hash mismatch: {paths[key]}")
        arrays = [np.load(paths[key], mmap_mode="r", allow_pickle=False) for key in ("crops", "labels", "samples", "cells")]
        validate_rgb_bank(*arrays)
        return {**prior, "status": "ALREADY_COMPLETE"}
    if any(present.values()):
        raise RuntimeError(f"partial strict RGB cropbank exists without manifest at {out_dir}")
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
        crops, keep = gather_rgb_crops(Path(he_of[sample]), xs, ys)
        dropped_blank += np.bincount(labels[~keep], minlength=N_CLASS)
        all_crops.extend(crops)
        all_labels.extend(labels[keep].tolist())
        all_samples.extend([sample] * int(keep.sum()))
        all_cells.extend(rows.iloc[np.flatnonzero(keep)]["cell_id"].astype(str).tolist())
        print(f"[strict RGB fold {fold}] {sample}: selected={len(rows)} kept={int(keep.sum())}", flush=True)

    if not all_crops:
        raise RuntimeError("no strict RGB crops survived common membership filtering")
    crops_array = np.stack(all_crops).astype(np.uint8, copy=False)
    labels_array = np.asarray(all_labels, dtype=np.int64)
    samples_array = np.asarray(all_samples, dtype=f"<U{max(map(len, all_samples))}")
    cells_array = np.asarray(all_cells, dtype=f"<U{max(map(len, all_cells))}")
    validate_rgb_bank(crops_array, labels_array, samples_array, cells_array)
    observed = set(samples_array.astype(str).tolist())
    test_samples = set(map(str, outer["test_samples"]))
    train_samples = set(map(str, outer["train_samples"]))
    if observed & test_samples:
        raise RuntimeError("strict RGB cropbank contains an outer-test sample")
    if observed - train_samples:
        raise RuntimeError("strict RGB cropbank contains a sample outside outer train")

    arrays_by_key = {
        "crops": crops_array,
        "labels": labels_array,
        "samples": samples_array,
        "cells": cells_array,
    }
    for key, array in arrays_by_key.items():
        atomic_npy(paths[key], array)
    output_records = {
        key: {
            "path": str(paths[key].resolve()),
            "sha256": sha256_file(paths[key]),
            "dtype": str(array.dtype),
            "shape": list(array.shape),
        }
        for key, array in arrays_by_key.items()
    }
    manifest = {
        **preflight,
        "status": "PASS",
        "evidence_status": config["evidence_status"],
        "selection_policy": "same uniform per-class outer-train selection as formal grayscale arms",
        "blank_filter": {
            "membership_filter_id": MEMBERSHIP_FILTER_ID,
            "algorithm": "Luster raw common to RGB and every grayscale arm",
            "pixel_threshold_strictly_greater_than": 240,
            "drop_if_fraction_strictly_greater_than": 0.9,
        },
        "crop_convention": {
            "source_mpp": 0.2125,
            "size_px": 224,
            "field_of_view_um": 47.6,
            "center": "Xenium cell centroid x_px/y_px",
            "padding_rgb": 255,
            "channel_order": "RGB",
        },
        "n_kept": int(len(labels_array)),
        "kept_by_class": {CLASSES[i]: int((labels_array == i).sum()) for i in range(N_CLASS)},
        "blank_dropped_by_class": {CLASSES[i]: int(dropped_blank[i]) for i in range(N_CLASS)},
        "n_samples_represented": len(observed),
        "membership_identity_sha256": membership_sha256(labels_array, samples_array, cells_array),
        "outer_train_samples_sha256": canonical_sha256(sorted(train_samples)),
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
    parser.add_argument("--fold", type=int, choices=range(5), required=True)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output-root", type=Path, default=CROPBANK_ROOT)
    parser.add_argument("--max-train", type=int)
    parser.add_argument("--seed-base", type=int)
    parser.add_argument("--audit-only", action="store_true")
    args = parser.parse_args()
    print(
        json.dumps(
            build(
                args.fold,
                args.config,
                args.output_root,
                args.max_train,
                args.seed_base,
                args.audit_only,
            ),
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
