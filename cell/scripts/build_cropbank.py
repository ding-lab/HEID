#!/usr/bin/env python3

from __future__ import annotations
from pathlib import Path
_RELEASE = Path(__file__).resolve().parents[2]

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import tifffile
import zarr

import contract as C

sys.path.insert(0, str(_RELEASE / "cell/scripts"))
from grayscale_preprocess import (
    MEMBERSHIP_FILTER_ID,
    is_blank_common_membership,
)

CROP = C.CROP
HALF = CROP // 2
CHUNK = 512
BAND_H = 4096
STRIP_PAD = HALF

PREPROCESSING_ID = "rgb_raw_v1"
INTERFACE_ID = "native_rgb_imagenet_v1"
DEFAULT_MAX_TRAIN = 50000
DEFAULT_SEED_BASE = 4200

CLASSES = C.CLASSES17
N_CLASS = C.N_CLASS


def scan_counts(train_samples: list[str]) -> tuple[dict[str, np.ndarray], np.ndarray, dict]:
    per_sample: dict[str, np.ndarray] = {}
    totals = np.zeros(N_CLASS, dtype=np.int64)
    index = {name: i for i, name in enumerate(CLASSES)}
    for sample in train_samples:
        rec = C.record(sample)
        ids = pd.read_parquet(rec["train_parquet"], columns=["cell_id"])
        ids["cell_id"] = ids["cell_id"].astype(str)
        labels = pd.read_parquet(C.labels_path(sample), columns=["cell_id", "cell_type_refine"])
        labels["cell_id"] = labels["cell_id"].astype(str)
        joined = ids.merge(labels, on="cell_id", how="inner")
        if len(joined) != int(rec["n_keep_cells_own"]):
            raise RuntimeError(
                f"{sample}: scan joined {len(joined)} cells, contract says "
                f"{rec['n_keep_cells_own']}"
            )
        counts = np.zeros(N_CLASS, dtype=np.int64)
        for name, count in joined["cell_type_refine"].astype(str).value_counts().items():
            if name not in index:
                raise RuntimeError(f"{sample}: label outside refine17: {name}")
            counts[index[name]] = int(count)
        if int(counts.sum()) != len(joined):
            raise RuntimeError(f"{sample}: label histogram lost rows")
        per_sample[sample] = counts
        totals += counts
    audit = {
        "label_mapping_id": C.LABEL_MAPPING_ID,
        "label_mapping_sha256": C.LABEL_MAPPING_SHA256,
        "population_by_class": {CLASSES[i]: int(totals[i]) for i in range(N_CLASS)},
        "n_rows": int(totals.sum()),
        "drop_policy": "none",
    }
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
        cls: (
            np.sort(rng.choice(int(totals[cls]), min(int(totals[cls]), per_class), replace=False))
            if totals[cls]
            else np.empty(0, dtype=np.int64)
        )
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


def make_rgb_crop(block: np.ndarray, yy: int, xx: int, bh: int, bw: int) -> np.ndarray | None:
    source = np.asarray(block)
    if source.dtype != np.uint8 or source.ndim != 3 or source.shape[2] != 3:
        raise ValueError(f"aligned H&E tile must be uint8 HxWx3 RGB, got {source.shape}")
    y0, x0 = yy - HALF, xx - HALF
    y1, x1 = y0 + CROP, x0 + CROP
    rgb = np.full((CROP, CROP, 3), 255, dtype=np.uint8)
    sy0, sx0 = max(0, y0), max(0, x0)
    sy1, sx1 = min(bh, y1), min(bw, x1)
    if sy1 > sy0 and sx1 > sx0:
        rgb[sy0 - y0:sy1 - y0, sx0 - x0:sx1 - x0] = source[sy0:sy1, sx0:sx1]
    if is_blank_common_membership(rgb):
        return None
    return rgb


def gather_crops(he_path: Path, xs: np.ndarray, ys: np.ndarray) -> tuple[list[np.ndarray], np.ndarray]:
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
        bands = list(range(ymin, ymax, BAND_H))
        for position, sy in enumerate(bands):


            lo = -(1 << 62) if position == 0 else sy
            hi = (1 << 62) if position == len(bands) - 1 else min(sy + BAND_H, ymax)
            selected = np.nonzero((ys >= lo) & (ys < hi))[0]
            if len(selected) == 0:
                continue
            ry0 = (max(0, sy - STRIP_PAD) // CHUNK) * CHUNK
            ry1 = min(height, ((min(height, sy + BAND_H + STRIP_PAD) + CHUNK - 1) // CHUNK) * CHUNK)
            band_x = xs[selected]
            rx0 = (max(0, int(band_x.min()) - HALF) // CHUNK) * CHUNK
            rx1 = min(width, ((int(band_x.max()) + HALF + 1 + CHUNK - 1) // CHUNK) * CHUNK)
            block = np.asarray(z[ry0:ry1, rx0:rx1, :])
            bh, bw = block.shape[:2]
            for global_index in selected:
                gathered[global_index] = make_rgb_crop(
                    block, int(ys[global_index]) - ry0, int(xs[global_index]) - rx0, bh, bw
                )
            del block
        keep = np.asarray([crop is not None for crop in gathered], dtype=bool)
        return [crop for crop in gathered if crop is not None], keep
    finally:
        close = getattr(store, "close", None)
        if callable(close):
            close()


def membership_sha256(labels: np.ndarray, samples: np.ndarray, cells: np.ndarray) -> str:
    import hashlib

    digest = hashlib.sha256()
    for label, sample, cell in zip(labels, samples, cells, strict=True):
        for value in (str(sample), str(cell), str(int(label))):
            encoded = value.encode("utf-8")
            digest.update(len(encoded).to_bytes(8, "little"))
            digest.update(encoded)
    return digest.hexdigest()


def build(fold: int, max_train: int, seed_base: int, audit_only: bool) -> dict[str, Any]:
    start = time.time()
    membership = C.fold_membership()[fold]
    train_samples = sorted(membership["train_samples"])
    C.assert_outside_fold(train_samples, fold, f"cropbank fold{fold} train list")

    per_sample, totals, label_audit = scan_counts(train_samples)
    seed = seed_base + fold
    plan, selected = select_local_indices(per_sample, totals, max_train, seed)
    preflight = {
        "status": "PREFLIGHT_PASS",
        "schema_version": "v8.rgb_cropbank.v1",
        "fold": fold,
        "canonical_fold_key": C.CANONICAL_FOLD_KEY,
        "contract_sha256": C.contract_sha256(),
        "preprocessing_id": PREPROCESSING_ID,
        "backbone_interface_id": INTERFACE_ID,
        "representation_arm": "rgb",
        "representation": "native_rgb_uint8_NHWC",
        "membership_filter_id": MEMBERSHIP_FILTER_ID,
        "label_mapping_id": C.LABEL_MAPPING_ID,
        "label_mapping_sha256": C.LABEL_MAPPING_SHA256,
        "label_audit": label_audit,
        "n_train_samples": len(train_samples),
        "n_test_samples": len(membership["test_samples"]),
        "population_by_class": {CLASSES[i]: int(totals[i]) for i in range(N_CLASS)},
        "selected_before_blank_filter": {CLASSES[i]: int(selected[i]) for i in range(N_CLASS)},
        "selected_total_before_blank_filter": int(selected.sum()),
        "max_train": int(max_train),
        "seed": int(seed),
        "fold_fingerprint": C.fold_fingerprint(fold),
    }
    if audit_only:
        return preflight

    out_dir = C.cropbank_dir(fold)
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
        if not all(present.values()):
            raise RuntimeError(f"incomplete existing cropbank at {out_dir}")
        expected = {
            "status": "PASS",
            "preprocessing_id": PREPROCESSING_ID,
            "label_mapping_sha256": C.LABEL_MAPPING_SHA256,
            "seed": seed,
            "max_train": max_train,
        }
        mismatch = {k: (v, prior.get(k)) for k, v in expected.items() if prior.get(k) != v}
        if mismatch:
            raise RuntimeError(f"incompatible existing cropbank: {mismatch}")


        C.assert_fold_binding(prior.get("fold_fingerprint"), fold, f"existing cropbank fold{fold}")
        for key in ("crops", "labels", "samples", "cells"):
            if C.sha256_file(paths[key]) != prior["outputs"][key]["sha256"]:
                raise RuntimeError(f"existing cropbank hash drift: {paths[key]}")
        return {**prior, "status": "ALREADY_COMPLETE"}
    if any(present.values()):
        raise RuntimeError(f"partial cropbank exists without a manifest at {out_dir}")
    out_dir.mkdir(parents=True, exist_ok=True)

    capacity = int(selected.sum())
    crops_array = np.empty((capacity, CROP, CROP, 3), dtype=np.uint8)
    filled = 0
    all_labels: list[int] = []
    all_samples: list[str] = []
    all_cells: list[str] = []
    dropped_blank = np.zeros(N_CLASS, dtype=np.int64)
    label_index = {name: i for i, name in enumerate(CLASSES)}

    for sample in train_samples:
        sample_plan = plan[sample]
        if not sample_plan:
            continue
        cells = C.load_cells(sample)
        if len(cells) != int(per_sample[sample].sum()):
            raise RuntimeError(f"{sample}: gather set differs from the scanned population")
        refine = cells["cell_type_refine"].astype(str).to_numpy()
        chosen: list[int] = []
        chosen_labels: list[int] = []
        for cls, local in sample_plan.items():
            positions = np.flatnonzero(refine == CLASSES[cls])
            if len(positions) != int(per_sample[sample][cls]):
                raise RuntimeError(f"{sample}: class {CLASSES[cls]} count moved between passes")
            indices = positions[local]
            chosen.extend(indices.tolist())
            chosen_labels.extend([cls] * len(indices))
        order = np.argsort(np.asarray(chosen, dtype=np.int64))
        row_indices = np.asarray(chosen, dtype=np.int64)[order]
        labels = np.asarray(chosen_labels, dtype=np.int64)[order]
        rows = cells.iloc[row_indices]
        xs = np.rint(rows["he_px"].to_numpy()).astype(np.int64)
        ys = np.rint(rows["he_py"].to_numpy()).astype(np.int64)
        crops, keep = gather_crops(C.he_path(sample), xs, ys)
        dropped_blank += np.bincount(labels[~keep], minlength=N_CLASS)
        for crop in crops:
            crops_array[filled] = crop
            filled += 1
        all_labels.extend(labels[keep].tolist())
        all_samples.extend([sample] * int(keep.sum()))
        all_cells.extend(rows.iloc[np.flatnonzero(keep)]["cell_id"].astype(str).tolist())
        print(f"[fold {fold}] {sample}: selected={len(rows)} kept={int(keep.sum())}", flush=True)
        del cells, crops

    if filled == 0:
        raise RuntimeError("no crop survived the membership blank filter")
    crops_array = crops_array[:filled]
    labels_array = np.asarray(all_labels, dtype=np.int64)
    samples_array = np.asarray(all_samples, dtype=f"<U{max(map(len, all_samples))}")
    cells_array = np.asarray(all_cells, dtype=f"<U{max(map(len, all_cells))}")
    if not (len(crops_array) == len(labels_array) == len(samples_array) == len(cells_array)):
        raise RuntimeError("cropbank arrays are misaligned")
    if crops_array.dtype != np.uint8 or tuple(crops_array.shape[1:]) != (CROP, CROP, 3):
        raise RuntimeError(f"cropbank is not uint8 [N,224,224,3]: {crops_array.shape}")

    observed = set(samples_array.astype(str).tolist())
    C.assert_outside_fold(observed, fold, f"cropbank fold{fold} observed samples")
    if observed - set(train_samples):
        raise RuntimeError("cropbank holds a sample outside the explicit fold train list")

    for key, array in (
        ("crops", crops_array),
        ("labels", labels_array),
        ("samples", samples_array),
        ("cells", cells_array),
    ):
        C.atomic_npy(paths[key], array)

    manifest = {
        **preflight,
        "status": "PASS",
        "selection_policy": (
            "uniform without replacement within each of 17 refine classes over the explicit "
            "fold_he_safe train slides"
        ),
        "blank_filter": {
            "membership_filter_id": MEMBERSHIP_FILTER_ID,
            "algorithm": "a1 arm-invariant Luster membership plane",
            "stage": "before anything arm-specific",
            "pixel_threshold_strictly_greater_than": 240,
            "drop_if_fraction_strictly_greater_than": 0.9,
        },
        "crop_convention": {
            "source_mpp": C.PX_UM,
            "size_px": CROP,
            "field_of_view_um": round(CROP * C.PX_UM, 4),
            "center": "aligned_he he_px/he_py, rounded with rint",
            "padding_rgb": 255,
            "channel_order": "RGB",
        },
        "n_kept": int(len(labels_array)),
        "kept_by_class": {CLASSES[i]: int((labels_array == i).sum()) for i in range(N_CLASS)},
        "blank_dropped_by_class": {CLASSES[i]: int(dropped_blank[i]) for i in range(N_CLASS)},
        "n_samples_represented": int(len(observed)),
        "membership_identity_sha256": membership_sha256(labels_array, samples_array, cells_array),
        "train_samples_sha256": C.sha256_list(train_samples),
        "sources": {
            "contract": {"path": str(C.CONTRACT), "sha256": C.contract_sha256()},
            "script": {
                "path": str(Path(__file__).resolve()),
                "sha256": C.sha256_file(Path(__file__)),
            },
            "a1_grayscale_preprocess": {
                "path": str((_RELEASE / "cell/scripts/grayscale_preprocess.py").resolve()),
                "sha256": C.sha256_file(_RELEASE / "cell/scripts/grayscale_preprocess.py"),
            },
        },
        "outputs": {
            key: {
                "path": str(paths[key].resolve()),
                "sha256": C.sha256_file(paths[key]),
                "dtype": str(array.dtype),
                "shape": list(array.shape),
            }
            for key, array in (
                ("crops", crops_array),
                ("labels", labels_array),
                ("samples", samples_array),
                ("cells", cells_array),
            )
        },
        "wall_sec": round(time.time() - start, 1),
        "node": os.environ.get("SLURMD_NODENAME", "login-or-local"),
        "job_id": os.environ.get("SLURM_JOB_ID"),
    }
    C.atomic_json(paths["manifest"], manifest)
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fold", type=int, choices=range(5))
    parser.add_argument("--task-id", type=int)
    parser.add_argument("--max-train", type=int, default=DEFAULT_MAX_TRAIN)
    parser.add_argument("--seed-base", type=int, default=DEFAULT_SEED_BASE)
    parser.add_argument("--audit-only", action="store_true")
    args = parser.parse_args()
    fold = args.fold if args.fold is not None else args.task_id
    if fold is None:
        raise ValueError("provide --fold or --task-id")
    result = build(int(fold), args.max_train, args.seed_base, args.audit_only)
    print(json.dumps({k: v for k, v in result.items() if k != "label_audit"},
                     indent=2, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
