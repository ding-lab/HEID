#!/usr/bin/env python3

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import sys
import tempfile
from collections import defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

import train_region_model as base
import validate_input_content_receipt as input_receipt_validator


SCHEMA = "he_boundary_prediction.v1.r4_dense_validation.v2"
SELECTION_METRIC = "inner_validation_sample_then_patient_equal_dense_dice"
_ORIGINAL_MAKE_TILE_DATASET = base.make_tile_dataset
_ORIGINAL_CREATE_LOADER = base.create_loader
_LAST_VALIDATION_CONTRACT: dict[str, Any] | None = None
_WRAPPER_ARGS: argparse.Namespace | None = None


def sha256_file(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
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
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{path}: expected a JSON object")
    return payload


def atomic_write_once_or_equal(path: Path, payload: Mapping[str, Any]) -> None:
    materialized = dict(payload)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if read_json(path) != materialized:
            raise FileExistsError(
                f"{path}: existing R4 validation contract differs"
            )
        return
    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        dir=path.parent,
        suffix=".json",
        delete=False,
    ) as handle:
        temporary = Path(handle.name)
        json.dump(materialized, handle, indent=2, sort_keys=True)
        handle.write("\n")
    try:
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        dir=path.parent,
        suffix=".json",
        delete=False,
    ) as handle:
        temporary = Path(handle.name)
        json.dump(dict(payload), handle, indent=2, sort_keys=True)
        handle.write("\n")
    try:
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _metadata_from_label(path: Path) -> dict[str, Any]:
    with np.load(path, allow_pickle=False) as archive:
        if "metadata_json" not in archive.files:
            raise ValueError(f"{path}: missing metadata_json")
        metadata = json.loads(str(np.asarray(archive["metadata_json"]).item()))
    if not isinstance(metadata, dict):
        raise ValueError(f"{path}: metadata_json is not an object")
    return metadata


class DenseValidationDataset(Dataset[dict[str, Any]]):

    def __init__(
        self,
        records: Sequence[base.SampleRecord],
        *,
        label_size: int,
        input_size: int,
        target_mpp_um: float,
        boundary_um: float,
        sdf_clip_um: float,
        normalization_mean: Sequence[float],
        normalization_std: Sequence[float],
        cache_size: int,
    ) -> None:
        if not records:
            raise ValueError("Dense validation has no records")
        self.records = {record.sample: record for record in records}
        if len(self.records) != len(records):
            raise ValueError("Dense validation records contain duplicate samples")
        self.patients = sorted({record.patient for record in records})
        self.label_size = int(label_size)
        self.input_size = int(input_size)
        self.target_mpp_um = float(target_mpp_um)
        self.field_um = self.label_size * base.LABEL_MPP_UM
        if not math.isclose(
            self.field_um,
            self.input_size * self.target_mpp_um,
            rel_tol=0.0,
            abs_tol=1e-6,
        ):
            raise ValueError("Dense RGB and label physical fields differ")
        self.boundary_um = float(boundary_um)
        self.sdf_clip_um = float(sdf_clip_um)
        self.mean = np.asarray(normalization_mean, dtype=np.float32)[:, None, None]
        self.std = np.asarray(normalization_std, dtype=np.float32)[:, None, None]
        self.cache_size = int(cache_size)
        self.label_cache = base.LabelCache(self.cache_size, self.boundary_um)
        self.reader_cache: base.WSIReaderCache | None = None
        self.entries: list[tuple[str, int, int]] = []
        sample_coverage: dict[str, dict[str, int]] = {}
        self.label_stride = self.label_size // 2
        if self.label_stride <= 0:
            raise ValueError("Dense validation label stride must be positive")

        for record in sorted(records, key=lambda item: item.sample):
            payload = base.load_label(record, self.boundary_um)
            height, width = payload.valid.shape
            total_valid = int(payload.valid.sum())
            if total_valid <= 0:
                raise ValueError(f"{record.sample}: validation label_valid is empty")
            coverage_hits = np.zeros((height, width), dtype=np.uint16)
            tile_count = 0
            for y0 in base.dense_origins(
                height, self.label_size, self.label_stride
            ):
                for x0 in base.dense_origins(
                    width, self.label_size, self.label_stride
                ):
                    valid = base.padded_crop(
                        payload.valid,
                        y0,
                        x0,
                        self.label_size,
                        False,
                    )
                    selected = int(valid.sum())
                    if selected == 0:
                        continue
                    self.entries.append((record.sample, y0, x0))
                    crop_height = min(self.label_size, height - y0)
                    crop_width = min(self.label_size, width - x0)
                    coverage_hits[
                        y0 : y0 + crop_height, x0 : x0 + crop_width
                    ] += valid[:crop_height, :crop_width].astype(np.uint16)
                    tile_count += 1
            covered_valid = int((coverage_hits[payload.valid] > 0).sum())
            if covered_valid != total_valid:
                raise AssertionError(
                    f"{record.sample}: dense validation coverage "
                    f"{covered_valid} != {total_valid}"
                )
            sample_coverage[record.sample] = {
                "valid_pixels": total_valid,
                "covered_valid_pixels": covered_valid,
                "valid_pixel_tile_exposures": int(
                    coverage_hits[payload.valid].sum()
                ),
                "minimum_valid_pixel_tile_exposures": int(
                    coverage_hits[payload.valid].min()
                ),
                "maximum_valid_pixel_tile_exposures": int(
                    coverage_hits[payload.valid].max()
                ),
                "tiles": tile_count,
                "height": height,
                "width": width,
            }
        if not self.entries:
            raise ValueError("Dense validation grid is empty")
        self.coverage = {
            "n_samples": len(records),
            "n_patients": len(self.patients),
            "n_tiles": len(self.entries),
            "valid_pixels": int(
                sum(row["valid_pixels"] for row in sample_coverage.values())
            ),
            "covered_valid_pixels": int(
                sum(
                    row["covered_valid_pixels"]
                    for row in sample_coverage.values()
                )
            ),
            "valid_pixel_tile_exposures": int(
                sum(
                    row["valid_pixel_tile_exposures"]
                    for row in sample_coverage.values()
                )
            ),
            "samples": sample_coverage,
        }

    def __len__(self) -> int:
        return len(self.entries)

    def set_epoch(self, epoch: int) -> None:
        if int(epoch) != 0:
            raise ValueError("Dense validation grid is fixed at epoch marker 0")

    def _reader(self, record: base.SampleRecord) -> base.PyramidWSIReader:
        if self.reader_cache is None:
            self.reader_cache = base.WSIReaderCache(
                self.cache_size, self.target_mpp_um
            )
        return self.reader_cache.get(record)

    def __getitem__(self, index: int) -> dict[str, Any]:
        sample, y0, x0 = self.entries[index]
        record = self.records[sample]
        payload = self.label_cache.get(record)
        tumor = base.padded_crop(
            payload.tumor, y0, x0, self.label_size, False
        )
        valid = base.padded_crop(
            payload.valid, y0, x0, self.label_size, False
        )
        sdf_um = base.padded_crop(
            payload.sdf_um, y0, x0, self.label_size, np.nan
        )
        image = base.prepare_dense_item(
            record,
            payload,
            self._reader(record),
            y0,
            x0,
            self.label_size,
            self.input_size,
            self.target_mpp_um,
            self.mean,
            self.std,
        )
        finite_sdf = valid & np.isfinite(sdf_um)
        sdf_normalized = np.clip(
            np.nan_to_num(sdf_um, nan=0.0),
            -self.sdf_clip_um,
            self.sdf_clip_um,
        ) / self.sdf_clip_um
        return {
            "image": image,
            "tumor": torch.from_numpy(tumor[None].astype(np.float32)),
            "valid": torch.from_numpy(valid[None].astype(bool)),
            "sdf_normalized": torch.from_numpy(
                sdf_normalized[None].astype(np.float32)
            ),
            "sdf_valid": torch.from_numpy(finite_sdf[None].astype(bool)),
            "sample": record.sample,
            "patient": record.patient,
            "cancer": record.cancer,
            "label_origin_yx": torch.tensor((y0, x0), dtype=torch.int64),
        }

    def __del__(self) -> None:
        if self.reader_cache is not None:
            self.reader_cache.close()


class SampleBoundaryBatchSampler:

    def __init__(
        self,
        entries: Sequence[tuple[str, int, int]],
        batch_size: int,
    ) -> None:
        self.batches: list[list[int]] = []
        if batch_size <= 0:
            raise ValueError("Dense validation batch size must be positive")
        start = 0
        while start < len(entries):
            sample = entries[start][0]
            end = start + 1
            while end < len(entries) and entries[end][0] == sample:
                end += 1
            for batch_start in range(start, end, batch_size):
                self.batches.append(
                    list(range(batch_start, min(batch_start + batch_size, end)))
                )
            start = end
        if not self.batches:
            raise ValueError("Dense validation batch sampler is empty")

    def __iter__(self) -> Any:
        return iter(self.batches)

    def __len__(self) -> int:
        return len(self.batches)


def _create_loader_r4(
    dataset: Any,
    batch_size: int,
    workers: int,
    device: torch.device,
    seed: int,
) -> Any:
    if not isinstance(dataset, DenseValidationDataset):
        return _ORIGINAL_CREATE_LOADER(
            dataset, batch_size, workers, device, seed
        )
    generator = torch.Generator()
    generator.manual_seed(seed)

    def initialize_worker(worker_id: int) -> None:
        worker_seed = base.stable_seed(seed, worker_id)
        random.seed(worker_seed)
        np.random.seed(worker_seed)
        torch.manual_seed(worker_seed)

    arguments: dict[str, Any] = {
        "dataset": dataset,
        "batch_sampler": SampleBoundaryBatchSampler(
            dataset.entries, batch_size
        ),
        "num_workers": workers,
        "pin_memory": device.type == "cuda",
        "generator": generator,
        "worker_init_fn": initialize_worker,
    }
    if workers:
        arguments["prefetch_factor"] = 2
    return DataLoader(**arguments)


def summarize_sample_counts(
    counts: Mapping[float, Mapping[str, np.ndarray]],
    identities: Mapping[str, tuple[str, str]],
    thresholds: Sequence[float],
) -> dict[str, Any]:

    expected_samples = set(identities)
    candidates: list[dict[str, Any]] = []
    for threshold_value in thresholds:
        threshold = float(threshold_value)
        threshold_counts = counts[threshold]
        if set(threshold_counts) != expected_samples:
            raise ValueError(
                f"threshold {threshold}: incomplete sample coverage"
            )
        patient_samples: dict[str, list[tuple[float, float]]] = defaultdict(
            list
        )
        for sample, (tp, fp, fn) in sorted(threshold_counts.items()):
            dice_denominator = 2 * tp + fp + fn
            iou_denominator = tp + fp + fn
            dice = (
                float(2 * tp / dice_denominator)
                if dice_denominator
                else 1.0
            )
            iou = (
                float(tp / iou_denominator) if iou_denominator else 1.0
            )
            patient = identities[sample][0]
            patient_samples[patient].append((dice, iou))
        patient_metrics = [
            {
                "patient": patient,
                "dice": float(np.mean([value[0] for value in values])),
                "iou": float(np.mean([value[1] for value in values])),
                "n_samples": len(values),
            }
            for patient, values in sorted(patient_samples.items())
        ]
        candidates.append(
            {
                "threshold": threshold,


                "patient_metrics": patient_metrics,
                "patient_equal_dice": float(
                    np.mean([row["dice"] for row in patient_metrics])
                ),
                "patient_equal_iou": float(
                    np.mean([row["iou"] for row in patient_metrics])
                ),
                "n_patients": len(patient_metrics),
                "n_samples": len(expected_samples),
            }
        )
    selected = min(
        candidates,
        key=lambda row: (
            -float(row["patient_equal_dice"]),
            abs(float(row["threshold"]) - 0.5),
            float(row["threshold"]),
        ),
    )
    return {"selected": selected, "candidates": candidates}


class StitchedSampleThenPatientAccumulator:

    def __init__(
        self,
        thresholds: Sequence[float],
        dataset: DenseValidationDataset,
    ) -> None:
        self.thresholds = tuple(float(value) for value in thresholds)
        if not self.thresholds:
            raise ValueError("Dense threshold grid is empty")
        if any(
            not math.isfinite(value) or not 0.0 <= value <= 1.0
            for value in self.thresholds
        ):
            raise ValueError("Dense threshold grid contains invalid values")
        self.label_size = int(dataset.label_size)
        self.weight = base.blend_window(self.label_size)
        self.identities: dict[str, tuple[str, str]] = {}
        self.truth: dict[str, np.ndarray] = {}
        self.valid: dict[str, np.ndarray] = {}
        self.probability_sum: dict[str, np.ndarray] = {}
        self.weight_sum: dict[str, np.ndarray] = {}
        for sample, record in sorted(dataset.records.items()):
            payload = base.load_label(record, dataset.boundary_um)
            shape = payload.valid.shape
            self.identities[sample] = (record.patient, record.cancer)
            self.truth[sample] = payload.tumor.astype(bool)
            self.valid[sample] = payload.valid.astype(bool)
            self.probability_sum[sample] = np.zeros(shape, dtype=np.float32)
            self.weight_sum[sample] = np.zeros(shape, dtype=np.float32)

    def update(
        self,
        logits: torch.Tensor,
        samples: Sequence[str],
        origins_yx: torch.Tensor | np.ndarray | Sequence[Sequence[int]],
    ) -> None:


        probabilities = patch_probabilities_for_stitching(logits)
        if isinstance(origins_yx, torch.Tensor):
            origins = origins_yx.detach().cpu().numpy()
        else:
            origins = np.asarray(origins_yx)
        if origins.shape != (len(samples), 2):
            raise ValueError(
                "Dense validation origins do not match the inference batch"
            )
        for batch_index, sample in enumerate(samples):
            sample = str(sample)
            if sample not in self.identities:
                raise ValueError(f"{sample}: absent from R4 validation dataset")
            y0, x0 = (int(value) for value in origins[batch_index])
            height, width = self.valid[sample].shape
            if y0 < 0 or x0 < 0 or y0 >= height or x0 >= width:
                raise ValueError(f"{sample}: invalid dense origin {(y0, x0)}")
            crop_height = min(self.label_size, height - y0)
            crop_width = min(self.label_size, width - x0)
            probability = probabilities[
                batch_index, 0, :crop_height, :crop_width
            ]
            weight = self.weight[:crop_height, :crop_width]
            self.probability_sum[sample][
                y0 : y0 + crop_height, x0 : x0 + crop_width
            ] += probability * weight
            self.weight_sum[sample][
                y0 : y0 + crop_height, x0 : x0 + crop_width
            ] += weight

    def summarize(self) -> dict[str, Any]:
        counts: dict[float, dict[str, np.ndarray]] = {
            threshold: {} for threshold in self.thresholds
        }
        coverage_rows: dict[str, dict[str, int]] = {}
        for sample in sorted(self.identities):
            valid = self.valid[sample]
            covered = self.weight_sum[sample] > 0
            missed_valid = int((valid & ~covered).sum())
            if missed_valid:
                raise RuntimeError(
                    f"{sample}: Hann stitching missed {missed_valid} valid pixels"
                )
            canvas = np.zeros(valid.shape, dtype=np.float32)
            canvas[covered] = (
                self.probability_sum[sample][covered]
                / self.weight_sum[sample][covered]
            )
            probability = probabilities_for_thresholding(canvas)
            truth = self.truth[sample]
            for threshold in self.thresholds:
                prediction = probability >= threshold
                tp = int((prediction & truth & valid).sum())
                fp = int((prediction & ~truth & valid).sum())
                fn = int((~prediction & truth & valid).sum())
                counts[threshold][sample] = np.asarray(
                    (tp, fp, fn), dtype=np.int64
                )
            coverage_rows[sample] = {
                "valid_pixels": int(valid.sum()),
                "covered_valid_pixels": int((valid & covered).sum()),
                "canvas_pixels_with_positive_weight": int(covered.sum()),
            }
        summary = summarize_sample_counts(
            counts, self.identities, self.thresholds
        )
        summary.update(
            {
            "aggregation": (
                "dense counts within sample; sample metric mean within patient; "
                "patient-equal cohort mean"
            ),
            "probability_precision": (
                "tile sigmoid float32; float32 Hann stitch; stitched canvas "
                "rounded through float16 once before thresholding"
            ),
            "stitching": {
                "label_stride": self.label_size // 2,
                "blend": "nonzero_hann",
                "coverage": coverage_rows,
            },
            }
        )
        return summary


def patch_probabilities_for_stitching(logits: torch.Tensor) -> np.ndarray:

    return torch.sigmoid(logits.detach().float()).cpu().numpy()


def probabilities_for_thresholding(canvas: np.ndarray) -> np.ndarray:

    return np.asarray(canvas, dtype=np.float32).astype(np.float16).astype(
        np.float32
    )


@torch.no_grad()
def validate_dense_tiles(
    model: torch.nn.Module,
    loader: Any,
    device: torch.device,
    thresholds: Sequence[float],
    loss_weights: Mapping[str, float],
    use_bf16: bool,
) -> dict[str, Any]:
    model.eval()
    if not isinstance(loader.dataset, DenseValidationDataset):
        raise TypeError("Dense validation requires DenseValidationDataset")
    accumulator = StitchedSampleThenPatientAccumulator(
        thresholds, loader.dataset
    )
    total_loss = 0.0
    n_batches = 0
    for batch in loader:
        for key in ("image", "tumor", "valid", "sdf_normalized", "sdf_valid"):
            batch[key] = batch[key].to(device, non_blocking=True)
        with base.autocast_context(device, use_bf16):
            outputs = model(batch["image"])
            loss, _ = base.masked_region_loss(outputs, batch, loss_weights)
        accumulator.update(
            outputs["tumor_logits"],
            list(batch["sample"]),
            batch["label_origin_yx"],
        )
        total_loss += float(loss)
        n_batches += 1
    summary = accumulator.summarize()
    summary["loss"] = total_loss / max(n_batches, 1)
    summary["metric_scope"] = (
        "full half-stride nonzero-Hann stitched valid tissue; each valid "
        "10-um pixel scored once after full-canvas float16 persistence"
    )
    return summary


def _make_tile_dataset_r4(
    records: Sequence[base.SampleRecord],
    *,
    experiment: Mapping[str, Any],
    candidates_document: Mapping[str, Any],
    tiles_per_patient: int,
    seed: int,
    training: bool,
    stain_augmentation: bool,
) -> Any:
    del tiles_per_patient, seed
    if training:
        return _ORIGINAL_MAKE_TILE_DATASET(
            records,
            experiment=experiment,
            candidates_document=candidates_document,
            tiles_per_patient=int(
                candidates_document["training_runtime"][
                    "tiles_per_patient_train"
                ]
            ),
            seed=base.stable_seed(
                int(experiment["training"]["seed"]),
                int(_WRAPPER_ARGS.outer_fold),
                str(_WRAPPER_ARGS.mode),
                "train",
            ),
            training=True,
            stain_augmentation=stain_augmentation,
        )
    runtime = candidates_document["training_runtime"]
    declared = runtime.get("validation_contract")
    required = {
        "grid": "full_valid_tissue_half_stride_hann",
        "origins": "dense_origins_stride_128_with_bottom_right_tail_alignment",
        "blend": "nonzero_hann",
        "batching": "batch_size_4_flush_at_each_sample_boundary",
        "coverage": "each_valid_10um_pixel_scored_once_after_stitching",
        "probability_precision": (
            "tile_sigmoid_float32_then_float32_hann_stitch_then_"
            "canvas_float16_roundtrip"
        ),
        "aggregation": "sample_metric_then_mean_within_patient_then_patient_equal",
        "threshold_tie_break": "closest_to_0.5_then_lower",
    }
    if not isinstance(declared, dict) or any(
        declared.get(key) != value for key, value in required.items()
    ):
        raise ValueError("Dense candidates config lacks the dense validation contract")
    preprocessing = experiment["preprocessing"]
    input_contract = candidates_document["input_contract"]
    dataset = DenseValidationDataset(
        records,
        label_size=int(preprocessing["label_shape_yx"][0]),
        input_size=int(preprocessing["model_input_shape_yx"][0]),
        target_mpp_um=float(preprocessing["model_input_mpp_um"]),
        boundary_um=float(experiment["sampling"]["boundary_candidate_um"]),
        sdf_clip_um=float(experiment["model"]["sdf_clip_um"]),
        normalization_mean=input_contract["normalization"]["mean"],
        normalization_std=input_contract["normalization"]["std"],
        cache_size=int(runtime["label_cache_size"]),
    )
    _write_validation_contract(dataset, experiment, candidates_document, declared)
    return dataset


def _write_validation_contract(
    dataset: DenseValidationDataset,
    experiment: Mapping[str, Any],
    candidates_document: Mapping[str, Any],
    declared: Mapping[str, Any],
) -> None:
    global _LAST_VALIDATION_CONTRACT
    assert _WRAPPER_ARGS is not None
    config_path = Path(_WRAPPER_ARGS.config).resolve()
    candidates_path = Path(_WRAPPER_ARGS.candidates_config).resolve()
    root = config_path.parent.parent
    source_paths = {
        "models": root / "scripts" / "models.py",
        "dataset": root / "scripts" / "he_mask_dataset.py",
        "preflight": root / "scripts" / "preflight_nested.py",
        "input_auditor": root / "scripts" / "audit_input_content.py",
        "input_receipt_validator": (
            root / "scripts" / "validate_input_content_receipt.py"
        ),
        "source_manifest": root / experiment["outputs"]["source_manifest"],
        "data_contract": root / experiment["outputs"]["data_contract"],
        "label_manifest": root / experiment["outputs"]["label_manifest"],
        "label_dataset_contract": (
            root / experiment["outputs"]["label_dataset_contract"]
        ),
        "input_content_receipt": (
            root / "data" / "manifests" / "input_content_receipt.json"
        ),
        "tile_index": root / "data" / "manifests" / "tile_index.parquet",
        "tile_index_receipt": (
            root / "data" / "manifests" / "tile_index_receipt.json"
        ),
    }
    missing_sources = [
        str(path) for path in source_paths.values() if not path.is_file()
    ]
    if missing_sources:
        raise FileNotFoundError(
            "Dense validation provenance sources are missing: "
            + ", ".join(missing_sources)
        )
    input_receipt = read_json(source_paths["input_content_receipt"])
    input_validation = input_receipt_validator.validate_receipt(
        root, source_paths["input_content_receipt"]
    )
    input_receipt_copy = dict(input_receipt)
    claimed_input_hash = input_receipt_copy.pop(
        "receipt_payload_sha256", None
    )
    if claimed_input_hash != canonical_sha256(input_receipt_copy):
        raise ValueError("Dense input-content receipt payload hash differs")
    if (
        input_receipt.get("status")
        != "COHORT_LABEL_AND_HE_CONTENT_VERIFIED"
    ):
        raise ValueError("Dense input-content receipt is incomplete")
    input_content_trees: dict[str, dict[str, Any]] = {}
    for name in ("aligned_he", "dense_labels"):
        tree = input_receipt.get("content_trees", {}).get(name, {})
        expected_files = int(input_receipt["scope"]["n_samples"])
        if int(tree.get("n_files", -1)) != expected_files:
            raise ValueError(
                f"Dense {name} input tree does not close the declared cohort "
                f"({expected_files} samples)"
            )
        input_content_trees[name] = {
            "n_files": expected_files,
            "receipt_tree_sha256": str(tree.get("tree_sha256", "")),
            "portable_tree_sha256": str(
                input_validation["content_trees"][name][
                    "portable_tree_sha256"
                ]
            ),
        }
    arm = "stainaug" if _WRAPPER_ARGS.stain_augmentation else "native_rgb"
    phase_root = (
        root
        / experiment["outputs"]["models"]
        / _WRAPPER_ARGS.candidate
        / arm
        / f"fold{_WRAPPER_ARGS.outer_fold}"
        / "tune"
    )
    path = phase_root / "VALIDATION_CONTRACT.json"
    payload: dict[str, Any] = {
        "schema_version": SCHEMA,
        "status": "FULL_VALID_TISSUE_GRID_VALIDATED",
        "candidate": _WRAPPER_ARGS.candidate,
        "arm": arm,
        "outer_fold": int(_WRAPPER_ARGS.outer_fold),
        "selection_metric": SELECTION_METRIC,
        "declared_contract": dict(declared),
        "threshold_candidates": [
            float(value)
            for value in candidates_document["training_runtime"][
                "threshold_candidates"
            ]
        ],
        "coverage": dataset.coverage,
        "input_content_trees": input_content_trees,
        "sources": {
            "wrapper": {
                "path": str(Path(__file__).resolve()),
                "sha256": sha256_file(Path(__file__).resolve()),
            },
            "base_training": {
                "path": str(Path(base.__file__).resolve()),
                "sha256": sha256_file(Path(base.__file__).resolve()),
            },
            "config": {
                "path": str(config_path),
                "sha256": sha256_file(config_path),
                "canonical_sha256": canonical_sha256(experiment),
            },
            "candidates_config": {
                "path": str(candidates_path),
                "sha256": sha256_file(candidates_path),
                "canonical_sha256": canonical_sha256(candidates_document),
            },
            **{
                name: {
                    "path": str(source_path.resolve()),
                    "sha256": sha256_file(source_path),
                    "size_bytes": source_path.stat().st_size,
                }
                for name, source_path in source_paths.items()
            },
        },
    }
    payload["contract_payload_sha256"] = canonical_sha256(payload)
    atomic_write_once_or_equal(path, payload)
    _LAST_VALIDATION_CONTRACT = {
        "path": str(path.resolve()),
        "sha256": sha256_file(path),
        "payload_sha256": payload["contract_payload_sha256"],
    }


def _postprocess_tune_summary() -> None:
    if _WRAPPER_ARGS is None or _WRAPPER_ARGS.mode != "tune":
        return
    if _LAST_VALIDATION_CONTRACT is None:
        raise RuntimeError("Dense tune completed without a validation contract")
    config = read_json(Path(_WRAPPER_ARGS.config))
    root = Path(_WRAPPER_ARGS.config).resolve().parent.parent
    arm = "stainaug" if _WRAPPER_ARGS.stain_augmentation else "native_rgb"
    tune_root = (
        root
        / config["outputs"]["models"]
        / _WRAPPER_ARGS.candidate
        / arm
        / f"fold{_WRAPPER_ARGS.outer_fold}"
        / "tune"
    )
    summary_path = tune_root / "summary.json"
    success_path = tune_root / "SUCCESS.json"
    summary = read_json(summary_path)
    success = read_json(success_path)
    if summary.get("phase") != "tune" or success.get("status") != "COMPLETED":
        raise ValueError("Dense tune did not produce a completed tune summary")
    summary["selection_metric"] = SELECTION_METRIC
    summary["r4_validation_contract"] = dict(_LAST_VALIDATION_CONTRACT)
    atomic_json(summary_path, summary)
    success["r4_validation_contract"] = dict(_LAST_VALIDATION_CONTRACT)
    success["r4_entrypoint_sha256"] = sha256_file(Path(__file__).resolve())
    atomic_json(success_path, success)


def parse_wrapper_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument(
        "--config",
        type=Path,
        default=root / "configs" / "experiment.json",
    )
    parser.add_argument(
        "--candidates-config",
        type=Path,
        default=root / "configs" / "model_candidates.json",
    )
    parser.add_argument("--mode", required=True)
    parser.add_argument("--candidate", required=True)
    parser.add_argument("--outer-fold", type=int, default=0)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--stain-augmentation", action="store_true")
    arguments, _ = parser.parse_known_args()
    arguments.config = arguments.config.resolve()
    arguments.candidates_config = arguments.candidates_config.resolve()
    if arguments.mode not in {"profile", "tune", "refit", "evaluate"}:
        raise ValueError(f"unsupported mode: {arguments.mode}")
    if arguments.outer_fold not in range(5):
        raise ValueError("outer fold must be in [0,4]")
    if (
        arguments.mode in {"tune", "refit", "evaluate"}
        and arguments.batch_size != 4
    ):
        raise ValueError(
            "Dense tune/refit/evaluate batch size is frozen at 4 so inner "
            "validation and outer OOF use identical forward batch shapes"
        )
    candidates = read_json(arguments.candidates_config)
    if arguments.candidate not in candidates.get("candidates", {}):
        raise ValueError(f"unknown candidate: {arguments.candidate}")
    if "validation_contract" not in candidates.get("training_runtime", {}):
        raise ValueError("Dense candidates config is required")
    return arguments


class _BaseTempfileProxy:

    @staticmethod
    def NamedTemporaryFile(*args: Any, **kwargs: Any) -> Any:
        if kwargs.get("suffix") == ".npz":
            kwargs.setdefault("prefix", ".r4tmp-")
        return tempfile.NamedTemporaryFile(*args, **kwargs)


def _recover_prediction_temporaries(arguments: argparse.Namespace) -> None:

    if arguments.mode != "evaluate":
        return
    experiment = read_json(arguments.config)
    root = arguments.config.parent.parent
    arm = "stainaug" if arguments.stain_augmentation else "native_rgb"
    prediction_root = (
        root
        / experiment["outputs"]["predictions"]
        / arguments.candidate
        / arm
        / f"fold{arguments.outer_fold}"
    )
    removed: list[str] = []
    if prediction_root.is_dir():
        for path in sorted(prediction_root.rglob(".r4tmp-*.npz")):
            if path.is_symlink() or not path.is_file():
                raise RuntimeError(
                    f"refusing unsafe R4 temporary recovery target: {path}"
                )
            path.unlink()
            removed.append(str(path))
    if removed:
        print(
            f"recovered_interrupted_r4_prediction_temporaries={len(removed)}",
            flush=True,
        )


def main() -> None:
    global _WRAPPER_ARGS
    if "--help" in sys.argv[1:] or "-h" in sys.argv[1:]:
        base.parse_args()
        return
    _WRAPPER_ARGS = parse_wrapper_args()
    _recover_prediction_temporaries(_WRAPPER_ARGS)
    base.make_tile_dataset = _make_tile_dataset_r4
    base.create_loader = _create_loader_r4
    base.validate_tiles = validate_dense_tiles
    base.tempfile = _BaseTempfileProxy()
    base.main()
    _postprocess_tune_summary()


if __name__ == "__main__":
    main()
