#!/usr/bin/env python3

from __future__ import annotations

import argparse
import contextlib
import hashlib
import json
import math
import os
import random
import resource
import tempfile
import time
from collections import OrderedDict, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd
import torch
from scipy.ndimage import binary_erosion, distance_transform_edt
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset

from models import build_model, summarize_model
from models_fm import build_fm_model, is_foundation_candidate
from he_mask_dataset import OmePyramidReader


SCHEMA = "he_boundary_prediction.v1.region_training.v1"
LABEL_MPP_UM = 10.0


def canonical_json_sha256(payload: Any) -> str:
    encoded = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def sha256_file(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def training_arm(stain_augmentation: bool) -> str:
    return "stainaug" if stain_augmentation else "native_rgb"


def artifact_paths(
    project_root: Path,
    experiment: Mapping[str, Any],
    candidate: str,
    arm: str,
    outer_fold: int,
) -> dict[str, Path]:
    if arm not in {"native_rgb", "stainaug"}:
        raise ValueError(f"unsupported training arm: {arm}")
    return {
        "model_root": (
            project_root
            / experiment["outputs"]["models"]
            / candidate
            / arm
            / f"fold{outer_fold}"
        ),
        "prediction_root": (
            project_root
            / experiment["outputs"]["predictions"]
            / candidate
            / arm
            / f"fold{outer_fold}"
        ),
        "result_path": (
            project_root
            / experiment["outputs"]["results"]
            / candidate
            / arm
            / f"fold{outer_fold}_outer_metrics.json"
        ),
    }


def stable_seed(*parts: Any) -> int:
    encoded = "\x1f".join(str(part) for part in parts).encode("utf-8")
    return int.from_bytes(hashlib.sha256(encoded).digest()[:8], "little") % (
        2**32
    )


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
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
    try:
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def atomic_torch_save(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        dir=path.parent, suffix=".pt", delete=False
    ) as handle:
        temporary = Path(handle.name)
    try:
        torch.save(dict(payload), temporary)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True, warn_only=True)


def capture_rng_state() -> dict[str, Any]:
    state: dict[str, Any] = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        state["cuda"] = torch.cuda.get_rng_state_all()
    return state


def restore_rng_state(state: Mapping[str, Any]) -> None:
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])
    if torch.cuda.is_available() and "cuda" in state:
        torch.cuda.set_rng_state_all(state["cuda"])


@dataclass(frozen=True)
class SampleRecord:
    sample: str
    patient: str
    cancer: str
    fold: int
    he_path: Path
    he_size_y_px: int
    he_size_x_px: int
    he_mpp_um: float
    label_path: Path


def load_records(manifest_path: Path) -> list[SampleRecord]:
    table = pd.read_csv(manifest_path, keep_default_na=False)
    required = {
        "sample",
        "patient",
        "cancer",
        "fold",
        "he_path",
        "he_size_y_px",
        "he_size_x_px",
        "he_mpp_um",
        "label_path",
    }
    missing = required.difference(table.columns)
    if missing:
        raise ValueError(f"source manifest is missing: {sorted(missing)}")
    if table["sample"].duplicated().any():
        raise ValueError("source manifest has duplicate samples")
    records = [
        SampleRecord(
            sample=str(row.sample),
            patient=str(row.patient),
            cancer=str(row.cancer),
            fold=int(row.fold),
            he_path=Path(str(row.he_path)),
            he_size_y_px=int(row.he_size_y_px),
            he_size_x_px=int(row.he_size_x_px),
            he_mpp_um=float(row.he_mpp_um),
            label_path=Path(str(row.label_path)),
        )
        for row in table.itertuples(index=False)
    ]
    patient_folds: dict[str, set[int]] = defaultdict(set)
    patient_cancers: dict[str, set[str]] = defaultdict(set)
    for record in records:
        patient_folds[record.patient].add(record.fold)
        patient_cancers[record.patient].add(record.cancer)
    bad_fold = {
        patient: sorted(folds)
        for patient, folds in patient_folds.items()
        if len(folds) != 1
    }
    bad_cancer = {
        patient: sorted(cancers)
        for patient, cancers in patient_cancers.items()
        if len(cancers) != 1
    }
    if bad_fold:
        raise ValueError(f"patients span outer folds: {bad_fold}")
    if bad_cancer:
        raise ValueError(f"patients span cancers: {bad_cancer}")
    return records


def make_nested_split(
    records: Sequence[SampleRecord],
    outer_fold: int,
    seed: int,
    validation_fraction: float,
) -> dict[str, Any]:
    if outer_fold not in range(5):
        raise ValueError("outer_fold must be in [0,4]")
    if not 0.0 < validation_fraction < 0.5:
        raise ValueError("inner validation fraction must be in (0,0.5)")
    patient_to_cancer = {
        record.patient: record.cancer for record in records
    }
    outer_test = sorted(
        {record.patient for record in records if record.fold == outer_fold}
    )
    outer_train = sorted(
        {record.patient for record in records if record.fold != outer_fold}
    )
    by_cancer: dict[str, list[str]] = defaultdict(list)
    for patient in outer_train:
        by_cancer[patient_to_cancer[patient]].append(patient)
    inner_validation: list[str] = []
    for cancer, patients in sorted(by_cancer.items()):
        ranked = sorted(
            patients,
            key=lambda patient: (
                stable_seed(seed, outer_fold, cancer, patient),
                patient,
            ),
        )
        count = int(round(len(ranked) * validation_fraction))
        if len(ranked) >= 2:
            count = max(1, min(count, len(ranked) - 1))
        else:
            count = 0
        inner_validation.extend(ranked[:count])
    inner_validation = sorted(inner_validation)
    inner_training = sorted(set(outer_train).difference(inner_validation))
    sets = [set(inner_training), set(inner_validation), set(outer_test)]
    if any(left.intersection(right) for i, left in enumerate(sets) for right in sets[i + 1 :]):
        raise AssertionError("nested patient sets overlap")
    if set(inner_training).union(inner_validation) != set(outer_train):
        raise AssertionError("inner split does not cover outer-training patients")
    payload = {
        "schema_version": f"{SCHEMA}.nested_split.v1",
        "outer_fold": outer_fold,
        "seed": seed,
        "validation_fraction": validation_fraction,
        "inner_training_patients": inner_training,
        "inner_validation_patients": inner_validation,
        "outer_training_patients": outer_train,
        "outer_test_patients": outer_test,
    }
    payload["split_sha256"] = canonical_json_sha256(payload)
    return payload


def select_records(
    records: Sequence[SampleRecord], patients: Iterable[str]
) -> list[SampleRecord]:
    allowed = set(patients)
    return [record for record in records if record.patient in allowed]


@dataclass
class LabelPayload:
    tumor: np.ndarray
    valid: np.ndarray
    sdf_um: np.ndarray
    metadata: dict[str, Any]
    candidate_pixels: dict[str, np.ndarray]


def load_label(record: SampleRecord, boundary_um: float) -> LabelPayload:
    if not record.label_path.is_file():
        raise FileNotFoundError(
            f"dense label is absent for {record.sample}: {record.label_path}; "
            "complete the full label export before training"
        )
    with np.load(record.label_path, allow_pickle=False) as archive:
        required = {"tumor_mask", "label_valid", "sdf_um", "metadata_json"}
        missing = required.difference(archive.files)
        if missing:
            raise ValueError(
                f"{record.sample}: label archive is missing {sorted(missing)}"
            )
        tumor = archive["tumor_mask"].astype(bool, copy=True)
        valid = archive["label_valid"].astype(bool, copy=True)
        sdf_um = archive["sdf_um"].astype(np.float32, copy=True)
        metadata = json.loads(str(archive["metadata_json"].item()))
    if tumor.shape != valid.shape or tumor.shape != sdf_um.shape:
        raise ValueError(f"{record.sample}: label array shapes differ")
    if metadata.get("sample") != record.sample:
        raise ValueError(f"{record.sample}: label metadata sample differs")
    if metadata.get("patient") != record.patient:
        raise ValueError(f"{record.sample}: label metadata patient differs")
    if int(metadata.get("fold", -1)) != record.fold:
        raise ValueError(f"{record.sample}: label metadata fold differs")
    resolution = float(metadata.get("resolution_um", float("nan")))
    if not math.isclose(resolution, LABEL_MPP_UM, rel_tol=0.0, abs_tol=1e-9):
        raise ValueError(f"{record.sample}: label resolution is not 10 um")
    finite = np.isfinite(sdf_um)
    candidate_masks = {
        "boundary": valid & finite & (np.abs(sdf_um) <= boundary_um),
        "tumor_interior": valid
        & tumor
        & finite
        & (sdf_um < -boundary_um),
        "normal_tissue": valid
        & ~tumor
        & finite
        & (sdf_um > boundary_um),
    }
    fallbacks = {
        "boundary": valid & finite,
        "tumor_interior": valid & tumor,
        "normal_tissue": valid & ~tumor,
    }
    candidates: dict[str, np.ndarray] = {}
    for name, mask in candidate_masks.items():
        if not mask.any():
            mask = fallbacks[name]
        candidates[name] = np.flatnonzero(mask)
    return LabelPayload(tumor, valid, sdf_um, metadata, candidates)


class LabelCache:
    def __init__(self, maximum: int, boundary_um: float) -> None:
        self.maximum = max(1, int(maximum))
        self.boundary_um = float(boundary_um)
        self._items: OrderedDict[str, LabelPayload] = OrderedDict()

    def get(self, record: SampleRecord) -> LabelPayload:
        if record.sample in self._items:
            payload = self._items.pop(record.sample)
            self._items[record.sample] = payload
            return payload
        payload = load_label(record, self.boundary_um)
        self._items[record.sample] = payload
        while len(self._items) > self.maximum:
            self._items.popitem(last=False)
        return payload


class PyramidWSIReader:

    def __init__(
        self,
        record: SampleRecord,
        target_mpp_um: float,
    ) -> None:
        if not record.he_path.is_file():
            raise FileNotFoundError(record.he_path)
        self.record = record
        self.target_mpp_um = float(target_mpp_um)
        self._reader = OmePyramidReader(
            record.he_path,
            (record.he_size_y_px, record.he_size_x_px),
            record.he_mpp_um,
            self.target_mpp_um,
        )
        self.level_index = self._reader.level
        self.level_mpp_y_um = self._reader.level_mpp_y_um
        self.level_mpp_x_um = self._reader.level_mpp_x_um

    def read_physical(
        self,
        x0_um: float,
        y0_um: float,
        field_um: float,
        output_size: int,
    ) -> np.ndarray:
        output_mpp_um = field_um / output_size
        if not math.isclose(
            output_mpp_um,
            self.target_mpp_um,
            rel_tol=0.0,
            abs_tol=1e-9,
        ):
            raise ValueError(
                "requested physical field does not match the model MPP"
            )
        return self._reader.read_physical_window(
            x0_um=x0_um,
            y0_um=y0_um,
            output_shape_yx=(output_size, output_size),
            output_mpp_um=output_mpp_um,
        )

    def close(self) -> None:
        reader = getattr(self, "_reader", None)
        if reader is not None:
            reader.close()
        self._reader = None

    def __del__(self) -> None:
        self.close()


class WSIReaderCache:
    def __init__(self, maximum: int, target_mpp_um: float) -> None:
        self.maximum = max(1, int(maximum))
        self.target_mpp_um = float(target_mpp_um)
        self._readers: OrderedDict[str, PyramidWSIReader] = OrderedDict()

    def get(self, record: SampleRecord) -> PyramidWSIReader:
        if record.sample in self._readers:
            reader = self._readers.pop(record.sample)
            self._readers[record.sample] = reader
            return reader
        reader = PyramidWSIReader(record, self.target_mpp_um)
        self._readers[record.sample] = reader
        while len(self._readers) > self.maximum:
            _, discarded = self._readers.popitem(last=False)
            discarded.close()
        return reader

    def close(self) -> None:
        for reader in self._readers.values():
            reader.close()
        self._readers.clear()


def padded_crop(
    array: np.ndarray,
    y0: int,
    x0: int,
    size: int,
    fill: float | bool,
) -> np.ndarray:
    output = np.full((size, size), fill, dtype=array.dtype)
    y1, x1 = y0 + size, x0 + size
    source_y0, source_x0 = max(0, y0), max(0, x0)
    source_y1, source_x1 = min(array.shape[0], y1), min(array.shape[1], x1)
    if source_y1 > source_y0 and source_x1 > source_x0:
        output[
            source_y0 - y0 : source_y1 - y0,
            source_x0 - x0 : source_x1 - x0,
        ] = array[source_y0:source_y1, source_x0:source_x1]
    return output


def choose_tile_origin(
    payload: LabelPayload,
    stratum: str,
    tile_size: int,
    minimum_valid_fraction: float,
    rng: np.random.Generator,
) -> tuple[int, int]:
    candidates = payload.candidate_pixels[stratum]
    if not len(candidates):
        raise ValueError(f"label has no candidate pixels for {stratum}")
    best: tuple[float, int, int] | None = None
    for _ in range(24):
        flat = int(candidates[int(rng.integers(0, len(candidates)))])
        center_y, center_x = np.unravel_index(flat, payload.valid.shape)
        y0 = int(center_y) - tile_size // 2
        x0 = int(center_x) - tile_size // 2
        valid = padded_crop(payload.valid, y0, x0, tile_size, False)
        fraction = float(valid.mean())
        if best is None or fraction > best[0]:
            best = (fraction, y0, x0)
        if fraction >= minimum_valid_fraction:
            return y0, x0
    if best is None or best[0] < minimum_valid_fraction:
        observed = 0.0 if best is None else best[0]
        raise ValueError(
            f"no {stratum} tile meets valid fraction "
            f"{minimum_valid_fraction:.3f}; best={observed:.3f}"
        )
    return best[1], best[2]


def optical_density_jitter(
    rgb: np.ndarray, rng: np.random.Generator
) -> np.ndarray:

    values = (rgb.astype(np.float32) + 1.0) / 256.0
    optical_density = -np.log(np.clip(values, 1.0 / 256.0, 1.0))
    scale = rng.uniform(0.88, 1.12, size=(1, 1, 3)).astype(np.float32)
    shift = rng.uniform(-0.04, 0.04, size=(1, 1, 3)).astype(np.float32)
    perturbed = np.exp(-(optical_density * scale + shift))
    return np.clip(perturbed * 256.0 - 1.0, 0, 255).astype(np.uint8)


def joint_spatial_augmentation(
    rgb: np.ndarray,
    tumor: np.ndarray,
    valid: np.ndarray,
    sdf_um: np.ndarray,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    rotations = int(rng.integers(0, 4))
    if rotations:
        rgb = np.rot90(rgb, rotations, axes=(0, 1))
        tumor = np.rot90(tumor, rotations)
        valid = np.rot90(valid, rotations)
        sdf_um = np.rot90(sdf_um, rotations)
    if bool(rng.integers(0, 2)):
        rgb = np.flip(rgb, axis=1)
        tumor = np.flip(tumor, axis=1)
        valid = np.flip(valid, axis=1)
        sdf_um = np.flip(sdf_um, axis=1)
    if bool(rng.integers(0, 2)):
        rgb = np.flip(rgb, axis=0)
        tumor = np.flip(tumor, axis=0)
        valid = np.flip(valid, axis=0)
        sdf_um = np.flip(sdf_um, axis=0)
    return tuple(
        np.ascontiguousarray(array)
        for array in (rgb, tumor, valid, sdf_um)
    )


class RegionTileDataset(Dataset[dict[str, Any]]):

    def __init__(
        self,
        records: Sequence[SampleRecord],
        tiles_per_patient: int,
        label_size: int,
        input_size: int,
        target_mpp_um: float,
        boundary_um: float,
        stratum_names: Sequence[str],
        stratum_weights: Sequence[float],
        minimum_valid_fraction: float,
        sdf_clip_um: float,
        normalization_mean: Sequence[float],
        normalization_std: Sequence[float],
        seed: int,
        training: bool,
        stain_augmentation: bool,
        cache_size: int,
    ) -> None:
        if not records:
            raise ValueError("tile dataset has no records")
        self.records_by_patient: dict[str, list[SampleRecord]] = defaultdict(list)
        for record in records:
            self.records_by_patient[record.patient].append(record)
        self.patients = sorted(self.records_by_patient)
        self.tiles_per_patient = int(tiles_per_patient)
        self.label_size = int(label_size)
        self.input_size = int(input_size)
        self.target_mpp_um = float(target_mpp_um)
        self.field_um = self.label_size * LABEL_MPP_UM
        expected_field = self.input_size * self.target_mpp_um
        if not math.isclose(
            self.field_um, expected_field, rel_tol=0.0, abs_tol=1e-6
        ):
            raise ValueError("RGB and label physical fields differ")
        self.stratum_names = tuple(stratum_names)
        weights = np.asarray(stratum_weights, dtype=np.float64)
        if len(weights) != len(self.stratum_names) or np.any(weights < 0):
            raise ValueError("invalid tile-stratum weights")
        self.stratum_probabilities = weights / weights.sum()
        self.minimum_valid_fraction = float(minimum_valid_fraction)
        self.sdf_clip_um = float(sdf_clip_um)
        self.mean = np.asarray(normalization_mean, dtype=np.float32)[:, None, None]
        self.std = np.asarray(normalization_std, dtype=np.float32)[:, None, None]
        self.seed = int(seed)
        self.training = bool(training)
        self.stain_augmentation = bool(stain_augmentation)
        self.epoch = 0
        self.label_cache = LabelCache(cache_size, boundary_um)
        self.reader_cache: WSIReaderCache | None = None
        self.cache_size = cache_size

    def __len__(self) -> int:
        return len(self.patients) * self.tiles_per_patient

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def _reader(self, record: SampleRecord) -> PyramidWSIReader:
        if self.reader_cache is None:
            self.reader_cache = WSIReaderCache(
                self.cache_size, self.target_mpp_um
            )
        return self.reader_cache.get(record)

    def __getitem__(self, index: int) -> dict[str, Any]:
        patient = self.patients[index % len(self.patients)]
        rng = np.random.default_rng(
            stable_seed(self.seed, self.epoch, index, patient)
        )
        records = self.records_by_patient[patient]
        failures: list[str] = []
        for _ in range(max(24, len(records) * 8)):
            record = records[int(rng.integers(0, len(records)))]
            payload = self.label_cache.get(record)
            available_strata = [
                name
                for name in self.stratum_names
                if len(payload.candidate_pixels[name])
            ]
            if not available_strata:
                failures.append(f"{record.sample}:no_nonempty_stratum")
                continue
            available_indices = [
                self.stratum_names.index(name) for name in available_strata
            ]
            available_probabilities = self.stratum_probabilities[
                available_indices
            ]
            available_probabilities = (
                available_probabilities / available_probabilities.sum()
            )
            stratum = str(
                rng.choice(available_strata, p=available_probabilities)
            )
            try:
                y0, x0 = choose_tile_origin(
                    payload,
                    stratum,
                    self.label_size,
                    self.minimum_valid_fraction,
                    rng,
                )
                break
            except ValueError as error:
                failures.append(f"{record.sample}:{stratum}:{error}")
        else:
            raise RuntimeError(
                f"no eligible tile for patient {patient}; "
                + "; ".join(failures[-4:])
            )
        tumor = padded_crop(
            payload.tumor, y0, x0, self.label_size, False
        )
        valid = padded_crop(
            payload.valid, y0, x0, self.label_size, False
        )
        sdf_um = padded_crop(
            payload.sdf_um, y0, x0, self.label_size, np.nan
        )
        x0_um = float(payload.metadata["x_min_um"]) + x0 * LABEL_MPP_UM
        y0_um = float(payload.metadata["y_min_um"]) + y0 * LABEL_MPP_UM
        rgb = self._reader(record).read_physical(
            x0_um,
            y0_um,
            self.field_um,
            self.input_size,
        )
        if self.training:
            rgb, tumor, valid, sdf_um = joint_spatial_augmentation(
                rgb, tumor, valid, sdf_um, rng
            )
            if self.stain_augmentation:
                rgb = optical_density_jitter(rgb, rng)
        image = np.moveaxis(rgb.astype(np.float32) / 255.0, -1, 0)
        image = (image - self.mean) / self.std
        finite_sdf = valid & np.isfinite(sdf_um)
        sdf_normalized = np.clip(
            np.nan_to_num(sdf_um, nan=0.0), -self.sdf_clip_um, self.sdf_clip_um
        ) / self.sdf_clip_um
        return {
            "image": torch.from_numpy(np.ascontiguousarray(image)),
            "tumor": torch.from_numpy(tumor[None].astype(np.float32)),
            "valid": torch.from_numpy(valid[None].astype(bool)),
            "sdf_normalized": torch.from_numpy(
                sdf_normalized[None].astype(np.float32)
            ),
            "sdf_valid": torch.from_numpy(finite_sdf[None].astype(bool)),
            "sample": record.sample,
            "patient": record.patient,
            "cancer": record.cancer,
            "stratum": stratum,
            "label_origin_yx": torch.tensor((y0, x0), dtype=torch.int64),
        }

    def __del__(self) -> None:
        if self.reader_cache is not None:
            self.reader_cache.close()


def masked_region_loss(
    outputs: Mapping[str, torch.Tensor],
    batch: Mapping[str, Any],
    weights: Mapping[str, float],
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    logits = outputs["tumor_logits"].float()
    sdf_prediction = outputs["sdf_normalized"].float()
    target = batch["tumor"].float()
    valid = batch["valid"].bool()
    sdf_target = batch["sdf_normalized"].float()
    sdf_valid = batch["sdf_valid"].bool() & valid
    if logits.shape != target.shape or sdf_prediction.shape != target.shape:
        raise ValueError(
            f"model output {tuple(logits.shape)} differs from "
            f"label {tuple(target.shape)}"
        )
    valid_float = valid.float()
    denominator = valid_float.sum().clamp_min(1.0)
    cross_entropy = (
        F.binary_cross_entropy_with_logits(logits, target, reduction="none")
        * valid_float
    ).sum() / denominator
    probability = torch.sigmoid(logits) * valid_float
    target_valid = target * valid_float
    intersection = (probability * target_valid).sum(dim=(1, 2, 3))
    dice = (
        2.0 * intersection + 1.0
    ) / (
        probability.sum(dim=(1, 2, 3))
        + target_valid.sum(dim=(1, 2, 3))
        + 1.0
    )
    dice_loss = 1.0 - dice.mean()
    if sdf_valid.any():
        sdf_loss = F.smooth_l1_loss(
            sdf_prediction[sdf_valid],
            sdf_target[sdf_valid],
            beta=0.1,
        )
    else:
        sdf_loss = sdf_prediction.sum() * 0.0
    total = (
        float(weights["masked_cross_entropy"]) * cross_entropy
        + float(weights["soft_dice"]) * dice_loss
        + float(weights["masked_sdf_smooth_l1"]) * sdf_loss
    )
    return total, {
        "cross_entropy": cross_entropy.detach(),
        "dice_loss": dice_loss.detach(),
        "sdf_loss": sdf_loss.detach(),
    }


class PatientThresholdAccumulator:
    def __init__(self, thresholds: Sequence[float]) -> None:
        self.thresholds = tuple(float(value) for value in thresholds)
        self.counts: dict[float, dict[str, np.ndarray]] = {
            threshold: defaultdict(lambda: np.zeros(3, dtype=np.int64))
            for threshold in self.thresholds
        }

    def update(
        self,
        logits: torch.Tensor,
        target: torch.Tensor,
        valid: torch.Tensor,
        patients: Sequence[str],
    ) -> None:
        probability = torch.sigmoid(logits.detach()).cpu().numpy()
        target_numpy = target.detach().cpu().numpy().astype(bool)
        valid_numpy = valid.detach().cpu().numpy().astype(bool)
        for batch_index, patient in enumerate(patients):
            truth = target_numpy[batch_index, 0]
            selected = valid_numpy[batch_index, 0]
            for threshold in self.thresholds:
                prediction = probability[batch_index, 0] >= threshold
                tp = int((prediction & truth & selected).sum())
                fp = int((prediction & ~truth & selected).sum())
                fn = int((~prediction & truth & selected).sum())
                self.counts[threshold][str(patient)] += np.asarray(
                    (tp, fp, fn), dtype=np.int64
                )

    def summarize(self) -> dict[str, Any]:
        candidates: list[dict[str, Any]] = []
        for threshold in self.thresholds:
            patient_metrics: list[dict[str, float | str]] = []
            for patient, (tp, fp, fn) in sorted(
                self.counts[threshold].items()
            ):
                dice_denom = 2 * tp + fp + fn
                iou_denom = tp + fp + fn
                patient_metrics.append(
                    {
                        "patient": patient,
                        "dice": (
                            float(2 * tp / dice_denom)
                            if dice_denom
                            else 1.0
                        ),
                        "iou": (
                            float(tp / iou_denom) if iou_denom else 1.0
                        ),
                    }
                )
            candidates.append(
                {
                    "threshold": threshold,
                    "patient_equal_dice": float(
                        np.mean([row["dice"] for row in patient_metrics])
                    ),
                    "patient_equal_iou": float(
                        np.mean([row["iou"] for row in patient_metrics])
                    ),
                    "n_patients": len(patient_metrics),
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


def create_loader(
    dataset: RegionTileDataset,
    batch_size: int,
    workers: int,
    device: torch.device,
    seed: int,
) -> DataLoader[dict[str, Any]]:
    generator = torch.Generator()
    generator.manual_seed(seed)

    def initialize_worker(worker_id: int) -> None:
        worker_seed = stable_seed(seed, worker_id)
        random.seed(worker_seed)
        np.random.seed(worker_seed)
        torch.manual_seed(worker_seed)

    arguments: dict[str, Any] = {
        "dataset": dataset,
        "batch_size": batch_size,
        "shuffle": False,
        "num_workers": workers,
        "pin_memory": device.type == "cuda",
        "drop_last": False,
        "generator": generator,
        "worker_init_fn": initialize_worker,
    }
    if workers:


        arguments["prefetch_factor"] = 2
    return DataLoader(**arguments)


def autocast_context(device: torch.device, use_bf16: bool) -> Any:
    if device.type == "cuda" and use_bf16:
        return torch.autocast(device_type="cuda", dtype=torch.bfloat16)
    return contextlib.nullcontext()


def train_epoch(
    model: nn.Module,
    loader: DataLoader[dict[str, Any]],
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    device: torch.device,
    loss_weights: Mapping[str, float],
    gradient_accumulation: int,
    gradient_clip_norm: float,
    use_bf16: bool,
) -> dict[str, float]:
    model.train()
    optimizer.zero_grad(set_to_none=True)
    totals: dict[str, float] = defaultdict(float)
    n_batches = 0
    for batch_index, batch in enumerate(loader):
        for key in ("image", "tumor", "valid", "sdf_normalized", "sdf_valid"):
            batch[key] = batch[key].to(device, non_blocking=True)
        with autocast_context(device, use_bf16):
            outputs = model(batch["image"])
            loss, components = masked_region_loss(
                outputs, batch, loss_weights
            )
            group_start = (
                batch_index // gradient_accumulation
            ) * gradient_accumulation
            group_size = min(
                gradient_accumulation, len(loader) - group_start
            )
            scaled_loss = loss / group_size
        scaled_loss.backward()
        should_step = (
            (batch_index + 1) % gradient_accumulation == 0
            or batch_index + 1 == len(loader)
        )
        if should_step:
            torch.nn.utils.clip_grad_norm_(
                model.parameters(), gradient_clip_norm
            )
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            scheduler.step()
        totals["loss"] += float(loss.detach())
        for name, value in components.items():
            totals[name] += float(value)
        n_batches += 1
    return {name: value / max(n_batches, 1) for name, value in totals.items()}


@torch.no_grad()
def validate_tiles(
    model: nn.Module,
    loader: DataLoader[dict[str, Any]],
    device: torch.device,
    thresholds: Sequence[float],
    loss_weights: Mapping[str, float],
    use_bf16: bool,
) -> dict[str, Any]:
    model.eval()
    accumulator = PatientThresholdAccumulator(thresholds)
    total_loss = 0.0
    n_batches = 0
    for batch in loader:
        for key in ("image", "tumor", "valid", "sdf_normalized", "sdf_valid"):
            batch[key] = batch[key].to(device, non_blocking=True)
        with autocast_context(device, use_bf16):
            outputs = model(batch["image"])
            loss, _ = masked_region_loss(outputs, batch, loss_weights)
        accumulator.update(
            outputs["tumor_logits"],
            batch["tumor"],
            batch["valid"],
            list(batch["patient"]),
        )
        total_loss += float(loss)
        n_batches += 1
    summary = accumulator.summarize()
    summary["loss"] = total_loss / max(n_batches, 1)
    summary["metric_scope"] = (
        "fixed patient-equal validation tiles; formal outer metrics use "
        "dense slide coverage"
    )
    return summary


def warmup_cosine_factor(
    step: int, total_steps: int, warmup_steps: int
) -> float:
    if step < warmup_steps:
        return float(step + 1) / max(1, warmup_steps)
    progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
    return 0.5 * (1.0 + math.cos(math.pi * min(max(progress, 0.0), 1.0)))


def checkpoint_payload(
    *,
    phase: str,
    candidate_name: str,
    arm: str,
    candidate: Mapping[str, Any],
    experiment_sha256: str,
    candidates_sha256: str,
    split: Mapping[str, Any],
    epoch: int,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    best: Mapping[str, Any],
    run_contract: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        "schema_version": f"{SCHEMA}.checkpoint.v1",
        "phase": phase,
        "candidate_name": candidate_name,
        "arm": arm,
        "candidate": dict(candidate),
        "experiment_sha256": experiment_sha256,
        "candidates_sha256": candidates_sha256,
        "split_sha256": split["split_sha256"],
        "outer_fold": split["outer_fold"],
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "best": dict(best),
        "run_contract": dict(run_contract),
        "run_contract_sha256": canonical_json_sha256(run_contract),
        "rng_state": capture_rng_state(),
    }


def restore_checkpoint(
    path: Path,
    *,
    model: nn.Module,
    optimizer: torch.optim.Optimizer | None,
    scheduler: torch.optim.lr_scheduler.LRScheduler | None,
    candidate_name: str,
    arm: str,
    experiment_sha256: str,
    candidates_sha256: str,
    split_sha256: str,
    restore_rng: bool,
    expected_run_contract: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    expected = {
        "candidate_name": candidate_name,
        "arm": arm,
        "experiment_sha256": experiment_sha256,
        "candidates_sha256": candidates_sha256,
        "split_sha256": split_sha256,
    }
    for key, value in expected.items():
        if payload.get(key) != value:
            raise ValueError(
                f"checkpoint {key} differs: {payload.get(key)!r} != {value!r}"
            )
    if expected_run_contract is not None:
        expected_run_sha = canonical_json_sha256(expected_run_contract)
        if payload.get("run_contract_sha256") != expected_run_sha:
            raise ValueError(
                "checkpoint runtime contract differs; do not resume with "
                "changed epochs, batches, sampling, or augmentation"
            )
    model.load_state_dict(payload["model_state_dict"], strict=True)
    if optimizer is not None:
        optimizer.load_state_dict(payload["optimizer_state_dict"])
    if scheduler is not None:
        scheduler.load_state_dict(payload["scheduler_state_dict"])
    if restore_rng:
        restore_rng_state(payload["rng_state"])
    return payload


def dense_origins(length: int, tile_size: int, stride: int) -> list[int]:
    if length <= tile_size:
        return [0]
    origins = list(range(0, length - tile_size + 1, stride))
    final = length - tile_size
    if origins[-1] != final:
        origins.append(final)
    return origins


def blend_window(size: int) -> np.ndarray:
    axis = np.hanning(size + 2)[1:-1].astype(np.float32)
    window = np.outer(axis, axis)
    return np.maximum(window, 0.05).astype(np.float32)


def prepare_dense_item(
    record: SampleRecord,
    payload: LabelPayload,
    reader: PyramidWSIReader,
    y0: int,
    x0: int,
    label_size: int,
    input_size: int,
    target_mpp_um: float,
    mean: np.ndarray,
    std: np.ndarray,
) -> torch.Tensor:
    field_um = label_size * LABEL_MPP_UM
    if not math.isclose(
        field_um, input_size * target_mpp_um, rel_tol=0.0, abs_tol=1e-6
    ):
        raise ValueError("dense RGB and label fields differ")
    x0_um = float(payload.metadata["x_min_um"]) + x0 * LABEL_MPP_UM
    y0_um = float(payload.metadata["y_min_um"]) + y0 * LABEL_MPP_UM
    rgb = reader.read_physical(x0_um, y0_um, field_um, input_size)
    image = np.moveaxis(rgb.astype(np.float32) / 255.0, -1, 0)
    image = (image - mean) / std
    return torch.from_numpy(np.ascontiguousarray(image))


def binary_surface(mask: np.ndarray, valid: np.ndarray) -> np.ndarray:
    selected = mask.astype(bool) & valid.astype(bool)
    if not selected.any():
        return np.zeros(selected.shape, dtype=bool)
    eroded = binary_erosion(
        selected, structure=np.ones((3, 3), dtype=bool), border_value=0
    )
    return selected & ~eroded


def region_metrics(
    probability: np.ndarray,
    truth: np.ndarray,
    valid: np.ndarray,
    threshold: float,
) -> dict[str, Any]:
    valid = valid.astype(bool)
    truth = truth.astype(bool) & valid
    prediction = (probability >= threshold) & valid
    tp = int((prediction & truth).sum())
    fp = int((prediction & ~truth & valid).sum())
    fn = int((~prediction & truth & valid).sum())
    dice_denominator = 2 * tp + fp + fn
    iou_denominator = tp + fp + fn
    valid_pixels = int(valid.sum())
    truth_surface = binary_surface(truth, valid)
    prediction_surface = binary_surface(prediction, valid)
    n_truth_surface = int(truth_surface.sum())
    n_prediction_surface = int(prediction_surface.sum())
    if not n_truth_surface and not n_prediction_surface:
        surface = {
            "surface_dice_20um": 1.0,
            "surface_dice_50um": 1.0,
            "assd_um": 0.0,
            "hd95_um": 0.0,
        }
    elif not n_truth_surface or not n_prediction_surface:
        surface = {
            "surface_dice_20um": 0.0,
            "surface_dice_50um": 0.0,
            "assd_um": None,
            "hd95_um": None,
        }
    else:
        distance_to_truth = (
            distance_transform_edt(~truth_surface) * LABEL_MPP_UM
        )
        distance_to_prediction = (
            distance_transform_edt(~prediction_surface) * LABEL_MPP_UM
        )
        pred_distances = distance_to_truth[prediction_surface]
        truth_distances = distance_to_prediction[truth_surface]
        all_distances = np.concatenate((pred_distances, truth_distances))
        surface = {
            "surface_dice_20um": float(
                (
                    (pred_distances <= 20.0).sum()
                    + (truth_distances <= 20.0).sum()
                )
                / (len(pred_distances) + len(truth_distances))
            ),
            "surface_dice_50um": float(
                (
                    (pred_distances <= 50.0).sum()
                    + (truth_distances <= 50.0).sum()
                )
                / (len(pred_distances) + len(truth_distances))
            ),
            "assd_um": float(all_distances.mean()),
            "hd95_um": float(np.percentile(all_distances, 95)),
        }
    metrics: dict[str, Any] = {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "valid_pixels": valid_pixels,
        "tumor_dice": (
            float(2 * tp / dice_denominator) if dice_denominator else 1.0
        ),
        "tumor_iou": (
            float(tp / iou_denominator) if iou_denominator else 1.0
        ),
        "truth_tumor_fraction": (
            float(truth.sum() / valid_pixels) if valid_pixels else None
        ),
        "predicted_tumor_fraction": (
            float(prediction.sum() / valid_pixels) if valid_pixels else None
        ),
        "tumor_fraction_absolute_error": (
            float(abs(int(prediction.sum()) - int(truth.sum())) / valid_pixels)
            if valid_pixels
            else None
        ),
        "tumor_area_bias_mm2": float(
            (int(prediction.sum()) - int(truth.sum()))
            * LABEL_MPP_UM**2
            / 1e6
        ),
        **surface,
    }
    return metrics


@torch.no_grad()
def infer_dense_sample(
    model: nn.Module,
    record: SampleRecord,
    payload: LabelPayload,
    device: torch.device,
    input_size: int,
    label_size: int,
    label_stride: int,
    target_mpp_um: float,
    normalization_mean: Sequence[float],
    normalization_std: Sequence[float],
    batch_size: int,
    use_bf16: bool,
) -> np.ndarray:
    model.eval()
    height, width = payload.valid.shape
    probability_sum = np.zeros((height, width), dtype=np.float32)
    weight_sum = np.zeros((height, width), dtype=np.float32)
    weight = blend_window(label_size)
    mean = np.asarray(normalization_mean, dtype=np.float32)[:, None, None]
    std = np.asarray(normalization_std, dtype=np.float32)[:, None, None]
    pending_images: list[torch.Tensor] = []
    pending_origins: list[tuple[int, int]] = []
    reader = PyramidWSIReader(record, target_mpp_um)

    def flush() -> None:
        if not pending_images:
            return
        images = torch.stack(pending_images).to(device, non_blocking=True)
        with autocast_context(device, use_bf16):
            logits = model(images)["tumor_logits"]
        probabilities = torch.sigmoid(logits.float()).cpu().numpy()[:, 0]
        for probability, (y0, x0) in zip(probabilities, pending_origins):
            crop_height = min(label_size, height - y0)
            crop_width = min(label_size, width - x0)
            probability_sum[
                y0 : y0 + crop_height, x0 : x0 + crop_width
            ] += probability[:crop_height, :crop_width] * weight[
                :crop_height, :crop_width
            ]
            weight_sum[
                y0 : y0 + crop_height, x0 : x0 + crop_width
            ] += weight[:crop_height, :crop_width]
        pending_images.clear()
        pending_origins.clear()

    try:
        for y0 in dense_origins(height, label_size, label_stride):
            for x0 in dense_origins(width, label_size, label_stride):
                valid_crop = padded_crop(
                    payload.valid, y0, x0, label_size, False
                )
                if not valid_crop.any():
                    continue
                pending_images.append(
                    prepare_dense_item(
                        record,
                        payload,
                        reader,
                        y0,
                        x0,
                        label_size,
                        input_size,
                        target_mpp_um,
                        mean,
                        std,
                    )
                )
                pending_origins.append((y0, x0))
                if len(pending_images) >= batch_size:
                    flush()
        flush()
    finally:
        reader.close()
    output = np.zeros((height, width), dtype=np.float32)
    covered = weight_sum > 0
    output[covered] = probability_sum[covered] / weight_sum[covered]
    if (payload.valid & ~covered).any():
        raise RuntimeError(f"{record.sample}: dense inference missed valid pixels")
    return output


def build_prediction_metadata(
    *,
    record: SampleRecord,
    payload: LabelPayload,
    outer_fold: int,
    decision_threshold: float,
    candidate: str,
    arm: str,
    checkpoint_path: Path,
    checkpoint_sha256: str,
    source_manifest_path: Path,
    source_manifest_sha256: str,
) -> dict[str, Any]:

    if record.fold != outer_fold:
        raise ValueError("prediction record is not owned by the outer fold")
    if not 0.0 <= decision_threshold <= 1.0:
        raise ValueError("decision threshold must be in [0,1]")
    geometry = {
        "axes": str(payload.metadata["axes"]),
        "shape_yx": [int(value) for value in payload.tumor.shape],
        "resolution_um": float(payload.metadata["resolution_um"]),
        "x_min_um": float(payload.metadata["x_min_um"]),
        "y_min_um": float(payload.metadata["y_min_um"]),
    }
    if geometry["axes"] != "YX":
        raise ValueError("prediction labels must use YX axes")
    return {
        "schema_version": f"{SCHEMA}.prediction.v1",
        "prediction_role": "outer_fold_oof",
        "sample": record.sample,
        "patient": record.patient,
        "cancer": record.cancer,
        "fold": record.fold,
        "outer_fold": outer_fold,
        **geometry,
        "decision_threshold": float(decision_threshold),
        "candidate": candidate,
        "arm": arm,
        "source_manifest_sha256": source_manifest_sha256,
        "source_manifest_path": str(source_manifest_path.resolve()),
        "label": {
            "path": str(record.label_path.resolve()),
            "sha256": sha256_file(record.label_path),
        },
        "producer": {
            "name": "scripts/train_region_model.py",
            "candidate": candidate,
            "arm": arm,
            "checkpoint_path": str(checkpoint_path.resolve()),
            "checkpoint_sha256": checkpoint_sha256,
        },
    }


def summarize_outer_metrics(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    patients: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        patients[str(row["patient"])].append(row)
    patient_rows: list[dict[str, Any]] = []
    for patient, samples in sorted(patients.items()):
        patient_rows.append(
            {
                "patient": patient,
                "cancer": str(samples[0]["cancer"]),
                "n_samples": len(samples),
                "tumor_dice": float(
                    np.mean([float(row["tumor_dice"]) for row in samples])
                ),
                "tumor_iou": float(
                    np.mean([float(row["tumor_iou"]) for row in samples])
                ),
            }
        )
    by_cancer: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in patient_rows:
        by_cancer[str(row["cancer"])].append(row)
    pooled_tp = sum(int(row["tp"]) for row in rows)
    pooled_fp = sum(int(row["fp"]) for row in rows)
    pooled_fn = sum(int(row["fn"]) for row in rows)
    pooled_dice_denominator = 2 * pooled_tp + pooled_fp + pooled_fn
    pooled_iou_denominator = pooled_tp + pooled_fp + pooled_fn
    return {
        "n_samples": len(rows),
        "n_patients": len(patient_rows),
        "patient_equal_tumor_dice": float(
            np.mean([row["tumor_dice"] for row in patient_rows])
        ),
        "patient_equal_tumor_iou": float(
            np.mean([row["tumor_iou"] for row in patient_rows])
        ),
        "sample_equal_tumor_dice": float(
            np.mean([float(row["tumor_dice"]) for row in rows])
        ),
        "sample_equal_tumor_iou": float(
            np.mean([float(row["tumor_iou"]) for row in rows])
        ),
        "patient_metrics": patient_rows,
        "by_cancer": {
            cancer: {
                "n_patients": len(cancer_patients),
                "patient_equal_tumor_dice": float(
                    np.mean(
                        [float(row["tumor_dice"]) for row in cancer_patients]
                    )
                ),
                "patient_equal_tumor_iou": float(
                    np.mean(
                        [float(row["tumor_iou"]) for row in cancer_patients]
                    )
                ),
            }
            for cancer, cancer_patients in sorted(by_cancer.items())
        },
        "pooled_pixel_diagnostic": {
            "tumor_dice": (
                float(2 * pooled_tp / pooled_dice_denominator)
                if pooled_dice_denominator
                else 1.0
            ),
            "tumor_iou": (
                float(pooled_tp / pooled_iou_denominator)
                if pooled_iou_denominator
                else 1.0
            ),
            "formal_metric": False,
        },
    }


def make_tile_dataset(
    records: Sequence[SampleRecord],
    *,
    experiment: Mapping[str, Any],
    candidates_document: Mapping[str, Any],
    tiles_per_patient: int,
    seed: int,
    training: bool,
    stain_augmentation: bool,
) -> RegionTileDataset:
    preprocessing = experiment["preprocessing"]
    sampling = experiment["sampling"]
    runtime = candidates_document["training_runtime"]
    input_contract = candidates_document["input_contract"]
    return RegionTileDataset(
        records=records,
        tiles_per_patient=tiles_per_patient,
        label_size=int(preprocessing["label_shape_yx"][0]),
        input_size=int(preprocessing["model_input_shape_yx"][0]),
        target_mpp_um=float(preprocessing["model_input_mpp_um"]),
        boundary_um=float(sampling["boundary_candidate_um"]),
        stratum_names=list(sampling["tile_strata"]),
        stratum_weights=list(sampling["tile_stratum_weights"]),
        minimum_valid_fraction=float(
            runtime["minimum_valid_tissue_fraction"]
        ),
        sdf_clip_um=float(experiment["model"]["sdf_clip_um"]),
        normalization_mean=input_contract["normalization"]["mean"],
        normalization_std=input_contract["normalization"]["std"],
        seed=seed,
        training=training,
        stain_augmentation=stain_augmentation,
        cache_size=int(runtime["label_cache_size"]),
    )


def resolve_device(requested: str) -> torch.device:
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(requested)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    return device


def profile_model(
    model: nn.Module,
    candidate_name: str,
    candidate: Mapping[str, Any],
    device: torch.device,
    input_size: int,
    batch_size: int,
) -> dict[str, Any]:
    model = model.to(device)
    model.eval()
    activation_bytes = 0
    handles: list[Any] = []

    def hook(_module: nn.Module, _inputs: Any, output: Any) -> None:
        nonlocal activation_bytes
        tensors: list[torch.Tensor] = []
        if torch.is_tensor(output):
            tensors = [output]
        elif isinstance(output, Mapping):
            tensors = [value for value in output.values() if torch.is_tensor(value)]
        elif isinstance(output, (tuple, list)):
            tensors = [value for value in output if torch.is_tensor(value)]
        activation_bytes += sum(
            tensor.numel() * tensor.element_size() for tensor in tensors
        )

    for module in model.modules():
        if not any(module.children()):
            handles.append(module.register_forward_hook(hook))
    rss_before = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    input_tensor = torch.zeros(
        (batch_size, 3, input_size, input_size),
        dtype=torch.float32,
        device=device,
    )
    start = time.perf_counter()
    with torch.inference_mode(), autocast_context(device, device.type == "cuda"):
        output = model(input_tensor)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elapsed = time.perf_counter() - start
    rss_after = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    for handle in handles:
        handle.remove()
    summary = summarize_model(model, str(candidate["architecture"]))
    result = {
        "schema_version": f"{SCHEMA}.profile.v1",
        "candidate_name": candidate_name,
        "candidate": dict(candidate),
        "device": str(device),
        "batch_size": batch_size,
        "input_shape": [batch_size, 3, input_size, input_size],
        "output_shapes": {
            name: list(tensor.shape) for name, tensor in output.items()
        },
        "trainable_parameters": summary.trainable_parameters,
        "parameter_bytes_fp32": summary.trainable_parameters * 4,
        "leaf_output_activation_bytes": activation_bytes,
        "forward_seconds": elapsed,
        "rss_peak_delta_kib": max(0, rss_after - rss_before),
        "activation_note": (
            "sum of leaf-module output tensor bytes for relative comparison; "
            "not a peak training-memory claim"
        ),
    }
    if device.type == "cuda":
        result["cuda_peak_allocated_bytes"] = torch.cuda.max_memory_allocated(
            device
        )
        result["cuda_peak_reserved_bytes"] = torch.cuda.max_memory_reserved(
            device
        )
    return result


def parse_args() -> argparse.Namespace:
    project_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=Path,
        default=project_root / "configs" / "experiment.json",
    )
    parser.add_argument(
        "--candidates-config",
        type=Path,
        default=project_root / "configs" / "model_candidates.json",
    )
    parser.add_argument(
        "--mode",
        required=True,
        choices=("profile", "tune", "refit", "evaluate"),
    )
    parser.add_argument("--candidate", required=True)
    parser.add_argument("--outer-fold", type=int, default=0)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--gradient-accumulation", type=int)
    parser.add_argument("--workers", type=int)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--stain-augmentation", action="store_true")
    parser.add_argument("--save-predictions", action="store_true")
    parser.add_argument("--profile-input-size", type=int)
    parser.add_argument("--profile-output", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config_path = args.config.resolve()
    candidates_path = args.candidates_config.resolve()
    project_root = config_path.parent.parent
    experiment = json.loads(config_path.read_text(encoding="utf-8"))
    candidates_document = json.loads(
        candidates_path.read_text(encoding="utf-8")
    )
    if args.candidate not in candidates_document["candidates"]:
        raise ValueError(f"unknown candidate: {args.candidate}")
    candidate = candidates_document["candidates"][args.candidate]
    experiment_sha256 = canonical_json_sha256(experiment)
    candidates_sha256 = canonical_json_sha256(candidates_document)
    seed = int(experiment["training"]["seed"])


    replicate = int(experiment["training"].get("replicate", 0))
    run_seed = seed if replicate == 0 else stable_seed(seed, "replicate", replicate)
    seed_everything(run_seed)
    device = resolve_device(args.device)


    builder = build_fm_model if is_foundation_candidate(candidate) else build_model
    model = builder(candidate).to(device)
    arm = training_arm(args.stain_augmentation)

    if args.mode == "profile":
        input_size = (
            args.profile_input_size
            if args.profile_input_size is not None
            else int(experiment["preprocessing"]["model_input_shape_yx"][0])
        )
        result = profile_model(
            model,
            args.candidate,
            candidate,
            device,
            input_size,
            args.batch_size or 1,
        )
        result["experiment_sha256"] = experiment_sha256
        result["candidates_sha256"] = candidates_sha256
        result["arm"] = arm
        print(json.dumps(result, indent=2, sort_keys=True))
        if args.profile_output is not None:
            output_path = args.profile_output.resolve()
            try:
                output_path.relative_to(project_root)
            except ValueError as error:
                raise ValueError(
                    "profile output must stay inside the release workspace"
                ) from error
            atomic_json(output_path, result)
        return

    manifest_path = project_root / experiment["outputs"]["source_manifest"]
    records = load_records(manifest_path)
    runtime = candidates_document["training_runtime"]
    split = make_nested_split(
        records,
        args.outer_fold,
        seed,
        float(runtime["inner_validation_fraction"]),
    )
    split_path = (
        project_root
        / experiment["outputs"]["manifests"]
        / "splits"
        / f"fold{args.outer_fold}_nested.json"
    )
    atomic_json(split_path, split)
    paths = artifact_paths(
        project_root,
        experiment,
        args.candidate,
        arm,
        args.outer_fold,
    )
    model_root = paths["model_root"]
    tune_root = model_root / "tune"
    refit_root = model_root / "refit"
    training_config = experiment["training"]
    batch_size = args.batch_size or int(training_config["batch_size_per_gpu"])
    accumulation = args.gradient_accumulation or int(
        training_config["gradient_accumulation_steps"]
    )
    workers = (
        args.workers
        if args.workers is not None
        else int(runtime["loader_workers"])
    )
    use_bf16 = (
        training_config["amp_dtype"] == "bfloat16" and device.type == "cuda"
    )

    if args.mode in {"tune", "refit"}:
        if args.mode == "tune":
            train_records = select_records(
                records, split["inner_training_patients"]
            )
            validation_records = select_records(
                records, split["inner_validation_patients"]
            )
            epochs = args.epochs or int(training_config["epochs"])
            phase_root = tune_root
        else:
            train_records = select_records(
                records, split["outer_training_patients"]
            )
            validation_records = []
            tune_summary_path = tune_root / "summary.json"
            if args.epochs is None:
                if not tune_summary_path.is_file():
                    raise FileNotFoundError(
                        "refit requires tune/summary.json or explicit --epochs"
                    )
                tune_summary = json.loads(
                    tune_summary_path.read_text(encoding="utf-8")
                )
                if tune_summary.get("arm") != arm:
                    raise ValueError(
                        "tune summary belongs to a different training arm"
                    )
                if tune_summary.get("split_sha256") != split["split_sha256"]:
                    raise ValueError("tune split receipt differs")
                epochs = int(tune_summary["best_epoch"])
            else:
                epochs = args.epochs
            phase_root = refit_root
        run_contract = {
            "phase": args.mode,
            "candidate": args.candidate,
            "arm": arm,
            "outer_fold": args.outer_fold,
            "epochs": epochs,
            "batch_size_per_gpu": batch_size,
            "gradient_accumulation": accumulation,
            "tiles_per_patient": int(runtime["tiles_per_patient_train"]),
            "stain_augmentation": args.stain_augmentation,


            "replicate": replicate,
            "training_samples": sorted(
                record.sample for record in train_records
            ),
            "validation_samples": sorted(
                record.sample for record in validation_records
            ),
            "optimizer": "adamw",
            "learning_rate": float(training_config["learning_rate"]),
            "weight_decay": float(training_config["weight_decay"]),
            "warmup_fraction": float(training_config["warmup_fraction"]),
            "scheduler": "cosine",
        }
        for record in train_records + validation_records:
            if not record.label_path.is_file():
                raise FileNotFoundError(
                    f"{record.sample}: dense label is absent; full label export "
                    "must pass before model training"
                )
        train_dataset = make_tile_dataset(
            train_records,
            experiment=experiment,
            candidates_document=candidates_document,
            tiles_per_patient=int(runtime["tiles_per_patient_train"]),
            seed=stable_seed(run_seed, args.outer_fold, args.mode, "train"),
            training=True,
            stain_augmentation=args.stain_augmentation,
        )
        train_loader = create_loader(
            train_dataset,
            batch_size,
            workers,
            device,
            stable_seed(run_seed, args.outer_fold, args.mode, "train_loader"),
        )
        validation_dataset: RegionTileDataset | None = None
        validation_loader: DataLoader[dict[str, Any]] | None = None
        if args.mode == "tune":
            validation_dataset = make_tile_dataset(
                validation_records,
                experiment=experiment,
                candidates_document=candidates_document,
                tiles_per_patient=int(
                    runtime["tiles_per_patient_validation"]
                ),
                seed=stable_seed(run_seed, args.outer_fold, "validation"),
                training=False,
                stain_augmentation=False,
            )
            validation_loader = create_loader(
                validation_dataset,
                batch_size,
                workers,
                device,
                stable_seed(run_seed, args.outer_fold, "validation_loader"),
            )


        base_lr = float(training_config["learning_rate"])
        encoder_lr_scale = float(candidate.get("encoder_lr_scale", 1.0))
        if encoder_lr_scale != 1.0:
            encoder_parameters, decoder_parameters = [], []
            for name, parameter in model.named_parameters():
                if not parameter.requires_grad:
                    continue
                (encoder_parameters if name.startswith("encoder.") else decoder_parameters).append(
                    parameter
                )
            if not encoder_parameters:
                raise ValueError("encoder_lr_scale set but no encoder.* parameters found")
            groups = [
                {"params": encoder_parameters, "lr": base_lr * encoder_lr_scale},
                {"params": decoder_parameters, "lr": base_lr},
            ]
        else:
            groups = [{"params": list(model.parameters()), "lr": base_lr}]
        optimizer = torch.optim.AdamW(
            groups,
            lr=base_lr,
            weight_decay=float(training_config["weight_decay"]),
        )
        optimizer_steps_per_epoch = math.ceil(
            len(train_loader) / accumulation
        )
        total_steps = max(1, epochs * optimizer_steps_per_epoch)
        warmup_steps = max(
            1, int(round(total_steps * float(training_config["warmup_fraction"])))
        )
        scheduler = torch.optim.lr_scheduler.LambdaLR(
            optimizer,
            lambda step: warmup_cosine_factor(
                step, total_steps, warmup_steps
            ),
        )
        latest_path = phase_root / "latest.pt"
        start_epoch = 0
        best: dict[str, Any] = {
            "score": -float("inf"),
            "epoch": 0,
            "threshold": 0.5,
        }
        if args.resume:
            if not latest_path.is_file():
                raise FileNotFoundError(latest_path)
            restored = restore_checkpoint(
                latest_path,
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                candidate_name=args.candidate,
                arm=arm,
                experiment_sha256=experiment_sha256,
                candidates_sha256=candidates_sha256,
                split_sha256=split["split_sha256"],
                restore_rng=True,
                expected_run_contract=run_contract,
            )
            start_epoch = int(restored["epoch"])
            best = dict(restored["best"])
        history: list[dict[str, Any]] = []
        history_path = phase_root / "history.json"
        if start_epoch and history_path.is_file():
            history = json.loads(history_path.read_text(encoding="utf-8"))[
                "epochs"
            ]
        for epoch_index in range(start_epoch, epochs):
            train_dataset.set_epoch(epoch_index)
            train_metrics = train_epoch(
                model,
                train_loader,
                optimizer,
                scheduler,
                device,
                training_config["loss"],
                accumulation,
                float(training_config["gradient_clip_norm"]),
                use_bf16,
            )
            row: dict[str, Any] = {
                "epoch": epoch_index + 1,
                "train": train_metrics,
                "learning_rate": optimizer.param_groups[0]["lr"],
            }
            if validation_loader is not None and validation_dataset is not None:


                validation_dataset.set_epoch(0)
                validation = validate_tiles(
                    model,
                    validation_loader,
                    device,
                    runtime["threshold_candidates"],
                    training_config["loss"],
                    use_bf16,
                )
                row["validation"] = validation
                selected = validation["selected"]
                score = float(selected["patient_equal_dice"])
                is_better = (
                    score > float(best["score"]) + 1e-12
                    or (
                        abs(score - float(best["score"])) <= 1e-12
                        and epoch_index + 1 < int(best["epoch"])
                    )
                )
                if is_better:
                    best = {
                        "score": score,
                        "epoch": epoch_index + 1,
                        "threshold": float(selected["threshold"]),
                    }
                    atomic_torch_save(
                        phase_root / "best.pt",
                        checkpoint_payload(
                            phase=args.mode,
                            candidate_name=args.candidate,
                            arm=arm,
                            candidate=candidate,
                            experiment_sha256=experiment_sha256,
                            candidates_sha256=candidates_sha256,
                            split=split,
                            epoch=epoch_index + 1,
                            model=model,
                            optimizer=optimizer,
                            scheduler=scheduler,
                            best=best,
                            run_contract=run_contract,
                        ),
                    )
            else:
                best = {
                    "score": None,
                    "epoch": epoch_index + 1,
                    "threshold": None,
                }
            history.append(row)
            checkpoint = checkpoint_payload(
                phase=args.mode,
                candidate_name=args.candidate,
                arm=arm,
                candidate=candidate,
                experiment_sha256=experiment_sha256,
                candidates_sha256=candidates_sha256,
                split=split,
                epoch=epoch_index + 1,
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                best=best,
                run_contract=run_contract,
            )
            atomic_torch_save(latest_path, checkpoint)
            atomic_json(
                history_path,
                {
                    "schema_version": f"{SCHEMA}.history.v1",
                    "candidate": args.candidate,
                    "arm": arm,
                    "outer_fold": args.outer_fold,
                    "phase": args.mode,
                    "epochs": history,
                },
            )
            print(
                f"{args.mode} fold={args.outer_fold} "
                f"epoch={epoch_index + 1}/{epochs} "
                f"loss={train_metrics['loss']:.6f}",
                flush=True,
            )
        if args.mode == "refit":


            final_payload = torch.load(
                latest_path, map_location="cpu", weights_only=False
            )
            atomic_torch_save(refit_root / "final.pt", final_payload)
        summary = {
            "schema_version": f"{SCHEMA}.summary.v1",
            "candidate": args.candidate,
            "arm": arm,
            "candidate_definition": candidate,
            "outer_fold": args.outer_fold,
            "phase": args.mode,
            "epochs_completed": epochs,
            "best_epoch": (
                int(best["epoch"]) if args.mode == "tune" else epochs
            ),
            "selected_threshold": (
                float(best["threshold"])
                if args.mode == "tune"
                else None
            ),
            "selection_metric": (
                "inner_validation_patient_equal_tile_dice"
                if args.mode == "tune"
                else "fixed_epoch_from_tune_or_explicit_authorized_override"
            ),
            "inner_validation_score": (
                float(best["score"]) if args.mode == "tune" else None
            ),
            "stain_augmentation": args.stain_augmentation,
            "n_training_patients": len(train_dataset.patients),
            "n_validation_patients": (
                len(validation_dataset.patients)
                if validation_dataset is not None
                else 0
            ),
            "experiment_sha256": experiment_sha256,
            "candidates_sha256": candidates_sha256,
            "split_sha256": split["split_sha256"],
            "device": str(device),
            "bf16": use_bf16,
            "run_contract_sha256": canonical_json_sha256(run_contract),
        }
        atomic_json(phase_root / "summary.json", summary)
        atomic_json(
            phase_root / "SUCCESS.json",
            {
                "schema_version": f"{SCHEMA}.success.v1",
                "status": "COMPLETED",
                "phase": args.mode,
                "candidate": args.candidate,
                "arm": arm,
                "outer_fold": args.outer_fold,
                "epochs_completed": epochs,
                "summary_path": str((phase_root / "summary.json").resolve()),
                "run_contract_sha256": canonical_json_sha256(run_contract),
            },
        )
        return

    if args.mode == "evaluate":
        tune_summary_path = tune_root / "summary.json"
        final_path = refit_root / "final.pt"
        if not tune_summary_path.is_file() or not final_path.is_file():
            raise FileNotFoundError(
                "outer evaluation requires tune/summary.json and refit/final.pt"
            )
        tune_summary = json.loads(
            tune_summary_path.read_text(encoding="utf-8")
        )
        if tune_summary.get("arm") != arm:
            raise ValueError("tune summary belongs to a different training arm")
        if tune_summary["split_sha256"] != split["split_sha256"]:
            raise ValueError("tune split receipt differs")
        threshold = float(tune_summary["selected_threshold"])
        restore_checkpoint(
            final_path,
            model=model,
            optimizer=None,
            scheduler=None,
            candidate_name=args.candidate,
            arm=arm,
            experiment_sha256=experiment_sha256,
            candidates_sha256=candidates_sha256,
            split_sha256=split["split_sha256"],
            restore_rng=False,
            expected_run_contract=None,
        )
        evaluation_records = select_records(
            records, split["outer_test_patients"]
        )
        for record in evaluation_records:
            if not record.label_path.is_file():
                raise FileNotFoundError(record.label_path)
        preprocessing = experiment["preprocessing"]
        input_contract = candidates_document["input_contract"]
        label_size = int(preprocessing["label_shape_yx"][0])
        input_size = int(preprocessing["model_input_shape_yx"][0])
        prediction_root = paths["prediction_root"]
        source_manifest_sha256 = sha256_file(manifest_path)
        checkpoint_sha256 = sha256_file(final_path)
        rows: list[dict[str, Any]] = []
        for record in evaluation_records:
            payload = load_label(
                record, float(experiment["sampling"]["boundary_candidate_um"])
            )
            probability = infer_dense_sample(
                model,
                record,
                payload,
                device,
                input_size=input_size,
                label_size=label_size,
                label_stride=label_size // 2,
                target_mpp_um=float(preprocessing["model_input_mpp_um"]),
                normalization_mean=input_contract["normalization"]["mean"],
                normalization_std=input_contract["normalization"]["std"],
                batch_size=batch_size,
                use_bf16=use_bf16,
            )
            metrics = region_metrics(
                probability,
                payload.tumor,
                payload.valid,
                threshold,
            )
            row = {
                "sample": record.sample,
                "patient": record.patient,
                "cancer": record.cancer,
                "fold": record.fold,
                **metrics,
            }
            rows.append(row)
            if args.save_predictions:
                output_path = (
                    prediction_root / record.cancer / f"{record.sample}.npz"
                )
                output_path.parent.mkdir(parents=True, exist_ok=True)
                prediction_metadata = build_prediction_metadata(
                    record=record,
                    payload=payload,
                    outer_fold=args.outer_fold,
                    decision_threshold=threshold,
                    candidate=args.candidate,
                    arm=arm,
                    checkpoint_path=final_path,
                    checkpoint_sha256=checkpoint_sha256,
                    source_manifest_path=manifest_path,
                    source_manifest_sha256=source_manifest_sha256,
                )
                with tempfile.NamedTemporaryFile(
                    dir=output_path.parent, suffix=".npz", delete=False
                ) as handle:
                    temporary = Path(handle.name)
                try:
                    np.savez_compressed(
                        temporary,
                        tumor_probability=probability.astype(np.float16),
                        label_valid=payload.valid.astype(np.uint8),
                        metadata_json=np.asarray(
                            json.dumps(
                                prediction_metadata,
                                sort_keys=True,
                                separators=(",", ":"),
                            )
                        ),
                    )
                    os.replace(temporary, output_path)
                finally:
                    temporary.unlink(missing_ok=True)
            print(
                f"evaluate {record.sample}: dice={metrics['tumor_dice']:.4f}",
                flush=True,
            )
        aggregate = summarize_outer_metrics(rows)
        result = {
            "schema_version": f"{SCHEMA}.outer_evaluation.v1",
            "candidate": args.candidate,
            "arm": arm,
            "outer_fold": args.outer_fold,
            "threshold_from_inner_validation": threshold,
            "checkpoint": str(final_path),
            "split_sha256": split["split_sha256"],
            "claim_boundary": experiment["task"]["claim_boundary"],
            "metric_resolution_um": LABEL_MPP_UM,
            "dense_inference": {
                "label_tile_size": label_size,
                "label_stride": label_size // 2,
                "blend": "nonzero_hann",
                "coverage": "every_valid_label_pixel",
            },
            "aggregate": aggregate,
            "samples": rows,
        }
        result_path = paths["result_path"]
        atomic_json(result_path, result)
        print(json.dumps(aggregate, indent=2, sort_keys=True))
        return

    raise AssertionError(f"unhandled mode: {args.mode}")


if __name__ == "__main__":
    main()
