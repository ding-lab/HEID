#!/usr/bin/env python3

from __future__ import annotations

import asyncio
import hashlib
import json
import math
import os
from collections import OrderedDict, defaultdict
from dataclasses import dataclass
from importlib import metadata as importlib_metadata
from pathlib import Path
from typing import Any, Iterable, Iterator, Mapping, Sequence

import cv2
import numpy as np
import pandas as pd
import tifffile
import torch
from torch.utils.data import Dataset, Sampler


REQUIRED_INDEX_COLUMNS = {
    "tile_id",
    "kind",
    "sample",
    "patient",
    "cancer",
    "fold",
    "stratum",
    "he_path",
    "he_sha256",
    "he_size_y_px",
    "he_size_x_px",
    "he_mpp_um",
    "label_path",
    "label_sha256",
    "label_size_y_px",
    "label_size_x_px",
    "label_mpp_um",
    "label_origin_x_um",
    "label_origin_y_um",
    "label_y0_px",
    "label_x0_px",
    "label_valid_h_px",
    "label_valid_w_px",
    "label_tile_h_px",
    "label_tile_w_px",
    "input_h_px",
    "input_w_px",
    "input_mpp_um",
    "fov_h_um",
    "fov_w_um",
}


def sha256_file(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def _major_version(distribution: str) -> int:
    value = importlib_metadata.version(distribution)
    return int(value.split(".", maxsplit=1)[0])


def zarr_major_version() -> int:

    return _major_version("zarr")


def validate_patient_fold_contract(table: pd.DataFrame) -> None:
    required = {"sample", "patient", "fold"}
    missing = required.difference(table.columns)
    if missing:
        raise ValueError(f"missing split columns: {sorted(missing)}")
    if table[list(required)].isna().any().any():
        raise ValueError("sample/patient/fold values must be non-null")
    sample_fold_counts = table.groupby("sample", sort=False)["fold"].nunique()
    patient_fold_counts = table.groupby("patient", sort=False)["fold"].nunique()
    if int(sample_fold_counts.max()) != 1:
        bad = sample_fold_counts[sample_fold_counts.ne(1)].index.tolist()
        raise ValueError(f"samples span folds: {bad[:10]}")
    if int(patient_fold_counts.max()) != 1:
        bad = patient_fold_counts[patient_fold_counts.ne(1)].index.tolist()
        raise ValueError(f"patients span folds: {bad[:10]}")


def load_index(path: Path) -> pd.DataFrame:
    if path.suffix.lower() == ".parquet":
        table = pd.read_parquet(path)
    elif path.suffix.lower() in {".csv", ".tsv"}:
        table = pd.read_csv(path, sep="\t" if path.suffix.lower() == ".tsv" else ",")
    else:
        raise ValueError(f"unsupported tile-index format: {path}")
    missing = REQUIRED_INDEX_COLUMNS.difference(table.columns)
    if missing:
        raise ValueError(f"tile index is missing columns: {sorted(missing)}")
    if table["tile_id"].duplicated().any():
        raise ValueError("tile IDs are not unique")
    validate_patient_fold_contract(table)
    return table


def _pyramid_mpp(
    level_shape_yxs: Sequence[int],
    level0_shape_yxs: Sequence[int],
    level0_mpp_um: float,
) -> tuple[float, float]:
    if len(level_shape_yxs) != 3 or len(level0_shape_yxs) != 3:
        raise ValueError("pyramid arrays must have YXS shape")
    level_mpp_y = (
        float(level0_mpp_um)
        * int(level0_shape_yxs[0])
        / int(level_shape_yxs[0])
    )
    level_mpp_x = (
        float(level0_mpp_um)
        * int(level0_shape_yxs[1])
        / int(level_shape_yxs[1])
    )
    return level_mpp_y, level_mpp_x


def choose_pyramid_level(
    level_shapes_yxs: Sequence[Sequence[int]],
    level0_mpp_um: float,
    target_mpp_um: float,
) -> tuple[int, float, float]:

    if target_mpp_um <= 0 or level0_mpp_um <= 0:
        raise ValueError("pixel sizes must be positive")
    level0_shape = tuple(level_shapes_yxs[0])
    candidates: list[tuple[int, float, float]] = []
    for level, shape in enumerate(level_shapes_yxs):
        mpp_y, mpp_x = _pyramid_mpp(shape, level0_shape, level0_mpp_um)
        if max(mpp_y, mpp_x) <= target_mpp_um * (1.0 + 1e-6):
            candidates.append((level, mpp_y, mpp_x))
    if not candidates:
        raise ValueError(
            f"target {target_mpp_um} um/px is finer than level 0 "
            f"({level0_mpp_um} um/px)"
        )
    return candidates[-1]


class OmePyramidReader:

    def __init__(
        self,
        path: Path,
        expected_l0_shape_yx: tuple[int, int],
        level0_mpp_um: float,
        target_mpp_um: float,
    ) -> None:
        import zarr

        self._async_loop: asyncio.AbstractEventLoop | None = None
        self.path = Path(path)
        self._tiff = tifffile.TiffFile(str(self.path))
        series = self._tiff.series[0]
        if series.axes != "YXS":
            self.close()
            raise ValueError(f"{self.path}: expected YXS axes, observed {series.axes}")
        if np.dtype(series.dtype) != np.dtype(np.uint8):
            self.close()
            raise ValueError(f"{self.path}: expected uint8, observed {series.dtype}")
        if tuple(series.shape[:2]) != tuple(expected_l0_shape_yx):
            self.close()
            raise ValueError(
                f"{self.path}: L0 shape {series.shape[:2]} differs from "
                f"contract {expected_l0_shape_yx}"
            )
        level_shapes = [tuple(level.shape) for level in series.levels]
        level, mpp_y, mpp_x = choose_pyramid_level(
            level_shapes, level0_mpp_um, target_mpp_um
        )
        self.level = level
        self.level_mpp_y_um = mpp_y
        self.level_mpp_x_um = mpp_x
        self._store = self._tiff.aszarr(series=0, level=level)
        asynchronous_api = getattr(
            getattr(zarr, "api", None), "asynchronous", None
        )
        if zarr_major_version() >= 3 and asynchronous_api is not None:


            self._async_loop = asyncio.new_event_loop()
            opened = self._async_loop.run_until_complete(
                asynchronous_api.open_array(store=self._store, mode="r")
            )
        else:
            opened = zarr.open(self._store, mode="r")
        if not hasattr(opened, "shape"):
            if str(level) not in opened:
                self.close()
                raise ValueError(f"{self.path}: Zarr group lacks level {level}")
            opened = opened[str(level)]
        self.array = opened
        expected_level_shape = level_shapes[level]
        if tuple(self.array.shape) != expected_level_shape:
            self.close()
            raise ValueError(
                f"{self.path}: lazy level shape differs from TIFF metadata"
            )

    def close(self) -> None:
        store = getattr(self, "_store", None)
        if store is not None and hasattr(store, "close"):
            store.close()
        tiff = getattr(self, "_tiff", None)
        if tiff is not None:
            tiff.close()
        async_loop = getattr(self, "_async_loop", None)
        if async_loop is not None:
            async_loop.close()
        self.array = None
        self._store = None
        self._tiff = None
        self._async_loop = None

    def __enter__(self) -> "OmePyramidReader":
        return self

    def __exit__(self, *_: Any) -> None:
        self.close()

    def read_physical_window(
        self,
        x0_um: float,
        y0_um: float,
        output_shape_yx: tuple[int, int],
        output_mpp_um: float,
    ) -> np.ndarray:

        if self.array is None:
            raise RuntimeError("reader is closed")
        out_h, out_w = map(int, output_shape_yx)
        if out_h <= 0 or out_w <= 0 or output_mpp_um <= 0:
            raise ValueError("output shape and mpp must be positive")
        target_x_um = x0_um + (np.arange(out_w, dtype=np.float64) + 0.5) * output_mpp_um
        target_y_um = y0_um + (np.arange(out_h, dtype=np.float64) + 0.5) * output_mpp_um
        source_x = target_x_um / self.level_mpp_x_um - 0.5
        source_y = target_y_um / self.level_mpp_y_um - 0.5


        radius = 4
        read_x0 = max(0, int(math.floor(float(source_x.min()))) - radius)
        read_y0 = max(0, int(math.floor(float(source_y.min()))) - radius)
        read_x1 = min(
            int(self.array.shape[1]),
            int(math.ceil(float(source_x.max()))) + radius + 1,
        )
        read_y1 = min(
            int(self.array.shape[0]),
            int(math.ceil(float(source_y.max()))) + radius + 1,
        )
        if read_x1 <= read_x0 or read_y1 <= read_y0:
            return np.full((out_h, out_w, 3), 255, dtype=np.uint8)

        selection = (
            slice(read_y0, read_y1),
            slice(read_x0, read_x1),
            slice(None),
        )
        if self._async_loop is not None:
            block = np.asarray(
                self._async_loop.run_until_complete(
                    self.array.getitem(selection)
                ),
                dtype=np.uint8,
            )
        else:
            block = np.asarray(self.array[selection], dtype=np.uint8)
        map_x, map_y = np.meshgrid(
            (source_x - read_x0).astype(np.float32),
            (source_y - read_y0).astype(np.float32),
        )
        output = cv2.remap(
            block,
            map_x,
            map_y,
            interpolation=cv2.INTER_LANCZOS4,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=(255, 255, 255),
        )
        if output.shape != (out_h, out_w, 3):
            raise RuntimeError(f"unexpected remap shape: {output.shape}")
        return output


@dataclass(frozen=True)
class LabelArchive:
    tumor_mask: np.ndarray
    label_valid: np.ndarray
    sdf_um: np.ndarray
    metadata: Mapping[str, Any]


def _same_path(left: str | Path, right: str | Path) -> bool:
    def key(value: str | Path) -> str:
        normalized = os.path.abspath(os.path.normpath(str(value)))
        alias_from = os.environ.get("PATH_ALIAS_FROM", "")
        alias_to = os.environ.get("PATH_ALIAS_TO", "")
        if alias_from and (normalized == alias_from or normalized.startswith(alias_from + "/")):
            normalized = alias_to + normalized[len(alias_from):]
        return normalized

    return key(left) == key(right)


def load_verified_label(row: Mapping[str, Any]) -> LabelArchive:
    path = Path(str(row["label_path"]))
    observed_sha = sha256_file(path)
    if observed_sha != str(row["label_sha256"]):
        raise ValueError(f"{path}: label SHA-256 differs from tile index")
    with np.load(path, allow_pickle=False) as archive:
        required = {"tumor_mask", "label_valid", "sdf_um", "metadata_json"}
        missing = required.difference(archive.files)
        if missing:
            raise ValueError(f"{path}: missing label arrays {sorted(missing)}")
        tumor_mask = np.asarray(archive["tumor_mask"], dtype=np.uint8)
        label_valid = np.asarray(archive["label_valid"], dtype=np.uint8)
        sdf_um = np.asarray(archive["sdf_um"], dtype=np.float32)
        metadata = json.loads(str(archive["metadata_json"].item()))
    expected_shape = (
        int(row["label_size_y_px"]),
        int(row["label_size_x_px"]),
    )
    if (
        tumor_mask.shape != expected_shape
        or label_valid.shape != expected_shape
        or sdf_um.shape != expected_shape
    ):
        raise ValueError(f"{path}: label array shape differs from tile index")
    if tuple(metadata.get("shape_yx", ())) != expected_shape:
        raise ValueError(f"{path}: embedded metadata shape differs")
    scalar_checks = {
        "sample": str(row["sample"]),
        "patient": str(row["patient"]),
        "cancer": str(row["cancer"]),
        "fold": int(row["fold"]),
    }
    for field, expected in scalar_checks.items():
        observed = metadata.get(field)
        if observed != expected:
            raise ValueError(
                f"{path}: embedded {field}={observed!r}, expected {expected!r}"
            )
    if not math.isclose(
        float(metadata["resolution_um"]),
        float(row["label_mpp_um"]),
        rel_tol=0.0,
        abs_tol=1e-9,
    ):
        raise ValueError(f"{path}: label resolution differs from tile index")
    if not math.isclose(
        float(metadata["x_min_um"]),
        float(row["label_origin_x_um"]),
        rel_tol=0.0,
        abs_tol=1e-6,
    ) or not math.isclose(
        float(metadata["y_min_um"]),
        float(row["label_origin_y_um"]),
        rel_tol=0.0,
        abs_tol=1e-6,
    ):
        raise ValueError(f"{path}: label origin differs from tile index")
    he = metadata.get("he", {})
    if not _same_path(he.get("path", ""), row["he_path"]):
        raise ValueError(f"{path}: embedded H&E path differs from tile index")
    if str(he.get("sha256", "")) != str(row["he_sha256"]):
        raise ValueError(f"{path}: embedded H&E SHA-256 differs from tile index")
    if tuple(he.get("shape_yx", ())) != (
        int(row["he_size_y_px"]),
        int(row["he_size_x_px"]),
    ):
        raise ValueError(f"{path}: embedded H&E shape differs from tile index")
    return LabelArchive(tumor_mask, label_valid, sdf_um, metadata)


class _LabelLru:
    def __init__(self, max_entries: int) -> None:
        if max_entries <= 0:
            raise ValueError("label cache must retain at least one slide")
        self.max_entries = int(max_entries)
        self.cache: OrderedDict[str, LabelArchive] = OrderedDict()

    def get(self, row: Mapping[str, Any]) -> LabelArchive:
        key = str(row["label_path"])
        value = self.cache.pop(key, None)
        if value is None:
            value = load_verified_label(row)
        self.cache[key] = value
        while len(self.cache) > self.max_entries:
            self.cache.popitem(last=False)
        return value


def extract_padded(
    array: np.ndarray,
    y0: int,
    x0: int,
    height: int,
    width: int,
    fill_value: float | int,
) -> np.ndarray:
    if y0 < 0 or x0 < 0:
        raise ValueError("label windows use non-negative array indices")
    output = np.full((height, width), fill_value, dtype=array.dtype)
    valid_h = max(0, min(height, array.shape[0] - y0))
    valid_w = max(0, min(width, array.shape[1] - x0))
    if valid_h and valid_w:
        output[:valid_h, :valid_w] = array[y0 : y0 + valid_h, x0 : x0 + valid_w]
    return output


def apply_d4(array: np.ndarray, transform_id: int) -> np.ndarray:

    transform_id = int(transform_id)
    if not 0 <= transform_id < 8:
        raise ValueError("D4 transform ID must be in [0, 7]")
    if array.ndim < 2:
        raise ValueError("D4 input needs at least two dimensions")
    output = np.rot90(array, k=transform_id % 4, axes=(-2, -1))
    if transform_id >= 4:
        output = np.flip(output, axis=-1)
    return np.ascontiguousarray(output)


def deterministic_d4(seed: int, epoch: int, tile_id: str) -> int:
    payload = f"{int(seed)}|{int(epoch)}|{tile_id}".encode("utf-8")
    value = int.from_bytes(hashlib.blake2b(payload, digest_size=8).digest(), "little")
    return value % 8


def blend_weight(shape_yx: tuple[int, int], floor: float = 1e-3) -> np.ndarray:

    height, width = shape_yx
    if height <= 0 or width <= 0 or not 0 < floor <= 1:
        raise ValueError("invalid blend-weight parameters")
    wy = np.hanning(height + 2)[1:-1] if height > 1 else np.ones(1)
    wx = np.hanning(width + 2)[1:-1] if width > 1 else np.ones(1)
    weight = np.outer(wy, wx)
    weight /= max(float(weight.max()), np.finfo(np.float64).eps)
    return np.maximum(weight, floor).astype(np.float32)


class HeMaskDataset(Dataset[dict[str, Any]]):

    def __init__(
        self,
        index: Path | pd.DataFrame,
        *,
        kind: str,
        outer_fold: int | None = None,
        role: str = "all",
        patients: Iterable[str] | None = None,
        geometric_augmentation: bool = False,
        seed: int = 24073,
        label_cache_entries: int = 2,
        slide_cache_entries: int = 1,
        image_mean: Sequence[float] = (0.5, 0.5, 0.5),
        image_std: Sequence[float] = (0.5, 0.5, 0.5),
        sdf_clip_um: float = 500.0,
    ) -> None:
        table = load_index(index) if isinstance(index, Path) else index.copy()
        missing = REQUIRED_INDEX_COLUMNS.difference(table.columns)
        if missing:
            raise ValueError(f"tile index is missing columns: {sorted(missing)}")
        validate_patient_fold_contract(table)
        table = table.loc[table["kind"].eq(kind)].copy()
        if role not in {"all", "train", "test"}:
            raise ValueError("role must be all, train, or test")
        if role != "all":
            if outer_fold is None:
                raise ValueError("outer_fold is required for train/test roles")
            in_fold = table["fold"].astype(int).eq(int(outer_fold))
            table = table.loc[~in_fold if role == "train" else in_fold].copy()
        if patients is not None:
            selected = {str(value) for value in patients}
            table = table.loc[table["patient"].astype(str).isin(selected)].copy()
        if table.empty:
            raise ValueError("dataset selection is empty")
        if role == "train" and kind != "train_candidate":
            raise ValueError("training role requires train_candidate tiles")
        if role == "test" and kind != "eval":
            raise ValueError("test role requires eval tiles")
        validate_patient_fold_contract(table)
        self.table = table.reset_index(drop=True)
        self.kind = kind
        self.geometric_augmentation = bool(geometric_augmentation)
        self.seed = int(seed)
        self.epoch = 0
        self._label_cache_entries = int(label_cache_entries)
        self._slide_cache_entries = int(slide_cache_entries)
        if self._slide_cache_entries <= 0:
            raise ValueError("slide cache must retain at least one slide")
        self.image_mean = np.asarray(image_mean, dtype=np.float32).reshape(3, 1, 1)
        self.image_std = np.asarray(image_std, dtype=np.float32).reshape(3, 1, 1)
        if not np.isfinite(self.image_mean).all() or not (
            np.isfinite(self.image_std).all() & (self.image_std > 0)
        ).all():
            raise ValueError("image mean/std must be finite and std positive")
        self.sdf_clip_um = float(sdf_clip_um)
        if not math.isfinite(self.sdf_clip_um) or self.sdf_clip_um <= 0:
            raise ValueError("SDF clip must be finite and positive")
        self._label_cache = _LabelLru(self._label_cache_entries)
        self._slide_cache: OrderedDict[str, OmePyramidReader] = OrderedDict()

    def __len__(self) -> int:
        return len(self.table)

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def close(self) -> None:
        for reader in self._slide_cache.values():
            reader.close()
        self._slide_cache.clear()

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            pass

    def __getstate__(self) -> dict[str, Any]:
        state = dict(self.__dict__)
        state["_label_cache"] = _LabelLru(self._label_cache_entries)
        state["_slide_cache"] = OrderedDict()
        return state

    def _reader(self, row: Mapping[str, Any]) -> OmePyramidReader:
        key = str(row["he_path"])
        reader = self._slide_cache.pop(key, None)
        if reader is None:
            reader = OmePyramidReader(
                Path(key),
                (
                    int(row["he_size_y_px"]),
                    int(row["he_size_x_px"]),
                ),
                float(row["he_mpp_um"]),
                float(row["input_mpp_um"]),
            )
        self._slide_cache[key] = reader
        while len(self._slide_cache) > self._slide_cache_entries:
            _, evicted = self._slide_cache.popitem(last=False)
            evicted.close()
        return reader

    def __getitem__(self, index: int) -> dict[str, Any]:
        row = self.table.iloc[int(index)]
        label = self._label_cache.get(row)
        label_y0 = int(row["label_y0_px"])
        label_x0 = int(row["label_x0_px"])
        label_h = int(row["label_tile_h_px"])
        label_w = int(row["label_tile_w_px"])
        tumor = extract_padded(
            label.tumor_mask, label_y0, label_x0, label_h, label_w, 0
        )
        valid = extract_padded(
            label.label_valid, label_y0, label_x0, label_h, label_w, 0
        ).astype(bool)
        sdf = extract_padded(
            label.sdf_um, label_y0, label_x0, label_h, label_w, np.nan
        )
        sdf_valid = valid & np.isfinite(sdf)
        sdf = np.where(sdf_valid, sdf, 0.0).astype(np.float32)

        x0_um = float(row["label_origin_x_um"]) + label_x0 * float(
            row["label_mpp_um"]
        )
        y0_um = float(row["label_origin_y_um"]) + label_y0 * float(
            row["label_mpp_um"]
        )
        expected_x0 = float(row["physical_x0_um"])
        expected_y0 = float(row["physical_y0_um"])
        if not math.isclose(x0_um, expected_x0, abs_tol=1e-6) or not math.isclose(
            y0_um, expected_y0, abs_tol=1e-6
        ):
            raise ValueError(f"{row['tile_id']}: physical origin contract mismatch")
        input_shape = (int(row["input_h_px"]), int(row["input_w_px"]))
        input_mpp = float(row["input_mpp_um"])
        if not math.isclose(
            input_shape[0] * input_mpp,
            float(row["fov_h_um"]),
            abs_tol=1e-6,
        ) or not math.isclose(
            input_shape[1] * input_mpp,
            float(row["fov_w_um"]),
            abs_tol=1e-6,
        ):
            raise ValueError(f"{row['tile_id']}: input FOV contract mismatch")
        rgb = self._reader(row).read_physical_window(
            x0_um=x0_um,
            y0_um=y0_um,
            output_shape_yx=input_shape,
            output_mpp_um=input_mpp,
        )

        image_chw = np.moveaxis(rgb, -1, 0)
        transform_id = 0
        if self.geometric_augmentation:
            if input_shape[0] != input_shape[1] or label_h != label_w:
                raise ValueError("D4 augmentation requires square image and label tiles")
            transform_id = deterministic_d4(self.seed, self.epoch, str(row["tile_id"]))
            image_chw = apply_d4(image_chw, transform_id)
            tumor = apply_d4(tumor, transform_id)
            valid = apply_d4(valid, transform_id)
            sdf = apply_d4(sdf, transform_id)
            sdf_valid = apply_d4(sdf_valid, transform_id)

        valid_h = int(row["label_valid_h_px"])
        valid_w = int(row["label_valid_w_px"])
        stitch = blend_weight((label_h, label_w))

        if valid_h < label_h:
            stitch[valid_h:, :] = 0.0
        if valid_w < label_w:
            stitch[:, valid_w:] = 0.0
        image_rgb_01 = image_chw.astype(np.float32) / 255.0
        image_normalized = (
            image_rgb_01 - self.image_mean
        ) / self.image_std
        tumor_tensor = torch.from_numpy(tumor[None].astype(np.float32))
        valid_tensor = torch.from_numpy(valid[None])
        sdf_tensor = torch.from_numpy(sdf[None])
        sdf_valid_tensor = torch.from_numpy(sdf_valid[None])
        sdf_normalized = torch.from_numpy(
            (
                np.clip(sdf, -self.sdf_clip_um, self.sdf_clip_um)
                / self.sdf_clip_um
            )[None].astype(np.float32)
        )
        return {
            "image": torch.from_numpy(np.ascontiguousarray(image_normalized)),
            "image_rgb_01": torch.from_numpy(np.ascontiguousarray(image_rgb_01)),

            "tumor_mask": tumor_tensor,
            "label_valid": valid_tensor,
            "sdf_um": sdf_tensor,

            "tumor": tumor_tensor,
            "valid": valid_tensor,
            "sdf_normalized": sdf_normalized,
            "sdf_valid": sdf_valid_tensor,
            "stitch_weight": torch.from_numpy(stitch[None]),
            "tile_id": str(row["tile_id"]),
            "sample": str(row["sample"]),
            "patient": str(row["patient"]),
            "cancer": str(row["cancer"]),
            "fold": int(row["fold"]),
            "stratum": str(row["stratum"]),
            "d4_transform": transform_id,
            "label_window_yxhw": torch.tensor(
                [label_y0, label_x0, valid_h, valid_w], dtype=torch.int64
            ),
            "label_canvas_shape_yx": torch.tensor(
                [int(row["label_size_y_px"]), int(row["label_size_x_px"])],
                dtype=torch.int64,
            ),
            "physical_window_xywh_um": torch.tensor(
                [
                    x0_um,
                    y0_um,
                    float(row["fov_w_um"]),
                    float(row["fov_h_um"]),
                ],
                dtype=torch.float64,
            ),
        }


class PatientEqualStratifiedSampler(Sampler[int]):

    def __init__(
        self,
        dataset: HeMaskDataset,
        *,
        epoch_size: int | None = None,
        stratum_weights: Mapping[str, float] | None = None,
        patient_block_size: int = 4,
        seed: int = 24073,
    ) -> None:
        if dataset.kind != "train_candidate":
            raise ValueError("patient-equal sampling requires train candidates")
        self.dataset = dataset
        self.epoch_size = int(epoch_size or len(dataset))
        if self.epoch_size <= 0 or patient_block_size <= 0:
            raise ValueError("epoch size and patient block size must be positive")
        self.patient_block_size = int(patient_block_size)
        self.seed = int(seed)
        self.epoch = 0
        self.stratum_weights = dict(
            stratum_weights
            or {
                "boundary": 0.4,
                "tumor_interior": 0.3,
                "normal_tissue": 0.3,
            }
        )
        if any(float(value) < 0 for value in self.stratum_weights.values()):
            raise ValueError("stratum weights must be non-negative")
        if sum(float(value) for value in self.stratum_weights.values()) <= 0:
            raise ValueError("at least one stratum weight must be positive")

        groups: dict[str, dict[str, dict[str, list[int]]]] = defaultdict(
            lambda: defaultdict(lambda: defaultdict(list))
        )
        for position, row in dataset.table.iterrows():
            groups[str(row["patient"])][str(row["stratum"])][
                str(row["sample"])
            ].append(int(position))
        self.groups = {
            patient: {
                stratum: dict(samples) for stratum, samples in strata.items()
            }
            for patient, strata in groups.items()
        }
        self.patients = sorted(self.groups)
        if not self.patients:
            raise ValueError("no patients available to sample")

    def __len__(self) -> int:
        return self.epoch_size

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)
        self.dataset.set_epoch(epoch)

    def __iter__(self) -> Iterator[int]:
        rng = np.random.default_rng(self.seed + self.epoch)
        n_patients = len(self.patients)
        base, remainder = divmod(self.epoch_size, n_patients)
        patient_order = list(rng.permutation(self.patients))
        remaining = {
            patient: base + int(patient in set(patient_order[:remainder]))
            for patient in self.patients
        }
        schedule: list[str] = []
        while any(value > 0 for value in remaining.values()):
            active = [patient for patient, value in remaining.items() if value > 0]
            for patient in rng.permutation(active):
                count = min(self.patient_block_size, remaining[str(patient)])
                schedule.extend([str(patient)] * count)
                remaining[str(patient)] -= count

        for patient in schedule:
            available = sorted(self.groups[patient])
            weights = np.asarray(
                [float(self.stratum_weights.get(name, 0.0)) for name in available],
                dtype=np.float64,
            )
            if not weights.any():
                weights[:] = 1.0
            weights /= weights.sum()
            stratum = str(rng.choice(available, p=weights))
            sample_groups = self.groups[patient][stratum]
            sample = str(rng.choice(sorted(sample_groups)))
            indices = sample_groups[sample]
            yield int(indices[int(rng.integers(0, len(indices)))])


def add_stitched_prediction(
    value_sum: np.ndarray,
    weight_sum: np.ndarray,
    prediction: np.ndarray,
    weight: np.ndarray,
    label_window_yxhw: Sequence[int],
) -> None:

    y0, x0, valid_h, valid_w = map(int, label_window_yxhw)
    if value_sum.shape != weight_sum.shape:
        raise ValueError("reconstruction accumulators must have equal shape")
    if prediction.shape != weight.shape:
        raise ValueError("prediction and weight shapes must match")
    if prediction.ndim != 2:
        raise ValueError("prediction tiles must be 2D")
    if valid_h < 0 or valid_w < 0:
        raise ValueError("valid tile extents must be non-negative")
    y1 = min(y0 + valid_h, value_sum.shape[0])
    x1 = min(x0 + valid_w, value_sum.shape[1])
    h = max(0, y1 - y0)
    w = max(0, x1 - x0)
    if h == 0 or w == 0:
        return
    local_weight = weight[:h, :w]
    value_sum[y0:y1, x0:x1] += prediction[:h, :w] * local_weight
    weight_sum[y0:y1, x0:x1] += local_weight


def finalize_stitched_prediction(
    value_sum: np.ndarray, weight_sum: np.ndarray
) -> np.ndarray:
    output = np.full(value_sum.shape, np.nan, dtype=np.float32)
    covered = weight_sum > 0
    output[covered] = (value_sum[covered] / weight_sum[covered]).astype(np.float32)
    return output
