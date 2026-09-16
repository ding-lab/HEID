#!/usr/bin/env python3

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Any

import numpy as np


DEFAULT_PREPROCESSING_ID = "luster_raw_gray_v1"
DEFAULT_RGB_PREPROCESSING_ID = "rgb_raw_v1"
FORMAL_GRAYSCALE_PREPROCESSING_IDS = (
    "luster_raw_gray_v1",
    "luster_slide_grayod_scale_v1",
    "intensity_raw_gray_v1",
    "bt601_raw_gray_v1",
)
FORMAL_RGB_PREPROCESSING_IDS = (DEFAULT_RGB_PREPROCESSING_ID,)
FORMAL_PREPROCESSING_IDS = (
    *FORMAL_GRAYSCALE_PREPROCESSING_IDS,
    *FORMAL_RGB_PREPROCESSING_IDS,
)

MEMBERSHIP_FILTER_ID = "clean177_luster_blank_v1"

_COMMON_SPEC: dict[str, Any] = {
    "schema_version": "a1.grayscale_preprocess.v1",
    "input": {
        "dtype": "uint8",
        "layout": "HWC",
        "channels": "RGB",
        "crop_size_px": 224,
        "padding_value": 255,
    },
    "blank_filter": {
        "membership_filter_id": MEMBERSHIP_FILTER_ID,
        "algorithm": "Luster raw plane shared by every gray and RGB arm",
        "stage": "before arm-specific preprocessing",
        "pixel_threshold_strictly_greater_than": 240,
        "drop_if_fraction_strictly_greater_than": 0.9,
    },
    "canonical_output": {
        "channels": 1,
        "layout": "HW",
        "dtype": "uint8",
        "colour_channels_forbidden": True,
    },
    "augmentation": {
        "allowed_spatial_only": ["horizontal_flip", "vertical_flip"],
        "forbidden": ["color_jitter", "channel_dropout", "channel_shuffle", "stain_jitter"],
    },
}


def _variant(
    preprocessing_id: str,
    algorithm: str,
    formula: str,
    *,
    percentile: bool,
    formal: bool,
    slide_grayod: bool = False,
) -> dict[str, Any]:
    spec = json.loads(json.dumps(_COMMON_SPEC))
    spec["preprocessing_id"] = preprocessing_id
    spec["representation_arm"] = "gray"
    spec["formal_launch_allowed"] = formal
    spec["evidence_status"] = (
        "formal comparison candidate; requires locked-split comparison before adoption"
        if formal
        else "rejected per-crop percentile diagnostic; formal launch forbidden"
    )
    spec["grayscale"] = {"algorithm": algorithm, "formula": formula}
    if algorithm == "BT.601 fixed-point luminance":
        spec["grayscale"].update({
            "integer_weights": [77, 150, 29],
            "denominator": 256,
            "rounding_offset": 128,
        })
    spec["intensity"] = {
        "enabled": percentile,
        "scope": "per crop over all pixels, including white padding",
        "lower_percentile": 1.0 if percentile else 0.0,
        "upper_percentile": 99.0 if percentile else 100.0,
        "percentile_method": "numpy linear",
        "minimum_range": 1.0,
        "output_range": [0, 255],
        "rounding": "numpy rint then uint8",
        "degenerate_range_fallback": "retain raw grayscale",
    }
    spec["slide_grayod"] = {
        "enabled": slide_grayod,
        "formula": "D=-ln((G+1)/256); D'=clip(D*s_ref/s_slide,0,D_cap); G'=rint(256*exp(-D')-1)",
        "statistic": "median gray optical density over deterministic tissue pixels",
        "reference": "equally slide-weighted median over applicable outer-training slides only",
        "target_statistic": "this slide only, unlabeled, never pooled across target slides",
        "fallback": "fatal",
    }
    return spec


def _rgb_variant(preprocessing_id: str) -> dict[str, Any]:
    spec = json.loads(json.dumps(_COMMON_SPEC))
    spec.update({
        "schema_version": "a1.rgb_preprocess.v1",
        "preprocessing_id": preprocessing_id,
        "representation_arm": "rgb",
        "formal_launch_allowed": True,
        "evidence_status": "formal independent RGB comparison arm; not an adopted winner",
        "canonical_output": {
            "channels": 3,
            "layout": "HWC",
            "dtype": "uint8",
            "colour_channels_forbidden": False,
            "channel_order": "RGB",
        },
        "rgb": {
            "algorithm": "identity",
            "formula": "uint8 RGB crop unchanged after common membership filter",
        },
        "intensity": {
            "enabled": False,
            "scope": "none",
            "output_range": [0, 255],
        },
        "slide_grayod": {"enabled": False, "fallback": "not_applicable"},
    })
    return spec


PREPROCESSING_SPECS: dict[str, dict[str, Any]] = {
    "luster_raw_gray_v1": _variant(
        "luster_raw_gray_v1",
        "Luster",
        "rint((min(R,G,B) + max(R,G,B)) / 2)",
        percentile=False,
        formal=True,
    ),
    "luster_slide_grayod_scale_v1": _variant(
        "luster_slide_grayod_scale_v1",
        "Luster",
        "rint((min(R,G,B) + max(R,G,B)) / 2)",
        percentile=False,
        formal=True,
        slide_grayod=True,
    ),
    "intensity_raw_gray_v1": _variant(
        "intensity_raw_gray_v1",
        "arithmetic RGB intensity",
        "rint((R + G + B) / 3)",
        percentile=False,
        formal=True,
    ),
    "bt601_raw_gray_v1": _variant(
        "bt601_raw_gray_v1",
        "BT.601 fixed-point luminance",
        "(77*R + 150*G + 29*B + 128) >> 8",
        percentile=False,
        formal=True,
    ),
    "rgb_raw_v1": _rgb_variant("rgb_raw_v1"),
    "bt601_p1p99_gray_v1": _variant(
        "bt601_p1p99_gray_v1",
        "BT.601 fixed-point luminance",
        "(77*R + 150*G + 29*B + 128) >> 8",
        percentile=True,
        formal=False,
    ),
    "intensity_p1p99_gray_v1": _variant(
        "intensity_p1p99_gray_v1",
        "arithmetic RGB intensity",
        "round((R + G + B) / 3)",
        percentile=True,
        formal=False,
    ),
    "lightness_p1p99_gray_v1": _variant(
        "lightness_p1p99_gray_v1",
        "Luster",
        "round((min(R,G,B) + max(R,G,B)) / 2)",
        percentile=True,
        formal=False,
    ),
}

PREPROCESS_SPEC = PREPROCESSING_SPECS[DEFAULT_PREPROCESSING_ID]

DEFAULT_BACKBONE_INTERFACE_ID = "folded_patch_projection_imagenet_v1"
DEFAULT_RGB_BACKBONE_INTERFACE_ID = "native_rgb_imagenet_v1"
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]
BACKBONE_INTERFACE_SPECS: dict[str, dict[str, Any]] = {
    "folded_patch_projection_imagenet_v1": {
        "schema_version": "a1.backbone_grayscale_interface.v1",
        "backbone_interface_id": "folded_patch_projection_imagenet_v1",
        "representation_arm": "gray",
        "canonical_input_channels": 1,
        "input": "one grayscale channel scaled to [0,1]",
        "patch_projection": "fold replicated-gray ImageNet affine into pretrained 3-channel patch Conv",
        "weight_formula": "W_gray=sum_c(W_c/std_c)",
        "bias_formula": "b_gray=b-sum_c,kernel(W_c*mean_c/std_c)",
        "imagenet_mean": IMAGENET_MEAN,
        "imagenet_std": IMAGENET_STD,
        "claim": "algebraically equivalent to replicated grayscale plus per-channel ImageNet affine for zero-padded-free patch Conv",
        "evidence_status": "preferred pilot interface because it preserves the pretrained first-layer computation exactly",
        "formal_launch_allowed": True,
    },
    "native_rgb_imagenet_v1": {
        "schema_version": "a1.backbone_rgb_interface.v1",
        "backbone_interface_id": "native_rgb_imagenet_v1",
        "representation_arm": "rgb",
        "canonical_input_channels": 3,
        "input": "native uint8 RGB scaled to [0,1], then channel-specific ImageNet affine",
        "imagenet_mean": IMAGENET_MEAN,
        "imagenet_std": IMAGENET_STD,
        "evidence_status": "formal independent RGB comparison interface",
        "formal_launch_allowed": True,
    },
    "replicate_imagenet_affine_v1": {
        "schema_version": "a1.backbone_grayscale_interface.v1",
        "backbone_interface_id": "replicate_imagenet_affine_v1",
        "representation_arm": "gray",
        "canonical_input_channels": 1,
        "input": "three byte-identical grayscale channels scaled to [0,1], then channel-specific affine",
        "imagenet_mean": IMAGENET_MEAN,
        "imagenet_std": IMAGENET_STD,
        "note": "post-affine channel values differ deterministically but contain no independent colour information",
        "evidence_status": "numerical QC oracle only; formal launch forbidden",
        "formal_launch_allowed": False,
    },
}

FORMAL_INTERFACE_MATRIX: dict[str, tuple[str, ...]] = {
    **{
        preprocessing_id: (DEFAULT_BACKBONE_INTERFACE_ID,)
        for preprocessing_id in FORMAL_GRAYSCALE_PREPROCESSING_IDS
    },
    DEFAULT_RGB_PREPROCESSING_ID: (DEFAULT_RGB_BACKBONE_INTERFACE_ID,),
}
FORMAL_BACKBONE_INTERFACE_IDS = tuple(sorted({
    interface_id
    for interface_ids in FORMAL_INTERFACE_MATRIX.values()
    for interface_id in interface_ids
}))


def _canonical_json(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")


def get_preprocess_spec(preprocessing_id: str = DEFAULT_PREPROCESSING_ID) -> dict[str, Any]:
    try:
        return PREPROCESSING_SPECS[preprocessing_id]
    except KeyError as exc:
        raise ValueError(
            f"unsupported preprocessing_id {preprocessing_id!r}; "
            f"choose one of {sorted(PREPROCESSING_SPECS)}"
        ) from exc


def preprocess_spec_sha256(preprocessing_id: str = DEFAULT_PREPROCESSING_ID) -> str:
    return hashlib.sha256(_canonical_json(get_preprocess_spec(preprocessing_id))).hexdigest()


def representation_arm(preprocessing_id: str) -> str:
    arm = get_preprocess_spec(preprocessing_id).get("representation_arm")
    if arm not in {"gray", "rgb"}:
        raise ValueError(f"preprocessing {preprocessing_id!r} has invalid representation arm")
    return str(arm)


PREPROCESS_SPEC_SHA256 = preprocess_spec_sha256()

SLIDE_GRAYOD_PARAMS_SCHEMA = "a1.slide_grayod_frozen_params.v1"


def slide_grayod_params_sha256(params: dict[str, Any]) -> str:
    return hashlib.sha256(_canonical_json(params)).hexdigest()


def validate_slide_grayod_params(
    params: dict[str, Any] | None,
    expected_sha256: str | None,
    *,
    expected_sample: str | None = None,
    expected_fold: int | None = None,
    expected_role: str | None = None,
    expected_split_id: str | None = None,
) -> dict[str, Any]:
    if not isinstance(params, dict) or not expected_sha256:
        raise ValueError("slide gray-OD requires explicit frozen params and params SHA256")
    observed_sha = slide_grayod_params_sha256(params)
    if observed_sha != expected_sha256:
        raise ValueError(
            f"slide gray-OD params SHA256 mismatch: expected {expected_sha256}, observed {observed_sha}"
        )
    required_scalar = ("s_slide", "s_ref", "epsilon", "d_cap")
    if params.get("schema_version") != SLIDE_GRAYOD_PARAMS_SCHEMA:
        raise ValueError("unsupported slide gray-OD params schema")
    if params.get("preprocessing_id") != "luster_slide_grayod_scale_v1":
        raise ValueError("slide gray-OD params preprocessing_id mismatch")
    for key in required_scalar:
        value = float(params.get(key, float("nan")))
        if not np.isfinite(value) or value <= 0:
            raise ValueError(f"invalid positive slide gray-OD scalar {key}={value}")
    tissue = params.get("tissue_mask")
    if not isinstance(tissue, dict):
        raise ValueError("slide gray-OD tissue_mask object missing")
    threshold = float(tissue.get("gray_od_strict_gt", float("nan")))
    minimum = int(tissue.get("min_tissue_pixels", -1))
    observed = int(tissue.get("n_tissue_pixels", -1))
    if not np.isfinite(threshold) or threshold < 0 or minimum < 1 or observed < minimum:
        raise ValueError("slide gray-OD tissue mask/count QC failed")
    if tissue.get("sampling") != "deterministic_slide_wide_single_plane":
        raise ValueError("slide gray-OD requires deterministic slide-wide single-plane sampling")
    statistic = params.get("slide_statistic")
    if not isinstance(statistic, dict) or statistic.get("name") != "median_gray_od":
        raise ValueError("slide gray-OD statistic must be median_gray_od")
    if statistic.get("uses_labels") is not False or statistic.get("pooled_across_slides") is not False:
        raise ValueError("slide gray-OD target statistic must be unlabeled and this-slide-only")
    reference = params.get("reference")
    if not isinstance(reference, dict):
        raise ValueError("slide gray-OD reference provenance missing")
    if (
        reference.get("statistic") != "median_s_slide"
        or reference.get("weighting") != "equal_slide"
        or reference.get("scope") != "outer_train_only"
        or int(reference.get("n_slides", 0)) < 1
        or not reference.get("slide_ids_sha256")
    ):
        raise ValueError("slide gray-OD reference is not an equal-slide outer-train reference")
    if params.get("fallback") != "fatal":
        raise ValueError("slide gray-OD silent fallback is forbidden")
    exact = {
        "sample": expected_sample,
        "adapter_fold": expected_fold,
        "role": expected_role,
    }
    for key, expected in exact.items():
        if expected is not None and params.get(key) != expected:
            raise ValueError(f"slide gray-OD {key} mismatch: {params.get(key)!r} != {expected!r}")
    if expected_split_id is not None and reference.get("split_id") != expected_split_id:
        raise ValueError("slide gray-OD reference split_id mismatch")
    if expected_fold is not None and int(reference.get("outer_fold", -1)) != expected_fold:
        raise ValueError("slide gray-OD reference outer_fold mismatch")
    return params


@dataclass(frozen=True)
class FrozenSlideGrayODParams:
    params: dict[str, Any]
    sha256: str


def freeze_slide_grayod_params(
    params: dict[str, Any] | None,
    expected_sha256: str | None,
    **expected_identity: Any,
) -> FrozenSlideGrayODParams:
    validated = validate_slide_grayod_params(
        params, expected_sha256, **expected_identity
    )
    canonical_copy = json.loads(_canonical_json(validated).decode("utf-8"))
    return FrozenSlideGrayODParams(canonical_copy, str(expected_sha256))


def gray_optical_density(gray: np.ndarray) -> np.ndarray:
    plane = np.asarray(gray)
    if plane.dtype != np.uint8:
        raise TypeError("gray optical density requires uint8 input")
    return -np.log((plane.astype(np.float32) + 1.0) / 256.0)


def estimate_slide_grayod_statistic(
    deterministic_gray_sample: np.ndarray,
    *,
    tissue_od_strict_gt: float = 0.12,
    min_tissue_pixels: int = 10_000,
) -> tuple[float, int]:
    od = gray_optical_density(np.asarray(deterministic_gray_sample, dtype=np.uint8))
    tissue = od[od > float(tissue_od_strict_gt)]
    if len(tissue) < int(min_tissue_pixels):
        raise ValueError(
            f"slide gray-OD tissue count {len(tissue)} is below {min_tissue_pixels}"
        )
    return float(np.median(tissue)), int(len(tissue))


def equal_slide_grayod_reference(slide_statistics: list[float]) -> float:
    values = np.asarray(slide_statistics, dtype=np.float64)
    if values.ndim != 1 or not len(values) or not np.all(np.isfinite(values)) or np.any(values <= 0):
        raise ValueError("outer-train slide statistics must be finite positive values")
    return float(np.median(values))


def apply_slide_grayod_scale(gray: np.ndarray, params: dict[str, Any]) -> np.ndarray:
    od = gray_optical_density(gray)
    scale = float(params["s_ref"]) / max(float(params["s_slide"]), float(params["epsilon"]))
    mapped_od = np.clip(od * scale, 0.0, float(params["d_cap"]))
    mapped = np.rint(256.0 * np.exp(-mapped_od) - 1.0)
    return np.clip(mapped, 0.0, 255.0).astype(np.uint8)


def get_backbone_interface_spec(
    interface_id: str = DEFAULT_BACKBONE_INTERFACE_ID,
) -> dict[str, Any]:
    try:
        return BACKBONE_INTERFACE_SPECS[interface_id]
    except KeyError as exc:
        raise ValueError(
            f"unsupported backbone_interface_id {interface_id!r}; "
            f"choose one of {sorted(BACKBONE_INTERFACE_SPECS)}"
        ) from exc


def backbone_interface_sha256(
    interface_id: str = DEFAULT_BACKBONE_INTERFACE_ID,
) -> str:
    return hashlib.sha256(_canonical_json(get_backbone_interface_spec(interface_id))).hexdigest()


def validate_formal_preprocessing_interface(
    preprocessing_id: str,
    interface_id: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    preprocess = get_preprocess_spec(preprocessing_id)
    interface = get_backbone_interface_spec(interface_id)
    if preprocess.get("formal_launch_allowed") is not True:
        raise ValueError(f"non-formal preprocessing_id {preprocessing_id!r}")
    if interface.get("formal_launch_allowed") is not True:
        raise ValueError(f"non-formal backbone interface {interface_id!r}")
    allowed = FORMAL_INTERFACE_MATRIX.get(preprocessing_id, ())
    if interface_id not in allowed:
        raise ValueError(
            f"illegal formal preprocessing/interface pair: {preprocessing_id!r} + "
            f"{interface_id!r}; allowed={list(allowed)}"
        )
    arm = representation_arm(preprocessing_id)
    if interface.get("representation_arm") != arm:
        raise ValueError("preprocessing/interface representation-arm mismatch")
    return preprocess, interface


@dataclass(frozen=True)
class CropPreprocessResult:
    canonical: np.ndarray | None
    is_blank: bool
    lower_intensity: float | None
    upper_intensity: float | None

    @property
    def gray(self) -> np.ndarray | None:
        if self.canonical is not None and self.canonical.ndim != 2:
            raise ValueError("RGB preprocessing has no canonical grayscale output")
        return self.canonical


def rgb_to_gray_uint8(
    rgb: np.ndarray, preprocessing_id: str = DEFAULT_PREPROCESSING_ID
) -> np.ndarray:
    arr = np.asarray(rgb)
    if arr.dtype != np.uint8:
        raise TypeError(f"expected uint8 RGB input, got {arr.dtype}")
    if arr.ndim != 3 or arr.shape[2] != 3:
        raise ValueError(f"expected HxWx3 RGB input, got shape {arr.shape}")
    spec = get_preprocess_spec(preprocessing_id)
    if spec.get("representation_arm") != "gray":
        raise ValueError(f"preprocessing_id {preprocessing_id!r} is not a grayscale arm")
    algorithm = spec["grayscale"]["algorithm"]
    wide = arr.astype(np.uint16, copy=False)
    if algorithm == "BT.601 fixed-point luminance":
        gray = (
            77 * wide[..., 0]
            + 150 * wide[..., 1]
            + 29 * wide[..., 2]
            + 128
        ) >> 8
    elif algorithm == "arithmetic RGB intensity":
        gray = np.rint(wide.astype(np.float32).mean(axis=2))
    elif algorithm == "Luster":
        gray = np.rint(
            (wide.min(axis=2).astype(np.float32) + wide.max(axis=2)) / 2.0
        )
    else:
        raise RuntimeError(f"unimplemented grayscale algorithm: {algorithm}")
    return gray.astype(np.uint8, copy=False)


def is_blank_gray(gray: np.ndarray) -> bool:
    if gray.dtype != np.uint8 or gray.ndim != 2:
        raise ValueError("blank detection requires a 2-D uint8 grayscale plane")
    return bool(np.mean(gray > 240) > 0.9)


def common_membership_plane(rgb: np.ndarray) -> np.ndarray:
    return rgb_to_gray_uint8(rgb, "luster_raw_gray_v1")


def is_blank_common_membership(rgb: np.ndarray) -> bool:
    return is_blank_gray(common_membership_plane(rgb))


def normalize_gray_uint8(
    gray: np.ndarray, preprocessing_id: str = DEFAULT_PREPROCESSING_ID
) -> tuple[np.ndarray, float, float]:
    if gray.dtype != np.uint8 or gray.ndim != 2:
        raise ValueError("intensity normalization requires 2-D uint8 grayscale")
    intensity = get_preprocess_spec(preprocessing_id)["intensity"]
    if not intensity["enabled"]:
        return gray.copy(), float(gray.min()), float(gray.max())
    lo, hi = np.percentile(
        gray,
        [intensity["lower_percentile"], intensity["upper_percentile"]],
        method="linear",
    )
    lo_f, hi_f = float(lo), float(hi)
    if hi_f - lo_f < 1.0:
        return gray.copy(), lo_f, hi_f
    scaled = np.rint(
        np.clip((gray.astype(np.float32) - lo_f) / (hi_f - lo_f), 0.0, 1.0)
        * 255.0
    ).astype(np.uint8)
    return scaled, lo_f, hi_f


def preprocess_rgb_crop(
    rgb: np.ndarray,
    preprocessing_id: str = DEFAULT_PREPROCESSING_ID,
    *,
    slide_grayod_params: dict[str, Any] | None = None,
    slide_grayod_params_sha256: str | None = None,
    slide_grayod_frozen: FrozenSlideGrayODParams | None = None,
) -> CropPreprocessResult:
    spec = get_preprocess_spec(preprocessing_id)
    if is_blank_common_membership(rgb):
        return CropPreprocessResult(None, True, None, None)
    if spec.get("representation_arm") == "rgb":
        if (
            slide_grayod_params is not None
            or slide_grayod_params_sha256 is not None
            or slide_grayod_frozen is not None
        ):
            raise ValueError("slide gray-OD params supplied to an RGB preprocessing ID")
        canonical = np.ascontiguousarray(np.asarray(rgb).copy())
        if canonical.dtype != np.uint8 or canonical.ndim != 3 or canonical.shape[2] != 3:
            raise TypeError("native RGB preprocessing requires HxWx3 uint8 input")
        return CropPreprocessResult(
            canonical,
            False,
            float(canonical.min()),
            float(canonical.max()),
        )
    gray = rgb_to_gray_uint8(rgb, preprocessing_id)
    if preprocessing_id == "luster_slide_grayod_scale_v1":
        if slide_grayod_frozen is not None:
            if slide_grayod_params is not None or slide_grayod_params_sha256 is not None:
                raise ValueError("provide either frozen or raw slide gray-OD params, not both")
            params = slide_grayod_frozen.params
        else:
            params = validate_slide_grayod_params(
                slide_grayod_params, slide_grayod_params_sha256
            )
        gray = apply_slide_grayod_scale(gray, params)
    elif (
        slide_grayod_params is not None
        or slide_grayod_params_sha256 is not None
        or slide_grayod_frozen is not None
    ):
        raise ValueError("slide gray-OD params supplied to a non-grayOD preprocessing ID")
    normalized, lo, hi = normalize_gray_uint8(gray, preprocessing_id)
    if normalized.ndim != 2:
        raise RuntimeError("canonical grayscale output must be one 2-D plane")
    return CropPreprocessResult(normalized, False, lo, hi)


def assert_gray3_equal(gray3: np.ndarray) -> None:
    arr = np.asarray(gray3)
    if arr.ndim != 3 or arr.shape[2] != 3:
        raise ValueError(f"expected HxWx3 backbone interface, got {arr.shape}")
    if not (
        np.array_equal(arr[..., 0], arr[..., 1])
        and np.array_equal(arr[..., 0], arr[..., 2])
    ):
        raise ValueError("a1 colour-channel prohibition violated: channels differ")


def gray_to_backbone_chw(
    gray: np.ndarray,
    interface_id: str = DEFAULT_BACKBONE_INTERFACE_ID,
) -> np.ndarray:
    plane = np.asarray(gray)
    if plane.dtype != np.uint8 or plane.ndim != 2:
        raise TypeError("backbone interface conversion requires one 2-D uint8 gray plane")
    interface = get_backbone_interface_spec(interface_id)
    scaled = plane.astype(np.float32) / np.float32(255.0)
    if interface_id == "folded_patch_projection_imagenet_v1":
        chw = scaled[None, ...]
    elif interface_id == "replicate_imagenet_affine_v1":
        chw = np.repeat(scaled[None, ...], 3, axis=0)
        mean = np.asarray(IMAGENET_MEAN, dtype=np.float32)[:, None, None]
        std = np.asarray(IMAGENET_STD, dtype=np.float32)[:, None, None]
        chw = (chw - mean) / std
    else:
        raise ValueError(f"unsupported backbone interface: {interface['backbone_interface_id']}")
    return np.ascontiguousarray(chw)


def rgb_to_backbone_chw(
    rgb: np.ndarray,
    interface_id: str = DEFAULT_RGB_BACKBONE_INTERFACE_ID,
) -> np.ndarray:
    arr = np.asarray(rgb)
    if arr.dtype != np.uint8 or arr.ndim != 3 or arr.shape[2] != 3:
        raise TypeError("native RGB interface requires one HxWx3 uint8 crop")
    interface = get_backbone_interface_spec(interface_id)
    if interface_id != DEFAULT_RGB_BACKBONE_INTERFACE_ID:
        raise ValueError(f"RGB crop is incompatible with interface {interface_id!r}")
    if interface.get("representation_arm") != "rgb":
        raise ValueError("native RGB interface registry arm mismatch")
    chw = np.moveaxis(arr.astype(np.float32) / np.float32(255.0), -1, 0)
    mean = np.asarray(IMAGENET_MEAN, dtype=np.float32)[:, None, None]
    std = np.asarray(IMAGENET_STD, dtype=np.float32)[:, None, None]
    return np.ascontiguousarray((chw - mean) / std)


def canonical_to_backbone_chw(
    canonical: np.ndarray,
    interface_id: str,
) -> np.ndarray:
    interface = get_backbone_interface_spec(interface_id)
    if interface.get("representation_arm") == "gray":
        return gray_to_backbone_chw(canonical, interface_id)
    if interface.get("representation_arm") == "rgb":
        return rgb_to_backbone_chw(canonical, interface_id)
    raise ValueError(f"unsupported representation arm for interface {interface_id!r}")


def gray3_to_backbone_chw(
    gray3: np.ndarray,
    interface_id: str = DEFAULT_BACKBONE_INTERFACE_ID,
) -> np.ndarray:
    assert_gray3_equal(gray3)
    return gray_to_backbone_chw(np.asarray(gray3)[..., 0], interface_id)


def fold_patch_projection_conv(conv: Any) -> Any:
    import torch
    import torch.nn as nn

    if not isinstance(conv, nn.Conv2d):
        raise TypeError(f"patch projection must be torch.nn.Conv2d, got {type(conv)}")
    if conv.in_channels != 3 or conv.groups != 1:
        raise ValueError("folding requires a dense three-input-channel patch Conv")
    padding = conv.padding if isinstance(conv.padding, tuple) else (conv.padding, conv.padding)
    if any(int(value) != 0 for value in padding):
        raise ValueError("exact ImageNet-affine folding requires zero patch-conv padding")
    device, dtype = conv.weight.device, conv.weight.dtype
    folded = nn.Conv2d(
        in_channels=1,
        out_channels=conv.out_channels,
        kernel_size=conv.kernel_size,
        stride=conv.stride,
        padding=conv.padding,
        dilation=conv.dilation,
        groups=1,
        bias=True,
        padding_mode=conv.padding_mode,
    ).to(device=device, dtype=dtype)
    mean = torch.tensor(IMAGENET_MEAN, device=device, dtype=dtype).view(1, 3, 1, 1)
    std = torch.tensor(IMAGENET_STD, device=device, dtype=dtype).view(1, 3, 1, 1)
    with torch.no_grad():
        folded.weight.copy_((conv.weight / std).sum(dim=1, keepdim=True))
        base_bias = (
            conv.bias
            if conv.bias is not None
            else torch.zeros(conv.out_channels, device=device, dtype=dtype)
        )
        affine_bias = (conv.weight * (mean / std)).sum(dim=(1, 2, 3))
        folded.bias.copy_(base_bias - affine_bias)
    return folded


def fold_model_patch_projection(model: Any, backbone: str) -> None:
    if backbone == "uni2":
        model.patch_embed.proj = fold_patch_projection_conv(model.patch_embed.proj)
    elif backbone == "phikon":
        patch = model.embeddings.patch_embeddings
        patch.projection = fold_patch_projection_conv(patch.projection)
        if not hasattr(model, "config") or not hasattr(model.config, "num_channels"):
            raise ValueError("Phikon model config does not expose num_channels")
        if not hasattr(patch, "num_channels"):
            raise ValueError("Phikon patch embeddings do not expose num_channels")
        model.config.num_channels = 1
        patch.num_channels = 1
    else:
        raise ValueError(f"unsupported backbone {backbone!r}")


def crop_digest(cell_id: str, crop: np.ndarray | None, *, blank: bool = False) -> bytes:
    h = hashlib.sha256()
    encoded = str(cell_id).encode("utf-8")
    h.update(len(encoded).to_bytes(8, "little"))
    h.update(encoded)
    if blank:
        h.update(b"A1_BLANK_CROP")
    elif crop is None:
        raise ValueError("non-blank digest requires crop bytes")
    else:
        arr = np.ascontiguousarray(crop)
        h.update(str(arr.dtype).encode("ascii"))
        h.update(np.asarray(arr.shape, dtype=np.int64).tobytes())
        h.update(arr.tobytes(order="C"))
    return h.digest()
