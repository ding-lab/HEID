#!/usr/bin/env python3

from __future__ import annotations
from pathlib import Path
_RELEASE = Path(__file__).resolve().parents[2]

import argparse
import hashlib
import importlib.util
import json
import os
import random
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler

from feature_contract import adapter_path
from grayscale_preprocess import (
    MEMBERSHIP_FILTER_ID,
    backbone_interface_sha256,
    fold_model_patch_projection,
    get_backbone_interface_spec,
    get_preprocess_spec,
    gray_to_backbone_chw,
    preprocess_spec_sha256,
)
from label_contract import (
    CANONICAL_CLASSES_17,
    LABEL_MAPPING_ID,
    LABEL_MAPPING_SHA256,
    contract_manifest_record,
)


A1 = Path(__file__).resolve().parents[1]
PC = A1.parent
SPLIT = A1 / "data/split_outer5_inner5.json"
COHORT = A1 / "data/cohort_clean177.csv"
DEFAULT_CONFIG = A1 / "configs/gray_adapter_formal_registry.json"
CROPBANK_ROOT = A1 / "outputs/cropbanks"
SHARED_SCRIPTS = _RELEASE / "cell/scripts"

CLASSES = list(CANONICAL_CLASSES_17)
N_CLASS = len(CLASSES)
FEATURE_DIM = {"uni2": 1536, "phikon": 1024}
BASE_MODEL_ID = {"uni2": "UNI2-h/vit_giant_patch14_224", "phikon": "owkin/phikon-v2"}


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


def atomic_torch_save(path: Path, value: Any) -> None:
    tmp = path.with_name(path.name + f".tmp.{os.getpid()}")
    torch.save(value, tmp)
    tmp.replace(path)


def load_run_config(path: Path) -> dict[str, Any]:
    config = json.loads(path.read_text())
    if config.get("schema_version") != "a1.gray_adapter_run.v1":
        raise ValueError(f"unsupported gray adapter config schema in {path}")
    split_id = json.loads(SPLIT.read_text()).get("split_id")
    if config.get("split_id") != split_id:
        raise ValueError("config split_id differs from the canonical reference split")
    if config.get("label_mapping_id") != LABEL_MAPPING_ID:
        raise ValueError("config label_mapping_id differs from frozen raw-to-refine17")
    if config.get("label_mapping_sha256") != LABEL_MAPPING_SHA256:
        raise ValueError("config label_mapping_sha256 differs from frozen raw-to-refine17")
    candidates = config.get("preprocessing_candidates")
    if not isinstance(candidates, list) or not candidates:
        raise ValueError("config must declare at least one preprocessing candidate")
    candidate_ids: list[str] = []
    for candidate in candidates:
        preprocessing_id = str(candidate.get("id", ""))
        preprocessing_spec = get_preprocess_spec(preprocessing_id)
        if preprocessing_spec.get("formal_launch_allowed") is not True:
            raise ValueError(f"non-formal preprocessing is forbidden: {preprocessing_id}")
        if candidate.get("sha256") != preprocess_spec_sha256(preprocessing_id):
            raise ValueError(
                f"config preprocessing hash differs from the runtime registry for {preprocessing_id}"
            )
        candidate_ids.append(preprocessing_id)
    if len(candidate_ids) != len(set(candidate_ids)):
        raise ValueError("config preprocessing candidate IDs are duplicated")
    interface = config.get("backbone_interface", {})
    interface_id = str(interface.get("id", ""))
    interface_spec = get_backbone_interface_spec(interface_id)
    if interface_spec.get("formal_launch_allowed") is not True:
        raise ValueError(f"non-formal backbone interface is forbidden: {interface_id}")
    if interface.get("sha256") != backbone_interface_sha256(interface_id):
        raise ValueError("config backbone-interface hash differs from the runtime registry")
    assert_single_plane_interface(interface_id)
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


def assert_single_plane_interface(interface_id: str) -> None:
    probe = gray_to_backbone_chw(np.zeros((224, 224), dtype=np.uint8), interface_id)
    if probe.dtype != np.float32 or probe.shape != (1, 224, 224):
        raise RuntimeError(
            "RGB/replicated adapter interface forbidden: gray adapter training requires "
            f"one channel, got {probe.shape} for {interface_id}"
        )


def cropbank_dir(preprocessing_id: str, fold: int, root: Path = CROPBANK_ROOT) -> Path:
    get_preprocess_spec(preprocessing_id)
    if Path(preprocessing_id).name != preprocessing_id:
        raise ValueError("preprocessing_id must be a single path component")
    return root / LABEL_MAPPING_ID / preprocessing_id / f"fold{fold}"


def validate_gray_cropbank_arrays(
    crops: np.ndarray,
    labels: np.ndarray,
    samples: np.ndarray,
    cells: np.ndarray,
) -> None:
    if crops.dtype != np.uint8 or crops.ndim != 3 or tuple(crops.shape[1:]) != (224, 224):
        raise RuntimeError(
            "RGB cropbank forbidden: expected uint8 [N,224,224] single-plane crops, "
            f"got {crops.dtype} {crops.shape}"
        )
    if labels.dtype != np.int64 or labels.ndim != 1:
        raise RuntimeError("gray cropbank labels must be int64 [N]")
    if samples.ndim != 1 or cells.ndim != 1:
        raise RuntimeError("gray cropbank identity arrays must be one-dimensional")
    if not (len(crops) == len(labels) == len(samples) == len(cells)):
        raise RuntimeError("gray cropbank arrays are not identity-aligned")
    if len(labels) == 0 or int(labels.min()) < 0 or int(labels.max()) >= N_CLASS:
        raise RuntimeError("gray cropbank labels violate the 17-class contract")


def load_gray_cropbank(
    fold: int,
    split: dict[str, Any],
    preprocessing_id: str,
    *,
    root: Path = CROPBANK_ROOT,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict[str, Any], str]:
    bank_dir = cropbank_dir(preprocessing_id, fold, root)
    paths = {
        "crops": bank_dir / "train_gray.uint8.npy",
        "labels": bank_dir / "train_labels.int64.npy",
        "samples": bank_dir / "train_sample_ids.npy",
        "cells": bank_dir / "train_cell_ids.npy",
        "manifest": bank_dir / "manifest.json",
    }
    missing = [str(path) for path in paths.values() if not path.exists()]
    if missing:
        raise FileNotFoundError(f"gray cropbank is incomplete: {missing}")
    manifest = json.loads(paths["manifest"].read_text())
    expected = {
        "status": "PASS",
        "schema_version": "a1.gray_cropbank.v1",
        "split_id": split["split_id"],
        "fold": int(fold),
        "preprocessing_id": preprocessing_id,
        "preprocessing_sha256": preprocess_spec_sha256(preprocessing_id),
        "representation_arm": "gray",
        "representation": "single_plane_uint8_NHW",
        "membership_filter_id": MEMBERSHIP_FILTER_ID,
        "label_mapping_id": LABEL_MAPPING_ID,
        "label_mapping_sha256": LABEL_MAPPING_SHA256,
    }
    mismatch = {
        key: {"expected": value, "observed": manifest.get(key)}
        for key, value in expected.items()
        if manifest.get(key) != value
    }
    if mismatch:
        raise RuntimeError(f"gray cropbank provenance mismatch: {mismatch}")
    mapping_source = manifest.get("label_mapping_source")
    if mapping_source != contract_manifest_record():
        raise RuntimeError(
            "gray cropbank label_mapping_source is absent or differs"
        )
    for key in ("crops", "labels", "samples", "cells"):
        record = manifest.get("outputs", {}).get(key, {})
        if record.get("path") != str(paths[key].resolve()):
            raise RuntimeError(f"gray cropbank manifest path mismatch for {key}")
        observed_sha = sha256_file(paths[key])
        if record.get("sha256") != observed_sha:
            raise RuntimeError(f"gray cropbank hash mismatch for {key}")
    crops = np.load(paths["crops"], mmap_mode="r", allow_pickle=False)
    labels = np.load(paths["labels"], mmap_mode="r", allow_pickle=False)
    samples = np.load(paths["samples"], mmap_mode="r", allow_pickle=False)
    cells = np.load(paths["cells"], mmap_mode="r", allow_pickle=False)
    validate_gray_cropbank_arrays(crops, labels, samples, cells)
    if int(manifest.get("n_kept", -1)) != len(labels):
        raise RuntimeError("gray cropbank n_kept differs from durable arrays")
    slide_grayod = manifest.get("slide_grayod")
    is_slide_grayod = preprocessing_id == "luster_slide_grayod_scale_v1"
    if not isinstance(slide_grayod, dict) or bool(slide_grayod.get("enabled")) != is_slide_grayod:
        raise RuntimeError("gray cropbank slide-grayOD enablement differs from preprocessing_id")
    if is_slide_grayod:
        params_manifest = slide_grayod.get("params_manifest")
        if not isinstance(params_manifest, dict):
            raise RuntimeError("grayOD cropbank has no bound params manifest")
        params_path = Path(str(params_manifest.get("path", "")))
        if not params_path.exists() or sha256_file(params_path) != params_manifest.get("sha256"):
            raise RuntimeError("grayOD params manifest is missing or hash-mismatched")
    outer = next(
        (item for item in split["outer_folds"] if int(item["outer_fold"]) == int(fold)),
        None,
    )
    if outer is None:
        raise ValueError(f"outer fold {fold} is absent")
    observed_samples = set(samples.astype(str).tolist())
    train_samples = set(map(str, outer["train_samples"]))
    test_samples = set(map(str, outer["test_samples"]))
    if observed_samples & test_samples:
        raise RuntimeError("gray cropbank contains an outer-test sample")
    if observed_samples - train_samples:
        raise RuntimeError("gray cropbank contains a sample outside explicit outer train")
    return crops, labels, samples, cells, manifest, sha256_file(paths["manifest"])


class GrayCropDataset(Dataset):

    def __init__(self, crops: np.ndarray, labels: np.ndarray, augment: bool) -> None:
        if crops.ndim != 3 or crops.dtype != np.uint8:
            raise RuntimeError("GrayCropDataset refuses non-single-plane crops")
        self.crops = crops
        self.labels = labels
        self.augment = bool(augment)

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int]:


        plane = torch.from_numpy(np.array(self.crops[index], copy=True)).unsqueeze(0)
        tensor = plane.to(dtype=torch.float32).div_(255.0)
        if self.augment:
            if bool(torch.rand(()) < 0.5):
                tensor = tensor.flip(-1)
            if bool(torch.rand(()) < 0.5):
                tensor = tensor.flip(-2)
        if tensor.shape != (1, 224, 224):
            raise RuntimeError("geometry augmentation changed the one-plane contract")
        return tensor.contiguous(), int(self.labels[index])


def _load_module(name: str, path: Path) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load inherited model implementation {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def resolve_base_model_sources(backbone: str, module: Any) -> dict[str, Any]:
    artifacts: list[Path] = []
    if backbone == "uni2":
        artifacts = [Path(path) for path in module.UNI2_PATHS if Path(path).exists()]
        if not artifacts:
            raise FileNotFoundError(f"UNI2 weights absent from {module.UNI2_PATHS}")
        artifacts = artifacts[:1]
    elif backbone == "phikon":
        from transformers.utils.hub import cached_file

        model_id = BASE_MODEL_ID[backbone]
        for single in ("model.safetensors", "pytorch_model.bin"):
            try:
                resolved = cached_file(model_id, single, local_files_only=True)
            except Exception:
                resolved = None
            if resolved:
                artifacts = [Path(resolved)]
                break
        if not artifacts:
            for index_name in ("model.safetensors.index.json", "pytorch_model.bin.index.json"):
                try:
                    index_path = cached_file(model_id, index_name, local_files_only=True)
                except Exception:
                    index_path = None
                if not index_path:
                    continue
                index = json.loads(Path(index_path).read_text())
                shard_names = sorted(set(index.get("weight_map", {}).values()))
                if not shard_names:
                    continue
                artifacts = [Path(index_path)]
                for shard_name in shard_names:
                    shard = cached_file(model_id, shard_name, local_files_only=True)
                    if not shard:
                        raise FileNotFoundError(f"cached Phikon shard missing: {shard_name}")
                    artifacts.append(Path(shard))
                break
        if not artifacts:
            raise FileNotFoundError("offline Phikon-v2 weight artifacts could not be resolved")
    else:
        raise ValueError(f"unsupported backbone {backbone!r}")
    records = [{"path": str(path.resolve()), "sha256": sha256_file(path)} for path in artifacts]
    return {
        "base_model_id": BASE_MODEL_ID[backbone],
        "artifacts": records,
        "source_sha256": canonical_sha256(records),
    }


def load_single_plane_model(
    backbone: str,
    device: torch.device,
    interface_id: str,
    *,
    lora_rank: int,
    lora_alpha: int,
    lora_dropout: float,
) -> tuple[nn.Module, Any, list[nn.Module], dict[str, Any]]:
    assert_single_plane_interface(interface_id)
    source = SHARED_SCRIPTS / ("train_lora_pc.py" if backbone == "uni2" else "train_lora_phikon.py")
    module = _load_module(f"gray_lora_{backbone}", source)
    base_provenance = resolve_base_model_sources(backbone, module)
    model = module.load_uni2(device) if backbone == "uni2" else module.load_phikon(device)
    fold_model_patch_projection(model, backbone)
    if backbone == "phikon":
        patch = model.embeddings.patch_embeddings
        if int(model.config.num_channels) != 1 or int(patch.num_channels) != 1:
            raise RuntimeError("Phikon num_channels metadata was not updated to one")
    modules = module.inject_lora(model, lora_rank, lora_alpha, lora_dropout)
    base_provenance.update({
        "model_source": str(source.resolve()),
        "model_source_sha256": sha256_file(source),
    })
    return model, module, modules, base_provenance


def cosine_warmup_lambda(epoch: int, warmup: int, total: int, minimum: float = 1e-6) -> float:
    if epoch < warmup:
        return (epoch + 1) / max(1, warmup)
    progress = (epoch - warmup) / max(1, total - warmup)
    return max(minimum, float(0.5 * (1.0 + np.cos(np.pi * progress))))


def checkpoint_path(
    backbone: str,
    fold: int,
    preprocessing_id: str,
    interface_id: str,
    output_root: Path | None = None,
) -> Path:


    if output_root is None:
        return adapter_path(backbone, fold, preprocessing_id, interface_id)
    name = f"lora_pc_fold{fold}.pt" if backbone == "uni2" else f"lora_phikon_fold{fold}.pt"
    return output_root / LABEL_MAPPING_ID / preprocessing_id / interface_id / backbone / name


def checkpoint_manifest_path(checkpoint: Path) -> Path:
    return checkpoint.with_suffix(".manifest.json")


def validate_adapter_contract(checkpoint: dict[str, Any], expected: dict[str, Any]) -> None:
    contract = checkpoint.get("a1_contract")
    if not isinstance(contract, dict):
        raise RuntimeError("checkpoint has no a1_contract")
    mismatch = {
        key: {"expected": value, "observed": contract.get(key)}
        for key, value in expected.items()
        if contract.get(key) != value
    }
    if mismatch:
        raise RuntimeError(f"adapter provenance mismatch: {mismatch}")
    if contract.get("input_channels") != 1:
        raise RuntimeError("RGB adapter forbidden: checkpoint input_channels is not one")
    if contract.get("outer_test_used_for_training") is not False:
        raise RuntimeError("adapter does not prove outer-test exclusion from training")
    if contract.get("outer_test_used_for_checkpoint_selection") is not False:
        raise RuntimeError("adapter does not prove outer-test exclusion from checkpoint selection")


def _frozen_contract(
    *,
    config: dict[str, Any],
    config_path: Path,
    split: dict[str, Any],
    fold: int,
    backbone: str,
    preprocessing_id: str,
    interface_id: str,
    cropbank_manifest: dict[str, Any],
    cropbank_manifest_sha: str,
    base_provenance: dict[str, Any],
    seed: int,
    epochs: int,
    run_kind: str,
) -> dict[str, Any]:
    slide_grayod = cropbank_manifest.get("slide_grayod", {})
    params_manifest = slide_grayod.get("params_manifest") or {}
    training = config["training"]
    return {
        "schema_version": "a1.gray_lora_checkpoint.v1",
        "split_id": split["split_id"],
        "outer_fold": int(fold),
        "backbone": backbone,
        "representation_arm": "gray",
        "run_kind": run_kind,
        "preprocessing_id": preprocessing_id,
        "preprocessing_sha256": preprocess_spec_sha256(preprocessing_id),
        "backbone_interface_id": interface_id,
        "backbone_interface_sha256": backbone_interface_sha256(interface_id),
        "input_channels": 1,
        "durable_crop_channels": 1,
        "selection_policy": "predeclared_fixed_final_epoch",
        "epochs": int(epochs),
        "lora_recipe": {
            key: training[key]
            for key in (
                "lora_rank",
                "lora_alpha",
                "lora_dropout",
                "lora_lr",
                "head_lr",
                "weight_decay",
                "warmup_epochs",
                "label_smoothing",
                "batch_size",
            )
        },
        "outer_test_used_for_training": False,
        "outer_test_used_for_checkpoint_selection": False,
        "n_validation_crops": 0,
        "seed": int(seed),
        "cropbank_manifest_sha256": cropbank_manifest_sha,
        "cropbank_gray_array_sha256": cropbank_manifest["outputs"]["crops"]["sha256"],
        "membership_filter_id": cropbank_manifest["membership_filter_id"],
        "membership_identity_sha256": cropbank_manifest["membership_identity_sha256"],
        "label_mapping_id": cropbank_manifest["label_mapping_id"],
        "label_mapping_sha256": cropbank_manifest["label_mapping_sha256"],
        "label_mapping_source": cropbank_manifest["label_mapping_source"],
        "cropbank_identity_sha256": canonical_sha256({
            key: cropbank_manifest["outputs"][key]["sha256"]
            for key in ("labels", "samples", "cells")
        }),
        "slide_grayod_params_manifest_sha256": params_manifest.get("sha256"),
        "slide_grayod_training_entries_sha256": params_manifest.get("entries_sha256"),
        "slide_grayod_reference_sha256": params_manifest.get("reference_sha256"),
        "base_model_id": base_provenance["base_model_id"],
        "base_model_source_sha256": base_provenance["source_sha256"],
        "base_model_source_artifacts": base_provenance["artifacts"],
        "model_source": base_provenance["model_source"],
        "model_source_sha256": base_provenance["model_source_sha256"],
        "config": str(config_path.resolve()),
        "config_sha256": sha256_file(config_path),
        "trainer": str(Path(__file__).resolve()),
        "trainer_sha256": sha256_file(Path(__file__)),
        "evidence_status": config["evidence_status"],
    }


def train(
    backbone: str,
    fold: int,
    config_path: Path,
    preprocessing_id: str,
    cropbank_root: Path = CROPBANK_ROOT,
    adapter_root: Path | None = None,
) -> dict[str, Any]:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not visible; a CUDA device is required")
    config = load_run_config(config_path)
    candidate = configured_candidate(config, preprocessing_id)
    interface_id = str(config["backbone_interface"]["id"])
    split = json.loads(SPLIT.read_text())
    crops, labels, samples, _cells, bank_manifest, bank_manifest_sha = load_gray_cropbank(
        fold, split, preprocessing_id, root=cropbank_root
    )
    train_cfg = config["training"]
    epochs = int(train_cfg["epochs"])
    if epochs < 1:
        raise ValueError("epochs must be positive and predeclared")
    seed = int(train_cfg["seed_base"]) + fold + (0 if backbone == "uni2" else 1000)
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    canonical_cropbanks = cropbank_root.resolve() == CROPBANK_ROOT.resolve()
    canonical_adapters = adapter_root is None
    if canonical_cropbanks != canonical_adapters:
        raise ValueError("cropbank and adapter roots must both be canonical or both be isolated smoke roots")
    run_kind = "formal_candidate" if canonical_cropbanks else "diagnostic_smoke"
    if run_kind == "diagnostic_smoke":
        scratch = (A1 / "scratch").resolve()
        if not cropbank_root.resolve().is_relative_to(scratch):
            raise ValueError("diagnostic cropbank root must be below the diagnostic scratch root")
        if adapter_root is None or not adapter_root.resolve().is_relative_to(scratch):
            raise ValueError("diagnostic adapter root must be below the diagnostic scratch root")
    checkpoint = checkpoint_path(
        backbone, fold, preprocessing_id, interface_id, adapter_root
    )
    manifest_path = checkpoint_manifest_path(checkpoint)
    present = checkpoint.exists(), manifest_path.exists()
    if any(present) and not all(present):
        raise RuntimeError(f"partial existing gray adapter output: {checkpoint}, {manifest_path}")

    device = torch.device("cuda")
    model, inherited, lora_modules, base_provenance = load_single_plane_model(
        backbone,
        device,
        interface_id,
        lora_rank=int(train_cfg["lora_rank"]),
        lora_alpha=int(train_cfg["lora_alpha"]),
        lora_dropout=float(train_cfg["lora_dropout"]),
    )
    contract = _frozen_contract(
        config={**config, "evidence_status": candidate["evidence_status"]},
        config_path=config_path,
        split=split,
        fold=fold,
        backbone=backbone,
        preprocessing_id=preprocessing_id,
        interface_id=interface_id,
        cropbank_manifest=bank_manifest,
        cropbank_manifest_sha=bank_manifest_sha,
        base_provenance=base_provenance,
        seed=seed,
        epochs=epochs,
        run_kind=run_kind,
    )
    if all(present):
        prior_manifest = json.loads(manifest_path.read_text())
        prior_checkpoint = torch.load(checkpoint, map_location="cpu", weights_only=False)
        validate_adapter_contract(prior_checkpoint, contract)
        if prior_manifest.get("checkpoint_sha256") != sha256_file(checkpoint):
            raise RuntimeError("existing gray adapter checkpoint hash mismatch")
        return {**prior_manifest, "status": "ALREADY_COMPLETE"}

    dataset = GrayCropDataset(crops, labels, augment=True)
    label_tensor = torch.from_numpy(np.asarray(labels, dtype=np.int64))
    counts = torch.bincount(label_tensor, minlength=N_CLASS).float().clamp(min=1)
    weights = (1.0 / counts)[label_tensor]
    generator = torch.Generator().manual_seed(seed)
    sampler = WeightedRandomSampler(weights, len(dataset), replacement=True, generator=generator)
    workers = min(int(train_cfg["num_workers"]), int(os.environ.get("SLURM_CPUS_PER_TASK", "8")))
    loader = DataLoader(
        dataset,
        batch_size=int(train_cfg["batch_size"]),
        sampler=sampler,
        num_workers=workers,
        pin_memory=True,
        persistent_workers=workers > 0,
        drop_last=False,
        generator=generator,
    )

    head = nn.Linear(FEATURE_DIM[backbone], N_CLASS).to(device)
    lora_params = [
        parameter
        for module in lora_modules
        for parameter in module.parameters()
        if parameter.requires_grad
    ]
    if not lora_params:
        raise RuntimeError("LoRA injection produced no trainable parameters")
    optimizer = torch.optim.AdamW(
        [
            {"params": lora_params, "lr": float(train_cfg["lora_lr"])},
            {"params": head.parameters(), "lr": float(train_cfg["head_lr"])},
        ],
        weight_decay=float(train_cfg["weight_decay"]),
    )
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lambda epoch: cosine_warmup_lambda(epoch, int(train_cfg["warmup_epochs"]), epochs),
    )
    criterion = nn.CrossEntropyLoss(label_smoothing=float(train_cfg["label_smoothing"]))
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True

    start = time.time()
    history: list[dict[str, Any]] = []
    for epoch in range(epochs):
        epoch_start = time.time()
        model.train()
        head.train()
        total_loss = 0.0
        correct = 0
        total = 0
        for images, targets in loader:
            if images.ndim != 4 or images.shape[1] != 1:
                raise RuntimeError("RGB tensor reached gray adapter training")
            images = images.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                output = model(images)
                features = output if backbone == "uni2" else output.last_hidden_state[:, 0, :]
                logits = head(features.float())
                loss = criterion(logits, targets)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(lora_params + list(head.parameters()), 1.0)
            optimizer.step()
            total_loss += float(loss.item()) * len(images)
            correct += int((logits.argmax(dim=1) == targets).sum().item())
            total += len(images)
        scheduler.step()
        record = {
            "epoch": epoch + 1,
            "train_loss": round(total_loss / max(1, total), 6),
            "train_accuracy_diagnostic_only": round(correct / max(1, total), 6),
            "learning_rate": [float(group["lr"]) for group in optimizer.param_groups],
            "sec": round(time.time() - epoch_start, 1),
            "selection_metric": None,
        }
        history.append(record)
        print(
            f"[gray LoRA {backbone} fold {fold}] epoch {epoch + 1}/{epochs} "
            f"loss={record['train_loss']:.6f} acc={record['train_accuracy_diagnostic_only']:.4f}",
            flush=True,
        )

    if len(history) != epochs or history[-1]["epoch"] != epochs:
        raise RuntimeError("training history does not end at the predeclared final epoch")
    lora_state = inherited.get_lora_state_dict(model)
    head_state = {name: value.detach().cpu() for name, value in head.state_dict().items()}
    payload = {
        "a1_contract": contract,
        "lora_state_dict": lora_state,
        "head_state_dict": head_state,
        "cell_types": CLASSES,
        "fold": fold,
        "backbone": backbone,
        "history": history,
        "n_train": int(len(labels)),
        "n_train_samples_represented": int(len(set(samples.astype(str).tolist()))),
        "n_val": 0,
        "val_auc": None,
        "saved_epoch": epochs,
        "note": "a1 single-plane grayscale LoRA; fixed final epoch; outer test never loaded or selected on.",
    }
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    atomic_torch_save(checkpoint, payload)
    result = {
        "status": "PASS",
        **contract,
        "checkpoint": str(checkpoint.resolve()),
        "checkpoint_sha256": sha256_file(checkpoint),
        "history": history,
        "n_train": int(len(labels)),
        "n_train_samples_represented": int(len(set(samples.astype(str).tolist()))),
        "wall_sec": round(time.time() - start, 1),
        "node": os.environ.get("SLURMD_NODENAME"),
        "job_id": os.environ.get("SLURM_JOB_ID"),
        "gpu": torch.cuda.get_device_name(0),
        "torch": torch.__version__,
    }
    atomic_json(manifest_path, result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--backbone", choices=sorted(FEATURE_DIM), required=True)
    parser.add_argument("--fold", type=int, choices=range(5), required=True)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--preprocessing-id", required=True)
    parser.add_argument("--cropbank-root", type=Path, default=CROPBANK_ROOT)
    parser.add_argument("--adapter-root", type=Path)
    args = parser.parse_args()
    print(
        json.dumps(
            train(
                args.backbone,
                args.fold,
                args.config,
                args.preprocessing_id,
                args.cropbank_root,
                args.adapter_root,
            ),
            indent=2,
            sort_keys=True,
        )
    )
