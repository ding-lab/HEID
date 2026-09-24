#!/usr/bin/env python3

from __future__ import annotations

import argparse
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

from build_rgb_cropbank import (
    CROPBANK_ROOT,
    DEFAULT_CONFIG,
    INTERFACE_ID,
    MEMBERSHIP_FILTER_ID,
    PREPROCESSING_ID,
    cropbank_dir,
    load_config,
    membership_sha256,
    validate_rgb_bank,
)
from feature_contract import adapter_path
from grayscale_preprocess import (
    backbone_interface_sha256,
    preprocess_spec_sha256,
    rgb_to_backbone_chw,
    validate_formal_preprocessing_interface,
)
from label_contract import LABEL_MAPPING_ID, LABEL_MAPPING_SHA256, contract_manifest_record
from train_gray_lora import (
    A1,
    CLASSES,
    FEATURE_DIM,
    N_CLASS,
    SHARED_SCRIPTS,
    SPLIT,
    _load_module,
    atomic_json,
    atomic_torch_save,
    canonical_sha256,
    cosine_warmup_lambda,
    resolve_base_model_sources,
    sha256_file,
)


def load_rgb_cropbank(
    fold: int,
    split: dict[str, Any],
    *,
    root: Path | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict[str, Any], str]:
    bank_dir = cropbank_dir(fold, root if root is not None else A1 / "outputs/cropbanks")
    paths = {
        "crops": bank_dir / "train_rgb.uint8.npy",
        "labels": bank_dir / "train_labels.int64.npy",
        "samples": bank_dir / "train_sample_ids.npy",
        "cells": bank_dir / "train_cell_ids.npy",
        "manifest": bank_dir / "manifest.json",
    }
    missing = [str(path) for path in paths.values() if not path.exists()]
    if missing:
        raise FileNotFoundError(f"strict RGB cropbank is incomplete: {missing}")
    manifest = json.loads(paths["manifest"].read_text())
    expected = {
        "status": "PASS",
        "schema_version": "a1.rgb_cropbank.v1",
        "split_id": split["split_id"],
        "fold": int(fold),
        "preprocessing_id": PREPROCESSING_ID,
        "preprocessing_sha256": preprocess_spec_sha256(PREPROCESSING_ID),
        "backbone_interface_id": INTERFACE_ID,
        "backbone_interface_sha256": backbone_interface_sha256(INTERFACE_ID),
        "representation_arm": "rgb",
        "representation": "native_rgb_uint8_NHWC",
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
        raise RuntimeError(f"strict RGB cropbank provenance mismatch: {mismatch}")
    mapping_source = manifest.get("label_mapping_source")
    if mapping_source != contract_manifest_record():
        raise RuntimeError(
            "RGB cropbank label_mapping_source is absent or differs"
        )
    for key in ("crops", "labels", "samples", "cells"):
        record = manifest.get("outputs", {}).get(key, {})
        if record.get("path") != str(paths[key].resolve()):
            raise RuntimeError(f"strict RGB cropbank path mismatch for {key}")
        if record.get("sha256") != sha256_file(paths[key]):
            raise RuntimeError(f"strict RGB cropbank hash mismatch for {key}")
    crops = np.load(paths["crops"], mmap_mode="r", allow_pickle=False)
    labels = np.load(paths["labels"], mmap_mode="r", allow_pickle=False)
    samples = np.load(paths["samples"], mmap_mode="r", allow_pickle=False)
    cells = np.load(paths["cells"], mmap_mode="r", allow_pickle=False)
    validate_rgb_bank(crops, labels, samples, cells)
    if int(manifest.get("n_kept", -1)) != len(labels):
        raise RuntimeError("strict RGB cropbank n_kept differs from arrays")
    if manifest.get("membership_identity_sha256") != membership_sha256(labels, samples, cells):
        raise RuntimeError("strict RGB cropbank membership identity digest mismatch")
    outer = next(
        (item for item in split["outer_folds"] if int(item["outer_fold"]) == int(fold)),
        None,
    )
    if outer is None:
        raise ValueError(f"outer fold {fold} is absent")
    observed = set(samples.astype(str).tolist())
    test_samples = set(map(str, outer["test_samples"]))
    train_samples = set(map(str, outer["train_samples"]))
    if observed & test_samples:
        raise RuntimeError("strict RGB cropbank contains an outer-test sample")
    if observed - train_samples:
        raise RuntimeError("strict RGB cropbank contains a sample outside outer train")
    return crops, labels, samples, cells, manifest, sha256_file(paths["manifest"])


class RGBCropDataset(Dataset):

    def __init__(self, crops: np.ndarray, labels: np.ndarray, augment: bool) -> None:
        if crops.dtype != np.uint8 or crops.ndim != 4 or crops.shape[-1] != 3:
            raise RuntimeError("RGBCropDataset requires native uint8 NHWC RGB")
        self.crops = crops
        self.labels = labels
        self.augment = bool(augment)

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int]:
        rgb = np.array(self.crops[index], copy=True)
        tensor = torch.from_numpy(rgb_to_backbone_chw(rgb, INTERFACE_ID))
        if self.augment:
            if bool(torch.rand(()) < 0.5):
                tensor = tensor.flip(-1)
            if bool(torch.rand(()) < 0.5):
                tensor = tensor.flip(-2)
        if tensor.dtype != torch.float32 or tensor.shape != (3, 224, 224):
            raise RuntimeError("native RGB interface changed channel or shape contract")
        return tensor.contiguous(), int(self.labels[index])


def load_native_rgb_model(
    backbone: str,
    device: torch.device,
    *,
    lora_rank: int,
    lora_alpha: int,
    lora_dropout: float,
) -> tuple[nn.Module, Any, list[nn.Module], dict[str, Any]]:
    validate_formal_preprocessing_interface(PREPROCESSING_ID, INTERFACE_ID)
    source = SHARED_SCRIPTS / ("train_lora_pc.py" if backbone == "uni2" else "train_lora_phikon.py")
    inherited = _load_module(f"rgb_lora_{backbone}", source)
    base_provenance = resolve_base_model_sources(backbone, inherited)
    model = inherited.load_uni2(device) if backbone == "uni2" else inherited.load_phikon(device)
    if backbone == "uni2":
        input_channels = int(model.patch_embed.proj.in_channels)
    else:
        patch = model.embeddings.patch_embeddings
        input_channels = int(patch.projection.in_channels)
        if int(model.config.num_channels) != 3 or int(patch.num_channels) != 3:
            raise RuntimeError("Phikon native-RGB channel metadata is not three")
    if input_channels != 3:
        raise RuntimeError("native RGB model patch projection is not three-channel")
    modules = inherited.inject_lora(model, lora_rank, lora_alpha, lora_dropout)
    base_provenance.update({
        "model_source": str(source.resolve()),
        "model_source_sha256": sha256_file(source),
    })
    return model, inherited, modules, base_provenance


def checkpoint_path(
    backbone: str,
    fold: int,
    output_root: Path | None = None,
) -> Path:
    if output_root is None:
        return adapter_path(backbone, fold, PREPROCESSING_ID, INTERFACE_ID)
    name = f"lora_pc_fold{fold}.pt" if backbone == "uni2" else f"lora_phikon_fold{fold}.pt"
    return output_root / LABEL_MAPPING_ID / PREPROCESSING_ID / INTERFACE_ID / backbone / name


def checkpoint_manifest_path(checkpoint: Path) -> Path:
    return checkpoint.with_suffix(".manifest.json")


def validate_rgb_adapter_contract(checkpoint: dict[str, Any], expected: dict[str, Any]) -> None:
    contract = checkpoint.get("a1_contract")
    if not isinstance(contract, dict):
        raise RuntimeError("checkpoint has no a1_contract")
    mismatch = {
        key: {"expected": value, "observed": contract.get(key)}
        for key, value in expected.items()
        if contract.get(key) != value
    }
    if mismatch:
        raise RuntimeError(f"strict RGB adapter provenance mismatch: {mismatch}")
    if contract.get("representation_arm") != "rgb" or contract.get("input_channels") != 3:
        raise RuntimeError("strict RGB adapter representation/channel contract is absent")
    if contract.get("outer_test_used_for_training") is not False:
        raise RuntimeError("RGB adapter does not prove outer-test exclusion from training")
    if contract.get("outer_test_used_for_checkpoint_selection") is not False:
        raise RuntimeError("RGB adapter does not prove outer-test exclusion from selection")


def frozen_contract(
    *,
    config: dict[str, Any],
    config_path: Path,
    split: dict[str, Any],
    fold: int,
    backbone: str,
    cropbank_manifest: dict[str, Any],
    cropbank_manifest_sha: str,
    base_provenance: dict[str, Any],
    seed: int,
    run_kind: str,
) -> dict[str, Any]:
    training = config["training"]
    return {
        "schema_version": "a1.rgb_lora_checkpoint.v1",
        "split_id": split["split_id"],
        "outer_fold": int(fold),
        "backbone": backbone,
        "representation_arm": "rgb",
        "run_kind": run_kind,
        "preprocessing_id": PREPROCESSING_ID,
        "preprocessing_sha256": preprocess_spec_sha256(PREPROCESSING_ID),
        "backbone_interface_id": INTERFACE_ID,
        "backbone_interface_sha256": backbone_interface_sha256(INTERFACE_ID),
        "input_channels": 3,
        "durable_crop_channels": 3,
        "membership_filter_id": MEMBERSHIP_FILTER_ID,
        "membership_identity_sha256": cropbank_manifest["membership_identity_sha256"],
        "label_mapping_id": cropbank_manifest["label_mapping_id"],
        "label_mapping_sha256": cropbank_manifest["label_mapping_sha256"],
        "label_mapping_source": cropbank_manifest["label_mapping_source"],
        "selection_policy": "predeclared_fixed_final_epoch",
        "epochs": int(training["epochs"]),
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
        "cropbank_rgb_array_sha256": cropbank_manifest["outputs"]["crops"]["sha256"],
        "cropbank_identity_sha256": canonical_sha256({
            key: cropbank_manifest["outputs"][key]["sha256"]
            for key in ("labels", "samples", "cells")
        }),
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
    cropbank_root: Path = CROPBANK_ROOT,
    adapter_root: Path | None = None,
) -> dict[str, Any]:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not visible; a CUDA device is required")
    config = load_config(config_path)
    split = json.loads(SPLIT.read_text())
    crops, labels, samples, _cells, bank_manifest, bank_manifest_sha = load_rgb_cropbank(
        fold, split, root=cropbank_root
    )
    training = config["training"]
    epochs = int(training["epochs"])
    if epochs < 1:
        raise ValueError("epochs must be positive and predeclared")
    seed = int(training["seed_base"]) + fold + (0 if backbone == "uni2" else 1000)
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    canonical_cropbanks = cropbank_root.resolve() == CROPBANK_ROOT.resolve()
    canonical_adapters = adapter_root is None
    if canonical_cropbanks != canonical_adapters:
        raise ValueError("RGB cropbank and adapter roots must both be canonical or isolated")
    run_kind = "formal_candidate" if canonical_cropbanks else "diagnostic_smoke"
    if run_kind == "diagnostic_smoke":
        scratch = (A1 / "scratch").resolve()
        if not cropbank_root.resolve().is_relative_to(scratch):
            raise ValueError("diagnostic RGB cropbank root must be below the diagnostic scratch root")
        if adapter_root is None or not adapter_root.resolve().is_relative_to(scratch):
            raise ValueError("diagnostic RGB adapter root must be below the diagnostic scratch root")
    checkpoint = checkpoint_path(backbone, fold, adapter_root)
    manifest_path = checkpoint_manifest_path(checkpoint)
    present = checkpoint.exists(), manifest_path.exists()
    if any(present) and not all(present):
        raise RuntimeError(f"partial existing strict RGB adapter: {checkpoint}, {manifest_path}")

    device = torch.device("cuda")
    model, inherited, lora_modules, base_provenance = load_native_rgb_model(
        backbone,
        device,
        lora_rank=int(training["lora_rank"]),
        lora_alpha=int(training["lora_alpha"]),
        lora_dropout=float(training["lora_dropout"]),
    )
    contract = frozen_contract(
        config=config,
        config_path=config_path,
        split=split,
        fold=fold,
        backbone=backbone,
        cropbank_manifest=bank_manifest,
        cropbank_manifest_sha=bank_manifest_sha,
        base_provenance=base_provenance,
        seed=seed,
        run_kind=run_kind,
    )
    if all(present):
        prior_manifest = json.loads(manifest_path.read_text())
        prior_checkpoint = torch.load(checkpoint, map_location="cpu", weights_only=False)
        validate_rgb_adapter_contract(prior_checkpoint, contract)
        if prior_manifest.get("checkpoint_sha256") != sha256_file(checkpoint):
            raise RuntimeError("existing strict RGB checkpoint hash mismatch")
        return {**prior_manifest, "status": "ALREADY_COMPLETE"}

    dataset = RGBCropDataset(crops, labels, augment=True)
    label_tensor = torch.from_numpy(np.asarray(labels, dtype=np.int64))
    counts = torch.bincount(label_tensor, minlength=N_CLASS).float().clamp(min=1)
    weights = (1.0 / counts)[label_tensor]
    generator = torch.Generator().manual_seed(seed)
    sampler = WeightedRandomSampler(weights, len(dataset), replacement=True, generator=generator)
    workers = min(int(training["num_workers"]), int(os.environ.get("SLURM_CPUS_PER_TASK", "8")))
    loader = DataLoader(
        dataset,
        batch_size=int(training["batch_size"]),
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
        raise RuntimeError("strict RGB LoRA injection produced no trainable parameters")
    optimizer = torch.optim.AdamW(
        [
            {"params": lora_params, "lr": float(training["lora_lr"])},
            {"params": head.parameters(), "lr": float(training["head_lr"])},
        ],
        weight_decay=float(training["weight_decay"]),
    )
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lambda epoch: cosine_warmup_lambda(
            epoch, int(training["warmup_epochs"]), epochs
        ),
    )
    criterion = nn.CrossEntropyLoss(label_smoothing=float(training["label_smoothing"]))
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
            if images.ndim != 4 or images.shape[1] != 3:
                raise RuntimeError("non-RGB tensor reached strict RGB adapter training")
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
        history.append({
            "epoch": epoch + 1,
            "train_loss": round(total_loss / max(1, total), 6),
            "train_accuracy_diagnostic_only": round(correct / max(1, total), 6),
            "learning_rate": [float(group["lr"]) for group in optimizer.param_groups],
            "sec": round(time.time() - epoch_start, 1),
            "selection_metric": None,
        })
        print(
            f"[strict RGB {backbone} fold {fold}] epoch {epoch + 1}/{epochs} "
            f"loss={history[-1]['train_loss']:.6f} "
            f"acc={history[-1]['train_accuracy_diagnostic_only']:.4f}",
            flush=True,
        )
    if len(history) != epochs or history[-1]["epoch"] != epochs:
        raise RuntimeError("RGB training history does not end at fixed final epoch")

    payload = {
        "a1_contract": contract,
        "lora_state_dict": inherited.get_lora_state_dict(model),
        "head_state_dict": {
            name: value.detach().cpu() for name, value in head.state_dict().items()
        },
        "cell_types": CLASSES,
        "fold": fold,
        "backbone": backbone,
        "history": history,
        "n_train": int(len(labels)),
        "n_train_samples_represented": int(len(set(samples.astype(str).tolist()))),
        "n_val": 0,
        "val_auc": None,
        "saved_epoch": epochs,
        "note": "a1 strict native-RGB comparison LoRA; fixed final epoch; no outer-test input or selection.",
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
    parser.add_argument("--cropbank-root", type=Path, default=CROPBANK_ROOT)
    parser.add_argument("--adapter-root", type=Path)
    args = parser.parse_args()
    print(
        json.dumps(
            train(
                args.backbone,
                args.fold,
                args.config,
                args.cropbank_root,
                args.adapter_root,
            ),
            indent=2,
            sort_keys=True,
        )
    )
