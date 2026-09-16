#!/usr/bin/env python3

from __future__ import annotations
from pathlib import Path
_RELEASE = Path(__file__).resolve().parents[2]

import argparse
import json
import os
import random
import sys
import time
from pathlib import Path
from typing import Any, Iterator

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, Sampler, WeightedRandomSampler

import contract as C

sys.path.insert(0, str(_RELEASE / "cell/scripts"))
from grayscale_preprocess import rgb_to_backbone_chw
from train_gray_lora import FEATURE_DIM, cosine_warmup_lambda
from train_rgb_lora import load_native_rgb_model

ARM = "raw_rgb"
INTERFACE_ID = "native_rgb_imagenet_v1"
TRAINING = {
    "epochs": 5,
    "batch_size": 64,
    "num_workers": 8,
    "seed_base": 5100,
    "lora_rank": 16,
    "lora_alpha": 32,
    "lora_dropout": 0.05,
    "lora_lr": 0.0001,
    "head_lr": 0.001,
    "weight_decay": 0.01,
    "warmup_epochs": 1,
    "label_smoothing": 0.1,
    "gradient_clip_norm": 1.0,
    "checkpoint_selection": "fixed_final_epoch_only",
    "outer_test_used_for_training": False,
    "outer_test_used_for_checkpoint_selection": False,
}


class ReplaySampler(Sampler):

    def __init__(self, base: WeightedRandomSampler) -> None:
        self.base = base
        self.epoch = 0

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def __iter__(self) -> Iterator[tuple[int, int, int]]:
        for draw, crop_index in enumerate(self.base):
            yield int(crop_index), self.epoch, int(draw)

    def __len__(self) -> int:
        return len(self.base)


class RawRgbDataset(Dataset):
    def __init__(self, crops: np.ndarray, labels: np.ndarray) -> None:
        self.crops = crops
        self.labels = labels

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, key) -> tuple[torch.Tensor, int]:
        index = int(key[0]) if isinstance(key, tuple) else int(key)
        rgb = np.array(self.crops[index], copy=True)
        tensor = torch.from_numpy(rgb_to_backbone_chw(rgb, INTERFACE_ID))
        if bool(torch.rand(()) < 0.5):
            tensor = tensor.flip(-1)
        if bool(torch.rand(()) < 0.5):
            tensor = tensor.flip(-2)
        if tensor.dtype != torch.float32 or tensor.shape != (3, C.CROP, C.CROP):
            raise RuntimeError("native RGB interface changed its channel or shape contract")
        return tensor.contiguous(), int(self.labels[index])


def load_cropbank(fold: int):
    directory = C.cropbank_dir(fold)
    manifest_path = directory / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    if manifest.get("status") != "PASS" or int(manifest.get("fold", -1)) != fold:
        raise RuntimeError(f"cropbank manifest is not a PASS fold-{fold} bank: {manifest_path}")
    C.assert_fold_binding(manifest.get("fold_fingerprint"), fold, f"cropbank fold{fold}")
    paths = {
        "crops": directory / "train_rgb.uint8.npy",
        "labels": directory / "train_labels.int64.npy",
        "samples": directory / "train_sample_ids.npy",
        "cells": directory / "train_cell_ids.npy",
    }
    for key, path in paths.items():
        if C.sha256_file(path) != manifest["outputs"][key]["sha256"]:
            raise RuntimeError(f"cropbank hash drift: {path}")
    crops = np.load(paths["crops"], mmap_mode="r", allow_pickle=False)
    labels = np.load(paths["labels"], allow_pickle=False)
    samples = np.load(paths["samples"], allow_pickle=False)
    cells = np.load(paths["cells"], allow_pickle=False)
    if crops.dtype != np.uint8 or tuple(crops.shape[1:]) != (C.CROP, C.CROP, 3):
        raise RuntimeError(f"cropbank is not uint8 [N,224,224,3]: {crops.shape}")
    if not (len(crops) == len(labels) == len(samples) == len(cells)):
        raise RuntimeError("cropbank arrays are misaligned")
    return crops, labels, samples, cells, manifest, C.sha256_file(manifest_path)


def train(backbone: str, fold: int) -> dict[str, Any]:
    started = time.time()
    environment = C.require_usable_gpu()
    membership = C.fold_membership()[fold]
    crops, labels, samples, cells, bank_manifest, bank_sha = load_cropbank(fold)

    observed = set(np.asarray(samples).astype(str).tolist())
    C.assert_outside_fold(observed, fold, f"adapter {backbone} fold{fold} cropbank")
    if observed - set(membership["train_samples"]):
        raise RuntimeError("adapter cropbank holds a slide outside the fold train list")

    seed = int(TRAINING["seed_base"]) + fold + (0 if backbone == "uni2" else 1000)
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    generator = torch.Generator().manual_seed(seed)

    out = C.adapter_path(backbone, fold)
    manifest_path = out.with_suffix(".manifest.json")
    fingerprint = {
        "schema_version": "v8.adapter_fingerprint.v1",
        "arm": ARM,
        "backbone": backbone,
        **C.fold_fingerprint(fold),
        "train_samples": membership["train_samples"],
        "cropbank_manifest_sha256": bank_sha,
        "cropbank_outputs": {
            key: bank_manifest["outputs"][key]["sha256"]
            for key in ("crops", "labels", "samples", "cells")
        },
        "cropbank_membership_identity_sha256": bank_manifest["membership_identity_sha256"],
        "n_observed_train_samples": len(observed),
        "observed_train_sample_sha256": C.sha256_list(sorted(observed)),
        "training": TRAINING,
        "seed": seed,
        "trainer_sha256": C.sha256_file(Path(__file__)),
    }
    fingerprint_sha = C.canonical_sha256(fingerprint)
    if out.exists() or manifest_path.exists():
        if not (out.exists() and manifest_path.exists()):
            raise RuntimeError("partial adapter output exists")
        prior = json.loads(manifest_path.read_text())
        if prior.get("fingerprint_sha256") != fingerprint_sha:
            raise RuntimeError("existing adapter fingerprint differs; refusing resume")
        if prior.get("checkpoint_sha256") != C.sha256_file(out):
            raise RuntimeError("existing adapter checkpoint hash differs")
        return {**prior, "status": "ALREADY_COMPLETE"}

    device = torch.device("cuda")
    model, inherited, lora_modules, base_provenance = load_native_rgb_model(
        backbone,
        device,
        lora_rank=int(TRAINING["lora_rank"]),
        lora_alpha=int(TRAINING["lora_alpha"]),
        lora_dropout=float(TRAINING["lora_dropout"]),
    )
    dataset = RawRgbDataset(crops, labels)
    label_tensor = torch.from_numpy(np.asarray(labels, dtype=np.int64))
    counts = torch.bincount(label_tensor, minlength=C.N_CLASS).float().clamp(min=1)
    weights = (1.0 / counts)[label_tensor]
    weighted = WeightedRandomSampler(weights, len(dataset), replacement=True, generator=generator)
    sampler = ReplaySampler(weighted)
    workers = min(int(TRAINING["num_workers"]), int(os.environ.get("SLURM_CPUS_PER_TASK", "8")))
    loader = DataLoader(
        dataset,
        batch_size=int(TRAINING["batch_size"]),
        sampler=sampler,
        num_workers=workers,
        pin_memory=True,
        persistent_workers=workers > 0,
        drop_last=False,
        generator=generator,
    )
    head = nn.Linear(int(FEATURE_DIM[backbone]), C.N_CLASS).to(device)
    lora_params = [p for module in lora_modules for p in module.parameters() if p.requires_grad]
    head_params = list(head.parameters())
    optimizer = torch.optim.AdamW(
        [
            {"params": lora_params, "lr": float(TRAINING["lora_lr"])},
            {"params": head_params, "lr": float(TRAINING["head_lr"])},
        ],
        weight_decay=float(TRAINING["weight_decay"]),
    )
    epochs = int(TRAINING["epochs"])
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lambda epoch: cosine_warmup_lambda(epoch, int(TRAINING["warmup_epochs"]), epochs),
    )
    criterion = nn.CrossEntropyLoss(label_smoothing=float(TRAINING["label_smoothing"]))
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True

    history = []
    for epoch in range(epochs):
        sampler.set_epoch(epoch)
        epoch_start = time.time()
        model.train()
        head.train()
        total_loss = 0.0
        correct = 0
        total = 0
        for images, targets in loader:
            images = images.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                output = model(images)
                features = output if backbone == "uni2" else output.last_hidden_state[:, 0, :]
                logits = head(features.float())
                loss = criterion(logits, targets)
            if not bool(torch.isfinite(loss)):
                raise RuntimeError("adapter loss became non-finite")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                lora_params + head_params, float(TRAINING["gradient_clip_norm"])
            )
            optimizer.step()
            total_loss += float(loss.item()) * len(images)
            correct += int((logits.argmax(1) == targets).sum().item())
            total += len(images)
        scheduler.step()
        history.append({
            "epoch": epoch + 1,
            "train_loss": round(total_loss / max(1, total), 6),
            "train_accuracy_diagnostic_only": round(correct / max(1, total), 6),
            "selection_metric": None,
            "sec": round(time.time() - epoch_start, 1),
        })
        print(f"[{ARM} {backbone} fold{fold}] epoch {epoch + 1}/{epochs} "
              f"loss={history[-1]['train_loss']}", flush=True)
    if history[-1]["epoch"] != epochs:
        raise RuntimeError("adapter did not reach the fixed final epoch")

    payload = {
        "schema_version": "v8.adapter_checkpoint.v1",
        "fingerprint": fingerprint,
        "fingerprint_sha256": fingerprint_sha,
        "lora_state_dict": inherited.get_lora_state_dict(model),
        "head_state_dict": {k: v.detach().cpu() for k, v in head.state_dict().items()},
        "cell_types": C.CLASSES17,
        "arm": ARM,
        "backbone": backbone,
        "fold": fold,
        "saved_epoch": epochs,
        "history": history,
        "n_train": int(len(labels)),
        "n_validation": 0,
        "base_model_provenance": base_provenance,
    }
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_name(out.name + f".tmp.{os.getpid()}")
    torch.save(payload, tmp)
    os.replace(tmp, out)

    result = {
        "schema_version": "v8.adapter_manifest.v1",
        "status": "PASS",
        "arm": ARM,
        "backbone": backbone,
        "outer_fold": fold,
        "canonical_fold_key": C.CANONICAL_FOLD_KEY,
        "excludes_fold": fold,
        "fingerprint_sha256": fingerprint_sha,
        "fold_fingerprint": C.fold_fingerprint(fold),
        "train_samples": membership["train_samples"],
        "train_sample_sha256": C.sha256_list(membership["train_samples"]),
        "held_out_samples": membership["test_samples"],
        "held_out_sample_sha256": C.sha256_list(membership["test_samples"]),
        "n_held_out_slides_seen_in_training": len(
            set(observed) & set(membership["test_samples"])
        ),
        "n_declared_train_slides_seen_in_training": len(
            set(observed) & set(membership["train_samples"])
        ),
        "checkpoint_path": str(out.resolve()),
        "checkpoint_sha256": C.sha256_file(out),
        "saved_epoch": epochs,
        "checkpoint_selection": TRAINING["checkpoint_selection"],
        "n_train": int(len(labels)),
        "n_train_samples": len(observed),
        "history": history,
        "contract_sha256": C.contract_sha256(),
        "cropbank_manifest_sha256": bank_sha,
        "wall_sec": round(time.time() - started, 1),
        "environment": environment,
    }
    C.atomic_json(manifest_path, result)
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--backbone", choices=C.BACKBONES)
    parser.add_argument("--fold", type=int, choices=range(5))
    parser.add_argument("--task-id", type=int,
                        help="0-4 -> uni2 fold0-4, 5-9 -> phikon fold0-4")
    args = parser.parse_args()
    if args.task_id is not None:
        if not 0 <= args.task_id < 10:
            raise ValueError("adapter task-id must be 0..9")
        backbone = C.BACKBONES[args.task_id // 5]
        fold = args.task_id % 5
    else:
        if args.backbone is None or args.fold is None:
            raise ValueError("provide --task-id or --backbone with --fold")
        backbone, fold = args.backbone, args.fold
    result = train(backbone, int(fold))
    print(json.dumps({k: v for k, v in result.items()
                      if k not in ("train_samples", "held_out_samples", "fold_fingerprint")},
                     indent=2, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
