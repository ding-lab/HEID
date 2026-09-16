#!/usr/bin/env python3

from __future__ import annotations
from pathlib import Path
_RELEASE = Path(__file__).resolve().parents[2]

import argparse
import importlib.util
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import tifffile
import torch
import zarr

import contract as C

SHARED_EXTRACTOR = _RELEASE / "cell/scripts/extract_lora_pc.py"
PHIKON_SOURCE = _RELEASE / "cell/scripts/train_lora_phikon.py"
FEATURE_DIM = {"uni2": 1536, "phikon": 1024}
OUT_ROOTS = {
    "uni2": C.FEATURE_ROOT / "frozen_uni2_cls",
    "phikon": C.FEATURE_ROOT / "frozen_phikon_cls",
}
MANIFEST_DIR = C.MANIFEST_ROOT / "frozen_cls"


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def target_samples(backbone: str) -> list[str]:
    return list(C.allowlist())


def run(backbone: str, sample: str) -> dict[str, Any]:
    started = time.time()
    environment = C.require_usable_gpu()
    out_path = OUT_ROOTS[backbone] / f"{sample}.pt"
    manifest_path = MANIFEST_DIR / f"{backbone}_{sample}.json"
    if out_path.is_file() and manifest_path.is_file():
        prior = json.loads(manifest_path.read_text())
        if prior.get("status") == "PASS" and prior.get("feature_file_sha256") == C.sha256_file(out_path):
            return {**prior, "status": "ALREADY_COMPLETE"}
        raise RuntimeError(f"existing frozen bank does not match its manifest: {out_path}")

    source = load_module(f"cell_frozen_{backbone}", SHARED_EXTRACTOR)
    device = torch.device("cuda")
    tf32_state = {
        "cuda_matmul_allow_tf32": bool(torch.backends.cuda.matmul.allow_tf32),
        "cudnn_allow_tf32": bool(torch.backends.cudnn.allow_tf32),
        "cudnn_benchmark": bool(torch.backends.cudnn.benchmark),
    }
    if backbone == "uni2":
        model = source.load_uni2(device)
    else:
        model = load_module("cell_frozen_phikon_src", PHIKON_SOURCE).load_phikon(device)
    model.eval()
    if any(".lora_" in name for name, _ in model.named_parameters()):
        raise RuntimeError("the frozen bank must have no LoRA parameters")

    cells = C.load_cells(sample)
    cids = cells["cell_id"].to_numpy().astype(str)
    refine = cells["cell_type_refine"].to_numpy().astype(str)
    coarse = cells["coarse11"].to_numpy().astype(str)
    xs = np.rint(cells["he_px"].to_numpy()).astype(np.int64)
    ys = np.rint(cells["he_py"].to_numpy()).astype(np.int64)
    n = len(cells)
    dim = FEATURE_DIM[backbone]
    feats = np.zeros((n, dim), dtype=np.float16)
    valid = np.zeros(n, dtype=bool)
    blank = np.zeros(n, dtype=bool)

    he_path = C.he_path(sample)
    zarr_store = tifffile.imread(str(he_path), aszarr=True, level=0)
    image = zarr.open(zarr_store, mode="r")
    height, width = int(image.shape[0]), int(image.shape[1])
    half = int(source.HALF)
    ymin = max(0, int(ys.min()) - half)
    ymax = min(height, int(ys.max()) + half + 1)
    xmin = max(0, int(xs.min()) - half)
    xmax = min(width, int(xs.max()) + half + 1)
    batch_size = int(source.BATCH)
    pending_crops: list[np.ndarray] = []
    pending_indices: list[int] = []

    def flush() -> None:
        nonlocal pending_crops, pending_indices
        if not pending_crops:
            return
        batch = torch.stack([source.TRANSFORM(c) for c in pending_crops]).to(
            device, non_blocking=True
        )
        with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            out = model(batch)
            cls = out if backbone == "uni2" else out.last_hidden_state[:, 0, :]
        block = cls.float().cpu().numpy().astype(np.float16)
        for k, index in enumerate(pending_indices):
            feats[index] = block[k]
            valid[index] = True
        pending_crops, pending_indices = [], []

    try:
        strip_starts = list(range(ymin, ymax, int(source.STRIP_H)))
        for sy in strip_starts:
            ry0 = max(0, sy - int(source.STRIP_PAD))
            ry1 = min(height, sy + int(source.STRIP_H) + int(source.STRIP_PAD))
            data = np.asarray(image[ry0:ry1, xmin:xmax, :])
            bh, bw = data.shape[:2]
            lo = -(1 << 62) if sy == strip_starts[0] else sy
            hi = (1 << 62) if sy == strip_starts[-1] else min(sy + int(source.STRIP_H), ymax)
            for idx in np.flatnonzero((ys >= lo) & (ys < hi)):
                crop = source.make_crop(data, int(ys[idx]) - ry0, int(xs[idx]) - xmin, bh, bw)
                if crop is None:
                    blank[idx] = True
                    continue
                pending_crops.append(crop)
                pending_indices.append(int(idx))
                if len(pending_crops) >= batch_size:
                    flush()
            del data
        flush()
    finally:
        close = getattr(zarr_store, "close", None)
        if callable(close):
            close()

    if int(blank.sum()) + int(valid.sum()) != n:
        raise RuntimeError("blank and extracted cells do not partition the cell set")
    keep = np.flatnonzero(valid)
    if len(keep) < 1:
        raise RuntimeError(f"{sample}: every crop was blank")
    vectors = torch.from_numpy(feats[keep])
    if not bool(torch.isfinite(vectors.float()).all()):
        raise RuntimeError(f"{sample}: frozen {backbone} bank holds a non-finite value")
    ids_out = cids[keep].tolist()
    refine_out = refine[keep].tolist()
    payload = {
        "schema_version": "v8.frozen_cls.v1",
        "features": vectors,
        "cell_ids": ids_out,
        "cell_type": refine_out,
        "coarse11": coarse[keep].tolist(),
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = out_path.with_name(out_path.name + f".tmp.{os.getpid()}")
    torch.save(payload, tmp)
    os.replace(tmp, out_path)

    result = {
        "schema_version": "v8.frozen_cls_manifest.v1",
        "status": "PASS",
        "sample": sample,
        "backbone": backbone,
        "frozen": True,
        "lora": None,
        "dimension": dim,
        "n_cells": int(n),
        "n_rows": int(len(keep)),
        "n_blank": int(blank.sum()),
        "cell_id_ordered_sha256": C.order_sha256(ids_out),
        "identity_sha256": C.identity_sha256(ids_out, refine_out),
        "label_vintage": "labels_v8 (current reannotation)",
        "centroid_convention": "aligned_he he_px/he_py (Cell canonical cell set)",
        "centroid_convention_note": (
            "All feature banks use the corrected aligned H&E canvas and the contracted cell set."
        ),
        "feature_path": str(out_path.resolve()),
        "feature_file_sha256": C.sha256_file(out_path),
        "he_path": str(he_path),
        "contract_sha256": C.contract_sha256(),
        "shared_extractor_sha256": C.sha256_file(SHARED_EXTRACTOR),
        "script_sha256": C.sha256_file(Path(__file__)),
        "tf32_state_at_run": tf32_state,
        "wall_sec": round(time.time() - started, 1),
        "environment": environment,
    }
    C.atomic_json(manifest_path, result)
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--backbone", choices=C.BACKBONES, required=True)
    parser.add_argument("--sample")
    parser.add_argument("--task-id", type=int)
    parser.add_argument("--list", action="store_true")
    args = parser.parse_args()
    todo = target_samples(args.backbone)
    if args.list:
        print(json.dumps({"backbone": args.backbone, "n_missing": len(todo), "samples": todo},
                         indent=2))
        return 0
    if args.sample:
        sample = args.sample
    elif args.task_id is not None:
        if not 0 <= args.task_id < len(todo):
            raise ValueError(f"task id outside the 0..{len(todo) - 1} cohort list")
        sample = todo[args.task_id]
    else:
        raise ValueError("provide --sample, --task-id or --list")
    result = run(args.backbone, sample)
    print(json.dumps(result, indent=2, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
