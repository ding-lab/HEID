#!/usr/bin/env python3

from __future__ import annotations
from pathlib import Path
_RELEASE = Path(__file__).resolve().parents[2]

import argparse
import fcntl
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
READOUT_SOURCE = _RELEASE / "cell/scripts/readout_lib.py"
CONTRACT_SOURCE = _RELEASE / "cell/scripts/feature_contract.py"

FEATURE_DIM = {"uni2": 1536, "phikon": 1024}
GRID = C.GRID
PATCH_PX = 14
PREFIX_TOKENS = 9
N_PATCH = GRID * GRID
RING_EDGES = ((0.0, 21.0), (21.0, 42.0), (42.0, 70.0), (70.0, 112.0), (112.0, np.inf))
RING_EXPECTED = (4, 28, 48, 128, 48)
RING_NAMES = C.RING_NAMES
POLY_NAMES = C.POLY_NAMES
WMAP_SCALE = 1.0 / 255.0
EXPECTED_GPU = "NVIDIA H100 80GB HBM3"

MANIFEST_DIRS = {
    "uni2": C.MANIFEST_ROOT / "features/uni2",
    "phikon": C.MANIFEST_ROOT / "features/phikon",
}
LOCK_DIR = C.LOG_ROOT / "locks"


def variants_for(backbone: str) -> tuple[str, ...]:
    return C.VARIANTS if backbone == "uni2" else ("cls",)


def feature_path(backbone: str, variant: str, role: str, fold: int, sample: str) -> Path:
    if backbone == "uni2":
        return C.uni2_feature_path(variant, role, fold, sample)
    if variant != "cls":
        raise ValueError("the phikon bank holds only the cls variant")
    return C.phikon_feature_path(role, fold, sample)


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def ring_masks() -> dict[str, np.ndarray]:
    offset = np.arange(GRID, dtype=np.float64) * PATCH_PX + PATCH_PX / 2.0 - 112.0
    radius = np.sqrt(offset[:, None] ** 2 + offset[None, :] ** 2)
    masks = {}
    for name, (lo, hi), expected in zip(RING_NAMES, RING_EDGES, RING_EXPECTED, strict=True):
        mask = ((radius >= lo) & (radius < hi)).astype(np.float32)
        if int(mask.sum()) != expected:
            raise RuntimeError(f"ring {name} holds {int(mask.sum())} patches, expected {expected}")
        masks[name] = mask
    total = int(sum(m.sum() for m in masks.values()))
    if total != N_PATCH:
        raise RuntimeError(f"ring bases cover {total} patches, expected {N_PATCH}")
    return masks


def load_wmaps(sample: str, cids: np.ndarray) -> dict[str, dict[str, Any]]:
    out = {}
    for name in POLY_NAMES:
        path = C.wmap_path(name, sample)
        with np.load(path) as handle:
            cell_id = handle["cell_id"].astype(str)
            w16 = handle["w16"]
            has_poly = handle["has_poly"]
        if w16.dtype != np.uint8 or w16.shape != (len(cids), GRID, GRID):
            raise RuntimeError(f"weight map shape/dtype differs: {path} {w16.shape} {w16.dtype}")
        if not np.array_equal(cell_id, cids):
            raise RuntimeError(f"weight map cell order differs from the cell set: {path}")
        out[name] = {
            "w16": w16,
            "path": str(path),
            "sha256": C.sha256_file(path),
            "n_cells_without_polygon": int((~has_poly).sum()),
        }
    return out


def load_adapters(backbone: str) -> dict[int, dict[str, Any]]:
    out: dict[int, dict[str, Any]] = {}
    for fold in range(5):
        path = C.adapter_path(backbone, fold)
        manifest_path = path.with_suffix(".manifest.json")
        if not path.is_file() or not manifest_path.is_file():
            raise FileNotFoundError(f"adapter or manifest absent: {path}")
        manifest = json.loads(manifest_path.read_text())
        if manifest.get("status") != "PASS" or int(manifest.get("outer_fold", -1)) != fold:
            raise RuntimeError(f"adapter manifest differs for {path}")
        if manifest.get("backbone") != backbone or manifest.get("arm") != C.ARM:
            raise RuntimeError(f"adapter identity differs for {path}")
        C.assert_fold_binding(manifest.get("fold_fingerprint"), fold, f"adapter {backbone} fold{fold}")
        checkpoint_sha = C.sha256_file(path)
        if manifest.get("checkpoint_sha256") != checkpoint_sha:
            raise RuntimeError(f"adapter checkpoint hash drift: {path}")
        expected = C.fold_fingerprint(fold)
        observed = manifest.get("fold_fingerprint", {})
        mismatch = {k: (v, observed.get(k)) for k, v in expected.items() if observed.get(k) != v}
        if mismatch:
            raise RuntimeError(f"adapter {path} fold fingerprint differs: {mismatch}")

        C.assert_outside_fold(manifest["train_samples"], fold, f"adapter {backbone} fold{fold}")
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
        if (
            checkpoint.get("arm") != C.ARM
            or checkpoint.get("backbone") != backbone
            or int(checkpoint.get("fold", -1)) != fold
            or int(checkpoint.get("saved_epoch", -1)) != int(manifest["saved_epoch"])
        ):
            raise RuntimeError(f"adapter checkpoint identity differs: {path}")
        if checkpoint.get("fingerprint_sha256") != manifest.get("fingerprint_sha256"):
            raise RuntimeError(f"adapter fingerprint drift between checkpoint and manifest: {path}")
        out[fold] = {
            "path": str(path),
            "sha256": checkpoint_sha,
            "lora_state_dict": checkpoint["lora_state_dict"],
            "fingerprint_sha256": manifest["fingerprint_sha256"],
            "train_sample_sha256": manifest["train_sample_sha256"],
        }
    return out


def route_positions(sample: str, cids: np.ndarray) -> tuple[dict[int, np.ndarray], dict[str, Any]]:
    fold_own = C.own_fold(sample)
    n = len(cids)
    index = {cell: i for i, cell in enumerate(cids.tolist())}
    positions: dict[int, np.ndarray] = {}
    selection: dict[str, Any] = {}
    for fold in range(5):
        if fold == fold_own:


            positions[fold] = np.arange(n, dtype=np.int64)
            selection[str(fold)] = {
                "role": "oof",
                "n_routed": int(n),
                "source": "every labelled cell of the slide",
            }
            continue


        C.assert_outside_fold([sample], fold, f"train route fold{fold}")
        manifest_path = C.head_selection_manifest(fold)
        manifest = json.loads(manifest_path.read_text())
        if manifest.get("status") != "PASS" or int(manifest.get("fold", -1)) != fold:
            raise RuntimeError(f"head selection manifest differs for fold {fold}")
        C.assert_fold_binding(
            manifest.get("fold_fingerprint"), fold, f"head selection fold{fold}"
        )
        srec = manifest.get("samples", {}).get(sample)
        if not isinstance(srec, dict):
            raise RuntimeError(f"{sample} is absent from the fold {fold} TRAIN selection")
        artifact = C.head_selection_artifact(fold, sample)
        if C.sha256_file(artifact) != srec["sha256"]:
            raise RuntimeError(f"head selection artifact hash drift: {artifact}")
        selected = np.load(artifact, allow_pickle=False).astype(str)
        if len(selected) != int(srec["n_cells"]):
            raise RuntimeError(f"head selection size differs for {sample}/fold{fold}")
        missing = [c for c in selected.tolist() if c not in index]
        if missing:


            raise RuntimeError(
                f"{sample}/fold{fold}: {len(missing)} selected cells are absent from the "
                f"canonical cell set, e.g. {missing[:5]}"
            )
        kept = np.asarray(sorted(index[c] for c in selected.tolist()), dtype=np.int64)
        if len(kept) < 1:
            raise RuntimeError(f"no fold {fold} training cell survives for {sample}")
        positions[fold] = kept
        selection[str(fold)] = {
            "role": "train",
            "n_routed": int(len(kept)),
            "head_selection_manifest_sha256": C.sha256_file(manifest_path),
            "head_selection_artifact_sha256": srec["sha256"],
        }
    return positions, selection


def already_complete(backbone: str, sample: str, task_identity: str) -> dict[str, Any] | None:
    path = MANIFEST_DIRS[backbone] / f"{sample}.json"
    if not path.is_file():
        return None
    manifest = json.loads(path.read_text())
    if manifest.get("status") != "PASS":
        raise RuntimeError(f"existing feature manifest is not PASS: {path}")
    if manifest.get("feature_task_identity_sha256") != task_identity:
        raise RuntimeError(f"existing feature manifest belongs to another task identity: {path}")
    outputs = manifest.get("outputs", [])
    expected = len(variants_for(backbone)) * 5
    if len(outputs) != expected:
        raise RuntimeError(f"existing manifest holds {len(outputs)} outputs, expected {expected}")
    for rec in outputs:


        target = feature_path(
            backbone, rec["variant"], rec["role"], int(rec["adapter_fold"]), sample
        )
        if Path(rec["path"]).name != target.name or not target.is_file():
            raise RuntimeError(f"manifest output is absent: {target}")
        if target.stat().st_size != int(rec["size_bytes"]):
            raise RuntimeError(f"manifest output size differs: {target}")
    return {**manifest, "status": "ALREADY_COMPLETE"}


def run(backbone: str, sample: str, expected_gpu: str = EXPECTED_GPU) -> dict[str, Any]:
    started_all = time.time()
    environment = C.require_usable_gpu(expected_gpu)
    variants = variants_for(backbone)
    dim = FEATURE_DIM[backbone]
    rec = C.record(sample)
    fold_own = C.own_fold(sample)

    adapters = load_adapters(backbone)
    cells = C.load_cells(sample)
    cids = cells["cell_id"].to_numpy().astype(str)
    refine = cells["cell_type_refine"].to_numpy().astype(str)
    coarse = cells["coarse11"].to_numpy().astype(str)
    xs = np.rint(cells["he_px"].to_numpy()).astype(np.int64)
    ys = np.rint(cells["he_py"].to_numpy()).astype(np.int64)
    n = len(cells)
    if n < 1 or len(np.unique(cids)) != n:
        raise RuntimeError(f"{sample}: cell set is empty or has duplicate cell_id")
    cell_order_sha = C.order_sha256(cids.tolist())

    fold_positions, route_record = route_positions(sample, cids)
    ring = ring_masks() if backbone == "uni2" else {}
    wmaps = load_wmaps(sample, cids) if backbone == "uni2" else {}
    he = C.he_path(sample)

    task_contract = {
        "schema_version": "v8.feature_task.v1",
        "sample": sample,
        "cancer": str(rec["cancer"]),
        "patient": str(rec["patient"]),
        "canonical_fold_key": C.CANONICAL_FOLD_KEY,
        "own_outer_fold": fold_own,
        "arm": C.ARM,
        "backbone": backbone,
        "variants": list(variants),
        "canvas": "aligned_he he_px/he_py on the level-0 aligned image",
        "crop_window": "[rint(he_px) - 112, rint(he_px) + 112)",
        "cell_set": "aligned-cell training table INNER JOIN labels_v8, cell_id ascending",
        "label_column": "cell_type_refine (refine17); coarse11 carried alongside",
        "n_cells": int(n),
        "cell_id_ordered_sha256": cell_order_sha,
        "contract_sha256": C.contract_sha256(),
        "v9_training_table": str(rec["train_parquet"]),
        "labels_parquet_sha256": C.sha256_file(C.labels_path(sample)),
        "he_path": str(he),
        "he_size_bytes": he.stat().st_size,
        "wmap_sha256": {name: wmaps[name]["sha256"] for name in wmaps},
        "adapter_sha256": {str(f): adapters[f]["sha256"] for f in range(5)},
        "adapter_fingerprint_sha256": {str(f): adapters[f]["fingerprint_sha256"] for f in range(5)},
        "inference_preprocessing": "native_rgb_imagenet_v1",
        "routing": (
            "for head fold k every row comes from adapter k; adapter k excluded fold k; "
            "the slide's own fold serves the oof route"
        ),
        "routes": route_record,
        "shared_extractor_sha256": C.sha256_file(SHARED_EXTRACTOR),
        "readout_source_sha256": C.sha256_file(READOUT_SOURCE),
        "feature_contract_sha256": C.sha256_file(CONTRACT_SOURCE),
        "extractor_sha256": C.sha256_file(Path(__file__)),
    }
    task_identity = C.canonical_sha256(task_contract)

    LOCK_DIR.mkdir(parents=True, exist_ok=True)
    lock_handle = (LOCK_DIR / f"v8_{backbone}_{sample}.lock").open("a+")
    try:
        fcntl.flock(lock_handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError as exc:
        lock_handle.close()
        raise RuntimeError(f"another Cell task holds this slide's lock: {sample}") from exc
    try:
        complete = already_complete(backbone, sample, task_identity)
        if complete is not None:
            print(json.dumps({"sample": sample, "backbone": backbone,
                              "status": "ALREADY_COMPLETE",
                              "n_outputs": len(complete["outputs"])}, indent=2), flush=True)
            return complete

        source = load_module(f"v8_extract_{backbone}", SHARED_EXTRACTOR)
        device = torch.device("cuda")
        tf32_state = {
            "cuda_matmul_allow_tf32": bool(torch.backends.cuda.matmul.allow_tf32),
            "cudnn_allow_tf32": bool(torch.backends.cudnn.allow_tf32),
            "cudnn_benchmark": bool(torch.backends.cudnn.benchmark),
            "float32_matmul_precision": torch.get_float32_matmul_precision(),
        }
        if backbone == "uni2":
            model = source.load_uni2(device)
            source.inject_lora(model, source.LORA_RANK, source.LORA_ALPHA)
        else:
            phikon = load_module("v8_phikon", _RELEASE / "cell/scripts/train_lora_phikon.py")
            model = phikon.load_phikon(device)
            phikon.inject_lora(model, source.LORA_RANK, source.LORA_ALPHA)
        model.eval()
        lora_param_names = {name for name, _ in model.named_parameters() if ".lora_" in name}
        for fold in range(5):
            keys = set(adapters[fold]["lora_state_dict"])
            if keys != lora_param_names:
                raise RuntimeError(
                    f"fold {fold} adapter covers {len(keys)} of the model's "
                    f"{len(lora_param_names)} LoRA parameters; rotation would leave stale weights"
                )

        ring_gpu = {name: torch.from_numpy(mask).to(device) for name, mask in ring.items()}
        store_arrays: dict[tuple[int, str], np.ndarray] = {}
        wsum_arrays: dict[tuple[int, str], np.ndarray] = {}
        for fold in range(5):
            size = len(fold_positions[fold])
            for variant in variants:
                store_arrays[(fold, variant)] = np.zeros((size, dim), dtype=np.float16)
            for variant in (POLY_NAMES if backbone == "uni2" else ()):
                wsum_arrays[(fold, variant)] = np.zeros(size, dtype=np.float32)

        local_index: dict[int, np.ndarray] = {}
        selected_mask: dict[int, np.ndarray] = {}
        valid: dict[int, np.ndarray] = {}
        for fold in range(5):
            lookup = np.full(n, -1, dtype=np.int64)
            lookup[fold_positions[fold]] = np.arange(len(fold_positions[fold]), dtype=np.int64)
            local_index[fold] = lookup
            mask = np.zeros(n, dtype=bool)
            mask[fold_positions[fold]] = True
            selected_mask[fold] = mask
            valid[fold] = np.zeros(len(fold_positions[fold]), dtype=bool)

        zarr_store = tifffile.imread(str(he), aszarr=True, level=0)
        image = zarr.open(zarr_store, mode="r")
        height, width = int(image.shape[0]), int(image.shape[1])
        half = int(source.HALF)
        ymin = max(0, int(ys.min()) - half)
        ymax = min(height, int(ys.max()) + half + 1)
        xmin = max(0, int(xs.min()) - half)
        xmax = min(width, int(xs.max()) + half + 1)
        batch_size = int(source.BATCH)
        blank_mask = np.zeros(n, dtype=bool)
        pending_crops: list[np.ndarray] = []
        pending_indices: list[int] = []
        control: dict[str, Any] = {"pending": True}
        stats: dict[str, Any] = {"n_batches": 0, "n_forwards": 0, "tok_shape": None}
        started = time.time()

        def forward(sub):
            if backbone != "uni2":
                out = model(sub)
                return out.last_hidden_state[:, 0, :], None
            tok = model.forward_features(sub)
            cls = model.forward_head(tok, pre_logits=True)
            if stats["tok_shape"] is None:
                stats["tok_shape"] = list(tok.shape)
            if tok.shape[1] != PREFIX_TOKENS + N_PATCH or tok.shape[2] != dim:
                raise RuntimeError(f"token layout differs: {tuple(tok.shape)}")
            b = int(sub.shape[0])
            patches = tok[:, PREFIX_TOKENS:, :].float().reshape(b, GRID, GRID, dim)
            return cls, patches

        def flush() -> None:
            nonlocal pending_crops, pending_indices
            if not pending_crops:
                return
            batch = torch.stack([source.TRANSFORM(crop) for crop in pending_crops]).to(
                device, non_blocking=True
            )
            global_indices = np.asarray(pending_indices, dtype=np.int64)
            stats["n_batches"] += 1
            for fold in range(5):
                in_fold = np.flatnonzero(selected_mask[fold][global_indices])
                if not len(in_fold):
                    continue
                source.load_lora_sd(model, adapters[fold]["lora_state_dict"])
                stats["n_forwards"] += 1
                chosen_global = global_indices[in_fold]
                chosen_local = local_index[fold][chosen_global]
                with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                    sub = batch[torch.as_tensor(in_fold, device=device)]
                    cls, patches = forward(sub)
                    pooled_out: dict[str, torch.Tensor] = {}
                    total_out: dict[str, torch.Tensor] = {}
                    if patches is not None:
                        b = int(sub.shape[0])
                        for variant in RING_NAMES + POLY_NAMES:
                            if variant in ring_gpu:
                                weight = ring_gpu[variant].unsqueeze(0).expand(b, GRID, GRID)
                            else:
                                picked = wmaps[variant]["w16"][chosen_global]
                                weight = torch.from_numpy(
                                    picked.astype(np.float32) * WMAP_SCALE
                                ).to(device)
                            total = weight.sum(dim=(1, 2))
                            pooled = torch.einsum("byx,byxd->bd", weight, patches)
                            pooled_out[variant] = pooled / total.clamp(min=1e-6).unsqueeze(1)
                            total_out[variant] = total
                    if control["pending"]:


                        control["batch"] = sub.detach().clone()
                        control["fold"] = fold
                        control["local"] = np.array(chosen_local, copy=True)
                        control["pending"] = False
                store_arrays[(fold, "cls")][chosen_local] = (
                    cls.float().cpu().numpy().astype(np.float16)
                )
                for variant in pooled_out:
                    store_arrays[(fold, variant)][chosen_local] = (
                        pooled_out[variant].cpu().numpy().astype(np.float16)
                    )
                for variant in total_out:
                    if (fold, variant) not in wsum_arrays:


                        continue
                    wsum_arrays[(fold, variant)][chosen_local] = (
                        total_out[variant].cpu().numpy().astype(np.float32)
                    )
                valid[fold][chosen_local] = True
                del cls, patches, pooled_out, total_out
            pending_crops, pending_indices = [], []

        try:


            strip_starts = list(range(ymin, ymax, int(source.STRIP_H)))
            for sy in strip_starts:
                ry0 = max(0, sy - int(source.STRIP_PAD))
                ry1 = min(height, sy + int(source.STRIP_H) + int(source.STRIP_PAD))
                block = np.asarray(image[ry0:ry1, xmin:xmax, :])
                bh, bw = block.shape[:2]
                lo = -(1 << 62) if sy == strip_starts[0] else sy
                hi = ((1 << 62) if sy == strip_starts[-1]
                      else min(sy + int(source.STRIP_H), ymax))
                selected = np.flatnonzero((ys >= lo) & (ys < hi))
                for idx in selected:
                    crop = source.make_crop(
                        block, int(ys[idx]) - ry0, int(xs[idx]) - xmin, bh, bw
                    )
                    if crop is None:
                        blank_mask[idx] = True
                        continue
                    pending_crops.append(crop)
                    pending_indices.append(int(idx))
                    if len(pending_crops) >= batch_size:
                        flush()
                del block
            flush()
        finally:
            close = getattr(zarr_store, "close", None)
            if callable(close):
                close()
        wall_forward = time.time() - started

        if int(blank_mask.sum()) + int(valid[fold_own].sum()) != n:
            raise RuntimeError("blank and extracted cells do not partition the oof cell set")
        for fold in range(5):
            routed = fold_positions[fold]
            expected = int(len(routed) - blank_mask[routed].sum())
            if int(valid[fold].sum()) != expected:
                raise RuntimeError(
                    f"fold {fold}: {int(valid[fold].sum())} extracted rows, expected {expected}"
                )


        control_batch = control.pop("batch", None)
        if control_batch is None:
            raise RuntimeError("no batch was captured for the direct-call control")
        control_fold = int(control.pop("fold"))
        control_local = control.pop("local")
        source.load_lora_sd(model, adapters[control_fold]["lora_state_dict"])
        with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            if backbone == "uni2":
                tok_replay = model.forward_features(control_batch)
                split_replay = model.forward_head(tok_replay, pre_logits=True)
                direct_replay = model(control_batch)
            else:


                split_replay = model(control_batch).last_hidden_state[:, 0, :]
                direct_replay = None
        replay_fp16 = torch.from_numpy(split_replay.float().cpu().numpy().astype(np.float16))
        stored_fp16 = torch.from_numpy(store_arrays[(control_fold, "cls")][control_local])
        split_applicable = direct_replay is not None
        gate = {
            "n_crops": int(control_batch.shape[0]),
            "adapter_fold": control_fold,
            "dtype_split": str(split_replay.dtype),
            "split_vs_direct_applicable": split_applicable,
            "split_vs_direct_why_not": (
                None if split_applicable
                else "phikon's CLS is read from a single forward; there is no split to check"
            ),
            "dtype_direct": str(direct_replay.dtype) if split_applicable else None,
            "same_dtype": (direct_replay.dtype == split_replay.dtype) if split_applicable else None,
            "split_vs_direct_torch_equal": (
                bool(torch.equal(direct_replay, split_replay)) if split_applicable else None
            ),
            "split_vs_direct_max_abs_delta": (
                float((direct_replay.to(torch.float64) - split_replay.to(torch.float64)).abs().max())
                if split_applicable else None
            ),
            "replay_vs_stored_torch_equal": bool(
                torch.equal(replay_fp16.view(torch.int16), stored_fp16.view(torch.int16))
            ),
            "replay_vs_stored_max_abs_delta": float(
                (replay_fp16.to(torch.float64) - stored_fp16.to(torch.float64)).abs().max()
            ),
        }
        gate["passed"] = bool(
            gate["replay_vs_stored_torch_equal"]
            and (
                not split_applicable
                or (gate["same_dtype"] and gate["split_vs_direct_torch_equal"])
            )
        )
        del control_batch, split_replay, direct_replay
        if not gate["passed"]:
            raise RuntimeError(f"in-run CLS identity gate failed: {gate}")
        print(json.dumps({"sample": sample, "backbone": backbone, "identity_gate": gate},
                         indent=2), flush=True)

        outputs = []
        for fold in range(5):
            role = "oof" if fold == fold_own else "train"
            local = np.flatnonzero(valid[fold])
            if len(local) < 1:
                raise RuntimeError(f"every selected crop was blank for fold {fold}")
            rows_here = fold_positions[fold][local]
            ids_out = cids[rows_here].tolist()
            refine_out = refine[rows_here].tolist()
            coarse_out = coarse[rows_here].tolist()
            identity = C.identity_sha256(ids_out, refine_out)
            n_blank = int(blank_mask[fold_positions[fold]].sum())
            whole = len(local) == len(fold_positions[fold])
            for variant in variants:
                block = store_arrays[(fold, variant)]
                vectors = torch.from_numpy(block if whole else block[local])
                if not bool(torch.isfinite(vectors.float()).all()):
                    raise RuntimeError(f"{variant}/fold{fold} holds a non-finite value")
                payload = {
                    "schema_version": "v8.feature.v1",
                    "features": vectors,
                    "cell_ids": ids_out,
                    "cell_type": refine_out,
                    "coarse11": coarse_out,
                    "contract": {
                        **task_contract,
                        "variant": variant,
                        "variant_kind": (
                            "cls" if variant == "cls"
                            else "concentric_ring" if variant in ring else "polygon_blur"
                        ),
                        "adapter_fold": fold,
                        "role": role,
                        "n_rows": int(len(local)),
                        "n_blank": n_blank,
                        "identity_sha256": identity,
                        "adapter_checkpoint_sha256": adapters[fold]["sha256"],
                        "adapter_fingerprint_sha256": adapters[fold]["fingerprint_sha256"],
                        "encoder_excluded_fold": fold,
                        "slide_fold_he_safe": fold_own,


                        "outer_clean": bool(
                            (fold != fold_own) if role == "train" else (fold == fold_own)
                        ),
                        "outer_clean_basis": (
                            "train route: slide_fold_he_safe != encoder_excluded_fold; "
                            "oof route: slide_fold_he_safe == encoder_excluded_fold"
                        ),
                        "normalised": "m_v = einsum(w, P) / clamp(sum(w), min=1e-6)",
                        "patch_grid": [GRID, GRID],
                        "prefix_tokens": PREFIX_TOKENS,
                        "identity_gate": gate,
                    },
                }
                if variant in ring:
                    payload["contract"]["n_patches_in_ring"] = int(ring[variant].sum())
                if (fold, variant) in wsum_arrays:
                    payload["w_sum"] = torch.from_numpy(wsum_arrays[(fold, variant)][local])
                    payload["contract"]["w_sum_dtype"] = "float32"
                    payload["contract"]["w_sum_meaning"] = (
                        "sum of the 16x16 normalised polygon weights for this cell; the "
                        "denominator of m_v, and 0 when the cell has no usable polygon"
                    )
                    payload["contract"]["wmap_path"] = wmaps[variant]["path"]
                    payload["contract"]["wmap_sha256"] = wmaps[variant]["sha256"]
                target = feature_path(backbone, variant, role, fold, sample)
                target.parent.mkdir(parents=True, exist_ok=True)
                tmp = target.with_name(target.name + f".tmp.{os.getpid()}")
                torch.save(payload, tmp)
                os.replace(tmp, target)
                record_out = {
                    "variant": variant,
                    "role": role,
                    "adapter_fold": fold,
                    "path": str(target),
                    "size_bytes": target.stat().st_size,
                    "n_rows": int(len(local)),
                    "n_blank": n_blank,
                    "identity_sha256": identity,
                }
                if (fold, variant) in wsum_arrays:
                    wsum = wsum_arrays[(fold, variant)][local]
                    record_out["w_sum_p50"] = round(float(np.percentile(wsum, 50)), 6)
                    record_out["w_sum_min"] = round(float(wsum.min()), 6)
                    record_out["n_cells_w_sum_zero"] = int((wsum == 0.0).sum())
                outputs.append(record_out)

        result = {
            "schema_version": "v8.feature_manifest.v1",
            "status": "PASS",
            "sample": sample,
            "backbone": backbone,
            "cancer": str(rec["cancer"]),
            "patient": str(rec["patient"]),
            "own_outer_fold": int(fold_own),
            "canonical_fold_key": C.CANONICAL_FOLD_KEY,
            "feature_task_identity_sha256": task_identity,
            "task_contract": task_contract,
            "n_cells": int(n),
            "n_blank_crops": int(blank_mask.sum()),
            "cell_id_ordered_sha256": cell_order_sha,
            "rows_per_route": {
                f"fold{fold}": {
                    "role": "oof" if fold == fold_own else "train",
                    "n_routed": int(len(fold_positions[fold])),
                    "n_extracted": int(valid[fold].sum()),
                    "encoder_adapter_fold": fold,
                    "encoder_excluded_fold": fold,
                }
                for fold in range(5)
            },
            "routes": route_record,
            "wmaps": {
                name: {
                    "path": wmaps[name]["path"],
                    "sha256": wmaps[name]["sha256"],
                    "n_cells_without_polygon": wmaps[name]["n_cells_without_polygon"],
                }
                for name in wmaps
            },
            "identity_gate": gate,
            "token_shape_observed": stats["tok_shape"],
            "n_batches": stats["n_batches"],
            "n_forwards": stats["n_forwards"],
            "tf32_state_at_run": tf32_state,
            "outputs": outputs,
            "wall_sec_forward": round(wall_forward, 1),
            "wall_sec_total": round(time.time() - started_all, 1),
            "rows_per_sec_forward": round(
                sum(int(valid[f].sum()) for f in range(5)) / max(wall_forward, 1e-6), 2
            ),
            "environment": environment,
        }
        manifest_out = MANIFEST_DIRS[backbone] / f"{sample}.json"
        C.atomic_json(manifest_out, result)
        print(json.dumps({k: v for k, v in result.items()
                          if k not in ("outputs", "task_contract")}, indent=2, sort_keys=True),
              flush=True)
        return result
    finally:
        fcntl.flock(lock_handle.fileno(), fcntl.LOCK_UN)
        lock_handle.close()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--backbone", choices=C.BACKBONES, required=True)
    parser.add_argument("--sample")
    parser.add_argument("--task-id", type=int)
    parser.add_argument("--expected-gpu", default=EXPECTED_GPU,
                        help="GPU name every sample of one feature library must be extracted on")
    args = parser.parse_args()
    if args.sample:
        sample = args.sample
    elif args.task_id is not None:
        allow = C.allowlist()
        if not 0 <= args.task_id < len(allow):
            raise ValueError(f"task id outside the 0..{len(allow) - 1} allowlist")
        sample = allow[args.task_id]
    else:
        raise ValueError("provide --sample or --task-id")
    result = run(args.backbone, sample, args.expected_gpu)
    return 0 if result["status"] in ("PASS", "ALREADY_COMPLETE") else 1


if __name__ == "__main__":
    sys.exit(main())
