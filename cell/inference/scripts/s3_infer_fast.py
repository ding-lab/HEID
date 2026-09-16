#!/usr/bin/env python -u
from __future__ import annotations
import os
PROJECTS_ROOT = os.environ.get("PROJECTS_ROOT", "/data/heid")
from pathlib import Path
_RELEASE = Path(__file__).resolve().parents[3]
import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path

for _var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
             "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_var, "1")

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

INFERENCE_ROOT = PROJECTS_ROOT + "/cell/inference"
DETECTION_ROOT = PROJECTS_ROOT + "/cell/inference/inputs"
COHORT = os.environ.get("CELL_COHORT", "tcga_pancancer")
CELLS_DIR = os.path.join(INFERENCE_ROOT, COHORT, "data", "cells")
WMAP_DIR = os.path.join(INFERENCE_ROOT, COHORT, "data", "wmaps")
GRID_DIR = os.path.join(INFERENCE_ROOT, COHORT, "data", "tissue_grid")
PRED_DIR = os.path.join(INFERENCE_ROOT, COHORT, "data", "predictions")

PC = Path(PROJECTS_ROOT + "/cell")
CELL_ROOT = PC
HEAD_DIR = CELL_ROOT / "outputs/models/head/cls_sigma3"
ADAPTER_DIR = CELL_ROOT / "outputs/models/adapters/raw_rgb"
HEAD_COMPATIBILITY = _RELEASE / "cell/configs/head_compatibility.json"

sys.path.insert(0, str(_RELEASE / "cell/scripts"))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, str(_RELEASE / "cell/inference/scripts"))
sys.path.insert(0, str(_RELEASE / "cell/inference/scripts"))

from band_reader import BandCachedSlide
from adapter_contract import load_inference_adapter

XPX = 0.2125
CROP = 224
HALF = CROP // 2
GRID_TOKENS = 16
PREFIX_TOKENS = 9
UNI2_DIM = 1536
PHIKON_DIM = 1024
GRID_DIM = 3072
FOLD = 0
WMAP_SCALE = 1.0 / 255.0
LORA_RANK = 16
LORA_ALPHA = 32
BATCH = 256
BAND_ROWS = 8192
COMPILE = os.environ.get("CELL_COMPILE", "1") not in ("0", "", "false")
FORCE_CUDNN_ATTENTION = False


CROP_WORKERS = 12


def log(message: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {message}", flush=True)


class BandCropDataset(torch.utils.data.Dataset):

    def __init__(self, view, x_canvas, y_canvas, rows, transform, white_thresh, blank_frac):
        self.view = view
        self.x_canvas = x_canvas
        self.y_canvas = y_canvas
        self.rows = rows
        self.transform = transform
        self.white_thresh = white_thresh
        self.blank_frac = blank_frac

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, i):
        index = int(self.rows[i])
        cx = int(np.rint(self.x_canvas[index])) - HALF
        cy = int(np.rint(self.y_canvas[index])) - HALF
        crop = self.view[cy:cy + CROP, cx:cx + CROP]
        if crop.shape[:2] != (CROP, CROP) or np.mean(crop > self.white_thresh) > self.blank_frac:
            return torch.zeros(3, CROP, CROP), index, False
        return self.transform(crop), index, True


class GatedDualBackboneContext(nn.Module):

    def __init__(self, *, uni2_dim, phikon_dim, grid_dim, projection_dim=256, hidden_dim=512,
                 n_identity=11, n_lineage=6, dropout=0.2, cell_feature_dropout=0.05,
                 grid_feature_dropout=0.15, necrosis_hidden_dim=512, necrosis_dropout=0.2):
        super().__init__()
        self.uni2 = nn.Sequential(nn.LayerNorm(uni2_dim), nn.Linear(uni2_dim, projection_dim),
                                  nn.GELU())
        self.phikon = nn.Sequential(nn.LayerNorm(phikon_dim),
                                    nn.Linear(phikon_dim, projection_dim), nn.GELU())
        self.gate = nn.Linear(projection_dim * 2, projection_dim)
        self.grid = nn.Sequential(nn.LayerNorm(grid_dim), nn.Linear(grid_dim, projection_dim),
                                  nn.GELU())
        self.trunk = nn.Sequential(nn.Linear(projection_dim * 2, hidden_dim),
                                   nn.LayerNorm(hidden_dim), nn.GELU(), nn.Dropout(dropout))
        self.identity_head = nn.Linear(hidden_dim, n_identity)
        self.lineage_head = nn.Linear(hidden_dim, n_lineage)
        self.necrosis_head = nn.Sequential(
            nn.LayerNorm(grid_dim), nn.Linear(grid_dim, necrosis_hidden_dim), nn.GELU(),
            nn.Dropout(necrosis_dropout), nn.Linear(necrosis_hidden_dim, 1))

    def forward(self, uni2, phikon, grid):
        u = self.uni2(uni2)
        p = self.phikon(phikon)
        gate = torch.sigmoid(self.gate(torch.cat((u, p), dim=-1)))
        cell = gate * u + (1.0 - gate) * p
        context = self.grid(grid)
        hidden = self.trunk(torch.cat((cell, context), dim=-1))
        return self.identity_head(hidden)


def load_head(device):
    path = HEAD_DIR / f"fold{FOLD}.pt"
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    compatibility = json.loads(HEAD_COMPATIBILITY.read_text())
    experiment_id = checkpoint.get("experiment_id")
    if checkpoint.get("arm") != "cls_sigma3" or int(checkpoint.get("fold", -1)) != FOLD:
        raise SystemExit(f"head identity drift at fold {FOLD}")
    if experiment_id not in compatibility["accepted_heads"]:
        raise SystemExit(f"head experiment_id {experiment_id!r} is not an accepted Cell head; see cell/configs/head_compatibility.json")
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    identity = {"experiment_id": experiment_id,
                "schwann_definition": compatibility["accepted_heads"][experiment_id]["schwann_definition"],
                "checkpoint_sha256": digest, "row_set_sha256": checkpoint.get("row_set_sha256"),
                "selection_sha256": checkpoint.get("selection_sha256"), "fold": FOLD}
    head = GatedDualBackboneContext(**dict(checkpoint["model_config"]))
    head.load_state_dict(checkpoint["model_state"], strict=True)
    return head.to(device).eval(), list(checkpoint["identity_classes"]), identity


def load_adapter(backbone: str):
    return load_inference_adapter(ADAPTER_DIR / backbone / f"fold{FOLD}.pt", backbone, FOLD)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--slide", required=True)
    parser.add_argument("--manifest",
                        default=os.path.join(INFERENCE_ROOT, "configs", COHORT, "run_manifest.tsv"))
    parser.add_argument("--batch", type=int, default=BATCH)
    parser.add_argument("--crop-workers", type=int, default=CROP_WORKERS)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        sys.exit("FATAL: no CUDA. The job landed without a GPU (pyxis not stripped at submit).")
    device = torch.device("cuda")
    log(f"gpu={torch.cuda.get_device_name(0)}")

    out_path = os.path.join(PRED_DIR, f"{args.slide}.parquet")
    if os.path.exists(out_path):
        log(f"exists -> skip ({args.slide})")
        return

    import extract_lora_pc as E
    import band_view as FM
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "train_lora_phikon", _RELEASE / "cell/scripts/train_lora_phikon.py")
    phikon_lib = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(phikon_lib)

    table = pd.read_csv(args.manifest, sep="\t").set_index("slide")
    row = table.loc[args.slide]
    native_mpp = float(row.native_mpp)

    cells = pd.read_parquet(os.path.join(CELLS_DIR, f"{args.slide}.parquet"))
    n = len(cells)
    with np.load(os.path.join(WMAP_DIR, f"{args.slide}.npz")) as handle:
        if not np.array_equal(handle["cell_id"].astype(str),
                              cells["cell_id"].to_numpy().astype(str)):
            sys.exit("FATAL: sigma3 map order differs from the cell table")
        w16 = handle["w16"]
        has_poly = handle["has_poly"]
    with np.load(os.path.join(GRID_DIR, f"{args.slide}.npz")) as handle:
        if not np.array_equal(handle["cell_id"].astype(str),
                              cells["cell_id"].to_numpy().astype(str)):
            sys.exit("FATAL: tissue-grid order differs from the cell table")


        scales = handle["scales"].tolist()
        tissue_grid = np.concatenate(
            [handle[f"emb_{sc}"][handle[f"idx_{sc}"]] for sc in scales], axis=1)
    log(f"{args.slide}: {n:,} cells, polygons {has_poly.mean():.4f}, grid {tissue_grid.shape}")

    uni2 = E.load_uni2(device)
    E.inject_lora(uni2, LORA_RANK, LORA_ALPHA)
    uni2.eval()
    phikon = phikon_lib.load_phikon(device)
    phikon_lib.inject_lora(phikon, LORA_RANK, LORA_ALPHA)
    phikon.eval()
    E.load_lora_sd(uni2, load_adapter("uni2"))
    E.load_lora_sd(phikon, load_adapter("phikon"))
    head, classes, head_identity = load_head(device)


    if COMPILE and FORCE_CUDNN_ATTENTION:
        sys.exit("FATAL: torch.compile with a forced cuDNN attention backend produces wrong "
                 "output (argmax agreement 0.17-0.26, accuracy 0.49 -> 0.13) at no speed "
                 "benefit. Enable one or neither.")
    if os.environ.get("CELL_COMPILE_MODE", "default") != "default":
        sys.exit("FATAL: only compile mode 'default' is validated. max-autotune disagrees with "
                 "the uncompiled model on 16-18% of cells.")
    if os.environ.get("CELL_FP8") or os.environ.get("CELL_TOKEN_MERGING"):
        sys.exit("FATAL: FP8 measured SLOWER than bf16 (2.49 vs 1.73 ms/crop); token merging "
                 "r=8 costs 4 points of Schwann recall for 1.23x. Neither is available.")
    if os.environ.get("CELL_SHARE_TILE_FEATURES"):
        sys.exit("FATAL: sharing one tile feature map across nearby cells changes the class of "
                 "37.9% of cells at the grid's typical off-centre distance. Per-cell 224 px "
                 "forward passes are the contract, not an inefficiency.")

    if COMPILE:


        uni2 = torch.compile(uni2, mode="default")
        phikon = torch.compile(phikon, mode="default")
        log("backbones compiled (mode=default)")
    log(f"two backbones and one head ready, fold {FOLD}")

    x_canvas = cells["x_um"].to_numpy(np.float64) / XPX
    y_canvas = cells["y_um"].to_numpy(np.float64) / XPX
    y_native = cells["y_px"].to_numpy(np.float64)


    canvas = BandCachedSlide(str(row.svs), band_rows=BAND_ROWS)
    height, width = canvas.H, canvas.W
    canvas_h = int(np.ceil(height * native_mpp / XPX))
    canvas_w = int(np.ceil(width * native_mpp / XPX))

    probs = np.zeros((n, len(classes)), np.float32)
    scored = np.zeros(n, bool)
    started = time.time()
    done = 0

    for band_y0, band_y1 in FM.bands_for(y_native, height):
        rows = np.flatnonzero((y_native >= band_y0) & (y_native < band_y1))
        if not len(rows):
            continue
        read_y0 = max(0, band_y0 - FM.CROP_HALF_NATIVE_PAD)
        read_y1 = min(height, band_y1 + FM.CROP_HALF_NATIVE_PAD)
        band = canvas.read(read_y0, 0, read_y1, width)
        view = FM.BandView(band, read_y0, native_mpp, canvas_h, canvas_w)

        loader = torch.utils.data.DataLoader(
            BandCropDataset(view, x_canvas, y_canvas, rows, E.TRANSFORM,
                            E.WHITE_THRESH, E.BLANK_FRAC),
            batch_size=args.batch, num_workers=args.crop_workers, shuffle=False,
            pin_memory=True, persistent_workers=False,
            prefetch_factor=4 if args.crop_workers > 0 else None)

        for batch, index_batch, usable in loader:
            usable = usable.numpy().astype(bool)
            if not usable.any():
                continue
            keep = index_batch.numpy()[usable]


            if not usable.all():
                batch = batch[torch.from_numpy(usable)]
            batch = batch.to(device, non_blocking=True)
            weight = torch.from_numpy(w16[keep].astype(np.float32) * WMAP_SCALE).to(device)
            total = weight.sum(dim=(1, 2))
            grid = torch.from_numpy(tissue_grid[keep].astype(np.float32)).to(device)

            with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                tokens = uni2.forward_features(batch)
                cls = uni2.forward_head(tokens, pre_logits=True)
                patches = tokens[:, PREFIX_TOKENS:, :].float().reshape(
                    len(keep), GRID_TOKENS, GRID_TOKENS, UNI2_DIM)
                sigma3 = torch.einsum("byx,byxd->bd", weight, patches)
                sigma3 = sigma3 / total.clamp(min=1e-6).unsqueeze(1)
                uni2_feature = torch.cat((cls.float(), sigma3), dim=1)
                phikon_feature = phikon(batch).last_hidden_state[:, 0, :].float()
                logits = head(uni2_feature, phikon_feature, grid)
                probability = torch.softmax(logits.float(), dim=1)
            probs[keep] = probability.cpu().numpy().astype(np.float32)
            scored[keep] = True
            done += len(keep)
        del band, view
        log(f"  band {band_y0}-{band_y1}: {done:,}/{n:,} "
            f"({done / max(time.time() - started, 1e-6):.0f} cells/s)")
    canvas.close()

    predicted = probs.argmax(axis=1)
    frame = pd.DataFrame({
        "cell_id": cells["cell_id"].to_numpy(),
        "x_um": cells["x_um"].to_numpy(),
        "y_um": cells["y_um"].to_numpy(),
        "scored": scored,
        "pred_class": predicted.astype(np.int16),
        "pred_class_name": [classes[i] for i in predicted],
        "schwann_prob": probs[:, classes.index("Schwann")],
    })
    for index, name in enumerate(classes):
        frame[f"prob_{index}"] = probs[:, index]
    frame.loc[~scored, "pred_class_name"] = pd.NA

    os.makedirs(PRED_DIR, exist_ok=True)
    temporary = f"{out_path}.tmp{os.getpid()}"
    frame.to_parquet(temporary, index=False)
    os.replace(temporary, out_path)

    counts = frame.loc[scored, "pred_class_name"].value_counts()
    meta = {
        "slide": args.slide, "cohort": COHORT, "n_cells": int(n), "n_scored": int(scored.sum()),
        "model": f"Cell cls_sigma3, single fold {FOLD}",
        "fold_rule": "model card 7: one fold matches the reported metric; "
                     "the five-model ensemble was never measured",
        "heads": str(HEAD_DIR), "adapters": str(ADAPTER_DIR), "head_identity": head_identity,
        "decision_layer": "not applied; raw model probabilities are reported",
        "classes": classes, "class_counts": {k: int(v) for k, v in counts.items()},
        "crop": {"px": CROP, "um_per_px": XPX, "stain": "raw RGB, ImageNet normalisation only"},
        "sigma3_polygon": "InstanSeg NUCLEUS contour (training uses Xenium WHOLE-CELL)",
        "seconds": round(time.time() - started, 1),
    }
    with open(os.path.join(PRED_DIR, f"{args.slide}.meta.json"), "w") as handle:
        json.dump(meta, handle, indent=2)


    import resource
    peak_self = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (1024 ** 2)
    peak_child = resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss / (1024 ** 2)
    log(f"DONE {args.slide}: scored {int(scored.sum()):,}/{n:,} in {meta['seconds']:.0f}s"
        f"  peak memory main process {peak_self:.1f} GB + max child process {peak_child:.1f} GB"
        f" (lower bound, requested 256 GB)")
    log(counts.to_string())


if __name__ == "__main__":
    main()
