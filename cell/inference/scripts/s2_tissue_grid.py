#!/usr/bin/env python -u
from __future__ import annotations
import os
PROJECTS_ROOT = os.environ.get("PROJECTS_ROOT", "/data/heid")
from pathlib import Path
_RELEASE = Path(__file__).resolve().parents[3]
import argparse
import json
import sys
import time
from pathlib import Path


_THREADS = int(os.environ.get("SLURM_CPUS_PER_TASK", "16"))

import cv2
import numpy as np
import pandas as pd
import torch
from scipy.spatial import cKDTree

INFERENCE_ROOT = PROJECTS_ROOT + "/cell/inference"
COHORT = os.environ.get("CELL_COHORT", "tcga_pancancer")
CELLS_DIR = os.path.join(INFERENCE_ROOT, COHORT, "data", "cells")
GRID_DIR = os.path.join(INFERENCE_ROOT, COHORT, "data", "tissue_grid")

PC = PROJECTS_ROOT + "/cell"
sys.path.insert(0, str(_RELEASE / "cell/scripts"))
sys.path.insert(0, str(_RELEASE / "cell/inference/scripts"))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

XPX = 0.2125
SCALES = (512, 1024)
BLANK_FRAC = 0.85
BLANK_SUM = 235 * 3
CROP = 224
BATCH = 256
DIM = 1536
BAND_NATIVE = 8192
DECODE_THREADS = 8


def log(message: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {message}", flush=True)


class ScaleState:

    def __init__(self, scale_canvas_px, native_mpp, width, height, bbox_native):
        self.scale = scale_canvas_px
        self.side = int(round(scale_canvas_px * XPX / native_mpp))
        self.stride = max(1, self.side // 2)
        x_lo, y_lo, x_hi, y_hi = bbox_native
        self.xs = np.arange(max(0, int(x_lo) - self.side),
                            min(width, int(x_hi) + self.side) + 1, self.stride)
        self.ys = np.arange(max(0, int(y_lo) - self.side),
                            min(height, int(y_hi) + self.side) + 1, self.stride)
        self.native_mpp = native_mpp
        self.centres: list[tuple[float, float]] = []
        self.embeddings: list[torch.Tensor] = []
        self.pending: list[np.ndarray] = []
        self.pending_centres: list[tuple[float, float]] = []


IMNET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
IMNET_STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)


def normalise_batch(tiles) -> torch.Tensor:
    x = torch.from_numpy(np.stack(tiles)).permute(0, 3, 1, 2).contiguous().float()
    return x.div_(255).sub_(IMNET_MEAN).div_(IMNET_STD)


def flush(state: ScaleState, model, device) -> None:
    if not state.pending:
        return
    batch = normalise_batch(state.pending).to(device, non_blocking=True)


    dev_type = "cuda" if str(device).startswith("cuda") else "cpu"
    with torch.no_grad(), torch.autocast(device_type=dev_type, dtype=torch.bfloat16):
        state.embeddings.append(model(batch).cpu().float())
    state.centres.extend(state.pending_centres)
    state.pending.clear()
    state.pending_centres.clear()


def cut_tiles(state: ScaleState, band, read_y0, band_rows, width, height,
              model, device) -> None:
    side = state.side
    band_h, band_w = band.shape[:2]
    rows = state.ys[(state.ys >= band_rows[0]) & (state.ys < band_rows[1])]
    if not len(rows):
        return
    factor = CROP / side
    small = cv2.resize(band, (int(round(band_w * factor)), int(round(band_h * factor))),
                       interpolation=cv2.INTER_AREA)
    small_h, small_w = small.shape[:2]
    for y0 in rows:
        top = int(round((int(y0) - read_y0) * factor))
        if top < 0 or top + CROP > small_h:
            continue
        strip = small[top:top + CROP]
        for x0 in state.xs:
            x0 = int(x0)
            left = int(round(x0 * factor))
            if left < 0 or left + CROP > small_w:
                continue
            tile = strip[:, left:left + CROP]
            if (tile.sum(2, dtype=np.uint16) >= BLANK_SUM).mean() > BLANK_FRAC:
                continue
            state.pending.append(np.ascontiguousarray(tile))
            state.pending_centres.append(((x0 + side / 2.0) * state.native_mpp / XPX,
                                          (y0 + side / 2.0) * state.native_mpp / XPX))
            if len(state.pending) >= BATCH:
                flush(state, model, device)


def run(slide: str, row: pd.Series) -> None:
    out_path = os.path.join(GRID_DIR, f"{slide}.npz")
    if os.path.exists(out_path):
        log(f"exists -> skip ({slide})")
        return

    import extract_lora_pc as E
    import slide_canvas as L
    cv2.setNumThreads(_THREADS)
    torch.set_num_threads(_THREADS)

    cells = pd.read_parquet(os.path.join(CELLS_DIR, f"{slide}.parquet"))
    x_canvas = cells["x_um"].to_numpy(np.float64) / XPX
    y_canvas = cells["y_um"].to_numpy(np.float64) / XPX
    native_mpp = float(row.native_mpp)

    device = torch.device(os.environ.get("CELL_DEVICE", "cuda"))
    model = E.load_uni2(device)
    model.eval()

    from fast_slide import FastSlide
    canvas = FastSlide(str(row.svs), n_threads=DECODE_THREADS)
    height, width = canvas.H, canvas.W
    bbox_native = (cells["x_px"].min(), cells["y_px"].min(),
                   cells["x_px"].max(), cells["y_px"].max())
    states = [ScaleState(scale, native_mpp, width, height, bbox_native) for scale in SCALES]
    max_side = max(state.side for state in states)

    started = time.time()
    y_start = min(int(state.ys[0]) for state in states if len(state.ys))
    y_stop = max(int(state.ys[-1]) for state in states if len(state.ys))
    n_bands = 0
    for band_start in range(y_start, y_stop + 1, BAND_NATIVE):
        read_y0 = band_start
        read_y1 = min(height, band_start + BAND_NATIVE + max_side)
        if read_y1 <= read_y0:
            continue
        band = canvas.read(read_y0, 0, read_y1, width)
        n_bands += 1
        for state in states:
            cut_tiles(state, band, read_y0, (band_start, band_start + BAND_NATIVE),
                      width, height, model, device)
        del band
        log(f"  band {read_y0}-{read_y1}: tiles so far "
            f"{[len(s.centres) + len(s.pending) for s in states]} "
            f"({time.time() - started:.0f}s)")
    for state in states:
        flush(state, model, device)
    canvas.close()


    payload, counts = {}, {}
    for state in states:
        counts[str(state.scale)] = len(state.centres)
        if not state.centres:
            raise RuntimeError(f"{slide}: scale {state.scale} produced no non-blank tile")
        embeddings = torch.cat(state.embeddings, 0).numpy().astype(np.float16)
        _, nearest = cKDTree(np.asarray(state.centres, np.float64)).query(
            np.column_stack([x_canvas, y_canvas]), k=1)
        if embeddings.shape[1] != DIM:
            raise RuntimeError(f"{slide}: scale {state.scale} embedding width "
                               f"{embeddings.shape[1]} != {DIM}")
        payload[f"emb_{state.scale}"] = embeddings
        payload[f"idx_{state.scale}"] = nearest.astype(np.uint32)

    os.makedirs(GRID_DIR, exist_ok=True)
    temporary = f"{out_path}.tmp{os.getpid()}.npz"
    np.savez_compressed(temporary, cell_id=cells["cell_id"].to_numpy().astype("U40"),
                        scales=np.array(SCALES, np.int32), **payload)
    os.replace(temporary, out_path)
    with open(os.path.join(GRID_DIR, f"{slide}.meta.json"), "w") as handle:
        json.dump({"slide": slide, "n_cells": int(len(cells)), "tiles": counts,
                   "scales_canvas_px": list(SCALES), "canvas_um_per_px": XPX,
                   "backbone": "UNI2-h frozen, no LoRA", "dim": DIM * len(SCALES),
                   "bands_decoded": n_bands, "device": str(device), "precision": "bfloat16",
                   "seconds": round(time.time() - started, 1)}, handle, indent=2)
    log(f"DONE {slide}: {len(cells):,} cells, tiles {counts}, {n_bands} bands, "
        f"{time.time() - started:.0f}s")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest",
                        default=os.path.join(INFERENCE_ROOT, "configs", COHORT, "run_manifest.tsv"))
    parser.add_argument("--slide")
    parser.add_argument("--shard", type=int, default=0)
    parser.add_argument("--n-shards", type=int, default=1)
    parser.add_argument("--device", default=None,
                        help="cpu is a fallback when no GPU is available; default requires CUDA and hard-exits without one")
    args = parser.parse_args()

    if args.device:
        os.environ["CELL_DEVICE"] = args.device
    if os.environ.get("CELL_DEVICE", "cuda").startswith("cuda"):


        if not torch.cuda.is_available():
            sys.exit("FATAL: a CUDA device is required.")
        log(f"gpu={torch.cuda.get_device_name(0)}")
    else:
        threads = int(os.environ.get("SLURM_CPUS_PER_TASK", "16"))
        torch.set_num_threads(threads)
        log(f"cpu fallback mode, threads {torch.get_num_threads()}")

    table = pd.read_csv(args.manifest, sep="\t").set_index("slide")
    slides = [args.slide] if args.slide else list(table.index)[args.shard::args.n_shards]
    for slide in slides:
        if not os.path.exists(os.path.join(CELLS_DIR, f"{slide}.parquet")):
            log(f"[wait] {slide}: no cells yet")
            continue
        try:
            run(slide, table.loc[slide])
        except Exception as exc:
            print(f"[FAIL] {slide}: {type(exc).__name__}: {exc}", flush=True)


if __name__ == "__main__":
    main()
