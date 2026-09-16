#!/usr/bin/env python3

from __future__ import annotations
import os
PROJECTS_ROOT = os.environ.get("PROJECTS_ROOT", "/data/heid")
from pathlib import Path
_RELEASE = Path(__file__).resolve().parents[2]

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import tifffile
import torch
from scipy.spatial import cKDTree
from torchvision import transforms

import contract as C

SCALES = [512, 1024]
BLANK_FRAC = 0.85
BLANK_LEVEL = 235
CROP = 224
BATCH = 256
DIM = 1536

TF = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

OUT_ROOT = C.FEATURE_ROOT / "tissue_grid"
MANIFEST_DIR = C.MANIFEST_ROOT / "tissue_grid"

DEPENDENCY_SHA256 = "42483aa28236094dec7ce2bd88991c3381854a1afc90f4b82f1e5ec38587d4f4"


def target_samples() -> list[str]:
    return list(C.allowlist())


def load_uni2_frozen(device):
    import timm
    from timm.layers import SwiGLUPacked

    paths = [
        Path(PROJECTS_ROOT + "/tools/uni2/pytorch_model.bin"),
        Path(PROJECTS_ROOT + "/tools/uni2/pytorch_model.bin"),
    ]
    weights = next((p for p in paths if p.exists()), None)
    if weights is None:
        raise FileNotFoundError("UNI2 weights not found")
    model = timm.create_model(
        "vit_giant_patch14_224", img_size=224, patch_size=14, depth=24, num_heads=24,
        init_values=1e-5, embed_dim=DIM, mlp_ratio=2.66667 * 2, num_classes=0,
        no_embed_class=True, mlp_layer=SwiGLUPacked, act_layer=torch.nn.SiLU,
        reg_tokens=8, dynamic_img_size=True,
    )
    model.load_state_dict(torch.load(weights, map_location="cpu"), strict=True)
    model = model.to(device).eval()
    if any(".lora_" in name for name, _ in model.named_parameters()):
        raise RuntimeError("the tissue grid must come from the frozen backbone, with no LoRA")
    return model, weights


def load_he(path: Path) -> np.ndarray:
    with tifffile.TiffFile(str(path)) as handle:
        array = handle.series[0].levels[0].asarray()
    if array.ndim == 2:
        array = np.stack([array] * 3, -1)
    if array.shape[-1] == 4:
        array = array[..., :3]
    return np.ascontiguousarray(array.astype(np.uint8))


def tile_grid_embeddings(he, model, device, T, stride, bbox):
    H, W = he.shape[:2]
    xmin, ymin, xmax, ymax = bbox
    xs = np.arange(max(0, int(xmin) - T), min(W, int(xmax) + T) + 1, stride)
    ys = np.arange(max(0, int(ymin) - T), min(H, int(ymax) + T) + 1, stride)
    centers, tiles = [], []
    for y0 in ys:
        for x0 in xs:
            x1, y1 = x0 + T, y0 + T
            tile = np.full((T, T, 3), 255, np.uint8)
            sx0, sy0 = max(0, x0), max(0, y0)
            sx1, sy1 = min(W, x1), min(H, y1)
            if sx1 > sx0 and sy1 > sy0:
                tile[sy0 - y0:sy1 - y0, sx0 - x0:sx1 - x0] = he[sy0:sy1, sx0:sx1]
            grey = tile.mean(2)
            if (grey >= BLANK_LEVEL).mean() > BLANK_FRAC:
                continue
            tiles.append(cv2.resize(tile, (CROP, CROP), interpolation=cv2.INTER_AREA))
            centers.append((x0 + T / 2.0, y0 + T / 2.0))
    if not tiles:
        return np.zeros((0, 2)), np.zeros((0, DIM), np.float32)
    embs = []
    with torch.no_grad():
        for i in range(0, len(tiles), BATCH):
            batch = torch.stack([TF(t) for t in tiles[i:i + BATCH]]).to(device, non_blocking=True)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                embs.append(model(batch).cpu().float())
    return np.array(centers, np.float64), torch.cat(embs, 0).numpy()


def run(sample: str) -> dict[str, Any]:
    started = time.time()
    environment = C.require_usable_gpu()
    out_path = OUT_ROOT / f"{sample}.pt"
    manifest_path = MANIFEST_DIR / f"{sample}.json"
    if out_path.is_file() and manifest_path.is_file():
        prior = json.loads(manifest_path.read_text())
        if prior.get("status") == "PASS" and prior.get("feature_file_sha256") == C.sha256_file(out_path):
            return {**prior, "status": "ALREADY_COMPLETE"}
        raise RuntimeError(f"existing tissue grid does not match its manifest: {out_path}")

    cells = C.load_cells(sample)
    cids = cells["cell_id"].astype(str).tolist()
    coords = cells[["he_px", "he_py"]].to_numpy().astype(np.float64)
    if np.isnan(coords).any():
        raise RuntimeError(f"{sample}: a Cell cell centroid is NaN on the aligned-cell canvas")

    device = torch.device("cuda")
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.benchmark = True

    he_path = C.he_path(sample)
    he = load_he(he_path)
    model, weights = load_uni2_frozen(device)
    print(f"[{sample}] HE {he.shape} {he.nbytes / 1e9:.1f}GB, {len(cids)} cells", flush=True)

    bbox = (coords[:, 0].min(), coords[:, 1].min(), coords[:, 0].max(), coords[:, 1].max())
    per_scale = []
    tiles_per_scale = {}
    for T in SCALES:
        centers, embs = tile_grid_embeddings(he, model, device, T, T // 2, bbox)
        tiles_per_scale[str(T)] = int(len(centers))
        if len(centers) == 0:
            raise RuntimeError(f"{sample}: scale {T} produced no non-blank tile")
        tree = cKDTree(centers)
        _, idx = tree.query(coords, k=1)
        per_scale.append(embs[idx])
        print(f"[{sample}] scale {T}: {len(centers)} tiles -> per-cell {embs[idx].shape}", flush=True)

    feats = np.concatenate(per_scale, 1).astype(np.float16)
    if feats.shape != (len(cids), DIM * len(SCALES)):
        raise RuntimeError(f"{sample}: tissue grid shape {feats.shape} is not [N, 3072]")
    if not np.isfinite(feats.astype(np.float32)).all():
        raise RuntimeError(f"{sample}: tissue grid holds a non-finite value")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = out_path.with_name(out_path.name + f".tmp.{os.getpid()}")
    torch.save(
        {"features": torch.from_numpy(feats), "cell_ids": cids,
         "scales": SCALES, "dim_per_scale": DIM},
        tmp,
    )
    os.replace(tmp, out_path)

    result = {
        "schema_version": "v8.tissue_grid.v1",
        "status": "PASS",
        "sample": sample,
        "cancer": str(C.record(sample)["cancer"]),
        "backbone": "UNI2-h_vit_giant_patch14_224_frozen",
        "lora": None,
        "base_weights_path": str(weights),
        "base_weights_sha256": C.sha256_file(weights),
        "feature_kind": "tissue_grid",
        "dimension": DIM * len(SCALES),
        "scales": SCALES,
        "stride_fraction": 0.5,
        "blank_level": BLANK_LEVEL,
        "blank_fraction": BLANK_FRAC,
        "mpp": C.PX_UM,
        "color_space": "native_rgb",
        "normalization": "imagenet_mean_0.485_0.456_0.406_std_0.229_0.224_0.225",
        "centroid_convention": "aligned_he he_px/he_py (Cell canonical cell set)",
        "centroid_convention_note": (
            "All feature banks use the corrected aligned H&E canvas and the contracted cell set."
        ),
        "row_count": int(len(cids)),
        "n_tiles_per_scale": tiles_per_scale,
        "ordered_cell_id_sha256": C.order_sha256(cids),
        "feature_path": str(out_path.resolve()),
        "feature_file_sha256": C.sha256_file(out_path),
        "he_path": str(he_path),
        "contract_sha256": C.contract_sha256(),
        "producer_sha256": DEPENDENCY_SHA256,
        "script_sha256": C.sha256_file(Path(__file__)),
        "wall_sec": round(time.time() - started, 1),
        "environment": environment,
    }
    C.atomic_json(manifest_path, result)
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sample")
    parser.add_argument("--task-id", type=int)
    parser.add_argument("--list", action="store_true", help="print the missing slides and exit")
    args = parser.parse_args()
    todo = target_samples()
    if args.list:
        print(json.dumps({"n_targets": len(todo), "samples": todo}, indent=2))
        return 0
    if args.sample:
        sample = args.sample
    elif args.task_id is not None:
        if not 0 <= args.task_id < len(todo):
            raise ValueError(f"task id outside the 0..{len(todo) - 1} cohort list")
        sample = todo[args.task_id]
    else:
        raise ValueError("provide --sample, --task-id or --list")
    result = run(sample)
    print(json.dumps(result, indent=2, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
