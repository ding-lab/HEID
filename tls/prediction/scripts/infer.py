#!/usr/bin/env python3

from __future__ import annotations

import argparse
import importlib
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch


MODULE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(MODULE_ROOT / "scripts"))
train = importlib.import_module("train")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--tiles", type=Path, required=True)
    parser.add_argument("--features", type=Path, required=True)
    parser.add_argument("--cancer", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--threshold", type=float)
    args = parser.parse_args()

    device = torch.device(args.device)
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    cancers = list(checkpoint["cancers"])
    if args.cancer not in cancers:
        raise ValueError(f"cancer {args.cancer!r} not in checkpoint vocabulary {cancers}")
    tiles = pd.read_parquet(args.tiles).reset_index(drop=True)
    feature_object = torch.load(args.features, map_location="cpu", weights_only=False)
    tile_index = np.asarray(feature_object["tile_index"], dtype=np.int64)
    features = np.asarray(feature_object["features"], dtype=np.float32)
    if len(tiles) != len(features) or not np.array_equal(
        tile_index, np.arange(len(tiles), dtype=np.int64)
    ):
        raise RuntimeError("feature/tile index contract failed")
    if not {"x_px", "y_px"}.issubset(tiles.columns):
        raise RuntimeError("tile parquet must contain x_px and y_px")
    frame = tiles[["x_px", "y_px"]].copy()
    frame["unit_id"] = "inference"
    nbr = train.build_neighbor_index(frame)
    onehot = np.zeros((len(frame), len(cancers)), dtype=np.float32)
    onehot[:, cancers.index(args.cancer)] = 1.0
    weights = np.ones(len(frame), dtype=np.float32)
    scaled = (
        features - np.asarray(checkpoint["scaler_mu"], dtype=np.float32)
    ) / np.asarray(checkpoint["scaler_sd"], dtype=np.float32)
    context = train.TensorContext(scaled, nbr, onehot, weights, device)
    variant = str(checkpoint["variant"])
    model = train.build_model(
        variant,
        int(checkpoint["architecture"]["d"]),
        int(checkpoint["architecture"]["n_onehot"]),
        device,
    )
    model.load_state_dict(checkpoint["state_dict"])
    indices = np.arange(len(frame), dtype=np.int64)
    probability = train.predict(model, context, indices, indices)
    threshold = (
        float(args.threshold)
        if args.threshold is not None
        else float(checkpoint["thresholds_insample_calibrated"][args.cancer])
    )
    output = tiles.copy()
    output["tls_probability"] = probability
    output["tls_call"] = probability >= threshold
    output["tls_threshold"] = threshold
    args.output.parent.mkdir(parents=True, exist_ok=True)
    output.to_parquet(args.output, index=False)
    print(
        f"saved {args.output}: tiles={len(output)} "
        f"positive_calls={int(output['tls_call'].sum())} threshold={threshold:.6f}"
    )


if __name__ == "__main__":
    main()
