#!/usr/bin/env python3
import argparse
from pathlib import Path
import numpy as np
import pandas as pd


def load_head(path):
    with np.load(path, allow_pickle=False) as data:
        head = {key: np.asarray(data[key]) for key in
                ("scaler_mean", "scaler_scale", "clf_coef", "clf_intercept", "clf_classes", "ridge_coef", "ridge_intercept")}
    for key in ("scaler_mean", "scaler_scale", "clf_coef", "ridge_coef"):
        if head[key].size != 3072 or not np.isfinite(head[key]).all():
            raise ValueError(f"Invalid 3072-dimensional parameter {key}")
        head[key] = head[key].reshape(3072)
    if np.any(head["scaler_scale"] <= 0) or not np.array_equal(head["clf_classes"], [0,1]):
        raise ValueError("Head scaling or class-order contract failed")
    for key in ("clf_intercept", "ridge_intercept"):
        if head[key].size != 1 or not np.isfinite(head[key]).all():
            raise ValueError(f"Invalid scalar parameter {key}")
    return head


def score_regions(features, region, head, sample):
    features = np.asarray(features, dtype=np.float32)
    raw_region = np.asarray(region)
    if features.ndim != 2 or features.shape[1] != 1536 or raw_region.shape != (len(features),):
        raise ValueError("Features must be [N,1536] with one matching region ID per tile")
    if not np.isfinite(features).all() or not np.isfinite(raw_region).all() or np.any(raw_region != np.floor(raw_region)):
        raise ValueError("Nonfinite feature or noninteger region ID")
    region = raw_region.astype(np.int64)
    rows = []
    for rid in np.unique(region[region > 0]):
        tiles = features[region == rid]
        if len(tiles) < 5:
            continue
        pooled = np.concatenate([tiles.mean(0), tiles.max(0)])

        scaled = pooled.copy()
        scaled -= head["scaler_mean"]
        scaled /= head["scaler_scale"]
        logit = float(head["clf_coef"] @ scaled + head["clf_intercept"].item())
        probability = float(np.exp(-np.logaddexp(0., -logit)))
        gc_hat = float((scaled[None] @ head["ridge_coef"] + head["ridge_intercept"])[0])
        rows.append(dict(sample=sample, region_id=int(rid), n_tiles=len(tiles), p_mature=probability, gc_hat=gc_hat))
    return pd.DataFrame(rows, columns=["sample", "region_id", "n_tiles", "p_mature", "gc_hat"])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--head", type=Path, required=True)
    parser.add_argument("--features", type=Path, required=True)
    parser.add_argument("--sample", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.features.suffix == ".npz":
        with np.load(args.features, allow_pickle=False) as source:
            features, regions = source["features"], source["region"]
    else:
        import torch
        source = torch.load(args.features, map_location="cpu", weights_only=False)
        features, regions = source["features"], source["region"]
    result = score_regions(features, regions, load_head(args.head), args.sample)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.output.suffix == ".parquet":result.to_parquet(args.output,index=False)
    else:result.to_csv(args.output,index=False)
    print(f"Saved {len(result)} supported regions to {args.output}")

if __name__ == "__main__":
    main()
