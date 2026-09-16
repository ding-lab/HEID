#!/usr/bin/env python3

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import platform
import random
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
import torch
import torch.nn as nn
from sklearn.metrics import average_precision_score, roc_auc_score


MODULE_ROOT = Path(__file__).resolve().parents[1]
HE_PRED = Path(os.environ.get("TLS_HE_PRED_DATA", str(Path(os.environ.get("PROJECTS_ROOT", "/data/heid")) / "tls/prediction")))
TLS = HE_PRED.parent
FEATURES = Path(os.environ.get("TLS_FEATURES", str(HE_PRED / "features")))
TILES_MAIN = HE_PRED / "data" / "xenium_5k" / "tiles"
TILES_EXCLUDED = HE_PRED / "data" / "xenium_5k" / "excluded" / "tiles"
SLIDES = MODULE_ROOT / "configs" / "slides_5k.tsv"
LABELS = Path(os.environ.get("TLS_LABELS", str(MODULE_ROOT / "data/labels")))
RUNS = MODULE_ROOT / "outputs" / "runs"
MODELS = MODULE_ROOT / "outputs" / "models"
INPUT_HASHES = MODULE_ROOT / "outputs" / "results" / "input_hashes.tsv"

STRIDE = 603
FIELD_PX = 1206
RINGS = (1, 2)
N_OUTER_FOLDS = 5
VAL_FRACTION_GROUPS = 0.20
MAX_EPOCHS = 40
PATIENCE = 6
BATCH_SIZE = 4096
LR = 1e-3
WEIGHT_DECAY = 1e-3
FOCAL_ALPHA = 0.75
FOCAL_GAMMA = 2.0
VARIANTS = ("baseline_groupval", "groupweight", "querypos_groupval")


def ring_offsets(radius: int) -> list[tuple[int, int]]:
    out = []
    for dx in range(-radius, radius + 1):
        for dy in range(-radius, radius + 1):
            if max(abs(dx), abs(dy)) == radius:
                out.append((dx, dy))
    return out


OFFSETS = [offset for radius in RINGS for offset in ring_offsets(radius)]


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def hash_frame(frame: pd.DataFrame) -> str:
    payload = frame.to_csv(sep="\t", index=False, lineterminator="\n").encode()
    return hashlib.sha256(payload).hexdigest()


def git_state() -> dict:
    try:
        sha = subprocess.check_output(
            ["git", "-C", str(HE_PRED), "rev-parse", "HEAD"],
            stderr=subprocess.DEVNULL,
            text=True,
        ).strip()
        dirty = bool(
            subprocess.check_output(
                ["git", "-C", str(HE_PRED), "status", "--porcelain"],
                stderr=subprocess.DEVNULL,
                text=True,
            ).strip()
        )
        return {"git_sha": sha, "git_dirty": dirty}
    except Exception as exc:
        return {"git_sha": None, "git_dirty": None, "git_error": str(exc)}


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True


def tiles_path(field: str) -> Path:
    for root in (TILES_MAIN, TILES_EXCLUDED):
        path = root / field / "tiles.parquet"
        if path.is_file():
            return path
    raise FileNotFoundError(f"tile grid missing for {field}")


def load_feature(field: str) -> tuple[np.ndarray, np.ndarray]:
    path = FEATURES / field / "uni2.pt"
    if not path.is_file():
        raise FileNotFoundError(f"canonical UNI2 feature missing for {field}")
    obj = torch.load(path, map_location="cpu", weights_only=False)
    return (
        np.asarray(obj["tile_index"], dtype=np.int64),
        np.asarray(obj["features"], dtype=np.float32),
    )


def load_data() -> tuple[pd.DataFrame, np.ndarray, dict]:
    slides = pd.read_csv(SLIDES, sep="\t")
    frames: list[pd.DataFrame] = []
    features: list[np.ndarray] = []
    input_rows: list[dict] = []
    cursor = 0
    for row in slides.itertuples(index=False):
        unit = str(row.unit_id)
        field = str(row.field)
        label_path = LABELS / f"{unit}_tiles.parquet"
        label = pd.read_parquet(label_path)
        grid_path = tiles_path(field)
        grid = pd.read_parquet(grid_path, columns=["x_px", "y_px"]).reset_index(drop=True)
        tile_index, feat = load_feature(field)
        if len(grid) != len(feat) or not np.array_equal(
            tile_index, np.arange(len(grid), dtype=np.int64)
        ):
            raise RuntimeError(f"{field}: feature/tile index contract failed")
        lookup = {
            (int(x), int(y)): i
            for i, (x, y) in enumerate(zip(grid["x_px"], grid["y_px"]))
        }
        rows = np.array(
            [
                lookup.get((int(x), int(y)), -1)
                for x, y in zip(label["x_px"], label["y_px"])
            ],
            dtype=np.int64,
        )
        if (rows < 0).any() or len(np.unique(rows)) != len(rows):
            raise RuntimeError(f"{unit}: label-to-feature grid mapping failed")
        frame = label.copy()
        frame["xenium_run"] = str(row.xenium_run_folder)
        frame["patient"] = str(row.patient)
        frame["_full"] = np.arange(cursor, cursor + len(frame), dtype=np.int64)
        frames.append(frame)
        features.append(feat[rows])
        cursor += len(frame)
        input_rows.append(
            {
                "unit_id": unit,
                "field": field,
                "label_sha256": sha256(label_path),
                "feature_path": str((FEATURES / field / "uni2.pt").resolve()),
                "feature_size": int((FEATURES / field / "uni2.pt").stat().st_size),
                "tile_grid_path": str(grid_path.resolve()),
                "tile_grid_size": int(grid_path.stat().st_size),
            }
        )
    full = pd.concat(frames, ignore_index=True)
    x_full = np.concatenate(features, axis=0).astype(np.float32)
    if len(full) != len(x_full):
        raise RuntimeError("metadata/feature concatenation mismatch")
    valid = full["label"].to_numpy(dtype=np.int8) >= 0
    centers = full.loc[valid].reset_index(drop=True).copy()
    center_full = np.flatnonzero(valid).astype(np.int64)
    y = centers["label"].to_numpy(dtype=np.int8)
    meta = {
        "full": full,
        "centers": centers,
        "center_full": center_full,
        "y": y,
        "input_rows": input_rows,
    }
    return full, x_full, meta


def build_neighbor_index(full: pd.DataFrame) -> np.ndarray:
    nbr = np.full((len(full), len(OFFSETS)), -1, dtype=np.int64)
    for unit, group in full.groupby("unit_id", sort=False):
        rows = group.index.to_numpy(dtype=np.int64)
        gx = np.rint(group["x_px"].to_numpy() / STRIDE).astype(np.int64)
        gy = np.rint(group["y_px"].to_numpy() / STRIDE).astype(np.int64)
        lookup = {(int(x), int(y)): int(r) for x, y, r in zip(gx, gy, rows)}
        for local, row in enumerate(rows):
            for k, (dx, dy) in enumerate(OFFSETS):
                nbr[row, k] = lookup.get((gx[local] + dx, gy[local] + dy), -1)
    return nbr


def balanced_group_kfold(groups: np.ndarray, y: np.ndarray, k: int = 5) -> np.ndarray:
    stats = pd.DataFrame({"group": groups, "y": y})
    positive = stats.groupby("group")["y"].sum().astype(int)
    size = stats.groupby("group").size().astype(int)
    positive_groups = sorted(
        [g for g in positive.index if positive[g] > 0],
        key=lambda g: (-int(positive[g]), str(g)),
    )
    negative_groups = sorted(
        [g for g in positive.index if positive[g] == 0],
        key=lambda g: (-int(size[g]), str(g)),
    )
    fold_pos_groups = [0] * k
    fold_pos = [0] * k
    fold_size = [0] * k
    mapping: dict[str, int] = {}
    for group in positive_groups:
        fold = min(
            range(k),
            key=lambda f: (fold_pos_groups[f], fold_pos[f], fold_size[f], f),
        )
        mapping[group] = fold
        fold_pos_groups[fold] += 1
        fold_pos[fold] += int(positive[group])
        fold_size[fold] += int(size[group])
    for group in negative_groups:
        fold = min(range(k), key=lambda f: (fold_size[f], f))
        mapping[group] = fold
        fold_size[fold] += int(size[group])
    return np.asarray([mapping[group] for group in groups], dtype=np.int8)


def group_val_split(
    indices: np.ndarray, groups: np.ndarray, y: np.ndarray, seed: int
) -> tuple[np.ndarray, np.ndarray]:
    local_groups = groups[indices]
    local_y = y[indices]
    n_groups = int(pd.unique(local_groups).size)
    k = max(2, min(int(round(1.0 / VAL_FRACTION_GROUPS)), n_groups))
    fold = balanced_group_kfold(local_groups, local_y, k=k)
    candidates = list(range(k))
    rng = np.random.default_rng(seed)
    rng.shuffle(candidates)
    selected = None
    for candidate in candidates:
        val_local = np.flatnonzero(fold == candidate)
        fit_local = np.flatnonzero(fold != candidate)
        if (
            len(val_local)
            and y[indices[val_local]].sum() > 0
            and y[indices[fit_local]].sum() > 0
        ):
            selected = candidate
            break
    if selected is None:
        raise RuntimeError("could not construct a group-disjoint inner validation set")
    val = indices[fold == selected]
    fit = indices[fold != selected]
    overlap = set(groups[fit]).intersection(set(groups[val]))
    if overlap:
        raise RuntimeError(f"inner split leaks groups: {sorted(overlap)[:5]}")
    return fit, val


class GatedAttentionHead(nn.Module):
    def __init__(
        self,
        d: int,
        n_onehot: int,
        attn_dim: int = 256,
        h1: int = 512,
        h2: int = 128,
        dropout: float = 0.4,
    ):
        super().__init__()
        self.v = nn.Linear(d, attn_dim)
        self.u = nn.Linear(d, attn_dim)
        self.w = nn.Linear(attn_dim, 1)
        self.mlp = nn.Sequential(
            nn.Linear(d + d + n_onehot, h1),
            nn.BatchNorm1d(h1),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(h1, h2),
            nn.BatchNorm1d(h2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(h2, 1),
        )

    def attention_logits(
        self, center: torch.Tensor, neigh: torch.Tensor
    ) -> torch.Tensor:
        del center
        return self.w(torch.tanh(self.v(neigh)) * torch.sigmoid(self.u(neigh))).squeeze(-1)

    def forward(
        self,
        center: torch.Tensor,
        neigh: torch.Tensor,
        mask: torch.Tensor,
        onehot: torch.Tensor,
    ) -> torch.Tensor:
        scores = self.attention_logits(center, neigh).masked_fill(mask == 0, -1e9)
        weights = torch.softmax(scores, dim=1) * mask
        weights = weights / weights.sum(dim=1, keepdim=True).clamp_min(1e-12)
        pooled = (weights.unsqueeze(-1) * neigh).sum(dim=1)
        return self.mlp(torch.cat([center, pooled, onehot], dim=-1)).squeeze(-1)


class QueryPositionAttentionHead(GatedAttentionHead):
    def __init__(self, d: int, n_onehot: int, query_dim: int = 64, **kwargs):
        super().__init__(d=d, n_onehot=n_onehot, **kwargs)
        self.query = nn.Linear(d, query_dim, bias=False)
        self.key = nn.Linear(d, query_dim, bias=False)
        self.position_bias = nn.Parameter(torch.zeros(len(OFFSETS)))
        self.query_scale = query_dim**-0.5

    def attention_logits(
        self, center: torch.Tensor, neigh: torch.Tensor
    ) -> torch.Tensor:
        gated = super().attention_logits(center, neigh)
        query = self.query(center).unsqueeze(1)
        key = self.key(neigh)
        conditioned = (query * key).sum(dim=-1) * self.query_scale
        return gated + conditioned + self.position_bias.unsqueeze(0)


def build_model(variant: str, d: int, n_onehot: int, device: torch.device) -> nn.Module:
    if variant == "querypos_groupval":
        return QueryPositionAttentionHead(d=d, n_onehot=n_onehot).to(device)
    return GatedAttentionHead(d=d, n_onehot=n_onehot).to(device)


def focal_loss(
    logits: torch.Tensor, target: torch.Tensor, weight: torch.Tensor | None = None
) -> torch.Tensor:
    ce = nn.functional.binary_cross_entropy_with_logits(
        logits.float(), target.float(), reduction="none"
    )
    pt = torch.exp(-ce)
    alpha = FOCAL_ALPHA * target + (1 - FOCAL_ALPHA) * (1 - target)
    loss = alpha * (1 - pt) ** FOCAL_GAMMA * ce
    if weight is None:
        return loss.mean()
    return (loss * weight).sum() / weight.sum().clamp_min(1e-12)


class TensorContext:
    def __init__(
        self,
        x_scaled: np.ndarray,
        nbr: np.ndarray,
        onehot_full: np.ndarray,
        row_weight_full: np.ndarray,
        device: torch.device,
    ):
        self.x = torch.as_tensor(x_scaled, device=device)
        nbr_t = torch.as_tensor(nbr, device=device)
        self.nbr_safe = nbr_t.clamp(min=0)
        self.nbr_mask = (nbr_t >= 0).float()
        self.onehot = torch.as_tensor(onehot_full, device=device)
        self.weight = torch.as_tensor(row_weight_full, device=device)
        self.device = device

    def gather(self, full_indices: np.ndarray):
        center = torch.as_tensor(full_indices, device=self.device)
        return (
            self.x[center],
            self.x[self.nbr_safe[center]],
            self.nbr_mask[center],
            self.onehot[center],
            self.weight[center],
        )


def train_fixed_epochs(
    variant: str,
    context: TensorContext,
    center_full: np.ndarray,
    y: np.ndarray,
    train_indices: np.ndarray,
    n_onehot: int,
    d: int,
    seed: int,
    n_epochs: int,
    device: torch.device,
) -> nn.Module:
    seed_everything(seed)
    model = build_model(variant, d, n_onehot, device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    target = torch.as_tensor(y, dtype=torch.float32, device=device)
    order = np.arange(len(train_indices))
    for epoch in range(n_epochs):
        rng = np.random.default_rng(seed + epoch)
        rng.shuffle(order)
        model.train()
        for start in range(0, len(order), BATCH_SIZE):
            selected = train_indices[order[start : start + BATCH_SIZE]]
            center, neigh, mask, onehot, weight = context.gather(center_full[selected])
            logits = model(center, neigh, mask, onehot)
            loss = focal_loss(
                logits,
                target[selected],
                weight if variant == "groupweight" else None,
            )
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    return model


def select_epochs(
    variant: str,
    context: TensorContext,
    center_full: np.ndarray,
    y: np.ndarray,
    fit_indices: np.ndarray,
    val_indices: np.ndarray,
    n_onehot: int,
    d: int,
    seed: int,
    device: torch.device,
    max_epochs: int,
) -> tuple[int, float, list[dict]]:
    seed_everything(seed)
    model = build_model(variant, d, n_onehot, device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    target = torch.as_tensor(y, dtype=torch.float32, device=device)
    order = np.arange(len(fit_indices))
    best_loss = float("inf")
    best_epoch = 0
    stale = 0
    history = []
    for epoch in range(max_epochs):
        rng = np.random.default_rng(seed + epoch)
        rng.shuffle(order)
        model.train()
        train_sum = 0.0
        train_batches = 0
        for start in range(0, len(order), BATCH_SIZE):
            selected = fit_indices[order[start : start + BATCH_SIZE]]
            center, neigh, mask, onehot, weight = context.gather(center_full[selected])
            logits = model(center, neigh, mask, onehot)
            loss = focal_loss(
                logits,
                target[selected],
                weight if variant == "groupweight" else None,
            )
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_sum += float(loss.item())
            train_batches += 1
        model.eval()
        val_sum = 0.0
        val_weight = 0
        with torch.no_grad():
            for start in range(0, len(val_indices), BATCH_SIZE):
                selected = val_indices[start : start + BATCH_SIZE]
                center, neigh, mask, onehot, weight = context.gather(center_full[selected])
                logits = model(center, neigh, mask, onehot)
                val_loss = focal_loss(
                    logits,
                    target[selected],
                    weight if variant == "groupweight" else None,
                )
                val_sum += float(val_loss.item()) * len(selected)
                val_weight += len(selected)
        val_mean = val_sum / max(val_weight, 1)
        history.append(
            {
                "epoch": epoch,
                "train_loss": train_sum / max(train_batches, 1),
                "val_loss": val_mean,
            }
        )
        if val_mean < best_loss - 1e-4:
            best_loss = val_mean
            best_epoch = epoch
            stale = 0
        else:
            stale += 1
            if stale >= PATIENCE:
                break
    return best_epoch, best_loss, history


def predict(
    model: nn.Module,
    context: TensorContext,
    center_full: np.ndarray,
    indices: np.ndarray,
) -> np.ndarray:
    model.eval()
    output = np.full(len(indices), np.nan, dtype=np.float32)
    with torch.no_grad():
        for start in range(0, len(indices), BATCH_SIZE):
            selected = indices[start : start + BATCH_SIZE]
            center, neigh, mask, onehot, _ = context.gather(center_full[selected])
            output[start : start + len(selected)] = (
                torch.sigmoid(model(center, neigh, mask, onehot))
                .detach()
                .cpu()
                .numpy()
            )
    return output


def safe_metric(fn, y: np.ndarray, p: np.ndarray) -> float:
    return float(fn(y, p)) if len(np.unique(y)) == 2 else float("nan")


def dice_threshold(y: np.ndarray, p: np.ndarray) -> float:
    if y.sum() == 0:
        raise RuntimeError("threshold calibration has no positive labels")
    thresholds = np.unique(np.quantile(p, np.linspace(0.0, 1.0, 401)))
    best = (-1.0, 0.5)
    for threshold in thresholds:
        pred = p >= threshold
        tp = int((pred & (y == 1)).sum())
        fp = int((pred & (y == 0)).sum())
        fn = int((~pred & (y == 1)).sum())
        value = 2 * tp / max(2 * tp + fp + fn, 1)
        if (value, -float(threshold)) > (best[0], -best[1]):
            best = (value, float(threshold))
    return best[1]


def crossfit_dice(oof: pd.DataFrame) -> tuple[float, list[dict]]:
    rows = []
    for cancer in sorted(oof["cancer"].unique()):
        cancer_data = oof.loc[oof["cancer"].eq(cancer)].copy()
        if cancer_data["y"].sum() == 0:
            continue
        predictions = np.zeros(len(cancer_data), dtype=bool)
        supported = np.zeros(len(cancer_data), dtype=bool)
        thresholds = {}
        for fold in sorted(cancer_data["fold_id"].unique()):
            test = cancer_data["fold_id"].eq(fold).to_numpy()
            calibration = ~test
            y_cal = cancer_data.loc[calibration, "y"].to_numpy()
            if y_cal.sum() == 0:
                thresholds[int(fold)] = None
                continue
            threshold = dice_threshold(
                y_cal, cancer_data.loc[calibration, "prob"].to_numpy()
            )
            thresholds[int(fold)] = threshold
            predictions[test] = cancer_data.loc[test, "prob"].to_numpy() >= threshold
            supported[test] = True
        y = cancer_data["y"].to_numpy()
        tp = int((predictions & (y == 1) & supported).sum())
        fp = int((predictions & (y == 0) & supported).sum())
        fn = int((~predictions & (y == 1) & supported).sum())
        dice = 2 * tp / max(2 * tp + fp + fn, 1)
        rows.append(
            {
                "cancer": cancer,
                "dice_crossfit_no_filter": dice,
                "n_tiles": int(len(cancer_data)),
                "n_positive": int(y.sum()),
                "n_supported": int(supported.sum()),
                "thresholds": thresholds,
            }
        )
    return (
        float(np.mean([row["dice_crossfit_no_filter"] for row in rows]))
        if rows
        else float("nan"),
        rows,
    )


def score_oof(oof: pd.DataFrame) -> dict:
    per_cancer = []
    for cancer, data in oof.groupby("cancer", sort=True):
        y = data["y"].to_numpy()
        p = data["prob"].to_numpy()
        per_cancer.append(
            {
                "cancer": cancer,
                "n_tiles": int(len(data)),
                "n_positive": int(y.sum()),
                "n_units": int(data["unit_id"].nunique()),
                "n_groups": int(data["cv_group"].nunique()),
                "n_positive_groups": int(
                    data.loc[data["y"].eq(1), "cv_group"].nunique()
                ),
                "auroc": safe_metric(roc_auc_score, y, p),
                "auprc": safe_metric(average_precision_score, y, p),
            }
        )
    macro_dice, dice_rows = crossfit_dice(oof)
    return {
        "pooled_auroc": safe_metric(
            roc_auc_score, oof["y"].to_numpy(), oof["prob"].to_numpy()
        ),
        "pooled_auprc": safe_metric(
            average_precision_score, oof["y"].to_numpy(), oof["prob"].to_numpy()
        ),
        "macro_auroc": float(np.nanmean([row["auroc"] for row in per_cancer])),
        "macro_auprc": float(np.nanmean([row["auprc"] for row in per_cancer])),
        "macro_dice_crossfit_no_filter": macro_dice,
        "per_cancer": per_cancer,
        "crossfit_dice": dice_rows,
    }


def data_tensors(
    full: pd.DataFrame,
    x_full: np.ndarray,
    meta: dict,
    nbr: np.ndarray,
    fit_for_scaler: np.ndarray,
    cancers: list[str],
    device: torch.device,
) -> tuple[TensorContext, np.ndarray, np.ndarray]:
    centers = meta["centers"]
    center_full = meta["center_full"]
    groups = centers["cv_group"].to_numpy()
    group_counts = pd.Series(groups).value_counts()
    row_weights = np.asarray(
        [len(groups) / (len(group_counts) * group_counts[g]) for g in groups],
        dtype=np.float32,
    )
    weight_full = np.zeros(len(full), dtype=np.float32)
    weight_full[center_full] = row_weights
    cancer_index = {cancer: i for i, cancer in enumerate(cancers)}
    onehot_full = np.zeros((len(full), len(cancers)), dtype=np.float32)
    onehot_full[
        center_full,
        centers["cancer"].map(cancer_index).to_numpy(dtype=np.int64),
    ] = 1.0
    fit_full = center_full[fit_for_scaler]
    mu = x_full[fit_full].mean(axis=0)
    sd = x_full[fit_full].std(axis=0) + 1e-6
    context = TensorContext(
        ((x_full - mu) / sd).astype(np.float32),
        nbr,
        onehot_full,
        weight_full,
        device,
    )
    return context, mu.astype(np.float32), sd.astype(np.float32)


def run_cv(args: argparse.Namespace) -> None:
    if args.variant not in VARIANTS:
        raise ValueError(args.variant)
    out = RUNS / args.run_id / args.variant / f"seed_{args.seed}"
    if (out / "SUCCESS").exists():
        raise FileExistsError(f"completed output already exists: {out}")
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    full, x_full, meta = load_data()
    centers = meta["centers"]
    center_full = meta["center_full"]
    y = meta["y"]
    groups = centers["cv_group"].to_numpy()
    fold_id = balanced_group_kfold(groups, y, k=N_OUTER_FOLDS)
    fold_map = (
        centers[["unit_id", "field", "cancer", "cv_group"]]
        .assign(fold_id=fold_id)
        .drop_duplicates()
        .sort_values(["fold_id", "cv_group", "unit_id"])
        .reset_index(drop=True)
    )
    if (fold_map.groupby("cv_group")["fold_id"].nunique() > 1).any():
        raise RuntimeError("outer split leaks cv_group")
    fold_map.to_csv(out / "fold_map.tsv", sep="\t", index=False)
    nbr = build_neighbor_index(full)
    cancers = sorted(centers["cancer"].unique())
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    probabilities = np.full(len(centers), np.nan, dtype=np.float32)
    histories = []
    max_epochs = args.max_epochs or MAX_EPOCHS
    folds = range(N_OUTER_FOLDS)
    if args.smoke_fold is not None:
        folds = [args.smoke_fold]

    for fold in folds:
        test = np.flatnonzero(fold_id == fold)
        outer_train = np.flatnonzero(fold_id != fold)
        inner_fit, inner_val = group_val_split(
            outer_train, groups, y, args.seed + fold * 101
        )
        select_context, _, _ = data_tensors(
            full, x_full, meta, nbr, inner_fit, cancers, device
        )
        best_epoch, best_loss, history = select_epochs(
            args.variant,
            select_context,
            center_full,
            y,
            inner_fit,
            inner_val,
            len(cancers),
            x_full.shape[1],
            args.seed,
            device,
            max_epochs,
        )
        del select_context
        if device.type == "cuda":
            torch.cuda.empty_cache()
        outer_context, _, _ = data_tensors(
            full, x_full, meta, nbr, outer_train, cancers, device
        )
        model = train_fixed_epochs(
            args.variant,
            outer_context,
            center_full,
            y,
            outer_train,
            len(cancers),
            x_full.shape[1],
            args.seed,
            best_epoch + 1,
            device,
        )
        probabilities[test] = predict(model, outer_context, center_full, test)
        histories.append(
            {
                "fold": fold,
                "best_epoch": best_epoch,
                "best_val_loss": best_loss,
                "n_outer_train": int(len(outer_train)),
                "n_test": int(len(test)),
                "n_inner_fit": int(len(inner_fit)),
                "n_inner_val": int(len(inner_val)),
                "inner_fit_groups": int(pd.unique(groups[inner_fit]).size),
                "inner_val_groups": int(pd.unique(groups[inner_val]).size),
                "inner_group_overlap": 0,
                "epochs": history,
            }
        )
        print(
            f"[fold {fold}] best_epoch={best_epoch} val={best_loss:.6f} "
            f"test={len(test)} pos={int(y[test].sum())}",
            flush=True,
        )
        del outer_context, model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    scored = np.isfinite(probabilities)
    oof = centers.loc[
        scored,
        [
            "unit_id",
            "field",
            "cancer",
            "cv_group",
            "x_px",
            "y_px",
            "region_frac",
            "rejected_tls_frac",
        ],
    ].copy()
    oof["y"] = y[scored]
    oof["fold_id"] = fold_id[scored]
    oof["prob"] = probabilities[scored]
    oof["variant"] = args.variant
    oof["seed"] = args.seed
    oof.to_parquet(out / "oof.parquet", index=False)
    metrics = score_oof(oof) if args.smoke_fold is None else {}
    result = {
        "run_id": args.run_id,
        "variant": args.variant,
        "seed": args.seed,
        "mode": "cv",
        "smoke_fold": args.smoke_fold,
        "n_full_grid_tiles": int(len(full)),
        "n_center_tiles": int(len(centers)),
        "n_scored_tiles": int(scored.sum()),
        "n_units": int(centers["unit_id"].nunique()),
        "n_cv_groups": int(centers["cv_group"].nunique()),
        "fold_map_sha256": hash_frame(fold_map),
        "empty_neighbor_centers": int(
            (nbr[center_full] < 0).all(axis=1).sum()
        ),
        "metrics": metrics,
        "folds": histories,
        "elapsed_seconds": round(time.time() - t0, 2),
    }
    (out / "results.json").write_text(json.dumps(result, indent=2, default=float) + "\n")
    if not INPUT_HASHES.is_file():
        raise FileNotFoundError(f"training input hash manifest missing: {INPUT_HASHES}")
    manifest = {
        **git_state(),
        "run_id": args.run_id,
        "variant": args.variant,
        "seed": args.seed,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "script": str(Path(__file__).resolve()),
        "script_sha256": sha256(Path(__file__).resolve()),
        "slides_sha256": sha256(SLIDES),
        "input_label_manifest_sha256": hash_frame(pd.DataFrame(meta["input_rows"])),
        "training_input_hash_manifest": str(INPUT_HASHES),
        "training_input_hash_manifest_sha256": sha256(INPUT_HASHES),
        "feature_root": str(FEATURES),
        "offsets": [list(offset) for offset in OFFSETS],
        "outer_folds": N_OUTER_FOLDS,
        "inner_validation": "coarse-group-disjoint, approximately 20%",
        "empty_neighbor_policy": "zero pooled vector",
        "loss": (
            "coarse-group-equal focal(alpha=0.75,gamma=2)"
            if args.variant == "groupweight"
            else "tile-mean focal(alpha=0.75,gamma=2)"
        ),
        "python": sys.version,
        "platform": platform.platform(),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "cublas_workspace_config": os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
        "device": str(device),
        "gpu": torch.cuda.get_device_name(0) if device.type == "cuda" else None,
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "slurm_array_task_id": os.environ.get("SLURM_ARRAY_TASK_ID"),
    }
    (out / "run_manifest.json").write_text(
        json.dumps(manifest, indent=2, default=str) + "\n"
    )
    required = [out / "oof.parquet", out / "results.json", out / "run_manifest.json"]
    if any(not path.is_file() or path.stat().st_size == 0 for path in required):
        raise RuntimeError("required output artifact missing or empty")
    (out / "SUCCESS").write_text("ok\n")
    print(f"[done] {out}", flush=True)


def run_full(args: argparse.Namespace) -> None:
    if args.variant not in VARIANTS:
        raise ValueError(args.variant)
    out = MODELS / args.run_id
    if (out / "SUCCESS").exists():
        raise FileExistsError(f"completed output already exists: {out}")
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    if args.evaluation_json is None or not args.evaluation_json.is_file():
        raise FileNotFoundError("--evaluation-json is required for a full production fit")
    evaluation = json.loads(args.evaluation_json.read_text())
    if evaluation["selected_variant"] != args.variant:
        raise RuntimeError(
            f"evaluation selected {evaluation['selected_variant']}, not {args.variant}"
        )
    cv_reference = evaluation["reports"][args.variant]["no_filter"]
    fold_policies = [
        row
        for row in evaluation["reports"][args.variant]["no_filter_fold_policies"]
        if row["supported"]
    ]
    oof_threshold_median = {
        cancer: float(
            np.median(
                [
                    row["threshold"]
                    for row in fold_policies
                    if row["cancer"] == cancer
                ]
            )
        )
        for cancer in sorted({row["cancer"] for row in fold_policies})
    }
    full, x_full, meta = load_data()
    centers = meta["centers"]
    center_full = meta["center_full"]
    y = meta["y"]
    groups = centers["cv_group"].to_numpy()
    all_indices = np.arange(len(centers), dtype=np.int64)
    inner_fit, inner_val = group_val_split(all_indices, groups, y, args.seed)
    nbr = build_neighbor_index(full)
    cancers = sorted(centers["cancer"].unique())
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    max_epochs = args.max_epochs or MAX_EPOCHS
    select_context, _, _ = data_tensors(
        full, x_full, meta, nbr, inner_fit, cancers, device
    )
    best_epoch, best_loss, history = select_epochs(
        args.variant,
        select_context,
        center_full,
        y,
        inner_fit,
        inner_val,
        len(cancers),
        x_full.shape[1],
        args.seed,
        device,
        max_epochs,
    )
    del select_context
    if device.type == "cuda":
        torch.cuda.empty_cache()
    final_context, mu, sd = data_tensors(
        full, x_full, meta, nbr, all_indices, cancers, device
    )
    model = train_fixed_epochs(
        args.variant,
        final_context,
        center_full,
        y,
        all_indices,
        len(cancers),
        x_full.shape[1],
        args.seed,
        best_epoch + 1,
        device,
    )
    insample_prob = predict(model, final_context, center_full, all_indices)
    thresholds = {}
    insample_metrics = {}
    for cancer in cancers:
        mask = centers["cancer"].eq(cancer).to_numpy()
        yc = y[mask]
        pc = insample_prob[mask]
        if yc.sum() == 0:
            continue
        threshold = dice_threshold(yc, pc)
        thresholds[cancer] = threshold
        insample_metrics[cancer] = {
            "threshold": threshold,
            "auroc": safe_metric(roc_auc_score, yc, pc),
            "auprc": safe_metric(average_precision_score, yc, pc),
            "note": "optimistic in-sample calibration diagnostic, not performance evidence",
        }
    arch = {
        "class": (
            "QueryPositionAttentionHead"
            if args.variant == "querypos_groupval"
            else "GatedAttentionHead"
        ),
        "d": int(x_full.shape[1]),
        "n_onehot": len(cancers),
        "attn_dim": 256,
        "h1": 512,
        "h2": 128,
        "dropout": 0.4,
        "query_dim": 64 if args.variant == "querypos_groupval" else None,
    }
    checkpoint = {
        "format": "xtls-v5-production-head-v1",
        "variant": args.variant,
        "state_dict": {key: value.detach().cpu() for key, value in model.state_dict().items()},
        "architecture": arch,
        "cancers": cancers,
        "scaler_mu": mu,
        "scaler_sd": sd,
        "offsets": [list(offset) for offset in OFFSETS],
        "stride": STRIDE,
        "field_px": FIELD_PX,
        "thresholds_insample_calibrated": thresholds,
        "thresholds_oof_fold_median_provenance": oof_threshold_median,
        "postprocessing": {
            "min_component_tiles": 1,
            "selection": (
                "cross-fitted nested k in {1,2,3} reduced macro Dice; "
                "no minimum-size filter retained"
            ),
        },
        "recipe": {
            "seed": args.seed,
            "best_epoch_zero_indexed": best_epoch,
            "final_epochs": best_epoch + 1,
            "best_inner_val_loss": best_loss,
            "inner_validation": "coarse-group-disjoint",
            "loss": (
                "coarse-group-equal focal(alpha=0.75,gamma=2)"
                if args.variant == "groupweight"
                else "tile-mean focal(alpha=0.75,gamma=2)"
            ),
            "lr": LR,
            "weight_decay": WEIGHT_DECAY,
            "batch_size": BATCH_SIZE,
        },
        "data": {
            "slides": str(SLIDES),
            "slides_sha256": sha256(SLIDES),
            "input_label_manifest_sha256": hash_frame(pd.DataFrame(meta["input_rows"])),
            "training_input_hash_manifest": str(INPUT_HASHES),
            "training_input_hash_manifest_sha256": sha256(INPUT_HASHES),
            "feature_root": str(FEATURES),
            "evaluation_json": str(args.evaluation_json.resolve()),
            "evaluation_json_sha256": sha256(args.evaluation_json),
        },
        "verified_cv_reference": cv_reference,
    }
    checkpoint_path = out / "production_head.pt"
    torch.save(checkpoint, checkpoint_path)

    loaded = torch.load(checkpoint_path, map_location=device, weights_only=False)
    reloaded = build_model(
        args.variant,
        int(loaded["architecture"]["d"]),
        int(loaded["architecture"]["n_onehot"]),
        device,
    )
    reloaded.load_state_dict(loaded["state_dict"])
    probe_indices = all_indices[: min(1024, len(all_indices))]
    before = predict(model, final_context, center_full, probe_indices)
    after = predict(reloaded, final_context, center_full, probe_indices)
    max_difference = float(np.max(np.abs(before - after)))
    if max_difference > 1e-7:
        raise RuntimeError(f"checkpoint round-trip prediction drift: {max_difference}")

    result = {
        "run_id": args.run_id,
        "variant": args.variant,
        "seed": args.seed,
        "mode": "full",
        "n_units": int(centers["unit_id"].nunique()),
        "n_cv_groups": int(centers["cv_group"].nunique()),
        "n_center_tiles": int(len(centers)),
        "n_positive": int(y.sum()),
        "best_epoch_zero_indexed": best_epoch,
        "final_epochs": best_epoch + 1,
        "best_inner_val_loss": best_loss,
        "inner_fit_groups": int(pd.unique(groups[inner_fit]).size),
        "inner_val_groups": int(pd.unique(groups[inner_val]).size),
        "insample_calibration": insample_metrics,
        "verified_cv_reference": cv_reference,
        "thresholds_oof_fold_median_provenance": oof_threshold_median,
        "postprocessing": checkpoint["postprocessing"],
        "checkpoint": str(checkpoint_path),
        "checkpoint_sha256": sha256(checkpoint_path),
        "roundtrip_smoke": {
            "ok": True,
            "n_predictions": int(len(probe_indices)),
            "max_abs_difference": max_difference,
        },
        "epoch_history": history,
        "elapsed_seconds": round(time.time() - t0, 2),
    }
    (out / "results.json").write_text(json.dumps(result, indent=2, default=float) + "\n")
    manifest = {
        **git_state(),
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "script": str(Path(__file__).resolve()),
        "script_sha256": sha256(Path(__file__).resolve()),
        "python": sys.version,
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "cublas_workspace_config": os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
        "device": str(device),
        "gpu": torch.cuda.get_device_name(0) if device.type == "cuda" else None,
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
    }
    (out / "run_manifest.json").write_text(
        json.dumps(manifest, indent=2, default=str) + "\n"
    )
    required = [
        checkpoint_path,
        out / "results.json",
        out / "run_manifest.json",
    ]
    if any(not path.is_file() or path.stat().st_size == 0 for path in required):
        raise RuntimeError("required production artifact missing or empty")
    (out / "SUCCESS").write_text("ok\n")
    print(f"[done] {out}", flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=["cv", "full"], default="cv")
    parser.add_argument("--variant", choices=VARIANTS, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--max-epochs", type=int)
    parser.add_argument("--smoke-fold", type=int, choices=range(N_OUTER_FOLDS))
    parser.add_argument("--evaluation-json", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.mode == "cv":
        run_cv(args)
    else:
        if args.smoke_fold is not None:
            raise ValueError("--smoke-fold is only valid in cv mode")
        run_full(args)


if __name__ == "__main__":
    main()
