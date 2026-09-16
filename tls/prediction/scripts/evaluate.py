#!/usr/bin/env python3

from __future__ import annotations
import os

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import ndimage as ndi
from sklearn.metrics import average_precision_score, roc_auc_score


MODULE_ROOT = Path(__file__).resolve().parents[1]
RUNS = MODULE_ROOT / "outputs" / "runs"
TLS_INSTANCES = MODULE_ROOT / "data" / "tls_instances.parquet"
VARIANTS = ("baseline_groupval", "groupweight", "querypos_groupval")
BASELINE = "baseline_groupval"
STRIDE = 603
TARGET_PREVALENCE = 0.0055


def safe_metric(fn, y: np.ndarray, p: np.ndarray) -> float:
    return float(fn(y, p)) if len(np.unique(y)) == 2 else float("nan")


def dice(y: np.ndarray, pred: np.ndarray) -> float:
    tp = int((pred & (y == 1)).sum())
    fp = int((pred & (y == 0)).sum())
    fn = int((~pred & (y == 1)).sum())
    return 2 * tp / max(2 * tp + fp + fn, 1)


def prevalence_matched_auprc(
    y: np.ndarray, p: np.ndarray, seed: int = 42, repeats: int = 100
) -> tuple[float, float]:
    positive = np.flatnonzero(y == 1)
    negative = np.flatnonzero(y == 0)
    if not len(positive) or not len(negative):
        return float("nan"), float("nan")
    observed = len(positive) / len(y)
    if np.isclose(observed, TARGET_PREVALENCE):
        value = float(average_precision_score(y, p))
        return value, 0.0
    rng = np.random.default_rng(seed)
    values = []
    for _ in range(repeats):
        if observed > TARGET_PREVALENCE:
            n_positive = max(
                1,
                min(
                    len(positive),
                    int(
                        round(
                            TARGET_PREVALENCE
                            / (1 - TARGET_PREVALENCE)
                            * len(negative)
                        )
                    ),
                ),
            )
            chosen = np.concatenate(
                [rng.choice(positive, n_positive, replace=False), negative]
            )
        else:
            n_negative = max(
                1,
                min(
                    len(negative),
                    int(
                        round(
                            (1 - TARGET_PREVALENCE)
                            / TARGET_PREVALENCE
                            * len(positive)
                        )
                    ),
                ),
            )
            chosen = np.concatenate(
                [positive, rng.choice(negative, n_negative, replace=False)]
            )
        values.append(float(average_precision_score(y[chosen], p[chosen])))
    return float(np.mean(values)), float(np.std(values))


def dice_threshold(y: np.ndarray, p: np.ndarray) -> float:
    if y.sum() == 0:
        raise RuntimeError("no positive calibration labels")
    candidates = np.unique(np.quantile(p, np.linspace(0.0, 1.0, 201)))
    scores = [(dice(y, p >= threshold), -float(threshold)) for threshold in candidates]
    _, negative_threshold = max(scores)
    return -negative_threshold


def filter_components(frame: pd.DataFrame, positive: np.ndarray, min_size: int) -> np.ndarray:
    kept = np.zeros(len(frame), dtype=bool)
    if min_size <= 1:
        return positive.copy()
    for _, positions in frame.groupby("unit_id", sort=False).indices.items():
        positions = np.asarray(positions, dtype=np.int64)
        pos_local = positive[positions]
        if not pos_local.any():
            continue
        x = np.rint(frame.iloc[positions]["x_px"].to_numpy() / STRIDE).astype(int)
        y = np.rint(frame.iloc[positions]["y_px"].to_numpy() / STRIDE).astype(int)
        x0, y0 = int(x.min()), int(y.min())
        canvas = np.zeros((int(y.max() - y0 + 1), int(x.max() - x0 + 1)), dtype=bool)
        canvas[y[pos_local] - y0, x[pos_local] - x0] = True
        labelled, n_components = ndi.label(canvas, structure=np.ones((3, 3), dtype=int))
        sizes = np.bincount(labelled.ravel(), minlength=n_components + 1)
        labels_at_tiles = labelled[y - y0, x - x0]
        kept[positions] = pos_local & (sizes[labels_at_tiles] >= min_size)
    return kept


def fit_policy(frame: pd.DataFrame) -> tuple[float, int, float]:
    y = frame["y"].to_numpy()
    p = frame["prob"].to_numpy()
    if y.sum() == 0:
        raise RuntimeError("no positives in policy calibration data")
    thresholds = np.unique(np.quantile(p, np.linspace(0.0, 1.0, 101)))
    best = (-1.0, -1, -1.0)
    for min_size in (1, 2, 3):
        for threshold in thresholds:
            raw = p >= threshold
            pred = filter_components(frame, raw, min_size)
            score = dice(y, pred)
            candidate = (score, -min_size, -float(threshold))
            if candidate > best:
                best = candidate
    return -best[2], -best[1], best[0]


def crossfit_predictions(
    frame: pd.DataFrame, tune_components: bool
) -> tuple[np.ndarray, list[dict]]:
    predictions = np.zeros(len(frame), dtype=bool)
    supported = np.zeros(len(frame), dtype=bool)
    policies = []
    for cancer in sorted(frame["cancer"].unique()):
        cancer_mask = frame["cancer"].eq(cancer).to_numpy()
        if frame.loc[cancer_mask, "y"].sum() == 0:
            continue
        for fold in sorted(frame.loc[cancer_mask, "fold_id"].unique()):
            test = cancer_mask & frame["fold_id"].eq(fold).to_numpy()
            calibration = cancer_mask & ~frame["fold_id"].eq(fold).to_numpy()
            if frame.loc[calibration, "y"].sum() == 0:
                policies.append(
                    {
                        "cancer": cancer,
                        "fold": int(fold),
                        "supported": False,
                        "reason": "no positive calibration labels",
                    }
                )
                continue
            calibration_frame = frame.loc[calibration].reset_index(drop=True)
            if tune_components:
                threshold, min_size, calibration_dice = fit_policy(calibration_frame)
            else:
                threshold = dice_threshold(
                    calibration_frame["y"].to_numpy(),
                    calibration_frame["prob"].to_numpy(),
                )
                min_size = 1
                calibration_dice = dice(
                    calibration_frame["y"].to_numpy(),
                    calibration_frame["prob"].to_numpy() >= threshold,
                )
            test_frame = frame.loc[test].reset_index(drop=True)
            raw = test_frame["prob"].to_numpy() >= threshold
            pred = filter_components(test_frame, raw, min_size)
            positions = np.flatnonzero(test)
            predictions[positions] = pred
            supported[positions] = True
            policies.append(
                {
                    "cancer": cancer,
                    "fold": int(fold),
                    "supported": True,
                    "threshold": float(threshold),
                    "min_component_tiles": int(min_size),
                    "calibration_dice": float(calibration_dice),
                }
            )
    return predictions & supported, policies


def macro_metrics(frame: pd.DataFrame, predicted: np.ndarray) -> dict:
    rows = []
    for cancer, data in frame.groupby("cancer", sort=True):
        positions = data.index.to_numpy(dtype=np.int64)
        y = data["y"].to_numpy()
        p = data["prob"].to_numpy()
        if y.sum() == 0:
            continue
        pmatch, pmatch_sd = prevalence_matched_auprc(y, p)
        rows.append(
            {
                "cancer": cancer,
                "n_tiles": int(len(data)),
                "n_positive": int(y.sum()),
                "n_units": int(data["unit_id"].nunique()),
                "n_groups": int(data["cv_group"].nunique()),
                "n_positive_groups": int(data.loc[data["y"].eq(1), "cv_group"].nunique()),
                "auroc": safe_metric(roc_auc_score, y, p),
                "auprc": safe_metric(average_precision_score, y, p),
                "prevalence_matched_auprc_0_0055": pmatch,
                "prevalence_matched_auprc_sd": pmatch_sd,
                "dice": dice(y, predicted[positions]),
            }
        )
    return {
        "macro_auroc": float(np.nanmean([row["auroc"] for row in rows])),
        "macro_auprc": float(np.nanmean([row["auprc"] for row in rows])),
        "macro_dice": float(np.nanmean([row["dice"] for row in rows])),
        "per_cancer": rows,
    }


def load_ensemble(run_id: str, variant: str, seeds: list[int]) -> pd.DataFrame:
    key = ["unit_id", "field", "cancer", "cv_group", "x_px", "y_px", "y", "fold_id"]
    frames = []
    for seed in seeds:
        root = RUNS / run_id / variant / f"seed_{seed}"
        if not (root / "SUCCESS").is_file():
            raise RuntimeError(f"missing SUCCESS marker: {root}")
        frame = pd.read_parquet(root / "oof.parquet").sort_values(key).reset_index(drop=True)
        frames.append(frame)
    reference = frames[0][key]
    for seed, frame in zip(seeds[1:], frames[1:]):
        if not reference.equals(frame[key]):
            raise RuntimeError(f"{variant} seed {seed} does not share identical OOF rows/folds")
    out = frames[0][key + ["region_frac", "rejected_tls_frac"]].copy()
    out["prob"] = np.mean([frame["prob"].to_numpy() for frame in frames], axis=0)
    out["variant"] = variant
    return out


def bootstrap_delta(
    baseline: pd.DataFrame,
    candidate: pd.DataFrame,
    baseline_pred: np.ndarray,
    candidate_pred: np.ndarray,
    iterations: int,
    seed: int,
) -> dict:
    keys = ["unit_id", "field", "cancer", "cv_group", "x_px", "y_px", "y", "fold_id"]
    if not baseline[keys].equals(candidate[keys]):
        raise RuntimeError("candidate does not share identical test rows and folds")
    groups = np.asarray(sorted(baseline["cv_group"].unique()))
    group_index = {group: index for index, group in enumerate(groups)}
    row_group = baseline["cv_group"].map(group_index).to_numpy(dtype=np.int16)
    cancers = [
        cancer
        for cancer, data in baseline.groupby("cancer", sort=True)
        if data["y"].sum() > 0
    ]
    cancer_positions = {
        cancer: np.flatnonzero(baseline["cancer"].eq(cancer).to_numpy())
        for cancer in cancers
    }
    y_all = baseline["y"].to_numpy(dtype=np.int8)
    baseline_prob = baseline["prob"].to_numpy()
    candidate_prob = candidate["prob"].to_numpy()
    baseline_order = {
        cancer: positions[
            np.argsort(-baseline_prob[positions], kind="stable")
        ]
        for cancer, positions in cancer_positions.items()
    }
    candidate_order = {
        cancer: positions[
            np.argsort(-candidate_prob[positions], kind="stable")
        ]
        for cancer, positions in cancer_positions.items()
    }
    rng = np.random.default_rng(seed)
    delta_auprc = []
    delta_dice = []

    def weighted_ap(order: np.ndarray, row_weight: np.ndarray) -> float:
        ys = y_all[order]
        ws = row_weight[order]
        total_positive = float((ws * ys).sum())
        if total_positive <= 0 or float((ws * (1 - ys)).sum()) <= 0:
            return float("nan")
        true_positive = np.cumsum(ws * ys)
        false_positive = np.cumsum(ws * (1 - ys))
        precision = true_positive / np.maximum(true_positive + false_positive, 1e-12)
        return float((precision * ws * ys).sum() / total_positive)

    def weighted_dice(
        y: np.ndarray, predicted: np.ndarray, weight: np.ndarray
    ) -> float:
        tp = float(weight[(predicted) & (y == 1)].sum())
        fp = float(weight[(predicted) & (y == 0)].sum())
        fn = float(weight[(~predicted) & (y == 1)].sum())
        return 2 * tp / max(2 * tp + fp + fn, 1e-12)

    for _ in range(iterations):
        sampled = rng.integers(0, len(groups), size=len(groups))
        group_weight = np.bincount(sampled, minlength=len(groups)).astype(float)
        row_weight = group_weight[row_group]
        baseline_auprc = []
        candidate_auprc = []
        baseline_dice = []
        candidate_dice = []
        for cancer in cancers:
            positions = cancer_positions[cancer]
            weight = row_weight[positions]
            y = y_all[positions]
            baseline_auprc.append(
                weighted_ap(baseline_order[cancer], row_weight)
            )
            candidate_auprc.append(
                weighted_ap(candidate_order[cancer], row_weight)
            )
            baseline_dice.append(
                weighted_dice(y, baseline_pred[positions], weight)
            )
            candidate_dice.append(
                weighted_dice(y, candidate_pred[positions], weight)
            )
        delta_auprc.append(
            float(np.nanmean(candidate_auprc) - np.nanmean(baseline_auprc))
        )
        delta_dice.append(
            float(np.nanmean(candidate_dice) - np.nanmean(baseline_dice))
        )

    def summarize(values: list[float]) -> dict:
        arr = np.asarray(values, dtype=float)
        finite = arr[np.isfinite(arr)]
        return {
            "mean": float(finite.mean()),
            "ci95": [float(np.quantile(finite, 0.025)), float(np.quantile(finite, 0.975))],
            "probability_gt_zero": float((finite > 0).mean()),
            "n_finite": int(len(finite)),
        }

    return {
        "sampling_unit": "coarse cv_group",
        "iterations": iterations,
        "delta_macro_auprc": summarize(delta_auprc),
        "delta_macro_dice_crossfit_no_filter": summarize(delta_dice),
    }


def region_metrics(frame: pd.DataFrame, predicted: np.ndarray) -> dict:
    instances = pd.read_parquet(TLS_INSTANCES)
    instances = instances.loc[instances["region_qc_usable"]].copy()
    positive_tiles = frame.loc[
        predicted, ["unit_id", "x_px", "y_px"]
    ].drop_duplicates()
    instance_hits = instances.merge(
        positive_tiles.assign(predicted=True),
        on=["unit_id", "x_px", "y_px"],
        how="left",
    )
    instance_level = (
        instance_hits.groupby(["sample", "tls_region"], sort=False)["predicted"]
        .any()
        .fillna(False)
    )
    return {
        "usable_tls_instances": int(len(instance_level)),
        "instances_detected_any_overlap": int(instance_level.sum()),
        "instance_recall_any_overlap": float(instance_level.mean()) if len(instance_level) else float("nan"),
        "definition": "a usable molecular TLS is detected if any predicted tile overlaps its raster",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--seeds", default="42,137,2026")
    parser.add_argument("--bootstrap-iterations", type=int, default=2000)
    args = parser.parse_args()
    seeds = [int(value) for value in args.seeds.split(",")]
    outdir = RUNS / args.run_id / "evaluation"
    if (outdir / "SUCCESS").exists():
        raise FileExistsError(f"completed evaluation already exists: {outdir}")
    outdir.mkdir(parents=True, exist_ok=True)

    ensembles = {
        variant: load_ensemble(args.run_id, variant, seeds) for variant in VARIANTS
    }
    key = ["unit_id", "field", "cancer", "cv_group", "x_px", "y_px", "y", "fold_id"]
    reference = ensembles[BASELINE][key]
    for variant in VARIANTS[1:]:
        if not reference.equals(ensembles[variant][key]):
            raise RuntimeError(f"{variant} does not share the frozen paired OOF contract")

    reports = {}
    no_filter_predictions = {}
    for variant, frame in ensembles.items():
        no_filter, no_filter_policy = crossfit_predictions(frame, tune_components=False)
        tuned, tuned_policy = crossfit_predictions(frame, tune_components=True)
        no_filter_predictions[variant] = no_filter
        report = {
            "no_filter": macro_metrics(frame, no_filter),
            "nested_component_policy": macro_metrics(frame, tuned),
            "region_metrics_no_filter": region_metrics(frame, no_filter),
            "region_metrics_nested_component_policy": region_metrics(frame, tuned),
            "no_filter_fold_policies": no_filter_policy,
            "nested_component_fold_policies": tuned_policy,
        }
        reports[variant] = report
        frame.assign(
            pred_no_filter=no_filter,
            pred_nested_component=tuned,
        ).to_parquet(outdir / f"ensemble_oof_{variant}.parquet", index=False)

    comparisons = {}
    baseline = ensembles[BASELINE]
    for variant in VARIANTS[1:]:
        comparisons[variant] = bootstrap_delta(
            baseline,
            ensembles[variant],
            no_filter_predictions[BASELINE],
            no_filter_predictions[variant],
            args.bootstrap_iterations,
            seed=4700 + VARIANTS.index(variant),
        )


    selected = BASELINE
    eligible = []
    base_metrics = reports[BASELINE]["no_filter"]
    for variant in VARIANTS[1:]:
        delta_point_dice = (
            reports[variant]["no_filter"]["macro_dice"]
            - base_metrics["macro_dice"]
        )
        probability = comparisons[variant]["delta_macro_auprc"]["probability_gt_zero"]
        mean_delta = comparisons[variant]["delta_macro_auprc"]["mean"]
        if probability >= 0.90 and mean_delta > 0 and delta_point_dice >= -0.01:
            eligible.append(
                (
                    reports[variant]["no_filter"]["macro_auprc"],
                    reports[variant]["no_filter"]["macro_dice"],
                    variant,
                )
            )
    if eligible:
        selected = max(eligible)[2]

    selection = {
        "selected_variant": selected,
        "baseline": BASELINE,
        "selection_rule": {
            "primary": "native-prevalence macro AUPRC across cancers with molecular positives",
            "candidate_probability_delta_gt_zero_min": 0.90,
            "candidate_mean_delta_must_be_positive": True,
            "maximum_point_macro_dice_regression": 0.01,
            "fallback": "baseline_groupval",
        },
        "seeds": seeds,
        "paired_rows": int(len(reference)),
        "cv_groups": int(reference["cv_group"].nunique()),
        "reports": reports,
        "paired_group_bootstrap": comparisons,
    }
    (outdir / "evaluation.json").write_text(
        json.dumps(selection, indent=2, default=float) + "\n"
    )
    (outdir / "selected_variant.txt").write_text(selected + "\n")
    required = [
        outdir / "evaluation.json",
        outdir / "selected_variant.txt",
        *[outdir / f"ensemble_oof_{variant}.parquet" for variant in VARIANTS],
    ]
    if any(not path.is_file() or path.stat().st_size == 0 for path in required):
        raise RuntimeError("required evaluation artifact missing or empty")
    (outdir / "SUCCESS").write_text("ok\n")
    print(json.dumps({"selected_variant": selected, "reports": reports}, indent=2, default=float))


if __name__ == "__main__":
    main()
