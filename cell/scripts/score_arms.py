#!/usr/bin/env python3
from __future__ import annotations
import os
PROJECTS_ROOT = os.environ.get("PROJECTS_ROOT", "/data/heid")
from pathlib import Path
_RELEASE = Path(__file__).resolve().parents[2]

import argparse
import glob
import hashlib
import json
import sys
from collections import defaultdict

import numpy as np
import pandas as pd

sys.path.insert(0, str(_RELEASE / "cell/scripts"))
import tme_lib as TL

OOF_ROOT = _RELEASE / "cell/outputs/oof"
DENOMINATOR = _RELEASE / "cell/configs/scoring_denominator.json"
EXCLUSION_LABELS = Path(PROJECTS_ROOT) / "cell/outputs/labels_region_schwann"
EPS = 1e-6
BASES = ("identity11", "scored10_renormalized")
REPORT_CLASSES = ("Schwann", "Myeloid_NOS", "B_cell", "NK_T", "Tumor", "Fibroblast")
SCHWANN = TL.COARSE11.index("Schwann")
TAUS = np.round(np.arange(0.05, 0.9501, 0.01), 2)


def load_arm(arm: str, root: Path, fold: int) -> dict:
    directory = root / f"fold{fold}"
    files = sorted(directory.glob("*.npz"))
    if not files:
        raise SystemExit(f"no held-out arrays for arm {arm} at {directory}")
    cells, y, pred, prob, patient, cancer, sample = [], [], [], [], [], [], []
    checkpoints = set()
    for path in files:
        with np.load(path, allow_pickle=False) as value:
            known = value["identity_known_mask"].astype(bool)
            p = value["prob"].astype(np.float64)
            yy = value["y"].astype(np.int64)
            cc = value["cell"].astype(str)
            n = int(known.sum())
            cells.append(cc[known])
            y.append(yy[known])
            prob.append(p[known])
            pred.append(np.argmax(p[known], axis=1).astype(np.int64))
            patient.append(np.repeat(str(value["patient"]), n))
            cancer.append(np.repeat(str(value["cancer"]), n))
            sample.append(np.repeat(str(value["sample"]), n))
            checkpoints.add(str(value["checkpoint_sha256"]))
    if len(checkpoints) != 1:
        raise SystemExit(f"arm {arm}: held-out files disagree on the checkpoint")
    order_key = np.concatenate([np.repeat(p.stem, 1) for p in files])
    n_rows = sum(len(c) for c in cells)
    return {
        "arm": arm,
        "n_files": len(files),
        "samples_in_order": list(order_key),
        "cell": np.concatenate(cells),
        "y": np.concatenate(y),
        "pred": np.concatenate(pred),
        "prob": np.concatenate(prob),
        "patient": np.concatenate(patient),
        "cancer": np.concatenate(cancer),
        "sample": np.concatenate(sample),
        "fold": np.full(n_rows, fold, dtype=np.int64),
        "checkpoint_sha256": sorted(checkpoints)[0],
    }


def load_arm_folds(arm: str, root: Path, folds: list[int]) -> dict:
    parts = [load_arm(arm, root, f) for f in folds]
    keys = set(parts[0])
    out: dict = {}
    for key in keys:
        values = [p[key] for p in parts]
        if isinstance(values[0], np.ndarray):
            out[key] = np.concatenate(values, axis=0)
        elif isinstance(values[0], (set, frozenset)):
            merged = set()
            for v in values:
                merged |= set(v)
            out[key] = merged
        elif isinstance(values[0], list):
            out[key] = [x for v in values for x in v]
        else:
            out[key] = values[0]
    if len(set(parts[i]["checkpoint_sha256"] for i in range(len(parts)))) != len(parts):
        raise SystemExit(f"arm {arm}: two folds share a checkpoint")
    return out


def excluded_rows(data: dict, labels_root: Path) -> np.ndarray:
    flagged: dict[str, set[str]] = {}
    for f in glob.glob(str(labels_root / "*.parquet")):
        t = pd.read_parquet(f, columns=["cell_id", "identity_excluded"])
        ex = t.loc[t["identity_excluded"].to_numpy(dtype=bool), "cell_id"].astype(str)
        if len(ex):
            flagged[Path(f).stem] = set(ex)
    if not flagged:
        raise SystemExit(f"no identity_excluded rows found under {labels_root}")
    mask = np.zeros(len(data["y"]), dtype=bool)
    samples = data["sample"]
    for sample, cells in flagged.items():
        rows = np.flatnonzero(samples == sample)
        if len(rows):
            mask[rows] = np.fromiter((c in cells for c in data["cell"][rows]), dtype=bool, count=len(rows))
    return mask


def restrict(data: dict, keep: np.ndarray) -> dict:
    out = dict(data)
    for key in ("cell", "y", "pred", "prob", "patient", "cancer", "sample", "fold"):
        out[key] = data[key][keep]
    return out


def assert_scoring_denominator(data: dict, excluded_per_patient: dict[str, int] | None) -> dict:
    spec = json.loads(DENOMINATOR.read_text())
    observed: dict[str, int] = {}
    for name in data["patient"].tolist():
        observed[name] = observed.get(name, 0) + 1
    total = sum(observed.values())
    n_excl = sum((excluded_per_patient or {}).values())
    if total != int(spec["n_scorable_cells"]) - n_excl:
        raise SystemExit(
            f"scored {total} rows, the denominator says {int(spec['n_scorable_cells']) - n_excl}"
        )
    mismatched = []
    for patient, entry in spec["per_patient"].items():
        want = int(entry["n_scorable"]) - int((excluded_per_patient or {}).get(patient, 0))
        got = int(observed.get(patient, 0))
        if got != want:
            mismatched.append({"patient": patient, "expected": want, "observed": got})
    empty = sorted(p for p, n in observed.items() if n == 0)
    if empty:
        raise SystemExit(f"patients scored with zero rows: {empty[:5]}")
    if mismatched:
        raise SystemExit(f"per-patient scorable counts differ from the registered denominator: {mismatched[:5]}")
    return {
        "denominator_rule": spec["rule"],
        "denominator_field": spec["denominator_field"],
        "n_scorable_cells": total,
        "n_excluded_blank_crop": int(spec["n_excluded_blank_crop"]),
        "n_excluded_no_identity_label": int(spec["n_excluded_no_identity_label"]),
        "n_excluded_region_ruler": int(n_excl),
        "n_patients_affected_by_exclusion": int(spec["n_patients_affected"]),
        "denominator_schema": spec["schema_version"],
        "checked_against": str(DENOMINATOR),
    }


def row_identity_sha256(data: dict) -> str:
    digest = hashlib.sha256()
    for s, c in zip(data["sample"], data["cell"], strict=True):
        digest.update(f"{s}\x1f{c}\x1e".encode("utf-8"))
    return digest.hexdigest()


def patient_counts(data: dict) -> tuple[np.ndarray, np.ndarray, list[str], list[str]]:
    patients = sorted(set(data["patient"].tolist()))
    index = {name: row for row, name in enumerate(patients)}
    true_counts = np.zeros((len(patients), TL.N11), dtype=np.int64)
    pred_counts = np.zeros((len(patients), TL.N11), dtype=np.int64)
    rows = np.asarray([index[name] for name in data["patient"]], dtype=np.int64)
    np.add.at(true_counts, (rows, data["y"]), 1)
    np.add.at(pred_counts, (rows, data["pred"]), 1)
    cancer_of: dict[str, str] = {}
    for name, cancer in zip(data["patient"], data["cancer"], strict=True):
        previous = cancer_of.setdefault(name, cancer)
        if previous != cancer:
            raise SystemExit(f"patient {name} appears under two cancers")
    return true_counts, pred_counts, patients, [cancer_of[p] for p in patients]


def arm_metrics(data: dict) -> dict:
    cm = TL.confusion11(data["y"], data["pred"])
    auroc, per_class_auroc, n_rows = TL.macro_ovr_auroc10(data["y"], data["prob"])
    true_counts, pred_counts, patients, cancers = patient_counts(data)
    out = {
        "arm": data["arm"],
        "n_cells_scored": int(len(data["y"])),
        "n_samples": int(len(set(data["sample"].tolist()))),
        "n_patients": len(patients),
        "checkpoint_sha256": data["checkpoint_sha256"],
        "row_identity_sha256": row_identity_sha256(data),
        "pooled_macro_f1_10": TL.macro_f1_10(cm),
        "pooled_macro_f1_11": TL.macro_f1_11(cm),
        "per_class_f1": {
            TL.COARSE11[i]: float(v) for i, v in enumerate(TL.per_class_f1(cm))
        },
        "macro_ovr_auroc_10": auroc,
        "per_class_auroc": per_class_auroc,
        "n_rows_for_auroc": n_rows,
        "confusion11": cm.tolist(),
        "composition": {},
    }
    for basis in BASES:
        metrics = TL.composition_metrics(true_counts, pred_counts, basis, EPS)
        out["composition"][basis] = TL.patient_equal_summary(metrics)
    out["_patient_axis"] = {
        "patients": patients,
        "cancers": cancers,
        "true_counts": true_counts,
        "pred_counts": pred_counts,
    }
    return out


def paired_bootstrap(
    baseline: dict, candidate: dict, basis: str, replicates: int, seed: int
) -> dict:
    axis_b, axis_c = baseline["_patient_axis"], candidate["_patient_axis"]
    if axis_b["patients"] != axis_c["patients"]:
        raise SystemExit("arms do not share a patient axis")
    if not np.array_equal(axis_b["true_counts"], axis_c["true_counts"]):
        raise SystemExit("arms disagree on the true patient composition")
    mb = TL.composition_metrics(
        axis_b["true_counts"], axis_b["pred_counts"], basis, EPS
    )
    mc = TL.composition_metrics(
        axis_c["true_counts"], axis_c["pred_counts"], basis, EPS
    )
    present = mb["present"]
    by_cancer = defaultdict(list)
    for row, cancer in enumerate(axis_b["cancers"]):
        by_cancer[cancer].append(row)
    groups = [np.asarray(v, dtype=np.int64) for v in by_cancer.values()]
    rng = np.random.RandomState(seed)

    def macro(lse: np.ndarray, pr: np.ndarray) -> float:
        values = [
            lse[pr[:, j], j].mean() for j in range(len(TL.SCORED10)) if pr[:, j].any()
        ]
        return float(np.mean(values)) if values else float("nan")

    d_macro = np.empty(replicates, dtype=np.float64)
    d_l1 = np.empty(replicates, dtype=np.float64)
    for b in range(replicates):
        draw = np.concatenate([g[rng.randint(0, len(g), len(g))] for g in groups])
        pr = present[draw]
        d_macro[b] = macro(mc["log_share_error"][draw], pr) - macro(
            mb["log_share_error"][draw], pr
        )
        d_l1[b] = (
            mc["composition_l1_10"][draw].mean() - mb["composition_l1_10"][draw].mean()
        )

    def summarize(values: np.ndarray, point: float) -> dict:
        finite = values[np.isfinite(values)]
        low, high = float(np.quantile(finite, 0.025)), float(np.quantile(finite, 0.975))
        return {
            "delta_point_estimate": float(point),
            "delta_bootstrap_median": float(np.median(finite)),
            "ci95": [low, high],
            "ci_contains_zero": bool(low <= 0.0 <= high),
            "n_finite_replicates": int(len(finite)),
        }

    sb, sc = TL.patient_equal_summary(mb), TL.patient_equal_summary(mc)
    return {
        "basis": basis,
        "replicates": int(replicates),
        "seed": int(seed),
        "stratification": "patients resampled with replacement within cancer, one draw scored for both arms",
        "macro_log_share_error_10": summarize(
            d_macro,
            sc["macro_log_share_error_10"] - sb["macro_log_share_error_10"],
        ),
        "mean_composition_l1_10": summarize(
            d_l1, sc["mean_composition_l1_10"] - sb["mean_composition_l1_10"]
        ),
    }


def per_patient_confusion(data: dict, patients: list[str]) -> np.ndarray:
    index = {name: row for row, name in enumerate(patients)}
    rows = np.asarray([index[name] for name in data["patient"]], dtype=np.int64)
    keep = (data["y"] >= 0) & (data["y"] < TL.N11)
    flat = (
        rows[keep] * (TL.N11 * TL.N11)
        + data["y"][keep] * TL.N11
        + data["pred"][keep]
    )
    counts = np.bincount(flat, minlength=len(patients) * TL.N11 * TL.N11)
    return counts.reshape(len(patients), TL.N11, TL.N11)


def paired_f1_bootstrap(
    baseline: dict, candidate: dict, base_raw: dict, cand_raw: dict,
    replicates: int, seed: int
) -> dict:
    patients = baseline["_patient_axis"]["patients"]
    cm_base = per_patient_confusion(base_raw, patients)
    cm_cand = per_patient_confusion(cand_raw, patients)
    by_cancer = defaultdict(list)
    for row, cancer in enumerate(baseline["_patient_axis"]["cancers"]):
        by_cancer[cancer].append(row)
    groups = [np.asarray(v, dtype=np.int64) for v in by_cancer.values()]
    rng = np.random.RandomState(seed)
    deltas = np.empty(replicates, dtype=np.float64)
    for b in range(replicates):
        draw = np.concatenate([g[rng.randint(0, len(g), len(g))] for g in groups])
        deltas[b] = TL.macro_f1_10(cm_cand[draw].sum(axis=0)) - TL.macro_f1_10(
            cm_base[draw].sum(axis=0)
        )
    low, high = float(np.quantile(deltas, 0.025)), float(np.quantile(deltas, 0.975))
    return {
        "delta_point_estimate": float(
            candidate["pooled_macro_f1_10"] - baseline["pooled_macro_f1_10"]
        ),
        "delta_bootstrap_median": float(np.median(deltas)),
        "ci95": [low, high],
        "ci_contains_zero": bool(low <= 0.0 <= high),
        "replicates": int(replicates),
        "seed": int(seed),
    }


def schwann_f1(cm: np.ndarray) -> float:
    tp = float(cm[SCHWANN, SCHWANN])
    fp = float(cm[:, SCHWANN].sum()) - tp
    fn = float(cm[SCHWANN, :].sum()) - tp
    return 2 * tp / (2 * tp + fp + fn) if tp else 0.0


def paired_schwann_bootstrap(
    baseline: dict, base_raw: dict, cand_raw: dict, replicates: int, seed: int
) -> dict:
    patients = baseline["_patient_axis"]["patients"]
    cm_base = per_patient_confusion(base_raw, patients)
    cm_cand = per_patient_confusion(cand_raw, patients)
    by_cancer = defaultdict(list)
    for row, cancer in enumerate(baseline["_patient_axis"]["cancers"]):
        by_cancer[cancer].append(row)
    groups = [np.asarray(v, dtype=np.int64) for v in by_cancer.values()]
    rng = np.random.RandomState(seed)
    deltas = np.empty(replicates, dtype=np.float64)
    for b in range(replicates):
        draw = np.concatenate([g[rng.randint(0, len(g), len(g))] for g in groups])
        deltas[b] = schwann_f1(cm_cand[draw].sum(axis=0)) - schwann_f1(cm_base[draw].sum(axis=0))
    low, high = float(np.quantile(deltas, 0.025)), float(np.quantile(deltas, 0.975))
    return {
        "delta_point_estimate": schwann_f1(cm_cand.sum(axis=0)) - schwann_f1(cm_base.sum(axis=0)),
        "delta_bootstrap_median": float(np.median(deltas)),
        "ci95": [low, high],
        "ci_above_zero": bool(low > 0.0),
        "replicates": int(replicates),
        "seed": int(seed),
        "stratification": "patients resampled with replacement within cancer",
    }


def nested_threshold(data: dict) -> dict:
    p = data["prob"][:, SCHWANN]
    pos = data["y"] == SCHWANN
    folds = sorted(set(data["fold"].tolist()))
    counts = {}
    t64 = TAUS.astype(np.float64)
    for k in folds:
        sel = data["fold"] == k
        ps, ns = np.sort(p[sel & pos]), np.sort(p[sel & ~pos])
        tp = len(ps) - np.searchsorted(ps, t64, side="left")
        fp = len(ns) - np.searchsorted(ns, t64, side="left")
        counts[k] = np.stack([tp, fp, len(ps) - tp], axis=1)
    total = np.zeros(3, dtype=np.int64)
    chosen = {}
    for k in folds:
        agg = sum(counts[j] for j in folds if j != k)
        f1 = np.where(agg[:, 0] > 0, 2 * agg[:, 0] / np.maximum(2 * agg[:, 0] + agg[:, 1] + agg[:, 2], 1), 0.0)
        t = int(np.argmax(f1))
        chosen[str(k)] = float(TAUS[t])
        total += counts[k][t]
    tp, fp, fn = (float(x) for x in total)
    return {"tau_by_fold": chosen, "tp": int(tp), "fp": int(fp), "fn": int(fn),
            "precision": tp / (tp + fp) if tp + fp else 0.0, "recall": tp / (tp + fn) if tp + fn else 0.0,
            "f1": 2 * tp / (2 * tp + fp + fn) if tp else 0.0}


def reference_check(metrics: dict, comparisons: dict, path: Path, baseline: str, arm: str) -> list[str]:
    ref = json.loads(path.read_text())
    fails = []

    def walk(a, b, where):
        if isinstance(b, dict):
            for key, value in b.items():
                if key in ("arm", "checkpoint_sha256"):
                    continue
                if key not in a:
                    fails.append(f"missing {where}.{key}")
                else:
                    walk(a[key], value, f"{where}.{key}")
        elif isinstance(b, list):
            if len(a) != len(b):
                fails.append(f"length {where}")
            else:
                for i, (x, y) in enumerate(zip(a, b)):
                    walk(x, y, f"{where}[{i}]")
        elif isinstance(b, (int, float)) and not isinstance(b, bool):
            if not (a == b or (isinstance(a, float) and np.isnan(a) and np.isnan(b))):
                fails.append(f"{where}: {a} != {b}")

    for name, ref_name in ((baseline, baseline.split("=", 1)[0]), (arm, arm.split("=", 1)[0])):
        walk(metrics[name], ref["per_arm"][ref_name], f"per_arm.{ref_name}")
    walk(comparisons[arm], ref["paired_bootstrap_vs_baseline"][arm.split("=", 1)[0]], f"paired.{arm}")
    return fails


def parse_arms(specs: list[str], default_root: Path) -> dict[str, Path]:
    roots: dict[str, Path] = {}
    for spec in specs:
        if "=" in spec:
            name, root = spec.split("=", 1)
            roots[name] = Path(root)
        else:
            roots[spec] = default_root / spec
    return roots


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--folds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    parser.add_argument("--baseline", type=str, required=True, help="arm name used as the reference")
    parser.add_argument("--arms", nargs="+", required=True,
                        help="NAME or NAME=OOF_ROOT; a bare NAME reads <oof-root>/NAME/fold<k>/")
    parser.add_argument("--oof-root", type=Path, default=OOF_ROOT)
    parser.add_argument("--ruler", choices=("region",), default="region",
                        help="region: drop identity_excluded rows for every arm")
    parser.add_argument("--exclusion-labels", type=Path, default=EXCLUSION_LABELS)
    parser.add_argument("--expect-excluded", type=int, default=None)
    parser.add_argument("--expect-schwann-remaining", type=int, default=None)
    parser.add_argument("--pairs", nargs="*", default=[], help="CANDIDATE:BASELINE pairs for the Schwann bootstrap")
    parser.add_argument("--replicates", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=20260730)
    parser.add_argument("--schwann-replicates", type=int, default=2000)
    parser.add_argument("--schwann-seed", type=int, default=20260910)
    parser.add_argument("--reference-check", type=Path, default=None)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    roots = parse_arms(args.arms, args.oof_root)
    if args.baseline not in roots:
        roots = {args.baseline: args.oof_root / args.baseline, **roots}
    arms = list(dict.fromkeys([args.baseline, *roots]))
    raw = {arm: load_arm_folds(arm, Path(roots[arm]), args.folds) for arm in arms}

    ruler_info: dict = {"ruler": args.ruler}
    excluded_per_patient = None
    if args.ruler == "region":
        base = raw[args.baseline]
        mask = excluded_rows(base, args.exclusion_labels)
        if not (base["y"][mask] == SCHWANN).all():
            raise SystemExit("an excluded row is not Schwann")
        n_excl = int(mask.sum())
        n_left = int(((base["y"] == SCHWANN) & ~mask).sum())
        if args.expect_excluded is not None and n_excl != args.expect_excluded:
            raise SystemExit(f"region ruler drops {n_excl} rows, expected {args.expect_excluded}")
        if args.expect_schwann_remaining is not None and n_left != args.expect_schwann_remaining:
            raise SystemExit(f"region ruler keeps {n_left} Schwann rows, expected {args.expect_schwann_remaining}")
        excluded_per_patient = defaultdict(int)
        for name in base["patient"][mask].tolist():
            excluded_per_patient[name] += 1
        ruler_info.update({"n_excluded": n_excl, "n_schwann_remaining": n_left,
                           "rule": "drop (sample, cell) with identity_excluded in the exclusion labels, for every arm"})
        for arm in arms:
            m = excluded_rows(raw[arm], args.exclusion_labels) if arm != args.baseline else mask
            raw[arm] = restrict(raw[arm], ~m)
    denominator = assert_scoring_denominator(raw[args.baseline], excluded_per_patient)
    metrics = {arm: arm_metrics(raw[arm]) for arm in arms}

    identities = {arm: metrics[arm]["row_identity_sha256"] for arm in arms}
    if len(set(identities.values())) != 1:
        raise SystemExit("arms were not scored on identical rows: " + json.dumps(identities, sort_keys=True))
    for arm in arms[1:]:
        if not np.array_equal(raw[args.baseline]["y"], raw[arm]["y"]):
            raise SystemExit(f"arm {arm}: label vector differs from the baseline")

    comparisons = {}
    for arm in arms:
        if arm == args.baseline:
            continue
        comparisons[arm] = {
            "vs_baseline": args.baseline,
            "composition": {
                basis: paired_bootstrap(metrics[args.baseline], metrics[arm], basis, args.replicates, args.seed)
                for basis in BASES
            },
            "pooled_macro_f1_10": paired_f1_bootstrap(
                metrics[args.baseline], metrics[arm], raw[args.baseline], raw[arm], args.replicates, args.seed),
        }

    if args.reference_check is not None:
        other = [a for a in arms if a != args.baseline]
        fails = reference_check(metrics, comparisons, args.reference_check, args.baseline, other[0])
        if fails:
            print(json.dumps({"reference_check": "FAIL", "n": len(fails), "first": fails[:20]}, indent=1))
            return 1
        print(json.dumps({"reference_check": "PASS", "against": str(args.reference_check)}))

    schwann_pairs = {}
    for spec in args.pairs:
        cand, base = spec.split(":", 1)
        schwann_pairs[spec] = paired_schwann_bootstrap(
            metrics[base], raw[base], raw[cand], args.schwann_replicates, args.schwann_seed)

    table = []
    for arm in arms:
        m = metrics[arm]
        row = {
            "arm": arm,
            "n_cells_scored": m["n_cells_scored"],
            "schwann_f1_argmax": m["per_class_f1"]["Schwann"],
            "schwann_nested_threshold": nested_threshold(raw[arm]),
            "pooled_macro_f1_10": m["pooled_macro_f1_10"],
            "macro_log_share_error_10_identity11": m["composition"]["identity11"]["macro_log_share_error_10"],
            "mean_composition_l1_10_identity11": m["composition"]["identity11"]["mean_composition_l1_10"],
            "macro_ovr_auroc_10": m["macro_ovr_auroc_10"],
        }
        for name in REPORT_CLASSES:
            row[f"f1_{name}"] = m["per_class_f1"][name]
        table.append(row)
    for m in metrics.values():
        m.pop("_patient_axis", None)

    payload = {
        "schema_version": "v12.arm_comparison.v1",
        "ruler": ruler_info,
        "folds": args.folds,
        "baseline": args.baseline,
        "arms": {arm: str(roots[arm]) for arm in arms},
        "eps": EPS,
        "scoring_denominator": denominator,
        "decision_layer": "none; predicted label is the argmax of the raw identity probability",
        "row_identity_sha256_by_arm": identities,
        "arm_table": table,
        "schwann_f1_paired_bootstrap": schwann_pairs,
        "per_arm": metrics,
        "paired_bootstrap_vs_baseline": comparisons,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    TL.write_json(args.out, payload)
    print(json.dumps({"event": "scored", "path": str(args.out)}))
    print(json.dumps(table, indent=1, sort_keys=True))
    print(json.dumps(schwann_pairs, indent=1, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
