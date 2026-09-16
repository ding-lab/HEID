#!/usr/bin/env python3

from __future__ import annotations
from pathlib import Path
_RELEASE = Path(__file__).resolve().parents[3]
import os
PROJECTS_ROOT = os.environ.get("PROJECTS_ROOT", "/data/heid")

import glob
import json
import os
import sys
from pathlib import Path


INPUT_ROOT = Path(PROJECTS_ROOT)
PREDICTION_ROOT = _RELEASE / "nerve/prediction"
EVALUATION = PREDICTION_ROOT / "data/evaluation"
FOLDS = PREDICTION_ROOT / "data/folds_he_safe.json"
GOLD = INPUT_ROOT / "nerve/prediction/gold"
CELL_OOF = INPUT_ROOT / "cell/outputs/oof/cls_sigma3"
OUT = EVALUATION / "transferred_operating_point.json"
IOU = "iou>=0.10"


def prf(tp, n_pred, n_gold):
    p = tp / n_pred if n_pred else 0.0
    r = tp / n_gold if n_gold else 0.0
    return dict(precision=round(p, 4), recall=round(r, 4),
                f1=round(2 * p * r / (p + r), 4) if p + r else 0.0,
                tp=tp, n_pred=n_pred, n_gold=n_gold)


def pooled(samples, key):
    tp = npred = ngold = 0
    used = skipped = 0
    for s in samples:
        b = s["per_tau"].get(key)
        if b is None:
            continue
        if b.get("skipped_degenerate"):
            skipped += 1
            continue
        used += 1
        tp += b["iou_tp"]
        npred += b["n_pred_scored"]
        ngold += b["n_gold"]
    return prf(tp, npred, ngold), used, skipped


def transfer_counts(replay, fold_of):
    keys = sorted({k for row in replay["per_sample"] for k in row["per_tau"]})
    report = {"schema": "heid.nerve.transferred_operating_point",
              "rule": "Choose the positive fraction by pooled region F1 on the other target folds, then accumulate held-out counts.",
              "operating_points": keys, "arms": {"full_frame": {}}}
    scope, arm, r = "full_frame", "CELL", replay
    ps = r["per_sample"]
    if not ps:
        return report
    folds_present = sorted({fold_of[s["sample"]] for s in ps if s["sample"] in fold_of})
    tp = npred = ngold = 0
    per_fold = []
    for k in folds_present:
        train = [s for s in ps if fold_of.get(s["sample"]) != k]
        held = [s for s in ps if fold_of.get(s["sample"]) == k]
        if not train or not held:
            continue
        scored = {key: pooled(train, key)[0]["f1"] for key in keys}
        chosen = max(scored, key=scored.get)
        b, used, skipped = pooled(held, chosen)
        per_fold.append({"fold": k, "n_held_slides": len(held),
                         "chosen_on_training_folds": chosen,
                         "train_f1_at_choice": scored[chosen],
                         "held_f1_at_choice": b["f1"],
                         "held_n_gold": b["n_gold"], "n_skipped": skipped})
        tp += b["tp"]
        npred += b["n_pred"]
        ngold += b["n_gold"]
    transferred = prf(tp, npred, ngold)
    oracle_key = max(keys, key=lambda key: pooled(ps, key)[0]["f1"])
    oracle = pooled(ps, oracle_key)[0]
    chosen_set = sorted({p["chosen_on_training_folds"] for p in per_fold})
    report["arms"][scope][arm] = {
        "n_slides": len(ps),
        "transferred": transferred,
        "oracle_same_slides": oracle,
        "oracle_operating_point": oracle_key,
        "oracle_minus_transferred": round(oracle["f1"] - transferred["f1"], 4),
        "operating_points_chosen_per_fold": chosen_set,
        "chose_one_point_in_every_fold": len(chosen_set) == 1,
        "per_fold": per_fold,
    }
    print(f"[{scope}/{arm}] transferred {transferred['f1']:.4f} "
          f"(P {transferred['precision']:.3f}/R {transferred['recall']:.3f}, "
          f"n_gold {transferred['n_gold']}) | oracle {oracle['f1']:.4f} @{oracle_key} "
          f"| gap {oracle['f1'] - transferred['f1']:+.4f} | chosen {chosen_set}",
          flush=True)
    return report


def threshold_drift(report, fold_of):
    import numpy as np
    import pandas as pd
    keys = report["operating_points"]
    gold = GOLD
    frame = json.loads((EVALUATION / "frame.json").read_text())
    per_slide_probs = {}
    for row in frame["slides"]:
        s = row["sample"]
        hits = sorted(glob.glob(str(CELL_OOF / f"fold*/{s}.npz")))
        if not hits:
            continue
        z = np.load(hits[0], allow_pickle=True)
        classes = [str(c) for c in z["identity_classes"]]
        k = classes.index("Schwann")
        m = dict(zip(z["cell"].astype(str), z["prob"][:, k].astype(np.float32)))
        g = pd.read_parquet(gold / f"{s}.parquet", columns=["cell_id", "x_um"])
        cids = g["cell_id"].astype(str).to_numpy()[g["x_um"].notna().to_numpy()]
        p = np.array([m.get(c, np.nan) for c in cids], dtype=np.float32)
        per_slide_probs[s] = p[np.isfinite(p)]
    if not per_slide_probs:
        raise ValueError("No Cell OOF probabilities available for threshold calibration")
    tgt = [float(k.split("_")[1]) for k in keys]
    allp = np.concatenate(list(per_slide_probs.values()))
    drift = []
    for k in sorted({fold_of[s] for s in per_slide_probs if s in fold_of}):
        tr = np.concatenate([v for s, v in per_slide_probs.items() if fold_of.get(s) != k])
        for f in tgt:
            t_all = float(np.quantile(allp, 1 - f))
            t_tr = float(np.quantile(tr, 1 - f))
            drift.append({"fold": k, "posfrac": f, "tau_all_slides": round(t_all, 8),
                          "tau_training_only": round(t_tr, 8),
                          "abs_rel_shift": round(abs(t_tr - t_all) / max(t_all, 1e-9), 5)})
    rel = np.array([d["abs_rel_shift"] for d in drift])
    report["threshold_calibration_leak_check"] = {
        "what": "posfrac was converted to a global threshold using ALL frame slides; this "
                "recomputes it from training-fold slides only, per fold",
        "n_checks": len(drift),
        "abs_rel_shift": {"p50": round(float(np.median(rel)), 5),
                          "p90": round(float(np.percentile(rel, 90)), 5),
                          "max": round(float(rel.max()), 5)},
        "reading": "a small shift means the operating-point grid itself carried little "
                   "held-out information; a large shift means the transferred number above "
                   "is still optimistic and the grid must be recalibrated per fold",
        "per_fold_posfrac": drift,
    }
    print(f"threshold leak check: |rel shift| p50 {np.median(rel):.5f} "
          f"p90 {np.percentile(rel, 90):.5f} max {rel.max():.5f}", flush=True)

    return report


def main() -> int:
    import argparse
    ap = argparse.ArgumentParser(description="Evaluate the Cell replay operating point across target folds.")
    ap.add_argument("--counts-only", action="store_true", help="Re-pool packaged counts without reading external probabilities or recalculating threshold drift.")
    ap.add_argument("--output", type=Path, default=OUT)
    args = ap.parse_args()
    folds = json.loads(FOLDS.read_text())
    fold_of = {r["sample"]: int(r["fold_he_safe"]) for r in folds["per_sample"]}
    replay = json.loads((EVALUATION / "operator_cell.json").read_text())
    report = transfer_counts(replay, fold_of)
    report["fold_source"] = str(FOLDS)
    report["fold_assignment_sha256"] = folds.get("assignment_sha256")
    report["threshold_calibration_recomputed"] = not args.counts_only
    if not args.counts_only:
        threshold_drift(report, fold_of)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=1) + "\n")
    print(f"wrote {args.output}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
