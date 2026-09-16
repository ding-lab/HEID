#!/usr/bin/env python3

from __future__ import annotations
import os
PROJECTS_ROOT = os.environ.get("PROJECTS_ROOT", "/data/heid")
from pathlib import Path
_RELEASE = Path(__file__).resolve().parents[3]

import argparse
import glob
import hashlib
import json
import sys
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd

INPUT_ROOT = Path(PROJECTS_ROOT)
PREDICTION_ROOT = _RELEASE / "nerve/prediction"
EVALUATION = PREDICTION_ROOT / "data/evaluation"
GOLD = INPUT_ROOT / "nerve/prediction/gold"
MASKS = INPUT_ROOT / "nerve/prediction/masks"
CELL_OOF = INPUT_ROOT / "cell/outputs/oof/cls_sigma3"
sys.path.insert(0, str(PREDICTION_ROOT / "scripts"))
import region_operator as OP

IOU_THR = 0.10
TARGET_POS_FRAC = [0.001, 0.002, 0.003, 0.00434, 0.006, 0.008, 0.012, 0.018,
                   0.025, 0.035, 0.05]
MAX_SLIDE_POS_FRAC = 0.10


def iou_match(pred_sets, gold_sets, thr):
    used, matched, pairs = set(), 0, []
    for gi, g in enumerate(gold_sets):
        best, bi = 0.0, None
        for i, ps in enumerate(pred_sets):
            if i in used or not ps:
                continue
            inter = len(ps & g)
            if not inter:
                continue
            iou = inter / len(ps | g)
            if iou > best:
                best, bi = iou, i
        if best >= thr:
            matched += 1
            used.add(bi)
            pairs.append((gi, bi, best))
    return matched, used, pairs


def score(pred_sets, gold_sets, ignore_sets, thr=IOU_THR):
    iou_tp, used, _ = iou_match(pred_sets, gold_sets, thr)

    gold_union = set().union(*gold_sets) if gold_sets else set()
    det_tp = sum(1 for ps in pred_sets if ps & gold_union)
    pred_union = set().union(*pred_sets) if pred_sets else set()
    det_rec = sum(1 for g in gold_sets if g & pred_union)

    n_ignored = 0
    if ignore_sets:
        for i, ps in enumerate(pred_sets):
            if i in used or not ps or (ps & gold_union):
                continue
            for ig in ignore_sets:
                if ps & ig and len(ps & ig) / len(ps | ig) >= thr:
                    n_ignored += 1
                    break
    return dict(n_pred=len(pred_sets), n_pred_ignored=n_ignored,
                n_pred_scored=len(pred_sets) - n_ignored, n_gold=len(gold_sets),
                iou_tp=iou_tp, det_tp=det_tp, det_rec=det_rec)


def load_probs(arm, sample, cell_ids):
    if arm != "CELL":
        return None, {"error": f"arm {arm} needs no probability"}
    hits = sorted(glob.glob(str(CELL_OOF / f"fold*/{sample}.npz")))
    if not hits:
        return None, {"error": "no Cell OOF file"}
    z = np.load(hits[0], allow_pickle=True)
    classes = [str(c) for c in z["identity_classes"]]
    if "Schwann" not in classes:
        return None, {"error": f"Schwann not in identity_classes: {classes}"}
    k = classes.index("Schwann")
    fold = int(z["fold"])
    pv = z["prob"][:, k].astype(np.float32)
    m = dict(zip(z["cell"].astype(str), pv))
    p = np.array([m.get(c, np.nan) for c in cell_ids], dtype=np.float32)
    return p, {"source": hits[0], "schwann_index": k,
               "n_rows_in_file": int(z["prob"].shape[0]), "fold_in_file": fold,
               "n_covered": int(np.isfinite(p).sum())}


def calibrate_taus(arm, rows):
    pool = []
    for r in rows:
        g = pd.read_parquet(GOLD / f"{r['sample']}.parquet", columns=["cell_id", "x_um"])
        keep = g["x_um"].notna().to_numpy()
        cids = g["cell_id"].astype(str).to_numpy()[keep]
        p, _ = load_probs(arm, r["sample"], cids)
        if p is None:
            continue
        pool.append(p[np.isfinite(p)])
    if not pool:
        return [], {"error": "no probabilities in frame"}
    allp = np.concatenate(pool)
    taus, table = [], []
    for f in TARGET_POS_FRAC:
        tau = float(np.quantile(allp, 1.0 - f))
        taus.append(tau)
        table.append({"target_pos_frac": f, "tau": round(tau, 8),
                      "achieved_pos_frac": round(float((allp > tau).mean()), 6)})
    meta = {"n_cells_pooled": int(len(allp)),
            "prob_quantiles": {q: round(float(np.quantile(allp, q)), 8)
                               for q in (0.5, 0.9, 0.99, 0.999, 0.9999)},
            "prob_max": round(float(allp.max()), 8),
            "grid": table}
    return taus, meta


def process(job):
    arm, row, taus = job
    sample, group, panel = row["sample"], row["group"], row["panel"]
    t0 = time.time()
    g = pd.read_parquet(GOLD / f"{sample}.parquet")
    has_xy = g["x_um"].notna().to_numpy() & g["y_um"].notna().to_numpy()
    n_no_xy = int((~has_xy).sum())
    g = g[has_xy].reset_index(drop=True)
    cell_ids = g["cell_id"].astype(str).to_numpy()
    ax, ay = g["x_um"].to_numpy(np.float64), g["y_um"].to_numpy(np.float64)

    z = np.load(MASKS / group / f"{sample}_mask.npz", allow_pickle=True)
    grid = {"x_min": float(z["x_min_um"]), "y_min": float(z["y_min_um"]),
            "nx": int(z["nx"]), "ny": int(z["ny"])}

    gold_sets = [set(np.where(g["gold_cluster_id"].to_numpy() == cl)[0].tolist())
                 for cl in pd.unique(g["gold_cluster_id"].dropna())]
    ignore_sets = [set(np.where(g["ignore_region_id"].to_numpy() == rid)[0].tolist())
                   for rid in pd.unique(g["ignore_region_id"].dropna())]

    out = {"sample": sample, "group": group, "panel": panel, "fold": row["fold"],
           "n_cells": int(len(g)), "n_cells_without_xy": n_no_xy,
           "n_gold": len(gold_sets), "n_ignore_regions": len(ignore_sets),
           "n_true_schwann": int(g["is_schwann"].sum()), "per_tau": {}}

    def run_once(is_sch):
        cg, clusters, meta = OP.run_operator(ax, ay, is_sch, panel=panel, grid=grid)
        rid = OP.membership(cg, ax, ay, grid["x_min"], grid["y_min"])
        pred_sets = [set(np.where(rid == c["cluster_id"])[0].tolist()) for c in clusters]
        s = score(pred_sets, gold_sets, ignore_sets)
        s["operator"] = {k: v for k, v in meta.items() if k not in ("x_min", "y_min")}
        return s

    if arm == "GOLD":
        out["per_tau"]["true_labels"] = run_once(g["is_schwann"].to_numpy())
    elif arm == "WHOLE_FIELD":

        s = score([set(range(len(g)))], gold_sets, ignore_sets)
        s["operator"] = {"note": "synthetic: whole slide = one region, operator bypassed"}
        out["per_tau"]["whole_slide"] = s
    else:
        p, cov = load_probs(arm, sample, cell_ids)
        out["prob_coverage"] = cov
        if p is None:
            out["skipped"] = cov.get("error", "no probability")
            return out
        finite = np.isfinite(p)
        out["prob_stats"] = {
            "n_covered": int(finite.sum()), "frac_covered": round(float(finite.mean()), 6),
            "p50": float(np.nanpercentile(p, 50)) if finite.any() else None,
            "p99": float(np.nanpercentile(p, 99)) if finite.any() else None,
            "max": float(np.nanmax(p)) if finite.any() else None,
        }
        pf = np.where(finite, p, -1.0)
        for frac, tau in zip(TARGET_POS_FRAC, taus):
            key = f"posfrac_{frac:.5f}"
            sel = pf > tau
            slide_frac = float(sel.mean())
            if slide_frac > MAX_SLIDE_POS_FRAC:
                out["per_tau"][key] = {"skipped_degenerate": True, "tau": tau,
                                       "slide_pos_frac": round(slide_frac, 5),
                                       "limit": MAX_SLIDE_POS_FRAC}
                continue
            s = run_once(sel)
            s["tau"] = tau
            s["n_pred_schwann_cells"] = int(sel.sum())
            s["slide_pos_frac"] = round(slide_frac, 6)
            out["per_tau"][key] = s
    out["wall_sec"] = round(time.time() - t0, 2)
    return out


def prf(tp, n_pred, n_gold, rec):
    prec = tp / n_pred if n_pred else 0.0
    recall = rec / n_gold if n_gold else 0.0
    f1 = 2 * prec * recall / (prec + recall) if prec + recall else 0.0
    return dict(precision=round(prec, 4), recall=round(recall, 4), f1=round(f1, 4),
                n_pred=n_pred, n_gold=n_gold, tp=tp)


def pool_key(samples, key):
    tot = {k: 0 for k in ("n_pred", "n_pred_ignored", "n_pred_scored", "n_gold",
                          "iou_tp", "det_tp", "det_rec")}
    n_used, n_skipped = 0, 0
    for s in samples:
        blk = s["per_tau"].get(key)
        if blk is None:
            continue
        if blk.get("skipped_degenerate"):
            n_skipped += 1
            continue
        n_used += 1
        for k in tot:
            tot[k] += blk[k]
    if n_skipped:
        tot["_n_slides_skipped_degenerate"] = n_skipped
    return {
        "n_slides": n_used, "n_slides_skipped_degenerate": n_skipped,
        f"iou>={IOU_THR:.2f}": prf(tot["iou_tp"], tot["n_pred_scored"], tot["n_gold"],
                                   tot["iou_tp"]),
        "detection": prf(tot["det_tp"], tot["n_pred_scored"], tot["n_gold"], tot["det_rec"]),
        "n_pred_raw": tot["n_pred"], "n_pred_ignored": tot["n_pred_ignored"],
        "n_pred_over_n_gold": round(tot["n_pred_scored"] / tot["n_gold"], 4) if tot["n_gold"] else None,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", required=True, choices=["CELL", "GOLD", "WHOLE_FIELD"])
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()

    frame = json.loads((EVALUATION / "frame.json").read_text())
    if frame["status"] != "HEADLINE":
        print(f"frame status is {frame['status']} -- refusing to score", flush=True)
        return 5
    rows = frame["slides"]
    print(f"arm {args.arm}: {len(rows)} slides, {args.workers} workers", flush=True)

    taus, tau_meta = ([], None)
    if args.arm == "CELL":
        t0 = time.time()
        taus, tau_meta = calibrate_taus(args.arm, rows)
        print(f"tau calibration ({time.time()-t0:.0f}s): "
              f"{json.dumps(tau_meta['grid'])}", flush=True)
        if not taus:
            print("no probabilities -- aborting", flush=True)
            return 6

    jobs = [(args.arm, r, taus) for r in rows]
    with Pool(args.workers) as pool:
        results = pool.map(process, jobs)

    keys = sorted({k for r in results for k in r["per_tau"]})
    pooled = {k: pool_key(results, k) for k in keys}
    report = {
        "schema": "heid.nerve.operator_run",
        "arm": args.arm,
        "frame_allowlist_sha256": frame["allowlist_sha256"],
        "operator": "gene-blind region_operator.py; expression-dependent small-core rescue is unavailable",
        "metric": f"greedy IoU over cell-id sets, threshold {IOU_THR}, pooled counts, "
                  f"zero-gold slides INCLUDED",
        "operating_point_axis": "predicted-positive fraction (raw tau does not transfer "
                                "between arms; see TARGET_POS_FRAC in this script)",
        "tau_calibration": tau_meta,
        "max_slide_pos_frac": MAX_SLIDE_POS_FRAC,
        "pooled": pooled,
        "n_slides_scored": len(results),
        "n_slides_skipped": sum(1 for r in results if "skipped" in r),
        "per_sample": results,
    }
    if args.arm == "CELL":
        f1s = {k: pooled[k][f"iou>={IOU_THR:.2f}"]["f1"] for k in keys}
        best = max(f1s, key=f1s.get)
        ends = (f"posfrac_{TARGET_POS_FRAC[0]:.5f}", f"posfrac_{TARGET_POS_FRAC[-1]:.5f}")
        report["best_operating_point"] = best
        report["best_f1"] = f1s[best]
        report["best_at_grid_endpoint"] = best in ends
        report["best_adjacent_to_degenerate_guard"] = bool(
            pooled[best]["n_slides_skipped_degenerate"] > 0)
        report["f1_range_over_grid"] = round(max(f1s.values()) - min(f1s.values()), 6)
        ng = {k: pooled[k][f"iou>={IOU_THR:.2f}"]["n_gold"] for k in keys}
        report["n_gold_constant_across_tau"] = len(set(ng.values())) == 1
        report["n_gold_values"] = sorted(set(ng.values()))

    out = EVALUATION / f"operator_{args.arm.lower()}.json"
    out.write_text(json.dumps(report, indent=1) + "\n")
    print(json.dumps({k: v for k, v in report.items()
                      if k not in ("per_sample", "pooled")}, indent=1), flush=True)
    for k in keys:
        b = pooled[k]
        print(f"  {k:>12}  IoU-F1 {b[f'iou>={IOU_THR:.2f}']['f1']:.4f} "
              f"(P {b[f'iou>={IOU_THR:.2f}']['precision']:.3f} / "
              f"R {b[f'iou>={IOU_THR:.2f}']['recall']:.3f})  "
              f"det-F1 {b['detection']['f1']:.4f}  n_pred/n_gold {b['n_pred_over_n_gold']}",
              flush=True)
    print(f"wrote {out}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
