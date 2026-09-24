#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

NBINS = 200_000
SCORED = 10
CURVE_POINTS = 1500


def excluded_by_sample(labels_root: Path) -> dict[str, set[str]]:
    out: dict[str, set[str]] = {}
    for path in sorted(labels_root.glob("*.parquet")):
        table = pd.read_parquet(path, columns=["cell_id", "identity_excluded"])
        flagged = table.loc[table["identity_excluded"].to_numpy(dtype=bool), "cell_id"].astype(str)
        if len(flagged):
            out[path.stem] = set(flagged)
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--oof-root", type=Path, required=True, help="directory holding fold{0..4}/<sample>.npz")
    parser.add_argument("--labels-root", type=Path, required=True, help="label parquets carrying identity_excluded")
    parser.add_argument("--results", type=Path, required=True, help="scoring JSON with per_arm.<arm-key>")
    parser.add_argument("--arm-key", default="cls_sigma3")
    parser.add_argument("--arm-name", default="cls_sigma3")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--expected-excluded", type=int, default=None)
    args = parser.parse_args()

    files = [(fold, p) for fold in range(5) for p in sorted((args.oof_root / f"fold{fold}").glob("*.npz"))]
    if not files:
        raise SystemExit(f"no held-out arrays under {args.oof_root}")
    excluded = excluded_by_sample(args.labels_root)
    classes = None
    checkpoints: dict[int, set[str]] = {}
    pos = np.zeros((SCORED, NBINS), dtype=np.int64)
    neg = np.zeros((SCORED, NBINS), dtype=np.int64)
    n_rows = 0
    n_excluded = 0
    t0 = time.perf_counter()
    for index, (fold, path) in enumerate(files, 1):
        with np.load(path, allow_pickle=False) as value:
            if classes is None:
                classes = [str(c) for c in value["identity_classes"]]
            checkpoints.setdefault(fold, set()).add(str(value["checkpoint_sha256"]))
            known = value["identity_known_mask"].astype(bool)
            flagged = excluded.get(path.stem)
            if flagged:
                cells = value["cell"].astype(str)
                drop = np.fromiter((c in flagged for c in cells), dtype=bool, count=len(cells))
                if not (value["y"][drop] == classes.index("Schwann")).all():
                    raise SystemExit(f"{path.stem}: an excluded row is not Schwann")
                n_excluded += int((drop & known).sum())
                known &= ~drop
            y = value["y"].astype(np.int64)[known]
            prob = value["prob"][known]
        n_rows += int(known.sum())
        for c in range(SCORED):
            binned = np.minimum((prob[:, c].astype(np.float64) * NBINS).astype(np.int64), NBINS - 1)
            is_pos = y == c
            pos[c] += np.bincount(binned[is_pos], minlength=NBINS)
            neg[c] += np.bincount(binned[~is_pos], minlength=NBINS)
        if index % 25 == 0 or index == len(files):
            print(f"[{index}/{len(files)}] rows={n_rows:,} excluded={n_excluded:,} {time.perf_counter() - t0:.0f}s", flush=True)
    for fold, digests in sorted(checkpoints.items()):
        if len(digests) != 1:
            raise SystemExit(f"fold {fold}: held-out files disagree on the checkpoint")
    if args.expected_excluded is not None and n_excluded != args.expected_excluded:
        raise SystemExit(f"excluded {n_excluded} rows, expected {args.expected_excluded}")

    reference = json.loads(args.results.read_text())["per_arm"][args.arm_key]
    if n_rows != int(reference["n_rows_for_auroc"]):
        raise SystemExit(f"accumulated {n_rows} rows, the results file reports {reference['n_rows_for_auroc']}")
    curves = {}
    for c in range(SCORED):
        name = classes[c]
        p_total = int(pos[c].sum())
        n_total = int(neg[c].sum())
        tp = np.cumsum(pos[c][::-1])
        fp = np.cumsum(neg[c][::-1])
        tpr = np.concatenate([[0.0], tp / p_total])
        fpr = np.concatenate([[0.0], fp / n_total])
        auc = float(np.trapezoid(tpr, fpr)) if hasattr(np, "trapezoid") else float(np.trapz(tpr, fpr))
        walk = fpr + tpr
        take = np.unique(np.concatenate([
            [0], np.searchsorted(walk, np.linspace(0.0, float(walk[-1]), CURVE_POINTS)), [len(walk) - 1]]))
        take = take[take < len(walk)]
        if p_total != int(reference["per_class_auroc"][name]["positives"]) or n_total != int(reference["per_class_auroc"][name]["negatives"]):
            raise SystemExit(f"{name}: positives/negatives differ from the results file")
        curves[name] = {
            "fpr": [round(float(v), 6) for v in fpr[take]],
            "tpr": [round(float(v), 6) for v in tpr[take]],
            "auc_binned": auc,
            "auroc_reported": reference["per_class_auroc"][name]["auroc"],
            "positives": p_total,
            "negatives": n_total,
        }
    payload = {
        "arm": args.arm_name,
        "classes": classes,
        "scored_classes": classes[:SCORED],
        "n_rows": n_rows,
        "n_rows_reported": reference["n_rows_for_auroc"],
        "n_excluded_region_ruler": n_excluded,
        "macro_auroc_reported": reference["macro_ovr_auroc_10"],
        "macro_auc_binned": float(np.mean([curves[c]["auc_binned"] for c in classes[:SCORED]])),
        "bins": NBINS,
        "checkpoint_sha256_by_fold": {str(f): sorted(d)[0] for f, d in sorted(checkpoints.items())},
        "curves": curves,
        "source": "held-out fold{0..4}/*.npz, identity_known_mask rows minus identity_excluded rows",
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=1))
    print(f"wrote {args.out}  rows={n_rows:,} macro_binned={payload['macro_auc_binned']:.5f} "
          f"macro_reported={payload['macro_auroc_reported']:.5f}", flush=True)


if __name__ == "__main__":
    main()
