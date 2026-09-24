#!/usr/bin/env python3

from __future__ import annotations
import os
PROJECTS_ROOT = os.environ.get("PROJECTS_ROOT", "/data/heid")

import json
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

PC = Path(PROJECTS_ROOT + "/cell")
CELL_ROOT = PC
DATA = Path(PROJECTS_ROOT + "/cell/inputs")
LABEL_TSV = DATA / "celltype/selected_metadata_kmean_k10_reannot.tsv"
CANDIDATES = CELL_ROOT / "outputs/manifests/cohort_candidates.json"
SCAN = CELL_ROOT / "outputs/manifests/label_scan.json"
OUT_JSON = CELL_ROOT / "outputs/manifests/label_alias_probe.json"

CHUNK = 4_000_000
BARCODE_LEN = 10


def main() -> int:
    cand = json.loads(CANDIDATES.read_text())
    scan = json.loads(SCAN.read_text())
    by_sample = {r["sample"]: r for r in cand["candidates"]}
    per = {r["sample"]: r for r in scan["per_sample"]}
    zero = list(scan["samples_with_zero_label_rows"])
    matched = {s for s, r in per.items() if r["n_label_rows_own"] > 0}
    print(f"{len(zero)} zero-label slides, {len(matched)} matched suffixes", flush=True)


    train = {}
    for sample in zero:
        table = pd.read_parquet(by_sample[sample]["train_parquet"], columns=["cell_id"])
        train[sample] = np.sort(table["cell_id"].astype(str).to_numpy().astype("S10"))
        print(f"  {sample}: {len(train[sample]):,} training barcodes", flush=True)


    held: dict[str, list[np.ndarray]] = defaultdict(list)
    n_rows = 0
    reader = pd.read_csv(LABEL_TSV, sep="\t", usecols=["cell_id"], dtype=str, chunksize=CHUNK)
    for chunk in reader:
        n_rows += len(chunk)
        suffix = chunk["cell_id"].str.slice(BARCODE_LEN + 1)
        keep = (~suffix.isin(matched)).to_numpy()
        if not keep.any():
            continue
        names = suffix[keep].to_numpy()
        bc = chunk["cell_id"][keep].str.slice(0, BARCODE_LEN).to_numpy().astype("S10")
        order = np.argsort(names, kind="stable")
        names, bc = names[order], bc[order]
        edges = np.flatnonzero(np.r_[True, names[1:] != names[:-1]])
        for start, stop in zip(edges, np.r_[edges[1:], len(names)], strict=True):
            held[names[start]].append(bc[start:stop])
        print(f"  {n_rows:,} rows read, {len(held)} unmatched suffixes seen", flush=True)

    pool = {k: np.unique(np.concatenate(v)) for k, v in held.items()}
    print(f"unmatched suffixes in the label table: {len(pool)}", flush=True)

    results = []
    for sample in zero:
        t = train[sample]
        scores = []
        for other, bc in pool.items():
            n = int(np.isin(t, bc, assume_unique=True).sum())
            if n:
                scores.append({"label_suffix": other, "n_shared_barcodes": n,
                               "frac_of_training": round(n / len(t), 6),
                               "n_label_barcodes": int(len(bc))})
        scores.sort(key=lambda r: -r["n_shared_barcodes"])
        best = scores[0] if scores else None
        results.append({
            "sample": sample,
            "cancer": by_sample[sample]["cancer"],
            "n_training_barcodes": int(len(t)),
            "n_unmatched_suffixes_with_any_overlap": len(scores),
            "best_match": best,
            "top5": scores[:5],
            "verdict": (
                "ALIAS" if best and best["frac_of_training"] >= 0.5
                else "NO_ALIAS"
            ),
        })
        print(f"[{sample}] best={best}", flush=True)

    payload = {
        "schema_version": "v8.label_alias_probe.v1",
        "built_at_utc": datetime.now(timezone.utc).isoformat(),
        "label_table": str(LABEL_TSV),
        "n_label_rows_read": n_rows,
        "n_zero_label_slides": len(zero),
        "n_unmatched_suffixes": len(pool),
        "unmatched_suffixes": sorted(pool),
        "alias_threshold_frac_of_training": 0.5,
        "n_alias": sum(1 for r in results if r["verdict"] == "ALIAS"),
        "results": results,
        "job_id": os.environ.get("SLURM_JOB_ID"),
    }
    OUT_JSON.write_text(json.dumps(payload, indent=2) + "\n")
    print(f"wrote {OUT_JSON}", flush=True)
    print(f"ALIAS verdicts: {payload['n_alias']} / {len(results)}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
