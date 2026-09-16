#!/usr/bin/env python3

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

import contract as C

DEFAULT_CAP = 400000
DEFAULT_SEED = 42


def stable_priority(sample: str, cells: np.ndarray, seed: int) -> np.ndarray:
    frame = pd.DataFrame({"sample": np.repeat(sample, len(cells)), "cell_id": cells.astype(str)})
    values = pd.util.hash_pandas_object(frame, index=False, categorize=False).to_numpy(np.uint64)
    salt = int.from_bytes(hashlib.sha256(str(seed).encode("ascii")).digest()[:8], "little")
    return values ^ np.uint64(salt)


def build(fold: int, cap: int, seed_base: int) -> dict:
    started = time.time()
    membership = C.fold_membership()[fold]
    train_samples = sorted(membership["train_samples"])
    C.assert_outside_fold(train_samples, fold, f"head selection fold{fold} train list")
    seed = seed_base + fold

    manifest_path = C.head_selection_manifest(fold)
    if manifest_path.exists():
        prior = json.loads(manifest_path.read_text())
        if prior.get("status") != "PASS" or prior.get("fold") != fold:
            raise RuntimeError("existing head selection manifest differs")
        for record in prior.get("samples", {}).values():
            path = Path(record["path"])
            if not path.is_file() or C.sha256_file(path) != record["sha256"]:
                raise RuntimeError(f"head selection artifact hash drift: {path}")
        return {**prior, "status": "ALREADY_COMPLETE"}

    classes = C.CLASSES17
    class_index = {name: i for i, name in enumerate(classes)}
    pools: dict[int, tuple[np.ndarray, np.ndarray, np.ndarray]] = {
        i: (
            np.zeros(0, dtype=np.uint64),
            np.zeros(0, dtype=np.uint16),
            np.zeros(0, dtype="<U1"),
        )
        for i in range(len(classes))
    }
    if len(train_samples) > np.iinfo(np.uint16).max:
        raise RuntimeError("sample code does not fit in uint16")
    sample_codes = {sample: i for i, sample in enumerate(train_samples)}
    population_counts = np.zeros(len(classes), dtype=np.int64)

    for sample in train_samples:
        rec = C.record(sample)
        ids = pd.read_parquet(rec["train_parquet"], columns=["cell_id"])
        ids["cell_id"] = ids["cell_id"].astype(str)
        labels_table = pd.read_parquet(
            C.labels_path(sample), columns=["cell_id", "cell_type_refine"]
        )
        labels_table["cell_id"] = labels_table["cell_id"].astype(str)
        joined = (
            ids.merge(labels_table, on="cell_id", how="inner")
            .sort_values("cell_id", kind="stable")
            .reset_index(drop=True)
        )
        if len(joined) != int(rec["n_keep_cells_own"]):
            raise RuntimeError(
                f"{sample}: joined {len(joined)} cells, contract says {rec['n_keep_cells_own']}"
            )
        cells = joined["cell_id"].to_numpy().astype(str)
        if len(set(cells.tolist())) != len(cells):
            raise RuntimeError(f"duplicate cell ID in {sample}")
        labels = joined["cell_type_refine"].to_numpy().astype(str)
        priorities = stable_priority(sample, cells, seed)
        for name, index in class_index.items():
            positions = np.flatnonzero(labels == name)
            population_counts[index] += len(positions)
            if not len(positions):
                continue
            old_h, old_s, old_c = pools[index]
            new_h = np.concatenate([old_h, priorities[positions]])
            new_s = np.concatenate([
                old_s,
                np.full(len(positions), sample_codes[sample], dtype=np.uint16),
            ])
            width = max(old_c.dtype.itemsize // 4, max(map(len, cells[positions].tolist())))
            new_c = np.concatenate([
                old_c.astype(f"<U{width}"),
                cells[positions].astype(f"<U{width}"),
            ])
            if len(new_h) > cap:
                keep = np.argpartition(new_h, cap - 1)[:cap]
                new_h, new_s, new_c = new_h[keep], new_s[keep], new_c[keep]
            pools[index] = new_h, new_s, new_c
        print(f"[head selection fold{fold}] scanned {sample} ({len(joined):,} cells)", flush=True)

    by_sample: dict[str, list[str]] = {sample: [] for sample in train_samples}
    selected_counts = np.zeros(len(classes), dtype=np.int64)
    for index in range(len(classes)):
        hashes, codes, cells = pools[index]
        order = np.lexsort((cells, codes, hashes))
        hashes, codes, cells = hashes[order], codes[order], cells[order]
        if len(np.unique(hashes)) != len(hashes):

            raise RuntimeError(f"priority hash collision in class {classes[index]}")
        selected_counts[index] = len(cells)
        for code, cell in zip(codes.tolist(), cells.tolist(), strict=True):
            by_sample[train_samples[int(code)]].append(str(cell))

    sample_records = {}
    identity_digest = hashlib.sha256()
    for sample in train_samples:
        cells = np.asarray(sorted(set(by_sample[sample])), dtype="<U64")
        if len(cells):
            width = max(map(len, cells.tolist()))
            cells = cells.astype(f"<U{width}")
        path = C.head_selection_artifact(fold, sample)
        C.atomic_npy(path, cells)
        for cell in cells.tolist():
            payload = f"{sample}\x1f{cell}".encode("utf-8")
            identity_digest.update(len(payload).to_bytes(8, "little"))
            identity_digest.update(payload)
        sample_records[sample] = {
            "path": str(path.resolve()),
            "sha256": C.sha256_file(path),
            "n_cells": int(len(cells)),
        }

    result = {
        "schema_version": "v8.head_selection.v1",
        "status": "PASS",
        "fold": fold,
        "canonical_fold_key": C.CANONICAL_FOLD_KEY,
        "algorithm": "global lowest pandas_hash(sample,cell_id) xor sha256(seed) per refine17 class",
        "pandas_version": pd.__version__,
        "numpy_version": np.__version__,
        "seed": seed,
        "cap_per_class": cap,
        "population_source": "aligned-cell training table INNER JOIN labels_v8, cell_id ascending",
        "label_column": "cell_type_refine",
        "label_mapping_id": C.LABEL_MAPPING_ID,
        "label_mapping_sha256": C.LABEL_MAPPING_SHA256,
        "n_train_samples": len(train_samples),
        "n_test_samples": len(membership["test_samples"]),
        "fold_fingerprint": C.fold_fingerprint(fold),
        "population_per_class": {
            classes[i]: int(population_counts[i]) for i in range(len(classes))
        },
        "selected_per_class": {classes[i]: int(selected_counts[i]) for i in range(len(classes))},
        "n_selected_occurrences": int(selected_counts.sum()),
        "selected_identity_sha256": identity_digest.hexdigest(),
        "samples": sample_records,
        "outer_test_read": False,
        "contract_sha256": C.contract_sha256(),
        "script_sha256": C.sha256_file(Path(__file__)),
        "wall_sec": round(time.time() - started, 1),
        "node": os.environ.get("SLURMD_NODENAME"),
        "job_id": os.environ.get("SLURM_JOB_ID"),
    }
    C.atomic_json(manifest_path, result)
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fold", type=int, choices=range(5))
    parser.add_argument("--task-id", type=int)
    parser.add_argument("--cap", type=int, default=DEFAULT_CAP)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    args = parser.parse_args()
    fold = args.fold if args.fold is not None else args.task_id
    if fold is None:
        raise ValueError("provide --fold or --task-id")
    result = build(int(fold), args.cap, args.seed)
    print(json.dumps({k: v for k, v in result.items() if k != "samples"},
                     indent=2, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
