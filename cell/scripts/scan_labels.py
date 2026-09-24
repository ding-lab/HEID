#!/usr/bin/env python3

from __future__ import annotations
from pathlib import Path
_RELEASE = Path(__file__).resolve().parents[2]
import os
PROJECTS_ROOT = os.environ.get("PROJECTS_ROOT", "/data/heid")

import hashlib
import json
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

PC = Path(PROJECTS_ROOT + "/cell")
COHORT_REFERENCE_ROOT = PC / "inputs/cohort_reference"
CELL_ROOT = PC
DATA = Path(PROJECTS_ROOT + "/cell/inputs")
LABEL_TSV = DATA / "celltype/selected_metadata_kmean_k10_reannot.tsv"
TAXONOMY = _RELEASE / "cell/configs/taxonomy_coarse11.json"
CANDIDATES = CELL_ROOT / "outputs/manifests/cohort_candidates.json"
COHORT_REFERENCE_LABELS = COHORT_REFERENCE_ROOT / "outputs/labels"
OUT_DIR = CELL_ROOT / "outputs/labels"
OUT_JSON = CELL_ROOT / "outputs/manifests/label_scan.json"

CHUNK = 4_000_000
BARCODE_LEN = 10
PX_UM = 0.2125
TRAIN_COLUMNS = ["cell_id", "he_px", "he_py", "x_um", "y_um", "dx_um", "dy_um", "block"]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 22), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> int:
    cand = json.loads(CANDIDATES.read_text())
    allow = list(cand["candidate_allowlist"])
    allow_set = set(allow)
    by_sample = {r["sample"]: r for r in cand["candidates"]}
    print(f"{len(allow)} candidate slides", flush=True)

    taxonomy = json.loads(TAXONOMY.read_text())
    refine17 = list(taxonomy["source_vocabularies"]["refine17"]["labels"])
    to_coarse = dict(taxonomy["source_vocabularies"]["refine17"]["source_to_identity"])
    vindex = {name: i for i, name in enumerate(refine17)}
    coarse_of_code = np.array([to_coarse[name] for name in refine17], dtype=object)


    train_stats = {}
    train_barcodes = {}
    max_resid_all = 0.0
    for i, sample in enumerate(allow, 1):
        path = Path(by_sample[sample]["train_parquet"])
        table = pd.read_parquet(path)
        if list(table.columns) != TRAIN_COLUMNS:
            raise RuntimeError(f"unexpected aligned-cell training schema for {sample}: {list(table.columns)}")
        cid = table["cell_id"].astype(str).to_numpy()
        n_dup = int(len(cid) - len(np.unique(cid)))
        resid = max(
            float(np.abs((table["x_um"] + table["dx_um"]) / PX_UM - table["he_px"]).max()),
            float(np.abs((table["y_um"] + table["dy_um"]) / PX_UM - table["he_py"]).max()),
        )
        max_resid_all = max(max_resid_all, resid)
        train_barcodes[sample] = np.sort(cid.astype("S10"))
        train_stats[sample] = {
            "n_training_rows": int(len(table)),
            "n_duplicate_cell_id": n_dup,
            "max_abs_coord_residual_px": resid,
            "n_blocks": int(table["block"].nunique()),
        }
        if i % 25 == 0 or i == len(allow):
            print(f"  training tables {i}/{len(allow)} (max residual so far {max_resid_all})",
                  flush=True)


    size = LABEL_TSV.stat().st_size
    print(f"hashing the {size/1e9:.2f} GB label table", flush=True)
    label_sha = sha256_file(LABEL_TSV)
    print(f"  sha256 {label_sha}", flush=True)

    barcodes: dict[str, list[np.ndarray]] = defaultdict(list)
    codes: dict[str, list[np.ndarray]] = defaultdict(list)
    unmapped: dict[str, dict] = defaultdict(lambda: defaultdict(int))
    suffix_counts: dict[str, int] = defaultdict(int)
    whole_table = defaultdict(int)
    n_rows = 0
    n_cohort_rows = 0

    reader = pd.read_csv(
        LABEL_TSV, sep="\t", usecols=["cell_id", "cell_type_refine"], dtype=str, chunksize=CHUNK
    )
    for chunk in reader:
        n_rows += len(chunk)
        for name, count in chunk["cell_type_refine"].value_counts().items():
            whole_table[str(name)] += int(count)
        suffix = chunk["cell_id"].str.slice(BARCODE_LEN + 1)
        for name, count in suffix.value_counts().items():
            suffix_counts[str(name)] += int(count)
        mask = suffix.isin(allow_set).to_numpy()
        if not mask.any():
            print(f"  {n_rows:,} rows read (no candidate row in this chunk)", flush=True)
            continue
        n_cohort_rows += int(mask.sum())
        sub_id = chunk["cell_id"][mask]
        sep = sub_id.str.slice(BARCODE_LEN, BARCODE_LEN + 1).to_numpy()
        if (sep != "_").any():
            raise RuntimeError("cell_id[10] is not '_' on a candidate row")
        bc_arr = sub_id.str.slice(0, BARCODE_LEN).to_numpy().astype("S10")
        names = suffix[mask].to_numpy()
        refine_vals = chunk["cell_type_refine"][mask]
        mapped = refine_vals.map(vindex)
        bad_mask = mapped.isna().to_numpy()
        if bad_mask.any():
            for nm, rv in zip(names[bad_mask], refine_vals.to_numpy()[bad_mask], strict=True):
                unmapped[str(nm)][str(rv)] += 1
        rc = mapped.fillna(-1).to_numpy().astype(np.int8)
        order = np.argsort(names, kind="stable")
        names, bc_arr, rc = names[order], bc_arr[order], rc[order]
        edges = np.flatnonzero(np.r_[True, names[1:] != names[:-1]])
        for start, stop in zip(edges, np.r_[edges[1:], len(names)], strict=True):
            barcodes[names[start]].append(bc_arr[start:stop])
            codes[names[start]].append(rc[start:stop])
        print(f"  {n_rows:,} rows read, {n_cohort_rows:,} candidate rows held", flush=True)

    print(f"label table: {n_rows:,} rows, {len(suffix_counts)} distinct sample suffixes",
          flush=True)


    OUT_DIR.mkdir(parents=True, exist_ok=True)
    records = []
    reference_compared = 0
    reference_mismatch = []
    per_class_total = defaultdict(int)
    for sample in allow:
        n_label = int(suffix_counts.get(sample, 0))
        stats = dict(train_stats[sample])
        stats["sample"] = sample
        stats["cancer"] = by_sample[sample]["cancer"]
        stats["n_label_rows_own"] = n_label
        stats["n_unmapped_refine_rows"] = int(sum(unmapped.get(sample, {}).values()))
        stats["unmapped_refine_values"] = dict(sorted(unmapped.get(sample, {}).items()))

        if n_label == 0:
            stats.update(
                n_join_cells=0, label_join_coverage_of_training_table=0.0,
                labels_parquet=None, per_class_refine17={},
            )
            records.append(stats)
            print(f"[{sample}] NO LABEL ROWS", flush=True)
            continue

        cell_id = np.concatenate(barcodes[sample])
        code = np.concatenate(codes[sample])
        if len(np.unique(cell_id)) != len(cell_id):
            raise RuntimeError(f"duplicate barcode inside one sample: {sample}")
        order = np.argsort(cell_id, kind="stable")
        cell_id, code = cell_id[order], code[order]

        n_join = int(np.isin(train_barcodes[sample], cell_id, assume_unique=True).sum())
        stats["n_join_cells"] = n_join
        stats["label_join_coverage_of_training_table"] = round(
            n_join / stats["n_training_rows"], 6) if stats["n_training_rows"] else 0.0

        refine = np.where(code >= 0, np.array(refine17 + ["<unmapped>"], dtype=object)[code], None)
        coarse = np.where(code >= 0,
                          np.array(list(coarse_of_code) + [None], dtype=object)[code], None)
        frame = pd.DataFrame(
            {
                "cell_id": np.char.decode(cell_id, "ascii").astype(object),
                "cell_type_refine": refine,
                "coarse11": coarse,
            }
        )
        for name, count in frame["cell_type_refine"].value_counts().items():
            per_class_total[str(name)] += int(count)
        stats["per_class_refine17"] = {
            str(k): int(v) for k, v in frame["cell_type_refine"].value_counts().items()
        }

        out = OUT_DIR / f"{sample}.parquet"
        tmp = out.with_name(out.name + f".tmp.{os.getpid()}")
        frame.to_parquet(tmp, index=False)
        os.replace(tmp, out)
        stats["labels_parquet"] = str(out)
        stats["labels_sha256"] = sha256_file(out)

        prev_path = COHORT_REFERENCE_LABELS / f"{sample}.parquet"
        if prev_path.is_file():
            prev = pd.read_parquet(prev_path)
            same = bool(
                prev.sort_values("cell_id", kind="stable").reset_index(drop=True)
                .equals(frame.sort_values("cell_id", kind="stable").reset_index(drop=True))
            )
            stats["matches_v7_labels_v9"] = same
            reference_compared += 1
            if not same:
                reference_mismatch.append(sample)
        else:
            stats["matches_v7_labels_v9"] = None

        records.append(stats)
        print(f"[{sample}] label {n_label:,}  train {stats['n_training_rows']:,}  "
              f"join {n_join:,}  cov {stats['label_join_coverage_of_training_table']}",
              flush=True)

    zero = [r["sample"] for r in records if r["n_label_rows_own"] == 0]
    cov = [r["label_join_coverage_of_training_table"] for r in records
           if r["n_label_rows_own"] > 0]

    payload = {
        "schema_version": "v8.label_scan.v1",
        "built_at_utc": datetime.now(timezone.utc).isoformat(),
        "candidates_manifest": str(CANDIDATES),
        "candidate_allowlist_sha256": cand["candidate_allowlist_sha256"],
        "label_table": {"path": str(LABEL_TSV), "sha256": label_sha, "size_bytes": size,
                        "n_rows": n_rows,
                        "n_distinct_sample_suffixes": len(suffix_counts)},
        "taxonomy": {"path": str(TAXONOMY), "sha256": sha256_file(TAXONOMY)},
        "n_candidates": len(allow),
        "n_rows_in_candidate_samples": n_cohort_rows,
        "max_abs_coord_residual_px_over_all_candidates": max_resid_all,
        "n_training_rows_total": int(sum(r["n_training_rows"] for r in records)),
        "n_join_cells_total": int(sum(r["n_join_cells"] for r in records)),
        "samples_with_zero_label_rows": zero,
        "join_coverage": {
            "min": min(cov) if cov else None,
            "median": float(np.median(cov)) if cov else None,
            "max": max(cov) if cov else None,
        },
        "n_samples_compared_against_v7_labels_v9": reference_compared,
        "v7_labels_mismatch": reference_mismatch,
        "whole_table_refine17_counts": dict(sorted(whole_table.items())),
        "candidate_refine17_counts": dict(sorted(per_class_total.items())),
        "per_sample": records,
        "job_id": os.environ.get("SLURM_JOB_ID"),
        "node": os.environ.get("SLURMD_NODENAME"),
    }
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(payload, indent=2) + "\n")
    print(f"wrote {OUT_JSON}", flush=True)
    print(f"zero-label slides: {zero}", flush=True)
    print(f"cohort-reference label files compared {reference_compared}, mismatches {reference_mismatch}", flush=True)
    print(f"max coord residual over all candidates: {max_resid_all}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
