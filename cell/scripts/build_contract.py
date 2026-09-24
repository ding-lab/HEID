#!/usr/bin/env python3

from __future__ import annotations
from pathlib import Path
_RELEASE = Path(__file__).resolve().parents[2]

import json
import os
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold

sys.path.insert(0, str(Path(__file__).resolve().parent))
from paths import (
    DATA, PC, COHORT_REFERENCE_ROOT, CELL_ROOT, ALIGN_ROOT, PX_UM, patient_of, sha256_file, sha256_list,
)

LABEL_TSV = DATA / "celltype/selected_metadata_kmean_k10_reannot.tsv"
TAXONOMY = _RELEASE / "cell/configs/taxonomy_coarse11.json"
CANDIDATES = CELL_ROOT / "outputs/manifests/cohort_candidates.json"
SCAN = CELL_ROOT / "outputs/manifests/label_scan.json"
ALIAS = CELL_ROOT / "outputs/manifests/label_alias_probe.json"
COHORT_REFERENCE_CONTRACT = COHORT_REFERENCE_ROOT / "outputs/manifests/cohort_contract.json"
LABEL_DIR = CELL_ROOT / "outputs/labels"
OUT = CELL_ROOT / "outputs/manifests/contract.json"

CHUNK = 4_000_000
BARCODE_LEN = 10
SEED = 42
N_SPLITS = 5


def keep_table_path(record: dict) -> Path:
    return Path(record["train_parquet"]).parent / "cell_keep" / f"{record['sample']}_cells.parquet"


def write_alias_labels(alias_map: dict[str, str]) -> dict[str, int]:
    taxonomy = json.loads(TAXONOMY.read_text())
    refine17 = list(taxonomy["source_vocabularies"]["refine17"]["labels"])
    to_coarse = dict(taxonomy["source_vocabularies"]["refine17"]["source_to_identity"])
    vindex = {name: i for i, name in enumerate(refine17)}
    coarse_of_code = np.array([to_coarse[n] for n in refine17], dtype=object)

    wanted = set(alias_map.values())
    held: dict[str, list[tuple[np.ndarray, np.ndarray]]] = defaultdict(list)
    reader = pd.read_csv(LABEL_TSV, sep="\t", usecols=["cell_id", "cell_type_refine"],
                         dtype=str, chunksize=CHUNK)
    n_rows = 0
    for chunk in reader:
        n_rows += len(chunk)
        suffix = chunk["cell_id"].str.slice(BARCODE_LEN + 1)
        mask = suffix.isin(wanted).to_numpy()
        if not mask.any():
            continue
        bc = chunk["cell_id"][mask].str.slice(0, BARCODE_LEN).to_numpy().astype("S10")
        mapped = chunk["cell_type_refine"][mask].map(vindex)
        if mapped.isna().any():
            bad = sorted(set(chunk["cell_type_refine"][mask][mapped.isna()]))
            raise RuntimeError(f"alias row with cell_type_refine outside refine17: {bad[:5]}")
        code = mapped.to_numpy().astype(np.int8)
        names = suffix[mask].to_numpy()
        for name in np.unique(names):
            sel = names == name
            held[str(name)].append((bc[sel], code[sel]))
        print(f"  alias pass: {n_rows:,} rows read", flush=True)

    written = {}
    for sample, suffix in alias_map.items():
        parts = held[suffix]
        cell_id = np.concatenate([p[0] for p in parts])
        code = np.concatenate([p[1] for p in parts])
        if len(np.unique(cell_id)) != len(cell_id):
            raise RuntimeError(f"duplicate barcode under alias suffix {suffix}")
        order = np.argsort(cell_id, kind="stable")
        cell_id, code = cell_id[order], code[order]
        frame = pd.DataFrame({
            "cell_id": np.char.decode(cell_id, "ascii").astype(object),
            "cell_type_refine": np.array(refine17, dtype=object)[code],
            "coarse11": coarse_of_code[code],
        })
        out = LABEL_DIR / f"{sample}.parquet"
        tmp = out.with_name(out.name + f".tmp.{os.getpid()}")
        frame.to_parquet(tmp, index=False)
        os.replace(tmp, out)
        written[sample] = int(len(frame))
        print(f"[alias {sample} <- {suffix}] {len(frame):,} label rows", flush=True)
    return written


def main() -> int:
    cand = json.loads(CANDIDATES.read_text())
    scan = json.loads(SCAN.read_text())
    alias = json.loads(ALIAS.read_text())
    cohort_reference = json.loads(COHORT_REFERENCE_CONTRACT.read_text())
    reference_by_sample = {r["sample"]: r for r in cohort_reference["per_sample"]}

    by_sample = {r["sample"]: r for r in cand["candidates"]}
    scan_by = {r["sample"]: r for r in scan["per_sample"]}
    alias_map = {r["sample"]: r["best_match"]["label_suffix"]
                 for r in alias["results"] if r["verdict"] == "ALIAS"}
    alias_evidence = {r["sample"]: r["best_match"] for r in alias["results"]
                      if r["verdict"] == "ALIAS"}
    no_labels = [r["sample"] for r in alias["results"] if r["verdict"] == "NO_ALIAS"]

    allow = sorted(s for s in cand["candidate_allowlist"]
                   if scan_by[s]["n_label_rows_own"] > 0 or s in alias_map)
    print(f"final allowlist: {len(allow)} "
          f"(candidates {len(cand['candidate_allowlist'])}, alias-recovered {len(alias_map)}, "
          f"dropped no-labels {len(no_labels)})", flush=True)

    alias_rows = write_alias_labels(alias_map) if alias_map else {}


    per_sample = []
    train_sha: dict[str, str] = {}
    per_class_total = defaultdict(int)
    coarse_total = defaultdict(int)
    max_resid = 0.0
    reference_join_agree, reference_join_disagree = 0, []
    for i, sample in enumerate(allow, 1):
        rec = by_sample[sample]
        tp = Path(rec["train_parquet"])
        table = pd.read_parquet(tp)
        resid = max(
            float(np.abs((table["x_um"] + table["dx_um"]) / PX_UM - table["he_px"]).max()),
            float(np.abs((table["y_um"] + table["dy_um"]) / PX_UM - table["he_py"]).max()),
        )
        max_resid = max(max_resid, resid)
        train_sha[sample] = sha256_file(tp)
        train_bc = table["cell_id"].astype(str).to_numpy()

        labels = pd.read_parquet(LABEL_DIR / f"{sample}.parquet")
        lab_bc = labels["cell_id"].to_numpy().astype(str)
        lab_index = pd.Index(lab_bc)
        hit = lab_index.get_indexer(train_bc)
        joined = hit >= 0
        n_join = int(joined.sum())
        refine = labels["cell_type_refine"].to_numpy()[hit[joined]]
        coarse = labels["coarse11"].to_numpy()[hit[joined]]
        per_class = pd.Series(refine).value_counts().to_dict()
        for k, v in per_class.items():
            per_class_total[str(k)] += int(v)
        for k, v in pd.Series(coarse).value_counts().to_dict().items():
            coarse_total[str(k)] += int(v)

        keep_path = keep_table_path(rec)
        keep_tbl = pd.read_parquet(keep_path, columns=["cell_id", "keep"])
        keep_bc = keep_tbl["cell_id"].astype(str).to_numpy()
        owned = lab_index.get_indexer(keep_bc) >= 0
        n_eval_own = int(owned.sum())
        n_keep_own = int((owned & keep_tbl["keep"].to_numpy()).sum())

        n_label_own = int(len(labels))
        entry = {
            "sample": sample,
            "cancer": rec["cancer"],
            "patient": patient_of(sample),
            "fold": None,
            "cohort": "5k",
            "dirname": rec["dirname"],
            "dir_suffix": rec["dir_suffix"],
            "train_parquet": str(tp),
            "keep_parquet": str(keep_path),
            "n_label_rows_own": n_label_own,
            "label_suffix_in_table": alias_map.get(sample, sample),
            "n_v9_training_rows_in_table": int(len(table)),
            "n_v9_evaluated_rows_in_table": int(len(keep_tbl)),
            "n_keep_cells_own": n_join,
            "n_evaluated_cells_own": n_eval_own,
            "n_keep_cells_own_from_keep_table": n_keep_own,
            "label_join_coverage_of_table": round(n_join / len(table), 6),
            "keep_frac_sample_recomputed": round(n_keep_own / n_eval_own, 6) if n_eval_own else None,
            "keep_frac_vs_label_rows": round(n_join / n_label_own, 6) if n_label_own else None,
            "shared_he_group": None,
            "per_class_refine17": {str(k): int(v) for k, v in sorted(per_class.items())},
            "keep_frac_cohort_summary_slide_level": rec["keep_frac_slide"],
            "max_abs_coord_residual_px": resid,
            "coord_identity_holds": resid == 0.0,
        }
        entry["keep_frac_delta_vs_slide_level"] = (
            None if entry["keep_frac_cohort_summary_slide_level"] is None
            else round(entry["keep_frac_sample_recomputed"]
                       - entry["keep_frac_cohort_summary_slide_level"], 6)
        )
        if sample in reference_by_sample:
            if n_join == int(reference_by_sample[sample]["n_keep_cells_own"]):
                reference_join_agree += 1
            else:
                reference_join_disagree.append(
                    {"sample": sample, "Cell": n_join,
                     "cohort_reference": int(reference_by_sample[sample]["n_keep_cells_own"])})
        per_sample.append(entry)
        if i % 25 == 0 or i == len(allow):
            print(f"  measured {i}/{len(allow)}", flush=True)

    if max_resid != 0.0:
        raise RuntimeError(f"coordinate identity broken, max residual {max_resid}")


    groups = defaultdict(list)
    for e in per_sample:
        groups[train_sha[e["sample"]]].append(e["sample"])
    shared = {k: v for k, v in groups.items() if len(v) > 1}
    he_group_of = {}
    shared_records = {}
    for gi, (digest, members) in enumerate(sorted(shared.items(), key=lambda kv: kv[1][0])):
        name = f"he_group_{gi}"
        for m in members:
            he_group_of[m] = name
        entry = {e["sample"]: e for e in per_sample}
        shared_records[name] = {
            "samples": members,
            "training_table_sha256": digest,
            "v9_training_tables_identical": True,
            "n_rows_each": entry[members[0]]["n_v9_training_rows_in_table"],
            "cancers": sorted({entry[m]["cancer"] for m in members}),
            "patients": sorted({entry[m]["patient"] for m in members}),
            "spans_multiple_patients": len({entry[m]["patient"] for m in members}) > 1,
            "per_member_keep_rows": {m: entry[m]["n_keep_cells_own"] for m in members},
        }
    for e in per_sample:
        e["shared_he_group"] = he_group_of.get(e["sample"])


    frame = pd.DataFrame([{"sample": e["sample"], "cancer": e["cancer"],
                           "patient": e["patient"]} for e in per_sample])
    frame = frame.sort_values("sample").reset_index(drop=True)
    multi = frame.groupby("patient")["cancer"].nunique()
    patients_multi_cancer = sorted(multi[multi > 1].index.tolist())

    def split(group_key: pd.Series) -> dict[str, int]:
        sgkf = StratifiedGroupKFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)
        assign = {}
        for fold, (_, test) in enumerate(sgkf.split(frame, frame["cancer"], group_key)):
            for idx in test:
                assign[frame.loc[idx, "sample"]] = fold
        if len(assign) != len(frame):
            raise RuntimeError("split did not cover every sample")
        return assign

    canonical = split(frame["patient"])


    parent = {p: p for p in frame["patient"]}

    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    pat_of = dict(zip(frame["sample"], frame["patient"], strict=True))
    for rec in shared_records.values():
        ps = [pat_of[m] for m in rec["samples"]]
        for p in ps[1:]:
            parent[find(p)] = find(ps[0])
    he_safe_key = frame["patient"].map(lambda p: find(p))
    he_safe = split(he_safe_key)

    for e in per_sample:
        e["fold"] = int(canonical[e["sample"]])
        e["fold_he_safe"] = int(he_safe[e["sample"]])

    def fold_table(assign: dict[str, int]) -> dict:
        rows = defaultdict(lambda: {"samples": 0, "patients": set(), "cancers": defaultdict(int),
                                    "cells": 0})
        for e in per_sample:
            f = assign[e["sample"]]
            rows[f]["samples"] += 1
            rows[f]["patients"].add(e["patient"])
            rows[f]["cancers"][e["cancer"]] += 1
            rows[f]["cells"] += e["n_keep_cells_own"]
        return {str(f): {"n_samples": v["samples"], "n_patients": len(v["patients"]),
                         "n_cells": v["cells"], "per_cancer": dict(sorted(v["cancers"].items()))}
                for f, v in sorted(rows.items())}

    canonical_table = fold_table(canonical)
    he_safe_table = fold_table(he_safe)

    leaked = []
    for name, rec in shared_records.items():
        folds = sorted({canonical[m] for m in rec["samples"]})
        rec["folds_canonical"] = folds
        rec["folds_he_safe"] = sorted({he_safe[m] for m in rec["samples"]})
        rec["split_across_folds_canonical"] = len(folds) > 1
        if len(folds) > 1:
            leaked.extend(rec["samples"])


    seen = defaultdict(set)
    for e in per_sample:
        seen[e["patient"]].add(e["fold"])
    bad = {p: sorted(f) for p, f in seen.items() if len(f) > 1}
    if bad:
        raise RuntimeError(f"patient appears in more than one fold: {bad}")


    excluded = list(cand["excluded"])
    grouped = {k: list(v) for k, v in cand["excluded_grouped_by_cause"].items()}
    grouped["no_labels_in_any_sample_suffix"] = sorted(no_labels)
    for s in no_labels:
        r = dict(by_sample[s])
        r["causes"] = ["no_labels"]
        r["n_training_rows"] = scan_by[s]["n_training_rows"]
        excluded.append(r)

    n_cells_lost_no_labels = int(sum(scan_by[s]["n_training_rows"] for s in no_labels))

    contract = {
        "schema_version": "v8.contract_all5k.v1",
        "built_at_utc": datetime.now(timezone.utc).isoformat(),
        "cohort": {
            "rule": (
                "cohort=='5k' and not lowqc and weak_frac<=0.5 and n_keep>0 and "
                "(field not in blur_rename_map or frac_keep>=0.50) and the slide's cells "
                "carry labels under some sample suffix of the reannotated label table"
            ),
            "rule_source": str(COHORT_REFERENCE_ROOT / "documents/COHORT_v9.md"),
            "n_samples": len(allow),
            "n_patients": int(frame["patient"].nunique()),
            "n_cancers": int(frame["cancer"].nunique()),
            "allowlist": allow,
            "allowlist_sha256": sha256_list(allow),
            "candidate_allowlist_sha256": cand["candidate_allowlist_sha256"],
            "n_candidates_before_label_gate": len(cand["candidate_allowlist"]),
            "excluded": excluded,
            "excluded_grouped_by_cause": grouped,
            "n_training_cells_lost_to_no_labels": n_cells_lost_no_labels,
            "alias_recovered": {
                s: {"label_suffix": alias_map[s], "n_label_rows": alias_rows[s],
                    "evidence": alias_evidence[s]} for s in sorted(alias_map)
            },
            "exclusion_evidence_sources": {
                "weak_correlation_gate": str(ALIGN_ROOT / "_report/weak_corr_gate.csv"),
                "lowqc_registry": str(ALIGN_ROOT / "_report/lowqc_samples.csv"),
                "blur_grading": str(ALIGN_ROOT / "_report/blur_rename_map.tsv"),
                "per_field_stage_and_counts": str(ALIGN_ROOT / "_report/cohort_summary.csv"),
                "label_coverage": str(SCAN),
                "label_alias_probe": str(ALIAS),
            },
            "per_fold_samples": {k: v["n_samples"] for k, v in canonical_table.items()},
            "per_fold_patients": {k: v["n_patients"] for k, v in canonical_table.items()},
            "per_cancer_samples": dict(sorted(frame["cancer"].value_counts().to_dict().items())),
            "v9_root": str(ALIGN_ROOT),
            "directory_index_source": "re-globbed in build_cohort.py; -blur dirs renamed upstream",
        },
        "folds": {
            "policy": (
                f"StratifiedGroupKFold(n_splits={N_SPLITS}, shuffle=True, random_state={SEED}), "
                "group=patient, stratify=cancer, drawn from scratch on the Cell patient set"
            ),
            "not_comparable_to": (
                "This split is NEW. The patient set changed (GBM added, 41 slides gated out, "
                "5 blur slides dropped), so a Cell fold is not the 177-section reference-cohort fold or the cohort-reference "
                "169-slide fold. Numbers computed on this split must never be compared with numbers from the "
                "177-section reference cohort, the necrosis-annotation contract or the cohort-reference set as if they estimated the same quantity."
            ),
            "seed": SEED,
            "per_fold": canonical_table,
            "he_safe_alternative": {
                "why": (
                    "grouping by patient alone lets one H&E image land in two folds when two "
                    "samples share a Xenium run and belong to different patients"
                ),
                "group_key": "connected component of patient union shared-H&E group",
                "per_fold": he_safe_table,
                "n_samples_moved_vs_canonical": int(sum(
                    1 for e in per_sample if e["fold"] != e["fold_he_safe"])),
            },
            "shared_he_slides_split_across_folds_in_canonical": sorted(leaked),
            "patients_mapped_to_multiple_cancers": patients_multi_cancer,
        },
        "coordinates": {
            "frame": "aligned_he he_px / he_py, level-0 pixels of the registered_he image",
            "pixel_size_um": PX_UM,
            "identity_checked": "he_px == (x_um + dx_um)/0.2125 and he_py == (y_um + dy_um)/0.2125",
            "max_abs_residual_px_over_all_samples": max_resid,
            "n_samples_checked": len(allow),
        },
        "labels": {
            "table": str(LABEL_TSV),
            "sha256": scan["label_table"]["sha256"],
            "n_rows_total": scan["label_table"]["n_rows"],
            "n_distinct_sample_suffixes_in_table": scan["label_table"]["n_distinct_sample_suffixes"],
            "column_used": "cell_type_refine",
            "taxonomy": str(TAXONOMY),
            "cell_id_split": {"barcode": "cell_id[:10]", "separator": "cell_id[10] == '_'",
                              "sample": "cell_id[11:]",
                              "why": "sample names contain underscores, so a split on '_' is wrong"},
            "labels_dir": str(LABEL_DIR),
            "kept_cells_per_refine17": dict(sorted(per_class_total.items())),
            "kept_cells_per_coarse11": dict(sorted(coarse_total.items())),
            "n_join_cells_total": int(sum(e["n_keep_cells_own"] for e in per_sample)),
            "label_join_coverage_of_v9_table": {
                "min": min(e["label_join_coverage_of_table"] for e in per_sample),
                "median": float(np.median([e["label_join_coverage_of_table"] for e in per_sample])),
                "max": max(e["label_join_coverage_of_table"] for e in per_sample),
            },
            "n_samples_agreeing_with_v7_n_keep_cells_own": reference_join_agree,
            "v7_join_disagreements": reference_join_disagree,
        },
        "shared_he_slides": {
            "n_groups": len(shared_records),
            "n_samples": int(sum(len(r["samples"]) for r in shared_records.values())),
            "detection": "identical sha256 of the aligned-cell cells_for_training parquet",
            "groups": shared_records,
        },
        "per_sample": per_sample,
        "job_id": os.environ.get("SLURM_JOB_ID"),
    }
    OUT.write_text(json.dumps(contract, indent=2) + "\n")

    print(f"\nwrote {OUT}")
    print(f"n_samples {len(allow)}  n_patients {contract['cohort']['n_patients']}  "
          f"n_cells {contract['labels']['n_join_cells_total']:,}")
    print("per fold:", json.dumps(canonical_table, indent=None)[:600])
    print(f"shared-H&E groups {len(shared_records)}, split across folds: {sorted(set(leaked))}")
    print(f"cohort-reference join agreement {reference_join_agree} samples, disagreements {reference_join_disagree}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
