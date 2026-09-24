#!/usr/bin/env python3
from __future__ import annotations
import os
PROJECTS_ROOT = os.environ.get("PROJECTS_ROOT", "/data/heid")
from pathlib import Path
_RELEASE = Path(__file__).resolve().parents[2]

import argparse
import glob
import json

import numpy as np
import pandas as pd
import torch

CELL_ROOT = _RELEASE / "cell"
SCHWANN = 9
IGNORE = -100
TOL_MACRO, TOL_SCHWANN = 0.0044, 0.0048
LABEL_KEYS = ("y", "identity_known_mask", "lineage_y", "necrosis_target", "nec_known_mask")
OOF_KEYS = ("cell", "y", "identity_known_mask", "source_label", "nec_target", "nec_known_mask")


def load(path: Path) -> dict:
    return torch.load(path, map_location="cpu", weights_only=False, mmap=True)


def excluded_cells(labels_root: Path) -> dict[str, set[str]]:
    out: dict[str, set[str]] = {}
    for path in glob.glob(str(labels_root / "*.parquet")):
        table = pd.read_parquet(path, columns=["cell_id", "identity_excluded"])
        excluded = table.loc[table["identity_excluded"].to_numpy(dtype=bool), "cell_id"].astype(str)
        if len(excluded):
            out[Path(path).stem] = set(excluded)
    return out


def scratch_root(arm: str) -> Path:
    config = json.loads((CELL_ROOT / "configs" / f"arm_{arm}.json").read_text())
    os.environ.setdefault("RELEASE_ROOT", str(_RELEASE))
    workspace = Path(os.path.expandvars(config["workspace_root"]))
    return workspace / config["outputs"]["scratch"]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--arm", required=True)
    parser.add_argument("--reference-arm", default="cls_sigma3_all_schwann")
    parser.add_argument("--fold", type=int, default=0)
    parser.add_argument("--null", action="store_true",
                        help="the arm carries no label change and must reproduce the reference within tolerance")
    parser.add_argument("--labels-root", type=Path, default=Path(PROJECTS_ROOT) / "cell/outputs/labels_region_schwann")
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()
    arm, reference, k = args.arm, args.reference_arm, args.fold
    fails: list[str] = []
    out: dict = {"arm": arm, "reference_arm": reference, "fold": k}
    outputs = CELL_ROOT / "outputs"

    summary = json.loads((outputs / f"results/{arm}/fold{k}_summary.json").read_text())
    reference_summary = json.loads((outputs / f"results/{reference}/fold{k}_summary.json").read_text())
    if summary["train_row_set_sha256"] != reference_summary["train_row_set_sha256"]:
        fails.append("train_row_set_sha256 differs")
    if summary["shared_row_identity"] != reference_summary["shared_row_identity"]:
        fails.append("shared_row_identity differs")
    stats, reference_stats = summary["statistics"], reference_summary["statistics"]
    out["necrosis_pos_weight"] = [stats["necrosis_pos_weight"], reference_stats["necrosis_pos_weight"]]
    if stats["necrosis_pos_weight"] != reference_stats["necrosis_pos_weight"]:
        fails.append("necrosis_pos_weight differs")
    masked = int(summary.get("identity_exclusion", {}).get("n_masked_train_rows", 0))
    out["n_masked_train_rows"] = masked
    for name, count in reference_stats["identity_counts"].items():
        want = count - masked if name == "Schwann" else count
        if stats["identity_counts"][name] != want:
            fails.append(f"identity_counts[{name}] {stats['identity_counts'][name]} != {want}")
    if args.null and masked:
        fails.append("null arm masked rows")

    excluded = {} if args.null else excluded_cells(args.labels_root)
    caches = sorted(glob.glob(str(scratch_root(arm) / f"fold{k}/selected_features/*.pt")))
    reference_cache_dir = scratch_root(reference) / f"fold{k}/selected_features"
    n_reference_caches = len(glob.glob(str(reference_cache_dir / "*.pt")))
    out["n_caches"] = len(caches)
    out["n_caches_reference"] = n_reference_caches
    if len(caches) != n_reference_caches:
        fails.append(f"{len(caches)} caches for {arm} but {n_reference_caches} for {reference}")
    n_diff_rows = 0
    for index, path in enumerate(caches):
        cache, reference_cache = load(Path(path)), load(reference_cache_dir / Path(path).name)
        if list(cache["cell"]) != list(reference_cache["cell"]):
            fails.append(f"{Path(path).stem}: cache cells differ")
            continue
        flagged = excluded.get(Path(path).stem, set())
        mask = np.array([c in flagged for c in cache["cell"]], dtype=bool)
        for key in LABEL_KEYS:
            value, reference_value = cache[key].numpy(), reference_cache[key].numpy()
            if key in ("y", "lineage_y"):
                if not np.array_equal(value[~mask], reference_value[~mask]) or (mask.any() and not (value[mask] == IGNORE).all()):
                    fails.append(f"{Path(path).stem}: {key} differs outside the masked rows")
            elif key == "identity_known_mask":
                if not np.array_equal(value[~mask], reference_value[~mask]) or value[mask].any():
                    fails.append(f"{Path(path).stem}: identity_known_mask wrong")
            elif not np.array_equal(value, reference_value):
                fails.append(f"{Path(path).stem}: {key} differs")
        if mask.any() and not (reference_cache["y"].numpy()[mask] == SCHWANN).all():
            fails.append(f"{Path(path).stem}: a masked row was not Schwann in the reference")
        n_diff_rows += int(mask.sum())
        if index < 5:
            for key in ("uni2", "phikon", "grid"):
                if not torch.equal(cache[key], reference_cache[key]):
                    fails.append(f"{Path(path).stem}: {key} features differ")
        del cache, reference_cache
    out["cache_rows_masked"] = n_diff_rows
    if n_diff_rows != masked:
        fails.append(f"cache masked rows {n_diff_rows} != summary {masked}")

    oof = sorted(glob.glob(str(outputs / f"oof/{arm}/fold{k}/*.npz")))
    reference_oof_dir = outputs / f"oof/{reference}/fold{k}"
    agree = total = 0
    max_diff = 0.0
    for path in oof:
        with np.load(path, allow_pickle=False) as value, np.load(reference_oof_dir / Path(path).name, allow_pickle=False) as reference_value:
            for key in OOF_KEYS:
                if not np.array_equal(value[key], reference_value[key]):
                    fails.append(f"{Path(path).stem}: OOF {key} differs")
            prob, reference_prob = value["prob"], reference_value["prob"]
            agree += int((prob.argmax(1) == reference_prob.argmax(1)).sum())
            total += len(prob)
            max_diff = max(max_diff, float(np.abs(prob.astype(np.float64) - reference_prob).max()))
    out["n_oof_files"] = len(oof)
    out["argmax_agreement_with_reference"] = agree / max(total, 1)
    out["max_abs_prob_diff"] = max_diff
    if len(oof) != len(glob.glob(str(reference_oof_dir / "*.npz"))):
        fails.append("held-out file count differs")

    identity, reference_identity = summary["identity"], reference_summary["identity"]
    out["macro_f1_10"] = [identity["macro_f1_10"], reference_identity["macro_f1_10"]]
    out["schwann_f1"] = [identity["per_class_f1"]["Schwann"], reference_identity["per_class_f1"]["Schwann"]]
    if args.null:
        if abs(identity["macro_f1_10"] - reference_identity["macro_f1_10"]) > TOL_MACRO:
            fails.append("macro_f1_10 outside tolerance")
        if abs(identity["per_class_f1"]["Schwann"] - reference_identity["per_class_f1"]["Schwann"]) > TOL_SCHWANN:
            fails.append("Schwann F1 outside tolerance")
    out["fails"] = fails
    out["status"] = "PASS" if not fails else "FAIL"
    destination = args.out or outputs / f"results/verify/{arm}_fold{k}.json"
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(out, indent=1))
    print(json.dumps({key: out[key] for key in out if key != "fails"}), flush=True)
    for failure in fails[:40]:
        print("FAIL", failure, flush=True)
    return 0 if not fails else 1


if __name__ == "__main__":
    raise SystemExit(main())
