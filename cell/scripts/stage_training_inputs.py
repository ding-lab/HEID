#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import contract as C

_RELEASE = Path(__file__).resolve().parents[2]
CONFIGS = _RELEASE / "cell/configs"
ARM_CONFIGS = sorted(CONFIGS.glob("arm_*.json"))
DENOMINATOR_SCHEMA = "v8.scoring_denominator.v2"
DENOMINATOR_RULE = ("true counts and predicted counts are BOTH taken over the extracted oof rows that "
                    "carry a known identity label. Two exclusions, for two different reasons: a cell whose "
                    "crop failed the blank filter has no prediction, and a Necrosis cell has no identity "
                    "class. Both are excluded from both sides, so neither can manufacture a share error.")
DENOMINATOR_FIELD = ("per-sample UNI2 feature-extraction receipt rows_per_route[own_fold].n_extracted, "
                     "restricted to rows with identity_known_mask true in the oof arrays")


def place(text: str, destination: Path, replace: bool) -> None:
    if destination.exists() and destination.read_text() != text and not replace:
        raise FileExistsError(f"{destination} differs; pass --replace to stage over it")
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(text)
    print(f"staged {destination}")


def stage_contract(args) -> None:
    source = Path(args.source or C.MANIFEST_ROOT / "contract.json")
    C.CONTRACT, C._CONTRACT_CACHE = source, None
    C.load_contract()
    place(source.read_text(), CONFIGS / "contract.json", args.replace)


def stage_data_contract(args) -> None:
    source = Path(args.source or C.MANIFEST_ROOT / "data_contract.json")
    record = json.loads(source.read_text())
    staged = C.sha256_file(CONFIGS / "contract.json")
    if record["provenance"]["contract_v8_sha256"] != staged:
        raise ValueError(f"{source} was not built from the staged contract")
    place(source.read_text(), CONFIGS / "data_contract.json", args.replace)


def stage_xenium_index(args) -> None:
    root = Path(args.xenium_root)
    index = {}
    for sample in C.allowlist():
        record = C.record(sample)
        path = root / str(record["cohort"]) / str(record["cancer"]) / sample
        if not path.is_dir():
            raise FileNotFoundError(f"{sample}: no Xenium output directory at {path}")
        index[sample] = {"path": str(path)}
    place(json.dumps(index, indent=1) + "\n", CONFIGS / "xenium_5k_index.json", args.replace)


def stage_exclusion(args) -> None:
    import pandas as pd

    source = Path(args.source or C.CELL_ROOT / "outputs/manifests/identity_exclusion_manifest.json")
    manifest = json.loads(source.read_text())
    records = manifest["per_sample"]
    if set(records) != set(C.allowlist()):
        raise ValueError("exclusion manifest does not cover the contract cohort exactly")
    checks = manifest["checks"]
    if any(value is False for value in checks.values()) or (checks.get("min_define_id_coverage") or 0.0) < 0.99:
        raise ValueError(f"exclusion manifest failed its own checks: {checks}")
    labels_root = Path(args.labels_root)
    for sample, record in records.items():
        path = labels_root / f"{sample}.parquet"
        if C.sha256_file(path) != record["sha256"] or C.sha256_file(C.labels_path(sample)) != record["source_label_sha256"]:
            raise ValueError(f"{sample}: exclusion label table or its source labels differ from the manifest")
        labels = pd.read_parquet(path)
        flagged = labels["identity_excluded"].to_numpy(dtype=bool)
        if len(labels) != record["n_rows"] or labels["cell_id"].duplicated().any():
            raise ValueError(f"{sample}: exclusion label rows differ from the manifest")
        if int(flagged.sum()) != record["n_excluded"] or (labels.loc[flagged, "cell_type_refine"] != "Schwann").any():
            raise ValueError(f"{sample}: excluded rows are not the manifest's Schwann rows")
    text = source.read_text()
    place(text, CONFIGS / "identity_exclusion_manifest.json", args.replace)
    digest = C.sha256_file(CONFIGS / "identity_exclusion_manifest.json")
    for path in ARM_CONFIGS:
        config = json.loads(path.read_text())
        block = config.get("inputs", {}).get("identity_exclusion")
        if block is None or block.get("manifest_sha256") == digest:
            continue
        if not args.replace:
            raise FileExistsError(f"{path} binds another exclusion manifest; pass --replace to rebind")
        block["manifest_sha256"] = digest
        path.write_text(json.dumps(config, indent=1) + "\n")
        print(f"bound {path.name} to manifest {digest}")


def stage_denominator(args) -> None:
    import numpy as np
    import torch

    taxonomy = json.loads((CONFIGS / "taxonomy_coarse11.json").read_text())
    supervised = taxonomy["source_vocabularies"]["refine17"]["identity_supervision_mask"]
    feature_root, receipt_root = Path(args.feature_root), Path(args.receipt_root)
    patients = defaultdict(lambda: {"n_excluded_blank_crop": 0, "n_excluded_no_identity_label": 0,
                                    "n_labelled": 0, "n_scorable": 0})
    for sample in C.allowlist():
        fold = C.own_fold(sample)
        receipt = json.loads((receipt_root / f"{sample}.json").read_text())
        if receipt["status"] != "PASS" or receipt["sample"] != sample or int(receipt["own_outer_fold"]) != fold:
            raise ValueError(f"{sample}: feature receipt is not a passing own-fold receipt")
        route = receipt["rows_per_route"][f"fold{fold}"]
        cells = C.load_cells(sample)
        if route["role"] != "oof" or int(route["n_routed"]) != len(cells):
            raise ValueError(f"{sample}: own-fold route does not cover the sample's cells")
        labels = dict(zip(cells["cell_id"].astype(str), cells["cell_type_refine"].astype(str)))
        if int(receipt["n_blank_crops"]):
            features = torch.load(feature_root / "oof" / f"fold{fold}" / f"{sample}.pt",
                                  map_location="cpu", weights_only=False)
            extracted = np.asarray(features["cell_ids"], dtype=str)
        else:
            extracted = np.asarray(list(labels), dtype=str)
        if len(extracted) != int(route["n_extracted"]):
            raise ValueError(f"{sample}: extracted rows differ from the receipt")
        known = sum(bool(supervised[labels[cell]]) for cell in extracted)
        entry = patients[str(C.record(sample)["patient"])]
        entry["n_labelled"] += len(cells)
        entry["n_scorable"] += known
        entry["n_excluded_blank_crop"] += len(cells) - len(extracted)
        entry["n_excluded_no_identity_label"] += len(extracted) - known
    affected = {patient: patients[patient] for patient in sorted(patients)
                if patients[patient]["n_scorable"] != patients[patient]["n_labelled"]}
    payload = {
        "schema_version": DENOMINATOR_SCHEMA,
        "rule": DENOMINATOR_RULE,
        "denominator_field": DENOMINATOR_FIELD,
        "n_scorable_cells": sum(p["n_scorable"] for p in patients.values()),
        "n_excluded_blank_crop": sum(p["n_excluded_blank_crop"] for p in patients.values()),
        "n_excluded_no_identity_label": sum(p["n_excluded_no_identity_label"] for p in patients.values()),
        "n_labelled_cells_on_oof_routes": sum(p["n_labelled"] for p in patients.values()),
        "n_patients": len(patients),
        "n_patients_affected": len(affected),
        "per_patient": affected,
    }
    place(json.dumps(payload, indent=2, sort_keys=True) + "\n", Path(args.output or CONFIGS / "scoring_denominator.json"),
          args.replace)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--replace", action="store_true")
    replace = argparse.ArgumentParser(add_help=False)
    replace.add_argument("--replace", action="store_true", default=argparse.SUPPRESS)
    sub = parser.add_subparsers(dest="command", required=True)
    for name in ("contract", "data-contract", "exclusion"):
        p = sub.add_parser(name, parents=[replace])
        p.add_argument("--source", default=None)
        if name == "exclusion":
            p.add_argument("--labels-root", default=str(C.CELL_ROOT / "outputs/labels_region_schwann"))
    p = sub.add_parser("xenium-index", parents=[replace])
    p.add_argument("--xenium-root", default=str(C.PC / "inputs/Xenium"))
    p = sub.add_parser("denominator", parents=[replace])
    p.add_argument("--feature-root", default=str(C.FEATURE_ROOT / "uni2/cls"))
    p.add_argument("--receipt-root", default=str(C.MANIFEST_ROOT / "features/uni2"))
    p.add_argument("--output", default=None)
    args = parser.parse_args()
    {"contract": stage_contract, "data-contract": stage_data_contract, "xenium-index": stage_xenium_index,
     "exclusion": stage_exclusion, "denominator": stage_denominator}[args.command](args)
    return 0


if __name__ == "__main__":
    sys.exit(main())
