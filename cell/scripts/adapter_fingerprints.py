#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import contract as C

_RELEASE = Path(__file__).resolve().parents[2]
COMPATIBILITY = _RELEASE / "cell/configs/adapter_compatibility.json"
OUTPUT = _RELEASE / "cell/configs/adapter_fingerprints.json"
SCHEMA = "heid.cell.adapter_fingerprints.v1"


def build() -> dict:
    accepted = [C.cohort_binding_sha256()]
    receipt = json.loads(COMPATIBILITY.read_text())
    semantic = {key: value for key, value in C.cohort_binding_payload().items() if key != "taxonomy"}
    if (C.canonical_sha256(semantic) == receipt["semantic_payload_sha256"]
            and C.sha256_file(_RELEASE / receipt["taxonomy_file"]) == receipt["taxonomy_content_sha256"]):
        accepted.append(receipt["source_cohort_binding_sha256"])
    folds = {}
    for fold in range(5):
        fingerprint = C.fold_fingerprint(fold)
        fingerprint.pop("cohort_binding_sha256")
        folds[str(fold)] = fingerprint
    return {"schema": SCHEMA, "accepted_cohort_binding_sha256": sorted(set(accepted)),
            "taxonomy_file": receipt["taxonomy_file"],
            "taxonomy_content_sha256": receipt["taxonomy_content_sha256"],
            "folds": folds}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--check", action="store_true")
    parser.add_argument("--output", type=Path, default=OUTPUT)
    args = parser.parse_args()
    payload = build()
    text = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    if args.check:
        if not args.output.is_file() or args.output.read_text() != text:
            print(f"{args.output} differs from the contract", file=sys.stderr)
            return 1
        print(f"{args.output} matches the contract")
        return 0
    args.output.write_text(text)
    print(f"wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
