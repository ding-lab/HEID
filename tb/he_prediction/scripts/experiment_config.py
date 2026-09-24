from __future__ import annotations

import json
from pathlib import Path
from typing import Any

PROJECT = Path(__file__).resolve().parents[1]
CONFIGS = PROJECT / "configs"
BINDING = PROJECT / "data" / "manifests" / "cohort_binding.json"
BINDING_SCHEMA = "heid.tb.cohort_binding.v1"


def binding_part(config_path: Path, binding_path: Path = BINDING) -> dict[str, Any] | None:
    try:
        key = config_path.resolve().relative_to(CONFIGS.resolve()).as_posix()
    except ValueError:
        return None
    if not binding_path.is_file():
        return None
    binding = json.loads(binding_path.read_text(encoding="utf-8"))
    if binding.get("schema") != BINDING_SCHEMA:
        raise ValueError(f"{binding_path}: expected schema {BINDING_SCHEMA}")
    return binding["configurations"].get(key)


def merge(config: dict[str, Any], part: dict[str, Any]) -> dict[str, Any]:
    merged = dict(config)
    for block, values in part.items():
        current = merged.get(block)
        if current is None:
            merged[block] = values
            continue
        if not isinstance(current, dict) or not isinstance(values, dict):
            raise ValueError(f"cohort binding block {block} cannot be joined")
        overlap = set(current) & set(values)
        if overlap:
            raise ValueError(f"cohort binding repeats configuration fields {sorted(overlap)} in {block}")
        merged[block] = {**current, **values}
    return merged


def load_experiment(config_path: Path, binding_path: Path = BINDING) -> dict[str, Any]:
    config = json.loads(Path(config_path).read_text(encoding="utf-8"))
    part = binding_part(Path(config_path), binding_path)
    return merge(config, part) if part else config


def canonical_sha256(payload: Any) -> str:
    import hashlib

    text = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


RELEASED = "experiment.json"
REFERENCE = "baseline_controls/experiment_baseline.json"


def cohort_size(configuration: str) -> dict[str, int]:
    cohort = load_experiment(CONFIGS / configuration)["cohort"]
    if "n_samples" not in cohort:
        raise ValueError(f"no cohort binding for configs/{configuration}; stage {BINDING}")
    return {key: int(cohort[key]) for key in ("n_samples", "n_patients", "n_cancers")}
