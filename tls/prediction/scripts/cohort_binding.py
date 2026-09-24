from __future__ import annotations

import json
from pathlib import Path
from typing import Any

MODULE_ROOT = Path(__file__).resolve().parents[1]
BINDING = MODULE_ROOT / "data" / "manifests" / "cohort_binding.json"
BINDING_SCHEMA = "heid.tls.cohort_binding.v1"


def merge(config: Any, part: Any, key: str = "") -> Any:
    if not isinstance(part, dict):
        if config is not None:
            raise ValueError(f"cohort binding repeats configuration field {key}")
        return part
    if config is None:
        return part
    if not isinstance(config, dict):
        raise ValueError(f"cohort binding block {key} cannot be joined")
    merged = dict(config)
    for name, value in part.items():
        merged[name] = merge(config.get(name), value, f"{key}.{name}".strip("."))
    return merged


def load_contract(config_path: Path, binding_path: Path = BINDING) -> dict[str, Any]:
    config = json.loads(Path(config_path).read_text(encoding="utf-8"))
    if not binding_path.is_file():
        raise FileNotFoundError(f"no cohort binding at {binding_path}; stage the binding supplied with the cohort")
    binding = json.loads(binding_path.read_text(encoding="utf-8"))
    if binding.pop("schema", None) != BINDING_SCHEMA:
        raise ValueError(f"{binding_path}: expected schema {BINDING_SCHEMA}")
    return merge(config, binding)
