#!/usr/bin/env python3

from __future__ import annotations
import os
PROJECTS_ROOT = os.environ.get("PROJECTS_ROOT", "/data/heid")

import hashlib
import re
from pathlib import Path

PC = Path(PROJECTS_ROOT + "/cell")
COHORT_REFERENCE_ROOT = PC / "inputs/cohort_reference"
CELL_ROOT = PC
DATA = Path(PROJECTS_ROOT + "/cell/inputs")
ALIGN_ROOT = Path(PROJECTS_ROOT) / "align/aligned_he"
PX_UM = 0.2125


def expand_env(value):
    if isinstance(value, dict):
        return {key: expand_env(item) for key, item in value.items()}
    if isinstance(value, list):
        return [expand_env(item) for item in value]
    if isinstance(value, str):
        value = value.replace("${PROJECTS_ROOT}", PROJECTS_ROOT)
        if value == "$PROJECTS_ROOT" or value.startswith("$PROJECTS_ROOT/"):
            value = PROJECTS_ROOT + value[len("$PROJECTS_ROOT"):]
        return os.path.expandvars(value)
    return value

_CASE = re.compile(r"^(\d+)(?:[A-Za-z]\d*)?$")
_BLOCK = re.compile(r"^([A-Za-z]+\d+)[A-Za-z]+\d*$")


def patient_of(sample: str) -> str:
    fields = sample.split("-")
    if fields[0] == "HS" and len(fields) >= 3:
        m = _CASE.match(fields[2])
        if m:
            return f"{fields[0]}-{fields[1]}-{m.group(1)}"
    if len(fields) >= 2:
        m = _CASE.match(fields[1])
        if m:
            return f"{fields[0]}-{m.group(1)}"
    m = _BLOCK.match(fields[0])
    return m.group(1) if m else fields[0]


def sha256_file(path: Path | str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 22), b""):
            digest.update(block)
    return digest.hexdigest()


def sha256_list(items) -> str:
    return hashlib.sha256("\0".join(items).encode()).hexdigest()


def normalise(path: Path) -> Path:
    text = str(path)
    alias_from = os.environ.get("PATH_ALIAS_FROM", "")
    alias_to = os.environ.get("PATH_ALIAS_TO", "")
    if alias_from and text.startswith(alias_from + "/"):
        text = alias_to + "/" + text.removeprefix(alias_from + "/")
    return Path(text)
