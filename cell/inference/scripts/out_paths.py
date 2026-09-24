import os
PROJECTS_ROOT = os.environ.get("PROJECTS_ROOT", "/data/heid")
import functools
from pathlib import Path

import pandas as pd

INFERENCE_ROOT = Path(PROJECTS_ROOT + "/cell/inference")
COHORT = os.environ.get("CELL_COHORT", "tcga_pancancer")
ROOT = INFERENCE_ROOT / COHORT
OUT = INFERENCE_ROOT
MANIFEST = INFERENCE_ROOT / "configs" / COHORT / "run_manifest.tsv"


@functools.lru_cache(maxsize=1)
def _project_of() -> dict:
    table = pd.read_csv(MANIFEST, sep="\t", usecols=["slide", "project_id"])
    return dict(zip(table["slide"], table["project_id"]))


def project_of(slide: str) -> str:
    return _project_of().get(slide, "UNKNOWN")


def slide_dir(slide: str) -> Path:
    return INFERENCE_ROOT / project_of(slide) / "outputs" / slide


def project_dir(slide_or_project: str) -> Path:
    name = slide_or_project
    if name not in set(_project_of().values()):
        name = project_of(name)
    return INFERENCE_ROOT / name / "outputs"


def summary_dir(project: str) -> Path:
    d = INFERENCE_ROOT / project / "summary"
    d.mkdir(parents=True, exist_ok=True)
    return d


def summary_path(project: str, shard=None) -> Path:
    stem = f"{project}_summary" + (f"_shard{shard}" if shard is not None else "")
    return summary_dir(project) / f"{stem}.csv"


def all_projects() -> list:
    return sorted(set(_project_of().values()))
