#!/bin/bash
export PROJECTS_ROOT=/path/to/data
export RELEASE_ROOT=/path/to/this/repository

export PYTHON=python
export GEOMETRY_PYTHON="$RELEASE_ROOT/.venv-geometry/bin/python"

export HF_HOME="$PROJECTS_ROOT/cache/hf_home"
export TORCHINDUCTOR_CACHE_DIR="$PROJECTS_ROOT/cache/inductor"



export PATH_ALIAS_FROM=
export PATH_ALIAS_TO=
export PROJECT_PATH_ALIASES="$PROJECTS_ROOT:$RELEASE_ROOT"

export CELL_COHORT=tcga_pancancer
export CELL_HE_SRC="$PROJECTS_ROOT/data/tcga_he/HE"
export CELL_PRAD_SVS="$PROJECTS_ROOT/cell/inference/data/tcga_prad/svs"
export DETECTION_POOL_CSV="$PROJECTS_ROOT/cell/detection/data/detection_pool_per_sample.csv"

export TLS_FEATURES="$PROJECTS_ROOT/tls/prediction/features"
export TLS_LABELS="$PROJECTS_ROOT/tls/prediction/labels"
export TLS_REPORT_OUTPUT="$PROJECTS_ROOT/tls/define/outputs/reporting"

export THREED_ROOT="$PROJECTS_ROOT/3d"
export THREED_DATA_ROOT="$PROJECTS_ROOT/data/3d"
export BLOCK_INFERENCE_ROOT="$THREED_ROOT/inference/blocks"
export BLOCK_SET=blocks
export CHROME=
export PAGE_HOST=
export PAGE_PORT=8931
