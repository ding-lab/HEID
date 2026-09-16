#!/bin/bash
export PROJECTS_ROOT=/path/to/data
export RELEASE_ROOT=/path/to/this/repository

export PYTHON=python
export INSTANSEG_PYTHON="$PYTHON"
export CONDA_SH="$HOME/miniconda3/etc/profile.d/conda.sh"
export CONDA_ENV=heid

export HF_HOME="$PROJECTS_ROOT/cache/hf_home"
export TORCHINDUCTOR_CACHE_DIR="$PROJECTS_ROOT/cache/inductor"

export LSF_GROUP=
export DATA_MOUNT=
export DATA_MOUNT_ALIAS=
export C1_HOST_PREFIX=
export C1_GPU_EXCLUDE=

export CELL_SHORT_PARTITION=short
export CELL_PREEMPT_CPU_PARTITION=preempt-cpu
export CELL_GPU_PARTITION=gpu
export CELL_PREEMPT_GPU_PARTITION=preempt-gpu

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
