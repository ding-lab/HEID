# HEID

Code for deriving cell types and tissue structures from H&E, with the paired spatial
data used to define them. Six branches:

| Branch | Contents |
|---|---|
| `cell/` | Cell-type classification: cohort preparation, crop banks, backbone adapters, feature extraction, head training, scoring, and the new-slide inference chain |
| `align/` | Xenium-to-H&E and CODEX-to-H&E registration, per-tissue refinement, cell-level quality control and publication of the aligned tree |
| `tb/` | Molecular tumor boundary from spatial cell identities, and its prediction from H&E |
| `tls/` | Tertiary lymphoid structure detection and reporting, H&E region prediction, and maturation scoring |
| `nerve/` | Schwann and glial region definition from spatial data, and a gene-blind region operator over H&E cell probabilities |
| `3d/` | Serial-section reconstruction: section preparation, block inference with the Cell head, pairwise and anchor-based alignment, volume and cell layers, and three-dimensional TLS, duct lumens and nerves |

## Setup

```bash
python3.10 -m venv .venv && . .venv/bin/activate
pip install -r requirements.txt
pip install --no-deps opencv-python-headless==4.12.0.88
python3.13 -m venv .venv-geometry
.venv-geometry/bin/pip install -r 3d/pipeline/requirements-geometry.txt
.venv-geometry/bin/pip install --no-deps opencv-python-headless==4.12.0.88
cp env.example.sh env.sh && $EDITOR env.sh && . env.sh
python materialize_configs.py --projects-root "$PROJECTS_ROOT" --dry-run
python materialize_configs.py --projects-root "$PROJECTS_ROOT"
python smoke_test.py
```

`PROJECTS_ROOT` is a writable workspace holding the inputs and most outputs; `RELEASE_ROOT`
is this checkout, which receives the outputs of the stages whose configuration names
`${RELEASE_ROOT}`. Configuration files carry both placeholders and `materialize_configs.py`
writes them out once. `smoke_test.py` runs synthetic CPU checks that need no data.
The 3D drivers run the registration and object stages with `GEOMETRY_PYTHON`, the Python 3.13
environment. OpenCV is installed without dependency resolution because its wheel declares
`numpy>=2,<2.3`, which neither NumPy pin satisfies.

## Inputs and commands

Each branch holds `configs/inputs.schema.json` files that list every external input with its
layout and columns, the command that produces or consumes it, and the order of the
preparation, training and evaluation commands. Command-line entry points are under the
branches' `scripts/` directories. Install InstanSeg and obtain UNI2-h and Phikon-v2
from their distributors under their own licences; see [documents/THIRD_PARTY_NOTICES.md](documents/THIRD_PARTY_NOTICES.md).
The fine-tuned nucleus detector, trained heads and adapters, cohort bindings and controlled
data come from their custodians.

## Licence and citation

Code is released under the Apache License 2.0 ([LICENSE](LICENSE)). Imported components
keep their own terms. Citation metadata is in [CITATION.cff](CITATION.cff).
