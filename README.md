# HEID

Code for deriving cell types and tissue structures from H&E, with the paired spatial
data used to define them. Five branches:

| Branch | Contents |
|---|---|
| `cell/` | Cell-type classification: crop banks, backbone adapters, feature extraction, head training, scoring, and the new-slide inference chain |
| `align/` | Xenium-to-H&E and CODEX-to-H&E image registration, per-tissue refinement, and cell-level quality control |
| `tb/` | Molecular tumor boundary from spatial cell identities, and its prediction from H&E |
| `tls/` | Tertiary lymphoid structure detection and reporting, H&E region prediction, and maturation scoring |
| `nerve/` | Schwann and glial region definition from spatial data, and a gene-blind region operator over H&E cell probabilities |

## Setup

```bash
python -m venv venv && . venv/bin/activate
pip install -r requirements.txt
cp env.example.sh env.sh && $EDITOR env.sh && . env.sh
python materialize_configs.py --projects-root "$PROJECTS_ROOT" --dry-run
python materialize_configs.py --projects-root "$PROJECTS_ROOT"
```

`PROJECTS_ROOT` is a writable workspace holding the inputs and receiving the outputs.
`RELEASE_ROOT` is this checkout. Shipped configuration files carry `${PROJECTS_ROOT}`
and `${RELEASE_ROOT}` placeholders; `materialize_configs.py` writes them out once.

`python smoke_test.py` runs the synthetic CPU checks and needs no data.

Each branch is organised as `<branch>/<stage>/scripts/`, with configuration under
`configs/`. Every script takes `--help`.

## Models and inputs

Obtain UNI2-h, Phikon-v2 and InstanSeg from their distributors under their own
licences; see [documents/THIRD_PARTY_NOTICES.md](documents/THIRD_PARTY_NOTICES.md).
Stage your slides, spatial tables and cell tables under `PROJECTS_ROOT` as the branch
configurations name them. The cell branch reads its cohort contract, scoring
denominator and input indexes from `cell/configs/` and `cell/data/`; write those for
your own cohort.

## Licence and citation

Code is released under the Apache License 2.0 ([LICENSE](LICENSE)). Imported
components keep their own terms. Citation metadata is in
[CITATION.cff](CITATION.cff).
