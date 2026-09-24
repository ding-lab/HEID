#!/usr/bin/env python
from __future__ import annotations

import importlib.util
import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
DET = Path(__file__).resolve().parents[2] / "cell/inference/scripts"
BLOCK_ROOT = Path(os.environ.get("BLOCK_INFERENCE_ROOT", os.path.join(
    os.environ.get("THREED_ROOT", os.path.join(os.environ.get("PROJECTS_ROOT", "/data/heid"), "3d")),
    "inference", "blocks")))
MANIFEST = BLOCK_ROOT / "configs" / os.environ.get("BLOCK_SET", "blocks") / "run_manifest.tsv"
OUTROOT = BLOCK_ROOT / "outputs"

STAGES = {"detect": "detect_cells.py",
          "grid": "tissue_grid.py",
          "infer": "infer_cells.py"}


def main():
    if len(sys.argv) < 3 or sys.argv[1] not in STAGES:
        raise SystemExit(f"usage: run_block_stage.py {{{'|'.join(STAGES)}}} <slide> [args]")
    stage, slide, rest = sys.argv[1], sys.argv[2], sys.argv[3:]

    import pandas as pd
    man = pd.read_csv(MANIFEST, sep="\t").set_index("slide")
    if slide not in man.index:
        raise SystemExit(f"FATAL: {slide} not in {MANIFEST}")
    r = man.loc[slide]


    os.environ["BLOCK_CROP"] = f"{r.crop_x0},{r.crop_y0},{r.crop_x1},{r.crop_y1}"

    bm = str(r.block_mask) if "block_mask" in man.columns and str(r.block_mask) not in ("", "nan") else ""
    if bm:
        os.environ["BLOCK_MASK"] = bm


    os.environ.setdefault("BLOCK_INFERENCE_ROOT", str(BLOCK_ROOT))
    os.environ.setdefault("BLOCK_INFERENCE_SET", ".")

    sys.path.insert(0, str(DET))
    sys.path.insert(0, str(HERE))

    import crop_view
    import slide_canvas
    b = crop_view.install(slide_canvas)
    print(f"[block] {slide} crop x{b[0]}-{b[2]} y{b[1]}-{b[3]}"
          + (f" + outline mask {os.path.basename(bm)}" if bm else "") + "  "
          f"({r.level0_W}x{r.level0_H} of {r.full_W}x{r.full_H})  "
          f"patient={r.patient} position={r.block_position}", flush=True)


    name = STAGES[stage][:-3]
    spec = importlib.util.spec_from_file_location(name, HERE / STAGES[stage])
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)

    sys.argv = [STAGES[stage], "--slide", slide,
                "--manifest", str(MANIFEST), *rest]
    mod.main()


if __name__ == "__main__":
    main()
