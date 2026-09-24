#!/usr/bin/env python3

import argparse
import json
from pathlib import Path

import pandas as pd


COLS = {"name": "name", "n_sections": "n_sections", "n_regions": "n_regions", "z_min_um": "z_min_um",
        "z_max_um": "z_max_um", "extent_um": "extent_um", "length_um": "length_um", "volume_um3": "volume_um3",
        "tube_radius_um": "tube_um", "max_diameter_um": "max_diameter_um", "max_diameter_z_um": "max_diameter_z_um",
        "curvature_radius_um": "radius_um", "model": "model", "high_confidence": "high_confidence"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--nerve-dir", required=True)
    ap.add_argument("--out-stem", required=True)
    a = ap.parse_args()
    d = Path(a.nerve_dir)
    rep = json.loads((d / "Schwann_nerves.json").read_text())
    rows = [{c: n.get(k) for c, k in COLS.items()} for n in rep["nerves"]]
    df = pd.DataFrame(rows, columns=list(COLS)).sort_values("name")
    df.insert(0, "rank", range(1, len(df) + 1))
    out = Path(a.out_stem)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out.with_suffix(".tsv"), sep="\t", index=False)
    print(f"{len(df)} cords -> {out.with_suffix('.tsv')}")
    print(f"  volume um3: median {df.volume_um3.median():,.0f}  max {df.volume_um3.max():,.0f}")
    print(f"  thickest um: median {df.max_diameter_um.median():.1f}  max {df.max_diameter_um.max():.1f}")
    print(f"  length um:  median {df.extent_um.median():.0f}  max {df.extent_um.max():.0f}")


if __name__ == "__main__":
    main()
