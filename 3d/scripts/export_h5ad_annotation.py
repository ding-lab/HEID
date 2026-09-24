#!/usr/bin/env python3
import argparse
import os

import h5py
import numpy as np
import pandas as pd


def col(f, name):
    g = f["obs"][name]
    if isinstance(g, h5py.Group):
        cats = np.array([c.decode() if isinstance(c, bytes) else c for c in g["categories"][:]], object)
        return cats[g["codes"][:]]
    v = g[:]
    return np.array([x.decode() if isinstance(x, bytes) else x for x in v], object)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--h5ad", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--type-col", default="cell_type", dest="type_col")
    ap.add_argument("--slide-col", default="sample_id", dest="slide_col")
    ap.add_argument("--id-col", default="cell_id", dest="id_col")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    with h5py.File(a.h5ad) as f:
        df = pd.DataFrame({"cell_id": col(f, a.id_col), "group": col(f, a.type_col),
                           "slide": col(f, a.slide_col)})
    for s, d in df.groupby("slide"):
        p = os.path.join(a.out, f"{s}_cell_annotation.csv")
        d[["cell_id", "group"]].to_csv(p, index=False)
        print(f"  {s}: {len(d):,} cells, {d.group.nunique()} classes -> {p}")
    print(f"{len(df):,} cells, classes: {df.group.value_counts().to_dict()}")


if __name__ == "__main__":
    main()
