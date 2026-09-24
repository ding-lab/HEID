#!/usr/bin/env python
from __future__ import annotations
import argparse
import csv
import importlib.util
import os
from pathlib import Path

import numpy as np
import pandas as pd

P3D = Path(os.environ.get("THREED_ROOT", os.path.join(os.environ.get("PROJECTS_ROOT", "/data/heid"), "3d")))
spec = importlib.util.spec_from_file_location("tls_he", Path(__file__).resolve().parent / "tls_define_he.py")
TD = importlib.util.module_from_spec(spec)
spec.loader.exec_module(TD)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sample", required=True)
    ap.add_argument("--annotation-dir", required=True, dest="annotation_dir")
    ap.add_argument("--runs-dir", required=True, dest="runs_dir")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    out = Path(a.out); pq_dir = out / "parquet" / "G_withdrawn"; out_dir = out / "G_withdrawn"
    for d in (pq_dir, out_dir):
        d.mkdir(parents=True, exist_ok=True)
    os.environ["TLS_SAVE_MASKS"] = "1"
    TD.PRED = str(pq_dir)
    runs = {}
    for d in sorted(Path(a.runs_dir).glob("output-XETG*")):
        if a.sample in d.name and (d / "cells.parquet").exists():
            c = pd.read_parquet(d / "cells.parquet", columns=["cell_id", "x_centroid", "y_centroid"])
            runs[d.name] = (c, set(c["cell_id"].astype(str)))
    rows, index = [], 0
    for f in sorted(Path(a.annotation_dir).glob(f"{a.sample}-*_cell_annotation.csv")):
        slide = f.name.replace("_cell_annotation.csv", "")
        with open(f) as fh:
            table = {r["cell_id"]: r["group"] for r in csv.DictReader(fh)}
        want = set(table)
        best, frac = None, 0.0
        for k, (c, ids) in runs.items():
            fr = len(want & ids) / max(len(want), 1)
            if fr > frac:
                best, frac = k, fr
        if best is None or frac < 0.90:
            print(f"  {slide}: no Xenium run reproduces its cell ids (best {frac:.1%}), skipped", flush=True)
            continue
        c = runs[best][0]
        c = c[c["cell_id"].astype(str).isin(want)].reset_index(drop=True)
        ids = c["cell_id"].astype(str).to_numpy()
        cls = np.array([table[i] for i in ids], object)
        x, y = c["x_centroid"].to_numpy(float), c["y_centroid"].to_numpy(float)
        pd.DataFrame({"x_um": x, "y_um": y, "pred_class_name": cls}).to_parquet(pq_dir / f"{slide}.parquet")
        TD.main(a.sample, slide, str(out_dir))
        m = np.load(out_dir / f"{slide}_masks.npz")
        cores, x0, y0, res = m["cores"], float(m["x_min"]), float(m["y_min"]), float(m["resolution"])
        tcsv = out_dir / f"{slide}_tls.csv"
        try:
            t = pd.read_csv(tcsv)
        except (pd.errors.EmptyDataError, FileNotFoundError):
            print(f"  {slide} ({best}): {len(ids):,} cells, 0 TLS", flush=True); index += 1
            continue
        gi = np.floor((y - y0) / res).astype(int); gj = np.floor((x - x0) / res).astype(int)
        ok = (gi >= 0) & (gj >= 0) & (gi < cores.shape[1]) & (gj < cores.shape[2])
        is_b = np.isin(cls, TD.B_TYPES)
        members = []
        for k in range(len(cores)):
            inside = np.zeros(len(ids), bool); inside[ok] = cores[k, gi[ok], gj[ok]]
            members.append((ids[inside], int((inside & is_b).sum())))
        assert len(members) == len(t), (slide, len(members), len(t))
        t.insert(0, "index", index); index += 1
        t["n_cells_core"] = [len(mm[0]) for mm in members]
        t["n_b_core_from_ids"] = [mm[1] for mm in members]
        t["cell_ids"] = [";".join(mm[0]) for mm in members]
        t["frame"] = "xenium_um"; t["xenium_run"] = best
        rows.append(t)
        print(f"  {slide} ({best}): {len(ids):,} cells, {len(t)} TLS; B+NK_T from ids vs n_b_core " +
              ", ".join(f"{mm[1]}/{r.n_b_core}" for r, mm in zip(t.itertuples(), members)), flush=True)
    if rows:
        pd.concat(rows).to_csv(out / "xenium_tls_table.tsv", sep="\t", index=False)
    print(f"wrote {out}: {sum(len(r) for r in rows)} TLS on {len(rows)} sections")


if __name__ == "__main__":
    main()
