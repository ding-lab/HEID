#!/usr/bin/env python
from __future__ import annotations
import argparse
import json
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
from scipy import sparse

MARKERS = {
    "B_cell": ["MS4A1", "CD79A", "CD79B", "CD19", "BANK1", "CR2", "PAX5"],
    "Plasma": ["MZB1", "JCHAIN", "TNFRSF17", "IGKC", "IGHG1", "IGHA1"],
    "NK_T": ["CD3D", "CD3E", "CD2", "CD247", "TRAC", "CD8A", "NKG7", "GZMA", "GNLY"],
    "Macrophage": ["CD68", "CD163", "C1QA", "C1QC", "LYZ", "MS4A7", "CD14"],
    "Tumor": ["EPCAM", "KRT19", "KRT8", "KRT18", "KRT7", "CDH1", "KRT17"],
    "Fibroblast": ["COL1A1", "COL1A2", "COL3A1", "LUM", "DCN", "PDGFRA", "FAP"],
    "Endothelial": ["PECAM1", "VWF", "CDH5", "CLDN5", "PLVAP"],
    "SMC": ["ACTA2", "MYH11", "DES", "TAGLN"],
}
MIN_COUNTS = 10


def read_matrix(run: Path):
    with h5py.File(run / "cell_feature_matrix.h5") as f:
        g = f["matrix"]
        m = sparse.csc_matrix((g["data"][:], g["indices"][:], g["indptr"][:]), shape=tuple(g["shape"][:]))
        names = np.array([x.decode() for x in g["features/name"][:]])
        ftype = np.array([x.decode() for x in g["features/feature_type"][:]])
        cells = np.array([x.decode() for x in g["barcodes"][:]])
    ge = ftype == "Gene Expression"
    return m[ge].tocsr(), names[ge], cells


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sections", required=True)
    ap.add_argument("--sample", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    secs = json.loads(Path(a.sections).read_text())["sections"]
    summary = []
    for sid, rec in sorted(secs.items()):
        m, genes, cells = read_matrix(Path(rec["xenium_dir"]))
        total = np.asarray(m.sum(0)).ravel()
        scale = 100.0 / np.maximum(total, 1)
        classes, scores, used = list(MARKERS), [], {}
        gi = {g: i for i, g in enumerate(genes)}
        for c in classes:
            rows = [gi[g] for g in MARKERS[c] if g in gi]
            used[c] = [g for g in MARKERS[c] if g in gi]
            if not rows:
                scores.append(np.zeros(len(cells)))
                continue
            x = m[rows].toarray().astype(np.float32) * scale
            scores.append(np.log1p(x).mean(0))
        s = np.vstack(scores)
        best = s.argmax(0)
        group = np.array(classes, object)[best]
        group[(s.max(0) <= 0) | (total < MIN_COUNTS)] = "Unknown"
        pd.DataFrame({"cell_id": cells, "group": group}).to_csv(out / f"{sid}_cell_annotation.csv", index=False)
        counts = pd.Series(group).value_counts()
        row = {"section_id": sid, "n_cells": len(cells), "n_genes_panel": len(genes)}
        row.update({c: int(counts.get(c, 0)) for c in classes + ["Unknown"]})
        row["markers_used"] = "; ".join(f"{c}: {','.join(used[c])}" for c in classes)
        summary.append(row)
        print(f"  {sid}: {len(cells):,} cells, {len(genes)} genes; " +
              ", ".join(f"{c} {int(counts.get(c, 0)):,}" for c in classes + ["Unknown"]), flush=True)
    pd.DataFrame(summary).to_csv(out / "marker_annotation_summary.tsv", sep="\t", index=False)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
