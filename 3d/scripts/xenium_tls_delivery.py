#!/usr/bin/env python
from __future__ import annotations
import argparse
import shutil
from pathlib import Path

import pandas as pd


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--table", required=True)
    ap.add_argument("--tls-dir", required=True, dest="tls_dir")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    if not Path(a.table).exists():
        out = Path(a.out); (out / "figures").mkdir(parents=True, exist_ok=True)
        cols = ["index", "sample", "slide", "tls_id", "n_b_core", "core_area_mm2", "cx_um", "cy_um", "n_cells_core", "cell_ids"]
        pd.DataFrame(columns=cols).to_csv(out / "4_xenium_tls.tsv", sep="\t", index=False)
        figs = sorted(Path(a.tls_dir).glob("*_tls.png"))
        for f in figs:
            shutil.copyfile(f, out / "figures" / f.name)
        print(f"wrote {out}/4_xenium_tls.tsv: 0 TLS; {len(figs)} figures")
        return
    t = pd.read_csv(a.table, sep="\t")
    if "cell_ids" not in t.columns:
        raise SystemExit(f"{a.table} has no cell_ids column: rebuild the Xenium layer with build_xenium_gt.py first")
    bad = t[t["n_b_core"] != t["n_b_core_from_ids"]]
    if len(bad):
        raise SystemExit("B cells (B_cell + NK_T, the geometry's B) from the cell_id lists differ from n_b_core:\n" +
                         bad[["slide", "tls_id", "n_b_core", "n_b_core_from_ids"]].to_string(index=False))
    out = Path(a.out); (out / "figures").mkdir(parents=True, exist_ok=True)
    cols = [c for c in t.columns if c not in ("index", "n_b_core_from_ids", "cell_ids")] + ["cell_ids"]
    t[cols].to_csv(out / "4_xenium_tls.tsv", sep="\t", index=False)
    figs = sorted(Path(a.tls_dir).glob("*_tls.png"))
    n_lab = 0
    for f in figs:
        n = redraw_with_ids(f, t, Path(a.tls_dir), out / "figures")
        if n is None:
            shutil.copyfile(f, out / "figures" / f.name)
        else:
            n_lab += n
    print(f"wrote {out}/4_xenium_tls.tsv: {len(t)} TLS on {t['slide'].nunique()} sections, "
          f"{int(t['n_cells_core'].sum()):,} cell ids; {len(figs)} figures, {n_lab} TLS labelled with their id")


def redraw_with_ids(fig_path, table, tls_dir, out_dir):
    import importlib.util
    import numpy as np
    slide = fig_path.name[:-len("_tls.png")]
    mp = tls_dir / f"{slide}_masks.npz"
    pq = tls_dir.parent / "parquet" / tls_dir.name / f"{slide}.parquet"
    if not (mp.exists() and pq.exists()):
        return None
    spec = importlib.util.spec_from_file_location("tls_define_he", Path(__file__).resolve().parent / "tls_define_he.py")
    TD = importlib.util.module_from_spec(spec); spec.loader.exec_module(TD)
    TD.CAVEAT = ("Cell types are the XENIUM annotation of this section (panel / marker-gene annotation), not an H&E model prediction; "
                 "TLS gates are the same as on H&E (B_cell + NK_T pooled as B).")
    mz = np.load(mp)
    pred = pd.read_parquet(pq, columns=["x_um", "y_um", "pred_class_name"])
    ct = pred["pred_class_name"].values
    cls = np.full(len(pred), "Other", dtype=object)
    cls[np.isin(ct, TD.B_TYPES)] = "B_cell"
    cls[np.isin(ct, TD.P_TYPES)] = "Plasma"
    df = pred.assign(**{"class": cls})
    sel_b, sel_t, sel_p = cls == "B_cell", np.isin(ct, TD.T_TYPES), cls == "Plasma"
    cores = list(mz["cores"]); tagg = list(mz["tagg"]) if "tagg" in mz else []; magg = list(mz["magg"]) if "magg" in mz else []
    ny, nx = (cores[0].shape if cores else (tagg[0].shape if tagg else (1, 1)))
    id_map = {str(x): str(x) for x in table.loc[table["slide"] == slide, "tls_id"]}
    TD.render_figure(slide, str(out_dir), df, sel_b, sel_t, sel_p, cores, tagg, magg,
                     float(mz["x_min"]), float(mz["x_max"]), float(mz["y_min"]), float(mz["y_max"]), ny, nx, id_map=id_map)
    return len(id_map)


if __name__ == "__main__":
    main()
