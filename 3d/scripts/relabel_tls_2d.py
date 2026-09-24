#!/usr/bin/env python3
import argparse
import importlib.util
import os
import sys

import numpy as np
import pandas as pd

P3D = os.environ.get("THREED_ROOT", os.path.join(os.environ.get("PROJECTS_ROOT", "/data/heid"), "3d"))
HERE = os.path.dirname(os.path.abspath(__file__))


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec)
    sys.modules[name] = m
    spec.loader.exec_module(m)
    return m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sample", required=True)
    ap.add_argument("--objects", required=True)
    ap.add_argument("--pred", required=True)
    ap.add_argument("--tls-root", default="", dest="tls_root")
    a = ap.parse_args()
    root = a.tls_root or os.path.join(P3D, "tls_define", a.sample)
    TD = _load("tls_he", os.path.join(HERE, "tls_define_he.py"))

    objs = pd.read_csv(a.objects, sep="\t")
    id_of = {}
    for o in objs.itertuples():
        for tok in str(o.member_tls).split(";"):
            if ":" in tok:
                s, t = tok.rsplit(":", 1)
                id_of[(s, t)] = o.object_id

    n_slides = n_lab = n_unlinked = n_nofig = 0
    for slide in sorted(os.listdir(root)):
        d = os.path.join(root, slide)
        tp, cp = os.path.join(d, f"{slide}_tls.csv"), os.path.join(d, f"{slide}_tls_cells.csv")
        if not (os.path.exists(tp) and os.path.exists(cp)):
            continue
        try:
            tdf = pd.read_csv(tp)
        except pd.errors.EmptyDataError:
            continue
        if "tls_id" not in tdf:
            continue


        if "tls_local" not in tdf:
            tdf["tls_local"] = tdf["tls_id"]
        tdf["tls_id"] = [id_of.get((slide, t), "") for t in tdf["tls_local"]]
        tdf = tdf.drop(columns=[c for c in ("tls_object",) if c in tdf])
        n_unlinked += int((tdf["tls_id"] == "").sum())
        tdf.to_csv(tp, index=False)

        pc = pd.read_csv(cp)
        for h in (50, 100, 150):
            col, loc = f"tls_region_halo{h}", f"tls_local_halo{h}"
            if col not in pc:
                continue
            if loc not in pc:
                pc[loc] = pc[col]
            pc[col] = [id_of.get((slide, t), "") if t != "Outside" else "Outside" for t in pc[loc].astype(str)]
        pc = pc.drop(columns=[c for c in ("tls_object",) if c in pc])
        pc.to_csv(cp, index=False)
        lab = pc["tls_local_halo150"].astype(str)
        n_slides += 1


        mp = os.path.join(d, f"{slide}_masks.npz")
        if not os.path.exists(mp):
            n_nofig += 1
            continue
        mz = np.load(mp)
        pred = pd.read_parquet(os.path.join(a.pred, slide + ".parquet"),
                               columns=["x_um", "y_um", "pred_class_name"])
        assert len(pred) == len(pc), (slide, len(pred), len(pc))
        ct = pred["pred_class_name"].values
        cls = np.full(len(pred), "Other", dtype=object)
        cls[np.isin(ct, TD.B_TYPES)] = "B_cell"
        cls[np.isin(ct, TD.P_TYPES)] = "Plasma"
        df = pred.assign(**{"class": cls})
        sel_b, sel_t, sel_p = cls == "B_cell", np.isin(ct, TD.T_TYPES), cls == "Plasma"
        cores = list(mz["cores"]); tagg = list(mz.get("tagg", [])) if "tagg" in mz else []
        magg = list(mz.get("magg", [])) if "magg" in mz else []
        ny, nx = (cores[0].shape if cores else (tagg[0].shape if tagg else (1, 1)))
        id_map = {t.tls_local: (t.tls_id or "unlinked") for t in tdf.itertuples()}
        TD.render_figure(slide, d, df, sel_b, sel_t, sel_p, cores, tagg, magg,
                         float(mz["x_min"]), float(mz.get("x_max", mz["x_min"] + nx * TD.RESOLUTION)),
                         float(mz["y_min"]), float(mz.get("y_max", mz["y_min"] + ny * TD.RESOLUTION)),
                         ny, nx, id_map=id_map)
        n_lab += len(cores)
    print(f"{a.sample}: {n_slides} slides relabelled, {n_lab} cores labelled on figures, "
          f"{n_unlinked} TLS without an object id, {n_nofig} slides without saved masks (figure not redrawn)")


if __name__ == "__main__":
    main()
