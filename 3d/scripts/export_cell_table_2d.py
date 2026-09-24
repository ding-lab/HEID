#!/usr/bin/env python3

import argparse
import csv
import json
import re
import struct
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

UM = 8.0
MATCH_UM = 15.0


def read_points(path):
    b = Path(path).read_bytes()
    magic, n, cw, ch, sub, ncls = struct.unpack_from("<8sIHHHH", b, 0)
    assert magic == b"CELLPT01"
    xy = np.frombuffer(b, np.uint16, 2 * n, 32).reshape(n, 2).astype(np.float64) / sub
    cls = np.frombuffer(b, np.uint8, n, 32 + 4 * n + n)
    return xy * UM, cls


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--volume", required=True)
    ap.add_argument("--sample", required=True)
    ap.add_argument("--pred", default="", help="H&E prediction parquet dir (cell_id, x_um, y_um, pred_class_name)")
    ap.add_argument("--join", default="", help="celltype_join dir: cells/<slide>.parquet + products/<slide>/<slide>_cells.csv")
    ap.add_argument("--out", required=True)
    ap.add_argument("--xenium-gt", default="", dest="xgt",
                    help="<raw>/xenium_gt_<panel> (canvas-placed Xenium cells)")
    ap.add_argument("--anno", default="", help="<raw>/xenium_annotation_<panel> (cell_id,group)")
    a = ap.parse_args()
    vol, out = Path(a.volume), Path(a.out) / a.sample
    out.mkdir(parents=True, exist_ok=True)
    meta = json.loads((vol / "cell_points_metadata.json").read_text())
    vmeta = json.loads((vol / "metadata.json").read_text())
    classes = meta["classes"]
    slide_of = {s["section_id"]: s["slide"] for s in meta["sections"]}
    planes = sorted(vmeta["encodings"]["G_withdrawn"]["planes"], key=lambda p: p["z_um"])
    native_mpp = {}
    if a.join:
        with (Path(a.join) / "run_manifest.tsv").open() as stream:
            for row in csv.DictReader(stream, delimiter="\t"):
                slide = row["slide"]
                if slide in native_mpp:
                    raise ValueError(f"Duplicate slide in join manifest: {slide}")
                mpp = float(row["native_mpp"])
                if not np.isfinite(mpp) or mpp <= 0:
                    raise ValueError(f"Invalid native_mpp for {slide}: {mpp}")
                native_mpp[slide] = mpp


    xen = {}
    if a.xgt:
        xm = json.loads((Path(a.xgt) / "xenium_gt_metadata.json").read_text())
        for f in xm["files"]:
            if f["basis"] == "G_withdrawn":
                xen[f["section_id"]] = Path(a.xgt) / f["file"]
    anno, anno_by_n = {}, {}
    if a.anno:
        for p in sorted(Path(a.anno).glob("*_cell_annotation.csv")):
            anno[p.name.split("_cell_annotation")[0]] = p
            n = sum(1 for _ in open(p)) - 1
            anno_by_n.setdefault(n, []).append(p)

    rows_sum = []
    for p in planes:
        sid = p["section_id"]
        slide = slide_of.get(sid)
        cand = sorted((vol / "G_withdrawn/cell_points").glob(f"z{p['index']:02d}_*.cells.bin"))
        pred = None
        if a.join and slide:


            cp = Path(a.join) / "cells" / f"{slide}.parquet"
            pp = Path(a.join) / "products" / slide / f"{slide}_cells.csv"
            if cp.exists() and pp.exists():
                xy = pd.read_parquet(cp, columns=["cell_id", "x_px", "y_px"])
                ct = pd.read_csv(pp, usecols=["cell_id", "cell_type"])
                pred = xy.merge(ct, on="cell_id", how="inner")


                mpp = native_mpp[slide]
                pred["x_um"] = pred["x_px"] * mpp
                pred["y_um"] = pred["y_px"] * mpp
                pred = pred.rename(columns={"cell_type": "pred_class_name"})
        elif a.pred and slide:
            pq = Path(a.pred) / f"{slide}.parquet"
            if not pq.exists():
                alt = Path(a.pred) / (re.sub(r"-P\d+_", "_", str(slide)) + ".parquet")
                if alt.exists():
                    pq = alt
            if pq.exists():
                pred = pd.read_parquet(pq, columns=["cell_id", "x_um", "y_um", "pred_class_name"])
        if not slide or not cand or pred is None:
            print(f"  {sid}: no prediction table, skipped", flush=True)
            continue
        xy_um, cls = read_points(cand[0])
        if len(pred) != len(xy_um):
            print(f"  {sid}: {len(pred)} predicted vs {len(xy_um)} placed, skipped", flush=True)
            continue
        idx = pred["pred_class_name"].map({c: i for i, c in enumerate(classes)}).to_numpy()
        assert np.array_equal(idx, cls), f"{sid}: class order differs between the two products"
        df = pd.DataFrame({
            "cell_id": pred["cell_id"].astype(str),
            "x_um": np.round(xy_um[:, 0], 2), "y_um": np.round(xy_um[:, 1], 2),
            "x_native_um": np.round(pred["x_um"].to_numpy(), 2),
            "y_native_um": np.round(pred["y_um"].to_numpy(), 2),
            "HE_celltype": pred["pred_class_name"].astype(str)})
        df["xenium_celltype"] = ""
        df["xenium_cell_id"] = ""
        df["xenium_dist_um"] = np.nan
        matched = 0
        if sid in xen:
            xxy, _xc = read_points(xen[sid])
            gt = None
            for k, pth in anno.items():
                if k == slide or k.replace("_", "") == str(slide).replace("_", ""):
                    gt = pd.read_csv(pth)
                    break
            ids = gt["cell_id"].astype(str).to_numpy() if gt is not None else None
            grp = gt["group"].astype(str).to_numpy() if gt is not None else None
            if ids is not None and len(ids) != len(xxy):


                alt = anno_by_n.get(len(xxy), [])
                if len(alt) == 1:
                    gt = pd.read_csv(alt[0])
                    ids = gt["cell_id"].astype(str).to_numpy()
                    grp = gt["group"].astype(str).to_numpy()
                    print(f"  {sid}: annotation file for this slide has {len(pd.read_csv(pth))} "
                          f"cells, placed {len(xxy)}; using {alt[0].name} (same count)", flush=True)
                else:
                    print(f"  {sid}: annotation {len(ids)} vs placed {len(xxy)} Xenium cells, "
                          f"ids not attached", flush=True)
                    ids = grp = None
            t = cKDTree(xxy)
            d, j = t.query(np.column_stack([df["x_um"], df["y_um"]]), distance_upper_bound=MATCH_UM)
            ok = np.isfinite(d)
            matched = int(ok.sum())
            if grp is not None:
                df.loc[ok, "xenium_celltype"] = grp[j[ok]]
                df.loc[ok, "xenium_cell_id"] = ids[j[ok]]
            df.loc[ok, "xenium_dist_um"] = np.round(d[ok], 2)
        f = out / f"{slide}.tsv"
        df.to_csv(f, sep="\t", index=False)
        agree = ""
        if matched:
            both = df[(df.xenium_celltype != "")]
            same = (both.HE_celltype.str.lower().str.replace("_", "")
                    == both.xenium_celltype.str.lower().str.replace("_", "")).mean()
            agree = round(float(same), 3)
        rows_sum.append({"section_id": sid, "slide": slide, "z_um": p["z_um"],
                         "n_cells": len(df), "n_xenium_matched": matched,
                         "frac_matched": round(matched / max(1, len(df)), 3),
                         "exact_label_agreement": agree, "file": f.name})
        print(f"  {sid}: {len(df)} cells, {matched} matched to Xenium "
              f"({100 * matched / max(1, len(df)):.1f}%)", flush=True)
    with open(out / "summary.tsv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows_sum[0].keys()), delimiter="\t")
        w.writeheader(); w.writerows(rows_sum)
    print(f"wrote {out}: {len(rows_sum)} slides", flush=True)


if __name__ == "__main__":
    main()
