#!/usr/bin/env python3
from __future__ import annotations

import importlib.util
import json
import os
import struct
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

P3D = Path(os.environ.get("THREED_ROOT", os.path.join(os.environ.get("PROJECTS_ROOT", "/data/heid"), "3d")))
VOL = P3D / "reconstruction/volume_8um"
XG = VOL / "xenium_gt"
OUTROOT = P3D / "tls_define/xenium_tls"
COLOUR = "#FF2D95"

spec = importlib.util.spec_from_file_location(
    "tls_he", Path(__file__).resolve().parent / "tls_define_he.py")
TD = importlib.util.module_from_spec(spec)
spec.loader.exec_module(TD)


def read_cells_bin(fp: Path):
    raw = fp.read_bytes()
    magic, n, W, H, sub, ncls, _ = struct.unpack("<8sIHHHHI", raw[:24])
    assert magic == b"CELLPT01", fp
    ip = np.frombuffer(raw, np.uint16, 2 * n, 32)
    off = 32 + 4 * n
    cls = np.frombuffer(raw, np.uint8, n, off + n)
    return ip[0::2] / sub, ip[1::2] / sub, cls, W, H


def main():
    global VOL, XG, OUTROOT
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--volume", default=str(VOL))
    ap.add_argument("--layer", default="", help="xenium_gt layer dir; default <volume>/xenium_gt")
    ap.add_argument("--out", default="", help="tables/masks root; default the 5k location")
    a = ap.parse_args()
    VOL = Path(a.volume)
    XG = Path(a.layer) if a.layer else VOL / "xenium_gt"
    if a.out:
        OUTROOT = Path(a.out)
    g = json.loads((XG / "xenium_gt_metadata.json").read_text())
    names = g["classes"]
    cm = json.loads((VOL / "cells_metadata.json").read_text())
    W, H = g["canvas_px"]["width"], g["canvas_px"]["height"]
    um_px = cm["canvas_mm"]["width"] * 1000.0 / W
    assert abs(um_px - cm["canvas_mm"]["height"] * 1000.0 / H) < 0.05, "anisotropic canvas"
    os.environ["TLS_SAVE_MASKS"] = "1"

    by_basis = {}
    for q in g["files"]:
        by_basis.setdefault(q["basis"], []).append(q)

    all_rows, meta_planes = [], {}
    for basis, files in sorted(by_basis.items()):
        pq_dir = OUTROOT / "parquet" / basis
        out_dir = OUTROOT / basis
        tls_dir = XG / basis / "tls"
        for d in (pq_dir, out_dir, tls_dir):
            d.mkdir(parents=True, exist_ok=True)
        TD.PRED = str(pq_dir)

        for q in sorted(files, key=lambda r: r["index"]):
            sid = q["section_id"]
            x_px, y_px, cls, bw, bh = read_cells_bin(XG / q["file"])
            assert (bw, bh) == (W, H), (sid, bw, bh)
            pd.DataFrame({
                "x_um": x_px * um_px, "y_um": y_px * um_px,
                "pred_class_name": np.array(names, object)[cls],
            }).to_parquet(pq_dir / f"{sid}.parquet")
            TD.main(sid.split("-")[0], sid, str(out_dir))

            m = np.load(out_dir / f"{sid}_masks.npz")
            cores, x0, y0, res = (m["cores"], float(m["x_min"]),
                                  float(m["y_min"]), float(m["resolution"]))

            members = None
            ids_fp = XG / q["file"].replace(".cells.bin", ".cell_ids.npy")
            if basis == "G_withdrawn" and ids_fp.exists() and len(cores):
                ids = np.load(ids_fp, allow_pickle=True).astype(str)
                assert len(ids) == len(x_px), (sid, len(ids), len(x_px))
                gi = np.floor((y_px * um_px - y0) / res).astype(int)
                gj = np.floor((x_px * um_px - x0) / res).astype(int)
                ok = (gi >= 0) & (gj >= 0) & (gi < cores.shape[1]) & (gj < cores.shape[2])
                is_b = np.isin(np.array(names, object)[cls], TD.B_TYPES)
                members = []
                for k in range(len(cores)):
                    inside = np.zeros(len(ids), bool)
                    inside[ok] = cores[k, gi[ok], gj[ok]]
                    members.append((ids[inside], int((inside & is_b).sum())))
            canvas = np.zeros((H, W), np.uint8)
            if len(cores):
                un = cores.any(0)
                gh, gw = un.shape

                tw = max(1, int(round(gw * res / um_px)))
                th = max(1, int(round(gh * res / um_px)))
                up = cv2.resize((un * 255).astype(np.uint8), (tw, th),
                                interpolation=cv2.INTER_LINEAR)
                ox = int(round(x0 / um_px)); oy = int(round(y0 / um_px))
                sx0, sy0 = max(0, -ox), max(0, -oy)
                dx0, dy0 = max(0, ox), max(0, oy)
                cw_ = min(tw - sx0, W - dx0); ch_ = min(th - sy0, H - dy0)
                if cw_ > 0 and ch_ > 0:
                    canvas[dy0:dy0+ch_, dx0:dx0+cw_] = up[sy0:sy0+ch_, sx0:sx0+cw_]
            fill = canvas > 127
            er = cv2.erode(fill.astype(np.uint8), np.ones((5, 5), np.uint8)) > 0
            rim = fill & ~er
            a8 = np.zeros((H, W), np.uint8)
            a8[fill] = 60
            a8[rim] = 255
            rgba = np.dstack([np.full_like(a8, 255)] * 3 + [a8])
            fn = f"z{q['index']:02d}_{sid.split('-')[1]}.tls.webp"
            from PIL import Image
            Image.fromarray(rgba).save(tls_dir / fn, format="WEBP",
                                       lossless=True, quality=100, method=4)

            tj = json.loads(open(out_dir / f"{sid}_meta.json").read()) \
                if (out_dir / f"{sid}_meta.json").exists() else {}
            n_tls = int(tj.get("n_tls_accepted", len(cores)))
            mp = meta_planes.setdefault(q["index"], {
                "index": q["index"], "section_id": sid, "n_tls": n_tls, "file": fn})
            assert mp["file"] == fn
            tcsv = out_dir / f"{sid}_tls.csv"
            if basis == "G_withdrawn" and tcsv.exists():
                try:
                    t = pd.read_csv(tcsv)
                    t.insert(0, "index", q["index"])
                    if members is not None and len(members) == len(t):
                        t["n_cells_core"] = [len(mm[0]) for mm in members]
                        t["n_b_core_from_ids"] = [mm[1] for mm in members]
                        t["cell_ids"] = [";".join(mm[0]) for mm in members]
                        print(f"    {sid}: cell ids per TLS core " + ", ".join(
                            f"{r.tls_id} {len(mm[0])} cells (B+NK_T {mm[1]} vs table n_b_core {r.n_b_core})"
                            for r, mm in zip(t.itertuples(), members)), flush=True)
                    all_rows.append(t)
                except pd.errors.EmptyDataError:
                    pass
            print(f"  {basis} {sid}: {n_tls} TLS", flush=True)

    (XG / "xenium_tls_metadata.json").write_text(json.dumps({
        "what": "TLS regions called on the Xenium annotation itself "
                "(TLS Define geometry: B_cell density core + halo gates)",
        "colour": COLOUR, "um_per_px": um_px,
        "planes": sorted(meta_planes.values(), key=lambda r: r["index"]),
        "claim": "Xenium ANNOTATION, not a model prediction; geometry-only TLS "
                 "call (no molecular grading)."}, indent=1))
    if all_rows:
        pd.concat(all_rows).to_csv(OUTROOT / "xenium_tls_table.tsv", sep="\t", index=False)
    n = sum(r["n_tls"] for r in meta_planes.values())
    print(f"wrote {len(meta_planes)} planes, {n} TLS total; metadata + table done")


if __name__ == "__main__":
    sys.exit(main())
