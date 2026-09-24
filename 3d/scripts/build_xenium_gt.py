#!/usr/bin/env python

from __future__ import annotations
import os

import argparse
import csv
import importlib.util
import json
import struct
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(os.environ.get("THREED_ROOT", os.path.join(os.environ.get("PROJECTS_ROOT", "/data/heid"), "3d")))
ANN = ROOT / "data/xenium_annotation"
ALIGN = ROOT / "xenium"


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec)
    sys.modules[name] = m
    spec.loader.exec_module(m)
    return m


import time as _time

def _lap(msg, _st=[None]):
    now=_time.time(); print(f"    timing: {msg} {now - (_st[0] or now):.0f} s", flush=True); _st[0]=now


def main():
    global ANN
    ap = argparse.ArgumentParser()
    ap.add_argument("--sample", default="HT891Z1")
    ap.add_argument("--volume", default=str(ROOT / "reconstruction/volume_8um"))
    ap.add_argument("--registration",
                    default=str(ALIGN / "outputs/registration.json"))
    ap.add_argument("--annotation-dir", default=str(ANN), dest="annotation_dir",
                    help="<sample>-<slide>_cell_annotation.csv files (cell_id, group)")
    ap.add_argument("--out", default="", help="output dir; default <volume>/xenium_gt")
    ap.add_argument("--placer-args", default="", dest="placer_args",
                    help="extra argv for the canvas placer, space-separated -- the "
                         "volume-specific prep/manifest/edges/anchor flags")
    ap.add_argument("--via-registration", action="store_true", dest="via_registration",
                    help="the sample's planes are H&E, not the Xenium image: take each cell from "
                         "Xenium microns to H&E microns through the registration's inverse, then "
                         "to H&E level-0 pixels, and place it through the H&E chain")
    a = ap.parse_args()

    vol = Path(a.volume)
    meta = json.loads((vol / "metadata.json").read_text())
    reg = json.loads(Path(a.registration).read_text())
    secs = reg["sections"]

    REG = _load("reg2", Path(__file__).resolve().parent / "xenium_register.py")
    rman = {r["slide"]: r for r in csv.DictReader(
        open(ROOT / "inference/cohort/configs/cohort/run_manifest.tsv"), delimiter="\t")}

    out_dir = Path(a.out) if a.out else vol / "xenium_gt"
    out_dir.mkdir(parents=True, exist_ok=True)
    ANN = Path(a.annotation_dir)
    classes, per_plane, dropped = Counter(), {}, {}


    ann = {}
    for f in sorted(ANN.glob(f"{a.sample}-*_cell_annotation.csv")):
        sid = f.name.replace("_cell_annotation.csv", "")
        with open(f) as fh:
            ann[sid] = {r["cell_id"]: r["group"] for r in csv.DictReader(fh)}
        classes.update(ann[sid].values())
    if not ann:
        raise SystemExit(f"no annotation files for {a.sample} under {ANN}")


    mj = json.loads((vol / "cells_metadata.json").read_text())["classes"]


    UI_HEAD = ["Tumor", "NK_T", "B_cell", "Schwann"]
    draw_order = [c["name"] for c in mj]
    def ui_rank(c):
        if c in UI_HEAD:
            return (0, UI_HEAD.index(c))
        if c == "Others":
            return (2, 0)
        return (1, draw_order.index(c) if c in draw_order else 99)
    model_order = sorted(draw_order, key=ui_rank)
    model_col = {c["name"]: c["colour"] for c in mj}
    seen = set(classes)
    names = [c for c in model_order if c in seen]
    names += [c for c, _ in classes.most_common() if c not in model_order]
    EXTRA = ["#E8743B", "#00A0B0", "#B07AA1", "#7FBF5F", "#9C755F",
             "#D4A017", "#5B8FF9", "#C0392B", "#7F8C8D"]
    colours = [model_col.get(c, EXTRA[i % len(EXTRA)])
               for i, c in enumerate(names)]
    idx = {c: i for i, c in enumerate(names)}
    _lap("read annotation")
    print(f"{a.sample}: {len(ann)} annotated planes, {len(names)} classes, "
          f"{sum(classes.values()):,} annotated cells")


    runs = {}
    for k, v in secs.items():
        d = Path(v["xenium_dir"])
        cells, *_ = REG.load_xenium(d)
        runs[k] = (v, cells, set(cells["cell_id"]))
    for sid, table in ann.items():
        want = set(table)
        best, frac = None, 0.0
        for k, (v, cells, ids) in runs.items():
            f = len(want & ids) / max(len(want), 1)
            if f > frac:
                best, frac = k, f
        if best is None or frac < 0.90:
            dropped[sid] = f"no Xenium run reproduces its cell ids (best {frac:.1%})"
            continue
        vol_sid, rec = best, runs[best][0]
        cells = runs[best][1]
        keep = cells["cell_id"].isin(table.keys())
        n_have, n_ann = int(keep.sum()), len(table)
        if not n_have:
            dropped[sid] = "no cell_id in common"
            continue
        sub = cells[keep]
        xy = np.stack([sub["x_centroid"].to_numpy(), sub["y_centroid"].to_numpy()])


        if a.via_registration:


            aff = rec["affine_he_um_to_xenium_um"]
            A, b = np.asarray(aff["A"], float), np.asarray(aff["b"], float)
            xy_he_um = np.linalg.solve(A, (xy.T - b).T).T
            per_plane[vol_sid] = {"slide": sid, "xy_he_um": xy_he_um.astype(np.float64),
                                  "xy_px0": None, "level0_wh": None,
                                  "cls": np.array([idx[table[c]] for c in sub["cell_id"]], np.uint8),
                                  "cell_id": sub["cell_id"].astype(str).to_numpy(),
                                  "n_annotated": n_ann, "n_placed": n_have}
            print(f"  {vol_sid:<14} {sid:<26} {n_have:>7,} / {n_ann:>7,} cells, via the registration"); _lap(f"place {vol_sid}")
            continue
        dapi_px = xy.T / float(rec["pixel_size_um"])


        px = float(rec["pixel_size_um"]) if rec.get("pixel_size_um") else None
        per_plane[vol_sid] = {
            "slide": sid,
            "xy_px0": dapi_px.astype(np.float32),
            "level0_wh": rec["dapi_level0_wh"],
            "cls": np.array([idx[table[c]] for c in sub["cell_id"]], np.uint8),
            "cell_id": sub["cell_id"].astype(str).to_numpy(),
            "n_annotated": n_ann, "n_placed": n_have,
        }
        print(f"  {vol_sid:<14} {sid:<26} {n_have:>7,} / {n_ann:>7,} cells placed"); _lap(f"place {vol_sid}")


    CP = _load("cp", Path(__file__).resolve().parent / "build_cell_points.py")


    place = CP.canvas_placer(vol, meta, ["--out", str(vol)] + a.placer_args.split())
    if a.via_registration:


        join = CP.CP.slide_for_section()
        for vol_sid, v in per_plane.items():
            if vol_sid not in join or vol_sid not in place["he_level0_wh"]:
                raise SystemExit(f"{vol_sid}: no H&E slide behind this plane (probe/manifest), cannot place its Xenium cells")
            mpp = float(rman[join[vol_sid]["slide"]]["native_mpp"])
            v["xy_px0"] = (v.pop("xy_he_um") / mpp).astype(np.float32)
            v["level0_wh"] = list(place["he_level0_wh"][vol_sid])
            inside = ((v["xy_px0"] >= 0) & (v["xy_px0"] < np.asarray(v["level0_wh"], float))).all(1).mean()
            print(f"  {vol_sid}: H&E slide {join[vol_sid]['slide']} at {mpp:.4f} um/px, "
                  f"{inside:.1%} of the Xenium cells land inside it", flush=True)
    out_pts = out_dir / "cell_points"
    out_pts.mkdir(parents=True, exist_ok=True)
    files = []
    mapped = {}
    for vol_sid, v in per_plane.items():
        for bk in place["bases"]:

            xy = place["to_canvas"](bk, vol_sid, v["xy_px0"], tuple(v["level0_wh"]))
            if xy is None:
                continue
            mp = mapped.setdefault(place["index_of"](vol_sid),
                                   {"cls": v["cls"], "xy": {}})
            mp["xy"][bk] = (xy[:, 0].astype(np.float32), xy[:, 1].astype(np.float32))
            n = len(xy)
            head = struct.pack("<8sIHHHHI", b"CELLPT01", n,
                               place["W"], place["H"], place["sub"], len(names), 0)
            head += b"\0" * (32 - len(head))
            xi = np.clip(np.round(xy[:, 0] * place["sub"]), 0,
                         place["W"] * place["sub"] - 1).astype(np.uint16)
            yi = np.clip(np.round(xy[:, 1] * place["sub"]), 0,
                         place["H"] * place["sub"] - 1).astype(np.uint16)
            ip = np.empty(2 * n, np.uint16); ip[0::2] = xi; ip[1::2] = yi
            r = np.full(n, place["r_default"], np.uint8)
            idx_i = place["index_of"](vol_sid)
            fn = f"z{idx_i:02d}_{vol_sid.split('-')[1]}.cells.bin"
            fp = out_pts / bk / fn
            fp.parent.mkdir(parents=True, exist_ok=True)
            assert n == len(v["cls"]) == len(v["cell_id"]), (vol_sid, n, len(v["cls"]))
            fp.write_bytes(head + ip.tobytes() + r.tobytes() + v["cls"].tobytes())

            np.save(fp.with_name(fn.replace(".cells.bin", ".cell_ids.npy")), v["cell_id"].astype("U64"))
            files.append({"index": idx_i, "basis": bk, "section_id": vol_sid,
                          "file": f"cell_points/{bk}/{fn}", "n_cells": n})


    CPL = _load("cpl", Path(__file__).resolve().parent / "build_cell_planes.py")
    have = sorted(mapped)
    dsids = {place["index_of"](sid): sid for sid in per_plane}
    zs = {i: float(meta["z_um"][i]) for i in have}
    dens_planes, dens_bytes, dens_norm = CPL.render_density_planes(
        mapped, have, names, idx, place["W"], place["H"], place["bases"],
        out_dir, dsids, zs)

    rep = {"what": "Xenium cell-type annotation in the volume's canvas frame",
           "files": files, "sub_px": place["sub"],
           "canvas_px": {"width": place["W"], "height": place["H"]},
           "sample": a.sample, "classes": names, "colours": colours,
           "counts": {c: int(classes[c]) for c in names},
           "shared_with_model": [c for c in names if c in model_col],
           "xenium_only": [c for c in names if c not in model_col],
           "n_planes": len(per_plane), "dropped": dropped,
           "density": {"planes": dens_planes,
                       "norm_full_alpha": {c: round(dens_norm[c], 5) for c in names}},
           "claim": "ANNOTATION MADE ON XENIUM, not a model prediction. A different "
                    "taxonomy from the H&E model layer; the two are never merged.",
           "planes": {k: {"n_annotated": v["n_annotated"], "n_placed": v["n_placed"]}
                      for k, v in per_plane.items()}}
    (out_dir / "xenium_gt_metadata.json").write_text(
        json.dumps(rep, indent=1, default=lambda o: int(o)))
    _lap("rasterise + write"); print(f"wrote {out_dir}")


if __name__ == "__main__":
    main()
