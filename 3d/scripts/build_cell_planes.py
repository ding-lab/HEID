#!/usr/bin/env python

from __future__ import annotations
import os

import argparse
import csv
import importlib.util as _iu
import json
import time
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from PIL import Image

ROOT = Path(os.environ.get("THREED_ROOT", os.path.join(os.environ.get("PROJECTS_ROOT", "/data/heid"), "3d")))
RUNS = ROOT / "registration/runs"
SEC = RUNS / "prep_sections/sections"
MANIFEST = RUNS / "prep_sections/manifest.tsv"
PROBE = RUNS / "prep_sections/source_probe.json"
EDGE_RUNS = ("pair_fits",)
SWAP = ROOT / "reconstruction/he_swap_planes"

INF = ROOT / "inference/cohort"
PRED = INF / "cohort/HT891Z1"
CELLS = INF / "cohort/data/cells"
RUNMAN = INF / "configs/cohort/run_manifest.tsv"

SRC_MPP, WG_MPP = 1.0, 2.0
PLANE_MPP = 4.0
BASES = {"G_withdrawn": "R", "G_not_withdrawn": "M"}


CLASS_COLOUR = {
    "Tumor":                    "#D62728",
    "Schwann":                  "#22B14C",
    "NonMalignant_Parenchymal": "#35A16B",
    "Fibroblast":               "#00C2A0",
    "SMC":                      "#8C6239",
    "NK_T":                     "#2E7DF7",
    "Endothelial":              "#FF61C6",
    "Myeloid_NOS":              "#F2C300",
    "B_cell":                   "#FFE100",
    "Plasma":                   "#FF8C1A",
    "Others":                   "#9E9E9E",
}
NOPRED_COLOUR = "#8A8F98"
DEFAULT_ON = []

SIGMA_PX = 1.5

NORM_PCTL = 99.0


NORM_GAMMA = 1.5
HATCH_PERIOD, HATCH_DUTY, HATCH_ALPHA = 10, 3, 0.55
CTRL_SHIFT_PX = 200.0


def _load(name, path):
    s = _iu.spec_from_file_location(name, path)
    m = _iu.module_from_spec(s)
    s.loader.exec_module(m)
    return m


CHAIN = _load("placement_chain", Path(__file__).resolve().parent / "placement_chain.py")


def pred_cells(slide):
    for d in [PRED / slide] + sorted(
            x for x in PRED.iterdir() if x.is_dir() and x.name.startswith(slide)):
        f = d / f"{slide}_cells.csv"
        if f.exists():
            return f
    return None


def slide_for_section():
    rm = {r["slide"]: r for r in csv.DictReader(open(RUNMAN), delimiter="\t")}
    by_file = {Path(r["svs"]).name: r for r in rm.values()}
    probe = {p["section_id"]: p for p in json.loads(PROBE.read_text())}
    out = {}
    for sid, p in probe.items():
        if p["modality"] == "he":
            f = Path(p["path"]).name
            r = by_file.get(f)
            if r is None:
                continue
            H0, W0 = p["level_yx"][0]
            if (int(r["level0_W"]), int(r["level0_H"])) != (W0, H0):
                raise SystemExit(f"{sid}: prep saw {W0}x{H0}, the classifier's "
                                 f"manifest says {r['level0_W']}x{r['level0_H']}")
            out[sid] = {"slide": r["slide"], "route": "he",
                        "level0": [W0, H0], "source": p["path"]}
    for g in sorted(SWAP.glob("*/geom.json")):
        geo = json.loads(g.read_text())
        sid = geo["section_id"]


        if "he_section_file" not in geo:
            continue
        r = by_file.get(geo["he_section_file"])
        if r is None:
            continue
        W0, H0 = int(r["level0_W"]), int(r["level0_H"])
        if geo["he_crop_native_px"] != [W0, H0]:
            raise SystemExit(f"{sid}: geom.json crop {geo['he_crop_native_px']} != "
                             f"the classifier's {W0}x{H0}")
        out[sid] = {"slide": r["slide"], "route": "swap", "level0": [W0, H0],
                    "source": str(Path(r["svs"])),
                    "matrix_plane_to_crop_px": geo["matrix_plane_to_crop_px"],
                    "plane_grid_px": geo["plane_grid_px"],
                    "slide_residual_note": geo["qc"]}
    return out


def read_cells(slide):
    xy = pd.read_parquet(CELLS / f"{slide}.parquet", columns=["cell_id", "x_px", "y_px"])


    f = pred_cells(slide)
    if f is None:
        raise SystemExit(f"no cell table for {slide} under {PRED}")
    cl = pd.read_csv(f, usecols=["cell_id", "cell_type"])
    if len(xy) == len(cl) and (xy["cell_id"].values == cl["cell_id"].values).all():
        d = xy.copy()
        d["cell_type"] = cl["cell_type"].values
    else:
        d = xy.merge(cl, on="cell_id", how="inner")
    return (d["x_px"].to_numpy(np.float64), d["y_px"].to_numpy(np.float64),
            d["cell_type"].to_numpy(), len(xy), len(cl))


def plane_size(shape_hw, out_mpp):
    h, w = shape_hw
    f = SRC_MPP / out_mpp
    return max(1, int(round(w * f))), max(1, int(round(h * f)))


def rescale(v, n_src, n_dst):
    return (v + 0.5) * (float(n_dst) / float(n_src)) - 0.5


_W = {}


def _winit(mapped, classes, cidx, CW, CH, sids, z, norm, subdir, out_root):
    _W.update(mapped=mapped, classes=classes, cidx=cidx, CW=CW, CH=CH,
              sids=sids, z=z, norm=norm, subdir=subdir, out_root=out_root)


def _wsample(args):
    i, bk = args
    ks = int(2 * round(3 * SIGMA_PX) + 1)
    d = _W["mapped"][i]
    X, Y = d["xy"][bk]
    out = {}
    for c in _W["classes"]:
        sel = d["cls"] == _W["cidx"][c]
        if not sel.any():
            continue
        g = np.zeros((_W["CH"], _W["CW"]), np.float32)
        xi = np.round(X[sel]).astype(np.int64)
        yi = np.round(Y[sel]).astype(np.int64)
        k = (xi >= 0) & (xi < _W["CW"]) & (yi >= 0) & (yi < _W["CH"])
        if k.any():
            np.add.at(g, (yi[k], xi[k]), 1.0)
        g = cv2.GaussianBlur(g, (ks, ks), SIGMA_PX)
        v = g[g > 1e-4]
        if v.size:
            out[c] = v[::7].astype(np.float32)
    return out


def _wrender(args):
    i, bk = args
    ks = int(2 * round(3 * SIGMA_PX) + 1)
    mp = _W["mapped"][i]
    X, Y = mp["xy"][bk]
    d = _W["out_root"] / bk / _W["subdir"]
    recs = []
    for c in _W["classes"]:
        sel = mp["cls"] == _W["cidx"][c]
        n = int(sel.sum())
        if not n:
            continue
        g = np.zeros((_W["CH"], _W["CW"]), np.float32)
        xi = np.round(X[sel]).astype(np.int64)
        yi = np.round(Y[sel]).astype(np.int64)
        k = (xi >= 0) & (xi < _W["CW"]) & (yi >= 0) & (yi < _W["CH"])
        if k.any():
            np.add.at(g, (yi[k], xi[k]), 1.0)
        g = cv2.GaussianBlur(g, (ks, ks), SIGMA_PX)
        al = np.clip(g / max(_W["norm"][c], 1e-9), 0, 1) ** NORM_GAMMA
        a8 = (al * 255.0 + 0.5).astype(np.uint8)
        if not a8.any():
            continue
        rgba = np.dstack([np.full_like(a8, 255)] * 3 + [a8])
        fc = c.replace("/", "_")
        fn = f"z{i:02d}_{_W['sids'][i].split('-')[1]}.{fc}.webp"
        fp = d / fn
        Image.fromarray(rgba).save(fp, format="WEBP", lossless=True,
                                   quality=100, method=6)
        recs.append({"index": i, "basis": bk, "section_id": _W["sids"][i],
                     "z_um": float(_W["z"][i]), "cell_class": c,
                     "n_cells": n, "file": f"{bk}/{_W['subdir']}/{fn}",
                     "bytes": int(fp.stat().st_size)})
    return recs


def render_density_planes(mapped, have, classes, cidx, CW, CH, bases, out_root,
                          sids, z, subdir="cells", workers=1):
    if workers > 1:


        import multiprocessing as _mp
        tasks = [(i, bk) for bk in bases for i in have]
        ctx = _mp.get_context("fork")

        with ctx.Pool(workers, initializer=_winit,
                      initargs=(mapped, classes, cidx, CW, CH, sids, z,
                                None, subdir, out_root)) as pool:
            samp = {c: [] for c in classes}
            for out in pool.imap_unordered(_wsample, tasks, chunksize=1):
                for c, v in out.items():
                    samp[c].append(v)
        norm = {}
        for c in classes:
            norm[c] = float(np.percentile(np.concatenate(samp[c]), NORM_PCTL)) \
                if samp[c] else 1.0
        del samp
        print("\nper-class normalisation (density at full alpha):")
        for c in classes:
            print(f"  {c:<26} {norm[c]:.4f}")
        for bk in bases:
            (out_root / bk / subdir).mkdir(parents=True, exist_ok=True)
        planes, nbytes = [], 0
        with ctx.Pool(workers, initializer=_winit,
                      initargs=(mapped, classes, cidx, CW, CH, sids, z,
                                norm, subdir, out_root)) as pool:
            for recs in pool.imap_unordered(_wrender, tasks, chunksize=1):
                planes.extend(recs)
                nbytes += sum(r["bytes"] for r in recs)
        for bk in bases:
            print(f"{bk}: written", flush=True)
        return planes, nbytes, norm

    ks = int(2 * round(3 * SIGMA_PX) + 1)

    def px_of(X, Y):
        xi = np.round(X).astype(np.int64)
        yi = np.round(Y).astype(np.int64)
        k = (xi >= 0) & (xi < CW) & (yi >= 0) & (yi < CH)
        return xi, yi, k

    def dens(X, Y, sel):
        g = np.zeros((CH, CW), np.float32)
        xi, yi, k = px_of(X[sel], Y[sel])
        if k.any():
            np.add.at(g, (yi[k], xi[k]), 1.0)
        return cv2.GaussianBlur(g, (ks, ks), SIGMA_PX)

    samp = {c: [] for c in classes}
    for i in have:
        d = mapped[i]
        for bk in bases:
            X, Y = d["xy"][bk]
            for c in classes:
                sel = d["cls"] == cidx[c]
                if not sel.any():
                    continue
                g = dens(X, Y, sel)
                v = g[g > 1e-4]
                if v.size:
                    samp[c].append(v[::7].astype(np.float32))
    norm = {}
    for c in classes:
        if samp[c]:
            norm[c] = float(np.percentile(np.concatenate(samp[c]), NORM_PCTL))
        else:
            norm[c] = 1.0
    del samp
    print("\nper-class normalisation (density at full alpha):")
    for c in classes:
        print(f"  {c:<26} {norm[c]:.4f}")

    planes, nbytes = [], 0
    for bk in bases:
        d = out_root / bk / subdir
        d.mkdir(parents=True, exist_ok=True)
        for i in have:
            mp = mapped[i]
            X, Y = mp["xy"][bk]
            for c in classes:
                sel = mp["cls"] == cidx[c]
                n = int(sel.sum())
                if not n:
                    continue
                g = dens(X, Y, sel)
                al = np.clip(g / max(norm[c], 1e-9), 0, 1) ** NORM_GAMMA
                a8 = (al * 255.0 + 0.5).astype(np.uint8)
                if not a8.any():
                    continue
                rgba = np.dstack([np.full_like(a8, 255)] * 3 + [a8])


                fc = c.replace("/", "_")
                fn = f"z{i:02d}_{sids[i].split('-')[1]}.{fc}.webp"
                fp = d / fn
                Image.fromarray(rgba).save(fp, format="WEBP", lossless=True,
                                           quality=100, method=6)
                nbytes += fp.stat().st_size
                planes.append({"index": i, "basis": bk, "section_id": sids[i],
                               "z_um": float(z[i]), "cell_class": c,
                               "n_cells": n, "file": f"{bk}/{subdir}/{fn}",
                               "bytes": int(fp.stat().st_size)})
        print(f"{bk}: written", flush=True)
    return planes, nbytes, norm


def main():
    global PRED, PROBE, CELLS, RUNMAN
    global SEC, MANIFEST, EDGE_RUNS
    ap = argparse.ArgumentParser()
    ap.add_argument("--mpp", type=float, default=8.0)
    ap.add_argument("--prep-root", default=str(SEC))
    ap.add_argument("--manifest", default=str(MANIFEST))
    ap.add_argument("--edge-runs", default=",".join(EDGE_RUNS))
    ap.add_argument("--suffix", default="")
    ap.add_argument("--out", default="", help="volume dir to write into; default reconstruction/volume_<mpp>um<suffix>")
    ap.add_argument("--limit", type=int, default=0, help="first N planes only (smoke)")
    ap.add_argument("--anchor", default="HT891Z1-U66",
                    help="must match the anchor the grey volume was built with")
    ap.add_argument("--sample", default="HT891Z1",
                    help="which sample's predictions to read")
    ap.add_argument("--probe", default=str(PROBE),
                    help="prep source_probe.json: section_id -> source file. The join to the classifier runs on the FILE NAME, never on the section id -- the two naming schemes disagree.")
    ap.add_argument("--pred-dir", default="", dest="pred_dir",
                    help="products root holding <slide>/<slide>_cells.csv; default "
                         "inference/cohort/<sample>")
    ap.add_argument("--cells-dir", default="", dest="cells_dir",
                    help="detection tables <slide>.parquet; default cohort's")
    ap.add_argument("--runman", default="",
                    help="classifier run manifest; default cohort's")
    ap.add_argument("--workers", type=int, default=1,
                    help="parallel render processes; 1 = the original serial path")
    ap.add_argument("--bases", default="",
                    help="comma list to build only these bases (parallel split); "
                         "default both")
    a = ap.parse_args()
    PRED = Path(a.pred_dir) if a.pred_dir else INF / "cohort" / a.sample
    if a.cells_dir:
        CELLS = Path(a.cells_dir)
    if a.runman:
        RUNMAN = Path(a.runman)
    PROBE = Path(a.probe)
    all_bases = dict(BASES)
    if a.bases:
        keep = set(a.bases.split(","))
        for k in list(BASES):
            if k not in keep:
                del BASES[k]
    out_mpp = float(a.mpp)
    SEC = Path(a.prep_root)
    MANIFEST = Path(a.manifest)
    EDGE_RUNS = tuple(s.strip() for s in a.edge_runs.split(",") if s.strip())
    out = Path(a.out) if a.out else ROOT / f"reconstruction/volume_{out_mpp:g}um{a.suffix}"
    gmeta = json.loads((out / "metadata.json").read_text())
    t0 = time.time()
    print(f"prep={SEC}\nedges={EDGE_RUNS}\nout={out}", flush=True)

    man = {r["section_id"]: r for r in csv.DictReader(open(MANIFEST), delimiter="\t")}
    sids = sorted(man, key=lambda s: float(man[s]["z_position_um"]))
    z = np.array([float(man[s]["z_position_um"]) for s in sids])


    m8, shapes = {}, {}
    for s in sids:
        m = cv2.imread(str(SEC / s / f"mask_mpp{SRC_MPP:g}.png"), cv2.IMREAD_GRAYSCALE)
        shapes[s] = m.shape
        w, h = plane_size(m.shape, out_mpp)


        m8[s] = cv2.resize((m > 127).astype(np.uint8), (w, h),
                           interpolation=cv2.INTER_AREA) > 0
    print(f"{len(sids)} masks read  ({time.time()-t0:.0f}s)", flush=True)

    edges = CHAIN.harvest(EDGE_RUNS)
    tree = CHAIN.prereg_tree(edges, sids)
    chains = {k: CHAIN.chain_on(tree, edges, b, a.anchor)
              for k, b in all_bases.items()}


    sizes = {s: plane_size(shapes[s], out_mpp) for s in sids}
    pts = []
    for ch in chains.values():
        for s in sids:
            w, h = sizes[s]
            c = np.array([[0, 0], [w, 0], [w, h], [0, h]], float) * (out_mpp / WG_MPP)
            pts.append((c @ ch[s][:2, :2].T + ch[s][:, 2]) * WG_MPP)
    pts = np.vstack(pts)
    o = pts.min(0) - 40.0
    W = int(np.ceil((pts[:, 0].max() + 40.0 - o[0]) / out_mpp))
    H = int(np.ceil((pts[:, 1].max() + 40.0 - o[1]) / out_mpp))

    def M_for(bk, s):
        M = np.zeros((2, 3))
        M[:2, :2] = chains[bk][s][:2, :2]
        M[:, 2] = (WG_MPP / out_mpp) * chains[bk][s][:, 2] - o / out_mpp
        return M

    anym = np.zeros((H, W), bool)
    for bk in chains:
        for s in sids:
            anym |= cv2.warpAffine(m8[s].astype(np.uint8) * 255, M_for(bk, s), (W, H),
                                   flags=cv2.INTER_NEAREST, borderValue=0) > 0
    ys, xs = np.nonzero(anym)
    p = int(round(80.0 / out_mpp))
    r0, r1 = max(0, ys.min() - p), min(H, ys.max() + p + 1)
    c0, c1 = max(0, xs.min() - p), min(W, xs.max() + p + 1)
    CW, CH = c1 - c0, r1 - r0
    want = (gmeta["canvas_px"]["width"], gmeta["canvas_px"]["height"])
    if gmeta.get("canvas_origin_um"):

        o = np.asarray(gmeta["canvas_origin_um"], float)
        c0, r0 = 0, 0
        CW, CH = want
        W, H = want
        print(f"canvas from the volume: {CW} x {CH} at origin {o[0]:.1f}, {o[1]:.1f} um", flush=True)
    print(f"canvas rebuilt {CW} x {CH}; the volume says {want[0]} x {want[1]}", flush=True)
    if (CW, CH) != want:
        raise SystemExit("canvas mismatch: refusing to place cells on a different grid")

    join = slide_for_section()
    have = [i for i, s in enumerate(sids) if s in join]
    lack = [i for i, s in enumerate(sids) if s not in join]
    if a.limit:
        have = have[:a.limit]
    print(f"{len(have)} planes with a prediction, {len(lack)} without", flush=True)


    CLASSES = list(CLASS_COLOUR)
    cidx = {c: i for i, c in enumerate(CLASSES)}
    mapped, rows = {}, []
    for i in have:
        s = sids[i]
        j = join[s]
        x0, y0, ct, n_xy, n_cl = read_cells(j["slide"])
        unk = sorted(set(ct) - set(CLASSES))
        if unk:
            raise SystemExit(f"{s}: unknown classes {unk}")
        w8, h8 = sizes[s]
        if j["route"] == "he":
            W0, H0 = j["level0"]
            x8 = rescale(x0, W0, w8)
            y8 = rescale(y0, H0, h8)
        else:
            A = np.array(j["matrix_plane_to_crop_px"], float)
            inv = np.linalg.inv(A[:, :2])
            q = np.stack([x0 - A[0, 2], y0 - A[1, 2]])
            x4 = inv[0, 0] * q[0] + inv[0, 1] * q[1]
            y4 = inv[1, 0] * q[0] + inv[1, 1] * q[1]
            w4, h4 = j["plane_grid_px"]
            if (w4, h4) != tuple(plane_size(shapes[s], PLANE_MPP)):
                raise SystemExit(f"{s}: geom grid {w4}x{h4} != prep's "
                                 f"{plane_size(shapes[s], PLANE_MPP)}")
            x8 = rescale(x4, w4, w8)
            y8 = rescale(y4, h4, h8)
        cls = np.array([cidx[c] for c in ct], np.int8)
        per = {}
        for bk in BASES:
            M = M_for(bk, s)
            X = M[0, 0] * x8 + M[0, 1] * y8 + M[0, 2] - c0
            Y = M[1, 0] * x8 + M[1, 1] * y8 + M[1, 2] - r0
            per[bk] = (X.astype(np.float32), Y.astype(np.float32))
        mapped[i] = {"cls": cls, "xy": per, "n": len(cls),
                     "route": j["route"], "slide": j["slide"]}
        rows.append({"index": i, "section_id": s, "slide": j["slide"],
                     "route": j["route"], "n_cells": int(len(cls)),
                     "n_detected": int(n_xy), "n_classified": int(n_cl)})
        print(f"  {s:<14} {j['slide']:<22} {j['route']:<4} {len(cls):7d} cells",
              flush=True)


    def alpha_of(bk, i):
        f = out / gmeta["encodings"][bk]["planes"][i]["webp_lossless"]
        return np.array(Image.open(f).convert("RGBA"))[..., 3]

    def px_of(X, Y):
        xi = np.round(X).astype(np.int64)
        yi = np.round(Y).astype(np.int64)
        k = (xi >= 0) & (xi < CW) & (yi >= 0) & (yi < CH)
        return xi, yi, k

    def hit_frac(X, Y, al):
        xi, yi, k = px_of(X, Y)
        if not len(xi):
            return 0.0, 0
        return float(al[yi[k], xi[k]].sum()) / len(xi), int((~k).sum())

    def foot(X, Y):
        g = np.zeros((CH, CW), np.uint8)
        xi, yi, k = px_of(X, Y)
        g[yi[k], xi[k]] = 255
        g = cv2.morphologyEx(g, cv2.MORPH_CLOSE, np.ones((9, 9), np.uint8))
        return g > 0

    checks = []
    for i in have:
        d = mapped[i]
        for bk in BASES:
            al = alpha_of(bk, i) > 0
            X, Y = d["xy"][bk]
            inm, noff = hit_frac(X, Y, al)
            f = foot(X, Y)
            iou = float((f & al).sum() / max(1, (f | al).sum()))


            far = sids[(i + len(sids) // 2) % len(sids)]
            Mf = M_for(bk, far)
            Mi = M_for(bk, sids[i])
            det = Mi[0, 0] * Mi[1, 1] - Mi[0, 1] * Mi[1, 0]
            Ii = np.array([[Mi[1, 1], -Mi[0, 1]], [-Mi[1, 0], Mi[0, 0]]]) / det
            px = Ii[0, 0] * (X + c0 - Mi[0, 2]) + Ii[0, 1] * (Y + r0 - Mi[1, 2])
            py = Ii[1, 0] * (X + c0 - Mi[0, 2]) + Ii[1, 1] * (Y + r0 - Mi[1, 2])
            Xf = Mf[0, 0] * px + Mf[0, 1] * py + Mf[0, 2] - c0
            Yf = Mf[1, 0] * px + Mf[1, 1] * py + Mf[1, 2] - r0
            ctl_far, _ = hit_frac(Xf, Yf, al)
            ctl_sh, _ = hit_frac(X + CTRL_SHIFT_PX, Y + CTRL_SHIFT_PX, al)
            checks.append({"index": i, "basis": bk, "section_id": sids[i],
                           "route": d["route"], "n_cells": int(d["n"]),
                           "off_canvas": int(noff),
                           "in_mask": round(inm, 4),
                           "iou_footprint_vs_alpha": round(iou, 4),
                           "control_far_section": round(ctl_far, 4),
                           "control_shift_200px": round(ctl_sh, 4)})
    worst = sorted(checks, key=lambda c: c["in_mask"])[:6]
    print("\nworst in_mask:")
    for c in worst:
        print(f"  {c['section_id']:<14} {c['basis']:<16} {c['route']:<4} "
              f"in_mask {c['in_mask']:.4f}  IoU {c['iou_footprint_vs_alpha']:.3f}  "
              f"far {c['control_far_section']:.3f}  shift {c['control_shift_200px']:.3f}",
              flush=True)


    planes, nbytes, norm = render_density_planes(
        mapped, have, CLASSES, cidx, CW, CH, list(BASES), out, sids, z,
        workers=a.workers)
    for bk in BASES:
        d = out / bk / "cells"


        for i in lack:
            al = alpha_of(bk, i) > 0
            yy, xx = np.mgrid[0:CH, 0:CW]
            hat = ((xx + yy) % HATCH_PERIOD) < HATCH_DUTY
            a8 = (al & hat).astype(np.uint8) * int(round(HATCH_ALPHA * 255))
            rgba = np.dstack([np.full_like(a8, 255)] * 3 + [a8])
            fn = f"z{i:02d}_{sids[i].split('-')[1]}.nopred.webp"
            fp = d / fn
            Image.fromarray(rgba).save(fp, format="WEBP", lossless=True, quality=100,
                                       method=6)
            nbytes += fp.stat().st_size
            planes.append({"index": i, "basis": bk, "section_id": sids[i],
                           "z_um": float(z[i]), "cell_class": None,
                           "n_cells": 0, "file": f"{bk}/cells/{fn}",
                           "bytes": int(fp.stat().st_size)})
        print(f"{bk}: written ({time.time()-t0:.0f}s)", flush=True)

    tot = {c: int(sum(int((mapped[i]["cls"] == cidx[c]).sum()) for i in have))
           for c in CLASSES}

    meta = {
        "what": "Cell cell-type predictions as plane textures on the volume's canvas",
        "claim": "MODEL PREDICTION on H&E, no spatial ground truth. The classifier is "
                 "pan-cancer Cell head (cls_sigma3, single fold), applied to H&E. Nothing "
                 "here was scored against a spatial truth, because this cohort has "
                 "none.",
        "in_plane_um_per_px": out_mpp,
        "canvas_px": {"width": int(CW), "height": int(CH)},
        "canvas_mm": gmeta["canvas_mm"],
        "n_planes_with_prediction": len(have),
        "n_planes_without": len(lack),
        "indices_with": [int(i) for i in have],
        "indices_without": [int(i) for i in lack],
        "without_reason": "these nine planes are Xenium sections: the volume's frame "
                          "for them is the Xenium DAPI image, and no transform from an "
                          "H&E scan into that frame exists in this reconstruction. "
                          "They are drawn as a hatched tissue mask so 'not predicted' "
                          "is not read as 'no cells'.",
        "geometry": {
            "source": "recomputed by the same code path as build_he_rgb.py and "
                      "build_volume_planes.py (the placement chain: harvest -> prereg "
                      "tree -> chain_on from the anchor), then asserted against the "
                      "grey volume's canvas_px",
            "edge_runs": list(EDGE_RUNS),
            "prep_root": str(SEC),
            "anchor": a.anchor,
            "steps_he": "slide level-0 px -> plane grid at this mpp (the resize applied "
                        "during section preparation, pixel-centre convention) -> chain matrix -> "
                        "canvas crop",
            "steps_swap": "H&E crop level-0 px -> inverse of geom.json's "
                          "matrix_plane_to_crop_px (4 um plane grid) -> plane grid at "
                          "this mpp -> chain matrix -> canvas crop",
            "swap_accuracy": "the slide-level placement's, 4.9-38.1 um depending on "
                             "the slide; no per-section fit exists and none was made",
        },
        "classes": [{"name": c, "colour": CLASS_COLOUR[c], "n_cells": tot[c],
                     "norm_density_full_alpha": round(norm[c], 5)} for c in CLASSES],
        "default_on": DEFAULT_ON,
        "nopred_colour": NOPRED_COLOUR,
        "encoding": {
            "format": "WebP lossless RGBA",
            "note": "RGB is constant 255; the DATA is the alpha channel. The page "
                    "multiplies it by the class colour with the tint path the shader "
                    "already has, so one class is exactly one colour.",
            "density": f"cells per {out_mpp:g} um pixel, Gaussian sigma {SIGMA_PX} px "
                       f"({SIGMA_PX*out_mpp:g} um)",
            "normalisation": f"per class, alpha = (density / p{NORM_PCTL:g}) ** "
                             f"{NORM_GAMMA}, clipped. The percentile is that class's "
                             f"own, pooled over every plane and both bases, so a class "
                             f"is on one ramp across the stack. Brightness is NOT "
                             f"comparable BETWEEN classes. The exponent is above 1 "
                             f"because the page composites 41 planes at 10% each: at "
                             f"0.65 the low end accumulated into a solid block with no "
                             f"structure left in it.",
            "bytes": int(nbytes),
        },
        "hatch": {"period_px": HATCH_PERIOD, "duty_px": HATCH_DUTY,
                  "alpha": HATCH_ALPHA},
        "sections": rows,


        "planes": planes,
        "checks": checks,
        "checks_note": "in_mask: cells landing on the plane's own tissue alpha. "
                       "iou_footprint_vs_alpha: the falsifiable one -- a wrong scale, "
                       "rotation or origin moves the footprint and IoU collapses while "
                       "in_mask barely moves. control_far_section: the same cells "
                       "through the pose of the section half a stack away. "
                       "control_shift_200px: the correct chain, translated 200 px.",
        "worst_in_mask": worst,
        "elapsed_s": round(time.time() - t0, 1),
        "read_only_upstream": True,
    }
    mname = ("cells_metadata_" + "_".join(BASES) + ".json") if a.bases \
        else "cells_metadata.json"
    (out / mname).write_text(json.dumps(meta, indent=1))
    print(f"\nwrote {out/'cells_metadata.json'}  {len(planes)} textures, "
          f"{nbytes/1e6:.1f} MB  ({meta['elapsed_s']} s)")


if __name__ == "__main__":
    main()
