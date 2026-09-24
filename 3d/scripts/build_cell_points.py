#!/usr/bin/env python

from __future__ import annotations

import argparse
import csv
import json
import struct
import time
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

import build_cell_planes as CP

ROOT = CP.ROOT
WMAPS = CP.INF / "cohort/data/wmaps"
SUB_PX = 16
KIND = "disc"


def load_points(slide):
    xy = pd.read_parquet(CP.CELLS / f"{slide}.parquet",
                         columns=["cell_id", "x_px", "y_px"])
    d = CP.PRED / slide
    if not (d / f"{slide}_cells.csv").exists():
        alt = sorted(x for x in CP.PRED.iterdir()
                     if x.is_dir() and x.name.startswith(slide)
                     and (x / f"{slide}_cells.csv").exists())
        if alt:
            d = alt[0]
    cl = pd.read_csv(d / f"{slide}_cells.csv",
                     usecols=["cell_id", "cell_type"])
    z = np.load(WMAPS / f"{slide}.npz")
    wm = pd.DataFrame({"cell_id": z["cell_id"].astype(str),
                       "area_px": z["area_px"].astype(np.float64)})
    d = xy.merge(cl, on="cell_id", how="inner").merge(wm, on="cell_id", how="inner")
    return d, len(xy), len(cl), len(wm)


def _to_canvas(M, xy_px0, wh0, wh8, c0, r0):
    import numpy as _np
    import build_cell_planes as _CP
    xy = _np.asarray(xy_px0, float)
    x8 = _CP.rescale(xy[:, 0], wh0[0], wh8[0])
    y8 = _CP.rescale(xy[:, 1], wh0[1], wh8[1])
    X = M[0, 0] * x8 + M[0, 1] * y8 + M[0, 2] - c0
    Y = M[1, 0] * x8 + M[1, 1] * y8 + M[1, 2] - r0
    return _np.stack([X, Y], 1)


PLACER = None


def canvas_placer(vol, meta, extra_argv=None):
    import sys as _s
    argv = _s.argv
    _s.argv = ["build_cell_points.py", "--mpp", "8"] + list(extra_argv or [])
    try:
        main(probe_only=True)
    finally:
        _s.argv = argv
    if PLACER is None:
        raise SystemExit("placement setup did not run")
    return PLACER


def main(probe_only=False):
    ap = argparse.ArgumentParser()
    ap.add_argument("--mpp", type=float, default=8.0)
    ap.add_argument("--prep-root", default=str(CP.SEC))
    ap.add_argument("--manifest", default=str(CP.MANIFEST))
    ap.add_argument("--chain-manifest", default="", dest="chain_manifest",
                    help="section set the chain is solved over (a swap arm passes the main manifest; "
                         "masks and plane sizes still come from --manifest / --prep-root)")
    ap.add_argument("--edge-runs", default=",".join(CP.EDGE_RUNS))
    ap.add_argument("--suffix", default="")
    ap.add_argument("--out", default="", help="volume dir to write into; default reconstruction/volume_<mpp>um<suffix>")
    ap.add_argument("--probe-only", action="store_true",
                    help="stop after the canvas is set up; used by canvas_placer")


    ap.add_argument("--anchor", default="HT891Z1-U66",
                    help="must match the anchor the volume was built with")
    ap.add_argument("--sample", default="HT891Z1",
                    help="which sample's predictions to read")
    ap.add_argument("--probe", default="",
                    help="prep source_probe.json; the join to the classifier runs "
                         "on the source FILE NAME, never on the section id")
    ap.add_argument("--pred-dir", default="", dest="pred_dir")
    ap.add_argument("--cells-dir", default="", dest="cells_dir")
    ap.add_argument("--wmaps-dir", default="", dest="wmaps_dir")
    ap.add_argument("--runman", default="")
    a = ap.parse_args()
    out_mpp = float(a.mpp)
    CP.SEC = Path(a.prep_root)
    SEC = CP.SEC
    MANIFEST = Path(a.manifest)
    EDGE_RUNS = tuple(s.strip() for s in a.edge_runs.split(",") if s.strip())
    if a.sample != "HT891Z1":
        CP.PRED = CP.INF / "cohort" / a.sample
    if getattr(a, "pred_dir", ""):
        CP.PRED = Path(a.pred_dir)
    if getattr(a, "cells_dir", ""):
        CP.CELLS = Path(a.cells_dir)
    if getattr(a, "runman", ""):
        CP.RUNMAN = Path(a.runman)
    if getattr(a, "wmaps_dir", ""):
        global WMAPS
        WMAPS = Path(a.wmaps_dir)
    if a.probe:
        CP.PROBE = Path(a.probe)
    out = Path(a.out) if a.out else ROOT / f"reconstruction/volume_{out_mpp:g}um{a.suffix}"
    gmeta = json.loads((out / "metadata.json").read_text())
    t0 = time.time()
    print(f"prep={SEC}\nedges={EDGE_RUNS}\nout={out}", flush=True)

    man = {r["section_id"]: r for r in csv.DictReader(open(MANIFEST), delimiter="\t")}


    cman = {r["section_id"]: r for r in csv.DictReader(open(a.chain_manifest or MANIFEST), delimiter="\t")}
    sids = sorted(cman, key=lambda s: float(cman[s]["z_position_um"]))
    psids = sorted(man, key=lambda s: float(man[s]["z_position_um"]))
    missing = [s for s in psids if s not in cman]
    if missing: raise SystemExit(f"{len(missing)} sections of {MANIFEST} are not in the chain manifest: {missing[:3]}")
    z_um = np.array([float(cman[s]["z_position_um"]) for s in sids])

    m8, shapes = {}, {}
    for s in psids:
        m = cv2.imread(str(SEC / s / f"mask_mpp{CP.SRC_MPP:g}.png"), cv2.IMREAD_GRAYSCALE)
        shapes[s] = m.shape
        w, h = CP.plane_size(m.shape, out_mpp)
        m8[s] = cv2.resize((m > 127).astype(np.uint8), (w, h),
                           interpolation=cv2.INTER_AREA) > 0
    print(f"{len(psids)} masks read ({time.time()-t0:.0f}s)", flush=True)

    edges = CP.CHAIN.harvest(EDGE_RUNS)
    tree = CP.CHAIN.prereg_tree(edges, sids)
    chains = {k: CP.CHAIN.chain_on(tree, edges, b, a.anchor) for k, b in CP.BASES.items()}

    sizes = {s: CP.plane_size(shapes[s], out_mpp) for s in psids}
    pts = []
    for ch in chains.values():
        for s in psids:
            w, h = sizes[s]
            c = np.array([[0, 0], [w, 0], [w, h], [0, h]], float) * (out_mpp / CP.WG_MPP)
            pts.append((c @ ch[s][:2, :2].T + ch[s][:, 2]) * CP.WG_MPP)
    pts = np.vstack(pts)
    o = pts.min(0) - 40.0
    W = int(np.ceil((pts[:, 0].max() + 40.0 - o[0]) / out_mpp))
    H = int(np.ceil((pts[:, 1].max() + 40.0 - o[1]) / out_mpp))

    def M_for(bk, s):
        M = np.zeros((2, 3))
        M[:2, :2] = chains[bk][s][:2, :2]
        M[:, 2] = (CP.WG_MPP / out_mpp) * chains[bk][s][:, 2] - o / out_mpp
        return M

    anym = np.zeros((H, W), bool)
    for bk in chains:
        for s in psids:
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
    if CW * SUB_PX > 65535 or CH * SUB_PX > 65535:
        raise SystemExit(f"canvas {CW}x{CH} at 1/{SUB_PX} px overflows uint16")


    _join0 = CP.slide_for_section()
    global PLACER
    PLACER = {"he_level0_wh": {s: tuple(_join0[s]["level0"])
                               for s in sids if s in _join0},
              "W": CW, "H": CH, "sub": SUB_PX, "r_default": 12,
              "bases": list(chains), "sids": list(sids),
              "index_of": lambda s: sids.index(s),
              "plane_wh8": {s: tuple(sizes[s]) for s in psids},
              "to_canvas": (lambda bk, s, xy_px0, wh0: None if s not in chains[bk] or s not in sizes
                            else _to_canvas(M_for(bk, s), xy_px0, wh0,
                                            sizes[s], c0, r0))}

    if probe_only:
        return
    join = CP.slide_for_section()
    have = [i for i, s in enumerate(sids) if s in join]
    CLASSES = list(CP.CLASS_COLOUR)
    cidx = {c: i for i, c in enumerate(CLASSES)}
    rm = {r["slide"]: r for r in csv.DictReader(open(CP.RUNMAN), delimiter="\t")}

    files, rows = [], []
    for bk in CP.BASES:
        (out / bk / "cell_points").mkdir(parents=True, exist_ok=True)
    for i in have:
        s = sids[i]
        j = join[s]
        d, n_xy, n_cl, n_wm = load_points(j["slide"])
        unk = sorted(set(d["cell_type"]) - set(CLASSES))
        if unk:
            raise SystemExit(f"{s}: unknown classes {unk}")
        x0 = d["x_px"].to_numpy(np.float64)
        y0 = d["y_px"].to_numpy(np.float64)
        w8, h8 = sizes[s]
        native_mpp = float(rm[j["slide"]]["native_mpp"])
        if j["route"] == "he":
            W0, H0 = j["level0"]
            x8 = CP.rescale(x0, W0, w8)
            y8 = CP.rescale(y0, H0, h8)


            sx, sy = w8 / W0, h8 / H0
        else:
            A = np.array(j["matrix_plane_to_crop_px"], float)
            inv = np.linalg.inv(A[:, :2])
            qx, qy = x0 - A[0, 2], y0 - A[1, 2]
            x4 = inv[0, 0] * qx + inv[0, 1] * qy
            y4 = inv[1, 0] * qx + inv[1, 1] * qy
            w4, h4 = j["plane_grid_px"]
            x8 = CP.rescale(x4, w4, w8)
            y8 = CP.rescale(y4, h4, h8)
            sx = abs(inv[0, 0]) * (w8 / w4)
            sy = abs(inv[1, 1]) * (h8 / h4)
        cls = np.array([cidx[c] for c in d["cell_type"]], np.uint8)


        r_src = np.sqrt(np.maximum(d["area_px"].to_numpy(np.float64), 0.0) / np.pi)
        for bk in CP.BASES:
            M = M_for(bk, s)
            X = M[0, 0] * x8 + M[0, 1] * y8 + M[0, 2] - c0
            Y = M[1, 0] * x8 + M[1, 1] * y8 + M[1, 2] - r0
            lin = np.sqrt(abs(M[0, 0] * M[1, 1] - M[0, 1] * M[1, 0]))
            r_canvas = r_src * np.sqrt(abs(sx * sy)) * lin
            xi = np.clip(np.round(X * SUB_PX), 0, CW * SUB_PX - 1).astype(np.uint16)
            yi = np.clip(np.round(Y * SUB_PX), 0, CH * SUB_PX - 1).astype(np.uint16)
            ri = np.clip(np.round(r_canvas * SUB_PX), 1, 255).astype(np.uint8)
            n = int(len(xi))
            head = struct.pack("<8sIHHHHI", b"CELLPT01", n, CW, CH, SUB_PX,
                               len(CLASSES), 0)
            head = head + b"\0" * (32 - len(head))
            xy = np.empty(2 * n, np.uint16); xy[0::2] = xi; xy[1::2] = yi
            blob = head + xy.tobytes() + ri.tobytes() + cls.tobytes()
            fn = f"z{i:02d}_{s.split('-')[1]}.cells.bin"
            fp = out / bk / "cell_points" / fn
            fp.write_bytes(blob)
            files.append({"index": i, "basis": bk, "section_id": s,
                          "file": f"{bk}/cell_points/{fn}", "n_cells": n,
                          "bytes": int(fp.stat().st_size)})
        rows.append({"index": i, "section_id": s, "slide": j["slide"],
                     "route": j["route"], "n_cells": int(len(cls)),
                     "n_detected": int(n_xy), "n_classified": int(n_cl),
                     "n_wmap": int(n_wm),
                     "median_radius_um": round(float(np.median(r_canvas) * out_mpp), 3)})
        print(f"  {s:<14} {j['slide']:<22} {len(cls):7d} cells  "
              f"median r {np.median(r_canvas)*out_mpp:.2f} um", flush=True)


    from PIL import Image
    checks = []
    for i in have:
        for bk in CP.BASES:
            f = out / gmeta["encodings"][bk]["planes"][i]["webp_lossless"]
            al = np.array(Image.open(f).convert("RGBA"))[..., 3] > 0
            rec = next(q for q in files if q["index"] == i and q["basis"] == bk)
            raw = (out / rec["file"]).read_bytes()
            n = struct.unpack_from("<I", raw, 8)[0]
            xy = np.frombuffer(raw, np.uint16, 2 * n, 32)
            X = xy[0::2].astype(np.float64) / SUB_PX
            Y = xy[1::2].astype(np.float64) / SUB_PX

            def hit(dx, dy):
                xi = np.round(X + dx).astype(np.int64)
                yi = np.round(Y + dy).astype(np.int64)
                k = (xi >= 0) & (xi < CW) & (yi >= 0) & (yi < CH)
                return float(al[yi[k], xi[k]].sum()) / max(1, len(xi))
            checks.append({"index": i, "basis": bk, "section_id": sids[i],
                           "n_cells": int(n),
                           "in_mask": round(hit(0, 0), 4),
                           "control_shift_200px": round(hit(200, 200), 4)})
    worst = sorted(checks, key=lambda c: c["in_mask"])[:6]
    print("\nworst in_mask (decoded points):")
    for c in worst:
        print(f"  {c['section_id']:<14} {c['basis']:<16} in_mask {c['in_mask']:.4f} "
              f"shift {c['control_shift_200px']:.4f}")

    meta = {
        "what": "one point per predicted cell, on the volume's canvas",
        "claim": json.loads((out / "cells_metadata.json").read_text())["claim"],
        "kind": KIND,
        "kind_note": "the shape the page draws for a cell. 'disc' carries a radius "
                     "only; a future 'polygon' value carries the nucleus outline and "
                     "gets its own renderer, so outlines are a new value of this "
                     "field rather than a change to the draw path. InstanSeg's "
                     "polygons were not kept by the detection step (wmaps.npz has "
                     "has_poly and a 16x16 weight map, not the polygon).",
        "format": {
            "header_bytes": 32,
            "header": "<8s magic 'CELLPT01'><u4 n><u2 canvas_w><u2 canvas_h>"
                      "<u2 sub_px><u2 n_classes><u4 reserved>, zero padded to 32",
            "body": "u2 xy[2n] interleaved, then u1 r[n], then u1 class[n]",
            "units": f"positions and radii are in 1/{SUB_PX} of a canvas pixel, i.e. "
                     f"{out_mpp/SUB_PX:g} um, which is the pitch the centroids were "
                     f"measured at",
            "bytes_per_cell": 6,
        },
        "in_plane_um_per_px": out_mpp,
        "canvas_px": {"width": int(CW), "height": int(CH)},
        "classes": CLASSES,
        "geometry": {
            "source": "recomputed by the same code path as build_cell_planes.py and "
                      "build_he_rgb.py, then asserted against the grey volume's "
                      "canvas_px",
            "edge_runs": list(EDGE_RUNS), "anchor": a.anchor,
            "radius": "sqrt(area_px/pi) from data/wmaps, carried through the source "
                      "grid's resize and the section's own chain scale",
        },
        "n_planes": len(have),
        "bytes_total": int(sum(q["bytes"] for q in files)),
        "sections": rows,
        "files": files,
        "checks": checks,
        "worst_in_mask": worst,
        "elapsed_s": round(time.time() - t0, 1),
        "read_only_upstream": True,
    }
    (out / "cell_points_metadata.json").write_text(json.dumps(meta, indent=1))
    print(f"\nwrote {out/'cell_points_metadata.json'}  {len(files)} files, "
          f"{meta['bytes_total']/1e6:.1f} MB  ({meta['elapsed_s']} s)")


if __name__ == "__main__":
    main()
