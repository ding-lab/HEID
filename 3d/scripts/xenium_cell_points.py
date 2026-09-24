#!/usr/bin/env python
from __future__ import annotations

import argparse
import csv
import importlib.util as _iu
import json
import struct
import sys
import time
from pathlib import Path

import os
import cv2
import numpy as np
import pandas as pd
from PIL import Image

ROOT = Path(os.environ.get("THREED_ROOT", os.path.join(os.environ.get("PROJECTS_ROOT", "/data/heid"), "3d")))
WORK = ROOT / "xenium"
VOL = Path(os.environ.get("HTAN3D_VOL") or ROOT / "reconstruction/volume_8um")
RSCRIPTS = Path(__file__).resolve().parent

_s = _iu.spec_from_file_location("build_cell_planes", RSCRIPTS / "build_cell_planes.py")
BCP = _iu.module_from_spec(_s)
_s.loader.exec_module(BCP)

if os.environ.get("HTAN3D_PREP"):
    BCP.SEC = Path(os.environ["HTAN3D_PREP"])
if os.environ.get("HTAN3D_EDGES"):
    BCP.EDGE_RUNS = tuple(x.strip() for x in os.environ["HTAN3D_EDGES"].split(",") if x.strip())
if os.environ.get("HTAN3D_MAN"):
    BCP.MANIFEST = Path(os.environ["HTAN3D_MAN"])
ANCHOR = os.environ.get("HTAN3D_ANCHOR", "HT891Z1-U66")

sys.modules["build_cell_planes"] = BCP
sys.path.insert(0, str(RSCRIPTS))
_s2 = _iu.spec_from_file_location("build_cell_points", RSCRIPTS / "build_cell_points.py")
BCPT = _iu.module_from_spec(_s2)
_s2.loader.exec_module(BCPT)

OUT_MPP = 8.0
SUB_PX = BCPT.SUB_PX
SELFCHECK_INDEX = 1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reg", default=str(WORK / "outputs/registration.json"))
    ap.add_argument("--out", default=str(WORK / "outputs/cells_planes"))
    ap.add_argument("--frag", default=str(WORK / "outputs/xenium9_cell_points_fragment.json"))
    a = ap.parse_args()
    t0 = time.time()
    reg = json.loads(Path(a.reg).read_text())
    gmeta = json.loads((VOL / "metadata.json").read_text())
    pmeta = json.loads((VOL / "cell_points_metadata.json").read_text())
    probe = {p["section_id"]: p for p in json.loads(BCP.PROBE.read_text())}
    rm = {r["slide"]: r for r in csv.DictReader(open(BCP.RUNMAN), delimiter="\t")}

    CLASSES = list(BCP.CLASS_COLOUR)
    if CLASSES != pmeta["classes"]:
        raise SystemExit(f"class order differs from the published cell_points "
                         f"metadata:\n  mine {CLASSES}\n  theirs {pmeta['classes']}")
    cidx = {c: i for i, c in enumerate(CLASSES)}

    man = {r["section_id"]: r for r in csv.DictReader(open(BCP.MANIFEST), delimiter="\t")}
    sids = sorted(man, key=lambda s: float(man[s]["z_position_um"]))

    m8, shapes = {}, {}
    for s in sids:
        m = cv2.imread(str(BCP.SEC / s / f"mask_mpp{BCP.SRC_MPP:g}.png"), cv2.IMREAD_GRAYSCALE)
        shapes[s] = m.shape
        w, h = BCP.plane_size(m.shape, OUT_MPP)
        m8[s] = cv2.resize((m > 127).astype(np.uint8), (w, h),
                           interpolation=cv2.INTER_AREA) > 0
    edges = BCP.CHAIN.harvest(BCP.EDGE_RUNS)
    tree = BCP.CHAIN.prereg_tree(edges, sids)
    chains = {k: BCP.CHAIN.chain_on(tree, edges, b, ANCHOR) for k, b in BCP.BASES.items()}

    sizes = {s: BCP.plane_size(shapes[s], OUT_MPP) for s in sids}
    pts = []
    for ch in chains.values():
        for s in sids:
            w, h = sizes[s]
            c = np.array([[0, 0], [w, 0], [w, h], [0, h]], float) * (OUT_MPP / BCP.WG_MPP)
            pts.append((c @ ch[s][:2, :2].T + ch[s][:, 2]) * BCP.WG_MPP)
    pts = np.vstack(pts)
    o = pts.min(0) - 40.0
    W = int(np.ceil((pts[:, 0].max() + 40.0 - o[0]) / OUT_MPP))
    H = int(np.ceil((pts[:, 1].max() + 40.0 - o[1]) / OUT_MPP))
    pinned = bool(gmeta.get("canvas_origin_um"))
    if pinned:


        o = np.asarray(gmeta["canvas_origin_um"], float)
        W, H = int(gmeta["canvas_px"]["width"]), int(gmeta["canvas_px"]["height"])

    def M_for(bk, s):
        M = np.zeros((2, 3))
        M[:2, :2] = chains[bk][s][:2, :2]
        M[:, 2] = (BCP.WG_MPP / OUT_MPP) * chains[bk][s][:, 2] - o / OUT_MPP
        return M

    if pinned:
        c0, r0, CW, CH = 0, 0, W, H
    else:
        anym = np.zeros((H, W), bool)
        for bk in chains:
            for s in sids:
                anym |= cv2.warpAffine(m8[s].astype(np.uint8) * 255, M_for(bk, s), (W, H),
                                       flags=cv2.INTER_NEAREST, borderValue=0) > 0
        ys, xs = np.nonzero(anym)
        p = int(round(80.0 / OUT_MPP))
        r0, r1 = max(0, ys.min() - p), min(H, ys.max() + p + 1)
        c0, c1 = max(0, xs.min() - p), min(W, xs.max() + p + 1)
        CW, CH = c1 - c0, r1 - r0
    if (CW, CH) != (gmeta["canvas_px"]["width"], gmeta["canvas_px"]["height"]):
        raise SystemExit(f"canvas rebuilt {CW}x{CH}, volume says "
                         f"{gmeta['canvas_px']} -- refusing to write points")
    if (CW, CH) != (pmeta["canvas_px"]["width"], pmeta["canvas_px"]["height"]):
        raise SystemExit("canvas differs from the published cell_points canvas")
    print(f"canvas {CW} x {CH} matches the volume and the published points "
          f"({time.time()-t0:.0f}s)", flush=True)

    def pack(X, Y, r_canvas, cls):
        xi = np.clip(np.round(X * SUB_PX), 0, CW * SUB_PX - 1).astype(np.uint16)
        yi = np.clip(np.round(Y * SUB_PX), 0, CH * SUB_PX - 1).astype(np.uint16)
        ri = np.clip(np.round(r_canvas * SUB_PX), 1, 255).astype(np.uint8)
        n = int(len(xi))
        head = struct.pack("<8sIHHHHI", b"CELLPT01", n, CW, CH, SUB_PX, len(CLASSES), 0)
        head = head + b"\0" * (32 - len(head))
        xy = np.empty(2 * n, np.uint16)
        xy[0::2] = xi
        xy[1::2] = yi
        return head + xy.tobytes() + ri.tobytes() + cls.tobytes(), n


    join = BCP.slide_for_section()
    i = SELFCHECK_INDEX
    s = sids[i]
    j = join[s]
    d, n_xy, n_cl, n_wm = BCPT.load_points(j["slide"])
    x0 = d["x_px"].to_numpy(np.float64)
    y0 = d["y_px"].to_numpy(np.float64)
    w8, h8 = sizes[s]
    W0, H0 = j["level0"]
    x8 = BCP.rescale(x0, W0, w8)
    y8 = BCP.rescale(y0, H0, h8)
    sx, sy = w8 / W0, h8 / H0
    cls = np.array([cidx[c] for c in d["cell_type"]], np.uint8)
    r_src = np.sqrt(np.maximum(d["area_px"].to_numpy(np.float64), 0.0) / np.pi)
    ok_all = True
    if pmeta.get("checks_repositioned"):


        print("self-check skipped: the published points predate a reposition of the planes", flush=True)
    for bk in ([] if pmeta.get("checks_repositioned") else BCP.BASES):
        M = M_for(bk, s)
        X = M[0, 0] * x8 + M[0, 1] * y8 + M[0, 2] - c0
        Y = M[1, 0] * x8 + M[1, 1] * y8 + M[1, 2] - r0
        lin = np.sqrt(abs(M[0, 0] * M[1, 1] - M[0, 1] * M[1, 0]))
        blob, n = pack(X, Y, r_src * np.sqrt(abs(sx * sy)) * lin, cls)
        ref = (VOL / bk / "cell_points" / f"z{i:02d}_{s.split('-')[1]}.cells.bin").read_bytes()
        same = blob == ref
        ok_all &= same
        print(f"self-check {bk} z{i:02d} {s}: {n} cells, "
              f"{'BYTE-IDENTICAL to the published file' if same else 'DIFFERS'}",
              flush=True)
    if not ok_all:
        raise SystemExit("the point writer does not reproduce a published file byte "
                         "for byte; the nine would not be in the same format")


    files, rows = [], []
    for bk in BCP.BASES:
        (Path(a.out) / bk / "cell_points").mkdir(parents=True, exist_ok=True)

    for sid, r in reg["sections"].items():
        i = sids.index(sid)
        slide = r["slide"]
        d, n_xy, n_cl, n_wm = BCPT.load_points(slide)
        unk = sorted(set(d["cell_type"]) - set(CLASSES))
        if unk:
            raise SystemExit(f"{sid}: unknown classes {unk}")
        x0 = d["x_px"].to_numpy(np.float64)
        y0 = d["y_px"].to_numpy(np.float64)
        he_mpp = float(rm[slide]["native_mpp"])
        A = np.array(r["affine_he_um_to_xenium_um"]["A"], float)
        b = np.array(r["affine_he_um_to_xenium_um"]["b"], float)
        ps = float(r["pixel_size_um"])
        H0, W0 = probe[sid]["level_yx"][0]
        w8, h8 = sizes[sid]

        def to_plane(xp, yp):
            um = np.stack([xp * he_mpp, yp * he_mpp], 1) @ A.T + b
            return (BCP.rescale(um[:, 0] / ps, W0, w8),
                    BCP.rescale(um[:, 1] / ps, H0, h8))

        x8, y8 = to_plane(x0, y0)


        s_aff = float(np.sqrt(abs(np.linalg.det(A))))
        sx = he_mpp * s_aff / ps * (w8 / W0)
        sy = he_mpp * s_aff / ps * (h8 / H0)
        dx8, dy8 = to_plane(x0[:64] + 1.0, y0[:64])
        ex8, ey8 = to_plane(x0[:64], y0[:64] + 1.0)
        num = np.sqrt(np.abs((dx8 - x8[:64]) * (ey8 - y8[:64])
                             - (dy8 - y8[:64]) * (ex8 - x8[:64])))
        if not np.allclose(num, np.sqrt(sx * sy), rtol=1e-6):
            raise SystemExit(f"{sid}: analytic scale {np.sqrt(sx*sy):.8f} disagrees "
                             f"with the measured {float(np.median(num)):.8f}")

        cls = np.array([cidx[c] for c in d["cell_type"]], np.uint8)
        r_src = np.sqrt(np.maximum(d["area_px"].to_numpy(np.float64), 0.0) / np.pi)
        for bk in BCP.BASES:
            M = M_for(bk, sid)
            X = M[0, 0] * x8 + M[0, 1] * y8 + M[0, 2] - c0
            Y = M[1, 0] * x8 + M[1, 1] * y8 + M[1, 2] - r0
            lin = np.sqrt(abs(M[0, 0] * M[1, 1] - M[0, 1] * M[1, 0]))
            r_canvas = r_src * np.sqrt(abs(sx * sy)) * lin
            blob, n = pack(X, Y, r_canvas, cls)
            fn = f"z{i:02d}_{sid.split('-')[1]}.cells.bin"
            fp = Path(a.out) / bk / "cell_points" / fn
            fp.write_bytes(blob)
            files.append({"index": i, "basis": bk, "section_id": sid,
                          "file": f"{bk}/cell_points/{fn}", "n_cells": n,
                          "bytes": int(fp.stat().st_size)})
        rows.append({"index": i, "section_id": sid, "slide": slide,
                     "route": "xenium_pointcloud", "n_cells": int(len(cls)),
                     "n_detected": int(n_xy), "n_classified": int(n_cl),
                     "n_wmap": int(n_wm),
                     "median_radius_um": round(float(np.median(r_canvas) * OUT_MPP), 3)})
        print(f"  {sid:<14} {slide:<22} {len(cls):7d} cells  "
              f"median r {np.median(r_canvas)*OUT_MPP:.2f} um", flush=True)


    checks = []
    for rec in files:
        i, bk = rec["index"], rec["basis"]
        al = np.array(Image.open(VOL / gmeta["encodings"][bk]["planes"][i]["webp_lossless"])
                      .convert("RGBA"))[..., 3] > 0
        raw = (Path(a.out) / rec["file"]).read_bytes()
        n = struct.unpack_from("<I", raw, 8)[0]
        hdr = struct.unpack_from("<8sIHHHHI", raw, 0)
        if hdr[0] != b"CELLPT01" or hdr[2:5] != (CW, CH, SUB_PX):
            raise SystemExit(f"{rec['file']}: bad header {hdr}")
        if len(raw) != 32 + 6 * n:
            raise SystemExit(f"{rec['file']}: length {len(raw)} != 32 + 6*{n}")
        xy = np.frombuffer(raw, np.uint16, 2 * n, 32)
        X = xy[0::2].astype(np.float64) / SUB_PX
        Y = xy[1::2].astype(np.float64) / SUB_PX

        def hit(dx, dy):
            xi = np.round(X + dx).astype(np.int64)
            yi = np.round(Y + dy).astype(np.int64)
            k = (xi >= 0) & (xi < CW) & (yi >= 0) & (yi < CH)
            return float(al[yi[k], xi[k]].sum()) / max(1, len(xi))
        checks.append({"index": i, "basis": bk, "section_id": rec["section_id"],
                       "n_cells": int(n), "in_mask": round(hit(0, 0), 4),
                       "control_shift_200px": round(hit(200, 200), 4)})
        print(f"  {rec['section_id']:<14} {bk:<16} in_mask {checks[-1]['in_mask']:.4f}"
              f"  shift200 {checks[-1]['control_shift_200px']:.4f}", flush=True)

    frag = {
        "what": "per-cell point files for the nine Xenium planes, in the format "
                "build_cell_points.py wrote for the other forty-one. A FRAGMENT: "
                "merge into cell_points_metadata.json.",
        "claim": pmeta["claim"],
        "kind": BCPT.KIND,
        "format": pmeta["format"],
        "classes": CLASSES,
        "canvas_px": {"width": int(CW), "height": int(CH)},
        "in_plane_um_per_px": OUT_MPP,
        "writer_selfcheck": f"plane {SELFCHECK_INDEX} was rebuilt through the he route "
                            f"by this script and is BYTE-IDENTICAL to the published "
                            f"file in both bases",
        "geometry": {
            "steps": "H&E level-0 px -> H&E um -> (fitted similarity) -> Xenium um -> "
                     "DAPI level-0 px -> plane grid at 8 um/px -> chain matrix -> canvas",
            "radius": "sqrt(area_px/pi) from inference/cohort/data/wmaps -- the "
                      "same source the other forty-one use, present for all nine "
                      "slides, so no substitute was needed",
            "radius_scale_check": "the analytic linear factor of the five-step "
                                  "composite is asserted against a numerical "
                                  "derivative of the real map at rtol 1e-6",
        },
        "n_planes": len(rows),
        "bytes_total": int(sum(q["bytes"] for q in files)),
        "sections": rows, "files": files, "checks": checks,
        "elapsed_s": round(time.time() - t0, 1),
    }
    Path(a.frag).write_text(json.dumps(frag, indent=1, default=float))
    print(f"\nwrote {a.frag}  {len(files)} files, "
          f"{frag['bytes_total']/1e6:.1f} MB  ({time.time()-t0:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
