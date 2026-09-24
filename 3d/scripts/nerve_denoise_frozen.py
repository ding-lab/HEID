#!/usr/bin/env python3
import argparse
import glob
import json
import os
import pickle
import struct
from pathlib import Path

import numpy as np
from scipy.ndimage import binary_dilation

RES = 10.0
TOL_UM = 60.0
OVERLAP_MIN = 0.3
BRANCH_MIN = 0.6
MIN_SECTIONS = 3


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--volume", required=True)
    a = ap.parse_args()
    VOL = Path(a.volume)
    OUT = Path(os.environ.get("HTAN3D_S12", str(Path(os.environ.get("HTAN3D_OBJ3D", VOL.parent / "objects_3d")) / "S12_cloud")))
    OUT.mkdir(parents=True, exist_ok=True)
    caches = sorted(glob.glob(str(VOL / "nerve_operator_cache" / "stage2_*.pkl")), key=os.path.getmtime)
    if not caches:
        raise SystemExit("no stage cache: run build_nerve_operator3d.py once first")
    st = pickle.load(open(caches[-1], "rb"))
    tau, data, cg = st["tau"], st["data"], st["cg"]
    order = sorted(data, key=lambda s: data[s][0])
    ny, nx = next(iter(cg.values()))[0].shape
    it = max(1, int(round(TOL_UM / RES)))

    node, bbs, area = {}, {}, {}
    for s in order:
        g2, ids = cg[s]
        for i in ids:
            ys, xs = np.nonzero(g2 == i)
            bbs[(s, i)] = (int(ys.min()), int(ys.max()) + 1, int(xs.min()), int(xs.max()) + 1)
            area[(s, i)] = int(len(ys))
            node[(s, i)] = (s, i)

    def find(x):
        while node[x] != x:
            node[x] = node[node[x]]; x = node[x]
        return x

    n_links = n_branch = 0
    for s0, s1 in zip(order, order[1:]):
        g0, ids0 = cg[s0]; g1, ids1 = cg[s1]
        cands = []
        for i in ids0:
            y0, y1, x0, x1 = bbs[(s0, i)]
            y0, y1 = max(0, y0 - it), min(ny, y1 + it); x0, x1 = max(0, x0 - it), min(nx, x1 + it)
            d0 = binary_dilation(g0[y0:y1, x0:x1] == i, iterations=it)
            w1 = g1[y0:y1, x0:x1]
            for j in ids1:
                b1 = bbs[(s1, j)]
                if b1[0] >= y1 or b1[1] <= y0 or b1[2] >= x1 or b1[3] <= x0:
                    continue
                inter = int((d0 & (w1 == j)).sum())
                if inter:
                    fr = inter / max(1, min(area[(s0, i)], area[(s1, j)]))
                    if fr >= OVERLAP_MIN:
                        cands.append((fr, i, j))
        cands.sort(key=lambda c: -c[0])
        up, down = set(), set()
        for fr, i, j in cands:
            taken = i in up or j in down
            if taken and fr < BRANCH_MIN:
                continue
            up.add(i); down.add(j)
            node[find((s0, i))] = find((s1, j)); n_links += 1; n_branch += int(taken)
    span = {}
    for k in node:
        span.setdefault(find(k), set()).add(k[0])
    keep = {k: len(span[find(k)]) >= MIN_SECTIONS for k in node}

    pts, per_sec = [], []
    for k, s in enumerate(order):
        z, xy, pr = data[s]
        g2, ids = cg[s]
        lut = np.zeros(max(ids, default=0) + 1, bool)
        for i in ids:
            lut[i] = keep[(s, i)]
        xi = np.clip((xy[:, 0] / RES).astype(int), 0, nx - 1)
        yi = np.clip((xy[:, 1] / RES).astype(int), 0, ny - 1)
        sel = (pr >= tau) & lut[g2[yi, xi]]
        n_tau = int((pr >= tau).sum())
        pts.append(np.column_stack([xy[sel, 0], xy[sel, 1], np.full(int(sel.sum()), k)]))
        per_sec.append({"section_id": s, "z_um": z, "n_tau": n_tau, "n_regions": len(ids),
                        "n_regions_kept": int(sum(keep[(s, i)] for i in ids)), "n_denoised": int(sel.sum())})
        print(f"  {s}: tau {n_tau} regions {len(ids)} kept {per_sec[-1]['n_regions_kept']} -> cells {int(sel.sum())}", flush=True)
    P = np.vstack(pts)
    assert P[:, :2].max() < 65535
    b = bytearray(8); b[7] = 1
    b += struct.pack("<I", len(P))
    arr = np.empty((len(P), 3), np.uint16)
    arr[:, 0] = np.round(P[:, 0]); arr[:, 1] = np.round(P[:, 1]); arr[:, 2] = P[:, 2]
    b += arr.tobytes()
    while len(b) % 4:
        b += b"\0"
    b += np.asarray([data[s][0] for s in order], np.float32).tobytes()
    (OUT / "nerve_cloud.bin").write_bytes(bytes(b))
    (OUT / "nerve_cloud.json").write_text(json.dumps({
        "what": "FROZEN Schwann denoise: tau cells inside Nerve operator regions whose stack spans >= %d consecutive measured sections" % MIN_SECTIONS,
        "classes": ["Schwann"], "n": int(len(P)), "n_planes": len(order), "um_px": 1.0, "tau": tau}, indent=1))
    (OUT / "denoise_frozen.json").write_text(json.dumps({
        "definition": __doc__.strip().split("\n\n")[1],
        "constants": {"RES": RES, "TOL_UM": TOL_UM, "OVERLAP_MIN": OVERLAP_MIN, "BRANCH_MIN": BRANCH_MIN,
                      "MIN_SECTIONS": MIN_SECTIONS, "tau": tau, "cache": os.path.basename(caches[-1])},
        "n_links": n_links, "n_branches": n_branch, "n_regions": len(keep), "n_regions_kept": int(sum(keep.values())),
        "n_cells": int(len(P)), "sections": per_sec}, indent=1))
    print(f"wrote {OUT / 'nerve_cloud.bin'}: {len(P):,} denoised Schwann cells; "
          f"{int(sum(keep.values())):,}/{len(keep):,} regions kept ({n_links} links, {n_branch} branches)", flush=True)


if __name__ == "__main__":
    main()
