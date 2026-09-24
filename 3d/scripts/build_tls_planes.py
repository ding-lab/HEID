#!/usr/bin/env python3
import argparse
import json
import os
import re
import struct

import numpy as np
import pandas as pd
from PIL import Image
from scipy.ndimage import binary_closing, binary_dilation, binary_fill_holes, label

UM = 8.0
COL3D, COL2D = "#1f5fd6", "#2ca02c"


def read_cells_bin(fp):
    raw = open(fp, "rb").read()
    magic, n, W, H, sub, ncls, _ = struct.unpack("<8sIHHHHI", raw[:24])
    assert magic == b"CELLPT01", fp
    ip = np.frombuffer(raw, np.uint16, 2 * n, 32)
    cls = np.frombuffer(raw, np.uint8, n, 32 + 5 * n)
    return ip[0::2] / sub, ip[1::2] / sub, cls, W, H


def ring(mask, fill_alpha=60, rim_px=2):
    fill = mask
    er = binary_dilation(~fill, iterations=rim_px)
    rim = fill & er
    a8 = np.zeros(mask.shape, np.uint8)
    a8[fill] = fill_alpha
    a8[rim] = 255
    return np.dstack([np.full_like(a8, 255)] * 3 + [a8])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--volume", required=True)
    ap.add_argument("--tls-root", required=True, dest="tls_root")
    ap.add_argument("--pred", required=True)
    a = ap.parse_args()
    vol = a.volume
    meta = json.load(open(os.path.join(vol, "cell_points_metadata.json")))
    vmeta = json.load(open(os.path.join(vol, "metadata.json")))
    classes = meta["classes"]
    slide_of = {s["section_id"]: s["slide"] for s in meta["sections"]}
    bases = sorted({f["basis"] for f in meta["files"]})
    planes = {}
    for f in meta["files"]:
        b, sid, idx = f["basis"], f["section_id"], f["index"]
        slide = slide_of.get(sid)
        cp = os.path.join(a.tls_root, slide or "-", f"{slide}_tls_cells.csv")
        if not os.path.exists(cp):


            m = re.search(r"U(\d+)[^0-9]*$", sid)
            cand = [d for d in os.listdir(a.tls_root)
                    if m and re.search(rf"U{m.group(1)}[^0-9]*$", d)
                    and os.path.exists(os.path.join(a.tls_root, d, f"{d}_tls_cells.csv"))] if m else []
            if len(cand) == 1:
                slide = cand[0]
                cp = os.path.join(a.tls_root, slide, f"{slide}_tls_cells.csv")
                print(f"  {sid}: detection found as {slide}", flush=True)
        if not slide or not os.path.exists(cp):
            continue
        x, y, cl, W, H = read_cells_bin(os.path.join(vol, f["file"]))
        pc = pd.read_csv(cp, usecols=["cell_type", "tls_region_halo50", "in_tls_core"])
        if len(pc) != len(x):
            print(f"  {sid} {b}: {len(pc)} cells in the detection vs {len(x)} points, skipped", flush=True)
            continue
        idx_c = pc["cell_type"].map({c: i for i, c in enumerate(classes)}).to_numpy()
        if not np.array_equal(idx_c, cl):
            print(f"  {sid} {b}: class sequence differs, skipped", flush=True)
            continue
        core = pc["in_tls_core"].astype(str).isin(["True", "true", "1"]).to_numpy()
        own = pc["tls_region_halo50"].astype(str).to_numpy()


        loc = (pc["tls_local_halo50"].astype(str).to_numpy() if "tls_local_halo50" in pc
               else np.full(len(pc), "", dtype=object))
        regs, imgs = [], {"3d": None, "2d": None}
        for tid in sorted(set(own[core])):
            if not (tid.startswith("3d-tls") or tid.startswith("2d-tls")):
                continue
            kind = "3d" if tid.startswith("3d") else "2d"
            sel_id = core & (own == tid)
            m_all = np.zeros((H, W), bool)
            for lobe in sorted(set(loc[sel_id])):
                sel = sel_id & (loc == lobe)
                m = np.zeros((H, W), bool)
                xi = np.clip(np.round(x[sel]).astype(int), 0, W - 1)
                yi = np.clip(np.round(y[sel]).astype(int), 0, H - 1)
                m[yi, xi] = True
                m = binary_fill_holes(binary_closing(binary_dilation(m, iterations=2), iterations=2))
                lb, n = label(m)
                if n > 1:
                    sizes = np.bincount(lb.ravel()); sizes[0] = 0
                    m = lb == int(sizes.argmax())
                r = ring(m)
                imgs[kind] = r if imgs[kind] is None else np.maximum(imgs[kind], r)
                m_all |= m
            m = m_all
            ys, xs = np.nonzero(m)
            regs.append({"id": tid, "kind": kind,
                         "cx_um": round(float(xs.mean()) * UM, 1), "cy_um": round(float(ys.mean()) * UM, 1),
                         "bbox_um": [int(xs.min()) * UM, int(ys.min()) * UM,
                                     (int(xs.max()) + 1) * UM, (int(ys.max()) + 1) * UM]})
        if not regs:
            continue
        d = os.path.join(vol, b, "tls")
        os.makedirs(d, exist_ok=True)
        u = sid.split("-")[-1]
        files = {}
        for kind in ("3d", "2d"):
            fn = f"z{idx:02d}_{u}.tls{kind}.webp"
            img = imgs[kind] if imgs[kind] is not None else ring(np.zeros((H, W), bool))
            Image.fromarray(img).save(os.path.join(d, fn), format="WEBP", lossless=True, quality=100, method=4)
            files[kind] = fn
        rec = planes.setdefault(idx, {"index": idx, "section_id": sid, "z_um": vmeta["z_um"][idx],
                                      "file3d": files["3d"], "file2d": files["2d"], "regions": regs,
                                      "n_3d": sum(r["kind"] == "3d" for r in regs),
                                      "n_2d": sum(r["kind"] == "2d" for r in regs)})
        print(f"  {sid} {b}: {rec['n_3d']} 3-D + {rec['n_2d']} 2-D TLS regions", flush=True)
    out = {"what": "TLS regions per section (core cells placed on the canvas), 3-D TLS and 2-D-only TLS as two ring layers",
           "colour3d": COL3D, "colour2d": COL2D, "bases": bases,
           "n_planes": len(planes), "planes": sorted(planes.values(), key=lambda r: r["index"])}
    json.dump(out, open(os.path.join(vol, "tls_regions_metadata.json"), "w"), indent=1)
    print(f"wrote {len(planes)} planes with TLS regions -> {vol}/tls_regions_metadata.json")


if __name__ == "__main__":
    main()
