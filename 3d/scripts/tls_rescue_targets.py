#!/usr/bin/env python3
import argparse
import json
import os
import struct

import numpy as np
import pandas as pd

UM_PER_CANVAS_PX = 8.0
TOL_BASE_UM = 150.0
FP_RES_UM = 10.0
FP_DILATE_UM = 20.0


def canvas_map(vol, pred, slide, basis):
    meta = json.load(open(os.path.join(vol, "cell_points_metadata.json")))
    secs = [s for s in meta["sections"] if s["slide"] == slide]
    if not secs:
        return None
    sid = secs[0]["section_id"]
    f = [x for x in meta["files"] if x["basis"] == basis and x["section_id"] == sid]
    if not f:
        return None
    raw = open(os.path.join(vol, f[0]["file"]), "rb").read()
    magic, n, cw, ch, sub, ncls, _ = struct.unpack("<8sIHHHHI", raw[:24])
    xy = np.frombuffer(raw[32:32 + 4 * n], dtype="<u2").reshape(n, 2).astype(np.float64)
    cl = np.frombuffer(raw[32 + 5 * n:32 + 6 * n], dtype=np.uint8)
    pq = os.path.join(pred, slide + ".parquet")
    if not os.path.exists(pq):
        return None
    pr = pd.read_parquet(pq, columns=["pred_class_name", "x_um", "y_um"])
    if len(pr) != n:
        return None
    idx = pr["pred_class_name"].map({c: i for i, c in enumerate(meta["classes"])}).to_numpy()
    if not np.array_equal(idx, cl):
        return None
    return xy / sub * UM_PER_CANVAS_PX, pr[["x_um", "y_um"]].to_numpy()


def affine_native_to_canvas(m):
    cxy, nxy = m
    X = np.column_stack([nxy, np.ones(len(nxy))])
    A, *_ = np.linalg.lstsq(X, cxy, rcond=None)
    return A


def core_footprint_canvas(tls_root, slide, tls_id, m):
    f = os.path.join(tls_root, slide, f"{slide}_masks.npz")
    if not os.path.exists(f):
        return None
    z = np.load(f, allow_pickle=False)
    cores = z["cores"]
    k = int(str(tls_id).split("-")[-1]) - 1
    if k < 0 or k >= len(cores):
        return None
    ys, xs = np.nonzero(cores[k])
    if not len(ys):
        return None
    res = float(z["resolution"]) if "resolution" in z else FP_RES_UM
    nat = np.column_stack([xs * res + float(z["x_min"]), ys * res + float(z["y_min"])])
    A = affine_native_to_canvas(m)
    return np.column_stack([nat, np.ones(len(nat))]) @ A


def footprint_bitmap(pts_canvas, m_target):
    cxy, nxy = m_target
    X = np.column_stack([cxy, np.ones(len(cxy))])
    A, *_ = np.linalg.lstsq(X, nxy, rcond=None)
    nat = np.column_stack([pts_canvas, np.ones(len(pts_canvas))]) @ A
    x0, y0 = float(nat[:, 0].min()) - FP_DILATE_UM, float(nat[:, 1].min()) - FP_DILATE_UM
    x1, y1 = float(nat[:, 0].max()) + FP_DILATE_UM, float(nat[:, 1].max()) + FP_DILATE_UM
    nx = int(np.ceil((x1 - x0) / FP_RES_UM)) + 1
    ny = int(np.ceil((y1 - y0) / FP_RES_UM)) + 1
    if nx * ny > 4_000_000:
        return None
    g = np.zeros((ny, nx), bool)
    gx = np.clip(((nat[:, 0] - x0) / FP_RES_UM).astype(int), 0, nx - 1)
    gy = np.clip(((nat[:, 1] - y0) / FP_RES_UM).astype(int), 0, ny - 1)
    g[gy, gx] = True
    r = max(1, int(round(FP_DILATE_UM / FP_RES_UM)))
    from scipy.ndimage import binary_dilation, binary_fill_holes
    g = binary_fill_holes(binary_dilation(g, iterations=r))
    return {"fp_x0": round(x0, 1), "fp_y0": round(y0, 1), "fp_res": FP_RES_UM,
            "fp_rows": ["".join("1" if v else "0" for v in row) for row in g]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sample", required=True)
    ap.add_argument("--objects", required=True)
    ap.add_argument("--table", required=True)
    ap.add_argument("--volume", required=True)
    ap.add_argument("--pred", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--min-sections", type=int, default=3)
    ap.add_argument("--bridges", default="", help="linker bridge_candidates.tsv: held-back column "
                    "bridges get targets on the empty planes between their two parts")
    ap.add_argument("--tls-root", default="", dest="tls_root",
                    help="per-slide TLS outputs: with it every target also carries the FOOTPRINT of "
                         "the members above and below, in this slide's own micrometres, and a "
                         "candidate that overlaps it is rescued however far its centroid sits")
    a = ap.parse_args()

    objs = pd.read_csv(a.objects, sep="\t")
    tab = pd.read_csv(a.table, sep="\t", comment="#")
    tab = tab[(tab.sample_id == a.sample) & tab.cx_um_canvas.notna()]
    row_of = {(r.slide_id, r.tls_id): r for r in tab.itertuples()}
    vm = json.load(open(os.path.join(a.volume, "metadata.json")))
    cm = json.load(open(os.path.join(a.volume, "cell_points_metadata.json")))
    zs = vm["z_um"]
    slide_by_z = {zs[s["index"]]: s["slide"] for s in cm["sections"]}
    basis = tab.canvas_basis.dropna().iloc[0] if "canvas_basis" in tab and tab.canvas_basis.notna().any() else "G_withdrawn"

    targets, maps = {}, {}
    n_obj = 0
    spans = []
    for o in objs.itertuples():
        if int(o.n_measured_sections) < a.min_sections:
            continue
        mem = []
        for tok in str(o.member_tls).split(";"):
            if ":" in tok:
                s, t = tok.rsplit(":", 1)
                r = row_of.get((s, t))
                if r is not None:
                    mem.append(r)
        mem.sort(key=lambda r: r.z_um)
        n_obj += 1
        spans.append((o.object_id, mem))
    if a.bridges and os.path.exists(a.bridges) and os.path.getsize(a.bridges) > 0:
        try:
            br = pd.read_csv(a.bridges, sep="\t")
        except pd.errors.EmptyDataError:
            br = pd.DataFrame(columns=["slide_lo", "tls_lo", "slide_hi", "tls_hi", "accepted"])
        for r in br[~br.accepted.astype(bool)].itertuples():
            lo, hi = row_of.get((r.slide_lo, r.tls_lo)), row_of.get((r.slide_hi, r.tls_hi))
            if lo is not None and hi is not None:
                spans.append((f"bridge {r.slide_lo}:{r.tls_lo}->{r.slide_hi}:{r.tls_hi}", [lo, hi]))
    for label, mem in spans:
        have = {r.z_um for r in mem}
        for z in zs:
            if z <= mem[0].z_um or z >= mem[-1].z_um or z in have or z not in slide_by_z:
                continue
            lo = max((r for r in mem if r.z_um < z), key=lambda r: r.z_um)
            hi = min((r for r in mem if r.z_um > z), key=lambda r: r.z_um)
            w = (z - lo.z_um) / (hi.z_um - lo.z_um)
            cx = lo.cx_um_canvas + w * (hi.cx_um_canvas - lo.cx_um_canvas)
            cy = lo.cy_um_canvas + w * (hi.cy_um_canvas - lo.cy_um_canvas)
            r_um = np.sqrt(max(float(lo.core_area_mm2), float(hi.core_area_mm2)) * 1e6 / np.pi)
            tol = max(TOL_BASE_UM, 0.8 * r_um) + 2.0 * min(z - lo.z_um, hi.z_um - z)
            slide = slide_by_z[z]
            if slide not in maps:
                maps[slide] = canvas_map(a.volume, a.pred, slide, basis)
            m = maps[slide]
            if m is None:
                print(f"  {label} z={z} {slide}: no canvas map, skipped")
                continue
            cxy, nxy = m
            k = int(np.argmin((cxy[:, 0] - cx) ** 2 + (cxy[:, 1] - cy) ** 2))
            if np.hypot(cxy[k, 0] - cx, cxy[k, 1] - cy) > tol:
                print(f"  {label} z={z} {slide}: no cell within tol of the target, skipped")
                continue
            t = {"cx_um": round(float(nxy[k, 0]), 1), "cy_um": round(float(nxy[k, 1]), 1),
                 "tol_um": round(float(tol), 1), "object_id": label, "z_um": float(z),
                 "z_below": float(lo.z_um), "z_above": float(hi.z_um),
                 "cx_um_canvas": round(float(cx), 1), "cy_um_canvas": round(float(cy), 1)}
            if a.tls_root:
                pts = []
                for nb in (lo, hi):
                    mn = maps.setdefault(nb.slide_id, canvas_map(a.volume, a.pred, nb.slide_id, basis))
                    if mn is None:
                        continue
                    fp = core_footprint_canvas(a.tls_root, nb.slide_id, nb.tls_id, mn)
                    if fp is not None:
                        pts.append(fp)
                if pts:
                    bm = footprint_bitmap(np.vstack(pts), m)
                    if bm:
                        t.update(bm)
            targets.setdefault(slide, []).append(t)
    json.dump(targets, open(a.out, "w"), indent=1)
    n_t = sum(len(v) for v in targets.values())
    print(f"{a.sample}: {n_obj} objects with >= {a.min_sections} sections -> "
          f"{n_t} rescue targets on {len(targets)} slides -> {a.out}")
    for s, v in targets.items():
        print("  " + s + ": " + "; ".join(f"{t['object_id']} z={t['z_um']:.0f} tol={t['tol_um']:.0f}" for t in v))


if __name__ == "__main__":
    main()
