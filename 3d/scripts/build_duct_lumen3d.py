#!/usr/bin/env python3

import argparse
import json
import os
import struct
import sys
from pathlib import Path

import numpy as np
from PIL import Image
from scipy.ndimage import (binary_closing, binary_dilation, binary_fill_holes, find_objects,
                           distance_transform_edt, gaussian_filter, label, shift as ndshift)

HERE = Path(__file__).resolve().parent


def _load(name, p):
    import importlib.util
    spec = importlib.util.spec_from_file_location(name, p)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


BM = _load("bm", HERE / "build_meshes.py")

RES = 10.0
UM = 8.0
EMPTY_R_UM = 12.0
GROW_R_UM = 12.0
MIN_AREA_MM2 = 0.003
MAX_AREA_MM2 = 2.0
RIM_UM = 30.0
EDGE_UM = 40.0
TISSUE_SIGMA_PX = 10
TISSUE_THR = 0.015
TISSUE_CLOSE_PX = 10
LINK_TOL_UM = 40.0
LINK_DRIFT_PER_UM = 1.0
LINK_MAX_DZ_UM = 60.0
LINK_OVERLAP_MIN = 0.3
MIN_SECTIONS = 2
TLS_ONLY = True
TLS_INSIDE_FRAC = 0.5
DUCT_CLASSES = ("Tumor", "NonMalignant_Parenchymal")
DUCT_COLOUR = "#f59e0b"


def read_plane(vol, p):
    f = sorted((vol / "G_withdrawn/cell_points").glob(f"z{p['index']:02d}_*.cells.bin"))
    if not f:
        return None
    raw = open(f[0], "rb").read()
    magic, n, _cw, _ch, sub, ncls, _ = struct.unpack("<8sIHHHHI", raw[:24])
    assert magic == b"CELLPT01"
    xy = np.frombuffer(raw[32:32 + 4 * n], dtype="<u2").reshape(n, 2).astype(np.float64) / sub * UM
    cl = np.frombuffer(raw[32 + 5 * n:32 + 6 * n], dtype=np.uint8)
    return xy, cl


def lumen_kind(counts, classes):
    n = sum(counts.values())
    if n < 10:
        return "cavity (unlined)"
    f = {c: counts.get(c, 0) / n for c in classes}
    if f.get("Tumor", 0) >= 0.5:
        return "tumor duct"
    if f.get("NonMalignant_Parenchymal", 0) >= 0.5:
        return "duct"
    if sum(f.get(c, 0) for c in DUCT_CLASSES) >= 0.5:
        return "duct (mixed lining)"
    if f.get("Endothelial", 0) >= 0.3:
        return "vessel"
    return "cavity"


import time as _time

def _lap(msg, _st=[None]):
    now=_time.time(); print(f"    timing: {msg} {now - (_st[0] or now):.0f} s", flush=True); _st[0]=now


class Win:
    __slots__ = ("y0", "y1", "x0", "x1", "m", "b")

    def __init__(self, y0, x0, m):
        ys, xs = np.nonzero(m)
        self.b = (y0 + int(ys.min()), y0 + int(ys.max()) + 1, x0 + int(xs.min()), x0 + int(xs.max()) + 1)

        self.m = m[ys.min():ys.max() + 1, xs.min():xs.max() + 1]
        self.y0, self.y1, self.x0, self.x1 = self.b

    def sum(self):
        return int(self.m.sum())

    def nonzero(self):
        ys, xs = np.nonzero(self.m)
        return ys + self.y0, xs + self.x0

    def crop(self, y0, y1, x0, x1):
        out = np.zeros((y1 - y0, x1 - x0), bool)
        a0, a1, b0, b1 = max(y0, self.y0), min(y1, self.y1), max(x0, self.x0), min(x1, self.x1)
        if a1 > a0 and b1 > b0:
            out[a0 - y0:a1 - y0, b0 - x0:b1 - x0] = self.m[a0 - self.y0:a1 - self.y0, b0 - self.x0:b1 - self.x0]
        return out

    def paste(self, full):
        full[self.y0:self.y1, self.x0:self.x1] |= self.m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--volume", required=True)
    ap.add_argument("--sample", required=True)
    ap.add_argument("--tls-near-um", type=float, default=50.0, dest="tls_near_um",
                    help="keep only the lumens that lie inside a TLS region of the same section or "
                         "within this many micrometres of one; 0 keeps every lumen")
    ap.add_argument("--from-2d", default="", dest="from_2d",
                    help="S15_duct_2d dir (build_duct_lumen2d.py): take the per-section lumens from "
                         "there (H&E colour at 1 um) instead of the cell-free cavities")
    a = ap.parse_args()
    F2D = Path(a.from_2d) if a.from_2d else None
    fine_path = {}
    VOL = Path(a.volume)
    obj = os.environ.get("HTAN3D_OBJ3D", "")
    OUT = (Path(obj) / "S15_duct_3d") if obj else HERE.parent / f"outputs/S15_duct_3d_{a.sample}"
    OUT.mkdir(parents=True, exist_ok=True)
    for old in OUT.glob("*.bin"):
        old.unlink()

    meta = json.loads((VOL / "cell_points_metadata.json").read_text())
    vmeta = json.loads((VOL / "metadata.json").read_text())
    classes = meta["classes"]
    cw, ch = meta["canvas_px"]["width"], meta["canvas_px"]["height"]
    nx, ny = int(np.ceil(cw * UM / RES)), int(np.ceil(ch * UM / RES))
    planes = sorted(vmeta["encodings"]["G_withdrawn"]["planes"], key=lambda p: p["z_um"])


    TLSNEAR, tls_meta = {}, VOL / "tls_regions_metadata.json"
    if a.tls_near_um > 0:
        if not tls_meta.exists():
            raise SystemExit(f"{tls_meta} is missing: build the TLS region layer first, or pass "
                             f"--tls-near-um 0 to keep every lumen")
        tm = json.loads(tls_meta.read_text())
        iy = np.clip((np.arange(ny) * RES / UM).astype(int), 0, ch - 1)
        ix = np.clip((np.arange(nx) * RES / UM).astype(int), 0, cw - 1)
        for q in tm["planes"]:
            reg = np.zeros((ch, cw), bool)
            for key in ("file3d", "file2d"):
                f = VOL / "G_withdrawn" / "tls" / q[key]
                if not f.exists():
                    continue
                arr = np.array(Image.open(f))
                reg |= (arr[..., -1] > 0) if arr.ndim == 3 else (arr > 0)
            if not reg.any():
                TLSNEAR[q["section_id"]] = None
                continue


            TLSNEAR[q["section_id"]] = distance_transform_edt(~reg[np.ix_(iy, ix)]) * RES <= a.tls_near_um
        n_with = sum(1 for v in TLSNEAR.values() if v is not None)
        print(f"  TLS gate: {n_with} of {len(TLSNEAR)} sections carry a TLS; a lumen is kept "
              f"within {a.tls_near_um:g} um of one", flush=True)

    def tls_ok(sid, y0, x0, m):
        if a.tls_near_um <= 0:
            return True
        g = TLSNEAR.get(sid)
        if g is None:
            return False
        return bool((g[y0:y0 + m.shape[0], x0:x0 + m.shape[1]] & m).any())


    data, masks, info, stats = {}, {}, {}, {}
    r_empty = max(1, int(round(EMPTY_R_UM / RES)))
    r_grow = int(round(GROW_R_UM / RES))
    r_rim = max(1, int(round(RIM_UM / RES)))
    px_min, px_max = MIN_AREA_MM2 * 1e6 / RES ** 2, MAX_AREA_MM2 * 1e6 / RES ** 2
    for p in planes:
        r = read_plane(VOL, p)
        if r is None:
            continue
        sid, z = p["section_id"], float(p["z_um"])
        xy, cl = r
        xi = np.clip((xy[:, 0] / RES).astype(int), 0, nx - 1)
        yi = np.clip((xy[:, 1] / RES).astype(int), 0, ny - 1)
        occ = np.zeros((ny, nx), bool)
        occ[yi, xi] = True
        dens = np.zeros((ny, nx), np.float32)
        np.add.at(dens, (yi, xi), 1.0)
        dens = gaussian_filter(dens, TISSUE_SIGMA_PX)
        tissue = dens > TISSUE_THR
        tissue = binary_closing(tissue, iterations=TISSUE_CLOSE_PX)
        tissue = binary_fill_holes(tissue)
        keep, k = {}, 0
        fp = (F2D / f"z{p['index']:02d}_{sid.split('-')[-1]}.lumens.npz") if F2D else None
        if fp is not None and fp.exists():


            zz = np.load(fp, allow_pickle=False)
            lab_f = zz["lab"]; fine_um = float(zz["um_per_px"])
            fine_path[sid] = fp
            r_rim = max(1, int(round(RIM_UM / RES)))
            ids = np.unique(lab_f); ids = ids[ids > 0]
            objs = find_objects(lab_f)
            for i in ids:
                slc = objs[int(i) - 1]
                mf = lab_f[slc] == i
                ysf, xsf = np.nonzero(mf)
                ysf = ysf + slc[0].start; xsf = xsf + slc[1].start


                yg = np.clip((ysf * fine_um / RES).astype(int), 0, ny - 1)
                xg = np.clip((xsf * fine_um / RES).astype(int), 0, nx - 1)
                pad = r_rim + 1
                y0, y1 = max(0, int(yg.min()) - pad), min(ny, int(yg.max()) + pad + 1)
                x0, x1 = max(0, int(xg.min()) - pad), min(nx, int(xg.max()) + pad + 1)
                mc = np.zeros((y1 - y0, x1 - x0), bool)
                mc[yg - y0, xg - x0] = True
                mc = binary_fill_holes(mc)
                ringc = binary_dilation(mc, iterations=r_rim) & ~mc
                inw = (yi >= y0) & (yi < y1) & (xi >= x0) & (xi < x1)
                sel = np.zeros(len(yi), bool)
                sel[inw] = ringc[yi[inw] - y0, xi[inw] - x0]
                if not tls_ok(sid, y0, x0, mc):
                    continue
                m = Win(y0, x0, mc)
                counts = {classes[c]: int(v) for c, v in
                          zip(*np.unique(cl[sel], return_counts=True))}
                k += 1
                keep[k] = m
                info[(sid, k)] = {"area_mm2": float(mf.sum() * fine_um ** 2 / 1e6),
                                  "cx_um": float(xsf.mean() * fine_um), "cy_um": float(ysf.mean() * fine_um),
                                  "rim": counts, "kind": lumen_kind(counts, classes), "fine_id": int(i)}
        elif fp is not None:
            print(f"  {sid}: no 2-D lumen file {fp.name}, section has no lumens", flush=True)
        else:


            empty = tissue & ~binary_dilation(occ, iterations=r_empty)
            occ_d = binary_dilation(occ, iterations=r_empty)
            edge = distance_transform_edt(tissue) * RES
            lab, n = label(empty)
            areas = np.bincount(lab.ravel())
            mins = np.full(n + 1, np.inf)
            np.minimum.at(mins, lab.ravel(), edge.ravel())
            for i in range(1, n + 1):
                if areas[i] < px_min or areas[i] > px_max or mins[i] < EDGE_UM:
                    continue
                m = lab == i
                if r_grow:
                    m = binary_dilation(m, iterations=r_grow) & tissue & ~occ
                    m = binary_fill_holes(m)
                ring = binary_dilation(m, iterations=r_rim) & ~m
                sel = ring[yi, xi]
                counts = {classes[c]: int(v) for c, v in
                          zip(*np.unique(cl[sel], return_counts=True))}
                if not tls_ok(sid, 0, 0, m):
                    continue
                ys, xs = np.nonzero(m)
                k += 1
                keep[k] = Win(0, 0, m)
                info[(sid, k)] = {"area_mm2": float(areas[i] * RES ** 2 / 1e6),
                                  "cx_um": float(xs.mean() * RES), "cy_um": float(ys.mean() * RES),
                                  "rim": counts, "kind": lumen_kind(counts, classes)}
        data[sid] = (z, xy, cl)
        masks[sid] = keep
        stats[sid] = {"z_um": z, "n_cells": int(len(xy)), "n_cavities": len(keep),
                      "tissue_mm2": float(tissue.sum() * RES ** 2 / 1e6)}
        print(f"  {sid}: z {z:.0f} cells {len(xy)} tissue {stats[sid]['tissue_mm2']:.2f} mm2 "
              f"cavities {len(keep)}", flush=True)
    order = sorted(masks, key=lambda s: data[s][0])


    node = {(s, i): (s, i) for s in order for i in masks[s]}

    def find(x):
        while node[x] != x:
            node[x] = node[node[x]]
            x = node[x]
        return x

    def bbox(m):
        return m.b

    BB = {s: {i: bbox(m) for i, m in masks[s].items()} for s in order}
    n_links = 0
    for skip in (1, 2, 3):
        for s0, s1 in zip(order, order[skip:]):
            dz = data[s1][0] - data[s0][0]
            if skip > 1 and dz > LINK_MAX_DZ_UM:
                continue
            it = max(1, int(round((LINK_TOL_UM + LINK_DRIFT_PER_UM * dz) / RES)))
            for i, m0 in masks[s0].items():
                y0, y1, x0, x1 = BB[s0][i]
                y0, y1 = max(0, y0 - it), min(ny, y1 + it)
                x0, x1 = max(0, x0 - it), min(nx, x1 + it)
                d0 = binary_dilation(m0.crop(y0, y1, x0, x1), iterations=it)
                for j, m1 in masks[s1].items():
                    b1 = BB[s1][j]
                    if b1[0] >= y1 or b1[1] <= y0 or b1[2] >= x1 or b1[3] <= x0:
                        continue
                    w1 = m1.crop(y0, y1, x0, x1)
                    inter = int((d0 & w1).sum())
                    if inter and inter >= LINK_OVERLAP_MIN * min(int(w1.sum()), int(m0.sum())):
                        ra, rb = find((s0, i)), find((s1, j))
                        if ra != rb:
                            node[ra] = rb
                            n_links += 1
    cavities = {}
    for k in node:
        cavities.setdefault(find(k), []).append(k)
    _lap("read 2-D lumens + per-section stats")
    print(f"overlap linking: {n_links} links, {len(cavities)} cavities "
          f"(tolerance {LINK_TOL_UM:g} um + {LINK_DRIFT_PER_UM:g} um/um, overlap >= "
          f"{LINK_OVERLAP_MIN:g} of the smaller, skips within {LINK_MAX_DZ_UM:g} um)", flush=True)


    def mesh_one(item):
        idx, root, members = item
        secs = sorted({s for s, _ in members}, key=lambda s: data[s][0])
        if len(secs) < MIN_SECTIONS:
            return ("rej", {"n_sections": len(secs), "why": f"single section"})
        mem_by_z, cen_by_z = {}, {}
        for s2, i2 in members:
            mem_by_z.setdefault(data[s2][0], []).append(masks[s2][i2])
        zs_m = sorted(mem_by_z)
        ws = [w for ws_ in mem_by_z.values() for w in ws_]
        ys0, ys1 = min(w.y0 for w in ws), max(w.y1 for w in ws) - 1
        xs0, xs1 = min(w.x0 for w in ws), max(w.x1 for w in ws) - 1
        pad = int(round(120.0 / RES))
        y0, y1 = max(0, ys0 - pad), min(ny, ys1 + pad + 1)
        x0, x1 = max(0, xs0 - pad), min(nx, xs1 + pad + 1)
        um_by_z = {}
        for z2, ws_ in mem_by_z.items():
            u = np.zeros((y1 - y0, x1 - x0), bool)
            for w in ws_:
                u |= w.crop(y0, y1, x0, x1)
            um_by_z[z2] = u
            ys, xs = np.nonzero(u)
            cen_by_z[z2] = np.array([ys.mean() + y0, xs.mean() + x0])
        sdf_by_z = {z2: BM.sdf(um_by_z[z2]) for z2 in zs_m}
        zg = np.arange(zs_m[0], zs_m[-1] + 0.5 * BM.Z_STEP, BM.Z_STEP, dtype=np.float32)
        vol = np.empty((len(zg), y1 - y0, x1 - x0), np.float32)
        uns = np.zeros(len(zg), bool)
        zs_a = np.asarray(zs_m, np.float32)
        for k2, z2 in enumerate(zg):
            j = int(np.searchsorted(zs_a, z2, "right")) - 1
            j = max(0, min(j, len(zs_a) - 2)) if len(zs_a) > 1 else 0
            za, zb = zs_a[j], zs_a[min(j + 1, len(zs_a) - 1)]
            span = float(zb - za)
            t = 0.0 if span <= 0 else min(1.0, max(0.0, float((z2 - za) / span)))
            ca, cb = cen_by_z[float(za)], cen_by_z[float(zb)]
            cc = (1.0 - t) * ca + t * cb
            fa = ndshift(sdf_by_z[float(za)], cc - ca, order=1, mode="nearest")
            fb = ndshift(sdf_by_z[float(zb)], cc - cb, order=1, mode="nearest")
            vol[k2] = (1.0 - t) * fa + t * fb
            if span > BM.GAP_UM and 0.0 < t < 1.0:
                uns[k2] = True
        m = BM.mesh_from(vol, zg, uns, RES / BM.UM_PX, smooth_vox=1.0)
        if m is None:
            return ("rej", {"n_sections": len(secs), "why": "empty isosurface"})
        m["verts"][:, 0] += x0 * RES
        m["verts"][:, 1] += y0 * RES
        e = BM.write_mesh(f"_tmp_lumen_{idx:04d}", m, OUT)
        ext = m["verts"].max(0) - m["verts"].min(0)
        v3, f3 = m["verts"], m["faces"]
        p0, p1, p2 = v3[f3[:, 0]], v3[f3[:, 1]], v3[f3[:, 2]]
        vol_um3 = float(abs(np.einsum("ij,ij->i", p0.astype(np.float64),
                                      np.cross(p1.astype(np.float64), p2.astype(np.float64))).sum()) / 6.0)
        rim = {}
        for mm in members:
            for c, v in info[mm]["rim"].items():
                rim[c] = rim.get(c, 0) + v
        n_rim = max(1, sum(rim.values()))
        e.update({"structure": "duct lumen", "kind": lumen_kind(rim, classes),
                  "n_sections": len(secs), "z_min_um": data[secs[0]][0], "z_max_um": data[secs[-1]][0],
                  "n_regions": len(members), "extent_um": round(float(np.sqrt((ext ** 2).sum())), 1),
                  "volume_um3": round(vol_um3, 1), "volume_mm3": vol_um3 / 1e9,
                  "max_area_mm2": round(max(info[mm]["area_mm2"] for mm in members), 4),
                  "rim_cells": int(n_rim),
                  "rim_tumor_frac": round(rim.get("Tumor", 0) / n_rim, 3),
                  "rim_parenchymal_frac": round(rim.get("NonMalignant_Parenchymal", 0) / n_rim, 3),
                  "rim_endothelial_frac": round(rim.get("Endothelial", 0) / n_rim, 3),
                  "members": [[s2, int(i2)] for s2, i2 in members], "_root": root})
        return ("ok", e)

    import multiprocessing as mp
    globals()["_MESH_ONE"] = mesh_one
    items = [(k, root, mem) for k, (root, mem) in
             enumerate(sorted(cavities.items(), key=lambda kv: -len(kv[1])), 1)]
    rep = {"basis": "G_withdrawn", "sample": a.sample,
           "how_defined": (f"H&E colour lumens of build_duct_lumen2d.py ({F2D.name}) on the {RES:g} um grid; "
                           if F2D else
                           f"cell-free pixels inside the tissue (no cell within {EMPTY_R_UM:g} um), "
                           f"components {MIN_AREA_MM2:g}-{MAX_AREA_MM2:g} mm2 at least {EDGE_UM:g} um "
                           f"from the tissue edge; ")
                          + f"lining = cells within {RIM_UM:g} um; linked across "
                          f"sections by >= {LINK_OVERLAP_MIN:g} overlap; meshed on >= {MIN_SECTIONS} sections"
                          + (f"; only the lumens inside a TLS region of their own section or within "
                             f"{a.tls_near_um:g} um of one." if a.tls_near_um > 0 else "."),
           "claim": ("GEOMETRY of H&E lumens" if F2D else "GEOMETRY of cell-free space")
                    + "; the lining kind is a majority of predicted cell classes.",
           "sections": stats, "lumens": [], "rejected": []}
    with mp.get_context("fork").Pool(min(16, mp.cpu_count())) as pool:
        for tag, val in pool.imap_unordered(_mesh_call, items):
            (rep["lumens"] if tag == "ok" else rep["rejected"]).append(val)


    if TLS_ONLY:
        tp = VOL / "tls_regions_metadata.json"
        if not tp.exists():
            print(f"  no TLS region layer at {tp}: every lumen is kept", flush=True)
        else:
            tm = json.loads(tp.read_text())
            base = (tm.get("bases") or ["G_withdrawn"])[0]
            tls_by_sec = {}
            for pl in tm["planes"]:
                if not any(g["kind"] == "3d" for g in pl.get("regions") or []):
                    continue
                f3 = VOL / base / "tls" / pl["file3d"]
                if not f3.exists():
                    continue

                al = np.asarray(Image.open(f3).convert("RGBA"))[..., 3] > 0
                m3 = np.asarray(Image.fromarray(al.astype(np.uint8) * 255).resize((nx, ny), Image.NEAREST)) > 0
                if m3.any():
                    tls_by_sec[pl["section_id"]] = m3
            keep, drop = [], 0
            for e in rep["lumens"]:
                hit = False
                for s2, i2 in e["members"]:
                    w = masks[s2][i2]
                    if a.tls_near_um > 0:


                        if tls_ok(s2, w.y0, w.x0, w.m):
                            hit = True; break
                        continue
                    t = tls_by_sec.get(s2)
                    if t is None:
                        continue
                    inside = (t[w.y0:w.y1, w.x0:w.x1] & w.m).sum()
                    if inside >= TLS_INSIDE_FRAC * w.m.sum():
                        hit = True; break
                if hit:
                    keep.append(e)
                else:
                    drop += 1
            rule = (f"a member cavity inside a TLS or within {a.tls_near_um:g} um of one" if a.tls_near_um > 0
                    else f"a member cavity >= {TLS_INSIDE_FRAC:.0%} under the filled 3-D region")
            print(f"  TLS filter: {len(keep)} lumens kept ({rule}), {drop} dropped", flush=True)
            for e in rep["lumens"]:
                if e not in keep:
                    f = OUT / e["file"]
                    if f.exists():
                        f.unlink()
            rep["lumens"] = keep
    rep["lumens"].sort(key=lambda e: -e["volume_um3"])
    region_id = {}
    for k, e in enumerate(rep["lumens"], 1):
        nid = f"duct-{k:02d}"
        (OUT / e["file"]).replace(OUT / f"{nid}.bin")
        e["name"], e["file"] = nid, f"{nid}.bin"
        e.pop("_root")
        for s2, i2 in e["members"]:
            region_id[(s2, i2)] = nid
    rep["n_lumens"] = len(rep["lumens"])
    rep["kinds"] = {k: sum(1 for e in rep["lumens"] if e["kind"] == k)
                    for k in sorted({e["kind"] for e in rep["lumens"]})}
    (OUT / "duct_lumens.json").write_text(json.dumps(rep, indent=1))


    TP = _load("tlsplanes", HERE / "build_tls_planes.py")
    carve, planes_out = {}, {}
    idx_of = {p["section_id"]: p for p in planes}
    for s in order:
        mem = [(i, region_id[(s, i)]) for i in masks[s] if (s, i) in region_id]
        if not mem:
            continue
        full = np.zeros((ny, nx), bool)
        regs = []
        for i, nid in mem:
            m = masks[s][i]
            m.paste(full)
            ys, xs = m.nonzero()
            regs.append({"id": nid, "kind": "duct", "cx_um": round(float(xs.mean()) * RES, 1),
                         "cy_um": round(float(ys.mean()) * RES, 1),
                         "bbox_um": [int(xs.min()) * RES, int(ys.min()) * RES,
                                     (int(xs.max()) + 1) * RES, (int(ys.max()) + 1) * RES]})
        ring_src = full
        if s in fine_path:

            zz = np.load(fine_path[s], allow_pickle=False)
            lab_f = zz["lab"]
            ring_src = np.isin(lab_f, [info[(s, i)]["fine_id"] for i, _ in mem])
        mc = np.asarray(Image.fromarray(full.astype(np.uint8) * 255).resize((cw, ch), Image.NEAREST)) > 0
        carve[s] = mc
        idx = idx_of[s]["index"]
        u = s.split("-")[-1]
        fn = f"z{idx:02d}_{u}.duct.webp"
        for b in sorted({f["basis"] for f in meta["files"]}):
            d = VOL / b / "duct"
            d.mkdir(parents=True, exist_ok=True)


            fine = s in fine_path
            img = TP.ring(ring_src if fine else mc, fill_alpha=0, rim_px=2 if fine else 2)
            if fine:


                oy, ox = ch * 2, cw * 2
                fy, fx = int(np.ceil(img.shape[0] / oy)), int(np.ceil(img.shape[1] / ox))
                pad = np.zeros((oy * fy, ox * fx, 4), np.uint8)
                pad[:img.shape[0], :img.shape[1]] = img
                img = pad.reshape(oy, fy, ox, fx, 4).max(axis=(1, 3))
            Image.fromarray(img).save(d / fn, format="WEBP", lossless=True, quality=100, method=4)
        planes_out[idx] = {"index": idx, "section_id": s, "z_um": data[s][0], "file": fn,
                           "regions": regs, "n_duct": len(regs)}
    np.savez_compressed(OUT / "duct_planes.npz", **{s: m for s, m in carve.items()})
    (VOL / "duct_regions_metadata.json").write_text(json.dumps(
        {"what": "3-D duct lumens per section (member cavity masks of duct-NN), one ring layer",
         "colour": DUCT_COLOUR, "bases": sorted({f["basis"] for f in meta["files"]}),
         "n_planes": len(planes_out), "planes": sorted(planes_out.values(), key=lambda r: r["index"])}, indent=1))
    _lap("link + TLS filter + ring layers + carve masks")
    print(f"duct ring layer: {len(planes_out)} planes; carve masks: {OUT / 'duct_planes.npz'}", flush=True)

    rows = ["section_id\tz_um\tcavity\tduct_id\tkind\tarea_mm2\tcx_um\tcy_um\trim_cells\trim_tumor_frac"]
    for s in order:
        for i in masks[s]:
            d = info[(s, i)]
            nr = max(1, sum(d["rim"].values()))
            rows.append(f"{s}\t{data[s][0]:g}\t{i}\t{region_id.get((s, i), '')}\t{d['kind']}\t"
                        f"{d['area_mm2']:.4f}\t{d['cx_um']:.0f}\t{d['cy_um']:.0f}\t{sum(d['rim'].values())}\t"
                        f"{d['rim'].get('Tumor', 0) / nr:.3f}")
    (OUT / "duct_cavities_per_section.tsv").write_text("\n".join(rows) + "\n")
    _lap("lumen meshes + report")
    print(f"wrote {OUT}: {rep['n_lumens']} lumens on >= {MIN_SECTIONS} sections "
          f"({len(rep['rejected'])} single-section cavities), kinds {rep['kinds']}", flush=True)


def _mesh_call(item):
    return _MESH_ONE(item)


if __name__ == "__main__":
    main()
