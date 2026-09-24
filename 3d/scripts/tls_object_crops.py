#!/usr/bin/env python3
from __future__ import annotations

import csv
import importlib.util
import json
import multiprocessing as mp
import os
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from skimage.measure import find_contours

ROOT = Path(os.environ.get("THREED_ROOT", os.path.join(os.environ.get("PROJECTS_ROOT", "/data/heid"), "3d")))
SAMPLE = "HT891Z1"
TLSD = ROOT / "tls_define"
OBJ = ROOT / "objects_3d/outputs/S9_tls_objects/objects.tsv"
VOL = ROOT / "reconstruction/volume_8um"
TILES = ROOT / "reconstruction/volume_tiles/G_withdrawn"
MASKS = TLSD / "outputs" / f"{SAMPLE}_masks"
TLSROOT = Path(os.environ["HTAN3D_TLSROOT"]) if os.environ.get("HTAN3D_TLSROOT") else None
OUT = TLSD / "outputs" / f"tls_object_crops_{SAMPLE}"
LEVEL_UM = 1.0
PAD_UM = 350.0
MIN_HALF_UM = 450.0
RES = 10.0
MOD = {"native_persection": "he", "restained_from_codex_slide": "codex",
       "xenium_slide_persection": "xenium"}
COL_THIS = (255, 45, 149)
COL_OTHER = (255, 255, 255)


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec)
    sys.modules[name] = m
    spec.loader.exec_module(m)
    return m


TD = _load("tls_he", Path(__file__).resolve().parent / "tls_define_he.py")


def mask_file(slide):
    if TLSROOT is not None and (TLSROOT / slide / f"{slide}_masks.npz").exists():
        return TLSROOT / slide / f"{slide}_masks.npz"
    return MASKS / f"{slide}_masks.npz"


def _mask_worker(arg):
    slide, expected = arg
    os.environ["TLS_SAVE_MASKS"] = "1"
    f = mask_file(slide)
    if f.parent != MASKS:
        return slide, "chain output"
    if f.exists():

        try:
            have = int(np.load(f)["cores"].shape[0])
        except Exception:
            have = -1
        if have >= expected:
            return slide, "cached"
        for g in MASKS.glob(f"{slide}_*"):
            g.unlink()
        print(f"  mask {slide}: cache had {have} cores, table lists {expected}: rebuilt", flush=True)
    try:
        TD.main(SAMPLE, slide, str(MASKS))
        return slide, "ok"
    except SystemExit as e:
        return slide, f"exit:{e}"


def read_tiles(sid, level_um, x0, y0, x1, y1, tiles=None):
    TILES = tiles or globals()["TILES"]
    idx = json.loads((TILES / sid / "index.json").read_text())
    lvs = [L for L in idx["levels"] if abs(L["mpp"] - level_um) < 1e-6]
    if not lvs:
        raise KeyError(f"{sid}: no {level_um:g} um tile level in {TILES / sid / 'index.json'} (tiles not built for this plane)")
    lv = lvs[0]
    ts = idx["tile_px"]
    W, H = lv["width"], lv["height"]
    px0, py0 = int(np.floor(x0 / level_um)), int(np.floor(y0 / level_um))
    px1, py1 = int(np.ceil(x1 / level_um)), int(np.ceil(y1 / level_um))
    out = np.full((py1 - py0, px1 - px0, 3), 255, np.uint8)
    present = set(lv.get("present", []))
    tdir = TILES / sid / f"{level_um:g}"
    rgb = TILES / sid / "rgb" / f"{level_um:g}"
    if rgb.is_dir():
        tdir, present = rgb, set()
    else:
        out[:] = 0
    for r in range(max(0, py0 // ts), min(lv["rows"], py1 // ts + 1)):
        for c in range(max(0, px0 // ts), min(lv["cols"], px1 // ts + 1)):
            key = f"{c}_{r}"
            f = tdir / f"{key}.webp"
            if present and key not in present or not f.exists():
                continue
            t = cv2.imread(str(f), cv2.IMREAD_COLOR)
            if t is None:
                continue
            tx0, ty0 = c * ts, r * ts
            sx0, sy0 = max(px0, tx0), max(py0, ty0)
            sx1, sy1 = min(px1, tx0 + t.shape[1], W), min(py1, ty0 + t.shape[0], H)
            if sx1 <= sx0 or sy1 <= sy0:
                continue
            out[sy0 - py0:sy1 - py0, sx0 - px0:sx1 - px0] = \
                t[sy0 - ty0:sy1 - ty0, sx0 - tx0:sx1 - tx0]
    return out, (px0, py0)


def canvas_affine(meta_cp, slide, basis="G_withdrawn", sid=None):
    import struct


    secs = [q for q in meta_cp["sections"] if q["slide"] == slide]
    if not secs and sid:
        secs = [q for q in meta_cp["sections"] if q["section_id"] == sid]
    if not secs:
        return None
    sid = secs[0]["section_id"]
    f = [x for x in meta_cp["files"] if x["basis"] == basis and x["section_id"] == sid]
    if not f:
        return None
    raw = (VOL / f[0]["file"]).read_bytes()
    magic, n, cw, ch, sub, ncls, _ = struct.unpack("<8sIHHHHI", raw[:24])
    assert magic == b"CELLPT01"
    xy = np.frombuffer(raw[32:32 + 4 * n], dtype="<u2").reshape(n, 2).astype(np.float64)
    cl = np.frombuffer(raw[32 + 5 * n:32 + 6 * n], dtype=np.uint8)
    pred = pd.read_parquet(TD.PRED + f"/{slide}.parquet",
                           columns=["pred_class_name", "x_um", "y_um"])
    if len(pred) != n:
        return None
    idx = pred["pred_class_name"].map({c: i for i, c in enumerate(meta_cp["classes"])}).to_numpy()
    if not np.array_equal(idx, cl):
        return None
    can = xy / sub * 8.0
    nat = pred[["x_um", "y_um"]].to_numpy(np.float64)
    X = np.column_stack([nat, np.ones(len(nat))])
    A, *_ = np.linalg.lstsq(X, can, rcond=None)
    res = float(np.abs(X @ A - can).max())
    return A, sid, res


XGT = VOL / "xenium_gt"


def draw_xenium_cells(img, sid, px0, py0, level_um, basis="G_withdrawn"):
    import struct
    g = json.loads((XGT / "xenium_gt_metadata.json").read_text())
    f = [q for q in g["files"] if q["basis"] == basis and q["section_id"] == sid]
    if not f:
        return img
    raw = (XGT / f[0]["file"]).read_bytes()
    magic, n, W, H, sub, ncls, _ = struct.unpack("<8sIHHHHI", raw[:24])
    assert magic == b"CELLPT01"
    ip = np.frombuffer(raw, np.uint16, 2 * n, 32)
    cls = np.frombuffer(raw, np.uint8, n, 32 + 4 * n + n)
    x = ip[0::2] / sub * 8.0 / level_um - px0
    y = ip[1::2] / sub * 8.0 / level_um - py0
    h, w = img.shape[:2]
    inside = (x >= 0) & (x < w) & (y >= 0) & (y < h)
    img[:] = (img.astype(np.float32) * 0.35).astype(np.uint8)
    r = max(2, int(round(4.0 / level_um)))
    counts = {}
    for ci in np.unique(cls[inside]):
        name = g["classes"][ci]
        hx = g["colours"][ci].lstrip("#")
        col = (int(hx[4:6], 16), int(hx[2:4], 16), int(hx[0:2], 16))
        sel = inside & (cls == ci)
        counts[name] = int(sel.sum())
        for xx, yy in zip(x[sel], y[sel]):
            cv2.circle(img, (int(round(xx)), int(round(yy))), r, col, -1, cv2.LINE_AA)

    y0 = 60
    for name, c in sorted(counts.items(), key=lambda kv: -kv[1]):
        ci = g["classes"].index(name)
        hx = g["colours"][ci].lstrip("#")
        col = (int(hx[4:6], 16), int(hx[2:4], 16), int(hx[0:2], 16))
        cv2.rectangle(img, (w - 250, y0 - 12), (w - 232, y0 + 4), col, -1)
        txt = f"{name} {c}"
        cv2.putText(img, txt, (w - 224, y0 + 3), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(img, txt, (w - 224, y0 + 3), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    (255, 255, 255), 1, cv2.LINE_AA)
        y0 += 22
    return img


def main():
    global SAMPLE, OBJ, VOL, TILES, MASKS, OUT, XGT
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--sample", default=SAMPLE)
    ap.add_argument("--objects", default="", help="3-D TLS objects.tsv; default the 891 one")
    ap.add_argument("--table", default="", help="tls_3d_table tsv; default the cohort table")
    ap.add_argument("--volume", default="", help="volume_8um dir; default the 891 one")
    ap.add_argument("--tiles", default="", help="tile pyramid basis dir; default the 891 one")
    ap.add_argument("--out", default="", help="crop root; default outputs/tls_object_crops_<sample>")
    ap.add_argument("--swap-tiles", default="", dest="swap_tiles",
                    help="the second-modality arm's tile pyramid basis dir (891: CODEX / Xenium scans); "
                         "a Xenium section's xenium/ image is read from it")
    ap.add_argument("--per-slide", default="", dest="per_slide",
                    help="cohort_per_slide.tsv for the modality of each section; "
                         "omit for a single-modality (H&E) sample")
    ap.add_argument("--shard", default="", help="K/N: render only objects K, K+N, K+2N, ... (one array element of N)")
    a = ap.parse_args()
    SAMPLE = a.sample
    if a.objects: OBJ = Path(a.objects)
    if a.volume: VOL = Path(a.volume)
    XGT = VOL / "xenium_gt"
    if a.tiles: TILES = Path(a.tiles)
    SWAP_TILES = Path(a.swap_tiles) if a.swap_tiles else None
    MASKS = TLSD / "outputs" / f"{SAMPLE}_masks"
    OUT = Path(a.out) if a.out else TLSD / "outputs" / f"tls_object_crops_{SAMPLE}"
    table = Path(a.table) if a.table else TLSD / "outputs/tls_3d_table.tsv"
    is_891 = SAMPLE == "HT891Z1"


    sys.path.insert(0, str(Path(__file__).resolve().parent))
    CP = _load("build_cell_planes", Path(__file__).resolve().parent / "build_cell_planes.py")
    BCP = _load("build_cell_points", Path(__file__).resolve().parent / "build_cell_points.py")
    meta = json.loads((VOL / "metadata.json").read_text())


    E = {"--prep-root": "HTAN3D_PREP", "--manifest": "HTAN3D_MAN",
         "--edge-runs": "HTAN3D_EDGES", "--anchor": "HTAN3D_ANCHOR"}
    pa = [x for f, v in E.items() if os.environ.get(v) for x in (f, os.environ[v])]
    place = BCP.canvas_placer(VOL, meta, pa + ["--out", str(VOL)] if pa else None)


    join = {}
    rm = ({r["slide"]: r for r in csv.DictReader(open(CP.RUNMAN), delimiter="\t")}
          if is_891 else {})
    meta_cp = json.loads((VOL / "cell_points_metadata.json").read_text())
    aff_cache = {}

    tab = pd.read_csv(table, sep="\t", comment="#")
    tab = tab[tab.sample_id == SAMPLE]
    per_path = Path(a.per_slide) if a.per_slide else (TLSD / "outputs/cohort_per_slide.tsv" if is_891 else None)
    mod_of_u = {}
    if per_path is not None and per_path.exists():
        per = pd.read_csv(per_path, sep="\t")
        per = per[per.sample_id == SAMPLE]
        mod_of_u = {int(r.section_u): MOD.get(r.he_plane_kind, r.he_plane_kind)
                    for r in per.itertuples()}


    u_of_sid = {}
    _sec = os.environ.get("TLS_SECTIONS", "")
    if _sec and os.path.exists(_sec):
        _t = pd.read_csv(_sec, sep="\t", dtype=str)
        if "u_number" in _t.columns:
            u_of_sid = {s: int(u) for s, u in zip(_t["section_id"], _t["u_number"])}

    def u_of(r):
        if hasattr(r, "section_u"):
            return int(r.section_u)
        if str(r.section_id) in u_of_sid:
            return u_of_sid[str(r.section_id)]
        import re
        return int(re.search(r"U(\d+)$", str(r.section_id)).group(1))
    row_of = {(r.slide_id, r.tls_id): r for r in tab.itertuples()}
    n_tls_of_slide = tab.groupby("slide_id").size().to_dict()

    objs = list(csv.DictReader(open(OBJ), delimiter="\t"))
    if a.shard:
        k_, n_ = (int(v) for v in a.shard.split("/"))
        objs = objs[k_::n_]
        print(f"shard {k_}/{n_}: {len(objs)} objects", flush=True)
    members = {}
    for o in objs:

        members[o["object_id"]] = [tuple(m.rsplit(":", 1)) for m in str(o.get("member_tls") or "").split(";") if ":" in m]


    slides = sorted({s for ms in members.values() for s, _ in ms})
    MASKS.mkdir(parents=True, exist_ok=True)
    with mp.get_context("fork").Pool(min(8, mp.cpu_count())) as pool:
        for s, st in pool.imap_unordered(_mask_worker, [(s, int(n_tls_of_slide.get(s, 0))) for s in slides]):
            print(f"  mask {s}: {st}", flush=True)


    n_img = 0
    for o in objs:
        oid = o["object_id"]
        mem = members[oid]
        if not mem:
            print(f"  {oid}: no per-section member, no crop", flush=True); continue
        rows = [row_of[(s, t)] for s, t in mem]
        cx = np.array([r.cx_um_canvas for r in rows]); cy = np.array([r.cy_um_canvas for r in rows])
        half = max(MIN_HALF_UM, 0.5 * max(np.ptp(cx), np.ptp(cy)) + PAD_UM)
        wx0, wx1 = cx.mean() - half, cx.mean() + half
        wy0, wy1 = cy.mean() - half, cy.mean() + half
        odir = OUT / oid
        for (slide, tid), r in zip(mem, rows):
            u = u_of(r)

            sid = str(r.section_id) if hasattr(r, "section_id") else f"{SAMPLE}-U{u}"
            mod = mod_of_u.get(u, "he")
            j = join.get(sid)
            aff = None
            if j is None:


                if slide not in aff_cache:
                    aff_cache[slide] = canvas_affine(meta_cp, slide, sid=sid)
                aff = aff_cache[slide]
                if aff is None:
                    print(f"  {oid} {sid}: no join and no cell map, skipped", flush=True); continue
                if aff[1] != sid:
                    print(f"  {oid} {sid}: cell map places {slide} on {aff[1]}", flush=True)
                    sid = aff[1]
                j = {"slide": slide, "route": "affine"}
                print(f"  {oid} {sid}: affine from cell map, max residual {aff[2]:.2f} um",
                      flush=True)
            elif j["slide"] != slide:
                print(f"  {oid} {sid}: table says {slide}, join says {j['slide']} -- using join",
                      flush=True)
            mf = mask_file(j["slide"])
            if not mf.exists():
                print(f"  {oid} {sid}: no masks", flush=True); continue
            mz = np.load(mf)
            cores, xm, ym = mz["cores"], float(mz["x_min"]), float(mz["y_min"])
            k = int(tid.split("-")[1]) - 1
            if k >= len(cores):
                print(f"  {oid} {sid}: {tid} beyond {len(cores)} cores", flush=True); continue
            mpp0 = float(rm[j["slide"]]["native_mpp"]) if j["slide"] in rm else None

            def to_canvas_um(pts_grid_rc):
                x_um = xm + pts_grid_rc[:, 1] * RES
                y_um = ym + pts_grid_rc[:, 0] * RES
                if j["route"] == "affine":
                    X = np.column_stack([x_um, y_um, np.ones(len(x_um))])
                    return X @ aff[0]
                x0, y0 = x_um / mpp0, y_um / mpp0
                if j["route"] == "he":
                    c8 = place["to_canvas"]("G_withdrawn", sid, np.stack([x0, y0], 1),
                                            tuple(j["level0"]))
                else:
                    A = np.array(j["matrix_plane_to_crop_px"], float)
                    inv = np.linalg.inv(A[:, :2])
                    qx, qy = x0 - A[0, 2], y0 - A[1, 2]
                    x4 = inv[0, 0] * qx + inv[0, 1] * qy
                    y4 = inv[1, 0] * qx + inv[1, 1] * qy
                    c8 = place["to_canvas"]("G_withdrawn", sid, np.stack([x4, y4], 1),
                                            tuple(j["plane_grid_px"]))
                return c8 * 8.0

            def render(img, px0, py0, mod_lab):

                for kk, core in enumerate(cores):
                    for cnt in find_contours(core.astype(float), 0.5):
                        pu = to_canvas_um(cnt)
                        pp = np.round(pu / LEVEL_UM - [px0, py0]).astype(np.int32)
                        bold = kk == k
                        col = COL_THIS if bold else COL_OTHER
                        cv2.polylines(img, [pp.reshape(-1, 1, 2)], True,
                                      (col[2], col[1], col[0]), 4 if bold else 1, cv2.LINE_AA)

                sb = int(200 / LEVEL_UM)
                cv2.rectangle(img, (20, img.shape[0] - 30), (20 + sb, img.shape[0] - 22), (0, 0, 0), -1)
                cv2.putText(img, "200 um", (20, img.shape[0] - 36), cv2.FONT_HERSHEY_SIMPLEX,
                            0.6, (0, 0, 0), 2, cv2.LINE_AA)

                lab = (f"{oid.replace('3d-tls-', '3D-').replace('2d-tls-', '2D-')}  U{u}  {mod_lab.upper()}  z={r.z_um:.0f} um  "
                       f"core {r.core_area_mm2:.4f} mm2")
                cv2.putText(img, lab, (20, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 4, cv2.LINE_AA)
                cv2.putText(img, lab, (20, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 0), 2, cv2.LINE_AA)
                d = odir / mod_lab
                d.mkdir(parents=True, exist_ok=True)
                cv2.imwrite(str(d / f"U{u:03d}.png"), img)

            wrote = 0
            if mod == "xenium":

                src = SWAP_TILES if SWAP_TILES is not None and (SWAP_TILES / sid / "index.json").exists() else None
                try:
                    img, (px0, py0) = read_tiles(sid, LEVEL_UM, wx0, wy0, wx1, wy1, tiles=src)
                    render(draw_xenium_cells(img, sid, px0, py0, LEVEL_UM), px0, py0, "xenium"); wrote += 1
                except (KeyError, FileNotFoundError) as e:
                    print(f"  skip {sid} xenium: {e}", flush=True)

                try:
                    img, (px0, py0) = read_tiles(sid, LEVEL_UM, wx0, wy0, wx1, wy1)
                    render(img, px0, py0, "he"); wrote += 1
                except (KeyError, FileNotFoundError) as e:
                    print(f"  skip {sid} he: {e}", flush=True)
            else:
                try:
                    img, (px0, py0) = read_tiles(sid, LEVEL_UM, wx0, wy0, wx1, wy1)
                except (KeyError, FileNotFoundError) as e:
                    print(f"  skip {sid}: {e}", flush=True); continue
                render(img, px0, py0, mod); wrote += 1
            n_img += wrote
        print(f"  {oid}: {len(mem)} members -> {odir}", flush=True)
    print(f"wrote {n_img} images under {OUT}")


if __name__ == "__main__":
    main()
