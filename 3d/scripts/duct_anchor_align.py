#!/usr/bin/env python3

import argparse
import json
import os
import time
from pathlib import Path

import struct

import cv2
import numpy as np
from PIL import Image, ImageDraw

Image.MAX_IMAGE_PIXELS = None
_nthr = int(os.environ.get("OMP_NUM_THREADS", "0") or 0)
if _nthr:
    cv2.setNumThreads(_nthr)

from scipy.ndimage import (binary_closing, binary_dilation, binary_fill_holes,
                           binary_opening, distance_transform_edt, find_objects,
                           gaussian_filter, label, maximum_filter)

RES_UM = 1.0
MIN_AREA_UM2 = 400.0
MAX_AREA_UM2 = 150000.0
SPLIT_MIN_SEP_UM = 60.0
SPLIT_SMOOTH_UM = 8.0
EDGE_UM = 12.0
CELL_FREE_UM = 3.0
WALL_LUM_MAX = 160
WALL_DILATE_UM = 2.0
CLOSE_R_FRAC = 1.0
CLOSE_R_MIN_UM = 4.0
CLOSE_R_MAX_UM = 40.0
WHITE_FRAC_MIN = 0.5
HOLE_MAX_FRAC = 0.5

OPEN_UM = 2.0
WHITE_MIN = 200
WHITE_SAT_MAX = 40


LOCAL_SE_UM = 25.0
LOCAL_TOPHAT_MIN = 22
LOCAL_LUM_MIN = 175
LOCAL_SAT_MAX = 55
LOCAL_MIN_THICK_UM = 3.0

LUMEN_FRAC = 0.20


MATCH_MAX_UM = 250.0
MATCH_R2_UM = 120.0
MATCH_IOU_R1 = 0.70
MATCH_IOU_R2 = 0.60
RESID_ANCHOR_MAX_UM = 40.0
MATCH_R3_UM = 100.0
MATCH_IOU_R3 = 0.50
GRID_N = 3
RESID_CELL_MAX_UM = 80.0
EDGE_R_UM = 250.0
EDGE_IOU = 0.55
EDGE_KEEP_UM = 120.0
GREY = (160, 160, 160)
T4_WEDGES = 4
T4_OUTER_FRAC = 0.85
EDGE4_RGB = (255, 225, 0)
T5_MARGIN_UM = 150.0
EDGE5_RGB = (255, 80, 255)
EDGE_MAX_TRIES = 4


ADAPT_Z0_UM = 10.0
ADAPT_Z_STEP = 0.10
ADAPT_IOU_STEP = 0.10
ADAPT_RELAX_STEP = 0.01
ADAPT_TOL_MAX = 2.0
SEARCH_ESCALATION = ((1.5, 0.05), (2.0, 0.10), (3.0, 0.15))
SIZE_STEP_MIN_MATCHES = 6
SIZE_STEP_AREA_TOL = 0.04
SIZE_STEP_RANGE = tuple(round(1.0 + 0.03 * k, 2) for k in range(-6, 7))
ADAPT_RELAX_MAX = 0.10
CORE_RGB = (70, 240, 240)
EDGE_RGB = (245, 130, 48)
N_WEDGES = 4
OUTER_FRAC = 0.6
MATCH_MIN_IOU = 0.55
MATCH_MIN_IOU_FLOOR = 0.40
N_ANCHORS_MIN = 5
N_TIER1 = 5
N_TIER2 = 5
SEP_DIV = 4.0
TIER2_POOL = 40

N_ANCHORS = N_TIER1 + N_TIER2
RESID_DROP_FACTOR = 2.5
RESID_DROP_MIN_UM = 15.0
FIG_SCALE = 1
OUTLINE_RGB = (20, 220, 50)
DARK_RGB = (40, 120, 255)
DARK_LUM_MAX = 105
DARK_GREY_FRAC = 0.80
DARK_CLOSE_UM = 4.0
DARK_OPEN_UM = 6.0
DARK_MIN_AREA_UM2 = 1500.0
DARK_MAX_AREA_UM2 = 150000.0
LABEL_FONT = os.path.join(__import__("matplotlib").get_data_path(), "fonts/ttf/DejaVuSans-Bold.ttf")
LABEL_PT = 22
SMOOTH_UM = 2.0
PAIR_COLOURS = [(230, 25, 75), (60, 180, 75), (0, 130, 200), (245, 130, 48), (145, 30, 180),
                (70, 240, 240), (240, 50, 230), (210, 245, 60), (250, 190, 190), (0, 128, 128)]
PAIR_RIM_PX = 4
MARK_R_PX = 22
MARK_W_PX = 5
ANCHOR_MIN_AREA_UM2 = 1500.0
AREA_REF_UM2 = 4000.0
LINE_W_PX = 6
MARGIN_PX = 0
LEADER_PT = 34
LEADER_UM = 90.0
FIG_MAX_MP = 24e6
CROP_PAD_UM = 120.0


def cell_mask(vol8, rec, shape, grid_um):
    cand = sorted((vol8 / "G_withdrawn/cell_points").glob(f"z{rec['index']:02d}_*.cells.bin"))
    m = np.zeros(shape, bool)
    if not cand:
        return m
    b = cand[0].read_bytes()
    magic, n, cw, ch, sub, ncls = struct.unpack_from("<8sIHHHH", b, 0)
    xy = np.frombuffer(b, np.uint16, 2 * n, 32).reshape(n, 2).astype(np.float64) / sub
    xi = np.clip((xy[:, 0] * 8.0 / grid_um).astype(int), 0, shape[1] - 1)
    yi = np.clip((xy[:, 1] * 8.0 / grid_um).astype(int), 0, shape[0] - 1)
    m[yi, xi] = True
    return binary_dilation(m, iterations=max(1, int(round(CELL_FREE_UM / grid_um))))


def tiles_root(vol, sid, vm):
    base = Path(str(vol).replace("volume_8um", "volume_tiles"))
    swap = Path(str(vol).replace("volume_8um", "volume_tiles_heswap"))
    mod = None
    if vm is not None:
        for q in vm.get("encodings", {}).get("G_withdrawn", {}).get("planes", []):
            if q["section_id"] == sid:
                mod = q.get("modality"); break
    order = [(base, "he"), (swap, "he_swap")] if mod in (None, "he") else [(swap, "he_swap"), (base, "he")]
    for root, src in order:
        if (root / "G_withdrawn" / sid / "index.json").exists():
            return root, src
    return None, None


def load_rgb_tiles(root, sid, mpp):
    d = root / "G_withdrawn" / sid
    ix = json.loads((d / "index.json").read_text())
    lv = [l for l in ix.get("rgb_levels", []) if abs(l["mpp"] - mpp) < 1e-6]
    if not lv:
        return None, None
    lv = lv[0]
    W, H, T = lv["width"], lv["height"], ix["tile_px"]
    rgb = np.zeros((H, W, 3), np.uint8)
    a = np.zeros((H, W), bool)
    for name in lv["present"]:
        c, r = (int(v) for v in name.split("_"))
        t = np.asarray(Image.open(d / "rgb" / f"{mpp:g}" / f"{name}.webp").convert("RGBA"))
        y0, x0 = r * T, c * T
        h, w = min(T, H - y0), min(T, W - x0)
        rgb[y0:y0 + h, x0:x0 + w] = t[:h, :w, :3]
        a[y0:y0 + h, x0:x0 + w] = t[:h, :w, 3] > 0
    return rgb, a


def load_tiles(root, sid, mpp):
    d = root / "G_withdrawn" / sid
    ix = json.loads((d / "index.json").read_text())
    lv = [l for l in ix["levels"] if abs(l["mpp"] - mpp) < 1e-6][0]
    W, H, T = lv["width"], lv["height"], ix["tile_px"]
    g = np.zeros((H, W), np.uint8)
    a = np.zeros((H, W), bool)
    for name in lv["present"]:
        c, r = (int(v) for v in name.split("_"))
        t = np.asarray(Image.open(d / f"{mpp:g}" / f"{name}.webp").convert("RGBA"))
        y0, x0 = r * T, c * T
        h, w = min(T, H - y0), min(T, W - x0)
        g[y0:y0 + h, x0:x0 + w] = t[:h, :w, 0]
        a[y0:y0 + h, x0:x0 + w] = t[:h, :w, 3] > 0
    return g.astype(np.float32), a


def swap_files(vol):
    out = {}
    for v in (Path(str(vol).replace("volume_8um", "volume_4um")), vol):
        p = v / "he_swap_metadata.json"
        if not p.exists():
            continue
        sm = json.loads(p.read_text())
        for q in sm["planes"]:
            if q["basis"] == "G_withdrawn" and q["section_id"] not in out:
                f = v / q["grey_file"]
                if f.exists():
                    out[q["section_id"]] = f
    return out


def white_mask(vol, rec, shape):
    v4 = Path(str(vol).replace("volume_8um", "volume_4um"))
    sid = rec["section_id"]
    p = None
    sm = v4 / "he_swap_metadata.json"
    if sm.exists():
        for q in json.loads(sm.read_text())["planes"]:
            if q["basis"] == "G_withdrawn" and q["section_id"] == sid and q.get("rgb_file"):
                p = v4 / q["rgb_file"]; break
    if p is None:
        cand = sorted((v4 / "G_withdrawn" / "rgb").glob(f"z{rec['index']:02d}_*.webp")) if (v4 / "G_withdrawn" / "rgb").exists() else []
        p = cand[0] if cand else None
    if p is None or not p.exists():
        print(f"    no 4 um RGB for {sid}: the white test is skipped", flush=True)
        return None
    a = np.asarray(Image.open(p).convert("RGBA")).astype(np.int16)
    w = (a[..., :3].min(2) >= WHITE_MIN) & ((a[..., :3].max(2) - a[..., :3].min(2)) <= WHITE_SAT_MAX) & (a[..., 3] > 0)
    im = Image.fromarray(w.astype(np.uint8) * 255).resize((shape[1], shape[0]), Image.NEAREST)
    return np.asarray(im) > 127


VM = None


def load_plane(vol, rec, swaps):
    sid = rec["section_id"]
    root, src = tiles_root(vol, sid, VM)
    if root is not None:
        g, a = load_tiles(root, sid, RES_UM)
        return g, a, src, root / "G_withdrawn" / sid / f"{RES_UM:g}"
    if sid in swaps:
        p, src = swaps[sid], "he_swap(4um)"
    else:
        p, src = vol / rec["webp_lossless"], rec.get("modality", "he") + "(8um)"
    im = np.asarray(Image.open(p).convert("RGBA"))
    return im[:, :, 0].astype(np.float32), im[:, :, 3] > 0, src, p


def _split_blob(m):
    sep_px = max(2, int(round(SPLIT_MIN_SEP_UM / RES_UM)))
    d = gaussian_filter(distance_transform_edt(m).astype(np.float32),
                        max(1.0, SPLIT_SMOOTH_UM / RES_UM))
    mx = maximum_filter(d, size=sep_px | 1)
    seeds = (d == mx) & (d > 0.3 * d.max()) & m
    sl, sn = label(seeds)
    if sn <= 1:
        return [m]


    from skimage.segmentation import watershed
    lab = watershed(-d, markers=sl, mask=m)
    return [pm for k in range(1, sn + 1) for pm in [(lab == k) & m] if pm.any()]


def darks(g, a, rgb=None, start_id=1):
    if rgb is not None:
        lum = rgb.astype(np.float32).mean(2)
        m = (lum <= DARK_LUM_MAX) & a
    else:
        v = g[a]; lo, hi = np.percentile(v, [5, 95])
        m = (g >= lo + DARK_GREY_FRAC * (hi - lo)) & a
    rc = max(1, int(round(DARK_CLOSE_UM / RES_UM))); ro = max(1, int(round(DARK_OPEN_UM / RES_UM)))
    yy, xx = np.ogrid[-rc:rc + 1, -rc:rc + 1]; dc = (xx * xx + yy * yy) <= rc * rc
    yy, xx = np.ogrid[-ro:ro + 1, -ro:ro + 1]; do = (xx * xx + yy * yy) <= ro * ro
    m = binary_opening(binary_closing(m, structure=dc), structure=do)
    m = binary_fill_holes(m)
    edge = distance_transform_edt(a).astype(np.float32) * RES_UM
    lb, n = label(m)
    sizes = np.bincount(lb.ravel()); objs = find_objects(lb)
    keep = np.zeros_like(m); feats = []; k = start_id
    for i in range(1, n + 1):
        area = sizes[i] * RES_UM ** 2
        if area < DARK_MIN_AREA_UM2 or area > DARK_MAX_AREA_UM2:
            continue
        sl0 = objs[i - 1]
        sl = (slice(max(0, sl0[0].start - 2), min(lb.shape[0], sl0[0].stop + 2)),
              slice(max(0, sl0[1].start - 2), min(lb.shape[1], sl0[1].stop + 2)))
        pm = lb[sl] == i
        if edge[sl][pm].min() < EDGE_UM:
            continue
        ys, xs = np.nonzero(pm)
        c = np.array([xs.mean(), ys.mean()])
        Q = np.column_stack([xs - c[0], ys - c[1]]).astype(float)
        w, vv = np.linalg.eigh(Q.T @ Q / max(1, len(Q)))
        elong = float(np.sqrt(max(w[1], 1e-6) / max(w[0], 1e-6)))
        ang = float(np.degrees(np.arctan2(vv[1, 1], vv[0, 1])) % 180.0)
        per = float((binary_dilation(pm) & ~pm).sum())
        circ = float(4 * np.pi * pm.sum() / max(1.0, per ** 2))
        f = {"id": k, "type": "dark", "label": f"D{k}",
             "cx_um": (c[0] + sl[1].start) * RES_UM, "cy_um": (c[1] + sl[0].start) * RES_UM,
             "area_um2": float(area), "elong": elong, "angle_deg": ang, "circ": circ, "sl": sl, "m": pm}
        keep[sl][pm] = True
        feats.append(f); k += 1
    return keep, feats


def paint(full, f, value=True):
    sl = f["sl"]
    full[sl][f["m"]] = value


def union(shape, feats):
    out = np.zeros(shape, bool)
    for f in feats:
        paint(out, f)
    return out


def ducts(g, a, cells=None, white=None, rgb=None, local_seed=False):


    if not a.any():
        return np.zeros_like(a, bool), []
    v = g[a]
    lo, hi = np.percentile(v, [5, 95])
    wall = None
    if rgb is not None:


        r16 = rgb.astype(np.int16)
        white_ = (r16.min(2) >= WHITE_MIN) & ((r16.max(2) - r16.min(2)) <= WHITE_SAT_MAX) & a
        lum = r16.mean(2)
        wall = binary_dilation(lum < WALL_LUM_MAX, iterations=max(1, int(round(WALL_DILATE_UM / RES_UM))))
        if local_seed is not False and np.any(local_seed):


            l8 = np.clip(lum, 0, 255).astype(np.uint8)
            rad = max(1, int(round(LOCAL_SE_UM / RES_UM)))
            se = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * rad + 1, 2 * rad + 1))
            th = cv2.morphologyEx(l8, cv2.MORPH_TOPHAT, se)
            loc = ((th >= LOCAL_TOPHAT_MIN) & (l8 >= LOCAL_LUM_MIN)
                   & ((r16.max(2) - r16.min(2)) <= LOCAL_SAT_MAX) & a)


            trad = max(1, int(round(LOCAL_MIN_THICK_UM / RES_UM)))
            tse = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * trad + 1, 2 * trad + 1))
            core = cv2.morphologyEx(loc.astype(np.uint8), cv2.MORPH_OPEN, tse) > 0
            if core.any():
                lb2, n2 = label(loc, structure=np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]]))
                keep_ids = np.unique(lb2[core])
                keep_ids = keep_ids[keep_ids > 0]
                loc = np.isin(lb2, keep_ids)
                del lb2
            else:
                loc[:] = False
            if local_seed is not True:
                loc &= local_seed
            white_ |= loc
            del th, l8, loc, core
        del lum
        wsel = None
    else:
        white_ = (g <= lo + LUMEN_FRAC * (hi - lo)) & a
        wsel = white
    if cells is not None:
        white_ &= ~cells
    white = white_


    _t0 = time.time()
    white = binary_opening(white, iterations=max(1, int(round(OPEN_UM / RES_UM))))
    edge = distance_transform_edt(a).astype(np.float32) * RES_UM
    lb, n = label(white, structure=np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]]))
    print(f"    lumens: {n} white components after {time.time() - _t0:.0f} s", flush=True)
    sizes = np.bincount(lb.ravel())
    objs = find_objects(lb)
    keep = np.zeros_like(white)
    feats = []
    pad = 2
    _done = 0
    for i in range(1, n + 1):
        if sizes[i] * RES_UM ** 2 < MIN_AREA_UM2:
            continue
        _done += 1
        if _done % 2000 == 0:
            print(f"    lumens: {_done} components handled, {len(feats)} kept, {time.time() - _t0:.0f} s", flush=True)
        sl0 = objs[i - 1]
        sl = (slice(max(0, sl0[0].start - pad), min(lb.shape[0], sl0[0].stop + pad)),
              slice(max(0, sl0[1].start - pad), min(lb.shape[1], sl0[1].stop + pad)))
        m = lb[sl] == i
        pieces = _split_blob(m) if sizes[i] * RES_UM ** 2 > MAX_AREA_UM2 else [m]
        sl_parent = sl
        for pm in pieces:
            sl = sl_parent
            if len(pieces) > 1:


                ys_, xs_ = np.nonzero(pm)
                if not len(ys_):
                    continue
                y0p, y1p, x0p, x1p = ys_.min(), ys_.max() + 1, xs_.min(), xs_.max() + 1
                sl = (slice(sl_parent[0].start + y0p, sl_parent[0].start + y1p),
                      slice(sl_parent[1].start + x0p, sl_parent[1].start + x1p))
                pm = pm[y0p:y1p, x0p:x1p]
            if wall is not None:


                req = float(np.sqrt(pm.sum() / np.pi)) * RES_UM
                rc = int(round(min(CLOSE_R_MAX_UM, max(CLOSE_R_MIN_UM, CLOSE_R_FRAC * req)) / RES_UM))
                if rc >= 1:
                    y0, y1 = max(0, sl[0].start - rc), min(lb.shape[0], sl[0].stop + rc)
                    x0, x1 = max(0, sl[1].start - rc), min(lb.shape[1], sl[1].stop + rc)
                    big = np.zeros((y1 - y0, x1 - x0), bool)
                    big[sl[0].start - y0:sl[0].start - y0 + pm.shape[0],
                        sl[1].start - x0:sl[1].start - x0 + pm.shape[1]] = pm


                    dsk = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * rc + 1, 2 * rc + 1))
                    grown = (cv2.morphologyEx(big.astype(np.uint8), cv2.MORPH_CLOSE, dsk) > 0) & ~wall[y0:y1, x0:x1] & a[y0:y1, x0:x1]
                    grown |= big

                    gl, _gn = label(grown)
                    ids = np.unique(gl[big])
                    ids = ids[ids > 0]
                    grown = np.isin(gl, ids)
                    sl = (slice(y0, y1), slice(x0, x1))
                    pm = grown


            holes = binary_fill_holes(pm) & ~pm
            if holes.any():
                hl, hn = label(holes)
                hs = np.bincount(hl.ravel())
                ok = np.zeros(hn + 1, bool)
                ok[1:] = hs[1:] <= HOLE_MAX_FRAC * pm.sum()
                pm = pm | ok[hl]


            sm = max(1, int(round(SMOOTH_UM / RES_UM)))
            yy, xx = np.ogrid[-sm:sm + 1, -sm:sm + 1]
            dsk2 = (xx * xx + yy * yy) <= sm * sm
            pm2 = binary_closing(binary_opening(pm, structure=dsk2), structure=dsk2)
            if wall is not None:
                pm2 &= ~wall[sl]
            pm2 = binary_fill_holes(pm2) & a[sl]
            if pm2.any():
                pm = pm2
            area = float(pm.sum()) * RES_UM ** 2
            if area < MIN_AREA_UM2 or area > MAX_AREA_UM2:
                continue
            if edge[sl][pm].min() < EDGE_UM:
                continue
            if wsel is not None and float(wsel[sl][pm].mean()) < WHITE_FRAC_MIN:
                continue
            ys, xs = np.nonzero(pm)
            c = np.array([xs.mean(), ys.mean()])
            Q = np.column_stack([xs - c[0], ys - c[1]]).astype(float)
            w, vv = np.linalg.eigh(Q.T @ Q / max(1, len(Q)))
            elong = float(np.sqrt(max(w[1], 1e-6) / max(w[0], 1e-6)))
            ang = float(np.degrees(np.arctan2(vv[1, 1], vv[0, 1])) % 180.0)
            per = float((binary_dilation(pm) & ~pm).sum())
            circ = float(4 * np.pi * pm.sum() / max(1.0, per ** 2))
            f = {"id": len(feats) + 1, "type": "lumen", "label": str(len(feats) + 1),
                 "cx_um": (c[0] + sl[1].start) * RES_UM, "cy_um": (c[1] + sl[0].start) * RES_UM,
                 "area_um2": area, "elong": elong, "angle_deg": ang, "circ": circ,
                 "sl": sl, "m": pm}
            paint(keep, f)
            feats.append(f)
    return keep, feats


def iou_centred(f, g):
    mf, mg = f["m"], g["m"]
    yf, xf = np.nonzero(mf); yg, xg = np.nonzero(mg)
    cf = (yf.mean(), xf.mean()); cg = (yg.mean(), xg.mean())
    H = int(max(mf.shape[0], mg.shape[0]) * 2 + 4); W = int(max(mf.shape[1], mg.shape[1]) * 2 + 4)
    A = np.zeros((H, W), bool); B = np.zeros((H, W), bool)
    A[(yf - cf[0] + H / 2).astype(int), (xf - cf[1] + W / 2).astype(int)] = True
    B[(yg - cg[0] + H / 2).astype(int), (xg - cg[1] + W / 2).astype(int)] = True
    u = (A | B).sum()
    return float((A & B).sum() / u) if u else 0.0


def _pix(f):
    c = f.get("_pix")
    if c is None:
        y, x = np.nonzero(f["m"])
        c = (y - y.mean(), x - x.mean()); f["_pix"] = c
    return c


def global_similarity(mask_a, mask_b, res):
    ya, xa = np.nonzero(mask_a); yb, xb = np.nonzero(mask_b)
    if len(ya) < 10 or len(yb) < 10:
        return 1.0, (0.0, 0.0), (0.0, 0.0)
    s = float(np.sqrt(len(yb) / len(ya)))
    return s, (float(xa.mean()) * res, float(ya.mean()) * res), (float(xb.mean()) * res, float(yb.mean()) * res)


def scaled_features(feats, s, ca, cb):
    if abs(s - 1.0) < 0.005:
        return feats
    out = []
    for f in feats:
        g = dict(f)
        g["cx_um"] = cb[0] + s * (f["cx_um"] - ca[0]); g["cy_um"] = cb[1] + s * (f["cy_um"] - ca[1])
        m = f["m"].astype(np.uint8)
        h, w = m.shape
        g["m"] = cv2.resize(m, (max(1, int(round(w * s))), max(1, int(round(h * s)))), interpolation=cv2.INTER_NEAREST) > 0
        g["area_um2"] = float(f["area_um2"]) * s * s
        g.pop("_pix", None)
        out.append(g)
    return out


def match(fa, fb, pre=None, max_um=MATCH_MAX_UM, min_iou=MATCH_IOU_R1):
    if not fa or not fb:
        return []
    from scipy.spatial import cKDTree
    ok_b = np.array([g["area_um2"] >= ANCHOR_MIN_AREA_UM2 for g in fb])
    cb = np.array([[g["cx_um"], g["cy_um"]] for g in fb], float).reshape(-1, 2)
    tree = cKDTree(cb) if len(cb) else None
    hmax = max([f["m"].shape[0] for f in fa] + [g["m"].shape[0] for g in fb])
    wmax = max([f["m"].shape[1] for f in fa] + [g["m"].shape[1] for g in fb])
    H = int(hmax * 2 + 4); W = int(wmax * 2 + 4)
    canvas = np.zeros((H, W), bool)


    ca = np.array([[f["cx_um"], f["cy_um"]] for f in fa], float).reshape(-1, 2)
    if callable(pre):
        PA = np.asarray(pre(ca), float).reshape(-1, 2)
    elif pre is not None:
        PA = ca @ np.asarray(pre, float)[:, :2].T + np.asarray(pre, float)[:, 2]
    else:
        PA = ca
    best_row = {}
    best_col = {}
    for i, f in enumerate(fa):
        if f["area_um2"] < ANCHOR_MIN_AREA_UM2 or tree is None:
            continue
        px, py = float(PA[i, 0]), float(PA[i, 1])
        cand = [j for j in tree.query_ball_point([px, py], max_um)
                if ok_b[j] and fb[j].get("type", "lumen") == f.get("type", "lumen")
                and min(f["area_um2"], fb[j]["area_um2"]) >= min_iou * max(f["area_um2"], fb[j]["area_um2"])]
        if not cand:
            continue
        fy, fx = _pix(f)
        yi = (fy + H / 2).astype(int); xi = (fx + W / 2).astype(int)
        canvas[yi, xi] = True; nf = len(yi)
        for j in cand:
            g = fb[j]
            gy, gx = _pix(g)
            gyi = (gy + H / 2).astype(int); gxi = (gx + W / 2).astype(int)
            inter = int(canvas[gyi, gxi].sum()); u = nf + len(gyi) - inter
            iou = inter / u if u else 0.0
            if iou < min_iou:
                continue
            d = float(np.hypot(px - g["cx_um"], py - g["cy_um"]))
            size = min(1.0, np.sqrt(min(f["area_um2"], g["area_um2"]) / AREA_REF_UM2))
            sc = iou * (1.0 - d / max_um) * size
            if sc <= 0:
                continue
            if i not in best_row or sc > best_row[i][0]:
                best_row[i] = (sc, j, iou, d)
            if j not in best_col or sc > best_col[j][0]:
                best_col[j] = (sc, i)
        canvas[yi, xi] = False
    out = []
    for i, (sc, j, iou, d) in best_row.items():
        if best_col.get(j, (None, None))[1] != i:
            continue
        out.append({"a": fa[i]["id"], "b": fb[j]["id"], "score": float(sc), "iou": float(iou),
                    "pa": (fa[i]["cx_um"], fa[i]["cy_um"]), "pb": (fb[j]["cx_um"], fb[j]["cy_um"]),
                    "dist_um": float(np.hypot(fa[i]["cx_um"] - fb[j]["cx_um"], fa[i]["cy_um"] - fb[j]["cy_um"])),
                    "pred_dist_um": float(d)})
    out.sort(key=lambda m: -m["score"])
    return out


def robust_affine(pairs):
    src = np.array([m["pa"] for m in pairs], float); dst = np.array([m["pb"] for m in pairs], float)
    w = np.array([m.get("iou", 1.0) for m in pairs], float)
    M = fit_affine_dir(src, dst, w)
    res = np.linalg.norm((M[:, :2] @ src.T).T + M[:, 2] - dst, axis=1)
    thr = max(RESID_DROP_MIN_UM, RESID_DROP_FACTOR * float(np.median(res)))
    keep = res <= thr
    if keep.sum() >= 4 and keep.sum() < len(pairs):
        M = fit_affine_dir(src[keep], dst[keep], w[keep])
        res = np.linalg.norm((M[:, :2] @ src.T).T + M[:, 2] - dst, axis=1)
    for m, r in zip(pairs, res):
        m["resid_affine_um"] = float(r)
    return M


def match_two_rounds(fa, fb, tol=1.0, relax=0.0):
    r1 = match(fa, fb, None, MATCH_MAX_UM * tol, MATCH_IOU_R1 - relax)
    if len(r1) < 4:
        r1 = match(fa, fb, None, MATCH_MAX_UM * tol, MATCH_IOU_R2 - relax)
    if len(r1) < 4:
        return r1, None
    M1 = robust_affine(r1)
    r2 = match(fa, fb, M1, MATCH_R2_UM * tol, MATCH_IOU_R2 - relax)
    if len(r2) < 4:
        return r1, M1
    M2 = robust_affine(r2)


    if len(r2) >= 6:
        FT2 = fit_affine_tps(r2)
        def pred(pts):
            M = FT2["M"]
            return (M[:, :2] @ pts.T).T + M[:, 2] + tps_eval(FT2["src"], FT2["W"], FT2["A"], pts)
        r3 = match(fa, fb, pred, MATCH_R3_UM * tol, MATCH_IOU_R3 - relax)
        if len(r3) >= len(r2):
            M3 = robust_affine(r3)
            return r3, M3
    return r2, M2


def _regions(feats, n_wedges=None, outer_frac=None):
    n_wedges = n_wedges or N_WEDGES
    outer_frac = OUTER_FRAC if outer_frac is None else outer_frac

    xs = np.array([f["cx_um"] for f in feats]); ys = np.array([f["cy_um"] for f in feats])
    x0, x1, y0, y1 = xs.min(), xs.max() + 1e-6, ys.min(), ys.max() + 1e-6
    cx, cy = xs.mean(), ys.mean()
    rad = np.hypot(xs - cx, ys - cy)
    wid = 360.0 / n_wedges
    def _w(x, y):
        if n_wedges == 4:
            return (0 if y < cy else 2) + (0 if x < cx else 1)
        return int((np.degrees(np.arctan2(y - cy, x - cx)) % 360.0) // wid)
    wid_all = np.array([_w(x, y) for x, y in zip(xs, ys)])
    reach = {}
    for w in range(n_wedges):
        sel = wid_all == w
        reach[w] = float(rad[sel].max()) if sel.any() else 0.0
    def cell_of(x, y):
        return (int((x - x0) / (x1 - x0) * GRID_N), int((y - y0) / (y1 - y0) * GRID_N))
    def wedge_of(x, y):
        w = _w(x, y); r = np.hypot(x - cx, y - cy)
        return w if reach[w] > 0 and r >= outer_frac * reach[w] else None
    wedge_of.whole = _w
    return cell_of, wedge_of


def edge_extension(fa, fb, core, pred, tol=1.0, relax=0.0, n_wedges=None, outer_frac=None,
                   prefix="E", start_k=0, min_r=None):
    n_wedges = n_wedges or N_WEDGES
    cell_of, wedge_of = _regions(fa, n_wedges, outer_frac)
    def quadrant(f):
        return wedge_of.whole(f["cx_um"], f["cy_um"])
    xs = np.array([f["cx_um"] for f in fa]); ys = np.array([f["cy_um"] for f in fa])
    cx0, cy0 = xs.mean(), ys.mean()
    def far_enough(f, w):
        return min_r is None or np.hypot(f["cx_um"] - cx0, f["cy_um"] - cy0) >= min_r.get(w, 0.0)
    taken_a = {m["a"] for m in core}; taken_b = {m["b"] for m in core}
    kept, dropped, k = [], [], start_k
    for w in range(n_wedges):
        pools = ([f for f in fa if wedge_of(f["cx_um"], f["cy_um"]) == w and f["id"] not in taken_a and far_enough(f, w)],
                 [f for f in fa if quadrant(f) == w and f["id"] not in taken_a and far_enough(f, w)])


        cands, seen = [], set()
        for sub in pools:
            if not sub:
                continue
            for iou in (EDGE_IOU, EDGE_IOU - 0.10):
                for c in match(sub, [g for g in fb if g["id"] not in taken_b], pred, EDGE_R_UM, iou):
                    if (c["a"], c["b"]) not in seen:
                        seen.add((c["a"], c["b"])); cands.append(c)
        tries = 0
        for c in cands:
            if tries >= EDGE_MAX_TRIES:
                break
            tries += 1
            pp = pred(np.array([c["pa"]], float))[0]
            c["edge_resid_um"] = float(np.hypot(pp[0] - c["pb"][0], pp[1] - c["pb"][1]))
            if c["edge_resid_um"] <= EDGE_KEEP_UM * tol:
                c["edge_label"] = f"{prefix}{w + 1}"
                c["role"] = "edge_kept"; kept.append(c)
                taken_a.add(c["a"]); taken_b.add(c["b"])
                break
            c["edge_label"] = f"{prefix}{w + 1}-{tries}"
            c["role"] = "edge_dropped"; dropped.append(c)
    return kept, dropped


def select_anchors_grid(pairs, feats_a):
    cell_of, wedge_of = _regions(feats_a)
    chosen = []
    for gy in range(GRID_N):
        for gx in range(GRID_N):
            inc = [m for m in pairs if cell_of(*m["pa"]) == (gx, gy)]
            good = [m for m in inc if m.get("resid_affine_um", 0.0) <= RESID_ANCHOR_MAX_UM]
            if good:
                chosen.append(max(good, key=lambda m: m["score"]))
            else:
                near = [m for m in inc if m.get("resid_affine_um", 1e9) <= RESID_CELL_MAX_UM]
                if near:
                    chosen.append(min(near, key=lambda m: m["resid_affine_um"]))
    ok = sorted((m for m in pairs if m.get("resid_affine_um", 0.0) <= RESID_ANCHOR_MAX_UM and m not in chosen),
                key=lambda m: -m["score"])
    sep = 600.0
    while len(chosen) < N_ANCHORS and ok:
        added = False
        for m in ok:
            if len(chosen) >= N_ANCHORS or m in chosen:
                continue
            if all(np.hypot(m["pa"][0] - c["pa"][0], m["pa"][1] - c["pa"][1]) >= sep for c in chosen):
                chosen.append(m); added = True
        if not added:
            if sep <= 200.0:
                break
            sep = max(200.0, sep - 100.0)
    chosen.sort(key=lambda m: -m["score"])
    return chosen[:max(N_ANCHORS, GRID_N * GRID_N)]


def select_anchors(pairs, feats_a=None, tissue_area_um2=None):
    from itertools import combinations
    ok = sorted((m for m in pairs if m.get("resid_affine_um", 0.0) <= RESID_ANCHOR_MAX_UM), key=lambda m: -m["score"])
    if not ok:
        return []
    tier1 = ok[:N_TIER1]
    if tissue_area_um2 is None:
        src = feats_a if feats_a else [{"cx_um": m["pa"][0], "cy_um": m["pa"][1]} for m in pairs]
        xs = np.array([f["cx_um"] for f in src]); ys = np.array([f["cy_um"] for f in src])
        tissue_area_um2 = float((xs.max() - xs.min()) * (ys.max() - ys.min()))
    sep = float(np.sqrt(tissue_area_um2) / SEP_DIV)
    pool = [m for m in ok[N_TIER1:N_TIER1 + TIER2_POOL]]
    P1 = np.array([m["pa"] for m in tier1], float)
    def d(m, n):
        return float(np.hypot(m["pa"][0] - n["pa"][0], m["pa"][1] - n["pa"][1]))
    tier2 = []
    while sep >= 150.0 and len(tier2) < N_TIER2:
        cand = [m for m in pool if np.hypot(P1[:, 0] - m["pa"][0], P1[:, 1] - m["pa"][1]).min() >= sep]
        best, best_sum = [], -1.0
        k = min(N_TIER2, len(cand))
        if k > 0:
            for combo in combinations(range(len(cand)), k):
                if all(d(cand[i], cand[j]) >= sep for i, j in combinations(combo, 2)):
                    ssum = sum(cand[i]["score"] for i in combo)
                    if ssum > best_sum:
                        best, best_sum = [cand[i] for i in combo], ssum
        if len(best) >= N_TIER2 or (best and sep <= 150.0):
            tier2 = best
            break
        if best and len(best) > len(tier2):
            tier2 = best
        sep *= 0.8
    select_anchors.sep_um = sep
    return tier1 + tier2


def _rigid(pairs):
    A = np.array([m["pa"] for m in pairs]); B = np.array([m["pb"] for m in pairs])
    ca, cb = A.mean(0), B.mean(0)
    H = (B - cb).T @ (A - ca)
    U, _S, Vt = np.linalg.svd(H)
    d = np.sign(np.linalg.det(Vt.T @ U.T))
    R = Vt.T @ np.diag([1, d]) @ U.T
    t = ca - R @ cb
    res = np.linalg.norm((R @ B.T).T + t - A, axis=1)
    return R, t, res


def fit_rigid(pairs):
    if len(pairs) < 2:
        return np.eye(2), np.zeros(2), 0.0
    R, t, res = _rigid(pairs)
    for m, r in zip(pairs, res):
        m["resid_um"] = float(r)
    thr = max(RESID_DROP_MIN_UM, RESID_DROP_FACTOR * float(np.median(res)))
    keep = [m for m, r in zip(pairs, res) if r <= thr]
    if len(keep) >= N_ANCHORS_MIN and len(keep) < len(pairs):
        R, t, res = _rigid(keep)
        for m in pairs:
            m["used"] = m in keep
        for m, r in zip(keep, res):
            m["resid_um"] = float(r)
    else:
        for m in pairs:
            m["used"] = True
    return R, t, float(np.sqrt((res ** 2).mean()))


def fit_affine(pairs):
    if len(pairs) < 3:
        return None, float("nan"), []
    def solve(ps):
        A = np.array([m["pa"] for m in ps]); B = np.array([m["pb"] for m in ps])
        X = np.column_stack([B, np.ones(len(B))])
        M, *_ = np.linalg.lstsq(X, A, rcond=None)
        res = np.linalg.norm(X @ M - A, axis=1)
        return M.T, res
    M, res = solve(pairs)
    thr = max(RESID_DROP_MIN_UM, RESID_DROP_FACTOR * float(np.median(res)))
    keep = [m for m, r in zip(pairs, res) if r <= thr]
    if len(keep) >= max(4, N_ANCHORS_MIN) and len(keep) < len(pairs):
        M, res = solve(keep)
        used = keep
    else:
        used = pairs
    for m in pairs:
        m["used_affine"] = m in used
    for m, r in zip(used, res):
        m["resid_affine_um"] = float(r)
    return M, float(np.sqrt((res ** 2).mean())), res


TPS_LAMBDAS = [0.0, 1e-3, 1e-2, 1e-1, 1.0, 10.0, 100.0]
TPS_GRID_PX = 8


def _tps_kernel(r2):
    with np.errstate(divide="ignore", invalid="ignore"):
        k = 0.5 * r2 * np.log(r2)
    k[r2 <= 0] = 0.0
    return k


def fit_affine_dir(src, dst, w=None):
    X = np.column_stack([src, np.ones(len(src))])
    if w is None:
        w = np.ones(len(src))
    sw = np.sqrt(w)[:, None]
    M, *_ = np.linalg.lstsq(X * sw, dst * sw, rcond=None)
    return M.T


def tps_solve(src, v, lam_rel):
    n = len(src)
    d2 = ((src[:, None, :] - src[None, :, :]) ** 2).sum(-1)
    K = _tps_kernel(d2)
    scale = np.abs(K).mean() if n > 1 else 1.0
    Kr = K + lam_rel * scale * np.eye(n)
    Pm = np.column_stack([np.ones(n), src])
    L = np.zeros((n + 3, n + 3)); L[:n, :n] = Kr; L[:n, n:] = Pm; L[n:, :n] = Pm.T
    rhs = np.zeros((n + 3, 2)); rhs[:n] = v
    sol = np.linalg.lstsq(L, rhs, rcond=None)[0]
    return sol[:n], sol[n:]


def tps_eval(src, W, A, pts):
    d2 = ((pts[:, None, :] - src[None, :, :]) ** 2).sum(-1)
    return _tps_kernel(d2) @ W + A[0] + pts @ A[1:]


def fit_affine_tps(pairs, M_fixed=None, lam_final=None):
    src = np.array([m["pa"] for m in pairs], float); dst = np.array([m["pb"] for m in pairs], float)
    w = np.array([m.get("iou", 1.0) for m in pairs], float)
    M = fit_affine_dir(src, dst, w) if M_fixed is None else np.asarray(M_fixed, float)
    pred = (M[:, :2] @ src.T).T + M[:, 2]
    v = dst - pred
    res_aff = np.linalg.norm(v, axis=1)
    n = len(pairs)
    best = (None, np.inf)
    if n >= 6:
        for lam in TPS_LAMBDAS:
            errs = []
            for i in range(n):
                keep = np.arange(n) != i
                Wk, Ak = tps_solve(src[keep], v[keep], lam)
                pv = tps_eval(src[keep], Wk, Ak, src[i:i + 1])[0]
                errs.append(np.linalg.norm(pv - v[i]))
            loo = float(np.sqrt(np.mean(np.square(errs))))
            if loo < best[1]:
                best = (lam, loo)
    lam = best[0] if best[0] is not None else 1.0
    if lam_final is not None:
        lam = lam_final
    W, A = tps_solve(src, v, lam)
    res_tps = np.linalg.norm(v - tps_eval(src, W, A, src), axis=1)
    return {"M": M, "src": src, "W": W, "A": A, "lam": lam,
            "rms_affine": float(np.sqrt((res_aff ** 2).mean())),
            "rms_tps": float(np.sqrt((res_tps ** 2).mean())),
            "rms_tps_loo": float(best[1]) if best[0] is not None else float("nan"),
            "res_affine": res_aff, "res_tps": res_tps, "n": n}


def warp_field(fit, H, W_, res_um):
    g = TPS_GRID_PX
    ys = np.arange(0, H, g); xs = np.arange(0, W_, g)
    gy, gx = np.meshgrid(ys, xs, indexing="ij")
    pts = np.stack([gx.ravel() * res_um, gy.ravel() * res_um], -1).astype(float)
    M = fit["M"]
    out = (M[:, :2] @ pts.T).T + M[:, 2]

    for i in range(0, len(pts), 200000):
        out[i:i + 200000] += tps_eval(fit["src"], fit["W"], fit["A"], pts[i:i + 200000])
    fx = out[:, 0].reshape(gy.shape) / res_um; fy = out[:, 1].reshape(gy.shape) / res_um
    from scipy.ndimage import zoom as _zoom
    zy, zx = H / fx.shape[0], W_ / fx.shape[1]
    sx = _zoom(fx, (zy, zx), order=1)[:H, :W_]; sy = _zoom(fy, (zy, zx), order=1)[:H, :W_]
    return sx, sy


def affine_params(M):
    a, b, tx = M[0]; c, d, ty = M[1]
    sx = float(np.hypot(a, c)); rot = float(np.degrees(np.arctan2(c, a)))
    sh = float((a * b + c * d) / max(sx * sx, 1e-12))
    sy = float((a * d - b * c) / max(sx, 1e-12))
    return {"tx_um": tx, "ty_um": ty, "rot_deg": rot, "scale_x": sx, "scale_y": sy, "shear": sh}


def rgb(g, a, tint):
    out = np.zeros(g.shape + (3,), np.uint8)
    v = (np.clip(g, 0, 255)).astype(np.uint8)
    for k in range(3):
        out[..., k] = (v * tint[k]).astype(np.uint8)
    out[~a] = 0
    return out


def save_fig(im, path):
    w, h = im.size
    if w * h > FIG_MAX_MP:
        k = (w * h / FIG_MAX_MP) ** 0.5
        im = im.resize((int(w / k), int(h / k)), Image.LANCZOS)
    im.save(path)


def crop(im, box):
    return im.crop(tuple(box)) if box else im


def up(im):
    if FIG_SCALE == 1:
        return im
    return im.resize((im.width * FIG_SCALE, im.height * FIG_SCALE), Image.LANCZOS)


def label_font():
    try:
        from PIL import ImageFont
        return ImageFont.truetype(LABEL_FONT, LABEL_PT)
    except Exception:
        return None


def put_label(dr, xy, text, fill=(0, 0, 0)):
    f = label_font()
    if f is not None:
        dr.text(xy, text, fill=fill, font=f, stroke_width=2, stroke_fill=(255, 255, 255))
    else:
        dr.text(xy, text, fill=fill)


def highlight(g, a, feats, colour=OUTLINE_RGB, base=None):
    o = rgb(g, a, (1, 1, 1)) if base is None else base.copy()
    if base is not None:
        o[~a] = 0
    for typ, col in (("lumen", colour), ("dark", DARK_RGB)):
        sub = [f for f in feats if f.get("type", "lumen") == typ]
        if not sub:
            continue
        m = union(g.shape, sub)
        rim = binary_dilation(m, iterations=2 if base is not None else 1) & ~m
        o[rim] = col
    return Image.fromarray(o)


def lumen_key(vm, rec):
    import hashlib
    sid = rec["section_id"]
    d = {"res": RES_UM, "origin": vm.get("canvas_origin_um"), "canvas": vm.get("canvas_px"),
         "corr": (vm.get("section_corrections_applied") or {}).get("sections", {}).get(sid),
         "resid": (vm.get("arm_residual_applied") or {}).get(sid),
         "params": [MIN_AREA_UM2, MAX_AREA_UM2, EDGE_UM, CELL_FREE_UM, WHITE_MIN, WHITE_SAT_MAX, CLOSE_R_FRAC, HOLE_MAX_FRAC, SMOOTH_UM, SPLIT_MIN_SEP_UM]}
    return hashlib.sha1(json.dumps(d, sort_keys=True).encode()).hexdigest()


def save_lumens(path, feats, key):
    tab = np.array([(f["id"], f.get("type", "lumen"), f.get("label", str(f["id"])), f["cx_um"], f["cy_um"], f["area_um2"],
                     f["elong"], f["angle_deg"], f["circ"], f["sl"][0].start, f["sl"][0].stop, f["sl"][1].start, f["sl"][1].stop)
                    for f in feats],
                   dtype=[("id", "i4"), ("type", "U8"), ("label", "U16"), ("cx", "f8"), ("cy", "f8"), ("area", "f8"), ("elong", "f8"),
                          ("angle", "f8"), ("circ", "f8"), ("y0", "i4"), ("y1", "i4"), ("x0", "i4"), ("x1", "i4")])
    masks = {f"m{i}": np.packbits(f["m"], axis=None) for i, f in enumerate(feats)}
    np.savez_compressed(path, key=np.array(key), tab=tab, **masks)


def load_lumens(path, key):
    if not Path(path).exists():
        return None
    z = np.load(path, allow_pickle=False)
    if str(z["key"]) != key:
        return None
    feats = []
    for i, r in enumerate(z["tab"]):
        sl = (slice(int(r["y0"]), int(r["y1"])), slice(int(r["x0"]), int(r["x1"])))
        h, w = sl[0].stop - sl[0].start, sl[1].stop - sl[1].start
        m = np.unpackbits(z[f"m{i}"])[:h * w].reshape(h, w).astype(bool)
        feats.append({"id": int(r["id"]), "type": str(r["type"]), "label": str(r["label"]), "cx_um": float(r["cx"]), "cy_um": float(r["cy"]),
                      "area_um2": float(r["area"]), "elong": float(r["elong"]), "angle_deg": float(r["angle"]), "circ": float(r["circ"]),
                      "sl": sl, "m": m})
    return feats


def main():
    global RES_UM
    ap = argparse.ArgumentParser()
    ap.add_argument("--volume", required=True)
    ap.add_argument("--sections", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--res", type=float, default=RES_UM,
                    help="plane grid in um/px for the lumens and the figures (a tile level: 1 or 2). The alignment is a "
                         "coarse affine per pair; 2 um is enough for it and 4x cheaper than 1 um")
    ap.add_argument("--lumen-cache", default="", dest="lumen_cache",
                    help="directory of per-section lumen extractions (<sid>.npz); a section found there with a matching key is not re-extracted")
    ap.add_argument("--extract-only", action="store_true", dest="extract_only",
                    help="extract the lumens of the given sections into --lumen-cache and stop (no pairs, no figures)")
    ap.add_argument("--figures", default="1,2,3,4",
                    help="which figures to write; the pipeline passes 3,4 (1 and 2 are per section and repeat for every pair)")
    ap.add_argument("--legacy", action="store_true",
                    help="the earlier alignment for comparison: grid-selected anchors, TPS on every accepted "
                         "pair with leave-one-out lambda, no edge tier")
    a = ap.parse_args()
    RES_UM = float(a.res)
    FIGS = {s.strip() for s in a.figures.split(",")}
    vol, out = Path(a.volume), Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    vm = json.loads((vol / "metadata.json").read_text())
    global VM; VM = vm
    planes = {}
    for q in vm["encodings"]["G_withdrawn"]["planes"]:
        sid = q["section_id"]
        planes[sid.split("-")[-1].upper()] = q
    for q in vm["encodings"]["G_withdrawn"]["planes"]:
        sid = q["section_id"]
        planes.setdefault("U%d" % int(sid[len(sid.rstrip("0123456789")):]), q)
    want = [s.strip().upper() for s in a.sections.split(",")]
    rows = []
    PAIR_FITS = []
    P = {}
    BOX = None
    swaps = swap_files(vol)

    _pl = sorted(vm["encodings"]["G_withdrawn"]["planes"], key=lambda p: p["z_um"])
    _ious = []
    _prev = None
    for _p in _pl:
        _a = np.asarray(Image.open(vol / _p["webp_lossless"]).convert("RGBA"))[..., 3] > 0
        if _prev is not None:
            _u = float((_prev | _a).sum())
            _ious.append(float((_prev & _a).sum()) / _u if _u else 0.0)
        _prev = _a
    IOU_REF = float(np.median(_ious)) if _ious else 0.9
    IOU_STEP = max(0.03, IOU_REF - float(np.percentile(_ious, 25))) if _ious else 0.05
    print(f"adjacent-pair tissue overlap over the sample: median {IOU_REF:.3f}, 25th pct "
          f"{np.percentile(_ious, 25):.3f}, 5th pct {np.percentile(_ious, 5):.3f} ({len(_ious)} pairs)", flush=True)
    cache_dir = Path(a.lumen_cache) if a.lumen_cache else None
    if cache_dir:
        cache_dir.mkdir(parents=True, exist_ok=True)
    for u in want:
        rec = planes[u]
        ckey = lumen_key(vm, rec)
        cached = load_lumens(cache_dir / f"{rec['section_id']}.npz", ckey) if cache_dir else None
        _t = time.time()
        g, msk, src, srcp = load_plane(vol, rec, swaps)
        if not src.startswith("he"):
            raise SystemExit(f"{u}: no H&E for this {rec.get('modality')} section in the swap arm "
                             f"({srcp}); the features must come from H&E")
        print(f"  {u}: image = {src} ({srcp}, {g.shape[1]}x{g.shape[0]} px at {RES_UM:g} um)", flush=True)
        root, _src = tiles_root(vol, rec["section_id"], vm)
        rgbp, _ra = load_rgb_tiles(root, rec["section_id"], RES_UM) if (root is not None and not a.extract_only) else (None, None)
        if cached is not None:
            feats = cached
            print(f"  {u}: {len(feats)} lumens from the cache ({cache_dir / (rec['section_id'] + '.npz')})", flush=True)
        else:
            cm = cell_mask(vol, rec, g.shape, RES_UM)
            if rgbp is None and root is not None:
                rgbp, _ra = load_rgb_tiles(root, rec["section_id"], RES_UM)
            wm = None if rgbp is not None else white_mask(vol, rec, g.shape)
            print(f"  {u}: lumen signal = {'H&E colour at %g um' % RES_UM if rgbp is not None else 'grey + 4 um colour test'}", flush=True)
            _tl = time.time() - _t; _t = time.time()
            white, feats = ducts(g, msk, cm, wm, rgbp)
            print(f"    timing {u}: load {_tl:.0f} s, lumens {time.time() - _t:.0f} s", flush=True)
            if cache_dir:
                save_lumens(cache_dir / f"{rec['section_id']}.npz", feats, ckey)
        if a.extract_only:
            print(f"  {u}: {len(feats)} lumens cached", flush=True)
            continue
        P[u] = {"g": g, "a": msk, "f": feats, "rec": rec, "rgb": rgbp}
        ys, xs = np.nonzero(msk)
        if len(ys):
            pad = int(round(CROP_PAD_UM / RES_UM))
            b = [max(0, xs.min() - pad), max(0, ys.min() - pad),
                 min(msk.shape[1], xs.max() + pad + 1), min(msk.shape[0], ys.max() + pad + 1)]
            BOX = b if BOX is None else [min(BOX[0], b[0]), min(BOX[1], b[1]),
                                         max(BOX[2], b[2]), max(BOX[3], b[3])]
        print(f"  {u}: {len(feats)} ducts "
              f"(area {np.median([f['area_um2'] for f in feats]):.0f} um2 median)" if feats
              else f"  {u}: no ducts", flush=True)
    if a.extract_only:
        return
    for u in want:
        rec = P[u]["rec"]; g, msk, feats = P[u]["g"], P[u]["a"], P[u]["f"]

        base = P[u].get("rgb")
        if base is not None:
            base = base.copy(); base[~msk] = 0
        if "1" in FIGS:
            up(crop(Image.fromarray(rgb(g, msk, (1, 1, 1)) if base is None else base), BOX)).save(out / f"1_raw_{u}.png")
        if "2" in FIGS:
            im = up(crop(highlight(g, msk, feats, base=base), BOX)).convert("RGB")
            dr = ImageDraw.Draw(im)
            for f in feats:
                put_label(dr, ((f["cx_um"] / RES_UM - BOX[0]) * FIG_SCALE + 4,
                               (f["cy_um"] / RES_UM - BOX[1]) * FIG_SCALE - LABEL_PT // 2), f.get("label", str(f["id"])))
            im.save(out / f"2_features_{u}.png")
        with open(out / f"features_{u}.tsv", "w") as fh:
            fh.write("id\ttype\tcx_um\tcy_um\tarea_um2\telong\tangle_deg\tcirc\n")
            for f in feats:
                fh.write(f"{f['id']}\t{f.get('type', 'lumen')}\t{f['cx_um']:.1f}\t{f['cy_um']:.1f}\t{f['area_um2']:.0f}\t"
                         f"{f['elong']:.2f}\t{f['angle_deg']:.1f}\t{f['circ']:.3f}\n")
    for u0, u1 in zip(want, want[1:]):
        A, B = P[u0], P[u1]


        inter = float((A["a"] & B["a"]).sum()); uni = float((A["a"] | B["a"]).sum())
        iou0 = inter / uni if uni else 0.0
        dz = abs(float(B["rec"]["z_um"]) - float(A["rec"]["z_um"]))
        steps_z = max(0.0, (dz - ADAPT_Z0_UM) / 5.0)
        steps_iou = max(0.0, (IOU_REF - iou0) / IOU_STEP)
        tol = min(ADAPT_TOL_MAX, (1.0 + ADAPT_Z_STEP * steps_z) * (1.0 + ADAPT_IOU_STEP * steps_iou))
        relax = min(ADAPT_RELAX_MAX, ADAPT_RELAX_STEP * (steps_z + steps_iou))
        print(f"  {u0}->{u1}: initial tissue overlap IoU {iou0:.3f} (sample median {IOU_REF:.3f}, step {IOU_STEP:.3f}), "
              f"z gap {dz:.0f} um -> tolerances x{tol:.2f}, shape gates -{relax:.2f}", flush=True)
        _t = time.time()


        s_area, c_a, c_b = global_similarity(A["a"], B["a"], RES_UM)


        n_one = len(match(A["f"], B["f"], None, MATCH_MAX_UM, MATCH_IOU_R2))
        best = (n_one, 1.0)


        if n_one < SIZE_STEP_MIN_MATCHES:
            for s_try in [s_ for s_ in SIZE_STEP_RANGE if abs(s_ - s_area) <= SIZE_STEP_AREA_TOL]:
                n_try = len(match(scaled_features(A["f"], s_try, c_a, c_b), B["f"], None, MATCH_MAX_UM, MATCH_IOU_R2))
                if n_try > best[0] or (n_try == best[0] and abs(s_try - 1.0) < abs(best[1] - 1.0)):
                    best = (n_try, s_try)
        s_g = best[1] if best[0] >= 4 and best[0] > n_one else 1.0
        fa_s = scaled_features(A["f"], s_g, c_a, c_b)
        pos_a = {f["id"]: (f["cx_um"], f["cy_um"]) for f in A["f"]}
        def _restore(ms_):
            for m in ms_:
                m["pa"] = pos_a[m["a"]]
                m["dist_um"] = float(np.hypot(m["pa"][0] - m["pb"][0], m["pa"][1] - m["pb"][1]))
            return ms_
        print(f"    global size step: scale {s_g:.3f} ({best[0]} first-round matches; tissue area ratio {s_area:.3f}), "
              f"centroid shift ({c_b[0] - c_a[0]:+.0f}, {c_b[1] - c_a[1]:+.0f}) um", flush=True)
        core, M_glob = match_two_rounds(fa_s, B["f"])
        _restore(core)
        print(f"    timing {u0}->{u1}: matching {time.time() - _t:.0f} s", flush=True); _t = time.time()
        for m in core:
            m["role"] = "core"
        area_um2 = float(A["a"].sum()) * RES_UM ** 2


        if a.legacy:
            ms = select_anchors_grid(core, A["f"]) if M_glob is not None else core[:N_ANCHORS]
        else:
            ms = select_anchors(core, A["f"], area_um2) if M_glob is not None else core[:N_ANCHORS]
        print(f"    tiers: {min(len(ms), N_TIER1)} + {max(0, len(ms) - N_TIER1)} anchors, "
              f"tier-2 separation {getattr(select_anchors, 'sep_um', 0):.0f} um "
              f"(sqrt(area)/{SEP_DIV:g}, tissue {area_um2 / 1e6:.1f} mm2)", flush=True)


        escalated = None
        if len(ms) < 6 and not a.legacy:
            for tol_e, relax_e in SEARCH_ESCALATION:
                core_e, M_e = match_two_rounds(fa_s, B["f"], tol=tol_e, relax=relax_e)
                _restore(core_e)
                for m in core_e:
                    m["role"] = "core"
                ms_e = select_anchors(core_e, A["f"], area_um2) if M_e is not None else core_e[:N_ANCHORS]
                print(f"    search escalation x{tol_e:.2f} / gates -{relax_e:.2f}: {len(core_e)} matches -> "
                      f"{len(ms_e)} anchors", flush=True)
                if len(ms_e) >= 6:
                    core, M_glob, ms, escalated = core_e, M_e, ms_e, (tol_e, relax_e)
                    break
        edge_kept, edge_dropped = [], []
        if len(ms) >= 6 and not a.legacy:
            FTc = fit_affine_tps(ms)
            def pred_core(pts, FTc=FTc):
                M = FTc["M"]
                return (M[:, :2] @ pts.T).T + M[:, 2] + tps_eval(FTc["src"], FTc["W"], FTc["A"], pts)
            edge_kept, edge_dropped = edge_extension(A["f"], B["f"], ms, pred_core, tol, relax)
            print(f"    edge extension: {len(edge_kept)} kept, {len(edge_dropped)} dropped "
                  f"({', '.join(m['edge_label'] + ' ' + str(round(m['edge_resid_um'])) + 'um' for m in edge_dropped)})", flush=True)

        if len(ms) >= 6 and not a.legacy:
            FT3 = fit_affine_tps(ms + edge_kept, M_fixed=FTc["M"], lam_final=0.0)
            def pred_t3(pts, FT3=FT3):
                M = FT3["M"]
                return (M[:, :2] @ pts.T).T + M[:, 2] + tps_eval(FT3["src"], FT3["W"], FT3["A"], pts)
            k4, d4 = edge_extension(A["f"], B["f"], ms + edge_kept + edge_dropped, pred_t3, tol, relax,
                                    n_wedges=T4_WEDGES, outer_frac=T4_OUTER_FRAC, prefix="F")
            for m in k4:
                m["role"] = "edge_kept"
            for m in d4:
                m["role"] = "edge_dropped"
            print(f"    tier 4 (outermost band, same quadrants): {len(k4)} kept, {len(d4)} dropped", flush=True)
            edge_kept = edge_kept + k4; edge_dropped = edge_dropped + d4

            FT4 = fit_affine_tps(ms + edge_kept, M_fixed=FTc["M"], lam_final=0.0)
            def pred_t4(pts, FT4=FT4):
                M = FT4["M"]
                return (M[:, :2] @ pts.T).T + M[:, 2] + tps_eval(FT4["src"], FT4["W"], FT4["A"], pts)
            _cell, _wedge = _regions(A["f"], N_WEDGES, OUTER_FRAC)
            xs_ = np.array([f["cx_um"] for f in A["f"]]); ys_ = np.array([f["cy_um"] for f in A["f"]])
            c0 = (xs_.mean(), ys_.mean())
            min_r = {}
            for m in edge_kept:
                w = _wedge.whole(*m["pa"])
                min_r[w] = max(min_r.get(w, 0.0), float(np.hypot(m["pa"][0] - c0[0], m["pa"][1] - c0[1])) + T5_MARGIN_UM)
            k5, d5 = edge_extension(A["f"], B["f"], ms + edge_kept + edge_dropped, pred_t4, tol, relax,
                                    n_wedges=N_WEDGES, outer_frac=0.0, prefix="G", min_r=min_r)
            for m in k5:
                m["role"] = "edge_kept"
            for m in d5:
                m["role"] = "edge_dropped"
            print(f"    tier 5 (farther out than E/F in each quadrant): {len(k5)} kept, {len(d5)} dropped", flush=True)
            edge_kept = edge_kept + k5; edge_dropped = edge_dropped + d5
        allp = (core if a.legacy else ms + edge_kept)
        M_core = FTc["M"] if (len(ms) >= 6 and not a.legacy) else None
        R, t, rms = fit_rigid(ms)
        ang = float(np.degrees(np.arctan2(R[1, 0], R[0, 0])))
        PAIR_FITS.append({"a": A["rec"]["section_id"], "b": B["rec"]["section_id"],
                          "z_a": float(A["rec"]["z_um"]), "z_b": float(B["rec"]["z_um"]),
                          "initial_tissue_iou": round(iou0, 4), "tolerance_factor": tol,
                          "affine_a_to_b_um": (FTc["M"].tolist() if len(ms) >= 6 and not a.legacy else None),
                          "affine_rms_um": (float(FTc["rms_affine"]) if len(ms) >= 6 and not a.legacy else None),
                          "n_core": len(ms), "n_edge_kept": len(edge_kept),
                          "search_escalation": (list(escalated) if escalated else None),
                          "global_size_scale": round(s_g, 4),
                          "anchors": [{"label": (f"#{k + 1}" if k < len(ms) else m.get("edge_label", "")),
                                       "pa": list(m["pa"]), "pb": list(m["pb"]), "iou": m.get("iou")}
                                      for k, m in enumerate(ms + edge_kept)]})
        MA, rms_aff, _ = fit_affine(ms)
        ap_ = affine_params(MA) if MA is not None else {}


        FT = (fit_affine_tps(allp) if a.legacy else fit_affine_tps(allp, M_fixed=M_core, lam_final=0.0)) \
            if len(allp) >= 4 else None
        rows.append({"pair": f"{u0}->{u1}", "initial_tissue_iou": round(iou0, 3), "z_gap_um": round(dz, 1),
                     "tolerance_factor": tol, "n_ducts_a": len(A["f"]), "n_ducts_b": len(B["f"]),
                     "n_matches": len(ms), "dx_um": round(float(t[0]), 1), "dy_um": round(float(t[1]), 1),
                     "rot_deg": round(ang, 3), "rms_rigid_um": round(rms, 1),
                     "rms_affine_um": round(rms_aff, 1),
                     "affine_tx_um": round(ap_.get("tx_um", float("nan")), 1),
                     "affine_ty_um": round(ap_.get("ty_um", float("nan")), 1),
                     "affine_rot_deg": round(ap_.get("rot_deg", float("nan")), 3),
                     "affine_scale_x": round(ap_.get("scale_x", float("nan")), 4),
                     "affine_scale_y": round(ap_.get("scale_y", float("nan")), 4),
                     "affine_shear": round(ap_.get("shear", float("nan")), 4),
                     "n_pairs_all": len(allp),
                     "rms_affine_all_um": round(FT["rms_affine"], 1) if FT else None,
                     "rms_tps_um": round(FT["rms_tps"], 1) if FT else None,
                     "rms_tps_loo_um": round(FT["rms_tps_loo"], 1) if FT else None,
                     "tps_lambda_rel": FT["lam"] if FT else None,
                     "median_pair_dist_um": round(float(np.median([m["dist_um"] for m in ms])), 1) if ms else None})
        print(f"  {u0}->{u1}: {len(ms)} anchors | rigid shift ({t[0]:+.0f}, {t[1]:+.0f}) um rot {ang:+.2f} deg "
              f"rms {rms:.0f} um | affine rms {rms_aff:.0f} um, scale ({ap_.get('scale_x', 0):.3f}, "
              f"{ap_.get('scale_y', 0):.3f}) shear {ap_.get('shear', 0):+.3f}"
              + (f" | all {FT['n']} pairs: affine {FT['rms_affine']:.0f} um, TPS {FT['rms_tps']:.0f} um "
                 f"(leave-one-out {FT['rms_tps_loo']:.0f} um, lambda {FT['lam']:g})" if FT else ""), flush=True)

        H, W = A["g"].shape
        S_ = FIG_SCALE
        gap = 12 * S_
        CW, CH = (BOX[2] - BOX[0]), (BOX[3] - BOX[1])


        fa_by = {f["id"]: f for f in A["f"]}; fb_by = {f["id"]: f for f in B["f"]}
        left = np.array(highlight(A["g"], A["a"], A["f"], base=A.get("rgb")).convert("RGB"))
        right = np.array(highlight(B["g"], B["a"], B["f"], base=B.get("rgb")).convert("RGB"))
        def paint_rim(img, f, col):
            sl = f["sl"]; pad = PAIR_RIM_PX + 1
            y0, y1 = max(0, sl[0].start - pad), min(img.shape[0], sl[0].stop + pad)
            x0, x1 = max(0, sl[1].start - pad), min(img.shape[1], sl[1].stop + pad)
            big = np.zeros((y1 - y0, x1 - x0), bool)
            big[sl[0].start - y0:sl[0].start - y0 + f["m"].shape[0],
                sl[1].start - x0:sl[1].start - x0 + f["m"].shape[1]] = f["m"]
            rim = binary_dilation(big, iterations=PAIR_RIM_PX) & ~big
            img[y0:y1, x0:x1][rim] = col
        shown_all = [(f"#{k + 1}", (255, 40, 40) if not m.get("used", True) else CORE_RGB, m)
                     for k, m in enumerate(ms)]
        shown_all += [(m["edge_label"], {"F": EDGE4_RGB, "G": EDGE5_RGB}.get(m["edge_label"][0], EDGE_RGB), m) for m in edge_kept]
        shown_all += [(m["edge_label"], GREY, m) for m in edge_dropped]
        for lab, col, m in shown_all:
            paint_rim(left, fa_by[m["a"]], col)
            paint_rim(right, fb_by[m["b"]], col)
        M = MARGIN_PX
        canv = Image.new("RGB", (M + CW * S_ * 2 + gap + M, CH * S_), (12, 12, 14))
        canv.paste(up(crop(Image.fromarray(left), BOX)), (M, 0))
        canv.paste(up(crop(Image.fromarray(right), BOX)), (M + CW * S_ + gap, 0))
        dr = ImageDraw.Draw(canv)
        try:
            from PIL import ImageFont
            lead_font = ImageFont.truetype(LABEL_FONT, LEADER_PT)
        except Exception:
            lead_font = None
        def near_label(dr, x, y, text, col, font):
            L = LEADER_UM / RES_UM * S_
            x1, y1 = x + L * 0.8, y - L * 0.6
            dr.line([x, y, x1, y1], fill=col, width=3)
            if font is not None:
                tx, ty = x1 + 4, y1 - LEADER_PT // 2
                bb = dr.textbbox((tx, ty), text, font=font)
                dr.rectangle([bb[0] - 6, bb[1] - 4, bb[2] + 6, bb[3] + 4], fill=(0, 0, 0))
                dr.text((tx, ty), text, fill=col, font=font)
            else:
                dr.text((x1 + 4, y1), text, fill=col)
        shown = {(m["a"], m["b"]) for m in ms}
        with open(out / f"matches_{u0}_{u1}.tsv", "w") as fh:
            fh.write("a\tb\trole\tlabel\tiou\tdist_um\tresid_rigid_um\tresid_affine_um\tresid_tps_um\tshown\n")
            for i, m in enumerate(allp + edge_dropped):
                lab = m.get("edge_label", "")
                if not lab and (m["a"], m["b"]) in shown:
                    lab = f"#{[ (x['a'], x['b']) for x in ms ].index((m['a'], m['b'])) + 1}"
                fh.write(f"{m['a']}\t{m['b']}\t{m.get('role', 'core')}\t{lab}\t{m['iou']:.3f}\t{m['dist_um']:.1f}\t"
                         f"{m.get('resid_um', float('nan')):.1f}\t"
                         f"{FT['res_affine'][i] if (FT and i < len(allp)) else m.get('edge_resid_um', float('nan')):.1f}\t"
                         f"{FT['res_tps'][i] if (FT and i < len(allp)) else float('nan'):.1f}\t"
                         f"{int((m['a'], m['b']) in shown or m.get('role', '') == 'edge_kept')}\n")
        for lab, col, m in shown_all:
            xa = (m["pa"][0] / RES_UM - BOX[0]) * S_ + M; ya = (m["pa"][1] / RES_UM - BOX[1]) * S_
            xb = (m["pb"][0] / RES_UM - BOX[0]) * S_ + M + CW * S_ + gap
            yb = (m["pb"][1] / RES_UM - BOX[1]) * S_


            near_label(dr, xa, ya, f"{lab} {fa_by[m['a']].get('label', m['a'])}", col, lead_font)
            near_label(dr, xb, yb, f"{lab} {fb_by[m['b']].get('label', m['b'])}", col, lead_font)
        print(f"    timing {u0}->{u1}: tiers + fits + fig 3 {time.time() - _t:.0f} s", flush=True); _t = time.time()
        save_fig(canv, out / f"3_matches_{u0}_{u1}.png")


        fa_by = {f["id"]: f for f in A["f"]}; fb_by = {f["id"]: f for f in B["f"]}
        pairs_used = [(k, m) for k, m in enumerate(ms) if m.get("used", True)]
        ov_items = [(f"#{k + 1}", CORE_RGB, m) for k, m in pairs_used]
        ov_items += [(m["edge_label"], {"F": EDGE4_RGB, "G": EDGE5_RGB}.get(m["edge_label"][0], EDGE_RGB), m) for m in edge_kept]
        ov_items += [(m["edge_label"], GREY, m) for m in edge_dropped]
        masks_a = {i: union(A["g"].shape, [fa_by[m["a"]]]) for i, (_l, _c, m) in enumerate(ov_items)}
        masks_b = {i: union(B["g"].shape, [fb_by[m["b"]]]) for i, (_l, _c, m) in enumerate(ov_items)}


        def darkness(Pn):
            if Pn.get("rgb") is not None:
                d = 255.0 - Pn["rgb"].astype(np.float32).mean(2)
                d[~Pn["a"]] = 0
                return d
            return Pn["g"]
        gA, gB = darkness(A), darkness(B)

        def overlay(T23):


            if isinstance(T23, tuple):
                sx = np.clip(np.rint(T23[0]), 0, W - 1).astype(int)
                sy = np.clip(np.rint(T23[1]), 0, H - 1).astype(int)
            else:
                yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
                pu = np.stack([xx * RES_UM, yy * RES_UM], -1).reshape(-1, 2)
                q = (T23[:, :2] @ pu.T).T + T23[:, 2]
                sx = np.clip(q[:, 0] / RES_UM, 0, W - 1).reshape(H, W).astype(int)
                sy = np.clip(q[:, 1] / RES_UM, 0, H - 1).reshape(H, W).astype(int)
            gb = gB[sy, sx]
            ab = B["a"][sy, sx]
            o = np.zeros((H, W, 3), np.uint8)
            o[..., 0] = np.where(A["a"], gA, 0).astype(np.uint8)
            o[..., 1] = np.where(ab, gb, 0).astype(np.uint8)
            marks = []
            for i, (lab, col, m) in enumerate(ov_items):
                ma = masks_a[i]
                mb = masks_b[i][sy, sx]
                ra = binary_dilation(ma, iterations=PAIR_RIM_PX) & ~ma
                rb = binary_dilation(mb, iterations=PAIR_RIM_PX) & ~mb
                o[ra] = col; o[rb] = col

                ys_b, xs_b = np.nonzero(mb)
                marks.append((lab, col, (m["pa"][0] / RES_UM, m["pa"][1] / RES_UM),
                              (xs_b.mean(), ys_b.mean()) if len(xs_b) else None))
            return Image.fromarray(o), marks
        before, marks_before = overlay(np.hstack([np.eye(2), np.zeros((2, 1))]))

        if FT is not None:
            after, marks_after = overlay(warp_field(FT, H, W, RES_UM))
        elif MA is not None:
            M33 = np.vstack([MA, [0, 0, 1]])
            after, marks_after = overlay(np.linalg.inv(M33)[:2])
        else:
            Rinv = np.linalg.inv(R)
            after, marks_after = overlay(np.hstack([Rinv, (-Rinv @ t)[:, None]]))
        S_ = FIG_SCALE
        gap = 12 * S_
        both = Image.new("RGB", (CW * S_ * 2 + gap, CH * S_), (12, 12, 14))
        both.paste(up(crop(before, BOX)), (0, 0)); both.paste(up(crop(after, BOX)), (CW * S_ + gap, 0))
        dr = ImageDraw.Draw(both)
        try:
            from PIL import ImageFont
            lead_font4 = ImageFont.truetype(LABEL_FONT, LEADER_PT)
        except Exception:
            lead_font4 = None
        for panel, marks in ((0, marks_before), (1, marks_after)):
            ox = panel * (CW * S_ + gap)
            for lab, col, pa, pb in marks:
                x = (pa[0] - BOX[0]) * S_ + ox; y = (pa[1] - BOX[1]) * S_
                near_label(dr, x, y, lab, col, lead_font4)
        save_fig(both, out / f"4_overlay_{u0}_{u1}.png")
        print(f"    timing {u0}->{u1}: fig 4 {time.time() - _t:.0f} s", flush=True)
    (out / "crop_box.json").write_text(json.dumps({"box_px": [int(v) for v in BOX], "grid_um": RES_UM,
                                                   "fig_scale": FIG_SCALE}))
    _applied = vm.get("section_corrections_applied") or None
    for _r in PAIR_FITS:
        _r["measured_with_corrections"] = {"sections": _applied["sections"]} if _applied else None
    (out / "pair_fits.json").write_text(json.dumps(PAIR_FITS, indent=1))
    import csv
    with open(out / "duct_anchor_fits.tsv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()), delimiter="\t")
        w.writeheader(); w.writerows(rows)
    print(f"wrote {out}", flush=True)


if __name__ == "__main__":
    main()
