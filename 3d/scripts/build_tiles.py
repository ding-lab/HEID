#!/usr/bin/env python

from __future__ import annotations

import argparse
import csv
import os
import importlib.util as _iu
import json
import os as _os
import time
from pathlib import Path

import cv2
import numpy as np
import tifffile
from PIL import Image

ROOT = Path(os.environ.get("THREED_ROOT", os.path.join(os.environ.get("PROJECTS_ROOT", "/data/heid"), "3d")))
RUNS = ROOT / "registration/runs"


SEC = RUNS / "prep_sections/sections"
MANIFEST = RUNS / "prep_sections/manifest.tsv"
EDGE_RUNS = ("pair_fits",)
BASE8 = ROOT / "reconstruction/volume_8um"
OUT = ROOT / "reconstruction/volume_tiles"
ANCHOR = "HT891Z1-U66"

WG_MPP = 2.0
BASE_MPP = 8.0
LEVELS = [4.0, 2.0, 1.0, 0.5]
TILE = 512
QUALITY = 90
BASES = {"G_withdrawn": "R", "G_not_withdrawn": "M"}


def _load(name, path):
    s = _iu.spec_from_file_location(name, path)
    m = _iu.module_from_spec(s)
    s.loader.exec_module(m)
    return m


CHAIN = _load("placement_chain", Path(__file__).resolve().parent / "placement_chain.py")


def _plain(o):
    return o.item() if hasattr(o, "item") else str(o)


def manifest(path=None):
    man = {r["section_id"]: r for r in
           csv.DictReader(open(path or MANIFEST), delimiter="\t")}
    sids = sorted(man, key=lambda s: float(man[s]["z_position_um"]))
    return man, sids


CHAIN_MANIFEST = ""


def chains(sids):
    if CHAIN_MANIFEST:
        _, sids = manifest(CHAIN_MANIFEST)
    edges = CHAIN.harvest(EDGE_RUNS)
    tree = CHAIN.prereg_tree(edges, sids)
    return {k: CHAIN.chain_on(tree, edges, b, ANCHOR) for k, b in BASES.items()}


def read_level(sid, src_mpp, out_mpp):
    img = tifffile.imread(str(SEC / sid / f"img_mpp{src_mpp:g}.tif"))
    if img.ndim == 3:
        img = img[..., 0]
    msk = cv2.imread(str(SEC / sid / f"mask_mpp{src_mpp:g}.png"),
                     cv2.IMREAD_GRAYSCALE) > 127
    f = src_mpp / out_mpp
    if abs(f - 1.0) > 1e-9:
        size = (max(1, int(round(img.shape[1] * f))), max(1, int(round(img.shape[0] * f))))
        img = cv2.resize(img, size, interpolation=cv2.INTER_AREA)
        msk = cv2.resize(msk.astype(np.uint8), size, interpolation=cv2.INTER_AREA) > 0
    return img, msk


def load_index(basis, sid):
    p = OUT / basis / sid / "index.json"
    try:
        return json.loads(p.read_text())
    except Exception:
        return None


def cached_window(sid):
    for basis in BASES:
        idx = load_index(basis, sid)
        c = (idx or {}).get("contrast")
        if c and c.get("lo") is not None and c.get("deployed_8um_window"):
            return (float(c["lo"]), float(c["hi"])), tuple(c["deployed_8um_window"])
    return None


def window_at(sid, mpp):
    src = 0.5 if mpp < 1.0 else 1.0
    img, msk = read_level(sid, src, mpp)
    v = img[msk]
    if v.size > 100:
        lo, hi = np.percentile(v, [1.0, 99.0])
        if hi > lo:
            return float(lo), float(hi)
    return None


def level_shape(sid, m):
    src = 0.5 if m < 1.0 else 1.0
    with tifffile.TiffFile(str(SEC / sid / f"img_mpp{src:g}.tif")) as tf:
        h, w = tf.pages[0].shape[:2]
    f = src / m
    if abs(f - 1.0) < 1e-9:
        return h, w
    return max(1, int(round(h * f))), max(1, int(round(w * f)))


def source_slide(sid):
    return Path(json.loads((SEC / sid / "meta.json").read_text())["source_filepath"])


_OME = {}


def ome_level(sid, m):
    key = sid
    if key not in _OME:
        with tifffile.TiffFile(str(source_slide(sid))) as tf:
            s = tf.series[0]
            shapes = [l.shape for l in s.levels]
        native = float(json.loads((SEC / sid / "meta.json").read_text())["mpp_native"])
        _OME[key] = {"shapes": shapes, "native": native}
    info = _OME[key]
    h0 = info["shapes"][0][0]
    best, best_mpp = 0, info["native"]
    for i, sh in enumerate(info["shapes"]):
        mpp = info["native"] * (h0 / sh[0])
        if mpp <= m + 1e-6 and mpp >= best_mpp:
            best, best_mpp = i, mpp
    return best, best_mpp


def read_rgb_level(sid, m):
    th, tw = level_shape(sid, m)
    if m >= 4.0 - 1e-9:
        img = tifffile.imread(str(SEC / sid / "rgb_mpp4.tif"))
    else:
        lvl, _ = ome_level(sid, m)
        img = tifffile.imread(str(source_slide(sid)), level=lvl)
    if img.ndim == 2:
        img = np.dstack([img] * 3)
    img = img[..., :3]
    if img.shape[0] != th or img.shape[1] != tw:
        shrink = (th * tw) < (img.shape[0] * img.shape[1])
        img = cv2.resize(img, (tw, th),
                         interpolation=cv2.INTER_AREA if shrink else cv2.INTER_LINEAR)
    return img


def warp_affine_big(img, M, wh, flags, borderValue):
    W, H = wh
    LIM = 32000
    if (W < LIM and H < LIM and img.shape[0] < LIM and img.shape[1] < LIM):
        return cv2.warpAffine(img, M, (W, H), flags=flags, borderValue=borderValue)
    shape = (H, W) if img.ndim == 2 else (H, W, img.shape[2])
    fill = borderValue if np.isscalar(borderValue) else borderValue[0]
    out = np.full(shape, fill, img.dtype)
    Minv = cv2.invertAffineTransform(M)
    T = 15000
    for y0 in range(0, H, T):
        for x0 in range(0, W, T):
            w = min(T, W - x0); h = min(T, H - y0)
            c = np.array([[x0, y0], [x0 + w, y0], [x0, y0 + h], [x0 + w, y0 + h]],
                         float)
            sc = c @ Minv[:2, :2].T + Minv[:, 2]
            sx0 = max(0, int(np.floor(sc[:, 0].min())) - 2)
            sx1 = min(img.shape[1], int(np.ceil(sc[:, 0].max())) + 2)
            sy0 = max(0, int(np.floor(sc[:, 1].min())) - 2)
            sy1 = min(img.shape[0], int(np.ceil(sc[:, 1].max())) + 2)
            if sx1 <= sx0 or sy1 <= sy0:
                continue
            Mt = M.copy()
            Mt[:, 2] = (M[:, 2] + M[:2, :2] @ np.array([sx0, sy0], float)
                        - np.array([x0, y0], float))
            out[y0:y0 + h, x0:x0 + w] = cv2.warpAffine(
                img[sy0:sy1, sx0:sx1], Mt, (w, h), flags=flags,
                borderValue=borderValue)
    return out


def read_mask_level(sid, m):
    src = 0.5 if m < 1.0 else 1.0
    msk = cv2.imread(str(SEC / sid / f"mask_mpp{src:g}.png"), cv2.IMREAD_GRAYSCALE) > 127
    th, tw = level_shape(sid, m)
    if msk.shape[0] != th or msk.shape[1] != tw:
        msk = cv2.resize(msk.astype(np.uint8), (tw, th),
                         interpolation=cv2.INTER_AREA) > 0
    return msk


def warp_rgb_level(sid, ch_b, m, O, W8, H8):
    img = read_rgb_level(sid, m)
    msk = read_mask_level(sid, m)
    img = img * msk[..., None]
    k = BASE_MPP / m
    W, H = int(round(W8 * k)), int(round(H8 * k))
    M = np.zeros((2, 3))
    M[:2, :2] = ch_b[:2, :2]
    M[:, 2] = (WG_MPP / m) * ch_b[:, 2] - O / m
    rgb = warp_affine_big(img, M, (W, H), cv2.INTER_LINEAR, 0)
    a = warp_affine_big(msk.astype(np.uint8) * 255, M, (W, H),
                        cv2.INTER_NEAREST, 0)
    return rgb, a


WEBP_METHOD = int(_os.environ.get("TILE_WEBP_METHOD", "2"))
TILE_WORKERS = int(_os.environ.get("TILE_WORKERS", "0")) or max(1, (_os.cpu_count() or 2))


def _encode_one(args):
    path, arr = args
    import io
    b = io.BytesIO()
    Image.fromarray(arr).save(b, format="WEBP", quality=QUALITY, method=WEBP_METHOD)
    data = b.getvalue()
    with open(path, "wb") as f:
        f.write(data)
    return len(data)


def _write_tiles(make_rgba, a, d):
    d.mkdir(parents=True, exist_ok=True)
    H, W = a.shape
    cols = int(np.ceil(W / TILE))
    rows = int(np.ceil(H / TILE))
    jobs, present, empty = [], [], 0
    for r in range(rows):
        y0, y1 = r * TILE, min(H, (r + 1) * TILE)
        arow = a[y0:y1]
        if not arow.any():
            empty += cols
            continue
        for c in range(cols):
            x0, x1 = c * TILE, min(W, (c + 1) * TILE)
            aa = arow[:, x0:x1]
            if not aa.any():
                empty += 1
                continue
            jobs.append((str(d / f"{c}_{r}.webp"), make_rgba(y0, y1, x0, x1, aa)))
            present.append(f"{c}_{r}")
    nbytes = 0
    if jobs:
        if len(jobs) >= 8 and TILE_WORKERS > 1:
            import multiprocessing as _mp
            with _mp.get_context("fork").Pool(min(TILE_WORKERS, len(jobs))) as pool:
                nbytes = int(sum(pool.imap_unordered(_encode_one, jobs, chunksize=8)))
        else:
            nbytes = int(sum(_encode_one(j) for j in jobs))
    return {"cols": cols, "rows": rows, "width": W, "height": H,
            "tiles_written": len(present), "tiles_empty": empty,
            "bytes": int(nbytes), "present": present}


def write_rgb_tiles(rgb, a, d):
    return _write_tiles(lambda y0, y1, x0, x1, aa: np.dstack([rgb[y0:y1, x0:x1], aa]), a, d)


def contrast_window(sid):
    hit = cached_window(sid)
    if hit:
        return hit
    fine = window_at(sid, min(LEVELS))
    base = window_at(sid, BASE_MPP)
    return fine, base


def apply_window(img, msk, win):
    if win is None:
        return (img * msk).astype(np.uint8), 0.0
    lo, hi = win
    f = img.astype(np.float32)
    inside = f[msk]
    clipped = float(np.mean((inside < lo) | (inside > hi))) if inside.size else 0.0
    f = np.clip((f - lo) * (255.0 / (hi - lo)), 0, 255)
    return (f * msk).astype(np.uint8), clipped


def geometry(sids, ch):
    shp = {}
    for s in sids:
        with tifffile.TiffFile(str(SEC / s / "img_mpp1.tif")) as tf:
            sh = tf.pages[0].shape
        h1, w1 = sh[0], sh[1]
        f = 1.0 / BASE_MPP
        shp[s] = (max(1, int(round(h1 * f))), max(1, int(round(w1 * f))), h1, w1)

    meta = json.loads((BASE8 / "metadata.json").read_text())
    if meta.get("canvas_origin_um"):


        O = np.asarray(meta["canvas_origin_um"], float)
        W8, H8 = meta["canvas_px"]["width"], meta["canvas_px"]["height"]
        print(f"canvas from the volume: {W8} x {H8} at origin {O[0]:.1f}, {O[1]:.1f} um", flush=True)
        return O, W8, H8, shp
    pts = []
    for c_ in ch.values():
        for s in sids:
            h, w = shp[s][0], shp[s][1]
            c = np.array([[0, 0], [w, 0], [w, h], [0, h]], float) * (BASE_MPP / WG_MPP)
            pts.append((c @ c_[s][:2, :2].T + c_[s][:, 2]) * WG_MPP)
    pts = np.vstack(pts)
    o = pts.min(0) - 40.0
    W = int(np.ceil((pts[:, 0].max() + 40.0 - o[0]) / BASE_MPP))
    H = int(np.ceil((pts[:, 1].max() + 40.0 - o[1]) / BASE_MPP))


    anym = np.zeros((H, W), bool)
    for bk, c_ in ch.items():
        for s in sids:
            msk = cv2.imread(str(SEC / s / "mask_mpp1.png"), cv2.IMREAD_GRAYSCALE) > 127
            hh, ww = shp[s][0], shp[s][1]
            msk = cv2.resize(msk.astype(np.uint8), (ww, hh),
                             interpolation=cv2.INTER_AREA) > 0
            M = np.zeros((2, 3))
            M[:2, :2] = c_[s][:2, :2]
            M[:, 2] = (WG_MPP / BASE_MPP) * c_[s][:, 2] - o / BASE_MPP
            a = cv2.warpAffine(msk.astype(np.uint8) * 255, M, (W, H),
                               flags=cv2.INTER_NEAREST, borderValue=0)
            anym |= a > 0
    ys, xs = np.nonzero(anym)
    p = int(round(80.0 / BASE_MPP))
    r0, r1 = max(0, ys.min() - p), min(H, ys.max() + p + 1)
    c0, c1 = max(0, xs.min() - p), min(W, xs.max() + p + 1)
    W8, H8 = int(c1 - c0), int(r1 - r0)
    O = o + np.array([c0, r0], float) * BASE_MPP

    meta = json.loads((BASE8 / "metadata.json").read_text())
    want = (meta["canvas_px"]["width"], meta["canvas_px"]["height"])
    if meta.get("canvas_origin_um"):

        O = np.asarray(meta["canvas_origin_um"], float)
        W8, H8 = want
        print(f"canvas from the volume: {W8} x {H8} at origin {O[0]:.1f}, {O[1]:.1f} um", flush=True)
        return O, W8, H8, shp
    assert (W8, H8) == want, (
        f"pinning failed: recomputed 8 um canvas {W8}x{H8}, the deployed volume is "
        f"{want[0]}x{want[1]}. Every tile would be offset; nothing was written.")
    return O, W8, H8, shp


def warp_level(sid, ch_b, win, m, O, W8, H8):
    src_mpp = 0.5 if m < 1.0 else 1.0
    img, msk = read_level(sid, src_mpp, m)
    img, clipped = apply_window(img, msk, win)
    k = BASE_MPP / m
    W, H = int(round(W8 * k)), int(round(H8 * k))
    M = np.zeros((2, 3))
    M[:2, :2] = ch_b[:2, :2]
    M[:, 2] = (WG_MPP / m) * ch_b[:, 2] - O / m
    g = warp_affine_big(img, M, (W, H), cv2.INTER_LINEAR, 0)
    a = warp_affine_big(msk.astype(np.uint8) * 255, M, (W, H),
                        cv2.INTER_NEAREST, 0)
    return g, a, clipped, src_mpp


def write_tiles(g, a, d):
    def mk(y0, y1, x0, x1, aa):
        gg = g[y0:y1, x0:x1]
        return np.dstack([gg, gg, gg, aa])
    return _write_tiles(mk, a, d)


def qc_plane(sid, basis, idx, O, W8, H8):
    lv = [l for l in idx["levels"] if abs(l["mpp"] - 0.5) < 1e-9][0]
    W, H = lv["width"], lv["height"]
    canvas = np.zeros((H, W), np.uint8)
    alpha = np.zeros((H, W), np.uint8)
    d = OUT / basis / sid / "0.5"
    for name in lv["present"]:
        c, r = (int(x) for x in name.split("_"))
        im = np.array(Image.open(d / f"{name}.webp").convert("RGBA"))
        y0, x0 = r * TILE, c * TILE
        canvas[y0:y0 + im.shape[0], x0:x0 + im.shape[1]] = im[..., 0]
        alpha[y0:y0 + im.shape[0], x0:x0 + im.shape[1]] = im[..., 3]
    k = int(round(BASE_MPP / 0.5))
    assert W == W8 * k and H == H8 * k, f"grid {W}x{H} is not {k}x the 8 um canvas"
    down = cv2.resize(canvas, (W8, H8), interpolation=cv2.INTER_AREA)
    downa = cv2.resize(alpha, (W8, H8), interpolation=cv2.INTER_AREA)

    meta = json.loads((BASE8 / "metadata.json").read_text())
    pl = [p for p in meta["encodings"][basis]["planes"] if p["section_id"] == sid][0]
    ref = np.array(Image.open(BASE8 / pl["png"]).convert("RGBA"))
    refg, refa = ref[..., 0], ref[..., 3]

    m = (refa > 0) | (downa > 0)
    x = down[m].astype(np.float64)
    y = refg[m].astype(np.float64)
    r = float(np.corrcoef(x, y)[0, 1])
    dif = np.abs(x - y)


    A = np.vstack([x, np.ones_like(x)]).T
    gain, off = np.linalg.lstsq(A, y, rcond=None)[0]
    dif_fit = np.abs(gain * x + off - y)
    shift, resp = cv2.phaseCorrelate(np.float32(down) * (refa > 0),
                                     np.float32(refg) * (refa > 0))
    inter = float((((refa > 0) & (downa > 0)).sum())
                  / max(1, ((refa > 0) | (downa > 0)).sum()))
    return {"section_id": sid, "basis": basis,
            "shift_px_at_8um": [round(float(shift[0]), 4), round(float(shift[1]), 4)],
            "shift_um": [round(float(shift[0]) * BASE_MPP, 3),
                         round(float(shift[1]) * BASE_MPP, 3)],
            "phase_correlation_response": round(float(resp), 4),
            "pearson_r": round(r, 6),
            "mean_abs_diff_grey": round(float(dif.mean()), 3),
            "mean_abs_diff_after_gain_fit": round(float(dif_fit.mean()), 3),
            "gain_fit": [round(float(gain), 4), round(float(off), 3)],
            "p99_abs_diff_grey": round(float(np.percentile(dif, 99)), 3),
            "max_abs_diff_grey": round(float(dif.max()), 1),
            "tissue_mask_iou": round(inter, 6),
            "compared_px": int(m.sum())}


def main():
    global ANCHOR, BASE8
    global SEC, MANIFEST, EDGE_RUNS, BASE8, OUT, LEVELS, CHAIN_MANIFEST
    ap = argparse.ArgumentParser()
    ap.add_argument("--planes", default="HT891Z1-U66",
                    help="comma separated section ids, or 'all'")
    ap.add_argument("--plane-index", type=int, default=-1,
                    help="one plane by its z order (0-49). For a SLURM array, where "
                         "the task id is the only thing the job knows about itself.")
    ap.add_argument("--bases", default=",".join(BASES))
    ap.add_argument("--levels", default=",".join(f"{l:g}" for l in LEVELS))
    ap.add_argument("--rgb", action="store_true",
                    help="build the H&E colour ladder instead of the grey one, on the "
                         "same grid. Non-H&E planes are skipped: they are single "
                         "channel and get their colour from a tint at draw time.")
    ap.add_argument("--qc", action="store_true",
                    help="stitch the finest level back and compare with volume_8um")
    ap.add_argument("--qc-only", action="store_true",
                    help="do not build anything: read the tiles already on disk and "
                         "run that same comparison. This is how a NON-ANCHOR plane "
                         "gets checked - the anchor's affine is the identity in both "
                         "bases, so checking only the anchor leaves the transform "
                         "itself untested.")
    ap.add_argument("--report", default="")
    ap.add_argument("--volume", default=str(BASE8),
                    help="the deployed volume the tiles are pinned against")
    ap.add_argument("--anchor", default=ANCHOR,
                    help="must match the anchor the volume was built with")

    ap.add_argument("--prep-root", default=str(SEC))
    ap.add_argument("--manifest", default=str(MANIFEST))
    ap.add_argument("--chain-manifest", default="", dest="chain_manifest", help="section set the chain is solved over (a swap arm passes the main manifest)")
    ap.add_argument("--edge-runs", default=",".join(EDGE_RUNS),
                    help="run directories under registration/runs holding the pairwise "
                         "results, EARLIEST WINNING LAST: harvest keeps the first "
                         "result it sees for a pair, so a re-solved run must be "
                         "listed before the run it supersedes")
    ap.add_argument("--base8", default=str(BASE8),
                    help="the 8 um volume this pyramid is pinned to and asserted against")
    ap.add_argument("--out", default=str(OUT))
    ap.add_argument("--threads", type=int, default=0,
                    help="cv2 thread cap. 0 leaves OpenCV alone; 1 is what a SLURM "
                         "array wants, because OpenCV sizes its pool from the NODE's "
                         "core count and several tasks on one node then contend for "
                         "the few cores each was actually given.")
    a = ap.parse_args()

    SEC = Path(a.prep_root)
    MANIFEST = Path(a.manifest)
    CHAIN_MANIFEST = a.chain_manifest
    EDGE_RUNS = tuple(s.strip() for s in a.edge_runs.split(",") if s.strip())
    BASE8 = Path(a.base8)
    OUT = Path(a.out)
    ANCHOR = a.anchor
    BASE8 = Path(a.volume)
    if a.threads:
        cv2.setNumThreads(a.threads)
    print(f"prep={SEC}\nmanifest={MANIFEST}\nedges={EDGE_RUNS}\nbase8={BASE8}\nout={OUT}",
          flush=True)

    t0 = time.time()
    man, sids = manifest()
    ch = chains(sids)
    O, W8, H8, shp = geometry(sids, ch)
    print(f"pinned to the 8 um canvas {W8} x {H8} px, world origin "
          f"({O[0]:.3f}, {O[1]:.3f}) um   [{time.time()-t0:.1f}s]", flush=True)

    if a.plane_index >= 0:
        planes = [sids[a.plane_index]]
    elif a.planes == "all":
        planes = sids
    else:
        planes = [s.strip() for s in a.planes.split(",")]
    levels = [float(x) for x in a.levels.split(",")]


    LEVELS = sorted(levels, reverse=True)
    bases = [b.strip() for b in a.bases.split(",")]
    report = {"canvas_8um_px": [W8, H8], "world_origin_um": [float(O[0]), float(O[1])],
              "tile_px": TILE, "webp_quality": QUALITY, "webp_method": WEBP_METHOD, "levels": levels,
              "planes": [], "qc": []}

    if a.qc_only:
        for sid in planes:
            for basis in bases:
                idx = json.loads((OUT / basis / sid / "index.json").read_text())
                A = ch[basis][sid][:2, :2]
                t = ch[basis][sid][:, 2]
                q = qc_plane(sid, basis, idx, O, W8, H8)


                q["affine_linear_part"] = [[round(float(v), 6) for v in row] for row in A]
                q["is_chain_anchor"] = bool(np.allclose(A, np.eye(2)))
                report["qc"].append(q)
                print(f"  QC {sid} {basis}: shift {q['shift_um']} um, r {q['pearson_r']}, "
                      f"mean|d| {q['mean_abs_diff_grey']}, anchor {q['is_chain_anchor']}",
                      flush=True)
        rp = Path(a.report) if a.report else (OUT / "qc_only_report.json")
        rp.write_text(json.dumps(report, indent=1, default=_plain))
        print(f"\nwrote {rp}  [{time.time()-t0:.1f}s]")
        return

    for sid in planes:
        if a.rgb and man[sid]["modality"] != "he":
            print(f"  {sid}: {man[sid]['modality']}, no colour pyramid (the two DAPI "
                  f"modalities are one channel and are tinted at draw time)", flush=True)
            continue
        win, win8 = (None, None) if a.rgb else contrast_window(sid)


        step = (None if (win is None or win8 is None) else
                round(((win8[1] - win8[0]) / (win[1] - win[0])), 4))
        i = sids.index(sid)
        for basis in bases:
            t1 = time.time()
            entry = {"section_id": sid, "index": i, "basis": basis,
                     "modality": man[sid]["modality"],
                     "z_um": float(man[sid]["z_position_um"]),
                     "tile_px": TILE, "webp_quality": QUALITY,
                     "world_origin_um": [float(O[0]), float(O[1])],
                     "canvas_8um_px": [W8, H8],
                     "contrast": None if win is None else
                                 {"lo": win[0], "hi": win[1],


                                  "measured_on_mpp": min(LEVELS),
                                  "rule": "1st/99th percentile of the tissue at the "
                                          "FINEST level, shared by every level so a "
                                          "zoom cannot change the values",
                                  "deployed_8um_window": None if win8 is None else
                                                         [win8[0], win8[1]],
                                  "gain_ratio_vs_deployed_8um": step},


                     "levels": [], "rgb_levels": []}
            for m in levels:
                if a.rgb:
                    rgb, al = warp_rgb_level(sid, ch[basis][sid], m, O, W8, H8)
                    info = write_rgb_tiles(rgb, al, OUT / basis / sid / "rgb" / f"{m:g}")
                    lvl, lvl_mpp = ome_level(sid, m)
                    info.update({"mpp": m,
                                 "source": ("prep rgb_mpp4.tif" if m >= 4 else
                                            f"source slide pyramid level {lvl} "
                                            f"({lvl_mpp:.4f} um/px)")})
                    entry["rgb_levels"].append(info)
                    del rgb, al
                else:
                    g, al, clipped, src_mpp = warp_level(sid, ch[basis][sid], win, m,
                                                         O, W8, H8)
                    info = write_tiles(g, al, OUT / basis / sid / f"{m:g}")
                    info.update({"mpp": m, "source_level_mpp": src_mpp,
                                 "clipped_tissue_fraction": round(clipped, 6)})
                    entry["levels"].append(info)
                    del g, al
                print(f"  {sid} {basis} {'rgb ' if a.rgb else ''}{m:g} um/px  "
                      f"{info['width']}x{info['height']} px "
                      f"-> {info['tiles_written']} tiles ({info['tiles_empty']} empty), "
                      f"{info['bytes']/1e6:.2f} MB  [{time.time()-t1:.1f}s]", flush=True)
            d = OUT / basis / sid
            d.mkdir(parents=True, exist_ok=True)


            old = load_index(basis, sid)


            if old and (not np.allclose(old.get("world_origin_um", [None, None]), entry["world_origin_um"], atol=1e-6)
                        or list(old.get("canvas_8um_px", [])) != list(entry["canvas_8um_px"])):
                print(f"  {sid} {basis}: canvas changed, levels of the earlier placement dropped from the index "
                      f"({[L['mpp'] for L in old.get('levels', [])]} grey, {[L['mpp'] for L in old.get('rgb_levels', [])]} colour)", flush=True)
                old = None
            if old:
                for key in ("levels", "rgb_levels"):
                    keep = [L for L in old.get(key, [])
                            if all(abs(L["mpp"] - n["mpp"]) > 1e-9 for n in entry[key])]
                    entry[key] = sorted(entry[key] + keep, key=lambda L: -L["mpp"])
                if win is None and old.get("contrast"):
                    entry["contrast"] = old["contrast"]
            (d / "index.json").write_text(json.dumps(entry, indent=1, default=_plain))
            report["planes"].append({k: v for k, v in entry.items() if k != "levels"}
                                    | {"levels": [{k: v for k, v in L.items()
                                                   if k != "present"}
                                                  for L in entry["levels"]]})
            if a.qc:
                q = qc_plane(sid, basis, entry, O, W8, H8)
                report["qc"].append(q)
                print(f"  QC {sid} {basis}: shift {q['shift_um']} um, r {q['pearson_r']}, "
                      f"mean|d| {q['mean_abs_diff_grey']} grey levels", flush=True)


    if all("area_mm2_mpp1" in man[s] for s in sids):
        areas = {s: float(man[s]["area_mm2_mpp1"]) for s in sids}
        for p in report["planes"]:
            base = areas[p["section_id"]]
            tot = sum(L["bytes"] for L in p["levels"])
            p["bytes_total"] = int(tot)
            p["projection_all_50_planes_2_bases_GB"] = round(
                sum(areas[s] / base for s in sids) * tot * len(BASES) / 1e9, 2)
    report["elapsed_s"] = round(time.time() - t0, 1)
    OUT.mkdir(parents=True, exist_ok=True)
    rp = Path(a.report) if a.report else (OUT / "phase1_report.json")

    rp.write_text(json.dumps(report, indent=1, default=_plain))
    print(f"\nwrote {rp}  [{report['elapsed_s']}s]")


if __name__ == "__main__":
    main()
