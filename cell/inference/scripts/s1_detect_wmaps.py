#!/usr/bin/env python -u
import os
PROJECTS_ROOT = os.environ.get("PROJECTS_ROOT", "/data/heid")
from pathlib import Path
_RELEASE = Path(__file__).resolve().parents[3]

import argparse
import hashlib
import json
import math
import os
import platform
import re
import sys
import time
import warnings
from concurrent.futures import ProcessPoolExecutor, FIRST_COMPLETED, wait

import cv2
import numpy as np
import pandas as pd
import tifffile
from scipy.spatial import cKDTree
from scipy.ndimage import find_objects

warnings.filterwarnings("ignore")

sys.path.insert(0, str(_RELEASE / "cell/inference/scripts"))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import slide_canvas as L
import instanseg_lib as SL
from band_reader import BandCachedSlide

BAND_ROWS = 8192

INFERENCE_ROOT = PROJECTS_ROOT + "/cell/inference"
COHORT = os.environ.get("CELL_COHORT", "tcga_pancancer")
CELLS_DIR = os.path.join(INFERENCE_ROOT, COHORT, "data", "cells")
WMAP_DIR = os.path.join(INFERENCE_ROOT, COHORT, "data", "wmaps")
LUNG_CKPT = PROJECTS_ROOT + "/cell/models/detection/instanseg_lung.pt"


POST = dict(seed_threshold=0.10, mask_threshold=0.35, peak_distance=3, min_size=6, max_seeds=8000)
TILE, OVERLAP, MIN_TISSUE = 512, 64, 0.02
MIN_TILE_PX = 64


HALO = 64


XPX = 0.2125
CROP_HALF = 112
BLANK_THR = 0.9

def log(m):
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


def sha256_of(path, limit=None):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        n = 0
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
            n += len(chunk)
            if limit is not None and n >= limit:
                break
    return h.hexdigest()


def read_native_mpp(svs):
    with tifffile.TiffFile(str(svs)) as tif:
        desc = tif.pages[0].description or ""
    m = re.search(r"MPP\s*=\s*([0-9.]+)", desc)
    if not m:
        raise ValueError(f"no MPP in SVS desc: {desc[:200]!r}")
    return float(m.group(1))


GRID = 16
PATCH = 14
FILL_SHIFT = 4
FILL_SCALE = 1 << FILL_SHIFT
SIGMA_PX = 3.0
QUANT_SCALE = 1.0 / 255.0


_BLUR_BUF = {}


def wmap_from_polygon(poly_local, crop_px):
    ksize = int(2 * round(3.0 * SIGMA_PX) + 1)
    r = ksize // 2
    lo = np.floor(poly_local.min(axis=0)).astype(np.int64) - r - 1
    hi = np.ceil(poly_local.max(axis=0)).astype(np.int64) + r + 2
    x0 = int(max(0, lo[0])); y0 = int(max(0, lo[1]))
    x1 = int(min(crop_px, hi[0])); y1 = int(min(crop_px, hi[1]))
    if x1 <= x0 or y1 <= y0:
        return np.zeros((GRID, GRID), np.uint8), 0

    mask = np.zeros((y1 - y0, x1 - x0), np.uint8)
    pts = np.rint((poly_local - np.array([x0, y0])) * FILL_SCALE).astype(np.int32).reshape(-1, 1, 2)
    cv2.fillPoly(mask, [pts], 1, lineType=cv2.LINE_8, shift=FILL_SHIFT)
    area = int(mask.sum())
    blurred = cv2.GaussianBlur(mask.astype(np.float32), (ksize, ksize), SIGMA_PX,
                               borderType=cv2.BORDER_CONSTANT)


    buf = _BLUR_BUF.get(crop_px)
    if buf is None:
        buf = _BLUR_BUF[crop_px] = np.zeros((crop_px, crop_px), np.float32)
    buf[y0:y1, x0:x1] = blurred
    pooled = buf.reshape(GRID, PATCH, GRID, PATCH).mean(axis=(1, 3))
    buf[y0:y1, x0:x1] = 0.0
    return np.rint(np.clip(pooled, 0.0, 1.0) / QUANT_SCALE).astype(np.uint8), area


def morpho_tile(task):
    cv2.setNumThreads(1)
    l0 = task["l0"]
    mpp = task["mpp"]
    h0, w0 = l0.shape[:2]


    fw = max(1, int(round(w0 * mpp / XPX)))
    fh = max(1, int(round(h0 * mpp / XPX)))
    fine = cv2.resize(l0, (fw, fh), interpolation=cv2.INTER_LINEAR)

    sx = fw / (w0 * mpp)
    sy = fh / (h0 * mpp)
    x0_um, y0_um = task["x0_um"], task["y0_um"]


    bright_sum = cv2.integral((fine > 240).sum(axis=2, dtype=np.uint8))

    out = []
    for (ridx, cx_um, cy_um, poly_um) in task["cells"]:
        xc = int(round((cx_um - x0_um) * sx))
        yc = int(round((cy_um - y0_um) * sy))
        poly_f = np.column_stack([(poly_um[:, 0] - x0_um) * sx, (poly_um[:, 1] - y0_um) * sy])


        x0, y0 = xc - CROP_HALF, yc - CROP_HALF
        crop_px = CROP_HALF * 2
        sx0, sy0 = max(0, x0), max(0, y0)
        sx1, sy1 = min(fw, x0 + crop_px), min(fh, y0 + crop_px)
        if sx1 > sx0 and sy1 > sy0:
            inside = int(bright_sum[sy1, sx1] - bright_sum[sy0, sx1]
                         - bright_sum[sy1, sx0] + bright_sum[sy0, sx0])
            n_inside = (sy1 - sy0) * (sx1 - sx0)
        else:
            inside, n_inside = 0, 0

        bright = inside + 3 * (crop_px * crop_px - n_inside)
        is_blank = bright / float(crop_px * crop_px * 3) > BLANK_THR

        if len(poly_f) >= 3 and not is_blank:
            w16, area = wmap_from_polygon(poly_f - np.array([x0, y0]), crop_px)
            valid = 1 if area > 0 else 0
        else:
            w16, area, valid = np.zeros((GRID, GRID), np.uint8), 0, 0
        out.append((ridx, (w16, area), valid))
    return out


def contours_from_labels(lab):
    slices = find_objects(lab)
    polys = []
    for i, sl in enumerate(slices):
        if sl is None:
            continue
        sub = (lab[sl] == (i + 1)).astype(np.uint8)
        cnts, _ = cv2.findContours(sub, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not cnts:
            polys.append(np.zeros((0, 2), np.float32))
            continue
        c = max(cnts, key=cv2.contourArea).reshape(-1, 2).astype(np.float32)
        c[:, 0] += sl[1].start
        c[:, 1] += sl[0].start
        polys.append(c)
    return polys


def build_slide(slide, svs, native_mpp, device, n_workers, executor,
                limit_cells=0, dry_run=False):
    import torch
    from instanseg.utils.tiling import _instanseg_padding, _recover_padding

    log(f"[detect+wmap] {slide}: single pass, cell table and weight maps come out together")

    model, method, _ = SL.build_model_and_loss(device=device, warmstart=False, verbose=False)
    sd = torch.load(LUNG_CKPT, map_location=device, weights_only=False)
    model.load_state_dict(sd["model_state_dict"], strict=True)
    model.eval()
    model_ps = sd.get("pixel_size", SL.MODEL_PIXEL_SIZE)

    canvas = BandCachedSlide(svs, band_rows=BAND_ROWS)
    H, W = canvas.H, canvas.W
    scale = model_ps / native_mpp
    vm = OVERLAP // 2
    step = TILE - OVERLAP
    Ht = int(np.ceil(H / scale)); Wt = int(np.ceil(W / scale))
    y_lo, x_lo, y_hi, x_hi = 0, 0, Ht, Wt
    by = list(range(y_lo, y_hi, step)); bx = list(range(x_lo, x_hi, step))
    n_blocks = len(by) * len(bx)
    log(f"[morpho] {slide} L0=({H},{W}) mpp={native_mpp:.4f} scale={scale:.5f} blocks={n_blocks}")

    rec_xy = []
    feats = {}
    pending = set()
    t0 = time.time()
    done = 0
    n_cells_seen = 0
    tissue_px_model = 0.0

    def drain():
        nonlocal pending
        if not pending:
            return
        finished, pending = wait(pending, return_when=FIRST_COMPLETED)
        for fut in finished:
            for (ridx, feat, valid) in fut.result():
                feats[ridx] = (feat, valid)

    for ty0 in by:
        for tx0 in bx:
            done += 1
            ty1 = min(y_hi, ty0 + TILE); tx1 = min(x_hi, tx0 + TILE)


            ry0 = max(0, int((ty0 - HALO) * scale)); rx0 = max(0, int((tx0 - HALO) * scale))
            ry1 = min(H, int(np.ceil((ty1 + HALO) * scale))); rx1 = min(W, int(np.ceil((tx1 + HALO) * scale)))
            halo = canvas.read(ry0, rx0, ry1, rx1)
            if halo.size == 0:
                continue

            sy_a = int(ty0 * scale) - ry0
            sx_a = int(tx0 * scale) - rx0
            sy_b = min(H, int(np.ceil(ty1 * scale))) - ry0
            sx_b = min(W, int(np.ceil(tx1 * scale))) - rx0
            rgb_c = halo[sy_a:sy_b, sx_a:sx_b]
            if rgb_c.shape[0] < 8 or rgb_c.shape[1] < 8:
                continue
            th, tw = ty1 - ty0, tx1 - tx0
            rgb = cv2.resize(rgb_c, (tw, th), interpolation=cv2.INTER_AREA)
            if L.tissue_fraction(rgb) < MIN_TISSUE:
                continue


            cr_a = vm if ty0 > y_lo else 0
            cr_b = th - vm if ty1 < y_hi else th
            cc_a = vm if tx0 > x_lo else 0
            cc_b = tw - vm if tx1 < x_hi else tw
            if cr_b > cr_a and cc_b > cc_a:
                core = rgb[cr_a:cr_b, cc_a:cc_b]
                tissue_px_model += L.tissue_fraction(core) * core.shape[0] * core.shape[1]


            if min(rgb.shape[0], rgb.shape[1]) < MIN_TILE_PX:
                continue
            with torch.no_grad():
                x = SL.normalize_rgb(rgb)[None].to(device)
                x, pad = _instanseg_padding(x, extra_pad=0, min_dim=32)
                out = model(x).float()
                out = _recover_padding(out, pad)
                try:
                    cents, scores, lab = SL.decode_centroids(method, out[0], return_scores=True, **POST)
                except Exception as exc:
                    raise RuntimeError(
                        f"{slide}: nucleus decoding failed at model-space tile "
                        f"({tx0}, {ty0})-({tx1}, {ty1})"
                    ) from exc
            polys = contours_from_labels(lab) if len(cents) else []
            if len(polys) != len(cents):
                raise RuntimeError(f"label/centroid count mismatch {len(polys)} vs {len(cents)} "
                                   f"at tile ({ty0},{tx0})")
            tile_cells = []
            for j, (cx, cy) in enumerate(cents):
                if tx0 > x_lo and cx < vm: continue
                if ty0 > y_lo and cy < vm: continue
                if tx1 < x_hi and cx > tw - vm: continue
                if ty1 < y_hi and cy > th - vm: continue
                gx = (tx0 + cx) * scale; gy = (ty0 + cy) * scale
                ridx = n_cells_seen; n_cells_seen += 1
                rec_xy.append((gx, gy))
                p = polys[j]
                if len(p) >= 3:
                    poly_um = np.column_stack([(tx0 + p[:, 0]) * model_ps, (ty0 + p[:, 1]) * model_ps])
                else:
                    poly_um = np.zeros((0, 2), np.float64)
                tile_cells.append((ridx, (tx0 + cx) * model_ps, (ty0 + cy) * model_ps, poly_um))
            if tile_cells:
                task = {"l0": np.ascontiguousarray(halo), "mpp": native_mpp,
                        "x0_um": rx0 * native_mpp, "y0_um": ry0 * native_mpp,
                        "cells": tile_cells}
                pending.add(executor.submit(morpho_tile, task))
                while len(pending) >= 3 * n_workers:
                    drain()
            if done % 500 == 0 or done == n_blocks:
                el = time.time() - t0
                log(f"[morpho] block {done}/{n_blocks} cells={n_cells_seen:,} "
                    f"done={len(feats):,} inflight={len(pending)} ({el:.0f}s)")
            if limit_cells and n_cells_seen >= limit_cells:
                break
        if limit_cells and n_cells_seen >= limit_cells:
            break
    while pending:
        drain()
    canvas.close()
    secs = round(time.time() - t0, 1)
    N = int(n_cells_seen)
    log(f"[detect+wmap] tiles done: {N:,} cells ({secs}s)")


    rec = np.asarray(rec_xy, np.float64) if rec_xy else np.zeros((0, 2))
    if N:
        xpx = np.clip(np.round(rec[:, 0]), 0, W - 1).astype(np.int64)
        ypx = np.clip(np.round(rec[:, 1]), 0, H - 1).astype(np.int64)
    else:
        xpx = np.array([], np.int64); ypx = np.array([], np.int64)
    cell_id = np.asarray([f"{slide}_{i}" for i in range(N)], dtype=str)
    if len(np.unique(cell_id)) != N:
        raise RuntimeError(f"{slide}: generated cell IDs are not unique")
    cells_df = pd.DataFrame({
        "cell_id": cell_id,
        "x_px": xpx, "y_px": ypx,
        "x_um": xpx.astype(np.float64) * native_mpp,
        "y_um": ypx.astype(np.float64) * native_mpp,
    })
    slide_w_um = W * native_mpp
    slide_h_um = H * native_mpp
    if N:
        assert cells_df["x_um"].max() <= slide_w_um + 1e-6, "x_um exceeds slide width in um"
        assert cells_df["y_um"].max() <= slide_h_um + 1e-6, "y_um exceeds slide height in um"
        assert cells_df["x_um"].min() >= -1e-9 and cells_df["y_um"].min() >= -1e-9, "negative um coord"

    w16 = np.zeros((N, GRID, GRID), np.uint8)
    area_px = np.zeros(N, np.int32)
    valid_arr = np.zeros(N, np.uint8)
    n_missing = 0
    for i in range(N):
        if i not in feats:
            n_missing += 1
            continue
        (this_w16, this_area), valid = feats[i]
        w16[i] = this_w16
        area_px[i] = this_area
        valid_arr[i] = valid

    tissue_um2 = tissue_px_model * (model_ps ** 2)
    tissue_mm2 = tissue_um2 / 1e6
    density = float(N / tissue_mm2) if tissue_mm2 > 0 else 0.0


    flags = []
    if N == 0:
        flags.append("ZERO_CELLS")
    if tissue_mm2 < 1.0:
        flags.append("LOW_TISSUE")
    if tissue_mm2 >= 1.0:
        if density < 1000:
            flags.append("LOW_CELL_DENSITY")
        elif density > 15000:
            flags.append("HIGH_CELL_DENSITY")
    poly_frac = int(valid_arr.sum()) / max(N, 1)
    if N and poly_frac < 0.80:


        flags.append("LOW_POLYGON_FRACTION")

    if not dry_run:
        os.makedirs(CELLS_DIR, exist_ok=True)
        out_pq = os.path.join(CELLS_DIR, f"{slide}.parquet")
        tmp_pq = f"{out_pq}.tmp{os.getpid()}.parquet"
        cells_df.to_parquet(tmp_pq, index=False)
        os.replace(tmp_pq, out_pq)

        os.makedirs(WMAP_DIR, exist_ok=True)
        out_npz = os.path.join(WMAP_DIR, f"{slide}.npz")
        tmp = f"{out_npz}.tmp{os.getpid()}.npz"
        np.savez_compressed(tmp, cell_id=cell_id,
                            w16=w16, area_px=area_px, has_poly=valid_arr.astype(bool))
        os.replace(tmp, out_npz)

    n_valid = int(valid_arr.sum())
    meta = {
        "slide": slide, "cohort": COHORT, "svs": svs,
        "n_cells": int(N), "n_valid": n_valid,
        "valid_frac": round(n_valid / N, 4) if N else 0.0,
        "n_missing_feature": int(n_missing),
        "tissue_mm2": float(tissue_mm2),
        "cells_per_mm2": density,
        "flags": flags,
        "native_mpp": native_mpp, "model_pixel_size": model_ps, "scale": scale,
        "crop": {"xpx_um": XPX, "crop_px": CROP_HALF * 2, "fov_um": CROP_HALF * 2 * XPX,
                 "resample": "INTER_LINEAR (native->0.2125um/px is an upsample for HNSC)",
                 },
        "nucleus_source": "InstanSeg label mask contour (cv2.findContours, model space 0.5um/px)",
        "nucleus_area_definition": ("polygon area of OUR contour; training used Xenium's stored "
                                    "nucleus_area, which was ~1/0.94x the area of its own polygon "
                                    "(calib_polyarea_over_nucarea 0.936-0.947) -- a known "
                                    "definitional offset, NOT corrected here"),
        "sigma3": {"grid": GRID, "patch_px": PATCH, "sigma_px": SIGMA_PX,
                   "fill_shift": FILL_SHIFT, "quant_scale": QUANT_SCALE,
                   "polygon_source": "InstanSeg NUCLEUS contour (detector trained on Xenium WHOLE-CELL)"},
        "detector": {"ckpt": LUNG_CKPT, "post": POST, "tile": TILE, "overlap": OVERLAP,
                     "min_tissue": MIN_TISSUE, "halo": HALO},
        "seconds": secs,
        "provenance": {
            "script": os.path.abspath(__file__),
            "script_sha256": sha256_of(os.path.abspath(__file__)),
            "torch": torch.__version__, "python": platform.python_version(),
            "host": platform.node(), "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
            "slurm_array_task_id": os.environ.get("SLURM_ARRAY_TASK_ID"),
            "n_workers": n_workers,
            "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        },
    }
    out_meta = os.path.join(WMAP_DIR, f"{slide}.meta.json")
    if not dry_run:
        tmpm = f"{out_meta}.tmp{os.getpid()}"
        with open(tmpm, "w") as f:
            json.dump(meta, f, indent=2)
        os.replace(tmpm, out_meta)
    else:
        log(f"[detect+wmap] DRY RUN -- nothing written. meta={json.dumps(meta)[:400]}")
    log(f"[detect+wmap] DONE {slide} cells {N:,} with-polygon {n_valid:,} "
        f"({100*n_valid/max(N,1):.1f}%) tissue {tissue_mm2:.1f}mm2 {density:.0f}/mm2 "
        f"({secs}s) -> {os.path.join(WMAP_DIR, slide + '.npz')}")
    return meta


def main():
    import torch
    ap = argparse.ArgumentParser()
    ap.add_argument("--slide", default=None)
    ap.add_argument("--index", type=int, default=None)
    ap.add_argument("--slide-list", default=None)
    ap.add_argument("--manifest", default=os.path.join(INFERENCE_ROOT, "configs", COHORT, "run_manifest.tsv"))
    ap.add_argument("--workers", type=int, default=None)
    ap.add_argument("--limit-cells", type=int, default=0,
                    help="debug: stop the tile loop after ~N cells; use with --dry-run")
    ap.add_argument("--dry-run", action="store_true", help="debug: compute but write nothing")
    ap.add_argument("--device", default=None, help="debug: force cpu")
    args = ap.parse_args()

    slide = args.slide
    if slide is None:
        payload = json.load(open(args.slide_list))
        slides = payload["slides"] if isinstance(payload, dict) else payload
        idx = args.index if args.index is not None else int(os.environ.get("SLURM_ARRAY_TASK_ID", "-1"))
        if not (0 <= idx < len(slides)):
            raise SystemExit(f"FATAL: index {idx} out of range ({len(slides)} slides)")
        slide = slides[idx]

    out_npz = os.path.join(WMAP_DIR, f"{slide}.npz")
    out_meta = os.path.join(WMAP_DIR, f"{slide}.meta.json")
    cells_pq = os.path.join(CELLS_DIR, f"{slide}.parquet")
    if all(os.path.exists(p) for p in (out_npz, out_meta, cells_pq)) and not args.dry_run:
        log(f"[detect+wmap] exists -> skip ({out_npz})")
        return


    import pandas as _pd
    _man = _pd.read_csv(args.manifest, sep="\t").set_index("slide")
    if slide not in _man.index:
        raise SystemExit(f"FATAL: {slide} not in {args.manifest}")
    svs = str(_man.loc[slide, "svs"])
    if not os.path.isfile(svs):
        raise SystemExit(f"FATAL: SVS not found: {svs}")

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    log(f"=== wmap {slide} device={device} torch={torch.__version__} ===")


    if device != "cuda" and args.device is None and not args.dry_run:
        log("FATAL: no CUDA visible (pyxis not stripped at submit time?). Abort.")
        sys.exit(2)

    nw = args.workers or max(1, int(os.environ.get("SLURM_CPUS_PER_TASK", "8")) - 2)


    cv2.setNumThreads(1)
    if device != "cuda":
        n_threads = int(os.environ.get("SLURM_CPUS_PER_TASK", "16"))
        torch.set_num_threads(n_threads)
        try:
            torch.set_num_interop_threads(max(2, n_threads // 4))
        except RuntimeError:
            pass
        log(f"[cpu] torch threads={torch.get_num_threads()} workers={nw}")
    native_mpp = read_native_mpp(svs)
    import multiprocessing as mp
    ctx = mp.get_context("spawn")
    with ProcessPoolExecutor(max_workers=nw, mp_context=ctx) as ex:
        build_slide(slide, svs, native_mpp, device, nw, ex,
                    limit_cells=args.limit_cells, dry_run=args.dry_run)


if __name__ == "__main__":
    main()
