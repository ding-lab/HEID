#!/usr/bin/env python3
from __future__ import annotations
import argparse
import json
import os
import sys
import time
from pathlib import Path

os.environ['PYTHONUNBUFFERED'] = '1'
import cv2
import numpy as np
from scipy.ndimage import binary_fill_holes

SCRIPTS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPTS_DIR))
from pack_pipeline import load_dapi, normalize_dapi


def detect_tissue_mask(dapi: np.ndarray, dapi_px: float,
                          work_size: int = 1500,
                          min_area_mm2: float = 0.5,
                          bright_otsu_scale: float = 0.10) -> tuple:
    ds = min(1.0, work_size / max(dapi.shape[:2]))
    small = cv2.resize(dapi, None, fx=ds, fy=ds,
                        interpolation=cv2.INTER_AREA) if ds < 0.95 else dapi
    norm = normalize_dapi(small)
    otsu_t, _ = cv2.threshold(norm, 0, 255,
                                  cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    t = max(5, int(otsu_t * bright_otsu_scale))
    _, base = cv2.threshold(norm, t, 255, cv2.THRESH_BINARY)


    close_r = max(1, int(round(15.0 / (dapi_px / ds))))
    base = cv2.morphologyEx(
        base, cv2.MORPH_CLOSE,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE,
                                  (2 * close_r + 1, 2 * close_r + 1)))
    mb = base > 0
    n_h, lab_h, st_h, _ = cv2.connectedComponentsWithStats(
        (~mb).astype(np.uint8), connectivity=8)
    max_hole_px = 0.1 * 1e6 / ((dapi_px / ds) ** 2)
    for hi in range(1, n_h):
        x0, y0, w0, h0, a0 = st_h[hi]
        touches = (x0 == 0 or y0 == 0
                   or x0 + w0 >= mb.shape[1] or y0 + h0 >= mb.shape[0])
        if (not touches) and a0 < max_hole_px:
            mb[lab_h == hi] = True
    base = (mb.astype(np.uint8)) * 255

    n, labels, stats, _ = cv2.connectedComponentsWithStats(base, connectivity=8)
    um2_per_px_small = (dapi_px / ds) ** 2
    min_area_px_small = int(min_area_mm2 * 1e6 / um2_per_px_small)
    out = np.zeros_like(base)
    biopsy_areas = []
    for i in range(1, n):
        a = int(stats[i, cv2.CC_STAT_AREA])
        if a < min_area_px_small:
            continue
        out[labels == i] = 255
        biopsy_areas.append(a * um2_per_px_small / 1e6)
    biopsy_areas.sort(reverse=True)
    return out, small, ds, biopsy_areas


def downsample_long(img, target_long_dim, interp=cv2.INTER_AREA):
    h, w = img.shape[:2]
    s = target_long_dim / max(h, w)
    if s >= 1.0:
        return img.copy()
    return cv2.resize(img, None, fx=s, fy=s, interpolation=interp)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--align', required=True)
    ap.add_argument('--output', required=True)
    ap.add_argument('--xenium', default=None)
    ap.add_argument('--debug-max', type=int, default=2400)
    args = ap.parse_args()

    out = Path(args.output); out.mkdir(parents=True, exist_ok=True)
    j = json.loads((Path(args.align) / 'global_alignment.json').read_text())
    xen_dir = args.xenium or j['inputs'].get('xenium_dir')

    t0 = time.time()
    print(f'[{time.time()-t0:.0f}s] Loading Xenium DAPI from {xen_dir}',
            flush=True)
    dapi, dapi_px = load_dapi(xenium_path=str(xen_dir))
    print(f'  shape={dapi.shape} dtype={dapi.dtype} px={dapi_px:.4f} µm',
            flush=True)

    print(f'[{time.time()-t0:.0f}s] Detecting tissue mask', flush=True)
    mask, small, ds, areas = detect_tissue_mask(dapi, dapi_px)
    print(f'  ds={ds:.4f} small_shape={small.shape} '
            f'{len(areas)} biopsies, areas={[f"{a:.2f}" for a in areas]} mm²',
            flush=True)
    print(f'  mask area = {100*mask.mean()/255:.1f}% of small frame',
            flush=True)


    norm = normalize_dapi(small)
    overlay = cv2.cvtColor(norm, cv2.COLOR_GRAY2BGR)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL,
                                       cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(overlay, contours, -1, (0, 255, 0), 3)
    out_path = out / 'xenium_mask.png'
    cv2.imwrite(str(out_path), downsample_long(overlay, args.debug_max))
    print(f'[{time.time()-t0:.0f}s] DONE → {out_path}', flush=True)


if __name__ == '__main__':
    main()
