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
import tifffile

SCRIPTS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPTS_DIR))
from pack_pipeline import load_dapi
from detect_codex_crop_mask import (
    load_codex_full_dapi, compute_codex_crop_mask,
)
from detect_he_crop_mask import downsample_long
from apply_he_alignment import to_u8_perc


def save_pyramidal_ome_tiff_gray(img, out_path, pixel_size_um, tile_size=512,
                                       compression='zlib'):
    print(f'  Writing pyramidal OME-TIFF → {out_path}', flush=True)
    pyramid = [img]
    cur = img
    while min(cur.shape[:2]) > 256:
        new_h, new_w = cur.shape[0] // 2, cur.shape[1] // 2
        if new_h < 256 or new_w < 256:
            break
        cur = cv2.resize(cur, (new_w, new_h), interpolation=cv2.INTER_AREA)
        pyramid.append(cur)
    print(f'    {len(pyramid)} levels, base={pyramid[0].shape}', flush=True)
    options = dict(
        tile=(tile_size, tile_size),
        compression=compression,
        photometric='minisblack',
        metadata={'axes': 'YX',
                  'PhysicalSizeX': float(pixel_size_um),
                  'PhysicalSizeXUnit': 'µm',
                  'PhysicalSizeY': float(pixel_size_um),
                  'PhysicalSizeYUnit': 'µm'},
    )
    with tifffile.TiffWriter(str(out_path), bigtiff=True, ome=True) as tif:
        tif.write(pyramid[0], subifds=len(pyramid) - 1, **options)
        for level in pyramid[1:]:
            tif.write(level, subfiletype=1, **options)
    sz_mb = Path(out_path).stat().st_size / (1024 ** 2)
    print(f'    {sz_mb:.1f} MB', flush=True)


def make_overlay_gray(cdx_on_xen_u16, xen_dapi_u8, target_long_dim=2400):
    cdx_u8 = to_u8_perc(cdx_on_xen_u16)
    h, w = cdx_u8.shape[:2]
    s = target_long_dim / max(h, w)
    if s < 1.0:
        cdx_small = cv2.resize(cdx_u8, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)
        xen_small = cv2.resize(xen_dapi_u8, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)
    else:
        cdx_small = cdx_u8; xen_small = xen_dapi_u8
    out = np.zeros((cdx_small.shape[0], cdx_small.shape[1], 3), dtype=np.uint8)
    out[..., 1] = cdx_small
    out[..., 2] = xen_small
    return out


def warp_codex_crop_to_xenium(cdx_crop, crop_x1, crop_y1, M_cdx_to_xen,
                                    xen_h, xen_w, tile=4096):
    A = M_cdx_to_xen[:2, :2]
    b = M_cdx_to_xen[:, 2] + A @ np.array([crop_x1, crop_y1])
    print(f'  Warping CODEX crop ({cdx_crop.shape}) → Xenium canvas '
          f'({xen_h}x{xen_w}) tile={tile}…', flush=True)
    out = np.zeros((xen_h, xen_w), dtype=cdx_crop.dtype)
    A_inv = np.linalg.inv(A)
    b_inv = -A_inv @ b
    src_h, src_w = cdx_crop.shape[:2]
    for y0 in range(0, xen_h, tile):
        for x0 in range(0, xen_w, tile):
            y1 = min(y0 + tile, xen_h); x1 = min(x0 + tile, xen_w)
            th, tw = y1 - y0, x1 - x0
            corners = np.array([[x0, y0], [x1, y0], [x1, y1], [x0, y1]],
                                 dtype=np.float64)
            src_corners = (A_inv @ corners.T).T + b_inv
            sx_min = max(0, int(np.floor(src_corners[:, 0].min())) - 4)
            sy_min = max(0, int(np.floor(src_corners[:, 1].min())) - 4)
            sx_max = min(src_w, int(np.ceil(src_corners[:, 0].max())) + 4)
            sy_max = min(src_h, int(np.ceil(src_corners[:, 1].max())) + 4)
            if sx_max <= sx_min or sy_max <= sy_min:
                continue
            src = cdx_crop[sy_min:sy_max, sx_min:sx_max]
            if src.size == 0:
                continue
            M_tile = np.array([
                [A[0, 0], A[0, 1],
                 A[0, 0] * sx_min + A[0, 1] * sy_min + b[0] - x0],
                [A[1, 0], A[1, 1],
                 A[1, 0] * sx_min + A[1, 1] * sy_min + b[1] - y0],
            ], dtype=np.float64)
            try:
                tile_warp = cv2.warpAffine(src, M_tile.astype(np.float32),
                                              (tw, th),
                                              flags=cv2.INTER_LINEAR,
                                              borderMode=cv2.BORDER_CONSTANT,
                                              borderValue=0)
                out[y0:y1, x0:x1] = tile_warp
            except cv2.error as e:
                print(f'    tile ({x0},{y0}) warp error: {e}', flush=True)
                continue
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--align', required=True,
                     help='Directory containing CODEX global_alignment.json')
    ap.add_argument('--codex', required=True, help='CODEX qptiff path')
    ap.add_argument('--output', required=True, help='Output dir')
    ap.add_argument('--xenium', default=None,
                     help='Xenium dir (else taken from align json)')
    ap.add_argument('--max-extend-um', type=float, default=150.0,
                     help='Geodesic extension distance into CODEX tissue '
                          '(default 150 µm)')
    ap.add_argument('--cdx-work-size', type=int, default=2400)
    ap.add_argument('--dilate-px', type=int, default=15,
                     help='Final-mask dilation in CODEX-FULL pixels (default 15)')
    ap.add_argument('--overlay-max', type=int, default=2400)
    args = ap.parse_args()

    out = Path(args.output); out.mkdir(parents=True, exist_ok=True)
    j = json.loads((Path(args.align) / 'global_alignment.json').read_text())
    M_xen_to_cdx = np.array(j['M_xen_full_to_cdx_full'])
    M_cdx_to_xen = np.array(j['M_cdx_full_to_xen_full'])
    xen_h, xen_w = j['full_resolution']['xenium_shape_hw']
    xen_px = j['full_resolution']['xenium_px_um']
    xenium_dir = args.xenium or j['inputs'].get('xenium_dir')

    t0 = time.time()
    print(f'[{time.time()-t0:.0f}s] Loading CODEX DAPI: {args.codex}', flush=True)
    cdx_dapi, cdx_px_full = load_codex_full_dapi(args.codex)
    cdx_h, cdx_w = cdx_dapi.shape[:2]
    print(f'  shape={cdx_dapi.shape} dtype={cdx_dapi.dtype} px={cdx_px_full:.4f} µm',
            flush=True)

    print(f'[{time.time()-t0:.0f}s] Loading Xenium DAPI: {xenium_dir}',
            flush=True)
    dapi, dapi_px = load_dapi(xenium_path=str(xenium_dir))
    print(f'  shape={dapi.shape} px={dapi_px:.4f} µm', flush=True)

    print(f'[{time.time()-t0:.0f}s] Computing CODEX crop mask', flush=True)
    r = compute_codex_crop_mask(M_xen_to_cdx, dapi, dapi_px, cdx_dapi, cdx_px_full,
                                      cdx_work_size=args.cdx_work_size,
                                      max_extend_um=args.max_extend_um)
    print(f'  Xenium: {len(r["xen_areas_mm2"])} biopsies, areas={r["xen_areas_mm2"]} mm²',
            flush=True)
    print(f'  CODEX geodesic extend: {args.max_extend_um:.0f} µm '
            f'= {r["extend_radius_px_small"]} small px', flush=True)
    if r['mask_small'].sum() == 0:
        print('  ERROR: projected Xenium mask has no overlap with CODEX tissue mask',
                flush=True)
        sys.exit(1)

    print(f'[{time.time()-t0:.0f}s] Upscaling mask to CODEX full ({cdx_h}×{cdx_w})',
            flush=True)
    mask_full = cv2.resize(r['mask_small'], (cdx_w, cdx_h),
                              interpolation=cv2.INTER_NEAREST)
    if args.dilate_px > 0:
        mask_full = cv2.dilate(mask_full,
                                  cv2.getStructuringElement(cv2.MORPH_ELLIPSE,
                                                              (args.dilate_px,
                                                               args.dilate_px)))

    ys, xs = np.where(mask_full > 0)
    bx1, bx2 = int(xs.min()), int(xs.max()) + 1
    by1, by2 = int(ys.min()), int(ys.max()) + 1
    print(f'  bbox=({bx1},{by1})→({bx2},{by2}) = {bx2-bx1}×{by2-by1}', flush=True)
    cdx_crop = cdx_dapi[by1:by2, bx1:bx2].copy()
    crop_mask = mask_full[by1:by2, bx1:bx2]
    cdx_crop[crop_mask == 0] = 0
    print(f'  CODEX crop shape={cdx_crop.shape} dtype={cdx_crop.dtype}',
            flush=True)

    save_pyramidal_ome_tiff_gray(cdx_crop, out / 'codex_crop.ome.tif',
                                       pixel_size_um=cdx_px_full)


    norm = cv2.normalize(r['cdx_small'], None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    overlay = cv2.cvtColor(norm, cv2.COLOR_GRAY2BGR)
    contours, _ = cv2.findContours(r['mask_small'], cv2.RETR_EXTERNAL,
                                       cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(overlay, contours, -1, (0, 255, 0), 3)
    cv2.imwrite(str(out / 'codex_whole_crop.png'),
                  downsample_long(overlay, args.overlay_max))

    del cdx_dapi, mask_full

    print(f'[{time.time()-t0:.0f}s] Warping CODEX crop into Xenium space',
            flush=True)
    cdx_on_xen = warp_codex_crop_to_xenium(cdx_crop, bx1, by1, M_cdx_to_xen,
                                                  xen_h, xen_w)
    save_pyramidal_ome_tiff_gray(cdx_on_xen, out / 'codex_on_xenium.ome.tif',
                                       pixel_size_um=xen_px)
    del cdx_crop

    print(f'[{time.time()-t0:.0f}s] Building coarse_overlay.jpg', flush=True)
    xen_u8 = to_u8_perc(dapi)
    overlay_jpg = make_overlay_gray(cdx_on_xen, xen_u8,
                                          target_long_dim=args.overlay_max)
    cv2.imwrite(str(out / 'coarse_overlay.jpg'), overlay_jpg)
    del cdx_on_xen, xen_u8, dapi

    np.savetxt(out / 'affine_local.csv', M_xen_to_cdx, delimiter=',')

    (out / 'apply_alignment.json').write_text(json.dumps({
        'source_alignment': str(Path(args.align) / 'global_alignment.json'),
        'codex_qptiff': str(args.codex),
        'xenium_dir': str(xenium_dir),
        'crop_bbox_codex': [bx1, by1, bx2, by2],
        'dilate_px': args.dilate_px,
        'max_extend_um': args.max_extend_um,
        'extend_radius_px_small': r['extend_radius_px_small'],
        'xen_biopsy_areas_mm2': r['xen_areas_mm2'],
        'M_xen_full_to_cdx_full': M_xen_to_cdx.tolist(),
        'M_cdx_full_to_xen_full': M_cdx_to_xen.tolist(),
        'codex_px_um_full': float(cdx_px_full),
        'xenium_px_um': float(xen_px),
        'xenium_shape_hw': [int(xen_h), int(xen_w)],
        'elapsed_seconds': float(time.time() - t0),
    }, indent=2))
    print(f'\n[{time.time()-t0:.0f}s] DONE → {out}', flush=True)


if __name__ == '__main__':
    main()
