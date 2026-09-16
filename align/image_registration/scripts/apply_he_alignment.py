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
from detect_he_crop_mask import (
    load_he_full_rgb, compute_he_crop_mask, compute_he_crop_samples,
    downsample_long, _json_safe,
    CLOSE_UM, COMPARABILITY_RATIO, NECK_SPLIT_FRAC, MIN_SAMPLE_AREA_MM2,
    EROSION_STEP_UM, MAX_EROSION_UM, NECK_NOISE_FRAC, MAX_SPLIT_DEPTH,
    CROP_GATE_PERIMETER_FRAC, GATE_BAND_UM, HE_SELECT_PROJ_FRAC,
    MAX_EXTEND_UM, NCC_LONG_DIM,
)


def save_pyramidal_ome_tiff(img, out_path, pixel_size_um, tile_size=512,
                              compression='jpeg'):
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
    photometric = 'rgb' if img.ndim == 3 and img.shape[-1] == 3 else 'minisblack'
    options = dict(
        tile=(tile_size, tile_size),
        compression=compression,
        photometric=photometric,
        metadata={'axes': 'YXC' if img.ndim == 3 else 'YX',
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


def to_u8_perc(img):
    fg = img[img > 0] if (img > 0).any() else img.ravel()
    p2, p98 = np.percentile(fg, [2, 98])
    return np.clip((img - p2) / (p98 - p2 + 1e-8) * 255, 0, 255).astype(np.uint8)


def make_overlay(he_on_xen_rgb, xen_dapi_u8, target_long_dim=2400):
    rg = he_on_xen_rgb[..., :2].astype(np.float32).mean(axis=-1)
    he_gray = (255 - rg).astype(np.uint8)
    h, w = he_gray.shape[:2]
    s = target_long_dim / max(h, w)
    if s < 1.0:
        he_small = cv2.resize(he_gray, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)
        xen_small = cv2.resize(xen_dapi_u8, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)
    else:
        he_small = he_gray; xen_small = xen_dapi_u8
    out = np.zeros((he_small.shape[0], he_small.shape[1], 3), dtype=np.uint8)
    out[..., 1] = he_small
    out[..., 2] = xen_small
    return out


def warp_he_crop_to_xenium(he_crop, crop_x1, crop_y1, M_he_to_xen,
                              xen_h, xen_w, tile=4096):
    A = M_he_to_xen[:2, :2]
    b = M_he_to_xen[:, 2] + A @ np.array([crop_x1, crop_y1])
    print(f'  Warping HE crop ({he_crop.shape}) → Xenium canvas '
          f'({xen_h}x{xen_w}) tile={tile}…', flush=True)
    out = np.full((xen_h, xen_w, 3), 255, dtype=np.uint8)
    A_inv = np.linalg.inv(A)
    b_inv = -A_inv @ b
    src_h, src_w = he_crop.shape[:2]
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
            src = he_crop[sy_min:sy_max, sx_min:sx_max]
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
                                              borderValue=(255, 255, 255))
                out[y0:y1, x0:x1] = tile_warp
            except cv2.error as e:
                print(f'    tile ({x0},{y0}) warp error: {e}', flush=True)
                continue
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--align', required=True,
                     help='Directory with global_alignment.json')
    ap.add_argument('--he', required=True, help='HE qptiff path')
    ap.add_argument('--output', required=True, help='Output dir')
    ap.add_argument('--xenium', default=None,
                     help='Xenium dir (else taken from align json)')
    ap.add_argument('--sat-thresh', type=int, default=25,
                     help='HSV saturation threshold for HE tissue (default 25)')
    ap.add_argument('--he-work-size', type=int, default=2400,
                     help='HE working canvas long axis px (default 2400)')
    ap.add_argument('--dilate-px', type=int, default=15,
                     help='Final-mask dilation in HE-FULL pixels (default 15)')
    ap.add_argument('--overlay-max', type=int, default=2400)
    args = ap.parse_args()

    out = Path(args.output); out.mkdir(parents=True, exist_ok=True)
    j = json.loads((Path(args.align) / 'global_alignment.json').read_text())
    M_xen_to_he = np.array(j['M_xen_full_to_he_full'])
    M_he_to_xen = np.array(j['M_he_full_to_xen_full'])
    xen_h, xen_w = j['full_resolution']['xenium_shape_hw']
    xen_px = j['full_resolution']['xenium_px_um']
    xenium_dir = args.xenium or j['inputs'].get('xenium_dir')

    t0 = time.time()
    print(f'[{time.time()-t0:.0f}s] Loading HE qptiff: {args.he}', flush=True)
    he_rgb, he_px_full = load_he_full_rgb(args.he)
    he_h, he_w = he_rgb.shape[:2]
    print(f'  shape={he_rgb.shape} px={he_px_full:.4f} µm', flush=True)

    print(f'[{time.time()-t0:.0f}s] Loading Xenium DAPI: {xenium_dir}',
            flush=True)
    dapi, dapi_px = load_dapi(xenium_path=str(xenium_dir))
    print(f'  shape={dapi.shape} px={dapi_px:.4f} µm', flush=True)

    print(f'[{time.time()-t0:.0f}s] Computing HE crop samples (gate)', flush=True)
    r = compute_he_crop_samples(M_xen_to_he, dapi, dapi_px, he_rgb, he_px_full,
                                he_work_size=args.he_work_size,
                                sat_thresh=args.sat_thresh)
    print(f'  Xenium: {len(r["xen_areas_mm2"])} biopsies, '
          f'areas={r["xen_areas_mm2"]} mm²', flush=True)
    print(f'  HE samples: {r["n_he_samples"]} | '
          f'Xenium samples: {r["xen_n_samples"]} | crops: {len(r["crops"])}',
          flush=True)
    n_crops = len(r['crops'])
    if n_crops == 0:
        print('  ERROR: no Xenium samples produced a crop', flush=True)
        sys.exit(1)

    he_small = r['he_small']


    written = []
    for c in r['crops']:
        cm = c['crop_mask_small']
        if cm.sum() == 0:
            print(f'  [xen_sample={c["xen_sample_id"]}] '
                  f'WARNING: empty crop mask, skipping', flush=True)
            continue
        written.append(c)


    uniq, seen = [], {}
    for c in written:
        key = hash(np.ascontiguousarray(c['crop_mask_small']).tobytes())
        if key in seen:
            seen[key].setdefault('merged_xen_sample_ids',
                                 [seen[key]['xen_sample_id']]).append(
                                     c['xen_sample_id'])
            continue
        seen[key] = c
        uniq.append(c)
    if len(uniq) != len(written):
        print(f'  dedup: {len(written)} crops → {len(uniq)} unique HE crop(s) '
              f'(identical HE region matched by multiple Xenium biopsies)',
              flush=True)
    written = uniq
    n_written = len(written)
    if n_written == 0:
        print('  ERROR: every crop mask was empty; nothing written', flush=True)
        sys.exit(1)

    def _suffix(i):

        return '' if n_written == 1 else f'_s{i + 1}'

    crop_records = []
    for ci, c in enumerate(written):
        sfx = _suffix(ci)
        xs_id = c['xen_sample_id']
        crop_small = c['crop_mask_small']
        print(f'[{time.time()-t0:.0f}s] === Crop {ci + 1}/{n_written} '
              f'xen_sample={xs_id} mode={c["crop_mode"]} '
              f'he_samples={c["he_sample_ids"]} '
              f'gate_contact={c["gate_contact_frac"]} '
              f'NCC(he→dapi)={c["ncc_he_to_dapi"]} '
              f'NCC(mask↔mask)={c["ncc_mask_to_mask"]} ===', flush=True)


        print(f'  Upscaling mask to HE full ({he_h}×{he_w})', flush=True)
        mask_full = cv2.resize(crop_small, (he_w, he_h),
                               interpolation=cv2.INTER_NEAREST)
        if args.dilate_px > 0:
            mask_full = cv2.dilate(mask_full,
                                   cv2.getStructuringElement(cv2.MORPH_ELLIPSE,
                                                             (args.dilate_px,
                                                              args.dilate_px)))


        ys, xs = np.where(mask_full > 0)
        bx1, bx2 = int(xs.min()), int(xs.max()) + 1
        by1, by2 = int(ys.min()), int(ys.max()) + 1
        print(f'  bbox=({bx1},{by1})→({bx2},{by2}) = {bx2-bx1}×{by2-by1}',
              flush=True)
        he_crop = he_rgb[by1:by2, bx1:bx2].copy()
        crop_mask = mask_full[by1:by2, bx1:bx2]
        he_crop[crop_mask == 0] = 255
        print(f'  HE crop shape={he_crop.shape}', flush=True)

        save_pyramidal_ome_tiff(he_crop, out / f'he_crop{sfx}.ome.tif',
                                pixel_size_um=he_px_full)


        overlay = cv2.cvtColor(he_small, cv2.COLOR_RGB2BGR)
        contours, _ = cv2.findContours(crop_small, cv2.RETR_EXTERNAL,
                                       cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(overlay, contours, -1, (0, 255, 0), 3)
        cv2.imwrite(str(out / f'he_whole_crop{sfx}.png'),
                    downsample_long(overlay, args.overlay_max))
        del mask_full, crop_mask, overlay, contours


        print(f'  Warping HE crop into Xenium space', flush=True)
        he_on_xen = warp_he_crop_to_xenium(he_crop, bx1, by1, M_he_to_xen,
                                           xen_h, xen_w)
        save_pyramidal_ome_tiff(he_on_xen, out / f'he_on_xenium{sfx}.ome.tif',
                                pixel_size_um=xen_px)
        del he_crop


        print(f'  Building coarse_overlay{sfx}.jpg', flush=True)
        xen_u8 = to_u8_perc(dapi)
        overlay_jpg = make_overlay(he_on_xen, xen_u8,
                                   target_long_dim=args.overlay_max)
        cv2.imwrite(str(out / f'coarse_overlay{sfx}.jpg'), overlay_jpg)
        del he_on_xen, xen_u8, overlay_jpg


        np.savetxt(out / f'affine_local{sfx}.csv', M_xen_to_he, delimiter=',')

        crop_records.append(dict(
            xen_sample_id=int(xs_id),
            merged_xen_sample_ids=c.get('merged_xen_sample_ids', [int(xs_id)]),
            he_sample_ids=c['he_sample_ids'],
            crop_mode=c['crop_mode'],
            crop_bbox_he=[bx1, by1, bx2, by2],
            gate_contact_frac=c['gate_contact_frac'],
            neck_connection_mm2=c['neck_connection_mm2'],
            neck_connection_ratio=c['neck_connection_ratio'],
            ncc_he_to_dapi=c['ncc_he_to_dapi'],
            ncc_mask_to_mask=c['ncc_mask_to_mask'],
            area_mm2=c['area_mm2'],
            outputs=dict(
                he_crop=f'he_crop{sfx}.ome.tif',
                he_on_xenium=f'he_on_xenium{sfx}.ome.tif',
                he_whole_crop=f'he_whole_crop{sfx}.png',
                coarse_overlay=f'coarse_overlay{sfx}.jpg',
                affine_local=f'affine_local{sfx}.csv',
            ),
            he_sample_selection=c.get('selection'),
        ))


    del he_rgb, dapi

    if not crop_records:
        print('  ERROR: every crop mask was empty; nothing written', flush=True)
        sys.exit(1)


    first_bbox = crop_records[0]['crop_bbox_he']


    he_px_small_um = r['he_px_small_um']
    extend_radius_px_small = max(0, int(MAX_EXTEND_UM / he_px_small_um))

    crop_gate_params = dict(
        close_um=CLOSE_UM,
        comparability_ratio=COMPARABILITY_RATIO,
        neck_split_frac=NECK_SPLIT_FRAC,
        min_sample_area_mm2=MIN_SAMPLE_AREA_MM2,
        erosion_step_um=EROSION_STEP_UM,
        max_erosion_um=MAX_EROSION_UM,
        neck_noise_frac=NECK_NOISE_FRAC,
        max_split_depth=MAX_SPLIT_DEPTH,
        crop_gate_perimeter_frac=CROP_GATE_PERIMETER_FRAC,
        gate_band_um=GATE_BAND_UM,
        he_select_proj_frac=HE_SELECT_PROJ_FRAC,
        max_extend_um=MAX_EXTEND_UM,
        ncc_long_dim=NCC_LONG_DIM,
    )

    (out / 'apply_alignment.json').write_text(json.dumps(_json_safe({
        'source_alignment': str(Path(args.align) / 'global_alignment.json'),
        'he_qptiff': str(args.he),
        'xenium_dir': str(xenium_dir),
        'crop_bbox_he': first_bbox,
        'extend_radius_px_small': extend_radius_px_small,
        'dilate_px': args.dilate_px,
        'max_extend_um': MAX_EXTEND_UM,
        'sat_thresh': args.sat_thresh,
        'xen_biopsy_areas_mm2': r['xen_areas_mm2'],
        'M_xen_full_to_he_full': M_xen_to_he.tolist(),
        'M_he_full_to_xen_full': M_he_to_xen.tolist(),
        'he_px_um_full': float(he_px_full),
        'xenium_px_um': float(xen_px),
        'xenium_shape_hw': [int(xen_h), int(xen_w)],
        'he_px_small_um': r['he_px_small_um'],
        'n_xenium_samples': r['xen_n_samples'],
        'n_he_samples': r['n_he_samples'],
        'crop_gate_params': crop_gate_params,
        'crops': crop_records,
        'elapsed_seconds': float(time.time() - t0),
    }), indent=2))
    print(f'\n[{time.time()-t0:.0f}s] DONE → {out} ({len(crop_records)} crop(s))',
          flush=True)


if __name__ == '__main__':
    main()
