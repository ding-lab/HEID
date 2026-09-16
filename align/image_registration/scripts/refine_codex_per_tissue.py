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
from align_xenium_to_he import (
    Context, top_k_translations_via_fft, normalize_dapi,
    extract_xen_tissue_mask, masked_ncc, affine_from_params,
)
from refine_he_per_tissue import (
    evaluate_local, evaluate_silhouette,
    _powell_refine, refine_per_cc,
    project_xen_bbox_to_he as project_xen_bbox,
    compose_local_M, invert_2x3_affine,
)
from detect_codex_crop_mask import (
    load_codex_full_dapi, load_codex_full_channel, codex_channel_info,
    detect_codex_tissue_mask,
)
from apply_codex_alignment import (
    save_pyramidal_ome_tiff_gray, make_overlay_gray,
    warp_codex_crop_to_xenium,
)
from apply_he_alignment import to_u8_perc
from detect_xenium_mask import detect_tissue_mask
from pack_codex_aligned_ome import (
    build_pyramid_memmaps as _xe_build_pyramid_memmaps,
    write_xenium_compatible_multichannel_ome as _xe_write_multichannel_ome,
    _resize_to as _xe_resize_to,
)


def build_cc_context(dapi, dapi_px, cdx_dapi_full, cdx_px,
                          M_xen_to_cdx, M_cdx_to_xen,
                          xen_x1, xen_y1, xen_x2, xen_y2,
                          margin_xen_px, margin_cdx_px,
                          work_size=1200):
    H_xen, W_xen = dapi.shape[:2]
    H_cdx, W_cdx = cdx_dapi_full.shape[:2]
    cx1 = max(0, xen_x1 - margin_xen_px); cy1 = max(0, xen_y1 - margin_xen_px)
    cx2 = min(W_xen, xen_x2 + margin_xen_px); cy2 = min(H_xen, xen_y2 + margin_xen_px)
    chip_w_full = cx2 - cx1; chip_h_full = cy2 - cy1
    ds_chip = min(1.0, work_size / max(chip_w_full, chip_h_full))
    chip_w = max(1, int(round(chip_w_full * ds_chip)))
    chip_h = max(1, int(round(chip_h_full * ds_chip)))

    xen_chip_full = dapi[cy1:cy2, cx1:cx2]
    if ds_chip < 0.95:
        xen_chip = cv2.resize(xen_chip_full, (chip_w, chip_h),
                                  interpolation=cv2.INTER_AREA)
    else:
        xen_chip = xen_chip_full
    xen_chip_norm = normalize_dapi(xen_chip)

    cdx1f, cdy1f, cdx2f, cdy2f = project_xen_bbox(M_xen_to_cdx,
                                                            cx1, cy1, cx2, cy2)
    cdx1 = max(0, int(cdx1f) - margin_cdx_px)
    cdy1 = max(0, int(cdy1f) - margin_cdx_px)
    cdx2_ = min(W_cdx, int(cdx2f) + margin_cdx_px)
    cdy2_ = min(H_cdx, int(cdy2f) + margin_cdx_px)
    cdx_subcrop = cdx_dapi_full[cdy1:cdy2_, cdx1:cdx2_]


    cdx_norm_sub = normalize_dapi(cdx_subcrop)
    _, cdx_mask_sub_u8 = cv2.threshold(cdx_norm_sub, 0, 255,
                                              cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    A_c2x = M_cdx_to_xen[:2, :2]
    A_use = ds_chip * A_c2x
    b_use = ds_chip * (-np.array([cx1, cy1]) + M_cdx_to_xen[:, 2]
                          + A_c2x @ np.array([cdx1, cdy1]))
    M_use = np.zeros((2, 3), dtype=np.float64)
    M_use[:2, :2] = A_use; M_use[:, 2] = b_use
    cdx_chip = cv2.warpAffine(cdx_norm_sub, M_use.astype(np.float32),
                                   (chip_w, chip_h), flags=cv2.INTER_LINEAR,
                                   borderValue=0)
    cdx_mask_chip_u8 = cv2.warpAffine(cdx_mask_sub_u8, M_use.astype(np.float32),
                                          (chip_w, chip_h),
                                          flags=cv2.INTER_NEAREST, borderValue=0)
    cdx_mask_chip = cdx_mask_chip_u8 > 127

    xen_mask_chip = extract_xen_tissue_mask(xen_chip_norm)


    xen_chip_clean = xen_chip_norm.copy()
    xen_chip_clean[~xen_mask_chip] = 0
    cdx_chip_clean = cdx_chip.copy()
    cdx_chip_clean[~cdx_mask_chip] = 0

    work_um_per_px = dapi_px / max(ds_chip, 1e-6)
    ctx = Context(xen_u8=xen_chip_clean, he_u8=cdx_chip_clean,
                       he_tissue_mask=cdx_mask_chip,
                       xen_tissue_mask=xen_mask_chip,
                       work_um_per_px=work_um_per_px)
    chip_origin = np.array([cx1, cy1], dtype=np.float64)
    return ctx, chip_origin, ds_chip, (cdx1, cdy1, cdx2_, cdy2_)


def assemble_refined_gray(out_canvas, cdx_dapi_full,
                                M_xen_to_cdx_local, cc_mask_full_bool):
    xen_h, xen_w = out_canvas.shape[:2]
    H_cdx, W_cdx = cdx_dapi_full.shape[:2]
    ys, xs = np.where(cc_mask_full_bool)
    bx1, bx2 = int(xs.min()), int(xs.max()) + 1
    by1, by2 = int(ys.min()), int(ys.max()) + 1
    cdx1f, cdy1f, cdx2f, cdy2f = project_xen_bbox(
        M_xen_to_cdx_local, bx1, by1, bx2, by2)
    margin = 64
    cdx1 = max(0, int(cdx1f) - margin); cdy1 = max(0, int(cdy1f) - margin)
    cdx2_ = min(W_cdx, int(cdx2f) + margin)
    cdy2_ = min(H_cdx, int(cdy2f) + margin)
    if cdx2_ <= cdx1 or cdy2_ <= cdy1:
        return
    cdx_sub = cdx_dapi_full[cdy1:cdy2_, cdx1:cdx2_]
    M_cdx_to_xen_local = invert_2x3_affine(M_xen_to_cdx_local)
    warped = warp_codex_crop_to_xenium(
        cdx_sub, cdx1, cdy1, M_cdx_to_xen_local, xen_h, xen_w)
    out_canvas[cc_mask_full_bool] = warped[cc_mask_full_bool]


def _build_pyramid_memmaps(level0_path, n_ch, h, w, dtype, tmp_dir, max_levels=8):
    levels = [(str(level0_path), (n_ch, h, w))]
    cur_path, cur_h, cur_w = str(level0_path), h, w
    while min(cur_h, cur_w) > 256 and len(levels) < max_levels:
        new_h, new_w = cur_h // 2, cur_w // 2
        if new_h < 256 or new_w < 256:
            break
        new_path = os.path.join(tmp_dir, f'lvl{len(levels)}.tmp')
        prev = np.memmap(cur_path, dtype=dtype, mode='r', shape=(n_ch, cur_h, cur_w))
        nxt = np.memmap(new_path, dtype=dtype, mode='w+', shape=(n_ch, new_h, new_w))
        for c in range(n_ch):
            nxt[c] = cv2.resize(np.asarray(prev[c]), (new_w, new_h),
                                interpolation=cv2.INTER_AREA)
        nxt.flush()
        del prev, nxt
        levels.append((new_path, (n_ch, new_h, new_w)))
        cur_path, cur_h, cur_w = new_path, new_h, new_w
    return levels


def _sanitize_channel_name(name):
    safe = ''.join(c if c.isalnum() or c in '-_' else '_' for c in str(name))
    safe = safe.strip('_') or 'unknown'
    return safe[:60]


def save_pyramidal_singlechannel_ome_tiff(arr, out_path, pixel_size_um,
                                                 channel_name,
                                                 tile_size=512,
                                                 compression='zstd',
                                                 compression_level=5,
                                                 maxworkers=4):
    if compression == 'zstd':
        try:
            import imagecodecs
            _ = imagecodecs.zstd_encode(b'probe', level=1)
        except Exception:
            compression = 'zlib'
            compression_level = 6
    pyramid = [arr]
    cur = arr
    while min(cur.shape[:2]) > 256:
        new_h, new_w = cur.shape[0] // 2, cur.shape[1] // 2
        if new_h < 256 or new_w < 256:
            break
        cur = cv2.resize(cur, (new_w, new_h), interpolation=cv2.INTER_AREA)
        pyramid.append(cur)
    options = dict(
        tile=(tile_size, tile_size),
        compression=compression,
        compressionargs={'level': compression_level},
        maxworkers=maxworkers,
        photometric='minisblack',
        metadata={'axes': 'YX',
                  'Channel': {'Name': [str(channel_name)]},
                  'PhysicalSizeX': float(pixel_size_um),
                  'PhysicalSizeXUnit': 'µm',
                  'PhysicalSizeY': float(pixel_size_um),
                  'PhysicalSizeYUnit': 'µm'},
    )
    with tifffile.TiffWriter(str(out_path), bigtiff=True, ome=True) as tif:
        tif.write(pyramid[0], subifds=len(pyramid) - 1, **options)
        for level in pyramid[1:]:
            tif.write(level, subfiletype=1, **options)


def save_pyramidal_multichannel_ome_tiff(levels, dtype, out_path,
                                                pixel_size_um, channel_names,
                                                tile_size=512, compression='zstd',
                                                compression_level=5, maxworkers=8):


    if compression == 'zstd':
        try:
            import imagecodecs
            _ = imagecodecs.zstd_encode(b'probe', level=1)
        except Exception as e:
            print(f'  zstd probe failed ({e}); falling back to zlib', flush=True)
            compression = 'zlib'
            compression_level = 6
    print(f'  Writing multi-channel pyramidal OME-TIFF → {out_path} '
            f'({len(levels)} levels, compression={compression}-{compression_level}, '
            f'maxworkers={maxworkers})', flush=True)
    options = dict(
        tile=(tile_size, tile_size),
        compression=compression,
        compressionargs={'level': compression_level},
        maxworkers=maxworkers,
        photometric='minisblack',
        metadata={'axes': 'CYX',
                  'Channel': {'Name': list(channel_names)},
                  'PhysicalSizeX': float(pixel_size_um),
                  'PhysicalSizeXUnit': 'µm',
                  'PhysicalSizeY': float(pixel_size_um),
                  'PhysicalSizeYUnit': 'µm'},
    )
    with tifffile.TiffWriter(str(out_path), bigtiff=True, ome=True) as tif:
        path0, shape0 = levels[0]
        arr0 = np.memmap(path0, dtype=dtype, mode='r', shape=shape0)
        tif.write(arr0, subifds=len(levels) - 1, **options)
        del arr0
        for path, shape in levels[1:]:
            arr = np.memmap(path, dtype=dtype, mode='r', shape=shape)
            tif.write(arr, subfiletype=1, **options)
            del arr
    sz_gb = Path(out_path).stat().st_size / (1024 ** 3)
    print(f'    {sz_gb:.2f} GB', flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--crop-dir', required=True,
                     help='CODEX alignment output directory of one field')
    ap.add_argument('--codex', default=None, help='CODEX qptiff override')
    ap.add_argument('--xenium', default=None, help='Xenium dir override')
    ap.add_argument('--search-um', type=float, default=150.0)
    ap.add_argument('--max-rot-deg', type=float, default=1.0)
    ap.add_argument('--scale-band', type=float, default=0.015)
    ap.add_argument('--low-ncc-threshold', type=float, default=0.50)
    ap.add_argument('--min-improve', type=float, default=0.03)
    ap.add_argument('--margin-xen-um', type=float, default=300.0)
    ap.add_argument('--margin-cdx-px', type=int, default=128)
    ap.add_argument('--work-size', type=int, default=2400)
    ap.add_argument('--overlay-max', type=int, default=2400)
    ap.add_argument('--benchmark-channels', type=int, default=0,
                     help='If >0, process only the first N channels (for IO/'
                          'compression benchmarking). 0 = all channels.')
    args = ap.parse_args()

    crop = Path(args.crop_dir)
    info = json.loads((crop / 'apply_alignment.json').read_text())
    align_dir = Path(info['source_alignment']).parent
    j = json.loads((align_dir / 'global_alignment.json').read_text())
    M_xen_to_cdx = np.array(j['M_xen_full_to_cdx_full'])
    M_cdx_to_xen = np.array(j['M_cdx_full_to_xen_full'])
    xen_h, xen_w = j['full_resolution']['xenium_shape_hw']
    xen_px = j['full_resolution']['xenium_px_um']

    codex_qptiff = args.codex or info.get('codex_qptiff')
    xenium_dir = args.xenium or info.get('xenium_dir')

    t0 = time.time()
    print(f'[{time.time()-t0:.0f}s] Loading CODEX DAPI: {codex_qptiff}',
            flush=True)
    cdx_dapi, cdx_px = load_codex_full_dapi(codex_qptiff)
    print(f'  shape={cdx_dapi.shape} dtype={cdx_dapi.dtype} px={cdx_px:.4f} µm',
            flush=True)

    print(f'[{time.time()-t0:.0f}s] Loading Xenium DAPI: {xenium_dir}',
            flush=True)
    dapi, dapi_px = load_dapi(xenium_path=str(xenium_dir))
    print(f'  shape={dapi.shape} px={dapi_px:.4f} µm', flush=True)

    print(f'[{time.time()-t0:.0f}s] Detecting Xenium tissue CCs', flush=True)
    union_mask_small, small_dapi, ds_xen, areas = detect_tissue_mask(
        dapi, dapi_px)
    n, labels, stats, _ = cv2.connectedComponentsWithStats(
        union_mask_small, connectivity=8)
    cc_indices = list(range(1, n))
    print(f'  ds_xen={ds_xen:.4f} small_shape={small_dapi.shape} '
            f'{len(cc_indices)} CCs', flush=True)

    margin_xen = max(50, int(args.margin_xen_um / dapi_px))

    per_cc_results = []
    for k, i in enumerate(cc_indices, 1):
        x_s = int(stats[i, cv2.CC_STAT_LEFT]); y_s = int(stats[i, cv2.CC_STAT_TOP])
        w_s = int(stats[i, cv2.CC_STAT_WIDTH]); h_s = int(stats[i, cv2.CC_STAT_HEIGHT])
        x1 = int(x_s / ds_xen); y1 = int(y_s / ds_xen)
        x2 = int((x_s + w_s) / ds_xen); y2 = int((y_s + h_s) / ds_xen)
        area_mm2 = float(stats[i, cv2.CC_STAT_AREA] * (dapi_px / ds_xen) ** 2 / 1e6)
        print(f'\n[{time.time()-t0:.0f}s] CC {k}/{len(cc_indices)} '
                f'area={area_mm2:.2f}mm² xen_bbox=({x1},{y1})→({x2},{y2})',
                flush=True)
        ctx, chip_origin, ds_chip, _ = build_cc_context(
            dapi, dapi_px, cdx_dapi, cdx_px,
            M_xen_to_cdx, M_cdx_to_xen, x1, y1, x2, y2,
            margin_xen, args.margin_cdx_px, work_size=args.work_size)
        search_radius_chip = max(1, int(args.search_um / dapi_px * ds_chip))
        print(f'  ctx chip={ctx.xen.shape} ds_chip={ds_chip:.4f} '
                f'search ±{search_radius_chip} chip-px '
                f'(±{args.search_um:.0f} µm full)', flush=True)
        params, score, ncc_id, mode = refine_per_cc(
            ctx, search_radius_xen_px=search_radius_chip,
            max_rot_deg=args.max_rot_deg, scale_band=args.scale_band,
            min_improve=args.min_improve,
            low_ncc_threshold=args.low_ncc_threshold)
        rot, scale, dx_s, dy_s = params
        dx_full = dx_s / max(ds_chip, 1e-6)
        dy_full = dy_s / max(ds_chip, 1e-6)
        print(f'  mode={mode} identity_score={ncc_id:.4f} → best_score={score:.4f}',
                flush=True)
        print(f'  resid: rot={rot:+.2f}° scale={scale:.4f} '
                f'dx={dx_full:+.1f} dy={dy_full:+.1f} (xen-full px '
                f'@ {dapi_px} µm/px)', flush=True)
        M_local = compose_local_M(M_xen_to_cdx, params, chip_origin, ds_chip)
        per_cc_results.append({
            'cc_idx': int(i),
            'area_mm2': area_mm2,
            'xen_bbox': [x1, y1, x2, y2],
            'mode': mode,
            'rot_deg': float(rot), 'scale': float(scale),
            'dx_xen_full_px': float(dx_full),
            'dy_xen_full_px': float(dy_full),
            'ds_chip': float(ds_chip),
            'score_id': float(ncc_id), 'score_best': float(score),
            'M_xen_full_to_cdx_full': M_local.tolist(),
            'cc_mask_small': labels == i,
        })


    s_xen_per_out = float(cdx_px) / float(xen_px)
    out_h = int(np.ceil(xen_h / s_xen_per_out))
    out_w = int(np.ceil(xen_w / s_xen_per_out))
    print(f'\n[{time.time()-t0:.0f}s] Option B canvas (CODEX-native): '
            f'{out_h}×{out_w} @ {cdx_px:.4f} µm/px '
            f'(xen×{1/s_xen_per_out:.3f}, scale_xen_per_out={s_xen_per_out:.4f})',
            flush=True)

    def compose_M_xen_to_out_frame(M_x2c, s):
        M2 = np.array(M_x2c, dtype=np.float64)
        M2[:2, :2] = M2[:2, :2] * s
        return M2


    print(f'[{time.time()-t0:.0f}s] Pre-computing per-CC out-resolution masks',
            flush=True)
    cc_out_masks = []
    cc_M_out_to_cdx = []
    for r in per_cc_results:
        cc_mask_small = r.pop('cc_mask_small').astype(np.uint8) * 255
        cc_mask_out = cv2.resize(cc_mask_small, (out_w, out_h),
                                       interpolation=cv2.INTER_NEAREST) > 127
        cc_out_masks.append(cc_mask_out if cc_mask_out.any() else None)
        M_xen_local = np.array(r['M_xen_full_to_cdx_full'])
        M_out_local = compose_M_xen_to_out_frame(M_xen_local, s_xen_per_out)
        cc_M_out_to_cdx.append(M_out_local)
        r['M_out_pixel_to_cdx_full'] = M_out_local.tolist()


    print(f'[{time.time()-t0:.0f}s] Building cc_label.tif', flush=True)
    cc_label_out = cv2.resize(labels.astype(np.int32), (out_w, out_h),
                                   interpolation=cv2.INTER_NEAREST)
    if cc_label_out.max() <= 255:
        cc_label_out = cc_label_out.astype(np.uint8)
    else:
        cc_label_out = cc_label_out.astype(np.uint16)
    tifffile.imwrite(str(crop / 'cc_label.tif'), cc_label_out,
                          photometric='minisblack', compression='zlib',
                          metadata={'axes': 'YX',
                                    'PhysicalSizeX': float(cdx_px),
                                    'PhysicalSizeXUnit': 'µm',
                                    'PhysicalSizeY': float(cdx_px),
                                    'PhysicalSizeYUnit': 'µm'})

    cdx_dtype = cdx_dapi.dtype
    del cdx_dapi

    n_channels, channel_names = codex_channel_info(codex_qptiff)
    print(f'[{time.time()-t0:.0f}s] CODEX has {n_channels} channels:', flush=True)
    for i, n in enumerate(channel_names):
        print(f'    ch{i:2d}: {n}', flush=True)
    if args.benchmark_channels > 0:
        n_channels = min(args.benchmark_channels, n_channels)
        channel_names = channel_names[:n_channels]
        print(f'[BENCHMARK] limiting to first {n_channels} channels', flush=True)


    tmp_root = crop / 'tmp_pyramid'
    tmp_root.mkdir(exist_ok=True)
    level0_path = tmp_root / 'lvl0.tmp'
    level0_shape = (n_channels, out_h, out_w)
    bytes_gb = (n_channels * out_h * out_w * np.dtype(cdx_dtype).itemsize) / (1024 ** 3)
    print(f'[{time.time()-t0:.0f}s] Level-0 memmap: {level0_shape} {cdx_dtype} '
            f'= {bytes_gb:.1f} GB → {level0_path}', flush=True)


    if level0_path.exists():
        print(f'  level0 memmap exists → resume mode', flush=True)
        level0 = np.memmap(level0_path, dtype=cdx_dtype, mode='r+',
                              shape=level0_shape)
    else:
        level0 = np.memmap(level0_path, dtype=cdx_dtype, mode='w+',
                              shape=level0_shape)

    for c in range(n_channels):
        sentinel = tmp_root / f'ch{c:02d}.done'
        if sentinel.exists():
            print(f'  [ch{c:2d}/{n_channels} {channel_names[c]}] SKIP (sentinel)',
                    flush=True)
            continue
        ch_t0 = time.time()
        print(f'\n[{time.time()-t0:.0f}s] [ch{c:2d}/{n_channels} {channel_names[c]}] '
                'loading + assembling', flush=True)
        cdx_full_c, _ = load_codex_full_channel(codex_qptiff, c)
        canvas = np.zeros((out_h, out_w), dtype=cdx_dtype)
        for cc_mask_out, M_out_local in zip(cc_out_masks, cc_M_out_to_cdx):
            if cc_mask_out is None:
                continue
            assemble_refined_gray(canvas, cdx_full_c, M_out_local, cc_mask_out)
        level0[c] = canvas


        level0.flush()
        sentinel.touch()
        del canvas, cdx_full_c
        print(f'  ch{c:2d} assembled in {time.time()-ch_t0:.1f}s', flush=True)

    level0.flush()
    del level0

    out_dir = crop / 'codex_aligned'
    out_dir.mkdir(exist_ok=True)
    print(f'\n[{time.time()-t0:.0f}s] Writing {n_channels} per-channel pyramidal '
          f'OME-TIFFs → {out_dir}', flush=True)
    level0_ro = np.memmap(level0_path, dtype=cdx_dtype, mode='r',
                                 shape=level0_shape)
    out_dapi = None
    per_channel_files = []
    for c in range(n_channels):
        name_safe = _sanitize_channel_name(channel_names[c])
        out_path = out_dir / f'ch{c:02d}-{name_safe}.ome.tif'
        ch_t0 = time.time()
        ch_arr = np.array(level0_ro[c])
        if c == 0:
            out_dapi = ch_arr.copy()
        save_pyramidal_singlechannel_ome_tiff(
            ch_arr, out_path,
            pixel_size_um=cdx_px,
            channel_name=channel_names[c],
            tile_size=512, compression='zstd',
            compression_level=5, maxworkers=4)
        sz_mb = out_path.stat().st_size / (1024 ** 2)
        per_channel_files.append(out_path.name)
        print(f'  [ch{c:2d}/{n_channels} {channel_names[c]}] {sz_mb:.1f} MB '
              f'({time.time()-ch_t0:.1f}s)', flush=True)
        del ch_arr
    del level0_ro

    print(f'[{time.time()-t0:.0f}s] Building fine_overlay.jpg', flush=True)


    xen_u8 = to_u8_perc(dapi)
    xen_u8_at_out = cv2.resize(xen_u8, (out_w, out_h),
                                   interpolation=cv2.INTER_AREA)
    fine_overlay = make_overlay_gray(out_dapi, xen_u8_at_out,
                                              target_long_dim=args.overlay_max)
    cv2.imwrite(str(crop / 'fine_overlay.jpg'), fine_overlay)
    del out_dapi, xen_u8_at_out


    print(f'\n[{time.time()-t0:.0f}s] Upsampling level0 ({n_channels}, {out_h}, '
            f'{out_w}) @ {cdx_px:.4f} µm/px → Xenium-native '
            f'({n_channels}, {xen_h}, {xen_w}) @ {xen_px:.4f} µm/px',
            flush=True)
    xe_level0_path = tmp_root / 'xe_lvl0.tmp'
    xe_bytes_gb = (n_channels * xen_h * xen_w *
                       np.dtype(cdx_dtype).itemsize) / (1024 ** 3)
    print(f'  xe_level0 memmap: {xe_bytes_gb:.1f} GB → {xe_level0_path}',
            flush=True)
    xe_level0 = np.memmap(xe_level0_path, dtype=cdx_dtype, mode='w+',
                              shape=(n_channels, xen_h, xen_w))
    level0_ro = np.memmap(level0_path, dtype=cdx_dtype, mode='r',
                                 shape=level0_shape)
    for c in range(n_channels):
        rt0 = time.time()
        xe_level0[c] = _xe_resize_to(np.asarray(level0_ro[c]),
                                            xen_h, xen_w,
                                            interp=cv2.INTER_LINEAR)
        print(f'  [ch{c:2d}/{n_channels} {channel_names[c]}] resized in '
                f'{time.time()-rt0:.1f}s', flush=True)
    xe_level0.flush()
    del xe_level0, level0_ro

    try:
        os.unlink(level0_path)
    except FileNotFoundError:
        pass

    print(f'\n[{time.time()-t0:.0f}s] Building Xenium-native pyramid memmaps',
            flush=True)
    pack_levels = _xe_build_pyramid_memmaps(
        xe_level0_path, n_channels, xen_h, xen_w, cdx_dtype, str(tmp_root))
    for path, shape in pack_levels:
        print(f'    pack level: {shape} → {path}', flush=True)
    _xe_write_multichannel_ome(
        pack_levels, cdx_dtype, crop / 'codex_aligned.ome.tif',
        pixel_size_um=xen_px, channel_names=channel_names,
        tile_size=512, compression='zlib', compression_level=6, maxworkers=8)
    for path, _ in pack_levels:
        try:
            os.unlink(path)
        except FileNotFoundError:
            pass


    try:
        os.unlink(level0_path)
    except FileNotFoundError:
        pass
    for sentinel in tmp_root.glob('ch*.done'):
        try:
            sentinel.unlink()
        except FileNotFoundError:
            pass
    try:
        tmp_root.rmdir()
    except OSError:
        pass

    for stale in ('codex_crop.ome.tif', 'codex_on_xenium.ome.tif',
                       'codex_on_xenium_bspline.ome.tif',
                       'codex_on_xenium_refined.ome.tif',
                       'bspline_displacement.npy'):
        p = crop / stale
        if p.exists():
            p.unlink()
            print(f'  cleaned intermediate: {stale}', flush=True)

    info['per_tissue_refinement'] = {
        'version': '1.0',
        'engine': 'refine_codex_per_tissue',
        'search_um': args.search_um,
        'max_rot_deg': args.max_rot_deg,
        'scale_band': args.scale_band,
        'low_ncc_threshold': args.low_ncc_threshold,
        'min_improve': args.min_improve,
        'n_ccs': len(per_cc_results),
        'per_cc': per_cc_results,
        'n_channels': int(n_channels),
        'channel_names': list(channel_names),
        'output_dir': 'codex_aligned',
        'output_layout': 'per_channel_files',
        'output_axes_per_channel': 'YX',
        'per_channel_files': per_channel_files,
        'per_channel_filename_pattern': 'ch{NN}-{marker}.ome.tif',
        'output_pixel_size_um': float(cdx_px),
        'output_shape_chw': [int(n_channels), int(out_h), int(out_w)],
        'output_resolution_choice': 'CODEX_native (Option B)',
        'scale_xen_per_out': s_xen_per_out,
        'overlap_policy': 'non-overlapping by connectedComponents partition construction',
        'cc_label_tif': 'cc_label.tif',
        'companion_files': ['cc_label.tif', 'fine_overlay.jpg',
                             'codex_aligned.ome.tif'],
        'fine_overlay_jpg': 'fine_overlay.jpg',
        'xenium_explorer_ome_tiff': 'codex_aligned.ome.tif',
        'xenium_explorer_ome_compression': 'zlib-6',
        'xenium_explorer_ome_shape_chw': [int(n_channels), int(xen_h),
                                              int(xen_w)],
        'xenium_explorer_ome_pixel_size_um': float(xen_px),
        'xenium_explorer_ome_pixel_grid': 'xenium_native_pixel_1_to_1',
        'validation_status': 'pending_A_vs_B_intensity_correlation',
        'compression': 'zstd-5',
        'elapsed_seconds': float(time.time() - t0),
    }
    (crop / 'apply_alignment.json').write_text(json.dumps(info, indent=2))
    print(f'\n[{time.time()-t0:.0f}s] DONE → {crop}', flush=True)


if __name__ == '__main__':
    main()
