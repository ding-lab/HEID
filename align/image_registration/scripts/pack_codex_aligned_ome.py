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


def build_pyramid_memmaps(level0_path, n_ch, h, w, dtype, tmp_dir, max_levels=8):
    levels = [(str(level0_path), (n_ch, h, w))]
    cur_path, cur_h, cur_w = str(level0_path), h, w
    while min(cur_h, cur_w) > 256 and len(levels) < max_levels:
        new_h, new_w = cur_h // 2, cur_w // 2
        if new_h < 256 or new_w < 256:
            break
        new_path = os.path.join(tmp_dir, f'lvl{len(levels)}.tmp')
        prev = np.memmap(cur_path, dtype=dtype, mode='r',
                            shape=(n_ch, cur_h, cur_w))
        nxt = np.memmap(new_path, dtype=dtype, mode='w+',
                           shape=(n_ch, new_h, new_w))
        for c in range(n_ch):
            nxt[c] = cv2.resize(np.asarray(prev[c]), (new_w, new_h),
                                   interpolation=cv2.INTER_AREA)
        nxt.flush()
        del prev, nxt
        levels.append((new_path, (n_ch, new_h, new_w)))
        cur_path, cur_h, cur_w = new_path, new_h, new_w
    return levels


def write_xenium_compatible_multichannel_ome(levels, dtype, out_path,
                                                  pixel_size_um, channel_names,
                                                  tile_size=512, compression='zlib',
                                                  compression_level=6,
                                                  maxworkers=8):
    if compression not in ('zlib', 'lzw', 'none'):
        raise ValueError(f'compression must be zlib/lzw/none, got {compression!r}')
    options = dict(
        tile=(tile_size, tile_size),
        compression=compression if compression != 'none' else None,
        maxworkers=maxworkers,
        photometric='minisblack',
        metadata={'axes': 'CYX',
                  'Channel': {'Name': list(channel_names)},
                  'PhysicalSizeX': float(pixel_size_um),
                  'PhysicalSizeXUnit': 'µm',
                  'PhysicalSizeY': float(pixel_size_um),
                  'PhysicalSizeYUnit': 'µm'},
    )
    if compression == 'zlib':
        options['compressionargs'] = {'level': compression_level}
    if options['compression'] is None:
        del options['compression']
    print(f'  Writing → {out_path} (compression={compression}, '
            f'{len(levels)} pyramid levels, tile={tile_size}, '
            f'maxworkers={maxworkers})', flush=True)
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


def _resize_to(arr, target_h, target_w, interp=cv2.INTER_LINEAR):
    LIMIT = 32000
    src_h, src_w = arr.shape
    if target_w <= LIMIT and target_h <= LIMIT:
        return cv2.resize(arr, (target_w, target_h), interpolation=interp)
    n_y = (target_h + LIMIT - 1) // LIMIT
    n_x = (target_w + LIMIT - 1) // LIMIT

    y_edges_t = [int(round(i * target_h / n_y)) for i in range(n_y + 1)]
    x_edges_t = [int(round(i * target_w / n_x)) for i in range(n_x + 1)]
    y_edges_s = [int(round(i * src_h    / n_y)) for i in range(n_y + 1)]
    x_edges_s = [int(round(i * src_w    / n_x)) for i in range(n_x + 1)]
    out = np.empty((target_h, target_w), dtype=arr.dtype)
    for i in range(n_y):
        ty1, ty2 = y_edges_t[i], y_edges_t[i + 1]
        sy1, sy2 = y_edges_s[i], y_edges_s[i + 1]
        for j in range(n_x):
            tx1, tx2 = x_edges_t[j], x_edges_t[j + 1]
            sx1, sx2 = x_edges_s[j], x_edges_s[j + 1]
            sub = arr[sy1:sy2, sx1:sx2]
            out[ty1:ty2, tx1:tx2] = cv2.resize(
                sub, (tx2 - tx1, ty2 - ty1), interpolation=interp)
    return out


def pack_from_per_channel_files(crop_dir, out_path=None, tmp_dir=None,
                                       compression='zlib', tile_size=512,
                                       maxworkers=8,
                                       target_shape_hw=None,
                                       target_pixel_size_um=None):
    crop = Path(crop_dir)
    info = json.loads((crop / 'apply_alignment.json').read_text())
    pt = info['per_tissue_refinement']
    channel_names = pt['channel_names']
    n_ch = int(pt['n_channels'])
    _, src_h, src_w = pt['output_shape_chw']
    src_h = int(src_h); src_w = int(src_w)
    src_pixel_size_um = float(pt['output_pixel_size_um'])
    per_channel_files = pt['per_channel_files']
    if len(per_channel_files) != n_ch:
        raise ValueError(f'per_channel_files length {len(per_channel_files)} != '
                              f'n_channels {n_ch}')

    if target_shape_hw is None:
        xen_shape = info.get('xenium_shape_hw') or pt.get('xenium_shape_hw')
        if xen_shape is None:
            raise ValueError('apply_alignment.json has no xenium_shape_hw; '
                                  'pass --target-shape-hw explicitly')
        target_shape_hw = (int(xen_shape[0]), int(xen_shape[1]))
    if target_pixel_size_um is None:
        target_pixel_size_um = float(info.get('xenium_px_um')
                                            or pt.get('xenium_px_um'))
    out_h, out_w = int(target_shape_hw[0]), int(target_shape_hw[1])
    do_resize = (out_h != src_h) or (out_w != src_w)

    out_path = Path(out_path) if out_path else (crop / 'codex_aligned.ome.tif')
    tmp_dir = Path(tmp_dir) if tmp_dir else (crop / 'tmp_pack')
    tmp_dir.mkdir(parents=True, exist_ok=True)

    src_dir = crop / 'codex_aligned'
    first = src_dir / per_channel_files[0]
    with tifffile.TiffFile(str(first)) as tf:
        s0 = tf.series[0]
        dtype = np.dtype(s0.dtype)
        if tuple(s0.shape) != (src_h, src_w):
            raise ValueError(f'shape mismatch: {s0.shape} vs ({src_h}, {src_w}) '
                                  f'in {first}')

    t0 = time.time()
    level0_path = tmp_dir / 'lvl0.tmp'
    bytes_gb = (n_ch * out_h * out_w * dtype.itemsize) / (1024 ** 3)
    src_bytes_gb = (n_ch * src_h * src_w * dtype.itemsize) / (1024 ** 3)
    print(f'[{time.time()-t0:.0f}s] Decoding {n_ch} channels '
            f'({src_h}×{src_w} @ {src_pixel_size_um} µm/px = {src_bytes_gb:.1f} GB) '
            f'→ {"upsample to " if do_resize else ""}level0 memmap '
            f'({out_h}×{out_w} @ {target_pixel_size_um} µm/px = '
            f'{bytes_gb:.1f} GB)', flush=True)
    print(f'  level0 memmap → {level0_path}', flush=True)
    level0 = np.memmap(level0_path, dtype=dtype, mode='w+',
                          shape=(n_ch, out_h, out_w))
    for c, fname in enumerate(per_channel_files):
        ct0 = time.time()
        fp = src_dir / fname
        with tifffile.TiffFile(str(fp)) as tf:
            arr = tf.series[0].levels[0].asarray()
        if arr.shape != (src_h, src_w):
            raise ValueError(f'{fp} level0 shape {arr.shape} '
                                  f'!= ({src_h}, {src_w})')
        if arr.dtype != dtype:
            raise ValueError(f'{fp} dtype {arr.dtype} != {dtype}')
        if do_resize:
            arr = _resize_to(arr, out_h, out_w, interp=cv2.INTER_LINEAR)
        level0[c] = arr
        del arr
        print(f'  [ch{c:2d}/{n_ch} {channel_names[c]}] done in '
                f'{time.time()-ct0:.1f}s', flush=True)
    level0.flush()
    del level0

    print(f'\n[{time.time()-t0:.0f}s] Building pyramid memmaps', flush=True)
    levels = build_pyramid_memmaps(level0_path, n_ch, out_h, out_w,
                                          dtype, str(tmp_dir))
    for path, shape in levels:
        print(f'  level: {shape} → {path}', flush=True)

    print(f'\n[{time.time()-t0:.0f}s] Writing Xenium Explorer–compatible '
            f'multi-channel OME-TIFF', flush=True)
    write_xenium_compatible_multichannel_ome(
        levels, dtype, out_path, target_pixel_size_um, channel_names,
        tile_size=tile_size, compression=compression, maxworkers=maxworkers)

    for path, _ in levels:
        try:
            os.unlink(path)
        except FileNotFoundError:
            pass
    try:
        tmp_dir.rmdir()
    except OSError:
        pass
    print(f'\n[{time.time()-t0:.0f}s] DONE → {out_path}', flush=True)
    return out_path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--crop-dir', required=True,
                     help='Sample dir containing apply_alignment.json + '
                          'codex_aligned/')
    ap.add_argument('--out', default=None,
                     help='Output OME-TIFF (default: '
                          '<crop-dir>/codex_aligned.ome.tif — overwrites)')
    ap.add_argument('--tmp-dir', default=None,
                     help='Tmp dir for memmaps (default: <crop-dir>/tmp_pack/)')
    ap.add_argument('--compression', default='zlib',
                     choices=['zlib', 'lzw', 'none'],
                     help='Xenium Explorer–supported codec (default zlib)')
    ap.add_argument('--tile-size', type=int, default=512)
    ap.add_argument('--maxworkers', type=int, default=8)
    ap.add_argument('--codex-native', action='store_true',
                     help='Keep CODEX-native pixel grid instead of upsampling '
                          'to Xenium pixel grid (the older layout — needs '
                          'Xenium Explorer alignment CSV).')
    args = ap.parse_args()
    kwargs = dict(out_path=args.out, tmp_dir=args.tmp_dir,
                  compression=args.compression, tile_size=args.tile_size,
                  maxworkers=args.maxworkers)
    if args.codex_native:
        import json as _json
        info = _json.loads(
            (Path(args.crop_dir) / 'apply_alignment.json').read_text())
        pt = info['per_tissue_refinement']
        _, h, w = pt['output_shape_chw']
        kwargs['target_shape_hw'] = (int(h), int(w))
        kwargs['target_pixel_size_um'] = float(pt['output_pixel_size_um'])
    pack_from_per_channel_files(args.crop_dir, **kwargs)


if __name__ == '__main__':
    main()
