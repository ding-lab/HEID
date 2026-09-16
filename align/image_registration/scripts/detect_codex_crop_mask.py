#!/usr/bin/env python3
from __future__ import annotations
import argparse
import json
import os
import sys
import time
import xml.etree.ElementTree as ET
from pathlib import Path

os.environ['PYTHONUNBUFFERED'] = '1'
import cv2
import numpy as np
import tifffile

SCRIPTS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPTS_DIR))
from pack_pipeline import load_dapi, normalize_dapi
from detect_xenium_mask import detect_tissue_mask
from detect_he_crop_mask import (
    project_xen_mask_to_he as project_xen_mask,
    geodesic_dilate, downsample_long,
)


def load_codex_full_dapi(codex_qptiff):
    return load_codex_full_channel(codex_qptiff, 0)


def load_codex_full_channel(codex_qptiff, channel_idx):
    with tifffile.TiffFile(str(codex_qptiff)) as tif:
        s = tif.series[0]
        lvl0 = s.levels[0]
        n_ch = lvl0.shape[0] if lvl0.ndim == 3 else 1
        if not (0 <= channel_idx < n_ch):
            raise ValueError(f'channel_idx={channel_idx} out of range [0,{n_ch})')
        if lvl0.ndim == 3 and lvl0.shape[0] > 1:
            ch = lvl0.asarray(key=channel_idx)
        else:
            ch = lvl0.asarray()
        full_px = None
        if tif.ome_metadata:
            try:
                root = ET.fromstring(tif.ome_metadata)
                ns = {'ome': 'http://www.openmicroscopy.org/Schemas/OME/2016-06'}
                px = root.find('.//ome:Pixels', ns)
                if px is not None and px.get('PhysicalSizeX') is not None:
                    full_px = float(px.get('PhysicalSizeX'))
            except Exception:
                pass
    if full_px is None or full_px <= 0:
        full_px = 0.5095
    return ch, full_px


def codex_channel_info(codex_qptiff):
    import re
    with tifffile.TiffFile(str(codex_qptiff)) as tif:
        s = tif.series[0]
        n_ch = s.shape[0] if s.axes.startswith('C') else 1
        names = []
        for i in range(n_ch):
            d = tif.pages[i].description or ''
            m = re.search(r'<Biomarker>([^<]+)</Biomarker>', d)
            names.append(m.group(1) if m else f'ch{i}')
    return n_ch, names


def detect_codex_tissue_mask(codex_dapi, codex_px_um,
                                  work_size: int = 2400,
                                  bright_otsu_scale: float = 0.10,
                                  min_area_mm2: float = 0.05) -> tuple:
    ds = min(1.0, work_size / max(codex_dapi.shape[:2]))
    small = cv2.resize(codex_dapi, None, fx=ds, fy=ds,
                        interpolation=cv2.INTER_AREA) if ds < 0.95 else codex_dapi
    norm = normalize_dapi(small)
    otsu_t, _ = cv2.threshold(norm, 0, 255,
                                  cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    t = max(5, int(otsu_t * bright_otsu_scale))
    _, base = cv2.threshold(norm, t, 255, cv2.THRESH_BINARY)
    from scipy.ndimage import binary_fill_holes
    base = (binary_fill_holes(base > 0).astype(np.uint8)) * 255
    n, labels, stats, _ = cv2.connectedComponentsWithStats(base, connectivity=8)
    um2_per_px = (codex_px_um / ds) ** 2
    min_area_px = int(min_area_mm2 * 1e6 / um2_per_px)
    out = np.zeros_like(base)
    for i in range(1, n):
        if int(stats[i, cv2.CC_STAT_AREA]) >= min_area_px:
            out[labels == i] = 255
    return out, small, ds


def compute_codex_crop_mask(M_xen_full_to_cdx_full, dapi, dapi_px,
                                cdx_dapi, cdx_px_um,
                                cdx_work_size: int = 2400,
                                bright_otsu_scale: float = 0.10,
                                cdx_min_area_mm2: float = 0.05,
                                max_extend_um: float = 150.0) -> dict:
    xen_mask, xen_small, ds_xen, xen_areas = detect_tissue_mask(dapi, dapi_px)
    cdx_mask, cdx_small, ds_cdx = detect_codex_tissue_mask(
        cdx_dapi, cdx_px_um, work_size=cdx_work_size,
        bright_otsu_scale=bright_otsu_scale,
        min_area_mm2=cdx_min_area_mm2)
    proj = project_xen_mask(xen_mask, M_xen_full_to_cdx_full,
                                ds_xen, ds_cdx, cdx_mask.shape)
    seed = cv2.bitwise_and(proj, cdx_mask)
    radius_px = max(0, int(max_extend_um / (cdx_px_um / ds_cdx)))
    final = geodesic_dilate(seed, cdx_mask, radius_px)
    return {
        'mask_small': final,
        'cdx_small': cdx_small,
        'cdx_tissue_small': cdx_mask,
        'ds_cdx': ds_cdx,
        'xen_mask_small': xen_mask,
        'xen_small': xen_small,
        'ds_xen': ds_xen,
        'xen_areas_mm2': xen_areas,
        'projection_small': proj,
        'extend_radius_px_small': radius_px,
        'max_extend_um': max_extend_um,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--align', required=True,
                     help='Directory containing CODEX global_alignment.json')
    ap.add_argument('--output', required=True, help='Flat output dir')
    ap.add_argument('--name', required=True, help='Sample short name (S1..)')
    ap.add_argument('--codex', default=None,
                     help='Override CODEX qptiff path')
    ap.add_argument('--xenium', default=None, help='Override Xenium dir')
    ap.add_argument('--max-extend-um', type=float, default=150.0)
    ap.add_argument('--cdx-work-size', type=int, default=2400)
    ap.add_argument('--debug-max', type=int, default=2400)
    args = ap.parse_args()

    out = Path(args.output); out.mkdir(parents=True, exist_ok=True)
    j = json.loads((Path(args.align) / 'global_alignment.json').read_text())
    M_xen_to_cdx = np.array(j['M_xen_full_to_cdx_full'])
    xen_dir = args.xenium or j['inputs'].get('xenium_dir')
    cdx_qptiff = args.codex or j['inputs'].get('codex_qptiff')

    t0 = time.time()
    print(f'[{time.time()-t0:.0f}s] Loading Xenium DAPI', flush=True)
    dapi, dapi_px = load_dapi(xenium_path=str(xen_dir))
    print(f'  shape={dapi.shape} px={dapi_px:.4f} µm', flush=True)
    print(f'[{time.time()-t0:.0f}s] Loading CODEX DAPI: {cdx_qptiff}',
            flush=True)
    cdx_dapi, cdx_px = load_codex_full_dapi(cdx_qptiff)
    print(f'  shape={cdx_dapi.shape} px={cdx_px:.4f} µm', flush=True)

    print(f'[{time.time()-t0:.0f}s] Computing CODEX crop mask', flush=True)
    r = compute_codex_crop_mask(M_xen_to_cdx, dapi, dapi_px, cdx_dapi, cdx_px,
                                      cdx_work_size=args.cdx_work_size,
                                      max_extend_um=args.max_extend_um)
    print(f'  Xenium: {len(r["xen_areas_mm2"])} biopsies, '
            f'areas={r["xen_areas_mm2"]} mm²', flush=True)
    print(f'  CODEX geodesic extend: {args.max_extend_um:.0f} µm '
            f'= {r["extend_radius_px_small"]} small px', flush=True)
    print(f'  final mask area = {100*r["mask_small"].mean()/255:.1f}% of small frame',
            flush=True)

    norm = normalize_dapi(r['cdx_small'])
    overlay = cv2.cvtColor(norm, cv2.COLOR_GRAY2BGR)
    contours, _ = cv2.findContours(r['mask_small'], cv2.RETR_EXTERNAL,
                                       cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(overlay, contours, -1, (0, 255, 0), 3)
    out_path = out / f'{args.name}_codex_crop_overlay.png'
    cv2.imwrite(str(out_path), downsample_long(overlay, args.debug_max))
    print(f'[{time.time()-t0:.0f}s] DONE → {out_path}', flush=True)


if __name__ == '__main__':
    main()
