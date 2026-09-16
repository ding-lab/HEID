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
from scipy.optimize import minimize


DEFAULT_CODEX_PX_UM = 0.5095
WORK_MAX_DIM = 1800


SILH_GATE_TOL = 0.05
NCC_ABSTAIN = 0.15


def get_ome_pixel_size(tif_path):
    with tifffile.TiffFile(str(tif_path)) as tif:
        if tif.ome_metadata:
            try:
                root = ET.fromstring(tif.ome_metadata)
                ns = {'ome': 'http://www.openmicroscopy.org/Schemas/OME/2016-06'}
                px = root.find('.//ome:Pixels', ns)
                if px is not None and px.get('PhysicalSizeX') is not None:
                    return float(px.get('PhysicalSizeX'))
            except Exception:
                pass
        page = tif.pages[0]
        res = page.tags.get('XResolution')
        unit = page.tags.get('ResolutionUnit')
        if res is not None and unit is not None:
            num, den = res.value
            if num > 0 and den > 0:
                ppu = num / den
                if unit.value == 3:   return 1e4 / ppu
                elif unit.value == 2: return 25400 / ppu
    return None


def load_xenium_dapi(xenium_dir):
    xen = Path(xenium_dir)
    focus_dir = xen / 'morphology_focus'
    if focus_dir.is_dir():
        files = sorted(focus_dir.glob('morphology_focus_*.ome.tif'))
        if files:
            print(f'  Loading {len(files)} Xenium DAPI z-planes (max projection)…',
                  flush=True)
            with tifffile.TiffFile(str(files[0])) as t:
                dapi = t.pages[0].asarray()
            for fp in files[1:]:
                with tifffile.TiffFile(str(fp)) as t:
                    np.maximum(dapi, t.pages[0].asarray(), out=dapi)
            px = get_ome_pixel_size(files[0]) or 0.2125
            return dapi, px
    for cand in (xen / 'morphology_focus.ome.tif', xen / 'morphology.ome.tif'):
        if cand.exists():
            with tifffile.TiffFile(str(cand)) as t:
                dapi = t.pages[0].asarray()
                if dapi.ndim == 3:
                    dapi = dapi.max(axis=0)
            px = get_ome_pixel_size(cand) or 0.2125
            return dapi, px
    raise FileNotFoundError(f'No Xenium DAPI under {xenium_dir}')


def load_codex_dapi(codex_qptiff, target_max_dim=WORK_MAX_DIM,
                     px_override=None):
    qp = Path(codex_qptiff)
    with tifffile.TiffFile(str(qp)) as tif:
        s = tif.series[0]
        levels = list(s.levels)
        chosen_idx = 0
        for i, lvl in enumerate(levels):
            shp = lvl.shape
            if max(shp[-2], shp[-1]) >= target_max_dim:
                chosen_idx = i
        chosen = levels[chosen_idx]
        if chosen.ndim == 3 and chosen.shape[0] > 1:
            img = chosen.asarray(key=0)
        else:
            img = chosen.asarray()
        full_h, full_w = levels[0].shape[-2], levels[0].shape[-1]

    full_px = px_override or get_ome_pixel_size(qp) or DEFAULT_CODEX_PX_UM
    lvl_max = max(chosen.shape[-2], chosen.shape[-1])
    full_max = max(full_h, full_w)
    lvl_px = full_px * (full_max / lvl_max)
    return img, lvl_px, chosen_idx, (full_h, full_w)


def to_u8(img):
    if img.dtype == np.uint8: return img
    fg = img[img > 0] if (img > 0).any() else img.ravel()
    p2, p98 = np.percentile(fg, [2, 98])
    return np.clip((img - p2) / (p98 - p2 + 1e-8) * 255, 0, 255).astype(np.uint8)


def normalize_dapi(img):
    if img.dtype != np.uint8:
        p2, p98 = np.percentile(img, [2, 98])
        u8 = np.clip((img - p2) / (p98 - p2 + 1e-8) * 255,
                       0, 255).astype(np.uint8)
    else:
        u8 = img
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    return clahe.apply(u8)


def downsample_to_um(img, src_px_um, target_px_um):
    if src_px_um >= target_px_um * 0.95:
        return img, src_px_um
    factor = src_px_um / target_px_um
    new_h = max(1, int(round(img.shape[0] * factor)))
    new_w = max(1, int(round(img.shape[1] * factor)))
    return cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA), target_px_um


def tissue_centroid(img_u8, threshold=8):
    mask = img_u8 > threshold
    if mask.sum() == 0:
        return np.array([img_u8.shape[1] / 2, img_u8.shape[0] / 2], dtype=np.float64)
    ys, xs = np.where(mask)
    return np.array([xs.mean(), ys.mean()], dtype=np.float64)


def affine_from_params(rot_deg, scale, dx, dy):
    th = np.deg2rad(rot_deg)
    c, s = np.cos(th), np.sin(th)
    A = scale * np.array([[c, -s], [s, c]])
    M = np.zeros((2, 3), dtype=np.float64)
    M[:2, :2] = A
    M[:, 2] = [dx, dy]
    return M


def masked_ncc(a, b, mask, min_overlap=5000):
    n = int(mask.sum())
    if n < min_overlap: return -1.0
    av = a[mask]; bv = b[mask]
    av = av - av.mean(); bv = bv - bv.mean()
    den = np.sqrt((av * av).sum()) * np.sqrt((bv * bv).sum())
    if den < 1e-6: return -1.0
    return float((av * bv).sum() / den)


def _otsu_tissue_mask(dapi_u8):
    _, mask = cv2.threshold(dapi_u8, 0, 255,
                              cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((7, 7), np.uint8))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN,  np.ones((3, 3), np.uint8))
    return mask > 0


class Context:
    def __init__(self, xen_u8, cdx_u8, work_um_per_px):
        self.xen = xen_u8
        self.cdx = cdx_u8
        self.cdx_f = cdx_u8.astype(np.float32)
        self.h, self.w = cdx_u8.shape[:2]
        self.xen_h, self.xen_w = xen_u8.shape[:2]


        xen_mask  = _otsu_tissue_mask(xen_u8)
        cdx_mask  = _otsu_tissue_mask(cdx_u8)
        self.xen_mask = xen_mask
        self.cdx_mask = cdx_mask
        self.xen_mask_u8 = xen_mask.astype(np.uint8) * 255
        self.cdx_mask_u8 = cdx_mask.astype(np.uint8) * 255
        self.xen_tissue_total = int(xen_mask.sum())
        self.um_per_px = work_um_per_px
        self.xen_centroid = tissue_centroid(xen_u8)
        self.cdx_centroid = tissue_centroid(cdx_u8)


def reverse_iou_metric(M, ctx):
    M_inv = cv2.invertAffineTransform(M)
    cdx_in_xen = cv2.warpAffine(ctx.cdx_mask_u8, M_inv,
                                  (ctx.xen_w, ctx.xen_h),
                                  flags=cv2.INTER_NEAREST, borderValue=0)
    cdx_bool = cdx_in_xen > 127
    xen_bool = ctx.xen_mask
    inter = int(np.logical_and(cdx_bool, xen_bool).sum())
    union = int(np.logical_or(cdx_bool,  xen_bool).sum())
    return inter / max(union, 1)


def silhouette_combine(forward_cov, reverse_iou):
    return min(forward_cov, reverse_iou)


def evaluate(params, ctx, return_warped=False):
    rot, scale, dx, dy = params


    if scale < 0.95 or scale > 1.05:
        if return_warped:
            return -2.0, np.zeros((ctx.h, ctx.w), dtype=np.uint8)
        return -2.0
    M = affine_from_params(rot, scale, dx, dy).astype(np.float32)
    warped = cv2.warpAffine(ctx.xen, M, (ctx.w, ctx.h),
                              flags=cv2.INTER_LINEAR, borderValue=0)
    warped_mask = cv2.warpAffine(ctx.xen_mask_u8, M, (ctx.w, ctx.h),
                                    flags=cv2.INTER_NEAREST,
                                    borderValue=0).astype(bool)
    overlap = ctx.cdx_mask & warped_mask
    n_warp = int(warped_mask.sum())
    n_over = int(overlap.sum())
    coverage = min(1.0, n_over / max(ctx.xen_tissue_total, 1))
    rev_iou = reverse_iou_metric(M, ctx)
    silhouette = silhouette_combine(coverage, rev_iou)
    if n_warp == 0 or n_over < 5000:

        if return_warped: return -1.5, warped
        return -1.5
    ncc_raw = masked_ncc(ctx.cdx_f, warped.astype(np.float32), overlap)
    s_violation = max(0.0, abs(scale - 1.0) - 0.02)
    ncc_clean = ncc_raw - 10.0 * s_violation
    score = ncc_clean + 2.0 * silhouette
    if return_warped: return score, warped
    return score


def shape_match_seeds(ctx, xen_mask, cdx_mask,
                         scale_mults=(0.85, 0.90, 0.95, 1.00, 1.05, 1.10, 1.15),
                         top_per_combo=2):
    cdx_mask_u8 = (cdx_mask * 255).astype(np.uint8)
    cdx_h, cdx_w = cdx_mask.shape
    xen_h0, xen_w0 = xen_mask.shape

    seeds = []
    for rot in (0.0, 90.0, 180.0, 270.0):
        for sm in scale_mults:
            M_rs = cv2.getRotationMatrix2D((xen_w0 / 2, xen_h0 / 2), rot, sm)
            xen_rs = cv2.warpAffine(
                (xen_mask * 255).astype(np.uint8), M_rs, (xen_w0, xen_h0),
                flags=cv2.INTER_NEAREST, borderValue=0)
            ys, xs = np.where(xen_rs > 127)
            if len(ys) < 1000:
                continue
            y_lo, y_hi = ys.min(), ys.max() + 1
            x_lo, x_hi = xs.min(), xs.max() + 1
            xen_crop = xen_rs[y_lo:y_hi, x_lo:x_hi]
            ch, cw = xen_crop.shape
            if ch >= cdx_h - 5 or cw >= cdx_w - 5:
                continue
            try:
                res = cv2.matchTemplate(cdx_mask_u8, xen_crop,
                                          cv2.TM_CCOEFF_NORMED)
            except cv2.error:
                continue
            res_w = res.copy()
            for _ in range(top_per_combo):
                _, max_val, _, max_loc = cv2.minMaxLoc(res_w)
                if max_val < -0.5:
                    break
                px, py = max_loc
                place = np.zeros((cdx_h, cdx_w), dtype=bool)
                x0 = max(0, px); x1 = min(cdx_w, px + cw)
                y0 = max(0, py); y1 = min(cdx_h, py + ch)
                sx0 = max(0, -px); sy0 = max(0, -py)
                place[y0:y1, x0:x1] = xen_crop[sy0:sy0+(y1-y0),
                                                  sx0:sx0+(x1-x0)] > 127
                inter = (place & cdx_mask).sum()
                xen_total = place.sum()
                if xen_total > 0:
                    coverage = inter / xen_total
                    full_tl_x = px - x_lo
                    full_tl_y = py - y_lo
                    th = np.deg2rad(rot)
                    cos_t, sin_t = np.cos(th), np.sin(th)
                    R = sm * np.array([[cos_t, -sin_t], [sin_t, cos_t]])
                    center = np.array([xen_w0 / 2, xen_h0 / 2])
                    t_origin = (center - R @ center
                                + np.array([full_tl_x, full_tl_y]))
                    params = (rot, sm,
                              float(t_origin[0]), float(t_origin[1]))
                    seeds.append((params, float(coverage)))
                ys_s = max(0, py - 30); ys_e = min(res_w.shape[0], py + 30)
                xs_s = max(0, px - 30); xs_e = min(res_w.shape[1], px + 30)
                res_w[ys_s:ys_e, xs_s:xs_e] = -np.inf

    seeds.sort(key=lambda x: -x[1])
    return seeds


def evaluate_with_coverage(params, ctx):
    rot, scale, dx, dy = params
    if scale < 0.95 or scale > 1.05:
        return -2.0, -2.0, 0.0
    M = affine_from_params(rot, scale, dx, dy).astype(np.float32)
    warped = cv2.warpAffine(ctx.xen, M, (ctx.w, ctx.h),
                              flags=cv2.INTER_LINEAR, borderValue=0)
    warped_mask = cv2.warpAffine(ctx.xen_mask_u8, M, (ctx.w, ctx.h),
                                    flags=cv2.INTER_NEAREST,
                                    borderValue=0).astype(bool)
    overlap = ctx.cdx_mask & warped_mask
    n_warp = int(warped_mask.sum())
    n_over = int(overlap.sum())
    coverage = min(1.0, n_over / max(ctx.xen_tissue_total, 1))
    rev_iou = reverse_iou_metric(M, ctx)
    silhouette = silhouette_combine(coverage, rev_iou)
    if n_warp == 0 or n_over < 5000:
        return -1.5, -1.5, silhouette
    ncc_raw = masked_ncc(ctx.cdx_f, warped.astype(np.float32), overlap)
    s_violation = max(0.0, abs(scale - 1.0) - 0.02)
    ncc_clean = ncc_raw - 10.0 * s_violation
    score = ncc_clean + 2.0 * silhouette
    return score, ncc_clean, silhouette


def rotate_xenium(xen_u8, rot_deg, scale=1.0):
    h, w = xen_u8.shape[:2]
    M = cv2.getRotationMatrix2D((w / 2, h / 2), rot_deg, scale)
    return cv2.warpAffine(xen_u8, M, (w, h),
                            flags=cv2.INTER_LINEAR, borderValue=0)


def top_k_translations_via_fft(moving_u8, fixed_u8, top_k=5,
                                  suppress_radius=40):
    fh, fw = fixed_u8.shape[:2]
    mh, mw = moving_u8.shape[:2]
    pad_h = fh + mh
    pad_w = fw + mw

    f_mask = (fixed_u8 > 8).astype(np.float32)
    m_mask = (moving_u8 > 8).astype(np.float32)
    f_mean = (fixed_u8.astype(np.float32) * f_mask).sum() / max(f_mask.sum(), 1)
    m_mean = (moving_u8.astype(np.float32) * m_mask).sum() / max(m_mask.sum(), 1)
    f_z = (fixed_u8.astype(np.float32) - f_mean) * f_mask
    m_z = (moving_u8.astype(np.float32) - m_mean) * m_mask

    F = np.zeros((pad_h, pad_w), dtype=np.float32)
    M = np.zeros((pad_h, pad_w), dtype=np.float32)
    F[:fh, :fw] = f_z
    M[:mh, :mw] = m_z
    Ff = np.fft.rfft2(F)
    Mf = np.fft.rfft2(M)
    corr = np.fft.irfft2(Ff * np.conj(Mf), s=(pad_h, pad_w)).copy()

    peaks = []
    for _ in range(top_k):
        idx = int(np.argmax(corr))
        peak_val = float(corr.flat[idx])
        if peak_val <= -1e30: break
        py = idx // pad_w
        px = idx % pad_w
        py_w = py - pad_h if py > pad_h // 2 else py
        px_w = px - pad_w if px > pad_w // 2 else px
        peaks.append((int(px_w), int(py_w), peak_val))

        y0 = max(0, py - suppress_radius); y1 = min(pad_h, py + suppress_radius)
        x0 = max(0, px - suppress_radius); x1 = min(pad_w, px + suppress_radius)
        corr[y0:y1, x0:x1] = -np.inf
    return peaks


def fft_seeds_for_rotation(ctx, rot_deg, expected_scale, top_k=5):
    moving = rotate_xenium(ctx.xen, rot_deg, expected_scale)
    peaks = top_k_translations_via_fft(moving, ctx.cdx, top_k=top_k)
    h, w = ctx.xen.shape[:2]
    th = np.deg2rad(rot_deg)
    c_, s_ = np.cos(th), np.sin(th)
    R = expected_scale * np.array([[c_, -s_], [s_, c_]])
    center = np.array([w / 2, h / 2])
    out = []
    for px, py, _ in peaks:
        t = center - R @ center + np.array([px, py])
        params = (rot_deg, expected_scale, float(t[0]), float(t[1]))
        ncc = evaluate(params, ctx)
        out.append((params, ncc))
    out.sort(key=lambda x: -x[1])
    return out


def powell(seed, ctx, maxiter=120):
    res = minimize(lambda p: -evaluate(p, ctx), np.array(seed),
                    method='Powell',
                    options={'xtol': 1e-3, 'ftol': 1e-4,
                             'maxiter': maxiter, 'disp': False})
    return -res.fun, res.x


def basin_hop(p0, ctx, n_epochs=60, patience=15, seed=42, min_silh=None):
    rng = np.random.default_rng(seed)
    base_sigma = np.array([2.0, 0.02, ctx.w * 0.05, ctx.h * 0.05])
    best_p = np.array(p0); best_ncc = evaluate(best_p, ctx)
    no_improve = 0; sigma_mult = 1.0
    history = []
    for epoch in range(1, n_epochs + 1):
        p_start = best_p + rng.normal(0.0, base_sigma * sigma_mult)
        ncc, p_e = powell(p_start, ctx, maxiter=80)
        silh_rej = False
        if min_silh is not None:
            _, _, silh_e = evaluate_with_coverage(p_e, ctx)
            if silh_e < min_silh:
                silh_rej = True
        improved = (not silh_rej) and ncc > best_ncc + 1e-5
        if improved:
            delta = ncc - best_ncc
            best_p = p_e.copy(); best_ncc = ncc
            no_improve = 0; sigma_mult = 1.0
            print(f'    epoch {epoch:3d}: NCC={best_ncc:.4f} (+{delta:.4f}) '
                  f'rot={best_p[0]:.2f}° sc={best_p[1]:.4f} '
                  f't=({best_p[2]:.0f},{best_p[3]:.0f})', flush=True)
        else:
            no_improve += 1
            if silh_rej and no_improve % 5 == 0:
                print(f'    epoch {epoch:3d}: silh-reject (silh<{min_silh:.3f}) '
                      f'NCC={best_ncc:.4f}', flush=True)
            elif no_improve % 5 == 0:
                print(f'    epoch {epoch:3d}: no-improve {no_improve}/{patience} '
                      f'NCC={best_ncc:.4f} sigma×{sigma_mult:.1f}', flush=True)
            if no_improve % 5 == 0 and sigma_mult < 4.0:
                sigma_mult = min(sigma_mult * 1.5, 4.0)
        history.append({'epoch': epoch, 'ncc': float(ncc),
                         'best_ncc': float(best_ncc),
                         'improved': bool(improved),
                         'silh_rej': bool(silh_rej),
                         'sigma_mult': float(sigma_mult)})
        if no_improve >= patience:
            print(f'    epoch {epoch}: patience exhausted ({patience}), stop',
                  flush=True)
            break
    return best_p, best_ncc, history


def fine_climb(p0, ctx, max_iter=200):
    base_steps = np.array([0.05, 0.0005, 1.5, 1.5])
    min_steps = base_steps * 0.05
    steps = base_steps.copy()
    best_p = np.array(p0); best_ncc = evaluate(best_p, ctx)
    moves = []
    for axis in range(4):
        for sgn in (+1, -1):
            v = np.zeros(4); v[axis] = sgn
            moves.append(v)
    no_improve = 0
    for _ in range(max_iter):
        best_mv = None; best_mv_ncc = best_ncc
        for mv in moves:
            v = evaluate(best_p + mv * steps, ctx)
            if v > best_mv_ncc + 1e-5:
                best_mv = mv; best_mv_ncc = v
        if best_mv is not None:
            best_p = best_p + best_mv * steps
            best_ncc = best_mv_ncc; no_improve = 0
        else:
            no_improve += 1
            if no_improve >= 8:
                ns = np.maximum(steps * 0.5, min_steps)
                if np.all(ns == steps): break
                steps = ns; no_improve = 0
    return best_p, best_ncc


def render_overlay(params, ctx, header):
    ncc, warped = evaluate(params, ctx, return_warped=True)
    out = np.zeros((ctx.h, ctx.w, 3), dtype=np.uint8)
    out[..., 2] = warped
    out[..., 1] = ctx.cdx
    cv2.rectangle(out, (0, 0), (ctx.w, 28), (0, 0, 0), -1)
    cv2.putText(out, header, (8, 20),
                 cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
    return out, ncc


def align(xenium_dir, codex_qptiff, output_dir,
            codex_px_override=None, work_max=WORK_MAX_DIM,
            n_epochs=60, patience=15, seed=42):
    output_dir = Path(output_dir); output_dir.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    print(f'[{time.time()-t0:.0f}s] Loading CODEX DAPI', flush=True)
    cdx_full, cdx_px, cdx_level, cdx_full_shape = load_codex_dapi(
        codex_qptiff, target_max_dim=work_max, px_override=codex_px_override)
    print(f'  CODEX level {cdx_level}: shape={cdx_full.shape} '
          f'px={cdx_px:.4f} μm', flush=True)

    print(f'[{time.time()-t0:.0f}s] Loading Xenium DAPI', flush=True)
    xen_full, xen_px = load_xenium_dapi(xenium_dir)
    print(f'  Xenium shape={xen_full.shape} px={xen_px:.4f} μm', flush=True)

    work_um = cdx_px
    xen_work, xen_work_px = downsample_to_um(xen_full, xen_px, work_um)

    cdx_u8 = normalize_dapi(cdx_full)
    xen_u8 = normalize_dapi(xen_work)
    ctx = Context(xen_u8, cdx_u8, work_um)
    print(f'  work canvas: CODEX={cdx_u8.shape} Xenium={xen_u8.shape} '
          f'{work_um:.3f} μm/px', flush=True)
    print(f'  tissue centroids — Xenium=({ctx.xen_centroid[0]:.0f},'
          f'{ctx.xen_centroid[1]:.0f}) CODEX=({ctx.cdx_centroid[0]:.0f},'
          f'{ctx.cdx_centroid[1]:.0f})', flush=True)
    expected_scale = xen_work_px / work_um


    print(f'\n[{time.time()-t0:.0f}s] Stage A — shape-mask seeds + FFT + grid '
          f'× 4 rotations', flush=True)
    shape_seeds = shape_match_seeds(ctx, ctx.xen_mask, ctx.cdx_mask)
    print(f'  top-10 shape seeds (by Xenium-coverage):', flush=True)
    for params, cov in shape_seeds[:10]:
        score_at = evaluate(params, ctx)
        M_seed = affine_from_params(*params).astype(np.float32)
        rev_seed = reverse_iou_metric(M_seed, ctx)
        print(f'    rot={params[0]:5.0f}° sc={params[1]:.2f} '
              f't=({params[2]:6.0f},{params[3]:6.0f}) '
              f'cov={cov:.3f} rev={rev_seed:.3f} score={score_at:+.4f}',
              flush=True)

    by_rot_shape = {}
    for params, cov in shape_seeds[:30]:
        rot_key = round(params[0])
        by_rot_shape.setdefault(rot_key, []).append((params, cov))

    GRID_X, GRID_Y = 5, 5
    grid_xs = np.linspace(0, ctx.w, GRID_X + 1)[:-1] + ctx.w / (2 * GRID_X)
    grid_ys = np.linspace(0, ctx.h, GRID_Y + 1)[:-1] + ctx.h / (2 * GRID_Y)

    all_seeds = []
    for rot in (0.0, 90.0, 180.0, 270.0):

        rot_key = round(rot)
        shape_cands = by_rot_shape.get(rot_key, [])
        candidates = [(p, c, 'shape') for p, c in shape_cands[:6]]

        fft_cands = fft_seeds_for_rotation(ctx, rot, expected_scale, top_k=5)
        for p, n in fft_cands:
            candidates.append((p, n, 'fft'))

        for ty in grid_ys:
            for tx in grid_xs:
                p = (rot, expected_scale, float(tx), float(ty))
                candidates.append((p, evaluate(p, ctx), 'grid'))


        best_for_rot = None
        best_key = None
        for params, ncc_seed, src in candidates:
            score_p, p_p = powell(params, ctx, maxiter=80)
            _, ncc_at, silh_at = evaluate_with_coverage(p_p, ctx)
            key = ncc_at + 2.0 * silh_at
            if best_key is None or key > best_key:
                best_key = key
                best_for_rot = (silh_at, ncc_at, score_p, p_p, ncc_seed, src)
        if best_for_rot is None:
            print(f'    rot init={rot:5.0f}°: NO VALID candidates', flush=True)
            continue
        silh_p, ncc_p, score_p, p_p, ncc_seed, src = best_for_rot

        M_p = affine_from_params(*p_p).astype(np.float32)
        warped_p = cv2.warpAffine(ctx.xen_mask_u8, M_p, (ctx.w, ctx.h),
                                       flags=cv2.INTER_NEAREST,
                                       borderValue=0).astype(bool)
        n_over_p = int((ctx.cdx_mask & warped_p).sum())
        fwd_cov_p = min(1.0, n_over_p / max(ctx.xen_tissue_total, 1))
        rev_iou_p = reverse_iou_metric(M_p, ctx)
        all_seeds.append((silh_p, ncc_p, score_p, p_p, rot, ncc_seed, src))
        print(f'    rot init={rot:5.0f}°: best seed [{src}] NCC={ncc_seed:.4f} '
              f'→ Powell cov(fwd)={fwd_cov_p:.3f} cov(rev)={rev_iou_p:.3f} '
              f'silhouette={silh_p:.3f} NCC={ncc_p:.4f} score={score_p:.4f} '
              f'(rot={p_p[0]:.2f}° sc={p_p[1]:.4f} '
              f't=({p_p[2]:.0f},{p_p[3]:.0f}))', flush=True)
    if not all_seeds:
        raise RuntimeError('Stage A: no valid candidates across any rotation')


    _composite = lambda s: s[1] + 2.0 * s[0]
    max_silh = max((s[0] for s in all_seeds), default=0.0)
    survivors = [s for s in all_seeds if s[0] >= max_silh - SILH_GATE_TOL] or all_seeds
    survivors.sort(key=lambda s: -_composite(s))
    all_seeds.sort(key=lambda s: -_composite(s))
    best_powell = survivors[0]
    seeds = all_seeds
    ncc_abstain = max((s[1] for s in survivors), default=-1.0) < NCC_ABSTAIN
    if ncc_abstain:
        print('  [registration] abstain_low_ncc: no survivor has trustworthy NCC — '
              'shape-only selection, flag for QC.', flush=True)
    print(f'  best Powell: init rot={best_powell[4]:.0f}° → '
          f'silh={best_powell[0]:.3f} NCC={best_powell[1]:.4f}', flush=True)


    stageA_silh = float(best_powell[0])
    silh_floor  = max(stageA_silh - 0.05, 0.65)
    print(f'\n[{time.time()-t0:.0f}s] Stage B — basin hopping '
          f'(epochs={n_epochs}, patience={patience}, '
          f'silh_floor={silh_floor:.3f} from S_A={stageA_silh:.3f})', flush=True)
    best_p, best_ncc, history = basin_hop(
        best_powell[3], ctx, n_epochs=n_epochs, patience=patience, seed=seed,
        min_silh=silh_floor)


    print(f'\n[{time.time()-t0:.0f}s] Stage C — fine hill climb', flush=True)
    best_p, best_ncc = fine_climb(best_p, ctx)
    print(f'  Stage C: NCC={best_ncc:.4f} rot={best_p[0]:.2f}° '
          f'sc={best_p[1]:.4f} t=({best_p[2]:.0f},{best_p[3]:.0f})', flush=True)


    txt = (f'{Path(xenium_dir).name}  NCC={best_ncc:.3f} '
           f'rot={best_p[0]:.2f}° sc={best_p[1]:.4f}')
    out_img, _ = render_overlay(best_p, ctx, txt)
    cv2.imwrite(str(output_dir / 'xenium_on_codex_full_optim.jpg'), out_img)


    xen_w2f = xen_full.shape[1] / xen_work.shape[1]
    cdx_w2f = cdx_full_shape[1] / cdx_full.shape[1]
    M_work = affine_from_params(*best_p)
    A_full = (cdx_w2f / xen_w2f) * M_work[:2, :2]
    b_full = cdx_w2f * M_work[:, 2]
    M_full = np.zeros((2, 3), dtype=np.float64)
    M_full[:2, :2] = A_full; M_full[:, 2] = b_full
    A_inv = np.linalg.inv(A_full)
    M_full_inv = np.hstack([A_inv, (-A_inv @ b_full).reshape(2, 1)])

    summary = {
        'inputs': {
            'xenium_dir': str(xenium_dir),
            'codex_qptiff': str(codex_qptiff),
            'output_dir': str(output_dir),
        },
        'final_ncc': float(best_ncc),
        'final_params_work_canvas': {
            'rot_deg': float(best_p[0]), 'scale': float(best_p[1]),
            'dx': float(best_p[2]), 'dy': float(best_p[3]),
        },
        'work_canvas': {
            'codex_shape_hw': list(cdx_full.shape[-2:]),
            'xenium_shape_hw': list(xen_work.shape[-2:]),
            'um_per_px': float(work_um),
            'codex_pyramid_level': int(cdx_level),
        },
        'full_resolution': {
            'codex_shape_hw': [int(cdx_full_shape[0]), int(cdx_full_shape[1])],
            'xenium_shape_hw': [int(xen_full.shape[0]), int(xen_full.shape[1])],
            'codex_px_um_full': float(cdx_px) * cdx_full_shape[0] / cdx_full.shape[0],
            'xenium_px_um': float(xen_px),
        },
        'M_xen_full_to_cdx_full': M_full.tolist(),
        'M_cdx_full_to_xen_full': M_full_inv.tolist(),
        'optimization': {

            'rotation_seeds': [
                {'init_rot': float(s[4]), 'seed_ncc': float(s[5]),
                 'powell_silh': float(s[0]), 'powell_ncc': float(s[1]),
                 'powell_score': float(s[2]), 'powell_p': s[3].tolist(),
                 'seed_src': s[6]}
                for s in seeds],
            'basin_hopping_history_tail': history[-30:],
            'abstain_low_ncc': bool(ncc_abstain),
        },
        'elapsed_seconds': float(time.time() - t0),
    }
    (output_dir / 'global_alignment.json').write_text(json.dumps(summary, indent=2))
    print(f'\n[{time.time()-t0:.0f}s] DONE NCC={best_ncc:.4f} → {output_dir}',
          flush=True)
    return summary


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                   formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--xenium', required=True, help='Xenium output dir')
    ap.add_argument('--codex', required=True, help='CODEX qptiff path')
    ap.add_argument('--output', required=True, help='Output dir')
    ap.add_argument('--codex-px', type=float, default=None,
                     help=f'CODEX μm/px (default {DEFAULT_CODEX_PX_UM})')
    ap.add_argument('--work-max', type=int, default=WORK_MAX_DIM)
    ap.add_argument('--n-epochs', type=int, default=60)
    ap.add_argument('--patience', type=int, default=15)
    ap.add_argument('--seed', type=int, default=42)
    args = ap.parse_args()
    align(args.xenium, args.codex, args.output,
          codex_px_override=args.codex_px,
          work_max=args.work_max,
          n_epochs=args.n_epochs,
          patience=args.patience,
          seed=args.seed)


if __name__ == '__main__':
    main()
