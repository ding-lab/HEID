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


DEFAULT_HE_PX_UM = 0.5
WORK_MAX_DIM = 1800


SILH_GATE_TOL = 0.05
NCC_ABSTAIN = 0.15


INVALID_SCORE = -1.4


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


def load_he_rgb(he_qptiff, target_max_dim=WORK_MAX_DIM, px_override=None):
    qp = Path(he_qptiff)
    with tifffile.TiffFile(str(qp)) as tif:
        s = tif.series[0]
        levels = list(s.levels)
        chosen_idx = 0
        for i, lvl in enumerate(levels):
            shp = lvl.shape
            if max(shp[0], shp[1]) >= target_max_dim:
                chosen_idx = i
        chosen = levels[chosen_idx]
        rgb = chosen.asarray()
        full_h, full_w = levels[0].shape[0], levels[0].shape[1]


    HARD_MAX = max(target_max_dim, 8000)
    cmax = max(rgb.shape[0], rgb.shape[1])
    if cmax > HARD_MAX:
        f = HARD_MAX / cmax
        rgb = cv2.resize(rgb, (max(1, int(round(rgb.shape[1] * f))),
                               max(1, int(round(rgb.shape[0] * f)))),
                         interpolation=cv2.INTER_AREA)

    full_px = px_override or get_ome_pixel_size(qp) or DEFAULT_HE_PX_UM
    lvl_max = max(rgb.shape[0], rgb.shape[1])
    full_max = max(full_h, full_w)
    lvl_px = full_px * (full_max / lvl_max)
    return rgb, lvl_px, chosen_idx, (full_h, full_w)


def extract_hematoxylin(rgb_img):
    img = rgb_img.astype(np.float32) / 255.0
    img = np.clip(img, 1e-6, 1.0)
    od = -np.log10(img)
    stain_matrix = np.array([
        [0.6442, 0.7166, 0.2668],
        [0.0928, 0.9541, 0.2831],
        [0.0,    0.0,    0.0   ],
    ], dtype=np.float32)
    stain_matrix[0] /= np.linalg.norm(stain_matrix[0])
    stain_matrix[1] /= np.linalg.norm(stain_matrix[1])
    stain_matrix[2] = np.cross(stain_matrix[0], stain_matrix[1])
    stain_matrix[2] /= np.linalg.norm(stain_matrix[2])
    inv_matrix = np.linalg.inv(stain_matrix.T)
    stain_conc = od.reshape(-1, 3) @ inv_matrix.T
    h_chan = stain_conc[:, 0].reshape(rgb_img.shape[:2])
    p1, p99 = np.percentile(h_chan, (1, 99))
    return np.clip((h_chan - p1) / (p99 - p1 + 1e-6) * 255,
                    0, 255).astype(np.uint8)


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


class Context:
    def __init__(self, xen_u8, he_u8, he_tissue_mask, xen_tissue_mask,
                  work_um_per_px):
        self.xen = xen_u8
        self.he = he_u8
        self.xen_mask_u8 = xen_tissue_mask.astype(np.uint8)
        self.he_mask = he_tissue_mask.astype(bool)
        self.he_mask_u8 = (he_tissue_mask.astype(np.uint8)) * 255
        self.he_f = he_u8.astype(np.float32)
        self.h, self.w = he_u8.shape[:2]
        self.xen_h, self.xen_w = xen_u8.shape[:2]
        self.um_per_px = work_um_per_px
        self.xen_tissue_total = int(self.xen_mask_u8.sum())


_SILHOUETTE_MODE = 'min'


def reverse_iou_metric(M, ctx):
    M_inv = cv2.invertAffineTransform(M)
    he_in_xen = cv2.warpAffine(ctx.he_mask_u8, M_inv, (ctx.xen_w, ctx.xen_h),
                                    flags=cv2.INTER_NEAREST, borderValue=0)
    he_bool = he_in_xen > 127
    xen_bool = ctx.xen_mask_u8 > 0
    inter = int(np.logical_and(he_bool, xen_bool).sum())
    union = int(np.logical_or(he_bool, xen_bool).sum())
    return inter / max(union, 1)


def silhouette_combine(forward_cov, reverse_iou, mode=None):
    m = mode or _SILHOUETTE_MODE
    if m == 'min':     return min(forward_cov, reverse_iou)
    if m == 'avg':     return 0.5 * (forward_cov + reverse_iou)
    if m == 'forward': return forward_cov
    if m == 'reverse': return reverse_iou
    return min(forward_cov, reverse_iou)


def evaluate(params, ctx, return_warped=False):
    rot, scale, dx, dy = params


    if scale < 0.95 or scale > 1.05:
        zero = np.zeros((ctx.h, ctx.w), dtype=np.uint8)
        return (-2.0, zero) if return_warped else -2.0

    M = affine_from_params(rot, scale, dx, dy).astype(np.float32)
    warped = cv2.warpAffine(ctx.xen, M, (ctx.w, ctx.h),
                              flags=cv2.INTER_LINEAR, borderValue=0)

    warped_mask = cv2.warpAffine(ctx.xen_mask_u8, M, (ctx.w, ctx.h),
                                    flags=cv2.INTER_NEAREST,
                                    borderValue=0).astype(bool)
    overlap = ctx.he_mask & warped_mask
    n_warp = int(warped_mask.sum())
    n_over = int(overlap.sum())
    expected_in_canvas = ctx.xen_tissue_total * (scale ** 2)
    in_canvas_ratio = n_warp / max(expected_in_canvas, 1)


    coverage = min(1.0, n_over / max(ctx.xen_tissue_total, 1))
    if (in_canvas_ratio < 0.80 or n_warp == 0
        or coverage < 0.85 or n_over < 5000):
        return (-1.5, warped) if return_warped else -1.5

    ncc = masked_ncc(ctx.he_f, warped.astype(np.float32), overlap)


    s_violation = max(0.0, abs(scale - 1.0) - 0.02)
    ncc -= 10.0 * s_violation


    rev_iou = reverse_iou_metric(M, ctx)
    silhouette = silhouette_combine(coverage, rev_iou)
    ncc_adj = ncc + 2.0 * silhouette

    if return_warped: return ncc_adj, warped
    return ncc_adj


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
    peaks = top_k_translations_via_fft(moving, ctx.he, top_k=top_k)
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


def extract_he_tissue_mask(rgb_img):
    hsv = cv2.cvtColor(rgb_img, cv2.COLOR_RGB2HSV)
    mask = hsv[:, :, 1] > 15
    mask = cv2.morphologyEx(mask.astype(np.uint8), cv2.MORPH_CLOSE,
                              np.ones((7, 7), np.uint8))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN,
                              np.ones((5, 5), np.uint8))
    return mask > 0


def extract_xen_tissue_mask(dapi_u8):
    _, mask = cv2.threshold(dapi_u8, 0, 255,
                              cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((7, 7), np.uint8))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    return mask > 0


def shape_match_seeds(ctx, xen_mask, he_mask,
                        scale_mults=(0.85, 0.90, 0.95, 1.00, 1.05, 1.10, 1.15),
                        top_per_combo=2):


    MT_MAX = 4096
    mt_ds = min(1.0, MT_MAX / max(he_mask.shape))
    if mt_ds < 1.0:
        def _ds_mask(m):
            return cv2.resize(m.astype(np.uint8),
                              (max(1, int(round(m.shape[1] * mt_ds))),
                               max(1, int(round(m.shape[0] * mt_ds)))),
                              interpolation=cv2.INTER_NEAREST).astype(bool)
        he_mask = _ds_mask(he_mask)
        xen_mask = _ds_mask(xen_mask)

    he_mask_u8 = (he_mask * 255).astype(np.uint8)
    he_h, he_w = he_mask.shape
    xen_h0, xen_w0 = xen_mask.shape
    min_tissue_px = max(50, int(1000 * mt_ds * mt_ds))

    seeds = []
    for rot in (0.0, 90.0, 180.0, 270.0):
        for sm in scale_mults:
            M_rs = cv2.getRotationMatrix2D((xen_w0 / 2, xen_h0 / 2), rot, sm)
            xen_rs = cv2.warpAffine(
                (xen_mask * 255).astype(np.uint8), M_rs, (xen_w0, xen_h0),
                flags=cv2.INTER_NEAREST, borderValue=0)
            ys, xs = np.where(xen_rs > 127)
            if len(ys) < min_tissue_px:
                continue

            y_lo, y_hi = ys.min(), ys.max() + 1
            x_lo, x_hi = xs.min(), xs.max() + 1
            xen_crop = xen_rs[y_lo:y_hi, x_lo:x_hi]
            ch, cw = xen_crop.shape
            if ch >= he_h - 5 or cw >= he_w - 5:
                continue
            try:
                res = cv2.matchTemplate(he_mask_u8, xen_crop,
                                          cv2.TM_CCOEFF_NORMED)
            except cv2.error:
                continue
            res_w = res.copy()
            for _ in range(top_per_combo):
                _, max_val, _, max_loc = cv2.minMaxLoc(res_w)
                if max_val < -0.5:
                    break
                px, py = max_loc

                place = np.zeros((he_h, he_w), dtype=bool)
                x0 = max(0, px); x1 = min(he_w, px + cw)
                y0 = max(0, py); y1 = min(he_h, py + ch)
                sx0 = max(0, -px); sy0 = max(0, -py)
                place[y0:y1, x0:x1] = xen_crop[sy0:sy0+(y1-y0),
                                                  sx0:sx0+(x1-x0)] > 127
                inter = (place & he_mask).sum()
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
                              float(t_origin[0]) / mt_ds,
                              float(t_origin[1]) / mt_ds)
                    seeds.append((params, float(coverage)))

                ys_s = max(0, py - 30); ys_e = min(res_w.shape[0], py + 30)
                xs_s = max(0, px - 30); xs_e = min(res_w.shape[1], px + 30)
                res_w[ys_s:ys_e, xs_s:xs_e] = -np.inf

    seeds.sort(key=lambda x: -x[1])
    return seeds


def powell(seed, ctx, maxiter=120):
    res = minimize(lambda p: -evaluate(p, ctx), np.array(seed),
                    method='Powell',
                    options={'xtol': 1e-3, 'ftol': 1e-4,
                             'maxiter': maxiter, 'disp': False})
    return -res.fun, res.x


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
    overlap = ctx.he_mask & warped_mask
    n_warp = int(warped_mask.sum())
    n_over = int(overlap.sum())
    coverage = min(1.0, n_over / max(ctx.xen_tissue_total, 1))
    rev_iou = reverse_iou_metric(M, ctx)
    silhouette = silhouette_combine(coverage, rev_iou)
    if n_warp == 0 or n_over < 5000:
        return -1.5, -1.5, silhouette
    ncc_raw = masked_ncc(ctx.he_f, warped.astype(np.float32), overlap)
    s_violation = max(0.0, abs(scale - 1.0) - 0.02)
    ncc_clean = ncc_raw - 10.0 * s_violation
    score = ncc_clean + 2.0 * silhouette
    return score, ncc_clean, silhouette


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
    out[..., 1] = ctx.he
    out[..., 2] = warped
    cv2.rectangle(out, (0, 0), (ctx.w, 28), (0, 0, 0), -1)
    cv2.putText(out, header, (8, 20),
                 cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
    return out, ncc


def align(xenium_dir, he_qptiff, output_dir,
            he_px_override=None, work_max=WORK_MAX_DIM,
            n_epochs=60, patience=15, seed=42):
    output_dir = Path(output_dir); output_dir.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    print(f'[{time.time()-t0:.0f}s] Loading H&E qptiff', flush=True)
    he_rgb, he_px, he_level, he_full_shape = load_he_rgb(
        he_qptiff, target_max_dim=work_max, px_override=he_px_override)
    print(f'  H&E level {he_level}: shape={he_rgb.shape} '
          f'px={he_px:.4f} μm', flush=True)
    print(f'[{time.time()-t0:.0f}s] Extracting hematoxylin channel', flush=True)
    he_hema = extract_hematoxylin(he_rgb)

    print(f'[{time.time()-t0:.0f}s] Loading Xenium DAPI', flush=True)
    xen_full, xen_px = load_xenium_dapi(xenium_dir)
    print(f'  Xenium shape={xen_full.shape} px={xen_px:.4f} μm', flush=True)

    work_um = he_px
    xen_work, xen_work_px = downsample_to_um(xen_full, xen_px, work_um)
    he_u8 = normalize_dapi(he_hema)
    xen_u8 = normalize_dapi(xen_work)

    he_tissue = extract_he_tissue_mask(he_rgb)
    xen_tissue = extract_xen_tissue_mask(xen_u8)
    ctx = Context(xen_u8, he_u8, he_tissue, xen_tissue, work_um)
    print(f'  work canvas: H&E={he_u8.shape} Xenium={xen_u8.shape} '
          f'{work_um:.3f} μm/px', flush=True)
    print(f'  HE tissue area: {he_tissue.sum()}px ({100*he_tissue.mean():.1f}%)',
          flush=True)
    print(f'  Xenium tissue area: {xen_tissue.sum()}px '
          f'({100*xen_tissue.mean():.1f}%)', flush=True)
    expected_scale = xen_work_px / work_um


    print(f'\n[{time.time()-t0:.0f}s] Stage A — shape-mask matching seeds',
          flush=True)
    shape_seeds = shape_match_seeds(ctx, xen_tissue, he_tissue)
    if not shape_seeds:
        print('  WARN: no shape seeds — falling back to FFT', flush=True)
    print(f'  top-10 shape seeds (by Xenium-coverage):', flush=True)
    for params, cov in shape_seeds[:10]:
        ncc_at_seed = evaluate(params, ctx)
        M_seed = affine_from_params(*params).astype(np.float32)
        rev_seed = reverse_iou_metric(M_seed, ctx)
        print(f'    rot={params[0]:5.0f}° sc={params[1]:.2f} '
              f't=({params[2]:6.0f},{params[3]:6.0f}) '
              f'cov={cov:.3f} rev={rev_seed:.3f} score={ncc_at_seed:+.4f}',
              flush=True)


    by_rot = {}
    for params, cov in shape_seeds[:30]:
        rot_key = round(params[0])
        if rot_key not in by_rot:
            by_rot[rot_key] = []
        by_rot[rot_key].append((params, cov))
    all_seeds = []
    for rot_key in (0, 90, 180, 270):
        cand_list = by_rot.get(rot_key, [])

        fft_cands = fft_seeds_for_rotation(ctx, float(rot_key), expected_scale,
                                              top_k=5)
        for params, cov in cand_list[:6]:
            cand_list_p = [(params, cov, 'shape')]
        cand_list_p = [(p, c, 'shape') for p, c in cand_list[:6]]
        for p, n in fft_cands:
            cand_list_p.append((p, n, 'fft'))

        for gy in np.linspace(0, ctx.h, 5 + 1)[:-1] + ctx.h / 10:
            for gx in np.linspace(0, ctx.w, 5 + 1)[:-1] + ctx.w / 10:
                p = (float(rot_key), expected_scale, float(gx), float(gy))
                cand_list_p.append((p, evaluate(p, ctx), 'grid'))


        best_for_rot = None
        best_key = None
        for params, seed_score, src in cand_list_p:
            score_p, p_p = powell(params, ctx, maxiter=80)
            _, ncc_at, cov_at = evaluate_with_coverage(p_p, ctx)
            key = ncc_at + 2.0 * cov_at
            if best_key is None or key > best_key:
                best_key = key
                best_for_rot = (cov_at, ncc_at, score_p, p_p, seed_score, src)
        if best_for_rot is None:
            print(f'    rot init={rot_key:3d}°: NO VALID candidates',
                  flush=True)
            continue
        cov_p, ncc_p, score_p, p_p, seed_score, src = best_for_rot
        all_seeds.append((cov_p, ncc_p, score_p, p_p, float(rot_key),
                            seed_score, src))

        M_p = affine_from_params(*p_p).astype(np.float32)
        warped_p = cv2.warpAffine(ctx.xen_mask_u8, M_p, (ctx.w, ctx.h),
                                       flags=cv2.INTER_NEAREST,
                                       borderValue=0).astype(bool)
        n_over_p = int((ctx.he_mask & warped_p).sum())
        fwd_cov_p = min(1.0, n_over_p / max(ctx.xen_tissue_total, 1))
        rev_iou_p = reverse_iou_metric(M_p, ctx)
        print(f'    rot init={rot_key:3d}°: best seed [{src}] '
              f'score={seed_score:.4f} → Powell '
              f'cov(fwd)={fwd_cov_p:.3f} cov(rev)={rev_iou_p:.3f} '
              f'silhouette={cov_p:.3f} NCC={ncc_p:.4f} score={score_p:.4f} '
              f'(rot={p_p[0]:.2f}° sc={p_p[1]:.4f} '
              f't=({p_p[2]:.0f},{p_p[3]:.0f}))', flush=True)
    if not all_seeds:
        raise RuntimeError('Stage A: no valid candidates across any rotation')


    _composite = lambda s: s[1] + 2.0 * s[0]

    valid = [s for s in all_seeds if s[2] > INVALID_SCORE]
    if valid:
        if len(valid) < len(all_seeds):
            rej = ', '.join(f'{s[4]:.0f}°' for s in all_seeds if s[2] <= INVALID_SCORE)
            print(f'  [registration] excluded from the silhouette gate ({rej}): rejected by '
                  'the objective (forward coverage < 0.85 or scale out of bounds)',
                  flush=True)
        max_silh = max(s[0] for s in valid)
        survivors = [s for s in valid if s[0] >= max_silh - SILH_GATE_TOL] or valid
    else:


        print('  [registration] no candidate satisfies the objective; ranking by composite '
              'with no silhouette gate', flush=True)
        survivors = list(all_seeds)
    survivors.sort(key=lambda s: -_composite(s))
    all_seeds.sort(key=lambda s: -_composite(s))
    best_powell = survivors[0]
    seeds = all_seeds
    ncc_abstain = max((s[1] for s in survivors), default=-1.0) < NCC_ABSTAIN
    if ncc_abstain:
        print('  [registration] abstain_low_ncc: no survivor has trustworthy NCC '
              f'(max={max((s[1] for s in survivors), default=-1.0):.3f}); '
              'shape-only selection — flag for QC.', flush=True)
    print(f'  best Powell: init rot={best_powell[4]:.0f}° → '
          f'cov={best_powell[0]:.3f} NCC={best_powell[1]:.4f}', flush=True)


    stageA_silh = float(best_powell[0])
    silh_floor = max(stageA_silh - 0.05, 0.65)
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
    cv2.imwrite(str(output_dir / 'xenium_on_he_full_optim.jpg'), out_img)


    xen_w2f = xen_full.shape[1] / xen_work.shape[1]
    he_w2f = he_full_shape[1] / he_rgb.shape[1]
    M_work = affine_from_params(*best_p)
    A_full = (he_w2f / xen_w2f) * M_work[:2, :2]
    b_full = he_w2f * M_work[:, 2]
    M_full = np.zeros((2, 3), dtype=np.float64)
    M_full[:2, :2] = A_full; M_full[:, 2] = b_full
    A_inv = np.linalg.inv(A_full)
    M_full_inv = np.hstack([A_inv, (-A_inv @ b_full).reshape(2, 1)])

    summary = {
        'inputs': {
            'xenium_dir': str(xenium_dir),
            'he_qptiff': str(he_qptiff),
            'output_dir': str(output_dir),
        },
        'final_ncc': float(best_ncc),
        'final_params_work_canvas': {
            'rot_deg': float(best_p[0]), 'scale': float(best_p[1]),
            'dx': float(best_p[2]), 'dy': float(best_p[3]),
        },
        'work_canvas': {
            'he_shape_hw': list(he_u8.shape[:2]),
            'xenium_shape_hw': list(xen_work.shape[:2]),
            'um_per_px': float(work_um),
            'he_pyramid_level': int(he_level),
        },
        'full_resolution': {
            'he_shape_hw': [int(he_full_shape[0]), int(he_full_shape[1])],
            'xenium_shape_hw': [int(xen_full.shape[0]), int(xen_full.shape[1])],
            'he_px_um_full': float(he_px) * he_full_shape[0] / he_u8.shape[0],
            'xenium_px_um': float(xen_px),
        },
        'M_xen_full_to_he_full': M_full.tolist(),
        'M_he_full_to_xen_full': M_full_inv.tolist(),
        'optimization': {
            'rotation_seeds': [

                {'init_rot': float(s[4]),
                 'seed_score': float(s[5]),
                 'powell_cov': float(s[0]),
                 'powell_ncc': float(s[1]),
                 'powell_score': float(s[2]),
                 'powell_p': s[3].tolist(),
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
    ap.add_argument('--he', required=True, help='H&E qptiff path')
    ap.add_argument('--output', required=True, help='Output dir')
    ap.add_argument('--he-px', type=float, default=None,
                     help=f'H&E μm/px override (default OME tags)')
    ap.add_argument('--work-max', type=int, default=WORK_MAX_DIM)
    ap.add_argument('--n-epochs', type=int, default=60)
    ap.add_argument('--patience', type=int, default=15)
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--silhouette-mode', default='min',
                     choices=['min', 'avg', 'forward', 'reverse'],
                     help='Bidirectional silhouette combine: min (both must '
                          'be high — default), avg (mean), forward (coverage '
                          'only), reverse (inverse-warp HE → Xen IoU only)')
    args = ap.parse_args()
    global _SILHOUETTE_MODE
    _SILHOUETTE_MODE = args.silhouette_mode
    print(f'[args] silhouette_mode={_SILHOUETTE_MODE}', flush=True)
    align(args.xenium, args.he, args.output,
          he_px_override=args.he_px,
          work_max=args.work_max,
          n_epochs=args.n_epochs,
          patience=args.patience,
          seed=args.seed)


if __name__ == '__main__':
    main()
