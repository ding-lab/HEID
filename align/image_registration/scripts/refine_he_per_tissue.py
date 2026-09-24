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
from scipy.optimize import minimize


from pack_pipeline import load_dapi

SCRIPTS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPTS_DIR))
from align_xenium_to_he import (
    Context, top_k_translations_via_fft, affine_from_params,
    extract_hematoxylin, extract_he_tissue_mask, extract_xen_tissue_mask,
    normalize_dapi, masked_ncc,
)
from apply_he_alignment import (
    save_pyramidal_ome_tiff, warp_he_crop_to_xenium, to_u8_perc, make_overlay,
)
from detect_he_crop_mask import load_he_full_rgb
from detect_xenium_mask import detect_tissue_mask


def project_xen_bbox_to_he(M_xen_to_he, x1, y1, x2, y2):
    corners = np.array([[x1, y1], [x2, y1], [x2, y2], [x1, y2]],
                          dtype=np.float64)
    A = M_xen_to_he[:2, :2]; b = M_xen_to_he[:, 2]
    he_pts = (A @ corners.T).T + b
    return (he_pts[:, 0].min(), he_pts[:, 1].min(),
              he_pts[:, 0].max(), he_pts[:, 1].max())


def _warp_affine_safe(src, M, out_w, out_h, flags, border=0, LIMIT=30000):
    M = M.astype(np.float64).copy()
    sh, sw = src.shape[:2]
    if max(sh, sw) >= LIMIT:
        f = (LIMIT - 1) / max(sh, sw)
        src = cv2.resize(src, (max(1, int(round(sw * f))),
                               max(1, int(round(sh * f)))),
                         interpolation=cv2.INTER_AREA)
        M[:, :2] = M[:, :2] / f
        sh, sw = src.shape[:2]
    if out_w < LIMIT and out_h < LIMIT:
        return cv2.warpAffine(src, M.astype(np.float32), (out_w, out_h),
                              flags=flags, borderValue=border)
    out = np.full((out_h, out_w), border, dtype=src.dtype)
    step = LIMIT - 1
    for oy in range(0, out_h, step):
        oh = min(step, out_h - oy)
        for ox in range(0, out_w, step):
            ow = min(step, out_w - ox)
            Mt = M.copy(); Mt[0, 2] -= ox; Mt[1, 2] -= oy
            out[oy:oy + oh, ox:ox + ow] = cv2.warpAffine(
                src, Mt.astype(np.float32), (ow, oh), flags=flags,
                borderValue=border)
    return out


def build_cc_context(dapi, dapi_px, he_full_rgb,
                          M_xen_to_he, M_he_to_xen,
                          xen_x1, xen_y1, xen_x2, xen_y2,
                          margin_xen_px, margin_he_px,
                          work_size=1200):
    H_xen, W_xen = dapi.shape[:2]
    H_he, W_he = he_full_rgb.shape[:2]
    cx1 = max(0, xen_x1 - margin_xen_px)
    cy1 = max(0, xen_y1 - margin_xen_px)
    cx2 = min(W_xen, xen_x2 + margin_xen_px)
    cy2 = min(H_xen, xen_y2 + margin_xen_px)
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

    hex1f, hey1f, hex2f, hey2f = project_xen_bbox_to_he(
        M_xen_to_he, cx1, cy1, cx2, cy2)
    hex1 = max(0, int(hex1f) - margin_he_px)
    hey1 = max(0, int(hey1f) - margin_he_px)
    hex2 = min(W_he, int(hex2f) + margin_he_px)
    hey2 = min(H_he, int(hey2f) + margin_he_px)

    he_subcrop = he_full_rgb[hey1:hey2, hex1:hex2]
    if he_subcrop.size == 0 or min(he_subcrop.shape[:2]) == 0:


        he_hema_sub = np.zeros((4, 4), dtype=np.uint8)
        he_mask_sub = np.zeros((4, 4), dtype=np.uint8)
    else:
        he_hema_sub = extract_hematoxylin(he_subcrop)
        he_mask_sub = extract_he_tissue_mask(he_subcrop).astype(np.uint8) * 255

    A_h2x = M_he_to_xen[:2, :2]


    A_use = ds_chip * A_h2x
    b_use = ds_chip * (-np.array([cx1, cy1]) + M_he_to_xen[:, 2]
                          + A_h2x @ np.array([hex1, hey1]))
    M_use = np.zeros((2, 3), dtype=np.float64)
    M_use[:2, :2] = A_use
    M_use[:, 2] = b_use
    he_hema_chip = _warp_affine_safe(he_hema_sub, M_use, chip_w, chip_h,
                                     cv2.INTER_LINEAR)
    he_mask_chip_u8 = _warp_affine_safe(he_mask_sub, M_use, chip_w, chip_h,
                                        cv2.INTER_NEAREST)
    he_mask_chip = he_mask_chip_u8 > 127

    xen_mask_chip = extract_xen_tissue_mask(xen_chip_norm)


    xen_chip_clean = xen_chip_norm.copy()
    xen_chip_clean[~xen_mask_chip] = 0
    he_hema_chip_clean = he_hema_chip.copy()
    he_hema_chip_clean[~he_mask_chip] = 0

    work_um_per_px = dapi_px / max(ds_chip, 1e-6)
    ctx = Context(xen_u8=xen_chip_clean, he_u8=he_hema_chip_clean,
                       he_tissue_mask=he_mask_chip,
                       xen_tissue_mask=xen_mask_chip,
                       work_um_per_px=work_um_per_px)
    chip_origin = np.array([cx1, cy1], dtype=np.float64)
    return ctx, chip_origin, ds_chip, (hex1, hey1, hex2, hey2)


def evaluate_silhouette(params, ctx, return_warped=False):
    rot, scale, dx, dy = params
    if scale < 0.95 or scale > 1.05:
        zero = np.zeros((ctx.h, ctx.w), dtype=np.uint8)
        return (-2.0, zero) if return_warped else -2.0
    M = affine_from_params(rot, scale, dx, dy).astype(np.float32)
    warped_mask = cv2.warpAffine(ctx.xen_mask_u8, M, (ctx.w, ctx.h),
                                       flags=cv2.INTER_NEAREST,
                                       borderValue=0).astype(bool)
    overlap = ctx.he_mask & warped_mask
    n_over = int(overlap.sum())
    if n_over < 100:
        return (-1.0, np.zeros((ctx.h, ctx.w), dtype=np.uint8)) if return_warped else -1.0
    union = int((ctx.he_mask | warped_mask).sum())
    iou = n_over / max(union, 1)
    s_violation = max(0.0, abs(scale - 1.0) - 0.02)
    iou -= 2.0 * s_violation
    if return_warped:
        warped = cv2.warpAffine(ctx.xen, M, (ctx.w, ctx.h),
                                  flags=cv2.INTER_LINEAR, borderValue=0)
        return iou, warped
    return iou


def evaluate_local(params, ctx, return_warped=False):
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
    n_over = int(overlap.sum())
    if n_over < 100:
        return (-1.0, warped) if return_warped else -1.0
    ncc = masked_ncc(ctx.he_f, warped.astype(np.float32), overlap,
                          min_overlap=100)
    s_violation = max(0.0, abs(scale - 1.0) - 0.02)
    ncc -= 5.0 * s_violation
    coverage = min(1.0, n_over / max(ctx.xen_tissue_total, 1))
    score = ncc + 0.5 * coverage
    if return_warped:
        return score, warped
    return score


def _powell_refine(x0, ctx, bounds, eval_fn=None,
                       xtol=0.01, ftol=1e-4, maxiter=120):
    if eval_fn is None:
        eval_fn = evaluate_local
    try:
        res = minimize(lambda p: -eval_fn(tuple(p), ctx), x0,
                          method='Powell', bounds=bounds,
                          options={'xtol': xtol, 'ftol': ftol,
                                    'maxiter': maxiter})
        return ((float(res.x[0]), float(res.x[1]),
                  float(res.x[2]), float(res.x[3])), -float(res.fun))
    except Exception:
        return None


def refine_per_cc(ctx, search_radius_xen_px,
                       max_rot_deg=1.0, scale_band=0.015,
                       fft_top_k=5,
                       rot_grid=(-1.0, -0.5, -0.25, 0.0, 0.25, 0.5, 1.0),
                       patience=8, max_basin_epochs=40, seed=42,
                       min_improve=0.03,
                       low_ncc_threshold=0.50):
    ncc_id = float(evaluate_local((0.0, 1.0, 0.0, 0.0), ctx))
    if ncc_id < low_ncc_threshold:
        eval_fn = evaluate_silhouette
        ncc_id_active = float(eval_fn((0.0, 1.0, 0.0, 0.0), ctx))
        mode = 'silhouette'

        min_improve_active = max(0.005, min_improve / 6.0)
    else:
        eval_fn = evaluate_local
        ncc_id_active = ncc_id
        mode = 'ncc'
        min_improve_active = min_improve

    bounds = [(-max_rot_deg, max_rot_deg),
                 (1.0 - scale_band, 1.0 + scale_band),
                 (-search_radius_xen_px, search_radius_xen_px),
                 (-search_radius_xen_px, search_radius_xen_px)]
    candidates = [((0.0, 1.0, 0.0, 0.0), ncc_id_active)]
    peaks = top_k_translations_via_fft(ctx.xen, ctx.he, top_k=fft_top_k)


    for rot_seed in rot_grid:
        for px, py, _ in peaks:
            if abs(px) > search_radius_xen_px or abs(py) > search_radius_xen_px:
                continue
            x0 = (rot_seed, 1.0, float(px), float(py))
            r = _powell_refine(x0, ctx, bounds, eval_fn=eval_fn,
                                  xtol=0.05, ftol=1e-3, maxiter=80)
            if r is not None:
                candidates.append(r)
    candidates.sort(key=lambda p: -p[1])
    best_params, best_score = candidates[0]


    rng = np.random.default_rng(seed)
    sigma_rot = 0.4
    sigma_scale = 0.005
    sigma_trans = max(2.0, search_radius_xen_px / 10.0)
    no_improve = 0; epoch = 0
    while no_improve < patience and epoch < max_basin_epochs:
        epoch += 1
        boost = 1.0 + 0.5 * no_improve
        perturb = np.array([
            rng.normal(0.0, sigma_rot * boost),
            rng.normal(0.0, sigma_scale * boost),
            rng.normal(0.0, sigma_trans * boost),
            rng.normal(0.0, sigma_trans * boost),
        ])
        x0 = np.clip(np.array(best_params) + perturb,
                       [b[0] for b in bounds],
                       [b[1] for b in bounds])
        r = _powell_refine(tuple(x0), ctx, bounds, eval_fn=eval_fn,
                              xtol=0.005, ftol=5e-5, maxiter=150)
        if r is None:
            no_improve += 1; continue
        params_new, score_new = r
        if score_new > best_score + 1e-4:
            best_params = params_new; best_score = score_new
            no_improve = 0
        else:
            no_improve += 1


    r = _powell_refine(best_params, ctx, bounds, eval_fn=eval_fn,
                          xtol=0.002, ftol=1e-5, maxiter=200)
    if r is not None and r[1] > best_score:
        best_params, best_score = r


    if best_score < ncc_id_active + min_improve_active:
        best_params = (0.0, 1.0, 0.0, 0.0)
        best_score = ncc_id_active


    rot_b, scale_b, dx_b, dy_b = best_params
    fine_bounds = [
        (max(-max_rot_deg, rot_b - 1.0), min(max_rot_deg, rot_b + 1.0)),
        (max(1.0 - scale_band, scale_b - 0.005),
          min(1.0 + scale_band, scale_b + 0.005)),
        (max(-search_radius_xen_px, dx_b - search_radius_xen_px / 3.0),
          min(search_radius_xen_px, dx_b + search_radius_xen_px / 3.0)),
        (max(-search_radius_xen_px, dy_b - search_radius_xen_px / 3.0),
          min(search_radius_xen_px, dy_b + search_radius_xen_px / 3.0)),
    ]
    r = _powell_refine(best_params, ctx, fine_bounds, eval_fn=eval_fn,
                          xtol=0.00005, ftol=1e-9, maxiter=800)
    if r is not None:
        best_params, best_score = r


    r = _powell_refine(best_params, ctx, fine_bounds, eval_fn=eval_fn,
                          xtol=1e-5, ftol=1e-10, maxiter=1500)
    if r is not None and r[1] >= best_score:
        best_params, best_score = r

    return best_params, best_score, ncc_id_active, mode


def compose_local_M(M_global_xen_to_he, params, chip_origin, ds_chip):
    rot, scale, dx_s, dy_s = params
    inv = 1.0 / max(ds_chip, 1e-6)
    dx_full = dx_s * inv; dy_full = dy_s * inv
    A_resid = affine_from_params(rot, scale, dx_full, dy_full)[:2, :2]
    b_resid = np.array([dx_full, dy_full]) + chip_origin - A_resid @ chip_origin
    A_global = np.array(M_global_xen_to_he)[:2, :2]
    b_global = np.array(M_global_xen_to_he)[:, 2]
    A_local = A_global @ A_resid
    b_local = A_global @ b_resid + b_global
    M_local = np.zeros((2, 3), dtype=np.float64)
    M_local[:2, :2] = A_local
    M_local[:, 2] = b_local
    return M_local


def invert_2x3_affine(M):
    A_inv = np.linalg.inv(M[:2, :2])
    b_inv = -A_inv @ M[:, 2]
    out = np.zeros((2, 3), dtype=np.float64)
    out[:2, :2] = A_inv
    out[:, 2] = b_inv
    return out


def assemble_refined(out_canvas, he_full_rgb, M_xen_to_he_local,
                          cc_mask_full_bool):
    xen_h, xen_w = out_canvas.shape[:2]
    H_he, W_he = he_full_rgb.shape[:2]


    ys, xs = np.where(cc_mask_full_bool)
    bx1, bx2 = int(xs.min()), int(xs.max()) + 1
    by1, by2 = int(ys.min()), int(ys.max()) + 1
    hex1f, hey1f, hex2f, hey2f = project_xen_bbox_to_he(
        M_xen_to_he_local, bx1, by1, bx2, by2)
    margin = 64
    hex1 = max(0, int(hex1f) - margin); hey1 = max(0, int(hey1f) - margin)
    hex2 = min(W_he, int(hex2f) + margin); hey2 = min(H_he, int(hey2f) + margin)
    if hex2 <= hex1 or hey2 <= hey1:
        return

    he_sub = he_full_rgb[hey1:hey2, hex1:hex2]
    M_he_to_xen_local = invert_2x3_affine(M_xen_to_he_local)
    warped = warp_he_crop_to_xenium(
        he_sub, hex1, hey1, M_he_to_xen_local, xen_h, xen_w)


    valid = (warped != 255).any(axis=-1)
    out_canvas[valid] = warped[valid]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--crop-dir', required=True,
                     help='alignment output directory of one field — reads '
                          'apply_alignment.json + global_alignment.json')
    ap.add_argument('--he', default=None, help='HE qptiff (else from json)')
    ap.add_argument('--xenium', default=None,
                     help='Xenium dir override (else from json)')
    ap.add_argument('--search-um', type=float, default=150.0,
                     help='Per-CC translation search radius µm (default 150).')
    ap.add_argument('--max-rot-deg', type=float, default=1.0,
                     help='Per-CC rotation bound deg (default 1)')
    ap.add_argument('--scale-band', type=float, default=0.015,
                     help='Per-CC scale bound around 1.0 (default 0.015 = ±1.5 %%)')
    ap.add_argument('--low-ncc-threshold', type=float, default=0.50,
                     help='Switch a CC to the silhouette-IoU objective when its '
                          'identity local score (NCC + 0.5·coverage, minus the '
                          'scale penalty) is below this (default 0.50).')
    ap.add_argument('--min-improve', type=float, default=0.03,
                     help='Require best_score − identity_score > this to '
                          'accept the per-CC refinement; otherwise the '
                          'identity transform is kept (default 0.03).')
    ap.add_argument('--margin-xen-um', type=float, default=300.0,
                     help='xen chip margin around CC bbox (default 300 µm)')
    ap.add_argument('--margin-he-px', type=int, default=128,
                     help='HE subcrop margin px (default 128)')
    ap.add_argument('--work-size', type=int, default=2400,
                     help='Per-CC chip long-axis px for Powell optimisation '
                          '(default 2400).')
    ap.add_argument('--overlay-max', type=int, default=2400)
    args = ap.parse_args()

    crop = Path(args.crop_dir)
    info = json.loads((crop / 'apply_alignment.json').read_text())
    align_dir = Path(info['source_alignment']).parent
    j = json.loads((align_dir / 'global_alignment.json').read_text())
    M_xen_to_he = np.array(j['M_xen_full_to_he_full'])
    M_he_to_xen = np.array(j['M_he_full_to_xen_full'])
    xen_h, xen_w = j['full_resolution']['xenium_shape_hw']
    xen_px = j['full_resolution']['xenium_px_um']

    he_qptiff = args.he or info.get('he_qptiff')
    xenium_dir = args.xenium or info.get('xenium_dir')

    t0 = time.time()
    print(f'[{time.time()-t0:.0f}s] Loading HE: {he_qptiff}', flush=True)
    he_full_rgb, he_px = load_he_full_rgb(he_qptiff)
    print(f'  shape={he_full_rgb.shape} px={he_px:.4f} µm', flush=True)

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
    margin_he = args.margin_he_px


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
            dapi, dapi_px, he_full_rgb, M_xen_to_he, M_he_to_xen,
            x1, y1, x2, y2, margin_xen, margin_he,
            work_size=args.work_size)

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
        M_local = compose_local_M(M_xen_to_he, params, chip_origin, ds_chip)
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
            'M_xen_full_to_he_full': M_local.tolist(),
            'cc_mask_small': labels == i,
        })


    print(f'\n[{time.time()-t0:.0f}s] Assembling refined HE-on-Xenium '
            f'({xen_h}x{xen_w})', flush=True)
    out = np.full((xen_h, xen_w, 3), 255, dtype=np.uint8)
    for r in per_cc_results:
        cc_mask_small = r.pop('cc_mask_small').astype(np.uint8) * 255
        cc_mask_full = cv2.resize(cc_mask_small, (xen_w, xen_h),
                                       interpolation=cv2.INTER_NEAREST) > 127
        if not cc_mask_full.any():
            continue
        M_local = np.array(r['M_xen_full_to_he_full'])
        assemble_refined(out, he_full_rgb, M_local, cc_mask_full)

    out_path = crop / 'he_aligned.ome.tif'
    save_pyramidal_ome_tiff(out, out_path, pixel_size_um=xen_px)

    for stale in ('he_crop.ome.tif', 'he_on_xenium.ome.tif',
                       'he_on_xenium_refined.ome.tif',
                       'he_on_xenium_bspline.ome.tif'):
        p = crop / stale
        if p.exists():
            p.unlink()
            print(f'  cleaned intermediate: {stale}', flush=True)


    for patt in ('he_crop_s*.ome.tif', 'he_on_xenium_s*.ome.tif',
                 'he_on_xenium_refined_s*.ome.tif',
                 'he_on_xenium_bspline_s*.ome.tif'):
        for p in crop.glob(patt):
            p.unlink()
            print(f'  cleaned intermediate: {p.name}', flush=True)


    print(f'[{time.time()-t0:.0f}s] Building fine_overlay.jpg', flush=True)
    xen_u8 = to_u8_perc(dapi)
    fine_overlay = make_overlay(out, xen_u8, target_long_dim=args.overlay_max)
    cv2.imwrite(str(crop / 'fine_overlay.jpg'), fine_overlay)
    del xen_u8

    info['per_tissue_refinement'] = {
        'engine': 'refine_he_per_tissue',
        'search_um': args.search_um,
        'max_rot_deg': args.max_rot_deg,
        'scale_band': args.scale_band,
        'n_ccs': len(per_cc_results),
        'per_cc': per_cc_results,
        'output_ome_tiff': 'he_aligned.ome.tif',
        'fine_overlay_jpg': 'fine_overlay.jpg',
        'elapsed_seconds': float(time.time() - t0),
    }
    (crop / 'apply_alignment.json').write_text(json.dumps(info, indent=2))
    print(f'\n[{time.time()-t0:.0f}s] DONE → {crop}', flush=True)


if __name__ == '__main__':
    main()
