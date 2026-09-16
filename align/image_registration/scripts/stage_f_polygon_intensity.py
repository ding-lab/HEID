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
import pandas as pd
import tifffile
import zarr
from scipy.ndimage import map_coordinates


def open_dapi_zarr(path):
    store = tifffile.imread(str(path), aszarr=True)
    z = zarr.open(store, mode='r')
    if hasattr(z, 'arrays'):
        keys = sorted(list(z.array_keys()), key=int)
        z = z[keys[0]]
    return z


def build_polygon_mask(nb_df, vx_col, vy_col, cell_col, x_off, y_off,
                              shape_hw, dilate_px=0):
    H, W = shape_hw
    mask = np.zeros((H, W), dtype=np.uint8)
    n = 0
    for cid, grp in nb_df.groupby(cell_col, sort=False):
        verts = np.column_stack([grp[vx_col].values - x_off,
                                          grp[vy_col].values - y_off]).astype(np.int32)
        if (verts[:, 0].max() < 0 or verts[:, 0].min() >= W
                or verts[:, 1].max() < 0 or verts[:, 1].min() >= H):
            continue
        cv2.fillPoly(mask, [verts], 1)
        n += 1
    if dilate_px > 0:
        mask = cv2.dilate(mask, cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (dilate_px * 2 + 1, dilate_px * 2 + 1)))
    return mask.astype(bool), n


def score_at(cdx_patch, mask_pts_yx, s, R, t):

    q = s * (mask_pts_yx @ R.T) + t
    vals = map_coordinates(cdx_patch, [q[:, 0], q[:, 1]], order=1,
                                  mode='constant', cval=0.0)
    return float(vals.sum())


def damped_grid_climb(cdx_patch, mask_pts_yx,
                            max_iters=15,
                            step_decay=0.5,
                            max_t_start_px=4.0,
                            max_rot_start_deg=0.05,
                            max_scale_start=0.0005,
                            min_step_t=0.005):
    s = 1.0
    R = np.eye(2)
    t = np.zeros(2)
    base = score_at(cdx_patch, mask_pts_yx, s, R, t)
    print(f'  iter -1: score={base:.4e} (identity)', flush=True)
    trail = [{'iter': -1, 'score': base, 'tx': 0.0, 'ty': 0.0,
                'rot_deg': 0.0, 's': 1.0}]
    max_t = max_t_start_px
    max_rot = max_rot_start_deg
    max_s_off = max_scale_start
    for it in range(max_iters):
        moves = []
        for dty in (-1.0, -0.5, 0.0, 0.5, 1.0):
            for dtx in (-1.0, -0.5, 0.0, 0.5, 1.0):
                for drot in (-1.0, 0.0, 1.0):
                    for ds in (-1.0, 0.0, 1.0):
                        if dty == dtx == drot == ds == 0:
                            continue
                        moves.append((dty, dtx, drot, ds))
        best_score = base
        best_state = None
        for (dty, dtx, drot, ds) in moves:
            t_new = t + np.array([dty * max_t, dtx * max_t])
            rot_cur = np.arctan2(R[1, 0], R[0, 0])
            rot_new = rot_cur + np.radians(drot * max_rot)
            c, sn = np.cos(rot_new), np.sin(rot_new)
            R_new = np.array([[c, -sn], [sn, c]])
            s_new = s + ds * max_s_off
            sc = score_at(cdx_patch, mask_pts_yx, s_new, R_new, t_new)
            if sc > best_score:
                best_score = sc
                best_state = (s_new, R_new, t_new, dty, dtx, drot, ds)
        if best_state is None:
            print(f'  iter {it:>2}: no improvement at step '
                    f'(t±{max_t:.3f}px, rot±{max_rot:.4f}°, s±{max_s_off:.5f})'
                    f' — shrinking', flush=True)
            max_t *= step_decay
            max_rot *= step_decay
            max_s_off *= step_decay
            trail.append({'iter': int(it), 'score': float(base),
                              'no_improvement': True,
                              'next_step': {'t_px': max_t,
                                                'rot_deg': max_rot,
                                                's_off': max_s_off}})
            if max_t < min_step_t:
                print(f'  step below floor {min_step_t} — done', flush=True)
                break
            continue
        s, R, t, dty, dtx, drot, ds = best_state
        improvement = best_score - base
        base = best_score
        print(f'  iter {it:>2}: dty={dty:+.1f} dtx={dtx:+.1f} '
                f'drot={drot:+.1f} ds={ds:+.1f}  '
                f'→ score={base:.4e} (+{improvement:.3e})  '
                f'acc t=({t[0]:+.3f},{t[1]:+.3f})px '
                f'rot={np.degrees(np.arctan2(R[1,0], R[0,0])):+.4f}° '
                f's={s:.6f}  step={max_t:.3f}px', flush=True)
        trail.append({'iter': int(it),
                          'score': float(base),
                          'improvement': float(improvement),
                          'best_probe': {'dty': dty, 'dtx': dtx,
                                              'drot': drot, 'ds': ds},
                          'step': {'t_px': float(max_t),
                                       'rot_deg': float(max_rot),
                                       's_off': float(max_s_off)},
                          'accumulated': {
                              'tx_px': float(t[1]),
                              'ty_px': float(t[0]),
                              'rot_deg': float(np.degrees(
                                  np.arctan2(R[1, 0], R[0, 0]))),
                              's': float(s),
                          }})

        max_t *= step_decay
        max_rot *= step_decay
        max_s_off *= step_decay
    return s, R, t, trail


def stage_f_for_sample(sample, aligned_root, xenium_local, out_dir,
                            **opt_kwargs):
    s_dir = Path(aligned_root) / sample
    info = json.loads((s_dir / 'apply_alignment.json').read_text())
    pt = info['per_tissue_refinement']
    cdx_px = float(pt['output_pixel_size_um'])
    xen_px = float(info['xenium_px_um'])
    out_h, out_w = pt['output_shape_chw'][1], pt['output_shape_chw'][2]
    t0 = time.time()
    print(f'\n[stage F-POLY] sample={sample}  cdx_px={cdx_px:.4f}', flush=True)

    nb = pd.read_parquet(Path(xenium_local) / sample / 'nucleus_boundaries.parquet')
    nb['vx_cdx'] = nb['vertex_x'] / cdx_px
    nb['vy_cdx'] = nb['vertex_y'] / cdx_px

    cdx_z = open_dapi_zarr(s_dir / 'codex_aligned' / 'ch00-DAPI.ome.tif')
    H, W = cdx_z.shape[-2:]

    per_cc_residuals = []
    for cc in pt['per_cc']:
        cc_idx = cc['cc_idx']
        x1, y1, x2, y2 = cc['xen_bbox']
        s_ratio = cdx_px / xen_px
        cx1 = max(0, int(round(x1 / s_ratio)))
        cy1 = max(0, int(round(y1 / s_ratio)))
        cx2 = min(W, int(round(x2 / s_ratio)))
        cy2 = min(H, int(round(y2 / s_ratio)))
        print(f'\n[{time.time()-t0:.0f}s] CC {cc_idx} '
                f'cdx-bbox=({cx1},{cy1})-({cx2},{cy2}) '
                f'= {cy2-cy1}×{cx2-cx1} px', flush=True)

        cell_cx = nb.groupby('cell_id')['vx_cdx'].mean()
        cell_cy = nb.groupby('cell_id')['vy_cdx'].mean()
        in_cc = ((cell_cx >= cx1) & (cell_cx < cx2) &
                       (cell_cy >= cy1) & (cell_cy < cy2))
        cells_in = set(in_cc[in_cc].index)
        nb_cc = nb[nb['cell_id'].isin(cells_in)].copy()
        print(f'  {len(cells_in)} polygons in CC', flush=True)

        mask, n_ras = build_polygon_mask(
            nb_cc, 'vx_cdx', 'vy_cdx', 'cell_id',
            x_off=cx1, y_off=cy1, shape_hw=(cy2 - cy1, cx2 - cx1))
        if mask.sum() < 1000:
            print('  WARN: mask too small, skip', flush=True); continue
        ys, xs = np.nonzero(mask)
        n_full = len(ys)


        SUB_N = 100_000
        if n_full > SUB_N:
            rng = np.random.default_rng(42)
            idx = rng.choice(n_full, size=SUB_N, replace=False)
            ys, xs = ys[idx], xs[idx]
        mask_pts = np.stack([ys.astype(np.float32),
                                      xs.astype(np.float32)], axis=1)
        print(f'  mask: {n_ras} polygons, {n_full} pixels '
                f'({100.0*mask.mean():.1f}% of patch area) '
                f'→ subsampled to {len(mask_pts)} for score eval', flush=True)

        cdx_patch = np.asarray(cdx_z[cy1:cy2, cx1:cx2]).astype(np.float32)

        s_acc, R_acc, t_acc, trail = damped_grid_climb(
            cdx_patch, mask_pts, **opt_kwargs)
        rot_deg = float(np.degrees(np.arctan2(R_acc[1, 0], R_acc[0, 0])))
        ty_px, tx_px = float(t_acc[0]), float(t_acc[1])
        score_id = score_at(cdx_patch, mask_pts, 1.0, np.eye(2), np.zeros(2))
        score_final = score_at(cdx_patch, mask_pts, s_acc, R_acc, t_acc)
        gain = 100.0 * (score_final - score_id) / max(score_id, 1.0)
        print(f'\n  RESULT: rot={rot_deg:+.5f}° s={s_acc:.7f} '
                f't=({ty_px:+.3f},{tx_px:+.3f})cdx-px '
                f'= ({ty_px*cdx_px:+.3f},{tx_px*cdx_px:+.3f})µm '
                f'score gain {gain:+.2f}%', flush=True)
        per_cc_residuals.append({
            'cc_idx': int(cc_idx),
            'xen_bbox': cc['xen_bbox'],
            'mode': cc['mode'],
            'baseline_score_best': cc['score_best'],
            'n_polygons_in_cc': int(n_ras),
            'mask_pixel_count': int(mask.sum()),
            'score_identity': float(score_id),
            'score_final': float(score_final),
            'gain_percent': float(gain),
            'similarity_transform_in_cc': {
                'scale': float(s_acc),
                'rotation_deg': rot_deg,
                'translation_y_cdx_px': ty_px,
                'translation_x_cdx_px': tx_px,
                'translation_y_um': float(ty_px * cdx_px),
                'translation_x_um': float(tx_px * cdx_px),
            },
            'stage_f_residual_dx_cdx_px': tx_px,
            'stage_f_residual_dy_cdx_px': ty_px,
            'stage_f_residual_dx_um': float(tx_px * cdx_px),
            'stage_f_residual_dy_um': float(ty_px * cdx_px),
            'optim_trail': trail,
        })

    out = {
        'sample': sample,
        'engine': 'stage_f_polygon_intensity_v5',
        'objective': ('Σ cdx_DAPI[T·polygon_pixel] over all Xenium nucleus '
                          'polygons in CC.'),
        'optimizer': 'damped grid climb (step *= step_decay each iter; '
                          'hard-clamped per-iter caps; no-improve → shrink + retry)',
        'opt_params': opt_kwargs,
        'cdx_px_um': float(cdx_px),
        'xen_px_um': float(xen_px),
        'per_cc': per_cc_residuals,
        'elapsed_seconds': float(time.time() - t0),
    }
    out_dir = Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f'{sample}.json'
    out_path.write_text(json.dumps(out, indent=2))
    print(f'\n[{time.time()-t0:.0f}s] DONE {sample} → {out_path}', flush=True)
    return out_path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--samples', nargs='+', required=True)
    ap.add_argument('--aligned-root',
                     default=os.environ.get("PROJECTS_ROOT", "/data/heid") + '/cell/inputs/aligned_Codex/multimodal' )
    ap.add_argument('--xenium-local',
                     default=os.environ.get("PROJECTS_ROOT", "/data/heid") + '/cell/inputs/Xenium/multimodal_localcopy' )
    ap.add_argument('--out-dir', required=True,
                     help='Where to write <sample>.json')
    ap.add_argument('--max-iters', type=int, default=15)
    ap.add_argument('--step-decay', type=float, default=0.5)
    ap.add_argument('--max-t-start-px', type=float, default=4.0)
    ap.add_argument('--max-rot-start-deg', type=float, default=0.05)
    ap.add_argument('--max-scale-start', type=float, default=0.0005)
    args = ap.parse_args()
    opt_kwargs = dict(
        max_iters=args.max_iters,
        step_decay=args.step_decay,
        max_t_start_px=args.max_t_start_px,
        max_rot_start_deg=args.max_rot_start_deg,
        max_scale_start=args.max_scale_start,
    )
    for s in args.samples:
        try:
            stage_f_for_sample(s, args.aligned_root, args.xenium_local,
                                    args.out_dir, **opt_kwargs)
        except Exception as e:
            import traceback; traceback.print_exc()
            print(f'[{s}] ERROR {type(e).__name__}: {e}', flush=True)


if __name__ == '__main__':
    main()
