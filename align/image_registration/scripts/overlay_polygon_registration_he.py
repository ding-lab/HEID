#!/usr/bin/env python3
from __future__ import annotations
import argparse
import json
import os
from pathlib import Path

os.environ['PYTHONUNBUFFERED'] = '1'
import numpy as np
import pandas as pd
import tifffile
import zarr
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.collections import PolyCollection


def open_he_zarr(path):
    store = tifffile.imread(str(path), aszarr=True)
    z = zarr.open(store, mode='r')
    if hasattr(z, 'arrays'):
        keys = sorted(list(z.array_keys()), key=int)
        z = z[keys[0]]
    return z


def cc_for_centroid(cx_xen, cy_xen, ccs):
    for cc in ccs:
        x1, y1, x2, y2 = cc['xen_bbox']
        if x1 <= cx_xen <= x2 and y1 <= cy_xen <= y2:
            return cc['cc_idx']
    return None


def overlay_one(sample, aligned_codex_root, xenium_local, stagef_dir,
                     out_dir, patch_px=1200):
    cdx_dir = Path(aligned_codex_root) / sample
    he_dir = Path(str(aligned_codex_root).replace('/codex', '/he')) / sample
    info = json.loads((cdx_dir / 'apply_alignment.json').read_text())
    pt = info['per_tissue_refinement']
    xen_px = float(info['xenium_px_um'])
    ccs = pt['per_cc']

    sf = json.loads((Path(stagef_dir) / f'{sample}.json').read_text())
    cc_to_sim = {}
    cc_bbox = {}
    for cc in sf['per_cc']:
        sim = cc['similarity_transform_in_cc']
        rot_rad = np.radians(sim['rotation_deg'])
        c, sn = np.cos(rot_rad), np.sin(rot_rad)
        sR = sim['scale'] * np.array([[c, -sn], [sn, c]])
        t = np.array([sim['translation_y_xen_px'],
                          sim['translation_x_xen_px']])
        cc_to_sim[cc['cc_idx']] = (sR, t)
        x1, y1, _, _ = cc['xen_bbox']
        cc_bbox[cc['cc_idx']] = (int(x1), int(y1))

    nb = pd.read_parquet(Path(xenium_local) / sample / 'nucleus_boundaries.parquet')
    nb['vx_he'] = nb['vertex_x'] / xen_px
    nb['vy_he'] = nb['vertex_y'] / xen_px

    z = open_he_zarr(he_dir / 'he_aligned.ome.tif')
    H, W = z.shape[:2]
    centroids = nb.groupby('cell_id', sort=False).agg(
        cx=('vx_he', 'mean'), cy=('vy_he', 'mean'))
    ny = max(1, H // patch_px); nx = max(1, W // patch_px)
    hist, ye, xe = np.histogram2d(centroids['cy'].values, centroids['cx'].values,
                                            bins=[ny, nx], range=[[0, H], [0, W]])
    yi, xi = np.unravel_index(hist.argmax(), hist.shape)
    y1 = int(ye[yi]); y2 = min(y1 + patch_px, H)
    x1 = int(xe[xi]); x2 = min(x1 + patch_px, W)
    print(f'[{sample}] he-patch yxyx=({y1},{x1})-({y2},{x2})  '
          f'~{(y2-y1)*xen_px:.0f}×{(x2-x1)*xen_px:.0f} µm', flush=True)
    patch = np.asarray(z[y1:y2, x1:x2, :])

    inside = ((nb['vx_he'] >= x1) & (nb['vx_he'] < x2) &
                  (nb['vy_he'] >= y1) & (nb['vy_he'] < y2))
    cells_in = nb.loc[inside, 'cell_id'].unique()
    nb_keep = nb[nb['cell_id'].isin(cells_in)].copy()
    cell_to_cc = {}
    for cid, grp in nb_keep.groupby('cell_id', sort=False):
        cx = float(grp['vx_he'].mean()); cy = float(grp['vy_he'].mean())
        cell_to_cc[cid] = cc_for_centroid(cx, cy, ccs)

    polys_left = []
    polys_right = []
    for cid, grp in nb_keep.groupby('cell_id', sort=False):
        verts_xy = np.column_stack([grp['vx_he'].values,
                                              grp['vy_he'].values])
        polys_left.append(verts_xy - np.array([x1, y1]))
        cc_idx = cell_to_cc.get(cid)
        if cc_idx is None or cc_idx not in cc_to_sim:
            polys_right.append(verts_xy - np.array([x1, y1]))
            continue
        sR, t_yx = cc_to_sim[cc_idx]
        bx, by = cc_bbox[cc_idx]
        v_yx_local = np.column_stack([grp['vy_he'].values - by,
                                                  grp['vx_he'].values - bx])
        q_yx_local = v_yx_local @ sR.T + t_yx
        v_global_xy = np.column_stack([q_yx_local[:, 1] + bx,
                                                    q_yx_local[:, 0] + by])
        polys_right.append(v_global_xy - np.array([x1, y1]))

    fig, axes = plt.subplots(1, 2, figsize=(16, 8 * (y2-y1)/(x2-x1)), dpi=180)
    titles = ['registration (before Stage F HE)',
                  'registration + Stage F HE (polygon-intensity max)']
    for ax, polys, title in zip(axes, [polys_left, polys_right], titles):
        ax.imshow(patch, interpolation='nearest', origin='upper')
        pc = PolyCollection(polys, closed=True, facecolors='none',
                                edgecolors='lime', linewidths=0.4, alpha=0.85)
        ax.add_collection(pc)
        ax.set_xlim(0, x2-x1); ax.set_ylim(y2-y1, 0)
        ax.set_aspect('equal')
        ax.set_title(title, fontsize=10)
    cc_used = set(cell_to_cc.values()) - {None}
    parts = []
    for cci in sorted(cc_used):
        cc_data = [c for c in sf['per_cc'] if c['cc_idx'] == cci][0]
        sim = cc_data['similarity_transform_in_cc']
        parts.append(f'CC{cci}: rot{sim["rotation_deg"]:+.4f}° '
                          f's={sim["scale"]:.6f} '
                          f't=({sim["translation_y_um"]:+.2f},{sim["translation_x_um"]:+.2f})µm '
                          f'gain={cc_data["gain_percent"]:+.2f}%')
    fig.suptitle(f'{sample}  {len(cells_in)} cells | ' + '  '.join(parts),
                    fontsize=10)
    out_path = Path(out_dir) / f'{sample}_stageF_v5_he.png'
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(str(out_path), dpi=180, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f'[{sample}] saved {out_path}', flush=True)
    return out_path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--samples', nargs='+', required=True)
    ap.add_argument('--aligned-codex-root',
                     default=os.environ.get("PROJECTS_ROOT", "/data/heid") + '/cell/inputs/aligned_Codex/multimodal' )
    ap.add_argument('--xenium-local',
                     default=os.environ.get("PROJECTS_ROOT", "/data/heid") + '/cell/inputs/Xenium/multimodal_localcopy' )
    ap.add_argument('--stagef-dir', required=True)
    ap.add_argument('--out-dir', required=True)
    ap.add_argument('--patch-px', type=int, default=1200,
                     help='HE patch size in xen-px (1200 ≈ 255 µm, matches '
                          'CODEX 500 cdx-px ≈ 255 µm).')
    args = ap.parse_args()
    for s in args.samples:
        try:
            overlay_one(s, args.aligned_codex_root, args.xenium_local,
                         args.stagef_dir, args.out_dir, args.patch_px)
        except Exception as e:
            import traceback; traceback.print_exc()
            print(f'[{s}] ERROR {type(e).__name__}: {e}', flush=True)


if __name__ == '__main__':
    main()
