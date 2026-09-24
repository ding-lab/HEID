#!/usr/bin/env python3
import struct
from pathlib import Path
import numpy as np
from scipy.spatial import cKDTree

MATCH_UM = 2.0


def _cloud_points(cloud, index=None):
    raw = Path(cloud).read_bytes()
    n = struct.unpack_from("<I", raw, 8)[0]
    arr = np.frombuffer(raw, np.uint16, 3 * n, 12).reshape(n, 3)
    off = 12 + 6 * n
    off += (-off) % 4
    n_planes = int(arr[:, 2].max()) + 1 if n else 0
    zt = np.frombuffer(raw, np.float32, n_planes, off)
    idx = np.arange(n) if index is None else index
    out = {}
    for k in range(n_planes):
        sel = idx[arr[idx, 2] == k]
        out[float(zt[k])] = arr[sel, :2].astype(np.float64)
    return out


def frozen_points(s12):
    f = Path(s12) / "nerve_cloud.bin"
    return _cloud_points(f) if f.exists() else None


def member_points(s14):
    f = Path(s14) / "nerve_members.npz"
    if not f.exists():
        return None
    z = np.load(f, allow_pickle=True)
    keys = [k for k in z.files if k.startswith("nerve-")]
    if not keys:
        return {}
    mem = np.unique(np.concatenate([z[k] for k in keys]))
    return _cloud_points(Path(str(z["cloud"])), mem)


def keep_flags(members, z_um, xy_px, um_per_px=8.0):
    pts = members.get(float(z_um))
    if pts is None:

        zs = np.array(list(members))
        if len(zs) == 0:
            return np.zeros(len(xy_px), bool)
        j = int(np.argmin(np.abs(zs - float(z_um))))
        if abs(zs[j] - float(z_um)) > 0.5:
            return np.zeros(len(xy_px), bool)
        pts = members[float(zs[j])]
    if len(pts) == 0 or len(xy_px) == 0:
        return np.zeros(len(xy_px), bool)
    d, _ = cKDTree(pts).query(np.asarray(xy_px, np.float64) * um_per_px, distance_upper_bound=MATCH_UM)
    return np.isfinite(d)


def apply_local_schwann_rescue(volume, planes, cells, keep_masks, classes, excluded=()):
    import json, hashlib
    volume = Path(volume)
    path = volume / 'nerve_local_rescue.json'
    if not path.exists():
        return 0
    data = json.loads(path.read_text())
    if data.get('version') != 1 or data.get('cell_class') != 'Schwann':
        raise ValueError('Unsupported local Schwann rescue manifest')
    by_index = {p['index']: k for k, p in enumerate(planes)}
    ci = classes.index('Schwann')
    added = 0
    for record in data['records']:
        k = by_index[record['index']]
        if planes[k]['section_id'] != record['section_id']:
            raise ValueError('Local rescue section identity changed')
        if record['section_id'] in excluded:
            continue
        files = list((volume / 'G_withdrawn/cell_points').glob(f"z{record['index']:02d}_*.cells.bin"))
        if len(files) != 1 or hashlib.sha256(files[0].read_bytes()).hexdigest() != record['cell_points_sha256']:
            raise ValueError('Local rescue cell rows require revalidation after source changes')
        rows = np.asarray(record['rows'], dtype=int)
        cl = cells[k][1]
        if (rows < 0).any() or (rows >= len(cl)).any() or not np.all(cl[rows] == ci):
            raise ValueError('Local rescue may only retain original Schwann predictions')
        added += int((~keep_masks[k][rows]).sum())
        keep_masks[k][rows] = True
    return added
