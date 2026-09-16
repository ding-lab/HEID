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
from scipy import ndimage as ndi
from scipy.ndimage import binary_fill_holes

SCRIPTS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPTS_DIR))
from pack_pipeline import load_dapi, normalize_dapi
from detect_xenium_mask import detect_tissue_mask


def load_he_full_rgb(he_qptiff):
    with tifffile.TiffFile(str(he_qptiff)) as tif:
        s = tif.series[0]
        rgb = s.levels[0].asarray()
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
        full_px = 0.25
    return rgb, full_px


def detect_he_tissue_mask(he_rgb, he_px_um,
                              work_size: int = 2400,
                              sat_thresh: int = 25,
                              min_area_mm2: float = 0.05) -> tuple:
    ds = min(1.0, work_size / max(he_rgb.shape[:2]))
    small = cv2.resize(he_rgb, None, fx=ds, fy=ds,
                        interpolation=cv2.INTER_AREA) if ds < 0.95 else he_rgb
    hsv = cv2.cvtColor(small, cv2.COLOR_RGB2HSV)
    mask = (hsv[:, :, 1] > sat_thresh).astype(np.uint8) * 255
    mask = (binary_fill_holes(mask > 0).astype(np.uint8)) * 255
    n, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    um2_per_px = (he_px_um / ds) ** 2
    min_area_px = int(min_area_mm2 * 1e6 / um2_per_px)
    out = np.zeros_like(mask)
    for i in range(1, n):
        if int(stats[i, cv2.CC_STAT_AREA]) >= min_area_px:
            out[labels == i] = 255
    return out, small, ds


def project_xen_mask_to_he(xen_mask_small, M_xen_full_to_he_full,
                              ds_xen, ds_he, he_small_shape):
    A = M_xen_full_to_he_full[:2, :2] * (ds_he / ds_xen)
    b = M_xen_full_to_he_full[:, 2] * ds_he
    M_use = np.zeros((2, 3), dtype=np.float64)
    M_use[:2, :2] = A
    M_use[:, 2] = b
    return cv2.warpAffine(xen_mask_small, M_use.astype(np.float32),
                              (he_small_shape[1], he_small_shape[0]),
                              flags=cv2.INTER_NEAREST, borderValue=0)


def geodesic_dilate(seed, allowed, max_iters):
    if max_iters <= 0:
        return seed
    cur = seed.copy()
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    for _ in range(max_iters):
        nxt = cv2.bitwise_and(cv2.dilate(cur, kernel), allowed)
        if np.array_equal(nxt, cur):
            break
        cur = nxt
    return cur


def compute_he_crop_mask(M_xen_full_to_he_full, dapi, dapi_px,
                              he_rgb, he_px_um,
                              he_work_size: int = 2400,
                              sat_thresh: int = 25,
                              he_min_area_mm2: float = 0.05,
                              max_extend_um: float = 150.0) -> dict:
    xen_mask, xen_small, ds_xen, xen_areas = detect_tissue_mask(dapi, dapi_px)
    he_mask, he_small, ds_he = detect_he_tissue_mask(
        he_rgb, he_px_um, work_size=he_work_size,
        sat_thresh=sat_thresh, min_area_mm2=he_min_area_mm2)
    proj = project_xen_mask_to_he(xen_mask, M_xen_full_to_he_full,
                                       ds_xen, ds_he, he_mask.shape)

    seed = cv2.bitwise_and(proj, he_mask)
    radius_px = max(0, int(max_extend_um / (he_px_um / ds_he)))
    final = geodesic_dilate(seed, he_mask, radius_px)
    return {
        'mask_small': final,
        'he_small': he_small,
        'ds_he': ds_he,
        'xen_mask_small': xen_mask,
        'xen_small': xen_small,
        'ds_xen': ds_xen,
        'xen_areas_mm2': xen_areas,
        'projection_small': proj,
        'he_tissue_small': he_mask,
        'extend_radius_px_small': radius_px,
        'max_extend_um': max_extend_um,
    }


GATE_HE_WORK_SIZE = 2400
GATE_HE_SAT_THRESH = 25
GATE_HE_MIN_AREA_MM2 = 0.05

MAX_EXTEND_UM = 150.0

CLOSE_UM = 50.0


COMPARABILITY_RATIO = 0.30
NECK_SPLIT_FRAC = 0.20
MIN_SAMPLE_AREA_MM2 = 1.0
EROSION_STEP_UM = 20.0
MAX_EROSION_UM = 3000.0
NECK_NOISE_FRAC = 0.02
MAX_SPLIT_DEPTH = 4
CROP_GATE_PERIMETER_FRAC = 0.20
GATE_BAND_UM = 40.0
HE_SELECT_PROJ_FRAC = 0.15

NCC_LONG_DIM = 512


def _disk(radius_px: int) -> np.ndarray:
    r = max(1, int(round(radius_px)))
    return cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1, 2 * r + 1))


def _bin01(m: np.ndarray) -> np.ndarray:
    return (m > 0).astype(np.uint8)


def _bbox(mask: np.ndarray):
    ys, xs = np.where(mask > 0)
    if len(xs) == 0:
        return None
    return int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())


def _ncc(a: np.ndarray, b: np.ndarray) -> float:
    a = a.astype(np.float64).ravel()
    b = b.astype(np.float64).ravel()
    if a.size < 100 or b.size != a.size:
        return 0.0
    a = (a - a.mean()) / (a.std() + 1e-8)
    b = (b - b.mean()) / (b.std() + 1e-8)
    return float(np.mean(a * b))


def _json_safe(o):
    if isinstance(o, dict):
        return {k: _json_safe(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_json_safe(v) for v in o]
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, (bool, np.bool_)):
        return bool(o)
    return o


def _locate_two_lobes(cc01: np.ndarray, px_um: float):
    base_area = int(cc01.sum())
    if base_area == 0:
        return None
    step_r = max(1, int(round(EROSION_STEP_UM / px_um)))
    max_steps = max(1, int(round(MAX_EROSION_UM / px_um / step_r)))
    noise_px = max(1, int(NECK_NOISE_FRAC * base_area))
    k = _disk(step_r)
    eroded = cc01.copy()
    pinch_r = 0
    for _ in range(1, max_steps + 1):
        eroded = cv2.erode(eroded, k)
        pinch_r += step_r
        if eroded.sum() == 0:
            return None
        n, lab, st, _ = cv2.connectedComponentsWithStats(eroded, connectivity=8)
        frags = [(int(st[i, cv2.CC_STAT_AREA]), i) for i in range(1, n)
                 if int(st[i, cv2.CC_STAT_AREA]) >= noise_px]
        if len(frags) >= 2:
            frags.sort(reverse=True)
            coreA = (lab == frags[0][1]).astype(np.uint8)
            coreB = (lab == frags[1][1]).astype(np.uint8)
            return coreA, coreB, pinch_r
    return None


def _partition_and_measure_neck(cc01: np.ndarray, coreA: np.ndarray,
                                coreB: np.ndarray, pinch_r: int, px_um: float):
    cc = (cc01 > 0)
    dA = ndi.distance_transform_edt(coreA == 0)
    dB = ndi.distance_transform_edt(coreB == 0)
    lobeA = cc & (dA <= dB)
    lobeB = cc & (dB < dA)
    aA, aB = int(lobeA.sum()), int(lobeB.sum())
    if aB > aA:
        lobeA, lobeB = lobeB, lobeA
        aA, aB = aB, aA

    cc_u8 = cc.astype(np.uint8)
    opened = cv2.morphologyEx(cc_u8, cv2.MORPH_OPEN, _disk(max(1, int(pinch_r))))
    removed = ((cc_u8 > 0) & (opened == 0)).astype(np.uint8)

    k1 = _disk(1)
    lAd = cv2.dilate(lobeA.astype(np.uint8), k1) > 0
    lBd = cv2.dilate(lobeB.astype(np.uint8), k1) > 0
    nb, blab, _, _ = cv2.connectedComponentsWithStats(removed, connectivity=8)
    neck = np.zeros_like(removed)
    for bi in range(1, nb):
        comp = (blab == bi)
        if (comp & lAd).any() and (comp & lBd).any():
            neck[comp] = 1
    aNeck = int(neck.sum())
    return dict(lobeA=lobeA.astype(np.uint8), lobeB=lobeB.astype(np.uint8),
                neck=neck.astype(np.uint8), areas_px=(aA, aB, aNeck))


def _neck_is_thin(aB: int, aNeck: int) -> bool:
    if aB <= 0:
        return False
    return (aNeck / aB) <= NECK_SPLIT_FRAC + 1e-9


def neck_split_cc(cc_mask: np.ndarray, px_um: float, depth: int = 0):
    cc01 = _bin01(cc_mask)
    px_mm2 = (px_um / 1000.0) ** 2
    if depth >= MAX_SPLIT_DEPTH or int(cc01.sum()) == 0:
        return [cc01], []

    cores = _locate_two_lobes(cc01, px_um)
    if cores is None:
        return [cc01], []
    coreA, coreB, pinch_r = cores
    m = _partition_and_measure_neck(cc01, coreA, coreB, pinch_r, px_um)
    aA, aB, aNeck = m['areas_px']
    if aA == 0 or aB == 0:
        return [cc01], []

    comp_ratio = aB / aA
    neck_ratio = aNeck / aB if aB > 0 else 1.0
    comparable = comp_ratio >= COMPARABILITY_RATIO
    thin = _neck_is_thin(aB, aNeck)
    a_area_mm2 = aA * px_mm2
    b_area_mm2 = aB * px_mm2

    if not (comparable and thin):
        reason = ('appendage' if not comparable else 'thick_neck')
        return [cc01], [dict(action='kept-' + reason,
                             a_area_mm2=round(a_area_mm2, 3),
                             b_area_mm2=round(b_area_mm2, 3),
                             neck_area_mm2=round(aNeck * px_mm2, 3),
                             comparability_ratio=round(comp_ratio, 3),
                             neck_ratio=round(neck_ratio, 3),
                             min_sample_area_ok=b_area_mm2 >= MIN_SAMPLE_AREA_MM2)]

    diag = dict(action='split',
                a_area_mm2=round(a_area_mm2, 3),
                b_area_mm2=round(b_area_mm2, 3),
                neck_area_mm2=round(aNeck * px_mm2, 3),
                comparability_ratio=round(comp_ratio, 3),
                neck_ratio=round(neck_ratio, 3),
                min_sample_area_ok=b_area_mm2 >= MIN_SAMPLE_AREA_MM2)

    lobeA = (m['lobeA'] | m['neck']) & cc01.astype(bool)
    lobeB = m['lobeB'] & cc01.astype(bool) & ~lobeA
    if lobeB.sum() == 0:
        return [cc01], [diag]

    atomsA, diagsA = neck_split_cc(lobeA.astype(np.uint8), px_um, depth + 1)
    atomsB, diagsB = neck_split_cc(lobeB.astype(np.uint8), px_um, depth + 1)
    return atomsA + atomsB, [diag] + diagsA + diagsB


def segment_samples(tissue_mask: np.ndarray, px_um: float):
    px_mm2 = (px_um / 1000.0) ** 2
    min_px = MIN_SAMPLE_AREA_MM2 * 1e6 / (px_um ** 2)
    orig = _bin01(binary_fill_holes(_bin01(tissue_mask)).astype(np.uint8))


    close_r = max(1, int(round(CLOSE_UM / px_um)))
    closed = cv2.morphologyEx(orig * 255, cv2.MORPH_CLOSE, _disk(close_r))
    closed = _bin01(binary_fill_holes(closed > 0).astype(np.uint8))
    n_grp, grp = cv2.connectedComponents(closed, connectivity=8)


    n_oc, oc = cv2.connectedComponents(orig, connectivity=8)

    atoms = []
    speck_px = 0.10 * min_px
    for c in range(1, n_oc):
        cc01 = (oc == c).astype(np.uint8)
        if int(cc01.sum()) < speck_px:
            continue
        gid = int(np.bincount(grp[cc01 > 0].ravel()).argmax())
        sub_atoms, diags = neck_split_cc(cc01, px_um)
        if len(sub_atoms) == 1:
            atoms.append(dict(mask=sub_atoms[0], group_id=gid, cc_id=c,
                              split_local=0, diags=diags))
        else:
            for k, a in enumerate(sub_atoms):
                atoms.append(dict(mask=a, group_id=gid, cc_id=c,
                                  split_local=k + 1, diags=diags))

    sample_of = {}
    for idx, a in enumerate(atoms):
        if a['split_local'] == 0:
            key = ('grp', a['group_id'])
        else:
            key = ('split', a['cc_id'], a['split_local'])
        sample_of.setdefault(key, []).append(idx)

    out = np.zeros(tissue_mask.shape[:2], dtype=np.int32)
    seed = np.zeros(tissue_mask.shape[:2], dtype=np.int32)
    for idx, a in enumerate(atoms):
        seed[a['mask'] > 0] = idx + 1
    if seed.max() > 0:
        _, (iy, ix) = ndi.distance_transform_edt(seed == 0, return_indices=True)
        full = seed.copy()
        unl = (seed == 0) & (closed > 0)
        full[unl] = seed[iy[unl], ix[unl]]
    else:
        full = seed

    info = []
    next_id = 1
    for key, atom_idxs in sample_of.items():
        smask = np.zeros(tissue_mask.shape[:2], dtype=bool)
        for idx in atom_idxs:
            smask |= (full == (idx + 1))
        area_px = int(smask.sum())
        if area_px * px_mm2 < MIN_SAMPLE_AREA_MM2:
            continue
        sid = next_id
        next_id += 1
        out[smask] = sid
        n_orig_pieces = len({atoms[i]['cc_id'] for i in atom_idxs})
        all_diags = []
        for i in atom_idxs:
            for d in atoms[i]['diags']:
                if d not in all_diags:
                    all_diags.append(d)
        was_split = key[0] == 'split'
        if was_split:
            action = 'neck-split-subpiece'
        elif n_orig_pieces >= 2:
            action = 'closing-merged'
        else:
            action = 'single-cc'
        split_diag = next((d for d in all_diags if d.get('action') == 'split'),
                          None)
        info.append(dict(
            id=sid,
            area_mm2=round(area_px * px_mm2, 3),
            action=action,
            n_original_pieces=n_orig_pieces,
            neck_area_mm2=(split_diag['neck_area_mm2'] if split_diag else None),
            neck_ratio=(split_diag['neck_ratio'] if split_diag else None),
            comparability_ratio=(split_diag['comparability_ratio']
                                 if split_diag else None),
            diags=all_diags,
        ))
    return out, info


def perimeter_contact_frac(direct_region: np.ndarray, sample_mask: np.ndarray,
                           px_um: float):
    region = _bin01(direct_region) & _bin01(sample_mask)
    if region.sum() == 0:
        return 0.0, 0.0, 0.0
    band_r = max(1, int(round(GATE_BAND_UM / px_um)))
    ring = (cv2.dilate(region * 255, _disk(band_r)) > 0) & (region == 0)
    ring_total = int(ring.sum())
    if ring_total == 0:
        return 1.0, 0.0, 0.0
    inside = ring & (_bin01(sample_mask) > 0)
    contact = int(inside.sum())
    return contact / ring_total, float(contact), float(ring_total)


def build_crop(he_tissue, he_samples, he_info, proj_region, px_um):
    px_mm2 = (px_um / 1000.0) ** 2
    proj = _bin01(proj_region)
    seed = proj & _bin01(he_tissue)
    radius_px = max(0, int(MAX_EXTEND_UM / px_um))
    direct = geodesic_dilate(seed * 255, he_tissue, radius_px)

    overlap_area = int((proj & _bin01(he_tissue)).sum())
    rows = []
    for s in he_info:
        sid = s['id']
        ov = int((proj & (he_samples == sid)).sum())
        if ov == 0:
            continue
        rows.append(dict(he_sample_id=sid, area_mm2=s['area_mm2'],
                         proj_overlap_mm2=round(ov * px_mm2, 3),
                         frac_of_proj=round(ov / max(1, overlap_area), 3)))
    rows.sort(key=lambda r: -r['proj_overlap_mm2'])
    sel_ids = [r['he_sample_id'] for r in rows
               if r['frac_of_proj'] >= HE_SELECT_PROJ_FRAC]
    if not sel_ids and rows:
        sel_ids = [rows[0]['he_sample_id']]

    if not sel_ids:
        return dict(he_sample_ids=[], crop_mode='direct', crop_mask=direct,
                    gate_contact_frac=0.0, gate_contact_px=0.0, gate_ring_px=0.0,
                    direct_region=direct,
                    whole_sample_mask=np.zeros_like(he_tissue), selection=rows)

    whole = np.zeros_like(he_tissue)
    for sid in sel_ids:
        whole[he_samples == sid] = 255

    contact_frac, contact_px, ring_px = perimeter_contact_frac(direct, whole,
                                                               px_um)
    if contact_frac >= CROP_GATE_PERIMETER_FRAC:
        crop_mode = 'whole_sample'
        crop_mask = whole.copy()
    else:
        crop_mode = 'direct'
        crop_mask = (_bin01(direct) & _bin01(whole)).astype(np.uint8) * 255

    return dict(he_sample_ids=sel_ids, crop_mode=crop_mode, crop_mask=crop_mask,
                gate_contact_frac=round(float(contact_frac), 3),
                gate_contact_px=contact_px, gate_ring_px=ring_px,
                direct_region=direct, whole_sample_mask=whole, selection=rows)


def bidirectional_ncc(crop_mask_small, he_small_rgb, he_tissue, xen_mask_small,
                      dapi, dapi_px, M, ds_xen, ds_he):
    crop = _bin01(crop_mask_small)
    if crop.sum() < 100:
        return 0.0, 0.0

    norm = normalize_dapi(cv2.resize(
        dapi, None, fx=ds_xen, fy=ds_xen, interpolation=cv2.INTER_AREA)
        if ds_xen < 0.95 else dapi)
    A = M[:2, :2] * (ds_he / ds_xen)
    b = M[:, 2] * ds_he
    M_use = np.zeros((2, 3), dtype=np.float32)
    M_use[:2, :2] = A
    M_use[:, 2] = b
    h, w = he_tissue.shape[:2]
    dapi_proj = cv2.warpAffine(norm, M_use, (w, h),
                               flags=cv2.INTER_LINEAR, borderValue=0)
    xen_proj = cv2.warpAffine(_bin01(xen_mask_small) * 255, M_use, (w, h),
                              flags=cv2.INTER_NEAREST, borderValue=0)

    he_gray = cv2.cvtColor(he_small_rgb, cv2.COLOR_RGB2GRAY)
    he_signal = ((255 - he_gray).astype(np.float32) * (crop > 0)).astype(np.uint8)

    eval_region = (crop > 0) & (xen_proj > 0)
    if eval_region.sum() < 100:
        eval_region = (crop > 0)

    def _ds(img):
        s = NCC_LONG_DIM / max(img.shape[:2])
        if s < 1.0:
            return cv2.resize(img, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)
        return img.copy()

    er = _ds((eval_region * 255).astype(np.uint8)) > 0
    he_d = _ds(he_signal)
    dp_d = _ds(dapi_proj)
    crop_d = _ds((crop * 255).astype(np.uint8))
    xen_d = _ds((xen_proj > 0).astype(np.uint8) * 255)
    if er.sum() < 100:
        return 0.0, 0.0
    ncc_fwd = _ncc(he_d[er], dp_d[er])
    ncc_bwd = _ncc(crop_d.astype(np.float32), xen_d.astype(np.float32))
    return round(float(ncc_fwd), 4), round(float(ncc_bwd), 4)


def compute_he_crop_samples(M_xen_full_to_he_full, dapi, dapi_px,
                            he_rgb, he_px_um,
                            he_work_size: int = 2400,
                            sat_thresh: int = 25,
                            he_min_area_mm2: float = GATE_HE_MIN_AREA_MM2) -> dict:
    xen_mask, xen_small, ds_xen, xen_areas = detect_tissue_mask(dapi, dapi_px)
    he_mask, he_small, ds_he = detect_he_tissue_mask(
        he_rgb, he_px_um, work_size=he_work_size,
        sat_thresh=sat_thresh, min_area_mm2=he_min_area_mm2)


    he_small_px_um = he_px_um * (he_rgb.shape[0] / he_small.shape[0])
    xen_small_px_um = dapi_px * (dapi.shape[0] / xen_small.shape[0])

    he_samples, he_info = segment_samples(he_mask, he_small_px_um)
    xen_samples, xen_sinfo = segment_samples(xen_mask, xen_small_px_um)
    px_mm2 = (he_small_px_um / 1000.0) ** 2

    xen_sids = [s['id'] for s in xen_sinfo] or [0]
    crops = []
    for xs_id in xen_sids:
        if xs_id == 0:
            xen_sub = xen_mask
        else:
            xen_sub = (xen_samples == xs_id).astype(np.uint8) * 255
        proj = project_xen_mask_to_he(xen_sub, M_xen_full_to_he_full,
                                      ds_xen, ds_he, he_mask.shape)
        crop = build_crop(he_mask, he_samples, he_info, proj, he_small_px_um)
        ncc_fwd, ncc_bwd = bidirectional_ncc(
            crop['crop_mask'], he_small, he_mask, xen_sub, dapi, dapi_px,
            M_xen_full_to_he_full, ds_xen, ds_he)


        sel_info = [s for s in he_info if s['id'] in crop['he_sample_ids']]
        split_sel = next((s for s in sel_info
                          if s['neck_area_mm2'] is not None), None)
        neck_mm2 = (split_sel['neck_area_mm2'] if split_sel else None)
        neck_ratio = (split_sel['neck_ratio'] if split_sel else None)


        crop_area_mm2 = round(int(_bin01(crop['crop_mask']).sum()) * px_mm2, 3)

        crops.append(dict(
            xen_sample_id=int(xs_id),
            crop_mask_small=crop['crop_mask'].astype(np.uint8),
            he_sample_ids=[int(i) for i in crop['he_sample_ids']],
            crop_mode=crop['crop_mode'],
            gate_contact_frac=crop['gate_contact_frac'],
            gate_contact_px=crop.get('gate_contact_px'),
            gate_ring_px=crop.get('gate_ring_px'),
            selection=crop['selection'],
            neck_connection_mm2=neck_mm2,
            neck_connection_ratio=neck_ratio,
            ncc_he_to_dapi=ncc_fwd,
            ncc_mask_to_mask=ncc_bwd,
            area_mm2=crop_area_mm2,
            he_sample_selection=sel_info,
        ))
        del proj

    return {
        'he_small': he_small,
        'he_px_small_um': float(he_small_px_um),
        'xen_px_small_um': float(xen_small_px_um),
        'he_samples': he_samples,
        'he_info': he_info,
        'xen_samples': xen_samples,
        'xen_info': xen_sinfo,
        'xen_n_samples': len(xen_sinfo),
        'n_he_samples': len(he_info),
        'ds_he': ds_he,
        'ds_xen': ds_xen,
        'he_tissue_small': he_mask,
        'xen_mask_small': xen_mask,
        'xen_areas_mm2': xen_areas,
        'crops': crops,
    }


def downsample_long(img, target_long_dim, interp=cv2.INTER_AREA):
    h, w = img.shape[:2]
    s = target_long_dim / max(h, w)
    if s >= 1.0:
        return img.copy()
    return cv2.resize(img, None, fx=s, fy=s, interpolation=interp)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--align', required=True)
    ap.add_argument('--output', required=True)
    ap.add_argument('--name', required=True)
    ap.add_argument('--he', default=None)
    ap.add_argument('--xenium', default=None)
    ap.add_argument('--max-extend-um', type=float, default=150.0,
                     help='Maximum geodesic extension distance into HE tissue '
                          'beyond the projected Xenium boundary (default 150 µm)')
    ap.add_argument('--sat-thresh', type=int, default=25)
    ap.add_argument('--he-work-size', type=int, default=2400)
    ap.add_argument('--debug-max', type=int, default=2400)
    args = ap.parse_args()

    out = Path(args.output); out.mkdir(parents=True, exist_ok=True)
    j = json.loads((Path(args.align) / 'global_alignment.json').read_text())
    M_xen_to_he = np.array(j['M_xen_full_to_he_full'])
    xen_dir = args.xenium or j['inputs'].get('xenium_dir')
    he_qptiff = args.he or j['inputs'].get('he_qptiff')

    t0 = time.time()
    print(f'[{time.time()-t0:.0f}s] Loading Xenium DAPI', flush=True)
    dapi, dapi_px = load_dapi(xenium_path=str(xen_dir))
    print(f'  shape={dapi.shape} px={dapi_px:.4f} µm', flush=True)
    print(f'[{time.time()-t0:.0f}s] Loading HE qptiff', flush=True)
    he_rgb, he_px_um = load_he_full_rgb(he_qptiff)
    print(f'  shape={he_rgb.shape} px={he_px_um:.4f} µm', flush=True)

    print(f'[{time.time()-t0:.0f}s] Computing HE crop mask', flush=True)
    r = compute_he_crop_mask(M_xen_to_he, dapi, dapi_px, he_rgb, he_px_um,
                                  he_work_size=args.he_work_size,
                                  sat_thresh=args.sat_thresh,
                                  max_extend_um=args.max_extend_um)
    print(f'  Xenium: {len(r["xen_areas_mm2"])} biopsies, '
            f'areas={r["xen_areas_mm2"]} mm²', flush=True)
    print(f'  HE geodesic extend: {args.max_extend_um:.0f} µm '
            f'= {r["extend_radius_px_small"]} small px', flush=True)
    print(f'  final mask area = {100*r["mask_small"].mean()/255:.1f}% of small frame',
            flush=True)

    overlay = cv2.cvtColor(r['he_small'], cv2.COLOR_RGB2BGR)
    contours, _ = cv2.findContours(r['mask_small'], cv2.RETR_EXTERNAL,
                                       cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(overlay, contours, -1, (0, 255, 0), 3)
    out_path = out / f'{args.name}_he_crop_overlay.png'
    cv2.imwrite(str(out_path), downsample_long(overlay, args.debug_max))
    print(f'[{time.time()-t0:.0f}s] DONE → {out_path}', flush=True)


if __name__ == '__main__':
    main()
