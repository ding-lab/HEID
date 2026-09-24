#!/usr/bin/env python

import argparse
import json
import os
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import tifffile
from scipy.ndimage import distance_transform_edt, map_coordinates


LEVELS_UM = (4.0, 1.0, 0.5)
JUDGE_LEVEL_UM = 1.0
MASK_BUILD_LEVEL_UM = 0.5

MASK_PARAM = {
    "he": {"rule": "hsv_saturation", "param": 25},
    "codex": {"rule": "otsu_scale", "param": 0.50},
    "xenium": {"rule": "otsu_scale", "param": 0.25},
}
MASK_PARAM_NOTE = (
    "Plateau operating points from the per-section area-versus-threshold curves. "
    "Held-out sections contribute nothing: none of the 34 sections with a threshold "
    "curve is in HOLDOUT_U. "
    "H&E: the curve's flattest point is saturation 10 but 25 is inside the same "
    "plateau (area drift 10->25 is -0.59%, within the +-1% plateau band) and 25 "
    "is what the Align registration ships, so 25 is kept."
)


DAPI_CLOSE_UM = 15.0
DAPI_HOLE_MM2 = 0.1
DAPI_MIN_AREA_MM2 = 0.5
HE_MIN_AREA_MM2 = 0.05


HE_CLOSE_UM = 0.0

CLAHE_CLIP = 3.0
CLAHE_GRID = 8

BOUNDARY_SPACING_UM = 2.0


HOLDOUT_PAIRS = [(43, 44), (68, 69), (102, 103), (113, 114)]
HOLDOUT_U = {u for p in HOLDOUT_PAIRS for u in p}
NO_PLATEAU_U = {93, 94, 109, 110, 111}
FRAME_CLIPPED_U = {104, 113, 116}
SPLIT_BLOCK_U = {103, 109}

POLARITY_MIN_CONTRAST = 0.05


POLARITY_SPEC = {
    "invariant": "both sides of a cross-modal pair must end in the same polarity",
    "canonical_here": "nuclei_bright",
    "branches": {
        "align_registration": {
            "he": "Macenko haematoxylin deconvolution -> nuclei bright",
            "dapi": "native -- DO NOT invert (it is already nuclei bright)",
            "result": "nuclei_bright",
        },
        "deeperhistreg": {
            "he": "kept as RGB -> nuclei dark",
            "dapi": "MUST be inverted",
            "result": "nuclei_dark",
            "why": "its flip_intensity is one global switch applied to both images",
        },
    },
    "assertion": "median(img[mask]) - median(img[~mask]) > 0.05*255, per section per level",
    "failure_mode": "a flipped sign turns the optimum into the worst point, so no "
                    "search window can find it -- the readout is indistinguishable "
                    "from 'window too small' and from 'no correspondence exists'. "
                    "The assertion has to run before the search, not after.",
}


def config_dict():
    return {
        "levels_um": list(LEVELS_UM),
        "judge_level_um": JUDGE_LEVEL_UM,
        "mask_build_level_um": MASK_BUILD_LEVEL_UM,
        "mask_param": MASK_PARAM,
        "mask_param_note": MASK_PARAM_NOTE,
        "dapi_close_um": DAPI_CLOSE_UM,
        "dapi_hole_mm2": DAPI_HOLE_MM2,
        "dapi_min_area_mm2": DAPI_MIN_AREA_MM2,
        "he_min_area_mm2": HE_MIN_AREA_MM2,
        "he_close_um": HE_CLOSE_UM,
        "clahe_clip": CLAHE_CLIP,
        "clahe_grid": CLAHE_GRID,
        "boundary_spacing_um": BOUNDARY_SPACING_UM,
        "polarity_convention": "nuclei bright, background dark",
        "polarity_spec": POLARITY_SPEC,
        "holdout_u": sorted(HOLDOUT_U),
        "holdout_pairs": [list(p) for p in HOLDOUT_PAIRS],
        "holdout_note": (
            "Enforced at section level. Any edge with either endpoint in "
            "holdout_u is unavailable for tuning; the four holdout_pairs are "
            "the final benchmark and must not be looked at until the "
            "parameters are frozen."),
        "no_plateau_u": sorted(NO_PLATEAU_U),
        "frame_clipped_u": sorted(FRAME_CLIPPED_U),
        "split_block_u": sorted(SPLIT_BLOCK_U),
    }


def _axes_of(series):
    return str(series.axes)


def _levels_of(tf, series):
    levels = list(series.levels)
    if len(levels) > 1:
        return [(tuple(l.shape), _axes_of(series), l.asarray) for l in levels]
    page0 = tf.pages[0]
    subs = list(getattr(page0, "pages", None) or [])
    if subs:
        return ([(tuple(page0.shape), _axes_of(series), page0.asarray)]
                + [(tuple(s.shape), _axes_of(series), s.asarray) for s in subs])
    return [(tuple(levels[0].shape), _axes_of(series), levels[0].asarray)]


def _yx_of(shape, axes):
    if "Y" in axes and "X" in axes and len(axes) == len(shape):
        return shape[axes.index("Y")], shape[axes.index("X")]
    if len(shape) == 3 and shape[-1] in (3, 4):
        return shape[0], shape[1]
    return shape[-2], shape[-1]


def find_xenium_morphology(run_dir: Path):
    focus = run_dir / "morphology_focus"
    if focus.is_dir():
        files = sorted(focus.glob("morphology_focus_*.ome.tif"))
        if files:
            return files[0], len(files)
    for rel in ("morphology_focus.ome.tif", "morphology_mip.ome.tif", "morphology.ome.tif"):
        if (run_dir / rel).exists():
            return run_dir / rel, 1
    raise FileNotFoundError(f"no morphology image under {run_dir}")


def source_path(filepath, modality):
    p = Path(filepath)
    if modality == "xenium":
        return find_xenium_morphology(p)[0]
    return p


def probe_source(filepath, modality):
    path = source_path(filepath, modality)
    import re
    with tifffile.TiffFile(str(path)) as tf:
        series = tf.series[0]
        lv = _levels_of(tf, series)
        xml = tf.ome_metadata or ""
        names = re.findall(r'<Channel[^>]*Name="([^"]*)"', xml)
        return {
            "path": str(path),
            "axes": _axes_of(series),
            "dtype": str(series.dtype),
            "level_shapes": [list(s) for s, _, _ in lv],
            "level_yx": [list(_yx_of(s, a)) for s, a, _ in lv],
            "channel_names": names[:6],
            "n_channels_declared": len(names),
        }


def read_plane_at_mpp(filepath, modality, mpp_native, target_mpp):
    path = source_path(filepath, modality)
    with tifffile.TiffFile(str(path)) as tf:
        series = tf.series[0]
        ladder = _levels_of(tf, series)
        yx = [_yx_of(s, a) for s, a, _ in ladder]
        mpps = [mpp_native * yx[0][1] / float(w) for _, w in yx]
        cand = [i for i, m in enumerate(mpps) if m <= target_mpp * 1.001]


        order = sorted(cand, reverse=True) or [0]
        arr = axes = lvl_mpp = None
        errors = []
        for i in order:
            try:
                arr = ladder[i][2]()
                axes, lvl_mpp = ladder[i][1], mpps[i]
                li = i
                break
            except Exception as e:
                errors.append(f"level {i} ({mpps[i]:.4f} um/px): {e!r}")
        if arr is None:
            raise RuntimeError("every pyramid level failed to decode: " + "; ".join(errors))
        decode_fallback = errors or None

    arr = np.squeeze(arr)


    axis_source = {"series_axes": axes, "rule": None, "channel_index": None}
    if modality == "he":
        if "S" in axes:
            si = axes.index("S")
            if si != arr.ndim - 1 and arr.ndim == 3:
                arr = np.moveaxis(arr, si, -1)
            axis_source["rule"] = "declared axes: S (samples-per-pixel) moved last"
        elif arr.ndim == 3 and arr.shape[0] == 3 and arr.shape[-1] != 3:
            arr = np.moveaxis(arr, 0, -1)
            axis_source["rule"] = "no S in declared axes; leading length-3 axis moved last"
        else:
            axis_source["rule"] = "declared axes already channel-last"
        img = np.ascontiguousarray(arr[..., :3])
    else:
        if "C" in axes and arr.ndim == 3:
            ci = axes.index("C")
            img = np.take(arr, 0, axis=ci)
            axis_source["rule"] = f"declared axes: C at position {ci}"
            axis_source["channel_index"] = 0
        elif arr.ndim == 3:
            img = arr[0]
            axis_source["rule"] = "no C in declared axes; leading axis taken"
            axis_source["channel_index"] = 0
        else:
            img = arr
            axis_source["rule"] = "single plane (pyramid SubIFD), no channel axis"
        img = np.squeeze(img)

    src_h, src_w = img.shape[:2]


    dst_w = max(1, int(round(src_w * lvl_mpp / target_mpp)))
    dst_h = max(1, int(round(src_h * lvl_mpp / target_mpp)))
    if (dst_w, dst_h) != (src_w, src_h):
        interp = cv2.INTER_AREA if dst_w < src_w else cv2.INTER_LINEAR
        img = cv2.resize(img, (dst_w, dst_h), interpolation=interp)
    mpp_actual = lvl_mpp * src_w / float(dst_w)
    info = {
        "level_index": li,
        "level_mpp": round(float(lvl_mpp), 6),
        "level_yx": [int(src_h), int(src_w)],
        "resample_factor": round(float(lvl_mpp / target_mpp), 6),
        "upsampled": bool(dst_w > src_w),
        "channel_axis_source": axis_source,
    }
    if decode_fallback:
        info["decode_fallback"] = decode_fallback
    return img, float(mpp_actual), info


def _pct(arr, qs, max_samples=8_000_000):
    if arr.size > max_samples:
        step = int(np.ceil(np.sqrt(arr.size / max_samples)))
        arr = arr[::step, ::step] if arr.ndim >= 2 else arr[::step]
    return np.percentile(arr, qs)


def _touches_frame(stats, shape):
    h, w = shape
    x0 = stats[:, cv2.CC_STAT_LEFT]
    y0 = stats[:, cv2.CC_STAT_TOP]
    bw = stats[:, cv2.CC_STAT_WIDTH]
    bh = stats[:, cv2.CC_STAT_HEIGHT]
    return (x0 == 0) | (y0 == 0) | (x0 + bw >= w) | (y0 + bh >= h)


def _select(labels, keep):
    keep = np.asarray(keep, bool).copy()
    keep[0] = False
    return keep[labels]


def _fill_holes_cc(mb):
    inv = (~mb).astype(np.uint8)
    n, lab, st, _ = cv2.connectedComponentsWithStats(inv, connectivity=4)
    return mb | _select(lab, ~_touches_frame(st, mb.shape))


def extract_haematoxylin(rgb):
    img = np.clip(rgb.astype(np.float32) / 255.0, 1e-6, 1.0)
    od = -np.log10(img)
    S = np.array([[0.6442, 0.7166, 0.2668],
                  [0.0928, 0.9541, 0.2831],
                  [0.0, 0.0, 0.0]], dtype=np.float32)
    S[0] /= np.linalg.norm(S[0])
    S[1] /= np.linalg.norm(S[1])
    S[2] = np.cross(S[0], S[1])
    S[2] /= np.linalg.norm(S[2])
    inv = np.linalg.inv(S.T)
    h = np.tensordot(od, inv[0].astype(np.float32), axes=([2], [0]))
    p1, p99 = _pct(h, (1, 99))
    u8 = np.clip((h - p1) / (p99 - p1 + 1e-6) * 255, 0, 255).astype(np.uint8)
    return u8, float(p1), float(p99)


def stain_matrix():
    S = np.array([[0.6442, 0.7166, 0.2668],
                  [0.0928, 0.9541, 0.2831],
                  [0.0, 0.0, 0.0]], dtype=np.float64)
    S[0] /= np.linalg.norm(S[0])
    S[1] /= np.linalg.norm(S[1])
    S[2] = np.cross(S[0], S[1])
    S[2] /= np.linalg.norm(S[2])
    return {"source": "align/image_registration/scripts/align_xenium_to_he.py:extract_hematoxylin",
            "rows": ["haematoxylin", "eosin", "residual (cross product)"],
            "matrix": [[round(v, 6) for v in row] for row in S],
            "channel_used": 0}


def _clahe(u8):
    return cv2.createCLAHE(clipLimit=CLAHE_CLIP,
                           tileGridSize=(CLAHE_GRID, CLAHE_GRID)).apply(u8)


def linear_u8(img):
    if img.dtype != np.uint8:
        p2, p98 = _pct(img, [2, 98])
        u8 = np.clip((img - p2) / (p98 - p2 + 1e-8) * 255, 0, 255).astype(np.uint8)
        return u8, float(p2), float(p98)
    return img, 0.0, 255.0


def normalize_dapi(img):
    return _clahe(linear_u8(img)[0])


def canonical_image(raw, modality, use_clahe=True):
    if modality == "he":
        if os.environ.get("HTAN3D_HE_REG", "") == "od":


            g = np.clip(raw.astype(np.float32).mean(2) / 255.0, 1e-6, 1.0)
            od = -np.log10(g)
            p1, p99 = _pct(od, (1, 99))
            u8 = np.clip((od - p1) / (p99 - p1 + 1e-6) * 255, 0, 255).astype(np.uint8)
            rec = {"lo": float(p1), "hi": float(p99),
                   "method": "luminance_od_p1_p99" + ("_clahe" if use_clahe else "")}
            return (_clahe(u8) if use_clahe else u8), rec
        h8, lo, hi = extract_haematoxylin(raw)
        rec = {"lo": lo, "hi": hi,
               "method": "macenko_haematoxylin_od_p1_p99" + ("_clahe" if use_clahe else "")}
        return (_clahe(h8) if use_clahe else h8), rec
    u8, lo, hi = linear_u8(raw)
    rec = {"lo": lo, "hi": hi,
           "method": "percentile_2_98" + ("_clahe" if use_clahe else "")}
    return (_clahe(u8) if use_clahe else u8), rec


def mask_input_image(raw, modality):
    return raw if modality == "he" else normalize_dapi(raw)


def polarity_gate_image(img, use_clahe):
    return img if use_clahe else _clahe(img)


def _disk(r):
    return cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1, 2 * r + 1))


def _keep_large(mb, mpp, min_area_mm2):
    n, labels, stats, _ = cv2.connectedComponentsWithStats(mb.astype(np.uint8), connectivity=8)
    min_px = int(min_area_mm2 * 1e6 / (mpp ** 2))
    return _select(labels, stats[:, cv2.CC_STAT_AREA] >= min_px)


def tissue_mask_dapi(norm_u8, mpp, otsu_scale):
    otsu_t, _ = cv2.threshold(norm_u8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    t = max(5, int(otsu_t * otsu_scale))
    _, base = cv2.threshold(norm_u8, t, 255, cv2.THRESH_BINARY)
    r = max(1, int(round(DAPI_CLOSE_UM / mpp)))
    base = cv2.morphologyEx(base, cv2.MORPH_CLOSE, _disk(r))
    mb = base > 0
    n_h, lab_h, st_h, _ = cv2.connectedComponentsWithStats((~mb).astype(np.uint8), connectivity=8)
    max_hole_px = DAPI_HOLE_MM2 * 1e6 / (mpp ** 2)
    small_inner = (~_touches_frame(st_h, mb.shape)) & (st_h[:, cv2.CC_STAT_AREA] < max_hole_px)
    mb |= _select(lab_h, small_inner)
    return _keep_large(mb, mpp, DAPI_MIN_AREA_MM2), int(otsu_t), int(t)


def tissue_mask_he(rgb, mpp, sat_thresh, close_um=None):
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    mb = hsv[:, :, 1] > sat_thresh
    close_um = HE_CLOSE_UM if close_um is None else close_um
    if close_um > 0:
        r = max(1, int(round(close_um / mpp)))
        mb = cv2.morphologyEx(mb.astype(np.uint8), cv2.MORPH_CLOSE, _disk(r)) > 0
    mb = _fill_holes_cc(mb)
    return _keep_large(mb, mpp, HE_MIN_AREA_MM2), None, int(sat_thresh)


def downsample_mask(mask, dst_mpp, dst_shape, modality):
    if mask.shape != dst_shape:
        small = cv2.resize(mask.astype(np.uint8) * 255, (dst_shape[1], dst_shape[0]),
                           interpolation=cv2.INTER_AREA)
        mask = small > 127
    min_area = HE_MIN_AREA_MM2 if modality == "he" else DAPI_MIN_AREA_MM2
    return _keep_large(mask, dst_mpp, min_area)


def _resample_contour(pts, spacing_px):
    p = pts.astype(np.float64)
    closed = np.vstack([p, p[:1]])
    seg = np.linalg.norm(np.diff(closed, axis=0), axis=1)
    s = np.concatenate([[0.0], np.cumsum(seg)])
    total = s[-1]
    if total < spacing_px * 3:
        return p
    n = max(3, int(round(total / spacing_px)))
    targets = np.linspace(0.0, total, n, endpoint=False)
    x = np.interp(targets, s, closed[:, 0])
    y = np.interp(targets, s, closed[:, 1])
    return np.stack([x, y], axis=1)


def boundary_points(mask, mpp, spacing_um=BOUNDARY_SPACING_UM):
    m = mask.astype(np.uint8)
    sdf = (distance_transform_edt(m == 0) - distance_transform_edt(m > 0)).astype(np.float32)
    gy, gx = np.gradient(sdf)

    n_cc, labels = cv2.connectedComponents(m, connectivity=8)
    xs, ys, nxs, nys, ccs = [], [], [], [], []
    spacing_px = spacing_um / mpp
    for ci in range(1, n_cc):
        comp = (labels == ci).astype(np.uint8)
        cnts, _ = cv2.findContours(comp, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        if not cnts:
            continue
        c = max(cnts, key=cv2.contourArea).reshape(-1, 2).astype(np.float64)
        rp = _resample_contour(c, spacing_px)
        rows = np.clip(rp[:, 1], 0, mask.shape[0] - 1)
        cols = np.clip(rp[:, 0], 0, mask.shape[1] - 1)
        vx = map_coordinates(gx, [rows, cols], order=1, mode="nearest")
        vy = map_coordinates(gy, [rows, cols], order=1, mode="nearest")
        norm = np.hypot(vx, vy)
        ok = norm > 1e-6
        vx = np.where(ok, vx / np.maximum(norm, 1e-6), 0.0)
        vy = np.where(ok, vy / np.maximum(norm, 1e-6), 0.0)
        xs.append((cols + 0.5) * mpp)
        ys.append((rows + 0.5) * mpp)
        nxs.append(vx)
        nys.append(vy)
        ccs.append(np.full(len(rp), ci, np.int16))
    if not xs:
        z = np.zeros((0,), np.float32)
        return z, z, z, z, np.zeros((0,), np.int16)
    return (np.concatenate(xs).astype(np.float32), np.concatenate(ys).astype(np.float32),
            np.concatenate(nxs).astype(np.float32), np.concatenate(nys).astype(np.float32),
            np.concatenate(ccs))


def boundary_roughness(mask, mpp):
    cnts, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not cnts:
        return {}
    c = max(cnts, key=cv2.contourArea)
    per = cv2.arcLength(c, True) * mpp
    hull = cv2.arcLength(cv2.convexHull(c), True) * mpp
    area = float(cv2.contourArea(c)) * mpp ** 2
    circle = 2.0 * np.sqrt(np.pi * max(area, 1e-9))
    return {
        "perimeter_traced_um": round(per, 1),
        "perimeter_convex_hull_um": round(hull, 1),
        "perimeter_equal_area_circle_um": round(circle, 1),
        "roughness_vs_hull": round(per / max(hull, 1e-9), 3),
        "roughness_vs_circle": round(per / max(circle, 1e-9), 3),
    }


def polarity_check(img_u8, mask):
    fg = img_u8[mask]
    bg = img_u8[~mask]
    if fg.size < 100 or bg.size < 100:
        return {"ok": False, "reason": "not enough pixels on one side"}
    fg_med, bg_med = float(np.median(fg)), float(np.median(bg))
    bg_mad = float(np.median(np.abs(bg - bg_med)))
    contrast = (fg_med - bg_med) / 255.0
    out = {
        "ok": bool(contrast > POLARITY_MIN_CONTRAST),
        "tissue_median": fg_med,
        "background_median": bg_med,
        "contrast_of_255": round(contrast, 4),
        "background_mad": bg_mad,
        "n_distinct_values": int(np.unique(img_u8[::17, ::17]).size),
    }


    out["contrast_in_bg_mad"] = (None if bg_mad < 0.5
                                 else round((fg_med - bg_med) / bg_mad, 2))
    return out


def hole_stats(mask, mpp):
    filled = _fill_holes_cc(mask)
    px = mpp ** 2 / 1e6
    a_mask = float(np.count_nonzero(mask)) * px
    a_filled = float(np.count_nonzero(filled)) * px
    out = {
        "area_mm2": round(a_mask, 4),
        "area_holes_filled_mm2": round(a_filled, 4),
        "hole_area_mm2": round(a_filled - a_mask, 4),
        "hole_fraction": round((a_filled - a_mask) / max(a_filled, 1e-9), 4),
    }


    ys, xs = np.nonzero(mask[::4, ::4])
    if xs.size > 10:
        x = xs.astype(np.float64) * 4 * mpp
        y = ys.astype(np.float64) * 4 * mpp
        cx, cy = x.mean(), y.mean()
        cov = np.cov(np.stack([x - cx, y - cy]))
        w, v = np.linalg.eigh(cov)
        major = v[:, int(np.argmax(w))]
        out.update({
            "centroid_um": [round(float(cx), 1), round(float(cy), 1)],
            "bbox_um": [round(float(x.min()), 1), round(float(y.min()), 1),
                        round(float(x.max()), 1), round(float(y.max()), 1)],

            "principal_axis_deg": round(float(np.degrees(np.arctan2(major[1], major[0])) % 180.0), 2),
            "elongation": round(float(np.sqrt(max(w) / max(min(w), 1e-9))), 3),
        })
    return out


def border_occupancy(mask):
    h, w = mask.shape
    edges = [mask[0], mask[-1], mask[:, 0], mask[:, -1]]
    return round(float(max(e.mean() for e in edges)), 4)


NO_PLATEAU_REASON = ("area-versus-threshold curve is steep at every threshold: no "
                     "operating point gives a stable mask calibre")
FRAME_CLIPPED_REASON = "tissue runs off the scanned frame; part of the outline is the frame edge"


def process_section(row, out_root, levels=LEVELS_UM, use_clahe=True):
    t0 = time.time()
    sid = row["section_id"]
    u = int(row["u_number"])
    modality = row["modality"]
    mpp_native = float(row["mpp"])
    out = Path(out_root) / sid
    out.mkdir(parents=True, exist_ok=True)

    meta = {
        "section_id": sid,
        "u_number": u,
        "z_position_um": float(row["z_position_um"]),
        "modality": modality,
        "sample": sid.split("-")[0],
        "source_filepath": row["filepath"],
        "source_image": str(source_path(row["filepath"], modality)),
        "mpp_native": mpp_native,
        "mpp_source": row.get("mpp_source", ""),
        "mask_rule": MASK_PARAM[modality]["rule"],
        "mask_param": MASK_PARAM[modality]["param"],
        "mask_input": ("raw RGB (HSV saturation); CLAHE not involved" if modality == "he"
                       else "percentile 2-98 + CLAHE (fixed, independent of the image variant)"),
        "polarity_convention": "nuclei bright, background dark",
        "image_variant": "clahe" if use_clahe else "linear",
        "n_tissue_blocks_zorder_table": int(row.get("n_tissue_blocks", 0) or 0),
        "mask_reliable": u not in NO_PLATEAU_U,
        "mask_unreliable_reason": (NO_PLATEAU_REASON if u in NO_PLATEAU_U else None),
        "flags": {
            "holdout": u in HOLDOUT_U,
            "calibration_eligible": u not in HOLDOUT_U,
            "mask_calibre_reliable": u not in NO_PLATEAU_U,
            "frame_clipped": u in FRAME_CLIPPED_U,
            "frame_clipped_reason": (FRAME_CLIPPED_REASON if u in FRAME_CLIPPED_U else None),
            "multi_block": u in SPLIT_BLOCK_U,
        },
        "levels": {},
        "files": {},
    }


    order = sorted(set(levels) | {MASK_BUILD_LEVEL_UM}, reverse=False)
    mask_fine = None
    fine_mpp = None
    for lv in order:
        raw, mpp_actual, rinfo = read_plane_at_mpp(row["filepath"], modality, mpp_native, lv)
        img, scaling = canonical_image(raw, modality, use_clahe=use_clahe)

        if lv == MASK_BUILD_LEVEL_UM:
            if modality == "he":
                mask, otsu_t, used_t = tissue_mask_he(raw, mpp_actual, MASK_PARAM["he"]["param"])
            else:
                mask, otsu_t, used_t = tissue_mask_dapi(
                    mask_input_image(raw, modality), mpp_actual, MASK_PARAM[modality]["param"])
            mask_fine = mask
            fine_mpp = mpp_actual
            meta["mask_otsu_raw"] = otsu_t
            meta["mask_threshold_applied"] = used_t
        else:
            assert mask_fine is not None, "mask level must be processed first"
            mask = downsample_mask(mask_fine, mpp_actual, img.shape[:2], modality)

        if lv not in levels:
            continue

        tag = f"mpp{lv:g}"
        img_p = out / f"img_{tag}.tif"
        mask_p = out / f"mask_{tag}.png"
        tifffile.imwrite(str(img_p), img, compression="deflate")
        cv2.imwrite(str(mask_p), (mask.astype(np.uint8) * 255))

        lvl_meta = {
            "mpp_requested": lv,
            "mpp_actual": round(mpp_actual, 6),
            "shape_hw": [int(img.shape[0]), int(img.shape[1])],
            "extent_um": [round(img.shape[0] * mpp_actual, 1), round(img.shape[1] * mpp_actual, 1)],
            "source_level": rinfo,
            "downsample_vs_native": round(float(mpp_actual / mpp_native), 6),
            "intensity_scaling": {"lo": round(scaling["lo"], 6), "hi": round(scaling["hi"], 6),
                                  "method": scaling["method"]},
            "clahe_tile_um": (round(img.shape[1] * mpp_actual / CLAHE_GRID, 1)
                              if use_clahe else None),


            "polarity": polarity_check(polarity_gate_image(img, use_clahe), mask),
            "polarity_variant_image": polarity_check(img, mask),
            "mask": hole_stats(mask, mpp_actual),
            "border_occupancy": border_occupancy(mask),
            "roughness": boundary_roughness(mask, mpp_actual),
            "n_components": int(cv2.connectedComponents(
                mask.astype(np.uint8), connectivity=8)[0] - 1),
        }
        meta["levels"][tag] = lvl_meta
        meta["files"][f"img_{tag}"] = img_p.name
        meta["files"][f"mask_{tag}"] = mask_p.name

        if modality == "he" and abs(lv - 4.0) < 1e-9:


            rgb_p = out / "rgb_mpp4.tif"
            tifffile.imwrite(str(rgb_p), raw, compression="deflate")
            meta["files"]["rgb_mpp4"] = rgb_p.name

        if abs(lv - JUDGE_LEVEL_UM) < 1e-9:
            bx, by, nx, ny, cc = boundary_points(mask, mpp_actual)
            bp = out / "boundary_mpp1.npz"
            np.savez_compressed(bp, x_um=bx, y_um=by, nx=nx, ny=ny, cc_id=cc,
                                mpp=np.float32(mpp_actual),
                                spacing_um=np.float32(BOUNDARY_SPACING_UM))
            meta["files"]["boundary_mpp1"] = bp.name
            meta["boundary"] = {
                "n_points": int(bx.size),
                "spacing_um": BOUNDARY_SPACING_UM,
                "perimeter_um": round(float(bx.size) * BOUNDARY_SPACING_UM, 1),
                "n_components": int(np.unique(cc).size),
            }

        del raw, img, mask


    a_ref = float(row["tissue_area_mm2"])
    a_now = meta["levels"][f"mpp{JUDGE_LEVEL_UM:g}"]["mask"]["area_mm2"]
    meta["qc"] = {
        "tissue_area_mm2_T1_table": a_ref,
        "tissue_area_mm2_here": a_now,
        "area_delta_pct": round(100.0 * (a_now - a_ref) / max(a_ref, 1e-9), 2),
        "elapsed_s": round(time.time() - t0, 1),
    }


    meta["polarity"] = POLARITY_SPEC["canonical_here"]
    meta["work_mpp_actual"] = {t: v["mpp_actual"] for t, v in meta["levels"].items()}
    meta["downsample_vs_native"] = {t: v["downsample_vs_native"] for t, v in meta["levels"].items()}
    meta["intensity_scaling"] = {t: v["intensity_scaling"] for t, v in meta["levels"].items()}
    meta["channel_axis_source"] = {t: v["source_level"]["channel_axis_source"]
                                   for t, v in meta["levels"].items()}
    meta["source_pyramid_level"] = {
        t: {"level_index": v["source_level"]["level_index"],
            "level_mpp": v["source_level"]["level_mpp"],
            "upsampled": v["source_level"]["upsampled"]}
        for t, v in meta["levels"].items()}
    meta["n_tissue_blocks"] = meta["levels"][f"mpp{JUDGE_LEVEL_UM:g}"]["n_components"]
    if modality == "he":
        meta["stain_matrix"] = stain_matrix()

    ok = all(v["polarity"]["ok"] for v in meta["levels"].values())
    meta["ok"] = bool(ok and a_now > 1.0)
    if not ok:
        meta["fail_reason"] = "polarity assertion failed"

    with open(out / "meta.json", "w") as f:
        json.dump(meta, f, indent=1)
    return meta


def load_sections(path):
    import csv
    with open(path) as f:
        return list(csv.DictReader(f, delimiter="\t"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sections", default=(
        os.environ.get("THREED_ROOT", os.path.join(os.environ.get("PROJECTS_ROOT", "/data/heid"), "3d")) + "/registration/runs/zorder/sections.tsv"))
    ap.add_argument("--out", default=(
        os.environ.get("THREED_ROOT", os.path.join(os.environ.get("PROJECTS_ROOT", "/data/heid"), "3d")) + "/registration/runs/prep_sections"))
    ap.add_argument("--index", type=int, default=None, help="array task index")
    ap.add_argument("--only-u", type=int, nargs="*", default=None)
    ap.add_argument("--levels", type=float, nargs="*", default=None)
    ap.add_argument("--no-clahe", action="store_true",
                    help="emit a linear (percentile-stretch only) registration image; "
                         "the mask path is unaffected and stays on the CLAHE image")
    ap.add_argument("--probe", action="store_true")
    args = ap.parse_args()

    rows = load_sections(args.sections)
    out_root = Path(args.out)
    (out_root / "sections").mkdir(parents=True, exist_ok=True)


    cfg_p = out_root / "prep_config.json"

    tmp = out_root / f"prep_config.{os.uname().nodename}.{os.environ.get('LSB_JOBINDEX', '0')}.{os.getpid()}.tmp"
    cfg = config_dict()
    cfg["image_variant"] = "linear" if args.no_clahe else "clahe"
    cfg["variant_note"] = (
        "Only the registration image differs between the clahe and linear variants. "
        "The mask path is identical in both: H&E from HSV saturation on raw RGB "
        "(CLAHE never involved), CODEX/Xenium from percentile+CLAHE, because the "
        "plateau operating points were calibrated on that image and Otsu x scale "
        "means something different on a differently-scaled one. The two variants "
        "therefore share byte-identical masks, boundary point sets, areas and QC.")
    with open(tmp, "w") as f:
        json.dump(cfg, f, indent=1)
    os.replace(tmp, cfg_p)

    if args.probe:
        rep = []
        for r in rows:
            try:
                rep.append({"section_id": r["section_id"], "modality": r["modality"],
                            "mpp_native": float(r["mpp"]), **probe_source(r["filepath"], r["modality"])})
                print("OK  ", r["section_id"], rep[-1]["axes"], rep[-1]["level_yx"][:2],
                      rep[-1]["channel_names"][:2], flush=True)
            except Exception as e:
                rep.append({"section_id": r["section_id"], "modality": r["modality"], "error": repr(e)})
                print("FAIL", r["section_id"], repr(e), flush=True)
        with open(out_root / "source_probe.json", "w") as f:
            json.dump(rep, f, indent=1)
        bad = [x for x in rep if "error" in x]
        print(f"\nprobe: {len(rep) - len(bad)}/{len(rep)} readable")
        return 1 if bad else 0

    if args.index is not None:
        rows = [rows[args.index]]
    elif args.only_u is not None:
        want = set(args.only_u)
        rows = [r for r in rows if int(r["u_number"]) in want]

    levels = tuple(args.levels) if args.levels else LEVELS_UM
    rc = 0
    for r in rows:
        try:
            m = process_section(r, out_root / "sections", levels=levels,
                                use_clahe=not args.no_clahe)
            print(f"{'OK ' if m['ok'] else 'BAD'} {m['section_id']:<14} {m['modality']:<7} "
                  f"area={m['qc']['tissue_area_mm2_here']:>7.3f} mm2 "
                  f"(area {m['qc']['area_delta_pct']:+.2f}%) "
                  f"pol={m['levels']['mpp1']['polarity']['contrast_of_255']:+.3f} "
                  f"bpts={m.get('boundary', {}).get('n_points', 0)} "
                  f"{m['qc']['elapsed_s']:.0f}s", flush=True)
            if not m["ok"]:
                rc = 1
        except Exception as e:
            import traceback
            traceback.print_exc()
            print(f"FAIL {r['section_id']}: {e!r}", flush=True)
            rc = 1
    return rc


if __name__ == "__main__":
    sys.exit(main())
