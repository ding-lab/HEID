#!/usr/bin/env python3

import numpy as np
import cv2
import tifffile
from pathlib import Path
import argparse
import json
import time
from datetime import datetime
from scipy.optimize import minimize
from multiprocessing import Pool, cpu_count
import xml.etree.ElementTree as ET

try:
    import SimpleITK as sitk
    HAS_SITK = True
except ImportError:
    HAS_SITK = False

try:
    from tqdm import tqdm
except ImportError:
    def tqdm(iterable, total=None, desc=None):
        return iterable


def get_ome_pixel_size(tif_path):
    with tifffile.TiffFile(tif_path) as tif:
        if tif.ome_metadata:
            root = ET.fromstring(tif.ome_metadata)
            ns = {'ome': 'http://www.openmicroscopy.org/Schemas/OME/2016-06'}
            pixels = root.find('.//ome:Pixels', ns)
            if pixels is not None:
                size_x = pixels.get('PhysicalSizeX')
                if size_x is not None:
                    return float(size_x)
        page = tif.pages[0]
        res_tag = page.tags.get('XResolution')
        unit_tag = page.tags.get('ResolutionUnit')
        if res_tag and unit_tag:
            num, den = res_tag.value
            if den > 0 and num > 0:
                pixels_per_unit = num / den
                unit = unit_tag.value
                if unit == 3:
                    return 1e4 / pixels_per_unit
                elif unit == 2:
                    return 25400 / pixels_per_unit
    return 1.0


def load_he_image(he_path):
    print(f"Loading HE: {he_path}")
    with tifffile.TiffFile(he_path) as tif:
        try:
            img = tif.series[0].levels[0].asarray()
        except Exception:
            img = tif.pages[0].asarray()
    if img.ndim == 3 and img.shape[0] in [3, 4]:
        img = np.transpose(img, (1, 2, 0))
    if img.ndim == 3 and img.shape[2] == 4:
        img = img[:, :, :3]
    pixel_size = get_ome_pixel_size(he_path)
    print(f"  Shape: {img.shape}, Pixel size: {pixel_size:.4f} um/px")
    return img, pixel_size


def load_dapi(dapi_path=None, xenium_path=None):
    if xenium_path:
        xenium_path = Path(xenium_path)
        focus_dir = xenium_path / "morphology_focus"
        if focus_dir.is_dir():
            focus_files = sorted(focus_dir.glob("morphology_focus_*.ome.tif"))
            if focus_files:
                if len(focus_files) > 1:
                    print(f"  Loading {len(focus_files)} DAPI z-planes (max projection)...")
                    with tifffile.TiffFile(str(focus_files[0])) as tif:
                        dapi = tif.pages[0].asarray()
                    for fp in focus_files[1:]:
                        with tifffile.TiffFile(str(fp)) as tif:
                            plane = tif.pages[0].asarray()
                        np.maximum(dapi, plane, out=dapi)
                    pixel_size = get_ome_pixel_size(focus_files[0])
                    print(f"  DAPI: {dapi.shape}, {pixel_size:.4f} um/px")
                    return dapi, pixel_size
                else:
                    dapi_path = focus_files[0]
        else:
            for c in [xenium_path / "morphology_focus.ome.tif", xenium_path / "morphology.ome.tif"]:
                if c.exists():
                    dapi_path = c
                    break

    if dapi_path is None or not Path(dapi_path).exists():
        raise FileNotFoundError("No DAPI image found")

    print(f"Loading DAPI: {dapi_path}")
    with tifffile.TiffFile(str(dapi_path)) as tif:
        dapi = tif.pages[0].asarray()
    pixel_size = get_ome_pixel_size(dapi_path)
    print(f"  Shape: {dapi.shape}, Pixel size: {pixel_size:.4f} um/px")
    return dapi, pixel_size


def extract_hematoxylin(rgb_img):
    img = rgb_img.astype(np.float32) / 255.0
    img = np.clip(img, 1e-6, 1.0)
    od = -np.log10(img)

    stain_matrix = np.array([
        [0.6442, 0.7166, 0.2668],
        [0.0928, 0.9541, 0.2831],
        [0.0, 0.0, 0.0]
    ], dtype=np.float32)
    stain_matrix[0] /= np.linalg.norm(stain_matrix[0])
    stain_matrix[1] /= np.linalg.norm(stain_matrix[1])
    stain_matrix[2] = np.cross(stain_matrix[0], stain_matrix[1])
    stain_matrix[2] /= np.linalg.norm(stain_matrix[2])

    inv_matrix = np.linalg.inv(stain_matrix.T)
    stain_conc = od.reshape(-1, 3) @ inv_matrix.T
    hematoxylin = stain_conc[:, 0].reshape(rgb_img.shape[:2])

    p1, p99 = np.percentile(hematoxylin, (1, 99))
    return np.clip((hematoxylin - p1) / (p99 - p1 + 1e-6) * 255, 0, 255).astype(np.uint8)


def normalize_dapi(dapi_img):
    p2, p98 = np.percentile(dapi_img, [2, 98])
    normalized = np.clip((dapi_img - p2) / (p98 - p2 + 1e-8), 0, 1)
    img_8bit = (normalized * 255).astype(np.uint8)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    return clahe.apply(img_8bit)


def rotate_img(img, deg):
    if deg == 90:
        return cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)
    elif deg == 180:
        return cv2.rotate(img, cv2.ROTATE_180)
    elif deg == 270:
        return cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
    return img


def normalized_cross_correlation(img1, img2, mask=None):
    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)
    if mask is not None:
        img1 = img1[mask]
        img2 = img2[mask]
    img1 = img1.ravel()
    img2 = img2.ravel()
    if len(img1) < 100:
        return 0.0
    img1 = (img1 - img1.mean()) / (img1.std() + 1e-8)
    img2 = (img2 - img2.mean()) / (img2.std() + 1e-8)
    return np.mean(img1 * img2)


def detect_tissues(dapi, min_area_um2=500000, dapi_px=0.2125):
    ds = 0.05
    small = cv2.resize(dapi, None, fx=ds, fy=ds, interpolation=cv2.INTER_AREA)
    small = normalize_dapi(small)
    _, mask = cv2.threshold(small, 20, 255, cv2.THRESH_BINARY)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask, connectivity=8)
    min_area_px = min_area_um2 / (dapi_px / ds) ** 2

    tissues = []
    for i in range(1, num_labels):
        area = stats[i, cv2.CC_STAT_AREA]
        if area < min_area_px:
            continue
        x = int(stats[i, cv2.CC_STAT_LEFT] / ds)
        y = int(stats[i, cv2.CC_STAT_TOP] / ds)
        w = int(stats[i, cv2.CC_STAT_WIDTH] / ds)
        h = int(stats[i, cv2.CC_STAT_HEIGHT] / ds)
        pad_x, pad_y = int(w * 0.05), int(h * 0.05)
        x1 = max(0, x - pad_x)
        y1 = max(0, y - pad_y)
        x2 = min(dapi.shape[1], x + w + pad_x)
        y2 = min(dapi.shape[0], y + h + pad_y)
        tissues.append((area, x1, y1, x2, y2))

    tissues.sort(reverse=True)
    if len(tissues) <= 1:
        return []

    print(f"  Detected {len(tissues)} tissue sections in DAPI:")
    for i, (area, x1, y1, x2, y2) in enumerate(tissues):
        print(f"    Tissue {i+1}: [{x1}:{x2}, {y1}:{y2}] = {(x2-x1)*dapi_px:.0f}x{(y2-y1)*dapi_px:.0f} um")

    return [(x1, y1, x2, y2) for (_, x1, y1, x2, y2) in tissues]


def detect_he_tissues(he_rgb, he_px, min_area_um2=1000000):
    ds = min(0.02, 2000 / max(he_rgb.shape[:2]))
    small = cv2.resize(he_rgb, None, fx=ds, fy=ds, interpolation=cv2.INTER_AREA)
    hsv = cv2.cvtColor(small, cv2.COLOR_RGB2HSV)

    mask = (hsv[:, :, 1] > 25).astype(np.uint8) * 255


    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask, connectivity=8)
    min_area_px = min_area_um2 / (he_px / ds) ** 2

    tissues = []
    for i in range(1, num_labels):
        area = stats[i, cv2.CC_STAT_AREA]
        if area < min_area_px:
            continue
        x = int(stats[i, cv2.CC_STAT_LEFT] / ds)
        y = int(stats[i, cv2.CC_STAT_TOP] / ds)
        w = int(stats[i, cv2.CC_STAT_WIDTH] / ds)
        h = int(stats[i, cv2.CC_STAT_HEIGHT] / ds)
        pad_x, pad_y = int(w * 0.05), int(h * 0.05)
        x1 = max(0, x - pad_x)
        y1 = max(0, y - pad_y)
        x2 = min(he_rgb.shape[1], x + w + pad_x)
        y2 = min(he_rgb.shape[0], y + h + pad_y)
        tissues.append((area, x1, y1, x2, y2))

    tissues.sort(reverse=True)
    if len(tissues) <= 1:
        return []

    print(f"  Detected {len(tissues)} tissue pieces on HE slide:")
    for i, (area, x1, y1, x2, y2) in enumerate(tissues):
        print(f"    Piece {i+1}: [{x1}:{x2}, {y1}:{y2}] = "
              f"{(x2-x1)*he_px:.0f}x{(y2-y1)*he_px:.0f} um, area={area*he_px/ds*he_px/ds/1e6:.1f} mm2")

    return [(x1, y1, x2, y2) for (_, x1, y1, x2, y2) in tissues]


def coarse_align_with_crop(he_rgb, dapi, he_px, dapi_px, work_size=2500):
    print("\n[Stage 1A] Coarse Alignment (Crop Mode)...")

    he_h, he_w = he_rgb.shape[:2]
    phys_scale = he_px / dapi_px

    he_factor = work_size / max(he_h, he_w)
    base_scale = he_factor / phys_scale
    rotations = [0, 90, 180, 270]
    scale_mults = [0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.8, 2.0, 2.5, 3.0]


    he_small_rgb = cv2.resize(he_rgb, None, fx=he_factor, fy=he_factor, interpolation=cv2.INTER_AREA)
    he_small = extract_hematoxylin(he_small_rgb)


    max_template_dim = max(dapi.shape[:2]) * base_scale * max(scale_mults)
    dapi_pre_factor = min(1.0, (max_template_dim * 3) / max(dapi.shape[:2]))
    if dapi_pre_factor < 0.95:
        dapi_pre = cv2.resize(dapi, None, fx=dapi_pre_factor, fy=dapi_pre_factor,
                              interpolation=cv2.INTER_AREA)
    else:
        dapi_pre = dapi
        dapi_pre_factor = 1.0
    dapi_norm = normalize_dapi(dapi_pre)
    adjusted_base_scale = base_scale / dapi_pre_factor

    best_per_rot = {}
    for rot in rotations:
        dapi_rot = rotate_img(dapi_norm, rot)
        best = {'ncc': -np.inf}

        for scale_mult in scale_mults:
            actual_scale = adjusted_base_scale * scale_mult
            dapi_small = cv2.resize(dapi_rot, None, fx=actual_scale, fy=actual_scale,
                                    interpolation=cv2.INTER_AREA)
            if dapi_small.shape[0] < 20 or dapi_small.shape[1] < 20:
                continue

            he_match = he_small
            pad_oy, pad_ox = 0, 0
            if dapi_small.shape[0] >= he_small.shape[0] - 5 or dapi_small.shape[1] >= he_small.shape[1] - 5:
                pad_y = max(0, dapi_small.shape[0] - he_small.shape[0] + 20)
                pad_x = max(0, dapi_small.shape[1] - he_small.shape[1] + 20)
                he_match = np.zeros((he_small.shape[0] + pad_y, he_small.shape[1] + pad_x), dtype=he_small.dtype)
                pad_oy, pad_ox = pad_y // 2, pad_x // 2
                he_match[pad_oy:pad_oy+he_small.shape[0], pad_ox:pad_ox+he_small.shape[1]] = he_small

            if dapi_small.shape[0] >= he_match.shape[0] or dapi_small.shape[1] >= he_match.shape[1]:
                continue

            try:
                res = cv2.matchTemplate(he_match.astype(np.float32),
                                       dapi_small.astype(np.float32),
                                       cv2.TM_CCOEFF_NORMED)
                _, ncc, _, loc = cv2.minMaxLoc(res)
                loc = (loc[0] - pad_ox, loc[1] - pad_oy)

                if ncc > best['ncc']:
                    best = {
                        'ncc': ncc, 'rotation': rot, 'scale_mult': scale_mult,
                        'loc_working': loc, 'shape_working': dapi_small.shape,
                        'he_factor': he_factor, 'base_scale': base_scale,
                    }
            except Exception:
                continue

        if best['ncc'] > -np.inf:
            x_work, y_work = best['loc_working']
            h_work, w_work = best['shape_working']
            best['cx_full'] = int((x_work + w_work / 2) / he_factor)
            best['cy_full'] = int((y_work + h_work / 2) / he_factor)
            best['x_full'] = int(x_work / he_factor)
            best['y_full'] = int(y_work / he_factor)
            best['w_full'] = int(w_work / he_factor)
            best['h_full'] = int(h_work / he_factor)
            best_per_rot[rot] = best
            print(f"  rot={rot:3d}deg: scale={best['scale_mult']:.2f}x, NCC={best['ncc']:.4f}")

    if not best_per_rot:
        print("ERROR: No valid match found!")
        return None

    return best_per_rot


def crop_he_region(he_rgb, coarse_result, dapi_shape=None, he_px=None, dapi_px=None,
                   padding_ratio=0.15):
    he_h, he_w = he_rgb.shape[:2]

    if dapi_shape is not None and he_px is not None and dapi_px is not None:
        dapi_h, dapi_w = dapi_shape[:2]
        if coarse_result['rotation'] in [90, 270]:
            dapi_h, dapi_w = dapi_w, dapi_h
        correct_w = int(dapi_w * dapi_px / he_px)
        correct_h = int(dapi_h * dapi_px / he_px)

        cx = coarse_result['cx_full']
        cy = coarse_result['cy_full']
        if correct_w > he_w * 0.8:
            cx = he_w // 2
        if correct_h > he_h * 0.8:
            cy = he_h // 2

        pad_x = int(correct_w * padding_ratio)
        pad_y = int(correct_h * padding_ratio)
        x1 = max(0, cx - correct_w // 2 - pad_x)
        y1 = max(0, cy - correct_h // 2 - pad_y)
        x2 = min(he_w, cx + correct_w // 2 + pad_x)
        y2 = min(he_h, cy + correct_h // 2 + pad_y)
    else:
        x, y = coarse_result['x_full'], coarse_result['y_full']
        w, h = coarse_result['w_full'], coarse_result['h_full']
        pad_x, pad_y = int(w * padding_ratio), int(h * padding_ratio)
        x1 = max(0, x - pad_x)
        y1 = max(0, y - pad_y)
        x2 = min(he_w, x + w + pad_x)
        y2 = min(he_h, y + h + pad_y)

    print(f"  Crop: x=[{x1}, {x2}], y=[{y1}, {y2}], size={x2-x1}x{y2-y1}")
    he_crop = he_rgb[y1:y2, x1:x2]
    crop_info = {'x1': x1, 'y1': y1, 'x2': x2, 'y2': y2, 'original_shape': list(he_rgb.shape)}
    return he_crop, crop_info


def detect_rotation(he_nuclei, dapi_norm, target_scale, work_size=500):
    he_scale = work_size / max(he_nuclei.shape[:2])
    xn_scale = work_size / max(dapi_norm.shape[:2])
    he_small = cv2.resize(he_nuclei, None, fx=he_scale, fy=he_scale, interpolation=cv2.INTER_AREA)
    xn_small = cv2.resize(dapi_norm, None, fx=xn_scale, fy=xn_scale, interpolation=cv2.INTER_AREA)

    best_ncc, best_rotation = -np.inf, 0
    for rotation_deg in [0, 90, 180, 270]:
        he_rot = rotate_img(he_small, rotation_deg)
        scale_to_xn = (xn_scale / he_scale) * target_scale
        he_scaled = cv2.resize(he_rot, None, fx=scale_to_xn, fy=scale_to_xn, interpolation=cv2.INTER_LINEAR)

        if he_scaled.shape[0] > xn_small.shape[0] or he_scaled.shape[1] > xn_small.shape[1]:
            ch = min(he_scaled.shape[0], xn_small.shape[0])
            cw = min(he_scaled.shape[1], xn_small.shape[1])
            ncc = normalized_cross_correlation(he_scaled[:ch, :cw], xn_small[:ch, :cw])
        else:
            try:
                result = cv2.matchTemplate(xn_small, he_scaled, cv2.TM_CCORR_NORMED)
                _, ncc, _, _ = cv2.minMaxLoc(result)
            except Exception:
                ncc = 0.0

        if ncc > best_ncc:
            best_ncc = ncc
            best_rotation = rotation_deg

    print(f"  Best rotation: {best_rotation} deg (NCC={best_ncc:.4f})")
    return best_rotation


def phase_correlation_alignment(he_nuclei, dapi_norm, rotation_deg, target_scale, work_size=1000):
    xn_scale = work_size / max(dapi_norm.shape[:2])
    xn_small = cv2.resize(dapi_norm, None, fx=xn_scale, fy=xn_scale, interpolation=cv2.INTER_AREA)

    he_rot = rotate_img(he_nuclei, rotation_deg)
    combined_scale = target_scale * xn_scale
    he_scaled = cv2.resize(he_rot, None, fx=combined_scale, fy=combined_scale, interpolation=cv2.INTER_LINEAR)

    h = max(he_scaled.shape[0], xn_small.shape[0])
    w = max(he_scaled.shape[1], xn_small.shape[1])

    he_padded = np.zeros((h, w), dtype=np.float32)
    xn_padded = np.zeros((h, w), dtype=np.float32)
    he_padded[:he_scaled.shape[0], :he_scaled.shape[1]] = he_scaled
    xn_padded[:xn_small.shape[0], :xn_small.shape[1]] = xn_small

    f_he = np.fft.fft2(he_padded)
    f_xn = np.fft.fft2(xn_padded)
    cross_power = (f_xn * np.conj(f_he)) / (np.abs(f_xn * np.conj(f_he)) + 1e-10)
    correlation = np.fft.ifft2(cross_power).real

    peak_y, peak_x = np.unravel_index(np.argmax(correlation), correlation.shape)
    if peak_y > h // 2:
        peak_y -= h
    if peak_x > w // 2:
        peak_x -= w

    return peak_x / xn_scale, peak_y / xn_scale


def fine_alignment(he_nuclei, dapi_norm, rotation_deg, target_scale, init_tx, init_ty,
                   he_rgb=None, work_size=3000, dapi_px=0.2125, screening=False):
    xn_scale = work_size / max(dapi_norm.shape[:2])
    xn_small = cv2.resize(dapi_norm, None, fx=xn_scale, fy=xn_scale, interpolation=cv2.INTER_AREA)

    he_rot = rotate_img(he_nuclei, rotation_deg)
    combined_scale = target_scale * xn_scale
    he_scaled = cv2.resize(he_rot, None, fx=combined_scale, fy=combined_scale, interpolation=cv2.INTER_LINEAR)


    um_per_px = dapi_px / xn_scale
    print(f"  Fine align: work_size={work_size}, xn_scale={xn_scale:.4f}, "
          f"dapi_px={dapi_px:.4f}, um_per_px={um_per_px:.2f}, "
          f"blur_200um={max(3, int(200.0/um_per_px)|1)}px, blur_50um={max(3, int(50.0/um_per_px)|1)}px")


    blur_k = max(3, int(min(he_scaled.shape[:2]) * 0.005) | 1)
    he_blur = cv2.GaussianBlur(he_scaled.astype(np.float32), (blur_k, blur_k), 0)
    he_grad = np.sqrt(cv2.Sobel(he_blur, cv2.CV_32F, 1, 0, ksize=3)**2 +
                      cv2.Sobel(he_blur, cv2.CV_32F, 0, 1, ksize=3)**2)
    he_grad = cv2.normalize(he_grad, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

    xn_blur = cv2.GaussianBlur(xn_small.astype(np.float32), (blur_k, blur_k), 0)
    xn_grad = np.sqrt(cv2.Sobel(xn_blur, cv2.CV_32F, 1, 0, ksize=3)**2 +
                      cv2.Sobel(xn_blur, cv2.CV_32F, 0, 1, ksize=3)**2)
    xn_grad = cv2.normalize(xn_grad, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

    he_sh, he_sw = he_grad.shape[:2]
    xn_sh, xn_sw = xn_grad.shape[:2]
    cx, cy = he_sw / 2, he_sh / 2
    init_tx_small = init_tx * xn_scale
    init_ty_small = init_ty * xn_scale

    def _make_warp_matrix(tx, ty, dtheta, ds):
        dtheta = np.clip(dtheta, -np.radians(15), np.radians(15))
        ds = np.clip(ds, 0.80, 1.20)
        cos_t, sin_t = np.cos(dtheta), np.sin(dtheta)
        a, b = ds * cos_t, -ds * sin_t
        c = tx + cx - ds * (cos_t * cx - sin_t * cy)
        d, e = ds * sin_t, ds * cos_t
        f = ty + cy - ds * (sin_t * cx + cos_t * cy)
        return np.array([[a, b, c], [d, e, f]], dtype=np.float64)

    def objective_grad(params):
        M = _make_warp_matrix(*params)
        he_warped = cv2.warpAffine(he_grad, M, (xn_sw, xn_sh), flags=cv2.INTER_LINEAR,
                                    borderMode=cv2.BORDER_CONSTANT, borderValue=0)
        m = he_warped > 0
        if np.sum(m) < 1000:
            return 1.0
        return -normalized_cross_correlation(he_warped, xn_grad, m)


    best_tx_init, best_ty_init = init_tx_small, init_ty_small
    best_tx_ncc = -1
    for tx_off in np.arange(-300, 301, 50):
        for ty_off in np.arange(-300, 301, 50):
            ncc = -objective_grad([init_tx_small + tx_off, init_ty_small + ty_off, 0.0, 1.0])
            if ncc > best_tx_ncc:
                best_tx_ncc = ncc
                best_tx_init = init_tx_small + tx_off
                best_ty_init = init_ty_small + ty_off
    coarse_tx, coarse_ty = best_tx_init, best_ty_init
    for tx_off in np.arange(-50, 51, 10):
        for ty_off in np.arange(-50, 51, 10):
            ncc = -objective_grad([coarse_tx + tx_off, coarse_ty + ty_off, 0.0, 1.0])
            if ncc > best_tx_ncc:
                best_tx_ncc = ncc
                best_tx_init = coarse_tx + tx_off
                best_ty_init = coarse_ty + ty_off
    print(f"  Tx grid: tx={best_tx_init/xn_scale:.1f}, ty={best_ty_init/xn_scale:.1f}, NCC={best_tx_ncc:.4f}")


    _, dapi_otsu = cv2.threshold(xn_small, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    dapi_tissue = dapi_otsu > 0
    he_rgb_scaled = None
    if he_rgb is not None:
        he_rgb_rot = rotate_img(he_rgb, rotation_deg)
        he_rgb_scaled = cv2.resize(he_rgb_rot, None, fx=combined_scale, fy=combined_scale,
                                    interpolation=cv2.INTER_LINEAR)

    def _he_tissue_mask(M):
        if he_rgb_scaled is not None:
            he_rgb_warped = cv2.warpAffine(he_rgb_scaled, M, (xn_sw, xn_sh),
                                            flags=cv2.INTER_LINEAR,
                                            borderMode=cv2.BORDER_CONSTANT, borderValue=0)
            hsv = cv2.cvtColor(he_rgb_warped, cv2.COLOR_RGB2HSV)
            return hsv[:, :, 1] > 15
        else:
            he_warped = cv2.warpAffine(he_scaled, M, (xn_sw, xn_sh), flags=cv2.INTER_LINEAR,
                                        borderMode=cv2.BORDER_CONSTANT, borderValue=0)
            return he_warped > 20

    def make_blur_objective(blur_um):
        blur_px = max(3, int(blur_um / um_per_px) | 1)
        xn_b = cv2.GaussianBlur(xn_small.astype(np.float32), (blur_px, blur_px), 0)
        def objective(params):
            M = _make_warp_matrix(*params)
            he_warped = cv2.warpAffine(he_scaled, M, (xn_sw, xn_sh), flags=cv2.INTER_LINEAR,
                                        borderMode=cv2.BORDER_CONSTANT, borderValue=0)
            mask = _he_tissue_mask(M) & dapi_tissue
            if np.sum(mask) < 1000:
                return 1.0
            he_b = cv2.GaussianBlur(he_warped.astype(np.float32), (blur_px, blur_px), 0)
            return -normalized_cross_correlation(
                he_b.astype(np.uint8), xn_b.astype(np.uint8), mask)
        return objective


    obj_coarse = make_blur_objective(200.0)

    best_tx_b, best_ty_b = best_tx_init, best_ty_init
    best_ncc_b = -1
    for tx_off in np.arange(-200, 201, 20):
        for ty_off in np.arange(-200, 201, 20):
            ncc = -obj_coarse([best_tx_init + tx_off, best_ty_init + ty_off, 0.0, 1.0])
            if ncc > best_ncc_b:
                best_ncc_b = ncc
                best_tx_b = best_tx_init + tx_off
                best_ty_b = best_ty_init + ty_off
    coarse_tx_b, coarse_ty_b = best_tx_b, best_ty_b
    for tx_off in np.arange(-30, 31, 5):
        for ty_off in np.arange(-30, 31, 5):
            ncc = -obj_coarse([coarse_tx_b + tx_off, coarse_ty_b + ty_off, 0.0, 1.0])
            if ncc > best_ncc_b:
                best_ncc_b = ncc
                best_tx_b = coarse_tx_b + tx_off
                best_ty_b = coarse_ty_b + ty_off


    best_init, best_init_ncc = (0.0, 1.0), -1
    for dtheta_deg in np.arange(-5.0, 5.5, 0.5):
        dtheta_rad = np.radians(dtheta_deg)
        for ds in [0.90, 0.92, 0.94, 0.96, 0.98, 1.0, 1.02, 1.04, 1.06, 1.08, 1.10]:
            ncc = -obj_coarse([best_tx_b, best_ty_b, dtheta_rad, ds])
            if ncc > best_init_ncc:
                best_init_ncc = ncc
                best_init = (dtheta_rad, ds)


    result = minimize(obj_coarse,
                      [best_tx_b, best_ty_b, best_init[0], best_init[1]],
                      method='Powell', options={'maxiter': 200, 'ftol': 1e-6})
    p = list(result.x)

    if screening:

        best_tx_small, best_ty_small, best_dtheta, best_ds = p
        best_dtheta = np.clip(best_dtheta, -np.radians(15), np.radians(15))
        best_ds = np.clip(best_ds, 0.80, 1.20)
        final_ncc = -obj_coarse([best_tx_small, best_ty_small, best_dtheta, best_ds])
    else:

        obj_fine = make_blur_objective(50.0)


        best_p = list(p)
        best_ncc_fine = -obj_fine(p)
        for tx_off in np.arange(-30, 31, 3):
            for ty_off in np.arange(-30, 31, 3):
                test = [p[0] + tx_off, p[1] + ty_off, p[2], p[3]]
                ncc = -obj_fine(test)
                if ncc > best_ncc_fine:
                    best_ncc_fine = ncc
                    best_p = list(test)


        for ds_off in np.arange(-0.05, 0.051, 0.005):
            test = [best_p[0], best_p[1], best_p[2], p[3] + ds_off]
            ncc = -obj_fine(test)
            if ncc > best_ncc_fine:
                best_ncc_fine = ncc
                best_p = list(test)


        result2 = minimize(obj_fine, best_p,
                           method='Powell', options={'maxiter': 200, 'ftol': 1e-7})
        p2 = list(result2.x)


        result3 = minimize(objective_grad, p2,
                           method='Powell', options={'maxiter': 200, 'ftol': 1e-7})

        best_tx_small, best_ty_small, best_dtheta, best_ds = result3.x
        best_dtheta = np.clip(best_dtheta, -np.radians(15), np.radians(15))
        best_ds = np.clip(best_ds, 0.80, 1.20)
        final_ncc = -obj_fine([best_tx_small, best_ty_small, best_dtheta, best_ds])

    print(f"  Fine: tx={best_tx_small/xn_scale:.1f}, ty={best_ty_small/xn_scale:.1f}, "
          f"dtheta={np.degrees(best_dtheta):.3f}deg, ds={best_ds:.4f}, NCC={final_ncc:.4f}")

    return best_tx_small / xn_scale, best_ty_small / xn_scale, best_dtheta, final_ncc, best_ds


def build_affine_matrix(he_shape, rotation_deg, scale, tx, ty, dtheta=0, ds=1.0):
    he_h, he_w = he_shape[:2]
    theta = np.radians(rotation_deg) + dtheta
    cos_t, sin_t = np.cos(theta), np.sin(theta)
    cx, cy = he_w / 2, he_h / 2

    s = scale * ds
    a = s * cos_t
    b = -s * sin_t
    c = s * (cx - cos_t * cx + sin_t * cy) + tx
    d = s * sin_t
    e = s * cos_t
    f = s * (cy - sin_t * cx - cos_t * cy) + ty

    return np.array([[a, b, c], [d, e, f]], dtype=np.float64)


def warp_full_resolution(he_rgb, M, output_shape, tile_size=4096):
    h, w = output_shape
    output = np.full((h, w, 3), 255, dtype=np.uint8)
    print(f"  Warping to {h}x{w}...")

    for y_start in range(0, h, tile_size):
        for x_start in range(0, w, tile_size):
            y_end = min(y_start + tile_size, h)
            x_end = min(x_start + tile_size, w)
            tile_h, tile_w = y_end - y_start, x_end - x_start

            M_tile = M.copy()
            M_tile[0, 2] -= x_start
            M_tile[1, 2] -= y_start
            M_inv = cv2.invertAffineTransform(M_tile)

            corners = np.array([[0, 0, 1], [tile_w, 0, 1], [tile_w, tile_h, 1], [0, tile_h, 1]], dtype=np.float64)
            src_corners = (M_inv @ corners.T).T
            src_x_min = max(0, int(np.floor(src_corners[:, 0].min())) - 10)
            src_y_min = max(0, int(np.floor(src_corners[:, 1].min())) - 10)
            src_x_max = min(he_rgb.shape[1], int(np.ceil(src_corners[:, 0].max())) + 10)
            src_y_max = min(he_rgb.shape[0], int(np.ceil(src_corners[:, 1].max())) + 10)

            if src_x_max <= src_x_min or src_y_max <= src_y_min:
                continue
            src_region = he_rgb[src_y_min:src_y_max, src_x_min:src_x_max]
            if src_region.size == 0:
                continue

            M_adjusted = M_tile.copy()
            M_adjusted[0, 2] += M_tile[0, 0] * src_x_min + M_tile[0, 1] * src_y_min
            M_adjusted[1, 2] += M_tile[1, 0] * src_x_min + M_tile[1, 1] * src_y_min

            try:
                tile = cv2.warpAffine(src_region, M_adjusted, (tile_w, tile_h),
                                       flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT,
                                       borderValue=(255, 255, 255))
                output[y_start:y_end, x_start:x_end] = tile
            except cv2.error:
                continue

    return output


def save_pyramidal_ome_tiff(img, output_path, pixel_size=0.2125):
    pyramid = [img]
    current = img
    while min(current.shape[:2]) > 256:
        h, w = current.shape[:2]
        new_h, new_w = h // 2, w // 2
        if new_h < 256 or new_w < 256:
            break
        current = cv2.resize(current, (new_w, new_h), interpolation=cv2.INTER_AREA)
        pyramid.append(current)

    print(f"  Saving pyramidal OME-TIFF ({len(pyramid)} levels)...")
    with tifffile.TiffWriter(str(output_path), bigtiff=True, ome=True) as tif:
        options = dict(
            tile=(256, 256), compression='jpeg', photometric='rgb',
            metadata={'axes': 'YXC', 'PhysicalSizeX': pixel_size, 'PhysicalSizeXUnit': 'um',
                      'PhysicalSizeY': pixel_size, 'PhysicalSizeYUnit': 'um'},
        )
        tif.write(pyramid[0], subifds=len(pyramid) - 1, **options)
        for level in pyramid[1:]:
            tif.write(level, subfiletype=1, **options)

    print(f"  Saved: {output_path} ({Path(output_path).stat().st_size / 1024**2:.1f} MB)")


def create_overlay(he_aligned, dapi, output_path, size=2000):
    scale = size / max(dapi.shape[:2])
    new_h, new_w = int(dapi.shape[0] * scale), int(dapi.shape[1] * scale)

    he_small = cv2.resize(he_aligned, (new_w, new_h), interpolation=cv2.INTER_AREA)
    dapi_small = cv2.resize(dapi, (new_w, new_h), interpolation=cv2.INTER_AREA)

    if len(he_small.shape) == 3:
        he_gray = cv2.cvtColor(he_small, cv2.COLOR_RGB2GRAY)
    else:
        he_gray = he_small

    he_norm = cv2.normalize(he_gray, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    dapi_norm = cv2.normalize(dapi_small, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    he_inverted = 255 - he_norm
    he_mask = he_gray > 5

    overlay = np.zeros((new_h, new_w, 3), dtype=np.uint8)
    overlay[:, :, 2] = np.where(he_mask, he_inverted, 0)
    overlay[:, :, 1] = dapi_norm
    cv2.imwrite(str(output_path), overlay)


def bspline_register_tile(he_tile, dapi_tile, grid_spacing=40, iterations=100):
    fixed = sitk.GetImageFromArray(dapi_tile.astype(np.float32))
    moving = sitk.GetImageFromArray(he_tile.astype(np.float32))

    h, w = he_tile.shape
    mesh_size = [max(2, w // grid_spacing), max(2, h // grid_spacing)]
    transform = sitk.BSplineTransformInitializer(fixed, mesh_size, order=3)

    method = sitk.ImageRegistrationMethod()
    method.SetMetricAsMattesMutualInformation(numberOfHistogramBins=50)
    method.SetMetricSamplingStrategy(method.RANDOM)
    method.SetMetricSamplingPercentage(0.15)
    method.SetInterpolator(sitk.sitkLinear)
    method.SetOptimizerAsLBFGSB(gradientConvergenceTolerance=1e-5, numberOfIterations=iterations,
                                 maximumNumberOfCorrections=5, maximumNumberOfFunctionEvaluations=500,
                                 costFunctionConvergenceFactor=1e7)
    method.SetShrinkFactorsPerLevel([2, 1])
    method.SetSmoothingSigmasPerLevel([1, 0])
    method.SmoothingSigmasAreSpecifiedInPhysicalUnitsOn()
    method.SetInitialTransform(transform, inPlace=True)

    try:
        final_transform = method.Execute(fixed, moving)
        disp_filter = sitk.TransformToDisplacementFieldFilter()
        disp_filter.SetReferenceImage(fixed)
        return sitk.GetArrayFromImage(disp_filter.Execute(final_transform))
    except Exception:
        return np.zeros((h, w, 2), dtype=np.float32)


def _process_bspline_tile(task):
    tile_idx, y, x, he_tile, dapi_tile, tile_h, tile_w, grid_spacing, iterations = task
    if he_tile.std() < 10 or dapi_tile.std() < 10:
        return {'tile_idx': tile_idx, 'y': y, 'x': x, 'tile_h': tile_h, 'tile_w': tile_w,
                'skipped': True, 'disp': None}

    current = he_tile.copy()
    tile_disp = np.zeros((tile_h, tile_w, 2), dtype=np.float32)
    for gs, iters in [(100, 60), (60, 80), (grid_spacing, iterations)]:
        disp = bspline_register_tile(current, dapi_tile, gs, iters)
        tile_disp += disp[:tile_h, :tile_w]
        coords_y, coords_x = np.meshgrid(np.arange(tile_h), np.arange(tile_w), indexing='ij')
        new_x = (coords_x + disp[:tile_h, :tile_w, 0]).astype(np.float32)
        new_y = (coords_y + disp[:tile_h, :tile_w, 1]).astype(np.float32)
        current = cv2.remap(current, new_x, new_y, cv2.INTER_LINEAR, borderValue=0)

    return {'tile_idx': tile_idx, 'y': y, 'x': x, 'tile_h': tile_h, 'tile_w': tile_w,
            'skipped': False, 'disp': tile_disp}


def compute_bspline_displacement(he_aligned, dapi, level=2, tile_size=512, overlap=128,
                                  grid_spacing=40, iterations=100, n_workers=32):
    print(f"\n[Stage 3] Computing B-spline Displacement Field...")
    t0 = time.time()

    scale_factor = 2 ** level
    h_small = he_aligned.shape[0] // scale_factor
    w_small = he_aligned.shape[1] // scale_factor
    print(f"  Level {level}: {h_small}x{w_small} (1/{scale_factor})")

    if len(he_aligned.shape) == 3:
        he_small = cv2.resize(extract_hematoxylin(he_aligned), (w_small, h_small), interpolation=cv2.INTER_AREA)
    else:
        he_small = cv2.resize(he_aligned, (w_small, h_small), interpolation=cv2.INTER_AREA)
    dapi_small = cv2.resize(normalize_dapi(dapi), (w_small, h_small), interpolation=cv2.INTER_AREA)

    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    he_clahe = clahe.apply(he_small)
    dapi_clahe = clahe.apply(dapi_small)

    h, w = he_clahe.shape
    step = tile_size - overlap
    tiles_y = list(range(0, max(1, h - tile_size + 1), step))
    tiles_x = list(range(0, max(1, w - tile_size + 1), step))
    if tiles_y[-1] + tile_size < h:
        tiles_y.append(h - tile_size)
    if tiles_x[-1] + tile_size < w:
        tiles_x.append(w - tile_size)

    tasks = []
    for tile_idx_counter, (ty, tx) in enumerate([(y, x) for y in tiles_y for x in tiles_x]):
        th = min(tile_size, h - ty)
        tw = min(tile_size, w - tx)
        tasks.append((tile_idx_counter, ty, tx,
                      he_clahe[ty:ty+th, tx:tx+tw].copy(),
                      dapi_clahe[ty:ty+th, tx:tx+tw].copy(),
                      th, tw, grid_spacing, iterations))

    n_tiles = len(tasks)
    effective_workers = max(1, min(n_workers, cpu_count() // 4, n_tiles))
    print(f"  Tiles: {n_tiles}, Workers: {effective_workers}")

    disp_field = np.zeros((h, w, 2), dtype=np.float32)
    weight_acc = np.zeros((h, w), dtype=np.float32)

    def create_weight_map(th, tw, margin=64):
        weight = np.ones((th, tw), dtype=np.float32)
        for i in range(margin):
            f = i / margin
            weight[i, :] *= f
            weight[th-1-i, :] *= f
            weight[:, i] *= f
            weight[:, tw-1-i] *= f
        return weight

    with Pool(effective_workers, maxtasksperchild=20) as pool:
        results = list(tqdm(pool.imap_unordered(_process_bspline_tile, tasks),
                           total=n_tiles, desc="B-spline"))

    processed = 0
    for result in results:
        if not result['skipped']:
            ry, rx = result['y'], result['x']
            th, tw = result['tile_h'], result['tile_w']
            weight = create_weight_map(th, tw, margin=overlap // 2)
            disp_field[ry:ry+th, rx:rx+tw, 0] += result['disp'][:, :, 0] * weight
            disp_field[ry:ry+th, rx:rx+tw, 1] += result['disp'][:, :, 1] * weight
            weight_acc[ry:ry+th, rx:rx+tw] += weight
            processed += 1

    mask = weight_acc > 0
    disp_field[mask, 0] /= weight_acc[mask]
    disp_field[mask, 1] /= weight_acc[mask]

    print(f"  Processed: {processed}/{n_tiles}, Max disp: {np.max(np.sqrt(disp_field[:,:,0]**2 + disp_field[:,:,1]**2)):.1f} px")
    print(f"  Time: {time.time() - t0:.1f}s")
    return disp_field, scale_factor


def apply_displacement(he_img, disp_field, scale_factor, tile_size=8192):
    print("  Applying displacement field...")
    H, W = he_img.shape[:2]
    is_rgb = len(he_img.shape) == 3

    disp_full = np.zeros((H, W, 2), dtype=np.float32)
    disp_full[:,:,0] = cv2.resize(disp_field[:,:,0], (W, H), interpolation=cv2.INTER_LINEAR) * scale_factor
    disp_full[:,:,1] = cv2.resize(disp_field[:,:,1], (W, H), interpolation=cv2.INTER_LINEAR) * scale_factor

    warped = np.zeros_like(he_img)
    margin = 64

    for ty in range((H + tile_size - 1) // tile_size):
        for tx in range((W + tile_size - 1) // tile_size):
            y1, y2 = ty * tile_size, min((ty + 1) * tile_size, H)
            x1, x2 = tx * tile_size, min((tx + 1) * tile_size, W)
            src_y1, src_y2 = max(0, y1 - margin), min(H, y2 + margin)
            src_x1, src_x2 = max(0, x1 - margin), min(W, x2 + margin)

            coords_y, coords_x = np.meshgrid(np.arange(y1, y2) - src_y1, np.arange(x1, x2) - src_x1, indexing='ij')
            new_x = (coords_x + disp_full[y1:y2, x1:x2, 0]).astype(np.float32)
            new_y = (coords_y + disp_full[y1:y2, x1:x2, 1]).astype(np.float32)
            src_tile = he_img[src_y1:src_y2, src_x1:src_x2]

            if is_rgb:
                for c in range(3):
                    warped[y1:y2, x1:x2, c] = cv2.remap(src_tile[:,:,c], new_x, new_y,
                                                         cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
            else:
                warped[y1:y2, x1:x2] = cv2.remap(src_tile, new_x, new_y,
                                                  cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)

    return warped


def run_pipeline(he_path, dapi_path, xenium_path, output_dir, crop_mode=False,
                 enable_bspline=False, bspline_level=2, bspline_workers=32):
    start_time = time.time()
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("HE-Xenium alignment pipeline")
    print(f"  Mode: {'Crop' if crop_mode else 'Direct'}")
    print(f"  B-spline: {'ON' if enable_bspline else 'OFF'}")
    print(f"  Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 70)


    he_rgb, he_px = load_he_image(he_path)
    dapi, dapi_px = load_dapi(dapi_path, xenium_path)
    target_scale = he_px / dapi_px
    print(f"  Target scale: {target_scale:.4f}")

    metadata = {
        'pipeline': 'standalone_pack',
        'he_path': str(he_path),
        'he_pixel_size': he_px,
        'dapi_pixel_size': dapi_px,
        'target_scale': target_scale,
        'crop_mode': crop_mode,
        'bspline': enable_bspline,
    }


    if crop_mode:
        print(f"\n[Stage 1] Crop Mode Alignment...")
        tissue_boxes = detect_tissues(dapi, dapi_px=dapi_px)

        if tissue_boxes:
            dapi_crops = [(dapi[y1:y2, x1:x2], x1, y1, x2, y2) for x1, y1, x2, y2 in tissue_boxes]
        else:
            dapi_crops = [(dapi, 0, 0, dapi.shape[1], dapi.shape[0])]


        he_tissue_pieces = detect_he_tissues(he_rgb, he_px)

        best_result = None

        def _try_fine_align(he_crop_rgb, dapi_crop, rot, crop_info,
                            d_x1, d_y1, d_x2, d_y2):
            fine_pre = 3000
            dapi_pre_f = min(1.0, fine_pre / max(dapi_crop.shape[:2]))
            if dapi_pre_f < 0.95:
                dapi_crop_small = cv2.resize(dapi_crop, None, fx=dapi_pre_f, fy=dapi_pre_f,
                                             interpolation=cv2.INTER_AREA)
            else:
                dapi_crop_small = dapi_crop
                dapi_pre_f = 1.0
            dapi_crop_norm = normalize_dapi(dapi_crop_small)
            he_pre_f = min(1.0, fine_pre / max(he_crop_rgb.shape[:2]))
            if he_pre_f < 0.95:
                he_crop_small = cv2.resize(he_crop_rgb, None, fx=he_pre_f, fy=he_pre_f,
                                           interpolation=cv2.INTER_AREA)
            else:
                he_crop_small = he_crop_rgb
                he_pre_f = 1.0
            he_nuclei = extract_hematoxylin(he_crop_small)
            adjusted_scale = target_scale * (dapi_pre_f / he_pre_f)
            ftx, fty, dtheta, fncc, ds = fine_alignment(
                he_nuclei, dapi_crop_norm, rot, adjusted_scale, 0.0, 0.0,
                he_rgb=he_crop_small, work_size=1500, dapi_px=dapi_px / dapi_pre_f,
                screening=True)
            ftx = ftx / dapi_pre_f
            fty = fty / dapi_pre_f

            if abs(np.degrees(dtheta)) > 10 or ds < 0.85 or ds > 1.15:
                fncc = fncc * 0.5
            return {
                'rot': rot, 'crop_info': crop_info,
                'he_crop': he_crop_rgb, 'ftx': ftx, 'fty': fty,
                'dtheta': dtheta, 'ds': ds, 'ncc': fncc,
                'dapi_box': (d_x1, d_y1, d_x2, d_y2), 'dapi_crop': dapi_crop,
            }


        coarse_ncc_per_rot = {}
        for dapi_crop, d_x1, d_y1, d_x2, d_y2 in dapi_crops:
            candidates = coarse_align_with_crop(he_rgb, dapi_crop, he_px, dapi_px)
            if candidates is None:
                continue
            for rot, coarse_result in candidates.items():
                cncc = coarse_result.get('ncc', 0)
                if rot not in coarse_ncc_per_rot or cncc > coarse_ncc_per_rot[rot]:
                    coarse_ncc_per_rot[rot] = cncc
                try:
                    he_crop, crop_info = crop_he_region(he_rgb, coarse_result,
                                                        dapi_shape=dapi_crop.shape, he_px=he_px, dapi_px=dapi_px)
                    result = _try_fine_align(he_crop, dapi_crop, rot, crop_info,
                                            d_x1, d_y1, d_x2, d_y2)
                    if best_result is None or result['ncc'] > best_result['ncc']:
                        best_result = result
                except Exception:
                    continue
        pre_sweep_ncc = best_result['ncc'] if best_result else 0.0


        if he_tissue_pieces:


            dapi_area_um2 = dapi.shape[0] * dapi.shape[1] * dapi_px ** 2
            min_piece_area_ratio = 0.30
            min_piece_area_um2 = dapi_area_um2 * min_piece_area_ratio

            eligible = []
            for hx1, hy1, hx2, hy2 in he_tissue_pieces:
                piece_area_um2 = (hx2 - hx1) * (hy2 - hy1) * he_px ** 2
                ratio = piece_area_um2 / dapi_area_um2
                if piece_area_um2 >= min_piece_area_um2:
                    eligible.append((hx1, hy1, hx2, hy2))
                else:
                    print(f"    Skipping piece [{hx1}:{hx2},{hy1}:{hy2}]: "
                          f"area={piece_area_um2/1e6:.1f}mm2 ({ratio:.0%} of DAPI) < {min_piece_area_ratio:.0%}")

            if eligible:
                print(f"\n  Trying {len(eligible)}/{len(he_tissue_pieces)} HE tissue pieces directly...")
                for piece_idx, (hx1, hy1, hx2, hy2) in enumerate(eligible):
                    he_piece = he_rgb[hy1:hy2, hx1:hx2]
                    piece_crop_info = {'x1': hx1, 'y1': hy1, 'x2': hx2, 'y2': hy2,
                                       'original_shape': list(he_rgb.shape)}
                    for dapi_crop, d_x1, d_y1, d_x2, d_y2 in dapi_crops:
                        for rot in [0, 90, 180, 270]:
                            try:
                                result = _try_fine_align(he_piece, dapi_crop, rot, piece_crop_info,
                                                        d_x1, d_y1, d_x2, d_y2)
                                marker = ""
                                if best_result is None or result['ncc'] > best_result['ncc']:
                                    best_result = result
                                    marker = " *NEW BEST*"
                                print(f"    Piece {piece_idx+1} rot={rot:3d}°: NCC={result['ncc']:.4f}{marker}")
                            except Exception:
                                continue


        ncc_threshold = 0.35
        if best_result is None or best_result['ncc'] < ncc_threshold:

            if coarse_ncc_per_rot:
                best_coarse = max(coarse_ncc_per_rot.values())
                sweep_rots = [r for r in [0, 90, 180, 270]
                              if coarse_ncc_per_rot.get(r, 0) >= best_coarse - 0.10]
                if not sweep_rots:
                    sweep_rots = [0, 90, 180, 270]
                print(f"  Sweep: filtering to rotations {sweep_rots} "
                      f"(coarse NCC: {coarse_ncc_per_rot})")
            else:
                sweep_rots = [0, 90, 180, 270]
            he_h, he_w = he_rgb.shape[:2]
            for dapi_crop, d_x1, d_y1, d_x2, d_y2 in dapi_crops:
                dapi_h_px = dapi_crop.shape[0]
                dapi_w_px = dapi_crop.shape[1]

                for rot in sweep_rots:
                    if rot in [0, 180]:
                        exp_h = int(dapi_h_px * dapi_px / he_px)
                        exp_w = int(dapi_w_px * dapi_px / he_px)
                    else:
                        exp_h = int(dapi_w_px * dapi_px / he_px)
                        exp_w = int(dapi_h_px * dapi_px / he_px)
                    pad = int(exp_h * 0.15)
                    crop_h = min(exp_h + 2 * pad, he_h)
                    step = max(2000, crop_h // 3)
                    y_positions = list(range(0, max(1, he_h - crop_h + 1), step))
                    if he_h > crop_h and he_h - crop_h > (y_positions[-1] if y_positions else -1):
                        y_positions.append(he_h - crop_h)
                    if not y_positions:
                        y_positions = [0]

                    for y1 in y_positions:
                        y2 = min(y1 + crop_h, he_h)
                        he_piece = he_rgb[y1:y2, :]
                        piece_crop_info = {'x1': 0, 'y1': y1, 'x2': he_w, 'y2': y2,
                                           'original_shape': list(he_rgb.shape)}
                        try:
                            result = _try_fine_align(he_piece, dapi_crop, rot, piece_crop_info,
                                                    d_x1, d_y1, d_x2, d_y2)
                            marker = ""

                            sweep_min = pre_sweep_ncc + 0.05
                            if (best_result is None or result['ncc'] > best_result['ncc']) \
                                    and result['ncc'] > sweep_min:
                                best_result = result
                                marker = " *NEW BEST*"
                            if result['ncc'] > 0.15 or marker:
                                print(f"  Sweep rot={rot:3d}° y=[{y1},{y2}]: "
                                      f"NCC={result['ncc']:.4f}{marker}")
                        except Exception:
                            continue

                    if best_result is not None and best_result['ncc'] >= ncc_threshold:
                        break
                if best_result is not None and best_result['ncc'] >= ncc_threshold:
                    break

        if best_result is None:
            print("FAILED: No valid alignment found!")
            return False


        print(f"\n[Stage 1C] High-resolution refinement (from NCC={best_result['ncc']:.4f})...")
        refine_pre = 8000
        he_crop_rgb = best_result['he_crop']
        dapi_crop_ref = best_result.get('dapi_crop', dapi)
        dapi_ref_f = min(1.0, refine_pre / max(dapi_crop_ref.shape[:2]))
        he_ref_f = min(1.0, refine_pre / max(he_crop_rgb.shape[:2]))
        if dapi_ref_f < 0.95:
            dapi_ref_small = cv2.resize(dapi_crop_ref, None, fx=dapi_ref_f, fy=dapi_ref_f,
                                        interpolation=cv2.INTER_AREA)
        else:
            dapi_ref_small = dapi_crop_ref
            dapi_ref_f = 1.0
        if he_ref_f < 0.95:
            he_ref_small = cv2.resize(he_crop_rgb, None, fx=he_ref_f, fy=he_ref_f,
                                      interpolation=cv2.INTER_AREA)
        else:
            he_ref_small = he_crop_rgb
            he_ref_f = 1.0
        dapi_ref_norm = normalize_dapi(dapi_ref_small)
        he_ref_nuclei = extract_hematoxylin(he_ref_small)
        ref_scale = target_scale * (dapi_ref_f / he_ref_f)

        ref_init_tx = best_result['ftx'] * dapi_ref_f
        ref_init_ty = best_result['fty'] * dapi_ref_f
        ref_ftx, ref_fty, ref_dtheta, ref_ncc, ref_ds = fine_alignment(
            he_ref_nuclei, dapi_ref_norm, best_result['rot'], ref_scale,
            ref_init_tx, ref_init_ty, he_rgb=he_ref_small, work_size=4000,
            dapi_px=dapi_px / dapi_ref_f)
        ref_ftx /= dapi_ref_f
        ref_fty /= dapi_ref_f
        print(f"  Refined: NCC {best_result['ncc']:.4f} -> {ref_ncc:.4f}")
        if ref_ncc >= best_result['ncc'] - 0.02:
            best_result['ftx'] = ref_ftx
            best_result['fty'] = ref_fty
            best_result['dtheta'] = ref_dtheta
            best_result['ds'] = ref_ds
            best_result['ncc'] = ref_ncc
            print(f"  Using refined result: tx={ref_ftx:.1f}, ty={ref_fty:.1f}, "
                  f"dtheta={np.degrees(ref_dtheta):.3f}deg, ds={ref_ds:.4f}")
        else:
            print(f"  Refinement degraded NCC, keeping original result")

        he_for_warp = best_result['he_crop']
        dapi_for_warp = best_result.get('dapi_crop', dapi)
        M = build_affine_matrix(he_for_warp.shape, best_result['rot'], target_scale,
                                best_result['ftx'], best_result['fty'], best_result['dtheta'], best_result['ds'])

        metadata['crop_info'] = best_result['crop_info']
        metadata['fine'] = {
            'rotation_deg': best_result['rot'], 'tx': float(best_result['ftx']),
            'ty': float(best_result['fty']), 'dtheta_deg': float(np.degrees(best_result['dtheta'])),
            'ds': float(best_result['ds']), 'ncc': float(best_result['ncc']),
        }
        if tissue_boxes and best_result.get('dapi_box'):
            metadata['dapi_tissue_box'] = list(best_result['dapi_box'])

    else:
        print(f"\n[Stage 1] Direct Mode Alignment...")
        he_nuclei = extract_hematoxylin(he_rgb)
        dapi_norm = normalize_dapi(dapi)

        rotation_deg = detect_rotation(he_nuclei, dapi_norm, target_scale)
        ptx, pty = phase_correlation_alignment(he_nuclei, dapi_norm, rotation_deg, target_scale)
        print(f"  Phase correlation: tx={ptx:.1f}, ty={pty:.1f}")
        ftx, fty, dtheta, ncc, ds = fine_alignment(he_nuclei, dapi_norm, rotation_deg, target_scale, ptx, pty,
                                                    he_rgb=he_rgb, dapi_px=dapi_px)

        he_for_warp = he_rgb
        M = build_affine_matrix(he_rgb.shape, rotation_deg, target_scale, ftx, fty, dtheta, ds)
        metadata['fine'] = {
            'rotation_deg': rotation_deg, 'tx': float(ftx), 'ty': float(fty),
            'dtheta_deg': float(np.degrees(dtheta)), 'ds': float(ds), 'ncc': float(ncc),
        }

    np.savetxt(output_dir / "affine_matrix.csv", M, delimiter=',')


    print(f"\n[Stage 2] Warping HE to DAPI space...")
    if crop_mode and best_result.get('dapi_box') and tissue_boxes:
        he_aligned_crop = warp_full_resolution(he_for_warp, M, dapi_for_warp.shape[:2])
        d_x1, d_y1, d_x2, d_y2 = best_result['dapi_box']
        he_aligned = np.full((dapi.shape[0], dapi.shape[1], 3), 255, dtype=np.uint8)
        he_aligned[d_y1:d_y1+he_aligned_crop.shape[0],
                   d_x1:d_x1+he_aligned_crop.shape[1]] = he_aligned_crop
    else:
        he_aligned = warp_full_resolution(he_for_warp, M, dapi.shape[:2])

    print(f"  Aligned shape: {he_aligned.shape}")
    del he_rgb, he_for_warp


    if enable_bspline:
        if not HAS_SITK:
            print("\n[Stage 3] WARNING: SimpleITK not available, skipping B-spline")
        else:
            disp_field, scale_factor = compute_bspline_displacement(
                he_aligned, dapi, level=bspline_level, n_workers=bspline_workers)
            np.save(output_dir / f"displacement_field_level{bspline_level}.npy", disp_field)

            he_aligned = apply_displacement(he_aligned, disp_field, scale_factor)
            metadata['bspline'] = {
                'level': bspline_level,
                'max_displacement': float(np.max(np.sqrt(disp_field[:,:,0]**2 + disp_field[:,:,1]**2))),
            }


    print(f"\n[Output] Saving...")
    save_pyramidal_ome_tiff(he_aligned, output_dir / "he_aligned.ome.tif", dapi_px)
    create_overlay(he_aligned, dapi, output_dir / "overlay.jpg")

    metadata['output_shape'] = list(he_aligned.shape[:2])
    metadata['runtime_seconds'] = time.time() - start_time
    with open(output_dir / "metadata.json", 'w') as f:
        json.dump(metadata, f, indent=2)

    print(f"\n{'='*70}")
    print(f"COMPLETED in {metadata['runtime_seconds']:.0f}s")
    print(f"  NCC: {metadata['fine']['ncc']:.4f}")
    print(f"  Output: {output_dir}")
    print(f"    he_aligned.ome.tif  (pyramidal OME-TIFF)")
    print(f"    overlay.jpg         (Red=HE, Green=DAPI)")
    print(f"    affine_matrix.csv")
    print(f"    metadata.json")
    if enable_bspline and HAS_SITK:
        print(f"    displacement_field_level{bspline_level}.npy")
    print(f"{'='*70}")

    return True


def main():
    parser = argparse.ArgumentParser(description='HE-Xenium alignment pipeline')
    parser.add_argument('--he', required=True, help='Path to HE image (OME-TIFF or QPTIFF)')
    parser.add_argument('--xenium', help='Path to Xenium output directory')
    parser.add_argument('--dapi', help='Path to DAPI image (auto-detected from --xenium if not set)')
    parser.add_argument('-o', '--output', required=True, help='Output directory')
    parser.add_argument('--crop', action='store_true', help='Crop mode: HE contains multiple samples')
    parser.add_argument('--bspline', action='store_true', help='Enable the B-spline refinement stage')
    parser.add_argument('--bspline-level', type=int, default=2, help='B-spline downsampling level (default: 2)')
    parser.add_argument('--bspline-workers', type=int, default=32, help='B-spline parallel workers (default: 32)')
    args = parser.parse_args()

    if not args.dapi and not args.xenium:
        parser.error("Must specify either --dapi or --xenium")

    success = run_pipeline(
        args.he, args.dapi, args.xenium, args.output,
        crop_mode=args.crop,
        enable_bspline=args.bspline,
        bspline_level=args.bspline_level,
        bspline_workers=args.bspline_workers,
    )
    return 0 if success else 1


if __name__ == '__main__':
    exit(main())
