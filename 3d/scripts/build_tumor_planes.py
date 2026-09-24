#!/usr/bin/env python3

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image
from scipy.ndimage import (binary_closing, binary_fill_holes, binary_opening,
                           gaussian_filter, label, zoom)

HERE = Path(__file__).resolve().parent


def _load(name, p):
    import importlib.util
    spec = importlib.util.spec_from_file_location(name, p)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


BM = _load("bm", HERE / "build_meshes.py")
TP = _load("tlsplanes", HERE / "build_tls_planes.py")

UM = 8.0
RESOLUTION = 10
SIGMA_ECO = 10
SIGMA_REGION = 5
CLOSING_R = 5
OPENING_R = 3
MIN_REGION_AREA = 500
MAX_HOLE_AREA = 50000
DENSITY_THRESH = 0.1
PARAMS_OVERRIDE = {
    "PDAC": {"SIGMA_REGION": 3, "CLOSING_R": 2, "MIN_REGION_AREA": 100, "DENSITY_THRESH": 0.03},
}
COLOUR = "#e0245e"


def disk(radius: int) -> np.ndarray:
    r = int(radius)
    y, x = np.ogrid[-r:r + 1, -r:r + 1]
    return (x ** 2 + y ** 2) <= r ** 2


def fill_small_holes(mask, max_hole_area):
    filled = binary_fill_holes(mask)
    holes = filled & ~mask
    if not holes.any():
        return filled
    lb_h, n_h = label(holes)
    sizes = np.bincount(lb_h.ravel())
    for idx in range(1, n_h + 1):
        if sizes[idx] > max_hole_area:
            filled[lb_h == idx] = False
    return filled


def clean_mask(mask, closing_r, min_region_area):
    mask = binary_closing(mask, structure=disk(closing_r))
    mask = fill_small_holes(mask, MAX_HOLE_AREA)
    mask = binary_opening(mask, structure=disk(OPENING_R))
    lb, n = label(mask)
    sizes = np.bincount(lb.ravel())
    for idx in range(1, n + 1):
        if sizes[idx] < min_region_area:
            mask[lb == idx] = False
    return mask


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--volume", required=True)
    ap.add_argument("--params", default="PDAC",
                    help="parameter block: PDAC = the clinical tumour territory (sigma 3, closing 2, min area 100 px, "
                         "0.03 of peak); '' = boundary defaults")
    a = ap.parse_args()
    vol = Path(a.volume)
    ovr = PARAMS_OVERRIDE.get(a.params, {})
    sigma_r = ovr.get("SIGMA_REGION", SIGMA_REGION)
    closing_r = ovr.get("CLOSING_R", CLOSING_R)
    min_area = ovr.get("MIN_REGION_AREA", MIN_REGION_AREA)
    dens_thr = ovr.get("DENSITY_THRESH", DENSITY_THRESH)
    print(f"tumour territory: raster {RESOLUTION} um, sigma_region {sigma_r}, "
          f"threshold {dens_thr} of peak, closing {closing_r}, opening {OPENING_R}, "
          f"min area {min_area} px, holes <= {MAX_HOLE_AREA} px"
          + (f"  [{a.params} override]" if ovr else "  [boundary defaults]"), flush=True)

    meta = json.loads((vol / "cell_points_metadata.json").read_text())
    vmeta = json.loads((vol / "metadata.json").read_text())
    classes = meta["classes"]
    tum = [i for i, c in enumerate(classes) if str(c).startswith("Tumor")]
    if not tum:
        raise SystemExit("no Tumor class in this volume")
    cw, ch = meta["canvas_px"]["width"], meta["canvas_px"]["height"]
    nx = int(np.ceil(cw * UM / RESOLUTION))
    ny = int(np.ceil(ch * UM / RESOLUTION))
    bases = sorted({f["basis"] for f in meta["files"]})
    planes = sorted(vmeta["encodings"]["G_withdrawn"]["planes"], key=lambda p: p["z_um"])
    out_planes, masks = [], {}
    for p in planes:
        cand = sorted((vol / "G_withdrawn/cell_points").glob(f"z{p['index']:02d}_*.cells.bin"))
        if not cand:
            continue
        xy, cls, _ = BM.read_points(cand[0])
        xi = np.clip((xy[:, 0] * UM / RESOLUTION).astype(int), 0, nx - 1)
        yi = np.clip((xy[:, 1] * UM / RESOLUTION).astype(int), 0, ny - 1)
        allr = np.zeros((ny, nx), np.float32)
        np.add.at(allr, (yi, xi), 1.0)
        tissue = gaussian_filter(allr, sigma=SIGMA_ECO) > 0.015
        tissue = binary_fill_holes(binary_closing(tissue, structure=disk(10)))
        sel = np.isin(cls, tum)
        if int(sel.sum()) < 10:
            m10 = np.zeros((ny, nx), bool)
        else:
            raw = np.zeros((ny, nx), np.float32)
            np.add.at(raw, (yi[sel], xi[sel]), 1.0)
            d = gaussian_filter(raw, sigma=sigma_r)
            m10 = (clean_mask(d > dens_thr * d.max(), closing_r, min_area)
                   if d.max() > 0 else np.zeros((ny, nx), bool))
        m10 &= tissue
        n = int(label(m10)[1])
        area = round(float(m10.sum()) * (RESOLUTION / 1000.0) ** 2, 3)
        tissue_area = round(float(tissue.sum()) * (RESOLUTION / 1000.0) ** 2, 3)

        m8 = np.asarray(Image.fromarray((m10 * 255).astype(np.uint8)).resize((cw, ch), Image.NEAREST)) > 127
        masks[p["section_id"]] = m8
        u = p["section_id"].split("-")[-1]
        fn = f"z{p['index']:02d}_{u}.tumor.webp"
        if m8.any():
            rgba = TP.ring(m8)
            for b in bases:
                d2 = vol / b / "tumor"
                d2.mkdir(parents=True, exist_ok=True)
                Image.fromarray(rgba).save(d2 / fn, format="WEBP", lossless=True, quality=100, method=4)
        out_planes.append({"index": p["index"], "section_id": p["section_id"], "z_um": p["z_um"],
                           "file": fn if m8.any() else None, "n_regions": n,
                           "n_tumor_cells": int(sel.sum()), "area_mm2": area,
                           "tissue_area_mm2": tissue_area})
        print(f"  {p['section_id']}: tumour {area} mm2 in {n} regions "
              f"({int(sel.sum())} tumour cells) of {tissue_area} mm2 tissue", flush=True)
    np.savez_compressed(vol / "tumor_planes.npz", **masks)
    (vol / "tumor_regions_metadata.json").write_text(json.dumps({
        "what": "2-D tumour territory per section, tumour-boundary recipe "
                f"(raster {RESOLUTION} um, tissue sigma {SIGMA_ECO} > 0.015 closed/filled; "
                f"tumour density sigma {sigma_r} > {dens_thr} of peak, closing {closing_r}, "
                f"holes <= {MAX_HOLE_AREA} px, opening {OPENING_R}, components >= {min_area} px, "
                "intersected with the tissue mask)",
        "source": "the tumour-boundary-area recipe (see this script's docstring for the full definition)",
        "params": (a.params or "boundary defaults"),
        "colour": COLOUR, "bases": bases,
        "n_planes": len([q for q in out_planes if q["file"]]),
        "planes": [q for q in out_planes if q["file"]],
        "sections_note": "every cell plane, with or without territory: tissue_area_mm2 is the recipe's "
                         "tissue mask (all-cell density), area_mm2 the tumour territory inside it",
        "sections": out_planes}, indent=1))
    tot = sum(q["area_mm2"] for q in out_planes)
    print(f"wrote {vol / 'tumor_regions_metadata.json'}: {len(out_planes)} planes, "
          f"{tot:.2f} mm2 of territory in total; masks in tumor_planes.npz", flush=True)


if __name__ == "__main__":
    main()
