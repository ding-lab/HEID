#!/usr/bin/env python
from __future__ import annotations
import argparse
import csv
import json
import re
from pathlib import Path

import numpy as np
import tifffile

MICRONS_PER_U = 5.0
TILE = 512
THUMB_LEVEL = 4


def pyramid_write(path: Path, img: np.ndarray, mpp: float, levels: int = 5):
    res = (1e4 / mpp, 1e4 / mpp)
    tmp = path.with_suffix(f".tmp{path.suffix}")
    with tifffile.TiffWriter(tmp, bigtiff=True, ome=True) as tw:
        opts = dict(tile=(TILE, TILE), compression="jpeg", compressionargs={"level": 92},
                    photometric="rgb", resolution=res, resolutionunit="CENTIMETER")
        tw.write(img, subifds=levels - 1,
                 metadata={"PhysicalSizeX": mpp, "PhysicalSizeXUnit": "µm",
                           "PhysicalSizeY": mpp, "PhysicalSizeYUnit": "µm"}, **opts)
        cur = img
        for _ in range(levels - 1):
            cur = cur[::2, ::2]
            tw.write(cur, subfiletype=1, **opts)
    tmp.replace(path)


def block_mask_for(row, X0, Y0, X1, Y1):
    f = (row.get("block_mask") or "").strip()
    if not f or f == "nan" or not Path(f).exists():
        return None
    z = np.load(f)
    m, sy, sx = z["mask"].astype(bool), float(z["scale_y"]), float(z["scale_x"])
    ys = np.clip((np.arange(Y0, Y1) / sy).astype(int), 0, m.shape[0] - 1)
    xs = np.clip((np.arange(X0, X1) / sx).astype(int), 0, m.shape[1] - 1)
    return m[ys][:, xs]


def tissue_area_mm2(q: Path, X0, Y0, X1, Y1, mpp, mask):
    from skimage.filters import threshold_otsu
    with tifffile.TiffFile(q) as tf:
        lv = tf.series[0].levels
        k = min(THUMB_LEVEL, len(lv) - 1)
        th = np.asarray(lv[k].asarray())
        H0 = int(tf.pages[0].shape[0])
    if th.ndim == 3:
        th = th.mean(2)
    s = H0 / th.shape[0]
    y0, y1, x0, x1 = int(Y0 / s), int(np.ceil(Y1 / s)), int(X0 / s), int(np.ceil(X1 / s))
    t = 255.0 - th[y0:y1, x0:x1].astype(np.float32)
    if t.size == 0:
        return 0.0
    tissue = t > threshold_otsu(t) if t.max() > t.min() else np.zeros_like(t, bool)
    if mask is not None:
        mk = mask[::max(1, int(round(s))), ::max(1, int(round(s)))]
        mk = mk[:tissue.shape[0], :tissue.shape[1]]
        tissue[:mk.shape[0], :mk.shape[1]] &= mk
    return round(float(tissue.sum()) * (mpp * s) ** 2 / 1e6, 4)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--sample", required=True)
    ap.add_argument("--out-dir", required=True, dest="out_dir")
    ap.add_argument("--sections-tsv", required=True, dest="sections_tsv")
    ap.add_argument("--index", type=int, default=None)
    ap.add_argument("--tsv-only", action="store_true", dest="tsv_only")
    a = ap.parse_args()
    OUT = Path(a.out_dir); OUT.mkdir(parents=True, exist_ok=True)
    man = [r for r in csv.DictReader(open(a.manifest), delimiter="\t") if r["sample"] == a.sample]
    man.sort(key=lambda r: int(r["section_number"]))
    rows = []
    for i, r in enumerate(man):
        u = int(r["section_number"]); q = Path(r["svs"]); mpp = float(r["native_mpp"])
        X0, Y0, X1, Y1 = (int(float(r[k])) for k in ("crop_x0", "crop_y0", "crop_x1", "crop_y1"))
        sid = f"{a.sample}_U{u}"
        out = OUT / f"{sid}.ome.tif"
        mask = None
        if a.tsv_only or a.index is None or a.index == i:
            mask = block_mask_for(r, X0, Y0, X1, Y1)
        area = tissue_area_mm2(q, X0, Y0, X1, Y1, mpp, mask) if (a.tsv_only or a.index is None) else None
        rows.append({
            "section_id": sid, "u_number": u, "z_position_um": round(u * MICRONS_PER_U, 3),
            "modality": "he", "filepath": str(out), "mpp": mpp, "n_tissue_blocks": 1,
            "u_from_filename": u, "u_source": "run_manifest",
            "size_x_px": X1 - X0, "size_y_px": Y1 - Y0,
            "width_um": round((X1 - X0) * mpp, 1), "height_um": round((Y1 - Y0) * mpp, 1),
            "tissue_area_mm2": area, "n_components": 1, "max_border_occupancy": 0.0,
            "n_pyramid_levels": 5, "mpp_source": r.get("mpp_source", "tiff_tag"),
            "mpp_from_tiff_tag": mpp, "ok": True, "mpp_assumed": False,
            "patient": "M", "block_position": r.get("block_position", ""),
            "section_inferred_from_position": r.get("section_inferred_from_position", "False"),
            "thickness": r.get("thickness", "standard"),
            "source_slide": r.get("slide", ""), "source_qptiff": str(q),
            "crop_x0": X0, "crop_y0": Y0, "crop_x1": X1, "crop_y1": Y1,
            "block_mask": r.get("block_mask", ""),
        })
        if a.tsv_only or (a.index is not None and a.index != i):
            continue
        if out.exists():
            print(f"  {sid}: exists, skipped", flush=True); continue
        with tifffile.TiffFile(q) as tf:
            import zarr
            zg = zarr.open(tf.series[0].aszarr(), mode="r")
            z0 = zg["0"] if hasattr(zg, "keys") and "0" in zg else zg
            img = np.asarray(z0[Y0:Y1, X0:X1])
        if img.ndim == 2:
            img = np.repeat(img[:, :, None], 3, 2)
        img = np.ascontiguousarray(img[:, :, :3])
        if mask is not None:
            img[~mask] = 255
        pyramid_write(out, img, mpp)
        print(f"  {sid}  {X1-X0}x{Y1-Y0}  mask={'yes' if mask is not None else 'no'}  {out.stat().st_size/1e6:.0f} MB", flush=True)
    if a.tsv_only or a.index is None:
        Path(a.sections_tsv).parent.mkdir(parents=True, exist_ok=True)
        with open(a.sections_tsv, "w", newline="") as fh:
            w = csv.DictWriter(fh, list(rows[0]), delimiter="\t"); w.writeheader(); w.writerows(rows)
        Path(a.sections_tsv).with_name("split_provenance.json").write_text(json.dumps({
            "source": a.manifest, "sample": a.sample, "microns_per_u": MICRONS_PER_U,
            "n_sections": len(rows), "u_numbers": [r["u_number"] for r in rows],
            "patients": {"M": [r["u_number"] for r in rows]}, "inferred_from_position": [],
            "masked": [r["section_id"] for r in rows if r["block_mask"]],
            "crop_boxes": "the inference run manifest (one source for prep, registration and the classifier)"}, indent=1))
        print(f"{len(rows)} sections -> {a.sections_tsv}")


if __name__ == "__main__":
    main()
