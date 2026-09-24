#!/usr/bin/env python

import argparse
import hashlib
import json
import os
import re
import sys
import traceback
from pathlib import Path

import numpy as np
import tifffile
from skimage.filters import threshold_otsu
from skimage.measure import label, regionprops
from skimage.morphology import binary_closing, disk, remove_small_holes
from skimage.transform import resize


MICRONS_PER_U = 5.0
Z_REFERENCE_U = 1


TARGET_THUMB_MPP = 16.0
CLOSE_RADIUS_UM = 120.0
MIN_BLOCK_AREA_MM2 = 0.10
HOLE_AREA_MM2 = 0.10


def parse_u_number(name: str):
    matches = re.findall(r"[Uu](\d+)(?![0-9])", str(name))
    return int(matches[-1]) if matches else None


def _pick_level(levels, mpp0, target_mpp):
    best, best_err = 0, None
    for i, lvl in enumerate(levels):
        shape = lvl.shape

        w = shape[1] if len(shape) == 3 and shape[2] == 3 else shape[-1]
        w0 = levels[0].shape[1] if len(levels[0].shape) == 3 and levels[0].shape[2] == 3 else levels[0].shape[-1]
        mpp = mpp0 * (w0 / float(w))
        err = abs(np.log(mpp / target_mpp))
        if best_err is None or err < best_err:
            best, best_err = i, err
    return best


def _ome_physical_size_x(tf):
    xml = tf.ome_metadata
    if not xml:
        return None, None
    m = re.search(r'PhysicalSizeX="([0-9.eE+-]+)"', xml)
    if not m:
        return None, None
    val = float(m.group(1))
    unit_m = re.search(r'PhysicalSizeXUnit="([^"]+)"', xml)
    unit = unit_m.group(1) if unit_m else "um"
    if unit in ("µm", "um", "micron", "microns"):
        return val, unit
    if unit == "nm":
        return val / 1000.0, unit
    if unit == "mm":
        return val * 1000.0, unit
    return val, unit


def _tiff_resolution_mpp(page):
    if "XResolution" not in page.tags:
        return None
    num, den = page.tags["XResolution"].value
    if not num or not den:
        return None
    per_unit = num / float(den)
    unit = str(page.tags["ResolutionUnit"].value) if "ResolutionUnit" in page.tags else ""
    if "CENTIMETER" in unit:
        return 1e4 / per_unit
    if "INCH" in unit:
        return 25400.0 / per_unit
    return None


def read_he(path, assumed_mpp=None):
    with tifffile.TiffFile(path) as tf:
        mpp, unit = _ome_physical_size_x(tf)
        source = "OME-XML PhysicalSizeX"
        tag_mpp = _tiff_resolution_mpp(tf.pages[0])
        if mpp is None:
            mpp, source = tag_mpp, "TIFF XResolution tag"
        if mpp is None:
            if assumed_mpp is None:
                raise ValueError("no pixel size in OME-XML nor in the TIFF resolution tags")
            mpp = float(assumed_mpp)
            source = "ASSUMED (file records no pixel size)"
        series = tf.series[0]
        levels = series.levels
        info = {
            "mpp_source": source,
            "mpp_unit": unit,
            "mpp_from_tiff_tag": tag_mpp,
            "shape_level0": list(series.shape),
            "axes": series.axes,
            "n_pyramid_levels": len(levels),
            "level_shapes": [list(l.shape) for l in levels],
            "dtype": str(series.dtype),
        }
        li = _pick_level(levels, mpp, TARGET_THUMB_MPP)
        img = levels[li].asarray()
    if img.ndim == 3 and img.shape[0] == 3 and img.shape[-1] != 3:
        img = np.moveaxis(img, 0, -1)
    img = img.astype(np.float32) / 255.0


    mx = img.max(axis=-1)
    mn = img.min(axis=-1)
    sat = np.where(mx > 0, (mx - mn) / np.maximum(mx, 1e-6), 0.0)
    info["thumb_level"] = li
    info["thumb_mpp"] = mpp * (levels[0].shape[1] / float(levels[li].shape[1]))
    return sat.astype(np.float32), mpp, info


def read_codex(path, dapi_channel=0):
    with tifffile.TiffFile(path) as tf:
        mpp, unit = _ome_physical_size_x(tf)
        series = tf.series[0]
        levels = series.levels
        xml = tf.ome_metadata or ""
        channels = re.findall(r'<Channel ID="Channel:\d+" Name="([^"]*)"', xml)
        info = {
            "mpp_source": "OME-XML PhysicalSizeX",
            "mpp_unit": unit,
            "shape_level0": list(series.shape),
            "axes": series.axes,
            "n_pyramid_levels": len(levels),
            "level_shapes": [list(l.shape) for l in levels],
            "dtype": str(series.dtype),
            "channels": channels,
        }
        if mpp is None:
            raise ValueError("no PhysicalSizeX in OME-XML")
        li = _pick_level(levels, mpp, TARGET_THUMB_MPP)
        arr = levels[li].asarray()
    img = arr[dapi_channel] if arr.ndim == 3 else arr
    img = img.astype(np.float32)
    info["thumb_level"] = li
    info["thumb_mpp"] = mpp * (levels[0].shape[-1] / float(levels[li].shape[-1]))
    info["dapi_channel_name"] = channels[dapi_channel] if channels else None
    return img, mpp, info


def _resolution_ladder(tf, series):
    levels = list(series.levels)
    if len(levels) > 1:
        return [(tuple(l.shape), l.asarray) for l in levels]
    page0 = tf.pages[0]
    subs = list(getattr(page0, "pages", None) or [])
    if subs:
        ladder = [(tuple(page0.shape), page0.asarray)]
        for s in subs:
            ladder.append((tuple(s.shape), s.asarray))
        return ladder
    return [(tuple(levels[0].shape), levels[0].asarray)]


XENIUM_MORPH_CANDIDATES = [
    "morphology_focus/morphology_focus_0000.ome.tif",
    "morphology_focus.ome.tif",
    "morphology_mip.ome.tif",
    "morphology.ome.tif",
]


def find_xenium_morphology(run_dir: Path):
    for rel in XENIUM_MORPH_CANDIDATES:
        p = run_dir / rel
        if p.exists():
            return p
    hits = sorted(run_dir.glob("morphology*"))
    for h in hits:
        if h.is_file() and h.suffix in (".tif", ".tiff"):
            return h
    return None


def read_xenium(run_dir):
    run_dir = Path(run_dir)
    info = {}
    exp = run_dir / "experiment.xenium"
    pixel_size_json = None
    if exp.exists():
        try:
            meta = json.loads(exp.read_text())
            pixel_size_json = meta.get("pixel_size")
            info["experiment_xenium"] = {
                k: meta.get(k)
                for k in (
                    "run_name",
                    "region_name",
                    "instrument_sn",
                    "analysis_sw_version",
                    "panel_name",
                    "num_cells",
                    "pixel_size",
                    "z_step_size",
                    "roi_area",
                )
                if k in meta
            }
        except Exception as exc:
            info["experiment_xenium_error"] = repr(exc)
    morph = find_xenium_morphology(run_dir)
    if morph is None:
        raise FileNotFoundError(f"no morphology image under {run_dir}")
    info["morphology_path"] = str(morph)
    with tifffile.TiffFile(morph) as tf:
        mpp, unit = _ome_physical_size_x(tf)
        series = tf.series[0]
        ladder = _resolution_ladder(tf, series)
        info.update(
            {
                "mpp_source": "morphology OME-XML PhysicalSizeX",
                "mpp_unit": unit,
                "shape_level0": list(series.shape),
                "axes": series.axes,
                "n_pyramid_levels": len(ladder),
                "level_shapes": [list(s) for s, _ in ladder],
                "dtype": str(series.dtype),
                "pixel_size_from_experiment_xenium": pixel_size_json,
            }
        )
        if mpp is None:
            if pixel_size_json is None:
                raise ValueError("no pixel size in morphology OME-XML nor experiment.xenium")
            mpp = float(pixel_size_json)
            info["mpp_source"] = "experiment.xenium pixel_size (OME-XML had none)"
        widths = [s[-1] for s, _ in ladder]
        li = int(np.argmin([abs(np.log((mpp * widths[0] / w) / TARGET_THUMB_MPP)) for w in widths]))
        arr = ladder[li][1]()
    while arr.ndim > 2:
        arr = arr[0]
    info["thumb_level"] = li
    info["thumb_mpp"] = mpp * (widths[0] / float(widths[li]))
    return arr.astype(np.float32), mpp, info


def count_tissue_blocks(thumb, thumb_mpp):
    x = thumb.astype(np.float32)
    finite = np.isfinite(x)
    if not finite.all():
        x = np.where(finite, x, 0.0)
    lo, hi = np.percentile(x, [1, 99.5])
    if hi <= lo:
        return 0, np.zeros(x.shape, bool), {"reason": "flat image"}
    xn = np.clip((x - lo) / (hi - lo), 0, 1)
    thr = float(threshold_otsu(xn))
    mask = xn > thr


    raw_border = {
        "top": round(float(mask[0].mean()), 4),
        "left": round(float(mask[:, 0].mean()), 4),
        "bottom": round(float(mask[-1].mean()), 4),
        "right": round(float(mask[:, -1].mean()), 4),
    }

    px_area_mm2 = (thumb_mpp / 1000.0) ** 2
    close_px = max(1, int(round(CLOSE_RADIUS_UM / thumb_mpp)))
    mask = binary_closing(mask, disk(close_px))
    hole_px = max(1, int(round(HOLE_AREA_MM2 / px_area_mm2)))
    mask = remove_small_holes(mask, area_threshold=hole_px)

    lab = label(mask, connectivity=2)
    min_px = max(1, int(round(MIN_BLOCK_AREA_MM2 / px_area_mm2)))
    blocks = []
    keep = np.zeros_like(mask)
    for r in regionprops(lab):
        if r.area >= min_px:
            blocks.append(
                {
                    "area_mm2": round(r.area * px_area_mm2, 4),
                    "bbox_px": [int(v) for v in r.bbox],
                    "centroid_px": [round(float(v), 1) for v in r.centroid],
                }
            )
            keep |= lab == r.label
    blocks.sort(key=lambda b: -b["area_mm2"])
    stats = {
        "otsu_threshold_normalised": round(thr, 4),
        "thumb_shape": list(x.shape),
        "thumb_mpp": round(float(thumb_mpp), 4),
        "close_radius_px": close_px,
        "min_block_area_px": min_px,
        "tissue_area_mm2": round(float(keep.sum()) * px_area_mm2, 4),
        "raw_border_occupancy": raw_border,
        "max_border_occupancy": round(max(raw_border.values()), 4),
        "blocks": blocks,
    }
    return len(blocks), keep, stats


def mask_signature(mask, size=64):
    small = resize(mask.astype(np.float32), (size, size), order=1, anti_aliasing=True)
    return small.astype(np.float32)


def sha256_head(path, nbytes=1 << 20):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        h.update(fh.read(nbytes))
    return h.hexdigest()


def collect_entries(data_root: Path, sample: str, corrections: dict):
    entries = []
    he_dir = data_root / "HE"
    for p in sorted(he_dir.glob(f"{sample}*")):
        if p.suffix.lower() not in (".tif", ".tiff", ".qptiff"):
            continue
        entries.append({"modality": "he", "path": p})
    for p in sorted((data_root / "Codex").glob(f"{sample}*")):
        if p.suffix.lower() not in (".tif", ".tiff"):
            continue
        entries.append({"modality": "codex", "path": p})
    for p in sorted((data_root / "Xenium").glob(f"*{sample}*")):
        entries.append({"modality": "xenium", "path": p})

    for e in entries:
        name = e["path"].name
        u_file = parse_u_number(name)
        key = f"{e['modality']}:{name}"
        e["u_from_filename"] = u_file
        if key in corrections:
            e["u_number"] = corrections[key]["u"]
            e["u_source"] = corrections[key]["why"]
        else:
            e["u_number"] = u_file
            e["u_source"] = "filename"
    return entries


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", default=os.environ.get("THREED_DATA_ROOT", os.path.join(os.environ.get("PROJECTS_ROOT", "/data/heid"), "data/3d")) + "")
    ap.add_argument("--sample", default="HT891Z1")
    ap.add_argument("--out", required=True)
    ap.add_argument("--corrections", default=None, help="JSON file with U-number corrections")
    ap.add_argument(
        "--assume-he-mpp",
        type=float,
        default=None,
        help="pixel size to use for H&E files that record none; flagged as ASSUMED in the output",
    )
    args = ap.parse_args()

    out = Path(args.out)
    (out / "thumbs").mkdir(parents=True, exist_ok=True)

    corrections = {}
    if args.corrections:
        corrections = json.loads(Path(args.corrections).read_text())

    entries = collect_entries(Path(args.data_root), args.sample, corrections)
    print(f"[z-order] {args.sample}: {len(entries)} candidate section files", flush=True)
    for e in entries:
        print(f"      {e['modality']:7s} U{e['u_number']}  {e['path'].name}", flush=True)

    readers = {"he": read_he, "codex": read_codex, "xenium": read_xenium}
    records = []
    signatures = {}
    for e in entries:
        rec = {
            "sample": args.sample,
            "modality": e["modality"],
            "filepath": str(e["path"]),
            "u_number": e["u_number"],
            "u_from_filename": e["u_from_filename"],
            "u_source": e["u_source"],
        }
        try:
            if e["modality"] == "he":
                thumb, mpp, info = read_he(e["path"], assumed_mpp=args.assume_he_mpp)
            else:
                thumb, mpp, info = readers[e["modality"]](e["path"])
            n_blocks, mask, stats = count_tissue_blocks(thumb, info["thumb_mpp"])
            rec.update(
                {
                    "ok": True,
                    "mpp": mpp,
                    "n_tissue_blocks": n_blocks,
                    "meta": info,
                    "tissue": stats,
                }
            )
            if e["modality"] != "xenium":
                rec["file_bytes"] = e["path"].stat().st_size
                rec["inode"] = e["path"].stat().st_ino
                rec["sha256_first_1MiB"] = sha256_head(e["path"])
            sig = mask_signature(mask)
            signatures[f"{e['modality']}:U{e['u_number']}"] = sig
            np.save(out / "thumbs" / f"{e['modality']}_U{e['u_number']}_mask64.npy", sig)
            print(
                f"[z-order] OK  {e['modality']:7s} U{e['u_number']:<4} mpp={mpp:.6f}  "
                f"blocks={n_blocks}  tissue={stats.get('tissue_area_mm2')} mm2",
                flush=True,
            )
        except Exception as exc:
            rec.update({"ok": False, "error": repr(exc), "traceback": traceback.format_exc()})
            print(f"[z-order] FAIL {e['modality']} U{e['u_number']} {e['path']}: {exc!r}", flush=True)
        records.append(rec)
        (out / "sections_full.json").write_text(json.dumps(records, indent=2, default=str))


    keys = sorted(signatures)
    pairs = []
    for i in range(len(keys)):
        for j in range(i + 1, len(keys)):
            a, b = signatures[keys[i]], signatures[keys[j]]
            az, bz = a - a.mean(), b - b.mean()
            denom = np.sqrt((az**2).sum() * (bz**2).sum())
            ncc = float((az * bz).sum() / denom) if denom > 0 else float("nan")
            pairs.append((keys[i], keys[j], round(ncc, 4)))
    pairs.sort(key=lambda t: -(t[2] if np.isfinite(t[2]) else -1))
    with open(out / "mask_similarity.tsv", "w") as fh:
        fh.write("section_a\tsection_b\tmask_ncc\n")
        for a, b, c in pairs:
            fh.write(f"{a}\t{b}\t{c}\n")


    cols = [
        "section_id",
        "u_number",
        "z_position_um",
        "modality",
        "filepath",
        "mpp",
        "n_tissue_blocks",
        "u_from_filename",
        "u_source",
        "size_x_px",
        "size_y_px",
        "width_um",
        "height_um",
        "tissue_area_mm2",
        "max_border_occupancy",
        "n_pyramid_levels",
        "mpp_source",
        "mpp_from_tiff_tag",
        "ok",
    ]
    rows = []
    for r in records:
        meta = r.get("meta", {})
        shape = meta.get("shape_level0") or []
        axes = meta.get("axes", "")
        if axes.startswith("YXS") and len(shape) == 3:
            sy, sx = shape[0], shape[1]
        elif len(shape) == 3:
            sy, sx = shape[1], shape[2]
        elif len(shape) == 2:
            sy, sx = shape[0], shape[1]
        else:
            sy = sx = None
        mpp = r.get("mpp")
        rows.append(
            {
                "section_id": f"{r['sample']}-U{r['u_number']}",
                "u_number": r["u_number"],
                "z_position_um": (r["u_number"] - Z_REFERENCE_U) * MICRONS_PER_U
                if r["u_number"] is not None
                else None,
                "modality": r["modality"],
                "filepath": r["filepath"],
                "mpp": round(mpp, 8) if mpp else None,
                "n_tissue_blocks": r.get("n_tissue_blocks"),
                "u_from_filename": r["u_from_filename"],
                "u_source": r["u_source"],
                "size_x_px": sx,
                "size_y_px": sy,
                "width_um": round(sx * mpp, 1) if (sx and mpp) else None,
                "height_um": round(sy * mpp, 1) if (sy and mpp) else None,
                "tissue_area_mm2": (r.get("tissue") or {}).get("tissue_area_mm2"),
                "max_border_occupancy": (r.get("tissue") or {}).get("max_border_occupancy"),
                "n_pyramid_levels": meta.get("n_pyramid_levels"),
                "mpp_source": meta.get("mpp_source"),
                "mpp_from_tiff_tag": meta.get("mpp_from_tiff_tag"),
                "ok": r.get("ok"),
            }
        )
    rows.sort(key=lambda d: (d["u_number"] if d["u_number"] is not None else 10**9, d["modality"]))
    with open(out / "sections.tsv", "w") as fh:
        fh.write("\t".join(cols) + "\n")
        for d in rows:
            fh.write("\t".join("" if d[c] is None else str(d[c]) for c in cols) + "\n")

    n_ok = sum(1 for r in records if r.get("ok"))
    print(f"[z-order] done: {n_ok}/{len(records)} sections read; wrote {out/'sections.tsv'}", flush=True)
    return 0 if n_ok == len(records) else 1


if __name__ == "__main__":
    sys.exit(main())
