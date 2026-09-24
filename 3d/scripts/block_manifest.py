#!/usr/bin/env python
from __future__ import annotations
import os

import argparse
import csv
import hashlib
import json
import re
from pathlib import Path

import numpy as np
import tifffile

ROOT = Path(os.environ.get("THREED_ROOT", os.path.join(os.environ.get("PROJECTS_ROOT", "/data/heid"), "3d")))
HE = Path(os.environ.get("THREED_DATA_ROOT", os.path.join(os.environ.get("PROJECTS_ROOT", "/data/heid"), "data/3d")) + "/HE")
PAD_UM = 200.0


def slide_numbers(name: str, block: str) -> list[int]:
    tail = re.split(block, name, flags=re.I)[-1]
    tail = re.sub(r"\(?\d+\s*UM[^)]*\)?", "", tail, flags=re.I)
    return [int(x) for x in re.findall(r"\d+", tail)]


def qptiff_of(d: Path) -> Path | None:
    c = sorted((d / "Scan1").glob("*.qptiff"))
    return c[0] if c else None


def blocks_from_image(qptiff, n_want: int, min_frac: float = 0.02, want_masks: bool = False):
    from scipy.ndimage import binary_closing, binary_fill_holes, label
    with tifffile.TiffFile(qptiff) as t:
        ser = t.series[0]
        H0, W0 = int(ser.shape[0]), int(ser.shape[1])
        img = ser.levels[len(ser.levels) - 1].asarray()
    grey = img[..., :3].mean(-1) if img.ndim == 3 else img
    tis = binary_fill_holes(binary_closing(grey < 225, np.ones((5, 5))))
    lb, _ = label(tis)
    size = np.bincount(lb.ravel()); size[0] = 0
    keep = [i for i in np.argsort(-size) if size[i] > tis.size * min_frac]
    if len(keep) != n_want:
        return None
    sy, sx = H0 / grey.shape[0], W0 / grey.shape[1]
    out = []
    for i in keep:
        ys, xs = np.nonzero(lb == i)
        out.append((int(ys.min() * sy), int((ys.max() + 1) * sy),
                    int(xs.min() * sx), int((xs.max() + 1) * sx), float(ys.mean()), lb == i))
    out.sort(key=lambda b: b[4])
    if want_masks:
        return [b[:4] for b in out], [b[5] for b in out], (sy, sx)
    return [b[:4] for b in out]


def bands(mask: np.ndarray, min_rows: int) -> list[tuple[int, int]]:
    ys = np.flatnonzero(mask.any(1))
    if not len(ys):
        return []
    out, cuts = [], np.flatnonzero(np.diff(ys) > 1)
    for b in np.split(ys, cuts + 1):
        if len(b) >= min_rows:
            out.append((int(b[0]), int(b[-1])))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--min-block-mm", type=float, default=2.0,
                    help="a band shorter than this is scanner dirt, not a block")
    ap.add_argument("--sample", default="S22-27909")
    ap.add_argument("--block", default="A26", help="specimen segment in the directory name")
    ap.add_argument("--patient-mode", choices=["position", "single"], default="position",
                    help="position: the two blocks of a slide are two patients (S22); "
                         "single: one specimen, every section joins one volume (S24)")
    ap.add_argument("--src", default=None, help="delivery directory (default HE/<sample>)")
    ap.add_argument("--out", default=None, help="config directory (default configs/<sample lowercased tail>)")
    ap.add_argument("--skip-dir", action="append", default=[],
                    help="a scan directory to leave out, e.g. a repeated scan of the same sections")
    ap.add_argument("--split-touching", action="store_true",
                    help="when a row projection finds fewer blocks than the name lists, read the blocks "
                         "from the image as connected components (blocks that meet at a corner)")
    a = ap.parse_args()
    SRC = Path(a.src) if a.src else HE / a.sample
    OUT = Path(a.out) if a.out else ROOT / "inference/S22-27909/configs" / a.sample.split("-")[0].lower()

    MASKDIR = OUT / "masks"
    rows, problems = [], []
    for d in sorted(SRC.iterdir()):
        if not d.is_dir() or d.name in a.skip_dir:
            continue
        q = qptiff_of(d)
        if q is None:
            problems.append({"slide": d.name, "why": "no qptiff under Scan1"})
            continue
        with tifffile.TiffFile(q) as t:
            p0 = t.pages[0]
            H0, W0 = int(p0.shape[0]), int(p0.shape[1])
            xr = p0.tags["XResolution"].value
            mpp = 1e4 / (xr[0] / xr[1])
            appmag = ""
            m = re.findall(r"Magnification[^0-9]{0,12}([0-9.]+)", p0.description or "")
            if m:
                appmag = m[0]
            n_levels = len(t.series[0].levels) if t.series else 1

        mp = d / "Scan1" / "SampleMask.tif"
        mk = tifffile.imread(mp)
        mk = (mk > 0) if mk.ndim == 2 else (mk[..., 0] > 0)
        sy, sx = H0 / mk.shape[0], W0 / mk.shape[1]
        bb = bands(mk, int(round(a.min_block_mm * 1000 / mpp / sy)))

        nums = sorted(slide_numbers(d.name, a.block))
        mid = mk.shape[0] / 2


        inferred = False
        img_boxes = None
        if a.split_touching and len(bb) < len(nums):
            got = blocks_from_image(q, len(nums), want_masks=True)
            if got is not None:
                img_boxes, img_masks, mask_scale = got
                bb = [(int(y0 / sy), int(y1 / sy) - 1) for y0, y1, _, _ in img_boxes]
        if img_boxes is not None:
            pass
        elif len(bb) == 1 and len(nums) == 2:
            y0, y1 = bb[0]
            hi = (y0 + y1) / 2 < mid
            nums = [nums[0] if hi else nums[1]]
            inferred = True
        elif len(bb) != len(nums):
            problems.append({"slide": d.name, "why": "block count != numbers in name",
                             "numbers": nums, "n_blocks": len(bb)})
            continue


        thick = "15um" if re.search(r"15\s*UM", d.name, re.I) else "standard"

        pad = int(round(PAD_UM / mpp))
        for k, ((y0, y1), num) in enumerate(zip(bb, nums)):
            hi = (y0 + y1) / 2 < mid
            pos = "top" if (len(bb) > 1 and k == 0) or (len(bb) == 1 and hi) else "bottom"
            mask_path = ""
            if img_boxes is not None:
                iy0, iy1, ix0, ix1 = img_boxes[k]
                X0, X1 = max(0, ix0 - pad), min(W0, ix1 + pad)
                Y0, Y1 = max(0, iy0 - pad), min(H0, iy1 + pad)
                MASKDIR.mkdir(parents=True, exist_ok=True)
                mask_path = str(MASKDIR / f"{a.sample}_U{num}.npz")
                np.savez_compressed(mask_path, mask=img_masks[k],
                                    scale_y=mask_scale[0], scale_x=mask_scale[1])
            else:
                sub = mk[y0:y1 + 1]
                xs = np.flatnonzero(sub.any(0))
                X0 = max(0, int(xs[0] * sx) - pad)
                X1 = min(W0, int((xs[-1] + 1) * sx) + pad)
                Y0 = max(0, int(y0 * sy) - pad)
                Y1 = min(H0, int((y1 + 1) * sy) + pad)
            rows.append({
                "slide": f"{a.sample}_U{num}",
                "svs": str(q),
                "native_mpp": mpp,
                "appmag": appmag,
                "level0_W": X1 - X0,
                "level0_H": Y1 - Y0,
                "crop_x0": X0, "crop_y0": Y0, "crop_x1": X1, "crop_y1": Y1,
                "full_W": W0, "full_H": H0,
                "sample": a.sample,


                "block_position": pos,
                "patient": a.sample if a.patient_mode == "single" else ("P_top" if pos == "top" else "P_bottom"),
                "section_number": num,
                "section_inferred_from_position": inferred,
                "blocks_from_image": img_boxes is not None,
                "block_mask": mask_path,
                "thickness": thick,
                "n_levels": n_levels,
                "source_dir": d.name,


                "tiled": False,
                "samples_per_pixel": 3,
                "mpp_source": "tiff_tag",
                "parent_slide": "",
                "file_name": q.name,
                "block": a.block,
                "he_delivery": f"{a.sample.split('-')[0].lower()}_scan_delivery",
                "restained": False,
            })

    OUT.mkdir(parents=True, exist_ok=True)
    f = OUT / "run_manifest.tsv"
    with open(f, "w", newline="") as fh:
        w = csv.DictWriter(fh, list(rows[0]), delimiter="\t")
        w.writeheader()
        w.writerows(rows)

    area = sum(r["level0_W"] * r["level0_H"] * r["native_mpp"] ** 2 for r in rows) / 1e6
    prov = {"source": str(SRC), "sample": a.sample, "block": a.block,
            "patient_mode": a.patient_mode, "skipped_dirs": a.skip_dir,
            "n_slides": len({r["source_dir"] for r in rows}),
            "n_blocks": len(rows), "bbox_area_mm2": round(area, 1),
            "patients": {p: sorted(r["section_number"] for r in rows if r["patient"] == p)
                         for p in sorted({r["patient"] for r in rows})},
            "problems": problems,
            "blocks_read_from_image": sorted({r["source_dir"] for r in rows if r["blocks_from_image"]}),
            "inferred_from_position": [
                {"slide": r["source_dir"], "section": r["section_number"],
                 "position": r["block_position"]}
                for r in rows if r["section_inferred_from_position"]],
            "thicker_sections": [r["slide"] for r in rows if r["thickness"] != "standard"],
            "manifest_sha256": hashlib.sha256(f.read_bytes()).hexdigest()}
    (OUT / "manifest_provenance.json").write_text(json.dumps(prov, indent=1))

    print(f"{len(rows)} blocks on {prov['n_slides']} slides, "
          f"bounding boxes cover {area:.0f} mm2")
    for p in sorted(prov["patients"]):
        s = prov["patients"][p]
        print(f"  {p}: {len(s)} sections  {s[:6]} ... {s[-3:]}")
    if problems:
        print(f"  {len(problems)} slides NOT written -- resolve before running:")
        for q in problems:
            print(f"    {q}")


if __name__ == "__main__":
    main()
