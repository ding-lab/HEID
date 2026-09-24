#!/usr/bin/env python
from __future__ import annotations
import argparse
import csv
import json
import re
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import tifffile
from matplotlib.patches import Rectangle
from scipy import ndimage as ndi

THUMB_UM = 20.0
EDGE_BAND_PX = 2
COLS = ["#1f77b4", "#ff7f0e", "#2ca02c", "#9467bd"]


def level_image(path, target_um, mpp0):
    with tifffile.TiffFile(path) as tf:
        s = tf.series[0]
        yi = s.axes.index("Y")
        h0 = s.levels[0].shape[yi]
        best = len(s.levels) - 1
        for k, lv in enumerate(s.levels):
            if mpp0 * h0 / lv.shape[yi] >= target_um * 0.5:
                best = k
                break
        lv = s.levels[best]
        best_mpp = mpp0 * h0 / lv.shape[yi]
        fluo = "C" in lv.axes and lv.shape[lv.axes.index("C")] > 4
        a = lv.pages[0].asarray() if fluo else lv.asarray()
    if fluo:
        g = a.astype(np.float32)
        lo, hi = np.percentile(g[g > 0], (1, 99.5)) if (g > 0).any() else (0.0, 1.0)
        g = np.clip((g - lo) / max(hi - lo, 1e-6), 0, 1)
        return (np.stack([g] * 3, -1) * 255).astype(np.uint8), best_mpp
    if a.ndim == 3 and a.shape[0] in (3, 4) and a.shape[-1] not in (3, 4):
        a = np.moveaxis(a, 0, -1)
    if a.ndim == 2:
        a = np.stack([a] * 3, -1)
    a = a[..., :3]
    if a.dtype != np.uint8:
        a = (a / max(1, a.max()) * 255).astype(np.uint8)
    return a, best_mpp


def slide_name(scan):
    p = Path(scan)
    return p.parent.parent.name if p.parent.name.lower().startswith("scan") else p.name


def tissue_mask(rgb):
    g = rgb.astype(np.float32)
    sat = g.max(-1) - g.min(-1)
    if sat.max() == 0:
        from skimage.filters import threshold_otsu
        d = ndi.gaussian_filter(g[..., 0], 2)
        m = d > np.expm1(threshold_otsu(np.log1p(d)))
    else:
        m = (g.mean(-1) < 215) & (sat > 18)
    m = ndi.binary_opening(m, iterations=1)
    m = ndi.binary_closing(m, iterations=2)
    m = ndi.binary_fill_holes(m)
    lab, n = ndi.label(m)
    if n:
        sizes = ndi.sum(m, lab, range(1, n + 1))
        keep = np.isin(lab, 1 + np.nonzero(sizes >= max(50, 0.002 * m.size))[0])
        m = keep
    return m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sections", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--only", default="", help="U numbers, e.g. 45,46 or U45,U46: only the slides that carry them")
    a = ap.parse_args()
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    rows = list(csv.DictReader(open(a.sections), delimiter="\t"))


    for js in {Path(r["filepath"]).parent for r in rows if r.get("filepath")}:
        for j in js.glob("*_shared_scan_crops.json"):
            prov = json.loads(j.read_text())
            for r in rows:
                c = prov["crops"].get(r["section_id"])
                if c:
                    r["source_qptiff"] = prov["scan"]
                    r["crop_x0"], r["crop_y0"], r["crop_x1"], r["crop_y1"] = c["crop_x0y0x1y1_px"]
    only = {int(re.sub(r"\D", "", t)) for t in a.only.split(",") if t.strip()}
    by_slide = {}
    for r in rows:
        if r.get("source_qptiff"):
            by_slide.setdefault(r["source_qptiff"], []).append(r)
    table = []
    for scan, secs in sorted(by_slide.items(), key=lambda kv: min(int(x["u_number"]) for x in kv[1])):
        us = sorted(int(x["u_number"]) for x in secs)
        if only and not (set(us) & only):
            continue
        mpp0 = float(secs[0]["mpp"])
        thumb, tm = level_image(scan, THUMB_UM, mpp0)
        f = mpp0 / tm
        tis = tissue_mask(thumb)
        lab, n = ndi.label(tis)
        inside_any = np.zeros_like(tis)
        boxes = []
        for r in sorted(secs, key=lambda x: int(x["u_number"])):
            x0, y0, x1, y1 = (float(r[k]) * f for k in ("crop_x0", "crop_y0", "crop_x1", "crop_y1"))
            X0, Y0 = max(0, int(np.floor(x0))), max(0, int(np.floor(y0)))
            X1, Y1 = min(tis.shape[1], int(np.ceil(x1))), min(tis.shape[0], int(np.ceil(y1)))
            inside_any[Y0:Y1, X0:X1] = True
            boxes.append((r, (x0, y0, x1, y1), (X0, Y0, X1, Y1)))
        outside = np.zeros_like(tis)
        px_mm2 = (tm / 1000.0) ** 2
        for r, _, (X0, Y0, X1, Y1) in boxes:
            inside = tis[Y0:Y1, X0:X1]

            comps = np.unique(lab[Y0:Y1, X0:X1][inside])
            comps = comps[comps > 0]
            block = np.isin(lab, comps)
            left_out = block & ~inside_any
            outside |= left_out
            b = EDGE_BAND_PX
            cut = []
            if inside[:, :b].any() and X0 > 0 and block[Y0:Y1, max(0, X0 - b):X0].any(): cut.append("left")
            if inside[:, -b:].any() and block[Y0:Y1, X1:X1 + b].any(): cut.append("right")
            if inside[:b, :].any() and Y0 > 0 and block[max(0, Y0 - b):Y0, X0:X1].any(): cut.append("top")
            if inside[-b:, :].any() and block[Y1:Y1 + b, X0:X1].any(): cut.append("bottom")
            table.append({"section_id": r["section_id"], "u_number": r["u_number"],
                          "slide": slide_name(scan), "block_position": r.get("block_position", ""),
                          "crop_x0": r["crop_x0"], "crop_y0": r["crop_y0"], "crop_x1": r["crop_x1"], "crop_y1": r["crop_y1"],
                          "tissue_in_box_mm2": round(float(inside.sum()) * px_mm2, 2),
                          "block_tissue_outside_boxes_mm2": round(float(left_out.sum()) * px_mm2, 2),
                          "edges_cutting_tissue": ";".join(cut)})

        ncrop = len(boxes)
        fig = plt.figure(figsize=(6 + 4.2 * ncrop, 7.5))
        gs = fig.add_gridspec(1, 1 + ncrop, width_ratios=[1.5] + [1] * ncrop, wspace=0.08)
        ax = fig.add_subplot(gs[0, 0])
        ax.imshow(thumb, interpolation="bilinear")
        if outside.any():
            red = np.zeros((*outside.shape, 4), np.float32); red[outside] = [0.9, 0.05, 0.05, 0.55]
            ax.imshow(red, interpolation="nearest")
        for k, (r, (x0, y0, x1, y1), _) in enumerate(boxes):
            c = COLS[k % len(COLS)]
            ax.add_patch(Rectangle((x0, y0), x1 - x0, y1 - y0, fill=False, ec=c, lw=2.2))
            ax.text(x0 + 6, y0 + 6, f"U{r['u_number']}  {r.get('block_position', '')}", color="white", fontsize=11,
                    fontweight="bold", va="top", bbox=dict(boxstyle="round,pad=0.2", fc=c, ec="none", alpha=0.9))
        ax.set_title(f"{slide_name(scan)}  ·  scan at {tm:.0f} µm/px\n"
                     "boxes = crop boxes of the sections table · red = tissue of a cropped block left outside the boxes",
                     fontsize=10)
        ax.axis("off")
        for k, (r, _, _) in enumerate(boxes):
            axc = fig.add_subplot(gs[0, 1 + k])
            try:
                im, cm = level_image(r["filepath"], THUMB_UM, float(r["mpp"]))
                axc.imshow(im, interpolation="bilinear")
                t = next(x for x in table if x["section_id"] == r["section_id"])
                axc.set_title(f"{r['section_id']} (crop file)\n{int(r['size_x_px'])}×{int(r['size_y_px'])} px · "
                              f"tissue {t['tissue_in_box_mm2']} mm²" + (f"\ncuts tissue: {t['edges_cutting_tissue']}"
                                                                        if t["edges_cutting_tissue"] else ""),
                              fontsize=9.5, color=("#c00000" if t["edges_cutting_tissue"] else "black"))
            except Exception as e:
                axc.text(0.5, 0.5, f"{r['section_id']}\ncannot read: {e}", ha="center", va="center", fontsize=9)
            for sp in axc.spines.values():
                sp.set_edgecolor(COLS[k % len(COLS)]); sp.set_linewidth(2.2)
            axc.set_xticks([]); axc.set_yticks([])
        name = "-".join(f"U{u}" for u in us) + "_crops.png"
        fig.savefig(out / name, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"  {name}: " + "; ".join(f"{t['section_id']} in {t['tissue_in_box_mm2']} mm2, block outside "
                                         f"{t['block_tissue_outside_boxes_mm2']} mm2, cuts [{t['edges_cutting_tissue']}]"
                                         for t in table if int(t["u_number"]) in us), flush=True)
    if table:
        with open(out / "3_crops.tsv", "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(table[0]), delimiter="\t")
            w.writeheader(); w.writerows(sorted(table, key=lambda t: int(t["u_number"])))
    print(f"wrote {out}: {len({t['slide'] for t in table})} slides, {len(table)} sections")


if __name__ == "__main__":
    main()
