#!/usr/bin/env python
from __future__ import annotations
import argparse
import csv
from pathlib import Path

import numpy as np
from scipy import ndimage as ndi

from crop_check_figures import level_image

THUMB_UM = 16.0
PAD_UM = 200.0
MIN_PIECE_MM2 = 0.05


def tissue(rgb):
    g = rgb.astype(np.float32)
    sat = g.max(-1) - g.min(-1)
    m = (g.mean(-1) < 220) & (sat > 15)
    m = ndi.binary_opening(m, iterations=1)
    m = ndi.binary_closing(m, iterations=2)
    return ndi.binary_fill_holes(m)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--sample", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--mask-dir", required=True)
    ap.add_argument("--near-mm", type=float, default=1.0)
    a = ap.parse_args()
    rows = list(csv.DictReader(open(a.manifest), delimiter="\t"))
    cols = list(rows[0].keys())
    by = {}
    for r in rows:
        if r["sample"] == a.sample:
            by.setdefault(r["svs"], []).append(r)
    for svs, blocks in by.items():
        blocks.sort(key=lambda r: int(r["section_number"]))
        mpp0 = float(blocks[0]["native_mpp"])
        th, tm = level_image(svs, THUMB_UM, mpp0)
        f = mpp0 / tm
        H0, W0 = int(blocks[0]["full_H"]), int(blocks[0]["full_W"])
        px_mm2 = (tm / 1000.0) ** 2
        lab, n = ndi.label(tissue(th))
        size = np.asarray(ndi.sum(np.ones_like(lab), lab, range(1, n + 1)))
        ok = np.zeros(n + 1, bool); ok[1:] = size * px_mm2 >= MIN_PIECE_MM2
        box = [tuple(float(r[k]) for k in ("crop_x0", "crop_y0", "crop_x1", "crop_y1")) for r in blocks]
        owner = np.full(n + 1, -1)
        for k, (x0, y0, x1, y1) in enumerate(box):
            ids = np.unique(lab[int(y0 * f):int(np.ceil(y1 * f)), int(x0 * f):int(np.ceil(x1 * f))])
            for i in ids[ids > 0]:
                if ok[i]:
                    owner[i] = k if owner[i] in (-1, k) else -2
        dist = []
        for k in range(len(blocks)):
            own = np.isin(lab, np.flatnonzero(owner == k))
            dist.append(ndi.distance_transform_edt(~own) if own.any() else np.full(lab.shape, np.inf))
        near_px = a.near_mm * 1000.0 / tm
        for i in np.flatnonzero(ok & (owner == -1)):
            pix = lab == i
            d = [float(D[pix].min()) for D in dist]
            k = int(np.argmin(d))
            if d[k] <= near_px:
                owner[i] = k
        pad = PAD_UM / tm
        for k, r in enumerate(blocks):
            tag = f"U{r['section_number']} {r.get('block_position', '')}"
            if (r.get("block_mask") or "").strip() not in ("", "nan"):
                print(f"  {tag}: has a block_mask, box and mask kept", flush=True)
                continue
            own = np.isin(lab, np.flatnonzero(owner == k))
            if not own.any():
                print(f"  {tag}: no tissue found in its box, kept", flush=True)
                continue
            ys, xs = np.nonzero(own)
            x0, y0, x1, y1 = box[k]
            X0 = max(0, int(np.floor(min(x0, (xs.min() - pad) / f))))
            Y0 = max(0, int(np.floor(min(y0, (ys.min() - pad) / f))))
            X1 = min(W0, int(np.ceil(max(x1, (xs.max() + 1 + pad) / f))))
            Y1 = min(H0, int(np.ceil(max(y1, (ys.max() + 1 + pad) / f))))
            others = np.isin(lab, np.flatnonzero((owner >= 0) & (owner != k)))
            t0, t1 = (int(Y0 * f), int(X0 * f)), (int(np.ceil(Y1 * f)), int(np.ceil(X1 * f)))
            gained = own.copy()
            gained[int(y0 * f):int(np.ceil(y1 * f)), int(x0 * f):int(np.ceil(x1 * f))] = False
            note = ""
            if others[t0[0]:t1[0], t0[1]:t1[1]].any():
                m = ndi.binary_dilation(own, iterations=max(1, int(round(pad)))) & ~others
                Path(a.mask_dir).mkdir(parents=True, exist_ok=True)
                mp = Path(a.mask_dir) / f"{a.sample}_U{r['section_number']}.npz"
                np.savez_compressed(mp, mask=m, scale_y=1.0 / f, scale_x=1.0 / f)
                r["block_mask"] = str(mp); note = "  block_mask (reaches the other block)"
            print(f"  {tag}: {int(x0)},{int(y0)}-{int(x1)},{int(y1)} -> {X0},{Y0}-{X1},{Y1}  "
                  f"+{gained.sum() * px_mm2:.1f} mm2 tissue{note}", flush=True)
            r.update(crop_x0=X0, crop_y0=Y0, crop_x1=X1, crop_y1=Y1, level0_W=X1 - X0, level0_H=Y1 - Y0)
        skipped = np.flatnonzero(ok & (owner == -1))
        if len(skipped):
            print(f"  {Path(svs).name}: {len(skipped)} pieces ({size[skipped - 1].sum() * px_mm2:.1f} mm2) "
                  f"farther than {a.near_mm} mm from every block, left out", flush=True)
    with open(a.out, "w", newline="") as fh:
        w = csv.DictWriter(fh, cols, delimiter="\t", lineterminator="\n")
        w.writeheader(); w.writerows(rows)
    print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
