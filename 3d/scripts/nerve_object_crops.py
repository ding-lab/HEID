#!/usr/bin/env python3

from __future__ import annotations
import os

import argparse
import importlib.util
import json
import re
import sys
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

ROOT = Path(os.environ.get("THREED_ROOT", os.path.join(os.environ.get("PROJECTS_ROOT", "/data/heid"), "3d")))
LEVEL_UM = 1.0
PAD_UM = 350.0
MIN_HALF_UM = 450.0
UM_PX = 8.0
COL_THIS = (240, 176, 0)
COL_OTHER = (255, 255, 255)


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec)
    sys.modules[name] = m
    spec.loader.exec_module(m)
    return m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sample", required=True)
    ap.add_argument("--volume", required=True)
    ap.add_argument("--tiles", required=True, help="tile pyramid basis dir (…/G_withdrawn)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--basis", default="G_withdrawn")
    a = ap.parse_args()
    VOL, OUT = Path(a.volume), Path(a.out)
    TC = _load("tls_object_crops", Path(__file__).resolve().parent / "tls_object_crops.py")
    TC.VOL, TC.TILES = VOL, Path(a.tiles)

    meta = json.loads((VOL / "nerve_regions_metadata.json").read_text())
    planes = [p for p in meta["planes"] if p.get("regions")]
    by_cord: dict[str, list] = {}
    for p in planes:
        for g in p["regions"]:
            by_cord.setdefault(g["id"], []).append((p, g))
    if not by_cord:
        raise SystemExit("no region ids in nerve_regions_metadata.json (rebuild the nerve layer)")

    n_img = 0
    for cid in sorted(by_cord):
        mem = by_cord[cid]
        bx = np.array([g["bbox_um"] for _, g in mem], float)
        cx, cy = bx[:, [0, 2]].mean(), bx[:, [1, 3]].mean()
        half = max(MIN_HALF_UM, 0.5 * max(bx[:, 2].max() - bx[:, 0].min(),
                                          bx[:, 3].max() - bx[:, 1].min()) + PAD_UM)
        wx0, wx1, wy0, wy1 = cx - half, cx + half, cy - half, cy + half
        odir = OUT / cid / "he"
        odir.mkdir(parents=True, exist_ok=True)
        done_planes = set()
        for p, g in mem:
            if p["index"] in done_planes:
                continue
            done_planes.add(p["index"])
            sid = p["section_id"]
            u = int(re.search(r"U(\d+)$", sid).group(1))
            try:
                img, (px0, py0) = TC.read_tiles(sid, LEVEL_UM, wx0, wy0, wx1, wy1)
            except (KeyError, FileNotFoundError) as e:
                print(f"  skip {sid}: {e}", flush=True); continue
            h, w = img.shape[:2]

            ring = np.asarray(Image.open(VOL / a.basis / "nerve" / p["file"]).convert("RGBA"))[:, :, 3]
            rx0, ry0 = int(np.floor(wx0 / UM_PX)), int(np.floor(wy0 / UM_PX))
            rx1, ry1 = int(np.ceil(wx1 / UM_PX)), int(np.ceil(wy1 / UM_PX))
            sub = np.zeros((ry1 - ry0, rx1 - rx0), np.uint8)
            sy0, sx0 = max(0, ry0), max(0, rx0)
            sy1, sx1 = min(ring.shape[0], ry1), min(ring.shape[1], rx1)
            if sy1 > sy0 and sx1 > sx0:
                sub[sy0 - ry0:sy1 - ry0, sx0 - rx0:sx1 - rx0] = ring[sy0:sy1, sx0:sx1]
            sub = cv2.resize(sub, (w, h), interpolation=cv2.INTER_NEAREST)
            rim = sub >= 200

            own = np.zeros((h, w), bool)
            for p2, g2 in mem:
                if p2["index"] != p["index"]:
                    continue
                x0, y0, x1, y1 = g2["bbox_um"]
                X0, Y0 = int((x0 - wx0) / LEVEL_UM), int((y0 - wy0) / LEVEL_UM)
                X1, Y1 = int((x1 - wx0) / LEVEL_UM), int((y1 - wy0) / LEVEL_UM)
                X0, Y0, X1, Y1 = max(0, X0), max(0, Y0), min(w, X1), min(h, Y1)
                if X1 > X0 and Y1 > Y0:
                    own[Y0:Y1, X0:X1] = True
            img[rim & ~own] = COL_OTHER
            bold = cv2.dilate((rim & own).astype(np.uint8), np.ones((3, 3), np.uint8)) > 0
            img[bold] = COL_THIS
            sb = int(200 / LEVEL_UM)
            cv2.rectangle(img, (20, h - 30), (20 + sb, h - 22), (0, 0, 0), -1)
            cv2.putText(img, "200 um", (20, h - 36), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2, cv2.LINE_AA)
            lab = f"{cid}  U{u}  HE  z={p['z_um']:.0f} um"
            cv2.putText(img, lab, (20, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 4, cv2.LINE_AA)
            cv2.putText(img, lab, (20, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 0), 2, cv2.LINE_AA)
            cv2.imwrite(str(odir / f"U{u:03d}.png"), img)
            n_img += 1
        print(f"  {cid}: {len(done_planes)} sections -> {odir.parent}", flush=True)
    print(f"wrote {n_img} images for {len(by_cord)} cords under {OUT}")


if __name__ == "__main__":
    main()
