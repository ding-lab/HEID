#!/usr/bin/env python3
import argparse
import importlib.util
import json
import re
import sys
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[3]
UM_PX = 8.0
LEVEL_UM = 1.0
PAD_UM = 120.0
MIN_HALF_UM = 250.0
COL_THIS = (240, 176, 0)
COL_OTHER = (255, 255, 255)


def display_sid(sid):
    return re.sub(r"-P\d+_", "_", str(sid))


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
    ap.add_argument("--tiles", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--basis", default="G_withdrawn")
    ap.add_argument("--per-slide", default="", dest="per_slide",
                    help="also write one ROI per region per section under <per-slide>/<section>/")
    ap.add_argument("--only", default="", help="one nerve id (nerve-83): only its ROI, nothing else touched")
    ap.add_argument("--clean", action="store_true", help="the H&E alone: no region outlines")
    ap.add_argument("--suffix", default="", help="file name suffix, e.g. _he -> nerve-83_he.png")
    ap.add_argument("--half-um", type=float, default=0.0, dest="half_um",
                    help="fixed half-size of every window (um), so a nerve's ROIs share one size across sections")
    a = ap.parse_args()
    VOL, OUT = Path(a.volume), Path(a.out)
    PS = Path(a.per_slide) if a.per_slide else None
    if PS is not None:
        PS.mkdir(parents=True, exist_ok=True)
        if not a.only:
            for old in PS.glob("*/nerve-*.png"):
                old.unlink()
    TC = _load("tls_object_crops", Path(__file__).resolve().parent / "tls_object_crops.py")
    TC.VOL, TC.TILES = VOL, Path(a.tiles)
    OUT.mkdir(parents=True, exist_ok=True)
    if not a.only:
        for old in OUT.glob("nerve-*.png"):
            old.unlink()

    meta = json.loads((VOL / "nerve_regions_metadata.json").read_text())
    by_nerve = {}
    for p in meta["planes"]:
        for g in p.get("regions") or []:
            by_nerve.setdefault(g["id"], []).append((p, g))
    def roi(nid, p, g, out_path):
        x0, y0, x1, y1 = g["bbox_um"]
        cx, cy = 0.5 * (x0 + x1), 0.5 * (y0 + y1)
        half = a.half_um or max(MIN_HALF_UM, 0.5 * max(x1 - x0, y1 - y0) + PAD_UM)
        wx0, wx1, wy0, wy1 = cx - half, cx + half, cy - half, cy + half
        sid = p["section_id"]
        try:
            img, (px0, py0) = TC.read_tiles(sid, LEVEL_UM, wx0, wy0, wx1, wy1)
        except (KeyError, FileNotFoundError) as e:
            print(f"  {nid} {sid}: {e}", flush=True); return False
        h, w = img.shape[:2]
        if g.get('polygons_um'):
            if not a.clean:
                for region in p['regions']:
                    own=region['id']==nid
                    paths=list(region.get('polygons_um',[]))
                    paths += [ring for group in region.get('polygon_holes_um',[]) for ring in group]
                    for path in paths:
                        q=np.rint((np.asarray(path)-[px0*LEVEL_UM,py0*LEVEL_UM])/LEVEL_UM).astype('i4')
                        cv2.polylines(img,[q],True,COL_THIS if own else COL_OTHER,3 if own else 1,cv2.LINE_AA)


            cv2.line(img,(20,h-24),(120,h-24),(0,0,0),3)
            cv2.putText(img,'100 um',(20,h-34),cv2.FONT_HERSHEY_SIMPLEX,.6,(0,0,0),2,cv2.LINE_AA)
            lab=f"{nid}  {display_sid(sid).split('_')[-1]}  HE  z={p['z_um']:.0f} um"
            cv2.putText(img,lab,(20,32),cv2.FONT_HERSHEY_SIMPLEX,.7,(255,255,255),4,cv2.LINE_AA)
            cv2.putText(img,lab,(20,32),cv2.FONT_HERSHEY_SIMPLEX,.7,(0,0,0),2,cv2.LINE_AA)
            out_path.parent.mkdir(parents=True,exist_ok=True);cv2.imwrite(str(out_path),img)
            return True
        ring = np.asarray(Image.open(VOL / a.basis / "nerve" / p["file"]).convert("RGBA"))[:, :, 3]
        rx0, ry0 = int(np.floor(wx0 / UM_PX)), int(np.floor(wy0 / UM_PX))
        rx1, ry1 = int(np.ceil(wx1 / UM_PX)), int(np.ceil(wy1 / UM_PX))
        sub = np.zeros((ry1 - ry0, rx1 - rx0), np.uint8)
        sy0, sx0 = max(0, ry0), max(0, rx0)
        sy1, sx1 = min(ring.shape[0], ry1), min(ring.shape[1], rx1)
        if sy1 > sy0 and sx1 > sx0:
            sub[sy0 - ry0:sy1 - ry0, sx0 - rx0:sx1 - rx0] = ring[sy0:sy1, sx0:sx1]


        fill8 = (sub > 0).astype(np.float32)
        fill1 = cv2.resize(fill8, (w, h), interpolation=cv2.INTER_LINEAR) > 0.5
        fill1 = cv2.GaussianBlur(fill1.astype(np.float32), (0, 0), 3) > 0.5
        X0, Y0 = max(0, int((x0 - wx0) / LEVEL_UM)), max(0, int((y0 - wy0) / LEVEL_UM))
        X1, Y1 = min(w, int((x1 - wx0) / LEVEL_UM)), min(h, int((y1 - wy0) / LEVEL_UM))
        cnts, _ = cv2.findContours(fill1.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for c in ([] if a.clean else cnts):
            cx_, cy_ = c[:, 0, 0].mean(), c[:, 0, 1].mean()
            own = X0 <= cx_ <= X1 and Y0 <= cy_ <= Y1
            cv2.polylines(img, [c], True, COL_THIS if own else COL_OTHER, 4 if own else 2, cv2.LINE_AA)
        sb = int(100 / LEVEL_UM)
        cv2.rectangle(img, (20, h - 30), (20 + sb, h - 22), (0, 0, 0), -1)
        cv2.putText(img, "100 um", (20, h - 36), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2, cv2.LINE_AA)
        u = display_sid(sid).split("_")[-1]
        lab = f"{nid}  {u}  HE  z={p['z_um']:.0f} um  ({len(by_nerve[nid])} sections)"
        cv2.putText(img, lab, (20, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 4, cv2.LINE_AA)
        cv2.putText(img, lab, (20, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 0), 2, cv2.LINE_AA)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(out_path), img)
        return True

    n_img = 0
    nids = [a.only] if a.only else sorted(by_nerve, key=lambda s: int(s.split("-")[1]))
    for nid in nids:

        p, g = max(by_nerve[nid], key=lambda pg: (pg[1]["bbox_um"][2] - pg[1]["bbox_um"][0]) * (pg[1]["bbox_um"][3] - pg[1]["bbox_um"][1]))
        n_img += int(roi(nid, p, g, OUT / f"{nid}{a.suffix}.png"))
    print(f"wrote {n_img} nerve ROIs under {OUT}", flush=True)
    if PS is not None:
        n_ps = 0
        for nid in nids:
            for p, g in by_nerve[nid]:
                n_ps += int(roi(nid, p, g, PS / display_sid(p["section_id"]) / f"{nid}{a.suffix}.png"))
        print(f"wrote {n_ps} per-section nerve ROIs under {PS}/<section>/", flush=True)


if __name__ == "__main__":
    main()
