#!/usr/bin/env python

from __future__ import annotations
import os

import argparse
import json
import struct
from pathlib import Path

import numpy as np

ROOT = Path(os.environ.get("THREED_ROOT", os.path.join(os.environ.get("PROJECTS_ROOT", "/data/heid"), "3d")))
VOL = ROOT / "reconstruction/volume_8um"
_OBJ3D = __import__("os").environ.get("HTAN3D_OBJ3D", "")
OUT = (Path(_OBJ3D) / "S12_cloud") if _OBJ3D else ROOT / "objects_3d/outputs/S12_cloud"
BASIS = "G_withdrawn"
BUDGET = 200_000
SEED = 20260819


def read_points(p: Path):
    b = p.read_bytes()
    magic, n, cw, ch, sub, ncls = struct.unpack_from("<8sIHHHH", b, 0)
    if magic != b"CELLPT01":
        raise ValueError(f"{p}: bad magic {magic!r}")
    off = 32
    xy = np.frombuffer(b, np.uint16, n * 2, off).reshape(-1, 2).astype(np.float32) / sub
    off += n * 4
    off += n
    cls = np.frombuffer(b, np.uint8, n, off)
    return xy, cls


_exf = __import__('pathlib').Path(__import__('os').environ.get("HTAN3D_EXCLUDE",
            os.environ.get("THREED_ROOT", os.path.join(os.environ.get("PROJECTS_ROOT", "/data/heid"), "3d")) + "/front/S22-27909/configs/exclude_sections_3d.json"))
EXCLUDE_3D = set()
if _exf.exists():
    import json as _json
    EXCLUDE_3D = set(_json.loads(_exf.read_text())["sections"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--budget", type=int, default=BUDGET,
                    help="cells kept per class; a class at or under it is kept whole")
    ap.add_argument("--volume", default="", help="volume dir; default HT891Z1's")
    ap.add_argument("--sample", default="HT891Z1",
                    help="non-default samples write S12_cloud_<sample>")
    a = ap.parse_args()
    global VOL, OUT
    if a.volume:
        VOL = Path(a.volume)
    if a.sample != "HT891Z1":
        OUT = OUT if _OBJ3D else OUT.parent / f"S12_cloud_{a.sample}"

    meta = json.loads((VOL / "metadata.json").read_text())
    classes = json.loads((VOL / "cell_points_metadata.json").read_text())["classes"]
    planes = meta["encodings"][BASIS]["planes"]
    pts_dir = VOL / BASIS / "cell_points"


    per = {c: [] for c in range(len(classes))}
    zs, zi = [], {}
    planes = [p for p in planes if p["section_id"] not in EXCLUDE_3D]
    for p in planes:
        cand = sorted(pts_dir.glob(f"z{p['index']:02d}_*.cells.bin"))
        if not cand:
            continue
        k = zi.setdefault(p["index"], len(zs))
        if k == len(zs):
            zs.append(float(p["z_um"]))
        xy, cls = read_points(cand[0])
        for c in np.unique(cls):
            m = cls == c
            per[int(c)].append((k, xy[m]))

    rng = np.random.default_rng(SEED)
    blocks, report = [], []
    for c, name in enumerate(classes):
        chunks = per[c]
        total = sum(len(x) for _, x in chunks)
        if not total:
            blocks.append(np.zeros((0, 3), np.uint16))
            report.append({"cell_class": name, "n_cells": 0, "n_points": 0,
                           "kept_fraction": 0.0})
            continue
        keep = min(a.budget, total)

        take = np.sort(rng.choice(total, size=keep, replace=False)) if keep < total \
            else np.arange(total)
        out, base = [], 0
        for k, xy in chunks:
            sel = take[(take >= base) & (take < base + len(xy))] - base
            if len(sel):
                q = np.empty((len(sel), 3), np.uint16)
                q[:, 0] = np.clip(xy[sel, 0], 0, 65535).astype(np.uint16)
                q[:, 1] = np.clip(xy[sel, 1], 0, 65535).astype(np.uint16)
                q[:, 2] = k
                out.append(q)
            base += len(xy)
        q = np.vstack(out) if out else np.zeros((0, 3), np.uint16)
        blocks.append(q)
        report.append({"cell_class": name, "n_cells": int(total),
                       "n_points": int(len(q)),
                       "kept_fraction": round(len(q) / total, 4)})

    OUT.mkdir(parents=True, exist_ok=True)
    f = OUT / "cloud.bin"
    with open(f, "wb") as fh:
        fh.write(b"CLOUD01" + bytes([len(classes)]))
        for q in blocks:
            fh.write(struct.pack("<I", len(q)))
        for q in blocks:
            fh.write(q.tobytes())
        if (sum(len(q) for q in blocks) * 6) % 4:
            fh.write(b"\0\0")
        fh.write(np.asarray(zs, np.float32).tobytes())

    rep = {"basis": BASIS, "classes": classes, "budget_per_class": a.budget,
           "seed": SEED, "n_planes": len(zs), "z_um": zs,
           "canvas_px": meta["canvas_px"], "per_class": report,
           "bytes": f.stat().st_size,
           "claim": "MODEL PREDICTION on H&E (pan-cancer Cell head, single fold). "
                    "No spatial ground truth in this cohort; nothing here is scored. "
                    "Abundant classes are subsampled to the per-class budget, so "
                    "point density is NOT cell density -- read kept_fraction."}
    (OUT / "cloud.json").write_text(json.dumps(rep, indent=1))
    kept = sum(r["n_points"] for r in report)
    print(f"{kept:,} points from {sum(r['n_cells'] for r in report):,} cells, "
          f"{f.stat().st_size/1e6:.1f} MB")
    for r in report:
        if r["n_cells"]:
            print(f"  {r['cell_class']:<26} {r['n_points']:>7,} / {r['n_cells']:>9,}"
                  f"  {r['kept_fraction']:6.1%}")


if __name__ == "__main__":
    main()
