#!/usr/bin/env python
from __future__ import annotations

import base64
import csv
import importlib.util as _iu
import json
import os
from pathlib import Path

import cv2
import numpy as np
import tifffile
from PIL import Image

ROOT = Path(os.environ.get("THREED_ROOT", os.path.join(os.environ.get("PROJECTS_ROOT", "/data/heid"), "3d")))
RUNS = ROOT / "registration/runs"


SEC = RUNS / "prep_sections/sections"
MANIFEST = RUNS / "prep_sections/manifest.tsv"
EDGE_RUNS = ("pair_fits",)
ANCHOR = "HT891Z1-U66"
OUT = ROOT / "reconstruction/volume_8um"

import argparse

SRC_MPP, WG_MPP = 1.0, 2.0
OUT_MPP = 8.0
GAP_UM = 25.0
BASES = {"G_withdrawn": ("R", "G withdrawn (R) - the design basis"),
         "G_not_withdrawn": ("M", "G NOT withdrawn (M = R o G) - the chain on disk")}


def _load(name, path):
    s = _iu.spec_from_file_location(name, path)
    m = _iu.module_from_spec(s)
    s.loader.exec_module(m)
    return m


CHAIN = _load("placement_chain", Path(__file__).resolve().parent / "placement_chain.py")


def source_plane(sid):
    img = tifffile.imread(str(SEC / sid / f"img_mpp{SRC_MPP:g}.tif"))
    if img.ndim == 3:
        img = img[..., 0]
    msk = cv2.imread(str(SEC / sid / f"mask_mpp{SRC_MPP:g}.png"),
                     cv2.IMREAD_GRAYSCALE) > 127
    f = SRC_MPP / OUT_MPP
    size = (max(1, int(round(img.shape[1] * f))), max(1, int(round(img.shape[0] * f))))
    img = cv2.resize(img, size, interpolation=cv2.INTER_AREA)
    msk = cv2.resize(msk.astype(np.uint8), size, interpolation=cv2.INTER_AREA) > 0
    v = img[msk]
    if v.size > 100:
        lo, hi = np.percentile(v, [1.0, 99.0])
        if hi > lo:
            img = np.clip((img.astype(np.float32) - lo) * (255.0 / (hi - lo)), 0, 255)
            img = (img * msk).astype(np.uint8)
    return img, msk


def b64len(n_bytes):
    return 4 * ((n_bytes + 2) // 3)


def _corrections_applied():
    p = os.environ.get("HTAN3D_SECTION_CORRECTIONS", "")
    if not p or not os.path.exists(p):
        return None
    d = json.load(open(p))
    return {"file": p, "chain_grid_um": float(d.get("chain_grid_um", 2.0)), "sections": d["sections"]}


def main():
    global OUT_MPP, OUT, SEC, MANIFEST, EDGE_RUNS, ANCHOR
    ap = argparse.ArgumentParser()
    ap.add_argument("--mpp", type=float, default=8.0,
                    help="in-plane um/px of the output volume (default 8)")
    ap.add_argument("--out", default="", help="output dir; default outputs/volume_<mpp>um")
    ap.add_argument("--skip-raw", action="store_true",
                    help="do not write the raw uint8 blobs (they lost the size "
                         "comparison by 5x; skip them once that is settled)")
    ap.add_argument("--anchor", default=ANCHOR,
                    help="section the chain is anchored to; sets the global frame "
                         "only, the relative geometry is the same for any choice")
    ap.add_argument("--prep-root", default=str(SEC))
    ap.add_argument("--manifest", default=str(MANIFEST))
    ap.add_argument("--edge-runs", default=",".join(EDGE_RUNS),
                    help="run directories under registration/runs holding the pairwise "
                         "results; a re-solved run must be listed BEFORE the run it "
                         "supersedes, because harvest keeps the first result per pair")
    ap.add_argument("--suffix", default="",
                    help="appended to the default output directory name, e.g. "
                         "--suffix _heswap -> outputs/volume_8um_heswap")
    ap.add_argument("--canvas-from", default="", dest="canvas_from",
                    help="a built volume (same mpp) whose canvas origin and size this one takes "
                         "instead of cropping to its own tissue hull: an arm the page draws on "
                         "the same quad as that volume must share its grid exactly")
    a = ap.parse_args()
    PIN = None
    if a.canvas_from:
        _pm = json.loads((Path(a.canvas_from) / "metadata.json").read_text())
        if abs(float(_pm["in_plane_um_per_px"]) - float(a.mpp)) > 1e-9 or not _pm.get("canvas_origin_um"):
            raise SystemExit(f"--canvas-from {a.canvas_from}: needs a volume at {a.mpp:g} um/px with canvas_origin_um")
        PIN = (np.asarray(_pm["canvas_origin_um"], float), int(_pm["canvas_px"]["width"]), int(_pm["canvas_px"]["height"]))
        print(f"canvas pinned to {a.canvas_from}: {PIN[1]} x {PIN[2]} px at origin ({PIN[0][0]:.3f}, {PIN[0][1]:.3f}) um")
    OUT_MPP = float(a.mpp)
    OUT = Path(a.out) if a.out else (
        ROOT / f"reconstruction/volume_{OUT_MPP:g}um{a.suffix}")
    SEC = Path(a.prep_root)
    MANIFEST = Path(a.manifest)
    EDGE_RUNS = tuple(s.strip() for s in a.edge_runs.split(",") if s.strip())
    ANCHOR = a.anchor
    print(f"prep={SEC}\nmanifest={MANIFEST}\nedges={EDGE_RUNS}", flush=True)
    a_skip_raw = bool(a.skip_raw)
    print(f"building at {OUT_MPP:g} um/px into {OUT}")
    OUT.mkdir(parents=True, exist_ok=True)
    man = {r["section_id"]: r for r in
           csv.DictReader(open(MANIFEST), delimiter="\t")}
    sids = sorted(man, key=lambda s: float(man[s]["z_position_um"]))
    z = np.array([float(man[s]["z_position_um"]) for s in sids])
    dz = np.diff(z)
    gaps = np.flatnonzero(dz >= GAP_UM)

    edges = CHAIN.harvest(EDGE_RUNS)
    tree = CHAIN.prereg_tree(edges, sids)
    chains = {k: CHAIN.chain_on(tree, edges, b, ANCHOR)
              for k, (b, _) in BASES.items()}
    print(f"{len(sids)} sections, z {z.min():.0f}-{z.max():.0f} um, "
          f"{len(gaps)} gaps >= {GAP_UM:g} um, tree {len(tree)} edges")

    src = {s: source_plane(s) for s in sids}
    print(f"sources read at {OUT_MPP:g} um/px; largest plane "
          f"{max(src[s][0].shape for s in sids)}")


    pts = []
    for ch in chains.values():
        for s in sids:
            h, w = src[s][0].shape
            c = np.array([[0, 0], [w, 0], [w, h], [0, h]], float) * (OUT_MPP / WG_MPP)
            pts.append((c @ ch[s][:2, :2].T + ch[s][:, 2]) * WG_MPP)
    pts = np.vstack(pts)
    o = pts.min(0) - 40.0
    W = int(np.ceil((pts[:, 0].max() + 40.0 - o[0]) / OUT_MPP))
    H = int(np.ceil((pts[:, 1].max() + 40.0 - o[1]) / OUT_MPP))
    if PIN is not None:
        o, W, H = PIN[0].copy(), PIN[1], PIN[2]

    vols = {}
    for bk, ch in chains.items():
        g = np.zeros((len(sids), H, W), np.uint8)
        a = np.zeros((len(sids), H, W), np.uint8)
        for i, s in enumerate(sids):
            M = np.zeros((2, 3)); M[:2, :2] = ch[s][:2, :2]
            M[:, 2] = (WG_MPP / OUT_MPP) * ch[s][:, 2] - o / OUT_MPP
            g[i] = cv2.warpAffine(src[s][0], M, (W, H), flags=cv2.INTER_LINEAR,
                                  borderValue=0)
            a[i] = cv2.warpAffine(src[s][1].astype(np.uint8) * 255, M, (W, H),
                                  flags=cv2.INTER_NEAREST, borderValue=0)
        vols[bk] = (g, a)

    anym = np.zeros((H, W), bool)
    for g, a in vols.values():
        anym |= (a > 0).any(0)
    ys, xs = np.nonzero(anym)
    p = int(round(80.0 / OUT_MPP))
    r0, r1 = max(0, ys.min() - p), min(H, ys.max() + p + 1)
    c0, c1 = max(0, xs.min() - p), min(W, xs.max() + p + 1)
    if PIN is not None:
        r0, r1, c0, c1 = 0, H, 0, W
    for bk in vols:
        g, a = vols[bk]
        vols[bk] = (np.ascontiguousarray(g[:, r0:r1, c0:c1]),
                    np.ascontiguousarray(a[:, r0:r1, c0:c1]))
    H, W = r1 - r0, c1 - c0


    CANVAS_ORIGIN_UM = [float(o[0] + c0 * OUT_MPP), float(o[1] + r0 * OUT_MPP)]
    print(f"shared canvas {W} x {H} px at {OUT_MPP:g} um/px "
          f"({W*OUT_MPP/1000:.2f} x {H*OUT_MPP/1000:.2f} mm), {len(sids)} planes")


    enc = {}
    cols = int(np.ceil(np.sqrt(len(sids))))
    rows = int(np.ceil(len(sids) / cols))
    for bk, (g, a) in vols.items():
        d = OUT / bk
        (d / "png").mkdir(parents=True, exist_ok=True)
        (d / "raw").mkdir(parents=True, exist_ok=True)

        png_bytes, planes = 0, []
        for i, s in enumerate(sids):


            fp = d / "png" / f"z{i:02d}_{s.split('-')[1]}.png"
            Image.fromarray(np.dstack([g[i], a[i]])).save(
                fp, format="PNG", optimize=True)
            png_bytes += fp.stat().st_size
            planes.append({"index": i, "section_id": s,
                           "z_um": float(z[i]),
                           "modality": man[s]["modality"],
                           "png": f"{bk}/png/{fp.name}",
                           "png_bytes": int(fp.stat().st_size)})

        atlas = np.zeros((rows * H, cols * W, 2), np.uint8)
        for i in range(len(sids)):
            rr, cc = divmod(i, cols)
            atlas[rr * H:(rr + 1) * H, cc * W:(cc + 1) * W, 0] = g[i]
            atlas[rr * H:(rr + 1) * H, cc * W:(cc + 1) * W, 1] = a[i]
        Image.fromarray(atlas).save(d / "atlas.png", format="PNG", optimize=True)
        atlas_bytes = (d / "atlas.png").stat().st_size

        raw_bytes = g.nbytes + a.nbytes
        if not a_skip_raw:
            (d / "raw" / "grey.u8").write_bytes(g.tobytes())
            (d / "raw" / "alpha.u8").write_bytes(a.tobytes())


        (d / "webp").mkdir(parents=True, exist_ok=True)
        wl_bytes, wq_bytes = 0, 0
        for i, s in enumerate(sids):
            la = np.dstack([g[i], a[i]])
            rgba = np.dstack([g[i], g[i], g[i], a[i]])
            f1 = d / "webp" / f"z{i:02d}_{s.split('-')[1]}.lossless.webp"
            Image.fromarray(rgba).save(f1, format="WEBP", lossless=True, quality=100,
                                       method=6)
            wl_bytes += f1.stat().st_size
            f2 = d / "webp" / f"z{i:02d}_{s.split('-')[1]}.q90.webp"
            Image.fromarray(rgba).save(f2, format="WEBP", quality=90, method=6)
            wq_bytes += f2.stat().st_size
            planes[i]["webp_lossless"] = f"{bk}/webp/{f1.name}"
            planes[i]["webp_lossless_bytes"] = int(f1.stat().st_size)
            planes[i]["webp_q90_lossy"] = f"{bk}/webp/{f2.name}"

        enc[bk] = {
            "webp_lossless": {"n_files": len(sids), "bytes": int(wl_bytes),
                              "base64_bytes": int(b64len(wl_bytes)),
                              "lossy": False,
                              "note": "RGBA (grey duplicated into R,G,B) because WebP "
                                      "has no grey+alpha mode; byte-exact"},
            "webp_q90_LOSSY": {"n_files": len(sids), "bytes": int(wq_bytes),
                               "base64_bytes": int(b64len(wq_bytes)),
                               "lossy": True,
                               "note": "quality 90. Pixel values are CHANGED. Size is "
                                       "reported for the decision; do not use it "
                                       "without saying on the page that it is lossy."},
            "per_plane_png": {"n_files": len(sids), "bytes": int(png_bytes),
                              "base64_bytes": int(b64len(png_bytes))},
            "atlas_png": {"grid_rows_cols": [rows, cols], "bytes": int(atlas_bytes),
                          "base64_bytes": int(b64len(atlas_bytes)),
                          "file": f"{bk}/atlas.png"},
            "raw_uint8": {"bytes": int(raw_bytes), "base64_bytes": int(b64len(raw_bytes)),
                          "grey": f"{bk}/raw/grey.u8", "alpha": f"{bk}/raw/alpha.u8",
                          "layout": "grey then alpha, each 50 x H x W uint8, "
                                    "C order, plane-major"},
            "planes": planes}
        LOSSLESS = ("per_plane_png", "atlas_png", "raw_uint8", "webp_lossless")
        best = min(LOSSLESS, key=lambda k: enc[bk][k]["base64_bytes"])
        enc[bk]["recommended"] = best
        enc[bk]["recommended_rule"] = ("smallest base64 among the LOSSLESS encodings; "
                                       "the lossy arm is listed but not recommended")
        print(f"\n  {bk}")
        for k in ("per_plane_png", "atlas_png", "raw_uint8", "webp_lossless",
                  "webp_q90_LOSSY"):
            e = enc[bk][k]
            print(f"    {k:<18} on disk {e['bytes']/1e6:8.2f} MB   "
                  f"base64 {e['base64_bytes']/1e6:8.2f} MB"
                  f"{'   <-- smallest lossless' if k == best else ''}"
                  f"{'   (LOSSY, not recommended)' if e.get('lossy') else ''}")


    meta = {
        "what": "display volume for an embeddable single-file page",
        "n_planes": len(sids),
        "interpolated": False,
        "interpolation_note": "zero interpolation along z: all 50 planes are "
                              "measured sections at their real z. In plane the data "
                              "were area-downsampled from the section-preparation 1 um/px level to "
                              f"{OUT_MPP:g} um/px.",
        "in_plane_um_per_px": OUT_MPP,
        "canvas_px": {"width": int(W), "height": int(H)},
        "canvas_origin_um": CANVAS_ORIGIN_UM,
        "canvas_mm": {"width": round(W * OUT_MPP / 1000, 4),
                      "height": round(H * OUT_MPP / 1000, 4)},
        "depth_mm": round(float(z.max() - z.min()) / 1000, 4),
        "true_aspect_ratio_xz": round(float(W * OUT_MPP / (z.max() - z.min())), 2),
        "aspect_note": f"in plane {W*OUT_MPP/1000:.2f} x {H*OUT_MPP/1000:.2f} mm "
                       f"against {float(z.max()-z.min())/1000:.3f} mm of depth, i.e. "
                       f"about 1:{W*OUT_MPP/(z.max()-z.min()):.0f}. Draw it to scale "
                       f"by default; if the depth axis is stretched, say so on the page.",
        "z_um": [float(v) for v in z],
        "z_spacing_um": {"values": [float(v) for v in dz],
                         "min": float(dz.min()), "max": float(dz.max()),
                         "mean": round(float(dz.mean()), 2),
                         "histogram": {str(int(k)): int((dz == k).sum())
                                       for k in sorted(set(dz.tolist()))}},
        "gaps_ge_25um": [{"after_index": int(i), "from_z_um": float(z[i]),
                          "to_z_um": float(z[i + 1]), "dz_um": float(dz[i])}
                         for i in gaps],
        "gaps_note": f"{len(gaps)} of {len(dz)} intervals are >= {GAP_UM:.0f} um "
                     f"(up to {dz.max():.0f} um). Mark them on the z axis; a reader "
                     f"who assumes even sampling will read the empty space as tissue.",
        "tree": "pre-registered minimum total |dz|, ties by section_id ascending",
        "tree_n_edges": len(tree),
        "anchor": ANCHOR,
        "bases": {k: {"code": b, "label": lab} for k, (b, lab) in BASES.items()},
        "bases_note": "the two bases are separate volumes on ONE shared canvas; do "
                      "not merge them. Per-section scale differs between them, which "
                      "is exactly what a depth view shows.",
        "alpha": "8-bit tissue mask; 0 outside the tissue so the page can render "
                 "without a black box. In the raw layout alpha is a second blob of "
                 "the same shape as grey.",
        "grey": "uint8, each section scaled by its own 1st/99th percentile inside "
                "its tissue mask (the same per-plane percentile rule used throughout)",
        "source": str(SEC),
        "manifest": str(MANIFEST),


        "edge_runs": list(EDGE_RUNS),


        "section_corrections_applied": _corrections_applied(),
        "encodings": enc,
        "read_only": True,
    }
    (OUT / "metadata.json").write_text(json.dumps(meta, indent=1))

    tot = {k: enc[k][enc[k]["recommended"]]["base64_bytes"] for k in enc}
    print(f"\nrecommended per basis: "
          + ", ".join(f"{k}={enc[k]['recommended']} "
                      f"({tot[k]/1e6:.1f} MB base64)" for k in enc))
    print(f"both bases together, base64: {sum(tot.values())/1e6:.1f} MB")
    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    main()
