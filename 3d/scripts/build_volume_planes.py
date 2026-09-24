#!/usr/bin/env python3

import argparse, csv, json, os, sys
from pathlib import Path
import numpy as np, cv2
from PIL import Image
import importlib.util as _iu

ROOT = Path(os.environ.get("THREED_ROOT", os.path.join(os.environ.get("PROJECTS_ROOT", "/data/heid"), "3d")))
_s = _iu.spec_from_file_location("bv", Path(__file__).resolve().parent / "build_volume.py"); bv = _iu.module_from_spec(_s); _s.loader.exec_module(bv)
CHAIN, BASES, GAP_UM, WG_MPP = bv.CHAIN, bv.BASES, bv.GAP_UM, bv.WG_MPP


def setup(a):
    bv.OUT_MPP = float(a.mpp); bv.SEC = Path(a.prep_root)
    man = {r["section_id"]: r for r in csv.DictReader(open(a.manifest), delimiter="\t")}
    sids = sorted(man, key=lambda s: float(man[s]["z_position_um"]))


    cman = {r["section_id"]: r for r in csv.DictReader(open(a.chain_manifest or a.manifest), delimiter="\t")}
    csids = sorted(cman, key=lambda s: float(cman[s]["z_position_um"]))
    missing = [s for s in sids if s not in cman]
    if missing: raise SystemExit(f"{len(missing)} sections of {a.manifest} are not in the chain manifest: {missing[:3]}")
    edge_runs = tuple(s.strip() for s in a.edge_runs.split(",") if s.strip())
    edges = CHAIN.harvest(edge_runs); tree = CHAIN.prereg_tree(edges, csids)
    chains = {k: CHAIN.chain_on(tree, edges, b, a.anchor) for k, (b, _) in BASES.items()}
    return man, sids, edge_runs, tree, chains


def plane_size(sid):
    m = cv2.imread(str(bv.SEC / sid / f"mask_mpp{bv.SRC_MPP:g}.png"), cv2.IMREAD_GRAYSCALE)
    f = bv.SRC_MPP / bv.OUT_MPP
    return (max(1, int(round(m.shape[1] * f))), max(1, int(round(m.shape[0] * f)))), m


def canvas(a, sids, chains):
    O = bv.OUT_MPP
    sizes, masks = {}, {}
    for s in sids:
        (w, h), m = plane_size(s)
        masks[s] = cv2.resize((m > 127).astype(np.uint8), (w, h), interpolation=cv2.INTER_AREA) > 0
        sizes[s] = (w, h)
    pts = []
    for ch in chains.values():
        for s in sids:
            w, h = sizes[s]
            c = np.array([[0, 0], [w, 0], [w, h], [0, h]], float) * (O / WG_MPP)
            pts.append((c @ ch[s][:2, :2].T + ch[s][:, 2]) * WG_MPP)
    pts = np.vstack(pts); o = pts.min(0) - 40.0
    W = int(np.ceil((pts[:, 0].max() + 40.0 - o[0]) / O)); H = int(np.ceil((pts[:, 1].max() + 40.0 - o[1]) / O))
    PIN = None
    if a.canvas_from:
        pm = json.loads((Path(a.canvas_from) / "metadata.json").read_text())
        if abs(float(pm["in_plane_um_per_px"]) - O) > 1e-9 or not pm.get("canvas_origin_um"):
            raise SystemExit(f"--canvas-from {a.canvas_from}: needs a volume at {O:g} um/px with canvas_origin_um")
        PIN = (np.asarray(pm["canvas_origin_um"], float), int(pm["canvas_px"]["width"]), int(pm["canvas_px"]["height"]))
        o, W, H = PIN[0].copy(), PIN[1], PIN[2]
    def M_of(ch_s):
        M = np.zeros((2, 3)); M[:2, :2] = ch_s[:2, :2]; M[:, 2] = (WG_MPP / O) * ch_s[:, 2] - o / O; return M
    anym = np.zeros((H, W), bool)
    for ch in chains.values():
        for s in sids:
            anym |= cv2.warpAffine(masks[s].astype(np.uint8) * 255, M_of(ch[s]), (W, H), flags=cv2.INTER_NEAREST, borderValue=0) > 0
    ys, xs = np.nonzero(anym); p = int(round(80.0 / O))
    r0, r1 = max(0, ys.min() - p), min(H, ys.max() + p + 1); c0, c1 = max(0, xs.min() - p), min(W, xs.max() + p + 1)
    if PIN is not None: r0, r1, c0, c1 = 0, H, 0, W
    return dict(o=o, W=W, H=H, r0=r0, r1=r1, c0=c0, c1=c1, M_of=M_of,
                origin=[float(o[0] + c0 * O), float(o[1] + r0 * O)], Wc=c1 - c0, Hc=r1 - r0)


def names(bk, i, s):
    u = s.split('-')[1]
    return {"png": f"{bk}/png/z{i:02d}_{u}.png", "webp_lossless": f"{bk}/webp/z{i:02d}_{u}.lossless.webp", "webp_q90_lossy": f"{bk}/webp/z{i:02d}_{u}.q90.webp"}


def do_plane(a, OUT, sids, chains, cv, i):
    s = sids[i]
    img, msk = bv.source_plane(s)
    for bk, ch in chains.items():
        M = cv["M_of"](ch[s])
        g = cv2.warpAffine(img, M, (cv["W"], cv["H"]), flags=cv2.INTER_LINEAR, borderValue=0)[cv["r0"]:cv["r1"], cv["c0"]:cv["c1"]]
        al = cv2.warpAffine(msk.astype(np.uint8) * 255, M, (cv["W"], cv["H"]), flags=cv2.INTER_NEAREST, borderValue=0)[cv["r0"]:cv["r1"], cv["c0"]:cv["c1"]]
        n = names(bk, i, s)
        for k in n: (OUT / n[k]).parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray(np.dstack([g, al])).save(OUT / n["png"], format="PNG", optimize=True)
        rgba = np.dstack([g, g, g, al])
        Image.fromarray(rgba).save(OUT / n["webp_lossless"], format="WEBP", lossless=True, quality=100, method=6)
        Image.fromarray(rgba).save(OUT / n["webp_q90_lossy"], format="WEBP", quality=90, method=6)
        print(f"  {s} {bk}: {g.shape[1]}x{g.shape[0]} px, tissue {int((al > 0).sum())} px", flush=True)


def finalize(a, OUT, man, sids, edge_runs, tree, chains, cv):
    O = bv.OUT_MPP
    z = np.array([float(man[s]["z_position_um"]) for s in sids]); dz = np.diff(z); gaps = np.flatnonzero(dz >= GAP_UM)
    enc = {}
    for bk in BASES:
        planes, wl, wq, pb = [], 0, 0, 0
        for i, s in enumerate(sids):
            n = names(bk, i, s)
            missing = [k for k in n if not (OUT / n[k]).exists()]
            if missing: raise SystemExit(f"{s} {bk}: missing {missing}; the plane array did not finish")
            with Image.open(OUT / n["png"]) as im: w, h = im.size
            if (w, h) != (cv["Wc"], cv["Hc"]): raise SystemExit(f"{s} {bk}: plane {w}x{h} is not the canvas {cv['Wc']}x{cv['Hc']}; rebuild the array")
            b1, b2, b3 = (OUT / n["webp_lossless"]).stat().st_size, (OUT / n["webp_q90_lossy"]).stat().st_size, (OUT / n["png"]).stat().st_size
            wl += b1; wq += b2; pb += b3
            planes.append({"index": i, "section_id": s, "z_um": float(z[i]), "modality": man[s]["modality"], "png": n["png"], "png_bytes": int(b3),
                           "webp_lossless": n["webp_lossless"], "webp_lossless_bytes": int(b1), "webp_q90_lossy": n["webp_q90_lossy"]})
        enc[bk] = {"webp_lossless": {"n_files": len(sids), "bytes": int(wl), "base64_bytes": int(bv.b64len(wl)), "lossy": False,
                                     "note": "RGBA (grey duplicated into R,G,B) because WebP has no grey+alpha mode; byte-exact"},
                   "webp_q90_LOSSY": {"n_files": len(sids), "bytes": int(wq), "base64_bytes": int(bv.b64len(wq)), "lossy": True,
                                      "note": "quality 90. Pixel values are CHANGED; the page says so where it uses it."},
                   "per_plane_png": {"n_files": len(sids), "bytes": int(pb), "base64_bytes": int(bv.b64len(pb))},
                   "recommended": "webp_lossless", "recommended_rule": "smallest lossless encoding", "planes": planes}
    W, H = cv["Wc"], cv["Hc"]
    meta = {"what": "display volume for an embeddable single-file page", "n_planes": len(sids), "interpolated": False,
            "interpolation_note": f"zero interpolation along z: every plane is a measured section at its real z. In plane the data were area-downsampled from the 1 um/px prep level to {O:g} um/px.",
            "in_plane_um_per_px": O, "canvas_px": {"width": int(W), "height": int(H)}, "canvas_origin_um": cv["origin"],
            "rasterization": {"warp_origin_um": [float(v) for v in cv["o"]],
                              "warp_canvas_px": {"width": int(cv["W"]), "height": int(cv["H"])},
                              "crop_xyxy_px": [int(cv[k]) for k in ("c0", "r0", "c1", "r1")],
                              "canvas_from": str(a.canvas_from or "")},
            "canvas_mm": {"width": round(W * O / 1000, 4), "height": round(H * O / 1000, 4)},
            "depth_mm": round(float(z.max() - z.min()) / 1000, 4), "true_aspect_ratio_xz": round(float(W * O / (z.max() - z.min())), 2),
            "z_um": [float(v) for v in z],
            "z_spacing_um": {"values": [float(v) for v in dz], "min": float(dz.min()), "max": float(dz.max()), "mean": round(float(dz.mean()), 2),
                             "histogram": {str(int(k)): int((dz == k).sum()) for k in sorted(set(dz.tolist()))}},
            "gaps_ge_25um": [{"after_index": int(i), "from_z_um": float(z[i]), "to_z_um": float(z[i + 1]), "dz_um": float(dz[i])} for i in gaps],
            "tree": "pre-registered minimum total |dz|, ties by section_id ascending", "tree_n_edges": len(tree), "anchor": a.anchor,
            "bases": {k: {"code": b, "label": lab} for k, (b, lab) in BASES.items()},
            "bases_note": "the two bases are separate volumes on ONE shared canvas; do not merge them.",
            "alpha": "8-bit tissue mask; 0 outside the tissue", "grey": "uint8, each section scaled by its own 1st/99th percentile inside its tissue mask",
            "source": str(bv.SEC), "manifest": str(a.manifest), "edge_runs": list(edge_runs),
            "section_corrections_applied": bv._corrections_applied(), "built_by": "build_volume_planes.py (one section per job)",
            "encodings": enc, "read_only": True}
    (OUT / "metadata.json").write_text(json.dumps(meta, indent=1))
    print(f"wrote {OUT / 'metadata.json'}: {len(sids)} planes, canvas {W} x {H} px at {O:g} um/px, origin ({cv['origin'][0]:.3f}, {cv['origin'][1]:.3f}) um", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mpp", type=float, default=8.0); ap.add_argument("--out", required=True)
    ap.add_argument("--anchor", required=True); ap.add_argument("--prep-root", required=True); ap.add_argument("--manifest", required=True)
    ap.add_argument("--edge-runs", required=True); ap.add_argument("--canvas-from", default="", dest="canvas_from")
    ap.add_argument("--chain-manifest", default="", dest="chain_manifest", help="section set the chain is solved over (a swap arm passes the main manifest)")
    ap.add_argument("--plane-index", type=int, default=-1, dest="plane_index"); ap.add_argument("--finalize", action="store_true")
    a = ap.parse_args()
    OUT = Path(a.out); OUT.mkdir(parents=True, exist_ok=True)
    man, sids, edge_runs, tree, chains = setup(a)
    cv = canvas(a, sids, chains)
    print(f"canvas {cv['Wc']} x {cv['Hc']} px at {bv.OUT_MPP:g} um/px, origin ({cv['origin'][0]:.3f}, {cv['origin'][1]:.3f}) um, {len(sids)} sections", flush=True)
    if a.plane_index >= 0:
        do_plane(a, OUT, sids, chains, cv, a.plane_index)
    if a.finalize:
        finalize(a, OUT, man, sids, edge_runs, tree, chains, cv)


if __name__ == "__main__":
    main()
