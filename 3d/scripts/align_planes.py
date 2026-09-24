#!/usr/bin/env python3

import argparse, csv, json, os, sys
from pathlib import Path
import numpy as np, cv2
from PIL import Image
import importlib.util as _iu

ROOT = Path(os.environ.get("THREED_ROOT", os.path.join(os.environ.get("PROJECTS_ROOT", "/data/heid"), "3d")))
WG_MPP, BASE_MPP = 2.0, 8.0
BASES = {"G_withdrawn": "R", "G_not_withdrawn": "M"}


def _load(name, path):
    s = _iu.spec_from_file_location(name, path); m = _iu.module_from_spec(s); s.loader.exec_module(m); return m


CHAIN = _load("placement_chain", Path(__file__).resolve().parent / "placement_chain.py")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prep-root", required=True); ap.add_argument("--manifest", required=True)
    ap.add_argument("--edge-runs", required=True); ap.add_argument("--anchor", required=True)
    ap.add_argument("--out", required=True, help="<align dir>/volume_8um")
    a = ap.parse_args()
    SEC, OUT = Path(a.prep_root), Path(a.out)
    man = {r["section_id"]: r for r in csv.DictReader(open(a.manifest), delimiter="\t")}
    sids = sorted(man, key=lambda s: float(man[s]["z_position_um"]))
    z = np.array([float(man[s]["z_position_um"]) for s in sids])
    edge_runs = tuple(s.strip() for s in a.edge_runs.split(",") if s.strip())
    edges = CHAIN.harvest(edge_runs); tree = CHAIN.prereg_tree(edges, sids)
    ch = {k: CHAIN.chain_on(tree, edges, b, a.anchor) for k, b in BASES.items()}


    shp, msk8 = {}, {}
    for s in sids:
        m = cv2.imread(str(SEC / s / "mask_mpp1.png"), cv2.IMREAD_GRAYSCALE) > 127
        h1, w1 = m.shape; f = 1.0 / BASE_MPP
        h, w = max(1, int(round(h1 * f))), max(1, int(round(w1 * f)))
        shp[s] = (h, w); msk8[s] = cv2.resize(m.astype(np.uint8), (w, h), interpolation=cv2.INTER_AREA) > 0
    pts = []
    for c_ in ch.values():
        for s in sids:
            h, w = shp[s]
            c = np.array([[0, 0], [w, 0], [w, h], [0, h]], float) * (BASE_MPP / WG_MPP)
            pts.append((c @ c_[s][:2, :2].T + c_[s][:, 2]) * WG_MPP)
    pts = np.vstack(pts); o = pts.min(0) - 40.0
    W = int(np.ceil((pts[:, 0].max() + 40.0 - o[0]) / BASE_MPP)); H = int(np.ceil((pts[:, 1].max() + 40.0 - o[1]) / BASE_MPP))
    alpha = {}
    anym = np.zeros((H, W), bool)
    for bk, c_ in ch.items():
        for s in sids:
            M = np.zeros((2, 3)); M[:2, :2] = c_[s][:2, :2]; M[:, 2] = (WG_MPP / BASE_MPP) * c_[s][:, 2] - o / BASE_MPP
            al = cv2.warpAffine(msk8[s].astype(np.uint8) * 255, M, (W, H), flags=cv2.INTER_NEAREST, borderValue=0)
            alpha[(bk, s)] = al; anym |= al > 0
    ys, xs = np.nonzero(anym); p = int(round(80.0 / BASE_MPP))
    r0, r1 = max(0, ys.min() - p), min(H, ys.max() + p + 1); c0, c1 = max(0, xs.min() - p), min(W, xs.max() + p + 1)
    W8, H8 = int(c1 - c0), int(r1 - r0)
    origin = [float(o[0] + c0 * BASE_MPP), float(o[1] + r0 * BASE_MPP)]
    print(f"canvas {W8} x {H8} px at 8 um, origin ({origin[0]:.3f}, {origin[1]:.3f}) um, {len(sids)} sections", flush=True)

    enc = {}
    for bk in BASES:
        d = OUT / bk / "webp"; d.mkdir(parents=True, exist_ok=True)
        planes = []
        for i, s in enumerate(sids):
            al = alpha[(bk, s)][r0:r1, c0:c1]
            fp = d / f"z{i:02d}_{s.split('-')[-1]}.webp"
            Image.fromarray(np.dstack([np.zeros_like(al), al])).convert("RGBA").save(fp, format="WEBP", lossless=True)
            planes.append({"index": i, "section_id": s, "z_um": float(z[i]), "modality": man[s]["modality"],
                           "webp_lossless": f"{bk}/webp/{fp.name}"})
        enc[bk] = {"planes": planes, "note": "alpha = the section's tissue mask on the canvas; grey is empty (alignment input only)"}
    meta = {"what": "3-D alignment input: canvas + tissue masks from the coarse placement (no image planes)",
            "n_planes": len(sids), "in_plane_um_per_px": BASE_MPP,
            "canvas_px": {"width": W8, "height": H8}, "canvas_origin_um": origin,
            "z_um": [float(v) for v in z], "anchor": a.anchor, "tree_n_edges": len(tree),
            "bases": {k: {"code": b} for k, b in BASES.items()}, "source": str(SEC), "manifest": a.manifest,
            "edge_runs": list(edge_runs), "encodings": enc,


            "section_corrections_applied": _corrections_applied()}
    (OUT / "metadata.json").write_text(json.dumps(meta, indent=1))


def _corrections_applied():
    p = os.environ.get("HTAN3D_SECTION_CORRECTIONS", "")
    if not p or not os.path.exists(p):
        return None
    d = json.load(open(p))
    return {"file": p, "chain_grid_um": float(d.get("chain_grid_um", 2.0)), "sections": d["sections"]}


if __name__ == "__main__":
    main()
