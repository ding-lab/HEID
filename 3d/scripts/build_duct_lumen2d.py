#!/usr/bin/env python3

import argparse
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent.parent
FINE_UM = 2.0


def tls_mask(vol, rec, shape):
    import numpy as _np
    from scipy.ndimage import binary_dilation as _dil
    mp = Path(vol) / "tls_regions_metadata.json"
    if not mp.exists():
        return False
    m = json.loads(mp.read_text())
    q = [x for x in m["planes"] if x["section_id"] == rec["section_id"]]
    if not q:
        return False
    from PIL import Image as _Im
    reg = None
    for key in ("file3d", "file2d"):
        f = Path(vol) / "G_withdrawn" / "tls" / q[0][key]
        if not f.exists():
            continue
        arr = _np.array(_Im.open(f))
        one = (arr[..., -1] > 0) if arr.ndim == 3 else (arr > 0)
        reg = one if reg is None else (reg | one)
    if reg is None or not reg.any():
        return False
    ch, cw = reg.shape
    H, W = shape
    iy = _np.clip((_np.arange(H) * DA.RES_UM / 8.0).astype(int), 0, ch - 1)
    ix = _np.clip((_np.arange(W) * DA.RES_UM / 8.0).astype(int), 0, cw - 1)
    grown = _dil(reg[_np.ix_(iy, ix)],
                 iterations=max(1, int(round(LOCAL_NEAR_TLS_UM / DA.RES_UM))))
    print(f"  local seed inside the TLS + {LOCAL_NEAR_TLS_UM:g} um: "
          f"{grown.sum() * DA.RES_UM ** 2 / 1e6:.2f} mm2", flush=True)
    return grown


def _load(name, p):
    spec = importlib.util.spec_from_file_location(name, p)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


DA = _load("duct_anchor", Path(__file__).resolve().parent / "duct_anchor_align.py")


LOCAL_NEAR_TLS_UM = 100.0


def colour_root(vol, sid):
    for arm in ("volume_tiles_heswap", "volume_tiles"):
        root = Path(str(vol).replace("volume_8um", arm))
        ix = root / "G_withdrawn" / sid / "index.json"
        if ix.exists():
            lv = json.loads(ix.read_text()).get("rgb_levels", [])
            if any(abs(l["mpp"] - DA.RES_UM) < 1e-6 for l in lv):
                return root, arm
    return None, None


def one_section(vol, rec, out):
    sid = rec["section_id"]
    u = sid.split("-")[-1]
    root, arm = colour_root(vol, sid)
    groot, _ = DA.tiles_root(vol, sid, None)
    if groot is None:
        print(f"  {sid}: no tile pyramid, skipped", flush=True)
        return None
    g, a = DA.load_tiles(groot, sid, DA.RES_UM)
    cm = DA.cell_mask(vol, rec, g.shape, DA.RES_UM)
    rgb = None
    if root is not None:
        rgb, _ra = DA.load_rgb_tiles(root, sid, DA.RES_UM)
    wm = None if rgb is not None else DA.white_mask(vol, rec, g.shape)
    white, feats = DA.ducts(g, a, cm, wm, rgb, local_seed=tls_mask(vol, rec, g.shape))
    src = f"{arm} colour {DA.RES_UM:g} um" if rgb is not None else "grey + 4 um colour test"

    lab = np.zeros(g.shape, np.int32)
    for f in feats:
        sl = f["sl"]
        sub = lab[sl]
        sub[f["m"] & (sub == 0)] = f["id"]
    k = int(round(FINE_UM / DA.RES_UM))
    H, W = lab.shape
    Hk, Wk = H // k * k, W // k * k
    lab_f = lab[:Hk, :Wk].reshape(Hk // k, k, Wk // k, k).max(axis=(1, 3))
    fn = f"z{rec['index']:02d}_{u}"
    np.savez_compressed(out / f"{fn}.lumens.npz", lab=lab_f, um_per_px=FINE_UM,
                        section_id=sid, z_um=float(rec["z_um"]), source=src)
    with open(out / f"{fn}.features.tsv", "w") as fh:
        fh.write("id\tcx_um\tcy_um\tarea_um2\telong\tangle_deg\tcirc\n")
        for f in feats:
            fh.write(f"{f['id']}\t{f['cx_um']:.1f}\t{f['cy_um']:.1f}\t{f['area_um2']:.0f}\t"
                     f"{f['elong']:.2f}\t{f['angle_deg']:.1f}\t{f['circ']:.3f}\n")
    print(f"  {sid}: {len(feats)} lumens from {src} "
          f"(median {np.median([f['area_um2'] for f in feats]):.0f} um2)" if feats
          else f"  {sid}: no lumens ({src})", flush=True)
    return len(feats)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--volume", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--plane-index", type=int, default=-1)
    ap.add_argument("--sections", default="", help="comma separated U numbers; default all")
    ap.add_argument("--min-area-um2", type=float, default=None, dest="min_area_um2",
                    help="smallest lumen kept (default: the anchor detector's own gate). "
                         "Set only here: the anchor path picks landmarks and keeps its own gate")
    ap.add_argument("--open-um", type=float, default=None, dest="open_um",
                    help="speckle thinner than this is not a lumen (default: the detector's own)")
    a = ap.parse_args()
    if a.min_area_um2 is not None:
        print(f"  min lumen area {DA.MIN_AREA_UM2:g} -> {a.min_area_um2:g} um2", flush=True)
        DA.MIN_AREA_UM2 = a.min_area_um2
    if a.open_um is not None:
        print(f"  opening {DA.OPEN_UM:g} -> {a.open_um:g} um", flush=True)
        DA.OPEN_UM = a.open_um
    vol, out = Path(a.volume), Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    vm = json.loads((vol / "metadata.json").read_text())
    planes = sorted(vm["encodings"]["G_withdrawn"]["planes"], key=lambda p: p["z_um"])
    if a.plane_index >= 0:
        planes = [planes[a.plane_index]]
    elif a.sections:
        want = {s.strip().upper() for s in a.sections.split(",")}
        planes = [p for p in planes if p["section_id"].split("-")[-1] in want]
    n = 0
    for p in planes:
        r = one_section(vol, p, out)
        n += 1 if r is not None else 0
    print(f"wrote {out}: {n} sections", flush=True)


if __name__ == "__main__":
    main()
