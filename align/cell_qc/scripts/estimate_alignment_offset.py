#!/usr/bin/env python
import argparse
import json
import os
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import tifffile
from scipy.spatial import cKDTree
from skimage.color import rgb2hed

UM_PER_PX = 0.2125
TILE = 1024
SEARCH = 96
BLUR_SIGMA = 1.5
MIN_CELLS_TILE = 20
MIN_HE_TISSUE = 0.15


WHITE_INTENSITY = 235
MAX_TILE_WHITE_FRAC = 0.40
MIN_MASK_FRAC = 0.02
MIN_PROM = 0.02
APPLICABLE_FRAC = 0.80

PROJ = Path(os.environ.get("PROJECTS_ROOT", "/data/heid"))
ALIGNED_ROOT = PROJ / "align/registered_he"
XENIUM_ROOT = PROJ / "data/X1000/Xenium"

_G = {}


def log(m):
    print(m, flush=True)


def _zarr_l0(path):
    import zarr

    g = zarr.open(tifffile.imread(str(path), aszarr=True), mode="r")
    if hasattr(g, "shape"):
        return g
    for k in ("0", 0):
        try:
            return g[k]
        except (TypeError, KeyError):
            continue
    raise RuntimeError(f"cannot resolve level 0 of {path}")


def haematoxylin(rgb):
    f = rgb.astype(np.float32) / 255.0
    np.clip(f, 1e-6, 1.0, out=f)
    return rgb2hed(f)[..., 0].astype(np.float32)


def blur(a, sigma=BLUR_SIGMA):
    k = int(max(3, round(sigma * 6) | 1))
    return cv2.GaussianBlur(a.astype(np.float32), (k, k), sigma)


def white_fraction(rgb):
    return float((rgb.astype(np.float32).mean(axis=2) > WHITE_INTENSITY).mean())


def tissue_fraction(rgb):
    return float((rgb.astype(np.float32).mean(axis=2) < 225).mean())


def read_he(x0, y0, w, h):
    a = _G["he"]
    H, W = a.shape[0], a.shape[1]
    sy0, sy1 = max(0, y0), min(H, y0 + h)
    sx0, sx1 = max(0, x0), min(W, x0 + w)
    out = np.full((h, w, 3), 255, np.uint8)
    if sy0 < sy1 and sx0 < sx1:
        out[sy0 - y0:sy1 - y0, sx0 - x0:sx1 - x0] = np.asarray(a[sy0:sy1, sx0:sx1])
    return out


def read_dapi(x0, y0, w, h):
    a = _G.get("dapi")
    if a is None:
        return None
    nd = len(a.shape)
    H, W = a.shape[-2], a.shape[-1]
    sy0, sy1 = max(0, y0), min(H, y0 + h)
    sx0, sx1 = max(0, x0), min(W, x0 + w)
    out = np.zeros((h, w), np.float32)
    if sy0 >= sy1 or sx0 >= sx1:
        return out
    if nd == 2:
        sub = np.asarray(a[sy0:sy1, sx0:sx1])
    elif nd == 3:
        sub = np.asarray(a[:, sy0:sy1, sx0:sx1]).max(axis=0)
    else:
        sub = np.asarray(a[0, :, sy0:sy1, sx0:sx1]).max(axis=0)
    out[sy0 - y0:sy1 - y0, sx0 - x0:sx1 - x0] = sub
    return out


def load_nuclei(xen_dir):
    nb = pd.read_parquet(Path(xen_dir) / "nucleus_boundaries.parquet",
                         columns=["cell_id", "vertex_x", "vertex_y"])
    nb = nb.sort_values("cell_id", kind="stable")
    cid = nb["cell_id"].to_numpy()
    vx = (nb["vertex_x"].to_numpy(np.float64) / UM_PER_PX).astype(np.float32)
    vy = (nb["vertex_y"].to_numpy(np.float64) / UM_PER_PX).astype(np.float32)
    change = np.empty(len(cid), bool)
    change[0] = True
    change[1:] = cid[1:] != cid[:-1]
    starts = np.flatnonzero(change)
    ends = np.append(starts[1:], len(cid))
    cx = np.add.reduceat(vx, starts) / (ends - starts)
    cy = np.add.reduceat(vy, starts) / (ends - starts)
    return {"verts_x": vx, "verts_y": vy, "starts": starts, "ends": ends,
            "tree": cKDTree(np.column_stack([cx, cy])), "n_cells": len(starts)}


def rasterise(x0, y0, w, h):
    nuc = _G["nuc"]
    m = np.zeros((h, w), np.uint8)
    r = 0.5 * np.hypot(w, h) + 40
    idx = nuc["tree"].query_ball_point([x0 + w / 2.0, y0 + h / 2.0], r)
    if not idx:
        return m, 0
    polys, n = [], 0
    for i in idx:
        a, b = nuc["starts"][i], nuc["ends"][i]
        px = np.round(nuc["verts_x"][a:b] - x0).astype(np.int32)
        py = np.round(nuc["verts_y"][a:b] - y0).astype(np.int32)
        if px.max() < 0 or py.max() < 0 or px.min() >= w or py.min() >= h:
            continue
        polys.append(np.column_stack([px, py]))
        n += 1
    if polys:
        cv2.fillPoly(m, polys, 1)
    return m, n


def match_shift(image, template, search=SEARCH):
    r = cv2.matchTemplate(image.astype(np.float32), template.astype(np.float32),
                          cv2.TM_CCOEFF_NORMED)
    if not np.isfinite(r).all():
        r = np.nan_to_num(r, nan=-1.0, posinf=-1.0, neginf=-1.0)
    j, i = np.unravel_index(int(np.argmax(r)), r.shape)
    peak = float(r[j, i])

    def refine(v0, vm, vp):
        den = vm - 2 * v0 + vp
        return 0.0 if abs(den) < 1e-12 else float(np.clip(0.5 * (vm - vp) / den, -1, 1))

    dx = refine(r[j, i], r[j, i - 1], r[j, i + 1]) if 0 < i < r.shape[1] - 1 else 0.0
    dy = refine(r[j, i], r[j - 1, i], r[j + 1, i]) if 0 < j < r.shape[0] - 1 else 0.0
    mask = np.ones_like(r, bool)
    mask[max(0, j - 10):j + 11, max(0, i - 10):i + 11] = False
    second = float(r[mask].max()) if mask.any() else -1.0
    return {"sx_px": (i - search) + dx, "sy_px": (j - search) + dy,
            "peak": peak, "second": second, "prominence": peak - second,
            "on_border": bool(i <= 1 or j <= 1 or i >= r.shape[1] - 2
                              or j >= r.shape[0] - 2)}


def _init(he_path, dapi_path, xen_dir):
    _G["he"] = _zarr_l0(he_path)
    _G["dapi"] = _zarr_l0(dapi_path) if dapi_path else None
    _G["nuc"] = load_nuclei(xen_dir)


def measure(args):
    x0, y0, do_dapi = args
    ex = TILE + 2 * SEARCH
    rgb = read_he(x0 - SEARCH, y0 - SEARCH, ex, ex)
    core = rgb[SEARCH:SEARCH + TILE, SEARCH:SEARCH + TILE]
    tf = tissue_fraction(core)
    wf = white_fraction(core)
    mask, ncell = rasterise(x0, y0, TILE, TILE)
    rec = {"x0": x0, "y0": y0, "cx_px": x0 + TILE / 2, "cy_px": y0 + TILE / 2,
           "n_nuclei": ncell, "he_tissue_frac": tf, "he_white_frac": wf,
           "mask_frac": float(mask.mean())}
    if ncell < MIN_CELLS_TILE or tf < MIN_HE_TISSUE or mask.mean() < MIN_MASK_FRAC:
        rec["status"] = "skip_lowcontent"
        return rec
    if wf >= MAX_TILE_WHITE_FRAC:
        rec["status"] = "skip_blank"
        return rec
    hem = blur(haematoxylin(rgb))
    if hem.std() < 1e-6:
        rec["status"] = "skip_dead_channel"
        return rec
    tmpl = blur(mask.astype(np.float32))
    m = match_shift(hem, tmpl)
    rec.update({f"he_{k}": v for k, v in m.items()})
    rec["he_sx_um"] = m["sx_px"] * UM_PER_PX
    rec["he_sy_um"] = m["sy_px"] * UM_PER_PX
    rec["status"] = "ok"
    if do_dapi:
        d = read_dapi(x0 - SEARCH, y0 - SEARCH, ex, ex)
        if d is not None:
            md = match_shift(blur(np.log1p(d)), tmpl)
            rec.update({f"dapi_{k}": v for k, v in md.items()})
            rec["dapi_sx_um"] = md["sx_px"] * UM_PER_PX
            rec["dapi_sy_um"] = md["sy_px"] * UM_PER_PX
    return rec


def estimate(field, aligned_dir, xen_dir, nproc, do_dapi, out_dir):
    he_path = Path(aligned_dir) / "he_aligned.ome.tif"
    if not he_path.exists():
        raise FileNotFoundError(he_path)
    dapi = Path(xen_dir) / "morphology_focus" / "morphology_focus_0000.ome.tif"
    if not dapi.exists():
        c = sorted((Path(xen_dir) / "morphology_focus").glob("*.ome.tif"))
        dapi = c[0] if c else None
    if dapi is None:
        do_dapi = False

    t0 = time.time()
    _init(he_path, dapi if do_dapi else None, xen_dir)
    H, W = _G["he"].shape[0], _G["he"].shape[1]
    log(f"[{field}] canvas=({H},{W})  nuclei={_G['nuc']['n_cells']}  dapi={bool(do_dapi)}")

    jobs = [(x, y, do_dapi)
            for y in range(0, H - TILE + 1, TILE)
            for x in range(0, W - TILE + 1, TILE)]
    log(f"[{field}] {len(jobs)} tiles, {nproc} proc")

    if nproc > 1:
        import multiprocessing as mp

        with mp.Pool(nproc, initializer=_init,
                     initargs=(he_path, dapi if do_dapi else None, xen_dir)) as p:
            recs = p.map(measure, jobs, chunksize=8)
    else:
        recs = [measure(j) for j in jobs]


    df = pd.DataFrame(recs)
    numeric_columns = ("x0", "y0", "cx_px", "cy_px", "n_nuclei",
                       "he_tissue_frac", "he_white_frac", "mask_frac")
    for column in numeric_columns:
        if column not in df:
            df[column] = pd.Series(index=df.index, dtype="float64")
    if "status" not in df:
        df["status"] = pd.Series(index=df.index, dtype="object")
    for prefix in ("he", "dapi") if do_dapi else ("he",):
        for suffix in ("sx_px", "sy_px", "peak", "second", "prominence", "sx_um", "sy_um"):
            column = f"{prefix}_{suffix}"
            if column not in df:
                df[column] = pd.Series(index=df.index, dtype="float64")
        column = f"{prefix}_on_border"
        df[column] = (df[column].astype("boolean") if column in df
                      else pd.Series(pd.NA, index=df.index, dtype="boolean"))
    df.insert(0, "field", field)
    ok = df[df["status"] == "ok"].copy()
    gated = ok[(ok["he_prominence"] >= MIN_PROM) & (~ok["he_on_border"])]

    frac_prom = float((ok["he_prominence"] >= MIN_PROM).mean()) if len(ok) else 0.0
    applicable = bool(frac_prom >= APPLICABLE_FRAC and len(gated) >= 20)

    n_blank = int((df["status"] == "skip_blank").sum()) if "status" in df else 0
    res = {
        "field": field,
        "tiles_skipped_blank": n_blank,
        "aligned_he": str(he_path),
        "xenium_dir": str(xen_dir),
        "params": {"TILE": TILE, "SEARCH": SEARCH, "BLUR_SIGMA": BLUR_SIGMA,
                   "MIN_CELLS_TILE": MIN_CELLS_TILE, "MIN_HE_TISSUE": MIN_HE_TISSUE,
                   "WHITE_INTENSITY": WHITE_INTENSITY,
                   "MAX_TILE_WHITE_FRAC": MAX_TILE_WHITE_FRAC,
                   "MIN_MASK_FRAC": MIN_MASK_FRAC, "MIN_PROM": MIN_PROM,
                   "APPLICABLE_FRAC": APPLICABLE_FRAC, "um_per_px": UM_PER_PX},
        "tiles": {"total": int(len(df)), "ok": int(len(ok)), "gated": int(len(gated))},
        "applicability": {"frac_prominence_ok": frac_prom,
                          "verdict": "APPLICABLE" if applicable else "EXCLUDE"},
        "sign_convention": "add the offset to the Xenium coordinate: "
                           "crop_centre = (x_um + dx_um, y_um + dy_um) / 0.2125",
    }

    if applicable:
        dx = float(gated["he_sx_um"].median())
        dy = float(gated["he_sy_um"].median())
        resid = np.hypot(gated["he_sx_um"] - dx, gated["he_sy_um"] - dy)
        rng = np.random.default_rng(0)
        n = len(gated)
        boot = [float(np.median(gated["he_sx_um"].to_numpy()[rng.integers(0, n, n)]))
                for _ in range(200)]
        booty = [float(np.median(gated["he_sy_um"].to_numpy()[rng.integers(0, n, n)]))
                 for _ in range(200)]
        res["offset_um"] = {"dx": dx, "dy": dy, "magnitude": float(np.hypot(dx, dy))}
        res["offset_se_um"] = {"dx": float(np.std(boot)), "dy": float(np.std(booty))}
        res["residual_um"] = {"median": float(np.median(resid)),
                              "p90": float(np.percentile(resid, 90)),
                              "p99": float(np.percentile(resid, 99))}
    else:
        res["offset_um"] = None
        if not len(df):
            res["reason"] = "no complete tile fits the image canvas; offset is unmeasurable"
        elif not len(ok):
            res["reason"] = "all scheduled tiles were skipped; offset is unmeasurable"
        else:
            res["reason"] = (f"{frac_prom:.1%} of measured tiles clear prominence "
                             f"{MIN_PROM}, with {len(gated)} non-border eligible tiles; "
                             f"requires at least {APPLICABLE_FRAC:.0%} and 20 tiles")

    if do_dapi and "dapi_sx_um" in ok.columns:
        dsub = ok.dropna(subset=["dapi_sx_um"])
        if len(dsub):
            res["self_control_um"] = {
                "dx": float(dsub["dapi_sx_um"].median()),
                "dy": float(dsub["dapi_sy_um"].median()),
                "magnitude": float(np.hypot(dsub["dapi_sx_um"].median(),
                                            dsub["dapi_sy_um"].median())),
                "note": "must be ~0; it is the noise floor of this method"}

    res["elapsed_seconds"] = time.time() - t0

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / f"{field}.json").write_text(json.dumps(res, indent=2))
    df.to_csv(out_dir / f"{field}_tiles.csv", index=False)

    v = res["applicability"]["verdict"]
    if applicable:
        o, s = res["offset_um"], res["offset_se_um"]
        log(f"[{field}] {v}  offset=({o['dx']:+.3f},{o['dy']:+.3f}) um "
            f"|d|={o['magnitude']:.3f}  SE=({s['dx']:.3f},{s['dy']:.3f})  "
            f"resid_med={res['residual_um']['median']:.3f}  "
            f"n_gated={len(gated)}  [{res['elapsed_seconds']:.0f}s]")
    else:
        log(f"[{field}] {v}  frac_prom={frac_prom:.3f}  [{res['elapsed_seconds']:.0f}s]")
    if "self_control_um" in res:
        log(f"[{field}] self-control |d|={res['self_control_um']['magnitude']:.3f} um "
            f"(must be ~0)")
    return res


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--field", action="append", required=True,
                    help="sample/field name; repeat for a batch")
    ap.add_argument("--panel", default="5k")
    ap.add_argument("--cancer", default=None,
                    help="cancer subdir under the panel; omit if --aligned-dir given")
    ap.add_argument("--aligned-dir", default=None,
                    help="override: directory holding he_aligned.ome.tif")
    ap.add_argument("--xenium-dir", default=None,
                    help="override: Xenium output dir")
    ap.add_argument("--aligned-root", default=str(ALIGNED_ROOT))
    ap.add_argument("--xenium-root", default=str(XENIUM_ROOT))
    ap.add_argument("--out-dir", default=str(
        Path(__file__).resolve().parents[1] / "outputs/offsets"))
    ap.add_argument("--nproc", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    ap.add_argument("--no-dapi", action="store_true",
                    help="skip the DAPI self-control arm")
    a = ap.parse_args()

    log(f"host={os.uname().nodename}")
    rc = 0
    for f in a.field:
        try:
            adir = a.aligned_dir or str(Path(a.aligned_root) / a.panel / a.cancer / f)
            xdir = a.xenium_dir or str(Path(a.xenium_root) / a.panel / a.cancer / f)
            estimate(f, adir, xdir, a.nproc, not a.no_dapi, a.out_dir)
        except Exception as exc:
            import traceback

            log(f"[{f}] ERROR {exc}")
            traceback.print_exc()
            rc = 1
    return rc


if __name__ == "__main__":
    sys.exit(main())
