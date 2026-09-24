#!/usr/bin/env python
from __future__ import annotations
import os

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import fft as sfft
from scipy import ndimage
from scipy.spatial import cKDTree

ROOT = Path(os.environ.get("THREED_ROOT", os.path.join(os.environ.get("PROJECTS_ROOT", "/data/heid"), "3d")))
WORK = ROOT / "xenium"
XEN = WORK / "data/xenium"
RUNS = ROOT / "registration/runs"
PROBE = RUNS / "prep_sections/source_probe.json"
CELLS = ROOT / "inference/cohort/data/cells"


PAIRS = {
    "HT891Z1-U1":   "HT891Z1-S2H3Fp1U32",
    "HT891Z1-U21":  "HT891Z1-S2H3Fp1U21",
    "HT891Z1-U31":  "HT891Z1-S2H3Fp1U1",
    "HT891Z1-U44":  "HT891Z1-S2H3Fp1U69",
    "HT891Z1-U59":  "HT891Z1-S2H3Fp1U59",
    "HT891Z1-U69":  "HT891Z1-S2H3Fp1U44",
    "HT891Z1-U81":  "HT891Z1-S2H3Fp1U81",
    "HT891Z1-U94":  "HT891Z1-S2H3Fp1U94",
    "HT891Z1-U104": "HT891Z1-S2H3Fp1U104",
}

SIL_MPP = 25.0

ANGLE_STEP = 0.5
MARGIN_UM = 1500.0
N_CAND = 6
ANGLE_NMS_DEG = 8.0

NUC_MPP_COARSE = 6.0
NUC_MPP_FINE = 2.0
NUC_SIGMA_PX = 1.2

NUC_HALF_DEG = 6.0
NUC_STEP_COARSE = 0.5
NUC_HALF_FINE, NUC_STEP_FINE = 0.75, 0.05
NUC_MARGIN_UM = 400.0
N_SHARPEN = 2
ICP_CAPS = [200.0, 100.0, 60.0, 30.0, 15.0, 8.0, 5.0, 4.0]
ICP_ITERS_PER_CAP = 6
TRIM_Q = 0.80
MATCH_R = 2.0


def xenium_dir(slide_u: str) -> Path:
    hits = sorted(XEN.glob(f"*__{slide_u}__*"))
    if len(hits) != 1:
        raise SystemExit(f"{slide_u}: {len(hits)} xenium dirs matched")
    return hits[0]


def load_xenium(d: Path):
    cells = pd.read_parquet(d / "cells.parquet",
                            columns=["cell_id", "x_centroid", "y_centroid",
                                     "nucleus_area", "cell_area", "transcript_counts"])
    nb = pd.read_parquet(d / "nucleus_boundaries.parquet")
    vx = [c for c in nb.columns if c.endswith("vertex_x")][0]
    vy = [c for c in nb.columns if c.endswith("vertex_y")][0]
    g = nb.groupby("cell_id", sort=False)[[vx, vy]].mean()
    g.columns = ["nx", "ny"]
    cells = cells.join(g, on="cell_id")
    n_poly = int(cells["nx"].notna().sum())
    cells["nx"] = cells["nx"].fillna(cells["x_centroid"])
    cells["ny"] = cells["ny"].fillna(cells["y_centroid"])
    meta = json.loads((d / "experiment.xenium").read_text())
    return cells, float(meta["pixel_size"]), meta, n_poly


def load_he(slide: str) -> pd.DataFrame:
    return pd.read_parquet(CELLS / f"{slide}.parquet",
                           columns=["cell_id", "x_px", "y_px", "x_um", "y_um"])


def silhouette(pts, mpp, origin, shape):
    ix = np.floor((pts[:, 0] - origin[0]) / mpp).astype(np.int64)
    iy = np.floor((pts[:, 1] - origin[1]) / mpp).astype(np.int64)
    k = (ix >= 0) & (ix < shape[1]) & (iy >= 0) & (iy < shape[0])
    g = np.bincount(iy[k] * shape[1] + ix[k], minlength=shape[0] * shape[1])
    m = g.reshape(shape) > 0
    return ndimage.binary_closing(m, np.ones((3, 3), bool)).astype(np.float32)


def best_shift(A, FB, sumB):
    c = np.fft.irfft2(np.conj(np.fft.rfft2(A)) * FB, s=A.shape)
    k = int(np.argmax(c))
    dy, dx = divmod(k, A.shape[1])
    if dy > A.shape[0] // 2:
        dy -= A.shape[0]
    if dx > A.shape[1] // 2:
        dx -= A.shape[1]
    return dy, dx, 2.0 * float(c.flat[k]) / max(1e-9, float(A.sum()) + sumB)


def rot(deg):
    t = np.deg2rad(deg)
    c, s = np.cos(t), np.sin(t)
    return np.array([[c, -s], [s, c]])


def enumerate_poses(he_xy, xe_xy):
    c_he, c_xe = he_xy.mean(0), xe_xy.mean(0)
    lo = xe_xy.min(0) - MARGIN_UM
    hi = xe_xy.max(0) + MARGIN_UM
    shape = (int(np.ceil((hi[1] - lo[1]) / SIL_MPP)),
             int(np.ceil((hi[0] - lo[0]) / SIL_MPP)))
    B = silhouette(xe_xy, SIL_MPP, lo, shape)
    FB, sumB = np.fft.rfft2(B), float(B.sum())

    scored = []
    for f in (1.0, -1.0):
        u0 = (he_xy - c_he) * np.array([f, 1.0])
        for th in np.arange(0.0, 360.0, ANGLE_STEP):
            v = u0 @ rot(th).T + c_xe
            dy, dx, dice = best_shift(silhouette(v, SIL_MPP, lo, shape), FB, sumB)
            scored.append((dice, f, float(th),
                           np.array([dx * SIL_MPP, dy * SIL_MPP], float)))
    scored.sort(key=lambda z: -z[0])
    kept = []
    for cand in scored:
        if all(not (cand[1] == k[1] and
                    min(abs(cand[2] - k[2]), 360 - abs(cand[2] - k[2])) < ANGLE_NMS_DEG)
               for k in kept):
            kept.append(cand)
        if len(kept) == N_CAND:
            break
    return kept, c_he, c_xe


def smooth_density(pts, mpp, lo, shape):
    ix = np.floor((pts[:, 0] - lo[0]) / mpp).astype(np.int64)
    iy = np.floor((pts[:, 1] - lo[1]) / mpp).astype(np.int64)
    k = (ix >= 0) & (ix < shape[1]) & (iy >= 0) & (iy < shape[0])
    g = np.bincount(iy[k] * shape[1] + ix[k],
                    minlength=shape[0] * shape[1]).reshape(shape).astype(np.float32)
    return ndimage.gaussian_filter(g, NUC_SIGMA_PX)


def nucleus_sweep(u0, xe_xy, th0, centre0, mpp, half_deg, step_deg):
    lo = xe_xy.min(0) - NUC_MARGIN_UM
    hi = xe_xy.max(0) + NUC_MARGIN_UM
    shape = (int(np.ceil((hi[1] - lo[1]) / mpp)), int(np.ceil((hi[0] - lo[0]) / mpp)))
    B = smooth_density(xe_xy, mpp, lo, shape)
    B -= B.mean()
    nB = float(np.linalg.norm(B))
    FB = sfft.rfft2(B, workers=-1)

    best = None
    for dth in np.arange(-half_deg, half_deg + 1e-9, step_deg):
        th = th0 + float(dth)
        A = smooth_density(u0 @ rot(th).T + centre0, mpp, lo, shape)
        A -= A.mean()
        nA = float(np.linalg.norm(A))
        c = sfft.irfft2(np.conj(sfft.rfft2(A, workers=-1)) * FB, s=shape, workers=-1)
        k = int(np.argmax(c))
        dy, dx = divmod(k, shape[1])
        if dy > shape[0] // 2:
            dy -= shape[0]
        if dx > shape[1] // 2:
            dx -= shape[1]
        ncc = float(c.flat[k]) / max(nA * nB, 1e-9)
        if best is None or ncc > best[2]:
            best = (th, centre0 + np.array([dx * mpp, dy * mpp], float), ncc)
    return best


def umeyama_sim(src, dst):
    mu_s, mu_d = src.mean(0), dst.mean(0)
    s0, d0 = src - mu_s, dst - mu_d
    H = (d0.T @ s0) / len(src)
    U, S, Vt = np.linalg.svd(H)
    D = np.eye(2)
    if np.linalg.det(U @ Vt) < 0:
        D[1, 1] = -1.0
    R = U @ D @ Vt
    var = (s0 ** 2).sum() / len(src)
    s = float((S * np.diag(D)).sum() / max(var, 1e-12))
    return s, R, mu_d - s * (R @ mu_s)


def icp(u, tree, s, R, t, caps, iters=ICP_ITERS_PER_CAP):
    hist = []
    for cap in caps:
        rmse, n_used = float("nan"), 0
        for _ in range(iters):
            v = s * (u @ R.T) + t
            d, j = tree.query(v, k=1, workers=-1)
            k = d < cap
            if k.sum() < 500:
                break
            dd = d[k]
            thr = np.quantile(dd, TRIM_Q)
            sel = np.where(k)[0][dd <= thr]
            if len(sel) < 500:
                break
            s, R, t = umeyama_sim(u[sel], tree.data[j[sel]])
            rmse = float(np.sqrt((dd[dd <= thr] ** 2).mean()))
            n_used = int(len(sel))
        hist.append({"cap_um": cap, "n_used": n_used, "rmse_um": round(rmse, 3)})
    return s, R, t, hist


def nn_median(u, tree, s, R, t):
    d, _ = tree.query(s * (u @ R.T) + t, k=1, workers=-1)
    return float(np.median(d))


def fit_pair(he_xy, xe_xy, tag, log, rng=None):
    cands, c_he, c_xe = enumerate_poses(he_xy, xe_xy)
    tree = cKDTree(xe_xy)

    ranked = []
    for dice, f, th, tr in cands:
        u0 = (he_xy - c_he) * np.array([f, 1.0])
        th1, ctr1, ncc = nucleus_sweep(u0, xe_xy, th, c_xe + tr, NUC_MPP_COARSE,
                                       NUC_HALF_DEG, NUC_STEP_COARSE)
        ranked.append({"mirror": f, "silhouette_deg": th, "dice": dice,
                       "coarse_deg": th1, "coarse_ncc": ncc, "centre": ctr1})
        log(f"    {tag:<18} cand mirror={f:+.0f} sil {th:6.1f} deg dice {dice:.4f}"
            f"  ->  nuc {th1:7.2f} deg  ncc {ncc:.4f}")
    ranked.sort(key=lambda z: -z["coarse_ncc"])

    sharp = []
    for c in ranked[:N_SHARPEN]:
        u0 = (he_xy - c_he) * np.array([c["mirror"], 1.0])
        th2, ctr2, ncc2 = nucleus_sweep(u0, xe_xy, c["coarse_deg"], c["centre"],
                                        NUC_MPP_FINE, NUC_HALF_FINE, NUC_STEP_FINE)
        sharp.append({**c, "fine_deg": th2, "fine_ncc": ncc2, "centre": ctr2})
        log(f"    {tag:<18} sharpen mirror={c['mirror']:+.0f} -> {th2:7.2f} deg  "
            f"ncc {ncc2:.4f}")
    sharp.sort(key=lambda z: -z["fine_ncc"])
    win = sharp[0]
    run = sharp[1] if len(sharp) > 1 else None

    f = win["mirror"]
    u0 = (he_xy - c_he) * np.array([f, 1.0])
    s, R, t, hist = icp(u0, tree, 1.0, rot(win["fine_deg"]), win["centre"], ICP_CAPS)

    F = np.array([[f, 0.0], [0.0, 1.0]])
    A = s * (R @ F)
    b = t - A @ c_he
    info = {
        "mirror": f,
        "silhouette_dice": win["dice"],
        "best_silhouette_dice_any_pose": max(c[0] for c in cands),
        "silhouette_deg": win["silhouette_deg"],
        "nucleus_deg": win["fine_deg"],
        "silhouette_minus_nucleus_deg": round(win["silhouette_deg"] - win["fine_deg"], 3),
        "nucleus_ncc": round(win["fine_ncc"], 4),
        "runner_up_nucleus_ncc": round(run["fine_ncc"], 4) if run else None,
        "candidates": [{k: (round(v, 4) if isinstance(v, float) else v)
                        for k, v in c.items() if k != "centre"} for c in ranked],
        "scale": float(s),
        "rot_deg": float(np.rad2deg(np.arctan2(R[1, 0], R[0, 0]))),
        "icp": hist,
    }
    return A, b, info, tree


def diagnose(he_xy, A, b, tree, xe_xy, occ_mpp=25.0):
    v = he_xy @ A.T + b
    lo = xe_xy.min(0) - 50.0
    hi = xe_xy.max(0) + 50.0
    shape = (int(np.ceil((hi[1] - lo[1]) / occ_mpp)),
             int(np.ceil((hi[0] - lo[0]) / occ_mpp)))
    occ = silhouette(xe_xy, occ_mpp, lo, shape) > 0
    ix = np.floor((v[:, 0] - lo[0]) / occ_mpp).astype(np.int64)
    iy = np.floor((v[:, 1] - lo[1]) / occ_mpp).astype(np.int64)
    ok = (ix >= 0) & (ix < shape[1]) & (iy >= 0) & (iy < shape[0])
    inside = np.zeros(len(v), bool)
    inside[ok] = occ[iy[ok], ix[ok]]

    d, _ = tree.query(v, k=1, workers=-1)
    di = d[inside] if inside.any() else d
    d9, _ = tree.query(xe_xy[::7], k=9, workers=-1)
    rho = float(np.median(8.0 / (np.pi * np.maximum(d9[:, 8], 1e-6) ** 2)))
    return {
        "n_he": int(len(v)),
        "n_he_inside_xenium": int(inside.sum()),
        "frac_he_inside_xenium": round(float(inside.mean()), 4),
        "nn_median_um": round(float(np.median(di)), 3),
        "nn_p90_um": round(float(np.percentile(di, 90)), 3),
        "frac_nn_lt_2um": round(float((di < MATCH_R).mean()), 4),
        "frac_nn_lt_5um": round(float((di < 5.0).mean()), 4),
        "chance_lt_2um": round(float(1.0 - np.exp(-rho * np.pi * MATCH_R ** 2)), 4),
        "xenium_density_per_mm2": round(rho * 1e6, 1),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(WORK / "outputs/registration.json"))
    ap.add_argument("--only", default="")
    a = ap.parse_args()
    t00 = time.time()

    def log(*m):
        print(*m, flush=True)

    probe = {p["section_id"]: p for p in json.loads(PROBE.read_text())}
    order = [s for s in PAIRS if not a.only or s in a.only.split(",")]

    xe_cache = {}
    for sid in list(PAIRS):
        u = sid.split("-U")[1]
        d = xenium_dir(f"HT891Z1-S2H3Fp1U{u}")
        cells, px, meta, n_poly = load_xenium(d)
        xe_cache[sid] = {"dir": str(d), "cells": cells, "pixel_size": px,
                         "region": meta["region_name"], "n_poly": n_poly,
                         "xy": np.stack([cells["nx"].to_numpy(),
                                         cells["ny"].to_numpy()], 1)}
        log(f"{sid}: xenium {meta['region_name']}  {len(cells)} cells "
            f"({n_poly} with a nucleus polygon)  pixel_size {px}")

    all_sids = list(PAIRS)
    results = {}
    for sid in order:
        i = all_sids.index(sid)
        slide = PAIRS[sid]
        log(f"\n=== {sid}  <-  {slide}   ({time.time()-t00:.0f}s)")
        he = load_he(slide)
        he_xy = np.stack([he["x_um"].to_numpy(), he["y_um"].to_numpy()], 1)
        xe = xe_cache[sid]
        xe_xy = xe["xy"]


        H0, W0 = probe[sid]["level_yx"][0]
        span = xe_xy.max(0) / xe["pixel_size"]
        if span[0] > W0 + 2 or span[1] > H0 + 2:
            raise SystemExit(f"{sid}: centroids reach {span} px but the DAPI image is "
                             f"{W0}x{H0} -- the micron->pixel contract is wrong")

        A, b, fit, tree = fit_pair(he_xy, xe_xy, sid, log)
        dg = diagnose(he_xy, A, b, tree, xe_xy)
        log(f"    -> WINNER mirror {fit['mirror']:+.0f}  rot {fit['rot_deg']:.3f} deg  "
            f"scale {fit['scale']:.5f}")
        log(f"    -> nn_median {dg['nn_median_um']:.2f} um  "
            f"frac<2um {dg['frac_nn_lt_2um']:.3f} (chance {dg['chance_lt_2um']:.3f})  "
            f"ncc {fit['nucleus_ncc']:.4f} (runner-up {fit['runner_up_nucleus_ncc']})"
            f"  sil-nuc angle gap {fit['silhouette_minus_nucleus_deg']:+.2f} deg")

        other = all_sids[(i + len(all_sids) // 2) % len(all_sids)]
        o_xy = xe_cache[other]["xy"]
        Ao, bo, fito, treeo = fit_pair(he_xy, o_xy, f"{sid}~ctl", log)
        dgo = diagnose(he_xy, Ao, bo, treeo, o_xy)
        log(f"    -> CONTROL vs {other}: dice {fito['silhouette_dice']:.4f}  "
            f"nn_median {dgo['nn_median_um']:.2f} um  "
            f"frac<2um {dgo['frac_nn_lt_2um']:.3f}")

        results[sid] = {
            "section_id": sid, "slide": slide,
            "xenium_dir": xe["dir"], "xenium_region": xe["region"],
            "pixel_size_um": xe["pixel_size"], "dapi_level0_wh": [W0, H0],
            "n_xenium_cells": int(len(xe["cells"])),
            "n_xenium_with_nucleus_polygon": int(xe["n_poly"]),
            "affine_he_um_to_xenium_um": {"A": A.tolist(), "b": b.tolist()},
            "fit": fit,
            "diagnostics": dg,
            "control_wrong_section": {
                "against": other,
                "affine_he_um_to_that_xenium_um": {"A": Ao.tolist(), "b": bo.tolist()},
                "silhouette_dice": float(fito["silhouette_dice"]),
                "scale": float(fito["scale"]), **dgo},
        }

    outp = Path(a.out)
    outp.parent.mkdir(parents=True, exist_ok=True)
    outp.write_text(json.dumps({
        "what": "similarity transforms taking H&E microns to Xenium microns for the "
                "nine HT891Z1 sections that carry a Xenium run",
        "model": "p_xenium_um = A . p_he_um + b, A = s.R.F; the mirror comes from the "
                 "winning candidate and there is no non-rigid term anywhere",
        "he_um_definition": "x_um,y_um in inference/cohort/data/cells/*.parquet, "
                            "i.e. level-0 H&E pixels x 0.5001807586523194 um/px",
        "xenium_um_definition": "nucleus polygon vertex mean from "
                                "nucleus_boundaries.parquet; cell centroid for cells "
                                "with no nucleus polygon",
        "pose_selection": "candidates enumerated by silhouette Dice at 25 um/px, then "
                          "decided by normalised cross-correlation of nucleus-density "
                          "maps swept over +-6 deg -- wider than the 8 deg window the "
                          "candidate list was thinned to, because the best silhouette "
                          "angle has been seen 4 deg off the true one and the thinning "
                          "then discards the true one as a near-duplicate.",
        "match_radius_um": MATCH_R,
        "elapsed_s": round(time.time() - t00, 1),
        "sections": results,
    }, indent=1, default=float))
    log(f"\nwrote {outp}  ({time.time()-t00:.0f}s)")


if __name__ == "__main__":
    main()
