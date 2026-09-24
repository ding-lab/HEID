#!/usr/bin/env python
from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path

os.environ.setdefault("PYTHONUNBUFFERED", "1")

import cv2
import numpy as np
import pandas as pd
import tifffile

TARGET_LONG_PX = 2000
N_PHI = 720
MIN_TISSUE_FRAC = 1e-4
SMOOTH_FRAC = 0.01


def _pick_level(tif, target_long):
    series = tif.series[0]
    try:
        levels = list(series.levels)
    except Exception:
        levels = [series]
    chosen, chosen_i = levels[0], 0
    for i, lvl in enumerate(levels):
        if max(lvl.shape[0], lvl.shape[1]) >= target_long:
            chosen, chosen_i = lvl, i
    return chosen, chosen_i, len(levels)


def load_plane(filepath: str, modality: str):
    p = Path(filepath)

    if modality == "xenium":
        focus = p / "morphology_focus"
        if focus.is_dir():
            files = sorted(focus.glob("morphology_focus_*.ome.tif"))
        else:
            files = []
        if not files:
            files = [c for c in (p / "morphology_focus.ome.tif",
                                 p / "morphology.ome.tif") if c.exists()]
        if not files:
            raise FileNotFoundError(f"no Xenium morphology image under {p}")
        with tifffile.TiffFile(str(files[0])) as tif:
            lvl, _, _ = _pick_level(tif, TARGET_LONG_PX)
            full_long = max(tif.series[0].shape[-2], tif.series[0].shape[-1])
            arr = lvl.asarray()
        if arr.ndim == 3:
            arr = arr.max(axis=0)

        for fp in files[1:]:
            with tifffile.TiffFile(str(fp)) as tif:
                lvl2, _, _ = _pick_level(tif, TARGET_LONG_PX)
                a2 = lvl2.asarray()
            if a2.ndim == 3:
                a2 = a2.max(axis=0)
            if a2.shape == arr.shape:
                np.maximum(arr, a2, out=arr)
        ds = full_long / max(arr.shape[0], arr.shape[1])
        return arr, ds

    with tifffile.TiffFile(str(p)) as tif:
        series = tif.series[0]
        axes = str(series.axes)
        s0 = series.shape
        yi, xi = axes.find("Y"), axes.find("X")
        full_long = max(s0[yi], s0[xi]) if yi >= 0 and xi >= 0 else max(s0[-2], s0[-1])

        levels = list(series.levels) if hasattr(series, "levels") else [series]
        order = sorted(range(len(levels)),
                       key=lambda i: -max(levels[i].shape[yi], levels[i].shape[xi]))

        cand = [i for i in order
                if max(levels[i].shape[yi], levels[i].shape[xi]) >= TARGET_LONG_PX]
        cand = (cand[::-1] or order)
        arr, last_err = None, None
        for i in cand:
            try:
                arr = levels[i].asarray()
                break
            except Exception as e:
                last_err = e
        if arr is None:
            raise RuntimeError(f"all pyramid levels failed to decode: {last_err}")


    if "S" in axes:
        si = axes.find("S")
        arr = np.moveaxis(arr, si, -1) if si != arr.ndim - 1 else arr
        arr = np.squeeze(arr)
    elif "C" in axes:
        ci = axes.find("C")
        chan0 = np.take(arr, 0, axis=ci)

        arr = chan0 if float(np.ptp(chan0)) > 0 else arr.max(axis=ci)
        arr = np.squeeze(arr)
    else:
        arr = np.squeeze(arr)
    if arr.ndim > 3:
        arr = arr.reshape(arr.shape[-3:]) if arr.shape[-1] <= 4 else arr[..., 0]

    ds = full_long / max(arr.shape[0], arr.shape[1])
    return arr, ds


def to_u8(a):
    if a.dtype == np.uint8:
        return a
    fg = a[a > 0] if (a > 0).any() else a.ravel()
    lo, hi = np.percentile(fg, [2, 98])
    return np.clip((a.astype(np.float32) - lo) / (hi - lo + 1e-8) * 255,
                   0, 255).astype(np.uint8)


def tissue_mask(arr, modality):
    if modality == "he" and arr.ndim == 3 and arr.shape[-1] >= 3:
        rgb = arr[..., :3]
        if rgb.dtype != np.uint8:
            rgb = to_u8(rgb)
        hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
        m = (hsv[:, :, 1] > 15).astype(np.uint8)
    else:
        if arr.ndim == 3:
            arr = arr.max(axis=-1) if arr.shape[-1] <= 8 else arr[..., 0]
        u8 = to_u8(arr)
        u8 = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8)).apply(u8)
        _, m = cv2.threshold(u8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        m = (m > 0).astype(np.uint8)

    k = max(3, int(round(min(m.shape) * 0.004)) | 1)
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, np.ones((k, k), np.uint8))
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((max(3, k // 2) | 1,) * 2, np.uint8))

    ff = m.copy()
    h, w = ff.shape
    cv2.floodFill(ff, np.zeros((h + 2, w + 2), np.uint8), (0, 0), 1)
    m = (m | (1 - ff)).astype(np.uint8)
    return m


MODES = {
    "iso": np.array([[1.0, 0.0], [0.0, 1.0]]),
    "rot": np.array([[0.0, -1.0], [1.0, 0.0]]),
    "d1":  np.array([[1.0, 0.0], [0.0, -1.0]]),
    "d2":  np.array([[0.0, 1.0], [1.0, 0.0]]),
}
MODE_ORDER = ["iso", "rot", "d1", "d2"]


def smooth_contour(c, frac):
    if frac <= 0:
        return c
    n = len(c)
    k = max(3, int(n * frac) | 1)
    pad = np.concatenate([c[-k:], c, c[:k]], axis=0)
    ker = np.ones(k) / k
    xs = np.convolve(pad[:, 0], ker, mode="same")[k:-k]
    ys = np.convolve(pad[:, 1], ker, mode="same")[k:-k]
    return np.stack([xs, ys], axis=1)


def boundary_gram(contour_xy, centroid, r_max):
    x = (contour_xy - centroid) / max(r_max, 1e-9)
    nxt = np.roll(x, -1, axis=0)
    prv = np.roll(x, 1, axis=0)
    tang = nxt - prv
    seg = np.linalg.norm(nxt - x, axis=1)
    tl = np.linalg.norm(tang, axis=1, keepdims=True)
    tang = tang / np.maximum(tl, 1e-12)

    nrm = np.stack([tang[:, 1], -tang[:, 0]], axis=1)
    flip = np.sign(np.einsum("ij,ij->i", nrm, x))
    flip[flip == 0] = 1.0
    nrm = nrm * flip[:, None]

    comp = np.stack([np.einsum("ij,ij->i", (x @ MODES[m].T), nrm)
                     for m in MODE_ORDER], axis=1)
    P = float(seg.sum())
    G = (comp * seg[:, None]).T @ comp / max(P, 1e-12)
    return G, P


def chirality(contour_xy, centroid):
    d = contour_xy - centroid
    phi = np.arctan2(d[:, 1], d[:, 0])
    r = np.linalg.norm(d, axis=1)
    order = np.argsort(phi)
    phi, r = phi[order], r[order]
    grid = np.linspace(-np.pi, np.pi, N_PHI, endpoint=False)
    rg = np.interp(grid, phi, r, period=2 * np.pi)
    rg = rg - rg.mean()
    if np.allclose(rg, 0):
        return 0.0

    rev = rg[::-1]
    F = np.fft.rfft(rg)
    Frev = np.fft.rfft(rev)
    cc = np.fft.irfft(F * np.conj(Frev), n=N_PHI)
    denom = float(rg @ rg)
    best = float(cc.max()) / max(denom, 1e-12)
    return float(np.clip(1.0 - best, 0.0, 1.0))


def analyse(mask, mpp_eff):
    n_lab, lab, stats, cents = cv2.connectedComponentsWithStats(mask, connectivity=8)
    if n_lab <= 1:
        return None, None
    areas = stats[1:, cv2.CC_STAT_AREA]
    blocks = int((areas >= max(50, 0.02 * areas.max())).sum())
    keep = 1 + int(np.argmax(areas))
    m1 = (lab == keep).astype(np.uint8)

    cnts, _ = cv2.findContours(m1, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not cnts:
        return None, None
    c = max(cnts, key=cv2.contourArea).reshape(-1, 2).astype(np.float64)
    if len(c) < 32:
        return None, None

    area_px = float(m1.sum())
    centroid = np.array([cents[keep][0], cents[keep][1]], dtype=np.float64)
    rad = np.linalg.norm(c - centroid, axis=1)
    r_max = float(rad.max())

    def _cond(cc):
        G_, _ = boundary_gram(cc, centroid, r_max)
        ev, evec = np.linalg.eigh(G_)
        ev = np.clip(ev, 0.0, None)
        return (G_, float(ev.max() / max(ev.min(), 1e-15)), float(ev.min()),
                float(ev.max()),
                MODE_ORDER[int(np.argmax(np.abs(evec[:, int(np.argmin(ev))])))])

    G_raw, cond_raw, _, _, _ = _cond(c)
    G, cond, eig_min, eig_max, weakest = _cond(smooth_contour(c, SMOOTH_FRAC))

    perim_px = float(cv2.arcLength(c.astype(np.float32).reshape(-1, 1, 2), True))
    out = {
        "area_um2": area_px * mpp_eff ** 2,
        "perimeter_um": perim_px * mpp_eff,
        "R_h_um": (area_px * mpp_eff ** 2) / max(perim_px * mpp_eff, 1e-9),
        "R_max_um": r_max * mpp_eff,
        "n_blocks": blocks,
        "gram_cond": cond,
        "gram_cond_raw": cond_raw,
        "gram_smooth_frac": SMOOTH_FRAC,
        "gram_weakest_mode": weakest,
        "gram_eigmin": eig_min,
        "gram_eigmax": eig_max,
        "chirality": chirality(c, centroid),
        "tissue_frac": area_px / float(mask.size),
        "mpp_eff_um": mpp_eff,
        "gram_convention": "G_ij=(1/P)*int (E_i x.n)(E_j x.n) ds, x centred & /R_max; cond from contour smoothed by gram_smooth_frac (raw has a ~35 ceiling from rasterisation)",
        "R_h_convention": "A/P",
    }
    for i, m in enumerate(MODE_ORDER):
        out[f"gram_eig_{m}_diag"] = float(G[i, i])


    R = max(r_max, 1e-9)
    G_um2 = G * (R * mpp_eff) ** 2
    out["gram_mode_order"] = MODE_ORDER
    out["gram_matrix_normalized"] = [[float(v) for v in row] for row in G]
    out["gram_matrix_um2"] = [[float(v) for v in row] for row in G_um2]
    try:
        Gi = np.linalg.inv(G)
        out["gram_inv_diag_normalized"] = [float(Gi[i, i]) for i in range(4)]
        out["gram_sqrt_inv_diag_normalized"] = [float(np.sqrt(max(Gi[i, i], 0.0)))
                                                for i in range(4)]
    except np.linalg.LinAlgError:
        out["gram_inv_diag_normalized"] = None
        out["gram_sqrt_inv_diag_normalized"] = None
    out["gram_units_note"] = ("gram_matrix_normalized is dimensionless (x/R_max); "
                              "gram_matrix_um2 has x in um. R_max_um and "
                              "perimeter_um are stored so either convention can "
                              "be reconstructed.")
    return out, m1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sections", required=True, help="section z-order table (TSV)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--index", type=int, default=-1,
                    help="SLURM array index; -1 = all rows in one process")
    args = ap.parse_args()

    out = Path(args.out)
    (out / "per_section").mkdir(parents=True, exist_ok=True)
    (out / "thumbs").mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.sections, sep="\t")
    rows = df.to_dict("records")
    todo = rows if args.index < 0 else [rows[args.index]]

    for r in todo:
        sid = str(r["section_id"])
        rec = {"section_id": sid, "u_number": int(r["u_number"]),
               "z_position_um": float(r["z_position_um"]),
               "modality": str(r["modality"]), "mpp_level0_um": float(r["mpp"]),
               "filepath": str(r["filepath"])}
        t0 = time.time()
        try:
            arr, ds = load_plane(str(r["filepath"]), str(r["modality"]))
            mpp_eff = float(r["mpp"]) * ds
            mask = tissue_mask(arr, str(r["modality"]))
            if mask.sum() < MIN_TISSUE_FRAC * mask.size:
                raise RuntimeError(f"tissue fraction too small: {mask.mean():.2e}")
            geom, m1 = analyse(mask, mpp_eff)
            if geom is None:
                raise RuntimeError("no usable contour")
            rec.update(geom)
            rec["downsample_vs_level0"] = ds
            rec["ok"] = True
            rec["error"] = ""
            thumb = cv2.resize((m1 * 255).astype(np.uint8), (512, 512),
                               interpolation=cv2.INTER_NEAREST)
            cv2.imwrite(str(out / "thumbs" / f"{sid}.png"), thumb)
        except Exception as e:
            rec["ok"] = False
            rec["error"] = f"{type(e).__name__}: {e}"
        rec["elapsed_s"] = round(time.time() - t0, 2)
        (out / "per_section" / f"{sid}.json").write_text(json.dumps(rec, indent=2))
        print(f"[{rec['elapsed_s']:6.1f}s] {sid:16s} {rec['modality']:7s} "
              f"ok={rec['ok']} "
              f"{'cond=%.2f chir=%.4f' % (rec.get('gram_cond', -1), rec.get('chirality', -1)) if rec['ok'] else rec['error'][:70]}",
              flush=True)

    if args.index >= 0:
        return

    aggregate(out, args.sections)


def _sha256(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for blk in iter(lambda: f.read(1 << 20), b""):
            h.update(blk)
    return h.hexdigest()


def aggregate(out: Path, sections_tsv: str):
    recs = [json.loads(p.read_text()) for p in sorted((out / "per_section").glob("*.json"))]
    if not recs:
        print("no per-section records to aggregate", flush=True)
        return
    df = pd.DataFrame(recs)
    df.to_parquet(out / "sections_geometry.parquet", index=False)
    ok = df[df["ok"]]

    def q(col, f):
        return float(f(ok[col])) if len(ok) and col in ok else None

    summary = {
        "n_sections": int(len(df)),
        "n_ok": int(len(ok)),
        "n_failed": int((~df["ok"]).sum()),
        "failed_ids": df.loc[~df["ok"], "section_id"].tolist(),
        "gram_cond": {"median": q("gram_cond", np.median),
                      "p90": q("gram_cond", lambda s: np.percentile(s, 90)),
                      "max": q("gram_cond", np.max),
                      "n_over_20": int((ok["gram_cond"] > 20).sum()) if len(ok) else None,
                      "ids_over_20": ok.loc[ok["gram_cond"] > 20, "section_id"].tolist() if len(ok) else []},
        "chirality": {"median": q("chirality", np.median),
                      "min": q("chirality", np.min),
                      "n_under_0p01": int((ok["chirality"] < 0.01).sum()) if len(ok) else None,
                      "ids_under_0p01": ok.loc[ok["chirality"] < 0.01, "section_id"].tolist() if len(ok) else []},
        "perimeter_um": {"median": q("perimeter_um", np.median)},
        "R_h_um": {"median": q("R_h_um", np.median)},
        "n_blocks_gt1": int((ok["n_blocks"] > 1).sum()) if len(ok) else None,
        "interpretation": {
            "gram_cond_gt_20": "section cannot pin 6 DOF from its outline alone -> needs a second channel",
            "chirality_near_0": "flipped mount NOT detectable from the contour -> must be flagged in QC and decided by a non-mask channel",
        },
    }
    (out / "results.json").write_text(json.dumps(summary, indent=2))

    try:
        git_sha = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True,
                                 text=True, cwd=str(Path(__file__).resolve().parent)
                                 ).stdout.strip() or "unknown"
    except Exception:
        git_sha = "unknown"
    manifest = {
        "run_id": "footprint_geometry",
        "stage": "footprint geometry (no registration)",
        "script": str(Path(__file__).resolve()),
        "script_git_sha": git_sha,
        "data_version": {"sections_tsv": str(sections_tsv),
                         "sections_tsv_sha256": _sha256(sections_tsv)},
        "seed": None,
        "deterministic": True,
        "python": sys.version.split()[0],
        "numpy": np.__version__, "opencv": cv2.__version__,
        "tifffile": tifffile.__version__, "pandas": pd.__version__,
        "slurm_job_id": os.environ.get("SLURM_ARRAY_JOB_ID") or os.environ.get("SLURM_JOB_ID"),
        "cpu_hours": None,
    }
    (out / "run_manifest.json").write_text(json.dumps(manifest, indent=2))

    print(f"\naggregated {len(ok)}/{len(df)} ok -> {out}", flush=True)
    print(json.dumps(summary["gram_cond"], indent=2), flush=True)
    print(json.dumps(summary["chirality"], indent=2), flush=True)


if __name__ == "__main__":
    main()
