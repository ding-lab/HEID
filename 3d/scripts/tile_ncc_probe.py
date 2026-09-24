#!/usr/bin/env python
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("PYTHONUNBUFFERED", "1")

import cv2
import numpy as np
import pandas as pd
import tifffile

sys.path.insert(0, str(Path(__file__).resolve().parent))
from importlib import util as _iu

_spec = _iu.spec_from_file_location(
    "s0", str(Path(__file__).resolve().parent / "footprint_geometry.py"))
_s0 = _iu.module_from_spec(_spec)
_spec.loader.exec_module(_s0)


def match_tile(image, template, search_px):
    r = cv2.matchTemplate(image.astype(np.float32), template.astype(np.float32),
                          cv2.TM_CCOEFF_NORMED)
    if not np.isfinite(r).all():
        r = np.nan_to_num(r, nan=-1.0, posinf=-1.0, neginf=-1.0)
    j, i = np.unravel_index(int(np.argmax(r)), r.shape)
    peak = float(r[j, i])
    on_border = bool(i <= 1 or j <= 1 or i >= r.shape[1] - 2 or j >= r.shape[0] - 2)

    def _par(v0, vm, vp):
        den = vm - 2 * v0 + vp
        return 0.0 if abs(den) < 1e-12 else float(np.clip(0.5 * (vm - vp) / den, -1, 1))

    if 0 < i < r.shape[1] - 1:
        dx_sub = _par(r[j, i], r[j, i - 1], r[j, i + 1])
    else:
        dx_sub = 0.0
    if 0 < j < r.shape[0] - 1:
        dy_sub = _par(r[j, i], r[j - 1, i], r[j + 1, i])
    else:
        dy_sub = 0.0

    H = None
    if 0 < i < r.shape[1] - 1 and 0 < j < r.shape[0] - 1:
        rxx = r[j, i + 1] - 2 * r[j, i] + r[j, i - 1]
        ryy = r[j + 1, i] - 2 * r[j, i] + r[j - 1, i]
        rxy = 0.25 * (r[j + 1, i + 1] - r[j + 1, i - 1] - r[j - 1, i + 1] + r[j - 1, i - 1])
        H = -np.array([[rxx, rxy], [rxy, ryy]], dtype=np.float64)

    mask = np.ones_like(r, bool)
    mask[max(0, j - 10):j + 11, max(0, i - 10):i + 11] = False
    second = float(r[mask].max()) if mask.any() else -1.0

    out = {"dx_px": (i - search_px) + dx_sub, "dy_px": (j - search_px) + dy_sub,
           "peak": peak, "second": second, "prominence": peak - second,
           "on_border": on_border}
    if H is not None:
        ev = np.linalg.eigvalsh(H)
        out.update({"H_xx": float(H[0, 0]), "H_xy": float(H[0, 1]),
                    "H_yy": float(H[1, 1]),
                    "H_eigmin": float(ev.min()), "H_eigmax": float(ev.max()),
                    "H_cond": float(ev.max() / ev.min()) if ev.min() > 1e-12 else np.inf,
                    "H_posdef": bool(ev.min() > 0)})
    else:
        out.update({"H_xx": np.nan, "H_xy": np.nan, "H_yy": np.nan,
                    "H_eigmin": np.nan, "H_eigmax": np.nan,
                    "H_cond": np.inf, "H_posdef": False})
    return out


def tile_grid(fixed, moving, tile_px, search_px, stride_px=None):
    stride_px = stride_px or tile_px
    H, W = fixed.shape[:2]
    recs = []
    for y in range(search_px, H - tile_px - search_px + 1, stride_px):
        for x in range(search_px, W - tile_px - search_px + 1, stride_px):
            tmpl = fixed[y:y + tile_px, x:x + tile_px]
            if float(tmpl.std()) < 1e-6:
                continue
            ex = tile_px + 2 * search_px
            img = moving[y - search_px:y - search_px + ex,
                         x - search_px:x - search_px + ex]
            if img.shape[0] != ex or img.shape[1] != ex:
                continue
            m = match_tile(img, tmpl, search_px)
            m.update({"x": x + tile_px / 2, "y": y + tile_px / 2,
                      "tmpl_std": float(tmpl.std())})
            recs.append(m)
    return pd.DataFrame(recs)


def hematoxylin_u8(rgb):
    img = np.clip(rgb.astype(np.float32) / 255.0, 1e-6, 1.0)
    od = -np.log10(img)
    S = np.array([[0.6442, 0.7166, 0.2668], [0.0928, 0.9541, 0.2831], [0, 0, 0]],
                 dtype=np.float32)
    S[0] /= np.linalg.norm(S[0]); S[1] /= np.linalg.norm(S[1])
    S[2] = np.cross(S[0], S[1]); S[2] /= np.linalg.norm(S[2])
    conc = od.reshape(-1, 3) @ np.linalg.inv(S.T).T
    h = conc[:, 0].reshape(rgb.shape[:2])
    p1, p99 = np.percentile(h, (1, 99))
    return np.clip((h - p1) / (p99 - p1 + 1e-6) * 255, 0, 255).astype(np.uint8)


def clahe(u8):
    return cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8)).apply(u8)


def to_common_grid(arr, mpp_in, mpp_out):
    f = mpp_in / mpp_out
    if abs(f - 1.0) < 1e-3:
        return arr
    return cv2.resize(arr, (max(1, int(round(arr.shape[1] * f))),
                            max(1, int(round(arr.shape[0] * f)))),
                      interpolation=cv2.INTER_AREA if f < 1 else cv2.INTER_LINEAR)


def load_channel(row, mpp_out):
    arr, ds = _s0.load_plane(str(row["filepath"]), str(row["modality"]))
    mpp_eff = float(row["mpp"]) * ds
    if str(row["modality"]) == "he" and arr.ndim == 3:
        chan = hematoxylin_u8(arr[..., :3] if arr.dtype == np.uint8
                              else _s0.to_u8(arr[..., :3]))
    else:
        if arr.ndim == 3:
            arr = arr[..., 0]
        chan = 255 - _s0.to_u8(arr)
    chan = clahe(chan)
    mask = _s0.tissue_mask(arr, str(row["modality"]))
    return (to_common_grid(chan, mpp_eff, mpp_out),
            to_common_grid(mask.astype(np.uint8), mpp_eff, mpp_out) > 0)


def rigid_prealign(fix_mask, mov_mask, mov_img):
    def cen(m):
        ys, xs = np.nonzero(m)
        return (xs.mean(), ys.mean()) if len(xs) else (0.0, 0.0)
    fx, fy = cen(fix_mask); mx, my = cen(mov_mask)
    M = np.float32([[1, 0, fx - mx], [0, 1, fy - my]])
    h, w = fix_mask.shape
    return (cv2.warpAffine(mov_img, M, (w, h), flags=cv2.INTER_LINEAR, borderValue=0),
            cv2.warpAffine(mov_mask.astype(np.uint8), M, (w, h),
                           flags=cv2.INTER_NEAREST, borderValue=0) > 0,
            (float(fx - mx), float(fy - my)))


def summarise(df, mpp, prom_gate, cond_gate):
    if len(df) == 0:
        return {"n_tiles": 0}
    good = df[(df["prominence"] >= prom_gate) & (~df["on_border"])
              & (df["H_posdef"]) & (df["H_cond"] <= cond_gate)]
    d = np.hypot(df["dx_px"], df["dy_px"]) * mpp
    dg = np.hypot(good["dx_px"], good["dy_px"]) * mpp if len(good) else pd.Series([], dtype=float)
    q = lambda s, p: float(np.percentile(s, p)) if len(s) else None
    return {
        "n_tiles": int(len(df)),
        "n_pass": int(len(good)),
        "frac_pass": float(len(good) / len(df)),
        "frac_on_border": float(df["on_border"].mean()),
        "frac_H_posdef": float(df["H_posdef"].mean()),
        "median_prominence": float(df["prominence"].median()),
        "H_cond_median": float(np.median(df["H_cond"][np.isfinite(df["H_cond"])]))
                         if np.isfinite(df["H_cond"]).any() else None,
        "disp_um_all": {"median": q(d, 50), "p90": q(d, 90), "p99": q(d, 99)},
        "disp_um_pass": {"median": q(dg, 50), "p90": q(dg, 90), "p99": q(dg, 99)},
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", required=True, choices=["selfcontrol", "crossmodal"])
    ap.add_argument("--sections", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--mpp", type=float, default=2.0, help="common grid um/px")
    ap.add_argument("--tile-um", type=float, default=220.0)
    ap.add_argument("--search-um", type=float, default=80.0,
                    help="capture radius; 20.4um is too small for serial")
    ap.add_argument("--prom-gate", type=float, default=0.02)
    ap.add_argument("--cond-gate", type=float, default=20.0)
    ap.add_argument("--index", type=int, default=-1)
    args = ap.parse_args()

    out = Path(args.out); (out / "per_pair").mkdir(parents=True, exist_ok=True)
    (out / "tiles").mkdir(exist_ok=True)
    df_sec = pd.read_csv(args.sections, sep="\t")
    tile_px = int(round(args.tile_um / args.mpp))
    search_px = int(round(args.search_um / args.mpp))

    jobs = []
    if args.mode == "selfcontrol":
        pick = (df_sec[df_sec.modality == "he"].head(2).to_dict("records")
                + df_sec[df_sec.modality == "codex"].head(2).to_dict("records")
                + df_sec[df_sec.modality == "xenium"].head(1).to_dict("records"))
        for r in pick:
            jobs.append({"kind": "self", "a": r})
    else:


        rows = df_sec.to_dict("records")
        for i, a in enumerate(rows):
            for b in rows[i + 1:]:
                dz = abs(float(a["z_position_um"]) - float(b["z_position_um"]))
                if a["modality"] == b["modality"] or dz < 6 or dz > 15:
                    continue
                jobs.append({"kind": "cross", "a": a, "b": b, "dz": dz})
        jobs = jobs[:6]

    todo = jobs if args.index < 0 else ([jobs[args.index]] if args.index < len(jobs) else [])
    for k, job in enumerate(todo):
        t0 = time.time()
        rec = {"mode": args.mode, "tile_um": args.tile_um, "search_um": args.search_um,
               "grid_mpp_um": args.mpp}
        try:
            if job["kind"] == "self":
                sid = str(job["a"]["section_id"])
                rec["pair_id"] = f"self::{sid}"; rec["section_id"] = sid
                rec["modality"] = str(job["a"]["modality"])
                img, msk = load_channel(job["a"], args.mpp)
                arms = {}

                arms["identity"] = summarise(tile_grid(img, img, tile_px, search_px),
                                             args.mpp, args.prom_gate, args.cond_gate)

                shift_um = (13.0, 29.0)
                Msh = np.float32([[1, 0, shift_um[0] / args.mpp],
                                  [0, 1, shift_um[1] / args.mpp]])
                sh = cv2.warpAffine(img, Msh, (img.shape[1], img.shape[0]),
                                    flags=cv2.INTER_LINEAR, borderValue=0)
                d = tile_grid(img, sh, tile_px, search_px)
                arms["known_shift"] = summarise(d, args.mpp, args.prom_gate, args.cond_gate)
                if len(d):
                    arms["known_shift"]["truth_um"] = list(shift_um)
                    arms["known_shift"]["recovered_um"] = [
                        float(np.median(d["dx_px"]) * args.mpp),
                        float(np.median(d["dy_px"]) * args.mpp)]
                rec["arms"] = arms
            else:
                a, b = job["a"], job["b"]
                rec["pair_id"] = f'{a["section_id"]}({a["modality"]})->{b["section_id"]}({b["modality"]})'
                rec["dz_um"] = job["dz"]
                rec["modality_pair"] = f'{a["modality"]}-{b["modality"]}'
                ia, ma = load_channel(a, args.mpp)
                ib, mb = load_channel(b, args.mpp)
                h = min(ia.shape[0], ib.shape[0]); w = min(ia.shape[1], ib.shape[1])
                ia, ma, ib, mb = ia[:h, :w], ma[:h, :w], ib[:h, :w], mb[:h, :w]
                ib2, mb2, pre = rigid_prealign(ma, mb, ib)
                rec["prealign_shift_px"] = pre
                d = tile_grid(ia, ib2, tile_px, search_px)
                rec["result"] = summarise(d, args.mpp, args.prom_gate, args.cond_gate)
                if len(d):
                    d.to_parquet(out / "tiles" / f'{rec["pair_id"].replace("/", "_")}.parquet',
                                 index=False)
            rec["ok"] = True; rec["error"] = ""
        except Exception as e:
            rec["ok"] = False; rec["error"] = f"{type(e).__name__}: {e}"
            rec.setdefault("pair_id", f"job{args.index}_{k}")
        rec["elapsed_s"] = round(time.time() - t0, 1)
        fn = str(rec["pair_id"]).replace("/", "_").replace("::", "_")
        (out / "per_pair" / f"{fn}.json").write_text(json.dumps(rec, indent=2))
        print(f'[{rec["elapsed_s"]:6.1f}s] {rec["pair_id"]} ok={rec["ok"]} '
              f'{rec.get("error","")[:60]}', flush=True)
        if rec["ok"] and args.mode == "crossmodal":
            r = rec["result"]
            print(f'    tiles={r["n_tiles"]} pass={r["n_pass"]} '
                  f'({100*r["frac_pass"]:.1f}%) disp_med='
                  f'{r["disp_um_pass"]["median"]} um p90={r["disp_um_pass"]["p90"]}',
                  flush=True)


if __name__ == "__main__":
    main()
