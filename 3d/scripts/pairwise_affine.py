#!/usr/bin/env python
from __future__ import annotations

import os
import numpy as np

try:
    import cv2
except ImportError:
    cv2 = None
from scipy.optimize import minimize


PARAM_NAMES = ("theta_deg", "sx", "sy", "shear", "dx", "dy")


def affine_from_params6(theta_deg, sx, sy, shear, dx, dy, flip=False):
    th = np.deg2rad(theta_deg)
    c, s = np.cos(th), np.sin(th)
    A = np.array([[c, -s], [s, c]], dtype=np.float64) @ \
        np.array([[sx, shear], [0.0, sy]], dtype=np.float64)
    if flip:
        A = A @ np.array([[-1.0, 0.0], [0.0, 1.0]])
    M = np.zeros((2, 3), dtype=np.float64)
    M[:2, :2] = A
    M[:, 2] = [dx, dy]
    return M


def strain_decomposition(M):
    A = np.asarray(M, dtype=np.float64)[:2, :2]
    det = float(np.linalg.det(A))
    s = np.linalg.svd(A, compute_uv=False)
    return {"iso_log_scale": float(0.5 * np.log(abs(det))) if det != 0 else None,
            "iso_scale_pct": (float(100.0 * (np.exp(0.5 * np.log(abs(det))) - 1.0))
                              if det != 0 else None),
            "eps_dev": float(0.5 * (np.log(s[0]) - np.log(s[1]))),
            "eps_dev_pct": float(100.0 * 0.5 * (np.log(s[0]) - np.log(s[1]))),
            "det_A": det, "singular_values": [float(s[0]), float(s[1])],
            "note": "iso and dev are orthogonal (m=0 vs m=2). Single-edge values "
                    "are dominated by per-section handling scatter; judge on the "
                    "z trend, never on one edge."}


def invert_affine(M):
    A_inv = np.linalg.inv(np.asarray(M)[:2, :2])
    out = np.zeros((2, 3), dtype=np.float64)
    out[:2, :2] = A_inv
    out[:, 2] = -A_inv @ np.asarray(M)[:, 2]
    return out


def compose_affine(M_outer, M_inner):
    Ao, bo = np.asarray(M_outer)[:2, :2], np.asarray(M_outer)[:, 2]
    Ai, bi = np.asarray(M_inner)[:2, :2], np.asarray(M_inner)[:, 2]
    out = np.zeros((2, 3), dtype=np.float64)
    out[:2, :2] = Ao @ Ai
    out[:, 2] = Ao @ bi + bo
    return out


def recenter_params(theta_deg, sx, sy, shear, dx, dy, center, flip=False):
    M = affine_from_params6(theta_deg, sx, sy, shear, 0.0, 0.0, flip)
    c = np.asarray(center, dtype=np.float64)
    M[:, 2] = c - M[:2, :2] @ c + np.array([dx, dy], dtype=np.float64)
    return M


def masked_ncc(a, b, mask, min_overlap=5000):
    n = int(mask.sum())
    if n < min_overlap:
        return -1.0
    av = a[mask]
    bv = b[mask]
    av = av - av.mean()
    bv = bv - bv.mean()
    den = np.sqrt((av * av).sum()) * np.sqrt((bv * bv).sum())
    if den < 1e-6:
        return -1.0
    return float((av * bv).sum() / den)


def iou_masks(m1, m2):
    inter = int(np.logical_and(m1, m2).sum())
    union = int(np.logical_or(m1, m2).sum())
    return inter / max(union, 1)


def ngf_field(img, eps):
    f = img.astype(np.float32)
    gx = cv2.Sobel(f, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(f, cv2.CV_32F, 0, 1, ksize=3)
    den = np.sqrt(gx * gx + gy * gy + np.float32(eps) ** 2)
    return gx / den, gy / den


def ngf_eps_from_image(img, mask, eta=1.0):
    f = img.astype(np.float32)
    g = np.hypot(cv2.Sobel(f, cv2.CV_32F, 1, 0, ksize=3),
                 cv2.Sobel(f, cv2.CV_32F, 0, 1, ksize=3))
    v = g[mask] if mask is not None and mask.any() else g.ravel()
    return float(eta * max(np.median(v), 1e-3))


def ngf_similarity(fx, fy, mov_img, mask, eps):
    mx, my = ngf_field(mov_img, eps)
    dot = fx[mask] * mx[mask] + fy[mask] * my[mask]
    return float(np.mean(dot * dot))


class PairContext:

    def __init__(self, fix_img, fix_mask, mov_img, mov_mask, mpp_um,
                 t_star_um=80.0, min_overlap_px=5000, intensity="ncc",
                 ngf_eta=1.0):
        self.fix = fix_img
        self.fix_f = fix_img.astype(np.float32)
        self.fix_mask = fix_mask.astype(bool)
        self.mov = mov_img
        self.mov_mask_u8 = mov_mask.astype(np.uint8)
        self.mpp = float(mpp_um)
        self.h, self.w = fix_img.shape[:2]
        self.mov_h, self.mov_w = mov_img.shape[:2]
        self.min_overlap_px = int(min_overlap_px)


        m8 = self.fix_mask.astype(np.uint8)
        cnts, _ = cv2.findContours(m8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        per_px = max(sum(cv2.arcLength(c, True) for c in cnts), 1e-9)
        self.R_h_um = float(m8.sum()) * self.mpp ** 2 / (per_px * self.mpp)
        self.t_star_um = float(t_star_um)

        self.w_silh = self.R_h_um / max(self.t_star_um, 1e-9)

        ys, xs = np.nonzero(self.fix_mask)
        self.fix_center = (float(xs.mean()), float(ys.mean())) if len(xs) else \
            (self.w / 2.0, self.h / 2.0)
        ys, xs = np.nonzero(self.mov_mask_u8)
        self.mov_center = (float(xs.mean()), float(ys.mean())) if len(xs) else \
            (self.mov_w / 2.0, self.mov_h / 2.0)


        self.intensity = intensity
        self.ngf_eps_fix = ngf_eps_from_image(fix_img, self.fix_mask, ngf_eta)
        self.ngf_eps_mov = ngf_eps_from_image(mov_img, mov_mask.astype(bool), ngf_eta)
        self._fix_ngf = (ngf_field(fix_img, self.ngf_eps_fix)
                         if intensity == "ngf" else (None, None))

    def similarity(self, warped, overlap):
        if self.intensity == "ngf":
            fx, fy = self._fix_ngf
            return ngf_similarity(fx, fy, warped, overlap, self.ngf_eps_mov)
        return masked_ncc(self.fix_f, warped.astype(np.float32), overlap,
                          min_overlap=self.min_overlap_px)

    def set_precompensation(self, G):
        self.pre_G = None if G is None else np.asarray(G, dtype=np.float64)

    def warp(self, M):
        M_eff = M if getattr(self, "pre_G", None) is None else \
            compose_affine(M, self.pre_G)
        M32 = np.asarray(M_eff, dtype=np.float32)
        img = cv2.warpAffine(self.mov, M32, (self.w, self.h),
                             flags=cv2.INTER_LINEAR, borderValue=0)
        msk = cv2.warpAffine(self.mov_mask_u8, M32, (self.w, self.h),
                             flags=cv2.INTER_NEAREST, borderValue=0) > 0
        return img, msk


SCALE_LO, SCALE_HI = 0.92, 1.08
SHEAR_MAX = 0.05


BOX_MODE = "legacy"
TOL_ISO = 0.03
TOL_DEV = 0.06
TOL_SHEAR = SHEAR_MAX


_LEGACY_BOUNDS = ((-180.0, 180.0), (SCALE_LO, SCALE_HI), (SCALE_LO, SCALE_HI),
                  (-SHEAR_MAX, SHEAR_MAX), (-1e9, 1e9), (-1e9, 1e9))
DEFAULT_BOUNDS = _LEGACY_BOUNDS


BOUND_PENALTY_PER_UNIT = 50.0


def set_boxes(mode="legacy", tol_iso=None, tol_dev=None, tol_shear=None):
    global BOX_MODE, TOL_ISO, TOL_DEV, TOL_SHEAR, DEFAULT_BOUNDS
    if mode not in ("legacy", "decoupled"):
        raise ValueError(f"unknown box mode {mode!r}")
    BOX_MODE = mode
    if tol_iso is not None:
        TOL_ISO = float(tol_iso)
    if tol_dev is not None:
        TOL_DEV = float(tol_dev)
    if tol_shear is not None:
        TOL_SHEAR = float(tol_shear)
    if mode == "legacy":
        DEFAULT_BOUNDS = _LEGACY_BOUNDS
    else:
        rail = float(np.exp(TOL_ISO + TOL_DEV) + 0.01)
        DEFAULT_BOUNDS = ((-180.0, 180.0), (1.0 / rail, rail), (1.0 / rail, rail),
                          (-3.0 * TOL_SHEAR, 3.0 * TOL_SHEAR),
                          (-1e9, 1e9), (-1e9, 1e9))
    return effective_boxes()


def effective_boxes():
    if BOX_MODE == "legacy":
        return {"mode": "legacy", "scale_lo": SCALE_LO, "scale_hi": SCALE_HI,
                "shear_max": SHEAR_MAX, "search_rails": DEFAULT_BOUNDS[1],
                "precompensation": "not applicable in this mode"}
    return {"mode": "decoupled", "tol_iso": TOL_ISO, "tol_dev": TOL_DEV,
            "tol_shear": TOL_SHEAR, "search_rails": DEFAULT_BOUNDS[1]}


def _strain_of(sx, sy, shear):
    det = sx * sy
    if det <= 0:
        return float("inf"), float("inf")
    s = np.linalg.svd(np.array([[sx, shear], [0.0, sy]], dtype=np.float64),
                      compute_uv=False)
    return 0.5 * float(np.log(det)), 0.5 * float(np.log(s[0] / s[1]))


def _bound_violation(sx, sy, shear):
    if BOX_MODE == "legacy":
        return (max(0.0, SCALE_LO - sx) + max(0.0, sx - SCALE_HI)
                + max(0.0, SCALE_LO - sy) + max(0.0, sy - SCALE_HI)
                + max(0.0, abs(shear) - SHEAR_MAX))
    if sx <= 0 or sy <= 0:
        return 1e3
    iso, dev = _strain_of(sx, sy, shear)
    return (max(0.0, abs(iso) - TOL_ISO)
            + max(0.0, dev - TOL_DEV)
            + max(0.0, abs(shear) - TOL_SHEAR))


def _clamp_into_box(sx, sy, shear, n_bisect=40):
    if _bound_violation(sx, sy, shear) <= 0:
        return sx, sy, shear
    if BOX_MODE == "legacy":

        return (min(max(sx, SCALE_LO), SCALE_HI),
                min(max(sy, SCALE_LO), SCALE_HI),
                min(max(shear, -SHEAR_MAX), SHEAR_MAX))
    l1, l2 = np.log(max(sx, 1e-9)), np.log(max(sy, 1e-9))

    def at(lam):
        return np.exp(lam * l1), np.exp(lam * l2), lam * shear

    lo, hi = 0.0, 1.0
    for _ in range(n_bisect):
        mid = 0.5 * (lo + hi)
        if _bound_violation(*at(mid)) <= 0:
            lo = mid
        else:
            hi = mid
    out = at(lo)
    for _ in range(60):
        if _bound_violation(*out) <= 0:
            return out
        lo *= 0.9
        out = at(lo)
    return 1.0, 1.0, 0.0


def objective(params, ctx, flip=False, detail=False):
    theta, sx, sy, shear, dx, dy = params
    viol = _bound_violation(sx, sy, shear)
    if viol > 0:
        sx, sy, shear = _clamp_into_box(sx, sy, shear)

    M = recenter_params(theta, sx, sy, shear, dx, dy, ctx.fix_center, flip)
    warped, wmask = ctx.warp(M)
    overlap = ctx.fix_mask & wmask
    n_over = int(overlap.sum())
    iou = iou_masks(ctx.fix_mask, wmask)
    pen = BOUND_PENALTY_PER_UNIT * viol

    if n_over < ctx.min_overlap_px:


        score = -1.0 - ctx.w_silh * (1.0 - iou) - pen
        return (score, {"intensity": None, "iou": iou, "n_overlap": n_over}
                ) if detail else score

    sim = ctx.similarity(warped, overlap)
    score = sim - ctx.w_silh * (1.0 - iou) - pen
    if detail:
        return score, {"intensity": float(sim), "intensity_kind": ctx.intensity,
                       "ncc": float(sim) if ctx.intensity == "ncc" else None,
                       "iou": float(iou), "n_overlap": n_over,
                       "bound_violation": float(viol),
                       "w_silh": ctx.w_silh, "R_h_um": ctx.R_h_um,
                       "t_star_um": ctx.t_star_um, "M": M.tolist()}
    return score


def polarity_gate(fix_img, mov_img, fix_mask, mov_mask):
    fy, fx = np.nonzero(fix_mask)
    my, mx = np.nonzero(mov_mask)
    if len(fx) == 0 or len(mx) == 0:
        return {"ok": False, "reason": "empty mask"}
    M = np.float32([[1, 0, fx.mean() - mx.mean()], [0, 1, fy.mean() - my.mean()]])
    h, w = fix_img.shape[:2]
    mv = cv2.warpAffine(mov_img, M, (w, h), flags=cv2.INTER_LINEAR, borderValue=0)
    mm = cv2.warpAffine(mov_mask.astype(np.uint8), M, (w, h),
                        flags=cv2.INTER_NEAREST, borderValue=0) > 0
    ov = fix_mask & mm
    if int(ov.sum()) < 1000:
        return {"ok": False, "reason": "centroid prealign gives no overlap"}
    same = masked_ncc(fix_img.astype(np.float32), mv.astype(np.float32), ov, 1000)
    inv = masked_ncc(fix_img.astype(np.float32),
                     (255.0 - mv.astype(np.float32)), ov, 1000)
    return {"ok": bool(same >= inv), "ncc_same_polarity": float(same),
            "ncc_inverted": float(inv)}


def top_k_translations_via_fft(moving_u8, fixed_u8, top_k=5, suppress_radius=40):
    fh, fw = fixed_u8.shape[:2]
    mh, mw = moving_u8.shape[:2]
    pad_h, pad_w = fh + mh, fw + mw

    f_mask = (fixed_u8 > 8).astype(np.float32)
    m_mask = (moving_u8 > 8).astype(np.float32)
    f_mean = (fixed_u8.astype(np.float32) * f_mask).sum() / max(f_mask.sum(), 1)
    m_mean = (moving_u8.astype(np.float32) * m_mask).sum() / max(m_mask.sum(), 1)
    F = np.zeros((pad_h, pad_w), dtype=np.float32)
    Mv = np.zeros((pad_h, pad_w), dtype=np.float32)
    F[:fh, :fw] = (fixed_u8.astype(np.float32) - f_mean) * f_mask
    Mv[:mh, :mw] = (moving_u8.astype(np.float32) - m_mean) * m_mask
    corr = np.fft.irfft2(np.fft.rfft2(F) * np.conj(np.fft.rfft2(Mv)),
                         s=(pad_h, pad_w)).copy()

    peaks = []
    for _ in range(top_k):
        idx = int(np.argmax(corr))
        val = float(corr.flat[idx])
        if not np.isfinite(val):
            break
        py, px = idx // pad_w, idx % pad_w
        peaks.append((int(px - pad_w if px > pad_w // 2 else px),
                      int(py - pad_h if py > pad_h // 2 else py), val))
        corr[max(0, py - suppress_radius):py + suppress_radius,
             max(0, px - suppress_radius):px + suppress_radius] = -np.inf
    return peaks


def _warp_moving_about_center(ctx, theta, sx, sy, shear, flip, what="both"):
    M = affine_from_params6(theta, sx, sy, shear, 0.0, 0.0, flip)
    c = np.asarray(ctx.mov_center)
    M[:, 2] = c - M[:2, :2] @ c
    M32 = M.astype(np.float32)
    img = msk = None
    if what in ("both", "img"):
        img = cv2.warpAffine(ctx.mov, M32, (ctx.mov_w, ctx.mov_h),
                             flags=cv2.INTER_LINEAR, borderValue=0)
    if what in ("both", "mask"):
        msk = cv2.warpAffine(ctx.mov_mask_u8 * 255, M32, (ctx.mov_w, ctx.mov_h),
                             flags=cv2.INTER_NEAREST, borderValue=0)
    return img, msk


def _peak_to_params(ctx, theta, sx, sy, shear, flip, px, py):
    A = affine_from_params6(theta, sx, sy, shear, 0.0, 0.0, flip)[:2, :2]
    cm = np.asarray(ctx.mov_center, dtype=np.float64)
    cf = np.asarray(ctx.fix_center, dtype=np.float64)


    t_total = cm - A @ cm + np.array([px, py], dtype=np.float64)
    d = t_total - (cf - A @ cf)
    return (float(theta), float(sx), float(sy), float(shear), float(d[0]), float(d[1]))


def _shrink(a, f):
    if f <= 1:
        return a
    return cv2.resize(a, (max(1, a.shape[1] // f), max(1, a.shape[0] // f)),
                      interpolation=cv2.INTER_AREA)


def mask_seeds(ctx, rot_grid, scale_grid, flip=False, top_k=3, seed_ds=4):
    fixed_m = _shrink(ctx.fix_mask.astype(np.uint8) * 255, seed_ds)
    out = []
    for theta in rot_grid:
        for sx, sy in scale_grid:
            _, mm = _warp_moving_about_center(ctx, theta, sx, sy, 0.0, flip, "mask")
            for px, py, _ in top_k_translations_via_fft(_shrink(mm, seed_ds),
                                                        fixed_m, top_k=top_k):
                p = _peak_to_params(ctx, theta, sx, sy, 0.0, flip,
                                    px * seed_ds, py * seed_ds)
                out.append((p, objective(p, ctx, flip)))
    out.sort(key=lambda z: -z[1])
    return out


def intensity_seeds(ctx, rot_grid, flip=False, top_k=3, seed_ds=4):
    fixed_i = _shrink(ctx.fix, seed_ds)
    out = []
    for theta in rot_grid:
        img, _ = _warp_moving_about_center(ctx, theta, 1.0, 1.0, 0.0, flip, "img")
        for px, py, _ in top_k_translations_via_fft(_shrink(img, seed_ds),
                                                    fixed_i, top_k=top_k):
            p = _peak_to_params(ctx, theta, 1.0, 1.0, 0.0, flip,
                                px * seed_ds, py * seed_ds)
            out.append((p, objective(p, ctx, flip)))
    out.sort(key=lambda z: -z[1])
    return out


def powell(seed, ctx, flip=False, maxiter=120):
    res = minimize(lambda p: -objective(p, ctx, flip), np.asarray(seed, float),
                   method="Powell",
                   options={"xtol": 1e-3, "ftol": 1e-4, "maxiter": maxiter,
                            "disp": False})
    return float(-res.fun), np.asarray(res.x, float)


def basin_hop(p0, ctx, flip=False, n_epochs=40, patience=12, seed=42, log=None):
    rng = np.random.default_rng(seed)
    sigma = np.array([2.0, 0.01, 0.01, 0.004, ctx.w * 0.03, ctx.h * 0.03])
    best_p = np.asarray(p0, float).copy()
    best = objective(best_p, ctx, flip)
    no_imp, mult, hist = 0, 1.0, []
    for epoch in range(1, n_epochs + 1):
        s, p = powell(best_p + rng.normal(0.0, sigma * mult), ctx, flip, maxiter=60)
        improved = s > best + 1e-5
        if improved:
            best, best_p, no_imp, mult = s, p.copy(), 0, 1.0
        else:
            no_imp += 1
            if no_imp % 4 == 0 and mult < 4.0:
                mult = min(mult * 1.5, 4.0)
        hist.append({"epoch": epoch, "score": float(s), "best": float(best),
                     "improved": bool(improved)})
        if log and (improved or epoch % 10 == 0):
            log(f"    basin epoch {epoch:3d}: best={best:.4f} "
                f"theta={best_p[0]:.2f} sx={best_p[1]:.4f} sy={best_p[2]:.4f}")
        if no_imp >= patience:
            break
    return best_p, float(best), hist


def fine_climb(p0, ctx, flip=False, max_iter=200):
    px_per_um = 1.0 / ctx.mpp
    base = np.array([0.05, 5e-4, 5e-4, 2e-4, 2.0 * px_per_um, 2.0 * px_per_um])
    floor = np.array([0.005, 5e-5, 5e-5, 2e-5, 0.1 * px_per_um, 0.1 * px_per_um])
    steps = base.copy()
    best_p = np.asarray(p0, float).copy()
    best = objective(best_p, ctx, flip)
    moves = []
    for ax in range(6):
        for sgn in (1.0, -1.0):
            v = np.zeros(6)
            v[ax] = sgn
            moves.append(v)
    stall = 0
    for _ in range(max_iter):
        cand, cand_s = None, best
        for mv in moves:
            v = objective(best_p + mv * steps, ctx, flip)
            if v > cand_s + 1e-6:
                cand, cand_s = mv, v
        if cand is not None:
            best_p = best_p + cand * steps
            best, stall = cand_s, 0
        else:
            stall += 1
            if stall >= 6:
                ns = np.maximum(steps * 0.5, floor)
                if np.all(ns == steps):
                    break
                steps, stall = ns, 0
    return best_p, float(best)


def _fork_map(fn, args_list, max_workers):
    import multiprocessing as _mp
    ctx = _mp.get_context("fork")


    prev_threads = cv2.getNumThreads()
    cv2.setNumThreads(1)
    out = [None] * len(args_list)
    pending = list(enumerate(args_list))
    running = []
    while pending or running:
        while pending and len(running) < max_workers:
            i, a = pending.pop(0)
            r, w = ctx.Pipe(duplex=False)

            def _child(a=a, w=w):
                try:
                    w.send((True, fn(a)))
                except BaseException as e:
                    w.send((False, repr(e)))
                finally:
                    w.close()
            pr = ctx.Process(target=_child)
            pr.start()
            w.close()
            running.append((i, pr, r))
        done = []
        for j, (i, pr, r) in enumerate(running):
            if r.poll(0.2):
                ok, val = r.recv()
                pr.join()
                if not ok:
                    raise RuntimeError(f"forked worker failed: {val}")
                out[i] = val
                done.append(j)
        for j in reversed(done):
            running.pop(j)
    cv2.setNumThreads(prev_threads)
    return out


def _powell_seed(args):
    p, ctx, flip = args
    return powell(p, ctx, flip, maxiter=120)


def _solve_branch(args):
    (ctx, flip, rot_grid, scale_grid, n_powell_seeds, n_epochs, patience, seed,
     seed_ds, workers) = args
    lines = []
    log = lines.append
    stages = {}
    ms = mask_seeds(ctx, rot_grid, scale_grid, flip=flip, seed_ds=seed_ds)
    isd = intensity_seeds(ctx, rot_grid, flip=flip, seed_ds=seed_ds)
    seeds = (ms[:n_powell_seeds] + isd[:n_powell_seeds])
    seeds.sort(key=lambda z: -z[1])
    stages[f"seeds_flip{int(flip)}"] = {
        "n_mask": len(ms), "n_intensity": len(isd),
        "best_mask_score": ms[0][1] if ms else None,
        "best_intensity_score": isd[0][1] if isd else None,
        "top": [{"params": list(map(float, p)), "score": float(s)}
                for p, s in seeds[:5]]}
    log(f"  [flip={flip}] seeds: mask best {ms[0][1]:.4f}  "
        f"intensity best {isd[0][1]:.4f}")

    jobs = [(p, ctx, flip) for p, _ in seeds[:n_powell_seeds]]
    if workers > 1 and len(jobs) > 1:
        pw = _fork_map(_powell_seed, jobs, min(workers, len(jobs)))
    else:
        pw = [_powell_seed(j) for j in jobs]
    pw.sort(key=lambda z: -z[0])
    log(f"  [flip={flip}] powell best {pw[0][0]:.4f}")

    bp, bs, hist = basin_hop(pw[0][1], ctx, flip, n_epochs=n_epochs,
                             patience=patience, seed=seed, log=log)
    fp, fs = fine_climb(bp, ctx, flip)
    log(f"  [flip={flip}] basin {bs:.4f} -> fine {fs:.4f}")
    stages[f"solve_flip{int(flip)}"] = {
        "powell_best": float(pw[0][0]), "basin_best": float(bs),
        "fine_best": float(fs), "basin_epochs": len(hist)}
    return fp, fs, stages, lines


def solve_pair(ctx, rot_grid=None, scale_grid=None, try_flip=False,
               n_powell_seeds=6, n_epochs=40, patience=12, seed=42,
               seed_ds=4, log=print):
    if rot_grid is None:


        rot_grid = list(np.arange(-180.0, 180.0, 10.0))
    if scale_grid is None:
        scale_grid = [(1.0, 1.0), (0.96, 1.0), (1.0, 0.96), (0.96, 0.96),
                      (1.04, 1.0), (1.0, 1.04)]

    report = {"stages": {}, "R_h_um": ctx.R_h_um, "t_star_um": ctx.t_star_um,
              "w_silh": ctx.w_silh}
    flips = (False, True) if try_flip else (False,)


    workers = int(os.environ.get("PAIRWISE_WORKERS", "1") or "1")
    branch_args = [(ctx, flip, rot_grid, scale_grid, n_powell_seeds, n_epochs,
                    patience, seed, seed_ds, max(1, workers // len(flips)))
                   for flip in flips]
    if workers > 1 and len(flips) > 1:
        outs = _fork_map(_solve_branch, branch_args, len(flips))
    else:
        outs = [_solve_branch(a) for a in branch_args]
    best_overall = None
    for flip, (fp, fs, stages, lines) in zip(flips, outs):
        for ln in lines:
            log(ln)
        report["stages"].update(stages)
        if best_overall is None or fs > best_overall[1]:
            best_overall = (fp, fs, flip)

    params, score, flip = best_overall
    _, det = objective(params, ctx, flip, detail=True)
    report["best"] = {"params": dict(zip(PARAM_NAMES, map(float, params))),
                      "flip": bool(flip), "score": float(score), **det}
    return params, flip, report


def refine_per_cc(ctx, M_global, flip=False, min_area_px=5000, margin_px=64,
                  max_rot_deg=1.0, scale_band=0.01, log=print):
    m8 = ctx.fix_mask.astype(np.uint8)
    n, lab, stats, _ = cv2.connectedComponentsWithStats(m8, connectivity=8)
    out = []
    for i in range(1, n):
        if stats[i, cv2.CC_STAT_AREA] < min_area_px:
            continue
        x, y, w, h = (stats[i, cv2.CC_STAT_LEFT], stats[i, cv2.CC_STAT_TOP],
                      stats[i, cv2.CC_STAT_WIDTH], stats[i, cv2.CC_STAT_HEIGHT])
        x0, y0 = max(0, x - margin_px), max(0, y - margin_px)
        x1, y1 = min(ctx.w, x + w + margin_px), min(ctx.h, y + h + margin_px)
        sub_fix = ctx.fix[y0:y1, x0:x1]
        sub_msk = (lab[y0:y1, x0:x1] == i)
        warped, wmask = ctx.warp(M_global)
        sub_ctx = PairContext(sub_fix, sub_msk, warped[y0:y1, x0:x1],
                              wmask[y0:y1, x0:x1], ctx.mpp, ctx.t_star_um,
                              min_overlap_px=max(500, min_area_px // 10))
        p0 = (0.0, 1.0, 1.0, 0.0, 0.0, 0.0)
        bnds = [(-max_rot_deg, max_rot_deg),
                (1 - scale_band, 1 + scale_band), (1 - scale_band, 1 + scale_band),
                (-0.01, 0.01), (-margin_px, margin_px), (-margin_px, margin_px)]
        r = minimize(lambda p: -objective(p, sub_ctx), np.array(p0),
                     method="Powell", bounds=bnds,
                     options={"xtol": 1e-3, "ftol": 1e-4, "maxiter": 80})
        resid = recenter_params(*r.x, center=sub_ctx.fix_center)

        shift = np.array([[1.0, 0.0, x0], [0.0, 1.0, y0]])
        M_local = compose_affine(compose_affine(shift, resid),
                                 invert_affine(shift))
        out.append({"cc": int(i), "area_px": int(stats[i, cv2.CC_STAT_AREA]),
                    "score_before": float(objective(p0, sub_ctx)),
                    "score_after": float(-r.fun),
                    "M_local": compose_affine(M_local, M_global).tolist()})
        log(f"    cc{i}: {out[-1]['score_before']:.4f} -> {out[-1]['score_after']:.4f}")
    return out


def intensity_profile(ctx, params, flip=False, max_um=800.0, n=17):
    p = np.asarray(params, float).copy()
    offs = np.linspace(0.0, max_um, n)
    px = offs / ctx.mpp
    prof = {"offsets_um": offs.tolist()}
    for axis, idx in (("x", 4), ("y", 5)):
        vals = []
        for d in px:
            q = p.copy()
            q[idx] = p[idx] + d
            s, det = objective(q, ctx, flip, detail=True)
            vals.append(det.get("intensity"))
        prof[f"similarity_{axis}"] = [None if v is None else float(v) for v in vals]
    v = np.array([[a if a is not None else np.nan for a in prof["similarity_x"]],
                  [a if a is not None else np.nan for a in prof["similarity_y"]]])
    m = np.nanmean(v, axis=0)
    prof["similarity_mean"] = [float(z) for z in m]
    if np.isfinite(m).sum() >= 3:

        d1 = offs[1]
        kappa = float(2.0 * (m[0] - m[1]) / max(d1 ** 2, 1e-9))
        prof["kappa_per_um2"] = kappa
        prof["t_star_self_consistent_um"] = (float(np.sqrt(0.6366 / kappa))
                                             if kappa > 0 else None)
        far = float(np.nanmin(m))
        half = far + 0.5 * (m[0] - far)
        below = np.where(m <= half)[0]
        prof["r_half_um"] = float(offs[below[0]]) if len(below) else None
        dm = np.diff(m)
        up = np.where(dm > 0)[0]
        prof["basin_um"] = float(offs[up[0]]) if len(up) else float(offs[-1])
        prof["basin_is_lower_bound"] = bool(len(up) == 0)
    return prof


compose_local_M = compose_affine
