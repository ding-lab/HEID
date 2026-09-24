#!/usr/bin/env python3

import argparse
import json
import os
import pickle
import struct
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw
from scipy.ndimage import binary_closing, binary_dilation, binary_fill_holes, shift as ndshift
from scipy.spatial import cKDTree

HERE = Path(__file__).resolve().parent


def _load(name, p):
    import importlib.util
    spec = importlib.util.spec_from_file_location(name, p)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


BM = _load("bm", HERE / "build_meshes.py")

RES = 10.0
UM = 8.0
R_TENSOR_UM = 100.0
LIN_SEED = 2.0
MIN_NB_SEED = 10
SEED_NMS_UM = 60.0
STEP_UM = 30.0
TUBE_R_UM = 60.0
TUBE_R_MAX_UM = 200.0
FIT_R_UM = 200.0
LOOK_UM = 100.0
R_MIN_UM = 250.0
DKAPPA_MAX = 0.5 / R_MIN_UM
K_MIN_AHEAD = 3
WEAK_FRAC = 0.3
PROBE_STEPS = 5
PROBE_STEPS_GAP = 7
MAX_STEPS = 400
GAP_MIN_DZ_UM = 7.5
L_MIN_UM = 300.0
DENS_LINE, DENS_CURVE = 6.0, 8.0
CONT_LINE, CONT_CURVE = 0.80, 0.80
BODY_MIN_CELLS = 600
BODY_MAX_ELONG = 1.8
BODY_MIN_WIDTH_UM = 400.0
S_MAX_TURN_DEG = 90.0
FIT_RMS_UM = 30.0
LINK_UM = 45.0
CORE_MIN = 5
WEAK_FRAC_SPLIT = 0.15
WEAK_BIN_UM = 30.0
SPLIT_GAP_UM = 100.0
CONTRAST_MIN = 3.0
NMS_OVERLAP = 0.3
MASK_R_UM = 30.0
TUBE_BODY_UM = 35.0
MIN_WAIST_UM = 25.0
LINE_COLOUR = "#ffd400"
REGION_COLOUR = "#00B0F0"


def _unit(v):
    n = np.linalg.norm(v)
    return v / n if n > 1e-9 else v


def _rotate_towards(d, dn, max_ang):
    c = float(np.clip(np.dot(d, dn), -1.0, 1.0))
    ang = float(np.arccos(c))
    if ang <= max_ang:
        return dn, ang
    perp = _unit(dn - c * d)
    return _unit(np.cos(max_ang) * d + np.sin(max_ang) * perp), max_ang


def _plane_frame(P):
    c = P.mean(0)
    Q = P - c
    _w, v = np.linalg.eigh(Q.T @ Q / max(1, len(P)))
    return c, v[:, 2], v[:, 1], v[:, 0]


def _rms_to_curve(P, curve):
    t = cKDTree(curve)
    d, _ = t.query(P)
    return float(np.sqrt(np.mean(d ** 2)))


def fit_models(P, tol=None, family=None):
    c, e1, e2, e3 = _plane_frame(P)
    u = (P - c) @ e1
    v = (P - c) @ e2
    out = []

    ts = np.arange(u.min(), u.max() + 1e-6, 10.0)
    cl = c + np.outer(ts, e1)
    out.append({"model": "line", "curve": cl, "kappa": 0.0, "rms": _rms_to_curve(P, cl)})

    if len(P) >= 4 and u.max() - u.min() > 50:
        A = np.column_stack([u ** 2, u, np.ones_like(u)])
        (pa, pb, pc), *_ = np.linalg.lstsq(A, v, rcond=None)
        kmax = 2.0 * abs(pa)
        if kmax <= 1.0 / R_MIN_UM:
            us = np.arange(u.min(), u.max() + 1e-6, 5.0)
            vs = pa * us ** 2 + pb * us + pc
            pts = c + np.outer(us, e1) + np.outer(vs, e2)
            seg = np.linalg.norm(np.diff(pts, axis=0), axis=1)
            s_at = np.concatenate([[0.0], np.cumsum(seg)])
            ss = np.arange(0.0, s_at[-1] + 1e-6, 10.0)
            pts = np.column_stack([np.interp(ss, s_at, pts[:, k]) for k in range(3)])
            out.append({"model": "parabola", "curve": pts, "kappa": kmax, "rms": _rms_to_curve(P, pts)})


    if len(P) >= 6 and u.max() - u.min() > 150:
        A = np.column_stack([u ** 3, u ** 2, u, np.ones_like(u)])
        (a3, a2, a1, a0), *_ = np.linalg.lstsq(A, v, rcond=None)
        us = np.arange(u.min(), u.max() + 1e-6, 5.0)
        vs = a3 * us ** 3 + a2 * us ** 2 + a1 * us + a0
        d1 = 3 * a3 * us ** 2 + 2 * a2 * us + a1
        d2 = 6 * a3 * us + 2 * a2
        kap = np.abs(d2) / (1 + d1 ** 2) ** 1.5
        seg = np.linalg.norm(np.diff(np.column_stack([us, vs]), axis=0), axis=1)
        turn = float(np.degrees(np.sum(np.abs(np.diff(np.arctan(d1))))))
        if kap.max() <= 1.0 / R_MIN_UM and turn <= S_MAX_TURN_DEG:
            pts = c + np.outer(us, e1) + np.outer(vs, e2)
            s_at = np.concatenate([[0.0], np.cumsum(seg)])
            ss = np.arange(0.0, s_at[-1] + 1e-6, 10.0)
            pts = np.column_stack([np.interp(ss, s_at, pts[:, k]) for k in range(3)])
            out.append({"model": "scurve", "curve": pts, "kappa": float(kap.max()), "rms": _rms_to_curve(P, pts)})

    if len(P) >= 4:
        A = np.column_stack([2 * u, 2 * v, np.ones_like(u)])
        b = u ** 2 + v ** 2
        try:
            (cx, cy, k), *_ = np.linalg.lstsq(A, b, rcond=None)
            r = float(np.sqrt(max(1e-9, k + cx ** 2 + cy ** 2)))
        except np.linalg.LinAlgError:
            r = 0.0
        if r >= R_MIN_UM:
            ang = np.arctan2(v - cy, u - cx)
            a0 = float(np.arctan2(np.sin(ang).mean(), np.cos(ang).mean()))
            rel = (ang - a0 + np.pi) % (2 * np.pi) - np.pi
            if rel.max() - rel.min() < np.pi:
                da = 10.0 / r
                angs = a0 + np.arange(rel.min(), rel.max() + 1e-9, da)
                pts = c + np.outer(cx + r * np.cos(angs), e1) + np.outer(cy + r * np.sin(angs), e2)
                out.append({"model": "arc", "curve": pts, "kappa": 1.0 / r, "rms": _rms_to_curve(P, pts)})
    if family is not None:
        return next((o for o in out if o["model"] == family), None)
    ok = [o for o in out if o["rms"] <= (FIT_RMS_UM if tol is None else tol)]
    if not ok:
        return None

    ok.sort(key=lambda o: o["rms"])
    best = ok[0]


    for fam in ("line", "parabola", "arc"):
        cand = next((o for o in ok if o["model"] == fam), None)
        if cand is not None and cand["rms"] <= 1.3 * best["rms"]:
            best = cand; break
    return best


G = {}


def _tensor_chunk(idx):
    P, tree = G["P"], G["tree"]
    lin = np.zeros(len(idx), np.float32)
    nnb = np.zeros(len(idx), np.int32)
    dirs = np.zeros((len(idx), 3), np.float32)
    nbs = tree.query_ball_point(P[idx], R_TENSOR_UM)
    for k, nb in enumerate(nbs):
        nnb[k] = len(nb)
        if len(nb) < 4:
            continue
        Q = P[nb] - P[nb].mean(0)
        w, v = np.linalg.eigh(Q.T @ Q / len(nb))
        l1, l2, l3 = w[2], w[1], w[0]
        lin[k] = l1 / max(1e-6, l2 + l3)
        dirs[k] = v[:, 2]
    return idx, lin, nnb, dirs


def _walk(p, d, gaps, force_enter=False):
    P, tree = G["P"], G["tree"]
    in_body = G.get("in_body")
    theta_max = STEP_UM / R_MIN_UM
    pts, kappa_prev, weak = [p.copy()], 0.0, 0
    ref, rad = [], TUBE_R_UM
    out_len, entered = (L_MIN_UM if force_enter else 0.0), False
    for _ in range(MAX_STEPS):
        centre = p + d * (LOOK_UM / 2)
        nb = tree.query_ball_point(centre, LOOK_UM / 2 + TUBE_R_MAX_UM)
        if nb:
            Q = P[nb] - p
            t = Q @ d
            perp = np.linalg.norm(Q - np.outer(t, d), axis=1)
            wide = (t > 0) & (t <= LOOK_UM) & (perp <= TUBE_R_MAX_UM)


            if wide.sum() >= K_MIN_AHEAD:
                rad = float(np.clip(1.5 * np.median(perp[wide]), TUBE_R_UM, TUBE_R_MAX_UM))
            sel = wide & (perp <= rad) & ~G["claimed0"][np.asarray(nb)]
            ahead = np.asarray(nb)[sel]
            if in_body is not None and len(ahead):


                fb = float(in_body[ahead].mean())
                if fb > 0.5 and out_len < L_MIN_UM and not entered:
                    ahead = ahead[~in_body[ahead]]
                elif fb > 0.5:
                    entered = True
                else:
                    out_len += STEP_UM
        else:
            ahead = np.zeros(0, int)
        need = K_MIN_AHEAD if not ref else max(K_MIN_AHEAD, WEAK_FRAC * float(np.median(ref[-5:])))
        if len(ahead) < need:
            z0, z1 = sorted([p[2], p[2] + d[2] * LOOK_UM])
            in_gap = any(g0 - 2.5 <= z1 and z0 <= g1 + 2.5 for g0, g1 in gaps)
            if weak < (PROBE_STEPS_GAP if in_gap else PROBE_STEPS):
                p = p + d * STEP_UM
                pts.append(p.copy()); weak += 1
                continue
            break
        ref.append(len(ahead)); weak = 0
        c = P[ahead].mean(0)
        dn = _unit(c - p)

        hi = min(theta_max, (kappa_prev + DKAPPA_MAX) * STEP_UM)
        dn, ang = _rotate_towards(d, dn, hi)
        kappa_prev = ang / STEP_UM

        off = (c - p) - np.dot(c - p, d) * d
        p = p + off + dn * STEP_UM
        d = dn
        pts.append(p.copy())
    while weak > 0 and len(pts) > 1:
        pts.pop(); weak -= 1
    return pts, rad, entered


def _track_seed(i):
    P, D, gaps = G["P"], G["D"], G["gaps"]
    fwd, r1, e1 = _walk(P[i].copy(), D[i].copy(), gaps)
    bwd, r2, e2 = _walk(P[i].copy(), -D[i].copy(), gaps)
    poly = np.array(bwd[::-1] + fwd[1:])
    return i, (poly, max(r1, r2), e1 or e2)


def _score(item, gaps, depth=0):
    P, tree = G["P"], G["tree"]
    poly, rad = item[0], item[1]
    if len(poly) < 3:
        return []
    seg = np.linalg.norm(np.diff(poly, axis=0), axis=1)
    if float(seg.sum()) < L_MIN_UM:
        return []
    best = fit_models(poly)
    if best is None:
        if depth >= 3 or len(poly) < 6:
            return []

        c, e1, _e2, _e3 = _plane_frame(poly)
        Q = poly - c
        dev = np.linalg.norm(Q - np.outer(Q @ e1, e1), axis=1)
        k = int(np.clip(np.argmax(dev), 2, len(poly) - 3))
        return _score((poly[:k + 1], rad), gaps, depth + 1) + _score((poly[k:], rad), gaps, depth + 1)


    curve = best["curve"]
    tube = float(np.clip(rad, TUBE_R_UM, TUBE_R_MAX_UM))
    for _it in range(5):
        nbs = tree.query_ball_point(curve, FIT_R_UM)
        idx = np.unique(np.concatenate([np.asarray(nb, int) for nb in nbs]) if any(len(nb) for nb in nbs) else np.zeros(0, int))
        if len(idx) < 8:
            break
        refit = fit_models(P[idx], family=best["model"])
        if refit is None:
            break
        refit["rms"] = best["rms"]
        moved = _rms_to_curve(curve, refit["curve"])
        best, curve = refit, refit["curve"]
        dperp, _ = cKDTree(curve).query(P[idx])
        tube = float(np.clip(1.5 * np.median(dperp), TUBE_R_UM, TUBE_R_MAX_UM))
        if moved < 5.0:
            break
    L = float(np.linalg.norm(np.diff(curve, axis=0), axis=1).sum())
    if L < L_MIN_UM:
        return []
    return _finish(curve, best, tube, gaps, split=True)


def _components(idx):
    P = G["P"]
    if len(idx) == 0:
        return []
    t = cKDTree(P[idx])
    pairs = np.array(list(t.query_pairs(LINK_UM)), int).reshape(-1, 2)
    deg = np.bincount(pairs.ravel(), minlength=len(idx)) if len(pairs) else np.zeros(len(idx), int)
    core = deg >= CORE_MIN
    parent = np.arange(len(idx))
    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]; x = parent[x]
        return x
    for a, b in pairs:
        if not (core[a] or core[b]):
            continue
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[ra] = rb
    roots = np.array([find(k) for k in range(len(idx))])
    comps = {}
    for k, r in enumerate(roots):
        comps.setdefault(r, []).append(idx[k])
    return sorted((np.array(c) for c in comps.values()), key=len, reverse=True)


def _rigid(curve, c0, ang_z, ang_t, shift):
    Q = curve - c0
    cz, sz = np.cos(ang_z), np.sin(ang_z)
    Q = Q @ np.array([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]]).T
    ct, st = np.cos(ang_t), np.sin(ang_t)
    Q = Q @ np.array([[ct, 0, st], [0, 1, 0], [-st, 0, ct]]).T
    return Q + c0 + shift


def _regression_score(curve, cells_xyz, sigma):
    d, _ = cKDTree(curve).query(cells_xyz)
    return float(np.exp(-(d / sigma) ** 2).sum())


def _optimize(curve, cells_xyz, tube, rng):
    sigma = max(20.0, 0.5 * tube)
    c0 = cells_xyz.mean(0)
    best, bs = curve, _regression_score(curve, cells_xyz, sigma)
    step_a, step_s = np.radians(6.0), 0.4 * tube
    for rnd in range(4):
        improved = False
        for _ in range(24):
            cand = _rigid(best, c0, rng.normal(0, step_a), rng.normal(0, step_a * 0.5), rng.normal(0, step_s, 3) * [1, 1, 0.3])
            sc = _regression_score(cand, cells_xyz, sigma)
            if sc > bs * 1.002:
                best, bs, improved = cand, sc, True
        if not improved:
            step_a *= 0.5; step_s *= 0.5
    return best, bs


def _assign(curve, tube):
    P, tree = G["P"], G["tree"]
    ss = np.arange(len(curve)) * 10.0
    nbs = tree.query_ball_point(curve, tube)
    cells, cell_s = {}, {}
    for k, nb in enumerate(nbs):
        for j in nb:
            dd = float(np.linalg.norm(P[j] - curve[k]))
            if j not in cells or dd < cells[j]:
                cells[j] = dd; cell_s[j] = ss[k]
    return cells, cell_s, ss


def _finish(curve, best, tube, gaps, split):
    P, tree = G["P"], G["tree"]
    cells, cell_s, ss = _assign(curve, tube)
    if not cells:
        return []
    if split:


        idx_all = np.fromiter(cells.keys(), int, len(cells))
        comps = _components(idx_all)
        if not comps:
            return []
        out = []
        for k, comp in enumerate(comps):
            if len(comp) < 8:
                continue
            ext = P[comp].max(0) - P[comp].min(0)
            if float(np.sqrt((ext ** 2).sum())) < L_MIN_UM:
                continue
            seg = fit_models(P[comp], tol=1e9) if k > 0 else best
            fam = seg["model"] if seg else best["model"]
            refit = fit_models(P[comp], family=fam)
            if refit is None:
                continue
            refit["rms"] = best["rms"]
            cur, _sc = _optimize(refit["curve"], P[comp], tube, np.random.default_rng(len(comp)))
            dperp, _ = cKDTree(cur).query(P[comp])
            t2 = float(np.clip(1.5 * np.median(dperp), TUBE_R_UM, TUBE_R_MAX_UM))
            out += _finish(cur, refit, t2, gaps, split=False)
        return out

    idx_all = np.fromiter(cells.keys(), int, len(cells))
    comps = _components(idx_all)
    if not comps:
        return []
    keep = set(comps[0].tolist())
    cells = {j: d for j, d in cells.items() if j in keep}
    cell_s = {j: v for j, v in cell_s.items() if j in keep}
    L = float(ss[-1])


    if not getattr(_finish, "_in_weak", False):
        nbw = max(1, int(np.ceil(L / WEAK_BIN_UM)))
        cw_ = np.zeros(nbw, int)
        for j, sj in cell_s.items():
            cw_[min(nbw - 1, int(sj // WEAK_BIN_UM))] += 1
        bz = np.interp(np.arange(nbw) * WEAK_BIN_UM + WEAK_BIN_UM / 2, ss, curve[:, 2])
        gapb = np.array([any(g0 - 2.5 <= z <= g1 + 2.5 for g0, g1 in gaps) for z in bz])
        med = float(np.median(cw_[~gapb])) if (~gapb).any() else 0.0
        weak = (cw_ < WEAK_FRAC_SPLIT * med) & ~gapb
        cuts, k = [], 0
        while k < nbw:
            if weak[k]:
                e = k
                while e < nbw and weak[e]:
                    e += 1
                if e - k >= 2 and k > 0 and e < nbw:
                    cuts.append((k * WEAK_BIN_UM, e * WEAK_BIN_UM))
                k = e
            else:
                k += 1
        if cuts and med >= 4:
            bounds = [0.0] + [c for cut in cuts for c in cut] + [L + 1.0]
            out = []
            for a, b in zip(bounds[0::2], bounds[1::2]):
                idx = np.fromiter((j for j, sj in cell_s.items() if a <= sj < b), int)
                if len(idx) < 30:
                    continue
                ext = P[idx].max(0) - P[idx].min(0)
                if float(np.sqrt((ext ** 2).sum())) < L_MIN_UM:
                    continue
                fam = fit_models(P[idx], tol=1e9)
                if fam is None:
                    continue
                cur, _sc = _optimize(fam["curve"], P[idx], tube, np.random.default_rng(len(idx)))
                dperp, _ = cKDTree(cur).query(P[idx])
                t2 = float(np.clip(1.5 * np.median(dperp), TUBE_R_UM, TUBE_R_MAX_UM))
                fam["rms"] = best["rms"]
                _finish._in_weak = True
                try:
                    out += _finish(cur, fam, t2, gaps, split=False)
                finally:
                    _finish._in_weak = False
            return out
    B = 50.0
    nb_bins = max(1, int(np.ceil(L / B)))
    counts = np.zeros(nb_bins, int)
    for j, sj in cell_s.items():
        counts[min(nb_bins - 1, int(sj // B))] += 1
    bin_z = np.interp(np.arange(nb_bins) * B + B / 2, ss, curve[:, 2])
    in_gap = np.array([any(g0 <= z <= g1 for g0, g1 in gaps) for z in bin_z])
    filled = (counts > 0) | in_gap
    if False:

        runs, k, need = [], 0, int(round(SPLIT_GAP_UM / B))
        while k < nb_bins:
            if not filled[k]:
                k += 1; continue
            a = k
            while k < nb_bins:
                if filled[k]:
                    k += 1; continue
                e = k
                while e < nb_bins and not filled[e]:
                    e += 1
                if e - k >= need:
                    break
                k = e
            runs.append((a, k))
        if len(runs) > 1 or (runs and (runs[0][0] > 0 or runs[0][1] < nb_bins)):
            out = []
            for a, b in runs:
                s0, s1 = a * B, min(L, b * B)
                if s1 - s0 < L_MIN_UM:
                    continue
                idx = np.fromiter((j for j, sj in cell_s.items() if s0 <= sj <= s1), int)
                if len(idx) < 8:
                    continue
                seg = curve[(ss >= s0) & (ss <= s1)]
                fam = fit_models(seg) if len(seg) >= 3 else None
                if fam is None:
                    continue
                refit = fit_models(P[idx], family=fam["model"])
                if refit is None:
                    continue
                refit["rms"] = fam["rms"]
                dperp, _ = cKDTree(refit["curve"]).query(P[idx])
                t2 = float(np.clip(1.5 * np.median(dperp), TUBE_R_UM, TUBE_R_MAX_UM))
                out += _finish(refit["curve"], refit, t2, gaps, split=False)
            return out
    if L < L_MIN_UM:
        return []
    n = len(cells)
    outer = set()
    for nb in tree.query_ball_point(curve, 2.0 * tube):
        outer.update(nb)
    n_ann = max(0, len(outer) - n)
    contrast = (n / 1.0) / max(0.5, n_ann / 3.0)
    if contrast < CONTRAST_MIN:
        return []
    ask = ~in_gap
    cont = float((counts[ask] > 0).mean()) if ask.any() else 0.0
    dens = n / (L / 100.0)
    rms = float(np.sqrt(np.mean(np.square(list(cells.values())))))
    model = best["model"]
    ok = (dens >= DENS_LINE and cont >= CONT_LINE) if model == "line" else (dens >= DENS_CURVE and cont >= CONT_CURVE)
    if not ok:
        return []
    kappa = best["kappa"]
    return [{"cells": np.fromiter(cells.keys(), int, len(cells)), "length_um": L, "n_cells": n,
             "density_per_100um": dens, "continuity": cont, "rms_um": rms, "model": model,
             "curvature_per_um": kappa, "radius_um": (1.0 / kappa if kappa > 0 else None),
             "fit_rms_um": best["rms"], "tube_um": tube, "contrast": contrast, "ok": True, "poly": curve[::2]}]


def _body_of(comp):
    P = G["P"]
    Q = P[comp] - P[comp].mean(0)
    w, v = np.linalg.eigh(Q.T @ Q / len(comp))
    l1, l2 = max(w[2], 1e-9), max(w[1], 1e-9)
    elong = float(np.sqrt(l1 / l2))
    width = float(2.0 * np.sqrt(l2) * 1.7)
    ext = P[comp].max(0) - P[comp].min(0)
    if not (elong < BODY_MAX_ELONG or width > BODY_MIN_WIDTH_UM):
        return None


    fam = fit_models(P[comp], tol=1e9)
    axis = fam["curve"][::2] if fam is not None else np.zeros((0, 3))
    return {"cells": comp, "length_um": float(np.sqrt((ext ** 2).sum())), "n_cells": int(len(comp)),
            "density_per_100um": 0.0, "continuity": 1.0, "rms_um": float(np.sqrt(l2)), "model": "body",
            "axis_model": (fam["model"] if fam is not None else ""),
            "curvature_per_um": (fam["kappa"] if fam is not None else 0.0), "radius_um": None, "fit_rms_um": 0.0,
            "tube_um": width / 2, "contrast": 0.0, "ok": True, "poly": axis, "elongation": elong, "width_um": width}


BRIDGE_BASE_UM = 150.0
BRIDGE_PER_DZ = 1.5
BRIDGE_ANG_DEG = 60.0
BRIDGE_EDGE_UM = 12.5
BRIDGE_MAX_DZ_UM = 60.0


def _end_tangent(poly, at_end):
    k = min(5, len(poly) - 1)
    v = (poly[-1] - poly[-1 - k]) if at_end else (poly[0] - poly[k])
    return _unit(v)


def _bridge_gaps(nerves, gaps):
    cosmin = float(np.cos(np.radians(BRIDGE_ANG_DEG)))
    changed = True
    while changed and gaps:
        changed = False
        ends = []
        for i, c in enumerate(nerves):
            if c["model"] == "body" or len(c["poly"]) < 6:
                continue
            for at_end in (True, False):
                pt = c["poly"][-1] if at_end else c["poly"][0]
                ends.append((i, at_end, pt, _end_tangent(c["poly"], at_end)))
        pairs = []


        for a in ends:
            if a[3][2] <= 0:
                continue
            for b in ends:
                if b[0] == a[0] or b[3][2] >= 0:
                    continue
                dz = float(b[2][2] - a[2][2])
                if dz <= 0 or dz > BRIDGE_MAX_DZ_UM:
                    continue
                g = next(((g0, g1) for g0, g1 in gaps if a[2][2] - BRIDGE_EDGE_UM <= g0 and g1 <= b[2][2] + BRIDGE_EDGE_UM), None)
                if g is None:
                    continue
                chord = b[2] - a[2]
                d = float(np.linalg.norm(chord))
                if d > BRIDGE_BASE_UM + BRIDGE_PER_DZ * dz or d < 1e-6:
                    continue
                u = chord / d
                if float(u @ a[3]) < cosmin or float(-u @ b[3]) < cosmin:
                    continue
                pairs.append((d, a, b, g))
        pairs.sort(key=lambda t: t[0])
        used = set()
        for d, a, b, g in pairs:
            if a[0] in used or b[0] in used:
                continue
            A, B = nerves[a[0]], nerves[b[0]]
            pa = A["poly"] if a[1] else A["poly"][::-1]
            pb = B["poly"] if not b[1] else B["poly"][::-1]
            A["poly"] = np.vstack([pa, pb])
            A["cells"] = np.unique(np.concatenate([A["cells"], B["cells"]]))
            A["n_cells"] = int(len(A["cells"]))
            A["length_um"] = float(A["length_um"] + B["length_um"] + d)
            A["bridged"] = A.get("bridged", 0) + 1
            print(f"  bridged across the gap z {g[0]:g}-{g[1]:g}: {A['model']} {int(A['length_um'] - B['length_um'] - d)} um "
                  f"+ {B['model']} {int(B['length_um'])} um, ends {d:.0f} um apart", flush=True)
            used.add(a[0]); used.add(b[0]); nerves[b[0]] = None
            changed = True
        nerves = [c for c in nerves if c is not None]
    return nerves


EXTEND_MIN_CELLS = 30
EXTEND_BEYOND_UM = 200.0
EXTEND_MAX_UM = 600.0


def _extend_across_gaps(nerves, claimed, gaps):
    P, tree = G["P"], G["tree"]
    in_body = G.get("in_body")
    for c in nerves:
        if c["model"] == "body" or len(c["poly"]) < 6:
            continue
        for at_end in (True, False):
            pt = c["poly"][-1] if at_end else c["poly"][0]
            tg = _end_tangent(c["poly"], at_end)
            if abs(tg[2]) < 1e-6:
                continue
            z_next = pt[2] + np.sign(tg[2]) * BRIDGE_MAX_DZ_UM
            lo, hi = sorted([pt[2], z_next])
            if not any(lo - BRIDGE_EDGE_UM <= g0 and g1 <= hi + BRIDGE_EDGE_UM for g0, g1 in gaps):
                continue
            pts, _rad, entered = _walk(pt.copy(), tg.copy(), gaps, force_enter=True)
            if len(pts) < 3:
                continue
            new = np.array(pts[1:])
            if not entered:


                g = next((g for g in gaps if lo - BRIDGE_EDGE_UM <= g[0] and g[1] <= hi + BRIDGE_EDGE_UM), None)
                far = g[1] if tg[2] > 0 else g[0]
                cap = min(EXTEND_MAX_UM, abs(far - pt[2]) / max(abs(tg[2]), 0.2) + EXTEND_BEYOND_UM)
                seg = np.linalg.norm(np.diff(np.vstack([pt[None], new]), axis=0), axis=1)
                kmax = int(np.searchsorted(np.cumsum(seg), cap)) + 1
                new = new[:kmax]
                if len(new) < 2:
                    continue
            tube = float(c.get("tube_um", TUBE_R_UM))
            nbs = tree.query_ball_point(new, tube)
            idx = np.unique(np.concatenate([np.asarray(nb, int) for nb in nbs]) if any(len(nb) for nb in nbs) else np.zeros(0, int))
            idx = idx[~claimed[idx]]
            if len(idx) < EXTEND_MIN_CELLS:
                continue
            claimed[idx] = True
            c["cells"] = np.unique(np.concatenate([c["cells"], idx]))
            c["n_cells"] = int(len(c["cells"]))
            add = float(np.linalg.norm(np.diff(np.vstack([pt[None], new]), axis=0), axis=1).sum())
            c["length_um"] = float(c["length_um"] + add)
            c["poly"] = np.vstack([c["poly"], new]) if at_end else np.vstack([new[::-1], c["poly"]])
            c["bridged"] = c.get("bridged", 0) + 1
            nb_body = int(in_body[idx].sum()) if in_body is not None else 0
            print(f"  walked on across a gap from z {pt[2]:.0f}: {c['model']} +{add:.0f} um, +{len(idx)} cells "
                  f"({nb_body} from a dense body){' [entered a body]' if entered else ''}", flush=True)
    return claimed


def region_mask(xs_um, ys_um, poly, zk, ny, nx):
    r_px = max(1, int(round(MASK_R_UM / RES)))
    xi = np.clip((np.asarray(xs_um, np.float64) / RES).astype(int), 0, nx - 1)
    yi = np.clip((np.asarray(ys_um, np.float64) / RES).astype(int), 0, ny - 1)


    pad = r_px + 2
    y0, y1 = max(0, yi.min() - pad), min(ny, yi.max() + pad + 1)
    x0, x1 = max(0, xi.min() - pad), min(nx, xi.max() + pad + 1)
    w = np.zeros((y1 - y0, x1 - x0), bool)
    w[yi - y0, xi - x0] = True
    w = binary_closing(binary_dilation(w, iterations=r_px), iterations=1)
    m = np.zeros((ny, nx), bool)
    m[y0:y1, x0:x1] = w


    near = (np.abs(poly[:, 2] - zk) <= 5.0) if len(poly) else np.zeros(0, bool)
    if near.any():
        rw = max(1, int(round(MIN_WAIST_UM / RES)))
        for cx, cy in poly[near][:, :2]:
            gx, gy = int(cx / RES), int(cy / RES)
            yy, xx = np.ogrid[max(0, gy - rw):min(ny, gy + rw + 1), max(0, gx - rw):min(nx, gx + rw + 1)]
            m[max(0, gy - rw):min(ny, gy + rw + 1), max(0, gx - rw):min(nx, gx + rw + 1)] |= ((yy - gy) ** 2 + (xx - gx) ** 2) <= rw * rw
    return m


def _thickest(masks, nid, secs, order, zs):
    from scipy.ndimage import distance_transform_edt, zoom
    best = None
    for ks in secs:
        m = masks.get((nid, order[ks]))
        if m is None or not m.any():
            continue
        ys, xs = np.nonzero(m)
        y0, y1, x0, x1 = ys.min(), ys.max() + 1, xs.min(), xs.max() + 1
        mm = zoom(m[y0:y1, x0:x1].astype(np.uint8), 2, order=0) > 0
        mm = np.pad(mm, 1)
        d2 = distance_transform_edt(mm)
        r = float(d2.max())
        if r > 0 and (best is None or r > best[0]):
            yy, xx = np.unravel_index(int(d2.argmax()), d2.shape)
            best = (r, float(zs[ks]), yy - 1 + 2 * y0, xx - 1 + 2 * x0, mm)
    if best is None:
        return {"max_diameter_um": None, "max_diameter_z_um": None, "max_diameter_line": None}
    r, zb, yy, xx, mm = best
    half = RES / 2.0
    dia = 2.0 * r * half
    ys, xs = np.nonzero(mm)
    pts = np.column_stack([xs - xs.mean(), ys - ys.mean()])
    _ev, evec = np.linalg.eigh(pts.T @ pts / max(1, len(pts)))
    mn = evec[:, 0]
    cx, cy = xx * half, yy * half
    hx, hy = mn[0] * dia / 2.0, mn[1] * dia / 2.0
    return {"max_diameter_um": round(dia, 1), "max_diameter_z_um": zb,
            "max_diameter_line": [[round(cx - hx, 1), round(cy - hy, 1), zb], [round(cx + hx, 1), round(cy + hy, 1), zb]]}


def _score_call(item):
    i, pr = item
    rs = _score(pr, G["gaps"])
    for r in rs:
        r["enter"] = bool(pr[2]) if len(pr) > 2 else False
    return i, rs


def _mesh_call(item):
    return _MESH_ONE(item)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--volume", required=True)
    ap.add_argument("--sample", required=True)
    ap.add_argument("--mesh", action="store_true", help="also stack the member masks into 3-D bodies")
    ap.add_argument("--workers", type=int, default=32)
    a = ap.parse_args()
    VOL = Path(a.volume)
    obj = os.environ.get("HTAN3D_OBJ3D", "")
    OUT = (Path(obj) / "S14_nerve_3d") if obj else HERE.parent / f"outputs/S14_nerve_3d_{a.sample}"
    OUT.mkdir(parents=True, exist_ok=True)
    caches = sorted((VOL / "nerve_operator_cache").glob("stage2_*.pkl"), key=os.path.getmtime)
    if not caches:
        raise SystemExit("no stage cache: run build_nerve_operator3d.py once first")
    st2 = pickle.load(open(caches[-1], "rb"))
    tau, data, cg = st2["tau"], st2["data"], st2["cg"]
    meta = json.loads((VOL / "cell_points_metadata.json").read_text())
    vmeta = json.loads((VOL / "metadata.json").read_text())
    cw, ch = meta["canvas_px"]["width"], meta["canvas_px"]["height"]
    planes = sorted(vmeta["encodings"]["G_withdrawn"]["planes"], key=lambda p: p["z_um"])
    idx_of = {p["section_id"]: p for p in planes}
    order = sorted(data, key=lambda s: data[s][0])
    ny, nx = next(iter(cg.values()))[0].shape


    s12 = Path(os.environ.get("HTAN3D_S12", str(OUT.parent / "S12_cloud")))
    cloud = s12 / "nerve_cloud.bin"
    if not cloud.exists():
        raise SystemExit(f"no denoised cloud at {cloud}: run nerve_denoise_frozen.py first")
    raw = cloud.read_bytes()
    n_pts = struct.unpack_from("<I", raw, 8)[0]
    arr = np.frombuffer(raw, np.uint16, 3 * n_pts, 12).reshape(n_pts, 3)
    off = 12 + 6 * n_pts
    off += (-off) % 4
    zt = np.frombuffer(raw, np.float32, len(order), off)
    pts = [np.column_stack([arr[:, 0].astype(np.float64), arr[:, 1].astype(np.float64), zt[arr[:, 2]].astype(np.float64)])]
    sec = [arr[:, 2].astype(np.int32)]
    print(f"denoised cloud {cloud}: {n_pts:,} points (the page's Denoised layer), {len(zt)} planes", flush=True)
    P = np.concatenate(pts).astype(np.float64)
    SEC = np.concatenate(sec)
    assert len(zt) == len(order) and np.allclose(np.sort(zt), [data[s2][0] for s2 in order]), "cloud planes != cache planes"
    zs = [data[s][0] for s in order]
    gaps = [(zs[k], zs[k + 1]) for k in range(len(zs) - 1) if zs[k + 1] - zs[k] > GAP_MIN_DZ_UM]
    print(f"denoised cells: {len(P):,} on {len(order)} sections (tau {tau:.4f}); "
          f"{len(gaps)} missing-section gaps {[(round(g0), round(g1)) for g0, g1 in gaps]}", flush=True)
    tree = cKDTree(P)
    G.update({"P": P, "tree": tree, "gaps": gaps})


    in_body = np.zeros(len(P), bool)
    body_comps = []
    for comp in _components(np.arange(len(P))):
        if len(comp) < BODY_MIN_CELLS:
            break
        if _body_of(comp) is not None:
            in_body[comp] = True
            body_comps.append(comp)
    claimed = np.zeros(len(P), bool)
    print(f"dense bodies: {len(body_comps)} clusters (>= {BODY_MIN_CELLS} cells, elongation < {BODY_MAX_ELONG:g} or width > "
          f"{BODY_MIN_WIDTH_UM:g} um), {int(in_body.sum()):,} cells; no seed inside them, curves from outside may enter", flush=True)

    import multiprocessing as mp
    W = min(a.workers, mp.cpu_count())

    LIN = np.zeros(len(P), np.float32); NNB = np.zeros(len(P), np.int32); D = np.zeros((len(P), 3), np.float32)
    chunks = np.array_split(np.arange(len(P)), max(1, len(P) // 2000))
    with mp.get_context("fork").Pool(W) as pool:
        for idx, lin, nnb, dirs in pool.imap_unordered(_tensor_chunk, chunks):
            LIN[idx] = lin; NNB[idx] = nnb; D[idx] = dirs
    G["D"] = D
    G["claimed0"] = claimed.copy()
    G["in_body"] = in_body

    cand = np.nonzero((((LIN >= LIN_SEED) & (NNB >= MIN_NB_SEED)) | (NNB >= 3 * MIN_NB_SEED)) & ~in_body)[0]
    cand = cand[np.argsort(-(LIN[cand] * NNB[cand]))]
    taken = np.zeros(len(P), bool)
    seeds = []
    for i in cand:
        if taken[i]:
            continue
        seeds.append(int(i))
        taken[tree.query_ball_point(P[i], SEED_NMS_UM)] = True
    print(f"structure tensor: {int((LIN >= LIN_SEED).sum()):,} linear cells (>= {LIN_SEED:g}); "
          f"{len(seeds):,} seeds after {SEED_NMS_UM:g} um thinning", flush=True)

    with mp.get_context("fork").Pool(W) as pool:
        tracks = list(pool.imap_unordered(_track_seed, seeds, chunksize=8))
        scored = list(pool.imap_unordered(_score_call, tracks, chunksize=8))
    cands = [r for _, rs in scored for r in rs]
    print(f"tracks: {len(tracks):,}; candidates passing the thresholds: {len(cands):,} "
          f"({sum(c['model'] == 'line' for c in cands)} line / {sum(c['model'] == 'arc' for c in cands)} arc / "
          f"{sum(c['model'] == 'parabola' for c in cands)} parabola)", flush=True)

    cands.sort(key=lambda c: -(c["n_cells"] * c["continuity"]))
    nerves = []
    for c in cands:
        ov = float(claimed[c["cells"]].mean())
        if ov >= NMS_OVERLAP:
            continue
        keep = c["cells"][~claimed[c["cells"]]]
        if not c.get("enter"):
            keep = keep[~in_body[keep]]


        comps = _components(keep)
        if not comps:
            continue
        keep = comps[0]
        ext = P[keep].max(0) - P[keep].min(0)
        if len(keep) < 8 or float(np.sqrt((ext ** 2).sum())) < L_MIN_UM:
            continue
        claimed[keep] = True
        c["cells"] = keep
        nerves.append(c)
    print(f"accepted: {len(nerves)} curves, {int(claimed.sum()):,} cells "
          f"({100 * claimed.mean():.1f}% of the denoised cells)", flush=True)
    nerves = _bridge_gaps(nerves, gaps)
    claimed = _extend_across_gaps(nerves, claimed, gaps)


    bodies = []
    for comp in body_comps:
        rest = comp[~claimed[comp]]
        b = _body_of(rest) if len(rest) >= BODY_MIN_CELLS else None
        if b is not None:
            claimed[rest] = True
            bodies.append(b)
        print(f"  body cluster {len(comp):,} cells: {len(comp) - len(rest):,} taken by curves -> "
              f"{'body of ' + str(len(rest)) + ' cells' if b is not None else 'remainder to the cluster pass'}", flush=True)
    nerves += bodies


    rest = np.nonzero(~claimed)[0]
    n_extra = 0
    for comp in _components(rest):
        if len(comp) < 30:
            continue
        ext = P[comp].max(0) - P[comp].min(0)
        if float(np.sqrt((ext ** 2).sum())) < L_MIN_UM:
            continue
        fam = fit_models(P[comp], tol=1e9)
        if fam is None:
            continue
        cur, _sc = _optimize(fam["curve"], P[comp], TUBE_R_UM, np.random.default_rng(len(comp)))
        dperp, _ = cKDTree(cur).query(P[comp])
        t2 = float(np.clip(1.5 * np.median(dperp), TUBE_R_UM, TUBE_R_MAX_UM))
        fam["rms"] = float(np.sqrt(np.mean(dperp ** 2)))
        for r in _finish(cur, fam, t2, gaps, split=False):
            keep = r["cells"][~claimed[r["cells"]]]
            comps2 = _components(keep)
            if not comps2 or len(comps2[0]) < 30:
                continue
            keep = comps2[0]
            claimed[keep] = True
            r["cells"] = keep
            nerves.append(r); n_extra += 1
    print(f"cluster pass: {n_extra} more nerves from unclaimed clusters; "
          f"{int(claimed.sum()):,} cells ({100 * claimed.mean():.1f}%) now in nerves", flush=True)
    nerves = _bridge_gaps(nerves, gaps)
    nerves.sort(key=lambda c: -c["length_um"])


    ndir = {b: (VOL / b / "nerve") for b in ("G_withdrawn", "G_not_withdrawn")}
    ldir = {b: (VOL / b / "nerveline") for b in ("G_withdrawn", "G_not_withdrawn")}
    for d in list(ndir.values()) + list(ldir.values()):
        d.mkdir(parents=True, exist_ok=True)
    masks = {}
    rep = []
    for k, c in enumerate(nerves, 1):
        nid = f"nerve-{k:02d}"
        by_sec = {}
        for j in c["cells"]:
            by_sec.setdefault(int(SEC[j]), []).append(j)
        secs = sorted(by_sec)
        for ks, js in by_sec.items():
            masks[(nid, order[ks])] = region_mask(P[js, 0], P[js, 1], c["poly"], zs[ks], ny, nx)


        dia = _thickest(masks, nid, secs, order, zs)
        poly = c["poly"]
        ext = (poly.max(0) - poly.min(0)) if len(poly) else (P[c["cells"]].max(0) - P[c["cells"]].min(0))
        rep.append({"name": nid, "model": c["model"], "length_um": round(c["length_um"], 1),
                    **dia,
                    "curvature_per_um": round(c["curvature_per_um"], 6),
                    "radius_um": (round(c["radius_um"], 1) if c["radius_um"] else None),
                    "n_cells": int(c["n_cells"]), "density_per_100um": round(c["density_per_100um"], 2),
                    "continuity": round(c["continuity"], 3), "rms_um": round(c["rms_um"], 1),
                    "fit_rms_um": round(c["fit_rms_um"], 1), "n_pieces": c.get("n_pieces", 1),
                    "tube_um": round(c.get("tube_um", TUBE_R_UM), 1), "contrast": round(c.get("contrast", 0.0), 1),
                    "n_sections": len(secs), "z_min_um": zs[secs[0]], "z_max_um": zs[secs[-1]],
                    "extent_um": round(float(np.sqrt((ext ** 2).sum())), 1),
                    "centreline_um": [[round(float(v), 1) for v in p] for p in poly],
                    "width_um": round(float(c.get("width_um", 0.0)), 1), "elongation": round(float(c.get("elongation", 0.0)), 2),
                    "axis_model": c.get("axis_model", ""),
                    "sections": [order[ks] for ks in secs]})
    nerves_by_id = {f"nerve-{k:02d}": c["cells"] for k, c in enumerate(nerves, 1)}
    TP = _load("tlsplanes", HERE / "build_tls_planes.py")

    plane_rows, line_rows = [], []
    z_of_idx = {idx_of[s]["index"]: data[s][0] for s in order}
    for s in order:
        pidx = idx_of[s]["index"]; z = data[s][0]
        ids = [(nid, m) for (nid, s2), m in masks.items() if s2 == s]


        lz = Image.new("RGBA", (cw, ch), (0, 0, 0, 0))
        dr = ImageDraw.Draw(lz)
        nline = 0
        for e in rep:
            pl = np.array(e["centreline_um"])
            if len(pl) < 2 or not (pl[:, 2].min() - 5.0 <= z <= pl[:, 2].max() + 5.0):
                continue


            xy = [(float(x / UM), float(y / UM)) for x, y, _z in pl]
            if len(xy) >= 2:
                dr.line(xy, fill=(255, 255, 255, 110), width=1)
            near = np.abs(pl[:, 2] - z) <= 5.0
            runs = np.split(np.nonzero(near)[0], np.nonzero(np.diff(np.nonzero(near)[0]) > 1)[0] + 1) if near.any() else []
            for run in runs:
                if len(run) >= 2:
                    dr.line([xy[i] for i in run], fill=(255, 255, 255, 255), width=3)
                elif len(run) == 1:
                    x, y = xy[run[0]]; dr.ellipse([x - 3, y - 3, x + 3, y + 3], fill=(255, 255, 255, 255))
            nline += 1
        if nline:
            fn = f"z{pidx:02d}_{s.split('-')[1]}.nerveline.webp"
            for d in ldir.values():
                lz.save(d / fn, format="WEBP", lossless=True, quality=100, method=4)
            line_rows.append({"index": pidx, "section_id": s, "z_um": z, "file": fn, "n_lines": nline})
        if not ids:
            continue


        k_s = order.index(s)
        fill = np.zeros((ch, cw), bool)
        for nid, _m in ids:
            js = [j for j in nerves_by_id[nid] if SEC[j] == k_s]
            if not js:
                continue
            xi = np.clip(np.round(P[js, 0] / UM).astype(int), 0, cw - 1)
            yi = np.clip(np.round(P[js, 1] / UM).astype(int), 0, ch - 1)
            m8 = np.zeros((ch, cw), bool); m8[yi, xi] = True
            y0, y1 = max(0, yi.min() - 6), min(ch, yi.max() + 7); x0, x1 = max(0, xi.min() - 6), min(cw, xi.max() + 7)
            w = m8[y0:y1, x0:x1]
            w = binary_fill_holes(binary_closing(binary_dilation(w, iterations=2), iterations=2))
            fill[y0:y1, x0:x1] |= w


        rim = fill & binary_dilation(~fill, iterations=1)
        a8 = np.zeros(fill.shape, np.uint8); a8[fill] = 1; a8[rim] = 255
        rgba = np.dstack([np.full_like(a8, 255)] * 3 + [a8])
        fn = f"z{pidx:02d}_{s.split('-')[1]}.nerve.webp"
        for d in ndir.values():
            Image.fromarray(rgba).save(d / fn, format="WEBP", lossless=True, quality=100, method=4)
        regs = []
        for nid, m in ids:
            ys, xs = np.nonzero(m)
            regs.append({"id": nid, "cx_um": round(float(xs.mean()) * RES, 1), "cy_um": round(float(ys.mean()) * RES, 1),
                         "bbox_um": [int(xs.min()) * RES, int(ys.min()) * RES, (int(xs.max()) + 1) * RES, (int(ys.max()) + 1) * RES]})
        plane_rows.append({"index": pidx, "section_id": s, "z_um": z, "n_regions": len(regs), "file": fn, "regions": regs})
    (VOL / "nerve_regions_metadata.json").write_text(json.dumps({
        "what": "nerve regions per section: the member cells of each fitted nerve curve, dilated and outlined",
        "colour": REGION_COLOUR, "tau": tau,
        "n_planes": len(plane_rows), "planes": sorted(plane_rows, key=lambda r: r["index"])}, indent=1))
    (VOL / "nerve_lines_metadata.json").write_text(json.dumps({
        "what": "fitted nerve centrelines (line / fixed-curvature arc / parabola) where they pass each section",
        "colour": LINE_COLOUR, "n_planes": len(line_rows), "planes": sorted(line_rows, key=lambda r: r["index"])}, indent=1))
    (OUT / "nerve_curves.json").write_text(json.dumps({
        "sample": a.sample, "tau": tau, "n_nerves": len(rep),
        "params": {"R_TENSOR_UM": R_TENSOR_UM, "LIN_SEED": LIN_SEED, "STEP_UM": STEP_UM, "TUBE_R_UM": TUBE_R_UM,
                   "LOOK_UM": LOOK_UM, "R_MIN_UM": R_MIN_UM, "DKAPPA_MAX": DKAPPA_MAX, "L_MIN_UM": L_MIN_UM,
                   "DENS_LINE": DENS_LINE, "DENS_CURVE": DENS_CURVE, "CONT_LINE": CONT_LINE, "CONT_CURVE": CONT_CURVE,
                   "FIT_RMS_UM": FIT_RMS_UM, "NMS_OVERLAP": NMS_OVERLAP, "MASK_R_UM": MASK_R_UM},
        "nerves": rep}, indent=1))
    rows = ["id\tmodel\tlength_um\tradius_um\tn_cells\tdensity_per_100um\tcontinuity\trms_um\tn_sections\tz_min_um\tz_max_um\textent_um"]
    for e in rep:
        rows.append("\t".join(str(e[k]) if e[k] is not None else "" for k in
                             ("name", "model", "length_um", "radius_um", "n_cells", "density_per_100um", "continuity",
                              "rms_um", "n_sections", "z_min_um", "z_max_um", "extent_um")))
    (OUT / "nerve_curves.tsv").write_text("\n".join(rows) + "\n")

    np.savez_compressed(OUT / "nerve_members.npz", cloud=str(cloud),
                        **{f"nerve-{k:02d}": np.asarray(c["cells"], np.int64) for k, c in enumerate(nerves, 1)})
    print(f"regions: {len(plane_rows)} planes; guide lines: {len(line_rows)} planes; "
          f"models {dict((m, sum(e['model'] == m for e in rep)) for m in ('line', 'arc', 'parabola', 'scurve', 'body'))}, "
          f"chains {sum(e['model'].startswith('chain') for e in rep)}", flush=True)

    if not a.mesh:
        print(f"wrote {OUT / 'nerve_curves.json'} ({len(rep)} nerves; no 3-D bodies, --mesh not given)", flush=True)
        return


    for old in OUT.glob("nerve-*.bin"):
        old.unlink()
    z_of = {s: data[s][0] for s in order}

    def _mesh_one(item):
        e = item
        ms = {z_of[s]: masks[(e["name"], s)].copy() for s in e["sections"]}
        zs_m = sorted(ms)
        if len(zs_m) < 2:
            return None


        pl = np.array(e["centreline_um"])
        def _tube(m, z2, r_um):
            if len(pl) < 2:
                return m
            pts = [q[:2] for q in pl if abs(q[2] - z2) <= 2.6]
            for a_, b_ in zip(pl[:-1], pl[1:]):
                if (a_[2] - z2) * (b_[2] - z2) < 0:
                    t = (z2 - a_[2]) / (b_[2] - a_[2])
                    pts.append(a_[:2] + t * (b_[:2] - a_[:2]))
            if not pts:
                return m
            rw = max(1, int(round(r_um / RES)))
            for cx, cy in pts:
                gx, gy = int(cx / RES), int(cy / RES)
                ya, yb, xa, xb = max(0, gy - rw), min(ny, gy + rw + 1), max(0, gx - rw), min(nx, gx + rw + 1)
                yy, xx = np.ogrid[ya:yb, xa:xb]
                m[ya:yb, xa:xb] |= ((yy - gy) ** 2 + (xx - gx) ** 2) <= rw * rw
            return m


        for s2 in order:
            z2 = z_of[s2]
            if zs_m[0] <= z2 <= zs_m[-1]:
                m2 = _tube(ms.get(z2, np.zeros((ny, nx), bool)), z2, TUBE_BODY_UM)
                if m2.any():
                    ms[z2] = m2
        zs_m = sorted(ms)
        cen = {z2: np.zeros(2) for z2 in zs_m}
        nz = [m for m in ms.values() if m.any()]
        ys0 = min(int(np.nonzero(m)[0].min()) for m in nz); ys1 = max(int(np.nonzero(m)[0].max()) for m in nz)
        xs0 = min(int(np.nonzero(m)[1].min()) for m in nz); xs1 = max(int(np.nonzero(m)[1].max()) for m in nz)
        pad = int(round(120.0 / RES))
        y0, y1 = max(0, ys0 - pad), min(ny, ys1 + pad + 1); x0, x1 = max(0, xs0 - pad), min(nx, xs1 + pad + 1)


        sdf = {z2: np.minimum(BM.sdf(ms[z2][y0:y1, x0:x1]), 40.0) for z2 in zs_m}
        zg = np.arange(zs_m[0], zs_m[-1] + 0.5 * BM.Z_STEP, BM.Z_STEP, dtype=np.float32)
        vol = np.empty((len(zg), y1 - y0, x1 - x0), np.float32); uns = np.zeros(len(zg), bool)
        za_ = np.asarray(zs_m, np.float32)
        for k2, z2 in enumerate(zg):
            j = int(np.searchsorted(za_, z2, "right")) - 1
            j = max(0, min(j, len(za_) - 2))
            za, zb = za_[j], za_[j + 1]
            span = float(zb - za); t = 0.0 if span <= 0 else min(1.0, max(0.0, float((z2 - za) / span)))
            ca, cb = cen[float(za)], cen[float(zb)]; cc = (1 - t) * ca + t * cb
            fa = ndshift(sdf[float(za)], cc - ca, order=1, mode="nearest"); fb = ndshift(sdf[float(zb)], cc - cb, order=1, mode="nearest")
            vol[k2] = (1 - t) * fa + t * fb
            if span > BM.GAP_UM and 0.0 < t < 1.0:
                uns[k2] = True
        from scipy.ndimage import label as _label3, gaussian_filter as _gf
        n_comp = int(_label3(_gf(vol, 1.0) < 0)[1])
        if n_comp > 1:


            cells_idx = nerves_by_id[e["name"]]
            for r_cell, r_try in ((40.0, 50.0), (50.0, 70.0), (60.0, 100.0), (80.0, 150.0), (100.0, 200.0)):
                rp = max(1, int(round(r_cell / RES)))
                for z2 in zs_m:
                    ms[z2] = np.zeros((ny, nx), bool)
                for j in cells_idx:
                    z2 = float(P[j, 2])
                    if z2 in ms:
                        gx, gy = int(P[j, 0] / RES), int(P[j, 1] / RES)
                        ya, yb, xa, xb = max(0, gy - rp), min(ny, gy + rp + 1), max(0, gx - rp), min(nx, gx + rp + 1)
                        yy, xx = np.ogrid[ya:yb, xa:xb]
                        ms[z2][ya:yb, xa:xb] |= ((yy - gy) ** 2 + (xx - gx) ** 2) <= rp * rp
                for z2 in zs_m:
                    ms[z2] = _tube(ms[z2], z2, r_try)
                sdf = {z2: np.minimum(BM.sdf(ms[z2][y0:y1, x0:x1]), 40.0) for z2 in zs_m}
                for z2 in zs_m:
                    if not ms[z2].any():
                        sdf[z2] = None
                zs_k = [z2 for z2 in zs_m if sdf[z2] is not None]
                za_ = np.asarray(zs_k, np.float32)
                for k2, z2 in enumerate(zg):
                    j = int(np.searchsorted(za_, z2, "right")) - 1
                    j = max(0, min(j, len(za_) - 2))
                    za, zb = za_[j], za_[j + 1]
                    span = float(zb - za); t = 0.0 if span <= 0 else min(1.0, max(0.0, float((z2 - za) / span)))
                    vol[k2] = (1 - t) * sdf[float(za)] + t * sdf[float(zb)]
                n_comp = int(_label3(_gf(vol, 1.0) < 0)[1])
                if n_comp <= 1:
                    break


        n_bridges = 0
        for _it in range(12):
            sm = _gf(vol, 1.0) < 0
            lb, n_comp = _label3(sm)
            if n_comp <= 1:
                break
            sizes = np.bincount(lb.ravel()); sizes[0] = 0
            main = int(sizes.argmax())
            others = [i for i in range(1, n_comp + 1) if i != main]
            A = np.argwhere(lb == main)[::7]
            for ci in others:
                B = np.argwhere(lb == ci)[::3]

                scale = np.array([BM.Z_STEP, RES, RES])
                d, k = cKDTree(A * scale).query(B * scale)
                b = B[int(np.argmin(d))]; a = A[int(k[int(np.argmin(d))])]
                nstep = int(max(2, np.ceil(np.linalg.norm((b - a) * scale) / 5.0)))
                rz, rxy = 6, 3
                for t in np.linspace(0.0, 1.0, nstep):
                    c0 = np.round(a + t * (b - a)).astype(int)
                    z0_, z1_ = max(0, c0[0] - rz), min(vol.shape[0], c0[0] + rz + 1)
                    y0_, y1_ = max(0, c0[1] - rxy), min(vol.shape[1], c0[1] + rxy + 1)
                    x0_, x1_ = max(0, c0[2] - rxy), min(vol.shape[2], c0[2] + rxy + 1)
                    zz, yy, xx = np.ogrid[z0_:z1_, y0_:y1_, x0_:x1_]
                    ball = ((zz - c0[0]) / rz) ** 2 + ((yy - c0[1]) / rxy) ** 2 + ((xx - c0[2]) / rxy) ** 2 <= 1.0
                    sub = vol[z0_:z1_, y0_:y1_, x0_:x1_]
                    sub[ball] = np.minimum(sub[ball], -15.0)
                n_bridges += 1
        n_comp = int(_label3(_gf(vol, 1.0) < 0)[1])
        m = BM.mesh_from(vol, zg, uns, RES / BM.UM_PX, smooth_vox=1.0)
        if m is None:
            return None
        m["verts"][:, 0] += x0 * RES; m["verts"][:, 1] += y0 * RES
        r = BM.write_mesh(e["name"], m, OUT)
        r["n_solids"] = n_comp
        r["n_bridges"] = n_bridges
        v3, f3 = m["verts"], m["faces"]
        p0, p1, p2 = v3[f3[:, 0]], v3[f3[:, 1]], v3[f3[:, 2]]
        vol_um3 = float(abs(np.einsum("ij,ij->i", p0.astype(np.float64), np.cross(p1.astype(np.float64), p2.astype(np.float64))).sum()) / 6.0)
        r.update({"structure": "nerve", "cell_class": "Schwann", "model": e["model"], "length_um": e["length_um"],
                  "radius_um": e["radius_um"], "n_sections": e["n_sections"], "z_min_um": e["z_min_um"],
                  "z_max_um": e["z_max_um"], "n_regions": e["n_sections"], "n_cells": e["n_cells"],
                  "extent_um": e["extent_um"], "volume_um3": round(vol_um3, 1), "high_confidence": True,
                  "tube_um": e.get("tube_um"), "max_diameter_um": e.get("max_diameter_um"),
                  "max_diameter_z_um": e.get("max_diameter_z_um"), "max_diameter_line": e.get("max_diameter_line"),
                  "centreline_um": e["centreline_um"]})
        return r

    globals()["_MESH_ONE"] = _mesh_one
    out = []
    with mp.get_context("fork").Pool(W) as pool:
        for r in pool.imap_unordered(_mesh_call, rep):
            if r is not None:
                out.append(r)
    out.sort(key=lambda r: int(r["name"].split("-")[1]))


    _bad = [r["name"] for r in out if r.get("n_solids", 1) > 1]
    if _bad:
        raise SystemExit(f"REFUSED: {len(_bad)} nerve bodies are not one solid: {_bad[:10]}")
    (OUT / "Schwann_nerves.json").write_text(json.dumps({
        "class": "Schwann", "basis": "G_withdrawn", "tau": tau,
        "how_defined": "smooth curves (line / fixed-curvature arc / parabola) tracked through the denoised "
                       "Schwann cells; a body is the member cells of each section, dilated and stacked.",
        "claim": "MODEL PREDICTION on H&E (pan-cancer Cell head, single fold). No spatial ground truth in this cohort.",
        "nerves": out, "rejected": [], "n_nerves": len(out),
        "display_rule": f"fitted curve length >= {L_MIN_UM:g} um"}, indent=1))
    print(f"wrote {OUT}: {len(out)} nerve bodies", flush=True)


if __name__ == "__main__":
    main()
