#!/usr/bin/env python3

import os
import argparse, csv, importlib.util, json, struct, sys
from pathlib import Path
import numpy as np
import pandas as pd
import cv2
from scipy.ndimage import binary_dilation

P3D = Path(os.environ.get("THREED_ROOT", os.path.join(os.environ.get("PROJECTS_ROOT", "/data/heid"), "3d")))
X16 = Path(__file__).resolve().parents[2] / "nerve/prediction/scripts/region_operator.py"
HERE = Path(__file__).resolve().parent

def _load(name, p):
    spec = importlib.util.spec_from_file_location(name, p)
    m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
    return m

V16 = _load("v16op", X16)
BM = _load("bm", HERE / "build_meshes.py")


_exf = Path(__import__('os').environ.get("HTAN3D_EXCLUDE",
            os.environ.get("THREED_ROOT", os.path.join(os.environ.get("PROJECTS_ROOT", "/data/heid"), "3d")) + "/front/S22-27909/configs/exclude_sections_3d.json"))
EXCLUDE_3D = set()
if _exf.exists():
    import json as _json
    EXCLUDE_3D = set(_json.loads(_exf.read_text())["sections"])

POSFRAC = 0.003
OVERLAP_DILATE_UM = 60.0
LINK_DRIFT_PER_UM = 2.0
LINK_MAX_DZ_UM = 60.0
LINK_OVERLAP_MIN = 0.3
STITCH_XY_UM = 150.0
STITCH_DRIFT_PER_UM = 1.5


BRIDGE_DILATE_UM = 90.0
MIN_SECTIONS = 3


def _mesh_call(item):
    return _MESH_ONE(item)


def _pool_call(item):
    return globals()["_one_section"](item)


def write_display(rep, OUT, MIN_DIAG_UM):
    slim = [n for n in rep["nerves"] if n["extent_um"] >= MIN_DIAG_UM]
    rep2 = dict(rep)
    rep2["nerves"] = sorted(slim, key=lambda n: -n["extent_um"])
    rep2["n_nerves"] = len(slim)
    rep2["display_rule"] = (f"bbox diagonal >= {MIN_DIAG_UM:g} um "
                            "(physical length, any orientation)")
    (OUT / "Schwann_nerves.json").write_text(json.dumps(rep2, indent=1))
    print(f"display filter: kept {len(slim)} cords >= {MIN_DIAG_UM:g} um, "
          f"dropped {len(rep['nerves']) - len(slim)}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--volume", required=True)
    ap.add_argument("--sample", required=True)
    ap.add_argument("--down", type=int, default=6)
    ap.add_argument("--refilter", action="store_true",
                    help="only re-apply the display rule to the last full run (no linking, no meshing)")
    a = ap.parse_args()
    VOL = Path(a.volume)
    _OBJ3D = __import__("os").environ.get("HTAN3D_OBJ3D", "")
    OUT = (Path(_OBJ3D) / "S14_nerve_3d") if _OBJ3D else HERE.parent / f"outputs/S14_nerve_3d_{a.sample}"
    OUT.mkdir(parents=True, exist_ok=True)

    MIN_DIAG_UM = float(__import__("os").environ.get("NERVE_MIN_EXTENT_UM", "1000"))
    if a.refilter:
        rep = json.loads((OUT.parent / f"Schwann_nerves_{a.sample}_full.json").read_text())
        write_display(rep, OUT, MIN_DIAG_UM)
        return

    G = Path(__import__('os').environ.get("HTAN3D_GEN", str(P3D / "front/S22-27909")))
    PRED = Path(__import__('os').environ.get("HTAN3D_PRED", str(P3D / "inference/S22-27909/data/predictions")))
    v3 = {}
    for r in csv.DictReader(open(__import__('os').environ.get("HTAN3D_RUNMAN", str(P3D / "inference/S22-27909/configs/s22/run_manifest.tsv"))), delimiter="\t"):
        u = int(r["section_number"])
        if u not in v3 or r["slide"].endswith("r"):
            v3[u] = r["slide"]

    meta = json.loads((VOL / "cell_points_metadata.json").read_text())
    vmeta = json.loads((VOL / "metadata.json").read_text())
    classes = meta["classes"]
    cw, ch = meta["canvas_px"]["width"], meta["canvas_px"]["height"]
    UM = 8.0
    grid = {"x_min": 0.0, "y_min": 0.0,
            "nx": int(np.ceil(cw * UM / V16.RESOLUTION)),
            "ny": int(np.ceil(ch * UM / V16.RESOLUTION))}
    planes = sorted(vmeta["encodings"]["G_withdrawn"]["planes"], key=lambda p: p["z_um"])

    def read_plane(p):
        f = sorted((VOL / "G_withdrawn/cell_points").glob(f"z{p['index']:02d}_*.cells.bin"))
        if not f:
            return None
        raw = open(f[0], "rb").read()
        magic, n, _cw, _ch, sub, ncls, _ = struct.unpack("<8sIHHHHI", raw[:24])
        assert magic == b"CELLPT01"
        xy = np.frombuffer(raw[32:32 + 4 * n], dtype="<u2").reshape(n, 2).astype(np.float64) / sub * UM
        cl = np.frombuffer(raw[32 + 5 * n:32 + 6 * n], dtype=np.uint8)
        u = int(p["section_id"].rsplit("_U", 1)[1])
        slide = v3[u]
        pred = pd.read_parquet(PRED / f"{slide}.parquet",
                               columns=["pred_class_name", "schwann_prob"])
        assert len(pred) == n, (p["section_id"], len(pred), n)
        idx = pred["pred_class_name"].map({c: i for i, c in enumerate(classes)}).to_numpy()
        assert np.array_equal(idx, cl), p["section_id"]
        return xy, pred["schwann_prob"].to_numpy(np.float64)


    SUP_GRID, SUP_TOL = 20.0, 60.0
    SUP_MIN_CELLS, SUP_MIN_SECTIONS, SUP_BIG = 8, 2, 50
    import pickle as _pk
    CACHE = VOL / "nerve_operator_cache"
    CACHE.mkdir(exist_ok=True)


    _cv = f"_c{cw}x{ch}"
    STAGE2 = CACHE / (f"stage2_pf{POSFRAC:g}_res{V16.RESOLUTION:g}_tissue1_sup{SUP_MIN_CELLS}-"
                      f"{SUP_MIN_SECTIONS}-{SUP_BIG}t{SUP_TOL:g}_x{len(EXCLUDE_3D)}{_cv}.pkl")
    masks, stats = {}, {}
    if STAGE2.exists():
        _st2 = _pk.load(open(STAGE2, "rb"))
        tau, data, stats = _st2["tau"], _st2["data"], _st2["stats"]
        masks = {sid: {i: (cg == i) for i in ids} for sid, (cg, ids) in _st2["cg"].items()}
        print(f"stage cache {STAGE2.name}: tau {tau:.4f}, {len(data)} sections, "
              f"{sum(len(m) for m in masks.values())} regions", flush=True)
    else:

        data, probs = {}, []
        for p in planes:
            r = read_plane(p)
            assert r is not None, f"no cell points for {p['section_id']}"
            data[p["section_id"]] = (float(p["z_um"]), r[0], r[1])
            probs.append(r[1])
        assert len(data) == len(planes)
        tau = float(np.quantile(np.concatenate(probs), 1.0 - POSFRAC))
        del probs
        print(f"cohort tau (posfrac {POSFRAC}) = {tau:.4f} over {len(data)} sections", flush=True)


        from scipy.ndimage import label as _lab3, binary_dilation as _bdil
        from PIL import Image as _Im
        _idx0 = {q["section_id"]: q for q in planes}
        work_sids = [sid for sid, _ in sorted(data.items(), key=lambda kv: kv[1][0])
                     if sid not in EXCLUDE_3D]
        sel0, cgrid = {}, {}
        ta0 = None
        for sid in work_sids:
            _z, xy, pr = data[sid]
            tf = sorted((VOL / "G_withdrawn/webp").glob(
                f"z{_idx0[sid]['index']:02d}_*.webp"))
            ta = np.asarray(_Im.open(tf[0]).convert("RGBA"))[:, :, 3]
            if ta0 is None:
                ta0 = ta
            cx = np.clip(np.round(xy[:, 0] / 8.0).astype(int), 0, ta.shape[1] - 1)
            cy = np.clip(np.round(xy[:, 1] / 8.0).astype(int), 0, ta.shape[0] - 1)
            sel0[sid] = (pr >= tau) & (ta[cy, cx] > 0)
            cgrid[sid] = (np.clip((xy[:, 0] / SUP_GRID).astype(int), 0, 10 ** 6),
                          np.clip((xy[:, 1] / SUP_GRID).astype(int), 0, 10 ** 6))
        gx = int(np.ceil(ta0.shape[1] * 8.0 / SUP_GRID))
        gy = int(np.ceil(ta0.shape[0] * 8.0 / SUP_GRID))
        occ = np.zeros((len(work_sids), gy, gx), bool)
        for k, sid in enumerate(work_sids):
            s0 = sel0[sid]
            xi = np.minimum(cgrid[sid][0], gx - 1)
            yi = np.minimum(cgrid[sid][1], gy - 1)
            cgrid[sid] = (xi, yi)
            occ[k, yi[s0], xi[s0]] = True
        _dil = max(1, int(round(SUP_TOL / SUP_GRID)))
        occ_d = np.stack([_bdil(occ[k], iterations=_dil) for k in range(len(work_sids))])
        _st = np.zeros((3, 3, 3), bool); _st[1] = True; _st[0, 1, 1] = True; _st[2, 1, 1] = True
        _lb, _n = _lab3(occ_d, structure=_st)
        _ncell = np.zeros(_n + 1, np.int64)
        _zspan = [set() for _ in range(_n + 1)]
        _ids = {}
        for k, sid in enumerate(work_sids):
            s0 = sel0[sid]; xi, yi = cgrid[sid]
            ids = _lb[k, yi[s0], xi[s0]]
            _ids[sid] = ids
            np.add.at(_ncell, ids, 1)
            for i2 in np.unique(ids):
                if i2:
                    _zspan[i2].add(k)
        _good = np.zeros(_n + 1, bool)
        for i2 in range(1, _n + 1):
            _good[i2] = (len(_zspan[i2]) >= SUP_MIN_SECTIONS
                         and _ncell[i2] >= SUP_MIN_CELLS) or _ncell[i2] >= SUP_BIG
        SEL3D, _drop = {}, 0
        for sid in work_sids:
            s0 = sel0[sid]
            sf = s0.copy()
            sf[np.where(s0)[0]] = _good[_ids[sid]]
            SEL3D[sid] = sf
            _drop += int(s0.sum() - sf.sum())
        _tot = sum(int(sel0[sid].sum()) for sid in work_sids)
        print(f"3-D support filter: kept {_tot - _drop:,}/{_tot:,} tau cells "
              f"({100 * (_tot - _drop) / max(1, _tot):.1f}%), "
              f"components {int(_good.sum())}/{_n}", flush=True)


        from PIL import Image as _Im
        idx_of0 = {q["section_id"]: q for q in planes}


        ckey = (f"tau{tau:.6f}_res{V16.RESOLUTION:g}_tissue1"
                f"_sup{SUP_MIN_CELLS}-{SUP_MIN_SECTIONS}-{SUP_BIG}t{SUP_TOL:g}{_cv}")

        global _NCTX
        _NCTX = {"tau": tau, "grid": grid, "idx_of0": idx_of0, "VOL": VOL,
                 "CACHE": CACHE, "ckey": ckey}

        def one_section(item):
            sid, (z, xy, pr) = item
            cf = CACHE / f"{sid}.{ckey}.npz"
            if cf.exists():
                d0 = np.load(cf)
                return sid, z, int(d0["n_sel"]), d0["cg"]
            sel = SEL3D[sid]
            cg, clusters, m = V16.run_operator(xy[:, 0], xy[:, 1], sel, panel="5k", grid=grid)
            np.savez_compressed(cf, cg=cg.astype(np.int32), n_sel=int(sel.sum()))
            return sid, z, int(sel.sum()), cg

        import multiprocessing as _mp
        globals()["_one_section"] = one_section
        work = [kv for kv in sorted(data.items(), key=lambda kv: kv[1][0])
                if kv[0] not in EXCLUDE_3D]
        if len(work) < len(data):
            print(f"excluded {len(data) - len(work)} flagged sections from every "
                  f"3-D product: {sorted(set(data) - {w[0] for w in work})}", flush=True)
        with _mp.get_context("fork").Pool(min(16, _mp.cpu_count())) as pool:
            for sid, z, n_sel, cg in pool.imap_unordered(_pool_call, work):
                ids = sorted(set(int(v) for v in np.unique(cg)) - {0})
                masks[sid] = {i: (cg == i) for i in ids}
                stats[sid] = {"z_um": z, "n_pred_schwann": n_sel, "n_regions": len(ids)}
                print(f"  {sid}: schwann {n_sel} regions {len(ids)}", flush=True)
        _cg = {}
        for sid, m in masks.items():
            g2 = np.zeros((grid["ny"], grid["nx"]), np.int32)
            for i in m:
                g2[m[i]] = i
            _cg[sid] = (g2, sorted(m))
        _pk.dump({"tau": tau, "data": data, "stats": stats, "cg": _cg}, open(STAGE2, "wb"), protocol=4)
        print(f"stage cache written: {STAGE2.name}", flush=True)


    BRIDGE_MAX_DZ_UM = 120.0
    RESCUE_R_UM, RESCUE_MIN_CELLS, RESCUE_POSFRAC_X = 150.0, SUP_MIN_CELLS, 2.0
    BRIDGE_FIT_PLANES = 5
    BRIDGE_MAX_BEND_DEG = 60.0
    BRIDGE_SUPPORT_GAIN = 1.0
    BRIDGE_PENDING_X = 2.0
    order = sorted(masks, key=lambda s: data[s][0])
    z_of = {s: data[s][0] for s in order}
    rescued = {}

    def tol_um(dz):
        return OVERLAP_DILATE_UM + LINK_DRIFT_PER_UM * dz

    def _centreline(mem):
        by_z = {}
        for s2, i2 in mem:
            by_z.setdefault(data[s2][0], []).append(masks[s2][i2])
        cl = {}
        for z2, ms2 in by_z.items():
            m2 = ms2[0]
            for extra in ms2[1:]:
                m2 = m2 | extra
            ys, xs = np.nonzero(m2)
            cl[z2] = np.array([xs.mean() * V16.RESOLUTION, ys.mean() * V16.RESOLUTION])
        return cl

    def _curve(cl, zs2, at_top):
        pts = zs2[-BRIDGE_FIT_PLANES:] if at_top else zs2[:BRIDGE_FIT_PLANES]
        z = np.array(pts, float); P = np.array([cl[p] for p in pts], float)
        deg = 2 if len(pts) >= 4 else (1 if len(pts) >= 2 else 0)
        if deg and (z[-1] - z[0]) < 1e-6:
            deg = 0
        coef = [np.polyfit(z, P[:, k], deg) for k in (0, 1)]
        def at(zq):
            return np.array([np.polyval(c, zq) for c in coef])
        def tang(zq):
            return np.array([np.polyval(np.polyder(c), zq) for c in coef]) if deg else np.zeros(2)
        return at, tang, deg >= 1

    def _bend_deg(t_xy, chord):
        t = np.array([t_xy[0], t_xy[1], 1.0]); c = np.asarray(chord, float)
        cosv = float(np.dot(t, c) / max(1e-9, np.linalg.norm(t) * np.linalg.norm(c)))
        return float(np.degrees(np.arccos(np.clip(cosv, -1.0, 1.0))))

    def _plane_union(mem, z2):
        m2 = None
        for s2, i2 in mem:
            if data[s2][0] == z2:
                m2 = masks[s2][i2] if m2 is None else (m2 | masks[s2][i2])
        return m2

    tau_r = float(np.quantile(np.concatenate([data[s][2] for s in order]),
                              1.0 - RESCUE_POSFRAC_X * POSFRAC))
    relaxed = {s: data[s][1][data[s][2] >= tau_r] for s in order}

    def _gap_support(pa_at, pb_at, zlo, zhi, empty):
        if not empty:
            return 0.0, {}
        counts, sup = {}, []
        for s in empty:
            z2 = z_of[s]; w = (z2 - zlo) / (zhi - zlo)
            c = (1.0 - w) * pa_at(z2) + w * pb_at(z2)
            xy = relaxed[s]
            n = int((np.hypot(xy[:, 0] - c[0], xy[:, 1] - c[1]) <= RESCUE_R_UM).sum()) if len(xy) else 0
            counts[s] = n; sup.append(min(1.0, n / RESCUE_MIN_CELLS))
        return float(np.mean(sup)), counts

    EDGES = []
    SHP = {}

    def _shape(s, i):
        k = (s, i)
        if k not in SHP:
            ys, xs = np.nonzero(masks[s][i])
            Pm = np.stack([xs, ys], 1).astype(float) * V16.RESOLUTION
            c = Pm.mean(0)
            if len(Pm) < 3:
                SHP[k] = (c, np.array([1.0, 0.0]), 0.0, 0.0)
            else:
                Q = Pm - c
                _w, _v = np.linalg.eigh(np.cov(Q.T))
                u = _v[:, -1]; t = Q @ u; t2 = Q @ _v[:, 0]
                SHP[k] = (c, u, float(max(t.max(), -t.min())), float(t2.max() - t2.min()))
        return SHP[k]

    def relink(new=None):
        node = {}
        for sid in order:
            for i in masks[sid]:
                node[(sid, i)] = (sid, i)
        def find(x):
            while node[x] != x:
                node[x] = node[node[x]]; x = node[x]
            return x


        def _bbox(m):
            ys, xs = np.nonzero(m)
            return (int(ys.min()), int(ys.max()) + 1, int(xs.min()), int(xs.max()) + 1)
        BB = {s: {i: _bbox(m) for i, m in masks[s].items()} for s in order}


        UP, DOWN = set(), set()
        def link_pairs(s0, s1):
            dz = data[s1][0] - data[s0][0]
            it = max(1, int(round(tol_um(dz) / V16.RESOLUTION)))
            H, W = grid["ny"], grid["nx"]
            cands = []
            for i, m0 in masks[s0].items():
                if (s0, i) in UP:
                    continue
                if new is not None and (s0, i) not in new:
                    js = [j for j in masks[s1] if (s1, j) in new]
                    if not js:
                        continue
                else:
                    js = list(masks[s1])
                y0, y1, x0, x1 = BB[s0][i]
                y0, y1 = max(0, y0 - it), min(H, y1 + it); x0, x1 = max(0, x0 - it), min(W, x1 + it)
                d0 = binary_dilation(m0[y0:y1, x0:x1], iterations=it)
                for j in js:
                    m1 = masks[s1][j]
                    b1 = BB[s1][j]
                    if b1[0] >= y1 or b1[1] <= y0 or b1[2] >= x1 or b1[3] <= x0:
                        continue


                    w1 = m1[y0:y1, x0:x1]
                    inter = int((d0 & w1).sum())
                    if (s1, j) in DOWN:
                        continue
                    if inter and inter >= LINK_OVERLAP_MIN * min(int(w1.sum()), int(m0.sum())):
                        cands.append((inter / min(int(w1.sum()), int(m0.sum())), (s0, i), (s1, j)))
            cands.sort(key=lambda c: -c[0])
            for _fr, ka, kb in cands:
                if ka in UP or kb in DOWN:
                    continue
                UP.add(ka); DOWN.add(kb)
                node[find(ka)] = find(kb)
                EDGES.append((ka, kb))
        n_pairs = 0
        if new is not None:
            for u1, v1 in EDGES:
                node[find(u1)] = find(v1)
                UP.add(u1); DOWN.add(v1)
            n_pairs = -len(EDGES)
        for skip in (1, 2, 3):
            for s0, s1 in zip(order, order[skip:]):
                dz = data[s1][0] - data[s0][0]
                if skip > 1 and dz > LINK_MAX_DZ_UM:
                    continue
                link_pairs(s0, s1); n_pairs += 1


        IN_PLANE_GAP_UM, IN_PLANE_ANGLE_DEG = 150.0, 30.0
        IN_PLANE_MIN_LEN_UM, IN_PLANE_ELONG = 100.0, 2.0
        _cosA = np.cos(np.radians(IN_PLANE_ANGLE_DEG))
        n_inplane = 0
        for s in order:
            el = []
            for i in masks[s]:
                c, u, Lh, wd = _shape(s, i)
                if 2 * Lh >= IN_PLANE_MIN_LEN_UM and Lh >= IN_PLANE_ELONG * max(wd, V16.RESOLUTION):
                    el.append((i, c, u, Lh))
            for a_ in range(len(el)):
                i, ci, ui, Li = el[a_]
                for b_ in range(a_ + 1, len(el)):
                    j, cj, uj, Lj = el[b_]
                    if abs(float(ui @ uj)) < _cosA:
                        continue
                    if float(np.hypot(*(cj - ci))) > Li + Lj + IN_PLANE_GAP_UM:
                        continue
                    ok = False
                    for sa in (-1.0, 1.0):
                        for sb in (-1.0, 1.0):
                            ea = ci + sa * ui * Li; eb = cj + sb * uj * Lj
                            g = eb - ea; d = float(np.hypot(*g))
                            if d > IN_PLANE_GAP_UM or d < 1e-6:
                                continue
                            if abs(float(g @ ui)) / d < _cosA or abs(float(g @ uj)) / d < _cosA:
                                continue
                            if float(g @ (sa * ui)) <= 0 or float((-g) @ (sb * uj)) <= 0:
                                continue
                            ok = True
                    if ok:
                        node[find((s, i))] = find((s, j)); n_inplane += 1
        print(f"in-plane collinear pieces joined: {n_inplane} (gap <= {IN_PLANE_GAP_UM:g} um, "
              f"axes within {IN_PLANE_ANGLE_DEG:g} deg)", flush=True)
        cords = {}
        for k in node:
            cords.setdefault(find(k), []).append(k)
        print(f"overlap linking over {n_pairs} section pairs (tolerance "
              f"{OVERLAP_DILATE_UM:g} um + {LINK_DRIFT_PER_UM:g} um/um of dz, "
              f"skips up to {LINK_MAX_DZ_UM:g} um): {len(cords)} cords", flush=True)
        it_res = max(1, int(round(OVERLAP_DILATE_UM / V16.RESOLUTION)))
        n_st_total, n_round, pending = 0, 0, []
        while True:
            n_round += 1
            cords = {}
            for k in node:
                cords.setdefault(find(k), []).append(k)
            cls = {r: _centreline(m) for r, m in cords.items()}
            zr = {r: sorted(c) for r, c in cls.items()}
            _ccache = {}
            def _end(r1, at_top):
                k1 = (r1, at_top)
                if k1 not in _ccache:
                    _ccache[k1] = _curve(cls[r1], zr[r1], at_top)
                return _ccache[k1]
            cand, pending = [], []
            roots = list(cords)
            for a1 in roots:
                za = zr[a1]
                for b1 in roots:
                    if a1 == b1:
                        continue
                    zb = zr[b1]

                    dz = zb[0] - za[-1]
                    if 0 < dz <= BRIDGE_MAX_DZ_UM:
                        ea, eb = cls[a1][za[-1]], cls[b1][zb[0]]
                        d0 = float(np.hypot(*(eb - ea)))
                        tol = STITCH_XY_UM + STITCH_DRIFT_PER_UM * dz
                        if d0 > BRIDGE_PENDING_X * tol + 2.0 * STITCH_DRIFT_PER_UM * dz * BRIDGE_FIT_PLANES:
                            continue
                        at_a, tg_a, ok_a = _end(a1, True)
                        at_b, tg_b, ok_b = _end(b1, False)
                        zm = 0.5 * (za[-1] + zb[0])
                        d_curve = float(np.hypot(*(at_b(zm) - at_a(zm))))
                        d = min(d_curve, d0)


                        chord = (eb[0] - ea[0], eb[1] - ea[1], dz)
                        bend = 0.0
                        if d0 > STITCH_XY_UM:
                            if ok_a:
                                bend = max(bend, _bend_deg(tg_a(za[-1]), chord))
                            if ok_b:
                                bend = max(bend, _bend_deg(tg_b(zb[0]), chord))
                        if bend > BRIDGE_MAX_BEND_DEG:
                            continue
                        empty = [s for s in order if za[-1] < z_of[s] < zb[0]]
                        support, _cnt = _gap_support(at_a, at_b, za[-1], zb[0], empty)
                        tol_eff = tol * (1.0 + BRIDGE_SUPPORT_GAIN * support)
                        if d <= tol_eff:
                            cand.append((d / tol_eff, a1, b1))
                        elif d <= BRIDGE_PENDING_X * tol:
                            pending.append((a1, b1, za[-1], zb[0], empty))
                        continue
            cand.sort(key=lambda c: c[0])
            n_st, used_top, used_bot = 0, set(), set()
            for d, a1, b1 in cand:
                if a1 in used_top or b1 in used_bot:
                    continue
                ra, rb = find(cords[a1][0]), find(cords[b1][0])
                if ra == rb:
                    continue
                used_top.add(a1); used_bot.add(b1)
                node[ra] = rb; n_st += 1
            n_st_total += n_st
            print(f"stitching round {n_round}: {n_st} joins", flush=True)
            if n_st == 0 or n_round >= 8:
                break
        cords = {}
        for k in node:
            cords.setdefault(find(k), []).append(k)
        print(f"stitching: {n_st_total} joins in {n_round} rounds "
              f"(end-to-end <= {BRIDGE_MAX_DZ_UM:g} um in z, curves through {BRIDGE_FIT_PLANES} planes "
              f"meeting within {STITCH_XY_UM:g} um + {STITCH_DRIFT_PER_UM:g} um/um (x up to "
              f"{1 + BRIDGE_SUPPORT_GAIN:g} with Schwann cells on the empty planes), bend <= "
              f"{BRIDGE_MAX_BEND_DEG:g} deg, one continuation per end; no side-by-side merge); "
              f"{len(cords)} cords; {len(pending)} bridges held back for the transition fit", flush=True)
        return cords, pending

    def rescue(cords, pending):
        targets = []
        for root, mem in cords.items():
            secs = sorted({s for s, _ in mem}, key=lambda s: z_of[s])
            if len(secs) < MIN_SECTIONS:
                continue
            cl = _centreline(mem); zs2 = sorted(cl)
            for s in order:
                z2 = z_of[s]
                if z2 <= zs2[0] or z2 >= zs2[-1] or z2 in cl:
                    continue
                lo = max(z for z in zs2 if z < z2); hi = min(z for z in zs2 if z > z2)
                w = (z2 - lo) / (hi - lo)
                c = cl[lo] + w * (cl[hi] - cl[lo])
                targets.append((s, float(c[0]), float(c[1])))
        for a1, b1, zlo, zhi, empty in pending:
            cla, clb = _centreline(cords[a1]), _centreline(cords[b1])
            at_a = _curve(cla, sorted(cla), True)[0]; at_b = _curve(clb, sorted(clb), False)[0]
            for s in empty:
                w = (z_of[s] - zlo) / (zhi - zlo)
                c = (1.0 - w) * at_a(z_of[s]) + w * at_b(z_of[s])
                targets.append((s, float(c[0]), float(c[1])))
        n_new, new_keys = 0, set()
        for s, cx, cy in targets:
            z2, xy, pr = data[s]
            near = (np.hypot(xy[:, 0] - cx, xy[:, 1] - cy) <= RESCUE_R_UM) & (pr >= tau_r)
            if int(near.sum()) < RESCUE_MIN_CELLS:
                continue
            m2 = np.zeros((grid["ny"], grid["nx"]), bool)
            xi = np.clip((xy[near, 0] / V16.RESOLUTION).astype(int), 0, grid["nx"] - 1)
            yi = np.clip((xy[near, 1] / V16.RESOLUTION).astype(int), 0, grid["ny"] - 1)
            m2[yi, xi] = True
            m2 = binary_dilation(m2, iterations=2)
            if any((m2 & mm).any() for mm in masks[s].values()):
                continue
            nid = (max(masks[s]) + 1) if masks[s] else 1
            masks[s][nid] = m2
            rescued[(s, nid)] = True
            new_keys.add((s, nid))
            stats[s]["n_regions"] = len(masks[s])
            n_new += 1
        print(f"transition fit: {len(targets)} empty planes looked at with tau {tau_r:.4f} "
              f"(posfrac x{RESCUE_POSFRAC_X:g}), {n_new} regions added", flush=True)
        return new_keys


    LINK_BRANCH_MIN = 0.6

    def relink_cells():
        node = {(s, i): (s, i) for s in order for i in masks[s]}

        def find(x):
            while node[x] != x:
                node[x] = node[node[x]]; x = node[x]
            return x

        def _bbox(m):
            ys, xs = np.nonzero(m)
            return (int(ys.min()), int(ys.max()) + 1, int(xs.min()), int(xs.max()) + 1)
        BB = {s: {i: _bbox(m) for i, m in masks[s].items()} for s in order}
        it = max(1, int(round(OVERLAP_DILATE_UM / V16.RESOLUTION)))
        H, W = grid["ny"], grid["nx"]
        n_links = n_branch = 0
        for s0, s1 in zip(order, order[1:]):
            cands = []
            for i, m0 in masks[s0].items():
                y0, y1, x0, x1 = BB[s0][i]
                y0, y1 = max(0, y0 - it), min(H, y1 + it); x0, x1 = max(0, x0 - it), min(W, x1 + it)
                d0 = binary_dilation(m0[y0:y1, x0:x1], iterations=it)
                a0 = int(m0.sum())
                for j, m1 in masks[s1].items():
                    b1 = BB[s1][j]
                    if b1[0] >= y1 or b1[1] <= y0 or b1[2] >= x1 or b1[3] <= x0:
                        continue
                    w1 = m1[y0:y1, x0:x1]
                    inter = int((d0 & w1).sum())
                    if inter:
                        fr = inter / max(1, min(a0, int(w1.sum())))
                        if fr >= LINK_OVERLAP_MIN:
                            cands.append((fr, i, j))
            cands.sort(key=lambda c: -c[0])
            up, down = set(), set()
            for fr, i, j in cands:
                taken = i in up or j in down
                if taken and fr < LINK_BRANCH_MIN:
                    continue
                up.add(i); down.add(j)
                node[find((s0, i))] = find((s1, j)); n_links += 1; n_branch += int(taken)
        cords = {}
        for k in node:
            cords.setdefault(find(k), []).append(k)
        gaps = [(order[k], order[k + 1], data[order[k + 1]][0] - data[order[k]][0])
                for k in range(len(order) - 1) if data[order[k + 1]][0] - data[order[k]][0] > 7.5]
        print(f"cell linking over {len(order) - 1} consecutive section pairs (tolerance {OVERLAP_DILATE_UM:g} um, "
              f"overlap >= {LINK_OVERLAP_MIN:g} of the smaller, strongest first, a branch needs >= {LINK_BRANCH_MIN:g}): "
              f"{n_links} links ({n_branch} branches), {len(cords)} cords; {len(gaps)} missing-section gaps treated as adjacent "
              f"{[round(g[2]) for g in gaps][:12]}", flush=True)
        return cords

    cords = relink_cells()


    down = a.down
    shape_d = (ch // down, cw // down)
    rep = {"class": "Schwann", "basis": "G_withdrawn", "tau": tau,
           "posfrac": POSFRAC,
           "how_defined": "Nerve region operator VERBATIM per section (imported, canvas-"
                          "pinned grid, tau at the cohort posfrac 0.003); a nerve is the "
                          "cells: regions joined only where their clusters overlap on "
                          f"consecutive measured sections ({OVERLAP_DILATE_UM:g} um tolerance, "
                          f">= {LINK_OVERLAP_MIN:g} of the smaller, strongest first); a missing "
                          "section is skipped as if the two around it were adjacent; a cord on "
                          f"fewer than {MIN_SECTIONS} sections is refused.",
           "constants_from": str(X16),
           "claim": "MODEL PREDICTION on H&E (pan-cancer Cell head, single fold). No "
                    "spatial ground truth in this cohort; nothing here is scored.",
           "sections": stats, "nerves": [], "rejected": []}
    sy = grid["ny"] / shape_d[0]


    def _mesh_one(item):
        idx, root, members = item
        secs = sorted({s for s, _ in members}, key=lambda s: data[s][0])
        if len(secs) < MIN_SECTIONS:
            return ("rej", {"n_sections": len(secs),
                                    "sections": secs,
                                    "why": f"fragment (<{MIN_SECTIONS} sections)"})


        _bb = [BB_ALL[s2][i2] for s2, i2 in members]
        _dx = (max(b[3] for b in _bb) - min(b[2] for b in _bb)) * V16.RESOLUTION
        _dy = (max(b[1] for b in _bb) - min(b[0] for b in _bb)) * V16.RESOLUTION
        _dz = data[secs[-1]][0] - data[secs[0]][0]
        _diag = float(np.sqrt(_dx ** 2 + _dy ** 2 + _dz ** 2))
        if _diag < MESH_MIN_DIAG_UM:
            return ("rej", {"n_sections": len(secs), "sections": secs, "extent_um": round(_diag, 1),
                            "why": f"below {MESH_MIN_DIAG_UM:g} um (not meshed)"})


        from scipy.ndimage import shift as _ndshift
        RES = V16.RESOLUTION
        zs_m, um_by_z, cen_by_z = [], {}, {}
        for s2, i2 in members:
            z2 = data[s2][0]
            um_by_z[z2] = masks[s2][i2] | um_by_z.get(z2, False)
        for z2, m2 in um_by_z.items():
            ys, xs = np.nonzero(m2)
            cen_by_z[z2] = np.array([ys.mean(), xs.mean()])
        zs_m = sorted(um_by_z)

        ys0 = min(int(np.nonzero(m2)[0].min()) for m2 in um_by_z.values())
        ys1 = max(int(np.nonzero(m2)[0].max()) for m2 in um_by_z.values())
        xs0 = min(int(np.nonzero(m2)[1].min()) for m2 in um_by_z.values())
        xs1 = max(int(np.nonzero(m2)[1].max()) for m2 in um_by_z.values())
        pad = int(round(120.0 / RES))
        y0, y1 = max(0, ys0 - pad), min(grid["ny"], ys1 + pad + 1)
        x0, x1 = max(0, xs0 - pad), min(grid["nx"], xs1 + pad + 1)
        hh2, ww2 = y1 - y0, x1 - x0
        sdf_by_z = {z2: BM.sdf(um_by_z[z2][y0:y1, x0:x1]) for z2 in zs_m}
        Z_STEP = BM.Z_STEP
        zg = np.arange(zs_m[0], zs_m[-1] + 0.5 * Z_STEP, Z_STEP, dtype=np.float32)
        vol = np.empty((len(zg), hh2, ww2), np.float32)
        uns = np.zeros(len(zg), bool)
        zs_a = np.asarray(zs_m, np.float32)
        for k2, z2 in enumerate(zg):
            j = int(np.searchsorted(zs_a, z2, "right")) - 1
            j = max(0, min(j, len(zs_a) - 2)) if len(zs_a) > 1 else 0
            za, zb = zs_a[j], zs_a[min(j + 1, len(zs_a) - 1)]
            span = float(zb - za)
            t = 0.0 if span <= 0 else min(1.0, max(0.0, float((z2 - za) / span)))
            ca, cb = cen_by_z[float(za)], cen_by_z[float(zb)]
            cc = (1.0 - t) * ca + t * cb
            fa = _ndshift(sdf_by_z[float(za)], cc - ca, order=1, mode="nearest")
            fb = _ndshift(sdf_by_z[float(zb)], cc - cb, order=1, mode="nearest")
            vol[k2] = (1.0 - t) * fa + t * fb
            if span > BM.GAP_UM and 0.0 < t < 1.0:
                uns[k2] = True
        m = BM.mesh_from(vol, zg, uns, RES / BM.UM_PX, smooth_vox=1.0)
        if m is None:
            return ("rej", {"n_sections": len(secs), "why": "empty isosurface"})

        m["verts"][:, 0] += x0 * RES
        m["verts"][:, 1] += y0 * RES
        name = f"_tmp_cord_{idx:04d}"
        e = BM.write_mesh(name, m, OUT)
        _ext = m["verts"].max(0) - m["verts"].min(0)
        e["extent_um"] = round(float(np.sqrt((_ext ** 2).sum())), 1)
        e["_root"] = root
        e.update({"structure": "nerve", "cell_class": "Schwann",
                  "n_sections": len(secs), "z_min_um": data[secs[0]][0],
                  "z_max_um": data[secs[-1]][0],
                  "n_regions": len(members), "high_confidence": len(secs) >= 3,
                  "n_rescued": int(sum(1 for mm in members if rescued.get(mm)))})


        _v3, _f3 = m["verts"], m["faces"]
        _p0, _p1, _p2 = _v3[_f3[:, 0]], _v3[_f3[:, 1]], _v3[_f3[:, 2]]
        e["volume_um3"] = round(float(abs(
            np.einsum("ij,ij->i", _p0.astype(np.float64),
                      np.cross(_p1.astype(np.float64),
                               _p2.astype(np.float64))).sum()) / 6.0), 1)
        from scipy.ndimage import distance_transform_edt as _edt, zoom as _zoom
        _best = None
        for _z2, _m2 in um_by_z.items():
            _mm = _zoom(_m2.astype(np.uint8), 2, order=0) > 0
            _d2 = _edt(_mm)
            _r = float(_d2.max())
            if _r > 0 and (_best is None or _r > _best[0]):
                _yy, _xx = np.unravel_index(int(_d2.argmax()), _d2.shape)
                _best = (_r, float(_z2), _yy, _xx, _mm)
        if _best:
            _r, _zb, _yy, _xx, _mm = _best
            _half = RES / 2.0
            _dia = 2.0 * _r * _half
            _ys, _xs = np.nonzero(_mm)
            _pts = np.column_stack([_xs - _xs.mean(), _ys - _ys.mean()])
            _ev, _evec = np.linalg.eigh(_pts.T @ _pts / max(1, len(_pts)))
            _mn = _evec[:, 0]
            _cx, _cy = _xx * _half, _yy * _half
            _hx, _hy = _mn[0] * _dia / 2.0, _mn[1] * _dia / 2.0
            e.update({"max_diameter_um": round(_dia, 1),
                      "max_diameter_z_um": _zb,
                      "max_diameter_line": [
                          [round(_cx - _hx, 1), round(_cy - _hy, 1), _zb],
                          [round(_cx + _hx, 1), round(_cy + _hy, 1), _zb]]})
        e["members"] = [[s2, int(i2)] for s2, i2 in members]
        print(f"  cord {name}: {len(secs)} sections z {data[secs[0]][0]:.0f}-"
              f"{data[secs[-1]][0]:.0f} um, {len(members)} regions", flush=True)
        return ("ok", e)
    globals()["_MESH_ONE"] = _mesh_one
    _items = [(k4, root, members) for k4, (root, members) in
              enumerate(sorted(cords.items(), key=lambda kv: -len(kv[1])), 1)]
    kept = 0
    import multiprocessing as _mp

    MESH_MIN_DIAG_UM = 0.5 * MIN_DIAG_UM
    def _bbx(m):
        ys, xs = np.nonzero(m)
        return (int(ys.min()), int(ys.max()) + 1, int(xs.min()), int(xs.max()) + 1)
    BB_ALL = {s2: {i2: _bbx(m2) for i2, m2 in masks[s2].items()} for s2 in order}
    with _mp.get_context("fork").Pool(min(32, _mp.cpu_count())) as _pool:
        for _tag, _val in _pool.imap_unordered(_mesh_call, _items):
            if _tag == "ok":
                rep["nerves"].append(_val); kept += 1
            else:
                rep["rejected"].append(_val)
    rep["n_nerves"] = kept


    rep["nerves"].sort(key=lambda n: -n["extent_um"])
    id_of_root = {}
    for k3, e3 in enumerate(rep["nerves"], 1):
        nid = f"nerve-{k3:02d}"
        old = OUT / e3["file"]
        new = OUT / f"{nid}.bin"
        old.replace(new)
        e3["name"], e3["file"] = nid, new.name
        id_of_root[e3.pop("_root")] = nid
    region_id = {}
    for root2, members in cords.items():
        nid = id_of_root.get(root2)
        if nid is None:
            continue
        for s, i in members:
            region_id[(s, i)] = nid


    kept_regions = {}
    for root2, members in cords.items():
        if root2 not in id_of_root:
            continue
        for s, i in members:
            kept_regions.setdefault(s, []).append(i)
    ndir = {b: (VOL / b / "nerve") for b in ("G_withdrawn", "G_not_withdrawn")}
    for d in ndir.values():
        d.mkdir(parents=True, exist_ok=True)
    plane_rows = []
    idx_of = {p["section_id"]: p for p in planes}
    for sid2, ids in kept_regions.items():
        m10 = np.zeros((grid["ny"], grid["nx"]), bool)
        for i in ids:
            m10 |= masks[sid2][i]


        fill = cv2.resize((m10 * 255).astype(np.uint8), (cw, ch),
                          interpolation=cv2.INTER_LINEAR) > 127


        er = cv2.erode(fill.astype(np.uint8), np.ones((7, 7), np.uint8)) > 0
        rim = fill & ~er
        a8 = np.zeros((ch, cw), np.uint8)
        a8[fill] = 60
        a8[rim] = 255
        rgba = np.dstack([np.full_like(a8, 255)] * 3 + [a8])
        pidx = idx_of[sid2]["index"]
        fn = f"z{pidx:02d}_{sid2.split('-')[1]}.nerve.webp"
        from PIL import Image
        for b, d in ndir.items():
            Image.fromarray(rgba).save(d / fn, format="WEBP", lossless=True,
                                       quality=100, method=4)


        regs = []
        for i in ids:
            ys, xs = np.nonzero(masks[sid2][i])
            regs.append({"id": region_id[(sid2, i)],
                         "cx_um": round(float(xs.mean()) * V16.RESOLUTION, 1),
                         "cy_um": round(float(ys.mean()) * V16.RESOLUTION, 1),
                         "bbox_um": [int(xs.min()) * V16.RESOLUTION, int(ys.min()) * V16.RESOLUTION,
                                     (int(xs.max()) + 1) * V16.RESOLUTION, (int(ys.max()) + 1) * V16.RESOLUTION]})
        plane_rows.append({"index": pidx, "section_id": sid2,
                           "z_um": data[sid2][0], "n_regions": len(ids),
                           "file": fn, "regions": regs})
    (VOL / "nerve_regions_metadata.json").write_text(json.dumps({
        "what": "Nerve operator regions per section, DENOISED: only regions that "
                "belong to a cord spanning >= %d measured sections" % MIN_SECTIONS,
        "colour": "#00B0F0", "tau": tau, "posfrac": POSFRAC,
        "n_planes": len(plane_rows), "planes": sorted(plane_rows, key=lambda r: r["index"]),
        "claim": rep["claim"]}, indent=1))
    print(f"denoised layer: {len(plane_rows)} planes with kept regions", flush=True)
    (OUT.parent / f"Schwann_nerves_{a.sample}_full.json").write_text(
        json.dumps(rep, indent=1))
    write_display(rep, OUT, MIN_DIAG_UM)
    print(f"wrote {OUT}: {kept} cords "
          f"({sum(len(v) for v in cords.values())} regions in {len(cords)} components)", flush=True)

if __name__ == "__main__":
    main()
