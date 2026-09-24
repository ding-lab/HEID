#!/usr/bin/env python3

import json
import os
import struct
import sys

import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter

P3D = os.environ.get("THREED_ROOT", os.path.join(os.environ.get("PROJECTS_ROOT", "/data/heid"), "3d"))
OBJ = os.path.join(P3D, "objects_3d")
CFG = os.environ.get("THREED_LINKAGE", os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "configs", "linkage.json"))
OUT = os.path.join(OBJ, "outputs/S9_tls_objects")
VOL = os.path.join(P3D, "reconstruction/volume_8um")
TLS_TABLE = os.path.join(P3D, "tls_define/tls_3d_table.tsv")

WIN_UM = 600.0
MAX_SHIFT_UM = 200.0


def mutual_links(a, c, tol, dz=0.0):
    if not len(a) or not len(c):
        return [], 0
    A = a[["cx_um_canvas", "cy_um_canvas"]].to_numpy()
    C = c[["cx_um_canvas", "cy_um_canvas"]].to_numpy()
    D = np.hypot(A[:, None, 0] - C[None, :, 0], A[:, None, 1] - C[None, :, 1])
    ra = np.sqrt(a["core_area_mm2"].fillna(0).to_numpy() * 1e6 / np.pi)
    rc = np.sqrt(c["core_area_mm2"].fillna(0).to_numpy() * 1e6 / np.pi)
    out = []
    for i in range(len(A)):
        j = int(np.argmin(D[i]))
        tol_ij = max(tol + 2.0 * dz, 0.8 * (ra[i] + rc[j]))
        if D[i, j] > tol_ij:
            continue
        if int(np.argmin(D[:, j])) != i:
            continue
        second = np.sort(D[i])[1] if D.shape[1] > 1 else np.inf
        margin = float(second / D[i, j]) if D[i, j] > 0 else np.inf
        out.append((i, j, float(D[i, j]), margin))
    return out, int(D.size)


def load_layer_field(vmeta, basis, section_id, res, ny, nx):
    f = [x for x in vmeta["files"] if x["basis"] == basis and x["section_id"] == section_id][0]
    raw = open(os.path.join(VOL, f["file"]), "rb").read()
    magic, n, cw, ch, sub, ncls, _ = struct.unpack("<8sIHHHHI", raw[:24])
    xy = np.frombuffer(raw[32:32 + 4 * n], dtype="<u2").reshape(n, 2).astype(np.float64) / sub * 8.0
    g = np.zeros((ny, nx), np.float32)
    ix = np.clip((xy[:, 0] / res).astype(int), 0, nx - 1)
    iy = np.clip((xy[:, 1] / res).astype(int), 0, ny - 1)
    np.add.at(g, (iy, ix), 1.0)
    return gaussian_filter(g, sigma=2.0)


def local_shift(fa, fb, cx, cy, res, win_um=WIN_UM, max_um=MAX_SHIFT_UM):
    w = int(win_um / res)
    ny, nx = fa.shape
    x0, x1 = int(cx / res) - w, int(cx / res) + w
    y0, y1 = int(cy / res) - w, int(cy / res) + w
    if x0 < 0 or y0 < 0 or x1 >= nx or y1 >= ny:
        return None, None, None
    ref = fa[y0:y1, x0:x1]
    if ref.std() < 1e-6:
        return None, None, None
    m = int(max_um / res)
    best, bd = -2.0, (0, 0)
    for dy in range(-m, m + 1):
        for dx in range(-m, m + 1):
            mov = fb[y0 + dy:y1 + dy, x0 + dx:x1 + dx]
            if mov.shape != ref.shape or mov.std() < 1e-6:
                continue
            r = float(np.corrcoef(ref.ravel(), mov.ravel())[0, 1])
            if r > best:
                best, bd = r, (dx, dy)
    return bd[0] * res, bd[1] * res, best


def main(do_residual=True):
    global VOL, TLS_TABLE, OUT, SAMPLE
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-residual", action="store_true")
    ap.add_argument("--sample", default="HT891Z1")
    ap.add_argument("--volume", default="")
    ap.add_argument("--table", default="")
    a = ap.parse_args()
    SAMPLE = a.sample
    if a.volume:
        VOL = a.volume
    if a.table:
        TLS_TABLE = a.table
    _obj3d = os.environ.get("HTAN3D_OBJ3D", "")
    if _obj3d:
        OUT = os.path.join(_obj3d, "S9_tls_objects")
    elif SAMPLE != "HT891Z1":
        OUT = os.path.join(os.path.dirname(OUT), f"S9_tls_objects_{SAMPLE}")
    do_residual = do_residual and not a.no_residual
    os.makedirs(OUT, exist_ok=True)
    cfg = json.load(open(CFG))
    rules = cfg["tls_linking"]
    tol = 150.0
    res = cfg["grid"]["resolution_um"]
    basis = cfg["grid"]["canvas_basis_891"]

    t3 = pd.read_csv(TLS_TABLE, sep="\t", comment="#")
    vmeta = json.load(open(os.path.join(VOL, "cell_points_metadata.json")))
    volmeta = json.load(open(os.path.join(VOL, "metadata.json")))
    zs = volmeta["z_um"]
    sec_by_z = {volmeta["z_um"][s["index"]]: s["section_id"] for s in vmeta["sections"]}
    cw, ch = vmeta["canvas_px"]["width"], vmeta["canvas_px"]["height"]
    nx, ny = int(np.ceil(cw * 8.0 / res)), int(np.ceil(ch * 8.0 / res))

    b = t3[(t3.sample_id == SAMPLE) & t3.cx_um_canvas.notna()].reset_index(drop=True)
    b["gid"] = b.index
    allz = sorted(zs)


    tls_z = sorted(b.z_um.unique())
    zpos = {z: k for k, z in enumerate(allz)}
    pairs = [(z0, z1) for z0, z1 in zip(tls_z, tls_z[1:])
             if zpos.get(z1, 0) - zpos.get(z0, 0) <= 3]


    link_rows, edges = [], []
    fields = {}
    for z0, z1 in pairs:
        a, c = b[b.z_um == z0], b[b.z_um == z1]
        lk, ncand = mutual_links(a, c, tol, dz=z1 - z0)
        if not lk:
            continue
        if do_residual:
            for z in (z0, z1):
                if z not in fields:


                    fields[z] = (load_layer_field(vmeta, basis, sec_by_z[z], res, ny, nx)
                                 if z in sec_by_z else None)
        for i, j, d, margin in lk:
            ra, rc = a.iloc[i], c.iloc[j]
            row = {"z_lo": z0, "z_hi": z1, "dz_um": z1 - z0,
                   "gid_lo": int(ra.gid), "gid_hi": int(rc.gid),
                   "tls_lo": ra.tls_id, "tls_hi": rc.tls_id,
                   "slide_lo": ra.slide_id, "slide_hi": rc.slide_id,
                   "dist_um": round(d, 1), "link_margin": round(margin, 2),
                   "gap_reliable": bool((z1 - z0) < 25),
                   "n_candidate_pairs": ncand}
            if do_residual:
                dx, dy, r = (None, None, None)
                if fields[z0] is not None and fields[z1] is not None:
                    dx, dy, r = local_shift(fields[z0], fields[z1],
                                            (ra.cx_um_canvas + rc.cx_um_canvas) / 2,
                                            (ra.cy_um_canvas + rc.cy_um_canvas) / 2, res)
                if dx is not None:
                    corr_d = float(np.hypot(rc.cx_um_canvas - dx - ra.cx_um_canvas,
                                            rc.cy_um_canvas - dy - ra.cy_um_canvas))
                    row.update({"local_shift_dx_um": dx, "local_shift_dy_um": dy,
                                "local_shift_mag_um": round(float(np.hypot(dx, dy)), 1),
                                "local_ncc": round(r, 4),
                                "dist_after_local_shift_um": round(corr_d, 1)})
                else:
                    row.update({"local_shift_dx_um": None, "local_shift_dy_um": None,
                                "local_shift_mag_um": None, "local_ncc": None,
                                "dist_after_local_shift_um": None})
            link_rows.append(row)
            edges.append((int(ra.gid), int(rc.gid)))
    links = pd.DataFrame(link_rows)
    links.to_csv(os.path.join(OUT, "links.tsv"), sep="\t", index=False)


    parent = {g: g for g in b.gid}

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for u, v in edges:
        parent[find(u)] = find(v)
    groups = {}
    for g in b.gid:
        groups.setdefault(find(g), []).append(g)


    BRIDGE_MAX_DZ_UM = 120.0
    BRIDGE_AREA_RATIO_MIN = 0.3
    BRIDGE_MAX_EMPTY_PLANES = 2
    ends = {}
    for root, members in groups.items():
        mm = b.loc[members].sort_values("z_um")
        if mm.z_um.nunique() >= 2:
            ends[root] = (mm.iloc[0], mm.iloc[-1])
    cand = {}
    for ra, (_, top) in ends.items():
        for rb, (bot, _) in ends.items():
            if ra == rb:
                continue
            dz = float(bot.z_um - top.z_um)
            if dz <= 0 or dz > BRIDGE_MAX_DZ_UM:
                continue
            d = float(np.hypot(bot.cx_um_canvas - top.cx_um_canvas, bot.cy_um_canvas - top.cy_um_canvas))
            r1 = np.sqrt(max(top.core_area_mm2, 0) * 1e6 / np.pi)
            r2 = np.sqrt(max(bot.core_area_mm2, 0) * 1e6 / np.pi)
            if d > max(tol + 2.0 * dz, 0.8 * (r1 + r2)):
                continue
            a1, a2 = float(top.core_area_mm2), float(bot.core_area_mm2)
            if min(a1, a2) / max(a1, a2, 1e-9) < BRIDGE_AREA_RATIO_MIN:
                continue
            m_empty = sum(1 for z in allz if top.z_um < z < bot.z_um)
            cand[(ra, rb)] = (d, dz, top, bot, m_empty)


    pd.DataFrame(columns=["slide_lo", "tls_lo", "z_lo", "slide_hi", "tls_hi", "z_hi", "dist_um", "dz_um",
                          "n_empty_planes", "accepted"]).to_csv(os.path.join(OUT, "bridge_candidates.tsv"), sep="\t", index=False)
    pd.DataFrame([{"slide_lo": t.slide_id, "tls_lo": t.tls_id, "z_lo": t.z_um,
                   "slide_hi": bt.slide_id, "tls_hi": bt.tls_id, "z_hi": bt.z_um,
                   "dist_um": round(d, 1), "dz_um": dz, "n_empty_planes": m,
                   "accepted": m <= BRIDGE_MAX_EMPTY_PLANES}
                  for (d, dz, t, bt, m) in cand.values()]).to_csv(
        os.path.join(OUT, "bridge_candidates.tsv"), sep="\t", index=False) if cand else None
    cand = {k: v[:4] for k, v in cand.items() if v[4] <= BRIDGE_MAX_EMPTY_PLANES}
    bridges = []
    for (ra, rb), (d, dz, top, bot) in cand.items():
        up = sorted(v[0] for (x, _y), v in cand.items() if x == ra)
        dn = sorted(v[0] for (_x, y), v in cand.items() if y == rb)
        if up[0] != d or dn[0] != d:
            continue
        second = min([u for u in up[1:]] + [v for v in dn[1:]] + [np.inf])
        margin = float(second / d) if d > 0 else np.inf
        bridges.append((ra, rb, d, dz, top, bot, margin))
    for ra, rb, d, dz, top, bot, margin in bridges:
        link_rows.append({"z_lo": float(top.z_um), "z_hi": float(bot.z_um), "dz_um": dz,
                          "gid_lo": int(top.gid), "gid_hi": int(bot.gid),
                          "tls_lo": top.tls_id, "tls_hi": bot.tls_id,
                          "slide_lo": top.slide_id, "slide_hi": bot.slide_id,
                          "dist_um": round(d, 1), "link_margin": round(margin, 2),
                          "gap_reliable": bool(dz < 25), "n_candidate_pairs": len(cand),
                          "bridge": True})
        edges.append((int(top.gid), int(bot.gid)))
        parent[find(int(top.gid))] = find(int(bot.gid))
    if bridges:
        links = pd.DataFrame(link_rows)
        links["bridge"] = links["bridge"].fillna(False).astype(bool)
        links.to_csv(os.path.join(OUT, "links.tsv"), sep="\t", index=False)
        groups = {}
        for g in b.gid:
            groups.setdefault(find(g), []).append(g)
    print(f"column bridges: {len(bridges)} "
          + "; ".join(f"{t.slide_id}:{t.tls_id} -> {bt.slide_id}:{bt.tls_id} d={d:.0f} dz={dz:.0f}"
                      for _a, _b, d, dz, t, bt, _m in bridges))


    JUNCTION_TOL_UM = 50.0
    JUNCTION_MAX_DIST_UM = 800.0
    JUNCTION_MIN_AREA_MM2 = 0.04
    tlsroot_j = os.environ.get("HTAN3D_TLSROOT", "")
    pred_j = os.environ.get("HTAN3D_PRED", "")
    junctions = []
    if tlsroot_j and pred_j:
        import importlib.util
        from scipy.spatial import cKDTree
        _sp = importlib.util.spec_from_file_location(
            "tls_rescue_targets", os.path.join(os.path.dirname(os.path.abspath(__file__)), "tls_rescue_targets.py"))
        RT = importlib.util.module_from_spec(_sp); _sp.loader.exec_module(RT)
        maps_j, fps_j = {}, {}

        def footprint(slide, tls_id):
            key = (slide, tls_id)
            if key not in fps_j:
                if slide not in maps_j:
                    maps_j[slide] = RT.canvas_map(VOL, pred_j, slide, basis)
                pts = (RT.core_footprint_canvas(tlsroot_j, slide, tls_id, maps_j[slide])
                       if maps_j[slide] is not None else None)
                fps_j[key] = cKDTree(pts) if pts is not None and len(pts) else None
            return fps_j[key]

        ends = set()
        for root, members in groups.items():
            mm = b.loc[members]
            ends.add(int(mm.z_um.idxmin())); ends.add(int(mm.z_um.idxmax()))
        linked = {(u, v) for u, v in edges} | {(v, u) for u, v in edges}
        for z0, z1 in pairs:
            a, c = b[b.z_um == z0], b[b.z_um == z1]
            for i in range(len(a)):
                for j in range(len(c)):
                    ra, rc = a.iloc[i], c.iloc[j]
                    gi, gj = int(ra.gid), int(rc.gid)
                    if (gi, gj) in linked or find(gi) == find(gj):
                        continue
                    if min(float(ra.core_area_mm2), float(rc.core_area_mm2)) < JUNCTION_MIN_AREA_MM2:
                        continue
                    if gi not in ends and gj not in ends:
                        continue
                    d = float(np.hypot(rc.cx_um_canvas - ra.cx_um_canvas, rc.cy_um_canvas - ra.cy_um_canvas))
                    if d > JUNCTION_MAX_DIST_UM:
                        continue
                    hit = False
                    for p, q in ((ra, rc), (rc, ra)):
                        t = footprint(q.slide_id, q.tls_id)
                        if t is not None and t.query([p.cx_um_canvas, p.cy_um_canvas])[0] <= JUNCTION_TOL_UM:
                            hit = True
                            break
                    if not hit:
                        continue
                    link_rows.append({"z_lo": z0, "z_hi": z1, "dz_um": z1 - z0,
                                      "gid_lo": gi, "gid_hi": gj, "tls_lo": ra.tls_id, "tls_hi": rc.tls_id,
                                      "slide_lo": ra.slide_id, "slide_hi": rc.slide_id,
                                      "dist_um": round(d, 1), "link_margin": 99.0,
                                      "gap_reliable": bool((z1 - z0) < 25), "n_candidate_pairs": len(a) * len(c),
                                      "junction": True})
                    edges.append((gi, gj)); linked.add((gi, gj)); linked.add((gj, gi))
                    parent[find(gi)] = find(gj)
                    junctions.append((ra, rc, d))
        if junctions:
            links = pd.DataFrame(link_rows)
            for col in ("bridge", "junction"):
                if col in links:
                    links[col] = links[col].fillna(False).astype(bool)
            links.to_csv(os.path.join(OUT, "links.tsv"), sep="\t", index=False)
            groups = {}
            for g in b.gid:
                groups.setdefault(find(g), []).append(g)
    print(f"junction links: {len(junctions)} "
          + "; ".join(f"{p.slide_id}:{p.tls_id} -> {q.slide_id}:{q.tls_id} d={d:.0f}" for p, q, d in junctions))


    LOBE_HALO_UM = 150.0
    LOBE_OVERLAP_MIN = 0.50
    LOBE_SECTION_FRAC = 0.5
    LOBE_MIN_SECTIONS = 2
    tlsroot = os.environ.get("HTAN3D_TLSROOT", "")
    _npz = {}
    _ov = {}

    def region_overlap(slide, ta, tb):
        key = (slide, str(ta), str(tb))
        if key in _ov:
            return _ov[key]
        _ov[key] = _region_overlap(slide, ta, tb)
        return _ov[key]

    def _halos_of(slide):
        if slide in _npz:
            return _npz[slide]
        f = os.path.join(tlsroot, slide, f"{slide}_masks.npz")
        if not os.path.exists(f):
            _npz[slide] = None
            return None
        from scipy.ndimage import binary_dilation
        z = np.load(f)
        cores, res = z["cores"], float(z["resolution"])
        r = max(1, int(round(LOBE_HALO_UM / res)))
        yy, xx = np.ogrid[-r:r + 1, -r:r + 1]
        disk = (xx * xx + yy * yy) <= r * r
        halos = []
        for core in cores:
            ys, xs = np.nonzero(core)
            if len(ys) == 0:
                halos.append(None)
                continue
            y0, y1 = max(0, ys.min() - r), min(core.shape[0], ys.max() + r + 1)
            x0, x1 = max(0, xs.min() - r), min(core.shape[1], xs.max() + r + 1)
            m = binary_dilation(core[y0:y1, x0:x1], structure=disk)
            halos.append((y0, y1, x0, x1, m, int(m.sum())))
        _npz[slide] = halos
        return halos

    def _region_overlap(slide, ta, tb):
        if not tlsroot:
            return None
        halos = _halos_of(slide)
        if halos is None:
            return None
        ia, ib = int(str(ta).split("-")[-1]) - 1, int(str(tb).split("-")[-1]) - 1
        if ia >= len(halos) or ib >= len(halos):
            return None
        A, B = halos[ia], halos[ib]
        if A is None or B is None:
            return 0.0
        ay0, ay1, ax0, ax1, am, an = A
        by0, by1, bx0, bx1, bm, bn = B
        y0, y1, x0, x1 = max(ay0, by0), min(ay1, by1), max(ax0, bx0), min(ax1, bx1)
        if y1 <= y0 or x1 <= x0:
            return 0.0
        inter = (am[y0 - ay0:y1 - ay0, x0 - ax0:x1 - ax0] & bm[y0 - by0:y1 - by0, x0 - bx0:x1 - bx0]).sum()
        return float(inter / max(1, min(an, bn)))

    merges = []
    changed = True
    while changed:
        changed = False
        groups = {}
        for g in b.gid:
            groups.setdefault(find(g), []).append(g)
        roots = list(groups)


        zmap = {r: {row.z_um: row for row in b.loc[groups[r]].itertuples()} for r in roots}
        for i in range(len(roots)):
            for j in range(i + 1, len(roots)):
                ra, rb = roots[i], roots[j]
                za, zb = zmap[ra], zmap[rb]
                shared = sorted(set(za) & set(zb))
                if not shared:
                    continue
                ov = [region_overlap(za[z].slide_id, za[z].tls_id, zb[z].tls_id) for z in shared]
                ov = [o for o in ov if o is not None]
                n_touch = sum(o >= LOBE_OVERLAP_MIN for o in ov)
                if n_touch >= LOBE_MIN_SECTIONS and n_touch >= int(np.ceil(LOBE_SECTION_FRAC * len(shared))):
                    merges.append((za[shared[0]].slide_id, za[shared[0]].tls_id,
                                   zb[shared[0]].slide_id, zb[shared[0]].tls_id, len(shared), n_touch,
                                   float(np.median(ov)) if ov else 0.0))
                    parent[find(ra)] = find(rb)
                    changed = True
                    break
            if changed:
                break
    groups = {}
    for g in b.gid:
        groups.setdefault(find(g), []).append(g)
    print(f"lobe merges: {len(merges)} "
          + "; ".join(f"{a}:{ta} + {c}:{tc} ({nt}/{ns} shared sections overlap, median {mo:.2f})"
                      for a, ta, c, tc, ns, nt, mo in merges))


    rng = np.random.default_rng(20260819)
    tzs = sorted(b.z_um.unique())
    null_len = []
    for _ in range(1000):
        perm = list(tzs)
        rng.shuffle(perm)
        mp = dict(zip(tzs, perm))
        par = {g: g for g in b.gid}

        def f2(x):
            while par[x] != x:
                par[x] = par[par[x]]
                x = par[x]
            return x
        for z0, z1 in pairs:
            if z0 not in mp or z1 not in mp or mp[z0] == mp[z1]:
                continue
            a, c = b[b.z_um == z0], b[b.z_um == mp[z1]]
            lk, _ = mutual_links(a, c, tol, dz=z1 - z0)
            for i, j, _d, _m in lk:
                par[f2(int(a.iloc[i].gid))] = f2(int(c.iloc[j].gid))
        gg = {}
        for g in b.gid:
            gg.setdefault(f2(g), []).append(g)
        null_len.append(max((len(set(b.loc[v, "z_um"])) for v in gg.values()), default=0))
    null_len = np.array(null_len)

    obj_rows = []
    for k, (root, members) in enumerate(sorted(groups.items(),
                                               key=lambda kv: -len(kv[1])), 1):
        mem = b[b.gid.isin(members)].sort_values("z_um")
        zsm = sorted(mem.z_um.unique())
        ml = links[links.gid_lo.isin(members) & links.gid_hi.isin(members)]
        n_meas = len(zsm)
        dzs = ml.dz_um.tolist()
        margins = ml.link_margin.tolist()
        crosses_big_gap = any(d >= 25 for d in dzs)
        frac_ge2 = float(np.mean([mg >= 2 for mg in margins])) if margins else 0.0
        if n_meas == 1:
            tier = "C"
        elif n_meas >= 4 and frac_ge2 >= 0.75:
            tier = "A"
        else:
            tier = "B"
        obj_rows.append({


            "object_id": f"tls-{k:02d}", "n_measured_sections": n_meas,
            "z_min_um": min(zsm), "z_max_um": max(zsm),
            "z_extent_um": max(zsm) - min(zsm),
            "n_cores": len(members),
            "member_tls": ";".join(f"{r.slide_id}:{r.tls_id}" for _, r in mem.iterrows()),
            "cx_um_canvas": round(float(mem.cx_um_canvas.mean()), 1),
            "cy_um_canvas": round(float(mem.cy_um_canvas.mean()), 1),
            "core_area_mm2_median": round(float(mem.core_area_mm2.median()), 4),
            "link_dist_um_median": (round(float(np.median(ml.dist_um)), 1) if len(ml) else None),
            "link_margin_min": (round(float(min(margins)), 2) if margins else None),
            "link_margin_frac_ge2": round(frac_ge2, 3),
            "max_dz_um": (max(dzs) if dzs else None),
            "crosses_gap_ge25um": bool(crosses_big_gap),
            "n_bridges": int(ml["bridge"].sum()) if "bridge" in ml else 0,
            "n_lobes_max": int(mem.groupby("z_um").size().max()),
            "n_rescued": int(mem["rescued"].astype(str).isin(["True", "true", "1"]).sum()) if "rescued" in mem else 0,
            "p_link": round(float((null_len >= n_meas).mean()), 4),
            "tier": tier,
        })
    objs = pd.DataFrame(obj_rows)
    objs.to_csv(os.path.join(OUT, "objects.tsv"), sep="\t", index=False)

    multi = objs[objs.n_measured_sections >= 2]
    ge4 = objs[objs.n_measured_sections >= 4]
    verdict = {
        "tolerance_um": tol,
        "rule": rules["accept_link_iff_both"],
        "n_cores_input": int(len(b)),
        "n_links_accepted": int(len(links)),
        "n_objects_total": int(len(objs)),
        "n_objects_single_layer": int((objs.n_measured_sections == 1).sum()),
        "n_objects_multi_layer": int(len(multi)),
        "n_objects_spanning_ge4_sections": int(len(ge4)),
        "acceptance_from_plan": ">=5 objects spanning >=4 measured sections",
        "acceptance_met": bool(len(ge4) >= 5),
        "max_sections_in_one_object": int(objs.n_measured_sections.max()),
        "tier_counts": objs.tier.value_counts().to_dict(),
        "shuffled_null_max_chain_median": float(np.median(null_len)),
        "shuffled_null_max_chain_p95": float(np.percentile(null_len, 95)),
    }
    if do_residual and "dist_after_local_shift_um" in links.columns:
        ok = links.dist_after_local_shift_um.notna()
        verdict["residual_decomposition"] = {
            "n_links_with_estimate": int(ok.sum()),
            "raw_link_distance_median_um": round(float(links.dist_um.median()), 1),
            "local_field_shift_magnitude_median_um":
                round(float(links.loc[ok, "local_shift_mag_um"].median()), 1),
            "distance_after_removing_local_shift_median_um":
                round(float(links.loc[ok, "dist_after_local_shift_um"].median()), 1),
            "local_ncc_median": round(float(links.loc[ok, "local_ncc"].median()), 3),
            "reading": ("the local shift is what a rigid registration residual would move "
                        "the whole neighbourhood by. If the link distance after removing it "
                        "drops well below the raw distance, that part of the 76.6 um was "
                        "registration, and what is left is the object's own displacement."),
        }
    with open(os.path.join(OUT, "verdict.json"), "w") as fh:
        json.dump(verdict, fh, indent=2)
    print(json.dumps(verdict, indent=2), flush=True)
    print("\n" + objs.head(25).to_string(index=False), flush=True)
    return 0


SAMPLE = "HT891Z1"

if __name__ == "__main__":
    sys.exit(main())
