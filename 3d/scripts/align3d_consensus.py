#!/usr/bin/env python3

import argparse
import csv
import glob
import json
import os
from pathlib import Path

import numpy as np

HUBER_UM = 30.0
W_FLOOR_UM = 10.0
WEAK_MIN_N = 3
WEAK_MAX_RMS_UM = 40.0
WEAK_MAX_SCALE_DEV = 0.05

WEAK_MAX_ROT_DEG = 3.0
WEAK_EXTRA_UM = 40.0
AFFINE_MIN_N = int(os.environ.get("HTAN3D_AFFINE_MIN_N", 6))
ITERS = 5
CHAIN_GRID_UM = 2.0
PRIOR_W = 1e-6
RIGID_L_UM = 1000.0
RIGID_W = float(os.environ.get("HTAN3D_RIGID_W", 0.3))


def params_of(M):
    a, b, tx = M[0]; c, d, ty = M[1]
    sx = float(np.hypot(a, c)); rot = float(np.degrees(np.arctan2(c, a)))
    sh = float((a * b + c * d) / max(sx * sx, 1e-12)); sy = float((a * d - b * c) / max(sx, 1e-12))
    return {"scale_x": sx, "scale_y": sy, "rot_deg": rot, "shear": sh, "tx_um": float(tx), "ty_um": float(ty)}


def of_params(q):
    th = np.radians(q["rot_deg"])
    return np.array([[q["scale_x"] * np.cos(th), q["scale_x"] * np.cos(th) * q["shear"] - q["scale_y"] * np.sin(th), q["tx_um"]],
                     [q["scale_x"] * np.sin(th), q["scale_x"] * np.sin(th) * q["shear"] + q["scale_y"] * np.cos(th), q["ty_um"]]])


def compose(M2, M1):
    A2 = np.vstack([M2, [0, 0, 1]]); A1 = np.vstack([M1, [0, 0, 1]])
    return (A2 @ A1)[:2]


def invert(M):
    return np.linalg.inv(np.vstack([M, [0, 0, 1]]))[:2]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--volume", required=True)
    ap.add_argument("--pairs-dir", required=True, dest="pairs")
    ap.add_argument("--out", required=True)
    ap.add_argument("--no-weak", action="store_true", dest="no_weak",
                    help="use only pairs with a full core affine; pairs with 3-5 anchors are dropped")
    a = ap.parse_args()
    vol, out = Path(a.volume), Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    vm = json.loads((vol / "metadata.json").read_text())
    planes = sorted(vm["encodings"]["G_withdrawn"]["planes"], key=lambda p: p["z_um"])
    sids = [p["section_id"] for p in planes]
    z = {p["section_id"]: float(p["z_um"]) for p in planes}
    idx = {s: i for i, s in enumerate(sids)}
    origin = vm.get("canvas_origin_um") or [0.0, 0.0]
    cw, ch = vm["canvas_px"]["width"], vm["canvas_px"]["height"]
    um = float(vm["in_plane_um_per_px"])

    pairs, n_weak = [], 0
    for f in sorted(glob.glob(str(Path(a.pairs) / "*" / "pair_fits.json"))):
        for r in json.loads(Path(f).read_text()):
            if r["a"] not in idx or r["b"] not in idx:
                continue
            if r.get("affine_a_to_b_um") and len(r.get("anchors") or []) >= AFFINE_MIN_N:
                pairs.append(r)
            elif not a.no_weak and len(r.get("anchors") or []) >= WEAK_MIN_N:


                P = np.array([q["pa"] for q in r["anchors"]], float); Q = np.array([q["pb"] for q in r["anchors"]], float)
                mp, mq = P.mean(0), Q.mean(0); X, Y = P - mp, Q - mq
                U, S, Vt = np.linalg.svd(X.T @ Y); d = np.sign(np.linalg.det(U @ Vt))
                R = (U @ np.diag([1, d]) @ Vt).T; sc = (S * [1, d]).sum() / max((X ** 2).sum(), 1e-9); t = mq - sc * R @ mp
                res = np.linalg.norm((sc * (P @ R.T) + t) - Q, axis=1); rms = float(np.sqrt((res ** 2).mean()))
                rot = float(np.degrees(np.arctan2(R[1, 0], R[0, 0])))
                if rms <= WEAK_MAX_RMS_UM and abs(sc - 1.0) <= WEAK_MAX_SCALE_DEV and abs(rot) <= WEAK_MAX_ROT_DEG:
                    r = dict(r, affine_a_to_b_um=np.column_stack([sc * R, t]).tolist(), affine_rms_um=rms + WEAK_EXTRA_UM, weak=True)
                    pairs.append(r); n_weak += 1
                    print(f"  weak link {r['a'].split('-')[-1]}->{r['b'].split('-')[-1]}: {len(P)} anchors, similarity scale {sc:.3f} rot {rot:+.2f} deg rms {rms:.1f} um", flush=True)
                else:
                    print(f"  weak pair {r['a'].split('-')[-1]}->{r['b'].split('-')[-1]} rejected: {len(P)} anchors, scale {sc:.3f} rot {rot:+.2f} deg rms {rms:.1f} um", flush=True)
    if not pairs:
        raise SystemExit("no pair fits found")
    if n_weak:
        print(f"  {n_weak} pairs with {WEAK_MIN_N}-5 anchors enter as similarities (rms <= {WEAK_MAX_RMS_UM:g} um, weight lowered by {WEAK_EXTRA_UM:g} um)", flush=True)


    par = {s: s for s in sids}
    def find(u):
        while par[u] != u:
            par[u] = par[par[u]]; u = par[u]
        return u
    for r in pairs:
        par[find(r["a"])] = find(r["b"])
    comp = {}
    for r in pairs:
        comp.setdefault(find(r["a"]), set()).update((r["a"], r["b"]))
    comps = sorted((sorted(c, key=lambda s: idx[s]) for c in comp.values()), key=lambda c: idx[c[0]])


    gauges = [c[len(c) // 2] for c in comps]
    if len(comps) > 1:
        print("  the stack falls into " + str(len(comps)) + " groups no pair links: "
              + "; ".join(f"{c[0].split('-')[-1]}..{c[-1].split('-')[-1]} ({len(c)})" for c in comps), flush=True)
    measured = sorted({s for c in comps for s in c}, key=lambda s: idx[s])
    stray = []
    print(f"{len(sids)} sections, {len(pairs)} measured pairs "
          f"({sum(1 for r in pairs if abs(idx[r['a']] - idx[r['b']]) == 1)} adjacent, "
          f"{sum(1 for r in pairs if abs(idx[r['a']] - idx[r['b']]) == 2)} one-apart); "
          f"{len(sids) - len(measured)} sections without a measurement", flush=True)


    W, H = cw * um, ch * um
    P = np.array([[W / 2, H / 2], [0.2 * W, 0.2 * H], [0.8 * W, 0.2 * H], [0.2 * W, 0.8 * H], [0.8 * W, 0.8 * H]])
    n = len(sids)

    def rows_for(i, p):
        rx = np.zeros(6 * n); ry = np.zeros(6 * n)
        rx[6 * i:6 * i + 3] = [p[0], p[1], 1.0]
        ry[6 * i + 3:6 * i + 6] = [p[0], p[1], 1.0]
        return rx, ry


    mwc = [r.get("measured_with_corrections") for r in pairs]
    prev = next((m for m in mwc if m), None)
    if prev and any(m != prev for m in mwc if m is not None):
        raise SystemExit("pair fits were measured on different placements; re-run the anchors on one volume")
    ox_, oy_ = float(origin[0]), float(origin[1])
    def prev_lin(s):
        if not prev or s not in prev["sections"]:
            return np.eye(2)
        return np.asarray(prev["sections"][s], float)[:, :2]

    rms = np.array([float(r["affine_rms_um"] or 0.0) for r in pairs])
    w = 1.0 / (rms + W_FLOOR_UM)
    w = w / w.max()
    x = None
    for it in range(ITERS):
        rows, rhs, wts = [], [], []
        for k, r in enumerate(pairs):
            i, j = idx[r["a"]], idx[r["b"]]
            A = np.array(r["affine_a_to_b_um"], float)
            for p in P:
                q = A[:, :2] @ p + A[:, 2]
                rxi, ryi = rows_for(i, p); rxj, ryj = rows_for(j, q)
                rows.append(rxj - rxi); rhs.append(0.0); wts.append(w[k])
                rows.append(ryj - ryi); rhs.append(0.0); wts.append(w[k])

        for i in range(n):
            for kk, val in enumerate([1, 0, 0, 0, 1, 0]):
                row = np.zeros(6 * n); row[6 * i + kk] = 1.0
                rows.append(row); rhs.append(float(val)); wts.append(PRIOR_W)


        for i, s_ in enumerate(sids):
            Pm = prev_lin(s_); p11, p12 = Pm[0]; p21, p22 = Pm[1]
            for coef, val in (([p11, p21, p12, p22], 2.0), ([p11, p21, -p12, -p22], 0.0), ([p12, p22, p11, p21], 0.0)):
                row = np.zeros(6 * n)
                row[6 * i + 0], row[6 * i + 1], row[6 * i + 3], row[6 * i + 4] = [c_ * RIGID_L_UM for c_ in coef]
                rows.append(row); rhs.append(val * RIGID_L_UM); wts.append(RIGID_W)
        Rw = np.array(rows) * np.sqrt(np.array(wts))[:, None]
        bw = np.array(rhs) * np.sqrt(np.array(wts))


        ident = np.array([1.0, 0, 0, 0, 1.0, 0])
        mcols = np.concatenate([np.arange(6 * idx[g], 6 * idx[g] + 6) for g in gauges])
        keep = np.setdiff1d(np.arange(6 * n), mcols)
        bw = bw - Rw[:, mcols] @ np.tile(ident, len(gauges))
        xk = np.linalg.lstsq(Rw[:, keep], bw, rcond=None)[0]
        x = np.zeros(6 * n); x[keep] = xk; x[mcols] = np.tile(ident, len(gauges))
        res = []
        for k, r in enumerate(pairs):
            i, j = idx[r["a"]], idx[r["b"]]
            Ci = x[6 * i:6 * i + 6].reshape(2, 3); Cj = x[6 * j:6 * j + 6].reshape(2, 3)
            A = np.array(r["affine_a_to_b_um"], float)
            e = []
            for p in P:
                q = A[:, :2] @ p + A[:, 2]
                e.append(np.linalg.norm(Cj[:, :2] @ q + Cj[:, 2] - (Ci[:, :2] @ p + Ci[:, 2])))
            res.append(float(np.sqrt(np.mean(np.square(e)))))
        res = np.array(res)
        hub = np.where(res <= HUBER_UM, 1.0, HUBER_UM / np.maximum(res, 1e-9))
        w = (1.0 / (rms + W_FLOOR_UM)) * hub
        w = w / w.max()
        print(f"  iteration {it + 1}: pair residual median {np.median(res):.1f} um, "
              f"p90 {np.percentile(res, 90):.1f} um, {int((res > HUBER_UM).sum())} pairs down-weighted", flush=True)

    C = {s: x[6 * i:6 * i + 6].reshape(2, 3).copy() for s, i in idx.items()}


    if len(comps) > 1:
        main = max(comps, key=len)
        for c in comps:
            if c is main:
                continue
            nb = min(main, key=lambda s: min(abs(idx[s] - idx[c[0]]), abs(idx[s] - idx[c[-1]])))
            bs = min(c, key=lambda s: abs(idx[s] - idx[nb]))
            G = compose(C[nb], invert(C[bs]))
            for s in c:
                C[s] = compose(G, C[s])
            print(f"  group {c[0].split('-')[-1]}..{c[-1].split('-')[-1]} carried: its {bs.split('-')[-1]} "
                  f"takes {nb.split('-')[-1]}'s correction, the rest follows", flush=True)

    mz = np.array([z[s] for s in measured]); mp = [params_of(C[s]) for s in measured]
    for s in sids:
        if s in measured:
            continue
        q = {k: float(np.interp(z[s], mz, [pq[k] for pq in mp])) for k in mp[0]}
        C[s] = of_params(q)

    pr = {s: params_of(C[s]) for s in sids}
    med = {k: float(np.median([pr[s][k] for s in measured])) for k in pr[sids[0]]}
    Gi = invert(of_params(med))
    C = {s: compose(Gi, C[s]) for s in sids}
    print("  median correction removed (vote): " + ", ".join(f"{k} {v:.3f}" for k, v in med.items()), flush=True)


    raw = {sids[n // 2]: np.array([[1.0, 0, 0], [0, 1.0, 0]])}
    adj = {(r["a"], r["b"]): np.array(r["affine_a_to_b_um"], float) for r in pairs if abs(idx[r["a"]] - idx[r["b"]]) == 1}
    for i in range(n // 2, n - 1):
        A = adj.get((sids[i], sids[i + 1]))
        raw[sids[i + 1]] = compose(invert(A), raw[sids[i]]) if A is not None else raw[sids[i]]
    for i in range(n // 2, 0, -1):
        A = adj.get((sids[i - 1], sids[i]))
        raw[sids[i - 1]] = compose(A, raw[sids[i]]) if A is not None else raw[sids[i]]

    ox, oy = float(origin[0]), float(origin[1])
    To = np.array([[1.0, 0, ox], [0, 1.0, oy]]); Ti = np.array([[1.0, 0, -ox], [0, 1.0, -oy]])
    world = {s: compose(To, compose(C[s], Ti)) for s in sids}


    if prev:
        for s in sids:
            if s in prev["sections"]:
                world[s] = compose(world[s], np.asarray(prev["sections"][s], float))
        print("  composed onto the corrections the pairs were measured with", flush=True)
    else:
        print("  pairs measured on the chain as solved: no composition", flush=True)
    pr2 = {s: params_of(compose(Ti, compose(world[s], To))) for s in sids}
    keep = {}
    old_f = out / "section_corrections.json"
    if old_f.exists():
        od = json.loads(old_f.read_text())
        keep = {k: od[k] for k in ("swap_to_main_residual", "swap_frame_cells", "swap_residual_source") if k in od}
    (out / "section_corrections.json").write_text(json.dumps({
        "what": "per-section affine correction in WORLD um (x' = M [x y 1]), composed onto the "
                "registration chain as solved; consensus of the duct-anchor pair fits, median section = identity",
        "chain_grid_um": CHAIN_GRID_UM, "canvas_origin_um": [ox, oy], **keep,
        "n_pairs": len(pairs), "unmeasured_sections": [s for s in sids if s not in measured],
        "groups": [[c[0], c[-1], len(c)] for c in comps],
        "sections": {s: world[s].tolist() for s in sids}}, indent=1))
    with open(out / "section_corrections.tsv", "w", newline="") as fh:
        wri = csv.writer(fh, delimiter="\t")
        wri.writerow(["section_id", "z_um", "measured", "scale_x", "scale_y", "rot_deg", "shear", "tx_um", "ty_um",
                      "raw_scale_x", "raw_scale_y", "raw_rot_deg", "raw_tx_um", "raw_ty_um"])
        for s in sids:
            q = pr2[s]; rq = params_of(raw[s])
            wri.writerow([s, z[s], int(s in measured)]
                         + [round(q[k], 4) for k in ("scale_x", "scale_y", "rot_deg", "shear", "tx_um", "ty_um")]
                         + [round(rq[k], 4) for k in ("scale_x", "scale_y", "rot_deg", "tx_um", "ty_um")])
    try:
        import matplotlib; matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        zs = [z[s] for s in sids]
        fig, ax = plt.subplots(3, 1, figsize=(11, 9), sharex=True)
        for key, lab, axi in (("scale_x", "scale x", ax[0]), ("rot_deg", "rotation (deg)", ax[1]), ("tx_um", "shift x (um)", ax[2])):
            axi.plot(zs, [params_of(raw[s])[key] for s in sids], "o-", color="#bbbbbb", label="raw chain (cumulative adjacent fits)")
            axi.plot(zs, [pr2[s][key] for s in sids], "o-", color="#0077cc", label="consensus correction")
            um_ = [z[s] for s in sids if s not in measured]
            if um_:
                axi.plot(um_, [pr2[s][key] for s in sids if s not in measured], "o", color="#ff7f0e", label="interpolated (no H&E)")
            axi.set_ylabel(lab); axi.grid(alpha=0.3)
        ax[0].legend(); ax[2].set_xlabel("z (um)")
        fig.suptitle("per-section correction: consensus of duct-anchor pair fits (median section = identity)")
        fig.tight_layout(); fig.savefig(out / "consensus_report.png", dpi=110)
    except Exception as e:
        print(f"  report figure skipped: {e}")
    big = [s for s in sids if abs(pr2[s]["scale_x"] - 1) > 0.05 or abs(pr2[s]["scale_y"] - 1) > 0.05
           or abs(pr2[s]["rot_deg"]) > 2 or np.hypot(pr2[s]["tx_um"], pr2[s]["ty_um"]) > 150]
    print(f"wrote {out}: {n} sections; {len(big)} get a correction beyond 5% scale / 2 deg / 150 um: "
          f"{', '.join(s.split('-')[-1] for s in big)}", flush=True)


    wild = [s for s in sids if not (0.85 < pr2[s]["scale_x"] < 1.15 and 0.85 < pr2[s]["scale_y"] < 1.15 and abs(pr2[s]["rot_deg"]) < 15)]
    if wild or np.median(res) > 50:
        (out / "section_corrections.json").rename(out / "section_corrections.REJECTED.json")
        raise SystemExit(f"REJECTED: median pair residual {np.median(res):.1f} um, out-of-range sections "
                         f"{', '.join(s.split('-')[-1] for s in wild)}; corrections not left in place")


if __name__ == "__main__":
    main()
