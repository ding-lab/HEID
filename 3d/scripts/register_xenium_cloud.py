#!/usr/bin/env python3
import argparse, json, struct
from pathlib import Path
import numpy as np, cv2
from PIL import Image
from scipy.ndimage import gaussian_filter, zoom
from scipy.spatial import cKDTree

DENS_UM = 16.0
ROT_STEP = 1.0
DENS_FINE_UM = 8.0
FINE_STEP = 0.1
FINE_WIN = 3.0
MIN_MASS = 0.3
ICP_R_UM = 15.0
MATCH_UM = 10.0
CONTROL_SHIFT_UM = 500.0


def read_bin(path):
    raw = Path(path).read_bytes()
    magic, n, cw, ch, sub, ncls, _ = struct.unpack("<8sIHHHHI", raw[:24])
    assert magic == b"CELLPT01", path
    return np.frombuffer(raw[32:32 + 4 * n], dtype="<u2").reshape(n, 2).astype(np.float64) / sub


def density(xy_px, H, W, f, sigma=1.5, pad=0):
    d = np.zeros((H + 2 * pad, W + 2 * pad), np.float32)
    ix = np.round(xy_px[:, 0]).astype(int) + pad; iy = np.round(xy_px[:, 1]).astype(int) + pad
    k = (ix >= 0) & (ix < d.shape[1]) & (iy >= 0) & (iy < d.shape[0])
    np.add.at(d, (iy[k], ix[k]), 1.0)
    return gaussian_filter(zoom(d, 1 / f, order=1), sigma)


def ncc(A, B, M):
    a, b = A[M], B[M]
    if a.size < 300: return -2.0
    a = a - a.mean(); b = b - b.mean(); d = np.sqrt((a * a).sum() * (b * b).sum())
    return float((a * b).sum() / d) if d > 0 else -2.0


class Shifter:
    def __init__(self, A, M):
        ys, xs = np.nonzero(M); self.y0, self.x0 = int(ys.min()), int(xs.min())
        self.T = (A * M)[self.y0:int(ys.max()) + 1, self.x0:int(xs.max()) + 1].astype(np.float32)
        self.A, self.M = A, M
    def __call__(self, B):
        Bf = B.astype(np.float32)
        res = cv2.matchTemplate(Bf, self.T, cv2.TM_CCOEFF_NORMED)


        th_, tw_ = self.T.shape
        mass = cv2.boxFilter(Bf, -1, (tw_, th_), anchor=(0, 0), normalize=False, borderType=cv2.BORDER_CONSTANT)
        res = np.where(mass[:res.shape[0], :res.shape[1]] >= MIN_MASS * Bf.sum(), np.nan_to_num(res, nan=-1.0), -1.0)
        my, mx = np.unravel_index(int(np.argmax(res)), res.shape)
        sx, sy = self.x0 - int(mx), self.y0 - int(my)
        return sx, sy, ncc(self.A, np.roll(np.roll(B, sy, 0), sx, 1), self.M)


def similarity_fit(P, Q):
    mp, mq = P.mean(0), Q.mean(0)
    X, Y = P - mp, Q - mq
    U, S, Vt = np.linalg.svd(X.T @ Y)
    D = np.eye(2); D[1, 1] = np.sign(np.linalg.det(U @ Vt))
    R = (U @ D @ Vt).T
    s = (S * np.diag(D)).sum() / (X ** 2).sum()
    t = mq - s * (R @ mp)
    return s, R, t


def apply(M, xy):
    return xy @ M[:, :2].T + M[:, 2]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--volume", required=True)
    ap.add_argument("--layer", required=True, help="xenium_gt layer dir built with an identity swap residual")
    ap.add_argument("--write-registry", default="", dest="registry")
    ap.add_argument("--min-ncc", type=float, default=0.45, dest="min_ncc", help="a section below this after refinement is reported, not written")
    ap.add_argument("--coarse-from", default="", dest="coarse_from", help="an earlier xenium_cloud_registration.json: keep its coarse pose per section, redo only the fine pass")
    a = ap.parse_args()
    V = Path(a.volume); L = Path(a.layer)
    prev = {}
    if a.coarse_from:
        for s, v in json.loads(Path(a.coarse_from).read_text())["sections"].items():
            prev[s] = dict(v, rot_coarse_deg=v.get("rot_coarse_deg", -v["rot_deg"]))
    vm = json.loads((V / "metadata.json").read_text()); mpp = float(vm["in_plane_um_per_px"])
    o = np.asarray(vm.get("canvas_origin_um") or [0.0, 0.0], float)
    grey = {q["section_id"]: q["webp_lossless"] for q in vm["encodings"]["G_withdrawn"]["planes"]}
    cp = {f["section_id"]: f["file"] for f in json.loads((V / "cell_points_metadata.json").read_text())["files"] if f["basis"] == "G_withdrawn"}
    xg = json.loads((L / "xenium_gt_metadata.json").read_text())
    xp = {f["section_id"]: f["file"] for f in xg["files"] if f.get("basis", "G_withdrawn") == "G_withdrawn"}
    f = DENS_UM / mpp
    out = {}
    for sid in sorted(xp, key=lambda s: [q["section_id"] for q in vm["encodings"]["G_withdrawn"]["planes"]].index(s)):
        g = np.asarray(Image.open(V / grey[sid]).convert("RGBA")); H, W = g.shape[:2]
        mask = g[:, :, 3] > 0
        pad = max(H, W)
        MA = zoom(np.pad(mask, pad).astype(np.float32), 1 / f, order=1) > 0.5
        he = read_bin(V / cp[sid]); xe = read_bin(L / xp[sid])
        A = density(he, H, W, f, pad=pad)
        r_id = ncc(A, density(xe, H, W, f, pad=pad), MA)

        c = xe.mean(0); best = None; shift = Shifter(A, MA)
        def rotated(flip, th):
            xf = xe.copy()
            if flip: xf[:, 0] = 2 * c[0] - xf[:, 0]
            return apply(cv2.getRotationMatrix2D((float(c[0]), float(c[1])), float(th), 1.0), xf)
        if sid in prev:
            flip, th = bool(prev[sid]["flip"]), float(prev[sid]["rot_coarse_deg"])
            sx, sy, r = shift(density(rotated(flip, th), H, W, f, pad=pad))
            best = (r, flip, th, sx * f, sy * f)
        for flip in ((False, True) if best is None else ()):
            for th in np.arange(-180, 180, ROT_STEP):
                sx, sy, r = shift(density(rotated(flip, th), H, W, f, pad=pad))
                if best is None or r > best[0]:
                    best = (r, flip, float(th), sx * f, sy * f)
        r_c, flip, th, sx, sy = best

        ff = DENS_FINE_UM / mpp
        Af = density(he, H, W, ff, pad=pad); MAf = zoom(np.pad(mask, pad).astype(np.float32), 1 / ff, order=1) > 0.5
        shift_f = Shifter(Af, MAf); fine = None
        for th2 in np.arange(th - FINE_WIN, th + FINE_WIN + 1e-9, FINE_STEP):
            sx2, sy2, r = shift_f(density(rotated(flip, th2), H, W, ff, pad=pad))
            if fine is None or r > fine[0]:
                fine = (r, float(th2), sx2 * ff, sy2 * ff)
        r_fine, th, sx, sy = fine
        M = cv2.getRotationMatrix2D((float(c[0]), float(c[1])), th, 1.0); M[:, 2] += [sx, sy]
        F = np.array([[-1.0, 0, 2 * c[0]], [0, 1.0, 0]]) if flip else np.array([[1.0, 0, 0], [0, 1.0, 0]])
        T = (np.vstack([M, [0, 0, 1]]) @ np.vstack([F, [0, 0, 1]]))[:2]

        tree = cKDTree(he)
        sub = xe if len(xe) <= 60000 else xe[np.random.default_rng(0).choice(len(xe), 60000, replace=False)]
        for it in range(8):
            P = apply(T, sub)
            d, j = tree.query(P, distance_upper_bound=ICP_R_UM / mpp)
            k = np.isfinite(d)
            if k.sum() < 200: break
            s, R, t = similarity_fit(P[k], he[j[k]])
            Tk = np.hstack([s * R, t[:, None]])
            T = (np.vstack([Tk, [0, 0, 1]]) @ np.vstack([T, [0, 0, 1]]))[:2]
        P = apply(T, xe)
        r_f = ncc(A, density(P, H, W, f, pad=pad), MA)
        d, _ = tree.query(P, distance_upper_bound=MATCH_UM / mpp); hit = float(np.isfinite(d).mean())
        d2, _ = tree.query(P + [CONTROL_SHIFT_UM / mpp, 0], distance_upper_bound=MATCH_UM / mpp); ctrl = float(np.isfinite(d2).mean())
        Lm = T[:, :2]; det = float(np.linalg.det(Lm)); rot = float(np.degrees(np.arctan2(Lm[1, 0], Lm[0, 0]))); sc = float(np.sqrt(abs(det)))
        S = np.array([[mpp, 0, o[0]], [0, mpp, o[1]], [0, 0, 1]]); Si = np.linalg.inv(S)
        T_um = (S @ np.vstack([T, [0, 0, 1]]) @ Si)[:2]
        out[sid] = {"ncc_identity": round(r_id, 3), "ncc_coarse": round(r_c, 3), "ncc_fine_8um": round(r_fine, 3), "ncc_refined": round(r_f, 3),
                    "flip": bool(flip), "rot_coarse_deg": round(float(best[2]), 2), "rot_fine_deg": round(th, 2),
                    "rot_deg": round(rot, 2), "scale": round(sc, 4), "det": round(det, 4),
                    "match_frac_10um": round(hit, 3), "match_frac_control_500um": round(ctrl, 3),
                    "n_xenium": int(len(xe)), "n_he": int(len(he)),
                    "swap_to_main_px": T.tolist(), "swap_to_main_um": T_um.tolist()}
        print(f"{sid:24s} identity r {r_id:.3f} -> coarse r {r_c:.3f} (flip={flip} rot {best[2]:+.0f}) -> fine rot {th:+.1f} r@8um {r_fine:.3f} -> refined r {r_f:.3f} "
              f"| rot {rot:+.1f} deg scale {sc:.3f} det {det:+.3f} | matched within {MATCH_UM:g} um {hit:.1%} (control {ctrl:.1%})", flush=True)
    rec = {"what": "Xenium cell cloud -> H&E cell cloud similarity per section (swap -> main), coarse density pose + ICP on nuclei",
           "params": {"DENS_UM": DENS_UM, "ROT_STEP": ROT_STEP, "DENS_FINE_UM": DENS_FINE_UM, "FINE_STEP": FINE_STEP, "FINE_WIN": FINE_WIN, "ICP_R_UM": ICP_R_UM, "MATCH_UM": MATCH_UM}, "sections": out}
    tmp = V / "xenium_cloud_registration.json.tmp"; tmp.write_text(json.dumps(rec, indent=1)); tmp.replace(V / "xenium_cloud_registration.json")
    if a.registry:
        reg_p = Path(a.registry); reg = json.loads(reg_p.read_text())
        ok = {s: v for s, v in out.items() if v["ncc_refined"] >= a.min_ncc}
        reg.setdefault("swap_to_main_residual", {})
        for s, v in ok.items():
            reg["swap_to_main_residual"][s] = v["swap_to_main_um"]
        reg["swap_residual_source"] = {"file": str(V / "xenium_cloud_registration.json"), "frame": "world um, swap scan (Xenium DAPI) -> H&E base",
                                       "method": "register_xenium_cloud.py: Xenium nuclei -> H&E-predicted nuclei similarity",
                                       "cells_rule": reg.get("swap_residual_source", {}).get("cells_rule", "")}
        reg_p.write_text(json.dumps(reg, indent=1))
        print(f"registry {reg_p}: swap_to_main_residual written for {len(ok)} of {len(out)} sections "
              f"(kept those with refined NCC >= {a.min_ncc}); below threshold: {sorted(set(out) - set(ok))}", flush=True)


if __name__ == "__main__":
    main()
