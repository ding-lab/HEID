#!/usr/bin/env python3
import os, sys, json, argparse
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import cv2
from scipy import ndimage
import align_xenium_to_he as A


OK_NCC = 0.30
OK_SILH = 0.75
OK_ORPHAN = 0.10


TRIG_ORPHAN = 0.15
TRIG_NCC = 0.15
ACC_SILH = 0.70
ACC_NCC = 0.20
ACC_SILH_STRONG = 0.80
DATALIM_SILH = 0.55


def load_ctx(g):
    he_rgb, he_px, _, _ = A.load_he_rgb(g["inputs"]["he_qptiff"], target_max_dim=A.WORK_MAX_DIM)
    he_hema = A.extract_hematoxylin(he_rgb)
    xen_full, xen_px = A.load_xenium_dapi(g["inputs"]["xenium_dir"])
    xen_work, _ = A.downsample_to_um(xen_full, xen_px, he_px)
    he_u8 = A.normalize_dapi(he_hema)
    xen_u8 = A.normalize_dapi(xen_work)
    he_tissue = A.extract_he_tissue_mask(he_rgb)
    xen_tissue = A.extract_xen_tissue_mask(xen_u8)
    ctx = A.Context(xen_u8, he_u8, he_tissue, xen_tissue, he_px)
    return ctx, he_u8, xen_u8, he_tissue, xen_tissue, he_px


def orphan_red_frac(p, xen_tissue, he_tissue, h, w):
    M = A.affine_from_params(*p).astype(np.float32)
    warped = cv2.warpAffine(xen_tissue.astype(np.uint8), M, (w, h),
                            flags=cv2.INTER_NEAREST, borderValue=0).astype(bool)
    tot = int(warped.sum())
    if tot == 0:
        return 1.0
    return int((warped & ~he_tissue).sum()) / tot


def rescue_search(he_u8, xen_u8, he_tissue, xen_tissue, he_px):
    he_lbl, hen = ndimage.label(he_tissue)
    xen_lbl, xnn = ndimage.label(xen_tissue)
    xt = int(xen_tissue.sum())
    xpieces = sorted([(i, int((xen_lbl == i).sum())) for i in range(1, xnn + 1)], key=lambda t: -t[1])
    xpieces = [(i, a) for i, a in xpieces if a >= 0.05 * xt][:6]
    big_x = xpieces[0][1] if xpieces else xt
    hfrags = sorted([(j, int((he_lbl == j).sum())) for j in range(1, hen + 1)], key=lambda t: -t[1])
    hfrags = [(j, a) for j, a in hfrags if a >= 0.25 * big_x][:20]
    best = None
    for xi, _ in xpieces:
        xp = (xen_lbl == xi)
        xen_u8_p = (xen_u8 * xp).astype(np.uint8)
        for hj, _ in hfrags:
            hf = (he_lbl == hj)
            ctx = A.Context(xen_u8_p, he_u8, hf, xp, he_px)
            try:
                seeds = A.shape_match_seeds(ctx, xp, hf)
            except Exception:
                continue
            for params, _cov in seeds[:6]:
                try:
                    _, p = A.powell(params, ctx, maxiter=80)
                    p, _ = A.fine_climb(p, ctx)
                    _, ncc, silh = A.evaluate_with_coverage(p, ctx)
                except Exception:
                    continue
                if best is None or silh > best["silh"]:
                    best = {"silh": float(silh), "ncc": float(ncc),
                            "p": [float(x) for x in p], "xen_piece": int(xi), "he_frag": int(hj)}
    return best


def lift_to_full(g, p):
    rot, scale, dx, dy = p
    th = np.deg2rad(rot); c, s = np.cos(th), np.sin(th)
    M_work = np.array([[scale * c, -scale * s, dx], [scale * s, scale * c, dy]])
    xen_w2f = g["full_resolution"]["xenium_shape_hw"][1] / g["work_canvas"]["xenium_shape_hw"][1]
    he_w2f = g["full_resolution"]["he_shape_hw"][1] / g["work_canvas"]["he_shape_hw"][1]
    A_full = (he_w2f / xen_w2f) * M_work[:2, :2]
    b_full = he_w2f * M_work[:, 2]
    M_full = np.zeros((2, 3)); M_full[:2, :2] = A_full; M_full[:, 2] = b_full
    A_inv = np.linalg.inv(A_full)
    M_inv = np.hstack([A_inv, (-A_inv @ b_full).reshape(2, 1)])
    return M_full.tolist(), M_inv.tolist()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--align", required=True, help="aligned sample dir (has global_alignment.json)")
    ap.add_argument("--report", default=None, help="optional cohort TSV to append one line to")
    ap.add_argument("--no-rescue", action="store_true", help="detection only, never rewrite")
    args = ap.parse_args()

    gp = os.path.join(args.align, "global_alignment.json")
    g = json.load(open(gp))
    ctx, he_u8, xen_u8, he_tissue, xen_tissue, he_px = load_ctx(g)
    h, w = he_tissue.shape
    fp = g["final_params_work_canvas"]
    p = [fp["rot_deg"], fp["scale"], fp["dx"], fp["dy"]]
    _, ncc, silh = A.evaluate_with_coverage(p, ctx)
    orf = orphan_red_frac(p, xen_tissue, he_tissue, h, w)
    abstain = bool(g.get("optimization", {}).get("abstain_low_ncc", False))
    qc = {"ncc": round(float(ncc), 4), "silhouette": round(float(silh), 4),
          "orphan_red_frac": round(float(orf), 4), "abstain_low_ncc": abstain, "rescued": False}

    ok = (ncc >= OK_NCC) or (silh >= OK_SILH and orf < OK_ORPHAN)
    if ok:
        qc["verdict"] = "OK"
    else:
        best = None
        if not args.no_rescue and (orf > TRIG_ORPHAN or ncc < TRIG_NCC):
            best = rescue_search(he_u8, xen_u8, he_tissue, xen_tissue, he_px)
        if best and best["silh"] > ACC_SILH and (best["ncc"] > ACC_NCC or best["silh"] > ACC_SILH_STRONG):
            M_full, M_inv = lift_to_full(g, best["p"])
            g["M_xen_full_to_he_full"] = M_full
            g["M_he_full_to_xen_full"] = M_inv
            g["final_params_work_canvas"] = {"rot_deg": best["p"][0], "scale": best["p"][1],
                                             "dx": best["p"][2], "dy": best["p"][3]}
            g.setdefault("optimization", {})["rescue"] = best
            qc.update({"verdict": "OK", "rescued": True, "rescue": best,
                       "ncc": round(best["ncc"], 4), "silhouette": round(best["silh"], 4)})
        elif best and best["silh"] < DATALIM_SILH:
            qc.update({"verdict": "DATA-LIMITED", "rescue_best": best})
        elif best:
            qc.update({"verdict": "DATA-LIMITED", "rescue_best": best})
        else:
            qc["verdict"] = "SUSPECT"

    g["qc"] = qc
    json.dump(g, open(gp, "w"), indent=2)
    line = (f"{os.path.basename(args.align)}\t{qc['verdict']}\t{qc['ncc']}\t{qc['silhouette']}"
            f"\t{qc['orphan_red_frac']}\t{qc['rescued']}")
    if args.report:
        hdr = "sample\tverdict\tncc\tsilhouette\torphan_red\trescued\n"
        newfile = not os.path.exists(args.report)
        with open(args.report, "a") as f:
            if newfile:
                f.write(hdr)
            f.write(line + "\n")
    print(f"[qc] {line}", flush=True)


if __name__ == "__main__":
    main()
