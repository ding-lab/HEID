#!/usr/bin/env python3

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
from PIL import Image
from scipy.ndimage import gaussian_filter, rotate

MAX_SHIFT_UM = 120.0
MAX_ROT_DEG = 2.0
ROT_STEP_DEG = 0.5


def load(p):
    im = np.asarray(Image.open(p).convert("RGBA")).astype(np.float32)
    return im[:, :, 0], im[:, :, 3] > 0


def ncc(A, B, M):
    a, b = A[M], B[M]
    if a.size < 1000:
        return -2.0
    a = a - a.mean(); b = b - b.mean()
    d = np.sqrt((a * a).sum() * (b * b).sum())
    return float((a * b).sum() / d) if d > 0 else -2.0


def search(A, B, M, rad_px, rots):
    best = (0, 0, 0.0, -2.0)
    for th in rots:
        Bt = B if th == 0 else rotate(B, th, reshape=False, order=1, mode="nearest")
        for dy in range(-rad_px, rad_px + 1):
            for dx in range(-rad_px, rad_px + 1):
                Bs = np.roll(np.roll(Bt, dy, 0), dx, 1)
                Ms = np.roll(np.roll(M, dy, 0), dx, 1) & M
                r = ncc(A, Bs, Ms)
                if r > best[3]:
                    best = (dx, dy, th, r)
    return best


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--volume", required=True)
    ap.add_argument("--fine", default="", help="4 um volume for the refinement pass")
    ap.add_argument("--registry", default="", help="section_corrections.json to write swap_to_main_residual into "
                    "(world um: moves a twice-imaged section's second scan and what sits in its frame onto the H&E base)")
    ap.add_argument("--swap-volume", default="", dest="swap_volume",
                    help="the swap arm's own grey volume (same mpp, same canvas): its planes are the swap planes. "
                         "Without it the swap planes are read from <volume>/he_swap_metadata.json")
    ap.add_argument("--cells-frame", default="auto", choices=("auto", "main", "swap"), dest="cells_frame",
                    help="which frame the predicted cells of the twice-imaged sections sit in: 'auto' = the "
                         "xenium sections' cells are Xenium cells in the swap frame (891); 'main' = every "
                         "section's cells were predicted on the H&E base and get no swap residual (HT206B1: "
                         "H&E is the main scan, the Xenium DAPI is the swap); 'swap' = all of them")
    ap.add_argument("--residual", default="fit", choices=("fit", "none", "cloud"),
                    help="'none' writes an identity residual for every section (the swap scan stays where the chain "
                         "put it) while still recording r0; 'cloud' takes each section's residual from "
                         "<volume>/xenium_cloud_registration.json (register_xenium_cloud.py: the Xenium nuclei "
                         "registered to the H&E nuclei, any rotation / flip / shift) and only scores it here")
    a = ap.parse_args()
    V = Path(a.volume)
    cloud = json.loads((V / "xenium_cloud_registration.json").read_text())["sections"] if a.residual == "cloud" else {}
    vm = json.loads((V / "metadata.json").read_text())
    mpp = float(vm["in_plane_um_per_px"])
    idx = {q["section_id"]: q for q in vm["encodings"]["G_withdrawn"]["planes"]}
    if a.swap_volume:
        SV = Path(a.swap_volume); svm = json.loads((SV / "metadata.json").read_text())
        if svm["canvas_px"] != vm["canvas_px"] or abs(float(svm["in_plane_um_per_px"]) - mpp) > 1e-9:
            raise SystemExit(f"{SV}: not on {V}'s canvas")
        sw = {q["section_id"]: {"grey_file": str(SV / q["webp_lossless"]), "modality_replaced": q.get("modality", "")}
              for q in svm["encodings"]["G_withdrawn"]["planes"] if q["section_id"] in idx}
    else:
        sm = json.loads((V / "he_swap_metadata.json").read_text())
        sw = {q["section_id"]: {"grey_file": str(V / q["grey_file"]), "modality_replaced": q.get("modality_replaced", "")}
              for q in sm["planes"] if q["basis"] == "G_withdrawn"}
    rad = int(round(MAX_SHIFT_UM / mpp))
    rots = np.arange(-MAX_ROT_DEG, MAX_ROT_DEG + 1e-9, ROT_STEP_DEG)
    out = {}
    for sid, q in sorted(sw.items(), key=lambda kv: idx[kv[0]]["index"]):
        A, M = load(V / idx[sid]["webp_lossless"])
        B, _ = load(Path(q["grey_file"]))
        A = gaussian_filter(A, 1.0); B = gaussian_filter(B, 1.0)
        r0 = ncc(A, B, M)
        if a.residual in ("none", "cloud"):
            dx, dy, th, r1 = 0, 0, 0.0, r0
        else:
            dx, dy, th, r1 = search(A, B, M, rad, rots)


        cy, cx = (np.array(B.shape[:2]) - 1) / 2.0
        best = None
        if a.residual == "cloud":
            if sid not in cloud:
                raise SystemExit(f"{sid}: no entry in {V / 'xenium_cloud_registration.json'}")
            Mpx = np.asarray(cloud[sid]["swap_to_main_px"], float)
            Bw = cv2.warpAffine(B, Mpx, (B.shape[1], B.shape[0]), flags=cv2.INTER_LINEAR, borderValue=0)
            Mw = cv2.warpAffine(M.astype(np.uint8), Mpx, (B.shape[1], B.shape[0]), flags=cv2.INTER_NEAREST, borderValue=0) > 0
            best = (ncc(A, Bw, Mw & M), Mpx)
            p = Mpx @ [cx, cy, 1.0]; dx, dy = p[0] - cx, p[1] - cy
            th = float(np.degrees(np.arctan2(Mpx[1, 0], Mpx[0, 0]))); r1 = best[0]
        for sgn in ((1.0, -1.0) if best is None else ()):
            Mpx = cv2.getRotationMatrix2D((float(cx), float(cy)), sgn * float(th), 1.0)
            Mpx[:, 2] += [dx, dy]
            Bw = cv2.warpAffine(B, Mpx, (B.shape[1], B.shape[0]), flags=cv2.INTER_LINEAR, borderValue=0)
            Mw = cv2.warpAffine(M.astype(np.uint8), Mpx, (B.shape[1], B.shape[0]), flags=cv2.INTER_NEAREST, borderValue=0) > 0
            rr = ncc(A, Bw, Mw & M)
            if best is None or rr > best[0]:
                best = (rr, Mpx)
        out[sid] = {"dx_um": round(dx * mpp, 1), "dy_um": round(dy * mpp, 1),
                    "theta_deg": round(float(th), 2), "r0": round(r0, 4), "r1": round(r1, 4),
                    "r_affine": round(float(best[0]), 4), "swap_to_main_px": best[1].tolist(),
                    "modality_replaced": q.get("modality_replaced", "")}
        print(f"  {sid}: shift ({dx * mpp:+.0f}, {dy * mpp:+.0f}) um, rot {th:+.2f} deg, "
              f"r {r0:.3f} -> {r1:.3f}", flush=True)
    (V / "he_swap_fit.json").write_text(json.dumps({
        "what": "per-section residual of the slide-level H&E placement, measured against the "
                "plane the registration chain solved; apply to the H&E-derived products "
                "(cell points, cell planes) and to the swapped plane itself",
        "search": {"MAX_SHIFT_UM": MAX_SHIFT_UM, "MAX_ROT_DEG": MAX_ROT_DEG,
                   "ROT_STEP_DEG": ROT_STEP_DEG, "grid_um": mpp},
        "sections": out}, indent=1))
    if a.registry:

        reg_p = Path(a.registry); reg = json.loads(reg_p.read_text()) if reg_p.exists() else {"sections": {}}
        o = np.asarray(vm.get("canvas_origin_um") or [0.0, 0.0], float)
        S = np.array([[mpp, 0, o[0]], [0, mpp, o[1]]]); Si = np.array([[1 / mpp, 0, -o[0] / mpp], [0, 1 / mpp, -o[1] / mpp]])
        def comp(A_, B_):
            return (np.vstack([A_, [0, 0, 1]]) @ np.vstack([B_, [0, 0, 1]]))[:2]
        mod = {sid: v["modality_replaced"] for sid, v in out.items()}
        reg["swap_to_main_residual"] = {sid: comp(S, comp(np.asarray(v["swap_to_main_px"], float), Si)).tolist() for sid, v in out.items()}
        if a.cells_frame == "auto":
            reg["swap_frame_cells"] = sorted(sid for sid in out if mod.get(sid) == "xenium")
            rule = "cells of xenium sections come from Xenium and sit in the swap frame; cells of codex sections were predicted on the H&E base"
        elif a.cells_frame == "swap":
            reg["swap_frame_cells"] = sorted(out)
            rule = "--cells-frame swap: every twice-imaged section's cells sit in the swap frame"
        else:
            reg["swap_frame_cells"] = []
            rule = "--cells-frame main: every section's cells were predicted on the H&E base (the main scan); no swap residual on cells"
        reg["swap_residual_source"] = {"file": str(V / "he_swap_fit.json"), "frame": "world um, swap scan (CODEX / Xenium) -> H&E base",
                                       "cells_rule": rule}
        for k in ("main_arm_residual", "main_arm_residual_cells", "main_arm_residual_source"):
            reg.pop(k, None)
        reg_p.write_text(json.dumps(reg, indent=1))
        with open(reg_p.with_name("section_transforms.tsv"), "w") as fh:
            fh.write("section_id\tmodality\tcorrection_scale_x\tcorrection_rot_deg\tcorrection_tx_um\tcorrection_ty_um\tswap_residual_dx_um\tswap_residual_dy_um\tswap_residual_rot_deg\tr_before\tr_after\tcells_frame\n")
            for q in sorted(vm["encodings"]["G_withdrawn"]["planes"], key=lambda z: z["z_um"]):
                sid = q["section_id"]; C = np.asarray(reg["sections"].get(sid, [[1, 0, 0], [0, 1, 0]]), float)
                sx = float(np.hypot(C[0, 0], C[1, 0])); rot = float(np.degrees(np.arctan2(C[1, 0], C[0, 0])))
                v = out.get(sid)
                fh.write(f"{sid}\t{q.get('modality')}\t{sx:.4f}\t{rot:.3f}\t{C[0, 2]:.1f}\t{C[1, 2]:.1f}\t"
                         + (f"{v['dx_um']:.1f}\t{v['dy_um']:.1f}\t{v['theta_deg']:.2f}\t{v['r0']:.3f}\t{v['r_affine']:.3f}\t{'swap' if sid in reg['swap_frame_cells'] else 'main'}" if v else "\t\t\t\t\tmain") + "\n")
        print(f"registry {reg_p}: swap_to_main_residual for {len(out)} sections, cells in the swap frame for {len(reg['swap_frame_cells'])}", flush=True)
    med = np.median([abs(v["dx_um"]) + abs(v["dy_um"]) for v in out.values()]) if out else 0
    print(f"wrote {V / 'he_swap_fit.json'}: {len(out)} sections, median |dx|+|dy| {med:.0f} um, "
          f"r improved on {sum(1 for v in out.values() if v['r1'] > v['r0'] + 0.01)} of them", flush=True)


if __name__ == "__main__":
    main()
