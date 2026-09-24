#!/usr/bin/env python
from __future__ import annotations

import argparse
import hashlib
import importlib.util as _iu
import json
import time
from pathlib import Path

import os
import cv2
import numpy as np
import pandas as pd
from PIL import Image

ROOT = Path(os.environ.get("THREED_ROOT", os.path.join(os.environ.get("PROJECTS_ROOT", "/data/heid"), "3d")))
WORK = ROOT / "xenium"
VOL = Path(os.environ.get("HTAN3D_VOL") or ROOT / "reconstruction/volume_8um")

_s = _iu.spec_from_file_location("bcp", Path(__file__).resolve().parent / "build_cell_planes.py")
BCP = _iu.module_from_spec(_s)
_s.loader.exec_module(BCP)

if os.environ.get("HTAN3D_PREP"):
    BCP.SEC = Path(os.environ["HTAN3D_PREP"])
if os.environ.get("HTAN3D_EDGES"):
    BCP.EDGE_RUNS = tuple(x.strip() for x in os.environ["HTAN3D_EDGES"].split(",") if x.strip())
if os.environ.get("HTAN3D_MAN"):
    BCP.MANIFEST = Path(os.environ["HTAN3D_MAN"])
ANCHOR = os.environ.get("HTAN3D_ANCHOR", "HT891Z1-U66")

OUT_MPP = 8.0
SELFCHECK_PLANES = 4
SELFCHECK_TEXTURE = ("G_withdrawn", 1, "Tumor")


def build_canvas(sids, shapes, m8, chains, gmeta):
    sizes = {s: BCP.plane_size(shapes[s], OUT_MPP) for s in sids}
    pts = []
    for ch in chains.values():
        for s in sids:
            w, h = sizes[s]
            c = np.array([[0, 0], [w, 0], [w, h], [0, h]], float) * (OUT_MPP / BCP.WG_MPP)
            pts.append((c @ ch[s][:2, :2].T + ch[s][:, 2]) * BCP.WG_MPP)
    pts = np.vstack(pts)
    o = pts.min(0) - 40.0
    W = int(np.ceil((pts[:, 0].max() + 40.0 - o[0]) / OUT_MPP))
    H = int(np.ceil((pts[:, 1].max() + 40.0 - o[1]) / OUT_MPP))
    if gmeta.get("canvas_origin_um"):


        o = np.asarray(gmeta["canvas_origin_um"], float)
        W, H = int(gmeta["canvas_px"]["width"]), int(gmeta["canvas_px"]["height"])

        def M_for(bk, s):
            M = np.zeros((2, 3))
            M[:2, :2] = chains[bk][s][:2, :2]
            M[:, 2] = (BCP.WG_MPP / OUT_MPP) * chains[bk][s][:, 2] - o / OUT_MPP
            return M
        return sizes, M_for, 0, 0, W, H

    def M_for(bk, s):
        M = np.zeros((2, 3))
        M[:2, :2] = chains[bk][s][:2, :2]
        M[:, 2] = (BCP.WG_MPP / OUT_MPP) * chains[bk][s][:, 2] - o / OUT_MPP
        return M

    anym = np.zeros((H, W), bool)
    for bk in chains:
        for s in sids:
            anym |= cv2.warpAffine(m8[s].astype(np.uint8) * 255, M_for(bk, s), (W, H),
                                   flags=cv2.INTER_NEAREST, borderValue=0) > 0
    ys, xs = np.nonzero(anym)
    p = int(round(80.0 / OUT_MPP))
    r0, r1 = max(0, ys.min() - p), min(H, ys.max() + p + 1)
    c0, c1 = max(0, xs.min() - p), min(W, xs.max() + p + 1)
    CW, CH = c1 - c0, r1 - r0
    want = (gmeta["canvas_px"]["width"], gmeta["canvas_px"]["height"])
    if (CW, CH) != want:
        raise SystemExit(f"canvas rebuilt {CW}x{CH}, the volume says {want} -- "
                         "refusing to place cells on a different grid")
    return sizes, M_for, c0, r0, CW, CH


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reg", default=str(WORK / "outputs/registration.json"))
    ap.add_argument("--out", default=str(WORK / "outputs/cells_planes"))
    ap.add_argument("--frag", default=str(WORK / "outputs/xenium9_cells_fragment.json"))
    ap.add_argument("--no-textures", action="store_true")
    a = ap.parse_args()
    t0 = time.time()
    reg = json.loads(Path(a.reg).read_text())
    gmeta = json.loads((VOL / "metadata.json").read_text())
    cmeta = json.loads((VOL / "cells_metadata.json").read_text())
    probe = {p["section_id"]: p for p in json.loads(BCP.PROBE.read_text())}

    norms = {bk: {c["name"]: float(c.get("norm_density_full_alpha") or c["norm_density_full_alpha_by_basis"][bk])
                  for c in cmeta["classes"]} for bk in BCP.BASES}
    CLASSES = list(BCP.CLASS_COLOUR)
    cidx = {c: i for i, c in enumerate(CLASSES)}

    import csv as _csv
    man = {r["section_id"]: r for r in _csv.DictReader(open(BCP.MANIFEST), delimiter="\t")}
    sids = sorted(man, key=lambda s: float(man[s]["z_position_um"]))

    m8, shapes = {}, {}
    for s in sids:
        m = cv2.imread(str(BCP.SEC / s / f"mask_mpp{BCP.SRC_MPP:g}.png"), cv2.IMREAD_GRAYSCALE)
        shapes[s] = m.shape
        w, h = BCP.plane_size(m.shape, OUT_MPP)
        m8[s] = cv2.resize((m > 127).astype(np.uint8), (w, h),
                           interpolation=cv2.INTER_AREA) > 0
    edges = BCP.CHAIN.harvest(BCP.EDGE_RUNS)
    tree = BCP.CHAIN.prereg_tree(edges, sids)
    chains = {k: BCP.CHAIN.chain_on(tree, edges, b, ANCHOR) for k, b in BCP.BASES.items()}
    sizes, M_for, c0, r0, CW, CH = build_canvas(sids, shapes, m8, chains, gmeta)
    print(f"canvas {CW} x {CH} matches the volume  ({time.time()-t0:.0f}s)", flush=True)


    def alpha_of(bk, i):
        return np.array(Image.open(VOL / gmeta["encodings"][bk]["planes"][i]["webp_lossless"])
                        .convert("RGBA"))[..., 3]

    def px_of(X, Y):
        xi = np.round(X).astype(np.int64)
        yi = np.round(Y).astype(np.int64)
        return xi, yi, (xi >= 0) & (xi < CW) & (yi >= 0) & (yi < CH)

    def hit_frac(X, Y, al):
        xi, yi, k = px_of(X, Y)
        if not len(xi):
            return 0.0, 0
        return float(al[yi[k], xi[k]].sum()) / len(xi), int((~k).sum())

    def foot(X, Y):
        g = np.zeros((CH, CW), np.uint8)
        xi, yi, k = px_of(X, Y)
        g[yi[k], xi[k]] = 255
        return cv2.morphologyEx(g, cv2.MORPH_CLOSE, np.ones((9, 9), np.uint8)) > 0

    ks = int(2 * round(3 * BCP.SIGMA_PX) + 1)

    def dens(X, Y, sel):
        g = np.zeros((CH, CW), np.float32)
        xi, yi, k = px_of(X[sel], Y[sel])
        if k.any():
            np.add.at(g, (yi[k], xi[k]), 1.0)
        return cv2.GaussianBlur(g, (ks, ks), BCP.SIGMA_PX)

    def four_checks(bk, i, X, Y, al):
        inm, noff = hit_frac(X, Y, al)
        f = foot(X, Y)
        iou = float((f & al).sum() / max(1, (f | al).sum()))
        far = sids[(i + len(sids) // 2) % len(sids)]
        Mf, Mi = M_for(bk, far), M_for(bk, sids[i])
        det = Mi[0, 0] * Mi[1, 1] - Mi[0, 1] * Mi[1, 0]
        Ii = np.array([[Mi[1, 1], -Mi[0, 1]], [-Mi[1, 0], Mi[0, 0]]]) / det
        px_ = Ii[0, 0] * (X + c0 - Mi[0, 2]) + Ii[0, 1] * (Y + r0 - Mi[1, 2])
        py_ = Ii[1, 0] * (X + c0 - Mi[0, 2]) + Ii[1, 1] * (Y + r0 - Mi[1, 2])
        Xf = Mf[0, 0] * px_ + Mf[0, 1] * py_ + Mf[0, 2] - c0
        Yf = Mf[1, 0] * px_ + Mf[1, 1] * py_ + Mf[1, 2] - r0
        ctl_far, _ = hit_frac(Xf, Yf, al)
        ctl_sh, _ = hit_frac(X + BCP.CTRL_SHIFT_PX, Y + BCP.CTRL_SHIFT_PX, al)
        return {"off_canvas": noff, "in_mask": round(inm, 4),
                "iou_footprint_vs_alpha": round(iou, 4),
                "control_far_section": round(ctl_far, 4),
                "control_shift_200px": round(ctl_sh, 4)}


    join = BCP.slide_for_section()
    stored = {(c["index"], c["basis"]): c for c in cmeta["checks"]}
    done, bad = 0, []
    if cmeta.get("checks_repositioned"):


        print("self-check 2 skipped: the stored checks predate a reposition of the planes", flush=True)
    for c in ([] if cmeta.get("checks_repositioned") else cmeta["checks"]):
        if done >= SELFCHECK_PLANES * len(BCP.BASES):
            break
        i, bk = c["index"], c["basis"]
        s = sids[i]
        j = join[s]
        x0, y0, ct, _, _ = BCP.read_cells(j["slide"])
        w8, h8 = sizes[s]
        if j["route"] == "he":
            W0, H0 = j["level0"]
            x8, y8 = BCP.rescale(x0, W0, w8), BCP.rescale(y0, H0, h8)
        else:
            A = np.array(j["matrix_plane_to_crop_px"], float)
            inv = np.linalg.inv(A[:, :2])
            q = np.stack([x0 - A[0, 2], y0 - A[1, 2]])
            x4 = inv[0, 0] * q[0] + inv[0, 1] * q[1]
            y4 = inv[1, 0] * q[0] + inv[1, 1] * q[1]
            w4, h4 = j["plane_grid_px"]
            x8, y8 = BCP.rescale(x4, w4, w8), BCP.rescale(y4, h4, h8)
        M = M_for(bk, s)


        X = (M[0, 0] * x8 + M[0, 1] * y8 + M[0, 2] - c0).astype(np.float32)
        Y = (M[1, 0] * x8 + M[1, 1] * y8 + M[1, 2] - r0).astype(np.float32)
        got = four_checks(bk, i, X, Y, alpha_of(bk, i) > 0)
        for k in ("in_mask", "iou_footprint_vs_alpha", "control_far_section",
                  "control_shift_200px"):
            if abs(got[k] - c[k]) > 1e-4:
                bad.append((i, bk, k, c[k], got[k]))
        done += 1
    if bad:
        for b in bad:
            print("MISMATCH", b, flush=True)
        raise SystemExit("recomputed checks do not reproduce the published ones -- the "
                         "imported chain is not the one that built the other planes")
    print(f"self-check: {done} published check rows reproduced exactly", flush=True)


    if cmeta.get("checks_repositioned"):
        print("self-check 3 skipped: the published texture predates a reposition of the planes", flush=True)
    else:
        bk, i, cls = SELFCHECK_TEXTURE
        s = sids[i]
        j = join[s]
        x0, y0, ct, _, _ = BCP.read_cells(j["slide"])
        w8, h8 = sizes[s]
        W0, H0 = j["level0"]
        x8, y8 = BCP.rescale(x0, W0, w8), BCP.rescale(y0, H0, h8)
        M = M_for(bk, s)
        X = (M[0, 0] * x8 + M[0, 1] * y8 + M[0, 2] - c0).astype(np.float32)
        Y = (M[1, 0] * x8 + M[1, 1] * y8 + M[1, 2] - r0).astype(np.float32)
        g = dens(X, Y, np.array([c == cls for c in ct]))
        al = np.clip(g / max(norms[bk][cls], 1e-9), 0, 1) ** BCP.NORM_GAMMA
        a8 = (al * 255.0 + 0.5).astype(np.uint8)
        ref = np.array(Image.open(VOL / bk / "cells" / f"z{i:02d}_{s.split('-')[1]}.{cls}.webp")
                       .convert("RGBA"))[..., 3]


        import io
        buf = io.BytesIO()
        Image.fromarray(np.dstack([np.full_like(a8, 255)] * 3 + [a8])).save(
            buf, format="WEBP", lossless=True, quality=100, method=6)
        a8rt = np.array(Image.open(io.BytesIO(buf.getvalue())).convert("RGBA"))[..., 3]
        d = np.abs(a8rt.astype(int) - ref.astype(int))
        n_diff = int((d > 0).sum())
        print(f"self-check texture {bk} z{i:02d} {cls}: encoder round trip is "
              f"{'exact' if np.array_equal(a8rt, a8) else 'LOSSY'}; vs published "
              f"max |diff| {d.max()}, {n_diff} of {d.size} pixels differ "
              f"({(d == 0).mean()*100:.4f}% identical)", flush=True)
        if n_diff:
            w = np.argsort(-d, axis=None)[:5]
            yy, xx = np.unravel_index(w, d.shape)
            print("  worst pixels (mine vs published, and the published alpha there):",
                  [(int(y), int(x), int(a8rt[y, x]), int(ref[y, x])) for y, x in zip(yy, xx)],
                  flush=True)
            print(f"  of the differing pixels, {(ref[d > 0] == 255).mean()*100:.1f}% are "
                  f"saturated in the published file and "
                  f"{(ref[d > 0] == 0).mean()*100:.1f}% are empty there", flush=True)


        if d.max() > 1:
            raise SystemExit("the density/alpha path does not reproduce the published "
                             "texture: a difference above one count is not the published "
                             "normalisation's rounding, it is a different density")


    mapped, rows, checks = {}, [], []
    for sid, r in reg["sections"].items():
        i = sids.index(sid)
        slide = r["slide"]
        he = pd.read_parquet(BCP.CELLS / f"{slide}.parquet",
                             columns=["cell_id", "x_um", "y_um"])
        cl = pd.read_csv(BCP.PRED / slide / f"{slide}_cells.csv",
                         usecols=["cell_id", "cell_type"])
        d = he.merge(cl, on="cell_id", how="inner")
        if len(d) != len(he) or len(d) != len(cl):
            raise SystemExit(f"{sid}: cell_id join lost rows "
                             f"({len(he)} detected, {len(cl)} classified, {len(d)} joined)")
        he_xy = np.stack([d["x_um"].to_numpy(), d["y_um"].to_numpy()], 1)
        ct = d["cell_type"].to_numpy()
        unk = sorted(set(ct) - set(CLASSES))
        if unk:
            raise SystemExit(f"{sid}: unknown classes {unk}")

        H0, W0 = probe[sid]["level_yx"][0]
        w8, h8 = sizes[sid]
        ps = r["pixel_size_um"]

        def to_canvas(aff, bk):
            xe = he_xy @ np.array(aff["A"]).T + np.array(aff["b"])
            x8 = BCP.rescale(xe[:, 0] / ps, W0, w8)
            y8 = BCP.rescale(xe[:, 1] / ps, H0, h8)
            M = M_for(bk, sid)
            return ((M[0, 0] * x8 + M[0, 1] * y8 + M[0, 2] - c0).astype(np.float32),
                    (M[1, 0] * x8 + M[1, 1] * y8 + M[1, 2] - r0).astype(np.float32))

        cls_i = np.array([cidx[c] for c in ct], np.int8)
        per = {}
        for bk in BCP.BASES:
            X, Y = to_canvas(r["affine_he_um_to_xenium_um"], bk)
            per[bk] = (X, Y)
            alp = alpha_of(bk, i) > 0
            ck = four_checks(bk, i, X, Y, alp)
            Xw, Yw = to_canvas(
                r["control_wrong_section"]["affine_he_um_to_that_xenium_um"], bk)
            ck_w, _ = hit_frac(Xw, Yw, alp)
            fw = foot(Xw, Yw)
            ck["control_wrong_section_in_mask"] = round(ck_w, 4)
            ck["control_wrong_section_iou"] = round(
                float((fw & alp).sum() / max(1, (fw | alp).sum())), 4)
            ck["control_wrong_section_against"] = r["control_wrong_section"]["against"]
            ck.update(index=i, basis=bk, section_id=sid, route="xenium_pointcloud",
                      n_cells=int(len(cls_i)))
            checks.append(ck)
            print(f"  {sid:<14} {bk:<16} in_mask {ck['in_mask']:.4f}  "
                  f"IoU {ck['iou_footprint_vs_alpha']:.4f}  "
                  f"shift200 {ck['control_shift_200px']:.4f}  "
                  f"far {ck['control_far_section']:.4f}  "
                  f"wrongsec IoU {ck['control_wrong_section_iou']:.4f}", flush=True)
        mapped[sid] = {"i": i, "cls": cls_i, "xy": per}
        rows.append({"index": i, "section_id": sid, "slide": slide,
                     "route": "xenium_pointcloud", "n_cells": int(len(cls_i)),
                     "n_detected": int(len(he)), "n_classified": int(len(cl))})


    planes, nbytes = [], 0
    if not a.no_textures:
        for bk in BCP.BASES:
            dd = Path(a.out) / bk / "cells"
            dd.mkdir(parents=True, exist_ok=True)
            for sid, mp in mapped.items():
                i = mp["i"]
                X, Y = mp["xy"][bk]
                for c in CLASSES:
                    sel = mp["cls"] == cidx[c]
                    n = int(sel.sum())
                    if not n:
                        continue
                    g = dens(X, Y, sel)
                    al = np.clip(g / max(norms[bk][c], 1e-9), 0, 1) ** BCP.NORM_GAMMA
                    a8 = (al * 255.0 + 0.5).astype(np.uint8)
                    if not a8.any():
                        continue
                    rgba = np.dstack([np.full_like(a8, 255)] * 3 + [a8])
                    fn = f"z{i:02d}_{sid.split('-')[1]}.{c}.webp"
                    fp = dd / fn
                    Image.fromarray(rgba).save(fp, format="WEBP", lossless=True,
                                               quality=100, method=6)
                    nbytes += fp.stat().st_size
                    planes.append({"index": i, "basis": bk, "section_id": sid,
                                   "z_um": float(man[sid]["z_position_um"]),
                                   "cell_class": c, "n_cells": n,
                                   "file": f"{bk}/cells/{fn}",
                                   "bytes": fp.stat().st_size})
                print(f"  wrote {bk} {sid}", flush=True)

    frag = {
        "what": "the nine Xenium planes of volume_8um, mapped from the Cell-model H&E "
                "predictions through a point-cloud fit of the H&E nuclei onto the "
                "Xenium nuclei. A FRAGMENT: merge into cells_metadata.json.",
        "claim": "MODEL PREDICTION on H&E, same as the other forty-one planes. What "
                 "is new here is the geometry, not the labels.",
        "checks_cannot_detect": {
            "what": "in_mask and iou_footprint_vs_alpha CANNOT tell that a plane has "
                    "been given the WRONG SECTION's cells. They test the pose, not the "
                    "identity, and serial sections share a silhouette.",
            "measured": "with four of these nine paired to the wrong Xenium section, "
                        "in_mask was still 0.9445-0.9929 and IoU still 0.8557-0.9114, "
                        "while control_shift_200px stayed low (0.38-0.45) so the check "
                        "suite looked healthy. The error moved IoU by about the same "
                        "amount that ordinary registration noise does.",
            "what_did_detect_it": "a per-object statistic: the median distance from "
                                  "each H&E nucleus to the nearest Xenium nucleus, "
                                  "1.0-1.6 um for the right section against 4.5-5.0 um "
                                  "-- the chance rate for this cell density -- for the "
                                  "wrong one.",
            "evidence": ["xenium/outputs/canvas_checks_under_named_pairing_CONTROL.json",
                         "xenium/outputs/slide_permutation.json"],
            "for_future_planes": "any check built only on tissue-shape agreement is "
                                 "blind to this class of error. Pair it with a "
                                 "per-object statistic whenever the candidate objects "
                                 "share a silhouette.",
        },
        "pairing_provenance": {
            "how": "MEASURED, not read from file names. All nine H&E-by-Xenium "
                   "combinations were fitted within each slide (27 cells; the slide "
                   "whose naming is confirmed correct was included as a positive "
                   "control and came out diagonal).",
            "result": "four of the nine H&E crops carry the wrong U number: the crops "
                      "named U1 and U32 are each other's sections, as are U44 and U69. "
                      "The Xenium region names, and therefore this volume's z "
                      "assignment, are the correct ones -- established separately by "
                      "locating each disputed tissue against the undisputed sections.",
            "evidence": ["xenium/outputs/slide_permutation.json",
                         "xenium/outputs/serial_position.json"],
        },
        "geometry": {
            "steps": "H&E level-0 px -> H&E um -> (fitted similarity) -> Xenium um -> "
                     "DAPI level-0 px -> plane grid at 8 um/px -> chain matrix -> canvas",
            "fitted_link": "affine_he_um_to_xenium_um in registration.json",
            "everything_else": "imported from build_cell_planes.py, not reimplemented",
            "selfcheck_published_rows_reproduced": done,
        },
        "indices": sorted(m["i"] for m in mapped.values()),
        "sections": rows, "checks": checks, "planes": planes,
        "texture_bytes": nbytes,
        "normalisation_source": "the per-class norm_density_full_alpha published in "
                                "cells_metadata.json, so these planes sit on the same "
                                "ramp as the other forty-one",
        "elapsed_s": round(time.time() - t0, 1),
    }
    Path(a.frag).write_text(json.dumps(frag, indent=1, default=float))
    print(f"\nwrote {a.frag}  ({time.time()-t0:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
