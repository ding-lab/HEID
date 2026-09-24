#!/usr/bin/env python

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path

os.environ.setdefault("PYTHONUNBUFFERED", "1")

import cv2
import numpy as np
import tifffile
import pandas as pd

_HERE = Path(__file__).resolve().parent
_MODELS = _HERE
from importlib import util as _iu


def _load(mod, path):
    s = _iu.spec_from_file_location(mod, str(path))
    m = _iu.module_from_spec(s)
    s.loader.exec_module(m)
    return m


_s0 = _load("s0", _HERE / "footprint_geometry.py")
_tp = _load("tp", _HERE / "tile_ncc_probe.py")
_eng = _load("eng", _MODELS / "pairwise_affine.py")
_pa = _load("pa", _MODELS / "polarity_audit.py")
_pc = _load("pc", _MODELS / "precompensation.py")

MASK_RULE = "prep"
GEOM = {}
PREP_LEVELS = (0.5, 1.0, 4.0)


def prepare_section(row, mpp_out):


    sec_dir = Path(row["prep_dir"])
    lvl = min(PREP_LEVELS, key=lambda L: abs(L - mpp_out))
    img_fp = sec_dir / f"img_mpp{lvl:g}.tif"
    msk_fp = sec_dir / f"mask_mpp{lvl:g}.png"
    if not img_fp.exists() or not msk_fp.exists():
        raise FileNotFoundError(f"section prep missing for {row['section_id']}: {img_fp}")
    dens = tifffile.imread(str(img_fp))
    if dens.ndim == 3:
        dens = dens[..., 0]
    if MASK_RULE == "legacy":


        mask = _s0.tissue_mask(dens if dens.ndim == 2 else dens[..., 0],
                               str(row["modality"])).astype(bool)
    else:
        mask = cv2.imread(str(msk_fp), cv2.IMREAD_GRAYSCALE) > 127
    dens = _tp.clahe(_s0.to_u8(dens))


    if abs(lvl - mpp_out) > 1e-6:
        dens = _tp.to_common_grid(dens, lvl, mpp_out)
        mask = _tp.to_common_grid(mask.astype(np.uint8), lvl, mpp_out) > 0
    return dens, mask, {"mpp_eff_um": float(lvl), "prep_level_mpp": float(lvl),
                        "source": "prep_sections", "mask_rule": MASK_RULE,
                        "level_long_px": int(max(dens.shape[:2]))}


def pad_to(img, mask, shape):
    H, W = shape
    out_i = np.zeros((H, W), dtype=img.dtype)
    out_m = np.zeros((H, W), dtype=bool)
    h, w = img.shape[:2]
    out_i[:h, :w] = img[:H, :W] if (h > H or w > W) else img
    out_m[:h, :w] = mask[:H, :W] if (h > H or w > W) else mask
    return out_i, out_m


def _signed_distance(mask):
    m = mask.astype(np.uint8)
    return (cv2.distanceTransform(1 - m, cv2.DIST_L2, 5)
            - cv2.distanceTransform(m, cv2.DIST_L2, 5))


def boundary_normal_residual_um(mask_a, mask_b, mpp):
    def one(ma, mb):
        a8 = ma.astype(np.uint8)
        edge = a8 - cv2.erode(a8, np.ones((3, 3), np.uint8))
        ys, xs = np.nonzero(edge)
        if len(ys) == 0:
            return None, np.array([])
        d = np.abs(_signed_distance(mb)[ys, xs]) * mpp
        return {"median": float(np.median(d)), "p90": float(np.percentile(d, 90)),
                "p99": float(np.percentile(d, 99)), "n": int(len(d))}, d

    fwd, df = one(mask_a, mask_b)
    rev, dr = one(mask_b, mask_a)
    both = np.concatenate([x for x in (df, dr) if len(x)]) if (len(df) or len(dr)) else np.array([])
    pooled = ({"median": float(np.median(both)), "p90": float(np.percentile(both, 90)),
               "p99": float(np.percentile(both, 99)), "n": int(len(both))}
              if len(both) else None)
    return {"fixed_to_moving": fwd, "moving_to_fixed": rev, "pooled": pooled}


def tile_residual_field(fix_img, mov_img, mpp, tile_um=220.0, search_um=80.0,
                        prom_gate=0.02, cond_gate=20.0):
    tile_px = int(round(tile_um / mpp))
    search_px = int(round(search_um / mpp))
    d = _tp.tile_grid(fix_img, mov_img, tile_px, search_px)
    if len(d) == 0:
        return d, {"n_tiles": 0}
    summ = _tp.summarise(d, mpp, prom_gate, cond_gate)


    rng = np.random.default_rng(0)
    u = np.hypot(rng.uniform(-search_um, search_um, 100000),
                 rng.uniform(-search_um, search_um, 100000))
    summ["search_um"] = search_um
    summ["null_median_um"] = float(np.median(u))
    obs = summ.get("disp_um_pass", {}).get("median")
    summ["obs_over_null"] = float(obs / summ["null_median_um"]) if obs else None
    return d, summ


def sweep_search_radius(fix_img, mov_img, mpp, radii_um, tile_um=220.0):
    out = []
    for R in radii_um:
        _, s = tile_residual_field(fix_img, mov_img, mpp, tile_um, R)
        out.append(s)
    ok = [s for s in out if s.get("n_tiles") and s.get("obs_over_null") is not None]
    verdict, evidence = "INCONCLUSIVE", {}
    if ok:
        ratios = [s["obs_over_null"] for s in ok]
        disp = [s["disp_um_pass"]["median"] for s in ok]
        border0 = ok[0]["frac_on_border"]

        spread = (max(disp) - min(disp)) / max(np.median(disp), 1e-9)
        evidence = {"obs_over_null_by_radius": ratios,
                    "disp_pass_median_by_radius": disp,
                    "frac_on_border_smallest_radius": border0,
                    "disp_relative_spread": float(spread)}
        if min(ratios) > 0.5:
            verdict = "NO_CORRESPONDENCE"
        elif border0 > 0.25 and spread > 0.5:
            verdict = "WINDOW_LIMITED"
        elif min(ratios) < 0.5 and spread <= 0.5:
            verdict = "RESOLVED"
    return {"radii_um": list(radii_um), "per_radius": out,
            "verdict": verdict, "evidence": evidence}


def tangential_slip_um(tiles, fix_mask, mpp, band_um=300.0,
                       prom_gate=0.02, cond_gate=20.0):
    if len(tiles) == 0:
        return {"n": 0}
    good = tiles[(tiles["prominence"] >= prom_gate) & (~tiles["on_border"])
                 & tiles["H_posdef"] & (tiles["H_cond"] <= cond_gate)]
    if len(good) == 0:
        return {"n": 0}
    phi = _signed_distance(fix_mask)
    gy, gx = np.gradient(phi.astype(np.float32))
    norm = np.hypot(gx, gy) + 1e-9
    xs = np.clip(good["x"].to_numpy().astype(int), 0, phi.shape[1] - 1)
    ys = np.clip(good["y"].to_numpy().astype(int), 0, phi.shape[0] - 1)
    nx, ny = gx[ys, xs] / norm[ys, xs], gy[ys, xs] / norm[ys, xs]
    dx = good["dx_px"].to_numpy() * mpp
    dy = good["dy_px"].to_numpy() * mpp
    dn = dx * nx + dy * ny
    dt = np.hypot(dx - dn * nx, dy - dn * ny)
    band = np.abs(phi[ys, xs]) * mpp <= band_um
    q = lambda a, p: float(np.percentile(a, p)) if len(a) else None
    return {"n": int(len(good)), "n_in_band": int(band.sum()), "band_um": band_um,
            "tangential_um": {"median": q(dt[band], 50), "p90": q(dt[band], 90),
                              "p99": q(dt[band], 99)},
            "normal_um": {"median": q(np.abs(dn[band]), 50),
                          "p90": q(np.abs(dn[band]), 90),
                          "p99": q(np.abs(dn[band]), 99)},
            "tangential_um_all_tiles": {"median": q(dt, 50), "p90": q(dt, 90),
                                        "p99": q(dt, 99)}}


def save_overlay(fix_img, warped, fix_mask, wmask, path_stem, header):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rgb = np.zeros(fix_img.shape[:2] + (3,), np.uint8)
    rgb[..., 1] = fix_img
    rgb[..., 0] = warped
    for m, col in ((fix_mask, (255, 255, 255)), (wmask, (80, 200, 255))):
        c, _ = cv2.findContours(m.astype(np.uint8), cv2.RETR_EXTERNAL,
                                cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(rgb, c, -1, col, 2)
    fig, ax = plt.subplots(figsize=(10, 10 * rgb.shape[0] / max(rgb.shape[1], 1)))
    ax.imshow(rgb)
    ax.set_title(header, fontsize=9)
    ax.axis("off")
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(f"{path_stem}.{ext}", dpi=150, bbox_inches="tight")
    plt.close(fig)


def _sha256(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for blk in iter(lambda: f.read(1 << 20), b""):
            h.update(blk)
    return h.hexdigest()


def write_manifest(out, sections_tsv, run_id, extra):
    try:
        sha = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True,
                             text=True, cwd=str(_HERE)).stdout.strip() or "unknown"
    except Exception:
        sha = "unknown"
    (out / "run_manifest.json").write_text(json.dumps({
        "run_id": run_id, "stage": "pairwise serial-section registration",
        "scripts": [str(_HERE / "pairwise_registration.py"), str(_MODELS / "pairwise_affine.py")],
        "script_git_sha": sha,
        "data_version": {"sections_tsv": str(sections_tsv),
                         "sections_tsv_sha256": _sha256(sections_tsv)},
        "model_spec_version": "pairwise_affine.py (affine6 + IoU-weighted NCC)",
        "seed": extra.get("seed"), "deterministic": True,
        "python": sys.version.split()[0], "numpy": np.__version__,
        "opencv": cv2.__version__, "pandas": pd.__version__,
        "slurm_job_id": os.environ.get("SLURM_ARRAY_JOB_ID") or os.environ.get("SLURM_JOB_ID"),
        "cpu_hours": None, **extra}, indent=2))


DEFAULT_PAIRS = [(4, 5, "he-he_ctrl"), (40, 41, "codex-codex_ctrl"),
                 (29, 31, "he-xenium"), (30, 32, "he-codex"),
                 (44, 46, "xenium-codex")]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sections", required=True)
    ap.add_argument("--prep-root", default="",
                    help="directory holding <section_id>/img_mpp*.tif from the "
                         "prep stage. Explicit on purpose — see main().")
    ap.add_argument("--out", required=True)
    ap.add_argument("--mask-rule", default="prep", choices=["prep", "legacy"],
                    help="which tissue mask to use; single-variable isolation arm")
    ap.add_argument("--pairs", default="",
                    help='comma-separated "<uA>-<uB>" overriding DEFAULT_PAIRS')
    ap.add_argument("--index", type=int, default=-1)
    ap.add_argument("--mpp", type=float, default=2.0, help="common grid um/px")
    ap.add_argument("--climb-levels", default="auto", dest="climb_levels",
                    help="after the coarse search: auto = hill-climb on --mpp*2 then --mpp (the full solve); "
                         "none = keep the coarse answer, only rescaled to --mpp (a coarse placement for a later refinement)")
    ap.add_argument("--search-mpp", type=float, default=8.0, dest="search_mpp",
                    help="coarse grid (um/px) for the seed / Powell / basin-hop search; the "
                         "result is then hill-climbed on --mpp/2 and on --mpp. 0 = search "
                         "on the --mpp grid directly")
    ap.add_argument("--t-star-um", type=float, default=80.0,
                    help="crossover radius that sets the footprint weight R_h/t*")
    ap.add_argument("--tile-um", type=float, default=220.0)
    ap.add_argument("--search-um", type=float, default=80.0)
    ap.add_argument("--geometry", default=None,
                    help="sections_geometry.parquet -- the registered area caliber")
    ap.add_argument("--precompensate", action="store_true",
                    help="apply G_ij = sqrt(area_i/area_j) as a recorded constant")
    ap.add_argument("--reach-cut-um", type=float, default=320.0,
                    help="route A if |log G|*R_edge <= this, else route B")
    ap.add_argument("--boxes", choices=("legacy", "decoupled"), default="legacy",
                    help="constraint set. legacy = the (sx,sy) box every previous "
                         "run used and the default, so those baselines stay "
                         "reproducible on this code; decoupled is opt-in.")
    ap.add_argument("--tol-iso", type=float, default=None)
    ap.add_argument("--tol-dev", type=float, default=None)
    ap.add_argument("--tol-shear", type=float, default=None)
    ap.add_argument("--rot-step", type=float, default=10.0)
    ap.add_argument("--rot-range", type=float, default=180.0,
                    help="half-width of the rotation seed grid, degrees; 180 = the full circle")
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--patience", type=int, default=12)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--intensity", default="ncc", choices=["ncc", "ngf"],
                    help="intensity term. ngf is invariant to a global gradient "
                         "sign flip, i.e. immune to the polarity failure")
    ap.add_argument("--ngf-eta", type=float, default=1.0)
    ap.add_argument("--t-star-sweep", default="",
                    help="comma-separated t* values in um; each is solved from "
                         "the SAME seeds and scored on readouts outside the "
                         "objective (tile residual + boundary residual)")
    ap.add_argument("--sweep-radii", default="20,40,80,160,320",
                    help="search radii (um) for the tile-correspondence sweep")
    ap.add_argument("--try-flip", action="store_true",
                    help="also search the reflected branch (det<0)")
    ap.add_argument("--per-cc", action="store_true",
                    help="enable the per-connected-component stage (unvalidated)")
    args = ap.parse_args()

    global MASK_RULE, GEOM
    MASK_RULE = args.mask_rule


    GEOM = {}
    if args.precompensate:
        if not args.geometry:
            raise SystemExit("--precompensate requires --geometry "
                             "(sections_geometry.parquet: the registered area caliber)")
        _g = pd.read_parquet(args.geometry)
        need = {"section_id", "area_um2", "R_max_um"}
        if not need.issubset(_g.columns):
            raise SystemExit(f"--geometry lacks {need - set(_g.columns)}")
        GEOM = {str(r["section_id"]): r for r in _g.to_dict("records")}
        print(f"[precomp] areas from {args.geometry} ({len(GEOM)} sections)", flush=True)


    BOXES = _eng.set_boxes(args.boxes, args.tol_iso, args.tol_dev, args.tol_shear)
    print(f"[boxes] {BOXES}", flush=True)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(args.sections, sep="\t")


    prep_root = Path(args.prep_root) if args.prep_root else \
        Path(args.sections).parent / "sections"
    df["prep_dir"] = [str(prep_root / str(sid)) for sid in df["section_id"]]
    missing = [s for s, d in zip(df["section_id"], df["prep_dir"])
               if not (Path(d) / "meta.json").exists()]
    if missing:

        raise SystemExit(f"prep_root={prep_root} has no prepared section for "
                         f"{len(missing)} ids (first: {missing[:3]}). "
                         f"Pass --prep-root explicitly.")
    by_u = {int(r["u_number"]): r for r in df.to_dict("records")}
    if args.pairs:


        want = []
        for tok in args.pairs.split(","):
            tok = tok.strip()
            if not tok:
                continue
            a, b = (int(x) for x in tok.split("-"))
            want.append((a, b, f"U{a}-U{b}"))
    else:
        want = DEFAULT_PAIRS
    pairs = [(a, b, t) for a, b, t in want if a in by_u and b in by_u]
    todo = pairs if args.index < 0 else (
        [pairs[args.index]] if args.index < len(pairs) else [])

    for ua, ub, tag in todo:
        pdir = out / tag
        pdir.mkdir(parents=True, exist_ok=True)
        a, b = by_u[ua], by_u[ub]
        rec = {"tag": tag, "fixed": str(a["section_id"]), "moving": str(b["section_id"]),
               "modality_pair": f'{b["modality"]}->{a["modality"]}',
               "dz_um": abs(float(a["z_position_um"]) - float(b["z_position_um"])),
               "grid_mpp_um": args.mpp, "t_star_um": args.t_star_um,
               "engine": "pairwise_affine.py", "ok": False, "error": ""}
        t0 = time.time()
        try:
            print(f"== {tag}: {a['section_id']}({a['modality']}) <- "
                  f"{b['section_id']}({b['modality']})  dz={rec['dz_um']}um", flush=True)
            fi, fm, fmeta = prepare_section(a, args.mpp)
            mi, mm, mmeta = prepare_section(b, args.mpp)
            H = max(fi.shape[0], mi.shape[0])
            W = max(fi.shape[1], mi.shape[1])
            fi, fm = pad_to(fi, fm, (H, W))
            mi, mm = pad_to(mi, mm, (H, W))
            rec["canvas_shape"] = [int(H), int(W)]
            rec["fixed_meta"], rec["moving_meta"] = fmeta, mmeta

            gate = _eng.polarity_gate(fi, mi, fm, mm)
            rec["polarity_gate"] = gate


            pol_state = _pa.edge_polarity_state(gate.get("ncc_same_polarity"))
            rec["polarity_state"] = pol_state
            rec["polarity_floor_used"] = _pa.NCC_INFORMATIVE_FLOOR
            if pol_state in ("INVERTED", "INVERTED_MARGINAL"):
                raise RuntimeError(
                    f"polarity gate FAILED: genuinely inverted ({pol_state}) {gate}")
            if pol_state == "NO_SIGNAL":
                rec["POSE_UNSUPPORTED_CANDIDATE"] = True
                print(f"   polarity: NO_SIGNAL (|ncc|={gate.get('ncc_same_polarity'):.5f} "
                      f"< floor {_pa.NCC_INFORMATIVE_FLOOR}) -> annotate, not exclude",
                      flush=True)

            ctx = _eng.PairContext(fi, fm, mi, mm, args.mpp,
                                   t_star_um=args.t_star_um,
                                   intensity=args.intensity, ngf_eta=args.ngf_eta)


            rec["precompensation_G_DO_NOT_INCLUDE_IN_A17C"] = None
            rec["route"] = "A"
            if args.precompensate:


                gf = GEOM[str(a["section_id"])]
                gm = GEOM[str(b["section_id"])]
                a_fix, a_mov = float(gf["area_um2"]), float(gm["area_um2"])
                log_g = _pc.log_precompensation(a_fix, a_mov)
                r_edge = min(float(gf["R_max_um"]), float(gm["R_max_um"]))
                route = _pc.classify_edge(log_g, r_edge, args.reach_cut_um)
                rec["route"] = route
                rec["pose_only"] = (route == "B")
                G = (_pc.precompensation_affine(log_g, center=ctx.mov_center)
                     if route == "A" else None)
                rec["precompensation_G_DO_NOT_INCLUDE_IN_A17C"] = {
                    "log_g": log_g if route == "A" else 0.0,
                    "log_g_computed": log_g,
                    "applied": route == "A",
                    "G_2x3": None if G is None else np.asarray(G).tolist(),
                    "about": "moving footprint centroid",
                    "area_fixed_um2": a_fix, "area_moving_um2": a_mov,
                    "section_fixed": str(a["section_id"]),
                    "section_moving": str(b["section_id"]),
                    "areas_from": str(args.geometry),
                    "R_edge_um": r_edge,
                    "reach_um": _pc.reach_um(log_g, r_edge),
                    "reach_cut_um": args.reach_cut_um,
                    "written_before_solve": True}


                probe = np.array([[0.0, 0.0], [r_edge, 0.0], [0.0, r_edge],
                                  [-r_edge, r_edge]], dtype=np.float64)
                rec["assert_roundtrip"] = \
                    _pc.assert_roundtrip_with_negative_control(probe, log_g)
                rec["assert_correspondence_frame"] = \
                    _pc.assert_correspondence_frame_with_negative_control(
                        probe, probe, log_g, "fixed_original")
                ctx.set_precompensation(G)
                print(f"   precompensation: route {route}  log_g={log_g:+.5f}  "
                      f"reach={_pc.reach_um(log_g, r_edge):.0f} um", flush=True)


            rec["boxes_effective"] = _eng.effective_boxes()
            rec["precompensation_enabled"] = bool(args.precompensate)
            rec["intensity_term"] = args.intensity
            rec["ngf_eps"] = {"fixed": ctx.ngf_eps_fix, "moving": ctx.ngf_eps_mov}
            rec["R_h_um"] = ctx.R_h_um
            rec["w_silhouette"] = ctx.w_silh
            rec["baseline"] = {
                "iou_no_registration": _eng.iou_masks(fm, mm),
                "boundary_um_no_registration": boundary_normal_residual_um(fm, mm, args.mpp)}


            _rr = float(getattr(args, "rot_range", 180.0))
            rot_grid = list(np.arange(-_rr, _rr, args.rot_step))
            if args.search_mpp and args.search_mpp > args.mpp:


                def _coarse(img, msk, f):
                    h2, w2 = max(1, int(round(img.shape[0] / f))), max(1, int(round(img.shape[1] / f)))
                    im2 = cv2.resize(img, (w2, h2), interpolation=cv2.INTER_AREA)
                    mk2 = cv2.resize(msk.astype(np.uint8) * 255, (w2, h2),
                                     interpolation=cv2.INTER_AREA) > 127
                    return im2, mk2
                ladder = [args.search_mpp]
                if args.mpp * 2 < args.search_mpp:
                    ladder.append(args.mpp * 2)
                t_c = time.time()
                f0 = args.search_mpp / args.mpp
                cfi, cfm = _coarse(fi, fm, f0); cmi, cmm = _coarse(mi, mm, f0)
                cctx = _eng.PairContext(cfi, cfm, cmi, cmm, args.search_mpp,
                                        t_star_um=args.t_star_um,
                                        intensity=args.intensity, ngf_eta=args.ngf_eta)
                if args.precompensate and rec["route"] == "A":
                    cctx.set_precompensation(
                        _pc.precompensation_affine(log_g, center=cctx.mov_center))
                params, flip, report = _eng.solve_pair(
                    cctx, rot_grid=rot_grid, try_flip=args.try_flip,
                    n_epochs=args.epochs, patience=args.patience, seed=args.seed)
                report["search_grid_mpp"] = args.search_mpp
                report["search_seconds"] = round(time.time() - t_c, 1)
                print(f"   coarse search at {args.search_mpp:g} um/px: score "
                      f"{report['best']['score']:.4f} flip={flip} in {time.time() - t_c:.0f} s", flush=True)
                prev_mpp = args.search_mpp
                climbs = []
                if args.climb_levels == "none":

                    params = np.asarray(params, float).copy()
                    params[4] *= prev_mpp / args.mpp; params[5] *= prev_mpp / args.mpp
                    prev_mpp = args.mpp
                for m_next in ([] if args.climb_levels == "none" else ([args.mpp * 2] if args.mpp * 2 < args.search_mpp else []) + [args.mpp]):
                    t_c = time.time()
                    f = prev_mpp / m_next
                    params = np.asarray(params, float).copy()
                    params[4] *= f; params[5] *= f
                    if abs(m_next - args.mpp) < 1e-9:
                        fctx = ctx
                    else:
                        ffi, ffm = _coarse(fi, fm, m_next / args.mpp); fmi, fmm = _coarse(mi, mm, m_next / args.mpp)
                        fctx = _eng.PairContext(ffi, ffm, fmi, fmm, m_next,
                                                t_star_um=args.t_star_um,
                                                intensity=args.intensity, ngf_eta=args.ngf_eta)
                        if args.precompensate and rec["route"] == "A":
                            fctx.set_precompensation(
                                _pc.precompensation_affine(log_g, center=fctx.mov_center))
                    params, sc = _eng.fine_climb(params, fctx, flip,
                                                 max_iter=200 if fctx is not ctx else 60)
                    climbs.append({"mpp": m_next, "score": float(sc), "seconds": round(time.time() - t_c, 1)})
                    print(f"   climb at {m_next:g} um/px: score {sc:.4f} in {time.time() - t_c:.0f} s", flush=True)
                    prev_mpp = m_next
                _, det = _eng.objective(params, ctx, flip, detail=True)
                report["best"] = {"params": dict(zip(_eng.PARAM_NAMES, map(float, params))),
                                  "flip": bool(flip), "score": float(det.get("intensity") if False else _eng.objective(params, ctx, flip)), **det}
                report["climbs"] = climbs
            else:
                params, flip, report = _eng.solve_pair(
                    ctx, rot_grid=rot_grid, try_flip=args.try_flip,
                    n_epochs=args.epochs, patience=args.patience, seed=args.seed)
            rec["solve"] = report
            rec["fit"] = dict(zip(_eng.PARAM_NAMES, map(float, params)))
            rec["fit"]["flip"] = bool(flip)
            rec["anisotropy_pct"] = 100.0 * abs(params[1] - params[2])

            R = _eng.recenter_params(*params, center=ctx.fix_center, flip=flip)


            rec["residual_fit_R"] = np.asarray(R).tolist()
            rec["strain_R_residual_only"] = _eng.strain_decomposition(R)
            _G = (rec.get("precompensation_G_DO_NOT_INCLUDE_IN_A17C") or {}).get("G_2x3")
            M = R if _G is None else _eng.compose_affine(R, np.asarray(_G))
            rec["M_moving_to_fixed"] = np.asarray(M).tolist()


            rec["strain"] = _eng.strain_decomposition(M)
            warped, wmask = ctx.warp(M)

            rec["iou_after"] = _eng.iou_masks(fm, wmask)
            rec["boundary_um_after"] = boundary_normal_residual_um(fm, wmask, args.mpp)
            tiles, tsum = tile_residual_field(fi, warped, args.mpp,
                                              args.tile_um, args.search_um)
            rec["tiles"] = tsum
            rec["tangential_slip"] = tangential_slip_um(tiles, fm, args.mpp)
            if len(tiles):
                tiles.to_parquet(pdir / "tile_residuals.parquet", index=False)


            rec["intensity_profile"] = _eng.intensity_profile(ctx, params, flip)


            radii = [float(x) for x in args.sweep_radii.split(",") if x.strip()]
            rec["radius_sweep"] = sweep_search_radius(fi, warped, args.mpp,
                                                      radii, args.tile_um)
            print(f'   radius sweep verdict: {rec["radius_sweep"]["verdict"]}', flush=True)


            if args.t_star_sweep:
                arms = []
                for ts in [float(x) for x in args.t_star_sweep.split(",") if x.strip()]:
                    c2 = _eng.PairContext(fi, fm, mi, mm, args.mpp, t_star_um=ts,
                                          intensity=args.intensity, ngf_eta=args.ngf_eta)
                    p2, f2, _ = _eng.solve_pair(
                        c2, rot_grid=rot_grid, try_flip=args.try_flip,
                        n_epochs=args.epochs, patience=args.patience,
                        seed=args.seed, log=lambda *_: None)
                    M2 = _eng.recenter_params(*p2, center=c2.fix_center, flip=f2)
                    w2, wm2 = c2.warp(M2)
                    t2, ts2 = tile_residual_field(fi, w2, args.mpp,
                                                  args.tile_um, args.search_um)
                    b2 = boundary_normal_residual_um(fm, wm2, args.mpp)
                    arms.append({
                        "t_star_um": ts, "w_silh": c2.w_silh,
                        "fit": dict(zip(_eng.PARAM_NAMES, map(float, p2))),
                        "anisotropy_pct": 100.0 * abs(p2[1] - p2[2]),
                        "OUT_OF_OBJECTIVE_tile_disp_median_um":
                            ts2.get("disp_um_pass", {}).get("median"),
                        "OUT_OF_OBJECTIVE_tile_disp_p90_um":
                            ts2.get("disp_um_pass", {}).get("p90"),
                        "OUT_OF_OBJECTIVE_boundary_pooled_p90_um":
                            (b2.get("pooled") or {}).get("p90"),
                        "iou": _eng.iou_masks(fm, wm2)})
                    print(f'   t*={ts:6.1f} w={c2.w_silh:7.2f} '
                          f'tile_med={arms[-1]["OUT_OF_OBJECTIVE_tile_disp_median_um"]} '
                          f'bnd_p90={arms[-1]["OUT_OF_OBJECTIVE_boundary_pooled_p90_um"]} '
                          f'aniso={arms[-1]["anisotropy_pct"]:.2f}%', flush=True)
                rec["t_star_arms"] = arms

            if args.per_cc:
                rec["per_cc"] = _eng.refine_per_cc(ctx, M, flip=flip)

            bb = rec["baseline"]["boundary_um_no_registration"]["pooled"]
            ba = rec["boundary_um_after"]["pooled"]
            save_overlay(fi, warped, fm, wmask, str(pdir / "overlay"),
                         f'{tag}  {rec["modality_pair"]}  dz={rec["dz_um"]:.0f}um   '
                         f'boundary(pooled) {bb["median"]:.1f} -> {ba["median"]:.1f} um   '
                         f'IoU {rec["baseline"]["iou_no_registration"]:.3f} -> '
                         f'{rec["iou_after"]:.3f}')
            rec["ok"] = True
            print(f'   boundary pooled med {bb["median"]:7.1f} -> {ba["median"]:7.1f} um  '
                  f'p90 {bb["p90"]:7.1f} -> {ba["p90"]:7.1f}  p99 {ba["p99"]:7.1f}',
                  flush=True)
            print(f'   IoU {rec["baseline"]["iou_no_registration"]:.3f} -> '
                  f'{rec["iou_after"]:.3f}   slip p50/p90 '
                  f'{rec["tangential_slip"].get("tangential_um", {}).get("median")}/'
                  f'{rec["tangential_slip"].get("tangential_um", {}).get("p90")} um',
                  flush=True)
            print(f'   fit theta={rec["fit"]["theta_deg"]:.2f} sx={rec["fit"]["sx"]:.4f} '
                  f'sy={rec["fit"]["sy"]:.4f} shear={rec["fit"]["shear"]:.4f} '
                  f'| anisotropy {rec["anisotropy_pct"]:.2f}%', flush=True)
        except Exception as e:
            import traceback
            rec["error"] = f"{type(e).__name__}: {e}"
            rec["traceback"] = traceback.format_exc()[-2000:]
            print(f"   FAILED: {rec['error']}", flush=True)
        rec["elapsed_s"] = round(time.time() - t0, 1)


        rec["slurm_job_id"] = (os.environ.get("SLURM_ARRAY_JOB_ID")
                               or os.environ.get("SLURM_JOB_ID"))
        rec["slurm_task_id"] = os.environ.get("SLURM_ARRAY_TASK_ID")
        rec["written_at"] = time.strftime("%Y-%m-%dT%H:%M:%S%z")

        (pdir / "results.json").write_text(json.dumps(rec, indent=2, default=str))

    write_manifest(out, args.sections, "pairwise",
                   {"seed": args.seed, "mpp": args.mpp, "t_star_um": args.t_star_um,
                    "tile_um": args.tile_um, "search_um": args.search_um,
                    "rot_step_deg": args.rot_step, "try_flip": bool(args.try_flip),
                    "per_cc": bool(args.per_cc)})


if __name__ == "__main__":
    main()
