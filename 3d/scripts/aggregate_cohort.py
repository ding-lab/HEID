#!/usr/bin/env python3

import argparse
import csv
import glob
import json
import os
import struct
import sys

import numpy as np
import pandas as pd

P3D = os.environ.get("THREED_ROOT", os.path.join(os.environ.get("PROJECTS_ROOT", "/data/heid"), "3d"))
TLS = os.path.join(P3D, "tls_define")
OUT = os.path.join(TLS, "outputs")
ROOTS = {}
CENSUS = os.path.join(P3D, "analysis/rare_event_census/rare_event_census.tsv")
VOL = os.path.join(P3D, "reconstruction/volume_8um")
VOLS = {"HT891Z1": VOL, "HT913Z1": os.path.join(P3D, "reconstruction/volume_8um_913")}
PRED = os.path.join(P3D, "inference/cohort/data/predictions")
UM_PER_CANVAS_PX = 8.0

CAVEATS = [
    "every cell type is a pan-cancer Cell MODEL PREDICTION on H&E; this cohort has no "
    "spatial ground truth, so no TLS here is scored",
    "no molecular grading: no expression in this cohort, and the Cell model has no DC class, so "
    "TLS_State / GC rung / development score have no implementation at all",
    "the T gate counts the merged NK_T class, so NK cells count toward MIN_T_IN_REGION",
    "HT913Z1 has never been registered: its canvas coordinates do not exist and are "
    "left empty, not substituted with native ones",
    "all seven acceptance gates are at the source values; none was relaxed",
]


def load_canvas_map(slide, basis, sample="HT891Z1"):
    vol = VOLS.get(sample, VOL)
    if not hasattr(load_canvas_map, "metas"):
        load_canvas_map.metas = {}
    if vol not in load_canvas_map.metas:
        mp = os.path.join(vol, "cell_points_metadata.json")
        load_canvas_map.metas[vol] = json.load(open(mp)) if os.path.exists(mp) else None
    meta = load_canvas_map.metas[vol]
    if meta is None:
        return None
    secs = [s for s in meta["sections"] if s["slide"] == slide]
    if not secs:
        return None
    sid = secs[0]["section_id"]
    f = [x for x in meta["files"] if x["basis"] == basis and x["section_id"] == sid]
    if not f:
        return None
    raw = open(os.path.join(vol, f[0]["file"]), "rb").read()
    magic, n, cw, ch, sub, ncls, _ = struct.unpack("<8sIHHHHI", raw[:24])
    assert magic == b"CELLPT01"
    xy = np.frombuffer(raw[32:32 + 4 * n], dtype="<u2").reshape(n, 2).astype(np.float64)
    cl = np.frombuffer(raw[32 + 5 * n:32 + 6 * n], dtype=np.uint8)
    pred = pd.read_parquet(os.path.join(PRED, slide + ".parquet"),
                           columns=["pred_class_name", "x_um", "y_um"])
    if len(pred) != n:
        return None
    idx = pred["pred_class_name"].map({c: i for i, c in enumerate(meta["classes"])}).to_numpy()
    if not np.array_equal(idx, cl):
        return None
    return {"canvas_um": xy / sub * UM_PER_CANVAS_PX,
            "native_um": pred[["x_um", "y_um"]].to_numpy(), "section_id": sid}


def main(basis):
    global OUT
    census = {r["slide_id"]: r for r in csv.DictReader(open(CENSUS), delimiter="\t")
              if r["status"] == "ok"}

    per_slide, tls_rows = [], []
    missing = []
    for slide, c in sorted(census.items(), key=lambda kv: (kv[1]["sample_id"],
                                                           int(kv[1]["section_u"]))):
        sample = c["sample_id"]
        d = os.path.join(ROOTS.get(sample, os.path.join(OUT, sample)), slide)
        mp = os.path.join(d, f"{slide}_meta.json")
        if not os.path.exists(mp):
            missing.append(slide)
            continue
        m = json.load(open(mp))
        row = {"sample_id": sample, "block_id": c["block_id"],
               "section_u": int(c["section_u"]), "z_um": float(c["z_um"]),
               "slide_id": slide, "he_plane_kind": c["he_plane_kind"],
               "n_cells": m["n_cells"], "n_B": m["n_B"], "b_fraction": m["b_fraction"],
               "n_T_NK": m["n_T_NK"], "n_Plasma": m["n_Plasma"],
               "tissue_area_mm2": m["tissue_area_mm2"],
               "n_candidate_cores": m["n_candidate_cores"],
               "n_tls": m["n_tls_accepted"],
               "n_tagg": m["n_tagg"], "n_magg": m["n_magg"]}
        for k, v in m["rejected_by"].items():
            row[f"rejected_{k}"] = v
        row["zero_reason"] = ""
        if m["n_tls_accepted"] == 0:
            if m["n_candidate_cores"] == 0:
                row["zero_reason"] = "no candidate core formed"
            else:
                worst = max(m["rejected_by"].items(), key=lambda kv: kv[1])
                row["zero_reason"] = f"all candidates rejected, most by: {worst[0]}"
        per_slide.append(row)

        tp = os.path.join(d, f"{slide}_tls.csv")
        if m["n_tls_accepted"] and os.path.exists(tp):
            cm = load_canvas_map(slide, basis, sample)
            tdf = pd.read_csv(tp)
            if "tls_local" in tdf:
                tdf["tls_id"] = tdf["tls_local"]

            cp = os.path.join(d, f"{slide}_tls_cells.csv")
            core_idx = None
            if cm is not None and os.path.exists(cp):
                pc = pd.read_csv(cp)
                if "tls_local_halo150" in pc:
                    pc["tls_region_halo150"] = pc["tls_local_halo150"]
                if len(pc) == len(cm["canvas_um"]):
                    core_idx = pc
            for _, t in tdf.iterrows():
                r = {"sample_id": sample, "block_id": c["block_id"],
                     "section_u": int(c["section_u"]), "z_um": float(c["z_um"]),
                     "slide_id": slide, "tls_id": t["tls_id"],
                     "core_area_mm2": t["core_area_mm2"],
                     "region_area_mm2_halo150": t["region_area_mm2_halo150"],
                     "n_b_core": t["n_b_core"],
                     "core_b_density_mm2": t["core_b_density_mm2"],
                     "core_class": t["core_class"],
                     "rescued": bool(t.get("rescued", False)),
                     "tls_composition_halo150": t.get("tls_composition_halo150"),
                     "plasma_fraction_halo150": t.get("plasma_fraction_halo150"),
                     "cx_um_native": round(float(t["cx_um"]), 1),
                     "cy_um_native": round(float(t["cy_um"]), 1),
                     "cx_um_canvas": None, "cy_um_canvas": None,
                     "canvas_basis": basis if cm is not None else ""}
                if core_idx is not None:


                    in_core = core_idx["in_tls_core"].astype(str).isin(
                        ["True", "true", "1"]).to_numpy()
                    cents = tdf[["cx_um", "cy_um"]].to_numpy()
                    nat = cm["native_um"]
                    d2 = ((nat[in_core][:, None, 0] - cents[None, :, 0]) ** 2 +
                          (nat[in_core][:, None, 1] - cents[None, :, 1]) ** 2)
                    owner = np.argmin(d2, axis=1)
                    sel = np.zeros(len(nat), bool)
                    sel[np.flatnonzero(in_core)[owner == (int(t["tls_id"].split("-")[1]) - 1)]] = True
                    if sel.sum():
                        pts = cm["canvas_um"][sel]
                        r["cx_um_canvas"] = round(float(pts[:, 0].mean()), 1)
                        r["cy_um_canvas"] = round(float(pts[:, 1].mean()), 1)
                tls_rows.append(r)

    ps = pd.DataFrame(per_slide).sort_values(["sample_id", "block_id", "section_u"])
    ps.to_csv(os.path.join(OUT, "cohort_per_slide.tsv"), sep="\t", index=False)
    t3 = pd.DataFrame(tls_rows)
    if len(t3):
        t3 = t3.sort_values(["sample_id", "block_id", "z_um", "tls_id"])

    p3 = os.path.join(OUT, "tls_3d_table.tsv")
    with open(p3, "w") as fh:
        fh.write("# ONLY cx_um_canvas / cy_um_canvas can be stacked along z.\n"
                 "# cx_um_native / cy_um_native are in each slide's OWN frame; consecutive\n"
                 "# slides differ by a translation and a rotation, so stacking the native\n"
                 "# pair produces a shape made of registration error, not anatomy.\n"
                 "# Canvas coordinates exist for HT891Z1 only (basis " + basis + "); HT913Z1 has\n"
                 "# never been registered and its canvas columns are EMPTY, not back-filled.\n"
                 "# Every cell type is a pan-cancer Cell MODEL PREDICTION on H&E; no ground truth.\n"
                 "# Read with: pd.read_csv(path, sep='\\t', comment='#')\n")
        t3.to_csv(fh, sep="\t", index=False)


    link = {}
    for s in (sorted(t3.sample_id.unique()) if len(t3) else []):
        sub = t3[(t3.sample_id == s) & t3.cx_um_canvas.notna()]
        if not len(sub):
            link[s] = {"status": "no canvas coordinates (sample not registered)"}
            continue
        for blk in sorted(sub.block_id.unique()):
            b = sub[sub.block_id == blk]
            zs = sorted(b.z_um.unique())


            allz = sorted(ps[(ps.sample_id == s) & (ps.block_id == blk)].z_um.unique())
            cross, pairs_used = [], 0
            for z0, z1 in zip(allz, allz[1:]):
                a = b[b.z_um == z0]
                c = b[b.z_um == z1]
                if not len(a) or not len(c):
                    continue
                pairs_used += 1
                for _, r0 in a.iterrows():
                    for _, r1 in c.iterrows():
                        cross.append(float(np.hypot(r0.cx_um_canvas - r1.cx_um_canvas,
                                                    r0.cy_um_canvas - r1.cy_um_canvas)))
            within = []
            for z in zs:
                g = b[b.z_um == z]
                v = g[["cx_um_canvas", "cy_um_canvas"]].to_numpy()
                for i in range(len(v)):
                    for j in range(i + 1, len(v)):
                        within.append(float(np.hypot(*(v[i] - v[j]))))

            nn = []
            for z0, z1 in zip(allz, allz[1:]):
                a = b[b.z_um == z0]
                c = b[b.z_um == z1]
                if not len(a) or not len(c):
                    continue
                cv = c[["cx_um_canvas", "cy_um_canvas"]].to_numpy()
                for _, r0 in a.iterrows():
                    d = np.hypot(cv[:, 0] - r0.cx_um_canvas, cv[:, 1] - r0.cy_um_canvas)
                    nn.append(float(d.min()))


            rng = np.random.default_rng(20260819)
            nullmed = []
            for _ in range(200):
                perm = list(zs)
                rng.shuffle(perm)
                mp = dict(zip(zs, perm))
                v = []
                for z0, z1 in zip(allz, allz[1:]):
                    if z0 in mp and z1 in mp and mp[z0] != mp[z1]:
                        a = b[b.z_um == z0]
                        c = b[b.z_um == mp[z1]]
                        if not len(a) or not len(c):
                            continue
                        cv = c[["cx_um_canvas", "cy_um_canvas"]].to_numpy()
                        for _, r0 in a.iterrows():
                            v.append(float(np.hypot(cv[:, 0] - r0.cx_um_canvas,
                                                    cv[:, 1] - r0.cy_um_canvas).min()))
                if v:
                    nullmed.append(float(np.median(v)))
            eqd = (2 * np.sqrt(b.core_area_mm2.median() * 1e6 / np.pi)) if len(b) else None
            link[f"{s}/{blk}"] = {
                "shuffled_layer_null_median_um": (round(float(np.median(nullmed)), 1)
                                                  if nullmed else None),
                "shuffled_layer_null_p05_p95": ([round(float(np.percentile(nullmed, 5)), 1),
                                                 round(float(np.percentile(nullmed, 95)), 1)]
                                                if nullmed else None),
                "p_empirical_null_le_observed": (round(float(np.mean(
                    np.array(nullmed) <= np.median(nn))), 4) if (nullmed and nn) else None),
                "median_core_equivalent_diameter_um": round(float(eqd), 0) if eqd else None,
                "frac_true_partners_within_one_core_diameter": (
                    round(float(np.mean(np.array(nn) < eqd)), 2) if (nn and eqd) else None),
                "n_sections_in_block": len(allz),
                "n_sections_with_tls": len(zs),
                "n_adjacent_section_pairs_both_with_tls": pairs_used,
                "cross_layer_all_pairs_n": len(cross),
                "cross_layer_median_um": round(float(np.median(cross)), 1) if cross else None,
                "cross_layer_nearest_partner_median_um": round(float(np.median(nn)), 1) if nn else None,
                "within_layer_control_n": len(within),
                "within_layer_control_median_um": round(float(np.median(within)), 1) if within else None,
                "reading": ("cross-layer nearest-partner distance well below the within-layer "
                            "control means adjacent sections cut the same object; comparable "
                            "medians mean the blobs are independent per layer"),
            }

    summary = {"basis_for_canvas_coords": basis, "caveats": CAVEATS,
               "z_linkage": link,
               "n_slides_expected": len(census), "n_slides_done": len(per_slide),
               "slides_missing": missing, "per_sample": {}}
    for s in sorted(ps.sample_id.unique()):
        sub = ps[ps.sample_id == s]
        sub_t = t3[t3.sample_id == s] if len(t3) else t3
        summary["per_sample"][s] = {
            "n_slides": int(len(sub)),
            "n_slides_with_tls": int((sub.n_tls > 0).sum()),
            "n_tls_total": int(sub.n_tls.sum()),
            "n_tagg_total": int(sub.n_tagg.sum()),
            "n_magg_total": int(sub.n_magg.sum()),
            "B_fraction_min_max": [float(sub.b_fraction.min()), float(sub.b_fraction.max())],
            "z_span_um_with_tls": ([float(sub_t.z_um.min()), float(sub_t.z_um.max())]
                                   if len(sub_t) else None),
            "core_area_mm2_median": (round(float(sub_t.core_area_mm2.median()), 4)
                                     if len(sub_t) else None),
            "zero_reasons": (sub[sub.n_tls == 0].zero_reason.value_counts().to_dict()),
            "canvas_coords_available": bool(len(sub_t) and sub_t.cx_um_canvas.notna().any()),
        }
    with open(os.path.join(OUT, "cohort_summary.json"), "w") as fh:
        json.dump(summary, fh, indent=2)
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--basis", default="G_withdrawn")
    ap.add_argument("--root", action="append", default=[],
                    help="SAMPLE=DIR: where that sample's per-slide products live (samples: <sample>/raw/tls_per_slide)")
    ap.add_argument("--volume", action="append", default=[],
                    help="SAMPLE=DIR: the volume whose cell points give that sample's canvas coordinates")
    ap.add_argument("--out-dir", default="", dest="out_dir",
                    help="where cohort_per_slide.tsv / tls_3d_table.tsv / cohort_summary.json go; default tls_define/outputs")
    a = ap.parse_args()
    for kv in a.root:
        k, v = kv.split("=", 1); ROOTS[k] = v
    for kv in a.volume:
        k, v = kv.split("=", 1); VOLS[k] = v
    if a.out_dir:
        OUT = a.out_dir
    sys.exit(main(a.basis))
