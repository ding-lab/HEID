#!/usr/bin/env python3
import argparse
import os
import shutil

import numpy as np
import pandas as pd

MIN_CORE_AREA_MM2 = 0.05


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--objects", required=True)
    ap.add_argument("--table", required=True)
    ap.add_argument("--sample", default="")
    ap.add_argument("--frozen", default="", help="tls3d_frozen.tsv: the sample's 3-D TLS and their "
                    "numbers, fixed; each is matched to the linker object sharing most members")
    a = ap.parse_args()

    objs = pd.read_csv(a.objects, sep="\t")
    raw = os.path.join(os.path.dirname(a.objects), "objects_linker.tsv")
    if not os.path.exists(raw):
        shutil.copy2(a.objects, raw)
    tab = pd.read_csv(a.table, sep="\t", comment="#")
    if a.sample and "sample_id" in tab:
        tab = tab[tab.sample_id == a.sample]
    row_of = {(r.slide_id, r.tls_id): r for r in tab.itertuples()}


    frac = objs["link_margin_frac_ge2"] if "link_margin_frac_ge2" in objs else (objs.link_margin_min.fillna(0) >= 2).astype(float)


    big = objs.core_area_mm2_median.fillna(0) >= MIN_CORE_AREA_MM2
    objs["tier"] = np.where((objs.n_measured_sections >= 4) & (frac.fillna(0) >= 0.75) & big, "A",
                            np.where(objs.n_measured_sections >= 2, "B", "C"))
    keep = objs[objs.tier == "A"].sort_values(["n_measured_sections", "z_min_um"], ascending=[False, True])
    out = []
    if a.frozen and os.path.exists(a.frozen):


        fz = pd.read_csv(a.frozen, sep="\t")
        members = {o.Index: set(str(o.member_tls).split(";")) for o in objs.itertuples()}
        taken, rows = set(), []
        for f in fz.itertuples():
            want = set(str(f.member_tls).split(";"))
            best, score = None, 0.0
            for i, m in members.items():
                if i in taken:
                    continue
                j = len(want & m) / max(len(want | m), 1)
                if j > score:
                    best, score = i, j
            if best is None or score < 0.25:


                print(f"  RETIRED {f.object_id}: no linker object shares its frozen members (best {score:.2f}); "
                      f"the number stays unused until the sample is re-frozen")
                continue
            if score < 0.5:
                print(f"  WARNING {f.object_id}: the linker object sharing most members overlaps only {score:.2f} "
                      f"(the linking changed under it, e.g. after a re-alignment); it keeps the id, see frozen_overlap")
            taken.add(best)
            d = objs.loc[best].to_dict(); d["object_id"] = f.object_id; d["tier"] = "A"
            d["frozen_overlap"] = round(score, 3)
            rows.append(d)
            print(f"  {f.object_id}: linker object with {len(members[best])} cores, overlap {score:.2f}")


        for o in keep.itertuples():
            if o.Index not in taken:
                print(f"  NOT FROZEN: a tier-A linker object with {int(o.n_measured_sections)} sections "
                      f"(z {o.z_min_um:g}-{o.z_max_um:g}, canvas {o.cx_um_canvas:.0f},{o.cy_um_canvas:.0f}) matches no "
                      f"frozen id and stays 2-D; re-freeze the sample to make it 3-D")
        rows.sort(key=lambda d: d["object_id"])
        out.extend(rows)
        objs["tier"] = np.where(objs.index.isin(taken), "A", np.where(objs.n_measured_sections >= 2, "B", "C"))
        keep = objs[objs.tier == "A"]
    else:
        for k, o in enumerate(keep.itertuples(), 1):
            d = o._asdict(); d.pop("Index", None); d["object_id"] = f"3d-tls-{k:02d}"
            out.append(d)
    loose = []
    for o in objs[objs.tier != "A"].itertuples():
        for tok in str(o.member_tls).split(";"):
            if ":" in tok:
                s, t = tok.rsplit(":", 1)
                r = row_of.get((s, t))
                loose.append((float(r.z_um) if r is not None else 1e9, s, t, r))
    for k, (z, s, t, r) in enumerate(sorted(loose, key=lambda q: (q[0], q[1], q[2])), 1):
        out.append({"object_id": f"2d-tls-{k:02d}", "n_measured_sections": 1,
                    "z_min_um": z, "z_max_um": z, "z_extent_um": 0.0, "n_cores": 1,
                    "member_tls": f"{s}:{t}",
                    "cx_um_canvas": r.cx_um_canvas if r is not None else np.nan,
                    "cy_um_canvas": r.cy_um_canvas if r is not None else np.nan,
                    "core_area_mm2_median": r.core_area_mm2 if r is not None else np.nan,
                    "link_dist_um_median": np.nan, "link_margin_min": np.nan, "link_margin_frac_ge2": np.nan,
                    "max_dz_um": np.nan, "n_bridges": 0, "n_rescued": 0,
                    "crosses_gap_ge25um": False, "p_link": 1.0, "tier": "C"})
    df = pd.DataFrame(out)
    df.to_csv(a.objects, sep="\t", index=False)
    n_b = int((objs.tier == "B").sum()); n_c = int((objs.tier == "C").sum())
    print(f"{a.sample or ''}: {len(keep)} tier-A objects -> 3d-tls-01..{len(keep):02d}; "
          f"{len(loose)} 2-D TLS from {n_b} tier-B and {n_c} tier-C objects -> 2d-tls-01..; "
          f"{len(tab)} 2-D TLS in the table")


if __name__ == "__main__":
    main()
