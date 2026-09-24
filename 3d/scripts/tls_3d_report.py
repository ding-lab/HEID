#!/usr/bin/env python3
import argparse
import json
import os
import re

import pandas as pd


_U_OF_SID = None


def _sections_u() -> dict:
    global _U_OF_SID
    if _U_OF_SID is None:
        _U_OF_SID = {}
        p = os.environ.get("TLS_SECTIONS", "")
        if p and os.path.exists(p):
            t = pd.read_csv(p, sep="\t", dtype=str)
            if "u_number" in t.columns:
                _U_OF_SID = {s: int(u) for s, u in zip(t["section_id"], t["u_number"])}
    return _U_OF_SID


def u_of(section_id: str) -> int:
    u = _sections_u().get(str(section_id))
    if u is not None:
        return u
    m = re.search(r"U(\d+)$", str(section_id))
    return int(m.group(1)) if m else -1


def u_of_row(r) -> int:
    if hasattr(r, "section_u"):
        return int(r.section_u)
    return u_of(r.section_id)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sample", required=True)
    ap.add_argument("--objects", required=True)
    ap.add_argument("--table", required=True)
    ap.add_argument("--meshes", required=True)
    ap.add_argument("--out", required=True, help="output path without extension")
    a = ap.parse_args()

    objs = pd.read_csv(a.objects, sep="\t")
    tab = pd.read_csv(a.table, sep="\t", comment="#")
    tab = tab[tab.sample_id == a.sample] if "sample_id" in tab else tab
    row_of = {(r.slide_id, r.tls_id): r for r in tab.itertuples()}
    vol_of = {}
    for m in json.load(open(a.meshes))["meshes"]:
        if m.get("structure") == "TLS object":
            vol_of[m["object_id"]] = m.get("volume_mm3")

    if "volume_mm3" in objs:
        for o in objs.itertuples():
            if pd.notna(o.volume_mm3):
                vol_of[o.object_id] = float(o.volume_mm3)

    orows, mrows = [], []
    for o in objs.itertuples():
        members = [tuple(t.rsplit(":", 1)) for t in str(o.member_tls).split(";") if ":" in t]
        orows.append({
            "tls_object": o.object_id, "tier": o.tier,
            "n_sections": int(o.n_measured_sections),
            "z_min_um": o.z_min_um, "z_max_um": o.z_max_um, "z_extent_um": o.z_extent_um,
            "volume_mm3": vol_of.get(o.object_id),
            "p_link": o.p_link, "above_chance": bool(o.n_measured_sections >= 5 and o.p_link <= 0.05),
            "cx_um_canvas": o.cx_um_canvas, "cy_um_canvas": o.cy_um_canvas,
            "core_area_mm2_median": o.core_area_mm2_median,
            "members": "; ".join(f"U{u_of_row(row_of[(s, t)]) if (s, t) in row_of else '?'}:{t}"
                                 for s, t in members)})
        for k, (s, t) in enumerate(members):
            r = row_of.get((s, t))
            mrows.append({
                "tls_object": o.object_id, "tier": o.tier,
                "section_u": u_of_row(r) if r is not None else None,
                "z_um": r.z_um if r is not None else None,
                "slide_tls_id": f"{s}:{t}",

                "n_cells_core": (getattr(r, "n_cells_core", None) if hasattr(r, "n_cells_core")
                                 else getattr(r, "n_b_core", None)) if r is not None else None,
                "core_area_mm2": getattr(r, "core_area_mm2", None) if r is not None else None,
                "object_volume_mm3": vol_of.get(o.object_id) if k == 0 else None})
    od, md = pd.DataFrame(orows), pd.DataFrame(mrows)
    md.to_csv(a.out + ".tsv", sep="\t", index=False)
    print(f"{a.sample}: {len(od)} objects, {len(md)} member TLS -> {a.out}.tsv")


if __name__ == "__main__":
    main()
