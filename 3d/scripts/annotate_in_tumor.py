#!/usr/bin/env python3
import argparse
import json
import os
import re
import struct
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.ndimage import binary_dilation

UM = 8.0
TOUCH_UM = 16.0


def read_cells_bin(fp):
    b = open(fp, "rb").read()
    magic, n, cw, ch, sub, ncls = struct.unpack_from("<8sIHHHH", b, 0)
    xy = np.frombuffer(b, np.uint16, 2 * n, 32).reshape(n, 2).astype(np.float64) / sub
    return xy, cw, ch


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--volume", required=True)
    ap.add_argument("--tls-root", required=True, dest="tls_root")
    ap.add_argument("--deliv", required=True)
    ap.add_argument("--s14", required=True)
    ap.add_argument("--planes", default="tumor_planes.npz",
                    help="the territory masks under <volume> (build_tumor_planes.py)")
    ap.add_argument("--regions", default="tumor_regions_metadata.json",
                    help="the per-section areas that go with --planes (tumor_regions_opt.json for the opt masks)")
    a = ap.parse_args()
    vol, deliv, S14 = Path(a.volume), Path(a.deliv), Path(a.s14)
    tp = np.load(vol / a.planes)
    it = max(1, int(round(TOUCH_UM / UM)))
    TM = {k: binary_dilation(tp[k], iterations=it) for k in tp.files}
    meta = json.loads((vol / "cell_points_metadata.json").read_text())
    vmeta = json.loads((vol / "metadata.json").read_text())
    planes = sorted(vmeta["encodings"]["G_withdrawn"]["planes"], key=lambda p: p["z_um"])
    slide_of = {s["section_id"]: s["slide"] for s in meta["sections"]}
    cw, ch = meta["canvas_px"]["width"], meta["canvas_px"]["height"]

    def inside(sid, xpx, ypx):
        m = TM.get(sid)
        if m is None:
            return np.zeros(len(xpx), bool)
        xi = np.clip(np.round(xpx).astype(int), 0, cw - 1)
        yi = np.clip(np.round(ypx).astype(int), 0, ch - 1)
        return m[yi, xi]


    tls_hit, tls_secs = {}, {}
    for p in planes:
        sid = p["section_id"]; slide = slide_of.get(sid)
        cp = Path(a.tls_root) / (slide or "-") / f"{slide}_tls_cells.csv"
        cand = sorted((vol / "G_withdrawn/cell_points").glob(f"z{p['index']:02d}_*.cells.bin"))
        if not slide or not cp.exists() or not cand:
            continue
        xy, _cw, _ch = read_cells_bin(cand[0])
        pc = pd.read_csv(cp, usecols=["tls_region_halo50", "in_tls_core"])
        if len(pc) != len(xy):
            continue
        own = pc["tls_region_halo50"].astype(str).to_numpy()
        core = pc["in_tls_core"].astype(str).isin(["True", "true", "1"]).to_numpy()
        for tid in set(own[core]):
            if not tid.startswith(("3d-tls", "2d-tls")):
                continue
            sel = core & (own == tid)
            tls_secs.setdefault(tid, set()).add(sid)
            if inside(sid, xy[sel, 0], xy[sel, 1]).any():
                tls_hit.setdefault(tid, set()).add(sid)
    rp = deliv / "tls" / "1_objects_report" / "1_objects_report.tsv"
    if rp.exists():
        df = pd.read_csv(rp, sep="\t")
        df["in_tumor"] = df["tls_object"].map(lambda t: bool(tls_hit.get(t)))
        df["tumor_sections"] = df["tls_object"].map(lambda t: len(tls_hit.get(t, ())))
        df["n_sections"] = df["tls_object"].map(lambda t: len(tls_secs.get(t, ())))
        df.to_csv(rp, sep="\t", index=False)
        print(f"  TLS: {int(df['in_tumor'].sum())} of {len(df)} rows in tumor", flush=True)


    nm = S14 / "nerve_members.npz"
    nerve_hit, nerve_secs = {}, {}
    if nm.exists():
        z = np.load(nm)
        raw = open(str(z["cloud"]), "rb").read()
        n = struct.unpack_from("<I", raw, 8)[0]
        arr = np.frombuffer(raw, np.uint16, 3 * n, 12).reshape(n, 3)
        sec_ids = [p["section_id"] for p in planes]
        for k in z.files:
            if not k.startswith("nerve-"):
                continue
            idx = z[k]
            hits, secs = set(), set()
            for ks in np.unique(arr[idx, 2]):
                sid = sec_ids[int(ks)]
                js = idx[arr[idx, 2] == ks]
                secs.add(sid)
                if inside(sid, arr[js, 0] / UM, arr[js, 1] / UM).any():
                    hits.add(sid)
            nerve_hit[k], nerve_secs[k] = hits, secs


        tm = json.loads((vol / a.regions).read_text())
        nkey = lambda k: int(k.split("-")[-1])
        per_rows = []
        for q in sorted(tm.get("sections") or tm["planes"], key=lambda q: q["z_um"]):
            sid = q["section_id"]
            tum = sorted((k for k in nerve_hit if sid in nerve_hit[k]), key=nkey)
            nor = sorted((k for k in nerve_secs if sid in nerve_secs[k] and sid not in nerve_hit[k]), key=nkey)
            ta, ti = q.get("area_mm2", 0.0), q.get("tissue_area_mm2")

            per_rows.append({"section_id": re.sub(r"-P\d+_", "_", sid), "z_um": q["z_um"],
                             "n_nerves_tumor": len(tum), "n_nerves_normal": len(nor),
                             "tumor_area_mm2": ta,
                             "normal_area_mm2": round(ti - ta, 3) if ti is not None else None,
                             "tissue_area_mm2": ti,
                             "tumor_nerves": ";".join(tum), "normal_nerves": ";".join(nor)})
        per_sec = pd.DataFrame(per_rows)
        pdir = deliv / "nerve" / "1_cords"
        if pdir.is_dir():
            per_sec.to_csv(pdir / "1_cords_per_section.tsv", sep="\t", index=False)
            print(f"  per section: {len(per_sec)} sections -> {pdir / '1_cords_per_section.tsv'}", flush=True)
        for jp in (S14 / "nerve_curves.json", S14 / "Schwann_nerves.json"):
            if jp.exists():
                r = json.loads(jp.read_text())
                for e in r["nerves"]:
                    e["in_tumor"] = bool(nerve_hit.get(e["name"]))
                    e["tumor_sections"] = len(nerve_hit.get(e["name"], ()))
                jp.write_text(json.dumps(r, indent=1))
        for tsv in (S14 / "nerve_curves.tsv", deliv / "nerve" / "1_cords" / "1_cords.tsv"):
            if tsv.exists():
                df = pd.read_csv(tsv, sep="\t")
                col = "id" if "id" in df.columns else "name"
                df["in_tumor"] = df[col].map(lambda t: bool(nerve_hit.get(t)))
                df["tumor_sections"] = df[col].map(lambda t: len(nerve_hit.get(t, ())))
                df.to_csv(tsv, sep="\t", index=False)
        print(f"  nerves: {sum(1 for v in nerve_hit.values() if v)} of {len(nerve_hit)} in tumor", flush=True)


if __name__ == "__main__":
    main()
