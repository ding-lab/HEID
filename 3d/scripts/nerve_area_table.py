#!/usr/bin/env python3
import argparse
import importlib.util
import json
import os
import pickle
import re
import struct
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent


def _load(name, p):
    spec = importlib.util.spec_from_file_location(name, p)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


BNC = _load("bnc", HERE / "build_nerve_curves3d.py")
RES = BNC.RES


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--nerve-dir", required=True, dest="nerve_dir")
    ap.add_argument("--volume", required=True)
    ap.add_argument("--sample", required=True)
    ap.add_argument("--out-stem", required=True, dest="out_stem")
    a = ap.parse_args()
    S14, VOL = Path(a.nerve_dir), Path(a.volume)


    region_file=VOL/'nerve_regions_metadata.json'
    region_data=json.loads(region_file.read_text()) if region_file.exists() else {}
    if region_data.get('definition')=='cell_supported_2d_v1':
        from shapely.geometry import Polygon
        from shapely.ops import unary_union
        rows=[]
        for plane in region_data['planes']:
            for r in plane['regions']:
                holes=r.get('polygon_holes_um',[[] for _ in r['polygons_um']])
                g=unary_union([Polygon(poly,holes[i]) for i,poly in enumerate(r['polygons_um'])])
                assert g.is_valid and abs(g.area-r['area_um2'])<.01
                rows.append(dict(name=r['id'],section_id=plane['section_id'],z_um=plane['z_um'],
                    n_cells=r['n_support_cells'],n_seed_cells=r['n_seed_cells'],area_um2=g.area,
                    definition=region_data['definition']))
        long=pd.DataFrame(rows);wide=[]
        for nid,group in long.groupby('name',sort=True):
            peak=group.loc[group.area_um2.idxmax()]
            row=dict(name=nid,n_sections=len(group),area_max_um2=peak.area_um2,area_max_z_um=peak.z_um,
                area_mean_um2=group.area_um2.mean(),area_sum_um2=group.area_um2.sum())
            for r in group.itertuples():row[re.sub(r'^P\d+_','',r.section_id.replace(a.sample+'-','').replace(a.sample+'_',''))]=r.area_um2
            wide.append(row)
        out=Path(a.out_stem);out.parent.mkdir(parents=True,exist_ok=True)
        pd.DataFrame(wide).to_csv(out.with_suffix('.tsv'),sep='\t',index=False)
        long.to_csv(out.with_name(out.name+'_per_section.tsv'),sep='\t',index=False)
        print(f'wrote exact revised 2D polygon areas: {len(wide)} nerves, {len(long)} profiles; retained 3D geometry is not remeasured')
        return

    caches = sorted((VOL / "nerve_operator_cache").glob("stage2_*.pkl"), key=os.path.getmtime)
    if not caches:
        raise SystemExit("no stage cache: the nerve chain has not run on this volume")
    st2 = pickle.load(open(caches[-1], "rb"))
    data, cg = st2["data"], st2["cg"]
    order = sorted(data, key=lambda s: data[s][0])
    zs = [float(data[s][0]) for s in order]
    ny, nx = next(iter(cg.values()))[0].shape


    mem = np.load(S14 / "nerve_members.npz")
    raw = Path(str(mem["cloud"])).read_bytes()
    n_pts = struct.unpack_from("<I", raw, 8)[0]
    arr = np.frombuffer(raw, np.uint16, 3 * n_pts, 12).reshape(n_pts, 3)
    P = arr[:, :2].astype(np.float64)
    SEC = arr[:, 2].astype(np.int32)
    rep = {e["name"]: e for e in json.loads((S14 / "Schwann_nerves.json").read_text())["nerves"]}
    names = sorted((k for k in mem.files if k.startswith("nerve-")), key=lambda k: int(k.split("-")[-1]))


    short = lambda sid: re.sub(r"^P\d+_", "", sid.replace(a.sample + "-", "").replace(a.sample + "_", ""))
    long_rows, wide_rows, bad, unchecked = [], [], [], []
    for nid in names:
        e = rep.get(nid, {})
        poly = np.array(e.get("centreline_um") or np.zeros((0, 3)), np.float64)
        idx = mem[nid]
        by_sec = {}
        for j in idx:
            by_sec.setdefault(int(SEC[j]), []).append(int(j))
        secs = sorted(by_sec)
        masks, areas = {}, {}
        for ks in secs:
            js = by_sec[ks]
            m = BNC.region_mask(P[js, 0], P[js, 1], poly, zs[ks], ny, nx)
            masks[(nid, order[ks])] = m
            area = int(m.sum()) * RES * RES
            areas[ks] = area
            long_rows.append({"name": nid, "section_id": order[ks], "z_um": zs[ks],
                              "n_cells": len(js), "area_um2": area})

        th = BNC._thickest(masks, nid, secs, order, zs)
        if e.get("max_diameter_um") is None:
            unchecked.append(nid)
        elif th["max_diameter_um"] != e["max_diameter_um"] or th["max_diameter_z_um"] != e.get("max_diameter_z_um"):
            bad.append(f"{nid}: rebuilt {th['max_diameter_um']} um @ z {th['max_diameter_z_um']} vs report "
                       f"{e.get('max_diameter_um')} um @ z {e.get('max_diameter_z_um')}")
        kmax = max(secs, key=lambda k: areas[k])
        row = {"name": nid, "n_sections": len(secs), "area_max_um2": areas[kmax], "area_max_z_um": zs[kmax],
               "area_mean_um2": round(float(np.mean([areas[k] for k in secs])), 1),
               "area_sum_um2": int(sum(areas.values()))}
        for ks in secs:
            row[short(order[ks])] = areas[ks]
        wide_rows.append(row)

    fixed = ["name", "n_sections", "area_max_um2", "area_max_z_um", "area_mean_um2", "area_sum_um2"]
    cols = fixed + [short(s) for s in order if any(short(s) in r for r in wide_rows)]
    wide = pd.DataFrame(wide_rows, columns=cols)
    out = Path(a.out_stem)
    out.parent.mkdir(parents=True, exist_ok=True)
    wide.to_csv(out.with_suffix(".tsv"), sep="\t", index=False)
    print(f"wrote {out.with_suffix('.tsv')}: {len(wide)} nerves on {len(cols) - len(fixed)} sections "
          f"({len(long_rows)} nerve-section cells; area = region mask on the {RES:g} um grid, "
          f"{BNC.MASK_R_UM:g} um margin around the member cells)", flush=True)
    print(f"self-check: thickest diameter reproduced for {len(wide) - len(bad) - len(unchecked)} of "
          f"{len(wide) - len(unchecked)} nerves with a max_diameter_um in the report"
          + (f"; {len(unchecked)} nerves have none (older curve run), not checked" if unchecked else ""), flush=True)
    if bad:
        for b in bad[:10]:
            print("  MISMATCH " + b)
        raise SystemExit(2)


if __name__ == "__main__":
    main()
