#!/usr/bin/env python3
import argparse, json, re
from pathlib import Path
import numpy as np, pandas as pd
from PIL import Image
from scipy.ndimage import label, distance_transform_edt

HALOS = [("inside", 0.0, 0.0), ("0-50um", 0.0, 50.0), ("50-100um", 50.0, 100.0)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--volume", required=True); ap.add_argument("--cells", required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--basis", default="G_withdrawn"); ap.add_argument("--type-col", default="HE_celltype", dest="type_col")
    ap.add_argument("--cords", default="", help="nerve cords table (delivery/nerve/1_cords.tsv): only its nerve ids are reported; default = <out dir>/1_cords.tsv when present")
    a = ap.parse_args()
    cords = Path(a.cords) if a.cords else Path(a.out).parent.parent / "1_cords" / "1_cords.tsv"
    keep_ids = None
    if cords.exists():
        keep_ids = {x.strip() for x in pd.read_csv(cords, sep="\t", usecols=lambda c: c.strip() == "name").iloc[:, 0]}
        print(f"  reporting the {len(keep_ids)} nerves of {cords}", flush=True)
    V, C = Path(a.volume), Path(a.cells)
    vm = json.loads((V / "metadata.json").read_text()); mpp = float(vm["in_plane_um_per_px"])
    nm = json.loads((V / "nerve_regions_metadata.json").read_text())
    rows = []
    for p in nm["planes"]:
        if not p.get("regions"):
            continue
        sid = p["section_id"]; tsv = C / f"{sid}.tsv"
        if not tsv.exists():
            print(f"  {sid}: no cell table, skipped", flush=True); continue
        cells = pd.read_csv(tsv, sep="\t", usecols=["x_um", "y_um", a.type_col]).dropna(subset=[a.type_col])
        im = np.asarray(Image.open(V / a.basis / "nerve" / p["file"]).convert("RGBA"))
        mask = im[..., 3] > 0
        lab, n = label(mask)

        comp_id = {}
        for r in p["regions"]:
            cy, cx = int(round(r["cy_um"] / mpp)), int(round(r["cx_um"] / mpp))
            if 0 <= cy < lab.shape[0] and 0 <= cx < lab.shape[1] and lab[cy, cx] > 0:
                comp_id.setdefault(lab[cy, cx], r["id"])
            else:
                x0, y0, x1, y1 = [int(round(v / mpp)) for v in r["bbox_um"]]
                sub = lab[max(0, y0):y1 + 1, max(0, x0):x1 + 1]
                ids = np.unique(sub[sub > 0])
                if len(ids): comp_id.setdefault(int(ids[0]), r["id"])
        cx_px = np.clip((cells["x_um"].to_numpy() / mpp).astype(int), 0, lab.shape[1] - 1)
        cy_px = np.clip((cells["y_um"].to_numpy() / mpp).astype(int), 0, lab.shape[0] - 1)
        types = cells[a.type_col].to_numpy()

        by_id = {}
        for comp, nid in comp_id.items():
            by_id.setdefault(nid, []).append(comp)
        if keep_ids is not None:
            by_id = {k: v for k, v in by_id.items() if k in keep_ids}
        for nid, comps in by_id.items():
            m = np.isin(lab, comps)
            d = distance_transform_edt(~m) * mpp
            dc = d[cy_px, cx_px]
            for name, lo, hi in HALOS:
                sel = (dc == 0) if name == "inside" else ((dc > lo) & (dc <= hi))
                if not sel.any():
                    continue
                vc = pd.Series(types[sel]).value_counts()
                tot = int(vc.sum())
                for t, n_ in vc.items():
                    rows.append({"nerve_id": nid, "section_id": re.sub(r"-P\d+_", "_", sid), "z_um": float(p["z_um"]), "n_regions": len(comps), "halo": name,
                                 "celltype": t, "n_cells": int(n_), "frac": round(float(n_) / tot, 4)})
        print(f"  {sid}: {len(comp_id)} nerve regions ({len(by_id)} nerves reported), {len(cells)} cells", flush=True)
    df = pd.DataFrame(rows)
    if df.empty:
        raise SystemExit("no rows: check the nerve layer and the cell tables")
    df = df.sort_values(["nerve_id", "z_um", "halo", "n_cells"], ascending=[True, True, True, False])
    out = Path(a.out); out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out.with_suffix(".tsv"), sep="\t", index=False)
    print(f"wrote {out.with_suffix('.tsv')} ({len(df)} rows, {df.nerve_id.nunique()} nerves)")


if __name__ == "__main__":
    main()
