import paths as P
import os
PROJECTS_ROOT = os.environ.get("PROJECTS_ROOT", "/data/heid")
import pandas as pd, numpy as np, glob, os, json

ROOT = str(P.WORKSPACE)
X1000 = str(P.PROJECTS)
TLS_OUTPUT_5K = str(P.ACCEPTED / "outputs/5k")
TUMOR_SIDE_5K = str(P.SIDE_ROOT / "5k/outputs")
OUT = str(P.ACCEPTED / "outputs/_tls_composition.csv")


BND = {os.path.basename(p).replace("_boundary_distance.csv", ""): p
       for p in glob.glob(f"{TUMOR_SIDE_5K}/*/*/*_boundary_distance.csv")}


def per_sample(scores_csv):
    tissue = scores_csv.split("/5k/")[1].split("/")[0]
    sample = os.path.basename(scores_csv).replace("_scores.csv", "")

    meta_p = os.path.join(os.path.dirname(scores_csv), f"{sample}_tls_meta.json")
    id2area = ({int(t["id"]): round(float(t.get("area_mm2", float("nan"))), 4)
                for t in json.load(open(meta_p)).get("tls", [])}
               if os.path.exists(meta_p) else {})
    d = pd.read_csv(scores_csv, usecols=["cell_id", "celltype", "region", "tls_region_number", "TLS_State"],
                    dtype={"cell_id": str, "TLS_State": str}, keep_default_na=False)
    d = d[d["region"] == "TLS"].copy()
    if d.empty:
        return []
    d["tls_region_number"] = pd.to_numeric(d["tls_region_number"], errors="coerce").astype("Int64")

    if sample in BND:
        b = pd.read_csv(BND[sample], usecols=["cell_id", "side"], dtype=str).drop_duplicates("cell_id")
        d["_side"] = d["cell_id"].str.split("_", n=1).str[0].map(b.set_index("cell_id")["side"])
        has_bnd = True
    else:
        d["_side"] = None
        has_bnd = False
    rows = []
    for rn, g in d.groupby("tls_region_number"):
        n = len(g)
        if has_bnd:
            n_tum = int((g["_side"] == "Tumor").sum())
            n_norm = int((g["_side"] == "Normal").sum())

            loc = ("on_boundary" if (n_tum > 0 and n_norm > 0) else
                   "inside_tumor" if n_tum > 0 else "outside_tumor")
            tf_out = round(n_tum / n, 3)
        else:
            loc, tf_out = "Lymph_node", ""
        state = g["TLS_State"].iloc[0] if "TLS_State" in g.columns and len(g) else ""
        row = {"sample": sample, "tissue": tissue, "tls_region": f"TLS-{int(rn)}",
               "TLS_State": state,
               "n_cells": n, "region_area_mm2": id2area.get(int(rn), ""),
               "location": loc, "tumor_frac": tf_out}
        comp = (g["celltype"].value_counts() / n)
        for ct, fr in comp.items():
            row[f"frac_{ct}"] = round(float(fr), 4)
        rows.append(row)

    sdf = pd.DataFrame(rows)
    idc = ["sample", "tissue", "tls_region", "TLS_State", "n_cells", "region_area_mm2", "location", "tumor_frac"]
    fr = sorted([c for c in sdf.columns if c.startswith("frac_")],
                key=lambda c: sdf[c].fillna(0).sum(), reverse=True)
    sdf = sdf[idc + fr]
    for c in fr:
        sdf[c] = pd.to_numeric(sdf[c], errors="coerce").fillna(0.0).round(4)
    sdf.to_csv(f"{os.path.dirname(scores_csv)}/{sample}_tls_composition.csv", index=False)
    return rows


def main():
    allrows = []
    for f in sorted(glob.glob(f"{TLS_OUTPUT_5K}/*/*/*_scores.csv")):
        allrows.extend(per_sample(f))
    df = pd.DataFrame(allrows)

    idc = ["sample", "tissue", "tls_region", "TLS_State", "n_cells", "region_area_mm2", "location", "tumor_frac"]
    fracs = [c for c in df.columns if c.startswith("frac_")]
    fracs = sorted(fracs, key=lambda c: df[c].fillna(0).sum(), reverse=True)
    df = df[idc + fracs].fillna(0.0 if False else "")
    for c in fracs:
        df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0.0).round(4)
    df.to_csv(OUT, index=False)
    print(f"wrote {OUT}: {len(df)} TLS rows, {len(fracs)} cell-type frac columns")
    print("location counts:", df["location"].value_counts().to_dict())


if __name__ == "__main__":
    main()
