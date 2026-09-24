import os
PROJECTS_ROOT = os.environ.get("PROJECTS_ROOT", "/data/heid")
import glob, json, os, sys
import numpy as np
import pandas as pd

INFERENCE_ROOT = PROJECTS_ROOT + "/cell/inference"
COHORT = os.environ.get("CELL_COHORT", "tcga_pancancer")
D = f"{INFERENCE_ROOT}/{COHORT}/data"
MAN = f"{INFERENCE_ROOT}/configs/{COHORT}/run_manifest.tsv"

DENS_LO, DENS_HI = 1000, 15000
POLY_MIN = 0.80


def audit():
    if not os.path.exists(MAN):
        return ["manifest not yet generated"], {}
    man = pd.read_csv(MAN, sep="\t").set_index("slide")
    metas = sorted(glob.glob(f"{D}/wmaps/*.meta.json"))
    if not metas:
        return ["no outputs yet"], {}
    rows, problems = [], []
    for m in metas:
        j = json.load(open(m))
        s = j["slide"]
        row = {"slide": s, "n_cells": j["n_cells"], "tissue_mm2": round(j["tissue_mm2"], 1),
               "cells_per_mm2": round(j["cells_per_mm2"], 0),
               "poly_frac": round(j["valid_frac"], 3), "flags": ",".join(j.get("flags", [])),
               "sec_detect": j.get("seconds")}
        if s in man.index:
            row["cancer"] = man.loc[s, "project_id"]
            row["appmag"] = man.loc[s, "appmag"]

        p = f"{D}/predictions/{s}.parquet"
        if os.path.exists(p):
            d = pd.read_parquet(p, columns=["scored", "pred_class_name", "schwann_prob"])
            d = d[d.scored]
            if len(d):
                vc = d.pred_class_name.value_counts(normalize=True)
                row["n_scored"] = len(d)
                row["top_class"] = vc.index[0]
                row["top_frac"] = round(float(vc.iloc[0]), 3)
                row["schwann_p999"] = round(float(np.quantile(d.schwann_prob, 0.999)), 4)
                row["frac_tumor"] = round(float(vc.get("Tumor", 0.0)), 3)
                if row["top_frac"] > 0.90:
                    problems.append(f"{s}: {row['top_class']} accounts for {row['top_frac']:.0%}, classification degenerate")

        g = f"{D}/tissue_grid/{s}.npz"
        if os.path.exists(g):
            with np.load(g) as h:
                if "scales" in h:
                    w = sum(h[f"emb_{sc}"].shape[1] for sc in h["scales"].tolist())
                    row["grid_dim"] = w
                    if w != 3072:
                        problems.append(f"{s}: wide-field feature reconstruction width {w} != 3072")
        rows.append(row)
        if j["n_cells"] == 0:
            problems.append(f"{s}: zero cells")
        elif j["tissue_mm2"] >= 1.0:
            dens = j["cells_per_mm2"]
            if dens < DENS_LO or dens > DENS_HI:
                problems.append(f"{s}: cell density {dens:.0f}/mm2 outside {DENS_LO}-{DENS_HI}")
        if j["n_cells"] and j["valid_frac"] < POLY_MIN:
            problems.append(f"{s}: polygon validity rate {j['valid_frac']:.1%} < {POLY_MIN:.0%}")
    return problems, pd.DataFrame(rows)


if __name__ == "__main__":
    problems, t = audit()
    if isinstance(t, dict) or not len(t):
        print("\n".join(problems) if problems else "no outputs")
        sys.exit(0)
    print(f"=== {len(t)} slides produced ===")
    cols = [c for c in ("slide", "cancer", "appmag", "n_cells", "cells_per_mm2", "poly_frac",
                        "n_scored", "top_class", "top_frac", "frac_tumor", "schwann_p999",
                        "grid_dim", "sec_detect", "flags") if c in t.columns]
    print(t[cols].to_string(index=False))
    if "appmag" in t and "cells_per_mm2" in t and t.appmag.nunique() > 1:
        print(f"\nmedian cell density by magnification:\n{t.groupby('appmag').cells_per_mm2.median().to_string()}")
        if "frac_tumor" in t:
            print(f"median tumor cell fraction by magnification:\n{t.groupby('appmag').frac_tumor.median().to_string()}")
    print(f"\n=== {len(problems)} problems ===")
    for p in problems[:25]:
        print(f"  {p}")
    os.makedirs(f"{INFERENCE_ROOT}/outputs", exist_ok=True)
    t.to_csv(f"{INFERENCE_ROOT}/outputs/first_slides_audit.csv", index=False)
    if problems:
        sys.exit(1)
