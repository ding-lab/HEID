import paths as P
import os
import pandas as pd, numpy as np, glob, os, sys
from scipy.ndimage import distance_transform_edt, binary_closing, binary_fill_holes
from skimage.morphology import disk

ROOT = str(P.WORKSPACE)
SCORE = str(P.PROGRAMS)
H5AD = str(P.CELLS)
TLS_OUTPUT_5K = str(P.ACCEPTED / "outputs/5k")
TUMOR_SIDE_5K = str(P.SIDE_ROOT / "5k/outputs")
TUMOR_FRAC_MIN = 0.5
DIST_RES_UM = 10
DIST_MARGIN_UM = 100
SCORE_COLS = ["score_B", "score_T", "score_DC"]
TLS_DIR = {}
for p in glob.glob(f"{TLS_OUTPUT_5K}/*/*/*_tls.csv"):
    samp = os.path.basename(p).replace("_tls.csv", "")
    TLS_DIR[samp] = (os.path.dirname(p), p.split("/5k/")[1].split("/")[0])
BND_DIR = {os.path.basename(p).replace("_boundary_distance.csv", ""): p
           for p in glob.glob(f"{TUMOR_SIDE_5K}/*/*/*_boundary_distance.csv")}


def assemble(sample):
    tls_dir, tissue = TLS_DIR[sample]
    sc = pd.read_parquet(f"{SCORE}/{sample}.parquet"); sc["cell_id"] = sc.cell_id.astype(str)
    if set(sc["scoring_method"].astype(str)) != {"ulm_pertype"}:
        raise ValueError(f"{sample}: program scores were not computed with ulm_pertype")
    t = pd.read_csv(f"{tls_dir}/{sample}_tls.csv",
                    usecols=["cell_id", "cell_type", "tls_region", "in_tls_core", "TLS_State"],
                    dtype={"cell_id": str, "TLS_State": str}, low_memory=False, keep_default_na=False)
    d = t.merge(sc[["cell_id"] + SCORE_COLS], on="cell_id", how="left")
    is_tls = d.tls_region.astype(str).str.startswith("TLS-")
    out = pd.DataFrame({"cell_id": d.cell_id, "celltype": d.cell_type})
    for c in SCORE_COLS:
        out[c] = d[c].astype(float)
    out["region"] = np.where(is_tls, "TLS", "Outside")
    out["tls_region_number"] = pd.to_numeric(
        d.tls_region.astype(str).str.extract(r"TLS-(\d+)")[0], errors="coerce").astype("Int64")

    out["TLS_State"] = np.where(is_tls, d["TLS_State"].astype(str).values, "")


    out["tls_class"] = ""
    if sample in BND_DIR:
        bnd = pd.read_csv(BND_DIR[sample], usecols=["cell_id", "side"], dtype=str)
        side = out["cell_id"].str.split("_", n=1).str[0].map(
            bnd.drop_duplicates("cell_id").set_index("cell_id")["side"])
        tn = out["tls_region_number"]
        frac = pd.Series((side == "Tumor").values, index=out.index)[tn.notna()].groupby(
            tn[tn.notna()]).mean()
        cls = frac.apply(lambda f: "Tumor" if f >= TUMOR_FRAC_MIN else "Normal")
        out.loc[tn.notna(), "tls_class"] = tn[tn.notna()].map(cls).values
    else:
        out.loc[out["region"] == "TLS", "tls_class"] = "Lymph_node"


    out["dist_to_tls_um"] = ""
    out["dist_to_tls_core_um"] = ""
    co = pd.read_parquet(f"{H5AD}/{sample}.parquet", columns=["cell_id", "x_centroid", "y_centroid"])
    co["cell_id"] = co.cell_id.astype(str)
    m = out.merge(co.drop_duplicates("cell_id"), on="cell_id", how="left")
    is_tumor = (m["celltype"].astype(str) == "Tumor").values
    is_region = (m["region"] == "TLS").values
    is_core = (d["in_tls_core"].astype(str).isin(["True", "true", "1"]).values
               if "in_tls_core" in d.columns else np.zeros(len(m), dtype=bool))
    if is_tumor.any() and m["x_centroid"].notna().any():
        x = m["x_centroid"].values; y = m["y_centroid"].values
        x0 = np.nanmin(x) - DIST_MARGIN_UM; y0 = np.nanmin(y) - DIST_MARGIN_UM
        nx = int(np.ceil((np.nanmax(x) + DIST_MARGIN_UM - x0) / DIST_RES_UM))
        ny = int(np.ceil((np.nanmax(y) + DIST_MARGIN_UM - y0) / DIST_RES_UM))
        ix = np.clip(((x - x0) / DIST_RES_UM).astype(int), 0, nx - 1)
        iy = np.clip(((y - y0) / DIST_RES_UM).astype(int), 0, ny - 1)
        ti = np.where(is_tumor)[0]

        def _dist(sel):
            if not sel.any():
                return None
            mk = np.zeros((ny, nx), dtype=bool)
            mk[iy[sel], ix[sel]] = True
            mk = binary_fill_holes(binary_closing(mk, structure=disk(3)))
            return np.round((distance_transform_edt(~mk) - distance_transform_edt(mk)) * DIST_RES_UM, 1)

        for colname, sel in (("dist_to_tls_um", is_region), ("dist_to_tls_core_um", is_core)):
            dmap = _dist(sel)
            if dmap is not None:
                vals = dmap[iy, ix]
                dcol = np.array([""] * len(out), dtype=object)
                dcol[ti] = vals[ti]
                out[colname] = dcol
    odir = tls_dir
    os.makedirs(odir, exist_ok=True)
    out.to_csv(f"{odir}/{sample}_scores.csv", index=False)
    return tissue, len(out), int(is_tls.sum())


def main():
    samples = [sys.argv[1]] if len(sys.argv) > 1 else sorted(TLS_DIR)
    done = miss = 0
    for s in samples:
        if not os.path.exists(f"{SCORE}/{s}.parquet"):
            print(f"SKIP {s}: no score sidecar"); miss += 1; continue
        tissue, n, ntls = assemble(s)
        print(f"{tissue}/{s}: {n} cells ({ntls} TLS) -> _scores.csv")
        done += 1
    print(f"\nassembled {done} samples ({miss} missing)")


if __name__ == "__main__":
    main()
