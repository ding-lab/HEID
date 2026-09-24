import os
PROJECTS_ROOT = os.environ.get("PROJECTS_ROOT", "/data/heid")
import argparse
import glob
import sys

import pandas as pd

INFERENCE_ROOT = PROJECTS_ROOT + "/cell/inference"
COHORT = os.environ.get("CELL_COHORT", "tcga_pancancer")
MANIFEST = f"{INFERENCE_ROOT}/configs/{COHORT}/run_manifest.tsv"


def gather(project: str) -> pd.DataFrame:
    patterns = [f"{INFERENCE_ROOT}/{project}/summary/*_summary_shard*.csv",
                f"{INFERENCE_ROOT}/{COHORT}/summary/*_summary_shard*.csv"]
    frames = []
    for pat in patterns:
        for path in glob.glob(pat):
            try:
                frames.append(pd.read_csv(path))
            except Exception as exc:
                print(f"  [warn] cannot read {os.path.basename(path)}: {exc}")
    if not frames:
        return pd.DataFrame()
    table = pd.concat(frames, ignore_index=True).drop_duplicates("slide")
    manifest = pd.read_csv(MANIFEST, sep="\t", usecols=["slide", "project_id", "patient"])
    table = table.merge(manifest, on="slide", how="left")
    return table[table.project_id == project].sort_values("slide").reset_index(drop=True)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--project", help="process only this one cancer type; default is all cancer types with existing outputs")
    args = ap.parse_args()

    manifest = pd.read_csv(MANIFEST, sep="\t")
    projects = [args.project] if args.project else sorted(manifest.project_id.unique())

    overview = []
    for project in projects:
        table = gather(project)
        if not len(table):
            continue
        expected = int((manifest.project_id == project).sum())
        cols = ["slide", "patient", "n_cells", "tissue_area_mm2", "n_tumor_cells",
                "tumor_area_mm2", "tumor_fraction_of_tissue", "n_schwann_cells",
                "n_regions", "region_area_mm2", "has_schwann"]
        cols = [c for c in cols if c in table.columns]
        out = table[cols].copy()


        total = {c: (out[c].sum() if pd.api.types.is_numeric_dtype(out[c]) else "")
                 for c in out.columns}
        total["slide"] = f"__TOTAL__ ({len(out)}/{expected} slides)"
        total["patient"] = ""
        if "tissue_area_mm2" in out and out.tissue_area_mm2.sum() > 0:
            total["tumor_fraction_of_tissue"] = round(
                out.tumor_area_mm2.sum() / out.tissue_area_mm2.sum(), 4)
        if "has_schwann" in out:
            total["has_schwann"] = int(out.has_schwann.sum()) if out.has_schwann.dtype != object else ""
        out = pd.concat([out, pd.DataFrame([total])], ignore_index=True)

        dest_dir = f"{INFERENCE_ROOT}/{project}/summary"
        os.makedirs(dest_dir, exist_ok=True)
        dest = f"{dest_dir}/{project}_cohort_table.csv"
        out.to_csv(dest, index=False)

        frac = out.tumor_area_mm2.iloc[:-1].sum() / max(out.tissue_area_mm2.iloc[:-1].sum(), 1e-9)
        print(f"{project:12s} {len(out)-1:4d}/{expected:4d} slides   "
              f"tissue {out.tissue_area_mm2.iloc[:-1].sum():9,.0f} mm²   "
              f"tumor {out.tumor_area_mm2.iloc[:-1].sum():9,.0f} mm² ({frac:5.1%})   -> {dest}")
        overview.append({"project_id": project, "n_slides_done": len(out) - 1,
                         "n_slides_expected": expected,
                         "tissue_area_mm2": round(out.tissue_area_mm2.iloc[:-1].sum(), 1),
                         "tumor_area_mm2": round(out.tumor_area_mm2.iloc[:-1].sum(), 1),
                         "tumor_fraction": round(frac, 4),
                         "n_schwann_regions": int(out.n_regions.iloc[:-1].sum())
                                              if "n_regions" in out else 0})

    if not overview:
        sys.exit("no cancer type has an available shard summary")
    across = pd.DataFrame(overview)
    dest = f"{INFERENCE_ROOT}/cohort_overview.csv"
    across.to_csv(dest, index=False)
    print(f"\ncross-cancer-type overview -> {dest}")
    print(across.to_string(index=False))


if __name__ == "__main__":
    main()
