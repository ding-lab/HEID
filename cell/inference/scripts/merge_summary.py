#!/usr/bin/env python -u
import os
PROJECTS_ROOT = os.environ.get("PROJECTS_ROOT", "/data/heid")
import ast
import collections
import sys

import pandas as pd

INFERENCE_ROOT = PROJECTS_ROOT + "/cell/inference"
COHORT = os.environ.get("CELL_COHORT", "tcga_pancancer")
OUT = os.path.join(INFERENCE_ROOT, COHORT, "outputs")


def main():
    manifest = pd.read_csv(os.path.join(INFERENCE_ROOT, "configs", COHORT, "run_manifest.tsv"), sep="\t")
    want = set(manifest["slide"])


    import glob as _glob
    shards = sorted(_glob.glob(os.path.join(OUT, "*", "*_summary_shard*.csv")))
    if not shards:
        sys.exit(f"FATAL: no summary shards found under {OUT}/*/")
    df = pd.concat([pd.read_csv(f) for f in shards], ignore_index=True)
    df = df.drop_duplicates("slide").sort_values("slide").reset_index(drop=True)

    missing = want - set(df["slide"])
    print(f"{len(shards)} shards | merged {len(df)}/{len(want)} slides")
    if missing:
        print(f"missing {len(missing)} slides: {sorted(missing)[:5]}{' ...' if len(missing) > 5 else ''}")
        sys.exit("FATAL: cohort incomplete, refusing to write a summary table that would be read as the full cohort")


    proj = manifest.set_index("slide")["project_id"]
    df["project_id"] = df["slide"].map(proj)
    for name, part in df.groupby("project_id"):
        dest = os.path.join(OUT, name, f"{name}_summary.csv")
        part.drop(columns=["project_id"]).to_csv(dest, index=False)
        print(f"  {name}: {len(part):5d} slides -> {dest}")
    dest = os.path.join(OUT, f"{COHORT}_summary.csv")
    df.to_csv(dest, index=False)
    print(f"full cohort -> {dest}")

    status = collections.Counter()
    for raw in df["status_counts"]:
        for key, value in ast.literal_eval(raw).items():
            status[key] += value
    n_reg = int(df["n_regions"].sum())
    print("\n--- cohort readout ---")
    print(f"total cells          {df['n_cells'].sum():,}")
    print(f"Schwann cells        {df['n_schwann_cells'].sum():,} "
          f"({100 * df['n_schwann_cells'].sum() / df['n_cells'].sum():.3f}%)")
    print(f"nerve regions        {n_reg:,}, total {df['region_area_mm2'].sum():.1f} mm2,"
          f"median per slide {df['n_regions'].median():.0f}")
    print(f"tumor cells          {df['n_tumor_cells'].sum():,} "
          f"({100 * df['n_tumor_cells'].sum() / df['n_cells'].sum():.1f}%)")
    print(f"tumor fraction of tissue  median {df['tumor_fraction_of_tissue'].median():.3f} "
          f"(IQR {df['tumor_fraction_of_tissue'].quantile(.25):.3f}-"
          f"{df['tumor_fraction_of_tissue'].quantile(.75):.3f})")
    print(f"tissue area          total {df['tissue_area_mm2'].sum():.0f} mm2")
    print("nerve region-to-tumor spatial relationship (4 categories):")
    for key in ("in tumor core", "in 50um halo zone", "100um halo", "-tumor"):
        print(f"    {key:20s} {status[key]:6,}  ({100 * status[key] / max(n_reg, 1):5.1f}%)")
    zero = int((df["n_regions"] == 0).sum())
    print(f"slides with zero nerve regions {zero}/{len(df)}")


if __name__ == "__main__":
    main()
