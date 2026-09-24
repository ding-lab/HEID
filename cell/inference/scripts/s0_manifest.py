import os
PROJECTS_ROOT = os.environ.get("PROJECTS_ROOT", "/data/heid")
import os, re, sys, json
import numpy as np
import pandas as pd
import tifffile

INFERENCE_ROOT = PROJECTS_ROOT + "/cell/inference"
COHORT = os.environ.get("CELL_COHORT", "tcga_pancancer")
SRC = f"{INFERENCE_ROOT}/data/{os.environ.get('CELL_SELECTED', 'cohort_selected')}.csv"
OUT_DIR = f"{INFERENCE_ROOT}/configs/{COHORT}"


def geometry(path):
    with tifffile.TiffFile(path) as tf:
        page = tf.pages[0]
        w, h = int(page.imagewidth), int(page.imagelength)
        desc = page.description or ""
    mpp = re.search(r"MPP\s*=\s*([\d.]+)", desc)
    mag = re.search(r"AppMag\s*=\s*([\d.]+)", desc)
    return w, h, (float(mpp.group(1)) if mpp else np.nan), \
        (float(mag.group(1)) if mag else np.nan)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Read slide headers and write the run manifest of a new-slide cohort.")
    parser.add_argument("--source", default=SRC, help="selection CSV; default from CELL_SELECTED")
    parser.add_argument("--output-dir", default=OUT_DIR, help="default from CELL_COHORT")
    args = parser.parse_args()
    SRC, OUT_DIR = args.source, args.output_dir
    os.makedirs(OUT_DIR, exist_ok=True)
    d = pd.read_csv(SRC)
    print(f"cohort table {len(d):,d} rows", flush=True)

    rows, skipped = [], []
    for i, r in enumerate(d.itertuples(), 1):
        if not os.path.exists(r.path):
            skipped.append({"file": r.file, "reason": "file does not exist"})
            continue
        try:
            w, h, mpp, mag = geometry(r.path)
        except Exception as e:
            skipped.append({"file": r.file, "reason": f"failed to read header {type(e).__name__}"})
            continue
        if not np.isfinite(mpp):

            skipped.append({"file": r.file, "reason": "no MPP in file header"})
            continue
        uuid = os.path.basename(os.path.dirname(r.path))
        rows.append({"slide": f"{r.patient}-{uuid[:8]}", "svs": r.path,
                     "native_mpp": mpp, "appmag": mag, "level0_W": w, "level0_H": h,
                     "patient": r.patient, "project_id": r.project_id, "pni": r.pni,
                     "flag_20x": bool(r.flag_20x), "flag_mpp_far": bool(r.flag_mpp_far),
                     "flag_small": bool(r.flag_small)})
        if i % 500 == 0:
            print(f"  {i}/{len(d)}", flush=True)

    m = pd.DataFrame(rows)


    ORDER = ["TCGA-COAD", "TCGA-PAAD", "TCGA-PRAD", "TCGA-BRCA", "TCGA-LUAD",
             "TCGA-HNSC", "TCGA-KIRC", "TCGA-SKCM", "TCGA-CHOL"]
    rank = {p: i for i, p in enumerate(ORDER)}
    m["_rank"] = m.project_id.map(rank).fillna(len(ORDER))
    m["_px"] = m.level0_W.astype(float) * m.level0_H.astype(float)
    m = m.sort_values(["_rank", "_px", "slide"]).drop(columns=["_rank", "_px"]).reset_index(drop=True)
    if m.slide.duplicated().any():
        dup = m.loc[m.slide.duplicated(keep=False), "slide"].unique()[:5]
        sys.exit(f"FATAL: slide id not unique, e.g. {list(dup)}")
    m.to_csv(f"{OUT_DIR}/run_manifest.tsv", sep="\t", index=False)
    json.dump(m.slide.tolist(), open(f"{OUT_DIR}/slides.json", "w"))
    if skipped:
        pd.DataFrame(skipped).to_csv(f"{OUT_DIR}/skipped.csv", index=False)

    print(f"\nmanifest {len(m):,d} slides (skipped {len(skipped)})", flush=True)
    print(f"resolution: median {m.native_mpp.median():.4f}  "
          f"range {m.native_mpp.min():.4f}-{m.native_mpp.max():.4f}", flush=True)
    print(f"magnification: {m.appmag.value_counts(dropna=False).to_dict()}", flush=True)
    print(f"cancer types: {m.project_id.value_counts().to_dict()}", flush=True)
    print("\nrun order (manifest order is run order):", flush=True)
    seen_rank = m.groupby("project_id").size()
    for i, proj in enumerate(ORDER, 1):
        n = int(seen_rank.get(proj, 0))
        print(f"  {i}. {proj:12s} {n:5,d} slides" + ("   (this cancer type not in this cohort)" if n == 0 else ""), flush=True)
    rest = sorted(set(m.project_id) - set(ORDER))
    if rest:
        print(f"  unlisted, placed last: {', '.join(f'{r}({int(seen_rank[r]):,d})' for r in rest)}",
              flush=True)
    print(f"with perineural invasion annotation: {int(m.pni.notna().sum()):,d}", flush=True)
    px = (m.level0_W.astype(float) * m.level0_H.astype(float)).sum()
    print(f"level-0 total {px/1e12:.2f} trillion pixels", flush=True)
    print(f"\n-> {OUT_DIR}/run_manifest.tsv", flush=True)
