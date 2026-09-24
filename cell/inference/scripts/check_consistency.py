import os
PROJECTS_ROOT = os.environ.get("PROJECTS_ROOT", "/data/heid")
import argparse, glob, json, os, re, sys
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from out_paths import slide_dir, all_projects

INFERENCE_ROOT = PROJECTS_ROOT + "/cell/inference"
COHORT = os.environ.get("CELL_COHORT", "tcga_pancancer")
OUT = f"{INFERENCE_ROOT}/{COHORT}/outputs"
DATA = f"{INFERENCE_ROOT}/{COHORT}/data"


def check_slide(slide):
    d = str(slide_dir(slide))
    problems = []
    if not os.path.isdir(d):
        return ["output directory does not exist"]

    stats_p = f"{d}/{slide}_cluster_stats.csv"
    cells_p = f"{d}/{slide}_cells.csv"
    roi_p = f"{d}/roi_{slide}_index.csv"
    for p in (stats_p, cells_p):
        if not os.path.exists(p):
            problems.append(f"missing {os.path.basename(p)}")
    if problems:
        return problems

    stats = pd.read_csv(stats_p)
    cells = pd.read_csv(cells_p, low_memory=False)
    region_ids = set(stats.cluster_id.astype(int)) if len(stats) else set()


    assigned = cells.schwann_cluster.dropna()
    cell_ids = set(assigned.astype(int)) if len(assigned) else set()
    if cell_ids - region_ids:
        problems.append(f"cells table has region numbers not in the region table {sorted(cell_ids - region_ids)[:5]}")
    if region_ids - cell_ids:
        problems.append(f"region table has region numbers with no cells {sorted(region_ids - cell_ids)[:5]}")
    for rid, n_expected in stats.set_index("cluster_id").n_schwann.items():
        n_actual = int((assigned.astype(float) == float(rid)).sum())


        if n_actual < n_expected:
            problems.append(f"region {rid}: table says {n_expected} Schwann cells, cells table only has {n_actual} members")


    whole = f"{d}/{slide}.png"
    if not os.path.exists(whole):
        problems.append("missing whole-slide image")
    rois = sorted(glob.glob(f"{d}/roi_{slide}_r*.png"))
    sheets = sorted(glob.glob(f"{d}/sheet_{slide}*.png"))
    if not region_ids:
        if rois or sheets:
            problems.append(f"zero regions but {len(rois)} ROI and {len(sheets)} contact sheet images remain (stale outputs)")
    else:
        if not sheets:
            problems.append("has regions but no contact sheet")
        if os.path.exists(roi_p):
            roi = pd.read_csv(roi_p)
            unknown = set(roi.region_id.astype(int)) - region_ids
            if unknown:
                problems.append(f"ROI index points to nonexistent regions {sorted(unknown)[:5]}")
            if len(roi) != len(rois):
                problems.append(f"ROI index has {len(roi)} rows, actual images {len(rois)}")
        elif rois:
            problems.append(f"has {len(rois)} ROI images but no index table")


    pred_p = f"{DATA}/predictions/{slide}.parquet"
    if os.path.exists(pred_p):
        pred = pd.read_parquet(pred_p, columns=["cell_id", "scored", "schwann_prob"])
        scored = pred[pred.scored]
        if len(scored) != len(cells):
            problems.append(f"predictions table has {len(scored)} scored cells, cells table has {len(cells)} rows")
        tau = json.load(open(f"{DATA}/operating_point.json"))["tau"]
        n_pos = int((scored.schwann_prob > tau).sum())
        n_tab = int((cells.schwann_prob > tau).sum())
        if n_pos != n_tab:
            problems.append(f"above threshold {tau}: predictions table has {n_pos}, cells table has {n_tab}")
    return problems


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--slides", nargs="*", help="default checks all slides with existing outputs")
    a = ap.parse_args()
    slides = a.slides or sorted(
        os.path.basename(p) for p in glob.glob(str(CELL_ROOT / "*" / "outputs" / "*")) if os.path.isdir(p))
    if not slides:
        sys.exit("no output directories found")

    op = f"{DATA}/operating_point.json"
    if os.path.exists(op):
        j = json.load(open(op))
        print(f"operating point: {j['rule']}  tau={j['tau']:.4f}  "
              f"achieved positive fraction {j.get('achieved_pos_frac', float('nan')):.5f}", flush=True)

    bad = {}
    for s in slides:
        p = check_slide(s)
        if p:
            bad[s] = p
    print(f"\nchecked {len(slides)} slides, consistent {len(slides) - len(bad)}, problematic {len(bad)}", flush=True)
    for s, ps in list(bad.items())[:20]:
        print(f"  [{s}]", flush=True)
        for x in ps:
            print(f"      {x}", flush=True)
    if bad:
        json.dump(bad, open(f"{OUT}/consistency_failures.json", "w"), ensure_ascii=False, indent=1)
        sys.exit(f"FATAL: outputs for {len(bad)} slides are self-contradictory")
    print("all consistent", flush=True)
