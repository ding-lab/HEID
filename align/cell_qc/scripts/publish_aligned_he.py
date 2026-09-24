#!/usr/bin/env python
import argparse
import csv
import json
import os
import shutil
import sys
from pathlib import Path

import pandas as pd

PROJ = Path(os.environ.get("PROJECTS_ROOT", "/data/heid"))
QC_ROOT = Path(__file__).resolve().parents[1]
PUBLISH_ROOT = PROJ / "align/aligned_he"
REGISTERED_ROOT = PROJ / "align/registered_he"
TILES = QC_ROOT / "outputs/offsets"
PER_BLOCK = QC_ROOT / "outputs/results/per_block"
KEEP = QC_ROOT / "outputs/results/cell_keep"

IMAGE = "he_aligned.ome.tif"
MIN_PROM = 0.02
MAX_WEAK_FRAC = 0.50
MIN_MEASURED = 30
MIN_KEEP = 0.80
SUFFIXES = ("-bad-lowqc", "-lowqc", "-weak", "-blur", "-bad")
SUMMARY_COLUMNS = ["cohort", "cancer", "field", "lowqc", "n_tiles", "n_measured",
                   "tiles_weak", "weak_frac", "n_cells", "n_keep", "keep_frac",
                   "blur_measured", "stage_reached"]


def log(m):
    print(m, flush=True)


def split_suffix(name):
    for suffix in SUFFIXES:
        if name.endswith(suffix):
            return name[: -len(suffix)], suffix
    return name, ""


def tile_stats(path):
    t = pd.read_csv(path)
    meas = t[t.status == "ok"]
    weak = int((meas.he_prominence < MIN_PROM).sum())
    return {"n_tiles": len(t), "n_measured": len(meas), "tiles_weak": weak,
            "weak_frac": weak / len(meas) if len(meas) else float("nan")}


def section_dirs(root):
    for cohort in sorted(p for p in root.iterdir() if p.is_dir() and not p.name.startswith("_")):
        for cancer in sorted(p for p in cohort.iterdir() if p.is_dir()):
            for d in sorted(p for p in cancer.iterdir() if p.is_dir()):
                yield cohort.name, cancer.name, d


def publish_field(a):
    keep_json = Path(a.keep_dir) / f"{a.field}.json"
    summary = json.loads(keep_json.read_text())
    if not summary.get("blur_measured"):
        log(f"  refusing to publish {a.field}: the image-quality rules were not measured")
        return 2
    tiles_csv = Path(a.tiles_dir) / f"{a.field}_tiles.csv"
    stats = tile_stats(tiles_csv)
    weak = stats["n_measured"] >= MIN_MEASURED and stats["weak_frac"] > MAX_WEAK_FRAC
    name = a.field + ("-weak" if weak else "")
    dest = Path(a.publish_root) / a.cohort / a.cancer / name
    micro = dest / "micro"
    for sub in ("offsets", "per_block", "cell_keep"):
        (micro / sub).mkdir(parents=True, exist_ok=True)

    registered = Path(a.registered_dir) if a.registered_dir else (
        REGISTERED_ROOT / a.cohort / a.cancer / a.field)
    image = registered / IMAGE
    if not image.is_file():
        raise FileNotFoundError(f"registered image missing: {image}")
    target = dest / IMAGE
    if not target.exists():
        if a.image == "link":
            target.symlink_to(image.resolve())
        else:
            shutil.copy2(image, target)

    shutil.copy2(tiles_csv, micro / "offsets" / tiles_csv.name)
    tiles_json = tiles_csv.with_name(f"{a.field}.json")
    if tiles_json.is_file():
        shutil.copy2(tiles_json, micro / "offsets" / tiles_json.name)
    shutil.copy2(Path(a.per_block_dir) / f"{a.field}.json", micro / "per_block" / f"{a.field}.json")
    shutil.copy2(Path(a.keep_dir) / f"{a.field}_cells.parquet", micro / "cell_keep" / f"{a.field}_cells.parquet")
    shutil.copy2(keep_json, micro / "cell_keep" / keep_json.name)
    shutil.copy2(Path(a.keep_dir) / f"{a.field}_cells_for_training.parquet",
                 micro / f"{a.field}_cells_for_training.parquet")
    log(f"  published {a.cohort}/{a.cancer}/{name}  weak {stats['tiles_weak']}/{stats['n_measured']}")
    return 0


def mark_blur(a):
    root = Path(a.publish_root)
    mapfile = root / "_report" / "blur_rename_map.tsv"
    rows = []
    for cohort, cancer, d in section_dirs(root):
        field, suffix = split_suffix(d.name)
        if suffix:
            continue
        keep_json = d / "micro" / "cell_keep" / f"{field}.json"
        if not keep_json.is_file():
            continue
        m = json.loads(keep_json.read_text())
        if m["n_keep"] == 0 or m["frac_keep"] > a.min_keep:
            continue
        rows.append({"cohort": cohort, "cancer": cancer, "field": field,
                     "n_cells": m["n_cells"], "n_keep": m["n_keep"],
                     "frac_keep": round(m["frac_keep"], 4),
                     "old": str(d), "new": str(d.with_name(field + "-blur"))})
    log(f"  {len(rows)} sections keep <= {a.min_keep:.2f} of their cells")
    if rows:
        mapfile.parent.mkdir(parents=True, exist_ok=True)
        existing = mapfile.is_file()
        with mapfile.open("a", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows[0]), delimiter="\t", lineterminator="\n")
            if not existing:
                w.writeheader()
            w.writerows(rows)
        for r in rows:
            Path(r["old"]).rename(r["new"])
    elif not mapfile.is_file():
        mapfile.parent.mkdir(parents=True, exist_ok=True)
        mapfile.write_text("\t".join(["cohort", "cancer", "field", "n_cells", "n_keep",
                                      "frac_keep", "old", "new"]) + "\n")
    return 0


def report(a):
    root = Path(a.publish_root)
    rows = []
    for cohort, cancer, d in section_dirs(root):
        field, suffix = split_suffix(d.name)
        row = {"cohort": cohort, "cancer": cancer, "field": field,
               "lowqc": suffix in ("-lowqc", "-bad-lowqc", "-bad")}
        tiles_csv = d / "micro" / "offsets" / f"{field}_tiles.csv"
        if tiles_csv.is_file():
            row.update(tile_stats(tiles_csv))
        keep_json = d / "micro" / "cell_keep" / f"{field}.json"
        if keep_json.is_file():
            m = json.loads(keep_json.read_text())
            row.update(n_cells=m["n_cells"], n_keep=m["n_keep"], keep_frac=m["frac_keep"],
                       blur_measured=m.get("blur_measured"))
        row["stage_reached"] = ("cell_keep" if keep_json.is_file() else
                                "offsets" if tiles_csv.is_file() else "none")
        rows.append(row)

    lowqc = pd.DataFrame(columns=["cohort", "cancer", "field", "reason"])
    if a.lowqc_list:
        lowqc = pd.read_csv(a.lowqc_list, sep="\t", dtype=str)
        missing = {"cohort", "cancer", "field", "reason"} - set(lowqc.columns)
        if missing:
            raise ValueError(f"{a.lowqc_list}: missing columns {sorted(missing)}")
        known = {(r["cohort"], r["cancer"], r["field"]) for r in rows}
        for r in lowqc.itertuples(index=False):
            if (r.cohort, r.cancer, r.field) not in known:
                rows.append({"cohort": r.cohort, "cancer": r.cancer, "field": r.field,
                             "lowqc": True, "stage_reached": "none"})
            else:
                for row in rows:
                    if (row["cohort"], row["cancer"], row["field"]) == (r.cohort, r.cancer, r.field):
                        row["lowqc"] = True

    df = pd.DataFrame(rows).reindex(columns=SUMMARY_COLUMNS)
    if df.duplicated(["cohort", "cancer", "field"]).any():
        raise RuntimeError("a field is published more than once")
    out = root / "_report"
    out.mkdir(parents=True, exist_ok=True)
    df.sort_values(["cohort", "cancer", "field"]).to_csv(out / "cohort_summary.csv", index=False)
    lowqc[["cohort", "cancer", "field", "reason"]].to_csv(out / "lowqc_samples.csv", index=False)
    log(f"  cohort_summary.csv {len(df)} sections;  lowqc_samples.csv {len(lowqc)} sections")
    return 0


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--publish-root", default=str(PUBLISH_ROOT))
    sub = ap.add_subparsers(dest="command", required=True)

    f = sub.add_parser("field", help="publish one processed section")
    f.add_argument("--field", required=True)
    f.add_argument("--cancer", required=True)
    f.add_argument("--cohort", default="5k")
    f.add_argument("--registered-dir", default=None,
                   help=f"directory holding {IMAGE}; default <registered root>/<cohort>/<cancer>/<field>")
    f.add_argument("--tiles-dir", default=str(TILES))
    f.add_argument("--per-block-dir", default=str(PER_BLOCK))
    f.add_argument("--keep-dir", default=str(KEEP))
    f.add_argument("--image", choices=("link", "copy"), default="link")

    b = sub.add_parser("mark-blur", help="suffix sections that keep too few cells")
    b.add_argument("--min-keep", type=float, default=MIN_KEEP)

    r = sub.add_parser("report", help="write the cohort tables under _report/")
    r.add_argument("--lowqc-list", default=None,
                   help="TSV with columns cohort, cancer, field, reason")

    a = ap.parse_args()
    return {"field": publish_field, "mark-blur": mark_blur, "report": report}[a.command](a)


if __name__ == "__main__":
    sys.exit(main())
