#!/usr/bin/env python
from __future__ import annotations
import argparse
import csv
import json
import shutil
from pathlib import Path

import pandas as pd


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sections", required=True)
    ap.add_argument("--run-manifest", required=True, dest="run_manifest")
    ap.add_argument("--inference-data", required=True, dest="inference_data")
    ap.add_argument("--sample", default=None, help="keep only this sample's manifest rows")
    ap.add_argument("--match", default="section", choices=["section", "file"],
                    help="how a section finds its manifest row: by section_number (a per-block inference run) "
                         "or by the source file name (a cohort run over whole per-section files)")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    DATA, out = Path(a.inference_data), Path(a.out)

    v3 = {}
    for r in csv.DictReader(open(a.run_manifest), delimiter="\t"):
        if a.sample and r.get("sample") not in (None, "", a.sample):
            continue
        if a.match == "file":
            v3[Path(r["svs"]).name.lower()] = r
            continue
        u = int(r["section_number"])

        if u not in v3 or r["slide"].endswith("r"):
            v3[u] = r

    (out / "cells").mkdir(parents=True, exist_ok=True)
    man_rows, probe = [], []
    for g in csv.DictReader(open(a.sections), delimiter="\t"):
        sid = g["section_id"]; u = int(g["u_number"])
        r = v3.get(Path(g["filepath"]).name.lower()) if a.match == "file" else v3.get(u)
        if r is None:
            if a.match == "file":
                print(f"{sid:<20} no inference row for {Path(g['filepath']).name}: no cells on this plane", flush=True)
                continue
            raise SystemExit(f"{sid}: no inference row for U{u}")
        if abs(float(r["native_mpp"]) - float(g["mpp"])) > 1e-6:
            raise SystemExit(f"{sid}: inference mpp {r['native_mpp']} != chain {g['mpp']}")
        dx = float(r.get("crop_x0") or 0) - float(g.get("crop_x0") or 0)
        dy = float(r.get("crop_y0") or 0) - float(g.get("crop_y0") or 0)

        det = pd.read_parquet(DATA / f"cells/{r['slide']}.parquet", columns=["cell_id", "x_px", "y_px"])
        det["x_px"] = det["x_px"] + dx
        det["y_px"] = det["y_px"] + dy
        det.to_parquet(out / "cells" / f"{sid}.parquet", index=False)

        pred = pd.read_parquet(DATA / f"predictions/{r['slide']}.parquet",
                               columns=["cell_id", "pred_class_name", "scored"])
        pred = pred[pred["scored"].astype(bool) & pred["pred_class_name"].notna()]
        pdir = out / "products" / sid
        pdir.mkdir(parents=True, exist_ok=True)
        pred.rename(columns={"pred_class_name": "cell_type"})[["cell_id", "cell_type"]].to_csv(
            pdir / f"{sid}_cells.csv", index=False)

        full = pd.read_parquet(DATA / f"predictions/{r['slide']}.parquet")
        full["x_um"] = full["x_um"] + dx * float(g["mpp"])
        full["y_um"] = full["y_um"] + dy * float(g["mpp"])
        (out / "predictions").mkdir(exist_ok=True)
        full.to_parquet(out / "predictions" / f"{sid}.parquet", index=False)
        (out / "wmaps").mkdir(exist_ok=True)
        shutil.copyfile(DATA / f"wmaps/{r['slide']}.npz", out / "wmaps" / f"{sid}.npz")

        man_rows.append({"slide": sid, "svs": g["filepath"], "native_mpp": g["mpp"],
                         "level0_W": g["size_x_px"], "level0_H": g["size_y_px"]})
        probe.append({"section_id": sid, "modality": "he", "path": g["filepath"],
                      "level_yx": [[int(g["size_y_px"]), int(g["size_x_px"])]]})
        print(f"{sid:<20} {len(det):>8,} cells  {len(pred):>8,} typed  offset ({dx:+.0f},{dy:+.0f}) px", flush=True)

    with open(out / "run_manifest.tsv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(man_rows[0]), delimiter="\t")
        w.writeheader(); w.writerows(man_rows)
    (out / "source_probe.json").write_text(json.dumps(probe, indent=1))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
