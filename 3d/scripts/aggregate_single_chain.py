#!/usr/bin/env python3
import argparse
import csv
import json
import os
import struct

import numpy as np
import pandas as pd

P3D = os.environ.get("THREED_ROOT", os.path.join(os.environ.get("PROJECTS_ROOT", "/data/heid"), "3d"))
UM = 8.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sample", required=True)
    ap.add_argument("--volume", required=True, help="volume_8um dir with cell_points_metadata.json")
    ap.add_argument("--pred", required=True, help="predictions dir, <slide>.parquet")
    ap.add_argument("--sections", required=True, help="sections tsv: section_id, z_position_um")
    ap.add_argument("--tls-root", default="", help="per-slide tls_define outputs; default tls_define/outputs/<sample>")
    ap.add_argument("--out", default="", help="default tls_define/outputs/tls_3d_table_<sample>.tsv")
    a = ap.parse_args()
    tls_root = a.tls_root or os.path.join(P3D, "tls_define", a.sample)
    out = a.out or os.path.join(P3D, "tls_define", f"tls_3d_table_{a.sample}.tsv")

    meta = json.load(open(os.path.join(a.volume, "cell_points_metadata.json")))
    cls_of = {c: i for i, c in enumerate(meta["classes"])}
    z_of = {r["section_id"]: float(r["z_position_um"])
            for r in csv.DictReader(open(a.sections), delimiter="\t")}

    def canvas_map(slide):
        sid = slide
        if not any(s["section_id"] == sid for s in meta["sections"]):
            return None, None
        f = [x for x in meta["files"] if x["basis"] == "G_withdrawn" and x["section_id"] == sid]
        if not f:
            return None, None
        raw = open(os.path.join(a.volume, f[0]["file"]), "rb").read()
        magic, n, cw, ch, sub, ncls, _ = struct.unpack("<8sIHHHHI", raw[:24])
        assert magic == b"CELLPT01"
        xy = np.frombuffer(raw[32:32 + 4 * n], dtype="<u2").reshape(n, 2).astype(np.float64)
        cl = np.frombuffer(raw[32 + 5 * n:32 + 6 * n], dtype=np.uint8)
        pred = pd.read_parquet(os.path.join(a.pred, slide + ".parquet"), columns=["pred_class_name"])
        if len(pred) != n:
            return None, None
        idx = pred["pred_class_name"].map(cls_of).to_numpy()
        if not np.array_equal(idx, cl):
            return None, None
        return xy / sub * UM, sid

    rows = []
    for slide in sorted(os.listdir(tls_root)):
        d = os.path.join(tls_root, slide)
        mp = os.path.join(d, f"{slide}_meta.json")
        tp = os.path.join(d, f"{slide}_tls.csv")
        cp = os.path.join(d, f"{slide}_tls_cells.csv")
        if not (os.path.exists(mp) and os.path.exists(tp) and os.path.exists(cp)):
            continue
        m = json.load(open(mp))
        if not m.get("n_tls_accepted"):
            continue
        cm, sid = canvas_map(slide)
        if cm is None:
            print(f"{slide}: canvas map unavailable, skipped"); continue
        pc = pd.read_csv(cp)
        if "tls_local_halo150" in pc:
            pc["tls_region_halo150"] = pc["tls_local_halo150"]
        assert len(pc) == len(cm), (slide, len(pc), len(cm))
        tdf = pd.read_csv(tp)
        if "tls_local" in tdf:
            tdf["tls_id"] = tdf["tls_local"]
        for _, t in tdf.iterrows():
            sel = (pc["in_tls_core"].astype(bool)
                   & (pc["tls_region_halo150"] == t["tls_id"])).to_numpy()
            if not sel.any():
                continue
            cx, cy = cm[sel, 0].mean(), cm[sel, 1].mean()
            rows.append({"sample_id": a.sample, "slide_id": slide,
                         "section_id": sid, "z_um": z_of[sid],
                         "tls_id": t["tls_id"], "n_cells_core": int(sel.sum()),
                         "rescued": bool(t.get("rescued", False)),
                         "core_area_mm2": t.get("core_area_mm2", np.nan),
                         "cx_um_canvas": round(cx, 1), "cy_um_canvas": round(cy, 1)})
    df = pd.DataFrame(rows).sort_values(["z_um", "tls_id"])
    df.to_csv(out, sep="\t", index=False)
    print(f"wrote {out}: {len(df)} TLS rows over {df['z_um'].nunique()} sections")


if __name__ == "__main__":
    main()
