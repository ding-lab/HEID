#!/usr/bin/env python3
import csv, json, os, struct
import numpy as np, pandas as pd

P3D = os.environ.get("THREED_ROOT", os.path.join(os.environ.get("PROJECTS_ROOT", "/data/heid"), "3d"))


TLS = os.path.join(P3D, "tls_define")
TLSROOT = os.environ.get("HTAN3D_TLSROOT", "")
VOL = os.environ.get("HTAN3D_VOL") or os.path.join(P3D, "reconstruction/volume_8um_s22M")
PRED = os.environ.get("HTAN3D_PRED") or os.path.join(P3D, "inference/S22-27909/data/predictions")
GEN = os.environ.get("HTAN3D_GEN") or os.path.join(P3D, "front/S22-27909")
TABLE = os.environ.get("HTAN3D_TABLE", "")
UM = 8.0

meta = json.load(open(os.path.join(VOL, "cell_points_metadata.json")))
cls_of = {c: i for i, c in enumerate(meta["classes"])}


_vz = json.load(open(os.path.join(VOL, "metadata.json")))["z_um"]
VOL_Z = {s["section_id"]: float(_vz[s["index"]]) for s in meta["sections"]}

import json as _json
_split = _json.load(open(os.path.join(GEN, "configs/split_provenance.json")))
_who = {u: p for p, us in _split["patients"].items() for u in us}

def sid_of(slide):
    u = int(slide.rsplit("_U", 1)[1].rstrip("r"))
    return f"S22-27909-{_who[u]}_U{u}"

def canvas_map(slide):
    sid = sid_of(slide)
    if not any(s["section_id"] == sid for s in meta["sections"]):
        return None, None
    f = [x for x in meta["files"] if x["basis"] == "G_withdrawn" and x["section_id"] == sid]
    if not f:
        return None, None
    raw = open(os.path.join(VOL, f[0]["file"]), "rb").read()
    magic, n, cw, ch, sub, ncls, _ = struct.unpack("<8sIHHHHI", raw[:24])
    assert magic == b"CELLPT01"
    xy = np.frombuffer(raw[32:32 + 4 * n], dtype="<u2").reshape(n, 2).astype(np.float64)
    cl = np.frombuffer(raw[32 + 5 * n:32 + 6 * n], dtype=np.uint8)
    pred = pd.read_parquet(os.path.join(PRED, slide + ".parquet"),
                           columns=["pred_class_name"])
    if len(pred) != n:
        return None, None
    idx = pred["pred_class_name"].map(cls_of).to_numpy()
    if not np.array_equal(idx, cl):
        return None, None
    return xy / sub * UM, sid

rows = []
z_of = {}
for pat in ("P1", "P2"):
    for r in csv.DictReader(open(f"{GEN}/configs/sections_{pat}.tsv"), delimiter="\t"):
        z_of[r["section_id"]] = float(r["z_position_um"])
for sample in ("S22-27909",):
    root = TLSROOT or os.path.join(TLS, sample)
    for slide in sorted(os.listdir(root)):
        d = os.path.join(root, slide)
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
            rows.append({"sample_id": "S22-27909", "slide_id": slide,
                         "section_id": sid, "z_um": VOL_Z[sid],
                         "tls_id": t["tls_id"], "n_cells_core": int(sel.sum()),
                         "core_area_mm2": t.get("core_area_mm2", np.nan),
                         "cx_um_canvas": round(cx, 1), "cy_um_canvas": round(cy, 1)})
df = pd.DataFrame(rows).sort_values(["z_um", "tls_id"])
out = TABLE or os.path.join(TLS, "tls_3d_table_S22-27909.tsv")
df.to_csv(out, sep="\t", index=False)
print(f"wrote {out}: {len(df)} TLS rows over {df['z_um'].nunique()} sections")
