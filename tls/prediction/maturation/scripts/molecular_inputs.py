import glob
import os
from pathlib import Path
import numpy as np
import pandas as pd
PARQ = F128 = LAB = MEMBERSHIP = RAW_DIR = None
RAW_GENES = ["BCL6", "AICDA", "MKI67", "CXCL13", "CR2", "CR1", "FCER2", "CXCL12", "CCL19", "PDPN", "CXCR5", "PDCD1", "ICOS", "IL21", "CD3E"]
LOCAL = RAW_GENES
UM = 0.2125
PAD = 200.0

def configure(config):
    global PARQ, F128, LAB, MEMBERSHIP, RAW_DIR
    PARQ = Path(config["cells"])
    F128 = Path(config["features"])
    LAB = Path(config["region_grids"])
    MEMBERSHIP = Path(config["tls_membership"])
    RAW_DIR = Path(config["raw_markers"])

def tls_csv(s):
    h = glob.glob(str(MEMBERSHIP / "5k/*" / s / f"{s}_tls.csv")); return h[0] if h else None

def load_sample(s):
    tp = tls_csv(s); pq = PARQ / f"{s}.parquet"; npz = LAB / s / f"{s}_tls_region_grid.npz"
    if not (tp and pq.exists() and npz.exists() and (F128 / s / "uni2.pt").exists()):
        return None
    use_raw = True
    t = pd.read_csv(tp, usecols=["cell_id", "tls_region"], low_memory=False); t = t[t.tls_region.notna()]
    t["cell_id"] = t.cell_id.astype(str)
    if use_raw:
        rp = RAW_DIR / f"{s}.parquet"
        if not rp.exists():
            return None
        p = pd.read_parquet(pq, columns=["cell_id", "x_centroid", "y_centroid", "cell_type"])
        p["cell_id"] = p.cell_id.astype(str)
        rg = pd.read_parquet(rp); rg["cell_id"] = rg["cell_id"].astype(str)
        p = p.merge(rg, on="cell_id", how="inner"); genes = RAW_GENES
    else:
        p = pd.read_parquet(pq, columns=["cell_id", "x_centroid", "y_centroid", "cell_type"] + LOCAL)
        p["cell_id"] = p.cell_id.astype(str); genes = LOCAL
    if not len(p):
        return None
    for g in genes:
        if g in p.columns:
            v = p[g].astype(float).values; p[g + "_z"] = (v - v.mean()) / (v.std() + 1e-9)
    x0 = p.x_centroid.min() - PAD; y0 = p.y_centroid.min() - PAD
    d = np.load(str(npz)); nxmin = float(d["x_min"]); nymin = float(d["y_min"])
    j = p.merge(t, on="cell_id", how="inner")
    if not len(j):
        return None
    j["hx"] = ((j.x_centroid - x0) + nxmin) / UM; j["hy"] = ((j.y_centroid - y0) + nymin) / UM
    return j
