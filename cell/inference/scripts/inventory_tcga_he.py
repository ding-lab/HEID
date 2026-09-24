import os
PROJECTS_ROOT = os.environ.get("PROJECTS_ROOT", "/data/heid")
import os, sys, json, re, time, hashlib
import numpy as np
import pandas as pd

SRC = os.environ.get("CELL_HE_SRC", PROJECTS_ROOT + "/data/tcga_he/HE")
INV = os.environ.get("CELL_INVENTORY", "tcga_he_inventory_raw")
OUT = PROJECTS_ROOT + "/cell/inference/data"


BARCODE = re.compile(r"(TCGA-[A-Z0-9]{2}-[A-Z0-9]{4})-(\d{2})([A-Z])")
TUMOUR_CODES = {"01", "02", "03", "04", "05", "06", "07", "08", "09"}
NORMAL_CODES = {"10", "11", "12", "13", "14", "15", "16", "17", "18", "19", "20"}


def slide_geometry(path):
    try:
        import openslide
        s = openslide.OpenSlide(path)
        w, h = s.level_dimensions[0]
        props = dict(s.properties)
        mpp = props.get("openslide.mpp-x") or props.get("aperio.MPP")
        mag = props.get("openslide.objective-power") or props.get("aperio.AppMag")
        s.close()
        return int(w), int(h), (float(mpp) if mpp else np.nan), \
            (float(mag) if mag else np.nan), "openslide"
    except Exception:
        pass
    try:
        import tifffile
        with tifffile.TiffFile(path) as tf:
            p = tf.pages[0]
            w, h = int(p.imagewidth), int(p.imagelength)
            desc = p.description or ""
            mpp = np.nan
            m = re.search(r"MPP\s*=\s*([\d.]+)", desc)
            if m:
                mpp = float(m.group(1))
            mag = np.nan
            m = re.search(r"AppMag\s*=\s*([\d.]+)", desc)
            if m:
                mag = float(m.group(1))
        return w, h, mpp, mag, "tifffile"
    except Exception as e:
        return -1, -1, np.nan, np.nan, f"unreadable:{type(e).__name__}"


if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True)
    if not os.path.isdir(SRC):
        print(f"[FATAL] source directory not visible: {SRC}", flush=True)
        sys.exit(2)

    files = []
    for root, _, names in os.walk(SRC):
        for n in names:
            if n.lower().endswith((".svs", ".tif", ".tiff", ".ndpi")):
                files.append(os.path.join(root, n))
    files.sort()
    print(f"found {len(files):,d} slide files", flush=True)

    rows, t0 = [], time.time()
    for i, p in enumerate(files, 1):
        base = os.path.basename(p)
        m = BARCODE.search(base)
        patient = m.group(1) if m else None
        stype = m.group(2) if m else None
        w, h, mpp, mag, reader = slide_geometry(p)
        rows.append({
            "file": base, "path": p, "rel_dir": os.path.relpath(os.path.dirname(p), SRC),
            "patient": patient, "sample_type_code": stype,
            "sample_class": ("tumour" if stype in TUMOUR_CODES else
                             "normal" if stype in NORMAL_CODES else "unknown"),
            "level0_W": w, "level0_H": h, "level0_MP": (w * h / 1e6) if w > 0 else np.nan,
            "native_mpp": mpp, "appmag": mag, "reader": reader,
            "bytes": os.path.getsize(p),
        })
        if i % 50 == 0:
            print(f"  {i}/{len(files)}  {time.time()-t0:.0f}s", flush=True)
    d = pd.DataFrame(rows)
    d.to_csv(f"{OUT}/{INV}.csv", index=False)

    print("\n=== inventory summary ===", flush=True)
    print(f"files {len(d):,d}; parseable barcodes {d.patient.notna().sum():,d};"
          f"unique patients {d.patient.nunique():,d}", flush=True)
    print(d.sample_class.value_counts().to_string(), flush=True)
    print(f"\nread failures {int((d.level0_W < 0).sum())}", flush=True)
    if (d.level0_W < 0).any():
        print(d.loc[d.level0_W < 0, ["file", "reader"]].head(10).to_string(), flush=True)
    print(f"\nnative_mpp resolution distribution:\n{d.native_mpp.describe().round(4).to_string()}", flush=True)
    print(f"appmag magnification:\n{d.appmag.value_counts(dropna=False).to_string()}", flush=True)
    print(f"\n-> {OUT}/{INV}.csv", flush=True)
