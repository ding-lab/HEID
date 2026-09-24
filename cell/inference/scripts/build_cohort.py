import os
PROJECTS_ROOT = os.environ.get("PROJECTS_ROOT", "/data/heid")
import json, os, sys, time
import numpy as np
import pandas as pd
import urllib.request
import urllib.parse

INFERENCE_ROOT = (PROJECTS_ROOT + "/cell/inference")
DATA = f"{INFERENCE_ROOT}/data"
BRCA_CLIN = f"{DATA}/BRCA_clinicalMatrix.tsv"


PRAD_SVS = os.environ.get("CELL_PRAD_SVS", f"{DATA}/tcga_prad/svs")
GDC_SHEET = f"{DATA}/gdc_sample_sheet.tsv"
PAM50_COL = "PAM50Call_RNAseq"
TUMOUR = {1, 2, 3, 4, 5, 6, 7, 8, 9}
EXCLUDE_PROJECTS = {"TCGA-READ"}
TARGET_MPP = 0.2125


def gdc_projects(patients, chunk=300):
    out = {}
    pats = sorted(patients)
    for i in range(0, len(pats), chunk):
        part = pats[i:i + chunk]
        body = json.dumps({
            "filters": {"op": "in", "content": {"field": "submitter_id", "value": part}},
            "fields": "submitter_id,project.project_id,"
                      "diagnoses.pathology_details.perineural_invasion_present",
            "size": len(part) + 10, "format": "JSON"}).encode()
        req = urllib.request.Request("https://api.gdc.cancer.gov/cases", data=body,
                                     headers={"Content-Type": "application/json"})
        for attempt in range(3):
            try:
                with urllib.request.urlopen(req, timeout=180) as r:
                    d = json.loads(r.read())
                break
            except Exception as e:
                if attempt == 2:
                    print(f"  [warn] GDC batch {i//chunk} failed: {e}", flush=True)
                    d = {"data": {"hits": []}}
                time.sleep(5)
        for h in d["data"]["hits"]:
            pni = None
            for dg in h.get("diagnoses", []) or []:
                for pd_ in dg.get("pathology_details", []) or []:
                    v = pd_.get("perineural_invasion_present")
                    if v is not None:
                        pni = str(v)
            out[h["submitter_id"]] = (h.get("project", {}).get("project_id"), pni)
        print(f"  GDC {min(i+chunk, len(pats))}/{len(pats)}", flush=True)
    return out


def fold_in_prad(d):
    import glob, re
    import tifffile
    files = sorted(glob.glob(f"{PRAD_SVS}/*.svs"))
    if not files:
        print(f"[warn] no PRAD slides found under {PRAD_SVS}", flush=True)
        return d
    rows = []
    for i, path in enumerate(files, 1):
        base = os.path.basename(path)
        m = re.match(r"(TCGA-[A-Z0-9]{2}-[A-Z0-9]{4})-(\d{2})([A-Z])", base)
        try:
            with tifffile.TiffFile(path) as tf:
                page = tf.pages[0]
                w, h = int(page.imagewidth), int(page.imagelength)
                desc = page.description or ""
            mm = re.search(r"MPP\s*=\s*([\d.]+)", desc)
            ma = re.search(r"AppMag\s*=\s*([\d.]+)", desc)
            mpp = float(mm.group(1)) if mm else np.nan
            mag = float(ma.group(1)) if ma else np.nan
        except Exception:
            w = h = -1
            mpp = mag = np.nan
        rows.append({"file": base, "path": path, "rel_dir": "tcga_prad",
                     "patient": m.group(1) if m else None,
                     "sample_type_code": int(m.group(2)) if m else None,
                     "sample_class": "tumour", "level0_W": w, "level0_H": h,
                     "level0_MP": (w * h / 1e6) if w > 0 else np.nan,
                     "native_mpp": mpp, "appmag": mag, "reader": "tifffile",
                     "bytes": os.path.getsize(path)})
        if i % 100 == 0:
            print(f"  PRAD {i}/{len(files)}", flush=True)
    print(f"merged in PRAD {len(rows):,d} slides", flush=True)
    return pd.concat([d, pd.DataFrame(rows)], ignore_index=True)


if __name__ == "__main__":
    INV = os.environ.get("CELL_INVENTORY", "tcga_he_inventory_raw")
    SEL = os.environ.get("CELL_SELECTED", "cohort_selected")
    d = pd.read_csv(f"{DATA}/{INV}.csv")
    print(f"raw inventory {len(d):,d} slides", flush=True)
    if os.environ.get("CELL_FOLD_PRAD", "1") == "1":
        d = fold_in_prad(d)
    dropped = []


    bad = ~d.sample_type_code.isin(TUMOUR)
    if bad.any():
        t = d[bad].copy(); t["drop_reason"] = "non-tumour sample type code"
        dropped.append(t)
    d = d[~bad].copy()
    print(f"after removing non-tumour {len(d):,d} slides (removed {int(bad.sum())})", flush=True)


    bad = d.level0_W <= 0
    if bad.any():
        t = d[bad].copy(); t["drop_reason"] = "could not read dimensions from file header"
        dropped.append(t)
    d = d[~bad].copy()


    d["_mpp_rank"] = d.native_mpp.fillna(9.9)
    d = d.sort_values(["patient", "level0_MP", "_mpp_rank", "file"],
                      ascending=[True, False, True, True])
    pick = d.groupby("patient", as_index=False).head(1).copy()
    rest = d[~d.index.isin(pick.index)].copy()
    if len(rest):
        rest["drop_reason"] = "smaller slide from the same patient"
        dropped.append(rest)
    print(f"after taking the largest per patient {len(pick):,d} slides (dropped {len(rest):,d} from the same patient)", flush=True)


    print("querying GDC for cancer type and perineural invasion annotation...", flush=True)
    mp = gdc_projects(pick.patient.dropna().unique().tolist())
    pick["project_id"] = pick.patient.map(lambda p: (mp.get(p) or (None, None))[0])

    miss = pick.project_id.isna() & (pick.rel_dir == "tcga_prad")
    pick.loc[miss, "project_id"] = "TCGA-PRAD"
    pick["pni"] = pick.patient.map(lambda p: (mp.get(p) or (None, None))[1])


    if os.path.exists(GDC_SHEET):
        sheet = pd.read_csv(GDC_SHEET, sep="\t", low_memory=False)
        desc = sheet["Tumor Descriptor"].fillna("").astype(str)
        keep_files = set(sheet.loc[desc.str.contains("Primary")
                                   & ~desc.str.contains("Metastatic"), "File Name"])
        known = pick.file.isin(sheet["File Name"])
        bad = known & ~pick.file.isin(keep_files)
        if bad.any():
            t = pick[bad].copy(); t["drop_reason"] = "not primary tumour (GDC Tumor Descriptor)"
            dropped.append(t)
        pick = pick[~bad].copy()
        print(f"primary tumour filter: removed {int(bad.sum())} slides;"
              f"not covered by the sheet {int((~known).sum())} slides (PRAD downloaded separately, not in this table)", flush=True)
    else:
        print(f"[warn] missing {GDC_SHEET}, skipping primary tumour filter", flush=True)


    drop_proj = pick.project_id.isin(EXCLUDE_PROJECTS)
    if drop_proj.any():
        t = pick[drop_proj].copy(); t["drop_reason"] = "cancer type outside this cohort's scope"
        dropped.append(t)
        pick = pick[~drop_proj].copy()
        print(f"excluding cancer types {sorted(EXCLUDE_PROJECTS)}: removed {int(drop_proj.sum())} slides", flush=True)


    if os.path.exists(BRCA_CLIN):
        clin = pd.read_csv(BRCA_CLIN, sep="\t", low_memory=False)
        clin["patient"] = clin["sampleID"].astype(str).str.slice(0, 12)
        pam = (clin.loc[clin[PAM50_COL].notna(), ["patient", PAM50_COL]]
               .drop_duplicates("patient").set_index("patient")[PAM50_COL])
        is_brca = pick.project_id == "TCGA-BRCA"
        has_pam = pick.patient.isin(pam.index)
        drop = is_brca & ~has_pam
        if drop.any():
            t = pick[drop].copy(); t["drop_reason"] = "BRCA missing PAM50 subtype"
            dropped.append(t)
        pick = pick[~drop].copy()
        pick["pam50"] = pick.patient.map(pam)
        print(f"BRCA PAM50 filter: removed {int(drop.sum())} slides, kept {int((pick.project_id=='TCGA-BRCA').sum())} BRCA slides",
              flush=True)
    else:
        print(f"[warn] missing {BRCA_CLIN}, skipping PAM50 filter", flush=True)
        pick["pam50"] = pd.NA


    pick["flag_mpp_missing"] = pick.native_mpp.isna()
    pick["flag_20x"] = pick.appmag == 20
    pick["flag_mpp_far"] = (~pick.native_mpp.isna()) & (
        (pick.native_mpp < 0.8 * TARGET_MPP) | (pick.native_mpp > 1.5 * TARGET_MPP))
    pick["flag_small"] = pick.level0_MP < 100
    pick["upsample_factor"] = pick.native_mpp / TARGET_MPP

    cols = ["patient", "file", "path", "project_id", "pni", "pam50", "sample_type_code",
            "level0_W", "level0_H", "level0_MP", "native_mpp", "appmag", "upsample_factor",
            "flag_mpp_missing", "flag_20x", "flag_mpp_far", "flag_small", "bytes"]
    pick[cols].sort_values(["project_id", "patient"]).to_csv(f"{DATA}/{SEL}.csv", index=False)
    if dropped:
        pd.concat(dropped)[["patient", "file", "path", "level0_MP", "native_mpp",
                            "drop_reason"]].to_csv(f"{DATA}/{SEL.replace('selected', 'dropped')}.csv", index=False)

    print("\n=== final cohort ===", flush=True)
    print(f"slides {len(pick):,d} = patients {pick.patient.nunique():,d} (one slide per patient)", flush=True)
    print(f"\nby cancer type:\n{pick.project_id.value_counts(dropna=False).to_string()}", flush=True)
    print(f"\naudit flags:", flush=True)
    for c in ("flag_20x", "flag_mpp_missing", "flag_mpp_far", "flag_small"):
        print(f"  {c:20s} {int(pick[c].sum()):5d}", flush=True)
    if "pam50" in pick:
        print(f"\nBRCA PAM50 subtype: {pick.pam50.value_counts(dropna=False).to_dict()}", flush=True)
    print(f"\nperineural invasion annotation available: {int(pick.pni.notna().sum()):,d} cases"
          f"({pick.pni.value_counts().to_dict()})", flush=True)
    print(f"\nlevel-0 pixels (millions):\n{pick.level0_MP.describe().round(1).to_string()}", flush=True)
    print(f"\n-> {DATA}/{SEL}.csv", flush=True)
