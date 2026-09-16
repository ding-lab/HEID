#!/usr/bin/env python3
import os
PROJECTS_ROOT = os.environ.get("PROJECTS_ROOT", "/data/heid")
import os
import csv
import json


PDC = PROJECTS_ROOT
HEID = os.path.join(PDC, "cell")
DETECTION_ROOT = os.path.join(HEID, "detection")


DETECTION_POOL_CSV = os.environ.get("DETECTION_POOL_CSV", os.path.join(DETECTION_ROOT, "data", "detection_pool_per_sample.csv"))


X1000_HE_ROOT = os.path.join(PDC, "align", "registered_he")
HE_FILENAMES = ("he_on_xenium.ome.tif", "he_aligned.ome.tif")


XCACHE_ROOT = os.path.join(DETECTION_ROOT, "data", "xenium_cache")


CELLTYPE_SPLIT = os.path.join(HEID, "inputs", "split_5fold.json")


PXSIZE_UM = 0.2125

VIABLE_CANCERS = ["BRCA", "CRC", "PDAC", "RCC", "HNSC", "LUNG"]
THIN_CANCERS = ["CHOL", "SKCM"]
BLOCKED_THIN = ["GBM"]
PRAD_GATED = "PRAD"


def resolve_he(cohort, cancer, sample):
    base = os.path.join(X1000_HE_ROOT, str(cohort), cancer, sample)
    for fn in HE_FILENAMES:
        p = os.path.join(base, fn)
        if os.path.exists(p):
            return p, fn
    return None, None


def he_align_json(cohort, cancer, sample):
    return os.path.join(X1000_HE_ROOT, str(cohort), cancer, sample, "global_alignment.json")


def xcache_path(cancer, sample, kind="nucleus_boundaries"):
    return os.path.join(XCACHE_ROOT, cancer, f"{sample}__{kind}.parquet")


def labels_root(cancer):
    return os.path.join(DETECTION_ROOT, "outputs", "labels", cancer)


def _load_celltype_folds():
    with open(CELLTYPE_SPLIT) as f:
        d = json.load(f)
    s2f, p2f = {}, {}
    for fo in d["folds"]:
        k = int(fo["fold"])
        for s in fo.get("test_samples", []):
            s2f[s] = k
        for p in fo.get("test_patients", []):
            p2f[p] = k
    return s2f, p2f


def _read_pool():
    with open(DETECTION_POOL_CSV) as f:
        return list(csv.DictReader(f))


def _assign_detection_only_fold(rows, s2f, p2f):
    per_cancer = {}
    for r in rows:
        if r["sample"] in s2f or r["patient"] in p2f:
            continue
        per_cancer.setdefault(r["cancer"], {}).setdefault(r["patient"], 0)
        per_cancer[r["cancer"]][r["patient"]] += 1
    extra = {}
    for cancer, pats in per_cancer.items():
        load = [0] * 5
        for pat in sorted(pats, key=lambda p: (-pats[p], p)):
            k = min(range(5), key=lambda i: (load[i], i))
            extra[pat] = k
            load[k] += pats[pat]
    return extra


def load_manifest(cancer=None, clean_only=True, include_flagged=False):
    rows = _read_pool()
    s2f, p2f = _load_celltype_folds()
    extra = _assign_detection_only_fold(rows, s2f, p2f)

    want = None if cancer is None else set(
        ([c.upper() for c in cancer] if isinstance(cancer, (list, tuple, set)) else [cancer.upper()]))
    out = []
    for r in rows:
        if want is not None and r["cancer"] not in want:
            continue
        if r["viable_detection"] != "True":
            continue
        flagged = (r["soft_flag_ncc<0.2"] == "True")
        if clean_only and flagged and not include_flagged:
            continue
        s, p = r["sample"], r["patient"]
        if s in s2f:
            fold = s2f[s]
        elif p in p2f:
            fold = p2f[p]
        else:
            fold = extra.get(p, 0)
        out.append(dict(
            sample=s, cohort=r["cohort"], cancer=r["cancer"], patient=p,
            fold=int(fold), typing_eligible=(s in s2f),
            ncc=float(r["ncc_he_to_dapi_max"]) if r["ncc_he_to_dapi_max"] else None,
            soft_flag=flagged, cells_rows=int(r["cells_rows"]) if r["cells_rows"] else 0,
        ))
    out.sort(key=lambda x: (x["cancer"], x["fold"], x["sample"]))
    return out


def fold_of(cancer, sample):
    for r in load_manifest(cancer=cancer, clean_only=False, include_flagged=True):
        if r["sample"] == sample:
            return r["fold"]
    return None


if __name__ == "__main__":
    import argparse
    from collections import Counter
    ap = argparse.ArgumentParser(description="print the detection manifest")
    ap.add_argument("--cancer", default=None, help="cancer code or omit for pancancer")
    ap.add_argument("--include_flagged", action="store_true")
    a = ap.parse_args()
    m = load_manifest(cancer=a.cancer, include_flagged=a.include_flagged)
    fold_c = Counter(r["fold"] for r in m)
    can_c = Counter(r["cancer"] for r in m)
    typ = sum(r["typing_eligible"] for r in m)
    print(f"manifest: {len(m)} samples  typing_eligible={typ}  "
          f"cancers={dict(can_c)}  folds={dict(sorted(fold_c.items()))}")
    for r in m[:6]:
        print("  ", r["cancer"], r["sample"], "fold", r["fold"],
              "typ" if r["typing_eligible"] else "det-only",
              f"ncc={r['ncc']}")
