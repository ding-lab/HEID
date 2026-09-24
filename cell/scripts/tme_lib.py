#!/usr/bin/env python
from __future__ import annotations
import os
PROJECTS_ROOT = os.environ.get("PROJECTS_ROOT", "/data/heid")

import json
import tempfile
from collections import defaultdict
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score

PC = Path(PROJECTS_ROOT + "/cell")


COARSE11 = [
    "Tumor", "Fibroblast", "NK_T", "Myeloid_NOS", "NonMalignant_Parenchymal",
    "Endothelial", "Plasma", "SMC", "B_cell", "Schwann", "Others",
]
N11 = 11
SCORED10 = list(range(10))
OTHERS = 10
C11IDX = {c: i for i, c in enumerate(COARSE11)}


COARSE14 = [
    "Tumor", "Fibroblast", "NK_T", "Myeloid_NOS", "NonMalignant_Parenchymal",
    "Endothelial", "Plasma", "Pericyte/vSMC", "SMC", "B_cell", "Necrosis",
    "Mast", "Schwann", "Others",
]

FINE2REFINE = {
    "Tumor": "Tumor", "Tumor_ccRCC": "Tumor", "Tumor_pRCC": "Tumor",
    "Tumor_chRCC": "Tumor", "Neuroendocrine": "Tumor",
    "Fibroblast": "Fibroblast", "CAF": "Fibroblast",
    "NK_T": "NK_T", "T_cell": "NK_T", "NK": "NK_T",
    "Macrophage": "Macrophage", "Macrophage_Monocyte": "Macrophage",
    "Microglia": "Macrophage", "Kupffer": "Macrophage",
    "Intestinal_epithelium": "NonMalignant_Parenchymal", "Basal": "NonMalignant_Parenchymal",
    "Luminal": "NonMalignant_Parenchymal", "Hepatocyte": "NonMalignant_Parenchymal",
    "Acinar": "NonMalignant_Parenchymal", "Club": "NonMalignant_Parenchymal",
    "Duct_like": "NonMalignant_Parenchymal", "AT2": "NonMalignant_Parenchymal",
    "Epidermal": "NonMalignant_Parenchymal", "Dysplasia": "NonMalignant_Parenchymal",
    "Cholangiocyte": "NonMalignant_Parenchymal", "Basal/Myoepithelial": "NonMalignant_Parenchymal",
    "Squamous_cell": "NonMalignant_Parenchymal", "Islet": "NonMalignant_Parenchymal",
    "Normal_Luminal": "NonMalignant_Parenchymal", "Ciliated": "NonMalignant_Parenchymal",
    "Squamous_epithelium": "NonMalignant_Parenchymal", "AT1": "NonMalignant_Parenchymal",
    "IPMN": "NonMalignant_Parenchymal", "Basal_keratinocyte": "NonMalignant_Parenchymal",
    "Secretory_Epithelium": "NonMalignant_Parenchymal", "Astrocyte": "NonMalignant_Parenchymal",
    "NonMalignant_Parenchymal": "NonMalignant_Parenchymal",
    "Endothelial": "Endothelial", "Sinusoidal_endothelial": "Endothelial",
    "Plasma": "Plasma",
    "Pericyte/vSMC": "Pericyte/vSMC", "Pericyte": "Pericyte/vSMC", "vSMC": "Pericyte/vSMC",
    "Smooth_muscle": "SMC", "SMC": "SMC",
    "B_cell": "B_cell",
    "Unknown": "Others", "Oligodendrocyte": "Others", "Neuron": "Others",
    "Skeletal_muscle": "Others", "Adipocyte": "Others", "Others": "Others",
    "Necrosis": "Necrosis", "Necrotic": "Necrosis",
    "cDC2": "DC", "Langerhans_cell": "DC", "mregDC": "DC", "pDC": "DC",
    "cDC1": "DC", "cDC": "DC", "DC": "DC",
    "Neutrophil": "Neutrophil",
    "Mast": "Granulocyte", "Granulocyte": "Granulocyte",
    "Lymphatic_Endothelial": "Lymphatic_Endothelial",
    "Schwann": "Schwann",
}
REFINE2COARSE11 = {
    "Tumor": "Tumor", "Fibroblast": "Fibroblast", "NK_T": "NK_T",
    "Macrophage": "Myeloid_NOS", "DC": "Myeloid_NOS", "Neutrophil": "Myeloid_NOS",
    "NonMalignant_Parenchymal": "NonMalignant_Parenchymal",
    "Endothelial": "Endothelial", "Lymphatic_Endothelial": "Endothelial",
    "Plasma": "Plasma",
    "Pericyte/vSMC": "SMC", "SMC": "SMC",
    "B_cell": "B_cell", "Schwann": "Schwann",
    "Others": "Others",
    "Necrosis": None, "Granulocyte": "Others",
}


def source_to_identity(label: str) -> int:
    ref = FINE2REFINE.get(label)
    if ref is None:
        return -1
    coarse = REFINE2COARSE11.get(ref)
    if coarse is None:
        return -1
    return C11IDX[coarse]


def src_lut(labels: np.ndarray) -> np.ndarray:
    uniq, inv = np.unique(labels, return_inverse=True)
    lut = np.array([source_to_identity(str(u)) for u in uniq.tolist()], dtype=np.int64)
    return lut[inv]


def a2_projection_matrix(mode: str) -> np.ndarray:
    if mode not in ("select", "marginalise"):
        raise ValueError(mode)
    m = np.zeros((14, N11), dtype=np.float32)
    for i, name in enumerate(COARSE14):
        if name in C11IDX:
            m[i, C11IDX[name]] = 1.0
        elif name == "Pericyte/vSMC" and mode == "marginalise":
            m[i, C11IDX["SMC"]] = 1.0
    return m


def a2_true_label_to_11(y14: np.ndarray) -> np.ndarray:
    lut = np.full(14, -1, dtype=np.int64)
    for i, name in enumerate(COARSE14):
        if name in C11IDX:
            lut[i] = C11IDX[name]
        elif name == "Pericyte/vSMC":
            lut[i] = C11IDX["SMC"]
    return lut[y14]


def confusion11(y: np.ndarray, pred: np.ndarray) -> np.ndarray:
    keep = (y >= 0) & (y < N11)
    z = y[keep].astype(np.int64) * N11 + pred[keep].astype(np.int64)
    return np.bincount(z, minlength=N11 * N11).reshape(N11, N11)


def per_class_f1(cm: np.ndarray) -> np.ndarray:
    tp = np.diag(cm).astype(np.float64)
    support = cm.sum(axis=1).astype(np.float64)
    predicted = cm.sum(axis=0).astype(np.float64)
    prec = np.divide(tp, predicted, out=np.zeros_like(tp), where=predicted > 0)
    rec = np.divide(tp, support, out=np.zeros_like(tp), where=support > 0)
    den = prec + rec
    return np.divide(2 * prec * rec, den, out=np.zeros_like(tp), where=den > 0)


def macro_f1_10(cm: np.ndarray) -> float:
    return float(np.mean(per_class_f1(cm)[SCORED10]))


def macro_f1_11(cm: np.ndarray) -> float:
    return float(np.mean(per_class_f1(cm)))


def macro_ovr_auroc10(y: np.ndarray, prob: np.ndarray):
    per, vals = {}, []
    for c in SCORED10:
        yc = (y == c).astype(np.int8)
        pos = int(yc.sum())
        if pos == 0 or pos == len(yc):
            per[COARSE11[c]] = {"auroc": None, "positives": pos,
                                "negatives": int(len(yc) - pos)}
            continue
        a = float(roc_auc_score(yc, prob[:, c]))
        per[COARSE11[c]] = {"auroc": a, "positives": pos,
                            "negatives": int(len(yc) - pos)}
        vals.append(a)
    return (float(np.mean(vals)) if vals else None), per, int(len(y))


def capped_rows_a4_protocol(sample_records, per_patient, global_per_class, seed):
    rng = np.random.RandomState(seed)
    quota = defaultdict(lambda: per_patient)
    y_parts = []
    extra_parts = defaultdict(list)
    for patient, y, extras in sample_records:
        for c in range(N11):
            key = (patient, c)
            remaining = quota[key]
            if remaining <= 0:
                continue
            idx = np.flatnonzero(y == c)
            if len(idx) > remaining:
                idx = rng.choice(idx, remaining, replace=False)
            quota[key] -= len(idx)
            if len(idx):
                y_parts.append(y[idx])
                for name, arr in extras.items():
                    extra_parts[name].append(arr[idx])
    if not y_parts:
        raise RuntimeError("no identity evaluation rows")
    y = np.concatenate(y_parts)
    out = {name: np.concatenate(parts) for name, parts in extra_parts.items()}
    keep_parts = []
    for c in range(N11):
        idx = np.flatnonzero(y == c)
        if len(idx) > global_per_class:
            idx = rng.choice(idx, global_per_class, replace=False)
        keep_parts.append(idx)
    keep = np.concatenate(keep_parts)
    return y[keep], {name: arr[keep] for name, arr in out.items()}


def fit_class_offset(logF_all, inner_rows, y_all, seed=0, n_sub=500000):
    rng = np.random.RandomState(seed)
    idx = rng.choice(len(inner_rows), min(n_sub, len(inner_rows)), replace=False)
    sel = inner_rows[idx]
    lf = logF_all[sel]
    yy = y_all[sel]
    b = np.zeros(N11, dtype=np.float64)
    cur = macro_f1_10(confusion11(yy, lf.argmax(1)))
    for _ in range(20):
        imp = False
        for c in range(N11):
            bd, bf = 0.0, cur
            for dl in (-0.5, -0.3, -0.15, -0.07, 0.07, 0.15, 0.3, 0.5):
                b2 = b.copy()
                b2[c] += dl
                f = macro_f1_10(confusion11(yy, (lf + b2).argmax(1)))
                if f > bf:
                    bf = f
                    bd = dl
            if bd:
                b[c] += bd
                cur = bf
                imp = True
        if not imp:
            break
    return b.astype(np.float32)


def shares_from_counts(counts: np.ndarray, basis: str) -> np.ndarray:
    counts = counts.astype(np.float64)
    if basis == "identity11":
        denom = counts.sum(axis=1, keepdims=True)
        out = np.zeros_like(counts)
        np.divide(counts, denom, out=out, where=denom > 0)
        return out
    if basis == "scored10_renormalized":


        sub = counts[:, :10]
        denom = sub.sum(axis=1, keepdims=True)
        out = np.zeros_like(counts)
        np.divide(sub, denom, out=out[:, :10], where=denom > 0)
        return out
    raise ValueError(basis)


def composition_metrics(true_counts, pred_counts, basis, eps):
    ts = shares_from_counts(true_counts, basis)
    ps = shares_from_counts(pred_counts, basis)
    present = true_counts[:, SCORED10] > 0
    lse = np.abs(np.log(ps[:, SCORED10] + eps) - np.log(ts[:, SCORED10] + eps))
    l1 = np.abs(ps[:, SCORED10] - ts[:, SCORED10]).sum(axis=1)
    l1_full = np.abs(ps - ts).sum(axis=1)
    return {
        "true_share": ts,
        "pred_share": ps,
        "present": present,
        "log_share_error": lse,
        "composition_l1_10": l1,
        "composition_l1_11": l1_full,
    }


def patient_equal_summary(m):
    present = m["present"]
    lse = m["log_share_error"]
    per_class, vals = {}, []
    for j, c in enumerate(SCORED10):
        mask = present[:, j]
        if not mask.any():
            per_class[COARSE11[c]] = {"mean_log_share_error": None, "n_patients": 0}
            continue
        v = float(lse[mask, j].mean())
        per_class[COARSE11[c]] = {"mean_log_share_error": v,
                                  "n_patients": int(mask.sum())}
        vals.append(v)
    return {
        "macro_log_share_error_10": float(np.mean(vals)) if vals else None,
        "per_class": per_class,
        "mean_composition_l1_10": float(m["composition_l1_10"].mean()),
        "mean_composition_l1_11": float(m["composition_l1_11"].mean()),
    }


def bootstrap_composition(true_counts, pred_counts, cancers, basis, eps,
                          replicates, seed):
    m = composition_metrics(true_counts, pred_counts, basis, eps)
    lse, present, l1 = m["log_share_error"], m["present"], m["composition_l1_10"]
    by_cancer = defaultdict(list)
    for i, c in enumerate(cancers):
        by_cancer[str(c)].append(i)
    groups = [np.asarray(v, dtype=np.int64) for v in by_cancer.values()]
    rng = np.random.RandomState(seed)
    macro_vals = np.empty(replicates, dtype=np.float64)
    l1_vals = np.empty(replicates, dtype=np.float64)
    for b in range(replicates):
        draw = np.concatenate([g[rng.randint(0, len(g), len(g))] for g in groups])
        pr = present[draw]
        le = lse[draw]
        cls_means = []
        for j in range(len(SCORED10)):
            mask = pr[:, j]
            if mask.any():
                cls_means.append(le[mask, j].mean())
        macro_vals[b] = np.mean(cls_means) if cls_means else np.nan
        l1_vals[b] = l1[draw].mean()
    def summarize(v):
        v = v[np.isfinite(v)]
        return {
            "median": float(np.median(v)),
            "ci95": [float(np.quantile(v, 0.025)), float(np.quantile(v, 0.975))],
            "n_finite_replicates": int(len(v)),
        }
    return {
        "replicates": int(replicates),
        "seed": int(seed),
        "stratification": "patients resampled with replacement within cancer",
        "macro_log_share_error_10": summarize(macro_vals),
        "mean_composition_l1_10": summarize(l1_vals),
    }


class Gate:

    def __init__(self, dump_path):
        self.rows = []
        self.dump_path = Path(dump_path)

    def check(self, name, ok, detail):
        self.rows.append({"assertion": name, "passed": bool(ok), "detail": str(detail)})
        print(f"  [{'PASS' if ok else 'FAIL'}] {name}: {detail}", flush=True)
        if not ok:
            self.dump()
            raise SystemExit(f"ASSERTION FAILED: {name} -- {detail}")
        return True

    def close(self, name, actual, expected, atol):
        return self.check(name, abs(float(actual) - float(expected)) <= atol,
                          f"{actual!r} vs {expected!r} (atol={atol})")

    def note(self, name, detail):
        self.rows.append({"assertion": name, "passed": True, "detail": str(detail)})
        print(f"  [NOTE] {name}: {detail}", flush=True)

    def dump(self):
        self.dump_path.parent.mkdir(parents=True, exist_ok=True)
        write_json(self.dump_path, {"assertions": self.rows})


def write_json(path, obj) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    with os.fdopen(fd, "w", encoding="utf-8") as handle:
        json.dump(obj, handle, indent=2, sort_keys=True, default=str)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(tmp, path)
