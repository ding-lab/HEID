#!/usr/bin/env python3
import argparse
import hashlib
import json
import os
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.metrics import roc_auc_score
import molecular_inputs as ENG
F128 = LAB = GROUP_MANIFEST = None
GC = ["BCL6", "AICDA", "MKI67"]
MIN_TILES = 5
C_LOGREG = 0.05
RIDGE_ALPHA = 10.0
SEED = 0
UM = 0.2125

def precompute(df):
    man = pd.read_parquet(GROUP_MANIFEST)
    cvmap = man.drop_duplicates("sample").set_index("sample")["cv_group"].to_dict()
    cancermap = df.drop_duplicates("sample").set_index("sample")["cancer"].to_dict()
    rows = []; feats = []; diag = []
    for s in sorted(df["sample"].unique()):
        cj = ENG.load_sample(s)
        npz = LAB / s / f"{s}_tls_region_grid.npz"
        fp = F128 / s / "uni2.pt"
        if cj is None or not npz.exists() or not fp.exists():
            continue
        d = torch.load(fp, map_location="cpu", weights_only=False)
        F = np.asarray(d["features"], np.float32); reg = np.asarray(d["region"]).astype(int)
        g = np.load(str(npz)); tg = g["tls_grid"].astype(np.int32)
        nxmin = float(g["x_min"]); nymin = float(g["y_min"]); res = float(g["res_um"]); ny, nx = tg.shape

        hx = cj.hx.values; hy = cj.hy.values
        gx = ((hx * UM - nxmin) / res).astype(int); gy = ((hy * UM - nymin) / res).astype(int)
        inb = (gx >= 0) & (gx < nx) & (gy >= 0) & (gy < ny)
        creg = np.zeros(len(cj), int)
        creg[inb] = tg[gy[inb], gx[inb]]
        isB = (cj.cell_type.values == "B_cell")
        zc = {gname: (cj[gname + "_z"].values if (gname + "_z") in cj.columns else np.zeros(len(cj))) for gname in GC}
        n_membership = cj.tls_region.nunique()
        regions_with_tiles = sorted(set(int(r) for r in np.unique(reg) if r > 0))
        for r in regions_with_tiles:
            tir = np.where(reg == r)[0]
            if len(tir) < MIN_TILES:
                continue
            cell_in = (creg == r)
            b_in = cell_in & isB
            if b_in.sum() == 0:
                continue
            gc_score = float(sum(zc[gname][b_in].mean() for gname in GC))
            pooled = np.concatenate([F[tir].mean(0), F[tir].max(0)])
            rows.append(dict(tls_key=f"{s}::grid{r}", sample=s, region=r, cancer=cancermap.get(s),
                             cv_group=cvmap.get(s, s), gc_score=gc_score,
                             n_tiles=int(len(tir)), n_bcells=int(b_in.sum())))
            feats.append(pooled)
        diag.append(dict(sample=s, cancer=cancermap.get(s), n_membership_TLS=int(n_membership),
                         n_grid_regions_total=int((tg > 0).any() and len(np.unique(tg[tg > 0]))),
                         n_regions_with_tiles=int(len(regions_with_tiles)),
                         n_kept=int(sum(1 for r in regions_with_tiles if (reg == r).sum() >= MIN_TILES))))
    return pd.DataFrame(rows), np.array(feats, np.float32), pd.DataFrame(diag)

def tertile_label(score):
    thr = float(np.quantile(score, 2 / 3)); return (score >= thr).astype(int), thr

def lopo_oof(X, y, groups):
    oof = np.full(len(y), np.nan)
    for tr, te in LeaveOneGroupOut().split(X, y, groups):
        if len(np.unique(y[tr])) < 2:
            continue
        sc = StandardScaler().fit(X[tr])
        m = LogisticRegression(max_iter=2000, C=C_LOGREG, class_weight="balanced").fit(sc.transform(X[tr]), y[tr])
        oof[te] = m.predict_proba(sc.transform(X[te]))[:, 1]
    return oof

def per_cancer(y, s, cancer):
    per = {}; num = den = 0.0
    for c in np.unique(cancer):
        sel = cancer == c; yy, ss = y[sel], s[sel]; ok = ~np.isnan(ss); yy, ss = yy[ok], ss[ok]
        if len(np.unique(yy)) < 2:
            continue
        a = float(roc_auc_score(yy, ss)); w = yy.sum() * (1 - yy).sum()
        per[c] = round(a, 3); num += a * w; den += w
    return (round(num / den, 3) if den else None), per

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    config = json.loads(os.path.expandvars(args.config.read_text()))
    unresolved = [k for k,v in config.items() if isinstance(v,str) and "${" in v]
    if unresolved:
        raise ValueError(f"Unresolved environment variables in {unresolved}")
    global F128, LAB, GROUP_MANIFEST
    ENG.configure(config)
    F128 = ENG.F128
    LAB = ENG.LAB
    GROUP_MANIFEST = Path(config["groups"])
    np.random.seed(SEED)
    roster = pd.read_parquet(config["samples"])
    metadata, features, diagnostics = precompute(roster)
    if metadata.empty:
        raise ValueError("No supported regions in the configured training inputs")
    target, threshold = tertile_label(metadata.gc_score.values)
    oof = lopo_oof(features, target, metadata.cv_group.values)
    within, per = per_cancer(target, oof, metadata.cancer.values)
    ok = np.isfinite(oof)
    scaler = StandardScaler().fit(features)
    scaled = scaler.transform(features)
    classifier = LogisticRegression(max_iter=2000, C=C_LOGREG, class_weight="balanced").fit(scaled, target)
    ridge = Ridge(alpha=RIDGE_ALPHA).fit(scaled, metadata.gc_score.values.astype(float))
    args.output.mkdir(parents=True, exist_ok=True)
    np.savez(args.output / "maturation_head.npz", scaler_mean=scaler.mean_, scaler_scale=scaler.scale_,
             clf_coef=classifier.coef_, clf_intercept=classifier.intercept_, clf_classes=classifier.classes_,
             ridge_coef=ridge.coef_, ridge_intercept=np.array([ridge.intercept_]),
             tertile_threshold=np.array([threshold]), feature_order=np.array(["mean1536||max1536"]))
    metadata = metadata.assign(label=target, oof_probability=oof)
    metadata.to_parquet(args.output / "region_evaluation.parquet", index=False)
    diagnostics.to_csv(args.output / "input_diagnostics.csv", index=False)
    result = dict(n_regions=len(metadata), n_positive=int(target.sum()), n_samples=int(metadata['sample'].nunique()),
                  n_groups=int(metadata.cv_group.nunique()), within_cancer_pair_weighted_AUROC=within,
                  per_cancer_AUROC=per, pooled_AUROC=float(roc_auc_score(target[ok],oof[ok])),
                  n_missing_oof=int((~ok).sum()), tertile_threshold=float(threshold),
                  validation_unit="recorded cv_group; adjudicate patient identity independently",
                  target_policy="complete-cohort upper tertile before group evaluation",
                  inputs=config, configuration_sha256=hashlib.sha256(args.config.read_bytes()).hexdigest())
    (args.output / "results.json").write_text(json.dumps(result,indent=2)+"\n")
    print(json.dumps({k:v for k,v in result.items() if k!='inputs'},indent=2))

if __name__ == "__main__":
    main()
