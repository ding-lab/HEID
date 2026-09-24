import paths as P
import h5py, numpy as np, pandas as pd, os, sys
from scipy.sparse import csr_matrix

H5 = str(P.FULL_EXPRESSION)
ROOT = str(P.WORKSPACE)
IDX = str(P.ROW_INDEX)
OUTDIR = str(P.PROGRAMS)


GENESETS = {
    "B":  ["BCL6", "MKI67", "AICDA", "CD38", "FAS"],
    "T":  ["CXCL13", "CXCR5", "PDCD1"],
    "DC": ["CR2", "FCER2", "CXCL13"],
}


def _dec(arr):
    return [x.decode() if isinstance(x, bytes) else str(x) for x in arr]


def _cat_codes(grp):
    return _dec(grp["categories"][:]), grp["codes"][:]


def read_block(f, r0, r1, n_genes):
    ip = f["X"]["indptr"]
    s, e = int(ip[r0]), int(ip[r1])
    cols = f["X"]["indices"][s:e]
    vals = f["X"]["data"][s:e]
    local_ip = (ip[r0:r1 + 1] - s).astype(np.int64)
    X = csr_matrix((vals, cols, local_ip), shape=(r1 - r0, n_genes))
    bc = _dec(f["obs"]["barcode"][r0:r1])
    ctcats, ctcodes = _cat_codes(f["obs"]["cell_type"])
    ct = np.array(ctcats, dtype=object)[ctcodes[r0:r1]]
    return X, bc, ct


def main(sample):
    idx = pd.read_csv(IDX, sep="\t")
    rows = idx[idx["sample"] == sample]
    if rows.empty:
        sys.exit(f"{sample} not in row index")
    os.makedirs(OUTDIR, exist_ok=True)
    blocks = list(zip(rows["row_start"].astype(int), rows["row_end"].astype(int)))
    with h5py.File(H5, "r") as f:
        genes = _dec(f["var"][dict(f["var"].attrs).get("_index", "_index")][:])
        n_genes = len(genes)
        gidx = {g: i for i, g in enumerate(genes)}
        present = {grp: [g for g in gs if g in gidx] for grp, gs in GENESETS.items()}
        Xs, bcs, cts = [], [], []
        for (r0, r1) in blocks:
            X, bc, ct = read_block(f, r0, r1, n_genes)
            Xs.append(X); bcs += bc; cts.append(ct)
    from scipy.sparse import vstack
    X = vstack(Xs).tocsr() if len(Xs) > 1 else Xs[0]
    ct = np.concatenate(cts)
    n = X.shape[0]


    method = "ulm_pertype"
    src_scores = {}
    import anndata as ad, decoupler as dc
    adata = ad.AnnData(X=X.tocsr()); adata.var_names = genes
    net = pd.concat([pd.DataFrame({"source": grp, "target": present[grp], "weight": 1.0})
                     for grp in GENESETS if present[grp]], ignore_index=True)
    tmin = min(len(present[grp]) for grp in GENESETS if present[grp])
    dc.mt.ulm(adata, net, tmin=max(1, tmin), verbose=False)
    key = "score_ulm" if "score_ulm" in adata.obsm else [k for k in adata.obsm if "ulm" in k.lower()][0]
    sdf = adata.obsm[key]
    for grp in GENESETS:
        if present[grp] and hasattr(sdf, "columns") and grp in sdf.columns:
            src_scores[grp] = np.asarray(sdf[grp].values, dtype=np.float64)
    if not src_scores:
        raise ValueError(f"{sample}: no ULM source columns")


    out = pd.DataFrame({"cell_id": bcs, "cell_type": ct})
    for grp in GENESETS:
        out[f"score_{grp}"] = (src_scores[grp].astype(np.float32) if grp in src_scores
                               else np.full(n, np.nan, np.float32))
    out["scoring_method"] = method
    out.to_parquet(f"{OUTDIR}/{sample}.parquet", index=False)
    fins = " ".join(f"{g}:fin={int(np.isfinite(out['score_'+g]).sum())}" for g in GENESETS)
    print(f"{sample}: cells={n} method={method} present={ {g: present[g] for g in GENESETS} } {fins}",
          flush=True)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Per-cell B, T and DC program scores (decoupler ULM) for one section.")
    parser.add_argument("sample")
    main(parser.parse_args().sample)
