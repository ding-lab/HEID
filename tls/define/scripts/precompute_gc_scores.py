import paths as P
import h5py, numpy as np, pandas as pd, os, sys
from scipy.sparse import csr_matrix

H5 = str(P.FULL_EXPRESSION)
ROOT = str(P.WORKSPACE)
IDX = str(P.ROW_INDEX)
GC_GENES = ["AICDA", "BCL6", "MKI67"]


GC_WEIGHTS = [float(x) for x in os.environ.get("TLS_GC_WEIGHTS", "1_1_1").replace(",", " ").replace("_", " ").split()]
assert len(GC_WEIGHTS) == len(GC_GENES), f"need {len(GC_GENES)} weights, got {GC_WEIGHTS}"
OUTDIR = str(P.GC)


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


def zsum_score(X, gene_idx):
    sub = np.asarray(X[:, gene_idx].todense(), dtype=np.float64)
    mu = sub.mean(0); sd = sub.std(0); sd[sd == 0] = 1.0
    return ((sub - mu) / sd).mean(1)


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
        present = [g for g in GC_GENES if g in gidx]
        Xs, bcs, cts = [], [], []
        for (r0, r1) in blocks:
            X, bc, ct = read_block(f, r0, r1, n_genes)
            Xs.append(X); bcs += bc; cts.append(ct)
    from scipy.sparse import vstack
    X = vstack(Xs).tocsr() if len(Xs) > 1 else Xs[0]
    ct = np.concatenate(cts)
    n = X.shape[0]
    method = "ulm"
    score = np.full(n, np.nan)
    if len(present) < 2:
        method = "zsum_lt2genes"
        score = zsum_score(X, [gidx[g] for g in present]) if present else np.zeros(n)
    else:
        import anndata as ad, decoupler as dc
        adata = ad.AnnData(X=X.tocsr())
        adata.var_names = genes
        w = [GC_WEIGHTS[GC_GENES.index(g)] for g in present]
        net = pd.DataFrame({"source": "GC", "target": present, "weight": w})
        net = net[net["weight"] != 0]

        dc.mt.ulm(adata, net, tmin=max(1, len(net)), verbose=False)
        obsm = adata.obsm
        key = "score_ulm" if "score_ulm" in obsm else [k for k in obsm if "ulm" in k.lower()][0]
        sdf = obsm[key]
        score = (sdf["GC"].values if hasattr(sdf, "columns") else np.asarray(sdf).ravel()).astype(np.float64)

        if not np.isfinite(score).any() or np.nanstd(score) == 0:
            raise ValueError(f"{sample}: degenerate ULM GC score")

    is_b = (ct == "B_cell")
    gc = np.where(is_b, score, np.nan)
    out = pd.DataFrame({"cell_id": bcs, "gc_score": gc.astype(np.float32)})
    out["scoring_method"] = method
    outp = f"{OUTDIR}/{sample}.parquet"
    out.to_parquet(outp, index=False)
    nb = int(is_b.sum()); nfin = int(np.isfinite(gc).sum())
    bsc = gc[np.isfinite(gc)]
    print(f"{sample}: cells={n} B={nb} gc_score_finite={nfin} method={method} "
          f"genes_present={present} | B-score mean={bsc.mean():.3f} std={bsc.std():.3f} "
          f"min={bsc.min():.3f} max={bsc.max():.3f}" if bsc.size else
          f"{sample}: cells={n} B={nb} NO finite B scores method={method}", flush=True)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Per-cell GC program score (decoupler ULM) for one section.")
    parser.add_argument("sample")
    main(parser.parse_args().sample)
