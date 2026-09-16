import paths as P
import os
PROJECTS_ROOT = os.environ.get("PROJECTS_ROOT", "/data/heid")
import os
import sys

import numpy as np
import pandas as pd
from scipy.ndimage import binary_closing, binary_dilation, gaussian_filter
from skimage.measure import find_contours
from skimage.morphology import disk

import matplotlib
matplotlib.use("Agg")
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator

REPO_ROOT = str(P.PROJECTS)
BASE = str(P.WORKSPACE)
H5AD_DIR = str(P.CELLS)
WORK_ROOT = str(P.ACCEPTED / "work")
TLS_OUT = str(P.ACCEPTED / "outputs")
OUT_ROOT = str(P.ACCEPTED / "outputs/t_cluster")

ANCHORS = {
    "T":   {"label": "T_cell", "tag": "tagg", "col": "#1f77b4",
            "types": ["T_cell", "NK_T", "CD4_T", "CD8_T", "T_reg", "Treg"]},
    "MAC": {"label": "Macrophage", "tag": "magg", "col": "#d81b60",
            "types": ["Macrophage", "Macrophage_Monocyte", "Monocyte"]},
}
RESOLUTION = 10
TISSUE_SIGMA = 10
TISSUE_THRESH = 0.015
HALO_RADIUS_UM = float(os.environ.get("AGG_HALO_RADIUS_UM", "150"))
CELL_POINT_SIZE = 1.0


def _label_fs(txt: str) -> float:
    return {1: 8.5, 2: 8.5, 3: 6.3}.get(len(txt), 5.3)


def main(dataset, cancer, sample, anchor_key="T"):
    A = ANCHORS[anchor_key]
    tag = A["tag"]; col = A["col"]


    if os.path.exists(f"{TLS_OUT}/{dataset}/{cancer}/{sample}/{sample}_tls.png"):
        print(f"  {sample}: has TLS -> handled by the TLS line; nothing to do here.", flush=True)
        return

    gp = f"{WORK_ROOT}/{dataset}/{cancer}/{sample}/{sample}_{tag}_grid.npz"
    if not os.path.exists(gp):
        print(f"  {sample}: no {tag} grid in work/; skip.", flush=True)
        return
    z = np.load(gp)
    if "grid" not in z.files or z["grid"].ndim != 2 or int(z["grid"].max()) == 0:
        print(f"  {sample}: no {A['label']} aggregate; skip.", flush=True)
        return
    g = z["grid"]
    stat = {int(i): r for i, r in zip(
        z["ids"], zip(z["n_cells"], z["core_area_mm2"], z["density_cells_mm2"], z["cx"], z["cy"]))}

    df = pd.read_parquet(f"{H5AD_DIR}/{sample}.parquet",
                         columns=["cell_id", "cell_type", "x_centroid", "y_centroid"])
    x_min = df.x_centroid.min() - 200; y_min = df.y_centroid.min() - 200
    x_max = df.x_centroid.max() + 200; y_max = df.y_centroid.max() + 200
    nx = int(np.ceil((x_max - x_min) / RESOLUTION)); ny = int(np.ceil((y_max - y_min) / RESOLUTION))
    if g.shape != (ny, nx):
        print(f"  {sample}: grid shape {g.shape} != {(ny, nx)}; skip.", flush=True)
        return
    ix = np.clip(((df.x_centroid.values - x_min) / RESOLUTION).astype(int), 0, nx - 1)
    iy = np.clip(((df.y_centroid.values - y_min) / RESOLUTION).astype(int), 0, ny - 1)
    all_raw = np.zeros((ny, nx), np.float32); np.add.at(all_raw, (iy, ix), 1)
    tissue = binary_closing(gaussian_filter(all_raw, sigma=TISSUE_SIGMA) > TISSUE_THRESH,
                            structure=disk(10))
    halo_px = int(round(HALO_RADIUS_UM / RESOLUTION))

    n = int(g.max())
    items, recs = [], []
    for i in range(1, n + 1):
        m = (g == i)
        if not m.any():
            continue
        rm = binary_dilation(m, structure=disk(halo_px)) & tissue
        nc, ca, dn, cx, cy = stat.get(i, (0, 0.0, 0.0, 0.0, 0.0))
        ys, xs = np.where(rm)
        items.append({"core": m, "region": rm, "cx": float(cx), "cy": float(cy),
                      "max_r_um": float(np.max(np.hypot(xs - xs.mean(), ys - ys.mean())) * RESOLUTION),
                      "label": f"{anchor_key[0]}{len(items)+1}"})


        in_reg = rm[iy, ix]
        n_reg = int(in_reg.sum())
        rec = {"id": len(recs) + 1, "n_cells_core": int(nc), "n_cells_region": n_reg,
               "core_area_mm2": round(float(ca), 4),
               "region_area_mm2": round(float(rm.sum() * (RESOLUTION ** 2) / 1e6), 4),
               "core_density_cells_mm2": round(float(dn), 1),
               "cx": float(cx), "cy": float(cy)}
        if n_reg > 0:
            vc = pd.Series(df.cell_type.values[in_reg]).value_counts()
            for ct, cnt in vc.items():
                rec[f"frac_{ct}"] = round(int(cnt) / n_reg, 4)
        recs.append(rec)
    if not items:
        print(f"  {sample}: no {A['label']} aggregate; skip.", flush=True)
        return

    outdir = f"{OUT_ROOT}/{dataset}/{cancer}/{sample}"
    os.makedirs(outdir, exist_ok=True)
    pd.DataFrame(recs).to_csv(f"{outdir}/{sample}_{tag}.csv", index=False)


    pfx = anchor_key[0]
    agg_reg = np.full(len(df), "Outside", dtype=object)
    agg_core = np.zeros(len(df), dtype=bool)
    for k, it in enumerate(items, 1):
        agg_reg[it["region"][iy, ix]] = f"{pfx}{k}"
        agg_core |= it["core"][iy, ix]
    pd.DataFrame({
        "cell_id": df["cell_id"].values,
        f"{pfx}agg_region": agg_reg,
        f"in_{pfx}agg_core": agg_core,
        "cell_type": df["cell_type"].values,
    }).to_csv(f"{outdir}/{sample}_{tag}_cells.csv", index=False)


    sub = df[df.cell_type.isin(A["types"])]
    gx0, gx1 = float(df.x_centroid.min()), float(df.x_centroid.max())
    gy0, gy1 = float(df.y_centroid.min()), float(df.y_centroid.max())
    ov_w = 16.0
    ov_h = float(np.clip(ov_w * (max(gy1 - gy0, 1.0) / max(gx1 - gx0, 1.0)), 4.0, 16.0))

    _upi = max((gx1 - gx0) / ov_w, (gy1 - gy0) / ov_h)
    cps = float(np.clip((72.0 * 13.5 / _upi) ** 2, CELL_POINT_SIZE, 9.0))
    fig, ax = plt.subplots(figsize=(ov_w, ov_h))
    other = df[~df.cell_type.isin(A["types"])]
    ax.scatter(other.x_centroid, other.y_centroid, c="#d5d5d5", s=cps,
               alpha=0.08, rasterized=True, zorder=0, edgecolors="none")
    overlay = np.zeros((ny, nx, 4), np.float32)
    rgba = list(mcolors.to_rgba(col)); rgba[3] = 0.20
    for it in items:
        overlay[it["region"]] = rgba
    ax.imshow(overlay, extent=[x_min, x_max, y_min, y_max], origin="lower",
              aspect="auto", zorder=2, interpolation="nearest")
    ax.scatter(sub.x_centroid, sub.y_centroid, c=col, s=cps, alpha=0.85,
               rasterized=True, zorder=5, edgecolors="none", label=f"{A['label']} ({len(sub)})")
    for it in items:
        for ctr in find_contours(it["region"].astype(float), 0.5):
            ax.plot(ctr[:, 1] * RESOLUTION + x_min, ctr[:, 0] * RESOLUTION + y_min,
                    color=col, lw=1.2, alpha=0.9, zorder=8)
        for ctr in find_contours(it["core"].astype(float), 0.5):
            ax.plot(ctr[:, 1] * RESOLUTION + x_min, ctr[:, 0] * RESOLUTION + y_min,
                    color=col, lw=0.9, alpha=0.85, zorder=9)
    mx = 0.015 * (gx1 - gx0); my = 0.015 * (gy1 - gy0)
    ax.set_xlim(gx0 - mx, gx1 + mx); ax.set_ylim(gy1 + my, gy0 - my); ax.set_aspect("equal")
    ax.set_xlabel("X (µm)", fontsize=11); ax.set_ylabel("Y (µm)", fontsize=11)
    ax.xaxis.set_major_locator(MultipleLocator(1000)); ax.yaxis.set_major_locator(MultipleLocator(1000))

    cxs = np.array([it["cx"] for it in items], float); cys = np.array([it["cy"] for it in items], float)
    rr = np.array([max(it["max_r_um"], 1.0) for it in items], float)
    pw, ph = (gx1 - gx0), (gy1 - gy0); diag = float(np.hypot(pw, ph))
    sep = 0.050 * diag; pad = 0.030 * diag
    dirx, diry = cxs - cxs.mean(), cys - cys.mean()


    deg = np.hypot(dirx, diry) < 1e-6
    dirx = np.where(deg, 1.0, dirx); diry = np.where(deg, -1.0, diry)
    dn_ = np.hypot(dirx, diry); dn_[dn_ < 1e-6] = 1.0
    lx = cxs + dirx / dn_ * (rr + pad); ly = cys + diry / dn_ * (rr + pad)
    for _ in range(400):
        dx = lx[:, None] - lx[None, :]; dy = ly[:, None] - ly[None, :]
        dist = np.hypot(dx, dy); np.fill_diagonal(dist, 1e9)
        with np.errstate(invalid="ignore", divide="ignore"):
            f = np.where(dist < sep, (sep - dist) / np.maximum(dist, 1e-6) * 0.5, 0.0)
        mvx = (dx * f).sum(1); mvy = (dy * f).sum(1)
        ex = lx[:, None] - cxs[None, :]; ey = ly[:, None] - cys[None, :]
        ed = np.hypot(ex, ey); need = (rr[None, :] + pad) - ed
        with np.errstate(invalid="ignore", divide="ignore"):
            gg = np.where(need > 0, need / np.maximum(ed, 1e-6), 0.0)
        mvx += (ex * gg).sum(1); mvy += (ey * gg).sum(1)
        lx += np.clip(mvx, -sep, sep) * 0.6; ly += np.clip(mvy, -sep, sep) * 0.6
        lx = np.clip(lx, gx0 - 0.10 * pw, gx1 + 0.10 * pw)
        ly = np.clip(ly, gy0 - 0.10 * ph, gy1 + 0.10 * ph)
    for i, it in enumerate(items):
        ax.plot([lx[i], it["cx"]], [ly[i], it["cy"]], color=col, lw=0.8, alpha=0.65,
                zorder=11, clip_on=False)
        ax.scatter([lx[i]], [ly[i]], s=210, marker="o", facecolor="white", edgecolor=col,
                   linewidth=1.8, alpha=0.98, zorder=12, clip_on=False)
        ax.text(lx[i], ly[i], it["label"], ha="center", va="center",
                fontsize=_label_fs(it["label"]),
                fontweight="bold", color=col, zorder=13, clip_on=False)
    ax.set_title(f"{sample} ({cancer}, {dataset}): NO TLS — {len(items)} independent "
                 f"{A['label']} aggregates", fontsize=12, fontweight="bold")
    ax.legend(loc="upper right", fontsize=9, markerscale=6)
    plt.tight_layout()
    fig.savefig(f"{outdir}/{sample}_{tag}.png", dpi=200, bbox_inches="tight")
    fig.savefig(f"{outdir}/{sample}_{tag}.pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"  {sample}: NO TLS, {len(items)} {A['label']} aggregates -> {outdir}", flush=True)


if __name__ == "__main__":
    if len(sys.argv) < 4:
        sys.exit("usage: agg_only.py <dataset> <cancer_type> <sample> [T|MAC]")
    main(sys.argv[1], sys.argv[2], sys.argv[3], (sys.argv[4] if len(sys.argv) > 4 else "T").upper())
