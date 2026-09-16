import paths as P
import os
import sys

import numpy as np
import pandas as pd
from scipy.ndimage import (binary_closing, binary_dilation, binary_fill_holes,
                           binary_opening, gaussian_filter, label)
from skimage.feature import peak_local_max
from skimage.measure import find_contours
from skimage.morphology import disk
from skimage.segmentation import watershed

import matplotlib
matplotlib.use("Agg")
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import MultipleLocator

REPO_ROOT = str(P.PROJECTS)
BASE = str(P.WORKSPACE)
SRC = str(P.ACCEPTED)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import tls_maturity as M

H5AD_DIR = str(P.CELLS)
OUT_ROOT = str(P.REPORT)

RESOLUTION = M.RESOLUTION
HALOS_UM = [50.0, 100.0, 150.0]
DRAW_HALO_UM = 150.0
CELL_POINT_SIZE = 1.0

COLORS = {"B_cell": "#8e44ad", "T_cell": "#3498db", "Plasma": "#e67e22",
          "Lymphatic_Endothelial": "#27ae60", "Other": "#d5d5d5"}
CELL_ALPHA = {"T_cell": 0.7, "Plasma": 0.8, "B_cell": 0.9, "Lymphatic_Endothelial": 0.95}
CELL_Z = {"Plasma": 3, "T_cell": 4, "B_cell": 5, "Lymphatic_Endothelial": 6}
CELL_LABEL = {"B_cell": "B_cell", "T_cell": "T_cell+NK_T", "Plasma": "Plasma",
              "Lymphatic_Endothelial": "Lymph. Endothelial"}
TLS_COL = "#d62728"
AGG_COL = {"tagg": "#1f77b4", "magg": "#d81b60"}


def build_cores(ny, nx, tissue, b_ix, b_iy):
    b_raw = np.zeros((ny, nx), np.float32)
    np.add.at(b_raw, (b_iy, b_ix), 1)
    b_density = gaussian_filter(b_raw, sigma=M.SIGMA_PX)
    lb_b, n_b_cc = label(M.b_density_mask(b_raw) & tissue)

    if n_b_cc > 0:
        sep_px = max(int(M.TLS_PEAK_MIN_SEP_UM / RESOLUTION), 3)
        new_lb = np.zeros_like(lb_b, dtype=np.int32)
        nxt = 1
        for cid in range(1, n_b_cc + 1):
            comp = lb_b == cid
            peaks = peak_local_max(b_density, min_distance=sep_px, labels=comp,
                                   exclude_border=False)
            if len(peaks) <= 1:
                new_lb[comp] = nxt; nxt += 1
                continue
            markers = np.zeros_like(comp, dtype=np.int32)
            for i, (yy, xx) in enumerate(peaks, 1):
                markers[yy, xx] = i
            ws = M.merge_shallow_basins(watershed(-b_density, markers, mask=comp),
                                        b_density, M.MERGE_SADDLE_FRAC)
            for j in range(1, ws.max() + 1):
                sm = ws == j
                if sm.any():
                    new_lb[sm] = nxt; nxt += 1
        lb_b, n_b_cc = new_lb, nxt - 1

    cores = []
    for cid in range(1, n_b_cc + 1):
        comp = lb_b == cid
        in_comp = comp[b_iy, b_ix]
        if not in_comp.any():
            continue
        is_big = int(comp.sum()) * (RESOLUTION ** 2) > M.BIG_COMP_AREA_UM2
        raw_c = np.zeros((ny, nx), dtype=np.float32)
        np.add.at(raw_c, (b_iy[in_comp], b_ix[in_comp]), 1)
        dens_c = gaussian_filter(raw_c, sigma=M.CORE_SIGMA_BIG_PX if is_big else M.CORE_SIGMA_SMALL_PX)
        if dens_c.max() <= 0:
            continue
        thr = M.CORE_THRESH_REL * dens_c.max()
        tight = dens_c > thr
        tight &= comp
        if M.CORE_CLOSING_PX > 0:
            tight = binary_closing(tight, structure=disk(M.CORE_CLOSING_PX))
        if M.CORE_OPENING_PX > 0:
            tight = binary_opening(tight, structure=disk(M.CORE_OPENING_PX))
        tight &= comp
        lb_p, n_p = label(tight)
        for pid in range(1, n_p + 1):
            for cc in M.clean_core(lb_p == pid, dens_c, thr):
                cores.append((cc, is_big))

    split_max_px = M.SPLIT_MAX_CORE_AREA_UM2 / (RESOLUTION ** 2)
    if cores:
        resplit = []
        for core, is_big in cores:
            if (not is_big) and int(core.sum()) > split_max_px:
                resplit.extend((p, is_big) for p in
                               M.split_oversized_core(core, b_density, b_iy, b_ix))
            else:
                resplit.append((core, is_big))
        cores = resplit
    return [c for c, _ in cores], b_density


def tls_region(core, tissue, ny, nx, halo_px):
    region = binary_dilation(core, structure=disk(halo_px)) & tissue
    props, _ = M.region_shape_stats(region, core)
    if props is None:
        return region
    M_full, m_full = props.major_axis_length, props.minor_axis_length
    ar = (m_full / M_full) if M_full > 0 else 1.0
    peri = props.perimeter
    circ = (4 * np.pi * props.area / (peri ** 2)) if peri > 0 else 0.0
    if (ar < M.MIN_REGION_AXIS_RATIO or circ < M.MIN_REGION_CIRCULARITY) and M_full > 0:
        ys_r, xs_r = np.where(region)
        if len(ys_r) >= 5:
            pts = np.column_stack([ys_r, xs_r]).astype(np.float64)
            ctr = pts.mean(axis=0)
            pts_c = pts - ctr
            cov = (pts_c.T @ pts_c) / len(pts_c)
            eigvals, eigvecs = np.linalg.eigh(cov)
            major_vec = eigvecs[:, int(np.argmax(eigvals))]
            proj = pts_c @ major_vec
            cyi, cxi = np.where(core)
            core_center = (float(np.median((np.column_stack([cyi, cxi]).astype(np.float64) - ctr)
                                           @ major_vec)) if len(cyi) else 0.0)
            half = min((proj.max() - proj.min()) / 2, m_full / M.MIN_REGION_AXIS_RATIO / 2)
            keep = np.abs(proj - core_center) <= half
            cand = np.zeros((ny, nx), dtype=bool)
            cand[ys_r[keep], xs_r[keep]] = True
            cand &= tissue
            if (cand & core).any():
                region = cand | (binary_dilation(core, structure=disk(halo_px)) & tissue)
    return region


def halo_cols(prefix, masks_by_halo, iy, ix, cell_type, area_of):
    out = {}
    for h, m in masks_by_halo.items():
        sel = m[iy, ix]
        n = int(sel.sum())
        tag = f"_halo{int(h)}"
        out[f"{prefix}n_cells{tag}"] = n
        out[f"{prefix}region_area_mm2{tag}"] = round(area_of(m), 4)
        if n:
            for ct, cnt in pd.Series(cell_type[sel]).value_counts().items():
                out[f"frac_{ct}{tag}"] = round(int(cnt) / n, 4)
    return out


def main(dataset, cancer, sample):
    sdir = f"{SRC}/outputs/{dataset}/{cancer}/{sample}"
    tls_csv = f"{sdir}/{sample}_tls.csv"
    has_tls = os.path.exists(tls_csv)
    if not has_tls:
        sdir = f"{SRC}/outputs/t_cluster/{dataset}/{cancer}/{sample}"
        if not os.path.isdir(sdir):
            print(f"  {sample}: nothing published; skip.", flush=True)
            return

    df = pd.read_parquet(f"{H5AD_DIR}/{sample}.parquet",
                         columns=["cell_id", "cell_type", "x_centroid", "y_centroid"])
    df["cell_id"] = df["cell_id"].astype(str)
    cls = np.full(len(df), "Other", dtype=object)
    cls[df.cell_type.isin(M.B_TYPES).values] = "B_cell"
    cls[df.cell_type.isin(M.T_TYPES).values] = "T_cell"
    cls[df.cell_type.isin(M.P_TYPES).values] = "Plasma"
    cls[df.cell_type.isin(["Lymphatic_Endothelial"]).values] = "Lymphatic_Endothelial"
    df["class"] = cls
    cell_type = df["cell_type"].values

    x_min = df.x_centroid.min() - 200; y_min = df.y_centroid.min() - 200
    x_max = df.x_centroid.max() + 200; y_max = df.y_centroid.max() + 200
    nx = int(np.ceil((x_max - x_min) / RESOLUTION)); ny = int(np.ceil((y_max - y_min) / RESOLUTION))
    ix = np.clip(((df.x_centroid.values - x_min) / RESOLUTION).astype(int), 0, nx - 1)
    iy = np.clip(((df.y_centroid.values - y_min) / RESOLUTION).astype(int), 0, ny - 1)
    all_raw = np.zeros((ny, nx), np.float32); np.add.at(all_raw, (iy, ix), 1)
    tissue = binary_fill_holes(binary_closing(
        gaussian_filter(all_raw, sigma=M.TISSUE_SIGMA) > M.TISSUE_THRESH, structure=disk(10)))
    area_of = lambda m: float(int(m.sum()) * (RESOLUTION ** 2) / 1e6)

    outdir = (f"{OUT_ROOT}/tls/{cancer}/{sample}" if has_tls
              else f"{OUT_ROOT}/t_cluster/{dataset}/{cancer}/{sample}")
    os.makedirs(outdir, exist_ok=True)


    tls_items = []
    if has_tls:
        bsel = (df["class"] == "B_cell").values
        cores, b_density = build_cores(ny, nx, tissue, ix[bsel], iy[bsel])
        pub = pd.read_csv(tls_csv, usecols=["cell_id", "tls_region", "in_tls_core"],
                          dtype=str, keep_default_na=False)
        pub["cell_id"] = pub["cell_id"].astype(str)
        lab = df[["cell_id"]].merge(pub, on="cell_id", how="left")
        pub_reg = lab["tls_region"].fillna("Outside").values
        is_core_cell = lab["in_tls_core"].isin(["True", "true", "1"]).values
        for core in cores:
            inside = core[iy, ix]
            if inside.any() and is_core_cell[inside].mean() >= 0.5:
                tls_items.append(core)


        meta_p = f"{sdir}/{sample}_tls_meta.json"
        if tls_items and os.path.exists(meta_p):
            import json
            pub_tls = [r for r in (json.loads(open(meta_p).read()).get("tls") or [])
                       if "cx" in r and "cy" in r]
            if len(pub_tls) == len(tls_items):
                from scipy.optimize import linear_sum_assignment
                peaks, areas = [], []
                for core in tls_items:
                    masked = np.where(core, b_density, -np.inf)
                    py, px = np.unravel_index(int(np.argmax(masked)), masked.shape)
                    peaks.append((px * RESOLUTION + x_min, py * RESOLUTION + y_min))
                    areas.append(round(area_of(core), 4))


                cost = np.zeros((len(tls_items), len(pub_tls)))
                for a, (pk, ar) in enumerate(zip(peaks, areas)):
                    for b, r in enumerate(pub_tls):
                        da = abs(ar - float(r.get("core_area_mm2", -1)))
                        dd = np.hypot(r["cx"] - pk[0], r["cy"] - pk[1])
                        cost[a, b] = da * 1e6 + dd
                rows, cols = linear_sum_assignment(cost)
                order = sorted(zip([pub_tls[b]["id"] for b in cols], rows))
                tls_items = [tls_items[k] for _, k in order]

    if tls_items:
        regions = {h: [tls_region(c, tissue, ny, nx, int(round(h / RESOLUTION)))
                       for c in tls_items] for h in HALOS_UM}


        owned = {}
        for h in HALOS_UM:
            claimed = np.zeros((ny, nx), bool)
            per = []
            for reg in regions[h]:
                own = reg & ~claimed
                claimed |= own
                per.append(own)
            owned[h] = per


        percell = {"cell_id": df["cell_id"].values}
        core_flag = np.zeros(len(df), bool)
        for k, core in enumerate(tls_items, 1):
            core_flag |= core[iy, ix]
        for h in HALOS_UM:
            lab_h = np.full(len(df), "Outside", dtype=object)
            for k, own in enumerate(owned[h], 1):
                lab_h[own[iy, ix]] = f"TLS-{k}"
            percell[f"tls_region_halo{int(h)}"] = lab_h
        percell["in_tls_core"] = core_flag
        percell["cell_type"] = cell_type
        pd.DataFrame(percell).to_csv(f"{outdir}/{sample}_tls.csv", index=False)

        rows = []
        for k, core in enumerate(tls_items, 1):
            ys, xs = np.where(core)
            row = {"sample": sample, "tissue": cancer, "tls_region": f"TLS-{k}",
                   "n_b_core": int(core[iy[(df['class'] == 'B_cell').values],
                                        ix[(df['class'] == 'B_cell').values]].sum()),
                   "core_area_mm2": round(area_of(core), 4),
                   "cx": float(xs.mean() * RESOLUTION + x_min),
                   "cy": float(ys.mean() * RESOLUTION + y_min)}
            row.update(halo_cols("", {h: owned[h][k - 1] for h in HALOS_UM},
                                 iy, ix, cell_type, area_of))
            rows.append(row)
        pd.DataFrame(rows).to_csv(f"{outdir}/{sample}_tls_composition.csv", index=False)


    agg_items = {"tagg": [], "magg": []}
    for tag in ("tagg", "magg"):
        gp = f"{SRC}/work/{dataset}/{cancer}/{sample}/{sample}_{tag}_grid.npz"
        kept_csv = f"{sdir}/{sample}_{tag}.csv"
        if not (os.path.exists(gp) and os.path.exists(kept_csv)):
            continue
        z = np.load(gp)
        if "grid" not in z.files or z["grid"].ndim != 2 or z["grid"].shape != (ny, nx):
            continue
        g = z["grid"]
        k = pd.read_csv(kept_csv)
        ids = (k["id_before_tls_dedup"] if "id_before_tls_dedup" in k.columns else k["id"]).tolist()
        stat = {int(i): r for i, r in zip(z["ids"], zip(z["n_cells"], z["core_area_mm2"],
                                                       z["density_cells_mm2"], z["cx"], z["cy"]))} \
            if "ids" in z.files else {}
        for gid in ids:
            m = (g == int(gid))
            if m.any():
                agg_items[tag].append((int(gid), m))

        if not agg_items[tag]:
            continue
        pfx = "T" if tag == "tagg" else "M"
        regs = {h: [binary_dilation(m, structure=disk(int(round(h / RESOLUTION)))) & tissue
                    for _, m in agg_items[tag]] for h in HALOS_UM}
        rows = []
        for j, (gid, m) in enumerate(agg_items[tag], 1):
            nc, ca, dn, cx, cy = stat.get(gid, (0, 0.0, 0.0, 0.0, 0.0))
            row = {"id": j, "id_before_tls_dedup": gid,
                   "n_cells_core": int(nc), "core_area_mm2": round(float(ca), 4),
                   "core_density_cells_mm2": round(float(dn), 1),
                   "cx": float(cx), "cy": float(cy)}
            row.update(halo_cols("", {h: regs[h][j - 1] for h in HALOS_UM},
                                 iy, ix, cell_type, area_of))
            rows.append(row)
        pd.DataFrame(rows).to_csv(f"{outdir}/{sample}_{tag}.csv", index=False)

        percell = {"cell_id": df["cell_id"].values}
        core_flag = np.zeros(len(df), bool)
        for _, m in agg_items[tag]:
            core_flag |= m[iy, ix]
        for h in HALOS_UM:
            lab_h = np.full(len(df), "Outside", dtype=object)
            taken = np.zeros(len(df), bool)
            for j, reg in enumerate(regs[h], 1):
                sel = reg[iy, ix] & ~taken
                lab_h[sel] = f"{pfx}{j}"
                taken |= sel
            percell[f"{pfx}agg_region_halo{int(h)}"] = lab_h
        percell[f"in_{pfx}agg_core"] = core_flag
        percell["cell_type"] = cell_type
        pd.DataFrame(percell).to_csv(f"{outdir}/{sample}_{tag}_cells.csv", index=False)


    draw_px = int(round(DRAW_HALO_UM / RESOLUTION))
    draw_tls = [{"core": c, "region": tls_region(c, tissue, ny, nx, draw_px)} for c in tls_items]
    draw_agg = {t: [{"core": m,
                     "region": binary_dilation(m, structure=disk(draw_px)) & tissue}
                    for _, m in agg_items[t]] for t in ("tagg", "magg")}

    gx0, gx1 = float(df.x_centroid.min()), float(df.x_centroid.max())
    gy0, gy1 = float(df.y_centroid.min()), float(df.y_centroid.max())
    ov_w = 16.0
    ov_h = float(np.clip(ov_w * (max(gy1 - gy0, 1.0) / max(gx1 - gx0, 1.0)), 4.0, 16.0))
    _upi = max((gx1 - gx0) / ov_w, (gy1 - gy0) / ov_h)
    cps = float(np.clip((72.0 * 13.5 / _upi) ** 2, CELL_POINT_SIZE, 9.0))
    fig, ax = plt.subplots(figsize=(ov_w, ov_h))

    other = df[df["class"] == "Other"]
    if len(other):
        ax.scatter(other.x_centroid, other.y_centroid, c=COLORS["Other"], s=cps,
                   alpha=0.08, rasterized=True, zorder=0, edgecolors="none")

    overlay = np.zeros((ny, nx, 4), np.float32)
    rgba = list(mcolors.to_rgba(TLS_COL)); rgba[3] = 0.20
    for it in draw_tls:
        overlay[it["core"]] = rgba
    for t in ("tagg", "magg"):
        rgba = list(mcolors.to_rgba(AGG_COL[t])); rgba[3] = 0.20
        for it in draw_agg[t]:
            overlay[it["core"]] = rgba
    ax.imshow(overlay, extent=[x_min, x_max, y_min, y_max], origin="lower",
              aspect="auto", zorder=2, interpolation="nearest")
    for c in ("T_cell", "Plasma", "B_cell", "Lymphatic_Endothelial"):
        sub = df[df["class"] == c]
        if len(sub):
            ax.scatter(sub.x_centroid, sub.y_centroid, c=COLORS[c], s=cps,
                       alpha=CELL_ALPHA[c], rasterized=True, zorder=CELL_Z[c], edgecolors="none")

    def outline(mask, col, lw, z, alpha):
        for ctr in find_contours(mask.astype(float), 0.5):
            ax.plot(ctr[:, 1] * RESOLUTION + x_min, ctr[:, 0] * RESOLUTION + y_min,
                    color=col, lw=lw, alpha=alpha, linestyle="-", zorder=z)

    for it in draw_tls:
        outline(it["core"], TLS_COL, 1.2, 9, 0.9)
    for t in ("tagg", "magg"):
        for it in draw_agg[t]:
            outline(it["core"], AGG_COL[t], 1.2, 9, 0.9)

    mx = 0.015 * (gx1 - gx0); my = 0.015 * (gy1 - gy0)
    ax.set_xlim(gx0 - mx, gx1 + mx); ax.set_ylim(gy1 + my, gy0 - my)
    ax.set_aspect("equal")
    ax.set_xlabel("X (µm)", fontsize=11); ax.set_ylabel("Y (µm)", fontsize=11)
    ax.xaxis.set_major_locator(MultipleLocator(1000)); ax.yaxis.set_major_locator(MultipleLocator(1000))

    handles = [Line2D([], [], marker="o", linestyle="none", markersize=7,
                      markerfacecolor=COLORS[c], markeredgecolor="none",
                      label=f"{CELL_LABEL[c]} ({int((df['class'] == c).sum())})")
               for c in ("B_cell", "T_cell", "Plasma", "Lymphatic_Endothelial")]
    ax.legend(handles=handles, loc="upper left", bbox_to_anchor=(1.01, 1.0),
              fontsize=10, framealpha=0.9, borderaxespad=0.0)
    ax.set_title(f"{sample} ({cancer}, {dataset})", fontsize=12, fontweight="bold")
    plt.tight_layout()
    fig.savefig(f"{outdir}/{sample}.png", dpi=200, bbox_inches="tight")
    fig.savefig(f"{outdir}/{sample}.pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"  {sample}: {len(tls_items)} TLS, {len(agg_items['tagg'])} T-agg, "
          f"{len(agg_items['magg'])} Mac-agg; halos {[int(h) for h in HALOS_UM]} -> {outdir}",
          flush=True)


if __name__ == "__main__":
    if len(sys.argv) < 4:
        sys.exit("usage: build_regions.py <dataset> <cancer_type> <sample>")
    main(sys.argv[1], sys.argv[2], sys.argv[3])
