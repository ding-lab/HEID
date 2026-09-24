#!/usr/bin/env python3
import argparse
import importlib.util
import json
import os
import pickle
import re
import struct
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from skimage.measure import find_contours

UM = 8.0
COL_NERVE = "#00B0F0"
HERE = Path(__file__).resolve().parent


def display_sid(sid):
    return re.sub(r"-P\d+_", "_", str(sid))


def _load(name, p):
    spec = importlib.util.spec_from_file_location(name, p)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m
CAVEAT = ("EVERY cell type is a MODEL PREDICTION on H&E (pan-cancer Cell head, single fold). "
          "No spatial ground truth in this cohort; nothing here is scored.")


def read_points(path):
    raw = Path(path).read_bytes()
    magic, n, cw, ch, sub, ncls, _ = struct.unpack("<8sIHHHHI", raw[:24])
    assert magic == b"CELLPT01"
    xy = np.frombuffer(raw[32:32 + 4 * n], dtype="<u2").reshape(n, 2).astype(np.float64) / sub * UM
    cl = np.frombuffer(raw[32 + 5 * n:32 + 6 * n], dtype=np.uint8)
    return xy, cl


def leader_labels(ax, cx, cy, rad, ids, X0, X1, Y0, Y1, col):
    cx, cy, rad = map(np.asarray, (cx, cy, rad))
    diag = float(np.hypot(X1 - X0, Y1 - Y0)); sep, pad = 0.05 * diag, 0.03 * diag
    d0x, d0y = cx - (X0 + X1) / 2, cy - (Y0 + Y1) / 2
    dn = np.hypot(d0x, d0y); dn[dn < 1e-6] = 1.0
    lx = cx + d0x / dn * (rad + pad); ly = cy + d0y / dn * (rad + pad)
    for _ in range(400):
        dx = lx[:, None] - lx[None, :]; dy = ly[:, None] - ly[None, :]
        dist = np.hypot(dx, dy); np.fill_diagonal(dist, 1e9)
        f = np.where(dist < sep, (sep - dist) / np.maximum(dist, 1e-6) * 0.5, 0.0)
        mvx = (dx * f).sum(1); mvy = (dy * f).sum(1)
        ex = lx[:, None] - cx[None, :]; ey = ly[:, None] - cy[None, :]
        ed = np.hypot(ex, ey); need = (rad[None, :] + pad) - ed
        g = np.where(need > 0, need / np.maximum(ed, 1e-6), 0.0)
        mvx += (ex * g).sum(1); mvy += (ey * g).sum(1)
        lx += np.clip(mvx, -sep, sep) * 0.35; ly += np.clip(mvy, -sep, sep) * 0.35
        lx = np.clip(lx, X0 - 0.10 * (X1 - X0), X1 + 0.10 * (X1 - X0))
        ly = np.clip(ly, Y0 - 0.10 * (Y1 - Y0), Y1 + 0.10 * (Y1 - Y0))
    for k in range(len(ids)):
        ax.annotate(ids[k], xy=(cx[k], cy[k]), xytext=(lx[k], ly[k]), fontsize=8, fontweight="bold",
                    color="#111", ha="center", va="center", zorder=13, annotation_clip=False,
                    arrowprops=dict(arrowstyle="-", color=col, lw=0.8, alpha=0.7, shrinkA=0, shrinkB=0),
                    bbox=dict(boxstyle="round,pad=0.25", fc="white", ec=col, lw=0.8, alpha=0.95))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--volume", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--basis", default="G_withdrawn")
    ap.add_argument("--volume4", default="", help="4 um volume with the H&E colour planes; default <volume> with 8um -> 4um")
    ap.add_argument("--nerve-dir", default=os.environ.get("HTAN3D_S14", ""), dest="nerve_dir",
                    help="the curve run's S14 directory (nerve_members.npz, Schwann_nerves.json); default $HTAN3D_S14")
    a = ap.parse_args()
    VOL, OUT = Path(a.volume), Path(a.out)
    S14 = Path(a.nerve_dir) if a.nerve_dir else None
    if S14 is None or not (S14 / "nerve_members.npz").exists():
        raise SystemExit(f"no nerve_members.npz under {S14}: give --nerve-dir (the curve run's S14 directory)")
    OUT.mkdir(parents=True, exist_ok=True)

    for old in list(OUT.glob("*_nerve.png")) + list(OUT.glob("*_nerve_he.png")) + list(OUT.glob("*/*_nerve.png")) + list(OUT.glob("*/*_nerve_he.png")):
        old.unlink()
    V4 = Path(a.volume4) if a.volume4 else VOL.parent / VOL.name.replace("8um", "4um")
    rgb_of, rgb_geo = {}, None
    if (V4 / "he_rgb_metadata.json").exists():
        rm = json.loads((V4 / "he_rgb_metadata.json").read_text())
        rgb_of = {f["section_id"]: V4 / f["file"] for f in rm["planes"] if f["basis"] == a.basis}
        m8 = json.loads((VOL / "metadata.json").read_text()); m4 = json.loads((V4 / "metadata.json").read_text())

        o8, o4 = m8.get("canvas_origin_um", [0, 0]), m4.get("canvas_origin_um", m8.get("canvas_origin_um", [0, 0]))
        rgb_geo = (float(o4[0] - o8[0]), float(o4[1] - o8[1]), float(m4["in_plane_um_per_px"]))
    meta = json.loads((VOL / "nerve_regions_metadata.json").read_text())
    cpm = json.loads((VOL / "cell_points_metadata.json").read_text())
    classes = cpm["classes"]
    schwann = classes.index("Schwann")
    files = {f["section_id"]: f["file"] for f in cpm["files"] if f["basis"] == a.basis}


    BNC = _load("bnc", HERE / "build_nerve_curves3d.py")
    res = float(BNC.RES)
    st2_files = sorted((VOL / "nerve_operator_cache").glob("stage2_*.pkl"), key=os.path.getmtime)
    if not st2_files:
        raise SystemExit("no stage-2 cache under the volume: the nerve chain has not run here")
    st2 = pickle.load(open(st2_files[-1], "rb"))
    data, cg_all = st2["data"], st2["cg"]
    order = sorted(data, key=lambda s: data[s][0])
    zs = [float(data[s][0]) for s in order]
    ny, nx = next(iter(cg_all.values()))[0].shape
    mem = np.load(S14 / "nerve_members.npz")
    raw = Path(str(mem["cloud"])).read_bytes()
    n_pts = struct.unpack_from("<I", raw, 8)[0]
    arr = np.frombuffer(raw, np.uint16, 3 * n_pts, 12).reshape(n_pts, 3)
    P = arr[:, :2].astype(np.float64)
    SEC = arr[:, 2].astype(np.int32)
    rep = {e["name"]: e for e in json.loads((S14 / "Schwann_nerves.json").read_text())["nerves"]}
    names = sorted((k for k in mem.files if k.startswith("nerve-")), key=lambda k: int(k.split("-")[-1]))
    by_nerve_sec = {}
    for nid in names:
        idx = np.asarray(mem[nid])
        for ks in np.unique(SEC[idx]):
            by_nerve_sec[(nid, int(ks))] = idx[SEC[idx] == ks]
    n_fig = 0
    for p in meta["planes"]:
        if not p.get("regions"):
            continue
        sid = p["section_id"]
        if sid not in files or sid not in order:
            print(f"  {sid}: no points / not in the curve run, skipped"); continue
        ks = order.index(sid)
        xy, cl = read_points(VOL / files[sid])
        X0, X1, Y0, Y1 = xy[:, 0].min(), xy[:, 0].max(), xy[:, 1].min(), xy[:, 1].max()
        other = cl != schwann
        cxs, cys, rads, ids, conts = [], [], [], [], []
        for nid in names:
            js = by_nerve_sec.get((nid, ks))
            if js is None:
                continue
            poly = np.array(rep.get(nid, {}).get("centreline_um") or np.zeros((0, 3)), np.float64)
            m = BNC.region_mask(P[js, 0], P[js, 1], poly, zs[ks], ny, nx)
            ys_, xs_ = np.nonzero(m)
            y0, y1, x0, x1 = max(0, ys_.min() - 2), min(ny, ys_.max() + 3), max(0, xs_.min() - 2), min(nx, xs_.max() + 3)
            for c in find_contours(m[y0:y1, x0:x1].astype(float), 0.5):
                conts.append(c + [y0, x0])
            cxs.append(xs_.mean() * res); cys.append(ys_.mean() * res); ids.append(nid)
            rads.append(np.sqrt(m.sum() * res * res / np.pi))
        regs = ids
        if len(ids) != len(p["regions"]):
            print(f"  {sid}: {len(ids)} member nerves vs {len(p['regions'])} regions in the metadata", flush=True)

        def finish(ax, fig, first_handle, fname):
            for c in conts:
                ax.plot(c[:, 1] * res, c[:, 0] * res, color=COL_NERVE, lw=1.3, alpha=0.9, zorder=9)
            leader_labels(ax, cxs, cys, rads, ids, X0, X1, Y0, Y1, COL_NERVE)
            ax.set_xlim(X0, X1); ax.set_ylim(Y1, Y0); ax.set_aspect("equal")
            ax.set_xlabel("X (µm, canvas)"); ax.set_ylabel("Y (µm, canvas)")
            handles = [first_handle,
                       Line2D([], [], color=COL_NERVE, lw=2, label=f"nerve region of an accepted cord ({len(regs)}), labelled by cord id")]
            ax.legend(handles=handles, loc="upper left", bbox_to_anchor=(1.01, 1.0), fontsize=9)
            ax.set_title(f"{display_sid(sid)} — {len(regs)} nerve regions", fontsize=12, fontweight="bold")
            fig.text(0.01, 0.005, CAVEAT, fontsize=7.5, color="#444", wrap=True)
            (OUT / display_sid(sid)).mkdir(parents=True, exist_ok=True)
            fig.savefig(OUT / display_sid(sid) / fname, dpi=180, bbox_inches="tight")
            plt.close(fig)

        fig, ax = plt.subplots(figsize=(14, 12))
        ax.scatter(xy[other, 0], xy[other, 1], c="#d5d5d5", s=0.6, alpha=0.08, rasterized=True,
                   zorder=0, edgecolors="none")
        ax.scatter(xy[~other, 0], xy[~other, 1], c="#6a3d9a", s=1.0, alpha=0.7, rasterized=True,
                   zorder=5, edgecolors="none")
        finish(ax, fig, Line2D([], [], marker="o", linestyle="none", markersize=7, markeredgecolor="none",
                               markerfacecolor="#6a3d9a", label=f"Schwann (predicted) ({int((~other).sum())})"),
               f"{display_sid(sid)}_nerve.png")
        n_fig += 1

        if sid in rgb_of and rgb_geo is not None:
            from PIL import Image
            Image.MAX_IMAGE_PIXELS = None
            rgba = Image.open(rgb_of[sid]).convert("RGBA")
            im = np.asarray(Image.alpha_composite(Image.new("RGBA", rgba.size, (255, 255, 255, 255)), rgba).convert("RGB"))
            ox, oy, mpp4 = rgb_geo
            fig, ax = plt.subplots(figsize=(14, 12))
            ax.imshow(im, extent=[ox, ox + im.shape[1] * mpp4, oy + im.shape[0] * mpp4, oy],
                      interpolation="bilinear", zorder=0)
            finish(ax, fig, Line2D([], [], marker="s", linestyle="none", markersize=7, markeredgecolor="none",
                                   markerfacecolor="#c08aa0", label="H&E (4 um/px colour plane, registered canvas)"),
                   f"{display_sid(sid)}_nerve_he.png")
            n_fig += 1
    print(f"wrote {n_fig} per-slide nerve figures under {OUT}")


if __name__ == "__main__":
    main()
