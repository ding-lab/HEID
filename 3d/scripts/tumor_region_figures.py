#!/usr/bin/env python3
import argparse
import json
import re
import struct
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.colors
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from PIL import Image
from skimage.measure import find_contours

UM = 8.0
C_TUMOR = "#D62728"
C_TERRITORY = "#7A1416"
C_NERVE = "#00B0F0"
UM_PER_INCH_WIDE = 1400.0
CAVEAT = ("EVERY cell type is a MODEL PREDICTION on H&E (pan-cancer Cell head, single fold); the territory is "
          "called from the predicted tumour cells. No spatial ground truth in this cohort; nothing here is scored.")


def display_sid(sid):
    return re.sub(r"-P\d+_", "_", str(sid))


def read_points(path):
    raw = Path(path).read_bytes()
    magic, n, cw, ch, sub, ncls, _ = struct.unpack("<8sIHHHHI", raw[:24])
    assert magic == b"CELLPT01"
    xy = np.frombuffer(raw[32:32 + 4 * n], dtype="<u2").reshape(n, 2).astype(np.float64) / sub * UM
    cl = np.frombuffer(raw[32 + 5 * n:32 + 6 * n], dtype=np.uint8)
    return xy, cl


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--volume", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--basis", default="G_withdrawn")
    ap.add_argument("--volume4", default="", help="4 um volume with the H&E colour planes; default <volume> with 8um -> 4um")
    a = ap.parse_args()
    VOL, OUT = Path(a.volume), Path(a.out)
    OUT.mkdir(parents=True, exist_ok=True)
    for old in OUT.glob("*_tumor_region.png"):
        old.unlink()
    V4 = Path(a.volume4) if a.volume4 else VOL.parent / VOL.name.replace("8um", "4um")

    tp = np.load(VOL / "tumor_planes.npz")
    tm = json.loads((VOL / "tumor_regions_metadata.json").read_text())
    area_of = {q["section_id"]: q for q in tm.get("sections") or tm["planes"]}
    cpm = json.loads((VOL / "cell_points_metadata.json").read_text())
    classes = cpm["classes"]
    tumour_cls = [i for i, c in enumerate(classes) if str(c).startswith("Tumor")]
    files = {f["section_id"]: f["file"] for f in cpm["files"] if f["basis"] == a.basis}
    cw, ch = cpm["canvas_px"]["width"], cpm["canvas_px"]["height"]
    nerve_regs = {}
    if (VOL / "nerve_regions_metadata.json").exists():
        nm = json.loads((VOL / "nerve_regions_metadata.json").read_text())
        nerve_regs = {p["section_id"]: p.get("regions") or [] for p in nm["planes"]}
    rgb_of, rgb_geo = {}, None
    if (V4 / "he_rgb_metadata.json").exists():
        rm = json.loads((V4 / "he_rgb_metadata.json").read_text())
        rgb_of = {f["section_id"]: V4 / f["file"] for f in rm["planes"] if f["basis"] == a.basis}
        m8 = json.loads((VOL / "metadata.json").read_text()); m4 = json.loads((V4 / "metadata.json").read_text())
        o8, o4 = m8.get("canvas_origin_um", [0, 0]), m4.get("canvas_origin_um", m8.get("canvas_origin_um", [0, 0]))
        rgb_geo = (float(o4[0] - o8[0]), float(o4[1] - o8[1]), float(m4["in_plane_um_per_px"]))
    Image.MAX_IMAGE_PIXELS = None

    n_fig = 0
    for sid in sorted(area_of, key=lambda s: area_of[s]["z_um"]):
        if sid not in files:
            continue
        xy, cl = read_points(VOL / files[sid])
        is_t = np.isin(cl, tumour_cls)
        mask = tp[sid] if sid in tp.files else np.zeros((ch, cw), bool)
        q = area_of[sid]
        t_mm2, ti_mm2 = float(q.get("area_mm2", 0.0)), float(q.get("tissue_area_mm2") or 0.0)
        X0, X1, Y0, Y1 = xy[:, 0].min(), xy[:, 0].max(), xy[:, 1].min(), xy[:, 1].max()
        span_x, span_y = X1 - X0, Y1 - Y0
        width_in = span_x / UM_PER_INCH_WIDE
        fig, axes = plt.subplots(1, 2, figsize=(2 * max(width_in, 4.0) + 3.5, max(span_y / UM_PER_INCH_WIDE, 4.0)))
        point_size = float(np.clip(1.2e4 / max(int(is_t.sum()), 1), 0.35, 1.0))
        bg = None
        if sid in rgb_of and rgb_geo is not None:
            rgba = Image.open(rgb_of[sid]).convert("RGBA")
            bg = np.asarray(Image.alpha_composite(Image.new("RGBA", rgba.size, (255, 255, 255, 255)), rgba).convert("RGB"))
        for ax, show in zip(axes, (False, True)):
            if bg is not None:
                ox, oy, mpp4 = rgb_geo
                ax.imshow(bg, extent=[ox, ox + bg.shape[1] * mpp4, oy + bg.shape[0] * mpp4, oy],
                          interpolation="bilinear", zorder=1)
            else:
                ax.scatter(xy[~is_t, 0], xy[~is_t, 1], s=0.5, c="#d5d5d5", alpha=0.15, linewidths=0,
                           rasterized=True, zorder=1)
            if show and mask.any():
                rgba_m = np.zeros((*mask.shape, 4), np.float32)
                rgba_m[mask] = list(matplotlib.colors.to_rgba(C_TERRITORY)[:3]) + [0.28]
                ax.imshow(rgba_m, extent=[0, mask.shape[1] * UM, mask.shape[0] * UM, 0],
                          interpolation="nearest", zorder=2)
                for contour in find_contours(mask.astype(float), 0.5):
                    ax.plot(contour[:, 1] * UM, contour[:, 0] * UM, color=C_TERRITORY, lw=0.8, alpha=0.9, zorder=3)
            ax.scatter(xy[is_t, 0], xy[is_t, 1], s=point_size, c=C_TUMOR, alpha=0.55, linewidths=0,
                       rasterized=True, zorder=4)
            regs = nerve_regs.get(sid, [])
            if regs:
                ax.scatter([r["cx_um"] for r in regs], [r["cy_um"] for r in regs], s=48, facecolors="none",
                           edgecolors=C_NERVE, linewidths=1.4, zorder=5)
            ax.set_xlim(X0, X1); ax.set_ylim(Y1, Y0); ax.set_aspect("equal")
            ax.set_xlabel("X (µm, canvas)")
            ax.set_title("predicted tumour cells" if not show else "+ tumour territory called from them", fontsize=11)
        axes[0].set_ylabel("Y (µm, canvas)")
        n_regs = len(nerve_regs.get(sid, []))
        fig.suptitle(f"{display_sid(sid)}  z={q['z_um']:.0f} um   {int(is_t.sum()):,} predicted tumour cells of {len(xy):,}  ·  "
                     f"territory {t_mm2:.1f} mm² ({(t_mm2 / ti_mm2 if ti_mm2 else 0):.0%} of tissue)"
                     + (f"  ·  {n_regs} nerve regions" if nerve_regs else "") + "   |   MODEL PREDICTION",
                     fontsize=11, fontweight="bold")
        handles = [Line2D([], [], marker="o", linestyle="none", markersize=6, markerfacecolor=C_TUMOR,
                          markeredgecolor="none", label="predicted tumour cell"),
                   Line2D([], [], color=C_TERRITORY, lw=1.2, label="tumour territory (clinical definition)")]
        if nerve_regs:
            handles.append(Line2D([], [], marker="o", linestyle="none", markersize=8, markerfacecolor="none",
                                  markeredgecolor=C_NERVE, label="3-D nerve region on this section"))
        axes[1].legend(handles=handles, loc="upper left", bbox_to_anchor=(1.01, 1.0), fontsize=9, borderaxespad=0.0)
        fig.text(0.01, 0.005, CAVEAT, fontsize=7.5, color="#444")
        plt.tight_layout()
        fig.savefig(OUT / f"{display_sid(sid)}_tumor_region.png", dpi=200, bbox_inches="tight")
        plt.close(fig)
        n_fig += 1
    print(f"wrote {n_fig} tumour-region figures under {OUT}")


if __name__ == "__main__":
    main()
