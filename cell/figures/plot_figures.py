#!/usr/bin/env python
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap

FIG = Path(__file__).resolve().parent
DATA = json.loads((FIG / "fig_data_cls_sigma3.json").read_text())

INK = "#0b0b0b"; INK2 = "#52514e"; MUTED = "#8a8985"; GRID = "#e6e5e1"; SURF = "#fcfcfb"
BLUE = "#2a78d6"; BLUE_LIGHT = "#dbe8f8"

plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 10,
    "axes.edgecolor": GRID, "axes.labelcolor": INK, "xtick.color": INK2, "ytick.color": INK2,
    "figure.facecolor": SURF, "axes.facecolor": SURF, "savefig.facecolor": SURF,
})

classes = DATA["classes"]
f1 = DATA["per_class_f1"]
cm = np.asarray(DATA["confusion11_rows_true_cols_pred"], dtype=np.int64)


def label(name):
    return (name.replace("NonMalignant_Parenchymal", "NonMalig. Parenchymal")
                .replace("NK_T", "NK/T").replace("_", " "))


def save(fig, stem):
    for suffix in (".png", ".pdf"):
        fig.savefig(FIG / f"{stem}{suffix}", dpi=400, bbox_inches="tight", pad_inches=0.15)
    plt.close(fig)


order = sorted(classes, key=lambda c: f1[c], reverse=True)
vals = [f1[c] for c in order]
colors = [MUTED if c == "Others" else BLUE for c in order]

fig, ax = plt.subplots(figsize=(8.0, 5.2))
y = np.arange(len(order))[::-1]
ax.barh(y, vals, height=0.62, color=colors, edgecolor=SURF, linewidth=1.0)
ax.set_yticks(y)
ax.set_yticklabels([label(c) for c in order])
ax.set_xlim(0, 1.0)
ax.set_xlabel("F1")
ax.xaxis.grid(True, color=GRID, linewidth=0.8); ax.set_axisbelow(True)
for s in ("top", "right", "left"): ax.spines[s].set_visible(False)
ax.tick_params(axis="y", length=0)
for yi, v in zip(y, vals):
    ax.text(v + 0.012, yi, f"{v:.3f}", va="center", ha="left", color=INK, fontsize=9.5)
ax.set_ylim(-0.7, y.max() + 0.7)
save(fig, "01_per_class_f1_cls_sigma3")

row_sum = cm.sum(axis=1, keepdims=True)
norm = cm / np.maximum(row_sum, 1)
cmap = LinearSegmentedColormap.from_list("blues_seq", ["#f7f9fc", BLUE_LIGHT, "#7fb0e6", BLUE, "#0b3f86"])

fig, ax = plt.subplots(figsize=(9.6, 8.6))
im = ax.imshow(norm, cmap=cmap, vmin=0, vmax=1, aspect="equal")
short = [label(c) for c in classes]
ax.set_xticks(range(len(classes))); ax.set_yticks(range(len(classes)))
ax.set_xticklabels(short, rotation=45, ha="right", fontsize=9)
ax.set_yticklabels(short, fontsize=9)
ax.set_xlabel("Predicted class")
ax.set_ylabel("True class")
ax.set_xticks(np.arange(-0.5, len(classes), 1), minor=True)
ax.set_yticks(np.arange(-0.5, len(classes), 1), minor=True)
ax.grid(which="minor", color=SURF, linewidth=1.5); ax.tick_params(which="minor", length=0)
for s in ax.spines.values(): s.set_visible(False)
for i in range(len(classes)):
    for j in range(len(classes)):
        v = norm[i, j]
        if v < 0.005:
            continue
        ax.text(j, i, f"{100*v:.0f}", ha="center", va="center", fontsize=8,
                color="white" if v > 0.55 else INK)
cb = fig.colorbar(im, ax=ax, fraction=0.035, pad=0.02)
cb.set_label("Share of true-class cells (%)", color=INK2)
cb.set_ticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
cb.set_ticklabels(["0", "20", "40", "60", "80", "100"])
cb.outline.set_visible(False)
save(fig, "02_confusion_matrix_cls_sigma3")

ROC = json.loads((FIG / "roc_data_cls_sigma3.json").read_text())
scored = ROC["scored_classes"]
curves = ROC["curves"]
by_auc = sorted(scored, key=lambda c: curves[c]["auroc_reported"], reverse=True)
palette = ["#2a78d6", "#d1495b", "#3f9b52", "#8656b3", "#e07b39",
           "#1b9aaa", "#c74f9e", "#6b7b8c", "#9c6b30", "#4f6fd0"]
hue = {name: palette[i] for i, name in enumerate(scored)}

fig, ax = plt.subplots(figsize=(6.4, 6.0))
ax.plot([0, 1], [0, 1], color=MUTED, linewidth=0.9, linestyle=(0, (4, 3)), zorder=1)
for name in by_auc:
    c = curves[name]
    ax.plot(c["fpr"], c["tpr"], color=hue[name], linewidth=1.6, zorder=2,
            label=f"{label(name)}  {c['auroc_reported']:.3f}")
ax.set_xlim(0, 1); ax.set_ylim(0, 1.002)
ax.set_xlabel("False positive rate")
ax.set_ylabel("True positive rate")
ax.set_aspect("equal")
ax.grid(True, color=GRID, linewidth=0.8); ax.set_axisbelow(True)
for s in ("top", "right"): ax.spines[s].set_visible(False)
legend = ax.legend(loc="lower right", frameon=False, fontsize=9, handlelength=1.6,
                   labelspacing=0.35, borderaxespad=0.6)
for text in legend.get_texts():
    text.set_color(INK)
save(fig, "03_roc_curves_cls_sigma3")

auroc = {name: curves[name]["auroc_reported"] for name in scored}
order = sorted(scored, key=lambda c: auroc[c], reverse=True)
vals = [auroc[c] for c in order]

fig, ax = plt.subplots(figsize=(8.0, 5.0))
y = np.arange(len(order))[::-1]
ax.barh(y, vals, height=0.62, color=BLUE, edgecolor=SURF, linewidth=1.0)
ax.set_yticks(y)
ax.set_yticklabels([label(c) for c in order])
ax.set_xlim(0.5, 1.0)
ax.set_xlabel("AUROC")
ax.xaxis.grid(True, color=GRID, linewidth=0.8); ax.set_axisbelow(True)
for s in ("top", "right", "left"): ax.spines[s].set_visible(False)
ax.tick_params(axis="y", length=0)
for yi, v in zip(y, vals):
    ax.text(v + 0.006, yi, f"{v:.3f}", va="center", ha="left", color=INK, fontsize=9.5)
ax.set_ylim(-0.7, y.max() + 0.7)
save(fig, "04_per_class_auroc_cls_sigma3")
