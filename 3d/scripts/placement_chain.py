#!/usr/bin/env python
from __future__ import annotations

import glob
import json
import os
from pathlib import Path

import numpy as np

ROOT = Path(os.environ.get("THREED_ROOT", os.path.join(os.environ.get("PROJECTS_ROOT", "/data/heid"), "3d")))
RUNS = ROOT / "registration/runs"


def invert(M):
    M = np.asarray(M, float)
    Ai = np.linalg.inv(M[:2, :2])
    o = np.zeros((2, 3)); o[:2, :2] = Ai; o[:, 2] = -Ai @ M[:, 2]
    return o


def compose(Mo, Mi):
    Mo, Mi = np.asarray(Mo, float), np.asarray(Mi, float)
    o = np.zeros((2, 3))
    o[:2, :2] = Mo[:2, :2] @ Mi[:2, :2]
    o[:, 2] = Mo[:2, :2] @ Mi[:, 2] + Mo[:, 2]
    return o


def iso_of(A):
    return 0.5 * np.log(abs(np.linalg.det(np.asarray(A, float)[:2, :2])))


def harvest(runs):
    rows, seen = {}, set()
    for run in runs:
        for f in sorted(glob.glob(str(RUNS / run / "*/results.json"))):
            try:
                d = json.load(open(f))
            except Exception:
                continue
            d = d[0] if isinstance(d, list) else d
            if not d.get("ok") or not d.get("M_moving_to_fixed"):
                continue
            k = tuple(sorted((d["fixed"], d["moving"])))
            if k in seen:
                continue
            seen.add(k)
            rows[os.path.basename(os.path.dirname(f))] = {
                "fixed": d["fixed"], "moving": d["moving"],
                "dz": abs(float(d.get("dz_um") or 0.0)),
                "M": np.array(d["M_moving_to_fixed"], float),
                "R": np.array(d["residual_fit_R"], float) if d.get("residual_fit_R") else None}
    return rows


def kruskal(keyed, edges, nodes):
    par = {n: n for n in nodes}; rank = {n: 0 for n in nodes}

    def find(x):
        while par[x] != x:
            par[x] = par[par[x]]; x = par[x]
        return x
    chosen = []
    for _, tag in sorted(keyed):
        e = edges[tag]
        a, b = find(e["fixed"]), find(e["moving"])
        if a == b:
            continue
        if rank[a] < rank[b]:
            a, b = b, a
        par[b] = a; rank[a] += (rank[a] == rank[b]); chosen.append(tag)
    return chosen


def prereg_tree(edges, nodes):


    keep = {t: e for t, e in edges.items() if e["fixed"] in nodes and e["moving"] in nodes}
    if len(keep) < len(edges):
        print(f"prereg_tree: {len(edges) - len(keep)} edges touch sections outside the "
              f"manifest and are ignored", flush=True)
    edges = keep
    return kruskal([((e["dz"], min(e["fixed"], e["moving"]), max(e["fixed"], e["moving"])), t)
                    for t, e in edges.items()], edges, nodes)


def chain_on(tree, edges, key, anchor):
    adj = {}
    for t in tree:
        e = edges[t]
        if e[key] is None:
            continue
        adj.setdefault(e["fixed"], []).append((e["moving"], e[key], +1))
        adj.setdefault(e["moving"], []).append((e["fixed"], e[key], -1))
    Ab = {anchor: np.array([[1.0, 0, 0], [0, 1.0, 0]])}
    seen, fr = {anchor}, [anchor]
    while fr:
        nx = []
        for u in fr:
            for v, M, s in adj.get(u, []):
                if v in seen:
                    continue
                Ab[v] = compose(Ab[u], M if s == +1 else invert(M))
                seen.add(v); nx.append(v)
        fr = nx
    return section_corrections(Ab)


def section_corrections(Ab):
    p = os.environ.get("HTAN3D_SECTION_CORRECTIONS", "")
    if not p:
        return Ab
    if not os.path.exists(p):
        if not getattr(section_corrections, "_warned", False):
            print(f"HTAN3D_SECTION_CORRECTIONS={p}: file not there, chain used as solved (uncorrected)",
                  flush=True)
            section_corrections._warned = True
        return Ab
    d = json.load(open(p))
    grid = float(d.get("chain_grid_um", 2.0))


    arm = os.environ.get("HTAN3D_ARM", "main")
    if arm == "he":
        arm = "swap"
    resid = d.get("swap_to_main_residual") or {}
    resid_cells = set(d.get("swap_frame_cells") or [])
    n = nr = 0
    for v, M in list(Ab.items()):
        C = d["sections"].get(v)
        R = resid.get(v) if (arm == "swap" or (arm == "cells" and v in resid_cells)) else None
        if C is None and R is None:
            continue
        T = np.asarray(C if C is not None else [[1.0, 0, 0], [0, 1.0, 0]], float)
        if R is not None:
            R = np.asarray(R, float)
            T = (np.vstack([R, [0, 0, 1]]) @ np.vstack([T, [0, 0, 1]]))[:2]
            nr += 1
        out = np.zeros((2, 3))
        out[:, :2] = T[:, :2] @ M[:, :2]
        out[:, 2] = T[:, :2] @ M[:, 2] + T[:, 2] / grid
        Ab[v] = out
        n += 1
    if not getattr(section_corrections, "_said", False):
        print(f"section corrections applied from {p}: {n} of {len(Ab)} sections "
              f"(frame {arm}, swap-to-main residual on {nr})", flush=True)
        section_corrections._said = True
    return Ab
