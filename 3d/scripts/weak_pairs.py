#!/usr/bin/env python3
import json
import sys
from pathlib import Path

vol, pairs = Path(sys.argv[1]), Path(sys.argv[2])
d = json.load(open(vol / "metadata.json"))
p = sorted(d["encodings"]["G_withdrawn"]["planes"], key=lambda q: q["z_um"])
n = len(p)
weak, secs = [], []
for k in range(2 * n - 3):
    i, j = (k, k + 1) if k < n - 1 else (k - (n - 1), k - (n - 1) + 2)
    a, b = p[i]["section_id"], p[j]["section_id"]
    ua, ub = a.split("-")[-1], b.split("-")[-1]
    f = pairs / f"{ua}_{ub}" / "pair_fits.json"
    ok = False
    if f.exists():
        r = json.load(open(f))
        r = r[0] if isinstance(r, list) and r else (r if isinstance(r, dict) else {})
        ok = bool(r.get("affine_a_to_b_um")) and int(r.get("n_core") or 0) >= 6
    if not ok:
        weak.append(f"{ua},{ub}")
        for s in (a, b):
            if s not in secs:
                secs.append(s)
for w in weak:
    print(w)
for s in secs:
    print("sec " + s)
