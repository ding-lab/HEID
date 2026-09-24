
import os
import importlib.util, csv, json
import numpy as np
from pathlib import Path
ROOT = Path(os.environ.get("THREED_ROOT", os.path.join(os.environ.get("PROJECTS_ROOT", "/data/heid"), "3d")))
spec = importlib.util.spec_from_file_location("placement_chain", Path(__file__).resolve().parent / "placement_chain.py")
CHAIN = importlib.util.module_from_spec(spec); spec.loader.exec_module(CHAIN)
G = ROOT / "front/S22-27909"

def chain_sign(pat, runs):
    man = {r["section_id"]: r for r in csv.DictReader(open(G / f"configs/sections_{pat}.tsv"), delimiter="\t")}
    sids = sorted(man, key=lambda s: float(man[s]["z_position_um"]))
    edges = CHAIN.harvest([str(G / "outputs" / r) for r in runs])
    tree = CHAIN.prereg_tree(edges, sids)
    ch = CHAIN.chain_on(tree, edges, "M", sids[len(sids)//2])
    return {s: float(np.sign(np.linalg.det(ch[s][:2, :2]))) for s in ch}, edges, tree

o1, e1, t1 = chain_sign("P1", ["reg_P1_fix3", "reg_P1_fix", "reg_P1_wg2"])
o2, e2, t2 = chain_sign("P2", ["reg_P2_fix3", "reg_P2_fix", "reg_P2_wg2"])


cross = CHAIN.harvest([str(G / "outputs/reg_M")])
rows = []
for t, e in cross.items():
    if e["fixed"] in o1 and e["moving"] in o2:
        oa, ob = o1[e["fixed"]], o2[e["moving"]]
    elif e["fixed"] in o2 and e["moving"] in o1:
        oa, ob = o2[e["fixed"]], o1[e["moving"]]
    else:
        continue
    d = float(np.sign(np.linalg.det(e["M"][:2, :2])))


    rows.append((t, d * oa * ob))
vals = [v for _, v in rows]
F = 1.0 if sum(vals) >= 0 else -1.0
bad = [t for t, v in rows if v != F]
print(f"cross edges voting: {len(rows)}; block-to-block flip F={'MIRROR' if F<0 else 'upright'}")
print(f"majority {vals.count(F)}, minority (flip-wrong) {len(bad)}:")
for t in sorted(bad):
    print("  ", t)
