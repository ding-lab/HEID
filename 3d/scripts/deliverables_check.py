#!/usr/bin/env python3
import re, sys, os, subprocess, time
from pathlib import Path

cfg = Path(sys.argv[1])
env = subprocess.run(["bash", "-c", f"source {cfg}; echo SAMPLE=$SAMPLE; echo G5=$G5; echo DELIV=$DELIV; echo PAGE=$PAGE; echo NERVE=${{NERVE:-1}}; echo XENIUM=${{XENIUM:-0}}; echo R3D=$R3D"],
                     capture_output=True, text=True).stdout
V = dict(l.split("=", 1) for l in env.strip().split("\n"))
D, P, G5, R3D = Path(V["DELIV"]), Path(V["PAGE"]), Path(V["G5"]), Path(V["R3D"])
S = V["SAMPLE"]; nerve = V["NERVE"] == "1"; xen = V["XENIUM"] == "1"
items = [
    ("A1", "3-D alignment", D / "align" / "1_section_corrections.tsv", True),
    ("A2", "alignment pair examples", D / "align" / "2_pair_examples", False),
    ("B1", "TLS object table", D / "tls" / "1_objects_report" / "1_objects_report.tsv", True),
    ("B2", "TLS object crops", D / "tls" / "2_objects", True),
    ("B3", "TLS per-section figures", D / "tls" / "3_per_slide", True),
    ("C1", "nerve cord table", D / "nerve" / "1_cords" / "1_cords.tsv", nerve),
    ("C2", "nerve object crops", D / "nerve" / "2_objects", nerve),
    ("C3", "nerve per-section figures", D / "nerve" / "3_per_slide", nerve),
    ("C4", "nerve halo cell-type table", D / "nerve" / "4_halo_celltypes" / "4_halo_celltypes.tsv", nerve),
    ("D1", "viewer page", R3D / "samples" / "viewer" / "index.html", True),
    ("D2", "sample data", P / "sample_data.js", True),
    ("D3", "2-D cell tables", R3D / "samples" / "cell_tables_2d" / S, True),
    ("E1", "Xenium TLS comparison", D / "other" / "xenium_tls_477", xen),
    ("E2", "scan previews", D / "other" / "scan_previews", False),
]
rows, missing = [], []
for code, name, path, required in items:
    if path.exists():
        n = len(list(path.iterdir())) if path.is_dir() else 1
        mt = time.strftime("%Y-%m-%d %H:%M", time.localtime(path.stat().st_mtime))
        rows.append((code, name, str(path.relative_to(R3D)), "present", str(n), mt))
    else:
        rows.append((code, name, str(path.relative_to(R3D)) if str(path).startswith(str(R3D)) else str(path), "MISSING" if required else "n/a", "0", ""))
        if required: missing.append(code)
D.mkdir(parents=True, exist_ok=True)
with open(D / "MANIFEST.tsv", "w") as fh:
    fh.write("code\tname\tpath\tstatus\tn_items\tmtime\n")
    for r in rows: fh.write("\t".join(r) + "\n")
for r in rows: print(f"  {r[0]:<3} {r[3]:<8} {r[4]:>5}  {r[5]:<16} {r[2]}")
print(f"{S}: {'OK' if not missing else 'MISSING ' + ' '.join(missing)}  -> {D / 'MANIFEST.tsv'}")
sys.exit(1 if missing else 0)
