#!/usr/bin/env python3
import csv, glob, json, os, re, struct, subprocess, sys
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).resolve().parent))


def cfg_env(cfg):
    keys = ["SAMPLE", "VOL", "DELIV", "PAGE", "S14", "OBJ3D", "TLSROOT", "NERVE", "RAW"]
    out = subprocess.run(["bash", "-c", "source " + cfg + "; " + "; ".join(f"echo {k}=${k}" for k in keys)],
                         capture_output=True, text=True, check=True).stdout
    return dict(l.split("=", 1) for l in out.strip().splitlines())


def read_points(f):
    raw = open(f, "rb").read()
    m = struct.unpack("<8sIHHHHI", raw[:24]); n, sub = m[1], m[4]
    xy = np.frombuffer(raw[32:32 + 4 * n], dtype="<u2").reshape(n, 2).astype(float) / sub * 8.0
    cl = np.frombuffer(raw[32 + 5 * n:32 + 6 * n], dtype=np.uint8)
    return xy, cl


def check_tls_centroids(table, metadata, volume):
    classes = metadata["classes"]
    lymphocytes = [classes.index(c) for c in ("B_cell", "NK_T") if c in classes]
    aliases = {}
    for section in metadata["sections"]:
        for name in (section.get("slide"), section["section_id"]):
            if name:
                aliases.setdefault(name, set()).add(section["section_id"])
    checked, low, missing = 0, [], []
    for slide, group in table.groupby("slide_id", dropna=False):
        explicit = set(group["section_id"].dropna().astype(str)) if "section_id" in group else set()
        explicit.discard("")
        candidates = explicit or aliases.get(slide, set())
        sid = next(iter(candidates)) if len(candidates) == 1 else None
        files = [f for f in metadata["files"]
                 if f["basis"] == "G_withdrawn" and f["section_id"] == sid]
        if not lymphocytes or len(files) != 1 or not (volume / files[0]["file"]).is_file():
            missing.extend(f"{slide}:{r.tls_id} (section/points unavailable or ambiguous)"
                           for r in group.itertuples())
            continue
        xy, cl = read_points(volume / files[0]["file"])
        points = xy[np.isin(cl, lymphocytes)]
        for row in group.itertuples():
            key = f"{slide}:{row.tls_id}"
            center = np.asarray([row.cx_um_canvas, row.cy_um_canvas], dtype=float)
            if not np.isfinite(center).all():
                missing.append(key + " (non-finite centroid)")
                continue
            checked += 1
            if (np.hypot(points[:, 0] - center[0], points[:, 1] - center[1]) < 150).sum() < 10:
                low.append(key)
    return checked, low, missing


def main():
    e = cfg_env(sys.argv[1])
    S, VOL, DELIV, PAGE, S14 = e["SAMPLE"], Path(e["VOL"]), Path(e["DELIV"]), Path(e["PAGE"]), Path(e["S14"])
    bad = 0
    def rep(ok, tag, msg):
        nonlocal bad
        bad += (not ok); print(("OK " if ok else "BAD") + f" {tag}: {msg}", flush=True)
    payload = (PAGE / "objects_payload.js").read_text() if (PAGE / "objects_payload.js").exists() else ""
    page_nerves = set(re.findall(r'"(nerve-\d+)"', payload))
    page_tls = set(re.findall(r'"(3d-tls-\d+)"', payload))


    if e.get("NERVE", "1") != "0" and (S14 / "Schwann_nerves.json").exists():
        rep_n = json.load(open(S14 / "Schwann_nerves.json"))["n_nerves"]
        tsv = DELIV / "nerve/1_cords/1_cords.tsv"
        rows = sum(1 for _ in open(tsv)) - 1 if tsv.exists() else -1
        folders = len(os.listdir(DELIV / "nerve/2_objects")) if (DELIV / "nerve/2_objects").is_dir() else -1
        rep(rows == rep_n == len(page_nerves) and folders <= rep_n and folders >= 0, "N1",
            f"table {rows}, report {rep_n}, page {len(page_nerves)}, crop folders {folders}")
        try:
            cmp_ = VOL / "keep2d_metadata.json"; cm = json.load(open(cmp_ if cmp_.exists() else VOL / "cells_denoised_metadata.json"))


            si = json.load(open(VOL / "cell_points_metadata.json"))["classes"].index("Schwann")
            worst = 0; n_pl = 0
            for r in cm["planes"]:
                kf = VOL / "keep2d" / f"z{r['index']:02d}.keep2d.bin"
                k3 = VOL / "keep3d" / f"z{r['index']:02d}.keep3d.bin"
                pf = sorted(glob.glob(str(VOL / "G_withdrawn/cell_points" / f"z{r['index']:02d}_*.cells.bin")))
                if not (kf.exists() and k3.exists() and len(pf) == 1):
                    continue
                keep = np.frombuffer(open(kf, "rb").read(), np.uint8).astype(bool)
                keep3 = np.frombuffer(open(k3, "rb").read(), np.uint8).astype(bool)
                xy, cl = read_points(pf[0])
                if not (len(keep) == len(keep3) == len(cl)):
                    raise ValueError(f"plane {r['index']}: keep2d/keep3d/classes lengths differ: "
                                     f"{len(keep)}/{len(keep3)}/{len(cl)}")
                bad_in = int((keep3 & (cl == si) & ~keep).sum()); bad_out = int((keep & ~keep3).sum())
                worst = max(worst, bad_in + bad_out); n_pl += 1
            rep(n_pl > 0 and n_pl == len(cm["planes"]) and worst <= 3, "N2",
                f"2-D keep2d vs 3-D keep3d Schwann on {n_pl}/{len(cm['planes'])} planes: worst plane {worst} cells differ")
            rep("FROZEN" in cm.get("what", ""), "N3", cm.get("what", "")[:60])
        except Exception as ex:
            rep(False, "N2", f"check failed: {ex}")
    else:
        print("--  N*: no nerve chain for this sample")


    obj = Path(e["OBJ3D"]) / "S9_tls_objects/objects.tsv"
    objs = list(csv.DictReader(open(obj), delimiter="\t")) if obj.exists() else []
    n3 = sum(o["object_id"].startswith("3d") for o in objs)
    rpt = DELIV / "tls/1_objects_report/1_objects_report.tsv"
    t3 = len({l.split("\t")[0] for l in open(rpt) if l.startswith("3d-tls")}) if rpt.exists() else -1
    rep(n3 == t3 == len(page_tls), "T1", f"objects.tsv {n3}, report {t3}, page {len(page_tls)}")
    tab = Path(e["RAW"]) / "tls_3d_table.tsv"
    if objs and tab.exists():
        import pandas as pd
        t = pd.read_csv(tab, sep="\t", comment="#"); t = t[t.sample_id == S] if "sample_id" in t else t
        cp = json.load(open(VOL / "cell_points_metadata.json"))
        n, low, missing = check_tls_centroids(t, cp, VOL)
        rep(n > 0 and n == len(t) and not low and not missing, "T2",
            f"{n}/{len(t)} table centroids checked, {len(low)} without >= 10 lymphocytes within 150 um {low[:5]}; "
            f"{len(missing)} unchecked {missing[:5]}")
        mism = []
        for o in objs:
            mem = [m for m in (o.get("member_tls") or "").split(";") if ":" in m]
            d = DELIV / "tls/2_objects" / o["object_id"]
            if not mem:
                continue
            nimg = {k: len(os.listdir(d / k)) for k in ("he", "codex", "xenium") if (d / k).is_dir()}

            nsec = len({m.split(":")[0] for m in mem})
            if sum(nimg.values()) != nsec + nimg.get("xenium", 0):
                mism.append(f"{o['object_id']}:{sum(nimg.values())}/{nsec}")
        rep(len(mism) == 0, "T3", f"{len(objs)} objects, {len(mism)} with missing crops {mism[:5]}")
        try:
            from PIL import Image
            fails = []; n_img = 0
            for f in sorted(glob.glob(str(DELIV / "tls/2_objects/*/*/U*.png"))):
                a = np.asarray(Image.open(f).convert("RGB")).astype(int)


                m = (abs(a[..., 0] - 255) <= 15) & (abs(a[..., 1] - 45) <= 20) & (abs(a[..., 2] - 149) <= 20); n_img += 1
                k = int(m.sum())
                if k < 800:
                    fails.append(os.path.basename(os.path.dirname(os.path.dirname(f))) + "/" + os.path.basename(f)); continue
                ys, xs = np.nonzero(m); h, w = m.shape


                if abs(xs.mean() - w / 2) > 0.85 * w / 2 or abs(ys.mean() - h / 2) > 0.85 * h / 2:
                    fails.append(os.path.basename(os.path.dirname(os.path.dirname(f))) + "/" + os.path.basename(f))
            rep(len(fails) == 0, "T4", f"{n_img} crops, {len(fails)} without a centred outline {fails[:5]}")
        except Exception as ex:
            rep(False, "T4", f"check failed: {ex}")
    elif objs:
        rep(False, "T2", f"TLS objects exist but the centroid table is missing: {tab}")
    print(f"RESULT {'CONSISTENT' if bad == 0 else 'INCONSISTENT (' + str(bad) + ' checks)'}: {S}")
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
