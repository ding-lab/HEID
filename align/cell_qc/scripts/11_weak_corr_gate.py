#!/usr/bin/env python
import argparse
import json
import os
import sys
from pathlib import Path

import pandas as pd

MIN_PROM = 0.02
MAX_WEAK_FRAC = 0.50
MIN_MEASURED = 30

PROJ = Path(os.environ.get("PROJECTS_ROOT", "/data/heid"))
OFFSETS_ROOT = Path(__file__).resolve().parents[1] / "outputs/offsets"
QC_ROOT = Path(__file__).resolve().parents[1]
ALI = PROJ / "align/registered_he"
OUT = QC_ROOT / "outputs/results/weak_gate"


def log(m):
    print(m, flush=True)


def known_lowqc():
    import re

    out = {}
    if not ALI.is_dir():
        log(f"low-QC comparison unavailable: {ALI} not present; already_lowqc/location columns stay empty")
        return out
    for coh in ("5k", "477"):
        root = ALI / coh
        if not root.is_dir():
            continue
        for canc in sorted(root.iterdir()):
            if not canc.is_dir():
                continue
            for d in canc.iterdir():
                if not d.is_dir():
                    continue
                if d.name.endswith("-lowqc") or "-bad" in d.name:
                    out[re.sub(r"(-bad)?-lowqc$", "", d.name)] = f"{coh}/{canc.name}"
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--min-prom", type=float, default=MIN_PROM)
    ap.add_argument("--max-weak-frac", type=float, default=MAX_WEAK_FRAC)
    ap.add_argument("--min-measured", type=int, default=MIN_MEASURED)
    ap.add_argument("--tiles-dir", action="append", default=None,
                    help="directory holding <field>_tiles.csv; repeatable, later "
                         "directories win. Defaults to outputs/offsets.")
    ap.add_argument("--out", default=str(OUT))
    a = ap.parse_args()
    log(f"host={os.uname().nodename}")

    dirs = [Path(x) for x in a.tiles_dir] if a.tiles_dir else [OFFSETS_ROOT]
    files = {}
    for d in dirs:
        if not d.is_dir():
            continue
        for f in sorted(d.glob("*_tiles.csv")):
            files[f.name[:-len("_tiles.csv")]] = f
    log(f"{len(files)} sections measured from {len(dirs)} directories")


    combos = {}
    for fld, f in files.items():
        j = f.with_name(f"{fld}.json")
        if not j.exists():
            continue
        pr = json.loads(j.read_text())
        pr = pr.get("params", pr)
        key = (pr.get("MIN_CELLS_TILE"), pr.get("MIN_HE_TISSUE"), pr.get("MIN_PROM"))
        combos.setdefault(key, []).append(fld)
    if len(combos) > 1:
        log("  WARNING sections were measured with different tile gates; the "
            "weak fractions below are not comparable:")
        for k, v in sorted(combos.items(), key=lambda x: -len(x[1])):
            log(f"    cells>={k[0]} tissue>={k[1]} prom>={k[2]}: {len(v)} sections")

    lq = known_lowqc()
    rows = []
    for fld, f in sorted(files.items()):
        t = pd.read_csv(f)
        meas = t[t.status == "ok"]
        weak = meas[meas.he_prominence < a.min_prom]
        n_m, n_w = len(meas), len(weak)
        frac = n_w / n_m if n_m else float("nan")
        judged = n_m >= a.min_measured
        rows.append(dict(field=fld, n_tiles=len(t), n_measured=n_m, n_weak=n_w,
                         weak_frac=frac, prom_median=float(meas.he_prominence.median())
                         if n_m else float("nan"),
                         peak_median=float(meas.he_peak.median()) if n_m else float("nan"),
                         judged=judged,
                         drop_weak=bool(judged and frac > a.max_weak_frac),
                         already_lowqc=fld in lq, location=lq.get(fld, "")))
    df = pd.DataFrame(rows).sort_values("weak_frac", ascending=False)
    outd = Path(a.out)
    outd.mkdir(parents=True, exist_ok=True)
    df.to_csv(outd / "weak_corr_gate.csv", index=False)

    hit = df[df.drop_weak & df.already_lowqc]
    new = df[df.drop_weak & ~df.already_lowqc]
    missed = df[~df.drop_weak & df.already_lowqc]
    clean = df[~df.drop_weak & ~df.already_lowqc]

    def show(title, sub):
        log(f"\n{title}  ({len(sub)})")
        if sub.empty:
            log("  --")
            return
        log(f"  {'field':<34}{'weak/measured':>16}{'frac':>8}{'prom med':>10}{'peak med':>10}")
        for _, r in sub.iterrows():
            log(f"  {r.field:<34}{f'{r.n_weak}/{r.n_measured}':>16}"
                f"{r.weak_frac:>8.3f}{r.prom_median:>10.4f}{r.peak_median:>10.3f}")

    show("flagged low-QC and caught by this gate (true positives)", hit)
    show("excluded by this gate and not marked low-QC on disk", new)
    show("flagged low-QC but passed by this gate (failure modes this gate cannot see)", missed)
    log(f"\npassed ({len(clean)} sections); the 5 highest weak_frac:")
    for _, r in clean.head(5).iterrows():
        log(f"  {r.field:<34}{r.weak_frac:>8.3f}")

    (outd / "weak_corr_gate.json").write_text(json.dumps(
        {"min_prom": a.min_prom, "max_weak_frac": a.max_weak_frac,
         "min_measured": a.min_measured, "n_sections": len(df),
         "n_drop": int(df.drop_weak.sum()),
         "n_true_positive": len(hit), "n_new": len(new), "n_missed": len(missed),
         "drop_fields": df.loc[df.drop_weak, "field"].tolist(),
         "new_fields": new.field.tolist()}, indent=2))
    log(f"\nwrote {outd}/weak_corr_gate.csv / .json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
