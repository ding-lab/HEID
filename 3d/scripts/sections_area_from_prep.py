#!/usr/bin/env python
import argparse
import csv
import json
from pathlib import Path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sections", required=True)
    ap.add_argument("--prep", required=True)
    a = ap.parse_args()
    p = Path(a.sections)
    rows = list(csv.DictReader(open(p), delimiter="\t"))
    n = 0
    for r in rows:
        m = Path(a.prep) / r["section_id"] / "meta.json"
        if not m.exists():
            continue
        s = json.dumps(json.load(open(m)))
        i = s.find('"area_mm2": ')
        r["tissue_area_mm2"] = round(float(s[i + 12:].split(",")[0].split("}")[0]), 4)
        n += 1
    with open(p, "w", newline="") as fh:
        w = csv.DictWriter(fh, list(rows[0]), delimiter="\t")
        w.writeheader(); w.writerows(rows)
    print(f"{n}/{len(rows)} tissue areas set from prep -> {p}")


if __name__ == "__main__":
    main()
