#!/usr/bin/env python3
import argparse
import re
from pathlib import Path

import cv2
import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-slide", required=True, dest="per_slide")
    ap.add_argument("--nerve", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--ncol", type=int, default=6)
    ap.add_argument("--sample", default="", help="sample id for the title (default: from the section directory names)")
    a = ap.parse_args()

    files = sorted(Path(a.per_slide).glob(f"*/{a.nerve}.png"),
                   key=lambda p: int(re.search(r"(\d+)$", p.parent.name).group(1)))
    if not files:
        files = sorted(Path(a.per_slide).glob(f"{a.nerve}_U*.png"),
                       key=lambda p: int(re.search(r"(\d+)$", p.stem).group(1)))
    if not files:
        raise SystemExit(f"no {a.nerve}.png under {a.per_slide}/<section>/ and no {a.nerve}_U*.png in it")
    ims = [cv2.imread(str(f)) for f in files]
    pw, ph = max(i.shape[1] for i in ims), max(i.shape[0] for i in ims)
    gap, top = 8, 44
    nrow = int(np.ceil(len(ims) / a.ncol))
    sheet = np.full((top + nrow * (ph + gap) + gap, a.ncol * (pw + gap) + gap, 3), 255, np.uint8)
    sample = a.sample or files[0].parent.name.rsplit('_', 1)[0]
    title = f"{sample}  {a.nerve}: H&E ROI on every section ({len(ims)} sections, 1 um/px, same scale)"
    cv2.putText(sheet, title, (gap, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2, cv2.LINE_AA)
    for k, im in enumerate(ims):
        r, c = divmod(k, a.ncol)
        y, x = top + gap + r * (ph + gap), gap + c * (pw + gap)
        sheet[y:y + im.shape[0], x:x + im.shape[1]] = im
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(a.out, sheet)
    print(f"wrote {a.out}: {len(ims)} panels {pw}x{ph}, sheet {sheet.shape[1]}x{sheet.shape[0]}")


if __name__ == "__main__":
    main()
