#!/usr/bin/env python3

import argparse, json, time
from pathlib import Path
import numpy as np
from PIL import Image


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--volume", required=True); ap.add_argument("--swap-volume", required=True, dest="swap_volume")
    ap.add_argument("--quality", type=int, default=90); ap.add_argument("--method", type=int, default=6)
    a = ap.parse_args()
    V, SV = Path(a.volume), Path(a.swap_volume)
    vm = json.loads((V / "metadata.json").read_text()); svm = json.loads((SV / "metadata.json").read_text())
    if svm["canvas_px"] != vm["canvas_px"] or abs(float(svm["in_plane_um_per_px"]) - float(vm["in_plane_um_per_px"])) > 1e-9:
        raise SystemExit(f"{SV} is not on {V}'s canvas ({svm['canvas_px']} vs {vm['canvas_px']})")
    t0 = time.time(); planes = []; sections = []
    for bk in vm["encodings"]:
        by = {q["section_id"]: q for q in vm["encodings"][bk]["planes"]}
        d = V / bk / "he_swap"; d.mkdir(parents=True, exist_ok=True)
        for q in svm["encodings"][bk]["planes"]:
            s = q["section_id"]
            if s not in by:
                raise SystemExit(f"{s}: in the swap arm but not in the main volume")
            im = np.asarray(Image.open(SV / q["webp_lossless"]).convert("RGBA"))
            tag = f"z{by[s]['index']:02d}_{s.split('-')[1]}"
            f_grey = d / f"{tag}.grey.q90.webp"
            Image.fromarray(im).save(f_grey, format="WEBP", quality=a.quality, method=a.method)
            planes.append({"index": by[s]["index"], "section_id": s, "basis": bk, "z_um": by[s]["z_um"],
                           "modality_replaced": q.get("modality", ""), "grey_file": f"{bk}/he_swap/{f_grey.name}",
                           "grey_bytes": int(f_grey.stat().st_size), "source": str(SV / q["webp_lossless"])})
            if bk == "G_withdrawn":
                sections.append({"section_id": s, "index": by[s]["index"], "modality": q.get("modality", "")})
    meta = {"what": "second-modality planes on the main volume's canvas (the swap arm), one grey WebP per plane and basis",
            "arm": "swap", "swap_volume": str(SV), "n_sections": len(sections), "sections": sections, "planes": planes,
            "elapsed_s": round(time.time() - t0, 1)}
    (V / "he_swap_metadata.json").write_text(json.dumps(meta, indent=1))
    print(f"wrote {V / 'he_swap_metadata.json'}: {len(sections)} sections x {len(vm['encodings'])} bases ({meta['elapsed_s']} s)")


if __name__ == "__main__":
    main()
