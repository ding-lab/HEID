
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
import os
VOL = Path(os.environ.get("HTAN3D_VOL") or ROOT / "outputs" / "volume_8um")
META = VOL / "cells_metadata.json"
POINT_META = VOL / "cell_points_metadata.json"
ALIGN = ROOT.parent / "xenium" / "outputs"
FRAGMENT = ALIGN / "xenium9_cells_fragment.json"
POINT_FRAGMENT = ALIGN / "xenium9_cell_points_fragment.json"
TEXTURES = ALIGN / "cells_planes"


ROUTE = "xenium_pointcloud"


def sha256(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def copy_in(rel: str, declared_bytes: int, dry: bool) -> str | None:
    src, dst = TEXTURES / rel, VOL / rel
    if dry:

        if not src.exists():
            return f"{rel} (source absent)"
        return (None if src.stat().st_size == declared_bytes
                else f"{rel} (source {src.stat().st_size} != {declared_bytes})")
    if not dst.exists():
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)
    if not dst.exists():
        return rel
    if dst.stat().st_size != declared_bytes:
        return f"{rel} ({dst.stat().st_size} != {declared_bytes})"
    if dst.read_bytes() != src.read_bytes():
        return f"{rel} (copy differs from the source byte for byte)"
    return None


def merge_points(a) -> None:
    frag_path = Path(a.points_fragment)
    if not frag_path.exists():
        print("no per-cell fragment; skipping that half")
        return
    frag = json.loads(frag_path.read_text())
    pm = json.loads(POINT_META.read_text())

    have = {(f["basis"], f["index"]) for f in pm["files"]}
    new = {(f["basis"], f["index"]) for f in frag["files"]}
    if new & have:
        print(f"per-cell: nothing to do, {len(new & have)} of {len(new)} already there")
        return
    if frag["classes"] != pm["classes"]:
        raise SystemExit("per-cell: class order differs from the published one; the "
                         "class byte is an index into that list")
    if (frag["canvas_px"] != pm["canvas_px"]
            or frag["in_plane_um_per_px"] != pm["in_plane_um_per_px"]):
        raise SystemExit("per-cell: canvas or pitch differs from the published one")

    bad = [b for b in (copy_in(f["file"], f["bytes"], a.dry_run)
                       for f in frag["files"]) if b]
    if bad:
        raise SystemExit(f"{len(bad)} per-cell files missing or altered: {bad[:3]}")

    pm["files"] = sorted(pm["files"] + frag["files"],
                         key=lambda f: (f["basis"], f["index"]))
    pm["sections"] = sorted(pm["sections"] + frag["sections"],
                            key=lambda s: s["index"])
    pm["checks"] = sorted(pm["checks"] + frag["checks"],
                          key=lambda c: (c["basis"], c["index"]))
    pm["n_planes"] = len(pm["sections"])
    pm["bytes_total"] = sum(f["bytes"] for f in pm["files"])
    pm["worst_in_mask"] = sorted(pm["checks"], key=lambda c: c["in_mask"])[:6]
    pm.setdefault("merged_from", []).append({
        "fragment": str(frag_path), "sha256": sha256(frag_path),
        "planes": sorted({f["index"] for f in frag["files"]}),
        "files": len(frag["files"]),
        "geometry": frag["geometry"],
        "writer_selfcheck": frag["writer_selfcheck"],
    })
    if not a.dry_run:
        POINT_META.write_text(json.dumps(pm, indent=1) + "\n")
    inm = sorted(pm["checks"], key=lambda c: c["in_mask"])
    print(f"per-cell: {pm['n_planes']} planes, {len(pm['files'])} files, "
          f"{pm['bytes_total'] / 1e6:.1f} MB, worst in_mask {inm[0]['in_mask']:.4f} "
          f"({inm[0]['section_id']} / {inm[0]['basis']})")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--fragment", default=str(FRAGMENT))
    ap.add_argument("--points-fragment", default=str(POINT_FRAGMENT))
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()

    merge_points(a)

    frag_path = Path(a.fragment)
    frag = json.loads(frag_path.read_text())
    cm = json.loads(META.read_text())

    new_idx = sorted(frag["indices"])
    already = set(cm.get("indices_with", []))
    if set(new_idx) & already:
        print(f"nothing to do: {sorted(set(new_idx) & already)} already predicted")
        return
    missing = set(new_idx) - set(cm.get("indices_without", []))
    if missing:
        raise SystemExit(f"fragment brings planes this file does not list as "
                         f"unpredicted: {sorted(missing)}")


    bad = [b for b in (copy_in(p["file"], p["bytes"], a.dry_run)
                       for p in frag["planes"]) if b]
    if bad:
        raise SystemExit(f"{len(bad)} textures missing or altered: {bad[:3]}")


    keep = [q for q in cm["planes"] if q["cell_class"] is not None]
    dropped = len(cm["planes"]) - len(keep)
    cm["planes"] = sorted(keep + frag["planes"],
                          key=lambda q: (q["basis"], q["index"], q["cell_class"]))
    cm["sections"] = sorted(cm["sections"] + frag["sections"], key=lambda s: s["index"])
    cm["checks"] = sorted(cm["checks"] + frag["checks"],
                          key=lambda c: (c["basis"], c["index"]))
    cm["indices_with"] = sorted(already | set(new_idx))
    cm["n_planes_with_prediction"] = len(cm["indices_with"])


    for dead in ("n_planes_without", "indices_without", "without_reason",
                 "nopred_colour", "hatch"):
        cm.pop(dead, None)


    cm["routes"] = {
        "he": "the section's own H&E, predicted and placed by this reconstruction's "
              "own chain",
        "he_swap": "a section imaged twice; predicted on the H&E scanned after the "
                   "CODEX run and mapped into the plane grid by that build's matrix",
        ROUTE: frag["geometry"]["steps"] + " -- the fitted link is "
               + frag["geometry"]["fitted_link"],
    }
    cm["merged_from"] = {
        "fragment": str(frag_path),
        "sha256": sha256(frag_path),
        "planes": sorted(new_idx),
        "textures": len(frag["planes"]),
        "texture_bytes": frag["texture_bytes"],
        "normalisation": frag["normalisation_source"],
    }
    cm["pairing"] = {
        "what": "four of these nine sections' H&E crops carry the wrong U number. "
                "Measured, not read from file names: all nine H&E-by-Xenium "
                "combinations were fitted within each slide and the assignment came "
                "out non-diagonal for two pairs.",
        "pairs": {"H&E U1": "Xenium U32", "H&E U32": "Xenium U1",
                  "H&E U44": "Xenium U69", "H&E U69": "Xenium U44"},
        "which_side_is_wrong": frag["pairing_provenance"]["result"],
        "z_positions": "unaffected on that finding: a plane's identity is the DAPI "
                       "image it carries, and the Xenium region names are the ones "
                       "this volume's z assignment uses. If that finding is ever "
                       "overturned in favour of the H&E names, planes 0, 10, 16 and 27 "
                       "change z and the stack order changes with them.",
        "status": "the pairing is measured and the direction is argued from serial "
                  "position; neither has been checked by anyone other than the "
                  "producer of xenium/.",
        "evidence": frag["pairing_provenance"]["evidence"],
        "blind_spot": frag["checks_cannot_detect"]["what"] + " "
                      + frag["checks_cannot_detect"]["what_did_detect_it"],
    }
    cm["checks_note"] += (
        " control_wrong_section_in_mask / _iou: the same cells against a DIFFERENT "
        "Xenium section's tissue. Shape-based checks cannot see a wrong-section "
        "pairing -- serial sections share a silhouette -- so the nine point-cloud "
        "planes also carry a per-object statistic, the median H&E-nucleus to "
        "Xenium-nucleus distance, which can.")

    inm = sorted(cm["checks"], key=lambda c: c["in_mask"])
    cm["worst_in_mask"] = inm[:6]

    if a.dry_run:
        print("dry run, nothing written")
    else:
        META.write_text(json.dumps(cm, indent=1) + "\n")

    print(f"planes with a prediction: {cm['n_planes_with_prediction']} of "
          f"{len(cm['sections'])} sections")
    print(f"plane rows {len(cm['planes'])} ({dropped} hatch rows dropped), "
          f"checks {len(cm['checks'])}, textures merged {len(frag['planes'])}")
    print(f"worst in_mask now {inm[0]['in_mask']:.4f} "
          f"({inm[0]['section_id']} / {inm[0]['basis']})")


if __name__ == "__main__":
    main()
