#!/usr/bin/env python3

from __future__ import annotations
import os

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd


MODULE_ROOT = Path(__file__).resolve().parents[1]
HE_PRED = Path(os.environ.get("TLS_HE_PRED_DATA", str(Path(os.environ.get("PROJECTS_ROOT", "/data/heid")) / "tls/prediction")))
TLS = HE_PRED.parent
FEATURES = Path(os.environ.get("TLS_FEATURES", str(HE_PRED / "features")))
TILES_MAIN = HE_PRED / "data" / "xenium_5k" / "tiles"
TILES_EXCLUDED = HE_PRED / "data" / "xenium_5k" / "excluded" / "tiles"
SLIDES = MODULE_ROOT / "configs" / "slides_5k.tsv"
LABELS = Path(os.environ.get("TLS_LABELS", str(MODULE_ROOT / "data/labels")))
OUT = MODULE_ROOT / "outputs" / "results" / "input_hashes.tsv"
SUMMARY = MODULE_ROOT / "outputs" / "results" / "input_hashes.json"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def tile_path(field: str) -> Path:
    for root in (TILES_MAIN, TILES_EXCLUDED):
        path = root / field / "tiles.parquet"
        if path.is_file():
            return path
    raise FileNotFoundError(field)


def main() -> None:
    slides = pd.read_csv(SLIDES, sep="\t")
    rows = []
    for index, row in enumerate(slides.itertuples(index=False), 1):
        label = LABELS / f"{row.unit_id}_tiles.parquet"
        feature = FEATURES / str(row.field) / "uni2.pt"
        tile = tile_path(str(row.field))
        paths = {"label": label, "feature": feature, "tile_grid": tile}
        if any(not path.is_file() for path in paths.values()):
            missing = [str(path) for path in paths.values() if not path.is_file()]
            raise FileNotFoundError(missing)
        item = {"unit_id": str(row.unit_id), "field": str(row.field)}
        for name, path in paths.items():
            item[f"{name}_path"] = str(path.resolve())
            item[f"{name}_size"] = int(path.stat().st_size)
            item[f"{name}_sha256"] = sha256(path)
        rows.append(item)
        print(f"[{index:3d}/{len(slides)}] {row.unit_id}", flush=True)
    frame = pd.DataFrame(rows)
    frame.to_csv(OUT, sep="\t", index=False)
    SUMMARY.write_text(
        json.dumps(
            {
                "generated_at_utc": datetime.now(timezone.utc).isoformat(),
                "units": int(len(frame)),
                "slides_sha256": sha256(SLIDES),
                "manifest_sha256": sha256(OUT),
                "total_input_bytes": int(
                    frame[
                        ["label_size", "feature_size", "tile_grid_size"]
                    ].to_numpy().sum()
                ),
            },
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )
    print(SUMMARY.read_text())


if __name__ == "__main__":
    main()
