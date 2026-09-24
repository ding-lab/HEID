#!/usr/bin/env python
from __future__ import annotations
import os
import json
import re
import sys
from pathlib import Path

import pandas as pd
import tifffile

D3 = Path(os.environ.get("THREED_DATA_ROOT", os.path.join(os.environ.get("PROJECTS_ROOT", "/data/heid"), "data/3d")) + "")
V1 = Path(os.environ.get("THREED_ROOT", os.path.join(os.environ.get("PROJECTS_ROOT", "/data/heid"), "3d")) + "/inference/cohort")
COHORT = "cohort"


COHORT_MPP = 0.5001807586523194


PLAUSIBLE_MPP = (0.1, 1.0)


BLOCKS_3D = ["HT913Z1", "HT891Z1", "HT704B1"]
ADDED = ["HT206B1", "HT397B1", "HT591B1", "HT814Z1", "HT817Z1", "HT832Z1", "HT849Z1", "HT852Z1",
         "SP003Z1", "SP004Z1", "SP005Z1", "SP006Z1", "SP007Z1", "SP008Z1", "SP009Z1", "SP011Z1",
         "SP012Z1", "SP014Z1", "SP015Z1", "SP016Z1", "SP017Z1", "SP048Z1", "SP050Z1", "SP051Z1"]
SAMPLE_GROUPS = [BLOCKS_3D, ADDED]
PATIENTS = BLOCKS_3D + ADDED


DELIVERY_ORDER = ["level3_manifest", "box_repository"]

EXPECTED = {"HT913Z1": 63, "HT891Z1": 50, "HT704B1": 23,
            "HT206B1": 6, "HT397B1": 2, "HT591B1": 1, "HT814Z1": 1, "HT817Z1": 1, "HT832Z1": 1,
            "HT849Z1": 1, "HT852Z1": 1, "SP003Z1": 2, "SP004Z1": 1, "SP005Z1": 1, "SP006Z1": 9,
            "SP007Z1": 1, "SP008Z1": 1, "SP009Z1": 1, "SP011Z1": 1, "SP012Z1": 1, "SP014Z1": 1,
            "SP015Z1": 1, "SP016Z1": 1, "SP017Z1": 9, "SP048Z1": 2, "SP050Z1": 1, "SP051Z1": 1}

SHARED_SCAN = re.compile(r"shared slide scan on disk:\s*(HE/_scan_pool/.+?\.qptiff)")


NOT_ONE_SECTION = re.compile(r"_Scan1\.qptiff$")


DELIVERY_FAMILY_MPP = {
    "A1_HT397B1-S1H3A1U1.tif": 0.4104082473577951,
    "B1_HT397B1-S1H3A1U21.tif": 0.4104082473577951,
}


def plausible(mpp) -> bool:
    return mpp is not None and PLAUSIBLE_MPP[0] <= mpp <= PLAUSIBLE_MPP[1]


def mpp_from_file(path: Path):
    with tifffile.TiffFile(str(path)) as tf:
        p0 = tf.series[0].levels[0].pages[0]
        desc = p0.description or ""
        m = re.search(r'PhysicalSizeX="([0-9.eE+-]+)"', desc)
        if m and plausible(float(m.group(1))):
            return float(m.group(1)), "ome_xml"
        xr, ru = p0.tags.get("XResolution"), p0.tags.get("ResolutionUnit")
        if xr is not None and ru is not None and xr.value[1]:
            ppu = xr.value[0] / xr.value[1]
            if ppu > 0 and int(ru.value) in (2, 3):
                value = 1e4 / ppu if int(ru.value) == 3 else 25400.0 / ppu
                if plausible(value):
                    return value, "tiff_tag"
    return None, None


def parent_mpp_for_crop(name: str, crop_map: dict, parents: dict):
    row = crop_map.get(name)
    if row is None:
        return None, None, None
    label = row["slide_label"]
    parent = parents.get(label)
    if parent is None:
        return None, None, label
    return parent["mpp"], "parent_qptiff", label


def shared_scan_mpp(detail: str, cache: dict):
    m = SHARED_SCAN.search(detail or "")
    if not m:
        return None, None, None
    relative = m.group(1).strip()
    path = D3 / relative
    if not path.exists():
        return None, None, None
    if relative not in cache:
        value, _ = mpp_from_file(path)
        cache[relative] = value
    if not plausible(cache[relative]):
        return None, None, None
    return cache[relative], "parent_qptiff", path.name


def load_parents() -> dict:
    smap = pd.read_csv(D3 / "documents" / "slide_section_map.tsv", sep="\t")
    out = {}
    for label, sub in smap.groupby("slide_label"):
        fname = str(sub.iloc[0]["he_file_SLIDE_LEVEL"])
        hits = list((D3 / "HE").rglob(fname))
        if not hits:
            continue
        mpp, src = mpp_from_file(hits[0])
        if mpp is None:
            continue
        with tifffile.TiffFile(str(hits[0])) as tf:
            samples = int(tf.series[0].levels[0].pages[0].samplesperpixel)
        out[label] = {"file": str(hits[0]), "mpp": mpp, "mpp_source": src, "samples": samples}
    return out


def load_ledger() -> pd.DataFrame:
    led = pd.read_csv(D3 / "documents" / "he_ledger.tsv", sep="\t", dtype=str)
    led.columns = [c.strip() for c in led.columns]
    for col in led.columns:
        led[col] = led[col].str.strip()
    led = led[led.sample_id.isin(PATIENTS) & (led.he_on_disk_persection == "yes")].copy()

    led["files"] = led.he_persection_file.str.split(";")
    led["n_files_on_row"] = led.files.map(len)
    led = led.explode("files").rename(columns={"files": "he_file"})
    led["he_file"] = led.he_file.str.strip()
    led["path"] = [D3 / p for p in led.he_file]

    missing = [str(p) for p in led.path if not p.exists()]
    if missing:
        sys.exit(f"FATAL: ledger lists {len(missing)} files that are not on disk: {missing[:3]}")
    unknown = sorted(set(led.he_persection_source) - set(DELIVERY_ORDER))
    if unknown:
        sys.exit(f"FATAL: unhandled delivery channel(s) {unknown}; add them to DELIVERY_ORDER "
                 "so their slides are ordered explicitly rather than dropped")

    shared = led[led.duplicated("he_file", keep=False)]
    for name, part in shared.groupby("he_file"):
        if part.sample_id.nunique() > 1:
            sys.exit(f"FATAL: {name} is claimed by more than one sample "
                     f"({sorted(part.sample_id.unique())}); the ledger cannot place it")
    led = led.drop_duplicates("he_file", keep="first")

    whole_slide = led.he_file.str.contains(NOT_ONE_SECTION)
    load_ledger.skipped = sorted(led.loc[whole_slide, "he_file"])
    return led[~whole_slide]


def crop_blocks() -> dict:
    smap = pd.read_csv(D3 / "documents" / "slide_section_map.tsv", sep="\t")
    return {str(r.he_section_file): str(r.block).upper() for r in smap.itertuples()
            if isinstance(r.he_section_file, str)}


def probe(sample: str, path: Path, block: str, delivery: str, restained: bool,
          crop_map: dict, parents: dict, detail: str = "",
          shared_cache: dict | None = None) -> dict:
    with tifffile.TiffFile(str(path)) as tf:
        s = tf.series[0]
        p0 = s.levels[0].pages[0]
        height, width = int(p0.imagelength), int(p0.imagewidth)
        n_levels = len(s.levels)
        tiled = bool(p0.is_tiled)
        samples = int(p0.samplesperpixel)

    mpp, src = mpp_from_file(path)
    parent_label = None
    if mpp is None and ".he_from_" in path.name:
        mpp, src, parent_label = parent_mpp_for_crop(path.name, crop_map, parents)
    if mpp is None:
        mpp, src, parent_label = shared_scan_mpp(detail, shared_cache if shared_cache
                                                 is not None else {})
    if mpp is None and path.name in DELIVERY_FAMILY_MPP:
        mpp, src = DELIVERY_FAMILY_MPP[path.name], "delivery_family"
    if mpp is None:
        mpp, src = COHORT_MPP, "cohort_const"

    return dict(slide=path.name.split(".")[0], svs=str(path), native_mpp=mpp,
                appmag=None, level0_W=width, level0_H=height,
                sample=sample, source_dir=path.parent.name, mpp_source=src,
                parent_slide=parent_label, n_levels=n_levels, tiled=tiled,
                samples_per_pixel=samples, file_name=path.name,
                block=block, he_delivery=delivery, restained=restained)


def main() -> None:
    crop_map = {}
    cm = D3 / "HE" / "HT891Z1" / "crop_manifest.tsv"
    if cm.exists():
        for _, r in pd.read_csv(cm, sep="\t").iterrows():
            crop_map[str(r["he_section_file"])] = r
    parents = load_parents()
    ledger = load_ledger()
    blocks = crop_blocks()
    shared_cache: dict = {}

    rows, keys = [], []
    for group in SAMPLE_GROUPS:
        for delivery in DELIVERY_ORDER:
            for sample in group:
                sub = ledger[(ledger.he_persection_source == delivery)
                             & (ledger.sample_id == sample)]
                for r in sorted(sub.itertuples(), key=lambda r: r.path.name):
                    rows.append(probe(sample, r.path, str(r.block_id).upper(), delivery,
                                      False, crop_map, parents, r.detail, shared_cache))


                    keys.append((sample, str(r.block_id).upper(), r.section_u,
                                 r.path.name if r.n_files_on_row > 1 else ""))
            if delivery == "level3_manifest" and group is BLOCKS_3D:


                for path in sorted((D3 / "HE" / "HT891Z1").glob("HT891Z1-*.he_from_*.ome.tif")):
                    rows.append(probe("HT891Z1", path, blocks.get(path.name, ""),
                                      "codex_restain_crop", True, crop_map, parents))
                    keys.append(("HT891Z1", blocks.get(path.name, ""),
                                 str(u_number(path.name)), ""))

    table = pd.DataFrame(rows)

    dup = table.slide[table.slide.duplicated()].tolist()
    if dup:
        sys.exit(f"FATAL: slide id is not unique: {dup[:5]}")
    seen = pd.Series(keys)
    dup_key = seen[seen.duplicated()].tolist()
    if dup_key:
        sys.exit(f"FATAL: same (patient, block, section) delivered twice: {dup_key[:5]}")
    counts = table["sample"].value_counts().to_dict()
    if counts != EXPECTED:
        only_here = {k: v for k, v in counts.items() if EXPECTED.get(k) != v}
        sys.exit(f"FATAL: slide counts differ from expected on {only_here} "
                 f"(expected {({k: EXPECTED.get(k) for k in only_here})})")
    if (table.samples_per_pixel != 3).any():
        sys.exit("FATAL: non-RGB slides present")
    if table.block.eq("").any():
        sys.exit(f"FATAL: no block for {table.loc[table.block.eq(''),'slide'].tolist()[:5]}")
    off_scale = table.loc[(table.native_mpp < PLAUSIBLE_MPP[0])
                          | (table.native_mpp > PLAUSIBLE_MPP[1]), "slide"].tolist()
    if off_scale:
        sys.exit(f"FATAL: implausible pixel size on {off_scale[:5]}")

    out = V1 / "configs" / COHORT / "run_manifest.tsv"
    out.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(out, sep="\t", index=False)

    prov = {"cohort": COHORT, "n_slides": int(len(table)),
            "by_sample": counts,
            "by_delivery": table.he_delivery.value_counts().to_dict(),
            "by_block": table.groupby(["sample", "block"]).size()
                             .rename("n").reset_index().to_dict("records"),
            "n_restained": int(table.restained.sum()),
            "discovery": "he_ledger.tsv he_persection_file for HE/, directory for HE_sections",
            "mpp_source_counts": table.mpp_source.value_counts().to_dict(),
            "native_mpp_values": {str(round(k, 10)): int(v)
                                  for k, v in table.native_mpp.value_counts().items()},
            "cohort_const_slides": table.loc[table.mpp_source == "cohort_const",
                                             "slide"].tolist(),


            "n_untiled": int((~table.tiled).sum()),
            "untiled_slides": table.loc[~table.tiled, "slide"].tolist(),
            "n_single_level": int((table.n_levels == 1).sum()),
            "parents": parents,
            "shared_scans_read": shared_cache,
            "unrun_ledger_files": getattr(load_ledger, "skipped", []),
            "total_level0_gigapixels": round(
                float((table.level0_W * table.level0_H).sum()) / 1e9, 3)}
    (V1 / "configs" / COHORT / "manifest_provenance.json").write_text(json.dumps(prov, indent=1))

    print(f"{len(table)} slides -> {out}")
    print("by sample :", counts)
    print("by channel:", prov["by_delivery"])
    print("by block  :", table.groupby(["sample", "block"]).size().to_dict())
    print("restained :", prov["n_restained"])
    print("mpp source:", table.mpp_source.value_counts().to_dict())
    print("native_mpp:", table.native_mpp.value_counts().to_dict())
    print("n_levels  :", table.n_levels.value_counts().to_dict())
    print("tiled     :", table.tiled.value_counts().to_dict())
    print(f"level-0 total: {prov['total_level0_gigapixels']} Gpx")


def u_number(file_name: str) -> int:
    m = re.search(r"_?U(\d+)", file_name.split(".")[0])
    if not m:
        raise ValueError(f"no U number in {file_name}")
    return int(m.group(1))


if __name__ == "__main__":
    main()
