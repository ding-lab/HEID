#!/usr/bin/env python3
import re, json, sys, os, glob, collections, struct
from PIL import Image
sd, root = sys.argv[1], sys.argv[2]
s = open(sd).read()
SRC = json.loads(re.search(r"window\.__SOURCE__\s*=\s*(.*?);\n", s, re.S).group(1))
META = json.loads(re.search(r"window\.__META__\s*=\s*(.*?);\n", s, re.S).group(1))
page_dir = os.path.join(root, "samples/viewer")
def P(rel): return os.path.normpath(os.path.join(page_dir, rel))
def dims(files):
    c = collections.Counter()
    for f in files:
        try: c[Image.open(f).size] += 1
        except Exception: c["ERR"] += 1
    return dict(c)
bad = 0
def report(name, want, got, extra=""):
    global bad
    ok = all(k == want for k in got) if got else True
    if not ok: bad += 1
    print(f"{'OK ' if ok else 'BAD'} {name:42s} want {want} got {got} {extra}")
print("page META canvas_px", META["canvas_px"])
for r, x in SRC["res"].items():
    d = P(x["dir"]); m = json.load(open(d + "/metadata.json")); want = (m["canvas_px"]["width"], m["canvas_px"]["height"])
    print(f"--- res {r}: dir {x['dir']} page says {x['canvas_px']} disk metadata {m['canvas_px']} origin {m.get('canvas_origin_um')}")
    if (x["canvas_px"]["width"], x["canvas_px"]["height"]) != want: bad += 1; print("BAD page canvas != disk metadata")
    for b, fs in x["files"].items(): report(f"res{r} grey {b}", want, dims(d + "/" + f for f in fs))
    for b, fs in (x.get("he_rgb") or {}).items(): report(f"res{r} colour {b}", want, dims(d + "/" + f for f in fs.values()))
    sw = x.get("swap")
    if sw:
        sdir = P(sw["dir"]); sm = json.load(open(sdir + "/metadata.json")); swant = (sm["canvas_px"]["width"], sm["canvas_px"]["height"])
        print(f"    swap arm dir {sw['dir']} disk metadata {sm['canvas_px']} origin {sm.get('canvas_origin_um')} same_geometry {sw.get('same_geometry')}")
        if swant != want: bad += 1; print("BAD swap arm canvas != main")
        for b, fs in sw["files"].items(): report(f"res{r} swap grey {b}", want, dims(sdir + "/" + f for f in fs))
        for b, fs in (sw.get("he_rgb") or {}).items(): report(f"res{r} swap colour {b}", want, dims(sdir + "/" + f for f in fs.values()))
m8 = SRC["res"][SRC.get("default_res", "8")]; d8 = P(m8["dir"]); mm = json.load(open(d8 + "/metadata.json")); want = (mm["canvas_px"]["width"], mm["canvas_px"]["height"])
cells = SRC.get("cells")
if cells:
    cd = P(cells["dir"]); print(f"--- cells: dir {cells['dir']} page canvas {cells.get('canvas_px')}")
    if cells.get("canvas_px") and (cells["canvas_px"]["width"], cells["canvas_px"]["height"]) != want: bad += 1; print("BAD cells canvas != volume")
    for b in ("G_withdrawn", "G_not_withdrawn"):
        fs = glob.glob(f"{cd}/{b}/cells/*.webp")
        if fs: report(f"cells {b} ({len(fs)} files)", want, dims(fs))
cp = SRC.get("cell_points")
if cp:
    cd = P(cp["dir"]); sizes = collections.Counter()
    for b, fs in cp["files"].items():
        for f in fs.values():
            with open(cd + "/" + f, "rb") as fh: h = fh.read(32)
            mag, n, W, H, sub, ncls, _ = struct.unpack("<8sIHHHHI", h[:24]); sizes[(W, H)] += 1
    report("cell_points headers", want, dict(sizes))
for key in ("tls_regions", "duct_regions", "nerve_regions", "tumor_regions", "nerve_lines", "cells_denoised", "cells_denoised3d"):
    L = SRC.get(key)
    if not L: continue
    ld = P(L["dir"]); fs = []
    for p in L.get("planes", []):
        for k in ("file", "file3d", "file2d"):
            if p.get(k):
                for b in ("G_withdrawn", "G_not_withdrawn"):
                    sub = {"tls_regions": "tls", "duct_regions": "duct", "nerve_regions": "nerve", "tumor_regions": "tumor", "nerve_lines": "nline"}.get(key, key)
                    g = glob.glob(f"{ld}/{b}/*/{p[k]}")
                    fs += g
    if fs: report(f"{key} ({len(fs)} files)", want, dims(fs))
xg = SRC.get("xenium_gt")
if xg:
    xd = P(xg["dir"]); fs = glob.glob(f"{xd}/**/*.webp", recursive=True)
    if fs: report(f"xenium_gt ({len(fs)} files)", want, dims(fs))
tiles = SRC.get("tiles")
for tkey in ("tiles", "tiles_swap"):
    T = SRC.get(tkey)
    if not T: continue
    td = P(T["dir"]); ors = collections.Counter(); w8 = collections.Counter(); n = 0
    for b in ("G_withdrawn", "G_not_withdrawn"):
        for q in glob.glob(f"{td}/{b}/*/index.json"):
            j = json.load(open(q)); n += 1; ors[str([round(v, 1) for v in j.get("world_origin_um", [])])] += 1
            L8 = [l for l in j["levels"] if abs(l["mpp"] - 8) < 1e-6]; w8[(L8[0]["width"], L8[0]["height"]) if L8 else None] += 1
    report(f"{tkey} level-8 size ({n} indexes)", want, dict(w8), f"origins {dict(ors)}")


try:
    import numpy as np, cv2, glob, json as _json
    from PIL import Image
    src = SRC
    vol = P(src["res"]["8"]["dir"]) if "8" in src.get("res", {}) else None
    v4 = P(src["res"]["4"]["dir"]) if "4" in src.get("res", {}) else None
    tiles = P(src.get("tiles", {}).get("dir", "")) if src.get("tiles") else None
    meta = _json.load(open(os.path.join(vol, "metadata.json")))
    planes = sorted(meta["encodings"]["G_withdrawn"]["planes"], key=lambda q: q["index"])
    step = max(1, len(planes) // 6)
    worst = 0.0; n_chk = 0
    for q in planes[::step][:7]:
        sid, i = q["section_id"], q["index"]
        f8 = glob.glob(os.path.join(vol, "G_withdrawn", "webp", f"z{i:02d}_*.q90.webp"))
        if not f8: continue
        a = np.array(Image.open(f8[0]).convert("L")).astype(np.float32)
        cands = []
        if v4:
            f4 = glob.glob(os.path.join(v4, "G_withdrawn", "webp", f"z{i:02d}_*.q90.webp"))
            if f4:
                b = np.array(Image.open(f4[0]).convert("L")); cands.append(("4um plane", cv2.resize(b, (a.shape[1], a.shape[0]), interpolation=cv2.INTER_AREA).astype(np.float32)))
        if tiles and os.path.exists(os.path.join(tiles, "G_withdrawn", sid, "index.json")):
            ix = _json.load(open(os.path.join(tiles, "G_withdrawn", sid, "index.json")))
            lv = [l for l in ix["levels"] if abs(l["mpp"] - 8) < 1e-6]
            if lv:
                lv = lv[0]; ts = ix["tile_px"]; img = np.full((lv["rows"] * ts, lv["cols"] * ts), 255, np.uint8)
                for tf in glob.glob(os.path.join(tiles, "G_withdrawn", sid, "8", "*.webp")):
                    c, r = map(int, os.path.basename(tf)[:-5].split("_")); t = np.array(Image.open(tf).convert("L")); img[r*ts:r*ts+t.shape[0], c*ts:c*ts+t.shape[1]] = t
                cands.append(("tiles L8", img[:lv["height"], :lv["width"]].astype(np.float32)))
        for name, b in cands:
            h = min(a.shape[0], b.shape[0]); w = min(a.shape[1], b.shape[1])
            (dx, dy), resp = cv2.phaseCorrelate(255 - a[:h, :w], 255 - b[:h, :w])
            d = float(max(abs(dx), abs(dy))); worst = max(worst, d); n_chk += 1
            if d > 1.5:
                bad += 1; print(f"BAD placement {sid} {name}: shifted ({dx:+.1f},{dy:+.1f}) px vs the 8 um plane")
    print(f"OK  placement: {n_chk} layer checks on {min(7, len(planes[::step]))} sections, worst shift {worst:.2f} px" if worst <= 1.5 else f"BAD placement: worst shift {worst:.2f} px")
except Exception as e:
    bad += 1; print("BAD placement check failed:", e)
print("RESULT", "CONSISTENT" if bad == 0 else f"{bad} INCONSISTENT FAMILIES")
