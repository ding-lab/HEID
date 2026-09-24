#!/usr/bin/env python3

import argparse
import json
import os
import re
import struct
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

Image.MAX_IMAGE_PIXELS = None
PAD_UM = 80.0
RASTER_DIRS = ("webp", "png", "rgb", "cells", "he_swap", "tls", "duct", "nerve", "xenium_gt")
PLANE_RE = re.compile(r"^z(\d+)_")


def compose(A, B):
    return (np.vstack([A, [0, 0, 1]]) @ np.vstack([B, [0, 0, 1]]))[:2]


def invert(A):
    return np.linalg.inv(np.vstack([A, [0, 0, 1]]))[:2]


def is_identity(M, tol=1e-9):
    return np.allclose(M, [[1, 0, 0], [0, 1, 0]], atol=tol)


def warp_matrix(D, o_old, o_new, mpp):
    S_old = np.array([[mpp, 0, o_old[0]], [0, mpp, o_old[1]]], float)
    S_new_inv = np.array([[1 / mpp, 0, -o_new[0] / mpp], [0, 1 / mpp, -o_new[1] / mpp]], float)
    return compose(S_new_inv, compose(D, S_old))


def warp_image(arr, M, size):
    W, H = size
    if arr.ndim == 2:
        return cv2.warpAffine(arr, M, (W, H), flags=cv2.INTER_LINEAR, borderValue=0)
    C = arr.shape[2]
    out = np.zeros((H, W, C), np.uint8)
    alpha = C in (2, 4)
    for c in range(C):
        flag = cv2.INTER_NEAREST if (alpha and c == C - 1) else cv2.INTER_LINEAR
        out[:, :, c] = cv2.warpAffine(np.ascontiguousarray(arr[:, :, c]), M, (W, H), flags=flag, borderValue=0)
    return out


def _do_raster(job):
    path, M, size, lossless_cells = job
    p = Path(path)
    try:
        im = Image.open(p)
        im.load()
    except Exception as e:
        print(f"  skipped (unreadable): {p} ({e})", flush=True)
        return path, None
    mode = im.mode
    arr = np.asarray(im)
    out = warp_image(arr, np.asarray(M, float), tuple(size))
    im2 = Image.fromarray(out, mode=mode)
    name = p.name.lower()
    if name.endswith(".png"):
        im2.save(p, format="PNG")
    elif ".lossless." in name or (p.parent.name == "cells" and lossless_cells):
        im2.save(p, format="WEBP", lossless=True, method=4)
    else:
        im2.save(p, format="WEBP", quality=90, method=4)
    return path, p.stat().st_size


def _do_points(job):
    path, D, o_old, o_new, W, H = job
    b = bytearray(Path(path).read_bytes())
    magic, n, cw, ch, sub, ncls = struct.unpack_from("<8sIHHHH", b, 0)
    assert magic == b"CELLPT01", path
    xy = np.frombuffer(bytes(b[32:32 + 4 * n]), np.uint16).reshape(n, 2).astype(np.float64) / sub
    r = np.frombuffer(bytes(b[32 + 4 * n:32 + 5 * n]), np.uint8).astype(np.float64) / sub

    mpp = 8.0
    world = xy * mpp + np.asarray(o_old)
    D = np.asarray(D, float)
    moved = world @ D[:, :2].T + D[:, 2]
    new = (moved - np.asarray(o_new)) / mpp
    scale = float(np.sqrt(abs(np.linalg.det(D[:, :2]))))
    q = np.clip(np.round(new * sub), 0, 65535).astype(np.uint16)
    rq = np.clip(np.round(r * scale * sub), 0, 255).astype(np.uint8)
    struct.pack_into("<8sIHHHH", b, 0, magic, n, int(W), int(H), sub, ncls)
    b[32:32 + 4 * n] = q.tobytes()
    b[32 + 4 * n:32 + 5 * n] = rq.tobytes()
    Path(path).write_bytes(bytes(b))
    return path, len(b)


def plane_index_of(name):
    m = PLANE_RE.match(name)
    return int(m.group(1)) if m else None


def update_json_canvas(obj, W, H, o_new, mpp, sizes):
    if isinstance(obj, dict):
        if "canvas_px" in obj and isinstance(obj["canvas_px"], dict):
            obj["canvas_px"] = {"width": int(W), "height": int(H)}
        if ("canvas_origin_um" in obj and isinstance(obj["canvas_origin_um"], list)) or ("canvas_px" in obj and "encodings" in obj):
            obj["canvas_origin_um"] = [float(o_new[0]), float(o_new[1])]
        if "canvas_mm" in obj and isinstance(obj["canvas_mm"], dict):
            obj["canvas_mm"] = {"width": round(W * mpp / 1000, 4), "height": round(H * mpp / 1000, 4)}
        for k in list(obj):
            v = obj[k]
            if isinstance(v, str) and v in sizes:
                bk = k + "_bytes" if k not in ("file", "grey_file", "rgb_file") else "bytes"
                if k == "file" and "bytes" in obj:
                    obj["bytes"] = sizes[v]
                elif k == "grey_file" and "grey_bytes" in obj:
                    obj["grey_bytes"] = sizes[v]
                elif k == "rgb_file" and "rgb_bytes" in obj:
                    obj["rgb_bytes"] = sizes[v]
                elif bk in obj:
                    obj[bk] = sizes[v]
            else:
                update_json_canvas(v, W, H, o_new, mpp, sizes)
    elif isinstance(obj, list):
        for v in obj:
            update_json_canvas(v, W, H, o_new, mpp, sizes)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--volume", required=True)
    ap.add_argument("--corrections", required=True)
    ap.add_argument("--canvas-from", default="", dest="canvas_from")
    ap.add_argument("--extra-dirs", default="", dest="extra",
                    help="other directories holding CELLPT01 bins / z##_ rasters on this volume's canvas")
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--skip-dirs", default="", dest="skip",
                    help="subdirectories already on the new placement (e.g. webp,png when the grey volume was rebuilt): left alone")
    ap.add_argument("--old-origin", default="", dest="old_origin",
                    help="ox,oy world um of canvas pixel (0,0) for a volume whose metadata predates the canvas contract")
    ap.add_argument("--assume-unapplied", action="store_true", dest="unapplied",
                    help="treat every product as built on the chain as solved, whatever metadata says was applied")
    a = ap.parse_args()
    V = Path(a.volume)
    meta = json.loads((V / "metadata.json").read_text())
    mpp = float(meta["in_plane_um_per_px"])
    if a.old_origin:
        o_old = np.asarray([float(v) for v in a.old_origin.split(",")], float)
    elif meta.get("canvas_origin_um"):
        o_old = np.asarray(meta["canvas_origin_um"], float)
    else:
        raise SystemExit(f"{V}: metadata has no canvas_origin_um; pass --old-origin ox,oy (the tile index world_origin_um for the 8 um volume)")
    W_old, H_old = meta["canvas_px"]["width"], meta["canvas_px"]["height"]
    corr = json.loads(Path(a.corrections).read_text())
    T_new = {s: np.asarray(m, float) for s, m in corr["sections"].items()}
    prev = {} if a.unapplied else (meta.get("section_corrections_applied") or {})
    T_old = {s: np.asarray(m, float) for s, m in (prev.get("sections") or {}).items()}
    bases = list(meta["encodings"])
    planes = {p["index"]: p["section_id"] for p in meta["encodings"][bases[0]]["planes"]}
    I2 = np.array([[1.0, 0, 0], [0, 1.0, 0]])
    D = {}
    for i, s in planes.items():
        D[i] = compose(T_new.get(s, I2), invert(T_old.get(s, I2)))


    R_new = {s: np.asarray(m, float) for s, m in (corr.get("swap_to_main_residual") or {}).items()}
    R_cells = set(corr.get("swap_frame_cells") or [])
    R_old = {} if a.unapplied else {s: np.asarray(m, float) for s, m in (meta.get("arm_residual_applied") or {}).items()}
    DR = {i: compose(R_new.get(s, I2), invert(R_old.get(s, I2))) for i, s in planes.items()}
    HE_DIRS = {"he_swap", "rgb"}; CELL_DIRS = {"cells", "cell_points"}
    def delta_for(sub, i):
        if sub in HE_DIRS:
            return D[i]
        if sub in CELL_DIRS:
            return compose(DR[i], D[i]) if planes[i] in R_cells else D[i]
        return compose(DR[i], D[i])
    moving = [i for i in D if not is_identity(D[i]) or not is_identity(DR[i])]
    print(f"{V.name}: {len(planes)} planes, {len(moving)} move; canvas now {W_old} x {H_old} px at {mpp:g} um, origin {o_old.round(1).tolist()}", flush=True)


    if a.canvas_from:
        om = json.loads((Path(a.canvas_from) / "metadata.json").read_text())


        o_new = np.asarray(om["canvas_origin_um"], float); f = float(om["in_plane_um_per_px"]) / mpp
        W_new, H_new = int(round(om["canvas_px"]["width"] * f)), int(round(om["canvas_px"]["height"] * f))
        print(f"  canvas from {a.canvas_from}: {W_new} x {H_new} at {o_new.round(1).tolist()} (x{f:g})", flush=True)
    else:


        pts = []
        for b in bases:
          for p in meta["encodings"][b]["planes"]:
            al = np.asarray(Image.open(V / p["webp_lossless"]).convert("RGBA"))[..., 3] > 0
            ys, xs = np.nonzero(al)
            if not len(ys):
                continue
            c = np.array([[xs.min(), ys.min()], [xs.max() + 1, ys.min()], [xs.min(), ys.max() + 1], [xs.max() + 1, ys.max() + 1]], float) * mpp + o_old
            Di = compose(DR[p["index"]], D[p["index"]])
            pts.append(c @ Di[:, :2].T + Di[:, 2])
        pts = np.vstack(pts)
        lo = pts.min(0) - PAD_UM; hi = pts.max(0) + PAD_UM
        o_new = lo
        W_new = int(np.ceil((hi[0] - lo[0]) / mpp)); H_new = int(np.ceil((hi[1] - lo[1]) / mpp))
        print(f"  new canvas {W_new} x {H_new} px, origin {o_new.round(1).tolist()}", flush=True)
    same_canvas = (W_new, H_new) == (W_old, H_old) and np.allclose(o_new, o_old, atol=1e-6)
    if not moving and same_canvas:
        print("  nothing moves and the canvas is unchanged: no-op", flush=True); return


    cells_meta = V / "cells_metadata.json"
    lossless_cells = "lossless" in json.dumps(json.loads(cells_meta.read_text()).get("encoding", "")).lower() if cells_meta.exists() else False
    rjobs, pjobs = [], []
    roots = [V / b for b in bases] + [Path(d) for d in a.extra.split(",") if d.strip()]
    for root in roots:
        if not root.exists():
            continue
        skip = {x.strip() for x in a.skip.split(",") if x.strip()}
        for sub in sorted(os.listdir(root)):
            d = root / sub
            if not d.is_dir() or sub in skip:
                continue
            for f in sorted(os.listdir(d)):
                i = plane_index_of(f)
                if i is None or i not in D:
                    continue
                fp = d / f
                Di = delta_for(sub, i)
                if f.endswith(".bin"):
                    pjobs.append((str(fp), Di.tolist(), o_old.tolist(), o_new.tolist(), W_new, H_new))
                elif f.lower().endswith((".webp", ".png")):
                    if is_identity(Di) and same_canvas:
                        continue
                    rjobs.append((str(fp), warp_matrix(Di, o_old, o_new, mpp).tolist(), (W_new, H_new), lossless_cells))
    print(f"  {len(rjobs)} rasters to warp, {len(pjobs)} point tables to transform", flush=True)
    sizes = {}
    with ProcessPoolExecutor(max_workers=a.workers) as ex:
        for path, nb in ex.map(_do_raster, rjobs, chunksize=4):
            if nb is not None:
                sizes[os.path.relpath(path, V)] = nb
        for path, nb in ex.map(_do_points, pjobs):
            sizes[os.path.relpath(path, V)] = nb

    for b in bases:
        enc = meta["encodings"][b]
        if "atlas_png" not in enc or "png" in a.skip.split(","):
            continue
        rows, cols = enc["atlas_png"]["grid_rows_cols"]
        atlas = np.zeros((rows * H_new, cols * W_new, 2), np.uint8)
        for p in enc["planes"]:
            la = np.asarray(Image.open(V / p["png"]).convert("LA"))
            rr, cc = divmod(p["index"], cols)
            atlas[rr * H_new:(rr + 1) * H_new, cc * W_new:(cc + 1) * W_new] = la
        ap_ = V / enc["atlas_png"]["file"]
        Image.fromarray(atlas, mode="LA").save(ap_, format="PNG")
        sizes[os.path.relpath(ap_, V)] = ap_.stat().st_size
        enc["atlas_png"]["bytes"] = sizes[os.path.relpath(ap_, V)]

    for mj in sorted(V.glob("*.json")):
        if ".bak" in mj.name:
            continue
        obj = json.loads(mj.read_text())
        update_json_canvas(obj, W_new, H_new, o_new, mpp, sizes)
        if mj.name in ("cells_metadata.json", "cell_points_metadata.json"):


            obj["checks_repositioned"] = True
        if mj.name == "metadata.json":
            obj["section_corrections_applied"] = {"file": str(Path(a.corrections).resolve()),
                                                  "chain_grid_um": float(corr.get("chain_grid_um", 2.0)),
                                                  "sections": corr["sections"]}
            obj["arm_residual_applied"] = {s: m.tolist() for s, m in R_new.items()}
            for b in bases:
                enc = obj["encodings"][b]
                for key in ("webp_lossless", "per_plane_png"):
                    if key in enc and isinstance(enc[key], dict) and "bytes" in enc[key]:
                        fk = "webp_lossless" if key == "webp_lossless" else "png"
                        enc[key]["bytes"] = int(sum(q.get(fk + "_bytes", 0) for q in enc["planes"]))
        mj.write_text(json.dumps(obj, indent=1))
    print(f"  done: {len(sizes)} files rewritten; metadata canvas {W_new} x {H_new}, origin {o_new.round(3).tolist()}", flush=True)


if __name__ == "__main__":
    main()
