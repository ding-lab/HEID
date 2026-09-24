#!/usr/bin/env python -u
import os
PROJECTS_ROOT = os.environ.get("PROJECTS_ROOT", "/data/heid")
import argparse
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import tifffile
import zarr
from torchvision import transforms

try:
    import timm
    from timm.layers import SwiGLUPacked
except ImportError:
    print("ERROR: timm not installed", flush=True)
    sys.exit(1)

PC = Path(PROJECTS_ROOT + "/cell")
COHORT_CSV = PC / "data/phase1_cohort_184.csv"
CELL_TABLE_DIR = PC / "data/cell_tables"
SPLIT_JSON = PC / "data/split_5fold.json"
LORA_DIR = PC / "shared/lora"

STAIN_MODE = os.environ.get("CELL_STAIN_MODE", "color")
_SFX = "" if STAIN_MODE == "color" else f"_{STAIN_MODE}"
OUT_OOF = PC / f"shared/features/lora_cls_oof{_SFX}"
OUT_MP_OOF = PC / f"shared/features/lora_meanpool_oof{_SFX}"
OUT_TRAIN = PC / "shared/features/lora_cls_train"
MANIFEST_DIR = PC / f"shared/features/manifests_lora{_SFX}"

UNI2_PATHS = [
    Path(PROJECTS_ROOT + "/tools/uni2/pytorch_model.bin"),
]

XPX = 0.2125
CROP = 224
HALF = CROP // 2
STRIP_H = 4096
STRIP_PAD = HALF
BATCH = 256
WHITE_THRESH = 240
BLANK_FRAC = 0.9
LORA_RANK = 16
LORA_ALPHA = 32

TRANSFORM = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


class LoRALinear(nn.Module):
    def __init__(self, in_f, out_f, rank=8, alpha=16):
        super().__init__()
        self.scaling = alpha / rank
        self.lora_A = nn.Parameter(torch.zeros(rank, in_f))
        self.lora_B = nn.Parameter(torch.zeros(out_f, rank))
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B)

    def forward(self, x):
        return (x @ self.lora_A.T @ self.lora_B.T) * self.scaling


class LoRAFusedQKV(nn.Module):
    def __init__(self, qkv, rank=8, alpha=16):
        super().__init__()
        self.original_qkv = qkv
        ed = qkv.in_features
        self.lora_q = LoRALinear(ed, ed, rank, alpha)
        self.lora_v = LoRALinear(ed, ed, rank, alpha)

    def forward(self, x):
        qkv = self.original_qkv(x)
        ed = self.original_qkv.in_features
        return torch.cat([
            qkv[..., :ed] + self.lora_q(x),
            qkv[..., ed:2 * ed],
            qkv[..., 2 * ed:] + self.lora_v(x),
        ], dim=-1)


def load_uni2(device):
    p = next((q for q in UNI2_PATHS if q.exists()), None)
    if p is None:
        print("FATAL: UNI2 not found", flush=True)
        sys.exit(2)
    m = timm.create_model(
        "vit_giant_patch14_224", img_size=224, patch_size=14, depth=24, num_heads=24,
        init_values=1e-5, embed_dim=1536, mlp_ratio=2.66667 * 2, num_classes=0,
        no_embed_class=True, mlp_layer=SwiGLUPacked, act_layer=torch.nn.SiLU,
        reg_tokens=8, dynamic_img_size=True,
    )
    m.load_state_dict(torch.load(p, map_location="cpu"), strict=True)
    return m.to(device)


def inject_lora(m, rank, alpha):
    for p in m.parameters():
        p.requires_grad = False
    dev = next(m.parameters()).device
    lqkvs = []
    for block in m.blocks:
        lqkv = LoRAFusedQKV(block.attn.qkv, rank=rank, alpha=alpha).to(dev)
        block.attn.qkv = lqkv
        lqkvs.append(lqkv)
    return lqkvs


def load_lora_sd(m, sd):
    from adapter_contract import copy_lora_state
    copy_lora_state(m, sd)


def sample_test_fold(sample, folds):
    for f in folds:
        if sample in f["test_samples"]:
            return f["fold"]
    raise ValueError(f"sample {sample} not in any fold's test_samples")


def make_crop(block, yy, xx, bh, bw):
    y0, x0 = yy - HALF, xx - HALF
    y1, x1 = y0 + CROP, x0 + CROP
    crop = np.full((CROP, CROP, 3), 255, np.uint8)
    sy0, sx0 = max(0, y0), max(0, x0)
    sy1, sx1 = min(bh, y1), min(bw, x1)
    if sy1 > sy0 and sx1 > sx0:
        crop[sy0 - y0:sy1 - y0, sx0 - x0:sx1 - x0] = block[sy0:sy1, sx0:sx1]
    if np.mean(crop > WHITE_THRESH) > BLANK_FRAC:
        return None
    return crop


@torch.no_grad()
def forward_cls(model, crops, device, want_meanpool=False):
    batch = torch.stack([TRANSFORM(c) for c in crops]).to(device, non_blocking=True)
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        if want_meanpool:
            tok = model.forward_features(batch)
            cls = model.forward_head(tok, pre_logits=True)
            patch = tok[:, 1 + 8:, :].mean(dim=1)
            return cls.cpu().float(), patch.cpu().float()
        cls = model(batch)
        return cls.cpu().float(), None


def extract_one_adapter(model, lqkvs, lora_sd, sample, he_path, ct, device,
                        want_meanpool, out_cls_path, out_mp_path):
    load_lora_sd(model, lora_sd)
    model.eval()

    cids = ct["cell_id"].values
    ctypes = ct["cell_type"].values
    xs = np.round(ct["x_px"].values).astype(np.int64)
    ys = np.round(ct["y_px"].values).astype(np.int64)
    n = len(ct)

    store = tifffile.imread(str(he_path), aszarr=True, level=0)
    z = zarr.open(store, mode="r")
    H, W = z.shape[0], z.shape[1]
    xmin = max(0, int(xs.min()) - HALF)
    xmax = min(W, int(xs.max()) + HALF + 1)
    ymin = max(0, int(ys.min()) - HALF)
    ymax = min(H, int(ys.max()) + HALF + 1)

    feats_cls = np.empty((n, 1536), dtype=np.float16)
    feats_mp = np.empty((n, 1536), dtype=np.float16) if want_meanpool else None
    valid = np.zeros(n, dtype=bool)
    pend_crops, pend_idx, n_blank = [], [], 0
    t0 = time.time()

    def flush():
        nonlocal pend_crops, pend_idx
        if not pend_crops:
            return
        cls, mp = forward_cls(model, pend_crops, device, want_meanpool)
        for k, gi in enumerate(pend_idx):
            feats_cls[gi] = cls[k].numpy().astype(np.float16)
            if want_meanpool:
                feats_mp[gi] = mp[k].numpy().astype(np.float16)
            valid[gi] = True
        pend_crops, pend_idx = [], []

    for sy in range(ymin, ymax, STRIP_H):
        ry0 = max(0, sy - STRIP_PAD)
        ry1 = min(H, sy + STRIP_H + STRIP_PAD)
        block = np.asarray(z[ry0:ry1, xmin:xmax, :])
        bh, bw = block.shape[0], block.shape[1]
        in_strip = np.nonzero((ys >= sy) & (ys < min(sy + STRIP_H, ymax)))[0]
        for gi in in_strip:
            c = make_crop(block, int(ys[gi]) - ry0, int(xs[gi]) - xmin, bh, bw)
            if c is None:
                n_blank += 1
                continue
            pend_crops.append(c)
            pend_idx.append(gi)
            if len(pend_crops) >= BATCH:
                flush()
        del block
    flush()

    sel = np.nonzero(valid)[0]
    out_cls_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"features": torch.from_numpy(feats_cls[sel]),
                "cell_ids": [str(c) for c in cids[sel]],
                "cell_type": [str(c) for c in ctypes[sel]]}, out_cls_path)
    if want_meanpool:
        out_mp_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"features": torch.from_numpy(feats_mp[sel]),
                    "cell_ids": [str(c) for c in cids[sel]],
                    "cell_type": [str(c) for c in ctypes[sel]]}, out_mp_path)
    wall = time.time() - t0
    print(f"[{sample}] adapter done: valid={len(sel)} blank={n_blank} "
          f"{wall:.0f}s ({len(sel)/max(1,wall):.0f} cells/s) -> {out_cls_path.name}", flush=True)
    return len(sel), n_blank, wall


def extract_multi_adapter(model, lqkvs, sample, he_path, ct, device, fold_paths, want_meanpool):
    cids = ct["cell_id"].values
    ctypes = ct["cell_type"].values
    xs = np.round(ct["x_px"].values).astype(np.int64)
    ys = np.round(ct["y_px"].values).astype(np.int64)
    n = len(ct)

    store = tifffile.imread(str(he_path), aszarr=True, level=0)
    z = zarr.open(store, mode="r")
    H, W = z.shape[0], z.shape[1]
    xmin = max(0, int(xs.min()) - HALF)
    xmax = min(W, int(xs.max()) + HALF + 1)
    ymin = max(0, int(ys.min()) - HALF)
    ymax = min(H, int(ys.max()) + HALF + 1)

    feats = {f: np.empty((n, 1536), dtype=np.float16) for f in fold_paths}
    valid = np.zeros(n, dtype=bool)
    pend_crops, pend_idx, n_blank = [], [], 0
    t0 = time.time()


    def apply_lora(sd):
        load_lora_sd(model, sd)

    def flush():
        nonlocal pend_crops, pend_idx
        if not pend_crops:
            return
        batch = torch.stack([TRANSFORM(c) for c in pend_crops]).to(device, non_blocking=True)
        for f, (sd, _, _) in fold_paths.items():
            apply_lora(sd)
            with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                cls = model(batch).cpu().float().numpy().astype(np.float16)
            for k, gi in enumerate(pend_idx):
                feats[f][gi] = cls[k]
        for gi in pend_idx:
            valid[gi] = True
        pend_crops, pend_idx = [], []

    model.eval()
    for sy in range(ymin, ymax, STRIP_H):
        ry0 = max(0, sy - STRIP_PAD)
        ry1 = min(H, sy + STRIP_H + STRIP_PAD)
        block = np.asarray(z[ry0:ry1, xmin:xmax, :])
        bh, bw = block.shape[0], block.shape[1]
        in_strip = np.nonzero((ys >= sy) & (ys < min(sy + STRIP_H, ymax)))[0]
        for gi in in_strip:
            c = make_crop(block, int(ys[gi]) - ry0, int(xs[gi]) - xmin, bh, bw)
            if c is None:
                n_blank += 1
                continue
            pend_crops.append(c)
            pend_idx.append(gi)
            if len(pend_crops) >= BATCH:
                flush()
        del block
    flush()

    sel = np.nonzero(valid)[0]
    for f, (_, out_cls_path, _) in fold_paths.items():
        out_cls_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"features": torch.from_numpy(feats[f][sel]),
                    "cell_ids": [str(c) for c in cids[sel]],
                    "cell_type": [str(c) for c in ctypes[sel]]}, out_cls_path)
    wall = time.time() - t0
    print(f"[{sample}] multi-adapter ({len(fold_paths)} folds) done: valid={len(sel)} "
          f"blank={n_blank} {wall:.0f}s -> {len(fold_paths)} files", flush=True)
    return len(sel), n_blank, wall


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sample", required=True)
    ap.add_argument("--mode", choices=["oof", "train"], default="oof")
    ap.add_argument("--fold", type=int, default=-1, help="for --mode train")
    ap.add_argument("--multi-adapter", action="store_true",
                    help="read crops once, run all 4 train-folds' LoRA (skip sample's own test fold)")
    ap.add_argument("--store-meanpool", action="store_true")
    ap.add_argument("--smoke-lora", default="", help="path to a specific LoRA .pt (overrides fold lookup; smoke)")
    ap.add_argument("--cell-subset", default="",
                    help="(--mode train) path to capped_selection/fold{F}.json; extract ONLY the "
                         "selected cell_ids for this sample (cap400k/class balanced train subset)")
    args = ap.parse_args()
    sample = args.sample
    t0 = time.time()

    co = pd.read_csv(COHORT_CSV)
    row = co[(co["sample"] == sample) & (co["qc_pass"] == True)]
    if len(row) == 0:
        print(f"FATAL: {sample} not in qc_pass cohort", flush=True)
        sys.exit(2)
    he_path = row.iloc[0]["aligned_he_path"]
    if not os.path.exists(he_path):
        print(f"FATAL: he_path missing {he_path}", flush=True)
        sys.exit(3)
    folds = json.loads(SPLIT_JSON.read_text())["folds"]


    if not args.smoke_lora and not args.multi_adapter:
        if args.mode == "oof":
            done_path = OUT_OOF / f"{sample}.pt"
        else:
            done_path = OUT_TRAIN / f"fold{args.fold}" / f"{sample}.pt"
        if done_path.exists():
            print(f"[{sample}] {args.mode} feature already exists at {done_path} — skip.", flush=True)
            return

    ct = pd.read_parquet(CELL_TABLE_DIR / f"{sample}.parquet")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.benchmark = True
    model = load_uni2(device)
    lqkvs = inject_lora(model, LORA_RANK, LORA_ALPHA)
    print(f"[{sample}] {len(ct)} cells; HE {he_path}", flush=True)

    MANIFEST_DIR.mkdir(parents=True, exist_ok=True)
    man = {"sample": sample, "he_path": str(he_path), "n_cells_table": int(len(ct)),
           "torch": torch.__version__, "timm": timm.__version__,
           "node": os.environ.get("SLURMD_NODENAME", "?"),
           "jobid": os.environ.get("SLURM_JOB_ID", "?"),
           "lora_rank": LORA_RANK, "lora_alpha": LORA_ALPHA}

    if args.smoke_lora:
        ckpt = torch.load(args.smoke_lora, map_location="cpu", weights_only=False)
        out_cls = OUT_OOF / f"{sample}_SMOKE.pt"
        out_mp = OUT_MP_OOF / f"{sample}_SMOKE.pt"
        nv, nb, w = extract_one_adapter(model, lqkvs, ckpt["lora_state_dict"], sample, he_path,
                                        ct, device, args.store_meanpool, out_cls, out_mp)
        man.update({"mode": "smoke", "lora": args.smoke_lora, "n_valid": nv,
                    "n_blank": nb, "wall_sec": round(w, 1)})
    elif args.multi_adapter:
        own = sample_test_fold(sample, folds)
        fold_paths = {}
        for f in range(5):
            if f == own:
                continue
            lp = LORA_DIR / f"lora_pc_fold{f}.pt"
            if not lp.exists():
                print(f"WARN: {lp} missing, skip fold {f}", flush=True)
                continue
            sd = torch.load(lp, map_location="cpu", weights_only=False)["lora_state_dict"]
            fold_paths[f] = (sd, OUT_TRAIN / f"fold{f}" / f"{sample}.pt", None)
        nv, nb, w = extract_multi_adapter(model, lqkvs, sample, he_path, ct, device,
                                          fold_paths, args.store_meanpool)
        man.update({"mode": "multi-adapter", "folds": list(fold_paths.keys()),
                    "own_test_fold": own, "n_valid": nv, "n_blank": nb, "wall_sec": round(w, 1)})
    elif args.mode == "oof":
        f = sample_test_fold(sample, folds)
        lp = LORA_DIR / f"lora_pc{_SFX}_fold{f}.pt"
        if not lp.exists():
            print(f"FATAL: {lp} missing — train fold {f} LoRA first", flush=True)
            sys.exit(4)
        sd = torch.load(lp, map_location="cpu", weights_only=False)["lora_state_dict"]
        out_cls = OUT_OOF / f"{sample}.pt"
        out_mp = OUT_MP_OOF / f"{sample}.pt"
        nv, nb, w = extract_one_adapter(model, lqkvs, sd, sample, he_path, ct, device,
                                        args.store_meanpool, out_cls, out_mp)
        man.update({"mode": "oof", "test_fold": f, "n_valid": nv, "n_blank": nb, "wall_sec": round(w, 1)})
    else:
        if args.fold < 0:
            print("FATAL: --mode train requires --fold", flush=True)
            sys.exit(2)
        own = sample_test_fold(sample, folds)
        if own == args.fold:
            print(f"FATAL: {sample} is a TEST sample of fold {args.fold} — leakage; "
                  f"do not extract train features for its own test fold", flush=True)
            sys.exit(5)
        lp = LORA_DIR / f"lora_pc_fold{args.fold}.pt"
        if not lp.exists():
            print(f"FATAL: {lp} missing", flush=True)
            sys.exit(4)
        sd = torch.load(lp, map_location="cpu", weights_only=False)["lora_state_dict"]

        n_before = len(ct)
        if args.cell_subset:
            sel = json.loads(Path(args.cell_subset).read_text())
            if int(sel.get("fold", -99)) != args.fold:
                print(f"FATAL: --cell-subset fold {sel.get('fold')} != --fold {args.fold}", flush=True)
                sys.exit(6)
            keep_ids = set(sel.get("samples", {}).get(sample, []))
            if not keep_ids:
                print(f"[{sample}] not in fold {args.fold} capped selection (0 cells) — "
                      f"writing empty feature file (defensive)", flush=True)
                out_cls = OUT_TRAIN / f"fold{args.fold}" / f"{sample}.pt"
                out_cls.parent.mkdir(parents=True, exist_ok=True)
                torch.save({"features": torch.zeros((0, 1536), dtype=torch.float16),
                            "cell_ids": [], "cell_type": []}, out_cls)
                man.update({"mode": "train", "fold": args.fold, "n_valid": 0, "n_blank": 0,
                            "n_before_subset": int(n_before), "n_subset": 0, "wall_sec": 0.0})
                (MANIFEST_DIR / f"{sample}_train.json").write_text(json.dumps(man, indent=2))
                print(f"[{sample}] DONE (empty) in {time.time()-t0:.0f}s", flush=True)
                return
            ct = ct[ct["cell_id"].astype(str).isin(keep_ids)].reset_index(drop=True)
            print(f"[{sample}] capped subset fold {args.fold}: {n_before} -> {len(ct)} cells", flush=True)
        out_cls = OUT_TRAIN / f"fold{args.fold}" / f"{sample}.pt"
        nv, nb, w = extract_one_adapter(model, lqkvs, sd, sample, he_path, ct, device,
                                        False, out_cls, None)
        man.update({"mode": "train", "fold": args.fold, "n_valid": nv, "n_blank": nb,
                    "n_before_subset": int(n_before), "n_subset": int(len(ct)),
                    "wall_sec": round(w, 1)})

    (MANIFEST_DIR / f"{sample}_{man['mode']}.json").write_text(json.dumps(man, indent=2))
    print(f"[{sample}] DONE in {time.time()-t0:.0f}s", flush=True)
