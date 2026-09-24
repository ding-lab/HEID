#!/usr/bin/env python -u
import os
PROJECTS_ROOT = os.environ.get("PROJECTS_ROOT", "/data/heid")
from pathlib import Path
_RELEASE = Path(__file__).resolve().parents[2]
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
import torch.nn.functional as F
import tifffile
import zarr
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torchvision import transforms
from sklearn.metrics import roc_auc_score

try:
    import timm
    from timm.layers import SwiGLUPacked
except ImportError:
    print("ERROR: timm not installed", flush=True)
    sys.exit(1)


PC = Path(PROJECTS_ROOT + "/cell")
COHORT_CSV = PC / "data/phase1_cohort_184.csv"
CELL_TABLE_DIR = PC / "data/cell_tables"
SUMMARY = PC / "data/cell_tables/_summary.csv"
SPLIT_JSON = PC / "data/split_5fold.json"
OUT_DIR = PC / "shared/lora"
LOG_DIR = PC / "shared/logs"


sys.path.insert(0, str(_RELEASE / "cell/scripts"))
from stain_transforms import get_stain_fn
STAIN_MODE = os.environ.get("CELL_STAIN_MODE", "color")
_SFX = "" if STAIN_MODE == "color" else f"_{STAIN_MODE}"

UNI2_PATHS = [
    Path(PROJECTS_ROOT + "/tools/uni2/pytorch_model.bin"),
]


CLASSES = [
    "Tumor", "Fibroblast", "NK_T", "Macrophage", "NonMalignant_Parenchymal",
    "Endothelial", "Plasma", "Pericyte/vSMC", "SMC", "B_cell", "Others",
    "Necrosis", "DC", "Neutrophil", "Granulocyte", "Lymphatic_Endothelial", "Schwann",
]
CT2IDX = {c: i for i, c in enumerate(CLASSES)}
N_CLASS = len(CLASSES)

XPX = 0.2125
CROP = 224
HALF = CROP // 2
STRIP_H = 4096
STRIP_PAD = HALF
CHUNK = 512
BAND_H = 4096
WHITE_THRESH = 240
BLANK_FRAC = 0.9


LORA_RANK = 16
LORA_ALPHA = 32
LORA_DROPOUT = 0.05
LORA_LR = 1e-4
LORA_HEAD_LR = 1e-3
LORA_EPOCHS = 5
LORA_BATCH_SIZE = 64
LORA_SUBSAMPLE = 50_000
LORA_WARMUP = 1
NUM_WORKERS = 8
VAL_SUBSET = 10_000


class LoRALinear(nn.Module):
    def __init__(self, in_f, out_f, rank=8, alpha=16, dropout=0.05):
        super().__init__()
        self.scaling = alpha / rank
        self.lora_A = nn.Parameter(torch.zeros(rank, in_f))
        self.lora_B = nn.Parameter(torch.zeros(out_f, rank))
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B)

    def forward(self, x):
        return (self.dropout(x) @ self.lora_A.T @ self.lora_B.T) * self.scaling


class LoRAFusedQKV(nn.Module):
    def __init__(self, qkv, rank=8, alpha=16, dropout=0.05):
        super().__init__()
        self.original_qkv = qkv
        ed = qkv.in_features
        self.lora_q = LoRALinear(ed, ed, rank, alpha, dropout)
        self.lora_v = LoRALinear(ed, ed, rank, alpha, dropout)

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
        print("FATAL: UNI2 weights not found in", [str(x) for x in UNI2_PATHS], flush=True)
        sys.exit(2)
    print(f"Loading UNI2-h from {p}", flush=True)
    m = timm.create_model(
        "vit_giant_patch14_224", img_size=224, patch_size=14, depth=24, num_heads=24,
        init_values=1e-5, embed_dim=1536, mlp_ratio=2.66667 * 2, num_classes=0,
        no_embed_class=True, mlp_layer=SwiGLUPacked, act_layer=torch.nn.SiLU,
        reg_tokens=8, dynamic_img_size=True,
    )
    m.load_state_dict(torch.load(p, map_location="cpu"), strict=True)
    return m.to(device)


def inject_lora(m, rank, alpha, dropout=0.05):
    for p in m.parameters():
        p.requires_grad = False
    dev = next(m.parameters()).device
    modules = []
    for block in m.blocks:
        lqkv = LoRAFusedQKV(block.attn.qkv, rank=rank, alpha=alpha, dropout=dropout).to(dev)
        block.attn.qkv = lqkv
        modules.append(lqkv)
    n_lora = sum(p.numel() for m_ in modules for p in m_.parameters() if p.requires_grad)
    n_total = sum(p.numel() for p in m.parameters())
    print(f"LoRA injected: {len(modules)} blocks, {n_lora:,} trainable / {n_total:,} total "
          f"({100 * n_lora / n_total:.2f}%)", flush=True)
    return modules


def get_lora_state_dict(m):
    return {name: prm.data.cpu() for name, prm in m.named_parameters() if prm.requires_grad}


TRANSFORM_TRAIN = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
])
TRANSFORM_VAL = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


class CropArrayDataset(Dataset):
    def __init__(self, crops, labels, transform, stain=False, seed=0):
        self.crops = crops
        self.labels = labels
        self.transform = transform
        self.stain_fn, self.needs_rng = (get_stain_fn(STAIN_MODE) if stain else (None, False))
        self._seed = int(seed)
        self._rng = None

    def _worker_rng(self):
        if self._rng is None:
            wi = torch.utils.data.get_worker_info()
            wid = wi.id if wi is not None else 0
            self._rng = np.random.default_rng(self._seed + 100003 * wid)
        return self._rng

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        crop = self.crops[idx]
        if self.stain_fn is not None:
            crop = self.stain_fn(crop, self._worker_rng()) if self.needs_rng else self.stain_fn(crop)
        return self.transform(crop), int(self.labels[idx])


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


def gather_crops_for_sample(he_path, xs, ys):
    store = tifffile.imread(str(he_path), aszarr=True, level=0)
    z = zarr.open(store, mode="r")
    H, W = z.shape[0], z.shape[1]

    n = len(xs)
    out_crops = [None] * n
    ymin = max(0, int(ys.min()) - HALF)
    ymax = min(H, int(ys.max()) + HALF + 1)
    for sy in range(ymin, ymax, BAND_H):
        in_band = np.nonzero((ys >= sy) & (ys < min(sy + BAND_H, ymax)))[0]
        if len(in_band) == 0:
            continue
        ry0 = (max(0, sy - STRIP_PAD) // CHUNK) * CHUNK
        ry1 = min(H, ((min(H, sy + BAND_H + STRIP_PAD) + CHUNK - 1) // CHUNK) * CHUNK)

        bx = xs[in_band]
        x0b = (max(0, int(bx.min()) - HALF) // CHUNK) * CHUNK
        x1b = min(W, ((int(bx.max()) + HALF + 1 + CHUNK - 1) // CHUNK) * CHUNK)
        block = np.asarray(z[ry0:ry1, x0b:x1b, :])
        bh, bw = block.shape[0], block.shape[1]
        for gi in in_band:
            c = make_crop(block, int(ys[gi]) - ry0, int(xs[gi]) - x0b, bh, bw)
            if c is not None:
                out_crops[gi] = c
        del block
    crops, keep = [], np.zeros(n, dtype=bool)
    for gi in range(n):
        if out_crops[gi] is not None:
            crops.append(out_crops[gi])
            keep[gi] = True
    return crops, keep


def load_split():
    sp = json.loads(SPLIT_JSON.read_text())
    return sp["folds"]


def qc_feat_samples():
    co = pd.read_csv(COHORT_CSV)
    qc = co[co["qc_pass"] == True].copy()
    he_of = dict(zip(qc["sample"], qc["aligned_he_path"]))
    return qc, he_of


def collect_fold_records(fold_idx, max_train, val_subset, smoke, seed=42):
    folds = load_split()
    fold = next(f for f in folds if f["fold"] == fold_idx)
    test_set = set(fold["test_samples"])
    co, he_of = qc_feat_samples()
    qc_samples = [s for s in co.loc[co["qc_pass"] == True, "sample"] if (CELL_TABLE_DIR / f"{s}.parquet").exists()]

    rng = np.random.RandomState(seed + fold_idx)
    if smoke:


        max_train = int(os.environ.get("SMOKE_CAP", 17_000))
        val_subset = int(os.environ.get("SMOKE_VAL", 4_000))


    train_pool = {c: [] for c in range(N_CLASS)}
    val_rows = []
    t0 = time.time()
    for s in qc_samples:
        is_test = s in test_set
        ct = pd.read_parquet(CELL_TABLE_DIR / f"{s}.parquet")
        ct = ct[ct["cell_type"].isin(CT2IDX)]
        if len(ct) == 0:
            continue
        xs = np.round(ct["x_px"].values).astype(np.int64)
        ys = np.round(ct["y_px"].values).astype(np.int64)
        labs = ct["cell_type"].map(CT2IDX).values.astype(np.int64)
        if is_test:
            for i in range(len(ct)):
                val_rows.append((s, xs[i], ys[i], labs[i]))
        else:
            for c in range(N_CLASS):
                idx = np.nonzero(labs == c)[0]
                for i in idx:
                    train_pool[c].append((s, xs[i], ys[i]))
    print(f"[fold {fold_idx}] label scan {time.time()-t0:.0f}s; "
          f"train_pool sizes={[len(train_pool[c]) for c in range(N_CLASS)]}", flush=True)


    per_class = max(1, max_train // N_CLASS)
    train_sel = []
    for c in range(N_CLASS):
        pool = train_pool[c]
        if len(pool) == 0:
            continue
        if len(pool) > per_class:
            pick = rng.choice(len(pool), per_class, replace=False)
            train_sel.extend((pool[i][0], pool[i][1], pool[i][2], c) for i in pick)
        else:
            train_sel.extend((p[0], p[1], p[2], c) for p in pool)
    rng.shuffle(train_sel)


    if len(val_rows) > val_subset:
        vidx = rng.choice(len(val_rows), val_subset, replace=False)
        val_sel = [val_rows[i] for i in vidx]
    else:
        val_sel = val_rows


    def gather(group_rows):
        by_sample = {}
        for r in group_rows:
            by_sample.setdefault(r[0], []).append(r)
        all_crops, all_labels = [], []
        for s, rows in by_sample.items():
            xs = np.array([r[1] for r in rows], dtype=np.int64)
            ys = np.array([r[2] for r in rows], dtype=np.int64)
            labs = np.array([r[3] for r in rows], dtype=np.int64)
            crops, keep = gather_crops_for_sample(he_of[s], xs, ys)
            all_crops.extend(crops)
            all_labels.extend(labs[keep].tolist())
        if not all_crops:
            return np.zeros((0, CROP, CROP, 3), np.uint8), np.zeros(0, np.int64)
        return np.stack(all_crops).astype(np.uint8), np.array(all_labels, dtype=np.int64)

    tg = time.time()
    train_crops, train_labels = gather(train_sel)
    val_crops, val_labels = gather(val_sel)
    print(f"[fold {fold_idx}] crop gather {time.time()-tg:.0f}s; "
          f"train={len(train_labels)} val={len(val_labels)} "
          f"train_mem={train_crops.nbytes/1e9:.1f}GB", flush=True)

    tc = np.bincount(train_labels, minlength=N_CLASS)
    print(f"[fold {fold_idx}] train class dist: "
          + json.dumps({CLASSES[c]: int(tc[c]) for c in range(N_CLASS)}), flush=True)
    return train_crops, train_labels, val_crops, val_labels


def cosine_warmup_lambda(epoch, warmup, total, min_lr_frac=1e-6):
    if epoch < warmup:
        return (epoch + 1) / warmup
    p = (epoch - warmup) / max(1, total - warmup)
    return max(min_lr_frac, 0.5 * (1 + np.cos(np.pi * p)))


def load_cropbank(fold_idx):
    p_tr = OUT_DIR / f"cropbank_fold{fold_idx}_train.npy"
    p_try = OUT_DIR / f"cropbank_fold{fold_idx}_train_y.npy"
    p_va = OUT_DIR / f"cropbank_fold{fold_idx}_val.npy"
    p_vay = OUT_DIR / f"cropbank_fold{fold_idx}_val_y.npy"
    if not all(p.exists() for p in (p_tr, p_try, p_va, p_vay)):
        return None
    tc = np.load(p_tr)
    tl = np.load(p_try).astype(np.int64)
    vc = np.load(p_va)
    vl = np.load(p_vay).astype(np.int64)
    print(f"[fold {fold_idx}] loaded crop bank: train={len(tl)} val={len(vl)} "
          f"({tc.nbytes/1e9:.1f}GB) from {OUT_DIR}", flush=True)
    return tc, tl, vc, vl


def train_fold(fold_idx, device, smoke, use_cropbank=False):
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    save_path = OUT_DIR / f"lora_pc{_SFX}_fold{fold_idx}.pt"
    if save_path.exists() and not smoke:
        print(f"Fold {fold_idx}: already exists at {save_path}, skip.", flush=True)
        return
    epochs = int(os.environ.get("SMOKE_EPOCHS", LORA_EPOCHS)) if smoke else LORA_EPOCHS
    cap = LORA_SUBSAMPLE

    print(f"\n{'='*60}\nUNI2-h LoRA — fold {fold_idx} "
          f"({'SMOKE' if smoke else 'FULL'}{', CROPBANK' if use_cropbank else ''})\n{'='*60}", flush=True)
    t0 = time.time()

    bank = load_cropbank(fold_idx) if use_cropbank else None
    if use_cropbank and bank is None:
        print(f"FATAL: --cropbank requested but bank for fold {fold_idx} not found in {OUT_DIR}", flush=True)
        sys.exit(6)
    if bank is not None:
        train_crops, train_labels, val_crops, val_labels = bank
    else:
        train_crops, train_labels, val_crops, val_labels = collect_fold_records(
            fold_idx, cap, VAL_SUBSET, smoke)
    if len(train_labels) == 0:
        print("FATAL: no train crops gathered", flush=True)
        sys.exit(3)

    train_ds = CropArrayDataset(train_crops, train_labels, TRANSFORM_TRAIN, stain=True, seed=42 + fold_idx)
    val_ds = CropArrayDataset(val_crops, val_labels, TRANSFORM_VAL)

    counts = torch.bincount(torch.tensor(train_labels), minlength=N_CLASS).float().clamp(min=1)
    weights = (1.0 / counts)[torch.tensor(train_labels)]
    sampler = WeightedRandomSampler(weights, len(train_ds), replacement=True)
    train_loader = DataLoader(train_ds, batch_size=LORA_BATCH_SIZE, sampler=sampler,
                              num_workers=NUM_WORKERS, pin_memory=True,
                              persistent_workers=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=LORA_BATCH_SIZE * 2, shuffle=False,
                            num_workers=NUM_WORKERS, pin_memory=True,
                            persistent_workers=(len(val_ds) > 0))

    model = load_uni2(device)
    lora_modules = inject_lora(model, LORA_RANK, LORA_ALPHA, LORA_DROPOUT)
    head = nn.Linear(1536, N_CLASS).to(device)

    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True

    lora_params = [p for m_ in lora_modules for p in m_.parameters() if p.requires_grad]
    opt = torch.optim.AdamW([
        {"params": lora_params, "lr": LORA_LR},
        {"params": head.parameters(), "lr": LORA_HEAD_LR},
    ], weight_decay=0.01)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda ep: cosine_warmup_lambda(ep, LORA_WARMUP, epochs))
    crit = nn.CrossEntropyLoss(label_smoothing=0.1)

    best_auc, best_lora_sd, best_head_sd = 0.0, None, None
    history = []

    for epoch in range(epochs):
        te = time.time()
        model.train(); head.train()
        tot_loss, correct, total = 0.0, 0, 0
        for imgs, labs in train_loader:
            imgs = imgs.to(device, non_blocking=True)
            labs = labs.to(device, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                feats = model(imgs)
                logits = head(feats.float())
            loss = crit(logits, labs)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(lora_params + list(head.parameters()), 1.0)
            opt.step()
            tot_loss += loss.item() * len(imgs)
            correct += (logits.argmax(1) == labs).sum().item()
            total += len(imgs)
        sched.step()
        train_loss = tot_loss / max(1, total)
        train_acc = correct / max(1, total)


        val_auc = 0.0
        if len(val_ds) > 0:
            model.eval(); head.eval()
            probs_list, labels_list = [], []
            with torch.no_grad():
                for imgs, labs in val_loader:
                    imgs = imgs.to(device, non_blocking=True)
                    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                        feats = model(imgs)
                        logits = head(feats.float())
                    probs_list.append(F.softmax(logits.float(), 1).cpu().numpy())
                    labels_list.extend(labs.numpy())
            probs = np.concatenate(probs_list)
            larr = np.array(labels_list)
            aucs = []
            for i in range(N_CLASS):
                b = (larr == i).astype(int)
                if 0 < b.sum() < len(b):
                    aucs.append(roc_auc_score(b, probs[:, i]))
            val_auc = float(np.mean(aucs)) if aucs else 0.0

        elapsed = time.time() - te
        rec = {"epoch": epoch + 1, "train_loss": round(train_loss, 4),
               "train_acc": round(train_acc, 4), "val_macro_auc": round(val_auc, 4),
               "sec": round(elapsed, 1)}
        history.append(rec)
        print(f"  Ep {epoch+1}/{epochs} | loss={train_loss:.4f} acc={train_acc:.3f} "
              f"| val_auc={val_auc:.3f} | {elapsed:.0f}s", flush=True)

        if val_auc >= best_auc:
            best_auc = val_auc
            best_lora_sd = get_lora_state_dict(model)
            best_head_sd = head.state_dict()
            print(f"    -> new best (AUC={val_auc:.3f})", flush=True)

    if best_lora_sd is None:
        best_lora_sd = get_lora_state_dict(model)
        best_head_sd = head.state_dict()

    tag = "smoke" if smoke else "full"
    out_path = OUT_DIR / (f"lora_pc{_SFX}_fold{fold_idx}.pt" if not smoke
                          else f"lora_pc{_SFX}_fold{fold_idx}_SMOKE.pt")
    torch.save({
        "lora_state_dict": best_lora_sd,
        "head_state_dict": best_head_sd,
        "val_auc": best_auc,
        "cell_types": CLASSES,
        "lora_config": {"rank": LORA_RANK, "alpha": LORA_ALPHA, "dropout": LORA_DROPOUT,
                        "lora_lr": LORA_LR, "head_lr": LORA_HEAD_LR, "epochs": epochs,
                        "subsample": cap, "label_smoothing": 0.1, "stain_mode": STAIN_MODE},
        "fold": fold_idx,
        "history": history,
        "n_train": int(len(train_labels)), "n_val": int(len(val_labels)),
        "note": f"UNI2-h 17-class LoRA r16 on unmasked 224px crops (he_aligned); stain_mode={STAIN_MODE}.",
    }, out_path)
    wall = time.time() - t0
    print(f"\nSaved: {out_path} (best val AUC={best_auc:.3f}); total {wall:.0f}s "
          f"({wall/3600:.2f}h)", flush=True)


    log = {"fold": fold_idx, "smoke": bool(smoke), "best_val_auc": round(best_auc, 4),
           "history": history, "n_train": int(len(train_labels)),
           "n_val": int(len(val_labels)), "wall_sec": round(wall, 1),
           "torch": torch.__version__, "timm": timm.__version__,
           "node": os.environ.get("SLURMD_NODENAME", "?"),
           "jobid": os.environ.get("SLURM_JOB_ID", "?")}
    (LOG_DIR / f"lora_train_fold{fold_idx}_{tag}.json").write_text(json.dumps(log, indent=2))

    del model, head, opt, sched
    torch.cuda.empty_cache()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fold", type=int, required=True, choices=[0, 1, 2, 3, 4])
    ap.add_argument("--smoke", action="store_true",
                    help="small subsample + 2 epochs; saves *_SMOKE.pt; no skip")
    ap.add_argument("--cropbank", action="store_true",
                    help="load pre-built Stage-1 crop bank (.npy) instead of inline gather ")
    args = ap.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device} torch={torch.__version__} cuda={torch.version.cuda}", flush=True)
    train_fold(args.fold, device, args.smoke, use_cropbank=args.cropbank)
