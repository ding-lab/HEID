#!/usr/bin/env python -u
import os
PROJECTS_ROOT = os.environ.get("PROJECTS_ROOT", "/data/heid")
from pathlib import Path
_RELEASE = Path(__file__).resolve().parents[2]
import argparse
import json
import math
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torchvision import transforms
from sklearn.metrics import roc_auc_score


PC = Path(PROJECTS_ROOT + "/cell")
CROPBANK_DIR = PC / "shared/lora"
OUT_DIR = PC / "shared/lora_phikon"
LOG_DIR = PC / "shared/logs"


sys.path.insert(0, str(_RELEASE / "cell/scripts"))
from stain_transforms import get_stain_fn
STAIN_MODE = os.environ.get("B2_STAIN_MODE", "color")
_SFX = "" if STAIN_MODE == "color" else f"_{STAIN_MODE}"


CLASSES = [
    "Tumor", "Fibroblast", "NK_T", "Macrophage", "NonMalignant_Parenchymal",
    "Endothelial", "Plasma", "Pericyte/vSMC", "SMC", "B_cell", "Others",
    "Necrosis", "DC", "Neutrophil", "Granulocyte", "Lymphatic_Endothelial", "Schwann",
]
N_CLASS = len(CLASSES)

CROP = 224
PHIKON_DIM = 1024


LORA_RANK = 16
LORA_ALPHA = 32
LORA_DROPOUT = 0.05
LORA_LR = 1e-4
LORA_HEAD_LR = 1e-3
LORA_EPOCHS = 5
LORA_BATCH_SIZE = 64
LORA_WARMUP = 1
NUM_WORKERS = 8


class LoRALinearWithBase(nn.Module):
    def __init__(self, base: nn.Linear, rank=8, alpha=16, dropout=0.05):
        super().__init__()
        self.base = base
        in_f, out_f = base.in_features, base.out_features
        self.scaling = alpha / rank
        self.lora_A = nn.Parameter(torch.zeros(rank, in_f))
        self.lora_B = nn.Parameter(torch.zeros(out_f, rank))
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B)

    def forward(self, x):
        return self.base(x) + (self.dropout(x) @ self.lora_A.T @ self.lora_B.T) * self.scaling


def load_phikon(device):
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    from transformers import AutoModel
    m = AutoModel.from_pretrained("owkin/phikon-v2", trust_remote_code=True)
    return m.to(device)


def inject_lora(m, rank, alpha, dropout=0.05):
    for p in m.parameters():
        p.requires_grad = False
    dev = next(m.parameters()).device
    modules = []
    for layer in m.encoder.layer:
        attn = layer.attention.attention
        lq = LoRALinearWithBase(attn.query, rank=rank, alpha=alpha, dropout=dropout).to(dev)
        lv = LoRALinearWithBase(attn.value, rank=rank, alpha=alpha, dropout=dropout).to(dev)
        attn.query = lq
        attn.value = lv
        modules.append(lq)
        modules.append(lv)
    n_lora = sum(p.numel() for mod in modules for p in mod.parameters() if p.requires_grad)
    n_total = sum(p.numel() for p in m.parameters())
    print(f"LoRA injected (Phikon Dinov2): {len(m.encoder.layer)} blocks x (Q,V), "
          f"{n_lora:,} trainable / {n_total:,} total ({100 * n_lora / n_total:.2f}%)", flush=True)
    return modules


def get_lora_state_dict(m):
    return {name: prm.data.cpu() for name, prm in m.named_parameters() if prm.requires_grad}


@torch.no_grad()
def forward_cls(model, batch):
    out = model(batch)
    return out.last_hidden_state[:, 0, :]


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


def load_cropbank(fold_idx):
    p_tr = CROPBANK_DIR / f"cropbank_fold{fold_idx}_train.npy"
    p_try = CROPBANK_DIR / f"cropbank_fold{fold_idx}_train_y.npy"
    p_va = CROPBANK_DIR / f"cropbank_fold{fold_idx}_val.npy"
    p_vay = CROPBANK_DIR / f"cropbank_fold{fold_idx}_val_y.npy"
    if not all(p.exists() for p in (p_tr, p_try, p_va, p_vay)):
        return None
    tc = np.load(p_tr)
    tl = np.load(p_try).astype(np.int64)
    vc = np.load(p_va)
    vl = np.load(p_vay).astype(np.int64)
    print(f"[fold {fold_idx}] loaded crop bank: train={len(tl)} val={len(vl)} "
          f"({tc.nbytes/1e9:.1f}GB) from {CROPBANK_DIR}", flush=True)
    return tc, tl, vc, vl


def cosine_warmup_lambda(epoch, warmup, total, min_lr_frac=1e-6):
    if epoch < warmup:
        return (epoch + 1) / warmup
    p = (epoch - warmup) / max(1, total - warmup)
    return max(min_lr_frac, 0.5 * (1 + np.cos(np.pi * p)))


def train_fold(fold_idx, device, smoke):
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    save_path = OUT_DIR / f"lora_phikon{_SFX}_fold{fold_idx}.pt"
    if save_path.exists() and not smoke:
        print(f"Fold {fold_idx}: already exists at {save_path}, skip.", flush=True)
        return
    epochs = int(os.environ.get("SMOKE_EPOCHS", LORA_EPOCHS)) if smoke else LORA_EPOCHS

    print(f"\n{'='*60}\nIter5 Phikon-v2 LoRA — fold {fold_idx} "
          f"({'SMOKE' if smoke else 'FULL'})\n{'='*60}", flush=True)
    t0 = time.time()

    bank = load_cropbank(fold_idx)
    if bank is None:
        print(f"FATAL: crop bank for fold {fold_idx} not found in {CROPBANK_DIR}", flush=True)
        sys.exit(6)
    train_crops, train_labels, val_crops, val_labels = bank

    if smoke:

        cap = int(os.environ.get("SMOKE_CAP", 17_000))
        vcap = int(os.environ.get("SMOKE_VAL", 4_000))
        rng = np.random.RandomState(123 + fold_idx)
        if len(train_labels) > cap:
            sel = rng.choice(len(train_labels), cap, replace=False)
            train_crops, train_labels = train_crops[sel], train_labels[sel]
        if len(val_labels) > vcap:
            sel = rng.choice(len(val_labels), vcap, replace=False)
            val_crops, val_labels = val_crops[sel], val_labels[sel]
        print(f"[fold {fold_idx}] SMOKE subsample: train={len(train_labels)} val={len(val_labels)}", flush=True)

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

    model = load_phikon(device)
    lora_modules = inject_lora(model, LORA_RANK, LORA_ALPHA, LORA_DROPOUT)
    head = nn.Linear(PHIKON_DIM, N_CLASS).to(device)

    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True

    lora_params = [p for mod in lora_modules for p in mod.parameters() if p.requires_grad]
    opt = torch.optim.AdamW([
        {"params": lora_params, "lr": LORA_LR},
        {"params": head.parameters(), "lr": LORA_HEAD_LR},
    ], weight_decay=0.01)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda ep: cosine_warmup_lambda(ep, LORA_WARMUP, epochs))
    crit = nn.CrossEntropyLoss(label_smoothing=0.1)

    best_auc, best_lora_sd, best_head_sd = -1.0, None, None
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
                feats = model(imgs).last_hidden_state[:, 0, :]
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
                        feats = model(imgs).last_hidden_state[:, 0, :]
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

    out_path = OUT_DIR / (f"lora_phikon{_SFX}_fold{fold_idx}.pt" if not smoke
                          else f"lora_phikon{_SFX}_fold{fold_idx}_SMOKE.pt")
    torch.save({
        "lora_state_dict": best_lora_sd,
        "head_state_dict": best_head_sd,
        "val_auc": best_auc,
        "cell_types": CLASSES,
        "lora_config": {"rank": LORA_RANK, "alpha": LORA_ALPHA, "dropout": LORA_DROPOUT,
                        "lora_lr": LORA_LR, "head_lr": LORA_HEAD_LR, "epochs": epochs,
                        "label_smoothing": 0.1, "inject": "Dinov2 query+value per block",
                        "stain_mode": STAIN_MODE},
        "backbone": "owkin/phikon-v2",
        "feat_dim": PHIKON_DIM,
        "fold": fold_idx,
        "history": history,
        "n_train": int(len(train_labels)), "n_val": int(len(val_labels)),
        "note": f"Iter5 Phikon-v2 17-class LoRA r16 (Q,V) on UNI2-built per-fold crop banks; stain_mode={STAIN_MODE}.",
    }, out_path)
    wall = time.time() - t0
    print(f"\nSaved: {out_path} (best val AUC={best_auc:.3f}); total {wall:.0f}s "
          f"({wall/3600:.2f}h)", flush=True)

    tag = "smoke" if smoke else "full"
    import transformers
    log = {"fold": fold_idx, "smoke": bool(smoke), "best_val_auc": round(best_auc, 4),
           "history": history, "n_train": int(len(train_labels)),
           "n_val": int(len(val_labels)), "wall_sec": round(wall, 1),
           "torch": torch.__version__, "transformers": transformers.__version__,
           "node": os.environ.get("SLURMD_NODENAME", "?"),
           "jobid": os.environ.get("SLURM_JOB_ID", "?")}
    (LOG_DIR / f"lora_phikon_train_fold{fold_idx}_{tag}.json").write_text(json.dumps(log, indent=2))

    del model, head, opt, sched
    torch.cuda.empty_cache()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fold", type=int, required=True, choices=[0, 1, 2, 3, 4])
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--cropbank", action="store_true",
                    help="(required) load pre-built per-fold crop bank (.npy)")
    args = ap.parse_args()
    if not args.cropbank:
        print("FATAL: --cropbank required (this script only consumes pre-built crop banks)", flush=True)
        sys.exit(2)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device} cuda_available={torch.cuda.is_available()} torch={torch.__version__}", flush=True)
    if device.type != "cuda":
        print("FATAL: no CUDA visible (pyxis?). Abort to avoid CPU-only LoRA train.", flush=True)
        sys.exit(2)
    train_fold(args.fold, device, args.smoke)


if __name__ == "__main__":
    main()
