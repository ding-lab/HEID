#!/usr/bin/env python3
import argparse
import json
import sys
import time
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import tifffile
import zarr
from PIL import Image
UM_PER_PX = 0.2125
FIELD_PX = 602
MODEL_INPUT_SIZE = 224
BATCH_SIZE = 64
FEAT_DIM = 1536
UNI2_WEIGHTS = None
IMAGENET_MEAN = np.array([0.485,0.456,0.406],dtype=np.float32)
IMAGENET_STD = np.array([0.229,0.224,0.225],dtype=np.float32)


def build_tiles(grid, xmin, ymin, resolution_um, native_mpp, region_ids=None):
    grid = np.asarray(grid)
    if grid.ndim != 2 or not np.isfinite([xmin, ymin, resolution_um, native_mpp]).all() or native_mpp <= 0 or resolution_um <= 0:
        raise ValueError("Require a two-dimensional region grid, finite origin and positive physical scales")
    if not np.isfinite(grid).all() or np.any(grid < 0) or np.any(grid != np.floor(grid)):
        raise ValueError("Region grid must contain finite nonnegative integer IDs")
    field_px = int(round(128.0/native_mpp))
    stride = field_px//2
    if stride < 1:
        raise ValueError("Native pixel scale is too coarse for 128 micrometer tiling")
    regions = set(int(r) for r in (np.unique(grid) if region_ids is None else region_ids) if r > 0)
    ny,nx=grid.shape
    max_x_px=(xmin+nx*resolution_um)/native_mpp
    max_y_px=(ymin+ny*resolution_um)/native_mpp
    rows=[]
    y_px=0
    while y_px+field_px <= max_y_px+field_px:
        x_px=0
        while x_px+field_px <= max_x_px+field_px:
            ix=int(np.floor(((x_px+field_px/2)*native_mpp-xmin)/resolution_um))
            iy=int(np.floor(((y_px+field_px/2)*native_mpp-ymin)/resolution_um))
            if 0<=ix<nx and 0<=iy<ny:
                rid=int(grid[iy,ix])
                if rid in regions:rows.append((int(x_px),int(y_px),rid))
            x_px+=stride
        y_px+=stride
    return pd.DataFrame(rows,columns=["x_px","y_px","region"])

def load_uni2(device: torch.device) -> torch.nn.Module:
    import timm
    from timm.layers import SwiGLUPacked

    model = timm.create_model(
        "vit_giant_patch14_224",
        img_size=MODEL_INPUT_SIZE, patch_size=14, depth=24, num_heads=24,
        init_values=1e-5, embed_dim=FEAT_DIM, mlp_ratio=2.66667 * 2,
        num_classes=0, no_embed_class=True, mlp_layer=SwiGLUPacked,
        act_layer=torch.nn.SiLU, reg_tokens=8, dynamic_img_size=True,
    )
    state_dict = torch.load(str(UNI2_WEIGHTS), map_location="cpu", weights_only=True)
    model.load_state_dict(state_dict, strict=True)
    model = model.to(device).eval()
    for parameter in model.parameters():
        parameter.requires_grad = False
    with torch.no_grad():
        output = model(torch.zeros(1, 3, MODEL_INPUT_SIZE, MODEL_INPUT_SIZE,
                                   device=device, dtype=torch.float32))
    if output.shape != (1, FEAT_DIM):
        raise ValueError(f"Unexpected UNI2 output shape {output.shape}; require (1, {FEAT_DIM})")
    return model

def normalize_tile(rgb_np: np.ndarray) -> torch.Tensor:
    x = rgb_np.astype(np.float32) / 255.0
    x = (x - IMAGENET_MEAN) / IMAGENET_STD

    x = np.transpose(x, (2, 0, 1))
    return torch.from_numpy(x)

def extract_features_for_core(core: str, tiles_df: pd.DataFrame,
                               model: torch.nn.Module, device: torch.device,
                               he_path: Path) -> torch.Tensor:
    n_tiles = len(tiles_df)
    print(f"  Total tiles to extract: {n_tiles}")


    tif = tifffile.TiffFile(str(he_path))
    store = tif.series[0].aszarr()
    grp = zarr.open(store, mode="r")
    wsi = grp if hasattr(grp, "shape") else grp["0"]
    img_h, img_w = int(wsi.shape[0]), int(wsi.shape[1])
    print(f"  WSI full-res: {img_h} x {img_w}")

    all_features = torch.zeros(n_tiles, FEAT_DIM, dtype=torch.float32)
    t0 = time.time()

    x_px_arr = tiles_df["x_px"].values
    y_px_arr = tiles_df["y_px"].values

    batch_tensors = []
    batch_indices = []

    def flush_batch():
        if not batch_tensors:
            return
        imgs = torch.stack(batch_tensors, dim=0).to(device, non_blocking=True)
        with torch.no_grad():
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16,
                                enabled=(device.type == "cuda")):
                feats = model(imgs)
        all_features[batch_indices] = feats.cpu().float()
        batch_tensors.clear()
        batch_indices.clear()

    for i in range(n_tiles):
        x = int(x_px_arr[i])
        y = int(y_px_arr[i])


        r_min = y
        r_max = y + FIELD_PX
        c_min = x
        c_max = x + FIELD_PX

        vr_min = max(0, r_min)
        vr_max = min(img_h, r_max)
        vc_min = max(0, c_min)
        vc_max = min(img_w, c_max)

        patch = np.zeros((FIELD_PX, FIELD_PX, 3), dtype=np.uint8)
        if vr_max > vr_min and vc_max > vc_min:
            region = wsi[vr_min:vr_max, vc_min:vc_max, :3]
            if not isinstance(region, np.ndarray):
                region = np.array(region)

            dst_r0 = vr_min - r_min
            dst_r1 = dst_r0 + (vr_max - vr_min)
            dst_c0 = vc_min - c_min
            dst_c1 = dst_c0 + (vc_max - vc_min)
            patch[dst_r0:dst_r1, dst_c0:dst_c1] = region


        pil_img = Image.fromarray(patch, mode="RGB")
        pil_img = pil_img.resize((MODEL_INPUT_SIZE, MODEL_INPUT_SIZE),
                                  Image.BICUBIC)
        tensor = normalize_tile(np.array(pil_img))
        batch_tensors.append(tensor)
        batch_indices.append(i)

        if len(batch_tensors) >= BATCH_SIZE:
            flush_batch()

        if (i + 1) % 500 == 0 or (i + 1) == n_tiles:
            elapsed = time.time() - t0
            rate = (i + 1) / elapsed
            print(f"  [{i+1}/{n_tiles}] {elapsed:.0f}s elapsed, "
                  f"{rate:.1f} tiles/s")

    flush_batch()
    store.close()
    tif.close()
    return all_features

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image",type=Path,required=True)
    parser.add_argument("--regions",type=Path,required=True,help="NPZ with tls_grid,x_min,y_min,res_um in the image physical coordinate frame")
    parser.add_argument("--native-mpp",type=float,required=True)
    parser.add_argument("--weights",type=Path,required=True)
    parser.add_argument("--output",type=Path,required=True)
    parser.add_argument("--region-ids",help="Optional comma-separated accepted region IDs")
    parser.add_argument("--device",default="cpu")
    parser.add_argument("--batch-size",type=int,default=64)
    args=parser.parse_args()
    with np.load(args.regions,allow_pickle=False) as grid:
        ids=None if args.region_ids is None else [int(x) for x in args.region_ids.split(",")]
        tiles=build_tiles(grid["tls_grid"],float(grid["x_min"]),float(grid["y_min"]),float(grid["res_um"]),args.native_mpp,ids)
    global UM_PER_PX,FIELD_PX,UNI2_WEIGHTS,BATCH_SIZE
    UM_PER_PX=args.native_mpp
    FIELD_PX=int(round(128.0/UM_PER_PX))
    UNI2_WEIGHTS=args.weights
    BATCH_SIZE=args.batch_size
    if BATCH_SIZE < 1:raise ValueError("Batch size must be positive")
    if len(tiles):
        device=torch.device(args.device)
        model=load_uni2(device)
        features=extract_features_for_core(args.image.stem,tiles,model,device,args.image)
    else:features=torch.empty((0,1536),dtype=torch.float32)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    torch.save({"features":features.numpy(),"region":tiles.region.to_numpy(np.int64),
                "x_px":tiles.x_px.to_numpy(np.int64),"y_px":tiles.y_px.to_numpy(np.int64),
                "field_um":128.,"native_mpp":UM_PER_PX,"field_px":FIELD_PX},args.output)
    print(f"Saved {len(tiles)} region-keyed feature tiles to {args.output}")

if __name__ == "__main__":main()
