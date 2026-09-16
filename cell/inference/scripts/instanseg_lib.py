#!/usr/bin/env python3
import os
import sys
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import slide_canvas as L
import det_config as C


def _stub_instanseg_viz_modules():
    import types
    specs = {
        "instanseg.utils.visualization": (
            "apply_cmap", "display_as_grid", "display_cells_and_nuclei",
            "display_colourized", "label_to_color_image", "plot_average",
            "save_image_with_label_overlay", "show_images", "_display_overlay"),
        "instanseg.utils.biological_utils": (
            "resolve_cell_and_nucleus_boundaries",),
    }
    for key, names in specs.items():
        if key in sys.modules:
            continue
        mod = types.ModuleType(key)
        for n in names:
            setattr(mod, n, lambda *a, **k: None)
        sys.modules[key] = mod


_stub_instanseg_viz_modules()

MODEL_PIXEL_SIZE = 0.5

WARMSTART_ARCH = dict(
    model_str="InstanSeg_UNet", dim_in=3, dim_coords=2, n_sigma=2, dim_seeds=1,
    cells_and_nuclei=False, multihead=False, layers=[32, 64, 128, 256],
    norm="BATCH", dropprob=0.0, mlp_width=5, feature_engineering="0",
)


def build_model_and_loss(device="cuda", warmstart=True, verbose=True):
    import torch
    from instanseg.utils.model_loader import build_model_from_dict
    from instanseg.utils.loss.instanseg_loss import InstanSeg as LossModule

    a = WARMSTART_ARCH
    model = build_model_from_dict(dict(a))
    method = LossModule(n_sigma=a["n_sigma"], dim_coords=a["dim_coords"],
                        dim_seeds=a["dim_seeds"], device=device,
                        feature_engineering_function=a["feature_engineering"])
    model = method.initialize_pixel_classifier(model, MLP_width=a["mlp_width"])

    matched, total = 0, len(model.state_dict())
    if warmstart:
        from instanseg.utils.utils import download_model
        ts = download_model("brightfield_nuclei", verbose=False)
        tsd = ts.state_dict()
        msd = model.state_dict()
        new_sd = {}
        for k, v in msd.items():
            tk = k if k.startswith("pixel_classifier") else "fcn." + k
            if tk in tsd and tuple(tsd[tk].shape) == tuple(v.shape):
                new_sd[k] = tsd[tk]; matched += 1
            else:
                new_sd[k] = v
        model.load_state_dict(new_sd, strict=True)
        if verbose:
            print(f"[warmstart] loaded {matched}/{total} params from brightfield_nuclei", flush=True)
    model = model.to(device)
    method.device = device
    method.pixel_classifier = method.pixel_classifier.to(device)
    return model, method, {"warmstart_matched": matched, "warmstart_total": total}


def normalize_rgb(rgb_uint8):
    import torch
    from instanseg.utils.utils import percentile_normalize
    x = rgb_uint8.astype(np.float32)
    x = percentile_normalize(x)
    x = np.ascontiguousarray(x.transpose(2, 0, 1))
    return torch.from_numpy(x)


def _read_rg(path, columns):
    pf = pq.ParquetFile(path)
    tabs = [pf.read_row_group(i, columns=list(columns)) for i in range(pf.num_row_groups)]
    return pa.concat_tables(tabs).to_pandas()


def read_nucleus_polygons(sample):
    path = C.xcache_path(L._cancer_of(sample), sample, "nucleus_boundaries")
    df = _read_rg(path, ["cell_id", "vertex_x", "vertex_y"])
    cid = df["cell_id"].to_numpy()
    vx = df["vertex_x"].to_numpy(np.float32)
    vy = df["vertex_y"].to_numpy(np.float32)
    change = np.empty(len(cid), dtype=bool)
    change[0] = True
    change[1:] = cid[1:] != cid[:-1]
    starts = np.flatnonzero(change)
    ends = np.append(starts[1:], len(cid))
    polys, cents = [], []
    for s, e in zip(starts, ends):
        v = np.stack([vx[s:e], vy[s:e]], axis=1)
        polys.append((v, cid[s]))
        cents.append(v.mean(axis=0))
    cents = np.asarray(cents, dtype=np.float32) if cents else np.zeros((0, 2), np.float32)
    return polys, cents


def read_ignore_centroids(sample):
    path = C.xcache_path(L._cancer_of(sample), sample, "cells")
    df = _read_rg(path, ["x_centroid", "y_centroid", "nucleus_area"])
    m = df["nucleus_area"].isna().to_numpy()
    return df.loc[m, ["x_centroid", "y_centroid"]].to_numpy(np.float32)


def load_offset_um(offset_json, sample):
    dxdy = np.zeros(2, np.float32)
    if offset_json and os.path.exists(offset_json):
        import json
        with open(offset_json) as f:
            d = json.load(f)
        entry = d.get(sample)
        if isinstance(entry, (list, tuple)):
            dxdy = np.asarray(entry, np.float32)
        elif isinstance(entry, dict) and "global" in entry:
            dxdy = np.asarray(entry["global"], np.float32)

    def apply(pts_um):
        if pts_um.size == 0:
            return pts_um
        return pts_um + dxdy
    apply.dxdy = dxdy
    return apply


def resnap_centroids(centroids_local_px, rgb_tile, enabled=False, **kw):
    if not enabled:
        return centroids_local_px
    raise NotImplementedError("SAM re-snap not yet implemented (hook reserved)")


def decode_centroids(method, pred_chw, return_scores=False, **post_kwargs):
    import torch
    from skimage.measure import regionprops
    with torch.no_grad():
        labels = method.postprocessing(pred_chw, **post_kwargs)
    lab = labels.squeeze().to("cpu").numpy().astype(np.int32)
    if lab.ndim != 2:
        lab = lab.reshape(lab.shape[-2], lab.shape[-1])
    if return_scores:
        seed = (pred_chw[-1].detach().float() / 15.0 + 0.5).to("cpu").numpy()
        cents, scores = [], []
        for p in regionprops(lab, intensity_image=seed):
            cy, cx = p.centroid
            cents.append((cx, cy))
            scores.append(float(p.intensity_max if hasattr(p, "intensity_max") else p.max_intensity))
        return (np.asarray(cents, np.float32) if cents else np.zeros((0, 2), np.float32),
                np.asarray(scores, np.float32) if scores else np.zeros(0, np.float32), lab)
    cents = []
    for p in regionprops(lab):
        cy, cx = p.centroid
        cents.append((cx, cy))
    return (np.asarray(cents, np.float32) if cents else np.zeros((0, 2), np.float32)), lab


def det_prf1(gt_xy, pred_xy, tau_px):
    n_gt, n_pred = len(gt_xy), len(pred_xy)
    m = L.greedy_match(gt_xy.astype(np.float64), pred_xy.astype(np.float64), tau_px, k=5)
    tp = int((m["pred_to_gt"] != -1).sum())
    fp, fn = n_pred - tp, n_gt - tp
    prec = tp / n_pred if n_pred else 0.0
    rec = tp / n_gt if n_gt else 0.0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
    return dict(precision=prec, recall=rec, det_f1=f1, tp=tp, fp=fp, fn=fn,
                n_gt=n_gt, n_pred=n_pred)
