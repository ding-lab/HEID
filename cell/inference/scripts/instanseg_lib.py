#!/usr/bin/env python3
import os
import sys
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)


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


