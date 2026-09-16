#!/usr/bin/env python
import numpy as np


_BT601 = np.array([0.299, 0.587, 0.114], np.float32)


def to_grayscale_bt601(rgb):
    a = rgb.astype(np.float32)
    y = a @ _BT601
    y = np.clip(np.rint(y), 0, 255).astype(np.uint8)
    return np.repeat(y[:, :, None], 3, axis=2)


def intensity_normalize(rgb, low=1.0, high=99.0, per_channel=True, eps=1e-6):
    a = rgb.astype(np.float32)
    if per_channel:
        out = np.empty_like(a)
        for c in range(a.shape[2]):
            lo = np.percentile(a[..., c], low)
            hi = np.percentile(a[..., c], high)
            out[..., c] = (a[..., c] - lo) / max(hi - lo, eps)
        out = np.clip(out, 0.0, 1.0) * 255.0
    else:
        luma = a @ _BT601
        lo = np.percentile(luma, low)
        hi = np.percentile(luma, high)
        out = np.clip((a - lo) / max(hi - lo, eps), 0.0, 1.0) * 255.0
    return np.rint(out).astype(np.uint8)


_HE_OD = np.array([[0.65, 0.70, 0.29],
                   [0.07, 0.99, 0.11],
                   [0.27, 0.57, 0.78]], np.float32)
_HE_INV = np.linalg.inv(_HE_OD).astype(np.float32)


def he_color_descriptor(rgb, eps=1e-6):
    a = rgb.astype(np.float32)
    mR = float(a[..., 0].mean()); mG = float(a[..., 1].mean()); mB = float(a[..., 2].mean())
    od = -np.log((a + 1.0) / 256.0)
    conc = od.reshape(-1, 3) @ _HE_INV.T
    h_mean = float(conc[:, 0].mean()); e_mean = float(conc[:, 1].mean())
    return np.array([mR, mG, mB, h_mean, e_mean], np.float32)


class RandStainNA:
    LAB_KEYS = ("L", "A", "B")

    def __init__(self, distribution=None, std_hyper=0.25, probability=1.0):
        self.dist = distribution
        self.std_hyper = float(std_hyper)
        self.p = float(probability)

    def __call__(self, rgb, rng):
        if rng.random() > self.p:
            return rgb
        try:
            import cv2
        except Exception as e:
            raise RuntimeError("RandStainNA needs OpenCV (cv2); not importable: %s" % e)
        lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB).astype(np.float32)
        out = np.empty_like(lab)
        for i, k in enumerate(self.LAB_KEYS):
            ch = lab[..., i]
            m = float(ch.mean()); s = float(ch.std()) + 1e-6
            if self.dist is not None:
                d = self.dist[k]
                tm = rng.normal(d["mean_mean"], d["mean_std"] * self.std_hyper)
                ts = rng.normal(d["std_mean"], d["std_std"] * self.std_hyper)
            else:
                tm = rng.normal(m, abs(m) * self.std_hyper + 1e-6)
                ts = rng.normal(s, s * self.std_hyper)
            ts = max(float(ts), 1e-6)
            out[..., i] = (ch - m) / s * ts + tm
        out = np.clip(out, 0.0, 255.0).astype(np.uint8)
        return cv2.cvtColor(out, cv2.COLOR_LAB2RGB)


def get_stain_fn(mode):
    if mode == "color":
        return None, False
    if mode == "gray":
        return (lambda crop, rng=None: to_grayscale_bt601(crop)), False
    if mode == "randstainna":
        rsn = RandStainNA(std_hyper=0.3, probability=1.0)
        return (lambda crop, rng: rsn(crop, rng)), True
    raise ValueError("unknown stain mode: %s" % mode)
