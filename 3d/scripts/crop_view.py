#!/usr/bin/env python
from __future__ import annotations

import os

import numpy as np


def box():
    v = os.environ.get("BLOCK_CROP", "").strip()
    if not v:
        return None
    x0, y0, x1, y1 = (int(float(t)) for t in v.split(","))
    return x0, y0, x1, y1


def tissue_mask():
    v = os.environ.get("BLOCK_MASK", "").strip()
    if not v or not os.path.exists(v):
        return None
    z = np.load(v)
    return z["mask"].astype(bool), float(z["scale_y"]), float(z["scale_x"])


def install(v22lib):
    b = box()
    if b is None or getattr(v22lib, "_BLOCK_CROPPED", False):
        return b
    x0, y0, x1, y1 = b
    mk = tissue_mask()
    Base = v22lib.Canvas

    class CropCanvas(Base):
        def __init__(self, *a, **k):
            super().__init__(*a, **k)
            self._ox, self._oy = x0, y0


            self.W = min(x1, self.W) - x0
            self.H = min(y1, self.H) - y0

        def read(self, ry0, rx0, ry1, rx1, *a, **k):


            ry0 = max(0, int(ry0)); rx0 = max(0, int(rx0))
            ry1 = min(self.H, int(ry1)); rx1 = min(self.W, int(rx1))
            if ry1 <= ry0 or rx1 <= rx0:
                return np.zeros((0, 0, 3), np.uint8)
            win = np.asarray(self.z[ry0 + self._oy: ry1 + self._oy,
                                    rx0 + self._ox: rx1 + self._ox])
            if win.ndim == 2:
                win = np.repeat(win[:, :, None], 3, 2)
            win = np.ascontiguousarray(win[:, :, :3])
            if mk is not None:
                m, sy, sx = mk
                iy = np.clip(((np.arange(ry0, ry1) + self._oy) / sy).astype(np.int64), 0, m.shape[0] - 1)
                ix = np.clip(((np.arange(rx0, rx1) + self._ox) / sx).astype(np.int64), 0, m.shape[1] - 1)
                win = np.where(m[np.ix_(iy, ix)][:, :, None], win, np.uint8(255))
            return win

    v22lib.Canvas = CropCanvas
    v22lib._BLOCK_CROPPED = True
    return b
