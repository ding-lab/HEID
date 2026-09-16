from __future__ import annotations

import numpy as np

XPX = 0.2125
BAND_NATIVE = 8192
CROP_HALF_NATIVE_PAD = 512


class BandView:
    def __init__(self, band: np.ndarray, y0_native: int, mpp: float, height: int, width: int):
        self.band = band
        self.y0 = y0_native
        self.mpp = mpp
        self.shape = (height, width, 3)

    def __getitem__(self, key):
        import cv2
        sy, sx = key[0], key[1]
        y0, y1, x0, x1 = int(sy.start), int(sy.stop), int(sx.start), int(sx.stop)
        h, w = y1 - y0, x1 - x0
        if h <= 0 or w <= 0:
            return np.zeros((max(h, 0), max(w, 0), 3), np.uint8)
        ny0 = int(np.floor(y0 * XPX / self.mpp)) - self.y0
        ny1 = int(np.ceil(y1 * XPX / self.mpp)) - self.y0
        nx0 = int(np.floor(x0 * XPX / self.mpp))
        nx1 = int(np.ceil(x1 * XPX / self.mpp))
        bh, bw = self.band.shape[0], self.band.shape[1]
        cy0, cy1 = max(0, ny0), min(bh, max(ny1, ny0 + 1))
        cx0, cx1 = max(0, nx0), min(bw, max(nx1, nx0 + 1))
        if cy1 <= cy0 or cx1 <= cx0:
            return np.full((h, w, 3), 255, np.uint8)
        sub = self.band[cy0:cy1, cx0:cx1]
        if (cy1 - cy0, cx1 - cx0) != (ny1 - ny0, nx1 - nx0):
            padded = np.full((ny1 - ny0, nx1 - nx0, 3), 255, np.uint8)
            padded[cy0 - ny0:cy1 - ny0, cx0 - nx0:cx1 - nx0] = sub
            sub = padded
        return cv2.resize(sub, (w, h), interpolation=cv2.INTER_LINEAR)


def bands_for(ys_native: np.ndarray, height: int) -> list[tuple[int, int]]:
    if not len(ys_native):
        return []
    lo = max(0, int(ys_native.min()) - CROP_HALF_NATIVE_PAD)
    hi = min(height, int(ys_native.max()) + CROP_HALF_NATIVE_PAD + 1)
    out = []
    for start in range(lo, hi, BAND_NATIVE):
        out.append((start, min(height, start + BAND_NATIVE)))
    return out
