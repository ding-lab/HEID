#!/usr/bin/env python
from __future__ import annotations

import os
import sys

import numpy as np


LOCAL_SCRIPTS = os.path.dirname(os.path.abspath(__file__))
if os.path.isdir(LOCAL_SCRIPTS) and LOCAL_SCRIPTS not in sys.path:
    sys.path.insert(0, LOCAL_SCRIPTS)

try:
    from fast_slide import FastSlide
except ImportError as exc:
    raise ImportError(
        f"BandCachedSlide needs `fast_slide`, expected in {LOCAL_SCRIPTS}. "
        f"Original error: {exc}"
    ) from exc


class BandCachedSlide:

    def __init__(self, path: str, band_rows: int = 8192, n_threads: int = 8):
        self._reader = FastSlide(path, n_threads=n_threads)
        self.path = path
        self.H = self._reader.H
        self.W = self._reader.W
        self.band_rows = int(band_rows)
        self._band = None
        self._b0 = 0
        self._b1 = 0
        self.n_band_decodes = 0
        self.rows_decoded = 0

    def read(self, y0, x0, y1, x1) -> np.ndarray:
        y0 = max(0, int(y0)); x0 = max(0, int(x0))
        y1 = min(self.H, int(y1)); x1 = min(self.W, int(x1))
        if y1 <= y0 or x1 <= x0:
            return np.zeros((0, 0, 3), np.uint8)
        if self._band is None or y0 < self._b0 or y1 > self._b1:
            b0 = y0
            b1 = min(self.H, max(y1, y0 + self.band_rows))
            self._band = None
            self._band = self._reader.read(b0, 0, b1, self.W)
            self._b0, self._b1 = b0, b1
            self.n_band_decodes += 1
            self.rows_decoded += b1 - b0
        return np.ascontiguousarray(self._band[y0 - self._b0:y1 - self._b0, x0:x1])

    def close(self):
        self._band = None
        self._reader.close()
