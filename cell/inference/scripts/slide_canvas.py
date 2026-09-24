#!/usr/bin/env python3
import os
from functools import lru_cache
import numpy as np

import det_config as C

PXSIZE_UM = C.PXSIZE_UM
UM_PER_PX = PXSIZE_UM
PX_PER_UM = 1.0 / PXSIZE_UM


def _target():
    cancer = os.environ.get("DET_CANCER", "pancancer")
    return (None if cancer.lower() in ("pancancer", "pan", "all", "") else cancer.upper())


def _include_flagged():
    return os.environ.get("DET_INCLUDE_FLAGGED", "0") == "1"


@lru_cache(maxsize=1)
def _sample_index():
    rows = C.load_manifest(cancer=None, clean_only=False, include_flagged=True)
    return {r["sample"]: r for r in rows}


def usable_samples():
    m = C.load_manifest(cancer=_target(), clean_only=True, include_flagged=_include_flagged())
    return [{"sample": r["sample"], "ncc": r["ncc"], "test_fold": r["fold"]} for r in m]


def find_he(sample):
    r = _sample_index().get(sample)
    if r is None:
        return None, None
    return C.resolve_he(r["cohort"], r["cancer"], sample)


class Canvas:

    def __init__(self, path):
        import tifffile
        import zarr
        self.path = path
        self._tif = tifffile.TiffFile(path)
        store = self._tif.series[0].aszarr()
        zg = zarr.open(store, mode="r")
        self.z = zg["0"] if isinstance(zg, zarr.hierarchy.Group) else zg
        self.H, self.W = int(self.z.shape[0]), int(self.z.shape[1])

    def read(self, y0, x0, y1, x1):
        y0 = max(0, int(y0)); x0 = max(0, int(x0))
        y1 = min(self.H, int(y1)); x1 = min(self.W, int(x1))
        if y1 <= y0 or x1 <= x0:
            return np.zeros((0, 0, 3), np.uint8)
        win = np.asarray(self.z[y0:y1, x0:x1])
        if win.ndim == 2:
            win = np.repeat(win[:, :, None], 3, axis=2)
        return win[:, :, :3].astype(np.uint8, copy=False)

    def close(self):
        try:
            self._tif.close()
        except Exception:
            pass


def tissue_fraction(rgb, white_thresh=220):
    if rgb.size == 0:
        return 0.0
    g = rgb.reshape(-1, rgb.shape[-1]).mean(axis=1)
    return float((g < white_thresh).mean())
