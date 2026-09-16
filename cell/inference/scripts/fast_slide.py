#!/usr/bin/env python
from __future__ import annotations
from pathlib import Path
_RELEASE = Path(__file__).resolve().parents[3]

from concurrent.futures import ThreadPoolExecutor

import numpy as np
import tifffile


class FastSlide:

    COALESCE_GAP = 1 << 20

    def __init__(self, path: str, n_threads: int = 8):
        self.path = path
        self._tif = tifffile.TiffFile(path)
        series = self._tif.series[0]
        self.page = series.levels[0].pages[0]
        if not self.page.is_tiled:
            raise RuntimeError(f"{path}: level 0 is not tiled; this reader needs tiles")
        self.H = int(self.page.imagelength)
        self.W = int(self.page.imagewidth)
        self.th = int(self.page.tilelength)
        self.tw = int(self.page.tilewidth)
        self.tiles_across = (self.W + self.tw - 1) // self.tw
        self.pool = ThreadPoolExecutor(max_workers=n_threads)
        self._fh = self._tif.filehandle


        self.jpegtables = getattr(self.page, "jpegtables", None)
        self.jpegheader = getattr(self.page, "jpegheader", None)

    def read(self, y0: int, x0: int, y1: int, x1: int) -> np.ndarray:
        y0 = max(0, int(y0)); x0 = max(0, int(x0))
        y1 = min(self.H, int(y1)); x1 = min(self.W, int(x1))
        if y1 <= y0 or x1 <= x0:
            return np.zeros((0, 0, 3), np.uint8)

        row0, row1 = y0 // self.th, (y1 - 1) // self.th
        col0, col1 = x0 // self.tw, (x1 - 1) // self.tw
        out = np.zeros((y1 - y0, x1 - x0, 3), np.uint8)

        wanted = []
        for row in range(row0, row1 + 1):
            for col in range(col0, col1 + 1):
                index = row * self.tiles_across + col
                count = int(self.page.databytecounts[index])
                if not count:
                    continue
                wanted.append((int(self.page.dataoffsets[index]), count, index, row, col))


        wanted.sort()
        jobs = []
        run_start = run_end = None
        run_members: list[tuple[int, int, int, int]] = []

        def drain_run():
            if run_start is None:
                return
            self._fh.seek(run_start)
            blob = self._fh.read(run_end - run_start)
            for offset, count, index, row, col in run_members:
                begin = offset - run_start
                jobs.append((index, row, col, blob[begin:begin + count]))

        for offset, count, index, row, col in wanted:
            if run_start is not None and offset - run_end <= self.COALESCE_GAP:
                run_end = max(run_end, offset + count)
                run_members.append((offset, count, index, row, col))
                continue
            drain_run()
            run_start, run_end = offset, offset + count
            run_members = [(offset, count, index, row, col)]
        drain_run()

        def decode(job):
            index, row, col, payload = job
            tile, _, shape = self.page.decode(payload, index,
                                              jpegtables=self.jpegtables,
                                              jpegheader=self.jpegheader)
            return row, col, np.asarray(tile).reshape(shape[1:])

        for row, col, tile in self.pool.map(decode, jobs):
            ty0, tx0 = row * self.th, col * self.tw
            sy0, sx0 = max(y0, ty0), max(x0, tx0)
            sy1 = min(y1, ty0 + tile.shape[0])
            sx1 = min(x1, tx0 + tile.shape[1])
            if sy1 <= sy0 or sx1 <= sx0:
                continue
            out[sy0 - y0:sy1 - y0, sx0 - x0:sx1 - x0] = \
                tile[sy0 - ty0:sy1 - ty0, sx0 - tx0:sx1 - tx0, :3]
        return out

    def close(self):
        self.pool.shutdown(wait=True)
        try:
            self._tif.close()
        except Exception:
            pass


def verify_against(path: str, y0: int, x0: int, y1: int, x1: int) -> dict:
    import sys
    import time
    sys.path.insert(0, str(_RELEASE / "cell/inference/scripts"))
    import slide_canvas as L

    started = time.time()
    canvas = L.Canvas(path)
    slow = canvas.read(y0, x0, y1, x1)
    slow_seconds = time.time() - started
    canvas.close()

    started = time.time()
    fast_reader = FastSlide(path)
    fast = fast_reader.read(y0, x0, y1, x1)
    fast_seconds = time.time() - started
    fast_reader.close()

    return {
        "shape_slow": list(slow.shape), "shape_fast": list(fast.shape),
        "identical": bool(slow.shape == fast.shape and np.array_equal(slow, fast)),
        "max_abs_diff": int(np.abs(slow.astype(np.int16) - fast.astype(np.int16)).max())
        if slow.shape == fast.shape else -1,
        "seconds_slow": round(slow_seconds, 1), "seconds_fast": round(fast_seconds, 1),
        "speedup": round(slow_seconds / max(fast_seconds, 1e-9), 2),
        "megapixels": round(slow.shape[0] * slow.shape[1] / 1e6, 1),
    }


if __name__ == "__main__":
    import argparse
    import json

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--svs", required=True)
    parser.add_argument("--y0", type=int, default=0)
    parser.add_argument("--rows", type=int, default=4096)
    args = parser.parse_args()
    with tifffile.TiffFile(args.svs) as handle:
        width = int(handle.series[0].levels[0].pages[0].imagewidth)
    print(json.dumps(verify_against(args.svs, args.y0, 0, args.y0 + args.rows, width),
                     indent=2))
