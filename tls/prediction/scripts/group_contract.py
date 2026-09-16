
from __future__ import annotations

import pandas as pd


class _UnionFind:

    def __init__(self):
        self._parent: dict[str, str] = {}

    def find(self, x: str) -> str:
        self._parent.setdefault(x, x)
        if self._parent[x] != x:
            self._parent[x] = self.find(self._parent[x])
        return self._parent[x]

    def union(self, a: str, b: str) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self._parent[rb] = ra

    def group_of(self, x: str) -> str:
        return self.find(x)


def build_coarse_groups(manifest: pd.DataFrame) -> pd.Series:
    uf = _UnionFind()


    for run, grp in manifest.groupby("xenium_run"):
        samples = list(grp["sample"])
        for s in samples[1:]:
            uf.union(samples[0], s)


    for pat, grp in manifest.groupby("patient_stem"):
        samples = list(grp["sample"])
        for s in samples[1:]:
            uf.union(samples[0], s)


    return manifest["sample"].apply(uf.group_of)
