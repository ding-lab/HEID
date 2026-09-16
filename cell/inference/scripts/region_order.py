#!/usr/bin/env python -u
from __future__ import annotations

import numpy as np


def serpentine_order(cx, cy) -> np.ndarray:
    cx = np.asarray(cx, float)
    cy = np.asarray(cy, float)
    n = len(cx)
    if n == 0:
        return np.zeros(0, int)
    if n == 1:
        return np.ones(1, int)

    n_bands = max(1, int(round(np.sqrt(n))))
    y_lo, y_hi = cy.min(), cy.max()
    span = y_hi - y_lo
    if span <= 0:
        band = np.zeros(n, int)
    else:
        band = np.clip(((cy - y_lo) / span * n_bands).astype(int), 0, n_bands - 1)

    order = []
    for b in range(n_bands):
        members = np.flatnonzero(band == b)
        if not len(members):
            continue


        members = members[np.argsort(cx[members] * (1 if b % 2 == 0 else -1))]
        order.extend(members.tolist())

    rank = np.zeros(n, int)
    for position, index in enumerate(order, 1):
        rank[index] = position
    return rank
