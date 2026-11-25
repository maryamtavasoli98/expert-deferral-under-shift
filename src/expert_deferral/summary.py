"""Aggregation over seeds (mean and sample std)."""

from __future__ import annotations

from collections import defaultdict

import numpy as np


def mean_std(vals) -> tuple[float, float]:
    a = np.asarray(vals, float)
    return float(a.mean()), float(a.std(ddof=1)) if len(a) > 1 else 0.0


def group(records: list[dict], keys: list[str], value: str) -> dict[tuple, tuple[float, float]]:
    g = defaultdict(list)
    for r in records:
        g[tuple(r[k] for k in keys)].append(r[value])
    return {k: mean_std(v) for k, v in g.items()}
