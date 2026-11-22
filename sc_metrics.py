"""Accuracy metrics."""

from __future__ import annotations

import numpy as np


def _arrays(actual, forecast):
    a = np.asarray(actual, dtype=float)
    f = np.asarray(forecast, dtype=float)
    mask = np.isfinite(a) & np.isfinite(f)
    return a[mask], f[mask]


def wape(actual, forecast) -> float:
    """Weighted absolute percentage error: sum|A - F| / sum|A|."""
    a, f = _arrays(actual, forecast)
    denom = np.abs(a).sum()
    return float(np.abs(a - f).sum() / denom) if denom > 0 else np.nan


def accuracy(actual, forecast) -> float:
    """1 - WAPE, floored at 0."""
    w = wape(actual, forecast)
    return float(max(0.0, 1.0 - w)) if np.isfinite(w) else np.nan


def bias(actual, forecast) -> float:
    """Signed error over actual: positive means the forecast was above the actual."""
    a, f = _arrays(actual, forecast)
    denom = np.abs(a).sum()
    return float((f - a).sum() / denom) if denom > 0 else np.nan


def rmse(actual, forecast) -> float:
    a, f = _arrays(actual, forecast)
    return float(np.sqrt(np.mean((a - f) ** 2))) if len(a) else np.nan
