"""Pure-NumPy versions of the C kernels.

Written for clarity rather than speed: they are the test oracle and the
"vectorised NumPy" baseline in perf.numpy_baseline.
"""
from __future__ import annotations

import numpy as np

# Same values as LOB_DEPTH / LOB_ASK_OFFSET in lob_engine.h; keep in sync.
DEPTH = 10
ASK = 2 * DEPTH  # first ask column (LOB_ASK_OFFSET)


def obi(levels: np.ndarray) -> np.ndarray:
    """Level-1 OBI; 0 for an empty top of book, NaN propagated (same as C)."""
    bv, av = levels[:, 1], levels[:, ASK + 1]
    total = bv + av
    with np.errstate(invalid="ignore", divide="ignore"):
        out = (bv - av) / total
    return np.where(total > 0, out, np.where(np.isnan(total), np.nan, 0.0))


def wobi(levels: np.ndarray, depth: int, alpha: float) -> tuple[np.ndarray, np.ndarray]:
    """Depth-weighted imbalance and micro-price with weights exp(-alpha * i).

    Same edge cases as the C kernel: NaN volume gives NaN imbalance, and the
    micro-price falls back to the mid when total volume is 0 or NaN.
    """
    # C uses a running product, so results may differ in the last bit.
    w = np.exp(-alpha) ** np.arange(depth)
    bp = levels[:, 0:2 * depth:2]
    bv = levels[:, 1:2 * depth:2]
    ap = levels[:, ASK:ASK + 2 * depth:2]
    av = levels[:, ASK + 1:ASK + 2 * depth:2]
    wb, wa = bv @ w, av @ w
    num = (bv * ap + av * bp) @ w
    total = wb + wa
    with np.errstate(invalid="ignore", divide="ignore"):
        imb = np.where(total > 0, (wb - wa) / total, np.where(np.isnan(total), np.nan, 0.0))
        micro = np.where(total > 0, num / total, 0.5 * (levels[:, 0] + levels[:, ASK]))
    return imb, micro


def legacy_obi_int(levels: np.ndarray) -> np.ndarray:
    """What the legacy (v1) engine computed: volumes truncated to int.

    Most BTC volumes are below 1, so the signal collapses onto {-1, 0, +1}.
    """
    bv = np.trunc(levels[:, 1])
    av = np.trunc(levels[:, ASK + 1])
    total = bv + av
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(total != 0, (bv - av) / total, 0.0)
