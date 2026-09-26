"""Shared helpers: synthetic order books and dataset fixtures."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

# So test modules (also in tests/bench/) can `from conftest import make_book`.
sys.path.insert(0, str(Path(__file__).parent))

from lobimb import data  # noqa: E402

TICK = 0.1


def make_levels(n: int, seed: int = 0, mid0: float = 17_000.0) -> np.ndarray:
    """Random well-formed books: best bid walks +/-1 tick, spread 1-2 ticks,
    exponential volumes with mean 2 BTC."""
    rng = np.random.default_rng(seed)
    best_bid = mid0 + np.cumsum(rng.integers(-1, 2, size=n)) * TICK
    spread_ticks = rng.integers(1, 3, size=n)
    levels = np.empty((n, data.ROW_WIDTH))
    for i in range(data.DEPTH):
        levels[:, 2 * i] = best_bid - i * TICK
        levels[:, data.ASK_OFFSET + 2 * i] = best_bid + (spread_ticks + i) * TICK
        levels[:, 2 * i + 1] = rng.exponential(2.0, size=n)
        levels[:, data.ASK_OFFSET + 2 * i + 1] = rng.exponential(2.0, size=n)
    return levels


def make_book(n: int, seed: int = 0, step_ms: int = 250) -> data.Book:
    """`make_levels` with regular timestamps (no gaps)."""
    ts = 1_673_302_660_926 + np.arange(n, dtype=np.int64) * step_ms
    return data.Book(ts, make_levels(n, seed))


def flat_book(bids: list[float], asks: list[float], step_ms: int = 250,
              volume: float = 1.0) -> data.Book:
    """Book with the given best bid/ask paths, for hand-computed PnL.

    Timestamps are 0, step_ms, 2*step_ms, ..., so H ms is exactly H / step_ms
    snapshots.
    """
    n = len(bids)
    levels = np.zeros((n, data.ROW_WIDTH))
    for i in range(data.DEPTH):
        levels[:, 2 * i] = np.asarray(bids) - i * TICK
        levels[:, data.ASK_OFFSET + 2 * i] = np.asarray(asks) + i * TICK
        levels[:, 2 * i + 1] = volume
        levels[:, data.ASK_OFFSET + 2 * i + 1] = volume
    ts = np.arange(n, dtype=np.int64) * step_ms
    return data.Book(ts, levels)


@pytest.fixture(scope="session")
def random_levels() -> np.ndarray:
    # 50k rows: above the engine's 16,384-row OpenMP threshold.
    return make_levels(50_000, seed=42)


@pytest.fixture(scope="session")
def real_book() -> data.Book:
    if not data.cache_exists():
        pytest.skip("dataset cache missing; run scripts/prepare_data.py")
    return data.load()
