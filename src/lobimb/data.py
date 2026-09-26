"""Loading and validation of order book snapshots.

Raw CSV (43 columns): row index, epoch-ms timestamp, timestamp string, then 10
bid and 10 ask levels as (price, volume), best first. Columns 3..42 are already
in engine layout, so `Book.levels` goes to C without reshuffling.

Parsing the 1.2 GB CSV is slow, so it is done once and cached as .npy files,
memory-mapped on later loads.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from .config import CACHE_DIR, RAW_CSV, SNAPSHOT_MS

# Must match LOB_DEPTH / LOB_ROW_WIDTH / LOB_ASK_OFFSET in lob_engine.h.
DEPTH = 10
ROW_WIDTH = 4 * DEPTH
ASK_OFFSET = 2 * DEPTH

CSV_TS_COL = 1
CSV_BOOK_COLS = list(range(3, 3 + ROW_WIDTH))  # columns 3..42


@dataclass(frozen=True)
class Book:
    """Time series of order book snapshots.

    Arrays may be read-only memory maps; slicing returns views.
    """

    ts: np.ndarray      # (n,) int64 epoch milliseconds
    levels: np.ndarray  # (n, 40) float64, engine layout

    def __post_init__(self) -> None:
        # Fail here rather than with out-of-bounds reads in C.
        if self.levels.ndim != 2 or self.levels.shape[1] != ROW_WIDTH:
            raise ValueError(f"levels must have shape (n, {ROW_WIDTH}), got {self.levels.shape}")
        if self.ts.shape != (self.levels.shape[0],):
            raise ValueError("ts and levels must have the same number of rows")

    def __len__(self) -> int:
        return self.ts.shape[0]

    def __getitem__(self, idx: slice) -> "Book":
        return Book(self.ts[idx], self.levels[idx])

    def bid_px(self, level: int = 0) -> np.ndarray:
        return self.levels[:, 2 * level]

    def bid_vol(self, level: int = 0) -> np.ndarray:
        return self.levels[:, 2 * level + 1]

    def ask_px(self, level: int = 0) -> np.ndarray:
        return self.levels[:, ASK_OFFSET + 2 * level]

    def ask_vol(self, level: int = 0) -> np.ndarray:
        return self.levels[:, ASK_OFFSET + 2 * level + 1]


def read_csv(path: Path | str = RAW_CSV, nrows: int | None = None) -> Book:
    """Parse the raw CSV (slow; `load` goes through the cache)."""
    usecols = [CSV_TS_COL, *CSV_BOOK_COLS]
    df = pd.read_csv(path, usecols=usecols, nrows=nrows, engine="c", dtype=np.float64)
    # Epoch ms (~1.7e12) are exact in float64.
    ts = df.iloc[:, 0].to_numpy(dtype=np.int64)
    levels = np.ascontiguousarray(df.iloc[:, 1:].to_numpy(dtype=np.float64))
    return Book(ts, levels)


def _cache_paths(cache_dir: Path) -> tuple[Path, Path]:
    return cache_dir / "ts.npy", cache_dir / "levels.npy"


def write_cache(book: Book, cache_dir: Path = CACHE_DIR) -> None:
    """Save a Book as ts.npy and levels.npy."""
    cache_dir.mkdir(parents=True, exist_ok=True)
    ts_path, lv_path = _cache_paths(cache_dir)
    np.save(ts_path, book.ts)
    np.save(lv_path, book.levels)


def load(cache_dir: Path = CACHE_DIR, csv_path: Path | str = RAW_CSV,
         mmap: bool = True) -> Book:
    """Load the dataset from the .npy cache, building it from the CSV if missing.

    mmap=True returns read-only memory maps; mmap=False reads into RAM (used by
    benchmarks so disk I/O is not timed).
    """
    ts_path, lv_path = _cache_paths(cache_dir)
    if not (ts_path.exists() and lv_path.exists()):
        write_cache(read_csv(csv_path), cache_dir)
    mode = "r" if mmap else None
    return Book(np.load(ts_path, mmap_mode=mode), np.load(lv_path, mmap_mode=mode))


def cache_exists(cache_dir: Path = CACHE_DIR) -> bool:
    return all(p.exists() for p in _cache_paths(cache_dir))


@dataclass(frozen=True)
class SanityReport:
    """Counts of data problems found by `sanity_check`.

    Gaps don't make the data unclean: they come from the feed and are handled
    by the backtests' gap flag.
    """

    n_rows: int
    non_monotonic_ts: int
    crossed: int          # best bid >= best ask
    non_positive_px: int  # any level price <= 0
    negative_vol: int     # any level volume < 0
    nan_rows: int
    gaps: int             # consecutive snapshots further apart than gap_ms
    max_gap_ms: int

    def is_clean(self) -> bool:
        return not (self.non_monotonic_ts or self.crossed or self.non_positive_px
                    or self.negative_vol or self.nan_rows)


def sanity_check(book: Book, gap_ms: int = 2 * SNAPSHOT_MS) -> SanityReport:
    """Count data problems instead of silently dropping rows."""
    lv = np.asarray(book.levels)
    dt = np.diff(np.asarray(book.ts))
    prices = lv[:, 0::2]
    vols = lv[:, 1::2]
    return SanityReport(
        n_rows=len(book),
        non_monotonic_ts=int((dt <= 0).sum()),
        crossed=int((book.bid_px() >= book.ask_px()).sum()),
        non_positive_px=int((prices <= 0).any(axis=1).sum()),
        negative_vol=int((vols < 0).any(axis=1).sum()),
        nan_rows=int(np.isnan(lv).any(axis=1).sum()),
        gaps=int((dt > gap_ms).sum()),
        max_gap_ms=int(dt.max()) if dt.size else 0,
    )


def in_sample_split(book: Book, fraction: float) -> tuple[Book, Book]:
    """Chronological split: first `fraction` of rows in-sample, the rest out-of-sample."""
    if not 0.0 < fraction < 1.0:
        raise ValueError("fraction must be in (0, 1)")
    cut = int(len(book) * fraction)
    return book[:cut], book[cut:]


def _utc(ms: int) -> str:
    return pd.Timestamp(int(ms), unit="ms", tz="UTC").isoformat(timespec="milliseconds")


def describe(book: Book, tick: float = 0.1, in_sample_fraction: float = 0.7) -> dict:
    """Descriptive statistics of a dataset as a JSON-serialisable dict.

    NumPy only (no C engine), so it works on any Book. Columns are processed
    one at a time so a memory-mapped Book is never copied whole. The
    best_volume shares below 1 BTC show how much the legacy int() cast lost.
    """
    n = len(book)
    if n < 2:
        raise ValueError("describe needs at least 2 snapshots")
    ts = np.asarray(book.ts)
    dt = np.diff(ts)
    bid, ask = np.asarray(book.bid_px()), np.asarray(book.ask_px())
    mid = 0.5 * (bid + ask)
    # Differences of 0.1-grid prices carry float noise, hence the rounding below.
    spread = ask - bid
    bv, av = np.asarray(book.bid_vol()), np.asarray(book.ask_vol())
    depth_bid = np.zeros(n)
    depth_ask = np.zeros(n)
    for i in range(DEPTH):
        depth_bid += book.bid_vol(i)
        depth_ask += book.ask_vol(i)
    tot = bv + av
    obi = np.divide(bv - av, tot, out=np.zeros(n), where=tot > 0)
    abs_obi = np.abs(obi)
    ins, oos = in_sample_split(book, in_sample_fraction)

    def vol_stats(v: np.ndarray) -> dict:
        return {"median": float(np.median(v)), "mean": float(v.mean()),
                "share_lt_0_01": float(np.mean(v < 0.01)), "share_lt_1": float(np.mean(v < 1.0))}

    return {
        "period": {"start_utc": _utc(ts[0]), "end_utc": _utc(ts[-1]),
                   "start_ms": int(ts[0]), "end_ms": int(ts[-1]),
                   "duration_hours": float((ts[-1] - ts[0]) / 3.6e6)},
        "n_snapshots": int(n),
        "sampling": {"median_dt_ms": float(np.median(dt)),
                     "p99_dt_ms": float(np.percentile(dt, 99)),
                     "max_dt_ms": int(dt.max()),
                     "gaps_gt_500ms": int((dt > 500).sum()),
                     "gaps_gt_2s": int((dt > 2_000).sum())},
        "price": {"mid_min": float(mid.min()), "mid_max": float(mid.max()),
                  "mid_first": float(mid[0]), "mid_last": float(mid[-1])},
        "spread": {"median": round(float(np.median(spread)), 6),
                   "mean": round(float(spread.mean()), 6),
                   "p99": round(float(np.percentile(spread, 99)), 6),
                   "p99_9": round(float(np.percentile(spread, 99.9)), 6),
                   "max": round(float(spread.max()), 6),
                   "share_one_tick": float(np.mean(np.abs(spread - tick) < tick * 1e-3)),
                   "tick": float(tick)},
        "best_volume": {"bid": vol_stats(bv), "ask": vol_stats(av)},
        "depth10_volume": {
            "bid": {"median": float(np.median(depth_bid)), "mean": float(depth_bid.mean())},
            "ask": {"median": float(np.median(depth_ask)), "mean": float(depth_ask.mean())}},
        "obi": {"mean_abs": float(abs_obi.mean()),
                "share_abs_gt_0_6": float(np.mean(abs_obi > 0.6)),
                "share_abs_eq_1": float(np.mean(abs_obi == 1.0))},
        "split": {"fraction": float(in_sample_fraction),
                  "in_sample_rows": len(ins), "out_of_sample_rows": len(oos),
                  "in_sample_start_utc": _utc(ins.ts[0]), "in_sample_end_utc": _utc(ins.ts[-1]),
                  "out_of_sample_start_utc": _utc(oos.ts[0]),
                  "out_of_sample_end_utc": _utc(oos.ts[-1])},
    }
