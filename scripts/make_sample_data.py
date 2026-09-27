"""Generate a synthetic order book CSV in the raw schema of the real dataset.

The full real dataset (Kaggle "Bitcoin Limit Order Book (LOB) Data", BTC/USDT
perpetual, MIT license) is not included because of its size (~1.2 GB); a real
excerpt with its first 100k snapshots ships as
sample_data/btc_lob_sample_100k.csv.gz. This script instead generates a
synthetic file with the same schema, so the pipeline and tests can run end to
end without any real data.

Nothing produced from this file is a research result. The mid-price is built
to respond weakly to the previous snapshot's OBI, so a small positive
OBI -> future-mid relationship appears by construction. All numbers in the
README and report come from the real dataset.

Model (seeded): ~250 ms sampling with a few 1-120 s gaps; 0.1 USDT ticks,
spread 1 tick (2 on ~1% of rows); best-level log-volumes are smoothed noise so
OBI persists for a few snapshots; the best bid moves one tick with probability
`move_prob`, up with probability 0.5 + `obi_coupling` * previous OBI.

Usage:
    python scripts/make_sample_data.py [--rows 200000] [--seed 7] [--out data/sample/sample.csv]
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DEPTH = 10
TICK = 0.1
START_MS = 1_672_531_200_000  # 2023-01-01 00:00:00 UTC (synthetic period)


def generate(n: int, seed: int = 7, mid0: float = 17_000.0, move_prob: float = 0.2,
             obi_coupling: float = 0.25, n_gaps: int = 5) -> pd.DataFrame:
    """Synthetic snapshots in the raw CSV layout; write with `df.to_csv(path)`."""
    if n < 2:
        raise ValueError("need at least 2 rows")
    rng = np.random.default_rng(seed)

    # 250 ms +/- 3 ms, plus `n_gaps` outages of 1-120 s.
    dt = 250 + rng.integers(-3, 4, size=n - 1)
    gap_at = rng.choice(n - 1, size=min(n_gaps, n - 1), replace=False)
    dt[gap_at] += rng.integers(1_000, 120_000, size=gap_at.size)
    ts = START_MS + np.concatenate([[0], np.cumsum(dt)]).astype(np.int64)

    # Smoothed log-normal best-level volumes (half-life ~3 snapshots), so OBI
    # is autocorrelated. Median ~1 BTC.
    kernel = 0.8 ** np.arange(30)
    kernel /= np.sqrt((kernel ** 2).sum())  # unit variance after smoothing

    def best_volume() -> np.ndarray:
        noise = rng.standard_normal(n + kernel.size - 1)
        smooth = np.convolve(noise, kernel, mode="valid")
        return np.maximum(np.round(np.exp(1.2 * smooth), 3), 0.001)

    bid_v0, ask_v0 = best_volume(), best_volume()
    obi = (bid_v0 - ask_v0) / (bid_v0 + ask_v0)

    # Best bid walk in ticks; a move at row t is tilted by OBI at t-1.
    moves = rng.random(n) < move_prob
    p_up = 0.5 + obi_coupling * np.concatenate([[0.0], obi[:-1]])
    step = np.where(rng.random(n) < p_up, 1, -1) * moves
    step[0] = 0
    best_bid_ticks = int(round(mid0 / TICK)) + np.cumsum(step)
    spread_ticks = np.where(rng.random(n) < 0.01, 2, 1)

    # Deeper levels one tick apart, occasionally two.
    bid_off = np.cumsum(1 + (rng.random((n, DEPTH)) < 0.1), axis=1) - 1
    bid_off[:, 0] = 0
    ask_off = np.cumsum(1 + (rng.random((n, DEPTH)) < 0.1), axis=1) - 1
    ask_off[:, 0] = 0
    bid_px = (best_bid_ticks[:, None] - bid_off) * TICK
    ask_px = (best_bid_ticks[:, None] + spread_ticks[:, None] + ask_off) * TICK

    # Mean 1 + 0.5 * level BTC.
    scale = 1.0 + 0.5 * np.arange(DEPTH)
    bid_vol = np.maximum(np.round(rng.exponential(scale, size=(n, DEPTH)), 3), 0.001)
    ask_vol = np.maximum(np.round(rng.exponential(scale, size=(n, DEPTH)), 3), 0.001)
    bid_vol[:, 0], ask_vol[:, 0] = bid_v0, ask_v0

    book = np.empty((n, 4 * DEPTH))
    book[:, 0:2 * DEPTH:2] = np.round(bid_px, 1)
    book[:, 1:2 * DEPTH:2] = bid_vol
    book[:, 2 * DEPTH::2] = np.round(ask_px, 1)
    book[:, 2 * DEPTH + 1::2] = ask_vol

    stamp = pd.to_datetime(ts, unit="ms", utc=True).strftime("%Y-%m-%d %H:%M:%S")
    df = pd.DataFrame(book, columns=range(2, 2 + 4 * DEPTH))
    df.insert(0, 0, ts)
    df.insert(1, 1, stamp)
    return df


def write_csv(path: Path | str, n: int, seed: int = 7, **kwargs) -> Path:
    """Generate `n` synthetic rows and write them in the raw CSV format."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    generate(n, seed, **kwargs).to_csv(path)
    return path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--rows", type=int, default=200_000)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--out", type=Path, default=ROOT / "data" / "sample" / "sample.csv")
    args = parser.parse_args()
    path = write_csv(args.out, args.rows, args.seed)
    print(f"wrote {args.rows:,} SYNTHETIC rows to {path} ({path.stat().st_size / 1e6:.1f} MB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
