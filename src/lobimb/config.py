"""Paths and strategy/cost parameters.

Parameter sets are frozen dataclasses; make variants with keyword arguments,
e.g. ``CostModel(fee_bps=0, latency_ms=0)``.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = Path(os.environ.get("LOBIMB_DATA_DIR", PROJECT_ROOT / "data"))
# Raw BTC/USDT snapshots (Jan 9 - Jan 20), ~1.2 GB of CSV.
RAW_CSV = DATA_DIR / "raw" / "1-09-1-20.csv"
# Parsed .npy arrays written by scripts/prepare_data.py.
CACHE_DIR = DATA_DIR / "cache"
RESULTS_DIR = PROJECT_ROOT / "results"

# Nominal sampling interval (4 snapshots/s); real gaps can be longer.
SNAPSHOT_MS = 250


@dataclass(frozen=True)
class CostModel:
    """Taker execution costs, per 1 BTC traded.

    fee_bps: fee per side in bps of notional (paid twice per round trip).
    slippage: extra adverse price per side, in USDT.
    latency_ms: delay between the decision snapshot and the fill snapshot.
        Default is one snapshot, the smallest realistic delay on this data.
    """

    fee_bps: float = 4.0
    slippage: float = 0.0
    latency_ms: int = SNAPSHOT_MS

    @property
    def fee_rate(self) -> float:
        """Fee as a fraction of notional (4 bps -> 0.0004)."""
        return self.fee_bps * 1e-4


@dataclass(frozen=True)
class ObiConfig:
    """Level-1 OBI strategy: long if OBI > threshold, short if OBI < -threshold."""

    threshold: float = 0.6
    horizons_ms: tuple[int, ...] = (1_000, 5_000, 20_000)


@dataclass(frozen=True)
class WobiConfig:
    """Depth-weighted imbalance strategy with a take-profit/stop-loss bracket.

    Long when micro - mid > edge_threshold and imbalance > imbalance_threshold
    (mirrored for short). take_profit/stop_loss are USDT distances from the
    fill price; max_hold_ms is measured from the fill.
    """

    alpha: float = 0.5
    depth: int = 10
    edge_threshold: float = 0.2
    imbalance_threshold: float = 0.6
    take_profit: float = 1.0
    stop_loss: float = 1.0
    max_hold_ms: int = 2_500
    # Also require the edge to cover spread + two fees (what the original
    # script attempted).
    require_edge_covers_cost: bool = False


@dataclass(frozen=True)
class BacktestConfig:
    """Settings shared by every backtest in lobimb.backtest."""

    costs: CostModel = field(default_factory=CostModel)
    # Trades spanning a longer data gap are flagged and excluded by
    # metrics.summarize: prices across a gap were not tradeable.
    max_gap_ms: int = 2_000
    in_sample_fraction: float = 0.7
    # Order size in BTC; only used to flag fills larger than the top level.
    quantity: float = 0.01
