"""Taker backtests on order book snapshots.

Execution model:
  * A signal at snapshot t fills at the first snapshot at or after
    t + latency_ms, crossing the spread (buy at ask, sell at bid) plus slippage.
    Filling at t's own prices or at the mid would show profits that can't be
    realised.
  * Fees on both legs, as a fraction of notional. PnL is per 1 BTC.
  * One position at a time; a new one may open on the snapshot where the
    previous one exits.
  * Horizons are in ms, not snapshots. Trades spanning a gap > max_gap_ms are
    kept but flagged `gap` (metrics.summarize drops them by default).
  * `size_capped` flags fills larger than the best-level volume; queue position
    and impact are otherwise ignored.

`side` has one entry per snapshot: +1 long, -1 short, 0 no signal. The loops
run over trades, not snapshots (next signal found by binary search).
"""
from __future__ import annotations

import bisect
from dataclasses import dataclass

import numpy as np
import pandas as pd

from .config import BacktestConfig
from .data import Book

# *_idx are row indices into the book; *_px include slippage; mid_move is the
# signed mid change from signal to exit; exit_reason is time/tp/sl/eod.
TRADE_COLUMNS = [
    "signal_idx", "entry_idx", "exit_idx", "side", "entry_ts", "exit_ts",
    "entry_px", "exit_px", "gross_pnl", "fees", "net_pnl", "mid_move",
    "exit_reason", "gap", "size_capped",
]


@dataclass(frozen=True)
class _Market:
    """Top-of-book columns as in-RAM arrays (materialises memory maps)."""

    ts: np.ndarray
    bid: np.ndarray
    ask: np.ndarray
    bid_vol: np.ndarray
    ask_vol: np.ndarray
    gap_count: np.ndarray  # gap_count[k] = number of gaps between rows 0..k

    @classmethod
    def from_book(cls, book: Book, max_gap_ms: int) -> "_Market":
        # Prefix sum of gaps: rows a < b straddle a gap iff gap_count differs.
        ts = np.asarray(book.ts, dtype=np.int64)
        gaps = np.concatenate([[0], np.cumsum(np.diff(ts) > max_gap_ms)])
        return cls(ts, np.asarray(book.bid_px()), np.asarray(book.ask_px()),
                   np.asarray(book.bid_vol()), np.asarray(book.ask_vol()), gaps)

    def index_after(self, idx: np.ndarray, delay_ms: int) -> np.ndarray:
        """First snapshot at or after ts[idx] + delay_ms (len(ts) if none)."""
        return np.searchsorted(self.ts, self.ts[idx] + delay_ms, side="left")


def _prepare(book: Book, side: np.ndarray, cfg: BacktestConfig):
    """Return (market, signal, entry), dropping signals that would fill past the end."""
    if len(side) != len(book):
        raise ValueError("side must have one entry per snapshot")
    m = _Market.from_book(book, cfg.max_gap_ms)
    signal = np.flatnonzero(side)
    entry = m.index_after(signal, cfg.costs.latency_ms)
    keep = entry < len(m.ts)
    return m, signal[keep], entry[keep]


def _select(signal: list[int], exit_idx: list[int]) -> list[int]:
    """Greedy non-overlapping selection: next signal at or after the previous exit."""
    chosen, k, n = [], 0, len(signal)
    while k < n:
        chosen.append(k)
        k = bisect.bisect_left(signal, exit_idx[k], lo=k + 1)
    return chosen


def _trades(m: _Market, side: np.ndarray, signal: np.ndarray, entry: np.ndarray,
            exit_idx: np.ndarray, reason: np.ndarray, cfg: BacktestConfig) -> pd.DataFrame:
    """Price a set of trades. Both fills are on the unfavourable side of the spread."""
    c = cfg.costs
    s = np.sign(side[signal]).astype(np.int64)
    long = s > 0
    entry_px = np.where(long, m.ask[entry] + c.slippage, m.bid[entry] - c.slippage)
    exit_px = np.where(long, m.bid[exit_idx] - c.slippage, m.ask[exit_idx] + c.slippage)
    gross = s * (exit_px - entry_px)
    fees = c.fee_rate * (entry_px + exit_px)
    mid = 0.5 * (m.bid + m.ask)
    best_vol = np.where(long, m.ask_vol[entry], m.bid_vol[entry])
    return pd.DataFrame({
        "signal_idx": signal, "entry_idx": entry, "exit_idx": exit_idx, "side": s,
        "entry_ts": m.ts[entry], "exit_ts": m.ts[exit_idx],
        "entry_px": entry_px, "exit_px": exit_px, "gross_pnl": gross, "fees": fees,
        "net_pnl": gross - fees, "mid_move": s * (mid[exit_idx] - mid[signal]),
        "exit_reason": reason, "gap": m.gap_count[exit_idx] != m.gap_count[signal],
        "size_capped": cfg.quantity > best_vol,
    }, columns=TRADE_COLUMNS)


def fixed_horizon(book: Book, side: np.ndarray, horizon_ms: int,
                  cfg: BacktestConfig = BacktestConfig()) -> pd.DataFrame:
    """Hold each position for `horizon_ms` after the fill, then exit at market.

    Trades whose exit would fall after the end of the data are dropped.
    """
    m, signal, entry = _prepare(book, side, cfg)
    exit_idx = m.index_after(entry, horizon_ms)
    keep = exit_idx < len(m.ts)
    signal, entry, exit_idx = signal[keep], entry[keep], exit_idx[keep]
    sel = _select(signal.tolist(), exit_idx.tolist())
    return _trades(m, side, signal[sel], entry[sel], exit_idx[sel],
                   np.full(len(sel), "time", dtype=object), cfg)


def bracket(book: Book, side: np.ndarray, take_profit: float, stop_loss: float,
            max_hold_ms: int, cfg: BacktestConfig = BacktestConfig()) -> pd.DataFrame:
    """Take-profit / stop-loss bracket with a time stop.

    Exits at the first snapshot after the fill where the exit price (bid for a
    long, ask for a short, including slippage, i.e. the price that gets booked)
    is take_profit above or stop_loss below the fill price; otherwise after
    max_hold_ms. Positions still open at the end of the data close on the last
    snapshot ("eod").
    """
    m, signal, entry = _prepare(book, side, cfg)
    n = len(m.ts)
    deadline = m.index_after(entry, max_hold_ms)
    end = np.minimum(deadline, n - 1)
    # A fill on the last snapshot leaves nothing to exit on.
    keep = end > entry
    signal, entry, deadline, end = signal[keep], entry[keep], deadline[keep], end[keep]
    s = np.sign(side[signal])
    slip = cfg.costs.slippage
    entry_px = np.where(s > 0, m.ask[entry] + slip, m.bid[entry] - slip)

    # The exit depends on the price path, so resolve it only for selected trades.
    signal_list = signal.tolist()
    chosen, exits, reasons = [], [], []
    k = 0
    while k < len(signal_list):
        e, stop = int(entry[k]), int(end[k])
        path = (m.bid[e + 1:stop + 1] - slip if s[k] > 0
                else m.ask[e + 1:stop + 1] + slip)
        move = s[k] * (path - entry_px[k])
        hit = np.flatnonzero((move >= take_profit) | (move <= -stop_loss))
        if hit.size:
            x, reason = e + 1 + int(hit[0]), "tp" if move[hit[0]] >= take_profit else "sl"
        else:
            x, reason = stop, "time" if deadline[k] < n else "eod"
        chosen.append(k)
        exits.append(x)
        reasons.append(reason)
        k = bisect.bisect_left(signal_list, x, lo=k + 1)

    return _trades(m, side, signal[chosen], entry[chosen], np.asarray(exits, dtype=np.int64),
                   np.asarray(reasons, dtype=object), cfg)


def with_fees(trades: pd.DataFrame, fee_bps: float) -> pd.DataFrame:
    """Re-price a trade list at another fee level (per side, in bps).

    Selection doesn't depend on fees, so a fee sweep can backtest once and
    re-price here.
    """
    out = trades.copy()
    out["fees"] = fee_bps * 1e-4 * (out["entry_px"] + out["exit_px"])
    out["net_pnl"] = out["gross_pnl"] - out["fees"]
    return out


def wobi_side(edge: np.ndarray, imbalance: np.ndarray, edge_threshold: float,
              imbalance_threshold: float, min_edge: np.ndarray | None = None) -> np.ndarray:
    """+1 when edge and imbalance both exceed their thresholds, -1 for the mirror case.

    `min_edge` (per snapshot) optionally raises the edge threshold, e.g. to the
    round-trip cost.
    """
    need = edge_threshold if min_edge is None else np.maximum(edge_threshold, min_edge)
    side = np.zeros(edge.shape[0], dtype=np.int8)
    side[(edge > need) & (imbalance > imbalance_threshold)] = 1
    side[(edge < -need) & (imbalance < -imbalance_threshold)] = -1
    return side


def threshold_side(signal: np.ndarray, threshold: float) -> np.ndarray:
    """+1 where signal > threshold, -1 where signal < -threshold, else 0 (NaN -> 0)."""
    side = np.zeros(signal.shape[0], dtype=np.int8)
    side[signal > threshold] = 1
    side[signal < -threshold] = -1
    return side
