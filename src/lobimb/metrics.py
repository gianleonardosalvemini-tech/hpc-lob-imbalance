"""Trade-level performance statistics.

Inputs are trade lists from lobimb.backtest (PnL in USDT per 1 BTC).
"""
from __future__ import annotations

import math

import numpy as np
import pandas as pd

MS_PER_DAY = 86_400_000


def max_drawdown(pnl: np.ndarray) -> float:
    """Largest peak-to-trough fall of cumulative PnL, as a positive number.

    Equity starts at 0, so an initial loss counts as drawdown.
    """
    if pnl.size == 0:
        return 0.0
    equity = np.concatenate([[0.0], np.cumsum(pnl)])
    return float(np.max(np.maximum.accumulate(equity) - equity))


def summarize(trades: pd.DataFrame, include_gaps: bool = False) -> dict[str, float]:
    """Summary KPIs for a trade list from `lobimb.backtest`.

    EV figures are per trade, per 1 BTC, in USDT. t_stat assumes independent
    trades; ci95 is mean +/- 1.96 SE. mid_move is the mean signed mid change
    from signal to exit (the signal's frictionless predictive power).
    sharpe_annual uses daily PnL x sqrt(365) and is only indicative on a few
    days of data. With no trades, only `trades` and `excluded_gap_trades` are
    returned.
    """
    t = trades if include_gaps else trades[~trades["gap"]]
    net = t["net_pnl"].to_numpy(dtype=float)
    n = net.size
    out: dict[str, float] = {"trades": n, "excluded_gap_trades": int(trades["gap"].sum())}
    if n == 0:
        return out

    std = float(net.std(ddof=1)) if n > 1 else float("nan")
    se = std / math.sqrt(n) if n > 1 else float("nan")
    mean = float(net.mean())
    wins, losses = net[net > 0].sum(), -net[net < 0].sum()

    # Daily PnL by UTC day of the exit.
    daily = pd.Series(net).groupby(t["exit_ts"].to_numpy() // MS_PER_DAY).sum()
    daily_sharpe = (daily.mean() / daily.std(ddof=1) * math.sqrt(365)
                    if len(daily) > 1 and daily.std(ddof=1) > 0 else float("nan"))

    out.update({
        "win_rate": float((net > 0).mean()),
        "ev_gross": float(t["gross_pnl"].mean()),
        "ev_net": mean,
        "ev_net_bps": float((net / t["entry_px"].to_numpy()).mean() * 1e4),
        "mid_move": float(t["mid_move"].mean()),
        "total_net": float(net.sum()),
        # NaN > 0 is False, so a NaN se also gives NaN here.
        "t_stat": mean / se if se and se > 0 else float("nan"),
        "ci95_low": mean - 1.96 * se,
        "ci95_high": mean + 1.96 * se,
        "profit_factor": float(wins / losses) if losses > 0 else float("inf"),
        "max_drawdown": max_drawdown(net),
        "sharpe_annual": float(daily_sharpe),
        "avg_hold_s": float((t["exit_ts"] - t["entry_ts"]).mean() / 1000),
        "size_capped_frac": float(t["size_capped"].mean()),
    })
    return out
