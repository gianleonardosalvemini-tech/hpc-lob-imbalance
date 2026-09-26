"""Depth-weighted imbalance (WOBI) + micro-price strategy, corrected.

Replaces legacy/src_python/WOBI/quant_strategy.py. Differences:
  * the edge filter compares USDT with USDT (the old one added `mid * fee`, ~7 USDT,
    to a signal bounded by ~1 USDT, so it could never fire);
  * take-profit / stop-loss are distances from the fill price, fees are charged
    on both legs, fills happen after a latency;
  * trades across data gaps are excluded, the position open at the end is closed;
  * results are reported in-sample and out-of-sample.

With --require-edge-covers-cost (WobiConfig.require_edge_covers_cost) an
entry also needs |micro - mid| > spread + 2 x fee x mid, the round-trip cost of
a taker at --cost-fee-bps; results then go to wobi_sweep_edge_covers_cost.csv.

Usage:
    python scripts/run_wobi_strategy.py [--alpha 0.5] [--fees 0 4]
                                        [--require-edge-covers-cost] [--cost-fee-bps 4]
                                        [--cache data/cache] [--out results]
"""
from __future__ import annotations

import argparse
import itertools
from pathlib import Path

import numpy as np
import pandas as pd

from lobimb import backtest, data, metrics, signals
from lobimb.config import CACHE_DIR, RESULTS_DIR, BacktestConfig, CostModel, WobiConfig

# metrics.summarize keys copied into each result row.
COLS = ["trades", "win_rate", "ev_gross", "ev_net", "mid_move", "t_stat", "total_net",
        "max_drawdown", "avg_hold_s"]


def main() -> int:
    d = WobiConfig()
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--alpha", type=float, default=d.alpha)
    parser.add_argument("--depth", type=int, default=d.depth)
    parser.add_argument("--edge", type=float, nargs="+", default=[0.05, 0.1, 0.2])
    parser.add_argument("--imbalance", type=float, default=d.imbalance_threshold)
    parser.add_argument("--brackets", type=float, nargs="+", default=[0.5, 1.0, 2.0],
                        help="symmetric take-profit/stop-loss distances in USDT")
    parser.add_argument("--max-hold-ms", type=int, default=d.max_hold_ms)
    parser.add_argument("--fees", type=float, nargs="+", default=[0.0, 4.0])
    parser.add_argument("--latency-ms", type=int, default=CostModel().latency_ms)
    parser.add_argument("--require-edge-covers-cost", action="store_true",
                        default=d.require_edge_covers_cost,
                        help="also require |micro - mid| > spread + 2 x fee x mid")
    parser.add_argument("--cost-fee-bps", type=float, default=CostModel().fee_bps,
                        help="taker fee per side used by the round-trip cost (default: %(default)s)")
    parser.add_argument("--cache", type=Path, default=CACHE_DIR,
                        help="directory with the ts.npy/levels.npy cache")
    parser.add_argument("--out", type=Path, default=RESULTS_DIR, help="output directory")
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    book = data.load(args.cache)
    w = signals.wobi(book, depth=args.depth, alpha=args.alpha)
    spread = signals.spread(book)
    mid = signals.mid(book)

    # Compare with the spread (>= 0.1 USDT) and the round-trip fee (~13.6 USDT).
    abs_edge = np.abs(w.edge)
    q = np.percentile(abs_edge, [50, 90, 99, 99.9, 100])
    print("|micro - mid| percentiles (USDT): "
          + ", ".join(f"p{p}={v:.3f}" for p, v in zip([50, 90, 99, 99.9, 100], q)))

    # Legacy entry filter (mixed units), for the record.
    legacy_cost = 0.2 + spread + mid * 0.0004
    legacy_long = (w.edge > legacy_cost) & (w.imbalance > 0.6)
    legacy_short = (-w.edge > legacy_cost) & (w.imbalance < -0.6)
    print(f"legacy entry condition true on {int(legacy_long.sum() + legacy_short.sum())} "
          f"of {len(book):,} snapshots")
    # Consistent version, all in USDT: does the edge cover spread + two fees?
    round_trip = spread + 2 * args.cost_fee_bps * 1e-4 * mid
    print(f"edge > spread + 2 x {args.cost_fee_bps:g} bps fee on "
          f"{int((abs_edge > round_trip).sum())} snapshots")
    min_edge = round_trip if args.require_edge_covers_cost else None

    ins, oos = data.in_sample_split(book, BacktestConfig().in_sample_fraction)
    n_in = len(ins)
    # Backtest fee-free, re-price per fee level below.
    cfg = BacktestConfig(costs=CostModel(fee_bps=0, latency_ms=args.latency_ms))
    rows = []
    for split, part, sl in (("in-sample", ins, slice(0, n_in)), ("out-of-sample", oos, slice(n_in, None))):
        for edge_thr, bracket_usdt in itertools.product(args.edge, args.brackets):
            side = backtest.wobi_side(w.edge[sl], w.imbalance[sl], edge_thr, args.imbalance,
                                      None if min_edge is None else min_edge[sl])
            trades = backtest.bracket(part, side, bracket_usdt, bracket_usdt, args.max_hold_ms, cfg)
            reasons = trades["exit_reason"].value_counts(normalize=True).to_dict()
            for fee in args.fees:
                s = metrics.summarize(backtest.with_fees(trades, fee))
                rows.append({"split": split, "edge": edge_thr, "tp_sl": bracket_usdt, "fee_bps": fee,
                             **{c: s.get(c, np.nan) for c in COLS},
                             **{f"exit_{r}": reasons.get(r, 0.0) for r in ("tp", "sl", "time", "eod")}})
    res = pd.DataFrame(rows)
    print(res[["split", "edge", "tp_sl", "fee_bps", "trades", "win_rate", "mid_move", "ev_gross",
               "ev_net", "t_stat", "exit_tp", "exit_sl"]].to_string(index=False, float_format="%.3f"))
    name = "wobi_sweep_edge_covers_cost.csv" if args.require_edge_covers_cost else "wobi_sweep.csv"
    res.to_csv(args.out / name, index=False)
    print(f"\nresults written to {args.out / name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
