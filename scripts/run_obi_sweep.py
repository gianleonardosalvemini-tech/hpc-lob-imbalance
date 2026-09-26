"""Level-1 OBI taker strategy: before/after comparison and parameter sweep.

Replaces legacy/src_python/OBI/backtest_strategy.py and the "EV" estimate in
`python historical_ingestion.py`. Outputs go to results/.

  A. Signal: correct OBI vs the legacy int-truncated OBI.
  B. Legacy methodology (first 100k rows, zero latency, no fees) re-run, to
     reproduce the original numbers.
  C. Corrected methodology on the full dataset: 250 ms latency, taker fees,
     wall-clock horizons, gap filtering, in-sample / out-of-sample split.

Defaults come from ObiConfig/CostModel; the sweep adds a lower and higher
threshold and a 60 s horizon.

Usage:
    python scripts/run_obi_sweep.py [--thresholds 0.4 0.6 0.8] [--fees 0 2 4 5]
                                    [--cache data/cache] [--out results]
                                    [--figures docs/figures]

--cache selects another .npy cache (e.g. the synthetic sample); --out and
--figures redirect outputs so a sample run doesn't overwrite real results.
The EV-decay chart is copied to each --figures directory (`--figures` with no
value skips the copy).
"""
from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import numpy as np
import pandas as pd

from lobimb import backtest, data, metrics, plots, reference, signals
from lobimb.config import (CACHE_DIR, PROJECT_ROOT, RESULTS_DIR, BacktestConfig, CostModel,
                           ObiConfig)

# metrics.summarize keys copied into each result row.
COLS = ["trades", "win_rate", "ev_gross", "ev_net", "mid_move", "t_stat", "total_net",
        "max_drawdown", "sharpe_annual", "size_capped_frac"]


def run(book: data.Book, obi: np.ndarray, threshold: float, horizon_ms: int,
        costs: CostModel, fees_bps: list[float], **tags) -> list[dict]:
    """Backtest once without fees, then re-price for each fee level.

    Fees don't change which trades are taken, so this matches one backtest per
    fee level. `costs.fee_bps` is ignored; `tags` are extra output columns.
    """
    cfg = BacktestConfig(costs=CostModel(fee_bps=0, slippage=costs.slippage,
                                         latency_ms=costs.latency_ms))
    trades = backtest.fixed_horizon(book, backtest.threshold_side(obi, threshold), horizon_ms, cfg)
    rows = []
    for fee in fees_bps:
        s = metrics.summarize(backtest.with_fees(trades, fee))
        rows.append({**tags, "threshold": threshold, "horizon_s": horizon_ms / 1000,
                     "fee_bps": fee, "slippage": costs.slippage, "latency_ms": costs.latency_ms,
                     **{c: s.get(c, np.nan) for c in COLS}})
    return rows


def main() -> int:
    o = ObiConfig()
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--thresholds", type=float, nargs="+",
                        default=sorted({0.4, o.threshold, 0.8}))
    parser.add_argument("--horizons-ms", type=int, nargs="+", default=[*o.horizons_ms, 60_000])
    parser.add_argument("--fees", type=float, nargs="+", default=[0.0, 2.0, 4.0])
    parser.add_argument("--latency-ms", type=int, default=CostModel().latency_ms)
    parser.add_argument("--cache", type=Path, default=CACHE_DIR,
                        help="directory with the ts.npy/levels.npy cache")
    parser.add_argument("--out", type=Path, default=RESULTS_DIR, help="output directory")
    parser.add_argument("--figures", type=Path, nargs="*", default=[PROJECT_ROOT / "docs" / "figures"],
                        help="directories that receive a copy of obi_ev_decay.png")
    args = parser.parse_args()
    out = args.out
    out.mkdir(parents=True, exist_ok=True)
    book = data.load(args.cache)

    # --- A. Signal distribution -------------------------------------------------
    obi = signals.obi(book)
    obi_int = reference.legacy_obi_int(np.asarray(book.levels))
    sig = pd.DataFrame({
        "metric": ["|OBI| == 1", "|OBI| > 0.6", "OBI == 0", "mean |OBI|"],
        "correct (float volumes)": [np.mean(np.abs(obi) == 1), np.mean(np.abs(obi) > 0.6),
                                    np.mean(obi == 0), np.mean(np.abs(obi))],
        "legacy (int volumes)": [np.mean(np.abs(obi_int) == 1), np.mean(np.abs(obi_int) > 0.6),
                                 np.mean(obi_int == 0), np.mean(np.abs(obi_int))],
    })
    print("\nA. Signal distribution\n", sig.to_string(index=False, float_format="%.4f"))
    plots.histogram({"correct": obi, "legacy int": obi_int}, out / "obi_distribution.png",
                    title="Level-1 OBI distribution: float vs int-truncated volumes")
    plots.price_vs_imbalance(book[:5000], obi[:5000], out / "price_vs_obi.png",
                             title="Mid price vs level-1 OBI (first 5,000 snapshots)")

    # --- B. Legacy methodology --------------------------------------------------
    # Original study: first 100k rows, zero latency, no fees.
    head = book[:100_000]
    legacy_rows = []
    for label, sig_arr in (("float OBI", obi[:100_000]), ("int OBI", obi_int[:100_000])):
        for h in o.horizons_ms:
            for slip in (0.0, 0.5, 1.0, 2.0):
                # Legacy slippage was per round trip; ours is per side.
                legacy_rows += run(head, sig_arr, o.threshold, h, CostModel(0, slip / 2, 0), [0.0],
                                   signal=label, sample="first 100k, legacy method")
    legacy = pd.DataFrame(legacy_rows)
    print("\nB. Legacy methodology\n",
          legacy[["signal", "horizon_s", "slippage", "trades", "win_rate", "ev_net"]]
          .to_string(index=False, float_format="%.3f"))

    # --- C. Corrected methodology -----------------------------------------------
    # OBI uses only the current snapshot, so computing it on the whole book and
    # slicing leaks nothing across the split.
    cfg_costs = CostModel(fee_bps=0, slippage=0.0, latency_ms=args.latency_ms)
    ins, oos = data.in_sample_split(book, BacktestConfig().in_sample_fraction)
    n_in = len(ins)
    rows = []
    for split, part, part_obi in (("in-sample", ins, obi[:n_in]), ("out-of-sample", oos, obi[n_in:])):
        for thr in args.thresholds:
            for h in args.horizons_ms:
                rows += run(part, part_obi, thr, h, cfg_costs, args.fees, split=split)
    sweep = pd.DataFrame(rows)
    print("\nC. Corrected sweep\n",
          sweep[["split", "threshold", "horizon_s", "fee_bps", "trades", "win_rate", "mid_move",
                 "ev_gross", "ev_net", "t_stat"]].to_string(index=False, float_format="%.3f"))

    sig.to_csv(out / "obi_signal_distribution.csv", index=False)
    legacy.to_csv(out / "obi_legacy_method.csv", index=False)
    sweep.to_csv(out / "obi_sweep.csv", index=False)
    oos_head = sweep[(sweep.split == "out-of-sample") & (sweep.threshold == o.threshold)]
    chart = plots.ev_decay(oos_head, out / "obi_ev_decay.png",
                           title=f"OBI > {o.threshold}, out-of-sample: net EV per trade vs horizon")
    for d in args.figures:
        d.mkdir(parents=True, exist_ok=True)
        if (d / chart.name).resolve() != chart.resolve():
            shutil.copyfile(chart, d / chart.name)
    print(f"\nresults written to {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
