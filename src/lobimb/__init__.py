"""lobimb - order book imbalance research toolkit.

Modules:
    config     paths and strategy/cost parameters (frozen dataclasses)
    data       load/validate snapshots (CSV -> cached .npy)
    engine     ctypes binding to the C/OpenMP batch kernels
    signals    OBI, depth-weighted imbalance, micro-price, mid, spread
    backtest   taker backtests with latency, fees and gap handling
    metrics    trade statistics (EV, win rate, Sharpe, drawdown, t-stat)
    reference  pure-NumPy versions of the kernels (test oracle, benchmark baseline)
    perf       latency/throughput measurements used by benchmarks and bench tests
    plots      matplotlib figures written to files

Typical flow: data.load() -> signals.obi()/wobi() -> backtest.*_side() ->
backtest.fixed_horizon()/bracket() -> metrics.summarize().
"""

__version__ = "1.0.0"
