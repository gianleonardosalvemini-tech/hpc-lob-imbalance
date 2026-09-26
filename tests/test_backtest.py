"""Execution model (lobimb.backtest) and statistics (lobimb.metrics).

Most tests use `flat_book` with 250 ms snapshots so indices and PnL can be
checked by hand.
"""
import numpy as np
import pytest

from conftest import flat_book, make_book
from lobimb import backtest, metrics
from lobimb.config import BacktestConfig, CostModel

NO_COST = BacktestConfig(costs=CostModel(fee_bps=0, slippage=0, latency_ms=0))


def side_at(n, idx, s=1):
    side = np.zeros(n, dtype=np.int8)
    side[idx] = s
    return side


def test_long_crosses_spread_both_ways():
    # buy at ask 100.1, sell at bid 101.0 two rows later; mid moved 1.0
    book = flat_book([100.0, 100.0, 101.0, 101.0], [100.1, 100.1, 101.1, 101.1])
    trades = backtest.fixed_horizon(book, side_at(4, 0), horizon_ms=500, cfg=NO_COST)
    assert len(trades) == 1
    t = trades.iloc[0]
    assert (t.entry_idx, t.exit_idx) == (0, 2)
    assert t.entry_px == pytest.approx(100.1)
    assert t.exit_px == pytest.approx(101.0)
    assert t.net_pnl == pytest.approx(0.9)
    assert t.mid_move == pytest.approx(1.0)


def test_short_pnl_sign():
    # sell at bid 100.0, buy back at ask 99.1
    book = flat_book([100.0, 99.0, 99.0], [100.1, 99.1, 99.1])
    trades = backtest.fixed_horizon(book, side_at(3, 0, -1), horizon_ms=250, cfg=NO_COST)
    assert trades.iloc[0].net_pnl == pytest.approx(100.0 - 99.1)


def test_latency_delays_fill():
    book = flat_book([100.0, 102.0, 102.0, 102.0], [100.1, 102.1, 102.1, 102.1])
    cfg = BacktestConfig(costs=CostModel(fee_bps=0, latency_ms=250))
    t = backtest.fixed_horizon(book, side_at(4, 0), horizon_ms=250, cfg=cfg).iloc[0]
    assert t.entry_idx == 1
    assert t.entry_px == pytest.approx(102.1)  # the move was missed


def test_fees_and_slippage():
    # locked book at 100: buy 100.05, sell 99.95 -> gross -0.1; fees 1e-3 per leg
    book = flat_book([100.0] * 3, [100.0] * 3)
    cfg = BacktestConfig(costs=CostModel(fee_bps=10, slippage=0.05, latency_ms=0))
    t = backtest.fixed_horizon(book, side_at(3, 0), horizon_ms=250, cfg=cfg).iloc[0]
    assert t.gross_pnl == pytest.approx(-0.1)
    assert t.fees == pytest.approx(1e-3 * (100.05 + 99.95))
    assert t.net_pnl == pytest.approx(t.gross_pnl - t.fees)


def test_no_overlapping_positions():
    n = 20
    book = flat_book([100.0] * n, [100.1] * n)
    trades = backtest.fixed_horizon(book, np.ones(n, dtype=np.int8), horizon_ms=1000, cfg=NO_COST)
    # 4-snapshot holds, each trade starts on the previous exit; the one from
    # row 16 would exit past the end.
    assert trades["signal_idx"].tolist() == [0, 4, 8, 12]
    assert np.all(trades["signal_idx"].to_numpy()[1:] >= trades["exit_idx"].to_numpy()[:-1])


def test_trades_past_end_of_data_are_dropped():
    book = flat_book([100.0] * 5, [100.1] * 5)
    trades = backtest.fixed_horizon(book, side_at(5, 3), horizon_ms=1000, cfg=NO_COST)
    assert trades.empty


def test_gap_trades_flagged_and_excluded():
    book = flat_book([100.0] * 6, [100.1] * 6)
    ts = book.ts.copy()
    ts[3:] += 60_000  # 60 s gap between rows 2 and 3
    book = type(book)(ts, book.levels)
    trades = backtest.fixed_horizon(book, side_at(6, 1), horizon_ms=500, cfg=NO_COST)
    assert trades.iloc[0].gap
    assert metrics.summarize(trades)["trades"] == 0


def test_bracket_take_profit_stop_loss_time():
    bids = [100.0, 100.0, 100.5, 101.2, 101.2, 101.2]
    book = flat_book(bids, [b + 0.1 for b in bids])
    tp = backtest.bracket(book, side_at(6, 0), take_profit=1.0, stop_loss=1.0,
                          max_hold_ms=5000, cfg=NO_COST).iloc[0]
    assert (tp.exit_idx, tp.exit_reason) == (3, "tp")  # 101.2 - 100.1 >= 1.0

    bids = [100.0, 100.0, 99.0, 99.0]
    book = flat_book(bids, [b + 0.1 for b in bids])
    sl = backtest.bracket(book, side_at(4, 0), 1.0, 1.0, 5000, NO_COST).iloc[0]
    assert (sl.exit_idx, sl.exit_reason) == (2, "sl")  # 99.0 - 100.1 <= -1.0

    book = flat_book([100.0] * 10, [100.1] * 10)
    tm = backtest.bracket(book, side_at(10, 0), 1.0, 1.0, 1000, NO_COST).iloc[0]
    assert (tm.exit_idx, tm.exit_reason) == (4, "time")

    eod = backtest.bracket(book, side_at(10, 7), 1.0, 1.0, 5000, NO_COST).iloc[0]
    assert (eod.exit_idx, eod.exit_reason) == (9, "eod")


def test_bracket_levels_include_exit_slippage():
    """TP/SL trigger on the exit price after slippage, i.e. the booked price."""
    cfg = BacktestConfig(costs=CostModel(fee_bps=0, slippage=0.1, latency_ms=0))
    # entry 100.2; row 2 sells at 101.15 (+0.95, no tp), row 3 at 101.25 (+1.05)
    bids = [100.0, 100.0, 101.25, 101.35, 101.35]
    book = flat_book(bids, [b + 0.1 for b in bids])
    tp = backtest.bracket(book, side_at(5, 0), 1.0, 1.0, 5000, cfg).iloc[0]
    assert (tp.exit_idx, tp.exit_reason) == (3, "tp")
    assert tp.gross_pnl == pytest.approx(1.05)

    # raw bid move is -0.95, but it sells at 99.15 -> -1.05
    bids = [100.0, 100.0, 99.25, 99.25]
    book = flat_book(bids, [b + 0.1 for b in bids])
    sl = backtest.bracket(book, side_at(4, 0), 1.0, 1.0, 5000, cfg).iloc[0]
    assert (sl.exit_idx, sl.exit_reason) == (2, "sl")
    assert sl.gross_pnl == pytest.approx(-1.05)

    book = make_book(20_000, seed=4)
    side = backtest.threshold_side(np.random.default_rng(3).uniform(-1, 1, 20_000), 0.9)
    trades = backtest.bracket(book, side, 0.5, 0.5, 10_000, cfg)
    tp, sl = trades[trades.exit_reason == "tp"], trades[trades.exit_reason == "sl"]
    assert len(tp) > 0 and len(sl) > 0 and set(trades.side) == {-1, 1}
    assert (tp.gross_pnl >= 0.5 - 1e-9).all()
    assert (sl.gross_pnl <= -0.5 + 1e-9).all()


def test_size_capped_flag():
    book = flat_book([100.0] * 3, [100.1] * 3, volume=0.001)
    cfg = BacktestConfig(costs=NO_COST.costs, quantity=0.01)
    t = backtest.fixed_horizon(book, side_at(3, 0), 250, cfg).iloc[0]
    assert t.size_capped


def test_with_fees_reprices_without_rerun():
    book = make_book(3000, seed=2)
    side = backtest.threshold_side(np.random.default_rng(1).uniform(-1, 1, 3000), 0.8)
    cfg = BacktestConfig(costs=CostModel(fee_bps=4))
    direct = backtest.fixed_horizon(book, side, 1000, cfg)
    base = backtest.fixed_horizon(book, side, 1000, BacktestConfig(costs=CostModel(fee_bps=0)))
    np.testing.assert_allclose(backtest.with_fees(base, 4)["net_pnl"], direct["net_pnl"])


def test_wobi_side():
    edge = np.array([0.3, 0.3, -0.3, 0.1, 0.3])
    imb = np.array([0.7, 0.5, -0.7, 0.9, 0.7])
    assert backtest.wobi_side(edge, imb, 0.2, 0.6).tolist() == [1, 0, -1, 0, 1]
    # min_edge 0.5 on the last row raises the required edge above 0.3
    min_edge = np.array([0.0, 0.0, 0.0, 0.0, 0.5])
    assert backtest.wobi_side(edge, imb, 0.2, 0.6, min_edge).tolist() == [1, 0, -1, 0, 0]


def test_threshold_side():
    sig = np.array([0.7, -0.7, 0.6, 0.0, -0.61])
    assert backtest.threshold_side(sig, 0.6).tolist() == [1, -1, 0, 0, -1]


def test_summary_metrics():
    book = make_book(5000, seed=1)
    side = backtest.threshold_side(np.random.default_rng(0).uniform(-1, 1, 5000), 0.9)
    trades = backtest.fixed_horizon(book, side, 1000, BacktestConfig())
    s = metrics.summarize(trades)
    assert s["trades"] == len(trades)
    assert 0 <= s["win_rate"] <= 1
    assert s["ci95_low"] <= s["ev_net"] <= s["ci95_high"]
    assert s["ev_net"] == pytest.approx(trades["net_pnl"].mean())


def test_max_drawdown():
    # equity 0, 1, -1, -0.5, -1.5, 1.5: peak 1, trough -1.5
    assert metrics.max_drawdown(np.array([1.0, -2.0, 0.5, -1.0, 3.0])) == pytest.approx(2.5)
    assert metrics.max_drawdown(np.array([])) == 0.0
