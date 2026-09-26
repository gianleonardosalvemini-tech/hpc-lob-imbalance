"""lobimb.data: Book, CSV parsing, cache, sanity checks, split and describe()."""
from pathlib import Path

import numpy as np
import pytest

from conftest import flat_book, make_book
from lobimb import backtest, data, metrics, signals
from lobimb.config import BacktestConfig


def test_book_shape_validation():
    with pytest.raises(ValueError):
        data.Book(np.arange(3), np.zeros((3, 39)))
    with pytest.raises(ValueError):
        data.Book(np.arange(4), np.zeros((3, 40)))


def test_accessors_follow_engine_layout():
    book = make_book(10)
    assert np.all(book.bid_px(0) == book.levels[:, 0])
    assert np.all(book.ask_vol(9) == book.levels[:, 20 + 19])
    assert np.all(book.bid_px() < book.ask_px())


def test_csv_roundtrip(tmp_path):
    book = make_book(200, seed=3)
    # raw layout: index, ts, datetime string, 40 book columns
    header = "," + ",".join(str(i) for i in range(42))
    lines = [header]
    for i in range(len(book)):
        vals = ",".join(repr(float(v)) for v in book.levels[i])
        lines.append(f"{i},{book.ts[i]},2023-01-09 22:17:40,{vals}")
    csv = tmp_path / "raw.csv"
    csv.write_text("\n".join(lines))

    parsed = data.read_csv(csv)
    np.testing.assert_array_equal(parsed.ts, book.ts)
    # pandas' fast float parser may differ in the last ULP
    np.testing.assert_allclose(parsed.levels, book.levels, rtol=1e-12)

    cache = tmp_path / "cache"
    loaded = data.load(cache_dir=cache, csv_path=csv)
    assert data.cache_exists(cache)
    np.testing.assert_array_equal(loaded.levels, parsed.levels)


def test_sanity_check_counts_problems():
    book = make_book(100)
    ts = book.ts.copy()
    lv = book.levels.copy()
    ts[50:] += 10_000                   # one gap of 10.25 s
    lv[10, 0] = lv[10, 20] + 1          # crossed book
    lv[20, 5] = np.nan
    lv[30, 3] = -1.0                    # negative volume
    report = data.sanity_check(data.Book(ts, lv))
    assert report.gaps == 1
    assert report.max_gap_ms == 10_250
    assert report.crossed == 1
    assert report.nan_rows == 1
    assert report.negative_vol == 1
    assert not report.is_clean()
    assert data.sanity_check(make_book(100)).is_clean()


def test_in_sample_split():
    book = make_book(100)
    ins, oos = data.in_sample_split(book, 0.7)
    assert len(ins) == 70 and len(oos) == 30
    assert ins.ts[-1] < oos.ts[0]


@pytest.mark.data
def test_real_dataset_is_clean(real_book):
    report = data.sanity_check(real_book)
    assert report.n_rows == 3_730_870
    assert report.crossed == 0 and report.nan_rows == 0 and report.non_positive_px == 0


def test_describe_on_hand_made_book():
    # spread 0.1 everywhere, 1 BTC per level, one 1.25 s interval before row 3;
    # row 0 has 3 BTC on the bid, so its OBI is 0.5 and the rest are 0
    base = flat_book([100.0, 100.1, 100.2, 100.3, 100.4],
                     [100.1, 100.2, 100.3, 100.4, 100.5])
    ts = base.ts.copy()
    ts[3:] += 1_000
    lv = base.levels.copy()
    lv[0, 1] = 3.0
    d = data.describe(data.Book(ts, lv), in_sample_fraction=0.6)
    assert d["n_snapshots"] == 5
    assert d["period"]["start_ms"] == 0 and d["period"]["end_ms"] == 2_000
    assert d["sampling"]["median_dt_ms"] == 250
    assert d["sampling"]["max_dt_ms"] == 1_250
    assert d["sampling"]["gaps_gt_500ms"] == 1 and d["sampling"]["gaps_gt_2s"] == 0
    assert d["price"]["mid_min"] == pytest.approx(100.05)
    assert d["price"]["mid_max"] == pytest.approx(100.45)
    assert d["spread"]["median"] == pytest.approx(0.1) and d["spread"]["share_one_tick"] == 1.0
    assert d["best_volume"]["bid"]["mean"] == pytest.approx((3 + 4 * 1) / 5)
    assert d["best_volume"]["ask"]["share_lt_1"] == 0.0
    assert d["depth10_volume"]["ask"]["median"] == pytest.approx(10.0)
    assert d["depth10_volume"]["bid"]["mean"] == pytest.approx((12 + 4 * 10) / 5)
    assert d["obi"]["mean_abs"] == pytest.approx(0.5 / 5)
    assert d["obi"]["share_abs_gt_0_6"] == 0.0 and d["obi"]["share_abs_eq_1"] == 0.0
    assert d["split"]["in_sample_rows"] == 3 and d["split"]["out_of_sample_rows"] == 2
    assert d["split"]["out_of_sample_start_utc"].startswith("1970-01-01T00:00:01.750")


def test_describe_small_volumes_and_obi_extremes():
    book = flat_book([100.0] * 4, [100.1] * 4)
    lv = book.levels.copy()
    lv[0, 1], lv[0, data.ASK_OFFSET + 1] = 0.005, 0.5
    lv[1, 1], lv[1, data.ASK_OFFSET + 1] = 2.0, 0.0   # OBI = +1
    d = data.describe(data.Book(book.ts, lv))
    assert d["best_volume"]["bid"]["share_lt_0_01"] == 0.25
    assert d["best_volume"]["bid"]["share_lt_1"] == 0.25
    assert d["best_volume"]["ask"]["share_lt_1"] == 0.5
    assert d["obi"]["share_abs_eq_1"] == 0.25
    assert d["obi"]["share_abs_gt_0_6"] == pytest.approx(0.5)  # rows 0 (0.98) and 1


def test_describe_is_json_serialisable_and_matches_sanity():
    import json

    book = make_book(1_000, seed=5)
    d = data.describe(book)
    assert json.loads(json.dumps(d)) == d
    report = data.sanity_check(book)
    assert d["n_snapshots"] == report.n_rows
    assert d["sampling"]["max_dt_ms"] == report.max_gap_ms
    ins, _ = data.in_sample_split(book, 0.7)
    assert d["split"]["in_sample_rows"] == len(ins)
    assert 0.0 <= d["spread"]["share_one_tick"] <= 1.0
    with pytest.raises(ValueError):
        data.describe(book[:1])


@pytest.mark.data
def test_real_dataset_matches_csv(real_book):
    """The .npy cache matches the first and last rows of the raw CSV."""
    import pandas as pd

    from lobimb.config import RAW_CSV

    if not RAW_CSV.exists():
        pytest.skip("raw CSV missing")
    first = data.read_csv(RAW_CSV, nrows=100)
    np.testing.assert_array_equal(real_book.ts[:100], first.ts)
    np.testing.assert_array_equal(real_book.levels[:100], first.levels)
    n = len(real_book)
    tail = pd.read_csv(RAW_CSV, skiprows=range(1, n - 1), header=0)
    assert len(tail) == 2 and int(tail.iloc[-1, 0]) == n - 1
    np.testing.assert_array_equal(real_book.ts[-2:], tail.iloc[:, 1].to_numpy(np.int64))
    np.testing.assert_array_equal(real_book.levels[-2:], tail.iloc[:, 3:].to_numpy(np.float64))


@pytest.mark.data
def test_real_dataset_summary(real_book):
    """Figures quoted in docs/report/data_summary.tex."""
    d = data.describe(real_book)
    assert d["period"]["start_utc"].startswith("2023-01-09T22:17:40")
    assert d["period"]["end_utc"].startswith("2023-01-20T18:10:48")
    assert d["sampling"]["median_dt_ms"] == 250
    assert d["spread"]["median"] == pytest.approx(0.1)
    assert d["spread"]["share_one_tick"] > 0.99


REAL_SAMPLE = Path(__file__).resolve().parents[1] / "sample_data" / "btc_lob_sample_100k.csv.gz"


@pytest.mark.skipif(not REAL_SAMPLE.exists(), reason="sample_data excerpt missing")
def test_real_sample_excerpt():
    # the 100k-row excerpt shipped in the repo: clean, and OBI predicts the 5 s mid move
    book = data.read_csv(REAL_SAMPLE)
    assert len(book) == 100_000
    assert data.sanity_check(book).is_clean()
    trades = backtest.fixed_horizon(book, backtest.threshold_side(signals.obi(book), 0.6),
                                    5_000, BacktestConfig())
    assert metrics.summarize(trades)["mid_move"] > 0
