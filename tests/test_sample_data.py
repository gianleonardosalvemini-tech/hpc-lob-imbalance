"""scripts/make_sample_data.py: raw schema and a run through the pipeline."""
import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from lobimb import backtest, data, metrics, signals
from lobimb.config import BacktestConfig

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "make_sample_data.py"


@pytest.fixture(scope="module")
def sample_module():
    # scripts/ is not a package
    spec = importlib.util.spec_from_file_location("make_sample_data", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def sample_csv(sample_module, tmp_path_factory):
    return sample_module.write_csv(tmp_path_factory.mktemp("sample") / "sample.csv", 5_000, seed=1)


def test_sample_csv_schema(sample_csv):
    with open(sample_csv) as fh:
        assert fh.readline().strip() == "," + ",".join(str(i) for i in range(42))
    raw = pd.read_csv(sample_csv)
    assert raw.shape == (5_000, 43)
    assert (raw.iloc[:, 0] == np.arange(5_000)).all()
    stamp = pd.to_datetime(raw.iloc[:, 1], unit="ms").dt.strftime("%Y-%m-%d %H:%M:%S")
    assert (stamp == raw.iloc[:, 2]).all()
    book = raw.iloc[:, 3:].to_numpy(float)
    bids, asks = book[:, 0:20:2], book[:, 20::2]
    assert (np.diff(bids, axis=1) < 0).all() and (np.diff(asks, axis=1) > 0).all()
    np.testing.assert_allclose(book[:, 0::2] * 10, np.round(book[:, 0::2] * 10), atol=1e-6)
    vols = book[:, 1::2]
    assert (vols > 0).all() and (vols != np.round(vols)).mean() > 0.9


def test_sample_runs_through_pipeline(sample_csv):
    book = data.read_csv(sample_csv)
    assert len(book) == 5_000
    report = data.sanity_check(book)
    assert report.is_clean() and report.gaps == 5
    obi = signals.obi(book)
    trades = backtest.fixed_horizon(book, backtest.threshold_side(obi, 0.6), 5_000, BacktestConfig())
    s = metrics.summarize(trades)
    assert s["trades"] > 50
    assert s["mid_move"] > 0  # built into the generator
    wide = signals.wobi(book)
    assert np.isfinite(wide.edge).all()
    assert backtest.bracket(book, backtest.wobi_side(wide.edge, wide.imbalance, 0.05, 0.6),
                            1.0, 1.0, 2_500, BacktestConfig()).shape[1] == len(backtest.TRADE_COLUMNS)
