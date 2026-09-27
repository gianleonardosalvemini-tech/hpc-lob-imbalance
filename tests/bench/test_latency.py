"""Latency / throughput KPI tests (`pytest -m bench`).

Budgets are loose, roughly 5-10x the numbers in docs/BENCHMARKS.md; scale them
with LOBIMB_BUDGET_SCALE on slower hardware. The test_bench_* functions only
report pytest-benchmark statistics; test_kpi_* assert budgets.
"""
import os

import numpy as np
import pytest

from conftest import make_levels
from lobimb import data, engine, perf

pytestmark = pytest.mark.bench

SCALE = float(os.environ.get("LOBIMB_BUDGET_SCALE", "1"))


def budget(value: float) -> float:
    return value * SCALE


@pytest.fixture(scope="module")
def levels() -> np.ndarray:
    """Real dataset in RAM when cached, otherwise 1M synthetic rows."""
    if data.cache_exists():
        return np.asarray(data.load(mmap=False).levels)
    return make_levels(1_000_000, seed=7)



def test_bench_obi_batch(benchmark, levels):
    benchmark(engine.obi, levels)


def test_bench_wobi_batch(benchmark, levels):
    benchmark(engine.wobi, levels)


def test_bench_wobi_batch_single_thread(benchmark, levels):
    with engine.num_threads(1):
        benchmark(engine.wobi, levels)


def test_bench_stream_wobi_call(benchmark, levels):
    stream = engine.StreamingEngine(levels)
    benchmark(stream.wobi, 12345)


# KPI budgets (assertions)

def test_kpi_batch_throughput(levels):
    # Low speedup bar: the kernels are memory-bandwidth bound.
    rows = perf.batch_throughput(levels, [1, engine.max_threads()], repeat=3)
    single, multi = rows
    print(f"\nWOBI batch: {single['wobi_ns_row']:.1f} ns/row (1 thr), "
          f"{multi['wobi_ns_row']:.1f} ns/row ({multi['threads']} thr), "
          f"speedup {multi['wobi_speedup']:.1f}x")
    assert multi["wobi_ns_row"] < budget(50)
    assert multi["obi_ns_row"] < budget(30)
    if engine.max_threads() >= 4:
        assert multi["wobi_speedup"] > 1.5 / SCALE


def test_kpi_streaming_tick_to_signal(levels):
    kpi = perf.streaming_latency(levels, n_calls=100_000)
    print(f"\nstream WOBI p50={kpi['stream_wobi_p50_ns']:.0f} ns "
          f"p99={kpi['stream_wobi_p99_ns']:.0f} ns p99.9={kpi['stream_wobi_p999_ns']:.0f} ns")
    assert kpi["stream_obi_p50_ns"] < budget(5_000)
    assert kpi["stream_wobi_p50_ns"] < budget(10_000)
    assert kpi["stream_wobi_p99_ns"] < budget(50_000)


def test_kpi_batch_beats_legacy_per_tick(levels):
    legacy = perf.legacy_wobi_loop(levels, nrows=20_000)
    batch = perf.batch_throughput(levels, [engine.max_threads()], repeat=3)[0]
    speedup = legacy["legacy_wobi_ns_row"] / batch["wobi_ns_row"]
    print(f"\nlegacy {legacy['legacy_wobi_ns_row']:.0f} ns/row vs batch "
          f"{batch['wobi_ns_row']:.1f} ns/row -> {speedup:.0f}x")
    assert speedup > 100 / SCALE


def test_kpi_native_floor():
    # needs build/bin/bench_engine
    kpi = perf.native_bench(rows=500_000, calls=200_000)
    print(f"\nC-only single WOBI p50={kpi['c_single_wobi_p50_ns']:.1f} ns, "
          f"batch {kpi['c_batch_wobi_ns_per_row']:.2f} ns/row")
    assert kpi["c_single_wobi_p50_ns"] < budget(500)
