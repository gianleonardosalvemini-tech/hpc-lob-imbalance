"""Latency/throughput measurements for scripts/run_benchmarks.py and tests/bench.

Throughput (ns/row of a batch call) is what research runs care about; latency
percentiles of single-snapshot calls are what a live system would pay. The
legacy_* functions replay the pre-refactor per-tick loops for comparison.
"""
from __future__ import annotations

import csv
import ctypes
import gc
import itertools
import subprocess
import time
from pathlib import Path
from typing import Callable

import numpy as np

from . import backtest, engine, reference, signals
from .config import PROJECT_ROOT, BacktestConfig
from .data import Book


def best_of(fn: Callable[[], object], repeat: int = 5) -> float:
    """Best wall time in seconds over `repeat` runs (noise only adds time)."""
    best = float("inf")
    for _ in range(repeat):
        t0 = time.perf_counter()
        fn()
        best = min(best, time.perf_counter() - t0)
    return best


def percentiles_ns(samples_ns: np.ndarray, prefix: str) -> dict[str, float]:
    """p50/p99/p99.9/max/mean, keyed "<prefix>_<stat>_ns".

    Interpolated, unlike the nearest rank in bench_engine.c; negligible
    difference at these sample counts.
    """
    p50, p99, p999 = np.percentile(samples_ns, [50, 99, 99.9])
    return {f"{prefix}_p50_ns": float(p50), f"{prefix}_p99_ns": float(p99),
            f"{prefix}_p999_ns": float(p999), f"{prefix}_max_ns": float(samples_ns.max()),
            f"{prefix}_mean_ns": float(samples_ns.mean())}


def timer_overhead_ns(n: int = 200_000) -> float:
    """Median cost of two back-to-back perf_counter_ns() reads."""
    clock = time.perf_counter_ns
    samples = np.empty(n)
    for i in range(n):
        t0 = clock()
        samples[i] = clock() - t0
    return float(np.median(samples))


# --------------------------------------------------------------------------- batch

def batch_throughput(levels: np.ndarray, threads: list[int], repeat: int = 5) -> list[dict]:
    """ns/row and rows/s of the C batch kernels for each thread count.

    Speedup is relative to the first entry, so pass 1 thread first.
    """
    rows = []
    n = levels.shape[0]
    for t in threads:
        with engine.num_threads(t):
            t_obi = best_of(lambda: engine.obi(levels), repeat)
            t_wobi = best_of(lambda: engine.wobi(levels), repeat)
        rows.append({"threads": t, "obi_ns_row": t_obi / n * 1e9, "wobi_ns_row": t_wobi / n * 1e9,
                     "obi_mrows_s": n / t_obi / 1e6, "wobi_mrows_s": n / t_wobi / 1e6})
    base_obi, base_wobi = rows[0]["obi_ns_row"], rows[0]["wobi_ns_row"]
    for r in rows:
        r["obi_speedup"] = base_obi / r["obi_ns_row"]
        r["wobi_speedup"] = base_wobi / r["wobi_ns_row"]
        r["wobi_efficiency"] = r["wobi_speedup"] / r["threads"]
    return rows


def numpy_baseline(levels: np.ndarray, repeat: int = 3) -> dict[str, float]:
    """Same formulas in vectorised NumPy (single-threaded, full-size temporaries)."""
    n = levels.shape[0]
    arr = np.ascontiguousarray(levels)
    return {"numpy_obi_ns_row": best_of(lambda: reference.obi(arr), repeat) / n * 1e9,
            "numpy_wobi_ns_row": best_of(lambda: reference.wobi(arr, 10, 0.5), repeat) / n * 1e9}


# ----------------------------------------------------------------- per-tick (streaming)

def streaming_latency(levels: np.ndarray, n_calls: int = 200_000) -> dict[str, float]:
    """Per-snapshot latency through Python + ctypes, each call timed individually.

    GC is disabled during the loop so a collection doesn't show up as a
    kernel outlier.
    """
    stream = engine.StreamingEngine(levels)
    clock = time.perf_counter_ns
    n_rows = levels.shape[0]
    out: dict[str, float] = {}
    for name, fn in (("stream_obi", stream.obi), ("stream_wobi", stream.wobi)):
        samples = np.empty(n_calls)
        gc.disable()
        try:
            for k in range(n_calls):
                i = k % n_rows
                t0 = clock()
                fn(i)
                samples[k] = clock() - t0
        finally:
            gc.enable()
        out.update(percentiles_ns(samples, name))
    return out


# ------------------------------------------------------------------------- legacy

# Mirrors of the structs in legacy/ (must match the C definitions). The int
# volume in _LegacyLevel is the truncation bug.

class _LegacyLevel(ctypes.Structure):
    _fields_ = [("price", ctypes.c_double), ("volume", ctypes.c_int)]


class _LegacyState(ctypes.Structure):
    _fields_ = [("timestamp", ctypes.c_longlong), ("best_bid", _LegacyLevel),
                ("best_ask", _LegacyLevel)]


class _WLevel(ctypes.Structure):
    _fields_ = [("price", ctypes.c_double), ("volume", ctypes.c_double)]


class _WState(ctypes.Structure):
    _fields_ = [("bids", _WLevel * 10), ("asks", _WLevel * 10)]


class _WMetrics(ctypes.Structure):
    _fields_ = [("weighted_micro_price", ctypes.c_double), ("imbalance_pressure", ctypes.c_double)]


def legacy_obi_csv_loop(csv_path: Path, nrows: int | None) -> dict[str, float]:
    """Replay the original historical_ingestion.py loop on the old -O0 DLL.

    csv.reader, int() truncation and two ctypes calls per tick. CSV parsing is
    timed too, as in the original pipeline.
    """
    lib = ctypes.CDLL(str(engine.library_path("lob_legacy.dll" if _win() else "lob_legacy.so")))
    lib.init_lob_state.argtypes = [ctypes.POINTER(_LegacyState), ctypes.c_longlong, ctypes.c_double,
                                   ctypes.c_int, ctypes.c_double, ctypes.c_int]
    lib.compute_raw_obi.argtypes = [ctypes.POINTER(_LegacyState)]
    lib.compute_raw_obi.restype = ctypes.c_double
    state = _LegacyState()
    ticks = 0
    t0 = time.perf_counter()
    with open(csv_path, newline="") as fh:
        reader = csv.reader(fh)
        next(reader)
        for row in itertools.islice(reader, nrows):
            lib.init_lob_state(ctypes.byref(state), int(row[1]), float(row[3]), int(float(row[4])),
                               float(row[23]), int(float(row[24])))
            lib.compute_raw_obi(ctypes.byref(state))
            ticks += 1
    elapsed = time.perf_counter() - t0
    return {"legacy_obi_rows": ticks, "legacy_obi_total_s": elapsed,
            "legacy_obi_ns_row": elapsed / ticks * 1e9}


def legacy_wobi_loop(levels: np.ndarray, nrows: int) -> dict[str, float]:
    """Replay the original WOBI/quant_strategy.py loop (CSV parsing excluded).

    The 40 ctypes field writes per tick dominate; the C part is negligible.
    """
    lib = ctypes.CDLL(str(engine.library_path("wobi_legacy.dll" if _win() else "wobi_legacy.so")))
    lib.compute_signals.restype = _WMetrics
    lib.compute_signals.argtypes = [ctypes.POINTER(_WState), ctypes.c_double]
    state = _WState()
    rows = levels[:nrows].tolist()
    samples = np.empty(len(rows))
    clock = time.perf_counter_ns
    t_all = time.perf_counter()
    for k, row in enumerate(rows):
        t0 = clock()
        for i in range(10):
            state.bids[i].price = row[2 * i]
            state.bids[i].volume = row[2 * i + 1]
            state.asks[i].price = row[20 + 2 * i]
            state.asks[i].volume = row[21 + 2 * i]
        lib.compute_signals(ctypes.byref(state), 0.5)
        samples[k] = clock() - t0
    elapsed = time.perf_counter() - t_all
    out = {"legacy_wobi_rows": len(rows), "legacy_wobi_ns_row": elapsed / len(rows) * 1e9}
    out.update(percentiles_ns(samples, "legacy_wobi"))
    return out


def _win() -> bool:
    import sys
    return sys.platform == "win32"


# ---------------------------------------------------------------------- end to end

def end_to_end(book: Book, cfg: BacktestConfig = BacktestConfig()) -> dict[str, float]:
    """Signals plus one OBI fixed-horizon and one WOBI bracket backtest, default params.

    Returns seconds per stage and trade counts.
    """
    t0 = time.perf_counter()
    obi = signals.obi(book)
    t1 = time.perf_counter()
    w = signals.wobi(book)
    t2 = time.perf_counter()
    trades = backtest.fixed_horizon(book, backtest.threshold_side(obi, 0.6), 1_000, cfg)
    t3 = time.perf_counter()
    side = backtest.wobi_side(w.edge, w.imbalance, 0.2, 0.6)
    wtrades = backtest.bracket(book, side, 1.0, 1.0, 2_500, cfg)
    t4 = time.perf_counter()
    return {"e2e_obi_s": t1 - t0, "e2e_wobi_s": t2 - t1, "e2e_obi_backtest_s": t3 - t2,
            "e2e_obi_trades": len(trades), "e2e_wobi_backtest_s": t4 - t3,
            "e2e_wobi_trades": len(wtrades), "e2e_total_s": t4 - t0}


def load_times(cache_dir: Path) -> dict[str, float]:
    """Time opening the memory map, a first scan through it, and a full np.load.

    The first scan is fast only when the OS page cache is warm.
    """
    from . import data
    t0 = time.perf_counter()
    book = data.load(cache_dir, mmap=True)
    t1 = time.perf_counter()
    float(np.asarray(book.levels[:, 0]).sum())  # touches every row
    t2 = time.perf_counter()
    in_ram = data.load(cache_dir, mmap=False)
    t3 = time.perf_counter()
    del in_ram
    return {"load_mmap_open_s": t1 - t0, "load_mmap_first_scan_s": t2 - t1,
            "load_npy_full_s": t3 - t2}


# ------------------------------------------------------------------------ native C

def native_bench(rows: int = 3_730_870, calls: int = 1_000_000,
                 threads: int | None = None) -> dict[str, float]:
    """Run build/bin/bench_engine and parse its key=value output.

    Keys get a "c_" prefix; `threads` sets OMP_NUM_THREADS for the child only.
    """
    exe = PROJECT_ROOT / "build" / "bin" / ("bench_engine.exe" if _win() else "bench_engine")
    env = None
    if threads is not None:
        import os
        env = {**os.environ, "OMP_NUM_THREADS": str(threads)}
    out = subprocess.run([str(exe), str(rows), str(calls)], capture_output=True, text=True,
                         check=True, env=env).stdout
    result = {}
    for line in out.splitlines():
        key, _, value = line.partition("=")
        try:
            result[f"c_{key}"] = float(value)
        except ValueError:
            result[f"c_{key}"] = value  # version string
    return result
