"""How much does the book imbalance predict, and how fast does it decay?

  1. Deciles: mean future mid move (1 s, 5 s, 20 s ahead) per decile of
     level-1 OBI and of the WOBI edge, full sample and out-of-sample. Windows
     containing a gap > max_gap_ms are dropped. Standard errors assume
     independent snapshots, but windows overlap, so the error bars are
     optimistic: they rule out pure noise, nothing more.
  2. Latency sweep: OBI > 0.6 taker (no fees, no slippage), out-of-sample,
     1 s and 5 s holds, fill latency 0-1000 ms.

A round trip needs a mid move above one full spread (0.1 USDT) to break even
before fees; the decile chart marks +/- half a spread (one fill).

Outputs:
    results/signal_deciles.csv, results/latency_sweep.csv
    obi_deciles.png and latency_decay.png in each --figures directory
    (default docs/figures and docs/report/figures)

Usage:
    python scripts/run_signal_analysis.py [--cache data/cache] [--out results]
                                          [--figures docs/figures docs/report/figures]
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from lobimb import backtest, data, metrics, signals  # noqa: E402
from lobimb.config import (CACHE_DIR, PROJECT_ROOT, RESULTS_DIR, BacktestConfig,  # noqa: E402
                           CostModel, ObiConfig)

TICK = 0.1
HALF_SPREAD = 0.05
DECILE_HORIZONS_MS = (1_000, 5_000, 20_000)
LATENCIES_MS = (0, 250, 500, 1_000)
LATENCY_HORIZONS_MS = (1_000, 5_000)
# One colour/marker per horizon.
COLORS = ["#2a78d6", "#eb6834", "#1baf7a"]
MARKERS = ["o", "s", "^"]


def future_move(book: data.Book, horizon_ms: int, max_gap_ms: int) -> tuple[np.ndarray, np.ndarray]:
    """Mid move to the first snapshot at or after t + h, and a validity mask.

    Invalid when the window runs past the data or contains a gap > max_gap_ms.
    """
    ts = np.asarray(book.ts)
    mid = np.asarray(signals.mid(book))
    j = np.searchsorted(ts, ts + horizon_ms, side="left")
    valid = j < len(ts)
    j = np.minimum(j, len(ts) - 1)
    gaps = np.concatenate([[0], np.cumsum(np.diff(ts) > max_gap_ms)])
    valid &= gaps[j] == gaps
    return mid[j] - mid, valid


def decile_table(x: np.ndarray, moves: dict[int, tuple[np.ndarray, np.ndarray]]) -> pd.DataFrame:
    """Mean future move and standard error per decile of `x`, for each horizon."""
    edges = np.quantile(x, np.linspace(0.1, 0.9, 9))
    bucket = np.searchsorted(edges, x, side="right")  # 0..9
    rows = []
    for h, (move, valid) in moves.items():
        b, m, s = bucket[valid], move[valid], x[valid]
        n = np.bincount(b, minlength=10)
        with np.errstate(invalid="ignore", divide="ignore"):
            mean = np.bincount(b, weights=m, minlength=10) / n
            # Two-pass variance (ddof=1) for numerical stability.
            var = np.bincount(b, weights=(m - mean[b]) ** 2, minlength=10) / (n - 1)
            se = np.sqrt(var / n)
            sig_mean = np.bincount(b, weights=s, minlength=10) / n
        for k in range(10):
            rows.append({"horizon_s": h / 1000, "decile": k + 1, "n": int(n[k]),
                         "signal_mean": sig_mean[k],
                         "signal_lo": edges[k - 1] if k > 0 else float(np.min(x)),
                         "signal_hi": edges[k] if k < 9 else float(np.max(x)),
                         "mean_move": mean[k], "se": se[k],
                         "mean_move_ticks": mean[k] / TICK, "se_ticks": se[k] / TICK})
    return pd.DataFrame(rows)


def style(ax: plt.Axes) -> None:
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(True, axis="y", alpha=0.25)
    ax.set_axisbelow(True)


def save(fig: plt.Figure, name: str, dirs: list[Path]) -> None:
    for d in dirs:
        d.mkdir(parents=True, exist_ok=True)
        fig.savefig(d / name, dpi=150)
    plt.close(fig)


def plot_deciles(tab: pd.DataFrame, dirs: list[Path], title: str) -> None:
    """Mean future mid move vs decile mean OBI, one line per horizon, 95% CI bars."""
    fig, ax = plt.subplots(figsize=(8, 5))
    for (h, grp), color, marker in zip(tab.groupby("horizon_s"), COLORS, MARKERS):
        ax.errorbar(grp.signal_mean, grp.mean_move, yerr=1.96 * grp.se, color=color,
                    marker=marker, markersize=6, linewidth=2, capsize=3, label=f"{h:g} s ahead")
    ax.axhline(0, color="#444444", linewidth=1)
    for y in (HALF_SPREAD, -HALF_SPREAD):
        ax.axhline(y, color="#888888", linewidth=1, linestyle="--")
    ax.text(ax.get_xlim()[0], HALF_SPREAD, " +half spread (0.05 USDT, one taker fill)",
            va="bottom", ha="left", fontsize=8, color="#555555")
    ax.text(ax.get_xlim()[1], -HALF_SPREAD, "-half spread ", va="top", ha="right",
            fontsize=8, color="#555555")
    ax.set_xlabel("Level-1 OBI (mean of decile)")
    ax.set_ylabel("Mean future mid move (USDT)")
    ax.set_title(title, fontsize=11)
    ax.legend(frameon=False, loc="upper left", bbox_to_anchor=(0, 0.93))
    style(ax)
    fig.tight_layout()
    save(fig, "obi_deciles.png", dirs)


def plot_latency(res: pd.DataFrame, dirs: list[Path], threshold: float) -> None:
    """Gross EV per trade vs fill latency, one line per holding horizon."""
    fig, ax = plt.subplots(figsize=(8, 5))
    for (h, grp), color, marker in zip(res.groupby("horizon_s"), COLORS, MARKERS):
        ax.plot(grp.latency_ms, grp.ev_gross, color=color, marker=marker, markersize=7,
                linewidth=2, label=f"hold {h:g} s")
        last = grp.iloc[-1]
        ax.annotate(f"{last.ev_gross:.2f}", (last.latency_ms, last.ev_gross),
                    textcoords="offset points", xytext=(6, 0), va="center", fontsize=8)
    ax.axhline(0, color="#444444", linewidth=1)
    ax.set_xticks(list(LATENCIES_MS))
    ax.set_xlabel("Fill latency after the signal (ms)")
    ax.set_ylabel("Gross EV per trade (USDT per BTC, before fees)")
    ax.set_title(f"OBI > {threshold}, out-of-sample: gross edge decays with latency", fontsize=11)
    if (res.ev_gross > 0).all():
        ax.set_ylim(bottom=0)
    ax.legend(frameon=False)
    style(ax)
    fig.tight_layout()
    save(fig, "latency_decay.png", dirs)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--cache", type=Path, default=CACHE_DIR,
                        help="directory with the ts.npy/levels.npy cache")
    parser.add_argument("--out", type=Path, default=RESULTS_DIR, help="output directory for CSVs")
    parser.add_argument("--figures", type=Path, nargs="*",
                        default=[PROJECT_ROOT / "docs" / "figures",
                                 PROJECT_ROOT / "docs" / "report" / "figures"])
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    cfg = BacktestConfig()
    threshold = ObiConfig().threshold

    book = data.load(args.cache)
    obi = signals.obi(book)
    edge = signals.wobi(book).edge
    _, oos = data.in_sample_split(book, cfg.in_sample_fraction)
    n_in = len(book) - len(oos)

    # --- 1. Deciles -------------------------------------------------------------
    tables = []
    for sample, part, sl in (("full", book, slice(None)), ("out-of-sample", oos, slice(n_in, None))):
        moves = {h: future_move(part, h, cfg.max_gap_ms) for h in DECILE_HORIZONS_MS}
        for name, x in (("obi", obi[sl]), ("wobi_edge", edge[sl])):
            t = decile_table(np.asarray(x), moves)
            t.insert(0, "signal", name)
            t.insert(0, "sample", sample)
            tables.append(t)
    deciles = pd.concat(tables, ignore_index=True)
    deciles.to_csv(args.out / "signal_deciles.csv", index=False)

    for (sample, name), grp in deciles.groupby(["sample", "signal"], sort=False):
        wide = grp.pivot(index="decile", columns="horizon_s", values="mean_move_ticks")
        wide.columns = [f"{h:g}s ticks" for h in wide.columns]
        se = grp[grp.horizon_s == DECILE_HORIZONS_MS[-1] / 1000].set_index("decile").se_ticks
        wide.insert(0, "mean " + name, grp.groupby("decile").signal_mean.first())
        wide[f"se {DECILE_HORIZONS_MS[-1] // 1000}s"] = se
        print(f"\n1. Future mid move by {name} decile ({sample}, n={grp.n.sum() // 3:,}), ticks of 0.1")
        print(wide.to_string(float_format="%.3f"))

    plot_deciles(deciles[(deciles["sample"] == "out-of-sample") & (deciles.signal == "obi")],
                 args.figures, "Future mid move by level-1 OBI decile (out-of-sample, 95% CI)")

    # --- 2. Latency sweep -------------------------------------------------------
    side = backtest.threshold_side(obi[n_in:], threshold)
    rows = []
    for h in LATENCY_HORIZONS_MS:
        for lat in LATENCIES_MS:
            bt = BacktestConfig(costs=CostModel(fee_bps=0, slippage=0.0, latency_ms=lat))
            s = metrics.summarize(backtest.fixed_horizon(oos, side, h, bt))
            rows.append({"split": "out-of-sample", "threshold": threshold, "horizon_s": h / 1000,
                         "latency_ms": lat, "fee_bps": 0.0, "trades": s["trades"],
                         "mid_move": s.get("mid_move", np.nan), "ev_gross": s.get("ev_gross", np.nan),
                         "t_stat": s.get("t_stat", np.nan)})
    lat_res = pd.DataFrame(rows)
    lat_res.to_csv(args.out / "latency_sweep.csv", index=False)
    print(f"\n2. Latency sweep: OBI > {threshold}, out-of-sample, no fees (gross = after spread)")
    print(lat_res[["horizon_s", "latency_ms", "trades", "mid_move", "ev_gross", "t_stat"]]
          .to_string(index=False, float_format="%.3f"))
    plot_latency(lat_res, args.figures, threshold)

    print(f"\nresults written to {args.out}; figures to {', '.join(map(str, args.figures))}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
