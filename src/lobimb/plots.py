"""Figures. Each function saves to a file, closes its figure and returns the path."""
from __future__ import annotations

from pathlib import Path

import matplotlib

# Headless backend; must be set before importing pyplot.
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from .data import Book  # noqa: E402
from .signals import mid  # noqa: E402


def price_vs_imbalance(book: Book, imbalance: np.ndarray, path: Path,
                       title: str = "Mid price vs order book imbalance") -> Path:
    """Mid price (left axis) and an imbalance series (right axis) over time.

    Pass a short slice: millions of points are slow and unreadable.
    """
    t = pd.to_datetime(np.asarray(book.ts), unit="ms")
    fig, ax1 = plt.subplots(figsize=(14, 6))
    ax1.plot(t, mid(book), color="#1f77b4", linewidth=1.2, label="Mid price")
    ax1.set_ylabel("Mid price (USDT)", color="#1f77b4")
    ax1.grid(True, alpha=0.3)
    ax2 = ax1.twinx()
    ax2.plot(t, imbalance, color="#d62728", alpha=0.35, linewidth=0.7, label="Imbalance")
    ax2.axhline(0, color="black", linestyle="--", linewidth=1)
    ax2.set_ylabel("Imbalance", color="#d62728")
    ax2.set_ylim(-1.05, 1.05)
    fig.legend(loc="upper left")
    ax1.set_title(title)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def ev_decay(results: pd.DataFrame, path: Path, x: str = "horizon_s",
             y: str = "ev_net", hue: str = "fee_bps",
             title: str = "Net EV per trade vs holding horizon") -> Path:
    """Sweep results: `y` against `x` (log scale), one line per value of `hue`."""
    fig, ax = plt.subplots(figsize=(10, 6))
    for key, grp in results.groupby(hue):
        grp = grp.sort_values(x)
        ax.plot(grp[x], grp[y], marker="o", label=f"{hue}={key}")
    ax.axhline(0, color="black", linestyle="--", alpha=0.7)
    ax.set_xscale("log")
    ax.set_xlabel(x)
    ax.set_ylabel(f"{y} (USDT per BTC)")
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def histogram(values: dict[str, np.ndarray], path: Path, bins: int = 101,
              title: str = "Distribution") -> Path:
    """Overlaid density histograms; non-finite values are dropped."""
    fig, ax = plt.subplots(figsize=(10, 5))
    for label, v in values.items():
        ax.hist(v[np.isfinite(v)], bins=bins, alpha=0.5, density=True, label=label)
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path
