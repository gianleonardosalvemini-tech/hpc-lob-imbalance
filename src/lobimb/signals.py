"""Signal API on top of the C engine.

Each function returns one value per snapshot, aligned with `book.ts`.
Prices are in USDT.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from . import engine
from .data import Book


def mid(book: Book) -> np.ndarray:
    """(best bid + best ask) / 2."""
    return 0.5 * (book.bid_px() + book.ask_px())


def spread(book: Book) -> np.ndarray:
    """Best ask - best bid."""
    return book.ask_px() - book.bid_px()


def obi(book: Book) -> np.ndarray:
    """Level-1 imbalance (Vb - Va) / (Vb + Va) in [-1, 1]."""
    return engine.obi(book.levels)


@dataclass(frozen=True)
class WobiSignals:
    """Outputs of `wobi`, one value per snapshot."""

    imbalance: np.ndarray  # depth-weighted imbalance in [-1, 1]
    micro: np.ndarray      # weighted micro-price
    edge: np.ndarray       # micro - mid, USDT


def wobi(book: Book, depth: int = 10, alpha: float = 0.5) -> WobiSignals:
    """Depth-weighted imbalance, micro-price and edge (micro - mid).

    The edge is in USDT, so it compares directly with spread and fees.
    """
    imbalance, micro = engine.wobi(book.levels, depth=depth, alpha=alpha)
    return WobiSignals(imbalance, micro, micro - mid(book))
