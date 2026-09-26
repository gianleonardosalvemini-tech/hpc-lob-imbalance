"""ctypes binding to the C batch engine (engine/src/lob_engine.c).

Each public function passes a whole array to C in one call: a ctypes call
costs around a microsecond, the kernel a few ns per row. Inputs that are
already float64 with contiguous rows are not copied.

The library is looked up at $LOBIMB_LIB, then build/bin/ (build it with
`python scripts/build_engine.py`).
"""
from __future__ import annotations

import contextlib
import ctypes
import os
import sys
from functools import lru_cache
from pathlib import Path
from typing import Iterator

import numpy as np

from .config import PROJECT_ROOT
from .data import DEPTH, ROW_WIDTH

_LIB_NAMES = {"win32": "lob_engine.dll", "darwin": "lob_engine.dylib"}
_DPTR = ctypes.POINTER(ctypes.c_double)

# LOB_ERR_NULL, LOB_ERR_ARG in lob_engine.h
_ERRORS = {-1: "null pointer", -2: "invalid argument"}


class EngineError(RuntimeError):
    """The C library is missing or a kernel returned an error status."""
    pass


def library_path(name: str | None = None) -> Path:
    """Path of a library in build/bin.

    $LOBIMB_LIB overrides the engine path, but not an explicit `name` (used by
    perf for the legacy libraries).
    """
    env = os.environ.get("LOBIMB_LIB")
    if env and name is None:
        return Path(env)
    name = name or _LIB_NAMES.get(sys.platform, "lob_engine.so")
    return PROJECT_ROOT / "build" / "bin" / name


@lru_cache(maxsize=None)
def _lib() -> ctypes.CDLL:
    """Load the engine once and declare the C signatures (ctypes defaults to int)."""
    path = library_path()
    if not path.exists():
        raise EngineError(f"{path} not found; run `python scripts/build_engine.py` first")
    lib = ctypes.CDLL(str(path))

    lib.lob_version.restype = ctypes.c_char_p
    lib.lob_version.argtypes = []
    lib.lob_max_threads.restype = ctypes.c_int
    lib.lob_max_threads.argtypes = []
    lib.lob_set_num_threads.restype = None
    lib.lob_set_num_threads.argtypes = [ctypes.c_int]

    # c_void_p so StreamingEngine can pass a plain int address (cheaper than
    # building a POINTER object per call).
    lib.lob_obi_one.restype = ctypes.c_double
    lib.lob_obi_one.argtypes = [ctypes.c_void_p]
    lib.lob_obi_batch.restype = ctypes.c_int
    lib.lob_obi_batch.argtypes = [_DPTR, ctypes.c_int64, ctypes.c_int64, _DPTR]

    lib.lob_wobi_one.restype = ctypes.c_int
    lib.lob_wobi_one.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_double, _DPTR, _DPTR]
    lib.lob_wobi_batch.restype = ctypes.c_int
    lib.lob_wobi_batch.argtypes = [_DPTR, ctypes.c_int64, ctypes.c_int64, ctypes.c_int,
                                   ctypes.c_double, _DPTR, _DPTR]
    return lib


def version() -> str:
    return _lib().lob_version().decode()


def max_threads() -> int:
    """OpenMP threads used by the batch kernels (1 for a serial build)."""
    return _lib().lob_max_threads()


def set_num_threads(n: int) -> None:
    """Set the OpenMP thread count (ignored if n <= 0)."""
    _lib().lob_set_num_threads(int(n))


@contextlib.contextmanager
def num_threads(n: int) -> Iterator[None]:
    """Temporarily change the OpenMP thread count."""
    previous = max_threads()
    set_num_threads(n)
    try:
        yield
    finally:
        set_num_threads(previous)


def _as_book(levels: np.ndarray) -> tuple[np.ndarray, int]:
    """Return a (n, 40) float64 array C can read, and its row stride in doubles.

    Values within a row must be adjacent, but rows may be further apart (e.g. a
    column slice of a wider array). Anything else, including a zero row stride
    (broadcast / `row[None, :]`) or a negative one (`levels[::-1]`), is copied.
    """
    arr = np.asarray(levels)
    if arr.ndim != 2 or arr.shape[1] != ROW_WIDTH:
        raise ValueError(f"expected shape (n, {ROW_WIDTH}), got {arr.shape}")
    if (arr.dtype != np.float64 or arr.strides[1] != 8 or arr.strides[0] % 8
            or arr.strides[0] < 8 * ROW_WIDTH):
        # Not np.ascontiguousarray: it leaves a (1, 40) view with row stride 0
        # as-is, since NumPy ignores the stride of a length-1 axis.
        arr = np.array(arr, dtype=np.float64, order="C", copy=True)
    return arr, arr.strides[0] // 8


def _check(rc: int) -> None:
    if rc != 0:
        raise EngineError(_ERRORS.get(rc, f"error code {rc}"))


def obi(levels: np.ndarray) -> np.ndarray:
    """Level-1 imbalance of every snapshot, shape (n,).

    0 for an empty top of book, NaN where a volume is NaN.
    """
    arr, stride = _as_book(levels)
    out = np.empty(arr.shape[0], dtype=np.float64)
    _check(_lib().lob_obi_batch(arr.ctypes.data_as(_DPTR), arr.shape[0], stride,
                                out.ctypes.data_as(_DPTR)))
    return out


def wobi(levels: np.ndarray, depth: int = DEPTH, alpha: float = 0.5) -> tuple[np.ndarray, np.ndarray]:
    """Depth-weighted imbalance and micro-price of every snapshot.

    Returns (imbalance, micro). Raises EngineError for depth outside [1, 10]
    or a negative/non-finite alpha.
    """
    arr, stride = _as_book(levels)
    imb = np.empty(arr.shape[0], dtype=np.float64)
    micro = np.empty(arr.shape[0], dtype=np.float64)
    _check(_lib().lob_wobi_batch(arr.ctypes.data_as(_DPTR), arr.shape[0], stride,
                                 int(depth), float(alpha),
                                 imb.ctypes.data_as(_DPTR), micro.ctypes.data_as(_DPTR)))
    return imb, micro


class StreamingEngine:
    """One C call per snapshot, as a live system would make them.

    Used by the latency benchmarks. Row addresses are precomputed so a call
    measures only FFI + kernel. Indices are bounds-checked because C gets raw
    addresses; the array is kept alive in `self._arr` for the same reason.
    """

    def __init__(self, levels: np.ndarray, depth: int = DEPTH, alpha: float = 0.5):
        self._arr, stride = _as_book(levels)
        base = self._arr.ctypes.data
        self._row_bytes = stride * 8
        self._n = self._arr.shape[0]
        self._base = base
        self._depth = int(depth)
        self._alpha = float(alpha)
        # Reused output slots: no allocation on the hot path.
        self._imb = ctypes.c_double()
        self._micro = ctypes.c_double()
        lib = _lib()
        self._obi_one = lib.lob_obi_one
        self._wobi_one = lib.lob_wobi_one

    def obi(self, i: int) -> float:
        if not 0 <= i < self._n:
            raise IndexError(f"snapshot index {i} out of range [0, {self._n})")
        return self._obi_one(self._base + i * self._row_bytes)

    def wobi(self, i: int) -> tuple[float, float]:
        """(imbalance, micro-price) of snapshot i."""
        if not 0 <= i < self._n:
            raise IndexError(f"snapshot index {i} out of range [0, {self._n})")
        _check(self._wobi_one(self._base + i * self._row_bytes, self._depth, self._alpha,
                              ctypes.byref(self._imb), ctypes.byref(self._micro)))
        return self._imb.value, self._micro.value
