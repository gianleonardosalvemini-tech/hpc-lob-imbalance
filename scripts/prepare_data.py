"""Parse the raw CSV once, validate it and cache it as memory-mappable .npy files.

Usage:
    python scripts/prepare_data.py [--csv data/raw/1-09-1-20.csv] [--cache data/cache] [--nrows N]

One-off step; later runs load the .npy cache in milliseconds. The sanity
report is printed, not enforced.
"""
from __future__ import annotations

import argparse
import time
from dataclasses import asdict
from pathlib import Path

from lobimb import data
from lobimb.config import CACHE_DIR, RAW_CSV


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", type=Path, default=RAW_CSV)
    parser.add_argument("--cache", type=Path, default=CACHE_DIR)
    parser.add_argument("--nrows", type=int, default=None)
    args = parser.parse_args()

    t0 = time.perf_counter()
    book = data.read_csv(args.csv, nrows=args.nrows)
    t_parse = time.perf_counter() - t0
    data.write_cache(book, args.cache)
    t_total = time.perf_counter() - t0

    # MB/s uses the whole file size, so it is overstated with --nrows.
    size_mb = args.csv.stat().st_size / 1e6
    print(f"rows={len(book):,}  parse={t_parse:.1f}s ({size_mb / t_parse:.0f} MB/s)  "
          f"total={t_total:.1f}s -> {args.cache}")
    for key, value in asdict(data.sanity_check(book)).items():
        print(f"  {key:>18}: {value:,}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
