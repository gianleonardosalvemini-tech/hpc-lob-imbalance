"""Configure and build the C engine with CMake (Ninja when available).

Usage:
    python scripts/build_engine.py [--no-openmp] [--no-native] [--debug] [--clean]

Outputs go to build/bin, where lobimb.engine looks for them. Use --clean to
switch generator (CMake caches it in build/CMakeCache.txt).
"""
from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BUILD = ROOT / "build"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--no-openmp", action="store_true")
    parser.add_argument("--no-native", action="store_true")
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--clean", action="store_true")
    args = parser.parse_args()

    if shutil.which("cmake") is None:
        print("cmake not found on PATH", file=sys.stderr)
        return 1
    if args.clean and BUILD.exists():
        shutil.rmtree(BUILD)

    configure = [
        "cmake", "-S", str(ROOT), "-B", str(BUILD),
        f"-DCMAKE_BUILD_TYPE={'Debug' if args.debug else 'Release'}",
        f"-DLOBIMB_OPENMP={'OFF' if args.no_openmp else 'ON'}",
        f"-DLOBIMB_NATIVE={'OFF' if args.no_native else 'ON'}",
    ]
    # The generator can only be chosen on the first configure.
    if shutil.which("ninja") and not (BUILD / "CMakeCache.txt").exists():
        configure += ["-G", "Ninja"]

    subprocess.run(configure, check=True)
    # --config only matters for multi-config generators (Visual Studio).
    subprocess.run(["cmake", "--build", str(BUILD), "--config", "Release"], check=True)
    print(f"Built libraries in {BUILD / 'bin'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
