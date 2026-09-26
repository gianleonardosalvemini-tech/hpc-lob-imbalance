# Legacy code (pre-refactor)

The original scripts and C sources, **unchanged**, kept only for before/after comparison.
Do not use them for research: they contain the bugs documented in `docs/report/strategy_report.tex`
(int-truncated volumes, broken DLL paths, mixed-unit entry filter, invented EV model).

- `src_c/OBI/lob_engine.c` and `src_c/WOBI/wobi_engine.c` are built as `lob_legacy` and
  `wobi_legacy` by CMake. They exist only so `lobimb.perf` can benchmark the old per-tick path.
- The replacement for each script:

| Legacy | Replacement |
|---|---|
| `OBI/backtest_strategy.py` | `scripts/run_obi_sweep.py` + `lobimb.backtest` |
| `OBI/python historical_ingestion.py` | `scripts/prepare_data.py`, `lobimb.data`, `lobimb.perf.legacy_obi_csv_loop` |
| `OBI/visualize_lob.py` | `lobimb.plots.price_vs_imbalance` |
| `OBI/check_data.py` | `lobimb.data.sanity_check` |
| `OBI/test_simulazione.py`, `OBI/quant_analyzer.py` | `tests/`, `tests/bench/` |
| `WOBI/quant_strategy.py` | `scripts/run_wobi_strategy.py` |
