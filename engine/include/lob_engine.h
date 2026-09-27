/*
 * lob_engine.h - batch order-book imbalance kernels.
 *
 * Each snapshot is one row of LOB_ROW_WIDTH doubles in the same column order
 * as the raw CSV (columns 3..42), so NumPy arrays can be passed without copies:
 *
 *   [bid_p0, bid_v0, ..., bid_p9, bid_v9, ask_p0, ask_v0, ..., ask_p9, ask_v9]
 *
 * Level 0 is the best price. Volumes are doubles because BTC trades in
 * fractions (the v1 engine stored them as int and truncated most levels to 0).
 *
 * `stride` is the distance in doubles between consecutive rows
 * (>= LOB_ROW_WIDTH), so a view of a wider array can be passed as-is.
 *
 * The *_batch functions exist because each FFI call from Python costs far more
 * than the per-row arithmetic; the *_one functions are for the streaming
 * latency benchmarks. All functions are thread-safe; batch functions use
 * OpenMP when available.
 */
#ifndef LOB_ENGINE_H
#define LOB_ENGINE_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* The library defines LOB_ENGINE_BUILD to export; elsewhere it is built with
 * -fvisibility=hidden, so only LOB_API symbols are public. */
#if defined(_WIN32)
#  if defined(LOB_ENGINE_BUILD)
#    define LOB_API __declspec(dllexport)
#  else
#    define LOB_API __declspec(dllimport)
#  endif
#else
#  define LOB_API __attribute__((visibility("default")))
#endif

/* Row layout (mirrored in lobimb/data.py and lobimb/reference.py). */
#define LOB_DEPTH 10                   /* price levels per side */
#define LOB_ROW_WIDTH (4 * LOB_DEPTH)  /* 40 doubles per row */
#define LOB_ASK_OFFSET (2 * LOB_DEPTH) /* first ask column (20) */

/* Status codes (mirrored in lobimb/engine.py). */
#define LOB_OK 0
#define LOB_ERR_NULL (-1)  /* NULL pointer argument */
#define LOB_ERR_ARG (-2)   /* n < 0, stride too small, bad depth or alpha */

LOB_API const char* lob_version(void);

/* Threads used by batch calls (1 without OpenMP). */
LOB_API int lob_max_threads(void);

/* Set the OpenMP thread count; ignored when n <= 0 or without OpenMP. */
LOB_API void lob_set_num_threads(int n);

/*
 * Level-1 imbalance (Vb - Va) / (Vb + Va) in [-1, 1], from the best bid/ask
 * volumes. Returns 0 for an empty book; NaN volumes propagate. Returns NaN if
 * `row` is NULL (no status code here).
 */
LOB_API double lob_obi_one(const double* row);

/*
 * out[r] = OBI of row r, for r in [0, n).
 * Returns LOB_OK, LOB_ERR_NULL or LOB_ERR_ARG (n < 0 or stride < LOB_ROW_WIDTH).
 */
LOB_API int lob_obi_batch(const double* book, int64_t n, int64_t stride,
                          double* out);

/*
 * Depth-weighted imbalance and micro-price over `depth` levels, with weights
 * w_i = exp(-alpha * i):
 *
 *   imbalance = sum w_i (Vb_i - Va_i) / sum w_i (Vb_i + Va_i)
 *   micro     = sum w_i (Vb_i * Pa_i + Va_i * Pb_i) / sum w_i (Vb_i + Va_i)
 *
 * alpha = 0 weights all levels equally; large alpha tends to level 1. Note the
 * cross-weighting in micro: a large bid queue pulls the fair value towards the
 * ask. The strategy uses micro - mid as its edge (in USDT).
 *
 * Empty book: imbalance = 0, micro = mid. `depth` must be in [1, LOB_DEPTH] and
 * `alpha` finite and >= 0, otherwise LOB_ERR_ARG and outputs are untouched.
 */
LOB_API int lob_wobi_one(const double* row, int depth, double alpha,
                         double* imbalance, double* micro);

/* Batch version of lob_wobi_one; same stride/error conventions as
 * lob_obi_batch. depth/alpha are checked before any row is processed. */
LOB_API int lob_wobi_batch(const double* book, int64_t n, int64_t stride,
                           int depth, double alpha,
                           double* imbalance, double* micro);

#ifdef __cplusplus
}
#endif

#endif /* LOB_ENGINE_H */
