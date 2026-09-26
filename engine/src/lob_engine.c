/*
 * lob_engine.c - see lob_engine.h for the row layout and formulas.
 *
 * The per-row kernels are shared by the *_one and *_batch functions, so both
 * paths give bit-identical results.
 */

/* Before the include, so LOB_API expands to dllexport on Windows. */
#define LOB_ENGINE_BUILD
#include "lob_engine.h"

#include <math.h>
#include <stddef.h>

#ifdef _OPENMP
#include <omp.h>
#endif

#define LOB_VERSION "1.0.0"

/*
 * Waking a thread team costs a few microseconds vs a few ns per row, so small
 * batches stay single-threaded. 16384 rows x 320 B = 5 MB, roughly where the
 * data stops fitting in cache and extra memory bandwidth starts to help.
 */
#define LOB_PARALLEL_MIN_ROWS 16384

const char* lob_version(void) { return LOB_VERSION; }

int lob_max_threads(void) {
#ifdef _OPENMP
    return omp_get_max_threads();
#else
    return 1;
#endif
}

void lob_set_num_threads(int n) {
#ifdef _OPENMP
    if (n > 0) omp_set_num_threads(n);
#else
    (void)n;
#endif
}

static inline double obi_row(const double* row) {
    const double bv = row[1];
    const double av = row[LOB_ASK_OFFSET + 1];
    const double total = bv + av;
    /* `total > 0` is false for NaN too, so propagate NaN explicitly. */
    if (total > 0.0) return (bv - av) / total;
    return isnan(total) ? total : 0.0;
}

/* wb, wa: weighted bid/ask volume; num: micro-price numerator. */
static inline void wobi_row(const double* row, int depth, const double* w,
                            double* imbalance, double* micro) {
    double wb = 0.0, wa = 0.0, num = 0.0;
    for (int i = 0; i < depth; i++) {
        const double bp = row[2 * i];
        const double bv = row[2 * i + 1];
        const double ap = row[LOB_ASK_OFFSET + 2 * i];
        const double av = row[LOB_ASK_OFFSET + 2 * i + 1];
        wb += w[i] * bv;
        wa += w[i] * av;
        num += w[i] * (bv * ap + av * bp);
    }
    const double total = wb + wa;
    if (total > 0.0) {
        *imbalance = (wb - wa) / total;
        *micro = num / total;
    } else {
        /* Empty book or NaN volume: fall back to the mid. */
        *imbalance = isnan(total) ? total : 0.0;
        *micro = 0.5 * (row[0] + row[LOB_ASK_OFFSET]);
    }
}

/* w[i] = exp(-alpha * i), computed once per call as a running product. */
static int fill_weights(int depth, double alpha, double* w) {
    if (depth < 1 || depth > LOB_DEPTH || !isfinite(alpha) || alpha < 0.0)
        return LOB_ERR_ARG;
    const double decay = exp(-alpha);
    w[0] = 1.0;
    for (int i = 1; i < depth; i++) w[i] = w[i - 1] * decay;
    return LOB_OK;
}

double lob_obi_one(const double* row) {
    return row ? obi_row(row) : NAN;
}

int lob_obi_batch(const double* book, int64_t n, int64_t stride, double* out) {
    if (!book || !out) return LOB_ERR_NULL;
    if (n < 0 || stride < LOB_ROW_WIDTH) return LOB_ERR_ARG;

    /* Every row costs the same, so a static schedule is enough. Only 2
     * doubles are read per 320-byte row: this is memory-bandwidth bound. */
    #pragma omp parallel for simd schedule(static) if (n >= LOB_PARALLEL_MIN_ROWS)
    for (int64_t r = 0; r < n; r++) {
        out[r] = obi_row(book + r * stride);
    }
    return LOB_OK;
}

int lob_wobi_one(const double* row, int depth, double alpha,
                 double* imbalance, double* micro) {
    if (!row || !imbalance || !micro) return LOB_ERR_NULL;
    double w[LOB_DEPTH];
    const int rc = fill_weights(depth, alpha, w);
    if (rc != LOB_OK) return rc;
    wobi_row(row, depth, w, imbalance, micro);
    return LOB_OK;
}

int lob_wobi_batch(const double* book, int64_t n, int64_t stride,
                   int depth, double alpha,
                   double* imbalance, double* micro) {
    if (!book || !imbalance || !micro) return LOB_ERR_NULL;
    if (n < 0 || stride < LOB_ROW_WIDTH) return LOB_ERR_ARG;
    double w[LOB_DEPTH];
    const int rc = fill_weights(depth, alpha, w);
    if (rc != LOB_OK) return rc;

    /* Each row is summed by a single thread, so results do not depend on the
     * thread count (checked bit-for-bit in tests/test_engine.py). */
    #pragma omp parallel for schedule(static) if (n >= LOB_PARALLEL_MIN_ROWS)
    for (int64_t r = 0; r < n; r++) {
        wobi_row(book + r * stride, depth, w, imbalance + r, micro + r);
    }
    return LOB_OK;
}
