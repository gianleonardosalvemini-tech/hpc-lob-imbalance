/*
 * bench_engine - latency of the C kernels with no Python in the loop, i.e. the
 * floor for the Python-side numbers.
 *
 * Usage: bench_engine [rows] [calls]
 * Prints "key=value" lines, parsed by lobimb.perf.native_bench.
 *   single_*: per-call latency percentiles of lob_*_one (ns, TSC-timed,
 *             includes timer_overhead_p50_ns)
 *   batch_*:  ns/row of lob_*_batch over `rows` rows, best of 5
 */
#include "lob_engine.h"

#include <stdio.h>
#include <stdlib.h>

/* Wall clock, used to calibrate the TSC (and as the timer on non-x86). */
#ifdef _WIN32
#include <windows.h>
static double wall_ns(void) {
    static LARGE_INTEGER freq;
    LARGE_INTEGER t;
    if (!freq.QuadPart) QueryPerformanceFrequency(&freq);
    QueryPerformanceCounter(&t);
    return (double)t.QuadPart * 1e9 / (double)freq.QuadPart;
}
static void sleep_ms(int ms) { Sleep(ms); }
#else
#include <time.h>
static double wall_ns(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec * 1e9 + ts.tv_nsec;
}
static void sleep_ms(int ms) {
    struct timespec ts = {ms / 1000, (ms % 1000) * 1000000L};
    nanosleep(&ts, NULL);
}
#endif

/* The OS clock has ~100 ns resolution on Windows, too coarse for single
 * calls, so use the invariant TSC. Its rate isn't exposed portably, so it is
 * calibrated against the wall clock over a 200 ms sleep. */
#if defined(__x86_64__) || defined(_M_X64)
#ifdef _MSC_VER
#include <intrin.h>
#else
#include <x86intrin.h>
#endif
static double ns_per_tick = 0.0;
static void calibrate(void) {
    const double w0 = wall_ns();
    const unsigned long long c0 = __rdtsc();
    sleep_ms(200);
    ns_per_tick = (wall_ns() - w0) / (double)(__rdtsc() - c0);
}
static double now_ns(void) { return (double)__rdtsc() * ns_per_tick; }
#else
/* Non-x86: no TSC, fall back to the OS clock. */
static void calibrate(void) {}
static double now_ns(void) { return wall_ns(); }
#endif

static int cmp_double(const void* a, const void* b) {
    const double x = *(const double*)a, y = *(const double*)b;
    return (x > y) - (x < y);
}

/*
 * Synthetic book: 0.1 USDT grid around 17000, volumes in [0, 4) BTC from a
 * fixed-seed LCG (Knuth's MMIX constants) so every run uses the same data.
 */
static void fill_book(double* book, int64_t n) {
    uint64_t s = 42;
    for (int64_t r = 0; r < n; r++) {
        double* row = book + r * LOB_ROW_WIDTH;
        for (int i = 0; i < LOB_DEPTH; i++) {
            s = s * 6364136223846793005ULL + 1442695040888963407ULL;
            row[2 * i] = 17000.0 - 0.1 * i;
            row[2 * i + 1] = (double)(s >> 40) / (1 << 22);
            row[LOB_ASK_OFFSET + 2 * i] = 17000.1 + 0.1 * i;
            row[LOB_ASK_OFFSET + 2 * i + 1] = (double)((s >> 16) & 0xFFFFFF) / (1 << 22);
        }
    }
}

/* Nearest-rank percentiles; sorts `lat` in place. The tail (p99.9, max) is
 * mostly OS interrupts and preemption, not the kernel. */
static void report_percentiles(const char* name, double* lat, int64_t n) {
    qsort(lat, (size_t)n, sizeof(double), cmp_double);
    printf("%s_p50_ns=%.1f\n", name, lat[n / 2]);
    printf("%s_p99_ns=%.1f\n", name, lat[(int64_t)(n * 0.99)]);
    printf("%s_p999_ns=%.1f\n", name, lat[(int64_t)(n * 0.999)]);
    printf("%s_max_ns=%.1f\n", name, lat[n - 1]);
}

int main(int argc, char** argv) {
    /* Default rows = size of the real dataset. */
    const int64_t rows = argc > 1 ? atoll(argv[1]) : 3730870;
    const int64_t calls = argc > 2 ? atoll(argv[2]) : 1000000;
    double* book = malloc(sizeof(double) * LOB_ROW_WIDTH * rows);
    double* out1 = malloc(sizeof(double) * rows);
    double* out2 = malloc(sizeof(double) * rows);
    double* lat = malloc(sizeof(double) * calls);
    if (!book || !out1 || !out2 || !lat) return 1;
    fill_book(book, rows);
    calibrate();

    printf("version=%s\nthreads=%d\nrows=%lld\n", lob_version(), lob_max_threads(),
           (long long)rows);

    /* Timer overhead (included in every single_* sample below). */
    for (int64_t i = 0; i < calls; i++) {
        const double t0 = now_ns();
        lat[i] = now_ns() - t0;
    }
    qsort(lat, (size_t)calls, sizeof(double), cmp_double);
    printf("timer_overhead_p50_ns=%.1f\n", lat[calls / 2]);

    /* Cycle through rows like a live feed instead of reusing one hot cache
     * line; `sink` keeps the calls from being optimised away. */
    volatile double sink = 0.0;
    for (int64_t i = 0; i < calls; i++) {
        const double* row = book + (i % rows) * LOB_ROW_WIDTH;
        const double t0 = now_ns();
        sink += lob_obi_one(row);
        lat[i] = now_ns() - t0;
    }
    report_percentiles("single_obi", lat, calls);

    /* Full depth, alpha = 0.5 (strategy default). */
    double imb, micro;
    for (int64_t i = 0; i < calls; i++) {
        const double* row = book + (i % rows) * LOB_ROW_WIDTH;
        const double t0 = now_ns();
        lob_wobi_one(row, LOB_DEPTH, 0.5, &imb, &micro);
        sink += imb;
        lat[i] = now_ns() - t0;
    }
    report_percentiles("single_wobi", lat, calls);

    /* Best of 5, since noise only ever adds time. Thread count comes from
     * OMP_NUM_THREADS (set by perf.native_bench). */
    double best_obi = 1e300, best_wobi = 1e300;
    for (int rep = 0; rep < 5; rep++) {
        double t0 = now_ns();
        lob_obi_batch(book, rows, LOB_ROW_WIDTH, out1);
        double dt = now_ns() - t0;
        if (dt < best_obi) best_obi = dt;
        t0 = now_ns();
        lob_wobi_batch(book, rows, LOB_ROW_WIDTH, LOB_DEPTH, 0.5, out1, out2);
        dt = now_ns() - t0;
        if (dt < best_wobi) best_wobi = dt;
    }
    printf("batch_obi_ns_per_row=%.3f\n", best_obi / rows);
    printf("batch_wobi_ns_per_row=%.3f\n", best_wobi / rows);
    printf("sink=%g\n", sink + out1[rows / 2]);

    free(book); free(out1); free(out2); free(lat);
    return 0;
}
