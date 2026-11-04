/*
 * kernels_f32.c — f32 x f32, bf16 x f32 and f16 x f32 dot products.
 *
 * Why this file exists: the matvec for unquantized weights used to be a plain
 * Zig `for` loop over
 *   for (rowv, xs) |mv, vv| acc += mv * vv;
 * Zig cannot reorder a floating-point reduction (it must preserve IEEE
 * evaluation order), so that loop compiles to one dependent `mulss`+`addss`
 * chain -- ~1 FLOP per ~2 instructions, no unrolling, no independent
 * accumulators. It is the C side's job to do this the way llama.cpp's
 * ggml_vec_dot_f32 does: N independent vector accumulators, fused
 * multiply-add, horizontal reduce once at the end.
 *
 * Who reaches which entry point: `.f32` tensors go to `f32_dot_f32`.  BF16 and
 * F16 weights stay packed at 2 B/elem (no f32 copy exists) and go to
 * `bf16_dot_f32` / `f16_dot_f32`, which widen element-by-element inside the
 * kernel -- see the `KF32_DOT_LOOP` block below.
 *
 * Contract (must match the Zig scalar reference bit-for-bit in the sense that
 * matters -- see the test in src/backend_cpu.zig "matvecF32 ..."):
 *   out[r] = sum_c a[r*cols + c] * b[c]
 * The reduction ORDER differs from a strictly sequential scalar sum (we sum
 * 8 lanes independently then combine), so results agree to ~1 ulp per term
 * rather than exactly. That is the same trade llama.cpp makes, and the f32
 * path is only used for unquantized weights where there is no integer-exact
 * reference to preserve.
 *
 * Compiled with AVX2+FMA (see build.zig KernelIsa.avx2). The scalar branch is
 * a complete, correct implementation -- never a placeholder (AGENTS.md 1.1).
 */

#include <stdint.h>
#include <stddef.h>
#include <immintrin.h>
#include "cpu_caps.h"

/* ---------------------------------------------------------------------- */
/* scalar reference: the exact sequential loop the Zig code used to run     */
/* ---------------------------------------------------------------------- */
static float kf32_dot_scalar(int n, const float* a, const float* b) {
    float acc = 0.0f;
    for (int i = 0; i < n; i++) acc += a[i] * b[i];
    return acc;
}

#if defined(__AVX2__) && defined(__FMA__)

/* 4 independent 8-wide accumulators = 32 floats in flight. Four accumulators
 * is what hides the FMA latency (4-5 cycles) at one FMA/cycle port pressure;
 * more than that just burns registers. */
static float kf32_dot_avx2(int n, const float* a, const float* b) {
    int i = 0;
    __m256 v0 = _mm256_setzero_ps();
    __m256 v1 = _mm256_setzero_ps();
    __m256 v2 = _mm256_setzero_ps();
    __m256 v3 = _mm256_setzero_ps();

    for (; i + 32 <= n; i += 32) {
        v0 = _mm256_fmadd_ps(_mm256_loadu_ps(a + i + 0), _mm256_loadu_ps(b + i + 0), v0);
        v1 = _mm256_fmadd_ps(_mm256_loadu_ps(a + i + 8), _mm256_loadu_ps(b + i + 8), v1);
        v2 = _mm256_fmadd_ps(_mm256_loadu_ps(a + i + 16), _mm256_loadu_ps(b + i + 16), v2);
        v3 = _mm256_fmadd_ps(_mm256_loadu_ps(a + i + 24), _mm256_loadu_ps(b + i + 24), v3);
    }
    for (; i + 8 <= n; i += 8) {
        v0 = _mm256_fmadd_ps(_mm256_loadu_ps(a + i), _mm256_loadu_ps(b + i), v0);
    }

    /* Reduce the 4 vector accumulators, then the 8 lanes, then the tail. */
    __m256 s = _mm256_add_ps(_mm256_add_ps(v0, v1), _mm256_add_ps(v2, v3));
    __m128 lo = _mm256_castps256_ps128(s);
    __m128 hi = _mm256_extractf128_ps(s, 1);
    __m128 t = _mm_add_ps(lo, hi);
    t = _mm_hadd_ps(t, t);
    t = _mm_hadd_ps(t, t);
    float acc = _mm_cvtss_f32(t);

    for (; i < n; i++) acc += a[i] * b[i];
    return acc;
}

/* ---------------------------------------------------------------------- */
/* bf16 x f32 / f16 x f32: weights stay in their 2-byte form, expanded on    */
/* the fly.                                                                  */
/*                                                                           */
/* Expanding bf16 -> f32 is a 16-bit left shift of the raw bits (no rounding  */
/* needed: bf16 is f32's top 16 bits), so this costs one shift per 8 values   */
/* and saves HALF the memory traffic versus the f32-resident copy that        */
/* `loadLinear` used to build. Traffic is what decode is bounded by.          */
/* f16 uses vcvtph2ps (F16C), which is also exact.                           */
/*                                                                           */
/* The accumulation structure is deliberately IDENTICAL to `kf32_dot_avx2`    */
/* (four 8-wide accumulators, same chunk order, same reduce, same tail).      */
/* The values being summed are exactly the f32 values that the materialized   */
/* copy held, so a bf16 row's dot here is BIT-IDENTICAL to the f32 row's dot  */
/* there -- the BF16 model's output stays the same as the f32-materialization  */
/* baseline, token for token. A different accumulator layout (the 2-accumulator */
/* version this function used to have) agrees only to ~1 ulp per term, which   */
/* is enough to flip an argmax in a near-tie and diverge the whole generation. */
/* ---------------------------------------------------------------------- */

#define KF32_BF16_X8(r) _mm256_castsi256_ps(_mm256_slli_epi32(_mm256_cvtepu16_epi32(_mm_loadu_si128((const __m128i*)(r))), 16))
#define KF32_F16_X8(r)  _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*)(r)))

#define KF32_DOT_LOOP(CVT, W)                       \
    int i = 0;                                    \
    __m256 v0 = _mm256_setzero_ps();              \
    __m256 v1 = _mm256_setzero_ps();              \
    __m256 v2 = _mm256_setzero_ps();              \
    __m256 v3 = _mm256_setzero_ps();              \
    for (; i + 32 <= n; i += 32) {                \
        v0 = _mm256_fmadd_ps(CVT(a + i + 0),  _mm256_loadu_ps(b + i + 0),  v0); \
        v1 = _mm256_fmadd_ps(CVT(a + i + 8),  _mm256_loadu_ps(b + i + 8),  v1); \
        v2 = _mm256_fmadd_ps(CVT(a + i + 16), _mm256_loadu_ps(b + i + 16), v2); \
        v3 = _mm256_fmadd_ps(CVT(a + i + 24), _mm256_loadu_ps(b + i + 24), v3); \
    }                                             \
    for (; i + 8 <= n; i += 8) {                  \
        v0 = _mm256_fmadd_ps(CVT(a + i), _mm256_loadu_ps(b + i), v0); \
    }                                             \
    __m256 s = _mm256_add_ps(_mm256_add_ps(v0, v1), _mm256_add_ps(v2, v3)); \
    __m128 lo = _mm256_castps256_ps128(s);        \
    __m128 hi = _mm256_extractf128_ps(s, 1);      \
    __m128 t = _mm_add_ps(lo, hi);                \
    t = _mm_hadd_ps(t, t);                        \
    t = _mm_hadd_ps(t, t);                        \
    float acc = _mm_cvtss_f32(t);                 \
    for (; i < n; i++) acc += W(a[i]) * b[i]; \
    return acc;

static inline float kf32_bf16_w(uint16_t h) {
    const uint32_t bits = (uint32_t)h << 16;
    float w;
    __builtin_memcpy(&w, &bits, sizeof(w));
    return w;
}

static inline float kf32_f16_w(uint16_t h) {
    /* BIT reinterpretation: `h` is the f16 encoding, not the integer value `h`.
     * Reinterpret the bits as _Float16, then widen.  This is exactly what
     * `vcvtph2ps` does in the AVX2 body, and exactly `k32_f16_to_f32` in
     * kernels32.c.  NOTE: `(float)((_Float16)h)` is NOT this -- C converts the
     * *integer* h into f16, so 0x2800 becomes 10240.0 instead of 0.03125.  The
     * tail loop only runs on n%8 lanes, so the damage is ~1/8 of the row's
     * weight values: the row sum is far from the truth, and `simd f32: bf16/f16
     * packed dot == ... f32-materialized dot` in src/simd_tests_f32.zig fails
     * (measured: relative error 0.987 at n = 33 and n = 100).  The bf16 tail
     * above (`kf32_bf16_w`) already uses this memcpy form; f16 must match it. */
    _Float16 f;
    __builtin_memcpy(&f, &h, sizeof(f));
    return (float)f;
}

static float kf32_dot_bf16_avx2(int n, const uint16_t* a, const float* b) {
    KF32_DOT_LOOP(KF32_BF16_X8, kf32_bf16_w)
}

static float kf32_dot_f16_avx2(int n, const uint16_t* a, const float* b) {
    KF32_DOT_LOOP(KF32_F16_X8, kf32_f16_w)
}

#else /* !(__AVX2__ && __FMA__) */

static float kf32_dot_avx2(int n, const float* a, const float* b) {
    return kf32_dot_scalar(n, a, b);
}

static float kf32_dot_bf16_avx2(int n, const uint16_t* a, const float* b) {
    /* correct scalar bf16->f32: bf16 is the high half of the f32 bit pattern */
    float acc = 0.0f;
    for (int i = 0; i < n; i++) {
        const uint32_t bits = (uint32_t)a[i] << 16;
        float w;
        __builtin_memcpy(&w, &bits, sizeof(w));
        acc += w * b[i];
    }
    return acc;
}

static float kf32_dot_f16_avx2(int n, const uint16_t* a, const float* b) {
    /* complete scalar f16->f32: same values as vcvtph2ps */
    float acc = 0.0f;
    for (int i = 0; i < n; i++) acc += kf32_f16_w(a[i]) * b[i];
    return acc;
}

#endif /* __AVX2__ && __FMA__ */

/* ---------------------------------------------------------------------- */
/* public entry points                                                      */
/* ---------------------------------------------------------------------- */

float f32_dot_f32(int n, const void* a, const void* b) {
    const float* pa = (const float*)a;
    const float* pb = (const float*)b;
    if (cpu_has_avx2() && cpu_has_fma()) return kf32_dot_avx2(n, pa, pb);
    return kf32_dot_scalar(n, pa, pb);
}

/// `a` holds `n` f16 values (2 bytes each), `b` holds `n` f32.
float f16_dot_f32(int n, const void* a, const void* b) {
    return kf32_dot_f16_avx2(n, (const uint16_t*)a, (const float*)b);
}

/// `a` holds `n` bf16 values (2 bytes each), `b` holds `n` f32.
float bf16_dot_f32(int n, const void* a, const void* b) {
    /* kf32_dot_bf16_avx2 is the AVX2 body when compiled with __AVX2__ &&
     * __FMA__, and the plain scalar shift-expand loop otherwise; both are
     * complete implementations, so no runtime branch is needed. */
    return kf32_dot_bf16_avx2(n, (const uint16_t*)a, (const float*)b);
}
