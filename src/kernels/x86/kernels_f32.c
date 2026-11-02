/*
 * kernels_f32.c — f32 x f32 dot products.
 *
 * Why this file exists: the `.f32` matvec path (reached by every bf16/f16/f32
 * tensor, i.e. the whole BF16 model) used to be a plain Zig `for` loop over
 *   for (rowv, xs) |mv, vv| acc += mv * vv;
 * Zig cannot reorder a floating-point reduction (it must preserve IEEE
 * evaluation order), so that loop compiles to one dependent `mulss`+`addss`
 * chain -- ~1 FLOP per ~2 instructions, no unrolling, no independent
 * accumulators. It is the C side's job to do this the way llama.cpp's
 * ggml_vec_dot_f32 does: N independent vector accumulators, fused
 * multiply-add, horizontal reduce once at the end.
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
/* bf16 x f32: weights stay in their 2-byte form, expanded on the fly.      */
/*                                                                          */
/* Expanding bf16 -> f32 is a 16-bit left shift of the raw bits (no rounding  */
/* needed: bf16 is f32's top 16 bits), so this costs one shift per 8 values   */
/* and saves HALF the memory traffic versus the f32-resident copy that       */
/* `loadLinear` currently builds. Traffic is what decode is bounded by.      */
/* ---------------------------------------------------------------------- */
static float kf32_dot_bf16_avx2(int n, const uint16_t* a, const float* b) {
    int i = 0;
    __m256 v0 = _mm256_setzero_ps();
    __m256 v1 = _mm256_setzero_ps();

    for (; i + 16 <= n; i += 16) {
        /* load 8 bf16, zero-extend each to 32 bits, shift into f32 position */
        __m128i r0 = _mm_loadu_si128((const __m128i*)(a + i));
        __m128i r1 = _mm_loadu_si128((const __m128i*)(a + i + 8));
        __m256i w0 = _mm256_slli_epi32(_mm256_cvtepu16_epi32(r0), 16);
        __m256i w1 = _mm256_slli_epi32(_mm256_cvtepu16_epi32(r1), 16);
        v0 = _mm256_fmadd_ps(_mm256_castsi256_ps(w0), _mm256_loadu_ps(b + i), v0);
        v1 = _mm256_fmadd_ps(_mm256_castsi256_ps(w1), _mm256_loadu_ps(b + i + 8), v1);
    }

    __m256 s = _mm256_add_ps(v0, v1);
    __m128 lo = _mm256_castps256_ps128(s);
    __m128 hi = _mm256_extractf128_ps(s, 1);
    __m128 t = _mm_add_ps(lo, hi);
    t = _mm_hadd_ps(t, t);
    t = _mm_hadd_ps(t, t);
    float acc = _mm_cvtss_f32(t);

    for (; i < n; i++) {
        const uint32_t bits = (uint32_t)a[i] << 16;
        float w;
        __builtin_memcpy(&w, &bits, sizeof(w));
        acc += w * b[i];
    }
    return acc;
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

/// `a` holds `n` bf16 values (2 bytes each), `b` holds `n` f32.
float bf16_dot_f32(int n, const void* a, const void* b) {
    /* kf32_dot_bf16_avx2 is the AVX2 body when compiled with __AVX2__ &&
     * __FMA__, and the plain scalar shift-expand loop otherwise; both are
     * complete implementations, so no runtime branch is needed. */
    return kf32_dot_bf16_avx2(n, (const uint16_t*)a, (const float*)b);
}
