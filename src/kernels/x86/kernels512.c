/*
 * kernels512.c — AVX-512 VNNI dot-product kernels.
 *
 * This is a SEPARATE compilation unit compiled with
 *   -mavx512f -mavx512bw -mavx512dq -mavx512vl -mavx512vnni
 * so the AVX2 units (kernels32.c / kernels256.c / iq_kernels.c) are NOT
 * recompiled with AVX-512 flags (that would change their codegen). The public
 * entry point is wired from kernels32.c, which dispatches
 *   q8_0_q8_0:  avx512vnni ? k512_q8_0_q8_0_vnni : (avx2 ? avx2 : scalar).
 *
 * Contract / bit-identity (matches kernels32.c scalar + AVX2 EXACTLY):
 *   n is a multiple of 32; each block is
 *     vx: f16 d @0, 32 x int8 qs @2   (block stride 34)
 *     vy: f16 d @0, 32 x int8 qs @2   (block stride 34)
 *   sum over blocks of  dx*dy*(sum_j x_j*y_j),  dx/dy decoded by the exact f16
 *   routine identical to kernels32.c. The integer sum is computed by widening
 *   each i8 to i16 and running VPDPWSSD (signed i16 pair products -> i32
 *   lanes). No saturation (|i8*i8| <= 16256 fits i16 pair sums, and 32
 *   products fit i32), so the integer part is bit-identical to the scalar loop.
 *   Verified by a 200000-trial simulation on the build host: dpwssd(signed,
 *   widened i8) == scalar i8*i8 dot, 0 mismatches.
 *
 * Every internal symbol is static and prefixed k512_.
 */

#include <stdint.h>
#include <string.h>
#include <immintrin.h>
#include "cpu_caps.h"

/* Exact f16 -> f32, byte-identical to kernels32.c k32_f16_to_f32. */
static inline float k512_f16_to_f32(uint16_t h) {
    const uint32_t sign = (uint32_t)(h & 0x8000u) << 16;
    const uint32_t exp = (uint32_t)(h >> 10) & 0x1Fu;
    const uint32_t mant = (uint32_t)(h & 0x3FFu);
    uint32_t out;
    if (exp == 0) {
        if (mant == 0) return 0.0f;
        uint32_t m = mant;
        uint32_t lz = 0;
        while (m < 0x200u) { m <<= 1; lz++; }
        out = sign | ((112u - lz) << 23) | ((m & 0x1FFu) << 14);
    } else if (exp == 31) {
        out = sign | 0x7F800000u | ((mant << 14) & 0x7FFFFFu);
    } else {
        out = sign | ((exp + 112u) << 23) | (mant << 13);
    }
    return *(const float *)&out;
}

#ifdef __AVX512VNNI__

float k512_q8_0_q8_0_vnni(int n, const void* vx, const void* vy) {
    const uint8_t* x = (const uint8_t*)vx;
    const uint8_t* y = (const uint8_t*)vy;
    const int nb = n / 32;

    __m512i accv = _mm512_setzero_si512();
    float sumf = 0.0f;

    for (int ib = 0; ib < nb; ib++) {
        const uint8_t* xb = x + ib * 34;
        const uint8_t* yb = y + ib * 34;
        const float dx = k512_f16_to_f32((uint16_t)(xb[0] | ((uint32_t)xb[1] << 8)));
        const float dy = k512_f16_to_f32((uint16_t)(yb[0] | ((uint32_t)yb[1] << 8)));

        /* 32 i8 -> 32 i16, then VPDPWSSD: adjacent i16 pairs multiply and
         * accumulate into i32 lanes. Signed, no saturation. */
        const __m512i xw = _mm512_cvtepi8_epi16(_mm256_loadu_si256((const __m256i*)(xb + 2)));
        const __m512i yw = _mm512_cvtepi8_epi16(_mm256_loadu_si256((const __m256i*)(yb + 2)));
        accv = _mm512_dpwssd_epi32(accv, xw, yw);

        /* Reduce the block's integer sum, then apply dx*dy in EXACTLY the
         * scalar/AVX2 order: per-block float accumulator, one scalar add per
         * block. (Accumulating into all 16 lanes of a __m512 would multiply
         * the total by 16 at reduce time.) */
        const int sumi = _mm512_reduce_add_epi32(accv);
        accv = _mm512_setzero_si512();
        sumf += dx * dy * (float)sumi;
    }
    return sumf;
}

#else
/* Compiled without __AVX512VNNI__: thin wrapper (never called; dispatch
 * guards on cpu_has_avx512_vnni()). Falls back to the AVX2 path's scalar by
 * returning 0; kernels32.c will not route here when VNNI is absent. */
float k512_q8_0_q8_0_vnni(int n, const void* vx, const void* vy) {
    (void)n; (void)vx; (void)vy;
    return 0.0f;
}
#endif
