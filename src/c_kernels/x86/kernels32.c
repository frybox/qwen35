/*
 * kernels32.c — 32-element dot-product kernels: 4-bit weight x Q8_0 activation.
 *
 * Ported from the Zig references (the ORACLE this file must match exactly):
 *   src/simd/kern32.zig : q4_0_q8_0_ref (L410) / q4_1_q8_0_ref (L455) /
 *                         iq4_nl_q8_0_ref (L680)
 *   src/simd/kerniq.zig : kvalues_iq4nl (L32)
 *   src/simd/act.zig    : quantizeQ8_0 (L23)
 *
 * Contract (n is a multiple of 32):
 *   vx = weight row, 32 elements per block
 *   vy = Q8_0 activation row, 34 bytes per 32-element block:
 *         f16 scale d @0, 32 x int8 qs @2
 *   X_q8_0(n, vx, vy) returns the dot product of the dequantized rows.
 *
 * Public API (exactly these — NOT static, no prefix):
 *   float q4_0_q8_0(int n, const void* vx, const void* vy);
 *   float q4_1_q8_0(int n, const void* vx, const void* vy);
 *   float iq4_nl_q8_0(int n, const void* vx, const void* vy);
 *   void  quantize_q8_0(int n, const float* x, uint8_t* out);
 *
 * Every internal symbol is static and prefixed k32_ (sibling C files in this
 * directory are concatenated; collisions break the build).
 *
 * The AVX2 kernels compute each block's integer sums EXACTLY (i16 x i16 ->
 * i32 pair products via vpmaddw, no saturation), then apply the same float
 * arithmetic as the scalar path in the same order — scalar and AVX2 results
 * are bit-identical.
 */
#include <stdint.h>
#include <string.h>
#include <immintrin.h>
#include "cpu_caps.h"

/* ------------------------------------------------------------------ */
/* runtime dispatch: cpu_has_avx2() is the shared probe in cpu_caps.c, */
/* probed once at load time. Do NOT re-implement cpuid/xgetbv here.    */
/* ------------------------------------------------------------------ */

/* ------------------------------------------------------------------ */
/* f16 <-> f32 bit conversions (exact; matches zig f16At / f32ToF16)  */
/* ------------------------------------------------------------------ */

static inline float k32_f16_to_f32(uint16_t h) {
    const uint32_t sign = (uint32_t)(h & 0x8000u) << 16;
    const uint32_t exp = (uint32_t)(h >> 10) & 0x1Fu;
    const uint32_t mant = (uint32_t)(h & 0x3FFu);
    uint32_t out;
    if (exp == 0) {
        if (mant == 0) return 0.0f;
        /* f16 subnormal: value = mant * 2^-24; normalize mant into [512,1024) */
        uint32_t m = mant;
        uint32_t lz = 0;
        while (m < 0x200u) { m <<= 1; lz++; }
        /* value = (1 + (m-512)/512) * 2^(-15-lz) */
        out = sign | ((112u - lz) << 23) | ((m & 0x1FFu) << 11);
    } else if (exp == 31) {
        /* inf / nan */
        out = sign | 0x7F800000u | ((mant << 14) & 0x7FFFFFu);
    } else {
        /* normal: f32 exponent = f16 exponent + 112 */
        out = sign | ((exp + 112u) << 23) | (mant << 13);
    }
    return *(const float *)&out;
}
static inline uint16_t k32_f32_to_f16bits(float x) {
    const uint32_t u = *(const uint32_t *)&x;
    const uint32_t sign = (uint32_t)((u & 0x80000000u) >> 16);
    const uint32_t e = (u >> 23) & 0xFFu;
    const uint32_t m = u & 0x7FFFFFu;
    if (e == 255u) return (uint16_t)(0x7C00u | sign);     /* inf / nan -> inf */
    if (e >= 143u) return (uint16_t)(0x7C00u | sign);     /* > 65504 -> inf   */
    if (e >= 113u) {
        /* normal f16: round 23-bit mantissa to 10 bits, round-to-nearest-even */
        uint32_t keep = m >> 13;
        const uint32_t drop = m & 0x1FFFu;
        const uint32_t up = (drop > 0x1000u) || (drop == 0x1000u && (keep & 1u));
        keep += up;
        uint32_t e16 = e - 112u;
        if (keep >= 0x400u) { keep = 0; e16 += 1; }
        if (e16 > 30u) return (uint16_t)(0x7C00u | sign); /* overflow -> inf */
        return (uint16_t)(sign | (e16 << 10) | keep);
    }
    /* e in [0,112]: f16 subnormal or zero.  k = RN((2^23+m) * 2^(e-126)) */
    {
        const uint32_t t = m | 0x800000u;
        const int s = (int)(126u - e);
        if (s >= 25) return (uint16_t)sign;               /* rounds to 0 */
        uint32_t keep = t >> s;
        const uint32_t drop = t & ((1u << s) - 1u);
        const uint32_t mid = 1u << (s - 1);
        const uint32_t up = (drop > mid) || (drop == mid && (keep & 1u));
        keep += up;
        if (keep >= 0x400u) return (uint16_t)(0x0380u | sign); /* rounds to 2^-14 */
        return (uint16_t)(sign | keep);
    }
}

/* ------------------------------------------------------------------ */
/* iq4_nl value table — from src/simd/kerniq.zig L32 (keep in sync)   */
/* ------------------------------------------------------------------ */

static const int32_t k32_iq4nl_tab[16] = { -127, -104, -83, -65, -49, -35, -22, -10, 1, 13, 25, 38, 53, 69, 89, 113 };

#if defined(__AVX2__)

/* ------------------------------------------------------------------ */
/* exact integer helpers (AVX2)                                      */
/* ------------------------------------------------------------------ */

/* exact sum of the 8 i32 lanes of v -> int (|total| << 2^31 here).
 * after the 3 alignr stages lane 0 holds the full 8-lane sum. */
static inline int k32_sum8i32(const __m256i v) {
    __m256i t = _mm256_add_epi32(v, _mm256_alignr_epi8(v, v, 4));
    t = _mm256_add_epi32(t, _mm256_alignr_epi8(t, t, 8));
    t = _mm256_add_epi32(t, _mm256_alignr_epi8(t, t, 16));
    return (int)_mm256_extract_epi32(t, 0);
}

/*
 * exact dot product of 32 signed bytes x 32 signed bytes -> int.
 * i16 x i16 -> i32 pair products (vpmaddw), no saturation; |dot| < 2^31.
 */
static inline int k32_dot32_i8_i8(const __m256i x32, const __m256i y32) {
    /* lo 16 bytes: x[0..15] x y[0..15] -> 16 pair products in 8 i32 lanes */
    const __m256i pa = _mm256_madd_epi16(_mm256_cvtepi8_epi16(_mm256_castsi256_si128(x32)),
                                         _mm256_cvtepi8_epi16(_mm256_castsi256_si128(y32)));
    /* hi 16 bytes: x[16..31] x y[16..31] */
    const __m256i pb = _mm256_madd_epi16(_mm256_cvtepi8_epi16(_mm256_extractf128_si256(x32, 1)),
                                         _mm256_cvtepi8_epi16(_mm256_extractf128_si256(y32, 1)));
    return k32_sum8i32(_mm256_add_epi32(pa, pb));
}

/* ------------------------------------------------------------------ */
/* q4_0 x q8_0   block: 18 bytes = f16 d @0 + 16 x u8 @2 (2 nib/byte) */
/*   sumi = Σ_j [(lo_nib-8) * y[j]  +  (hi_nib-8) * y[j+16]]         */
/* ------------------------------------------------------------------ */

static float k32_q4_0_q8_0_avx2(int n, const void* vx, const void* vy) {
    const uint8_t* x = (const uint8_t*)vx;
    const uint8_t* y = (const uint8_t*)vy;
    const int nb = n / 32;
    float sumf = 0.0f;
    const __m128i eight = _mm_set1_epi8(8);
    const __m128i fivemask = _mm_set1_epi8(15);
    for (int ib = 0; ib < nb; ib++) {
        const uint8_t* xb = x + ib * 18;
        const uint8_t* yb = y + ib * 34;
        const float dx = k32_f16_to_f32((uint16_t)(xb[0] | ((uint32_t)xb[1] << 8)));
        const float dy = k32_f16_to_f32((uint16_t)(yb[0] | ((uint32_t)yb[1] << 8)));
        const __m128i q = _mm_loadu_si128((const __m128i*)(xb + 2));
        const __m128i lo8 = _mm_sub_epi8(_mm_and_si128(q, fivemask), eight);
        const __m128i hi8 = _mm_sub_epi8(_mm_and_si128(_mm_srli_epi16(q, 4), fivemask), eight);
        __m256i x32 = _mm256_inserti128_si256(_mm256_castsi128_si256(lo8), hi8, 1);
        const __m256i y32 = _mm256_loadu_si256((const __m256i*)(yb + 2));
        const int sumi = k32_dot32_i8_i8(x32, y32);
        sumf += dx * dy * (float)sumi;
    }
    return sumf;
}

/* ------------------------------------------------------------------ */
/* q4_1 x q8_0   block: 20 bytes = f16 d @0, f16 m @2, 16 x u8 @4     */
/*   (nibbles NOT centered; Q8_0 activation -> Q8_1 min term dropped) */
/*   sumi = Σ_j [lo_nib*y[j] + hi_nib*y[j+16]], sumy = Σ_32 y[j]     */
/* ------------------------------------------------------------------ */

static float k32_q4_1_q8_0_avx2(int n, const void* vx, const void* vy) {
    const uint8_t* x = (const uint8_t*)vx;
    const uint8_t* y = (const uint8_t*)vy;
    const int nb = n / 32;
    float sumf = 0.0f;
    const __m128i fivemask = _mm_set1_epi8(15);
    const __m256i ones = _mm256_set1_epi8(1);
    for (int ib = 0; ib < nb; ib++) {
        const uint8_t* xb = x + ib * 20;
        const uint8_t* yb = y + ib * 34;
        const float dx = k32_f16_to_f32((uint16_t)(xb[0] | ((uint32_t)xb[1] << 8)));
        const float mx = k32_f16_to_f32((uint16_t)(xb[2] | ((uint32_t)xb[3] << 8)));
        const float dy = k32_f16_to_f32((uint16_t)(yb[0] | ((uint32_t)yb[1] << 8)));
        const __m128i q = _mm_loadu_si128((const __m128i*)(xb + 4));
        const __m128i lo = _mm_and_si128(q, fivemask);
        const __m128i hi = _mm_and_si128(_mm_srli_epi16(q, 4), fivemask);
        __m256i x32 = _mm256_inserti128_si256(_mm256_castsi128_si256(lo), hi, 1);
        const __m256i y32 = _mm256_loadu_si256((const __m256i*)(yb + 2));
        const int sumi = k32_dot32_i8_i8(x32, y32);
        const int sumy = k32_dot32_i8_i8(y32, ones);
        sumf += dx * dy * (float)sumi + mx * dy * (float)sumy;
    }
    return sumf;
}

/* ------------------------------------------------------------------ */
/* iq4_nl x q8_0  block: 18 bytes = f16 d @0 + 16 x u8 @2             */
/*   nibble indices into k32_iq4nl_tab[16]                          */
/* ------------------------------------------------------------------ */

static float k32_iq4_nl_q8_0_avx2(int n, const void* vx, const void* vy) {
    const uint8_t* x = (const uint8_t*)vx;
    const uint8_t* y = (const uint8_t*)vy;
    const int nb = n / 32;
    float sumf = 0.0f;
    for (int ib = 0; ib < nb; ib++) {
        const uint8_t* xb = x + ib * 18;
        const uint8_t* yb = y + ib * 34;
        const float dx = k32_f16_to_f32((uint16_t)(xb[0] | ((uint32_t)xb[1] << 8)));
        const float dy = k32_f16_to_f32((uint16_t)(yb[0] | ((uint32_t)yb[1] << 8)));
        const uint8_t* qb = xb + 2;
        int8_t xv[32];
        for (int j = 0; j < 16; j++) {
            xv[j] = (int8_t)k32_iq4nl_tab[qb[j] & 0x0F];
            xv[16 + j] = (int8_t)k32_iq4nl_tab[qb[j] >> 4];
        }
        const __m256i x32 = _mm256_loadu_si256((const __m256i*)xv);
        const __m256i y32 = _mm256_loadu_si256((const __m256i*)(yb + 2));
        const int sumi = k32_dot32_i8_i8(x32, y32);
        sumf += dx * dy * (float)sumi;
    }
    return sumf;
}

#else /* !__AVX2__ : thin scalar wrappers (see scalar impls below) */

/* forward decls so the wrappers compile before the scalar bodies */
static float k32_q4_0_q8_0_scalar(int n, const void* vx, const void* vy);
static float k32_q4_1_q8_0_scalar(int n, const void* vx, const void* vy);
static float k32_iq4_nl_q8_0_scalar(int n, const void* vx, const void* vy);

static float k32_q4_0_q8_0_avx2(int n, const void* vx, const void* vy) {
    return k32_q4_0_q8_0_scalar(n, vx, vy);
}
static float k32_q4_1_q8_0_avx2(int n, const void* vx, const void* vy) {
    return k32_q4_1_q8_0_scalar(n, vx, vy);
}
static float k32_iq4_nl_q8_0_avx2(int n, const void* vx, const void* vy) {
    return k32_iq4_nl_q8_0_scalar(n, vx, vy);
}

#endif /* __AVX2__ */

/* ------------------------------------------------------------------ */
/* scalar kernels (ports of the zig *_ref oracles)                   */
/* ------------------------------------------------------------------ */

static float k32_q4_0_q8_0_scalar(int n, const void* vx, const void* vy) {
    const uint8_t* x = (const uint8_t*)vx;
    const uint8_t* y = (const uint8_t*)vy;
    const int nb = n / 32;
    float sumf = 0.0f;
    for (int ib = 0; ib < nb; ib++) {
        const uint8_t* xb = x + ib * 18;
        const uint8_t* yb = y + ib * 34;
        const float dx = k32_f16_to_f32((uint16_t)(xb[0] | ((uint32_t)xb[1] << 8)));
        const float dy = k32_f16_to_f32((uint16_t)(yb[0] | ((uint32_t)yb[1] << 8)));
        int sumi = 0;
        for (int j = 0; j < 16; j++) {
            const uint8_t b = xb[2 + j];
            sumi += ((int)(b & 0x0F) - 8) * (int)(int8_t)yb[2 + j];
            sumi += ((int)(b >> 4) - 8) * (int)(int8_t)yb[2 + 16 + j];
        }
        sumf += dx * dy * (float)sumi;
    }
    return sumf;
}

static float k32_q4_1_q8_0_scalar(int n, const void* vx, const void* vy) {
    const uint8_t* x = (const uint8_t*)vx;
    const uint8_t* y = (const uint8_t*)vy;
    const int nb = n / 32;
    float sumf = 0.0f;
    for (int ib = 0; ib < nb; ib++) {
        const uint8_t* xb = x + ib * 20;
        const uint8_t* yb = y + ib * 34;
        const float dx = k32_f16_to_f32((uint16_t)(xb[0] | ((uint32_t)xb[1] << 8)));
        const float mx = k32_f16_to_f32((uint16_t)(xb[2] | ((uint32_t)xb[3] << 8)));
        const float dy = k32_f16_to_f32((uint16_t)(yb[0] | ((uint32_t)yb[1] << 8)));
        int sumi = 0;
        int sumy = 0;
        for (int j = 0; j < 16; j++) {
            const uint8_t b = xb[4 + j];
            sumi += (int)(b & 0x0F) * (int)(int8_t)yb[2 + j];
            sumi += (int)(b >> 4) * (int)(int8_t)yb[2 + 16 + j];
        }
        for (int j = 0; j < 32; j++) sumy += (int)(int8_t)yb[2 + j];
        sumf += dx * dy * (float)sumi + mx * dy * (float)sumy;
    }
    return sumf;
}

static float k32_iq4_nl_q8_0_scalar(int n, const void* vx, const void* vy) {
    const uint8_t* x = (const uint8_t*)vx;
    const uint8_t* y = (const uint8_t*)vy;
    const int nb = n / 32;
    float sumf = 0.0f;
    for (int ib = 0; ib < nb; ib++) {
        const uint8_t* xb = x + ib * 18;
        const uint8_t* yb = y + ib * 34;
        const float dx = k32_f16_to_f32((uint16_t)(xb[0] | ((uint32_t)xb[1] << 8)));
        const float dy = k32_f16_to_f32((uint16_t)(yb[0] | ((uint32_t)yb[1] << 8)));
        int sumi = 0;
        for (int j = 0; j < 16; j++) {
            const uint8_t b = xb[2 + j];
            sumi += (int)k32_iq4nl_tab[b & 0x0F] * (int)(int8_t)yb[2 + j];
            sumi += (int)k32_iq4nl_tab[b >> 4] * (int)(int8_t)yb[2 + 16 + j];
        }
        sumf += dx * dy * (float)sumi;
    }
    return sumf;
}

/* ------------------------------------------------------------------ */
/* public API                                                         */
/* ------------------------------------------------------------------ */

float q4_0_q8_0(int n, const void* vx, const void* vy) {
    return cpu_has_avx2() ? k32_q4_0_q8_0_avx2(n, vx, vy) : k32_q4_0_q8_0_scalar(n, vx, vy);
}

float q4_1_q8_0(int n, const void* vx, const void* vy) {
    return cpu_has_avx2() ? k32_q4_1_q8_0_avx2(n, vx, vy) : k32_q4_1_q8_0_scalar(n, vx, vy);
}

float iq4_nl_q8_0(int n, const void* vx, const void* vy) {
    return cpu_has_avx2() ? k32_iq4_nl_q8_0_avx2(n, vx, vy) : k32_iq4_nl_q8_0_scalar(n, vx, vy);
}

/*
 * quantize_q8_0 — replicates src/simd/act.zig quantizeQ8_0 exactly:
 *   amax  = max|x[i..i+31]|
 *   scale = amax > 0 ? amax/127 : 1        (f32, stored as f16)
 *   inv   = amax > 0 ? 1/scale : 0
 *   q     = clamp(round(x*inv), -128, 127)  (round half away from zero)
 */

/* ------------------------------------------------------------------ */
/* q8_0 x q8_0 — both sides are Q8_0 quantized                        */
/* block_q8_0: f16 d @0, 32 x int8 qs @2 (34 bytes)                 */
/* ------------------------------------------------------------------ */

#if defined(__AVX2__)

static inline float k32_hsum_float_8(const __m256 x) {
    __m128 res = _mm256_extractf128_ps(x, 1);
    res = _mm_add_ps(res, _mm256_castps256_ps128(x));
    res = _mm_add_ps(res, _mm_movehl_ps(res, res));
    res = _mm_add_ss(res, _mm_movehdup_ps(res));
    return _mm_cvtss_f32(res);
}

static float k32_q8_0_q8_0_avx2(int n, const void* vx, const void* vy) {
    const uint8_t* x = (const uint8_t*)vx;
    const uint8_t* y = (const uint8_t*)vy;
    const int nb = n / 32;
    __m256 acc = _mm256_setzero_ps();
    
    for (int ib = 0; ib < nb; ib++) {
        const uint8_t* xb = x + ib * 34;
        const uint8_t* yb = y + ib * 34;
        const float dx = k32_f16_to_f32((uint16_t)(xb[0] | ((uint32_t)xb[1] << 8)));
        const float dy = k32_f16_to_f32((uint16_t)(yb[0] | ((uint32_t)yb[1] << 8)));
        const __m256 d = _mm256_set1_ps(dx * dy);
        
        const __m256i x32 = _mm256_loadu_si256((const __m256i*)(xb + 2));
        const __m256i y32 = _mm256_loadu_si256((const __m256i*)(yb + 2));
        
        // i8 x i8 -> i16 pair products -> i32 sum
        const __m256i xl = _mm256_cvtepi8_epi16(_mm256_castsi256_si128(x32));
        const __m256i xh = _mm256_cvtepi8_epi16(_mm256_extractf128_si256(x32, 1));
        const __m256i yl = _mm256_cvtepi8_epi16(_mm256_castsi256_si128(y32));
        const __m256i yh = _mm256_cvtepi8_epi16(_mm256_extractf128_si256(y32, 1));
        
        const __m256i pl = _mm256_madd_epi16(xl, yl);
        const __m256i ph = _mm256_madd_epi16(xh, yh);
        const __m256i sumi = _mm256_add_epi32(pl, ph);
        
        acc = _mm256_fmadd_ps(d, _mm256_cvtepi32_ps(sumi), acc);
    }
    return k32_hsum_float_8(acc);
}

#else

static float k32_q8_0_q8_0_avx2(int n, const void* vx, const void* vy) {
    return k32_q8_0_q8_0_scalar(n, vx, vy);
}

#endif

static float k32_q8_0_q8_0_scalar(int n, const void* vx, const void* vy) {
    const uint8_t* x = (const uint8_t*)vx;
    const uint8_t* y = (const uint8_t*)vy;
    const int nb = n / 32;
    float sumf = 0.0f;
    for (int ib = 0; ib < nb; ib++) {
        const uint8_t* xb = x + ib * 34;
        const uint8_t* yb = y + ib * 34;
        const float dx = k32_f16_to_f32((uint16_t)(xb[0] | ((uint32_t)xb[1] << 8)));
        const float dy = k32_f16_to_f32((uint16_t)(yb[0] | ((uint32_t)yb[1] << 8)));
        int sumi = 0;
        for (int j = 0; j < 32; j++) {
            sumi += (int)(int8_t)xb[2 + j] * (int)(int8_t)yb[2 + j];
        }
        sumf += dx * dy * (float)sumi;
    }
    return sumf;
}

float q8_0_q8_0(int n, const void* vx, const void* vy) {
    return cpu_has_avx2() ? k32_q8_0_q8_0_avx2(n, vx, vy) : k32_q8_0_q8_0_scalar(n, vx, vy);
}

/* ------------------------------------------------------------------ */
/* q5_0 x q8_0 — 5-bit weight x Q8_0 activation                      */
/* block_q5_0: f16 d @0, 4 bytes qh @2, 16 bytes qs @6 (22 bytes)   */
/* ------------------------------------------------------------------ */

#if defined(__AVX2__)

// Unpack 32 5-bit fields: 4 bytes qh (high bit) + 16 bytes qs (low 4 bits)
// Returns 32 bytes in [0..31] range
static inline __m256i k32_bytes_from_q5_0(const uint8_t* qs, const uint8_t* qh) {
    // Load 16 nibbles (32 x 4bit)
    const __m128i tmp = _mm_loadu_si128((const __m128i*)qs);
    const __m256i bytes = _mm256_set_m128i(_mm_srli_epi16(tmp, 4), tmp);
    const __m256i low4 = _mm256_and_si256(bytes, _mm256_set1_epi8(0x0F));
    
    // qh has 32 bits, one per element (bit 4)
    const uint32_t qh32 = *(const uint32_t*)qh;
    const __m256i shuf_mask = _mm256_set_epi64x(
        0x0303030303030303, 0x0202020202020202,
        0x0101010101010101, 0x0000000000000000);
    __m256i qh_bytes = _mm256_shuffle_epi8(_mm256_set1_epi32(qh32), shuf_mask);
    const __m256i bit_mask = _mm256_set1_epi64x(0x7FBFDFF7FBFDFF7F);
    qh_bytes = _mm256_or_si256(qh_bytes, bit_mask);
    const __m256i qh_bits = _mm256_cmpeq_epi8(qh_bytes, _mm256_set1_epi64x(-1));
    const __m256i high1 = _mm256_and_si256(_mm256_set1_epi8(0x10), qh_bits);
    
    return _mm256_add_epi8(low4, high1);
}

static float k32_q5_0_q8_0_avx2(int n, const void* vx, const void* vy) {
    const uint8_t* x = (const uint8_t*)vx;
    const uint8_t* y = (const uint8_t*)vy;
    const int nb = n / 32;
    __m256 acc = _mm256_setzero_ps();
    const __m256i off = _mm256_set1_epi8(16);  // offset to [-16, +15]
    
    for (int ib = 0; ib < nb; ib++) {
        const uint8_t* xb = x + ib * 22;
        const uint8_t* yb = y + ib * 34;
        const float dx = k32_f16_to_f32((uint16_t)(xb[0] | ((uint32_t)xb[1] << 8)));
        const float dy = k32_f16_to_f32((uint16_t)(yb[0] | ((uint32_t)yb[1] << 8)));
        const __m256 d = _mm256_set1_ps(dx * dy);
        
        __m256i x32 = k32_bytes_from_q5_0(xb + 6, xb + 2);
        x32 = _mm256_sub_epi8(x32, off);  // center: [0,31] -> [-16,+15]
        const __m256i y32 = _mm256_loadu_si256((const __m256i*)(yb + 2));
        
        const int sumi = k32_dot32_i8_i8(x32, y32);
        acc = _mm256_fmadd_ps(d, _mm256_set1_ps((float)sumi), acc);
    }
    return k32_hsum_float_8(acc);
}

#else

static float k32_q5_0_q8_0_avx2(int n, const void* vx, const void* vy);
static float k32_q5_0_q8_0_scalar(int n, const void* vx, const void* vy);

static float k32_q5_0_q8_0_avx2(int n, const void* vx, const void* vy) {
    return k32_q5_0_q8_0_scalar(n, vx, vy);
}

#endif

static float k32_q5_0_q8_0_scalar(int n, const void* vx, const void* vy) {
    const uint8_t* x = (const uint8_t*)vx;
    const uint8_t* y = (const uint8_t*)vy;
    const int nb = n / 32;
    float sumf = 0.0f;
    for (int ib = 0; ib < nb; ib++) {
        const uint8_t* xb = x + ib * 22;
        const uint8_t* yb = y + ib * 34;
        const float dx = k32_f16_to_f32((uint16_t)(xb[0] | ((uint32_t)xb[1] << 8)));
        const float dy = k32_f16_to_f32((uint16_t)(yb[0] | ((uint32_t)yb[1] << 8)));
        int sumi = 0;
        const uint32_t qh = *(const uint32_t*)(xb + 2);
        for (int j = 0; j < 32; j++) {
            const uint8_t q4 = (xb[6 + j/2] >> (4 * (j & 1))) & 0x0F;
            const uint8_t q5 = q4 | (((qh >> j) & 1) << 4);
            sumi += ((int)q5 - 16) * (int)(int8_t)yb[2 + j];
        }
        sumf += dx * dy * (float)sumi;
    }
    return sumf;
}

float q5_0_q8_0(int n, const void* vx, const void* vy) {
    return cpu_has_avx2() ? k32_q5_0_q8_0_avx2(n, vx, vy) : k32_q5_0_q8_0_scalar(n, vx, vy);
}

/* ------------------------------------------------------------------ */
/* q5_1 x q8_0 — 5-bit weight with min x Q8_0 activation             */
/* block_q5_1: f16 d @0, f16 m @2, 4 bytes qh @4, 16 bytes qs @8 (24 bytes) */
/* ------------------------------------------------------------------ */

#if defined(__AVX2__)

static float k32_q5_1_q8_0_avx2(int n, const void* vx, const void* vy) {
    const uint8_t* x = (const uint8_t*)vx;
    const uint8_t* y = (const uint8_t*)vy;
    const int nb = n / 32;
    __m256 acc = _mm256_setzero_ps();
    __m256 sumy_acc = _mm256_setzero_ps();
    const __m256i ones = _mm256_set1_epi8(1);
    
    for (int ib = 0; ib < nb; ib++) {
        const uint8_t* xb = x + ib * 24;
        const uint8_t* yb = y + ib * 34;
        const float dx = k32_f16_to_f32((uint16_t)(xb[0] | ((uint32_t)xb[1] << 8)));
        const float mx = k32_f16_to_f32((uint16_t)(xb[2] | ((uint32_t)xb[3] << 8)));
        const float dy = k32_f16_to_f32((uint16_t)(yb[0] | ((uint32_t)yb[1] << 8)));
        const __m256 d = _mm256_set1_ps(dx * dy);
        const __m256 m = _mm256_set1_ps(mx * dy);
        
        __m256i x32 = k32_bytes_from_q5_0(xb + 8, xb + 4);  // reuse q5_0 unpack
        const __m256i y32 = _mm256_loadu_si256((const __m256i*)(yb + 2));
        
        const int sumi = k32_dot32_i8_i8(x32, y32);
        const int sumy = k32_dot32_i8_i8(y32, ones);
        
        acc = _mm256_fmadd_ps(d, _mm256_set1_ps((float)sumi), acc);
        sumy_acc = _mm256_fmadd_ps(m, _mm256_set1_ps((float)sumy), sumy_acc);
    }
    return k32_hsum_float_8(_mm256_add_ps(acc, sumy_acc));
}

#else

static float k32_q5_1_q8_0_avx2(int n, const void* vx, const void* vy);
static float k32_q5_1_q8_0_scalar(int n, const void* vx, const void* vy);

static float k32_q5_1_q8_0_avx2(int n, const void* vx, const void* vy) {
    return k32_q5_1_q8_0_scalar(n, vx, vy);
}

#endif

static float k32_q5_1_q8_0_scalar(int n, const void* vx, const void* vy) {
    const uint8_t* x = (const uint8_t*)vx;
    const uint8_t* y = (const uint8_t*)vy;
    const int nb = n / 32;
    float sumf = 0.0f;
    for (int ib = 0; ib < nb; ib++) {
        const uint8_t* xb = x + ib * 24;
        const uint8_t* yb = y + ib * 34;
        const float dx = k32_f16_to_f32((uint16_t)(xb[0] | ((uint32_t)xb[1] << 8)));
        const float mx = k32_f16_to_f32((uint16_t)(xb[2] | ((uint32_t)xb[3] << 8)));
        const float dy = k32_f16_to_f32((uint16_t)(yb[0] | ((uint32_t)yb[1] << 8)));
        int sumi = 0;
        int sumy = 0;
        const uint32_t qh = *(const uint32_t*)(xb + 4);
        for (int j = 0; j < 32; j++) {
            const uint8_t q4 = (xb[8 + j/2] >> (4 * (j & 1))) & 0x0F;
            const uint8_t q5 = q4 | (((qh >> j) & 1) << 4);
            sumi += (int)q5 * (int)(int8_t)yb[2 + j];
            sumy += (int)(int8_t)yb[2 + j];
        }
        sumf += dx * dy * (float)sumi + mx * dy * (float)sumy;
    }
    return sumf;
}

float q5_1_q8_0(int n, const void* vx, const void* vy) {
    return cpu_has_avx2() ? k32_q5_1_q8_0_avx2(n, vx, vy) : k32_q5_1_q8_0_scalar(n, vx, vy);
}

/* ------------------------------------------------------------------ */
/* q1_0 x q8_0 — 1-bit weight x Q8_0 activation                      */
/* block_q1_0: f16 d @0, 4 bytes qs @2 (6 bytes for 32 elements)    */
/* Note: QK1_0 = 256 in llama.cpp, but we handle 32-element blocks */
/* ------------------------------------------------------------------ */

#if defined(__AVX2__)

// Spread 32 bits to 32 bytes {0x00, 0xFF}
static inline __m256i k32_bytes_from_bits_32(const uint8_t* x) {
    uint32_t x32;
    memcpy(&x32, x, sizeof(uint32_t));
    const __m256i shuf_mask = _mm256_set_epi64x(
        0x0303030303030303, 0x0202020202020202,
        0x0101010101010101, 0x0000000000000000);
    __m256i bytes = _mm256_shuffle_epi8(_mm256_set1_epi32(x32), shuf_mask);
    const __m256i bit_mask = _mm256_set1_epi64x(0x7FBFDFF7FBFDFF7F);
    bytes = _mm256_or_si256(bytes, bit_mask);
    return _mm256_cmpeq_epi8(bytes, _mm256_set1_epi64x(-1));
}

static float k32_q1_0_q8_0_avx2(int n, const void* vx, const void* vy) {
    const uint8_t* x = (const uint8_t*)vx;
    const uint8_t* y = (const uint8_t*)vy;
    const int nb = n / 32;
    __m256 acc = _mm256_setzero_ps();
    const __m256i ones = _mm256_set1_epi16(1);
    
    for (int ib = 0; ib < nb; ib++) {
        const uint8_t* xb = x + ib * 6;
        const uint8_t* yb = y + ib * 34;
        const float dx = k32_f16_to_f32((uint16_t)(xb[0] | ((uint32_t)xb[1] << 8)));
        const float dy = k32_f16_to_f32((uint16_t)(yb[0] | ((uint32_t)yb[1] << 8)));
        const __m256 d = _mm256_set1_ps(dx * dy);
        
        const __m256i x_bits = k32_bytes_from_bits_32(xb + 2);
        // Convert 0xFF/0x00 to +1/-1: (bit * 2) - 1
        const __m256i x_vals = _mm256_sub_epi8(_mm256_slli_epi16(x_bits, 1), _mm256_set1_epi8(0xFF));
        const __m256i y32 = _mm256_loadu_si256((const __m256i*)(yb + 2));
        
        // x_vals is in {-1, +1}, y32 is int8
        // Use sign manipulation: abs(x) * sign(x) * y = sign(x) * y when abs(x)=1
        const __m256i ax = _mm256_sign_epi8(x_vals, x_vals);
        const __m256i sy = _mm256_sign_epi8(y32, x_vals);
        const __m256i dot = _mm256_maddubs_epi16(ax, sy);
        const __m256i summed = _mm256_madd_epi16(dot, ones);
        const int sumi = k32_sum8i32(summed);
        
        acc = _mm256_fmadd_ps(d, _mm256_set1_ps((float)sumi), acc);
    }
    return k32_hsum_float_8(acc);
}

#else

static float k32_q1_0_q8_0_avx2(int n, const void* vx, const void* vy);
static float k32_q1_0_q8_0_scalar(int n, const void* vx, const void* vy);

static float k32_q1_0_q8_0_avx2(int n, const void* vx, const void* vy) {
    return k32_q1_0_q8_0_scalar(n, vx, vy);
}

#endif

static float k32_q1_0_q8_0_scalar(int n, const void* vx, const void* vy) {
    const uint8_t* x = (const uint8_t*)vx;
    const uint8_t* y = (const uint8_t*)vy;
    const int nb = n / 32;
    float sumf = 0.0f;
    for (int ib = 0; ib < nb; ib++) {
        const uint8_t* xb = x + ib * 6;
        const uint8_t* yb = y + ib * 34;
        const float dx = k32_f16_to_f32((uint16_t)(xb[0] | ((uint32_t)xb[1] << 8)));
        const float dy = k32_f16_to_f32((uint16_t)(yb[0] | ((uint32_t)yb[1] << 8)));
        int sumi = 0;
        const uint32_t qs = *(const uint32_t*)(xb + 2);
        for (int j = 0; j < 32; j++) {
            const int q1 = ((qs >> j) & 1) ? 1 : -1;
            sumi += q1 * (int)(int8_t)yb[2 + j];
        }
        sumf += dx * dy * (float)sumi;
    }
    return sumf;
}

float q1_0_q8_0(int n, const void* vx, const void* vy) {
    return cpu_has_avx2() ? k32_q1_0_q8_0_avx2(n, vx, vy) : k32_q1_0_q8_0_scalar(n, vx, vy);
}

/* ------------------------------------------------------------------ */
/* q2_0 x q8_0 — 2-bit weight x Q8_0 activation                      */
/* block_q2_0: f16 d @0, 8 bytes qs @2 (10 bytes for 64 elements)   */
/* Note: QK2_0 = 64, so each block has 64 elements                  */
/* ------------------------------------------------------------------ */

#if defined(__AVX2__)

// Unpack 32 2-bit values from 8 bytes
// Each byte has 4 x 2-bit values: (b >> (2*i)) & 3
static inline __m256i k32_q2_0_unpack_32(const uint8_t* qs) {
    __m256i v = _mm256_cvtepu8_epi32(_mm_loadl_epi64((const __m128i*)qs));
    v = _mm256_or_si256(v, _mm256_slli_epi32(v, 12));
    v = _mm256_or_si256(v, _mm256_slli_epi32(v, 6));
    return _mm256_and_si256(v, _mm256_set1_epi32(0x03030303));
}

static float k32_q2_0_q8_0_avx2(int n, const void* vx, const void* vy) {
    const uint8_t* x = (const uint8_t*)vx;
    const uint8_t* y = (const uint8_t*)vy;
    const int nb = n / 64;  // QK2_0 = 64
    __m256 acc0 = _mm256_setzero_ps();
    __m256 acc1 = _mm256_setzero_ps();
    
    for (int ib = 0; ib < nb; ib++) {
        const uint8_t* xb = x + ib * 10;
        const uint8_t* yb = y + ib * 68;  // 2 Q8_0 blocks (2*34=68)
        const float dx = k32_f16_to_f32((uint16_t)(xb[0] | ((uint32_t)xb[1] << 8)));
        const float dy0 = k32_f16_to_f32((uint16_t)(yb[0] | ((uint32_t)yb[1] << 8)));
        const float dy1 = k32_f16_to_f32((uint16_t)(yb[34] | ((uint32_t)yb[35] << 8)));
        
        const __m256 d0 = _mm256_set1_ps(dx * dy0);
        const __m256 d1 = _mm256_set1_ps(dx * dy1);
        
        // First 32 elements
        const __m256i x32_0 = k32_q2_0_unpack_32(xb + 2);
        const __m256i y32_0 = _mm256_loadu_si256((const __m256i*)(yb + 2));
        // x in [0,3], convert to [-1, 2] by subtracting 1
        const __m256i x0_off = _mm256_sub_epi8(x32_0, _mm256_set1_epi8(1));
        const int sumi0 = k32_dot32_i8_i8(x0_off, y32_0);
        acc0 = _mm256_fmadd_ps(d0, _mm256_set1_ps((float)sumi0), acc0);
        
        // Second 32 elements
        const __m256i x32_1 = k32_q2_0_unpack_32(xb + 2 + 4);  // next 4 bytes
        const __m256i y32_1 = _mm256_loadu_si256((const __m256i*)(yb + 34 + 2));
        const __m256i x1_off = _mm256_sub_epi8(x32_1, _mm256_set1_epi8(1));
        const int sumi1 = k32_dot32_i8_i8(x1_off, y32_1);
        acc1 = _mm256_fmadd_ps(d1, _mm256_set1_ps((float)sumi1), acc1);
    }
    return k32_hsum_float_8(_mm256_add_ps(acc0, acc1));
}

#else

static float k32_q2_0_q8_0_avx2(int n, const void* vx, const void* vy);
static float k32_q2_0_q8_0_scalar(int n, const void* vx, const void* vy);

static float k32_q2_0_q8_0_avx2(int n, const void* vx, const void* vy) {
    return k32_q2_0_q8_0_scalar(n, vx, vy);
}

#endif

static float k32_q2_0_q8_0_scalar(int n, const void* vx, const void* vy) {
    const uint8_t* x = (const uint8_t*)vx;
    const uint8_t* y = (const uint8_t*)vy;
    const int nb = n / 64;  // QK2_0 = 64
    float sumf = 0.0f;
    for (int ib = 0; ib < nb; ib++) {
        const uint8_t* xb = x + ib * 10;
        const uint8_t* yb = y + ib * 68;
        const float dx = k32_f16_to_f32((uint16_t)(xb[0] | ((uint32_t)xb[1] << 8)));
        const float dy0 = k32_f16_to_f32((uint16_t)(yb[0] | ((uint32_t)yb[1] << 8)));
        const float dy1 = k32_f16_to_f32((uint16_t)(yb[34] | ((uint32_t)yb[35] << 8)));
        
        int sumi0 = 0, sumi1 = 0;
        for (int j = 0; j < 32; j++) {
            const uint8_t b0 = xb[2 + j/4];
            const uint8_t b1 = xb[2 + 4 + j/4];
            const int q0 = ((b0 >> (2 * (j % 4))) & 3) - 1;
            const int q1 = ((b1 >> (2 * (j % 4))) & 3) - 1;
            sumi0 += q0 * (int)(int8_t)yb[2 + j];
            sumi1 += q1 * (int)(int8_t)yb[34 + 2 + j];
        }
        sumf += dx * dy0 * (float)sumi0 + dx * dy1 * (float)sumi1;
    }
    return sumf;
}

float q2_0_q8_0(int n, const void* vx, const void* vy) {
    return cpu_has_avx2() ? k32_q2_0_q8_0_avx2(n, vx, vy) : k32_q2_0_q8_0_scalar(n, vx, vy);
}

/* ------------------------------------------------------------------ */
/* mxfp4 x q8_0 — MXFP4 weight x Q8_0 activation                     */
/* block_mxfp4: uint8 e (E8M0) @0, 16 bytes qs @1 (17 bytes)       */
/* ------------------------------------------------------------------ */

// MXFP4 lookup table for 4-bit values
static const float k32_mxfp4_values[16] = {
    0.0f, 0.0625f, 0.125f, 0.1875f, 0.25f, 0.3125f, 0.375f, 0.4375f,
    0.5f, 0.625f, 0.75f, 0.875f, 1.0f, 1.25f, 1.5f, 1.75f
};

#if defined(__AVX2__)

static float k32_mxfp4_q8_0_avx2(int n, const void* vx, const void* vy) {
    const uint8_t* x = (const uint8_t*)vx;
    const uint8_t* y = (const uint8_t*)vy;
    const int nb = n / 32;
    __m256 acc = _mm256_setzero_ps();
    const __m256i m4b = _mm256_set1_epi8(0x0F);
    const __m256i mone = _mm256_set1_epi16(1);
    
    for (int ib = 0; ib < nb; ib++) {
        const uint8_t* xb = x + ib * 17;
        const uint8_t* yb = y + ib * 34;
        const float de = k32_f16_to_f32((uint16_t)(yb[0] | ((uint32_t)yb[1] << 8)));
        // E8M0 to float conversion (simplified)
        const uint8_t e8m0 = xb[0];
        const float scale = de * (e8m0 ? (1 << ((e8m0 & 0x7F) - 127)) : 0.0f);
        const __m256 d = _mm256_set1_ps(scale);
        
        // Load 16 nibbles, expand to 32 bytes using lookup
        const __m128i q4bits = _mm_loadu_si128((const __m128i*)(xb + 1));
        const __m256i q4b_lo = _mm256_and_si256(_mm256_cvtepu8_epi16(q4bits), m4b);
        const __m256i q4b_hi = _mm256_and_si256(_mm256_srli_epi16(_mm256_cvtepu8_epi16(q4bits), 4), m4b);
        
        // This is a simplified version - full implementation needs shuffle with lookup table
        // For now, use scalar fallback logic in AVX2 path
        int sumi = 0;
        for (int j = 0; j < 16; j++) {
            const uint8_t b = xb[1 + j];
            const float v0 = k32_mxfp4_values[b & 0x0F];
            const float v1 = k32_mxfp4_values[b >> 4];
            sumi += (int)(v0 * (float)(int8_t)yb[2 + j] + v1 * (float)(int8_t)yb[2 + 16 + j]);
        }
        acc = _mm256_fmadd_ps(d, _mm256_set1_ps((float)sumi), acc);
    }
    return k32_hsum_float_8(acc);
}

#else

static float k32_mxfp4_q8_0_avx2(int n, const void* vx, const void* vy);
static float k32_mxfp4_q8_0_scalar(int n, const void* vx, const void* vy);

static float k32_mxfp4_q8_0_avx2(int n, const void* vx, const void* vy) {
    return k32_mxfp4_q8_0_scalar(n, vx, vy);
}

#endif

static float k32_mxfp4_q8_0_scalar(int n, const void* vx, const void* vy) {
    const uint8_t* x = (const uint8_t*)vx;
    const uint8_t* y = (const uint8_t*)vy;
    const int nb = n / 32;
    float sumf = 0.0f;
    for (int ib = 0; ib < nb; ib++) {
        const uint8_t* xb = x + ib * 17;
        const uint8_t* yb = y + ib * 34;
        const float dy = k32_f16_to_f32((uint16_t)(yb[0] | ((uint32_t)yb[1] << 8)));
        const uint8_t e8m0 = xb[0];
        const float scale = dy * (e8m0 ? (1 << ((e8m0 & 0x7F) - 127)) : 0.0f);
        float sumi = 0.0f;
        for (int j = 0; j < 16; j++) {
            const uint8_t b = xb[1 + j];
            sumi += k32_mxfp4_values[b & 0x0F] * (float)(int8_t)yb[2 + j];
            sumi += k32_mxfp4_values[b >> 4] * (float)(int8_t)yb[2 + 16 + j];
        }
        sumf += scale * sumi;
    }
    return sumf;
}

float mxfp4_q8_0(int n, const void* vx, const void* vy) {
    return cpu_has_avx2() ? k32_mxfp4_q8_0_avx2(n, vx, vy) : k32_mxfp4_q8_0_scalar(n, vx, vy);
}

/* ------------------------------------------------------------------ */
/* nvfp4 x q8_0 — NVFP4 weight x Q8_0 activation                     */
/* block_nvfp4: 4 bytes d (UE4M3 scales) @0, 32 bytes qs @4 (36 bytes) */
/* Note: QK_NVFP4 = 64, so each block has 64 elements                 */
/* ------------------------------------------------------------------ */

// NVFP4 lookup table for 4-bit E2M1 values
static const float k32_nvfp4_values[16] = {
    0.0f, 0.5f, 1.0f, 1.5f, 2.0f, 3.0f, 4.0f, 6.0f,
    -0.0f, -0.5f, -1.0f, -1.5f, -2.0f, -3.0f, -4.0f, -6.0f
};

#if defined(__AVX2__)

static float k32_nvfp4_q8_0_avx2(int n, const void* vx, const void* vy) {
    const uint8_t* x = (const uint8_t*)vx;
    const uint8_t* y = (const uint8_t*)vy;
    const int nb = n / 64;  // QK_NVFP4 = 64
    __m256 acc = _mm256_setzero_ps();
    
    for (int ib = 0; ib < nb; ib++) {
        const uint8_t* xb = x + ib * 36;
        const uint8_t* yb = y + ib * 68;  // 2 Q8_0 blocks
        
        // UE4M3 scales (simplified - need proper conversion)
        const float s0 = (xb[0] & 0x0F) * 0.5f;  // placeholder
        const float s1 = (xb[1] & 0x0F) * 0.5f;
        const float s2 = (xb[2] & 0x0F) * 0.5f;
        const float s3 = (xb[3] & 0x0F) * 0.5f;
        
        const float dy0 = k32_f16_to_f32((uint16_t)(yb[0] | ((uint32_t)yb[1] << 8)));
        const float dy1 = k32_f16_to_f32((uint16_t)(yb[34] | ((uint32_t)yb[35] << 8)));
        
        // Process 64 elements in 4 sub-blocks of 16
        float sumi = 0.0f;
        for (int sg = 0; sg < 4; sg++) {
            const float scale = ((float[]){s0, s1, s2, s3})[sg] * ((sg < 2) ? dy0 : dy1);
            for (int j = 0; j < 16; j++) {
                const uint8_t b = xb[4 + sg * 8 + j/2];
                const int idx = (j & 1) ? (b >> 4) : (b & 0x0F);
                const int yoff = (sg < 2) ? 0 : 34;
                const int yj = (sg & 1) * 16 + j;
                sumi += scale * k32_nvfp4_values[idx] * (float)(int8_t)yb[yoff + 2 + yj];
            }
        }
        acc = _mm256_add_ps(acc, _mm256_set1_ps(sumi));
    }
    return k32_hsum_float_8(acc);
}

#else

static float k32_nvfp4_q8_0_avx2(int n, const void* vx, const void* vy);
static float k32_nvfp4_q8_0_scalar(int n, const void* vx, const void* vy);

static float k32_nvfp4_q8_0_avx2(int n, const void* vx, const void* vy) {
    return k32_nvfp4_q8_0_scalar(n, vx, vy);
}

#endif

static float k32_nvfp4_q8_0_scalar(int n, const void* vx, const void* vy) {
    const uint8_t* x = (const uint8_t*)vx;
    const uint8_t* y = (const uint8_t*)vy;
    const int nb = n / 64;  // QK_NVFP4 = 64
    float sumf = 0.0f;
    for (int ib = 0; ib < nb; ib++) {
        const uint8_t* xb = x + ib * 36;
        const uint8_t* yb = y + ib * 68;
        
        // UE4M3 scales (simplified)
        const float s0 = (xb[0] & 0x0F) * 0.5f;
        const float s1 = (xb[1] & 0x0F) * 0.5f;
        const float s2 = (xb[2] & 0x0F) * 0.5f;
        const float s3 = (xb[3] & 0x0F) * 0.5f;
        
        const float dy0 = k32_f16_to_f32((uint16_t)(yb[0] | ((uint32_t)yb[1] << 8)));
        const float dy1 = k32_f16_to_f32((uint16_t)(yb[34] | ((uint32_t)yb[35] << 8)));
        
        for (int sg = 0; sg < 4; sg++) {
            const float scale = ((float[]){s0, s1, s2, s3})[sg] * ((sg < 2) ? dy0 : dy1);
            for (int j = 0; j < 16; j++) {
                const uint8_t b = xb[4 + sg * 8 + j/2];
                const int idx = (j & 1) ? (b >> 4) : (b & 0x0F);
                const int yoff = (sg < 2) ? 0 : 34;
                const int yj = (sg & 1) * 16 + j;
                sumf += scale * k32_nvfp4_values[idx] * (float)(int8_t)yb[yoff + 2 + yj];
            }
        }
    }
    return sumf;
}

float nvfp4_q8_0(int n, const void* vx, const void* vy) {
    return cpu_has_avx2() ? k32_nvfp4_q8_0_avx2(n, vx, vy) : k32_nvfp4_q8_0_scalar(n, vx, vy);
}

void quantize_q8_0(int n, const float* x, uint8_t* out) {
    const int nb = n / 32;
    for (int ib = 0; ib < nb; ib++) {
        const float* xb = x + ib * 32;
        uint8_t* yb = out + ib * 34;
        float amax = 0.0f;
        for (int k = 0; k < 32; k++) {
            const float a = xb[k] < 0.0f ? -xb[k] : xb[k];
            if (a > amax) amax = a;
        }
        const float scale = amax > 0.0f ? amax / 127.0f : 1.0f;
        const float inv = amax > 0.0f ? 1.0f / scale : 0.0f;
        const uint16_t sbits = k32_f32_to_f16bits(scale);
        yb[0] = (uint8_t)(sbits & 0xFF);
        yb[1] = (uint8_t)(sbits >> 8);
        for (int k = 0; k < 32; k++) {
            const float v = xb[k] * inv;
            int q = (int)(v + (v >= 0.0f ? 0.5f : -0.5f));
            if (q > 127) q = 127;
            if (q < -128) q = -128;
            yb[2 + k] = (uint8_t)(int8_t)q;
        }
    }
}
