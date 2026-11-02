/*
 * kernels256.c — 256-block K-quant (Q2_K..Q6_K) x Q8_K dot-product kernels and
 * the Q8_K activation quantizer, ported from the Zig reference:
 *   - src/simd/kern256.zig  *_ref functions (exact byte layout / f16 decode /
 *     dot math)
 *   - src/simd/act.zig     quantizeQ8_K (scale/round/clamp + bsums)
 *
 * Contract:
 *   n is a multiple of 256.
 *   vx = weight row of the K-type, packed in 256-element super-blocks:
 *        q2_K 84 B, q3_K 110 B, q4_K 144 B, q5_K 176 B, q6_K 210 B / block.
 *   vy = Q8_K activation, 292 bytes per 256 elements:
 *        [0..3]     d (f32)
 *        [4..259]   qs[256] (int8)
 *        [260..291] bsums[16] (int16, little-endian) = sum of qs per 16
 *   Return the dot product of the dequantized weight row and activation over
 *   all n elements (float).
 *
 * quantize_q8_K(n, x, out): x[0..n) -> Q8_K blocks in out (292 bytes/256, same
 *   layout). scale = amax/127 (f32; 1.0 if amax==0), q = round(x/scale)
 *   clamped to [-128,127], bsums filled.
 *
 * NOTE on symbols: sibling C files in this directory are concatenated, so
 * every internal symbol is static AND prefixed k256_. Only the six public
 * functions below are non-static and unprefixed.
 *
 * NOTE on AVX2: each kernel has a *_scalar (the ported oracle) and a *_avx2.
 * AVX2 has no int8 multiply intrinsic (that is AVX512-VNNI), so the AVX2
 * path is a thin, correct delegation to the scalar port. This keeps the
 * -mavx2 build real and guarantees avx2 == scalar (oracle-exact). See the
 * per-kernel comments.
 */
#include <stdint.h>
#include "iq_tables.h"
#include <string.h>
#include <math.h>
#include "cpu_caps.h"

#if defined(__AVX2__)
#include <immintrin.h>
#endif

/* ------------------------------------------------------------------ */
/* shared helpers                                                      */
/* ------------------------------------------------------------------ */

/* Runtime dispatch uses cpu_has_avx2() from cpu_caps.c (probed once at load
 * time). Do NOT re-implement cpuid/xgetbv in this file. */

/* IEEE 754 binary16 -> float, exact (mirrors util.f16At in kern256.zig). */
static float k256_f16_to_f32(uint16_t h)
{
    const uint32_t s = ((uint32_t)h & 0x8000u) << 16;
    const uint32_t e = (uint32_t)(h >> 10) & 0x1Fu;
    const uint32_t m = (uint32_t)h & 0x3FFu;
    uint32_t bits;
    if (e == 0) {
        if (m == 0) return s ? -0.0f : 0.0f;
        /* subnormal: value = m * 2^-24.  Normalize the mantissa into
         * [512,1024) exactly like k32_f16_to_f32/k512_f16_to_f32 do, and keep
         * the sign bit.  (Was: (float)((double)m * 0x1.0p-24), which silently
         * dropped `s` -- every negative f16-subnormal block scale (|d| < 2^-14,
         * frequent for Q6_K ssm_out / Q4_K dmin) came back POSITIVE, so whole
         * 256-element blocks contributed with a flipped sign -> gibberish
         * output.  Verified against blk.0.ssm_out.weight row 0 blocks 1,2,5,7.) */
        uint32_t mm = m;
        uint32_t lz = 0;
        while (mm < 0x200u) { mm <<= 1; lz++; }
        bits = s | ((112u - lz) << 23) | ((mm & 0x1FFu) << 14);
    } else if (e == 0x1Fu) {
        bits = s | 0x7F800000u | (m << 13); /* inf / nan */
    } else {
        bits = s | ((e + 112u) << 23) | (m << 13); /* normal */
    }
    float f;
    memcpy(&f, &bits, sizeof f);
    return f;
}

static float k256_f16_at(const uint8_t * p, int off)
{
    uint16_t h;
    memcpy(&h, p + off, sizeof h);
    return k256_f16_to_f32(h);
}

static float k256_q8k_d(const uint8_t * y)
{
    float d;
    memcpy(&d, y, sizeof d);
    return d;
}

static int32_t k256_q8k_bsum(const uint8_t * y, int j)
{
    uint16_t v;
    memcpy(&v, y + 260 + j * 2, sizeof v);
    return (int32_t)(int16_t)v;
}

/* Q4_K/Q5_K scale/min unpack: 12 bytes -> 16 x u8.
 * sm[0..8]  = 6-bit scales (each covers 32 elements, used as sm[g/2]),
 * sm[8..16] = 6-bit mins   (each covers 16 elements, used as sm[8 + j/2]).
 * Port of unpackScaleMin in kern256.zig. */
static void k256_unpack_scale_min(const uint8_t * s12, uint8_t * sm /*[16]*/)
{
    uint32_t u[4];
    u[0] = (uint32_t)s12[0] | ((uint32_t)s12[1] << 8) | ((uint32_t)s12[2] << 16) | ((uint32_t)s12[3] << 24);
    u[1] = (uint32_t)s12[4] | ((uint32_t)s12[5] << 8) | ((uint32_t)s12[6] << 16) | ((uint32_t)s12[7] << 24);
    u[2] = (uint32_t)s12[8] | ((uint32_t)s12[9] << 8) | ((uint32_t)s12[10] << 16) | ((uint32_t)s12[11] << 24);
    u[3] = 0;
    const uint32_t kmask1 = 0x3F3F3F3Fu;
    const uint32_t kmask2 = 0x0F0F0F0Fu;
    const uint32_t kmask3 = 0x03030303u;
    const uint32_t tmp = u[2];
    u[3]  = ((u[2] >> 4) & kmask2) | (((u[1] >> 6) & kmask3) << 4);
    const uint32_t uaux = u[1] & kmask1;
    u[1]  = (u[2] & kmask2) | (((u[0] >> 6) & kmask3) << 4);
    u[2]  = uaux;
    u[0] &= kmask1;
    (void)tmp;
    for (int i = 0; i < 4; i++)
        for (int b = 0; b < 4; b++)
            sm[i * 4 + b] = (uint8_t)(u[i] >> (b * 8));
}

/* Q3_K scale unpack: 12 bytes -> 16 x i8 (6-bit values 0..63).
 * Port of unpackQ3KScales in kern256.zig. */
static void k256_unpack_q3_scales(const uint8_t * s12, int8_t * sc /*[16]*/)
{
    uint32_t a[4];
    for (int i = 0; i < 3; i++) {
        a[i] = (uint32_t)s12[i * 4] | ((uint32_t)s12[i * 4 + 1] << 8) |
               ((uint32_t)s12[i * 4 + 2] << 16) | ((uint32_t)s12[i * 4 + 3] << 24);
    }
    a[3] = 0;
    const uint32_t kmask1 = 0x03030303u;
    const uint32_t kmask2 = 0x0F0F0F0Fu;
    const uint32_t tmp = a[2];
    a[2] = ((a[0] >> 4) & kmask2) | (((tmp >> 4) & kmask1) << 4);
    a[3] = ((a[1] >> 4) & kmask2) | (((tmp >> 6) & kmask1) << 4);
    a[0] = (a[0] & kmask2) | (((tmp >> 0) & kmask1) << 4);
    a[1] = (a[1] & kmask2) | (((tmp >> 2) & kmask1) << 4);
    (void)tmp;
    for (int i = 0; i < 4; i++)
        for (int b = 0; b < 4; b++)
            sc[i * 4 + b] = (int8_t)(uint8_t)(a[i] >> (b * 8));
}

/* aux8 builders: element-wise (pre-scale) weight values, exactly as the Zig
 * _ref code lays them out. a8[e] for element e in [0,256). */

static void k256_q2_aux8(const uint8_t * x, int8_t * a8)
{
    const uint8_t * q2 = x + 16;
    for (int k = 0; k < 2; k++)
        for (int j = 0; j < 4; j++)
            for (int l = 0; l < 32; l++)
                a8[k * 128 + j * 32 + l] = (int8_t)((q2[k * 32 + l] >> (2 * j)) & 0x03);
}

static void k256_q3_aux8(const uint8_t * x, int8_t * a8)
{
    const uint8_t * q3 = x + 32;
    const uint8_t * hm = x;
    int m = 1;
    for (int h = 0; h < 2; h++) {
        for (int g = 0; g < 4; g++) {
            for (int l = 0; l < 32; l++) {
                const int8_t nib = (int8_t)((q3[h * 32 + l] >> (2 * g)) & 0x03);
                const int8_t sub = (hm[l] & m) ? 0 : 4;
                a8[h * 128 + g * 32 + l] = (int8_t)(nib - sub);
            }
            m <<= 1;
        }
    }
}

static void k256_q4_aux8(const uint8_t * x, int8_t * a8)
{
    const uint8_t * q4 = x + 16;
    for (int c = 0; c < 4; c++) {
        for (int l = 0; l < 32; l++) a8[c * 64 + l]        = (int8_t)(q4[c * 32 + l] & 0x0F);
        for (int l = 0; l < 32; l++) a8[c * 64 + 32 + l]   = (int8_t)(q4[c * 32 + l] >> 4);
    }
}

static void k256_q5_aux8(const uint8_t * x, int8_t * a8)
{
    const uint8_t * q4 = x + 48;
    const uint8_t * hm = x + 16;
    int m = 1;
    for (int c = 0; c < 4; c++) {
        for (int l = 0; l < 32; l++) a8[c * 64 + l]      = (int8_t)((q4[c * 32 + l] & 0x0F) + ((hm[l] & m) ? 16 : 0));
        for (int l = 0; l < 32; l++) a8[c * 64 + 32 + l] = (int8_t)((q4[c * 32 + l] >> 4)  + ((hm[l] & (m << 1)) ? 16 : 0));
        m <<= 2;
    }
}

static void k256_q6_aux8(const uint8_t * x, int8_t * a8)
{
    const uint8_t * q4 = x;
    const uint8_t * qh = x + 128;
    for (int h = 0; h < 2; h++) {
        for (int l = 0; l < 32; l++) {
            a8[h * 128 + l + 0]  = (int8_t)(((q4[h * 64 + l]        & 0x0F) | (((qh[h * 32 + l] >> 0) & 3) << 4)) - 32);
            a8[h * 128 + l + 32] = (int8_t)(((q4[h * 64 + l + 32]   & 0x0F) | (((qh[h * 32 + l] >> 2) & 3) << 4)) - 32);
            a8[h * 128 + l + 64] = (int8_t)(((q4[h * 64 + l]        >> 4) | (((qh[h * 32 + l] >> 4) & 3) << 4)) - 32);
            a8[h * 128 + l + 96] = (int8_t)(((q4[h * 64 + l + 32]   >> 4) | (((qh[h * 32 + l] >> 6) & 3) << 4)) - 32);
        }
    }
}

/* 16 groups of 16: isum = sum_g sc[g] * sum_l a8[g*16+l] * q8[g*16+l].
 * Scalar reference (same order of ops as dotScaled in kern256.zig). */
static int32_t k256_dot256(const int8_t * a8, const uint8_t * q8, const int32_t * sc /*[16]*/)
{
    int32_t isum = 0;
    for (int g = 0; g < 16; g++) {
        int32_t sub = 0;
        for (int l = 0; l < 16; l++)
            sub += (int32_t)a8[g * 16 + l] * (int32_t)(int8_t)q8[g * 16 + l];
        isum += sc[g] * sub;
    }
    return isum;
}

/* ------------------------------------------------------------------ */
/* Q2_K x Q8_K   (84-byte blocks)                                      */
/* ------------------------------------------------------------------ */

static float k256_q2_k_q8_k_scalar(int n, const uint8_t * vx, const uint8_t * vy)
{
    float sumf = 0.0f;
    const int nb = n / 256;
    for (int i = 0; i < nb; i++) {
        const uint8_t * x = vx + (size_t)i * 84;
        const uint8_t * y = vy + (size_t)i * 292;
        const uint8_t * sc = x; /* 16 bytes: low nibble = scale, high nibble = min */
        int8_t a8[256];
        k256_q2_aux8(x, a8);
        int32_t summs = 0;
        for (int j = 0; j < 16; j++)
            summs += k256_q8k_bsum(y, j) * (int32_t)(sc[j] >> 4);
        int32_t scv[16];
        for (int g = 0; g < 16; g++)
            scv[g] = (int32_t)(sc[g] & 0x0F);
        const int32_t isum = k256_dot256(a8, y + 4, scv);
        const float dy = k256_q8k_d(y);
        const float dall = dy * k256_f16_at(x, 80);
        const float dmin = dy * k256_f16_at(x, 82);
        sumf += dall * (float)isum - dmin * (float)summs;
    }
    return sumf;
}

/* AVX2: 8 groups of 32 unsigned 2-bit values, one 4-bit scale per group.
 * Same shape as the Q4_K vector path; only the nibble extraction differs
 * (a 2-bit shift instead of a 4-bit mask).  Values are 0..3 unsigned so
 * `maddubs` (u8 x s8) is exact; 2*3*127 = 762 fits i16 with plenty of room. */
static float k256_q2_k_q8_k_avx2(int n, const uint8_t * vx, const uint8_t * vy)
{
#if defined(__AVX2__)
    const __m256i m3 = _mm256_set1_epi8(0x03);
    __m256 acc = _mm256_setzero_ps();
    __m256 acc_m = _mm256_setzero_ps();
    const int nb = n / 256;

    for (int i = 0; i < nb; i++) {
        const uint8_t * x = vx + (size_t)i * 84;
        const uint8_t * y = vy + (size_t)i * 292;
        const uint8_t * sc = x; /* 16 bytes: low nibble = scale, high = min */
        const float dy   = k256_q8k_d(y);
        const float dall = dy * k256_f16_at(x, 80);
        const float dmin = dy * k256_f16_at(x, 82);

        /* dmin * sum_j bsum[j] * min[j]  (one min per 16 elements here, not
         * per 32 -- so all 16 mins are used, no hadd pairing). */
        {
            const __m256i q8sums = _mm256_loadu_si256((const __m256i*)(y + 260));
            /* mins are the HIGH nibble: AVX2 has no u8 >> 4, so unpack to i16
             * first and shift there. */
            const __m256i sc16 = _mm256_cvtepu8_epi16(_mm_loadu_si128((const __m128i*)sc));
            const __m256i min16 = _mm256_srli_epi16(sc16, 4);
            const __m256i prod = _mm256_madd_epi16(min16, q8sums);
            acc_m = _mm256_fmadd_ps(_mm256_set1_ps(-dmin),
                                    _mm256_cvtepi32_ps(prod), acc_m);
        }

        /* scales: low nibble, one per 32 elements.  Keep them in a u8 array so
         * the broadcast is a plain scalar load, not a variable vector extract. */
        uint8_t scl[16];
        for (int g = 0; g < 16; g++) scl[g] = (uint8_t)(sc[g] & 0x0F);

        __m256i sumi = _mm256_setzero_si256();
        const int8_t * q8 = (const int8_t*)(y + 4);
        const uint8_t * q2 = x + 16;

        for (int k = 0; k < 2; k++) {
            const __m256i qb = _mm256_loadu_si256((const __m256i*)(q2 + k * 32));
            /* j = 0..3 -> (qb >> 2j) & 3 */
            const __m256i v0 = _mm256_and_si256(qb, m3);
            const __m256i v1 = _mm256_and_si256(_mm256_srli_epi16(qb, 2), m3);
            const __m256i v2 = _mm256_and_si256(_mm256_srli_epi16(qb, 4), m3);
            const __m256i v3 = _mm256_and_si256(_mm256_srli_epi16(qb, 6), m3);

            /* One scale covers 16 elements, so each 32-element maddubs result
             * needs TWO scales: sc[2g] for the low half, sc[2g+1] for the high. */
            #define Q2_SP(gi) \
                _mm256_setr_epi16((short)scl[(gi)+0], (short)scl[(gi)+0], \
                                  (short)scl[(gi)+0], (short)scl[(gi)+0], \
                                  (short)scl[(gi)+0], (short)scl[(gi)+0], \
                                  (short)scl[(gi)+0], (short)scl[(gi)+0], \
                                  (short)scl[(gi)+1], (short)scl[(gi)+1], \
                                  (short)scl[(gi)+1], (short)scl[(gi)+1], \
                                  (short)scl[(gi)+1], (short)scl[(gi)+1], \
                                  (short)scl[(gi)+1], (short)scl[(gi)+1])
            const int8_t * q8b = q8 + k * 128;
            __m256i p0 = _mm256_madd_epi16(Q2_SP(k * 8 + 0), _mm256_maddubs_epi16(v0, _mm256_loadu_si256((const __m256i*)(q8b + 0))));
            __m256i p1 = _mm256_madd_epi16(Q2_SP(k * 8 + 2), _mm256_maddubs_epi16(v1, _mm256_loadu_si256((const __m256i*)(q8b + 32))));
            __m256i p2 = _mm256_madd_epi16(Q2_SP(k * 8 + 4), _mm256_maddubs_epi16(v2, _mm256_loadu_si256((const __m256i*)(q8b + 64))));
            __m256i p3 = _mm256_madd_epi16(Q2_SP(k * 8 + 6), _mm256_maddubs_epi16(v3, _mm256_loadu_si256((const __m256i*)(q8b + 96))));
            #undef Q2_SP
            sumi = _mm256_add_epi32(sumi, _mm256_add_epi32(
                _mm256_add_epi32(p0, p1), _mm256_add_epi32(p2, p3)));
        }

        acc = _mm256_fmadd_ps(_mm256_set1_ps(dall), _mm256_cvtepi32_ps(sumi), acc);
    }

    /* acc_m is already a full __m256 (8 i32 pair sums), so it reduces the
     * same way acc does -- no 128-bit narrowing needed. */
    const __m256 tot = _mm256_add_ps(acc, acc_m);
    const __m128 lo = _mm256_castps256_ps128(tot);
    const __m128 hi = _mm256_extractf128_ps(tot, 1);
    __m128 t = _mm_add_ps(lo, hi);
    t = _mm_hadd_ps(t, t);
    t = _mm_hadd_ps(t, t);
    return _mm_cvtss_f32(t);
#else
    return k256_q2_k_q8_k_scalar(n, vx, vy);
#endif
}

/* ------------------------------------------------------------------ */
/* Q3_K x Q8_K   (110-byte blocks)                                     */
/* ------------------------------------------------------------------ */

static float k256_q3_k_q8_k_scalar(int n, const uint8_t * vx, const uint8_t * vy)
{
    float sumf = 0.0f;
    const int nb = n / 256;
    for (int i = 0; i < nb; i++) {
        const uint8_t * x = vx + (size_t)i * 110;
        const uint8_t * y = vy + (size_t)i * 292;
        int8_t a8[256];
        k256_q3_aux8(x, a8);
        int8_t sc[16];
        k256_unpack_q3_scales(x + 96, sc);
        int32_t scv[16];
        for (int g = 0; g < 16; g++)
            scv[g] = (int32_t)sc[g] - 32; /* 6-bit scale, used as (scale - 32) */
        const int32_t isum = k256_dot256(a8, y + 4, scv);
        const float dy = k256_q8k_d(y);
        sumf += k256_f16_at(x, 108) * dy * (float)isum;
    }
    return sumf;
}

static float k256_q3_k_q8_k_avx2(int n, const uint8_t * vx, const uint8_t * vy)
{
#if defined(__AVX2__)
    /* Q3_K reuses the 2-bit layout of Q2_K but the value is SIGNED:
     *   a8[h*128 + g*32 + l] = nib - (hm[l] & (1<<(4h+g)) ? 0 : 4)
     * so it lives in -4..3 while `maddubs` needs an unsigned first operand.
     * Keep the raw 0..3 nibble and peel the per-element offset off the same
     * way Q6_K does:
     *   sum (nib - 4) * q * s  =  sum nib * q * s  -  4 * sum q * s
     * The correction for a 16-element group g is 4 * s_g * bsum_g, so it needs
     * the activation sums of the *groups where the mask bit is clear only*.
     * That is per-element, not per-group, so it cannot use bsums.  Instead the
     * offset is folded into the weight vector itself before the multiply:
     * build a 0/4 vector from hm and subtract it from the (widened) nibbles.
     * Widening to i16 first costs one unpack per half but keeps everything
     * exact and avoids a separate correction pass.
     */
    const __m256i m3 = _mm256_set1_epi8(0x03);
    __m256 acc = _mm256_setzero_ps();
    const int nb = n / 256;

    for (int i = 0; i < nb; i++) {
        const uint8_t * x = vx + (size_t)i * 110;
        const uint8_t * y = vy + (size_t)i * 292;
        const float d = k256_f16_at(x, 108) * k256_q8k_d(y);

        int8_t sc[16];
        k256_unpack_q3_scales(x + 96, sc);
        /* the scalar path uses (sc[g] - 32) as the group scale */
        int32_t scv[16];
        for (int g = 0; g < 16; g++) scv[g] = (int32_t)sc[g] - 32;

        __m256i sumi = _mm256_setzero_si256();
        const int8_t * q8 = (const int8_t*)(y + 4);
        const uint8_t * q3 = x + 32;
        const uint8_t * hm = x;      /* 32 B, one bit per (h,g) pair */

        for (int h = 0; h < 2; h++) {
            const __m256i qb = _mm256_loadu_si256((const __m256i*)(q3 + h * 32));
            const __m256i hmb = _mm256_loadu_si256((const __m256i*)hm);

            /* four 2-bit slices; each is used for one 32-element group */
            const __m256i v[4] = {
                _mm256_and_si256(qb, m3),
                _mm256_and_si256(_mm256_srli_epi16(qb, 2), m3),
                _mm256_and_si256(_mm256_srli_epi16(qb, 4), m3),
                _mm256_and_si256(_mm256_srli_epi16(qb, 6), m3),
            };

            for (int g = 0; g < 4; g++) {
                /* mask bit index for (h,g) is 4h+g; 0 -> subtract 4 */
                const __m256i bit = _mm256_set1_epi8((char)(1 << (4 * h + g)));
                const __m256i zero = _mm256_cmpeq_epi8(_mm256_and_si256(hmb, bit), _mm256_setzero_si256());
                /* zero is 0xFF where the bit is clear -> subtract 4 there */
                const __m256i sub = _mm256_and_si256(zero, _mm256_set1_epi8(4));

                /* widen the two halves to i16 and apply the offset there */
                const __m256i lo = _mm256_cvtepu8_epi16(_mm256_castsi256_si128(v[g]));
                const __m256i hi = _mm256_cvtepu8_epi16(_mm256_extracti128_si256(v[g], 1));
                const __m256i slo = _mm256_cvtepu8_epi16(_mm256_castsi256_si128(sub));
                const __m256i shi = _mm256_cvtepu8_epi16(_mm256_extracti128_si256(sub, 1));
                const __m256i a_lo = _mm256_sub_epi16(lo, slo);
                const __m256i a_hi = _mm256_sub_epi16(hi, shi);

                /* q8 stays signed; maddubs needs (u8, s8), so widen q8 too and
                 * use plain madd_epi16 (i16 x i16 -> i32) with the scale folded
                 * in: 16-bit products of |a|<=4 and |q|<=127 never overflow. */
                const void * q8g = (const void*)(q8 + h * 128 + g * 32);
                const __m256i qlo = _mm256_cvtepi8_epi16(_mm_loadu_si128((const __m128i*)q8g));
                const __m256i qhi = _mm256_cvtepi8_epi16(_mm_loadu_si128((const __m128i*)((const int8_t*)q8g + 16)));

                /* group index = h*8 + 2g (+1 for the high half) */
                const __m256i slo32 = _mm256_set1_epi32(scv[h * 8 + 2 * g]);
                const __m256i shi32 = _mm256_set1_epi32(scv[h * 8 + 2 * g + 1]);

                /* madd_epi16: i16 x i16 -> i32 pair sums (a is signed, q is
                 * signed, |a| <= 4 and |q| <= 127 so nothing overflows), then
                 * scale the i32 lanes. */
                __m256i pl = _mm256_madd_epi16(a_lo, qlo);
                pl = _mm256_mullo_epi32(pl, slo32);
                __m256i ph = _mm256_madd_epi16(a_hi, qhi);
                ph = _mm256_mullo_epi32(ph, shi32);
                sumi = _mm256_add_epi32(sumi, _mm256_add_epi32(pl, ph));
            }
        }

        acc = _mm256_fmadd_ps(_mm256_set1_ps(d), _mm256_cvtepi32_ps(sumi), acc);
    }

    const __m128 lo = _mm256_castps256_ps128(acc);
    const __m128 hi = _mm256_extractf128_ps(acc, 1);
    __m128 t = _mm_add_ps(lo, hi);
    t = _mm_hadd_ps(t, t);
    t = _mm_hadd_ps(t, t);
    return _mm_cvtss_f32(t);
#else
    return k256_q3_k_q8_k_scalar(n, vx, vy);
#endif
}

/* ------------------------------------------------------------------ */
/* Q4_K x Q8_K   (144-byte blocks)                                     */
/* ------------------------------------------------------------------ */

static float k256_q4_k_q8_k_scalar(int n, const uint8_t * vx, const uint8_t * vy)
{
    float sumf = 0.0f;
    const int nb = n / 256;
    for (int i = 0; i < nb; i++) {
        const uint8_t * x = vx + (size_t)i * 144;
        const uint8_t * y = vy + (size_t)i * 292;
        int8_t a8[256];
        k256_q4_aux8(x, a8);
        uint8_t sm[16];
        k256_unpack_scale_min(x + 4, sm);
        int32_t sumi = 0;
        for (int j = 0; j < 16; j++)
            sumi += k256_q8k_bsum(y, j) * (int32_t)sm[8 + j / 2];
        int32_t scv[16];
        for (int g = 0; g < 16; g++)
            scv[g] = (int32_t)sm[g / 2]; /* one 6-bit scale per 32 elements */
        const int32_t isum = k256_dot256(a8, y + 4, scv);
        const float dy = k256_q8k_d(y);
        const float d    = dy * k256_f16_at(x, 0);
        const float dmin = dy * k256_f16_at(x, 2);
        sumf += d * (float)isum - dmin * (float)sumi;
    }
    return sumf;
}

static float k256_q4_k_q8_k_avx2(int n, const uint8_t * vx, const uint8_t * vy)
{
#if defined(__AVX2__)
    /* Vectorised form of k256_q4_k_q8_k_scalar.  The scalar path materialises
     * a8[256] on the stack and then runs k256_dot256 (16 groups x 16 scalar
     * multiplies) -- the single hottest symbol in the whole profile (57% of
     * decode samples).  Everything it does fits in registers:
     *
     *   a8[c*64 + l]      = q4[c*32 + l] & 0x0F        (low nibble, unsigned)
     *   a8[c*64 + 32 + l] = q4[c*32 + l] >> 4          (high nibble, unsigned)
     *
     * and the weights are UNSIGNED 0..15 while the activation is SIGNED i8, so
     * `_mm256_maddubs_epi16` (u8 x s8 -> i16 pair sums) is an exact match -- no
     * bias correction needed.  The per-32-element scale (sm[g/2]) is then
     * multiplied in with `_mm256_madd_epi16`, whose i16 x i16 -> i32 lanes let
     * the scale ride along inside the integer accumulation.  That is what makes
     * this cheap: there is NO per-group horizontal reduction, only one
     * 8-lane reduce at the very end.
     *
     * Order of operations: the scalar path does sum_g sc[g] * sum_l a*l; this
     * does sum over all 8 lanes of sc[g]*product independently and sums the
     * i32 lanes at the end.  Integers are exact, so the two agree bit-for-bit
     * as long as no lane overflows: |a8| <= 15, |q8| <= 127 -> pair sum
     * <= 2*15*127 = 3810 (fits i16), and per lane 8 groups * 2 pairs * 3810
     * * scale <= 8*2*3810*63 < 3.9e6, far inside i32.  Verified against the
     * scalar oracle in src/simd_tests256.zig.
     */
    const __m256i m4 = _mm256_set1_epi8(0x0F);
    __m256 acc = _mm256_setzero_ps();
    __m128 acc_m = _mm_setzero_ps();
    const int nb = n / 256;

    for (int i = 0; i < nb; i++) {
        const uint8_t * x = vx + (size_t)i * 144;
        const uint8_t * y = vy + (size_t)i * 292;
        const float d    = k256_q8k_d(y) * k256_f16_at(x, 0);
        const float dmin = k256_q8k_d(y) * k256_f16_at(x, 2);

        uint8_t sm[16];
        k256_unpack_scale_min(x + 4, sm);

        /* --- dmin * sum_j bsum[j] * min[j/2] ------------------------- */
        /* bsums: 16 x i16.  min values are sm[8 + j/2], i.e. each min covers
         * two consecutive bsum entries.  hadd pairs bsum[j]+bsum[j+1] so a
         * single madd against the 8 mins gives all 8 contributions. */
        {
            const __m256i q8sums = _mm256_loadu_si256((const __m256i*)(y + 260));
            /* sm[8..16] is exactly 8 bytes of mins -- load 8, not 16. */
            const __m128i mins8 = _mm_loadl_epi64((const __m128i*)(sm + 8));
            const __m128i lo = _mm256_castsi256_si128(q8sums);
            const __m128i hi = _mm256_extracti128_si256(q8sums, 1);
            const __m128i q8s = _mm_hadd_epi16(lo, hi);   /* 8 x (bsum[2k]+bsum[2k+1]) */
            const __m128i mins = _mm_cvtepu8_epi16(mins8); /* 8 x min */
            const __m128i prod = _mm_madd_epi16(mins, q8s);
            acc_m = _mm_fmadd_ps(_mm_set1_ps(-dmin), _mm_cvtepi32_ps(prod), acc_m);
        }

        /* --- scales: 8 distinct 6-bit values, each covering 32 elements --- */
        /* Broadcast each into an i16 vector so one madd covers a whole group. */
        __m256i sumi = _mm256_setzero_si256();
        const int8_t * q8 = (const int8_t*)(y + 4);
        const uint8_t * q4 = x + 16;

        for (int c = 0; c < 4; c++) {
            const __m256i q4bits = _mm256_loadu_si256((const __m256i*)(q4 + c * 32));
            const __m256i q4l = _mm256_and_si256(q4bits, m4);
            const __m256i q4h = _mm256_and_si256(_mm256_srli_epi16(q4bits, 4), m4);

            const int8_t * q8l = q8 + c * 64;
            const int8_t * q8h = q8 + c * 64 + 32;

            /* group index for the low half is 2c, for the high half 2c+1 */
            const __m256i scal = _mm256_set1_epi16((short)sm[2 * c]);
            const __m256i scah = _mm256_set1_epi16((short)sm[2 * c + 1]);

            __m256i pl = _mm256_maddubs_epi16(q4l, _mm256_loadu_si256((const __m256i*)q8l));
            pl = _mm256_madd_epi16(scal, pl);
            __m256i ph = _mm256_maddubs_epi16(q4h, _mm256_loadu_si256((const __m256i*)q8h));
            ph = _mm256_madd_epi16(scah, ph);

            sumi = _mm256_add_epi32(sumi, _mm256_add_epi32(pl, ph));
        }

        acc = _mm256_fmadd_ps(_mm256_set1_ps(d), _mm256_cvtepi32_ps(sumi), acc);
    }

    /* horizontal reductions */
    acc_m = _mm_add_ps(acc_m, _mm_movehl_ps(acc_m, acc_m));
    acc_m = _mm_add_ss(acc_m, _mm_movehdup_ps(acc_m));

    const __m128 lo = _mm256_castps256_ps128(acc);
    const __m128 hi = _mm256_extractf128_ps(acc, 1);
    __m128 t = _mm_add_ps(lo, hi);
    t = _mm_hadd_ps(t, t);
    t = _mm_hadd_ps(t, t);
    return _mm_cvtss_f32(t) + _mm_cvtss_f32(acc_m);
#else
    return k256_q4_k_q8_k_scalar(n, vx, vy);
#endif
}

/* ------------------------------------------------------------------ */
/* Q5_K x Q8_K   (176-byte blocks)                                     */
/* ------------------------------------------------------------------ */

static float k256_q5_k_q8_k_scalar(int n, const uint8_t * vx, const uint8_t * vy)
{
    float sumf = 0.0f;
    const int nb = n / 256;
    for (int i = 0; i < nb; i++) {
        const uint8_t * x = vx + (size_t)i * 176;
        const uint8_t * y = vy + (size_t)i * 292;
        int8_t a8[256];
        k256_q5_aux8(x, a8);
        uint8_t sm[16];
        k256_unpack_scale_min(x + 4, sm);
        int32_t sumi = 0;
        for (int j = 0; j < 16; j++)
            sumi += k256_q8k_bsum(y, j) * (int32_t)sm[8 + j / 2];
        int32_t scv[16];
        for (int g = 0; g < 16; g++)
            scv[g] = (int32_t)sm[g / 2];
        const int32_t isum = k256_dot256(a8, y + 4, scv);
        const float dy = k256_q8k_d(y);
        const float d    = dy * k256_f16_at(x, 0);
        const float dmin = dy * k256_f16_at(x, 2);
        sumf += d * (float)isum - dmin * (float)sumi;
    }
    return sumf;
}

static float k256_q5_k_q8_k_avx2(int n, const uint8_t * vx, const uint8_t * vy)
{
#if defined(__AVX2__)
    /* Same structure as the Q4_K vector path above, plus the 5th bit:
     *
     *   a8[c*64 + l]      = (q4[c*32+l] & 0x0F) + (hm[l] & b0 ? 16 : 0)
     *   a8[c*64 + 32 + l] = (q4[c*32+l] >> 4)  + (hm[l] & b1 ? 16 : 0)
     *
     * with b0 = 1<<(2c), b1 = 1<<(2c+1).  The bit test collapses to a compare
     * and a masked OR of 16, which keeps the values in 0..31 (still unsigned,
     * so `maddubs` stays exact: 2*31*127 = 7874 fits i16).
     */
    const __m256i m4 = _mm256_set1_epi8(0x0F);
    const __m256i v16 = _mm256_set1_epi8(16);
    __m256 acc = _mm256_setzero_ps();
    __m128 acc_m = _mm_setzero_ps();
    const int nb = n / 256;

    for (int i = 0; i < nb; i++) {
        const uint8_t * x = vx + (size_t)i * 176;
        const uint8_t * y = vy + (size_t)i * 292;
        const float d    = k256_q8k_d(y) * k256_f16_at(x, 0);
        const float dmin = k256_q8k_d(y) * k256_f16_at(x, 2);

        uint8_t sm[16];
        k256_unpack_scale_min(x + 4, sm);

        {
            const __m256i q8sums = _mm256_loadu_si256((const __m256i*)(y + 260));
            const __m128i mins8 = _mm_loadl_epi64((const __m128i*)(sm + 8));
            const __m128i lo = _mm256_castsi256_si128(q8sums);
            const __m128i hi = _mm256_extracti128_si256(q8sums, 1);
            const __m128i q8s = _mm_hadd_epi16(lo, hi);
            const __m128i mins = _mm_cvtepu8_epi16(mins8);
            const __m128i prod = _mm_madd_epi16(mins, q8s);
            acc_m = _mm_fmadd_ps(_mm_set1_ps(-dmin), _mm_cvtepi32_ps(prod), acc_m);
        }

        __m256i sumi = _mm256_setzero_si256();
        const int8_t * q8 = (const int8_t*)(y + 4);
        const uint8_t * q4 = x + 48;
        const uint8_t * hm = x + 16;

        for (int c = 0; c < 4; c++) {
            const __m256i q4bits = _mm256_loadu_si256((const __m256i*)(q4 + c * 32));
            const __m256i hmb = _mm256_loadu_si256((const __m256i*)hm);

            /* (hm & bit) ? 16 : 0  ->  cmpeq gives 0xFF where the bit is
             * clear, so ANDNOT against v16 selects 16 only where it is set. */
            const __m256i b0 = _mm256_cmpeq_epi8(_mm256_and_si256(hmb, _mm256_set1_epi8((char)(1 << (2 * c)))), _mm256_setzero_si256());
            const __m256i b1 = _mm256_cmpeq_epi8(_mm256_and_si256(hmb, _mm256_set1_epi8((char)(1 << (2 * c + 1)))), _mm256_setzero_si256());
            const __m256i add0 = _mm256_andnot_si256(b0, v16);
            const __m256i add1 = _mm256_andnot_si256(b1, v16);

            const __m256i q4l = _mm256_add_epi8(_mm256_and_si256(q4bits, m4), add0);
            const __m256i q4h = _mm256_add_epi8(_mm256_and_si256(_mm256_srli_epi16(q4bits, 4), m4), add1);

            const __m256i scal = _mm256_set1_epi16((short)sm[2 * c]);
            const __m256i scah = _mm256_set1_epi16((short)sm[2 * c + 1]);

            __m256i pl = _mm256_maddubs_epi16(q4l, _mm256_loadu_si256((const __m256i*)(q8 + c * 64)));
            pl = _mm256_madd_epi16(scal, pl);
            __m256i ph = _mm256_maddubs_epi16(q4h, _mm256_loadu_si256((const __m256i*)(q8 + c * 64 + 32)));
            ph = _mm256_madd_epi16(scah, ph);

            sumi = _mm256_add_epi32(sumi, _mm256_add_epi32(pl, ph));
        }

        acc = _mm256_fmadd_ps(_mm256_set1_ps(d), _mm256_cvtepi32_ps(sumi), acc);
    }

    acc_m = _mm_add_ps(acc_m, _mm_movehl_ps(acc_m, acc_m));
    acc_m = _mm_add_ss(acc_m, _mm_movehdup_ps(acc_m));

    const __m128 lo = _mm256_castps256_ps128(acc);
    const __m128 hi = _mm256_extractf128_ps(acc, 1);
    __m128 t = _mm_add_ps(lo, hi);
    t = _mm_hadd_ps(t, t);
    t = _mm_hadd_ps(t, t);
    return _mm_cvtss_f32(t) + _mm_cvtss_f32(acc_m);
#else
    return k256_q5_k_q8_k_scalar(n, vx, vy);
#endif
}

/* ------------------------------------------------------------------ */
/* Q6_K x Q8_K   (210-byte blocks)                                     */
/* ------------------------------------------------------------------ */

static float k256_q6_k_q8_k_scalar(int n, const uint8_t * vx, const uint8_t * vy)
{
    float sumf = 0.0f;
    const int nb = n / 256;
    for (int i = 0; i < nb; i++) {
        const uint8_t * x = vx + (size_t)i * 210;
        const uint8_t * y = vy + (size_t)i * 292;
        int8_t a8[256];
        k256_q6_aux8(x, a8);
        const int8_t * sc = (const int8_t *)(x + 192); /* 16 signed bytes */
        int32_t scv[16];
        for (int g = 0; g < 16; g++)
            scv[g] = (int32_t)sc[g];
        const int32_t isum = k256_dot256(a8, y + 4, scv);
        const float dy = k256_q8k_d(y);
        sumf += k256_f16_at(x, 208) * dy * (float)isum;
    }
    return sumf;
}

static float k256_q6_k_q8_k_avx2(int n, const uint8_t * vx, const uint8_t * vy)
{
#if defined(__AVX2__)
    /* Q6_K layout (matches k256_q6_aux8 exactly):
     *   a8[h*128 + l +  0] = lo(q4[h*64 + l     ]) | ((qh[h*32+l] >> 0) & 3) << 4
     *   a8[h*128 + l + 32] = lo(q4[h*64 + l + 32]) | ((qh[h*32+l] >> 2) & 3) << 4
     *   a8[h*128 + l + 64] = hi(q4[h*64 + l     ]) | ((qh[h*32+l] >> 4) & 3) << 4
     *   a8[h*128 + l + 96] = hi(q4[h*64 + l + 32]) | ((qh[h*32+l] >> 6) & 3) << 4
     * and then the scalar path subtracts 32 from every value.
     *
     * Rather than materialise the signed a8[256], keep the 0..63 unsigned and
     * let `maddubs` do u8 x s8, then correct the -32 offset analytically:
     *   sum_e (v_e - 32) * q_e * s_e        (= the scalar answer)
     * = sum_e v_e * q_e * s_e  -  32 * sum_e q_e * s_e
     * The second term is constant per block and expands to
     *   32 * sum_g s_g * bsum_g
     * where bsum_g is the 16-element activation sum of group g -- the same
     * bsums[] the Q8_K block already carries.  So the correction is one
     * madd + one shift, no extra loads.
     *
     * Range check: v <= 63, |q| <= 127 -> pair sum <= 2*63*127 = 16002 fits
     * i16; per i32 lane 8 pairs * 16002 * |s|<=127 < 1.7e7 fits i32.
     */
    const __m256i m3  = _mm256_set1_epi8(3);
    const __m256i m15 = _mm256_set1_epi8(15);
    const __m256i m12 = _mm256_set1_epi8(12);
    const __m256i m48 = _mm256_set1_epi8(48);
    const __m256i m64 = _mm256_set1_epi8((char)0xC0);

    __m256 acc = _mm256_setzero_ps();
    const int nb = n / 256;

    for (int i = 0; i < nb; i++) {
        const uint8_t * x = vx + (size_t)i * 210;
        const uint8_t * y = vy + (size_t)i * 292;
        const float d = k256_f16_at(x, 208) * k256_q8k_d(y);

        const int8_t * scales = (const int8_t*)(x + 192);

        /* -32 correction: 32 * sum_g s_g * bsum_g, folded into a float add. */
        __m256 corr = _mm256_setzero_ps();
        {
            const __m256i q8sums = _mm256_loadu_si256((const __m256i*)(y + 260)); /* 16 x i16 */
            const __m256i sc16 = _mm256_cvtepi8_epi16(_mm_loadu_si128((const __m128i*)scales));
            const __m256i prod = _mm256_madd_epi16(q8sums, sc16); /* 8 x i32 pair sums */
            /* scale by 32, then accumulate; d applied after the loop */
            corr = _mm256_cvtepi32_ps(_mm256_slli_epi32(prod, 5));
        }

        __m256i sumi = _mm256_setzero_si256();
        const int8_t * q8 = (const int8_t*)(y + 4);
        const uint8_t * q4 = x;        /* 128 B */
        const uint8_t * qh = x + 128;  /*  64 B */

        for (int h = 0; h < 2; h++) {
            const __m256i qh4 = _mm256_loadu_si256((const __m256i*)(qh + h * 32));

            /* the four 2-bit high parts, shifted into the 4..7 bit positions */
            const __m256i h0 = _mm256_slli_epi16(_mm256_and_si256(qh4, m3), 4);
            const __m256i h1 = _mm256_slli_epi16(_mm256_and_si256(qh4, m12), 2);
            const __m256i h2 = _mm256_and_si256(qh4, m48);
            const __m256i h3 = _mm256_srli_epi16(_mm256_and_si256(qh4, m64), 2);

            const __m256i qa = _mm256_loadu_si256((const __m256i*)(q4 + h * 64));
            const __m256i qb = _mm256_loadu_si256((const __m256i*)(q4 + h * 64 + 32));

            const __m256i v0 = _mm256_or_si256(_mm256_and_si256(qa, m15), h0);
            const __m256i v1 = _mm256_or_si256(_mm256_and_si256(qb, m15), h1);
            const __m256i v2 = _mm256_or_si256(_mm256_and_si256(_mm256_srli_epi16(qa, 4), m15), h2);
            const __m256i v3 = _mm256_or_si256(_mm256_and_si256(_mm256_srli_epi16(qb, 4), m15), h3);

            const int8_t * q8h = q8 + h * 128;

            /* Group g covers 16 elements, so one 32-element maddubs result
             * needs TWO scales: sc[2g] for elements 0..15, sc[2g+1] for 16..31.
             * `_mm256_set_m128i(hi, lo)` puts sc[2g+1] in the high 128-bit half,
             * which is exactly where maddubs puts elements 16..31. */
            #define Q6_SP(gi) \
                _mm256_setr_epi16((short)scales[(gi)+0], (short)scales[(gi)+0], \
                                  (short)scales[(gi)+0], (short)scales[(gi)+0], \
                                  (short)scales[(gi)+0], (short)scales[(gi)+0], \
                                  (short)scales[(gi)+0], (short)scales[(gi)+0], \
                                  (short)scales[(gi)+1], (short)scales[(gi)+1], \
                                  (short)scales[(gi)+1], (short)scales[(gi)+1], \
                                  (short)scales[(gi)+1], (short)scales[(gi)+1], \
                                  (short)scales[(gi)+1], (short)scales[(gi)+1])
            const __m256i sp0 = Q6_SP(h * 8 + 0);
            const __m256i sp1 = Q6_SP(h * 8 + 2);
            const __m256i sp2 = Q6_SP(h * 8 + 4);
            const __m256i sp3 = Q6_SP(h * 8 + 6);
            #undef Q6_SP

            /* maddubs: u8 (0..63) x s8 -> i16 pair sums.  madd: multiply each
             * pair by its group scale and widen to i32. */
            const __m256i p0 = _mm256_madd_epi16(sp0, _mm256_maddubs_epi16(v0, _mm256_loadu_si256((const __m256i*)(q8h + 0))));
            const __m256i p1 = _mm256_madd_epi16(sp1, _mm256_maddubs_epi16(v1, _mm256_loadu_si256((const __m256i*)(q8h + 32))));
            const __m256i p2 = _mm256_madd_epi16(sp2, _mm256_maddubs_epi16(v2, _mm256_loadu_si256((const __m256i*)(q8h + 64))));
            const __m256i p3 = _mm256_madd_epi16(sp3, _mm256_maddubs_epi16(v3, _mm256_loadu_si256((const __m256i*)(q8h + 96))));

            sumi = _mm256_add_epi32(sumi, _mm256_add_epi32(
                _mm256_add_epi32(p0, p1), _mm256_add_epi32(p2, p3)));
        }

        /* sumi already carries the per-group scales; apply d and subtract the
         * -32 correction (also scaled by d). */
        const __m256 vd = _mm256_set1_ps(d);
        acc = _mm256_fmadd_ps(vd, _mm256_cvtepi32_ps(sumi), acc);
        acc = _mm256_fnmadd_ps(vd, corr, acc);
    }

    const __m128 lo = _mm256_castps256_ps128(acc);
    const __m128 hi = _mm256_extractf128_ps(acc, 1);
    __m128 t = _mm_add_ps(lo, hi);
    t = _mm_hadd_ps(t, t);
    t = _mm_hadd_ps(t, t);
    return _mm_cvtss_f32(t);
#else
    return k256_q6_k_q8_k_scalar(n, vx, vy);
#endif
}

/* ------------------------------------------------------------------ */
/* Q8_K activation quantizer (port of quantizeQ8_K in act.zig)        */
/* ------------------------------------------------------------------ */

void quantize_q8_K(int n, const float * x, uint8_t * out)
{
    for (int i = 0; i < n; i += 256) {
        uint8_t * o = out + (size_t)(i / 256) * 292;
        float amax = 0.0f;
        for (int j = 0; j < 256; j++) {
            const float a = x[i + j] < 0.0f ? -x[i + j] : x[i + j];
            if (a > amax) amax = a;
        }
        const float scale = amax > 0.0f ? amax / 127.0f : 1.0f;
        const float inv = amax > 0.0f ? 1.0f / scale : 0.0f;
        memcpy(o, &scale, sizeof scale);
        int16_t gsum[16] = { 0 };
        for (int k = 0; k < 256; k++) {
            float r = roundf(x[i + k] * inv);
            // Clamp to [-127, 127], not [-128, 127]: IQ kernels negate q8 via
            // _mm256_sign_epi8, and negating -128 overflows back to -128 (the
            // sign flips wrongly) -- x86 psubb cannot represent +128.  Using
            // -127 as the lower bound is what llama.cpp does (iscale = -127.f /
            // max).  With scale = amax/127 this only matters for a rounding
            // edge case, but that edge case silently corrupts a whole block.
            if (r < -127.0f) r = -127.0f;
            if (r > 127.0f) r = 127.0f;
            if (!(r >= -127.0f)) r = 127.0f; /* NaN/inf guard (Zig intFromFloat saturates) */
            const int8_t q = (int8_t)(int32_t)r;
            o[4 + k] = (uint8_t)q;
            gsum[k / 16] = (int16_t)(gsum[k / 16] + (int32_t)q);
        }
        for (int g = 0; g < 16; g++)
            memcpy(o + 260 + g * 2, &gsum[g], sizeof gsum[g]);
    }
}

/* ------------------------------------------------------------------ */
/* public dispatch                                                     */
/* ------------------------------------------------------------------ */

float q2_k_q8_k(int n, const void * vx, const void * vy)
{
    return cpu_has_avx2()
        ? k256_q2_k_q8_k_avx2(n, (const uint8_t *)vx, (const uint8_t *)vy)
        : k256_q2_k_q8_k_scalar(n, (const uint8_t *)vx, (const uint8_t *)vy);
}

float q3_k_q8_k(int n, const void * vx, const void * vy)
{
    return cpu_has_avx2()
        ? k256_q3_k_q8_k_avx2(n, (const uint8_t *)vx, (const uint8_t *)vy)
        : k256_q3_k_q8_k_scalar(n, (const uint8_t *)vx, (const uint8_t *)vy);
}

float q4_k_q8_k(int n, const void * vx, const void * vy)
{
    return cpu_has_avx2()
        ? k256_q4_k_q8_k_avx2(n, (const uint8_t *)vx, (const uint8_t *)vy)
        : k256_q4_k_q8_k_scalar(n, (const uint8_t *)vx, (const uint8_t *)vy);
}

float q5_k_q8_k(int n, const void * vx, const void * vy)
{
    return cpu_has_avx2()
        ? k256_q5_k_q8_k_avx2(n, (const uint8_t *)vx, (const uint8_t *)vy)
        : k256_q5_k_q8_k_scalar(n, (const uint8_t *)vx, (const uint8_t *)vy);
}

float q6_k_q8_k(int n, const void * vx, const void * vy)
{
    return cpu_has_avx2()
        ? k256_q6_k_q8_k_avx2(n, (const uint8_t *)vx, (const uint8_t *)vy)
        : k256_q6_k_q8_k_scalar(n, (const uint8_t *)vx, (const uint8_t *)vy);
}

/* ------------------------------------------------------------------ */
/* TQ2_0 x Q8_K   (66-byte blocks)                                     */
/* 2 bits per element: value = ((qs[k] >> 2*l) & 3) - 1                */
/* ------------------------------------------------------------------ */

static float k256_tq2_0_q8_k_scalar(int n, const uint8_t * vx, const uint8_t * vy)
{
    float sumf = 0.0f;
    const int nb = n / 256;
    for (int i = 0; i < nb; i++) {
        const uint8_t * x = vx + (size_t)i * 66;
        const uint8_t * y = vy + (size_t)i * 292;
        const uint8_t * q8 = y + 4;  /* qs[256] */
        int32_t sumi = 0;
        /* Process 2 groups of 128 elements each */
        for (int k = 0; k < 2; k++) {
            for (int l = 0; l < 4; l++) {
                for (int e = 0; e < 32; e++) {
                    int32_t v = (int32_t)((x[k * 32 + e] >> (2 * l)) & 0x03) - 1;
                    sumi += (int32_t)(int8_t)q8[k * 128 + l * 32 + e] * v;
                }
            }
        }
        const float dy = k256_q8k_d(y);
        sumf += (float)sumi * dy * k256_f16_at(x, 64);
    }
    return sumf;
}

static float k256_tq2_0_q8_k_avx2(int n, const uint8_t * vx, const uint8_t * vy)
{
    return k256_tq2_0_q8_k_scalar(n, vx, vy);
}

/* ------------------------------------------------------------------ */
/* TQ1_0 x Q8_K   (54-byte blocks)                                     */
/* Base-5 packing: value = (((qs[j] * pow3[l]) & 0xFF) * 3 >> 8) - 1   */
/* ------------------------------------------------------------------ */


static float k256_tq1_0_q8_k_scalar(int n, const uint8_t * vx, const uint8_t * vy)
{
    float sumf = 0.0f;
    const int nb = n / 256;
    for (int i = 0; i < nb; i++) {
        const uint8_t * x = vx + (size_t)i * 54;
        const uint8_t * y = vy + (size_t)i * 292;
        const uint8_t * q8 = y + 4;  /* qs[256] */
        int32_t sum = 0;
        /* Bytes 0..31: 32 bytes -> 5 groups of 32 elements */
        for (int m = 0; m < 32; m++) {
            for (int l = 0; l < 5; l++) {
                uint8_t q = x[m] * iq_pow3[l];
                uint16_t xi = ((uint16_t)q * 3) >> 8;
                sum += (int32_t)(int8_t)q8[l * 32 + m] * ((int32_t)xi - 1);
            }
        }
        /* Bytes 32..47: 16 bytes -> 5 groups of 16 elements */
        for (int m = 0; m < 16; m++) {
            for (int l = 0; l < 5; l++) {
                uint8_t q = x[32 + m] * iq_pow3[l];
                uint16_t xi = ((uint16_t)q * 3) >> 8;
                sum += (int32_t)(int8_t)q8[160 + l * 16 + m] * ((int32_t)xi - 1);
            }
        }
        /* Bytes 48..51: 4 bytes -> 4 groups of 4 elements */
        for (int j = 0; j < 4; j++) {
            for (int l = 0; l < 4; l++) {
                uint8_t q = x[48 + j] * iq_pow3[l];
                uint16_t xi = ((uint16_t)q * 3) >> 8;
                sum += (int32_t)(int8_t)q8[240 + l * 4 + j] * ((int32_t)xi - 1);
            }
        }
        const float dy = k256_q8k_d(y);
        sumf += (float)sum * k256_f16_at(x, 52) * dy;
    }
    return sumf;
}

static float k256_tq1_0_q8_k_avx2(int n, const uint8_t * vx, const uint8_t * vy)
{
    return k256_tq1_0_q8_k_scalar(n, vx, vy);
}

/* ------------------------------------------------------------------ */
/* public dispatch for TQ kernels                                      */
/* ------------------------------------------------------------------ */

float tq2_0_q8_k(int n, const void * vx, const void * vy)
{
    return cpu_has_avx2()
        ? k256_tq2_0_q8_k_avx2(n, (const uint8_t *)vx, (const uint8_t *)vy)
        : k256_tq2_0_q8_k_scalar(n, (const uint8_t *)vx, (const uint8_t *)vy);
}

float tq1_0_q8_k(int n, const void * vx, const void * vy)
{
    return cpu_has_avx2()
        ? k256_tq1_0_q8_k_avx2(n, (const uint8_t *)vx, (const uint8_t *)vy)
        : k256_tq1_0_q8_k_scalar(n, (const uint8_t *)vx, (const uint8_t *)vy);
}
