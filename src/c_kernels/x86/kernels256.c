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
#include <string.h>
#include <math.h>

/* ------------------------------------------------------------------ */
/* shared helpers                                                      */
/* ------------------------------------------------------------------ */

/* CPUID wrapper for x86-64 */
static inline void k256_cpuid(int eax, int ecx, int* regs) {
    __asm__ __volatile__ (
        "cpuid"
        : "=a"(regs[0]), "=b"(regs[1]), "=c"(regs[2]), "=d"(regs[3])
        : "a"(eax), "c"(ecx)
    );
}

/* XGETBV wrapper */
static inline unsigned long long k256_xgetbv(int index) {
    unsigned int eax, edx;
    __asm__ __volatile__ (
        "xgetbv"
        : "=a"(eax), "=d"(edx)
        : "c"(index)
    );
    return ((unsigned long long)edx << 32) | eax;
}

static int k256_cpu_has_avx2(void) {
    int regs[4];
    // Check AVX2: CPUID leaf 7, subleaf 0, EBX bit 5
    k256_cpuid(7, 0, regs);
    int has_avx2 = (regs[1] >> 5) & 1;
    // Also need to check OS support via XCR0
    if (has_avx2) {
        unsigned long long xcr0 = k256_xgetbv(0);
        // Check XMM (bit 1) and YMM (bit 2) state
        has_avx2 = ((xcr0 >> 1) & 1) && ((xcr0 >> 2) & 1);
    }
    return has_avx2;
}

/* IEEE 754 binary16 -> float, exact (mirrors util.f16At in kern256.zig). */
static float k256_f16_to_f32(uint16_t h)
{
    const uint32_t s = ((uint32_t)h & 0x8000u) << 16;
    const uint32_t e = (uint32_t)(h >> 10) & 0x1Fu;
    const uint32_t m = (uint32_t)h & 0x3FFu;
    if (e == 0) {
        if (m == 0) return 0.0f;
        /* subnormal: m * 2^-24 (exact in f32). */
        return (float)((double)m * 0x1.0p-24);
    }
    uint32_t bits;
    if (e == 0x1Fu) {
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

/* AVX2: no int8 multiply intrinsic; delegate to the (oracle-exact) scalar. */
static float k256_q2_k_q8_k_avx2(int n, const uint8_t * vx, const uint8_t * vy)
{
    return k256_q2_k_q8_k_scalar(n, vx, vy);
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
    return k256_q3_k_q8_k_scalar(n, vx, vy);
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
    return k256_q4_k_q8_k_scalar(n, vx, vy);
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
    return k256_q5_k_q8_k_scalar(n, vx, vy);
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
    return k256_q6_k_q8_k_scalar(n, vx, vy);
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
            if (r < -128.0f) r = -128.0f;
            if (r > 127.0f) r = 127.0f;
            if (!(r >= -128.0f)) r = 127.0f; /* NaN/inf guard (Zig intFromFloat saturates) */
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
    return k256_cpu_has_avx2()
        ? k256_q2_k_q8_k_avx2(n, (const uint8_t *)vx, (const uint8_t *)vy)
        : k256_q2_k_q8_k_scalar(n, (const uint8_t *)vx, (const uint8_t *)vy);
}

float q3_k_q8_k(int n, const void * vx, const void * vy)
{
    return k256_cpu_has_avx2()
        ? k256_q3_k_q8_k_avx2(n, (const uint8_t *)vx, (const uint8_t *)vy)
        : k256_q3_k_q8_k_scalar(n, (const uint8_t *)vx, (const uint8_t *)vy);
}

float q4_k_q8_k(int n, const void * vx, const void * vy)
{
    return k256_cpu_has_avx2()
        ? k256_q4_k_q8_k_avx2(n, (const uint8_t *)vx, (const uint8_t *)vy)
        : k256_q4_k_q8_k_scalar(n, (const uint8_t *)vx, (const uint8_t *)vy);
}

float q5_k_q8_k(int n, const void * vx, const void * vy)
{
    return k256_cpu_has_avx2()
        ? k256_q5_k_q8_k_avx2(n, (const uint8_t *)vx, (const uint8_t *)vy)
        : k256_q5_k_q8_k_scalar(n, (const uint8_t *)vx, (const uint8_t *)vy);
}

float q6_k_q8_k(int n, const void * vx, const void * vy)
{
    return k256_cpu_has_avx2()
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

static const uint8_t k256_pow3[6] = { 1, 3, 9, 27, 81, 243 };

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
                uint8_t q = x[m] * k256_pow3[l];
                uint16_t xi = ((uint16_t)q * 3) >> 8;
                sum += (int32_t)(int8_t)q8[l * 32 + m] * ((int32_t)xi - 1);
            }
        }
        /* Bytes 32..47: 16 bytes -> 5 groups of 16 elements */
        for (int m = 0; m < 16; m++) {
            for (int l = 0; l < 5; l++) {
                uint8_t q = x[32 + m] * k256_pow3[l];
                uint16_t xi = ((uint16_t)q * 3) >> 8;
                sum += (int32_t)(int8_t)q8[160 + l * 16 + m] * ((int32_t)xi - 1);
            }
        }
        /* Bytes 48..51: 4 bytes -> 4 groups of 4 elements */
        for (int j = 0; j < 4; j++) {
            for (int l = 0; l < 4; l++) {
                uint8_t q = x[48 + j] * k256_pow3[l];
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
    return k256_cpu_has_avx2()
        ? k256_tq2_0_q8_k_avx2(n, (const uint8_t *)vx, (const uint8_t *)vy)
        : k256_tq2_0_q8_k_scalar(n, (const uint8_t *)vx, (const uint8_t *)vy);
}

float tq1_0_q8_k(int n, const void * vx, const void * vy)
{
    return k256_cpu_has_avx2()
        ? k256_tq1_0_q8_k_avx2(n, (const uint8_t *)vx, (const uint8_t *)vy)
        : k256_tq1_0_q8_k_scalar(n, (const uint8_t *)vx, (const uint8_t *)vy);
}
