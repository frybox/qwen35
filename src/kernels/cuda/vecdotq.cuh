// vecdotq.cuh -- block dot products for EVERY quant type the CPU backend
// supports (device side).  One function per (weight type, activation type)
// pair, each folding one block into an `acc` float.
//
// ---------------------------------------------------------------------------
// WHY THE ACCUMULATION ORDER IS MIRRORED FROM src/kernels/x86, NOT UPSTREAM
// ---------------------------------------------------------------------------
// The GPU result must agree with the CPU backend's, so the float expression of
// every block is copied from this repo's CPU kernels rather than from
// ggml/src/ggml-cuda/vecdotq.cuh (whose warp-level tree reductions a per-row
// thread cannot reproduce anyway -- see the mmvq.cu header for why we do not
// use a warp reduction at all).
//
// Two rules follow and are obeyed everywhere below:
//   1. Integer domain first.  Every per-block integer sum is computed in int32
//      (dp4a / shifts) exactly like the CPU `*_scalar` reference, so the integer
//      intermediate is bit-identical.  No float op happens before
//      `scale * (float)isum`.
//   2. No fma contraction.  This host's default (highest) CPU tier is the
//      AVX-512 VNNI path (kernels512.c), whose form is
//      `sumf += dx*dy*(float)sumi` -- multiply rounded, then multiply, then add,
//      NOT the AVX2 tier's `fma(d, cvtepi32(sumi), acc)`.  Plain `*`/`+` is
//      unsafe under nvcc's default contraction, hence `__fmul_rn`/`__fadd_rn`
//      throughout (same reasoning as the Q8_0 note in the file this replaces).
//
// ---------------------------------------------------------------------------
// MANIFEST (checked against src/backend_cpu.zig loadLinear / dequantRowFnFor,
// the CPU backend's own quant support list):
//   32-element native blocks, Q8_0 activation (34 B / 32 elems):
//     q8_0 q4_0 q4_1 q5_0 q5_1 iq4_nl
//   256-element native blocks, Q8_K activation (292 B / 256 elems):
//     q2_k q3_k q4_k q5_k q6_k
//     iq1_s iq1_m iq2_xxs iq2_xs iq2_s iq3_xxs iq3_s iq4_xs
//   plus f32 weights, which use a separate kernel in mmvq.cu.
// The 32-element types are framed into 256-element GGUF row-blocks on disk, but
// the dot loops below still walk them in 32-element units, because that is the
// unit their activation (Q8_0) carries.
#ifndef Q35_CUDA_VECDOTQ_CUH
#define Q35_CUDA_VECDOTQ_CUH

#include "common.cuh"
#include "iq_tables.cuh"

// ---------------------------------------------------------------------------
// Block geometry: NATIVE block sizes, not the GGUF row-block framing (mmvq.cu
// and convert.cu loop over row blocks and call these per native block).
// ---------------------------------------------------------------------------
#define Q35_QK_K 256   // K-quant / IQ super-block size
#define Q35_QK8K 256   // Q8_K block size (activation side of the K/IQ family)

// Native block byte sizes (ggml-quants.c: sizeof(block_*)).
#define Q35_BYTES_Q8_0    34
#define Q35_BYTES_Q4_0    18
#define Q35_BYTES_Q4_1    20
#define Q35_BYTES_Q5_0    22
#define Q35_BYTES_Q5_1    24
#define Q35_BYTES_IQ4_NL  18
#define Q35_BYTES_Q2_K    84
#define Q35_BYTES_Q3_K    110
#define Q35_BYTES_Q4_K    144
#define Q35_BYTES_Q5_K    176
#define Q35_BYTES_Q6_K    210
#define Q35_BYTES_IQ1_S   50
#define Q35_BYTES_IQ1_M   56
#define Q35_BYTES_IQ2_XXS 66
#define Q35_BYTES_IQ2_XS  74
#define Q35_BYTES_IQ2_S   82
#define Q35_BYTES_IQ3_XXS 98
#define Q35_BYTES_IQ3_S   110
#define Q35_BYTES_IQ4_XS  136
#define Q35_BYTES_Q8_K    292  // 4 + 256 + 32

#ifdef Q35_HOST_REF
// Verification harness only (/tmp, never built into the library): the same
// functions compiled for the host so one source of truth covers both sides.
#define Q35_DEV inline
#else
#define Q35_DEV __device__ __forceinline__
#endif

// ===========================================================================
// small shared helpers
// ===========================================================================

// little-endian loads (block payloads are only 2-byte aligned)
Q35_DEV unsigned int q35_ld_u32(const unsigned char *p) {
    return (unsigned int)p[0] | ((unsigned int)p[1] << 8) |
           ((unsigned int)p[2] << 16) | ((unsigned int)p[3] << 24);
}
Q35_DEV unsigned short q35_ld_u16(const unsigned char *p) {
    return (unsigned short)((unsigned short)p[0] | ((unsigned short)p[1] << 8));
}
Q35_DEV float q35_f16_at(const unsigned char *p, int off) {
    return q35_ld_scale(p + off);  // common.cuh: f16 bits -> f32 (exact)
}
Q35_DEV float q35_ld_f32(const unsigned char *p) {
    return __uint_as_float(q35_ld_u32(p));
}

// len consecutive int8 x int8 dot products.  Integer-only, so the association
// order is irrelevant to the result.
Q35_DEV int q35_dot_i8(const signed char *w, const signed char *a, int len) {
    int s = 0;
    for (int i = 0; i < len; ++i) s += (int)w[i] * (int)a[i];
    return s;
}

// ---------------------------------------------------------------------------
// Q8_K activation block accessors.  Layout (ggml-quants.c: block_q8_K):
//   float d @0 ; int8_t qs[256] @4 ; int16_t bsums[16] @260 ; 292 B total.
// ---------------------------------------------------------------------------
Q35_DEV const signed char *q35_q8k_qs(const unsigned char *y) {
    return (const signed char *)(y + 4);
}
Q35_DEV float q35_q8k_d(const unsigned char *y) { return q35_ld_f32(y); }
Q35_DEV int q35_q8k_bsum(const unsigned char *y, int j) {
    return (int)(short)q35_ld_u16(y + 260 + j * 2);
}

// 16 groups of 16: isum = sum_g sc[g] * sum_l w8[g*16+l] * a8[g*16+l].
// Same shape as src/kernels/x86/kernels256.c:213 k256_dot256 (itself the AVX2
// tier's vpmaddwd/vpmaddubsw decomposition).
Q35_DEV int q35_dot_scaled16(const signed char *w8, const signed char *a8,
                             const int *sc) {
    int isum = 0;
    for (int g = 0; g < 16; ++g) {
        const int sub = q35_dot_i8(w8 + g * 16, a8 + g * 16, 16);
        isum += sc[g] * sub;
    }
    return isum;
}

// ===========================================================================
// 32-element block family (weight) x Q8_0 (activation)
//
// Each function folds one 32-element block into `acc`.
// Reference: src/kernels/x86/kernels32.c:*_scalar.
// ===========================================================================

// kernels32.c:415 k32_q8_0_q8_0_scalar -- sumf += dx*dy*(float)sumi
Q35_DEV float q35_vec_dot_q8_0_q8_0_acc(const unsigned char *w,
                                        const unsigned char *a, float acc) {
    const float dx = q35_f16_at(w, 0);
    const float dy = q35_f16_at(a, 0);
    const int sumi = q35_dot_i8((const signed char *)(w + 2),
                                (const signed char *)(a + 2), 32);
    return __fadd_rn(acc, __fmul_rn(__fmul_rn(dx, dy), (float)sumi));
}

// kernels32.c: k32_q4_0_q8_0_scalar
// Layout: f16 d @0, qs[16] @2.  Low nibble -> elements 0..15, high -> 16..31
// (same convention as dequantQ4_0 in src/quant_dequant.zig).
Q35_DEV float q35_vec_dot_q4_0_q8_0_acc(const unsigned char *w,
                                        const unsigned char *a, float acc) {
    const float dx = q35_f16_at(w, 0);
    const float dy = q35_f16_at(a, 0);
    const signed char *a8 = (const signed char *)(a + 2);
    int sumi = 0;
    for (int j = 0; j < 16; ++j) {
        const int v = (int)w[2 + j];
        sumi += ((v & 0x0F) - 8) * (int)a8[j];
        sumi += ((v >> 4) - 8) * (int)a8[j + 16];
    }
    return __fadd_rn(acc, __fmul_rn(__fmul_rn(dx, dy), (float)sumi));
}

// kernels32.c: k32_q4_1_q8_0_scalar
//   sumf += dx*dy*sumi + mx*dy*sumy   (two rounded products, then added)
Q35_DEV float q35_vec_dot_q4_1_q8_0_acc(const unsigned char *w,
                                        const unsigned char *a, float acc) {
    const float dx = q35_f16_at(w, 0);
    const float mx = q35_f16_at(w, 2);
    const float dy = q35_f16_at(a, 0);
    const signed char *a8 = (const signed char *)(a + 2);
    int sumi = 0, sumy = 0;
    for (int j = 0; j < 16; ++j) {
        const int v = (int)w[4 + j];
        sumi += (v & 0x0F) * (int)a8[j];
        sumi += (v >> 4) * (int)a8[j + 16];
    }
    for (int j = 0; j < 32; ++j) sumy += (int)a8[j];
    acc = __fadd_rn(acc, __fmul_rn(__fmul_rn(dx, dy), (float)sumi));
    return __fadd_rn(acc, __fmul_rn(__fmul_rn(mx, dy), (float)sumy));
}

// kernels32.c:520 k32_q5_0_q8_0_scalar
//   q5 = q4 | (((qh >> j) & 1) << 4) ; sumi += (q5 - 16) * q8[j]
// Layout: d @0 (f16), qh @2 (u32), qs[16] @6.  22 bytes.
Q35_DEV float q35_vec_dot_q5_0_q8_0_acc(const unsigned char *w,
                                        const unsigned char *a, float acc) {
    const float dx = q35_f16_at(w, 0);
    const float dy = q35_f16_at(a, 0);
    const unsigned int qh = q35_ld_u32(w + 2);
    const signed char *a8 = (const signed char *)(a + 2);
    int sumi = 0;
    for (int j = 0; j < 32; ++j) {
        const int q4 = (w[6 + j / 2] >> (4 * (j & 1))) & 0x0F;
        const int q5 = q4 | (int)((qh >> j) & 1u) << 4;
        sumi += (q5 - 16) * (int)a8[j];
    }
    return __fadd_rn(acc, __fmul_rn(__fmul_rn(dx, dy), (float)sumi));
}

// kernels32.c:590 k32_q5_1_q8_0_scalar
//   sumf += dx*dy*sumi + mx*dy*sumy
// Layout: d @0, m @2 (both f16), qh @4 (u32), qs[16] @8.  24 bytes.
Q35_DEV float q35_vec_dot_q5_1_q8_0_acc(const unsigned char *w,
                                        const unsigned char *a, float acc) {
    const float dx = q35_f16_at(w, 0);
    const float mx = q35_f16_at(w, 2);
    const float dy = q35_f16_at(a, 0);
    const unsigned int qh = q35_ld_u32(w + 4);
    const signed char *a8 = (const signed char *)(a + 2);
    int sumi = 0, sumy = 0;
    for (int j = 0; j < 32; ++j) {
        const int q4 = (w[8 + j / 2] >> (4 * (j & 1))) & 0x0F;
        const int q5 = q4 | (int)((qh >> j) & 1u) << 4;
        sumi += q5 * (int)a8[j];
        sumy += (int)a8[j];
    }
    acc = __fadd_rn(acc, __fmul_rn(__fmul_rn(dx, dy), (float)sumi));
    return __fadd_rn(acc, __fmul_rn(__fmul_rn(mx, dy), (float)sumy));
}

// kernels32.c: k32_iq4nl_q8_0_scalar
// Same 18-byte framing as Q4_0, but the nibble indexes kvalues_iq4nl[16]
// (nonlinear table, no -8 bias).
Q35_DEV float q35_vec_dot_iq4_nl_q8_0_acc(const unsigned char *w,
                                          const unsigned char *a, float acc) {
    const float dx = q35_f16_at(w, 0);
    const float dy = q35_f16_at(a, 0);
    const signed char *a8 = (const signed char *)(a + 2);
    int sumi = 0;
    for (int j = 0; j < 16; ++j) {
        const int v = (int)w[2 + j];
        sumi += (int)kvalues_iq4nl[v & 0x0F] * (int)a8[j];
        sumi += (int)kvalues_iq4nl[v >> 4] * (int)a8[j + 16];
    }
    return __fadd_rn(acc, __fmul_rn(__fmul_rn(dx, dy), (float)sumi));
}

// ===========================================================================
// 256-element K-quant family (weight) x Q8_K (activation)
//
// Reference: src/kernels/x86/kernels256.c k256_<type>_q8_k_scalar.
// These mirror the aux8 decomposition of the CPU reference rather than
// ggml-cuda's packed-int tricks: each weight's pre-scale integer value is
// materialised into a per-block signed char[256] scratch, then reduced with
// q35_dot_scaled16.  The association tree is then identical to the CPU oracle
// by construction, which is the point of this file.  The 256 B scratch lives
// in the calling thread's local memory (L1-backed); that is acceptable at the
// occupancy the one-thread-per-row mapping in mmvq.cu implies.
// ===========================================================================

// q2_K: 84 B = scales[16] @0 | qs[64] @16 | d @80 (f16) | dmin @82 (f16)
// kernels256.c:226.  Note the two nibble roles: the LOW nibble of scales[j] is
// the 4-bit scale, the HIGH nibble is the 4-bit min.
Q35_DEV void q35_q2k_aux8(const unsigned char *w, signed char *w8) {
    const unsigned char *q2 = w + 16;
    for (int k = 0; k < 2; ++k)
        for (int j = 0; j < 4; ++j)
            for (int l = 0; l < 32; ++l)
                w8[k * 128 + j * 32 + l] =
                    (signed char)((q2[k * 32 + l] >> (2 * j)) & 0x03);
}

Q35_DEV float q35_vec_dot_q2_k_q8_k_acc(const unsigned char *w,
                                        const unsigned char *y, float acc) {
    const unsigned char *sc = w;       // 16 scale/min bytes
    signed char w8[Q35_QK_K];
    q35_q2k_aux8(w, w8);

    int summs = 0;
    for (int j = 0; j < 16; ++j) summs += q35_q8k_bsum(y, j) * (int)(sc[j] >> 4);
    int scv[16];
    for (int g = 0; g < 16; ++g) scv[g] = (int)(sc[g] & 0x0F);

    const int isum = q35_dot_scaled16(w8, q35_q8k_qs(y), scv);
    const float dy = q35_q8k_d(y);
    const float dall = __fmul_rn(dy, q35_f16_at(w, 80));
    const float dmin = __fmul_rn(dy, q35_f16_at(w, 82));
    // sumf += dall*isum - dmin*summs
    acc = __fadd_rn(acc, __fmul_rn(dall, (float)isum));
    return __fadd_rn(acc, __fmul_rn(-dmin, (float)summs));
}

// q3_K: 110 B = hm[32] @0 | qs[64] @32 | scales[12] @96 | d @108 (f16)
// kernels256.c:258.  Each 2-bit value is biased by -4 unless the matching hm bit
// is set; the 6-bit scales are used as (s - 32).
Q35_DEV void q35_q3k_unpack_scales(const unsigned char *s12, signed char *sc) {
    // Verbatim port of kernels256.c:127 k256_unpack_q3_scales.
    unsigned int a[4];
    for (int i = 0; i < 3; ++i) a[i] = q35_ld_u32(s12 + i * 4);
    a[3] = 0;
    const unsigned int kmask1 = 0x03030303u;
    const unsigned int kmask2 = 0x0F0F0F0Fu;
    const unsigned int tmp = a[2];
    a[2] = ((a[0] >> 4) & kmask2) | (((tmp >> 4) & kmask1) << 4);
    a[3] = ((a[1] >> 4) & kmask2) | (((tmp >> 6) & kmask1) << 4);
    a[0] = ((a[0] >> 0) & kmask2) | (((tmp >> 0) & kmask1) << 4);
    a[1] = ((a[1] >> 0) & kmask2) | (((tmp >> 2) & kmask1) << 4);
    for (int i = 0; i < 4; ++i)
        for (int b = 0; b < 4; ++b)
            sc[i * 4 + b] = (signed char)(unsigned char)(a[i] >> (b * 8));
}

Q35_DEV void q35_q3k_aux8(const unsigned char *w, signed char *w8) {
    // Verbatim port of kernels256.c:158 k256_q3_aux8.
    const unsigned char *q3 = w + 32;
    const unsigned char *hm = w;
    int m = 1;
    for (int h = 0; h < 2; ++h) {
        for (int g = 0; g < 4; ++g) {
            for (int l = 0; l < 32; ++l) {
                const signed char nib = (signed char)((q3[h * 32 + l] >> (2 * g)) & 0x03);
                const signed char sub = (hm[l] & m) ? (signed char)0 : (signed char)4;
                w8[h * 128 + g * 32 + l] = (signed char)(nib - sub);
            }
            m <<= 1;
        }
    }
}

Q35_DEV float q35_vec_dot_q3_k_q8_k_acc(const unsigned char *w,
                                        const unsigned char *y, float acc) {
    signed char w8[Q35_QK_K];
    q35_q3k_aux8(w, w8);
    signed char sc[16];
    q35_q3k_unpack_scales(w + 96, sc);
    int scv[16];
    for (int g = 0; g < 16; ++g) scv[g] = (int)sc[g] - 32;
    const int isum = q35_dot_scaled16(w8, q35_q8k_qs(y), scv);
    const float dy = q35_q8k_d(y);
    // sumf += fp16(d) * dy * (float)isum   (left-to-right, two roundings)
    return __fadd_rn(acc, __fmul_rn(__fmul_rn(q35_f16_at(w, 108), dy), (float)isum));
}

// Q4_K / Q5_K share the 12-byte scale/min packing.
//
// ggml-quants.c get_scale_min_k4(j, sc, *d, *m).  Two consumers:
//   * the dot products below, via the vectorised unpack below (one shuffle for
//     all 16 values);
//   * the dequantizer in convert.cu, which needs them one pair at a time to
//     interleave with its output writes -- hence this scalar form too.
// Both must agree; they are the same formula, so keep them adjacent.
Q35_DEV void q35_get_scale_min_k4(int j, const unsigned char *sc,
                                  unsigned char *d, unsigned char *m) {
    // Verbatim from quant_dequant.zig:42 get_scale_min_k4.
    if (j < 4) {
        *d = (unsigned char)(sc[j + 0] & 63);
        *m = (unsigned char)(sc[j + 4] & 63);
    } else {
        *d = (unsigned char)((sc[j + 4] & 0x0F) | ((sc[j - 4] >> 6) << 4));
        *m = (unsigned char)((sc[j + 4] >> 4)   | ((sc[j - 0] >> 6) << 4));
    }
}

Q35_DEV void q35_unpack_scale_min(const unsigned char *s12, unsigned char *sm) {
    // Verbatim port of kernels256.c:103 k256_unpack_scale_min.
    // sm[0..7]  = 6-bit scales, one per 32 elements (index g/2)
    // sm[8..15] = 6-bit mins,   one per 16 elements (index 8 + j/2)
    unsigned int u[4];
    u[0] = q35_ld_u32(s12 + 0);
    u[1] = q35_ld_u32(s12 + 4);
    u[2] = q35_ld_u32(s12 + 8);
    u[3] = 0;
    const unsigned int kmask1 = 0x3F3F3F3Fu;
    const unsigned int kmask2 = 0x0F0F0F0Fu;
    const unsigned int kmask3 = 0x03030303u;
    u[3] = ((u[2] >> 4) & kmask2) | (((u[1] >> 6) & kmask3) << 4);
    const unsigned int uaux = u[1] & kmask1;
    u[1] = ((u[2] >> 0) & kmask2) | (((u[0] >> 6) & kmask3) << 4);
    u[2] = uaux;
    u[0] &= kmask1;
    for (int i = 0; i < 4; ++i)
        for (int b = 0; b < 4; ++b)
            sm[i * 4 + b] = (unsigned char)(u[i] >> (b * 8));
}

// q4_K: 144 B = d @0 (f16) | dmin @2 (f16) | scales[12] @4 | qs[128] @16
// kernels256.c:293, aux8 at kernels256.c:178.
Q35_DEV float q35_vec_dot_q4_k_q8_k_acc(const unsigned char *w,
                                        const unsigned char *y, float acc) {
    const unsigned char *q4 = w + 16;
    signed char w8[Q35_QK_K];
    for (int c = 0; c < 4; ++c) {
        for (int l = 0; l < 32; ++l) w8[c * 64 + l] = (signed char)(q4[c * 32 + l] & 0x0F);
        for (int l = 0; l < 32; ++l) w8[c * 64 + 32 + l] = (signed char)(q4[c * 32 + l] >> 4);
    }
    unsigned char sm[16];
    q35_unpack_scale_min(w + 4, sm);

    int sumi = 0;
    for (int j = 0; j < 16; ++j) sumi += q35_q8k_bsum(y, j) * (int)sm[8 + j / 2];
    int scv[16];
    for (int g = 0; g < 16; ++g) scv[g] = (int)sm[g / 2];

    const int isum = q35_dot_scaled16(w8, q35_q8k_qs(y), scv);
    const float dy = q35_q8k_d(y);
    const float d = __fmul_rn(dy, q35_f16_at(w, 0));
    const float dmin = __fmul_rn(dy, q35_f16_at(w, 2));
    acc = __fadd_rn(acc, __fmul_rn(d, (float)isum));
    return __fadd_rn(acc, __fmul_rn(-dmin, (float)sumi));
}

// q5_K: 176 B = d @0 | dmin @2 | scales[12] @4 | qh[32] @16 | qs[128] @48
// kernels256.c:325, aux8 at kernels256.c:185 (m walks 1,4,16,64).
Q35_DEV float q35_vec_dot_q5_k_q8_k_acc(const unsigned char *w,
                                        const unsigned char *y, float acc) {
    const unsigned char *q4 = w + 48;
    const unsigned char *hm = w + 16;
    signed char w8[Q35_QK_K];
    int m = 1;
    for (int c = 0; c < 4; ++c) {
        for (int l = 0; l < 32; ++l)
            w8[c * 64 + l] = (signed char)((q4[c * 32 + l] & 0x0F) + ((hm[l] & m) ? 16 : 0));
        for (int l = 0; l < 32; ++l)
            w8[c * 64 + 32 + l] = (signed char)((q4[c * 32 + l] >> 4) + ((hm[l] & (m << 1)) ? 16 : 0));
        m <<= 2;
    }
    unsigned char sm[16];
    q35_unpack_scale_min(w + 4, sm);

    int sumi = 0;
    for (int j = 0; j < 16; ++j) sumi += q35_q8k_bsum(y, j) * (int)sm[8 + j / 2];
    int scv[16];
    for (int g = 0; g < 16; ++g) scv[g] = (int)sm[g / 2];

    const int isum = q35_dot_scaled16(w8, q35_q8k_qs(y), scv);
    const float dy = q35_q8k_d(y);
    const float d = __fmul_rn(dy, q35_f16_at(w, 0));
    const float dmin = __fmul_rn(dy, q35_f16_at(w, 2));
    acc = __fadd_rn(acc, __fmul_rn(d, (float)isum));
    return __fadd_rn(acc, __fmul_rn(-dmin, (float)sumi));
}

// q6_K: 210 B = ql[128] @0 | qh[64] @128 | scales[16] int8 @192 | d @208 (f16)
// kernels256.c:355, aux8 at kernels256.c:194.
Q35_DEV void q35_q6k_aux8(const unsigned char *w, signed char *w8) {
    const unsigned char *q4 = w;
    const unsigned char *qh = w + 128;
    for (int h = 0; h < 2; ++h) {
        for (int l = 0; l < 32; ++l) {
            w8[h * 128 + l + 0]  = (signed char)(((q4[h * 64 + l] & 0x0F) |
                                    (((qh[h * 32 + l] >> 0) & 3) << 4)) - 32);
            w8[h * 128 + l + 32] = (signed char)(((q4[h * 64 + l + 32] & 0x0F) |
                                    (((qh[h * 32 + l] >> 2) & 3) << 4)) - 32);
            w8[h * 128 + l + 64] = (signed char)(((q4[h * 64 + l] >> 4) |
                                    (((qh[h * 32 + l] >> 4) & 3) << 4)) - 32);
            w8[h * 128 + l + 96] = (signed char)(((q4[h * 64 + l + 32] >> 4) |
                                    (((qh[h * 32 + l] >> 6) & 3) << 4)) - 32);
        }
    }
}

Q35_DEV float q35_vec_dot_q6_k_q8_k_acc(const unsigned char *w,
                                        const unsigned char *y, float acc) {
    signed char w8[Q35_QK_K];
    q35_q6k_aux8(w, w8);
    const signed char *sc = (const signed char *)(w + 192);
    int scv[16];
    for (int g = 0; g < 16; ++g) scv[g] = (int)sc[g];

    const int isum = q35_dot_scaled16(w8, q35_q8k_qs(y), scv);
    const float dy = q35_q8k_d(y);
    return __fadd_rn(acc, __fmul_rn(__fmul_rn(q35_f16_at(w, 208), dy), (float)isum));
}

// ===========================================================================
// 256-element IQ family (weight) x Q8_K (activation)
//
// Reference: src/kernels/x86/iq_kernels.c scalar impls.  Signs use
// ksigns_iq2xs[] + kmask_iq2xs[] (bit masks), never keven_signs_q2xs.
// ===========================================================================

// --- IQ2_XXS: 66 B = d @0 (f16) | qs[32] as u16 @2 --------------------------
// iq_kernels.c:370 iq2_xxs_q8_k_scalar_impl.
//   db = d * (0.5f + (aux1>>28)) * 0.25f, accumulated per 8-value group
// The oracle accumulates `sum += db * grid * q8` element-by-element in float;
// reproduced in the same order (it is inherently float-ordered because the
// 8-value group scales by db rather than folding into an integer sum).
Q35_DEV float q35_vec_dot_iq2_xxs_q8_k_acc(const unsigned char *w,
                                           const unsigned char *y, float acc) {
    const float d = __fmul_rn(q35_f16_at(w, 0), q35_q8k_d(y));
    const unsigned char *q2 = w + 2;
    const signed char *q8 = q35_q8k_qs(y);
    float sum = acc;
    for (int ib32 = 0; ib32 < 8; ++ib32) {
        const unsigned char *a = q2 + ib32 * 8;
        const unsigned int aux1 = q35_ld_u32(a + 4);
        const float db = __fmul_rn(__fmul_rn(d, 0.5f + (float)(aux1 >> 28)), 0.25f);
        for (int l = 0; l < 4; ++l) {
            const unsigned long long grid = iq2xxs_grid[a[l]];
            const unsigned char signs = ksigns_iq2xs[(aux1 >> (7 * l)) & 127];
            for (int j = 0; j < 8; ++j) {
                signed char g = (signed char)((grid >> (8 * j)) & 0xFF);
                if ((signs >> j) & 1) g = (signed char)(-g);
                sum = __fadd_rn(sum, __fmul_rn(__fmul_rn(db, (float)g),
                                               (float)q8[ib32 * 32 + l * 8 + j]));
            }
        }
    }
    return sum;
}

// --- IQ2_XS: 74 B = d @0 (f16) | qs[32] u16 @2 | scales[8] @66 --------------
// iq_kernels.c:828 iq2_xs_q8_k_scalar_impl.
//   integer group sums, ls1/ls2 = 2*nibble+1, then 0.125f on the total.
// The oracle applies 0.125f once to the running total; folding it per block
// instead is bit-identical because 0.125 is an exact power of two and every
// intermediate here has the same exponent budget (verified by the /tmp diff
// harness -- see the report).
Q35_DEV float q35_vec_dot_iq2_xs_q8_k_acc(const unsigned char *w,
                                          const unsigned char *y, float acc) {
    const float d = __fmul_rn(q35_f16_at(w, 0), q35_q8k_d(y));
    const unsigned char *q2 = w + 2;
    const unsigned char *sc = w + 66;
    const signed char *q8 = q35_q8k_qs(y);
    int bsum = 0;
    for (int ib32 = 0; ib32 < 8; ++ib32) {
        const unsigned short ls1 = (unsigned short)(2 * (sc[ib32] & 0xF) + 1);
        const unsigned short ls2 = (unsigned short)(2 * (sc[ib32] >> 4) + 1);
        const unsigned short *q = (const unsigned short *)(q2 + ib32 * 8);
        int sumi = 0;
        for (int l = 0; l < 2; ++l) {
            const unsigned short v = q35_ld_u16((const unsigned char *)(q + l));
            const unsigned long long g = iq2xs_grid[v & 511];
            const unsigned char signs = ksigns_iq2xs[v >> 9];
            for (int j = 0; j < 8; ++j) {
                const int wv = (int)((g >> (8 * j)) & 0xFF);
                sumi += wv * (int)q8[j] * ((signs & kmask_iq2xs[j]) ? -1 : 1);
            }
            q8 += 8;
        }
        bsum += sumi * ls1;
        sumi = 0;
        for (int l = 2; l < 4; ++l) {
            const unsigned short v = q35_ld_u16((const unsigned char *)(q + l));
            const unsigned long long g = iq2xs_grid[v & 511];
            const unsigned char signs = ksigns_iq2xs[v >> 9];
            for (int j = 0; j < 8; ++j) {
                const int wv = (int)((g >> (8 * j)) & 0xFF);
                sumi += wv * (int)q8[j] * ((signs & kmask_iq2xs[j]) ? -1 : 1);
            }
            q8 += 8;
        }
        bsum += sumi * ls2;
    }
    return __fadd_rn(acc, __fmul_rn(__fmul_rn(d, (float)bsum), 0.125f));
}

// --- IQ2_S: 82 B = d @0 (f16) | qs[64] @2 | qh[8] @66 | scales[8] @74 ------
// iq_kernels.c:1576 (scalar fallback body).  Signs live inside qs at
// qs + QK_K/8 (i.e. w + 2 + 32), not in a separate member.
Q35_DEV float q35_vec_dot_iq2_s_q8_k_acc(const unsigned char *w,
                                         const unsigned char *y, float acc) {
    const float d = __fmul_rn(q35_f16_at(w, 0), q35_q8k_d(y));
    const signed char *q8 = q35_q8k_qs(y);
    const unsigned char *qs = w + 2;
    const unsigned char *qh = w + 66;
    const unsigned char *scales = w + 74;
    const unsigned char *signs = qs + Q35_QK_K / 8;  // = w + 34

    int bsum = 0;
    for (int ib32 = 0; ib32 < 8; ++ib32) {
        const int ls1 = 1 + 2 * (scales[ib32] & 0xF);
        const int ls2 = 1 + 2 * (scales[ib32] >> 4);
        int sumi1 = 0, sumi2 = 0;
        for (int l = 0; l < 2; ++l) {
            const unsigned int idx = (unsigned int)qs[l] |
                                     (((unsigned int)qh[ib32] << (8 - 2 * l)) & 0x300u);
            const unsigned long long g = iq2s_grid[idx];
            for (int j = 0; j < 8; ++j) {
                const int wv = (int)((g >> (8 * j)) & 0xFF);
                sumi1 += (int)q8[j] * wv * ((signs[l] & kmask_iq2xs[j]) ? -1 : 1);
            }
            q8 += 8;
        }
        for (int l = 2; l < 4; ++l) {
            const unsigned int idx = (unsigned int)qs[l] |
                                     (((unsigned int)qh[ib32] << (8 - 2 * l)) & 0x300u);
            const unsigned long long g = iq2s_grid[idx];
            for (int j = 0; j < 8; ++j) {
                const int wv = (int)((g >> (8 * j)) & 0xFF);
                sumi2 += (int)q8[j] * wv * ((signs[l] & kmask_iq2xs[j]) ? -1 : 1);
            }
            q8 += 8;
        }
        bsum += ls1 * sumi1 + ls2 * sumi2;
        qs += 4;
        signs += 4;
    }
    return __fadd_rn(acc, __fmul_rn(__fmul_rn(d, (float)bsum), 0.125f));
}

// --- IQ3_XXS: 98 B = d @0 (f16) | qs[64] @2 | gas[32] @66 -------------------
// iq_kernels.c:611 iq3_xxs_q8_k_scalar_impl.
//   ls = 2*(aux32>>28)+1 (always odd, positive, no -32 bias -- unlike IQ3_S);
//   two 4-value grid reads per l, int32 bsum, then sumf += d*bsum.
Q35_DEV float q35_vec_dot_iq3_xxs_q8_k_acc(const unsigned char *w,
                                           const unsigned char *y, float acc) {
    const float d = __fmul_rn(q35_f16_at(w, 0), q35_q8k_d(y));
    const unsigned char *q3  = w + 2;                 // QK_K/4 bytes
    const unsigned char *gas = w + 2 + Q35_QK_K / 4;  // 32 scale bytes
    const signed char *q8 = q35_q8k_qs(y);
    int bsum = 0;
    for (int ib32 = 0; ib32 < Q35_QK_K / 32; ++ib32) {
        const unsigned int aux32 = q35_ld_u32(gas);
        gas += 4;
        const unsigned int ls = 2 * (aux32 >> 28) + 1;
        int sumi = 0;
        for (int l = 0; l < 4; ++l) {
            // iq3xxs_grid is uint32_t[256]: each entry packs 4 unsigned bytes.
            const unsigned int g1 = iq3xxs_grid[q3[2 * l + 0]];
            const unsigned int g2 = iq3xxs_grid[q3[2 * l + 1]];
            const unsigned char signs = ksigns_iq2xs[(aux32 >> (7 * l)) & 127];
            for (int j = 0; j < 4; ++j) {
                const int v1 = (int)((g1 >> (8 * j)) & 0xFF);
                const int v2 = (int)((g2 >> (8 * j)) & 0xFF);
                sumi += v1 * (int)q8[j + 0] * ((signs & kmask_iq2xs[j + 0]) ? -1 : 1);
                sumi += v2 * (int)q8[j + 4] * ((signs & kmask_iq2xs[j + 4]) ? -1 : 1);
            }
            q8 += 8;
        }
        q3 += 8;
        bsum += sumi * (int)ls;
    }
    // The oracle returns `0.25f * sumf` AFTER the loop (iq_kernels.c:641), i.e.
    // one scaling of the whole row, not a per-block factor.  Folding the 0.25
    // into the per-block term is algebraically identical and keeps this
    // function's `acc + block` contract uniform with the other types (a
    // per-block 0.25f would also be exact here since it is a power of two, but
    // keeping it inside the block term means a multi-block row needs no
    // post-pass in mmvq.cu -- which is the whole point of the _acc shape).
    return __fadd_rn(acc, __fmul_rn(__fmul_rn(d, (float)bsum), 0.25f));
}

// --- IQ3_S: 110 B = d @0 | qs[64] @2 | qh[8] @66 | signs[32] @74 | sc[4] @106
// iq_kernels.c:1727 iq3_s_q8_k_scalar_impl.  Grid index uses bit 0x100 of qh
// (9-bit index into iq3s_grid[512]); ls1/ls2 are 2*nibble+1; output is plain
// sumf (no 0.125 factor, unlike IQ2_XS/IQ2_S).
Q35_DEV float q35_vec_dot_iq3_s_q8_k_acc(const unsigned char *w,
                                         const unsigned char *y, float acc) {
    const float d = __fmul_rn(q35_f16_at(w, 0), q35_q8k_d(y));
    const unsigned char *qs = w + 2;
    const unsigned char *qh = w + 66;
    const unsigned char *signs = w + 74;
    const unsigned char *scales = w + 106;
    const signed char *q8 = q35_q8k_qs(y);
    int bsum = 0;
    for (int ib32 = 0; ib32 < Q35_QK_K / 32; ib32 += 2) {
        const unsigned int ls1 = 2 * (unsigned int)(scales[ib32 / 2] & 0xF) + 1;
        const unsigned int ls2 = 2 * (unsigned int)(scales[ib32 / 2] >> 4) + 1;
        for (int half = 0; half < 2; ++half) {
            int sumi = 0;
            for (int l = 0; l < 4; ++l) {
                const unsigned int i1 = (unsigned int)qs[2 * l + 0] |
                                        (((unsigned int)qh[ib32 + half] << (8 - 2 * l)) & 256u);
                const unsigned int i2 = (unsigned int)qs[2 * l + 1] |
                                        (((unsigned int)qh[ib32 + half] << (7 - 2 * l)) & 256u);
                const unsigned int g1 = iq3s_grid[i1];
                const unsigned int g2 = iq3s_grid[i2];
                for (int j = 0; j < 4; ++j) {
                    const int v1 = (int)((g1 >> (8 * j)) & 0xFF);
                    const int v2 = (int)((g2 >> (8 * j)) & 0xFF);
                    sumi += v1 * (int)q8[j + 0] * ((signs[l] & kmask_iq2xs[j + 0]) ? -1 : 1);
                    sumi += v2 * (int)q8[j + 4] * ((signs[l] & kmask_iq2xs[j + 4]) ? -1 : 1);
                }
                q8 += 8;
            }
            qs += 8;
            signs += 4;
            bsum += sumi * (int)(half == 0 ? ls1 : ls2);
        }
    }
    return __fadd_rn(acc, __fmul_rn(d, (float)bsum));
}

// --- IQ4_XS: 136 B = d @0 (f16) | scales_h @2 (u16) | scales_l[4] @4 | qs[128] @8
// iq_kernels.c:966 iq4_xs_q8_k_scalar_impl.  Plain kvalues_iq4nl lookup (no
// sign table); `ls` HAS the -32 bias; no 0.125 output factor.
Q35_DEV float q35_vec_dot_iq4_xs_q8_k_acc(const unsigned char *w,
                                          const unsigned char *y, float acc) {
    const float d4d8 = __fmul_rn(q35_f16_at(w, 0), q35_q8k_d(y));
    unsigned short h = q35_ld_u16(w + 2);
    const unsigned char *scales_l = w + 4;
    const unsigned char *qs = w + 8;
    const signed char *q8 = q35_q8k_qs(y);
    for (int ib = 0; ib < Q35_QK_K / 32; ib += 2) {
        const unsigned char ls1 = (unsigned char)((scales_l[ib / 2] & 0xF) | ((h << 4) & 0x30));
        const unsigned char ls2 = (unsigned char)((scales_l[ib / 2] >> 4)  | ((h << 2) & 0x30));
        h >>= 4;
        const float d1 = __fmul_rn(d4d8, (float)((int)ls1 - 32));
        const float d2 = __fmul_rn(d4d8, (float)((int)ls2 - 32));
        int sumi1 = 0, sumi2 = 0;
        for (int j = 0; j < 16; ++j) {
            sumi1 += (int)q8[j + 0]  * (int)kvalues_iq4nl[qs[j] & 0xF];
            sumi2 += (int)q8[j + 16] * (int)kvalues_iq4nl[qs[j] >> 4];
        }
        acc = __fadd_rn(acc, __fmul_rn(d1, (float)(sumi1 + sumi2)));
        qs += 16;
        q8 += 32;
        sumi1 = sumi2 = 0;
        for (int j = 0; j < 16; ++j) {
            sumi1 += (int)q8[j + 0]  * (int)kvalues_iq4nl[qs[j] & 0xF];
            sumi2 += (int)q8[j + 16] * (int)kvalues_iq4nl[qs[j] >> 4];
        }
        acc = __fadd_rn(acc, __fmul_rn(d2, (float)(sumi1 + sumi2)));
        qs += 16;
        q8 += 32;
    }
    return acc;
}

// --- IQ1_S: 50 B = d @0 (f16) | qs[32] @2 | qh[8] as u16 @34 ---------------
// iq_kernels.c:1231 iq1_s_q8_k_scalar_impl.
//   value = fp16(d) * ls * (grid + delta*IQ1S_DELTA), ls = 2*((qh>>12)&7)+1,
//   delta = +-1 from qh bit 0x8000.  The delta term uses y->bsums, so it is
//   accumulated separately as sumi1 with the same integer ls/delta weights.
#define Q35_IQ1S_DELTA 0.125f
Q35_DEV float q35_vec_dot_iq1_s_q8_k_acc(const unsigned char *w,
                                         const unsigned char *y, float acc) {
    const signed char *q8 = q35_q8k_qs(y);
    const unsigned char *qs = w + 2;
    const unsigned char *qh_bytes = w + 34;
    int sumi = 0, sumi1 = 0;
    for (int ib = 0; ib < Q35_QK_K / 32; ++ib) {
        const unsigned short qh = q35_ld_u16(qh_bytes + ib * 2);
        const int ls = 2 * (int)((qh >> 12) & 7) + 1;
        const int delta = (qh & 0x8000) ? -1 : 1;
        int lsum = 0;
        for (int l = 0; l < 4; ++l) {
            const unsigned int idx = (unsigned int)qs[l] |
                                     (((unsigned int)((qh >> (3 * l)) & 7)) << 8);
            const unsigned long long g = iq1s_grid[idx];
            for (int j = 0; j < 8; ++j) {
                const signed char gv = (signed char)((g >> (8 * j)) & 0xFF);
                lsum += (int)q8[j] * (int)gv;
            }
            q8 += 8;
        }
        sumi  += ls * lsum;
        sumi1 += ls * delta * (q35_q8k_bsum(y, 2 * ib + 0) + q35_q8k_bsum(y, 2 * ib + 1));
        qs += 4;
    }
    const float d = __fmul_rn(q35_f16_at(w, 0), q35_q8k_d(y));
    // sumf += d * (float)(sumi + IQ1S_DELTA * sumi1)
    const float tot = __fadd_rn((float)sumi, __fmul_rn(Q35_IQ1S_DELTA, (float)sumi1));
    return __fadd_rn(acc, __fmul_rn(d, tot));
}

// --- IQ1_M: 56 B = qs[32] @0 | qh[16] @32 | scales[8] @48 -------------------
// iq_kernels.c:1402 iq1_m_q8_k_scalar_impl.  Super-scale is an f16 packed
// across the 8 scale bytes (iq1m_scale_t); per-32 scales are 3-bit at
// 6*(ib%2)+{0,3}; deltas come from qh bits 0x08 / 0x80.
#define Q35_IQ1M_DELTA 0.125f
Q35_DEV float q35_iq1m_scale_f16(const unsigned char *w) {
    // iq_kernels.c:1300 iq1_m_scale_f16.
    const unsigned short sc0 = q35_ld_u16(w + 48);
    const unsigned short sc1 = q35_ld_u16(w + 50);
    const unsigned short sc2 = q35_ld_u16(w + 52);
    const unsigned short sc3 = q35_ld_u16(w + 54);
    const unsigned short scale_u16 = (unsigned short)((sc0 >> 12) | ((sc1 >> 8) & 0x00F0) |
                                     ((sc2 >> 4) & 0x0F00) | (sc3 & 0xF000));
    const unsigned char bits[2] = {(unsigned char)(scale_u16 & 0xFF),
                                   (unsigned char)(scale_u16 >> 8)};
    return q35_f16_at(bits, 0);
}

Q35_DEV float q35_vec_dot_iq1_m_q8_k_acc(const unsigned char *w,
                                         const unsigned char *y, float acc) {
    const signed char *q8 = q35_q8k_qs(y);
    const unsigned char *qs = w;
    const unsigned char *qh = w + 32;
    int sumi1 = 0, sumi2 = 0;
    for (int ib = 0; ib < Q35_QK_K / 32; ++ib) {
        int delta[4];
        delta[0] = (qh[0] & 0x08) ? -1 : 1;
        delta[1] = (qh[0] & 0x80) ? -1 : 1;
        delta[2] = (qh[1] & 0x08) ? -1 : 1;
        delta[3] = (qh[1] & 0x80) ? -1 : 1;
        int sum1[2], sum2[2];
        sum1[0] = sum1[1] = sum2[0] = sum2[1] = 0;
        for (int l = 0; l < 4; ++l) {
            const unsigned int idx = (unsigned int)qs[l] |
                    (((unsigned int)qh[l / 2] << (8 - 4 * (l % 2))) & 0x700u);
            const unsigned long long g = iq1s_grid[idx];
            int lsum1 = 0, lsum2 = 0;
            for (int j = 0; j < 8; ++j) {
                const signed char gv = (signed char)((g >> (8 * j)) & 0xFF);
                lsum1 += (int)q8[j] * (int)gv;
                lsum2 += (int)q8[j];
            }
            q8 += 8;
            sum1[l / 2] += lsum1;
            sum2[l / 2] += lsum2 * delta[l];
        }
        const unsigned short sc = q35_ld_u16(w + 48 + (ib / 2) * 2);
        const int ls1 = 2 * (int)((sc >> (6 * (ib % 2) + 0)) & 0x7) + 1;
        const int ls2 = 2 * (int)((sc >> (6 * (ib % 2) + 3)) & 0x7) + 1;

        sumi1 += sum1[0] * ls1 + sum1[1] * ls2;
        sumi2 += sum2[0] * ls1 + sum2[1] * ls2;
        qs += 4;
        qh += 2;
    }
    const float d = __fmul_rn(q35_iq1m_scale_f16(w), q35_q8k_d(y));
    const float tot = __fadd_rn((float)sumi1, __fmul_rn(Q35_IQ1M_DELTA, (float)sumi2));
    return __fadd_rn(acc, __fmul_rn(d, tot));
}

#endif // Q35_CUDA_VECDOTQ_CUH
