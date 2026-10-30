// convert.cu -- activation quantizers (f32 -> block_q8_0 / block_q8_K) and
// dequantizers (every supported weight type -> f32), plus their C launchers.
//
// Anchors:
//   ggml/src/ggml-cuda/convert.cu            dequantize_block_q8_0
//   ggml/src/ggml-quants.c                   quantize_row_q8_0_ref,
//                                            dequantize_row_*
// Numeric order is anchored to THIS REPO's implementations instead, because the
// point of the exercise is a bit-identical backend:
//   src/kernels/x86/kernels32.c:1033  quantize_q8_0
//   src/kernels/x86/kernels256.c:392  quantize_q8_K
//   src/quant_dequant.zig             dequantQ4_0 / dequantQ4_1 / dequantQ4K /
//                                     dequantQ5K / dequantQ6K / dequantQ3K /
//                                     dequantQ2K / dequantIQ*  (the Zig
//                                     reference the CPU side itself is tested
//                                     against)
//
// Two traps this file deliberately avoids in quantize_q8_0:
//  (1) src/kernel.cu's k_quantize_q8 divides by the f16-truncated scale via
//      roundf(x/scale); that is a different rounding and is not reused here.
//  (2) The caller passes the *number of blocks*, not elements.
//
// Dequantization has one more requirement: it must be BIT-EXACT with the Zig
// reference, not merely close, because the CUDA backend uses it for the
// embedding / lm_head rows and any 1-ulp drift compounds.  The rules that fall
// out of the Zig sources and are obeyed below:
//   * f16 -> f32 via q35_ld_scale (exact, common.cuh), never `__half2float`.
//   * Expression order copied literally, including the `d*x - min` vs
//     `(d*x) - min` grouping and the `d * 0.5f * (0.5f + s)` triples used by
//     the IQ dequantizers.
//   * IQ sign masks come from ksigns_iq2xs + kmask_iq2xs only (see
//     iq_tables.cuh on why keven_signs_q2xs is not usable here).
#include "common.cuh"
#include "vecdotq.cuh"
#include "api.h"

// ---------------------------------------------------------------------------
// device kernels: activation quantizers
// ---------------------------------------------------------------------------

// One thread per 32-element block.
__global__ void q35_k_quantize_q8_0(const float *__restrict__ x,
                                    unsigned char *__restrict__ out,
                                    int nblocks) {
    const int b = blockIdx.x * blockDim.x + threadIdx.x;
    if (b >= nblocks) return;
    const float *xb = x + (size_t)b * Q35_QK8_0;
    unsigned char *ob = out + (size_t)b * Q35_Q8_0_BLOCK_BYTES;

    float amax = 0.0f;
#pragma unroll
    for (int k = 0; k < Q35_QK8_0; ++k) {
        const float a = fabsf(xb[k]);
        if (a > amax) amax = a;  // same comparison chain as the CPU loop
    }
    const float scale = amax > 0.0f ? amax / 127.0f : 1.0f;
    const float inv = amax > 0.0f ? 1.0f / scale : 0.0f;
    q35_store_scale(ob, scale);

    signed char *qs = (signed char *)(ob + 2);
#pragma unroll
    for (int k = 0; k < Q35_QK8_0; ++k) {
        // round half away from zero, matching `(int)(v + (v>=0 ? 0.5f : -0.5f))`
        // in kernels32.c:1059.  rintf() would be round-half-to-even -- wrong.
        const float v = xb[k] * inv;
        int q = (int)(v + (v >= 0.0f ? 0.5f : -0.5f));
        if (q > 127) q = 127;
        if (q < -128) q = -128;
        qs[k] = (signed char)q;
    }
}

// ---------------------------------------------------------------------------
// device kernels: dequantizers, one thread per NATIVE block
//
// Each writes exactly `block_elems` floats.  The launchers below convert a
// byte/element range into a block count and rely on cols being a multiple of
// the native block size (validated on the host).
// ---------------------------------------------------------------------------

// --------------------------------------------------------------- 32-elem family

// block_q8_0 -> 32 floats.  quant_dequant.zig has no q8_0 dequant of its own;
// the reference is the CPU dequantRowInto q8_0 bypass, i.e. d * qs[k].
__global__ void q35_k_dequantize_q8_0(const unsigned char *__restrict__ src,
                                      float *__restrict__ dst, int nblocks) {
    const int b = blockIdx.x * blockDim.x + threadIdx.x;
    if (b >= nblocks) return;
    const unsigned char *sb = src + (size_t)b * Q35_BYTES_Q8_0;
    float *db = dst + (size_t)b * 32;
    const float d = q35_ld_scale(sb);
    const signed char *qs = (const signed char *)(sb + 2);
#pragma unroll
    for (int k = 0; k < 32; ++k) db[k] = __fmul_rn(d, (float)qs[k]);
}

// block_q4_0 -> 32 floats.  quant_dequant.zig:78 dequantQ4_0.
//   y[j]    = ((qs[j] & 0xF) - 8) * d
//   y[16+j] = ((qs[j] >>  4) - 8) * d
__global__ void q35_k_dequantize_q4_0(const unsigned char *__restrict__ src,
                                      float *__restrict__ dst, int nblocks) {
    const int b = blockIdx.x * blockDim.x + threadIdx.x;
    if (b >= nblocks) return;
    const unsigned char *sb = src + (size_t)b * Q35_BYTES_Q4_0;
    float *db = dst + (size_t)b * 32;
    const float d = q35_ld_scale(sb);
#pragma unroll
    for (int j = 0; j < 16; ++j) {
        const int v = (int)sb[2 + j];
        db[j]      = __fmul_rn((float)((v & 0x0F) - 8), d);
        db[16 + j] = __fmul_rn((float)((v >> 4) - 8), d);
    }
}

// block_q4_1 -> 32 floats.  quant_dequant.zig:92 dequantQ4_1.
//   y = q*d + m   (multiply then add, both rounded)
__global__ void q35_k_dequantize_q4_1(const unsigned char *__restrict__ src,
                                      float *__restrict__ dst, int nblocks) {
    const int b = blockIdx.x * blockDim.x + threadIdx.x;
    if (b >= nblocks) return;
    const unsigned char *sb = src + (size_t)b * Q35_BYTES_Q4_1;
    float *db = dst + (size_t)b * 32;
    const float d = q35_ld_scale(sb);
    const float m = q35_ld_scale(sb + 2);
#pragma unroll
    for (int j = 0; j < 16; ++j) {
        const int v = (int)sb[4 + j];
        db[j]      = __fadd_rn(__fmul_rn((float)(v & 0x0F), d), m);
        db[16 + j] = __fadd_rn(__fmul_rn((float)(v >> 4), d), m);
    }
}

// block_iq4_nl -> 32 floats.  quant_dequant.zig:347 dequantIQ4NL.
//   y = d * kvalues_iq4nl[nibble]      (no -8 bias; nonlinear table)
__global__ void q35_k_dequantize_iq4_nl(const unsigned char *__restrict__ src,
                                        float *__restrict__ dst, int nblocks) {
    const int b = blockIdx.x * blockDim.x + threadIdx.x;
    if (b >= nblocks) return;
    const unsigned char *sb = src + (size_t)b * Q35_BYTES_IQ4_NL;
    float *db = dst + (size_t)b * 32;
    const float d = q35_ld_scale(sb);
#pragma unroll
    for (int j = 0; j < 16; ++j) {
        const int v = (int)sb[2 + j];
        db[j]      = __fmul_rn(d, (float)(int)kvalues_iq4nl[v & 0x0F]);
        db[16 + j] = __fmul_rn(d, (float)(int)kvalues_iq4nl[v >> 4]);
    }
}

// f16 / bf16 -> 32 floats.  2 B per element, so 64 B per 32-element block.
// bf16 -> f32 is exact (truncated high half of an f32: a pure 16-bit shift,
// kernels32.c:1105 k32_bf16_to_f32).
__global__ void q35_k_dequantize_f16(const unsigned char *__restrict__ src,
                                     float *__restrict__ dst, int nblocks) {
    const int b = blockIdx.x * blockDim.x + threadIdx.x;
    if (b >= nblocks) return;
    const unsigned char *sb = src + (size_t)b * 64;
    float *db = dst + (size_t)b * 32;
#pragma unroll
    for (int k = 0; k < 32; ++k) db[k] = q35_f16_at(sb, 2 * k);
}

__global__ void q35_k_dequantize_bf16(const unsigned char *__restrict__ src,
                                      float *__restrict__ dst, int nblocks) {
    const int b = blockIdx.x * blockDim.x + threadIdx.x;
    if (b >= nblocks) return;
    const unsigned char *sb = src + (size_t)b * 64;
    float *db = dst + (size_t)b * 32;
#pragma unroll
    for (int k = 0; k < 32; ++k) {
        const unsigned int bits = (unsigned int)q35_ld_u16(sb + 2 * k) << 16;
        db[k] = __uint_as_float(bits);
    }
}

// f32 passthrough: keeps the type switch uniform for callers.
__global__ void q35_k_copy_f32(const float *__restrict__ src, float *__restrict__ dst,
                               int nblocks) {
    const size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    const size_t n = (size_t)nblocks * 32;
    if (i < n) dst[i] = src[i];
}

// -------------------------------------------------------------- 256-elem family

// Q2_K -> 256 floats.  quant_dequant.zig:892 dequantQ2K.
//   84 B = scales[16] @0 | qs[64] @16 | d @80 (f16) | dmin @82 (f16)
//   Two 128-element chunks; within a chunk, 4 bit-planes (j -> shift 2j) of
//   16 bytes each; element = dl * 2bit - ml.  NOTE the per-16-element
//   scale/min alternation (sc0 -> bytes 0..15, sc1 -> bytes 16..31) and that
//   the low nibble of scales[] is the scale, the high nibble the min.
__global__ void q35_k_dequantize_q2_k(const unsigned char *__restrict__ src,
                                      float *__restrict__ dst, int nblocks) {
    const int b = blockIdx.x * blockDim.x + threadIdx.x;
    if (b >= nblocks) return;
    const unsigned char *sb = src + (size_t)b * Q35_BYTES_Q2_K;
    float *db = dst + (size_t)b * Q35_QK_K;
    const float d = q35_f16_at(sb, 80);
    const float min = q35_f16_at(sb, 82);
    const unsigned char *scales = sb;
    size_t y = 0;
    int is = 0;
    for (int chunk = 0; chunk < 2; ++chunk) {
        const unsigned char *q = sb + 16 + 32 * chunk;
        for (int j = 0; j < 4; ++j) {
            const int sh = j * 2;
            const unsigned char sc0 = scales[is++];
            const float dl0 = __fmul_rn(d, (float)(int)(sc0 & 0x0F));
            const float ml0 = __fmul_rn(min, (float)(int)(sc0 >> 4));
            for (int l = 0; l < 16; ++l)
                db[y + l] = __fadd_rn(__fmul_rn(dl0, (float)(int)((q[l] >> sh) & 3)), -ml0);
            const unsigned char sc1 = scales[is++];
            const float dl1 = __fmul_rn(d, (float)(int)(sc1 & 0x0F));
            const float ml1 = __fmul_rn(min, (float)(int)(sc1 >> 4));
            for (int l = 0; l < 16; ++l)
                db[y + 16 + l] = __fadd_rn(__fmul_rn(dl1, (float)(int)((q[l + 16] >> sh) & 3)), -ml1);
            y += 32;
        }
    }
}

// Q3_K -> 256 floats.  quant_dequant.zig:252 dequantQ3K.
//   110 B = hmask[32] @0 | qs[64] @32 | scales[12] @96 | d @108 (f16)
//   `m` walks 1,2,...,128 over the whole block and is NOT reset per 128-chunk
//   (the Zig/C reference comment calls this out); the 6-bit scale is used
//   minus 32, and the 2-bit value becomes `2bit - 4` when the hmask bit is
//   clear, `2bit` when it is set.
__global__ void q35_k_dequantize_q3_k(const unsigned char *__restrict__ src,
                                      float *__restrict__ dst, int nblocks) {
    const int b = blockIdx.x * blockDim.x + threadIdx.x;
    if (b >= nblocks) return;
    const unsigned char *sb = src + (size_t)b * Q35_BYTES_Q3_K;
    float *db = dst + (size_t)b * Q35_QK_K;
    const float d_all = q35_f16_at(sb, 108);
    const unsigned char *q = sb + 32;
    const unsigned char *hm = sb;
    signed char scales[16];
    q35_q3k_unpack_scales(sb + 96, scales);

    int is = 0;
    size_t y = 0;
    unsigned int m = 1;  // NOT reset per 128-chunk (matches the reference)
    size_t q_off = 0;
    for (int n = 0; n < Q35_QK_K; n += 128) {
        for (int j = 0; j < 4; ++j) {
            const int shift = 2 * j;
            const int sval = (int)scales[is++] - 32;
            const float dl = __fmul_rn(d_all, (float)sval);
            for (int l = 0; l < 16; ++l) {
                const int raw = (int)((q[q_off + l] >> shift) & 3);
                const int vi = (hm[l] & (unsigned char)m) ? raw : raw - 4;
                db[y + l] = __fmul_rn(dl, (float)vi);
            }
            m <<= 1;
            for (int l = 0; l < 16; ++l) {
                const int raw = (int)((q[q_off + 16 + l] >> shift) & 3);
                const int vi = (hm[16 + l] & (unsigned char)m) ? raw : raw - 4;
                db[y + 16 + l] = __fmul_rn(dl, (float)vi);
            }
            m <<= 1;
            y += 32;
        }
        q_off += 32;
    }
}

// Q4_K -> 256 floats.  quant_dequant.zig:54 dequantQ4K +
// get_scale_min_k4 (quant_dequant.zig:42).
//   144 B = d @0 (f16) | dmin @2 (f16) | scales[12] @4 | qs[128] @16
//   y = d1*q4 - m1 (low nibble), d2*q4 - m2 (high nibble), 4 chunks of 64.
__global__ void q35_k_dequantize_q4_k(const unsigned char *__restrict__ src,
                                      float *__restrict__ dst, int nblocks) {
    const int b = blockIdx.x * blockDim.x + threadIdx.x;
    if (b >= nblocks) return;
    const unsigned char *sb = src + (size_t)b * Q35_BYTES_Q4_K;
    float *db = dst + (size_t)b * Q35_QK_K;
    const float d = q35_f16_at(sb, 0);
    const float min = q35_f16_at(sb, 2);
    const unsigned char *scales = sb + 4;
    const unsigned char *q = sb + 16;
    size_t y = 0;
    int is = 0;
    for (int c = 0; c < 4; ++c) {
        unsigned char sc0, m0, sc1, m1;
        q35_get_scale_min_k4(is + 0, scales, &sc0, &m0);
        q35_get_scale_min_k4(is + 1, scales, &sc1, &m1);
        const float d1 = __fmul_rn(d, (float)(int)sc0);
        const float mm1 = __fmul_rn(min, (float)(int)m0);
        const float d2 = __fmul_rn(d, (float)(int)sc1);
        const float mm2 = __fmul_rn(min, (float)(int)m1);
        for (int l = 0; l < 32; ++l)
            db[y + l] = __fadd_rn(__fmul_rn(d1, (float)(int)(q[l] & 0x0F)), -mm1);
        for (int l = 0; l < 32; ++l)
            db[y + 32 + l] = __fadd_rn(__fmul_rn(d2, (float)(int)(q[l] >> 4)), -mm2);
        y += 64;
        q += 32;
        is += 2;
    }
}

// Q5_K -> 256 floats.  quant_dequant.zig:188 dequantQ5K.
//   176 B = d @0 | dmin @2 | scales[12] @4 | qh[32] @16 | ql[128] @48
//   bm1 starts at 1 and bm2 at 2, both <<= 2 per chunk (bit pairs per 64-elem
//   group), so the 5th bit came from qh[l] & {1,2,4,8} then {16,32,64,128}.
__global__ void q35_k_dequantize_q5_k(const unsigned char *__restrict__ src,
                                      float *__restrict__ dst, int nblocks) {
    const int b = blockIdx.x * blockDim.x + threadIdx.x;
    if (b >= nblocks) return;
    const unsigned char *sb = src + (size_t)b * Q35_BYTES_Q5_K;
    float *db = dst + (size_t)b * Q35_QK_K;
    const float d = q35_f16_at(sb, 0);
    const float min = q35_f16_at(sb, 2);
    const unsigned char *scales = sb + 4;
    const unsigned char *qh = sb + 16;
    const unsigned char *ql = sb + 48;
    size_t y = 0;
    int is = 0;
    unsigned short bm1 = 1, bm2 = 2;
    for (int c = 0; c < 4; ++c) {
        unsigned char sc0, m0, sc1, m1;
        q35_get_scale_min_k4(is + 0, scales, &sc0, &m0);
        q35_get_scale_min_k4(is + 1, scales, &sc1, &m1);
        const float d1 = __fmul_rn(d, (float)(int)sc0);
        const float mm1 = __fmul_rn(min, (float)(int)m0);
        const float d2 = __fmul_rn(d, (float)(int)sc1);
        const float mm2 = __fmul_rn(min, (float)(int)m1);
        for (int l = 0; l < 32; ++l) {
            const int hi = (qh[l] & bm1) ? 16 : 0;
            db[y + l] = __fadd_rn(__fmul_rn(d1, (float)(int)((ql[l] & 0x0F) | hi)), -mm1);
        }
        for (int l = 0; l < 32; ++l) {
            const int hi = (qh[l] & bm2) ? 16 : 0;
            db[y + 32 + l] = __fadd_rn(__fmul_rn(d2, (float)(int)((ql[l] >> 4) | hi)), -mm2);
        }
        y += 64;
        ql += 32;
        is += 2;
        bm1 = (unsigned short)(bm1 << 2);
        bm2 = (unsigned short)(bm2 << 2);
    }
}

// Q6_K -> 256 floats.  quant_dequant.zig:222 dequantQ6K.
//   210 B = ql[128] @0 | qh[64] @128 | scales[16] i8 @192 | d @208 (f16)
//   Six-bit values are (4 low bits from ql) | (2 high bits from qh) - 32, and
//   the per-16-element scale index is scb + is + {0,2,4,6}.
__global__ void q35_k_dequantize_q6_k(const unsigned char *__restrict__ src,
                                      float *__restrict__ dst, int nblocks) {
    const int b = blockIdx.x * blockDim.x + threadIdx.x;
    if (b >= nblocks) return;
    const unsigned char *sb = src + (size_t)b * Q35_BYTES_Q6_K;
    float *db = dst + (size_t)b * Q35_QK_K;
    const float d = q35_f16_at(sb, 208);
    const unsigned char *ql = sb;
    const unsigned char *qh = sb + 128;
    signed char sc[16];
    for (int i = 0; i < 16; ++i) sc[i] = (signed char)sb[192 + i];

    for (int n = 0; n < Q35_QK_K; n += 128) {
        const int qlb = n / 2;   // 0 or 64
        const int qhb = n / 4;   // 0 or 32
        const int scb = n / 16;  // 0 or 8
        for (int l = 0; l < 32; ++l) {
            const int is = l / 16;  // 0 or 1
            const int q1 = (int)((ql[qlb + l] & 0x0F) | (((qh[qhb + l] >> 0) & 3) << 4)) - 32;
            const int q2 = (int)((ql[qlb + l + 32] & 0x0F) | (((qh[qhb + l] >> 2) & 3) << 4)) - 32;
            const int q3 = (int)((ql[qlb + l] >> 4) | (((qh[qhb + l] >> 4) & 3) << 4)) - 32;
            const int q4 = (int)((ql[qlb + l + 32] >> 4) | (((qh[qhb + l] >> 6) & 3) << 4)) - 32;
            // d * sc * q, left to right (two roundings)
            db[n + l]      = __fmul_rn(__fmul_rn(d, (float)(int)sc[scb + is + 0]), (float)q1);
            db[n + l + 32] = __fmul_rn(__fmul_rn(d, (float)(int)sc[scb + is + 2]), (float)q2);
            db[n + l + 64] = __fmul_rn(__fmul_rn(d, (float)(int)sc[scb + is + 4]), (float)q3);
            db[n + l + 96] = __fmul_rn(__fmul_rn(d, (float)(int)sc[scb + is + 6]), (float)q4);
        }
    }
}

// IQ4_XS -> 256 floats.  quant_dequant.zig:324 dequantIQ4XS.
//   136 B = d @0 (f16) | scales_h @2 (u16) | scales_l[4] @4 | qs[128] @8
//   ls = 4 low bits of scales_l[ib/2] (shifted by 4*(ib%2)) | 2 bits of
//   scales_h (shifted by 2*ib) << 4, then (ls - 32) scales d.
__global__ void q35_k_dequantize_iq4_xs(const unsigned char *__restrict__ src,
                                        float *__restrict__ dst, int nblocks) {
    const int b = blockIdx.x * blockDim.x + threadIdx.x;
    if (b >= nblocks) return;
    const unsigned char *sb = src + (size_t)b * Q35_BYTES_IQ4_XS;
    float *db = dst + (size_t)b * Q35_QK_K;
    const float d = q35_f16_at(sb, 0);
    const unsigned int sh = (unsigned int)q35_ld_u16(sb + 2);
    const unsigned char *sl = sb + 4;
    const unsigned char *qs = sb + 8;
    size_t y = 0;
    for (int ib = 0; ib < 8; ++ib) {
        const unsigned int sl4 = (unsigned int)sl[ib / 2] >> (4 * (ib % 2));
        const unsigned int sh4 = sh >> (2 * ib);
        const unsigned int ls = (sl4 & 0x0Fu) | ((sh4 & 3u) << 4);
        const float dl = __fmul_rn(d, (float)((int)ls - 32));
        for (int j = 0; j < 16; ++j) {
            const int q = (int)qs[16 * ib + j];
            db[y + j]      = __fmul_rn(dl, (float)(int)kvalues_iq4nl[q & 0x0F]);
            db[y + 16 + j] = __fmul_rn(dl, (float)(int)kvalues_iq4nl[q >> 4]);
        }
        y += 32;
    }
}

// IQ3_XXS -> 256 floats.  quant_dequant.zig:423 dequantIQ3XXS.
//   98 B = d @0 (f16) | qs[64] @2 | scales_and_signs[32] @66
//   db    = d * (0.5 + (aux32 >> 28)) * 0.5
//   signs = ksigns_iq2xs[(aux32 >> 7*l) & 127]; grid = iq3xxs_grid[qs[...]]
// Byte j of the packed u32 is value j (little-endian, hence the >> 8*j below).
__global__ void q35_k_dequantize_iq3_xxs(const unsigned char *__restrict__ src,
                                         float *__restrict__ dst, int nblocks) {
    const int b = blockIdx.x * blockDim.x + threadIdx.x;
    if (b >= nblocks) return;
    const unsigned char *sb = src + (size_t)b * Q35_BYTES_IQ3_XXS;
    float *db = dst + (size_t)b * Q35_QK_K;
    const float d = q35_f16_at(sb, 0);
    const unsigned char *qs = sb + 2;
    const unsigned char *ss = sb + 66;
    size_t y = 0;
    for (int ib32 = 0; ib32 < 8; ++ib32) {
        const unsigned int aux32 = q35_ld_u32(ss + 4 * ib32);
        const float dbk = __fmul_rn(__fmul_rn(d, 0.5f + (float)(aux32 >> 28)), 0.5f);
        for (int l = 0; l < 4; ++l) {
            const unsigned char signs = ksigns_iq2xs[(aux32 >> (7 * l)) & 127];
            const unsigned int g1 = iq3xxs_grid[qs[8 * ib32 + 2 * l + 0]];
            const unsigned int g2 = iq3xxs_grid[qs[8 * ib32 + 2 * l + 1]];
            for (int j = 0; j < 4; ++j) {
                const int v1 = (int)((g1 >> (8 * j)) & 0xFF);
                const int v2 = (int)((g2 >> (8 * j)) & 0xFF);
                const float s1 = (signs & kmask_iq2xs[j]) ? -1.0f : 1.0f;
                const float s2 = (signs & kmask_iq2xs[4 + j]) ? -1.0f : 1.0f;
                db[y + j]     = __fmul_rn(__fmul_rn(dbk, (float)v1), s1);
                db[y + 4 + j] = __fmul_rn(__fmul_rn(dbk, (float)v2), s2);
            }
            y += 8;
        }
    }
}

// IQ3_S -> 256 floats.  quant_dequant.zig:362 dequantIQ3S.
//   110 B = d @0 | qs[64] @2 | qh[8] @66 | signs[32] @74 | scales[4] @106
//   qh bit 0x100 selects the grid high bit; db1 from the low nibble of
//   scales, db2 from the high nibble; each p covers 64 elements.
__global__ void q35_k_dequantize_iq3_s(const unsigned char *__restrict__ src,
                                       float *__restrict__ dst, int nblocks) {
    const int b = blockIdx.x * blockDim.x + threadIdx.x;
    if (b >= nblocks) return;
    const unsigned char *sb = src + (size_t)b * Q35_BYTES_IQ3_S;
    float *db = dst + (size_t)b * Q35_QK_K;
    const float d = q35_f16_at(sb, 0);
    const unsigned char *qs = sb + 2;
    const unsigned char *qh = sb + 66;
    const unsigned char *signs = sb + 74;
    const unsigned char *scales = sb + 106;
    size_t y = 0;
    for (int p = 0; p < 4; ++p) {
        const unsigned char sbyte = scales[p];
        const float db1 = __fmul_rn(d, (float)(1 + 2 * (int)(sbyte & 0x0F)));
        const float db2 = __fmul_rn(d, (float)(1 + 2 * (int)(sbyte >> 4)));
        for (int half = 0; half < 2; ++half) {
            const unsigned int qhv = (unsigned int)qh[2 * p + half];
            const float dh = half == 0 ? db1 : db2;
            for (int l = 0; l < 4; ++l) {
                const unsigned int bit1 = (qhv << (8 - 2 * l)) & 0x100u;
                const unsigned int bit2 = (qhv << (7 - 2 * l)) & 0x100u;
                const unsigned int g1 = iq3s_grid[qs[16 * p + 8 * half + 2 * l + 0] | bit1];
                const unsigned int g2 = iq3s_grid[qs[16 * p + 8 * half + 2 * l + 1] | bit2];
                const unsigned char sg = signs[8 * p + 4 * half + l];
                for (int j = 0; j < 4; ++j) {
                    const int v1 = (int)((g1 >> (8 * j)) & 0xFF);
                    const int v2 = (int)((g2 >> (8 * j)) & 0xFF);
                    const float s1 = (sg & kmask_iq2xs[j]) ? -1.0f : 1.0f;
                    const float s2 = (sg & kmask_iq2xs[4 + j]) ? -1.0f : 1.0f;
                    db[y + l * 8 + j]     = __fmul_rn(__fmul_rn(dh, (float)v1), s1);
                    db[y + l * 8 + 4 + j] = __fmul_rn(__fmul_rn(dh, (float)v2), s2);
                }
            }
        }
        y += 64;
    }
}

// IQ2_XXS -> 256 floats.  quant_dequant.zig:751 dequantIQ2XXS.
//   66 B = d @0 (f16) | qs[32] (as u16) @2
//   db = d * (0.5 + (aux1>>28)) * 0.25 ; 8 bytes per grid entry.
__global__ void q35_k_dequantize_iq2_xxs(const unsigned char *__restrict__ src,
                                         float *__restrict__ dst, int nblocks) {
    const int b = blockIdx.x * blockDim.x + threadIdx.x;
    if (b >= nblocks) return;
    const unsigned char *sb = src + (size_t)b * Q35_BYTES_IQ2_XXS;
    float *db = dst + (size_t)b * Q35_QK_K;
    const float d = q35_f16_at(sb, 0);
    const unsigned char *qs = sb + 2;
    size_t y = 0;
    for (int ib32 = 0; ib32 < 8; ++ib32) {
        const unsigned char *a = qs + 8 * ib32;
        const unsigned int aux1 = q35_ld_u32(a + 4);
        const float dbk = __fmul_rn(__fmul_rn(d, 0.5f + (float)(aux1 >> 28)), 0.25f);
        for (int l = 0; l < 4; ++l) {
            const unsigned long long g = iq2xxs_grid[a[l]];
            const unsigned char signs = ksigns_iq2xs[(aux1 >> (7 * l)) & 127];
            for (int j = 0; j < 8; ++j) {
                const int v = (int)((g >> (8 * j)) & 0xFF);
                const float s = (signs & kmask_iq2xs[j]) ? -1.0f : 1.0f;
                db[y + j] = __fmul_rn(__fmul_rn(dbk, (float)v), s);
            }
            y += 8;
        }
    }
}

// IQ2_XS -> 256 floats.  quant_dequant.zig:793 dequantIQ2XS.
//   74 B = d @0 (f16) | qs[32] as u16 @2 | scales[8] @66
//   db0 from the low nibble of scales[ib32], db1 from the high nibble.
//   NOTE this uses f16 d * (0.5+nibble) * 0.25 -- the *dequant* form -- not the
//   integer `2*nibble+1` form the dot product uses (both are equal in exact
//   arithmetic; the dequant reference is the float one, so we match that here).
__global__ void q35_k_dequantize_iq2_xs(const unsigned char *__restrict__ src,
                                        float *__restrict__ dst, int nblocks) {
    const int b = blockIdx.x * blockDim.x + threadIdx.x;
    if (b >= nblocks) return;
    const unsigned char *sb = src + (size_t)b * Q35_BYTES_IQ2_XS;
    float *db = dst + (size_t)b * Q35_QK_K;
    const float d = q35_f16_at(sb, 0);
    const unsigned char *qs16 = sb + 2;
    const unsigned char *scales = sb + 66;
    size_t y = 0;
    for (int ib32 = 0; ib32 < 8; ++ib32) {
        const unsigned char sc = scales[ib32];
        const float db0 = __fmul_rn(__fmul_rn(d, 0.5f + (float)(int)(sc & 0x0F)), 0.25f);
        const float db1 = __fmul_rn(__fmul_rn(d, 0.5f + (float)(int)(sc >> 4)), 0.25f);
        for (int l = 0; l < 4; ++l) {
            const unsigned int q16 = (unsigned int)q35_ld_u16(qs16 + (4 * ib32 + l) * 2);
            const unsigned long long g = iq2xs_grid[q16 & 511u];
            const unsigned char signs = ksigns_iq2xs[q16 >> 9];
            const float dl = (l / 2 == 0) ? db0 : db1;
            for (int j = 0; j < 8; ++j) {
                const int v = (int)((g >> (8 * j)) & 0xFF);
                const float s = (signs & kmask_iq2xs[j]) ? -1.0f : 1.0f;
                db[y + j] = __fmul_rn(__fmul_rn(dl, (float)v), s);
            }
            y += 8;
        }
    }
}

// IQ2_S -> 256 floats.  quant_dequant.zig:842 dequantIQ2S.
//   82 B = d @0 | qs[64] @2 | qh[8] @66 | scales[8] @74
//   Signs live inside qs at +32 (row[34..66]); the grid index takes 2 bits of
//   qh[ib32] at 8-2*l (mask 0x300, so a 10-bit index into iq2s_grid[1024]).
__global__ void q35_k_dequantize_iq2_s(const unsigned char *__restrict__ src,
                                       float *__restrict__ dst, int nblocks) {
    const int b = blockIdx.x * blockDim.x + threadIdx.x;
    if (b >= nblocks) return;
    const unsigned char *sb = src + (size_t)b * Q35_BYTES_IQ2_S;
    float *db = dst + (size_t)b * Q35_QK_K;
    const float d = q35_f16_at(sb, 0);
    size_t y = 0;
    for (int ib32 = 0; ib32 < 8; ++ib32) {
        const unsigned char sc = sb[74 + ib32];
        const float db0 = __fmul_rn(__fmul_rn(d, 0.5f + (float)(int)(sc & 0x0F)), 0.25f);
        const float db1 = __fmul_rn(__fmul_rn(d, 0.5f + (float)(int)(sc >> 4)), 0.25f);
        const unsigned int qhv = (unsigned int)sb[66 + ib32];
        for (int l = 0; l < 4; ++l) {
            const float dl = (l / 2 == 0) ? db0 : db1;
            const unsigned int qsv = (unsigned int)sb[2 + 4 * ib32 + l];
            const unsigned int idx = qsv | ((qhv << (8 - 2 * l)) & 0x300u);
            const unsigned long long g = iq2s_grid[idx];
            const unsigned char sign_byte = sb[34 + 4 * ib32 + l];
            for (int j = 0; j < 8; ++j) {
                const int v = (int)((g >> (8 * j)) & 0xFF);
                const float s = (sign_byte & kmask_iq2xs[j]) ? -1.0f : 1.0f;
                db[y + j] = __fmul_rn(__fmul_rn(dl, (float)v), s);
            }
            y += 8;
        }
    }
}

// IQ1_S -> 256 floats.  quant_dequant.zig:954 dequantIQ1S.
//   50 B = d @0 (f16) | qs[32] @2 | qh[8] as u16 @34
//   y = dl * (grid + delta), dl = d*(2*((qh>>12)&7)+1), delta = +-0.125.
//   The grid entry is SIGNED (int8 +-1-packed), hence the signed char cast.
__global__ void q35_k_dequantize_iq1_s(const unsigned char *__restrict__ src,
                                       float *__restrict__ dst, int nblocks) {
    const int b = blockIdx.x * blockDim.x + threadIdx.x;
    if (b >= nblocks) return;
    const unsigned char *sb = src + (size_t)b * Q35_BYTES_IQ1_S;
    float *db = dst + (size_t)b * Q35_QK_K;
    const float d = q35_f16_at(sb, 0);
    const unsigned char *qs = sb + 2;
    size_t y = 0;
    for (int ib = 0; ib < 8; ++ib) {
        const unsigned int qh16 = (unsigned int)q35_ld_u16(sb + 34 + 2 * ib);
        const float dl = __fmul_rn(d, (float)(2 * (int)((qh16 >> 12) & 7) + 1));
        const float delta = (qh16 & 0x8000u) ? -0.125f : 0.125f;
        for (int l = 0; l < 4; ++l) {
            const unsigned int idx = (unsigned int)qs[4 * ib + l] |
                                     (((qh16 >> (3 * l)) & 7u) << 8);
            const unsigned long long g = iq1s_grid[idx];
            for (int j = 0; j < 8; ++j) {
                const int gv = (int)(signed char)((g >> (8 * j)) & 0xFF);
                db[y + j] = __fmul_rn(dl, __fadd_rn((float)gv, delta));
            }
            y += 8;
        }
    }
}

// IQ1_M -> 256 floats.  quant_dequant.zig:998 dequantIQ1M.
//   56 B = qs[32] @0 | qh[16] @32 | scales[8] @48
//   d is an f16 assembled from the top bits of the four 16-bit scale words
//   (iq1m_scale_t); per-32 scales are 3-bit at 6*(ib%2)+{0,3}; deltas come from
//   qh bits 0x08/0x80.
__global__ void q35_k_dequantize_iq1_m(const unsigned char *__restrict__ src,
                                       float *__restrict__ dst, int nblocks) {
    const int b = blockIdx.x * blockDim.x + threadIdx.x;
    if (b >= nblocks) return;
    const unsigned char *sb = src + (size_t)b * Q35_BYTES_IQ1_M;
    float *db = dst + (size_t)b * Q35_QK_K;
    const unsigned int s0 = (unsigned int)q35_ld_u16(sb + 48);
    const unsigned int s1 = (unsigned int)q35_ld_u16(sb + 50);
    const unsigned int s2 = (unsigned int)q35_ld_u16(sb + 52);
    const unsigned int s3 = (unsigned int)q35_ld_u16(sb + 54);
    const unsigned int scale_u16 = (s0 >> 12) | ((s1 >> 8) & 0x00F0u) |
                                   ((s2 >> 4) & 0x0F00u) | (s3 & 0xF000u);
    const unsigned char bits[2] = {(unsigned char)(scale_u16 & 0xFFu),
                                   (unsigned char)(scale_u16 >> 8)};
    const float d = q35_f16_at(bits, 0);
    const unsigned int sc_u16[4] = {s0, s1, s2, s3};
    const unsigned char *qs = sb;
    const unsigned char *qh = sb + 32;
    size_t y = 0;
    for (int ib = 0; ib < 8; ++ib) {
        const unsigned int sh = sc_u16[ib / 2];
        const float dl1 = __fmul_rn(d, (float)(2 * (int)((sh >> (6 * (ib % 2) + 0)) & 7) + 1));
        const float dl2 = __fmul_rn(d, (float)(2 * (int)((sh >> (6 * (ib % 2) + 3)) & 7) + 1));
        const unsigned int qh0 = (unsigned int)qh[2 * ib];
        const unsigned int qh1 = (unsigned int)qh[2 * ib + 1];
        const unsigned int idx0 = (unsigned int)qs[4 * ib + 0] | ((qh0 << 8) & 0x700u);
        const unsigned int idx1 = (unsigned int)qs[4 * ib + 1] | ((qh0 << 4) & 0x700u);
        const unsigned int idx2 = (unsigned int)qs[4 * ib + 2] | ((qh1 << 8) & 0x700u);
        const unsigned int idx3 = (unsigned int)qs[4 * ib + 3] | ((qh1 << 4) & 0x700u);
        const float d0 = (qh0 & 0x08u) ? -0.125f : 0.125f;
        const float d1 = (qh0 & 0x80u) ? -0.125f : 0.125f;
        const float d2 = (qh1 & 0x08u) ? -0.125f : 0.125f;
        const float d3 = (qh1 & 0x80u) ? -0.125f : 0.125f;
        const unsigned long long g0 = iq1s_grid[idx0];
        const unsigned long long g1 = iq1s_grid[idx1];
        const unsigned long long g2 = iq1s_grid[idx2];
        const unsigned long long g3 = iq1s_grid[idx3];
        for (int j = 0; j < 8; ++j) {
            const int v0 = (int)(signed char)((g0 >> (8 * j)) & 0xFF);
            const int v1 = (int)(signed char)((g1 >> (8 * j)) & 0xFF);
            const int v2 = (int)(signed char)((g2 >> (8 * j)) & 0xFF);
            const int v3 = (int)(signed char)((g3 >> (8 * j)) & 0xFF);
            db[y + j]      = __fmul_rn(dl1, __fadd_rn((float)v0, d0));
            db[y + 8 + j]  = __fmul_rn(dl1, __fadd_rn((float)v1, d1));
            db[y + 16 + j] = __fmul_rn(dl2, __fadd_rn((float)v2, d2));
            db[y + 24 + j] = __fmul_rn(dl2, __fadd_rn((float)v3, d3));
        }
        y += 32;
    }
}

// ---------------------------------------------------------------------------
// launchers
//
// Grid sizing lives here because Zig cannot express `<<< >>>`.  Errors are
// returned as the numeric cudaError_t (0 == cudaSuccess).
// ---------------------------------------------------------------------------

extern "C" int q35_launch_quantize_q8_0(const float *d_x, void *d_out_v,
                                        int nblocks, void *stream_v) {
    unsigned char *d_out = (unsigned char *)d_out_v;
    cudaStream_t stream = (cudaStream_t)stream_v;
    if (nblocks <= 0) return (int)cudaSuccess;
    const int threads = 128;
    const int blocks = (nblocks + threads - 1) / threads;
    q35_k_quantize_q8_0<<<blocks, threads, 0, stream>>>(d_x, d_out, nblocks);
    return (int)cudaGetLastError();
}

// Type dispatch for dequantization.  `ttype` is the Q35_TT_* id from api.h,
// which mirrors src/gguf.zig's TensorType enumeration.  Native block sizes are
// the ones in vecdotq.cuh (Q35_BYTES_*), not the 256-element GGUF row framing.
//
// The switch is factored into q35_launch_dequantize() so the row entry point
// below shares it: both differ only in how they compute `nblocks`.
static int q35_launch_dequantize_impl(int ttype, const void *d_src_v, float *d_dst,
                                      int nblocks, void *stream_v) {
    const unsigned char *d_src = (const unsigned char *)d_src_v;
    cudaStream_t stream = (cudaStream_t)stream_v;
    if (nblocks <= 0) return (int)cudaSuccess;
    const int threads = 128;
    const int blocks = (nblocks + threads - 1) / threads;
    switch (ttype) {
        case Q35_TT_Q8_0:
            q35_k_dequantize_q8_0<<<blocks, threads, 0, stream>>>(d_src, d_dst, nblocks);
            break;
        case Q35_TT_Q4_0:
            q35_k_dequantize_q4_0<<<blocks, threads, 0, stream>>>(d_src, d_dst, nblocks);
            break;
        case Q35_TT_Q4_1:
            q35_k_dequantize_q4_1<<<blocks, threads, 0, stream>>>(d_src, d_dst, nblocks);
            break;
        case Q35_TT_IQ4_NL:
            q35_k_dequantize_iq4_nl<<<blocks, threads, 0, stream>>>(d_src, d_dst, nblocks);
            break;
        case Q35_TT_F16:
            q35_k_dequantize_f16<<<blocks, threads, 0, stream>>>(d_src, d_dst, nblocks);
            break;
        case Q35_TT_BF16:
            q35_k_dequantize_bf16<<<blocks, threads, 0, stream>>>(d_src, d_dst, nblocks);
            break;
        case Q35_TT_F32:
            q35_k_copy_f32<<<blocks, threads, 0, stream>>>((const float *)d_src_v, d_dst,
                                                           nblocks);
            break;
        case Q35_TT_Q2_K:
            q35_k_dequantize_q2_k<<<blocks, threads, 0, stream>>>(d_src, d_dst, nblocks);
            break;
        case Q35_TT_Q3_K:
            q35_k_dequantize_q3_k<<<blocks, threads, 0, stream>>>(d_src, d_dst, nblocks);
            break;
        case Q35_TT_Q4_K:
            q35_k_dequantize_q4_k<<<blocks, threads, 0, stream>>>(d_src, d_dst, nblocks);
            break;
        case Q35_TT_Q5_K:
            q35_k_dequantize_q5_k<<<blocks, threads, 0, stream>>>(d_src, d_dst, nblocks);
            break;
        case Q35_TT_Q6_K:
            q35_k_dequantize_q6_k<<<blocks, threads, 0, stream>>>(d_src, d_dst, nblocks);
            break;
        case Q35_TT_IQ4_XS:
            q35_k_dequantize_iq4_xs<<<blocks, threads, 0, stream>>>(d_src, d_dst, nblocks);
            break;
        case Q35_TT_IQ3_XXS:
            q35_k_dequantize_iq3_xxs<<<blocks, threads, 0, stream>>>(d_src, d_dst, nblocks);
            break;
        case Q35_TT_IQ3_S:
            q35_k_dequantize_iq3_s<<<blocks, threads, 0, stream>>>(d_src, d_dst, nblocks);
            break;
        case Q35_TT_IQ2_XXS:
            q35_k_dequantize_iq2_xxs<<<blocks, threads, 0, stream>>>(d_src, d_dst, nblocks);
            break;
        case Q35_TT_IQ2_XS:
            q35_k_dequantize_iq2_xs<<<blocks, threads, 0, stream>>>(d_src, d_dst, nblocks);
            break;
        case Q35_TT_IQ2_S:
            q35_k_dequantize_iq2_s<<<blocks, threads, 0, stream>>>(d_src, d_dst, nblocks);
            break;
        case Q35_TT_IQ1_S:
            q35_k_dequantize_iq1_s<<<blocks, threads, 0, stream>>>(d_src, d_dst, nblocks);
            break;
        case Q35_TT_IQ1_M:
            q35_k_dequantize_iq1_m<<<blocks, threads, 0, stream>>>(d_src, d_dst, nblocks);
            break;
        default:
            return (int)cudaErrorInvalidValue;
    }
    return (int)cudaGetLastError();
}

// Native block geometry for a type, for the Zig caller's row-offset math.
// Returns 0 on an unknown type; otherwise *block_elems / *block_bytes are set.
// Lives in convert.cu rather than mmvq.cu because both files need it and each
// .cu is a separate translation unit.
extern "C" int q35_quant_geom(int ttype, int *block_elems, int *block_bytes) {
    switch (ttype) {
        case Q35_TT_Q8_0:    if (block_elems) *block_elems = 32;  if (block_bytes) *block_bytes = Q35_BYTES_Q8_0;    return 1;
        case Q35_TT_Q4_0:    if (block_elems) *block_elems = 32;  if (block_bytes) *block_bytes = Q35_BYTES_Q4_0;    return 1;
        case Q35_TT_Q4_1:    if (block_elems) *block_elems = 32;  if (block_bytes) *block_bytes = Q35_BYTES_Q4_1;    return 1;
        case Q35_TT_Q5_0:    if (block_elems) *block_elems = 32;  if (block_bytes) *block_bytes = Q35_BYTES_Q5_0;    return 1;
        case Q35_TT_Q5_1:    if (block_elems) *block_elems = 32;  if (block_bytes) *block_bytes = Q35_BYTES_Q5_1;    return 1;
        case Q35_TT_IQ4_NL:  if (block_elems) *block_elems = 32;  if (block_bytes) *block_bytes = Q35_BYTES_IQ4_NL;  return 1;
        case Q35_TT_F16:
        case Q35_TT_BF16:    if (block_elems) *block_elems = 32;  if (block_bytes) *block_bytes = 64;               return 1;
        case Q35_TT_F32:     if (block_elems) *block_elems = 32;  if (block_bytes) *block_bytes = 128;              return 1;
        case Q35_TT_Q2_K:    if (block_elems) *block_elems = 256; if (block_bytes) *block_bytes = Q35_BYTES_Q2_K;    return 1;
        case Q35_TT_Q3_K:    if (block_elems) *block_elems = 256; if (block_bytes) *block_bytes = Q35_BYTES_Q3_K;    return 1;
        case Q35_TT_Q4_K:    if (block_elems) *block_elems = 256; if (block_bytes) *block_bytes = Q35_BYTES_Q4_K;    return 1;
        case Q35_TT_Q5_K:    if (block_elems) *block_elems = 256; if (block_bytes) *block_bytes = Q35_BYTES_Q5_K;    return 1;
        case Q35_TT_Q6_K:    if (block_elems) *block_elems = 256; if (block_bytes) *block_bytes = Q35_BYTES_Q6_K;    return 1;
        case Q35_TT_IQ1_S:   if (block_elems) *block_elems = 256; if (block_bytes) *block_bytes = Q35_BYTES_IQ1_S;   return 1;
        case Q35_TT_IQ1_M:   if (block_elems) *block_elems = 256; if (block_bytes) *block_bytes = Q35_BYTES_IQ1_M;   return 1;
        case Q35_TT_IQ2_XXS: if (block_elems) *block_elems = 256; if (block_bytes) *block_bytes = Q35_BYTES_IQ2_XXS; return 1;
        case Q35_TT_IQ2_XS:  if (block_elems) *block_elems = 256; if (block_bytes) *block_bytes = Q35_BYTES_IQ2_XS;  return 1;
        case Q35_TT_IQ2_S:   if (block_elems) *block_elems = 256; if (block_bytes) *block_bytes = Q35_BYTES_IQ2_S;   return 1;
        case Q35_TT_IQ3_XXS: if (block_elems) *block_elems = 256; if (block_bytes) *block_bytes = Q35_BYTES_IQ3_XXS; return 1;
        case Q35_TT_IQ3_S:   if (block_elems) *block_elems = 256; if (block_bytes) *block_bytes = Q35_BYTES_IQ3_S;   return 1;
        case Q35_TT_IQ4_XS:  if (block_elems) *block_elems = 256; if (block_bytes) *block_bytes = Q35_BYTES_IQ4_XS;  return 1;
        default:             return 0;  // type has no CUDA implementation
    }
}

extern "C" int q35_launch_dequantize(int ttype, const void *d_src, float *d_dst,
                                     int nblocks, void *stream) {
    return q35_launch_dequantize_impl(ttype, d_src, d_dst, nblocks, stream);
}

// Row entry point: dequantize `cols` floats starting at d_src_row.  Mirrors
// src/backend_cpu.zig dequantRowInto, which walks the row in row-block units
// and never allocates.  `cols` must be a multiple of the type's native block
// size (for the 32-element family -- q4_0, q4_1, q5_0, q5_1, iq4_nl, q8_0 --
// the byte offset of a row is still cols/block_elems*block_bytes, so any
// natural GGUF width works; we do NOT require cols % 256 == 0).
extern "C" int q35_launch_dequantize_row(int ttype, const void *d_src_row,
                                         float *d_dst, int cols, void *stream) {
    if (cols <= 0) return (int)cudaSuccess;
    int block_elems = 0, block_bytes = 0;
    if (!q35_quant_geom(ttype, &block_elems, &block_bytes)) return (int)cudaErrorInvalidValue;
    (void)block_bytes;
    if (cols % block_elems != 0) return (int)cudaErrorInvalidValue;
    return q35_launch_dequantize_impl(ttype, d_src_row, d_dst, cols / block_elems,
                                      stream);
}
