// dotq.cuh -- the per-type dot dispatch (`Q35QuantInfo` + `q35_dot_1blk` +
// `q35_dot_row`) factored out of mmvq.cu so a second .cu can call it.
//
// The bodies moved here VERBATIM from mmvq.cu (2026-11, when mmq.cu needed the
// same table); mmvq.cu now just includes this header.  They live in a header
// rather than a .cu because `q35_dot_1blk` is `__forceinline__ device` and its
// whole point is that the caller's per-type loop unrolls -- going through a
// real function would put the block bodies behind a call and the per-type
// switch would no longer be resolved at compile time (the mmvq.cu header
// explains that design).
//
// Anchor for the numerics: vecdotq.cuh (per-type block bodies, ported from
// ggml-quants.c) and mmvq.cu (the reduction-order contract).
#ifndef Q35_CUDA_DOTQ_CUH
#define Q35_CUDA_DOTQ_CUH

#include "common.cuh"
#include "vecdotq.cuh"
#include "api.h"
#include <cstdio>
#include <cstdlib>

// ---------------------------------------------------------------------------
// per-type geometry.  MUST agree with src/gguf.zig blockElems/blockBytes for
// the *native* block (not the 256-element GGUF row-block framing).
// ---------------------------------------------------------------------------
struct Q35QuantInfo {
    int block_elems;    // elements per native block (32 or 256)
    int block_bytes;    // bytes per native WEIGHT block (18 for q4_0 ...)
    int act_bytes;      // bytes per ACTIVATION block: 34 (Q8_0) or 292 (Q8_K)
    int wants_q8k;      // 1 -> Q8_K activation, 0 -> Q8_0 activation
};

// __host__ __device__ on purpose: the host launchers validate `ttype` and
// `cols` with the same table the kernel uses, so a mismatch between the two
// cannot happen.
__host__ __device__ __forceinline__ Q35QuantInfo q35_quant_info(int ttype) {
    Q35QuantInfo info;
    info.block_elems = 0;
    info.block_bytes = 0;
    // The activation block is a SEPARATE layout from the weight block and has
    // its own stride.  For the 32-element family the weight block is 18-24 B
    // but the activation is always a 34 B block_q8_0; for the 256-element
    // family it is always a 292 B block_q8_K.  Walking both with the weight
    // stride (the obvious bug) silently desynchronises the activation after
    // the first block -- and q8_0 is the one type where the two strides
    // coincide, so it passes while every other type is wrong.
    info.act_bytes = Q35_Q8_0_BLOCK_BYTES;
    info.wants_q8k = 0;
    switch (ttype) {
        case Q35_TT_Q8_0:    info.block_elems = 32;  info.block_bytes = Q35_BYTES_Q8_0;    break;
        case Q35_TT_Q4_0:    info.block_elems = 32;  info.block_bytes = Q35_BYTES_Q4_0;    break;
        case Q35_TT_Q4_1:    info.block_elems = 32;  info.block_bytes = Q35_BYTES_Q4_1;    break;
        case Q35_TT_Q5_0:    info.block_elems = 32;  info.block_bytes = Q35_BYTES_Q5_0;    break;
        case Q35_TT_Q5_1:    info.block_elems = 32;  info.block_bytes = Q35_BYTES_Q5_1;    break;
        case Q35_TT_IQ4_NL:  info.block_elems = 32;  info.block_bytes = Q35_BYTES_IQ4_NL;  break;
        case Q35_TT_Q2_K:    info.block_elems = 256; info.block_bytes = Q35_BYTES_Q2_K;    info.wants_q8k = 1; break;
        case Q35_TT_Q3_K:    info.block_elems = 256; info.block_bytes = Q35_BYTES_Q3_K;    info.wants_q8k = 1; break;
        case Q35_TT_Q4_K:    info.block_elems = 256; info.block_bytes = Q35_BYTES_Q4_K;    info.wants_q8k = 1; break;
        case Q35_TT_Q5_K:    info.block_elems = 256; info.block_bytes = Q35_BYTES_Q5_K;    info.wants_q8k = 1; break;
        case Q35_TT_Q6_K:    info.block_elems = 256; info.block_bytes = Q35_BYTES_Q6_K;    info.wants_q8k = 1; break;
        case Q35_TT_IQ1_S:   info.block_elems = 256; info.block_bytes = Q35_BYTES_IQ1_S;   info.wants_q8k = 1; break;
        case Q35_TT_IQ1_M:   info.block_elems = 256; info.block_bytes = Q35_BYTES_IQ1_M;   info.wants_q8k = 1; break;
        case Q35_TT_IQ2_XXS: info.block_elems = 256; info.block_bytes = Q35_BYTES_IQ2_XXS; info.wants_q8k = 1; break;
        case Q35_TT_IQ2_XS:  info.block_elems = 256; info.block_bytes = Q35_BYTES_IQ2_XS;  info.wants_q8k = 1; break;
        case Q35_TT_IQ2_S:   info.block_elems = 256; info.block_bytes = Q35_BYTES_IQ2_S;   info.wants_q8k = 1; break;
        case Q35_TT_IQ3_XXS: info.block_elems = 256; info.block_bytes = Q35_BYTES_IQ3_XXS; info.wants_q8k = 1; break;
        case Q35_TT_IQ3_S:   info.block_elems = 256; info.block_bytes = Q35_BYTES_IQ3_S;   info.wants_q8k = 1; break;
        case Q35_TT_IQ4_XS:  info.block_elems = 256; info.block_bytes = Q35_BYTES_IQ4_XS;  info.wants_q8k = 1; break;
        default:                                                                          break;
    }
    if (info.block_elems != 0 && info.wants_q8k) info.act_bytes = Q35_BYTES_Q8_K;
    return info;
}

// Sub-block count of a 256-element block for `ttype`, and whether the split-lane
// decode kernel has a partial-dot implementation for it.  The split kernel
// subdivides a block into `SPLIT` pieces of equal sub-block count, which is why
// only types whose 256 elements factor evenly AND that have a `_part` body are
// accepted.  Everything else keeps the whole-block path (`q35_dot_1blk`).
__device__ __host__ __forceinline__ int q35_split_units(int ttype) {
    // The unit is the native per-32 scale group, so a 256-element block is
    // always 8 units.  IQ3_XXS is the first type with a `_part` body; add to
    // this switch as each one is implemented and verified with MVCHECK.
    switch (ttype) {
        case Q35_TT_IQ3_XXS: return 8;
        case Q35_TT_IQ2_S:   return 8;
        // IQ3_S packs two 32-element sub-blocks per `scales[]` byte, so its
        // `_part` body can only start on a PAIR boundary; `q35_split_min_per`
        // reports that constraint and the launcher clamps `split` to respect it.
        case Q35_TT_IQ3_S:   return 8;
        // Q4_K splits on its four 64-element q4 BYTE ROWS, not on the sixteen
        // 16-element sub-blocks `q35_dot_scaled16` uses: only the row boundary
        // keeps each lane's 32-byte read contiguous.  Slicing on the arithmetic
        // sub-block boundary instead measured 298 -> 316 GB/s (nothing) on
        // 27B 17408x5120 because every lane read 16 strided bytes, while row
        // slicing gives 298 -> 553 GB/s.  See `q35_vec_dot_q4_k_q8_k_part`.
        case Q35_TT_Q4_K:    return 4;
        default:             return 0;   // no split path: caller uses 1blk
    }
}

// Smallest valid slice width (in 32-element units) for `ttype`'s `_part` body.
// Most bodies are fine with any width; IQ3_S packs two sub-blocks per scale byte
// and therefore needs an even, non-zero slice.  The launcher clamps `split` down
// until `units / split >= min_per`, so a body can never be handed a range it
// would have to reject.
__device__ __host__ __forceinline__ int q35_split_min_per(int ttype) {
    switch (ttype) {
        case Q35_TT_IQ3_S: return 2;
        default:           return 1;
    }
}

// The `i`-th slice `[lo, hi)` of the `units` sub-blocks of one 256-element block.
__device__ __forceinline__ float q35_dot_blk_part(int ttype, const unsigned char *w,
                                                  const unsigned char *a, int part, int split) {
    const int units = q35_split_units(ttype);
    const int per = units / split;          // units must be a multiple of split
    const int lo = part * per;
    const int hi = lo + per;
    // `split > units` would make `per == 0`: an EMPTY range, i.e. the block
    // would silently contribute nothing.  That is rejected here rather than
    // clamped, because a silent all-zeros dot looks like a fast kernel
    // (measured: SPLIT=16 on IQ3_XXS "ran" at 20 tok/s with mismatched=0
    // because it computed zero blocks).  The launcher already clamps `split`
    // down to a divisor of `units`; this is the second line of defence.
    if (per <= 0) return 0.0f;
    switch (ttype) {
        case Q35_TT_IQ3_XXS: return q35_vec_dot_iq3_xxs_q8_k_part(w, a, lo, hi);
        case Q35_TT_IQ2_S:   return q35_vec_dot_iq2_s_q8_k_part(w, a, lo, hi);
        case Q35_TT_IQ3_S:   return q35_vec_dot_iq3_s_q8_k_part(w, a, lo, hi);
        case Q35_TT_Q4_K:    return q35_vec_dot_q4_k_q8_k_part(w, a, lo, hi);
        default:             return 0.0f;
    }
}

// One activation block, folded into `acc` for weight type `ttype`.
// `a` points at ONE activation block (34 B for Q8_0, 292 B for Q8_K); the
// per-type reference is named in vecdotq.cuh next to each function.
__device__ __forceinline__ float q35_dot_1blk(int ttype, const unsigned char *w,
                                              const unsigned char *a, float acc) {
    switch (ttype) {
        case Q35_TT_Q8_0:    return q35_vec_dot_q8_0_q8_0_acc(w, a, acc);
        case Q35_TT_Q4_0:    return q35_vec_dot_q4_0_q8_0_acc(w, a, acc);
        case Q35_TT_Q4_1:    return q35_vec_dot_q4_1_q8_0_acc(w, a, acc);
        case Q35_TT_Q5_0:    return q35_vec_dot_q5_0_q8_0_acc(w, a, acc);
        case Q35_TT_Q5_1:    return q35_vec_dot_q5_1_q8_0_acc(w, a, acc);
        case Q35_TT_IQ4_NL:  return q35_vec_dot_iq4_nl_q8_0_acc(w, a, acc);
        case Q35_TT_Q2_K:    return q35_vec_dot_q2_k_q8_k_acc(w, a, acc);
        case Q35_TT_Q3_K:    return q35_vec_dot_q3_k_q8_k_acc(w, a, acc);
        case Q35_TT_Q4_K:    return q35_vec_dot_q4_k_q8_k_acc(w, a, acc);
        case Q35_TT_Q5_K:    return q35_vec_dot_q5_k_q8_k_acc(w, a, acc);
        case Q35_TT_Q6_K:    return q35_vec_dot_q6_k_q8_k_acc(w, a, acc);
        case Q35_TT_IQ1_S:   return q35_vec_dot_iq1_s_q8_k_acc(w, a, acc);
        case Q35_TT_IQ1_M:   return q35_vec_dot_iq1_m_q8_k_acc(w, a, acc);
        case Q35_TT_IQ2_XXS: return q35_vec_dot_iq2_xxs_q8_k_acc(w, a, acc);
        case Q35_TT_IQ2_XS:  return q35_vec_dot_iq2_xs_q8_k_acc(w, a, acc);
        case Q35_TT_IQ2_S:   return q35_vec_dot_iq2_s_q8_k_acc(w, a, acc);
        case Q35_TT_IQ3_XXS: return q35_vec_dot_iq3_xxs_q8_k_acc(w, a, acc);
        case Q35_TT_IQ3_S:   return q35_vec_dot_iq3_s_q8_k_acc(w, a, acc);
        case Q35_TT_IQ4_XS:  return q35_vec_dot_iq4_xs_q8_k_acc(w, a, acc);
        default:             return acc;  // unreachable: host validates ttype
    }
}

// f16 / bf16 weights (2 bytes per element, no block metadata) against the
// engine's f32 activation, left to right.  The widening is EXACT for both
// types, so the products are the same f32 values the materialized-f32 kernel
// multiplied; the loop order here is the same `c = 0..cols-1` fold that
// `q35_k_mul_mat_vec_f32` uses, INCLUDING its `fmaf` (fused multiply-add, one
// rounding).  A `__fmul_rn`+`__fadd_rn` pair would round twice per term and
// drift ~1 ulp per block, enough to flip an argmax -- see the w16 header in
// mmvq.cu for the measured consequence on the BF16 model.
__device__ __forceinline__ float q35_dot_row_w16(int ttype, const unsigned char *wrow,
                                                 const float *xrow, int cols) {
    float acc = 0.0f;
    const uint16_t *w = (const uint16_t *)wrow;
    if (ttype == Q35_TT_BF16) {
        // bf16 is the top half of an f32: shift the raw bits into the exponent
        // field.  Exact, no rounding.
        for (int c = 0; c < cols; ++c) {
            acc = fmaf(__uint_as_float((uint32_t)w[c] << 16), xrow[c], acc);
        }
    } else {
        for (int c = 0; c < cols; ++c) {
            acc = fmaf(__half2float(__ushort_as_half(w[c])), xrow[c], acc);
        }
    }
    return acc;
}

// Whole-row dot: walk `nblk` native blocks of one weight row against one
// activation row, strictly left to right (see the header on reduction order).
__device__ __forceinline__ float q35_dot_row(int ttype, const unsigned char *wrow,
                                             const unsigned char *arow,
                                             int weight_bytes, int act_bytes, int nblk) {
    // Two independent strides on purpose: weight block != activation block.
    float acc = 0.0f;
    for (int b = 0; b < nblk; ++b) {
        acc = q35_dot_1blk(ttype, wrow + (size_t)b * (size_t)weight_bytes,
                           arow + (size_t)b * (size_t)act_bytes, acc);
    }
    return acc;
}

// A warp is 32 lanes; the coalescing width of both kernels below.
#define Q35_WARP 32

#endif // Q35_CUDA_DOTQ_CUH
