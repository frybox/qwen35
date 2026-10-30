// mmvq.cu -- quantized matrix-vector products, ALL weight types, plus the
// activation quantizers they need (Q8_0 for the 32-element block family, Q8_K
// for the 256-element one) and the f32 path.
//
// ---------------------------------------------------------------------------
// DESIGN: one thread per output element, one type-dispatched dot function
// ---------------------------------------------------------------------------
// Anchors:
//   ggml/src/ggml-cuda/mmvq.cu          mul_mat_vec_q (grid/shape idea)
//   ggml/src/ggml-cuda/vecdotq.cuh      per-type vec_dot_* (numerics live in
//                                       this repo's vecdotq.cuh instead)
//   src/kernel.cu:239                   k_matvec_q8 (dp4a + prefetch idea)
//
// DELIBERATE DIFFERENCE from src/kernel.cu: that kernel parallelises a row
// across a warp and reduces with shuffles.  Here one thread owns one whole row
// and accumulates strictly left to right (block by block), which is exactly
// the CPU kernel's float addition tree.  A warp reduction reorders that tree,
// was measured to be ~1 ulp off, and compounds over 150+ matvecs per token into
// a different sampled token.  The acceptance criterion is an identical token
// stream, not a tolerance -- so the one-thread-per-row mapping stays, and the
// per-type functions in vecdotq.cuh are written to fold one block at a time.
//
// The "type table" is a compile-time switch (`q35_dot_row`) rather than an
// array of device function pointers: nvcc resolves the switch to a branch, and
// keeping it as code lets every type's block loop be unrolled.  The type list,
// block sizes and activation choice come from ONE table (q35_quant_info below)
// so a new type cannot be half-wired.
//
// ---------------------------------------------------------------------------
// ACTIVATION SIDE
// ---------------------------------------------------------------------------
// 32-element family (q8_0/q4_0/q4_1/q5_0/q5_1/iq4_nl) uses Q8_0 activations:
//   34 B per 32 elements, one quantized copy per (token, column range).
// 256-element family (q2_k..q6_k, iq1_s..iq4_xs) uses Q8_K activations:
//   292 B per 256 elements, d(f32) @0, qs[256] @4, bsums[16] @260.
// Both are produced ON THE GPU (q35_launch_quantize_q8_0 / _q8_K); nothing
// falls back to the CPU.
#include "common.cuh"
#include "vecdotq.cuh"
#include "api.h"

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

// Whole-row dot: walk `nblk` native blocks of one weight row against one
// activation row, strictly left to right (see the header on reduction order).
//   float dot_row(ttype, wrow, arow, row_block_elems)
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

// ---------------------------------------------------------------------------
// matvec kernel: out[t*rows + r] = W[r] . x[t]
// ---------------------------------------------------------------------------
__global__ void q35_k_mul_mat_vec_q(const unsigned char *__restrict__ w,
                                    const unsigned char *__restrict__ xq,
                                    float *__restrict__ out,
                                    int rows, int cols, int tokens, int ttype) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int total = rows * tokens;
    if (idx >= total) return;
    const Q35QuantInfo info = q35_quant_info(ttype);
    const int nblk = cols / info.block_elems;
    const int t = idx / rows;
    const int r = idx - t * rows;

    const unsigned char *wrow = w + (size_t)r * (size_t)nblk * (size_t)info.block_bytes;
    const unsigned char *arow = xq + (size_t)t * (size_t)nblk * (size_t)info.act_bytes;
    out[(size_t)t * rows + r] =
        q35_dot_row(ttype, wrow, arow, info.block_bytes, info.act_bytes, nblk);
}

// f32 weights x f32 activation, same one-thread-per-element mapping.
__global__ void q35_k_mul_mat_vec_f32(const float *__restrict__ w,
                                      const float *__restrict__ x,
                                      float *__restrict__ out,
                                      int rows, int cols, int tokens) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= rows * tokens) return;
    const int t = idx / rows;
    const int r = idx - t * rows;
    const float *wrow = w + (size_t)r * (size_t)cols;
    const float *xrow = x + (size_t)t * (size_t)cols;
    float acc = 0.0f;
    for (int c = 0; c < cols; ++c) acc = fmaf(wrow[c], xrow[c], acc);
    out[(size_t)t * rows + r] = acc;
}

// ---------------------------------------------------------------------------
// Q8_K activation quantizer (f32 -> block_q8_K, 292 B / 256 elems)
//
// Bit-exact port of src/kernels/x86/kernels256.c:392 quantize_q8_K.  Three
// details that are easy to get wrong and that this reproduces deliberately:
//   1. `scale` goes into the header as a raw f32 (NOT rounded to f16 as Q8_0
//      does), and `inv` is built from that same f32 scale.
//   2. Rounding is roundf() = round half away from zero.
//   3. The clamp is [-127, 127], NOT [-128, 127]: the IQ kernels negate q8
//      (x86 _mm256_sign_epi8 in the CPU kernel, a plain negate on device), and
//      -128 has no positive counterpart.  The `!(r >= -127) -> 127` line is the
//      NaN/Inf guard (NaN compares false and would otherwise become garbage).
//   4. bsums[16] are int16 sums over each 16-element group of qs, little-endian,
//      consumed by the Q2_K/Q4_K/Q5_K min terms and the IQ1 delta terms.
// One thread per block (a block is 256 elements, so per-thread work is fine).
// ---------------------------------------------------------------------------
__global__ void q35_k_quantize_q8_K(const float *__restrict__ x,
                                    unsigned char *__restrict__ out,
                                    int nblocks) {
    const int b = blockIdx.x * blockDim.x + threadIdx.x;
    if (b >= nblocks) return;
    const float *xb = x + (size_t)b * Q35_QK8K;
    unsigned char *ob = out + (size_t)b * Q35_BYTES_Q8_K;

    float amax = 0.0f;
    for (int k = 0; k < Q35_QK8K; ++k) {
        const float a = xb[k] < 0.0f ? -xb[k] : xb[k];
        if (a > amax) amax = a;
    }
    const float scale = amax > 0.0f ? amax / 127.0f : 1.0f;
    const float inv = amax > 0.0f ? 1.0f / scale : 0.0f;
    const unsigned int sbits = __float_as_uint(scale);
    ob[0] = (unsigned char)(sbits & 0xFFu);
    ob[1] = (unsigned char)((sbits >> 8) & 0xFFu);
    ob[2] = (unsigned char)((sbits >> 16) & 0xFFu);
    ob[3] = (unsigned char)((sbits >> 24) & 0xFFu);

    int gsum[16];
#pragma unroll
    for (int g = 0; g < 16; ++g) gsum[g] = 0;
    for (int k = 0; k < Q35_QK8K; ++k) {
        float r = roundf(xb[k] * inv);
        if (r < -127.0f) r = -127.0f;
        if (r > 127.0f) r = 127.0f;
        if (!(r >= -127.0f)) r = 127.0f;  // NaN/Inf guard, see note 3
        const signed char q = (signed char)(int)r;
        ob[4 + k] = (unsigned char)q;
        gsum[k / 16] += (int)q;
    }
    for (int g = 0; g < 16; ++g) {
        const short v = (short)gsum[g];
        ob[260 + g * 2 + 0] = (unsigned char)((unsigned short)v & 0xFFu);
        ob[260 + g * 2 + 1] = (unsigned char)(((unsigned short)v >> 8) & 0xFFu);
    }
}

// ---------------------------------------------------------------------------
// launchers
//
// Grid sizing lives here because Zig cannot express `<<< >>>`.  Errors are
// returned as the numeric cudaError_t (0 == cudaSuccess) so the Zig side can
// surface cudaGetErrorString().
// ---------------------------------------------------------------------------

extern "C" int q35_launch_mul_mat_vec(const void *d_w_v, const void *d_xq_v,
                                      float *d_out, int rows, int cols,
                                      int tokens, int ttype, void *stream_v) {
    const unsigned char *d_w = (const unsigned char *)d_w_v;
    const unsigned char *d_xq = (const unsigned char *)d_xq_v;
    cudaStream_t stream = (cudaStream_t)stream_v;
    if (rows <= 0 || cols <= 0 || tokens <= 0) return (int)cudaSuccess;
    // Host-side validation: a wrong ttype or a cols that is not a multiple of
    // the type's block size would silently read out of bounds in the kernel.
    const Q35QuantInfo info = q35_quant_info(ttype);
    if (info.block_elems == 0) return (int)cudaErrorInvalidValue;
    if (cols % info.block_elems != 0) return (int)cudaErrorInvalidValue;
    const int threads = 128;
    const int total = rows * tokens;
    const int blocks = (total + threads - 1) / threads;
    q35_k_mul_mat_vec_q<<<blocks, threads, 0, stream>>>(d_w, d_xq, d_out, rows, cols,
                                                        tokens, ttype);
    return (int)cudaGetLastError();
}

extern "C" int q35_launch_mul_mat_vec_q8_0(const void *d_w, const void *d_xq,
                                           float *d_out, int rows, int cols,
                                           int tokens, void *stream) {
    // Backwards-compatible entry point: identical to
    // q35_launch_mul_mat_vec(..., Q35_TT_Q8_0, ...).
    return q35_launch_mul_mat_vec(d_w, d_xq, d_out, rows, cols, tokens,
                                  Q35_TT_Q8_0, stream);
}

extern "C" int q35_launch_mul_mat_vec_f32(const float *d_w, const float *d_x,
                                          float *d_out, int rows, int cols,
                                          int tokens, void *stream_v) {
    cudaStream_t stream = (cudaStream_t)stream_v;
    if (rows <= 0 || cols <= 0 || tokens <= 0) return (int)cudaSuccess;
    const int threads = 256;
    const int total = rows * tokens;
    const int blocks = (total + threads - 1) / threads;
    q35_k_mul_mat_vec_f32<<<blocks, threads, 0, stream>>>(d_w, d_x, d_out, rows, cols,
                                                          tokens);
    return (int)cudaGetLastError();
}

extern "C" int q35_launch_quantize_q8_K(const float *d_x, void *d_out_v,
                                        int nblocks, void *stream_v) {
    unsigned char *d_out = (unsigned char *)d_out_v;
    cudaStream_t stream = (cudaStream_t)stream_v;
    if (nblocks <= 0) return (int)cudaSuccess;
    const int threads = 128;
    const int blocks = (nblocks + threads - 1) / threads;
    q35_k_quantize_q8_K<<<blocks, threads, 0, stream>>>(d_x, d_out, nblocks);
    return (int)cudaGetLastError();
}

// Which activation layout a type needs, for the Zig caller's buffer sizing.
// Returns 256 if the type uses Q8_K activations, 32 if it uses Q8_0 ones, and
// 0 for an unknown type.
extern "C" int q35_quant_act_block_elems(int ttype) {
    const Q35QuantInfo info = q35_quant_info(ttype);
    if (info.block_elems == 0) return 0;
    return info.wants_q8k ? Q35_QK8K : 32;
}
