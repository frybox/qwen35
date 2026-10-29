// mmvq.cu -- quantized matrix-vector products for the incremental CUDA backend.
//
// Scope of this file: Q8_0 weights x Q8_0 activation, one thread per output
// element, plus the f32-weight fallback used for the few weights that are not
// Q8_0 in a mixed GGUF (or that the backend chose to dequantize on upload).
//
// Anchors:
//   ggml/src/ggml-cuda/mmvq.cu                mul_mat_vec_q (grid/shape idea)
//   ggml/src/ggml-cuda/vecdotq.cuh:851        vec_dot_q8_0_q8_1
//   src/kernel.cu:239                         k_matvec_q8 (dp4a + prefetch idea)
//
// Difference from src/kernel.cu on purpose: that kernel parallelises a row
// across a warp and reduces with shuffles.  Here one thread owns one whole row
// and accumulates strictly left to right with fmaf, which is exactly the CPU
// kernel's float addition tree (kernels32.c:379) -- required because the
// acceptance criterion is an identical token-id stream, not a tolerance.
// A warp reduction was measured to be ~1 ulp off and would eventually flip a
// sampled token.
#include "common.cuh"
#include "vecdotq.cuh"
#include "api.h"

// out[t*rows + r] = sum_c W[r,c] * x[t,c] for T rows of activations.
// `xq` is Q8_0-quantized: T * (cols/32) blocks, 34 bytes each.
__global__ void q35_k_mul_mat_vec_q8_0(const unsigned char *__restrict__ w,
                                       const unsigned char *__restrict__ xq,
                                       float *__restrict__ out,
                                       int rows, int cols, int tokens) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int nblk = cols >> 5; // 32 elements per Q8_0 block
    if (idx >= rows * tokens) return;
    const int t = idx / rows;
    const int r = idx - t * rows;

    const unsigned char *wrow = w + (size_t)r * (size_t)nblk * Q35_Q8_0_BLOCK_BYTES;
    const unsigned char *arow = xq + (size_t)t * (size_t)nblk * Q35_Q8_0_BLOCK_BYTES;

    float acc = 0.0f;
    for (int b = 0; b < nblk; ++b) {
        acc = q35_vec_dot_q8_0_q8_0_acc(wrow + (size_t)b * Q35_Q8_0_BLOCK_BYTES,
                                        arow + (size_t)b * Q35_Q8_0_BLOCK_BYTES,
                                        acc);
    }
    out[(size_t)t * rows + r] = acc;
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
// launchers
// ---------------------------------------------------------------------------

extern "C" int q35_launch_mul_mat_vec_q8_0(const void *d_w_v, const void *d_xq_v,
                                          float *d_out, int rows, int cols,
                                          int tokens, void *stream_v) {
    const unsigned char *d_w = (const unsigned char *)d_w_v;
    const unsigned char *d_xq = (const unsigned char *)d_xq_v;
    cudaStream_t stream = (cudaStream_t)stream_v;
    if (rows <= 0 || cols <= 0 || tokens <= 0) return (int)cudaSuccess;
    const int threads = 128;
    const int total = rows * tokens;
    const int blocks = (total + threads - 1) / threads;
    q35_k_mul_mat_vec_q8_0<<<blocks, threads, 0, stream>>>(d_w, d_xq, d_out, rows, cols,
                                                           tokens);
    return (int)cudaGetLastError();
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
