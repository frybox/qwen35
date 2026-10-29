// convert.cu -- Q8_0 activation quantize (f32 -> block_q8_0) and Q8_0 block
// dequantize (block_q8_0 -> f32), plus their C launchers.
//
// Anchors:
//   ggml/src/ggml-cuda/convert.cu            dequantize_block_q8_0
//   ggml/src/ggml-quants.c                   quantize_row_q8_0_ref
// Numeric order is anchored to this repo's CPU kernel instead, because the
// point of the exercise is a bit-identical backend:
//   src/kernels/x86/kernels32.c:1033 quantize_q8_0
//     amax  = max|x[b*32 + k]|
//     scale = amax > 0 ? amax/127 : 1        (f32; the f16 in the header is
//                                             only the stored copy)
//     inv   = amax > 0 ? 1/scale : 0         (built from the *f32* scale --
//                                             using the f16-rounded scale here
//                                             would change q and break the md5)
//     q     = clamp(rint_half_away(x*inv), -128, 127)
// Two traps this file deliberately avoids:
//  (1) src/kernel.cu's k_quantize_q8 divides by the f16-truncated scale via
//      roundf(x/scale); that is a different rounding and is not reused here.
//  (2) The caller passes the *number of 32-element blocks*, not elements.
#include "common.cuh"
#include "vecdotq.cuh"
#include "api.h"

// ---------------------------------------------------------------------------
// device kernels
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
        if (a > amax) amax = a; // same comparison chain as the CPU loop
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

// One thread per 32-element block; writes 32 floats.
__global__ void q35_k_dequantize_q8_0(const unsigned char *__restrict__ src,
                                      float *__restrict__ dst,
                                      int nblocks) {
    const int b = blockIdx.x * blockDim.x + threadIdx.x;
    if (b >= nblocks) return;
    const unsigned char *sb = src + (size_t)b * Q35_Q8_0_BLOCK_BYTES;
    float *db = dst + (size_t)b * Q35_QK8_0;
    const float d = q35_ld_scale(sb);
    const signed char *qs = (const signed char *)(sb + 2);
#pragma unroll
    for (int k = 0; k < Q35_QK8_0; ++k) db[k] = d * (float)qs[k];
}

// ---------------------------------------------------------------------------
// C launchers.  Straight host code -- `<<< >>>` is not expressible from Zig, so
// the grid sizing lives here.  Launch errors are surfaced through
// cudaGetLastError() and turned into a Zig error with cudaGetErrorString().
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

extern "C" int q35_launch_dequantize_q8_0(const void *d_src_v, float *d_dst,
                                         int nblocks, void *stream_v) {
    const unsigned char *d_src = (const unsigned char *)d_src_v;
    cudaStream_t stream = (cudaStream_t)stream_v;
    if (nblocks <= 0) return (int)cudaSuccess;
    const int threads = 128;
    const int blocks = (nblocks + threads - 1) / threads;
    q35_k_dequantize_q8_0<<<blocks, threads, 0, stream>>>(d_src, d_dst, nblocks);
    return (int)cudaGetLastError();
}
