// norm.cu -- the two normalisation slots (rmsnorm, l2norm) moved off the CPU in
// P2b batch 2.  Both are "one small reduction, then a per-element scale", so
// they are the first kernels in this backend that do NOT reduce bit-exactly:
//
//   rmsnorm  out[i] = x[i] * rsqrt(sum(x^2)/n + eps) * w[i]
//   l2norm   x[i]  *= rsqrt(sum(x^2) + eps)
//
// ---------------------------------------------------------------------------
// WHY THE ELEMENTWISE PART IS BIT-EXACT AND THE REDUCTION IS NOT
// ---------------------------------------------------------------------------
// Measured on sm_89 / CUDA 13 against glibc on this host (1 Mi samples):
//
//   sqrtf   bit-identical  (correctly rounded by IEEE-754, both sides)
//   expf    up to 2 ULP, ~30 % of inputs differ
//   cosf/sinf/powf  up to 2 ULP
//
// So `sqrtf` and the four basic operations are exact, while every libm
// transcendental in the device runtime (`__nv_expf` & co.) is a *different*
// implementation from glibc's with its own final rounding.  `rsqrtf` is worse
// than that: it is an approximation instruction, NOT correctly rounded.
//
// Consequence, and the reason this file is spelled the way it is:
//   * the sum of squares is a parallel tree reduction (`shfl_down` over a
//     warp, then over warps through shared memory), whereas the CPU reference
//     (src/elementwise.zig `rmsNorm` / `l2norm`) sums left-to-right in f32.
//     Floating-point addition is not associative, so the two sums differ in
//     the last bits.  No amount of care recovers the CPU's ORDER here without
//     a serial kernel, which would be slower than the CPU path it replaces.
//   * therefore `scale` is spread over a small neighbourhood, and it is
//     computed with `1.0f / sqrtf(...)` rather than `rsqrtf`: `sqrtf` + `/`
//     are the same two correctly-rounded operations the CPU performs, so the
//     only difference left is the sum's summation order (one rounding), not a
//     second approximation on top of it.
//
// MEASURED ERROR (200 trials x hidden=1024, random inputs; harness
// /tmp/q35parity/driver.cu, rebuilt from this file):
//   rmsnorm  max 13 ULP, max relative 8.75e-7, max absolute 1.9e-6
//   l2norm   max 12 ULP, max relative 8.33e-7, max absolute 4.1e-8
// The ULP figure is not 2 because the scale is a *product* factor: a 1-ULP
// difference in `scale` moves small elements several ULP, and `out` is then
// `x*scale*w`.  The bound that matters is the relative one, ~1e-6, and it is
// dominated by the summation order, which is the only thing this kernel does
// differently from the CPU.
//
// The elementwise multiply itself uses `__fmul_rn` for the same reason as
// elementwise.cu: nvcc's default `-fmad` may weld `x * scale * w` into an fma
// with a DIFFERENT result, so the uncontractable spelling pins it.
//
// ---------------------------------------------------------------------------
// LAYOUT / CONTRACT
// ---------------------------------------------------------------------------
// Both take raw device pointers and an f32 ELEMENT count.  `w` is the length-n
// weight of one normalised row; the engine calls these on a single contiguous
// row (`engine.zig` `self.exec.rmsnorm(qs, qs, ...)` over `q.span(h*hd, hd)`),
// never over a rows*n matrix -- so `n` IS the row, and no stride is needed.
// The `rows > 1` case is supported anyway by launching one block per row with
// `w` indexed modulo `n`: that keeps the signature able to serve the
// prefill-shaped call if the engine ever makes one, without a second kernel.
//
// One block per row, `kThreads` threads, one thread per element plus a
// grid-stride-free stride loop for rows longer than the block.  `n <= 0`
// returns cudaSuccess without launching.
//
// The reduction reads `x` twice (once for the sum, once for the scale).  `x`
// is in no case the same pointer as `out` at a different address: the engine's
// in-place form passes the SAME buffer for both, which is safe because the
// reduction finishes (and the block syncs) before any element is written.
#include "common.cuh"

namespace {

constexpr int kThreads = 256;

// Block-wide sum of `v`, in the tree order described above.
__device__ __forceinline__ float block_reduce(float v) {
    __shared__ float warp_sums[kThreads / 32];
    v += __shfl_down_sync(0xffffffffu, v, 16);
    v += __shfl_down_sync(0xffffffffu, v, 8);
    v += __shfl_down_sync(0xffffffffu, v, 4);
    v += __shfl_down_sync(0xffffffffu, v, 2);
    v += __shfl_down_sync(0xffffffffu, v, 1);
    const int lane = threadIdx.x & 31;
    const int wid = threadIdx.x >> 5;
    if (lane == 0) warp_sums[wid] = v;
    __syncthreads();
    // Only warp 0 finishes the job; the others read the broadcast result.
    v = (threadIdx.x < kThreads / 32) ? warp_sums[threadIdx.x] : 0.0f;
    if (wid == 0) {
        v += __shfl_down_sync(0xffffffffu, v, 16);
        v += __shfl_down_sync(0xffffffffu, v, 8);
        v += __shfl_down_sync(0xffffffffu, v, 4);
        v += __shfl_down_sync(0xffffffffu, v, 2);
        v += __shfl_down_sync(0xffffffffu, v, 1);
    }
    if (threadIdx.x == 0) warp_sums[0] = v;
    __syncthreads();
    return warp_sums[0];
}

// The elementwise tail of rmsnorm: out = x * scale * w.
// `__fmul_rn` twice, left to right, exactly as elementwise.zig `rmsNorm`
// spells `x * scale * w`.
__global__ void k_rmsnorm(float *__restrict__ out, const float *__restrict__ x,
                          const float *__restrict__ w, int n, float eps) {
    const int row = blockIdx.x;
    const int stride = blockDim.x;
    const float *xr = x + (size_t)row * (size_t)n;
    float *orow = out + (size_t)row * (size_t)n;

    float ss = 0.0f;
    for (int i = threadIdx.x; i < n; i += stride) ss += __fmul_rn(xr[i], xr[i]);
    // mean(x^2) + eps, then one correctly-rounded 1/sqrt -- the CPU's two
    // operations, applied to a differently-ordered sum.
    const float scale = 1.0f / sqrtf(block_reduce(ss) / (float)n + eps);
    for (int i = threadIdx.x; i < n; i += stride)
        orow[i] = __fmul_rn(__fmul_rn(xr[i], scale), w[i]);
}

__global__ void k_l2norm(float *__restrict__ x, int n, float eps) {
    const int row = blockIdx.x;
    const int stride = blockDim.x;
    float *xr = x + (size_t)row * (size_t)n;

    float ss = 0.0f;
    for (int i = threadIdx.x; i < n; i += stride) ss += __fmul_rn(xr[i], xr[i]);
    // eps INSIDE the sum, per elementwise.zig `l2norm` and SPEC A3.
    const float scale = 1.0f / sqrtf(block_reduce(ss) + eps);
    for (int i = threadIdx.x; i < n; i += stride) xr[i] = __fmul_rn(xr[i], scale);
}

} // namespace

extern "C" int q35_launch_rmsnorm(float *d_out, const float *d_x, const float *d_w,
                                  int rows, int n, float eps, void *stream_v) {
    if (n <= 0 || rows <= 0) return (int)cudaSuccess;
    cudaStream_t stream = (cudaStream_t)stream_v;
    k_rmsnorm<<<rows, kThreads, 0, stream>>>(d_out, d_x, d_w, n, eps);
    return (int)cudaGetLastError();
}

extern "C" int q35_launch_l2norm(float *d_x, int rows, int n, float eps, void *stream_v) {
    if (n <= 0 || rows <= 0) return (int)cudaSuccess;
    cudaStream_t stream = (cudaStream_t)stream_v;
    k_l2norm<<<rows, kThreads, 0, stream>>>(d_x, n, eps);
    return (int)cudaGetLastError();
}
