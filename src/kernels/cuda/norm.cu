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

// Same computation, but the `rows` rows are `row_stride` floats apart instead of
// packed at `row * n`.  Prefill needs this: one token's q run lives at
// `qkvK[t * conv_dim]`, so the same head across `tokens` is a strided row set.
// blockIdx.y is the token, blockIdx.x the row within it, so the arithmetic per
// row is unchanged (own sum of squares, same eps, same reduce order).
__global__ void k_l2norm_strided(float *__restrict__ x, int n, float eps,
                                 int rows, int row_stride) {
    const int row = blockIdx.x;
    const int tok = blockIdx.y;
    const int stride = blockDim.x;
    float *xr = x + (size_t)tok * (size_t)row_stride + (size_t)row * (size_t)n;

    float ss = 0.0f;
    for (int i = threadIdx.x; i < n; i += stride) ss += __fmul_rn(xr[i], xr[i]);
    const float scale = 1.0f / sqrtf(block_reduce(ss) + eps);
    for (int i = threadIdx.x; i < n; i += stride) xr[i] = __fmul_rn(xr[i], scale);
}

// --- fused GDN q/k normalisation -------------------------------------------
//
// The GDN prelude normalises `nk` q rows and `nk` k rows that sit in ONE
// contiguous buffer (`qkv`: q at 0, k at `kq_dim`, v after), then scales only
// the q rows by 1/sqrt(kd).  That was three launches per layer -- two
// `q35_launch_l2norm` plus one `q35_launch_scale_vec` -- 144 per token across
// the 48 recurrent layers.
//
// Why it is worth fusing.  A decode token issues 1220 kernels, 723 of them
// non-matvec, and measured standalone each of those costs 3.7 us
// (/tmp/shp/lch.cu: l2norm 16x128 = 3.77, scale_vec = 3.69, zero_buf = 3.87 --
// all far more than a 128-element reduction, so it is launch latency, not
// work).  723 x 3.7 us is 2.7 ms of a 19.5 ms token.  Folding three launches
// into one removes 96 of them.
//
// Numerics: this is the SAME arithmetic in the SAME order.  Each row's sum of
// squares is computed by one block over that row alone, `scale = 1/sqrt(ss+eps)`
// is applied to the row, and only then is the q row multiplied by `qscale` --
// exactly `l2normRows` x2 followed by `scaleVec`.  `__fmul_rn` is kept so the
// two multiplies are not contracted into an FMA.
__global__ void k_l2norm_qk_scale(float *__restrict__ x, int nk, int kd,
                                  int k_offset, float eps, float qscale) {
    // One block per (row, which-run): blockIdx.x < nk is the q run, the rest is
    // the k run.  Both runs are `nk` rows of `kd` floats.
    const int b = blockIdx.x;
    const int row = b < nk ? b : b - nk;
    const bool is_q = b < nk;
    float *xr = x + (size_t)(is_q ? 0 : k_offset) + (size_t)row * (size_t)kd;
    const int stride = blockDim.x;
    float ss = 0.0f;
    for (int i = threadIdx.x; i < kd; i += stride) ss += __fmul_rn(xr[i], xr[i]);
    const float scale = 1.0f / sqrtf(block_reduce(ss) + eps);
    if (is_q) {
        for (int i = threadIdx.x; i < kd; i += stride)
            xr[i] = __fmul_rn(__fmul_rn(xr[i], scale), qscale);
    } else {
        for (int i = threadIdx.x; i < kd; i += stride)
            xr[i] = __fmul_rn(xr[i], scale);
    }
}

} // namespace

extern "C" int q35_launch_l2norm_qk_scale(float *d_x, int nk, int kd, int k_offset,
                                          float eps, float qscale, void *stream_v) {
    if (nk <= 0 || kd <= 0) return (int)cudaSuccess;
    if (k_offset < nk * kd) return (int)cudaErrorInvalidValue;
    cudaStream_t stream = (cudaStream_t)stream_v;
    k_l2norm_qk_scale<<<2 * nk, kThreads, 0, stream>>>(d_x, nk, kd, k_offset, eps, qscale);
    return (int)cudaGetLastError();
}

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

// `tokens * rows` normalisations in ONE launch, token t's row set starting
// `row_stride` floats past token t-1's.  `row_offset` selects which run inside
// the row block (the q run at 0, the k run at `kq_dim`); it is an element offset
// so the caller does not have to slice the Buf.
extern "C" int q35_launch_l2norm_strided(float *d_x, int row_offset, int rows, int n,
                                        float eps, int tokens, int row_stride,
                                        void *stream_v) {
    if (n <= 0 || rows <= 0 || tokens <= 0) return (int)cudaSuccess;
    if (row_stride < row_offset + rows * n) return (int)cudaErrorInvalidValue;
    cudaStream_t stream = (cudaStream_t)stream_v;
    const dim3 grid((unsigned)rows, (unsigned)tokens);
    k_l2norm_strided<<<grid, kThreads, 0, stream>>>(
        d_x + (size_t)row_offset, n, eps, rows, row_stride);
    return (int)cudaGetLastError();
}
