// elementwise.cu -- the four pure elementwise (per-element, no reduction)
// activation operations, moved off the CPU in P2b:
//
//   scaleVec   x[i] *= s
//   addResid   dst[i] += src[i]
//   copyRange  dst[i]  = src[i]
//   zeroBuf    buf[i]  = 0
//
// ---------------------------------------------------------------------------
// WHY THESE ARE TRIVIAL AND WHY THAT IS THE POINT
// ---------------------------------------------------------------------------
// None of the four has a reduction, a lookup table or a transcendental, so
// there is no summation order to preserve and no table to share.  Each element
// is one IEEE-754 single operation on the same input the CPU kernel sees,
// therefore GPU and CPU results must be **bit-identical**, not "within a
// tolerance" (the acceptance probe compares token ids, which requires exactly
// this).  Keep it that way: no fused multiply-add, no fast-math reassociation,
// no `-use_fast_math`.
//
// To make that guarantee independent of nvcc's default contraction (`-fmad`,
// which is ON and is allowed to weld a `*` and a neighbouring `+` into one
// fma), the two arithmetic kernels use the explicit round-to-nearest
// intrinsics `__fmul_rn` / `__fadd_rn`.  Those are the *uncontractable*
// spelling of exactly the operation the CPU performs, so a later edit that
// happens to add an add next to the multiply cannot silently change the
// numerics, and the file needs no special nvcc flag (the shared stage-1 flags
// in build.zig stay uniform across .cu files).
//
// Copies and zero-fill are bit moves / exact zero, so they need no intrinsic.
//
// ---------------------------------------------------------------------------
// LAYOUT / CONTRACT
// ---------------------------------------------------------------------------
// All four take raw device pointers plus an ELEMENT count (`n` of f32), never a
// byte count, and never a stride: every buffer the engine hands over is a
// contiguous span of f32 (backend.Buf).  `n <= 0` returns cudaSuccess without
// launching, which is what the Zig side does for an empty Buf -- the alternative
// (launching a zero-thread grid) is a cudaErrorInvalidConfiguration.
//
// There is one thread per element from a single flat index (no grid-stride
// loop), so the mapping is exact: element i is handled by exactly one thread
// and no element is read or written twice.  That is what makes the in-place
// `addResid` / `scaleVec` safe, and it is also why the result does not depend
// on the block size.
#include "common.cuh"

namespace {

__global__ void k_scale_vec(float *__restrict__ x, float s, int n) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) x[i] = __fmul_rn(x[i], s);
}

__global__ void k_add_resid(float *__restrict__ dst, const float *__restrict__ src, int n) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) dst[i] = __fadd_rn(dst[i], src[i]);
}

__global__ void k_copy_range(float *__restrict__ dst, const float *__restrict__ src, int n) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) dst[i] = src[i];
}

__global__ void k_zero_buf(float *__restrict__ buf, int n) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) buf[i] = 0.0f;
}

// 256 threads: same choice as mmvq.cu / convert.cu (an integer multiple of the
// warp size that keeps the launch config identical across the kernel set, so a
// profiler comparison between kernels is not confounded by occupancy).
constexpr int kThreads = 256;

inline int gridFor(int n) { return (n + kThreads - 1) / kThreads; }

} // namespace

extern "C" int q35_launch_scale_vec(float *d_x, float s, int n, void *stream_v) {
    if (n <= 0) return (int)cudaSuccess;
    cudaStream_t stream = (cudaStream_t)stream_v;
    k_scale_vec<<<gridFor(n), kThreads, 0, stream>>>(d_x, s, n);
    return (int)cudaGetLastError();
}

extern "C" int q35_launch_add_resid(float *d_dst, const float *d_src, int n, void *stream_v) {
    if (n <= 0) return (int)cudaSuccess;
    cudaStream_t stream = (cudaStream_t)stream_v;
    k_add_resid<<<gridFor(n), kThreads, 0, stream>>>(d_dst, d_src, n);
    return (int)cudaGetLastError();
}

extern "C" int q35_launch_copy_range(float *d_dst, const float *d_src, int n, void *stream_v) {
    if (n <= 0) return (int)cudaSuccess;
    cudaStream_t stream = (cudaStream_t)stream_v;
    k_copy_range<<<gridFor(n), kThreads, 0, stream>>>(d_dst, d_src, n);
    return (int)cudaGetLastError();
}

extern "C" int q35_launch_zero_buf(float *d_buf, int n, void *stream_v) {
    if (n <= 0) return (int)cudaSuccess;
    cudaStream_t stream = (cudaStream_t)stream_v;
    k_zero_buf<<<gridFor(n), kThreads, 0, stream>>>(d_buf, n);
    return (int)cudaGetLastError();
}
