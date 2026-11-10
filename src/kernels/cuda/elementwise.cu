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
#include <math_constants.h>

namespace {

__global__ void k_scale_vec(float *__restrict__ x, float s, int n) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) x[i] = __fmul_rn(x[i], s);
}

// `tokens` runs of `n` elements starting `stride` floats apart (see
// `q35_launch_scale_vec_strided`).  Same `__fmul_rn` per element.
__global__ void k_scale_vec_strided(float *__restrict__ x, float s, int n, int stride) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    float *xp = x + (size_t)blockIdx.y * (size_t)stride + (size_t)i;
    *xp = __fmul_rn(*xp, s);
}

__global__ void k_add_resid(float *__restrict__ dst, const float *__restrict__ src, int n) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) dst[i] = __fadd_rn(dst[i], src[i]);
}

__global__ void k_copy_range(float *__restrict__ dst, const float *__restrict__ src, int n) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) dst[i] = src[i];
}

// Split `[h][q_head | gate_head]` into two dense head-major arrays.
//
// One block per head, `hd` elements in, `2*hd` out: the source block for head
// `h` is contiguous (two `hd` runs), so a block copy is the natural shape and
// the whole thing is one launch for the entire projection instead of `2*heads`
// launches of one head each.  That is the entire point -- the bytes are the
// same, the launch count is not (`PROF-COPY` put the per-head form at 768
// calls/token and 99% of the `copy_range` slot).
//
// EXACTNESS: a pure move, no arithmetic, so it is bit-identical to
// `elementwise.bufDeinterleaveQG` by construction.
__global__ void k_deinterleave_qg(float *__restrict__ q, float *__restrict__ gate,
                                 const float *__restrict__ src, int hd) {
    const int h = blockIdx.x;
    const float *s = src + (size_t)h * 2 * hd;
    for (int i = threadIdx.x; i < hd; i += blockDim.x) {
        q[(size_t)h * hd + i] = s[i];
        gate[(size_t)h * hd + i] = s[hd + i];
    }
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

// `tokens * n` elements scaled in ONE launch: token t's run starts `stride`
// floats past token t-1's.  Same per-element multiply as `q35_launch_scale_vec`,
// so a prefill chunk gets one launch where it used to get one per token.
extern "C" int q35_launch_scale_vec_strided(float *d_x, int offset, float s, int n,
                                          int tokens, int stride, void *stream_v) {
    if (n <= 0 || tokens <= 0) return (int)cudaSuccess;
    if (stride < offset + n) return (int)cudaErrorInvalidValue;
    cudaStream_t stream = (cudaStream_t)stream_v;
    k_scale_vec_strided<<<dim3((unsigned)gridFor(n), (unsigned)tokens), kThreads, 0,
                          stream>>>(d_x + (size_t)offset, s, n, stride);
    return (int)cudaGetLastError();
}

extern "C" int q35_launch_add_resid(float *d_dst, const float *d_src, int n, void *stream_v) {
    if (n <= 0) return (int)cudaSuccess;
    cudaStream_t stream = (cudaStream_t)stream_v;
    k_add_resid<<<gridFor(n), kThreads, 0, stream>>>(d_dst, d_src, n);
    return (int)cudaGetLastError();
}

// Host-stub address of the copy kernel, so a replaying decode graph can find
// this node's captured arguments (see `q35_func_rope64` for the mechanism).
// The KV append is why it matters: its destination is `cache + slot * kv_dim`,
// a pointer that moves once per token.
extern "C" void *q35_func_copy_range(void) { return (void *)&k_copy_range; }

extern "C" int q35_launch_copy_range(float *d_dst, const float *d_src, int n, void *stream_v) {
    if (n <= 0) return (int)cudaSuccess;
    cudaStream_t stream = (cudaStream_t)stream_v;
    k_copy_range<<<gridFor(n), kThreads, 0, stream>>>(d_dst, d_src, n);
    return (int)cudaGetLastError();
}

extern "C" int q35_launch_deinterleave_qg(float *d_q, float *d_gate, const float *d_src,
                                          int heads, int hd, void *stream_v) {
    if (heads <= 0 || hd <= 0) return (int)cudaSuccess;
    cudaStream_t stream = (cudaStream_t)stream_v;
    // `hd` threads per block is 256 for every model in this family (head_dim is
    // 128 or 256), i.e. the same as `kThreads`; the loop in the kernel handles
    // a larger `hd` without changing the launch config.
    k_deinterleave_qg<<<heads, hd < kThreads ? hd : kThreads, 0, stream>>>(d_q, d_gate, d_src, hd);
    return (int)cudaGetLastError();
}

// Publish one int32 into device memory from a 1-thread kernel.
//
// This is the CUDA-graph escape hatch for "a scalar launch argument that changes
// every token": if the value lives in DEVICE memory, the captured node's
// argument block and shared-memory size stay constant, so the node never needs
// `cudaGraphExecKernelNodeSetParams`.  Same trick as `k_rope64_setenv`
// (rope.cu:182), generalised so any kernel can take a per-token int this way.
// Measured motivation: the per-layer, per-token SetParams on the attn_decode
// node was costing 14.3 ms/token of the 27B's 33.8 ms decode.
__global__ void k_set_int1(int *d, int v) { *d = v; }

extern "C" int q35_launch_set_int1(int *d, int v, void *stream_v) {
    if (d == nullptr) return (int)cudaErrorInvalidValue;
    k_set_int1<<<1, 1, 0, (cudaStream_t)stream_v>>>(d, v);
    return (int)cudaGetLastError();
}

extern "C" int q35_launch_zero_buf(float *d_buf, int n, void *stream_v) {
    if (n <= 0) return (int)cudaSuccess;
    cudaStream_t stream = (cudaStream_t)stream_v;
    k_zero_buf<<<gridFor(n), kThreads, 0, stream>>>(d_buf, n);
    return (int)cudaGetLastError();
}

// ---------------------------------------------------------------------------
// argmax -- the one REDUCTION here, because it cannot be avoided: `sample`
// needs the index of the largest logit and the engine holds logits in device
// memory (P2c).  Doing it on the host would mean a D2H of the whole vocabulary
// per token, which is exactly the kind of transfer P2c removes.
//
// EXACTNESS: a max over floats is order-independent (no rounding), so the
// requirement is only that the TIE-BREAK matches `elementwise.argmaxKernel`:
// it uses `v > best` scanning ascending, so the FIRST index of the maximum
// wins.  This kernel therefore reduces (value, index) pairs with
// "greater value wins; equal value keeps the SMALLER index".
// ---------------------------------------------------------------------------

// Block reduction over (value, index).  Each thread scans a strided slice, then
// the block folds 256 candidates down in shared memory.
__global__ void k_argmax(const float *__restrict__ d_x, int n, int *d_out) {
    __shared__ float s_val[256];
    __shared__ int s_idx[256];
    const int tid = threadIdx.x;
    float best = -CUDART_INF_F;
    int best_i = 0;
    for (int i = tid; i < n; i += blockDim.x) {
        const float v = d_x[i];
        // Strict > keeps the first index on ties within a thread's slice; the
        // (value, index) combine below extends that across threads.
        if (v > best) { best = v; best_i = i; }
    }
    s_val[tid] = best;
    s_idx[tid] = best_i;
    __syncthreads();
    for (int stride = blockDim.x >> 1; stride > 0; stride >>= 1) {
        if (tid < stride) {
            if (s_val[tid + stride] > s_val[tid] ||
                (s_val[tid + stride] == s_val[tid] && s_idx[tid + stride] < s_idx[tid])) {
                s_val[tid] = s_val[tid + stride];
                s_idx[tid] = s_idx[tid + stride];
            }
        }
        __syncthreads();
    }
    if (tid == 0) *d_out = s_idx[0];
}

// ---------------------------------------------------------------------------
// argmax: TWO-STAGE, MANY BLOCKS.
//
// The single-block version above (`k_argmax<<<1,256>>>`) scans all `n` logits
// from ONE SM.  On the 27B the lm_head is 248320 rows, i.e. ~1 MB of f32, and a
// lone SM streams that at roughly 1/114 of the chip's bandwidth: measured
// 13.4 ms of DEVICE time per token for `argmax` alone, which was 34% of the
// whole decode step (35.4 ms total, of which the matvecs were only 22.4 ms).
// The wait showed up as "sample"/"argmax" in PROF-SLOT and looked like host
// overhead; it was one block reading 1 MB.
//
// Stage 1 runs `blocks` blocks of 256 threads over strided slices and writes
// one (value, index) pair each.  Stage 2 folds those `blocks` pairs with a
// single block.  Ties still resolve to the SMALLER index -- the same
// "value, then index" combine, so the answer is identical to the serial scan --
// which is why this can replace the old kernel without changing the token
// stream.
// ---------------------------------------------------------------------------
constexpr int kArgmaxThreads = 256;
constexpr int kArgmaxMaxBlocks = 512;

__device__ __forceinline__ void argmax_combine(float &bv, int &bi, float v, int i) {
    if (v > bv || (v == bv && i < bi)) { bv = v; bi = i; }
}

// Reduce (value, index) across one block into lane 0 of `s_val`/`s_idx`.
__device__ __forceinline__ void argmax_block_reduce(float *s_val, int *s_idx, int tid) {
    __syncthreads();
    for (int stride = kArgmaxThreads >> 1; stride > 0; stride >>= 1) {
        if (tid < stride) {
            float v = s_val[tid + stride];
            int i = s_idx[tid + stride];
            argmax_combine(s_val[tid], s_idx[tid], v, i);
        }
        __syncthreads();
    }
}

__global__ void k_argmax_partial(const float *__restrict__ d_x, int n,
                                 float *__restrict__ d_pv, int *__restrict__ d_pi,
                                 int nblocks) {
    __shared__ float s_val[kArgmaxThreads];
    __shared__ int s_idx[kArgmaxThreads];
    const int tid = threadIdx.x;
    // Interleave block and thread so consecutive warps read consecutive memory
    // and the whole grid strides the array exactly once.
    const int stride = blockDim.x * nblocks;
    float best = -CUDART_INF_F;
    int best_i = 0;
    for (int i = blockIdx.x * blockDim.x + tid; i < n; i += stride) {
        const float v = d_x[i];
        if (v > best) { best = v; best_i = i; }
    }
    s_val[tid] = best;
    s_idx[tid] = best_i;
    argmax_block_reduce(s_val, s_idx, tid);
    if (tid == 0) { d_pv[blockIdx.x] = s_val[0]; d_pi[blockIdx.x] = s_idx[0]; }
}

__global__ void k_argmax_final(const float *__restrict__ d_pv, const int *__restrict__ d_pi,
                               int m, int *__restrict__ d_out) {
    __shared__ float s_val[kArgmaxThreads];
    __shared__ int s_idx[kArgmaxThreads];
    const int tid = threadIdx.x;
    float best = -CUDART_INF_F;
    int best_i = 0;
    for (int i = tid; i < m; i += blockDim.x) argmax_combine(best, best_i, d_pv[i], d_pi[i]);
    s_val[tid] = best;
    s_idx[tid] = best_i;
    argmax_block_reduce(s_val, s_idx, tid);
    if (tid == 0) *d_out = s_idx[0];
}

extern "C" int q35_launch_argmax(const float *d_x, int n, int *d_out, void *stream_v) {
    if (n <= 0) return (int)cudaSuccess;
    cudaStream_t stream = (cudaStream_t)stream_v;
    if (n <= kArgmaxThreads * 8) {
        k_argmax<<<1, 256, 0, stream>>>(d_x, n, d_out);
        return (int)cudaGetLastError();
    }
    // ONE scratch allocation, made on the first call and never resized: this
    // launcher also runs INSIDE a CUDA graph capture (decode replays `argmax`
    // as a graph node), and `cudaMalloc` during a capture is illegal -- it
    // aborts the graph, exactly as `ensureRopeDyn`'s comment records for the
    // rope channel.  Fixed size for `kArgmaxMaxBlocks`, allocated once.
    static float *s_pv = nullptr;
    static int *s_pi = nullptr;
    if (s_pv == nullptr) {
        if (cudaMalloc((void **)&s_pv, sizeof(float) * kArgmaxMaxBlocks) != cudaSuccess) {
            s_pv = nullptr;
            return (int)cudaGetLastError();
        }
        if (cudaMalloc((void **)&s_pi, sizeof(int) * kArgmaxMaxBlocks) != cudaSuccess) {
            cudaFree(s_pv);
            s_pv = nullptr;
            return (int)cudaGetLastError();
        }
    }
    int blocks = (n + kArgmaxThreads * 4 - 1) / (kArgmaxThreads * 4);
    if (blocks > kArgmaxMaxBlocks) blocks = kArgmaxMaxBlocks;
    if (blocks < 1) blocks = 1;
    k_argmax_partial<<<blocks, kArgmaxThreads, 0, stream>>>(d_x, n, s_pv, s_pi, blocks);
    k_argmax_final<<<1, kArgmaxThreads, 0, stream>>>(s_pv, s_pi, blocks, d_out);
    return (int)cudaGetLastError();
}
