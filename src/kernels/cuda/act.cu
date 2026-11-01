// act.cu -- the activation slots (silu, mulSilu, mulSigmoid, swiglu, softmax)
// moved off the CPU in P2b batch 2, and the GDN causal convolution
// (convStep).
//
//   silu        x[i]  = x[i] / (1 + exp(-x[i]))
//   mulSilu     out[i] = out[i] * (g / (1 + exp(-g)))          (g = gate[i])
//   mulSigmoid  out[i] = out[i] * (1 / (1 + exp(-g)))
//   swiglu      gate[i] = silu(gate[i]) * up[i]
//   softmax     x[i]  = exp(x[i]-max) / sum(exp(x-max))        (whole slice)
//   convStep    y[i]  = silu(4-tap FIR over the per-channel history), then the
//                       history shifts by one
//
// ---------------------------------------------------------------------------
// THE SIGMOID IS BIT-EXACT; THE TRANSCENDENTAL IS NOT
// ---------------------------------------------------------------------------
// Every one of these reduces to `expf` and then only IEEE additions,
// multiplications and one division.  Of those, `+ - * /` and `sqrtf` are
// correctly rounded on both sides, but `expf` is NOT the same implementation
// on the two sides.  Measured here (1 Mi samples across [-20,20], CUDA 13 /
// glibc, sm_89):
//
//   expf   max 2 ULP, ~30 % of inputs differ in the last bit(s)
//
// So the GPU activations are within 2 ULP of the CPU's, never bit-identical,
// and no flag fixes that (`-use_fast_math` would widen the gap, not close it;
// it is deliberately not used anywhere in this backend).  What the kernels
// below do guarantee is that the *expression* matches elementwise.zig
// character for character, so the only error introduced is the device
// runtime's `expf`, and it appears exactly once per element:
//
//   * `expf`, not `__expf`: the fast intrinsic is a ~2^-22-relative-error
//     approximation, ~10x worse in ULP terms and with its own bias; it would
//     make the gap between the two backends large enough to change sampled
//     token ids in the tail.
//   * `1.0f / (1.0f + expf(-g))` and NOT the algebraically equivalent
//     `expf(g) / (1.0f + expf(g))` -- elementwise.zig
//     `sigmoid`/`sigmoidVec`/`silu`/`mulSiluVec`/`mulSigmoidVec` all use the
//     `-g` form, and the two forms differ by up to an ULP even with a perfect
//     exponential.  Spelling the same one keeps the gap at one `expf`.
//   * `__fadd_rn`/`__fmul_rn`/`__fdiv_rn` wherever `-fmad` contraction could
//     otherwise weld neighbours: `x * (g / (1+e))` in mulSilu may not become
//     an fma with a different rounding.
//
// MEASURED ERROR (1 Mi random inputs over [-20,20]; softmax over n=1024 x 200
// trials; harness /tmp/q35parity/driver.cu, rebuilt from this file):
//   silu        max  4 ULP, max relative 3.5e-7
//   mulSilu     max  5 ULP, max relative 3.6e-7
//   mulSigmoid  max  5 ULP, max relative 3.5e-7
//   swiglu      max  5 ULP, max relative 3.7e-7
//   softmax     max 34 ULP, max relative 2.1e-6
// The activations land at ~2x the `expf` figure because the subsequent divide
// rounds again; softmax is the outlier because it stacks three sources -- the
// exp error, a block-tree denominator (the CPU sums serially), and the final
// divide -- on values that are all < 1, where ULP is tiny in absolute terms
// (max absolute error 3.2e-8).
//
// convStep's arithmetic is one multiply per tap and a fixed left-to-right sum
// per element (elementwise.zig `convStepKernel` pins the order with
// `prod[0]+prod[1]+prod[2]+prod[3]`), so the GPU match is exact up to the
// trailing `silu`.  MEASURED: max 3 ULP / 2.4e-7 relative on y, and the
// history shifts are bit-identical (they are copies).
//
// ---------------------------------------------------------------------------
// LAYOUT / CONTRACT
// ---------------------------------------------------------------------------
// Flat, one thread per element, `kThreads` = 256 like the rest of the kernel
// set, `n <= 0` returns cudaSuccess without launching (a zero-thread grid is
// cudaErrorInvalidConfiguration).  `out` and `gate` are distinct buffers in
// the engine's calls, but the kernels tolerate aliasing because each element
// is read then written by the same thread exactly once.
//
// softmax is the only reduction here: one block, `kThreads` threads, stride
// loop, two block reductions (max then sum), so it is limited to the block's
// own shared-memory tree rather than a grid sync.  The engine only ever calls
// it on one attention row (head_dim = 256 floats), which is far below that.
#include "common.cuh"

namespace {

constexpr int kThreads = 256;

inline int gridFor(int n) { return (n + kThreads - 1) / kThreads; }

// `g / (1 + exp(-g))` -- elementwise.zig `silu`.
__device__ __forceinline__ float silu_dev(float g) {
    return __fdiv_rn(g, __fadd_rn(1.0f, expf(-g)));
}

__global__ void k_silu(float *__restrict__ x, int n) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) x[i] = silu_dev(x[i]);
}

// `x * (g / (1 + exp(-g)))` -- elementwise.zig `mulSiluVec`.
__global__ void k_mul_silu(float *__restrict__ out, const float *__restrict__ gate, int n) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = __fmul_rn(out[i], silu_dev(gate[i]));
}

// `x * (1 / (1 + exp(-g)))` -- elementwise.zig `mulSigmoidVec`.
__global__ void k_mul_sigmoid(float *__restrict__ out, const float *__restrict__ gate, int n) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        const float g = gate[i];
        const float s = __fdiv_rn(1.0f, __fadd_rn(1.0f, expf(-g)));
        out[i] = __fmul_rn(out[i], s);
    }
}

// `gate = silu(gate) * up` -- elementwise.zig `bufSwiglu`, which multiplies the
// two AFTER the silu, not as `gate * up` then reduce.
__global__ void k_swiglu(float *__restrict__ gate, const float *__restrict__ up, int n) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) gate[i] = __fmul_rn(silu_dev(gate[i]), up[i]);
}

// Same tree as norm.cu's block_reduce but for both max and sum; kept local so
// this file does not grow a shared header for one extra helper.
template <bool IsMax>
__device__ __forceinline__ float block_reduce(float v) {
    __shared__ float warp_vals[kThreads / 32];
    for (int off = 16; off > 0; off >>= 1) {
        const float o = __shfl_down_sync(0xffffffffu, v, off);
        v = IsMax ? fmaxf(v, o) : __fadd_rn(v, o);
    }
    const int lane = threadIdx.x & 31;
    const int wid = threadIdx.x >> 5;
    if (lane == 0) warp_vals[wid] = v;
    __syncthreads();
    v = (threadIdx.x < kThreads / 32) ? warp_vals[threadIdx.x]
                                      : (IsMax ? -INFINITY : 0.0f);
    if (wid == 0) {
        for (int off = 16; off > 0; off >>= 1) {
            const float o = __shfl_down_sync(0xffffffffu, v, off);
            v = IsMax ? fmaxf(v, o) : __fadd_rn(v, o);
        }
    }
    if (threadIdx.x == 0) warp_vals[0] = v;
    __syncthreads();
    return warp_vals[0];
}

// softmax over the whole slice, exactly elementwise.zig `softmax`'s three
// passes: max, then exp(v-max) accumulated, then divide.  The max is essential
// (a raw exp of a logit-style value would overflow) and the CPU takes it too,
// so the two paths agree on which value is subtracted.
__global__ void k_softmax(float *__restrict__ x, int n) {
    float m = -INFINITY;
    for (int i = threadIdx.x; i < n; i += blockDim.x) m = fmaxf(m, x[i]);
    m = block_reduce<true>(m);

    float s = 0.0f;
    for (int i = threadIdx.x; i < n; i += blockDim.x) {
        const float e = expf(__fadd_rn(x[i], -m));
        x[i] = e;
        s = __fadd_rn(s, e);
    }
    s = block_reduce<false>(s);

    const float inv = __fdiv_rn(1.0f, s);
    for (int i = threadIdx.x; i < n; i += blockDim.x) x[i] = __fmul_rn(x[i], inv);
}

// GDN causal convolution: 4 taps per channel, history is 3 f32 per channel
// (oldest..newest) and is shifted in place, so `conv` and `y` are both read
// and written and `conv` is NOT optional state -- it is the sliding window.
// One thread per channel; the 4-element sum is spelled left to right to match
// elementwise.zig `convStepKernel`, and `silu` is applied afterwards over the
// whole row (which is what the CPU does, as a second pass).
__global__ void k_conv_step(float *__restrict__ conv, float *__restrict__ y,
                            const float *__restrict__ wconv, int n) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    float *hp = conv + (size_t)i * 3;
    const float cur = y[i];
    const float *wp = wconv + (size_t)i * 4;
    // `a*b` then the three additions, left to right: `__fmul_rn` per product
    // and `__fadd_rn` per add stops `-fmad` from folding `p0+p1` into the
    // next product's rounding.
    const float p0 = __fmul_rn(hp[0], wp[0]);
    const float p1 = __fmul_rn(hp[1], wp[1]);
    const float p2 = __fmul_rn(hp[2], wp[2]);
    const float p3 = __fmul_rn(cur, wp[3]);
    y[i] = __fadd_rn(__fadd_rn(__fadd_rn(p0, p1), p2), p3);
    hp[0] = hp[1];
    hp[1] = hp[2];
    hp[2] = cur;
}

__global__ void k_silu_tail(float *__restrict__ y, int n) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) y[i] = silu_dev(y[i]);
}

} // namespace

extern "C" int q35_launch_silu(float *d_x, int n, void *stream_v) {
    if (n <= 0) return (int)cudaSuccess;
    k_silu<<<gridFor(n), kThreads, 0, (cudaStream_t)stream_v>>>(d_x, n);
    return (int)cudaGetLastError();
}

extern "C" int q35_launch_mul_silu(float *d_out, const float *d_gate, int n, void *stream_v) {
    if (n <= 0) return (int)cudaSuccess;
    k_mul_silu<<<gridFor(n), kThreads, 0, (cudaStream_t)stream_v>>>(d_out, d_gate, n);
    return (int)cudaGetLastError();
}

extern "C" int q35_launch_mul_sigmoid(float *d_out, const float *d_gate, int n, void *stream_v) {
    if (n <= 0) return (int)cudaSuccess;
    k_mul_sigmoid<<<gridFor(n), kThreads, 0, (cudaStream_t)stream_v>>>(d_out, d_gate, n);
    return (int)cudaGetLastError();
}

extern "C" int q35_launch_swiglu(float *d_gate, const float *d_up, int n, void *stream_v) {
    if (n <= 0) return (int)cudaSuccess;
    k_swiglu<<<gridFor(n), kThreads, 0, (cudaStream_t)stream_v>>>(d_gate, d_up, n);
    return (int)cudaGetLastError();
}

extern "C" int q35_launch_softmax(float *d_x, int n, void *stream_v) {
    if (n <= 0) return (int)cudaSuccess;
    // ONE block: the reduction is block-wide, so `n` must fit the stride loop.
    k_softmax<<<1, kThreads, 0, (cudaStream_t)stream_v>>>(d_x, n);
    return (int)cudaGetLastError();
}

extern "C" int q35_launch_conv_step(float *d_conv, float *d_y, const float *d_wconv,
                                    int n, void *stream_v) {
    if (n <= 0) return (int)cudaSuccess;
    cudaStream_t stream = (cudaStream_t)stream_v;
    k_conv_step<<<gridFor(n), kThreads, 0, stream>>>(d_conv, d_y, d_wconv, n);
    const cudaError_t rc = cudaGetLastError();
    if (rc != cudaSuccess) return (int)rc;
    // Second launch for the activation tail; its error is what the caller sees.
    k_silu_tail<<<gridFor(n), kThreads, 0, stream>>>(d_y, n);
    return (int)cudaGetLastError();
}
