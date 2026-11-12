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

// GDN causal convolution: 4 taps per channel, history is 3 f32 per channel.
//
// Batched twin of the decode kernel below: `tokens` consecutive rows of `y`,
// `y_stride` floats apart, sharing ONE per-channel history block `conv`.
//
// Why this exists: prefill used to call `convStep` once per token (525 launches
// per GDN layer for a 525-token prompt, times two because of the silu tail).
// The convolution is a per-channel sliding window, so the tokens are NOT
// independent -- token t's taps read the history token t-1 left behind.  What
// makes batching legal is that every channel's chain is independent of every
// other channel's: one thread owns one channel and walks its own k tokens in
// ascending order, which is exactly the sequence of per-token launches the
// engine used to issue.  Same 4 products, same left-to-right `__fadd_rn` chain,
// same history shift -- the result is bit-identical.
//
// `silu` is a separate pass over the whole `tokens` x `n` region afterwards,
// preserving "silu after the entire conv" (what the engine's two-pass CPU form
// does; the old launcher also ran it as its own kernel per token).
__global__ void k_conv_step_rows(float *__restrict__ conv, float *__restrict__ y,
                                 const float *__restrict__ wconv, int n,
                                 int tokens, int y_stride) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    float *hp = conv + (size_t)i * 3;
    const float *wp = wconv + (size_t)i * 4;
    float *yp = y + (size_t)i;
    for (int t = 0; t < tokens; ++t, yp += y_stride) {
        const float cur = *yp;
        // Identical spelling to `k_conv_step_silu`: `__fmul_rn` per product and
        // `__fadd_rn` per add, so `-fmad` cannot fold `p0 + p1` into `p3`'s
        // rounding.  This is the whole numeric contract of the kernel.
        const float p0 = __fmul_rn(hp[0], wp[0]);
        const float p1 = __fmul_rn(hp[1], wp[1]);
        const float p2 = __fmul_rn(hp[2], wp[2]);
        const float p3 = __fmul_rn(cur, wp[3]);
        *yp = __fadd_rn(__fadd_rn(__fadd_rn(p0, p1), p2), p3);
        hp[0] = hp[1];
        hp[1] = hp[2];
        hp[2] = cur;
    }
}

// silu over `tokens` rows of `y` (`y_stride` apart), one thread per element run.
__global__ void k_silu_tail_rows(float *__restrict__ y, int n, int tokens, int y_stride) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    float *yp = y + (size_t)i;
    for (int t = 0; t < tokens; ++t, yp += y_stride) *yp = silu_dev(*yp);
}


// Decode-rung fused twin of `k_conv_step` + `k_silu_tail`: one launch where the
// launcher used to issue two.
//
// `q35_launch_conv_step` was already one C call, so this saves no host work --
// it saves a LAUNCH, and launch latency is what this rung is made of: a decode
// token issues 1220 kernels, 723 of them non-matvec, and each costs ~3.7 us
// standalone no matter how little it computes (/tmp/shp/lch.cu).  `conv_step`
// and `silu_tail` are 48+48 of those 723.
//
// Fusing them is sound because the two passes touch the same element with the
// same thread: `k_conv_step`'s thread i writes only `y[i]`, and
// `k_silu_tail`'s thread i reads only `y[i]`.  So applying `silu_dev` to the
// value this thread just stored is exactly the second pass, with no ordering
// question and no change to the conv arithmetic (`__fmul_rn`/`__fadd_rn` chain
// kept verbatim, which is the kernel's whole numeric contract).
__global__ void k_conv_step_silu(float *__restrict__ conv, float *__restrict__ y,
                                 const float *__restrict__ wconv, int n) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    float *hp = conv + (size_t)i * 3;
    const float cur = y[i];
    const float *wp = wconv + (size_t)i * 4;
    const float p0 = __fmul_rn(hp[0], wp[0]);
    const float p1 = __fmul_rn(hp[1], wp[1]);
    const float p2 = __fmul_rn(hp[2], wp[2]);
    const float p3 = __fmul_rn(cur, wp[3]);
    y[i] = silu_dev(__fadd_rn(__fadd_rn(__fadd_rn(p0, p1), p2), p3));
    hp[0] = hp[1];
    hp[1] = hp[2];
    hp[2] = cur;
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
    // One launch, not two: see `k_conv_step_silu`.  The activation tail used to
    // be a second kernel over the same row with the same thread mapping.
    k_conv_step_silu<<<gridFor(n), kThreads, 0, stream>>>(d_conv, d_y, d_wconv, n);
    return (int)cudaGetLastError();
}

// Batched form: `tokens` rows of `d_y`, consecutive rows `y_stride` floats
// apart, one shared per-channel history `d_conv`.  Equivalent to calling
// `q35_launch_conv_step` on each row in ascending order, in TWO launches
// instead of `2 * tokens`.  See `k_conv_step_rows` for why the per-token
// coupling does not forbid batching.
extern "C" int q35_launch_conv_step_rows(float *d_conv, float *d_y,
                                         const float *d_wconv, int n, int tokens,
                                         int y_stride, void *stream_v) {
    if (n <= 0 || tokens <= 0) return (int)cudaSuccess;
    // A stride narrower than the row would make consecutive tokens alias.
    if (y_stride < n) return (int)cudaErrorInvalidValue;
    cudaStream_t stream = (cudaStream_t)stream_v;
    k_conv_step_rows<<<gridFor(n), kThreads, 0, stream>>>(
        d_conv, d_y, d_wconv, n, tokens, y_stride);
    const cudaError_t rc = cudaGetLastError();
    if (rc != cudaSuccess) return (int)rc;
    k_silu_tail_rows<<<gridFor(n), kThreads, 0, stream>>>(d_y, n, tokens, y_stride);
    return (int)cudaGetLastError();
}
