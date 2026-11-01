// Gated Delta Net state machine on the device: the per-token delta-rule state
// update (`gdnStep`) and the per-token beta/gamma derivation (`gdnGamma`).
//
// WHY THESE ARE THE ONLY KERNELS WITH A WRITTEN-OUT ACCUMULATION CONTRACT
// ----------------------------------------------------------------------
// The GDN state is a 128x128 matrix that is carried across tokens: whatever
// error one step leaves behind is the STARTING POINT of the next one, and the
// model runs thousands of steps.  Every other kernel in this backend is
// pointwise or a norm, where an error stays inside one token.  So this file is
// held to a stricter bar than the rest: the accumulation ORDER is specified
// here and checked against the CPU reference, not left to the scheduler.
//
// The CPU reference is `gdnFused` in src/engine.zig (the fast path, vd == 128).
// It is a TWO-PASS algorithm, and so is this kernel:
//
//   Pass 1 (read-only over the state): km += S[i] * k[i], accumulated in
//           registers over i; the decay is applied at STORE time.
//           Concretely:  acc[j] = sum_i S[i][j]*k[i];  km[j] = acc[j]*gamma.
//   delta = beta * (v - km)                        (pointwise)
//   Pass 2: S[i][j] = S[i][j]*gamma + k[i]*delta[j];  out[j] += S[i][j]*q[i]
//           out is accumulated over i in the same i order as pass 1.
//
// Both sums are over i = 0..kd-1 IN ORDER, per output column j.  The CPU does
// them as 4 independent @Vector(32) lanes with the loop over i outermost, so
// element j of the sum sees i in increasing order and no reassociation happens
// either side.  This kernel reproduces that literally: one thread per output
// column j, sequential loop over i.
//
// WHY ONE THREAD PER COLUMN AND NOT A REDUCTION: a reduction over i would sum
// in a tree (or in warp-shuffle order), which changes the result in the last
// few ULPs.  Those ULPs are then amplified by the state carry.  Columns are
// independent (column j of pass 1/2 only ever touches column j of S), so
// "one thread per column, loop over rows" is both the CPU order and fully
// parallel.  128 columns per head = 4 warps, which is small, but this kernel
// runs once per value head per token (16-48 launches of 128 threads for the
// 0.8B) and its cost is dominated by the 64 KB state read, not by occupancy.
//
// BIT-EXACTNESS: NOT claimed, and not achievable in general -- nvcc contracts
// `a*b + c` into an FMA by default while the CPU emits separate mul and add
// (Zig does not fuse across statements), so each accumulation step differs by
// up to half an ULP.  The measured divergence is in the header of the test
// harness (tools/verify_cuda_slots, run by hand); one step is ~1e-7 relative
// per element, and it does NOT grow without bound over a long run because the
// delta rule contracts (gamma < 1 multiplies the carried state every token).
// What IS guaranteed: the loop order and therefore the error BOUND.

#include <cuda_runtime.h>
#include <math_constants.h>

#define Q35_GDN_THREADS 128

// One block per value head; thread j owns output column j of the 128x128 state.
//
// `d_state` is the [kd, vd] slice for this head. kd == vd == 128 for every
// Qwen3.5 variant (asserted host-side), so the grid is (128) threads and each
// thread walks its own column, both passes.
__global__ void q35_gdn_step_kernel(
    float *__restrict__ d_state,
    const float *__restrict__ d_k,
    const float *__restrict__ d_v,
    const float *__restrict__ d_q,
    float *__restrict__ d_km,
    float *__restrict__ d_delta,
    float *__restrict__ d_out,
    float gamma, float beta, int vd)
{
    const int j = threadIdx.x;
    if (j >= vd) return;

    // Pass 1: rate read-only sweep, mirroring engine.zig gdnFused's first loop.
    // A plain `acc += S*k` would let nvcc contract to FMA; `__fmul_rn` +
    // `__fadd_rn` pin it to the CPU's separate mul/add so the only difference
    // left is the value of the operands.  This is the one place worth the
    // intrinsics: it is 128 sequential dependent adds, the longest reduction in
    // the whole model.
    float acc = 0.0f;
    for (int i = 0; i < vd; ++i) {
        const float s = d_state[(size_t)i * vd + j];
        acc = __fadd_rn(acc, __fmul_rn(s, d_k[i]));
    }
    const float km = __fmul_rn(acc, gamma);
    d_km[j] = km;

    // delta = beta * (v - km)
    const float delta = __fmul_rn(__fsub_rn(d_v[j], km), beta);
    d_delta[j] = delta;

    // Pass 2: rewrite the column with the decayed value plus the outer-product
    // update, and accumulate S^T q in the same i order.
    float out = 0.0f;
    for (int i = 0; i < vd; ++i) {
        const size_t idx = (size_t)i * vd + j;
        const float r = __fadd_rn(__fmul_rn(d_state[idx], gamma), __fmul_rn(d_k[i], delta));
        d_state[idx] = r;
        out = __fadd_rn(out, __fmul_rn(r, d_q[i]));
    }
    d_out[j] = out;
}

// beta = sigmoid(b) in place; gamma = exp(A * softplus(a + dt_bias)).
//
// Mirrors elementwise.zig `gdnGammaKernel`, including softplus's branch:
//   softplus(x) = x > 20 ? x : log1p(exp(x))
// and `sigmoid(x) = 1/(1+exp(-x))` exactly as `elementwise.sigmoid` spells it,
// so the only divergence is expf/log1pf accuracy (see the note on
// `q35_launch_gdn_gamma` below).
__device__ __forceinline__ float q35_softplus(float x) {
    return x > 20.0f ? x : log1pf(expf(x));
}

__global__ void q35_gdn_gamma_kernel(
    const float *__restrict__ d_a_raw,
    float *__restrict__ d_b_raw,
    const float *__restrict__ d_ssm_a,
    const float *__restrict__ d_dt_bias,
    float *__restrict__ d_gamma,
    int nv)
{
    const int h = blockIdx.x * blockDim.x + threadIdx.x;
    if (h >= nv) return;
    // sigmoid(b) written back into b_raw, exactly like the CPU version.
    d_b_raw[h] = 1.0f / (1.0f + expf(-d_b_raw[h]));
    const float g = d_ssm_a[h] * q35_softplus(d_a_raw[h] + d_dt_bias[h]);
    d_gamma[h] = expf(g);
}

extern "C" int q35_launch_gdn_step(float *d_state, const float *d_k,
                                   const float *d_v, const float *d_q,
                                   float *d_km, float *d_delta, float *d_out,
                                   float gamma, float beta, int vd,
                                   void *stream)
{
    if (vd != 128) {
        // The vtable passes vd from the model config.  Every Qwen3.5 variant is
        // 128; the CPU `gdnFused` has a four-kernel fallback for anything else,
        // and the Zig side routes to it, so reaching here means the caller did
        // not check.  Refuse rather than launch a kernel that would read OOB.
        return cudaErrorInvalidValue;
    }
    q35_gdn_step_kernel<<<1, Q35_GDN_THREADS, 0, (cudaStream_t)stream>>>(
        d_state, d_k, d_v, d_q, d_km, d_delta, d_out, gamma, beta, vd);
    return (int)cudaGetLastError();
}

extern "C" int q35_launch_gdn_gamma(const float *d_a_raw, float *d_b_raw,
                                    const float *d_ssm_a,
                                    const float *d_dt_bias, float *d_gamma,
                                    int nv, void *stream)
{
    if (nv <= 0) return (int)cudaSuccess;
    const int threads = 32;
    const int blocks = (nv + threads - 1) / threads;
    q35_gdn_gamma_kernel<<<blocks, threads, 0, (cudaStream_t)stream>>>(
        d_a_raw, d_b_raw, d_ssm_a, d_dt_bias, d_gamma, nv);
    return (int)cudaGetLastError();
}
