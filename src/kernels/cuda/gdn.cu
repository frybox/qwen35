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

// Batched form: one block per value head, same one-thread-per-column mapping
// and the SAME accumulation statements as `q35_gdn_step_kernel` above (the
// __fmul_rn/__fadd_rn pinning is repeated verbatim, not factored behind a
// helper, so that a diff of the two bodies stays a line-by-line read).
//
// Every operand is a strided base: head h reads
//   d_state + h*state_stride, d_v/d_km/d_delta/d_out + h*vd,
//   d_k/d_q at kv head (h % nk) * kd, and its own gamma[h] / beta[h].
// Rows do not overlap (`state_stride` is kd*vd per head; each block only ever
// touches its own head's state block and its own vd-wide scratch slice), so
// `heads` blocks in parallel are race-free -- which is exactly why km/delta had
// to stop being single shared vd-float buffers (heads blocks writing one slice
// would be a silent wrong answer, not a crash).
__global__ void q35_gdn_step_rows_kernel(
    float *__restrict__ d_state, int state_stride,
    const float *__restrict__ d_k, int k_stride,
    const float *__restrict__ d_v, int v_stride,
    const float *__restrict__ d_q, int q_stride,
    float *__restrict__ d_km, float *__restrict__ d_delta,
    float *__restrict__ d_out,
    const float *__restrict__ d_gamma, const float *__restrict__ d_beta,
    int nk, int kd, int vd)
{
    const int h = blockIdx.x;
    const int j = threadIdx.x;
    if (j >= vd) return;

    const int kh = h % nk;
    const float gamma = d_gamma[h];
    const float beta = d_beta[h];
    const float *__restrict__ k_h = d_k + (size_t)kh * k_stride;
    const float *__restrict__ q_h = d_q + (size_t)kh * q_stride;
    const float *__restrict__ v_h = d_v + (size_t)h * v_stride;
    float *__restrict__ state = d_state + (size_t)h * state_stride;
    float *__restrict__ km_p = d_km + (size_t)h * vd;
    float *__restrict__ delta_p = d_delta + (size_t)h * vd;
    float *__restrict__ out_p = d_out + (size_t)h * vd;

    // Pass 1: read-only sweep.  Same intrinsics, same order as the single-row
    // kernel (see the contract at the top of this file).
    float acc = 0.0f;
    for (int i = 0; i < kd; ++i) {
        const float s = state[(size_t)i * vd + j];
        acc = __fadd_rn(acc, __fmul_rn(s, k_h[i]));
    }
    const float km = __fmul_rn(acc, gamma);
    km_p[j] = km;

    // delta = beta * (v - km)
    const float delta = __fmul_rn(__fsub_rn(v_h[j], km), beta);
    delta_p[j] = delta;

    // Pass 2: rewrite the column, accumulate S^T q in the same i order.
    float out = 0.0f;
    for (int i = 0; i < kd; ++i) {
        const size_t idx = (size_t)i * vd + j;
        const float r = __fadd_rn(__fmul_rn(state[idx], gamma), __fmul_rn(k_h[i], delta));
        state[idx] = r;
        out = __fadd_rn(out, __fmul_rn(r, q_h[i]));
    }
    out_p[j] = out;
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

extern "C" int q35_launch_gdn_step_rows(float *d_state, int state_stride,
                                        const float *d_k, int k_stride,
                                        const float *d_v, int v_stride,
                                        const float *d_q, int q_stride,
                                        float *d_km, float *d_delta,
                                        float *d_out,
                                        const float *d_gamma,
                                        const float *d_beta,
                                        int heads, int nk, int kd, int vd,
                                        void *stream)
{
    // Same shape gate as the single-row launcher: the kernel is written for the
    // 128-wide fast path only.  A zero/negative head count launches nothing
    // (the empty batch is not an error).
    if (heads <= 0) return (int)cudaSuccess;
    if (vd != 128 || kd != 128) return cudaErrorInvalidValue;
    if (nk <= 0) return cudaErrorInvalidValue;
    // The kv-head index the kernel computes must land inside the caller's row
    // array; nk > heads would make `h % nk` a head the caller never addressed.
    if (nk > heads) return cudaErrorInvalidValue;
    q35_gdn_step_rows_kernel<<<heads, Q35_GDN_THREADS, 0, (cudaStream_t)stream>>>(
        d_state, state_stride, d_k, k_stride, d_v, v_stride, d_q, q_stride,
        d_km, d_delta, d_out, d_gamma, d_beta, nk, kd, vd);
    return (int)cudaGetLastError();
}

// Chunk form: one block per value head, and the per-token loop runs INSIDE the
// kernel instead of once per launch.
//
// The steps are sequential by construction (`state` is the recurrence), so this
// buys no parallelism over `q35_gdn_step_rows_kernel` -- it buys the launch.
// A 525-token prefill used to issue 525 launches per GDN layer (10,500 total at
// 48 layers); each ran ~4 us of work but the host could not submit them faster
// than ~57 us apart, so 641 ms of a 1752 ms prefill was the host, not the GPU.
// Removing the loop entirely (measurement only, wrong results) took prefill
// 299 -> 453 tok/s, i.e. the whole overhead was ~594 ms.
//
// Everything below is `q35_gdn_step_rows_kernel`'s body verbatim -- same
// intrinsics, same i-order, same pass structure -- with the token index `t`
// advancing every strided base by one row.  The only edits are the loop and the
// per-row pointer bumps, which is what keeps the two versions comparable
// line by line.
__global__ void q35_gdn_step_rows_chunk_kernel(
    float *__restrict__ d_state, int state_stride,
    const float *__restrict__ d_k, int k_stride,
    const float *__restrict__ d_v, int v_stride,
    const float *__restrict__ d_q, int q_stride,
    float *__restrict__ d_km, int km_stride,
    float *__restrict__ d_delta, int delta_stride,
    float *__restrict__ d_out, int out_stride,
    const float *__restrict__ d_gamma, int gamma_stride,
    const float *__restrict__ d_beta, int beta_stride,
    int nk, int kd, int vd, int rows)
{
    const int h = blockIdx.x;
    const int j = threadIdx.x;
    if (j >= vd) return;

    const int kh = h % nk;
    // Per-head bases that do NOT move with the token.
    float *__restrict__ state = d_state + (size_t)h * state_stride;
    const float *__restrict__ k_h0 = d_k + (size_t)kh * kd;
    const float *__restrict__ q_h0 = d_q + (size_t)kh * kd;

    for (int t = 0; t < rows; ++t) {
        const float gamma = d_gamma[(size_t)t * gamma_stride + h];
        const float beta = d_beta[(size_t)t * beta_stride + h];
        const float *__restrict__ k_h = k_h0 + (size_t)t * k_stride;
        const float *__restrict__ q_h = q_h0 + (size_t)t * q_stride;
        const float *__restrict__ v_h = d_v + (size_t)t * v_stride + (size_t)h * vd;
        float *__restrict__ km_p = d_km + (size_t)t * km_stride + (size_t)h * vd;
        float *__restrict__ delta_p = d_delta + (size_t)t * delta_stride + (size_t)h * vd;
        float *__restrict__ out_p = d_out + (size_t)t * out_stride + (size_t)h * vd;

        // Pass 1: read-only sweep.
        float acc = 0.0f;
        for (int i = 0; i < kd; ++i) {
            const float s = state[(size_t)i * vd + j];
            acc = __fadd_rn(acc, __fmul_rn(s, k_h[i]));
        }
        const float km = __fmul_rn(acc, gamma);
        km_p[j] = km;

        // delta = beta * (v - km)
        const float delta = __fmul_rn(__fsub_rn(v_h[j], km), beta);
        delta_p[j] = delta;

        // Pass 2: rewrite the column, accumulate S^T q in the same i order.
        float out = 0.0f;
        for (int i = 0; i < kd; ++i) {
            const size_t idx = (size_t)i * vd + j;
            const float r = __fadd_rn(__fmul_rn(state[idx], gamma), __fmul_rn(k_h[i], delta));
            state[idx] = r;
            out = __fadd_rn(out, __fmul_rn(r, q_h[i]));
        }
        out_p[j] = out;
    }
}

extern "C" int q35_launch_gdn_step_rows_chunk(float *d_state, int state_stride,
                                              const float *d_k, int k_stride,
                                              const float *d_v, int v_stride,
                                              const float *d_q, int q_stride,
                                              float *d_km, int km_stride,
                                              float *d_delta, int delta_stride,
                                              float *d_out, int out_stride,
                                              const float *d_gamma, int gamma_stride,
                                              const float *d_beta, int beta_stride,
                                              int heads, int nk, int kd, int vd,
                                              int rows, void *stream)
{
    // Same shape gate as the per-row launcher: 128-wide fast path only, and a
    // zero/negative head count or row count launches nothing (an empty batch is
    // not an error).
    if (heads <= 0 || rows <= 0) return (int)cudaSuccess;
    if (vd != 128 || kd != 128) return cudaErrorInvalidValue;
    if (nk <= 0) return cudaErrorInvalidValue;
    if (nk > heads) return cudaErrorInvalidValue;
    q35_gdn_step_rows_chunk_kernel<<<heads, Q35_GDN_THREADS, 0, (cudaStream_t)stream>>>(
        d_state, state_stride, d_k, k_stride, d_v, v_stride, d_q, q_stride,
        d_km, km_stride, d_delta, delta_stride, d_out, out_stride,
        d_gamma, gamma_stride, d_beta, beta_stride, nk, kd, vd, rows);
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

// `tokens` tokens' worth of gamma/beta in ONE launch.  Token t reads its own
// `nv`-wide a/b slices (`a_stride`/`b_stride` floats apart) and writes its own
// `gamma_stride`-apart gamma row; `d_ssm_a`/`d_dt_bias` are per-head model
// weights shared by every token.  Per token the arithmetic is identical to the
// single-token launcher, so a prefill chunk replaces `tokens` launches.
__global__ void q35_gdn_gamma_rows_kernel(
    const float *__restrict__ d_a_raw, float *__restrict__ d_b_raw,
    const float *__restrict__ d_ssm_a, const float *__restrict__ d_dt_bias,
    float *__restrict__ d_gamma, int nv, int a_stride, int b_stride,
    int gamma_stride)
{
    const int h = blockIdx.x * blockDim.x + threadIdx.x;
    const int t = blockIdx.y;
    if (h >= nv) return;
    const int a_off = t * a_stride + h;
    const int b_off = t * b_stride + h;
    // sigmoid(b) written back into b_raw, exactly like the CPU version.
    d_b_raw[b_off] = 1.0f / (1.0f + expf(-d_b_raw[b_off]));
    const float g = d_ssm_a[h] * q35_softplus(d_a_raw[a_off] + d_dt_bias[h]);
    d_gamma[t * gamma_stride + h] = expf(g);
}

extern "C" int q35_launch_gdn_gamma_rows(const float *d_a_raw, float *d_b_raw,
                                         const float *d_ssm_a,
                                         const float *d_dt_bias, float *d_gamma,
                                         int nv, int tokens, int a_stride,
                                         int b_stride, int gamma_stride,
                                         void *stream)
{
    if (nv <= 0 || tokens <= 0) return (int)cudaSuccess;
    if (a_stride < nv || b_stride < nv || gamma_stride < nv)
        return (int)cudaErrorInvalidValue;
    const int threads = 32;
    const dim3 blocks((unsigned)((nv + threads - 1) / threads), (unsigned)tokens);
    q35_gdn_gamma_rows_kernel<<<blocks, threads, 0, (cudaStream_t)stream>>>(
        d_a_raw, d_b_raw, d_ssm_a, d_dt_bias, d_gamma, nv, a_stride, b_stride,
        gamma_stride);
    return (int)cudaGetLastError();
}
