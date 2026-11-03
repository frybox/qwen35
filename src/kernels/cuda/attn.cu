// Full-attention decode for one token, in ONE launch: scores(q, K) -> softmax
// -> P.V -> sigmoid gate.  The KV cache lives on the device, so this is the
// slot that makes device-resident KV worthwhile -- the alternative is shipping
// (slot+1)*kv_dim floats across PCIe per layer per token.
//
// NUMERIC CONTRACT vs src/elementwise.zig `attnDecodeKernel`
// ----------------------------------------------------------
// That function is the reference and this kernel follows its structure exactly:
//
//   for h in heads:                       (this kernel: blockIdx.x == h)
//     kvh = h / gqa
//     scores[t] = (sum_j q[h*hd+j] * kcache[t*kv_dim + kvh*hd + j]) * scale
//     softmax(scores[0..slot])            (in place, CPU `softmax`)
//     accv[j] = sum_t scores[t] * vcache[t*kv_dim + kvh*hd + j]
//     gout[h*hd+j] = accv[j]
//   gout[i] *= sigmoid(gate[i])           (ONE pass over all heads, after)
//
// The three things that decide the result:
//
//   1. The dot product over j.  CPU is a serial `for (qs, ks) |a,b| acc += a*b`,
//      i.e. strictly increasing j.  Here that is a block-wide sum over j; using
//      a shared-memory/warp tree would reassociate and change the last ULPs.
//      hd is 128 for every Qwen3.5 variant, so ONE warp per head with a
//      shuffle-free approach is not possible without reassociation.  Instead:
//      thread t handles time step t (t = slot index), and the j-sum inside it is
//      serial in increasing j -- exactly the CPU order, no tree anywhere.
//   2. The softmax.  `elementwise.softmax` is: max, then exp, then sum in index
//      order, then divide.  Reproduced per the same "thread per element, reduce
//      serially" pattern below.
//   3. The gate multiply, on every head's output, AFTER all heads are written
//      (the CPU does it in `mulSigmoidVec` on the whole gout slice).  Keeping
//      that as a separate kernel preserves it.
//
// BIT-EXACTNESS: not claimed, and not needed -- nothing here is carried
// across tokens (gout is re-derived each token from the current scores), so an
// ULP cannot accumulate.  Measurement (RTX 4090 D sm_89, harness
// /tmp/slot_harness.cu vs an independent C transcription of
// `elementwise.attnDecodeKernel`):
//
//   attn probs  n=6   (slot=5,  heads=4, gqa=2, hd=8)   max_abs=0        0 ULP
//   attn gout   n=32  (slot=5,  heads=4, gqa=2, hd=8)   max_abs=5.96e-08 2 ULP
//   attn probs  n=64  (slot=63, heads=8, gqa=4, hd=16)  max_abs=1.86e-09 1 ULP
//   attn gout   n=128 (slot=63, heads=8, gqa=4, hd=16)  max_abs=2.24e-08 3 ULP
//
// A SHARED-MEMORY TRAP worth remembering (this harness caught it): static
// `__shared__` scalars and the dynamic `extern __shared__` block share one
// region, statics first.  A separate `__shared__ float s_max` therefore
// aliases `s_scores[0]` and silently destroys row 0's score -- the
// probabilities came out near-uniform with no error reported.  The max/sum are
// carved out of the dynamic block below; keep them there.
//
// LAYOUT ASSUMPTION (checked host-side): kcache/vcache rows are CONTIGUOUS,
// `slot+1` rows of kv_dim, i.e. `Buf.span(0, (slot+1)*kv_dim)` of the layer's
// cache.  `d_att` is (slot+1) floats of scratch reused by each head.

#include <cuda_runtime.h>
#include <math_constants.h>

// Upper bound for the dynamic shared-memory scratch (`scores`, 1 float per
// written row).  Qwen3.5 max_seq is well under this; a longer sequence would
// exceed the 48 KB default smem budget, so the launcher checks and refuses
// rather than silently truncating the attention window.
#define Q35_ATTN_MAX_ROWS 8192

// One block per head.  Threads cooperate differently in the three phases, but
// every sum keeps the CPU's index order.
__global__ void q35_attn_decode_kernel(
    const float *__restrict__ d_q,
    const float *__restrict__ d_gate,
    const float *__restrict__ d_kcache,
    const float *__restrict__ d_vcache,
    float *__restrict__ d_att,
    float *__restrict__ d_gout,
    int slot, int heads, int gqa, int hd, int kv_dim, float scale)
{
    const int h = blockIdx.x;
    const int tid = threadIdx.x;
    const int nt = blockDim.x;
    if (h >= heads) return;
    const int n = slot + 1;             // rows written so far
    const int kvh = h / gqa;
    const float *__restrict__ qs = d_q + (size_t)h * hd;

    // Dynamic: `n` score floats plus two scalars, ONE allocation.
    //
    // The scalars MUST be carved out of the dynamic block rather than declared
    // `__shared__` on their own: static and dynamic shared memory occupy the
    // same region, with the statics placed first, so a separate
    // `__shared__ float s_max` aliases `s_scores[0]` and the max overwrites
    // row 0's score.  (This harness caught it: at slot=63/hd=16 the device
    // probabilities came out near-uniform instead of matching the CPU --
    // every score's max was being clobbered.  Smells like UB; it is not, it is
    // the documented layout, but it is silent.)
    //
    // Layout: [0, n) scores, [n] max, [n+1] sum.
    extern __shared__ float s_scores[];
    float *const s_max = &s_scores[n];
    float *const s_sum = &s_scores[n + 1];

    // ---- scores[t] = dot(q, kcache[t]) * scale -----------------------------
    // Thread t owns row t; the j-sum is SERIAL and increasing, like the CPU.
    // A thread must not spin past `n`; rows beyond it are simply not touched.
    for (int t = tid; t < n; t += nt) {
        const float *__restrict__ ks = d_kcache + (size_t)t * kv_dim + (size_t)kvh * hd;
        float acc = 0.0f;
        for (int j = 0; j < hd; ++j) acc = __fadd_rn(acc, __fmul_rn(qs[j], ks[j]));
        s_scores[t] = __fmul_rn(acc, scale);
    }
    __syncthreads();

    // ---- softmax over scores[0..n) in place --------------------------------
    // `elementwise.softmax` order: find max (first occurrence wins on ties via
    // strict >), exp, sum in index order, divide.
    // CPU: `var max = -inf; for (x) |v| max = @max(max, v);` -- @max keeps the
    // larger operand, so on a NaN-free input this is the same as a running max.
    if (tid == 0) {
        float m = -CUDART_INF_F;
        for (int t = 0; t < n; ++t) m = fmaxf(m, s_scores[t]);
        *s_max = m;
    }
    __syncthreads();
    for (int t = tid; t < n; t += nt) s_scores[t] = expf(s_scores[t] - *s_max);
    __syncthreads();
    if (tid == 0) {
        float s = 0.0f;
        for (int t = 0; t < n; ++t) s += s_scores[t];
        *s_sum = s;
    }
    __syncthreads();
    for (int t = tid; t < n; t += nt) s_scores[t] = s_scores[t] / *s_sum;
    __syncthreads();

    // Publish this head's probabilities into the caller's scratch, matching the
    // CPU reference (which leaves them in `att`).
    for (int t = tid; t < n; t += nt) d_att[t] = s_scores[t];
    __syncthreads();

    // ---- accv[j] = sum_t scores[t] * vcache[t][j] --------------------------
    // Thread j owns output element j; the t-sum is SERIAL and increasing, which
    // is the CPU's order (its outer loop is over t, accumulating into accv[j]).
    float *__restrict__ gout = d_gout + (size_t)h * hd;
    for (int j = tid; j < hd; j += nt) {
        float acc = 0.0f;
        for (int t = 0; t < n; ++t) {
            const float w = s_scores[t];
            const float vv = d_vcache[(size_t)t * kv_dim + (size_t)kvh * hd + j];
            acc = __fadd_rn(acc, __fmul_rn(w, vv));
        }
        gout[j] = acc;
    }
}

// Gate multiply, applied to the WHOLE gout after every head is written -- the
// CPU does this as one `mulSigmoidVec(gout, gate)` call.
__global__ void q35_attn_gate_kernel(float *__restrict__ d_gout,
                                     const float *__restrict__ d_gate, int n)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    const float g = d_gate[i];
    d_gout[i] = d_gout[i] * (1.0f / (1.0f + expf(-g)));
}

// Host-stub address of the decode-attention kernel; see `q35_func_rope64` in
// rope.cu for why the graph path needs it.  Only the score/softmax/PV kernel is
// reported: its per-token variable is `slot`.  The gate kernel that follows it
// in the same launcher has no per-token argument, so it is never retargeted.
extern "C" void *q35_func_attn_decode(void) { return (void *)&q35_attn_decode_kernel; }

extern "C" int q35_launch_attn_decode(const float *d_q, const float *d_gate,
                                      const float *d_kcache,
                                      const float *d_vcache, float *d_att,
                                      float *d_gout, int slot, int heads,
                                      int gqa, int hd, int kv_dim, float scale,
                                      void *stream)
{
    if (heads <= 0 || hd <= 0) return (int)cudaSuccess;
    const int n = slot + 1;
    if (n > Q35_ATTN_MAX_ROWS) return cudaErrorInvalidValue;
    // 128 threads: enough to cover hd (128) in one pass for the per-element
    // phases, and 128 rows in flight for the per-row phases.
    // +2 floats: the two shared scalars that share the dynamic block.
    q35_attn_decode_kernel<<<heads, 128, (size_t)(n + 2) * sizeof(float), (cudaStream_t)stream>>>(
        d_q, d_gate, d_kcache, d_vcache, d_att, d_gout, slot, heads, gqa, hd,
        kv_dim, scale);
    int rc = (int)cudaGetLastError();
    if (rc != (int)cudaSuccess) return rc;
    const int total = heads * hd;
    const int threads = 128;
    const int blocks = (total + threads - 1) / threads;
    q35_attn_gate_kernel<<<blocks, threads, 0, (cudaStream_t)stream>>>(d_gout, d_gate, total);
    return (int)cudaGetLastError();
}
