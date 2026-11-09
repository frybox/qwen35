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


#ifndef Q35_ATTN_SMEM_DYNSLOT
#define Q35_ATTN_SMEM_DYNSLOT ((Q35_ATTN_MAX_ROWS + 2) * (int)sizeof(float))
#endif

// Device-slot variant of the kernel above: `n` comes from DEVICE memory and the
// dynamic shared block is always `Q35_ATTN_SMEM_DYNSLOT` bytes.
//
// WHY THIS EXISTS -- it is a CUDA-graph fix, not a kernel change.
// The score/softmax/PV kernel was the one node whose retarget needed
// `cudaGraphExecKernelNodeSetParams` on EVERY layer for EVERY token, because
// BOTH of its per-token inputs change:
//   * `slot` (the KV prefix length), and
//   * `shared_mem_bytes` (`(n+2)*sizeof(float)`), which the graph exec
//     snapshots at SetParams time and cannot be updated any other way.
// Measured on the 27B IQ3_XXS: with the graph enabled, ablating `attn_decode`
// took the token from 33.8 to 19.4 ms -- 14.3 ms, 43% of decode -- while the
// same kernel measured ALONE is 0.058 ms at n=300 (x24 layers = 1.4 ms).
// With `QWEN35_CUDA_GRAPH=0` the same ablation saves only 2.2 ms.  So the cost
// was never the kernel: it was ~24 `GraphExecKernelNodeSetParams` calls per
// token, each of which forces the graph exec to re-apply its node list.
//
// Fix: make `n` a pointer argument and keep `shared_mem_bytes` CONSTANT, so the
// decode graph never retargets this node at all.  The block is sized for the
// maximum window; shared memory does not limit occupancy here (the limit is 24
// blocks x 128 threads on 114 SMs).
//
// The arithmetic is the scalar-`slot` kernel's, term for term: same phase order,
// same increasing CPU index order in every reduction, same `-inf` start.
__global__ void q35_attn_decode_kernel_dyn(
    const float *__restrict__ d_q,
    const float *__restrict__ d_gate,
    const float *__restrict__ d_kcache,
    const float *__restrict__ d_vcache,
    float *__restrict__ d_att,
    float *__restrict__ d_gout,
    const int *__restrict__ d_slot,
    int heads, int gqa, int hd, int kv_dim, float scale)
{
    const int h = blockIdx.x;
    const int tid = threadIdx.x;
    const int nt = blockDim.x;
    if (h >= heads) return;
    const int n = *d_slot + 1;

    const int kvh = h / gqa;
    const float *__restrict__ qs = d_q + (size_t)h * hd;

    extern __shared__ float s_scores[];
    float *const s_max = &s_scores[n];
    float *const s_sum = &s_scores[n + 1];

    for (int t = tid; t < n; t += nt) {
        const float *__restrict__ ks = d_kcache + (size_t)t * kv_dim + (size_t)kvh * hd;
        float acc = 0.0f;
        for (int j = 0; j < hd; ++j) acc = __fadd_rn(acc, __fmul_rn(qs[j], ks[j]));
        s_scores[t] = __fmul_rn(acc, scale);
    }
    __syncthreads();

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

    for (int t = tid; t < n; t += nt) d_att[t] = s_scores[t];
    __syncthreads();

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

// Address of the device-slot kernel for the graph-capture path.
extern "C" void *q35_func_attn_decode_dyn(void) { return (void *)&q35_attn_decode_kernel_dyn; }

// Launch the device-slot variant: `slot` is read from DEVICE memory (`d_slot`,
// one int) and the dynamic shared block is the fixed maximum, so neither the
// argument block nor `shared_mem_bytes` changes between tokens and a captured
// node never needs retargeting.  The caller publishes `d_slot` (see
// `q35_launch_set_int1`).
extern "C" int q35_launch_attn_decode_dyn(const float *d_q, const float *d_gate,
                                          const float *d_kcache,
                                          const float *d_vcache, float *d_att,
                                          float *d_gout, const int *d_slot,
                                          int heads, int gqa, int hd, int kv_dim,
                                          float scale, void *stream)
{
    if (heads <= 0 || hd <= 0) return (int)cudaSuccess;
    if (d_slot == nullptr) return (int)cudaErrorInvalidValue;
    q35_attn_decode_kernel_dyn<<<heads, 128, (size_t)Q35_ATTN_SMEM_DYNSLOT, (cudaStream_t)stream>>>(
        d_q, d_gate, d_kcache, d_vcache, d_att, d_gout, d_slot, heads, gqa, hd,
        kv_dim, scale);
    int rc = (int)cudaGetLastError();
    if (rc != (int)cudaSuccess) return rc;
    const int total = heads * hd;
    const int threads = 128;
    const int blocks = (total + threads - 1) / threads;
    q35_attn_gate_kernel<<<blocks, threads, 0, (cudaStream_t)stream>>>(d_gout, d_gate, total);
    return (int)cudaGetLastError();
}

// ---------------------------------------------------------------------------
// TUNED decode attention (block = one head, but the row loops are WARP-TILED).
//
// WHY: the naive kernel above assigns row `t` to thread `t`, so consecutive
// threads read `d_kcache + t*kv_dim + ...`, i.e. 4 KB apart.  Every 32-thread
// warp then touches 32 distinct cache lines per j-step and uses only `hd` of the
// `kv_dim` floats in each row: the read is uncoalesced AND over-fetched by
// `kv_dim/hd` (= 4x for the 27B, where kv_dim=1024 but hd=256).
//
// MEASURED (`/tmp/mb/mb_attn2.cu`, 24 layers with DISTINCT KV caches so the
// working set is 102-180 MB and cannot hide in the 72 MB L2): the naive kernel
// does 33-37 GB/s, i.e. 3.0 ms/token at slot=518 and 5.0 ms at slot=917.
// (A first measurement of 0.058 ms at n=300 was an L2 artifact -- the same trap
// documented for the matvec microbenches.)
//
// FIX: one warp per (row group, j-tile).  Lane `l` handles j = jt + l, so a warp
// reads ONE contiguous 128-byte segment per iteration, and rows are processed
// `warps_per_block` at a time.  The j-sum for a score is then a warp reduction;
// to keep the CPU's strict increasing-index order the reduction uses
// `__shfl_down_sync` on a tree -- which is NOT the serial order, so the scores
// differ from the reference in the last bits.
//
// This variant is therefore OPT-IN and NOT the default.  The default keeps the
// serial order because the project's gate for a non-split change is bit-exact
// token equality; it is kept here with its measurement so the trade-off is
// visible rather than rediscovered.
__global__ void q35_attn_decode_kernel_coal(
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
    const int warp = tid >> 5, lane = tid & 31;
    const int nwarps = nt >> 5;
    if (h >= heads) return;
    const int n = slot + 1;
    const int kvh = h / gqa;
    const float *__restrict__ qs = d_q + (size_t)h * hd;

    extern __shared__ float s_scores[];
    float *const s_max = &s_scores[n];
    float *const s_sum = &s_scores[n + 1];

    // ---- scores: one warp per row, lanes tile j ----------------------------
    for (int t = warp; t < n; t += nwarps) {
        const float *__restrict__ ks = d_kcache + (size_t)t * kv_dim + (size_t)kvh * hd;
        float acc = 0.0f;
        for (int j = lane; j < hd; j += 32) acc = __fadd_rn(acc, __fmul_rn(qs[j], ks[j]));
        for (int o = 16; o > 0; o >>= 1) acc = __fadd_rn(acc, __shfl_down_sync(0xffffffffu, acc, o));
        if (lane == 0) s_scores[t] = __fmul_rn(acc, scale);
    }
    __syncthreads();

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
    for (int t = tid; t < n; t += nt) d_att[t] = s_scores[t];
    __syncthreads();

    // ---- PV: warp per j-tile, rows t walked in order -----------------------
    // Lane `l` holds j = jt + l, so the warp reads one contiguous 128-byte run of
    // `vcache[t]` per row.  Every lane walks `t` in the SAME increasing order, so
    // each output element keeps the CPU's accumulation order exactly (this is
    // why the t-loop is outside the shuffle: unlike the score reduction, no
    // reassociation happens here).
    float *__restrict__ gout = d_gout + (size_t)h * hd;
    float acc = 0.0f;
    int j = warp * 32 + lane;
    const int jstep = nwarps * 32;
    for (; j < hd; j += jstep) {
        float a = 0.0f;
        const float *__restrict__ vcol = d_vcache + (size_t)kvh * hd + j;
        for (int t = 0; t < n; ++t)
            a = __fadd_rn(a, __fmul_rn(s_scores[t], vcol[(size_t)t * kv_dim]));
        gout[j] = a;
    }
    (void)acc;
}

extern "C" void *q35_func_attn_decode_coal(void) { return (void *)&q35_attn_decode_kernel_coal; }

// Launch the warp-tiled variant.  Uses 256 threads (8 warps) per block so a
// block covers 8 rows at once instead of 4 with 128 -- the score phase is the
// uncoalesced one and this is where the width pays.
extern "C" int q35_launch_attn_decode_coal(const float *d_q, const float *d_gate,
                                           const float *d_kcache,
                                           const float *d_vcache, float *d_att,
                                           float *d_gout, int slot, int heads,
                                           int gqa, int hd, int kv_dim, float scale,
                                           void *stream)
{
    if (heads <= 0 || hd <= 0) return (int)cudaSuccess;
    const int n = slot + 1;
    if (n > Q35_ATTN_MAX_ROWS) return cudaErrorInvalidValue;
    q35_attn_decode_kernel_coal<<<heads, 256, (size_t)(n + 2) * sizeof(float), (cudaStream_t)stream>>>(
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

// ---------------------------------------------------------------------------
// SPLIT variant: the row walk is divided across `SPLIT` blocks per head.
//
// The warp-tiled kernel above is still only 68-73 GB/s because ONE block per head
// gives 24 blocks x 256 threads = 6144 threads on 114 SMs (21% occupancy) and
// every lane then walks all `n` rows serially.  This variant launches
// `heads * SPLIT` blocks; block `s` handles rows `t % SPLIT == s`... no: it
// handles the contiguous slice [s*n/SPLIT, (s+1)*n/SPLIT), and the PV partials
// are summed in slice order by a second kernel.
//
// CONSEQUENCE: the score for row t is still the serial j-sum (bit-exact), but the
// PV sum is now a 2-level tree instead of one serial walk, so `gout` differs from
// the reference in the last bits.  That is the SAME trade the decode matvec split
// path already makes (`mmvq.cu`), and it is gated the same way: integer-exact
// token IDs plus zero MVCHECK mismatches, not bit equality.
#define Q35_ATTN_SPLIT_MAX 16

__global__ void q35_attn_decode_kernel_split(
    const float *__restrict__ d_q,
    const float *__restrict__ d_gate,
    const float *__restrict__ d_kcache,
    const float *__restrict__ d_vcache,
    float *__restrict__ d_att,
    float *__restrict__ d_gout,
    float *__restrict__ d_part,   // [heads][SPLIT][hd]
    const float *__restrict__ d_prob,  // [n] probabilities from the score kernel
    int slot, int heads, int gqa, int hd, int kv_dim, float scale, int split)
{
    const int h = blockIdx.x / split;
    const int s = blockIdx.x % split;
    const int tid = threadIdx.x;
    const int nt = blockDim.x;
    const int warp = tid >> 5, lane = tid & 31;
    const int nwarps = nt >> 5;
    if (h >= heads) return;
    const int n = slot + 1;
    const int kvh = h / gqa;
    const float *__restrict__ qs = d_q + (size_t)h * hd;

    // Contiguous slice of rows for this block (last block takes the remainder).
    const int per = (n + split - 1) / split;
    const int t0 = s * per;
    const int t1 = min(n, t0 + per);
    const int cnt = t1 > t0 ? t1 - t0 : 0;
    extern __shared__ float s_scratch_unused[];

    // NOTE the softmax is NOT computed here.  Each block used to normalise by
    // its OWN slice's max/sum, which is wrong: the softmax is over the whole row.
    // The first version did exactly that and was caught by the microbench
    // (`mismatched=6144`, worst error 0.35 = the scale of the output itself).
    // So probabilities come from `d_prob`, written by the score kernel.
    const float *__restrict__ prob = d_prob + (size_t)h * (size_t)Q35_ATTN_MAX_ROWS + (size_t)t0;

    // PV over this block's slice -> d_part[h][s][j].
    float *__restrict__ part = d_part + ((size_t)h * split + s) * (size_t)hd;
    for (int j = warp * 32 + lane; j < hd; j += nwarps * 32) {
        float a = 0.0f;
        const float *__restrict__ vcol = d_vcache + (size_t)kvh * hd + j;
        for (int i = 0; i < cnt; ++i)
            a = __fadd_rn(a, __fmul_rn(prob[i], vcol[(size_t)(t0 + i) * kv_dim]));
        part[j] = a;
    }
    (void)qs; (void)s_scratch_unused;
}

// Reduce the per-block PV partials in slice order, apply the gate.
__global__ void q35_attn_reduce_kernel(const float *__restrict__ d_part,
                                       const float *__restrict__ d_gate,
                                       float *__restrict__ d_gout, int heads,
                                       int hd, int split)
{
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= heads * hd) return;
    const int h = idx / hd, j = idx % hd;
    const float *__restrict__ p = d_part + ((size_t)h * split) * (size_t)hd + j;
    float a = 0.0f;
    for (int s = 0; s < split; ++s) a = __fadd_rn(a, p[(size_t)s * hd]);
    const float g = d_gate[idx];
    d_gout[idx] = a * (1.0f / (1.0f + expf(-g)));
}


// ---------------------------------------------------------------------------
// FULL split: BOTH passes divide the row walk across `split` blocks per head.
//
// Pass 1 writes raw scores (no softmax) into `d_att`, so block boundaries cannot
// change the result.  A host-visible global max/sum is then computed by
// `q35_attn_softmax_kernel` (one block per head, so it sees every row), and the
// probabilities are written back in place.  Pass 2 (the PV walk above) consumes
// them.
//
// Split over the SCORE walk is safe for exactness: each score is still its own
// serial j-sum, and the softmax that follows is the CPU's order over the full
// row.  The only re-associated sum is the PV one, which is why the split path is
// gated on integer-exact token IDs.


// Softmax over the full row of `d_att[0..n)` for every head, in place.




// ---------------------------------------------------------------------------
// Device-slot form of the split path, so a CUDA-graph node never retargets.
//
// Same three phases as `q35_launch_attn_decode_split2`, but `slot` comes from
// DEVICE memory and `split` is a constant, so the argument blocks are frozen for
// the whole run.  The dynamic shared block is sized for the MAXIMUM window --
// the split kernels use no shared memory (their partials live in `d_part`), so
// this costs nothing.
__global__ void q35_attn_score_split_kernel_dyn(
    const float *__restrict__ d_q,
    const float *__restrict__ d_kcache,
    float *__restrict__ d_att,
    const int *__restrict__ d_slot,
    int heads, int gqa, int hd, int kv_dim, float scale, int split)
{
    const int h = blockIdx.x / split;
    const int s = blockIdx.x % split;
    const int tid = threadIdx.x;
    if (h >= heads) return;
    const int n = *d_slot + 1;
    const int kvh = h / gqa;
    const float *__restrict__ qs = d_q + (size_t)h * hd;
    const int warp = tid >> 5, lane = tid & 31;
    const int nwarps = blockDim.x >> 5;
    const int per = (n + split - 1) / split;
    const int t0 = s * per, t1 = min(n, t0 + per);
    float *__restrict__ mine = d_att + (size_t)h * (size_t)Q35_ATTN_MAX_ROWS;
    for (int t = t0 + warp; t < t1; t += nwarps) {
        const float *__restrict__ ks = d_kcache + (size_t)t * kv_dim + (size_t)kvh * hd;
        float acc = 0.0f;
        for (int j = lane; j < hd; j += 32) acc = __fadd_rn(acc, __fmul_rn(qs[j], ks[j]));
        for (int o = 16; o > 0; o >>= 1) acc = __fadd_rn(acc, __shfl_down_sync(0xffffffffu, acc, o));
        if (lane == 0) mine[t] = __fmul_rn(acc, scale);
    }
}

__global__ void q35_attn_softmax_kernel_dyn(float *__restrict__ d_att,
                                            const int *__restrict__ d_slot,
                                            int heads)
{
    const int h = blockIdx.x;
    if (h >= heads) return;
    const int n = *d_slot + 1;
    float *__restrict__ a = d_att + (size_t)h * (size_t)Q35_ATTN_MAX_ROWS;
    const int tid = threadIdx.x, nt = blockDim.x;
    __shared__ float s_max, s_sum;
    __shared__ float s_w[256];
    float m = -CUDART_INF_F;
    for (int t = tid; t < n; t += nt) m = fmaxf(m, a[t]);
    s_w[tid] = m;
    __syncthreads();
    for (int st = nt / 2; st > 0; st >>= 1) {
        if (tid < st) s_w[tid] = fmaxf(s_w[tid], s_w[tid + st]);
        __syncthreads();
    }
    if (tid == 0) s_max = s_w[0];
    __syncthreads();
    for (int t = tid; t < n; t += nt) a[t] = expf(a[t] - s_max);
    __syncthreads();
    // Serial sum, CPU order -- see the note in `q35_attn_softmax_kernel`.
    if (tid == 0) {
        float sm = 0.0f;
        for (int t = 0; t < n; ++t) sm += a[t];
        s_sum = sm;
    }
    __syncthreads();
    for (int t = tid; t < n; t += nt) a[t] = a[t] / s_sum;
}

__global__ void q35_attn_pv_split_kernel_dyn(
    const float *__restrict__ d_vcache,
    float *__restrict__ d_part,
    const float *__restrict__ d_prob,
    const int *__restrict__ d_slot,
    int heads, int gqa, int hd, int kv_dim, int split)
{
    const int h = blockIdx.x / split;
    const int s = blockIdx.x % split;
    const int tid = threadIdx.x;
    if (h >= heads) return;
    const int n = *d_slot + 1;
    const int kvh = h / gqa;
    const int warp = tid >> 5, lane = tid & 31;
    const int nwarps = blockDim.x >> 5;
    const int per = (n + split - 1) / split;
    const int t0 = s * per, t1 = min(n, t0 + per);
    const int cnt = t1 > t0 ? t1 - t0 : 0;
    const float *__restrict__ prob = d_prob + (size_t)h * (size_t)Q35_ATTN_MAX_ROWS + (size_t)t0;
    float *__restrict__ part = d_part + ((size_t)h * split + s) * (size_t)hd;
    for (int j = warp * 32 + lane; j < hd; j += nwarps * 32) {
        float a = 0.0f;
        const float *__restrict__ vcol = d_vcache + (size_t)kvh * hd + j;
        for (int i = 0; i < cnt; ++i)
            a = __fadd_rn(a, __fmul_rn(prob[i], vcol[(size_t)(t0 + i) * kv_dim]));
        part[j] = a;
    }
}

extern "C" void *q35_func_attn_score_split_dyn(void) { return (void *)&q35_attn_score_split_kernel_dyn; }
extern "C" void *q35_func_attn_pv_split_dyn(void) { return (void *)&q35_attn_pv_split_kernel_dyn; }
extern "C" void *q35_func_attn_softmax_dyn(void) { return (void *)&q35_attn_softmax_kernel_dyn; }
extern "C" void *q35_func_attn_reduce(void) { return (void *)&q35_attn_reduce_kernel; }

// Four nodes, all with constant arguments: score-split, softmax, PV-split,
// reduce+gate.  `split` is a constant so the grid is fixed too.
extern "C" int q35_launch_attn_decode_split_dyn(
    const float *d_q, const float *d_gate, const float *d_kcache,
    const float *d_vcache, float *d_att, float *d_gout, float *d_part,
    const int *d_slot, int heads, int gqa, int hd, int kv_dim, float scale,
    int split, void *stream)
{
    if (heads <= 0 || hd <= 0) return (int)cudaSuccess;
    if (d_slot == nullptr || d_part == nullptr) return (int)cudaErrorInvalidValue;
    if (split < 2) split = 2;
    if (split > Q35_ATTN_SPLIT_MAX) split = Q35_ATTN_SPLIT_MAX;
    q35_attn_score_split_kernel_dyn<<<heads * split, 256, 0, (cudaStream_t)stream>>>(
        d_q, d_kcache, d_att, d_slot, heads, gqa, hd, kv_dim, scale, split);
    int rc = (int)cudaGetLastError();
    if (rc != (int)cudaSuccess) return rc;
    q35_attn_softmax_kernel_dyn<<<heads, 256, 0, (cudaStream_t)stream>>>(d_att, d_slot, heads);
    rc = (int)cudaGetLastError();
    if (rc != (int)cudaSuccess) return rc;
    q35_attn_pv_split_kernel_dyn<<<heads * split, 256, 0, (cudaStream_t)stream>>>(
        d_vcache, d_part, d_att, d_slot, heads, gqa, hd, kv_dim, split);
    rc = (int)cudaGetLastError();
    if (rc != (int)cudaSuccess) return rc;
    const int total = heads * hd;
    q35_attn_reduce_kernel<<<(total + 255) / 256, 256, 0, (cudaStream_t)stream>>>(
        d_part, d_gate, d_gout, heads, hd, split);
    return (int)cudaGetLastError();
}

__global__ void q35_attn_score_split_kernel(
    const float *__restrict__ d_q,
    const float *__restrict__ d_kcache,
    float *__restrict__ d_att,
    int slot, int heads, int gqa, int hd, int kv_dim, float scale, int split)
{
    const int h = blockIdx.x / split;
    const int s = blockIdx.x % split;
    const int tid = threadIdx.x;
    if (h >= heads) return;
    const int n = slot + 1;
    const int kvh = h / gqa;
    const float *__restrict__ qs = d_q + (size_t)h * hd;
    const int warp = tid >> 5, lane = tid & 31;
    const int nwarps = blockDim.x >> 5;
    const int per = (n + split - 1) / split;
    const int t0 = s * per, t1 = min(n, t0 + per);
    float *__restrict__ mine = d_att + (size_t)h * (size_t)Q35_ATTN_MAX_ROWS;
    for (int t = t0 + warp; t < t1; t += nwarps) {
        const float *__restrict__ ks = d_kcache + (size_t)t * kv_dim + (size_t)kvh * hd;
        float acc = 0.0f;
        for (int j = lane; j < hd; j += 32) acc = __fadd_rn(acc, __fmul_rn(qs[j], ks[j]));
        for (int o = 16; o > 0; o >>= 1) acc = __fadd_rn(acc, __shfl_down_sync(0xffffffffu, acc, o));
        if (lane == 0) mine[t] = __fmul_rn(acc, scale);
    }
}
__global__ void q35_attn_softmax_kernel(float *__restrict__ d_att, int slot, int heads)
{
    const int h = blockIdx.x;
    if (h >= heads) return;
    const int n = slot + 1;
    float *__restrict__ a = d_att + (size_t)h * (size_t)Q35_ATTN_MAX_ROWS;
    const int tid = threadIdx.x, nt = blockDim.x;
    __shared__ float s_max, s_sum;
    __shared__ float s_w[256];
    // Parallel max with the CPU's strict-> tie-free order: max is associative and
    // commutative on the max operation's operand choice here, so a tree is exact.
    float m = -CUDART_INF_F;
    for (int t = tid; t < n; t += nt) m = fmaxf(m, a[t]);
    s_w[tid] = m;
    __syncthreads();
    for (int st = nt / 2; st > 0; st >>= 1) {
        if (tid < st) s_w[tid] = fmaxf(s_w[tid], s_w[tid + st]);
        __syncthreads();
    }
    if (tid == 0) s_max = s_w[0];
    __syncthreads();
    for (int t = tid; t < n; t += nt) a[t] = expf(a[t] - s_max);
    __syncthreads();
    // The SUM must keep the CPU's serial index order -- a tree would re-associate
    // it and move the last bits of every probability.  Thread 0 walks the row;
    // `n` is a few hundred, so this is a few microseconds, not a bottleneck.
    if (tid == 0) {
        float sm = 0.0f;
        for (int t = 0; t < n; ++t) sm += a[t];
        s_sum = sm;
    }
    __syncthreads();
    for (int t = tid; t < n; t += nt) a[t] = a[t] / s_sum;
}
extern "C" void *q35_func_attn_score_split(void) { return (void *)&q35_attn_score_split_kernel; }

extern "C" int q35_launch_attn_decode_split2(const float *d_q, const float *d_gate,
                                             const float *d_kcache,
                                             const float *d_vcache, float *d_att,
                                             float *d_gout, float *d_part, int slot,
                                             int heads, int gqa, int hd, int kv_dim,
                                             float scale, int split, void *stream)
{
    if (heads <= 0 || hd <= 0) return (int)cudaSuccess;
    const int n = slot + 1;
    if (n > Q35_ATTN_MAX_ROWS) return cudaErrorInvalidValue;
    if (split < 1) split = 1;
    if (split > Q35_ATTN_SPLIT_MAX) split = Q35_ATTN_SPLIT_MAX;
    if (split > n) split = n;
    if (split == 1 || d_part == nullptr)
        return q35_launch_attn_decode_coal(d_q, d_gate, d_kcache, d_vcache, d_att,
                                           d_gout, slot, heads, gqa, hd, kv_dim,
                                           scale, stream);
    q35_attn_score_split_kernel<<<heads * split, 256, 0, (cudaStream_t)stream>>>(
        d_q, d_kcache, d_att, slot, heads, gqa, hd, kv_dim, scale, split);
    int rc = (int)cudaGetLastError();
    if (rc != (int)cudaSuccess) return rc;
    q35_attn_softmax_kernel<<<heads, 256, 0, (cudaStream_t)stream>>>(d_att, slot, heads);
    rc = (int)cudaGetLastError();
    if (rc != (int)cudaSuccess) return rc;
    q35_attn_decode_kernel_split<<<heads * split, 256, 0, (cudaStream_t)stream>>>(
        d_q, d_gate, d_kcache, d_vcache, d_att, d_gout, d_part, d_att, slot, heads,
        gqa, hd, kv_dim, scale, split);
    rc = (int)cudaGetLastError();
    if (rc != (int)cudaSuccess) return rc;
    const int total = heads * hd;
    q35_attn_reduce_kernel<<<(total + 255) / 256, 256, 0, (cudaStream_t)stream>>>(
        d_part, d_gate, d_gout, heads, hd, split);
    return (int)cudaGetLastError();
}
