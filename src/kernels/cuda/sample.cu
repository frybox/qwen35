// sample.cu -- device-side sampling: temperature -> top-k -> top-p -> draw.
//
// ---------------------------------------------------------------------------
// WHY THIS EXISTS
// ---------------------------------------------------------------------------
// llama.cpp puts its sampler in the SAME `ggml_cgraph` as the model, so a decode
// step is one submission from logits to token id and the sampler's nodes are
// captured by the CUDA graph like every other kernel
// (`ggml_backend_cuda_graph_compute`).  qwen35 used to stop the graph at the
// logits and sample on the HOST: a blocking D2H of the whole vocabulary row
// (~1 MB on the 27B) per generated token whenever `temperature > 0`, with the
// sampler outside the graph at the one point where the device is already idle.
//
// ---------------------------------------------------------------------------
// THE GRAPH CONSTRAINT THAT SHAPES THE DESIGN
// ---------------------------------------------------------------------------
// A captured kernel node keeps the PARAMETERS it was recorded with.  `pos` and
// `slot` escape that via device memory (`graphReplayInner` in backend_cuda.zig),
// but retargeting every sampler node per token would re-introduce exactly the
// per-layer `SetParams` cost the decode graph exists to avoid.  So EVERY value
// that changes per token travels through ONE device control block, rewritten by
// a single 64-byte H2D before the replay; the kernels read it through a pointer
// that never changes, so no node is ever retargeted.
//
// The same constraint forbids host synchronisation in this file: a
// `cudaStreamSynchronize` or blocking `cudaMemcpy` inside a capture aborts the
// graph.  The selection below is therefore multi-pass ON THE DEVICE -- never
// "loop until the host learns the prefix".
//
// Control block (`q35_sample_ctl`): 16 x int32, so any field can carry an f32
// bit pattern or half of a u64.  Mirrored EXACTLY by `SampleCtl` in
// src/backend_cuda.zig -- change both or neither.
//
//   [0]  vocab         int       logits length
//   [1]  temperature   f32 bits  0 => greedy argmax
//   [2]  top_k         int       0 => off
//   [3]  top_p         f32 bits  <= 0 => off
//   [4]  rng_lo        u32       PRNG state low half   (in)
//   [5]  rng_hi        u32       PRNG state high half  (in)
//   [6]  token_out     int       RESULT -- the only field the host reads back
//   [7]  rng_out_lo    u32       PRNG state after the draw (out)
//   [8]  rng_out_hi    u32       ditto                   (out)
//   [9]  thresh        f32 bits  effective keep-threshold (out)
//   [10] flags         int       bit0 = k active, bit1 = p active
//   [11] sel_prefix    u32       radix-select accumulator (internal)
//   [12] sel_mask      u32       radix-select accumulator (internal)
//   [13] sel_rank      int       radix-select accumulator (internal)
//   [14..15] reserved
//
// ---------------------------------------------------------------------------
// NUMERIC CONTRACT
// ---------------------------------------------------------------------------
// The reference is `src/sampler.zig` (`Sampler.sample`).  This file mirrors it
// op for op so CPU and CUDA pick the same token from the same logits and seed:
//
//   * argmax ties -> LOWEST index (sampler.zig:73).
//   * softmax     -> shift by the running max, `exp((v - max) / T)`, mass
//                    accumulated in f64 in INDEX order.
//   * top-k       -> the k-th largest value acts as a THRESHOLD and ties at it
//                    are all kept (sampler.zig:93).  Value-based, not "keep k
//                    elements": LLM logits do tie, and the CPU rule is the
//                    contract.
//   * top-p       -> a second threshold on the same ordering: the smallest
//                    prefix of the descending order whose mass reaches p of the
//                    total.  Implemented as a binary search over the ORDERED KEY
//                    space, so no permutation is ever materialised and the
//                    "keep everything at or above this value" tie rule survives.
//   * the draw    -> `Sampler.nextU64` is xorshift64*; `nextF32` takes the TOP
//                    24 bits over 2^24; the walk is `coin < acc / sum` over
//                    surviving indices in INCREASING order, with sampler.zig's
//                    two fallbacks (last survivor, then 0).
//
// `double` is IEEE-754 exact on sm_60+, so the f64 mass is bit-identical to the
// CPU's.  It runs at 1/32 rate on consumer parts, but the sampler touches
// `vocab` elements ONCE per token -- the lm_head matvec just produced the same
// `vocab` values -- so it is nowhere near the critical path and exactness is
// worth more than its FLOPs.
//
// The one thing NOT bit-identical is `expf` (device libm != glibc's, ~1 ULP).
// That perturbs the mass, never the ORDERING that top-k/top-p threshold on, so
// the token only changes when two candidates sit within ~1 ULP of the draw --
// the same divergence class AGENTS.md §4 already documents.
//
// ---------------------------------------------------------------------------
// PARALLELISM: the two order-independent reductions are parallel; everything
// that accumulates mass is a single thread, BY DESIGN.
// ---------------------------------------------------------------------------
// Only the running max and the radix histogram that locates the k-th largest are
// multi-block.  A tree reduction over the mass would change the accumulation
// order and break the CPU match.  At vocab <= 248320 that walk is ~0.25 M f64
// adds -- well under a millisecond, once per generated token.
#include "common.cuh"
#include <math_constants.h>

namespace {

// Control block field indices.  MUST match `SampleCtl` in backend_cuda.zig.
enum : int {
    CTL_VOCAB = 0,
    CTL_TEMPERATURE = 1,
    CTL_TOP_K = 2,
    CTL_TOP_P = 3,
    CTL_RNG_LO = 4,
    CTL_RNG_HI = 5,
    CTL_TOKEN_OUT = 6,
    CTL_RNG_OUT_LO = 7,
    CTL_RNG_OUT_HI = 8,
    CTL_THRESH = 9,
    CTL_FLAGS = 10,
    CTL_SEL_PREFIX = 11,
    CTL_SEL_MASK = 12,
    CTL_SEL_RANK = 13,
    CTL_INTS = 16,
};

constexpr int kThreads = 256;
// Histogram blocks.  FIXED (not derived from `n`) so the scratch size is a
// compile-time constant the Zig side can query, and so `k_radix_pick`'s merge
// loop is bounded: 64 blocks x 256 buckets = 16384 ints to fold, which one
// thread does in a few microseconds.
constexpr int kHistBlocks = 64;
constexpr int kRadixBits = 8;
constexpr int kRadixBuckets = 1 << kRadixBits;
constexpr int kRadixPasses = 32 / kRadixBits;

// Scratch layout, in INTS, so the Zig side can size it without reasoning about
// float/double alignment.  `cudaMalloc` returns 256-byte alignment and every
// offset below is even, so the `double` slots are 8-byte aligned.
constexpr int OFF_PV = 0;                        // float[kHistBlocks]
constexpr int OFF_PI = OFF_PV + kHistBlocks;     // int  [kHistBlocks]
constexpr int OFF_MAXV = OFF_PI + kHistBlocks;   // float[2] (maxv, pad)
constexpr int OFF_HIST = OFF_MAXV + 2;           // int[kHistBlocks * kRadixBuckets]
// The histogram is 16384 ints, i.e. a multiple of 2, so `OFF_HIST` being even
// keeps everything after it 8-byte aligned.  `OFF_MAXV` is padded to 2 ints for
// exactly that reason: with it at +1 the `double` below landed on a 4-byte
// boundary and the kernel died with "misaligned address" (CUDA requires natural
// alignment for 64-bit types).  Do not "save" the padding int.
constexpr int OFF_SUM = OFF_HIST + kHistBlocks * kRadixBuckets;  // double
constexpr int OFF_COIN = OFF_SUM + 2;            // float (a double's worth)
constexpr int SCRATCH_INTS = OFF_COIN + 2;
static_assert(OFF_SUM % 2 == 0, "double must be 8-byte aligned in the scratch block");
static_assert(OFF_HIST % 2 == 0, "everything after the histogram inherits this alignment");

__device__ __forceinline__ float ctlF(const int *__restrict__ ctl, int at) {
    return __int_as_float(ctl[at]);
}

// xorshift64*, identical to `Sampler.nextU64` (sampler.zig:64).  Unsigned 64-bit
// wrap-around is plain C++ defined behaviour, which is what Zig's `%` on u64
// compiles to as well.
__device__ __forceinline__ uint64_t xs64star(uint64_t *s) {
    uint64_t x = *s;
    x ^= x >> 12;
    x ^= x << 25;
    x ^= x >> 27;
    *s = x;
    return x * 0x2545F4914F6CDD1DULL;
}

// IEEE-754 ascending -> unsigned ascending, so "largest float" is "largest
// bucket" and a DESCENDING bucket walk is a descending VALUE walk.  This is what
// makes the k-th-largest select a value threshold instead of a rank.
__device__ __forceinline__ uint32_t floatToOrdered(float v) {
    const uint32_t bits = __float_as_uint(v);
    const uint32_t mask = (uint32_t)(-(int32_t)(bits >> 31)) | 0x80000000U;
    return bits ^ mask;
}

// The INVERSE of `floatToOrdered`.  Note it is NOT the same expression: the
// forward map sends a negative float to a key with bit31 CLEAR and a positive one
// to a key with bit31 SET, so the inverse must branch on the KEY's sign bit, not
// on the float's.  Using the forward expression here (the obvious-looking
// "symmetric" form) round-trips -5.0f to -0.875f -- it was the bug that made the
// top-p threshold meaningless while still producing plausible token ids.
__device__ __forceinline__ float orderedToFloat(uint32_t k) {
    const uint32_t mask = (k & 0x80000000U) ? 0x80000000U : 0xFFFFFFFFU;
    return __uint_as_float(k ^ mask);
}

// Flag bits of `ctl[CTL_FLAGS]`, set by `k_plan`.
enum : int {
    PLAN_GREEDY = 1,
    PLAN_TOP_K = 2,
    PLAN_TOP_P = 4,
};

// A value survives when no ACTIVE threshold excludes it.  "Active" is derived
// from the flag bits, never from a sentinel threshold: an inactive threshold has
// to admit EVERYTHING, including NaN logits, and the CPU admits those too
// (`sampler.zig` compares with `>=` against a threshold that is only consulted
// when the filter is on).
__device__ __forceinline__ bool keep(float v, float thresh, int plan) {
    // With no active filter `thresh` is -FLT_MAX (set by `k_plan`), so the
    // unconditional `v >= thresh` below reproduces sampler.zig's comparison in
    // ALL cases -- including NaN, which the CPU's `v >= thresh` also rejects.
    // THE CPU COMPARISON IS THE CONTRACT (`sampler.zig:110` `if (logits[i] <
    // thresh) continue;`), so this is deliberately the same expression with the
    // same operand order, not an `active == 0 -> true` shortcut.
    (void)plan;
    return !(v < thresh);
}

// ---------------------------------------------------------------------------
// Stage 1: running max -> scratch[OFF_MAXV].  Needed by the numerically stable
// softmax, and it is a plain order-independent reduction.
// ---------------------------------------------------------------------------
__global__ void k_max_partial(const float *__restrict__ x, int n,
                              float *__restrict__ pm, int nblocks) {
    __shared__ float s[kThreads];
    const int tid = threadIdx.x;
    const int stride = blockDim.x * nblocks;
    float best = -CUDART_INF_F;
    for (int i = blockIdx.x * blockDim.x + tid; i < n; i += stride) {
        if (x[i] > best) best = x[i];
    }
    s[tid] = best;
    __syncthreads();
    for (int t = kThreads >> 1; t > 0; t >>= 1) {
        if (tid < t and s[tid + t] > s[tid]) s[tid] = s[tid + t];
        __syncthreads();
    }
    if (tid == 0) pm[blockIdx.x] = s[0];
}

__global__ void k_max_final(const float *__restrict__ pm, int m,
                            float *__restrict__ out_max) {
    __shared__ float s[kThreads];
    const int tid = threadIdx.x;
    float best = -CUDART_INF_F;
    for (int i = tid; i < m; i += blockDim.x) {
        if (pm[i] > best) best = pm[i];
    }
    s[tid] = best;
    __syncthreads();
    for (int t = kThreads >> 1; t > 0; t >>= 1) {
        if (tid < t and s[tid + t] > s[tid]) s[tid] = s[tid + t];
        __syncthreads();
    }
    if (tid == 0) *out_max = s[0];
}

// ---------------------------------------------------------------------------
// Stage 2: k-th largest VALUE, by radix select over four 8-bit passes.
//
// WHY FOUR SEPARATE LAUNCHES, NOT ONE KERNEL: a pass must AGGREGATE across blocks
// between passes, and a device-wide barrier inside one kernel needs cooperative
// launch.  Four small launches are simpler, capture into the graph just as well,
// and cost four submissions per token -- negligible next to the hundreds of
// launches of the forward pass they follow.
//
// After `kRadixPasses` passes `ctl[CTL_THRESH]` holds the k-th largest VALUE;
// ties at that value are all kept, which is exactly `kthLargest`'s meaning.
// The pass state (prefix/mask/rank) rides in `ctl[11..13]`, i.e. device memory,
// so nothing returns to the host mid-chain.
// ---------------------------------------------------------------------------
__global__ void k_hist_clear(int *__restrict__ hist, int count) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < count) hist[i] = 0;
}

// Seed one token's radix-select state IN DEVICE MEMORY, and decide the whole
// token's plan in `CTL_FLAGS`.
//
// Two reasons this is a kernel and not host code:
//
//   1. `cudaMemcpyAsync` of the rank would be an H2D copy inside the capture
//      interval; CUDA forbids those unless the source is pinned, and even then
//      a 1-thread kernel is simpler and measurably cheaper at 8 bytes.
//   2. MORE IMPORTANTLY: `d_ctl` is DEVICE memory, so the host launcher cannot
//      read `temperature`/`top_k`/`top_p` at all.  Every branch on a
//      control-block field must therefore happen in a kernel -- which is
//      exactly what makes the node list below FLAT and identical every token,
//      the property a captured graph needs.
//
// Flags (in CTL_FLAGS, read by every later kernel):
//   bit0  greedy      -- temperature <= 0, so `k_draw` does a pure argmax and
//                        every distribution kernel early-outs
//   bit1  top-k active
//   bit2  top-p active
// `CTL_SEL_RANK <= 0` additionally turns the radix select into a no-op, so the
// unconditional four-pass loop below costs one shared-memory clear per pass when
// top-k is off -- microseconds, and it keeps the graph shape constant.
__global__ void k_plan(int *__restrict__ ctl, int n) {
    if (blockIdx.x != 0 or threadIdx.x != 0) return;
    const float temperature = ctlF(ctl, CTL_TEMPERATURE);
    const int top_k = ctl[CTL_TOP_K];
    const float top_p = ctlF(ctl, CTL_TOP_P);
    const int greedy = !(temperature > 0.0f);
    const int use_k = (!greedy && top_k > 0 && top_k < n) ? 1 : 0;
    const int use_p = (!greedy && top_p > 0.0f && top_p < 1.0f) ? 1 : 0;
    ctl[CTL_FLAGS] = (greedy ? 1 : 0) | (use_k ? 2 : 0) | (use_p ? 4 : 0);
    ctl[CTL_THRESH] = __float_as_int(-CUDART_INF_F);
    ctl[CTL_SEL_PREFIX] = 0;
    ctl[CTL_SEL_MASK] = 0;
    ctl[CTL_SEL_RANK] = use_k ? top_k : 0;
}

__global__ void k_radix_hist(const float *__restrict__ x, int n,
                             const int *__restrict__ ctl, int shift,
                             int *__restrict__ hist) {
    const int tid = threadIdx.x;
    __shared__ int s[kRadixBuckets];
    for (int b = tid; b < kRadixBuckets; b += blockDim.x) s[b] = 0;
    __syncthreads();
    // `CTL_SEL_RANK <= 0` means top-k is off (or the token is greedy): emit an
    // all-zero histogram and let `k_radix_pick` see the no-op.
    const uint32_t prefix = (uint32_t)ctl[CTL_SEL_PREFIX];
    const uint32_t mask = (uint32_t)ctl[CTL_SEL_MASK];
    const int active = ctl[CTL_SEL_RANK] > 0;
    if (active) {
        const int stride = blockDim.x * gridDim.x;
        for (int i = blockIdx.x * blockDim.x + tid; i < n; i += stride) {
            const uint32_t key = floatToOrdered(x[i]);
            if ((key & mask) != prefix) continue;
            atomicAdd(&s[(key >> shift) & (kRadixBuckets - 1)], 1);
        }
        __syncthreads();
    }
    int *dst = hist + blockIdx.x * kRadixBuckets;
    for (int b = tid; b < kRadixBuckets; b += blockDim.x) dst[b] = s[b];
}

__global__ void k_radix_pick(int *__restrict__ ctl, const int *__restrict__ hist,
                             int nblocks, int shift, int last_pass) {
    __shared__ int s[kRadixBuckets];
    const int tid = threadIdx.x;
    for (int b = tid; b < kRadixBuckets; b += blockDim.x) {
        int sum = 0;
        for (int blk = 0; blk < nblocks; ++blk) sum += hist[blk * kRadixBuckets + b];
        s[b] = sum;
    }
    __syncthreads();
    if (tid != 0) return;
    const int rank = ctl[CTL_SEL_RANK]; // 1-indexed rank among survivors
    if (rank <= 0) return; // top-k off: leave the select state untouched
    const uint32_t prefix = (uint32_t)ctl[CTL_SEL_PREFIX];
    int acc = 0, chosen = 0, chosen_acc = 0;
    for (int d = kRadixBuckets - 1; d >= 0; --d) {
        const int c = s[d];
        if (c == 0) continue;
        if (acc + c >= rank) { chosen = d; chosen_acc = acc; break; }
        acc += c;
    }
    const uint32_t next_prefix = prefix | ((uint32_t)chosen << shift);
    ctl[CTL_SEL_PREFIX] = (int)next_prefix;
    ctl[CTL_SEL_MASK] = (int)((uint32_t)ctl[CTL_SEL_MASK] |
                              ((uint32_t)(kRadixBuckets - 1) << shift));
    ctl[CTL_SEL_RANK] = rank - chosen_acc;
    // The last pass leaves a complete 32-bit key: publish it as the f32 cutoff.
    if (last_pass) ctl[CTL_THRESH] = __float_as_int(orderedToFloat(next_prefix));
}

// Fold top-k and top-p into ONE effective cutoff plus the active-flag bits.
//
// `flags` reports what was actually applied, so `keep` never filters on a
// threshold that was not computed.  top_p's own cutoff is produced by
// `k_top_p`, which runs between the radix select and this kernel.
__global__ void k_top_p(const float *__restrict__ x, int n,
                        const float *__restrict__ d_max, int *__restrict__ ctl) {
    if (blockIdx.x != 0 or threadIdx.x != 0) return;
    const int plan = ctl[CTL_FLAGS];
    const int greedy = plan & 1;
    const int use_k = (plan >> 1) & 1;
    const int use_p = (plan >> 2) & 1;
    // Greedy never reaches the distribution kernels; nothing to fold.
    if (greedy || !use_p) return;

    const float temperature = ctlF(ctl, CTL_TEMPERATURE);
    const float top_p = ctlF(ctl, CTL_TOP_P);
    const float maxv = d_max[0];
    // The top-k cutoff, if any, is already in CTL_THRESH.  When top-k is OFF
    // there is no cutoff at all, so the only filter inside top-p's search is
    // "is the candidate at or above the probe" -- a `0.0f` sentinel here would
    // silently drop every negative logit, which on real logits is most of them.
    const int flags_in = use_k ? PLAN_TOP_K : 0;
    const float thresh = use_k ? ctlF(ctl, CTL_THRESH) : -CUDART_INF_F;
    double total = 0.0;
    for (int i = 0; i < n; ++i) {
        const float v = x[i];
        if (!keep(v, thresh, flags_in)) continue;
        total += (double)expf((v - maxv) / temperature);
    }
    const double target = total * (double)top_p;
    // Binary search the ORDERED KEY space for the largest t whose mass >= target.
    // Monotone decreasing in t, so the search is well defined.
    uint32_t lo = 0, hi = 0xFFFFFFFFu;
    while (lo < hi) {
        const uint32_t mid = lo + (uint32_t)(((uint64_t)hi - lo + 1) >> 1);
        const float t = orderedToFloat(mid);
        double m = 0.0;
        for (int i = 0; i < n; ++i) {
            const float v = x[i];
            if (!keep(v, thresh, flags_in)) continue;
            if (v < t) continue;
            m += (double)expf((v - maxv) / temperature);
        }
        if (m >= target) lo = mid; else hi = mid - 1;
    }
    const float p_thresh = orderedToFloat(lo);
    // Compose with top-k: the effective cutoff can only be RAISED, so keep the
    // stricter of the two.  The flag bits say which thresholds `keep` applies;
    // top-k stays applied through top-p, so bit1 is preserved.
    if (!(use_k && p_thresh < thresh)) {
        ctl[CTL_THRESH] = __float_as_int(p_thresh);
    }
}

// Total surviving mass, f64, index order.
__global__ void k_mass_sum(const float *__restrict__ x, int n,
                           const float *__restrict__ d_max, const int *__restrict__ ctl,
                           double *__restrict__ out_sum) {
    if (blockIdx.x != 0 or threadIdx.x != 0) return;
    const float maxv = d_max[0];
    const float temperature = ctlF(ctl, CTL_TEMPERATURE);
    const float thresh = ctlF(ctl, CTL_THRESH);
    const int plan = ctl[CTL_FLAGS];
    // Greedy takes no distribution; a zero sum keeps `k_draw`'s fallbacks honest
    // if it ever ran.
    if (plan & PLAN_GREEDY) { out_sum[0] = 0.0; return; }
    double sum = 0.0;
    for (int i = 0; i < n; ++i) {
        const float v = x[i];
        if (v < thresh) continue;
        sum += (double)expf((v - maxv) / temperature);
    }
    out_sum[0] = sum;
}

// One xorshift64* word -> the uniform float `Sampler.nextF32` would produce, and
// the advanced state written back into the control block so the next token
// continues the same stream.
__global__ void k_coin(int *__restrict__ ctl, float *__restrict__ out_coin) {
    if (blockIdx.x != 0 or threadIdx.x != 0) return;
    // Greedy consumes no randomness: advancing the stream there would make the
    // token sequence depend on the temperature schedule, which the CPU sampler
    // does not do either.
    if (ctl[CTL_FLAGS] & PLAN_GREEDY) return;
    uint64_t s = ((uint64_t)(uint32_t)ctl[CTL_RNG_HI] << 32) |
                 (uint64_t)(uint32_t)ctl[CTL_RNG_LO];
    // Zero is a fixed point of xorshift; `Sampler.init` never produces one, but a
    // caller that hands us 0 must not get 0 forever.
    if (s == 0) s = 0x9E3779B97F4A7C15ULL;
    const uint64_t word = xs64star(&s);
    ctl[CTL_RNG_OUT_LO] = (int)(uint32_t)(s & 0xFFFFFFFFu);
    ctl[CTL_RNG_OUT_HI] = (int)(uint32_t)(s >> 32);
    out_coin[0] = (float)(word >> 40) / 16777216.0f;
}

// The draw: `coin < acc / sum` walked over surviving indices in INCREASING order,
// with sampler.zig:99-107's two fallbacks (last survivor, then 0).
__global__ void k_draw(const float *__restrict__ x, int n,
                       const float *__restrict__ d_max, const int *__restrict__ ctl,
                       const double *__restrict__ d_sum, const float *__restrict__ d_coin,
                       int *__restrict__ out_tok) {
    if (blockIdx.x != 0 or threadIdx.x != 0) return;
    const float maxv = d_max[0];
    const float temperature = ctlF(ctl, CTL_TEMPERATURE);
    const float thresh = ctlF(ctl, CTL_THRESH);
    const int plan = ctl[CTL_FLAGS];

    // Greedy: the argmax `k_amax_final` already wrote into `ctl[CTL_TOKEN_OUT]`
    // from the same serial thread?  No -- `k_amax_*` is a separate reduction, so
    // this kernel leaves the greedy result alone (see the launcher: the argmax
    // pair runs for every token, and this kernel only overwrites the token when a
    // draw actually happened).
    if (plan & PLAN_GREEDY) return;

    const float coin = d_coin[0];
    const double sum = d_sum[0];
    double acc = 0.0;
    for (int i = 0; i < n; ++i) {
        const float v = x[i];
        if (v < thresh) continue;
        acc += (double)expf((v - maxv) / temperature);
        if ((double)coin < acc / sum) { out_tok[0] = i; return; }
    }
    for (int i = n - 1; i >= 0; --i) {
        if (x[i] < thresh) continue;
        out_tok[0] = i;
        return;
    }
    out_tok[0] = 0;
}

// Greedy: temperature <= 0 is a pure argmax with the CPU's lowest-index tie rule
// (sampler.zig:73).  Two stages so a 248k vocabulary streams from many SMs rather
// than one -- the same measurement that made elementwise.cu's argmax two-stage.
__global__ void k_amax_partial(const float *__restrict__ x, int n,
                               float *__restrict__ pv, int *__restrict__ pi, int nblocks) {
    __shared__ float sv[kThreads];
    __shared__ int si[kThreads];
    const int tid = threadIdx.x;
    const int stride = blockDim.x * nblocks;
    float best = -CUDART_INF_F;
    int best_i = 0;
    for (int i = blockIdx.x * blockDim.x + tid; i < n; i += stride) {
        const float v = x[i];
        if (v > best) { best = v; best_i = i; }
    }
    sv[tid] = best;
    si[tid] = best_i;
    __syncthreads();
    for (int t = kThreads >> 1; t > 0; t >>= 1) {
        if (tid < t) {
            const float v = sv[tid + t];
            const int i = si[tid + t];
            if (v > sv[tid] or (v == sv[tid] and i < si[tid])) { sv[tid] = v; si[tid] = i; }
        }
        __syncthreads();
    }
    if (tid == 0) { pv[blockIdx.x] = sv[0]; pi[blockIdx.x] = si[0]; }
}

__global__ void k_amax_final(const float *__restrict__ pv, const int *__restrict__ pi,
                             int m, int *__restrict__ out_tok) {
    __shared__ float sv[kThreads];
    __shared__ int si[kThreads];
    const int tid = threadIdx.x;
    float best = -CUDART_INF_F;
    int best_i = 0;
    for (int i = tid; i < m; i += blockDim.x) {
        const float v = pv[i];
        const int idx = pi[i];
        if (v > best or (v == best and idx < best_i)) { best = v; best_i = idx; }
    }
    sv[tid] = best;
    si[tid] = best_i;
    __syncthreads();
    for (int t = kThreads >> 1; t > 0; t >>= 1) {
        if (tid < t) {
            const float v = sv[tid + t];
            const int i = si[tid + t];
            if (v > sv[tid] or (v == sv[tid] and i < si[tid])) { sv[tid] = v; si[tid] = i; }
        }
        __syncthreads();
    }
    if (tid == 0) out_tok[0] = si[0];
}

} // namespace

// ===========================================================================
// Launchers
// ===========================================================================
//
// Every launcher takes the SAME device arguments (`d_logits`, `d_ctl`,
// `d_scratch`) plus the stream, so a captured graph node's parameter list never
// changes from token to token -- the whole point of the control block.
// Nothing here allocates: the two regions come from `q35_launch_sample_init`,
// which the host runs before any capture opens (`cudaMalloc` inside a capture is
// illegal).
//
// THE NODE LIST IS FLAT AND UNCONDITIONAL.  That is not laziness: `d_ctl` is
// device memory, so this host code cannot read `temperature`/`top_k`/`top_p` to
// choose a path.  `k_plan` publishes the decision into `ctl[CTL_FLAGS]`, and each
// later kernel early-outs on its own flag.  Cost of replaying a disabled stage:
// one shared-memory clear or one early return, i.e. microseconds, against the
// hundreds of forward-pass launches that precede it.
extern "C" {

int q35_sample_scratch_ints(void) { return SCRATCH_INTS; }
int q35_sample_ctl_ints(void) { return CTL_INTS; }

// Allocate both regions once.  Idempotent: a NULL input means "not allocated
// yet".  Returns a cudaError_t value.
int q35_launch_sample_init(void **d_ctl, void **d_scratch) {
    if (d_ctl == nullptr || d_scratch == nullptr) return (int)cudaErrorInvalidValue;
    if (*d_ctl == nullptr) {
        if (cudaMalloc(d_ctl, (size_t)CTL_INTS * sizeof(int)) != cudaSuccess) {
            *d_ctl = nullptr;
            return (int)cudaGetLastError();
        }
        cudaMemset(*d_ctl, 0, (size_t)CTL_INTS * sizeof(int));
    }
    if (*d_scratch == nullptr) {
        if (cudaMalloc(d_scratch, (size_t)SCRATCH_INTS * sizeof(int)) != cudaSuccess) {
            cudaFree(*d_ctl);
            *d_ctl = nullptr;
            return (int)cudaGetLastError();
        }
        cudaMemset(*d_scratch, 0, (size_t)SCRATCH_INTS * sizeof(int));
    }
    return (int)cudaSuccess;
}

void q35_launch_sample_deinit(void **d_ctl, void **d_scratch) {
    if (d_ctl && *d_ctl) { cudaFree(*d_ctl); *d_ctl = nullptr; }
    if (d_scratch && *d_scratch) { cudaFree(*d_scratch); *d_scratch = nullptr; }
}

// The whole sampler: logits in, `ctl[CTL_TOKEN_OUT]` out.
//
// Called either directly (prefill's single logits row, or a cold start) or
// recorded as nodes of a captured decode graph -- `stream` is the capture stream
// in the latter case, so every kernel below lands inside the graph.
//
// Order (each stage only consumes what the previous published into d_ctl /
// d_scratch, so the chain has no host dependency):
//   k_plan       decide greedy / top-k / top-p, seed the radix-select state
//   argmax       two-stage max, ALWAYS run: it is the greedy answer, and its
//                result is only overwritten by k_draw when a draw happens
//   max          running max for the stable softmax
//   radix x 4    k-th largest value -> CTL_THRESH
//   k_top_p      refine that cutoff to the top-p boundary
//   k_mass_sum   f64 total mass over survivors
//   k_coin       one xorshift64* word -> uniform f32, advance the state
//   k_draw       the cumulative walk
// `n` (the vocabulary length) is a LAUNCH ARGUMENT, not a control-block read:
// it decides the grid shape, so the host has to know it, and it is constant for
// the whole run (the lm_head row length never changes).  Everything that varies
// per token stays in `d_ctl`, which this function never dereferences.
int q35_launch_sample(const float *d_logits, int n, int *d_ctl, void *d_scratch,
                      void *stream_v) {
    if (d_logits == nullptr || d_ctl == nullptr || d_scratch == nullptr) {
        return (int)cudaErrorInvalidValue;
    }
    if (n <= 0) return (int)cudaErrorInvalidValue;
    cudaStream_t stream = (cudaStream_t)stream_v;

    int *si = (int *)d_scratch;
    float *pv = (float *)(si + OFF_PV);
    int *pi = si + OFF_PI;
    float *d_max = (float *)(si + OFF_MAXV);
    int *d_hist = si + OFF_HIST;
    double *d_sum = (double *)(si + OFF_SUM);
    float *d_coin = (float *)(si + OFF_COIN);
    int *d_tok = d_ctl + CTL_TOKEN_OUT;

    int nblocks = (n + kThreads * 4 - 1) / (kThreads * 4);
    if (nblocks > kHistBlocks) nblocks = kHistBlocks;
    if (nblocks < 1) nblocks = 1;

    // ---- plan + greedy argmax ---------------------------------------------
    // The argmax pair runs UNCONDITIONALLY and writes CTL_TOKEN_OUT first, so
    // the greedy answer is in place before anything else touches the token.  It
    // is not wasted work when sampling: it is the same `vocab`-element stream the
    // running max does, and the two share the L2-resident logits row.
    k_plan<<<1, 1, 0, stream>>>(d_ctl, n);
    k_amax_partial<<<nblocks, kThreads, 0, stream>>>(d_logits, n, pv, pi, nblocks);
    k_amax_final<<<1, kThreads, 0, stream>>>(pv, pi, nblocks, d_tok);

    // ---- running max (numerically stable softmax) -------------------------
    k_max_partial<<<nblocks, kThreads, 0, stream>>>(d_logits, n, pv, nblocks);
    k_max_final<<<1, kThreads, 0, stream>>>(pv, nblocks, d_max);

    // ---- k-th largest value: four device-side radix passes ----------------
    k_hist_clear<<<(kHistBlocks * kRadixBuckets + kThreads - 1) / kThreads, kThreads, 0,
                   stream>>>(d_hist, kHistBlocks * kRadixBuckets);
    for (int pass = 0; pass < kRadixPasses; ++pass) {
        const int shift = 32 - kRadixBits * (pass + 1);
        const int last = (pass == kRadixPasses - 1) ? 1 : 0;
        k_radix_hist<<<kHistBlocks, kThreads, 0, stream>>>(d_logits, n, d_ctl, shift, d_hist);
        k_radix_pick<<<1, kRadixBuckets, 0, stream>>>(d_ctl, d_hist, kHistBlocks, shift, last);
    }

    // ---- top-p refine, mass, coin, draw -----------------------------------
    k_top_p<<<1, 1, 0, stream>>>(d_logits, n, d_max, d_ctl);
    k_mass_sum<<<1, 1, 0, stream>>>(d_logits, n, d_max, d_ctl, d_sum);
    k_coin<<<1, 1, 0, stream>>>(d_ctl, d_coin);
    k_draw<<<1, 1, 0, stream>>>(d_logits, n, d_max, d_ctl, d_sum, d_coin, d_tok);
    return (int)cudaGetLastError();
}

} // extern "C"
