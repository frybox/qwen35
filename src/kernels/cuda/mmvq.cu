// mmvq.cu -- quantized matrix-vector products, ALL weight types, plus the
// activation quantizers they need (Q8_0 for the 32-element block family, Q8_K
// for the 256-element one) and the f32 path.
//
// ---------------------------------------------------------------------------
// DESIGN: one thread per output element, one type-dispatched dot function
// ---------------------------------------------------------------------------
// Anchors:
//   ggml/src/ggml-cuda/mmvq.cu          mul_mat_vec_q (grid/shape idea)
//   ggml/src/ggml-cuda/vecdotq.cuh      per-type vec_dot_* (numerics live in
//                                       this repo's vecdotq.cuh instead)
//   src/kernel.cu:239 (deleted)           k_matvec_q8 (dp4a + prefetch idea)
//
// DELIBERATE DIFFERENCE from the retired src/kernel.cu (deleted): that kernel
// parallelises a row across a warp and reduces with shuffles.  Here one thread
// owns one whole row
// and accumulates strictly left to right (block by block), which is exactly
// the CPU kernel's float addition tree.  A warp reduction reorders that tree,
// was measured to be ~1 ulp off, and compounds over 150+ matvecs per token into
// a different sampled token.  The acceptance criterion is an identical token
// stream, not a tolerance -- so the one-thread-per-row mapping stays, and the
// per-type functions in vecdotq.cuh are written to fold one block at a time.
//
// The "type table" is a compile-time switch (`q35_dot_row`) rather than an
// array of device function pointers: nvcc resolves the switch to a branch, and
// keeping it as code lets every type's block loop be unrolled.  The type list,
// block sizes and activation choice come from ONE table (q35_quant_info below)
// so a new type cannot be half-wired.
//
// ---------------------------------------------------------------------------
// ACTIVATION SIDE
// ---------------------------------------------------------------------------
// 32-element family (q8_0/q4_0/q4_1/q5_0/q5_1/iq4_nl) uses Q8_0 activations:
//   34 B per 32 elements, one quantized copy per (token, column range).
// 256-element family (q2_k..q6_k, iq1_s..iq4_xs) uses Q8_K activations:
//   292 B per 256 elements, d(f32) @0, qs[256] @4, bsums[16] @260.
// Both are produced ON THE GPU (q35_launch_quantize_q8_0 / _q8_K); nothing
// falls back to the CPU.
#include "common.cuh"
#include "vecdotq.cuh"
#include "api.h"

#include <stdlib.h>
#include <string.h>

// The per-type geometry table and the block/row dot helpers moved to
// dotq.cuh so that mmq.cu (the prefill tile kernel) can use the exact same
// definitions instead of a second copy (AGENTS.md: one source for one fact).
#include "dotq.cuh"
// ---------------------------------------------------------------------------
// matvec kernel: out[t*rows + r] = W[r] . x[t]
// ---------------------------------------------------------------------------
__global__ void q35_k_mul_mat_vec_q(const unsigned char *__restrict__ w,
                                    const unsigned char *__restrict__ xq,
                                    float *__restrict__ out,
                                    int rows, int cols, int tokens, int ttype) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int total = rows * tokens;
    if (idx >= total) return;
    const Q35QuantInfo info = q35_quant_info(ttype);
    const int nblk = cols / info.block_elems;
    const int t = idx / rows;
    const int r = idx - t * rows;

    const unsigned char *wrow = w + (size_t)r * (size_t)nblk * (size_t)info.block_bytes;
    const unsigned char *arow = xq + (size_t)t * (size_t)nblk * (size_t)info.act_bytes;
    out[(size_t)t * rows + r] =
        q35_dot_row(ttype, wrow, arow, info.block_bytes, info.act_bytes, nblk);
}

// ---------------------------------------------------------------------------
// Cooperative-row variant: one warp per output row.
//
// Why this exists.  The kernel above maps thread -> row, so the 32 threads of a
// warp walk 32 *different* rows: at any step their addresses are
// `nblk * block_bytes` apart (2200 B for IQ3_XXS at cols=5120), i.e. every load
// touches its own 128 B sector and the warp uses 1/128th of each.  Measured
// effect: the batchK path ran at ~80 GB/s of the 4090D's ~1008 GB/s.
// Flipping the mapping so a warp covers consecutive *blocks of one row* makes
// the warp read a contiguous 32 x block_bytes stretch -> fully coalesced.
//
// Reduced precision concern: the reduction order is NOT changed.  Each lane
// computes one block's term with `q35_dot_1blk`, the partial sums go to shared
// memory, and lane 0 folds them strictly b = 0 .. nblk-1 -- the same left-to-
// right order `q35_dot_row` uses, so the result is bit-identical (the file
// header forbids reordering; this respects it).
//
// Block shape is a template parameter (WARPS = warps per block, one row per
// warp).  `part` must be exactly WARPS deep: it is indexed by the warp number,
// which runs 0..WARPS-1, so the launcher's block size (WARPS*Q35_WARP threads)
// and this dimension are ONE decision -- q35_coop_warps() below picks it and
// the host-side table instantiates every shape in the 1..8 domain.  WARPS=4
// (128 threads) is the measured-best shape on sm_89; see q35_coop_warps().
// ---------------------------------------------------------------------------
// A warp owns one output row and its lanes sweep the row's `nblk` blocks in
// strides of Q35_WARP: at every step the 32 lanes read 32 *consecutive*
// blocks, which is what makes the load coalesced (see the file header).
//
// The partials live in DYNAMIC shared (extern __shared__): the launcher sizes
// it WARPS * nblk floats and passes the byte count at launch, so nblk is
// unbounded.  Slot (warp, b) is part_dyn[warp*nblk + b] (a row per warp);
// lane `l` writes block `l, l+32, l+64, ...` into that warp's row, so each
// block's term lands in its own slot regardless of how many blocks a lane
// owned.  Lane 0 then folds strictly b = 0..nblk-1 with __fadd_rn -- the
// same left-to-right order q35_dot_row produces, so the value is
// bit-identical to the thread-per-row mapping.  Shared-memory writes are
// conflict-free: within a warp the lane `l` of step `k` writes slot
// `l + 32k`, bank `(l + 32k) % 32 == l`, one per bank.
template <int WARPS>
__global__ void q35_k_mul_mat_vec_q_coop_t(const unsigned char *__restrict__ w,
                                           const unsigned char *__restrict__ xq,
                                           float *__restrict__ out,
                                           int rows, int cols, int tokens, int ttype,
                                           int memonly, int rpb) {
    const Q35QuantInfo info = q35_quant_info(ttype);
    const int nblk = cols / info.block_elems;
    const int lane = threadIdx.x & (Q35_WARP - 1);
    const int warp = threadIdx.x / Q35_WARP;

    // Dynamic shared: the launcher sizes it WARPS * nblk floats (one slot per
    // (warp, block) -- the exact shape the static part[WARPS][Q35_COOP_MAXNBLK]
    // used to have, just with the 256 bound gone) and passes the byte count as
    // the third <<<>>> argument, so nblk is unbounded.  Slot (warp, b) lives at
    // part_dyn[warp*nblk + b]; lane `l` writes block l, l+32, ... and lane 0
    // folds b = 0..nblk-1 strictly left to right, so the value stays
    // bit-identical to the static shape (see the header on reduction order).
    extern __shared__ float part_dyn[];
    // Each row the warp owns gets its OWN nblk-slot region, so the lane-0 fold
    // below is the same strict b = 0..nblk-1 order for every row -- multi-row
    // per warp must not change the reduction order (see the header contract).
    float *part_w = part_dyn + (size_t)warp * (size_t)rpb * nblk;

    // A warp owns `rpb` CONSECUTIVE rows: row (warp*rpb + k).  The base is
    // WARPS*rpb per block, so rows stay covered in order with no gaps.
    const int row_base = blockIdx.x * WARPS * rpb + warp * rpb;
    if (row_base >= rows) return;
    const size_t row_stride = (size_t)nblk * (size_t)info.block_bytes;

    // Loop over tokens so a prefill chunk reuses the same tile mapping.
    for (int t = 0; t < tokens; ++t) {
        const unsigned char *arow = xq + (size_t)t * (size_t)nblk * (size_t)info.act_bytes;

        // Lane `l` owns blocks l, l+32, l+64, ...; each step the warp reads
        // 32 consecutive blocks, coalesced.  Each term is stored at its own
        // block index so the lane-0 fold below is order-independent of strides.
        for (int b = lane; b < nblk; b += Q35_WARP) {
            // All rpb rows in one pass: the rpb q35_dot_1blk chains are
            // independent (different rows, different weights), so issuing them
            // back to back gives the scheduler rpb-way ILP where the old
            // single-row mapping had one chain per warp.
            for (int k = 0; k < rpb; ++k) {
                const int row = row_base + k;
                if (row >= rows) break;
                const unsigned char *wrow = w + (size_t)row * row_stride;
                const unsigned char *pb = wrow + (size_t)b * (size_t)info.block_bytes;
                float *slot = part_w + (size_t)k * nblk + b;
                if (memonly) {
                    // Load-only: touch every byte of the block, no IQ math.  The
                    // address pattern, the lane->block mapping and the shared
                    // write are identical to the real path, so the difference in
                    // time is the IQ dot product plus its dependent loads.
                    int s = 0;
                    for (int i = 0; i < info.block_bytes; ++i) s += pb[i];
                    *slot = (float)s;
                } else {
                    *slot = q35_dot_1blk(ttype, pb,
                                         arow + (size_t)b * (size_t)info.act_bytes, 0.0f);
                }
            }
        }
        __syncwarp();

        // Fold `rpb` rows in PARALLEL: lane `k` folds row `k`.  Each row keeps
        // the strict b = 0..nblk-1 left-to-right fold that q35_dot_row uses, so
        // every value is bit-identical (the header's reduction-order contract).
        // Folding everything on lane 0 would serialize rpb*nblk FADDs and make
        // the reduction the bottleneck as rpb grows.
        if (lane < rpb && row_base + lane < rows) {
            const float *slot_row = part_w + (size_t)lane * nblk;
            float acc = 0.0f;
            for (int b = 0; b < nblk; ++b) acc = __fadd_rn(acc, slot_row[b]);
            out[(size_t)t * rows + row_base + lane] = acc;
        }
    }
}

// ---------------------------------------------------------------------------
// K-SPLIT decode kernel: `SPLIT` lanes cooperate on ONE 256-element weight block
// ---------------------------------------------------------------------------
// Why this exists.  `q35_k_mul_mat_vec_q_coop_t` above gives each lane an entire
// 256-element block, so a lane's dot is a strict serial chain of eight
// 32-element sub-blocks (8 x (2 grid lookups + 2 dp4a) into ONE accumulator).
// Measured on 27B IQ3_XXS decode (5120x10240 qkv), the coop kernel moves the
// weights at 251 GB/s while the SAME loads with the arithmetic replaced by a
// byte checksum reach 885 GB/s: decode is not bandwidth-starved, it is bound by
// that per-lane chain.  A microbenchmark over the identical bytes measured
// 251 GB/s at one lane per block, 285 at 2, 453 at 4 and 694 at 8 -- the chain,
// not the parallelism, is the limit.  llama.cpp reaches the same conclusion by
// different means (`vdr` splits K across lanes).
//
// Shape here.  Lane `l` of a warp works on block `b` and sub-block slice
// `l % SPLIT`, and the SPLIT lanes of one block are lane-consecutive so their
// weight reads for that block are one coalesced region.  Blocks stride by
// `Q35_WARP / SPLIT`, so the warp still covers 32 blocks per step (unchanged
// coalescing) and every block is covered exactly once.
//
// REDUCTION.  Each lane holds a partial sum.  They are folded with a
// `SPLIT`-wide butterfly (`__shfl_down_sync` by 1, 2, 4) inside each subgroup,
// then the SPLIT-wide results are folded across the warp exactly like the coop
// kernel does -- except that the cross-warp part still goes through
// `part_dyn[block]` + a lane-0 strict `b = 0..nblk-1` fold, so the value stays
// in the same reduction class as the coop kernel (checked by QWEN35_MVCHECK).
//
// Deliberately NO `rpb`: multi-row-per-warp was measured worse on this model
// (17.8 -> 12.6 tok/s at rpb=4), and it multiplies the shared-memory footprint
// which is the other thing this kernel must keep small.
// `WARPS` warps cooperate on ONE row (the shape llama.cpp's `mul_mat_vec_q`
// uses: `blocks_per_iter = vdr*nwarps*32/qi`, so the whole block covers the row
// in a single pass).  The alternative -- each warp taking its own row, which is
// what this kernel originally did via `row = blockIdx.x*WARPS + warp` -- only
// ever gave one warp's worth of loads per row in flight, and on 27B IQ3_XXS
// decode that left the reduce cost dominant: a 20-block row needs 20 serial
// `__fadd_rn` plus 20 shared stores per token, against only 5 rounds of actual
// dot work when SPLIT=8.  Measured on the 27B: 26.9 tok/s with one warp per row
// vs a warp-group per row below.
//
// Lane->work mapping: lane `l` handles block `b` and the `l % SPLIT` slice of it,
// blocks stride by `WARPS*Q35_WARP/SPLIT` so the warp group covers the row in
// `nblk / (WARPS * Q35_WARP / SPLIT)` rounds and each lane's reads for a block
// stay contiguous.
template <int WARPS, int SPLIT>
__global__ void q35_k_mul_mat_vec_q_split_t(const unsigned char *__restrict__ w,
                                            const unsigned char *__restrict__ xq,
                                            float *__restrict__ out,
                                            int rows, int cols, int tokens, int ttype) {
    const Q35QuantInfo info = q35_quant_info(ttype);
    const int nblk = cols / info.block_elems;
    const int lane = threadIdx.x & (Q35_WARP - 1);
    const int warp = threadIdx.x / Q35_WARP;

    // One float per block for the whole warp group: each block's FULL sum lands
    // here, so the lane-0 fold below is the same strict b = 0..nblk-1 order the
    // coop kernel uses (bit-identical reduction order, checked by MVCHECK).
    extern __shared__ float part_dyn[];
    float *part_w = part_dyn;

    const int row = blockIdx.x;
    if (row >= rows) return;
    const size_t row_stride = (size_t)nblk * (size_t)info.block_bytes;

    const int part = lane & (SPLIT - 1);        // which slice of the block
    const int lblock = lane / SPLIT;            // which lane-slot within a block
    constexpr int kBlocksPerStep = Q35_WARP / SPLIT;
    // All WARPS warps walk one interleaved block sequence so no block is covered
    // twice: warp `w` starts at `lblock + w*kBlocksPerStep` and every step
    // advances by the full group width.
    constexpr int kGroupStep = WARPS * kBlocksPerStep;
    // Round the block count UP to a multiple of `kGroupStep` so every lane
    // executes the same number of loop rounds.  This is not cosmetic: with a
    // ragged trip count the warp is left DIVERGENT at the loop exit, and the
    // `__shfl_down_sync` below would have to run on a partial mask.  (The
    // original version used a full-mask `__syncwarp()` there instead and hung
    // the device outright whenever `nblk % kBlocksPerStep != 0`; measured:
    // SPLIT=2/4 hung on the very first block, while SPLIT=8 survived only
    // because 20 % 4 == 0.)
    const int rounds = (nblk + kGroupStep - 1) / kGroupStep;
    // Fast path: when `nblk` divides evenly the padding never triggers, so the
    // in-range test inside the loop is pure overhead on the hot path.
    const bool even = (nblk % kGroupStep) == 0;

    for (int t = 0; t < tokens; ++t) {
        const unsigned char *arow = xq + (size_t)t * (size_t)nblk * (size_t)info.act_bytes;
        for (int r = 0; r < rounds; ++r) {
            const int b = lblock + warp * kBlocksPerStep + r * kGroupStep;
            float v = 0.0f;
            if (even || b < nblk) {
                const unsigned char *pb = w + (size_t)row * row_stride +
                                          (size_t)b * (size_t)info.block_bytes;
                const unsigned char *ab = arow + (size_t)b * (size_t)info.act_bytes;
                v = q35_dot_blk_part(ttype, pb, ab, part, SPLIT);
            }
            // Butterfly within the SPLIT-lane subgroup.  Lanes 0..SPLIT-1 of
            // each group all end up with the same value; only `part == 0` keeps
            // it (the store below).  Executed by EVERY lane, including the ones
            // whose `b` is out of range, to stay converged.
            #pragma unroll
            for (int o = 1; o < SPLIT; o <<= 1) v += __shfl_down_sync(0xffffffffu, v, o);
            if (part == 0 && (even || b < nblk)) part_w[b] = v;
        }
        // Block-wide barrier, NOT `__syncwarp()`: the WARPS warps of this block
        // all write into the shared `part_w`, so the fold has to wait for every
        // warp, not just this one.  It must be reached by ALL threads
        // unconditionally -- a `__syncthreads()` inside the `warp == 0` branch
        // deadlocks the block.
        __syncthreads();
        // Strict left-to-right fold, matching q35_dot_row's order, on one lane.
        if (warp == 0 && lane == 0) {
            float acc = 0.0f;
            for (int b = 0; b < nblk; ++b) acc = __fadd_rn(acc, part_w[b]);
            out[(size_t)t * rows + row] = acc;
        }
    }
}

// ---------------------------------------------------------------------------
// f16 / bf16 weights x f32 activation (the "w16" family).
//
// f16 and bf16 have no block metadata, so there is nothing to quantize: the
// weights stay in their native 2-byte form on the device (one u16 per element)
// and are expanded INSIDE the kernel.  No f32 copy of the weights exists
// anywhere -- resident footprint is 2 B/elem, not 4.
//
// The activation is the engine's own f32 vector, NOT a quantized copy:
//   - the widening is exact for both types (bf16 is the truncated high half of
//     an f32; f16 -> f32 is exact), so the products are the same f32 values the
//     materialized-f32 kernel multiplied;
//   - quantizing the activation to Q8_0 costs ~5 significant digits, and on a
//     0.8B BF16 model that is enough to flip an argmax in a near-tie and
//     diverge the rest of the generation.  Measured: the Q8_0-activation CPU
//     path matched the baseline for 42 tokens, then diverged.
//
// Accumulation order is one thread per output row, `c = 0..cols-1` left to
// right -- the same fold `q35_k_mul_mat_vec_f32` below uses, so a bf16 row here
// is bit-identical to the same row run as f32, and the CUDA BF16 token stream
// stays identical to the f32-materialization baseline.  A warp-cooperative
// reduction would agree only to ~1 ulp; deliberately not used (see the file
// header's reduction-order contract).
// ---------------------------------------------------------------------------
__global__ void q35_k_mul_mat_vec_w16(const unsigned char *__restrict__ w,
                                      const float *__restrict__ x,
                                      float *__restrict__ out,
                                      int rows, int cols, int tokens, int ttype) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int total = rows * tokens;
    if (idx >= total) return;
    const int t = idx / rows;
    const int r = idx - t * rows;
    const unsigned char *wrow = w + (size_t)r * (size_t)cols * (size_t)Q35_BYTES_F16;
    const float *xrow = x + (size_t)t * (size_t)cols;
    out[(size_t)t * rows + r] = q35_dot_row_w16(ttype, wrow, xrow, cols);
}

// f32 weights x f32 activation, same one-thread-per-element mapping.
__global__ void q35_k_mul_mat_vec_f32(const float *__restrict__ w,
                                      const float *__restrict__ x,
                                      float *__restrict__ out,
                                      int rows, int cols, int tokens) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= rows * tokens) return;
    const int t = idx / rows;
    const int r = idx - t * rows;
    const float *wrow = w + (size_t)r * (size_t)cols;
    const float *xrow = x + (size_t)t * (size_t)cols;
    float acc = 0.0f;
    for (int c = 0; c < cols; ++c) acc = fmaf(wrow[c], xrow[c], acc);
    out[(size_t)t * rows + r] = acc;
}

// ---------------------------------------------------------------------------
// Q8_K activation quantizer (f32 -> block_q8_K, 292 B / 256 elems)
//
// Bit-exact port of src/kernels/x86/kernels256.c:392 quantize_q8_K.  Three
// details that are easy to get wrong and that this reproduces deliberately:
//   1. `scale` goes into the header as a raw f32 (NOT rounded to f16 as Q8_0
//      does), and `inv` is built from that same f32 scale.
//   2. Rounding is roundf() = round half away from zero.
//   3. The clamp is [-127, 127], NOT [-128, 127]: the IQ kernels negate q8
//      (x86 _mm256_sign_epi8 in the CPU kernel, a plain negate on device), and
//      -128 has no positive counterpart.  The `!(r >= -127) -> 127` line is the
//      NaN/Inf guard (NaN compares false and would otherwise become garbage).
//   4. bsums[16] are int16 sums over each 16-element group of qs, little-endian,
//      consumed by the Q2_K/Q4_K/Q5_K min terms and the IQ1 delta terms.
// One thread per block (a block is 256 elements, so per-thread work is fine).
// ---------------------------------------------------------------------------
__global__ void q35_k_quantize_q8_K(const float *__restrict__ x,
                                    unsigned char *__restrict__ out,
                                    int nblocks) {
    const int b = blockIdx.x * blockDim.x + threadIdx.x;
    if (b >= nblocks) return;
    const float *xb = x + (size_t)b * Q35_QK8K;
    unsigned char *ob = out + (size_t)b * Q35_BYTES_Q8_K;

    float amax = 0.0f;
    for (int k = 0; k < Q35_QK8K; ++k) {
        const float a = xb[k] < 0.0f ? -xb[k] : xb[k];
        if (a > amax) amax = a;
    }
    const float scale = amax > 0.0f ? amax / 127.0f : 1.0f;
    const float inv = amax > 0.0f ? 1.0f / scale : 0.0f;
    const unsigned int sbits = __float_as_uint(scale);
    ob[0] = (unsigned char)(sbits & 0xFFu);
    ob[1] = (unsigned char)((sbits >> 8) & 0xFFu);
    ob[2] = (unsigned char)((sbits >> 16) & 0xFFu);
    ob[3] = (unsigned char)((sbits >> 24) & 0xFFu);

    int gsum[16];
#pragma unroll
    for (int g = 0; g < 16; ++g) gsum[g] = 0;
    for (int k = 0; k < Q35_QK8K; ++k) {
        float r = roundf(xb[k] * inv);
        if (r < -127.0f) r = -127.0f;
        if (r > 127.0f) r = 127.0f;
        if (!(r >= -127.0f)) r = 127.0f;  // NaN/Inf guard, see note 3
        const signed char q = (signed char)(int)r;
        ob[4 + k] = (unsigned char)q;
        gsum[k / 16] += (int)q;
    }
    for (int g = 0; g < 16; ++g) {
        const short v = (short)gsum[g];
        ob[260 + g * 2 + 0] = (unsigned char)((unsigned short)v & 0xFFu);
        ob[260 + g * 2 + 1] = (unsigned char)(((unsigned short)v >> 8) & 0xFFu);
    }
}

// ---------------------------------------------------------------------------
// launchers
//
// Grid sizing lives here because Zig cannot express `<<< >>>`.  Errors are
// returned as the numeric cudaError_t (0 == cudaSuccess) so the Zig side can
// surface cudaGetErrorString().
// ---------------------------------------------------------------------------

// ---------------------------------------------------------------------------
// `$QWEN35_CUDA_MMQ=0|1`: which kernel serves a prefill (tokens > 1) matvec.
//
//   1 (default)  the tile kernel in mmq.cu -- K-dimension reuse (a weight block
//                is dotted into every token of the chunk) plus coalesced loads
//                (a warp reads 32 consecutive blocks of ONE row), which is
//                what the thread-per-row kernel below cannot do, and what the
//                warp-per-row kernel can only do per token, serially.
//   0            the thread-per-row kernel, kept as the A/B reference (and as
//                the fallback, see below).
//
// Any other value (including unset) keeps the default.  Read once and cached:
// this is per launch and `getenv` per launch is the kind of syscall this
// backend exists to avoid.  A launch failure is NOT fatal and is NOT silent
// either: the message names the kernel and the tile, and the old kernel runs.
static bool q35_mmq_enabled() {
    static int cached = -1;
    if (cached < 0) {
        // Default ON.  This switch now gates the whole mmq.cu tensor-core
        // prefill path, and `q35_launch_mul_mat_vec_tile` decides per TYPE
        // whether the tensor-core kernel serves it: it returns 0 (handled) for
        // the 32-element family, the K family and IQ3_XXS, and -1 ("use the
        // fallback") for everything still without an mma kernel -- so enabling
        // this cannot route a type to the old dp4a tile kernel, which was the
        // reason the default used to be OFF (dp4a tile 19.3 tok/s vs 36.2
        // thread-per-row on 27B IQ3_XXS).  `=0` stays the A/B reference: it
        // forces the thread-per-row kernel for every type.
        cached = 1;
        const char *raw = getenv("QWEN35_CUDA_MMQ");
        if (raw != NULL) {
            const char *v = raw;
            while (*v == ' ' || *v == '\t') ++v;
            const bool tail_ok = (v[1] == '\0' || v[1] == ' ' || v[1] == '\t' ||
                                  v[1] == '\r' || v[1] == '\n');
            if (v[0] == '0' && tail_ok) {
                cached = 0;
            } else if (v[0] == '1' && tail_ok) {
                cached = 1;
            } else {
                // Not 0/1: keep the default and say so, exactly like
                // $QWEN35_CUDA_ROPE does for a bad value in backend_cuda.zig.
                fprintf(stderr, "cuda kernels: QWEN35_CUDA_MMQ='%s' is neither '0' nor '1'; using 1\n", v);
            }
        }
        if (cached == 0) fprintf(stderr, "cuda kernels: prefill matvec = thread-per-row ($QWEN35_CUDA_MMQ=1 for the tile kernel)\n");
        else fprintf(stderr, "cuda kernels: prefill matvec = mmq tile ($QWEN35_CUDA_MMQ)\n");
    }
    return cached == 1;
}

// Defined in mmq.cu.  Return value contract (different from cudaError_t, which
// is >= 0): cudaSuccess (0) == handled, every output element written; -1 ==
// "not my shape, use the fallback"; anything else == a real launch failure.
// Declared weak so a build without mmq.cu still links; nvcc rejects the GNU
// attribute on a bare declaration, so it goes through a pointer.
extern "C" int q35_launch_mul_mat_vec_tile(const void *d_w, const void *d_xq,
                                           float *d_out, int rows, int cols,
                                           int tokens, int ttype, void *stream)
    __attribute__((weak));

// ---------------------------------------------------------------------------
// Warp-per-row block shape: ONE decision, picked once, from the GPU.
//
// The decode (tokens == 1) coop kernel is launched with WARPS warps per block
// (one row per warp, block size WARPS*Q35_WARP threads).  The shared array
// `part` inside the kernel is exactly WARPS deep, so the launcher's block
// size and the kernel's `part` dimension are the SAME number -- this function
// is the single place that fixes it.
//
// The shape is a hardware fact, probed once via caps.cu (the CUDA-side mirror
// of src/kernels/x86/cpu_caps.c):
//   * sm_89:  4 warps (128 threads).  The measured-best shape on this box
//     (4090D): decode 128-thread -> 124 tok/s vs 256-thread -> 77 tok/s.
//   * any other compute capability: the occupancy formula -- the warp count
//     that maximises RESIDENT warps per SM, queried from the compiled kernel
//     via cudaOccupancyMaxActiveBlocksPerMultiprocessor.  The decode matvec is
//     bandwidth-bound, so resident-warps-per-SM is the right proxy; the
//     occupancy API accounts for register / shared / thread limits the way
//     the hardware actually does.
//
// $QWEN35_CUDA_COOP_WARPS=<1..8> overrides the shape (A/B only, like
// $QWEN35_CUDA_MMQ).  An empty or out-of-range value is ignored with a note.
// A failed probe (no usable device) keeps the sm_89-measured 4, which is the
// value every prior build used, so an unprobed device never regresses.
// ---------------------------------------------------------------------------
// Occupancy formula for a non-sm_89 device: the warp count that maximises
// RESIDENT warps per SM.  A kernel function pointer may not be stored in an
// object-pointer variable (C++ [conv.ptr] defines the pointer-to-function ->
// pointer-to-void conversion only at a function call), so each instantiation
// is passed as a first-class function pointer (the `auto` parameter) and the
// `void*` conversion happens right at the occupancy call.
static int q35_coop_occ_best_warps(int nblk) {
    int best_warps = 0;
    int best_resident = -1;
    auto occ = [&](auto kern, int warps) {
        int active = 0;
        // The kernel's `part` is dynamic shared sized WARPS * nblk floats, so
        // each candidate's per-block shared depends on that candidate's warp
        // count: pass it so the resident-block count is what the hardware
        // actually gives (passing 0 would over-estimate residency once nblk is
        // large enough that shared, not threads, becomes the limit).
        const size_t shmem = (size_t)warps * (size_t)nblk * sizeof(float);
        if (cudaOccupancyMaxActiveBlocksPerMultiprocessor(
                &active, (const void *)kern, warps * Q35_WARP, shmem) == cudaSuccess
            && active * warps > best_resident) {
            best_resident = active * warps;
            best_warps = warps;
        }
    };
    occ(q35_k_mul_mat_vec_q_coop_t<1>, 1);
    occ(q35_k_mul_mat_vec_q_coop_t<2>, 2);
    occ(q35_k_mul_mat_vec_q_coop_t<3>, 3);
    occ(q35_k_mul_mat_vec_q_coop_t<4>, 4);
    occ(q35_k_mul_mat_vec_q_coop_t<5>, 5);
    occ(q35_k_mul_mat_vec_q_coop_t<6>, 6);
    occ(q35_k_mul_mat_vec_q_coop_t<7>, 7);
    occ(q35_k_mul_mat_vec_q_coop_t<8>, 8);
    return best_warps;
}

/// `$QWEN35_CUDA_MEMONLY`: load-only decode matvec (see the memonly branch in
/// q35_k_mul_mat_vec_q_coop_t).  Diagnostic only -- it produces garbage output,
/// so it exists to answer "how much of the decode matvec time is the IQ dot
/// product" and must never be used for a real run.
static int q35_memonly() {
    static int cached = -1;
    if (cached >= 0) return cached;
    const char *raw = getenv("QWEN35_CUDA_MEMONLY");
    cached = (raw != NULL && *raw && strcmp(raw, "0") != 0) ? 1 : 0;
    if (cached) fprintf(stderr, "cuda kernels: decode matvec = LOAD ONLY (no IQ math); output is garbage\n");
    return cached;
}

/// `$QWEN35_CUDA_RPB`: rows per warp in the decode matvec (see q35_k_mul_mat_vec_q_coop_t).
/// 1 = the original one-row-per-warp mapping.
///
/// MEASURED, AND IT DOES NOT WORK: on 27B IQ3_XXS (RTX 4090 D, 4 warps, decode
/// tok/s) rpb = 1/2/4/8/16 gives 17.82 / 17.35 / 16.84 / 15.04 / 12.62 --
/// strictly WORSE as rpb grows.  The reasoning that motivated it (nblk = 20 < 32,
/// so a warp's lanes each do one block per row, giving rpb independent
/// q35_dot_1blk chains) is correct about ILP and wrong about everything else:
/// with rpb = 1 the 1280 blocks already fill all 114 SMs, so there is no
/// parallelism left to add, and the only thing rpb changes is the shared-memory
/// footprint (warps * rpb * nblk floats = 20 KiB at rpb = 16), which lowers
/// occupancy.  Rows-per-warp is NOT the lever; the chain length inside
/// q35_dot_1blk (256 elements per lane) is.  Keep the knob as an A/B tool, do
/// not tune it.
static int q35_rpb() {
    static int cached = -1;
    if (cached >= 0) return cached;
    int v = 1;
    const char *raw = getenv("QWEN35_CUDA_RPB");
    if (raw != NULL && *raw) {
        int t = atoi(raw);
        if (t >= 1 && t <= 32) v = t;
        else fprintf(stderr, "cuda kernels: QWEN35_CUDA_RPB='%s' is not in 1..32; using 1\n", raw);
    }
    cached = v;
    fprintf(stderr, "cuda kernels: decode matvec rows-per-warp = %d ($QWEN35_CUDA_RPB overrides)\n", cached);
    return cached;
}

static int q35_coop_warps(int nblk) {
    static int cached = -1;
    if (cached >= 0) return cached;
    int warps = 0;
    const char *raw = getenv("QWEN35_CUDA_COOP_WARPS");
    if (raw != NULL && *raw) {
        int v = atoi(raw);
        if (v >= 1 && v <= 8) {
            warps = v;
        } else {
            fprintf(stderr,
                    "cuda kernels: QWEN35_CUDA_COOP_WARPS='%s' is not in 1..8; ignoring\n",
                    raw);
        }
    }
    if (warps == 0) {
        const int cc = q35_gpu_cc();
        if (cc == 809) {
            // sm_89 measured best: 4 warps (128 threads).  See the header
            // comment above for the 124 vs 77 tok/s decode measurement.
            warps = 4;
        } else if (cc > 0) {
            const int best = q35_coop_occ_best_warps(nblk);
            warps = (best >= 1 && best <= 8) ? best : 4;
        } else {
            // No usable device (probe failed): keep the sm_89-measured value.
            warps = 4;
        }
    }
    cached = warps;
    fprintf(stderr,
            "cuda kernels: decode matvec block = %d warps/block (%d threads); "
            "device cc=%d \"%s\" %d SMs ($QWEN35_CUDA_COOP_WARPS overrides)\n",
            cached, cached * Q35_WARP, q35_gpu_cc(), q35_gpu_name(), q35_gpu_sm_count());
    return cached;
}

// How many lanes cooperate on one 256-element weight block in the decode
// kernel.  0 (or 1) means "use the whole-block coop kernel".
// `$QWEN35_CUDA_SPLIT` overrides; accepted values 0/1/2/4/8/16/32.  A value the
// type cannot support (`q35_split_units(ttype) % split != 0`) is clamped down to
// the largest supported divisor at the call site, so the env var can never make
// the kernel read a slice that does not exist.
static int q35_split() {
    static int cached = -1;
    if (cached >= 0) return cached;
    int v = 8;   // measured-best on sm_89 for IQ3_XXS (see the kernel header)
    const char *raw = getenv("QWEN35_CUDA_SPLIT");
    if (raw != NULL && *raw) {
        int t = atoi(raw);
        if (t == 0 || t == 1 || t == 2 || t == 4 || t == 8 || t == 16 || t == 32) v = t;
        else fprintf(stderr, "cuda kernels: QWEN35_CUDA_SPLIT='%s' is not one of 0/1/2/4/8/16/32; using 8\n", raw);
    }
    cached = v;
    fprintf(stderr, "cuda kernels: decode K-split lanes/block = %d ($QWEN35_CUDA_SPLIT overrides)\n", cached);
    return cached;
}

extern "C" int q35_launch_mul_mat_vec(const void *d_w_v, const void *d_xq_v,
                                      float *d_out, int rows, int cols,
                                      int tokens, int ttype, void *stream_v) {
    const unsigned char *d_w = (const unsigned char *)d_w_v;
    const unsigned char *d_xq = (const unsigned char *)d_xq_v;
    cudaStream_t stream = (cudaStream_t)stream_v;
    if (rows <= 0 || cols <= 0 || tokens <= 0) return (int)cudaSuccess;
    // Host-side validation: a wrong ttype or a cols that is not a multiple of
    // the type's block size would silently read out of bounds in the kernel.
    const Q35QuantInfo info = q35_quant_info(ttype);
    if (info.block_elems == 0) return (int)cudaErrorInvalidValue;
    if (cols % info.block_elems != 0) return (int)cudaErrorInvalidValue;

    // Prefill only: a single token has nothing to reuse, and the warp-per-row
    // kernel already coalesces it (below).
    if (tokens > 1 && q35_mmq_enabled() && q35_launch_mul_mat_vec_tile != 0) {
        const int rc = q35_launch_mul_mat_vec_tile(d_w, d_xq, d_out, rows, cols,
                                                   tokens, ttype, stream_v);
        if (rc == (int)cudaSuccess) return rc;   // the tile kernel owns the output
        if (rc != -1) {
            // A real launch failure: nothing was written, so report it and run
            // the old kernel below, which is correct for every shape.
            fprintf(stderr,
                    "cuda kernels: mmq tile launch failed (%d: %s) for rows=%d cols=%d "
                    "tokens=%d ttype=%d; falling back to the thread-per-row kernel\n",
                    rc, cudaGetErrorString((cudaError_t)rc), rows, cols, tokens, ttype);
            (void)cudaGetLastError();   // clear so the fallback's check is clean
        }
        // rc == -1: shape the tile was not sized for -> silent, intended fallback.
    }
    const int total = rows * tokens;
    const int nblk = cols / info.block_elems;
    // The warp-per-row mapping wins only for single-token decoding: with a
    // prefill chunk the tile is reused across `tokens`, and the per-token
    // `__syncwarp()` plus lane-0 fold cost more than the coalescing saves
    // (measured: 27B IQ3_XXS prefill 11.3 -> 10.1 t/s).  Keep the thread-per-row
    // kernel for tokens > 1, where each token already fills the SMs.
    if (tokens == 1) {
        // K-split kernel first: it needs a type with a `_part` body and a block
        // whose sub-block count divides `SPLIT` evenly.  `q35_split()` returns
        // the wanted lane count; clamp to the largest divisor of the type's unit
        // count so an unsupported value degrades to a valid shape instead of
        // reading a slice that does not exist.
        {
            const int units = q35_split_units(ttype);
            int split = q35_split();
            if (units > 0 && split > 1) {
                // Clamp to the largest divisor of the unit count, capped at the
                // unit count itself: `split > units` would give every lane an
                // EMPTY sub-block range and the dot would contribute nothing
                // (measured: IQ3_XXS at SPLIT=16 "ran" fine and fast on a
                // zero-valued dot product).
                while (split > 1 && (units % split) != 0) split >>= 1;
                if (split > units) split = units;
                // A type whose `_part` body needs a MINIMUM slice width (IQ3_S
                // can only start on a pair boundary, so `per` must be even and
                // at least 2) would otherwise be handed a one-unit slice that
                // its body rejects -- and a rejected slice returns 0.0f, i.e.
                // the block silently contributes nothing while the run stays
                // fast and the output looks plausible.  Clamp instead of
                // trusting the body's own guard, so `split` can never describe
                // a shape the body cannot compute.
                const int min_per = q35_split_min_per(ttype);
                while (split > 1 && (units / split) < min_per) split >>= 1;
            } else {
                split = 0;
            }
            if (split >= 2 && q35_memonly() == 0) {
                const int warps_s = q35_coop_warps(nblk);
                const size_t shmem_s = (size_t)nblk * sizeof(float);
                if (shmem_s > 48u * 1024u) {
                    static size_t max_set_s = 0;
                    if (shmem_s > max_set_s) {
                        auto opt = [&](auto kern) {
                            cudaError_t e = cudaFuncSetAttribute((const void *)kern,
                                cudaFuncAttributeMaxDynamicSharedMemorySize, (int)shmem_s);
                            if (e != cudaSuccess) {
                                // Silently ignoring this used to leave the launch
                                // asking for more dynamic shmem than the variant
                                // was ever granted, i.e. an immediate crash on the
                                // first block.  Fail the launch instead, so the
                                // caller reports the kernel as unsupported.
                                fprintf(stderr, "[cuda] dynshmem %zu B rejected: %s\n",
                                        shmem_s, cudaGetErrorString(e));
                                cudaGetLastError();
                            }
                        };
                        opt(q35_k_mul_mat_vec_q_split_t<1, 2>);
                        opt(q35_k_mul_mat_vec_q_split_t<1, 4>);
                        opt(q35_k_mul_mat_vec_q_split_t<1, 8>);
                        opt(q35_k_mul_mat_vec_q_split_t<2, 2>);
                        opt(q35_k_mul_mat_vec_q_split_t<2, 4>);
                        opt(q35_k_mul_mat_vec_q_split_t<2, 8>);
                        opt(q35_k_mul_mat_vec_q_split_t<4, 2>);
                        opt(q35_k_mul_mat_vec_q_split_t<4, 4>);
                        opt(q35_k_mul_mat_vec_q_split_t<4, 8>);
                        opt(q35_k_mul_mat_vec_q_split_t<8, 2>);
                        opt(q35_k_mul_mat_vec_q_split_t<8, 4>);
                        opt(q35_k_mul_mat_vec_q_split_t<8, 8>);
                        max_set_s = shmem_s;
                    }
                }
                // One block per ROW now: the WARPS warps inside it split that
                // row's K dimension between them (see the kernel header), so
                // the shared buffer is one `nblk`-float row, not WARPS of them.
                const int blocks_s = rows;
                #define Q35_SPLIT_LAUNCH(W, S) \
                    q35_k_mul_mat_vec_q_split_t<W, S><<<blocks_s, W * Q35_WARP, shmem_s, stream>>>( \
                        d_w, d_xq, d_out, rows, cols, tokens, ttype)
                #define Q35_SPLIT_SW(S) \
                    switch (warps_s) { \
                    case 1:  Q35_SPLIT_LAUNCH(1, S); break; \
                    case 2:  Q35_SPLIT_LAUNCH(2, S); break; \
                    case 4:  Q35_SPLIT_LAUNCH(4, S); break; \
                    default: Q35_SPLIT_LAUNCH(8, S); break; \
                    }
                if (getenv("QWEN35_SPLITDBG")) {
                    fprintf(stderr, "[split] ttype=%d rows=%d cols=%d nblk=%d split=%d warps=%d shmem=%zu\n",
                            ttype, rows, cols, nblk, split, warps_s, shmem_s);
                }
                switch (split) {
                case 2:  Q35_SPLIT_SW(2); break;
                case 4:  Q35_SPLIT_SW(4); break;
                default: Q35_SPLIT_SW(8); break;
                }
                #undef Q35_SPLIT_SW
                #undef Q35_SPLIT_LAUNCH
                return (int)cudaGetLastError();
            }
        }
        // One warp per row: coalesced because the warp's lanes read consecutive
        // blocks of a single row.  See q35_k_mul_mat_vec_q_coop for why this is
        // numerically identical to the thread-per-row mapping (bit-identical:
        // part[warp][b] holds block b's term, lane 0 folds b = 0..nblk-1 with
        // __fadd_rn -- the exact order q35_dot_row uses).
        // nblk is unbounded: `part` is dynamic shared, sized WARPS * nblk floats
        // and passed as the third <<<>>> argument (was a static part[WARPS]
        // [Q35_COOP_MAXNBLK] with nblk<=256; the bound is gone).
        const int warps = q35_coop_warps(nblk);
        const int memonly = q35_memonly();
        const int rpb = q35_rpb();
        const int rows_per_block = warps * rpb;
        const int blocks = (rows + rows_per_block - 1) / rows_per_block;
        const size_t shmem = (size_t)warps * (size_t)rpb * (size_t)nblk * sizeof(float);
        if (shmem > 48u * 1024u) {
            // Above the default dynamic-shared ceiling (48 KiB/block).  Raise it
            // (sticky attribute, set on every instantiation; the launch below
            // uses the matching one) -- without this the launch fails and
            // decode silently degrades to thread-per-row for wide rows.
            // NOTE: the attribute can only ever GROW (a smaller set returns
            // cudaErrorInvalidValue), so track the high water mark and skip
            // the call when nothing grew.
            static size_t max_set = 0;
            if (shmem > max_set) {
                auto opt = [&](auto kern) {
                    cudaFuncSetAttribute(
                        (const void *)kern,
                        cudaFuncAttributeMaxDynamicSharedMemorySize, (int)shmem);
                };
                opt(q35_k_mul_mat_vec_q_coop_t<1>);
                opt(q35_k_mul_mat_vec_q_coop_t<2>);
                opt(q35_k_mul_mat_vec_q_coop_t<3>);
                opt(q35_k_mul_mat_vec_q_coop_t<4>);
                opt(q35_k_mul_mat_vec_q_coop_t<5>);
                opt(q35_k_mul_mat_vec_q_coop_t<6>);
                opt(q35_k_mul_mat_vec_q_coop_t<7>);
                opt(q35_k_mul_mat_vec_q_coop_t<8>);
                max_set = shmem;
            }
        }
        switch (warps) {
        case 1:  q35_k_mul_mat_vec_q_coop_t<1><<<blocks, 1 * Q35_WARP, shmem, stream>>>(d_w, d_xq, d_out, rows, cols, tokens, ttype, memonly, rpb); break;
        case 2:  q35_k_mul_mat_vec_q_coop_t<2><<<blocks, 2 * Q35_WARP, shmem, stream>>>(d_w, d_xq, d_out, rows, cols, tokens, ttype, memonly, rpb); break;
        case 3:  q35_k_mul_mat_vec_q_coop_t<3><<<blocks, 3 * Q35_WARP, shmem, stream>>>(d_w, d_xq, d_out, rows, cols, tokens, ttype, memonly, rpb); break;
        case 4:  q35_k_mul_mat_vec_q_coop_t<4><<<blocks, 4 * Q35_WARP, shmem, stream>>>(d_w, d_xq, d_out, rows, cols, tokens, ttype, memonly, rpb); break;
        case 5:  q35_k_mul_mat_vec_q_coop_t<5><<<blocks, 5 * Q35_WARP, shmem, stream>>>(d_w, d_xq, d_out, rows, cols, tokens, ttype, memonly, rpb); break;
        case 6:  q35_k_mul_mat_vec_q_coop_t<6><<<blocks, 6 * Q35_WARP, shmem, stream>>>(d_w, d_xq, d_out, rows, cols, tokens, ttype, memonly, rpb); break;
        case 7:  q35_k_mul_mat_vec_q_coop_t<7><<<blocks, 7 * Q35_WARP, shmem, stream>>>(d_w, d_xq, d_out, rows, cols, tokens, ttype, memonly, rpb); break;
        default: q35_k_mul_mat_vec_q_coop_t<8><<<blocks, 8 * Q35_WARP, shmem, stream>>>(d_w, d_xq, d_out, rows, cols, tokens, ttype, memonly, rpb); break;
        }
        return (int)cudaGetLastError();
    }
    const int threads = 128;
    const int blocks = (total + threads - 1) / threads;
    q35_k_mul_mat_vec_q<<<blocks, threads, 0, stream>>>(d_w, d_xq, d_out, rows, cols,
                                                        tokens, ttype);
    return (int)cudaGetLastError();
}

extern "C" int q35_launch_mul_mat_vec_q8_0(const void *d_w, const void *d_xq,
                                           float *d_out, int rows, int cols,
                                           int tokens, void *stream) {
    // Backwards-compatible entry point: identical to
    // q35_launch_mul_mat_vec(..., Q35_TT_Q8_0, ...).
    return q35_launch_mul_mat_vec(d_w, d_xq, d_out, rows, cols, tokens,
                                  Q35_TT_Q8_0, stream);
}

extern "C" int q35_launch_mul_mat_vec_f32(const float *d_w, const float *d_x,
                                          float *d_out, int rows, int cols,
                                          int tokens, void *stream_v) {
    cudaStream_t stream = (cudaStream_t)stream_v;
    if (rows <= 0 || cols <= 0 || tokens <= 0) return (int)cudaSuccess;
    const int threads = 256;
    const int total = rows * tokens;
    const int blocks = (total + threads - 1) / threads;
    // Scalar order on purpose: the warp-per-row fold (~1 ulp different) changed
    // the BF16 token stream on the 0.8B probe for only +12% throughput, which is
    // not a trade worth making while the quantized path is the hot one.
    q35_k_mul_mat_vec_f32<<<blocks, threads, 0, stream>>>(d_w, d_x, d_out, rows, cols,
                                                          tokens);
    return (int)cudaGetLastError();
}

// f16/bf16 weights x f32 activation.  `d_w` is `rows*cols` u16 (2 B/elem),
// `d_x` is `tokens*cols` f32.  Same thread mapping as the f32 kernel, so the
// result is bit-identical to the f32-materialization path (see the w16 header
// above).  `ttype` picks bf16 (bits << 16) vs f16 (__half2float).
extern "C" int q35_launch_mul_mat_vec_w16(const void *d_w, const float *d_x,
                                          float *d_out, int rows, int cols,
                                          int tokens, int ttype, void *stream_v) {
    cudaStream_t stream = (cudaStream_t)stream_v;
    if (rows <= 0 || cols <= 0 || tokens <= 0) return (int)cudaSuccess;
    if (ttype != Q35_TT_F16 && ttype != Q35_TT_BF16) return (int)cudaErrorInvalidValue;
    const int threads = 256;
    const int total = rows * tokens;
    const int blocks = (total + threads - 1) / threads;
    q35_k_mul_mat_vec_w16<<<blocks, threads, 0, stream>>>(
        (const unsigned char *)d_w, d_x, d_out, rows, cols, tokens, ttype);
    return (int)cudaGetLastError();
}

extern "C" int q35_launch_quantize_q8_K(const float *d_x, void *d_out_v,
                                        int nblocks, void *stream_v) {
    unsigned char *d_out = (unsigned char *)d_out_v;
    cudaStream_t stream = (cudaStream_t)stream_v;
    if (nblocks <= 0) return (int)cudaSuccess;
    const int threads = 128;
    const int blocks = (nblocks + threads - 1) / threads;
    q35_k_quantize_q8_K<<<blocks, threads, 0, stream>>>(d_x, d_out, nblocks);
    return (int)cudaGetLastError();
}

// Which activation layout a type needs, for the Zig caller's buffer sizing.
// Returns 256 if the type uses Q8_K activations, 32 if it uses Q8_0 ones, and
// 0 for an unknown type.
extern "C" int q35_quant_act_block_elems(int ttype) {
    const Q35QuantInfo info = q35_quant_info(ttype);
    if (info.block_elems == 0) return 0;
    return info.wants_q8k ? Q35_QK8K : 32;
}
