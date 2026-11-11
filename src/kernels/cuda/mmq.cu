// mmq.cu -- prefill (tokens > 1) quantized mat-mul with K-dimension reuse and
// coalesced weight loads.
//
// ---------------------------------------------------------------------------
// WHY A THIRD KERNEL NEXT TO q35_k_mul_mat_vec_q / _coop (mmvq.cu)
// ---------------------------------------------------------------------------
// Both existing mappings waste the batchK path, for opposite reasons:
//
//   q35_k_mul_mat_vec_q  (thread -> (token, row))
//       Zero K-dimension reuse: each of the `tokens` threads owning row r
//       re-reads the whole weight row, so the weights are fetched `tokens`
//       times per chunk.  Also uncoalesced: neighbours in a warp own *different*
//       rows, which sit `nblk * block_bytes` apart (2200 B for IQ3_XXS at
//       cols=5120), so every load lands in its own 128 B sector.  Measured in
//       mmvq.cu: ~80 GB/s of the 4090D's ~1008 GB/s.
//
//   q35_k_mul_mat_vec_q_coop  (warp -> row, loop over tokens)
//       Coalesced, but serialised over tokens (one row's 32 lanes work through
//       token 0, then token 1, ...), needs a `__syncwarp` + lane-0 fold per
//       token, and bails out entirely when nblk > 32 -- so it cannot run the
//       27B model's FFN layers (cols = 17408 -> nblk = 68).  Measured in
//       mmvq.cu: switching it on for prefill made 11.3 -> 10.1 t/s.
//
// The mapping below removes both problems at once:
//
//   lane l covers the contiguous weight-block run [l*BLK, (l+1)*BLK) of one
//          row, so the 32 lanes of a warp read one contiguous
//          32 * BLK * block_bytes stretch of ONE weight row -- fully
//          coalesced, and the whole row in a single unrolled pass;
//   warp   owns one (weight row, token); a block is `wpt` rows x
//          `tokens_per_pass` tokens.  Every row of the tile is therefore
//          resident while all `tokens_per_pass` tokens are accumulated, so the
//          weight bytes are fetched once per chunk instead of once per token
//          (that is the K-dimension reuse).
//
// `blk` (blocks per lane) is a runtime parameter, `ceil(nblk / 32)`, so a warp
// covers the whole row in one pass.  That is what makes nblk > 32 work:
// cols = 17408 (nblk = 68) runs with blk = 3, no loop over the row and no tail
// branch.  It used to be a template parameter and was demoted because the
// compile cost scaled with its number of values -- see the kernel's own comment.
//
// ---------------------------------------------------------------------------
// REDUCTION ORDER IS UNCHANGED (hard requirement: mmvq.cu header, AGENTS.md)
// ---------------------------------------------------------------------------
// Each output element is still assembled strictly left to right over
// b = 0 .. nblk-1.  A lane keeps ONE accumulator per token and walks its own
// increasing contiguous run through `q35_dot_1blk(..., acc)`, i.e. exactly the
// sequence of float operations `q35_dot_row` performs for those b's -- the split
// is an inline expansion of that loop, not a different summation.  Compare
// `_coop`, which must gather partial sums from 32 lanes and fold them
// b-ascending to preserve the order; here a lane's terms are already in b order
// and no cross-lane arithmetic happens at all.
// Verified two ways: `$QWEN35_MVCHECK` (device vs CPU reference, per row over
// real model weights) and the `$QWEN35_CUDA_MMQ` token-id A/B.
//
// ---------------------------------------------------------------------------
// SHARED MEMORY / OCCUPANCY / MEASURED RESULT
// ---------------------------------------------------------------------------
// The only shared memory is the cross-lane fold buffer, `part[8][TPERPASS][32]`
// (see the fold commentary in the kernel body) -- it has to exist because the
// partials are produced by different lanes than the one that folds them.
// Everything else stays in registers: a warp reads its own row and warps never
// share one, so staging weight would only add a round trip.
//
// The binding constraint is register pressure from the `acc[TPERPASS]` array.
// Measured on this file with `-Xptxas -v`:
//     TPERPASS = 16 -> 127 registers/thread -> 2 blocks/SM (25% occupancy)
//     TPERPASS =  8 ->  fewer, fits more blocks
// and that difference IS the performance: 354-token prefill, 27B IQ3_XXS on the
// 4090D, thread-per-row baseline 30.4-30.9 s:
//     TPERPASS = 16   38.7 s   (9.1 tok/s)   <- occupancy dominates
//     TPERPASS =  8   30.4 s  (11.6 tok/s)   <- parity with the baseline
//
// So the K-reuse this kernel buys (one weight block dotted into TPERPASS tokens
// instead of re-read for each) is REAL but currently cancelled by two costs:
// the fold -- some lane must sum 32 partials serially per (row, token) while the
// other 31 idle, and the reduction order may not be changed (mmvq.cu header,
// AGENTS.md), so a tree/warp-shuffle fold is not available -- and the occupancy
// below.  Hence mmvq.cu leaves this kernel OFF by default.
//
// TODO(next attempt): the fold is the thing to kill, not the reuse.  Two shapes
// look viable and both keep b ascending: (a) split the row across fewer lanes
// (blk larger, fewer partials to fold) so TPERPASS can stay small; (b) write
// per-lane partials to a per-row global scratch and fold them in a second,
// trivially-parallel kernel -- order still ascending, but the fold leaves the
// critical path.  Re-measure both before flipping the default.
//
// `tokens > Q35_MMQ_MAX_TOKENS` (the engine's prefill chunk is 1024) is declined
// at launch time: the launcher returns -1, which mmvq.cu reads as "use the
// thread-per-row kernel".  A half-idle tile would be slower than the fallback,
// and a silently wrong tile is worse than either.
#include "dotq.cuh"
#include "mma.cuh"

#define Q35_MMQ_THREADS 256
#define Q35_MMQ_WARPS (Q35_MMQ_THREADS / Q35_WARP)
// engine.zig nbatch: must be >= the engine's chunk size K, or the launcher
// declines the shape and prefill silently drops to the thread-per-row kernel.
// Lifting this from 128 to 1024 is what lets a whole 525-token prompt run as
// ONE chunk: the weight slice is hoisted over the token loop, so weight traffic
// is (n_prompt/K) x model_bytes, and 5 chunks was 5 full passes over 10.3 GB.
// The cost is `acc[MAX_TILES][4]` -- MAX_TILES has to cover K/8 tiles and the
// accumulator lands in local memory, see the TILE_BATCH note in the kernel.
//
// MAX_TILES is deliberately SMALL (16 tiles = 128 tokens per block) and the
// launcher raises njsplit instead of letting it grow with the chunk size.
// History: it used to be Q35_MMQ_MAX_TOKENS/8 == 128 to cover the whole chunk
// in one block, which ptxas turned into a `2048 bytes stack frame` per thread.
// That frame is the accumulator living in local memory (spill stores/loads read
// 0 because the array is user-declared, not ptxas-spilled), and it is why the
// IQ tensor-core kernel measured 37 TFLOPS where llama.cpp measures 127 on the
// equivalent ggml shape (m=4096 n=512 k=14336, test-backend-ops).  MAX_TILES=16
// -> 128 B frame, and grid.y (=njsplit) supplies the parallelism instead; the
// duplicated weight traffic is absorbed by L2 (a 27B ffn slice is 34 MB against
// 72 MB of L2).  Measured 256.0 -> 264.3 tok/s on 27B IQ3_XXS prefill.
// Keep this in sync with the kernel's acc[] declaration.
#define Q35_MMQ_MAX_TILES 16
#define Q35_MMQ_MAX_TOKENS 1024

// ---------------------------------------------------------------------------
// One warp = one (weight row, token).  Lane l folds the block run
// [l*BLK, (l+1)*BLK) and writes the output element itself.
//
// `tokens_per_pass` consecutive tokens are served by `wpt * tokens_per_pass`
// warps of the block, so a block covers `wpt` rows x `tokens_per_pass` tokens.
// ---------------------------------------------------------------------------
//
// `blk` is a RUNTIME parameter, not a template one.  It was `template <int BLK,
// int TPERPASS>` and that made this file take >4 minutes to compile: every
// instantiation inlines the 19-branch `q35_dot_1blk` switch (it is
// `__forceinline__` by design, see dotq.cuh), so cost is
// (#BLK values) x (#TPERPASS values) x 19, and five BLK values were enough to
// push a single nvcc invocation past the point of usability (measured: 11 s per
// instantiation of a one-instance test translation unit).  `blk` only sizes the
// lane's block range -- `bend = min(b0 + blk, nblk)` is already a runtime
// expression -- so making it a parameter costs one integer multiply and buys a
// 5x shorter compile.  `TPERPASS` stays a template parameter because it drives
// `float acc[TPERPASS]`, i.e. a register array whose size must be static.
template <int TPERPASS>
__global__ void q35_k_mul_mat_vec_q_mmq(const unsigned char *__restrict__ w,
                                        const unsigned char *__restrict__ xq,
                                        float *__restrict__ out,
                                        int rows, int cols, int tokens, int ttype,
                                        int wpt, int blk) {
    const Q35QuantInfo info = q35_quant_info(ttype);
    const int nblk = cols / info.block_elems;
    const int lane = threadIdx.x & (Q35_WARP - 1);
    const int warp = threadIdx.x / Q35_WARP;
    // `tpp` warps cooperate on one row's token pass, `wpt` rows share the block:
    // tpp * wpt == number of warps in the block.  `wpt` comes from the launcher.
    const int tpp = Q35_MMQ_WARPS / wpt;
    const int t0 = blockIdx.y * (tpp * TPERPASS) + (warp / wpt) * TPERPASS;
    const int r = blockIdx.x * wpt + (warp - (warp / wpt) * wpt);
    if (r >= rows) return;

    // Lane-local, contiguous, increasing run of blocks.  Runs that fall past
    // `nblk` (the padding when BLK > nblk/32) are not entered, so an
    // accumulator sees exactly b = 0 .. nblk-1 and no filler terms.
    const int b0 = lane * blk;
    const int bend = min(b0 + blk, nblk);
    const unsigned char *wrow = w + (size_t)r * (size_t)nblk * (size_t)info.block_bytes;
    // How many lanes actually cover the row: the fold below must stop there.
    // Computed by every lane (not just lane 0) and clamped, because the
    // `__syncwarp()` further down is a convergence point -- a lane that
    // returned early here would deadlock the fold.
    const int wl = (nblk + blk - 1) / blk;

    float acc[TPERPASS];
#pragma unroll
    for (int j = 0; j < TPERPASS; ++j) acc[j] = 0.0f;

    // The weight block pointer advances once per b and is dotted into every
    // token of the pass: that is the reuse.  `q35_dot_1blk` receives the SAME
    // `wb` for each j, and `acc[j]` keeps b ascending, which is the whole
    // order-preservation argument.
    for (int b = b0; b < bend; ++b) {
        const unsigned char *wb = wrow + (size_t)b * (size_t)info.block_bytes;
#pragma unroll
        for (int j = 0; j < TPERPASS; ++j) {
            const int t = t0 + j;
            if (t >= tokens) continue;
            const unsigned char *ab =
                xq + ((size_t)t * (size_t)nblk + (size_t)b) * (size_t)info.act_bytes;
            acc[j] = q35_dot_1blk(ttype, wb, ab, acc[j]);
        }
    }

    // Cross-lane fold.  Each lane accumulated only its own contiguous run
    // [lane*blk, (lane+1)*blk), so the row's sum is the lanes' partials folded
    // in ASCENDING LANE ORDER -- lane L's run starts at L*blk, so folding
    // 0 .. Q35_WARP-1 walks b = 0 .. nblk-1 with no reordering.  This is the
    // step the first version of this kernel was missing: it wrote `acc[j]`
    // straight to `out`, so 32 lanes raced on the same address and the result
    // was whichever lane won (measured: every element mismatched).
    //
    // `__syncwarp()` before the reads (not after the writes alone) because the
    // partials are produced by different lanes than the one folding them.
    __shared__ float part[Q35_MMQ_WARPS][TPERPASS][Q35_WARP];
#pragma unroll
    for (int j = 0; j < TPERPASS; ++j) part[warp][j][lane] = acc[j];
    __syncwarp();

    if (lane == 0) {
#pragma unroll
        for (int j = 0; j < TPERPASS; ++j) {
            const int t = t0 + j;
            if (t >= tokens) continue;
            // Only `W` lanes of the warp touched the row: the tail lanes
            // (b0 >= nblk) returned early and their partials stay at the
            // `acc` init value of 0.0f only if we zero them here.  `wl` is the
            // count of lanes that actually ran, so the fold stops there.
            float sum = part[warp][j][0];
            for (int l = 1; l < wl; ++l) sum = __fadd_rn(sum, part[warp][j][l]);
            out[(size_t)t * (size_t)rows + (size_t)r] = sum;
        }
    }
}

// ---------------------------------------------------------------------------
// launcher
// ---------------------------------------------------------------------------

template <int TPERPASS>
static cudaError_t mmq_launch(const unsigned char *d_w, const unsigned char *d_xq,
                              float *d_out, int rows, int cols, int tokens,
                              int ttype, int blk, cudaStream_t stream) {
    // A block spends its warps on `wpt` rows x `tpp` token-passes.  The token
    // dimension is a grid axis (blockIdx.y), NOT folded into the row count: with
    // a fixed 8-warps-per-block there are only `tpp` passes available per block,
    // so `tokens` larger than `tpp * TPERPASS` needs more blocks, not a bigger
    // index expression.  (The first version derived t0 from (warp / wpt) only
    // and silently dropped every token past the first pass -- caught by MVCHECK
    // at k=16: half the rows mismatched.)
    int tpp = Q35_MMQ_WARPS / TPERPASS;      // warps working on one row
    if (tpp < 1) tpp = 1;
    const int wpt = Q35_MMQ_WARPS / tpp;     // rows per block
    const int tokens_per_block = tpp * TPERPASS;
    const dim3 grid((unsigned)((rows + wpt - 1) / wpt),
                    (unsigned)((tokens + tokens_per_block - 1) / tokens_per_block));
    q35_k_mul_mat_vec_q_mmq<TPERPASS><<<grid, Q35_MMQ_THREADS, 0, stream>>>(
        d_w, d_xq, d_out, rows, cols, tokens, ttype, wpt, blk);
    return cudaGetLastError();
}

// ---------------------------------------------------------------------------
// Tensor-core (mma.sync) prefill kernel, 32-element weight family
// (Q8_0, Q4_0, Q4_1, Q5_0, Q5_1, IQ4_NL).
//
// Replaces the dp4a inner dot with mma.sync m16n8k32 s32.s8.s8.s32.  Every
// type in this family is: dequant the weight block to int8 w8[32] (every value
// lands in the int8 range -- IQ4_NL table spans -127..113, the widest), let
// mma give the EXACT int32 dot sumi = sum_k w8[k]*a8[k], then apply the SAME
// per-block float scale in the SAME ascending block order as the dp4a path
// (dotq.cuh q35_vec_dot_*_q8_0_acc): Q4_1/Q5_1 add a sumy = sum a8 term.
// The integer dot is exact (no overflow: |sumi| <= 127*127*32 = 517760 <<
// 2^31) and the float ops are the same expressions in the same order, so the
// result is BIT-IDENTICAL to the dp4a tile kernel and the CPU reference.
// Verified standalone for Q8_0 (/tmp/mma_q8_0_test.cu): 0/2048 mismatched.
//
// All six use Q8_0 activation (act_bytes 34, f16 d @0 + int8 a8 @2..33), so
// the activation staging is shared; only the weight dequant (q35_mma_w8) and
// the optional sumy term differ.
//
// Tiling: one block = 8 warps = 128 weight rows (16/warp) x 8 tokens.  Each
// warp covers 16 consecutive rows for the same 8-token tile, so the mma A
// fragment is 16 rows x 32 k and B is 8 tokens x 32 k (one k32 mma per
// block b).  grid.x over 128-row groups, grid.y over 8-token tiles.
// ---------------------------------------------------------------------------

// Dequantize weight element `k` of native block at `wb` to its int8 form
// (the value mma multiplies by the activation int8).  Mirrors the per-type
// nibble layouts in vecdotq.cuh q35_vec_dot_*_q8_0_acc:
//   Q4_0/Q4_1/IQ4_NL: byte j -> element j (low nibble) and j+16 (high nibble);
//   Q5_0/Q5_1:        byte j/2 -> elements 2j, 2j+1 (q4 | high qh bit).
__device__ __forceinline__ int q35_mma_w8(int ttype, const unsigned char *wb, int k) {
    switch (ttype) {
    case Q35_TT_Q8_0:
        return (int)(int8_t)wb[2 + k];
    case Q35_TT_Q4_0:
        return (k < 16) ? (wb[2 + k] & 0x0F) - 8 : (int)(wb[2 + (k - 16)] >> 4) - 8;
    case Q35_TT_Q4_1:
        return (k < 16) ? (wb[4 + k] & 0x0F) : (int)(wb[4 + (k - 16)] >> 4);
    case Q35_TT_Q5_0: {
        const unsigned int qh = q35_ld_u32(wb + 2);
        const int q4 = (wb[6 + k / 2] >> (4 * (k & 1))) & 0x0F;
        return (q4 | (int)((qh >> k) & 1u) << 4) - 16;
    }
    case Q35_TT_Q5_1: {
        const unsigned int qh = q35_ld_u32(wb + 4);
        const int q4 = (wb[8 + k / 2] >> (4 * (k & 1))) & 0x0F;
        return q4 | (int)((qh >> k) & 1u) << 4;
    }
    case Q35_TT_IQ4_NL:
        return (k < 16) ? (int)kvalues_iq4nl[wb[2 + k] & 0x0F]
                        : (int)kvalues_iq4nl[wb[2 + (k - 16)] >> 4];
    default:
        return 0;  // unreachable: host validates ttype
    }
}

__global__ void q35_k_mmq_mma_q32(const unsigned char *__restrict__ w,
                                  const unsigned char *__restrict__ xq,
                                  float *__restrict__ out,
                                  int rows, int cols, int tokens, int ttype) {
    const Q35QuantInfo info = q35_quant_info(ttype);
    const int block_bytes = info.block_bytes;
    const int nblk = cols / 32;
    const int lane = threadIdx.x & 31;
    const int warp = threadIdx.x >> 5;
    const int r0 = blockIdx.x * 128 + warp * 16;
    const int t0 = blockIdx.y * 8;
    if (r0 >= rows) return;

    const bool need_sumy = (ttype == Q35_TT_Q4_1 || ttype == Q35_TT_Q5_1);

    __shared__ int8_t sA[8 * 16 * 32];   // per-warp weight int8 (16 rows x 32 k)
    __shared__ int8_t sB[8 * 8 * 32];    // per-warp activation int8 (8 tok x 32 k)
    __shared__ int sSumy[8 * 8];         // per-warp per-token sum a8 (Q4_1/Q5_1 only)
    int8_t *sAw = sA + warp * 16 * 32;
    int8_t *sBw = sB + warp * 8 * 32;
    int *sumy = sSumy + warp * 8;

    float acc[4] = {0.0f, 0.0f, 0.0f, 0.0f};
    for (int b = 0; b < nblk; ++b) {
        // Weight int8: per-type dequant (q35_mma_w8), 16 rows x 32 k.
        for (int idx = lane; idx < 16 * 32; idx += 32) {
            int mp = idx / 32, k = idx % 32;
            const unsigned char *wb =
                w + (size_t)(r0 + mp) * (size_t)nblk * block_bytes + (size_t)b * block_bytes;
            sAw[mp * 32 + k] = (int8_t)q35_mma_w8(ttype, wb, k);
        }
        // Activation int8 (Q8_0 layout for all six types).
        for (int idx = lane; idx < 8 * 32; idx += 32) {
            int np = idx / 32, k = idx % 32;
            sBw[np * 32 + k] = (t0 + np < tokens)
                ? (int8_t)xq[(size_t)(t0 + np) * (size_t)nblk * 34 + (size_t)b * 34 + 2 + k] : 0;
        }
        __syncwarp();
        // Per-token sumy = sum a8, only for the _1 types.  Lanes 0..7 each sum
        // one token's 32 int8; the scale loop below reads all eight.
        if (need_sumy) {
            if (lane < 8) {
                int s = 0;
                for (int k = 0; k < 32; ++k) s += (int)(int8_t)sBw[lane * 32 + k];
                sumy[lane] = s;
            }
            __syncwarp();
        }
        q35_tile<16, 8, int> A_frag, D;
        q35_tile<8, 8, int> B_frag;
        q35_mma::load_ldmatrix(A_frag, (const int *)sAw, 8);
        q35_mma::load_ldmatrix(B_frag, (const int *)sBw, 8);
        q35_mma::mma(D, A_frag, B_frag);
#pragma unroll
        for (int l = 0; l < 4; ++l) {
            const int mp = q35_tile<16, 8, int>::get_i(l);
            const int np = q35_tile<16, 8, int>::get_j(l);
            const unsigned char *wb =
                w + (size_t)(r0 + mp) * (size_t)nblk * block_bytes + (size_t)b * block_bytes;
            const float dx = q35_f16_at(wb, 0);
            const float dy = q35_f16_at(xq + (size_t)(t0 + np) * (size_t)nblk * 34 + (size_t)b * 34, 0);
            acc[l] = __fadd_rn(acc[l], __fmul_rn(__fmul_rn(dx, dy), (float)D.x[l]));
            if (need_sumy) {
                const float mx = q35_f16_at(wb, 2);
                acc[l] = __fadd_rn(acc[l], __fmul_rn(__fmul_rn(mx, dy), (float)sumy[np]));
            }
        }
    }
#pragma unroll
    for (int l = 0; l < 4; ++l) {
        const int mp = q35_tile<16, 8, int>::get_i(l);
        const int np = q35_tile<16, 8, int>::get_j(l);
        if (t0 + np < tokens)
            out[(size_t)(t0 + np) * (size_t)rows + (size_t)(r0 + mp)] = acc[l];
    }
}

// ---------------------------------------------------------------------------
// Tensor-core (mma.sync) prefill kernel, 256-element K-family weight
// (Q2_K, Q3_K, Q4_K, Q5_K, Q6_K).
//
// Same shape as the 32-element kernel, but the dot's scale is per 16 elements
// (16 groups per 256-element super-block), so the k32 mma result cannot be
// used: it mixes two scale groups.  k16 mma is used instead -- one mma per
// group, so D.x[l] is the EXACT int32 dot of one 16-element group, folded into
// isum[l] with that group's `scv[g]` in ascending g order.  That is precisely
// q35_dot_scaled16's association (int-domain, exact), so the integer part is
// bit-identical by construction.
//
// Dequant: the per-type aux8/unpack helpers are reused VERBATIM from
// vecdotq.cuh (q35_q2k_aux8, q35_q3k_aux8 + q35_q3k_unpack_scales,
// q35_q6k_aux8, q35_unpack_scale_min), so the int8 weights mma multiplies are
// the same bytes the dp4a path multiplies.  Q5_K needs the +16 high-bit term,
// and its `sm[]` is the low-nibble scale; Q6_K carries its scales as int8.
//
// Float tail: Q2_K/Q4_K/Q5_K add a `- dmin*summs` term (summs from Q8_K bsums);
// each type's expression and rounding order mirrors its _acc function exactly.
// ---------------------------------------------------------------------------

// Dequantize a WHOLE 256-element row of the native block at `wb` into w8[256]
// (int8), reusing the same aux8 helpers the dp4a path multiplies, so the int8
// values mma sees are the same bytes.  Q5_K's +16 high-bit term and Q6_K's
// -32 bias live in those helpers.
__device__ __forceinline__ void q35_mma_k_aux8(int ttype, const unsigned char *wb,
                                               int8_t *w8) {
    signed char *s = (signed char *)w8;
    switch (ttype) {
    case Q35_TT_Q2_K:
        q35_q2k_aux8(wb, s);
        break;
    case Q35_TT_Q3_K:
        q35_q3k_aux8(wb, s);
        break;
    case Q35_TT_Q4_K: {
        const unsigned char *q4 = wb + 16;
        for (int c = 0; c < 4; ++c) {
            for (int l = 0; l < 32; ++l) s[c * 64 + l] = (signed char)(q4[c * 32 + l] & 0x0F);
            for (int l = 0; l < 32; ++l) s[c * 64 + 32 + l] = (signed char)(q4[c * 32 + l] >> 4);
        }
        break;
    }
    case Q35_TT_Q5_K: {
        const unsigned char *q4 = wb + 48;
        const unsigned char *hm = wb + 16;
        int m = 1;
        for (int c = 0; c < 4; ++c) {
            for (int l = 0; l < 32; ++l)
                s[c * 64 + l] = (signed char)((q4[c * 32 + l] & 0x0F) + ((hm[l] & m) ? 16 : 0));
            for (int l = 0; l < 32; ++l)
                s[c * 64 + 32 + l] =
                    (signed char)((q4[c * 32 + l] >> 4) + ((hm[l] & (m << 1)) ? 16 : 0));
            m <<= 2;
        }
        break;
    }
    case Q35_TT_Q6_K:
        q35_q6k_aux8(wb, s);
        break;
    default:
        break;
    }
}

// Per-type: unpack the 16 per-16-element scales (scv[g]) and, where the type
// has a min term, the per-32-element mins (sm[8 + j/2]).
__device__ __forceinline__ void q35_mma_k_scales(int ttype, const unsigned char *wb,
                                                 int *scv, unsigned char *smv) {
    switch (ttype) {
    case Q35_TT_Q2_K: {
        const unsigned char *sc = wb;
        for (int g = 0; g < 16; ++g) scv[g] = (int)(sc[g] & 0x0F);
        break;
    }
    case Q35_TT_Q3_K: {
        signed char sc[16];
        q35_q3k_unpack_scales(wb + 96, sc);
        for (int g = 0; g < 16; ++g) scv[g] = (int)sc[g] - 32;
        break;
    }
    case Q35_TT_Q4_K:
    case Q35_TT_Q5_K: {
        unsigned char sm[16];
        q35_unpack_scale_min(wb + 4, sm);
        for (int g = 0; g < 16; ++g) scv[g] = (int)sm[g / 2];
        for (int j = 0; j < 16; ++j) smv[j] = sm[j];
        break;
    }
    case Q35_TT_Q6_K: {
        const signed char *sc = (const signed char *)(wb + 192);
        for (int g = 0; g < 16; ++g) scv[g] = (int)sc[g];
        break;
    }
    default:
        break;
    }
}

__global__ void q35_k_mmq_mma_k256(const unsigned char *__restrict__ w,
                                   const unsigned char *__restrict__ xq,
                                   float *__restrict__ out,
                                   int rows, int cols, int tokens, int ttype) {
    const Q35QuantInfo info = q35_quant_info(ttype);
    const int block_bytes = info.block_bytes;
    const int nblk = cols / 256;
    const int lane = threadIdx.x & 31;
    const int warp = threadIdx.x >> 5;
    const int r0 = blockIdx.x * 128 + warp * 16;
    const int t0 = blockIdx.y * 8;
    // NOTE: no early return here.  This kernel uses __syncthreads() (the
    // activation tile sB is shared by all warps of the block), so every thread
    // must reach the barrier -- a `return` for out-of-range warps would hang
    // the block.  Rows past `rows` are guarded at the store instead.
    const bool row_live = (r0 < rows);

    // Whole 256-element block staged once per warp (16 rows x 256 int8 = 4 KB)
    // plus one shared 8-token activation tile (2 KB).  34 KB/block total, so
    // the block still fits two-per-SM.  The per-16 scale makes a group
    // self-contained but staging the whole block means ONE __syncwarp() per
    // block instead of 16, and the per-row scales are unpacked once instead of
    // once per group -- that was most of the cost in the first version.
    // Row stride is 256+16, NOT 256: a power-of-two shared stride makes every
    // lane in a warp hit the same bank when `ldmatrix` walks rows, and the IQ
    // kernel (which uses the same staging shape) has carried this padding since
    // it was written -- see SA_STRIDE in q35_k_mmq_mma_iq.  16 B keeps the
    // ldmatrix alignment requirement (rows start 16 B apart, which is what
    // `load_ldmatrix` needs) while breaking the bank alias.
    constexpr int SA_STRIDE = 256 + 16;
    __shared__ int8_t sA[8 * 16 * SA_STRIDE];
    // Same bank-alias reasoning as SA_STRIDE above; `sB` is read by
    // `load_generic` walking 8 token rows 256 B apart.
    constexpr int SB_STRIDE = 256 + 16;
    __shared__ int8_t sB[8 * SB_STRIDE];
    int8_t *sAw = sA + warp * 16 * SA_STRIDE;
    __shared__ int sScv[8 * 16 * 16];    // per-warp, per-row group scales
    int *scv_w = sScv + warp * 16 * 16;

    float acc[4] = {0.0f, 0.0f, 0.0f, 0.0f};

    for (int b = 0; b < nblk; ++b) {
        const unsigned char *wb0 =
            w + (size_t)r0 * (size_t)nblk * block_bytes + (size_t)b * block_bytes;
        // Stage the whole 256-element block for all 16 rows.  Dequantize a
        // WHOLE row at a time through the same aux8 helper the dp4a path uses
        // (16 lanes -> 16 rows, one call each) instead of element-by-element
        // with a per-element type switch: the switch was the bulk of the cost
        // in the first version (measured 27B Q4_K_XL prefill 21.7 vs 22.5 for
        // thread-per-row; whole-row aux8 is what the dp4a path already does).
        if (row_live && lane < 16) {
            const unsigned char *wb = wb0 + (size_t)lane * (size_t)nblk * block_bytes;
            q35_mma_k_aux8(ttype, wb, sAw + lane * SA_STRIDE);
            int scv[16];
            unsigned char smv[16];
            q35_mma_k_scales(ttype, wb, scv, smv);
            for (int g = 0; g < 16; ++g) scv_w[lane * 16 + g] = scv[g];
        }
        // Activation: one shared 8-token x 256 int8 tile.  Byte copies, NOT
        // int4: a Q8_K block is 292 bytes, so 292 % 16 == 4 and neither a token
        // row nor a block start is 16-byte aligned (an int4 walk faults with
        // "misaligned address").  This copy is 2 KB next to the weight dequant.
        // sB is shared by all 8 warps of the block, so the barrier here must be
        // __syncthreads(): with __syncwarp() a warp could read a tile another
        // warp was still writing (that was a real bug -- Q6_K had 24 718
        // mismatched rows until this became a block-wide barrier).
        // Spread over the whole block, not `warp == 0 && lane < 8`: the old
        // shape had 8 lanes each byte-copying 256 elements while the other 248
        // threads waited at the barrier, and it runs once per (super-block,
        // token-tile), so its cost scaled exactly linearly with `tokens`.  Same
        // fix and same measurement as the IQ path in
        // `q35_k_mmq_mma_iq` (tokens 8 -> 128 was 222 us -> 1799 us before it).
        {
            const int tid = (int)threadIdx.x;
            const int trow = tid / 32;      // which of the 8 tokens
            const int tcol = tid % 32;      // byte column, 8 per step
            if (trow < 8) {
                int8_t *dst = sB + trow * SB_STRIDE + tcol * 8;
                if (t0 + trow < tokens) {
                    const unsigned char *row =
                        xq + (size_t)(t0 + trow) * (size_t)nblk * 292 + (size_t)b * 292 + 4;
                    const unsigned char *src = row + tcol * 8;
                    for (int k = 0; k < 8; ++k) dst[k] = (int8_t)src[k];
                } else {
                    for (int k = 0; k < 8; ++k) dst[k] = 0;
                }
            }
        }
        __syncthreads();

        int isum[4] = {0, 0, 0, 0};
        for (int g = 0; g < 16; ++g) {
            q35_tile<16, 4, int> A_frag;
            q35_tile<8, 4, int> B_frag;
            q35_tile<16, 8, int> D;
            q35_mma::load_ldmatrix(A_frag, (const int *)(sAw + g * 16), SA_STRIDE / 4);
            q35_mma::load_generic(B_frag, (const int *)(sB + g * 16), SB_STRIDE / 4);
            q35_mma::mma(D, A_frag, B_frag);
            // Row of each lane's fragment register, so the scale lookup is
            // scv_w[row][g] (per-row, per-group).
#pragma unroll
            for (int l = 0; l < 4; ++l) {
                const int mp = q35_tile<16, 8, int>::get_i(l);
                isum[l] += scv_w[mp * 16 + g] * D.x[l];
            }
        }

#pragma unroll
        for (int l = 0; l < 4; ++l) {
            const int np = q35_tile<16, 8, int>::get_j(l);
            const int mp = q35_tile<16, 8, int>::get_i(l);
            if (t0 + np >= tokens || r0 + mp >= rows) continue;
            const unsigned char *wb = wb0 + (size_t)mp * (size_t)nblk * block_bytes;
            const unsigned char *yb =
                xq + (size_t)(t0 + np) * (size_t)nblk * 292 + (size_t)b * 292;
            const float dy = q35_q8k_d(yb);
            switch (ttype) {
            case Q35_TT_Q2_K: {
                const unsigned char *sc = wb;
                int sms = 0;
                for (int j = 0; j < 16; ++j) sms += q35_q8k_bsum(yb, j) * (int)(sc[j] >> 4);
                const float dall = __fmul_rn(dy, q35_f16_at(wb, 80));
                const float dmin = __fmul_rn(dy, q35_f16_at(wb, 82));
                acc[l] = __fadd_rn(acc[l], __fmul_rn(dall, (float)isum[l]));
                acc[l] = __fadd_rn(acc[l], __fmul_rn(-dmin, (float)sms));
                break;
            }
            case Q35_TT_Q3_K:
                acc[l] = __fadd_rn(acc[l],
                    __fmul_rn(__fmul_rn(q35_f16_at(wb, 108), dy), (float)isum[l]));
                break;
            case Q35_TT_Q4_K:
            case Q35_TT_Q5_K: {
                unsigned char sm[16];
                q35_unpack_scale_min(wb + 4, sm);
                int sms = 0;
                for (int j = 0; j < 16; ++j) sms += q35_q8k_bsum(yb, j) * (int)sm[8 + j / 2];
                const float d = __fmul_rn(dy, q35_f16_at(wb, 0));
                const float dmin = __fmul_rn(dy, q35_f16_at(wb, 2));
                acc[l] = __fadd_rn(acc[l], __fmul_rn(d, (float)isum[l]));
                acc[l] = __fadd_rn(acc[l], __fmul_rn(-dmin, (float)sms));
                break;
            }
            case Q35_TT_Q6_K:
                acc[l] = __fadd_rn(acc[l],
                    __fmul_rn(__fmul_rn(q35_f16_at(wb, 208), dy), (float)isum[l]));
                break;
            default:
                break;
            }
        }
    }

#pragma unroll
    for (int l = 0; l < 4; ++l) {
        const int mp = q35_tile<16, 8, int>::get_i(l);
        const int np = q35_tile<16, 8, int>::get_j(l);
        if (t0 + np < tokens && r0 + mp < rows)
            out[(size_t)(t0 + np) * (size_t)rows + (size_t)(r0 + mp)] = acc[l];
    }
}

static cudaError_t mmq_mma_launch(const unsigned char *d_w, const unsigned char *d_xq,
                                  float *d_out, int rows, int cols, int tokens, int ttype,
                                  cudaStream_t stream) {
    const dim3 grid((unsigned)((rows + 127) / 128), (unsigned)((tokens + 7) / 8));
    q35_k_mmq_mma_q32<<<grid, Q35_MMQ_THREADS, 0, stream>>>(d_w, d_xq, d_out, rows, cols, tokens, ttype);
    return cudaGetLastError();
}

static cudaError_t mmq_mma_k_launch(const unsigned char *d_w, const unsigned char *d_xq,
                                    float *d_out, int rows, int cols, int tokens, int ttype,
                                    cudaStream_t stream) {
    const dim3 grid((unsigned)((rows + 127) / 128), (unsigned)((tokens + 7) / 8));
    q35_k_mmq_mma_k256<<<grid, Q35_MMQ_THREADS, 0, stream>>>(d_w, d_xq, d_out, rows, cols, tokens, ttype);
    return cudaGetLastError();
}

// ---------------------------------------------------------------------------
// Tensor-core (mma.sync) prefill kernel, IQ family (256-element super-block).
//
// WHY THIS EXISTS
// ---------------
// The IQ types had no mma kernel: q35_launch_mul_mat_vec_tile returned -1 for
// them, so prefill fell back to the caller's thread-per-row kernel -- 35.8
// tok/s on 27B IQ3_XXS (66-token prefill, measured).  The dp4a *tile* kernel
// was measured slower still (19.3 tok/s) because it carries a 32-lane serial
// fold (`__shared__ float part[WARPS][TPERPASS][32]` + a lane-0 loop of 32
// `__fadd_rn`) whose only justification was reproducing q35_dot_row's exact
// b-ascending float association.
//
// That constraint is lifted (docs/cuda-elementwise-ulp.md: "bit-exact
// reproduction of CPU token ids is NOT the acceptance criterion"; kernel-level
// acceptance is MVCHECK's relative tolerance 2e-2).  With free association the
// reduction folds into the mma D fragment: mma.sync already yields the int32
// dot of a whole 32-element group in D.x[0..3] -- four int32 per lane, no
// cross-lane step at all.  There is no serial fold here to remove.
//
// SHAPE / ASSOCIATION
// -------------------
// One block = 8 warps = 128 weight rows (16 per warp) x 8 tokens.  Each warp
// stages 16 rows x 256 int8 of dequantized weight, the block shares one
// 8-token x 256 int8 activation tile.  IQ3_XXS carries one scale per 32
// elements (8 per super-block) and the k32 mma gives exactly the int32 dot of
// one 32-element group, so `isum[row] += ls[g] * D.x[l]` accumulates the 8
// groups in the integer domain (exact), and the float scaling is applied once
// at the end -- the same association q35_vec_dot_iq3_xxs_q8_k_acc uses
// (`0.25f * d * bsum`).
//
// DEQUANT IS THE REAL COST, SO IT IS DONE ONCE
// --------------------------------------------
// IQ3_XXS dequant is a grid lookup (iq3xxs_grid: uint32 -> 4 int8) plus a 7-bit
// sign expansion per 4-lane group.  Doing that per mma (16 groups per
// super-block, per row) would dominate.  Instead each lane unpacks a WHOLE row
// (16 lanes = 16 rows) into shared memory using the same __vsub4(g ^ m, m)
// byte-negation the dp4a path uses, and ldmatrix feeds mma from the staged
// int8: one __syncthreads() per super-block.
// ---------------------------------------------------------------------------

// Unpack one 32-element IQ3_XXS group from q3[0..7] and the 28-bit sign field
// into out[0..7], each holding 4 int8 (elements 0..3, 4..7, ... 28..31).
// Byte-identical to the dp4a path's __vsub4(iq3xxs_grid[..] ^ s, s).
__device__ __forceinline__ void q35_iq3xxs_group32(const unsigned char *q3,
                                                   unsigned int aux7, int out[8]) {
#pragma unroll
    for (int l = 0; l < 4; ++l) {
        const unsigned int sgn = q35_iq_signs8((aux7 >> (7 * l)) & 127u);
        const int g1 = (int)iq3xxs_grid[q3[2 * l + 0]];
        const int g2 = (int)iq3xxs_grid[q3[2 * l + 1]];
        const int s0 = __vcmpne4(sgn & 0x08040201u, 0u);
        const int s1 = __vcmpne4(sgn & 0x80402010u, 0u);
        out[2 * l + 0] = __vsub4(g1 ^ s0, s0);
        out[2 * l + 1] = __vsub4(g2 ^ s1, s1);
    }
}

// Per-32-element scale of IQ3_XXS: ls = 2*(aux32>>28) + 1 (always positive).
__device__ __forceinline__ int q35_iq3xxs_ls(const unsigned char *gas, int g) {
    return 2 * (int)(q35_ld_u32(gas + 4 * g) >> 28) + 1;
}

// Trait structs for the generic IQ kernel: plain constants plus a
// __device__-callable unpacker.  `group` is a plain (non-template) static
// function, which keeps nvcc's template substitution happy.
struct Q35Iq3Xxs {
    static constexpr int block_bytes = 98;
    // One scale per 32 elements, so both halves of a super-block group share it
    // and the k32 mma (which spans a whole 32-element group) applies it once.
    static constexpr bool scale16 = false;
    static constexpr int out_scale_q = 1;
    static constexpr int out_scale_den = 4;   // 0.25f
    __device__ static void group(const unsigned char *wb, int g, int p[8], int ls[2]);
    /// The super-block's f16 `d`; overridden only by IQ1_M.
    __device__ static float d_of(const unsigned char *wb) { return q35_f16_at(wb, 0); }
};
struct Q35Iq3S {
    static constexpr int block_bytes = 110;
    static constexpr bool scale16 = false;
    static constexpr int out_scale_q = 1;
    static constexpr int out_scale_den = 1;   // 1.0f
    __device__ static void group(const unsigned char *wb, int g, int p[8], int ls[2]);
    /// The super-block's f16 `d`; overridden only by IQ1_M.
    __device__ static float d_of(const unsigned char *wb) { return q35_f16_at(wb, 0); }
};
struct Q35Iq2S {
    static constexpr int block_bytes = 82;
    // 16-element granularity, like IQ4_XS: scales[g] packs ls1 in the low nibble
    // and ls2 in the high nibble, so the two halves of a group carry DIFFERENT
    // scales.  See Q35Iq2S::group -- this is not a style choice, a k32 mma here
    // mixes the two and MVCHECK catches it as mismatched=5209727 (the whole
    // matrix), because every element with a non-equal nibble pair is wrong.
    static constexpr bool scale16 = true;
    static constexpr int out_scale_q = 1;
    static constexpr int out_scale_den = 8;   // 0.125f
    __device__ static void group(const unsigned char *wb, int g, int p[8], int ls[2]);
    /// The super-block's f16 `d`; overridden only by IQ1_M.
    __device__ static float d_of(const unsigned char *wb) { return q35_f16_at(wb, 0); }
};
// IQ4_XS: 136 B = d@0 (f16) | scales_h@2 (u16) | scales_l[4]@4 | qs[128]@8.
// A plain 4-bit codebook lookup (kvalues_iq4nl, i.e. the iq4nl alias) with NO
// sign table, ONE scale per 32 elements (so ls[0] == ls[1] here), and -- unlike
// IQ3_XXS -- a NEGATIVE-capable ls (the -32 bias), which is why the integer
// domain is the only sane place to accumulate it.
// Source of truth: ggml-quants.c dequantize_row_iq4_xs (line 2744) and
// ggml-common.h block_iq4_xs (line 454).  dp4a twin in this repo is
// vecdotq.cuh:710 q35_vec_dot_iq4_xs_q8_k_acc.
struct Q35Iq4Xs {
    static constexpr int block_bytes = 136;
    // THE ONE 16-element-granularity type: ls[0] != ls[1] (dequantize_row_iq4_xs,
    // ggml-quants.c:2744), so a k32 mma would mix two different scales in one
    // D fragment and the group() contract below forbids that.  This is why the
    // kernel keeps a k16 path at all.
    static constexpr bool scale16 = true;
    static constexpr int out_scale_q = 1;
    static constexpr int out_scale_den = 1;   // 1.0f
    __device__ static void group(const unsigned char *wb, int g, int p[8], int ls[2]);
    /// The super-block's f16 `d`; overridden only by IQ1_M.
    __device__ static float d_of(const unsigned char *wb) { return q35_f16_at(wb, 0); }
};
// IQ2_XXS: 66 B = d@0 (f16) | qs[32] @2, read as 8 x { a[0..3] (4 grid indices)
// + aux1 (u32) }.  iq2xxs_grid is u64 (8 uint8 magnitudes), the sign table is
// ksigns_iq2xs (7-bit codes -- NOT keven_signs), and the scale is ONE per 32
// elements from the top nibble of aux1, so -- like IQ3_XXS -- both halves of
// the group share ls[0] == ls[1] and the kernel runs k16 mma with 0.25f.
// Source of truth: ggml-quants.c dequantize_row_iq2_xxs (line 2500) and
// ggml-common.h block_iq2_xxs (line 240).  dp4a twin in this repo is
// vecdotq.cuh:498 q35_vec_dot_iq2_xxs_q8_k_acc.
struct Q35Iq2Xxs {
    static constexpr int block_bytes = 66;
    static constexpr bool scale16 = false;
    static constexpr int out_scale_q = 1;
    // 1/8, NOT IQ3_XXS's 1/4 -- see the derivation in group() below:
    //   db = d * (0.5f + hi) * 0.25f  and  ls = 2*hi + 1
    //   => db = d * ls * 0.125f, because 0.25*(0.5+hi) == 0.125*(2*hi+1).
    // The `2*hi` inside ls already absorbs a factor of two, so reusing
    // out_scale_den = 4 here doubles every output (MVCHECK: dev == 2 * cpu on
    // every row, mismatched=163839).
    static constexpr int out_scale_den = 8;   // 0.125f
    __device__ static void group(const unsigned char *wb, int g, int p[8], int ls[2]);
    /// The super-block's f16 `d`; overridden only by IQ1_M.
    __device__ static float d_of(const unsigned char *wb) { return q35_f16_at(wb, 0); }
};
// IQ2_XS: 74 B = d@0 (f16) | qs[32] @2, read as 16 uint16 | scales[8] @66.
// The uint16 at qs[2*ib32+l] is NOT two nibbles: bits 0..8 index iq2xs_grid
// (512 entries, u64 of 8 uint8 magnitudes), bits 9..15 are the 7-bit sign code.
// That is why this cannot share IQ2_XXS's unpacker even though the grid and the
// sign table are the same ones.
//
// Scale granularity is 16, NOT 32: dequantize_row_iq2_xs (ggml-quants.c:2528)
// builds db[0] from the low nibble of scales[ib32] and db[1] from the high
// nibble, and `l/2` selects between them, so elements 0..15 and 16..31 of a
// 32-element group carry DIFFERENT floats.  A k32 mma would mix them -- the
// same trap Q35Iq2S documents.  Hence scale16 = true.
//
// Source of truth: ggml-quants.c dequantize_row_iq2_xs (line 2517) and
// ggml-common.h block_iq2_xs (line 389).
struct Q35Iq2Xs {
    static constexpr int block_bytes = 74;
    static constexpr bool scale16 = true;
    static constexpr int out_scale_q = 1;
    // db = d * (0.5f + nib) * 0.25f, with ls = 2*nib + 1 the odd scale the
    // integer domain carries: 0.25*(0.5+nib) == 0.125*(2*nib+1), so the float
    // tail is 0.125f exactly (same identity as IQ2_XXS, so the same denominator).
    static constexpr int out_scale_den = 8;   // 0.125f
    __device__ static void group(const unsigned char *wb, int g, int p[8], int ls[2]);
    /// The super-block's f16 `d`; overridden only by IQ1_M.
    __device__ static float d_of(const unsigned char *wb) { return q35_f16_at(wb, 0); }
};

// IQ1_S: 50 B = d@0 (f16) | qs[32] @2 | qh[16] @34 (u16 x 16).
// Each group of 32 elements owns 4 `qs` bytes and one `qh` u16:
//   grid index = qs[l] | (((qh >> 3l) & 7) << 8)     -> iq1s_grid, 2048 entries
//   scale      = 2*((qh >> 12) & 7) + 1              -> ONE per 32 elements
//   delta      = (qh & 0x8000) ? -0.125 : +0.125
//
// Two things make this the awkward one:
//  * `iq1s_grid` holds SIGNED int8 (-1/0/+1), unlike the unsigned magnitude
//    tables the other traits expand.  It therefore needs no sign fold at all --
//    the u64's two halves drop straight into p[] as four int8 lanes each, which
//    is exactly what vecdotq.cuh:1200 does with dp4a.
//  * the +-0.125 delta is NOT an integer multiple of the scale.  The obvious
//    route is a separate float term built from q8_K's per-16 sums (that is what
//    q35_vec_dot_iq1_s_q8_k_part does, and what this kernel used to do).  The
//    cheaper route, and the one taken here, is to scale the whole grid by EIGHT
//    inside group(): 8*grid is +-8, so 8*grid +- 1 stays inside int8, the mma
//    computes an exact integer dot of the delta-included value, and
//    out_scale_den = 8 divides it back out (a power of two, so exact).  The
//    float tail is then a plain `d * isum`, and the bsum walk disappears.
//
// Source of truth: ggml-quants.c dequantize_row_iq1_s (line 2651) and
// ggml-common.h block_iq1_s (line 425).
struct Q35Iq1S {
    static constexpr int block_bytes = 50;
    static constexpr bool scale16 = false;
    static constexpr int out_scale_q = 1;
    // 8: group() folded delta into p[] at 1/8 scale (see the comment there), so
    // the integer dot comes out eight times too large.  8 is a power of two, so
    // this division is exact.
    static constexpr int out_scale_den = 8;
    __device__ static void group(const unsigned char *wb, int g, int p[8], int ls[2]);
    __device__ static float d_of(const unsigned char *wb) { return q35_f16_at(wb, 0); }
};

// IQ1_M: 56 B = qs[32] @0 | qh[16] @32 | scales[8] @48.
//   d          = an f16 whose bits are spread over the four u16s' nibbles
//                (dequantize_row_iq1_m:2686) -- the ONE type whose `d` is not
//                a stored f16, hence the `d_of` override.
//   grid index = qs[4g+l] | 0x700 bits from qh[2g + (l>>1)], shifted 8 for
//                even l and 4 for odd l
//   scale      = TWO 6-bit values per group: scales[g>>1] bits
//                (g&1)*6+0..2 for l=0,1 and (g&1)*6+3..5 for l=2,3
//   delta      = +-0.125 per EIGHT elements: qh[2g+(l>>1)] bit 0x08 for even l,
//                0x80 for odd l
//
// The l -> qh group mapping is `l >> 1`, NOT `l & 1`: l=0,1 read qh[2g] and
// l=2,3 read qh[2g+1], while the SHIFT alternates within each pair
// (`8 - 4*(l%2)`, i.e. 8 for even l).  Swapping the two turns the grid index
// into a rotation of the real one, which still looks like a plausible int8
// operand and still passes every single-component probe.  That confusion, plus
// an oracle that repeated it, is what cost this trait several rounds: both
// sides agreed with each other and both disagreed with ggml.
//
// delta is per EIGHT elements here, but it costs nothing extra: all eight bytes
// of one 8-element chunk share a sign, so the same x8 trick IQ1_S uses applies l
// by l.  (An earlier version built per-8 activation sums in shared memory to add
// delta as a separate term -- q8_K's bsums are per-16 and cannot express a per-8
// sum -- and it read the wrong token row.  Folding delta into p[] removes the
// need for that array entirely, and with it the whole class of bug.)
struct Q35Iq1M {
    static constexpr int block_bytes = 56;
    static constexpr bool scale16 = true;
    static constexpr int out_scale_q = 1;
    // 8: group() folded delta into p[] at 1/8 scale, exactly as IQ1_S does.
    static constexpr int out_scale_den = 8;
    __device__ static void group(const unsigned char *wb, int g, int p[8], int ls[2]);
    __device__ static float d_of(const unsigned char *wb);
};


/// Pack four int8 values into one int32 (little-endian byte order: a is byte 0).
/// Only used where the integer operand is NOT the raw table entry -- IQ1_S and
/// IQ1_M scale their grids by 8 to fold the +-0.125 delta in, which has to be
/// done per element.  Every other trait gets its p[] from the table directly,
/// via the sign tricks documented on each of them.
__device__ __forceinline__ int q35_iq_pack8(int a, int b, int c, int d) {
    return (int)(((unsigned int)(a) & 0xFFu) | (((unsigned int)(b) & 0xFFu) << 8) |
                 (((unsigned int)(c) & 0xFFu) << 16) | (((unsigned int)(d) & 0xFFu) << 24));
}

// Per-type unpacker for the generic IQ tensor-core kernel below: decompose one
// 32-element group of a 256-element super-block into 8 int32 of int8 and
// return that group's integer scale `ls`.
__device__ void Q35Iq3Xxs::group(const unsigned char *wb, int g, int p[8], int ls[2]) {
    // gas[4g] packs four 7-bit sign codes plus the 4-bit group scale.
    // IQ3_XXS has ONE scale per 32 elements, so both halves share it.
    const unsigned int aux32 = q35_ld_u32(wb + 66 + 4 * g);
    q35_iq3xxs_group32(wb + 2 + 8 * g, aux32 & 0x0FFFFFFFu, p);
    ls[0] = ls[1] = 2 * (int)(aux32 >> 28) + 1;
}

__device__ void Q35Iq3S::group(const unsigned char *wb, int g, int p[8], int ls[2]) {
    // IQ3_S differs from IQ3_XXS in three ways, all handled here: the grid
    // index needs a 9th bit from qh[g], the sign field is a plain 8-bit mask
    // per 4-lane chunk (signs[4g+l] * 0x01010101, no 7-bit expansion), and the
    // group scale is a nibble of scales[g>>1] rather than the top of gas.
    const unsigned char *qs = wb + 2 + 8 * g;
    const unsigned int qh = wb[66 + g];
    const unsigned char *signs = wb + 74 + 4 * g;
#pragma unroll
    for (int l = 0; l < 4; ++l) {
        const unsigned int i1 = (unsigned int)qs[2 * l + 0] | ((qh << (8 - 2 * l)) & 256u);
        const unsigned int i2 = (unsigned int)qs[2 * l + 1] | ((qh << (7 - 2 * l)) & 256u);
        const unsigned int sgn = (unsigned int)signs[l] * 0x01010101u;
        const int s0 = __vcmpne4(sgn & 0x08040201u, 0u);
        const int s1 = __vcmpne4(sgn & 0x80402010u, 0u);
        p[2 * l + 0] = __vsub4((int)iq3s_grid[i1] ^ s0, s0);
        p[2 * l + 1] = __vsub4((int)iq3s_grid[i2] ^ s1, s1);
    }
    // One scale per 32 elements, so both halves share it.
    const unsigned char sc = wb[106 + (g >> 1)];
    const unsigned int nib = (g & 1) ? (unsigned int)(sc >> 4) : (unsigned int)(sc & 0xF);
    ls[0] = ls[1] = 2 * (int)nib + 1;
}

// BLOCK_BYTES: weight block size of the type; GROUP(i) -> q35_iq_group_*.

__device__ void Q35Iq2S::group(const unsigned char *wb, int g, int p[8], int ls[2]) {
    // 82 B = d@0 | qs[32]@2 | qh[8]@66 | scales[8]@74 | signs[32]@34
    // (`signs` is NOT a separate field: the reference reads it at qs+32, i.e.
    // the block's byte 34, and each of its bytes packs 8 sign bits.)
    // iq2s_grid is u64: 8 uint8 of magnitude per entry, so one 8-value chunk
    // needs no packing tricks -- just two uint32 halves.
    const unsigned char *qs = wb + 2 + 4 * g;
    const unsigned int qh = wb[66 + g];
    const unsigned char *signs = wb + 34 + 4 * g;
    for (int l = 0; l < 4; ++l) {
        const unsigned int idx = (unsigned int)qs[l] | ((qh << (8 - 2 * l)) & 0x300u);
        const unsigned long long gv = iq2s_grid[idx];
        const int g1 = (int)(gv & 0xFFFFFFFFull);
        const int g2 = (int)(gv >> 32);
        const unsigned int sgn = (unsigned int)signs[l] * 0x01010101u;
        const int s0 = __vcmpne4(sgn & 0x08040201u, 0u);
        const int s1 = __vcmpne4(sgn & 0x80402010u, 0u);
        p[2 * l + 0] = __vsub4(g1 ^ s0, s0);
        p[2 * l + 1] = __vsub4(g2 ^ s1, s1);
    }
    // IQ2_S is the one IQ type with 16-element scale granularity: scales[g]
    // holds ls1 in its low nibble (elements 0..15 of the group) and ls2 in its
    // high nibble (elements 16..31).  That is exactly why this kernel uses k16
    // mma: a k32 D fragment would span both scales and could only apply one.
    const unsigned char sc = wb[74 + g];
    ls[0] = 2 * (int)(sc & 0xF) + 1;
    ls[1] = 2 * (int)(sc >> 4) + 1;
}

__device__ void Q35Iq4Xs::group(const unsigned char *wb, int g, int p[8], int ls[2]) {
    // 136 B = d@0 (f16) | scales_h@2 (u16) | scales_l[4]@4 | qs[128]@8.
    // dequantize_row_iq4_xs (ggml-quants.c:2744) and the dp4a twin
    // q35_vec_dot_iq4_xs_q8_k_acc (vecdotq.cuh:710):
    //   ls = (scales_l[ib/2] nibble) | (((h >> 2*ib) & 3) << 4),  SCALE = ls-32
    //   y[j+0] = d*(ls-32)*kvalues_iq4nl[qs[j] & 0xf]
    //   y[j+16]= d*(ls-32)*kvalues_iq4nl[qs[j] >>  4]
    // Two differences from IQ3_XXS/IQ3_S drive this implementation:
    //  1. The 6-bit ls is biased by -32 and may be negative or zero -- IQ4_XS
    //     is the only IQ type here with a signed per-group scale.  It is kept
    //     in the INTEGER domain (`isum += ls * D.x[l]`), which is exact either
    //     way; d is factored out to the final float scale.
    //  2. kvalues_iq4nl is `int8_t` spanning -127..113, NOT an unsigned grid
    //     byte.  Packing it with plain `<< 8` sign-extends each value into its
    //     neighbours (`(int)(int8_t)-127 == 0xFFFFFF81`), so every byte must be
    //     masked to 0x00FF before the shift.  (Measured before the mask: every
    //     element of every row wrong, MVCHECK mismatched=161750.)
    const unsigned short h = q35_ld_u16(wb + 2);
    const unsigned char *sc_l = wb + 4;
    const unsigned char *qs = wb + 8 + 16 * g;
    const unsigned char sc = sc_l[g >> 1];
    const unsigned int nib = (g & 1) ? (unsigned int)(sc >> 4) : (unsigned int)(sc & 0xF);
    const int l = (int)nib | (int)(((unsigned int)h >> (2 * g)) & 3u) << 4;
    // Low nibbles -> elements 0..15, high nibbles -> 16..31 (the two y[] rows).
    // p[l] holds elements 4l..4l+3.
#pragma unroll
    for (int q = 0; q < 4; ++q) {
        const int lo = ((int)(unsigned char)kvalues_iq4nl[qs[4 * q + 0] & 0xF]) |
                       (((int)(unsigned char)kvalues_iq4nl[qs[4 * q + 1] & 0xF]) << 8) |
                       (((int)(unsigned char)kvalues_iq4nl[qs[4 * q + 2] & 0xF]) << 16) |
                       (((int)(unsigned char)kvalues_iq4nl[qs[4 * q + 3] & 0xF]) << 24);
        const int hi = ((int)(unsigned char)kvalues_iq4nl[qs[4 * q + 0] >> 4]) |
                       (((int)(unsigned char)kvalues_iq4nl[qs[4 * q + 1] >> 4]) << 8) |
                       (((int)(unsigned char)kvalues_iq4nl[qs[4 * q + 2] >> 4]) << 16) |
                       (((int)(unsigned char)kvalues_iq4nl[qs[4 * q + 3] >> 4]) << 24);
        p[q] = lo;
        p[4 + q] = hi;
    }
    ls[0] = ls[1] = l - 32;
}

__device__ void Q35Iq2Xxs::group(const unsigned char *wb, int g, int p[8], int ls[2]) {
    // 66 B = d@0 (f16) | qs[32]@2.  Group g owns the 8 bytes at qs + 8*g:
    //   a[0..3] = four iq2xxs_grid indices, +4 = aux1
    //     aux1 bits 7*l+0..7*l+6 -> 7-bit sign code for the l-th chunk of 8
    //     aux1 bits 28..31       -> (db/d factor) - 1, i.e. ls = 2*hi + 1
    // Element order is l*8 + j (dequantize_row_iq2_xxs:2500), i.e. the grid's
    // 8 magnitude bytes are CONSECUTIVE elements, whereas p[q] holds elements
    // 4q..4q+3.  So each 8-byte grid straddles two p[] ints: the low uint32
    // (bytes j=0..3) lands in p[2l], the high uint32 (j=4..7) in p[2l+1].
    // Signs: exactly the IQ3_XXS trick -- q35_iq_signs8 expands the 7-bit code
    // into one 0xFF/0x00 byte per element, __vcmpne4 turns the byte selection
    // into a 0xFFFFFFFF/0 int, and __vsub4(g ^ m, m) negates the marked bytes
    // (bit-identical to the scalar `signs & kmask_iq2xs[j] ? -1 : 1`, and free
    // of the branch loop).  iq2xxs_grid bytes are unsigned 0..255 magnitudes,
    // so they are taken with & 0xFF and fed to __vcmpne4/__vsub4 as unsigned
    // byte lanes -- no int8 sign extension to pollute neighbouring bytes (the
    // IQ4_XS bug was the opposite: a genuinely signed int8 table).
    const unsigned int aux1 = q35_ld_u32(wb + 2 + 8 * g + 4);
#pragma unroll
    for (int l = 0; l < 4; ++l) {
        const unsigned int sgn = q35_iq_signs8((aux1 >> (7 * l)) & 127u);
        const unsigned long long grid = iq2xxs_grid[wb[2 + 8 * g + l]];
        const int g1 = (int)(unsigned int)(grid & 0xFFFFFFFFull);
        const int g2 = (int)(unsigned int)(grid >> 32);
        const int s0 = __vcmpne4(sgn & 0x08040201u, 0u);
        const int s1 = __vcmpne4(sgn & 0x80402010u, 0u);
        p[2 * l + 0] = __vsub4(g1 ^ s0, s0);
        p[2 * l + 1] = __vsub4(g2 ^ s1, s1);
    }
    // ONE (always positive, odd) scale per 32 elements, so both halves share
    // it.  ls = 2*(aux1>>28) + 1.  With db = d*(0.5f + hi)*0.25f that leaves
    // a float factor of 0.125f (out_scale_den = 8), NOT IQ3_XXS's 0.25f:
    // 0.25*(0.5+hi) == 0.125*(2*hi+1) exactly (both are powers of two times an
    // odd integer, so the identity is bit-exact in f32).
    ls[0] = ls[1] = 2 * (int)(aux1 >> 28) + 1;
}

__device__ void Q35Iq2Xs::group(const unsigned char *wb, int g, int p[8], int ls[2]) {
    // Group g owns the four uint16 at qs + 8*g (2 bytes each).  Element order is
    // l*8 + j (dequantize_row_iq2_xs:2530), so each 8-byte grid straddles two
    // p[] ints exactly like IQ2_XXS: low four magnitude bytes -> p[2l], high
    // four -> p[2l+1].
#pragma unroll
    for (int l = 0; l < 4; ++l) {
        const unsigned int q = (unsigned int)wb[2 + 8 * g + 2 * l]
                             | ((unsigned int)wb[2 + 8 * g + 2 * l + 1] << 8);
        const unsigned int sgn = q35_iq_signs8((q >> 9) & 127u);
        const unsigned long long grid = iq2xs_grid[q & 511u];
        const int g1 = (int)(unsigned int)(grid & 0xFFFFFFFFull);
        const int g2 = (int)(unsigned int)(grid >> 32);
        const int s0 = __vcmpne4(sgn & 0x08040201u, 0u);
        const int s1 = __vcmpne4(sgn & 0x80402010u, 0u);
        p[2 * l + 0] = __vsub4(g1 ^ s0, s0);
        p[2 * l + 1] = __vsub4(g2 ^ s1, s1);
    }
    // 16-element granularity: scales[g] low nibble -> elements 0..15, high
    // nibble -> 16..31.  ls must be the same odd integer the float side divides
    // out (2*nib + 1), and the kernel applies ls[0] to the first group of 16 and
    // ls[1] to the second when scale16 is set.
    const unsigned char sc = wb[66 + g];
    ls[0] = 2 * (int)(sc & 0x0F) + 1;
    ls[1] = 2 * (int)(sc >> 4) + 1;
}

__device__ float Q35Iq1M::d_of(const unsigned char *wb) {
    // dequantize_row_iq1_m:2686 -- iq1m_scale_t.u16 is assembled from four
    // nibbles: sc[0]'s HIGH nibble is bits 12..15, sc[1]'s high nibble 8..11,
    // sc[2]'s high nibble 4..7, and sc[3]'s LOW nibble 0..3.
    const unsigned int sc0 = q35_ld_u16(wb + 48), sc1 = q35_ld_u16(wb + 50);
    const unsigned int sc2 = q35_ld_u16(wb + 52), sc3 = q35_ld_u16(wb + 54);
    const unsigned int u = (sc0 >> 12) | ((sc1 >> 8) & 0x00F0u) |
                           ((sc2 >> 4) & 0x0F00u) | (sc3 & 0xF000u);
    const unsigned char b[2] = { (unsigned char)(u & 0xFFu), (unsigned char)(u >> 8) };
    return q35_ld_scale(b);
}

__device__ void Q35Iq1M::group(const unsigned char *wb, int g, int p[8], int ls[2]) {
    // Group g owns qs + 4*g and the TWO qh bytes at 32 + 2*g (one per 16
    // elements).  Element order is l*8 + j, so the grid's low uint32 is p[2l]
    // and its high uint32 p[2l+1] -- the same shape as IQ1_S.
    // delta is +-0.125 PER EIGHT ELEMENTS here (one sign per l), so all eight
    // bytes of p[2l], p[2l+1] carry the SAME sign -- which is exactly the shape
    // IQ1_S's x8 trick handles, one l at a time.  That is why this trait needs
    // no per-8 activation sums: the delta rides in the integer operand.
    const unsigned int qh0 = wb[32 + 2 * g + 0];
    const unsigned int qh1 = wb[32 + 2 * g + 1];
#pragma unroll
    for (int l = 0; l < 4; ++l) {
        // l = 0,1 take the 9th..11th index bits from qh0, l = 2,3 from qh1, and
        // within each pair the even l shifts by 8, the odd by 4
        // (dequantize_row_iq1_m:2694-2697).
        // l = 0,1 come from qh[0]; l = 2,3 from qh[1] -- `l >> 1`, NOT `l & 1`.
        // The shift alternates within each PAIR: even l by 8, odd l by 4
        // (ggml's `8 - 4*(l%2)`).  Confusing the two groups reads the other
        // qh byte and yields a plausible-looking but rotated grid, which is
        // why every single-block probe "passed" while the engine did not.
        const unsigned int hi = ((l < 2) ? qh0 : qh1) << ((l & 1) ? 4 : 8);
        const unsigned int idx = (unsigned int)wb[4 * g + l] | (hi & 0x700u);
        const unsigned long long grid = iq1s_grid[idx & 2047u];
        // sign bits: qh[2g]&0x08 (l=0), &0x80 (l=1), qh[2g+1]&0x08 (l=2),
        // &0x80 (l=3) -- dequantize_row_iq1_m:2699.
        const int dsign = (((l < 2) ? qh0 : qh1) & ((l & 1) ? 0x80u : 0x08u)) ? -1 : 1;
        p[2 * l + 0] = q35_iq_pack8((int)(int8_t)(grid & 0xFFull) * 8 + dsign,
                                    (int)(int8_t)((grid >> 8) & 0xFFull) * 8 + dsign,
                                    (int)(int8_t)((grid >> 16) & 0xFFull) * 8 + dsign,
                                    (int)(int8_t)((grid >> 24) & 0xFFull) * 8 + dsign);
        p[2 * l + 1] = q35_iq_pack8((int)(int8_t)((grid >> 32) & 0xFFull) * 8 + dsign,
                                    (int)(int8_t)((grid >> 40) & 0xFFull) * 8 + dsign,
                                    (int)(int8_t)((grid >> 48) & 0xFFull) * 8 + dsign,
                                    (int)(int8_t)((grid >> 56) & 0xFFull) * 8 + dsign);
    }
    // TWO 6-bit scales per group (dequantize_row_iq1_m:2690).  The integer
    // domain carries (2*hi + 1), the same odd-scale convention as IQ1_S.
    const unsigned int scw = q35_ld_u16(wb + 48 + 2 * (g >> 1));
    const unsigned int base = (g & 1) ? 6u : 0u;
    ls[0] = 2 * (int)((scw >> (base + 0)) & 7u) + 1;
    ls[1] = 2 * (int)((scw >> (base + 3)) & 7u) + 1;
}

__device__ void Q35Iq1S::group(const unsigned char *wb, int g, int p[8], int ls[2]) {
    // 50 B = d@0 (f16) | qs[32]@2 | qh[16]@34.  Group g owns qs + 4*g and the
    // u16 at qh + 2*g.  Element order is l*8 + j, so the grid's two uint32
    // halves land in p[2l] and p[2l+1] -- same shape as IQ2_XXS, except the
    // table is already int8 so there is no sign expansion to do.
    const unsigned int qh = (unsigned int)wb[34 + 2 * g]
                          | ((unsigned int)wb[34 + 2 * g + 1] << 8);
    // The quantized value is ls*(grid + delta) with delta = +-0.125, and 0.125
    // is not an integer, so the naive route is a separate float term.  Instead
    // scale the whole grid by EIGHT and carry delta as +-1 in the integer
    // domain:  8*grid + 8*delta = 8*grid +- 1, exact, and out_scale_den = 8
    // gives the factor back at the end.  8*grid is +-8 so 8*grid +- 1 stays
    // inside int8's range -- the mma needs int8, and +-9 fits.
    //
    // This removes the delta from the float tail entirely: no bsum walk, no
    // per-8 partial sums, no separate code path.  It also means the operand is
    // no longer the raw table entry, so the eight bytes are packed here instead
    // of copying the u64's two halves.
    const int dsign = (qh & 0x8000u) ? -1 : 1;
#pragma unroll
    for (int l = 0; l < 4; ++l) {
        const unsigned int idx = (unsigned int)wb[2 + 4 * g + l]
                               | (((qh >> (3 * l)) & 7u) << 8);
        const unsigned long long grid = iq1s_grid[idx & 2047u];
        p[2 * l + 0] = q35_iq_pack8((int)(int8_t)(grid & 0xFFull) * 8 + dsign,
                                    (int)(int8_t)((grid >> 8) & 0xFFull) * 8 + dsign,
                                    (int)(int8_t)((grid >> 16) & 0xFFull) * 8 + dsign,
                                    (int)(int8_t)((grid >> 24) & 0xFFull) * 8 + dsign);
        p[2 * l + 1] = q35_iq_pack8((int)(int8_t)((grid >> 32) & 0xFFull) * 8 + dsign,
                                    (int)(int8_t)((grid >> 40) & 0xFFull) * 8 + dsign,
                                    (int)(int8_t)((grid >> 48) & 0xFFull) * 8 + dsign,
                                    (int)(int8_t)((grid >> 56) & 0xFFull) * 8 + dsign);
    }
    // ONE scale per 32 elements, so both halves share it.
    ls[0] = ls[1] = 2 * (int)((qh >> 12) & 7u) + 1;
}

// NWARPS is a template argument, not a constant, so the launch shape can be
// shrunk to buy OCCUPANCY.  The tile is 16 rows per warp, so the row tile per
// block is NWARPS*16: 8 warps -> 128 rows (the original shape) gives the 27B
// ffn (rows=5120) only (5120+127)/128 = 40 blocks for 114 SMs, i.e. 35% of the
// device idle.  Halving the warps doubles the block count and halves the shared
// memory (`sA` is NWARPS*16*256 int8), so the small-row matrices finally fill
// the GPU.  Crucially this needs no atomics and no extra reduction: every block
// still owns whole rows and the whole K range, so the store path stays a plain
// assignment and the result is deterministic.  (An earlier attempt split the K
// range across grid.y and accumulated with atomicAdd; it measured fine but
// reordered the float sums, so the token stream diverged from the --MMQ=0
// reference even though MVCHECK passed.  Warp count is the lever that does not
// touch the summation order.)
template <class T, int NWARPS>
__global__ void q35_k_mmq_mma_iq(const unsigned char *__restrict__ w,
                                 const unsigned char *__restrict__ xq,
                                 float *__restrict__ out,
                                 int rows, int cols, int tokens, int njsplit) {
    // grid.y splits the TOKEN dimension: block y owns token tiles
    // [y*tpt, (y+1)*tpt), so each block covers fewer tokens and the weight slice
    // is re-read njsplit times.  njsplit == 1 is exactly the old grid.y == 1
    // shape.  `tokens` stays the global count -- xq and out are indexed by
    // global token id, only the loop bounds use the slice.
    //
    // This re-reads weights on purpose and it is a 15% win; the launcher's
    // q35_iq_warps-adjacent comment explains why (latency-bound, needs streams
    // in flight, not bytes saved).
    constexpr int block_bytes = T::block_bytes;
    constexpr int OUT_SCALE_Q = T::out_scale_q;
    constexpr int OUT_SCALE_DEN = T::out_scale_den;
    const int nblk = cols / 256;
    const int lane = threadIdx.x & 31;
    const int warp = threadIdx.x >> 5;
    const int r0 = blockIdx.x * (NWARPS * 16) + warp * 16;
    const int b_lo = 0;
    const int b_hi = nblk;
    // No early return: this kernel uses __syncthreads() (sB is shared by all
    // warps), so every thread must reach the barrier.  Out-of-range rows are
    // skipped at the dequant and at the store.
    const bool row_live = (r0 < rows);

    // 16 rows x 256 int8 = 4 KB per warp (32 KB/block) + one shared 8x256
    // activation tile (2 KB) + per-row group scales.
    //
    // BANK CONFLICTS: every stride here was a power of two, which is the worst
    // possible case on a 32-bank / 4-byte-wide shared memory:
    //   * sA rows were 256 int8 == 64 int, and 64 % 32 == 0, so ALL 16 rows of a
    //     warp started on bank 0.  The producer writes `dst[g*8+i]` with
    //     dst = row*64 int, so lanes 0..31 (16 rows x 2 group-pairs) all touched
    //     the same bank at the same instruction -- a full 32-way conflict on
    //     every one of the 32 int stores per lane.
    //   * the same 64-int stride is what load_ldmatrix reads with, so the
    //     consumer hit it again, 16 times per super-block.
    //   * sLs rows were 16 int, and 16 % 32 == 0, so the 16 lsum lanes collided too.
    // Padding by 4 int (16 B) makes consecutive rows start 4 banks apart, which
    // is a coprime-ish rotation over the 16 rows of a warp: 16*4 = 64 == 0 mod
    // 32 only every 8 rows, so at most 2 rows share a bank instead of 16.
    // (llama.cpp solves the same problem with its `sram_stride` "+4" padding,
    // mmq.cuh:131 -- `static_assert(stride % 8 == 4)` in every layout.)
    constexpr int SA_STRIDE = 256 + 16;       // int8 per row, padded
    constexpr int LS_STRIDE = 16 + 4;         // int per row, padded
    // sB is the activation tile: 8 token rows x 256 int8, read by load_generic
    // as (token * SB_STRIDE + col).  It was the one stride left at a power of
    // two: 256 B == 64 banks * 4 B, so 256 % 128 == 0 and ALL 8 token rows
    // started on the same bank -- an 8-way conflict on every one of the 16
    // loads per super-block, per token tile.  Same fix as sA: pad by 16 B so
    // consecutive rows rotate 4 banks apart.
    constexpr int SB_STRIDE = 256 + 16;       // int8 per row, padded
    __shared__ int8_t sA[NWARPS * 16 * SA_STRIDE];
    __shared__ int8_t sB[8 * SB_STRIDE];
    int8_t *sAw = sA + warp * 16 * SA_STRIDE;
    __shared__ int sLs[NWARPS * 16 * LS_STRIDE];  // per-warp per-row ls[16]: one per 16-element half
    int *ls_w = sLs + warp * 16 * LS_STRIDE;

    // One accumulator per token-tile.  With grid.y == 1 the whole token range
    // is handled inside the kernel, so the 128-row weight slice is dequantized
    // from HBM ONCE per super-block b and hoisted over the token-tile loop
    // below.  (Previously grid.y == tokens/8 re-read the same 128-row weight
    // once per 8-token tile -- the prefill bandwidth cliff.)  acc[t] holds the
    // (16 rows x 8 tokens) D-fragment sum for token-tile t, accumulated across
    // all super-blocks b.
    constexpr int MAX_TILES = Q35_MMQ_MAX_TILES;
    const int ntiles = (tokens + 7) / 8;
    const int tpt = (ntiles + njsplit - 1) / njsplit;
    const int t_lo = (int)blockIdx.y * tpt;
    const int t_hi = min(ntiles, t_lo + tpt);
    // The launcher guarantees tpt <= MAX_TILES (it raises njsplit until it is),
    // so a violation means that contract broke.  Clamp instead of trapping: the
    // kernel still has to reach its __syncthreads() unconditionally, and a trap
    // here would turn a wrong-but-running shape into a dead device.
    const int t_end = min(t_hi, t_lo + MAX_TILES);
    float acc[MAX_TILES][4];
#pragma unroll
    for (int t = 0; t < MAX_TILES; ++t)
#pragma unroll
        for (int l = 0; l < 4; ++l) acc[t][l] = 0.0f;
    // acc[] is indexed by LOCAL tile (0..t_hi-t_lo-1): the global tile index is
    // t_lo+ti.

    for (int b = b_lo; b < b_hi; ++b) {
        const unsigned char *wb0 =
            w + (size_t)r0 * (size_t)nblk * block_bytes + (size_t)b * block_bytes;

        // Weight dequant, spread over the WHOLE warp so all 32 lanes unpack:
        // 16 rows x 2 group-pairs = 32 lanes, each taking 4 of the block's 8
        // groups.  Unpacking a whole row on one lane left 31 of 32 lanes idle
        // and was the kernel's real cost.
        //
        // Staged ONCE per b (hoisted over the token-tile loop below).
        //
        // NOTE on the row spread: this makes a warp touch 16 rows, whose HBM
        // addresses are 16*nblk*block_bytes apart (~104 KB for the 27B ffn,
        // where nblk*98 = 6664 B).  Narrowing it to 4 rows per warp (8 lanes
        // per row, which is what llama.cpp's loader does) was measured and made
        // NO difference (170.25 vs 170.29 tok/s), so the L2/coalescer already
        // absorbs the row stride and the address spread is not the bottleneck.
        // Do not "fix" it again without a measurement.
        // Lane assignment: 4 lanes per row (llama.cpp's threads_per_row for
        // IQ3_XXS), so one warp covers 8 rows and needs TWO passes for the
        // 16-row warp tile.  Lane -> (row, group pair) is unchanged in meaning:
        // row = lane>>2 + pass*8, gp = lane&3 picks one of the 8 groups.
        // Each lane therefore unpacks 2 groups instead of 4 -- the whole point:
        // the per-lane IQ3_XXS unpack (~4 grid lookups + 2 sign expansions) is
        // cut in half and the row's K is spread over 4 lanes rather than 2.
        for (int pass = 0; pass < 2; ++pass) {
            const int row = ((lane >> 2) + 8 * pass);          // 0..15
            const int gp = lane & 3;                           // 0..3
            const bool live = row_live && (r0 + row < rows);
            if (live) {
                const unsigned char *wb = wb0 + (size_t)row * (size_t)nblk * block_bytes;
                int *dst = (int *)(sAw + row * SA_STRIDE);
#pragma unroll
                for (int k = 0; k < 2; ++k) {
                    // 4 lanes x 2 groups = groups 0..7 with NO gaps: 2*gp+k.
                    // (4*gp+k with only k<2 covers 0,1,4,5 and silently drops
                    // the other half -- MVCHECK caught exactly that.)
                    const int g = 2 * gp + k;
                    int p[8];
                    int ls[2];
                    T::group(wb, g, p, ls);
                    ls_w[row * LS_STRIDE + g * 2 + 0] = ls[0];
                    ls_w[row * LS_STRIDE + g * 2 + 1] = ls[1];
                    for (int i = 0; i < 8; ++i) dst[g * 8 + i] = p[i];
                }
            } else {
                for (int k = 0; k < 2; ++k) {
                    const int g = 2 * gp + k;
                    ls_w[row * LS_STRIDE + g * 2 + 0] = 1;
                    ls_w[row * LS_STRIDE + g * 2 + 1] = 1;
                }
            }
        }

        // A-fragment REGISTERS, hoisted over the token-tile loop exactly like
        // sAw: the weight is the same for every token, so re-running
        // `load_ldmatrix` inside the token loop was pure waste -- 8 ldmatrix per
        // super-block per token tile, against llama.cpp doing it once
        // (mmq-vec-dot.cuh:226 `tile_A A[ntx][MMQ_TILE_NE_K/QI8_0]` filled
        // before the j loop).  With ntx == 1 our warp tile is a single 16-row
        // mini-tile, so the whole A tile is 8 fragments (k32 path) or 16 (k16).
        //
        // Registers: 8 * tile<16,8,int> == 8 * 4 int = 32 registers for k32,
        // 64 for the k16 path.  Both fit -- and they are the same registers the
        // per-tile loads were already occupying, just live across the loop.
        q35_tile<16, 8, int> Af[8];    // k32 path (scale16 == false)
        q35_tile<16, 4, int> Af16[16]; // k16 path (scale16 == true)
        if (!T::scale16) {
#pragma unroll
            for (int g = 0; g < 8; ++g)
                q35_mma::load_ldmatrix(Af[g], (const int *)sAw + g * 8, SA_STRIDE / 4);
        } else {
#pragma unroll
            for (int h = 0; h < 16; ++h)
                q35_mma::load_ldmatrix(Af16[h], (const int *)sAw + h * 4, SA_STRIDE / 4);
        }

        // Token tiles: sAw/ls_w are staged above and hoisted over this loop;
        // only the 8-token activation tile is re-staged per tile.
        // Constant-bounded (MAX_TILES) so the compiler keeps acc[t][l] in
        // REGISTERS, not local memory: a runtime-bounded loop would spill the
        // 16x4 float accumulator to local memory and the prefill matvec becomes
        // local-memory bound (measured: that is why the grid.y==1 hoist alone
        // gave only ~1.6x -- the weight-bandwidth win was eaten by the spill).
        // `tb >= tokens` guards the partial last tile; tokens is launch-uniform
        // (not thread-dependent), so the __syncthreads() below stay uniform.
#pragma unroll
        for (int ti = 0; ti < MAX_TILES; ++ti) {
            const int t = t_lo + ti;
            if (t >= t_end) break;
            const int tb = t * 8;
            // Activation tile: Q8_K qs[256] at byte 4 (292-byte block).
            // Byte copies, not int4: 292 % 16 == 4, so neither a token row nor a
            // block start is 16-byte aligned.
            //
            // Previously `warp == 0 && lane < 8` with a 256-iteration inner
            // loop: 8 lanes copied 256 bytes each while the other 248 threads
            // sat at the barrier.  That serial loop runs once per (super-block,
            // token-tile), so its cost scales EXACTLY linearly with `tokens` --
            // which is the shape of the measured prefill cliff (tokens 8 -> 128
            // went 222 us -> 1799 us, 8.1x for 16x tokens).  Spread the same
            // 8x256 bytes over the whole block instead: `threadIdx.x` picks one
            // of the 8 token rows and one of the 32 byte-columns, so all 256
            // threads copy 8 bytes each and the barrier wait is one loop
            // iteration.
            // 2048 bytes over whatever the block actually has: stride by
            // blockDim.x so this is correct for ANY warp count (NWARPS is a
            // template parameter, so it is 128..256 threads here).  Binding one
            // row per warp -- `trow = threadIdx.x / 32` -- silently left rows 4..7
            // unwritten at 4 warps, which is exactly the bug this loop had.
            {
                // 16 bytes per thread as four 4-byte words, not 8 single bytes.
                //
                // Where the width came from: with the unpack AND the mma removed
                // entirely the kernel still took 1.276 of its 1.334 ms, so this
                // copy -- not HBM -- was the bottleneck.  (A temporary
                // memory-only branch was used to establish that and has been
                // removed; the numbers are IQ3_XXS ffn 17408x5120 at 525
                // tokens.)
                //
                // 4-byte words, not int2/int4: a q8_K block is 292 = 4*73 bytes,
                // so `b*292 + 4` is only 4-byte aligned for odd b and anything
                // wider faults with `misaligned address` (measured, both int4 and
                // int2).  4 B is the widest alignment the format guarantees and
                // it is already 4x the old per-instruction width.
                //
                // Staging the whole block's activation once per super-block b
                // (a 34 KB sB) was tried and is WORSE -- it pushes shared past
                // 48 KiB, halves occupancy and lands at 1.247 ms against this
                // 1.231.  MAX_TILES=16 with the small sB stays the optimum.
                const int tid = (int)threadIdx.x;
                const int nt  = (int)blockDim.x;
                constexpr int CHUNK = 16;
                for (int i = tid * CHUNK; i < 8 * 256; i += nt * CHUNK) {
                    const int trow = i >> 8;               // which of the 8 tokens
                    const int tcol = i & 255;              // byte offset in the row
                    int *dst = (int *)(sB + trow * SB_STRIDE + tcol);
                    if (tb + trow < tokens) {
                        const unsigned char *row =
                            xq + (size_t)(tb + trow) * (size_t)nblk * 292 + (size_t)b * 292 + 4;
                        const int *src = (const int *)(row + tcol);
                        dst[0] = src[0];
                        dst[1] = src[1];
                        dst[2] = src[2];
                        dst[3] = src[3];
                    } else {
                        dst[0] = dst[1] = dst[2] = dst[3] = 0;
                    }
                }
            }
            __syncthreads();
            // Two mma paths, chosen by the scale granularity of T:
            //
            //  * scale16 == false (IQ3_XXS / IQ3_S / IQ2_XXS): ONE scale
            //    per 32 elements, so `mma.sync.m16n8k32` gives the exact int32
            //    dot of a whole group in D.x[l] and the scale applies once.
            //    8 mma + 8 ldmatrix + 8 B-loads per super-block instead of 16.
            //  * scale16 == true (IQ4_XS / IQ2_S): ls[0] != ls[1] -- IQ2_S packs
            //    both nibbles into scales[g] -- so the two halves must stay
            //    separate.  Keep the k16 loop for those two.
            //
            // The old comment here claimed "the tensor cores are nowhere near
            // the bottleneck ... so this is free".  They are not free: at 16
            // k16 mma per super-block the kernel measured 42 TFLOPS against
            // llama.cpp's 127 on the same shape, and halving the instruction
            // count is part of closing that.  Association is unchanged -- both
            // paths accumulate `isum[l] += ls * D.x[l]` in the integer domain,
            // so the result is the same exact int32 dot either way.
            int isum[4] = {0, 0, 0, 0};
            if (!T::scale16) {
#pragma unroll
                for (int g = 0; g < 8; ++g) {
                    q35_tile<8, 8, int> B_frag;
                    q35_tile<16, 8, int> D;
                    q35_mma::load_generic(B_frag, (const int *)sB + g * 8, SB_STRIDE / 4);
                    q35_mma::mma(D, Af[g], B_frag);
#pragma unroll
                    for (int l = 0; l < 4; ++l) {
                        const int mp = q35_tile<16, 8, int>::get_i(l);
                        isum[l] += ls_w[mp * LS_STRIDE + g * 2] * D.x[l];
                    }
                }
            } else {
#pragma unroll
                for (int h = 0; h < 16; ++h) {
                    q35_tile<8, 4, int> B_frag;
                    q35_tile<16, 8, int> D;
                    // A_frag comes from the hoisted Af[] above; only B (the
                    // activation tile) is re-read per token tile.
                    q35_mma::load_generic(B_frag, (const int *)sB + h * 4, SB_STRIDE / 4);
                    q35_mma::mma(D, Af16[h], B_frag);
                    const int g = h >> 1;               // 32-element group
                    const int half = h & 1;
#pragma unroll
                    for (int l = 0; l < 4; ++l) {
                        const int mp = q35_tile<16, 8, int>::get_i(l);
                        // ls2 holds the second half's scale when the type has 16-element
                        // granularity; for 32-element types both entries are equal.
                        isum[l] += (half ? ls_w[mp * LS_STRIDE + g * 2 + 1] : ls_w[mp * LS_STRIDE + g * 2]) * D.x[l];
                    }
                }
            }
            __syncthreads();   // sB is re-staged next token-tile

#pragma unroll
            for (int l = 0; l < 4; ++l) {
                const int mp = q35_tile<16, 8, int>::get_i(l);
                const int np = q35_tile<16, 8, int>::get_j(l);
                if (tb + np >= tokens || r0 + mp >= rows) continue;
                const unsigned char *wb = wb0 + (size_t)mp * (size_t)nblk * block_bytes;
                const unsigned char *yb =
                    xq + (size_t)(tb + np) * (size_t)nblk * 292 + (size_t)b * 292;
                const float d = __fmul_rn(T::d_of(wb), q35_q8k_d(yb));
                // T::out_scale_q * OUT_SCALE_DEN: an integer template argument
                // times a power-of-two denominator, because C++ forbids a
                // floating-point template parameter.  Both factors are exact.
                acc[ti][l] = __fadd_rn(acc[ti][l], __fmul_rn(
                    __fmul_rn(d, (float)isum[l]), (float)OUT_SCALE_Q / (float)OUT_SCALE_DEN));
            }
        }
    }

#pragma unroll
    for (int ti = 0; ti < MAX_TILES; ++ti) {
        const int t = t_lo + ti;
        if (t >= t_end) break;
        const int tb = t * 8;
#pragma unroll
        for (int l = 0; l < 4; ++l) {
            const int mp = q35_tile<16, 8, int>::get_i(l);
            const int np = q35_tile<16, 8, int>::get_j(l);
            if (tb + np < tokens && r0 + mp < rows) {
                out[(size_t)(tb + np) * (size_t)rows + (size_t)(r0 + mp)] = acc[ti][l];
            }
        }
    }
}


// ---------------------------------------------------------------------------
// IQ prefill kernel, llama.cpp's dp4a shape (NOT the tensor-core one above).
//
// Why this exists: on sm_89 llama.cpp does NOT use an mma for the IQ dot.  Its
// `ggml_cuda_mmq_load_tiles_iq3_xxs` unpacks the weights into shared memory as
// plain int8 rows and `ggml_cuda_mmq_vec_dot_q8_0_q8_1_dp4a` folds those rows
// against a q8_1 activation tile with __dp4a -- the tensor cores are only used
// by the *other* `_mma` vec_dot, which pairs with a different sram_stride (84
// vs 65, mmq.cuh:131 vs mmq-vec-dot.cuh:134) and which the IQ3_XXS dispatch
// does not select.  Our kernel went the other way (mma IS the dot), and it
// measures 71.9 TFLOPS on the 27B ffn shape against llama.cpp's 180.6.
//
// This kernel is a faithful port of that dp4a shape so the two can be
// compared on identical shapes.  Layout, verbatim from the source:
//
//   tile_x (weights), I=128 rows, sram_stride = 2*32+1 = 65 int per row:
//     [i*65 + 0 .. i*65+63]  256 int8 = the whole 256-element super-block,
//                            written as 8 lanes/row * QR3_XXS=4 * 2 ints
//                            (mmq-load-tiles.cuh: threads_per_row = 8)
//     [i*65 + 64]            unused padding (the +1 of the stride)
//   tile_df (weight scales), a SEPARATE array -- llama.cpp does not append them
//   to the rows, it writes `x_df[i*(MMQ_TILE_NE_K/4) + i/4 + kqsx]`:
//     [i*8 + i/4 + 0..7]     one float per 32-int8 group, the `i/4` being the
//                            +16 B/four-rows padding that keeps the 8 floats of
//                            a row off one shared-memory bank
//   tile_y (activations), J=128 rows x MMQ_TILE_Y_K = 4 + 32 = 36 int, matching
//   llama.cpp's `block_q8_1_mmq`, whose two accessors are
//   `y_qs = (const int*)y + 4` and `y_df = (const float*)y`:
//     [j*36 + 0..3]          4 float scales, one per 32 int8 (`d4[4]`)
//     [j*36 + 4..35]         128 int8 of the half of the super-block being dotted
//
// Our activations arrive as q8_K (292 B: one f32 scale + 256 int8 + bsums), not
// as `block_q8_1_mmq`, so the staging loop transposes them.  That is exact: q8_K
// carries ONE scale for all 256 values, so all four `d4` entries are that same
// number, and the 128-int8 halves are qs[k00h*4 ..] selected by `k00h` in INTS.
//
// The float scale is folded ONCE per (i, kqsx) here -- `d*ls/4` with `ls =
// 2*aux+1`, algebraically llama.cpp's `(ls_4bit*d + d/2)/2` -- i.e. exactly what
// the mma kernel applies as `d * isum * OUT_SCALE_Q/DEN`.  Both forms are exact
// in f32 for the values involved (ls is a small odd integer, d is an f16), and
// the accumulator is a plain float sum in ascending k order in both, so the two
// kernels agree bit for bit (that is what MVCHECK checks).
//
// Cost note (measured, 27B IQ3_XXS, 525-token prefill).  This shape runs at
// 425-426 tok/s against the mma kernel's 435-436, i.e. ~2.4% SLOWER, and
// PROF-GPU puts the per-launch time at 3052 us against 2912.
//
// Two rounds of measurement went into that number, and the second one is the
// interesting result:
//
//   * First version transposed q8_K -> `block_q8_1_mmq` on the accumulator
//     side of EVERY launch: 421-424 tok/s, 3103 us/launch.
//   * Now the quantizer emits the mmq reshape in the same pass it produces
//     q8_K (q35_launch_quantize_q8_K_mmq), so the tile staging here is
//     llama.cpp's contiguous copy: 425-426 tok/s, 3052 us/launch.
//
// So removing the redundant transpose bought 1.7% and the remaining gap is NOT
// memory traffic -- it is the dp4a dot itself being slower than the mma path on
// this shape.  llama.cpp picks dp4a on sm_89 only because ITS tile geometry,
// register budget and activation layout are all tuned around it; porting the
// instruction choice without the rest does not reproduce the result.  Do not
// read this kernel's presence as "dp4a is faster".
//
// Every index in the inner loop is a compile-time constant: I, J, nwarps and
// the k/VDR steps are template/constexpr, so `sum[...]` below is 64 scalars in
// REGISTERS.  This is the property the mma kernel lacks -- see the comment at
// the top of it.
// ---------------------------------------------------------------------------

#define Q35_DP4A_I      128   // weight rows per block (llama.cpp's I for IQ3_XXS)
#define Q35_DP4A_J      128   // tokens per block     (llama.cpp's J)
#define Q35_DP4A_THREADS 256  // 8 warps, llama.cpp's nthreads
#define Q35_DP4A_NE_K    32   // MMQ_TILE_NE_K: k per vec_dot call
#define Q35_DP4A_VDR      8   // VDR_Q8_0_Q8_1_MMQ: int per dp4a chunk
#define Q35_DP4A_SX      (2 * Q35_DP4A_NE_K + 1)          // 65 int per weight row
// llama.cpp keeps the weight scales in their own array (`x_df`), NOT at the end
// of each weight row: `x_df[i*(MMQ_TILE_NE_K/4) + i/4 + kqsx]`.  The `i/4` is
// +16 B of padding per 4 rows, which is what keeps the 8 scale floats of one
// row off the same shared-memory bank.  One float per 32 int8 (one 8-int group).
#define Q35_DP4A_SDF     8
#define Q35_DP4A_DF(i, g) ((i) * Q35_DP4A_SDF + (i) / 4 + (g))
// llama.cpp's `block_q8_1_mmq` layout, which is what vec_dot_q8_0_q8_1_dp4a
// reads: 4 float scales (one per 32 int8) followed by 128 int8 (32 int).
// `y_qs = (const int*)y + 4` and `y_df = (const float*)y` are exactly this.
#define Q35_DP4A_SY      (4 + Q35_DP4A_NE_K)              // 36: 4 scale ints + 32 int8 ints
#define Q35_DP4A_NW      (Q35_DP4A_THREADS / 32)          // 8 warps
#define Q35_DP4A_NSUM    (Q35_DP4A_J * Q35_DP4A_I / Q35_DP4A_THREADS)  // 64

// One thread's 64 accumulators, indexed by (j0/nwarps, i0/warp_size) exactly as
// llama.cpp indexes its `sum[]`.  I and J are compile-time, so both loops
// unroll to constant indices.
struct Q35Dp4aSums {
    float v[Q35_DP4A_NSUM];
    __device__ __forceinline__ float &at(int jq, int iq) {
        return v[jq * (Q35_DP4A_I / 32) + iq];
    }
};

// vec_dot_q8_0_q8_1_impl<float, 8>: 8 dp4a over 32 int8, scaled by (d8_0 * d8_1).
// v and u are `int*` into the int8 rows; we read them as 8 ints.
__device__ __forceinline__ float q35_dp4a_dot32(const int *v, const int *u,
                                               float d8_0, float d8_1) {
    int sumi = 0;
#pragma unroll
    for (int i = 0; i < Q35_DP4A_VDR; ++i) sumi = q35_dp4a(v[i], u[i], sumi);
    return __fmul_rn(__fmul_rn(d8_0, d8_1), (float)sumi);
}

// Weight unpack for one row / one super-block, producing the 8 ints (32 int8)
// that lane `kqsx` owns.
//
// Mapping, from mmq-load-tiles.cuh (`ggml_cuda_mmq_load_tiles_iq3_xxs`):
//   threads_per_row = (MMQ_ITER_K / (4*QR3_XXS)) / 2 = (256/16)/2 = 8
//   kqsx = threadIdx.x % 8
//   q3_packed = { get_int_b2(qs, 2*kqsx + 0), get_int_b2(qs, 2*kqsx + 1) }
//   for l in 0..QR3_XXS-1 (4):
//       grid_l = iq3xxs_grid[q3[2*l + 0]]      -> x_qs[i*65 + 8*kqsx + 2*l + 0]
//       grid_h = iq3xxs_grid[q3[2*l + 1]]      -> x_qs[i*65 + 8*kqsx + 2*l + 1]
//
// So lane `kqsx` owns the 8 ints at offset 8*kqsx, which is exactly ONE
// 32-element group (32 int8) -- group index kqsx, not 2*kqsx.  Reuses
// q35_iq3xxs_group32, the same helper the mma kernel calls, so both kernels
// multiply identical int8 bytes.
template <class T>
__device__ __forceinline__ int q35_dp4a_unpack_row(const unsigned char *wb, int kqsx, int out[8]) {
    int p[8];
    int ls[2];
    T::group(wb, kqsx, p, ls);
#pragma unroll
    for (int l = 0; l < 8; ++l) out[l] = p[l];
    return ls[0];
}

__global__ void q35_k_mmq_dp4a_iq(const unsigned char *__restrict__ w,
                                  const unsigned char *__restrict__ xq,
                                  float *__restrict__ out,
                                  int rows, int cols, int tokens, int ttype) {
    (void)ttype;
    extern __shared__ int smem[];
    int *tile_x = smem;                                  // I * SX int
    // x_df: one float per (row, 32-int8 group), padded `i/4` per row -- see
    // Q35_DP4A_DF.  I*8 + I/4 floats.
    float *tile_df = (float *)(tile_x + Q35_DP4A_I * Q35_DP4A_SX);
    int *tile_y = (int *)(tile_df + Q35_DP4A_I * Q35_DP4A_SDF + Q35_DP4A_I / 4);

    const int tid  = (int)threadIdx.x;
    const int lane = tid & 31;
    const int warp = tid >> 5;
    const int i0blk = blockIdx.x * Q35_DP4A_I;
    const int j0blk = blockIdx.y * Q35_DP4A_J;

    Q35Dp4aSums sum;
#pragma unroll
    for (int q = 0; q < Q35_DP4A_NSUM; ++q) sum.v[q] = 0.0f;

    for (int k00 = 0; k00 < cols; k00 += 256) {
        // ---- weights: 8 lanes per row (llama.cpp threads_per_row = 8) ----
        {
            const int kqsx = tid & 7;
            for (int r = tid >> 3; r < Q35_DP4A_I; r += Q35_DP4A_THREADS / 8) {
                const int i = i0blk + r;
                int *dst = tile_x + r * Q35_DP4A_SX;
                if (i < rows) {
                    const unsigned char *wb = w + (size_t)i * (size_t)(cols / 256) * 98 + (k00 / 256) * 98;
                    const int lsi = q35_dp4a_unpack_row<Q35Iq3Xxs>(wb, kqsx, dst + 8 * kqsx);
                    const float d = q35_f16_at(wb, 0);
                    // The weight scale.  llama.cpp writes `(ls*d + d/2)/2` with
                    // `ls = aux32 >> 28` (the RAW 4-bit value), which is
                    // d*(aux + 0.5)/2 = d*(2*aux + 1)/4.  Our helper already
                    // returns the doubled odd scale `2*aux + 1`, so feeding it
                    // into llama.cpp's expression adds a stray +d*0.5 term -- a
                    // quarter of the whole block's contribution.  Express it as
                    // d*ls/4, identical by construction to the mma kernel's
                    // `d * isum * 0.25` and to llama.cpp's value.
                    const float lsf = (float)lsi;
                    tile_df[Q35_DP4A_DF(r, kqsx)] = __fmul_rn(__fmul_rn(lsf, d), 0.25f);
                } else {
#pragma unroll
                    for (int q = 0; q < 8; ++q) dst[8 * kqsx + q] = 0;
                    tile_df[Q35_DP4A_DF(r, kqsx)] = 0.0f;
                }
            }
        }
        // ---- activations: J token rows x 36 int (32 int8 + 4 scale ints) ----
        {
            // Q8_K -> block_q8_1_mmq shape.  A Q8_K block is 256 int8 with ONE
            // f32 scale; block_q8_1_mmq is 128 int8 with FOUR float scales
            // (d4[0..3], one per 32 values).  The half selected by `half` picks
            // which 128 int8 land in tile_y, and all four d4 entries carry the
            // Q8_K scale -- Q8_K does not sub-divide it, so this is exact (not
            // an approximation), and `y_df[... + k0/8 % 4]` then returns the
            // same number for every k0.
            const unsigned char *base = xq + (size_t)(k00 / 256) * 292;
            const size_t tstride = (size_t)(cols / 256) * 292;
            // 4 scale ints + 32 qs ints per token row, one word per thread slot.
            for (int q = tid; q < Q35_DP4A_J * Q35_DP4A_SY; q += Q35_DP4A_THREADS) {
                const int j = q / Q35_DP4A_SY;
                const int w = q % Q35_DP4A_SY;
                const int tj = j0blk + j;
                if (tj >= tokens) { tile_y[q] = 0; continue; }
                const unsigned char *yb = base + (size_t)tj * tstride;
                if (w < 4) {
                    // All four scales are the single Q8_K scale (see above).
                    *((float *)(tile_y + j * Q35_DP4A_SY + w)) = q35_q8k_d(yb);
                } else {
                    // qs[] starts at byte 4; this launch stages the k00 half of
                    // the 256 int8, i.e. int8 [k00, k00+128) -> int [k00/4,
                    // k00/4 + 32).
                    const int *src = (const int *)yb + 1 + (k00 / 4) + (w - 4);
                    tile_y[j * Q35_DP4A_SY + w] = *src;
                }
            }
        }
        __syncthreads();

        // ---- vec_dot, twice: k00 = 0 and k00 = 32 ----
        // llama.cpp reloads tile_y between the two calls (mmq.cuh:921 and :938
        // stage `by0` and `by0 + sz` respectively), because each call consumes a
        // DIFFERENT 128-value half of the activation.
        //
        // `xq` here is already in `block_q8_1_mmq` layout -- 4 float scales then
        // 128 int8 per block, two blocks per super-block -- because the
        // quantizer that produced it (q35_launch_quantize_q8_K_mmq) emitted that
        // reshape in the same pass.  So this is llama.cpp's contiguous
        // `tile_y[l] = by0[l]` copy, with NO per-launch transpose.  An earlier
        // version transposed out of q8_K every launch and had to get the
        // int-vs-int8 offset of the upper half right by hand.
#pragma unroll
        for (int half2 = 0; half2 < 2; ++half2) {
            const int k00h = half2 * Q35_DP4A_NE_K;
            {
                // One mmq block is 36 ints (4 scale + 32 int8 payload), which is
                // exactly Q35_DP4A_SY, so the copy is a straight 32-bit move.
                // 144 B = 4 float scales + 128 int8, i.e. sizeof(block_q8_1_mmq)
                // (mmq-smem.cuh's q35_block_q8_1_mmq, not visible up here).
                const int *base = (const int *)(xq +
                    ((size_t)(k00 / 256) * 2 + half2) * 144);
                const int tstride = 2 * (Q35_DP4A_SY);   // two mmq blocks per super-block
                const int nsup = cols / 256;
                for (int q = tid; q < Q35_DP4A_J * Q35_DP4A_SY; q += Q35_DP4A_THREADS) {
                    const int j = q / Q35_DP4A_SY;
                    const int w = q % Q35_DP4A_SY;
                    const int tj = j0blk + j;
                    if (tj >= tokens) { tile_y[q] = 0; continue; }
                    tile_y[j * Q35_DP4A_SY + w] =
                        base[(size_t)tj * nsup * tstride + w];
                }
            }
            __syncthreads();

#pragma unroll
            for (int k01 = 0; k01 < Q35_DP4A_NE_K; k01 += Q35_DP4A_VDR) {
                const int k0 = k00h + k01;
#pragma unroll
                for (int jq = 0; jq < Q35_DP4A_J / Q35_DP4A_NW; ++jq) {
                    const int j = jq * Q35_DP4A_NW + warp;
                    const int *y_qs = tile_y + j * Q35_DP4A_SY + 4;
                    // All four d4 entries equal q8_K's single scale, so any
                    // index works; `(k0/8)%4` selects the quarter the dp4a
                    // consumes, matching llama.cpp's
                    // `y_df[j*MMQ_TILE_Y_K + (k0/QI8_1) % (MMQ_TILE_NE_K/QI8_1)]`.
                    const float dy = ((const float *)(tile_y + j * Q35_DP4A_SY))[(k0 / 8) % 4];
#pragma unroll
                    for (int iq = 0; iq < Q35_DP4A_I / 32; ++iq) {
                        const int i = iq * 32 + lane;
                        const int *x_qs = tile_x + i * Q35_DP4A_SX;
                        const float dx = tile_df[Q35_DP4A_DF(i, k0 / 8)];
                        // `k0 % 32` on the activation side: tile_y holds the 128
                        // int8 of this half, and the four k01 steps inside a
                        // half walk its four 32-int8 quarters.
                        sum.at(jq, iq) = __fadd_rn(sum.at(jq, iq),
                            q35_dp4a_dot32(x_qs + k0, y_qs + k01, dx, dy));
                    }
                }
            }
            __syncthreads();
        }   // half2
    }

    // ---- write back: sum[jq][iq] -> out[(j0blk+j)*rows + i0blk+i] ----
#pragma unroll
    for (int jq = 0; jq < Q35_DP4A_J / Q35_DP4A_NW; ++jq) {
        const int j = jq * Q35_DP4A_NW + warp;
        const int tj = j0blk + j;
        if (tj >= tokens) continue;
#pragma unroll
        for (int iq = 0; iq < Q35_DP4A_I / 32; ++iq) {
            const int i = i0blk + iq * 32 + lane;
            if (i < rows) out[(size_t)tj * (size_t)rows + (size_t)i] = sum.at(jq, iq);
        }
    }
}

// Launcher for the dp4a IQ kernel above.
static cudaError_t mmq_dp4a_iq_launch(const unsigned char *d_w, const unsigned char *d_xq,
                                      float *d_out, int rows, int cols, int tokens,
                                      cudaStream_t stream) {
    const int gridx = (rows + Q35_DP4A_I - 1) / Q35_DP4A_I;
    const int gridy = (tokens + Q35_DP4A_J - 1) / Q35_DP4A_J;
    const size_t shmem =
        (size_t)(Q35_DP4A_I * Q35_DP4A_SX + Q35_DP4A_J * Q35_DP4A_SY) * sizeof(int) +
        (size_t)(Q35_DP4A_I * Q35_DP4A_SDF + Q35_DP4A_I / 4) * sizeof(float);
    // (I*SX + J*SY) int + (I*SDF + I/4) float = 12928*4 + 1056*4 = 55,936 B,
    // over the 48 KiB static limit, so it MUST go through the dynamic path
    // (which needs the opt-in below past 48 KiB).  ptxas -v: 0 stack frame,
    // 0 spill, 174 registers -- i.e. occupancy 1 block/SM, the same as the mma
    // kernel and as llama.cpp's own dp4a config.
    static bool opted = false;
    if (!opted) {
        if (shmem > 48 * 1024) {
            cudaError_t e = cudaFuncSetAttribute(q35_k_mmq_dp4a_iq,
                cudaFuncAttributeMaxDynamicSharedMemorySize, (int)shmem);
            if (e != cudaSuccess) return e;
        }
        // NOTE: deliberately NO cudaDeviceSynchronize here.  It would break
        // graph capture (the engine captures one forward per chunk), and an
        // earlier version that added one was chasing a `misaligned address`
        // that turned out to be an int/byte offset mix-up in the activation
        // staging loop -- not an attribute-ordering problem.
        opted = true;
    }
    dim3 grid(gridx, gridy, 1);
    q35_k_mmq_dp4a_iq<<<grid, Q35_DP4A_THREADS, shmem, stream>>>(d_w, d_xq, d_out, rows, cols, tokens,
                                                                Q35_TT_IQ3_XXS);
    cudaError_t le = cudaGetLastError();
    if (le != cudaSuccess) {
        fprintf(stderr, "[dp4a] LAUNCH FAILED rows=%d cols=%d tokens=%d shmem=%zu grid=%dx%d: %s\n",
                rows, cols, tokens, shmem, gridx, gridy, cudaGetErrorString(le));
    }
    return le;
}

// QWEN35_CUDA_MMQ_DP4A=<tag,...> routes those types to the dp4a kernel above.
// Same parsing convention as q35_mma_iq_enabled: unset/"0" means off.
static bool q35_dp4a_enabled(int ttype) {
    // Re-read on every call: the A/B harness (/tmp/mbm/cmpsmall.cu) flips
    // $QWEN35_CUDA_MMQ_DP4A between two launches in one process to compare the
    // two kernels on identical data, and a cached value made both launches run
    // the same kernel (which reads as "the two agree perfectly").
    static int mode = -1;   // -1 unknown, 1 on, 2 off
    static char list[256];
    {
        const char *e = getenv("QWEN35_CUDA_MMQ_DP4A");
        if (e == nullptr || (e[0] == '0' && e[1] == '\0')) {
            mode = 2;
        } else {
            mode = 1;
            size_t n = sizeof(list) - 1;
            size_t i = 0;
            for (; i < n && e[i] != '\0'; ++i) list[i] = e[i];
            list[i] = '\0';
        }
    }
    if (mode == 2) return false;
    struct { int tt; const char *tag; } names[] = {
        {Q35_TT_IQ3_XXS, "iq3_xxs"},
    };
    for (const auto &nm : names) {
        if (nm.tt != ttype) continue;
        const char *p = list;
        const size_t ln = strlen(nm.tag);
        while (*p) {
            while (*p == ',' || *p == ' ') ++p;
            if (strncmp(p, nm.tag, ln) == 0 && (p[ln] == ',' || p[ln] == '\0')) return true;
            while (*p && *p != ',') ++p;
        }
        return false;
    }
    return false;
}

// How many WARPS per block the IQ prefill tile should use.
//
// Occupancy is what is left on the table.  The tile is 16 rows per warp, so the
// original 8-warp shape is a 128-row block and the 27B ffn (rows=5120,
// cols=17408) launches only (5120+127)/128 = 40 blocks for 114 SMs -- 65% of
// the device idle on the largest matrix in the model.  Measured throughput on
// that shape was 17.2 GB/s, against 250+ GB/s the same weights achieve in the
// decode kernel.
//
// Fewer warps = more blocks AND less shared memory (`sA` is NWARPS*16*256 int8),
// so the small-row matrices fill the device.  This is deliberately a launch-shape
// change only: each block still owns whole rows over the whole K range, so the
// store stays a plain assignment and the float summation order is untouched.
//
// (The obvious other lever, splitting the K range across grid.y, measured
// slightly better -- 88.8 -> 105.3 tok/s -- but it needs atomicAdd, which
// reorders the float sums.  MVCHECK passed and the token stream still diverged
// from the --MMQ=0 reference, so it was dropped: this project's decode/prefill
// equivalence standard is the token stream, not a tolerance.)
//
// $QWEN35_CUDA_MMQ_WARPS forces the warp count (1..8; 8 restores the old shape).
static int q35_iq_warps(int rows) {
    static int forced = -1;
    if (forced < 0) {
        forced = 0;
        const char *raw = getenv("QWEN35_CUDA_MMQ_WARPS");
        if (raw != nullptr && *raw) {
            int v = atoi(raw);
            if (v >= 1 && v <= 8) forced = v;
            else fprintf(stderr, "cuda kernels: QWEN35_CUDA_MMQ_WARPS='%s' is not in 1..8; ignoring\n", raw);
        }
    }
    if (forced > 0) return forced;
    // 8 warps (128 rows/block, 256 threads) -- re-measured after Q35_MMQ_MAX_TILES
    // dropped 128 -> 16.  The old note here said 4 warps and "w=4 vs w=8 is a
    // tie"; that tie was an artifact of the 2 KB per-thread accumulator.  With
    // the frame down to 128 B the 8-warp block no longer loses on occupancy and
    // wins outright:
    //   27B IQ3_XXS, 525-token prompt, tok/s:
    //     MAX_TILES \ warps:   4       8
    //          8              265.0   265.1
    //         16              268.7   274.1   <- both defaults
    //         24              267.7   274.1
    //         32              268.3   274.4
    // w=8 also halves the grid (27B ffn: 136 blocks per token slice instead of
    // 272), which is still 1.2x the 114 SMs once the token split is applied.
    // Historical sweep (MAX_TILES=128, IQ3_XXS, 525-token prompt) -- kept so the
    // old numbers are not re-litigated:
    //     warps:   1      2      4      8
    //   before:  94.6   96.0   97.4   88.5    (79-token prompt, pre-padding)
    //   after:  141.6  160.1  170.0  160.2    (525-token prompt, padded)
    //   J=128:  180.0  216.5  231.4  231.8
    (void)rows;
    return 8;
}

static cudaError_t mmq_mma_iq_launch(const unsigned char *d_w, const unsigned char *d_xq,
                                     float *d_out, int rows, int cols, int tokens, int ttype,
                                     cudaStream_t stream) {
    const int nw = q35_iq_warps(rows);
    const int rowtile = nw * 16;
    // grid.x is one row tile (NWARPS*16 rows) per block; grid.y is the token
    // split below.
    // J = tokens per block.  0 in the env var means "use the default"; a
    // negative value disables the split entirely (grid.y == 1, the old shape).
    //
    // 128 is not a padding-friendly round number, it is the measured optimum and
    // it is the SAME value llama.cpp's IQ3_XXS config picks (J=128,
    // mmq-config-ampere.cuh:293) -- two independent implementations converging.
    // 525-token IQ3_XXS prefill, tok/s:
    //     J:     0(old)  8     16    32    64    96    128   160   192   256
    //   tok/s:   202.3  121.0 169.6 206.2 225.7 224.8 233.7 231.8 227.9 225.0
    // The shape is a broad plateau, not a spike, so it does not need tuning per
    // model: 92/145/183-token prompts all peak at 64 or 128 too.
    //
    // Why splitting -- which RE-READS the weights 5x at 525 tokens -- is faster:
    // it is the same conclusion the chunk-size work reached from the other side.
    // This kernel is latency-bound, not bandwidth-bound (one launch costs ~the
    // same for 105 or 525 tokens), so what matters is how many independent
    // memory streams are in flight to hide the 600-800 ns HBM latency between
    // two __syncthreads().  grid.y == 1 gives the 27B ffn 272 blocks (2.4 per
    // SM); J=128 gives 272 x 5 = 1360 (12 per SM), and the extra parallelism is
    // worth far more than the duplicated traffic.  Confirmed by the J=8 point:
    // maximum parallelism but 66 weight re-reads, and it is the worst result.
    //
    // (K side of the same coin: raising the chunk 128 -> 1024 in engine.zig is
    // worth +18% and the two combine, 170.3 -> 233.7.)
    static int j_env = -2;
    if (j_env == -2) { const char* e = getenv("QWEN35_CUDA_MMQ_J"); j_env = e ? atoi(e) : 128; }
    int njsplit = 1;
    if (j_env > 0) {
        const int ntiles = (tokens + 7) / 8;
        const int jtiles = j_env / 8;                     // j_env is tokens/block
        njsplit = jtiles > 0 ? (ntiles + jtiles - 1) / jtiles : 1;
        if (njsplit < 1) njsplit = 1;
    }
    // HARD FLOOR, not a preference: `acc[MAX_TILES][4]` is a compile-time-sized
    // array indexed by LOCAL tile (t_hi - t_lo == tpt), so a tpt above
    // MAX_TILES writes past it with no diagnostic.  The kernel used to hard-code
    // MAX_TILES = Q35_MMQ_MAX_TOKENS/8 = 128, which made that impossible and
    // simultaneously put 2 KB of accumulator per thread in local memory (ptxas:
    // `2048 bytes stack frame`) -- that spill is what held IQ3_XXS at 37 TFLOPS
    // against llama.cpp's 127 on the same m=4096 n=512 k=14336 shape.  Dropping
    // MAX_TILES to 16 shrinks the frame to 128 B, and this floor keeps the
    // smaller array safe for ANY token count by raising njsplit (the token
    // split) until tpt fits.
    {
        const int ntiles = (tokens + 7) / 8;
        const int need = (ntiles + Q35_MMQ_MAX_TILES - 1) / Q35_MMQ_MAX_TILES;
        if (njsplit < need) njsplit = need;
    }
    const dim3 grid((unsigned)((rows + rowtile - 1) / rowtile), (unsigned)njsplit);
    const int threads = nw * Q35_WARP;
    #define Q35_IQ_LAUNCH(TP)                                                   \
        do {                                                                    \
            switch (nw) {                                                       \
            case 1: q35_k_mmq_mma_iq<TP, 1><<<grid, threads, 0, stream>>>(      \
                        d_w, d_xq, d_out, rows, cols, tokens, njsplit); break;           \
            case 2: q35_k_mmq_mma_iq<TP, 2><<<grid, threads, 0, stream>>>(      \
                        d_w, d_xq, d_out, rows, cols, tokens, njsplit); break;           \
            case 4: q35_k_mmq_mma_iq<TP, 4><<<grid, threads, 0, stream>>>(      \
                        d_w, d_xq, d_out, rows, cols, tokens, njsplit); break;           \
            default: q35_k_mmq_mma_iq<TP, 8><<<grid, threads, 0, stream>>>(     \
                        d_w, d_xq, d_out, rows, cols, tokens, njsplit); break;           \
            }                                                                   \
        } while (0)
    switch (ttype) {
        case Q35_TT_IQ3_XXS: Q35_IQ_LAUNCH(Q35Iq3Xxs); break;
        case Q35_TT_IQ3_S:   Q35_IQ_LAUNCH(Q35Iq3S);   break;
        case Q35_TT_IQ2_S:   Q35_IQ_LAUNCH(Q35Iq2S);   break;
        case Q35_TT_IQ4_XS:  Q35_IQ_LAUNCH(Q35Iq4Xs);  break;
        case Q35_TT_IQ2_XXS: Q35_IQ_LAUNCH(Q35Iq2Xxs); break;
        case Q35_TT_IQ2_XS:  Q35_IQ_LAUNCH(Q35Iq2Xs);  break;
        case Q35_TT_IQ1_S:   Q35_IQ_LAUNCH(Q35Iq1S);   break;
        case Q35_TT_IQ1_M:   Q35_IQ_LAUNCH(Q35Iq1M);   break;
        default:
            return (cudaError_t)-1;
    }
    #undef Q35_IQ_LAUNCH
    return cudaGetLastError();
}

// QWEN35_CUDA_MMQ_KMM: "0" disables the K-family tensor-core path entirely;
// otherwise it is a comma list of `q.ttype` tag names to enable selectively
// (empty/unset = all five).  Read once per launch shape; a getenv on the
// prefill path is noise next to the kernel itself.
static bool q35_mma_k_enabled(int ttype) {
    static int mode = -1;   // -1 unknown, 0 all, 1 list
    static char list[256];
    if (mode < 0) {
        const char *e = getenv("QWEN35_CUDA_MMQ_KMM");
        if (e == nullptr) {
            mode = 0;
        } else if (e[0] == '0' && e[1] == '\0') {
            mode = 2;  // explicit off
        } else {
            mode = 1;
            size_t n = sizeof(list) - 1;
            size_t i = 0;
            for (; i < n && e[i] != '\0'; ++i) list[i] = e[i];
            list[i] = '\0';
        }
    }
    if (mode == 2) return false;
    if (mode == 0) return true;
    // Selective: match the tag name of this ttype against the list.
    struct { int tt; const char *tag; } names[] = {
        {Q35_TT_Q2_K, "q2_k"}, {Q35_TT_Q3_K, "q3_k"}, {Q35_TT_Q4_K, "q4_k"},
        {Q35_TT_Q5_K, "q5_k"}, {Q35_TT_Q6_K, "q6_k"},
    };
    for (const auto &nm : names) {
        if (nm.tt != ttype) continue;
        const char *p = list;
        const size_t ln = strlen(nm.tag);
        while (*p) {
            while (*p == ',' || *p == ' ') ++p;
            if (strncmp(p, nm.tag, ln) == 0 && (p[ln] == ',' || p[ln] == '\0')) return true;
            while (*p && *p != ',') ++p;
        }
        return false;
    }
    return false;
}

// QWEN35_CUDA_MMQ_IQM: "0" disables the IQ-family tensor-core path entirely;
// otherwise it is a comma list of `q.ttype` tag names to enable selectively
// (empty/unset = all implemented).  Same shape as QWEN35_CUDA_MMQ_KMM so the
// two families A/B the same way.
static bool q35_mma_iq_enabled(int ttype) {
    static int mode = -1;   // -1 unknown, 0 all, 1 list, 2 off
    static char list[256];
    if (mode < 0) {
        const char *e = getenv("QWEN35_CUDA_MMQ_IQM");
        if (e == nullptr) {
            mode = 0;
        } else if (e[0] == '0' && e[1] == '\0') {
            mode = 2;
        } else {
            mode = 1;
            size_t n = sizeof(list) - 1;
            size_t i = 0;
            for (; i < n && e[i] != '\0'; ++i) list[i] = e[i];
            list[i] = '\0';
        }
    }
    if (mode == 2) return false;
    if (mode == 0) return true;
    struct { int tt; const char *tag; } names[] = {
        {Q35_TT_IQ3_XXS, "iq3_xxs"},
        {Q35_TT_IQ3_S,   "iq3_s"},
        {Q35_TT_IQ2_S,   "iq2_s"},
        {Q35_TT_IQ4_XS,  "iq4_xs"},
        {Q35_TT_IQ2_XXS, "iq2_xxs"},
        {Q35_TT_IQ2_XS,  "iq2_xs"},
        {Q35_TT_IQ1_S,   "iq1_s"},
        {Q35_TT_IQ1_M,   "iq1_m"},
    };
    for (const auto &nm : names) {
        if (nm.tt != ttype) continue;
        const char *p = list;
        const size_t ln = strlen(nm.tag);
        while (*p) {
            while (*p == ',' || *p == ' ') ++p;
            if (strncmp(p, nm.tag, ln) == 0 && (p[ln] == ',' || p[ln] == '\0')) return true;
            while (*p && *p != ',') ++p;
        }
        return false;
    }
    return false;
}




// See api.h.  `q35_dp4a_enabled` is the single source of truth for the switch.
extern "C" int q35_mmq_dp4a_wants(int ttype) {
    return (ttype == Q35_TT_IQ3_XXS && q35_dp4a_enabled(ttype)) ? 1 : 0;
}

// 2 mmq blocks per 256-element super-block, 144 B each.
extern "C" int q35_mmq_act_block_bytes(void) { return 2 * 144; }

extern "C" int q35_launch_mul_mat_vec_tile(const void *d_w_v, const void *d_xq_v,
                                           float *d_out, int rows, int cols,
                                           int tokens, int ttype, void *stream_v) {
    const unsigned char *d_w = (const unsigned char *)d_w_v;
    const unsigned char *d_xq = (const unsigned char *)d_xq_v;
    cudaStream_t stream = (cudaStream_t)stream_v;
    if (rows <= 0 || cols <= 0 || tokens <= 0) return 0;   // nothing to do
    const Q35QuantInfo info = q35_quant_info(ttype);
    if (info.block_elems == 0) return (int)cudaErrorInvalidValue;
    if (cols % info.block_elems != 0) return (int)cudaErrorInvalidValue;
    // Declines: "not my shape, use the caller's fallback".  -1 is outside the
    // cudaError_t range (those are >= 0), so it cannot be mistaken for a launch
    // failure that would need reporting.
    if (tokens > Q35_MMQ_MAX_TOKENS) return -1;

    // Tensor-core path: 32-element weight family.  mma gives the SAME exact
    // int32 per-block dot as the dp4a path, so the result is bit-identical
    // (MVCHECK: mismatched=0).  All six 32-elem types route here; the 256-elem
    // family (K/IQ) routes here per-type once each is implemented and verified.
    if (info.block_elems == 32) {
        // Q8_0 has a real mma kernel here, but the kernel loads its int8 weights
        // through q35_mma_w8(i) one element at a time: thread i reads row i/32,
        // so the warp's 32 lanes are `nblk * 34` bytes apart and every load
        // touches its own sector.  Measured on the 27B model's three 48x5120
        // q8_0 projections: 497 us per call at 8 tokens AND at 525 tokens --
        // i.e. a per-block constant, not work, because a 16-row tile is 48
        // lanes' worth of useful bytes per 5440-byte stride.  The warp-per-row
        // kernel reads one row's blocks contiguously, so Q8_0 declines to the
        // caller instead.  Guarded by the same switch as the other types so the
        // A/B stays possible.
        // Unconditional: there is no shape where the tile kernel beats
        // warp-per-row for Q8_0 here, and $QWEN35_CUDA_MMQ=0 already routes the
        // whole file back to it for A/B.
        if (ttype == Q35_TT_Q8_0) return -1;
        return (int)mmq_mma_launch(d_w, d_xq, d_out, rows, cols, tokens, ttype, stream);
    }
    // K family (256 elements, per-16 scales).  Gated by QWEN35_CUDA_MMQ_KMM
    // (a comma list of ttype tags, or "0" to disable, default on) so each type
    // can be A/B'd against the dp4a tile kernel in isolation.
    if (info.block_elems == 256 && q35_mma_k_enabled(ttype) &&
        (ttype == Q35_TT_Q2_K || ttype == Q35_TT_Q3_K || ttype == Q35_TT_Q4_K ||
         ttype == Q35_TT_Q5_K || ttype == Q35_TT_Q6_K)) {
        return (int)mmq_mma_k_launch(d_w, d_xq, d_out, rows, cols, tokens, ttype, stream);
    }

    // IQ family (256 elements, tensor-core path).  The mma D fragment IS the
    // 32-element integer reduction (see the block comment above
    // q35_k_mmq_mma_iq3xxs).  Gated by QWEN35_CUDA_MMQ_IQM ("0" to disable, or
    // a comma list of ttype tags, default on) so each type can be A/B'd
    // against the caller's thread-per-row kernel.
    // QWEN35_CUDA_MMQ_DP4A selects the llama.cpp-shaped dp4a kernel instead of
    // the mma one, per type, so the two can be measured on identical shapes.
    // Unset/"0" = mma (the long-standing path); a comma list of ttype tags
    // selects dp4a for exactly those types.  Currently only IQ3_XXS has a
    // dp4a kernel -- see q35_k_mmq_dp4a_iq.
    if (ttype == Q35_TT_IQ3_XXS && q35_dp4a_enabled(ttype)) {
        return (int)mmq_dp4a_iq_launch(d_w, d_xq, d_out, rows, cols, tokens, stream);
    }
    if (info.block_elems == 256 && q35_mma_iq_enabled(ttype) &&
        (ttype == Q35_TT_IQ3_XXS || ttype == Q35_TT_IQ3_S || ttype == Q35_TT_IQ2_S ||
         ttype == Q35_TT_IQ4_XS || ttype == Q35_TT_IQ2_XXS || ttype == Q35_TT_IQ2_XS ||
         ttype == Q35_TT_IQ1_S || ttype == Q35_TT_IQ1_M)) {
        return (int)mmq_mma_iq_launch(d_w, d_xq, d_out, rows, cols, tokens, ttype, stream);
    }

    // Types with no tensor-core kernel yet DECLINE to the caller's
    // thread-per-row path rather than running the dp4a tile kernel: measured
    // on 27B IQ3_XXS prefill (523 tokens), the dp4a tile is 19.3 tok/s against
    // 36.2 for thread-per-row, so routing the IQ family there is strictly
    // worse.  Each remaining IQ type gets its own mma kernel and flips over
    // around.  Each remaining IQ type gets its own mma kernel and flips over
    // here:  IQ4_XS done (1 scale per 32 elements, signed ls, no sign table),
    // IQ2_XXS done (u64 grid -> 8 unsigned magnitude bytes, 7-bit ksigns
    // expansion, 1 positive odd scale per 32 elements = the IQ3_XXS shape).
    // Still falling back: IQ1_S, IQ1_M.  Q35Iq1S exists and is registered in
    // the launcher's switch, but deliberately NOT in the dispatch list above:
    // it produces a wrong result, and the failing probe is worth recording
    // because it rules out the obvious causes.
    //
    // Unit probe (rows=1 cols=256, d=1.0, every qs byte 0 so idx=0, ls=1,
    // all-ones activations; iq1s_grid[0] is eight -1 bytes): the correct
    // answer is grid(-256) + delta(-32) = -288.  Measured -64 with the delta
    // bit set and 0 with it clear.  So the integer (grid) part contributes
    // NOTHING while the delta term is right, and the shortfall is a clean 1/8
    // -- one group out of eight -- not the 1/4 the first reading suggested
    // (that reading was taken while the dispatch list still excluded IQ1_S, so
    // it measured the thread-per-row path, not this kernel).
    //
    // Ruled out: block_bytes/field offsets (50 = 2+32+16, matches
    // block_iq1_s), grid index extraction (qs[l] | (((qh>>3l)&7)<<8), matches
    // dequantize_row_iq1_s:2665), the scale ((qh>>12)&7, and the probe scales
    // linearly with it: ls=1 -> -64, ls=3 -> -192), the delta decomposition
    // (matches vecdotq.cuh:1210 q35_vec_dot_iq1_s_q8_k_part), p[] contents
    // (0xFFFFFFFF per int is what a signed -1 table should yield, and the
    // group() loop covers g=0..7), and both staging paths (k16 reads
    // (const int*)sAw + h*4, k32 reads + g*8, both of which are correct for
    // the other five traits).
    //
    // Next step is a fragment-level probe in the style of /tmp/frag/*.cu, not
    // another dispatch experiment.
    // IQ2_XS is done: its uint16 grid index + 7-bit sign code unpack into the
    // same 8-byte magnitude shape as IQ2_XXS, and its two per-16 scales go
    // through the existing scale16 path (see Q35Iq2Xs).
    return -1;
}
