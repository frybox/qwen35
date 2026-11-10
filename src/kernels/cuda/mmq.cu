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
// Measured (27B IQ3_XXS, 525-token prompt, tok/s): K=128 170.3, K=256 184.1,
// K=512 187.3, K=1024 200.5.  Sublinear because the kernel is latency-bound:
// one launch costs ~the same for 105 or 525 tokens (2.68 vs 2.83 ms/call).
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
    __shared__ int8_t sA[8 * 16 * 256];
    __shared__ int8_t sB[8 * 256];
    int8_t *sAw = sA + warp * 16 * 256;
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
            q35_mma_k_aux8(ttype, wb, sAw + lane * 256);
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
                int8_t *dst = sB + trow * 256 + tcol * 8;
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
            q35_mma::load_ldmatrix(A_frag, (const int *)(sAw + g * 16), 64);
            q35_mma::load_generic(B_frag, (const int *)(sB + g * 16), 64);
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
    static constexpr int out_scale_q = 1;
    static constexpr int out_scale_den = 4;   // 0.25f
    __device__ static void group(const unsigned char *wb, int g, int p[8], int ls[2]);
};
struct Q35Iq3S {
    static constexpr int block_bytes = 110;
    static constexpr int out_scale_q = 1;
    static constexpr int out_scale_den = 1;   // 1.0f
    __device__ static void group(const unsigned char *wb, int g, int p[8], int ls[2]);
};
struct Q35Iq2S {
    static constexpr int block_bytes = 82;
    static constexpr int out_scale_q = 1;
    static constexpr int out_scale_den = 8;   // 0.125f
    __device__ static void group(const unsigned char *wb, int g, int p[8], int ls[2]);
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
    static constexpr int out_scale_q = 1;
    static constexpr int out_scale_den = 1;   // 1.0f
    __device__ static void group(const unsigned char *wb, int g, int p[8], int ls[2]);
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
    static constexpr int out_scale_q = 1;
    // 1/8, NOT IQ3_XXS's 1/4 -- see the derivation in group() below:
    //   db = d * (0.5f + hi) * 0.25f  and  ls = 2*hi + 1
    //   => db = d * ls * 0.125f, because 0.25*(0.5+hi) == 0.125*(2*hi+1).
    // The `2*hi` inside ls already absorbs a factor of two, so reusing
    // out_scale_den = 4 here doubles every output (MVCHECK: dev == 2 * cpu on
    // every row, mismatched=163839).
    static constexpr int out_scale_den = 8;   // 0.125f
    __device__ static void group(const unsigned char *wb, int g, int p[8], int ls[2]);
};

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
                                 int rows, int cols, int tokens) {
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
    __shared__ int8_t sA[NWARPS * 16 * SA_STRIDE];
    __shared__ int8_t sB[8 * 256];
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
    constexpr int MAX_TILES = 128;  // >= Q35_MMQ_MAX_TOKENS / 8 == 128 tiles
    const int ntiles = (tokens + 7) / 8;
    float acc[MAX_TILES][4];
    for (int t = 0; t < MAX_TILES; ++t)
        for (int l = 0; l < 4; ++l) acc[t][l] = 0.0f;

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
        const int row = lane >> 1;              // 0..15
        const int gp = lane & 1;                // 0..1
        if (row_live) {
            const unsigned char *wb = wb0 + (size_t)row * (size_t)nblk * block_bytes;
            int *dst = (int *)(sAw + row * SA_STRIDE);
#pragma unroll
            for (int k = 0; k < 4; ++k) {
                const int g = 4 * gp + k;
                int p[8];
                int ls[2];
                T::group(wb, g, p, ls);
                ls_w[row * LS_STRIDE + g * 2 + 0] = ls[0];
                ls_w[row * LS_STRIDE + g * 2 + 1] = ls[1];
                for (int i = 0; i < 8; ++i) dst[g * 8 + i] = p[i];
            }
        } else {
            for (int k = 0; k < 4; ++k) {
                ls_w[row * LS_STRIDE + (4 * gp + k) * 2 + 0] = 1;
                ls_w[row * LS_STRIDE + (4 * gp + k) * 2 + 1] = 1;
            }
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
        for (int t = 0; t < MAX_TILES; ++t) {
            const int tb = t * 8;
            if (tb >= tokens) continue;
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
                const int tid = (int)threadIdx.x;
                const int nt  = (int)blockDim.x;
                for (int i = tid * 8; i < 8 * 256; i += nt * 8) {
                    const int trow = i >> 8;               // which of the 8 tokens
                    const int tcol = i & 255;              // byte offset in the row
                    int8_t *dst = sB + i;
                    if (tb + trow < tokens && tcol + 8 <= 256) {
                        const unsigned char *row =
                            xq + (size_t)(tb + trow) * (size_t)nblk * 292 + (size_t)b * 292 + 4;
                        const unsigned char *src = row + tcol;
                        for (int k = 0; k < 8; ++k) dst[k] = (int8_t)src[k];
                    } else {
                        for (int k = 0; k < 8; ++k) dst[k] = 0;
                    }
                }
            }
            __syncthreads();

            // k16 mma, one per 16-element HALF-group: D.x[l] is then the exact
            // int32 dot of 16 elements, so the per-half scale is applied without
            // mixing.  k16 costs 16 mma instead of 8 per super-block; the tensor
            // cores are nowhere near the bottleneck (dequant + shared staging is),
            // so this is free.
            int isum[4] = {0, 0, 0, 0};
#pragma unroll
            for (int h = 0; h < 16; ++h) {
                q35_tile<16, 4, int> A_frag;
                q35_tile<8, 4, int> B_frag;
                q35_tile<16, 8, int> D;
                // Shared rows are SA_STRIDE int8 wide, so the ldmatrix stride is
                // SA_STRIDE/4 int (68 with the padding) and a 16-element half
                // starts at int offset h*4.
                q35_mma::load_ldmatrix(A_frag, (const int *)sAw + h * 4, SA_STRIDE / 4);
                q35_mma::load_generic(B_frag, (const int *)sB + h * 4, 64);
                q35_mma::mma(D, A_frag, B_frag);
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
            __syncthreads();   // sB is re-staged next token-tile

#pragma unroll
            for (int l = 0; l < 4; ++l) {
                const int mp = q35_tile<16, 8, int>::get_i(l);
                const int np = q35_tile<16, 8, int>::get_j(l);
                if (tb + np >= tokens || r0 + mp >= rows) continue;
                const unsigned char *wb = wb0 + (size_t)mp * (size_t)nblk * block_bytes;
                const unsigned char *yb =
                    xq + (size_t)(tb + np) * (size_t)nblk * 292 + (size_t)b * 292;
                const float d = __fmul_rn(q35_f16_at(wb, 0), q35_q8k_d(yb));
                // T::out_scale_q * OUT_SCALE_DEN: an integer template argument
                // times a power-of-two denominator, because C++ forbids a
                // floating-point template parameter.  Both factors are exact.
                acc[t][l] = __fadd_rn(acc[t][l], __fmul_rn(
                    __fmul_rn(d, (float)isum[l]), (float)OUT_SCALE_Q / (float)OUT_SCALE_DEN));
            }
        }
    }

    for (int t = 0; t < ntiles; ++t) {
        const int tb = t * 8;
#pragma unroll
        for (int l = 0; l < 4; ++l) {
            const int mp = q35_tile<16, 8, int>::get_i(l);
            const int np = q35_tile<16, 8, int>::get_j(l);
            if (tb + np < tokens && r0 + mp < rows) {
                out[(size_t)(tb + np) * (size_t)rows + (size_t)(r0 + mp)] = acc[t][l];
            }
        }
    }
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
    // 4 warps (64 rows/block) is the measured optimum on the 27B shapes, and the
    // 1..8 sweep is flat enough around it that a per-shape search is not worth a
    // getenv-free table.  525-token IQ3_XXS prefill, tok/s, before and after the
    // shared-memory bank-conflict padding (SA_STRIDE/LS_STRIDE above):
    //     warps:   1      2      4      8
    //   before:  94.6   96.0   97.4   88.5    (79-token prompt, pre-padding)
    //   after:  141.6  160.1  170.0  160.2    (525-token prompt, padded)
    // The padding is what pulled w=8 up from the clear loser to a near tie: the
    // 8-warp shape is the one whose shared tiles suffered the worst conflicts.
    // w=4 still wins on block count -- the 8-warp 128-row block would give the 27B ffn
    // (rows=5120) only (5120+127)/128 = 40 blocks for 114 SMs.
    //
    // The reason 4 wins over 2 and 1 despite fewer blocks per SM: the block's
    // shared-memory tile is the same 8-token activation tile whatever the warp
    // count, and each block pays one __syncthreads() per token tile.  Halving
    // the warps halves the rows that barrier work is amortized over, so past
    // w=4 that cost overtakes the occupancy gain.  The 8-warp shape is the clear
    // loser because it underfills the device (27B ffn: 40 blocks for 114 SMs).
    (void)rows;
    return 4;
}

static cudaError_t mmq_mma_iq_launch(const unsigned char *d_w, const unsigned char *d_xq,
                                     float *d_out, int rows, int cols, int tokens, int ttype,
                                     cudaStream_t stream) {
    const int nw = q35_iq_warps(rows);
    const int rowtile = nw * 16;
    // grid.x is one row tile per block; grid.y stays 1 because the kernel loops
    // over the 8-token tiles itself, so the weight slice is dequantized from HBM
    // once per super-block instead of once per tile.
    const dim3 grid((unsigned)((rows + rowtile - 1) / rowtile), 1);
    const int threads = nw * Q35_WARP;
    #define Q35_IQ_LAUNCH(TP)                                                   \
        do {                                                                    \
            switch (nw) {                                                       \
            case 1: q35_k_mmq_mma_iq<TP, 1><<<grid, threads, 0, stream>>>(      \
                        d_w, d_xq, d_out, rows, cols, tokens); break;           \
            case 2: q35_k_mmq_mma_iq<TP, 2><<<grid, threads, 0, stream>>>(      \
                        d_w, d_xq, d_out, rows, cols, tokens); break;           \
            case 4: q35_k_mmq_mma_iq<TP, 4><<<grid, threads, 0, stream>>>(      \
                        d_w, d_xq, d_out, rows, cols, tokens); break;           \
            default: q35_k_mmq_mma_iq<TP, 8><<<grid, threads, 0, stream>>>(     \
                        d_w, d_xq, d_out, rows, cols, tokens); break;           \
            }                                                                   \
        } while (0)
    switch (ttype) {
        case Q35_TT_IQ3_XXS: Q35_IQ_LAUNCH(Q35Iq3Xxs); break;
        case Q35_TT_IQ3_S:   Q35_IQ_LAUNCH(Q35Iq3S);   break;
        case Q35_TT_IQ2_S:   Q35_IQ_LAUNCH(Q35Iq2S);   break;
        case Q35_TT_IQ4_XS:  Q35_IQ_LAUNCH(Q35Iq4Xs);  break;
        case Q35_TT_IQ2_XXS: Q35_IQ_LAUNCH(Q35Iq2Xxs); break;
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
    if (info.block_elems == 256 && q35_mma_iq_enabled(ttype) &&
        (ttype == Q35_TT_IQ3_XXS || ttype == Q35_TT_IQ3_S || ttype == Q35_TT_IQ2_S ||
         ttype == Q35_TT_IQ4_XS || ttype == Q35_TT_IQ2_XXS)) {
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
    // Still falling back: IQ1_S/IQ1_M (1 element per 8-bit grid entry, so the
    // int8 staging this kernel does would need a different unpack), IQ2_XS
    // (8-bit grid entries + 7-bit signs, no k32-friendly scale).
    return -1;
}
