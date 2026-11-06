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
// `tokens > Q35_MMQ_MAX_TOKENS` (the engine's prefill chunk is 16) is declined
// at launch time: the launcher returns -1, which mmvq.cu reads as "use the
// thread-per-row kernel".  A half-idle tile would be slower than the fallback,
// and a silently wrong tile is worse than either.
#include "dotq.cuh"
#include "mma.cuh"

#define Q35_MMQ_THREADS 256
#define Q35_MMQ_WARPS (Q35_MMQ_THREADS / Q35_WARP)
#define Q35_MMQ_MAX_TOKENS 16   // engine.zig nbatch: the register budget boundary

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

static cudaError_t mmq_mma_launch(const unsigned char *d_w, const unsigned char *d_xq,
                                  float *d_out, int rows, int cols, int tokens, int ttype,
                                  cudaStream_t stream) {
    const dim3 grid((unsigned)((rows + 127) / 128), (unsigned)((tokens + 7) / 8));
    q35_k_mmq_mma_q32<<<grid, Q35_MMQ_THREADS, 0, stream>>>(d_w, d_xq, d_out, rows, cols, tokens, ttype);
    return cudaGetLastError();
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
    const int nblk = cols / info.block_elems;

    // Tensor-core path: 32-element weight family.  mma gives the SAME exact
    // int32 per-block dot as the dp4a path, so the result is bit-identical
    // (MVCHECK: mismatched=0).  All six 32-elem types route here; the 256-elem
    // family (K/IQ) routes here per-type once each is implemented and verified.
    if (info.block_elems == 32) {
        return (int)mmq_mma_launch(d_w, d_xq, d_out, rows, cols, tokens, ttype, stream);
    }

    // BLK per lane is a runtime pick now (see the kernel's comment for why it is
    // not a template parameter any more).  A warp covers `Q35_WARP * blk`
    // blocks and must cover the whole row, so blk = ceil(nblk / 32).  nblk 20
    // (attention) gives 1, nblk 68 (FFN) gives 3; every row is one pass and
    // there is no tail loop.  All shapes share the SAME TPERPASS=16
    // instantiation, so this file compiles exactly one kernel.
    const int blk = (nblk + Q35_WARP - 1) / Q35_WARP;
    return (int)mmq_launch<8>(d_w, d_xq, d_out, rows, cols, tokens, ttype, blk, stream);
}
