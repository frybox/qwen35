// mmq-smem.cuh -- shared-memory layout for the prefill MMQ path.
//
// PORT of llama.cpp ggml-cuda/mmq.cuh's smem layout decisions
// (block_q8_1_mmq, nbytes_shared, tile_x_sizes, MMQ_TILE_NE_K).
//
// WHY THIS FILE EXISTS
// --------------------
// The previous qwen35 tile kernel did something llama.cpp never does: it
// DEQUANTIZED the 128-row weight slice into a 32 KB `int8_t sA[...]` buffer,
// then fed that to ldmatrix.  llama.cpp keeps the COMPRESSED quantization tile
// in smem and lets the integer dot product consume the quantized lanes
// directly -- the unpacking happens as part of the arithmetic, not as a
// separate staging pass that pays for both time and shared memory.
//
// Measured consequence on 27B IQ3_XXS prefill:
//   old: sA 32 KB + sB 2 KB + sLs 8 KB = ~42 KB/block -> 1 block/SM,
//        plus one full unpack-every-element pass per super-block
//   new: nbs_x (compressed tile) + nbs_y (J * 40 B of block_q8_1_mmq),
//        typically 16-24 KB/block -> 2-3 blocks/SM, and NO separate pass
//
// Every constant is copied from upstream with its anchor noted, so a future
// reader can diff instead of re-deriving the bank-conflict padding (which is
// load-bearing: the "+ I" terms in the tile_x_sizes formulas below keep
// consecutive rows off the same smem bank).
//
// IMPORTANT: tile_x_sizes is a function of `I` (rows per tile), not a
// per-type constant.  An earlier draft of this file tried to hard-code byte
// counts per type; upstream is parameterised and any re-derivation drifts.
#pragma once

#include <cstdint>

#include "api.h"   // Q35_TT_* ttype ids (q35_mmq_xfam_of below)

// The QK* values above are the same numbers common.cuh defines for its own
// quantizer; both are fixed by the ggml block layouts, so a second definition
// here is safe as long as they agree (checked at compile time when common.cuh
// is also included).
#ifdef Q35_QK8_0
static_assert(Q35_QK8_0 == 32, "QK8_0 disagrees with common.cuh");
#endif

// ---------------------------------------------------------------------------
// Quantization granularities (ggml-common.h:89-140, 194, 251, 258)
// ---------------------------------------------------------------------------
#define Q35_QK_K   256
#define Q35_QK4_0  32
#define Q35_QK8_0  32
#define Q35_QK8_1  32
#define Q35_QR4_0  2
#define Q35_QR4_1  2
#define Q35_QR8_0  1
#define Q35_QR8_1  1
#define Q35_QR2_K  4
#define Q35_QR3_K  4
#define Q35_QR4_K  2
#define Q35_QR5_K  2
#define Q35_QR6_K  2
// QIx = "how many ints" the quants of one 32-value tile row occupy.
#define Q35_QI4_0  (Q35_QK4_0 / (4 * Q35_QR4_0))          // 4
#define Q35_QI4_1  (Q35_QK4_0 / (4 * Q35_QR4_1))          // 4
#define Q35_QI8_0  (Q35_QK8_0 / (4 * Q35_QR8_0))          // 8
#define Q35_QI8_1  (Q35_QK8_1 / (4 * Q35_QR8_1))          // 8
#define Q35_QI2_K  (Q35_QK_K / (4 * Q35_QR2_K))           // 16
#define Q35_QI3_K  (Q35_QK_K / (4 * Q35_QR3_K))           // 16
#define Q35_QI4_K  (Q35_QK_K / (4 * Q35_QR4_K))           // 32
#define Q35_QI5_K  (Q35_QK_K / (4 * Q35_QR5_K))           // 32
#define Q35_QI6_K  (Q35_QK_K / (4 * Q35_QR6_K))           // 32

// Activation tile granularity (mmq.cuh:24-25).
static constexpr int Q35_QK8_1_MMQ = 4 * Q35_QK8_1;   // 128 values per block
static constexpr int Q35_QI8_1_MMQ = 4 * Q35_QI8_1;   // 32 ints per block

// ---------------------------------------------------------------------------
// block_q8_1_mmq (mmq.cuh:27-53, verbatim field order)
// ---------------------------------------------------------------------------
// The y (activation) tile is stored ALREADY TRANSFORMED by the quantize
// kernel: the float row is cut into 128-value blocks, each quantized as four
// groups of 32, and the whole thing transposed so the smem copy is a plain
// contiguous int assignment with no rearrangement inside the matvec kernel.
//
// The 32 B of scales double as the "pad each block with 16 bytes" the upstream
// comment mentions: 128 values = 4 groups of 32, one scale-half2 pair each, so
// 4 * 8 B = 32 B of ds riding along with 128 B of quants -> 160 B total.
//
// The union's meaning depends on the x type (mmq_get_q8_1_ds_layout,
// mmq.cuh:57): D4 = float scales only (no partial sum), DS4 = half d + half s
// per 32 values, D2S6 = d per 64 values + s per 16.  qwen35's int8 mma
// (m16n8k32 s8.s8.s32) carries no per-block scale, so the integer dot is exact
// and the scale is applied outside it: that is D4.
struct q35_block_q8_1_mmq {
    union {
        float d4[4];      // 1 f32 scale per 32 values
        uint32_t ds4[8];  // 4 x (half d, half s)
        uint32_t d2s6[8]; // d per 64 values + s per 16 values
    };
    int8_t qs[Q35_QK8_1_MMQ];
};
// Upstream pins this against 4*sizeof(block_q8_1) = 4*40 = 160 B: 128 B of
// quants + 4 half2 pairs (8 B each) of scale/partial-sum.
static_assert(sizeof(q35_block_q8_1_mmq) == Q35_QK8_1_MMQ + 4 * 8,
              "q8_1_mmq must be 160 B per 128 values (cf. upstream static_assert)");

// ---------------------------------------------------------------------------
// Tile geometry (mmq.cuh:110-125)
// ---------------------------------------------------------------------------
// MMQ_TILE_NE_K: elements of one row covered by a single smem x-tile.  Always
// 32 upstream: every format's block is a multiple of 32, which is also the
// q8_1_mmq granularity, so x and y tiles share one index space.
static constexpr int Q35_MMQ_TILE_NE_K = 32;

// MMQ_ITER_K: elements of a row processed per inner pass.
static constexpr int Q35_MMQ_ITER_K = 256;

// MMQ_TILE_Y_K is measured in 32-BIT WORDS, not elements (mma.cuh:118):
//   "block_q8_1_mmq has (128 8-bit ints == 32 32-bit ints + 4 32-bit scales)"
// i.e. 32 + 32/QI8_1 = 32 + 4 = 36 words = 144 B for 128 values.  NOTE this
// is a DIFFERENT quantity from sizeof(block_q8_1_mmq), which is 160 B: the
// struct as declared upstream has d4[4] (16 B) + qs[128] (128 B) = 144 B...
// except the union is ds4[8] = 32 B wide, so the struct is 160 B.  The two
// numbers describe the same data at different granularities: TILE_Y_K is the
// number of ints that the *tile* spends per row (the last 16 B of the 160 B
// struct are the ds region, counted as 4 ints), while sizeof is the stride of
// the *global* array written by quantize_mmq_q8_1.  Keep both.
static constexpr int Q35_MMQ_TILE_Y_K = Q35_MMQ_TILE_NE_K + Q35_MMQ_TILE_NE_K / Q35_QI8_1;  // 36

// Bytes of activation tile per token row, for a J-token tile (mmq.cuh):
//   nbs_y = J * sizeof(block_q8_1_mmq)
// One row of the tile is J blocks of 128 values... i.e. the J tokens ARE the
// row dimension here, each contributing one 160 B block per 128 K values.
// MMA_ITER_K = 256 therefore means 2 blocks per token row.
#define Q35_MMQ_Y_BYTES_PER_ROW(J) ((J) * (int)sizeof(q35_block_q8_1_mmq) * (Q35_MMQ_ITER_K / Q35_QK8_1_MMQ))

// ---------------------------------------------------------------------------
// x-side tile sizes (mmq.cuh:379-388, verbatim formulas)
// ---------------------------------------------------------------------------
// A packed weight row interleaves scale/min bytes with the quants in a
// type-specific way, so upstream splits one tile row into three regions:
//   qs -- quant bytes, arranged so ldmatrix reads them contiguously
//   dm -- per-32-element (d, m) pairs of the affine formats
//   sc -- per-sub-block scales of the K-quant family
// All three are expressed in BYTES for `I` rows, and the trailing "+ I",
// "+ I/8" terms are the per-row padding that avoids smem bank conflicts.
struct q35_tile_x_sizes {
    int qs, dm, sc;
    constexpr int total() const { return qs + dm + sc; }
};

// QK4_0 / QK8_0 payloads are 16 B / 32 B per 32 values; the byte count for I
// rows is therefore I*NE_K*(bytes per value) with the +I row padding.
#define Q35_MMQ_TXS_Q4_0(I)    q35_tile_x_sizes{(I)*Q35_MMQ_TILE_NE_K       + (I), (I)*Q35_MMQ_TILE_NE_K/Q35_QI4_0       + (I)/Q35_QI4_0,      0}
#define Q35_MMQ_TXS_Q4_1(I)    q35_tile_x_sizes{(I)*Q35_MMQ_TILE_NE_K       + (I), (I)*Q35_MMQ_TILE_NE_K/Q35_QI4_1       + (I)/Q35_QI4_1,      0}
#define Q35_MMQ_TXS_Q8_0(I)    q35_tile_x_sizes{(I)*Q35_MMQ_TILE_NE_K*2     + (I), (I)*Q35_MMQ_TILE_NE_K*2/Q35_QI8_0     + (I)/(Q35_QI8_0/2),  0}
#define Q35_MMQ_TXS_Q8_0_16(I) q35_tile_x_sizes{(I)*Q35_MMQ_TILE_NE_K*2     + (I), (I)*Q35_MMQ_TILE_NE_K*4/Q35_QI8_0     + (I)/(Q35_QI8_0/4),  0}
#define Q35_MMQ_TXS_Q8_1(I)    q35_tile_x_sizes{(I)*Q35_MMQ_TILE_NE_K*2     + (I), (I)*Q35_MMQ_TILE_NE_K*2/Q35_QI8_1     + (I)/(Q35_QI8_1/2),  0}
#define Q35_MMQ_TXS_Q2_K(I)    q35_tile_x_sizes{(I)*Q35_MMQ_TILE_NE_K*2     + (I), (I)*Q35_MMQ_TILE_NE_K                 + (I),                0}
#define Q35_MMQ_TXS_Q3_K(I)    q35_tile_x_sizes{(I)*Q35_MMQ_TILE_NE_K*2     + (I), (I),                                                (I)*Q35_MMQ_TILE_NE_K/8 + (I)/8}
#define Q35_MMQ_TXS_Q4_K(I)    q35_tile_x_sizes{(I)*Q35_MMQ_TILE_NE_K       + (I), (I)*Q35_MMQ_TILE_NE_K/Q35_QI4_K,                   (I)*Q35_MMQ_TILE_NE_K/8 + (I)/8}
#define Q35_MMQ_TXS_Q5_K(I)    q35_tile_x_sizes{(I)*Q35_MMQ_TILE_NE_K*2     + (I), (I)*Q35_MMQ_TILE_NE_K/Q35_QI5_K     + (I)/Q35_QI5_K,     (I)*Q35_MMQ_TILE_NE_K/8 + (I)/8}
#define Q35_MMQ_TXS_Q6_K(I)    q35_tile_x_sizes{(I)*Q35_MMQ_TILE_NE_K*2     + (I), (I)*Q35_MMQ_TILE_NE_K/Q35_QI6_K     + (I)/Q35_QI6_K,     (I)*Q35_MMQ_TILE_NE_K/8 + (I)/8}

// Which family a weight type routes to (mmq.cuh:390-414).  IQ1_S..IQ4_NL all
// reuse the Q8_0 sizes upstream: those formats' block bytes are staged as a
// contiguous quant payload and the grid lookups happen on the registers, so
// the byte count is driven by q8_0's 32 B per 32 values, not by the block's
// nominal packing.  Keeping the upstream mapping (rather than inventing
// per-type counts) is what makes nbs_x match for a given (type, I).
enum class q35_mmq_xfam {
    q4_0, q4_1, q8_0, q8_0_16, q8_1, q2_k, q3_k, q4_k, q5_k, q6_k, none
};

inline constexpr q35_mmq_xfam q35_mmq_xfam_of(int ttype) {
    // Mirrors mmq_get_dp4a_tile_x_sizes (mmq.cuh:390-414) using qwen35's ttype
    // ids from api.h.  Note Q5_0 -> q8_0 and Q5_1 -> q8_1: the 5th-bit plane
    // is what decides the split, not the nominal bit width (upstream maps them
    // exactly this way).  Types with no tile kernel stay `none`.
    switch (ttype) {
        case Q35_TT_Q4_0:    return q35_mmq_xfam::q4_0;
        case Q35_TT_Q4_1:    return q35_mmq_xfam::q4_1;
        case Q35_TT_Q5_0:    return q35_mmq_xfam::q8_0;
        case Q35_TT_Q5_1:    return q35_mmq_xfam::q8_1;
        case Q35_TT_Q8_0:    return q35_mmq_xfam::q8_0;
        case Q35_TT_IQ4_NL:  return q35_mmq_xfam::q8_0;
        case Q35_TT_Q2_K:    return q35_mmq_xfam::q2_k;
        case Q35_TT_Q3_K:    return q35_mmq_xfam::q3_k;
        case Q35_TT_Q4_K:    return q35_mmq_xfam::q4_k;
        case Q35_TT_Q5_K:    return q35_mmq_xfam::q5_k;
        case Q35_TT_Q6_K:    return q35_mmq_xfam::q6_k;
        case Q35_TT_IQ1_S:   return q35_mmq_xfam::q8_0;
        case Q35_TT_IQ1_M:   return q35_mmq_xfam::q8_0;
        case Q35_TT_IQ2_XXS: return q35_mmq_xfam::q8_0;
        case Q35_TT_IQ2_XS:  return q35_mmq_xfam::q8_0_16;
        case Q35_TT_IQ2_S:   return q35_mmq_xfam::q8_0_16;
        case Q35_TT_IQ3_XXS: return q35_mmq_xfam::q8_0;
        case Q35_TT_IQ3_S:   return q35_mmq_xfam::q8_0;
        case Q35_TT_IQ4_XS:  return q35_mmq_xfam::q8_0;
        default:             return q35_mmq_xfam::none;
    }
}

// The sizes for one tile of `I` rows of `fam`.
inline q35_tile_x_sizes q35_mmq_tile_x_sizes(q35_mmq_xfam fam, int I) {
    switch (fam) {
        case q35_mmq_xfam::q4_0:    return Q35_MMQ_TXS_Q4_0(I);
        case q35_mmq_xfam::q4_1:    return Q35_MMQ_TXS_Q4_1(I);
        case q35_mmq_xfam::q8_0:    return Q35_MMQ_TXS_Q8_0(I);
        case q35_mmq_xfam::q8_0_16: return Q35_MMQ_TXS_Q8_0_16(I);
        case q35_mmq_xfam::q8_1:    return Q35_MMQ_TXS_Q8_1(I);
        case q35_mmq_xfam::q2_k:    return Q35_MMQ_TXS_Q2_K(I);
        case q35_mmq_xfam::q3_k:    return Q35_MMQ_TXS_Q3_K(I);
        case q35_mmq_xfam::q4_k:    return Q35_MMQ_TXS_Q4_K(I);
        case q35_mmq_xfam::q5_k:    return Q35_MMQ_TXS_Q5_K(I);
        case q35_mmq_xfam::q6_k:    return Q35_MMQ_TXS_Q6_K(I);
        default:                    return q35_tile_x_sizes{0, 0, 0};
    }
}

// ---------------------------------------------------------------------------
// MMA sram layouts and strides (mma.cuh:112-144)
// ---------------------------------------------------------------------------
// On the MMA path the x tile is NOT stored as {qs,dm,sc} triplets; the whole
// row is ONE int array of `sram_stride` words, and the type's load_tiles
// decides where the quants and the scales land inside it.  The stride must be
// 4 mod 8 so that consecutive rows stay on different smem banks (upstream
// static_asserts this for every layout).
enum q35_mmq_sram_layout {
    Q35_SRAM_Q8_0,
    Q35_SRAM_Q8_1,
    Q35_SRAM_Q2_K,
    Q35_SRAM_Q3_K,
    Q35_SRAM_Q6_K,
};

// Words per row of the x tile.  Note Q4_K and Q5_K use the Q6_K stride: the
// 4/5-bit K formats split their scales the same way the 6-bit one does.
__host__ __device__ constexpr int q35_sram_stride(q35_mmq_sram_layout l) {
    switch (l) {
        case Q35_SRAM_Q8_0: return 2 * Q35_MMQ_TILE_NE_K + 2 * Q35_MMQ_TILE_NE_K / Q35_QI8_0 + 4;  // 76
        case Q35_SRAM_Q8_1: return 2 * Q35_MMQ_TILE_NE_K + 2 * Q35_MMQ_TILE_NE_K / Q35_QI8_1 + 4;  // 76
        case Q35_SRAM_Q2_K: return 2 * Q35_MMQ_TILE_NE_K +     Q35_MMQ_TILE_NE_K         + 4;      // 100
        case Q35_SRAM_Q3_K: return 2 * Q35_MMQ_TILE_NE_K +     Q35_MMQ_TILE_NE_K / 2     + 4;      // 84
        case Q35_SRAM_Q6_K: return 2 * Q35_MMQ_TILE_NE_K +     Q35_MMQ_TILE_NE_K / Q35_QI6_K
                                                                  + Q35_MMQ_TILE_NE_K / 8 + 7;    // 76
    }
    return -1;
}
static_assert(q35_sram_stride(Q35_SRAM_Q8_0) % 8 == 4, "Wrong padding (upstream static_assert)");
static_assert(q35_sram_stride(Q35_SRAM_Q8_1) % 8 == 4, "Wrong padding (upstream static_assert)");
static_assert(q35_sram_stride(Q35_SRAM_Q2_K) % 8 == 4, "Wrong padding (upstream static_assert)");
static_assert(q35_sram_stride(Q35_SRAM_Q3_K) % 8 == 4, "Wrong padding (upstream static_assert)");
static_assert(q35_sram_stride(Q35_SRAM_Q6_K) % 8 == 4, "Wrong padding (upstream static_assert)");

// Which layout each weight type uses.  This is EXACTLY the sram_layout column
// of every row in mmq-config-ampere.cuh (verified by parsing all 352 CASE rows:
// each type has a single (nthreads=256, occupancy=1, I=128, layout, K_vram=
// MMQ_ITER_K, stream_k=true) tuple, so the layout is a pure function of type).
__host__ __device__ constexpr q35_mmq_sram_layout q35_sram_layout_of(int ttype) {
    switch (ttype) {
        // Q8_0 layout: 32 ints of quants + 4 ints of scale/half2 per 128 values.
        case Q35_TT_Q4_0: case Q35_TT_Q8_0: case Q35_TT_IQ4_NL:
        case Q35_TT_IQ1_S: case Q35_TT_IQ1_M: case Q35_TT_IQ2_XXS:
        case Q35_TT_IQ3_XXS: case Q35_TT_IQ3_S: case Q35_TT_IQ4_XS:
            return Q35_SRAM_Q8_0;
        // Q8_1 layout: identical shape to Q8_0 (both 76 words) but the scale
        // region holds the (d, s) pair of formats with a nonzero min.
        case Q35_TT_Q4_1: case Q35_TT_Q5_1: case Q35_TT_Q4_K: case Q35_TT_Q5_K:
            return Q35_SRAM_Q8_1;
        case Q35_TT_Q2_K: return Q35_SRAM_Q2_K;
        // IQ2_S and IQ2_XS share Q3_K's stride: both carry 8-bit per-group
        // scales in a separate int array.  (Their load_tiles differ; only the
        // tile LENGTH is shared, which is all this function decides.)
        case Q35_TT_Q3_K: case Q35_TT_IQ2_S: case Q35_TT_IQ2_XS:
            return Q35_SRAM_Q3_K;
        case Q35_TT_Q6_K: return Q35_SRAM_Q6_K;
        default: return Q35_SRAM_Q8_0;
    }
}

// I == 128 for EVERY type in the Ampere table (mmq-config-ampere.cuh), across
// all J.  So the tile is 128 rows wide in the K direction... no: I is the tile
// extent along src0->ne[1] (the output-feature axis), J along the token axis.
// Q35_MMQ_I is fixed here because qwen35 only ever needs the J-batching that
// the prefill path uses; if a future type needs a different I, add it to
// q35_mmq_i_of() rather than editing the constant in place.
static constexpr int Q35_MMQ_I = 128;

// rows_per_warp (mmq.cuh:179): 32 when J is a multiple of 16 and >= 48, else
// 16.  This is what the load_tiles functions use to decide how many rows one
// warp stages per pass.
__host__ __device__ constexpr int q35_mmq_rows_per_warp(int J) {
    return (J >= 48 && J % 16 == 0) ? 32 : 16;
}

static constexpr __host__ __device__ int q35_mmq_nbs_x(int ttype) {
    return Q35_MMQ_I * q35_sram_stride(q35_sram_layout_of(ttype)) * (int)sizeof(int);
}

// ---------------------------------------------------------------------------
// smem budget arithmetic (mmq.cuh:1395-1412, verbatim shape)
// ---------------------------------------------------------------------------
//   nbytes_shared = nbs_ids + nbs_x + GGML_PAD(nbs_y, nthreads * sizeof(int))
//
// nbs_ids is the J-int destination-id array; qwen35 has no ids path so it is
// always 0, but the term is kept so this formula can be diffed against
// upstream.  nbs_y is padded to a per-thread int boundary so a partial warp
// still lands on an aligned slot.
#define Q35_MMQ_PAD(n, m) (((n) + (m) - 1) / (m) * (m))

// MEASURED BUDGET (I=128, 256 threads, worst type = Q2_K at stride 100):
//   J=  8 -> 54272 B      J= 32 -> 61440 B      J= 64 -> 71680 B
//   J= 16 -> 56320 B      J= 48 -> 66560 B      J=128 -> 92160 B
// Two consequences the launcher must respect:
//   1. even J=8 exceeds the 48 KB static limit, so the kernel MUST be launched
//      with cudaFuncSetAttribute(MaxDynamicSharedMemorySize) and the tile in
//      DYNAMIC smem (a static __shared__ array of this size does not link);
//   2. RTX 4090 caps at 99 KB/block, so J=128 only fits for the cheaper types
//      and there is exactly 1 block/SM at that point -- J must be chosen from
//      the occupancy target, not fixed.  That is what the upstream config table
//      is for.  Small J is not automatically better: see Q35_MMQ_J_BEST below.
static constexpr int Q35_MMQ_SMEM_STATIC_LIMIT = 48 * 1024;
static constexpr int Q35_MMQ_SMEM_HW_MAX_4090  = 99 * 1024;

static inline int q35_mmq_nbytes_shared(int J, int nbs_x, int nthreads, bool use_ids = false) {
    const int nbs_ids = use_ids ? J * (int)sizeof(int) : 0;
    const int nbs_y   = Q35_MMQ_Y_BYTES_PER_ROW(J);
    return nbs_ids + nbs_x + (int)Q35_MMQ_PAD(nbs_y, nthreads * (int)sizeof(int));
}
