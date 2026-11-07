// MMQ x-tile loaders for the K-quant family -- ported from llama.cpp
//   ggml/src/ggml-cuda/mmq-load-tiles.cuh:537-1035
//     q2_K :537  q3_K :598  q4_K :711  q5_K :822  q6_K :946
//
// Only the MMA branch is carried.  Upstream guards two branches per function,
// `#if defined(TURING_MMA_AVAILABLE) || ...` (this file) and the dp4a `#else`.
// qwen35 never runs the dp4a one: sm_75+ always has the int8 mma we need, and
// keeping a second layout would defeat the point of this port (see the note at
// the top of mmq-smem.cuh about dropping the sA staging buffer).
//
// The MMA x-tile is ONE int array of sram_stride words per row; where the
// quants and the scales land inside it is decided here, per type.  Layout per
// 128 K-values (k = intra-tile element index, 0..2*MMQ_TILE_NE_K-1):
//   x_qs[0 .. 2*NE_K)        the unpacked int8 quants, 32 ints
//   x_dm / x_df / x_sc       the scale region, offset 2*NE_K words
// The scale region's shape is what distinguishes the layouts:
//   Q8_0 (76 w): half2 dm[16]          -> 32 words
//   Q8_1 (76 w): same shape, (d,s) pair instead of (d,min)
//   Q2_K (100 w): half2 dm[32]         <- 2 scale bytes per 16 elements
//   Q3_K (84 w):  float df[32] + 8-bit scales packed in the tail
//   Q6_K (76 w):  half2 dm[16] + 8-bit int8 scales in the tail (+7 pad)
#ifndef Q35_MMQ_LOAD_TILES_K_CUH
#define Q35_MMQ_LOAD_TILES_K_CUH

#include "common.cuh"
#include "vecdotq.cuh"   // Q35_BYTES_Q*_K block sizes
#include "mmq-config.cuh"
#include "mmq-smem.cuh"

namespace q35_mmq {

// Physical warp is always 32 here (no AMD wave64 target).
static constexpr int WARP = 32;

// threadIdx.x/y decomposition shared by every loader.  `threads_per_row` is how
// many threads cooperate on ONE row's 256 K-values; a warp therefore covers
// WARP/threads_per_row rows per pass.
template <int nthreads, int I>
struct tile_ctx {
    static constexpr int nwarps = nthreads / WARP;
    int tx, ty;
    __device__ __forceinline__ tile_ctx() : tx(threadIdx.x), ty(threadIdx.y) {}
};

// ---------------------------------------------------------------------------
// q2_K (mmq-load-tiles.cuh:537)
// ---------------------------------------------------------------------------
template <int nthreads, int I, bool fallback>
__device__ __forceinline__ void load_tiles_q2_K(const char *__restrict__ x,
                                                int *__restrict__ x_tile,
                                                int kbx0, int i_max, int stride) {
    constexpr int sram_stride = q35_sram_stride(Q35_SRAM_Q2_K);
    int   *x_qs = (int *)x_tile;
    half2 *x_dm = (half2 *)(x_qs + 2 * Q35_MMQ_TILE_NE_K);

    constexpr int threads_per_row = Q35_MMQ_ITER_K / (4 * Q35_QR2_K);   // 256/16 = 16
    constexpr int nrows = WARP / threads_per_row;                        // 2
    constexpr int nwarps = nthreads / WARP;
    const int kqsx = threadIdx.x % threads_per_row;

#pragma unroll
    for (int i0 = 0; i0 < I; i0 += nrows * nwarps) {
        int i = i0 + threadIdx.y * nrows + threadIdx.x / threads_per_row;
        if (fallback) i = min(i, i_max);

        const unsigned char *bxi = (const unsigned char *)x + (size_t)(kbx0 + i * stride) * Q35_BYTES_Q2_K;
        const int x_ql_0 = q35_get_int_b2_u(bxi + 16);   // qs[] at offset 16

#pragma unroll
        for (int l = 0; l < Q35_QR2_K; ++l) {
            const int k = (kqsx / 8) * 32 + l * 8 + kqsx % 8;
            const int x_qs_k = (x_ql_0 >> (2 * l)) & 0x03030303;
            x_qs[i * sram_stride + k] = x_qs_k;
        }

        // scales[kqsx] packs (scale & 0x0F, min >> 4) for one 16-value group.
        const int sc_m = bxi[80 + kqsx];
        const half2 dm = *(const half2 *)(bxi);          // (d, min)
        const half2 sc2 = __floats2half2_rn((float)(sc_m & 0x0F), (float)(sc_m >> 4));
        const half2 x_dm_ik = __hmul2(dm, sc2);
        x_dm[i * sram_stride + kqsx] = x_dm_ik;
    }
}

// ---------------------------------------------------------------------------
// q3_K (mmq-load-tiles.cuh:598)
// ---------------------------------------------------------------------------
template <int nthreads, int I, bool fallback>
__device__ __forceinline__ void load_tiles_q3_K(const char *__restrict__ x,
                                                int *__restrict__ x_tile,
                                                int kbx0, int i_max, int stride) {
    constexpr int sram_stride = q35_sram_stride(Q35_SRAM_Q3_K);
    int   *x_qs = (int *)x_tile;
    float *x_df = (float *)(x_qs + 2 * Q35_MMQ_TILE_NE_K);
    // NOTE: the MMA branch of q3_K writes ONLY x_df (2*NE_K floats); there is
    // no x_sc store.  Upstream's x_sc store lives in the dp4a #else branch
    // (mmq-load-tiles.cuh:677) and its separate d-only loop is guarded by
    // `#if !(MMA)` (:687) -- both are absent here.  The stride 84 = 64 qs + 32
    // df + 4 pad has nothing left over for packed scales, which is the
    // structural reason the two are not interchangeable.

    constexpr int threads_per_row = Q35_MMQ_ITER_K / (4 * Q35_QR3_K);   // 16
    constexpr int nrows = WARP / threads_per_row;                        // 2
    constexpr int nwarps = nthreads / WARP;
    const int kqsx = threadIdx.x % threads_per_row;

#pragma unroll
    for (int i0 = 0; i0 < I; i0 += nrows * nwarps) {
        int i = i0 + threadIdx.y * nrows + threadIdx.x / threads_per_row;
        if (fallback) i = min(i, i_max);

        // block_q3_K layout: hmask[32] qs[64] scales[12] d(fp16) -> 110 B,
        // but the struct is padded to 110.   qs at 32, hmask at 0.
        const unsigned char *bxi = (const unsigned char *)x + (size_t)(kbx0 + i * stride) * Q35_BYTES_Q3_K;
        const int x_ql_0 = q35_get_int_b2_u(bxi + 32 + 2 * kqsx);              // qs[kqsx]
        const int x_qh_0 = q35_get_int_b2_u(bxi + 2 * (kqsx % (Q35_QI3_K / 2)))
                           >> (4 * (kqsx / (Q35_QI3_K / 2)));                   // hmask

#pragma unroll
        for (int l = 0; l < Q35_QR3_K; ++l) {
            const int k = (kqsx / 8) * 32 + l * 8 + kqsx % 8;
            const int x_ql_k =  (x_ql_0 >> (2 * l))       & 0x03030303;
            const int x_qh_k = ((x_qh_0 >>    l)  << 2) & 0x04040404;
            x_qs[i * sram_stride + k] = __vsub4(x_ql_k | x_qh_k, 0x04040404);
        }
    }

    // 8-bit signed sub-block scales -> floats, staged in x_df then packed into
    // the x_sc tail as ints (one int per 4 sub-blocks of one row).
    constexpr int rows_per_warp = WARP / 4;   // 8
#pragma unroll
    for (int i0 = 0; i0 < I; i0 += nwarps * rows_per_warp) {
        int i = i0 + threadIdx.y * rows_per_warp + threadIdx.x / 4;
        if (fallback) i = min(i, i_max);

        const unsigned char *bxi = (const unsigned char *)x + (size_t)(kbx0 + i * stride) * Q35_BYTES_Q3_K;
        const int ksc = threadIdx.x % 4;

        const int ksc_low  = ksc % (Q35_QI3_K / 8);
        const int shift_low = 4 * (ksc / (Q35_QI3_K / 8));
        const int sc_low = (q35_get_int_b2_u(bxi + 96 + 2 * ksc_low) >> shift_low) & 0x0F0F0F0F;

        const int ksc_high = Q35_QI3_K / 8;
        const int shift_high = 2 * ksc;
        const int sc_high = ((q35_get_int_b2_u(bxi + 96 + 2 * ksc_high) >> shift_high) << 4) & 0x30303030;

        const int sc = __vsub4(sc_low | sc_high, 0x20202020);

        const float d = *(const half *)(bxi + 108);
        const int8_t *sc8 = (const int8_t *)&sc;

#pragma unroll
        for (int l = 0; l < (int)sizeof(int); ++l) {
            x_df[i * sram_stride + sizeof(int) * ksc + l] = d * sc8[l];
        }
    }
}

// ---------------------------------------------------------------------------
// q4_K (mmq-load-tiles.cuh:711)
// ---------------------------------------------------------------------------
// unpack_scales_q45_K: pull the 6-bit (scale, min) of sub-block `k` out of the
// interleaved 12-byte scales array.  Verbatim from common.cuh.
__device__ __forceinline__ int q35_unpack_scales_q45_K(const int *scales, int ksc) {
    // Verbatim from mmq-load-tiles.cuh:703, including its index tricks:
    //   ksc == 0: sc0..sc3   ksc == 1: sc4..sc7
    //   ksc == 2:  m0..m3    ksc == 3:  m4..m7
    return ((scales[(ksc % 2) + (ksc != 0)] >> (4 * (ksc & (ksc / 2)))) & 0x0F0F0F0F) |  // lower 4 bits
           ((scales[ksc / 2]              >> (2 * (ksc % 2)))       & 0x30303030);      // upper 2 bits
}

template <int nthreads, int I, bool fallback>
__device__ __forceinline__ void load_tiles_q4_K(const char *__restrict__ x,
                                                int *__restrict__ x_tile,
                                                int kbx0, int i_max, int stride) {
    constexpr int sram_stride = q35_sram_stride(Q35_SRAM_Q8_1);
    int   *x_qs = (int *)x_tile;
    half2 *x_dm = (half2 *)(x_qs + 2 * Q35_MMQ_TILE_NE_K);

    constexpr int threads_per_row = Q35_MMQ_ITER_K / (4 * Q35_QR4_K);   // 256/8 = 32
    constexpr int nrows = WARP / threads_per_row;                        // 1
    constexpr int nwarps = nthreads / WARP;
    const int txi = WARP > threads_per_row ? threadIdx.x % threads_per_row : threadIdx.x;

#pragma unroll
    for (int i0 = 0; i0 < I; i0 += nrows * nwarps) {
        int i = i0 + (nrows == 1 ? threadIdx.y : threadIdx.y * nrows + threadIdx.x / threads_per_row);
        if (fallback) i = min(i, i_max);

        // block_q4_K: dm(half2) scales[12] qs[128] = 144 B
        const unsigned char *bxi = (const unsigned char *)x + (size_t)(kbx0 + i * stride) * Q35_BYTES_Q4_K;
        const int qs0 = q35_get_int_b4((const signed char *)(bxi + 16 + 4 * txi));

        x_qs[i * sram_stride + 16 * (txi / 8) + txi % 8 + 0] = (qs0 >> 0) & 0x0F0F0F0F;
        x_qs[i * sram_stride + 16 * (txi / 8) + txi % 8 + 8] = (qs0 >> 4) & 0x0F0F0F0F;
    }

    // Scale region: half2 (d*sc, -d*min) for each of the 32 sub-blocks.
    constexpr int rows_per_warp = WARP / 2;   // 16
#pragma unroll
    for (int i0 = 0; i0 < I; i0 += nwarps * rows_per_warp) {
        int i = i0 + threadIdx.y * rows_per_warp + threadIdx.x / 2;
        if (fallback) i = min(i, i_max);

        const unsigned char *bxi = (const unsigned char *)x + (size_t)(kbx0 + i * stride) * Q35_BYTES_Q4_K;

        const int *scales = (const int *)(bxi + 4);
        const int ksc = threadIdx.x % 2;

        const int sc32 = q35_unpack_scales_q45_K(scales, ksc + 0);
        const int  m32 = q35_unpack_scales_q45_K(scales, ksc + 2);

        const uint8_t *sc8 = (const uint8_t *)&sc32;
        const uint8_t *m8  = (const uint8_t *)&m32;

        // Verbatim upstream: dm * make_half2(1.0f, -1.0f) negates the min
        // half, then each (sc, m) pair is scaled by it.
        const half2 dm = __hmul2(*(const half2 *)(bxi), __floats2half2_rn(1.0f, -1.0f));
#pragma unroll
        for (int l = 0; l < (int)sizeof(int); ++l) {
            x_dm[i * sram_stride + sizeof(int) * ksc + l] =
                __hmul2(dm, __floats2half2_rn((float)sc8[l], (float)m8[l]));
        }
    }
}

// ---------------------------------------------------------------------------
// q5_K (mmq-load-tiles.cuh:822)
// ---------------------------------------------------------------------------
template <int nthreads, int I, bool fallback>
__device__ __forceinline__ void load_tiles_q5_K(const char *__restrict__ x,
                                                int *__restrict__ x_tile,
                                                int kbx0, int i_max, int stride) {
    constexpr int sram_stride = q35_sram_stride(Q35_SRAM_Q8_1);
    int   *x_qs = (int *)x_tile;
    half2 *x_dm = (half2 *)(x_qs + 2 * Q35_MMQ_TILE_NE_K);

    constexpr int threads_per_row = Q35_MMQ_ITER_K / (4 * Q35_QR5_K);   // 32
    constexpr int nrows = WARP / threads_per_row;                        // 1
    constexpr int nwarps = nthreads / WARP;
    const int txi = WARP > threads_per_row ? threadIdx.x % threads_per_row : threadIdx.x;

#pragma unroll
    for (int i0 = 0; i0 < I; i0 += nrows * nwarps) {
        int i = i0 + (nrows == 1 ? threadIdx.y : threadIdx.y * nrows + threadIdx.x / threads_per_row);
        if (fallback) i = min(i, i_max);

        // block_q5_K: dm(half2) scales[12] qh[32] qs[128] = 176 B
        const unsigned char *bxi = (const unsigned char *)x + (size_t)(kbx0 + i * stride) * Q35_BYTES_Q5_K;
        const int qs0 = q35_get_int_b4((const signed char *)(bxi + 48 + 4 * txi));
        const int qh = q35_get_int_b4((const signed char *)(bxi + 16 + txi % 8 * 4));
        const int qh0 = ((qh >> (txi / 8)) << 4) & 0x10101010;
        const int qh1 = qh0 >> 4;

        x_qs[i * sram_stride + 16 * (txi / 8) + txi % 8 + 0] = ((qs0 >> 0) & 0x0F0F0F0F) | qh0;
        x_qs[i * sram_stride + 16 * (txi / 8) + txi % 8 + 8] = ((qs0 >> 4) & 0x0F0F0F0F) | qh1;
    }

    constexpr int rows_per_warp = WARP / 2;
#pragma unroll
    for (int i0 = 0; i0 < I; i0 += nwarps * rows_per_warp) {
        int i = i0 + threadIdx.y * rows_per_warp + threadIdx.x / 2;
        if (fallback) i = min(i, i_max);

        const unsigned char *bxi = (const unsigned char *)x + (size_t)(kbx0 + i * stride) * Q35_BYTES_Q5_K;
        const int *scales = (const int *)(bxi + 4);
        const int ksc = threadIdx.x % 2;

        const int sc32 = q35_unpack_scales_q45_K(scales, ksc + 0);
        const int  m32 = q35_unpack_scales_q45_K(scales, ksc + 2);

        const uint8_t *sc8 = (const uint8_t *)&sc32;
        const uint8_t *m8  = (const uint8_t *)&m32;

        const half2 dm = __hmul2(*(const half2 *)(bxi), __floats2half2_rn(1.0f, -1.0f));
#pragma unroll
        for (int l = 0; l < (int)sizeof(int); ++l) {
            x_dm[i * sram_stride + sizeof(int) * ksc + l] =
                __hmul2(dm, __floats2half2_rn((float)sc8[l], (float)m8[l]));
        }
    }
}

// ---------------------------------------------------------------------------
// q6_K (mmq-load-tiles.cuh:946)
// ---------------------------------------------------------------------------
template <int nthreads, int I, bool fallback>
__device__ __forceinline__ void load_tiles_q6_K(const char *__restrict__ x,
                                                int *__restrict__ x_tile,
                                                int kbx0, int i_max, int stride) {
    constexpr int sram_stride = q35_sram_stride(Q35_SRAM_Q6_K);
    int   *x_qs = (int *)x_tile;
    float *x_df = (float *)(x_qs + 2 * Q35_MMQ_TILE_NE_K);
    int   *x_sc = (int *)(x_df + 2 * Q35_MMQ_TILE_NE_K);   // int8 sub-block scales, packed

    constexpr int threads_per_row = Q35_MMQ_ITER_K / (4 * Q35_QR6_K);   // 32
    constexpr int nrows = WARP / threads_per_row;                       // 1
    constexpr int nwarps = nthreads / WARP;
    const int txi = WARP > threads_per_row ? threadIdx.x % threads_per_row : threadIdx.x;

#pragma unroll
    for (int i0 = 0; i0 < I; i0 += nrows * nwarps) {
        int i = i0 + (nrows == 1 ? threadIdx.y : threadIdx.y * nrows + threadIdx.x / threads_per_row);
        if (fallback) i = min(i, i_max);

        // block_q6_K: ql[128] qh[64] scales[16] d(fp16) = 210 B
        const unsigned char *bxi = (const unsigned char *)x + (size_t)(kbx0 + i * stride) * Q35_BYTES_Q6_K;
        const int ql = q35_get_int_b4((const signed char *)(bxi + 4 * txi));
        const int qh = q35_get_int_b4((const signed char *)(bxi + 128 + 2 * (txi % 16)));

        const int is = txi / 16;
        const int qh0 = ((qh >> (4 * is)) << 4) & 0x30303030;
        const int qh1 = qh0 >> 4;
        const int ql0 = (ql >> 0) & 0x0F0F0F0F;
        const int ql1 = (ql >> 4) & 0x0F0F0F0F;

        // Upstream kq0/kq1 (mmq-load-tiles.cuh:986): the two halves of a
        // thread's 8-value group land QI6_K/2 apart, not 8 apart.
        const int kq0 = 2 * txi - txi % (Q35_QI6_K / 2) + 0;
        const int kq1 = 2 * txi - txi % (Q35_QI6_K / 2) + Q35_QI6_K / 2;

        x_qs[i * sram_stride + kq0] = __vsub4(ql0 | qh0, 0x20202020);
        x_qs[i * sram_stride + kq1] = __vsub4(ql1 | qh1, 0x20202020);
    }

    // Super-block scale d, one float per row at the head of the scale region.
#pragma unroll
    for (int i0 = 0; i0 < I; i0 += nwarps * WARP) {
        int i = (i0 + threadIdx.y * WARP + threadIdx.x) % I;
        if (fallback) i = min(i, i_max);

        const unsigned char *bxi = (const unsigned char *)x + (size_t)(kbx0 + i * stride) * Q35_BYTES_Q6_K;
        x_df[i * sram_stride] = *(const half *)(bxi + 208);
    }

    // int8 sub-block scales, 2 bytes per thread into x_sc.
    constexpr int rows_per_warp = WARP / 4;   // 8
#pragma unroll
    for (int i0 = 0; i0 < I; i0 += nwarps * rows_per_warp) {
        int i = (i0 + threadIdx.y * rows_per_warp + threadIdx.x / (Q35_MMQ_TILE_NE_K / 8)) % I;
        if (fallback) i = min(i, i_max);

        // Upstream advances the block pointer by (threadIdx.x % 4)/4 == 0 when
        // the remainder is over 4 -- i.e. it never advances; every thread of
        // the group reads the SAME block and picks a different 2-byte window.
        const unsigned char *bxi = (const unsigned char *)x + (size_t)(kbx0 + i * stride) * Q35_BYTES_Q6_K;
        x_sc[i * sram_stride + threadIdx.x % 4] =
            q35_get_int_b2_u(bxi + 192 + (threadIdx.x % (Q35_QI6_K / 8)));
    }
}

}  // namespace q35_mmq

#endif  // Q35_MMQ_LOAD_TILES_K_CUH
