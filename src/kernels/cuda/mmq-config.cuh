// MMQ tile configuration and J selection -- ported from llama.cpp
//   ggml/src/ggml-cuda/mmq.cuh:164-236   (struct + CASE macro + dispatch)
//   ggml/src/ggml-cuda/mmq.cu:189-225    (J_best search)
//   ggml/src/ggml-cuda/mmq-config-ampere.cuh (the per-type table)
//
// SCOPE.  qwen35 ships one CUDA backend for Turing/Ampere/Ada (sm_75+), so the
// config table here is the Ampere one verbatim, for the 22 quant types qwen35
// can load.  The AMD/pascal/GCN branches are deliberately absent: there is no
// second target to keep in sync, and every field those branches change
// (nthreads, occupancy, sram_layout) is a tuning constant, not semantics.
//
// The table is (type, J, fallback) -> config.  `fallback` is true when
// src0->ne[1] (the row count) is not a multiple of I, in which case load_tiles
// clamps the row index with min(i, i_max) instead of relying on the padding.
#ifndef Q35_MMQ_CONFIG_CUH
#define Q35_MMQ_CONFIG_CUH

#include "api.h"
#include "mmq-smem.cuh"

// Sentinel for "this (type, J, fallback) has no row", the analogue of upstream's
// GGML_TYPE_COUNT return.  q35 cannot use a type tag here because Q35_TT_* are
// preprocessor macros with no room for a sentinel value.
#define Q35_MMQ_TT_NONE (-1)

struct q35_mmq_config {
    int ttype;          // src0 type (Q35_TT_*)
    int nthreads;       // threads per CUDA block
    int occupancy;      // targeted occupancy for the MMA kernel
    int I;              // smem tile extent along the output-feature axis
    int J;              // smem tile extent along the token axis
    q35_mmq_sram_layout sram_layout;
    int K_vram;         // VRAM tile extent along K
    bool stream_k;
    bool fallback;

    __host__ __device__ constexpr int rows_per_warp() const {
        // mmq.cuh:179 -- Ampere branch (no AMD MFMA/WMMA).
        return (J >= 48 && J % 16 == 0) ? 32 : 16;
    }
};

// The Ampere table, restricted to what qwen35 can load.  Every entry upstream
// uses nthreads=256, occupancy=1, I=128, K_vram=MMQ_ITER_K=256, stream_k=true;
// the only per-entry variation is J and the sram layout, and the layout is a
// pure function of the type.  So the table collapses to: "every J in the
// type's J-set is valid, with stride from q35_sram_layout_of(type)".  Keeping
// the collapse explicit (instead of copying 205 lines) is safe BECAUSE the
// upstream rows are that regular -- Q35_MMQ_J_LIST below is the only
// information not derivable from the layout, and it is copied verbatim:
//   J with fallback=false : {8,16,24,32,40,48,64,80,96,112,128}
//   J with fallback=true  : {8,16,32,64,128}
// (8/16/24/32/40/48/64/80/96/112/128 and 8/16/32/64/128 respectively, for
// every type in the Ampere table.)
static constexpr int Q35_MMQ_J_LIST_NF[] = {8, 16, 24, 32, 40, 48, 64, 80, 96, 112, 128};
static constexpr int Q35_MMQ_J_LIST_F[]  = {8, 16, 32, 64, 128};
static constexpr int Q35_MMQ_J_LIST_NF_N = 11;
static constexpr int Q35_MMQ_J_LIST_F_N  = 5;

// Default config when no (type, J, fallback) row matches: mmq-config-ampere.cuh
// ends with GGML_TYPE_COUNT (== "no config"), which callers detect and skip.
// We use the same sentinel instead of guessing a J.
static constexpr __host__ __device__ q35_mmq_config q35_mmq_config_none() {
    return q35_mmq_config{Q35_MMQ_TT_NONE, 256, 1, 128, 64, Q35_SRAM_Q8_0, Q35_MMQ_ITER_K, false, true};
}

static constexpr __host__ __device__ bool q35_mmq_j_in(const int *list, int n, int J) {
    for (int i = 0; i < n; ++i) {
        if (list[i] == J) return true;
    }
    return false;
}

// (type, J, fallback) -> config.  Mirrors the CASE macro: a miss returns
// q35_mmq_config_none() so the callers can skip the entry, exactly like
// upstream's `if (config.type == GGML_TYPE_COUNT) continue;`.
static constexpr __host__ __device__ q35_mmq_config q35_mmq_get_config(int ttype, int J, bool fallback) {
    if (ttype == Q35_MMQ_TT_NONE) return q35_mmq_config_none();
    const bool j_ok = fallback ? q35_mmq_j_in(Q35_MMQ_J_LIST_F, Q35_MMQ_J_LIST_F_N, J)
                               : q35_mmq_j_in(Q35_MMQ_J_LIST_NF, Q35_MMQ_J_LIST_NF_N, J);
    if (!j_ok) return q35_mmq_config_none();
    return q35_mmq_config{ttype, 256, 1, Q35_MMQ_I, J,
                          q35_sram_layout_of(ttype), Q35_MMQ_ITER_K, true, fallback};
}

// Total smem for a config: ids scratch (only used by the MoE path, which
// qwen35 does not have, so nbs_ids = 0) + x tile + padded y tile.
// mmq.cuh:1415:
//     nbs_x = I * sram_stride * 4            (MMA data layout)
//     nbs_y = J * sizeof(block_q8_1_mmq)
//     total = nbs_ids + nbs_x + GGML_PAD(nbs_y, nthreads * sizeof(int))
static constexpr __host__ __device__ int q35_mmq_nbytes_shared(q35_mmq_config c) {
    const int nbs_x = c.I * q35_sram_stride(c.sram_layout) * (int)sizeof(int);
    const int nbs_y = c.J * (int)sizeof(q35_block_q8_1_mmq);
    return nbs_x + Q35_MMQ_PAD(nbs_y, c.nthreads * (int)sizeof(int));
}

// ---------------------------------------------------------------------------
// J_best (mmq.cu:204-225, verbatim algorithm)
// ---------------------------------------------------------------------------
// Walk J = 8,16,...,128; skip configs that do not exist or do not fit in the
// smem-per-block limit; keep the J that minimises the number of token tiles.
// Stop early once a single tile covers everything (ntiles_J_best == 1).
//
// The upstream loop increments J by 8 and *continues* past missing entries, so
// it does reach 24/40/48/56/... even though only some are in the fallback
// list.  We reproduce that by asking q35_mmq_get_config for each J.
struct q35_mmq_j_selection {
    int J;
    int nthreads;
    bool ok;
};

static inline q35_mmq_j_selection q35_mmq_pick_j(int ttype, int ncols, bool fallback,
                                                 int smem_per_block_limit) {
    q35_mmq_j_selection best{0, 0, false};
    int ntiles_best = 0x7fffffff;
    for (int J = 8; J <= 128 && ntiles_best > 1; J += 8) {
        const q35_mmq_config c = q35_mmq_get_config(ttype, J, fallback);
        if (c.ttype == Q35_MMQ_TT_NONE) continue;
        if (q35_mmq_nbytes_shared(c) > smem_per_block_limit) continue;
        const int ntiles = (ncols + c.J - 1) / c.J;
        if (ntiles < ntiles_best) {
            best.J = J;
            best.nthreads = c.nthreads;
            best.ok = true;
            ntiles_best = ntiles;
        }
    }
    return best;
}

// Padding of the q8_1 activation buffer, quoted from mmq.cu:229-231: a J-tile
// can read up to J-1 extra columns, and upstream rounds that up to a multiple
// of nthreads*sizeof(int) so the parallel load never faults.
static inline size_t q35_mmq_src1_q8_1_padding(int J_best, int nthreads_best) {
    const size_t chunk = (size_t)nthreads_best * sizeof(int);
    return ((J_best * sizeof(q35_block_q8_1_mmq) + chunk - 1) / chunk) * chunk;
}

#endif  // Q35_MMQ_CONFIG_CUH
