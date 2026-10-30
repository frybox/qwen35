/*
 * @file iq_tables.cu
 * @brief The single translation unit that *defines* the IQ lookup tables in
 *        device code.
 *
 * `src/iq_tables.h` (repo root) holds the one copy of the data.  Defining
 * Q35_TABLES_CUDA before including it makes its Q35_TAB_DECL macro expand to
 *
 *     __device__ <type> <name>[<n]> = { ... };
 *
 * so the same source text that the x86 kernels compile as plain `const` and
 * Zig's translate-c step turns into `pub export const` lands in device global
 * memory here.  Every other .cu includes `iq_tables.cuh` and sees only `extern
 * __device__` declarations, so the codebooks are instantiated exactly once in
 * libqwen35-cuda-kernels.so.
 *
 * Why global memory and not `__constant__`: the grids are indexed by a
 * per-element codebook byte (`iq3xxs_grid[qs[j]]`), so the lanes of a warp
 * touch different addresses and the constant cache serialises the load into
 * one transaction per distinct offset -- it is the slow path for these tables.
 * Plain `__device__` arrays are served by the normal L1/L2 path, which handles
 * a divergent gather far better.  Constant memory is only a win for
 * warp-uniform accesses (cf. `kvalues_iq4nl`, still read by index but small
 * enough to stay resident).
 *
 * Why this is a separate .cu and not a header: an initialized `__device__`
 * definition in a header would be compiled into every including translation
 * unit as its own object.  The tables are ~96 KB, so with two including .cu it
 * was ~192 KB of duplicated memory -- and, worse, a silent divergence risk if
 * only one .cu's copy was ever updated.  Declaration/definition split avoids
 * both.
 */

#define Q35_TABLES_CUDA 1
#include "../../iq_tables.h"
