/*
 * @file iq_tables.cuh
 * @brief Declarations of the IQ lookup tables for device code.
 *
 * The data lives in exactly ONE place: `src/iq_tables.h` (repo root), included
 * by `iq_tables.cu` with Q35_TABLES_CUDA defined so it becomes
 *
 *     __device__ <type> <name>[<n>] = { ... };
 *
 * Every other .cu includes *this* header and sees
 *
 *     extern __device__ <type> <name>[<n>];
 *
 * which is the same object, resolved by the device link step.  That split is
 * what keeps the .so from carrying two copies of the ~96 KB codebooks: an
 * initialized `__device__` *definition* in a header included by N translation
 * units produces N separate objects, and nothing merges them.
 *
 * Why global memory rather than `__constant__`: the grids are indexed per
 * element (`iq3xxs_grid[qs[j]]`, `iq1s_grid[...]`), so a warp's lanes read
 * different offsets.  The constant cache serialises such an access into one
 * transaction per distinct address -- the worst case for these tables -- while
 * the ordinary L1/L2 path handles a divergent gather.  The tables are ~96 KB,
 * small enough to stay hot in L1 without meaningfully displacing weights in L2.
 *
 * Why this shim exists instead of each .cu writing `#include "../../iq_tables.h"`
 * itself: the path traversal stays in one file, and a host C compiler including
 * it gets a clear #error rather than a silently empty translation unit.
 */

#ifndef Q35_CUDA_IQ_TABLES_CUH
#define Q35_CUDA_IQ_TABLES_CUH

#ifndef __CUDACC__
#error "iq_tables.cuh is device-only (__device__ __constant__); include ../../iq_tables.h from host code"
#endif

#include <cstdint>

extern __device__ uint8_t  kmask_iq2xs[8];
extern __device__ uint8_t  ksigns_iq2xs[128];
extern __device__ uint64_t ksigns64[128];
extern __device__ uint64_t iq2xxs_grid[256];
extern __device__ uint64_t iq2xs_grid[512];
extern __device__ uint64_t iq2s_grid[1024];
extern __device__ uint32_t iq3xxs_grid[256];
extern __device__ uint32_t iq3s_grid[512];
extern __device__ int8_t   kvalues_iq4nl[16];
extern __device__ int8_t   kvalues_fp4[16];
extern __device__ uint64_t iq1s_grid[2048];
extern __device__ uint8_t  iq_pow3[6];

#endif /* Q35_CUDA_IQ_TABLES_CUH */
