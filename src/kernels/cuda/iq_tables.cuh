/*
 * @file iq_tables.cuh
 * @brief Declarations of the IQ lookup tables for device code.
 *
 * The data lives in exactly ONE place: `src/iq_tables.h` (repo root), included
 * by `iq_tables.cu` with Q35_TABLES_CUDA defined so it becomes
 *
 *     __device__ __constant__ const <type> <name>[<n>] = { ... };
 *
 * Every other .cu includes *this* header and sees
 *
 *     extern __device__ __constant__ <type> <name>[<n>];
 *
 * which is the same object, resolved by the device link step.  That split is
 * what keeps the .so from carrying two copies of the ~96 KB codebooks: a
 * `__constant__` *definition* in a header included by N translation units
 * produces N separate constant banks, and nothing merges them.
 *
 * Why __constant__ rather than __global__: the IQ codebooks are small (~96 KB
 * for all twelve tables), read-only, index-computed, and read once per
 * sub-block.  On sm_89 a constant-bank load is a couple of cycles (the hardware
 * broadcasts a warp-uniform address), against tens for a global load, and the
 * tables do not compete with the weights for L2.
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

extern __device__ __constant__ uint8_t  kmask_iq2xs[8];
extern __device__ __constant__ uint8_t  ksigns_iq2xs[128];
extern __device__ __constant__ uint64_t ksigns64[128];
extern __device__ __constant__ uint64_t iq2xxs_grid[256];
extern __device__ __constant__ uint64_t iq2xs_grid[512];
extern __device__ __constant__ uint64_t iq2s_grid[1024];
extern __device__ __constant__ uint32_t iq3xxs_grid[256];
extern __device__ __constant__ uint32_t iq3s_grid[512];
extern __device__ __constant__ int8_t   kvalues_iq4nl[16];
extern __device__ __constant__ int8_t   kvalues_fp4[16];
extern __device__ __constant__ uint64_t iq1s_grid[2048];
extern __device__ __constant__ uint8_t  iq_pow3[6];

#endif /* Q35_CUDA_IQ_TABLES_CUH */
