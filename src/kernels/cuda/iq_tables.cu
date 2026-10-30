/*
 * @file iq_tables.cu
 * @brief The single translation unit that *defines* the IQ lookup tables in
 *        device code.
 *
 * `src/iq_tables.h` (repo root) holds the one copy of the data.  Defining
 * Q35_TABLES_CUDA before including it makes its Q35_TAB_DECL macro expand to
 *
 *     __device__ __constant__ const <type> <name>[<n]> = { ... };
 *
 * so the same source text that the x86 kernels compile as plain `const` and
 * Zig's translate-c step turns into `pub export const` lands in the constant
 * bank here.  Every other .cu includes `iq_tables.cuh` and sees only `extern
 * __device__ __constant__` declarations, so the codebooks are instantiated
 * exactly once in libqwen35-cuda-kernels.so.
 *
 * Why this is a separate .cu and not a header: an initialized `__constant__`
 * definition in a header produces one private constant bank per including
 * translation unit (nothing merges them across shared objects).  The tables are
 * ~96 KB, so with two including .cu it was ~192 KB of duplicated constant
 * memory -- and, worse, a silent divergence risk if only one .cu's budget was
 * ever exhausted.  Declaration/definition split avoids both.
 */

#define Q35_TABLES_CUDA 1
#include "../../iq_tables.h"
