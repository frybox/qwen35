/*
 * @file iq_tables.cuh
 * @brief Device-side access to the twelve IQ lookup tables.
 *
 * THE DATA HAS EXACTLY ONE DEFINITION, IN THE EXECUTABLE
 * ------------------------------------------------------
 * The twelve codebooks are written down once, in `src/iq_tables.zig` (generated
 * by scripts/gen_iq_tables.py), as `pub export const` arrays.  They are
 * ordinary *host* objects of the `qwen35` executable; the x86 kernels link
 * against them through `extern const` declarations
 * (src/kernels/x86/iq_tables.h), and so does the CUDA upload code now that the
 * .cu objects are linked into that same executable.  AGENTS.md §1.0 is explicit
 * that this is the only copy -- a second definition, whether a `= { 0x.., .. }`
 * table in a .cu or a `__device__` array, is forbidden.
 *
 * SO HOW DOES DEVICE CODE READ THEM?
 * ----------------------------------
 * A CUDA kernel cannot dereference a host pointer, so at start-up
 * `q35_launch_init_tables()` (src/kernels/cuda/iq_tables.cu) copies the twelve
 * host arrays into one device buffer and points a device global at each.  Every
 * kernel that needs a table reads it through a pointer that this header
 * declares:
 *
 *     extern __device__ const uint8_t *kmask_iq2xs;   // a POINTER, not an array
 *
 * The pointer keeps the table's *name*, so every existing call site --
 * `iq1s_grid[idx]`, `ksigns_iq2xs[(aux1 >> 7*l) & 127]`, ... -- compiles
 * unchanged and keeps meaning exactly what it meant before: index the table.
 * That is deliberate: the kernels' numeric expressions are a line-by-line port
 * of ggml-quants.c / ggml-cuda, so a mechanical rewrite of 64 index expressions
 * would be unreviewable and would invite `ksigns_iq2xs` / `keven_signs_q2xs`
 * (two different sign forms, see AGENTS.md §4) to be crossed by accident.  Under
 * this scheme the diff of convert.cu / vecdotq.cuh / mmvq.cu is empty.
 *
 * WHERE THE HOST ADDRESSES COME FROM
 * ---------------------------------
 * From `src/kernels/cuda/iq_tables_host.c`, a plain C file that declares the
 * arrays with their real (unprefixed) names and exposes them through the small
 * API declared at the bottom of this header:
 *
 *     void q35_host_tables(const q35_host_table **out, int *count);
 *
 * Record i is the host address and byte count of the i-th table of
 * iq_tables_list.inc, which is also the order of the twelve pointers below --
 * that .inc is the single place where a table name is associated with its
 * geometry, so the two sides cannot drift.  (`iq_tables_host.c` exists as a
 * separate C translation unit because a host `extern const uint64_t iq1s_grid`
 * declared next to the `__device__` pointer of the same name in one CUDA
 * translation unit is folded by nvcc into one HIDDEN host object -- see that
 * file's comment.)
 *
 * Device symbols contributed by this scheme: exactly the 12 pointers (96 bytes).
 * The ~40 KB of data is a heap buffer owned by iq_tables.cu, uploaded once.
 *
 * Why the pointers are `__device__` globals rather than a data-carrying
 * `__constant__` array: a device *array* would be a symbol definition carrying a
 * copy of the data, which is what this header exists to avoid.
 *
 * Why global memory rather than `__constant__`: the grids are indexed per
 * element (`iq3xxs_grid[qs[j]]`), so a warp's lanes read different offsets.  The
 * constant cache serialises such an access into one transaction per distinct
 * address -- the worst case for these tables -- while the ordinary L1/L2 path
 * handles a divergent gather.  The tables are ~40 KB, small enough to stay hot
 * in L1 without meaningfully displacing weights in L2.
 *
 * Why a shim header instead of each .cu declaring the pointers itself: the
 * declarations must match the definitions in iq_tables.cu exactly, and one
 * wrong `const` would be an nvlink error far from the edit.  A host C compiler
 * including this file gets a clear #error rather than a silently empty
 * translation unit.
 */

#ifndef Q35_CUDA_IQ_TABLES_CUH
#define Q35_CUDA_IQ_TABLES_CUH

#ifdef __CUDACC__
#include <cstdint>

/* The single table list (name, element type, element count), shared with
 * iq_tables_host.c -- which is where the host addresses and byte counts come
 * from.  Expanding it here is what keeps this header free of table names. */
#include "iq_tables_list.inc"

/* The 12 device table pointers: element type and name from iq_tables_list.inc
 * (shared with iq_tables_host.c), so this header spells no table name at all.
 *
 * DECLARED here, DEFINED in src/kernels/cuda/iq_tables.cu, assigned by
 * q35_launch_init_tables() before the first kernel launch.
 *
 * WHY THE SYMBOL NAME IS `q35_d_<table>` AND NOT JUST `<table>`
 * -----------------------------------------------------------
 * A `__device__ const T *iq1s_grid;` definition emits a HOST object too: 8 bytes
 * in .bss (the device-pointer stub cudaMemcpyToSymbol writes through), and
 * `readelf -sW` shows it as `OBJECT GLOBAL HIDDEN ... 8 iq1s_grid`.  Now that the
 * .cu objects are linked into the SAME executable as src/iq_tables.zig, that stub
 * collides with Zig's `pub export const iq1s_grid[2048]` and the linker stops:
 *
 *   $ nvcc -rdc=true -o probe probe.cu iq.o tables.o
 *   /usr/bin/ld: tables.o:(.data.rel.ro+0x6280): multiple definition of `iq1s_grid';
 *                 iq.o:(.bss+0x20): first defined here
 *
 * (measured; forcing it through with --allow-multiple-definition links but
 * segfaults at run time, so it is not a packaging detail).  The kernel library
 * used to live in its own libqwen35-cuda-kernels.so, where `extern __device__`
 * only produced an undefined reference and there was nothing to collide with.
 *
 * So the PTX symbol is `q35_d_iq1s_grid`, `q35_d_iq_pow3`, ... and the host-side
 * `#define`s at the bottom of this block restore the plain table name for every
 * device-side use site, which is why convert.cu / vecdotq.cuh / mmvq.cu are
 * unchanged and still read `iq1s_grid[idx]`.  The macro is scoped to
 * __CUDACC__ translation units and only rewrites the identifier inside device
 * code that included this header: nothing else in those .cu files uses those
 * names (verified: 22 references, all table indexing).
 *
 * The defining translation unit sets Q35_IQ_TABLES_DEFINING_POINTERS before
 * including this header, so `extern __device__ const T *p;` is not emitted next
 * to the definition -- that pair is "redefinition" whenever nvcc runs WITHOUT
 * -rdc (it treats the extern declaration as a static definition, warning
 * #20044-D).  The guard is what lets `zig build cuda-check` (no -rdc) compile
 * both this file and its consumers.
 */
#ifndef Q35_IQ_TABLES_DEFINING_POINTERS
#define Q35_DEV_PTR_DECL(name, elem, count) \
    extern __device__ const elem *q35_d_##name;
Q35_IQ_TABLE_LIST(Q35_DEV_PTR_DECL)
#undef Q35_DEV_PTR_DECL
#endif

/* Restore the plain table names for device code.  This is the one place in the
 * backend that repeats the twelve identifiers of iq_tables_list.inc -- a macro
 * cannot #define another macro, so the alias list has to be written out.  It is
 * a NAME mapping, not a second "table <-> address/size" correspondence (that
 * lives only in the .inc); the scripts/ check for drift is `zig build test`
 * failing to compile every consumer if a name is missing, which is loud. */
#define kmask_iq2xs      q35_d_kmask_iq2xs
#define ksigns_iq2xs     q35_d_ksigns_iq2xs
#define ksigns64         q35_d_ksigns64
#define iq2xxs_grid      q35_d_iq2xxs_grid
#define iq2xs_grid       q35_d_iq2xs_grid
#define iq2s_grid        q35_d_iq2s_grid
#define iq3xxs_grid      q35_d_iq3xxs_grid
#define iq3s_grid        q35_d_iq3s_grid
#define kvalues_iq4nl    q35_d_kvalues_iq4nl
#define kvalues_fp4      q35_d_kvalues_fp4
#define iq1s_grid        q35_d_iq1s_grid
#define iq_pow3          q35_d_iq_pow3

extern "C" {

/* Host side of the upload: the twelve arrays DEFINED in src/iq_tables.zig, as
 * seen from the executable's own link.  Implemented in iq_tables_host.c (plain
 * C, no device namespace); that file must be on the same link line as the .cu
 * objects or this reference stays undefined. */
typedef struct {
    const void *host;               /* address of the table in this executable */
    unsigned long long bytes;       /* its size in bytes, straight from sizeof */
} q35_host_table;

/* Fills `*out` with a pointer to a static array of `*count` records, one per
 * table, in iq_tables_list.inc order; `*count == Q35_HOST_TABLE_COUNT`.  Either
 * out-parameter may be null.  The records are host memory owned by
 * iq_tables_host.c -- copy the addresses out, do not keep the pointer past the
 * upload (deinit re-arms init, so a later call re-reads them). */
void q35_host_tables(const q35_host_table **out, int *count);

/* The upload itself, implemented in iq_tables.cu; the authoritative
 * declarations are in api.h (with the rest of the q35_launch_* family) -- these
 * repeats are for device-side-only translation units and tests that include this
 * header without api.h.  Keep the signatures byte-identical to api.h. */
int q35_launch_init_tables(void *stream);
unsigned long long q35_iq_tables_bytes(void);
int q35_iq_tables_ready(void);
void q35_launch_deinit_tables(void);

} /* extern "C" */

#else /* !__CUDACC__ */

/* Plain C/C++ callers of the API above (a test, say) can include this header for
 * `q35_host_table` / `q35_host_tables` -- the device declarations are skipped,
 * since a non-CUDA compiler cannot parse `__device__`. */
#include <stddef.h>

typedef struct {
    const void *host;
    unsigned long long bytes;
} q35_host_table;

void q35_host_tables(const q35_host_table **out, int *count);

#endif /* __CUDACC__ */

#endif /* Q35_CUDA_IQ_TABLES_CUH */
