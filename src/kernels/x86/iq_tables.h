/*
 * IQ-series lookup tables as seen by the x86 C kernels: declarations only.
 *
 * The data is written once, in the repo-root src/iq_tables.h.  For host code its
 * single definition is the translate-c output that Zig builds from that header
 * (see src/iq_tables.zig), which is why everything here is `extern const`: the
 * x86 kernels reference the Zig-emitted objects, they do not compile their own
 * copy.  CUDA cannot reference host objects, so the device image instantiates
 * the same header separately as __constant__ (src/kernels/cuda/iq_tables.cu).
 *
 * An earlier revision did compile a third copy here; the linker kept its
 * symbol and translate-c's apart, and the "one copy, not two" test in
 * src/iq_table_tests.zig caught it.  Do not reintroduce a definition here.
 *
 * The derived tables (keven_signs_q2xs, ksigns64_derived, k32_iq4nl_tab,
 * k32_nvfp4_values) are pure functions of the canonical tables; Zig computes
 * them at comptime and exports them, so they are declared `extern` below too.
 */

#ifndef IQ_TABLES_H
#define IQ_TABLES_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* Canonical tables: defined in src/iq_tables.h, compiled by build.zig's
 * qwen35_iq_tables unit, consumed by the kernels through these declarations. */
extern const uint8_t  kmask_iq2xs[8];      /* ggml-common.h:509 */
extern const uint8_t  ksigns_iq2xs[128];   /* ggml-common.h:513 */
extern const uint64_t ksigns64[128];       /* ggml-common.h:524 */
extern const uint64_t iq2xxs_grid[256];    /* ggml-common.h:560 */
extern const uint64_t iq2xs_grid[512];     /* ggml-common.h:627 */
extern const uint64_t iq2s_grid[1024];     /* ggml-common.h:758 */
extern const uint32_t iq3xxs_grid[256];    /* ggml-common.h:1017 */
extern const uint32_t iq3s_grid[512];      /* ggml-common.h:1052 */
extern const int8_t   kvalues_iq4nl[16];   /* ggml-common.h:1120 */
extern const int8_t   kvalues_fp4[16];     /* ggml-common.h:1126 */
extern const uint64_t iq1s_grid[2048];     /* ggml-common.h:1135 */
extern const uint8_t  iq_pow3[6];          /* ggml-cpu/quants.c:493 */

/* Derived tables, computed at Zig comptime from the canonical ones and exported
 * by src/iq_tables.zig. */
extern const int8_t   keven_signs_q2xs[1024];  /* sign expansion of ksigns_iq2xs */
extern const uint64_t ksigns64_derived[128];   /* must equal ksigns64 */
extern const int32_t  k32_iq4nl_tab[16];       /* int32 mirror of kvalues_iq4nl */
extern const float    k32_nvfp4_values[16];    /* f32 view of kvalues_fp4 */

/*
 * Cross-language spot check.  Returns the number of mismatches between the
 * tables the C kernels actually see and the expected upstream values.  Called
 * from the Zig test suite (src/iq_table_tests.zig), so a table that drifts in
 * either language fails a test rather than silently producing wrong dot
 * products.
 */
int iq_table_spotcheck(void);


#ifdef __cplusplus
}
#endif

#endif /* IQ_TABLES_H */
