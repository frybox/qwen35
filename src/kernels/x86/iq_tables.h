/*
 * IQ-series lookup tables: declarations only.
 *
 * The data lives in src/iq_tables.zig, generated from llama.cpp's
 * ggml/src/ggml-common.h by scripts/gen_iq_tables.py.  Zig emits these as
 * global objects (`pub export const`), so every C kernel here shares one copy
 * of the data instead of carrying its own.  This is the C-side equivalent of
 * llama.cpp's GGML_TABLE_BEGIN/GGML_TABLE_END mechanism (ggml-common.h:469-502),
 * except there is no per-TU copy: the symbol is defined once.
 *
 * Do not add a table definition to this header.  If a kernel needs a new
 * table, add it to src/iq_tables.zig (canonical data) or derive it there
 * (comptime function), then declare it here.
 */

#ifndef IQ_TABLES_H
#define IQ_TABLES_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* Canonical tables, one value per upstream ggml-common.h entry. */
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

/* Derived tables, computed at Zig comptime from the canonical ones. */
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
