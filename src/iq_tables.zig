//! The IQ lookup tables, bridged from their single C definition.
//!
//! The data itself lives in exactly one place: `src/iq_tables.h`, which every
//! consumer includes with a different macro defined --
//!
//!   * `Q35_TABLES_CUDA`  -> `__device__ __constant__`  (src/kernels/cuda/)
//!   * (no macro)         -> plain `const`               (Zig, via translate-c)
//!   * nothing defined    -> plain `const`               (this @cImport)
//!
//! so the Zig half and the C/CUDA kernels read the *same* objects.  This module
//! exists only to (a) re-export those symbols under the names the Zig side uses
//! and (b) *derive* the four tables that are pure functions of them.  Deriving
//! rather than storing is deliberate: it is what lets `iq_table_tests.zig`
//! cross-check two independent forms of the sign tables instead of comparing a
//! hand-maintained copy against itself.
//!
//! Before this existed the tables were a generated `src/iq_tables.zig` plus a
//! hand-copied `src/kernels/cuda/iq_tables.cuh` -- three copies of a 12-table
//! codebook (Zig source, CUDA copy, generated binding), which is exactly how
//! the Zig oracle and the C kernels drifted apart in the past.

const std = @import("std");

/// src/iq_tables.h translated to Zig at build time.
///
/// Zig 0.17 removed the `@cImport` builtin; `build.zig` runs a translate-c step
/// over the header and this imports its output.  The translated `const`
/// definitions come out as `pub export const`, so **this is the definition** the
/// whole program uses: the x86 kernels reach it through the `extern const`
/// declarations in src/kernels/x86/iq_tables.h, and the CUDA kernels get their
/// own `__device__ __constant__` instantiation of the same header (device code
/// cannot reference host objects).
///
/// So the data is written once, in src/iq_tables.h, and instantiated twice: once
/// here for host+Zig, once in the device image.  The x86 side deliberately does
/// NOT compile its own copy: a first attempt at a C-side definition unit produced
/// two symbols for one table, which the identity test in src/iq_table_tests.zig caught.
const c = @import("iq_tables_c");

// ---------------------------------------------------------------------------
// Canonical tables -- defined ONCE, by the translate-c output of
// src/iq_tables.h.  These aliases exist so the rest of the code says
// `tables.iq2xxs_grid` rather than `tables.c.iq2xxs_grid`.
// ---------------------------------------------------------------------------

pub const kmask_iq2xs = c.kmask_iq2xs;
pub const ksigns_iq2xs = c.ksigns_iq2xs;
pub const ksigns64 = c.ksigns64;
pub const iq2xxs_grid = c.iq2xxs_grid;
pub const iq2xs_grid = c.iq2xs_grid;
pub const iq2s_grid = c.iq2s_grid;
pub const iq3xxs_grid = c.iq3xxs_grid;
pub const iq3s_grid = c.iq3s_grid;
pub const kvalues_iq4nl = c.kvalues_iq4nl;
pub const kvalues_fp4 = c.kvalues_fp4;
pub const iq1s_grid = c.iq1s_grid;
pub const iq_pow3 = c.iq_pow3;

// The derived tables below are computed from the canonical ones above.

// ---------------------------------------------------------------------------
// Derived tables -- computed from the canonical tables above at comptime.
// ---------------------------------------------------------------------------

// keven_signs_q2xs: the SIMD-only int8 +-1 expansion of ksigns_iq2xs.  There is
// no such table in upstream ggml-common.h; every x86/riscv kernel in llama.cpp
// hand-writes it (ggml-cpu/arch/x86/quants.c:2688).  Generating it here removes
// the class of bug where the Zig oracle and the C kernel use different sign
// tables (see the kerniq runtime-quantization failures).
pub export const keven_signs_q2xs: [1024]i8 align(64) = blk: {
    @setEvalBranchQuota(100000);
    var t: [1024]i8 = undefined;
    for (ksigns_iq2xs, 0..) |mask, i| {
        for (0..8) |b| {
            t[i * 8 + b] = if (mask & (1 << @as(u3, @intCast(b))) != 0) -1 else 1;
        }
    }
    break :blk t;
};

// ksigns64: the same sign mask as keven_signs_q2xs, packed into one u64 per
// entry, with -1 encoded as 0xFF and +1 as 0x00.  Upstream ships this table
// (ggml-common.h); we derive it and assert against the upstream copy below.
pub export const ksigns64_derived: [128]u64 align(64) = blk: {
    @setEvalBranchQuota(100000);
    var t: [128]u64 = undefined;
    for (ksigns_iq2xs, 0..) |mask, i| {
        var w: u64 = 0;
        for (0..8) |b| {
            if (mask & (1 << @as(u3, @intCast(b))) != 0) w |= 0xFF << @as(u64, b * 8);
        }
        t[i] = w;
    }
    break :blk t;
};

// int32 mirror of kvalues_iq4nl for the AVX2 hot path in kernels32.c (widening
// is free there, so the table is stored as i32 to avoid a per-element cvt).
pub export const k32_iq4nl_tab: [16]i32 align(64) = blk: {
    var t: [16]i32 = undefined;
    for (kvalues_iq4nl, 0..) |v, i| t[i] = @as(i32, v);
    break :blk t;
};

// f32 view of kvalues_fp4 for the nvfp4 kernel, which needs float lanes.
// kvalues_mxfp4 is an alias of kvalues_fp4 upstream (ggml-common.h:1129), so
// the int8 table above serves the mxfp4 path; the upper half of the table is
// already negative, so no sign flip is needed here.
pub export const k32_nvfp4_values: [16]f32 align(64) = blk: {
    var t: [16]f32 = undefined;
    for (kvalues_fp4, 0..) |v, i| t[i] = @as(f32, v);
    break :blk t;
};

// ---------------------------------------------------------------------------
// Comptime self-checks: these fail the build, not the test run.
// ---------------------------------------------------------------------------
comptime {
    @setEvalBranchQuota(100000);
    // 1. every derived sign lane must agree with its canonical mask bit
    for (ksigns_iq2xs, 0..) |mask, i| {
        for (0..8) |b| {
            const want: i8 = if (mask & (1 << @as(u3, @intCast(b))) != 0) -1 else 1;
            if (keven_signs_q2xs[i * 8 + b] != want) {
                @compileError("keven_signs_q2xs disagrees with ksigns_iq2xs");
            }
        }
    }
    // 2. derived ksigns64 must equal the upstream table byte for byte
    for (ksigns64, 0..) |want, i| {
        if (ksigns64_derived[i] != want) {
            @compileError("derived ksigns64 disagrees with the upstream table");
        }
    }
    // 3. IQ4_NL values must fit in int8_t (the C kernels cast them)
    for (k32_iq4nl_tab) |v| {
        if (v < -128 or v > 127) @compileError("kvalues_iq4nl entry does not fit in int8_t");
    }
    // 4. grid sizes the kernels index into must match the documented widths
    if (iq3s_grid.len != 512) @compileError("iq3s_grid must be 512 entries");
    if (iq3xxs_grid.len != 256) @compileError("iq3xxs_grid must be 256 entries");
    if (iq2xs_grid.len != 512) @compileError("iq2xs_grid must be 512 entries");
    if (iq2xxs_grid.len != 256) @compileError("iq2xxs_grid must be 256 entries");
    if (iq2s_grid.len != 1024) @compileError("iq2s_grid must be 1024 entries");
    if (iq1s_grid.len != 2048) @compileError("iq1s_grid must be 2048 entries");
}
