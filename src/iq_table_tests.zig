//! Cross-language table consistency tests.
//!
//! src/iq_tables.zig is the single definition of the IQ lookup tables; the C
//! kernels in src/kernels/x86/ reach the same objects through `extern const`
//! declarations (src/kernels/x86/iq_tables.h).  These tests check three things:
//!
//!   1. the canonical tables still match the upstream ggml values they were
//!      extracted from (spot values, per table);
//!   2. the derived tables (keven_signs_q2xs, ksigns64_derived, k32_iq4nl_tab,
//!      k32_nvfp4_values) are consistent with their canonical sources;
//!   3. the C kernels see exactly the same bytes as the Zig side, via
//!      iq_table_spotcheck() in src/kernels/x86/iq_table_check.c.
//!
//! A failure here means a table drifted, which is the root cause class behind
//! the "SIMD (3 tiers) + _ref vs engine rowDot" runtime-quantization failures.

const std = @import("std");
const tables = @import("iq_tables.zig");

extern fn iq_table_spotcheck() c_int;

test "iq tables: kmask_iq2xs is the bit mask" {
    for (tables.kmask_iq2xs, 0..) |m, i| {
        try std.testing.expectEqual(@as(u8, @as(u8, 1) << @as(u3, @intCast(i))), m);
    }
}

test "iq tables: keven_signs_q2xs is the expansion of ksigns_iq2xs" {
    for (tables.ksigns_iq2xs, 0..) |mask, i| {
        for (0..8) |b| {
            const want: i8 = if (mask & (@as(u8, 1) << @as(u3, @intCast(b))) != 0) -1 else 1;
            try std.testing.expectEqual(want, tables.keven_signs_q2xs[i * 8 + b]);
        }
    }
}

test "iq tables: derived ksigns64 equals the upstream table" {
    for (tables.ksigns64, 0..) |want, i| {
        try std.testing.expectEqual(want, tables.ksigns64_derived[i]);
    }
}

test "iq tables: iq4nl mirrors are consistent" {
    for (tables.kvalues_iq4nl, 0..) |v, i| {
        try std.testing.expectEqual(@as(i32, v), tables.k32_iq4nl_tab[i]);
    }
    for (tables.kvalues_fp4, 0..) |v, i| {
        try std.testing.expectEqual(@as(f32, v), tables.k32_nvfp4_values[i]);
    }
}

test "iq tables: grid sizes match what the kernels index" {
    try std.testing.expectEqual(@as(usize, 2048), tables.iq1s_grid.len);
    try std.testing.expectEqual(@as(usize, 256), tables.iq2xxs_grid.len);
    try std.testing.expectEqual(@as(usize, 512), tables.iq2xs_grid.len);
    try std.testing.expectEqual(@as(usize, 1024), tables.iq2s_grid.len);
    try std.testing.expectEqual(@as(usize, 256), tables.iq3xxs_grid.len);
    try std.testing.expectEqual(@as(usize, 512), tables.iq3s_grid.len);
}

test "iq tables: C kernels read the same tables as Zig" {
    const bad = iq_table_spotcheck();
    if (bad != 0) {
        std.debug.print("iq_table_spotcheck: {d} table entries disagree with upstream\n", .{bad});
    }
    try std.testing.expectEqual(@as(c_int, 0), bad);
}
