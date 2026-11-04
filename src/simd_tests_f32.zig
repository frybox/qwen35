//! Cross-checks for the f16/bf16 x f32 dot kernels in `kernels_f32.c`
//! (`bf16_dot_f32`, `f16_dot_f32`, `f32_dot_f32`).
//!
//! These are the kernels the BF16 model runs on: weights stay in their native
//! 2-byte form (the mmap bytes, no expanded copy) and the kernel widens each
//! element as it multiplies.  Three properties must hold:
//!
//!   1. **Bit-identity with the f32-materialized path.**  `bf16_dot_f32` over a
//!      packed bf16 row must equal `f32_dot_f32` over the same row expanded to
//!      f32, exactly.  The widening is exact (bf16 is the top half of an f32)
//!      and the accumulate structure is deliberately shared, so the result is
//!      the same set of products summed in the same order.  This is the property
//!      that makes dropping the f32 copy a *no-op numerically* -- if it drifts
//!      by even 1 ulp, an argmax in a near-tie can flip and the whole
//!      generation diverges (measured on the 0.8B BF16 model).
//!   2. **A Zig scalar oracle.**  Left-to-right, one accumulator.  The SIMD
//!      path has four accumulators, so this is a tolerance check, not bit
//!      equality.
//!   3. **f16 gets the same treatment** (F16C's `vcvtph2ps` is also exact).
//!
//! Weight bytes are built with constrained exponent fields: raw random bytes
//! land a large share of lanes in the NaN/Inf/large-exponent range, which
//! poisons the tolerance for reasons unrelated to the kernel.  Values are
//! all-positive so the row sum stays far from zero (see the note on
//! `fillRowPacked` in `simd_tests32.zig`).

const std = @import("std");
const simd_api = @import("simd_api.zig");

const rel_eps: f32 = 1e-6;

fn relErr(a: f32, b: f32) f32 {
    const denom = @max(@abs(a), @abs(b));
    if (denom == 0) return 0;
    return @abs(a - b) / denom;
}

fn fillW16(alloc: std.mem.Allocator, n: usize, seed: u64, comptime is_bf16: bool) ![]u8 {
    const b = try alloc.alloc(u8, n * 2);
    var rng = std.Random.DefaultPrng.init(seed);
    const r = rng.random();
    const v: [*]align(1) u16 = @ptrCast(b.ptr);
    for (0..n) |i| {
        if (comptime is_bf16) {
            // bf16 = top 16 bits of an f32: sign(1) exp(8) mant(7), bias 127.
            const exp: u16 = 120 + @as(u16, @intCast(r.intRangeAtMost(u8, 0, 10)));
            v[i] = (exp << 7) | @as(u16, r.intRangeAtMost(u8, 0, 127));
        } else {
            // f16: sign(1) exp(5) mant(10), bias 15.
            const exp: u16 = 10 + @as(u16, @intCast(r.intRangeAtMost(u8, 0, 10)));
            v[i] = (exp << 10) | @as(u16, r.intRangeAtMost(u16, 0, 1023));
        }
    }
    return b;
}

fn makeXPos(buf: []f32, seed: u64) void {
    var rng = std.Random.DefaultPrng.init(seed);
    const r = rng.random();
    for (buf) |*v| v.* = r.float(f32) * 4.0 + 0.5;
}

fn widened(alloc: std.mem.Allocator, w: []const u8, comptime is_bf16: bool) ![]f32 {
    const n = w.len / 2;
    const f = try alloc.alloc(f32, n);
    const src: [*]align(1) const u16 = @ptrCast(w.ptr);
    for (0..n) |i| {
        f[i] = if (is_bf16)
            @as(f32, @bitCast(@as(u32, @intCast(src[i])) << 16))
        else
            @as(f32, @floatCast(@as(f16, @bitCast(src[i]))));
    }
    return f;
}

/// The oracle: same as the C scalar fallback, one accumulator, k = 0..n-1.
fn refW16(w: []const u8, x: []const f32, comptime is_bf16: bool) f32 {
    const src: [*]align(1) const u16 = @ptrCast(w.ptr);
    var acc: f32 = 0;
    for (0..w.len / 2) |i| {
        const wv = if (is_bf16)
            @as(f32, @bitCast(@as(u32, @intCast(src[i])) << 16))
        else
            @as(f32, @floatCast(@as(f16, @bitCast(src[i]))));
        acc += wv * x[i];
    }
    return acc;
}

const lengths = [_]usize{ 32, 33, 64, 100, 256, 1024, 4096 };

test "simd f32: bf16/f16 packed dot == scalar oracle, and bit-identical to the f32-materialized dot" {
    const alloc = std.testing.allocator;
    var worst_ref: f32 = 0;
    var worst_mat: f32 = 0;
    var bit_exact: bool = true;

    inline for ([_]bool{ true, false }) |is_bf16| {
        const dot = if (is_bf16) simd_api.dotBf16F32 else simd_api.dotF16F32;
        inline for (lengths) |n| {
            const w = try fillW16(alloc, n, 0x5EED + n, is_bf16);
            defer alloc.free(w);
            var x: [4096]f32 = undefined;
            makeXPos(x[0..n], 0xB16C0DE + n);

            const got = dot(@intCast(n), w.ptr, @ptrCast(&x));
            const ref = refW16(w, x[0..n], is_bf16);
            const e1 = relErr(got, ref);
            if (e1 > worst_ref) worst_ref = e1;

            // The decisive check: no expanded copy exists, so the packed kernel
            // must reproduce the expanded one EXACTLY.  Not a tolerance -- bit
            // equality, because the widening is exact and the accumulate
            // structure is shared with `f32_dot_f32`.
            const f32row = try widened(alloc, w, is_bf16);
            defer alloc.free(f32row);
            const got_f32 = simd_api.dotF32(@intCast(n), @ptrCast(f32row.ptr), @ptrCast(&x));
            const e2 = relErr(got, got_f32);
            if (e2 > worst_mat) worst_mat = e2;
            if (got != got_f32) {
                std.debug.print("  packed/f32 diverge at n={d} is_bf16={}: {d} vs {d}\n", .{ n, is_bf16, got, got_f32 });
                bit_exact = false;
            }
        }
    }
    std.debug.print("  simd f32 w16: worst vs scalar oracle = {d:.6}\n", .{worst_ref});
    std.debug.print("  simd f32 w16: worst vs f32-materialized dot = {d:.6}\n", .{worst_mat});
    try std.testing.expect(worst_ref < 1e-3);
    try std.testing.expect(worst_mat == 0);
    try std.testing.expect(bit_exact);
}

test "simd f32: bf16/f16 dot handles negative weights and zero activations" {
    // Sign coverage.  The positive-value cases keep the sum far from zero (so
    // relative error is meaningful), which is why the main test uses them.  This
    // case exercises the sign bits and exact zeros, where cancellation is
    // expected -- so the check is against the f32-materialized dot (bit
    // equality), not the scalar oracle at a fixed tolerance.
    const alloc = std.testing.allocator;
    const n = 1024;
    const w = try fillW16(alloc, n, 0x2024, true);
    defer alloc.free(w);
    const src: [*]align(1) u16 = @ptrCast(w.ptr);
    var rng = std.Random.DefaultPrng.init(77);
    const r = rng.random();
    for (0..n) |i| {
        if (r.boolean()) src[i] |= 0x8000; // negative weights
        if (i % 7 == 0) src[i] = 0; // +/-0 weights
    }
    var x: [1024]f32 = undefined;
    for (0..n) |i| x[i] = if (i % 5 == 0) 0.0 else @as(f32, @floatFromInt(r.intRangeAtMost(i8, -127, 127)));
    const got = simd_api.dotBf16F32(@intCast(n), w.ptr, @ptrCast(&x));
    const f32row = try widened(alloc, w, true);
    defer alloc.free(f32row);
    const got_f32 = simd_api.dotF32(@intCast(n), @ptrCast(f32row.ptr), @ptrCast(&x));
    std.debug.print("  simd f32 w16 signed: packed={d} materialized={d}\n", .{ got, got_f32 });
    try std.testing.expect(got == got_f32);
    // And the scalar oracle, at a cancellation-aware tolerance.
    const ref = refW16(w, &x, true);
    try std.testing.expect(relErr(got, ref) < 1e-3);
}
