//! Temporary benchmark for the vectorized Q4_0 / Q4_1 rowDot kernels.
//! Run from the repo root:
//!   zig run -OReleaseFast src/bench.zig
//! Prints (a) per-row exactness of vector vs scalar (max abs / max rel diff over
//! 96 rows of testdata/udtest/Q4_0.bin and Q4_1.bin against A_x.f32), and
//! (b) scalar vs vector wall-clock over `iters` passes of 96 rows, plus speedup.

const std = @import("std");
const quant = @import("quant_dequant.zig");

const TESTDATA = "testdata/udtest/";

fn asF32(b: []const u8) []const f32 {
    const p: [*]const f32 = @alignCast(@ptrCast(@constCast(&b[0])));
    return p[0 .. b.len / @sizeOf(f32)];
}

fn cmp(bin: []const u8, rb: usize, ax: []const f32, comptime vec: fn ([]const u8, []const f32) f32, comptime sca: fn ([]const u8, []const f32) f32, name: []const u8) void {
    var max_abs: f32 = 0;
    var max_rel: f32 = 0;
    var worst_row: usize = 0;
    const nrows = bin.len / rb;
    for (0..nrows) |r| {
        const row = bin[r * rb .. (r + 1) * rb];
        const s = sca(row, ax);
        const v = vec(row, ax);
        const e = @abs(s - v);
        if (e > max_abs) {
            max_abs = e;
            worst_row = r;
        }
        const rel = e / @max(@abs(s), 1);
        if (rel > max_rel) max_rel = rel;
    }
    std.debug.print("{s} vec-vs-scalar: max_abs={d} max_rel={d} (worst row {d})\n", .{ name, max_abs, max_rel, worst_row });
}

fn timeIt(io: std.Io, label: []const u8, comptime iters: usize, bin: []const u8, rb: usize, ax: []const f32, comptime f: fn ([]const u8, []const f32) f32) f64 {
    var sum: f32 = 0;
    const t0 = std.Io.Clock.Timestamp.now(io, .awake);
    for (0..iters) |_| {
        for (0..96) |r| sum += f(bin[r * rb .. (r + 1) * rb], ax);
    }
    const t1 = std.Io.Clock.Timestamp.now(io, .awake);
    const ms: f64 = @as(f64, @floatFromInt(t1.raw.nanoseconds - t0.raw.nanoseconds)) / 1e6;
    std.debug.print("{s}: {d} ms  sum={d}\n", .{ label, ms, sum });
    return ms;
}

pub fn main() !void {
    const alloc = std.heap.page_allocator;
    var thr = std.Io.Threaded.init(alloc, .{});
    const io = thr.io();
    const bin0 = try std.Io.Dir.cwd().readFileAlloc(io, TESTDATA ++ "Q4_0.bin", alloc, .unlimited);
    const bin1 = try std.Io.Dir.cwd().readFileAlloc(io, TESTDATA ++ "Q4_1.bin", alloc, .unlimited);
    const axb = try std.Io.Dir.cwd().readFileAlloc(io, TESTDATA ++ "A_x.f32", alloc, .unlimited);
    const ax = asF32(axb);

    cmp(bin0, 144, ax, quant.rowDotQ4_0, quant.rowDotQ4_0_scalar, "Q4_0");
    cmp(bin1, 160, ax, quant.rowDotQ4_1, quant.rowDotQ4_1_scalar, "Q4_1");

    const iters = 100000;
    const ts0 = timeIt(io, "Q4_0 scalar", iters, bin0, 144, ax, quant.rowDotQ4_0_scalar);
    const tv0 = timeIt(io, "Q4_0 vector", iters, bin0, 144, ax, quant.rowDotQ4_0);
    std.debug.print("Q4_0 speedup = {d}\n", .{ ts0 / tv0 });

    const ts1 = timeIt(io, "Q4_1 scalar", iters, bin1, 160, ax, quant.rowDotQ4_1_scalar);
    const tv1 = timeIt(io, "Q4_1 vector", iters, bin1, 160, ax, quant.rowDotQ4_1);
    std.debug.print("Q4_1 speedup = {d}\n", .{ ts1 / tv1 });
}
