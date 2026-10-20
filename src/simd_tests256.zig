//! Parity tests for the 256-block K-quant / ternary SIMD kernels.
//!
//!   1. SIMD (each ISA tier) vs the scalar `_ref` twin,
//!   2. `_ref` vs the engine's existing scalar `rowDot*` on real testdata rows.
//!
//! The activation fed to the engine's rowDot is the *exact* f32 view of the
//! Q8_K block (d * q), so activation-quantization error does not pollute the
//! kernel parity measurement.
//!
//! Tier modules are injected by build.zig -> run via `zig build test`.

const std = @import("std");
const quant = @import("quant_dequant.zig");
const simd_base = @import("simd_baseline");
const simd_avx2 = @import("simd_avx2");
const simd_avx512 = @import("simd_avx512");

const TESTDATA = "/mnt/llm-dev/qwen35/testdata/udtest/";
const q8k_bytes = 292;
const DotFn = *const fn (n: usize, vx: [*]const u8, vy: [*]const u8) callconv(.c) f32;

const tier_list = .{
    .{ .name = "baseline", .m = simd_base },
    .{ .name = "avx2", .m = simd_avx2 },
    .{ .name = "avx512", .m = simd_avx512 },
};

fn relErr(a: f32, b: f32) f32 {
    return @abs(a - b) / @max(@abs(b), 1.0);
}

fn makeX(buf: []f32, seed: u64) void {
    var rng = std.Random.DefaultPrng.init(seed);
    const r = rng.random();
    for (buf) |*v| v.* = (r.float(f32) * 2.0 - 1.0) * 4.0;
}

/// Random packed weight bytes; the f16 scale fields are forced small and finite.
fn fillWeight(alloc: std.mem.Allocator, rb: usize, seed: u64, comptime scales: []const usize) ![]u8 {
    const b = try alloc.alloc(u8, rb);
    var rng = std.Random.DefaultPrng.init(seed);
    const r = rng.random();
    r.bytes(b);
    inline for (scales) |off| {
        if (off + 1 < rb) {
            const raw = std.mem.readInt(u16, b[off..][0..2], .little);
            std.mem.writeInt(u16, b[off..][0..2], 0x2C00 | (raw & 0x03FF), .little);
        }
    }
    return b;
}

const cases = .{
    .{ .name = "q2_k_q8_k", .rb = 84, .scales = &[_]usize{ 80, 82 } },
    .{ .name = "q3_k_q8_k", .rb = 110, .scales = &[_]usize{0} },
    .{ .name = "q4_k_q8_k", .rb = 144, .scales = &[_]usize{ 0, 2 } },
    .{ .name = "q5_k_q8_k", .rb = 176, .scales = &[_]usize{ 0, 2 } },
    .{ .name = "q6_k_q8_k", .rb = 210, .scales = &[_]usize{208} },
    .{ .name = "tq1_0_q8_k", .rb = 54, .scales = &[_]usize{52} },
    .{ .name = "tq2_0_q8_k", .rb = 66, .scales = &[_]usize{64} },
};

fn runCase(
    comptime m: type,
    comptime name: []const u8,
    comptime rb: usize,
    comptime scales: []const usize,
    q8k: []const u8,
    alloc: std.mem.Allocator,
) !f32 {
    const simd_f: DotFn = @field(m.dot256, name);
    const ref_f: DotFn = @field(m.dot256, name ++ "_ref");
    const w = try fillWeight(alloc, rb, 0x2000, scales);
    defer alloc.free(w);
    const got = simd_f(256, w.ptr, q8k.ptr);
    const ref = ref_f(256, w.ptr, q8k.ptr);
    const e = relErr(got, ref);
    if (e > 1e-3) std.debug.print("      [{s}] e={} got={d} ref={d}\n", .{ name, e, got, ref });
    return e;
}

test "simd kern256: SIMD (3 tiers) vs scalar ref" {
    const alloc = std.testing.allocator;
    var x: [256]f32 = undefined;
    makeX(&x, 0xBEEF);
    const q8k = try alloc.alloc(u8, q8k_bytes);
    defer alloc.free(q8k);
    simd_base.act.quantizeQ8_K(256, &x, q8k.ptr);

    var worst: f32 = 0;
    var worst_case: []const u8 = "";
    var worst_tier: []const u8 = "";
    inline for (tier_list) |t| {
        inline for (cases) |c| {
            const e = try runCase(t.m, c.name, c.rb, c.scales, q8k, alloc);
            std.debug.print("    {s}/{s} = {}\n", .{ c.name, t.name, e });
            if (e > worst) {
                worst = e;
                worst_case = c.name;
                worst_tier = t.name;
            }
        }
    }
    std.debug.print("  kern256 simd-vs-ref worst = {} ({s}/{s})\n", .{ worst, worst_case, worst_tier });
    try std.testing.expect(worst < 1e-3);
}

const Case2 = struct {
    file: []const u8,
    rb: usize,
    simd: DotFn,
    rowdot: *const fn (row: []const u8, x: []const f32) f32,
};

const cases2 = [_]Case2{
    .{ .file = "Q4_K.bin", .rb = 144, .simd = &simd_base.dot256.q4_k_q8_k, .rowdot = &quant.rowDotQ4K },
    .{ .file = "Q5_K.bin", .rb = 176, .simd = &simd_base.dot256.q5_k_q8_k, .rowdot = &quant.rowDotQ5K },
    .{ .file = "Q6_K.bin", .rb = 210, .simd = &simd_base.dot256.q6_k_q8_k, .rowdot = &quant.rowDotQ6K },
    .{ .file = "Q3_K.bin", .rb = 110, .simd = &simd_base.dot256.q3_k_q8_k, .rowdot = &quant.rowDotQ3K },
};

test "simd kern256: real testdata rows vs engine rowDot" {
    const alloc = std.testing.allocator;
    const io = std.testing.io;

    const axb = try readFile(alloc, io, TESTDATA ++ "A_x.f32");
    defer alloc.free(axb);
    const ax = asF32(axb);

    const q8k = try alloc.alloc(u8, q8k_bytes);
    defer alloc.free(q8k);
    simd_base.act.quantizeQ8_K(256, ax.ptr, q8k.ptr);

    // exact f32 view of the Q8_K activation
    var xq: [256]f32 = undefined;
    const d: f32 = @bitCast(std.mem.readInt(u32, q8k[0..4], .little));
    for (0..256) |j| xq[j] = d * @as(f32, @floatFromInt(@as(i8, @bitCast(q8k[4 + j]))));

    var worst: f32 = 0;
    var worst_file: []const u8 = "";
    for (cases2) |c| {
        const path = std.fmt.allocPrint(alloc, "{s}{s}", .{ TESTDATA, c.file }) catch unreachable;
        defer alloc.free(path);
        const bin = readFile(alloc, io, path) catch {
            std.debug.print("  (skip {s})\n", .{c.file});
            continue;
        };
        defer alloc.free(bin);
        const nrows = bin.len / c.rb;
        for (0..nrows) |r| {
            const row = bin[r * c.rb .. (r + 1) * c.rb];
            const e = relErr(c.simd(256, row.ptr, q8k.ptr), c.rowdot(row, &xq));
            if (e > worst) {
                worst = e;
                worst_file = c.file;
            }
        }
    }
    std.debug.print("  kern256 testdata-vs-rowDot worst = {} ({s})\n", .{ worst, worst_file });
    try std.testing.expect(worst < 5e-3);
}

fn readFile(alloc: std.mem.Allocator, io: std.Io, path: []const u8) ![]u8 {
    return std.Io.Dir.cwd().readFileAlloc(io, path, alloc, std.Io.Limit.unlimited);
}

fn asF32(b: []const u8) []const f32 {
    const p: [*]const f32 = @alignCast(@ptrCast(@constCast(&b[0])));
    return p[0 .. b.len / @sizeOf(f32)];
}
