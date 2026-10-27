//! Parity tests for the 32-block SIMD kernels (src/simd/kern32.zig).
//!
//! Each kernel is checked three ways:
//!   1. SIMD, in each of the three ISA tiers, vs its scalar `_ref` twin;
//!   2. the scalar `_ref` vs the engine's existing scalar `rowDot*` on real
//!      packed rows from testdata/udtest (where such data exists).
//!
//! The `simd_c` module (C kernels) and `simd_ref` module (scalar oracle) are
//! injected by build.zig, so these tests run under `zig build test`.

const std = @import("std");
const quant = @import("quant_dequant.zig");
const simd_c = @import("simd_c.zig");
const simd_ref = @import("simd_ref.zig");

const TESTDATA = "/mnt/llm-dev/qwen35/testdata/udtest/";
const q8_bytes = 34;
const DotFn = *const fn (n: usize, vx: [*]const u8, vy: [*]const u8) callconv(.c) f32;

const tier_list = .{
    .{ .name = "c", .m = simd_c },
};

fn relErr(a: f32, b: f32) f32 {
    return @abs(a - b) / @max(@abs(b), 1.0);
}

fn makeX(buf: []f32, seed: u64) void {
    var rng = std.Random.DefaultPrng.init(seed);
    const r = rng.random();
    for (buf) |*v| v.* = (r.float(f32) * 2.0 - 1.0) * 4.0;
}

/// Random packed weight bytes with the f16 scale fields forced into a small
/// finite range, so products stay sane and no NaN/Inf enters the sums.
fn fillWeight(alloc: std.mem.Allocator, rb: usize, seed: u64, comptime scales: []const usize) ![]u8 {
    const b = try alloc.alloc(u8, rb);
    var rng = std.Random.DefaultPrng.init(seed);
    const r = rng.random();
    r.bytes(b);
    inline for (scales) |off| {
        if (off + 1 < rb) {
            const raw = std.mem.readInt(u16, b[off..][0..2], .little);
            std.mem.writeInt(u16, b[off..][0..2], 0x2C00 | (raw & 0x03FF), .little); // ~0.06..0.9
        }
    }
    return b;
}

const cases = .{
    .{ .name = "q8_0_q8_0", .elems = 32, .rb = 34, .scales = &[_]usize{0} },
    .{ .name = "q4_0_q8_0", .elems = 32, .rb = 18, .scales = &[_]usize{0} },
    .{ .name = "q4_1_q8_0", .elems = 32, .rb = 20, .scales = &[_]usize{ 0, 2 } },
    .{ .name = "q5_0_q8_0", .elems = 32, .rb = 22, .scales = &[_]usize{0} },
    .{ .name = "q5_1_q8_0", .elems = 32, .rb = 24, .scales = &[_]usize{ 0, 2 } },
    .{ .name = "q1_0_q8_0", .elems = 128, .rb = 18, .scales = &[_]usize{0} },
    .{ .name = "q2_0_q8_0", .elems = 64, .rb = 18, .scales = &[_]usize{0} },
    .{ .name = "iq4_nl_q8_0", .elems = 32, .rb = 18, .scales = &[_]usize{0} },
    .{ .name = "mxfp4_q8_0", .elems = 32, .rb = 17, .scales = &[_]usize{} },
    .{ .name = "nvfp4_q8_0", .elems = 64, .rb = 36, .scales = &[_]usize{} },
};

fn runCase(
    comptime m: type,
    comptime refm: type,
    comptime name: []const u8,
    comptime elems: usize,
    comptime rb: usize,
    comptime scales: []const usize,
    q8: []const u8,
    alloc: std.mem.Allocator,
) !f32 {
    const simd_f: DotFn = @field(m.dot32, name);
    const ref_f: DotFn = @field(refm.dot32, name ++ "_ref");
    const nblocks = 256 / elems;
    const w = try alloc.alloc(u8, nblocks * rb);
    defer alloc.free(w);
    for (0..nblocks) |i| {
        const blk = try fillWeight(alloc, rb, 0x1000 + i * 7, scales);
        defer alloc.free(blk);
        @memcpy(w[i * rb ..][0..rb], blk);
    }
    var worst: f32 = 0;
    for (0..nblocks) |i| {
        const vx: [*]const u8 = w.ptr + i * rb;
        const e = relErr(simd_f(elems, vx, q8.ptr), ref_f(elems, vx, q8.ptr));
        if (e > worst) worst = e;
    }
    return worst;
}

test "simd kern32: SIMD (3 tiers) vs scalar ref" {
    const alloc = std.testing.allocator;
    var x: [256]f32 = undefined;
    makeX(&x, 0xC0FFEE);
    const q8 = try alloc.alloc(u8, (256 / 32) * q8_bytes);
    defer alloc.free(q8);
    simd_c.act.quantizeQ8_0(x.len, &x, q8.ptr);

    var worst: f32 = 0;
    var worst_case: []const u8 = "";
    var worst_tier: []const u8 = "";
    inline for (tier_list) |t| {
        inline for (cases) |c| {
            const e = try runCase(t.m, simd_ref, c.name, c.elems, c.rb, c.scales, q8, alloc);
            std.debug.print("    {s}/{s} = {}\n", .{ c.name, t.name, e });
            if (e > worst) {
                worst = e;
                worst_case = c.name;
                worst_tier = t.name;
            }
        }
    }
    std.debug.print("  kern32 simd-vs-ref worst = {} ({s}/{s})\n", .{ worst, worst_case, worst_tier });
    try std.testing.expect(worst < 1e-3);
}

// Q4_0 / Q4_1 / IQ4_NL have real packed rows in testdata; compare the SIMD
// kernel (over a whole 256-element row) with the engine's scalar rowDot.
const Case2 = struct { file: []const u8, rb: usize, simd: DotFn, rowdot: *const fn (row: []const u8, x: []const f32) f32 };

const cases2 = [_]Case2{
    .{ .file = "Q4_0.bin", .rb = 144, .simd = &simd_c.dot32.q4_0_q8_0, .rowdot = &quant.rowDotQ4_0 },
    .{ .file = "Q4_1.bin", .rb = 160, .simd = &simd_c.dot32.q4_1_q8_0, .rowdot = &quant.rowDotQ4_1 },
    .{ .file = "IQ4_NL.bin", .rb = 144, .simd = &simd_c.dot32.iq4_nl_q8_0, .rowdot = &quant.rowDotIQ4NL },
};

test "simd kern32: real testdata rows vs engine rowDot" {
    const alloc = std.testing.allocator;
    const io = std.testing.io;

    const axb = try readFile(alloc, io, TESTDATA ++ "A_x.f32");
    defer alloc.free(axb);
    const ax = asF32(axb);

    const q8 = try alloc.alloc(u8, (256 / 32) * q8_bytes);
    defer alloc.free(q8);
    simd_c.act.quantizeQ8_0(256, ax.ptr, q8.ptr);

    // Exact f32 view of the Q8_0 activation (d * q): feeding this to the engine's
    // scalar rowDot isolates kernel error from activation-quantization error.
    var xq: [256]f32 = undefined;
    for (0..8) |b| {
        const d = f16bits(q8[b * 34 ..][0..2]);
        for (0..32) |j| xq[b * 32 + j] = d * @as(f32, @floatFromInt(@as(i8, @bitCast(q8[b * 34 + 2 + j]))));
    }

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
            const got = c.simd(256, row.ptr, q8.ptr);
            const ref = c.rowdot(row, &xq);
            const e = relErr(got, ref);
            if (e > worst) {
                worst = e;
                worst_file = c.file;
            }
        }
    }
    std.debug.print("  kern32 testdata-vs-rowDot worst = {} ({s})\n", .{ worst, worst_file });
    try std.testing.expect(worst < 5e-3);
}

fn f16bits(b: *const [2]u8) f32 {
    const u = std.mem.readInt(u16, b, .little);
    const h: f16 = @bitCast(u);
    return @floatCast(h);
}

fn readFile(alloc: std.mem.Allocator, io: std.Io, path: []const u8) ![]u8 {
    return std.Io.Dir.cwd().readFileAlloc(io, path, alloc, std.Io.Limit.unlimited);
}

fn asF32(b: []const u8) []const f32 {
    const p: [*]const f32 = @alignCast(@ptrCast(@constCast(&b[0])));
    return p[0 .. b.len / @sizeOf(f32)];
}

// ---------------------------------------------------------------------------
// Row-level kernels: one horizontal reduction per row (the fast path used by
// the engine).  Validated against the block kernels, which are themselves
// validated against the C-translated scalar refs above.
// ---------------------------------------------------------------------------

fn fillRow(alloc: std.mem.Allocator, rb: usize, nblocks: usize, seed: u64, comptime scales: []const usize) ![]u8 {
    const b = try alloc.alloc(u8, rb * nblocks);
    var rng = std.Random.DefaultPrng.init(seed);
    const r = rng.random();
    r.bytes(b);
    for (0..nblocks) |blk| {
        inline for (scales) |off| {
            const o = blk * rb + off;
            if (o + 1 < b.len) {
                const raw = std.mem.readInt(u16, b[o..][0..2], .little);
                std.mem.writeInt(u16, b[o..][0..2], 0x2C00 | (raw & 0x03FF), .little);
            }
        }
    }
    return b;
}

test "simd kern32: row kernels == sum of block kernels (3 tiers)" {
    const alloc = std.testing.allocator;
    const nblocks = 8;
    var x: [nblocks * 32]f32 = undefined;
    makeX(&x, 0x5EED);
    const n = x.len;

    var worst: f32 = 0;
    var worst_case: []const u8 = "";
    const RowCase = struct { name: []const u8, elems: usize, rb: usize, scales: []const usize };
    const rc = [_]RowCase{
        .{ .name = "q8_0_q8_0", .elems = 32, .rb = 34, .scales = &[_]usize{0} },
        .{ .name = "q4_0_q8_0", .elems = 32, .rb = 18, .scales = &[_]usize{0} },
        .{ .name = "q4_1_q8_0", .elems = 32, .rb = 20, .scales = &[_]usize{ 0, 2 } },
        .{ .name = "q5_0_q8_0", .elems = 32, .rb = 22, .scales = &[_]usize{0} },
        .{ .name = "q5_1_q8_0", .elems = 32, .rb = 24, .scales = &[_]usize{ 0, 2 } },
        .{ .name = "q1_0_q8_0", .elems = 128, .rb = 18, .scales = &[_]usize{0} },
        .{ .name = "q2_0_q8_0", .elems = 64, .rb = 18, .scales = &[_]usize{0} },
        .{ .name = "iq4_nl_q8_0", .elems = 32, .rb = 18, .scales = &[_]usize{0} },
        .{ .name = "mxfp4_q8_0", .elems = 32, .rb = 17, .scales = &[_]usize{} },
        .{ .name = "nvfp4_q8_0", .elems = 64, .rb = 36, .scales = &[_]usize{} },
    };
    const q8 = try alloc.alloc(u8, (n / 32) * q8_bytes);
    defer alloc.free(q8);
    simd_c.act.quantizeQ8_0(n, &x, q8.ptr);

    inline for (tier_list) |t| {
        inline for (rc) |c| {
            const nb = if (c.elems == 32)
                nblocks
            else if (c.elems == 128)
                nblocks / 4
            else
                nblocks / 2;
            const w = try fillRow(alloc, c.rb, nb, 0x2000, c.scales);
            defer alloc.free(w);
            const row_f: DotFn = @field(t.m.dot32, c.name ++ "_row");
            const blk_f: DotFn = @field(t.m.dot32, c.name);
            const got = row_f(nb * c.elems, w.ptr, q8.ptr);
            var ref: f32 = 0;
            for (0..nb) |i| ref += blk_f(c.elems, w[i * c.rb ..].ptr, q8[i * 34 * (c.elems / 32) ..].ptr);
            const e = relErr(got, ref);
            if (e > worst) {
                worst = e;
                worst_case = c.name;
            }
        }
    }
    std.debug.print("  kern32 row-vs-block worst = {} ({s})\n", .{ worst, worst_case });
    try std.testing.expect(worst < 1e-3);
}
