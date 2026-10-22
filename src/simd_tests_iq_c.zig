//! Parity tests for the IQ1_S / IQ1_M 256-block kernels (src/simd/kerniq3.zig).
//! Structure mirrors src/simd_tests256.zig:
//!   1. SIMD (each ISA tier, currently forwarding) + scalar `_ref` vs engine
//!      rowDotIQ1S / rowDotIQ1M on synthetic rows,
//!   2. same on real testdata rows (skipped if no .bin).
//! Tier modules injected by build.zig -> run via `zig build test`.
//!
//! Note: IQ1_S has d(f16)@0.  IQ1_M packs d from the 4 scale u16s (@48..55),
//! so its synthetic fill forces all four scale words to keep the assembled
//! f16 finite (random bytes could otherwise assemble an inf/NaN scale).

const std = @import("std");
const quant = @import("quant_dequant.zig");
const simd_base = @import("simd_baseline");
const simd_avx2 = @import("simd_avx2");
const simd_avx512 = @import("simd_avx512");
const simd_avx512vnni = @import("simd_avx512vnni");

const TESTDATA = "/mnt/llm-dev/qwen35/testdata/udtest/";
const q8k_bytes = 292;
const DotFn = *const fn (n: usize, vx: [*]const u8, vy: [*]const u8) callconv(.c) f32;
const RowDotFn = *const fn (row: []const u8, x: []const f32) f32;

const tier_list = .{
    .{ .name = "baseline", .m = simd_base },
    .{ .name = "avx2", .m = simd_avx2 },
    .{ .name = "avx512", .m = simd_avx512 },
    .{ .name = "avx512vnni", .m = simd_avx512vnni },
};

fn relErr(a: f32, b: f32) f32 {
    return @abs(a - b) / @max(@abs(b), 1.0);
}

fn makeX(buf: []f32, seed: u64) void {
    var rng = std.Random.DefaultPrng.init(seed);
    const r = rng.random();
    for (buf) |*v| v.* = (r.float(f32) * 2.0 - 1.0) * 4.0;
}

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

const Case = struct { name: []const u8, rb: usize, rowdot: RowDotFn, file: []const u8, scales: []const usize, simd: DotFn };

const cases = [_]Case{
    .{ .name = "iq1_s_q8_k", .rb = 50, .rowdot = &quant.rowDotIQ1S, .file = "IQ1_S.bin", .scales = &[_]usize{0}, .simd = &simd_base.dotiq3.iq1_s_q8_k },
    .{ .name = "iq1_m_q8_k", .rb = 56, .rowdot = &quant.rowDotIQ1M, .file = "IQ1_M.bin", .scales = &[_]usize{ 48, 50, 52, 54 }, .simd = &simd_base.dotiq3.iq1_m_q8_k },
};

test "simd kerniq3: SIMD (3 tiers) + _ref vs engine rowDot (synthetic)" {
    const alloc = std.testing.allocator;
    var x: [256]f32 = undefined;
    makeX(&x, 0xBEEF);
    const q8k = try alloc.alloc(u8, q8k_bytes);
    defer alloc.free(q8k);
    simd_base.act.quantizeQ8_K(256, &x, q8k.ptr);
    var xq: [256]f32 = undefined;
    const d: f32 = @bitCast(std.mem.readInt(u32, q8k[0..4], .little));
    for (0..256) |j| xq[j] = d * @as(f32, @floatFromInt(@as(i8, @bitCast(q8k[4 + j]))));

    var worst: f32 = 0;
    var worst_name: []const u8 = "";
    inline for (cases) |c| {
        const w = try fillWeight(alloc, c.rb, 0x2000, c.scales);
        defer alloc.free(w);
        const ref: f32 = c.rowdot(w, &xq);
        inline for (tier_list) |t| {
            const simd_f: DotFn = @field(t.m.dotiq3, c.name);
            const got = simd_f(256, w.ptr, q8k.ptr);
            const e = relErr(got, ref);
            std.debug.print("    {s}/{s} = {}\n", .{ c.name, t.name, e });
            if (e > worst) {
                worst = e;
                worst_name = c.name;
            }
        }
    }
    std.debug.print("  kerniq3 synthetic worst = {} ({s})\n", .{ worst, worst_name });
    try std.testing.expect(worst < 1e-3);
}

test "simd kerniq3: real testdata rows vs engine rowDot" {
    const alloc = std.testing.allocator;
    const io = std.testing.io;
    const axb = try readFile(alloc, io, TESTDATA ++ "A_x.f32");
    defer alloc.free(axb);
    const ax = asF32(axb);
    const q8k = try alloc.alloc(u8, q8k_bytes);
    defer alloc.free(q8k);
    simd_base.act.quantizeQ8_K(256, ax.ptr, q8k.ptr);
    var xq: [256]f32 = undefined;
    const d: f32 = @bitCast(std.mem.readInt(u32, q8k[0..4], .little));
    for (0..256) |j| xq[j] = d * @as(f32, @floatFromInt(@as(i8, @bitCast(q8k[4 + j]))));

    var worst: f32 = 0;
    var worst_file: []const u8 = "";
    for (cases) |c| {
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
    std.debug.print("  kerniq3 testdata-vs-rowDot worst = {} ({s})\n", .{ worst, worst_file });
    try std.testing.expect(worst < 5e-3);
}

fn readFile(alloc: std.mem.Allocator, io: std.Io, path: []const u8) ![]u8 {
    return std.Io.Dir.cwd().readFileAlloc(io, path, alloc, std.Io.Limit.unlimited);
}

fn asF32(b: []const u8) []const f32 {
    const p: [*]const f32 = @alignCast(@ptrCast(@constCast(&b[0])));
    return p[0 .. b.len / @sizeOf(f32)];
}
