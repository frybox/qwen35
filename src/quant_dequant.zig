// Port of ggml dequantize_row_{q4_K,q5_K,q6_K,q3_K,iq4_xs,iq4_nl,iq3_s} to pure Zig.
// Each function takes one row of raw quantized bytes (exactly as stored in an
// mmap'd .bin file, little-endian) and writes the 256 dequantized f32 values,
// mirroring /tmp/ref_dequant.c (llama.cpp ggml-quants.c) expression-for-expression,
// so output is bit-identical to C for the same bytes.
//
// Row byte layouts (little-endian), one 256-element block = 1 row:
//   Q4_K   : d(2) dmin(2) scales[12] qs[128]              = 144
//   Q5_K   : d(2) dmin(2) scales[12] qh[32] qs[128]       = 176
//   Q6_K   : ql[128] qh[64] scales[i8:16] d(2)           = 210
//   Q3_K   : hmask[32] qs[64] scales[12] d(2)            = 110
//   IQ4_XS : d(2) scales_h(2) scales_l[4] qs[128]        = 136
//   IQ4_NL : 8 x ( d(2) qs[16] )                          = 144
//   IQ3_S  : d(2) qs[64] qh[8] signs[32] scales[4]        = 110

const std = @import("std");
const ggml_tables = @import("ggml_tables.zig");

const QK_K: usize = 256;
const QK4_NL: usize = 32;

const kvalues_iq4nl: [16]i8 = .{ -127, -104, -83, -65, -49, -35, -22, -10, 1, 13, 25, 38, 53, 69, 89, 113 };
const kmask_iq2xs: [8]u8 = .{ 1, 2, 4, 8, 16, 32, 64, 128 };

// f16 field at row[off..off+2] (little-endian) -> f32, exact.
inline fn f16_to_f32(row: []const u8, off: usize) f32 {
    const bits: u16 = @as(u16, row[off]) | (@as(u16, row[off + 1]) << 8);
    return @as(f32, @floatCast(@as(f16, @bitCast(bits))));
}

// ggml-quants.c get_scale_min_k4(j, sc, *d, *m):
//   if (j < 4)  { *d = sc[j] & 63; *m = sc[j+4] & 63; }
//   else       { *d = (sc[j+4] & 0xF) | ((sc[j-4] >> 6) << 4);
//               *m = (sc[j+4] >>  4) | ((sc[j+0] >> 6) << 4); }
fn get_scale_min_k4(is: usize, sc: []const u8) struct { sc: u8, m: u8 } {
    if (is < 4) {
        return .{ .sc = sc[is] & 63, .m = sc[is + 4] & 63 };
    } else {
        const s: u16 = (@as(u16, sc[is + 4] & 0x0F)) | (@as(u16, sc[is - 4] >> 6) << 4);
        const m: u16 = (@as(u16, sc[is + 4] >> 4)) | (@as(u16, sc[is] >> 6) << 4);
        return .{ .sc = @intCast(s), .m = @intCast(m) };
    }
}

// ---------------- Q4_K: d/dmin @0,2; scales[12] @4..15; qs[128] @16..143
pub fn dequantQ4K(row: []const u8, out: []f32) void {
    const d = f16_to_f32(row, 0);
    const min = f16_to_f32(row, 2);
    const scales = row[4..16];
    var q: []const u8 = row[16..144];
    var y: usize = 0;
    var is: usize = 0;
    for (0..4) |_| {
        const p0 = get_scale_min_k4(is, scales);
        const d1: f32 = d * @as(f32, @floatFromInt(@as(i32, p0.sc)));
        const m1: f32 = min * @as(f32, @floatFromInt(@as(i32, p0.m)));
        const p1 = get_scale_min_k4(is + 1, scales);
        const d2: f32 = d * @as(f32, @floatFromInt(@as(i32, p1.sc)));
        const m2: f32 = min * @as(f32, @floatFromInt(@as(i32, p1.m)));
        for (0..32) |l| out[y + l] = d1 * @as(f32, @floatFromInt(@as(i32, q[l] & 0x0F))) - m1;
        for (0..32) |l| out[y + 32 + l] = d2 * @as(f32, @floatFromInt(@as(i32, q[l] >> 4))) - m2;
        y += 64;
        q = q[32..];
        is += 2;
    }
}

// ---------------- Q5_K: d/dmin @0,2; scales[12] @4..15; qh[32] @16..47; qs[128] @48..175
pub fn dequantQ5K(row: []const u8, out: []f32) void {
    const d = f16_to_f32(row, 0);
    const min = f16_to_f32(row, 2);
    const scales = row[4..16];
    const qh = row[16..48];
    var ql: []const u8 = row[48..176];
    var y: usize = 0;
    var is: usize = 0;
    var bm1: u16 = 1;
    var bm2: u16 = 2;
    for (0..4) |_| {
        const p0 = get_scale_min_k4(is, scales);
        const d1: f32 = d * @as(f32, @floatFromInt(@as(i32, p0.sc)));
        const m1: f32 = min * @as(f32, @floatFromInt(@as(i32, p0.m)));
        const p1 = get_scale_min_k4(is + 1, scales);
        const d2: f32 = d * @as(f32, @floatFromInt(@as(i32, p1.sc)));
        const m2: f32 = min * @as(f32, @floatFromInt(@as(i32, p1.m)));
        for (0..32) |l| {
            const hi: u8 = if (qh[l] & bm1 != 0) 16 else 0;
            out[y + l] = d1 * @as(f32, @floatFromInt(@as(i32, ql[l] & 0x0F | hi))) - m1;
        }
        for (0..32) |l| {
            const hi: u8 = if (qh[l] & bm2 != 0) 16 else 0;
            out[y + 32 + l] = d2 * @as(f32, @floatFromInt(@as(i32, ql[l] >> 4 | hi))) - m2;
        }
        y += 64;
        ql = ql[32..];
        is += 2;
        bm1 <<= 2;
        bm2 <<= 2;
    }
}

// ---------------- Q6_K: ql[128] @0..127; qh[64] @128..191; scales[i8:16] @192..207; d @208..209
pub fn dequantQ6K(row: []const u8, out: []f32) void {
    const d = f16_to_f32(row, 208);
    const ql = row[0..128];
    const qh = row[128..192];
    var sc: [16]i8 = undefined;
    for (0..16) |i| sc[i] = @as(i8, @bitCast(row[192 + i]));
    var n: usize = 0;
    while (n < QK_K) : (n += 128) {
        const qlb: usize = n / 2; // 0 or 64
        const qhb: usize = n / 4; // 0 or 32
        const scb: usize = n / 16; // 0 or 8
        for (0..32) |l| {
            const is: usize = l / 16; // 0 or 1
            const q1u: u8 = (ql[qlb + l] & 0x0F) | (((qh[qhb + l] >> 0) & 3) << 4);
            const q2u: u8 = (ql[qlb + l + 32] & 0x0F) | (((qh[qhb + l] >> 2) & 3) << 4);
            const q3u: u8 = (ql[qlb + l] >> 4) | (((qh[qhb + l] >> 4) & 3) << 4);
            const q4u: u8 = (ql[qlb + l + 32] >> 4) | (((qh[qhb + l] >> 6) & 3) << 4);
            const q1: i8 = @as(i8, @bitCast(q1u)) - 32;
            const q2: i8 = @as(i8, @bitCast(q2u)) - 32;
            const q3: i8 = @as(i8, @bitCast(q3u)) - 32;
            const q4: i8 = @as(i8, @bitCast(q4u)) - 32;
            out[n + l] = d * @as(f32, @floatFromInt(sc[scb + is + 0])) * @as(f32, @floatFromInt(q1));
            out[n + l + 32] = d * @as(f32, @floatFromInt(sc[scb + is + 2])) * @as(f32, @floatFromInt(q2));
            out[n + l + 64] = d * @as(f32, @floatFromInt(sc[scb + is + 4])) * @as(f32, @floatFromInt(q3));
            out[n + l + 96] = d * @as(f32, @floatFromInt(sc[scb + is + 6])) * @as(f32, @floatFromInt(q4));
        }
    }
}

// ---------------- Q3_K: hmask[32] @0..31; qs[64] @32..95; scales[12] @96..107; d @108..109
pub fn dequantQ3K(row: []const u8, out: []f32) void {
    const d_all = f16_to_f32(row, 108);
    const q = row[32..96];
    const hm = row[0..32];
    const s = row[96..108];

    const kmask1: u32 = 0x03030303;
    const kmask2: u32 = 0x0F0F0F0F;
    // memcpy(aux, scales, 12) then the 4-line shuffle, verbatim from C ref.
    var aux0: u32 = 0;
    var aux1: u32 = 0;
    var aux2: u32 = 0;
    {
        var b0: [4]u8 = .{ s[0], s[1], s[2], s[3] };
        var b1: [4]u8 = .{ s[4], s[5], s[6], s[7] };
        var b2: [4]u8 = .{ s[8], s[9], s[10], s[11] };
        aux0 = std.mem.readInt(u32, &b0, .little);
        aux1 = std.mem.readInt(u32, &b1, .little);
        aux2 = std.mem.readInt(u32, &b2, .little);
    }
    const tmp = aux2;
    aux2 = ((aux0 >> 4) & kmask2) | (((tmp >> 4) & kmask1) << 4);
    const aux3: u32 = ((aux1 >> 4) & kmask2) | (((tmp >> 6) & kmask1) << 4);
    aux0 = (aux0 & kmask2) | (((tmp >> 0) & kmask1) << 4);
    aux1 = (aux1 & kmask2) | (((tmp >> 2) & kmask1) << 4);
    // scales = (int8_t*)aux : 16 bytes, little-endian per u32.
    const scales: [16]u8 = blk: {
        var v: [16]u8 = undefined;
        std.mem.writeInt(u32, v[0..4], aux0, .little);
        std.mem.writeInt(u32, v[4..8], aux1, .little);
        std.mem.writeInt(u32, v[8..12], aux2, .little);
        std.mem.writeInt(u32, v[12..16], aux3, .little);
        break :blk v;
    };

    var is: usize = 0;
    var y: usize = 0;
    var m: u16 = 1; // NOT reset per 128-chunk (matches C): cycles 1..128 (u16 so no u8 wrap).
    var q_off: usize = 0;
    var n: usize = 0;
    while (n < QK_K) : (n += 128) {
        for (0..4) |j| {
            const shift: u3 = @intCast(2 * j); // 0,2,4,6 (C uses int; no overflow here)
            const sval: i32 = @as(i32, @as(i8, @bitCast(scales[is]))) - 32;
            is += 1;
            const dl: f32 = d_all * @as(f32, @floatFromInt(sval));
            for (0..16) |l| {
                const raw: u8 = (q[q_off + l] >> shift) & 3;
                const vi: i8 = @intCast(raw);
                const sub: i8 = if (hm[l] & m != 0) 0 else 4;
                const v: i8 = vi - sub;
                out[y + l] = dl * @as(f32, @floatFromInt(v));
            }
            y += 16;
            const sval2: i32 = @as(i32, @as(i8, @bitCast(scales[is]))) - 32;
            is += 1;
            const dl2: f32 = d_all * @as(f32, @floatFromInt(sval2));
            for (0..16) |l| {
                const raw2: u8 = (q[q_off + l + 16] >> shift) & 3;
                const vi2: i8 = @intCast(raw2);
                const sub2: i8 = if (hm[l + 16] & m != 0) 0 else 4;
                const v: i8 = vi2 - sub2;
                out[y + l] = dl2 * @as(f32, @floatFromInt(v));
            }
            y += 16;
            m <<= 1;
        }
        q_off += 32;
    }
}

// ---------------- IQ4_XS: d @0..1; scales_h(u16) @2..3; scales_l[4] @4..7; qs[128] @8..135
pub fn dequantIQ4XS(row: []const u8, out: []f32) void {
    const d = f16_to_f32(row, 0);
    const sh: u16 = @as(u16, row[2]) | (@as(u16, row[3]) << 8);
    const sl = row[4..8];
    const qs = row[8..136];
    var y: usize = 0;
    for (0..8) |ib| {
        const sl_amt: u3 = @intCast(4 * (ib % 2)); // u8 >> wants u3
        const sh_amt: u4 = @intCast(2 * ib);
        const sl4: u16 = sl[ib / 2] >> sl_amt;
        const sh4: u16 = sh >> sh_amt;
        const ls: u16 = (sl4 & 0x0F) | ((sh4 & 3) << 4);
        const dl: f32 = d * @as(f32, @floatFromInt(@as(i32, ls) - 32));
        for (0..16) |j| {
            const q = qs[16 * ib + j];
            out[y + j] = dl * @as(f32, @floatFromInt(kvalues_iq4nl[q & 0x0F]));
            out[y + 16 + j] = dl * @as(f32, @floatFromInt(kvalues_iq4nl[q >> 4]));
        }
        y += 32;
    }
}

// ---------------- IQ4_NL: 8 blocks of (d(2) qs[16]) = 18 bytes each
pub fn dequantIQ4NL(row: []const u8, out: []f32) void {
    var y: usize = 0;
    for (0..8) |i| {
        const base = i * 18;
        const d = f16_to_f32(row, base);
        const qs = row[base + 2 .. base + 18];
        for (0..16) |j| {
            out[y + j] = d * @as(f32, @floatFromInt(kvalues_iq4nl[qs[j] & 0x0F]));
            out[y + 16 + j] = d * @as(f32, @floatFromInt(kvalues_iq4nl[qs[j] >> 4]));
        }
        y += 32;
    }
}

// ---------------- IQ3_S: d @0..1; qs[64] @2..65; qh[8] @66..73; signs[32] @74..105; scales[4] @106..109
pub fn dequantIQ3S(row: []const u8, out: []f32) void {
    const d = f16_to_f32(row, 0);
    const qs = row[2..66];
    const qh = row[66..74];
    const signs = row[74..106];
    const scales = row[106..110];
    var y: usize = 0;
    for (0..4) |p| { // p = ib32/2 = 0..3; each p covers 64 elements
        const sbyte = scales[p];
        const db1: f32 = d * @as(f32, @floatFromInt(1 + 2 * (sbyte & 0x0F)));
        const db2: f32 = d * @as(f32, @floatFromInt(1 + 2 * (sbyte >> 4)));
        // block1: qh[2p], qs[16p..16p+7], signs[8p..8p+3]
        const qh0: u16 = qh[2 * p];
        for (0..4) |l| {
            const b1_amt: u4 = @intCast(8 - 2 * l);
            const b2_amt: u4 = @intCast(7 - 2 * l);
            // C: (qh[0] << (8-2*l)) & 256  (0x100 selects the grid high-byte bit)
            const bit1: u16 = (qh0 << b1_amt) & 0x100;
            const bit2: u16 = (qh0 << b2_amt) & 0x100;
            const g1 = ggml_tables.iq3s_grid[qs[16 * p + 2 * l] | bit1];
            const g2 = ggml_tables.iq3s_grid[qs[16 * p + 2 * l + 1] | bit2];
            const sg = signs[8 * p + l];
            for (0..4) |j| {
                const gb1: [4]u8 = @bitCast(g1);
            const b1: u8 = gb1[j];
                const gb2: [4]u8 = @bitCast(g2);
                const b2: u8 = gb2[j];
                out[y + l * 8 + j] = db1 * @as(f32, @floatFromInt(@as(i32, b1))) * @as(f32, if (sg & kmask_iq2xs[j] != 0) -1.0 else 1.0);
                out[y + l * 8 + 4 + j] = db1 * @as(f32, @floatFromInt(@as(i32, b2))) * @as(f32, if (sg & kmask_iq2xs[4 + j] != 0) -1.0 else 1.0);
            }
        }
        // block2: qh[2p+1], qs[16p+8..16p+15], signs[8p+4..8p+7]
        const qh1: u16 = qh[2 * p + 1];
        for (0..4) |l| {
            const b1_amt: u4 = @intCast(8 - 2 * l);
            const b2_amt: u4 = @intCast(7 - 2 * l);
            // C: (qh[1] << (8-2*l)) & 256
            const bit1: u16 = (qh1 << b1_amt) & 0x100;
            const bit2: u16 = (qh1 << b2_amt) & 0x100;
            const g1 = ggml_tables.iq3s_grid[qs[16 * p + 8 + 2 * l] | bit1];
            const g2 = ggml_tables.iq3s_grid[qs[16 * p + 8 + 2 * l + 1] | bit2];
            const sg = signs[8 * p + 4 + l];
            for (0..4) |j| {
                const gb1: [4]u8 = @bitCast(g1);
            const b1: u8 = gb1[j];
                const gb2: [4]u8 = @bitCast(g2);
                const b2: u8 = gb2[j];
                out[y + 32 + l * 8 + j] = db2 * @as(f32, @floatFromInt(@as(i32, b1))) * @as(f32, if (sg & kmask_iq2xs[j] != 0) -1.0 else 1.0);
                out[y + 32 + l * 8 + 4 + j] = db2 * @as(f32, @floatFromInt(@as(i32, b2))) * @as(f32, if (sg & kmask_iq2xs[4 + j] != 0) -1.0 else 1.0);
            }
        }
        y += 64;
    }
}

// ---------------- IQ3_XXS: d @0..1; qs[64] @2..65; scales_and_signs[32] @66..97
// Port of ggml dequantize_row_iq3_xxs (ggml-quants.c:2575), expression-for-expression:
//   db    = d * (0.5f + (aux32 >> 28)) * 0.5f          (sub-scale from aux32 top 4 bits)
//   signs = ksigns_iq2xs[(aux32 >> 7*l) & 127]         (7-bit sign word per 8-element group)
//   grid  = iq3xxs_grid[qs[2*l+k]] as 4 x u4 values    (byte j = value j)
//   y     = db * grid[j] * (signs & kmask_iq2xs[...] ? -1 : 1)
pub fn dequantIQ3XXS(row: []const u8, out: []f32) void {
    const d = f16_to_f32(row, 0);
    const qs = row[2..66];
    const ss = row[66..98]; // scales_and_signs[32]
    var y: usize = 0;
    for (0..8) |ib32| {
        const aux32: u32 = @bitCast([4]u8{ ss[4 * ib32], ss[4 * ib32 + 1], ss[4 * ib32 + 2], ss[4 * ib32 + 3] });
        const db: f32 = d * (0.5 + @as(f32, @floatFromInt(aux32 >> 28))) * 0.5;
        for (0..4) |l| {
            const sidx: u8 = @intCast((aux32 >> @as(u5, @intCast(7 * l))) & 127);
            const signs: u8 = ggml_tables.ksigns_iq2xs[sidx];
            // C advances qs by 8 per ib32; grid row index is the 2-bit value.
            const g1 = ggml_tables.iq3xxs_grid[qs[8 * ib32 + 2 * l]];
            const g2 = ggml_tables.iq3xxs_grid[qs[8 * ib32 + 2 * l + 1]];
            const gb1: [4]u8 = @bitCast(g1);
            const gb2: [4]u8 = @bitCast(g2);
            for (0..4) |j| {
                out[y + j] = db * @as(f32, @floatFromInt(gb1[j])) * @as(f32, if (signs & ggml_tables.kmask_iq2xs[j] != 0) -1.0 else 1.0);
                out[y + 4 + j] = db * @as(f32, @floatFromInt(gb2[j])) * @as(f32, if (signs & ggml_tables.kmask_iq2xs[4 + j] != 0) -1.0 else 1.0);
            }
            y += 8;
        }
    }
}

// ============================================================ tests
// Reads the real testdata (testdata/udtest) and verifies every row: dot product
// of the dequantized row with A_x.f32 vs the C ground-truth *_expected.f32.
const DequantFn = *const fn (row: []const u8, out: []f32) void;

const TESTDATA = "/mnt/llm-dev/qwen35/testdata/udtest/";

fn readFile(alloc: std.mem.Allocator, io: std.Io, path: []const u8) ![]u8 {
    return std.Io.Dir.cwd().readFileAlloc(io, path, alloc, std.Io.Limit.unlimited);
}

fn asF32(b: []const u8) []const f32 {
    const p: [*]const f32 = @alignCast(@ptrCast(@constCast(&b[0])));
    return p[0 .. b.len / @sizeOf(f32)];
}

fn joinPath(alloc: std.mem.Allocator, name: []const u8, suffix: []const u8) []u8 {
    return std.fmt.allocPrint(alloc, TESTDATA ++ "{s}" ++ "{s}", .{ name, suffix }) catch unreachable;
}

fn checkType(alloc: std.mem.Allocator, io: std.Io, name: []const u8, rowbytes: usize, ax: []const f32, deq: DequantFn) !f32 {
    const pbin = joinPath(alloc, name, ".bin");
    defer alloc.free(pbin);
    const bin = try readFile(alloc, io, pbin);
    defer alloc.free(bin);
    const pexp = joinPath(alloc, name, "_expected.f32");
    defer alloc.free(pexp);
    const exb = try readFile(alloc, io, pexp);
    defer alloc.free(exb);
    const exp = asF32(exb);
    const nrows: usize = bin.len / rowbytes;
    var tmp: [256]f32 = undefined;
    var max_err: f32 = 0;
    var worst_row: usize = 0;
    for (0..nrows) |r| {
        deq(bin[r * rowbytes .. (r + 1) * rowbytes], tmp[0..]);
        var dot: f32 = 0;
        for (0..256) |i| dot += tmp[i] * ax[i];
        const e = @abs(dot - exp[r]);
        if (e > max_err) {
            max_err = e;
            worst_row = r;
        }
    }
    std.debug.print("  {s}: max_err = {} (row {})\n", .{ name, max_err, worst_row });
    return max_err;
}

test "quant_dequant: all match C ground-truth (96 rows x 256)" {
    const alloc = std.testing.allocator;
    const io = std.testing.io;
    const axb = try readFile(alloc, io, TESTDATA ++ "A_x.f32");
    defer alloc.free(axb);
    const ax = asF32(axb);
    try std.testing.expectEqual(@as(usize, 256), ax.len);
    var worst: f32 = 0;
    var e: f32 = try checkType(alloc, io, "Q4_K", 144, ax, dequantQ4K);
    if (e > worst) worst = e;
    e = try checkType(alloc, io, "Q5_K", 176, ax, dequantQ5K);
    if (e > worst) worst = e;
    e = try checkType(alloc, io, "Q6_K", 210, ax, dequantQ6K);
    if (e > worst) worst = e;
    e = try checkType(alloc, io, "Q3_K", 110, ax, dequantQ3K);
    if (e > worst) worst = e;
    e = try checkType(alloc, io, "IQ4_XS", 136, ax, dequantIQ4XS);
    if (e > worst) worst = e;
    e = try checkType(alloc, io, "IQ4_NL", 144, ax, dequantIQ4NL);
    if (e > worst) worst = e;
    e = try checkType(alloc, io, "IQ3_XXS", 98, ax, dequantIQ3XXS);
    if (e > worst) worst = e;
    std.debug.print("  worst overall = {}\n", .{worst});
    try std.testing.expect(worst < 1e-3);
}
