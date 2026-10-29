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
const ggml_tables = @import("iq_tables.zig");
const simd_ref = @import("simd_ref.zig");

const QK_K: usize = 256;

/// Q8_0 block geometry (ggml `block_q8_0`: f16 d + 32 x i8 = 34 bytes).
/// Spelled out here rather than imported from gguf so this file stays a
/// dependency-free reference implementation.
const QK8_0: usize = 32;
const block_q8_0_bytes: usize = 2 + QK8_0;
const QK4_NL: usize = 32;

pub const kvalues_iq4nl = ggml_tables.kvalues_iq4nl;
const kmask_iq2xs = ggml_tables.kmask_iq2xs;
const iq1s_delta: f32 = 0.125;

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

// ---------------- Q4_0: 8 x ( d(u16) @0,2; qs[16] @2..17 ) = 8*18 = 144
// Per 32-elem sub-block: y[i]=((qs[i]&0xF)-8)*d ; y[i+16]=((qs[i]>>4)-8)*d.
pub fn dequantQ4_0(row: []const u8, out: []f32) void {
    var y: usize = 0;
    for (0..8) |ib| {
        const b = ib * 18;
        const d = f16_to_f32(row, b);
        const qs = row[b + 2 .. b + 18];
        for (0..16) |j| out[y + j] = (@as(f32, @floatFromInt(@as(i32, qs[j] & 0x0F))) - 8) * d;
        for (0..16) |j| out[y + 16 + j] = (@as(f32, @floatFromInt(@as(i32, qs[j] >> 4))) - 8) * d;
        y += 32;
    }
}

// ---------------- Q4_1: 8 x ( d(u16) @0,2; m(u16) @2,4; qs[16] @4..19 ) = 8*20 = 160
// Per 32-elem sub-block: y[i]=((qs[i]&0xF))*d + m ; y[i+16]=((qs[i]>>4))*d + m.
pub fn dequantQ4_1(row: []const u8, out: []f32) void {
    var y: usize = 0;
    for (0..8) |ib| {
        const b = ib * 20;
        const d = f16_to_f32(row, b);
        const m = f16_to_f32(row, b + 2);
        const qs = row[b + 4 .. b + 20];
        for (0..16) |j| out[y + j] = @as(f32, @floatFromInt(qs[j] & 0x0F)) * d + m;
        for (0..16) |j| out[y + 16 + j] = @as(f32, @floatFromInt(qs[j] >> 4)) * d + m;
        y += 32;
    }
}

// SIMD helpers: 16 u8 nibbles -> two @Vector(16) f32 lanes (low nibble, high
// nibble), multiplied against x in two f32 lane-accumulators; single @reduce at
// the end.  Sub-block order and low-then-high order match the scalar loop, so
// only the final tree-reduce reorders the summation (rel err ~2e-6).
const V16u8 = @Vector(16, u8);
const V16f32 = @Vector(16, f32);

inline fn niblo(q: V16u8) V16f32 {
    return @as(V16f32, @floatFromInt(q & @as(V16u8, @splat(0x0F))));
}
inline fn nibhi(q: V16u8) V16f32 {
    return @as(V16f32, @floatFromInt(q >> @as(V16u8, @splat(4))));
}

// rowDot over one 256-elem Q4_0 row (144 bytes): sum over sub-blocks of
// ((qs&0xF)-8)*d * x and ((qs>>4)-8)*d * x.  Vectorized.
pub fn rowDotQ4_0(row: []const u8, x: []const f32) f32 {
    var a16: V16f32 = @splat(0);
    var a16b: V16f32 = @splat(0);
    for (0..8) |ib| {
        const b = ib * 18;
        const d = f16_to_f32(row, b);
        const q: V16u8 = @as(*const [16]u8, @ptrCast(row[b + 2 .. b + 18])).*;
        const ylo = (niblo(q) - @as(V16f32, @splat(8))) * @as(V16f32, @splat(d));
        const yhi = (nibhi(q) - @as(V16f32, @splat(8))) * @as(V16f32, @splat(d));
        const xp: [*]const f32 = @ptrCast(x[ib * 32 ..]);
        a16 += ylo * @as(*const [16]f32, @ptrCast(xp)).*;
        a16b += yhi * @as(*const [16]f32, @ptrCast(xp + 16)).*;
    }
    return @reduce(.Add, a16) + @reduce(.Add, a16b);
}

// rowDot over one 256-elem Q4_1 row (160 bytes): sum of ((qs&0xF)*d+m)*x and
// ((qs>>4)*d+m)*x per sub-block.  Vectorized.
pub fn rowDotQ4_1(row: []const u8, x: []const f32) f32 {
    var a16: V16f32 = @splat(0);
    var a16b: V16f32 = @splat(0);
    for (0..8) |ib| {
        const b = ib * 20;
        const d = f16_to_f32(row, b);
        const m = f16_to_f32(row, b + 2);
        const q: V16u8 = @as(*const [16]u8, @ptrCast(row[b + 4 .. b + 20])).*;
        const ylo = niblo(q) * @as(V16f32, @splat(d)) + @as(V16f32, @splat(m));
        const yhi = nibhi(q) * @as(V16f32, @splat(d)) + @as(V16f32, @splat(m));
        const xp: [*]const f32 = @ptrCast(x[ib * 32 ..]);
        a16 += ylo * @as(*const [16]f32, @ptrCast(xp)).*;
        a16b += yhi * @as(*const [16]f32, @ptrCast(xp + 16)).*;
    }
    return @reduce(.Add, a16) + @reduce(.Add, a16b);
}

// Scalar reference (ground truth for the vectorized kernels above; used by the
// bench test to report speedup and by the exactness test to compare row-by-row).
pub fn rowDotQ4_0_scalar(row: []const u8, x: []const f32) f32 {
    var acc: f32 = 0;
    var y: usize = 0;
    for (0..8) |ib| {
        const b = ib * 18;
        const d = f16_to_f32(row, b);
        const qs = row[b + 2 .. b + 18];
        for (0..16) |j| acc += ((@as(f32, @floatFromInt(@as(i32, qs[j] & 0x0F))) - 8) * d) * x[y + j];
        for (0..16) |j| acc += ((@as(f32, @floatFromInt(@as(i32, qs[j] >> 4))) - 8) * d) * x[y + 16 + j];
        y += 32;
    }
    return acc;
}

pub fn rowDotQ4_1_scalar(row: []const u8, x: []const f32) f32 {
    var acc: f32 = 0;
    var y: usize = 0;
    for (0..8) |ib| {
        const b = ib * 20;
        const d = f16_to_f32(row, b);
        const m = f16_to_f32(row, b + 2);
        const qs = row[b + 4 .. b + 20];
        for (0..16) |j| acc += (@as(f32, @floatFromInt(qs[j] & 0x0F)) * d + m) * x[y + j];
        for (0..16) |j| acc += (@as(f32, @floatFromInt(qs[j] >> 4)) * d + m) * x[y + 16 + j];
        y += 32;
    }
    return acc;
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

// ============================================================
// Quantized-domain row dot products (Phase 2a).
//
// Each rowDotXXX computes sum_c W[c] * x[c] for ONE packed 256-element row,
// reading the packed GGUF bytes directly and accumulating in f32 — the same
// per-element expressions as dequantXXX, inlined against x (no whole-row
// f32 materialization, per the "no dequant-then-matmul" requirement).
// rowDotXXX(row, x) must equal dot(dequantXXX(row), x).
// ============================================================

// Q4_K: d/dmin @0,2; scales[12] @4..15; qs[128] @16..143 (per 256-elem row)
pub fn rowDotQ4K(row: []const u8, x: []const f32) f32 {
    const d = f16_to_f32(row, 0);
    const min = f16_to_f32(row, 2);
    const scales = row[4..16];
    var q: []const u8 = row[16..144];
    var y: usize = 0;
    var is: usize = 0;
    var acc: f32 = 0;
    for (0..4) |_| {
        const p0 = get_scale_min_k4(is, scales);
        const d1: f32 = d * @as(f32, @floatFromInt(@as(i32, p0.sc)));
        const m1: f32 = min * @as(f32, @floatFromInt(@as(i32, p0.m)));
        const p1 = get_scale_min_k4(is + 1, scales);
        const d2: f32 = d * @as(f32, @floatFromInt(@as(i32, p1.sc)));
        const m2: f32 = min * @as(f32, @floatFromInt(@as(i32, p1.m)));
        for (0..32) |l| acc += (d1 * @as(f32, @floatFromInt(@as(i32, q[l] & 0x0F))) - m1) * x[y + l];
        for (0..32) |l| acc += (d2 * @as(f32, @floatFromInt(@as(i32, q[l] >> 4))) - m2) * x[y + 32 + l];
        y += 64;
        q = q[32..];
        is += 2;
    }
    return acc;
}

// Q5_K: d/dmin @0,2; scales[12] @4..15; qh[32] @16..47; qs[128] @48..175
pub fn rowDotQ5K(row: []const u8, x: []const f32) f32 {
    const d = f16_to_f32(row, 0);
    const min = f16_to_f32(row, 2);
    const scales = row[4..16];
    const qh = row[16..48];
    var ql: []const u8 = row[48..176];
    var y: usize = 0;
    var is: usize = 0;
    var bm1: u16 = 1;
    var bm2: u16 = 2;
    var acc: f32 = 0;
    for (0..4) |_| {
        const p0 = get_scale_min_k4(is, scales);
        const d1: f32 = d * @as(f32, @floatFromInt(@as(i32, p0.sc)));
        const m1: f32 = min * @as(f32, @floatFromInt(@as(i32, p0.m)));
        const p1 = get_scale_min_k4(is + 1, scales);
        const d2: f32 = d * @as(f32, @floatFromInt(@as(i32, p1.sc)));
        const m2: f32 = min * @as(f32, @floatFromInt(@as(i32, p1.m)));
        for (0..32) |l| {
            const hi: u8 = if (qh[l] & bm1 != 0) 16 else 0;
            acc += (d1 * @as(f32, @floatFromInt(@as(i32, ql[l] & 0x0F | hi))) - m1) * x[y + l];
        }
        for (0..32) |l| {
            const hi: u8 = if (qh[l] & bm2 != 0) 16 else 0;
            acc += (d2 * @as(f32, @floatFromInt(@as(i32, ql[l] >> 4 | hi))) - m2) * x[y + 32 + l];
        }
        y += 64;
        ql = ql[32..];
        is += 2;
        bm1 <<= 2;
        bm2 <<= 2;
    }
    return acc;
}

// Q6_K: ql[128] @0..127; qh[64] @128..191; scales[i8:16] @192..207; d @208..209
pub fn rowDotQ6K(row: []const u8, x: []const f32) f32 {
    const d = f16_to_f32(row, 208);
    const ql = row[0..128];
    const qh = row[128..192];
    var sc: [16]i8 = undefined;
    for (0..16) |i| sc[i] = @as(i8, @bitCast(row[192 + i]));
    var acc: f32 = 0;
    var n: usize = 0;
    while (n < QK_K) : (n += 128) {
        const qlb: usize = n / 2;
        const qhb: usize = n / 4;
        const scb: usize = n / 16;
        for (0..32) |l| {
            const is: usize = l / 16;
            const q1u: u8 = (ql[qlb + l] & 0x0F) | (((qh[qhb + l] >> 0) & 3) << 4);
            const q2u: u8 = (ql[qlb + l + 32] & 0x0F) | (((qh[qhb + l] >> 2) & 3) << 4);
            const q3u: u8 = (ql[qlb + l] >> 4) | (((qh[qhb + l] >> 4) & 3) << 4);
            const q4u: u8 = (ql[qlb + l + 32] >> 4) | (((qh[qhb + l] >> 6) & 3) << 4);
            const q1: i8 = @as(i8, @bitCast(q1u)) - 32;
            const q2: i8 = @as(i8, @bitCast(q2u)) - 32;
            const q3: i8 = @as(i8, @bitCast(q3u)) - 32;
            const q4: i8 = @as(i8, @bitCast(q4u)) - 32;
            acc += d * @as(f32, @floatFromInt(sc[scb + is + 0])) * @as(f32, @floatFromInt(q1)) * x[n + l];
            acc += d * @as(f32, @floatFromInt(sc[scb + is + 2])) * @as(f32, @floatFromInt(q2)) * x[n + l + 32];
            acc += d * @as(f32, @floatFromInt(sc[scb + is + 4])) * @as(f32, @floatFromInt(q3)) * x[n + l + 64];
            acc += d * @as(f32, @floatFromInt(sc[scb + is + 6])) * @as(f32, @floatFromInt(q4)) * x[n + l + 96];
        }
    }
    return acc;
}

// Q3_K: hmask[32] @0..31; qs[64] @32..95; scales[12] @96..107; d @108..109
pub fn rowDotQ3K(row: []const u8, x: []const f32) f32 {
    const d_all = f16_to_f32(row, 108);
    const q = row[32..96];
    const hm = row[0..32];
    const s = row[96..108];
    const kmask1: u32 = 0x03030303;
    const kmask2: u32 = 0x0F0F0F0F;
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
    var m: u16 = 1; // NOT reset per 128-chunk (matches C): cycles 1..128
    var q_off: usize = 0;
    var acc: f32 = 0;
    var nch: usize = 0;
    while (nch < QK_K) : (nch += 128) {
        for (0..4) |j| {
            const shift: u3 = @intCast(2 * j);
            const sval: i32 = @as(i32, @as(i8, @bitCast(scales[is]))) - 32;
            is += 1;
            const dl: f32 = d_all * @as(f32, @floatFromInt(sval));
            for (0..16) |l| {
                const raw: u8 = (q[q_off + l] >> shift) & 3;
                const vi: i8 = @intCast(raw);
                const sub: i8 = if (hm[l] & m != 0) 0 else 4;
                const v: i8 = vi - sub;
                acc += dl * @as(f32, @floatFromInt(v)) * x[y + l];
            }
            y += 16;
            const sval2: i32 = @as(i32, @as(i8, @bitCast(scales[is]))) - 32;
            is += 1;
            const dl2: f32 = d_all * @as(f32, @floatFromInt(sval2));
            for (0..16) |l| {
                const raw2: u8 = (q[q_off + l + 16] >> shift) & 3;
                const vi2: i8 = @intCast(raw2);
                const sub2: i8 = if (hm[l + 16] & m != 0) 0 else 4;
                const v2: i8 = vi2 - sub2;
                acc += dl2 * @as(f32, @floatFromInt(v2)) * x[y + l];
            }
            y += 16;
            m <<= 1;
        }
        q_off += 32;
    }
    return acc;
}

// IQ4_XS: d @0..1; scales_h(u16) @2..3; scales_l[4] @4..7; qs[128] @8..135
pub fn rowDotIQ4XS(row: []const u8, x: []const f32) f32 {
    const d = f16_to_f32(row, 0);
    const sh: u16 = @as(u16, row[2]) | (@as(u16, row[3]) << 8);
    const sl = row[4..8];
    const qs = row[8..136];
    var y: usize = 0;
    var acc: f32 = 0;
    for (0..8) |ib| {
        const sl_amt: u3 = @intCast(4 * (ib % 2));
        const sh_amt: u4 = @intCast(2 * ib);
        const sl4: u16 = sl[ib / 2] >> sl_amt;
        const sh4: u16 = sh >> sh_amt;
        const ls: u16 = (sl4 & 0x0F) | ((sh4 & 3) << 4);
        const dl: f32 = d * @as(f32, @floatFromInt(@as(i32, ls) - 32));
        for (0..16) |j| {
            const q = qs[16 * ib + j];
            acc += dl * @as(f32, @floatFromInt(kvalues_iq4nl[q & 0x0F])) * x[y + j];
            acc += dl * @as(f32, @floatFromInt(kvalues_iq4nl[q >> 4])) * x[y + 16 + j];
        }
        y += 32;
    }
    return acc;
}

// IQ4_NL: 8 blocks of (d(2) qs[16]) = 18 bytes each
pub fn rowDotIQ4NL(row: []const u8, x: []const f32) f32 {
    var y: usize = 0;
    var acc: f32 = 0;
    for (0..8) |i| {
        const base = i * 18;
        const d = f16_to_f32(row, base);
        const qs = row[base + 2 .. base + 18];
        for (0..16) |j| {
            acc += d * @as(f32, @floatFromInt(kvalues_iq4nl[qs[j] & 0x0F])) * x[y + j];
            acc += d * @as(f32, @floatFromInt(kvalues_iq4nl[qs[j] >> 4])) * x[y + 16 + j];
        }
        y += 32;
    }
    return acc;
}

// IQ3_S: d @0..1; qs[64] @2..65; qh[8] @66..73; signs[32] @74..105; scales[4] @106..109
pub fn rowDotIQ3S(row: []const u8, x: []const f32) f32 {
    const d = f16_to_f32(row, 0);
    const qs = row[2..66];
    const qh = row[66..74];
    const signs = row[74..106];
    const scales = row[106..110];
    var y: usize = 0;
    var acc: f32 = 0;
    for (0..4) |p| { // p = ib32/2 = 0..3; each p covers 64 elements
        const sbyte = scales[p];
        const db1: f32 = d * @as(f32, @floatFromInt(1 + 2 * (sbyte & 0x0F)));
        const db2: f32 = d * @as(f32, @floatFromInt(1 + 2 * (sbyte >> 4)));
        const qh0: u16 = qh[2 * p];
        for (0..4) |l| {
            const b1_amt: u4 = @intCast(8 - 2 * l);
            const b2_amt: u4 = @intCast(7 - 2 * l);
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
                acc += db1 * @as(f32, @floatFromInt(@as(i32, b1))) * @as(f32, if (sg & kmask_iq2xs[j] != 0) -1.0 else 1.0) * x[y + l * 8 + j];
                acc += db1 * @as(f32, @floatFromInt(@as(i32, b2))) * @as(f32, if (sg & kmask_iq2xs[4 + j] != 0) -1.0 else 1.0) * x[y + l * 8 + 4 + j];
            }
        }
        const qh1: u16 = qh[2 * p + 1];
        for (0..4) |l| {
            const b1_amt: u4 = @intCast(8 - 2 * l);
            const b2_amt: u4 = @intCast(7 - 2 * l);
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
                acc += db2 * @as(f32, @floatFromInt(@as(i32, b1))) * @as(f32, if (sg & kmask_iq2xs[j] != 0) -1.0 else 1.0) * x[y + 32 + l * 8 + j];
                acc += db2 * @as(f32, @floatFromInt(@as(i32, b2))) * @as(f32, if (sg & kmask_iq2xs[4 + j] != 0) -1.0 else 1.0) * x[y + 32 + l * 8 + 4 + j];
            }
        }
        y += 64;
    }
    return acc;
}

// IQ3_XXS: d @0..1; qs[64] @2..65; scales_and_signs[32] @66..97
pub fn rowDotIQ3XXS(row: []const u8, x: []const f32) f32 {
    const d = f16_to_f32(row, 0);
    const qs = row[2..66];
    const ss = row[66..98];
    var y: usize = 0;
    var acc: f32 = 0;
    for (0..8) |ib32| {
        const aux32: u32 = @bitCast([4]u8{ ss[4 * ib32], ss[4 * ib32 + 1], ss[4 * ib32 + 2], ss[4 * ib32 + 3] });
        const db: f32 = d * (0.5 + @as(f32, @floatFromInt(aux32 >> 28))) * 0.5;
        for (0..4) |l| {
            const sidx: u8 = @intCast((aux32 >> @as(u5, @intCast(7 * l))) & 127);
            const signs: u8 = ggml_tables.ksigns_iq2xs[sidx];
            const g1 = ggml_tables.iq3xxs_grid[qs[8 * ib32 + 2 * l]];
            const g2 = ggml_tables.iq3xxs_grid[qs[8 * ib32 + 2 * l + 1]];
            const gb1: [4]u8 = @bitCast(g1);
            const gb2: [4]u8 = @bitCast(g2);
            for (0..4) |j| {
                acc += db * @as(f32, @floatFromInt(gb1[j])) * @as(f32, if (signs & ggml_tables.kmask_iq2xs[j] != 0) -1.0 else 1.0) * x[y + j];
                acc += db * @as(f32, @floatFromInt(gb2[j])) * @as(f32, if (signs & ggml_tables.kmask_iq2xs[4 + j] != 0) -1.0 else 1.0) * x[y + 4 + j];
            }
            y += 8;
        }
    }
    return acc;
}

// =====================================================================
// Phase 2b: 6 missing 27B types (IQ2_S, IQ2_XXS, Q2_K, IQ2_XS, IQ1_S, IQ1_M)
// dequantXXX + rowDotXXX, mirroring ggml-quants.c expression-for-expression.
// =====================================================================

// ---------------- IQ2_XXS: d(2) @0; qs[32]u16 @2 (64B)
// C: dequantize_row_iq2_xxs
pub fn dequantIQ2XXS(row: []const u8, out: []f32) void {
    const d = f16_to_f32(row, 0);
    const qs = row[2..66];
    var y: usize = 0;
    for (0..8) |ib32| {
        const a = qs[8 * ib32 .. 8 * ib32 + 8];
        const aux1: u32 = @bitCast([4]u8{ a[4], a[5], a[6], a[7] });
        const db = d * (0.5 + @as(f32, @floatFromInt(aux1 >> 28))) * 0.25;
        for (0..4) |l| {
            const grid: [8]u8 = @bitCast(ggml_tables.iq2xxs_grid[a[l]]);
            const signs: u8 = ggml_tables.ksigns_iq2xs[(aux1 >> @as(u5, @intCast(7 * l))) & 127];
            for (0..8) |j| {
                out[y + j] = db * @as(f32, @floatFromInt(grid[j])) * @as(f32, if (signs & ggml_tables.kmask_iq2xs[j] != 0) -1.0 else 1.0);
            }
            y += 8;
        }
    }
}

pub fn rowDotIQ2XXS(row: []const u8, x: []const f32) f32 {
    const d = f16_to_f32(row, 0);
    const qs = row[2..66];
    var y: usize = 0;
    var acc: f32 = 0;
    for (0..8) |ib32| {
        const a = qs[8 * ib32 .. 8 * ib32 + 8];
        const aux1: u32 = @bitCast([4]u8{ a[4], a[5], a[6], a[7] });
        const db = d * (0.5 + @as(f32, @floatFromInt(aux1 >> 28))) * 0.25;
        for (0..4) |l| {
            const grid: [8]u8 = @bitCast(ggml_tables.iq2xxs_grid[a[l]]);
            const signs: u8 = ggml_tables.ksigns_iq2xs[(aux1 >> @as(u5, @intCast(7 * l))) & 127];
            for (0..8) |j| {
                acc += db * @as(f32, @floatFromInt(grid[j])) * @as(f32, if (signs & ggml_tables.kmask_iq2xs[j] != 0) -1.0 else 1.0) * x[y + j];
            }
            y += 8;
        }
    }
    return acc;
}

// ---------------- IQ2_XS: d(2)@0; qs[32]u16@2 (64B); scales[8]@66
// C: dequantize_row_iq2_xs
pub fn dequantIQ2XS(row: []const u8, out: []f32) void {
    const d = f16_to_f32(row, 0);
    const qs16 = row[2..66];
    const scales = row[66..74];
    var y: usize = 0;
    for (0..8) |ib32| {
        const sc = scales[ib32];
        const db0 = d * (0.5 + @as(f32, @floatFromInt(sc & 0xf))) * 0.25;
        const db1 = d * (0.5 + @as(f32, @floatFromInt(sc >> 4))) * 0.25;
        for (0..4) |l| {
            const q16: u16 = @bitCast([2]u8{ qs16[(4 * ib32 + l) * 2], qs16[(4 * ib32 + l) * 2 + 1] });
            const grid: [8]u8 = @bitCast(ggml_tables.iq2xs_grid[q16 & 511]);
            const signs: u8 = ggml_tables.ksigns_iq2xs[q16 >> 9];
            const db: f32 = if (l / 2 == 0) db0 else db1;
            for (0..8) |j| {
                out[y + j] = db * @as(f32, @floatFromInt(grid[j])) * @as(f32, if (signs & ggml_tables.kmask_iq2xs[j] != 0) -1.0 else 1.0);
            }
            y += 8;
        }
    }
}

pub fn rowDotIQ2XS(row: []const u8, x: []const f32) f32 {
    const d = f16_to_f32(row, 0);
    const qs16 = row[2..66];
    const scales = row[66..74];
    var y: usize = 0;
    var acc: f32 = 0;
    for (0..8) |ib32| {
        const sc = scales[ib32];
        const db0 = d * (0.5 + @as(f32, @floatFromInt(sc & 0xf))) * 0.25;
        const db1 = d * (0.5 + @as(f32, @floatFromInt(sc >> 4))) * 0.25;
        for (0..4) |l| {
            const q16: u16 = @bitCast([2]u8{ qs16[(4 * ib32 + l) * 2], qs16[(4 * ib32 + l) * 2 + 1] });
            const grid: [8]u8 = @bitCast(ggml_tables.iq2xs_grid[q16 & 511]);
            const signs: u8 = ggml_tables.ksigns_iq2xs[q16 >> 9];
            const db: f32 = if (l / 2 == 0) db0 else db1;
            for (0..8) |j| {
                acc += db * @as(f32, @floatFromInt(grid[j])) * @as(f32, if (signs & ggml_tables.kmask_iq2xs[j] != 0) -1.0 else 1.0) * x[y + j];
            }
            y += 8;
        }
    }
    return acc;
}

// ---------------- IQ2_S: d(2)@0; qs[64]@2; qh[8]@66; scales[8]@74
// signs = qs+32 (row[34..66])
// C: dequantize_row_iq2_s
pub fn dequantIQ2S(row: []const u8, out: []f32) void {
    const d = f16_to_f32(row, 0);
    var y: usize = 0;
    for (0..8) |ib32| {
        const sc = row[74 + ib32];
        const db0 = d * (0.5 + @as(f32, @floatFromInt(sc & 0xf))) * 0.25;
        const db1 = d * (0.5 + @as(f32, @floatFromInt(sc >> 4))) * 0.25;
        const qh_val: u16 = row[66 + ib32];
        for (0..4) |l| {
            const dl: f32 = if (l / 2 == 0) db0 else db1;
            const qsv = row[2 + 4 * ib32 + l];
            const shifted: u16 = (qh_val << @as(u4, @intCast(8 - 2 * l))) & 0x300;
            const idx: u16 = qsv | shifted;
            const grid: [8]u8 = @bitCast(ggml_tables.iq2s_grid[idx]);
            const sign_byte = row[34 + 4 * ib32 + l];
            for (0..8) |j| {
                out[y + j] = dl * @as(f32, @floatFromInt(grid[j])) * @as(f32, if (sign_byte & ggml_tables.kmask_iq2xs[j] != 0) -1.0 else 1.0);
            }
            y += 8;
        }
    }
}

pub fn rowDotIQ2S(row: []const u8, x: []const f32) f32 {
    const d = f16_to_f32(row, 0);
    var y: usize = 0;
    var acc: f32 = 0;
    for (0..8) |ib32| {
        const sc = row[74 + ib32];
        const db0 = d * (0.5 + @as(f32, @floatFromInt(sc & 0xf))) * 0.25;
        const db1 = d * (0.5 + @as(f32, @floatFromInt(sc >> 4))) * 0.25;
        const qh_val: u16 = row[66 + ib32];
        for (0..4) |l| {
            const dl: f32 = if (l / 2 == 0) db0 else db1;
            const qsv = row[2 + 4 * ib32 + l];
            const shifted: u16 = (qh_val << @as(u4, @intCast(8 - 2 * l))) & 0x300;
            const idx: u16 = qsv | shifted;
            const grid: [8]u8 = @bitCast(ggml_tables.iq2s_grid[idx]);
            const sign_byte = row[34 + 4 * ib32 + l];
            for (0..8) |j| {
                acc += dl * @as(f32, @floatFromInt(grid[j])) * @as(f32, if (sign_byte & ggml_tables.kmask_iq2xs[j] != 0) -1.0 else 1.0) * x[y + j];
            }
            y += 8;
        }
    }
    return acc;
}

// ---------------- Q2_K: scales[16]@0; qs[64]@16; d@80 dmin@82
// C: dequantize_row_q2_K
pub fn dequantQ2K(row: []const u8, out: []f32) void {
    const d = f16_to_f32(row, 80);
    const min = f16_to_f32(row, 82);
    const scales = row[0..16];
    var y: usize = 0;
    var is: usize = 0;
    for (0..2) |chunk| {
        const q = row[16 + 32 * chunk .. 16 + 32 * chunk + 32];
        for (0..4) |j| {
            const sh: u3 = @intCast(j * 2);
            const sc0 = scales[is];
            is += 1;
            const dl0: f32 = d * @as(f32, @floatFromInt(sc0 & 0xf));
            const ml0: f32 = min * @as(f32, @floatFromInt(sc0 >> 4));
            for (0..16) |l| {
                out[y + l] = dl0 * @as(f32, @floatFromInt((q[l] >> sh) & 3)) - ml0;
            }
            const sc1 = scales[is];
            is += 1;
            const dl1: f32 = d * @as(f32, @floatFromInt(sc1 & 0xf));
            const ml1: f32 = min * @as(f32, @floatFromInt(sc1 >> 4));
            for (0..16) |l| {
                out[y + 16 + l] = dl1 * @as(f32, @floatFromInt((q[l + 16] >> sh) & 3)) - ml1;
            }
            y += 32;
        }
    }
}

pub fn rowDotQ2K(row: []const u8, x: []const f32) f32 {
    const d = f16_to_f32(row, 80);
    const min = f16_to_f32(row, 82);
    const scales = row[0..16];
    var y: usize = 0;
    var is: usize = 0;
    var acc: f32 = 0;
    for (0..2) |chunk| {
        const q = row[16 + 32 * chunk .. 16 + 32 * chunk + 32];
        for (0..4) |j| {
            const sh: u3 = @intCast(j * 2);
            const sc0 = scales[is];
            is += 1;
            const dl0: f32 = d * @as(f32, @floatFromInt(sc0 & 0xf));
            const ml0: f32 = min * @as(f32, @floatFromInt(sc0 >> 4));
            for (0..16) |l| {
                acc += (dl0 * @as(f32, @floatFromInt((q[l] >> sh) & 3)) - ml0) * x[y + l];
            }
            const sc1 = scales[is];
            is += 1;
            const dl1: f32 = d * @as(f32, @floatFromInt(sc1 & 0xf));
            const ml1: f32 = min * @as(f32, @floatFromInt(sc1 >> 4));
            for (0..16) |l| {
                acc += (dl1 * @as(f32, @floatFromInt((q[l + 16] >> sh) & 3)) - ml1) * x[y + 16 + l];
            }
            y += 32;
        }
    }
    return acc;
}

// ---------------- IQ1_S: d(2)@0; qs[32]@2; qh[8]u16@34
// C: dequantize_row_iq1_s
pub fn dequantIQ1S(row: []const u8, out: []f32) void {
    const d = f16_to_f32(row, 0);
    const qs = row[2..34];
    const qh = row[34..50];
    var y: usize = 0;
    for (0..8) |ib| {
        const qh16: u16 = @bitCast([2]u8{ qh[2 * ib], qh[2 * ib + 1] });
        const dl = d * @as(f32, @floatFromInt(2 * ((qh16 >> 12) & 7) + 1));
        const delta: f32 = if (qh16 & 0x8000 != 0) -iq1s_delta else iq1s_delta;
        for (0..4) |l| {
            const idx: u16 = qs[4 * ib + l] | @as(u16, ((qh16 >> @as(u4, @intCast(3 * l))) & 7) << 8);
            const grid: [8]i8 = @bitCast(ggml_tables.iq1s_grid[idx]);
            for (0..8) |j| {
                out[y + j] = dl * (@as(f32, @floatFromInt(grid[j])) + delta);
            }
            y += 8;
        }
    }
}

pub fn rowDotIQ1S(row: []const u8, x: []const f32) f32 {
    const d = f16_to_f32(row, 0);
    const qs = row[2..34];
    const qh = row[34..50];
    var y: usize = 0;
    var acc: f32 = 0;
    for (0..8) |ib| {
        const qh16: u16 = @bitCast([2]u8{ qh[2 * ib], qh[2 * ib + 1] });
        const dl = d * @as(f32, @floatFromInt(2 * ((qh16 >> 12) & 7) + 1));
        const delta: f32 = if (qh16 & 0x8000 != 0) -iq1s_delta else iq1s_delta;
        for (0..4) |l| {
            const idx: u16 = qs[4 * ib + l] | @as(u16, ((qh16 >> @as(u4, @intCast(3 * l))) & 7) << 8);
            const grid: [8]i8 = @bitCast(ggml_tables.iq1s_grid[idx]);
            for (0..8) |j| {
                acc += dl * (@as(f32, @floatFromInt(grid[j])) + delta) * x[y + j];
            }
            y += 8;
        }
    }
    return acc;
}

// ---------------- IQ1_M: qs[32]@0; qh[16]@32; scales[8]@48 (d packed from scales)
// C: dequantize_row_iq1_m
pub fn dequantIQ1M(row: []const u8, out: []f32) void {
    const sc = row[48..56];
    const s0: u16 = @bitCast([2]u8{ sc[0], sc[1] });
    const s1: u16 = @bitCast([2]u8{ sc[2], sc[3] });
    const s2: u16 = @bitCast([2]u8{ sc[4], sc[5] });
    const s3: u16 = @bitCast([2]u8{ sc[6], sc[7] });
    const scale_u16: u16 = (s0 >> 12) | ((s1 >> 8) & 0x00f0) | ((s2 >> 4) & 0x0f00) | (s3 & 0xf000);
    const d: f32 = @as(f32, @floatCast(@as(f16, @bitCast(scale_u16))));
    const sc_u16: [4]u16 = .{ s0, s1, s2, s3 };
    const qs = row[0..32];
    const qh = row[32..48];
    var y: usize = 0;
    for (0..8) |ib| {
        const sh: u16 = sc_u16[ib / 2];
        const dl1 = d * @as(f32, @floatFromInt(2 * ((sh >> @as(u4, @intCast(6 * (ib % 2)))) & 7) + 1));
        const dl2 = d * @as(f32, @floatFromInt(2 * ((sh >> @as(u4, @intCast(6 * (ib % 2) + 3))) & 7) + 1));
        const qh0: u16 = qh[2 * ib];
        const qh1: u16 = qh[2 * ib + 1];
        const idx0: u16 = qs[4 * ib + 0] | @as(u16, ((qh0 << 8) & 0x700));
        const idx1: u16 = qs[4 * ib + 1] | @as(u16, ((qh0 << 4) & 0x700));
        const idx2: u16 = qs[4 * ib + 2] | @as(u16, ((qh1 << 8) & 0x700));
        const idx3: u16 = qs[4 * ib + 3] | @as(u16, ((qh1 << 4) & 0x700));
        const d0: f32 = if (qh0 & 0x08 != 0) -iq1s_delta else iq1s_delta;
        const d1: f32 = if (qh0 & 0x80 != 0) -iq1s_delta else iq1s_delta;
        const d2: f32 = if (qh1 & 0x08 != 0) -iq1s_delta else iq1s_delta;
        const d3: f32 = if (qh1 & 0x80 != 0) -iq1s_delta else iq1s_delta;
        const g0: [8]i8 = @bitCast(ggml_tables.iq1s_grid[idx0]);
        const g1: [8]i8 = @bitCast(ggml_tables.iq1s_grid[idx1]);
        const g2: [8]i8 = @bitCast(ggml_tables.iq1s_grid[idx2]);
        const g3: [8]i8 = @bitCast(ggml_tables.iq1s_grid[idx3]);
        for (0..8) |j| {
            out[y + j] = dl1 * (@as(f32, @floatFromInt(g0[j])) + d0);
            out[y + 8 + j] = dl1 * (@as(f32, @floatFromInt(g1[j])) + d1);
            out[y + 16 + j] = dl2 * (@as(f32, @floatFromInt(g2[j])) + d2);
            out[y + 24 + j] = dl2 * (@as(f32, @floatFromInt(g3[j])) + d3);
        }
        y += 32;
    }
}

pub fn rowDotIQ1M(row: []const u8, x: []const f32) f32 {
    const sc = row[48..56];
    const s0: u16 = @bitCast([2]u8{ sc[0], sc[1] });
    const s1: u16 = @bitCast([2]u8{ sc[2], sc[3] });
    const s2: u16 = @bitCast([2]u8{ sc[4], sc[5] });
    const s3: u16 = @bitCast([2]u8{ sc[6], sc[7] });
    const scale_u16: u16 = (s0 >> 12) | ((s1 >> 8) & 0x00f0) | ((s2 >> 4) & 0x0f00) | (s3 & 0xf000);
    const d: f32 = @as(f32, @floatCast(@as(f16, @bitCast(scale_u16))));
    const sc_u16: [4]u16 = .{ s0, s1, s2, s3 };
    const qs = row[0..32];
    const qh = row[32..48];
    var y: usize = 0;
    var acc: f32 = 0;
    for (0..8) |ib| {
        const sh: u16 = sc_u16[ib / 2];
        const dl1 = d * @as(f32, @floatFromInt(2 * ((sh >> @as(u4, @intCast(6 * (ib % 2)))) & 7) + 1));
        const dl2 = d * @as(f32, @floatFromInt(2 * ((sh >> @as(u4, @intCast(6 * (ib % 2) + 3))) & 7) + 1));
        const qh0: u16 = qh[2 * ib];
        const qh1: u16 = qh[2 * ib + 1];
        const idx0: u16 = qs[4 * ib + 0] | @as(u16, ((qh0 << 8) & 0x700));
        const idx1: u16 = qs[4 * ib + 1] | @as(u16, ((qh0 << 4) & 0x700));
        const idx2: u16 = qs[4 * ib + 2] | @as(u16, ((qh1 << 8) & 0x700));
        const idx3: u16 = qs[4 * ib + 3] | @as(u16, ((qh1 << 4) & 0x700));
        const d0: f32 = if (qh0 & 0x08 != 0) -iq1s_delta else iq1s_delta;
        const d1: f32 = if (qh0 & 0x80 != 0) -iq1s_delta else iq1s_delta;
        const d2: f32 = if (qh1 & 0x08 != 0) -iq1s_delta else iq1s_delta;
        const d3: f32 = if (qh1 & 0x80 != 0) -iq1s_delta else iq1s_delta;
        const g0: [8]i8 = @bitCast(ggml_tables.iq1s_grid[idx0]);
        const g1: [8]i8 = @bitCast(ggml_tables.iq1s_grid[idx1]);
        const g2: [8]i8 = @bitCast(ggml_tables.iq1s_grid[idx2]);
        const g3: [8]i8 = @bitCast(ggml_tables.iq1s_grid[idx3]);
        for (0..8) |j| {
            acc += dl1 * (@as(f32, @floatFromInt(g0[j])) + d0) * x[y + j];
            acc += dl1 * (@as(f32, @floatFromInt(g1[j])) + d1) * x[y + 8 + j];
            acc += dl2 * (@as(f32, @floatFromInt(g2[j])) + d2) * x[y + 16 + j];
            acc += dl2 * (@as(f32, @floatFromInt(g3[j])) + d3) * x[y + 24 + j];
        }
        y += 32;
    }
    return acc;
}

const DequantFn = *const fn (row: []const u8, out: []f32) void;
const RowDotFn = *const fn (row: []const u8, x: []const f32) f32;

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

test "quant_dequant: rowDot kernels match dot(dequant(row)) (8 types x 96 rows)" {
    const alloc = std.testing.allocator;
    const io = std.testing.io;
    const axb = try readFile(alloc, io, TESTDATA ++ "A_x.f32");
    defer alloc.free(axb);
    const ax = asF32(axb);

    var worst: f32 = 0;
    var worst_name: []const u8 = "";
    const Case = struct { name: []const u8, rb: usize, dot: RowDotFn, deq: DequantFn };
    const cases: []const Case = &.{
        .{ .name = "Q4_0", .rb = 144, .dot = rowDotQ4_0, .deq = dequantQ4_0 },
        .{ .name = "Q4_1", .rb = 160, .dot = rowDotQ4_1, .deq = dequantQ4_1 },
        .{ .name = "Q4_K", .rb = 144, .dot = rowDotQ4K, .deq = dequantQ4K },
        .{ .name = "Q5_K", .rb = 176, .dot = rowDotQ5K, .deq = dequantQ5K },
        .{ .name = "Q6_K", .rb = 210, .dot = rowDotQ6K, .deq = dequantQ6K },
        .{ .name = "Q3_K", .rb = 110, .dot = rowDotQ3K, .deq = dequantQ3K },
        .{ .name = "IQ4_XS", .rb = 136, .dot = rowDotIQ4XS, .deq = dequantIQ4XS },
        .{ .name = "IQ4_NL", .rb = 144, .dot = rowDotIQ4NL, .deq = dequantIQ4NL },
        .{ .name = "IQ3_S", .rb = 110, .dot = rowDotIQ3S, .deq = dequantIQ3S },
        .{ .name = "IQ3_XXS", .rb = 98, .dot = rowDotIQ3XXS, .deq = dequantIQ3XXS },
    };
    for (cases) |c| {
        const pbin = joinPath(alloc, c.name, ".bin");
        defer alloc.free(pbin);
        const bin = try readFile(alloc, io, pbin);
        defer alloc.free(bin);
        const nrows: usize = bin.len / c.rb;
        var tmp: [256]f32 = undefined;
        var max_err: f32 = 0;
        for (0..nrows) |r| {
            const row = bin[r * c.rb .. (r + 1) * c.rb];
            const direct = c.dot(row, ax);
            c.deq(row, tmp[0..]);
            var ref: f32 = 0;
            for (0..256) |i| ref += tmp[i] * ax[i];
            const e = @abs(direct - ref) / @max(@abs(ref), 1);
            if (e > max_err) max_err = e;
        }
        std.debug.print("  rowDot {s}: max rel err = {}\n", .{ c.name, max_err });
        if (max_err > worst) {
            worst = max_err;
            worst_name = c.name;
        }
    }
    std.debug.print("  rowDot worst overall = {} ({s})\n", .{ worst, worst_name });
    try std.testing.expect(worst < 1e-3);
}

test "quant_dequant: all match C ground-truth (96 rows x 256)" {
    const alloc = std.testing.allocator;
    const io = std.testing.io;
    const axb = try readFile(alloc, io, TESTDATA ++ "A_x.f32");
    defer alloc.free(axb);
    const ax = asF32(axb);
    try std.testing.expectEqual(@as(usize, 256), ax.len);
    var worst: f32 = 0;
    var e: f32 = try checkType(alloc, io, "Q4_0", 144, ax, dequantQ4_0);
    if (e > worst) worst = e;
    e = try checkType(alloc, io, "Q4_1", 160, ax, dequantQ4_1);
    if (e > worst) worst = e;
    e = try checkType(alloc, io, "Q4_K", 144, ax, dequantQ4K);
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
    e = try checkType(alloc, io, "IQ3_S", 110, ax, dequantIQ3S);
    if (e > worst) worst = e;
    e = try checkType(alloc, io, "IQ3_XXS", 98, ax, dequantIQ3XXS);
    if (e > worst) worst = e;
    std.debug.print("  worst overall = {}\n", .{worst});
    try std.testing.expect(worst < 1e-3);
}

// ---------------------------------------------------------------------------
// f16 -> f32 exhaustive sweep.
//
// History: an f16 subnormal bug shipped and passed the whole suite, because no
// test ever exercised the exponent==0 path.  Those 1024 patterns are only
// 15% of the space, so we simply enumerate ALL 65536 codes here.
//
// The oracle is written from the IEEE-754 half specification in *pure integer
// bit arithmetic*, deliberately NOT using Zig's @floatCast(f16->f32): using the
// builtin would validate it against itself and could not catch a bug in the
// path being tested (which is itself @floatCast-based).  Every f16 value is
// exactly representable in f32, so the comparison is exact -- any deviation is
// a real bug, never rounding.
fn f16ToF32ByBits(h: u16) f32 {
    const sign: u32 = @as(u32, h >> 15) << 31;
    const exp: u32 = @as(u32, (h >> 10) & 0x1F);
    const man: u32 = h & 0x3FF;

    const bits: u32 = if (exp == 0) blk: {
        if (man == 0) break :blk sign; // +/- zero
        // Subnormal: value = man * 2^-24.  Normalise: shift the mantissa left
        // until bit 10 is set, decreasing the f32 exponent accordingly.
        var m = man;
        var e: i32 = -14; // f16 subnormal bias step
        while ((m & 0x400) == 0) : (m <<= 1) e -= 1;
        m &= 0x3FF;
        const f32exp: u32 = @intCast(e + 127);
        break :blk sign | (f32exp << 23) | (m << 13);
    } else if (exp == 0x1F) blk: {
        // Inf / NaN.  Preserve NaN-ness; the payload is not semantically
        // meaningful for our kernels, so we only require NaN->NaN.
        break :blk sign | 0x7F80_0000 | (if (man != 0) @as(u32, 0x0040_0000) else 0);
    } else blk: {
        const f32exp: u32 = exp + 127 - 15;
        break :blk sign | (f32exp << 23) | (man << 13);
    };
    return @bitCast(bits);
}

test "quant_dequant: f16 -> f32 exhaustive over all 65536 codes" {
    // Path A: the row accessor used by every rowDot kernel.
    var mismatches: usize = 0;
    var worst_rel: f32 = 0;
    var first_bad: u16 = 0;
    var have_bad = false;

    var h: u32 = 0;
    while (h < 0x10000) : (h += 1) {
        const code: u16 = @intCast(h);
        const row = [_]u8{ @truncate(code), @as(u8, @truncate(code >> 8)) };
        const got = f16_to_f32(&row, 0);
        const want = f16ToF32ByBits(code);

        const both_nan = std.math.isNan(got) and std.math.isNan(want);
        if (both_nan) continue;

        const same = @as(u64, @bitCast(@as(f64, got))) == @as(u64, @bitCast(@as(f64, want)));
        if (!same) {
            mismatches += 1;
            if (!have_bad) {
                have_bad = true;
                first_bad = code;
            }
            const denom = @max(@abs(want), 1e-30);
            const rel = @abs(got - want) / denom;
            if (rel > worst_rel) worst_rel = rel;
        }
    }
    if (mismatches != 0) {
        std.debug.print("  f16 sweep: {d} mismatches, first code 0x{X:0>4}, worst rel {e}\n", .{ mismatches, first_bad, worst_rel });
    }
    try std.testing.expectEqual(@as(usize, 0), mismatches);

    // Path B: simd_ref's pointer-based accessor must agree with path A.
    var h2: u32 = 0;
    var mismatches_b: usize = 0;
    while (h2 < 0x10000) : (h2 += 1) {
        const code: u16 = @intCast(h2);
        const bytes = [_]u8{ @truncate(code), @as(u8, @truncate(code >> 8)) };
        const got = simd_ref.util.f16At(&bytes, 0);
        const want = f16ToF32ByBits(code);
        if (std.math.isNan(got) and std.math.isNan(want)) continue;
        if (@as(u64, @bitCast(@as(f64, got))) != @as(u64, @bitCast(@as(f64, want)))) mismatches_b += 1;
    }
    try std.testing.expectEqual(@as(usize, 0), mismatches_b);
}

// ---------------------------------------------------------------------------
// f16 / bf16 rows (unquantized 2-byte weights, 256 elements per "row block").
// These are the scalar per-element references the f16_q8_0 / bf16_q8_0 C
// kernels are A/B'd against (QWEN35_KERNEL_CHECK=1), and what the old
// f32-materialization path effectively computed.  Same accumulation shape as
// the kernels: one f32 accumulator per 32-element chunk, added row-major.
// ---------------------------------------------------------------------------

fn f16BitsToF32(h: u16) f32 {
    return @floatCast(@as(f16, @bitCast(h)));
}

fn bf16BitsToF32(h: u16) f32 {
    return @as(f32, @bitCast(@as(u32, h) << 16));
}

/// Dot of one f16 row (512 B = 256 x 2 B) with an f32 activation.
pub fn rowDotF16(row: []const u8, x: []const f32) f32 {
    // Same layout rule as bf16 below: 32 x f16 per block, no block metadata.
    // Cover the WHOLE row (nblk from x.len), not a hardcoded 8 blocks.
    std.debug.assert(x.len % 32 == 0);
    std.debug.assert(row.len >= x.len * 2);
    var sum: f32 = 0;
    const nblk = x.len / 32;
    const w: [*]const u16 = @ptrCast(@alignCast(row.ptr));
    for (0..nblk) |blk| {
        var acc: f32 = 0;
        for (0..32) |k| acc += f16BitsToF32(w[blk * 32 + k]) * x[blk * 32 + k];
        sum += acc;
    }
    return sum;
}

/// Dot of one bf16 row (512 B = 256 x 2 B) with an f32 activation.
pub fn rowDotBF16(row: []const u8, x: []const f32) f32 {
    // One 32-element block = 32 x bf16 = 64 bytes; no per-block scale (bf16 IS
    // the value).  The row length is implied by `x`, so cover ALL of x -- the
    // old version stopped after 8 blocks (256 elements), silently truncating
    // every row wider than 256 and producing garbage for real models.
    std.debug.assert(x.len % 32 == 0);
    std.debug.assert(row.len >= x.len * 2);
    var sum: f32 = 0;
    const nblk = x.len / 32;
    const w: [*]const u16 = @ptrCast(@alignCast(row.ptr));
    for (0..nblk) |blk| {
        var acc: f32 = 0;
        for (0..32) |k| acc += bf16BitsToF32(w[blk * 32 + k]) * x[blk * 32 + k];
        sum += acc;
    }
    return sum;
}

/// Dequantize Q8_0 into f32 (used for small tensors / tests).
pub fn dequantizeQ8_0(data: []const u8, out: []f32) !void {
    if (out.len % QK8_0 != 0) return error.NotMultipleOfBlock;
    if (data.len != out.len / QK8_0 * block_q8_0_bytes) return error.SizeMismatch;
    var blk: usize = 0;
    var base: usize = 0;
    while (blk < out.len) : ({
        blk += QK8_0;
        base += block_q8_0_bytes;
    }) {
        const d: f16 = @bitCast(std.mem.readInt(u16, data[base..][0..2], .little));
        const df: f32 = @floatCast(d);
        const qs = data[base + 2 ..][0..QK8_0];
        for (qs, 0..) |q, j| out[blk + j] = df * @as(f32, @floatFromInt(@as(i8, @bitCast(q))));
    }
}

pub fn quantizeQ8_0(data: []const f32, out: []u8) !void {
    if (data.len % QK8_0 != 0) return error.NotMultipleOfBlock;
    if (out.len != data.len / QK8_0 * block_q8_0_bytes) return error.SizeMismatch;
    var idata: usize = 0;
    var iout: usize = 0;
    while (idata < data.len) : ({
        idata += QK8_0;
        iout += block_q8_0_bytes;
    }) {
        var max: f32 = 0;
        for (data[idata..][0..QK8_0]) |d| {
            const v = @abs(d);
            if (v > max) max = v;
        }
        const scale32: f32 = max / 127.0;
        const scale: f16 = @floatCast(scale32);
        std.mem.writeInt(u16, out[iout..][0..2], @bitCast(scale), .little);
        for (0..QK8_0) |i| {
            out[iout + 2 + i] = @bitCast(@as(i8, @intFromFloat(@round(data[idata + i] / scale32))));
        }
    }
}
