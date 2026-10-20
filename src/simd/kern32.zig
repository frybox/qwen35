//! 32-element super-block dot products (packed weight x Q8_0 activation).
//!
//! Ported from llama.cpp:
//!   scalar semantics: ggml/src/ggml-cpu/quants.c        `ggml_vec_dot_*_generic`
//!   vectorization:    ggml/src/ggml-cpu/arch/x86/quants.c
//! Pure Zig `@Vector` -- no C, no C++, no inline asm.  build.zig compiles one
//! copy of this file per ISA tier, so LLVM widens the same IR to ymm (AVX2) and
//! zmm (AVX-512); the source is written at 128-bit widths and stays tier-free.
//!
//! Signature (single-row form of llama.cpp's
//! `ggml_vec_dot_X_q8_0(n, s, bs, vx, bx, vy, by, nrc)`):
//!
//!     pub fn x_q8_0(n: usize, vx: [*]const u8, vy: [*]const u8) callconv(.c) f32
//!
//! `vy` is a Q8_0 block array: 2-byte f16 scale + 32 x i8 = 34 bytes per 32
//! elements.  Each kernel also has a `_ref` scalar twin translated line by line
//! from the C `*_generic`, used as the parity oracle in tests.
//!
//! Block layouts (ggml-common.h):
//!   Q8_0   34 B / 32 el : d(f16) qs[32]
//!   Q4_0   18 B / 32 el : d(f16) qs[16]            v = (nib - 8) * d
//!   Q4_1   20 B / 32 el : d(f16) m(f16) qs[16]     v = nib * d + m
//!   Q5_0   22 B / 32 el : d(f16) qh(u32) qs[16]    v = ((nib | bit<<4) - 16) * d
//!   Q5_1   24 B / 32 el : d(f16) m(f16) qh(u32) qs[16]  v = (nib | bit<<4) * d + m
//!   Q1_0   10 B / 128 el: d(f16) bits[8]           v = +-1 * d
//!   Q2_0   18 B / 64 el : d(f16) qs[16]            v = (2bit - 1) * d
//!   IQ4_NL 18 B / 32 el : d(f16) qs[16]            v = kvalues[nib] * d

const std = @import("std");
const util = @import("util.zig");
const tier = @import("tier.zig");

const V16u8 = util.V16u8;
const V16i8 = util.V16i8;
const V32i8 = util.V32i8;
const V8f32 = util.V8f32;
const V16f32 = util.V16f32;

const q8_bytes = 34; // per 32 elements

// ---------------------------------------------------------------------------
// shared helpers
// ---------------------------------------------------------------------------

/// IQ4_NL codebook (quant_dequant.zig:23, ggml kvalues_iq4nl).
const kvalues_iq4nl: [16]i8 = .{ -127, -104, -83, -65, -49, -35, -22, -10, 1, 13, 25, 38, 53, 69, 89, 113 };

inline fn sumI8(v: V32i8) i32 {
    return @reduce(.Add, @as(@Vector(32, i32), v));
}

inline fn loadQs8(p: anytype) @Vector(8, u8) {
    return util.load(@Vector(8, u8), p);
}

inline fn loadQs16(p: anytype) V16u8 {
    return util.load(V16u8, p);
}

inline fn loadQy(p: anytype) V32i8 {
    return util.load(V32i8, p);
}

/// Q8_0 quant at `off` sign-extended to i32 (the raw byte is an i8).
inline fn qi8(p: [*]const u8, off: usize) i32 {
    return @as(i32, @as(i8, @bitCast(p[off])));
}

inline fn ylo(qy: V32i8) V16i8 {
    const a: [32]i8 = @bitCast(qy);
    return @bitCast(a[0..16].*);
}

inline fn yhi(qy: V32i8) V16i8 {
    const a: [32]i8 = @bitCast(qy);
    return @bitCast(a[16..32].*);
}

// comptime shuffles that spread bytes out to one lane per bit
const spread4: [32]i32 = blk: {
    var v: [32]i32 = undefined;
    for (0..32) |l| v[l] = @intCast(l / 8);
    break :blk v;
};
const shift4: [32]u3 = blk: {
    var v: [32]u3 = undefined;
    for (0..32) |l| v[l] = @intCast(l % 8);
    break :blk v;
};
const spread8x4: [32]i32 = blk: {
    var v: [32]i32 = undefined;
    for (0..32) |l| v[l] = @intCast(l / 4);
    break :blk v;
};
const shift8x4: [32]u3 = blk: {
    var v: [32]u3 = undefined;
    for (0..32) |l| v[l] = @intCast((l % 4) * 2);
    break :blk v;
};
const spread16lo: [16]i32 = blk: {
    var v: [16]i32 = undefined;
    for (0..16) |l| v[l] = @intCast(l / 8);
    break :blk v;
};
const spread16hi: [16]i32 = blk: {
    var v: [16]i32 = undefined;
    for (0..16) |l| v[l] = @intCast(2 + l / 8);
    break :blk v;
};
const shift16: [16]u3 = blk: {
    var v: [16]u3 = undefined;
    for (0..16) |l| v[l] = @intCast(l % 8);
    break :blk v;
};

/// 32 bits (4 bytes, little-endian bit order) -> one i8 per bit: bit set => +1,
/// clear => -1.  This is the Q1_0 sign pattern; vpsrlvb/vpshufb-shaped.
inline fn signs32(bits: *const [4]u8) V32i8 {
    const b: @Vector(4, u8) = @bitCast(bits.*);
    const spread: V32u8Local = @shuffle(u8, b, undefined, spread4);
    const sh: @Vector(32, u3) = @bitCast(shift4);
    const bitv: V32u8Local = (spread >> sh) & @as(V32u8Local, @splat(1));
    const s: V32i8 = @bitCast(bitv);
    return s *% @as(V32i8, @splat(2)) -% @as(V32i8, @splat(1));
}

const V32u8Local = @Vector(32, u8);

/// 8 bytes -> 32 lanes, each byte spread to 4 lanes and its 2-bit field
/// extracted, mapped {0,1,2,3} -> {-1,0,1,2} (Q2_0 quants).
inline fn q2vals(qs: *const [8]u8) V32i8 {
    const b: @Vector(8, u8) = @bitCast(qs.*);
    const spread: V32u8Local = @shuffle(u8, b, undefined, spread8x4);
    const sh: @Vector(32, u3) = @bitCast(shift8x4);
    const v: V32u8Local = (spread >> sh) & @as(V32u8Local, @splat(3));
    const sv: V32i8 = @bitCast(v);
    return sv -% @as(V32i8, @splat(1));
}

/// The 16 high bits of a Q5_0/Q5_1 block (qh bits 0..15) as 0/16 per lane.
inline fn qhLo(qh: u32) V16i8 {
    const b: @Vector(4, u8) = @bitCast(qh);
    const spread: @Vector(16, u8) = @shuffle(u8, b, undefined, spread16lo);
    const sh: @Vector(16, u3) = @bitCast(shift16);
    const bitv: @Vector(16, u8) = (spread >> sh) & @as(@Vector(16, u8), @splat(1));
    const s: V16i8 = @bitCast(bitv);
    return s *% @as(V16i8, @splat(16));
}

/// The 16 high bits of a Q5_0/Q5_1 block (qh bits 16..31) as 0/16 per lane.
inline fn qhHi(qh: u32) V16i8 {
    const b: @Vector(4, u8) = @bitCast(qh);
    const spread: @Vector(16, u8) = @shuffle(u8, b, undefined, spread16hi);
    const sh: @Vector(16, u3) = @bitCast(shift16);
    const bitv: @Vector(16, u8) = (spread >> sh) & @as(@Vector(16, u8), @splat(1));
    const s: V16i8 = @bitCast(bitv);
    return s *% @as(V16i8, @splat(16));
}

// ---------------------------------------------------------------------------
// Q8_0 x Q8_0
// ---------------------------------------------------------------------------
// C: ggml_vec_dot_q8_0_q8_0_generic (quants.c:451), AVX2 arch/x86/quants.c:1372
pub fn q8_0_q8_0(n: usize, vx: [*]const u8, vy: [*]const u8) callconv(.c) f32 {
    const nb = n / 32;
    var acc: f32 = 0;
    var ib: usize = 0;
    while (ib < nb) : (ib += 1) {
        const x = vx + ib * q8_bytes;
        const y = vy + ib * q8_bytes;
        const sumi = util.dot32i8(loadQy(x + 2), loadQy(y + 2));
        acc += util.f16At(x, 0) * @as(f32, @floatFromInt(sumi)) * util.f16At(y, 0);
    }
    return acc;
}

pub fn q8_0_q8_0_ref(n: usize, vx: [*]const u8, vy: [*]const u8) callconv(.c) f32 {
    return q8_0_q8_0(n, vx, vy);
}

// ---------------------------------------------------------------------------
// Q4_0 x Q8_0
// ---------------------------------------------------------------------------
// C: ggml_vec_dot_q4_0_q8_0_generic (quants.c:225), AVX2 arch/x86/quants.c:765
pub fn q4_0_q8_0(n: usize, vx: [*]const u8, vy: [*]const u8) callconv(.c) f32 {
    const nb = n / 32;
    var acc: f32 = 0;
    var ib: usize = 0;
    while (ib < nb) : (ib += 1) {
        const x = vx + ib * 18;
        const y = vy + ib * q8_bytes;
        const q = loadQs16(x + 2);
        const qy = loadQy(y + 2);
        const lo = util.nibloI8(q) -% @as(V16i8, @splat(8));
        const hi = util.nibhiI8(q) -% @as(V16i8, @splat(8));
        const sumi = util.dot16i8(lo, ylo(qy)) + util.dot16i8(hi, yhi(qy));
        acc += util.f16At(x, 0) * @as(f32, @floatFromInt(sumi)) * util.f16At(y, 0);
    }
    return acc;
}

pub fn q4_0_q8_0_ref(n: usize, vx: [*]const u8, vy: [*]const u8) callconv(.c) f32 {
    const nb = n / 32;
    var sumf: f32 = 0;
    var ib: usize = 0;
    while (ib < nb) : (ib += 1) {
        const x = vx + ib * 18;
        const y = vy + ib * q8_bytes;
        const dx = util.f16At(x, 0);
        const dy = util.f16At(y, 0);
        var sumi: i32 = 0;
        for (0..16) |j| {
            const b = x[2 + j];
            sumi += (@as(i32, b & 0x0F) - 8) * qi8(y, 2 + j);
            sumi += (@as(i32, b >> 4) - 8) * qi8(y, 2 + 16 + j);
        }
        sumf += dx * dy * @as(f32, @floatFromInt(sumi));
    }
    return sumf;
}

// ---------------------------------------------------------------------------
// Q4_1 x Q8_0
// ---------------------------------------------------------------------------
// The engine quantizes activations to Q8_0, not Q8_1, so this is the Q4_1
// weight paired with a Q8_0 activation.  Algebra (from
// ggml_vec_dot_q4_1_q8_1_generic, quants.c:262, with the Q8_1 min term dropped
// because Q8_0 has no min):
//   dot = d_x * sum(nib * qy) + m_x * sum(qy)
pub fn q4_1_q8_0(n: usize, vx: [*]const u8, vy: [*]const u8) callconv(.c) f32 {
    const nb = n / 32;
    var acc: f32 = 0;
    var ib: usize = 0;
    while (ib < nb) : (ib += 1) {
        const x = vx + ib * 20;
        const y = vy + ib * q8_bytes;
        const q = loadQs16(x + 4);
        const qy = loadQy(y + 2);
        const sumi = util.dot16i8(util.nibloI8(q), ylo(qy)) + util.dot16i8(util.nibhiI8(q), yhi(qy));
        // y_j = qy_j * d_y, so sum(m_x * y_j) = m_x * d_y * sum(qy_j)
        acc += util.f16At(x, 0) * @as(f32, @floatFromInt(sumi)) * util.f16At(y, 0) +
            util.f16At(x, 2) * util.f16At(y, 0) * @as(f32, @floatFromInt(sumI8(qy)));
    }
    return acc;
}

pub fn q4_1_q8_0_ref(n: usize, vx: [*]const u8, vy: [*]const u8) callconv(.c) f32 {
    const nb = n / 32;
    var sumf: f32 = 0;
    var ib: usize = 0;
    while (ib < nb) : (ib += 1) {
        const x = vx + ib * 20;
        const y = vy + ib * q8_bytes;
        const dx = util.f16At(x, 0);
        const mx = util.f16At(x, 2);
        const dy = util.f16At(y, 0);
        var sumi: i32 = 0;
        var sumy: i32 = 0;
        for (0..16) |j| {
            const b = x[4 + j];
            sumi += @as(i32, b & 0x0F) * qi8(y, 2 + j);
            sumi += @as(i32, b >> 4) * qi8(y, 2 + 16 + j);
        }
        for (0..32) |j| sumy += qi8(y, 2 + j);
        sumf += dx * dy * @as(f32, @floatFromInt(sumi)) + mx * dy * @as(f32, @floatFromInt(sumy));
    }
    return sumf;
}

// ---------------------------------------------------------------------------
// Q5_0 x Q8_0
// ---------------------------------------------------------------------------
// C: ggml_vec_dot_q5_0_q8_0_generic (quants.c:365), AVX2 arch/x86/quants.c:1206
pub fn q5_0_q8_0(n: usize, vx: [*]const u8, vy: [*]const u8) callconv(.c) f32 {
    const nb = n / 32;
    var acc: f32 = 0;
    var ib: usize = 0;
    while (ib < nb) : (ib += 1) {
        const x = vx + ib * 22;
        const y = vy + ib * q8_bytes;
        const q = loadQs16(x + 6);
        const qh = util.u32At(x + 2);
        const qy = loadQy(y + 2);
        const lo = util.nibloI8(q) + qhLo(qh) -% @as(V16i8, @splat(16));
        const hi = util.nibhiI8(q) + qhHi(qh) -% @as(V16i8, @splat(16));
        const sumi = util.dot16i8(lo, ylo(qy)) + util.dot16i8(hi, yhi(qy));
        acc += util.f16At(x, 0) * @as(f32, @floatFromInt(sumi)) * util.f16At(y, 0);
    }
    return acc;
}

pub fn q5_0_q8_0_ref(n: usize, vx: [*]const u8, vy: [*]const u8) callconv(.c) f32 {
    const nb = n / 32;
    var sumf: f32 = 0;
    var ib: usize = 0;
    while (ib < nb) : (ib += 1) {
        const x = vx + ib * 22;
        const y = vy + ib * q8_bytes;
        const qh = util.u32At(x + 2);
        const dx = util.f16At(x, 0);
        const dy = util.f16At(y, 0);
        var sumi: i32 = 0;
        for (0..16) |j| {
            const xh0: u8 = @intCast(((qh & (@as(u32, 1) << @intCast(j))) >> @intCast(j)) << 4);
            const xh1: u8 = @intCast(((qh & (@as(u32, 1) << @intCast(j + 16))) >> @intCast(j + 12)));
            const b = x[6 + j];
            const x0: i8 = @bitCast(@as(u8, b & 0x0F) | xh0);
            const x1: i8 = @bitCast(@as(u8, b >> 4) | xh1);
            sumi += (@as(i32, x0) - 16) * qi8(y, 2 + j);
            sumi += (@as(i32, x1) - 16) * qi8(y, 2 + 16 + j);
        }
        sumf += dx * dy * @as(f32, @floatFromInt(sumi));
    }
    return sumf;
}

// ---------------------------------------------------------------------------
// Q5_1 x Q8_0
// ---------------------------------------------------------------------------
// C: ggml_vec_dot_q5_1_q8_1_generic (quants.c:408), paired with Q8_0 as above:
//   dot = d_x * sum(q5 * qy) + m_x * sum(qy)
pub fn q5_1_q8_0(n: usize, vx: [*]const u8, vy: [*]const u8) callconv(.c) f32 {
    const nb = n / 32;
    var acc: f32 = 0;
    var ib: usize = 0;
    while (ib < nb) : (ib += 1) {
        const x = vx + ib * 24;
        const y = vy + ib * q8_bytes;
        const q = loadQs16(x + 8);
        const qh = util.u32At(x + 4);
        const qy = loadQy(y + 2);
        const lo = util.nibloI8(q) + qhLo(qh);
        const hi = util.nibhiI8(q) + qhHi(qh);
        const sumi = util.dot16i8(lo, ylo(qy)) + util.dot16i8(hi, yhi(qy));
        // y_j = qy_j * d_y, so sum(m_x * y_j) = m_x * d_y * sum(qy_j)
        acc += util.f16At(x, 0) * @as(f32, @floatFromInt(sumi)) * util.f16At(y, 0) +
            util.f16At(x, 2) * util.f16At(y, 0) * @as(f32, @floatFromInt(sumI8(qy)));
    }
    return acc;
}

pub fn q5_1_q8_0_ref(n: usize, vx: [*]const u8, vy: [*]const u8) callconv(.c) f32 {
    const nb = n / 32;
    var sumf: f32 = 0;
    var ib: usize = 0;
    while (ib < nb) : (ib += 1) {
        const x = vx + ib * 24;
        const y = vy + ib * q8_bytes;
        const qh = util.u32At(x + 4);
        const dx = util.f16At(x, 0);
        const mx = util.f16At(x, 2);
        const dy = util.f16At(y, 0);
        var sumi: i32 = 0;
        var sumy: i32 = 0;
        for (0..16) |j| {
            const xh0: u8 = @intCast(((qh & (@as(u32, 1) << @intCast(j))) >> @intCast(j)) << 4);
            const xh1: u8 = @intCast(((qh & (@as(u32, 1) << @intCast(j + 16))) >> @intCast(j + 12)));
            const b = x[8 + j];
            const x0: i8 = @bitCast(@as(u8, b & 0x0F) | xh0);
            const x1: i8 = @bitCast(@as(u8, b >> 4) | xh1);
            sumi += @as(i32, x0) * qi8(y, 2 + j);
            sumi += @as(i32, x1) * qi8(y, 2 + 16 + j);
        }
        for (0..32) |j| sumy += qi8(y, 2 + j);
        sumf += dx * dy * @as(f32, @floatFromInt(sumi)) + mx * dy * @as(f32, @floatFromInt(sumy));
    }
    return sumf;
}

// ---------------------------------------------------------------------------
// Q1_0 x Q8_0   (128 elements per weight block = 4 Q8_0 blocks)
// ---------------------------------------------------------------------------
// C: ggml_vec_dot_q1_0_q8_0_generic (quants.c:127), AVX2 arch/x86/quants.c:555
pub fn q1_0_q8_0(n: usize, vx: [*]const u8, vy: [*]const u8) callconv(.c) f32 {
    const nb = n / 128;
    var sumf: f32 = 0;
    var ib: usize = 0;
    while (ib < nb) : (ib += 1) {
        const x = vx + ib * 10;
        const d0 = util.f16At(x, 0);
        var sumi: f32 = 0;
        for (0..4) |k| {
            const y = vy + (ib * 4 + k) * q8_bytes;
            const bits: *const [4]u8 = @ptrCast(x + 2 + k * 4);
            const signs = signs32(bits);
            const dot = util.dot32i8(signs, loadQy(y + 2));
            sumi += util.f16At(y, 0) * @as(f32, @floatFromInt(dot));
        }
        sumf += d0 * sumi;
    }
    return sumf;
}

pub fn q1_0_q8_0_ref(n: usize, vx: [*]const u8, vy: [*]const u8) callconv(.c) f32 {
    const nb = n / 128;
    var sumf: f32 = 0;
    var ib: usize = 0;
    while (ib < nb) : (ib += 1) {
        const x = vx + ib * 10;
        const d0 = util.f16At(x, 0);
        var sumi: f32 = 0;
        for (0..4) |k| {
            const y = vy + (ib * 4 + k) * q8_bytes;
            const d1 = util.f16At(y, 0);
            var blk: i32 = 0;
            for (0..4) |b| {
                const mask = x[2 + k * 4 + b];
                for (0..8) |j| {
                    const v: i32 = qi8(y, 2 + 8 * b + j);
                    blk += if (mask & (@as(u8, 1) << @intCast(j)) != 0) v else -v;
                }
            }
            sumi += d1 * @as(f32, @floatFromInt(blk));
        }
        sumf += d0 * sumi;
    }
    return sumf;
}

// ---------------------------------------------------------------------------
// Q2_0 x Q8_0   (64 elements per weight block = 2 Q8_0 blocks)
// ---------------------------------------------------------------------------
// C: ggml_vec_dot_q2_0_q8_0_generic (quants.c:177), AVX2 arch/x86/quants.c:734
pub fn q2_0_q8_0(n: usize, vx: [*]const u8, vy: [*]const u8) callconv(.c) f32 {
    const nb = n / 64;
    var sumf: f32 = 0;
    var ib: usize = 0;
    while (ib < nb) : (ib += 1) {
        const x = vx + ib * 18;
        const d0 = util.f16At(x, 0);
        var sumi: f32 = 0;
        for (0..2) |k| {
            const y = vy + (ib * 2 + k) * q8_bytes;
            const qs: *const [8]u8 = @ptrCast(x + 2 + k * 8);
            const v = q2vals(qs);
            const dot = util.dot32i8(v, loadQy(y + 2));
            sumi += util.f16At(y, 0) * @as(f32, @floatFromInt(dot));
        }
        sumf += d0 * sumi;
    }
    return sumf;
}

pub fn q2_0_q8_0_ref(n: usize, vx: [*]const u8, vy: [*]const u8) callconv(.c) f32 {
    const nb = n / 64;
    var sumf: f32 = 0;
    var ib: usize = 0;
    while (ib < nb) : (ib += 1) {
        const x = vx + ib * 18;
        const d0 = util.f16At(x, 0);
        var sumi: f32 = 0;
        for (0..2) |k| {
            const y = vy + (ib * 2 + k) * q8_bytes;
            const d1 = util.f16At(y, 0);
            var blk: i32 = 0;
            for (0..8) |b| {
                const byte = x[2 + k * 8 + b];
                for (0..4) |j| {
                    const q = @as(i32, (byte >> @intCast(j * 2)) & 3) - 1;
                    blk += q * qi8(y, 2 + b * 4 + j);
                }
            }
            sumi += d1 * @as(f32, @floatFromInt(blk));
        }
        sumf += d0 * sumi;
    }
    return sumf;
}

// ---------------------------------------------------------------------------
// IQ4_NL x Q8_0
// ---------------------------------------------------------------------------
// C: ggml_vec_dot_iq4_nl_q8_0_generic (quants.c:1254), AVX2 arch/x86/quants.c:3984
// Codebook lookup is a 16-entry i8 table; the products stay in i8 so the dot is
// still vpmaddwd-shaped (the table read is scalar, the multiply-accumulate is
// vector).
pub fn iq4_nl_q8_0(n: usize, vx: [*]const u8, vy: [*]const u8) callconv(.c) f32 {
    const nb = n / 32;
    var acc: f32 = 0;
    var ib: usize = 0;
    while (ib < nb) : (ib += 1) {
        const x = vx + ib * 18;
        const y = vy + ib * q8_bytes;
        var qx: [32]i8 = undefined;
        for (0..16) |j| {
            const b = x[2 + j];
            qx[j] = kvalues_iq4nl[b & 0x0F];
            qx[16 + j] = kvalues_iq4nl[b >> 4];
        }
        const qv: V32i8 = @bitCast(qx);
        const sumi = util.dot32i8(qv, loadQy(y + 2));
        acc += util.f16At(x, 0) * @as(f32, @floatFromInt(sumi)) * util.f16At(y, 0);
    }
    return acc;
}

pub fn iq4_nl_q8_0_ref(n: usize, vx: [*]const u8, vy: [*]const u8) callconv(.c) f32 {
    const nb = n / 32;
    var sumf: f32 = 0;
    var ib: usize = 0;
    while (ib < nb) : (ib += 1) {
        const x = vx + ib * 18;
        const y = vy + ib * q8_bytes;
        const dx = util.f16At(x, 0);
        const dy = util.f16At(y, 0);
        var sumi: i32 = 0;
        for (0..16) |j| {
            const b = x[2 + j];
            sumi += @as(i32, kvalues_iq4nl[b & 0x0F]) * qi8(y, 2 + j);
            sumi += @as(i32, kvalues_iq4nl[b >> 4]) * qi8(y, 2 + 16 + j);
        }
        sumf += dx * dy * @as(f32, @floatFromInt(sumi));
    }
    return sumf;
}

// ---------------------------------------------------------------------------
// MXFP4 x Q8_0   (block: e(u8 E8M0 scale) + qs[16] packed nibbles, 32 el)
// ---------------------------------------------------------------------------
// C: ggml_vec_dot_mxfp4_q8_0_generic (quants.c:298), AVX2 arch/x86/quants.c:982
// Weight value = 0.5 * 2^(e-127) * kvalues_fp4[nib]; kvalues_fp4 stores the
// e2m1 values doubled (0,1,2,3,4,6,8,12,0,-1,...,-12), so the 0.5 cancels and
// the integer dot uses the table as-is.
const kvalues_fp4: [16]i8 = .{ 0, 1, 2, 3, 4, 6, 8, 12, 0, -1, -2, -3, -4, -6, -8, -12 };

inline fn e8m0ToHalf(e: u8) f32 {
    const bits: u32 = if (e == 0) 0x00200000 else @as(u32, e - 1) << 23;
    return @bitCast(bits);
}

fn mxfp4Vals(qs: *const [16]u8) V32i8 {
    var v: [32]i8 = undefined;
    for (0..16) |j| {
        v[j] = kvalues_fp4[qs[j] & 0x0F];
        v[16 + j] = kvalues_fp4[qs[j] >> 4];
    }
    return @bitCast(v);
}

pub fn mxfp4_q8_0(n: usize, vx: [*]const u8, vy: [*]const u8) callconv(.c) f32 {
    const nb = n / 32;
    var acc: f32 = 0;
    var ib: usize = 0;
    while (ib < nb) : (ib += 1) {
        const x = vx + ib * 17;
        const y = vy + ib * q8_bytes;
        const d = util.f16At(y, 0) * e8m0ToHalf(x[0]);
        const sumi = util.dot32i8(mxfp4Vals(@ptrCast(x + 1)), loadQy(y + 2));
        acc += d * @as(f32, @floatFromInt(sumi));
    }
    return acc;
}

pub fn mxfp4_q8_0_ref(n: usize, vx: [*]const u8, vy: [*]const u8) callconv(.c) f32 {
    const nb = n / 32;
    var sumf: f32 = 0;
    var ib: usize = 0;
    while (ib < nb) : (ib += 1) {
        const x = vx + ib * 17;
        const y = vy + ib * q8_bytes;
        const d = util.f16At(y, 0) * e8m0ToHalf(x[0]);
        var sumi: i32 = 0;
        for (0..16) |j| {
            sumi += kvalues_fp4[x[1 + j] & 0x0F] * qi8(y, 2 + j);
            sumi += kvalues_fp4[x[1 + j] >> 4] * qi8(y, 2 + 16 + j);
        }
        sumf += d * @as(f32, @floatFromInt(sumi));
    }
    return sumf;
}

// ---------------------------------------------------------------------------
// NVFP4 x Q8_0   (block: d[4] UE4M3 + qs[32], 64 el = 2 Q8_0 blocks)
// ---------------------------------------------------------------------------
// C: ggml_vec_dot_nvfp4_q8_0_generic (quants.c:330), AVX2 arch/x86/quants.c:1068
inline fn ue4m3ToHalf(x: u8) f32 {
    if (x == 0 or x == 0x7F) return 0.0;
    const exp: u32 = (x >> 3) & 0xF;
    const man: u32 = x & 0x7;
    const e: i32 = @bitCast(exp);
    const raw: f32 = if (exp == 0)
        @as(f32, @floatFromInt(man)) * std.math.pow(f32, 2.0, -9.0)
    else
        (1.0 + @as(f32, @floatFromInt(man)) / 8.0) * std.math.pow(f32, 2.0, @floatFromInt(e - 7));
    return raw * 0.5;
}

pub fn nvfp4_q8_0(n: usize, vx: [*]const u8, vy: [*]const u8) callconv(.c) f32 {
    const nb = n / 64;
    var sumf: f32 = 0;
    var ib: usize = 0;
    while (ib < nb) : (ib += 1) {
        const x = vx + ib * 36;
        for (0..4) |s| {
            const y = vy + (ib * 2 + s / 2) * q8_bytes;
            const off = (s % 2) * 16;
            const d = ue4m3ToHalf(x[s]) * util.f16At(y, 0);
            const qs: *const [8]u8 = @ptrCast(x + 4 + s * 8);
            var v16: [32]i8 = undefined;
            for (0..8) |j| {
                v16[j] = kvalues_fp4[qs[j] & 0x0F];
                v16[8 + j] = kvalues_fp4[qs[j] >> 4];
            }
            const qy: @Vector(16, i8) = @bitCast(@as(*align(1) const [16]u8, @ptrCast(y + 2 + off)).*);
            const vv: @Vector(16, i8) = @bitCast(v16[0..16].*);
            const sumi = util.dot16i8(vv, qy);
            sumf += d * @as(f32, @floatFromInt(sumi));
        }
    }
    return sumf;
}

pub fn nvfp4_q8_0_ref(n: usize, vx: [*]const u8, vy: [*]const u8) callconv(.c) f32 {
    const nb = n / 64;
    var sumf: f32 = 0;
    var ib: usize = 0;
    while (ib < nb) : (ib += 1) {
        const x = vx + ib * 36;
        for (0..4) |s| {
            const y = vy + (ib * 2 + s / 2) * q8_bytes;
            const off = (s % 2) * 16;
            const d = ue4m3ToHalf(x[s]) * util.f16At(y, 0);
            var sumi: i32 = 0;
            for (0..8) |j| {
                const qv = x[4 + s * 8 + j];
                sumi += kvalues_fp4[qv & 0x0F] * qi8(y, 2 + off + j);
                sumi += kvalues_fp4[qv >> 4] * qi8(y, 2 + off + 8 + j);
            }
            sumf += d * @as(f32, @floatFromInt(sumi));
        }
    }
    return sumf;
}

// ===========================================================================
// Row-level kernels: one call per weight row, ONE horizontal reduction per row
// ===========================================================================
// The block-level kernels above reduce every 32 elements.  For a decode matvec
// that costs a full cross-lane reduction per block, which measured SLOWER than
// the auto-vectorized scalar loop it replaced.  These variants keep a 16-lane
// f32 accumulator across all blocks of a row and reduce once at the end.
// Same signature as the block kernels (n = elements in the row), so they are
// drop-in for the dispatch table.

/// 16 i32 partial sums from the 32 elements at x/y, scaled by s and folded into
/// the row accumulator.
inline fn accBlock(acc: *V16f32, p: @Vector(16, i32), s: f32) void {
    acc.* += @as(V16f32, @floatFromInt(p)) * @as(V16f32, @splat(s));
}

/// Join two 8-lane partial-sum vectors into one 16-lane accumulator update.
inline fn accPair(acc: *V16f32, plo: @Vector(8, i32), phi: @Vector(8, i32), s: f32) void {
    var p: [16]i32 = undefined;
    const a: [8]i32 = @bitCast(plo);
    const b: [8]i32 = @bitCast(phi);
    for (0..8) |i| {
        p[i] = a[i];
        p[8 + i] = b[i];
    }
    acc.* += @as(V16f32, @floatFromInt(@as(@Vector(16, i32), @bitCast(p)))) * @as(V16f32, @splat(s));
}

pub fn q8_0_q8_0_row(n: usize, vx: [*]const u8, vy: [*]const u8) callconv(.c) f32 {
    var acc: V16f32 = @splat(0);
    for (0..n / 32) |ib| {
        const x = vx + ib * q8_bytes;
        const y = vy + ib * q8_bytes;
        accBlock(&acc, util.madd16i8(loadQy(x + 2), loadQy(y + 2)), util.f16At(x, 0) * util.f16At(y, 0));
    }
    return util.sumF16(acc);
}
pub fn q8_0_q8_0_row_ref(n: usize, vx: [*]const u8, vy: [*]const u8) callconv(.c) f32 {
    return q8_0_q8_0(n, vx, vy);
}

pub fn q4_0_q8_0_row(n: usize, vx: [*]const u8, vy: [*]const u8) callconv(.c) f32 {
    var acc: V16f32 = @splat(0);
    for (0..n / 32) |ib| {
        const x = vx + ib * 18;
        const y = vy + ib * q8_bytes;
        const q = loadQs16(x + 2);
        const qy = loadQy(y + 2);
        const lo = util.nibloI8(q) -% @as(V16i8, @splat(8));
        const hi = util.nibhiI8(q) -% @as(V16i8, @splat(8));
        accPair(&acc, util.madd8i8(lo, ylo(qy)), util.madd8i8(hi, yhi(qy)),
            util.f16At(x, 0) * util.f16At(y, 0));
    }
    return util.sumF16(acc);
}
pub fn q4_0_q8_0_row_ref(n: usize, vx: [*]const u8, vy: [*]const u8) callconv(.c) f32 {
    return q4_0_q8_0(n, vx, vy);
}

pub fn q4_1_q8_0_row(n: usize, vx: [*]const u8, vy: [*]const u8) callconv(.c) f32 {
    var acc: V16f32 = @splat(0);
    var accmin: V16f32 = @splat(0);
    for (0..n / 32) |ib| {
        const x = vx + ib * 20;
        const y = vy + ib * q8_bytes;
        const q = loadQs16(x + 4);
        const qy = loadQy(y + 2);
        const dy = util.f16At(y, 0);
        accPair(&acc, util.madd8i8(util.nibloI8(q), ylo(qy)), util.madd8i8(util.nibhiI8(q), yhi(qy)),
            util.f16At(x, 0) * dy);
        // min term: m_x * sum(y_j) = m_x * d_y * sum(qy_j); keep the sum as a
        // 16-lane vector so no per-block reduction is needed.
        const w = util.widenI8x32(qy);
        accmin += @as(V16f32, @floatFromInt(w[0] +% w[1])) * @as(V16f32, @splat(util.f16At(x, 2) * dy));
    }
    return util.sumF16(acc) + util.sumF16(accmin);
}
pub fn q4_1_q8_0_row_ref(n: usize, vx: [*]const u8, vy: [*]const u8) callconv(.c) f32 {
    return q4_1_q8_0(n, vx, vy);
}

pub fn q5_0_q8_0_row(n: usize, vx: [*]const u8, vy: [*]const u8) callconv(.c) f32 {
    var acc: V16f32 = @splat(0);
    for (0..n / 32) |ib| {
        const x = vx + ib * 22;
        const y = vy + ib * q8_bytes;
        const q = loadQs16(x + 6);
        const qh = util.u32At(x + 2);
        const qy = loadQy(y + 2);
        const lo = util.nibloI8(q) + qhLo(qh) -% @as(V16i8, @splat(16));
        const hi = util.nibhiI8(q) + qhHi(qh) -% @as(V16i8, @splat(16));
        accPair(&acc, util.madd8i8(lo, ylo(qy)), util.madd8i8(hi, yhi(qy)),
            util.f16At(x, 0) * util.f16At(y, 0));
    }
    return util.sumF16(acc);
}
pub fn q5_0_q8_0_row_ref(n: usize, vx: [*]const u8, vy: [*]const u8) callconv(.c) f32 {
    return q5_0_q8_0(n, vx, vy);
}

pub fn q5_1_q8_0_row(n: usize, vx: [*]const u8, vy: [*]const u8) callconv(.c) f32 {
    var acc: V16f32 = @splat(0);
    var accmin: V16f32 = @splat(0);
    for (0..n / 32) |ib| {
        const x = vx + ib * 24;
        const y = vy + ib * q8_bytes;
        const q = loadQs16(x + 8);
        const qh = util.u32At(x + 4);
        const qy = loadQy(y + 2);
        const dy = util.f16At(y, 0);
        const lo = util.nibloI8(q) + qhLo(qh);
        const hi = util.nibhiI8(q) + qhHi(qh);
        accPair(&acc, util.madd8i8(lo, ylo(qy)), util.madd8i8(hi, yhi(qy)), util.f16At(x, 0) * dy);
        const w = util.widenI8x32(qy);
        accmin += @as(V16f32, @floatFromInt(w[0] +% w[1])) * @as(V16f32, @splat(util.f16At(x, 2) * dy));
    }
    return util.sumF16(acc) + util.sumF16(accmin);
}
pub fn q5_1_q8_0_row_ref(n: usize, vx: [*]const u8, vy: [*]const u8) callconv(.c) f32 {
    return q5_1_q8_0(n, vx, vy);
}

pub fn iq4_nl_q8_0_row(n: usize, vx: [*]const u8, vy: [*]const u8) callconv(.c) f32 {
    var acc: V16f32 = @splat(0);
    for (0..n / 32) |ib| {
        const x = vx + ib * 18;
        const y = vy + ib * q8_bytes;
        var qx: [32]i8 = undefined;
        for (0..16) |j| {
            const b = x[2 + j];
            qx[j] = kvalues_iq4nl[b & 0x0F];
            qx[16 + j] = kvalues_iq4nl[b >> 4];
        }
        accBlock(&acc, util.madd16i8(@bitCast(qx), loadQy(y + 2)), util.f16At(x, 0) * util.f16At(y, 0));
    }
    return util.sumF16(acc);
}
pub fn iq4_nl_q8_0_row_ref(n: usize, vx: [*]const u8, vy: [*]const u8) callconv(.c) f32 {
    return iq4_nl_q8_0(n, vx, vy);
}

pub fn mxfp4_q8_0_row(n: usize, vx: [*]const u8, vy: [*]const u8) callconv(.c) f32 {
    var acc: V16f32 = @splat(0);
    for (0..n / 32) |ib| {
        const x = vx + ib * 17;
        const y = vy + ib * q8_bytes;
        const qs: *const [16]u8 = @ptrCast(x + 1);
        var v: [32]i8 = undefined;
        for (0..16) |j| {
            v[j] = kvalues_fp4[qs[j] & 0x0F];
            v[16 + j] = kvalues_fp4[qs[j] >> 4];
        }
        accBlock(&acc, util.madd16i8(@bitCast(v), loadQy(y + 2)),
            e8m0ToHalf(x[0]) * util.f16At(y, 0));
    }
    return util.sumF16(acc);
}
pub fn mxfp4_q8_0_row_ref(n: usize, vx: [*]const u8, vy: [*]const u8) callconv(.c) f32 {
    return mxfp4_q8_0(n, vx, vy);
}

/// Q1_0: one weight block covers 128 elements = 4 Q8_0 blocks sharing d0.
pub fn q1_0_q8_0_row(n: usize, vx: [*]const u8, vy: [*]const u8) callconv(.c) f32 {
    var acc: V16f32 = @splat(0);
    for (0..n / 128) |ib| {
        const x = vx + ib * 10;
        var inner: V16f32 = @splat(0);
        for (0..4) |k| {
            const y = vy + (ib * 4 + k) * q8_bytes;
            const signs = signs32(@ptrCast(x + 2 + k * 4));
            accBlock(&inner, util.madd16i8(signs, loadQy(y + 2)), util.f16At(y, 0));
        }
        acc += inner * @as(V16f32, @splat(util.f16At(x, 0)));
    }
    return util.sumF16(acc);
}
pub fn q1_0_q8_0_row_ref(n: usize, vx: [*]const u8, vy: [*]const u8) callconv(.c) f32 {
    return q1_0_q8_0(n, vx, vy);
}

/// Q2_0: one weight block covers 64 elements = 2 Q8_0 blocks sharing d0.
pub fn q2_0_q8_0_row(n: usize, vx: [*]const u8, vy: [*]const u8) callconv(.c) f32 {
    var acc: V16f32 = @splat(0);
    for (0..n / 64) |ib| {
        const x = vx + ib * 18;
        var inner: V16f32 = @splat(0);
        for (0..2) |k| {
            const y = vy + (ib * 2 + k) * q8_bytes;
            accBlock(&inner, util.madd16i8(q2vals(@ptrCast(x + 2 + k * 8)), loadQy(y + 2)), util.f16At(y, 0));
        }
        acc += inner * @as(V16f32, @splat(util.f16At(x, 0)));
    }
    return util.sumF16(acc);
}
pub fn q2_0_q8_0_row_ref(n: usize, vx: [*]const u8, vy: [*]const u8) callconv(.c) f32 {
    return q2_0_q8_0(n, vx, vy);
}

/// NVFP4: 64 elements per weight block, 4 sub-blocks of 16 with their own
/// UE4M3 scale, each pair of sub-blocks sharing one Q8_0 activation block.
pub fn nvfp4_q8_0_row(n: usize, vx: [*]const u8, vy: [*]const u8) callconv(.c) f32 {
    var acc: V16f32 = @splat(0);
    for (0..n / 64) |ib| {
        const x = vx + ib * 36;
        for (0..4) |s| {
            const y = vy + (ib * 2 + s / 2) * q8_bytes;
            const off = (s % 2) * 16;
            const qs: *const [8]u8 = @ptrCast(x + 4 + s * 8);
            var v16: [16]i8 = undefined;
            for (0..8) |j| {
                v16[j] = kvalues_fp4[qs[j] & 0x0F];
                v16[8 + j] = kvalues_fp4[qs[j] >> 4];
            }
            const qy: @Vector(16, i8) = @bitCast(@as(*align(1) const [16]u8, @ptrCast(y + 2 + off)).*);
            const p = util.madd8i8(@bitCast(v16), qy);
            const a: [8]f32 = @bitCast(@as(@Vector(8, f32), @floatFromInt(p)) *
                @as(@Vector(8, f32), @splat(ue4m3ToHalf(x[s]) * util.f16At(y, 0))));
            var acc16: [16]f32 = @bitCast(acc);
            for (0..8) |i| acc16[i] += a[i];
            acc = @bitCast(acc16);
        }
    }
    return util.sumF16(acc);
}
pub fn nvfp4_q8_0_row_ref(n: usize, vx: [*]const u8, vy: [*]const u8) callconv(.c) f32 {
    return nvfp4_q8_0(n, vx, vy);
}
