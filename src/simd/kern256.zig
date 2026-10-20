//! 256-element super-block dot products: K-quant / ternary weights x Q8_K.
//!
//! Ported from llama.cpp:
//!   * semantics:  `ggml/src/ggml-cpu/quants.c`  (`*_q8_K_generic`)
//!   * vectorized: `ggml/src/ggml-cpu/arch/x86/quants.c`
//! Pure Zig: `@Vector` + std, no C, no C++, no inline asm.  build.zig compiles
//! one copy of this directory per `-mcpu`, so the same 128-bit `@Vector` source
//! becomes ymm code on the AVX2 tier and zmm code on the AVX-512 tier.
//!
//! Block layouts (ggml-common.h), all 256 elements per super-block:
//!   Q2_K  84 B : scales[16]@0 qs[64]@16 d(f16)@80 dmin(f16)@82
//!   Q3_K 110 B : hmask[32]@0 qs[64]@32 scales[12]@96 d(f16)@108
//!   Q4_K 144 B : d(f16)@0 dmin(f16)@2 scales[12]@4 qs[128]@16
//!   Q5_K 176 B : d(f16)@0 dmin(f16)@2 scales[12]@4 qh[32]@16 qs[128]@48
//!   Q6_K 210 B : ql[128]@0 qh[64]@128 scales[16](i8)@192 d(f16)@208
//!   TQ1_0 54 B : qs[48]@0 qh[4]@48 d(f16)@52
//!   TQ2_0 66 B : qs[64]@0 d(f16)@64
//! Activation Q8_K (292 B): d(f32)@0, qs[256](i8)@4, bsums[16](i16)@260.
//!
//! Every kernel has a scalar `_ref` twin translated line-by-line from the C
//! `*_generic` function; the tests compare them.

const std = @import("std");
const util = @import("util.zig");
const tier = @import("tier.zig");

const V16u8 = @Vector(16, u8);
const V32u8 = @Vector(32, u8);
const V32i8 = @Vector(32, i8);
const V32i16 = @Vector(32, i16);
const V32i32 = @Vector(32, i32);
const V32u16 = @Vector(32, u16);
const V32u5 = @Vector(32, u5);
const V32u3 = @Vector(32, u3);

const q8k_bytes = 292;

/// unit scale table for the types with no per-group scale (TQ1_0, TQ2_0)
const ones: [16]i32 = blk: {
    var t: [16]i32 = undefined;
    @memset(&t, 1);
    break :blk t;
};
const qk_k = 256;

// ---------------------------------------------------------------------------
// shared helpers
// ---------------------------------------------------------------------------

inline fn q8kD(y: [*]const u8) f32 {
    return @bitCast(util.u32At(y));
}

inline fn q8kQs(y: [*]const u8) [*]const u8 {
    return y + 4;
}

inline fn q8kBsum(y: [*]const u8, j: usize) i16 {
    const p: *align(1) const i16 = @ptrCast(y + 260 + j * 2);
    return p.*;
}

/// 32 low / high nibbles of 32 bytes as i8 (values 0..15).
inline fn niblo32(q: V32u8) V32i8 {
    return @bitCast(q & @as(V32u8, @splat(0x0F)));
}
inline fn nibhi32(q: V32u8) V32i8 {
    return @bitCast(q >> @as(V32u8, @splat(4)));
}

/// Broadcast a pair of i32 to lanes 0..15 and 16..31.
inline fn splat2(a: i32, b: i32) V32i32 {
    var arr: [32]i32 = undefined;
    for (0..16) |i| arr[i] = a;
    for (0..16) |i| arr[16 + i] = b;
    return @bitCast(arr);
}

/// Vector -> array, for @memcpy targets (memcpy's slice arg gives @bitCast no
/// result type, so the conversion must be explicit).
inline fn arr32(v: V32i8) [32]i8 {
    return @bitCast(v);
}

inline fn arr32u8(v: V32u8) [32]i8 {
    return @bitCast(v);
}

inline fn loadI8(p: [*]const u8) V32i8 {
    return @bitCast(@as(*align(1) const [32]u8, @ptrCast(p)).*);
}

/// sum over 16-element groups of (scale[g] * sum(a*q8)), with one i32 vector
/// accumulator and a single @reduce at the end.  `scales[g]` is the integer
/// scale of group g (16 elements), precomputed by the caller.
inline fn dotScaled(a: *const [256]i8, q8: [*]const u8, scales: *const [16]i32) i32 {
    var acc: V32i32 = @splat(0);
    var i: usize = 0;
    while (i < 8) : (i += 1) {
        // 32 elements at a time = two 16-element groups, one i32 vector each
        const av: V32i8 = @as(*const [32]i8, @ptrCast(a[i * 32 ..])).*;
        const qv = loadI8(q8 + i * 32);
        const p: V32i16 = @as(V32i16, av) *% @as(V32i16, qv);
        acc +%= splat2(scales[2 * i], scales[2 * i + 1]) *% @as(V32i32, p);
    }
    return @reduce(.Add, acc);
}

// ---------------------------------------------------------------------------
// Q4_K x Q8_K   (quants.c:696 generic, arch/x86/quants.c:2102)
// ---------------------------------------------------------------------------
// aux8 = the 128 bytes unpacked as 4 x (32 low nibbles, 32 high nibbles).
// dot = d*sum_groups(scale_g * sum(a*q8)) - dmin*sum_j(bsums[j]*min[j/2])

inline fn unpackQ4K(qs: *const [128]u8, out: *[256]i8) void {
    for (0..4) |c| {
        const q: V32u8 = @as(*const [32]u8, @ptrCast(qs[c * 32 ..])).*;
        @memcpy(out[c * 64 ..][0..32], &arr32(niblo32(q)));
        @memcpy(out[c * 64 + 32 ..][0..32], &arr32(nibhi32(q)));
    }
}

/// Q4_K/Q5_K 6+4-bit scale/min unpacking (the utmp shuffle in the C code).
inline fn unpackScaleMin(scales12: *const [12]u8) [16]u8 {
    var utmp: [4]u32 = .{ 0, 0, 0, 0 };
    for (0..3) |i| {
        utmp[i] = @as(u32, scales12[i * 4]) |
            (@as(u32, scales12[i * 4 + 1]) << 8) |
            (@as(u32, scales12[i * 4 + 2]) << 16) |
            (@as(u32, scales12[i * 4 + 3]) << 24);
    }
    const kmask1: u32 = 0x3f3f3f3f;
    const kmask2: u32 = 0x0f0f0f0f;
    const kmask3: u32 = 0x03030303;
    utmp[3] = ((utmp[2] >> 4) & kmask2) | (((utmp[1] >> 6) & kmask3) << 4);
    const uaux = utmp[1] & kmask1;
    utmp[1] = (utmp[2] & kmask2) | (((utmp[0] >> 6) & kmask3) << 4);
    utmp[2] = uaux;
    utmp[0] &= kmask1;
    var out: [16]u8 = undefined;
    for (0..4) |i| {
        for (0..4) |b| out[i * 4 + b] = @truncate(utmp[i] >> @intCast(b * 8));
    }
    return out;
}

pub fn q4_k_q8_k(n: usize, vx: [*]const u8, vy: [*]const u8) callconv(.c) f32 {
    const nb = n / qk_k;
    var sumf: f32 = 0;
    var i: usize = 0;
    while (i < nb) : (i += 1) {
        const x = vx + i * 144;
        const y = vy + i * q8k_bytes;
        var aux8: [256]i8 = undefined;
        unpackQ4K(@ptrCast(x + 16), &aux8);
        const sm = unpackScaleMin(@ptrCast(x + 4));
        const dy = q8kD(y);
        const d = util.f16At(x, 0) * dy;
        const dmin = util.f16At(x, 2) * dy;

        var sumi: i32 = 0;
        for (0..16) |j| sumi += @as(i32, q8kBsum(y, j)) * @as(i32, sm[8 + j / 2]);

        // 8 sub-block scales, each covering 32 elements -> group g (16 el) uses sm[g/2]
        var sc: [16]i32 = undefined;
        for (0..16) |g| sc[g] = @as(i32, sm[g / 2]);
        const isum = dotScaled(&aux8, q8kQs(y), &sc);
        sumf += d * @as(f32, @floatFromInt(isum)) - dmin * @as(f32, @floatFromInt(sumi));
    }
    return sumf;
}

pub fn q4_k_q8_k_ref(n: usize, vx: [*]const u8, vy: [*]const u8) callconv(.c) f32 {
    const nb = n / qk_k;
    var sumf: f32 = 0;
    var i: usize = 0;
    while (i < nb) : (i += 1) {
        const x = vx + i * 144;
        const y = vy + i * q8k_bytes;
        const q4 = x + 16;
        const q8 = q8kQs(y);
        const sm = unpackScaleMin(@ptrCast(x + 4));
        var aux8: [256]u8 = undefined;
        for (0..4) |c| {
            for (0..32) |l| aux8[c * 64 + l] = q4[c * 32 + l] & 0x0F;
            for (0..32) |l| aux8[c * 64 + 32 + l] = q4[c * 32 + l] >> 4;
        }
        var sumi: i32 = 0;
        for (0..16) |j| sumi += @as(i32, q8kBsum(y, j)) * @as(i32, sm[8 + j / 2]);
        var isum: i32 = 0;
        for (0..16) |g| {
            var sub: i32 = 0;
            for (0..16) |l| sub += @as(i32, @as(i8, @bitCast(q8[g * 16 + l]))) * @as(i32, aux8[g * 16 + l]);
            isum += @as(i32, sm[g / 2]) * sub; // one scale per 32 elements
        }
        const d = util.f16At(x, 0) * q8kD(y);
        const dmin = util.f16At(x, 2) * q8kD(y);
        sumf += d * @as(f32, @floatFromInt(isum)) - dmin * @as(f32, @floatFromInt(sumi));
    }
    return sumf;
}

// ---------------------------------------------------------------------------
// Q5_K x Q8_K   (quants.c:771 generic, arch/x86/quants.c:2280)
// ---------------------------------------------------------------------------
// Same as Q4_K, plus the 5th bit from qh[32]: group 2c supplies bit 2c to the
// low-nibble half and bit 2c+1 to the high-nibble half.

pub fn q5_k_q8_k(n: usize, vx: [*]const u8, vy: [*]const u8) callconv(.c) f32 {
    const nb = n / qk_k;
    var sumf: f32 = 0;
    var i: usize = 0;
    while (i < nb) : (i += 1) {
        const x = vx + i * 176;
        const y = vy + i * q8k_bytes;
        var aux8: [256]i8 = undefined;
        for (0..4) |c| {
            const q: V32u8 = @as(*const [32]u8, @ptrCast(x + 48 + c * 32)).*;
            const qh: V32u8 = @as(*const [32]u8, @ptrCast(x + 16)).*;
            const b0: V32u8 = @splat(@as(u8, 1) << @as(u3, @intCast(2 * c)));
            const b1: V32u8 = @splat(@as(u8, 1) << @as(u3, @intCast(2 * c + 1)));
            const m0: @Vector(32, bool) = (qh & b0) != @as(V32u8, @splat(0));
            const m1: @Vector(32, bool) = (qh & b1) != @as(V32u8, @splat(0));
            const h0: V32i8 = @bitCast(@select(u8, m0, @as(V32u8, @splat(16)), @as(V32u8, @splat(0))));
            const h1: V32i8 = @bitCast(@select(u8, m1, @as(V32u8, @splat(16)), @as(V32u8, @splat(0))));
            @memcpy(aux8[c * 64 ..][0..32], &arr32(niblo32(q) +% h0));
            @memcpy(aux8[c * 64 + 32 ..][0..32], &arr32(nibhi32(q) +% h1));
        }
        const sm = unpackScaleMin(@ptrCast(x + 4));
        const dy = q8kD(y);
        const d = util.f16At(x, 0) * dy;
        const dmin = util.f16At(x, 2) * dy;

        var sumi: i32 = 0;
        for (0..16) |j| sumi += @as(i32, q8kBsum(y, j)) * @as(i32, sm[8 + j / 2]);

        // 8 sub-block scales, each covering 32 elements -> group g (16 el) uses sm[g/2]
        var sc: [16]i32 = undefined;
        for (0..16) |g| sc[g] = @as(i32, sm[g / 2]);
        const isum = dotScaled(&aux8, q8kQs(y), &sc);
        sumf += d * @as(f32, @floatFromInt(isum)) - dmin * @as(f32, @floatFromInt(sumi));
    }
    return sumf;
}

pub fn q5_k_q8_k_ref(n: usize, vx: [*]const u8, vy: [*]const u8) callconv(.c) f32 {
    const nb = n / qk_k;
    var sumf: f32 = 0;
    var i: usize = 0;
    while (i < nb) : (i += 1) {
        const x = vx + i * 176;
        const y = vy + i * q8k_bytes;
        const q4 = x + 48;
        const hm = x + 16;
        const q8 = q8kQs(y);
        const sm = unpackScaleMin(@ptrCast(x + 4));
        var aux8: [256]u8 = undefined;
        var m: u8 = 1;
        for (0..4) |c| {
            for (0..32) |l| aux8[c * 64 + l] = (q4[c * 32 + l] & 0x0F) + (if (hm[l] & m != 0) @as(u8, 16) else 0);
            for (0..32) |l| aux8[c * 64 + 32 + l] = (q4[c * 32 + l] >> 4) + (if (hm[l] & (m << 1) != 0) @as(u8, 16) else 0);
            m <<= 2;
        }
        var sumi: i32 = 0;
        for (0..16) |j| sumi += @as(i32, q8kBsum(y, j)) * @as(i32, sm[8 + j / 2]);
        var isum: i32 = 0;
        for (0..16) |g| {
            var sub: i32 = 0;
            for (0..16) |l| sub += @as(i32, @as(i8, @bitCast(q8[g * 16 + l]))) * @as(i32, aux8[g * 16 + l]);
            isum += @as(i32, sm[g / 2]) * sub; // one scale per 32 elements
        }
        const d = util.f16At(x, 0) * q8kD(y);
        const dmin = util.f16At(x, 2) * q8kD(y);
        sumf += d * @as(f32, @floatFromInt(isum)) - dmin * @as(f32, @floatFromInt(sumi));
    }
    return sumf;
}

// ---------------------------------------------------------------------------
// Q6_K x Q8_K   (quants.c:851 generic, arch/x86/quants.c:2490)
// ---------------------------------------------------------------------------
// aux8[128] per half: from 32 ql bytes and 32 qh bytes, 4 interleaved groups.

pub fn q6_k_q8_k(n: usize, vx: [*]const u8, vy: [*]const u8) callconv(.c) f32 {
    const nb = n / qk_k;
    var sumf: f32 = 0;
    var i: usize = 0;
    while (i < nb) : (i += 1) {
        const x = vx + i * 210;
        const y = vy + i * q8k_bytes;
        var aux8: [256]i8 = undefined;
        for (0..2) |h| {
            const q4: V32u8 = @as(*const [32]u8, @ptrCast(x + h * 64)).*;
            const q4b: V32u8 = @as(*const [32]u8, @ptrCast(x + h * 64 + 32)).*;
            const qh: V32u8 = @as(*const [32]u8, @ptrCast(x + 128 + h * 32)).*;
            const sh: [4]V32u5 = .{
                @splat(@as(u5, 0)), @splat(@as(u5, 2)), @splat(@as(u5, 4)), @splat(@as(u5, 6)),
            };
            const lo = niblo32(q4);
            const lo2 = niblo32(q4b);
            const hi = nibhi32(q4);
            const hi2 = nibhi32(q4b);
            const base = @as(V32i8, @splat(-32));
            inline for (0..4) |g| {
                const extra: V32i8 = @as(V32i8, @bitCast((qh >> sh[g]) & @as(V32u8, @splat(0x03)))) *% @as(V32i8, @splat(16));
                const src = if (g == 0) lo else if (g == 1) lo2 else if (g == 2) hi else hi2;
                @memcpy(aux8[h * 128 + g * 32 ..][0..32], &arr32(src +% extra +% base));
            }
        }
        const dy = q8kD(y);
        const d = util.f16At(x, 208) * dy;
        var sc: [16]i32 = undefined;
        for (0..16) |g| sc[g] = @as(i32, @as(*const [16]i8, @ptrCast(x + 192))[g]);
        const isum = dotScaled(&aux8, q8kQs(y), &sc);
        sumf += d * @as(f32, @floatFromInt(isum));
    }
    return sumf;
}

pub fn q6_k_q8_k_ref(n: usize, vx: [*]const u8, vy: [*]const u8) callconv(.c) f32 {
    const nb = n / qk_k;
    var sumf: f32 = 0;
    var i: usize = 0;
    while (i < nb) : (i += 1) {
        const x = vx + i * 210;
        const y = vy + i * q8k_bytes;
        const q4 = x;
        const qh = x + 128;
        const q8 = q8kQs(y);
        var aux8: [256]i8 = undefined;
        for (0..2) |h| {
            for (0..32) |l| {
                // q4 advances 64 bytes and qh 32 bytes per 128-element chunk (as in C)
                aux8[h * 128 + l + 0] = @truncate(@as(i32, (q4[h * 64 + l + 0] & 0x0F) | (((qh[h * 32 + l] >> 0) & 3) << 4)) - 32);
                aux8[h * 128 + l + 32] = @truncate(@as(i32, (q4[h * 64 + l + 32] & 0x0F) | (((qh[h * 32 + l] >> 2) & 3) << 4)) - 32);
                aux8[h * 128 + l + 64] = @truncate(@as(i32, (q4[h * 64 + l + 0] >> 4) | (((qh[h * 32 + l] >> 4) & 3) << 4)) - 32);
                aux8[h * 128 + l + 96] = @truncate(@as(i32, (q4[h * 64 + l + 32] >> 4) | (((qh[h * 32 + l] >> 6) & 3) << 4)) - 32);
            }
        }
        var isum: i32 = 0;
        for (0..16) |g| {
            const scale: i32 = @as(i32, @as(*const [16]i8, @ptrCast(x + 192))[g]);
            var sub: i32 = 0;
            for (0..16) |l| sub += @as(i32, @as(i8, @bitCast(q8[g * 16 + l]))) * @as(i32, aux8[g * 16 + l]);
            isum += scale * sub;
        }
        sumf += util.f16At(x, 208) * q8kD(y) * @as(f32, @floatFromInt(isum));
    }
    return sumf;
}

// ---------------------------------------------------------------------------
// Q3_K x Q8_K   (quants.c:617 generic, arch/x86/quants.c:1830)
// ---------------------------------------------------------------------------
// aux8 values are (2-bit nibble) - (hmask bit set ? 0 : 4), in [-4, 3].
// Scales are 6-bit signed, used as (scale - 32).

inline fn unpackQ3KScales(scales12: *const [12]u8) [16]i8 {
    var auxs: [4]u32 = .{ 0, 0, 0, 0 };
    for (0..3) |i| {
        auxs[i] = @as(u32, scales12[i * 4]) |
            (@as(u32, scales12[i * 4 + 1]) << 8) |
            (@as(u32, scales12[i * 4 + 2]) << 16) |
            (@as(u32, scales12[i * 4 + 3]) << 24);
    }
    const kmask1: u32 = 0x03030303;
    const kmask2: u32 = 0x0f0f0f0f;
    const tmp = auxs[2];
    auxs[2] = ((auxs[0] >> 4) & kmask2) | (((tmp >> 4) & kmask1) << 4);
    auxs[3] = ((auxs[1] >> 4) & kmask2) | (((tmp >> 6) & kmask1) << 4);
    auxs[0] = (auxs[0] & kmask2) | (((tmp >> 0) & kmask1) << 4);
    auxs[1] = (auxs[1] & kmask2) | (((tmp >> 2) & kmask1) << 4);
    var out: [16]i8 = undefined;
    for (0..4) |i| {
        for (0..4) |b| out[i * 4 + b] = @bitCast(@as(u8, @truncate(auxs[i] >> @intCast(b * 8))));
    }
    return out;
}

pub fn q3_k_q8_k(n: usize, vx: [*]const u8, vy: [*]const u8) callconv(.c) f32 {
    const nb = n / qk_k;
    var sumf: f32 = 0;
    var i: usize = 0;
    while (i < nb) : (i += 1) {
        const x = vx + i * 110;
        const y = vy + i * q8k_bytes;
        var aux8: [256]i8 = undefined;
        const qh: V32u8 = @as(*const [32]u8, @ptrCast(x)).*;
        for (0..2) |h| {
            const q3: V32u8 = @as(*const [32]u8, @ptrCast(x + 32 + h * 32)).*;
            inline for (0..4) |g| {
                const sh: V32u5 = @splat(@as(u5, @intCast(2 * g)));
                const nib: V32i8 = @bitCast((q3 >> sh) & @as(V32u8, @splat(0x03)));
                const bitv: V32u8 = @splat(@as(u8, 1) << @as(u3, @intCast(4 * h + g)));
                const m: @Vector(32, bool) = (qh & bitv) != @as(V32u8, @splat(0));
                const sub: V32i8 = @bitCast(@select(u8, m, @as(V32u8, @splat(0)), @as(V32u8, @splat(4))));
                @memcpy(aux8[h * 128 + g * 32 ..][0..32], &arr32(nib -% sub));
            }
        }
        const sc = unpackQ3KScales(@ptrCast(x + 96));
        const d = util.f16At(x, 108) * q8kD(y);
        var scv: [16]i32 = undefined;
        for (0..16) |g| scv[g] = @as(i32, sc[g]) - 32;
        const isum = dotScaled(&aux8, q8kQs(y), &scv);
        sumf += d * @as(f32, @floatFromInt(isum));
    }
    return sumf;
}

pub fn q3_k_q8_k_ref(n: usize, vx: [*]const u8, vy: [*]const u8) callconv(.c) f32 {
    const nb = n / qk_k;
    var sumf: f32 = 0;
    var i: usize = 0;
    while (i < nb) : (i += 1) {
        const x = vx + i * 110;
        const y = vy + i * q8k_bytes;
        const q3 = x + 32;
        const hm = x;
        const q8 = q8kQs(y);
        var aux8: [256]i8 = undefined;
        var m: u8 = 1;
        for (0..2) |h| {
            inline for (0..4) |g| {
                for (0..32) |l| {
                    const nib: i8 = @bitCast((q3[h * 32 + l] >> (2 * g)) & 0x03);
                    const sub: i8 = if (hm[l] & m != 0) 0 else 4;
                    aux8[h * 128 + g * 32 + l] = nib - sub;
                }
                m <<= 1;
            }
        }
        const sc = unpackQ3KScales(@ptrCast(x + 96));
        var isum: i32 = 0;
        for (0..16) |g| {
            const scale: i32 = @as(i32, sc[g]) - 32;
            var sub: i32 = 0;
            for (0..16) |l| sub += @as(i32, @as(i8, @bitCast(q8[g * 16 + l]))) * @as(i32, aux8[g * 16 + l]);
            isum += scale * sub;
        }
        sumf += util.f16At(x, 108) * q8kD(y) * @as(f32, @floatFromInt(isum));
    }
    return sumf;
}

// ---------------------------------------------------------------------------
// Q2_K x Q8_K   (quants.c:565 generic, arch/x86/quants.c:1638)
// ---------------------------------------------------------------------------
// aux8[k*128 + j*32 + l] = (qs[k*32+l] >> 2j) & 3, for k<2, j<4, l<32.
// Group g = k*8 + 2j (16 lanes) uses sc[g] & 0xF, group g+1 uses sc[g+1] & 0xF.
// Minus term: sum_j bsums[j] * (sc[j] >> 4).

pub fn q2_k_q8_k(n: usize, vx: [*]const u8, vy: [*]const u8) callconv(.c) f32 {
    const nb = n / qk_k;
    var sumf: f32 = 0;
    var i: usize = 0;
    while (i < nb) : (i += 1) {
        const x = vx + i * 84;
        const y = vy + i * q8k_bytes;
        const sc: *const [16]u8 = @ptrCast(x + 0);
        var aux8: [256]i8 = undefined;
        for (0..2) |k| {
            const q2: V32u8 = @as(*const [32]u8, @ptrCast(x + 16 + k * 32)).*;
            inline for (0..4) |j| {
                const sh: V32u5 = @splat(@as(u5, @intCast(2 * j)));
                @memcpy(aux8[k * 128 + j * 32 ..][0..32], &arr32u8((q2 >> sh) & @as(V32u8, @splat(0x03))));
            }
        }
        var summs: i32 = 0;
        for (0..16) |j| summs += @as(i32, q8kBsum(y, j)) * @as(i32, sc[j] >> 4);
        var scv: [16]i32 = undefined;
        for (0..16) |g| scv[g] = @as(i32, sc[g] & 0x0F);
        const isum = dotScaled(&aux8, q8kQs(y), &scv);
        const dall = q8kD(y) * util.f16At(x, 80);
        const dmin = q8kD(y) * util.f16At(x, 82);
        sumf += dall * @as(f32, @floatFromInt(isum)) - dmin * @as(f32, @floatFromInt(summs));
    }
    return sumf;
}

pub fn q2_k_q8_k_ref(n: usize, vx: [*]const u8, vy: [*]const u8) callconv(.c) f32 {
    const nb = n / qk_k;
    var sumf: f32 = 0;
    var i: usize = 0;
    while (i < nb) : (i += 1) {
        const x = vx + i * 84;
        const y = vy + i * q8k_bytes;
        const q2 = x + 16;
        const q8 = q8kQs(y);
        const sc: *const [16]u8 = @ptrCast(x + 0);
        var summs: i32 = 0;
        for (0..16) |j| summs += @as(i32, q8kBsum(y, j)) * @as(i32, sc[j] >> 4);
        var isum: i32 = 0;
        var is: usize = 0;
        for (0..2) |k| {
            for (0..4) |j| {
                const shift: u3 = @intCast(2 * j);
                var sub: i32 = 0;
                for (0..16) |l| sub += @as(i32, @as(i8, @bitCast(q8[k * 128 + j * 32 + l]))) * @as(i32, (q2[k * 32 + l] >> shift) & 0x03);
                isum += @as(i32, sc[is] & 0x0F) * sub;
                is += 1;
                sub = 0;
                for (0..16) |l| sub += @as(i32, @as(i8, @bitCast(q8[k * 128 + j * 32 + 16 + l]))) * @as(i32, (q2[k * 32 + 16 + l] >> shift) & 0x03);
                isum += @as(i32, sc[is] & 0x0F) * sub;
                is += 1;
            }
        }
        const dall = q8kD(y) * util.f16At(x, 80);
        const dmin = q8kD(y) * util.f16At(x, 82);
        sumf += dall * @as(f32, @floatFromInt(isum)) - dmin * @as(f32, @floatFromInt(summs));
    }
    return sumf;
}

// ---------------------------------------------------------------------------
// TQ2_0 x Q8_K   (quants.c:533 generic, arch/x86/quants.c:1572)
// ---------------------------------------------------------------------------
// 2 bits per element, ternary-ish: value = ((qs[k] >> 2l) & 3) - 1.

pub fn tq2_0_q8_k(n: usize, vx: [*]const u8, vy: [*]const u8) callconv(.c) f32 {
    const nb = n / qk_k;
    var sumf: f32 = 0;
    var i: usize = 0;
    while (i < nb) : (i += 1) {
        const x = vx + i * 66;
        const y = vy + i * q8k_bytes;
        var aux8: [256]i8 = undefined;
        for (0..2) |k| {
            const q: V32u8 = @as(*const [32]u8, @ptrCast(x + k * 32)).*;
            inline for (0..4) |l| {
                const sh: V32u8 = @splat(@as(u8, @intCast(2 * l)));
                const v: V32i8 = @bitCast((q >> sh) & @as(V32u8, @splat(0x03)));
                @memcpy(aux8[k * 128 + l * 32 ..][0..32], &arr32(v -% @as(V32i8, @splat(1))));
            }
        }
        const isum = dotScaled(&aux8, q8kQs(y), &ones);
        sumf += @as(f32, @floatFromInt(isum)) * (q8kD(y) * util.f16At(x, 64));
    }
    return sumf;
}

pub fn tq2_0_q8_k_ref(n: usize, vx: [*]const u8, vy: [*]const u8) callconv(.c) f32 {
    const nb = n / qk_k;
    var sumf: f32 = 0;
    var i: usize = 0;
    while (i < nb) : (i += 1) {
        const x = vx + i * 66;
        const y = vy + i * q8k_bytes;
        const q8 = q8kQs(y);
        var sumi: i32 = 0;
        for (0..2) |k| {
            for (0..4) |l| {
                for (0..32) |e| {
                    const v: i32 = @as(i32, (x[k * 32 + e] >> @intCast(2 * l)) & 0x03) - 1;
                    sumi += @as(i32, @as(i8, @bitCast(q8[k * 128 + l * 32 + e]))) * v;
                }
            }
        }
        sumf += @as(f32, @floatFromInt(sumi)) * (q8kD(y) * util.f16At(x, 64));
    }
    return sumf;
}

// ---------------------------------------------------------------------------
// TQ1_0 x Q8_K   (quants.c:481 generic, arch/x86/quants.c:1440)
// ---------------------------------------------------------------------------
// Base-5 packing: 48 bytes x 5 = 240 elements, then qh[4] x 4 = 16 elements.
// value = (((qs[j] * pow3[l]) & 0xFF) * 3 >> 8) - 1, i.e. the l-th ternary digit.

const pow3 = [6]u8{ 1, 3, 9, 27, 81, 243 };

inline fn tq1Vals(bytes: V32u8, comptime l: usize) V32i8 {
    const q: V32u8 = bytes *% @as(V32u8, @splat(pow3[l]));
    const xi: V32u16 = (@as(V32u16, q) *% @as(V32u16, @splat(@as(u16, 3)))) >> @as(V32u16, @splat(8));
    const xa: [32]u16 = @bitCast(xi);
    var out: [32]i8 = undefined;
    for (0..32) |k| out[k] = @bitCast(@as(u8, @truncate(xa[k])));
    const v: V32i8 = @bitCast(out);
    return v -% @as(V32i8, @splat(1));
}

pub fn tq1_0_q8_k(n: usize, vx: [*]const u8, vy: [*]const u8) callconv(.c) f32 {
    const nb = n / qk_k;
    var sumf: f32 = 0;
    var i: usize = 0;
    while (i < nb) : (i += 1) {
        const x = vx + i * 54;
        const y = vy + i * q8k_bytes;
        var aux8: [256]i8 = undefined;
        // 32 bytes -> 5 groups of 32 elements (activation index l*32 + m)
        const b0: V32u8 = @as(*const [32]u8, @ptrCast(x)).*;
        inline for (0..5) |l| {
            @memcpy(aux8[l * 32 ..][0..32], &arr32(tq1Vals(b0, l)));
        }
        // bytes 32..47 (16) -> 5 groups of 16 (activation index 160 + l*16 + m)
        for (0..5) |l| {
            for (0..16) |m| {
                const q: u8 = x[32 + m] *% pow3[l];
                aux8[160 + l * 16 + m] = @as(i8, @intCast((@as(u16, q) * 3) >> 8)) - 1;
            }
        }
        // qh[4] x 4 -> activation index 240 + l*4 + j
        for (0..4) |l| {
            for (0..4) |j| {
                const q: u8 = x[48 + j] *% pow3[l];
                aux8[240 + l * 4 + j] = @as(i8, @intCast((@as(u16, q) * 3) >> 8)) - 1;
            }
        }
        const isum = dotScaled(&aux8, q8kQs(y), &ones);
        sumf += @as(f32, @floatFromInt(isum)) * (util.f16At(x, 52) * q8kD(y));
    }
    return sumf;
}

pub fn tq1_0_q8_k_ref(n: usize, vx: [*]const u8, vy: [*]const u8) callconv(.c) f32 {
    const nb = n / qk_k;
    var sumf: f32 = 0;
    var i: usize = 0;
    while (i < nb) : (i += 1) {
        const x = vx + i * 54;
        const y = vy + i * q8k_bytes;
        const q8 = q8kQs(y);
        var sum: i32 = 0;
        for (0..32) |m| {
            for (0..5) |l| {
                const q: u8 = x[m] *% pow3[l];
                const xi: u16 = (@as(u16, q) * 3) >> 8;
                sum += @as(i32, @as(i8, @bitCast(q8[l * 32 + m]))) * (@as(i32, xi) - 1);
            }
        }
        for (0..16) |m| {
            for (0..5) |l| {
                const q: u8 = x[32 + m] *% pow3[l];
                const xi: u16 = (@as(u16, q) * 3) >> 8;
                sum += @as(i32, @as(i8, @bitCast(q8[160 + l * 16 + m]))) * (@as(i32, xi) - 1);
            }
        }
        for (0..4) |l| {
            for (0..4) |j| {
                const q: u8 = x[48 + j] *% pow3[l];
                const xi: u16 = (@as(u16, q) * 3) >> 8;
                sum += @as(i32, @as(i8, @bitCast(q8[240 + l * 4 + j]))) * (@as(i32, xi) - 1);
            }
        }
        sumf += @as(f32, @floatFromInt(sum)) * (util.f16At(x, 52) * q8kD(y));
    }
    return sumf;
}

/// Diagnostic: expose the SIMD aux8 builders so tests can diff them against the
/// scalar twins (the two paths must agree element-by-element, not just in sums).
pub fn dbgAuxQ5K(x: [*]const u8) [256]i8 {
    var aux8: [256]i8 = undefined;
    for (0..4) |c| {
        const q: V32u8 = @as(*const [32]u8, @ptrCast(x + 48 + c * 32)).*;
        const qh: V32u8 = @as(*const [32]u8, @ptrCast(x + 16)).*;
        const b0: V32u8 = @splat(@as(u8, 1) << @as(u3, @intCast(2 * c)));
        const b1: V32u8 = @splat(@as(u8, 1) << @as(u3, @intCast(2 * c + 1)));
        const m0: @Vector(32, bool) = (qh & b0) != @as(V32u8, @splat(0));
        const m1: @Vector(32, bool) = (qh & b1) != @as(V32u8, @splat(0));
        const h0: V32i8 = @bitCast(@select(u8, m0, @as(V32u8, @splat(16)), @as(V32u8, @splat(0))));
        const h1: V32i8 = @bitCast(@select(u8, m1, @as(V32u8, @splat(16)), @as(V32u8, @splat(0))));
        @memcpy(aux8[c * 64 ..][0..32], &arr32(niblo32(q) +% h0));
        @memcpy(aux8[c * 64 + 32 ..][0..32], &arr32(nibhi32(q) +% h1));
    }
    return aux8;
}

pub fn dbgAuxQ6K(x: [*]const u8) [256]i8 {
    var aux8: [256]i8 = undefined;
    for (0..2) |h| {
        const q4: V32u8 = @as(*const [32]u8, @ptrCast(x + h * 64)).*;
        const q4b: V32u8 = @as(*const [32]u8, @ptrCast(x + h * 64 + 32)).*;
        const qh: V32u8 = @as(*const [32]u8, @ptrCast(x + 128 + h * 32)).*;
        const sh: [4]V32u5 = .{
            @splat(@as(u5, 0)), @splat(@as(u5, 2)), @splat(@as(u5, 4)), @splat(@as(u5, 6)),
        };
        const lo = niblo32(q4);
        const lo2 = niblo32(q4b);
        const hi = nibhi32(q4);
        const hi2 = nibhi32(q4b);
        const base = @as(V32i8, @splat(-32));
        inline for (0..4) |g| {
            const extra: V32i8 = @as(V32i8, @bitCast((qh >> sh[g]) & @as(V32u8, @splat(0x03)))) *% @as(V32i8, @splat(16));
            const src = if (g == 0) lo else if (g == 1) lo2 else if (g == 2) hi else hi2;
            @memcpy(aux8[h * 128 + g * 32 ..][0..32], &arr32(src +% extra +% base));
        }
    }
    return aux8;
}
