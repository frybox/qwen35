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
//!   Q1_0   18 B / 128 el: d(f16) qs[16] (1 bit each) v = +-1 * d  (ggml-common.h block_q1_0: d(2)+qs[16]=18)
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

// ---------------------------------------------------------------------------
// IQ4_NL x Q8_0 -- AVX-512 path (zmm, VNNI)
// ---------------------------------------------------------------------------
// The scalar version needs 32 codebook lookups per 32-element block, which is
// why iq4_nl (17.6 ns/element) was 3x the cost of q4_0 (5.5) on the same
// payload.  Here the codebook is a 16-byte vpshufb table replicated to all four
// 128-bit lanes, the qs bytes are broadcast to a full zmm so each lane
// independently shuffles the lo and hi nibbles, and vpdpbusd accumulates the
// codebook bytes against the Q8_0 qs bytes.
//
// vpdpbusd treats its first operand as UNSIGNED, so codebook entries are
// stored as (kvalue + 128) in u8 and the +128 bias is removed once per
// sub-block from the i32 accumulator (see kerniq.zig's iq4_xs/iq3_xxs paths,
// which use the same correction).
//
// Layout: the zmm lanes 0..15 carry the lo nibbles (block elements 0..15),
// lanes 16..31 the hi nibbles (elements 16..31).  Lanes 32..63 are a copy of
// lanes 0..31 (vbroadcasti32x4 replicates the 16-byte qs group four times), so
// vpdpbusd sums four identical 16-element partial dot products.  The Q8_0 side
// is built from two 16-byte broadcasts (qs[0..15] into lanes 0..15 + 32..47,
// qs[16..31] into lanes 16..31 + 48..63), giving 2x the true block sum in the
// lanes that carry nibbles; the +128 bias and the doubled partial sums are
// folded out in f32 by the factor 0.5.  No 64-byte read past the Q8_0 block
// (34 bytes) is needed.
const IQZ = @Vector(64, u8);
const IQI32 = @Vector(16, i32);

inline fn iqnBcast128(p: [*]const u8) IQZ {
    return asm ("vbroadcasti32x4 (%[src]), %[dst]" : [dst] "=v" (-> IQZ) : [src] "r" (p));
}
inline fn iqnBcast256(p: [*]const u8) IQZ {
    return asm ("vbroadcasti32x8 (%[src]), %[dst]" : [dst] "=v" (-> IQZ) : [src] "r" (p));
}
inline fn iqnPsrld4(a: IQZ) IQZ {
    return asm ("vpsrld $4, %[a], %[dst]" : [dst] "=v" (-> IQZ) : [a] "v" (a));
}
inline fn iqnPand(a: IQZ, b: IQZ) IQZ {
    return asm ("vpandd %[b], %[a], %[dst]" : [dst] "=v" (-> IQZ) : [a] "v" (a), [b] "v" (b));
}
inline fn iqnPor(a: IQZ, b: IQZ) IQZ {
    return asm ("vpord %[b], %[a], %[dst]" : [dst] "=v" (-> IQZ) : [a] "v" (a), [b] "v" (b));
}
inline fn iqnPaddb(a: IQZ, b: IQZ) IQZ {
    return asm ("vpaddb %[b], %[a], %[dst]" : [dst] "=v" (-> IQZ) : [a] "v" (a), [b] "v" (b));
}
inline fn iqnShufb(idx: IQZ, tab: IQZ) IQZ {
    return asm ("vpshufb %[idx], %[tab], %[dst]" : [dst] "=v" (-> IQZ) : [idx] "v" (idx), [tab] "v" (tab));
}
inline fn iqnDpbusd0(a: IQZ, b: IQZ) IQI32 {
    var out: IQI32 = @splat(0);
    asm ("vpdpbusd %[b], %[a], %[out]" : [out] "+v" (out) : [a] "v" (a), [b] "v" (b));
    return out;
}
inline fn iqnPslld7(a: IQI32) IQI32 {
    return asm ("vpslld $7, %[a], %[dst]" : [dst] "=v" (-> IQI32) : [a] "v" (a));
}
inline fn iqnPsubd(a: IQI32, b: IQI32) IQI32 {
    return asm ("vpsubd %[b], %[a], %[dst]" : [dst] "=v" (-> IQI32) : [a] "v" (a), [b] "v" (b));
}
inline fn iqnPandI(a: IQI32, b: IQI32) IQI32 {
    return asm ("vpandd %[b], %[a], %[dst]" : [dst] "=v" (-> IQI32) : [a] "v" (a), [b] "v" (b));
}

/// Codebook stored as (kvalue + 128) in u8 so vpshufb yields the biased byte
/// directly; vpdpbusd reads its first operand as unsigned, and the +128 bias is
/// removed from the i32 accumulator once per block (see the loop).
const iqn_tab: IQZ = blk: {
    var a: [64]u8 = undefined;
    for (0..64) |i| a[i] = @as(u8, @intCast(@as(i16, kvalues_iq4nl[i % 16]) + 128));
    break :blk @bitCast(a);
};
const iqn_m0f: IQZ = @splat(@as(u8, 0x0f));
const iqn_ones: IQZ = @splat(@as(u8, 1));
/// x side: keep shuffled lo nibbles in lanes 0..15, hi nibbles in lanes 16..31.
const iqn_mlo: IQZ = blk: {
    var a: [64]u8 = undefined;
    for (0..64) |i| a[i] = if (i < 16) @as(u8, 0xff) else 0;
    break :blk @bitCast(a);
};
const iqn_mhi: IQZ = blk: {
    var a: [64]u8 = undefined;
    for (0..64) |i| a[i] = if (i >= 16 and i < 32) @as(u8, 0xff) else 0;
    break :blk @bitCast(a);
};
/// +128 bias only applies to the lanes carrying codebook bytes.
const iqn_add128: IQZ = blk: {
    var a: [64]u8 = undefined;
    for (0..64) |i| a[i] = if (i < 32) @as(u8, 128) else 0;
    break :blk @bitCast(a);
};
const iqn_lanes07: IQI32 = @bitCast([16]i32{ -1, -1, -1, -1, -1, -1, -1, -1, 0, 0, 0, 0, 0, 0, 0, 0 });

/// Q4_0 nibble table: nibble -> (nib - 8 + 128) = nib + 120, replicated to all
/// four 128-bit lanes so one vpshufb resolves the 16 qs bytes.  The +128 bias is
/// folded out below by `pslld7(dpbusd(ones, yq))` = 128 * S.
const q40_tab: IQZ = blk: {
    var a: [64]u8 = undefined;
    for (0..64) |i| a[i] = @as(u8, @intCast(i % 16 + 120));
    break :blk @bitCast(a);
};
/// Q4_1 nibble table: nibble -> nib + 128 (no -8, Q4_1 keeps the min term).
const q41_tab: IQZ = blk: {
    var a: [64]u8 = undefined;
    for (0..64) |i| a[i] = @as(u8, @intCast(i % 16 + 128));
    break :blk @bitCast(a);
};
/// ones in lanes 0..31 only: vpdpbusd against the Q8_0 zmm broadcast (which
/// replicates the 32 qs bytes into both halves) sums the block exactly once.
/// A full-lane ones table would count every byte twice.
const iqn_ones32: IQZ = blk: {
    var a: [64]u8 = undefined;
    for (0..64) |i| a[i] = if (i < 32) @as(u8, 1) else 0;
    break :blk @bitCast(a);
};

/// Q4_0 / Q4_1 x Q8_0, zmm + VNNI.  Same shape as `iq4_nl_q8_0_vnni_impl`: the
/// 16 qs bytes are broadcast to every 128-bit lane, each lane shuffles its own
/// nibble, lo nibbles land in lanes 0..15 and hi nibbles in lanes 16..31, and
/// the Q8_0 side is broadcast into the low 256 bits (upper lanes zero, so no
/// read past the 34-byte block and no double counting).
/// `gs` = dpbusd(ones, yq): lanes 0..31 carry the 4-byte partial sums of the
/// block, which sum to S over the 16 lanes, and lanes 32..63 are zero.  So
/// `pslld7(gs)` = 128*S is exactly the bias to fold out of dpbusd(w, yq), and
/// floatFromInt(gs) feeds the f32 accumulator as S once per block (Q4_1's
/// min term m_x * d_y * S).
fn q4_vnni_impl(comptime tab: IQZ, comptime rb: usize, comptime m_off: usize, n: usize, vx: [*]const u8, vy: [*]const u8) f32 {
    if (comptime tier.isAvx512Vnni and @import("builtin").mode != .debug) {
        const nb = n / 32;
        var acc_out: V16f32 = @splat(0);
        var accmin: V16f32 = @splat(0);
        for (0..nb) |ib| {
            const x = vx + ib * rb;
            const y = vy + ib * q8_bytes;
            const xq = iqnBcast128(x + 2 + m_off);
            const lo = iqnPand(xq, iqn_m0f);
            const hi = iqnPand(iqnPsrld4(xq), iqn_m0f);
            const w = iqnPor(iqnPand(iqnShufb(lo, tab), iqn_mlo),
                iqnPand(iqnShufb(hi, tab), iqn_mhi));
            const yq = iqnBcast256(y + 2);
            const gs = iqnDpbusd0(iqn_ones32, yq);
            const dot = iqnPsubd(iqnDpbusd0(w, yq), iqnPslld7(gs));
            acc_out = acc_out + @as(V16f32, @splat(util.f16At(x, 0) * util.f16At(y, 0))) *
                @as(V16f32, @floatFromInt(dot));
            if (comptime m_off != 0) {
                accmin = accmin + @as(V16f32, @splat(util.f16At(x, 2) * util.f16At(y, 0))) *
                    @as(V16f32, @floatFromInt(gs));
            }
        }
        return @reduce(.Add, acc_out) + @reduce(.Add, accmin);
    }
    return 0;
}

fn iq4_nl_q8_0_vnni_impl(n: usize, vx: [*]const u8, vy: [*]const u8) callconv(.c) f32 {
    if (comptime tier.isAvx512Vnni and @import("builtin").mode != .debug) {
        const nb = n / 32;
        var acc_out: V16f32 = @splat(0);
        for (0..nb) |ib| {
            const x = vx + ib * 18;
            const y = vy + ib * q8_bytes;
            // vbroadcasti32x8 copies the 16 qs bytes into the low 256 bits and
            // leaves the high 256 bits zero, so the shuffled nibbles in the
            // upper lanes (masked out below) and their Q8_0 counterparts cost
            // nothing, and the i32 lane sums of the upper half are zero.
            const xq = iqnBcast128(x + 2);
            const lo = iqnPand(xq, iqn_m0f);
            const hi = iqnPand(iqnPsrld4(xq), iqn_m0f);
            const w = iqnPor(iqnPand(iqnShufb(lo, iqn_tab), iqn_mlo),
                iqnPand(iqnShufb(hi, iqn_tab), iqn_mhi));
            const wu = w; // table already carries the +128 bias
            const yq = iqnBcast256(y + 2);
            const acc = iqnDpbusd0(wu, yq);
            // vbroadcasti32x8 replicates the 32 qs bytes into BOTH 256-bit
            // halves, so a full ones table would see the block sum twice.
            // ones32 leaves the upper 256 bits masked: S lands in lanes 0..7,
            // lanes 8..15 stay zero, and pslld7 gives exactly the 128*S bias.
            const dot = iqnPsubd(acc, iqnPslld7(iqnDpbusd0(iqn_ones32, yq)));
            const dv: V16f32 = @splat(util.f16At(x, 0) * util.f16At(y, 0));
            acc_out = acc_out + dv * @as(V16f32, @floatFromInt(dot));
        }
        return @reduce(.Add, acc_out);
    }
    return 0;
}

pub fn iq4_nl_q8_0(n: usize, vx: [*]const u8, vy: [*]const u8) callconv(.c) f32 {
    if (comptime tier.isAvx512Vnni and @import("builtin").mode != .debug) {
        return iq4_nl_q8_0_vnni_impl(n, vx, vy);
    }
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
        const x = vx + ib * 18; // block_q1_0 = d(2) + qs[16] = 18 bytes
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
        const x = vx + ib * 18; // block_q1_0 = d(2) + qs[16] = 18 bytes
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

// IQ4_NL x Q8_0: C ggml_vec_dot_iq4_nl_q8_0_generic (quants.c:1254).  The
// implementation (with its AVX-512 VNNI path) is above, next to the codebook.
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

// ---------------------------------------------------------------------------
// AVX-512 (BW) path for Q8_0 x Q8_0 rows: vpdpwssd.
//
// The `madd16i8` version above widens 32 i8 to 32 i16, multiplies, then
// pairwise-adds through an array -- 32 imul + a shuffle-heavy reduce per 32
// elements.  `vpdpwssd` (AVX-512BW, signed i16 x i16 -> i32) does the whole
// 32-element block dot into 16 i32 lanes in one instruction: lane i gets
// a[2i]*b[2i] + a[2i+1]*b[2i+1], so the horizontal sum of the 16 lanes is
// exactly the 32-element block dot.  Four independent block chains are
// interleaved per iteration to expose ILP (measured 84 GB/s vs 70 GB/s for
// two chains on the real 4.5 GB mmap'd weight file, 16 threads).
//
// NOTE on the previous version: it used `vpdpbusd` (VNNI), which interprets
// its FIRST operand as UNSIGNED u8.  Q8_0 quants are signed i8, so every
// negative weight byte was read as 256-|w| -- a ~26% relative error against
// the scalar reference, and garbage output end to end.  vpdpwssd is signed on
// both operands and is exact.  Gating is `tier.isAvx512` (true for both the
// avx512 and avx512vnni tiers), so neither tier needs VNNI here.
//
// Same gating style as kern256.zig's vnni impls: the zmm-only register
// constraints are only semanalyzed when the branch is reachable, i.e. in an
// avx512* module at non-Debug optimization.  Debug builds (tests) and the
// baseline/avx2 tiers take the madd16i8 path below.
// ---------------------------------------------------------------------------
const Z16i16 = @Vector(32, i16);
const Z16i32 = @Vector(16, i32);

/// Load 32 bytes and sign-extend every i8 into its own i16 lane.
inline fn vloadSx32(p: [*]const u8) Z16i16 {
    return asm ("vpmovsxbw (%[src]), %[dst]" : [dst] "=v" (-> Z16i16) : [src] "r" (p));
}
/// dst[i] = a[2i]*b[2i] + a[2i+1]*b[2i+1], signed i16*i16 -> i32.
inline fn vdpwssd(a: Z16i16, b: Z16i16) Z16i32 {
    var out: Z16i32 = @splat(0);
    asm ("vpdpwssd %[b], %[a], %[out]" : [out] "+v" (out) : [a] "v" (a), [b] "v" (b));
    return out;
}
/// Fold one 16-lane i32 partial-sum vector into the f32 row accumulator.
inline fn accDp(acc: *V16f32, dp: Z16i32, s: f32) void {
    acc.* += @as(V16f32, @floatFromInt(dp)) * @as(V16f32, @splat(s));
}

pub fn q8_0_q8_0_row(n: usize, vx: [*]const u8, vy: [*]const u8) callconv(.c) f32 {
    if (comptime tier.isAvx512 and @import("builtin").mode != .debug) {
        const nb = n / 32;
        var accA: V16f32 = @splat(0);
        var accB: V16f32 = @splat(0);
        var accC: V16f32 = @splat(0);
        var accD: V16f32 = @splat(0);
        var i: usize = 0;
        while (i + 3 < nb) : (i += 4) {
            const x0 = vx + i * q8_bytes;
            const y0 = vy + i * q8_bytes;
            const dp0 = vdpwssd(vloadSx32(x0 + 2), vloadSx32(y0 + 2));
            const dp1 = vdpwssd(vloadSx32(x0 + 34 + 2), vloadSx32(y0 + 34 + 2));
            const dp2 = vdpwssd(vloadSx32(x0 + 68 + 2), vloadSx32(y0 + 68 + 2));
            const dp3 = vdpwssd(vloadSx32(x0 + 102 + 2), vloadSx32(y0 + 102 + 2));
            accDp(&accA, dp0, util.f16At(x0, 0) * util.f16At(y0, 0));
            accDp(&accB, dp1, util.f16At(x0 + 34, 0) * util.f16At(y0 + 34, 0));
            accDp(&accC, dp2, util.f16At(x0 + 68, 0) * util.f16At(y0 + 68, 0));
            accDp(&accD, dp3, util.f16At(x0 + 102, 0) * util.f16At(y0 + 102, 0));
        }
        while (i < nb) : (i += 1) {
            const x = vx + i * q8_bytes;
            const y = vy + i * q8_bytes;
            accDp(&accA, vdpwssd(vloadSx32(x + 2), vloadSx32(y + 2)),
                util.f16At(x, 0) * util.f16At(y, 0));
        }
        return util.sumF16(accA) + util.sumF16(accB) + util.sumF16(accC) + util.sumF16(accD);
    }
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
    if (comptime tier.isAvx512Vnni and @import("builtin").mode != .debug) {
        return q4_vnni_impl(q40_tab, 18, 0, n, vx, vy);
    }
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
    if (comptime tier.isAvx512Vnni and @import("builtin").mode != .debug) {
        return q4_vnni_impl(q41_tab, 20, 2, n, vx, vy);
    }
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
    if (comptime tier.isAvx512Vnni and @import("builtin").mode != .debug) {
        return iq4_nl_q8_0_vnni_impl(n, vx, vy);
    }
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
        const x = vx + ib * 18; // block_q1_0 = d(2) + qs[16] = 18 bytes
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
