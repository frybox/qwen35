//! 256-element super-block dot products for IQ2_S and IQ3_S.
//!
//! Ported from llama.cpp `ggml-cpu/quants.c` (`ggml_vec_dot_iq2_s_q8_K_generic`
//! / `_iq3_s`).  The `<name>` function is the SIMD kernel; `<name>_ref` is a
//! scalar twin (dequant both sides to f32, then dot) mirroring the engine's
//! verified `rowDotIQ2S` / `rowDotIQ3S` in `src/quant_dequant.zig`.  Until the
//! vectorized inner loop lands, `<name>` forwards to `_ref` so the dispatch
//! table is wired and the build/tests are green with correct semantics.
//!
//! `vx` packed weight row (one 256-element super-block per row), `vy` Q8_K
//! activation (292 bytes/256: f32 d@0, qs[256] i8@4), `n` element count.
//!
//! Block layouts (bytes, little-endian):
//!   IQ2_S 82 B : d(f16)@0  qs[64]@2  qh[8]@66  scales[8]@74  (signs = qs+32)
//!   IQ3_S 110 B: d(f16)@0  qs[64]@2  qh[8]@66  signs[32]@74  scales[4]@106

const std = @import("std");
const util = @import("util.zig");
const tier = @import("tier.zig");
const tables = @import("tables.zig");

// ---------------------------------------------------------------------------
// scalar dequant (256 f32) -- mirrors src/quant_dequant.zig exactly
// ---------------------------------------------------------------------------

fn deqIQ2S(row: [*]const u8, out: *[256]f32) void {
    const d = util.f16At(row, 0);
    var y: usize = 0;
    for (0..8) |ib32| {
        const sc: u8 = row[74 + ib32];
        const db0 = d * (0.5 + @as(f32, @floatFromInt(sc & 0xf))) * 0.25;
        const db1 = d * (0.5 + @as(f32, @floatFromInt(sc >> 4))) * 0.25;
        const qh_val: u16 = row[66 + ib32];
        for (0..4) |l| {
            const dl: f32 = if (l / 2 == 0) db0 else db1;
            const qsv: u16 = row[2 + 4 * ib32 + l];
            const shifted: u16 = (qh_val << @as(u4, @intCast(8 - 2 * l))) & 0x300;
            const idx: u16 = qsv | shifted;
            const grid: [8]u8 = @bitCast(tables.iq2s_grid[idx]);
            const sign_byte: u8 = row[34 + 4 * ib32 + l];
            for (0..8) |j| {
                out[y + j] = dl * @as(f32, @floatFromInt(grid[j])) * @as(f32, if (sign_byte & tables.kmask_iq2xs[j] != 0) -1.0 else 1.0);
            }
            y += 8;
        }
    }
}

fn deqIQ3S(row: [*]const u8, out: *[256]f32) void {
    const d = util.f16At(row, 0);
    var y: usize = 0;
    for (0..4) |p| { // p = 0..3, each covers 64 elements
        const sbyte: u8 = row[106 + p];
        const db1: f32 = d * @as(f32, @floatFromInt(1 + 2 * (sbyte & 0x0F)));
        const db2: f32 = d * @as(f32, @floatFromInt(1 + 2 * (sbyte >> 4)));
        const qh0: u16 = row[66 + 2 * p];
        for (0..4) |l| {
            const bit1: u16 = (qh0 << @as(u4, @intCast(8 - 2 * l))) & 0x100;
            const bit2: u16 = (qh0 << @as(u4, @intCast(7 - 2 * l))) & 0x100;
            const g1: u32 = tables.iq3s_grid[row[2 + 16 * p + 2 * l] | bit1];
            const g2: u32 = tables.iq3s_grid[row[2 + 16 * p + 2 * l + 1] | bit2];
            const sg: u8 = row[74 + 8 * p + l];
            const gb1: [4]u8 = @bitCast(g1);
            const gb2: [4]u8 = @bitCast(g2);
            for (0..4) |j| {
                out[y + l * 8 + j] = db1 * @as(f32, @floatFromInt(@as(i32, gb1[j]))) * @as(f32, if (sg & tables.kmask_iq2xs[j] != 0) -1.0 else 1.0);
                out[y + l * 8 + 4 + j] = db1 * @as(f32, @floatFromInt(@as(i32, gb2[j]))) * @as(f32, if (sg & tables.kmask_iq2xs[4 + j] != 0) -1.0 else 1.0);
            }
        }
        const qh1: u16 = row[66 + 2 * p + 1];
        for (0..4) |l| {
            const bit1: u16 = (qh1 << @as(u4, @intCast(8 - 2 * l))) & 0x100;
            const bit2: u16 = (qh1 << @as(u4, @intCast(7 - 2 * l))) & 0x100;
            const g1: u32 = tables.iq3s_grid[row[2 + 16 * p + 8 + 2 * l] | bit1];
            const g2: u32 = tables.iq3s_grid[row[2 + 16 * p + 8 + 2 * l + 1] | bit2];
            const sg: u8 = row[74 + 8 * p + 4 + l];
            const gb1: [4]u8 = @bitCast(g1);
            const gb2: [4]u8 = @bitCast(g2);
            for (0..4) |j| {
                out[y + 32 + l * 8 + j] = db2 * @as(f32, @floatFromInt(@as(i32, gb1[j]))) * @as(f32, if (sg & tables.kmask_iq2xs[j] != 0) -1.0 else 1.0);
                out[y + 32 + l * 8 + 4 + j] = db2 * @as(f32, @floatFromInt(@as(i32, gb2[j]))) * @as(f32, if (sg & tables.kmask_iq2xs[4 + j] != 0) -1.0 else 1.0);
            }
        }
        y += 64;
    }
}


// ---------------------------------------------------------------------------
// AVX-512 VNNI path (IQ2_S / IQ3_S)
// ---------------------------------------------------------------------------
// Private zmm-only inline-asm helpers; they are only semantic-analyzed when
// reached, i.e. from the `comptime tier.isAvx512Vnni and mode != .debug`
// branch below.  Debug test builds and the non-VNNI tiers never analyze them,
// which is what keeps the zmm-only "=v"/"+v" constraints legal (pattern copied
// from src/simd/kerniq.zig `iq4_xs_q8_k`, commit 16ee77f).
//
// Lane layout (both kernels): zmm zi (0..3) carries elements 64*zi..64*zi+63
// in ELEMENT order (the `gb` buffer below is built in element order), i32 lane
// k = elements 4k..4k+3 of that zmm.  The Q8_K qs loads are element order, so
// they align 1:1.
//
// Codebook bytes are SCALAR loads: iq2s_grid (1024 entries) / iq3s_grid (512)
// exceed 128 entries, and Zig 0.17 inline asm cannot express AVX-512 mask
// registers (no vpgatherdd), so the gather must be scalar (llama.cpp does the
// same).  Signs sit in contiguous 8-byte groups per zmm (IQ2_S row[34+8*zi],
// IQ3_S row[74+8*zi]) -> vpbroadcastq + vpshufb expansion (BW-only, so the
// avx512vnni tier gets the path too; no VBMI required).
//
// +128 trick (see /tmp/ZIGFACTS.md): codebook value v is signed (+/-1..43).
// Store u8 = v + 128; vpdpbusd against signed qs accumulates sum((v+128)*qs)
// per i32 lane; subtract 128*sum(qs in lane) (vpdpbusd with an all-ones
// multiplier, << 7) to recover sum(v*qs).  Multiply by the integer scale
// coefficient c = 1 + 2*nibble; the common factor d*dy (x0.25 for IQ2_S) is
// folded in f32 once per block.  Everything stays integer inside the loop.

const Z = @Vector(64, i8);
const U = @Vector(64, u8);
const I32 = @Vector(16, i32);
const F32 = @Vector(16, f32);

inline fn vload64(p: [*]const u8) Z {
    return asm ("vmovdqu8 (%[src]), %[dst]" : [dst] "=v" (-> Z) : [src] "r" (p));
}
inline fn vshufbZ(idx: Z, tab: Z) Z { // dst[i] = tab[idx[i]] per 128-bit lane
    return asm ("vpshufb %[idx], %[tab], %[dst]" : [dst] "=v" (-> Z) : [idx] "v" (idx), [tab] "v" (tab));
}
inline fn vpbroadcastqZ(x: u64) Z {
    return asm ("vpbroadcastq (%[src]), %[dst]" : [dst] "=v" (-> Z) : [src] "r" (@as([*]const u8, @ptrCast(&x))));
}
inline fn vdpbusdZ(acc: Z, a: Z, b: Z) Z { // acc += (unsigned a) * (signed b) -> 16 i32
    var out: Z = acc;
    asm ("vpdpbusd %[b], %[a], %[out]" : [out] "+v" (out) : [a] "v" (a), [b] "v" (b));
    return out;
}
inline fn vpadddZ(a: Z, b: Z) Z {
    return asm ("vpaddd %[b], %[a], %[dst]" : [dst] "=v" (-> Z) : [a] "v" (a), [b] "v" (b));
}
inline fn vpsubdZ(a: Z, b: Z) Z {
    return asm ("vpsubd %[b], %[a], %[dst]" : [dst] "=v" (-> Z) : [a] "v" (a), [b] "v" (b));
}
inline fn vmulldZ(a: Z, b: Z) Z {
    return asm ("vpmulld %[b], %[a], %[dst]" : [dst] "=v" (-> Z) : [a] "v" (a), [b] "v" (b));
}
inline fn vpslld7Z(a: Z) Z { // i32 lanes << 7 == *128
    return asm ("vpslld $7, %[a], %[dst]" : [dst] "=v" (-> Z) : [a] "v" (a));
}

const v_zero: Z = @splat(0);
const v_ones: Z = @bitCast(@as(U, @splat(@as(u8, 1))));
const v_128: U = @splat(@as(u8, 128));
const v_zero_u: U = @splat(@as(u8, 0));

/// kmask_iq2xs replicated 8x (element k tests bit k % 8).
const kmask_pat: U = blk: {
    var a: [64]u8 = undefined;
    for (0..8) |i| {
        for (0..8) |j| {
            a[i * 8 + j] = tables.kmask_iq2xs[j];
        }
    }
    break :blk @bitCast(a);
};

/// vpshufb expansion: one sign byte g -> 8 copies (byte k selects g = k / 8).
const v_expand8: Z = blk: {
    var a: [64]u8 = undefined;
    for (0..64) |k| {
        a[k] = @as(u8, @intCast(k / 8));
    }
    break :blk @bitCast(a);
};

fn bcast8(x: [*]const u8, off: usize) u64 {
    return @bitCast(@as(*const [8]u8, @ptrCast(x + off)).*);
}

fn iq2_s_q8_k_vnni_impl(n: usize, vx: [*]const u8, vy: [*]const u8) callconv(.c) f32 {
    if (comptime tier.isAvx512Vnni and @import("builtin").mode != .debug) {
        const nb = n / 256;
        var acc_out: F32 = @splat(0);
        for (0..nb) |b| {
            const x = vx + b * 82;
            const y = vy + b * 292;
            // codebook bytes in element order: group (ib32, l) -> elements
            // ib32*32 + l*8, idx = qs | ((qh << (8-2l)) & 0x300)
            var gb: [256]u8 = undefined;
            var o: usize = 0;
            for (0..8) |ib| {
                const qhv: u16 = x[66 + ib];
                for (0..4) |l| {
                    const idx: u16 = @as(u16, x[2 + 4 * ib + l]) |
                        ((qhv << @as(u4, @intCast(8 - 2 * l))) & 0x300);
                    const g8: [8]u8 = @bitCast(tables.iq2s_grid[idx]);
                    for (0..8) |j| { gb[o + j] = g8[j]; }
                    o += 8;
                }
            }
            const gptr: [*]const u8 = @as([*]const u8, @ptrCast(&gb));
            var acc: Z = v_zero;
            for (0..4) |zi| {
                const g = vload64(gptr + 64 * zi);
                const sg = vshufbZ(v_expand8, vpbroadcastqZ(bcast8(x, 34 + 8 * zi)));
                const m: @Vector(64, bool) = (@as(U, @bitCast(sg)) & kmask_pat) != v_zero_u;
                const wu: U = @as(U, @bitCast(@select(i8, m, v_zero - g, g))) +% v_128;
                const q = vload64(y + 4 + 64 * zi);
                const dot = vpsubdZ(vdpbusdZ(v_zero, @bitCast(wu), q),
                    vpslld7Z(vdpbusdZ(v_zero, v_ones, q)));
                // lane k: ib32 = 2*zi + k/8, nibble db0 (k%8 < 4) / db1
                var cv: [16]i32 = undefined;
                for (0..16) |k| {
                    const sc: u8 = x[74 + 2 * zi + k / 8];
                    const nib = if ((k % 8) / 4 == 0) sc & 0x0f else sc >> 4;
                    cv[k] = 1 + 2 * @as(i32, nib);
                }
                acc = vpadddZ(acc, vmulldZ(dot, @bitCast(cv)));
            }
            const dv: F32 = @splat(0.25 * util.f16At(x, 0) * @as(f32, @bitCast(util.u32At(y))));
            acc_out = acc_out + dv * @as(F32, @floatFromInt(@as(I32, @bitCast(acc))));
        }
        return @reduce(.Add, acc_out);
    }
    return 0;
}

fn iq3_s_q8_k_vnni_impl(n: usize, vx: [*]const u8, vy: [*]const u8) callconv(.c) f32 {
    if (comptime tier.isAvx512Vnni and @import("builtin").mode != .debug) {
        const nb = n / 256;
        var acc_out: F32 = @splat(0);
        for (0..nb) |b| {
            const x = vx + b * 110;
            const y = vy + b * 292;
            // element order within p: group l = bytes g1[0..4] ++ g2[0..4];
            // elements 32..63 of p come from qh1 / qs offset +8
            var gb: [256]u8 = undefined;
            for (0..4) |p| {
                const qh0: u16 = x[66 + 2 * p];
                const qh1: u16 = x[66 + 2 * p + 1];
                const qb: usize = 2 + 16 * p;
                var o: usize = 64 * p;
                for (0..4) |l| {
                    const gi1: u16 = @as(u16, x[qb + 2 * l]) |
                        ((qh0 << @as(u4, @intCast(8 - 2 * l))) & 0x100);
                    const gi2: u16 = @as(u16, x[qb + 2 * l + 1]) |
                        ((qh0 << @as(u4, @intCast(7 - 2 * l))) & 0x100);
                    const g4a: [4]u8 = @bitCast(tables.iq3s_grid[gi1]);
                    const g4b: [4]u8 = @bitCast(tables.iq3s_grid[gi2]);
                    for (0..4) |j| { gb[o + j] = g4a[j]; gb[o + 4 + j] = g4b[j]; }
                    o += 8;
                }
                o = 64 * p + 32;
                for (0..4) |l| {
                    const gi1: u16 = @as(u16, x[qb + 8 + 2 * l]) |
                        ((qh1 << @as(u4, @intCast(8 - 2 * l))) & 0x100);
                    const gi2: u16 = @as(u16, x[qb + 8 + 2 * l + 1]) |
                        ((qh1 << @as(u4, @intCast(7 - 2 * l))) & 0x100);
                    const g4a: [4]u8 = @bitCast(tables.iq3s_grid[gi1]);
                    const g4b: [4]u8 = @bitCast(tables.iq3s_grid[gi2]);
                    for (0..4) |j| { gb[o + j] = g4a[j]; gb[o + 4 + j] = g4b[j]; }
                    o += 8;
                }
            }
            const gptr: [*]const u8 = @as([*]const u8, @ptrCast(&gb));
            var acc: Z = v_zero;
            for (0..4) |zi| {
                const g = vload64(gptr + 64 * zi);
                const sg = vshufbZ(v_expand8, vpbroadcastqZ(bcast8(x, 74 + 8 * zi)));
                const m: @Vector(64, bool) = (@as(U, @bitCast(sg)) & kmask_pat) != v_zero_u;
                const wu: U = @as(U, @bitCast(@select(i8, m, v_zero - g, g))) +% v_128;
                const q = vload64(y + 4 + 64 * zi);
                const dot = vpsubdZ(vdpbusdZ(v_zero, @bitCast(wu), q),
                    vpslld7Z(vdpbusdZ(v_zero, v_ones, q)));
                // lanes 0..7 (elements 0..31 of p) use db1's coefficient,
                // lanes 8..15 db2's
                const sbyte: u8 = x[106 + zi];
                const c1: i32 = 1 + 2 * @as(i32, sbyte & 0x0f);
                const c2: i32 = 1 + 2 * @as(i32, sbyte >> 4);
                var cv: [16]i32 = undefined;
                for (0..8) |k| {
                    cv[k] = c1;
                    cv[8 + k] = c2;
                }
                acc = vpadddZ(acc, vmulldZ(dot, @bitCast(cv)));
            }
            const dv: F32 = @splat(util.f16At(x, 0) * @as(f32, @bitCast(util.u32At(y))));
            acc_out = acc_out + dv * @as(F32, @floatFromInt(@as(I32, @bitCast(acc))));
        }
        return @reduce(.Add, acc_out);
    }
    return 0;
}

// ---------------------------------------------------------------------------
// kernels: SIMD (currently forwards to _ref) + scalar _ref twin
// ---------------------------------------------------------------------------

/// True SIMD IQ2_S: scalar codebook dequant (both sides) + 8-wide `@Vector`
/// FMA accumulate with horizontal reduction (pattern as iq4_xs_q8_k / iq3_s_q8_k).
/// Math is element-identical to `iq2_s_q8_k_ref`.
pub fn iq2_s_q8_k(n: usize, vx: [*]const u8, vy: [*]const u8) callconv(.c) f32 {
    if (comptime tier.isAvx512Vnni and @import("builtin").mode != .debug) {
        return iq2_s_q8_k_vnni_impl(n, vx, vy);
    }
    const rb: usize = 82;
    const V8f32 = @Vector(8, f32);
    var w: [256]f32 = undefined;
    var a: [256]f32 = undefined;
    var accv: V8f32 = @splat(0);
    for (0 .. n / 256) |b| {
        deqIQ2S(vx + b * rb, &w);
        util.deqQ8K(vy + b * 292, &a);
        var i: usize = 0;
        while (i < 256) : (i += 8) {
            const wv = util.loadF32(8, &w[i]);
            const av = util.loadF32(8, &a[i]);
            accv = accv + wv * av;
        }
    }
    return @reduce(.Add, accv);
}

pub fn iq2_s_q8_k_ref(n: usize, vx: [*]const u8, vy: [*]const u8) callconv(.c) f32 {
    const rb: usize = 82;
    var w: [256]f32 = undefined;
    var a: [256]f32 = undefined;
    var acc: f32 = 0;
    for (0 .. n / 256) |b| {
        deqIQ2S(vx + b * rb, &w);
        util.deqQ8K(vy + b * 292, &a);
        for (0..256) |i| acc += w[i] * a[i];
    }
    return acc;
}

/// True SIMD IQ3_S: scalar codebook dequant (both sides) + 8-wide `@Vector`
/// FMA accumulate with horizontal reduction (see iq4_xs_q8_k in kerniq.zig for
/// the same pattern and why the dequant stays scalar: codebook gather has no
/// portable `@Vector` form in Zig 0.17).  Math is element-identical to
/// `iq3_s_q8_k_ref` (only accumulation order differs).
pub fn iq3_s_q8_k(n: usize, vx: [*]const u8, vy: [*]const u8) callconv(.c) f32 {
    if (comptime tier.isAvx512Vnni and @import("builtin").mode != .debug) {
        return iq3_s_q8_k_vnni_impl(n, vx, vy);
    }
    const rb: usize = 110;
    const V8f32 = @Vector(8, f32);
    var w: [256]f32 = undefined;
    var a: [256]f32 = undefined;
    var accv: V8f32 = @splat(0);
    for (0 .. n / 256) |b| {
        deqIQ3S(vx + b * rb, &w);
        util.deqQ8K(vy + b * 292, &a);
        var i: usize = 0;
        while (i < 256) : (i += 8) {
            const wv = util.loadF32(8, &w[i]);
            const av = util.loadF32(8, &a[i]);
            accv = accv + wv * av;
        }
    }
    return @reduce(.Add, accv);
}

pub fn iq3_s_q8_k_ref(n: usize, vx: [*]const u8, vy: [*]const u8) callconv(.c) f32 {
    const rb: usize = 110;
    var w: [256]f32 = undefined;
    var a: [256]f32 = undefined;
    var acc: f32 = 0;
    for (0 .. n / 256) |b| {
        deqIQ3S(vx + b * rb, &w);
        util.deqQ8K(vy + b * 292, &a);
        for (0..256) |i| acc += w[i] * a[i];
    }
    return acc;
}
