//! 256-element super-block dot products for the importance-sampled IQ types.
//!
//! Ported from llama.cpp `ggml-cpu/quants.c` (`ggml_vec_dot_iq*_q8_K_generic`):
//!   * semantics:  /mnt/strata/llama.cpp/ggml/src/ggml-cpu/quants.c
//!   * vectorized: /mnt/strata/llama.cpp/ggml/src/ggml-cpu/arch/x86/quants.c
//!
//! Pure Zig: `@Vector` + std, no C, no C++, no inline asm.  build.zig compiles
//! one copy of this directory per `-mcpu` (see `src/simd/util.zig`), so the
//! same source becomes 128/256/512-bit code per tier.
//!
//! The `<name>` function is the SIMD kernel; `<name>_ref` is a scalar twin
//! (dequant both sides to f32, then dot) that mirrors the engine's verified
//! `rowDotIQ*` in `src/quant_dequant.zig`.  Until the vectorized inner loop
//! lands, `<name>` forwards to `_ref` so the dispatch table is wired and the
//! build/tests are green with correct semantics.
//!
//! `vx` is the packed weight row (one 256-element super-block per row), `vy`
//! the Q8_K activation (292 bytes/256: f32 d@0, qs[256] i8@4).  `n` is the
//! element count; the kernel processes `n/256` super-blocks.
//!
//! Block layouts (bytes, little-endian; `d`/scale fields decoded as f16):
//!   IQ2_XXS 64 B : d(f16)@0  qs[32]u16@2
//!   IQ2_XS  74 B : d(f16)@0  qs[32]u16@2  scales[8]@66
//!   IQ3_XXS 98 B : d(f16)@0  qs[64]@2  scales_and_signs[32]@66
//!   IQ4_XS  136 B: d(f16)@0  scales_h(u16)@2  scales_l[4]@4  qs[128]@8

const std = @import("std");
const util = @import("util.zig");
const tier = @import("tier.zig");
const tables = @import("tables.zig");

const kvalues_iq4nl: [16]i8 = .{ -127, -104, -83, -65, -49, -35, -22, -10, 1, 13, 25, 38, 53, 69, 89, 113 };
const iq1s_delta: f32 = 0.125;

// ---------------------------------------------------------------------------
// scalar dequant (256 f32) -- mirrors src/quant_dequant.zig rowDotIQ* exactly
// ---------------------------------------------------------------------------

fn deqIQ2XXS(row: [*]const u8, out: *[256]f32) void {
    const d = util.f16At(row, 0);
    var y: usize = 0;
    for (0..8) |ib32| {
        const aux1: u32 = util.u32At(row + 2 + 8 * ib32 + 4);
        const db = d * (0.5 + @as(f32, @floatFromInt(aux1 >> 28))) * 0.25;
        for (0..4) |l| {
            const grid: [8]u8 = @bitCast(tables.iq2xxs_grid[row[2 + 8 * ib32 + l]]);
            const signs: u8 = tables.ksigns_iq2xs[(aux1 >> @as(u5, @intCast(7 * l))) & 127];
            for (0..8) |j| {
                out[y + j] = db * @as(f32, @floatFromInt(grid[j])) * @as(f32, if (signs & tables.kmask_iq2xs[j] != 0) -1.0 else 1.0);
            }
            y += 8;
        }
    }
}

fn deqIQ2XS(row: [*]const u8, out: *[256]f32) void {
    const d = util.f16At(row, 0);
    var y: usize = 0;
    for (0..8) |ib32| {
        const sc: u8 = row[66 + ib32];
        const db0 = d * (0.5 + @as(f32, @floatFromInt(sc & 0xf))) * 0.25;
        const db1 = d * (0.5 + @as(f32, @floatFromInt(sc >> 4))) * 0.25;
        for (0..4) |l| {
            const q16: u16 = util.u16At(row + 2 + (4 * ib32 + l) * 2);
            const grid: [8]u8 = @bitCast(tables.iq2xs_grid[q16 & 511]);
            const signs: u8 = tables.ksigns_iq2xs[q16 >> 9];
            const db: f32 = if (l / 2 == 0) db0 else db1;
            for (0..8) |j| {
                out[y + j] = db * @as(f32, @floatFromInt(grid[j])) * @as(f32, if (signs & tables.kmask_iq2xs[j] != 0) -1.0 else 1.0);
            }
            y += 8;
        }
    }
}

fn deqIQ3XXS(row: [*]const u8, out: *[256]f32) void {
    const d = util.f16At(row, 0);
    var y: usize = 0;
    for (0..8) |ib32| {
        const aux32: u32 = util.u32At(row + 66 + 4 * ib32);
        const db: f32 = d * (0.5 + @as(f32, @floatFromInt(aux32 >> 28))) * 0.5;
        for (0..4) |l| {
            const sidx: u8 = @intCast((aux32 >> @as(u5, @intCast(7 * l))) & 127);
            const signs: u8 = tables.ksigns_iq2xs[sidx];
            const g1: u32 = tables.iq3xxs_grid[row[2 + 8 * ib32 + 2 * l]];
            const g2: u32 = tables.iq3xxs_grid[row[2 + 8 * ib32 + 2 * l + 1]];
            const gb1: [4]u8 = @bitCast(g1);
            const gb2: [4]u8 = @bitCast(g2);
            for (0..4) |j| {
                out[y + j] = db * @as(f32, @floatFromInt(gb1[j])) * @as(f32, if (signs & tables.kmask_iq2xs[j] != 0) -1.0 else 1.0);
                out[y + 4 + j] = db * @as(f32, @floatFromInt(gb2[j])) * @as(f32, if (signs & tables.kmask_iq2xs[4 + j] != 0) -1.0 else 1.0);
            }
            y += 8;
        }
    }
}

fn deqIQ4XS(row: [*]const u8, out: *[256]f32) void {
    const d = util.f16At(row, 0);
    const sh: u16 = @as(u16, row[2]) | (@as(u16, row[3]) << 8);
    var y: usize = 0;
    for (0..8) |ib| {
        const sl_amt: u3 = @intCast(4 * (ib % 2));
        const sh_amt: u4 = @intCast(2 * ib);
        const sl4: u16 = row[4 + ib / 2] >> sl_amt;
        const sh4: u16 = sh >> sh_amt;
        const ls: u16 = (sl4 & 0x0F) | ((sh4 & 3) << 4);
        const dl: f32 = d * @as(f32, @floatFromInt(@as(i32, ls) - 32));
        for (0..16) |j| {
            const q: u8 = row[8 + 16 * ib + j];
            out[y + j] = dl * @as(f32, @floatFromInt(kvalues_iq4nl[q & 0x0F]));
            out[y + 16 + j] = dl * @as(f32, @floatFromInt(kvalues_iq4nl[q >> 4]));
        }
        y += 32;
    }
}

// ---------------------------------------------------------------------------
// AVX-512 VNNI paths for IQ2_XXS / IQ2_XS / IQ3_XXS x Q8_K
// ---------------------------------------------------------------------------
// Same structure as iq2_s_q8_k_vnni_impl in src/simd/kerniq2.zig (commit
// e982273, measured 5.44x): scalar codebook lookup fills a 256-byte `gb`
// buffer in ELEMENT order; vectorized `vpdpbusd` accumulation with the +128
// trick (subtract 128 * per-lane qs sum via an all-ones multiplier), integer
// scale coefficient c per lane, common factor d*dy folded in f32 once per
// block.  Codebook bytes are SCALAR loads (grids exceed 128 entries, no
// gather).  Sign bytes sit in contiguous 8-byte groups per zmm ->
// `vpbroadcastq` + `vpshufb` expansion (BW-only, no VBMI needed).
//
// Lane layout (all three kernels): zmm zi (0..3) carries elements
// 64*zi..64*zi+63 in element order; i32 lane k = elements 4k..4k+3.  The
// group (ib32, l) covers elements ib32*32 + l*8 .. +7, so lanes 0..7 of zmm
// zi carry ib32 = 2*zi and lanes 8..15 carry ib32 = 2*zi+1.  Q8_K qs loads
// are element order, so they align 1:1.
//
// These zmm-only helpers are private and only semantic-analyzed from the
// `comptime tier.isAvx512Vnni and mode != .debug` branches below (same
// pattern as iq4_xs_q8_k above); `signs` is the sign byte for elements
// 8*ib32..+7, so element k tests bit k%8 -- identical to the kmask_pat
// expansion kerniq2.zig uses.

inline fn vpbroadcastqZ(x: u64) Z {
    return asm ("vpbroadcastq (%[src]), %[dst]" : [dst] "=v" (-> Z) : [src] "r" (@as([*]const u8, @ptrCast(&x))));
}

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
    return @bitCast(@as(*align(1) const [8]u8, @ptrCast(x + off)).*);
}

/// IQ2_XXS: mirrors deqIQ2XXS exactly -- group (ib, l) -> elements ib*32 + l*8,
/// grid bytes written to gb in element order, one sign byte per 8 elements into
/// sb[4*ib + l], integer coefficient c = 1 + 2*(aux1 >> 28) per ib into cs.
/// zmm zi holds elements 64*zi..+63 in element order, so the sign bytes of
/// those elements are the contiguous 8 bytes sb[8*zi..+7] (`vpbroadcastq` +
/// `vpshufb`), and i32 lane k covers elements 4k..4k+3 -> ib = 2*zi + k/8.
fn iq2_xxs_q8_k_vnni_impl(n: usize, vx: [*]const u8, vy: [*]const u8) callconv(.c) f32 {
    if (comptime tier.isAvx512Vnni and @import("builtin").mode != .debug) {
        const nb = n / 256;
        var acc_out: F32 = @splat(0);
        for (0..nb) |b| {
            const x = vx + b * 66;
            const y = vy + b * 292;
            var gb: [256]u8 = undefined;
            var sb: [32]u8 = undefined;
            var cs: [8]i32 = undefined;
            for (0..8) |ib| {
                const aux1: u32 = util.u32At(x + 6 + 8 * ib);
                cs[ib] = 1 + 2 * @as(i32, @intCast(aux1 >> 28));
                for (0..4) |l| {
                    @as(*align(1) u64, @ptrCast(&gb[32 * ib + 8 * l])).* =
                        tables.iq2xxs_grid[x[2 + 8 * ib + l]];
                    sb[4 * ib + l] = tables.ksigns_iq2xs[(aux1 >> @as(u5, @intCast(7 * l))) & 127];
                }
            }
            const gptr: [*]const u8 = @as([*]const u8, @ptrCast(&gb));
            const sptr: [*]const u8 = @as([*]const u8, @ptrCast(&sb));
            var acc: Z = v_zero;
            for (0..4) |zi| {
                const g = vload64(gptr + 64 * zi);
                const sg = vshufb(v_expand8, vpbroadcastqZ(bcast8(sptr, 8 * zi)));
                const m: @Vector(64, bool) = (@as(U, @bitCast(sg)) & kmask_pat) != v_zero_u;
                const wu: U = @as(U, @bitCast(@select(i8, m, v_zero - g, g))) +% v_128;
                const q = vload64(y + 4 + 64 * zi);
                const dot = vpsubd(vdpbusd(v_zero, @bitCast(wu), q),
                    vpslld7(vdpbusd(v_zero, v_ones, q)));
                var cv: [16]i32 = undefined;
                for (0..16) |k| {
                    cv[k] = cs[2 * zi + k / 8];
                }
                acc = vpaddd(acc, vmulld(dot, @bitCast(cv)));
            }
            const dv: F32 = @splat(0.25 * util.f16At(x, 0) * @as(f32, @bitCast(util.u32At(y))));
            acc_out = acc_out + dv * @as(F32, @floatFromInt(@as(I32, @bitCast(acc))));
        }
        return @reduce(.Add, acc_out);
    }
    return 0;
}

/// IQ2_XS: mirrors deqIQ2XS -- q16 = u16 @ 2 + 2*(4*ib + l), grid index q16 & 511
/// (u64, 8 bytes in element order), signs = ksigns[q16 >> 9] into sb[4*ib + l],
/// scale nibble db0 (sc & 0x0f) for l < 2 and db1 (sc >> 4) for l >= 2,
/// c = 1 + 2*nibble, common factor 0.25.  lane k -> ib = 2*zi + k/8,
/// l = (k/2) % 4.
fn iq2_xs_q8_k_vnni_impl(n: usize, vx: [*]const u8, vy: [*]const u8) callconv(.c) f32 {
    if (comptime tier.isAvx512Vnni and @import("builtin").mode != .debug) {
        const nb = n / 256;
        var acc_out: F32 = @splat(0);
        for (0..nb) |b| {
            const x = vx + b * 74;
            const y = vy + b * 292;
            var gb: [256]u8 = undefined;
            var sb: [32]u8 = undefined;
            for (0..8) |ib| {
                for (0..4) |l| {
                    const q16: u16 = util.u16At(x + 2 + 2 * (4 * ib + l));
                    @as(*align(1) u64, @ptrCast(&gb[32 * ib + 8 * l])).* =
                        tables.iq2xs_grid[q16 & 511];
                    sb[4 * ib + l] = tables.ksigns_iq2xs[q16 >> 9];
                }
            }
            const gptr: [*]const u8 = @as([*]const u8, @ptrCast(&gb));
            const sptr: [*]const u8 = @as([*]const u8, @ptrCast(&sb));
            var acc: Z = v_zero;
            for (0..4) |zi| {
                const g = vload64(gptr + 64 * zi);
                const sg = vshufb(v_expand8, vpbroadcastqZ(bcast8(sptr, 8 * zi)));
                const m: @Vector(64, bool) = (@as(U, @bitCast(sg)) & kmask_pat) != v_zero_u;
                const wu: U = @as(U, @bitCast(@select(i8, m, v_zero - g, g))) +% v_128;
                const q = vload64(y + 4 + 64 * zi);
                const dot = vpsubd(vdpbusd(v_zero, @bitCast(wu), q),
                    vpslld7(vdpbusd(v_zero, v_ones, q)));
                var cv: [16]i32 = undefined;
                for (0..16) |k| {
                    const sc: u8 = x[66 + 2 * zi + k / 8];
                    const nib = if ((k / 2 % 4) < 2) sc & 0x0f else sc >> 4;
                    cv[k] = 1 + 2 * @as(i32, nib);
                }
                acc = vpaddd(acc, vmulld(dot, @bitCast(cv)));
            }
            const dv: F32 = @splat(0.25 * util.f16At(x, 0) * @as(f32, @bitCast(util.u32At(y))));
            acc_out = acc_out + dv * @as(F32, @floatFromInt(@as(I32, @bitCast(acc))));
        }
        return @reduce(.Add, acc_out);
    }
    return 0;
}

/// IQ3_XXS: mirrors deqIQ3XXS -- aux32 = u32 @ 66 + 4*ib, c = 1 + 2*(aux32 >> 28)
/// (factor 0.5), two u32 grid entries per l written to gb[32*ib + 8*l .. +7],
/// signs = ksigns[(aux32 >> 7l) & 127] into sb[4*ib + l].  Same lane layout.
fn iq3_xxs_q8_k_vnni_impl(n: usize, vx: [*]const u8, vy: [*]const u8) callconv(.c) f32 {
    if (comptime tier.isAvx512Vnni and @import("builtin").mode != .debug) {
        const nb = n / 256;
        var acc_out: F32 = @splat(0);
        for (0..nb) |b| {
            const x = vx + b * 98;
            const y = vy + b * 292;
            var gb: [256]u8 = undefined;
            var sb: [32]u8 = undefined;
            var cs: [8]i32 = undefined;
            for (0..8) |ib| {
                const aux32: u32 = util.u32At(x + 66 + 4 * ib);
                cs[ib] = 1 + 2 * @as(i32, @intCast(aux32 >> 28));
                for (0..4) |l| {
                    @as(*align(1) u32, @ptrCast(&gb[32 * ib + 8 * l])).* =
                        tables.iq3xxs_grid[x[2 + 8 * ib + 2 * l]];
                    @as(*align(1) u32, @ptrCast(&gb[32 * ib + 8 * l + 4])).* =
                        tables.iq3xxs_grid[x[2 + 8 * ib + 2 * l + 1]];
                    sb[4 * ib + l] = tables.ksigns_iq2xs[(aux32 >> @as(u5, @intCast(7 * l))) & 127];
                }
            }
            const gptr: [*]const u8 = @as([*]const u8, @ptrCast(&gb));
            const sptr: [*]const u8 = @as([*]const u8, @ptrCast(&sb));
            var acc: Z = v_zero;
            for (0..4) |zi| {
                const g = vload64(gptr + 64 * zi);
                const sg = vshufb(v_expand8, vpbroadcastqZ(bcast8(sptr, 8 * zi)));
                const m: @Vector(64, bool) = (@as(U, @bitCast(sg)) & kmask_pat) != v_zero_u;
                const wu: U = @as(U, @bitCast(@select(i8, m, v_zero - g, g))) +% v_128;
                const q = vload64(y + 4 + 64 * zi);
                const dot = vpsubd(vdpbusd(v_zero, @bitCast(wu), q),
                    vpslld7(vdpbusd(v_zero, v_ones, q)));
                var cv: [16]i32 = undefined;
                for (0..16) |k| {
                    cv[k] = cs[2 * zi + k / 8];
                }
                acc = vpaddd(acc, vmulld(dot, @bitCast(cv)));
            }
            const dv: F32 = @splat(0.5 * util.f16At(x, 0) * @as(f32, @bitCast(util.u32At(y))));
            acc_out = acc_out + dv * @as(F32, @floatFromInt(@as(I32, @bitCast(acc))));
        }
        return @reduce(.Add, acc_out);
    }
    return 0;
}

/// True SIMD IQ2_XXS: scalar codebook dequant (both sides) + 8-wide `@Vector`
/// FMA accumulate with horizontal reduction (pattern as iq4_xs_q8_k).  Math is
/// element-identical to `iq2_xxs_q8_k_ref`.  The AVX-512 VNNI kernel above runs
/// in the vnni/vbmi tiers at non-Debug optimization; other tiers and Debug
/// builds use the element-identical f32 path below.
pub fn iq2_xxs_q8_k(n: usize, vx: [*]const u8, vy: [*]const u8) callconv(.c) f32 {
    if (comptime tier.isAvx512Vnni and @import("builtin").mode != .debug) {
        return iq2_xxs_q8_k_vnni_impl(n, vx, vy);
    }
    const rb: usize = 66;
    const V8f32 = @Vector(8, f32);
    var w: [256]f32 = undefined;
    var a: [256]f32 = undefined;
    var accv: V8f32 = @splat(0);
    for (0 .. n / 256) |b| {
        deqIQ2XXS(vx + b * rb, &w);
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

pub fn iq2_xxs_q8_k_ref(n: usize, vx: [*]const u8, vy: [*]const u8) callconv(.c) f32 {
    const rb: usize = 66;
    var w: [256]f32 = undefined;
    var a: [256]f32 = undefined;
    var acc: f32 = 0;
    for (0 .. n / 256) |b| {
        deqIQ2XXS(vx + b * rb, &w);
        util.deqQ8K(vy + b * 292, &a);
        for (0..256) |i| acc += w[i] * a[i];
    }
    return acc;
}

/// True SIMD IQ2_XS: scalar codebook dequant (both sides) + 8-wide `@Vector`
/// FMA accumulate with horizontal reduction (pattern as iq4_xs_q8_k).  Math is
/// element-identical to `iq2_xs_q8_k_ref`.
pub fn iq2_xs_q8_k(n: usize, vx: [*]const u8, vy: [*]const u8) callconv(.c) f32 {
    if (comptime tier.isAvx512Vnni and @import("builtin").mode != .debug) {
        return iq2_xs_q8_k_vnni_impl(n, vx, vy);
    }
    const rb: usize = 74;
    const V8f32 = @Vector(8, f32);
    var w: [256]f32 = undefined;
    var a: [256]f32 = undefined;
    var accv: V8f32 = @splat(0);
    for (0 .. n / 256) |b| {
        deqIQ2XS(vx + b * rb, &w);
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

pub fn iq2_xs_q8_k_ref(n: usize, vx: [*]const u8, vy: [*]const u8) callconv(.c) f32 {
    const rb: usize = 74;
    var w: [256]f32 = undefined;
    var a: [256]f32 = undefined;
    var acc: f32 = 0;
    for (0 .. n / 256) |b| {
        deqIQ2XS(vx + b * rb, &w);
        util.deqQ8K(vy + b * 292, &a);
        for (0..256) |i| acc += w[i] * a[i];
    }
    return acc;
}

/// True SIMD IQ3_XXS: scalar codebook dequant (both sides) + 8-wide `@Vector`
/// FMA accumulate with horizontal reduction (pattern as iq4_xs_q8_k).  Math is
/// element-identical to `iq3_xxs_q8_k_ref`.
pub fn iq3_xxs_q8_k(n: usize, vx: [*]const u8, vy: [*]const u8) callconv(.c) f32 {
    if (comptime tier.isAvx512Vnni and @import("builtin").mode != .debug) {
        return iq3_xxs_q8_k_vnni_impl(n, vx, vy);
    }
    const rb: usize = 98;
    const V8f32 = @Vector(8, f32);
    var w: [256]f32 = undefined;
    var a: [256]f32 = undefined;
    var accv: V8f32 = @splat(0);
    for (0 .. n / 256) |b| {
        deqIQ3XXS(vx + b * rb, &w);
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

pub fn iq3_xxs_q8_k_ref(n: usize, vx: [*]const u8, vy: [*]const u8) callconv(.c) f32 {
    const rb: usize = 98;
    var w: [256]f32 = undefined;
    var a: [256]f32 = undefined;
    var acc: f32 = 0;
    for (0 .. n / 256) |b| {
        deqIQ3XXS(vx + b * rb, &w);
        util.deqQ8K(vy + b * 292, &a);
        for (0..256) |i| acc += w[i] * a[i];
    }
    return acc;
}

/// True SIMD IQ4_XS: scalar codebook dequant (both sides) + 8-wide `@Vector`
/// FMA accumulate with horizontal reduction.  The weight dequant is a 512-row
/// codebook gather (`iq4xs_grid[q16&511]`) which Zig 0.17 `@Vector` cannot
/// express portably (no gather/intrinsic index load), so the dequant stays
/// scalar; the 256-wide dot's multiply-accumulate is the vectorized part.
// ---------------------------------------------------------------------------
// AVX-512 VNNI path for IQ4_XS x Q8_K (vpdpbusd int8 dot-product accumulators)
// ---------------------------------------------------------------------------
// Same trick as q4_k_q8_k in kern256.zig, adapted to IQ4_XS: the codebook is
// a 16-entry LUT (`kvalues_iq4nl`), made unsigned by adding 128, replicated 4x
// into one zmm so `vpshufb` (per-128-bit-lane) can index it.  Because the
// weights are +128-offset, every lane's contribution carries 128*(sum of that
// lane's 4 activation bytes), which is subtracted with a second vpdpbusd using
// an all-ones multiplier (llama.cpp's standard compensation).
//
// The asm helpers are private to this file and are only *analyzed* when
// reached, i.e. from the `comptime tier.isAvx512Vnni and mode != .debug`
// branch below.  In the baseline/avx2/avx512 tiers and in Debug builds the
// bodies are never semanalyzed, which is what keeps the zmm-only
// "=v"/"+v" register constraints legal (see src/simd/kern256.zig:158-166).
//
// Lane layout derived from the IQ4_XS block layout (qs byte p -> lo nibble is
// element (p/16)*32 + p%16, hi nibble is element (p/16)*32 + 16 + p%16):
//   E0 = elements   0..63  -> lanes 0..7 = ib0, lanes 8..15 = ib1
//   E1 = elements  64..127 -> lanes 0..7 = ib2, lanes 8..15 = ib3
//   E2 = elements 128..191 -> lanes 0..7 = ib4, lanes 8..15 = ib5
//   E3 = elements 192..255 -> lanes 0..7 = ib6, lanes 8..15 = ib7
// q8 loads q0..q3 are the activation qs[256] in element order, so they align
// 1:1 with E0..E3; sc0..sc3 broadcast the two per-vector sub-block scales.

const Z = @Vector(64, i8);
const U = @Vector(64, u8);
const I32 = @Vector(16, i32);
const F32 = @Vector(16, f32);

inline fn vload64(p: [*]const u8) Z {
    return asm ("vmovdqu8 (%[src]), %[dst]" : [dst] "=v" (-> Z) : [src] "r" (p));
}
inline fn vandd(a: Z, b: Z) Z {
    return asm ("vpandd %[b], %[a], %[dst]" : [dst] "=v" (-> Z) : [a] "v" (a), [b] "v" (b));
}
inline fn vsrlw4(a: Z) Z { // 16-bit lanes >> 4
    return asm ("vpsrlw $4, %[a], %[dst]" : [dst] "=v" (-> Z) : [a] "v" (a));
}
inline fn vsrlw12(a: Z) Z { // 16-bit lanes >> 12
    return asm ("vpsrlw $12, %[a], %[dst]" : [dst] "=v" (-> Z) : [a] "v" (a));
}
inline fn vsllw8(a: Z) Z { // 16-bit lanes << 8
    return asm ("vpsllw $8, %[a], %[dst]" : [dst] "=v" (-> Z) : [a] "v" (a));
}
/// byte-granular high nibbles.  `vpsrlw $4` + `&0x0f` puts hi(b_even) into the
/// even byte of each 16-bit pair and ZERO into the odd byte; the odd-byte hi
/// nibbles are recovered with `vpsrlw $12` << 8.  (A plain i8-lane `A >> 4` is
/// an arithmetic shift, so the word-lane pair is the cheap way out.)
inline fn hiNibble(a: Z) Z {
    return (vsrlw4(a) & v_m4) | vsllw8(vsrlw12(a));
}
inline fn vshufb(idx: Z, tab: Z) Z { // dst[i] = tab[idx[i]] per 128-bit lane
    return asm ("vpshufb %[idx], %[tab], %[dst]" : [dst] "=v" (-> Z) : [idx] "v" (idx), [tab] "v" (tab));
}
/// qword-granularity 2-source permute, VERIFIED AT&T form (see /tmp/ZIGFACTS.md):
/// Intel `VPERMT2Q zmm1, zmm2, zmm3` prints AT&T as `vpermt2q %zmm3, %zmm2, %zmm1`
/// and zmm1 (= dst) *is* the index vector, zmm2/3 the two sources.  So
/// `out qword j = (idx[j] < 8) ? a[idx[j]] : b[idx[j] & 7]`, with `out` seeded
/// from `idx`.
inline fn vpermt2q(idx: Z, a: Z, b: Z) Z {
    var out: Z = idx;
    asm ("vpermt2q %[b], %[a], %[out]" : [out] "+v" (out) : [a] "v" (a), [b] "v" (b));
    return out;
}
/// dword-granularity permute: dst[i] = src[idx[i]]
inline fn vpermd(src: Z, idx: Z) Z {
    return asm ("vpermd %[src], %[idx], %[dst]" : [dst] "=v" (-> Z) : [src] "v" (src), [idx] "v" (idx));
}
inline fn vdpbusd(acc: Z, a: Z, b: Z) Z { // acc += (unsigned a) * (signed b)
    var out: Z = acc;
    asm ("vpdpbusd %[b], %[a], %[out]" : [out] "+v" (out) : [a] "v" (a), [b] "v" (b));
    return out;
}
inline fn vpaddd(a: Z, b: Z) Z {
    return asm ("vpaddd %[b], %[a], %[dst]" : [dst] "=v" (-> Z) : [a] "v" (a), [b] "v" (b));
}
inline fn vpsubd(a: Z, b: Z) Z {
    return asm ("vpsubd %[b], %[a], %[dst]" : [dst] "=v" (-> Z) : [a] "v" (a), [b] "v" (b));
}
inline fn vmulld(a: Z, b: Z) Z {
    return asm ("vpmulld %[b], %[a], %[dst]" : [dst] "=v" (-> Z) : [a] "v" (a), [b] "v" (b));
}
inline fn vpslld7(a: Z) Z { // i32 lanes << 7 == *128
    return asm ("vpslld $7, %[a], %[dst]" : [dst] "=v" (-> Z) : [a] "v" (a));
}

// nibble mask / all-ones multiplier / zero accumulator seed
const v_m4: Z = @bitCast(@as(U, @splat(@as(u8, 0x0f))));
const v_ones: Z = @bitCast(@as(U, @splat(@as(u8, 1))));
const v_zero: Z = @splat(0);

// qword permute indices: E0/E2 (even) and E1/E3 (odd) -- see header comment
const v_pA: Z = @bitCast([8]i64{ 0, 1, 8, 9, 2, 3, 10, 11 });
const v_pB: Z = @bitCast([8]i64{ 4, 5, 12, 13, 6, 7, 14, 15 });
const v_sc0: Z = @bitCast([16]i32{ 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1 });
const v_sc1: Z = @bitCast([16]i32{ 2, 2, 2, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3, 3, 3 });
const v_sc2: Z = @bitCast([16]i32{ 4, 4, 4, 4, 4, 4, 4, 4, 5, 5, 5, 5, 5, 5, 5, 5 });
const v_sc3: Z = @bitCast([16]i32{ 6, 6, 6, 6, 6, 6, 6, 6, 7, 7, 7, 7, 7, 7, 7, 7 });

/// kvalues_iq4nl + 128 (unsigned), replicated 4x so every 128-bit lane of the
/// zmm holds the full 16-entry table for vpshufb.  Built with a comptime loop
/// (array repetition `tab16 ** 4` does not compile on Zig 0.17).
const tab16x4: Z = blk: {
    var a: [64]u8 = undefined;
    for (0..4) |i| {
        for (0..16) |j| {
            a[i * 16 + j] = @as(u8, @intCast(@as(i32, kvalues_iq4nl[j]) + 128));
        }
    }
    break :blk @bitCast(a);
};

fn iq4_xs_q8_k_vnni_impl(n: usize, vx: [*]const u8, vy: [*]const u8) callconv(.c) f32 {
    if (comptime tier.isAvx512Vnni and @import("builtin").mode != .debug) {
        const nb = n / 256;
        // Per-block f32 contributions accumulate in zmm lanes; the horizontal
        // reduction happens ONCE after the block loop.
        var acc_out: F32 = @splat(0);
        for (0..nb) |b| {
            const x = vx + b * 136;
            const y = vy + b * 292;
            // 128 qs bytes -> lo/hi nibbles, then qword permute into element order
            const A0 = vload64(x + 8);
            const A1 = vload64(x + 72);
            const lo0 = vandd(A0, v_m4);
            const hi0 = hiNibble(A0);
            const lo1 = vandd(A1, v_m4);
            const hi1 = hiNibble(A1);
            const w0 = vshufb(vpermt2q(v_pA, lo0, hi0), tab16x4);
            const w1 = vshufb(vpermt2q(v_pB, lo0, hi0), tab16x4);
            const w2 = vshufb(vpermt2q(v_pA, lo1, hi1), tab16x4);
            const w3 = vshufb(vpermt2q(v_pB, lo1, hi1), tab16x4);
            const q0 = vload64(y + 4);
            const q1 = vload64(y + 68);
            const q2 = vload64(y + 132);
            const q3 = vload64(y + 196);
            // scales: ls[ib] - 32, ib = 0..8, in lanes 0..7 (8..15 unused)
            var ls16: [16]i32 = @splat(0);
            const sh: u16 = @as(u16, x[2]) | (@as(u16, x[3]) << 8);
            for (0..8) |ib| {
                const sl4: u16 = x[4 + ib / 2] >> @as(u3, @intCast(4 * (ib % 2)));
                const sh4: u16 = sh >> @as(u4, @intCast(2 * ib));
                const ls: u16 = (sl4 & 0x0F) | ((sh4 & 3) << 4);
                ls16[ib] = @as(i32, ls) - 32;
            }
            const lsv: Z = @bitCast(ls16);
            const sc0 = vpermd(lsv, v_sc0);
            const sc1 = vpermd(lsv, v_sc1);
            const sc2 = vpermd(lsv, v_sc2);
            const sc3 = vpermd(lsv, v_sc3);
            // per lane: scale * (sum(lut+128)*q8 - 128*sum(q8))
            var acc = vpsubd(vmulld(vdpbusd(v_zero, w0, q0), sc0), vpslld7(vmulld(vdpbusd(v_zero, v_ones, q0), sc0)));
            acc = vpaddd(acc, vpsubd(vmulld(vdpbusd(v_zero, w1, q1), sc1), vpslld7(vmulld(vdpbusd(v_zero, v_ones, q1), sc1))));
            acc = vpaddd(acc, vpsubd(vmulld(vdpbusd(v_zero, w2, q2), sc2), vpslld7(vmulld(vdpbusd(v_zero, v_ones, q2), sc2))));
            acc = vpaddd(acc, vpsubd(vmulld(vdpbusd(v_zero, w3, q3), sc3), vpslld7(vmulld(vdpbusd(v_zero, v_ones, q3), sc3))));
            const dy = @as(f32, @bitCast(util.u32At(y)));
            const dv: F32 = @splat(util.f16At(x, 0) * dy);
            acc_out = acc_out + dv * @as(F32, @floatFromInt(@as(I32, @bitCast(acc))));
        }
        return @reduce(.Add, acc_out);
    }
    return 0;
}

/// Math is element-identical to `iq4_xs_q8_k_ref` (same dequant, only the
/// accumulation order differs, so tests bound it to the 1e-3/5e-3 parity).
/// The AVX-512 VNNI kernel (above) runs in the vnni tier at non-Debug
/// optimization; every other tier/mode uses the @Vector f32 path below.
pub fn iq4_xs_q8_k(n: usize, vx: [*]const u8, vy: [*]const u8) callconv(.c) f32 {
    // The VNNI kernel is only analyzed in the vnni module at non-Debug
    // optimization; Debug builds and the other tiers fall through to the
    // @Vector f32 path below, which is element-identical.
    if (comptime tier.isAvx512Vnni and @import("builtin").mode != .debug) {
        return iq4_xs_q8_k_vnni_impl(n, vx, vy);
    }
    const rb: usize = 136;
    const V8f32 = @Vector(8, f32);
    var w: [256]f32 = undefined;
    var a: [256]f32 = undefined;
    var acc: f32 = 0;
    var accv: V8f32 = @splat(0);
    for (0 .. n / 256) |b| {
        deqIQ4XS(vx + b * rb, &w);
        util.deqQ8K(vy + b * 292, &a);
        var i: usize = 0;
        while (i < 256) : (i += 8) {
            const wv = util.loadF32(8, &w[i]);
            const av = util.loadF32(8, &a[i]);
            accv = accv + wv * av;
        }
    }
    acc = @reduce(.Add, accv);
    return acc;
}

pub fn iq4_xs_q8_k_ref(n: usize, vx: [*]const u8, vy: [*]const u8) callconv(.c) f32 {
    const rb: usize = 136;
    var w: [256]f32 = undefined;
    var a: [256]f32 = undefined;
    var acc: f32 = 0;
    for (0 .. n / 256) |b| {
        deqIQ4XS(vx + b * rb, &w);
        util.deqQ8K(vy + b * 292, &a);
        for (0..256) |i| acc += w[i] * a[i];
    }
    return acc;
}
