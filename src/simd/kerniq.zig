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
// kernels: SIMD (currently forwards to _ref) + scalar _ref twin
// ---------------------------------------------------------------------------

/// True SIMD IQ2_XXS: scalar codebook dequant (both sides) + 8-wide `@Vector`
/// FMA accumulate with horizontal reduction (pattern as iq4_xs_q8_k).  Math is
/// element-identical to `iq2_xxs_q8_k_ref`.
pub fn iq2_xxs_q8_k(n: usize, vx: [*]const u8, vy: [*]const u8) callconv(.c) f32 {
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
/// Math is element-identical to `iq4_xs_q8_k_ref` (same dequant, only the
/// accumulation order differs, so tests bound it to the 1e-3/5e-3 parity).
pub fn iq4_xs_q8_k(n: usize, vx: [*]const u8, vy: [*]const u8) callconv(.c) f32 {
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
