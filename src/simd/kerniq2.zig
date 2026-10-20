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
// kernels: SIMD (currently forwards to _ref) + scalar _ref twin
// ---------------------------------------------------------------------------

/// True SIMD IQ2_S: scalar codebook dequant (both sides) + 8-wide `@Vector`
/// FMA accumulate with horizontal reduction (pattern as iq4_xs_q8_k / iq3_s_q8_k).
/// Math is element-identical to `iq2_s_q8_k_ref`.
pub fn iq2_s_q8_k(n: usize, vx: [*]const u8, vy: [*]const u8) callconv(.c) f32 {
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
