//! 256-element super-block dot products for IQ1_S and IQ1_M.
//!
//! Ported from llama.cpp `ggml-cpu/quants.c` (`ggml_vec_dot_iq1_s_q8_K_generic`
//! / `_iq1_m`).  The `<name>` function is the SIMD kernel; `<name>_ref` is a
//! scalar twin (dequant both sides to f32, then dot) mirroring the engine's
//! verified `rowDotIQ1S` / `rowDotIQ1M` in `src/quant_dequant.zig`.  Until the
//! vectorized inner loop lands, `<name>` forwards to `_ref` so the dispatch
//! table is wired and the build/tests are green with correct semantics.
//!
//! `vx` packed weight row (one 256-element super-block per row), `vy` Q8_K
//! activation (292 bytes/256: f32 d@0, qs[256] i8@4), `n` element count.
//!
//! Block layouts (bytes, little-endian):
//!   IQ1_S 42 B : d(f16)@0  qs[32]@2  qh[8]u16@34
//!   IQ1_M 56 B : qs[32]@0  qh[16]@32  scales[8]@48  (d packed from scales)

const std = @import("std");
const util = @import("util.zig");
const tier = @import("tier.zig");
const tables = @import("tables.zig");

const iq1s_delta: f32 = 0.125;

// ---------------------------------------------------------------------------
// scalar dequant (256 f32) -- mirrors src/quant_dequant.zig exactly
// ---------------------------------------------------------------------------

fn deqIQ1S(row: [*]const u8, out: *[256]f32) void {
    const d = util.f16At(row, 0);
    var y: usize = 0;
    for (0..8) |ib| {
        const qh16: u16 = util.u16At(row + 34 + 2 * ib);
        const dl = d * @as(f32, @floatFromInt(2 * ((qh16 >> 12) & 7) + 1));
        const delta: f32 = if (qh16 & 0x8000 != 0) -iq1s_delta else iq1s_delta;
        for (0..4) |l| {
            const idx: u16 = row[2 + 4 * ib + l] | @as(u16, ((qh16 >> @as(u4, @intCast(3 * l))) & 7) << 8);
            const grid: [8]i8 = @bitCast(tables.iq1s_grid[idx]);
            for (0..8) |j| {
                out[y + j] = dl * (@as(f32, @floatFromInt(grid[j])) + delta);
            }
            y += 8;
        }
    }
}

fn deqIQ1M(row: [*]const u8, out: *[256]f32) void {
    const sc0: u16 = util.u16At(row + 48);
    const sc1: u16 = util.u16At(row + 50);
    const sc2: u16 = util.u16At(row + 52);
    const sc3: u16 = util.u16At(row + 54);
    const scale_u16: u16 = (sc0 >> 12) | ((sc1 >> 8) & 0x00f0) | ((sc2 >> 4) & 0x0f00) | (sc3 & 0xf000);
    const d: f32 = @as(f32, @floatCast(@as(f16, @bitCast(scale_u16))));
    const sc_u16: [4]u16 = .{ sc0, sc1, sc2, sc3 };
    var y: usize = 0;
    for (0..8) |ib| {
        const sh: u16 = sc_u16[ib / 2];
        const dl1 = d * @as(f32, @floatFromInt(2 * ((sh >> @as(u4, @intCast(6 * (ib % 2)))) & 7) + 1));
        const dl2 = d * @as(f32, @floatFromInt(2 * ((sh >> @as(u4, @intCast(6 * (ib % 2) + 3))) & 7) + 1));
        const qh0: u16 = row[32 + 2 * ib];
        const qh1: u16 = row[32 + 2 * ib + 1];
        const idx0: u16 = row[4 * ib + 0] | @as(u16, ((qh0 << 8) & 0x700));
        const idx1: u16 = row[4 * ib + 1] | @as(u16, ((qh0 << 4) & 0x700));
        const idx2: u16 = row[4 * ib + 2] | @as(u16, ((qh1 << 8) & 0x700));
        const idx3: u16 = row[4 * ib + 3] | @as(u16, ((qh1 << 4) & 0x700));
        const d0: f32 = if (qh0 & 0x08 != 0) -iq1s_delta else iq1s_delta;
        const d1: f32 = if (qh0 & 0x80 != 0) -iq1s_delta else iq1s_delta;
        const d2: f32 = if (qh1 & 0x08 != 0) -iq1s_delta else iq1s_delta;
        const d3: f32 = if (qh1 & 0x80 != 0) -iq1s_delta else iq1s_delta;
        const g0: [8]i8 = @bitCast(tables.iq1s_grid[idx0]);
        const g1: [8]i8 = @bitCast(tables.iq1s_grid[idx1]);
        const g2: [8]i8 = @bitCast(tables.iq1s_grid[idx2]);
        const g3: [8]i8 = @bitCast(tables.iq1s_grid[idx3]);
        for (0..8) |j| {
            out[y + j] = dl1 * (@as(f32, @floatFromInt(g0[j])) + d0);
            out[y + 8 + j] = dl1 * (@as(f32, @floatFromInt(g1[j])) + d1);
            out[y + 16 + j] = dl2 * (@as(f32, @floatFromInt(g2[j])) + d2);
            out[y + 24 + j] = dl2 * (@as(f32, @floatFromInt(g3[j])) + d3);
        }
        y += 32;
    }
}

// ---------------------------------------------------------------------------
// kernels: SIMD (currently forwards to _ref) + scalar _ref twin
// ---------------------------------------------------------------------------

/// True SIMD IQ1_S: scalar codebook dequant (both sides) + 8-wide `@Vector`
/// FMA accumulate with horizontal reduction (pattern as iq4_xs_q8_k).  Math is
/// element-identical to `iq1_s_q8_k_ref`.
pub fn iq1_s_q8_k(n: usize, vx: [*]const u8, vy: [*]const u8) callconv(.c) f32 {
    const rb: usize = 50;
    const V8f32 = @Vector(8, f32);
    var w: [256]f32 = undefined;
    var a: [256]f32 = undefined;
    var accv: V8f32 = @splat(0);
    for (0 .. n / 256) |b| {
        deqIQ1S(vx + b * rb, &w);
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

pub fn iq1_s_q8_k_ref(n: usize, vx: [*]const u8, vy: [*]const u8) callconv(.c) f32 {
    const rb: usize = 50;
    var w: [256]f32 = undefined;
    var a: [256]f32 = undefined;
    var acc: f32 = 0;
    for (0 .. n / 256) |b| {
        deqIQ1S(vx + b * rb, &w);
        util.deqQ8K(vy + b * 292, &a);
        for (0..256) |i| acc += w[i] * a[i];
    }
    return acc;
}

/// True SIMD IQ1_M: scalar codebook dequant (both sides) + 8-wide `@Vector`
/// FMA accumulate with horizontal reduction (pattern as iq4_xs_q8_k).  Math is
/// element-identical to `iq1_m_q8_k_ref`.
pub fn iq1_m_q8_k(n: usize, vx: [*]const u8, vy: [*]const u8) callconv(.c) f32 {
    const rb: usize = 56;
    const V8f32 = @Vector(8, f32);
    var w: [256]f32 = undefined;
    var a: [256]f32 = undefined;
    var accv: V8f32 = @splat(0);
    for (0 .. n / 256) |b| {
        deqIQ1M(vx + b * rb, &w);
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

pub fn iq1_m_q8_k_ref(n: usize, vx: [*]const u8, vy: [*]const u8) callconv(.c) f32 {
    const rb: usize = 56;
    var w: [256]f32 = undefined;
    var a: [256]f32 = undefined;
    var acc: f32 = 0;
    for (0 .. n / 256) |b| {
        deqIQ1M(vx + b * rb, &w);
        util.deqQ8K(vy + b * 292, &a);
        for (0..256) |i| acc += w[i] * a[i];
    }
    return acc;
}
