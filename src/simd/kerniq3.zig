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

// ---------------------------------------------------------------------------
// AVX-512 VNNI paths
// ---------------------------------------------------------------------------
// Both IQ1 kernels dequant to `dl * (grid[j] + delta)` per 8-element group, so
// the dot with the Q8_K qs bytes splits into two integer sums per group:
//   dl * sum(grid[j] * qs[j])  +  dl * delta * sum(qs[j])
// grid bytes are built into a 256-byte buffer by scalar table reads (the grid
// has 2048 entries -- no gather is available in Zig inline asm), then each
// 64-byte chunk is fed to vpdpbusd with the +128 bias removed once per chunk
// (same correction as kerniq.zig's IQ2/IQ3/IQ4_XS paths).  Per i32 lane the
// scale coefficients are gathered into f32 vectors and applied once.

const Q1Z = @Vector(64, i8);
const Q1U = @Vector(64, u8);
const Q1I32 = @Vector(16, i32);
const Q1F32 = @Vector(16, f32);

inline fn q1load64(p: [*]const u8) Q1Z {
    return asm ("vmovdqu8 (%[src]), %[dst]" : [dst] "=v" (-> Q1Z) : [src] "r" (p));
}
inline fn q1addb(a: Q1U, b: Q1U) Q1U {
    return asm ("vpaddb %[b], %[a], %[dst]" : [dst] "=v" (-> Q1U) : [a] "v" (a), [b] "v" (b));
}
inline fn q1dpbusd0(a: Q1U, b: Q1Z) Q1I32 {
    var out: Q1I32 = @splat(0);
    asm ("vpdpbusd %[b], %[a], %[out]" : [out] "+v" (out) : [a] "v" (a), [b] "v" (b));
    return out;
}
inline fn q1paddd(a: Q1I32, b: Q1I32) Q1I32 {
    return asm ("vpaddd %[b], %[a], %[dst]" : [dst] "=v" (-> Q1I32) : [a] "v" (a), [b] "v" (b));
}
inline fn q1psubd(a: Q1I32, b: Q1I32) Q1I32 {
    return asm ("vpsubd %[b], %[a], %[dst]" : [dst] "=v" (-> Q1I32) : [a] "v" (a), [b] "v" (b));
}
inline fn q1pslld7(a: Q1I32) Q1I32 {
    return asm ("vpslld $7, %[a], %[dst]" : [dst] "=v" (-> Q1I32) : [a] "v" (a));
}

const q1_zero: Q1I32 = @splat(0);
const q1_ones: Q1U = @bitCast(@as(Q1U, @splat(@as(u8, 1))));
const q1_128: Q1U = @splat(@as(u8, 128));

/// IQ1_S: 50 B per 256 (d@0, qs[32]@2, qh[8]u16@34).  Per ib: dl = d*(2*((qh16>>12)&7)+1),
/// delta = +-0.125 from bit 0x8000, and 4 grid entries at 32*ib + 8*l.
fn iq1_s_q8_k_vnni_impl(n: usize, vx: [*]const u8, vy: [*]const u8) callconv(.c) f32 {
    if (comptime tier.isAvx512Vnni and @import("builtin").mode != .debug) {
        const nb = n / 256;
        var acc_out: Q1F32 = @splat(0);
        for (0..nb) |b| {
            const x = vx + b * 50;
            const y = vy + b * 292;
            const d = util.f16At(x, 0);
            var gb: [256]u8 = undefined;
            var dls: [8]f32 = undefined;
            var dlt: [8]f32 = undefined;
            for (0..8) |ib| {
                const qh16: u16 = util.u16At(x + 34 + 2 * ib);
                dls[ib] = d * @as(f32, @floatFromInt(2 * ((qh16 >> 12) & 7) + 1));
                dlt[ib] = if (qh16 & 0x8000 != 0) -iq1s_delta else iq1s_delta;
                for (0..4) |l| {
                    const idx: usize = x[2 + 4 * ib + l] | (@as(usize, (qh16 >> @as(u4, @intCast(3 * l))) & 7) << 8);
                    @as(*align(1) u64, @ptrCast(&gb[32 * ib + 8 * l])).* = tables.iq1s_grid[idx];
                }
            }
            const gptr: [*]const u8 = @as([*]const u8, @ptrCast(&gb));
            for (0..4) |zi| {
                const g = q1load64(gptr + 64 * zi);
                const wu = q1addb(@as(Q1U, @bitCast(g)), q1_128);
                const q = q1load64(y + 4 + 64 * zi);
                const acc = q1dpbusd0(wu, q);
                const gsum = q1dpbusd0(q1_ones, q);
                const dot = q1psubd(acc, q1pslld7(gsum));
                var cv: [16]f32 = undefined;
                var cvd: [16]f32 = undefined;
                for (0..16) |k| {
                    // lane k sums bytes 4k..4k+3 of the chunk => sub-block
                    // ib = 2*zi + k/8, which owns one dl and one delta.
                    const ib = 2 * zi + k / 8;
                    cv[k] = dls[ib];
                    cvd[k] = dls[ib] * dlt[ib];
                }
                acc_out = acc_out + @as(Q1F32, @bitCast(cv)) * @as(Q1F32, @floatFromInt(dot)) +
                    @as(Q1F32, @bitCast(cvd)) * @as(Q1F32, @floatFromInt(gsum));
            }
            const dy: Q1F32 = @splat(@as(f32, @bitCast(util.u32At(y))));
            acc_out = dy * acc_out;
        }
        return @reduce(.Add, acc_out);
    }
    return 0;
}

pub fn iq1_s_q8_k(n: usize, vx: [*]const u8, vy: [*]const u8) callconv(.c) f32 {
    if (comptime tier.isAvx512Vnni and @import("builtin").mode != .debug) {
        return iq1_s_q8_k_vnni_impl(n, vx, vy);
    }
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

/// IQ1_M: 56 B per 256 (qs[32]@0, qh[16]@32, scales[8]@48, d packed from them).
/// Per ib: two scales (dl1 for groups 0,1 and dl2 for groups 2,3) and four
/// deltas (two from qh0, two from qh1); grid entries at 32*ib + 8*l.
fn iq1_m_q8_k_vnni_impl(n: usize, vx: [*]const u8, vy: [*]const u8) callconv(.c) f32 {
    if (comptime tier.isAvx512Vnni and @import("builtin").mode != .debug) {
        const nb = n / 256;
        var acc_out: Q1F32 = @splat(0);
        for (0..nb) |b| {
            const x = vx + b * 56;
            const y = vy + b * 292;
            const sc0: u16 = util.u16At(x + 48);
            const sc1: u16 = util.u16At(x + 50);
            const sc2: u16 = util.u16At(x + 52);
            const sc3: u16 = util.u16At(x + 54);
            const scale_u16: u16 = (sc0 >> 12) | ((sc1 >> 8) & 0x00f0) | ((sc2 >> 4) & 0x0f00) | (sc3 & 0xf000);
            const d: f32 = @as(f32, @floatCast(@as(f16, @bitCast(scale_u16))));
            const sc_u16: [4]u16 = .{ sc0, sc1, sc2, sc3 };
            var gb: [256]u8 = undefined;
            var dls1: [8]f32 = undefined;
            var dls2: [8]f32 = undefined;
            var dlt: [8][4]f32 = undefined;
            for (0..8) |ib| {
                const sh: u16 = sc_u16[ib / 2];
                dls1[ib] = d * @as(f32, @floatFromInt(2 * ((sh >> @as(u4, @intCast(6 * (ib % 2)))) & 7) + 1));
                dls2[ib] = d * @as(f32, @floatFromInt(2 * ((sh >> @as(u4, @intCast(6 * (ib % 2) + 3))) & 7) + 1));
                const qh0: u16 = x[32 + 2 * ib];
                const qh1: u16 = x[32 + 2 * ib + 1];
                dlt[ib][0] = if (qh0 & 0x08 != 0) -iq1s_delta else iq1s_delta;
                dlt[ib][1] = if (qh0 & 0x80 != 0) -iq1s_delta else iq1s_delta;
                dlt[ib][2] = if (qh1 & 0x08 != 0) -iq1s_delta else iq1s_delta;
                dlt[ib][3] = if (qh1 & 0x80 != 0) -iq1s_delta else iq1s_delta;
                const idx0: usize = x[4 * ib + 0] | @as(usize, ((qh0 << 8) & 0x700));
                const idx1: usize = x[4 * ib + 1] | @as(usize, ((qh0 << 4) & 0x700));
                const idx2: usize = x[4 * ib + 2] | @as(usize, ((qh1 << 8) & 0x700));
                const idx3: usize = x[4 * ib + 3] | @as(usize, ((qh1 << 4) & 0x700));
                @as(*align(1) u64, @ptrCast(&gb[32 * ib + 0])).* = tables.iq1s_grid[idx0];
                @as(*align(1) u64, @ptrCast(&gb[32 * ib + 8])).* = tables.iq1s_grid[idx1];
                @as(*align(1) u64, @ptrCast(&gb[32 * ib + 16])).* = tables.iq1s_grid[idx2];
                @as(*align(1) u64, @ptrCast(&gb[32 * ib + 24])).* = tables.iq1s_grid[idx3];
            }
            const gptr: [*]const u8 = @as([*]const u8, @ptrCast(&gb));
            for (0..4) |zi| {
                const g = q1load64(gptr + 64 * zi);
                const wu = q1addb(@as(Q1U, @bitCast(g)), q1_128);
                const q = q1load64(y + 4 + 64 * zi);
                const acc = q1dpbusd0(wu, q);
                const gsum = q1dpbusd0(q1_ones, q);
                const dot = q1psubd(acc, q1pslld7(gsum));
                var cv: [16]f32 = undefined;
                var cvd: [16]f32 = undefined;
                for (0..16) |k| {
                    // lane k sums bytes 4k..4k+3 => group l = (k % 8) / 2 of
                    // sub-block ib = 2*zi + k/8; groups 0,1 use dl1, 2,3 use dl2.
                    const ib = 2 * zi + k / 8;
                    const l = (k % 8) / 2;
                    const sc = if (l < 2) dls1[ib] else dls2[ib];
                    cv[k] = sc;
                    cvd[k] = sc * dlt[ib][l];
                }
                acc_out = acc_out + @as(Q1F32, @bitCast(cv)) * @as(Q1F32, @floatFromInt(dot)) +
                    @as(Q1F32, @bitCast(cvd)) * @as(Q1F32, @floatFromInt(gsum));
            }
            const dy: Q1F32 = @splat(@as(f32, @bitCast(util.u32At(y))));
            acc_out = dy * acc_out;
        }
        return @reduce(.Add, acc_out);
    }
    return 0;
}

pub fn iq1_m_q8_k(n: usize, vx: [*]const u8, vy: [*]const u8) callconv(.c) f32 {
    if (comptime tier.isAvx512Vnni and @import("builtin").mode != .debug) {
        return iq1_m_q8_k_vnni_impl(n, vx, vy);
    }
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
