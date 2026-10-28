//! Test-only scalar oracle for the C kernels (src/kernels/x86/*).
//!
//! The `_ref` bodies below are the pure-scalar twins extracted from the old
//! src/simd/kern32.zig / kern256.zig / kerniq*.zig (commit f0e36ea^), stripped
//! of every ISA tier branch.  The old `X_row_ref` entries were aliases to the
//! SIMD row kernels (which no longer exist); they are rewritten here to forward
//! to the real scalar `X_ref`.
const std = @import("std");
const tables = @import("ggml_tables.zig");
const tables_x = @import("ggml_tables_extra.zig");

const iq2xxs_grid = tables_x.iq2xxs_grid;
const iq2xs_grid = tables_x.iq2xs_grid;
const iq2s_grid = tables_x.iq2s_grid;
const iq1s_grid = tables_x.iq1s_grid;
const iq3s_grid = tables.iq3s_grid;
const iq3xxs_grid = tables.iq3xxs_grid;
const kmask_iq2xs = tables.kmask_iq2xs;
const ksigns_iq2xs = tables.ksigns_iq2xs;

const q8_bytes = 34;
const q8k_bytes = 292;
const qk_k = 256;
const iq1s_delta = 0.125;
const pow3 = [6]u8{ 1, 3, 9, 27, 81, 243 };
const kvalues_iq4nl = [16]i8{ -127, -104, -83, -65, -49, -35, -22, -10, 1, 13, 25, 38, 53, 69, 89, 113 };
const kvalues_fp4 = [16]i8{ 0, 1, 2, 3, 4, 6, 8, 12, 0, -1, -2, -3, -4, -6, -8, -12 };

pub const util = struct {
    pub fn f16At(p: [*]const u8, off: usize) f32 {
        const h: f16 = @bitCast(@as(*align(1) const u16, @ptrCast(p + off)).*);
        return @floatCast(h);
    }
    pub fn u32At(p: [*]const u8) u32 {
        return @as(*align(1) const u32, @ptrCast(p)).*;
    }
    pub fn u16At(p: [*]const u8) u16 {
        return @as(*align(1) const u16, @ptrCast(p)).*;
    }
    pub fn deqQ8K(p: [*]const u8, out: *[256]f32) void {
        const d: f32 = @bitCast(util.u32At(p));
        for (0..256) |i| out[i] = d * @as(f32, @floatFromInt(@as(i8, @bitCast(p[4 + i]))));
    }
};

inline fn qi8(p: [*]const u8, off: usize) i32 {
    return @as(i32, @as(i8, @bitCast(p[off])));
}
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
inline fn e8m0ToHalf(e: u8) f32 {
    const bits: u32 = if (e == 0) 0x00200000 else @as(u32, e - 1) << 23;
    return @bitCast(bits);
}
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

fn deqIQ2XXS(row: [*]const u8, out: *[256]f32) void {
    const d = util.f16At(row, 0);
    var y: usize = 0;
    for (0..8) |ib32| {
        const aux1: u32 = util.u32At(row + 2 + 8 * ib32 + 4);
        const db = d * (0.5 + @as(f32, @floatFromInt(aux1 >> 28))) * 0.25;
        for (0..4) |l| {
            const grid: [8]u8 = @bitCast(iq2xxs_grid[row[2 + 8 * ib32 + l]]);
            const signs: u8 = ksigns_iq2xs[(aux1 >> @as(u5, @intCast(7 * l))) & 127];
            for (0..8) |j| {
                out[y + j] = db * @as(f32, @floatFromInt(grid[j])) * @as(f32, if (signs & kmask_iq2xs[j] != 0) -1.0 else 1.0);
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
            const grid: [8]u8 = @bitCast(iq2xs_grid[q16 & 511]);
            const signs: u8 = ksigns_iq2xs[q16 >> 9];
            const db: f32 = if (l / 2 == 0) db0 else db1;
            for (0..8) |j| {
                out[y + j] = db * @as(f32, @floatFromInt(grid[j])) * @as(f32, if (signs & kmask_iq2xs[j] != 0) -1.0 else 1.0);
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
            const signs: u8 = ksigns_iq2xs[sidx];
            const g1: u32 = iq3xxs_grid[row[2 + 8 * ib32 + 2 * l]];
            const g2: u32 = iq3xxs_grid[row[2 + 8 * ib32 + 2 * l + 1]];
            const gb1: [4]u8 = @bitCast(g1);
            const gb2: [4]u8 = @bitCast(g2);
            for (0..4) |j| {
                out[y + j] = db * @as(f32, @floatFromInt(gb1[j])) * @as(f32, if (signs & kmask_iq2xs[j] != 0) -1.0 else 1.0);
                out[y + 4 + j] = db * @as(f32, @floatFromInt(gb2[j])) * @as(f32, if (signs & kmask_iq2xs[4 + j] != 0) -1.0 else 1.0);
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
            const grid: [8]u8 = @bitCast(iq2s_grid[idx]);
            const sign_byte: u8 = row[34 + 4 * ib32 + l];
            for (0..8) |j| {
                out[y + j] = dl * @as(f32, @floatFromInt(grid[j])) * @as(f32, if (sign_byte & kmask_iq2xs[j] != 0) -1.0 else 1.0);
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
            const g1: u32 = iq3s_grid[row[2 + 16 * p + 2 * l] | bit1];
            const g2: u32 = iq3s_grid[row[2 + 16 * p + 2 * l + 1] | bit2];
            const sg: u8 = row[74 + 8 * p + l];
            const gb1: [4]u8 = @bitCast(g1);
            const gb2: [4]u8 = @bitCast(g2);
            for (0..4) |j| {
                out[y + l * 8 + j] = db1 * @as(f32, @floatFromInt(@as(i32, gb1[j]))) * @as(f32, if (sg & kmask_iq2xs[j] != 0) -1.0 else 1.0);
                out[y + l * 8 + 4 + j] = db1 * @as(f32, @floatFromInt(@as(i32, gb2[j]))) * @as(f32, if (sg & kmask_iq2xs[4 + j] != 0) -1.0 else 1.0);
            }
        }
        const qh1: u16 = row[66 + 2 * p + 1];
        for (0..4) |l| {
            const bit1: u16 = (qh1 << @as(u4, @intCast(8 - 2 * l))) & 0x100;
            const bit2: u16 = (qh1 << @as(u4, @intCast(7 - 2 * l))) & 0x100;
            const g1: u32 = iq3s_grid[row[2 + 16 * p + 8 + 2 * l] | bit1];
            const g2: u32 = iq3s_grid[row[2 + 16 * p + 8 + 2 * l + 1] | bit2];
            const sg: u8 = row[74 + 8 * p + 4 + l];
            const gb1: [4]u8 = @bitCast(g1);
            const gb2: [4]u8 = @bitCast(g2);
            for (0..4) |j| {
                out[y + 32 + l * 8 + j] = db2 * @as(f32, @floatFromInt(@as(i32, gb1[j]))) * @as(f32, if (sg & kmask_iq2xs[j] != 0) -1.0 else 1.0);
                out[y + 32 + l * 8 + 4 + j] = db2 * @as(f32, @floatFromInt(@as(i32, gb2[j]))) * @as(f32, if (sg & kmask_iq2xs[4 + j] != 0) -1.0 else 1.0);
            }
        }
        y += 64;
    }
}
fn deqIQ1S(row: [*]const u8, out: *[256]f32) void {
    const d = util.f16At(row, 0);
    var y: usize = 0;
    for (0..8) |ib| {
        const qh16: u16 = util.u16At(row + 34 + 2 * ib);
        const dl = d * @as(f32, @floatFromInt(2 * ((qh16 >> 12) & 7) + 1));
        const delta: f32 = if (qh16 & 0x8000 != 0) -iq1s_delta else iq1s_delta;
        for (0..4) |l| {
            const idx: u16 = row[2 + 4 * ib + l] | @as(u16, ((qh16 >> @as(u4, @intCast(3 * l))) & 7) << 8);
            const grid: [8]i8 = @bitCast(iq1s_grid[idx]);
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
        const g0: [8]i8 = @bitCast(iq1s_grid[idx0]);
        const g1: [8]i8 = @bitCast(iq1s_grid[idx1]);
        const g2: [8]i8 = @bitCast(iq1s_grid[idx2]);
        const g3: [8]i8 = @bitCast(iq1s_grid[idx3]);
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
// kern32 refs
// ---------------------------------------------------------------------------
pub const dot32 = struct {
    pub fn q8_0_q8_0_ref(n: usize, vx: [*]const u8, vy: [*]const u8) callconv(.c) f32 {
        const nb = n / 32;
        var sumf: f32 = 0;
        for (0..nb) |ib| {
            const xb = vx + ib * 34;
            const yb = vy + ib * 34;
            const dx = util.f16At(xb, 0);
            const dy = util.f16At(yb, 0);
            var sumi: i32 = 0;
            for (0..32) |j| sumi += @as(i32, @as(i8, @bitCast(xb[2 + j]))) * @as(i32, @as(i8, @bitCast(yb[2 + j])));
            sumf += dx * dy * @as(f32, @floatFromInt(sumi));
        }
        return sumf;
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
    // q5_0 block layout (ggml-common.h block_q5_0 / ggml-quants.c
    // dequantize_row_q5_0): f16 d @0, uint8 qh[4] @2, uint8 qs[16] @6.
    // qh is 4 bytes (not a u32 @2 with qs already at 6) and qs follows it.
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
                const b = x[6 + j];  // qs[16] @6: low nibble -> elem j, high nibble -> elem j+16
                const x0: i8 = @bitCast(@as(u8, b & 0x0F) | xh0);
                const x1: i8 = @bitCast(@as(u8, b >> 4) | xh1);
                sumi += (@as(i32, x0) - 16) * qi8(y, 2 + j);
                sumi += (@as(i32, x1) - 16) * qi8(y, 2 + 16 + j);
            }
            sumf += dx * dy * @as(f32, @floatFromInt(sumi));
        }
        return sumf;
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
    pub fn q1_0_q8_0_ref(n: usize, vx: [*]const u8, vy: [*]const u8) callconv(.c) f32 {
        const nb = n / 128;
        var sumf: f32 = 0;
        var ib: usize = 0;
        while (ib < nb) : (ib += 1) {
            const x = vx + ib * 18;
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
    // q2_0 block layout (ggml-common.h block_q2_0, QK2_0 = 64):
    // f16 d @0, uint8 qs[16] @2 (18 bytes/block -- QK2_0/4 = 16 qs bytes, NOT 8).
    // Element j -> byte j/4, bits (j%4)*2; value = code - 1
    // (00=-1, 01=0, 10=+1, 11=+2), see ggml dequantize_row_q2_0.
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
                for (0..32) |j| {
                    const byte = x[2 + k * 8 + j / 4];
                    const q = @as(i32, (byte >> @intCast((j % 4) * 2)) & 3) - 1;
                    blk += q * qi8(y, 2 + j);
                }
                sumi += d1 * @as(f32, @floatFromInt(blk));
            }
            sumf += d0 * sumi;
        }
        return sumf;
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
    // row aliases: the old *_row_ref were SIMD aliases -> point at the real scalar ref
    pub fn q8_0_q8_0_row_ref(n: usize, vx: [*]const u8, vy: [*]const u8) callconv(.c) f32 {
        return q8_0_q8_0_ref(n, vx, vy);
    }
    pub fn q4_0_q8_0_row_ref(n: usize, vx: [*]const u8, vy: [*]const u8) callconv(.c) f32 {
        return q4_0_q8_0_ref(n, vx, vy);
    }
    pub fn q4_1_q8_0_row_ref(n: usize, vx: [*]const u8, vy: [*]const u8) callconv(.c) f32 {
        return q4_1_q8_0_ref(n, vx, vy);
    }
    pub fn q5_0_q8_0_row_ref(n: usize, vx: [*]const u8, vy: [*]const u8) callconv(.c) f32 {
        return q5_0_q8_0_ref(n, vx, vy);
    }
    pub fn q5_1_q8_0_row_ref(n: usize, vx: [*]const u8, vy: [*]const u8) callconv(.c) f32 {
        return q5_1_q8_0_ref(n, vx, vy);
    }
    pub fn q1_0_q8_0_row_ref(n: usize, vx: [*]const u8, vy: [*]const u8) callconv(.c) f32 {
        return q1_0_q8_0_ref(n, vx, vy);
    }
    pub fn q2_0_q8_0_row_ref(n: usize, vx: [*]const u8, vy: [*]const u8) callconv(.c) f32 {
        return q2_0_q8_0_ref(n, vx, vy);
    }
    pub fn iq4_nl_q8_0_row_ref(n: usize, vx: [*]const u8, vy: [*]const u8) callconv(.c) f32 {
        return iq4_nl_q8_0_ref(n, vx, vy);
    }
    pub fn mxfp4_q8_0_row_ref(n: usize, vx: [*]const u8, vy: [*]const u8) callconv(.c) f32 {
        return mxfp4_q8_0_ref(n, vx, vy);
    }
    pub fn nvfp4_q8_0_row_ref(n: usize, vx: [*]const u8, vy: [*]const u8) callconv(.c) f32 {
        return nvfp4_q8_0_ref(n, vx, vy);
    }
};

// ---------------------------------------------------------------------------
// kern256 refs
// ---------------------------------------------------------------------------
pub const dot256 = struct {
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
    // row aliases (see note at dot32)
    pub fn q2_k_q8_k_row_ref(n: usize, vx: [*]const u8, vy: [*]const u8) callconv(.c) f32 {
        return q2_k_q8_k_ref(n, vx, vy);
    }
    pub fn q3_k_q8_k_row_ref(n: usize, vx: [*]const u8, vy: [*]const u8) callconv(.c) f32 {
        return q3_k_q8_k_ref(n, vx, vy);
    }
    pub fn q4_k_q8_k_row_ref(n: usize, vx: [*]const u8, vy: [*]const u8) callconv(.c) f32 {
        return q4_k_q8_k_ref(n, vx, vy);
    }
    pub fn q5_k_q8_k_row_ref(n: usize, vx: [*]const u8, vy: [*]const u8) callconv(.c) f32 {
        return q5_k_q8_k_ref(n, vx, vy);
    }
    pub fn q6_k_q8_k_row_ref(n: usize, vx: [*]const u8, vy: [*]const u8) callconv(.c) f32 {
        return q6_k_q8_k_ref(n, vx, vy);
    }
    pub fn tq1_0_q8_k_row_ref(n: usize, vx: [*]const u8, vy: [*]const u8) callconv(.c) f32 {
        return tq1_0_q8_k_ref(n, vx, vy);
    }
    pub fn tq2_0_q8_k_row_ref(n: usize, vx: [*]const u8, vy: [*]const u8) callconv(.c) f32 {
        return tq2_0_q8_k_ref(n, vx, vy);
    }
};

// ---------------------------------------------------------------------------
// kerniq refs
// ---------------------------------------------------------------------------
pub const dotiq = struct {
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
    pub fn iq2_xxs_q8_k_row_ref(n: usize, vx: [*]const u8, vy: [*]const u8) callconv(.c) f32 {
        return iq2_xxs_q8_k_ref(n, vx, vy);
    }
    pub fn iq2_xs_q8_k_row_ref(n: usize, vx: [*]const u8, vy: [*]const u8) callconv(.c) f32 {
        return iq2_xs_q8_k_ref(n, vx, vy);
    }
    pub fn iq3_xxs_q8_k_row_ref(n: usize, vx: [*]const u8, vy: [*]const u8) callconv(.c) f32 {
        return iq3_xxs_q8_k_ref(n, vx, vy);
    }
    pub fn iq4_xs_q8_k_row_ref(n: usize, vx: [*]const u8, vy: [*]const u8) callconv(.c) f32 {
        return iq4_xs_q8_k_ref(n, vx, vy);
    }
};

// ---------------------------------------------------------------------------
// kerniq2 refs
// ---------------------------------------------------------------------------
pub const dotiq2 = struct {
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
    pub fn iq2_s_q8_k_row_ref(n: usize, vx: [*]const u8, vy: [*]const u8) callconv(.c) f32 {
        return iq2_s_q8_k_ref(n, vx, vy);
    }
    pub fn iq3_s_q8_k_row_ref(n: usize, vx: [*]const u8, vy: [*]const u8) callconv(.c) f32 {
        return iq3_s_q8_k_ref(n, vx, vy);
    }
};

// ---------------------------------------------------------------------------
// kerniq3 refs
// ---------------------------------------------------------------------------
pub const dotiq3 = struct {
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
    pub fn iq1_s_q8_k_row_ref(n: usize, vx: [*]const u8, vy: [*]const u8) callconv(.c) f32 {
        return iq1_s_q8_k_ref(n, vx, vy);
    }
    pub fn iq1_m_q8_k_row_ref(n: usize, vx: [*]const u8, vy: [*]const u8) callconv(.c) f32 {
        return iq1_m_q8_k_ref(n, vx, vy);
    }
};
