//! Activation quantizers: f32 -> Q8_0 / Q8_K blocks.
//!
//! Ported of `quantize_row_q8_0_ref` / `quantize_row_q8_K_ref`
//! (ggml/src/ggml-cpu/quants.c) and of the engine's inline quantizer
//! (src/engine.zig matvecQ8_0).  Semantics are kept identical to
//! `quantizeQ8_0` in src/quant_dequant.zig:
//!
//!   scale = amax / 127 (f32), stored as f16 in Q8_0, as f32 in Q8_K
//!   q     = round(x / scale), clamped to [-128, 127]
//!   amax == 0  => scale = 1, all q = 0   (engine convention)

const std = @import("std");
const util = @import("util.zig");
const tier = @import("tier.zig");

const V32f32 = @Vector(32, f32);
const V32i8 = @Vector(32, i8);
const V32i32 = @Vector(32, i32);

const q8k_bytes = 292;

/// Q8_0: 32 elements per block, 2-byte f16 scale + 32 x i8 = 34 bytes.
pub fn quantizeQ8_0(n: usize, x: [*]const f32, out: [*]u8) callconv(.c) void {
    std.debug.assert(n % 32 == 0);
    var i: usize = 0;
    var o: usize = 0;
    while (i < n) : ({
        i += 32;
        o += 34;
    }) {
        // Runtime-indexed vector element access is not allowed in Zig, so the
        // block is materialized as an array; LLVM keeps the loads vectorized.
        const va: [32]f32 = @bitCast(util.loadF32(32, x + i));
        var amax: f32 = 0;
        for (va) |a| {
            const m = @abs(a);
            if (m > amax) amax = m;
        }
        const scale: f32 = if (amax > 0) amax / 127.0 else 1.0;
        const inv: f32 = if (amax > 0) 1.0 / scale else 0.0;
        std.mem.writeInt(u16, out[o..][0..2], util.f32ToF16Bits(scale), .little);
        var k: usize = 0;
        while (k < 32) : (k += 1) {
            const q: i32 = @intFromFloat(@round(va[k] * inv));
            out[o + 2 + k] = @bitCast(@as(i8, @intCast(std.math.clamp(q, -128, 127))));
        }
    }
}

/// Q8_K: 256 elements per block.  Layout (ggml-common.h block_q8_K, 292 bytes):
///   0..3    d (f32)
///   4..259  qs[256] (i8)
///   260..291 bsums[16] (i16) -- sum of qs in each group of 16
/// bsums is what the K-quant min-terms read (q4_K/q5_K/q6_K), so it must be
/// filled here; a 260-byte block is not a valid Q8_K.
pub fn quantizeQ8_K(n: usize, x: [*]const f32, out: [*]u8) callconv(.c) void {
    std.debug.assert(n % 256 == 0);
    var i: usize = 0;
    var o: usize = 0;
    while (i < n) : ({
        i += 256;
        o += q8k_bytes;
    }) {
        var amax: f32 = 0;
        var j: usize = 0;
        while (j < 256) : (j += 1) {
            const a = @abs(x[i + j]);
            if (a > amax) amax = a;
        }
        const scale: f32 = if (amax > 0) amax / 127.0 else 1.0;
        const inv: f32 = if (amax > 0) 1.0 / scale else 0.0;
        std.mem.writeInt(u32, out[o..][0..4], @bitCast(scale), .little);
        var gsum: [16]i16 = undefined;
        @memset(&gsum, 0);
        var k: usize = 0;
        while (k < 256) : (k += 1) {
            const r: f32 = @round(x[i + k] * inv);
            const ri: i32 = @intFromFloat(r);
            const qi: i32 = std.math.clamp(ri, -128, 127);
            const q: i8 = @bitCast(@as(i8, @intCast(qi)));
            out[o + 4 + k] = @bitCast(q);
            gsum[k / 16] += @as(i16, q);
        }
        for (0..16) |g| std.mem.writeInt(i16, out[o + 260 + g * 2 ..][0..2], gsum[g], .little);
    }
}
