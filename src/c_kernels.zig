//! Zig wrapper for C IQ quantization kernels.
//!
//! This module provides a Zig interface to the C-optimized SIMD kernels
//! ported from llama.cpp. The C implementation uses AVX2 intrinsics
//! for maximum performance on x86-64 CPUs.

const std = @import("std");

// C function declarations
extern fn cpu_has_avx2() c_int;
extern fn cpu_has_avx512f() c_int;
extern fn cpu_has_avx512_vnni() c_int;
extern fn cpu_get_features_string() [*:0]const u8;

// 256-element IQ dot-product kernels (Q8_K activation)
extern fn iq2_xxs_q8_k(n: c_int, vx: ?*const anyopaque, vy: ?*const anyopaque) f32;
extern fn iq2_xs_q8_k(n: c_int, vx: ?*const anyopaque, vy: ?*const anyopaque) f32;
extern fn iq2_s_q8_k(n: c_int, vx: ?*const anyopaque, vy: ?*const anyopaque) f32;
extern fn iq3_xxs_q8_k(n: c_int, vx: ?*const anyopaque, vy: ?*const anyopaque) f32;
extern fn iq3_s_q8_k(n: c_int, vx: ?*const anyopaque, vy: ?*const anyopaque) f32;
extern fn iq1_s_q8_k(n: c_int, vx: ?*const anyopaque, vy: ?*const anyopaque) f32;
extern fn iq1_m_q8_k(n: c_int, vx: ?*const anyopaque, vy: ?*const anyopaque) f32;
extern fn iq4_xs_q8_k(n: c_int, vx: ?*const anyopaque, vy: ?*const anyopaque) f32;

// 256-element K-quant dot-product kernels (Q8_K activation)
extern fn q2_k_q8_k(n: c_int, vx: ?*const anyopaque, vy: ?*const anyopaque) f32;
extern fn q3_k_q8_k(n: c_int, vx: ?*const anyopaque, vy: ?*const anyopaque) f32;
extern fn q4_k_q8_k(n: c_int, vx: ?*const anyopaque, vy: ?*const anyopaque) f32;
extern fn q5_k_q8_k(n: c_int, vx: ?*const anyopaque, vy: ?*const anyopaque) f32;
extern fn q6_k_q8_k(n: c_int, vx: ?*const anyopaque, vy: ?*const anyopaque) f32;

// 32-element dot-product kernels
extern fn q4_0_q8_0(n: c_int, vx: ?*const anyopaque, vy: ?*const anyopaque) f32;
extern fn q4_1_q8_0(n: c_int, vx: ?*const anyopaque, vy: ?*const anyopaque) f32;
extern fn iq4_nl_q8_0(n: c_int, vx: ?*const anyopaque, vy: ?*const anyopaque) f32;
extern fn q8_0_q8_0(n: c_int, vx: ?*const anyopaque, vy: ?*const anyopaque) f32;
extern fn q5_0_q8_0(n: c_int, vx: ?*const anyopaque, vy: ?*const anyopaque) f32;
extern fn q5_1_q8_0(n: c_int, vx: ?*const anyopaque, vy: ?*const anyopaque) f32;
extern fn q1_0_q8_0(n: c_int, vx: ?*const anyopaque, vy: ?*const anyopaque) f32;
extern fn q2_0_q8_0(n: c_int, vx: ?*const anyopaque, vy: ?*const anyopaque) f32;
extern fn mxfp4_q8_0(n: c_int, vx: ?*const anyopaque, vy: ?*const anyopaque) f32;
extern fn nvfp4_q8_0(n: c_int, vx: ?*const anyopaque, vy: ?*const anyopaque) f32;
extern fn quantize_q8_0(n: c_int, x: ?*const anyopaque, out: ?*anyopaque) void;
extern fn quantize_q8_K(n: c_int, x: ?*const anyopaque, out: ?*anyopaque) void;

/// CPU feature detection results
pub const CpuFeatures = struct {
    avx2: bool,
    avx512f: bool,
    avx512_vnni: bool,

    pub fn detect() CpuFeatures {
        return .{
            .avx2 = cpu_has_avx2() != 0,
            .avx512f = cpu_has_avx512f() != 0,
            .avx512_vnni = cpu_has_avx512_vnni() != 0,
        };
    }

    pub fn format(_: CpuFeatures, writer: anytype) !void {
        const str = cpu_get_features_string();
        try writer.print("{s}", .{std.mem.span(str)});
    }
};

/// Compute dot product of IQ2_XXS quantized vector with Q8_K quantized vector.
///
/// Parameters:
///   n: Number of elements (must be multiple of 256)
///   vx: Pointer to IQ2_XXS quantized data
///   vy: Pointer to Q8_K quantized data
///
/// Returns: The dot product as a float
pub fn dotIq2XxsQ8K(n: usize, vx: [*]const u8, vy: [*]const u8) callconv(.c) f32 {
    return iq2_xxs_q8_k(@intCast(n), vx, vy);
}

/// Compute dot product of IQ2_XS quantized vector with Q8_K quantized vector.
pub fn dotIq2XsQ8K(n: usize, vx: [*]const u8, vy: [*]const u8) callconv(.c) f32 {
    return iq2_xs_q8_k(@intCast(n), vx, vy);
}

/// Compute dot product of IQ2_S quantized vector with Q8_K quantized vector.
pub fn dotIq2SQ8K(n: usize, vx: [*]const u8, vy: [*]const u8) callconv(.c) f32 {
    return iq2_s_q8_k(@intCast(n), vx, vy);
}

/// Compute dot product of IQ3_XXS quantized vector with Q8_K quantized vector.
pub fn dotIq3XxsQ8K(n: usize, vx: [*]const u8, vy: [*]const u8) callconv(.c) f32 {
    return iq3_xxs_q8_k(@intCast(n), vx, vy);
}

/// Compute dot product of IQ3_S quantized vector with Q8_K quantized vector.
pub fn dotIq3SQ8K(n: usize, vx: [*]const u8, vy: [*]const u8) callconv(.c) f32 {
    return iq3_s_q8_k(@intCast(n), vx, vy);
}

/// Compute dot product of IQ1_S quantized vector with Q8_K quantized vector.
pub fn dotIq1SQ8K(n: usize, vx: [*]const u8, vy: [*]const u8) callconv(.c) f32 {
    return iq1_s_q8_k(@intCast(n), vx, vy);
}

/// Compute dot product of IQ1_M quantized vector with Q8_K quantized vector.
pub fn dotIq1MQ8K(n: usize, vx: [*]const u8, vy: [*]const u8) callconv(.c) f32 {
    return iq1_m_q8_k(@intCast(n), vx, vy);
}

/// Compute dot product of IQ4_XS quantized vector with Q8_K quantized vector.
pub fn dotIq4XsQ8K(n: usize, vx: [*]const u8, vy: [*]const u8) callconv(.c) f32 {
    return iq4_xs_q8_k(@intCast(n), vx, vy);
}

// 256-element K-quant dot-product kernels (Q8_K activation)

/// Compute dot product of Q2_K quantized vector with Q8_K quantized vector.
pub fn dotQ2KQ8K(n: usize, vx: [*]const u8, vy: [*]const u8) callconv(.c) f32 {
    return q2_k_q8_k(@intCast(n), vx, vy);
}

/// Compute dot product of Q3_K quantized vector with Q8_K quantized vector.
pub fn dotQ3KQ8K(n: usize, vx: [*]const u8, vy: [*]const u8) callconv(.c) f32 {
    return q3_k_q8_k(@intCast(n), vx, vy);
}

/// Compute dot product of Q4_K quantized vector with Q8_K quantized vector.
pub fn dotQ4KQ8K(n: usize, vx: [*]const u8, vy: [*]const u8) callconv(.c) f32 {
    return q4_k_q8_k(@intCast(n), vx, vy);
}

/// Compute dot product of Q5_K quantized vector with Q8_K quantized vector.
pub fn dotQ5KQ8K(n: usize, vx: [*]const u8, vy: [*]const u8) callconv(.c) f32 {
    return q5_k_q8_k(@intCast(n), vx, vy);
}

/// Compute dot product of Q6_K quantized vector with Q8_K quantized vector.
pub fn dotQ6KQ8K(n: usize, vx: [*]const u8, vy: [*]const u8) callconv(.c) f32 {
    return q6_k_q8_k(@intCast(n), vx, vy);
}

// 32-element dot-product kernels

/// Compute dot product of Q4_0 quantized vector with Q8_0 quantized vector.
pub fn dotQ4_0Q8_0(n: usize, vx: [*]const u8, vy: [*]const u8) callconv(.c) f32 {
    return q4_0_q8_0(@intCast(n), vx, vy);
}

/// Compute dot product of Q4_1 quantized vector with Q8_0 quantized vector.
pub fn dotQ4_1Q8_0(n: usize, vx: [*]const u8, vy: [*]const u8) callconv(.c) f32 {
    return q4_1_q8_0(@intCast(n), vx, vy);
}

/// Compute dot product of IQ4_NL quantized vector with Q8_0 quantized vector.
pub fn dotIq4NlQ8_0(n: usize, vx: [*]const u8, vy: [*]const u8) callconv(.c) f32 {
    return iq4_nl_q8_0(@intCast(n), vx, vy);
}

/// Compute dot product of Q8_0 quantized vector with Q8_0 quantized vector.
pub fn dotQ8_0Q8_0(n: usize, vx: [*]const u8, vy: [*]const u8) callconv(.c) f32 {
    return q8_0_q8_0(@intCast(n), vx, vy);
}

/// Compute dot product of Q5_0 quantized vector with Q8_0 quantized vector.
pub fn dotQ5_0Q8_0(n: usize, vx: [*]const u8, vy: [*]const u8) callconv(.c) f32 {
    return q5_0_q8_0(@intCast(n), vx, vy);
}

/// Compute dot product of Q5_1 quantized vector with Q8_0 quantized vector.
pub fn dotQ5_1Q8_0(n: usize, vx: [*]const u8, vy: [*]const u8) callconv(.c) f32 {
    return q5_1_q8_0(@intCast(n), vx, vy);
}

/// Compute dot product of Q1_0 quantized vector with Q8_0 quantized vector.
pub fn dotQ1_0Q8_0(n: usize, vx: [*]const u8, vy: [*]const u8) callconv(.c) f32 {
    return q1_0_q8_0(@intCast(n), vx, vy);
}

/// Compute dot product of Q2_0 quantized vector with Q8_0 quantized vector.
pub fn dotQ2_0Q8_0(n: usize, vx: [*]const u8, vy: [*]const u8) callconv(.c) f32 {
    return q2_0_q8_0(@intCast(n), vx, vy);
}

/// Compute dot product of MXFP4 quantized vector with Q8_0 quantized vector.
pub fn dotMxfp4Q8_0(n: usize, vx: [*]const u8, vy: [*]const u8) callconv(.c) f32 {
    return mxfp4_q8_0(@intCast(n), vx, vy);
}

/// Compute dot product of NVFP4 quantized vector with Q8_0 quantized vector.
pub fn dotNvfp4Q8_0(n: usize, vx: [*]const u8, vy: [*]const u8) callconv(.c) f32 {
    return nvfp4_q8_0(@intCast(n), vx, vy);
}

/// Quantize float array to Q8_0 format.
pub fn quantizeQ8_0(n: usize, x: [*]const f32, out: [*]u8) callconv(.c) void {
    quantize_q8_0(@intCast(n), x, out);
}

/// Quantize float array to Q8_K format.
pub fn quantizeQ8_K(n: usize, x: [*]const f32, out: [*]u8) callconv(.c) void {
    quantize_q8_K(@intCast(n), x, out);
}

// Test the C kernels
test "CPU feature detection" {
    const feats = CpuFeatures.detect();
    std.debug.print("CPU features: AVX2={}, AVX512F={}, AVX512_VNNI={}\n", .{
        feats.avx2, feats.avx512f, feats.avx512_vnni,
    });
}