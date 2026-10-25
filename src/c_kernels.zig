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

extern fn iq2_xxs_q8_k(n: c_int, vx: ?*const anyopaque, vy: ?*const anyopaque) f32;
extern fn iq2_xs_q8_k(n: c_int, vx: ?*const anyopaque, vy: ?*const anyopaque) f32;
extern fn iq3_xxs_q8_k(n: c_int, vx: ?*const anyopaque, vy: ?*const anyopaque) f32;
extern fn iq4_xs_q8_k(n: c_int, vx: ?*const anyopaque, vy: ?*const anyopaque) f32;

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

    pub fn format(self: CpuFeatures, writer: anytype) !void {
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
pub fn dotIq2XxsQ8K(n: usize, vx: [*]const u8, vy: [*]const u8) f32 {
    return iq2_xxs_q8_k(@intCast(n), vx, vy);
}

/// Compute dot product of IQ2_XS quantized vector with Q8_K quantized vector.
pub fn dotIq2XsQ8K(n: usize, vx: [*]const u8, vy: [*]const u8) f32 {
    return iq2_xs_q8_k(@intCast(n), vx, vy);
}

/// Compute dot product of IQ3_XXS quantized vector with Q8_K quantized vector.
pub fn dotIq3XxsQ8K(n: usize, vx: [*]const u8, vy: [*]const u8) f32 {
    return iq3_xxs_q8_k(@intCast(n), vx, vy);
}

/// Compute dot product of IQ4_XS quantized vector with Q8_K quantized vector.
pub fn dotIq4XsQ8K(n: usize, vx: [*]const u8, vy: [*]const u8) f32 {
    return iq4_xs_q8_k(@intCast(n), vx, vy);
}

// Test the C kernels
test "CPU feature detection" {
    const feats = CpuFeatures.detect();
    std.debug.print("CPU features: AVX2={}, AVX512F={}, AVX512_VNNI={}\n", .{
        feats.avx2, feats.avx512f, feats.avx512_vnni,
    });
}