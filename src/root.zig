//! qwen35: a from-scratch Qwen3.5 inference engine in Zig.
//!
//! Module root. Everything public here is reachable as `@import("qwen35").xxx`.

const std = @import("std");

pub const gguf = @import("gguf.zig");
pub const tokenizer = @import("tokenizer.zig");
pub const engine = @import("engine.zig");
pub const quant_dequant = @import("quant_dequant.zig");

test {
    std.testing.refAllDecls(@This());
}

test {
    _ = @import("simd_tests32.zig");
}

test {
    _ = @import("simd_tests256.zig");
}

test {
    _ = @import("simd_tests_iq_a.zig");
}

test {
    _ = @import("simd_tests_iq_b.zig");
}

test {
    _ = @import("simd_tests_iq_c.zig");
}

