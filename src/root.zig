//! qwen35: a from-scratch Qwen3.5 inference engine in Zig.
//!
//! Module root. Everything public here is reachable as `@import("qwen35").xxx`.

const std = @import("std");

pub const gguf = @import("gguf.zig");
pub const tokenizer = @import("tokenizer.zig");
pub const engine = @import("engine.zig");

test {
    std.testing.refAllDecls(@This());
}

