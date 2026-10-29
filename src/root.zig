//! qwen35: a from-scratch Qwen3.5 inference engine in Zig.
//!
//! Module root. Everything public here is reachable as `@import("qwen35").xxx`.

const std = @import("std");

pub const gguf = @import("gguf.zig");
pub const tokenizer = @import("tokenizer.zig");
pub const engine = @import("engine.zig");
pub const quant_dequant = @import("quant_dequant.zig");
pub const pool = @import("pool.zig");

// The IQ lookup tables are defined here once and consumed by the C kernels via
// `extern const` (src/kernels/x86/iq_tables.h).  Importing them from the root
// module guarantees the symbols are emitted into the final binary regardless of
// which Zig code path reaches them.
pub const iq_tables = @import("iq_tables.zig");

test {
    std.testing.refAllDecls(@This());
}

test {
    _ = @import("iq_table_tests.zig");
}

