//! qwen35: a from-scratch Qwen3.5 inference engine in Zig.
//!
//! Module root. Everything public here is reachable as `@import("qwen35").xxx`.

const std = @import("std");

pub const gguf = @import("gguf.zig");
pub const backend = @import("backend.zig");
pub const backend_cpu = @import("backend_cpu.zig");
pub const backend_cuda = @import("backend_cuda.zig");
pub const cudart = @import("cudart.zig");
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

// C-kernel parity coverage (restored after `33e2bb8` unmounted them).
// Each file compares the C kernels (`simd_c.zig`, extern symbols from
// src/kernels/x86/) against the Zig scalar oracle (`simd_ref.zig`), and against
// the engine's own `rowDot*` on real packed rows from testdata/udtest.
// This is the only automated guard for the AGENTS.md §4 symbol-table hazard
// (`ksigns_iq2xs` vs `keven_signs_q2xs`): no test catches a swapped table if
// these files are unmounted.
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

test {
    _ = @import("backend_cpu.zig");
    // Referenced so the trampoline is type-checked by `zig build test` even
    // before anything selects `.cuda`; the CUDA backend has no test of its own
    // (it is a pure forwarding layer) so this is its only compile-time guard.
    _ = @import("backend_cuda.zig");
}

