//! Test-only forwarding module: exposes the C kernels (src/c_kernels/x86/*)
//! under the same shape the SIMD test files expect from a tier module
//! (`dot32`, `dot256`, `dotiq`, `dotiq2`, `dotiq3`, `act`).
//!
//! The C kernels loop over blocks internally, so the "row" entry point and the
//! "block" entry point are the same C function: `NAME` and `NAME_row` both
//! forward to one C symbol.  This is the single tier the tests iterate over
//! (see `tier_list` in the test files).
//!
//! The externs below name the C symbols directly and are resolved by linking
//! libiq_kernels.a (build.zig wires `linkLibrary(c_kernels)` on the test
//! artifact).  We do NOT `@import("c_kernels.zig")` here: that file is already
//! part of the root module (via simd_dispatch.zig), and a second module
//! containing the same file is a compile error.
//!
//! `tq1_0_q8_k` / `tq2_0_q8_k` exist in kernels256.c but have no wrapper in
//! `c_kernels.zig`; declaring them here keeps production untouched.

const C = struct {
    extern fn q8_0_q8_0(n: c_int, vx: ?*const anyopaque, vy: ?*const anyopaque) f32;
    extern fn q4_0_q8_0(n: c_int, vx: ?*const anyopaque, vy: ?*const anyopaque) f32;
    extern fn q4_1_q8_0(n: c_int, vx: ?*const anyopaque, vy: ?*const anyopaque) f32;
    extern fn q5_0_q8_0(n: c_int, vx: ?*const anyopaque, vy: ?*const anyopaque) f32;
    extern fn q5_1_q8_0(n: c_int, vx: ?*const anyopaque, vy: ?*const anyopaque) f32;
    extern fn q1_0_q8_0(n: c_int, vx: ?*const anyopaque, vy: ?*const anyopaque) f32;
    extern fn q2_0_q8_0(n: c_int, vx: ?*const anyopaque, vy: ?*const anyopaque) f32;
    extern fn iq4_nl_q8_0(n: c_int, vx: ?*const anyopaque, vy: ?*const anyopaque) f32;
    extern fn mxfp4_q8_0(n: c_int, vx: ?*const anyopaque, vy: ?*const anyopaque) f32;
    extern fn nvfp4_q8_0(n: c_int, vx: ?*const anyopaque, vy: ?*const anyopaque) f32;

    extern fn q2_k_q8_k(n: c_int, vx: ?*const anyopaque, vy: ?*const anyopaque) f32;
    extern fn q3_k_q8_k(n: c_int, vx: ?*const anyopaque, vy: ?*const anyopaque) f32;
    extern fn q4_k_q8_k(n: c_int, vx: ?*const anyopaque, vy: ?*const anyopaque) f32;
    extern fn q5_k_q8_k(n: c_int, vx: ?*const anyopaque, vy: ?*const anyopaque) f32;
    extern fn q6_k_q8_k(n: c_int, vx: ?*const anyopaque, vy: ?*const anyopaque) f32;
    extern fn tq1_0_q8_k(n: c_int, vx: ?*const anyopaque, vy: ?*const anyopaque) f32;
    extern fn tq2_0_q8_k(n: c_int, vx: ?*const anyopaque, vy: ?*const anyopaque) f32;

    extern fn iq2_xxs_q8_k(n: c_int, vx: ?*const anyopaque, vy: ?*const anyopaque) f32;
    extern fn iq2_xs_q8_k(n: c_int, vx: ?*const anyopaque, vy: ?*const anyopaque) f32;
    extern fn iq3_xxs_q8_k(n: c_int, vx: ?*const anyopaque, vy: ?*const anyopaque) f32;
    extern fn iq4_xs_q8_k(n: c_int, vx: ?*const anyopaque, vy: ?*const anyopaque) f32;
    extern fn iq2_s_q8_k(n: c_int, vx: ?*const anyopaque, vy: ?*const anyopaque) f32;
    extern fn iq3_s_q8_k(n: c_int, vx: ?*const anyopaque, vy: ?*const anyopaque) f32;
    extern fn iq1_s_q8_k(n: c_int, vx: ?*const anyopaque, vy: ?*const anyopaque) f32;
    extern fn iq1_m_q8_k(n: c_int, vx: ?*const anyopaque, vy: ?*const anyopaque) f32;

    extern fn quantize_q8_0(n: c_int, x: ?*const anyopaque, out: ?*anyopaque) void;
    extern fn quantize_q8_K(n: c_int, x: ?*const anyopaque, out: ?*anyopaque) void;
};

pub const dot32 = struct {
    pub fn q8_0_q8_0(n: usize, vx: [*]const u8, vy: [*]const u8) callconv(.c) f32 {
        return C.q8_0_q8_0(@intCast(n), vx, vy);
    }
    pub fn q4_0_q8_0(n: usize, vx: [*]const u8, vy: [*]const u8) callconv(.c) f32 {
        return C.q4_0_q8_0(@intCast(n), vx, vy);
    }
    pub fn q4_1_q8_0(n: usize, vx: [*]const u8, vy: [*]const u8) callconv(.c) f32 {
        return C.q4_1_q8_0(@intCast(n), vx, vy);
    }
    pub fn q5_0_q8_0(n: usize, vx: [*]const u8, vy: [*]const u8) callconv(.c) f32 {
        return C.q5_0_q8_0(@intCast(n), vx, vy);
    }
    pub fn q5_1_q8_0(n: usize, vx: [*]const u8, vy: [*]const u8) callconv(.c) f32 {
        return C.q5_1_q8_0(@intCast(n), vx, vy);
    }
    pub fn q1_0_q8_0(n: usize, vx: [*]const u8, vy: [*]const u8) callconv(.c) f32 {
        return C.q1_0_q8_0(@intCast(n), vx, vy);
    }
    pub fn q2_0_q8_0(n: usize, vx: [*]const u8, vy: [*]const u8) callconv(.c) f32 {
        return C.q2_0_q8_0(@intCast(n), vx, vy);
    }
    pub fn iq4_nl_q8_0(n: usize, vx: [*]const u8, vy: [*]const u8) callconv(.c) f32 {
        return C.iq4_nl_q8_0(@intCast(n), vx, vy);
    }
    pub fn mxfp4_q8_0(n: usize, vx: [*]const u8, vy: [*]const u8) callconv(.c) f32 {
        return C.mxfp4_q8_0(@intCast(n), vx, vy);
    }
    pub fn nvfp4_q8_0(n: usize, vx: [*]const u8, vy: [*]const u8) callconv(.c) f32 {
        return C.nvfp4_q8_0(@intCast(n), vx, vy);
    }

    // Row entry points: the C kernels iterate over blocks, so a row and a
    // single block go through the same code path.
    pub fn q8_0_q8_0_row(n: usize, vx: [*]const u8, vy: [*]const u8) callconv(.c) f32 {
        return q8_0_q8_0(n, vx, vy);
    }
    pub fn q4_0_q8_0_row(n: usize, vx: [*]const u8, vy: [*]const u8) callconv(.c) f32 {
        return q4_0_q8_0(n, vx, vy);
    }
    pub fn q4_1_q8_0_row(n: usize, vx: [*]const u8, vy: [*]const u8) callconv(.c) f32 {
        return q4_1_q8_0(n, vx, vy);
    }
    pub fn q5_0_q8_0_row(n: usize, vx: [*]const u8, vy: [*]const u8) callconv(.c) f32 {
        return q5_0_q8_0(n, vx, vy);
    }
    pub fn q5_1_q8_0_row(n: usize, vx: [*]const u8, vy: [*]const u8) callconv(.c) f32 {
        return q5_1_q8_0(n, vx, vy);
    }
    pub fn q1_0_q8_0_row(n: usize, vx: [*]const u8, vy: [*]const u8) callconv(.c) f32 {
        return q1_0_q8_0(n, vx, vy);
    }
    pub fn q2_0_q8_0_row(n: usize, vx: [*]const u8, vy: [*]const u8) callconv(.c) f32 {
        return q2_0_q8_0(n, vx, vy);
    }
    pub fn iq4_nl_q8_0_row(n: usize, vx: [*]const u8, vy: [*]const u8) callconv(.c) f32 {
        return iq4_nl_q8_0(n, vx, vy);
    }
    pub fn mxfp4_q8_0_row(n: usize, vx: [*]const u8, vy: [*]const u8) callconv(.c) f32 {
        return mxfp4_q8_0(n, vx, vy);
    }
    pub fn nvfp4_q8_0_row(n: usize, vx: [*]const u8, vy: [*]const u8) callconv(.c) f32 {
        return nvfp4_q8_0(n, vx, vy);
    }
};

pub const dot256 = struct {
    pub fn q2_k_q8_k(n: usize, vx: [*]const u8, vy: [*]const u8) callconv(.c) f32 {
        return C.q2_k_q8_k(@intCast(n), vx, vy);
    }
    pub fn q3_k_q8_k(n: usize, vx: [*]const u8, vy: [*]const u8) callconv(.c) f32 {
        return C.q3_k_q8_k(@intCast(n), vx, vy);
    }
    pub fn q4_k_q8_k(n: usize, vx: [*]const u8, vy: [*]const u8) callconv(.c) f32 {
        return C.q4_k_q8_k(@intCast(n), vx, vy);
    }
    pub fn q5_k_q8_k(n: usize, vx: [*]const u8, vy: [*]const u8) callconv(.c) f32 {
        return C.q5_k_q8_k(@intCast(n), vx, vy);
    }
    pub fn q6_k_q8_k(n: usize, vx: [*]const u8, vy: [*]const u8) callconv(.c) f32 {
        return C.q6_k_q8_k(@intCast(n), vx, vy);
    }
    pub fn tq1_0_q8_k(n: usize, vx: [*]const u8, vy: [*]const u8) callconv(.c) f32 {
        return C.tq1_0_q8_k(@intCast(n), vx, vy);
    }
    pub fn tq2_0_q8_k(n: usize, vx: [*]const u8, vy: [*]const u8) callconv(.c) f32 {
        return C.tq2_0_q8_k(@intCast(n), vx, vy);
    }

    pub fn q2_k_q8_k_row(n: usize, vx: [*]const u8, vy: [*]const u8) callconv(.c) f32 {
        return q2_k_q8_k(n, vx, vy);
    }
    pub fn q3_k_q8_k_row(n: usize, vx: [*]const u8, vy: [*]const u8) callconv(.c) f32 {
        return q3_k_q8_k(n, vx, vy);
    }
    pub fn q4_k_q8_k_row(n: usize, vx: [*]const u8, vy: [*]const u8) callconv(.c) f32 {
        return q4_k_q8_k(n, vx, vy);
    }
    pub fn q5_k_q8_k_row(n: usize, vx: [*]const u8, vy: [*]const u8) callconv(.c) f32 {
        return q5_k_q8_k(n, vx, vy);
    }
    pub fn q6_k_q8_k_row(n: usize, vx: [*]const u8, vy: [*]const u8) callconv(.c) f32 {
        return q6_k_q8_k(n, vx, vy);
    }
    pub fn tq1_0_q8_k_row(n: usize, vx: [*]const u8, vy: [*]const u8) callconv(.c) f32 {
        return tq1_0_q8_k(n, vx, vy);
    }
    pub fn tq2_0_q8_k_row(n: usize, vx: [*]const u8, vy: [*]const u8) callconv(.c) f32 {
        return tq2_0_q8_k(n, vx, vy);
    }
};

pub const dotiq = struct {
    pub fn iq2_xxs_q8_k(n: usize, vx: [*]const u8, vy: [*]const u8) callconv(.c) f32 {
        return C.iq2_xxs_q8_k(@intCast(n), vx, vy);
    }
    pub fn iq2_xs_q8_k(n: usize, vx: [*]const u8, vy: [*]const u8) callconv(.c) f32 {
        return C.iq2_xs_q8_k(@intCast(n), vx, vy);
    }
    pub fn iq3_xxs_q8_k(n: usize, vx: [*]const u8, vy: [*]const u8) callconv(.c) f32 {
        return C.iq3_xxs_q8_k(@intCast(n), vx, vy);
    }
    pub fn iq4_xs_q8_k(n: usize, vx: [*]const u8, vy: [*]const u8) callconv(.c) f32 {
        return C.iq4_xs_q8_k(@intCast(n), vx, vy);
    }

    pub fn iq2_xxs_q8_k_row(n: usize, vx: [*]const u8, vy: [*]const u8) callconv(.c) f32 {
        return iq2_xxs_q8_k(n, vx, vy);
    }
    pub fn iq2_xs_q8_k_row(n: usize, vx: [*]const u8, vy: [*]const u8) callconv(.c) f32 {
        return iq2_xs_q8_k(n, vx, vy);
    }
    pub fn iq3_xxs_q8_k_row(n: usize, vx: [*]const u8, vy: [*]const u8) callconv(.c) f32 {
        return iq3_xxs_q8_k(n, vx, vy);
    }
    pub fn iq4_xs_q8_k_row(n: usize, vx: [*]const u8, vy: [*]const u8) callconv(.c) f32 {
        return iq4_xs_q8_k(n, vx, vy);
    }
};

pub const dotiq2 = struct {
    pub fn iq2_s_q8_k(n: usize, vx: [*]const u8, vy: [*]const u8) callconv(.c) f32 {
        return C.iq2_s_q8_k(@intCast(n), vx, vy);
    }
    pub fn iq3_s_q8_k(n: usize, vx: [*]const u8, vy: [*]const u8) callconv(.c) f32 {
        return C.iq3_s_q8_k(@intCast(n), vx, vy);
    }

    pub fn iq2_s_q8_k_row(n: usize, vx: [*]const u8, vy: [*]const u8) callconv(.c) f32 {
        return iq2_s_q8_k(n, vx, vy);
    }
    pub fn iq3_s_q8_k_row(n: usize, vx: [*]const u8, vy: [*]const u8) callconv(.c) f32 {
        return iq3_s_q8_k(n, vx, vy);
    }
};

pub const dotiq3 = struct {
    pub fn iq1_s_q8_k(n: usize, vx: [*]const u8, vy: [*]const u8) callconv(.c) f32 {
        return C.iq1_s_q8_k(@intCast(n), vx, vy);
    }
    pub fn iq1_m_q8_k(n: usize, vx: [*]const u8, vy: [*]const u8) callconv(.c) f32 {
        return C.iq1_m_q8_k(@intCast(n), vx, vy);
    }

    pub fn iq1_s_q8_k_row(n: usize, vx: [*]const u8, vy: [*]const u8) callconv(.c) f32 {
        return iq1_s_q8_k(n, vx, vy);
    }
    pub fn iq1_m_q8_k_row(n: usize, vx: [*]const u8, vy: [*]const u8) callconv(.c) f32 {
        return iq1_m_q8_k(n, vx, vy);
    }
};

pub const act = struct {
    pub fn quantizeQ8_0(n: usize, x: [*]const f32, out: [*]u8) callconv(.c) void {
        C.quantize_q8_0(@intCast(n), x, out);
    }
    pub fn quantizeQ8_K(n: usize, x: [*]const f32, out: [*]u8) callconv(.c) void {
        C.quantize_q8_K(@intCast(n), x, out);
    }
};
