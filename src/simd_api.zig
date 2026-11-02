//! Public API of the ISA-specialized quantization kernels.
//!
//! One function-pointer type per kernel family; every tier (baseline / AVX2 /
//! AVX512) exports functions with these signatures, and `simd_dispatch.zig`
//! binds them to the widest tier the running CPU supports.
//!
//! Signatures mirror llama.cpp's `ggml_vec_dot_<x>_q8_<y>(n, s, bs, vx, bx,
//! vy, by, nrc)` entry points reduced to the single-row case (`nrc == 1`) that
//! this engine uses: `n` = element count, `vx` = packed weight row, `vy` =
//! quantized activation, return = dot product in f32.
//!
//!   * `*Q8_0` kernels: `n % 32 == 0`, `vy` is a Q8_0 block array
//!     (2-byte f16 scale + 32 x i8, 34 bytes per 32 elements).
//!   * `*Q8_K` kernels: `n % 256 == 0`, `vy` is a Q8_K block array
//!     (f32 scale + 256 x i8, 260 bytes per 256 elements).

pub const Dot32 = *const fn (n: usize, vx: [*]const u8, vy: [*]const u8) callconv(.c) f32;
pub const Dot256 = *const fn (n: usize, vx: [*]const u8, vy: [*]const u8) callconv(.c) f32;

/// Activation quantizers: `x[0..n]` -> packed Q8_0 / Q8_K blocks in `out`.
pub const QuantizeQ8_0 = *const fn (n: usize, x: [*]const f32, out: [*]u8) callconv(.c) void;
pub const QuantizeQ8_K = *const fn (n: usize, x: [*]const f32, out: [*]u8) callconv(.c) void;

/// f32 x f32 dot (`src/kernels/x86/kernels_f32.c`). Used by the `.f32` matvec
/// path, which is what every bf16/f16/f32 tensor goes through.  `n` is `c_int`
/// (matching the C prototype) rather than `usize` like the quantized kernels,
/// because these entry points have no ISA-specific tiers to bind through.
pub const DotF32 = *const fn (n: c_int, a: [*]const u8, b: [*]const u8) callconv(.c) f32;

extern fn f32_dot_f32(n: c_int, a: [*]const u8, b: [*]const u8) callconv(.c) f32;
extern fn bf16_dot_f32(n: c_int, a: [*]const u8, b: [*]const u8) callconv(.c) f32;

/// Bound directly to the C symbol: there is only one f32 implementation tier
/// (AVX2+FMA with a complete scalar fallback inside the same unit).
pub const dotF32: DotF32 = f32_dot_f32;
pub const dotBf16F32: DotF32 = bf16_dot_f32;
