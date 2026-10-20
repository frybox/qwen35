//! ISA-specialized quantization kernels, compiled once per CPU tier.
//!
//! Ported from llama.cpp `ggml/src/ggml-cpu/arch/x86/quants.c` (AVX2 / AVX
//! paths) and `ggml/src/ggml-cpu/quants.c` (the `*_generic` fallbacks).  Pure
//! Zig: no C, no C++, no inline assembly.  The SIMD is `@Vector`; LLVM emits
//! the AVX2 (ymm) / AVX-512 (zmm) instructions because build.zig compiles one
//! copy of this directory per `-mcpu` (see the `simdTier` comment there).
//!
//! Every tier exports the same API; `src/simd_dispatch.zig` binds the function
//! pointers to the widest tier the running CPU supports, so a baseline build
//! still runs at full speed on an AVX-512 box.
//!
//! Kernel signatures mirror llama.cpp's
//! `ggml_vec_dot_<x>_q8_<y>(n, s, bs, vx, bx, vy, by, nrc)` reduced to the
//! single-row case this engine needs (`nrc == 1`):
//!
//!   * 32-element super-blocks (`vy` = Q8_0): Q8_0, Q4_0, Q4_1, Q5_0, Q5_1,
//!     Q1_0, Q2_0, MXFP4, NVFP4, IQ4_NL.
//!   * 256-element super-blocks (`vy` = Q8_K): Q2_K, Q3_K, Q4_K, Q5_K, Q6_K,
//!     TQ1_0, TQ2_0, IQ1_S, IQ1_M, IQ2_XXS, IQ2_XS, IQ2_S, IQ3_XXS, IQ3_S,
//!     IQ4_XS.
//!
//! `vx` is the packed weight, `vy` the quantized activation, `n` the element
//! count.  Accumulation order follows the C reference so results match the
//! existing scalar `quant_dequant.zig` rowDot within its test tolerance.

pub const tier = @import("tier.zig").tier;
pub const util = @import("util.zig");

pub const dot32 = @import("kern32.zig"); // 32-block types
pub const dot256 = @import("kern256.zig"); // 256-block K-quant types
pub const dotiq = @import("kerniq.zig"); // 256-block IQ types (IQ2_XXS/XS, IQ3_XXS, IQ4_XS)
pub const dotiq2 = @import("kerniq2.zig"); // 256-block IQ types (IQ2_S, IQ3_S)
pub const dotiq3 = @import("kerniq3.zig"); // 256-block IQ types (IQ1_S, IQ1_M)
pub const act = @import("act.zig"); // activation quantizers (Q8_0 / Q8_K)
