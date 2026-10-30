// api.h -- the only surface the Zig backend sees of the CUDA kernels.
//
// Why a separate header: `src/backend_cuda.zig` declares these with `extern`
// and never includes a CUDA header, so the Zig side cannot accidentally pick
// up a link-time dependency on cudart.  Everything that touches the CUDA
// runtime lives in the .cu files on the other side of this boundary, and those
// .cu files are compiled into a plain .o that is linked into the executable.
//
// The launchers return the numeric value of `cudaError_t` (0 == cudaSuccess)
// cast to int, so the caller can turn a failure into a named Zig error plus
// cudaGetErrorString() text instead of panicking.  Device pointers are opaque
// `void *` and the stream is `void *` too, which keeps this header free of
// cuda_runtime.h (and therefore includable from plain C/C++ if ever needed).
#ifndef Q35_CUDA_API_H
#define Q35_CUDA_API_H

#ifdef __cplusplus
extern "C" {
#endif

// Layout the host side must satisfy (mirrors ggml block_q8_0):
//   32 elements per block, 34 bytes per block (f16 d @0, int8 qs @2).
#define Q35_API_QK8_0 32
#define Q35_API_Q8_0_BLOCK_BYTES 34

// Layout of a Q8_K activation block (mirrors ggml block_q8_K):
//   float d @0, int8 qs[256] @4, int16 bsums[16] @260 -> 292 bytes / 256 elems.
#define Q35_API_QK_K 256
#define Q35_API_Q8_K_BLOCK_BYTES 292

// ---------------------------------------------------------------------------
// TensorType ids.
//
// MUST MATCH src/gguf.zig's `TensorType` enum, both in VALUE and in NAME --
// the numeric values below are copied from that enum, so Zig can pass
// `@intFromEnum(ttype)` straight through without a translation table.
// Renumbering anything here without renumbering gguf.zig silently makes every
// CUDA kernel read the wrong block geometry (the kernels then read
// out of bounds rather than producing obviously-wrong numbers).
//
// Types with no CUDA dot/dequant implementation are deliberately absent; the
// host launchers reject unknown ids with cudaErrorInvalidValue.
// ---------------------------------------------------------------------------
#define Q35_TT_F32      0
#define Q35_TT_F16      1
#define Q35_TT_Q4_0     2
#define Q35_TT_Q4_1     3
#define Q35_TT_Q5_0     6
#define Q35_TT_Q5_1     7
#define Q35_TT_Q8_0     8
#define Q35_TT_Q8_1     9
#define Q35_TT_Q2_K     10
#define Q35_TT_Q3_K     11
#define Q35_TT_Q4_K     12
#define Q35_TT_Q5_K     13
#define Q35_TT_Q6_K     14
#define Q35_TT_Q8_K     15
#define Q35_TT_IQ2_XXS  16
#define Q35_TT_IQ2_XS   17
#define Q35_TT_IQ3_XXS  18
#define Q35_TT_IQ1_S    19
#define Q35_TT_IQ4_NL   20
#define Q35_TT_IQ3_S    21
#define Q35_TT_IQ2_S    22
#define Q35_TT_IQ4_XS   23
#define Q35_TT_BF16     30
#define Q35_TT_IQ1_M    29

// --- activation quantization (src/kernels/cuda/convert.cu, mmvq.cu) --------
// `nblocks` is a count of 32-ELEMENT blocks, not a count of elements.
int q35_launch_quantize_q8_0(const float *d_x, void *d_out, int nblocks,
                             void *stream);
// `nblocks` is a count of 256-ELEMENT blocks.  Bit-exact with the CPU
// quantize_q8_K (src/kernels/x86/kernels256.c:392).
int q35_launch_quantize_q8_K(const float *d_x, void *d_out, int nblocks,
                             void *stream);

// --- dequantization (src/kernels/cuda/convert.cu) --------------------------
// `nblocks` is a count of the type's NATIVE blocks (32 elements for
// q8_0/q4_0/q4_1/q5_0/q5_1/iq4_nl, 256 for the K/IQ family), NOT elements.
int q35_launch_dequantize(int ttype, const void *d_src, float *d_dst, int nblocks,
                          void *stream);
// Dequantize one weight row of `cols` floats.  `cols` must be a multiple of the
// type's native block size.  Same semantics as src/backend_cpu.zig's
// dequantRowInto: handles the row block by block, so it never needs a
// temporary allocation.
int q35_launch_dequantize_row(int ttype, const void *d_src_row, float *d_dst,
                              int cols, void *stream);

// --- matvec (src/kernels/cuda/mmvq.cu) -------------------------------------
// out[t*rows + r] = W[r] . xq[t] for `tokens` activation rows.
// `xq` must be quantized in the layout the type asks for: Q8_0 (34 B / 32
// elems) for the 32-element family, Q8_K (292 B / 256 elems) for the 256-elem
// family -- see q35_quant_act_block_elems().  `cols` must be a multiple of the
// type's native block size.  Returns cudaErrorInvalidValue for an unknown type.
int q35_launch_mul_mat_vec(const void *d_w, const void *d_xq, float *d_out,
                           int rows, int cols, int tokens, int ttype,
                           void *stream);

// Backwards-compatible alias for
// q35_launch_mul_mat_vec(..., Q35_TT_Q8_0, ...).
int q35_launch_mul_mat_vec_q8_0(const void *d_w, const void *d_xq, float *d_out,
                                int rows, int cols, int tokens, void *stream);

int q35_launch_mul_mat_vec_f32(const float *d_w, const float *d_x, float *d_out,
                               int rows, int cols, int tokens, void *stream);

// --- geometry queries (pure host code, no launch) --------------------------
// 256 for types that want Q8_K activations, 32 for Q8_0, 0 for unknown types.
int q35_quant_act_block_elems(int ttype);
// Native block geometry of a type.  Returns 1 and fills the out-params on
// success, 0 for an unknown type.
int q35_quant_geom(int ttype, int *block_elems, int *block_bytes);

#ifdef __cplusplus
}
#endif

#endif // Q35_CUDA_API_H
