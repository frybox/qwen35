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

// --- kernels (src/kernels/cuda/convert.cu, mmvq.cu) ------------------------
// `nblocks` is a count of 32-ELEMENT blocks, not a count of elements.
int q35_launch_quantize_q8_0(const float *d_x, void *d_out, int nblocks,
                             void *stream);
int q35_launch_dequantize_q8_0(const void *d_src, float *d_dst, int nblocks,
                               void *stream);

// out[t*rows + r] = W[r] . xq[t].  `xq` is Q8_0-quantized activations, one
// 34-byte block per 32 elements, tokens x cols.
int q35_launch_mul_mat_vec_q8_0(const void *d_w, const void *d_xq, float *d_out,
                                int rows, int cols, int tokens, void *stream);

int q35_launch_mul_mat_vec_f32(const float *d_w, const float *d_x, float *d_out,
                               int rows, int cols, int tokens, void *stream);

#ifdef __cplusplus
}
#endif

#endif // Q35_CUDA_API_H
