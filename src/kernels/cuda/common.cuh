// common.cuh -- shared device helpers for the qwen35 Q8_0 CUDA kernels.
//
// Why this file exists: the incremental `--backend cuda` backend needs only a
// handful of kernels (Q8_0 activation quantize + Q8_0 x Q8_0 matvec + Q8_0
// dequant), not the whole engine port that lives in src/kernel.cu.  Keeping
// them in their own translation unit means the host object can be produced by
// a single `nvcc -c` invocation with no relocatable-device-code (-rdc) link
// step and no CUB dependency.
//
// Source anchors (all bit-level conventions copied from upstream, not from
// memory):
//   ggml/src/ggml-quants.c  block_q8_0 layout / quantize_row_q8_0_ref
//   ggml/src/ggml-cuda/convert.cu:... dequantize_block_q8_0
//   ggml/src/ggml-cuda/vecdotq.cuh:246 vec_dot_q8_0_q8_1_impl
// The *numeric order* is anchored to this repo instead, because the goal is a
// bit-identical result against the CPU backend:
//   src/kernels/x86/kernels32.c:379 k32_q8_0_q8_0_avx2  (acc order)
//   src/kernels/x86/kernels32.c:1033 quantize_q8_0      (activation quantize)
#ifndef Q35_CUDA_COMMON_CUH
#define Q35_CUDA_COMMON_CUH

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <stdint.h>

// Q8_0 block geometry: one f16 scale followed by 32 signed bytes, packed
// back-to-back at 34 bytes, so the int8 payload of a block is only 2-byte
// aligned.  Every load below therefore goes through byte or uint16 access --
// a 4-byte `int` load would be both misaligned and out of bounds on the last
// block.
#define Q35_QK8_0 32
#define Q35_Q8_0_BLOCK_BYTES 34

// f16 bits -> f32.  `__half2float` is the same IEEE-754 binary16 decode the
// CPU path performs in k32_f16_to_f32 (kernels32.c:45), including subnormals.
__device__ __forceinline__ float q35_ld_scale(const unsigned char *p) {
    const unsigned short u = (unsigned short)p[0] | ((unsigned short)p[1] << 8);
    return __half2float(__ushort_as_half(u));
}

// f32 -> f16 bits, round-to-nearest-even.  The CPU quantizer stores the scale
// through k32_f32_to_f16bits (round-to-nearest-even, kernels32.c:1050) and
// keeps using the *unrounded* f32 scale to build `inv` -- so this helper is
// only for writing the block header; the inverse must use from_f32(scale).
//
// Why this is hand-rolled integer arithmetic and NOT `__half_as_ushort(
// __float2half(s))`: ptxas miscompiles the intrinsic pair when only the low
// byte of the result is consumed.  `cvt.rn.f16.f32 %rs1, %f; st.global.u8
// [p], %rs1;` comes out of ptxas with the low 8 bits of the mantissa zeroed --
// the observed block header was 0x2b00 where 0x2b33 is correct, i.e. every
// quantized activation carried a scale ~0.06 % low, and because the block is
// then decoded with that same scale the whole tensor drifted (measured:
// MVCHECK mismatched=248320 of 248320 rows).  Reproduced standalone on
// sm_89 with nvcc -O3, and it disappears if a device printf keeps the value
// live -- a textbook dead-store elimination on a value the compiler believes
// is only used through its high byte.
//
// The integer form below is also exactly the CPU helper's algorithm
// (kernels32.c:67 k32_f32_to_f16bits), which is what the bit-identical
// requirement wants; divergences in the subnormal range are listed there.
__device__ __forceinline__ unsigned short q35_st_scale_bits(float x) {
    const unsigned u = __float_as_uint(x);
    const unsigned sign = (u >> 16) & 0x8000u;
    const unsigned e = (u >> 23) & 0xFFu;
    const unsigned m = u & 0x7FFFFFu;
    if (e == 255u) return (unsigned short)(0x7C00u | sign); // inf / nan -> inf
    if (e >= 143u) return (unsigned short)(0x7C00u | sign); // > 65504 -> inf
    if (e >= 113u) {
        // normal f16: round the 23-bit mantissa to 10 bits, nearest-even.
        unsigned keep = m >> 13;
        const unsigned drop = m & 0x1FFFu;
        const unsigned up = (drop > 0x1000u) || (drop == 0x1000u && (keep & 1u));
        keep += up;
        unsigned e16 = e - 112u;
        if (keep >= 0x400u) { keep = 0; e16 += 1; }        // mantissa carry
        if (e16 > 30u) return (unsigned short)(0x7C00u | sign);
        return (unsigned short)(sign | (e16 << 10) | keep);
    }
    // e in [0,112]: f16 subnormal or zero.  Same recurrence as the CPU
    // helper (kernels32.c:85-97) so the header bits are identical even when a
    // whole activation block is denormal-small.
    {
        const unsigned t = m | 0x800000u;
        const int s = (int)(126u - e);
        if (s >= 25) return (unsigned short)sign; // rounds to 0
        unsigned keep = t >> s;
        const unsigned drop = t & ((1u << s) - 1u);
        const unsigned mid = 1u << (s - 1);
        const unsigned up = (drop > mid) || (drop == mid && (keep & 1u));
        keep += up;
        if (keep >= 0x400u) return (unsigned short)(0x0380u | sign);
        return (unsigned short)(sign | keep);
    }
}

__device__ __forceinline__ void q35_store_scale(unsigned char *p, float s) {
    const unsigned short u = q35_st_scale_bits(s);
    p[0] = (unsigned char)(u & 0xffu);
    p[1] = (unsigned char)(u >> 8);
}

// 4-way int8 dot.  sm_61+ has the hardware instruction (target: 4090D, sm_89);
// the fallback keeps the file compilable for lower targets and is a faithful
// scalar equivalent, not a stub (AGENTS.md §1).
__device__ __forceinline__ int q35_dp4a(int a, int b, int c) {
#if __CUDA_ARCH__ >= 610
    return __dp4a(a, b, c);
#else
    const signed char *aa = (const signed char *)&a;
    const signed char *bb = (const signed char *)&b;
    return c + aa[0] * bb[0] + aa[1] * bb[1] + aa[2] * bb[2] + aa[3] * bb[3];
#endif
}

// Little-endian 32-bit view of 4 consecutive int8 payload bytes.  qs+2 is
// 2-byte aligned only, so this is two uint16 loads glued together instead of
// a single unaligned int load.
__device__ __forceinline__ int q35_get_int_b4(const signed char *p) {
    const unsigned char *u = (const unsigned char *)p;
    const unsigned int lo = (unsigned int)u[0] | ((unsigned int)u[1] << 8);
    const unsigned int hi = (unsigned int)u[2] | ((unsigned int)u[3] << 8);
    return (int)(lo | (hi << 16));
}

#endif // Q35_CUDA_COMMON_CUH
