// vecdotq.cuh -- Q8_0 x Q8_0 block dot product (device side).
//
// Anchor: ggml/src/ggml-cuda/vecdotq.cuh:246 vec_dot_q8_0_q8_1_impl
//         ggml/src/ggml-cuda/vecdotq.cuh:851  vec_dot_q8_0_q8_1
//
// Upstream returns `d8_0*d8_1*sumi` per block and lets the caller accumulate.
// This repo's CPU reference (which the GPU result must match bit-for-bit)
// fuses the accumulation:
//   src/kernels/x86/kernels32.c:394  acc = _mm256_fmadd_ps(d, cvtepi32(sumi), acc)
//   src/kernels/x86/kernels32.c:430  sumf += dx*dy*sumi            (scalar path)
// `fmaf` below is the same single-rounding multiply-add as _mm256_fmadd_ps,
// and the caller walks blocks strictly left to right, so the float addition
// tree is identical.  A warp-per-row reduction (as src/kernel.cu does) would
// reorder that tree and drift by an ulp, which is why mmvq.cu uses one thread
// per row instead.
#ifndef Q35_CUDA_VECDOTQ_CUH
#define Q35_CUDA_VECDOTQ_CUH

#include "common.cuh"

// 32 int8 pairs -> one int32 sum.  dp4a is exact (int32, no saturation in the
// range reachable here: |sumi| <= 32*128*128), so this is integer-identical to
// the CPU kernel's vpmaddwd path.
__device__ __forceinline__ int q35_dot_i8x32(const signed char *w,
                                             const signed char *a) {
    int sumi = 0;
#pragma unroll
    for (int i = 0; i < 8; ++i) {
        sumi = q35_dp4a(q35_get_int_b4(w + 4 * i), q35_get_int_b4(a + 4 * i), sumi);
    }
    return sumi;
}

// acc += (d_w * d_a) * sumi, evaluated in EXACTLY the CPU tier's order.
//
// Which tier matters: kernels32.c:379 (AVX2) does
//     acc = _mm256_fmadd_ps(d, cvtepi32_ps(sumi), acc)
// i.e. one rounding for the whole `d*sumi + acc`, while kernels512.c:76 (the
// AVX-512 VNNI path that this host actually selects) does
//     sumf += dx * dy * (float)sumi;
// i.e. `dx*dy` rounded, then times `sumi` rounded, then added.  The two differ
// by an ulp, which compounds over 150+ matvecs per token into different logits
// and eventually a different token -- so mirror the *VNNI* form, and reproduce
// it with plain multiply/add rather than fmaf (the compiler must not contract
// them back into an fma, hence `__fmul_rn` / `__fadd_rn`).
//
// The AVX2 tier's fma form is therefore *not* bit-identical to this kernel.
// That is accepted: the target is the default (highest) tier, and matching two
// different CPU tiers at once is impossible with one float expression.
__device__ __forceinline__ float q35_vec_dot_q8_0_q8_0_acc(
    const unsigned char *w, const unsigned char *a, float acc) {
    const float dx = q35_ld_scale(w);
    const float dy = q35_ld_scale(a);
    const int sumi = q35_dot_i8x32((const signed char *)(w + 2),
                                   (const signed char *)(a + 2));
    // __fmul_rn / __fadd_rn are the intrinsics that forbid fma contraction.
    const float d = __fmul_rn(dx, dy);
    return __fadd_rn(acc, __fmul_rn(d, (float)sumi));
}

#endif // Q35_CUDA_VECDOTQ_CUH
