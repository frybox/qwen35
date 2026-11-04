/*
 * kernels_f32.h — Public API for f32/bf16 dot-product kernels
 */
#ifndef KERNELS_F32_H
#define KERNELS_F32_H

#ifdef __cplusplus
extern "C" {
#endif

/* out[r] = sum_c a[r*cols + c] * b[c]  (a: n f32, b: n f32) */
float f32_dot_f32(int n, const void* a, const void* b);

/* Same, but `a` holds n bf16 (2 bytes each) expanded on the fly. */
float bf16_dot_f32(int n, const void* a, const void* b);

/* Same, but `a` holds n f16 (2 bytes each) expanded on the fly. */
float f16_dot_f32(int n, const void* a, const void* b);

#ifdef __cplusplus
}
#endif

#endif /* KERNELS_F32_H */
