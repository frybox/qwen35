/*
 * kernels32.h — Public API for 32-element dot-product kernels
 */
#ifndef KERNELS32_H
#define KERNELS32_H

#ifdef __cplusplus
extern "C" {
#endif

/* 32-element dot-product kernels: weight x Q8_0 activation */
float q4_0_q8_0(int n, const void* vx, const void* vy);
float q4_1_q8_0(int n, const void* vx, const void* vy);
float iq4_nl_q8_0(int n, const void* vx, const void* vy);

/* Additional kernels (newly added) */
float q8_0_q8_0(int n, const void* vx, const void* vy);
float q5_0_q8_0(int n, const void* vx, const void* vy);
float q5_1_q8_0(int n, const void* vx, const void* vy);
float q1_0_q8_0(int n, const void* vx, const void* vy);
float q2_0_q8_0(int n, const void* vx, const void* vy);
float mxfp4_q8_0(int n, const void* vx, const void* vy);
float nvfp4_q8_0(int n, const void* vx, const void* vy);

/* Quantization */
void quantize_q8_0(int n, const float* x, uint8_t* out);

#ifdef __cplusplus
}
#endif

#endif /* KERNELS32_H */
