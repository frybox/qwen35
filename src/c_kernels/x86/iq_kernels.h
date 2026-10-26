/**
 * @file iq_kernels.h
 * @brief IQ quantization kernels with CPU feature detection and runtime dispatch.
 *
 * This file provides a unified interface for IQ quantized matrix multiplication
 * kernels that automatically dispatch to the best available implementation
 * (AVX2, AVX, or scalar) based on CPU features detected at runtime.
 *
 * Supported quantization types:
 *   - IQ2_XXS: 2-bit quantization (2.0625 bpw)
 *   - IQ2_XS:  2-bit quantization (2.3125 bpw)
 *   - IQ3_XXS: 3-bit quantization (3.0625 bpw)
 *   - IQ4_XS:  4-bit quantization (4.5 bpw)
 *
 * Ported from llama.cpp: ggml/src/ggml-cpu/arch/x86/quants.c
 */

#ifndef IQ_KERNELS_H
#define IQ_KERNELS_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* ============================================================================
 * CPU Feature Detection
 * ============================================================================ */

/**
 * @brief Check if the CPU supports AVX2 instructions.
 * @return 1 if AVX2 is supported, 0 otherwise.
 */
int cpu_has_avx2(void);

/**
 * @brief Check if the CPU supports AVX-512F instructions.
 * @return 1 if AVX-512F is supported, 0 otherwise.
 */
int cpu_has_avx512f(void);

/**
 * @brief Check if the CPU supports AVX-512 VNNI instructions.
 * @return 1 if AVX-512 VNNI is supported, 0 otherwise.
 */
int cpu_has_avx512_vnni(void);

/**
 * @brief Get a string describing the CPU features.
 * @return A static string describing detected CPU features.
 */
const char* cpu_get_features_string(void);

/* ============================================================================
 * IQ Quantization Kernels
 * ============================================================================ */

/**
 * @brief Compute dot product of IQ2_XXS quantized vector with Q8_K quantized vector.
 *
 * This function computes: sum(vx[i] * vy[i]) for i in 0..n-1
 * where vx is IQ2_XXS quantized and vy is Q8_K quantized.
 *
 * @param n   Number of elements (must be a multiple of 256).
 * @param vx  Pointer to IQ2_XXS quantized data (block_iq2_xxs format).
 * @param vy  Pointer to Q8_K quantized data (block_q8_K format).
 * @return    The dot product as a float.
 */
float iq2_xxs_q8_k(int n, const void* vx, const void* vy);

/**
 * @brief Compute dot product of IQ2_XS quantized vector with Q8_K quantized vector.
 *
 * @param n   Number of elements (must be a multiple of 256).
 * @param vx  Pointer to IQ2_XS quantized data (block_iq2_xs format).
 * @param vy  Pointer to Q8_K quantized data (block_q8_K format).
 * @return    The dot product as a float.
 */
float iq2_xs_q8_k(int n, const void* vx, const void* vy);

/**
 * @brief Compute dot product of IQ3_XXS quantized vector with Q8_K quantized vector.
 *
 * @param n   Number of elements (must be a multiple of 256).
 * @param vx  Pointer to IQ3_XXS quantized data (block_iq3_xxs format).
 * @param vy  Pointer to Q8_K quantized data (block_q8_K format).
 * @return    The dot product as a float.
 */
float iq3_xxs_q8_k(int n, const void* vx, const void* vy);

/**
 * @brief Compute dot product of IQ4_XS quantized vector with Q8_K quantized vector.
 *
 * @param n   Number of elements (must be a multiple of 256).
 * @param vx  Pointer to IQ4_XS quantized data (block_iq4_xs format).
 * @param vy  Pointer to Q8_K quantized data (block_q8_K format).
 * @return    The dot product as a float.
 */
float iq4_xs_q8_k(int n, const void* vx, const void* vy);

/**
 * @brief Compute dot product of TQ2_0 quantized vector with Q8_K quantized vector.
 *
 * TQ2_0: 2-bit ternary-ish quantization, 66 bytes per 256 elements.
 * Value = ((qs >> 2*l) & 3) - 1, giving {-1, 0, 1, 2}.
 *
 * @param n   Number of elements (must be a multiple of 256).
 * @param vx  Pointer to TQ2_0 quantized data.
 * @param vy  Pointer to Q8_K quantized data (block_q8_K format).
 * @return    The dot product as a float.
 */
float tq2_0_q8_k(int n, const void* vx, const void* vy);

/**
 * @brief Compute dot product of TQ1_0 quantized vector with Q8_K quantized vector.
 *
 * TQ1_0: Base-5 packing ternary quantization, 54 bytes per 256 elements.
 * Value = (((qs * pow3[l]) & 0xFF) * 3 >> 8) - 1, giving {-1, 0, 1}.
 *
 * @param n   Number of elements (must be a multiple of 256).
 * @param vx  Pointer to TQ1_0 quantized data.
 * @param vy  Pointer to Q8_K quantized data (block_q8_K format).
 * @return    The dot product as a float.
 */
float tq1_0_q8_k(int n, const void* vx, const void* vy);

/**
 * @brief Compute dot product of IQ1_S quantized vector with Q8_K quantized vector.
 *
 * IQ1_S: 1-bit quantization with separate scales, 64 bytes per 256 elements.
 * Uses iq1s_grid lookup table with delta=0.125f.
 *
 * @param n   Number of elements (must be a multiple of 256).
 * @param vx  Pointer to IQ1_S quantized data (block_iq1_s format).
 * @param vy  Pointer to Q8_K quantized data (block_q8_K format).
 * @return    The dot product as a float.
 */
float iq1_s_q8_k(int n, const void* vx, const void* vy);

/**
 * @brief Compute dot product of IQ1_M quantized vector with Q8_K quantized vector.
 *
 * IQ1_M: 1-bit quantization with extra scale factors, 64 bytes per 256 elements.
 * Uses iq1s_grid lookup table with delta=0.125f.
 *
 * @param n   Number of elements (must be a multiple of 256).
 * @param vx  Pointer to IQ1_M quantized data (block_iq1_m format).
 * @param vy  Pointer to Q8_K quantized data (block_q8_K format).
 * @return    The dot product as a float.
 */
float iq1_m_q8_k(int n, const void* vx, const void* vy);

/**
 * @brief Compute dot product of IQ2_S quantized vector with Q8_K quantized vector.
 *
 * IQ2_S: 2-bit quantization with scales, 128 bytes per 256 elements.
 * Uses iq2s_grid and kmask_iq2xs lookup tables.
 *
 * @param n   Number of elements (must be a multiple of 256).
 * @param vx  Pointer to IQ2_S quantized data (block_iq2_s format).
 * @param vy  Pointer to Q8_K quantized data (block_q8_K format).
 * @return    The dot product as a float.
 */
float iq2_s_q8_k(int n, const void* vx, const void* vy);

/**
 * @brief Compute dot product of IQ3_S quantized vector with Q8_K quantized vector.
 *
 * IQ3_S: 3-bit quantization with scales, 160 bytes per 256 elements.
 * Uses iq3s_grid and kmask_iq2xs lookup tables.
 *
 * @param n   Number of elements (must be a multiple of 256).
 * @param vx  Pointer to IQ3_S quantized data (block_iq3_s format).
 * @param vy  Pointer to Q8_K quantized data (block_q8_K format).
 * @return    The dot product as a float.
 */
float iq3_s_q8_k(int n, const void* vx, const void* vy);

#ifdef __cplusplus
}
#endif

#endif /* IQ_KERNELS_H */