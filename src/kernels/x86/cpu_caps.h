/**
 * @file cpu_caps.h
 * @brief Single source of truth for x86 CPU feature detection.
 *
 * Every C kernel file in src/kernels/x86/ MUST use these probes instead of
 * writing its own cpuid/xgetbv code. The features are probed exactly once, at
 * library load time (via a constructor), so a kernel dispatcher can read a
 * cached int on every call — cpuid/xgetbv costs ~1600 ns on this VM (VM-exit)
 * and must never be on the hot path.
 *
 * All probes return 1 (supported) or 0 (not), and include the OS state check
 * (OSXSAVE + XCR0 bits), so a bare CPUID bit is not enough to report support.
 */

#ifndef CPU_CAPS_H
#define CPU_CAPS_H

#ifdef __cplusplus
extern "C" {
#endif

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
 * @brief Check if the CPU supports AVX-512BW instructions.
 * @return 1 if AVX-512BW is supported, 0 otherwise.
 */
int cpu_has_avx512bw(void);

/**
 * @brief Check if the CPU supports AVX-512VL instructions.
 * @return 1 if AVX-512VL is supported, 0 otherwise.
 */
int cpu_has_avx512vl(void);

/**
 * @brief Check if the CPU supports AVX-512DQ instructions.
 * @return 1 if AVX-512DQ is supported, 0 otherwise.
 */
int cpu_has_avx512dq(void);

/**
 * @brief Check if the CPU supports AVX-512 VNNI (vpdpwssd/vpdpbusd).
 * @return 1 if supported, 0 otherwise. Implies AVX-512F.
 */
int cpu_has_avx512_vnni(void);

/**
 * @brief Check if the CPU supports AVX-512 VBMI.
 * @return 1 if supported, 0 otherwise. Implies AVX-512F.
 */
int cpu_has_avx512_vbmi(void);

/**
 * @brief Check if the CPU supports FMA (leaf 1, ECX bit 12).
 * @return 1 if supported, 0 otherwise.
 */
int cpu_has_fma(void);

/**
 * @brief Check if the CPU supports F16C (leaf 1, ECX bit 29).
 * @return 1 if supported, 0 otherwise.
 */
int cpu_has_f16c(void);

/**
 * @brief Check if the CPU supports SSSE3 (leaf 1, ECX bit 9).
 * @return 1 if supported, 0 otherwise.
 */
int cpu_has_ssse3(void);

/**
 * @brief Get a string describing the CPU features.
 * @return A static string describing detected CPU features.
 */
const char* cpu_get_features_string(void);

#ifdef __cplusplus
}
#endif

#endif /* CPU_CAPS_H */
