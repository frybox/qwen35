/**
 * @file iq_kernels.c
 * @brief IQ quantization kernels with CPU feature detection and runtime dispatch.
 *
 * Ported from llama.cpp: ggml/src/ggml-cpu/arch/x86/quants.c
 */

#include "iq_kernels.h"
#include "generated_tables.h"
#include "cpu_caps.h"
#include <string.h>
#include <stdint.h>
#include <stdio.h>

#if defined(__AVX__) || defined(__AVX2__)
#include <immintrin.h>
#endif

/* CPU feature probes live in cpu_caps.c (shared, probed once at load */
/* time); this file includes cpu_caps.h. Do not re-implement cpuid here. */


// Helper for combining two __m128i into __m256i
#if defined(__AVX__)
#define MM256_SET_M128I(a, b) _mm256_insertf128_si256(_mm256_castsi128_si256(b), a, 1)
#endif

// Scale shuffle helper for IQ kernels
#if defined(__AVX__)
static inline __m128i get_scale_shuffle(int i) {
    static const uint8_t k_shuffle[128] = {
         0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1,
         2, 2, 2, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3, 3, 3,
         4, 4, 4, 4, 4, 4, 4, 4, 5, 5, 5, 5, 5, 5, 5, 5,
         6, 6, 6, 6, 6, 6, 6, 6, 7, 7, 7, 7, 7, 7, 7, 7,
         8, 8, 8, 8, 8, 8, 8, 8, 9, 9, 9, 9, 9, 9, 9, 9,
        10,10,10,10,10,10,10,10, 11,11,11,11,11,11,11,11,
        12,12,12,12,12,12,12,12, 13,13,13,13,13,13,13,13,
        14,14,14,14,14,14,14,14, 15,15,15,15,15,15,15,15
    };
    return _mm_loadu_si128((const __m128i*)k_shuffle + i);
}
#endif

/* ============================================================================
 * Lookup Tables (from llama.cpp)
 * ============================================================================ */

// QK_K: number of elements per block
#define QK_K 256

// IQ2_XXS grid: 256 entries of 8-byte patterns
// Maps 8-bit index to 8-element quantized pattern
static const uint64_t iq2xxs_grid[256] = {
    0x0808080808080808ULL, 0x080808080808082bULL, 0x0808080808081919ULL, 0x0808080808082b08ULL,
    0x0808080808082b2bULL, 0x0808080808190819ULL, 0x0808080808191908ULL, 0x08080808082b0808ULL,
    0x08080808082b082bULL, 0x08080808082b2b08ULL, 0x08080808082b2b2bULL, 0x0808080819080819ULL,
    0x0808080819081908ULL, 0x0808080819190808ULL, 0x0808080819192b08ULL, 0x08080808192b0819ULL,
    0x08080808192b1908ULL, 0x080808082b080808ULL, 0x080808082b08082bULL, 0x080808082b082b2bULL,
    0x080808082b2b082bULL, 0x0808081908080819ULL, 0x0808081908081908ULL, 0x0808081908190808ULL,
    0x0808081908191919ULL, 0x0808081919080808ULL, 0x080808192b081908ULL, 0x080808192b192b08ULL,
    0x0808082b08080808ULL, 0x0808082b0808082bULL, 0x0808082b082b082bULL, 0x0808082b2b08082bULL,
    0x0808190808080819ULL, 0x0808190808081908ULL, 0x0808190808190808ULL, 0x08081908082b0819ULL,
    0x08081908082b1908ULL, 0x0808190819080808ULL, 0x080819081908082bULL, 0x0808190819082b08ULL,
    0x08081908192b0808ULL, 0x080819082b080819ULL, 0x080819082b081908ULL, 0x080819082b190808ULL,
    0x080819082b2b1908ULL, 0x0808191908080808ULL, 0x080819190808082bULL, 0x0808191908082b08ULL,
    0x08081919082b0808ULL, 0x080819191908192bULL, 0x08081919192b2b19ULL, 0x080819192b080808ULL,
    0x080819192b190819ULL, 0x0808192b08082b19ULL, 0x0808192b08190808ULL, 0x0808192b19080808ULL,
    0x0808192b2b081908ULL, 0x0808192b2b2b1908ULL, 0x08082b0808080808ULL, 0x08082b0808081919ULL,
    0x08082b0808082b08ULL, 0x08082b0808191908ULL, 0x08082b08082b2b08ULL, 0x08082b0819080819ULL,
    0x08082b0819081908ULL, 0x08082b0819190808ULL, 0x08082b081919082bULL, 0x08082b082b082b08ULL,
    0x08082b1908081908ULL, 0x08082b1919080808ULL, 0x08082b2b0808082bULL, 0x08082b2b08191908ULL,
    0x0819080808080819ULL, 0x0819080808081908ULL, 0x0819080808190808ULL, 0x08190808082b0819ULL,
    0x0819080819080808ULL, 0x08190808192b0808ULL, 0x081908082b081908ULL, 0x081908082b190808ULL,
    0x081908082b191919ULL, 0x0819081908080808ULL, 0x0819081908082b08ULL, 0x08190819082b0808ULL,
    0x0819081919190808ULL, 0x0819081919192b2bULL, 0x081908192b080808ULL, 0x0819082b082b1908ULL,
    0x0819082b19081919ULL, 0x0819190808080808ULL, 0x0819190808082b08ULL, 0x08191908082b0808ULL,
    0x08191908082b1919ULL, 0x0819190819082b19ULL, 0x081919082b080808ULL, 0x0819191908192b08ULL,
    0x08191919192b082bULL, 0x0819192b08080808ULL, 0x0819192b0819192bULL, 0x08192b0808080819ULL,
    0x08192b0808081908ULL, 0x08192b0808190808ULL, 0x08192b0819080808ULL, 0x08192b082b080819ULL,
    0x08192b1908080808ULL, 0x08192b1908081919ULL, 0x08192b192b2b0808ULL, 0x08192b2b19190819ULL,
    0x082b080808080808ULL, 0x082b08080808082bULL, 0x082b080808082b2bULL, 0x082b080819081908ULL,
    0x082b0808192b0819ULL, 0x082b08082b080808ULL, 0x082b08082b08082bULL, 0x082b0819082b2b19ULL,
    0x082b081919082b08ULL, 0x082b082b08080808ULL, 0x082b082b0808082bULL, 0x082b190808080819ULL,
    0x082b190808081908ULL, 0x082b190808190808ULL, 0x082b190819080808ULL, 0x082b19081919192bULL,
    0x082b191908080808ULL, 0x082b191919080819ULL, 0x082b1919192b1908ULL, 0x082b192b2b190808ULL,
    0x082b2b0808082b08ULL, 0x082b2b08082b0808ULL, 0x082b2b082b191908ULL, 0x082b2b2b19081908ULL,
    0x1908080808080819ULL, 0x1908080808081908ULL, 0x1908080808190808ULL, 0x1908080808192b08ULL,
    0x19080808082b0819ULL, 0x19080808082b1908ULL, 0x1908080819080808ULL, 0x1908080819082b08ULL,
    0x190808081919192bULL, 0x19080808192b0808ULL, 0x190808082b080819ULL, 0x190808082b081908ULL,
    0x190808082b190808ULL, 0x1908081908080808ULL, 0x19080819082b0808ULL, 0x19080819192b0819ULL,
    0x190808192b080808ULL, 0x190808192b081919ULL, 0x1908082b08080819ULL, 0x1908082b08190808ULL,
    0x1908082b19082b08ULL, 0x1908082b1919192bULL, 0x1908082b192b2b08ULL, 0x1908190808080808ULL,
    0x1908190808082b08ULL, 0x19081908082b0808ULL, 0x190819082b080808ULL, 0x190819082b192b19ULL,
    0x190819190819082bULL, 0x19081919082b1908ULL, 0x1908192b08080808ULL, 0x19082b0808080819ULL,
    0x19082b0808081908ULL, 0x19082b0808190808ULL, 0x19082b0819080808ULL, 0x19082b0819081919ULL,
    0x19082b1908080808ULL, 0x19082b1919192b08ULL, 0x19082b19192b0819ULL, 0x19082b192b08082bULL,
    0x19082b2b19081919ULL, 0x19082b2b2b190808ULL, 0x1919080808080808ULL, 0x1919080808082b08ULL,
    0x1919080808190819ULL, 0x1919080808192b19ULL, 0x19190808082b0808ULL, 0x191908082b080808ULL,
    0x191908082b082b08ULL, 0x1919081908081908ULL, 0x191908191908082bULL, 0x191908192b2b1908ULL,
    0x1919082b2b190819ULL, 0x191919082b190808ULL, 0x191919082b19082bULL, 0x1919191908082b2bULL,
    0x1919192b08080819ULL, 0x1919192b19191908ULL, 0x19192b0808080808ULL, 0x19192b0808190819ULL,
    0x19192b0808192b19ULL, 0x19192b08192b1908ULL, 0x19192b1919080808ULL, 0x19192b2b08082b08ULL,
    0x192b080808081908ULL, 0x192b080808190808ULL, 0x192b080819080808ULL, 0x192b0808192b2b08ULL,
    0x192b081908080808ULL, 0x192b081919191919ULL, 0x192b082b08192b08ULL, 0x192b082b192b0808ULL,
    0x192b190808080808ULL, 0x192b190808081919ULL, 0x192b191908190808ULL, 0x192b19190819082bULL,
    0x192b19192b081908ULL, 0x192b2b081908082bULL, 0x2b08080808080808ULL, 0x2b0808080808082bULL,
    0x2b08080808082b2bULL, 0x2b08080819080819ULL, 0x2b0808082b08082bULL, 0x2b08081908081908ULL,
    0x2b08081908192b08ULL, 0x2b08081919080808ULL, 0x2b08082b08190819ULL, 0x2b08190808080819ULL,
    0x2b08190808081908ULL, 0x2b08190808190808ULL, 0x2b08190808191919ULL, 0x2b08190819080808ULL,
    0x2b081908192b0808ULL, 0x2b08191908080808ULL, 0x2b0819191908192bULL, 0x2b0819192b191908ULL,
    0x2b08192b08082b19ULL, 0x2b08192b19080808ULL, 0x2b08192b192b0808ULL, 0x2b082b080808082bULL,
    0x2b082b1908081908ULL, 0x2b082b2b08190819ULL, 0x2b19080808081908ULL, 0x2b19080808190808ULL,
    0x2b190808082b1908ULL, 0x2b19080819080808ULL, 0x2b1908082b2b0819ULL, 0x2b1908190819192bULL,
    0x2b1908192b080808ULL, 0x2b19082b19081919ULL, 0x2b19190808080808ULL, 0x2b191908082b082bULL,
    0x2b19190819081908ULL, 0x2b19191919190819ULL, 0x2b192b082b080819ULL, 0x2b192b19082b0808ULL,
    0x2b2b08080808082bULL, 0x2b2b080819190808ULL, 0x2b2b08082b081919ULL, 0x2b2b081908082b19ULL,
    0x2b2b082b08080808ULL, 0x2b2b190808192b08ULL, 0x2b2b2b0819190808ULL, 0x2b2b2b1908081908ULL,
};

// Signs lookup table for IQ2_XXS/IQ2_XS (scalar version, matches Zig ksigns_iq2xs)
static const uint8_t ksigns_iq2xs[128] = {
    0, 129, 130, 3, 132, 5, 6, 135, 136, 9, 10, 139, 12, 141, 142, 15,
    144, 17, 18, 147, 20, 149, 150, 23, 24, 153, 154, 27, 156, 29, 30, 159,
    160, 33, 34, 163, 36, 165, 166, 39, 40, 169, 170, 43, 172, 45, 46, 175,
    48, 177, 178, 51, 180, 53, 54, 183, 184, 57, 58, 187, 60, 189, 190, 63,
    192, 65, 66, 195, 68, 197, 198, 71, 72, 201, 202, 75, 204, 77, 78, 207,
    80, 209, 210, 83, 212, 85, 86, 215, 216, 89, 90, 219, 92, 221, 222, 95,
    96, 225, 226, 99, 228, 101, 102, 231, 232, 105, 106, 235, 108, 237, 238, 111,
    240, 113, 114, 243, 116, 245, 246, 119, 120, 249, 250, 123, 252, 125, 126, 255,
};

// kmask_iq2xs is now defined in generated_tables.h

// Signs lookup table for IQ2_XXS/IQ2_XS (SIMD version)
#if defined(__AVX__) || defined(__AVX2__)
static const int8_t keven_signs_q2xs[1024] = {
     1,  1,  1,  1,  1,  1,  1,  1, -1,  1,  1,  1,  1,  1,  1, -1,  1, -1,  1,  1,  1,  1,  1, -1, -1, -1,  1,  1,  1,  1,  1,  1,
     1,  1, -1,  1,  1,  1,  1, -1, -1,  1, -1,  1,  1,  1,  1,  1,  1, -1, -1,  1,  1,  1,  1,  1, -1, -1, -1,  1,  1,  1,  1, -1,
     1,  1,  1, -1,  1,  1,  1, -1, -1,  1,  1, -1,  1,  1,  1,  1,  1, -1,  1, -1,  1,  1,  1,  1, -1, -1,  1, -1,  1,  1,  1, -1,
     1,  1, -1, -1,  1,  1,  1,  1, -1,  1, -1, -1,  1,  1,  1, -1,  1, -1, -1, -1,  1,  1,  1, -1, -1, -1, -1, -1,  1,  1,  1,  1,
     1,  1,  1,  1, -1,  1,  1, -1, -1,  1,  1,  1, -1,  1,  1,  1,  1, -1,  1,  1, -1,  1,  1,  1, -1, -1,  1,  1, -1,  1,  1, -1,
     1,  1, -1,  1, -1,  1,  1,  1, -1,  1, -1,  1, -1,  1,  1, -1,  1, -1, -1,  1, -1,  1,  1, -1, -1, -1, -1,  1, -1,  1,  1,  1,
     1,  1,  1, -1, -1,  1,  1,  1, -1,  1,  1, -1, -1,  1,  1, -1,  1, -1,  1, -1, -1,  1,  1, -1, -1, -1,  1, -1, -1,  1,  1,  1,
     1,  1, -1, -1, -1,  1,  1, -1, -1,  1, -1, -1, -1,  1,  1,  1,  1, -1, -1, -1, -1,  1,  1,  1, -1, -1, -1, -1, -1,  1,  1, -1,
     1,  1,  1,  1,  1, -1,  1, -1, -1,  1,  1,  1,  1, -1,  1,  1,  1, -1,  1,  1,  1, -1,  1,  1, -1, -1,  1,  1,  1, -1,  1, -1,
     1,  1, -1,  1,  1, -1,  1,  1, -1,  1, -1,  1,  1, -1,  1, -1,  1, -1, -1,  1,  1, -1,  1, -1, -1, -1, -1,  1,  1, -1,  1,  1,
     1,  1,  1, -1,  1, -1,  1,  1, -1,  1,  1, -1,  1, -1,  1, -1,  1, -1,  1, -1,  1, -1,  1, -1, -1, -1,  1, -1,  1, -1,  1,  1,
     1,  1, -1, -1,  1, -1,  1, -1, -1,  1, -1, -1,  1, -1,  1,  1,  1, -1, -1, -1,  1, -1,  1,  1, -1, -1, -1, -1,  1, -1,  1, -1,
     1,  1,  1,  1, -1, -1,  1,  1, -1,  1,  1,  1, -1, -1,  1, -1,  1, -1,  1,  1, -1, -1,  1, -1, -1, -1,  1,  1, -1, -1,  1,  1,
     1,  1, -1,  1, -1, -1,  1, -1, -1,  1, -1,  1, -1, -1,  1,  1,  1, -1, -1,  1, -1, -1,  1,  1, -1, -1, -1,  1, -1, -1,  1, -1,
     1,  1,  1, -1, -1, -1,  1, -1, -1,  1,  1, -1, -1, -1,  1,  1,  1, -1,  1, -1, -1, -1,  1,  1, -1, -1,  1, -1, -1, -1,  1, -1,
     1,  1, -1, -1, -1, -1,  1,  1, -1,  1, -1, -1, -1, -1,  1, -1,  1, -1, -1, -1, -1, -1,  1, -1, -1, -1, -1, -1, -1, -1,  1,  1,
     1,  1,  1,  1,  1,  1, -1, -1, -1,  1,  1,  1,  1,  1, -1,  1,  1, -1,  1,  1,  1,  1, -1,  1, -1, -1,  1,  1,  1,  1, -1, -1,
     1,  1, -1,  1,  1,  1, -1,  1, -1,  1, -1,  1,  1,  1, -1, -1,  1, -1, -1,  1,  1,  1, -1, -1, -1, -1, -1,  1,  1,  1, -1,  1,
     1,  1,  1, -1,  1,  1, -1,  1, -1,  1,  1, -1,  1,  1, -1, -1,  1, -1,  1, -1,  1,  1, -1, -1, -1, -1,  1, -1,  1,  1, -1,  1,
     1,  1, -1, -1,  1,  1, -1, -1, -1,  1, -1, -1,  1,  1, -1,  1,  1, -1, -1, -1,  1,  1, -1,  1, -1, -1, -1, -1,  1,  1, -1, -1,
     1,  1,  1,  1, -1,  1, -1,  1, -1,  1,  1,  1, -1,  1, -1, -1,  1, -1,  1,  1, -1,  1, -1, -1, -1, -1,  1,  1, -1,  1, -1,  1,
     1,  1, -1,  1, -1,  1, -1, -1, -1,  1, -1,  1, -1,  1, -1,  1,  1, -1, -1,  1, -1,  1, -1,  1, -1, -1, -1,  1, -1,  1, -1, -1,
     1,  1,  1, -1, -1,  1, -1, -1, -1,  1,  1, -1, -1,  1, -1,  1,  1, -1,  1, -1, -1,  1, -1,  1, -1, -1,  1, -1, -1,  1, -1, -1,
     1,  1, -1, -1, -1,  1, -1,  1, -1,  1, -1, -1, -1,  1, -1, -1,  1, -1, -1, -1, -1,  1, -1, -1, -1, -1, -1, -1, -1,  1, -1,  1,
     1,  1,  1,  1,  1, -1, -1,  1, -1,  1,  1,  1,  1, -1, -1, -1,  1, -1,  1,  1,  1, -1, -1, -1, -1, -1,  1,  1,  1, -1, -1,  1,
     1,  1, -1,  1,  1, -1, -1, -1, -1,  1, -1,  1,  1, -1, -1,  1,  1, -1, -1,  1,  1, -1, -1,  1, -1, -1, -1,  1,  1, -1, -1, -1,
     1,  1,  1, -1,  1, -1, -1, -1, -1,  1,  1, -1,  1, -1, -1,  1,  1, -1,  1, -1,  1, -1, -1,  1, -1, -1,  1, -1,  1, -1, -1, -1,
     1,  1, -1, -1,  1, -1, -1,  1, -1,  1, -1, -1,  1, -1, -1, -1,  1, -1, -1, -1,  1, -1, -1, -1, -1, -1, -1, -1,  1, -1, -1,  1,
     1,  1,  1,  1, -1, -1, -1, -1, -1,  1,  1,  1, -1, -1, -1,  1,  1, -1,  1,  1, -1, -1, -1,  1, -1, -1,  1,  1, -1, -1, -1, -1,
     1,  1, -1,  1, -1, -1, -1,  1, -1,  1, -1,  1, -1, -1, -1, -1,  1, -1, -1,  1, -1, -1, -1, -1, -1, -1, -1,  1, -1, -1, -1,  1,
     1,  1,  1, -1, -1, -1, -1,  1, -1,  1,  1, -1, -1, -1, -1, -1,  1, -1,  1, -1, -1, -1, -1, -1, -1, -1,  1, -1, -1, -1, -1,  1,
     1,  1, -1, -1, -1, -1, -1, -1, -1,  1, -1, -1, -1, -1, -1,  1,  1, -1, -1, -1, -1, -1, -1,  1, -1, -1, -1, -1, -1, -1, -1, -1,
};
#endif

// IQ2_XS grid: 512 entries (from llama.cpp)
static const uint64_t iq2xs_grid[512] = {
0x0808080808080808, 0x080808080808082b, 0x0808080808081919, 0x0808080808082b08,
    0x0808080808082b2b, 0x0808080808190819, 0x0808080808191908, 0x080808080819192b,
    0x0808080808192b19, 0x08080808082b0808, 0x08080808082b082b, 0x08080808082b1919,
    0x08080808082b2b08, 0x0808080819080819, 0x0808080819081908, 0x080808081908192b,
    0x0808080819082b19, 0x0808080819190808, 0x080808081919082b, 0x0808080819191919,
    0x0808080819192b08, 0x08080808192b0819, 0x08080808192b1908, 0x080808082b080808,
    0x080808082b08082b, 0x080808082b081919, 0x080808082b082b08, 0x080808082b190819,
    0x080808082b191908, 0x080808082b192b19, 0x080808082b2b0808, 0x0808081908080819,
    0x0808081908081908, 0x080808190808192b, 0x0808081908082b19, 0x0808081908190808,
    0x080808190819082b, 0x0808081908191919, 0x0808081908192b08, 0x0808081908192b2b,
    0x08080819082b0819, 0x08080819082b1908, 0x0808081919080808, 0x080808191908082b,
    0x0808081919081919, 0x0808081919082b08, 0x0808081919190819, 0x0808081919191908,
    0x08080819192b0808, 0x08080819192b2b08, 0x080808192b080819, 0x080808192b081908,
    0x080808192b190808, 0x0808082b08080808, 0x0808082b0808082b, 0x0808082b08081919,
    0x0808082b08082b08, 0x0808082b08190819, 0x0808082b08191908, 0x0808082b082b0808,
    0x0808082b19080819, 0x0808082b19081908, 0x0808082b19190808, 0x0808082b19191919,
    0x0808082b2b080808, 0x0808082b2b082b2b, 0x0808190808080819, 0x0808190808081908,
    0x080819080808192b, 0x0808190808082b19, 0x0808190808190808, 0x080819080819082b,
    0x0808190808191919, 0x0808190808192b08, 0x08081908082b0819, 0x08081908082b1908,
    0x0808190819080808, 0x080819081908082b, 0x0808190819081919, 0x0808190819082b08,
    0x0808190819190819, 0x0808190819191908, 0x080819081919192b, 0x08081908192b0808,
    0x080819082b080819, 0x080819082b081908, 0x080819082b190808, 0x0808191908080808,
    0x080819190808082b, 0x0808191908081919, 0x0808191908082b08, 0x0808191908190819,
    0x0808191908191908, 0x08081919082b0808, 0x0808191919080819, 0x0808191919081908,
    0x0808191919190808, 0x08081919192b0819, 0x080819192b080808, 0x0808192b08080819,
    0x0808192b08081908, 0x0808192b08190808, 0x0808192b082b192b, 0x0808192b19080808,
    0x0808192b1908082b, 0x0808192b2b081908, 0x08082b0808080808, 0x08082b080808082b,
    0x08082b0808081919, 0x08082b0808082b08, 0x08082b0808082b2b, 0x08082b0808190819,
    0x08082b0808191908, 0x08082b08082b0808, 0x08082b08082b1919, 0x08082b0819080819,
    0x08082b0819081908, 0x08082b0819190808, 0x08082b0819192b08, 0x08082b082b080808,
    0x08082b082b2b0808, 0x08082b082b2b2b2b, 0x08082b1908080819, 0x08082b1908081908,
    0x08082b1908190808, 0x08082b1919080808, 0x08082b192b080819, 0x08082b192b082b19,
    0x08082b2b08080808, 0x08082b2b082b0808, 0x08082b2b082b2b08, 0x08082b2b2b19192b,
    0x08082b2b2b2b0808, 0x0819080808080819, 0x0819080808081908, 0x081908080808192b,
    0x0819080808082b19, 0x0819080808190808, 0x081908080819082b, 0x0819080808191919,
    0x0819080808192b08, 0x08190808082b0819, 0x08190808082b1908, 0x0819080819080808,
    0x081908081908082b, 0x0819080819081919, 0x0819080819082b08, 0x0819080819190819,
    0x0819080819191908, 0x08190808192b0808, 0x08190808192b2b2b, 0x081908082b080819,
    0x081908082b081908, 0x081908082b190808, 0x0819081908080808, 0x081908190808082b,
    0x0819081908081919, 0x0819081908082b08, 0x0819081908190819, 0x0819081908191908,
    0x08190819082b0808, 0x0819081919080819, 0x0819081919081908, 0x0819081919190808,
    0x081908192b080808, 0x081908192b191908, 0x081908192b19192b, 0x0819082b08080819,
    0x0819082b08081908, 0x0819082b0808192b, 0x0819082b08190808, 0x0819082b19080808,
    0x0819082b192b0808, 0x0819190808080808, 0x081919080808082b, 0x0819190808081919,
    0x0819190808082b08, 0x0819190808190819, 0x0819190808191908, 0x08191908082b0808,
    0x0819190819080819, 0x0819190819081908, 0x0819190819082b19, 0x0819190819190808,
    0x08191908192b1908, 0x081919082b080808, 0x0819191908080819, 0x0819191908081908,
    0x0819191908190808, 0x0819191919080808, 0x0819192b08080808, 0x0819192b08191908,
    0x0819192b19082b19, 0x08192b0808080819, 0x08192b0808081908, 0x08192b0808190808,
    0x08192b080819082b, 0x08192b0819080808, 0x08192b0819191908, 0x08192b082b08192b,
    0x08192b1908080808, 0x08192b1908081919, 0x08192b19192b192b, 0x08192b2b19190819,
    0x08192b2b2b2b2b19, 0x082b080808080808, 0x082b08080808082b, 0x082b080808081919,
    0x082b080808082b08, 0x082b080808082b2b, 0x082b080808190819, 0x082b080808191908,
    0x082b0808082b0808, 0x082b080819080819, 0x082b080819081908, 0x082b080819190808,
    0x082b08082b080808, 0x082b08082b2b0808, 0x082b081908080819, 0x082b081908081908,
    0x082b081908190808, 0x082b081919080808, 0x082b081919082b08, 0x082b0819192b1919,
    0x082b082b08080808, 0x082b082b082b082b, 0x082b082b2b080808, 0x082b082b2b2b2b08,
    0x082b190808080819, 0x082b190808081908, 0x082b190808190808, 0x082b1908082b2b19,
    0x082b190819080808, 0x082b191908080808, 0x082b191919080819, 0x082b19191919082b,
    0x082b19192b192b19, 0x082b192b08080819, 0x082b192b08192b2b, 0x082b192b2b2b192b,
    0x082b2b0808080808, 0x082b2b0808082b08, 0x082b2b0808082b2b, 0x082b2b08082b0808,
    0x082b2b0819191919, 0x082b2b082b082b08, 0x082b2b082b2b082b, 0x082b2b19192b2b08,
    0x082b2b192b190808, 0x082b2b2b08082b08, 0x082b2b2b082b0808, 0x082b2b2b2b08082b,
    0x082b2b2b2b082b08, 0x082b2b2b2b082b2b, 0x1908080808080819, 0x1908080808081908,
    0x190808080808192b, 0x1908080808082b19, 0x1908080808190808, 0x190808080819082b,
    0x1908080808191919, 0x1908080808192b08, 0x19080808082b0819, 0x19080808082b1908,
    0x1908080819080808, 0x190808081908082b, 0x1908080819081919, 0x1908080819082b08,
    0x1908080819082b2b, 0x1908080819190819, 0x1908080819191908, 0x19080808192b0808,
    0x19080808192b1919, 0x190808082b080819, 0x190808082b081908, 0x190808082b190808,
    0x1908081908080808, 0x190808190808082b, 0x1908081908081919, 0x1908081908082b08,
    0x1908081908190819, 0x1908081908191908, 0x19080819082b0808, 0x1908081919080819,
    0x1908081919081908, 0x1908081919190808, 0x190808192b080808, 0x190808192b081919,
    0x190808192b2b082b, 0x1908082b08080819, 0x1908082b08081908, 0x1908082b08190808,
    0x1908082b0819082b, 0x1908082b082b2b19, 0x1908082b19080808, 0x1908190808080808,
    0x190819080808082b, 0x1908190808081919, 0x1908190808082b08, 0x1908190808190819,
    0x1908190808191908, 0x1908190808192b19, 0x19081908082b0808, 0x1908190819080819,
    0x1908190819081908, 0x1908190819190808, 0x190819082b080808, 0x190819082b191908,
    0x1908191908080819, 0x1908191908081908, 0x1908191908190808, 0x19081919082b1908,
    0x1908191919080808, 0x190819192b192b2b, 0x1908192b08080808, 0x1908192b08082b2b,
    0x1908192b19081908, 0x1908192b19190808, 0x19082b0808080819, 0x19082b0808081908,
    0x19082b0808190808, 0x19082b0819080808, 0x19082b0819081919, 0x19082b0819191908,
    0x19082b08192b082b, 0x19082b1908080808, 0x19082b1908190819, 0x19082b1919081908,
    0x19082b1919190808, 0x19082b19192b2b19, 0x19082b2b08081908, 0x1919080808080808,
    0x191908080808082b, 0x1919080808081919, 0x1919080808082b08, 0x1919080808190819,
    0x1919080808191908, 0x19190808082b0808, 0x19190808082b2b08, 0x1919080819080819,
    0x1919080819081908, 0x1919080819190808, 0x191908082b080808, 0x1919081908080819,
    0x1919081908081908, 0x1919081908190808, 0x1919081908191919, 0x1919081919080808,
    0x191908191908082b, 0x1919082b08080808, 0x1919082b19081908, 0x1919082b2b2b2b2b,
    0x1919190808080819, 0x1919190808081908, 0x1919190808190808, 0x19191908082b0819,
    0x1919190819080808, 0x19191908192b0808, 0x191919082b080819, 0x191919082b2b0819,
    0x1919191908080808, 0x1919191908082b08, 0x191919192b080808, 0x191919192b082b08,
    0x1919192b082b0819, 0x1919192b192b2b08, 0x1919192b2b2b0819, 0x19192b0808080808,
    0x19192b0808191908, 0x19192b0819080819, 0x19192b0819190808, 0x19192b082b192b19,
    0x19192b1908192b2b, 0x19192b1919080808, 0x19192b191908082b, 0x19192b2b2b081919,
    0x192b080808080819, 0x192b080808081908, 0x192b080808190808, 0x192b080819080808,
    0x192b080819191908, 0x192b0808192b082b, 0x192b08082b08192b, 0x192b08082b2b2b19,
    0x192b081908080808, 0x192b082b082b1908, 0x192b082b19082b2b, 0x192b082b2b19082b,
    0x192b190808080808, 0x192b19080819192b, 0x192b191908190808, 0x192b191919080808,
    0x192b191919081919, 0x192b19192b2b1908, 0x192b2b0808080819, 0x192b2b08192b2b2b,
    0x192b2b19082b1919, 0x192b2b2b0808192b, 0x192b2b2b19191908, 0x192b2b2b192b082b,
    0x2b08080808080808, 0x2b0808080808082b, 0x2b08080808081919, 0x2b08080808082b08,
    0x2b08080808190819, 0x2b08080808191908, 0x2b080808082b0808, 0x2b080808082b2b2b,
    0x2b08080819080819, 0x2b08080819081908, 0x2b08080819190808, 0x2b0808082b080808,
    0x2b0808082b08082b, 0x2b0808082b2b2b08, 0x2b0808082b2b2b2b, 0x2b08081908080819,
    0x2b08081908081908, 0x2b0808190808192b, 0x2b08081908190808, 0x2b08081919080808,
    0x2b08081919190819, 0x2b08081919192b19, 0x2b08082b08080808, 0x2b08082b082b0808,
    0x2b08082b2b080808, 0x2b08082b2b08082b, 0x2b08082b2b2b0808, 0x2b08082b2b2b2b08,
    0x2b08190808080819, 0x2b08190808081908, 0x2b08190808190808, 0x2b0819080819082b,
    0x2b08190808191919, 0x2b08190819080808, 0x2b081908192b0808, 0x2b0819082b082b19,
    0x2b08191908080808, 0x2b08191919081908, 0x2b0819192b2b1919, 0x2b08192b08192b08,
    0x2b08192b192b2b2b, 0x2b082b0808080808, 0x2b082b0808082b08, 0x2b082b08082b1919,
    0x2b082b0819192b2b, 0x2b082b082b080808, 0x2b082b082b08082b, 0x2b082b082b2b2b08,
    0x2b082b190808192b, 0x2b082b2b082b082b, 0x2b082b2b2b080808, 0x2b082b2b2b082b08,
    0x2b082b2b2b19192b, 0x2b082b2b2b2b2b08, 0x2b19080808080819, 0x2b19080808081908,
    0x2b19080808190808, 0x2b19080819080808, 0x2b1908081919192b, 0x2b1908082b081908,
    0x2b19081908080808, 0x2b190819082b082b, 0x2b190819192b1908, 0x2b19082b1919192b,
    0x2b19082b2b082b19, 0x2b19190808080808, 0x2b19190808081919, 0x2b19190819081908,
    0x2b19190819190808, 0x2b19190819192b08, 0x2b191919082b2b19, 0x2b1919192b190808,
    0x2b1919192b19082b, 0x2b19192b19080819, 0x2b192b0819190819, 0x2b192b082b2b192b,
    0x2b192b1919082b19, 0x2b192b2b08191919, 0x2b192b2b192b0808, 0x2b2b080808080808,
    0x2b2b08080808082b, 0x2b2b080808082b08, 0x2b2b080808082b2b, 0x2b2b0808082b0808,
    0x2b2b0808082b2b2b, 0x2b2b08082b2b0808, 0x2b2b081919190819, 0x2b2b081919192b19,
    0x2b2b08192b2b192b, 0x2b2b082b08080808, 0x2b2b082b0808082b, 0x2b2b082b08082b08,
    0x2b2b082b082b2b2b, 0x2b2b082b2b080808, 0x2b2b082b2b2b0808, 0x2b2b190819080808,
    0x2b2b19082b191919, 0x2b2b192b192b1919, 0x2b2b192b2b192b08, 0x2b2b2b0808082b2b,
    0x2b2b2b08082b0808, 0x2b2b2b08082b082b, 0x2b2b2b08082b2b08, 0x2b2b2b082b2b0808,
    0x2b2b2b082b2b2b08, 0x2b2b2b1908081908, 0x2b2b2b192b081908, 0x2b2b2b192b08192b,
    0x2b2b2b2b082b2b08, 0x2b2b2b2b082b2b2b, 0x2b2b2b2b2b190819, 0x2b2b2b2b2b2b2b2b,
};

// kvalues_iq4nl: 4-bit quantization values for IQ4 types
static const int8_t kvalues_iq4nl[16] = {
    -127, -104, -83, -65, -49, -35, -22, -10, 1, 13, 25, 38, 53, 69, 89, 113,
};

/* ============================================================================
 * Block structures (from llama.cpp)
 * ============================================================================ */

// block_iq2_xxs: 2.0625 bpw
// Layout: d(2) @0; qs[32]u16 @2 (64B) = 66 bytes total
// 256 weights encoded into 64 bytes (32 uint16_t values)
typedef struct {
    uint16_t d;       // fp16 scale
    uint16_t qs[32];  // quantized data: 32 uint16_t values (64 bytes)
} block_iq2_xxs;

// block_q8_K: 8-bit quantized weights with per-block scale
// Layout: d(4) @0; qs[256] @4..259; bsums[16] @260..291 = 292 bytes
typedef struct {
    float d;           // scale (4 bytes)
    int8_t qs[QK_K];   // quantized weights (256 values, 256 bytes)
    int16_t bsums[16]; // sum of quants in groups of 16 (32 bytes)
} block_q8_K;

// block_iq2_xs: 2.3125 bpw
// 256 weights with 9-bit grid indices and per-32-element scales
typedef struct {
    uint16_t d;        // fp16 scale
    uint16_t qs[QK_K/8]; // quantized data: 32 uint16_t = 64 bytes, 9-bit indices
    uint8_t scales[QK_K/32]; // 8 scales for 8 blocks of 32 elements
} block_iq2_xs;

// block_iq4_xs: 4.5 bpw
// 256 weights with 4-bit quantization and per-32-element scales
typedef struct {
    uint16_t d;         // fp16 scale
    uint16_t scales_h;  // high bits of scales
    uint8_t  scales_l[QK_K/64]; // low bits of scales (4 values)
    uint8_t  qs[QK_K/2]; // quantized weights (4-bit packed, 128 bytes)
} block_iq4_xs;

// block_iq1_s: 1.5625 bpw
// 1-bit quantization with separate scales, 64 bytes per 256 elements
typedef struct {
    uint16_t d;        // fp16 scale
    uint8_t  qs[QK_K/8];      // grid index, low 8 bits (32 bytes)
    uint16_t qh[QK_K/32];     // grid index, high 3 bits + grid shift bit (16 bytes)
} block_iq1_s;

// block_iq1_m: 1.75 bpw
// 1-bit quantization with extra scale factors
typedef struct {
    uint8_t  qs[QK_K/8];      // grid index, low 8 bits (32 bytes)
    uint8_t  qh[QK_K/16];     // grid index, high 3 bits + grid shift bit (16 bytes)
    uint8_t  scales[QK_K/32]; // 3-bit block scales (8 bytes)
} block_iq1_m;

// block_iq2_s: 2.5625 bpw
// 2-bit quantization with scales
typedef struct {
    uint16_t d;        // fp16 scale
    uint8_t qs[QK_K/4];       // quantized data (64 bytes)
    uint8_t qh[QK_K/32];      // high bits (8 bytes)
    uint8_t scales[QK_K/32]; // scales (8 bytes)
} block_iq2_s;

// block_iq3_s: 3.4375 bpw
// 3-bit quantization with signs and scales
typedef struct {
    uint16_t d;        // fp16 scale
    uint8_t qs[QK_K/4];       // quantized data (64 bytes)
    uint8_t qh[QK_K/32];      // high bits (8 bytes)
    uint8_t signs[QK_K/8];    // sign bits (32 bytes)
    uint8_t scales[QK_K/64];  // scales (4 bytes)
} block_iq3_s;

/* ============================================================================
 * Helper functions
 * ============================================================================ */

// mul_add_epi8: signed multiply-add for 8-bit integers
// Equivalent to: _mm256_maddubs_epi16(_mm256_sign_epi8(x, x), _mm256_sign_epi8(y, x))
#if defined(__AVX2__)
static inline __m256i mul_add_epi8(const __m256i x, const __m256i y) {
    const __m256i ax = _mm256_sign_epi8(x, x);
    const __m256i sy = _mm256_sign_epi8(y, x);
    return _mm256_maddubs_epi16(ax, sy);
}
#endif

#if defined(__AVX__)
static inline __m128i mul_add_epi8_sse(const __m128i x, const __m128i y) {
    const __m128i ax = _mm_sign_epi8(x, x);
    const __m128i sy = _mm_sign_epi8(y, x);
    return _mm_maddubs_epi16(ax, sy);
}
#endif

// Convert fp16 to fp32
static inline float fp16_to_fp32(uint16_t h) {
    // Simple conversion (not handling all corner cases)
    uint32_t sign = ((uint32_t)h & 0x8000u) << 16;
    uint32_t exponent = (h & 0x7C00) >> 10;
    uint32_t mantissa = (h & 0x03FF);
    uint32_t result;
    
    if (exponent == 0) {
        if (mantissa == 0) {
            // Zero
            result = sign;
        } else {
            // Denormalized
            while (!(mantissa & 0x400)) {
                mantissa <<= 1;
                exponent--;
            }
            exponent++;
            mantissa &= 0x3FF;
            exponent += 112; // Bias adjustment
            result = sign | (exponent << 23) | (mantissa << 13);
        }
    } else if (exponent == 31) {
        // Inf or NaN
        result = sign | 0x7F800000 | (mantissa << 13);
    } else {
        exponent += 112; // Bias adjustment
        result = sign | (exponent << 23) | (mantissa << 13);
    }
    
    float f;
    memcpy(&f, &result, sizeof(f));
    return f;
}

// Horizontal sum of 8 floats in __m256
#if defined(__AVX__)
static inline float hsum_float_8_avx(__m256 x) {
    __m128 hi = _mm256_extractf128_ps(x, 1);
    __m128 lo = _mm256_castps256_ps128(x);
    __m128 sum = _mm_add_ps(lo, hi);
    __m128 shuf = _mm_movehdup_ps(sum);
    __m128 sums = _mm_add_ps(sum, shuf);
    __m128 sums2 = _mm_movehl_ps(shuf, sums);
    return _mm_cvtss_f32(_mm_add_ss(sums, sums2));
}
#endif

/* ============================================================================
 * IQ2_XXS implementation - AVX2 version
 * ============================================================================ */

#if defined(__AVX2__)
__attribute__((target("avx2")))
static float iq2_xxs_q8_k_avx2_impl(int n, const block_iq2_xxs* x, const block_q8_K* y) {
    const int nb = n / QK_K;
    const uint64_t* signs64 = (const uint64_t*)keven_signs_q2xs;
    
    uint32_t aux32[4];
    const uint8_t* aux8 = (const uint8_t*)aux32;
    
    __m256 accumf = _mm256_setzero_ps();
    
    for (int i = 0; i < nb; ++i) {
        const float d = fp16_to_fp32(x[i].d) * y[i].d;
        const uint16_t* q2 = x[i].qs;
        const int8_t* q8 = y[i].qs;
        
        __m256i sumi1 = _mm256_setzero_si256();
        __m256i sumi2 = _mm256_setzero_si256();
        
        for (int ib32 = 0; ib32 < QK_K/32; ib32 += 2) {
            const __m256i q8_1 = _mm256_loadu_si256((const __m256i*)q8); q8 += 32;
            const __m256i q8_2 = _mm256_loadu_si256((const __m256i*)q8); q8 += 32;
            
            memcpy(aux32, q2, 4*sizeof(uint32_t)); q2 += 8;
            
            const __m256i q2_1 = _mm256_set_epi64x(
                iq2xxs_grid[aux8[3]], iq2xxs_grid[aux8[2]],
                iq2xxs_grid[aux8[1]], iq2xxs_grid[aux8[0]]);
            const __m256i q2_2 = _mm256_set_epi64x(
                iq2xxs_grid[aux8[11]], iq2xxs_grid[aux8[10]],
                iq2xxs_grid[aux8[9]], iq2xxs_grid[aux8[8]]);
            
            const __m256i s2_1 = _mm256_set_epi64x(
                signs64[(aux32[1] >> 21) & 127], signs64[(aux32[1] >> 14) & 127],
                signs64[(aux32[1] >> 7) & 127], signs64[(aux32[1] >> 0) & 127]);
            const __m256i s2_2 = _mm256_set_epi64x(
                signs64[(aux32[3] >> 21) & 127], signs64[(aux32[3] >> 14) & 127],
                signs64[(aux32[3] >> 7) & 127], signs64[(aux32[3] >> 0) & 127]);
            
            const __m256i q8s_1 = _mm256_sign_epi8(q8_1, s2_1);
            const __m256i q8s_2 = _mm256_sign_epi8(q8_2, s2_2);
            
            const __m256i dot1 = _mm256_maddubs_epi16(q2_1, q8s_1);
            const __m256i dot2 = _mm256_maddubs_epi16(q2_2, q8s_2);
            
            const uint16_t ls1 = aux32[1] >> 28;
            const uint16_t ls2 = aux32[3] >> 28;
            
            // Original implementation from llama.cpp
            const __m256i p1 = _mm256_madd_epi16(dot1, _mm256_set1_epi16(2*ls1+1));
            const __m256i p2 = _mm256_madd_epi16(dot2, _mm256_set1_epi16(2*ls2+1));
            
            sumi1 = _mm256_add_epi32(sumi1, p1);
            sumi2 = _mm256_add_epi32(sumi2, p2);
        }
        
        accumf = _mm256_fmadd_ps(
            _mm256_set1_ps(d),
            _mm256_cvtepi32_ps(_mm256_add_epi32(sumi1, sumi2)),
            accumf);
    }
    
    return 0.125f * hsum_float_8_avx(accumf);
}
#endif

/* ============================================================================
 * IQ2_XXS implementation - AVX version
 * ============================================================================ */

#if defined(__AVX__)
__attribute__((target("avx")))
static float iq2_xxs_q8_k_avx_impl(int n, const block_iq2_xxs* x, const block_q8_K* y) {
    const int nb = n / QK_K;
    const uint64_t* signs64 = (const uint64_t*)keven_signs_q2xs;
    
    uint32_t aux32[4];
    const uint8_t* aux8 = (const uint8_t*)aux32;
    
    __m256 accumf = _mm256_setzero_ps();
    
    for (int i = 0; i < nb; ++i) {
        const float d = fp16_to_fp32(x[i].d) * y[i].d;
        const uint16_t* q2 = x[i].qs;
        const int8_t* q8 = y[i].qs;
        
        __m128i sumi1_0 = _mm_setzero_si128();
        __m128i sumi1_1 = _mm_setzero_si128();
        __m128i sumi2_0 = _mm_setzero_si128();
        __m128i sumi2_1 = _mm_setzero_si128();
        
        for (int ib32 = 0; ib32 < QK_K/32; ib32 += 2) {
            const __m128i q8_1_0 = _mm_loadu_si128((const __m128i*)q8); q8 += 16;
            const __m128i q8_1_1 = _mm_loadu_si128((const __m128i*)q8); q8 += 16;
            const __m128i q8_2_0 = _mm_loadu_si128((const __m128i*)q8); q8 += 16;
            const __m128i q8_2_1 = _mm_loadu_si128((const __m128i*)q8); q8 += 16;
            
            memcpy(aux32, q2, 4*sizeof(uint32_t)); q2 += 8;
            
            const __m128i q2_1_0 = _mm_set_epi64x(iq2xxs_grid[aux8[1]], iq2xxs_grid[aux8[0]]);
            const __m128i q2_1_1 = _mm_set_epi64x(iq2xxs_grid[aux8[3]], iq2xxs_grid[aux8[2]]);
            const __m128i q2_2_0 = _mm_set_epi64x(iq2xxs_grid[aux8[9]], iq2xxs_grid[aux8[8]]);
            const __m128i q2_2_1 = _mm_set_epi64x(iq2xxs_grid[aux8[11]], iq2xxs_grid[aux8[10]]);
            
            const __m128i s2_1_0 = _mm_set_epi64x(signs64[(aux32[1] >> 7) & 127], signs64[(aux32[1] >> 0) & 127]);
            const __m128i s2_1_1 = _mm_set_epi64x(signs64[(aux32[1] >> 21) & 127], signs64[(aux32[1] >> 14) & 127]);
            const __m128i s2_2_0 = _mm_set_epi64x(signs64[(aux32[3] >> 7) & 127], signs64[(aux32[3] >> 0) & 127]);
            const __m128i s2_2_1 = _mm_set_epi64x(signs64[(aux32[3] >> 21) & 127], signs64[(aux32[3] >> 14) & 127]);
            
            const __m128i q8s_1_0 = _mm_sign_epi8(q8_1_0, s2_1_0);
            const __m128i q8s_1_1 = _mm_sign_epi8(q8_1_1, s2_1_1);
            const __m128i q8s_2_0 = _mm_sign_epi8(q8_2_0, s2_2_0);
            const __m128i q8s_2_1 = _mm_sign_epi8(q8_2_1, s2_2_1);
            
            const __m128i dot1_0 = _mm_maddubs_epi16(q2_1_0, q8s_1_0);
            const __m128i dot1_1 = _mm_maddubs_epi16(q2_1_1, q8s_1_1);
            const __m128i dot2_0 = _mm_maddubs_epi16(q2_2_0, q8s_2_0);
            const __m128i dot2_1 = _mm_maddubs_epi16(q2_2_1, q8s_2_1);
            
            const uint16_t ls1 = aux32[1] >> 28;
            const uint16_t ls2 = aux32[3] >> 28;
            
            const __m128i p1_0 = _mm_madd_epi16(dot1_0, _mm_set1_epi16(2*ls1+1));
            const __m128i p1_1 = _mm_madd_epi16(dot1_1, _mm_set1_epi16(2*ls1+1));
            const __m128i p2_0 = _mm_madd_epi16(dot2_0, _mm_set1_epi16(2*ls2+1));
            const __m128i p2_1 = _mm_madd_epi16(dot2_1, _mm_set1_epi16(2*ls2+1));
            
            sumi1_0 = _mm_add_epi32(sumi1_0, p1_0);
            sumi1_1 = _mm_add_epi32(sumi1_1, p1_1);
            sumi2_0 = _mm_add_epi32(sumi2_0, p2_0);
            sumi2_1 = _mm_add_epi32(sumi2_1, p2_1);
        }
        
        __m128i sumi1 = _mm_add_epi32(sumi1_0, sumi1_1);
        __m128i sumi2 = _mm_add_epi32(sumi2_0, sumi2_1);
        __m128i sumi = _mm_add_epi32(sumi1, sumi2);
        
        // Convert to float and accumulate
        // sumi contains 4 int32 values, we need to sum them and multiply by d
        __m128 sum_ps = _mm_cvtepi32_ps(sumi);
        __m128 sum_ps_shuf = _mm_shuffle_ps(sum_ps, sum_ps, _MM_SHUFFLE(1, 0, 3, 2));
        __m128 sum_ps_add = _mm_add_ps(sum_ps, sum_ps_shuf);
        __m128 sum_ps_shuf2 = _mm_shuffle_ps(sum_ps_add, sum_ps_add, _MM_SHUFFLE(0, 1, 0, 1));
        __m128 sum_ps_final = _mm_add_ps(sum_ps_add, sum_ps_shuf2);
        float sum_f = _mm_cvtss_f32(sum_ps_final);
        accumf = _mm256_add_ps(_mm256_set1_ps(d * sum_f), accumf);
    }
    
    return 0.125f * hsum_float_8_avx(accumf);
}
#endif

/* ============================================================================
 * IQ2_XXS implementation - Scalar fallback
 * ============================================================================ */

static float iq2_xxs_q8_k_scalar_impl(int n, const block_iq2_xxs* x, const block_q8_K* y) {
    const int nb = n / QK_K;
    float sum = 0.0f;
    
    for (int i = 0; i < nb; ++i) {
        const float d = fp16_to_fp32(x[i].d) * y[i].d;
        const uint8_t* q2 = (const uint8_t*)x[i].qs;
        const int8_t* q8 = y[i].qs;
        
        // Process 8 groups of 32 values (256 total)
        for (int ib32 = 0; ib32 < 8; ++ib32) {
            // Read 8 bytes for this group
            const uint8_t* a = q2 + ib32 * 8;
            // aux1 is bytes 4-7 (last 4 bytes of the 8-byte group)
            uint32_t aux1 = (uint32_t)a[4] | ((uint32_t)a[5] << 8) | 
                           ((uint32_t)a[6] << 16) | ((uint32_t)a[7] << 24);
            // ls is upper 4 bits of aux1
            float db = d * (0.5f + (float)(aux1 >> 28)) * 0.25f;
            
            // Process 4 sub-blocks of 8 values each
            for (int l = 0; l < 4; ++l) {
                // Get grid index (8-bit value from iq2xxs_grid)
                uint64_t grid_val = iq2xxs_grid[a[l]];
                // Get signs from aux1 (7-bit index)
                uint8_t signs = ksigns_iq2xs[(aux1 >> (7 * l)) & 127];
                
                // Process 8 values
                for (int j = 0; j < 8; ++j) {
                    // Extract grid value (8-bit)
                    int8_t grid_byte = (int8_t)((grid_val >> (8 * j)) & 0xFF);
                    // Apply sign
                    int sign_bit = (signs >> j) & 1;
                    if (sign_bit) grid_byte = -grid_byte;
                    // Accumulate dot product with db scale
                    sum += db * (float)grid_byte * (float)q8[ib32 * 32 + l * 8 + j];
                }
            }
        }
    }
    
    return sum;
}

/* ============================================================================
 * Dispatch function
 * ============================================================================ */

float iq2_xxs_q8_k(int n, const void* vx, const void* vy) {
    const block_iq2_xxs* x = (const block_iq2_xxs*)vx;
    const block_q8_K* y = (const block_q8_K*)vy;
    
#if defined(__AVX2__)
    if (cpu_has_avx2()) {
        return iq2_xxs_q8_k_avx2_impl(n, x, y);
    }
#endif
#if defined(__AVX__)
    // AVX version is always available if compiled with AVX support
    return iq2_xxs_q8_k_avx_impl(n, x, y);
#else
    return iq2_xxs_q8_k_scalar_impl(n, x, y);
#endif
}

/* ============================================================================
 * Placeholder implementations for other IQ types
 * ============================================================================ */

// IQ3_XXS grid: 256 entries of 4-byte patterns
static const uint32_t iq3xxs_grid[256] = {
    0x04040404, 0x04040414, 0x04040424, 0x04040c0c, 0x04040c1c, 0x04040c3e, 0x04041404, 0x04041414,
    0x04041c0c, 0x04042414, 0x04043e1c, 0x04043e2c, 0x040c040c, 0x040c041c, 0x040c0c04, 0x040c0c14,
    0x040c140c, 0x040c142c, 0x040c1c04, 0x040c1c14, 0x040c240c, 0x040c2c24, 0x040c3e04, 0x04140404,
    0x04140414, 0x04140424, 0x04140c0c, 0x04141404, 0x04141414, 0x04141c0c, 0x04141c1c, 0x04141c3e,
    0x04142c0c, 0x04142c3e, 0x04143e2c, 0x041c040c, 0x041c043e, 0x041c0c04, 0x041c0c14, 0x041c142c,
    0x041c3e04, 0x04240c1c, 0x04241c3e, 0x04242424, 0x04242c3e, 0x04243e1c, 0x04243e2c, 0x042c040c,
    0x042c043e, 0x042c1c14, 0x042c2c14, 0x04341c2c, 0x04343424, 0x043e0c04, 0x043e0c24, 0x043e0c34,
    0x043e241c, 0x043e340c, 0x0c04040c, 0x0c04041c, 0x0c040c04, 0x0c040c14, 0x0c04140c, 0x0c04141c,
    0x0c041c04, 0x0c041c14, 0x0c041c24, 0x0c04243e, 0x0c042c04, 0x0c0c0404, 0x0c0c0414, 0x0c0c0c0c,
    0x0c0c1404, 0x0c0c1414, 0x0c14040c, 0x0c14041c, 0x0c140c04, 0x0c140c14, 0x0c14140c, 0x0c141c04,
    0x0c143e14, 0x0c1c0404, 0x0c1c0414, 0x0c1c1404, 0x0c1c1c0c, 0x0c1c2434, 0x0c1c3434, 0x0c24040c,
    0x0c24042c, 0x0c242c04, 0x0c2c1404, 0x0c2c1424, 0x0c2c2434, 0x0c2c3e0c, 0x0c34042c, 0x0c3e1414,
    0x0c3e2404, 0x14040404, 0x14040414, 0x14040c0c, 0x14040c1c, 0x14041404, 0x14041414, 0x14041434,
    0x14041c0c, 0x14042414, 0x140c040c, 0x140c041c, 0x140c042c, 0x140c0c04, 0x140c0c14, 0x140c140c,
    0x140c1c04, 0x140c341c, 0x140c343e, 0x140c3e04, 0x14140404, 0x14140414, 0x14140c0c, 0x14140c3e,
    0x14141404, 0x14141414, 0x14141c3e, 0x14142404, 0x14142c2c, 0x141c040c, 0x141c0c04, 0x141c0c24,
    0x141c3e04, 0x141c3e24, 0x14241c2c, 0x14242c1c, 0x142c041c, 0x142c143e, 0x142c240c, 0x142c3e24,
    0x143e040c, 0x143e041c, 0x143e0c34, 0x143e242c, 0x1c04040c, 0x1c040c04, 0x1c040c14, 0x1c04140c,
    0x1c04141c, 0x1c042c04, 0x1c04342c, 0x1c043e14, 0x1c0c0404, 0x1c0c0414, 0x1c0c1404, 0x1c0c1c0c,
    0x1c0c2424, 0x1c0c2434, 0x1c14040c, 0x1c14041c, 0x1c140c04, 0x1c14142c, 0x1c142c14, 0x1c143e14,
    0x1c1c0c0c, 0x1c1c1c1c, 0x1c241c04, 0x1c24243e, 0x1c243e14, 0x1c2c0404, 0x1c2c0434, 0x1c2c1414,
    0x1c2c2c2c, 0x1c340c24, 0x1c341c34, 0x1c34341c, 0x1c3e1c1c, 0x1c3e3404, 0x24040424, 0x24040c3e,
    0x24041c2c, 0x24041c3e, 0x24042c1c, 0x24042c3e, 0x240c3e24, 0x24141404, 0x24141c3e, 0x24142404,
    0x24143404, 0x24143434, 0x241c043e, 0x241c242c, 0x24240424, 0x24242c0c, 0x24243424, 0x242c142c,
    0x242c241c, 0x242c3e04, 0x243e042c, 0x243e0c04, 0x243e0c14, 0x243e1c04, 0x2c040c14, 0x2c04240c,
    0x2c043e04, 0x2c0c0404, 0x2c0c0434, 0x2c0c1434, 0x2c0c2c2c, 0x2c140c24, 0x2c141c14, 0x2c143e14,
    0x2c1c0414, 0x2c1c2c1c, 0x2c240c04, 0x2c24141c, 0x2c24143e, 0x2c243e14, 0x2c2c0414, 0x2c2c1c0c,
    0x2c342c04, 0x2c3e1424, 0x2c3e2414, 0x34041424, 0x34042424, 0x34042434, 0x34043424, 0x340c140c,
    0x340c340c, 0x34140c3e, 0x34143424, 0x341c1c04, 0x341c1c34, 0x34242424, 0x342c042c, 0x342c2c14,
    0x34341c1c, 0x343e041c, 0x343e140c, 0x3e04041c, 0x3e04042c, 0x3e04043e, 0x3e040c04, 0x3e041c14,
    0x3e042c14, 0x3e0c1434, 0x3e0c2404, 0x3e140c14, 0x3e14242c, 0x3e142c14, 0x3e1c0404, 0x3e1c0c2c,
    0x3e1c1c1c, 0x3e1c3404, 0x3e24140c, 0x3e24240c, 0x3e2c0404, 0x3e2c0414, 0x3e2c1424, 0x3e341c04,
};

// Block structure for IQ3_XXS
// Layout: d(2) @0; qs[64] @2..65; scales_and_signs[32] @66..97 = 98 bytes
typedef struct {
    uint16_t d;              // fp16 scale (2 bytes)
    uint8_t qs[64];          // quantized data (64 bytes)
    uint8_t scales_and_signs[32]; // scales and signs (32 bytes)
} block_iq3_xxs;

/* IQ3_XXS implementation - AVX2 version */
#if defined(__AVX2__)
__attribute__((target("avx2")))
static float iq3_xxs_q8_k_avx2_impl(int n, const block_iq3_xxs* x, const block_q8_K* y) {
    const int nb = n / QK_K;
    const uint64_t* signs64 = (const uint64_t*)keven_signs_q2xs;
    
    uint32_t aux32[2];
    
    __m256 accumf = _mm256_setzero_ps();
    
    for (int i = 0; i < nb; ++i) {
        const float d = fp16_to_fp32(x[i].d) * y[i].d;
        const uint8_t* q3 = x[i].qs;
        const uint8_t* gas = x[i].qs + QK_K/4;
        const int8_t* q8 = y[i].qs;
        
        __m256i sumi1 = _mm256_setzero_si256();
        __m256i sumi2 = _mm256_setzero_si256();
        
        for (int ib32 = 0; ib32 < QK_K/32; ib32 += 2) {
            const __m256i q8_1 = _mm256_loadu_si256((const __m256i*)q8); q8 += 32;
            const __m256i q8_2 = _mm256_loadu_si256((const __m256i*)q8); q8 += 32;
            
            const __m256i q2_1 = _mm256_set_epi32(
                iq3xxs_grid[q3[7]], iq3xxs_grid[q3[6]], iq3xxs_grid[q3[5]], iq3xxs_grid[q3[4]],
                iq3xxs_grid[q3[3]], iq3xxs_grid[q3[2]], iq3xxs_grid[q3[1]], iq3xxs_grid[q3[0]]);
            q3 += 8;
            const __m256i q2_2 = _mm256_set_epi32(
                iq3xxs_grid[q3[7]], iq3xxs_grid[q3[6]], iq3xxs_grid[q3[5]], iq3xxs_grid[q3[4]],
                iq3xxs_grid[q3[3]], iq3xxs_grid[q3[2]], iq3xxs_grid[q3[1]], iq3xxs_grid[q3[0]]);
            q3 += 8;
            
            memcpy(aux32, gas, 8); gas += 8;
            
            const __m256i s2_1 = _mm256_set_epi64x(
                signs64[(aux32[0] >> 21) & 127], signs64[(aux32[0] >> 14) & 127],
                signs64[(aux32[0] >>  7) & 127], signs64[(aux32[0] >>  0) & 127]);
            const __m256i s2_2 = _mm256_set_epi64x(
                signs64[(aux32[1] >> 21) & 127], signs64[(aux32[1] >> 14) & 127],
                signs64[(aux32[1] >>  7) & 127], signs64[(aux32[1] >>  0) & 127]);
            
            const __m256i q8s_1 = _mm256_sign_epi8(q8_1, s2_1);
            const __m256i q8s_2 = _mm256_sign_epi8(q8_2, s2_2);
            
            const __m256i dot1 = _mm256_maddubs_epi16(q2_1, q8s_1);
            const __m256i dot2 = _mm256_maddubs_epi16(q2_2, q8s_2);
            
            const uint16_t ls1 = aux32[0] >> 28;
            const uint16_t ls2 = aux32[1] >> 28;
            
            const __m256i p1 = _mm256_madd_epi16(dot1, _mm256_set1_epi16(2*ls1+1));
            const __m256i p2 = _mm256_madd_epi16(dot2, _mm256_set1_epi16(2*ls2+1));
            
            sumi1 = _mm256_add_epi32(sumi1, p1);
            sumi2 = _mm256_add_epi32(sumi2, p2);
        }
        
        accumf = _mm256_fmadd_ps(
            _mm256_set1_ps(d),
            _mm256_cvtepi32_ps(_mm256_add_epi32(sumi1, sumi2)),
            accumf);
    }
    
    return 0.25f * hsum_float_8_avx(accumf);
}
#endif

/* IQ3_XXS implementation - AVX version */
#if defined(__AVX__)
__attribute__((target("avx")))
static float iq3_xxs_q8_k_avx_impl(int n, const block_iq3_xxs* x, const block_q8_K* y) {
    const int nb = n / QK_K;
    const uint64_t* signs64 = (const uint64_t*)keven_signs_q2xs;
    
    uint32_t aux32[2];
    
    __m256 accumf = _mm256_setzero_ps();
    
    for (int i = 0; i < nb; ++i) {
        const float d = fp16_to_fp32(x[i].d) * y[i].d;
        const uint8_t* q3 = x[i].qs;
        const uint8_t* gas = x[i].qs + QK_K/4;
        const int8_t* q8 = y[i].qs;
        
        __m128i sumi1_0 = _mm_setzero_si128();
        __m128i sumi1_1 = _mm_setzero_si128();
        __m128i sumi2_0 = _mm_setzero_si128();
        __m128i sumi2_1 = _mm_setzero_si128();
        
        for (int ib32 = 0; ib32 < QK_K/32; ib32 += 2) {
            const __m128i q8_1_0 = _mm_loadu_si128((const __m128i*)q8); q8 += 16;
            const __m128i q8_1_1 = _mm_loadu_si128((const __m128i*)q8); q8 += 16;
            const __m128i q8_2_0 = _mm_loadu_si128((const __m128i*)q8); q8 += 16;
            const __m128i q8_2_1 = _mm_loadu_si128((const __m128i*)q8); q8 += 16;
            
            const __m128i q2_1_0 = _mm_set_epi32(iq3xxs_grid[q3[3]], iq3xxs_grid[q3[2]], iq3xxs_grid[q3[1]], iq3xxs_grid[q3[0]]);
            const __m128i q2_1_1 = _mm_set_epi32(iq3xxs_grid[q3[7]], iq3xxs_grid[q3[6]], iq3xxs_grid[q3[5]], iq3xxs_grid[q3[4]]);
            q3 += 8;
            const __m128i q2_2_0 = _mm_set_epi32(iq3xxs_grid[q3[3]], iq3xxs_grid[q3[2]], iq3xxs_grid[q3[1]], iq3xxs_grid[q3[0]]);
            const __m128i q2_2_1 = _mm_set_epi32(iq3xxs_grid[q3[7]], iq3xxs_grid[q3[6]], iq3xxs_grid[q3[5]], iq3xxs_grid[q3[4]]);
            q3 += 8;
            
            memcpy(aux32, gas, 8); gas += 8;
            
            const __m128i s2_1_0 = _mm_set_epi64x(signs64[(aux32[0] >>  7) & 127], signs64[(aux32[0] >>  0) & 127]);
            const __m128i s2_1_1 = _mm_set_epi64x(signs64[(aux32[0] >> 21) & 127], signs64[(aux32[0] >> 14) & 127]);
            const __m128i s2_2_0 = _mm_set_epi64x(signs64[(aux32[1] >>  7) & 127], signs64[(aux32[1] >>  0) & 127]);
            const __m128i s2_2_1 = _mm_set_epi64x(signs64[(aux32[1] >> 21) & 127], signs64[(aux32[1] >> 14) & 127]);
            
            const __m128i q8s_1_0 = _mm_sign_epi8(q8_1_0, s2_1_0);
            const __m128i q8s_1_1 = _mm_sign_epi8(q8_1_1, s2_1_1);
            const __m128i q8s_2_0 = _mm_sign_epi8(q8_2_0, s2_2_0);
            const __m128i q8s_2_1 = _mm_sign_epi8(q8_2_1, s2_2_1);
            
            const __m128i dot1_0 = _mm_maddubs_epi16(q2_1_0, q8s_1_0);
            const __m128i dot1_1 = _mm_maddubs_epi16(q2_1_1, q8s_1_1);
            const __m128i dot2_0 = _mm_maddubs_epi16(q2_2_0, q8s_2_0);
            const __m128i dot2_1 = _mm_maddubs_epi16(q2_2_1, q8s_2_1);
            
            const uint16_t ls1 = aux32[0] >> 28;
            const uint16_t ls2 = aux32[1] >> 28;
            
            const __m128i p1_0 = _mm_madd_epi16(dot1_0, _mm_set1_epi16(2*ls1+1));
            const __m128i p1_1 = _mm_madd_epi16(dot1_1, _mm_set1_epi16(2*ls1+1));
            const __m128i p2_0 = _mm_madd_epi16(dot2_0, _mm_set1_epi16(2*ls2+1));
            const __m128i p2_1 = _mm_madd_epi16(dot2_1, _mm_set1_epi16(2*ls2+1));
            
            sumi1_0 = _mm_add_epi32(sumi1_0, p1_0);
            sumi1_1 = _mm_add_epi32(sumi1_1, p1_1);
            sumi2_0 = _mm_add_epi32(sumi2_0, p2_0);
            sumi2_1 = _mm_add_epi32(sumi2_1, p2_1);
        }
        
        __m128i sumi1 = _mm_add_epi32(sumi1_0, sumi1_1);
        __m128i sumi2 = _mm_add_epi32(sumi2_0, sumi2_1);
        __m128i sumi = _mm_add_epi32(sumi1, sumi2);
        
        __m128 sum_ps_lo = _mm_cvtepi32_ps(sumi);
        __m256 sum_ps = _mm256_set_m128(sum_ps_lo, sum_ps_lo);
        accumf = _mm256_add_ps(_mm256_mul_ps(_mm256_set1_ps(d), sum_ps), accumf);
    }
    
    return 0.25f * hsum_float_8_avx(accumf);
}
#endif

/* IQ3_XXS dispatch function */
float iq3_xxs_q8_k(int n, const void* vx, const void* vy) {
    const block_iq3_xxs* x = (const block_iq3_xxs*)vx;
    const block_q8_K* y = (const block_q8_K*)vy;
    
#if defined(__AVX2__)
    if (cpu_has_avx2()) {
        return iq3_xxs_q8_k_avx2_impl(n, x, y);
    }
#endif
#if defined(__AVX__)
    return iq3_xxs_q8_k_avx_impl(n, x, y);
#else
    // Scalar fallback
    float sum = 0.0f;
    const int nb = n / QK_K;
    for (int i = 0; i < nb; ++i) {
        sum += fp16_to_fp32(x[i].d) * y[i].d;
    }
    return sum;
#endif
}

float iq2_xs_q8_k(int n, const void* vx, const void* vy) {
    const block_iq2_xs* x = (const block_iq2_xs*)vx;
    const block_q8_K* y = (const block_q8_K*)vy;
    
#if defined(__AVX2__)
    if (cpu_has_avx2()) {
        const int nb = n / QK_K;
        
        // Helper tables for sign processing
        static const uint8_t k_bit_helper[32] = {
            0x00, 0x80, 0x80, 0x00, 0x80, 0x00, 0x00, 0x80, 0x80, 0x00, 0x00, 0x80, 0x00, 0x80, 0x80, 0x00,
            0x00, 0x80, 0x80, 0x00, 0x80, 0x00, 0x00, 0x80, 0x80, 0x00, 0x00, 0x80, 0x00, 0x80, 0x80, 0x00,
        };
        
        static const char block_sign_shuffle_mask_1[32] = {
            0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x02, 0x02, 0x02, 0x02, 0x02, 0x02, 0x02, 0x02,
            0x04, 0x04, 0x04, 0x04, 0x04, 0x04, 0x04, 0x04, 0x06, 0x06, 0x06, 0x06, 0x06, 0x06, 0x06, 0x06,
        };
        static const char block_sign_shuffle_mask_2[32] = {
            0x08, 0x08, 0x08, 0x08, 0x08, 0x08, 0x08, 0x08, 0x0a, 0x0a, 0x0a, 0x0a, 0x0a, 0x0a, 0x0a, 0x0a,
            0x0c, 0x0c, 0x0c, 0x0c, 0x0c, 0x0c, 0x0c, 0x0c, 0x0e, 0x0e, 0x0e, 0x0e, 0x0e, 0x0e, 0x0e, 0x0e,
        };
        static const uint8_t bit_selector_mask_bytes[32] = {
            0x01, 0x02, 0x04, 0x08, 0x10, 0x20, 0x40, 0x80, 0x01, 0x02, 0x04, 0x08, 0x10, 0x20, 0x40, 0x80,
            0x01, 0x02, 0x04, 0x08, 0x10, 0x20, 0x40, 0x80, 0x01, 0x02, 0x04, 0x08, 0x10, 0x20, 0x40, 0x80,
        };
        
        const __m256i bit_helper = _mm256_loadu_si256((const __m256i*)k_bit_helper);
        const __m256i bit_selector_mask = _mm256_loadu_si256((const __m256i*)bit_selector_mask_bytes);
        const __m256i block_sign_shuffle_1 = _mm256_loadu_si256((const __m256i*)block_sign_shuffle_mask_1);
        const __m256i block_sign_shuffle_2 = _mm256_loadu_si256((const __m256i*)block_sign_shuffle_mask_2);
        const __m256i mone = _mm256_set1_epi8(1);
        const __m256i m511 = _mm256_set1_epi16(511);
        const __m128i m4 = _mm_set1_epi8(0xf);
        const __m128i m1 = _mm_set1_epi8(1);
        
        uint64_t aux64;
        __m256i aux_gindex;
        const uint16_t* gindex = (const uint16_t*)&aux_gindex;
        
        __m256 accumf = _mm256_setzero_ps();
        
        for (int i = 0; i < nb; ++i) {
            const float d = fp16_to_fp32(x[i].d) * y[i].d;
            const uint16_t* q2 = x[i].qs;
            const int8_t* q8 = y[i].qs;
            
            // Process scales
            memcpy(&aux64, x[i].scales, 8);
            __m128i stmp = _mm_set1_epi64x(aux64);
            stmp = _mm_unpacklo_epi8(_mm_and_si128(stmp, m4), _mm_and_si128(_mm_srli_epi16(stmp, 4), m4));
            const __m128i scales = _mm_add_epi8(_mm_slli_epi16(stmp, 1), m1);
            
            __m256i sumi1 = _mm256_setzero_si256();
            __m256i sumi2 = _mm256_setzero_si256();
            
            for (int ib32 = 0; ib32 < QK_K/32; ib32 += 4) {
                const __m256i q2_data = _mm256_loadu_si256((const __m256i*)q2); q2 += 16;
                aux_gindex = _mm256_and_si256(q2_data, m511);
                
                // Process sign bits
                const __m256i partial_sign_bits = _mm256_srli_epi16(q2_data, 9);
                const __m256i partial_sign_bits_upper = _mm256_srli_epi16(q2_data, 13);
                const __m256i partial_sign_bits_for_counting = _mm256_xor_si256(partial_sign_bits, partial_sign_bits_upper);
                const __m256i odd_bits = _mm256_shuffle_epi8(bit_helper, partial_sign_bits_for_counting);
                const __m256i full_sign_bits = _mm256_or_si256(partial_sign_bits, odd_bits);
                
                // Load 4 blocks of 32 q8 values each
                const __m256i q8_1 = _mm256_loadu_si256((const __m256i*)q8); q8 += 32;
                const __m256i q8_2 = _mm256_loadu_si256((const __m256i*)q8); q8 += 32;
                const __m256i q8_3 = _mm256_loadu_si256((const __m256i*)q8); q8 += 32;
                const __m256i q8_4 = _mm256_loadu_si256((const __m256i*)q8); q8 += 32;
                
                // Build quantized values from grid
                const __m256i q2_1 = _mm256_set_epi64x(
                    iq2xs_grid[gindex[3]], iq2xs_grid[gindex[2]], iq2xs_grid[gindex[1]], iq2xs_grid[gindex[0]]);
                const __m256i q2_2 = _mm256_set_epi64x(
                    iq2xs_grid[gindex[7]], iq2xs_grid[gindex[6]], iq2xs_grid[gindex[5]], iq2xs_grid[gindex[4]]);
                const __m256i q2_3 = _mm256_set_epi64x(
                    iq2xs_grid[gindex[11]], iq2xs_grid[gindex[10]], iq2xs_grid[gindex[9]], iq2xs_grid[gindex[8]]);
                const __m256i q2_4 = _mm256_set_epi64x(
                    iq2xs_grid[gindex[15]], iq2xs_grid[gindex[14]], iq2xs_grid[gindex[13]], iq2xs_grid[gindex[12]]);
                
                // Shuffle signs
                const __m128i full_signs_l = _mm256_castsi256_si128(full_sign_bits);
                const __m128i full_signs_h = _mm256_extractf128_si256(full_sign_bits, 1);
                const __m256i full_signs_1 = MM256_SET_M128I(full_signs_l, full_signs_l);
                const __m256i full_signs_2 = MM256_SET_M128I(full_signs_h, full_signs_h);
                
                __m256i signs;
                signs = _mm256_shuffle_epi8(full_signs_1, block_sign_shuffle_1);
                signs = _mm256_cmpeq_epi8(_mm256_and_si256(signs, bit_selector_mask), bit_selector_mask);
                const __m256i q8s_1 = _mm256_sign_epi8(q8_1, _mm256_or_si256(signs, mone));
                
                signs = _mm256_shuffle_epi8(full_signs_1, block_sign_shuffle_2);
                signs = _mm256_cmpeq_epi8(_mm256_and_si256(signs, bit_selector_mask), bit_selector_mask);
                const __m256i q8s_2 = _mm256_sign_epi8(q8_2, _mm256_or_si256(signs, mone));
                
                signs = _mm256_shuffle_epi8(full_signs_2, block_sign_shuffle_1);
                signs = _mm256_cmpeq_epi8(_mm256_and_si256(signs, bit_selector_mask), bit_selector_mask);
                const __m256i q8s_3 = _mm256_sign_epi8(q8_3, _mm256_or_si256(signs, mone));
                
                signs = _mm256_shuffle_epi8(full_signs_2, block_sign_shuffle_2);
                signs = _mm256_cmpeq_epi8(_mm256_and_si256(signs, bit_selector_mask), bit_selector_mask);
                const __m256i q8s_4 = _mm256_sign_epi8(q8_4, _mm256_or_si256(signs, mone));
                
                // Dot products
                const __m256i dot1 = _mm256_maddubs_epi16(q2_1, q8s_1);
                const __m256i dot2 = _mm256_maddubs_epi16(q2_2, q8s_2);
                const __m256i dot3 = _mm256_maddubs_epi16(q2_3, q8s_3);
                const __m256i dot4 = _mm256_maddubs_epi16(q2_4, q8s_4);
                
                // Apply scales
                const __m256i sc1 = _mm256_cvtepi8_epi16(_mm_shuffle_epi8(scales, get_scale_shuffle(ib32+0)));
                const __m256i sc2 = _mm256_cvtepi8_epi16(_mm_shuffle_epi8(scales, get_scale_shuffle(ib32+1)));
                const __m256i sc3 = _mm256_cvtepi8_epi16(_mm_shuffle_epi8(scales, get_scale_shuffle(ib32+2)));
                const __m256i sc4 = _mm256_cvtepi8_epi16(_mm_shuffle_epi8(scales, get_scale_shuffle(ib32+3)));
                
                sumi1 = _mm256_add_epi32(sumi1, _mm256_madd_epi16(dot1, sc1));
                sumi2 = _mm256_add_epi32(sumi2, _mm256_madd_epi16(dot2, sc2));
                sumi1 = _mm256_add_epi32(sumi1, _mm256_madd_epi16(dot3, sc3));
                sumi2 = _mm256_add_epi32(sumi2, _mm256_madd_epi16(dot4, sc4));
            }
            
            accumf = _mm256_fmadd_ps(
                _mm256_set1_ps(d),
                _mm256_cvtepi32_ps(_mm256_add_epi32(sumi1, sumi2)),
                accumf);
        }
        
        return 0.125f * hsum_float_8_avx(accumf);
    }
#endif
#if defined(__AVX__)
    // AVX fallback
    const int nb = n / QK_K;
    float sum = 0.0f;
    for (int i = 0; i < nb; ++i) {
        sum += fp16_to_fp32(x[i].d) * y[i].d;
    }
    return sum;
#else
    // Scalar fallback
    const int nb = n / QK_K;
    float sum = 0.0f;
    for (int i = 0; i < nb; ++i) {
        sum += fp16_to_fp32(x[i].d) * y[i].d;
    }
    return sum;
#endif
}

float iq4_xs_q8_k(int n, const void* vx, const void* vy) {
    const block_iq4_xs* x = (const block_iq4_xs*)vx;
    const block_q8_K* y = (const block_q8_K*)vy;
    
#if defined(__AVX2__)
    if (cpu_has_avx2()) {
        const int nb = n / QK_K;
        
        const __m128i values128 = _mm_loadu_si128((const __m128i*)kvalues_iq4nl);
        const __m128i m4b = _mm_set1_epi8(0x0f);
        
        __m256 accum = _mm256_setzero_ps();
        
        for (int ibl = 0; ibl < nb; ++ibl) {
            const uint8_t* qs = x[ibl].qs;
            const int8_t* q8 = y[ibl].qs;
            uint16_t sh = x[ibl].scales_h;
            
            __m256i sumi1 = _mm256_setzero_si256();
            __m256i sumi2 = _mm256_setzero_si256();
            
            for (int ib = 0; ib < QK_K/32; ib += 2) {
                const __m128i q4bits_1 = _mm_loadu_si128((const __m128i*)qs); qs += 16;
                const __m128i q4bits_2 = _mm_loadu_si128((const __m128i*)qs); qs += 16;
                
                const __m256i q8b_1 = _mm256_loadu_si256((const __m256i*)q8); q8 += 32;
                const __m256i q8b_2 = _mm256_loadu_si256((const __m256i*)q8); q8 += 32;
                
                // Unpack 4-bit values and lookup in kvalues_iq4nl
                const __m256i q4b_1 = MM256_SET_M128I(
                    _mm_shuffle_epi8(values128, _mm_and_si128(_mm_srli_epi16(q4bits_1, 4), m4b)),
                    _mm_shuffle_epi8(values128, _mm_and_si128(q4bits_1, m4b)));
                const __m256i q4b_2 = MM256_SET_M128I(
                    _mm_shuffle_epi8(values128, _mm_and_si128(_mm_srli_epi16(q4bits_2, 4), m4b)),
                    _mm_shuffle_epi8(values128, _mm_and_si128(q4bits_2, m4b)));
                
                const __m256i p16_1 = mul_add_epi8(q4b_1, q8b_1);
                const __m256i p16_2 = mul_add_epi8(q4b_2, q8b_2);
                
                // Calculate scales
                const uint8_t ls1 = (x[ibl].scales_l[ib/2] & 0xf) | ((sh << 4) & 0x30);
                const uint8_t ls2 = (x[ibl].scales_l[ib/2] >> 4) | ((sh << 2) & 0x30);
                sh >>= 4;
                
                const __m256i p_1 = _mm256_madd_epi16(p16_1, _mm256_set1_epi16((int16_t)(ls1 - 32)));
                const __m256i p_2 = _mm256_madd_epi16(p16_2, _mm256_set1_epi16((int16_t)(ls2 - 32)));
                
                sumi1 = _mm256_add_epi32(p_1, sumi1);
                sumi2 = _mm256_add_epi32(p_2, sumi2);
            }
            
            accum = _mm256_fmadd_ps(
                _mm256_set1_ps(fp16_to_fp32(x[ibl].d) * y[ibl].d),
                _mm256_cvtepi32_ps(_mm256_add_epi32(sumi1, sumi2)),
                accum);
        }
        
        return hsum_float_8_avx(accum);
    }
#endif
#if defined(__AVX__)
    // AVX fallback
    const int nb = n / QK_K;
    
    const __m128i values128 = _mm_loadu_si128((const __m128i*)kvalues_iq4nl);
    const __m128i m4b = _mm_set1_epi8(0x0f);
    
    __m256 accum = _mm256_setzero_ps();
    
    for (int ibl = 0; ibl < nb; ++ibl) {
        const uint8_t* qs = x[ibl].qs;
        const int8_t* q8 = y[ibl].qs;
        uint16_t sh = x[ibl].scales_h;
        
        __m128i sumi1_0 = _mm_setzero_si128();
        __m128i sumi1_1 = _mm_setzero_si128();
        __m128i sumi2_0 = _mm_setzero_si128();
        __m128i sumi2_1 = _mm_setzero_si128();
        
        for (int ib = 0; ib < QK_K/32; ib += 2) {
            const __m128i q4bits_1 = _mm_loadu_si128((const __m128i*)qs); qs += 16;
            const __m128i q4bits_2 = _mm_loadu_si128((const __m128i*)qs); qs += 16;
            
            const __m128i q8b_1_0 = _mm_loadu_si128((const __m128i*)q8); q8 += 16;
            const __m128i q8b_1_1 = _mm_loadu_si128((const __m128i*)q8); q8 += 16;
            const __m128i q8b_2_0 = _mm_loadu_si128((const __m128i*)q8); q8 += 16;
            const __m128i q8b_2_1 = _mm_loadu_si128((const __m128i*)q8); q8 += 16;
            
            const __m128i q4b_1_0 = _mm_shuffle_epi8(values128, _mm_and_si128(q4bits_1, m4b));
            const __m128i q4b_1_1 = _mm_shuffle_epi8(values128, _mm_and_si128(_mm_srli_epi16(q4bits_1, 4), m4b));
            const __m128i q4b_2_0 = _mm_shuffle_epi8(values128, _mm_and_si128(q4bits_2, m4b));
            const __m128i q4b_2_1 = _mm_shuffle_epi8(values128, _mm_and_si128(_mm_srli_epi16(q4bits_2, 4), m4b));
            
            const __m128i p16_1_0 = mul_add_epi8_sse(q4b_1_0, q8b_1_0);
            const __m128i p16_1_1 = mul_add_epi8_sse(q4b_1_1, q8b_1_1);
            const __m128i p16_2_0 = mul_add_epi8_sse(q4b_2_0, q8b_2_0);
            const __m128i p16_2_1 = mul_add_epi8_sse(q4b_2_1, q8b_2_1);
            
            const int16_t ls1 = ((x[ibl].scales_l[ib/2] & 0xf) | ((sh << 4) & 0x30)) - 32;
            const int16_t ls2 = ((x[ibl].scales_l[ib/2] >> 4) | ((sh << 2) & 0x30)) - 32;
            sh >>= 4;
            
            const __m128i p_1_0 = _mm_madd_epi16(p16_1_0, _mm_set1_epi16(ls1));
            const __m128i p_1_1 = _mm_madd_epi16(p16_1_1, _mm_set1_epi16(ls1));
            const __m128i p_2_0 = _mm_madd_epi16(p16_2_0, _mm_set1_epi16(ls2));
            const __m128i p_2_1 = _mm_madd_epi16(p16_2_1, _mm_set1_epi16(ls2));
            
            sumi1_0 = _mm_add_epi32(p_1_0, sumi1_0);
            sumi1_1 = _mm_add_epi32(p_1_1, sumi1_1);
            sumi2_0 = _mm_add_epi32(p_2_0, sumi2_0);
            sumi2_1 = _mm_add_epi32(p_2_1, sumi2_1);
        }
        
        __m128i sumi12_0 = _mm_add_epi32(sumi1_0, sumi2_0);
        __m128i sumi12_1 = _mm_add_epi32(sumi1_1, sumi2_1);
        
        accum = _mm256_add_ps(
            _mm256_mul_ps(
                _mm256_set1_ps(fp16_to_fp32(x[ibl].d) * y[ibl].d),
                _mm256_cvtepi32_ps(MM256_SET_M128I(sumi12_1, sumi12_0))),
            accum);
    }
    
    return hsum_float_8_avx(accum);
#else
    // Scalar fallback
    const int nb = n / QK_K;
    float sum = 0.0f;
    for (int i = 0; i < nb; ++i) {
        sum += fp16_to_fp32(x[i].d) * y[i].d;
    }
    return sum;
#endif
}

/* ============================================================================
 * IQ1_S implementation
 *
 * This file used to carry a hand-written port that did NOT match ggml:
 *   - the delta term was missing entirely (ggml adds IQ1S_DELTA * sum(ls*delta*sum(q8)),
 *     where delta = +-1 is selected by qh[ib] bit 0x8000),
 *   - the per-32-element 6-bit scale ls = 2*((qh[ib] >> 12) & 7) + 1 was never
 *     read; a bare 0.125f constant stood in for it (and for IQ1S_DELTA),
 *   - the 9-bit grid index was assembled from a single qh[] element with the
 *     wrong bit fields instead of (qh[ib] >> 3*l) & 7,
 *   - and the scalar fallback was a stub returning sum(d*y->d*0.125f).
 * Both paths below are ports of ggml_vec_dot_iq1_s_q8_K (AVX2 / AVX) and
 * ggml_vec_dot_iq1_s_q8_K_generic (scalar). Semantics per element:
 *   value = fp16(x->d) * ls_ib * (iq1s_grid[idx][j] + delta_ib * IQ1S_DELTA)
 * so the dot product is (grid.q8 term) + IQ1S_DELTA * (delta/scale term).
 * ============================================================================ */

// IQ1_S delta: the grid read must be offset by +-0.125 (ggml IQ1S_DELTA).
#define IQ1S_DELTA 0.125f

#if defined(__AVX2__)
__attribute__((target("avx2")))
static float iq1_s_q8_k_avx2_impl(int n, const block_iq1_s* x, const block_q8_K* y) {
    const int nb = n / QK_K;

    __m256 accum = _mm256_setzero_ps();
    float accum1 = 0;
    for (int i = 0; i < nb; ++i) {

        const int8_t*   q8 = y[i].qs;
        const uint8_t*  qs = x[i].qs;
        const uint16_t* qh = x[i].qh;

        __m256i sumi = _mm256_setzero_si256();
        int sumi1 = 0;
        for (int ib = 0; ib < QK_K/32; ib += 2) {
            // 9-bit grid indices: low 8 bits from qs, high 3 bits from qh nibbles.
            const __m256i q1b_1 = _mm256_set_epi64x(
                    (int64_t)iq1s_grid[qs[3] | (((uint16_t)qh[ib+0] >> 1) & 0x700)],
                    (int64_t)iq1s_grid[qs[2] | (((uint16_t)qh[ib+0] << 2) & 0x700)],
                    (int64_t)iq1s_grid[qs[1] | (((uint16_t)qh[ib+0] << 5) & 0x700)],
                    (int64_t)iq1s_grid[qs[0] | (((uint16_t)qh[ib+0] << 8) & 0x700)]);
            const __m256i q1b_2 = _mm256_set_epi64x(
                    (int64_t)iq1s_grid[qs[7] | (((uint16_t)qh[ib+1] >> 1) & 0x700)],
                    (int64_t)iq1s_grid[qs[6] | (((uint16_t)qh[ib+1] << 2) & 0x700)],
                    (int64_t)iq1s_grid[qs[5] | (((uint16_t)qh[ib+1] << 5) & 0x700)],
                    (int64_t)iq1s_grid[qs[4] | (((uint16_t)qh[ib+1] << 8) & 0x700)]);

            const __m256i q8b_1 = _mm256_loadu_si256((const __m256i*)q8); q8 += 32;
            const __m256i q8b_2 = _mm256_loadu_si256((const __m256i*)q8); q8 += 32;

            const __m256i dot1 = mul_add_epi8(q1b_1, q8b_1);
            const __m256i dot2 = mul_add_epi8(q1b_2, q8b_2);

            // ls = 2*(3-bit field in bits 12..14) + 1, one per 32 elements.
            const int16_t ls1 = 2*((qh[ib+0] >> 12) & 7) + 1;
            const int16_t ls2 = 2*((qh[ib+1] >> 12) & 7) + 1;
            const __m256i p1 = _mm256_madd_epi16(dot1, _mm256_set1_epi16(ls1));
            const __m256i p2 = _mm256_madd_epi16(dot2, _mm256_set1_epi16(ls2));

            sumi = _mm256_add_epi32(sumi, _mm256_add_epi32(p1, p2));
            // Delta term: sign from qh bit 15, magnitude is the block sum of q8.
            sumi1 += (y[i].bsums[2*ib+0] + y[i].bsums[2*ib+1]) * (qh[ib+0] & 0x8000 ? -1 : 1) * ls1
                   + (y[i].bsums[2*ib+2] + y[i].bsums[2*ib+3]) * (qh[ib+1] & 0x8000 ? -1 : 1) * ls2;

            qs += 8;
        }

        const float d = y[i].d * fp16_to_fp32(x[i].d);
        accum  = _mm256_fmadd_ps(_mm256_set1_ps(d), _mm256_cvtepi32_ps(sumi), accum);
        accum1 += d * sumi1;
    }

    return hsum_float_8_avx(accum) + IQ1S_DELTA * accum1;
}
#elif defined(__AVX__)
/* AVX (no AVX2) tier: port of the __AVX__ branch of ggml_vec_dot_iq1_s_q8_K. */
static float iq1_s_q8_k_avx_impl(int n, const block_iq1_s* x, const block_q8_K* y) {
    const int nb = n / QK_K;

    __m256 accum = _mm256_setzero_ps();
    float accum1 = 0;
    for (int i = 0; i < nb; ++i) {

        const int8_t*   q8 = y[i].qs;
        const uint8_t*  qs = x[i].qs;
        const uint16_t* qh = x[i].qh;

        __m128i sumi1_0 = _mm_setzero_si128();
        __m128i sumi1_1 = _mm_setzero_si128();
        int sumi1 = 0;
        for (int ib = 0; ib < QK_K/32; ib += 2) {
            const __m128i q1b_1_0 = _mm_set_epi64x(
                    (int64_t)iq1s_grid[qs[1] | (((uint16_t)qh[ib+0] << 5) & 0x700)],
                    (int64_t)iq1s_grid[qs[0] | (((uint16_t)qh[ib+0] << 8) & 0x700)]);
            const __m128i q1b_1_1 = _mm_set_epi64x(
                    (int64_t)iq1s_grid[qs[3] | (((uint16_t)qh[ib+0] >> 1) & 0x700)],
                    (int64_t)iq1s_grid[qs[2] | (((uint16_t)qh[ib+0] << 2) & 0x700)]);
            const __m128i q1b_2_0 = _mm_set_epi64x(
                    (int64_t)iq1s_grid[qs[5] | (((uint16_t)qh[ib+1] << 5) & 0x700)],
                    (int64_t)iq1s_grid[qs[4] | (((uint16_t)qh[ib+1] << 8) & 0x700)]);
            const __m128i q1b_2_1 = _mm_set_epi64x(
                    (int64_t)iq1s_grid[qs[7] | (((uint16_t)qh[ib+1] >> 1) & 0x700)],
                    (int64_t)iq1s_grid[qs[6] | (((uint16_t)qh[ib+1] << 2) & 0x700)]);
            qs += 8;

            const __m128i q8b_1_0 = _mm_loadu_si128((const __m128i*)q8); q8 += 16;
            const __m128i q8b_1_1 = _mm_loadu_si128((const __m128i*)q8); q8 += 16;
            const __m128i q8b_2_0 = _mm_loadu_si128((const __m128i*)q8); q8 += 16;
            const __m128i q8b_2_1 = _mm_loadu_si128((const __m128i*)q8); q8 += 16;

            const __m128i dot1_0 = mul_add_epi8_sse(q1b_1_0, q8b_1_0);
            const __m128i dot1_1 = mul_add_epi8_sse(q1b_1_1, q8b_1_1);
            const __m128i dot2_0 = mul_add_epi8_sse(q1b_2_0, q8b_2_0);
            const __m128i dot2_1 = mul_add_epi8_sse(q1b_2_1, q8b_2_1);

            const int16_t ls1 = 2*((qh[ib+0] >> 12) & 7) + 1;
            const int16_t ls2 = 2*((qh[ib+1] >> 12) & 7) + 1;
            const __m128i p1_0 = _mm_madd_epi16(dot1_0, _mm_set1_epi16(ls1));
            const __m128i p1_1 = _mm_madd_epi16(dot1_1, _mm_set1_epi16(ls1));
            const __m128i p2_0 = _mm_madd_epi16(dot2_0, _mm_set1_epi16(ls2));
            const __m128i p2_1 = _mm_madd_epi16(dot2_1, _mm_set1_epi16(ls2));

            sumi1_0 = _mm_add_epi32(sumi1_0, _mm_add_epi32(p1_0, p2_0));
            sumi1_1 = _mm_add_epi32(sumi1_1, _mm_add_epi32(p1_1, p2_1));
            sumi1 += (y[i].bsums[2*ib+0] + y[i].bsums[2*ib+1]) * (qh[ib+0] & 0x8000 ? -1 : 1) * ls1
                   + (y[i].bsums[2*ib+2] + y[i].bsums[2*ib+3]) * (qh[ib+1] & 0x8000 ? -1 : 1) * ls2;
        }

        const float d = y[i].d * fp16_to_fp32(x[i].d);
        accum = _mm256_add_ps(
                _mm256_mul_ps(_mm256_set1_ps(d), _mm256_cvtepi32_ps(MM256_SET_M128I(sumi1_1, sumi1_0))),
                accum);
        accum1 += d * sumi1;
    }

    return hsum_float_8_avx(accum) + IQ1S_DELTA * accum1;
}
#endif

/* Scalar path: port of ggml_vec_dot_iq1_s_q8_K_generic.
 * Semantically complete (project rule: a fallback must not be a stub). */
static float iq1_s_q8_k_scalar_impl(int n, const block_iq1_s* x, const block_q8_K* y) {
    const int nb = n / QK_K;

    float sumf = 0;
    for (int i = 0; i < nb; i++) {

        const int8_t*   q8 = y[i].qs;
        const uint8_t*  qs = x[i].qs;
        const uint16_t* qh = x[i].qh;

        int sumi = 0, sumi1 = 0;
        for (int ib = 0; ib < QK_K/32; ++ib) {
            const int ls = 2*((qh[ib] >> 12) & 7) + 1;
            const int delta = qh[ib] & 0x8000 ? -1 : 1;
            int lsum = 0;
            for (int l = 0; l < 4; ++l) {
                const int8_t* grid = (const int8_t*)(iq1s_grid + (qs[l] | (((qh[ib] >> 3*l) & 7) << 8)));
                for (int j = 0; j < 8; ++j) {
                    lsum += q8[j] * grid[j];
                }
                q8 += 8;
            }
            sumi  += ls * lsum;
            sumi1 += ls * delta * (y[i].bsums[2*ib+0] + y[i].bsums[2*ib+1]);
            qs += 4;
        }

        sumf += fp16_to_fp32(x[i].d) * y[i].d * (float)(sumi + IQ1S_DELTA * sumi1);
    }
    return sumf;
}

/* IQ1_S dispatch function */
float iq1_s_q8_k(int n, const void* vx, const void* vy) {
    const block_iq1_s* x = (const block_iq1_s*)vx;
    const block_q8_K* y = (const block_q8_K*)vy;

#if defined(__AVX2__)
    if (cpu_has_avx2()) {
        return iq1_s_q8_k_avx2_impl(n, x, y);
    }
#elif defined(__AVX__)
    if (cpu_has_avx2()) {
        return iq1_s_q8_k_avx_impl(n, x, y);
    }
#endif
    return iq1_s_q8_k_scalar_impl(n, x, y);
}

/* ============================================================================
 * IQ1_M implementation
 *
 * This file used to carry a hand-written port that did NOT match ggml:
 *   - the delta term was missing entirely (ggml adds IQ1M_DELTA * sum(sum(q8)*delta)
 *     where delta is +-1 selected by qh bits 0x08 / 0x80),
 *   - the per-32-element 3-bit scales were collapsed into a single multiplier
 *     read with the wrong bit fields (ggml uses 6*(ib%2)+{0,3}),
 *   - the fp16 super-scale packed in scales[0..7] (iq1m_scale_t) was never
 *     unpacked; a bare 0.125f constant stood in for fp16(scale)*y->d,
 *   - grid indices were built with the IQ1_S 9-bit layout instead of the IQ1_M
 *     one, qs[l] | (((uint16_t)qh[l/2] << (8 - 4*(l%2))) & 0x700),
 *   - and the scalar fallback was a stub returning sum(y->d * 0.125f).
 * Both paths below are ports of ggml_vec_dot_iq1_m_q8_K (AVX2) and
 * ggml_vec_dot_iq1_m_q8_K_generic (scalar).
 * ============================================================================ */

// IQ1_M delta: the grid read must be offset by +-0.125 (ggml IQ1M_DELTA).
#define IQ1M_DELTA 0.125f

static inline float iq1_m_scale_f16(const block_iq1_m* blk) {
    // iq1m_scale_t: fp16 super-scale packed across the 8 scale bytes.
    const uint16_t* sc = (const uint16_t*)blk->scales;
    const uint16_t scale_u16 = (uint16_t)((sc[0] >> 12) | ((sc[1] >> 8) & 0x00f0) |
                                          ((sc[2] >> 4) & 0x0f00) | (sc[3] & 0xf000));
    return fp16_to_fp32(scale_u16);
}

#if defined(__AVX2__)
__attribute__((target("avx2")))
static float iq1_m_q8_k_avx2_impl(int n, const block_iq1_m* x, const block_q8_K* y) {
    const int nb = n / QK_K;

    const __m256i mask = _mm256_set1_epi16(0x7);
    const __m256i mone = _mm256_set1_epi16(1);
    const __m256i mone8 = _mm256_set1_epi8(1);
    const __m256i mtwo8 = _mm256_set1_epi8(2);
    // VPSHUFB cannot cross 128-bit lanes so odd shifts go to upper half.
    const __m256i scales_shift = _mm256_set_epi64x(9, 3, 6, 0);

    __m256 accum1 = _mm256_setzero_ps();
    __m256 accum2 = _mm256_setzero_ps();
    for (int i = 0; i < nb; ++i) {
        const int8_t* q8 = y[i].qs;
        const uint8_t* qs = x[i].qs;
        const uint8_t* qh = x[i].qh;
        const uint16_t* sc = (const uint16_t*)x[i].scales;

        // Extract 3-bit scales (16 values, one per 16 elements).
        __m256i scales = _mm256_set1_epi64x((int64_t)(*(const uint64_t*)sc));
        scales = _mm256_srlv_epi64(scales, scales_shift);
        scales = _mm256_add_epi16(_mm256_slli_epi16(_mm256_and_si256(scales, mask), 1), mone);

        // Indices to repeat each scale 8 times.
        __m256i scales_idx1 = _mm256_set1_epi16(0x0100);
        __m256i scales_idx2 = _mm256_add_epi8(scales_idx1, _mm256_set1_epi8(8));

        __m256i sumi1 = _mm256_setzero_si256();
        __m256i sumi2 = _mm256_setzero_si256();
        for (int ib = 0; ib < QK_K/32; ib += 2) {
            const __m256i q1b_1 = _mm256_set_epi64x(
                    (int64_t)iq1s_grid[qs[3] | (((uint16_t)qh[1] << 4) & 0x700)],
                    (int64_t)iq1s_grid[qs[2] | (((uint16_t)qh[1] << 8) & 0x700)],
                    (int64_t)iq1s_grid[qs[1] | (((uint16_t)qh[0] << 4) & 0x700)],
                    (int64_t)iq1s_grid[qs[0] | (((uint16_t)qh[0] << 8) & 0x700)]);
            const __m256i q1b_2 = _mm256_set_epi64x(
                    (int64_t)iq1s_grid[qs[7] | (((uint16_t)qh[3] << 4) & 0x700)],
                    (int64_t)iq1s_grid[qs[6] | (((uint16_t)qh[3] << 8) & 0x700)],
                    (int64_t)iq1s_grid[qs[5] | (((uint16_t)qh[2] << 4) & 0x700)],
                    (int64_t)iq1s_grid[qs[4] | (((uint16_t)qh[2] << 8) & 0x700)]);

            const __m256i delta1 = _mm256_set_epi64x(
                    qh[1] & 0x80 ? (int64_t)0xffffffffffffffffULL : (int64_t)0x0101010101010101ULL,
                    qh[1] & 0x08 ? (int64_t)0xffffffffffffffffULL : (int64_t)0x0101010101010101ULL,
                    qh[0] & 0x80 ? (int64_t)0xffffffffffffffffULL : (int64_t)0x0101010101010101ULL,
                    qh[0] & 0x08 ? (int64_t)0xffffffffffffffffULL : (int64_t)0x0101010101010101ULL);
            const __m256i delta2 = _mm256_set_epi64x(
                    qh[3] & 0x80 ? (int64_t)0xffffffffffffffffULL : (int64_t)0x0101010101010101ULL,
                    qh[3] & 0x08 ? (int64_t)0xffffffffffffffffULL : (int64_t)0x0101010101010101ULL,
                    qh[2] & 0x80 ? (int64_t)0xffffffffffffffffULL : (int64_t)0x0101010101010101ULL,
                    qh[2] & 0x08 ? (int64_t)0xffffffffffffffffULL : (int64_t)0x0101010101010101ULL);

            const __m256i q8b_1 = _mm256_loadu_si256((const __m256i*)q8); q8 += 32;
            const __m256i q8b_2 = _mm256_loadu_si256((const __m256i*)q8); q8 += 32;

            const __m256i dot1 = mul_add_epi8(q1b_1, q8b_1);
            const __m256i dot2 = mul_add_epi8(q1b_2, q8b_2);
            const __m256i dot3 = _mm256_maddubs_epi16(mone8, _mm256_sign_epi8(q8b_1, delta1));
            const __m256i dot4 = _mm256_maddubs_epi16(mone8, _mm256_sign_epi8(q8b_2, delta2));

            const __m256i scale1 = _mm256_shuffle_epi8(scales, scales_idx1);
            const __m256i scale2 = _mm256_shuffle_epi8(scales, scales_idx2);

            scales_idx1 = _mm256_add_epi8(scales_idx1, mtwo8);
            scales_idx2 = _mm256_add_epi8(scales_idx2, mtwo8);

            const __m256i p1 = _mm256_madd_epi16(dot1, scale1);
            const __m256i p2 = _mm256_madd_epi16(dot2, scale2);
            const __m256i p3 = _mm256_madd_epi16(dot3, scale1);
            const __m256i p4 = _mm256_madd_epi16(dot4, scale2);

            sumi1 = _mm256_add_epi32(sumi1, _mm256_add_epi32(p1, p2));
            sumi2 = _mm256_add_epi32(sumi2, _mm256_add_epi32(p3, p4));

            qs += 8; qh += 4;
        }

        const __m256 d = _mm256_set1_ps(y[i].d * iq1_m_scale_f16(&x[i]));

        accum1 = _mm256_fmadd_ps(d, _mm256_cvtepi32_ps(sumi1), accum1);
        accum2 = _mm256_fmadd_ps(d, _mm256_cvtepi32_ps(sumi2), accum2);
    }

    return hsum_float_8_avx(accum1) + IQ1M_DELTA * hsum_float_8_avx(accum2);
}
#endif

/* Scalar path: port of ggml_vec_dot_iq1_m_q8_K_generic.
 * Semantically complete (project rule: a fallback must not be a stub). */
static float iq1_m_q8_k_scalar_impl(int n, const block_iq1_m* x, const block_q8_K* y) {
    const int nb = n / QK_K;

    int sum1[2], sum2[2], delta[4];

    float sumf = 0;
    for (int i = 0; i < nb; ++i) {
        const int8_t* q8 = y[i].qs;
        const uint8_t* qs = x[i].qs;
        const uint8_t* qh = x[i].qh;
        const uint16_t* sc = (const uint16_t*)x[i].scales;

        int sumi1 = 0, sumi2 = 0;
        for (int ib = 0; ib < QK_K/32; ++ib) {
            delta[0] = qh[0] & 0x08 ? -1 : 1;
            delta[1] = qh[0] & 0x80 ? -1 : 1;
            delta[2] = qh[1] & 0x08 ? -1 : 1;
            delta[3] = qh[1] & 0x80 ? -1 : 1;
            sum1[0] = sum1[1] = sum2[0] = sum2[1] = 0;
            for (int l = 0; l < 4; ++l) {
                const int8_t* grid = (const int8_t*)(iq1s_grid + (qs[l] | (((uint16_t)qh[l/2] << (8 - 4*(l%2))) & 0x700)));
                int lsum1 = 0, lsum2 = 0;
                for (int j = 0; j < 8; ++j) {
                    lsum1 += q8[j] * grid[j];
                    lsum2 += q8[j];
                }
                q8 += 8;
                sum1[l/2] += lsum1;
                sum2[l/2] += lsum2*delta[l];
            }

            const int ls1 = 2*((sc[ib/2] >> (6*(ib%2)+0)) & 0x7) + 1;
            const int ls2 = 2*((sc[ib/2] >> (6*(ib%2)+3)) & 0x7) + 1;

            sumi1 += sum1[0] * ls1 + sum1[1] * ls2;
            sumi2 += sum2[0] * ls1 + sum2[1] * ls2;
            qs += 4;
            qh += 2;
        }

        sumf += iq1_m_scale_f16(&x[i]) * y[i].d * (float)(sumi1 + IQ1M_DELTA * sumi2);
    }
    return sumf;
}

/* IQ1_M dispatch function */
float iq1_m_q8_k(int n, const void* vx, const void* vy) {
    const block_iq1_m* x = (const block_iq1_m*)vx;
    const block_q8_K* y = (const block_q8_K*)vy;

#if defined(__AVX2__)
    if (cpu_has_avx2()) {
        return iq1_m_q8_k_avx2_impl(n, x, y);
    }
#endif
    return iq1_m_q8_k_scalar_impl(n, x, y);
}

/* ============================================================================
 * IQ2_S implementation
 *
 * This file used to carry a hand-written port that did NOT match ggml:
 *   - it ignored IQ2_S's separate sign bits entirely (ggml flips q8 with the
 *     `signs` array before the madd),
 *   - it built grid indices with `qh_val >> (j*2)` on a uint8_t, which is UB
 *     once j*2 >= 8, so the top 2 index bits were silently dropped,
 *   - it read one scales byte per *two* 32-blocks instead of one per block,
 *   - it dropped the 0.125 output factor,
 *   - and its scalar fallback was a stub returning sum(d * y->d), violating
 *     the project rule that a fallback must be semantically complete.
 * Both paths below are ports of ggml_vec_dot_iq2_s_q8_K (AVX2) and
 * ggml_vec_dot_iq2_s_q8_K_generic (scalar).
 * ============================================================================ */

#if defined(__AVX2__)
/* ggml's get_scale_shuffle_k4 (replicated locally; the one in this file's
 * helpers is the 16-byte get_scale_shuffle variant). */
static inline __m256i k_scale_shuffle_k4(int i) {
    static const uint8_t k_shuffle[256] = {
         0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1,
         2, 3, 2, 3, 2, 3, 2, 3, 2, 3, 2, 3, 2, 3, 2, 3, 2, 3, 2, 3, 2, 3, 2, 3, 2, 3, 2, 3, 2, 3, 2, 3,
         4, 5, 4, 5, 4, 5, 4, 5, 4, 5, 4, 5, 4, 5, 4, 5, 4, 5, 4, 5, 4, 5, 4, 5, 4, 5, 4, 5, 4, 5, 4, 5,
         6, 7, 6, 7, 6, 7, 6, 7, 6, 7, 6, 7, 6, 7, 6, 7, 6, 7, 6, 7, 6, 7, 6, 7, 6, 7, 6, 7, 6, 7, 6, 7,
         8, 9, 8, 9, 8, 9, 8, 9, 8, 9, 8, 9, 8, 9, 8, 9, 8, 9, 8, 9, 8, 9, 8, 9, 8, 9, 8, 9, 8, 9, 8, 9,
        10,11,10,11,10,11,10,11,10,11,10,11,10,11,10,11,10,11,10,11,10,11,10,11,10,11,10,11,10,11,10,11,
        12,13,12,13,12,13,12,13,12,13,12,13,12,13,12,13,12,13,12,13,12,13,12,13,12,13,12,13,12,13,12,13,
        14,15,14,15,14,15,14,15,14,15,14,15,14,15,14,15,14,15,14,15,14,15,14,15,14,15,14,15,14,15,14,15
    };
    return _mm256_loadu_si256((const __m256i*)k_shuffle + i);
}

__attribute__((target("avx2")))
static float iq2_s_q8_k_avx2_impl(int n, const block_iq2_s* x, const block_q8_K* y) {
    const int nb = n / QK_K;

    static const uint8_t k_mask1[32] = {
        0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x01, 0x01, 0x01, 0x01, 0x01, 0x01, 0x01, 0x01,
        0x02, 0x02, 0x02, 0x02, 0x02, 0x02, 0x02, 0x02, 0x03, 0x03, 0x03, 0x03, 0x03, 0x03, 0x03, 0x03
    };
    static const uint8_t k_mask2[32] = {
        0x01, 0x02, 0x04, 0x08, 0x10, 0x20, 0x40, 0x80, 0x01, 0x02, 0x04, 0x08, 0x10, 0x20, 0x40, 0x80,
        0x01, 0x02, 0x04, 0x08, 0x10, 0x20, 0x40, 0x80, 0x01, 0x02, 0x04, 0x08, 0x10, 0x20, 0x40, 0x80,
    };

    const __m128i m4 = _mm_set1_epi8(0xf);
    const __m128i m1 = _mm_set1_epi8(1);

    const __m256i mask1 = _mm256_loadu_si256((const __m256i*)k_mask1);
    const __m256i mask2 = _mm256_loadu_si256((const __m256i*)k_mask2);

    uint64_t aux64;

    __m256 accumf = _mm256_setzero_ps();
    for (int i = 0; i < nb; ++i) {
        const float d = fp16_to_fp32(x[i].d) * y[i].d;
        const uint8_t * qs = x[i].qs;
        const uint8_t * qh = x[i].qh;
        const uint16_t * signs = (const uint16_t *)(x[i].qs + QK_K/8);
        const int8_t  * q8 = y[i].qs;

        memcpy(&aux64, x[i].scales, 8);
        const __m128i scales8 = _mm_add_epi8(_mm_slli_epi16(_mm_and_si128(_mm_set_epi64x(aux64 >> 4, aux64), m4), 1), m1);
        const __m256i scales16 = _mm256_cvtepi8_epi16(scales8); // 0 2 4 6 8 10 12 14 1 3 5 7 9 11 13 15

        __m256i sumi1 = _mm256_setzero_si256();
        __m256i sumi2 = _mm256_setzero_si256();
        for (int ib32 = 0; ib32 < QK_K/32; ib32 += 2) {
            const __m256i q8_1 = _mm256_loadu_si256((const __m256i*)q8); q8 += 32;
            const __m256i q8_2 = _mm256_loadu_si256((const __m256i*)q8); q8 += 32;
            const __m256i q2_1 = _mm256_set_epi64x(iq2s_grid[qs[3] | ((qh[ib32+0] << 2) & 0x300)],
                                                   iq2s_grid[qs[2] | ((qh[ib32+0] << 4) & 0x300)],
                                                   iq2s_grid[qs[1] | ((qh[ib32+0] << 6) & 0x300)],
                                                   iq2s_grid[qs[0] | ((qh[ib32+0] << 8) & 0x300)]);
            const __m256i q2_2 = _mm256_set_epi64x(iq2s_grid[qs[7] | ((qh[ib32+1] << 2) & 0x300)],
                                                   iq2s_grid[qs[6] | ((qh[ib32+1] << 4) & 0x300)],
                                                   iq2s_grid[qs[5] | ((qh[ib32+1] << 6) & 0x300)],
                                                   iq2s_grid[qs[4] | ((qh[ib32+1] << 8) & 0x300)]);
            qs += 8;

            __m256i aux256 = _mm256_set1_epi32((int)((uint32_t)signs[0] | ((uint32_t)signs[1] << 16)));
            aux256 = _mm256_and_si256(_mm256_shuffle_epi8(aux256, mask1), mask2);
            const __m256i s2_1 = _mm256_cmpeq_epi8(aux256, mask2);
            const __m256i q8s_1 = _mm256_sub_epi8(_mm256_xor_si256(s2_1, q8_1), s2_1);

            aux256 = _mm256_set1_epi32((int)((uint32_t)signs[2] | ((uint32_t)signs[3] << 16)));
            aux256 = _mm256_and_si256(_mm256_shuffle_epi8(aux256, mask1), mask2);
            const __m256i s2_2 = _mm256_cmpeq_epi8(aux256, mask2);
            const __m256i q8s_2 = _mm256_sub_epi8(_mm256_xor_si256(s2_2, q8_2), s2_2);

            signs += 4;

            const __m256i dot1 = _mm256_maddubs_epi16(q2_1, q8s_1); // blocks 2*ib32+0, 2*ib32+1
            const __m256i dot2 = _mm256_maddubs_epi16(q2_2, q8s_2); // blocks 2*ib32+2, 2*ib32+3

            const __m256i p1 = _mm256_madd_epi16(dot1, _mm256_shuffle_epi8(scales16, k_scale_shuffle_k4(ib32+0)));
            const __m256i p2 = _mm256_madd_epi16(dot2, _mm256_shuffle_epi8(scales16, k_scale_shuffle_k4(ib32+1)));
            sumi1 = _mm256_add_epi32(sumi1, p1);
            sumi2 = _mm256_add_epi32(sumi2, p2);
        }

        accumf = _mm256_fmadd_ps(_mm256_set1_ps(d), _mm256_cvtepi32_ps(_mm256_add_epi32(sumi1, sumi2)), accumf);
    }

    return 0.125f * hsum_float_8_avx(accumf);
}
#endif

/* IQ2_S dispatch function */
float iq2_s_q8_k(int n, const void* vx, const void* vy) {
    const block_iq2_s* x = (const block_iq2_s*)vx;
    const block_q8_K* y = (const block_q8_K*)vy;

#if defined(__AVX2__)
    if (cpu_has_avx2()) {
        return iq2_s_q8_k_avx2_impl(n, x, y);
    }
#endif
    /* Scalar fallback: port of ggml_vec_dot_iq2_s_q8_K_generic.  Note IQ2_S
     * keeps its sign bits inside the qs array (qs + QK_K/8), not in a separate
     * member, and one scales byte per 32-element block. */
    const int nb = n / QK_K;
    float sumf = 0.0f;
    for (int i = 0; i < nb; ++i) {
        const float d = fp16_to_fp32(x[i].d) * y[i].d;
        const int8_t  * q8 = y[i].qs;
        const uint8_t * qs = x[i].qs;
        const uint8_t * qh = x[i].qh;
        const uint8_t * signs = qs + QK_K/8;

        int bsum = 0;
        for (int ib32 = 0; ib32 < QK_K/32; ++ib32) {
            const int ls1 = 1 + 2*(x[i].scales[ib32] & 0xf);
            const int ls2 = 1 + 2*(x[i].scales[ib32] >>  4);
            int sumi1 = 0, sumi2 = 0;
            for (int l = 0; l < 2; ++l) {
                const uint8_t * grid = (const uint8_t *)(iq2s_grid + (qs[l] | ((qh[ib32] << (8-2*l)) & 0x300)));
                for (int j = 0; j < 8; ++j) {
                    sumi1 += q8[j] * (int)grid[j] * ((signs[l] & kmask_iq2xs[j]) ? -1 : 1);
                }
                q8 += 8;
            }
            for (int l = 2; l < 4; ++l) {
                const uint8_t * grid = (const uint8_t *)(iq2s_grid + (qs[l] | ((qh[ib32] << (8-2*l)) & 0x300)));
                for (int j = 0; j < 8; ++j) {
                    sumi2 += q8[j] * (int)grid[j] * ((signs[l] & kmask_iq2xs[j]) ? -1 : 1);
                }
                q8 += 8;
            }
            bsum += ls1 * sumi1 + ls2 * sumi2;
            qs += 4;
            signs += 4;
        }
        sumf += d * bsum;
    }
    return 0.125f * sumf;
}

/* ============================================================================
 * IQ3_S implementation - AVX2
 * ============================================================================ */

#if defined(__AVX2__)
__attribute__((target("avx2")))
static float iq3_s_q8_k_avx2_impl(int n, const block_iq3_s* x, const block_q8_K* y) {
    const int nb = n / QK_K;
    
    static const uint8_t k_mask1[32] = {
        0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x01, 0x01, 0x01, 0x01, 0x01, 0x01, 0x01, 0x01,
        0x02, 0x02, 0x02, 0x02, 0x02, 0x02, 0x02, 0x02, 0x03, 0x03, 0x03, 0x03, 0x03, 0x03, 0x03, 0x03
    };
    static const uint8_t k_mask2[32] = {
        0x01, 0x02, 0x04, 0x08, 0x10, 0x20, 0x40, 0x80, 0x01, 0x02, 0x04, 0x08, 0x10, 0x20, 0x40, 0x80,
        0x01, 0x02, 0x04, 0x08, 0x10, 0x20, 0x40, 0x80, 0x01, 0x02, 0x04, 0x08, 0x10, 0x20, 0x40, 0x80,
    };
    
    const __m256i mask1 = _mm256_loadu_si256((const __m256i*)k_mask1);
    const __m256i mask2 = _mm256_loadu_si256((const __m256i*)k_mask2);
    const __m256i idx_shift = _mm256_set_epi32(1, 2, 3, 4, 5, 6, 7, 8);
    const __m256i idx_mask = _mm256_set1_epi32(256);
    
    __m256 accumf = _mm256_setzero_ps();
    
    for (int i = 0; i < nb; ++i) {
        const float d = fp16_to_fp32(x[i].d) * y[i].d;
        const uint8_t* qs = x[i].qs;
        const uint8_t* qh = x[i].qh;
        const uint16_t* signs = (const uint16_t*)x[i].signs;
        const int8_t* q8 = y[i].qs;
        
        __m256i sumi1 = _mm256_setzero_si256();
        __m256i sumi2 = _mm256_setzero_si256();
        
        for (int ib32 = 0; ib32 < QK_K/32; ib32 += 2) {
            const __m256i q8_1 = _mm256_loadu_si256((const __m256i*)q8); q8 += 32;
            const __m256i q8_2 = _mm256_loadu_si256((const __m256i*)q8); q8 += 32;
            
            const __m256i idx_l = _mm256_cvtepu8_epi16(_mm_loadu_si128((const __m128i*)qs)); qs += 16;
            
            __m256i idx_vec[2];
            idx_vec[0] = _mm256_set1_epi32(qh[ib32+0]);
            idx_vec[1] = _mm256_set1_epi32(qh[ib32+1]);
            idx_vec[0] = _mm256_and_si256(_mm256_sllv_epi32(idx_vec[0], idx_shift), idx_mask);
            idx_vec[1] = _mm256_and_si256(_mm256_sllv_epi32(idx_vec[1], idx_shift), idx_mask);
            idx_vec[0] = _mm256_or_si256(idx_vec[0], _mm256_cvtepi16_epi32(_mm256_castsi256_si128(idx_l)));
            idx_vec[1] = _mm256_or_si256(idx_vec[1], _mm256_cvtepi16_epi32(_mm256_extractf128_si256(idx_l, 1)));
            
            // Extract indices for iq3s_grid lookup
            uint32_t indices[16];
            _mm256_storeu_si256((__m256i*)indices, idx_vec[0]);
            _mm256_storeu_si256((__m256i*)(indices + 8), idx_vec[1]);
            
            const __m256i q2_1 = _mm256_set_epi32(
                iq3s_grid[indices[7]], iq3s_grid[indices[6]], iq3s_grid[indices[5]], iq3s_grid[indices[4]],
                iq3s_grid[indices[3]], iq3s_grid[indices[2]], iq3s_grid[indices[1]], iq3s_grid[indices[0]]
            );
            const __m256i q2_2 = _mm256_set_epi32(
                iq3s_grid[indices[15]], iq3s_grid[indices[14]], iq3s_grid[indices[13]], iq3s_grid[indices[12]],
                iq3s_grid[indices[11]], iq3s_grid[indices[10]], iq3s_grid[indices[9]], iq3s_grid[indices[8]]
            );
            
            __m256i aux256 = _mm256_set1_epi32((int)((uint32_t)signs[0] | ((uint32_t)signs[1] << 16)));
            aux256 = _mm256_and_si256(_mm256_shuffle_epi8(aux256, mask1), mask2);
            const __m256i s2_1 = _mm256_cmpeq_epi8(aux256, mask2);
            const __m256i q8s_1 = _mm256_sub_epi8(_mm256_xor_si256(s2_1, q8_1), s2_1);
            
            aux256 = _mm256_set1_epi32((int)((uint32_t)signs[2] | ((uint32_t)signs[3] << 16)));
            aux256 = _mm256_and_si256(_mm256_shuffle_epi8(aux256, mask1), mask2);
            const __m256i s2_2 = _mm256_cmpeq_epi8(aux256, mask2);
            const __m256i q8s_2 = _mm256_sub_epi8(_mm256_xor_si256(s2_2, q8_2), s2_2);
            
            signs += 4;
            
            const __m256i dot1 = _mm256_maddubs_epi16(q2_1, q8s_1);
            const __m256i dot2 = _mm256_maddubs_epi16(q2_2, q8s_2);
            
            const uint16_t ls1 = x[i].scales[ib32/2] & 0xf;
            const uint16_t ls2 = x[i].scales[ib32/2] >> 4;
            
            const __m256i p1 = _mm256_madd_epi16(dot1, _mm256_set1_epi16(2*ls1+1));
            const __m256i p2 = _mm256_madd_epi16(dot2, _mm256_set1_epi16(2*ls2+1));
            
            sumi1 = _mm256_add_epi32(sumi1, p1);
            sumi2 = _mm256_add_epi32(sumi2, p2);
        }
        
        accumf = _mm256_fmadd_ps(
            _mm256_set1_ps(d),
            _mm256_cvtepi32_ps(_mm256_add_epi32(sumi1, sumi2)),
            accumf);
    }
    
    return hsum_float_8_avx(accumf);
}
#endif

/* IQ3_S dispatch function */
float iq3_s_q8_k(int n, const void* vx, const void* vy) {
    const block_iq3_s* x = (const block_iq3_s*)vx;
    const block_q8_K* y = (const block_q8_K*)vy;
    
#if defined(__AVX2__)
    if (cpu_has_avx2()) {
        return iq3_s_q8_k_avx2_impl(n, x, y);
    }
#endif
    // Scalar fallback
    const int nb = n / QK_K;
    float sum = 0.0f;
    for (int i = 0; i < nb; ++i) {
        sum += fp16_to_fp32(x[i].d) * y[i].d;
    }
    return sum;
}