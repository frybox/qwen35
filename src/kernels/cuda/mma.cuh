// mma.cuh -- int8 tensor-core (mma.sync) primitives for the prefill MMQ path.
//
// VERBATIM PORT of the NVIDIA int8 subset of llama.cpp ggml-cuda/mma.cuh
// (tile fragment layouts, swizzle, load_ldmatrix, and the two int8 mma
// overloads).  Only the shapes the int8 MMQ path instantiates are kept:
// int tiles 8x4 / 8x8 / 16x8 / 16x16 / 32x8, both data layouts, the b16
// ldmatrix loaders, and mma.sync m16n8k16 / m16n8k32 s32.s8.s8.s32.
// AMD / Volta / f16 branches are dropped (qwen35 targets NVIDIA only).
//
// The get_i/get_j fragment layouts are the make-or-break part: they must
// match the mma.sync register fragment layout EXACTLY, so this file is a
// port, not a re-derivation.  Any "cleanup" that changes an offset breaks
// correctness in ways that look like random garbage.
//
// Naming: `q35_tile` instead of `tile` to avoid clashing with the C
// standard library / other headers; everything else is named as in the
// source (load_ldmatrix, swizzle, mma) so the port can be diffed line by
// line against llama.cpp.
#pragma once

#include <cstdint>

#define Q35_MMA_WARP_SIZE 32
#define Q35_CC_AMPERE 800

enum q35_data_layout {
    Q35_DATA_LAYOUT_I_MAJOR,
    Q35_DATA_LAYOUT_J_MAJOR,
};

template <int I_, int J_, typename T = int, q35_data_layout dl = Q35_DATA_LAYOUT_I_MAJOR>
struct q35_tile;

// I_MAJOR (the common case).  `x` holds `ne` register values; for int tiles
// each int is FOUR int8 elements, so the tile spans I rows x J*4 int8 in
// the J direction (that is why A = tile<16,8,int> is a 16 x 32-int8 tile).
template <int I_, int J_, typename T, q35_data_layout dl>
struct q35_tile {
    static constexpr int I = I_;
    static constexpr int J = J_;
    static constexpr q35_data_layout dl_ = dl;

    static constexpr int ne = I * J / Q35_MMA_WARP_SIZE;
    T x[ne] = {0};

    static constexpr __device__ bool supported() {
        if (I ==  8 && J ==  4) return true;
        if (I ==  8 && J ==  8) return true;
        if (I == 16 && J ==  8) return true;
        if (I == 16 && J == 16) return true;
        if (I == 32 && J ==  8) return true;
        return false;
    }

    // Element coordinates of register `l`.  For int tiles the j returned
    // here is in UNITS OF int (4 int8), i.e. the int8 column is
    // 4 * get_j(l) + {0,1,2,3}; the row i is an exact element index.
    static __device__ __forceinline__ int get_i(const int l) {
        if constexpr (I == 8 && J == 4) {
            return (threadIdx.x & 31) / 4;
        } else if constexpr (I == 8 && J == 8) {
            return (threadIdx.x & 31) / 4;
        } else if constexpr (I == 16 && J == 8) {
            return ((l / 2) * 8) + ((threadIdx.x & 31) / 4);
        } else if constexpr (I == 16 && J == 16) {
            return (((l / 2) % 2) * 8) + ((threadIdx.x & 31) / 4);
        } else if constexpr (I == 32 && J == 8) {
            return q35_tile<16, 8, T>::get_i(l); // repeated pattern in i direction
        } else {
            return -1;
        }
    }

    static __device__ __forceinline__ int get_j(const int l) {
        if constexpr (I == 8 && J == 4) {
            return (threadIdx.x & 31) % 4;
        } else if constexpr (I == 8 && J == 8) {
            return (l * 4) + ((threadIdx.x & 31) % 4);
        } else if constexpr (I == 16 && J == 8) {
            return (((threadIdx.x & 31) % 4) * 2) + (l % 2);
        } else if constexpr (I == 16 && J == 16) {
            return ((l / 4) * 8) + (((threadIdx.x & 31) % 4) * 2) + (l % 2);
        } else if constexpr (I == 32 && J == 8) {
            return q35_tile<16, 8, T>::get_j(l); // repeated pattern in i direction
        } else {
            return -1;
        }
    }
};

// J_MAJOR: same ne, get_i/get_j swapped (used when the same physical tile is
// addressed transposed).
template <int I_, int J_, typename T>
struct q35_tile<I_, J_, T, Q35_DATA_LAYOUT_J_MAJOR> {
    static constexpr int I = I_;
    static constexpr int J = J_;
    static constexpr q35_data_layout dl_ = Q35_DATA_LAYOUT_J_MAJOR;

    static constexpr int ne = q35_tile<I_, J_, T, Q35_DATA_LAYOUT_I_MAJOR>::ne;
    T x[ne] = {0};

    static constexpr __device__ bool supported() {
        return q35_tile<I_, J_, T, Q35_DATA_LAYOUT_I_MAJOR>::supported();
    }

    static __device__ __forceinline__ int get_i(const int l) {
        return q35_tile<I_, J_, T, Q35_DATA_LAYOUT_I_MAJOR>::get_j(l);
    }

    static __device__ __forceinline__ int get_j(const int l) {
        return q35_tile<I_, J_, T, Q35_DATA_LAYOUT_I_MAJOR>::get_i(l);
    }
};

namespace q35_mma {

// Element-wise (no ldmatrix) load of a tile from row-major shared memory.
// llama.cpp's MMQ uses this for the activation B tile because it is faster
// than ldmatrix there (see mmq-vec-dot.cuh: "faster than load_ldmatrix").
template <int I, int J, typename T, q35_data_layout dl>
static __device__ __forceinline__ void load_generic(q35_tile<I, J, T, dl> & t,
                                                    const T * __restrict__ xs0,
                                                    const int stride) {
#pragma unroll
    for (int l = 0; l < t.ne; ++l) {
        t.x[l] = xs0[t.get_i(l) * stride + t.get_j(l)];
    }
}

// Bank-conflict swizzle for the ldmatrix rows.  `offset` is in units of T,
// `i` is the row index being addressed.  Verbatim from the source.
template <int stride, typename T>
static __device__ __forceinline__ uint32_t swizzle(const uint32_t offset, const uint32_t i) {
    static_assert(sizeof(T) <= 4, "unsupported type size");
    constexpr int stride_bytes = stride * sizeof(T);
    static_assert(stride_bytes % 16 == 0, "bad stride");
    constexpr uint32_t shift = sizeof(T) == 1 ? 4 : (sizeof(T) == 2 ? 3 : 2);
    if (stride_bytes % 32 != 0) {
        return offset; // Equivalent to padding with 16 bytes.
    }
    if (stride_bytes % 64 != 0) {
        return offset ^ (((i / 4) % 2) << shift);
    }
    if (stride_bytes % 128 != 0) {
        return offset ^ (((i / 2) % 4) << shift);
    }
    return offset ^ ((i % 8) << shift);
}

// 8x8 int tile: 8 rows x 8 int (32 int8).  Loads 2 m8n8 b16 matrices.
static __device__ __forceinline__ void load_ldmatrix(
        q35_tile<8, 8, int> & t, const int * __restrict__ xs0, const int stride) {
    int * xi = (int *) t.x;
    const int * xs = xs0 + ((threadIdx.x & 31) % 8) * stride + (((threadIdx.x & 31) / 8) * (8 / 2)) % 8;
    asm volatile("ldmatrix.sync.aligned.m8n8.x2.b16 {%0, %1}, [%2];"
        : "=r"(xi[0]), "=r"(xi[1])
        : "l"(xs));
}

// 16x4 int tile: 16 rows x 4 int (16 int8).
static __device__ __forceinline__ void load_ldmatrix(
        q35_tile<16, 4, int> & t, const int * __restrict__ xs0, const int stride) {
    int * xi = (int *) t.x;
    const int * xs = xs0 + ((threadIdx.x & 31) % 16) * stride;
    asm volatile("ldmatrix.sync.aligned.m8n8.x2.b16 {%0, %1}, [%2];"
        : "=r"(xi[0]), "=r"(xi[1])
        : "l"(xs));
}

// 16x8 int tile: 16 rows x 8 int (32 int8).
static __device__ __forceinline__ void load_ldmatrix(
        q35_tile<16, 8, int> & t, const int * __restrict__ xs0, const int stride) {
    int * xi = (int *) t.x;
    const int * xs = xs0 + ((threadIdx.x & 31) % 16) * stride + ((threadIdx.x & 31) / 16) * (8 / 2);
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.b16 {%0, %1, %2, %3}, [%4];"
        : "=r"(xi[0]), "=r"(xi[1]), "=r"(xi[2]), "=r"(xi[3])
        : "l"(xs));
}

// int8 mma, k = 16: D[m,n] += sum_k A[m,k]*B[k,n],  A 16x16, B 16x8, int8.
// A fragment: tile<16,4,int> (2 int = 8 int8 per thread), B: tile<8,4,int>
// (1 int = 4 int8), D: tile<16,8,int> (4 int32).
static __device__ __forceinline__ void mma(
        q35_tile<16, 8, int> & D, const q35_tile<16, 4, int> & A, const q35_tile<8, 4, int> & B) {
#if __CUDA_ARCH__ >= Q35_CC_AMPERE
    asm("mma.sync.aligned.m16n8k16.row.col.s32.s8.s8.s32 {%0, %1, %2, %3}, {%4, %5}, {%6}, {%0, %1, %2, %3};"
        : "+r"(D.x[0]), "+r"(D.x[1]), "+r"(D.x[2]), "+r"(D.x[3])
        : "r"(A.x[0]), "r"(A.x[1]), "r"(B.x[0]));
#else
    // Turing: no m16n8k16, use 2x m8n8k16 instead.
    asm("mma.sync.aligned.m8n8k16.row.col.s32.s8.s8.s32 {%0, %1}, {%2}, {%3}, {%0, %1};"
        : "+r"(D.x[0]), "+r"(D.x[1])
        : "r"(A.x[0]), "r"(B.x[0]));
    asm("mma.sync.aligned.m8n8k16.row.col.s32.s8.s8.s32 {%0, %1}, {%2}, {%3}, {%0, %1};"
        : "+r"(D.x[2]), "+r"(D.x[3])
        : "r"(A.x[1]), "r"(B.x[0]));
#endif
}

// int8 mma, k = 32: A 16x32, B 32x8, int8.
// A fragment: tile<16,8,int> (4 int = 16 int8), B: tile<8,8,int> (2 int =
// 8 int8), D: tile<16,8,int> (4 int32).
static __device__ __forceinline__ void mma(
        q35_tile<16, 8, int> & D, const q35_tile<16, 8, int> & A, const q35_tile<8, 8, int> & B) {
#if __CUDA_ARCH__ >= Q35_CC_AMPERE
    asm("mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};"
        : "+r"(D.x[0]), "+r"(D.x[1]), "+r"(D.x[2]), "+r"(D.x[3])
        : "r"(A.x[0]), "r"(A.x[1]), "r"(A.x[2]), "r"(A.x[3]), "r"(B.x[0]), "r"(B.x[1]));
#else
    // Turing: 4x m8n8k16.
    asm("mma.sync.aligned.m8n8k16.row.col.s32.s8.s8.s32 {%0, %1}, {%2}, {%3}, {%0, %1};"
        : "+r"(D.x[0]), "+r"(D.x[1])
        : "r"(A.x[0]), "r"(B.x[0]));
    asm("mma.sync.aligned.m8n8k16.row.col.s32.s8.s8.s32 {%0, %1}, {%2}, {%3}, {%0, %1};"
        : "+r"(D.x[2]), "+r"(D.x[3])
        : "r"(A.x[1]), "r"(B.x[0]));
    asm("mma.sync.aligned.m8n8k16.row.col.s32.s8.s8.s32 {%0, %1}, {%2}, {%3}, {%0, %1};"
        : "+r"(D.x[0]), "+r"(D.x[1])
        : "r"(A.x[2]), "r"(B.x[1]));
    asm("mma.sync.aligned.m8n8k16.row.col.s32.s8.s8.s32 {%0, %1}, {%2}, {%3}, {%0, %1};"
        : "+r"(D.x[2]), "+r"(D.x[3])
        : "r"(A.x[3]), "r"(B.x[1]));
#endif
}

} // namespace q35_mma
