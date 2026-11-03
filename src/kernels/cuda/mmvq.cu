// mmvq.cu -- quantized matrix-vector products, ALL weight types, plus the
// activation quantizers they need (Q8_0 for the 32-element block family, Q8_K
// for the 256-element one) and the f32 path.
//
// ---------------------------------------------------------------------------
// DESIGN: one thread per output element, one type-dispatched dot function
// ---------------------------------------------------------------------------
// Anchors:
//   ggml/src/ggml-cuda/mmvq.cu          mul_mat_vec_q (grid/shape idea)
//   ggml/src/ggml-cuda/vecdotq.cuh      per-type vec_dot_* (numerics live in
//                                       this repo's vecdotq.cuh instead)
//   src/kernel.cu:239 (deleted)           k_matvec_q8 (dp4a + prefetch idea)
//
// DELIBERATE DIFFERENCE from the retired src/kernel.cu (deleted): that kernel
// parallelises a row across a warp and reduces with shuffles.  Here one thread
// owns one whole row
// and accumulates strictly left to right (block by block), which is exactly
// the CPU kernel's float addition tree.  A warp reduction reorders that tree,
// was measured to be ~1 ulp off, and compounds over 150+ matvecs per token into
// a different sampled token.  The acceptance criterion is an identical token
// stream, not a tolerance -- so the one-thread-per-row mapping stays, and the
// per-type functions in vecdotq.cuh are written to fold one block at a time.
//
// The "type table" is a compile-time switch (`q35_dot_row`) rather than an
// array of device function pointers: nvcc resolves the switch to a branch, and
// keeping it as code lets every type's block loop be unrolled.  The type list,
// block sizes and activation choice come from ONE table (q35_quant_info below)
// so a new type cannot be half-wired.
//
// ---------------------------------------------------------------------------
// ACTIVATION SIDE
// ---------------------------------------------------------------------------
// 32-element family (q8_0/q4_0/q4_1/q5_0/q5_1/iq4_nl) uses Q8_0 activations:
//   34 B per 32 elements, one quantized copy per (token, column range).
// 256-element family (q2_k..q6_k, iq1_s..iq4_xs) uses Q8_K activations:
//   292 B per 256 elements, d(f32) @0, qs[256] @4, bsums[16] @260.
// Both are produced ON THE GPU (q35_launch_quantize_q8_0 / _q8_K); nothing
// falls back to the CPU.
#include "common.cuh"
#include "vecdotq.cuh"
#include "api.h"

#include <stdlib.h>

// The per-type geometry table and the block/row dot helpers moved to
// dotq.cuh so that mmq.cu (the prefill tile kernel) can use the exact same
// definitions instead of a second copy (AGENTS.md: one source for one fact).
#include "dotq.cuh"
// ---------------------------------------------------------------------------
// matvec kernel: out[t*rows + r] = W[r] . x[t]
// ---------------------------------------------------------------------------
__global__ void q35_k_mul_mat_vec_q(const unsigned char *__restrict__ w,
                                    const unsigned char *__restrict__ xq,
                                    float *__restrict__ out,
                                    int rows, int cols, int tokens, int ttype) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int total = rows * tokens;
    if (idx >= total) return;
    const Q35QuantInfo info = q35_quant_info(ttype);
    const int nblk = cols / info.block_elems;
    const int t = idx / rows;
    const int r = idx - t * rows;

    const unsigned char *wrow = w + (size_t)r * (size_t)nblk * (size_t)info.block_bytes;
    const unsigned char *arow = xq + (size_t)t * (size_t)nblk * (size_t)info.act_bytes;
    out[(size_t)t * rows + r] =
        q35_dot_row(ttype, wrow, arow, info.block_bytes, info.act_bytes, nblk);
}

// ---------------------------------------------------------------------------
// Cooperative-row variant: one warp per output row.
//
// Why this exists.  The kernel above maps thread -> row, so the 32 threads of a
// warp walk 32 *different* rows: at any step their addresses are
// `nblk * block_bytes` apart (2200 B for IQ3_XXS at cols=5120), i.e. every load
// touches its own 128 B sector and the warp uses 1/128th of each.  Measured
// effect: the batchK path ran at ~80 GB/s of the 4090D's ~1008 GB/s.
// Flipping the mapping so a warp covers consecutive *blocks of one row* makes
// the warp read a contiguous 32 x block_bytes stretch -> fully coalesced.
//
// Reduced precision concern: the reduction order is NOT changed.  Each lane
// computes one block's term with `q35_dot_1blk`, the partial sums go to shared
// memory, and lane 0 folds them strictly b = 0 .. nblk-1 -- the same left-to-
// right order `q35_dot_row` uses, so the result is bit-identical (the file
// header forbids reordering; this respects it).
//
// Block shape is a template parameter (WARPS = warps per block, one row per
// warp).  `part` must be exactly WARPS deep: it is indexed by the warp number,
// which runs 0..WARPS-1, so the launcher's block size (WARPS*Q35_WARP threads)
// and this dimension are ONE decision -- q35_coop_warps() below picks it and
// the host-side table instantiates every shape in the 1..8 domain.  WARPS=4
// (128 threads) is the measured-best shape on sm_89; see q35_coop_warps().
// ---------------------------------------------------------------------------
template <int WARPS>
__global__ void q35_k_mul_mat_vec_q_coop_t(const unsigned char *__restrict__ w,
                                           const unsigned char *__restrict__ xq,
                                           float *__restrict__ out,
                                           int rows, int cols, int tokens, int ttype) {
    __shared__ float part[WARPS][Q35_WARP];
    const Q35QuantInfo info = q35_quant_info(ttype);
    const int nblk = cols / info.block_elems;
    const int lane = threadIdx.x & (Q35_WARP - 1);
    const int warp = threadIdx.x / Q35_WARP;
    if (nblk > Q35_WARP) return;  // caller picks the other kernel in that case

    const int row_in_block = blockIdx.x * WARPS + warp;
    if (row_in_block >= rows) return;

    const unsigned char *wrow = w + (size_t)row_in_block * (size_t)nblk * (size_t)info.block_bytes;

    // Loop over tokens so a prefill chunk reuses the same tile mapping.
    for (int t = 0; t < tokens; ++t) {
        const unsigned char *arow = xq + (size_t)t * (size_t)nblk * (size_t)info.act_bytes;

        // Lane `b` owns block b: the warp reads nblk consecutive blocks, one per
        // lane, which is what makes the load coalesced.
        float s = 0.0f;
        if (lane < nblk) {
            s = q35_dot_1blk(ttype, wrow + (size_t)lane * (size_t)info.block_bytes,
                             arow + (size_t)lane * (size_t)info.act_bytes, 0.0f);
        }
        part[warp][lane] = s;
        __syncwarp();

        if (lane == 0) {
            // Strict left-to-right fold over b = 0..nblk-1: the same order
            // q35_dot_row produces, so the value is bit-identical.
            float acc = 0.0f;
            for (int b = 0; b < nblk; ++b) acc = __fadd_rn(acc, part[warp][b]);
            out[(size_t)t * rows + row_in_block] = acc;
        }
    }
}

// f32 weights x f32 activation, same one-thread-per-element mapping.
__global__ void q35_k_mul_mat_vec_f32(const float *__restrict__ w,
                                      const float *__restrict__ x,
                                      float *__restrict__ out,
                                      int rows, int cols, int tokens) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= rows * tokens) return;
    const int t = idx / rows;
    const int r = idx - t * rows;
    const float *wrow = w + (size_t)r * (size_t)cols;
    const float *xrow = x + (size_t)t * (size_t)cols;
    float acc = 0.0f;
    for (int c = 0; c < cols; ++c) acc = fmaf(wrow[c], xrow[c], acc);
    out[(size_t)t * rows + r] = acc;
}

// ---------------------------------------------------------------------------
// Q8_K activation quantizer (f32 -> block_q8_K, 292 B / 256 elems)
//
// Bit-exact port of src/kernels/x86/kernels256.c:392 quantize_q8_K.  Three
// details that are easy to get wrong and that this reproduces deliberately:
//   1. `scale` goes into the header as a raw f32 (NOT rounded to f16 as Q8_0
//      does), and `inv` is built from that same f32 scale.
//   2. Rounding is roundf() = round half away from zero.
//   3. The clamp is [-127, 127], NOT [-128, 127]: the IQ kernels negate q8
//      (x86 _mm256_sign_epi8 in the CPU kernel, a plain negate on device), and
//      -128 has no positive counterpart.  The `!(r >= -127) -> 127` line is the
//      NaN/Inf guard (NaN compares false and would otherwise become garbage).
//   4. bsums[16] are int16 sums over each 16-element group of qs, little-endian,
//      consumed by the Q2_K/Q4_K/Q5_K min terms and the IQ1 delta terms.
// One thread per block (a block is 256 elements, so per-thread work is fine).
// ---------------------------------------------------------------------------
__global__ void q35_k_quantize_q8_K(const float *__restrict__ x,
                                    unsigned char *__restrict__ out,
                                    int nblocks) {
    const int b = blockIdx.x * blockDim.x + threadIdx.x;
    if (b >= nblocks) return;
    const float *xb = x + (size_t)b * Q35_QK8K;
    unsigned char *ob = out + (size_t)b * Q35_BYTES_Q8_K;

    float amax = 0.0f;
    for (int k = 0; k < Q35_QK8K; ++k) {
        const float a = xb[k] < 0.0f ? -xb[k] : xb[k];
        if (a > amax) amax = a;
    }
    const float scale = amax > 0.0f ? amax / 127.0f : 1.0f;
    const float inv = amax > 0.0f ? 1.0f / scale : 0.0f;
    const unsigned int sbits = __float_as_uint(scale);
    ob[0] = (unsigned char)(sbits & 0xFFu);
    ob[1] = (unsigned char)((sbits >> 8) & 0xFFu);
    ob[2] = (unsigned char)((sbits >> 16) & 0xFFu);
    ob[3] = (unsigned char)((sbits >> 24) & 0xFFu);

    int gsum[16];
#pragma unroll
    for (int g = 0; g < 16; ++g) gsum[g] = 0;
    for (int k = 0; k < Q35_QK8K; ++k) {
        float r = roundf(xb[k] * inv);
        if (r < -127.0f) r = -127.0f;
        if (r > 127.0f) r = 127.0f;
        if (!(r >= -127.0f)) r = 127.0f;  // NaN/Inf guard, see note 3
        const signed char q = (signed char)(int)r;
        ob[4 + k] = (unsigned char)q;
        gsum[k / 16] += (int)q;
    }
    for (int g = 0; g < 16; ++g) {
        const short v = (short)gsum[g];
        ob[260 + g * 2 + 0] = (unsigned char)((unsigned short)v & 0xFFu);
        ob[260 + g * 2 + 1] = (unsigned char)(((unsigned short)v >> 8) & 0xFFu);
    }
}

// ---------------------------------------------------------------------------
// launchers
//
// Grid sizing lives here because Zig cannot express `<<< >>>`.  Errors are
// returned as the numeric cudaError_t (0 == cudaSuccess) so the Zig side can
// surface cudaGetErrorString().
// ---------------------------------------------------------------------------

// ---------------------------------------------------------------------------
// `$QWEN35_CUDA_MMQ=0|1`: which kernel serves a prefill (tokens > 1) matvec.
//
//   1 (default)  the tile kernel in mmq.cu -- K-dimension reuse (a weight block
//                is dotted into every token of the chunk) plus coalesced loads
//                (a warp reads 32 consecutive blocks of ONE row), which is
//                what the thread-per-row kernel below cannot do, and what the
//                warp-per-row kernel can only do per token, serially.
//   0            the thread-per-row kernel, kept as the A/B reference (and as
//                the fallback, see below).
//
// Any other value (including unset) keeps the default.  Read once and cached:
// this is per launch and `getenv` per launch is the kind of syscall this
// backend exists to avoid.  A launch failure is NOT fatal and is NOT silent
// either: the message names the kernel and the tile, and the old kernel runs.
static bool q35_mmq_enabled() {
    static int cached = -1;
    if (cached < 0) {
        // Default OFF.  The tile kernel is correct (bit-exact against the CPU
        // reference for all 13 quant types on the 27B model, and it keeps the
        // b-ascending reduction order) but it is not FASTER, so it must not be
        // the default: measured on 354-token prefill, Qwen3.8-27B IQ3_XXS,
        // 4090D -- thread-per-row 30.4-30.9 s vs tile 30.4 s at TPERPASS=8
        // (parity) and 38.7 s at TPERPASS=16 (22% slower).
        //
        // The reuse it buys is real (a weight block is dotted into 8 tokens
        // instead of re-read for each), but keeping the reduction order means
        // some lane must fold 32 partials serially per (row, token) while the
        // other 31 wait; that, plus the occupancy lost to the 16 accumulators,
        // cancels the gain.  Kept in tree, off by default, because the K-reuse
        // is the right shape for the next attempt (see the TODO in mmq.cu).
        cached = 0;
        const char *raw = getenv("QWEN35_CUDA_MMQ");
        if (raw != NULL) {
            const char *v = raw;
            while (*v == ' ' || *v == '\t') ++v;
            if (!(v[0] == '0' && (v[1] == '\0' || v[1] == ' ' || v[1] == '\t' || v[1] == '\r' || v[1] == '\n'))) {
                if (v[0] == '1' && (v[1] == '\0' || v[1] == ' ' || v[1] == '\t' || v[1] == '\r' || v[1] == '\n')) {
                    cached = 1;
                } else {
                    // Not 0/1: keep the default and say so, exactly like
                    // $QWEN35_CUDA_ROPE does for a bad value in backend_cuda.zig.
                    fprintf(stderr, "cuda kernels: QWEN35_CUDA_MMQ='%s' is neither '0' nor '1'; using 0\n", v);
                }
            }
        }
        if (cached == 0) fprintf(stderr, "cuda kernels: prefill matvec = thread-per-row ($QWEN35_CUDA_MMQ=1 for the tile kernel)\n");
        else fprintf(stderr, "cuda kernels: prefill matvec = mmq tile ($QWEN35_CUDA_MMQ)\n");
    }
    return cached == 1;
}

// Defined in mmq.cu.  Return value contract (different from cudaError_t, which
// is >= 0): cudaSuccess (0) == handled, every output element written; -1 ==
// "not my shape, use the fallback"; anything else == a real launch failure.
// Declared weak so a build without mmq.cu still links; nvcc rejects the GNU
// attribute on a bare declaration, so it goes through a pointer.
extern "C" int q35_launch_mul_mat_vec_tile(const void *d_w, const void *d_xq,
                                           float *d_out, int rows, int cols,
                                           int tokens, int ttype, void *stream)
    __attribute__((weak));

// ---------------------------------------------------------------------------
// Warp-per-row block shape: ONE decision, picked once, from the GPU.
//
// The decode (tokens == 1) coop kernel is launched with WARPS warps per block
// (one row per warp, block size WARPS*Q35_WARP threads).  The shared array
// `part` inside the kernel is exactly WARPS deep, so the launcher's block
// size and the kernel's `part` dimension are the SAME number -- this function
// is the single place that fixes it.
//
// The shape is a hardware fact, probed once via caps.cu (the CUDA-side mirror
// of src/kernels/x86/cpu_caps.c):
//   * sm_89:  4 warps (128 threads).  The measured-best shape on this box
//     (4090D): decode 128-thread -> 124 tok/s vs 256-thread -> 77 tok/s.
//   * any other compute capability: the occupancy formula -- the warp count
//     that maximises RESIDENT warps per SM, queried from the compiled kernel
//     via cudaOccupancyMaxActiveBlocksPerMultiprocessor.  The decode matvec is
//     bandwidth-bound, so resident-warps-per-SM is the right proxy; the
//     occupancy API accounts for register / shared / thread limits the way
//     the hardware actually does.
//
// $QWEN35_CUDA_COOP_WARPS=<1..8> overrides the shape (A/B only, like
// $QWEN35_CUDA_MMQ).  An empty or out-of-range value is ignored with a note.
// A failed probe (no usable device) keeps the sm_89-measured 4, which is the
// value every prior build used, so an unprobed device never regresses.
// ---------------------------------------------------------------------------
// Occupancy formula for a non-sm_89 device: the warp count that maximises
// RESIDENT warps per SM.  A kernel function pointer may not be stored in an
// object-pointer variable (C++ [conv.ptr] defines the pointer-to-function ->
// pointer-to-void conversion only at a function call), so each instantiation
// is passed as a first-class function pointer (the `auto` parameter) and the
// `void*` conversion happens right at the occupancy call.
static int q35_coop_occ_best_warps(void) {
    int best_warps = 0;
    int best_resident = -1;
    auto occ = [&](auto kern, int warps) {
        int active = 0;
        if (cudaOccupancyMaxActiveBlocksPerMultiprocessor(
                &active, (const void *)kern, warps * Q35_WARP, 0) == cudaSuccess
            && active * warps > best_resident) {
            best_resident = active * warps;
            best_warps = warps;
        }
    };
    occ(q35_k_mul_mat_vec_q_coop_t<1>, 1);
    occ(q35_k_mul_mat_vec_q_coop_t<2>, 2);
    occ(q35_k_mul_mat_vec_q_coop_t<3>, 3);
    occ(q35_k_mul_mat_vec_q_coop_t<4>, 4);
    occ(q35_k_mul_mat_vec_q_coop_t<5>, 5);
    occ(q35_k_mul_mat_vec_q_coop_t<6>, 6);
    occ(q35_k_mul_mat_vec_q_coop_t<7>, 7);
    occ(q35_k_mul_mat_vec_q_coop_t<8>, 8);
    return best_warps;
}

static int q35_coop_warps(void) {
    static int cached = -1;
    if (cached >= 0) return cached;
    int warps = 0;
    const char *raw = getenv("QWEN35_CUDA_COOP_WARPS");
    if (raw != NULL && *raw) {
        int v = atoi(raw);
        if (v >= 1 && v <= 8) {
            warps = v;
        } else {
            fprintf(stderr,
                    "cuda kernels: QWEN35_CUDA_COOP_WARPS='%s' is not in 1..8; ignoring\n",
                    raw);
        }
    }
    if (warps == 0) {
        const int cc = q35_gpu_cc();
        if (cc == 809) {
            // sm_89 measured best: 4 warps (128 threads).  See the header
            // comment above for the 124 vs 77 tok/s decode measurement.
            warps = 4;
        } else if (cc > 0) {
            const int best = q35_coop_occ_best_warps();
            warps = (best >= 1 && best <= 8) ? best : 4;
        } else {
            // No usable device (probe failed): keep the sm_89-measured value.
            warps = 4;
        }
    }
    cached = warps;
    fprintf(stderr,
            "cuda kernels: decode matvec block = %d warps/block (%d threads); "
            "device cc=%d \"%s\" %d SMs ($QWEN35_CUDA_COOP_WARPS overrides)\n",
            cached, cached * Q35_WARP, q35_gpu_cc(), q35_gpu_name(), q35_gpu_sm_count());
    return cached;
}

extern "C" int q35_launch_mul_mat_vec(const void *d_w_v, const void *d_xq_v,
                                      float *d_out, int rows, int cols,
                                      int tokens, int ttype, void *stream_v) {
    const unsigned char *d_w = (const unsigned char *)d_w_v;
    const unsigned char *d_xq = (const unsigned char *)d_xq_v;
    cudaStream_t stream = (cudaStream_t)stream_v;
    if (rows <= 0 || cols <= 0 || tokens <= 0) return (int)cudaSuccess;
    // Host-side validation: a wrong ttype or a cols that is not a multiple of
    // the type's block size would silently read out of bounds in the kernel.
    const Q35QuantInfo info = q35_quant_info(ttype);
    if (info.block_elems == 0) return (int)cudaErrorInvalidValue;
    if (cols % info.block_elems != 0) return (int)cudaErrorInvalidValue;

    // Prefill only: a single token has nothing to reuse, and the warp-per-row
    // kernel already coalesces it (below).
    if (tokens > 1 && q35_mmq_enabled() && q35_launch_mul_mat_vec_tile != 0) {
        const int rc = q35_launch_mul_mat_vec_tile(d_w, d_xq, d_out, rows, cols,
                                                   tokens, ttype, stream_v);
        if (rc == (int)cudaSuccess) return rc;   // the tile kernel owns the output
        if (rc != -1) {
            // A real launch failure: nothing was written, so report it and run
            // the old kernel below, which is correct for every shape.
            fprintf(stderr,
                    "cuda kernels: mmq tile launch failed (%d: %s) for rows=%d cols=%d "
                    "tokens=%d ttype=%d; falling back to the thread-per-row kernel\n",
                    rc, cudaGetErrorString((cudaError_t)rc), rows, cols, tokens, ttype);
            (void)cudaGetLastError();   // clear so the fallback's check is clean
        }
        // rc == -1: shape the tile was not sized for -> silent, intended fallback.
    }
    const int total = rows * tokens;
    const int nblk = cols / info.block_elems;
    // The warp-per-row mapping wins only for single-token decoding: with a
    // prefill chunk the tile is reused across `tokens`, and the per-token
    // `__syncwarp()` plus lane-0 fold cost more than the coalescing saves
    // (measured: 27B IQ3_XXS prefill 11.3 -> 10.1 t/s).  Keep the thread-per-row
    // kernel for tokens > 1, where each token already fills the SMs.
    if (tokens == 1 && nblk <= Q35_WARP) {
        // One warp per row: coalesced because the warp's lanes read consecutive
        // blocks of a single row.  See q35_k_mul_mat_vec_q_coop for why this is
        // numerically identical to the thread-per-row mapping.
        const int warps = q35_coop_warps();
        const int blocks = (rows + warps - 1) / warps;
        switch (warps) {
        case 1:  q35_k_mul_mat_vec_q_coop_t<1><<<blocks, 1 * Q35_WARP, 0, stream>>>(d_w, d_xq, d_out, rows, cols, tokens, ttype); break;
        case 2:  q35_k_mul_mat_vec_q_coop_t<2><<<blocks, 2 * Q35_WARP, 0, stream>>>(d_w, d_xq, d_out, rows, cols, tokens, ttype); break;
        case 3:  q35_k_mul_mat_vec_q_coop_t<3><<<blocks, 3 * Q35_WARP, 0, stream>>>(d_w, d_xq, d_out, rows, cols, tokens, ttype); break;
        case 4:  q35_k_mul_mat_vec_q_coop_t<4><<<blocks, 4 * Q35_WARP, 0, stream>>>(d_w, d_xq, d_out, rows, cols, tokens, ttype); break;
        case 5:  q35_k_mul_mat_vec_q_coop_t<5><<<blocks, 5 * Q35_WARP, 0, stream>>>(d_w, d_xq, d_out, rows, cols, tokens, ttype); break;
        case 6:  q35_k_mul_mat_vec_q_coop_t<6><<<blocks, 6 * Q35_WARP, 0, stream>>>(d_w, d_xq, d_out, rows, cols, tokens, ttype); break;
        case 7:  q35_k_mul_mat_vec_q_coop_t<7><<<blocks, 7 * Q35_WARP, 0, stream>>>(d_w, d_xq, d_out, rows, cols, tokens, ttype); break;
        default: q35_k_mul_mat_vec_q_coop_t<8><<<blocks, 8 * Q35_WARP, 0, stream>>>(d_w, d_xq, d_out, rows, cols, tokens, ttype); break;
        }
        return (int)cudaGetLastError();
    }
    const int threads = 128;
    const int blocks = (total + threads - 1) / threads;
    q35_k_mul_mat_vec_q<<<blocks, threads, 0, stream>>>(d_w, d_xq, d_out, rows, cols,
                                                        tokens, ttype);
    return (int)cudaGetLastError();
}

extern "C" int q35_launch_mul_mat_vec_q8_0(const void *d_w, const void *d_xq,
                                           float *d_out, int rows, int cols,
                                           int tokens, void *stream) {
    // Backwards-compatible entry point: identical to
    // q35_launch_mul_mat_vec(..., Q35_TT_Q8_0, ...).
    return q35_launch_mul_mat_vec(d_w, d_xq, d_out, rows, cols, tokens,
                                  Q35_TT_Q8_0, stream);
}

extern "C" int q35_launch_mul_mat_vec_f32(const float *d_w, const float *d_x,
                                          float *d_out, int rows, int cols,
                                          int tokens, void *stream_v) {
    cudaStream_t stream = (cudaStream_t)stream_v;
    if (rows <= 0 || cols <= 0 || tokens <= 0) return (int)cudaSuccess;
    const int threads = 256;
    const int total = rows * tokens;
    const int blocks = (total + threads - 1) / threads;
    // Scalar order on purpose: the warp-per-row fold (~1 ulp different) changed
    // the BF16 token stream on the 0.8B probe for only +12% throughput, which is
    // not a trade worth making while the quantized path is the hot one.
    q35_k_mul_mat_vec_f32<<<blocks, threads, 0, stream>>>(d_w, d_x, d_out, rows, cols,
                                                          tokens);
    return (int)cudaGetLastError();
}

extern "C" int q35_launch_quantize_q8_K(const float *d_x, void *d_out_v,
                                        int nblocks, void *stream_v) {
    unsigned char *d_out = (unsigned char *)d_out_v;
    cudaStream_t stream = (cudaStream_t)stream_v;
    if (nblocks <= 0) return (int)cudaSuccess;
    const int threads = 128;
    const int blocks = (nblocks + threads - 1) / threads;
    q35_k_quantize_q8_K<<<blocks, threads, 0, stream>>>(d_x, d_out, nblocks);
    return (int)cudaGetLastError();
}

// Which activation layout a type needs, for the Zig caller's buffer sizing.
// Returns 256 if the type uses Q8_K activations, 32 if it uses Q8_0 ones, and
// 0 for an unknown type.
extern "C" int q35_quant_act_block_elems(int ttype) {
    const Q35QuantInfo info = q35_quant_info(ttype);
    if (info.block_elems == 0) return 0;
    return info.wants_q8k ? Q35_QK8K : 32;
}
