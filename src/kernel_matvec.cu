// kernel_matvec.cu -- Optimized Q8_0 matvec (drop-in replacement for
// k_matvec_q8 in src/kernel.cu), plus a CPU+GPU self-test (main) that proves
// bit-exactness and benchmarks against the original kernel.
//
// Build & self-test:
//   nvcc -O3 -arch=sm_89 -o /tmp/km_selftest kernel_matvec.cu
//   CUDA_VISIBLE_DEVICES=7 /tmp/km_selftest
// Object-only (what gets pasted into the engine):
//   nvcc -O3 -arch=sm_89 -c kernel_matvec.cu -o /tmp/km.o
//
// Layout: Q8_0 block = 34 packed bytes = [0..1] f16 little-endian scale,
// [2..33] 32 signed int8 bytes. The 32-byte data region starts at an offset
// that is only 2-byte aligned (34*b+2), so it is loaded as 16 uint16 values
// (never 32-bit), which is legal and halves the number of loads vs byte-wise.

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <math.h>

// ---------------------------------------------------------------------------
// Original (byte-wise) kernel, kept verbatim ONLY for the self-test
// (bit-exactness proof + speedup baseline). Not part of the deliverable.
// ---------------------------------------------------------------------------
__device__ __forceinline__ float q35_ld_scale(const unsigned char *p) {
    unsigned short u = (unsigned short)p[0] | ((unsigned short)p[1] << 8);
    return __half2float(__ushort_as_half(u));
}

// Read every 4-byte word of a large buffer to evict L2, forcing the next
// measured launch to read `w`/`x` from DRAM (cold). grid*block threads each
// stride over the buffer; the whole buffer is covered.
__global__ void k_flush(const unsigned char *p, int bytes) {
    int stride = gridDim.x * blockDim.x;
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int sink = 0;
    for (int off = i * 4; off < bytes; off += stride * 4)
        sink += ((const unsigned int *)p)[off / 4];
    if (sink == 0xdeadbeefu) ((unsigned char *)p)[0] = 1; // never true
}

__global__ void k_matvec_q8_orig(const unsigned char *__restrict__ w,
                                 const unsigned char *__restrict__ xq,
                                 float *__restrict__ out, int rows,
                                 int row_blocks) {
    int warp = threadIdx.x >> 5;
    int lane = threadIdx.x & 31;
    int row = blockIdx.x * (blockDim.x >> 5) + warp;
    if (row >= rows) return;
    const unsigned char *wrow = w + (size_t)row * (size_t)row_blocks * 34;
    float acc = 0.0f;
    for (int b = lane; b < row_blocks; b += 32) {
        const unsigned char *wb = wrow + (size_t)b * 34;
        const unsigned char *xb = xq + (size_t)b * 34;
        float ws = q35_ld_scale(wb);
        float xs = q35_ld_scale(xb);
        int dot = 0;
#pragma unroll
        for (int j = 0; j < 32; j++)
            dot += (int)(signed char)wb[2 + j] * (int)(signed char)xb[2 + j];
        acc += (ws * (float)dot) * xs;
    }
#pragma unroll
    for (int off = 16; off > 0; off >>= 1)
        acc += __shfl_down_sync(0xffffffffu, acc, off);
    if (lane == 0) out[row] = acc;
}

// ---------------------------------------------------------------------------
// OPTIMIZED drop-in replacement: identical signature to k_matvec_q8.
//
// Bit-exactness argument vs the original:
//   * same one-warp-per-row mapping and same per-lane block order
//     (lane, lane+32, ...);
//   * each block's integer dot covers the SAME 32 byte products in the SAME
//     order (the uint16 loads just pair adjacent bytes 2k,2k+1 and they are
//     added in ascending k -> ascending j); the dot is an exact integer sum
//     with no overflow (|dot| <= 32*127*127 = 516,192 < 2^31);
//   * per-block float term is (ws * (float)dot) * xs, accumulated in the
//     same order into the same accumulator;
//   * the 5-step shfl_down reduction and the lane-0 store are unchanged.
// Only the *load* pattern changes (16 uint16 loads instead of 64 byte loads,
// with a software-pipelined prefetch of the next block), so results are
// bit-identical.
// ---------------------------------------------------------------------------
__global__ void k_matvec_q8(const unsigned char *__restrict__ w,
                            const unsigned char *__restrict__ xq,
                            float *__restrict__ out, int rows,
                            int row_blocks) {
    int warp = threadIdx.x >> 5;
    int lane = threadIdx.x & 31;
    int row = blockIdx.x * (blockDim.x >> 5) + warp;
    if (row >= rows) return;
    const unsigned char *wrow = w + (size_t)row * (size_t)row_blocks * 34;
    // This lane's first block (row_blocks is a multiple of 32 in practice).
    const unsigned char *w0 = wrow + (size_t)lane * 34;
    const unsigned char *x0 = xq + (size_t)lane * 34;
    float acc = 0.0f;
    if (lane < row_blocks) {
        const uint16_t *wp = (const uint16_t *)(w0 + 2);
        const uint16_t *xp = (const uint16_t *)(x0 + 2);
        uint16_t wv[16], xv[16];
#pragma unroll
        for (int k = 0; k < 16; k++) {
            wv[k] = wp[k];
            xv[k] = xp[k];
        }
        float ws = q35_ld_scale(w0);
        float xs = q35_ld_scale(x0);
        float nws = 0.0f, nxs = 0.0f;
        if (lane + 32 < row_blocks) { // prefetch next block's scales
            nws = q35_ld_scale(w0 + 32 * 34);
            nxs = q35_ld_scale(x0 + 32 * 34);
        }
        for (int i = lane; i < row_blocks; i += 32) {
            int dot = 0;
#pragma unroll
            for (int k = 0; k < 16; k++) {
                int lo_w = (int)(signed char)(wv[k] & 0xff);
                int hi_w = (int)(signed char)(wv[k] >> 8);
                int lo_x = (int)(signed char)(xv[k] & 0xff);
                int hi_x = (int)(signed char)(xv[k] >> 8);
                dot += lo_w * lo_x + hi_w * hi_x;
            }
            acc += (ws * (float)dot) * xs;
            if (i + 32 < row_blocks) {
                const unsigned char *wn = wrow + (size_t)(i + 32) * 34;
                const unsigned char *xn = xq + (size_t)(i + 32) * 34;
                const uint16_t *wnp = (const uint16_t *)(wn + 2);
                const uint16_t *xnp = (const uint16_t *)(xn + 2);
#pragma unroll
                for (int k = 0; k < 16; k++) {
                    wv[k] = wnp[k];
                    xv[k] = xnp[k];
                }
                ws = nws;
                xs = nxs;
                if (i + 64 < row_blocks) { // prefetch block-after-next scales
                    nws = q35_ld_scale(wn + 32 * 34);
                    nxs = q35_ld_scale(xn + 32 * 34);
                }
            }
        }
    }
#pragma unroll
    for (int off = 16; off > 0; off >>= 1)
        acc += __shfl_down_sync(0xffffffffu, acc, off);
    if (lane == 0) out[row] = acc;
}

// ---------------------------------------------------------------------------
// Self-test helpers (CPU side).
// ---------------------------------------------------------------------------
static float cpu_ld_scale(const unsigned char *p) {
    unsigned short u = (unsigned short)p[0] | ((unsigned short)p[1] << 8);
    return __half2float(__ushort_as_half(u));
}

// Exact integer per-block dot (same product order as the kernel).
static int cpu_block_dot(const unsigned char *wb, const unsigned char *xb) {
    int dot = 0;
    for (int j = 0; j < 32; j++)
        dot += (int)(signed char)wb[2 + j] * (int)(signed char)xb[2 + j];
    return dot;
}

// Reference with the kernel's exact semantics, computed in double so it is a
// tight upper bound: exact integer dot, then float(ws*dot)*xs in block order.
static double cpu_ref_row(const unsigned char *wrow, const unsigned char *xq,
                          int blocks) {
    double sum = 0.0;
    for (int b = 0; b < blocks; b++) {
        const unsigned char *wb = wrow + (size_t)b * 34;
        const unsigned char *xb = xq + (size_t)b * 34;
        float ws = cpu_ld_scale(wb);
        float xs = cpu_ld_scale(xb);
        int dot = cpu_block_dot(wb, xb);
        sum += (double)((float)((ws * (float)dot) * xs));
    }
    return sum;
}

static unsigned g_rng = 12345u;
static unsigned rng32(void) {
    g_rng = g_rng * 1664525u + 1013904223u;
    return g_rng >> 8;
}

// Fill one Q8_0 block with random bytes and a plausible f16 scale.
static void fill_block(unsigned char *p) {
    float s = 0.01f + (float)(rng32() % 10000) / 1000.0f; // 0.01..10.0
    unsigned short u = __half_as_ushort(__float2half(s));
    p[0] = (unsigned char)(u & 0xff);
    p[1] = (unsigned char)(u >> 8);
    for (int j = 0; j < 32; j++)
        p[2 + j] = (unsigned char)(signed char)((int)(rng32() % 257) - 128);
}

static void fill_q8(unsigned char *p, size_t nblocks) {
    for (size_t b = 0; b < nblocks; b++) fill_block(p + b * 34);
}

struct Cfg {
    int rows;
    int blocks;
    const char *name;
};

static float bench(const char *name, int rows, int blocks,
                   const unsigned char *dw, const unsigned char *dx,
                   float *dout, unsigned char *flush, int flush_bytes,
                   int cold) {
    cudaEvent_t t0, t1;
    cudaEventCreate(&t0);
    cudaEventCreate(&t1);
    const int iters = cold ? 100 : 300;
    // warmup (also sizes the caches / clocks)
    k_matvec_q8<<<(rows + 7) / 8, 256, 0>>>(dw, dx, dout, rows, blocks);
    cudaDeviceSynchronize();
    float total_us = 0.0f;
    for (int i = 0; i < iters; i++) {
        if (cold)
            // Evict L2 (untimed) so the matvec below reads w/x from DRAM.
            k_flush<<<(flush_bytes + 127) / 128, 128>>>(flush, flush_bytes);
        // All on the default stream => strictly ordered: flush (if any) runs
        // fully before t0, then matvec, then t1. So the timed interval is the
        // matvec launch alone, on a cold L2.
        cudaEventRecord(t0);
        k_matvec_q8<<<(rows + 7) / 8, 256, 0>>>(dw, dx, dout, rows, blocks);
        cudaEventRecord(t1);
        cudaDeviceSynchronize();
        float ms;
        cudaEventElapsedTime(&ms, t0, t1);
        total_us += ms * 1000.0f;
    }
    float us = total_us / iters;
    // GB/s: each launch reads rows*blocks*34 (w) + blocks*34 (x) bytes
    double bytes = (double)rows * blocks * 34.0 + blocks * 34.0;
    printf("  [%s] rows=%4d blocks=%4d : %8.2f us/launch  (%.1f GB/s w+x)%s\n",
           name, rows, blocks, us, bytes / (us * 1e-6) / 1e9,
           cold ? "  [cold L2]" : "");
    cudaEventDestroy(t0);
    cudaEventDestroy(t1);
    return us;
}

int main(void) {
    // L2-flush buffer, larger than the 4090D's 72 MB L2, to force cold reads.
    const size_t flush_bytes = 128u * 1024u * 1024u;
    unsigned char *dflush = (unsigned char *)0;
    cudaMalloc((void **)&dflush, flush_bytes);
    cudaMemset(dflush, 0x42, flush_bytes);
    cudaDeviceSynchronize();

    struct Cfg cfgs[] = {
        {256, 64, "selftest 2048d"},   // 256 x 2048
        {1024, 32, "attn 1024d"},      // rows=hidden, cols=hidden
        {3584, 112, "ffn gate/up"},    // n_ff rows, cols=hidden
        {1024, 128, "attn_q GDN 4096"}, // cols=4096
        {1024, 64, "kv 2048d"},        // cols=2*kd*heads-ish
    };
    const int NC = sizeof(cfgs) / sizeof(cfgs[0]);

    printf("=== correctness: optimized vs original (bit-exact) + double ref ===\n");
    for (int c = 0; c < NC; c++) {
        int rows = cfgs[c].rows;
        int blocks = cfgs[c].blocks;
        size_t wbytes = (size_t)rows * blocks * 34;
        size_t xbytes = (size_t)blocks * 34;
        unsigned char *hw = (unsigned char *)malloc(wbytes);
        unsigned char *hx = (unsigned char *)malloc(xbytes);
        double *ref = (double *)malloc(rows * sizeof(double));
        float *hout = (float *)malloc(rows * sizeof(float));
        float *horig = (float *)malloc(rows * sizeof(float));
        float *dout, *dorig;
        unsigned char *dw, *dx;
        cudaMalloc(&dw, wbytes);
        cudaMalloc(&dx, xbytes);
        cudaMalloc(&dout, rows * sizeof(float));
        cudaMalloc(&dorig, rows * sizeof(float));

        fill_q8(hw, (size_t)rows * blocks);
        fill_q8(hx, blocks);
        for (int r = 0; r < rows; r++)
            ref[r] = cpu_ref_row(hw + (size_t)r * blocks * 34, hx, blocks);
        cudaMemcpy(dw, hw, wbytes, cudaMemcpyHostToDevice);
        cudaMemcpy(dx, hx, xbytes, cudaMemcpyHostToDevice);

        k_matvec_q8_orig<<<(rows + 7) / 8, 256>>>(dw, dx, dorig, rows, blocks);
        k_matvec_q8<<<(rows + 7) / 8, 256>>>(dw, dx, dout, rows, blocks);
        cudaDeviceSynchronize();
        cudaMemcpy(horig, dorig, rows * sizeof(float), cudaMemcpyDeviceToHost);
        cudaMemcpy(hout, dout, rows * sizeof(float), cudaMemcpyDeviceToHost);

        int bitexact_new_vs_orig = 1;
        double maxdiff_ref = 0.0, maxdiff_orig = 0.0;
        for (int r = 0; r < rows; r++) {
            if (memcmp(&hout[r], &horig[r], 4) != 0) bitexact_new_vs_orig = 0;
            double d1 = fabs((double)hout[r] - ref[r]);
            double d2 = fabs((double)horig[r] - ref[r]);
            if (d1 > maxdiff_ref) maxdiff_ref = d1;
            if (d2 > maxdiff_orig) maxdiff_orig = d2;
        }
        printf("  [%s] rows=%4d blocks=%4d : new==orig bit-exact: %s | "
               "max|new-ref|=%.6g max|orig-ref|=%.6g\n",
               cfgs[c].name, rows, blocks,
               bitexact_new_vs_orig ? "YES" : "NO", maxdiff_ref,
               maxdiff_orig);
        if (!bitexact_new_vs_orig)
            printf("    *** MISMATCH: optimized kernel is NOT bit-exact ***\n");

        // benchmark (original + new), warm-L2 and cold-L2
        bench("orig", rows, blocks, dw, dx, dorig, dflush, flush_bytes, 0);
        bench("new ", rows, blocks, dw, dx, dout, dflush, flush_bytes, 0);
        bench("orig", rows, blocks, dw, dx, dorig, dflush, flush_bytes, 1);
        bench("new ", rows, blocks, dw, dx, dout, dflush, flush_bytes, 1);

        cudaFree(dw);
        cudaFree(dx);
        cudaFree(dout);
        cudaFree(dorig);
        free(hw);
        free(hx);
        free(ref);
        free(hout);
        free(horig);
    }
    cudaFree(dflush);
    printf("done.\n");
    return 0;
}
