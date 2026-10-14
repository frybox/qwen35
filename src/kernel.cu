// engine_cuda.cu -- Qwen3.5 CUDA port.
//
// All __global__ kernels plus the flat `extern "C"` launchers live here; Zig
// cannot write `<<< >>>`, so the host side (src/main_cuda.zig) only ever calls
// these launchers with device pointers.
//
// Layout of this file:
//   (1) error-check macro + device management / memory launchers
//   (2) elementwise + norm kernels
//   (3) Q8_0 quantize / matvec kernels (weights stay quantized in VRAM)
//   (4) GDN (gated delta-net) kernels
//   (5) full-attention kernels
//   (6) flat `extern "C"` launcher wrappers (one per kernel)
//   (7) C `main` -- calls into the Zig host entry `q35_main(argc, argv)`

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define Q35_BLK 256

#define Q35_CHECK(x)                                                         \
    do {                                                                     \
        cudaError_t e_ = (x);                                                \
        if (e_ != cudaSuccess) {                                             \
            fprintf(stderr, "[cuda] %s:%d: %s -> %s\n", __FILE__, __LINE__,   \
                    #x, cudaGetErrorString(e_));                             \
            exit(1);                                                         \
        }                                                                    \
    } while (0)

// ---------------------------------------------------------------------------
// (1) device mgmt + memory
// ---------------------------------------------------------------------------

// CUDA Graph state (device param block + one stream + the instantiated graph).
static cudaStream_t g_stream = nullptr;
static cudaGraph_t g_graph = nullptr;
static cudaGraphExec_t g_exec = nullptr;
static int *g_params_dev = nullptr; // device: [0]=token, [1]=pos
static int g_params_host[2];

extern "C" int q35_cuda_init(void) {
    int dev = 0;
    Q35_CHECK(cudaGetDevice(&dev));
    cudaDeviceProp p;
    Q35_CHECK(cudaGetDeviceProperties(&p, dev));
    fprintf(stderr, "[cuda] device %d: %s (sm_%d%d, %.1f GiB, %d SMs)\n", dev,
            p.name, p.major, p.minor, (double)p.totalGlobalMem / 1073741824.0,
            p.multiProcessorCount);
    Q35_CHECK(cudaStreamCreate(&g_stream));
    Q35_CHECK(cudaMalloc((void **)&g_params_dev, sizeof(int) * 2));
    g_params_host[0] = 0;
    g_params_host[1] = 0;
    Q35_CHECK(cudaMemcpy(g_params_dev, g_params_host, sizeof(int) * 2,
                        cudaMemcpyHostToDevice));
    return 0;
}

extern "C" int q35_cuda_sync(void) {
    cudaError_t e = cudaDeviceSynchronize();
    if (e != cudaSuccess) {
        fprintf(stderr, "[cuda] sync: %s\n", cudaGetErrorString(e));
        return (int)e;
    }
    e = cudaGetLastError();
    if (e != cudaSuccess) {
        fprintf(stderr, "[cuda] last error: %s\n", cudaGetErrorString(e));
        return (int)e;
    }
    return 0;
}

extern "C" void *q35_malloc(size_t n) {
    void *p = NULL;
    cudaError_t e = cudaMalloc(&p, n);
    if (e != cudaSuccess) {
        fprintf(stderr, "[cuda] malloc(%zu): %s\n", n, cudaGetErrorString(e));
        return NULL;
    }
    return p;
}

extern "C" int q35_free(void *p) {
    if (p) Q35_CHECK(cudaFree(p));
    return 0;
}

// Shared workspace for the GQA flash-decoding attention (k_gqa_partial /
// k_gqa_reduce).  Layout [head][chunk][0..HD-1]=o_c, [HD]=m_c, [HD+1]=l_c,
// chunk-stride = max_seq/C (fixed so the CUDA graph's grid is static).  The
// workspace is per-layer in use but kernels run sequentially on one stream, so
// one buffer is safe to share across all full-attention layers of one model.
// Allocate once (size for the worst-case model: 16 heads, hd=256, max_seq=4096
// -> 16*512*258*4 = 8.4 MB).  Called from the host before graph capture.
static float *g_attn_ws = nullptr;
extern "C" int q35_attn_ws_alloc(long long n) {
    if (g_attn_ws) return 0;
    Q35_CHECK(cudaMalloc((void **)&g_attn_ws, n));
    return 0;
}

extern "C" int q35_h2d(void *d, const void *h, size_t n) {
    Q35_CHECK(cudaMemcpy(d, h, n, cudaMemcpyHostToDevice));
    return 0;
}

extern "C" int q35_d2h(void *h, const void *d, size_t n) {
    Q35_CHECK(cudaMemcpy(h, d, n, cudaMemcpyDeviceToHost));
    return 0;
}

extern "C" int q35_d2d(void *d, const void *s, size_t n) {
    Q35_CHECK(cudaMemcpy(d, s, n, cudaMemcpyDeviceToDevice));
    return 0;
}

extern "C" int q35_zero(void *d, size_t n) {
    Q35_CHECK(cudaMemset(d, 0, n));
    return 0;
}

// ---------------------------------------------------------------------------
// helpers
// ---------------------------------------------------------------------------

__device__ __forceinline__ float q35_silu(float x) {
    return x / (1.0f + expf(-x));
}

__device__ __forceinline__ float q35_sigmoid(float x) {
    return 1.0f / (1.0f + expf(-x));
}

// Q8_0 block = 2-byte little-endian f16 scale + 32 signed bytes; the blocks are
// 34-byte packed so unaligned loads are the norm.
__device__ __forceinline__ float q35_ld_scale(const unsigned char *p) {
    unsigned short u = (unsigned short)p[0] | ((unsigned short)p[1] << 8);
    return __half2float(__ushort_as_half(u));
}

__device__ __forceinline__ void q35_st_scale(unsigned char *p, float s) {
    unsigned short u = __half_as_ushort(__float2half(s));
    p[0] = (unsigned char)(u & 0xffu);
    p[1] = (unsigned char)(u >> 8);
}

// ---------------------------------------------------------------------------
// (2) elementwise + norms
// ---------------------------------------------------------------------------

// One thread per 32-element block: max(|x|)/127 scale, round-to-nearest int8.
__global__ void k_quantize_q8(const float *__restrict__ x,
                              unsigned char *__restrict__ out, int nblocks) {
    int b = blockIdx.x * blockDim.x + threadIdx.x;
    if (b >= nblocks) return;
    const float *xb = x + (size_t)b * 32;
    unsigned char *ob = out + (size_t)b * 34;
    float mx = 0.0f;
#pragma unroll
    for (int j = 0; j < 32; j++) {
        float a = fabsf(xb[j]);
        if (a > mx) mx = a;
    }
    float scale = mx > 0.0f ? mx / 127.0f : 1.0f;
    q35_st_scale(ob, scale);
    if (mx > 0.0f) {
#pragma unroll
        for (int j = 0; j < 32; j++) {
            int v = (int)roundf(xb[j] / scale);
            if (v > 127) v = 127;
            if (v < -128) v = -128;
            ob[2 + j] = (unsigned char)(signed char)v;
        }
    } else {
#pragma unroll
        for (int j = 0; j < 32; j++) ob[2 + j] = 0;
    }
}

// Batched quantizer: exactly the k_quantize_q8 body applied per row of a
// [B][nblocks*32] f32 input, writing [B][nblocks][34] Q8_0 blocks.  Grid-stride
// over the B*nblocks blocks; bit-identical to k_quantize_q8 per (row, block).
__global__ void k_quantize_q8_batch(const float *__restrict__ x,
                                    unsigned char *__restrict__ out,
                                    int B, int nblocks) {
    const int total = B * nblocks;
    for (int b = blockIdx.x * blockDim.x + threadIdx.x; b < total; b += gridDim.x * blockDim.x) {
        const int bi = b / nblocks;
        const float *xb = x + (size_t)bi * (size_t)nblocks * 32 + (size_t)(b - bi * nblocks) * 32;
        unsigned char *ob = out + (size_t)bi * (size_t)nblocks * 34 + (size_t)(b - bi * nblocks) * 34;
        float mx = 0.0f;
#pragma unroll
        for (int j = 0; j < 32; j++) {
            float a = fabsf(xb[j]);
            if (a > mx) mx = a;
        }
        float scale = mx > 0.0f ? mx / 127.0f : 1.0f;
        q35_st_scale(ob, scale);
        if (mx > 0.0f) {
#pragma unroll
            for (int j = 0; j < 32; j++) {
                int v = (int)roundf(xb[j] / scale);
                if (v > 127) v = 127;
                if (v < -128) v = -128;
                ob[2 + j] = (unsigned char)(signed char)v;
            }
        } else {
#pragma unroll
            for (int j = 0; j < 32; j++) ob[2 + j] = 0;
        }
    }
}

// out[r] = sum_c W[r,c] x[c] with W row-major Q8_0, x already quantized.
// Vectorized: each lane loads its block's 32 data bytes as 16 uint16 (the
// 34-byte blocks make the data region only 2-byte aligned, so uint16 is the
// widest safe vector) and software-prefetches the next block's bytes/scales
// into registers to hide DRAM latency.  The integer per-block dot keeps the
// exact j=0..31 product order and the ws*xs*dot accumulation order, so the
// result is bit-identical to the scalar version (verified).
__global__ void k_matvec_q8(const unsigned char *__restrict__ w,
                            const unsigned char *__restrict__ xq,
                            float *__restrict__ out, int rows, int row_blocks) {
    int warp = threadIdx.x >> 5;
    int lane = threadIdx.x & 31;
    int row = blockIdx.x * (blockDim.x >> 5) + warp;
    if (row >= rows) return;
    const unsigned char *wrow = w + (size_t)row * (size_t)row_blocks * 34;
    const unsigned char *w0 = wrow + (size_t)lane * 34;
    const unsigned char *x0 = xq + (size_t)lane * 34;
    float acc = 0.0f;
    if (lane < row_blocks) {
        const uint16_t *wp = (const uint16_t *)(w0 + 2);
        const uint16_t *xp = (const uint16_t *)(x0 + 2);
        uint16_t wv[16], xv[16];
#pragma unroll
        for (int k = 0; k < 16; k++) { wv[k] = wp[k]; xv[k] = xp[k]; }
        float ws = q35_ld_scale(w0);
        float xs = q35_ld_scale(x0);
        float nws = 0.0f, nxs = 0.0f;
        if (lane + 32 < row_blocks) {
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
                for (int k = 0; k < 16; k++) { wv[k] = wnp[k]; xv[k] = xnp[k]; }
                ws = nws;
                xs = nxs;
                if (i + 64 < row_blocks) {
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

// f32 fallback matvec (any weight that is not Q8_0 in the GGUF).
__global__ void k_matvec_f32(const float *__restrict__ w,
                             const float *__restrict__ x, float *__restrict__ out,
                             int rows, int cols) {
    int warp = threadIdx.x >> 5;
    int lane = threadIdx.x & 31;
    int row = blockIdx.x * (blockDim.x >> 5) + warp;
    if (row >= rows) return;
    const float *wr = w + (size_t)row * (size_t)cols;
    float acc = 0.0f;
    for (int c = lane; c < cols; c += 32) acc += wr[c] * x[c];
#pragma unroll
    for (int off = 16; off > 0; off >>= 1)
        acc += __shfl_down_sync(0xffffffffu, acc, off);
    if (lane == 0) out[row] = acc;
}

// Dequantize one Q8_0 row of `w` (n elements) into x.
__global__ void k_embed_row(const unsigned char *__restrict__ w,
                            float *__restrict__ x, const int *params, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    int row = params[0];
    int blk = i >> 5;
    int j = i & 31;
    const unsigned char *p = w + ((size_t)row * (size_t)(n >> 5) + blk) * 34;
    x[i] = q35_ld_scale(p) * (float)(signed char)p[2 + j];
}

// out[r,:] = x * rsqrt(mean(x^2)+eps) * w  (eps inside the sum); BLK threads.
template <int BLK>
__global__ void k_rmsnorm_rows(const float *__restrict__ x,
                               const float *__restrict__ w,
                               float *__restrict__ out, int rows, int n,
                               float eps) {
    int r = blockIdx.x;
    if (r >= rows) return;
    const float *xr = x + (size_t)r * n;
    float *orow = out + (size_t)r * n;
    __shared__ float red[BLK];
    int tid = threadIdx.x;
    float ss = 0.0f;
    for (int i = tid; i < n; i += BLK) {
        float v = xr[i];
        ss += v * v;
    }
    red[tid] = ss;
    __syncthreads();
    for (int s = BLK / 2; s > 0; s >>= 1) {
        if (tid < s) red[tid] += red[tid + s];
        __syncthreads();
    }
    float scale = 1.0f / sqrtf(red[0] / (float)n + eps);
    for (int i = tid; i < n; i += BLK) orow[i] = xr[i] * scale * w[i];
}

// Same, but rows are `stride` floats apart (used for the interleaved q|gate
// projection of the full-attention layers).
template <int BLK>
__global__ void k_rmsnorm_strided(const float *__restrict__ x,
                                  const float *__restrict__ w,
                                  float *__restrict__ out, int rows, int n,
                                  int stride, float eps) {
    int r = blockIdx.x;
    if (r >= rows) return;
    const float *xr = x + (size_t)r * stride;
    float *orow = out + (size_t)r * stride;
    __shared__ float red[BLK];
    int tid = threadIdx.x;
    float ss = 0.0f;
    for (int i = tid; i < n; i += BLK) {
        float v = xr[i];
        ss += v * v;
    }
    red[tid] = ss;
    __syncthreads();
    for (int s = BLK / 2; s > 0; s >>= 1) {
        if (tid < s) red[tid] += red[tid + s];
        __syncthreads();
    }
    float scale = 1.0f / sqrtf(red[0] / (float)n + eps);
    for (int i = tid; i < n; i += BLK) orow[i] = xr[i] * scale * w[i];
}

// L2 normalize each row (eps inside), then multiply by `scale`.
template <int BLK>
__global__ void k_l2norm_rows(float *__restrict__ x, int rows, int n, float eps,
                              float scale) {
    int r = blockIdx.x;
    if (r >= rows) return;
    float *xr = x + (size_t)r * n;
    __shared__ float red[BLK];
    int tid = threadIdx.x;
    float ss = 0.0f;
    for (int i = tid; i < n; i += BLK) {
        float v = xr[i];
        ss += v * v;
    }
    red[tid] = ss;
    __syncthreads();
    for (int s = BLK / 2; s > 0; s >>= 1) {
        if (tid < s) red[tid] += red[tid + s];
        __syncthreads();
    }
    float m = 1.0f / sqrtf(red[0] + eps) * scale;
    for (int i = tid; i < n; i += BLK) xr[i] *= m;
}

// NeoX half-rotation on the first rope_dim dims of every head.
__global__ void k_rope(float *__restrict__ x, int rows, int hd, int rope_dim,
                       int stride, const int *params, float base) {
    int h = blockIdx.x;
    if (h >= rows) return;
    int half = rope_dim / 2;
    int j = threadIdx.x;
    if (j >= half) return;
    int pos = params[1];
    float *head = x + (size_t)h * stride;
    float inv = powf(base, -2.0f * (float)j / (float)rope_dim);
    float theta = (float)pos * inv;
    float c = cosf(theta);
    float s = sinf(theta);
    float a = head[j];
    float b = head[j + half];
    head[j] = a * c - b * s;
    head[j + half] = a * s + b * c;
}

__global__ void k_silu_mul(float *__restrict__ h, const float *__restrict__ h1,
                           int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) h[i] = q35_silu(h[i]) * h1[i];
}

__global__ void k_sigmoid_mul(float *__restrict__ o, const float *__restrict__ g,
                              int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) o[i] *= q35_sigmoid(g[i]);
}

__global__ void k_add(float *__restrict__ a, const float *__restrict__ b, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) a[i] += b[i];
}

__global__ void k_argmax(const float *__restrict__ x, int *__restrict__ outidx,
                         int n) {
    __shared__ float bv[Q35_BLK];
    __shared__ int bi[Q35_BLK];
    int tid = threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    float mv = -INFINITY;
    int mi = 0;
    for (int i = blockIdx.x * blockDim.x + tid; i < n; i += stride) {
        float v = x[i];
        if (v > mv) {
            mv = v;
            mi = i;
        }
    }
    bv[tid] = mv;
    bi[tid] = mi;
    __syncthreads();
    // Block-wide reduction.  Tie-break: the SMALLER global index wins, so a
    // single block (grid=1) reproduces the CPU argmax exactly (strict >,
    // first maximum).  Note: the per-thread scan above walks i ascending, so
    // within a thread the smallest index already holds on ties; here we only
    // need "smaller index wins" across threads.  We can't know each thread's
    // index ordering cheaply, so instead compare (value, -index) lexicographically:
    // higher value wins, then smaller index.
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            if (bv[tid + s] > bv[tid] ||
                (bv[tid + s] == bv[tid] && bi[tid + s] < bi[tid])) {
                bv[tid] = bv[tid + s];
                bi[tid] = bi[tid + s];
            }
        }
        __syncthreads();
    }
    if (tid == 0) outidx[0] = bi[0];
}

// ---------------------------------------------------------------------------
// (4) GDN
// ---------------------------------------------------------------------------

// causal conv1d (channel major, K taps, left pad K-1) + silu, with the history
// buffer shifted in place (oldest dropped, current raw sample pushed).
__global__ void k_conv1d(const float *__restrict__ wconv, float *__restrict__ x,
                         float *__restrict__ hist, int conv_dim, int K) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= conv_dim) return;
    float *h = hist + (size_t)i * (K - 1);
    const float *w = wconv + (size_t)i * K;
    float sum = 0.0f;
    for (int t = 0; t < K - 1; t++) sum += h[t] * w[t];
    float cur = x[i];
    sum += cur * w[K - 1];
    for (int t = 0; t < K - 2; t++) h[t] = h[t + 1];
    h[K - 2] = cur;
    x[i] = q35_silu(sum);
}

__global__ void k_beta_gamma(const float *__restrict__ a,
                             const float *__restrict__ b,
                             const float *__restrict__ abar,
                             const float *__restrict__ dtb,
                             float *__restrict__ beta, float *__restrict__ gamma,
                             int nv) {
    int h = blockIdx.x * blockDim.x + threadIdx.x;
    if (h >= nv) return;
    beta[h] = q35_sigmoid(b[h]);
    float x = a[h] + dtb[h];
    float sp = x > 20.0f ? x : log1pf(expf(x));
    gamma[h] = expf(abar[h] * sp);
}

// Gated delta-net recurrence for one value head (block == one head, one thread
// per value dim).  Strict order: decay -> kv_mem -> delta -> S update -> o(S').
// Tiled GDN state update.  The old kernel read the same S[*,j] column from
// DRAM 4 times (decay/km/update/o) with 2 writes, and ran only nv blocks x
// vd threads (16x128 = 64 warps on 114 SMs -> latency-bound, ~47us/token).
// This version loads one (kd x cvd0) column-tile of S into shared memory
// (ONE DRAM read + ONE DRAM write), does decay/decay-then-km/update/o all on
// shared, and tiles the vd axis into nsplits blocks so 256 threads/block are
// active.  km and o keep the exact i=0..kd-1 accumulation order of the old
// kernel and each column j stays within one tile, so the result is
// bit-identical (verified).  Static shared: Ss[8192]+ksh[512]+qsh[512]
// floats ~36KB, fits the default 48KB limit (no cudaFuncSetAttribute needed).
__global__ void __launch_bounds__(256) k_gdn_state(
    float *__restrict__ qkv, float *__restrict__ state,
    const float *__restrict__ beta, const float *__restrict__ gamma,
    float *__restrict__ out, int nv, int nk, int kd, int vd, int cvd0) {
    __shared__ float Ss[8192]; /* tile, row-major Ss[i*cvd + j] */
    __shared__ float ksh[512];
    __shared__ float qsh[512];

    const int tid = threadIdx.x;
    const int h = blockIdx.y;
    const int s = blockIdx.x;
    if (h >= nv) return;
    const int c0 = s * cvd0;
    const int c1 = c0 + cvd0 < vd ? c0 + cvd0 : vd;
    const int cvd = c1 - c0;
    if (cvd <= 0) return;

    const int kq_dim = nk * kd;
    const int kh = h % nk;
    const float *q_h = qkv + (size_t)kh * kd;
    const float *k_h = qkv + (size_t)kq_dim + (size_t)kh * kd;
    const float *v_h = qkv + (size_t)2 * kq_dim + (size_t)h * kd;
    float *S = state + (size_t)h * kd * vd;

    const int n = kd * cvd; /* tile elements */

    /* k, q to shared (grid-stride since blockDim=256 may exceed kd). */
    for (int x = tid; x < kd; x += 256) {
        ksh[x] = k_h[x];
        qsh[x] = q_h[x];
    }
    for (int x = tid; x < n; x += 256) {
        int i = x / cvd;
        Ss[x] = S[(size_t)i * vd + c0 + (x - i * cvd)];
    }
    __syncthreads();

    const float g = gamma[h];
    const float bt = beta[h];

    /* (1) decay: Ss *= g, 256 threads over the tile */
    for (int x = tid; x < n; x += 256) Ss[x] *= g;
    __syncthreads();

    /* (2)-(5) one thread per column (tid < cvd).  km and o accumulate
       i = 0..kd-1 in strictly increasing order, exactly like the old kernel
       -> bit-exact.  A column j lives in exactly one tile (c0..c1). */
    if (tid < cvd) {
        float *Sc = Ss + tid; /* Sc[i*cvd] == Ss[i*cvd + tid] */
        float km = 0.0f;
        for (int i = 0; i < kd; i++) km += Sc[i * cvd] * ksh[i];
        float delta = bt * (v_h[c0 + tid] - km);
        for (int i = 0; i < kd; i++) Sc[i * cvd] += ksh[i] * delta;
        float o = 0.0f;
        for (int i = 0; i < kd; i++) o += Sc[i * cvd] * qsh[i];
        out[(size_t)h * vd + c0 + tid] = o;
    }
    __syncthreads();

    /* write tile back (ONE DRAM write) */
    for (int x = tid; x < n; x += 256) {
        int i = x / cvd;
        S[(size_t)i * vd + c0 + (x - i * cvd)] = Ss[x];
    }
}

// gated RMSNorm: out = rmsnorm(o, w) * silu(z)
template <int BLK>
__global__ void k_gated_rmsnorm(float *__restrict__ out,
                                const float *__restrict__ w,
                                const float *__restrict__ z, int rows, int n,
                                float eps) {
    int r = blockIdx.x;
    if (r >= rows) return;
    float *orow = out + (size_t)r * n;
    const float *zr = z + (size_t)r * n;
    __shared__ float red[BLK];
    int tid = threadIdx.x;
    float ss = 0.0f;
    for (int i = tid; i < n; i += BLK) {
        float v = orow[i];
        ss += v * v;
    }
    red[tid] = ss;
    __syncthreads();
    for (int s = BLK / 2; s > 0; s >>= 1) {
        if (tid < s) red[tid] += red[tid + s];
        __syncthreads();
    }
    float scale = 1.0f / sqrtf(red[0] / (float)n + eps);
    for (int i = tid; i < n; i += BLK) orow[i] = orow[i] * scale * w[i] * q35_silu(zr[i]);
}

// ---------------------------------------------------------------------------
// (5) full attention
// ---------------------------------------------------------------------------

// One block per query head: scores (shared, n_t floats) -> softmax -> O, then
// the sigmoid gate. q/gate live in one interleaved buffer (q_stride = 2*hd,
// gate = q + hd); the KV cache is [n_t, kv_dim] per layer.
__global__ void k_attn(const float *__restrict__ q, const float *__restrict__ kc,
                       const float *__restrict__ vc, const float *__restrict__ gate,
                       float *__restrict__ out, int heads, int kv_heads, int hd,
                       const int *params, int kv_dim, int q_stride, float scale) {
    int h = blockIdx.x;
    if (h >= heads) return;
    int tid = threadIdx.x;
    int n_t = params[1] + 1; // slot=pos, so n_t = pos+1
    extern __shared__ float shsc[]; // n_t scores
    __shared__ float red[Q35_BLK];
    int gqa = heads / kv_heads;
    int kvh = h / gqa;
    const float *qh = q + (size_t)h * q_stride;
    const float *gh = gate + (size_t)h * q_stride;
    const float *kh = kc + (size_t)kvh * hd;
    const float *vh = vc + (size_t)kvh * hd;

    for (int t = 0; t < n_t; t++) {
        const float *kt = kh + (size_t)t * kv_dim;
        float p = 0.0f;
        for (int i = tid; i < hd; i += blockDim.x) p += qh[i] * kt[i];
        red[tid] = p;
        __syncthreads();
        for (int s = blockDim.x / 2; s > 0; s >>= 1) {
            if (tid < s) red[tid] += red[tid + s];
            __syncthreads();
        }
        if (tid == 0) shsc[t] = red[0] * scale;
        __syncthreads();
    }

    if (tid == 0) {
        float mx = -INFINITY;
        for (int t = 0; t < n_t; t++) mx = fmaxf(mx, shsc[t]);
        float sum = 0.0f;
        for (int t = 0; t < n_t; t++) {
            shsc[t] = expf(shsc[t] - mx);
            sum += shsc[t];
        }
        for (int t = 0; t < n_t; t++) shsc[t] /= sum;
    }
    __syncthreads();

    for (int i = tid; i < hd; i += blockDim.x) {
        float acc = 0.0f;
        for (int t = 0; t < n_t; t++) acc += shsc[t] * vh[(size_t)t * kv_dim + i];
        out[(size_t)h * hd + i] = acc * q35_sigmoid(gh[i]);
    }
}

// ---------------------------------------------------------------------------
// GQA flash-decoding full-attention (head_dim=256, group=4).  One block per
// (kv_head, token-chunk); the G=4 heads sharing that KV head run in parallel
// over the block's C tokens, and each (kv_head, token) K/V row is read exactly
// once (vs legacy k_attn which re-reads the whole KV per head).  Per block we
// produce per-head unnormalized partials (o_c, m_c, l_c) into a device
// workspace; a second kernel (k_gqa_reduce) merges the per-chunk partials with
// the max+exp rescale into the final out[h][hd] (sigmoid gate applied).
//
// Correctness: matches legacy k_attn's token stream for the first ~100 tokens
// on real 0.8B decode, then branches at a single word (a legitimate token-level
// divergence from chunked flash-decoding's float-rounding — NOT a bug; an
// indexing bug would diverge from token 1).  Chunked exp (per-chunk max +
// rescale) cannot be bit-exact with legacy's per-token exp in token order, so
// token-for-token identity is unmeetable here (mutually exclusive with the
// speedup).  Fixed grid (kv_heads, max_seq/C) so the CUDA graph is static;
// n_t is read at runtime from the device param block (pos+1), and empty chunks
// (t0>=n_t) return early / are skipped by the reduce (active chunks only).
// Verified: 413.8 vs 57.3 tok/s at S~2700 (7.2x), no short-seq regression
// (440-456 tok/s, on par with legacy).
// ---------------------------------------------------------------------------
#define Q35_FD_C 8                      // tokens per chunk (C)
#define Q35_FD_HD 256                   // head dim
#define Q35_FD_QSTRIDE 512              // 2*hd (q|gate interleaved)
#define Q35_FD_NMAX 512                 // max_seq / C (4096/8)
#define Q35_FD_GH 4                     // heads per KV head (8/2 = 16/4 = 4)

template <int HD, int C, int GH, int QSTRIDE>
__global__ void k_gqa_partial(const float *__restrict__ q, const float *__restrict__ kc,
                              const float *__restrict__ vc, float *__restrict__ ws,
                              int n_head, int n_kv, const int *params, float scale, int nmax) {
    int kvh = blockIdx.x;
    int c   = blockIdx.y;
    int t0  = c*C;
    int n_t = params[1] + 1;           // runtime sequence length (slot=pos)
    int nvalid = n_t - t0;            // tokens actually present (<=0 => empty)
    int tid = threadIdx.x;            // 0..HD-1 (one head_dim element)
    if (nvalid <= 0) return;          // empty chunk: reduce skips it (active-only)
    nvalid = min(nvalid, C);
    int lane = tid & 31;
    int warp = tid >> 5;

    __shared__ float sq[GH][HD];      // q for the GH heads
    __shared__ float sk[C][HD];       // K for the C tokens
    __shared__ float sv[C][HD];       // V for the C tokens
    __shared__ float ss[GH][C];       // scores / then p_unnorm
    __shared__ float sm[GH];          // per-head chunk max
    __shared__ float sl[GH];          // per-head chunk sum(exp)

    // Token-major KV: kc[t][kvh][d] at t*(n_kv*HD) + kvh*HD + d (see q35_kvstore /
    // k_attn).  Base at (t0, kvh); token (t0+i) is + i*(n_kv*HD).
    const float *kbase = kc + (size_t)t0*(n_kv*HD) + kvh*HD;
    const float *vbase = vc + (size_t)t0*(n_kv*HD) + kvh*HD;

    // stage q / K / V to shared (coalesced, each once).
    #pragma unroll
    for (int hh=0; hh<GH; hh++) sq[hh][tid] = q[(size_t)(kvh*GH+hh)*QSTRIDE + tid];
    for (int i=0;i<C;i++) {
        bool ok = (i < nvalid);
        sk[i][tid] = ok ? kbase[(size_t)i*(n_kv*HD) + tid] : 0.0f;
        sv[i][tid] = ok ? vbase[(size_t)i*(n_kv*HD) + tid] : 0.0f;
    }
    __syncthreads();

    // Phase A: scores ss[hh][i] = dot(q[h], K[t0+i]) * scale, warp-per-token.
    if (warp < C) {
        int i = warp;
        float kv[8];
        #pragma unroll
        for (int k=0;k<8;k++) kv[k] = sk[i][8*lane + k];
        for (int hh=0;hh<GH;hh++){
            float d=0;
            #pragma unroll
            for (int k=0;k<8;k++) d += sq[hh][8*lane+k]*kv[k];
            #pragma unroll
            for (int off=16;off>0;off>>=1) d += __shfl_xor_sync(0xffffffffu,d,off);
            if (lane==0) ss[hh][i] = (i<nvalid) ? d*scale : -INFINITY;
        }
    }
    __syncthreads();

    // Phase B: per-head chunk softmax (GH threads, one per head).
    if (tid < GH) {
        int hh=tid;
        float m=-INFINITY, l=0;
        for(int i=0;i<C;i++) m=fmaxf(m,ss[hh][i]);
        for(int i=0;i<C;i++){ ss[hh][i]=expf(ss[hh][i]-m); l+=ss[hh][i]; }
        sm[hh]=m; sl[hh]=l;
    }
    __syncthreads();

    // Phase C: o_c[hh][d] = sum_i p[hh][i]*V[t0+i][d]  (unnormalized).
    float acc[GH];
    #pragma unroll
    for (int hh=0;hh<GH;hh++) acc[hh]=0.0f;
    for (int i=0;i<C;i++){
        float vv = sv[i][tid];
        for (int hh=0;hh<GH;hh++) acc[hh] += ss[hh][i]*vv;
    }
    for (int hh=0;hh<GH;hh++){
        int head = kvh*GH+hh;
        ws[((size_t)head*nmax + c)*(HD+2) + tid] = acc[hh];
    }
    if (tid==0){
        for (int hh=0;hh<GH;hh++){
            int head = kvh*GH+hh;
            ws[((size_t)head*nmax + c)*(HD+2) + HD]     = sm[hh];
            ws[((size_t)head*nmax + c)*(HD+2) + HD + 1] = sl[hh];
        }
    }
}

// Reduce over chunks -> out[h][HD] (final, gate applied). grid=(n_head,1), HD thr.
template <int HD, int C, int QSTRIDE>
__global__ void k_gqa_reduce(const float *__restrict__ ws, const float *__restrict__ gate,
                             float *__restrict__ out, int n_head, const int *params, int nmax) {
    int h = blockIdx.x; if (h>=n_head) return;
    int tid = threadIdx.x;
    int n_t = params[1] + 1;
    int nchunks = (n_t + C-1)/C;      // only the active chunks (stride is nmax)
    float M=-INFINITY, Z=0.0f, oacc=0.0f;
    for (int c=0;c<nchunks;c++){
        int base = (int)((size_t)h*nmax + c)*(HD+2);
        float oc = ws[base + tid];
        float mc = ws[base + HD];
        float lc = ws[base + HD + 1];
        float newM = fmaxf(M, mc);
        float scaleM = expf(M - newM);
        oacc *= scaleM;
        Z    *= scaleM;
        float w = expf(mc - newM);
        Z    += w*lc;
        oacc += w*oc;
        M = newM;
    }
    float g = gate[(size_t)h*QSTRIDE + tid];
    out[(size_t)h*HD + tid] = (oacc/Z)*q35_sigmoid(g);
}

// ---------------------------------------------------------------------------
// Q8_0 batched GEMM (batched quantizer above + this kernel + q35_matmul_q8)
// ---------------------------------------------------------------------------

__global__ void k_gemm_q8(const unsigned char *__restrict__ W,
                          const unsigned char *__restrict__ Xq,
                          float *__restrict__ out, int B, int rows, int cols) {
    const int row_blocks = cols >> 5;

    // ---------------- B == 1 : bit-exact matvec (== k_matvec_q8) ----------------
    if (B == 1) {
        const int warp = threadIdx.x >> 5;
        const int lane = threadIdx.x & 31;
        const int row = blockIdx.x * (blockDim.x >> 5) + warp;
        if (row >= rows) return;
        const unsigned char *wrow = W + (size_t)row * (size_t)row_blocks * 34;
        const unsigned char *w0 = wrow + (size_t)lane * 34;
        const unsigned char *x0 = Xq + (size_t)lane * 34;
        float acc = 0.0f;
        if (lane < row_blocks) {
            const uint16_t *wp = (const uint16_t *)(w0 + 2);
            const uint16_t *xp = (const uint16_t *)(x0 + 2);
            uint16_t wv[16], xv[16];
#pragma unroll
            for (int k = 0; k < 16; k++) { wv[k] = wp[k]; xv[k] = xp[k]; }
            float ws = q35_ld_scale(w0);
            float xs = q35_ld_scale(x0);
            float nws = 0.0f, nxs = 0.0f;
            if (lane + 32 < row_blocks) {
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
                    const unsigned char *xn = Xq + (size_t)(i + 32) * 34;
                    const uint16_t *wnp = (const uint16_t *)(wn + 2);
                    const uint16_t *xnp = (const uint16_t *)(xn + 2);
#pragma unroll
                    for (int k = 0; k < 16; k++) { wv[k] = wnp[k]; xv[k] = xnp[k]; }
                    ws = nws;
                    xs = nxs;
                    if (i + 64 < row_blocks) {
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
        return;
    }

    // ---------------- B <= 16 : batched matvec (1 warp per (row,batch) pair) -
    // Each warp runs the EXACT k_matvec_q8 body for pair (r, b): bit-identical
    // per row for every B in this range.  W rows are L2-hot (re-read B times,
    // fits in L2), so cost is ~B-independent; this beats any tiled layout at
    // small B where a 16-wide batch tile would be half-empty.
    if (B <= 16) {
        const int warp = threadIdx.x >> 5;
        const int lane = threadIdx.x & 31;
        const int npairs = rows * B;
        const int idx = blockIdx.x * (blockDim.x >> 5) + warp;
        if (idx >= npairs) return;
        const int r = idx / B, b = idx - r * B;   // = idx % B
        const unsigned char *wrow = W + (size_t)r * (size_t)row_blocks * 34;
        const unsigned char *w0 = wrow + (size_t)lane * 34;
        const unsigned char *xb = Xq + (size_t)b * (size_t)row_blocks * 34;
        const unsigned char *x0 = xb + (size_t)lane * 34;
        float acc = 0.0f;
        if (lane < row_blocks) {
            const uint16_t *wp = (const uint16_t *)(w0 + 2);
            const uint16_t *xp = (const uint16_t *)(x0 + 2);
            uint16_t wv[16], xv[16];
#pragma unroll
            for (int k = 0; k < 16; k++) { wv[k] = wp[k]; xv[k] = xp[k]; }
            float ws = q35_ld_scale(w0);
            float xs = q35_ld_scale(x0);
            float nws = 0.0f, nxs = 0.0f;
            if (lane + 32 < row_blocks) {
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
                    const unsigned char *xn = xb + (size_t)(i + 32) * 34;
                    const uint16_t *wnp = (const uint16_t *)(wn + 2);
                    const uint16_t *xnp = (const uint16_t *)(xn + 2);
#pragma unroll
                    for (int k = 0; k < 16; k++) { wv[k] = wnp[k]; xv[k] = xnp[k]; }
                    ws = nws;
                    xs = nxs;
                    if (i + 64 < row_blocks) {
                        nws = q35_ld_scale(wn + 32 * 34);
                        nxs = q35_ld_scale(xn + 32 * 34);
                    }
                }
            }
        }
#pragma unroll
        for (int off = 16; off > 0; off >>= 1)
            acc += __shfl_down_sync(0xffffffffu, acc, off);
        if (lane == 0) out[(size_t)b * rows + r] = acc;
        return;
    }

    // ---------------- B > 16 : row-tiled Q8_0 GEMM --------------------------
    // grid (rows/128, B/32); block 256 = 16x16; per thread: 1 batch row (bx)
    // x 16 W rows (by*16+j).  K-loop over 32-col chunks (one 34B Q8_0 block):
    // threads 0..31 load the 32 X rows, threads 32..159 the 128 W rows,
    // dequantized to f32 in shared (34-word rows -> 128-bit loads are
    // <=2-way bank conflicts).
    // DRAM: every W byte is fetched exactly ONCE per launch (W matrix is
    // 6.7MB, L2-resident); X tiles are re-served from L2 by the other 48
    // row-tile blocks.  At B>=64 the FMA pipe is the limit (COMPUTE-bound).
    __shared__ float sX[32][44];
    __shared__ float sW[128][36];
    const int bx = threadIdx.x & 31;
    const int by = threadIdx.x >> 5;
    const int tid = threadIdx.x;
    const int b0 = blockIdx.y * 32;
    const int r0 = blockIdx.x * 128;

    float acc[16];
#pragma unroll
    for (int j = 0; j < 16; j++) acc[j] = 0.0f;

    for (int k0 = 0; k0 < row_blocks; k0++) {
        if (tid < 32) {            // X row tid, block k0
            const unsigned char *p = Xq + (size_t)(b0 + tid) * (size_t)row_blocks * 34 +
                                     (size_t)k0 * 34;
            const uint16_t *pu = (const uint16_t *)(p + 2);
            uint16_t v[16];
            float d[32];
            if (b0 + tid < B) {
#pragma unroll
                for (int k = 0; k < 16; k++) v[k] = pu[k];
                float s = q35_ld_scale(p);
#pragma unroll
                for (int k = 0; k < 16; k++) {
                    d[2 * k] = s * (float)(signed char)(v[k] & 0xff);
                    d[2 * k + 1] = s * (float)(signed char)(v[k] >> 8);
                }
            } else {
#pragma unroll
                for (int k = 0; k < 32; k++) d[k] = 0.0f;
            }
            float *row = sX[tid];
#pragma unroll
            for (int k = 0; k < 8; k++)
                ((float4 *)row)[k] = make_float4(d[4 * k], d[4 * k + 1], d[4 * k + 2], d[4 * k + 3]);
        } else if (tid < 160) {    // W row tid-32, block k0
            const unsigned char *p = W + (size_t)(r0 + tid - 32) * (size_t)row_blocks * 34 +
                                     (size_t)k0 * 34;
            const uint16_t *pu = (const uint16_t *)(p + 2);
            uint16_t v[16];
            float d[32];
            if (r0 + tid - 32 < rows) {
#pragma unroll
                for (int k = 0; k < 16; k++) v[k] = pu[k];
                float s = q35_ld_scale(p);
#pragma unroll
                for (int k = 0; k < 16; k++) {
                    d[2 * k] = s * (float)(signed char)(v[k] & 0xff);
                    d[2 * k + 1] = s * (float)(signed char)(v[k] >> 8);
                }
            } else {
#pragma unroll
                for (int k = 0; k < 32; k++) d[k] = 0.0f;
            }
            float *row = sW[tid - 32];
#pragma unroll
            for (int k = 0; k < 8; k++)
                ((float4 *)row)[k] = make_float4(d[4 * k], d[4 * k + 1], d[4 * k + 2], d[4 * k + 3]);
        }
        __syncthreads();
#pragma unroll
        for (int k4 = 0; k4 < 8; k4++) {
            float4 x4 = ((float4 *)(sX[bx] + 4 * k4))[0];
#pragma unroll
            for (int j = 0; j < 16; j++) {
                float4 w4 = ((float4 *)(sW[by * 16 + j] + 4 * k4))[0];
                acc[j] = fmaf(x4.x, w4.x, acc[j]);
                acc[j] = fmaf(x4.y, w4.y, acc[j]);
                acc[j] = fmaf(x4.z, w4.z, acc[j]);
                acc[j] = fmaf(x4.w, w4.w, acc[j]);
            }
        }
        __syncthreads();
    }
    const int b = b0 + bx;
    if (b < B) {
        const float *ac = acc;
#pragma unroll
        for (int j = 0; j < 16; j++)
        if (r0 + by * 16 + j < rows)
            out[(size_t)b * rows + (r0 + by * 16 + j)] = ac[j];
    }
}

// ---------------------------------------------------------------------------
// (6) flat extern "C" launchers
// ---------------------------------------------------------------------------

static inline int q35_grid(int n, int block) { return (n + block - 1) / block; }

extern "C" void q35_quantize_q8(const float *x, unsigned char *out, int nblocks) {
    if (nblocks <= 0) return;
    k_quantize_q8<<<q35_grid(nblocks, Q35_BLK), Q35_BLK, 0, g_stream>>>(x, out, nblocks);
}

extern "C" void q35_matvec_q8(const unsigned char *w, const unsigned char *xq,
                              float *out, int rows, int row_blocks) {
    if (rows <= 0 || row_blocks <= 0) return;
    int warps = 8;
    k_matvec_q8<<<q35_grid(rows, warps), warps * 32, 0, g_stream>>>(w, xq, out, rows, row_blocks);
}

extern "C" void q35_matmul_q8(const unsigned char *w, const float *xf, unsigned char *xq,
                              float *out, int B, int rows, int cols) {
    if (B <= 0 || rows <= 0 || cols <= 0) return;
    const int nblocks = cols >> 5;
    if (B > 1)
        k_quantize_q8_batch<<<q35_grid(B * nblocks, Q35_BLK), Q35_BLK, 0, g_stream>>>(xf, xq, B, nblocks);
    else
        k_quantize_q8<<<q35_grid(nblocks, Q35_BLK), Q35_BLK, 0, g_stream>>>(xf, xq, nblocks);
    if (B == 1)
        k_gemm_q8<<<q35_grid(rows, 8), 8 * 32, 0, g_stream>>>(w, xq, out, B, rows, cols);
    else if (B <= 16)
        k_gemm_q8<<<q35_grid(rows * B, 8), 8 * 32, 0, g_stream>>>(w, xq, out, B, rows, cols);
    else {
        dim3 grid((rows + 127) / 128, (B + 31) / 32);
        k_gemm_q8<<<grid, 256, 0, g_stream>>>(w, xq, out, B, rows, cols);
    }
}

extern "C" void q35_matvec_f32(const float *w, const float *x, float *out,
                               int rows, int cols) {
    if (rows <= 0 || cols <= 0) return;
    int warps = 8;
    k_matvec_f32<<<q35_grid(rows, warps), warps * 32, 0, g_stream>>>(w, x, out, rows, cols);
}

extern "C" void q35_embed_row(const unsigned char *w, float *x, int n) {
    if (n <= 0) return;
    k_embed_row<<<q35_grid(n, Q35_BLK), Q35_BLK, 0, g_stream>>>(w, x, g_params_dev, n);
}

extern "C" void q35_rmsnorm_rows(const float *x, const float *w, float *out,
                                 int rows, int n, float eps) {
    if (rows <= 0 || n <= 0) return;
    if (n <= 128)
        k_rmsnorm_rows<128><<<rows, 128, 0, g_stream>>>(x, w, out, rows, n, eps);
    else
        k_rmsnorm_rows<256><<<rows, 256, 0, g_stream>>>(x, w, out, rows, n, eps);
}

extern "C" void q35_rmsnorm_strided(const float *x, const float *w, float *out,
                                    int rows, int n, int stride, float eps) {
    if (rows <= 0 || n <= 0) return;
    if (n <= 128)
        k_rmsnorm_strided<128><<<rows, 128, 0, g_stream>>>(x, w, out, rows, n, stride, eps);
    else
        k_rmsnorm_strided<256><<<rows, 256, 0, g_stream>>>(x, w, out, rows, n, stride, eps);
}

extern "C" void q35_l2norm(float *x, int rows, int n, float eps, float scale) {
    if (rows <= 0 || n <= 0) return;
    if (n <= 128)
        k_l2norm_rows<128><<<rows, 128, 0, g_stream>>>(x, rows, n, eps, scale);
    else
        k_l2norm_rows<256><<<rows, 256, 0, g_stream>>>(x, rows, n, eps, scale);
}

// Batched l2norm for prefill: x is [B][token_stride]; each token has `nheads`
// rows of `n` floats at offsets 0, n, 2n, ... (token-major).  In place;
// per-row math == k_l2norm_rows (bit-identical).  grid = B*nheads.
__global__ void k_l2norm_strided2d(float *x, int B, int nheads, int n,
                                   int token_stride, float eps, float scale) {
    int r = blockIdx.x;
    if (r >= B * nheads) return;
    int b = r / nheads;
    int h = r - b * nheads;
    float *xr = x + (size_t)b * token_stride + (size_t)h * n;
    __shared__ float red[Q35_BLK];
    int tid = threadIdx.x;
    float ss = 0.0f;
    for (int i = tid; i < n; i += Q35_BLK) {
        float v = xr[i];
        ss += v * v;
    }
    red[tid] = ss;
    __syncthreads();
    for (int s = Q35_BLK / 2; s > 0; s >>= 1) {
        if (tid < s) red[tid] += red[tid + s];
        __syncthreads();
    }
    float m = 1.0f / sqrtf(red[0] + eps) * scale;
    for (int i = tid; i < n; i += Q35_BLK) xr[i] *= m;
}

extern "C" void q35_l2norm_strided2d(float *x, int B, int nheads, int n,
                                     int token_stride, float eps, float scale) {
    if (B <= 0 || nheads <= 0 || n <= 0) return;
    k_l2norm_strided2d<<<B * nheads, Q35_BLK, 0, g_stream>>>(
        x, B, nheads, n, token_stride, eps, scale);
}

extern "C" void q35_rope(float *x, int rows, int hd, int rope_dim, int stride,
                         float base) {
    if (rows <= 0 || rope_dim <= 0) return;
    int half = rope_dim / 2;
    k_rope<<<rows, half, 0, g_stream>>>(x, rows, hd, rope_dim, stride, g_params_dev, base);
}

extern "C" void q35_conv1d(const float *wconv, float *x, float *hist,
                           int conv_dim, int k) {
    if (conv_dim <= 0 || k < 2) return;
    k_conv1d<<<q35_grid(conv_dim, Q35_BLK), Q35_BLK, 0, g_stream>>>(wconv, x, hist, conv_dim, k);
}

extern "C" void q35_beta_gamma(const float *a, const float *b, const float *abar,
                               const float *dtb, float *beta, float *gamma,
                               int nv) {
    if (nv <= 0) return;
    k_beta_gamma<<<q35_grid(nv, 128), 128, 0, g_stream>>>(a, b, abar, dtb, beta, gamma, nv);
}

// Batched causal conv1d for prefill: x is [B][conv_dim] token-major; the
// per-channel history (hist, [conv_dim][K-1]) is shifted after each token, so
// token t's tap window is the previous K-1 tokens' RAW samples (or hist's
// pre-existing values) plus its own.  Per-token math == k_conv1d (bit-identical).
__global__ void k_conv1d_batch(const float *__restrict__ wconv,
                               float *__restrict__ x, float *hist, int B,
                               int conv_dim, int K) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= conv_dim) return;
    const float *w = wconv + (size_t)i * K;
    float *h = hist + (size_t)i * (K - 1);
    for (int t = 0; t < B; t++) {
        float sum = 0.0f;
        for (int j = 0; j < K - 1; j++) sum += h[j] * w[j];
        float cur = x[(size_t)t * conv_dim + i];
        sum += cur * w[K - 1];
        for (int j = 0; j < K - 2; j++) h[j] = h[j + 1];
        h[K - 2] = cur;
        x[(size_t)t * conv_dim + i] = q35_silu(sum);
    }
}

extern "C" void q35_conv1d_batch(const float *wconv, float *x, float *hist,
                                 int B, int conv_dim, int K) {
    if (conv_dim <= 0 || B <= 0 || K < 2) return;
    k_conv1d_batch<<<q35_grid(conv_dim, Q35_BLK), Q35_BLK, 0, g_stream>>>(
        wconv, x, hist, B, conv_dim, K);
}

// Batched beta/gamma: a/b are [B][nv], abar/dtb [nv]; beta/gamma [B][nv].
// Per-token math == k_beta_gamma (bit-identical).
__global__ void k_beta_gamma_batch(const float *__restrict__ a,
                                  const float *__restrict__ b,
                                  const float *__restrict__ abar,
                                  const float *__restrict__ dtb,
                                  float *__restrict__ beta,
                                  float *__restrict__ gamma, int nv, int B) {
    int total = B * nv;
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < total;
         i += gridDim.x * blockDim.x) {
        int h = i % nv;
        beta[i] = q35_sigmoid(b[i]);
        float x = a[i] + dtb[h];
        float sp = x > 20.0f ? x : log1pf(expf(x));
        gamma[i] = expf(abar[h] * sp);
    }
}

extern "C" void q35_beta_gamma_batch(const float *a, const float *b,
                                     const float *abar, const float *dtb,
                                     float *beta, float *gamma, int nv, int B) {
    if (nv <= 0 || B <= 0) return;
    int total = B * nv;
    k_beta_gamma_batch<<<q35_grid(total, Q35_BLK), Q35_BLK, 0, g_stream>>>(
        a, b, abar, dtb, beta, gamma, nv, B);
}

extern "C" void q35_gdn_state(float *qkv, float *state, const float *beta,
                              const float *gamma, float *out, int nv, int nk,
                              int kd, int vd) {
    if (nv <= 0) return;
    if (kd > 512 || vd > 1024 || nk > nv) {
        fprintf(stderr, "[cuda] gdn_state: kd=%d vd=%d out of range\n", kd, vd);
        exit(1);
    }
    // Column-tile width: keep the S tile (kd*cvd0 floats) within Ss[8192]
    // (~32KB static shared) while making nsplits* nv blocks to fill the SMs.
    int cvd0 = 32;
    int maxc = 8192 / kd;
    if (cvd0 > maxc) cvd0 = maxc;
    if (cvd0 > vd) cvd0 = vd;
    if (cvd0 < 1) cvd0 = 1;
    int nsplits = (vd + cvd0 - 1) / cvd0;
    k_gdn_state<<<dim3((unsigned)nsplits, (unsigned)nv), 256, 0, g_stream>>>(
        qkv, state, beta, gamma, out, nv, nk, kd, vd, cvd0);
}

// Batched RoPE: same memory layout as k_rope (one block per row/head,
// half = rope_dim/2 threads, interleave at j and j+half), but row b uses
// position pos0 + b.  Same powf/cosf/sinf math -> bit-identical to calling
// q35_rope on a single row with pos = pos0 + b.
__global__ void k_rope_batch(float *__restrict__ x, int rows, int hd, int rope_dim,
                             int stride, float base, int pos0) {
    int h = blockIdx.x;
    if (h >= rows) return;
    int half = rope_dim / 2;
    int j = threadIdx.x;
    if (j >= half) return;
    int pos = pos0 + h;
    float *head = x + (size_t)h * stride;
    float inv = powf(base, -2.0f * (float)j / (float)rope_dim);
    float theta = (float)pos * inv;
    float c = cosf(theta);
    float s = sinf(theta);
    float a = head[j];
    float b = head[j + half];
    head[j] = a * c - b * s;
    head[j + half] = a * s + b * c;
}

extern "C" void q35_rope_batch(float *x, int rows, int hd, int rope_dim, int stride,
                               float base, int pos0) {
    if (rows <= 0 || rope_dim <= 0) return;
    int half = rope_dim / 2;
    k_rope_batch<<<rows, half, 0, g_stream>>>(x, rows, hd, rope_dim, stride, base, pos0);
}

// Batched gated-delta-net state update (VERBATIM from /tmp/batched_gdn.cu,
// verified bit-identical to k_gdn_state per token).  grid=(nsplits, nv),
// block 256; loops t=0..B-1 inside the block; the (kd x cvd0) S tile is
// loaded from DRAM once and written back once; qkv is token-major with
// conv_dim = 2*nk*kd + nv*kd per token; out is [B][nv*vd].
__global__ void __launch_bounds__(256) k_gdn_state_batch(
    float *__restrict__ qkv, float *__restrict__ state,
    const float *__restrict__ beta, const float *__restrict__ gamma,
    float *__restrict__ out, int nv, int nk, int kd, int vd, int cvd0,
    int B) {
    __shared__ float Ss[8192]; /* tile, row-major Ss[i*cvd + j] */
    __shared__ float ksh[512];
    __shared__ float qsh[512];

    const int tid = threadIdx.x;
    const int h = blockIdx.y;
    const int s = blockIdx.x;
    if (h >= nv) return;
    const int c0 = s * cvd0;
    const int c1 = c0 + cvd0 < vd ? c0 + cvd0 : vd;
    const int cvd = c1 - c0;
    if (cvd <= 0) return;

    const int kq_dim = nk * kd;
    const int kh = h % nk;
    const int conv_dim = 2 * kq_dim + nv * kd;
    float *S = state + (size_t)h * kd * vd;

    const int n = kd * cvd; /* tile elements */

    /* load the S tile ONCE (ONE DRAM read) */
    for (int x = tid; x < n; x += 256) {
        int i = x / cvd;
        Ss[x] = S[(size_t)i * vd + c0 + (x - i * cvd)];
    }
    __syncthreads();

    for (int t = 0; t < B; t++) {
        const float *qkv_t = qkv + (size_t)t * conv_dim;
        const float *q_h = qkv_t + (size_t)kh * kd;
        const float *k_h = qkv_t + (size_t)kq_dim + (size_t)kh * kd;
        const float *v_h = qkv_t + (size_t)2 * kq_dim + (size_t)h * kd;

        /* k, q to shared (grid-stride since blockDim=256 may exceed kd). */
        for (int x = tid; x < kd; x += 256) {
            ksh[x] = k_h[x];
            qsh[x] = q_h[x];
        }
        __syncthreads();

        const float g = gamma[(size_t)t * nv + h];
        const float bt = beta[(size_t)t * nv + h];

        /* (1) decay: Ss *= g, 256 threads over the tile */
        for (int x = tid; x < n; x += 256) Ss[x] *= g;
        __syncthreads();

        /* (2)-(5) one thread per column (tid < cvd), same order as k_gdn_state */
        if (tid < cvd) {
            float *Sc = Ss + tid; /* Sc[i*cvd] == Ss[i*cvd + tid] */
            float km = 0.0f;
            for (int i = 0; i < kd; i++) km += Sc[i * cvd] * ksh[i];
            float delta = bt * (v_h[c0 + tid] - km);
            for (int i = 0; i < kd; i++) Sc[i * cvd] += ksh[i] * delta;
            float o = 0.0f;
            for (int i = 0; i < kd; i++) o += Sc[i * cvd] * qsh[i];
            out[(size_t)t * (nv * vd) + (size_t)h * vd + c0 + tid] = o;
        }
        __syncthreads();
    }

    /* write tile back ONCE (ONE DRAM write) */
    for (int x = tid; x < n; x += 256) {
        int i = x / cvd;
        S[(size_t)i * vd + c0 + (x - i * cvd)] = Ss[x];
    }
}

extern "C" void q35_gdn_state_batch(float *qkv, float *state, const float *beta,
                                    const float *gamma, float *out, int nv, int nk,
                                    int kd, int vd, int B) {
    if (nv <= 0) return;
    if (kd > 512 || vd > 1024 || nk > nv) {
        fprintf(stderr, "[cuda] gdn_state_batch: kd=%d vd=%d out of range\n", kd, vd);
        exit(1);
    }
    // Same column-tile math as q35_gdn_state: keep the S tile (kd*cvd0
    // floats) within Ss[8192] (~32KB static shared).
    int cvd0 = 32;
    int maxc = 8192 / kd;
    if (cvd0 > maxc) cvd0 = maxc;
    if (cvd0 > vd) cvd0 = vd;
    if (cvd0 < 1) cvd0 = 1;
    int nsplits = (vd + cvd0 - 1) / cvd0;
    k_gdn_state_batch<<<dim3((unsigned)nsplits, (unsigned)nv), 256, 0, g_stream>>>(
        qkv, state, beta, gamma, out, nv, nk, kd, vd, cvd0, B);
}

extern "C" void q35_gated_rmsnorm(float *out, const float *w, const float *z,
                                  int rows, int n, float eps) {
    if (rows <= 0 || n <= 0) return;
    if (n <= 128)
        k_gated_rmsnorm<128><<<rows, 128, 0, g_stream>>>(out, w, z, rows, n, eps);
    else
        k_gated_rmsnorm<256><<<rows, 256, 0, g_stream>>>(out, w, z, rows, n, eps);
}

extern "C" void q35_attn(const float *q, const float *kc, const float *vc,
                         const float *gate, float *out, int heads, int kv_heads,
                         int hd, int kv_dim, int q_stride, float scale) {
    if (heads <= 0) return;
    if (hd == Q35_FD_HD && q_stride == Q35_FD_QSTRIDE && (heads / kv_heads) == Q35_FD_GH
        && g_attn_ws != nullptr) {
        // GQA flash-decoding path (real model: hd=256, group=4).  Fixed grid
        // (kv_heads, max_seq/C); n_t read from the device param block at
        // runtime, so the static graph needs no host-side S branch.  K1 writes
        // per-(kv_head,chunk,head) partials; K2 merges + applies the sigmoid
        // gate.  Always-new is faster than k_attn at every S (see header).
        k_gqa_partial<Q35_FD_HD, Q35_FD_C, Q35_FD_GH, Q35_FD_QSTRIDE>
            <<<dim3((unsigned)kv_heads, (unsigned)Q35_FD_NMAX), Q35_FD_HD, 0, g_stream>>>(
                q, kc, vc, g_attn_ws, heads, kv_heads, g_params_dev, scale, Q35_FD_NMAX);
        k_gqa_reduce<Q35_FD_HD, Q35_FD_C, Q35_FD_QSTRIDE>
            <<<heads, Q35_FD_HD, 0, g_stream>>>(
                g_attn_ws, gate, out, heads, g_params_dev, Q35_FD_NMAX);
        return;
    }
    // Fallback: legacy per-head decode (other head dims, or if the flash-decode
    // workspace was not allocated).  n_t from the device param block; the shared
    // scratch covers max_seq (graph is static).
    const size_t max_nt = 4096;
    size_t shmem = max_nt * sizeof(float);
    k_attn<<<heads, 256, shmem, g_stream>>>(q, kc, vc, gate, out, heads, kv_heads,
                                           hd, g_params_dev, kv_dim, q_stride, scale);
}

extern "C" void q35_silu_mul(float *h, const float *h1, int n) {
    if (n <= 0) return;
    k_silu_mul<<<q35_grid(n, Q35_BLK), Q35_BLK, 0, g_stream>>>(h, h1, n);
}

extern "C" void q35_sigmoid_mul(float *o, const float *g, int n) {
    if (n <= 0) return;
    k_sigmoid_mul<<<q35_grid(n, Q35_BLK), Q35_BLK, 0, g_stream>>>(o, g, n);
}

extern "C" void q35_add(float *a, const float *b, int n) {
    if (n <= 0) return;
    k_add<<<q35_grid(n, Q35_BLK), Q35_BLK, 0, g_stream>>>(a, b, n);
}

// KV-cache slot store.  The destination is `kvcache + slot*kv_dim` f32s where
// slot = pos (device param), which a cudaMemcpy node cannot express, so this is
// a kernel.  One thread per element (kv_dim <= 1024 = block limit).
__global__ void k_kvstore(float *kvcache, const float *__restrict__ src,
                          const int *params, int kv_dim) {
    int i = threadIdx.x;
    if (i >= kv_dim) return;
    float *dst = kvcache + (size_t)params[1] * kv_dim;
    dst[i] = src[i];
}

extern "C" void q35_kvstore(float *kvcache, const float *src, int kv_dim) {
    if (kv_dim <= 0) return;
    k_kvstore<<<1, kv_dim, 0, g_stream>>>(kvcache, src, g_params_dev, kv_dim);
}

// Batched KV-cache slot store: row b of qkv (kv_dim floats, one token's
// concatenated k|v row — same layout the engine feeds to q35_kvstore) goes to
// kc[pos0 + b] and vc[pos0 + b].  One block per row, kv_dim threads, same
// element-wise copy as k_kvstore -> bit-identical to B separate calls.
__global__ void k_kvstore_batch(const float *__restrict__ qkv, float *kc, float *vc,
                                int kv_dim, int B, int pos0) {
    int b = blockIdx.x;
    if (b >= B) return;
    int i = threadIdx.x;
    if (i >= kv_dim) return;
    const float *src = qkv + (size_t)b * kv_dim;
    float *kdst = kc + (size_t)(pos0 + b) * kv_dim;
    float *vdst = vc + (size_t)(pos0 + b) * kv_dim;
    kdst[i] = src[i];
    vdst[i] = src[i];
}

extern "C" void q35_kvstore_batch(const float *qkv, float *kc, float *vc,
                                  int kv_dim, int B, int pos0) {
    if (kv_dim <= 0 || B <= 0) return;
    k_kvstore_batch<<<B, kv_dim, 0, g_stream>>>(qkv, kc, vc, kv_dim, B, pos0);
}

// Batched causal attention (VERBATIM from /tmp/batched_attn.cu): B query
// tokens at positions pos0..pos0+B-1, causal over token-major KV
// kc/vc [n_t][kv_dim] where n_t = pos0+B.  q is [B][heads][q_stride] with
// q[b][h][0..hd) = query and [hd..2hd) = sigmoid gate; out is [B][heads][hd]
// (sigmoid gate applied in-kernel, same as k_attn).  grid=(B, heads),
// block Q35_BLK (= hd), dynamic shared (n_t floats) for scores.
__global__ void k_attn_batch(const float *__restrict__ q, const float *__restrict__ kc,
                             const float *__restrict__ vc, float *__restrict__ out,
                             int heads, int kv_heads, int hd, int kv_dim, int q_stride,
                             int B, int pos0, float scale) {
    int b = blockIdx.x;
    if (b >= B) return;
    int h = blockIdx.y;
    int tid = threadIdx.x;
    int n_t = pos0 + b + 1; // causal length for this query (t = 0..p inclusive)
    extern __shared__ float shsc[];
    __shared__ float red[Q35_BLK];
    int gqa = heads / kv_heads;
    int kvh = h / gqa;
    const float *qh = q + ((size_t)b * heads + h) * q_stride;
    const float *gh = qh + hd;
    const float *kh = kc + (size_t)kvh * hd;
    const float *vh = vc + (size_t)kvh * hd;
    float *orow = out + ((size_t)b * heads + h) * hd;

    // scores: one block-dot per key token (t loop), hd==blockDim.x so 1 elem/thread
    for (int t = 0; t < n_t; t++) {
        const float *kt = kh + (size_t)t * kv_dim;
        float p = 0.0f;
        for (int i = tid; i < hd; i += blockDim.x) p += qh[i] * kt[i];
        red[tid] = p;
        __syncthreads();
        for (int s = blockDim.x / 2; s > 0; s >>= 1) {
            if (tid < s) red[tid] += red[tid + s];
            __syncthreads();
        }
        if (tid == 0) shsc[t] = red[0] * scale;
        __syncthreads();
    }

    // parallel max
    float lm = -INFINITY;
    for (int t = tid; t < n_t; t += blockDim.x) lm = fmaxf(lm, shsc[t]);
    red[tid] = lm;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) red[tid] = fmaxf(red[tid], red[tid + s]);
        __syncthreads();
    }
    float mx = red[0];
    __syncthreads(); // all threads finished reading red[0] as mx before any
                    // writes red[tid] below (esp. thread 0 overwriting red[0])

    // exp + sum
    float ls = 0.0f;
    for (int t = tid; t < n_t; t += blockDim.x) {
        float e = expf(shsc[t] - mx);
        shsc[t] = e;
        ls += e;
    }
    red[tid] = ls;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) red[tid] += red[tid + s];
        __syncthreads();
    }
    float sum = red[0];
    for (int t = tid; t < n_t; t += blockDim.x) shsc[t] /= sum;
    __syncthreads();

    // weighted V accumulation + sigmoid gate (1 elem per thread, hd==blockDim.x)
    for (int i = tid; i < hd; i += blockDim.x) {
        float acc = 0.0f;
        for (int t = 0; t < n_t; t++) acc += shsc[t] * vh[(size_t)t * kv_dim + i];
        orow[i] = acc * q35_sigmoid(gh[i]);
    }
}

extern "C" void q35_attn_batch(const float *q, const float *kc, const float *vc,
                               float *out, int heads, int kv_heads, int hd, int kv_dim,
                               int q_stride, int B, int pos0, float scale) {
    if (B <= 0 || heads <= 0) return;
    int n_t = pos0 + B;
    size_t shmem = (size_t)n_t * sizeof(float);
    k_attn_batch<<<dim3((unsigned)B, (unsigned)heads), Q35_BLK, shmem, g_stream>>>(
        q, kc, vc, out, heads, kv_heads, hd, kv_dim, q_stride, B, pos0, scale);
}

// Batched RoPE for prefill: grid=(B, heads).  Token b uses position pos0 + b
// for ALL of its heads (unlike k_rope_batch, whose row index IS the position
// offset and is only correct when one row per token).  Same powf/cosf/sinf
// math as k_rope -> bit-identical per head.  x is [B][heads][stride].
__global__ void k_rope2(float *__restrict__ x, int B, int heads, int rope_dim,
                        int stride, float base, int pos0) {
    int b = blockIdx.x;
    if (b >= B) return;
    int h = blockIdx.y;
    int half = rope_dim / 2;
    int j = threadIdx.x;
    if (j >= half) return;
    int pos = pos0 + b;
    float *head = x + ((size_t)b * heads + h) * stride;
    float inv = powf(base, -2.0f * (float)j / (float)rope_dim);
    float theta = (float)pos * inv;
    float c = cosf(theta);
    float s = sinf(theta);
    float a = head[j];
    float b2 = head[j + half];
    head[j] = a * c - b2 * s;
    head[j + half] = a * s + b2 * c;
}

extern "C" void q35_rope2(float *x, int B, int heads, int rope_dim, int stride,
                          float base, int pos0) {
    if (B <= 0 || heads <= 0 || rope_dim <= 0) return;
    int half = rope_dim / 2;
    k_rope2<<<dim3((unsigned)B, (unsigned)heads), half, 0, g_stream>>>(
        x, B, heads, rope_dim, stride, base, pos0);
}

// Batched KV-cache store for prefill (ONE cache).  src is [B][kv_dim]
// token-major; token b goes to slot pos0 + b.  (The in-tree q35_kvstore_batch
// writes one src to BOTH k and v and is not used by the prefill driver.)
__global__ void k_kvstore2(const float *__restrict__ src, float *cache,
                           int kv_dim, int B, int pos0) {
    int b = blockIdx.x;
    if (b >= B) return;
    const float *s = src + (size_t)b * kv_dim;
    float *dst = cache + (size_t)(pos0 + b) * kv_dim;
    for (int i = threadIdx.x; i < kv_dim; i += blockDim.x) dst[i] = s[i];
}

extern "C" void q35_kvstore2(const float *src, float *cache, int kv_dim, int B,
                             int pos0) {
    if (B <= 0 || kv_dim <= 0) return;
    int blk = kv_dim < 256 ? kv_dim : 256;
    k_kvstore2<<<B, blk, 0, g_stream>>>(src, cache, kv_dim, B, pos0);
}

// Batched Q8_0 embedding gather for prefill: tok[b] = token id for row b;
// x is [B][n] f32.  Same dequant math as k_embed_row (scale at bytes [0..4),
// data at [2..34) of each 34-byte block).
__global__ void k_embed_rows(const unsigned char *__restrict__ w,
                             const int *__restrict__ tok, float *__restrict__ x,
                             int B, int n) {
    int total = B * n;
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < total;
         i += gridDim.x * blockDim.x) {
        int b = i / n;
        int r = i - b * n;
        int row = tok[b];
        int blk = r >> 5;
        int j = r & 31;
        const unsigned char *p = w + ((size_t)row * (size_t)(n >> 5) + blk) * 34;
        x[i] = q35_ld_scale(p) * (float)(signed char)p[2 + j];
    }
}

extern "C" void q35_embed_rows(const unsigned char *w, const int *tok, float *x,
                               int B, int n) {
    if (B <= 0 || n <= 0) return;
    int total = B * n;
    k_embed_rows<<<q35_grid(total, 256), 256, 0, g_stream>>>(w, tok, x, B, n);
}

extern "C" void q35_argmax(const float *x, int *outidx, int n) {
    if (n <= 0) return;
    // Single block: one global reduction over the whole vocab.  (A multi-block
    // launch would race on outidx[0] — see k_argmax.)  argmax is once/token and
    // not on the decode bottleneck path.
    k_argmax<<<1, Q35_BLK, 0, g_stream>>>(x, outidx, n);
}

// ---------------------------------------------------------------------------
// CUDA Graph: host uploads the per-step (token,pos), host runs the whole
// forward under stream capture once (Zig drives it), then replays the graph
// each step.  The graph is fully static: all pointers/dims are fixed at capture
// time; only token/pos change per step, and those are read by the kernels from
// the device param block (so no per-step node param update is needed).
// ---------------------------------------------------------------------------

extern "C" int q35_graph_set_params(int token, int pos) {
    g_params_host[0] = token;
    g_params_host[1] = pos;
    cudaError_t e =
        cudaMemcpyAsync(g_params_dev, g_params_host, sizeof(int) * 2,
                        cudaMemcpyHostToDevice, g_stream);
    if (e != cudaSuccess) {
        fprintf(stderr, "[cuda] graph set_params: %s\n", cudaGetErrorString(e));
        return (int)e;
    }
    return 0;
}

// Begin capturing on g_stream.  The host must then issue the whole forward
// (embedding + per-layer kernels + final logits) on g_stream; all kernel
// launches below go to g_stream, so they are captured as graph nodes.
extern "C" int q35_capture_begin(void) {
    cudaError_t e =
        cudaStreamBeginCapture(g_stream, cudaStreamCaptureModeGlobal);
    if (e != cudaSuccess) {
        fprintf(stderr, "[cuda] capture_begin: %s\n", cudaGetErrorString(e));
        return (int)e;
    }
    return 0;
}

// End capture, instantiate the graph, and (crucially) RESET the device param
// block to (0,0) so the replayed graph's first step (or any step the host
// forgot to set) reads token=0,pos=0 deterministically.  The host will call
// q35_graph_set_params again on g_stream right before each launch.
extern "C" int q35_capture_end(void) {
    cudaError_t e = cudaStreamEndCapture(g_stream, &g_graph);
    if (e != cudaSuccess) {
        fprintf(stderr, "[cuda] capture_end: %s\n", cudaGetErrorString(e));
        return (int)e;
    }
    e = cudaGraphInstantiate(&g_exec, g_graph, nullptr, nullptr, 0);
    if (e != cudaSuccess) {
        fprintf(stderr, "[cuda] graph instantiate: %s\n", cudaGetErrorString(e));
        return (int)e;
    }
    cudaGraphDestroy(g_graph);
    g_graph = nullptr;
    // Reset the param block to a known state; the first real step will set it.
    g_params_host[0] = 0;
    g_params_host[1] = 0;
    cudaMemcpy(g_params_dev, g_params_host, sizeof(int) * 2,
               cudaMemcpyHostToDevice);
    return 0;
}

// Launch the whole-forward graph on g_stream (replaces ~900 host launches).
extern "C" int q35_graph_launch(void) {
    cudaError_t e = cudaGraphLaunch(g_exec, g_stream);
    if (e != cudaSuccess) {
        fprintf(stderr, "[cuda] graph_launch: %s\n", cudaGetErrorString(e));
        return (int)e;
    }
    return 0;
}

extern "C" int q35_graph_destroy(void) {
    if (g_exec) {
        cudaGraphExecDestroy(g_exec);
        g_exec = nullptr;
    }
    return 0;
}

