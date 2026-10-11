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

extern "C" int q35_cuda_init(void) {
    int dev = 0;
    Q35_CHECK(cudaGetDevice(&dev));
    cudaDeviceProp p;
    Q35_CHECK(cudaGetDeviceProperties(&p, dev));
    fprintf(stderr, "[cuda] device %d: %s (sm_%d%d, %.1f GiB, %d SMs)\n", dev,
            p.name, p.major, p.minor, (double)p.totalGlobalMem / 1073741824.0,
            p.multiProcessorCount);
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

// out[r] = sum_c W[r,c] x[c] with W row-major Q8_0, x already quantized.
__global__ void k_matvec_q8(const unsigned char *__restrict__ w,
                            const unsigned char *__restrict__ xq,
                            float *__restrict__ out, int rows, int row_blocks) {
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
                            float *__restrict__ x, int row, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
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
                       int stride, int pos, float base) {
    int h = blockIdx.x;
    if (h >= rows) return;
    int half = rope_dim / 2;
    int j = threadIdx.x;
    if (j >= half) return;
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
__global__ void k_gdn_state(float *__restrict__ qkv, float *__restrict__ state,
                            const float *__restrict__ beta,
                            const float *__restrict__ gamma,
                            float *__restrict__ out, int nv, int nk, int kd,
                            int vd) {
    int h = blockIdx.x;
    if (h >= nv) return;
    int j = threadIdx.x;
    if (j >= vd) return;
    const int kq_dim = nk * kd;
    int kh = h % nk;
    const float *q_h = qkv + (size_t)kh * kd;
    const float *k_h = qkv + (size_t)kq_dim + (size_t)kh * kd;
    const float *v_h = qkv + (size_t)2 * kq_dim + (size_t)h * kd;
    float *S = state + (size_t)h * kd * vd;

    __shared__ float ksh[512];
    __shared__ float qsh[512];
    if (j < kd) {
        ksh[j] = k_h[j];
        qsh[j] = q_h[j];
    }
    __syncthreads();

    float g = gamma[h];
    float bt = beta[h];

    // (1) decay
    for (int i = 0; i < kd; i++) S[(size_t)i * vd + j] *= g;
    // (2) kv_mem = sum_i S[i,j] * k[i]
    float km = 0.0f;
    for (int i = 0; i < kd; i++) km += S[(size_t)i * vd + j] * ksh[i];
    // (3) delta
    float delta = bt * (v_h[j] - km);
    // (4) rank-1 update
    for (int i = 0; i < kd; i++) S[(size_t)i * vd + j] += ksh[i] * delta;
    // (5) o = S'^T q
    float o = 0.0f;
    for (int i = 0; i < kd; i++) o += S[(size_t)i * vd + j] * qsh[i];
    out[(size_t)h * vd + j] = o;
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
                       int n_t, int kv_dim, int q_stride, float scale) {
    int h = blockIdx.x;
    if (h >= heads) return;
    int tid = threadIdx.x;
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
// (6) flat extern "C" launchers
// ---------------------------------------------------------------------------

static inline int q35_grid(int n, int block) { return (n + block - 1) / block; }

extern "C" void q35_quantize_q8(const float *x, unsigned char *out, int nblocks) {
    if (nblocks <= 0) return;
    k_quantize_q8<<<q35_grid(nblocks, Q35_BLK), Q35_BLK>>>(x, out, nblocks);
}

extern "C" void q35_matvec_q8(const unsigned char *w, const unsigned char *xq,
                              float *out, int rows, int row_blocks) {
    if (rows <= 0 || row_blocks <= 0) return;
    int warps = 8;
    k_matvec_q8<<<q35_grid(rows, warps), warps * 32>>>(w, xq, out, rows, row_blocks);
}

extern "C" void q35_matvec_f32(const float *w, const float *x, float *out,
                               int rows, int cols) {
    if (rows <= 0 || cols <= 0) return;
    int warps = 8;
    k_matvec_f32<<<q35_grid(rows, warps), warps * 32>>>(w, x, out, rows, cols);
}

extern "C" void q35_embed_row(const unsigned char *w, float *x, int row, int n) {
    if (n <= 0) return;
    k_embed_row<<<q35_grid(n, Q35_BLK), Q35_BLK>>>(w, x, row, n);
}

extern "C" void q35_rmsnorm_rows(const float *x, const float *w, float *out,
                                 int rows, int n, float eps) {
    if (rows <= 0 || n <= 0) return;
    if (n <= 128)
        k_rmsnorm_rows<128><<<rows, 128>>>(x, w, out, rows, n, eps);
    else
        k_rmsnorm_rows<256><<<rows, 256>>>(x, w, out, rows, n, eps);
}

extern "C" void q35_rmsnorm_strided(const float *x, const float *w, float *out,
                                    int rows, int n, int stride, float eps) {
    if (rows <= 0 || n <= 0) return;
    if (n <= 128)
        k_rmsnorm_strided<128><<<rows, 128>>>(x, w, out, rows, n, stride, eps);
    else
        k_rmsnorm_strided<256><<<rows, 256>>>(x, w, out, rows, n, stride, eps);
}

extern "C" void q35_l2norm(float *x, int rows, int n, float eps, float scale) {
    if (rows <= 0 || n <= 0) return;
    if (n <= 128)
        k_l2norm_rows<128><<<rows, 128>>>(x, rows, n, eps, scale);
    else
        k_l2norm_rows<256><<<rows, 256>>>(x, rows, n, eps, scale);
}

extern "C" void q35_rope(float *x, int rows, int hd, int rope_dim, int stride,
                         int pos, float base) {
    if (rows <= 0 || rope_dim <= 0) return;
    int half = rope_dim / 2;
    k_rope<<<rows, half>>>(x, rows, hd, rope_dim, stride, pos, base);
}

extern "C" void q35_conv1d(const float *wconv, float *x, float *hist,
                           int conv_dim, int k) {
    if (conv_dim <= 0 || k < 2) return;
    k_conv1d<<<q35_grid(conv_dim, Q35_BLK), Q35_BLK>>>(wconv, x, hist, conv_dim, k);
}

extern "C" void q35_beta_gamma(const float *a, const float *b, const float *abar,
                               const float *dtb, float *beta, float *gamma,
                               int nv) {
    if (nv <= 0) return;
    k_beta_gamma<<<q35_grid(nv, 128), 128>>>(a, b, abar, dtb, beta, gamma, nv);
}

extern "C" void q35_gdn_state(float *qkv, float *state, const float *beta,
                              const float *gamma, float *out, int nv, int nk,
                              int kd, int vd) {
    if (nv <= 0) return;
    if (kd > 512 || vd > 1024) {
        fprintf(stderr, "[cuda] gdn_state: kd=%d vd=%d out of range\n", kd, vd);
        exit(1);
    }
    k_gdn_state<<<nv, vd>>>(qkv, state, beta, gamma, out, nv, nk, kd, vd);
}

extern "C" void q35_gated_rmsnorm(float *out, const float *w, const float *z,
                                  int rows, int n, float eps) {
    if (rows <= 0 || n <= 0) return;
    if (n <= 128)
        k_gated_rmsnorm<128><<<rows, 128>>>(out, w, z, rows, n, eps);
    else
        k_gated_rmsnorm<256><<<rows, 256>>>(out, w, z, rows, n, eps);
}

extern "C" void q35_attn(const float *q, const float *kc, const float *vc,
                         const float *gate, float *out, int heads, int kv_heads,
                         int hd, int n_t, int kv_dim, int q_stride, float scale) {
    if (heads <= 0 || n_t <= 0) return;
    size_t shmem = (size_t)n_t * sizeof(float);
    k_attn<<<heads, 256, shmem>>>(q, kc, vc, gate, out, heads, kv_heads, hd, n_t,
                                 kv_dim, q_stride, scale);
}

extern "C" void q35_silu_mul(float *h, const float *h1, int n) {
    if (n <= 0) return;
    k_silu_mul<<<q35_grid(n, Q35_BLK), Q35_BLK>>>(h, h1, n);
}

extern "C" void q35_sigmoid_mul(float *o, const float *g, int n) {
    if (n <= 0) return;
    k_sigmoid_mul<<<q35_grid(n, Q35_BLK), Q35_BLK>>>(o, g, n);
}

extern "C" void q35_add(float *a, const float *b, int n) {
    if (n <= 0) return;
    k_add<<<q35_grid(n, Q35_BLK), Q35_BLK>>>(a, b, n);
}

extern "C" void q35_argmax(const float *x, int *outidx, int n) {
    if (n <= 0) return;
    // Single block: one global reduction over the whole vocab.  (A multi-block
    // launch would race on outidx[0] — see k_argmax.)  argmax is once/token and
    // not on the decode bottleneck path.
    k_argmax<<<1, Q35_BLK>>>(x, outidx, n);
}

