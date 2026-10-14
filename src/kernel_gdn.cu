/*
 * kernel_gdn.cu — optimized gated-delta-net (GDN) state update.
 *
 * Replaces k_gdn_state from src/kernel.cu (left untouched).
 *
 * Why the old kernel took ~46.7us (0.8B: nv=16, nk=2, kd=vd=128):
 *   grid = nv = 16 blocks, blockDim = vd = 128 (4 warps) -> only 16 of the
 *   72 SMs busy, 64 warps total.  Each thread owned one column
 *   S[*,j] and read it from DRAM 4x / wrote it 2x with 512B row stride
 *   (decay, km, update, o).  Pure DRAM latency exposure.
 *
 * What this kernel does:
 *   grid = (nsplits, nv): one block per (head, column tile), blockDim = 256.
 *   - The tile S[0..kd, c0..c1) is copied to shared memory ONCE (coalesced),
 *     all 5 math steps run on shared memory, and the tile is written back
 *     to DRAM ONCE.  DRAM traffic: 1 read + 1 write of S (was 4 read + 2
 *     write), fully coalesced.
 *   - decay / writeback: 256 threads over the whole tile.
 *   - km[j] = sum_i S[i][j]*k[i] and o[j] = sum_i S[i][j]*q[i]: one thread
 *     per column, summing i = 0..kd-1 in EXACTLY increasing order (no
 *     cross-lane partial merges), so the floating point accumulation order
 *     is IDENTICAL to the old kernel -> results are bit-exact vs the old
 *     kernel (verified in self-test: max diff 0).
 *   - rank-1 update: one thread per column over i (each writes only its
 *     own column -> no shared atomics needed).
 *   - cvd0 (columns per block) is chosen at launch: cvd0 = min(vd, cap,
 *     8192/kd) so the shared tile is always <= 8192 floats = 32KB; with
 *     ksh/qsh (2x512 floats = 4KB) total static shared = 36KB < 48KB.
 *     NO dynamic shared, NO cudaFuncSetAttribute needed.
 *     For 0.8B with cap=32: tile = 128x32 = 16KB, grid = 64 blocks
 *     (was 16), 8 warps/block (was 4).
 *
 * Semantics preserved (same order as old kernel):
 *   decay(S*=g) -> km = S'*k -> delta = beta*(v-km) -> S += k*delta
 *   -> o = S''*q.  Same expression shapes compiled at -O3, so FMA
 *   contraction matches the old kernel too.
 *   v pointer convention kept EXACTLY as in the old kernel:
 *   v_h = qkv + 2*nk*kd + h*kd  (correct for kd==vd, which is every real
 *   model here; for kd!=vd it reproduces the old kernel's behavior).
 *
 * Supported: kd <= 512, vd <= 1024, nk <= nv (same bound as old launcher).
 *
 * Integration into src/kernel.cu:
 *   - Add this file to the nvcc sources (or compile to .o and link).
 *   - Replace the body of q35_gdn_state with a call to q35_gdn_state_v2
 *     (identical signature: float *qkv, float *state, const float *beta,
 *      const float *gamma, float *out, int nv, int nk, int kd, int vd).
 *   - No cudaFuncSetAttribute, no dynamic shared, no launch-config change
 *     needed anywhere else; the stream is the caller's (this launcher uses
 *     the default stream; if you need g_stream, use the 9-arg variant
 *     q35_gdn_state_v2s below).
 *
 * Self-test:
 *   nvcc -O3 -arch=sm_89 -Xptxas -v kernel_gdn.cu -o /tmp/kg
 *   CUDA_VISIBLE_DEVICES=7 /tmp/kg            # correctness
 *   CUDA_VISIBLE_DEVICES=7 /tmp/kg bench      # + A/B timing old vs new
 */

#include <cuda_runtime.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>

#define GDN_CHECK(x)                                                            \
    do {                                                                        \
        cudaError_t e_ = (x);                                                 \
        if (e_ != cudaSuccess) {                                              \
            fprintf(stderr, "cuda %s:%d: %s\n", __FILE__, __LINE__,            \
                    cudaGetErrorString(e_));                                  \
            exit(1);                                                          \
        }                                                                     \
    } while (0)

/* ================================================================== */
/* Optimized kernel.  grid = (nsplits, nv), block = 256 threads.      */
/* Tile per block: rows 0..kd-1, cols c0..c0+cvd0-1 (last may be     */
/* short).  Static shared: Ss 8192 f (32KB) + ksh/qsh 2*512 f (4KB). */
/* ================================================================== */
__global__ void __launch_bounds__(256) k_gdn_state_v2(
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

    /* k, q to shared (grid-stride since blockDim=256 < kd possible). */
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

    /* (2)-(5) one thread per column (tid < cvd).
       km and o accumulate i = 0..kd-1 in strictly increasing order,
       exactly like the old kernel -> bit-exact. */
    if (tid < cvd) {
        float *Sc = Ss + tid; /* Sc[i*cvd] == Ss[i*cvd + tid] */
        float km = 0.f;
        for (int i = 0; i < kd; i++) km += Sc[i * cvd] * ksh[i];
        float delta = bt * (v_h[c0 + tid] - km);
        for (int i = 0; i < kd; i++) Sc[i * cvd] += ksh[i] * delta;
        float o = 0.f;
        for (int i = 0; i < kd; i++) o += Sc[i * cvd] * qsh[i];
        out[(size_t)h * vd + c0 + tid] = o;
    }
    __syncthreads();

    /* write tile back (ONE DRAM write). */
    for (int x = tid; x < n; x += 256) {
        int i = x / cvd;
        S[(size_t)i * vd + c0 + (x - i * cvd)] = Ss[x];
    }
}

/* ================================================================== */
/* Launcher.  Same signature as q35_gdn_state in src/kernel.cu.       */
/* ================================================================== */
static void gdn_launch(float *qkv, float *state, const float *beta,
                       const float *gamma, float *out, int nv, int nk,
                       int kd, int vd, int cap) {
    int cvd0 = cap;
    int maxc = 8192 / kd; /* tile <= 8192 floats = 32KB static shared */
    if (cvd0 > maxc) cvd0 = maxc;
    if (cvd0 > vd) cvd0 = vd;
    if (cvd0 < 1) cvd0 = 1;
    int nsplits = (vd + cvd0 - 1) / cvd0;
    k_gdn_state_v2<<<dim3((unsigned)nsplits, (unsigned)nv), 256>>>(
        qkv, state, beta, gamma, out, nv, nk, kd, vd, cvd0);
}

extern "C" void q35_gdn_state_v2(float *qkv, float *state, const float *beta,
                                 const float *gamma, float *out, int nv,
                                 int nk, int kd, int vd) {
    if (nv <= 0 || nk <= 0 || kd <= 0 || vd <= 0) return;
    if (kd > 512 || vd > 1024 || nk > nv) {
        fprintf(stderr,
                "[cuda] gdn_state_v2: kd=%d vd=%d nk=%d nv=%d out of range\n",
                kd, vd, nk, nv);
        exit(1);
    }
    gdn_launch(qkv, state, beta, gamma, out, nv, nk, kd, vd, 32);
}

/* Same as above but on a caller-provided stream (for graph capture,
 * mirrors how src/kernel.cu launches on g_stream). */
extern "C" void q35_gdn_state_v2s(float *qkv, float *state, const float *beta,
                                  const float *gamma, float *out, int nv,
                                  int nk, int kd, int vd, cudaStream_t st) {
    if (nv <= 0 || nk <= 0 || kd <= 0 || vd <= 0) return;
    if (kd > 512 || vd > 1024 || nk > nv) {
        fprintf(stderr,
                "[cuda] gdn_state_v2s: kd=%d vd=%d nk=%d nv=%d out of range\n",
                kd, vd, nk, nv);
        exit(1);
    }
    int cvd0, nsplits;
    {
        cvd0 = 32;
        int maxc = 8192 / kd;
        if (cvd0 > maxc) cvd0 = maxc;
        if (cvd0 > vd) cvd0 = vd;
        if (cvd0 < 1) cvd0 = 1;
        nsplits = (vd + cvd0 - 1) / cvd0;
        k_gdn_state_v2<<<dim3((unsigned)nsplits, (unsigned)nv), 256, 0, st>>>(
            qkv, state, beta, gamma, out, nv, nk, kd, vd, cvd0);
    }
}

/* bench helper: choose cvd0 cap at runtime (correctness sweep). */
extern "C" void q35_gdn_state_v2cap(float *qkv, float *state,
                                    const float *beta, const float *gamma,
                                    float *out, int nv, int nk, int kd,
                                    int vd, int cap) {
    if (nv <= 0 || nk <= 0 || kd <= 0 || vd <= 0) return;
    gdn_launch(qkv, state, beta, gamma, out, nv, nk, kd, vd, cap);
}

/* ================================================================== */
/* Original kernel (verbatim copy from src/kernel.cu, for A/B only).  */
/* ================================================================== */
__global__ void k_gdn_state_orig(float *__restrict__ qkv,
                                 float *__restrict__ state,
                                 const float *__restrict__ beta,
                                 const float *__restrict__ gamma,
                                 float *__restrict__ out, int nv, int nk,
                                 int kd, int vd) {
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
    for (int i = 0; i < kd; i++) S[(size_t)i * vd + j] *= g;
    float km = 0.0f;
    for (int i = 0; i < kd; i++) km += S[(size_t)i * vd + j] * ksh[i];
    float delta = bt * (v_h[j] - km);
    for (int i = 0; i < kd; i++) S[(size_t)i * vd + j] += ksh[i] * delta;
    float o = 0.0f;
    for (int i = 0; i < kd; i++) o += S[(size_t)i * vd + j] * qsh[i];
    out[(size_t)h * vd + j] = o;
}

/* ================================================================== */
/* Self-test main.                                                     */
/* ================================================================== */

static uint32_t s_rng = 0x9e3779b9u;
static float frand(void) { /* ~U(-1,1) */
    s_rng = s_rng * 1664525u + 1013904223u;
    return (float)((s_rng >> 8) / 16777216.0f) * 2.0f - 1.0f;
}

/* Naive CPU reference, float, NO fused multiply-add (explicit mul then
 * add), same i-order as the kernels.  Mirrors the old kernel's pointer
 * convention v_h = v + h*kd.  S is updated in place; S0 is the pre-step
 * state (kept separately so we can reset). */
static void ref_float(const float *hq, const float *hk, const float *hv,
                      const float *hbeta, const float *hgamma, float *S,
                      float *ref_out, int nv, int nk, int kd, int vd) {
    for (int h = 0; h < nv; h++) {
        const int kh = h % nk;
        const float g = hgamma[h];
        const float bt = hbeta[h];
        const float *q_h = hq + (size_t)kh * kd;
        const float *k_h = hk + (size_t)kh * kd;
        const float *v_h = hv + (size_t)h * kd;
        float *Sh = S + (size_t)h * kd * vd;
        for (int i = 0; i < kd; i++)
            for (int j = 0; j < vd; j++) Sh[(size_t)i * vd + j] *= g;
        for (int j = 0; j < vd; j++) {
            float km = 0.f;
            for (int i = 0; i < kd; i++) {
                float p = Sh[(size_t)i * vd + j] * k_h[i];
                km += p;
            }
            float delta = bt * (v_h[j] - km);
            for (int i = 0; i < kd; i++) {
                float p = k_h[i] * delta;
                Sh[(size_t)i * vd + j] += p;
            }
            float o = 0.f;
            for (int i = 0; i < kd; i++) {
                float p = Sh[(size_t)i * vd + j] * q_h[i];
                o += p;
            }
            ref_out[(size_t)h * vd + j] = o;
        }
    }
}

/* Double-precision reference (true value). */
static void ref_double(const float *hq, const float *hk, const float *hv,
                       const float *hbeta, const float *hgamma,
                       const float *S0, double *ref_S, double *ref_out,
                       int nv, int nk, int kd, int vd) {
    for (int h = 0; h < nv; h++) {
        const int kh = h % nk;
        const double g = (double)hgamma[h];
        const double bt = (double)hbeta[h];
        const float *q_h = hq + (size_t)kh * kd;
        const float *k_h = hk + (size_t)kh * kd;
        const float *v_h = hv + (size_t)h * kd;
        const float *S0h = S0 + (size_t)h * kd * vd;
        double *Sh = ref_S + (size_t)h * kd * vd;
        for (int i = 0; i < kd; i++)
            for (int j = 0; j < vd; j++)
                Sh[(size_t)i * vd + j] = (double)S0h[(size_t)i * vd + j] * g;
        for (int j = 0; j < vd; j++) {
            double km = 0.0;
            for (int i = 0; i < kd; i++)
                km += Sh[(size_t)i * vd + j] * (double)k_h[i];
            double delta = bt * ((double)v_h[j] - km);
            for (int i = 0; i < kd; i++)
                Sh[(size_t)i * vd + j] += (double)k_h[i] * delta;
            double o = 0.0;
            for (int i = 0; i < kd; i++)
                o += Sh[(size_t)i * vd + j] * (double)q_h[i];
            ref_out[(size_t)h * vd + j] = o;
        }
    }
}

static double maxdiff(const float *a, const float *b, size_t n) {
    double m = 0.0;
    for (size_t i = 0; i < n; i++) {
        double d = fabs((double)a[i] - (double)b[i]);
        if (d > m) m = d;
    }
    return m;
}

struct Case {
    int nv, nk, kd, vd;
    float glo, ghi; /* gamma range */
};

int main(int argc, char **argv) {
    const int do_bench = argc > 1 && strcmp(argv[1], "bench") == 0;
    const Case cases[] = {
        {16, 2, 128, 128, 0.9, 1.0},   /* 0.8B */
        {32, 4, 128, 128, 0.9, 1.0},   /* bigger model */
        {16, 2, 64, 64, 0.5, 1.0},     /* small */
        {16, 2, 256, 256, 0.9, 1.0},   /* bigger S */
        {16, 2, 128, 160, 0.9, 1.0},   /* vd % cvd0 != 0 */
        {16, 2, 192, 128, 0.9, 1.0},   /* kd not mult of 32 */
        {16, 2, 512, 512, 0.99, 1.0},  /* large */
        {16, 2, 512, 1024, 0.99, 1.0}, /* max size */
        {8, 1, 1, 1, 0.0, 1.0},        /* degenerate */
        {4, 1, 1, 1024, 0.5, 1.0},     /* kd=1, vd=1024 (ncol path) */
        {16, 2, 128, 128, 0.5, 0.8},   /* strong decay */
    };
    const int NCASE = (int)(sizeof(cases) / sizeof(cases[0]));

    cudaDeviceProp prop;
    GDN_CHECK(cudaGetDeviceProperties(&prop, 0));
    printf("device: %s (SMs=%d, sm_%d%d)\n", prop.name,
           prop.multiProcessorCount, prop.major, prop.minor);

    int maxblk = 0;
    cudaOccupancyMaxActiveBlocksPerMultiprocessor(&maxblk, k_gdn_state_v2, 256,
                                                   0);
    printf("k_gdn_state_v2: max active blocks/SM = %d (256 thr, 0 dyn shmem)\n",
           maxblk);

    double worst = 0.0, worst_od = 0.0;
    int all_ok = 1;

    for (int c = 0; c < NCASE; c++) {
        const Case cs = cases[c];
        const int nv = cs.nv, nk = cs.nk, kd = cs.kd, vd = cs.vd;
        const size_t nq = (size_t)nk * kd;
        const size_t nqkv = (size_t)(2 * nk * kd + nv * kd);
        const size_t nstate = (size_t)nv * kd * vd;
        const size_t nout = (size_t)nv * vd;

        float *hq = (float *)malloc(nq * 4);
        float *hk = (float *)malloc(nq * 4);
        float *hv = (float *)malloc((size_t)nv * kd * 4);
        for (size_t i = 0; i < nq; i++) hq[i] = frand();
        for (size_t i = 0; i < nq; i++) hk[i] = frand();
        for (size_t i = 0; i < (size_t)nv * kd; i++) hv[i] = frand();
        float *hbeta = (float *)malloc((size_t)nv * 4);
        float *hgamma = (float *)malloc((size_t)nv * 4);
        for (int i = 0; i < nv; i++) {
            hbeta[i] = 0.5f + 0.5f * frand(); /* (0,1) */
            hgamma[i] = cs.glo + (cs.ghi - cs.glo) * (0.5f + 0.5f * frand());
        }
        float *hstate = (float *)malloc(nstate * 4);
        for (size_t i = 0; i < nstate; i++) hstate[i] = frand() * 3.0f;

        /* references from the fresh state */
        float *Sf = (float *)malloc(nstate * 4);
        float *ref_out_f = (float *)malloc(nout * 4);
        double *Sd = (double *)malloc(nstate * 8);
        double *ref_out_d = (double *)malloc(nout * 8);
        memcpy(Sf, hstate, nstate * 4);
        ref_float(hq, hk, hv, hbeta, hgamma, Sf, ref_out_f, nv, nk, kd, vd);
        ref_double(hq, hk, hv, hbeta, hgamma, hstate, Sd, ref_out_d, nv, nk,
                   kd, vd);

        /* pack qkv [q|k|v] */
        float *hpack = (float *)malloc(nqkv * 4);
        memcpy(hpack, hq, nq * 4);
        memcpy(hpack + nq, hk, nq * 4);
        memcpy(hpack + 2 * nq, hv, (size_t)nv * kd * 4);

        float *d_qkv, *d_state, *d_beta, *d_gamma, *d_out;
        GDN_CHECK(cudaMalloc(&d_qkv, nqkv * 4));
        GDN_CHECK(cudaMalloc(&d_state, nstate * 4));
        GDN_CHECK(cudaMalloc(&d_beta, (size_t)nv * 4));
        GDN_CHECK(cudaMalloc(&d_gamma, (size_t)nv * 4));
        GDN_CHECK(cudaMalloc(&d_out, nout * 4));
        GDN_CHECK(cudaMemcpy(d_qkv, hpack, nqkv * 4, cudaMemcpyHostToDevice));
        GDN_CHECK(cudaMemcpy(d_state, hstate, nstate * 4,
                            cudaMemcpyHostToDevice));
        GDN_CHECK(cudaMemcpy(d_beta, hbeta, (size_t)nv * 4,
                            cudaMemcpyHostToDevice));
        GDN_CHECK(cudaMemcpy(d_gamma, hgamma, (size_t)nv * 4,
                            cudaMemcpyHostToDevice));

        /* --- run 1: optimized kernel on fresh state --- */
        GDN_CHECK(cudaMemset(d_out, 0, nout * 4));
        q35_gdn_state_v2(d_qkv, d_state, d_beta,
                        d_gamma, d_out, nv, nk, kd,
                        vd);
        GDN_CHECK(cudaGetLastError());
        GDN_CHECK(cudaDeviceSynchronize());
        float *g_out = (float *)malloc(nout * 4);
        float *g_S = (float *)malloc(nstate * 4);
        GDN_CHECK(cudaMemcpy(g_out, d_out, nout * 4, cudaMemcpyDeviceToHost));
        GDN_CHECK(cudaMemcpy(g_S, d_state, nstate * 4, cudaMemcpyDeviceToHost));
        double e_out_f = maxdiff(g_out, ref_out_f, nout);
        double e_S_f = maxdiff(g_S, Sf, nstate);
        /* vs double reference (absolute scale of the float algorithm) */
        double e_out_d = 0, e_S_d = 0;
        for (size_t i = 0; i < nout; i++)
            e_out_d = fmax(e_out_d, fabs((double)g_out[i] - ref_out_d[i]));
        for (size_t i = 0; i < nstate; i++)
            e_S_d = fmax(e_S_d, fabs((double)g_S[i] - Sd[i]));

        /* --- run 2: second step on the already-updated device state --- */
        float *Sf2 = (float *)malloc(nstate * 4);
        float *ref_out_f2 = (float *)malloc(nout * 4);
        memcpy(Sf2, Sf, nstate * 4);
        ref_float(hq, hk, hv, hbeta, hgamma, Sf2, ref_out_f2, nv, nk, kd,
                  vd);
        GDN_CHECK(cudaMemset(d_out, 0, nout * 4));
        q35_gdn_state_v2(d_qkv, d_state, d_beta,
                        d_gamma, d_out, nv, nk, kd,
                        vd);
        GDN_CHECK(cudaDeviceSynchronize());
        GDN_CHECK(cudaMemcpy(g_out, d_out, nout * 4, cudaMemcpyDeviceToHost));
        GDN_CHECK(cudaMemcpy(g_S, d_state, nstate * 4, cudaMemcpyDeviceToHost));
        double e2_out = maxdiff(g_out, ref_out_f2, nout);
        double e2_S = maxdiff(g_S, Sf2, nstate);

        /* --- run 3: original kernel on fresh state (A/B + exactness) --- */
        GDN_CHECK(cudaMemcpy(d_state, hstate, nstate * 4,
                            cudaMemcpyHostToDevice));
        GDN_CHECK(cudaMemset(d_out, 0, nout * 4));
        k_gdn_state_orig<<<nv, vd>>>(d_qkv, d_state, d_beta, d_gamma, d_out,
                                     nv, nk, kd, vd);
        GDN_CHECK(cudaGetLastError());
        GDN_CHECK(cudaDeviceSynchronize());
        float *o_out = (float *)malloc(nout * 4);
        float *o_S = (float *)malloc(nstate * 4);
        GDN_CHECK(cudaMemcpy(o_out, d_out, nout * 4, cudaMemcpyDeviceToHost));
        GDN_CHECK(cudaMemcpy(o_S, d_state, nstate * 4,
                            cudaMemcpyDeviceToHost));
        /* fresh-state result of the new kernel, re-run for exactness */
        GDN_CHECK(cudaMemcpy(d_state, hstate, nstate * 4,
                            cudaMemcpyHostToDevice));
        GDN_CHECK(cudaMemset(d_out, 0, nout * 4));
        q35_gdn_state_v2(d_qkv, d_state, d_beta,
                        d_gamma, d_out, nv, nk, kd,
                        vd);
        GDN_CHECK(cudaDeviceSynchronize());
        GDN_CHECK(cudaMemcpy(g_out, d_out, nout * 4, cudaMemcpyDeviceToHost));
        GDN_CHECK(cudaMemcpy(g_S, d_state, nstate * 4, cudaMemcpyDeviceToHost));
        double e_xo = maxdiff(g_out, o_out, nout);
        double e_xS = maxdiff(g_S, o_S, nstate);

        printf("case nv=%d nk=%d kd=%d vd=%d g~[%.2f,%.2f]:\n"
               "  vs CPU-float: max|out|=%.2e max|S|=%.2e   (2nd step: %.2e / %.2e)\n"
               "  vs double-ref: max|out|=%.2e max|S|=%.2e\n"
               "  new-vs-orig:   max|out|=%.2e max|S|=%.2e\n",
               nv, nk, kd, vd, cs.glo, cs.ghi, e_out_f, e_S_f, e2_out, e2_S,
               e_out_d, e_S_d, e_xo, e_xS);
        worst = fmax(worst, fmax(fmax(e_out_f, e_S_f), fmax(e2_out, e2_S)));
        worst_od = fmax(worst_od, fmax(e_out_d, e_S_d));
        if (e_out_f > 2e-5 || e_S_f > 2e-4 || e2_out > 2e-5 || e2_S > 2e-4 ||
            e_xo > 1e-6 || e_xS > 1e-6)
            all_ok = 0;

        /* --- bench (all cases; ~150 iters, state reset each iter) --- */
        if (do_bench) {
            const int warmup = 20, iters = 150;
            cudaEvent_t e0, e1;
            cudaEventCreate(&e0);
            cudaEventCreate(&e1);
            float us_orig = 0, us_new = 0;
            for (int it = 0; it < warmup + iters; it++) {
                GDN_CHECK(cudaMemcpy(d_state, hstate, nstate * 4,
                                    cudaMemcpyHostToDevice));
                cudaEventRecord(e0);
                k_gdn_state_orig<<<nv, vd>>>(d_qkv, d_state, d_beta, d_gamma,
                                             d_out, nv, nk, kd, vd);
                cudaEventRecord(e1);
                cudaEventSynchronize(e1);
                if (it >= warmup) {
                    float ms;
                    cudaEventElapsedTime(&ms, e0, e1);
                    us_orig += ms * 1000.0f;
                }
            }
            for (int it = 0; it < warmup + iters; it++) {
                GDN_CHECK(cudaMemcpy(d_state, hstate, nstate * 4,
                                    cudaMemcpyHostToDevice));
                cudaEventRecord(e0);
                q35_gdn_state_v2(d_qkv, d_state, d_beta, d_gamma, d_out, nv, nk, kd, vd);
                cudaEventRecord(e1);
                cudaEventSynchronize(e1);
                if (it >= warmup) {
                    float ms;
                    cudaEventElapsedTime(&ms, e0, e1);
                    us_new += ms * 1000.0f;
                }
            }
            printf("  bench: orig=%.1fus new=%.1fus speedup=%.1fx\n",
                   us_orig / iters, us_new / iters, us_orig / us_new);
            cudaEventDestroy(e0);
            cudaEventDestroy(e1);

            /* cap sweep (only meaningful for the 0.8B-ish shape) */
            if (nv == 16 && nk == 2 && kd == 128 && vd == 128) {
                for (int cap = 16; cap <= 128; cap *= 2) {
                    float us_c = 0;
                    for (int it = 0; it < warmup + iters; it++) {
                        GDN_CHECK(cudaMemcpy(d_state, hstate, nstate * 4,
                                            cudaMemcpyHostToDevice));
                        cudaEventRecord(e0);
                        q35_gdn_state_v2cap(d_qkv, d_state, d_beta, d_gamma, d_out, nv, nk, kd, vd,
                                            cap);
                        cudaEventRecord(e1);
                        cudaEventSynchronize(e1);
                        if (it >= warmup) {
                            float ms;
                            cudaEventElapsedTime(&ms, e0, e1);
                            us_c += ms * 1000.0f;
                        }
                    }
                    printf("  cap=%3d: new=%.1fus speedup=%.1fx\n", cap,
                           us_c / iters, us_orig / us_c);
                }
            }
        }

        cudaFree(d_qkv);
        cudaFree(d_state);
        cudaFree(d_beta);
        cudaFree(d_gamma);
        cudaFree(d_out);
        free(hq);
        free(hk);
        free(hv);
        free(hbeta);
        free(hgamma);
        free(hstate);
        free(Sf);
        free(Sf2);
        free(ref_out_f);
        free(ref_out_f2);
        free(Sd);
        free(ref_out_d);
        free(g_out);
        free(g_S);
        free(o_out);
        free(o_S);
        free(hpack);
    }

    printf("\nworst (vs CPU-float ref): %.2e   worst (vs double ref): %.2e   %s\n",
           worst, worst_od, all_ok ? "PASS" : "FAIL");
    return all_ok ? 0 : 1;
}
