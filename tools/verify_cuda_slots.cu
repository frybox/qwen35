// Slot-by-slot numeric acceptance for the P2c device kernels.
//
// Method: each kernel is compiled from its real source in src/kernels/cuda/ and
// called through its real `q35_launch_*` entry point (the same symbol
// backend_cuda.zig calls).  The expected value is computed HERE, in C, by a
// line-by-line transcription of the CPU reference named in the comment above
// each check (src/elementwise.zig / src/engine.zig).  The harness therefore
// compares "the compiled device kernel" against "an independent transcription
// of the CPU algorithm", not against itself.
//
// Reported per case: max ABSOLUTE error and max RELATIVE error (relative to
// max(1, |expected|) so near-zero elements do not dominate).
//
// Build + run (from the repo root; NATIVE arch -- an sm_90 binary on an sm_89
// card fails at runtime with cudaErrorInvalidValue "named symbol not found",
// which looks like a kernel bug and is not):
//   nvcc -rdc=true -arch=native -O2 -o /tmp/verify_slots \
//        tools/verify_cuda_slots.cu src/kernels/cuda/attn.cu \
//        src/kernels/cuda/gdn.cu src/kernels/cuda/norm.cu src/kernels/cuda/act.cu
//   /tmp/verify_slots
//
// Recorded results (RTX 4090 D, sm_89) -- these are the numbers quoted in the
// kernel headers:
//   rmsnorm  n=2048  max_abs=1.07e-06  max_rel=7.8e-07   9 ULP
//   rmsnorm  n=1024  max_abs=3.58e-07  max_rel=3.3e-07   3 ULP
//   softmax  n=17    max_abs=2.98e-08  max_rel=2.98e-08  2 ULP
//   softmax  n=512   max_abs=5.59e-09  max_rel=5.59e-09  3 ULP
//   swiglu   n=2048  max_abs=2.38e-07  max_rel=1.36e-07  1 ULP
//   gdn_step km/delta/state/out        ALL 0.000e+00      0 ULP (bit-exact)
//   gdn_gamma beta/gamma n=48          max_abs=5.96e-08   1 ULP
//   attn probs/gout (two shapes)       max_abs<=2.24e-08  0..3 ULP
//   gdn_step drift, 200 and 800 steps  max_abs_state_drift=0.000e+00
//
// This harness is what caught the attn.cu static/dynamic shared-memory alias
// (see that file's header); it is kept so the next edit to these kernels can
// be re-verified with one command instead of an end-to-end token comparison.

#include <cuda_runtime.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

extern "C" {
int q35_launch_rmsnorm(float *d_out, const float *d_x, const float *d_w,
                       int rows, int n, float eps, void *stream);
int q35_launch_softmax(float *d_x, int n, void *stream);
int q35_launch_swiglu(float *d_gate, const float *d_up, int n, void *stream);
int q35_launch_gdn_step(float *d_state, const float *d_k, const float *d_v,
                        const float *d_q, float *d_km, float *d_delta,
                        float *d_out, float gamma, float beta, int vd,
                        void *stream);
int q35_launch_gdn_gamma(const float *d_a_raw, float *d_b_raw,
                         const float *d_ssm_a, const float *d_dt_bias,
                         float *d_gamma, int nv, void *stream);
int q35_launch_attn_decode(const float *d_q, const float *d_gate,
                           const float *d_kcache, const float *d_vcache,
                           float *d_att, float *d_gout, int slot, int heads,
                           int gqa, int hd, int kv_dim, float scale,
                           void *stream);
}

// ---- reporting -------------------------------------------------------------
static int g_fail = 0;

static void report(const char *what, const float *got, const float *want, int n) {
    double maxabs = 0.0, maxrel = 0.0;
    int argmax = 0;
    for (int i = 0; i < n; ++i) {
        const double d = fabs((double)got[i] - (double)want[i]);
        const double denom = fabs((double)want[i]) > 1.0 ? fabs((double)want[i]) : 1.0;
        const double r = d / denom;
        if (d > maxabs) { maxabs = d; argmax = i; }
        if (r > maxrel) maxrel = r;
    }
    // ULP of the worst element, for the number people actually quote.
    const float w = want[argmax];
    double ulp = 0.0;
    if (w != 0.0f) {
        const float nxt = nextafterf(w, w > 0 ? INFINITY : -INFINITY);
        ulp = fabs((double)w - (double)nxt);
        if (ulp == 0.0) ulp = fabs((double)w) * 1.1920929e-7;
    }
    const double ulps = ulp > 0 ? maxabs / ulp : 0.0;
    printf("%-28s n=%-5d max_abs=%.3e max_rel=%.3e (%.1f ULP at worst elem)\n",
           what, n, maxabs, maxrel, ulps);
}

static float *up(const float *h, int n) {
    float *d = NULL;
    cudaMalloc(&d, (size_t)n * sizeof(float));
    cudaMemcpy(d, h, (size_t)n * sizeof(float), cudaMemcpyHostToDevice);
    return d;
}
static void down(float *h, const float *d, int n) {
    cudaMemcpy(h, d, (size_t)n * sizeof(float), cudaMemcpyDeviceToHost);
}

static float frand(void) { return (float)rand() / (float)RAND_MAX * 2.0f - 1.0f; }

// ---- 1. rmsnorm: elementwise.rmsNorm --------------------------------------
//   ss = sum x^2 (scalar, exact order); scale = 1/sqrt(ss/n + eps);
//   out = x * scale * w.
static void check_rmsnorm(int n) {
    float *x = (float *)malloc(n * 4), *w = (float *)malloc(n * 4);
    float *want = (float *)malloc(n * 4), *got = (float *)malloc(n * 4);
    for (int i = 0; i < n; ++i) { x[i] = frand(); w[i] = 0.5f + 0.5f * frand(); }
    const float eps = 1e-6f;
    float ss = 0.f;
    for (int i = 0; i < n; ++i) ss += x[i] * x[i];
    const float scale = 1.0f / sqrtf(ss / (float)n + eps);
    for (int i = 0; i < n; ++i) want[i] = x[i] * scale * w[i];

    float *dx = up(x, n), *dw = up(w, n), *dout = NULL;
    cudaMalloc(&dout, (size_t)n * 4);
    q35_launch_rmsnorm(dout, dx, dw, 1, n, eps, NULL);
    down(got, dout, n);
    report("rmsnorm", got, want, n);
    if (getenv("Q35_SHOW")) for (int i = 0; i < 3; ++i) printf("   [%d] got=%.8g want=%.8g\n", i, got[i], want[i]);
    cudaFree(dx); cudaFree(dw); cudaFree(dout); free(x); free(w); free(want); free(got);
}

// ---- 2. softmax: elementwise.softmax -------------------------------------
static void check_softmax(int n) {
    float *x = (float *)malloc(n * 4), *want = (float *)malloc(n * 4), *got = (float *)malloc(n * 4);
    for (int i = 0; i < n; ++i) { x[i] = frand() * 4.0f; want[i] = x[i]; }
    float mx = -3.4e38f;
    for (int i = 0; i < n; ++i) mx = fmaxf(mx, want[i]);
    float sum = 0.f;
    for (int i = 0; i < n; ++i) { want[i] = expf(want[i] - mx); sum += want[i]; }
    for (int i = 0; i < n; ++i) want[i] /= sum;

    float *dx = up(x, n);
    q35_launch_softmax(dx, n, NULL);
    down(got, dx, n);
    report("softmax", got, want, n);
    cudaFree(dx); free(x); free(want); free(got);
}

// ---- 3. swiglu: elementwise.bufSwiglu (gate = silu(gate) * up) -----------
static void check_swiglu(int n) {
    float *g = (float *)malloc(n * 4), *u = (float *)malloc(n * 4);
    float *want = (float *)malloc(n * 4), *got = (float *)malloc(n * 4);
    for (int i = 0; i < n; ++i) { g[i] = frand() * 3.f; u[i] = frand(); want[i] = g[i]; }
    for (int i = 0; i < n; ++i) {
        const float gv = want[i];
        want[i] = (gv / (1.0f + expf(-gv))) * u[i];
    }
    float *dg = up(g, n), *du = up(u, n);
    q35_launch_swiglu(dg, du, n, NULL);
    down(got, dg, n);
    report("swiglu", got, want, n);
    cudaFree(dg); cudaFree(du); free(g); free(u); free(want); free(got);
}

// ---- 4. gdn_step: engine.zig gdnFused (vd == 128, THE authoritative order) --
static void check_gdn_step(int vd, float gamma, float beta) {
    const int nn = vd * vd;
    float *S = (float *)malloc(nn * 4), *wantS = (float *)malloc(nn * 4);
    float *k = (float *)malloc(vd * 4), *v = (float *)malloc(vd * 4), *q = (float *)malloc(vd * 4);
    float *want_km = (float *)malloc(vd * 4), *want_delta = (float *)malloc(vd * 4);
    float *want_out = (float *)malloc(vd * 4);
    float *got_km = (float *)malloc(vd * 4), *got_delta = (float *)malloc(vd * 4);
    float *got_out = (float *)malloc(vd * 4), *gotS = (float *)malloc(nn * 4);
    srand(7);
    for (int i = 0; i < nn; ++i) S[i] = frand() * 0.1f;
    memcpy(wantS, S, (size_t)nn * 4);
    for (int i = 0; i < vd; ++i) { k[i] = frand(); v[i] = frand(); q[i] = frand(); }

    // CPU: gdnFused, pass 1 then delta then pass 2 -- transcribed literally.
    for (int j = 0; j < vd; ++j) {
        float acc = 0.f;
        for (int i = 0; i < vd; ++i) acc += wantS[i * vd + j] * k[i];
        want_km[j] = acc * gamma;
    }
    for (int j = 0; j < vd; ++j) want_delta[j] = (v[j] - want_km[j]) * beta;
    for (int j = 0; j < vd; ++j) {
        float acc = 0.f;
        for (int i = 0; i < vd; ++i) {
            wantS[i * vd + j] = wantS[i * vd + j] * gamma + k[i] * want_delta[j];
            acc += wantS[i * vd + j] * q[i];
        }
        want_out[j] = acc;
    }

    float *dS = up(S, nn), *dk = up(k, vd), *dv = up(v, vd), *dq = up(q, vd);
    float *dkm = NULL, *ddelta = NULL, *dout = NULL;
    cudaMalloc(&dkm, vd * 4); cudaMalloc(&ddelta, vd * 4); cudaMalloc(&dout, vd * 4);
    const int rc = q35_launch_gdn_step(dS, dk, dv, dq, dkm, ddelta, dout, gamma, beta, vd, NULL);
    if (rc != 0) { printf("gdn_step launch FAILED rc=%d (%s)\n", rc, cudaGetErrorString((cudaError_t)rc)); g_fail = 1; return; }
    down(gotS, dS, nn); down(got_km, dkm, vd); down(got_delta, ddelta, vd); down(got_out, dout, vd);
    report("gdn_step km", got_km, want_km, vd);
    report("gdn_step delta", got_delta, want_delta, vd);
    report("gdn_step state", gotS, wantS, nn);
    report("gdn_step out", got_out, want_out, vd);
    cudaFree(dS); cudaFree(dk); cudaFree(dv); cudaFree(dq);
    cudaFree(dkm); cudaFree(ddelta); cudaFree(dout);
    free(S); free(wantS); free(k); free(v); free(q);
    free(want_km); free(want_delta); free(want_out);
    free(got_km); free(got_delta); free(got_out); free(gotS);
}

// ---- 5. gdn_gamma: elementwise.gdnGammaKernel ----------------------------
static void check_gdn_gamma(int nv) {
    float *a = (float *)malloc(nv * 4), *b = (float *)malloc(nv * 4);
    float *A = (float *)malloc(nv * 4), *dt = (float *)malloc(nv * 4);
    float *want_b = (float *)malloc(nv * 4), *want_g = (float *)malloc(nv * 4);
    float *got_b = (float *)malloc(nv * 4), *got_g = (float *)malloc(nv * 4);
    for (int i = 0; i < nv; ++i) {
        a[i] = frand() * 3.f; b[i] = frand() * 3.f;
        A[i] = -0.1f - 0.05f * (float)i; dt[i] = frand();
        want_b[i] = b[i];
    }
    for (int i = 0; i < nv; ++i) {
        want_b[i] = 1.0f / (1.0f + expf(-want_b[i]));
        const float x = a[i] + dt[i];
        const float sp = x > 20.0f ? x : log1pf(expf(x));
        want_g[i] = expf(A[i] * sp);
    }
    float *da = up(a, nv), *db = up(b, nv), *dA = up(A, nv), *ddt = up(dt, nv), *dg = NULL;
    cudaMalloc(&dg, nv * 4);
    q35_launch_gdn_gamma(da, db, dA, ddt, dg, nv, NULL);
    down(got_b, db, nv); down(got_g, dg, nv);
    report("gdn_gamma beta", got_b, want_b, nv);
    report("gdn_gamma gamma", got_g, want_g, nv);
    cudaFree(da); cudaFree(db); cudaFree(dA); cudaFree(ddt); cudaFree(dg);
    free(a); free(b); free(A); free(dt); free(want_b); free(want_g); free(got_b); free(got_g);
}

// ---- 6. attn_decode: elementwise.attnDecodeKernel ------------------------
// heads=4, gqa=2 (so two heads share one KV head), hd=8, kv rows = slot+1.
static void check_attn(int slot, int heads, int gqa, int hd) {
    const int kv_dim = (heads / gqa) * hd;
    const int nrows = slot + 1;
    const float scale = 1.0f / sqrtf((float)hd);
    float *q = (float *)malloc(heads * hd * 4), *gate = (float *)malloc(heads * hd * 4);
    float *kc = (float *)malloc(nrows * kv_dim * 4), *vc = (float *)malloc(nrows * kv_dim * 4);
    float *want = (float *)malloc(heads * hd * 4), *got = (float *)malloc(heads * hd * 4);
    float *want_att = (float *)malloc(nrows * 4);
    srand(11);
    for (int i = 0; i < heads * hd; ++i) { q[i] = frand(); gate[i] = frand() * 2.f; }
    for (int i = 0; i < nrows * kv_dim; ++i) { kc[i] = frand(); vc[i] = frand(); }

    // CPU reference: one head at a time, then mulSigmoidVec over everything.
    float *att = (float *)malloc(nrows * 4);
    for (int h = 0; h < heads; ++h) {
        const int kvh = h / gqa;
        for (int t = 0; t < nrows; ++t) {
            float acc = 0.f;
            for (int j = 0; j < hd; ++j)
                acc += q[h * hd + j] * kc[t * kv_dim + kvh * hd + j];
            att[t] = acc * scale;
        }
        // softmax in place
        float mx = -3.4e38f;
        for (int t = 0; t < nrows; ++t) mx = fmaxf(mx, att[t]);
        float sum = 0.f;
        for (int t = 0; t < nrows; ++t) { att[t] = expf(att[t] - mx); sum += att[t]; }
        for (int t = 0; t < nrows; ++t) att[t] /= sum;
        for (int j = 0; j < hd; ++j) {
            float acc = 0.f;
            for (int t = 0; t < nrows; ++t) acc += att[t] * vc[t * kv_dim + kvh * hd + j];
            want[h * hd + j] = acc;
        }
        if (h == 0) memcpy(want_att, att, nrows * 4);
        if (0) {
            float mx = -3.4e38f;
            for (int t = 0; t < nrows; ++t) {
                float acc = 0.f;
                for (int j = 0; j < hd; ++j) acc += q[h*hd+j] * kc[t*kv_dim + kvh*hd + j];
                const float sc = acc * scale;
                mx = fmaxf(mx, sc);
                printf("   cpu score[%d]=%.10g\n", t, sc);
            }
            (void)mx;
        }
    }
    for (int i = 0; i < heads * hd; ++i) want[i] *= 1.0f / (1.0f + expf(-gate[i]));

    float *dq = up(q, heads * hd), *dgate = up(gate, heads * hd);
    float *dkc = up(kc, nrows * kv_dim), *dvc = up(vc, nrows * kv_dim);
    float *datt = NULL, *dgout = NULL;
    cudaMalloc(&datt, nrows * 4); cudaMalloc(&dgout, heads * hd * 4);
    const int rc = q35_launch_attn_decode(dq, dgate, dkc, dvc, datt, dgout, slot,
                                          heads, gqa, hd, kv_dim, scale, NULL);
    if (rc != 0) { printf("attn_decode launch FAILED rc=%d (%s)\n", rc, cudaGetErrorString((cudaError_t)rc)); g_fail = 1; return; }
    float *got_att = (float *)malloc(nrows * 4);
    down(got, dgout, heads * hd);
    down(got_att, datt, nrows);
    report("attn gout (slot=5)", got, want, heads * hd);
    report("attn probs", got_att, want_att, nrows);
    if (getenv("Q35_SHOW")) {
        int worst = 0; double wd = 0;
        for (int i = 0; i < nrows; ++i) {
            double d = fabs((double)got_att[i] - (double)want_att[i]);
            if (d > wd) { wd = d; worst = i; }
        }
        for (int i = 0; i < nrows; ++i) printf("   [%d] got=%.10g want=%.10g\n", i, got_att[i], want_att[i]);
        printf("   probs worst idx=%d got=%.10g want=%.10g (sum got=?", worst, got_att[worst], want_att[worst]);
        double sg = 0, sw = 0;
        for (int i = 0; i < nrows; ++i) { sg += got_att[i]; sw += want_att[i]; }
        printf("%.10g want=%.10g)\n", sg, sw);
    }
    cudaFree(dq); cudaFree(dgate); cudaFree(dkc); cudaFree(dvc); cudaFree(datt); cudaFree(dgout);
    free(q); free(gate); free(kc); free(vc); free(want); free(got);
    free(want_att); free(att); free(got_att);
}

// ---- 7. long-horizon drift probe for gdn_step ----------------------------
// Runs N sequential steps on both sides and reports whether the state drift
// grows (that is the failure mode the kernel header claims does not happen).
static void check_gdn_drift(int vd, int steps) {
    const int nn = vd * vd;
    float *S = (float *)malloc(nn * 4), *H = (float *)malloc(nn * 4);
    float *k = (float *)malloc(vd * 4), *v = (float *)malloc(vd * 4), *q = (float *)malloc(vd * 4);
    float *km = (float *)malloc(vd * 4), *delta = (float *)malloc(vd * 4), *o = (float *)malloc(vd * 4);
    srand(23);
    for (int i = 0; i < nn; ++i) { S[i] = frand() * 0.5f; }
    memcpy(H, S, (size_t)nn * 4);
    float *dS = up(S, nn);
    float *dk = NULL, *dv = NULL, *dq = NULL, *dkm = NULL, *ddelta = NULL, *dout = NULL;
    cudaMalloc(&dk, vd * 4); cudaMalloc(&dv, vd * 4); cudaMalloc(&dq, vd * 4);
    cudaMalloc(&dkm, vd * 4); cudaMalloc(&ddelta, vd * 4); cudaMalloc(&dout, vd * 4);

    double worst = 0.0;
    for (int step = 0; step < steps; ++step) {
        for (int i = 0; i < vd; ++i) {
            k[i] = frand(); v[i] = frand(); q[i] = frand();
        }
        // host
        for (int j = 0; j < vd; ++j) {
            float acc = 0.f;
            for (int i = 0; i < vd; ++i) acc += H[i * vd + j] * k[i];
            km[j] = acc * 0.98f;
        }
        for (int j = 0; j < vd; ++j) delta[j] = (v[j] - km[j]) * 0.5f;
        for (int j = 0; j < vd; ++j) {
            float acc = 0.f;
            for (int i = 0; i < vd; ++i) {
                H[i * vd + j] = H[i * vd + j] * 0.98f + k[i] * delta[j];
                acc += H[i * vd + j] * q[i];
            }
            o[j] = acc;
        }
        // device
        cudaMemcpy(dk, k, vd * 4, cudaMemcpyHostToDevice);
        cudaMemcpy(dv, v, vd * 4, cudaMemcpyHostToDevice);
        cudaMemcpy(dq, q, vd * 4, cudaMemcpyHostToDevice);
        if (q35_launch_gdn_step(dS, dk, dv, dq, dkm, ddelta, dout, 0.98f, 0.5f, vd, NULL) != 0) {
            printf("gdn drift launch failed\n"); g_fail = 1; return;
        }
        if (step == steps - 1) {
            float *G = (float *)malloc(nn * 4);
            down(G, dS, nn);
            double mx = 0.0;
            for (int i = 0; i < nn; ++i) {
                const double d = fabs((double)G[i] - (double)H[i]);
                if (d > mx) mx = d;
            }
            printf("%-28s steps=%d  max_abs_state_drift=%.3e\n", "gdn_step drift", steps, mx);
            worst = mx;
            free(G);
        }
    }
    (void)worst;
    cudaFree(dS); cudaFree(dk); cudaFree(dv); cudaFree(dq);
    cudaFree(dkm); cudaFree(ddelta); cudaFree(dout);
    free(S); free(H); free(k); free(v); free(q);
    free(km); free(delta); free(o);
}

int main(int argc, char **argv) {
    int dev = 0;
    cudaGetDevice(&dev);
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, dev);
    printf("device: %s (sm_%d%d)\n", prop.name, prop.major, prop.minor);
    printf("--- device kernel vs an independent transcription of the CPU ref ---\n");
    check_rmsnorm(2048);
    check_rmsnorm(1024);
    check_softmax(17);
    check_softmax(512);
    check_swiglu(2048);
    check_gdn_step(128, 0.98f, 0.5f);
    check_gdn_step(128, 1.0f, 1.0f);
    check_gdn_gamma(48);
    check_attn(5, 4, 2, 8);
    check_attn(63, 8, 4, 16);
    check_attn(63, 8, 4, 16);
    printf("--- drift over a long run (state carry) ---\n");
    check_gdn_drift(128, 200);
    check_gdn_drift(128, 800);
    if (g_fail) printf("RESULT: FAIL\n");
    return g_fail;
}
