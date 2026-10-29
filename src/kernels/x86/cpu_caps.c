/**
 * @file cpu_caps.c
 * @brief x86 CPU feature probes — single source of truth for src/kernels/x86.
 *
 * Features are probed ONCE in a constructor at load time. The probes below are
 * pure reads of cached ints, safe to call from a kernel dispatcher on every
 * dot-product call (cpuid/xgetbv cost ~1600 ns per call on a KVM guest).
 *
 * Do not add a second probe implementation anywhere in src/. Zig callers use
 * c_kernels.cpuHasAvx2() etc., which forward here.
 */

#include "cpu_caps.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* DEBUG ONLY: QWEN35_FORCE_SCALAR=<comma-separated probe names> forces those
 * probes to 0, so a kernel dispatcher takes its scalar (oracle) path.  Used to
 * A/B "is the garbage coming from the SIMD kernel or from the engine?".
 * Names: avx2 avx512f avx512bw avx512vl avx512dq vnni vbmi fma f16c ssse3
 */
static int force_mask_avx2        = 0;
static int force_mask_avx512f     = 0;
static int force_mask_avx512bw    = 0;
static int force_mask_avx512vl    = 0;
static int force_mask_avx512dq    = 0;
static int force_mask_vnni        = 0;
static int force_mask_vbmi        = 0;
static int force_mask_fma         = 0;
static int force_mask_f16c        = 0;
static int force_mask_ssse3       = 0;

static void cpu_caps_apply_env(void) {
    const char* e = getenv("QWEN35_FORCE_SCALAR");
    if (!e || !*e) return;
    if (strstr(e, "avx2"))     force_mask_avx2    = 1;
    if (strstr(e, "avx512f"))  force_mask_avx512f = 1;
    if (strstr(e, "avx512bw")) force_mask_avx512bw = 1;
    if (strstr(e, "avx512vl")) force_mask_avx512vl = 1;
    if (strstr(e, "avx512dq")) force_mask_avx512dq = 1;
    if (strstr(e, "avx512"))   force_mask_avx512f = force_mask_avx512bw = force_mask_avx512vl = force_mask_avx512dq = 1;
    if (strstr(e, "vnni"))     force_mask_vnni    = 1;
    if (strstr(e, "vbmi"))     force_mask_vbmi    = 1;
    if (strstr(e, "fma"))      force_mask_fma     = 1;
    if (strstr(e, "f16c"))     force_mask_f16c    = 1;
    if (strstr(e, "ssse3"))    force_mask_ssse3   = 1;
    if (strstr(e, "all"))      force_mask_avx2 = force_mask_avx512f = force_mask_avx512bw =
                              force_mask_avx512vl = force_mask_avx512dq = force_mask_vnni =
                              force_mask_vbmi = force_mask_fma = force_mask_f16c = force_mask_ssse3 = 1;
}

/* CPUID wrapper for x86-64 (inline asm, no GCC built-ins) */
static inline void cpuid(int eax, int ecx, int* regs) {
    __asm__ __volatile__ (
        "cpuid"
        : "=a"(regs[0]), "=b"(regs[1]), "=c"(regs[2]), "=d"(regs[3])
        : "a"(eax), "c"(ecx)
    );
}

/* XGETBV wrapper */
static inline unsigned long long xgetbv(int index) {
    unsigned int eax, edx;
    __asm__ __volatile__ (
        "xgetbv"
        : "=a"(eax), "=d"(edx)
        : "c"(index)
    );
    return ((unsigned long long)edx << 32) | eax;
}

/* Cached feature flags, filled once by cpu_caps_init(). -1 = not probed yet. */
static int avx2_cached        = -1;
static int avx512f_cached     = -1;
static int avx512bw_cached    = -1;
static int avx512vl_cached    = -1;
static int avx512dq_cached    = -1;
static int avx512_vnni_cached = -1;
static int avx512_vbmi_cached = -1;
static int fma_cached         = -1;
static int f16c_cached        = -1;
static int ssse3_cached       = -1;

/* One CPUID leaf-7 probe fills every leaf-7 feature at once. */
__attribute__((constructor)) static void cpu_caps_init(void) {
    int regs[4];
    cpuid(7, 0, regs);
    const int ebx = regs[1], ecx = regs[2];

    /* OS must enable the vector state: OSXSAVE (leaf 1, ECX bit 27) + XCR0. */
    int one[4];
    cpuid(1, 0, one);
    int osxsave = (one[2] >> 27) & 1;
    unsigned long long xcr0 = osxsave ? xgetbv(0) : 0;

    /* XMM(1)+YMM(2) for AVX/AVX2; opmask(5)+ZMM_hi256(6)+Hi16_ZMM(7) for AVX-512. */
    const int ymm_ok = ((xcr0 >> 1) & 1) && ((xcr0 >> 2) & 1);
    const int zmm_ok = ((xcr0 >> 5) & 1) && ((xcr0 >> 6) & 1) && ((xcr0 >> 7) & 1);

    cpu_caps_apply_env();

    avx2_cached        = osxsave && ymm_ok && ((ebx >> 5)  & 1);
    avx512f_cached     = osxsave && zmm_ok && ((ebx >> 16) & 1);
    avx512dq_cached    = avx512f_cached && ((ebx >> 17) & 1);   /* leaf7 EBX bit17 */
    avx512bw_cached    = avx512f_cached && ((ebx >> 30) & 1);
    avx512vl_cached    = avx512f_cached && ((ebx >> 31) & 1);
    avx512_vnni_cached = avx512f_cached && ((ecx >> 11) & 1);   /* leaf7 ECX bit11 */
    avx512_vbmi_cached = avx512f_cached && ((ecx >> 24) & 1);   /* leaf7 ECX bit24 */

    /* Leaf 1 features (ECX): SSSE3 bit9, FMA bit12, AVX bit28, F16C bit29.
     * FMA/F16C need the YMM state too (they are AVX-class instructions). */
    ssse3_cached = (one[2] >> 9)  & 1;
    fma_cached   = ymm_ok && ((one[2] >> 12) & 1);
    f16c_cached  = ymm_ok && ((one[2] >> 29) & 1);

    /* Apply QWEN35_FORCE_SCALAR LAST, after every bit has been probed.
     * This ran here-but-never-actually-happened before: cpu_caps_apply_env()
     * only *parsed* the env var into the force_mask_* globals and nothing ever
     * read them back, so the documented debug switch was a silent no-op and
     * every "force scalar" A/B compared a kernel against itself. Clearing a
     * cached flag here (rather than at probe time) also keeps the mask from
     * disabling a feature before its dependency chain is resolved. */
    if (force_mask_avx2)        avx2_cached        = 0;
    if (force_mask_avx512f)     avx512f_cached     = 0;
    if (force_mask_avx512bw)    avx512bw_cached    = 0;
    if (force_mask_avx512vl)    avx512vl_cached    = 0;
    if (force_mask_avx512dq)    avx512dq_cached    = 0;
    if (force_mask_vnni)        avx512_vnni_cached = 0;
    if (force_mask_vbmi)        avx512_vbmi_cached = 0;
    if (force_mask_fma)         fma_cached         = 0;
    if (force_mask_f16c)        f16c_cached        = 0;
    if (force_mask_ssse3)       ssse3_cached       = 0;

    /* AVX-512 features are only reachable when the 512-bit state is on; keep
     * the dependency invariant the probe built (avx512*_cached implies
     * avx512f_cached) so a mask on avx512f cannot leave VNNI claiming support. */
    if (!avx512f_cached) {
        avx512bw_cached = 0;
        avx512vl_cached = 0;
        avx512dq_cached = 0;
        avx512_vnni_cached = 0;
        avx512_vbmi_cached = 0;
    }
}

int cpu_has_avx2(void)        { return avx2_cached < 0 ? (cpu_caps_init(), avx2_cached)        : avx2_cached; }
int cpu_has_avx512f(void)     { return avx512f_cached < 0 ? (cpu_caps_init(), avx512f_cached)  : avx512f_cached; }
int cpu_has_avx512bw(void)    { return avx512bw_cached < 0 ? (cpu_caps_init(), avx512bw_cached) : avx512bw_cached; }
int cpu_has_avx512vl(void)    { return avx512vl_cached < 0 ? (cpu_caps_init(), avx512vl_cached) : avx512vl_cached; }
int cpu_has_avx512dq(void)    { return avx512dq_cached < 0 ? (cpu_caps_init(), avx512dq_cached) : avx512dq_cached; }
int cpu_has_avx512_vnni(void) { return avx512_vnni_cached < 0 ? (cpu_caps_init(), avx512_vnni_cached) : avx512_vnni_cached; }
int cpu_has_avx512_vbmi(void) { return avx512_vbmi_cached < 0 ? (cpu_caps_init(), avx512_vbmi_cached) : avx512_vbmi_cached; }
int cpu_has_fma(void)         { return fma_cached < 0 ? (cpu_caps_init(), fma_cached) : fma_cached; }
int cpu_has_f16c(void)        { return f16c_cached < 0 ? (cpu_caps_init(), f16c_cached) : f16c_cached; }
int cpu_has_ssse3(void)       { return ssse3_cached < 0 ? (cpu_caps_init(), ssse3_cached) : ssse3_cached; }

const char* cpu_get_features_string(void) {
    static char buf[128];
    snprintf(buf, sizeof(buf), "AVX2=%d AVX512F=%d AVX512BW=%d AVX512VL=%d AVX512DQ=%d AVX512_VNNI=%d AVX512_VBMI=%d",
             cpu_has_avx2(), cpu_has_avx512f(), cpu_has_avx512bw(), cpu_has_avx512vl(),
             cpu_has_avx512dq(), cpu_has_avx512_vnni(), cpu_has_avx512_vbmi());
    return buf;
}
