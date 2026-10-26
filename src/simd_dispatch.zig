//! Runtime ISA dispatch for the ported quantization kernels.
//!
//! This version uses only C kernels (src/c_kernels/x86/), no Zig SIMD.

const std = @import("std");
const api = @import("simd_api.zig");
const c_kernels = @import("c_kernels.zig");

pub const Tier = enum { baseline, avx2, avx512, avx512vnni, avx512vbmi };

pub var tier: Tier = .baseline;

/// TEMP(debug): force a specific tier, bypassing CPUID detection.
/// Set from main.zig via QWEN35_SIMD_TIER (baseline|avx2|avx512).
pub var force_tier: ?Tier = null;

// ---------------------------------------------------------------------------
// CPUID / XGETBV
// ---------------------------------------------------------------------------

inline fn cpuid(eax_in: u32, ecx_in: u32) [4]u32 {
    var eax: u32 = eax_in;
    var ebx: u32 = 0;
    var ecx: u32 = ecx_in;
    var edx: u32 = 0;
    asm volatile ("cpuid"
        : [eax] "+{eax}" (eax),
          [ebx] "={ebx}" (ebx),
          [ecx] "+{ecx}" (ecx),
          [edx] "={edx}" (edx),
    );
    return .{ eax, ebx, ecx, edx };
}

inline fn xgetbv(ecx_in: u32) u32 {
    return asm volatile ("xgetbv"
        : [a] "={eax}" (-> u32),
        : [c] "{ecx}" (ecx_in),
    );
}

inline fn hasOsxsave() bool {
    return cpuid(1, 0)[2] & (1 << 27) != 0;
}

/// XCR0 bits the OS has enabled: x87(0) SSE(1) YMM(2) opmask(5) ZMM_hi256(6)
/// Hi16_ZMM(7).  AVX2 needs 0x7, AVX-512 needs 0xE6, both plus x87/SSE.
inline fn xcr0Ok(mask: u32) bool {
    if (!hasOsxsave()) return false;
    const xcr0 = xgetbv(0);
    return (xcr0 & 0x6) == 0x6 and (xcr0 & mask) == mask;
}

pub const Features = struct {
    avx2: bool = false,
    avx512f: bool = false,
    avx512bw: bool = false,
    avx512vl: bool = false,
    avx512dq: bool = false,
    avx512vnni: bool = false,
    avx512vbmi: bool = false,
    f16c: bool = false,
    fma: bool = false,
    ssse3: bool = false,
};

pub fn detect() Features {
    var f: Features = .{};
    const c1 = cpuid(1, 0);
    f.ssse3 = c1[2] & (1 << 9) != 0;
    f.fma = c1[2] & (1 << 12) != 0;
    const avx = c1[2] & (1 << 28) != 0;
    f.f16c = c1[2] & (1 << 29) != 0;

    const c7 = cpuid(7, 0);
    f.avx2 = c7[1] & (1 << 5) != 0;
    f.avx512f = c7[1] & (1 << 16) != 0;
    f.avx512dq = c7[1] & (1 << 17) != 0;
    f.avx512bw = c7[1] & (1 << 30) != 0;
    f.avx512vl = c7[1] & (1 << 31) != 0;
    f.avx512vnni = c7[1] & (1 << 11) != 0;
    f.avx512vbmi = c7[1] & (1 << 31) != 0;

    // OS state: AVX2 needs YMM, AVX-512 needs opmask + the two ZMM halves.
    if (!(avx and f.fma and f.f16c and xcr0Ok(0x6))) {
        f.avx2 = false;
        f.fma = false;
    }
    if (!(f.avx2 and f.avx512f and f.avx512bw and f.avx512vl and f.avx512dq and xcr0Ok(0xE6))) {
        f.avx512f = false;
        f.avx512bw = false;
        f.avx512vl = false;
        f.avx512dq = false;
        f.avx512vnni = false;
    }
    return f;
}

// ---------------------------------------------------------------------------
// bound kernels - using C kernels only
// ---------------------------------------------------------------------------

pub const Dot32Table = struct {
    q8_0_q8_0: api.Dot32,
    q4_0_q8_0: api.Dot32,
    q4_1_q8_0: api.Dot32,
    q5_0_q8_0: api.Dot32,
    q5_1_q8_0: api.Dot32,
    q1_0_q8_0: api.Dot32,
    q2_0_q8_0: api.Dot32,
    iq4_nl_q8_0: api.Dot32,
    mxfp4_q8_0: api.Dot32,
    nvfp4_q8_0: api.Dot32,
};

fn makeDot32() Dot32Table {
    return .{
        .q8_0_q8_0 = c_kernels.dotQ8_0Q8_0,
        .q4_0_q8_0 = c_kernels.dotQ4_0Q8_0,
        .q4_1_q8_0 = c_kernels.dotQ4_1Q8_0,
        .q5_0_q8_0 = c_kernels.dotQ5_0Q8_0,
        .q5_1_q8_0 = c_kernels.dotQ5_1Q8_0,
        .q1_0_q8_0 = c_kernels.dotQ1_0Q8_0,
        .q2_0_q8_0 = c_kernels.dotQ2_0Q8_0,
        .iq4_nl_q8_0 = c_kernels.dotIq4NlQ8_0,
        .mxfp4_q8_0 = c_kernels.dotMxfp4Q8_0,
        .nvfp4_q8_0 = c_kernels.dotNvfp4Q8_0,
    };
}

/// 32-block kernels (Q8_0 activation), bound to the selected tier.
pub var d32: Dot32Table = makeDot32();

/// Row-level 32-block kernels: one call per weight row, one horizontal
/// reduction per row.  These are the fast path (the block-level table reduces
/// every 32 elements, which measured slower than the scalar loop).
pub const Dot32RowTable = struct {
    q8_0_q8_0: api.Dot32,
    q4_0_q8_0: api.Dot32,
    q4_1_q8_0: api.Dot32,
    q5_0_q8_0: api.Dot32,
    q5_1_q8_0: api.Dot32,
    q1_0_q8_0: api.Dot32,
    q2_0_q8_0: api.Dot32,
    iq4_nl_q8_0: api.Dot32,
    mxfp4_q8_0: api.Dot32,
    nvfp4_q8_0: api.Dot32,
};

fn makeDot32Row() Dot32RowTable {
    return .{
        .q8_0_q8_0 = c_kernels.dotQ8_0Q8_0,
        .q4_0_q8_0 = c_kernels.dotQ4_0Q8_0,
        .q4_1_q8_0 = c_kernels.dotQ4_1Q8_0,
        .q5_0_q8_0 = c_kernels.dotQ5_0Q8_0,
        .q5_1_q8_0 = c_kernels.dotQ5_1Q8_0,
        .q1_0_q8_0 = c_kernels.dotQ1_0Q8_0,
        .q2_0_q8_0 = c_kernels.dotQ2_0Q8_0,
        .iq4_nl_q8_0 = c_kernels.dotIq4NlQ8_0,
        .mxfp4_q8_0 = c_kernels.dotMxfp4Q8_0,
        .nvfp4_q8_0 = c_kernels.dotNvfp4Q8_0,
    };
}

/// 32-block row kernels (Q8_0 activation), bound to the selected tier.
pub var d32row: Dot32RowTable = makeDot32Row();

pub const Dot256Table = struct {
    iq2_xxs_q8_k: api.Dot256,
    iq2_xs_q8_k: api.Dot256,
    iq2_s_q8_k: api.Dot256,
    iq3_xxs_q8_k: api.Dot256,
    iq3_s_q8_k: api.Dot256,
    iq1_s_q8_k: api.Dot256,
    iq1_m_q8_k: api.Dot256,
    iq4_xs_q8_k: api.Dot256,
    q2_k_q8_k: api.Dot256,
    q3_k_q8_k: api.Dot256,
    q4_k_q8_k: api.Dot256,
    q5_k_q8_k: api.Dot256,
    q6_k_q8_k: api.Dot256,
};

fn makeDot256() Dot256Table {
    return .{
        .iq2_xxs_q8_k = c_kernels.dotIq2XxsQ8K,
        .iq2_xs_q8_k = c_kernels.dotIq2XsQ8K,
        .iq2_s_q8_k = c_kernels.dotIq2SQ8K,
        .iq3_xxs_q8_k = c_kernels.dotIq3XxsQ8K,
        .iq3_s_q8_k = c_kernels.dotIq3SQ8K,
        .iq1_s_q8_k = c_kernels.dotIq1SQ8K,
        .iq1_m_q8_k = c_kernels.dotIq1MQ8K,
        .iq4_xs_q8_k = c_kernels.dotIq4XsQ8K,
        .q2_k_q8_k = c_kernels.dotQ2KQ8K,
        .q3_k_q8_k = c_kernels.dotQ3KQ8K,
        .q4_k_q8_k = c_kernels.dotQ4KQ8K,
        .q5_k_q8_k = c_kernels.dotQ5KQ8K,
        .q6_k_q8_k = c_kernels.dotQ6KQ8K,
    };
}

/// 256-block IQ kernels (Q8_K activation), using C implementation.
pub var d256: Dot256Table = makeDot256();

/// IQ kernels (Q8_K activation) - C implementation.
pub var diq: Dot256Table = makeDot256();

/// Activation quantizers - using C implementation.
pub var actQ8_0: api.QuantizeQ8_0 = &c_kernels.quantizeQ8_0;
pub var actQ8_K: api.QuantizeQ8_K = &c_kernels.quantizeQ8_K;

/// Choose the widest tier the running CPU supports. Idempotent.
/// Note: Currently all tiers use the same C kernels (AVX2 optimized).
/// The tier is detected but doesn't change the kernel selection.
pub fn init() void {
    if (force_tier) |ft| {
        tier = ft;
        return;
    }
    const f = detect();
    if (f.avx512f and f.avx512bw and f.avx512vl and f.avx512dq) {
        if (f.avx512vbmi and f.avx512vnni) {
            tier = .avx512vbmi;
        } else if (f.avx512vnni) {
            tier = .avx512vnni;
        } else {
            tier = .avx512;
        }
    } else if (f.avx2) {
        tier = .avx2;
    } else {
        tier = .baseline;
    }
    // Note: All tiers currently use the same C kernels
    // The C kernels are compiled with AVX2 and use runtime dispatch internally
}

pub fn features() Features {
    return detect();
}
