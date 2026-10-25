//! Runtime ISA dispatch for the ported quantization kernels.
//!
//! Zig 0.17 has no function multiversioning, so build.zig compiles src/simd/
//! three times (baseline / AVX2 / AVX512) and this file picks one at runtime.
//! Detection is CPUID + XGETBV in pure Zig inline assembly -- the same checks
//! gcc's __builtin_cpu_supports does:
//!
//!   * the CPU must report the feature, AND
//!   * the OS must have enabled the corresponding XCR0 state bits (otherwise
//!     the instruction #GP's, even on a CPU that supports it).
//!
//! `init()` is idempotent and cheap; call it once at startup.  Before it is
//! called, every pointer is the baseline tier, so the engine is always safe to
//! use.

const std = @import("std");
const api = @import("simd_api.zig");
const simd_base = @import("simd_baseline");
const simd_avx2 = @import("simd_avx2");
const simd_avx512 = @import("simd_avx512");
const simd_avx512vnni = @import("simd_avx512vnni");
const simd_avx512vbmi = @import("simd_avx512vbmi");

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
// bound kernels
// ---------------------------------------------------------------------------
// One struct per kernel family; `init()` swaps the whole table to the widest
// tier the CPU supports, so adding a kernel means adding one field here and one
// line in make* -- no other file changes.

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

fn makeDot32(comptime m: type) Dot32Table {
    const k = m.dot32;
    return .{
        .q8_0_q8_0 = k.q8_0_q8_0,
        .q4_0_q8_0 = k.q4_0_q8_0,
        .q4_1_q8_0 = k.q4_1_q8_0,
        .q5_0_q8_0 = k.q5_0_q8_0,
        .q5_1_q8_0 = k.q5_1_q8_0,
        .q1_0_q8_0 = k.q1_0_q8_0,
        .q2_0_q8_0 = k.q2_0_q8_0,
        .iq4_nl_q8_0 = k.iq4_nl_q8_0,
        .mxfp4_q8_0 = k.mxfp4_q8_0,
        .nvfp4_q8_0 = k.nvfp4_q8_0,
    };
}

/// 32-block kernels (Q8_0 activation), bound to the selected tier.
pub var d32: Dot32Table = makeDot32(simd_base);

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

fn makeDot32Row(comptime m: type) Dot32RowTable {
    const k = m.dot32;
    return .{
        .q8_0_q8_0 = k.q8_0_q8_0_row,
        .q4_0_q8_0 = k.q4_0_q8_0_row,
        .q4_1_q8_0 = k.q4_1_q8_0_row,
        .q5_0_q8_0 = k.q5_0_q8_0_row,
        .q5_1_q8_0 = k.q5_1_q8_0_row,
        .q1_0_q8_0 = k.q1_0_q8_0_row,
        .q2_0_q8_0 = k.q2_0_q8_0_row,
        .iq4_nl_q8_0 = k.iq4_nl_q8_0_row,
        .mxfp4_q8_0 = k.mxfp4_q8_0_row,
        .nvfp4_q8_0 = k.nvfp4_q8_0_row,
    };
}

/// 32-block row kernels (Q8_0 activation), bound to the selected tier.
pub var d32row: Dot32RowTable = makeDot32Row(simd_base);

pub const Dot256Table = struct {
    q2_k_q8_k: api.Dot256,
    q3_k_q8_k: api.Dot256,
    q4_k_q8_k: api.Dot256,
    q5_k_q8_k: api.Dot256,
    q6_k_q8_k: api.Dot256,
    tq1_0_q8_k: api.Dot256,
    tq2_0_q8_k: api.Dot256,
};

fn makeDot256(comptime m: type) Dot256Table {
    const k = m.dot256;
    return .{
        .q2_k_q8_k = k.q2_k_q8_k,
        .q3_k_q8_k = k.q3_k_q8_k,
        .q4_k_q8_k = k.q4_k_q8_k,
        .q5_k_q8_k = k.q5_k_q8_k,
        .q6_k_q8_k = k.q6_k_q8_k,
        .tq1_0_q8_k = k.tq1_0_q8_k,
        .tq2_0_q8_k = k.tq2_0_q8_k,
    };
}

/// 256-block kernels (Q8_K activation), bound to the selected tier.
pub var d256: Dot256Table = makeDot256(simd_base);

/// Importance-sampled types (Q8_K activation).  Split over two kernel files
/// (kerniq.zig / kerniq2.zig) so the ports stay reviewable.
pub const DotIqTable = struct {
    iq2_xxs_q8_k: api.Dot256,
    iq2_xs_q8_k: api.Dot256,
    iq2_s_q8_k: api.Dot256,
    iq3_xxs_q8_k: api.Dot256,
    iq3_s_q8_k: api.Dot256,
    iq1_s_q8_k: api.Dot256,
    iq1_m_q8_k: api.Dot256,
    iq4_xs_q8_k: api.Dot256,
};

fn makeDotIq(comptime m: type) DotIqTable {
    const a = m.dotiq;
    const b = m.dotiq2;
    const c = m.dotiq3;
    return .{
        .iq2_xxs_q8_k = a.iq2_xxs_q8_k,
        .iq2_xs_q8_k = a.iq2_xs_q8_k,
        .iq3_xxs_q8_k = a.iq3_xxs_q8_k,
        .iq4_xs_q8_k = a.iq4_xs_q8_k,
        .iq2_s_q8_k = b.iq2_s_q8_k,
        .iq3_s_q8_k = b.iq3_s_q8_k,
        .iq1_s_q8_k = c.iq1_s_q8_k,
        .iq1_m_q8_k = c.iq1_m_q8_k,
    };
}

/// IQ kernels (Q8_K activation), bound to the selected tier.
pub var diq: DotIqTable = makeDotIq(simd_base);

/// Activation quantizers (identical semantics in every tier; the AVX tiers get
/// vectorized loads by LLVM).
pub var actQ8_0: api.QuantizeQ8_0 = &simd_base.act.quantizeQ8_0;
pub var actQ8_K: api.QuantizeQ8_K = &simd_base.act.quantizeQ8_K;

fn bindTier(comptime t: Tier) void {
    tier = t;
    const m = switch (t) {
        .baseline => simd_base,
        .avx2 => simd_avx2,
        .avx512 => simd_avx512,
        .avx512vnni => simd_avx512vnni,
        .avx512vbmi => simd_avx512vbmi,
    };
    d32 = makeDot32(m);
    d256 = makeDot256(m);
    d32row = makeDot32Row(m);
    diq = makeDotIq(m);
    actQ8_0 = &m.act.quantizeQ8_0;
    actQ8_K = &m.act.quantizeQ8_K;
}

/// Choose the widest tier the running CPU supports.  Idempotent.
pub fn init() void {
    if (force_tier) |ft| {
        switch (ft) {
            .baseline => bindTier(.baseline),
            .avx2 => bindTier(.avx2),
            .avx512 => bindTier(.avx512),
            .avx512vnni => bindTier(.avx512vnni),
            .avx512vbmi => bindTier(.avx512vbmi),
        }
        return;
    }
    const f = detect();
    if (f.avx512f and f.avx512bw and f.avx512vl and f.avx512dq) {
        // Split tier selection: Q-family (kern256) can use VNNI safely,
        // but IQ-family (kerniq/kerniq2/kerniq3) VNNI kernels have bugs.
        // This gives ~2x speedup for Q-family while keeping IQ correct.
        const use_vnni = f.avx512vnni;
        if (use_vnni) {
            tier = .avx512vnni;
            d32 = makeDot32(simd_avx512vnni);
            d256 = makeDot256(simd_avx512vnni);
            d32row = makeDot32Row(simd_avx512vnni);
            diq = makeDotIq(simd_avx512vnni); // enable VNNI for IQ
            actQ8_0 = &simd_avx512vnni.act.quantizeQ8_0;
            actQ8_K = &simd_avx512vnni.act.quantizeQ8_K;
        } else {
            bindTier(.avx512);
        }
    } else if (f.avx2) {
        tier = .avx2;
        d32 = makeDot32(simd_avx2);
        d256 = makeDot256(simd_avx2);
        d32row = makeDot32Row(simd_avx2);
        diq = makeDotIq(simd_avx2);
        actQ8_0 = &simd_avx2.act.quantizeQ8_0;
        actQ8_K = &simd_avx2.act.quantizeQ8_K;
    } else {
        tier = .baseline;
        d32 = makeDot32(simd_base);
        d256 = makeDot256(simd_base);
        d32row = makeDot32Row(simd_base);
        diq = makeDotIq(simd_base);
        actQ8_0 = &simd_base.act.quantizeQ8_0;
        actQ8_K = &simd_base.act.quantizeQ8_K;
    }
}

pub fn features() Features {
    return detect();
}
