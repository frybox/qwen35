//! 256-element super-block dot products: K-quant / ternary weights x Q8_K.
//!
//! Ported from llama.cpp:
//!   * semantics:  `ggml/src/ggml-cpu/quants.c`  (`*_q8_K_generic`)
//!   * vectorized: `ggml/src/ggml-cpu/arch/x86/quants.c`
//! Pure Zig: `@Vector` + std, no C, no C++, no inline asm.  build.zig compiles
//! one copy of this directory per `-mcpu`, so the same 128-bit `@Vector` source
//! becomes ymm code on the AVX2 tier and zmm code on the AVX-512 tier.
//!
//! Block layouts (ggml-common.h), all 256 elements per super-block:
//!   Q2_K  84 B : scales[16]@0 qs[64]@16 d(f16)@80 dmin(f16)@82
//!   Q3_K 110 B : hmask[32]@0 qs[64]@32 scales[12]@96 d(f16)@108
//!   Q4_K 144 B : d(f16)@0 dmin(f16)@2 scales[12]@4 qs[128]@16
//!   Q5_K 176 B : d(f16)@0 dmin(f16)@2 scales[12]@4 qh[32]@16 qs[128]@48
//!   Q6_K 210 B : ql[128]@0 qh[64]@128 scales[16](i8)@192 d(f16)@208
//!   TQ1_0 54 B : qs[48]@0 qh[4]@48 d(f16)@52
//!   TQ2_0 66 B : qs[64]@0 d(f16)@64
//! Activation Q8_K (292 B): d(f32)@0, qs[256](i8)@4, bsums[16](i16)@260.
//!
//! Every kernel has a scalar `_ref` twin translated line-by-line from the C
//! `*_generic` function; the tests compare them.

const std = @import("std");
const util = @import("util.zig");
const tier = @import("tier.zig");

const V16u8 = @Vector(16, u8);
const V32u8 = @Vector(32, u8);
const V32i8 = @Vector(32, i8);
const V32i16 = @Vector(32, i16);
const V32i32 = @Vector(32, i32);
const V16i32 = @Vector(16, i32);
const V32u16 = @Vector(32, u16);
const V32u5 = @Vector(32, u5);
const V32u3 = @Vector(32, u3);

const q8k_bytes = 292;

/// unit scale table for the types with no per-group scale (TQ1_0, TQ2_0)
const ones: [16]i32 = blk: {
    var t: [16]i32 = undefined;
    @memset(&t, 1);
    break :blk t;
};
const qk_k = 256;

// ---------------------------------------------------------------------------
// shared helpers
// ---------------------------------------------------------------------------

inline fn q8kD(y: [*]const u8) f32 {
    return @bitCast(util.u32At(y));
}

inline fn q8kQs(y: [*]const u8) [*]const u8 {
    return y + 4;
}

inline fn q8kBsum(y: [*]const u8, j: usize) i16 {
    const p: *align(1) const i16 = @ptrCast(y + 260 + j * 2);
    return p.*;
}

/// 32 low / high nibbles of 32 bytes as i8 (values 0..15).
inline fn niblo32(q: V32u8) V32i8 {
    return @bitCast(q & @as(V32u8, @splat(0x0F)));
}
inline fn nibhi32(q: V32u8) V32i8 {
    return @bitCast(q >> @as(V32u8, @splat(4)));
}

/// Broadcast a pair of i32 to lanes 0..7 and 8..15 (one 16-element group
/// each -- the vpmaddwd lane layout after a 32-byte chunk).
inline fn splat16(a: i32, b: i32) V16i32 {
    var arr: [16]i32 = undefined;
    for (0..8) |i| arr[i] = a;
    for (0..8) |i| arr[8 + i] = b;
    return @bitCast(arr);
}

/// Broadcast a pair of i32 to lanes 0..15 and 16..31.
inline fn splat2(a: i32, b: i32) V32i32 {
    var arr: [32]i32 = undefined;
    for (0..16) |i| arr[i] = a;
    for (0..16) |i| arr[16 + i] = b;
    return @bitCast(arr);
}

/// Vector -> array, for @memcpy targets (memcpy's slice arg gives @bitCast no
/// result type, so the conversion must be explicit).
inline fn arr32(v: V32i8) [32]i8 {
    return @bitCast(v);
}

inline fn arr32u8(v: V32u8) [32]i8 {
    return @bitCast(v);
}

inline fn loadI8(p: [*]const u8) V32i8 {
    return @bitCast(@as(*align(1) const [32]u8, @ptrCast(p)).*);
}

/// sum over 16-element groups of (scale[g] * sum(a*q8)), with one i32 vector
/// accumulator and a single @reduce at the end.  `scales[g]` is the integer
/// scale of group g (16 elements), precomputed by the caller.
inline fn dotScaled(a: *const [256]i8, q8: [*]const u8, scales: *const [16]i32) i32 {
    var acc: V32i32 = @splat(0);
    var i: usize = 0;
    while (i < 8) : (i += 1) {
        // 32 elements at a time = two 16-element groups, one i32 vector each
        const av: V32i8 = @as(*const [32]i8, @ptrCast(a[i * 32 ..])).*;
        const qv = loadI8(q8 + i * 32);
        const p: V32i16 = @as(V32i16, av) *% @as(V32i16, qv);
        acc +%= splat2(scales[2 * i], scales[2 * i + 1]) *% @as(V32i32, p);
    }
    return @reduce(.Add, acc);
}

// ---------------------------------------------------------------------------
// Q4_K x Q8_K   (quants.c:696 generic, arch/x86/quants.c:2102)
// ---------------------------------------------------------------------------
// aux8 = the 128 bytes unpacked as 4 x (32 low nibbles, 32 high nibbles).
// dot = d*sum_groups(scale_g * sum(a*q8)) - dmin*sum_j(bsums[j]*min[j/2])

inline fn unpackQ4K(qs: *const [128]u8, out: *[256]i8) void {
    for (0..4) |c| {
        const q: V32u8 = @as(*const [32]u8, @ptrCast(qs[c * 32 ..])).*;
        @memcpy(out[c * 64 ..][0..32], &arr32(niblo32(q)));
        @memcpy(out[c * 64 + 32 ..][0..32], &arr32(nibhi32(q)));
    }
}

/// Q4_K/Q5_K 6+4-bit scale/min unpacking (the utmp shuffle in the C code).
inline fn unpackScaleMin(scales12: *const [12]u8) [16]u8 {
    var utmp: [4]u32 = .{ 0, 0, 0, 0 };
    for (0..3) |i| {
        utmp[i] = @as(u32, scales12[i * 4]) |
            (@as(u32, scales12[i * 4 + 1]) << 8) |
            (@as(u32, scales12[i * 4 + 2]) << 16) |
            (@as(u32, scales12[i * 4 + 3]) << 24);
    }
    const kmask1: u32 = 0x3f3f3f3f;
    const kmask2: u32 = 0x0f0f0f0f;
    const kmask3: u32 = 0x03030303;
    utmp[3] = ((utmp[2] >> 4) & kmask2) | (((utmp[1] >> 6) & kmask3) << 4);
    const uaux = utmp[1] & kmask1;
    utmp[1] = (utmp[2] & kmask2) | (((utmp[0] >> 6) & kmask3) << 4);
    utmp[2] = uaux;
    utmp[0] &= kmask1;
    var out: [16]u8 = undefined;
    for (0..4) |i| {
        for (0..4) |b| out[i * 4 + b] = @truncate(utmp[i] >> @intCast(b * 8));
    }
    return out;
}


// ---------------------------------------------------------------------------
// AVX-512 VNNI path for Q4_K x Q8_K (vpdpbusd int8 dot-product accumulators).
// The asm helpers below are only *analyzed* when reached: the kernel is gated
// by `if (comptime tier.isAvx512Vnni)`, so in the baseline/avx2/avx512
// compilations the asm bodies are never semanalyzed (Zig lazily compiles),
// which is what keeps the zmm-only 'v' register constraint legal.
// Bit-identical to the scalar path below (verified over 16384+4096 blocks in
// /tmp/microbench).  ~2x faster than the avx512 tier kernel.
// ---------------------------------------------------------------------------
const Z = @Vector(64, i8);
const I32 = @Vector(16, i32);
const F32 = @Vector(16, f32);
const W4 = @Vector(4, u32);

inline fn vload64(p: [*]const u8) Z {
    return asm ("vmovdqu8 (%[src]), %[dst]" : [dst] "=v" (-> Z) : [src] "r" (p));
}
inline fn vsrlw4(a: Z) Z {
    return asm ("vpsrlw $4, %[a], %[dst]" : [dst] "=v" (-> Z) : [a] "v" (a));
}
inline fn vandd(a: Z, b: Z) Z {
    return asm ("vpandd %[b], %[a], %[dst]" : [dst] "=v" (-> Z) : [a] "v" (a), [b] "v" (b));
}
inline fn vpermi2d(idx: Z, a: Z, b: Z) Z {
    var out: Z = idx;
    asm ("vpermi2d %[b], %[a], %[out]" : [out] "+v" (out) : [a] "v" (a), [b] "v" (b));
    return out;
}
inline fn vzxbd16(p: [*]const u8) Z {
    return asm ("vpmovzxbd (%[src]), %[dst]" : [dst] "=v" (-> Z) : [src] "r" (p));
}
inline fn vsxwd16(p: [*]const u8) Z {
    return asm ("vpmovsxwd (%[src]), %[dst]" : [dst] "=v" (-> Z) : [src] "r" (p));
}
inline fn vmulld(a: Z, b: Z) Z {
    return asm ("vpmulld %[b], %[a], %[dst]" : [dst] "=v" (-> Z) : [a] "v" (a), [b] "v" (b));
}
/// dst[i] = src[idx[i]]
inline fn vpermd(src: Z, idx: Z) Z {
    return asm ("vpermd %[src], %[idx], %[dst]" : [dst] "=v" (-> Z) : [src] "v" (src), [idx] "v" (idx));
}
inline fn vdpbusd(acc: Z, a: Z, b: Z) Z {
    var out: Z = acc;
    asm ("vpdpbusd %[b], %[a], %[out]" : [out] "+v" (out) : [a] "v" (a), [b] "v" (b));
    return out;
}
inline fn vaddd(a: Z, b: Z) Z {
    return asm ("vpaddd %[b], %[a], %[dst]" : [dst] "=v" (-> Z) : [a] "v" (a), [b] "v" (b));
}
inline fn vload16w(p: [*]const u8) W4 {
    return asm ("vmovdqu (%[src]), %[dst]" : [dst] "=x" (-> W4) : [src] "r" (p));
}
inline fn vandd4(a: W4, b: W4) W4 {
    return asm ("vpandd %[b], %[a], %[dst]" : [dst] "=x" (-> W4) : [a] "x" (a), [b] "x" (b));
}
inline fn vpor4(a: W4, b: W4) W4 {
    return asm ("vpor %[b], %[a], %[dst]" : [dst] "=x" (-> W4) : [a] "x" (a), [b] "x" (b));
}
inline fn vsrld4x(a: W4) W4 {
    return asm ("vpsrld $4, %[a], %[dst]" : [dst] "=x" (-> W4) : [a] "x" (a));
}
inline fn vsrld2x(a: W4) W4 {
    return asm ("vpsrld $2, %[a], %[dst]" : [dst] "=x" (-> W4) : [a] "x" (a));
}
inline fn vshufd00(a: W4) W4 {
    return asm ("vpshufd $0, %[a], %[dst]" : [dst] "=x" (-> W4) : [a] "x" (a));
}
inline fn vshufd10(a: W4) W4 {
    return asm ("vpshufd $16, %[a], %[dst]" : [dst] "=x" (-> W4) : [a] "x" (a));
}
inline fn vshufdaa(a: W4) W4 {
    return asm ("vpshufd $170, %[a], %[dst]" : [dst] "=x" (-> W4) : [a] "x" (a));
}
inline fn vshufd40(a: W4) W4 {
    return asm ("vpshufd $64, %[a], %[dst]" : [dst] "=x" (-> W4) : [a] "x" (a));
}
inline fn vslld4x(a: W4) W4 {
    return asm ("vpslld $4, %[a], %[dst]" : [dst] "=x" (-> W4) : [a] "x" (a));
}
inline fn vzxbdX(x: W4) Z {
    return asm ("vpmovzxbd %[src], %[dst]" : [dst] "=v" (-> Z) : [src] "x" (x));
}

const v_idxA: Z = @bitCast([16]i32{ 0, 1, 2, 3, 4, 5, 6, 7, 16, 17, 18, 19, 20, 21, 22, 23 });
const v_idxB: Z = @bitCast([16]i32{ 8, 9, 10, 11, 12, 13, 14, 15, 24, 25, 26, 27, 28, 29, 30, 31 });
const v_sidx0: Z = @bitCast([16]i32{ 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1 });
const v_sidx1: Z = @bitCast([16]i32{ 2, 2, 2, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3, 3, 3 });
const v_sidx2: Z = @bitCast([16]i32{ 4, 4, 4, 4, 4, 4, 4, 4, 5, 5, 5, 5, 5, 5, 5, 5 });
const v_sidx3: Z = @bitCast([16]i32{ 6, 6, 6, 6, 6, 6, 6, 6, 7, 7, 7, 7, 7, 7, 7, 7 });
const v_midx: Z = @bitCast([16]i32{ 8, 8, 9, 9, 10, 10, 11, 11, 12, 12, 13, 13, 14, 14, 15, 15 });

const v_m3f: W4 = @splat(0x3f3f3f3f);
const v_m0f: W4 = @splat(0x0f0f0f0f);
const v_m30: W4 = @splat(0x30303030);
const v_mL0: W4 = @bitCast([4]u32{ 0xffff_ffff, 0, 0, 0 });
const v_mL1: W4 = @bitCast([4]u32{ 0, 0xffff_ffff, 0, 0 });
const v_mL2: W4 = @bitCast([4]u32{ 0, 0, 0xffff_ffff, 0 });
const v_mL3: W4 = @bitCast([4]u32{ 0, 0, 0, 0xffff_ffff });

/// Q4_K scale+min unpack -> 16 bytes = sm[0..15] (8 scales + 8 mins),
/// identical values to unpackScaleMin.
inline fn vsmUtmp(p: [*]const u8) W4 {
    const v = vload16w(p + 4);
    const A = vandd4(v, v_m3f);
    const Q = vandd4(v, v_m0f);
    const Q4 = vandd4(vsrld4x(v), v_m0f);
    const S = vandd4(vsrld2x(v), v_m30);
    const p0 = vandd4(vshufd00(A), v_mL0);
    const p1 = vandd4(vshufdaa(Q), v_mL1);
    const p2 = vandd4(vshufd00(S), v_mL1);
    const p3 = vandd4(vshufd10(A), v_mL2);
    const p4 = vandd4(vshufdaa(Q4), v_mL3);
    const p5 = vandd4(vshufd40(S), v_mL3);
    return vpor4(vpor4(vpor4(vpor4(vpor4(p0, p1), p2), p3), p4), p5);
}

fn q4_k_q8_k_vnni_impl(n: usize, vx: [*]const u8, vy: [*]const u8) callconv(.c) f32 {
    // In Debug the compiler eagerly semanalyzes every function body, which
    // rejects the zmm-only register constraints even in the vnni module; in
    // ReleaseFast the comptime-true branch is the only one analyzed.  Debug
    // builds (tests) fall through to the scalar path, which is bit-identical.
    if (comptime tier.isAvx512Vnni and @import("builtin").mode != .debug) {
        const nb = n / qk_k;
        var i: usize = 0;
        const m4: Z = @splat(0x0f);
        // Per-block f32 contributions are accumulated in zmm vectors; the two
        // lane reductions happen ONCE per row, after the block loop.
        var acc_out: F32 = @splat(0);
        var acc_min: F32 = @splat(0);
        while (i < nb) : (i += 1) {
            const x = vx + i * 144;
            const y = vy + i * q8k_bytes;
            const A = vload64(x + 16);
            const B = vload64(x + 80);
            const loA = vandd(A, m4);
            const hiA = vandd(vsrlw4(A), m4);
            const loB = vandd(B, m4);
            const hiB = vandd(vsrlw4(B), m4);
            const aux0 = vpermi2d(v_idxA, loA, hiA);
            const aux1 = vpermi2d(v_idxB, loA, hiA);
            const aux2 = vpermi2d(v_idxA, loB, hiB);
            const aux3 = vpermi2d(v_idxB, loB, hiB);
            const smv = vzxbdX(vsmUtmp(x));
            const sc0 = vpermd(smv, v_sidx0);
            const sc1 = vpermd(smv, v_sidx1);
            const sc2 = vpermd(smv, v_sidx2);
            const sc3 = vpermd(smv, v_sidx3);
            const q0 = vload64(y + 4);
            const q1 = vload64(y + 68);
            const q2 = vload64(y + 132);
            const q3 = vload64(y + 196);
            const dp0 = vdpbusd(@splat(0), aux0, q0);
            const dp1 = vdpbusd(@splat(0), aux1, q1);
            const dp2 = vdpbusd(@splat(0), aux2, q2);
            const dp3 = vdpbusd(@splat(0), aux3, q3);
            var acc = vaddd(vmulld(dp0, sc0), vmulld(dp1, sc1));
            acc = vaddd(acc, vaddd(vmulld(dp2, sc2), vmulld(dp3, sc3)));
            const bs = vsxwd16(y + 260);
            const mn = vpermd(smv, v_midx);
            const dy = @as(f32, @bitCast(util.u32At(y)));
            const dv: F32 = @splat(util.f16At(x, 0) * dy);
            const dnv: F32 = @splat(util.f16At(x, 2) * dy);
            // d_b * isum_b and dmin_b * sumi_b as 16-lane f32 vectors: no
            // cross-lane reduction inside the block loop.  Two independent
            // accumulator chains keep ILP exposed.
            acc_out = acc_out + dv * @as(F32, @floatFromInt(@as(I32, @bitCast(acc))));
            acc_min = acc_min + dnv * @as(F32, @floatFromInt(@as(I32, @bitCast(vmulld(bs, mn)))));
        }
        return @reduce(.Add, acc_out) - @reduce(.Add, acc_min);
    }
    return 0;
}
pub fn q4_k_q8_k(n: usize, vx: [*]const u8, vy: [*]const u8) callconv(.c) f32 {
    // The VNNI kernel is only analyzed in the vnni module at non-Debug
    // optimization (Debug eagerly semanalyzes zmm asm constraints); Debug
    // builds use the scalar path below, which is bit-identical.
    if (comptime tier.isAvx512Vnni and @import("builtin").mode != .debug) {
        return q4_k_q8_k_vnni_impl(n, vx, vy);
    }
    const nb = n / qk_k;
    var sumf: f32 = 0;
    var i: usize = 0;
    while (i < nb) : (i += 1) {
        const x = vx + i * 144;
        const y = vy + i * q8k_bytes;
        var aux8: [256]i8 = undefined;
        unpackQ4K(@ptrCast(x + 16), &aux8);
        const sm = unpackScaleMin(@ptrCast(x + 4));
        const dy = q8kD(y);
        const d = util.f16At(x, 0) * dy;
        const dmin = util.f16At(x, 2) * dy;

        var sumi: i32 = 0;
        for (0..16) |j| sumi += @as(i32, q8kBsum(y, j)) * @as(i32, sm[8 + j / 2]);

        // 8 sub-block scales, each covering 32 elements -> group g (16 el) uses sm[g/2]
        var sc: [16]i32 = undefined;
        for (0..16) |g| sc[g] = @as(i32, sm[g / 2]);
        const isum = dotScaled(&aux8, q8kQs(y), &sc);
        sumf += d * @as(f32, @floatFromInt(isum)) - dmin * @as(f32, @floatFromInt(sumi));
    }
    return sumf;
}

pub fn q4_k_q8_k_ref(n: usize, vx: [*]const u8, vy: [*]const u8) callconv(.c) f32 {
    const nb = n / qk_k;
    var sumf: f32 = 0;
    var i: usize = 0;
    while (i < nb) : (i += 1) {
        const x = vx + i * 144;
        const y = vy + i * q8k_bytes;
        const q4 = x + 16;
        const q8 = q8kQs(y);
        const sm = unpackScaleMin(@ptrCast(x + 4));
        var aux8: [256]u8 = undefined;
        for (0..4) |c| {
            for (0..32) |l| aux8[c * 64 + l] = q4[c * 32 + l] & 0x0F;
            for (0..32) |l| aux8[c * 64 + 32 + l] = q4[c * 32 + l] >> 4;
        }
        var sumi: i32 = 0;
        for (0..16) |j| sumi += @as(i32, q8kBsum(y, j)) * @as(i32, sm[8 + j / 2]);
        var isum: i32 = 0;
        for (0..16) |g| {
            var sub: i32 = 0;
            for (0..16) |l| sub += @as(i32, @as(i8, @bitCast(q8[g * 16 + l]))) * @as(i32, aux8[g * 16 + l]);
            isum += @as(i32, sm[g / 2]) * sub; // one scale per 32 elements
        }
        const d = util.f16At(x, 0) * q8kD(y);
        const dmin = util.f16At(x, 2) * q8kD(y);
        sumf += d * @as(f32, @floatFromInt(isum)) - dmin * @as(f32, @floatFromInt(sumi));
    }
    return sumf;
}

// ---------------------------------------------------------------------------
// Q5_K x Q8_K   (quants.c:771 generic, arch/x86/quants.c:2280)
// ---------------------------------------------------------------------------
// Same as Q4_K, plus the 5th bit from qh[32]: group 2c supplies bit 2c to the
// low-nibble half and bit 2c+1 to the high-nibble half.

pub fn q5_k_q8_k(n: usize, vx: [*]const u8, vy: [*]const u8) callconv(.c) f32 {
    const nb = n / qk_k;
    var sumf: f32 = 0;
    var i: usize = 0;
    while (i < nb) : (i += 1) {
        const x = vx + i * 176;
        const y = vy + i * q8k_bytes;
        var aux8: [256]i8 = undefined;
        for (0..4) |c| {
            const q: V32u8 = @as(*const [32]u8, @ptrCast(x + 48 + c * 32)).*;
            const qh: V32u8 = @as(*const [32]u8, @ptrCast(x + 16)).*;
            const b0: V32u8 = @splat(@as(u8, 1) << @as(u3, @intCast(2 * c)));
            const b1: V32u8 = @splat(@as(u8, 1) << @as(u3, @intCast(2 * c + 1)));
            const m0: @Vector(32, bool) = (qh & b0) != @as(V32u8, @splat(0));
            const m1: @Vector(32, bool) = (qh & b1) != @as(V32u8, @splat(0));
            const h0: V32i8 = @bitCast(@select(u8, m0, @as(V32u8, @splat(16)), @as(V32u8, @splat(0))));
            const h1: V32i8 = @bitCast(@select(u8, m1, @as(V32u8, @splat(16)), @as(V32u8, @splat(0))));
            @memcpy(aux8[c * 64 ..][0..32], &arr32(niblo32(q) +% h0));
            @memcpy(aux8[c * 64 + 32 ..][0..32], &arr32(nibhi32(q) +% h1));
        }
        const sm = unpackScaleMin(@ptrCast(x + 4));
        const dy = q8kD(y);
        const d = util.f16At(x, 0) * dy;
        const dmin = util.f16At(x, 2) * dy;

        var sumi: i32 = 0;
        for (0..16) |j| sumi += @as(i32, q8kBsum(y, j)) * @as(i32, sm[8 + j / 2]);

        // 8 sub-block scales, each covering 32 elements -> group g (16 el) uses sm[g/2]
        var sc: [16]i32 = undefined;
        for (0..16) |g| sc[g] = @as(i32, sm[g / 2]);
        const isum = dotScaled(&aux8, q8kQs(y), &sc);
        sumf += d * @as(f32, @floatFromInt(isum)) - dmin * @as(f32, @floatFromInt(sumi));
    }
    return sumf;
}

pub fn q5_k_q8_k_ref(n: usize, vx: [*]const u8, vy: [*]const u8) callconv(.c) f32 {
    const nb = n / qk_k;
    var sumf: f32 = 0;
    var i: usize = 0;
    while (i < nb) : (i += 1) {
        const x = vx + i * 176;
        const y = vy + i * q8k_bytes;
        const q4 = x + 48;
        const hm = x + 16;
        const q8 = q8kQs(y);
        const sm = unpackScaleMin(@ptrCast(x + 4));
        var aux8: [256]u8 = undefined;
        var m: u8 = 1;
        for (0..4) |c| {
            for (0..32) |l| aux8[c * 64 + l] = (q4[c * 32 + l] & 0x0F) + (if (hm[l] & m != 0) @as(u8, 16) else 0);
            for (0..32) |l| aux8[c * 64 + 32 + l] = (q4[c * 32 + l] >> 4) + (if (hm[l] & (m << 1) != 0) @as(u8, 16) else 0);
            m <<= 2;
        }
        var sumi: i32 = 0;
        for (0..16) |j| sumi += @as(i32, q8kBsum(y, j)) * @as(i32, sm[8 + j / 2]);
        var isum: i32 = 0;
        for (0..16) |g| {
            var sub: i32 = 0;
            for (0..16) |l| sub += @as(i32, @as(i8, @bitCast(q8[g * 16 + l]))) * @as(i32, aux8[g * 16 + l]);
            isum += @as(i32, sm[g / 2]) * sub; // one scale per 32 elements
        }
        const d = util.f16At(x, 0) * q8kD(y);
        const dmin = util.f16At(x, 2) * q8kD(y);
        sumf += d * @as(f32, @floatFromInt(isum)) - dmin * @as(f32, @floatFromInt(sumi));
    }
    return sumf;
}

// ---------------------------------------------------------------------------
// Q6_K x Q8_K   (quants.c:851 generic, arch/x86/quants.c:2490)
// ---------------------------------------------------------------------------
// aux8[128] per half: from 32 ql bytes and 32 qh bytes, 4 interleaved groups.

pub fn q6_k_q8_k(n: usize, vx: [*]const u8, vy: [*]const u8) callconv(.c) f32 {
    const nb = n / qk_k;
    var sumf: f32 = 0;
    var i: usize = 0;
    while (i < nb) : (i += 1) {
        const x = vx + i * 210;
        const y = vy + i * q8k_bytes;
        const q8 = q8kQs(y);
        const sc: *const [16]i8 = @ptrCast(x + 192);

        // One 16-lane accumulator for all 16 scale-groups; reduced once per block.
        var acc: V16i32 = @splat(0);
        for (0..2) |h| {
            const q4: V32u8 = @as(*const [32]u8, @ptrCast(x + h * 64)).*;
            const q4b: V32u8 = @as(*const [32]u8, @ptrCast(x + h * 64 + 32)).*;
            const qh: V32u8 = @as(*const [32]u8, @ptrCast(x + 128 + h * 32)).*;

            // aux8 for this half, in chunk order g=0..3 (each 32 elements):
            //   g0 = lo nibbles of q4[0..32]   g1 = lo nibbles of q4[32..64]
            //   g2 = hi nibbles of q4[0..32]   g3 = hi nibbles of q4[32..64]
            // plus ((qh >> 2g) & 3) * 16, minus 32.
            const lo = niblo32(q4);
            const lo2 = niblo32(q4b);
            const hi = nibhi32(q4);
            const hi2 = nibhi32(q4b);

            inline for (0..4) |g| {
                const k = h * 4 + g;
                const extra: V32i8 = @as(V32i8, @bitCast((qh >> @as(V32u5, @splat(@as(u5, 2 * g)))) & @as(V32u8, @splat(0x03)))) *% @as(V32i8, @splat(16));
                const src = if (g == 0) lo else if (g == 1) lo2 else if (g == 2) hi else hi2;
                const av = src +% extra +% @as(V32i8, @splat(-32));
                const qv = loadI8(q8 + k * 32);
                // 16 vpmaddwd partial sums: lanes 0..7 are scale-group k,
                // lanes 8..15 are scale-group k+1.
                acc +%= splat16(@as(i32, sc[2 * k]), @as(i32, sc[2 * k + 1])) *% util.madd16i8(av, qv);
            }
        }
        const dy = q8kD(y);
        const d = util.f16At(x, 208) * dy;
        sumf += d * @as(f32, @floatFromInt(@reduce(.Add, acc)));
    }
    return sumf;
}

pub fn q6_k_q8_k_ref(n: usize, vx: [*]const u8, vy: [*]const u8) callconv(.c) f32 {
    const nb = n / qk_k;
    var sumf: f32 = 0;
    var i: usize = 0;
    while (i < nb) : (i += 1) {
        const x = vx + i * 210;
        const y = vy + i * q8k_bytes;
        const q4 = x;
        const qh = x + 128;
        const q8 = q8kQs(y);
        var aux8: [256]i8 = undefined;
        for (0..2) |h| {
            for (0..32) |l| {
                // q4 advances 64 bytes and qh 32 bytes per 128-element chunk (as in C)
                aux8[h * 128 + l + 0] = @truncate(@as(i32, (q4[h * 64 + l + 0] & 0x0F) | (((qh[h * 32 + l] >> 0) & 3) << 4)) - 32);
                aux8[h * 128 + l + 32] = @truncate(@as(i32, (q4[h * 64 + l + 32] & 0x0F) | (((qh[h * 32 + l] >> 2) & 3) << 4)) - 32);
                aux8[h * 128 + l + 64] = @truncate(@as(i32, (q4[h * 64 + l + 0] >> 4) | (((qh[h * 32 + l] >> 4) & 3) << 4)) - 32);
                aux8[h * 128 + l + 96] = @truncate(@as(i32, (q4[h * 64 + l + 32] >> 4) | (((qh[h * 32 + l] >> 6) & 3) << 4)) - 32);
            }
        }
        var isum: i32 = 0;
        for (0..16) |g| {
            const scale: i32 = @as(i32, @as(*const [16]i8, @ptrCast(x + 192))[g]);
            var sub: i32 = 0;
            for (0..16) |l| sub += @as(i32, @as(i8, @bitCast(q8[g * 16 + l]))) * @as(i32, aux8[g * 16 + l]);
            isum += scale * sub;
        }
        sumf += util.f16At(x, 208) * q8kD(y) * @as(f32, @floatFromInt(isum));
    }
    return sumf;
}

// ---------------------------------------------------------------------------
// Q3_K x Q8_K   (quants.c:617 generic, arch/x86/quants.c:1830)
// ---------------------------------------------------------------------------
// aux8 values are (2-bit nibble) - (hmask bit set ? 0 : 4), in [-4, 3].
// Scales are 6-bit signed, used as (scale - 32).

inline fn unpackQ3KScales(scales12: *const [12]u8) [16]i8 {
    var auxs: [4]u32 = .{ 0, 0, 0, 0 };
    for (0..3) |i| {
        auxs[i] = @as(u32, scales12[i * 4]) |
            (@as(u32, scales12[i * 4 + 1]) << 8) |
            (@as(u32, scales12[i * 4 + 2]) << 16) |
            (@as(u32, scales12[i * 4 + 3]) << 24);
    }
    const kmask1: u32 = 0x03030303;
    const kmask2: u32 = 0x0f0f0f0f;
    const tmp = auxs[2];
    auxs[2] = ((auxs[0] >> 4) & kmask2) | (((tmp >> 4) & kmask1) << 4);
    auxs[3] = ((auxs[1] >> 4) & kmask2) | (((tmp >> 6) & kmask1) << 4);
    auxs[0] = (auxs[0] & kmask2) | (((tmp >> 0) & kmask1) << 4);
    auxs[1] = (auxs[1] & kmask2) | (((tmp >> 2) & kmask1) << 4);
    var out: [16]i8 = undefined;
    for (0..4) |i| {
        for (0..4) |b| out[i * 4 + b] = @bitCast(@as(u8, @truncate(auxs[i] >> @intCast(b * 8))));
    }
    return out;
}

// AVX-512 VNNI path for Q3_K x Q8_K (vpdpbusd int8 dot-product accumulators).
// Same gate/structure as q4_k_q8_k above: the zmm-only asm helpers below are
// private to this file and are only semanalyzed inside the
// `tier.isAvx512Vnni and mode != .debug` branch; Debug tests use the @Vector
// path, which is bit-identical to q3_k_q8_k_ref.
//
// aux8[f] (f = h*128 + g'*32 + l') = nib - sub, nib = (qs[h*32+l'] >> 2*g') & 3,
// sub = (hmask[l'] & (1 << (4h + g'))) != 0 ? 0 : 4.
// Feed dpbusd the UNSIGNED value aux8 + 4 = nib + 4*bit in [0,7]; then
//   sum_g scale'_g * sum(aux8*q8) = sum_g scale'_g * (dpbusd_g - 4 * bsums_g)
// with scale'_g = sc[g] - 32 and bsums_g = sum q8[g*16..g*16+15] (the Q8_K
// bsums at y+260), so the hmask "min" term comes from bsums, exactly like the
// q4_k min term.  Byte layout: aux vector k's 16-byte chunks are groups
//   aux0: g0,g1,g2,g3   aux1: g8,g9,g10,g11   aux2: g4..g7   aux3: g12..g15
// (f = h*128 + g'*32 + l' maps to group h*8 + 2*g' + (l'/16)), paired with the
// q8 loads y+4 (g0..3), y+68 (g4..7), y+132 (g8..11), y+196 (g12..15).
inline fn vporz(a: Z, b: Z) Z {
    return asm ("vpord %[b], %[a], %[dst]" : [dst] "=v" (-> Z) : [a] "v" (a), [b] "v" (b));
}
inline fn vpsubd(a: Z, b: Z) Z {
    return asm ("vpsubd %[b], %[a], %[dst]" : [dst] "=v" (-> Z) : [a] "v" (a), [b] "v" (b));
}
inline fn vsrlw2(a: Z) Z {          // 16-bit lanes shifted right by 2
    return asm ("vpsrlw $2, %[a], %[dst]" : [dst] "=v" (-> Z) : [a] "v" (a));
}
inline fn vsrlw6(a: Z) Z {          // 16-bit lanes shifted right by 6
    return asm ("vpsrlw $6, %[a], %[dst]" : [dst] "=v" (-> Z) : [a] "v" (a));
}

/// broadcast hmask[0..15] / hmask[16..31] over both 32-byte halves
const v_hbcast: Z = @bitCast([16]i32{ 0, 1, 2, 3, 4, 5, 6, 7, 0, 1, 2, 3, 4, 5, 6, 7 });
/// scale broadcast: each 4-lane chunk = one 16-element group
const v_sg0: Z = @bitCast([16]i32{ 0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3 });
const v_sg1: Z = @bitCast([16]i32{ 4, 4, 4, 4, 5, 5, 5, 5, 6, 6, 6, 6, 7, 7, 7, 7 });
const v_sg2: Z = @bitCast([16]i32{ 8, 8, 8, 8, 9, 9, 9, 9, 10, 10, 10, 10, 11, 11, 11, 11 });
const v_sg3: Z = @bitCast([16]i32{ 12, 12, 12, 12, 13, 13, 13, 13, 14, 14, 14, 14, 15, 15, 15, 15 });

/// hmask bit masks per (g'): low 32 bytes = h = 0 (bits 0..3), high 32 bytes =
/// h = 1 (bits 4..7); value 4*bit, OR-ed into the 0..3 nibble (no carry).
const cmask0: Z = blk: {
    var a: [64]u8 = undefined;
    for (0..32) |i| {
        a[i] = 4;
        a[32 + i] = 16;
    }
    break :blk @bitCast(a);
};
const cmask1: Z = blk: {
    var a: [64]u8 = undefined;
    for (0..32) |i| {
        a[i] = 8;
        a[32 + i] = 32;
    }
    break :blk @bitCast(a);
};
const cmask2: Z = blk: {
    var a: [64]u8 = undefined;
    for (0..32) |i| {
        a[i] = 16;
        a[32 + i] = 64;
    }
    break :blk @bitCast(a);
};
const cmask3: Z = blk: {
    var a: [64]u8 = undefined;
    for (0..32) |i| {
        a[i] = 32;
        a[32 + i] = 128;
    }
    break :blk @bitCast(a);
};

fn q3_k_q8_k_vnni_impl(n: usize, vx: [*]const u8, vy: [*]const u8) callconv(.c) f32 {
    if (comptime tier.isAvx512Vnni and @import("builtin").mode != .debug) {
        const nb = n / qk_k;
        var i: usize = 0;
        const z0: Z = @splat(0);
        const m3: Z = @splat(3);
        const v32z: Z = @splat(32);
        const v4z: Z = @splat(4);
        // per-block f32 contributions kept as 16-lane vectors; the two lane
        // reductions happen ONCE per row, after the block loop
        var acc_out: F32 = @splat(0);
        var acc_min: F32 = @splat(0);
        while (i < nb) : (i += 1) {
            const x = vx + i * 110;
            const y = vy + i * q8k_bytes;
            const A = vload64(x + 32);
            const H = vpermd(vload64(x), v_hbcast);
            const n0 = vandd(A, m3);
            const n1 = vandd(vsrlw2(A), m3);
            const n2 = vandd(vsrlw4(A), m3);
            const n3 = vandd(vsrlw6(A), m3);
            const a0 = vporz(n0, vandd(H, cmask0));
            const a1 = vporz(n1, vandd(H, cmask1));
            const a2 = vporz(n2, vandd(H, cmask2));
            const a3 = vporz(n3, vandd(H, cmask3));
            const aux0 = vpermi2d(v_idxA, a0, a1);
            const aux1 = vpermi2d(v_idxB, a0, a1);
            const aux2 = vpermi2d(v_idxA, a2, a3);
            const aux3 = vpermi2d(v_idxB, a2, a3);
            const dp0 = vdpbusd(z0, aux0, vload64(y + 4));
            const dp1 = vdpbusd(z0, aux1, vload64(y + 132));
            const dp2 = vdpbusd(z0, aux2, vload64(y + 68));
            const dp3 = vdpbusd(z0, aux3, vload64(y + 196));
            const sc16: [16]u8 = @bitCast(unpackQ3KScales(@ptrCast(x + 96)));
            const scv = vpsubd(vzxbd16(@as([*]const u8, @ptrCast(&sc16))), v32z);
            const sc4 = vmulld(scv, v4z);
            var acc = vaddd(vmulld(dp0, vpermd(scv, v_sg0)), vmulld(dp3, vpermd(scv, v_sg3)));
            acc = vaddd(acc, vaddd(vmulld(dp2, vpermd(scv, v_sg1)), vmulld(dp1, vpermd(scv, v_sg2))));
            const bs = vsxwd16(y + 260);
            var mn = vaddd(vmulld(bs, vpermd(sc4, v_sg0)), vmulld(bs, vpermd(sc4, v_sg3)));
            mn = vaddd(mn, vaddd(vmulld(bs, vpermd(sc4, v_sg1)), vmulld(bs, vpermd(sc4, v_sg2))));
            const dy: f32 = @bitCast(util.u32At(y));
            const dv: F32 = @splat(util.f16At(x, 108) * dy);
            // d_b * isum_b and d_b * 4*sum_g bsums_g*scale'_g as 16-lane f32
            // vectors: no cross-lane reduction inside the block loop, two
            // independent accumulator chains keep ILP exposed.
            acc_out = acc_out + dv * @as(F32, @floatFromInt(@as(I32, @bitCast(acc))));
            acc_min = acc_min + dv * @as(F32, @floatFromInt(@as(I32, @bitCast(mn))));
        }
        return @reduce(.Add, acc_out) - @reduce(.Add, acc_min);
    }
    return 0;
}

pub fn q3_k_q8_k(n: usize, vx: [*]const u8, vy: [*]const u8) callconv(.c) f32 {
    if (comptime tier.isAvx512Vnni and @import("builtin").mode != .debug) {
        return q3_k_q8_k_vnni_impl(n, vx, vy);
    }
    const nb = n / qk_k;
    var sumf: f32 = 0;
    var i: usize = 0;
    while (i < nb) : (i += 1) {
        const x = vx + i * 110;
        const y = vy + i * q8k_bytes;
        var aux8: [256]i8 = undefined;
        const qh: V32u8 = @as(*const [32]u8, @ptrCast(x)).*;
        for (0..2) |h| {
            const q3: V32u8 = @as(*const [32]u8, @ptrCast(x + 32 + h * 32)).*;
            inline for (0..4) |g| {
                const sh: V32u5 = @splat(@as(u5, @intCast(2 * g)));
                const nib: V32i8 = @bitCast((q3 >> sh) & @as(V32u8, @splat(0x03)));
                const bitv: V32u8 = @splat(@as(u8, 1) << @as(u3, @intCast(4 * h + g)));
                const m: @Vector(32, bool) = (qh & bitv) != @as(V32u8, @splat(0));
                const sub: V32i8 = @bitCast(@select(u8, m, @as(V32u8, @splat(0)), @as(V32u8, @splat(4))));
                @memcpy(aux8[h * 128 + g * 32 ..][0..32], &arr32(nib -% sub));
            }
        }
        const sc = unpackQ3KScales(@ptrCast(x + 96));
        const d = util.f16At(x, 108) * q8kD(y);
        var scv: [16]i32 = undefined;
        for (0..16) |g| scv[g] = @as(i32, sc[g]) - 32;
        const isum = dotScaled(&aux8, q8kQs(y), &scv);
        sumf += d * @as(f32, @floatFromInt(isum));
    }
    return sumf;
}

pub fn q3_k_q8_k_ref(n: usize, vx: [*]const u8, vy: [*]const u8) callconv(.c) f32 {
    const nb = n / qk_k;
    var sumf: f32 = 0;
    var i: usize = 0;
    while (i < nb) : (i += 1) {
        const x = vx + i * 110;
        const y = vy + i * q8k_bytes;
        const q3 = x + 32;
        const hm = x;
        const q8 = q8kQs(y);
        var aux8: [256]i8 = undefined;
        var m: u8 = 1;
        for (0..2) |h| {
            inline for (0..4) |g| {
                for (0..32) |l| {
                    const nib: i8 = @bitCast((q3[h * 32 + l] >> (2 * g)) & 0x03);
                    const sub: i8 = if (hm[l] & m != 0) 0 else 4;
                    aux8[h * 128 + g * 32 + l] = nib - sub;
                }
                m <<= 1;
            }
        }
        const sc = unpackQ3KScales(@ptrCast(x + 96));
        var isum: i32 = 0;
        for (0..16) |g| {
            const scale: i32 = @as(i32, sc[g]) - 32;
            var sub: i32 = 0;
            for (0..16) |l| sub += @as(i32, @as(i8, @bitCast(q8[g * 16 + l]))) * @as(i32, aux8[g * 16 + l]);
            isum += scale * sub;
        }
        sumf += util.f16At(x, 108) * q8kD(y) * @as(f32, @floatFromInt(isum));
    }
    return sumf;
}

// ---------------------------------------------------------------------------
// Q2_K x Q8_K   (quants.c:565 generic, arch/x86/quants.c:1638)
// ---------------------------------------------------------------------------
// aux8[k*128 + j*32 + l] = (qs[k*32+l] >> 2j) & 3, for k<2, j<4, l<32.
// Group g = k*8 + 2j (16 lanes) uses sc[g] & 0xF, group g+1 uses sc[g+1] & 0xF.
// Minus term: sum_j bsums[j] * (sc[j] >> 4).

pub fn q2_k_q8_k(n: usize, vx: [*]const u8, vy: [*]const u8) callconv(.c) f32 {
    const nb = n / qk_k;
    var sumf: f32 = 0;
    var i: usize = 0;
    while (i < nb) : (i += 1) {
        const x = vx + i * 84;
        const y = vy + i * q8k_bytes;
        const sc: *const [16]u8 = @ptrCast(x + 0);
        var aux8: [256]i8 = undefined;
        for (0..2) |k| {
            const q2: V32u8 = @as(*const [32]u8, @ptrCast(x + 16 + k * 32)).*;
            inline for (0..4) |j| {
                const sh: V32u5 = @splat(@as(u5, @intCast(2 * j)));
                @memcpy(aux8[k * 128 + j * 32 ..][0..32], &arr32u8((q2 >> sh) & @as(V32u8, @splat(0x03))));
            }
        }
        var summs: i32 = 0;
        for (0..16) |j| summs += @as(i32, q8kBsum(y, j)) * @as(i32, sc[j] >> 4);
        var scv: [16]i32 = undefined;
        for (0..16) |g| scv[g] = @as(i32, sc[g] & 0x0F);
        const isum = dotScaled(&aux8, q8kQs(y), &scv);
        const dall = q8kD(y) * util.f16At(x, 80);
        const dmin = q8kD(y) * util.f16At(x, 82);
        sumf += dall * @as(f32, @floatFromInt(isum)) - dmin * @as(f32, @floatFromInt(summs));
    }
    return sumf;
}

pub fn q2_k_q8_k_ref(n: usize, vx: [*]const u8, vy: [*]const u8) callconv(.c) f32 {
    const nb = n / qk_k;
    var sumf: f32 = 0;
    var i: usize = 0;
    while (i < nb) : (i += 1) {
        const x = vx + i * 84;
        const y = vy + i * q8k_bytes;
        const q2 = x + 16;
        const q8 = q8kQs(y);
        const sc: *const [16]u8 = @ptrCast(x + 0);
        var summs: i32 = 0;
        for (0..16) |j| summs += @as(i32, q8kBsum(y, j)) * @as(i32, sc[j] >> 4);
        var isum: i32 = 0;
        var is: usize = 0;
        for (0..2) |k| {
            for (0..4) |j| {
                const shift: u3 = @intCast(2 * j);
                var sub: i32 = 0;
                for (0..16) |l| sub += @as(i32, @as(i8, @bitCast(q8[k * 128 + j * 32 + l]))) * @as(i32, (q2[k * 32 + l] >> shift) & 0x03);
                isum += @as(i32, sc[is] & 0x0F) * sub;
                is += 1;
                sub = 0;
                for (0..16) |l| sub += @as(i32, @as(i8, @bitCast(q8[k * 128 + j * 32 + 16 + l]))) * @as(i32, (q2[k * 32 + 16 + l] >> shift) & 0x03);
                isum += @as(i32, sc[is] & 0x0F) * sub;
                is += 1;
            }
        }
        const dall = q8kD(y) * util.f16At(x, 80);
        const dmin = q8kD(y) * util.f16At(x, 82);
        sumf += dall * @as(f32, @floatFromInt(isum)) - dmin * @as(f32, @floatFromInt(summs));
    }
    return sumf;
}

// ---------------------------------------------------------------------------
// TQ2_0 x Q8_K   (quants.c:533 generic, arch/x86/quants.c:1572)
// ---------------------------------------------------------------------------
// 2 bits per element, ternary-ish: value = ((qs[k] >> 2l) & 3) - 1.

pub fn tq2_0_q8_k(n: usize, vx: [*]const u8, vy: [*]const u8) callconv(.c) f32 {
    const nb = n / qk_k;
    var sumf: f32 = 0;
    var i: usize = 0;
    while (i < nb) : (i += 1) {
        const x = vx + i * 66;
        const y = vy + i * q8k_bytes;
        var aux8: [256]i8 = undefined;
        for (0..2) |k| {
            const q: V32u8 = @as(*const [32]u8, @ptrCast(x + k * 32)).*;
            inline for (0..4) |l| {
                const sh: V32u8 = @splat(@as(u8, @intCast(2 * l)));
                const v: V32i8 = @bitCast((q >> sh) & @as(V32u8, @splat(0x03)));
                @memcpy(aux8[k * 128 + l * 32 ..][0..32], &arr32(v -% @as(V32i8, @splat(1))));
            }
        }
        const isum = dotScaled(&aux8, q8kQs(y), &ones);
        sumf += @as(f32, @floatFromInt(isum)) * (q8kD(y) * util.f16At(x, 64));
    }
    return sumf;
}

pub fn tq2_0_q8_k_ref(n: usize, vx: [*]const u8, vy: [*]const u8) callconv(.c) f32 {
    const nb = n / qk_k;
    var sumf: f32 = 0;
    var i: usize = 0;
    while (i < nb) : (i += 1) {
        const x = vx + i * 66;
        const y = vy + i * q8k_bytes;
        const q8 = q8kQs(y);
        var sumi: i32 = 0;
        for (0..2) |k| {
            for (0..4) |l| {
                for (0..32) |e| {
                    const v: i32 = @as(i32, (x[k * 32 + e] >> @intCast(2 * l)) & 0x03) - 1;
                    sumi += @as(i32, @as(i8, @bitCast(q8[k * 128 + l * 32 + e]))) * v;
                }
            }
        }
        sumf += @as(f32, @floatFromInt(sumi)) * (q8kD(y) * util.f16At(x, 64));
    }
    return sumf;
}

// ---------------------------------------------------------------------------
// TQ1_0 x Q8_K   (quants.c:481 generic, arch/x86/quants.c:1440)
// ---------------------------------------------------------------------------
// Base-5 packing: 48 bytes x 5 = 240 elements, then qh[4] x 4 = 16 elements.
// value = (((qs[j] * pow3[l]) & 0xFF) * 3 >> 8) - 1, i.e. the l-th ternary digit.

const pow3 = [6]u8{ 1, 3, 9, 27, 81, 243 };

inline fn tq1Vals(bytes: V32u8, comptime l: usize) V32i8 {
    const q: V32u8 = bytes *% @as(V32u8, @splat(pow3[l]));
    const xi: V32u16 = (@as(V32u16, q) *% @as(V32u16, @splat(@as(u16, 3)))) >> @as(V32u16, @splat(8));
    const xa: [32]u16 = @bitCast(xi);
    var out: [32]i8 = undefined;
    for (0..32) |k| out[k] = @bitCast(@as(u8, @truncate(xa[k])));
    const v: V32i8 = @bitCast(out);
    return v -% @as(V32i8, @splat(1));
}

pub fn tq1_0_q8_k(n: usize, vx: [*]const u8, vy: [*]const u8) callconv(.c) f32 {
    const nb = n / qk_k;
    var sumf: f32 = 0;
    var i: usize = 0;
    while (i < nb) : (i += 1) {
        const x = vx + i * 54;
        const y = vy + i * q8k_bytes;
        var aux8: [256]i8 = undefined;
        // 32 bytes -> 5 groups of 32 elements (activation index l*32 + m)
        const b0: V32u8 = @as(*const [32]u8, @ptrCast(x)).*;
        inline for (0..5) |l| {
            @memcpy(aux8[l * 32 ..][0..32], &arr32(tq1Vals(b0, l)));
        }
        // bytes 32..47 (16) -> 5 groups of 16 (activation index 160 + l*16 + m)
        for (0..5) |l| {
            for (0..16) |m| {
                const q: u8 = x[32 + m] *% pow3[l];
                aux8[160 + l * 16 + m] = @as(i8, @intCast((@as(u16, q) * 3) >> 8)) - 1;
            }
        }
        // qh[4] x 4 -> activation index 240 + l*4 + j
        for (0..4) |l| {
            for (0..4) |j| {
                const q: u8 = x[48 + j] *% pow3[l];
                aux8[240 + l * 4 + j] = @as(i8, @intCast((@as(u16, q) * 3) >> 8)) - 1;
            }
        }
        const isum = dotScaled(&aux8, q8kQs(y), &ones);
        sumf += @as(f32, @floatFromInt(isum)) * (util.f16At(x, 52) * q8kD(y));
    }
    return sumf;
}

pub fn tq1_0_q8_k_ref(n: usize, vx: [*]const u8, vy: [*]const u8) callconv(.c) f32 {
    const nb = n / qk_k;
    var sumf: f32 = 0;
    var i: usize = 0;
    while (i < nb) : (i += 1) {
        const x = vx + i * 54;
        const y = vy + i * q8k_bytes;
        const q8 = q8kQs(y);
        var sum: i32 = 0;
        for (0..32) |m| {
            for (0..5) |l| {
                const q: u8 = x[m] *% pow3[l];
                const xi: u16 = (@as(u16, q) * 3) >> 8;
                sum += @as(i32, @as(i8, @bitCast(q8[l * 32 + m]))) * (@as(i32, xi) - 1);
            }
        }
        for (0..16) |m| {
            for (0..5) |l| {
                const q: u8 = x[32 + m] *% pow3[l];
                const xi: u16 = (@as(u16, q) * 3) >> 8;
                sum += @as(i32, @as(i8, @bitCast(q8[160 + l * 16 + m]))) * (@as(i32, xi) - 1);
            }
        }
        for (0..4) |l| {
            for (0..4) |j| {
                const q: u8 = x[48 + j] *% pow3[l];
                const xi: u16 = (@as(u16, q) * 3) >> 8;
                sum += @as(i32, @as(i8, @bitCast(q8[240 + l * 4 + j]))) * (@as(i32, xi) - 1);
            }
        }
        sumf += @as(f32, @floatFromInt(sum)) * (util.f16At(x, 52) * q8kD(y));
    }
    return sumf;
}

/// Diagnostic: expose the SIMD aux8 builders so tests can diff them against the
/// scalar twins (the two paths must agree element-by-element, not just in sums).
pub fn dbgAuxQ5K(x: [*]const u8) [256]i8 {
    var aux8: [256]i8 = undefined;
    for (0..4) |c| {
        const q: V32u8 = @as(*const [32]u8, @ptrCast(x + 48 + c * 32)).*;
        const qh: V32u8 = @as(*const [32]u8, @ptrCast(x + 16)).*;
        const b0: V32u8 = @splat(@as(u8, 1) << @as(u3, @intCast(2 * c)));
        const b1: V32u8 = @splat(@as(u8, 1) << @as(u3, @intCast(2 * c + 1)));
        const m0: @Vector(32, bool) = (qh & b0) != @as(V32u8, @splat(0));
        const m1: @Vector(32, bool) = (qh & b1) != @as(V32u8, @splat(0));
        const h0: V32i8 = @bitCast(@select(u8, m0, @as(V32u8, @splat(16)), @as(V32u8, @splat(0))));
        const h1: V32i8 = @bitCast(@select(u8, m1, @as(V32u8, @splat(16)), @as(V32u8, @splat(0))));
        @memcpy(aux8[c * 64 ..][0..32], &arr32(niblo32(q) +% h0));
        @memcpy(aux8[c * 64 + 32 ..][0..32], &arr32(nibhi32(q) +% h1));
    }
    return aux8;
}

pub fn dbgAuxQ6K(x: [*]const u8) [256]i8 {
    var aux8: [256]i8 = undefined;
    for (0..2) |h| {
        const q4: V32u8 = @as(*const [32]u8, @ptrCast(x + h * 64)).*;
        const q4b: V32u8 = @as(*const [32]u8, @ptrCast(x + h * 64 + 32)).*;
        const qh: V32u8 = @as(*const [32]u8, @ptrCast(x + 128 + h * 32)).*;
        const sh: [4]V32u5 = .{
            @splat(@as(u5, 0)), @splat(@as(u5, 2)), @splat(@as(u5, 4)), @splat(@as(u5, 6)),
        };
        const lo = niblo32(q4);
        const lo2 = niblo32(q4b);
        const hi = nibhi32(q4);
        const hi2 = nibhi32(q4b);
        const base = @as(V32i8, @splat(-32));
        inline for (0..4) |g| {
            const extra: V32i8 = @as(V32i8, @bitCast((qh >> sh[g]) & @as(V32u8, @splat(0x03)))) *% @as(V32i8, @splat(16));
            const src = if (g == 0) lo else if (g == 1) lo2 else if (g == 2) hi else hi2;
            @memcpy(aux8[h * 128 + g * 32 ..][0..32], &arr32(src +% extra +% base));
        }
    }
    return aux8;
}
