//! Shared SIMD primitives for the ported llama.cpp quantization kernels.
//!
//! Written with `@Vector` at 128-bit widths (16 x i8, 8 x f32, ...).  LLVM
//! widens the same IR to 256-bit (ymm) on the AVX2 tier and 512-bit (zmm) on
//! the AVX-512 tier, because build.zig compiles one copy of this directory per
//! `-mcpu`.  No inline assembly, no C, no C++.

const std = @import("std");
const tier = @import("tier.zig");

pub const V8u8 = @Vector(8, u8);
pub const V16u8 = @Vector(16, u8);
pub const V16i8 = @Vector(16, i8);
pub const V32i8 = @Vector(32, i8);
pub const V32u8 = @Vector(32, u8);
pub const V64i8 = @Vector(64, i8);
pub const V64u8 = @Vector(64, u8);
pub const V8f32 = @Vector(8, f32);
pub const V16f32 = @Vector(16, f32);
pub const V16i16 = @Vector(16, i16);
pub const V16i32 = @Vector(16, i32);
pub const V32i16 = @Vector(32, i16);
pub const V32i32 = @Vector(32, i32);

// ---------------------------------------------------------------------------
// unaligned loads / stores
// ---------------------------------------------------------------------------

/// Load a @Vector from an unaligned pointer.
pub inline fn load(comptime T: type, p: anytype) T {
    const n = comptime @typeInfo(T).vector.len * @sizeOf(@typeInfo(T).vector.child);
    const ptr: *align(1) const [n]u8 = @ptrCast(@alignCast(p));
    return @bitCast(ptr.*);
}

pub inline fn loadF32(comptime N: usize, p: anytype) @Vector(N, f32) {
    const ptr: *align(1) const [N]f32 = @ptrCast(@alignCast(p));
    return ptr.*;
}

pub inline fn storeF32(p: anytype, v: anytype) void {
    const n = comptime @typeInfo(@TypeOf(v)).vector.len;
    const ptr: *align(1) [n]f32 = @ptrCast(@alignCast(p));
    ptr.* = v;
}

pub inline fn u16At(p: anytype) u16 {
    const ptr: *align(1) const u16 = @ptrCast(@alignCast(p));
    return ptr.*;
}

pub inline fn u32At(p: anytype) u32 {
    const ptr: *align(1) const u32 = @ptrCast(@alignCast(p));
    return ptr.*;
}

/// Dequantize one 256-element Q8_K activation block (d(f32)@0, qs[256] i8@4)
/// to f32: out[i] = d * qs[i].  Matches the engine's exact f32 view.
///
/// Vectorized: 8 chunks of 32 i8, each sign-extended to i32 lane-by-lane
/// (`@as`, which is the vpmovsxbd/vpmovsxwd family on AVX2/AVX-512) and
/// multiplied by the broadcast d.  |qs| <= 127 is exactly representable in
/// f32, and f32 multiplication is bit-commutative, so every element is
/// bit-identical to the scalar `d * @floatFromInt(q)` form.
pub inline fn deqQ8K(p: [*]const u8, a: *[256]f32) void {
    const d: f32 = @bitCast(u32At(p));
    const dv16: V16f32 = @splat(d);
    const qs: [*]const u8 = p + 4;
    inline for (0..8) |c| {
        const q8: V32i8 = load(V32i8, qs + c * 32);
        const q32: V32i32 = @as(V32i32, q8);
        const arr: [32]i32 = @bitCast(q32);
        const lo: @Vector(16, i32) = @bitCast(arr[0..16].*);
        const hi: @Vector(16, i32) = @bitCast(arr[16..32].*);
        storeF32(a[c * 32 ..], @as(V16f32, @floatFromInt(lo)) * dv16);
        storeF32(a[c * 32 + 16 ..], @as(V16f32, @floatFromInt(hi)) * dv16);
    }
}

// ---------------------------------------------------------------------------
// f16
// ---------------------------------------------------------------------------

pub inline fn f16At(p: anytype, off: usize) f32 {
    const h: f16 = @bitCast(u16At(@as([*]const u8, @ptrCast(p)) + off));
    return @floatCast(h);
}

/// 8 f16 -> 8 f32 (vpmovzxwd+vcmpps on AVX2/AVX512; scalar cvtss on baseline).
pub inline fn f16x8(v: @Vector(8, u16)) @Vector(8, f32) {
    const h: @Vector(8, f16) = @bitCast(v);
    return @floatCast(h);
}

pub inline fn f16x4(v: @Vector(4, u16)) @Vector(4, f32) {
    const h: @Vector(4, f16) = @bitCast(v);
    return @floatCast(h);
}

pub inline fn f16x2(v: @Vector(2, u16)) @Vector(2, f32) {
    const h: @Vector(2, f16) = @bitCast(v);
    return @floatCast(h);
}

pub inline fn f32ToF16Bits(x: f32) u16 {
    const h: f16 = @floatCast(x);
    return @bitCast(h);
}

// ---------------------------------------------------------------------------
// integer dot products (vpmaddwd / vpmaddubsw + vpaddd)
// ---------------------------------------------------------------------------

/// Horizontal sum of a i16 vector as i32 (widens in halves: no @extend in 0.17).
pub inline fn sumI16(p: @Vector(16, i16)) i32 {
    const arr: [16]i16 = @bitCast(p);
    const lo: @Vector(8, i16) = @bitCast(arr[0..8].*);
    const hi: @Vector(8, i16) = @bitCast(arr[8..16].*);
    return @reduce(.Add, @as(@Vector(8, i32), lo)) + @reduce(.Add, @as(@Vector(8, i32), hi));
}

pub inline fn sumI16x32(p: @Vector(32, i16)) i32 {
    const arr: [32]i16 = @bitCast(p);
    const lo: @Vector(16, i16) = @bitCast(arr[0..16].*);
    const hi: @Vector(16, i16) = @bitCast(arr[16..32].*);
    return @reduce(.Add, @as(@Vector(16, i32), lo)) + @reduce(.Add, @as(@Vector(16, i32), hi));
}

pub inline fn sumI16x64(p: @Vector(64, i16)) i32 {
    const arr: [64]i16 = @bitCast(p);
    const lo: @Vector(32, i16) = @bitCast(arr[0..32].*);
    const hi: @Vector(32, i16) = @bitCast(arr[32..64].*);
    return @reduce(.Add, @as(@Vector(32, i32), lo)) + @reduce(.Add, @as(@Vector(32, i32), hi));
}

/// sum of 32 i8*i8 products.
pub inline fn dot32i8(x: @Vector(32, i8), y: @Vector(32, i8)) i32 {
    return sumI16x32(@as(@Vector(32, i16), x) *% @as(@Vector(32, i16), y));
}

/// sum of 16 i8*i8 products.
pub inline fn dot16i8(x: @Vector(16, i8), y: @Vector(16, i8)) i32 {
    return sumI16(@as(@Vector(16, i16), x) *% @as(@Vector(16, i16), y));
}

/// sum of 64 i8*i8 products.
pub inline fn dot64i8(x: @Vector(64, i8), y: @Vector(64, i8)) i32 {
    return sumI16x64(@as(@Vector(64, i16), x) *% @as(@Vector(64, i16), y));
}

/// sum of 32 u8*i8 products (x zero-extends, y sign-extends).
/// NOTE: widening is `@as` (element-wise), NOT `@bitCast` -- @bitCast from a
/// 256-bit u8 vector to a 512-bit i16 vector is a compile error.
pub inline fn dot32u8i8(x: @Vector(32, u8), y: @Vector(32, i8)) i32 {
    return sumI16x32(@as(@Vector(32, i16), x) *% @as(@Vector(32, i16), y));
}

/// sum of 16 u8*i8 products.
pub inline fn dot16u8i8(x: @Vector(16, u8), y: @Vector(16, i8)) i32 {
    return sumI16(@as(@Vector(16, i16), x) *% @as(@Vector(16, i16), y));
}

/// sum of 64 u8*i8 products.
pub inline fn dot64u8i8(x: @Vector(64, u8), y: @Vector(64, i8)) i32 {
    return sumI16x64(@as(@Vector(64, i16), x) *% @as(@Vector(64, i16), y));
}

// ---------------------------------------------------------------------------
// nibble unpacking
// ---------------------------------------------------------------------------

/// low nibbles of 16 bytes as f32.
pub inline fn niblo(q: V16u8) V16f32 {
    return @as(V16f32, @floatFromInt(q & @as(V16u8, @splat(0x0F))));
}

/// high nibbles of 16 bytes as f32.
pub inline fn nibhi(q: V16u8) V16f32 {
    return @as(V16f32, @floatFromInt(q >> @as(V16u8, @splat(4))));
}

pub inline fn niblo8(q: V8u8) V8f32 {
    return @as(V8f32, @floatFromInt(q & @as(V8u8, @splat(0x0F))));
}

pub inline fn nibhi8(q: V8u8) V8f32 {
    return @as(V8f32, @floatFromInt(q >> @as(V8u8, @splat(4))));
}

/// 16 low nibbles as i8.
pub inline fn nibloI8(q: V16u8) V16i8 {
    return @as(V16i8, @bitCast(q & @as(V16u8, @splat(0x0F))));
}

/// 16 high nibbles as i8.
pub inline fn nibhiI8(q: V16u8) V16i8 {
    return @as(V16i8, @bitCast(q >> @as(V16u8, @splat(4))));
}

pub inline fn nibloI8_8(q: V8u8) @Vector(8, i8) {
    return @as(@Vector(8, i8), @bitCast(q & @as(V8u8, @splat(0x0F))));
}

pub inline fn nibhiI8_8(q: V8u8) @Vector(8, i8) {
    return @as(@Vector(8, i8), @bitCast(q >> @as(V8u8, @splat(4))));
}

pub inline fn bit(v: u32, comptime bitpos: usize) u32 {
    return (v >> bitpos) & 1;
}

// ---------------------------------------------------------------------------
// partial-sum forms (no horizontal reduction)
//
// The 32-block kernels are called once per 32 elements, so a kernel that
// horizontal-reduces every block pays a full cross-lane reduction per 32
// elements -- which measured SLOWER than the auto-vectorized scalar loop it
// replaced.  These helpers return the vpmaddwd-shaped partial sums so the
// caller can keep a vector accumulator across all blocks of a row and reduce
// exactly once per row.
// ---------------------------------------------------------------------------

/// 32 i8*i8 products as 16 i32 partial sums (adjacent pairs pre-added).
pub inline fn madd16i8(x: @Vector(32, i8), y: @Vector(32, i8)) @Vector(16, i32) {
    const p: @Vector(32, i16) = @as(@Vector(32, i16), x) *% @as(@Vector(32, i16), y);
    const pa: [32]i16 = @bitCast(p);
    var s: [16]i32 = undefined;
    for (0..16) |i| s[i] = @as(i32, pa[2 * i]) + @as(i32, pa[2 * i + 1]);
    return @bitCast(s);
}

/// 16 i8*i8 products as 8 i32 partial sums.
pub inline fn madd8i8(x: @Vector(16, i8), y: @Vector(16, i8)) @Vector(8, i32) {
    const p: @Vector(16, i16) = @as(@Vector(16, i16), x) *% @as(@Vector(16, i16), y);
    const pa: [16]i16 = @bitCast(p);
    var s: [8]i32 = undefined;
    for (0..8) |i| s[i] = @as(i32, pa[2 * i]) + @as(i32, pa[2 * i + 1]);
    return @bitCast(s);
}

/// Horizontal sum of a 16-lane f32 accumulator.
pub inline fn sumF16(v: @Vector(16, f32)) f32 {
    const a: [16]f32 = @bitCast(v);
    const lo: @Vector(8, f32) = @bitCast(a[0..8].*);
    const hi: @Vector(8, f32) = @bitCast(a[8..16].*);
    return @reduce(.Add, lo) + @reduce(.Add, hi);
}

/// Widen a 32-lane i8 vector into two 16-lane i32 vectors (for accumulating
/// sums of activation quants across blocks).
pub inline fn widenI8x32(v: @Vector(32, i8)) [2]@Vector(16, i32) {
    const a: [32]i8 = @bitCast(v);
    const lo: @Vector(16, i8) = @bitCast(a[0..16].*);
    const hi: @Vector(16, i8) = @bitCast(a[16..32].*);
    return .{ @as(@Vector(16, i32), lo), @as(@Vector(16, i32), hi) };
}
