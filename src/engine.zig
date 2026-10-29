//! Qwen3.5 CPU scalar inference engine (Q8_0 GGUF only).
//!
//! Linear weights stay in their native Q8_0 blocks (zero-copy views into the
//! mmapped GGUF); matvecs quantize the activation on the fly and accumulate
//! per-block dot products in i32. Norms and elementwise ops stay f32.
//!
//! Numerics follow ref/docs/SPEC.md (authoritative) and ref/docs/GDN_IMPL.md;
//! control flow mirrors ref/c/qwen35_engine.c. GGUF-specific notes:
//!   - GGUF tensor layout differs from the C ref's safetensors (see briefing):
//!     linear layers have attn_qkv [hidden, nk*kd|nk*kd|nv*kd] = [Q|K|V],
//!     attn_gate [hidden, nv*kd] (= z), ssm_alpha [hidden, nv], ssm_beta
//!     [hidden, nk], ssm_a [nv] is already -exp(A_log) (negative values,
//!     verified numerically against SPEC A7's A_log.exp() list), ssm_conv1d
//!     is F32 [4, conv_dim] channel-major taps.
//!     0.8B: hidden=1024, nv=nk=16, kd=128 (conv_dim 6144); 4B: hidden=2560,
//!     nv=32, nk=16, kd=128 (conv_dim 8192).
//!   - GGUF Qwen3_5RMSNorm weights are already (1+w) folded at conversion time
//!     (verified: blk.0.attn_norm[0] == 2.0, q_norm mean ~ 1.4), so they are
//!     applied directly; the gated ssm_norm (Qwen3_5RMSNormGated) is also
//!     pre-folded and applied directly.
//!   - llama.cpp conversion (conversion/qwen.py:394-404) fixes +1 for all
//!     Qwen3_5RMSNorm-family weights and stores ssm_a = -A_log.exp().

const std = @import("std");
const gguf = @import("gguf.zig");
const backend = @import("backend.zig");
const backend_cpu = @import("backend_cpu.zig");
const simd_ref = @import("simd_ref.zig");
const sampler = @import("sampler.zig");
const quant = @import("quant_dequant.zig");
const simd = @import("simd_dispatch.zig");
const simd_api = @import("simd_api.zig");
const pool_mod = @import("pool.zig");
const Io = std.Io;

/// Re-export so the CPU and CUDA hosts share one sampler implementation.
pub const Sampler = sampler.Sampler;

pub const Config = struct {
    n_layers: u32 = 24,
    full_attention_interval: u32 = 4,
    hidden: u32 = 1024,
    n_heads: u32 = 8,
    n_kv_heads: u32 = 2,
    head_dim: u32 = 256,
    n_ff: u32 = 3584,
    rms_eps: f32 = 1e-6,
    vocab: u32 = 248320,
    rope_freq_base: f32 = 1e7,
    /// B9/B10: [11,11,10] mrope sections on 32 freq pairs (rotary_dim=64);
    /// 4th section 0 = padding to 64 dims. Pure text => all three grids equal.
    rope_sections: [4]u32 = .{ 11, 11, 10, 0 },
    /// rotary_dim = head_dim * 0.25 = 64 (un-rotated tail = 192).
    rotary_dim: u32 = 64,
    // GDN
    gdn_n_heads: u32 = 16, // linear_num_value_heads (0.8B: 16, 4B: 32)
    /// k/q head count (ssm.group_count); v-head h uses k-head h / (nv/nk).
    /// 0.8B: 16 (nv == nk), 4B: 16 (nv/nk = 2).
    gdn_num_k_heads: u32 = 16,
    gdn_head_dim: u32 = 128,
    conv_kernel: u32 = 4,
    eos_token_id: u32 = 248046,

    pub fn isFullAttention(self: Config, layer: u32) bool {
        return self.full_attention_interval > 0 and
            layer % self.full_attention_interval == self.full_attention_interval - 1;
    }
};

// ---------------------------------------------------------------------------
// Scalar math (f32; SPEC C15: all norms computed in f32)
// ---------------------------------------------------------------------------

/// RMSNorm, out = x * rsqrt(mean(x^2) + eps) * w. eps inside the sum (SPEC).
/// The sum of squares stays scalar (exact order); the elementwise scale is
/// vectorized.
pub fn rmsNorm(out: []f32, x: []const f32, w: []const f32, eps: f32) void {
    std.debug.assert(out.len == x.len and out.len == w.len);
    var ss: f32 = 0;
    for (x) |v| ss += v * v;
    const scale = 1.0 / @sqrt(ss / @as(f32, @floatFromInt(x.len)) + eps);
    const sv: @Vector(GV, f32) = @splat(scale);
    const xp: [*]const f32 = x.ptr;
    const wp: [*]const f32 = w.ptr;
    const op: [*]f32 = out.ptr;
    for (0..out.len / GV) |b| {
        const o = b * GV;
        vstoreF32(GV, op + o, vloadF32(GV, xp + o) * sv * vloadF32(GV, wp + o));
    }
    for (out.len - out.len % GV..out.len) |i| out[i] = x[i] * scale * w[i];
}

pub fn sigmoid(x: f32) f32 {
    return 1.0 / (1.0 + @exp(-x));
}

/// Elementwise sigmoid over a whole slice (bit-identical to the scalar form).
fn sigmoidVec(x: []f32) void {
    const one: @Vector(GV, f32) = @splat(1.0);
    const p: [*]f32 = x.ptr;
    for (0..x.len / GV) |b| {
        const o = b * GV;
        vstoreF32(GV, p + o, one / (one + @exp(-vloadF32(GV, p + o))));
    }
}

/// out[i] *= silu(g[i]) for the whole slice.
fn mulSiluVec(out: []f32, gate: []const f32) void {
    const one: @Vector(GV, f32) = @splat(1.0);
    const op: [*]f32 = out.ptr;
    const gp: [*]const f32 = gate.ptr;
    for (0..out.len / GV) |b| {
        const o = b * GV;
        const g = vloadF32(GV, gp + o);
        vstoreF32(GV, op + o, vloadF32(GV, op + o) * (g / (one + @exp(-g))));
    }
}

pub fn silu(x: f32) f32 {
    return x / (1.0 + @exp(-x));
}

/// torch F.softplus default beta=1, threshold=20 (SPEC A4.4).
pub fn softplus(x: f32) f32 {
    return if (x > 20.0) x else std.math.log1p(@exp(x));
}

pub fn softmax(x: []f32) void {
    var max: f32 = -std.math.floatMax(f32);
    for (x) |v| max = @max(max, v);
    var sum: f32 = 0;
    for (x) |*v| {
        v.* = @exp(v.* - max);
        sum += v.*;
    }
    for (x) |*v| v.* /= sum;
}

/// FLA l2norm (SPEC A3): x * rsqrt(sum(x^2) + eps), eps inside the sum.
/// Sum of squares stays scalar (exact order); the scale is vectorized.
pub fn l2norm(x: []f32) void {
    var ss: f32 = 0;
    for (x) |v| ss += v * v;
    const inv = 1.0 / @sqrt(ss + 1e-6);
    scaleVec(x, inv);
}

// ---------------------------------------------------------------------------
// GDN delta-rule state kernels, @Vector-ized (SPEC A4)
// ---------------------------------------------------------------------------
//
// The state is [nv, kd, vd] row-major with kd == vd == 128, so every row is
// an exact number of 16-element f32 vectors.  All four per-head operations
// (decay, S^T k, outer-product update, S^T q) are written as @Vector(16, f32)
// loads/stores; the accumulation order per output element is unchanged from
// the scalar version, so results are bit-identical.  Loads/stores are
// unaligned because the state buffer is only 4-byte aligned.

inline fn vloadF32(comptime N: usize, p: [*]const f32) @Vector(N, f32) {
    const ptr: *align(1) const [N]f32 = @ptrCast(p);
    return @bitCast(ptr.*);
}

inline fn vstoreF32(comptime N: usize, p: [*]f32, v: @Vector(N, f32)) void {
    const ptr: *align(1) [N]f32 = @ptrCast(p);
    ptr.* = @bitCast(v);
}

const GV = 16; // f32 lanes per vector used by the GDN kernels

/// Elementwise x *= s (order-preserving, bit-identical to the scalar loop).
fn scaleVec(x: []f32, s: f32) void {
    const sv: @Vector(GV, f32) = @splat(s);
    const p: [*]f32 = x.ptr;
    for (0..x.len / GV) |b| {
        const o = b * GV;
        vstoreF32(GV, p + o, vloadF32(GV, p + o) * sv);
    }
    for (x.len - x.len % GV..x.len) |i| x[i] *= s;
}

/// (1) S *= gamma.
fn gdnDecay(sh: []f32, gamma: f32) void {
    const g: @Vector(GV, f32) = @splat(gamma);
    const p: [*]f32 = sh.ptr;
    for (0..sh.len / GV) |b| {
        const o = b * GV;
        vstoreF32(GV, p + o, vloadF32(GV, p + o) * g);
    }
    for (sh.len - sh.len % GV..sh.len) |i| sh[i] *= gamma;
}

/// (2) km[j] = sum_i sh[i*vd + j] * k[i].  km must be vd long; vd % GV == 0.
/// Accumulates in place in km (same per-element order as the scalar loop).
fn gdnKvMem(sh: []const f32, k: []const f32, km: []f32, vd: usize) void {
    std.debug.assert(vd % GV == 0 and km.len == vd);
    @memset(km, 0);
    const p: [*]const f32 = sh.ptr;
    const q: [*]f32 = km.ptr;
    for (0..k.len) |i| {
        const ki: @Vector(GV, f32) = @splat(k[i]);
        const row = p + i * vd;
        for (0..vd / GV) |b| {
            const o = b * GV;
            vstoreF32(GV, q + o, vloadF32(GV, q + o) + vloadF32(GV, row + o) * ki);
        }
    }
}

/// (3) delta = beta * (v - km);  S += k (outer) delta.
fn gdnOuter(sh: []f32, k: []const f32, v: []const f32, km: []const f32, delta: []f32, beta: f32, vd: usize) void {
    const b16: @Vector(GV, f32) = @splat(beta);
    const vp: [*]const f32 = v.ptr;
    const kp: [*]const f32 = km.ptr;
    const dp: [*]f32 = delta.ptr;
    for (0..vd / GV) |b| {
        const o = b * GV;
        vstoreF32(GV, dp + o, (vloadF32(GV, vp + o) - vloadF32(GV, kp + o)) * b16);
    }
    const p: [*]f32 = sh.ptr;
    for (0..k.len) |i| {
        const ki: @Vector(GV, f32) = @splat(k[i]);
        const row = p + i * vd;
        for (0..vd / GV) |b| {
            const o = b * GV;
            vstoreF32(GV, row + o, vloadF32(GV, row + o) + ki * vloadF32(GV, dp + o));
        }
    }
}

/// (4) out[j] = sum_i sh[i*vd + j] * q[i]  (row-wise accumulation, same order
/// as the scalar column loop).  Accumulates in place in out.
fn gdnStateOut(sh: []const f32, q: []const f32, out: []f32, vd: usize) void {
    std.debug.assert(vd % GV == 0 and out.len == vd);
    @memset(out, 0);
    const p: [*]const f32 = sh.ptr;
    const o: [*]f32 = out.ptr;
    for (0..q.len) |i| {
        const qi: @Vector(GV, f32) = @splat(q[i]);
        const row = p + i * vd;
        for (0..vd / GV) |b| {
            const ob = b * GV;
            vstoreF32(GV, o + ob, vloadF32(GV, o + ob) + vloadF32(GV, row + ob) * qi);
        }
    }
}

/// Fused GDN state update: (1) decay + (2) kv-mem in ONE pass over the state,
/// then (3) delta + outer-update and (4) S^T q in a SECOND pass.  Semantically
/// the same as gdnDecay; gdnKvMem; gdnOuter; gdnStateOut back to back, but the
/// 128x128 (2.1 MB) state is traversed twice instead of four times, and km/out
/// are accumulated in vector registers rather than load/store'd per row.
/// Fast path requires vd == 128 (all Qwen3.5 variants); otherwise it falls back
/// to the four separate kernels.
inline fn gdnFused(sh: []f32, k: []const f32, v: []const f32, q: []const f32, km: []f32, delta: []f32, out: []f32, gamma: f32, beta: f32, vd: usize) void {
    std.debug.assert(k.len == q.len and km.len == vd and out.len == vd and delta.len == vd);
    if (vd != 128) {
        gdnDecay(sh, gamma);
        gdnKvMem(sh, k, km, vd);
        gdnOuter(sh, k, v, km, delta, beta, vd);
        gdnStateOut(sh, q, out, vd);
        return;
    }
    const W = 4; // 128 / 32 lanes per @Vector(32, f32)
    const sp: [*]f32 = sh.ptr;
    const gp: [*]f32 = km.ptr;
    const op: [*]f32 = out.ptr;
    const g: @Vector(32, f32) = @splat(gamma);

    // Pass 1: read-only sweep, km += (S[i] * gamma) * k[i], accumulated in
    // registers.  The decayed rows are NOT stored: pass 2 recomputes them.
    var acc: [W]@Vector(32, f32) = undefined;
    for (0..W) |b| acc[b] = @splat(@as(f32, 0.0));
    for (0..k.len) |i| {
        const row = sp + i * vd;
        const ki: @Vector(32, f32) = @splat(k[i]);
        for (0..W) |b| acc[b] += vloadF32(32, row + b * 32) * ki;
    }
    // km is stored already decayed (S is overwritten below), so callers that
    // read km afterwards see the same value as gdnKvMem after gdnDecay.
    for (0..W) |b| {
        acc[b] *= g;
        vstoreF32(32, gp + b * 32, acc[b]);
    }

    // delta = beta * (v - km)
    const bp: [*]const f32 = v.ptr;
    const dp: [*]f32 = delta.ptr;
    const bb: @Vector(32, f32) = @splat(beta);
    for (0..W) |b| {
        const o = b * 32;
        vstoreF32(32, dp + o, (vloadF32(32, bp + o) - vloadF32(32, gp + o)) * bb);
    }

    // Pass 2: S[i] = S[i] * gamma + k[i] * delta (one store), out += S[i] * q[i].
    var dvec: [W]@Vector(32, f32) = undefined;
    for (0..W) |b| dvec[b] = vloadF32(32, dp + b * 32);
    var oacc: [W]@Vector(32, f32) = undefined;
    for (0..W) |b| oacc[b] = @splat(@as(f32, 0.0));
    for (0..k.len) |i| {
        const row = sp + i * vd;
        const ki: @Vector(32, f32) = @splat(k[i]);
        const qi: @Vector(32, f32) = @splat(q[i]);
        for (0..W) |b| {
            const o = b * 32;
            const r = vloadF32(32, row + o) * g + ki * dvec[b];
            vstoreF32(32, row + o, r);
            oacc[b] += r * qi;
        }
    }
    for (0..W) |b| vstoreF32(32, op + b * 32, oacc[b]);
}

/// Vectorized silu over a whole slice: x / (1 + exp(-x)).
fn siluVec(x: []f32) void {
    const p: [*]f32 = x.ptr;
    const one: @Vector(GV, f32) = @splat(1.0);
    for (0..x.len / GV) |b| {
        const o = b * GV;
        const xv = vloadF32(GV, p + o);
        vstoreF32(GV, p + o, xv / (one + @exp(-xv)));
    }
    for (x.len - x.len % GV..x.len) |i| x[i] = silu(x[i]);
}

/// out[i] *= sigmoid(g[i]) for the whole slice.
fn mulSigmoidVec(out: []f32, gate: []const f32) void {
    const one: @Vector(GV, f32) = @splat(1.0);
    const op: [*]f32 = out.ptr;
    const gp: [*]const f32 = gate.ptr;
    for (0..out.len / GV) |b| {
        const o = b * GV;
        const g = vloadF32(GV, gp + o);
        vstoreF32(GV, op + o, vloadF32(GV, op + o) * (one / (one + @exp(-g))));
    }
    for (out.len - out.len % GV..out.len) |i| out[i] *= sigmoid(gate[i]);
}

/// Activation context for the f64 reference: `x` is the serial input
/// (k == 1) or the [k, cols] row-major batch input (k > 1, out is [k, rows]).
const DumpX = struct { x: []const f32, k: usize };

/// f64 golden reference error for one matvec of any ttype: dequantize each
/// weight row with the pure Zig scalar dequant (dequantToRows' per-block
/// dequantRowFn, verified byte-identical to the C scalar kernels) and dot
/// against the ORIGINAL x in f64.  Returns
/// max_i |out[i] - ref[i]| / max_i |ref[i]|, i.e. the maximum relative error
/// of `out` against the dequantized-f32/f64 ground truth.
///
/// Reusable for tests and for validating new kernels (f32, Q8_0 and every
/// packed K/IQ ttype) without any debug-print scaffolding.
/// Negative return = reference itself not computable (unknown ttype / OOB x).
pub fn refErrF64(w: Engine.LinearW, out: []const f32, dx: DumpX, cols: usize) f64 {
    const k: usize = @max(1, dx.k);
    const rows = out.len / k;
    if (rows * k != out.len or rows == 0) return -1;
    var rowbuf: [256]f32 = undefined;
    var max_ref: f64 = 0;
    var max_err: f64 = 0;
    var vi: usize = 0;
    while (vi < out.len) : (vi += 1) {
        const t = vi / rows;
        const rr = vi % rows;
        const xoff = t * cols;
        if (xoff + cols > dx.x.len) return -3; // activation slice OOB
        var acc: f64 = 0;
        switch (w) {
            .quant => |q| {
                if (q.ttype == .q8_0) {
                    // Q8_0: 32-element blocks, f16 block scale + i8 quants.
                    const blen = q.rowBytes();
                    if (rr * blen + blen > q.data.len) return -4;
                    const wrow = q.data[rr * blen ..][0..blen];
                    for (0..q.row_blocks()) |b| {
                        const wb = wrow[b * q8_0_block_bytes ..][0..q8_0_block_bytes];
                        const d: f64 = @floatCast(@as(f16, @bitCast(std.mem.readInt(u16, wb[0..2], .little))));
                        for (0..q8_0_block_elems) |j| {
                            const wv: f64 = d * @as(f64, @as(i8, @bitCast(wb[2 + j])));
                            acc += wv * @as(f64, dx.x[xoff + b * q8_0_block_elems + j]);
                        }
                    }
                } else {
                    const bb = q.blockBytes(); // one block of this type
                    const be = q.blockElems();
                    const deq = dequantRowFnFor(q.ttype) orelse return -2;
                    for (0..q.blocks_per_row) |b| {
                        const i = rr * q.blocks_per_row + b;
                        if (i * bb + bb > q.data.len) return -4;
                        deq(q.data[i * bb ..][0..bb], rowbuf[0..be]);
                        for (0..be) |j| acc += @as(f64, rowbuf[j]) * @as(f64, dx.x[xoff + b * be + j]);
                    }
                }
            },
            .f32 => |m| {
                if (rr * cols + cols > m.len) return -4;
                const rowv = m[rr * cols ..][0..cols];
                for (0..cols) |c| acc += @as(f64, rowv[c]) * @as(f64, dx.x[xoff + c]);
            },
        }
        const e = @abs(acc - @as(f64, out[vi]));
        if (e > max_err) max_err = e;
        const a = @abs(acc);
        if (a > max_ref) max_ref = a;
    }
    if (max_ref == 0) return 0;
    return max_err / max_ref;
}

/// Row-major [rows, cols] matvec: out[r] = sum_c mat[r*cols + c] * x[c].
pub fn matvec(out: []f32, mat: []const f32, x: []const f32, rows: usize, cols: usize, pool: ?*pool_mod.Pool) void {
    std.debug.assert(mat.len == rows * cols and out.len == rows and x.len == cols);
    pool_mod.parallel(pool, .{ .f32 = .{
        .out = out.ptr,
        .mat = mat,
        .x = x.ptr,
        .x_len = cols,
        .cols = cols,
        .rows = rows,
        .k = 1,
    } });
}


// ---------------------------------------------------------------------------
// Q8_0 block format (ggml: 34 bytes per 32 elements: f16 scale + 32 x i8)
// ---------------------------------------------------------------------------

/// Block geometry of a quantized ttype: ONE source, `gguf.TensorType
/// .blockLayout()`, which mirrors ggml's `ggml_blck_size` / `ggml_type_size`
/// pair (`ggml.c` type table; per-block sizes asserted in ggml-common.h e.g.
/// `static_assert(sizeof(block_q8_0) == ...)`).  Every block-elems / block-bytes
/// number in engine.zig is derived here -- there is deliberately no
/// "row bytes" table any more: a row is `blocks_per_row * blockElemBytes()`.
///
/// Panics on a layout-less ttype: 0 was the old "unknown" sentinel and each
/// caller turned it into its own error, but every caller is either guarded by
/// `gguf.File` rejecting unknown types at load (`gguf.zig` `isKnown()`) or
/// behind an exhaustive dispatch switch, so reaching here means a real bug.
pub const blockElemsOf = gguf.blockElems;

/// Packed bytes of one block of `t` (`blockLayout().bytes`).
pub const blockElemBytes = gguf.blockBytes;

pub const q8_0_block_elems = blockElemsOf(.q8_0);
pub const q8_0_block_bytes = blockElemBytes(.q8_0);

/// Elements in one GGUF row-block: the unit `QuantRow.blocks_per_row` counts
/// for every type except Q8_0 (whose rows are plain 32-element blocks).
/// All framed types pack 8 x 32 = 256 elements per row-block; it is spelled
/// as an expression over the layout table rather than a bare 256 so that the
/// 32-element block size still has a single source.
pub const rowBlockElems = q8_0_block_elems * 8;

/// Bytes `q8_scratch` must hold to quantize a matvec INPUT of `cols` elements
/// (ceil(cols/32) blocks of 34 bytes).  Every Q8_0 matvec writes its quantized
/// input into this scratch, so the scratch is sized by the widest INPUT, never
/// by an output row.
pub fn q8ScratchBytes(cols: usize) usize {
    return (cols + q8_0_block_elems - 1) / q8_0_block_elems * q8_0_block_bytes;
}

/// Bytes `q8k_scratch` must hold to Q8_K-quantize a matvec INPUT of `cols`
/// elements (292 bytes per 256 elements: f32 scale + 256 x i8 + 16 x i16 bsums).
pub fn q8kScratchBytes(cols: usize) usize {
    const be = blockElemsOf(.q8_k);
    return (cols + be - 1) / be * blockElemBytes(.q8_k);
}

/// The widest Q8_0 matvec INPUT in a model of this shape, derived from the
/// config.  Each Q8_0 matvec quantizes its input activation into q8_scratch,
/// so the scratch must cover the widest input:
///   ffn_down        <- n_ff        (mlp_h)
///   attn_output     <- q_dim       (gout, full-attention layers)
///   ssm_out         <- value_dim    (gout, GDN layers)
///   token_embd / lm_head / in_proj / ffn_gate / ffn_up <- hidden
/// Sizing to conv_dim / 2*q_dim (the OUTPUT widths of attn_qkv / attn_q) is
/// the classic miss: n_ff is the widest input in every Qwen3.5 config
/// (27B: n_ff=17408 > conv_dim=10240 > q_dim=6144 > hidden=5120), so ffn_down
/// writes past the arena end -- assert in Debug, silent heap corruption in
/// ReleaseFast.  Same root cause as the CUDA-side 2B/4B fix (main_cuda.zig:
/// max(hidden, value_dim, q_dim, n_ff)).
pub fn maxQ8Cols(c: Config) usize {
    const n_ff: usize = c.n_ff;
    const hidden: usize = c.hidden;
    const value_dim: usize = @as(usize, c.gdn_n_heads) * c.gdn_head_dim;
    const q_dim: usize = @as(usize, c.n_heads) * c.head_dim;
    return @max(@max(n_ff, hidden), @max(value_dim, q_dim));
}

/// Anti-regression guard: every Q8_0 matvec's input length must fit in
/// q8_scratch.  Returns the widest input seen, so a miss is reported at load
/// time instead of as an out-of-bounds write inside the first matvec.
pub fn checkQ8Scratch(scratch: []const u8, cols: usize) !void {
    if (scratch.len < q8ScratchBytes(cols)) return error.Q8ScratchTooSmall;
}

// ---------------------------------------------------------------------------
// Q8_0-native matvec: weights stay in quantized blocks (zero-copy mmap
// view), dot products accumulate in i32 per block and rescale by the f16
// block scale. x is f32, quantized on the fly into a scratch buffer.
// ---------------------------------------------------------------------------

// ---------------------------------------------------------------------------
// Generic per-row dequant: a tensor of a K-quantized type is a sequence of
// fixed-size rows (each `row_bytes` bytes == 256 elements); dequantize each
// row into its 256 f32 values and append to `out`. `out.len` must equal the
// element count of the tensor (rows * 256).
// ---------------------------------------------------------------------------

const dequantRowFn = *const fn ([]const u8, []f32) void;

/// The pure-Zig scalar row dequant for a K/IQ type (null if unsupported).
fn dequantRowFnFor(ttype: gguf.TensorType) ?dequantRowFn {
    return switch (ttype) {
        .q4_0 => &quant.dequantQ4_0,
        .q4_1 => &quant.dequantQ4_1,
        .q4_k => &quant.dequantQ4K,
        .q5_k => &quant.dequantQ5K,
        .q6_k => &quant.dequantQ6K,
        .q3_k => &quant.dequantQ3K,
        .iq4_xs => &quant.dequantIQ4XS,
        .iq4_nl => &quant.dequantIQ4NL,
        .iq3_s => &quant.dequantIQ3S,
        .iq3_xxs => &quant.dequantIQ3XXS,
        .iq2_s => &quant.dequantIQ2S,
        .iq2_xxs => &quant.dequantIQ2XXS,
        .iq2_xs => &quant.dequantIQ2XS,
        .iq1_s => &quant.dequantIQ1S,
        .iq1_m => &quant.dequantIQ1M,
        .q2_k => &quant.dequantQ2K,
        else => null,
    };
}

fn dequantRowF16(row: []const u8, out: []f32) void {
    const src: [*]const u16 = @ptrCast(@alignCast(row.ptr));
    for (0..out.len) |i| out[i] = @floatCast(@as(f16, @bitCast(src[i])));
}

fn dequantRowBF16(row: []const u8, out: []f32) void {
    const src: [*]const u16 = @ptrCast(@alignCast(row.ptr));
    for (0..out.len) |i| out[i] = @as(f32, @bitCast(@as(u32, src[i]) << 16));
}

/// Dequantize a flat packed block stream into f32, one block per iteration.
/// The unit walked here is the type's own block, taken from the layout table
/// (`blockElemsOf`/`blockElemBytes`) -- the old per-type row-byte table
/// (144 for Q4_0, 512 for f16, ...) was a second, divergent copy of exactly
/// that geometry.
fn dequantToRows(alloc: std.mem.Allocator, ttype: gguf.TensorType, data: []const u8, out: []f32) !void {
    const deq: dequantRowFn = dequantRowFnFor(ttype) orelse switch (ttype) {
        .f16 => &dequantRowF16,
        .bf16 => &dequantRowBF16,
        else => return error.UnsupportedTensorType,
    };
    const row_elems = 256; // every K-quant type packs 256 elems per row
    // byte size of one such row, from the block geometry (was a per-type
    // table: 144 for Q4_0, 512 for f16, ...)
    const row_bytes: usize = row_elems / blockElemsOf(ttype) * blockElemBytes(ttype);
    const nf: usize = out.len;
    if (nf % row_elems != 0) return error.NotMultipleOfRow;
    const nrows = nf / row_elems;
    const expected_bytes = nrows * row_bytes;
    if (data.len < expected_bytes) return error.SizeMismatch;

    const row = try alloc.alloc(f32, row_elems);
    defer alloc.free(row);
    for (0..nrows) |r| {
        const in = data[r * row_bytes .. (r + 1) * row_bytes];
        deq(in, row);
        const o0 = r * row_elems;
        @memcpy(out[o0..][0..row_elems], row);
    }
}

fn loadTensorF32(alloc: std.mem.Allocator, g: *const gguf.File, name: []const u8) ![]f32 {
    const t = g.findTensor(name) orelse return error.MissingTensor;
    const n: usize = @intCast(t.nElems());
    const out = try alloc.alloc(f32, n);
    errdefer alloc.free(out);
    const data = try g.tensorData(t);
    switch (t.ttype) {
        .f32 => @memcpy(out, @as([*]const f32, @ptrCast(@alignCast(data)))[0..n]),
        .f16 => {
            const src: [*]const u16 = @ptrCast(@alignCast(data));
            for (0..n) |i| out[i] = @floatCast(@as(f16, @bitCast(src[i])));
        },
        .bf16 => {
            // BF16 is the high 16 bits of an f32: widen by shifting into the
            // top half of the significand (exact, no rounding).
            const src: [*]const u16 = @ptrCast(@alignCast(data));
            for (0..n) |i| out[i] = @as(f32, @bitCast(@as(u32, @intCast(src[i])) << 16));
        },
        .q8_0 => try quant.dequantizeQ8_0(data, out),
        .q4_0, .q4_1, .q4_k, .q5_k, .q6_k, .q3_k, .iq4_xs, .iq4_nl, .iq3_s => try dequantToRows(alloc, t.ttype, data, out),
        else => return error.UnsupportedTensorType,
    }
    return out;
}

/// Load a 2D linear weight: Q8_0 and K/IQ quant types stay in their native
/// packed format (zero-copy view into the mmap); F32/F16 load as f32.
fn loadLinear(alloc: std.mem.Allocator, g: *gguf.File, name: []const u8) !Engine.LinearW {
    const t = g.findTensor(name) orelse return error.MissingTensor;
    switch (t.ttype) {
        .q8_0, .q4_0, .q4_1, .q4_k, .q5_k, .q6_k, .q3_k, .iq4_xs, .iq4_nl, .iq3_s, .iq3_xxs, .iq2_s, .iq2_xxs, .q2_k, .iq2_xs, .iq1_s, .iq1_m => {
            return .{ .quant = try Engine.quantRowOf(g, t) };
        },
        // REVERTED: f16/bf16 used to take the zero-copy packed quant path
        // (matvecQuant + simd.d32row.f16_q8_0/bf16_q8_0).  That path produced
        // garbage output on real models -- the packed row layout (2 B/element,
        // no block metadata) was framed as 256-element blocks inconsistently
        // between matvecCols / rowDot / the q8_0 kernel, so every row wider
        // than one block was mis-strided.  Correctness first: go back to
        // materialising f32.  The 3.5x speedup is NOT worth garbled text; if we
        // want it back it needs a proper block layout + coverage tests (there
        // is currently no bf16/f16 kernel test at all).
        .f16, .bf16, .f32 => return .{ .f32 = try loadTensorF32(alloc, g, name) },
        else => return error.UnsupportedTensorType,
    }
}

// ---------------------------------------------------------------------------
// RoPE (SPEC B9/B10/B11): rotary_dim=64, cross-half pairing (i, i+32),
// cos/sin layout [c0..c31, c0..c31], inv_freq[j] = base^(-2j/64).
// Only dims 0..63 rotate; dims 64..255 pass through unchanged.
// mrope sections [11,11,10] only reorder WHICH section uses which grid
// position; for pure text all grids are identical, so sections only
// determine the frequency grouping: pairs 0..10 use pos of grid dim0,
// 11..21 grid dim1, 22..31 grid dim2 -- all == pos here (B10).
// ---------------------------------------------------------------------------

/// Rotate a 64-dim rotary block in place (cross-half pairing).
pub fn applyRope64(x: []f32, pos: u32, base: f32) void {
    const half = 32;
    // freq j uses exponent 2j/rotary_dim with rotary_dim=64
    for (0..half) |j| {
        const theta = @as(f32, @floatFromInt(pos)) * std.math.pow(f32, base, -@as(f32, @floatFromInt(2 * j)) / 64.0);
        const c = @cos(theta);
        const s = @sin(theta);
        const a = x[j];
        const b = x[j + half];
        x[j] = a * c - b * s;
        x[j + half] = a * s + b * c;
    }
}

// ---------------------------------------------------------------------------
// Engine
// ---------------------------------------------------------------------------

pub var prof_io: Io = undefined;

/// Disable the K-token prefill chunking (QWEN35_NO_BATCH); every token then
/// runs through the single-token path.
pub var no_batch = false;
/// Enable the per-phase timing report (QWEN35_PROF).
pub var prof = false;

pub fn profReport() void {
    const tot: f64 = prof_emb + prof_gdn + prof_full + prof_mlp + prof_lm;
    std.debug.print("PROF emb={d:.2}ms gdn={d:.2}ms full={d:.2}ms mlp={d:.2}ms lm={d:.2}ms total={d:.2}\n", .{ prof_emb / 1e6, prof_gdn / 1e6, prof_full / 1e6, prof_mlp / 1e6, prof_lm / 1e6, tot / 1e6 });
    pool_mod.profReport();
}

fn profNs(io: Io, t0: Io.Clock.Timestamp) f64 {
    return @as(f64, @floatFromInt(t0.durationTo(Io.Clock.Timestamp.now(io, .awake)).raw.nanoseconds));
}

pub var prof_gdn: f64 = 0;
pub var prof_full: f64 = 0;
pub var prof_mlp: f64 = 0;
pub var prof_lm: f64 = 0;
pub var prof_emb: f64 = 0;

pub const Engine = struct {
    alloc: std.mem.Allocator,
    cfg: Config,
    g: gguf.File,
    mm: std.Io.File.MemoryMap,
    file: std.Io.File,
    io: std.Io,
    /// set by `load` when it created the Threaded instance; freed in deinit
    owns_io: ?*std.Io.Threaded = null,

    token_embd: LinearW, // [vocab, hidden] row-major (embedding)
    /// untied lm_head (27B+); null in the Engine means "tied" -> use
    /// token_embd.  GGUF `output.weight` has the SAME [vocab, hidden]
    /// row-major layout as token_embd.
    output_weight: ?LinearW,
    output_norm: []f32, // [hidden]
    /// Hardware boundary: matvecs, weight loading, dequantization and the
    /// activation scratch the kernels need.  The engine never learns which
    /// implementation this is -- CPU today, CUDA later.
    exec: backend.Exec,
    layers: []LayerWeights,
    kv: []KvCache,
    gdn: []GdnState,
    /// worker count (1 = serial); set by load from `n_threads`
    n_threads: usize = 1,

    // scratch
    x: []f32, // hidden residual stream
    x1: []f32, // hidden normed input
    x2: []f32, // hidden output projection target
    qkv: []f32, // GDN conv_dim / reused for q,k,v of full layers
    z: []f32, // gate (GDN value_dim / full-attn q_dim)
    ab: []f32, // a|b (2*nv scalars)
    gdn_out: []f32, // GDN output / full-attn attention out
    q: []f32, // full-attn q
    k: []f32, // full-attn k
    v: []f32, // full-attn v
    attn: []f32, // full-attn scores [max_seq]
    mlp_h: []f32, // n_ff
    mlp_h1: []f32, // n_ff
    logits: []f32,
    delta: []f32, // 128 per-head scratch
    kv_mem: []f32, // 128
    /// Number of tokens batched per prefill chunk (K-token batched projection:
    /// each layer's weights are read once per chunk and reused across K tokens
    /// through cache).  0 => the batched path is unavailable.
    nbatch: usize = 0,
    /// K-slot scratch for the batched prefill path.  `xK[t*hidden..]` is the
    /// residual stream of token t of the current chunk; `x1K[t*hidden..]` the
    /// RMS-normed input; `qkvK/gateK/abK/qK/kK/vK` the batched projection
    /// outputs; `goutK/x2K` per-token attention/GDN output and out-projection
    /// result; `mlpK/mlpK1` the ffn intermediates; `q8kK/q8K` the per-token
    /// quantized activations for the batched matvecs.
    xK: []f32 = &.{},
    x1K: []f32 = &.{},
    qkvK: []f32 = &.{},
    gateK: []f32 = &.{},
    abK: []f32 = &.{},
    qK: []f32 = &.{},
    kK: []f32 = &.{},
    vK: []f32 = &.{},
    goutK: []f32 = &.{},
    x2K: []f32 = &.{},
    mlpK: []f32 = &.{},
    mlpK1: []f32 = &.{},
    /// Top-k order-statistic workspace (quickselect); sized vocab.
    sample_scratch: []f32 = &.{},
    /// Sampling policy applied to every produced token.  Default is greedy
    /// (temperature 0), i.e. exactly the historical behaviour; the CLI sets
    /// it from -t/-k/-s.
    sampler: Sampler = Sampler.init(.{}),
    prefill_ms: f64 = 0,

    // ---- MTP / nextn speculative decoding (blk.64, present only on 27B) ----
    /// nextn decoder-layer weights (blk.64, full-attention; same shape as a main
    /// full-attn layer).  Null when the model has no nextn layer (0.8B/2B/4B).
    nextn_layer: ?LayerWeights = null,
    /// Independent KV cache for the nextn layer (blk.64 full attention).
    nextn_kv: ?KvCache = null,
    /// nextn-specific tensors (blk.64.nextn.*).
    nextn_eh_proj: ?LinearW = null, // [2*hidden, hidden]
    nextn_enorm: ?[]f32 = null, // [hidden]
    nextn_hnorm: ?[]f32 = null, // [hidden]
    nextn_shnorm: ?[]f32 = null, // [hidden] shared_head_norm
    /// true once the model's nextn layer is loaded (27B).
    has_nextn: bool = false,
    /// scratch for nextn_forward: concat (2*hidden) + h/e/z rows (hidden each).
    nxt_concat: []f32 = &.{},
    nxt_h: []f32 = &.{},
    nxt_e: []f32 = &.{},
    nxt_z: []f32 = &.{},
    /// per-position final-normed hidden states from the last batched verify
    /// (K rows of `hidden`), plus per-position logits row for argmax.
    verify_h: []f32 = &.{}, // [K*hidden]
    verify_logits: []f32 = &.{}, // [K*vocab]
    /// GDN rollback snapshots: `mtp_snap` holds, per GDN layer, K copies of the
    /// state (delta-rule [nv*kd*kd] + conv [conv_dim*(conv_kernel-1)]) taken
    /// after each consumed token during the batched verify.
    mtp_snap_state: [][]f32 = &.{}, // [gdn_layers][K] state snapshots
    mtp_snap_conv: [][]f32 = &.{}, // [gdn_layers][K] conv snapshots
    mtp_snap_len: []u32 = &.{}, // [K] full-attn kv.len after each consumed token
    /// number of draft tokens per MTP step (N).
    mtp_n: usize = 3,


    /// Packed weight types are defined by the GGUF module (`gguf.QuantRow` is
    /// pure layout); the backend owns `LinearW` because whether a weight lives
    /// packed in host memory, expanded to f32, or uploaded to device memory is
    /// a backend decision.
    pub const QuantRow = gguf.QuantRow;
    pub const LinearW = backend.LinearW;

    /// [rows, cols] so cols = len / rows.
    fn matvecCols(w: LinearW, out_len: usize) usize {
        return switch (w) {
            .quant => |q| q.cols(),
            .f32 => |m| m.len / out_len,
        };
    }

    const LayerWeights = struct {
        attn_norm: []f32,
        post_attention_norm: []f32,
        // GDN-only
        attn_qkv: ?LinearW = null,
        attn_gate: ?LinearW = null,
        ssm_alpha: ?LinearW = null, // a (decay input)
        ssm_beta: ?LinearW = null, // b (beta input)
        ssm_conv1d: ?[]f32 = null, // F32 [4, conv_dim]: w[ch*k + t]
        ssm_dt_bias: ?[]f32 = null, // [nv], already softplus-anchored
        ssm_a: ?[]f32 = null, // [nv], already -exp(A_log)
        ssm_norm: ?[]f32 = null, // [128]
        ssm_out: ?LinearW = null,
        // full-attention-only
        attn_q: ?LinearW = null, // [1024, 4096] = per-head [q|gate] interleaved
        attn_k: ?LinearW = null,
        attn_v: ?LinearW = null,
        attn_q_norm: ?[]f32 = null, // [256]
        attn_k_norm: ?[]f32 = null, // [256]
        attn_output: ?LinearW = null,
        // shared
        ffn_gate: LinearW,
        ffn_up: LinearW,
        ffn_down: LinearW,
    };

    const KvCache = struct {
        k: []f32,
        v: []f32,
        len: u32,
    };

    /// Causal conv state: [conv_dim] history of the last K-1=3 raw samples
    /// (channel-major, same order as the C ref's per-channel rows).
    /// delta-rule state: [nv, kd, kd] row-major S[k][v].
    const GdnState = struct {
        conv: []f32,
        conv_len: u32,
        state: []f32,
    };

    pub const max_seq: u32 = 4096;

    /// Debug switch: force a SIMD tier (baseline|avx2|avx512) before load.
    /// Force a SIMD tier (QWEN35_SIMD_TIER).  Static because it is applied
    /// before the backend exists (the tier only changes which C kernels the
    /// dispatch tables point at), so it forwards to the CPU backend's setter.
    pub fn setSimdTier(name: []const u8) void {
        backend_cpu.setSimdTier(name);
    }

    pub fn load(alloc: std.mem.Allocator, path: []const u8, io: std.Io, n_threads: usize, backend_kind: backend.Kind) !Engine {
        // Bind the SIMD kernels to the widest ISA tier this CPU supports
        // (CPUID + XGETBV).  Idempotent; before this, pointers are baseline.
        simd.init();
        var nt = n_threads;
        if (nt == 0) {
            const ncpu = std.Thread.getCpuCount() catch 1;
            nt = @min(16, ncpu);
        }
        const file = std.Io.Dir.cwd().openFile(io, path, .{}) catch return error.OpenFailed;
        errdefer file.close(io);
        const stat = file.stat(io) catch return error.OpenFailed;
        const file_size: usize = @intCast(stat.size);
        var mm = std.Io.File.MemoryMap.create(io, file, .{
            .len = file_size,
            .protection = .{ .read = true },
        }) catch |e2| {
            file.close(io);
            return e2;
        };
        errdefer mm.destroy(io);

        var g = try gguf.parse(alloc, mm.memory);
        errdefer g.deinit();

        // Create the hardware backend now that the file is parsed: the CPU
        // backend keeps `&g` to resolve tensor names into mmap views.  Its
        // allocation lives in the same arena as the engine.
        const BackendImpl = backend.selectBackend(backend_kind);
        const backend_ptr = try alloc.create(BackendImpl);
        backend_ptr.* = BackendImpl.init(alloc, &g);
        const exec = backend_ptr.exec();

        var cfg = Config{};
        // block_count includes the trailing MTP/nextn prediction layer(s)
        // (27B: block_count=65, nextn=1 -> main decoder = layers 0..63).
        // The 0.8B/2B/4B files have no nextn key -> 0 -> unchanged.
        var n_nextn: u32 = 0;
        if (g.getUint("qwen35.block_count")) |v| {
            cfg.n_layers = @intCast(v);
            if (g.getUint("qwen35.nextn_predict_layers")) |n| n_nextn = @intCast(n);
        }
        if (n_nextn > 0 and n_nextn < cfg.n_layers) cfg.n_layers -= n_nextn;
        if (g.getUint("qwen35.embedding_length")) |v| cfg.hidden = @intCast(v);
        if (g.getUint("qwen35.feed_forward_length")) |v| cfg.n_ff = @intCast(v);
        if (g.getUint("qwen35.attention.head_count")) |v| cfg.n_heads = @intCast(v);
        if (g.getUint("qwen35.attention.head_count_kv")) |v| cfg.n_kv_heads = @intCast(v);
        if (g.getUint("qwen35.attention.key_length")) |v| cfg.head_dim = @intCast(v);
        if (g.getUint("qwen35.full_attention_interval")) |v| cfg.full_attention_interval = @intCast(v);
        if (g.getFloat("qwen35.attention.layer_norm_rms_epsilon")) |v| cfg.rms_eps = @floatCast(v);
        if (g.getUint("qwen35.ssm.state_size")) |v| cfg.gdn_head_dim = @intCast(v);
        if (g.getUint("qwen35.ssm.time_step_rank")) |_| {} // informational
        if (g.getUint("qwen35.ssm.conv_kernel")) |v| cfg.conv_kernel = @intCast(v);
        // qwen35.ssm.inner_size = value_dim = num_v_heads * head_dim
        if (g.getUint("qwen35.ssm.inner_size")) |v| {
            cfg.gdn_n_heads = @intCast(v / cfg.gdn_head_dim);
        }
        // qwen35.ssm.group_count = k/q head count (== v heads on 0.8B, half on 4B)
        if (g.getUint("qwen35.ssm.group_count")) |v| {
            cfg.gdn_num_k_heads = @intCast(v);
        }
        if (g.getFloat("qwen35.rope.freq_base")) |v| cfg.rope_freq_base = @floatCast(v);
        if (g.getUint("qwen35.rope.dimension_count")) |v| {
            cfg.rotary_dim = @intCast(v);
        }
        if (g.findKey("qwen35.rope.dimension_sections")) |v| {
            if (v.* == .array and v.array.element_type == .int32 and v.array.count == 4) {
                var r = gguf.arrayReader(&g, v.array);
                for (&cfg.rope_sections) |*sec| {
                    sec.* = @intCast(try gguf.arrayElemU64(&r, .int32));
                }
            }
        }
        if (g.getUint("tokenizer.ggml.eos_token_id")) |v| cfg.eos_token_id = @intCast(v);
        if (g.getUint("tokenizer.ggml.vocab_size")) |v| cfg.vocab = @intCast(v);

        var e = Engine{
            .alloc = alloc,
            .cfg = cfg,
            .g = g,
            .mm = mm,
            .file = file,
            .io = io,
            .owns_io = null,
            .token_embd = .{ .f32 = &.{} },
            .output_weight = null,
            .output_norm = &.{},
            .exec = exec,
            .layers = &.{},
            .kv = &.{},
            .gdn = &.{},
            .n_threads = 1,
            .x = &.{},
            .x1 = &.{},
            .x2 = &.{},
            .qkv = &.{},
            .z = &.{},
            .ab = &.{},
            .gdn_out = &.{},
            .q = &.{},
            .k = &.{},
            .v = &.{},
            .attn = &.{},
            .mlp_h = &.{},
            .mlp_h1 = &.{},
            .logits = &.{},
            .delta = &.{},
            .kv_mem = &.{},
        };
        errdefer e.freeAll();

        e.n_threads = nt;
        // The thread pool belongs to the backend: it is how the *hardware*
        // parallelizes a matvec, and a GPU backend has no use for it.
        exec.setThreads(io, nt);

        e.token_embd = try exec.loadLinear("token_embd.weight");
        // 27B+ has an untied lm_head (`output.weight`, same [vocab,hidden]
        // layout); 0.8B/2B/4B tie it to the embedding (key absent -> null).
        e.output_weight = exec.loadLinear("output.weight") catch |err| switch (err) {
            error.MissingTensor => null,
            else => return err,
        };
        e.output_norm = try loadTensorF32(alloc, &g, "output_norm.weight");

        const n_layers = cfg.n_layers;
        e.layers = try alloc.alloc(LayerWeights, n_layers);
        var loaded: usize = 0;
        errdefer for (e.layers[0..loaded]) |lw| freeLayerWeights(alloc, lw);
        for (0..n_layers) |li| {
            var b1: [64]u8 = undefined;
            var b2: [64]u8 = undefined;
            const idx: u32 = @intCast(li);
            const lw = LayerWeights{
                .attn_norm = try loadTensorF32(alloc, &g, std.fmt.bufPrint(&b1, "blk.{d}.attn_norm.weight", .{idx}) catch unreachable),
                .post_attention_norm = try loadTensorF32(alloc, &g, std.fmt.bufPrint(&b2, "blk.{d}.post_attention_norm.weight", .{idx}) catch unreachable),
                .ffn_gate = try exec.loadLinear(tname(&b1, "blk.{d}.ffn_gate.weight", idx)),
                .ffn_up = try exec.loadLinear(tname(&b2, "blk.{d}.ffn_up.weight", idx)),
                .ffn_down = try exec.loadLinear(tname(&b1, "blk.{d}.ffn_down.weight", idx)),
            };
            e.layers[li] = lw;
            loaded += 1;
        }

        // caches (alloc + init)
        e.kv = try alloc.alloc(KvCache, n_layers);
        e.gdn = try alloc.alloc(GdnState, n_layers);
        var inited: usize = 0;
        errdefer e.freeCachesUpTo(inited);
        const kv_dim: usize = @as(usize, cfg.n_kv_heads) * cfg.head_dim;
        const gdn_kv: usize = @as(usize, cfg.gdn_n_heads) * cfg.gdn_head_dim * cfg.gdn_head_dim;
        const gdn_conv_dim: usize =
            2 * @as(usize, cfg.gdn_num_k_heads) * cfg.gdn_head_dim +
            @as(usize, cfg.gdn_n_heads) * cfg.gdn_head_dim;
        for (0..n_layers) |li| {
            if (cfg.isFullAttention(@intCast(li))) {
                e.kv[li] = .{
                    .k = try alloc.alloc(f32, @as(usize, max_seq) * kv_dim),
                    .v = try alloc.alloc(f32, @as(usize, max_seq) * kv_dim),
                    .len = 0,
                };
                e.gdn[li] = .{ .conv = &.{}, .conv_len = 0, .state = &.{} };
            } else {
                e.kv[li] = .{ .k = &.{}, .v = &.{}, .len = 0 };
                e.gdn[li] = .{
                    .conv = try alloc.alloc(f32, gdn_conv_dim * (@as(usize, cfg.conv_kernel) - 1)),
                    .conv_len = 0,
                    .state = try alloc.alloc(f32, gdn_kv),
                };
                @memset(e.gdn[li].state, 0);
            }
            inited += 1;
        }

        // load the per-type weights now that layer type is known
        var loaded2: usize = 0;
        errdefer for (e.layers[0..loaded2]) |*lw| freeLayerTyped(alloc, lw);
        for (0..n_layers) |li| {
            const idx: u32 = @intCast(li);
            var b1: [64]u8 = undefined;
            var b2: [64]u8 = undefined;
            if (cfg.isFullAttention(@intCast(li))) {
                e.layers[li].attn_q = try exec.loadLinear(tname(&b1, "blk.{d}.attn_q.weight", idx));
                e.layers[li].attn_k = try exec.loadLinear(tname(&b2, "blk.{d}.attn_k.weight", idx));
                e.layers[li].attn_v = try exec.loadLinear(tname(&b1, "blk.{d}.attn_v.weight", idx));
                e.layers[li].attn_q_norm = try loadTensorF32(alloc, &g, tname(&b2, "blk.{d}.attn_q_norm.weight", idx));
                e.layers[li].attn_k_norm = try loadTensorF32(alloc, &g, tname(&b1, "blk.{d}.attn_k_norm.weight", idx));
                e.layers[li].attn_output = try exec.loadLinear(tname(&b2, "blk.{d}.attn_output.weight", idx));
            } else {
                e.layers[li].attn_qkv = try exec.loadLinear(tname(&b1, "blk.{d}.attn_qkv.weight", idx));
                e.layers[li].attn_gate = try exec.loadLinear(tname(&b2, "blk.{d}.attn_gate.weight", idx));
                e.layers[li].ssm_alpha = try exec.loadLinear(tname(&b1, "blk.{d}.ssm_alpha.weight", idx));
                e.layers[li].ssm_beta = try exec.loadLinear(tname(&b2, "blk.{d}.ssm_beta.weight", idx));
                e.layers[li].ssm_conv1d = try loadTensorF32(alloc, &g, tname(&b1, "blk.{d}.ssm_conv1d.weight", idx));
                e.layers[li].ssm_dt_bias = try loadTensorF32(alloc, &g, tname(&b2, "blk.{d}.ssm_dt.bias", idx));
                e.layers[li].ssm_a = try loadTensorF32(alloc, &g, tname(&b1, "blk.{d}.ssm_a", idx));
                e.layers[li].ssm_norm = try loadTensorF32(alloc, &g, tname(&b2, "blk.{d}.ssm_norm.weight", idx));
                e.layers[li].ssm_out = try exec.loadLinear(tname(&b1, "blk.{d}.ssm_out.weight", idx));
            }
            loaded2 += 1;
        }

        // scratch
        e.x = try alloc.alloc(f32, cfg.hidden);
        e.x1 = try alloc.alloc(f32, cfg.hidden);
        e.x2 = try alloc.alloc(f32, cfg.hidden);
        const gdn_nv: usize = @as(usize, cfg.gdn_n_heads);
        const gdn_kd: usize = @as(usize, cfg.gdn_head_dim);
        const gdn_value_dim: usize = gdn_nv * gdn_kd;
        // q8_scratch holds the Q8_0-quantized INPUT x of each Q8_0 matvec, so
        // its size is driven by the widest matvec INPUT (cols), not by any
        // output row: ffn_down's input is n_ff, attn_output's is q_dim,
        // ssm_out's is gdn_value_dim, token_embd / lm_head / in_proj /
        // ffn_gate / ffn_up's is hidden.
        // Sizing to conv_dim / 2*q_dim (the OUTPUT widths of attn_qkv / attn_q)
        // under-allocates ffn_down's scratch, because n_ff is the widest input
        // in every Qwen3.5 config (27B: n_ff=17408 vs conv_dim=10240,
        // q_dim=6144, hidden=5120): the quantize loop writes past the end of
        // the arena, so the assert fires in Debug and it silently corrupts the
        // heap in ReleaseFast.  Same root cause as the CUDA-side 2B/4B fix
        // (see main_cuda.zig: max(hidden, value_dim, q_dim, n_ff)).
        //
        // Size from the config, then MEASURE the widest input over the weights
        // that were actually loaded and assert the scratch covers it.  A
        // sizing miss (n_ff left out of the max, a new layer shape, a config
        // key read wrong) then fails here at load time with a named error,
        // instead of as an out-of-bounds write inside the first matvec.
        const max_q8_cols = maxQ8Cols(cfg);
        try e.exec.allocScratch(max_q8_cols);

        // measured widest input, straight from the loaded tensors
        var measured: usize = 0;
        measured = @max(measured, matvecCols(e.token_embd, cfg.vocab));
        if (e.output_weight) |ow| measured = @max(measured, matvecCols(ow, cfg.vocab));
        for (0..n_layers) |li| {
            const lw = &e.layers[li];
            if (cfg.isFullAttention(@intCast(li))) {
                const q_dim: usize = @as(usize, cfg.n_heads) * cfg.head_dim;
                const o_kv_dim: usize = @as(usize, cfg.n_kv_heads) * cfg.head_dim;
                measured = @max(measured, matvecCols(lw.attn_q.?, q_dim));
                measured = @max(measured, matvecCols(lw.attn_k.?, o_kv_dim));
                measured = @max(measured, matvecCols(lw.attn_v.?, o_kv_dim));
                measured = @max(measured, matvecCols(lw.attn_output.?, cfg.hidden));
            } else {
                const conv_dim: usize = 2 * @as(usize, cfg.gdn_num_k_heads) * cfg.gdn_head_dim + gdn_value_dim;
                measured = @max(measured, matvecCols(lw.attn_qkv.?, conv_dim));
                measured = @max(measured, matvecCols(lw.attn_gate.?, gdn_value_dim));
                measured = @max(measured, matvecCols(lw.ssm_alpha.?, gdn_nv));
                measured = @max(measured, matvecCols(lw.ssm_beta.?, gdn_nv));
                measured = @max(measured, matvecCols(lw.ssm_out.?, gdn_value_dim));
            }
            measured = @max(measured, matvecCols(lw.ffn_gate, cfg.n_ff));
            measured = @max(measured, matvecCols(lw.ffn_up, cfg.n_ff));
            measured = @max(measured, matvecCols(lw.ffn_down, cfg.hidden));
        }
        std.debug.assert(measured == max_q8_cols);

        e.qkv = try alloc.alloc(f32, @max(gdn_conv_dim, 2 * @as(usize, cfg.n_heads) * cfg.head_dim));
        e.z = try alloc.alloc(f32, @max(gdn_value_dim, @as(usize, cfg.n_heads) * cfg.head_dim));
        e.ab = try alloc.alloc(f32, 2 * gdn_nv);
        e.gdn_out = try alloc.alloc(f32, @max(gdn_value_dim, @as(usize, cfg.n_heads) * cfg.head_dim));
        e.q = try alloc.alloc(f32, @as(usize, cfg.n_heads) * cfg.head_dim);
        e.k = try alloc.alloc(f32, @as(usize, cfg.n_kv_heads) * cfg.head_dim);
        e.v = try alloc.alloc(f32, @as(usize, cfg.n_kv_heads) * cfg.head_dim);
        e.attn = try alloc.alloc(f32, max_seq);
        e.mlp_h = try alloc.alloc(f32, cfg.n_ff);
        e.mlp_h1 = try alloc.alloc(f32, cfg.n_ff);
        e.logits = try alloc.alloc(f32, cfg.vocab);
        e.delta = try alloc.alloc(f32, gdn_kd);
        e.kv_mem = try alloc.alloc(f32, gdn_kd);
        e.sample_scratch = try alloc.alloc(f32, cfg.vocab);

        // ---- K-token batched prefill scratch (K = 16) ----
        const K: usize = 16;
        const q_dimK: usize = @as(usize, cfg.n_heads) * cfg.head_dim;
        const kv_dimK: usize = @as(usize, cfg.n_kv_heads) * cfg.head_dim;
        e.nbatch = if (no_batch) 0 else K;
        e.xK = try alloc.alloc(f32, K * cfg.hidden);
        e.x1K = try alloc.alloc(f32, K * cfg.hidden);
        e.qkvK = try alloc.alloc(f32, K * @max(gdn_conv_dim, 2 * q_dimK));
        e.gateK = try alloc.alloc(f32, K * @max(gdn_value_dim, q_dimK));
        e.abK = try alloc.alloc(f32, K * 2 * gdn_nv);
        e.qK = try alloc.alloc(f32, K * q_dimK);
        e.kK = try alloc.alloc(f32, K * kv_dimK);
        e.vK = try alloc.alloc(f32, K * kv_dimK);
        e.goutK = try alloc.alloc(f32, K * @max(gdn_value_dim, q_dimK));
        e.x2K = try alloc.alloc(f32, K * cfg.hidden);
        e.mlpK = try alloc.alloc(f32, K * cfg.n_ff);
        e.mlpK1 = try alloc.alloc(f32, K * cfg.n_ff);
        try e.exec.allocScratchK(max_q8_cols, K);
        prof_io = io;
        return e;
    }

    fn tname(buf: []u8, comptime fmt: []const u8, idx: u32) []const u8 {
        return std.fmt.bufPrint(buf, fmt, .{idx}) catch unreachable;
    }


    fn freeLayerWeights(alloc: std.mem.Allocator, lw: LayerWeights) void {
        alloc.free(lw.attn_norm);
        alloc.free(lw.post_attention_norm);
        lw.ffn_gate.deinit(alloc);
        lw.ffn_up.deinit(alloc);
        lw.ffn_down.deinit(alloc);
    }

    fn freeLayerTyped(alloc: std.mem.Allocator, lw: *LayerWeights) void {
        // linear weights (q8 views point into the mmap, f32 copies are ours)
        inline for (.{
            &lw.attn_qkv, &lw.attn_gate, &lw.ssm_alpha, &lw.ssm_beta,    &lw.ssm_out,
            &lw.attn_q,   &lw.attn_k,    &lw.attn_v,    &lw.attn_output,
        }) |p| {
            if (p.*) |w| w.deinit(alloc);
        }
        // plain f32 scratch tensors
        inline for (.{
            &lw.ssm_conv1d,  &lw.ssm_dt_bias, &lw.ssm_a, &lw.ssm_norm,
            &lw.attn_q_norm, &lw.attn_k_norm,
        }) |p| {
            if (p.*) |w| alloc.free(w);
        }
    }

    fn freeCachesUpTo(self: *Engine, n: usize) void {
        // Robust to partially-constructed state: `load`'s `errdefer freeAll`
        // may run before kv/gdn are allocated (e.g. an unsupported tensor type
        // fails at loadLinear before the caches exist), in which case
        // self.kv/self.gdn are still the empty sentinel.  Clamp to the real
        // allocated length so a clean error return does not read OOB.
        const nk = @min(n, self.kv.len);
        for (self.kv[0..nk]) |c| {
            if (c.k.len > 0) self.alloc.free(c.k);
            if (c.v.len > 0) self.alloc.free(c.v);
        }
        const ng = @min(n, self.gdn.len);
        for (self.gdn[0..ng]) |s| {
            if (s.conv.len > 0) self.alloc.free(s.conv);
            if (s.state.len > 0) self.alloc.free(s.state);
        }
    }

    fn freeAll(self: *Engine) void {
        const a = self.alloc;
        self.token_embd.deinit(a);
        if (self.output_weight) |w| w.deinit(a);
        if (self.output_norm.len > 0) a.free(self.output_norm);
        self.exec.freeScratch();
        if (self.layers.len > 0) {
            for (self.layers) |lw| freeLayerWeights(a, lw);
            for (self.layers) |*lw| freeLayerTyped(a, lw);
            a.free(self.layers);
        }
        self.freeCachesUpTo(self.cfg.n_layers);
        if (self.kv.len > 0) a.free(self.kv);
        if (self.gdn.len > 0) a.free(self.gdn);
        for ([_]*[]f32{ &self.x, &self.x1, &self.x2, &self.qkv, &self.z, &self.ab, &self.gdn_out, &self.q, &self.k, &self.v, &self.attn, &self.mlp_h, &self.mlp_h1, &self.logits, &self.delta, &self.kv_mem, &self.xK, &self.x1K, &self.qkvK, &self.gateK, &self.abK, &self.qK, &self.kK, &self.vK, &self.goutK, &self.x2K, &self.mlpK, &self.mlpK1 }) |p| {
            if (p.len > 0) a.free(p.*);
        }
    }

    pub fn deinit(self: *Engine) void {
        self.freeAll();
        self.g.deinit();
        self.mm.destroy(self.io);
        self.file.close(self.io);
        // destroy the Threaded instance iff `load` created it
        if (self.owns_io) |t| {
            self.alloc.destroy(t);
            t.deinit();
        }
    }

    // -----------------------------------------------------------------
    // Forward: one token at position `pos`
    // -----------------------------------------------------------------

    /// Run the whole model for one token and return the raw logits (no
    // -----------------------------------------------------------------
    // Forward: one token at position `pos`
    // -----------------------------------------------------------------

    // Forward: one token at position `pos`
    // -----------------------------------------------------------------

    /// Run the whole model for one token and return the raw logits (no
    // -----------------------------------------------------------------
    // Forward: one token at position `pos`
    // -----------------------------------------------------------------

    /// Run the whole model for one token and return the raw logits (no
    /// sampling, no PRNG draw).  Prefill uses this so the sampler's coin
    /// stream is untouched by the prompt.
    pub fn forwardLogits(self: *Engine, token: u32, pos: u32) []f32 {
        const c = self.cfg;
        const hidden: usize = c.hidden;
        const io = self.io;
        if (prof) {
            prof_gdn = 0;
            prof_full = 0;
            prof_mlp = 0;
            prof_lm = 0;
            prof_emb = 0;
        }
        const te = Io.Clock.Timestamp.now(io, .awake);
        // embedding lookup: dequantize the token row straight into x
        self.embedTo(token, self.x[0..hidden]);
        if (prof) prof_emb += profNs(io, te);

        for (self.layers, 0..) |*lw, li| {
            if (c.isFullAttention(@intCast(li))) {
                const a = Io.Clock.Timestamp.now(io, .awake);
                self.forwardFull(lw, li, pos);
                if (prof) prof_full += profNs(io, a);
            } else {
                const a = Io.Clock.Timestamp.now(io, .awake);
                self.forwardGdn(lw, li);
                if (prof) prof_gdn += profNs(io, a);
            }
            const a = Io.Clock.Timestamp.now(io, .awake);
            self.forwardMlp(lw);
            if (prof) prof_mlp += profNs(io, a);
        }
        const b = Io.Clock.Timestamp.now(io, .awake);

        rmsNorm(self.x1, self.x, self.output_norm, c.rms_eps);
        // lm_head: untied `output.weight` when present, else tied token_embd.
        // Both are [vocab, hidden] row-major.
        if (self.output_weight) |ow| {
            switch (ow) {
                .quant, .f32 => self.exec.matvec(self.logits, ow, self.x1),
            }
        } else {
            // tied lm_head: logits = W_emb @ x1 with W_emb row-major [vocab, hidden]
            switch (self.token_embd) {
                .quant => |qrow| self.exec.matvec(self.logits, .{ .quant = qrow }, self.x1),
                .f32 => |m| {
                    const vocab: usize = c.vocab;
                    for (0..vocab) |vi| {
                        const rowv = m[vi * hidden ..][0..hidden];
                        var acc: f32 = 0;
                        for (rowv, self.x1) |mv, vv| acc += mv * vv;
                        self.logits[vi] = acc;
                    }
                },
            }
        }
        if (prof) prof_lm += profNs(io, b);
        return self.logits;
    }

    pub fn profReport() void {
        const tot: f64 = prof_gdn + prof_full + prof_mlp + prof_lm;
        std.debug.print("PROF emb={d:.2}ms gdn={d:.2}ms full={d:.2}ms mlp={d:.2}ms lm={d:.2}ms total={d:.2}\n", .{ prof_emb / 1e6, prof_gdn / 1e6, prof_full / 1e6, prof_mlp / 1e6, prof_lm / 1e6, tot / 1e6 });
    }

    /// Forward + sample one token.  Greedy (argmax) when the sampler's
    /// temperature is 0 — the historical default.
    pub fn step(self: *Engine, token: u32, pos: u32) u32 {
        const logits = self.forwardLogits(token, pos);
        return self.sampler.sample(logits, self.sample_scratch);
    }

    /// Dequantize the embedding row of `token` into `dst` (hidden wide).
    fn embedTo(self: *Engine, token: u32, dst: []f32) void {
        const hidden: usize = self.cfg.hidden;
        std.debug.assert(dst.len == hidden);
        // Expanding one row to f32 is the one place the model needs *values*
        // rather than a dot product, so it is a backend call: which
        // dequantizer runs (Zig reference on CPU, a device kernel on GPU) is
        // the backend's business.  Q8_0's 32-element framing is handled
        // there rather than special-cased here.
        switch (self.token_embd) {
            .f32 => |m| @memcpy(dst, m[token * hidden ..][0..hidden]),
            .quant => |qrow| self.exec.dequantRow(dst, qrow, token),
        }
    }

    // -----------------------------------------------------------------
    // K-token batched prefill chunk
    // -----------------------------------------------------------------

    /// Run `tokens.len` (<= K) tokens of prefill starting at position `pos0`
    /// and return the logits of the LAST token.  Every weight-reading linear
    /// projection (qkv / gate / alpha-beta / q / k / v / out-proj / ffn
    /// gate-up-down) is executed ONCE per chunk as a K-token batched matvec,
    /// so each layer's weight bytes pass through cache once for K tokens; the
    /// state kernels (GDN conv + 128x128 state, full-attention scores, KV
    /// cache writes) run token-by-token in exactly the serial order, so the
    /// KV cache and GDN state end up bit-identical to the serial prefill.
    pub fn forwardChunk(self: *Engine, tokens: []const u32, pos0: u32) []f32 {
        const k = tokens.len;
        std.debug.assert(k > 0 and k <= self.nbatch);
        if (k == 1) return self.forwardLogits(tokens[0], pos0);

        const c = self.cfg;
        const hidden: usize = c.hidden;
        const q_dim: usize = @as(usize, c.n_heads) * c.head_dim;
        const kv_dim: usize = @as(usize, c.n_kv_heads) * c.head_dim;
        const nv: usize = c.gdn_n_heads;
        const nk: usize = c.gdn_num_k_heads;
        const kd: usize = c.gdn_head_dim;
        const conv_dim: usize = 2 * nk * kd + nv * kd;
        const value_dim: usize = nv * kd;
        const n_ff: usize = c.n_ff;

        var t: usize = 0;
        while (t < k) : (t += 1) self.embedTo(tokens[t], self.xK[t * hidden ..][0..hidden]);

        for (self.layers, 0..) |*lw, li| {
            // ---- per-token RMSNorm of the residual stream, then the batched
            // projections (weight bytes read once for the whole chunk) ----
            t = 0;
            while (t < k) : (t += 1)
                rmsNorm(self.x1K[t * hidden ..][0..hidden], self.xK[t * hidden ..][0..hidden], lw.attn_norm, c.rms_eps);

            if (c.isFullAttention(@intCast(li))) {
                self.exec.matvecBatchK(self.qkvK[0 .. k * 2 * q_dim], lw.attn_q.?, self.x1K, k);
                self.exec.matvecBatchK(self.kK[0 .. k * kv_dim], lw.attn_k.?, self.x1K, k);
                self.exec.matvecBatchK(self.vK[0 .. k * kv_dim], lw.attn_v.?, self.x1K, k);
                t = 0;
                while (t < k) : (t += 1) self.forwardFullCore(lw, li, t, pos0 + @as(u32, @intCast(t)));
                self.exec.matvecBatchK(self.x2K[0 .. k * hidden], lw.attn_output.?, self.goutK, k);
            } else {
                self.exec.matvecBatchK(self.qkvK[0 .. k * conv_dim], lw.attn_qkv.?, self.x1K, k);
                self.exec.matvecBatchK(self.gateK[0 .. k * value_dim], lw.attn_gate.?, self.x1K, k);
                self.exec.matvecBatchK(self.abK[0 .. k * nv], lw.ssm_alpha.?, self.x1K, k);
                self.exec.matvecBatchK(self.abK[k * nv ..][0 .. k * nv], lw.ssm_beta.?, self.x1K, k);
                t = 0;
                while (t < k) : (t += 1) self.forwardGdnCore(lw, li, t, k);
                self.exec.matvecBatchK(self.x2K[0 .. k * hidden], lw.ssm_out.?, self.goutK, k);
            }

            // ---- per-token residual add ----
            t = 0;
            while (t < k) : (t += 1) {
                for (self.xK[t * hidden ..][0..hidden], self.x2K[t * hidden ..][0..hidden]) |*xo, v| xo.* += v;
            }

            // ---- MLP: norm, batched gate/up, per-token SwiGLU, batched down, residual ----
            t = 0;
            while (t < k) : (t += 1)
                rmsNorm(self.x1K[t * hidden ..][0..hidden], self.xK[t * hidden ..][0..hidden], lw.post_attention_norm, c.rms_eps);
            self.exec.matvecBatchK(self.mlpK[0 .. k * n_ff], lw.ffn_gate, self.x1K, k);
            self.exec.matvecBatchK(self.mlpK1[0 .. k * n_ff], lw.ffn_up, self.x1K, k);
            t = 0;
            while (t < k) : (t += 1) {
                for (self.mlpK[t * n_ff ..][0..n_ff], self.mlpK1[t * n_ff ..][0..n_ff]) |*gi, uv|
                    gi.* = silu(gi.*) * uv;
            }
            self.exec.matvecBatchK(self.x2K[0 .. k * hidden], lw.ffn_down, self.mlpK, k);
            t = 0;
            while (t < k) : (t += 1) {
                for (self.xK[t * hidden ..][0..hidden], self.x2K[t * hidden ..][0..hidden]) |*xo, v| xo.* += v;
            }
        }

        // ---- last token: final RMSNorm + lm_head on a single token (serial
        // path, so the first sampled logits are bit-identical) ----
        rmsNorm(self.x1, self.xK[(k - 1) * hidden ..][0..hidden], self.output_norm, c.rms_eps);
        if (self.output_weight) |ow| {
            switch (ow) {
                .quant, .f32 => self.exec.matvec(self.logits, ow, self.x1),
            }
        } else {
            switch (self.token_embd) {
                .quant => |qrow| self.exec.matvec(self.logits, .{ .quant = qrow }, self.x1),
                .f32 => |m| {
                    const vocab: usize = c.vocab;
                    for (0..vocab) |vi| {
                        const rowv = m[vi * hidden ..][0..hidden];
                        var acc: f32 = 0;
                        for (rowv, self.x1) |mv, vv| acc += mv * vv;
                        self.logits[vi] = acc;
                    }
                },
            }
        }
        return self.logits;
    }

    pub fn argmax(self: *Engine) u32 {
        var best: u32 = 0;
        var bv = -std.math.floatMax(f32);
        for (self.logits, 0..) |v, i| {
            if (v > bv) {
                bv = v;
                best = @intCast(i);
            }
        }
        return best;
    }

    // ------------- GDN block (SPEC A, GDN_IMPL §2) -------------

    fn forwardGdn(self: *Engine, lw: *LayerWeights, li: usize) void {
        const c = self.cfg;
        const hidden: usize = c.hidden;
        const nv: usize = c.gdn_n_heads; // value heads (0.8B: 16, 4B: 32)
        const nk: usize = c.gdn_num_k_heads; // k/q heads (0.8B: 16, 4B: 16)
        const kd: usize = c.gdn_head_dim; // 128
        const vd: usize = c.gdn_head_dim; // 128
        const conv_dim: usize = 2 * nk * kd + nv * kd; // [Q|K|V] = nk*kd + nk*kd + nv*kd
        const kq_dim: usize = nk * kd; // Q and K sections are nk*kd wide each
        const value_dim: usize = nv * vd; // 2048 (0.8B) / 4096 (4B)

        rmsNorm(self.x1, self.x, lw.attn_norm, c.rms_eps);

        // in_proj_qkv [hidden, conv_dim]: [Q|K|V] contiguous (SPEC A1);
        // Q,K are nk*kd wide, V is nv*kd wide
        const qkv = self.qkv[0..conv_dim];
        self.exec.matvec(qkv, lw.attn_qkv.?, self.x1);

        // z projection (attn_gate [hidden, value_dim])
        const z = self.z[0..value_dim];
        self.exec.matvec(z, lw.attn_gate.?, self.x1);

        // a/b projections: ssm_alpha = a (decay), ssm_beta = b (beta input),
        // one scalar per VALUE head (C ref: gdn_a/gdn_b/A/dt all sized
        // gdn_vheads; GGUF rows are [hidden, nv] on 0.8B/2B/4B alike)
        const ab = self.ab[0 .. 2 * nv];
        self.exec.matvec(ab[0..nv], lw.ssm_alpha.?, self.x1);
        self.exec.matvec(ab[nv..], lw.ssm_beta.?, self.x1);

        // causal conv1d with left-pad K-1, silu after (SPEC A2, C ref 520-556)
        // convStep handles history + silu for all channels; beta/gamma
        // are computed after, on the RAW a/b projections (not conv'd).
        const st = &self.gdn[li];
        self.convStep(st, qkv, lw.ssm_conv1d.?, c.conv_kernel);

        // beta = sigmoid(b); g = A * softplus(a + dt_bias) with A = ssm_a
        // (GGUF ssm_a is already -exp(A_log), i.e. negative, verified.)
        const beta = self.ab[0..nv]; // reuse: store beta in first half
        var gamma: [64]f32 = undefined;
        for (0..nv) |h| {
            const a_raw = ab[h];
            const b_raw = ab[nv + h];
            beta[h] = sigmoid(b_raw);
            const g = lw.ssm_a.?[h] * softplus(a_raw + lw.ssm_dt_bias.?[h]);
            gamma[h] = @exp(g); // g <= 0 -> gamma in (0,1]
        }

        const state = st.state; // [nv, kd, vd] row-major S[k][v]
        const gout = self.gdn_out[0..value_dim];
        for (0..nv) |h| {
            // GGUF v slots are in "tiled" order (converter _reorder_v_heads):
            // slot h <-> HF v head (h%nk)*nvp + h/nk, whose k head is h%nk.
            const kh = h % nk;
            const q_h = qkv[kh * kd ..][0..kd];
            const k_h = qkv[kq_dim + kh * kd ..][0..kd];
            const v_h = qkv[2 * kq_dim + h * kd ..][0..kd];
            const sh = state[h * kd * vd ..][0 .. kd * vd];

            // L2 normalize q,k (eps inside), then scale q by 1/sqrt(128)
            l2norm(q_h);
            l2norm(k_h);
            const qscale = 1.0 / @sqrt(@as(f32, @floatFromInt(kd)));
            scaleVec(q_h, qscale);

            // (1)+(2) decay & kv_mem in one pass, (3)+(4) outer update & S^T q
            // in a second pass (see gdnFused).
            const km = self.kv_mem[0..vd];
            const out_h = gout[h * vd ..][0..vd];
            gdnFused(sh, k_h, v_h, q_h, km, self.delta[0..vd], out_h, gamma[h], beta[h], vd);
        }

        // gated RMSNorm: norm -> weight -> silu(gate) (SPEC A6)
        for (0..nv) |h| {
            const out_h = gout[h * vd ..][0..vd];
            const zh = z[h * vd ..][0..vd];
            rmsNorm(out_h, out_h, lw.ssm_norm.?, c.rms_eps);
            mulSiluVec(out_h, zh);
        }

        // out projection [2048 -> 1024]
        self.exec.matvec(self.x2[0..hidden], lw.ssm_out.?, gout);
        for (0..hidden) |i| self.x[i] += self.x2[i];
    }

    /// Causal conv step: out[i] = sum_t w[i][t] * hist[i][t], where hist =
    /// [3 previous raw samples (oldest..newest), current raw sample].
    fn convStep(self: *Engine, st: *GdnState, qkv: []f32, wconv: []const f32, K: usize) void {
        _ = self;
        std.debug.assert(K == 4);
        const h: [*]f32 = st.conv.ptr;
        const w: [*]const f32 = wconv.ptr;
        const out: [*]f32 = qkv.ptr;
        for (0..qkv.len) |i| {
            const hp = h + i * 3;
            const cur = out[i];
            // 4 taps per channel in one @Vector multiply; the sum keeps the
            // exact scalar order so results are bit-identical.
            const hv: @Vector(4, f32) = .{ hp[0], hp[1], hp[2], cur };
            const wv: @Vector(4, f32) = .{ w[i * 4], w[i * 4 + 1], w[i * 4 + 2], w[i * 4 + 3] };
            const prod: [4]f32 = @bitCast(hv * wv);
            out[i] = prod[0] + prod[1] + prod[2] + prod[3];
            // shift: drop oldest, append current raw
            hp[0] = hp[1];
            hp[1] = hp[2];
            hp[2] = cur;
        }
        siluVec(qkv);
    }

    // ------------- Full attention block (SPEC B) -------------

    /// Batched-chunk twin of `forwardFull`: the q / k / v projections have
    /// already been run as K-token batched matvecs into qkvK / kK / vK; this
    /// runs the per-token attention kernel for slot `t`, writing KV at
    /// position `pos`, and leaves the gated attention output in goutK[t].
    fn forwardFullCore(self: *Engine, lw: *LayerWeights, li: usize, t: usize, pos: u32) void {
        const c = self.cfg;
        const heads: usize = c.n_heads;
        const kv_heads: usize = c.n_kv_heads;
        const hd: usize = c.head_dim;
        const q_dim: usize = heads * hd;
        const kv_dim: usize = kv_heads * hd;
        const gqa: usize = heads / kv_heads;

        const qg = self.qkvK[t * 2 * q_dim ..][0 .. 2 * q_dim];
        const q = self.qK[t * q_dim ..][0..q_dim];
        const gate = self.gateK[t * q_dim ..][0..q_dim];
        for (0..heads) |h| {
            @memcpy(q[h * hd ..][0..hd], qg[h * 2 * hd ..][0..hd]);
            @memcpy(gate[h * hd ..][0..hd], qg[h * 2 * hd + hd ..][0..hd]);
        }
        const k = self.kK[t * kv_dim ..][0..kv_dim];
        const v = self.vK[t * kv_dim ..][0..kv_dim];

        // q_norm/k_norm BEFORE RoPE (B12); GGUF weights pre-folded (1+w)
        for (0..heads) |h| {
            const qs = q[h * hd ..][0..hd];
            rmsNorm(qs, qs, lw.attn_q_norm.?, c.rms_eps);
        }
        for (0..kv_heads) |h| {
            const ks = k[h * hd ..][0..hd];
            rmsNorm(ks, ks, lw.attn_k_norm.?, c.rms_eps);
        }
        for (0..heads) |h| applyRope64(q[h * hd ..][0..hd], pos, c.rope_freq_base);
        for (0..kv_heads) |h| applyRope64(k[h * hd ..][0..hd], pos, c.rope_freq_base);

        // KV cache: slot == cache.len, exactly as the serial path
        const cache = &self.kv[li];
        std.debug.assert(cache.len < max_seq);
        const slot: usize = cache.len;
        @memcpy(cache.k[slot * kv_dim ..][0..kv_dim], k);
        @memcpy(cache.v[slot * kv_dim ..][0..kv_dim], v);
        cache.len += 1;

        const scale = 1.0 / @sqrt(@as(f32, @floatFromInt(hd)));
        const gout = self.goutK[t * q_dim ..][0..q_dim];
        for (0..heads) |h| {
            const kvh = h / gqa;
            const qs = q[h * hd ..][0..hd];
            const att = self.attn[0 .. slot + 1];
            for (0..slot + 1) |ti| {
                const ks = cache.k[ti * kv_dim + kvh * hd ..][0..hd];
                var acc: f32 = 0;
                for (qs, ks) |a, b| acc += a * b;
                att[ti] = acc * scale;
            }
            softmax(att);
            var accv: [256]f32 = @splat(0);
            for (0..slot + 1) |ti| {
                const w = att[ti];
                const vs = cache.v[ti * kv_dim + kvh * hd ..][0..hd];
                for (vs, 0..) |vv, j| accv[j] += w * vv;
            }
            @memcpy(gout[h * hd ..][0..hd], accv[0..hd]);
        }
        mulSigmoidVec(gout[0..q_dim], gate[0..q_dim]);
    }

    /// Batched-chunk twin of `forwardGdn`: qkv / gate / alpha / beta were
    /// batched into qkvK / gateK / abK; this runs the per-token conv + GDN
    /// state kernel for slot `t` (updating the shared conv history and 128x128
    /// state in serial order), writing the gated RMS-normed output to goutK[t].
    fn forwardGdnCore(self: *Engine, lw: *LayerWeights, li: usize, t: usize, k: usize) void {
        const c = self.cfg;
        const nv: usize = c.gdn_n_heads;
        const nk: usize = c.gdn_num_k_heads;
        const kd: usize = c.gdn_head_dim;
        const vd: usize = c.gdn_head_dim;
        const conv_dim: usize = 2 * nk * kd + nv * kd;
        const kq_dim: usize = nk * kd;
        const value_dim: usize = nv * vd;

        const qkv = self.qkvK[t * conv_dim ..][0..conv_dim];
        const z = self.gateK[t * value_dim ..][0..value_dim];
        // forwardChunk wrote alpha as abK[0..k*nv] (token t at t*nv) and beta
        // as abK[k*nv..2*k*nv] (token t at k*nv + t*nv) — two half-regions,
        // matching the two linMatvecBatchK calls.  Read them back in exactly
        // that layout; beta is stored in place (same order as the serial path).
        const ab_a = self.abK[t * nv ..][0..nv];
        const beta = self.abK[k * nv + t * nv ..][0..nv];
        const st = &self.gdn[li];
        self.convStep(st, qkv, lw.ssm_conv1d.?, c.conv_kernel);
        var gamma: [64]f32 = undefined;
        for (0..nv) |h| {
            const a_raw = ab_a[h];
            const b_raw = beta[h];
            beta[h] = sigmoid(b_raw);
            const g = lw.ssm_a.?[h] * softplus(a_raw + lw.ssm_dt_bias.?[h]);
            gamma[h] = @exp(g);
        }

        const state = st.state;
        const gout = self.goutK[t * value_dim ..][0..value_dim];
        for (0..nv) |h| {
            const kh = h % nk;
            const q_h = qkv[kh * kd ..][0..kd];
            const k_h = qkv[kq_dim + kh * kd ..][0..kd];
            const v_h = qkv[2 * kq_dim + h * kd ..][0..kd];
            const sh = state[h * kd * vd ..][0 .. kd * vd];

            l2norm(q_h);
            l2norm(k_h);
            const qscale = 1.0 / @sqrt(@as(f32, @floatFromInt(kd)));
            scaleVec(q_h, qscale);

            const km = self.kv_mem[0..vd];
            const out_h = gout[h * vd ..][0..vd];
            gdnFused(sh, k_h, v_h, q_h, km, self.delta[0..vd], out_h, gamma[h], beta[h], vd);
        }

        for (0..nv) |h| {
            const out_h = gout[h * vd ..][0..vd];
            const zh = z[h * vd ..][0..vd];
            rmsNorm(out_h, out_h, lw.ssm_norm.?, c.rms_eps);
            mulSiluVec(out_h, zh);
        }
    }

    fn forwardFull(self: *Engine, lw: *LayerWeights, li: usize, pos: u32) void {
        const c = self.cfg;
        const hidden: usize = c.hidden;
        const heads: usize = c.n_heads;
        const kv_heads: usize = c.n_kv_heads;
        const hd: usize = c.head_dim; // 256
        const q_dim: usize = heads * hd; // 2048
        const kv_dim: usize = kv_heads * hd; // 512
        const gqa: usize = heads / kv_heads; // 4

        rmsNorm(self.x1, self.x, lw.attn_norm, c.rms_eps);

        // q_proj [1024, 4096]: view(8, 512) then chunk -> per-head [q|gate]
        const qg = self.qkv[0 .. 2 * q_dim];
        self.exec.matvec(qg, lw.attn_q.?, self.x1);
        const q = self.q[0..q_dim];
        const gate = self.z[0..q_dim];
        for (0..heads) |h| {
            // interleave: q(256) = qg[h*512 .. h*512+256], gate = next 256
            @memcpy(q[h * hd ..][0..hd], qg[h * 2 * hd ..][0..hd]);
            @memcpy(gate[h * hd ..][0..hd], qg[h * 2 * hd + hd ..][0..hd]);
        }
        self.exec.matvec(self.k[0..kv_dim], lw.attn_k.?, self.x1);
        self.exec.matvec(self.v[0..kv_dim], lw.attn_v.?, self.x1);

        // q_norm/k_norm BEFORE RoPE (B12); GGUF weights pre-folded (1+w)
        for (0..heads) |h| {
            const qs = q[h * hd ..][0..hd];
            rmsNorm(qs, qs, lw.attn_q_norm.?, c.rms_eps);
        }
        for (0..kv_heads) |h| {
            const ks = self.k[h * hd ..][0..hd];
            rmsNorm(ks, ks, lw.attn_k_norm.?, c.rms_eps);
        }
        // RoPE on first 64 dims of each q/k head
        for (0..heads) |h| {
            applyRope64(q[h * hd ..][0..hd], pos, c.rope_freq_base);
        }
        for (0..kv_heads) |h| {
            applyRope64(self.k[h * hd ..][0..hd], pos, c.rope_freq_base);
        }

        // KV cache
        const cache = &self.kv[li];
        std.debug.assert(cache.len < max_seq);
        const slot: usize = cache.len;
        @memcpy(cache.k[slot * kv_dim ..][0..kv_dim], self.k[0..kv_dim]);
        @memcpy(cache.v[slot * kv_dim ..][0..kv_dim], self.v[0..kv_dim]);
        cache.len += 1;

        // attention: kv = h // gqa (B13), scale = hd^-0.5
        const scale = 1.0 / @sqrt(@as(f32, @floatFromInt(hd)));
        const gout = self.gdn_out[0..q_dim];
        for (0..heads) |h| {
            const kvh = h / gqa;
            const qs = q[h * hd ..][0..hd];
            const att = self.attn[0 .. slot + 1];
            for (0..slot + 1) |t| {
                const ks = cache.k[t * kv_dim + kvh * hd ..][0..hd];
                var acc: f32 = 0;
                for (qs, ks) |a, b| acc += a * b;
                att[t] = acc * scale;
            }
            softmax(att);
            var accv: [256]f32 = @splat(0);
            for (0..slot + 1) |t| {
                const w = att[t];
                const vs = cache.v[t * kv_dim + kvh * hd ..][0..hd];
                for (vs, 0..) |vv, j| accv[j] += w * vv;
            }
            @memcpy(gout[h * hd ..][0..hd], accv[0..hd]);
        }

        // sigmoid gate (B14), then o_proj [2048 -> 1024]
        mulSigmoidVec(gout[0..q_dim], gate[0..q_dim]);
        self.exec.matvec(self.x2[0..hidden], lw.attn_output.?, gout);
        for (0..hidden) |i| self.x[i] += self.x2[i];
    }

    // ------------- MLP (SPEC C16) -------------

    fn forwardMlp(self: *Engine, lw: *LayerWeights) void {
        const c = self.cfg;
        const hidden: usize = c.hidden;
        rmsNorm(self.x1, self.x, lw.post_attention_norm, c.rms_eps);
        self.exec.matvec(self.mlp_h, lw.ffn_gate, self.x1);
        self.exec.matvec(self.mlp_h1, lw.ffn_up, self.x1);
        for (0..c.n_ff) |i| self.mlp_h[i] = silu(self.mlp_h[i]) * self.mlp_h1[i];
        self.exec.matvec(self.x2[0..hidden], lw.ffn_down, self.mlp_h);
        for (0..hidden) |i| self.x[i] += self.x2[i];
    }

    // -----------------------------------------------------------------
    // Generation
    // -----------------------------------------------------------------

    pub fn reset(self: *Engine) void {
        for (self.kv) |*cache| cache.len = 0;
        for (self.gdn) |*st| {
            if (st.conv.len > 0) @memset(st.conv, 0);
            if (st.state.len > 0) @memset(st.state, 0);
            st.conv_len = 0;
        }
    }

    pub const GenResult = struct {
        tokens: []u32,
        prefill_ms: f64,
        /// average wall time per generated token (ms)
        ms_per_token: f64,
        /// total wall time spent in decode steps, forward passes only (ms)
        decode_ms: f64,
        /// true when generation stopped because the next sampled token was EOS
        hit_eos: bool,
    };

    /// Prefill prompt (forward-only, no PRNG draws), then generate up to
    /// max_tokens tokens with `self.sampler` (greedy argmax when its
    /// temperature is 0 — the historical default), stopping at EOS.  The
    /// first generated token is sampled from the last prefill position, so
    /// the PRNG stream starts at the same point on CPU and CUDA.  Caller
    /// owns result.tokens.
    pub fn generate(
        self: *Engine,
        alloc: std.mem.Allocator,
        prompt: []const u32,
        max_tokens: usize,
        progress: ?*const fn (ctx: ?*anyopaque, tok: u32) void,
        progress_ctx: ?*anyopaque,
    ) !GenResult {
        // Restart the PRNG at its seed so every `generate` call (and every
        // process started with the same -s) samples identically.  The prompt
        // itself never draws (forwardLogits below), so the first coin spent
        // is the first generated token — on CPU and CUDA alike.
        self.sampler.reseed();
        self.reset();

        const io = self.io;
        // Profiling covers prefill too: prof_phase used to be set only after
        // the prefill loop, so every prefill sample reported calls=0 and the
        // batched kernels -- exactly the part worth profiling -- were invisible.
        pool_mod.profReset();
        pool_mod.prof_phase = true;
        const t_start = Io.Clock.Timestamp.now(io, .awake);
        var last_logits: []f32 = &.{};
        if (self.nbatch == 0) {
            for (prompt, 0..) |t, i| last_logits = self.forwardLogits(t, @intCast(i));
        } else {
            // K-token batched prefill: every layer's weights are read once per
            // chunk (batched projections) and reused across the chunk's K
            // tokens; the KV cache / GDN state updates happen token-by-token in
            // the same order as the serial loop, so the resulting logits of the
            // last prompt token are bit-identical to the serial prefill.
            var i: usize = 0;
            while (i < prompt.len) {
                const n = @min(self.nbatch, prompt.len - i);
                last_logits = self.forwardChunk(prompt[i .. i + n], @intCast(i));
                i += n;
            }
        }
        const t_after = Io.Clock.Timestamp.now(io, .awake);
        self.prefill_ms = @as(f64, @floatFromInt(t_start.durationTo(t_after).raw.nanoseconds)) / 1e6;

        var out: std.ArrayList(u32) = .empty;
        errdefer out.deinit(alloc);
        var pos: u32 = @intCast(prompt.len);
        var gen_ns: i96 = 0;
        var hit_eos = false;
        // prof_phase stays on across prefill and decode (set above); do not
        // reset here or the prefill samples are thrown away.
        var next: u32 = self.sampler.sample(last_logits, self.sample_scratch);
        while (pos < max_tokens) : (pos += 1) {
            if (next == self.cfg.eos_token_id) {
                hit_eos = true;
                break;
            }
            try out.append(alloc, next);
            if (progress) |f| f(progress_ctx, next);
            const t0 = Io.Clock.Timestamp.now(io, .awake);
            next = self.step(next, pos);
            gen_ns += t0.durationTo(Io.Clock.Timestamp.now(io, .awake)).raw.nanoseconds;
        }
        pool_mod.prof_phase = false;
        const per: f64 = if (out.items.len > 0)
            @as(f64, @floatFromInt(gen_ns)) / 1e6 / @as(f64, @floatFromInt(out.items.len))
        else
            0;
        return .{
            .tokens = try out.toOwnedSlice(alloc),
            .prefill_ms = self.prefill_ms,
            .ms_per_token = per,
            .decode_ms = @as(f64, @floatFromInt(gen_ns)) / 1e6,
            .hit_eos = hit_eos,
        };
    }
};

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

test "rms norm: x*rsqrt(mean(x^2)+eps)*w" {
    const w = [_]f32{ 1, 1 };
    var out: [2]f32 = undefined;
    rmsNorm(&out, &.{ 3, 4 }, &w, 0);
    const inv = 1.0 / @sqrt(12.5);
    try std.testing.expectApproxEqAbs(@as(f32, 3 * inv), out[0], 1e-6);
    try std.testing.expectApproxEqAbs(@as(f32, 4 * inv), out[1], 1e-6);
    const w2 = [_]f32{ 2, 0.5 };
    rmsNorm(&out, &.{ 3, 4 }, &w2, 0);
    try std.testing.expectApproxEqAbs(@as(f32, 3 * inv * 2), out[0], 1e-6);
    try std.testing.expectApproxEqAbs(@as(f32, 4 * inv * 0.5), out[1], 1e-6);
}

test "silu / sigmoid / softplus reference values" {
    try std.testing.expectApproxEqAbs(@as(f32, 0), silu(0), 1e-6);
    try std.testing.expectApproxEqAbs(@as(f32, 0.7310586), silu(1), 1e-5);
    try std.testing.expectApproxEqAbs(@as(f32, -0.26894143), silu(-1), 1e-5);
    try std.testing.expectApproxEqAbs(@as(f32, 0.5), sigmoid(0), 1e-6);
    try std.testing.expectApproxEqAbs(@as(f32, @log(@as(f32, 2))), softplus(0), 1e-6);
    try std.testing.expectApproxEqAbs(@as(f32, 25), softplus(25), 1e-5); // threshold branch
}

test "softmax sums to one and is shift-stable" {
    var x = [_]f32{ 1, 2, 1000, -5 };
    softmax(&x);
    var sum: f32 = 0;
    for (x) |v| sum += v;
    try std.testing.expectApproxEqAbs(@as(f32, 1), sum, 1e-5);
    try std.testing.expect(x[2] > 0.99);
}

test "l2norm eps inside the sum" {
    var x = [_]f32{ 3, 4 };
    l2norm(&x);
    // 1/sqrt(25 + 1e-6): almost exactly 0.2
    try std.testing.expectApproxEqAbs(@as(f32, 3 * 0.2), x[0], 1e-6);
    try std.testing.expectApproxEqAbs(@as(f32, 4 * 0.2), x[1], 1e-6);
}

test "rope64: pos 0 identity, cross-half pairing, tail untouched" {
    var x: [256]f32 = undefined;
    for (&x, 0..) |*p, i| p.* = @floatFromInt(i);
    applyRope64(x[0..64], 0, 1e7);
    for (0..64) |i| try std.testing.expectApproxEqAbs(@as(f32, @floatFromInt(i)), x[i], 1e-4);
    for (64..256) |i| try std.testing.expectEqual(@as(f32, @floatFromInt(i)), x[i]);

    // pos 1: pair (0, 32) rotates by theta=1 (inv_freq[0]=1)
    var y: [256]f32 = undefined;
    for (&y, 0..) |*p, i| p.* = @floatFromInt(i);
    applyRope64(y[0..64], 1, 1e7);
    // SPEC B10 cos/sin table: cos(theta=1)=0.54030234, sin=0.84147096
    // y[0] = 0*cos - 32*sin ; y[32] = 0*sin + 32*cos
    try std.testing.expectApproxEqAbs(@as(f32, -32 * 0.84147096), y[0], 1e-3);
    try std.testing.expectApproxEqAbs(@as(f32, 32 * 0.54030234), y[32], 1e-3);
    // pair (1, 33): theta = 0.60429639 (inv_freq[1])
    const c1: f32 = 0.82290208;
    const s1: f32 = 0.56818324;
    try std.testing.expectApproxEqAbs(1 * c1 - 33 * s1, y[1], 1e-3);
    try std.testing.expectApproxEqAbs(1 * s1 + 33 * c1, y[33], 1e-3);
}

test "rope64: inv_freq[31] matches 1e7^-62/64" {
    // pair (31,63) rotates with inv_freq[31] = 1e7^(-62/64) = 1.6548171e-07
    var x: [64]f32 = undefined;
    @memset(&x, 0);
    x[31] = 1;
    applyRope64(&x, 1000000, 1e7);
    const expected_cos = @cos(@as(f32, 1000000) * 1.6548171e-07);
    try std.testing.expectApproxEqAbs(expected_cos, x[31], 1e-4);
    // also check pos=0 gives cos(theta0)=cos(1e6) for the pair (0,32)
    var y: [64]f32 = undefined;
    @memset(&y, 0);
    y[0] = 1;
    applyRope64(&y, 1000000, 1e7);
    try std.testing.expectApproxEqAbs(@cos(@as(f32, 1000000)), y[0], 1e-4);
}

test "config layer typing" {
    const c = Config{};
    try std.testing.expect(!c.isFullAttention(0));
    try std.testing.expect(!c.isFullAttention(2));
    try std.testing.expect(c.isFullAttention(3));
    try std.testing.expect(c.isFullAttention(23));
}

test "matvec 2x3 hand computed" {
    const m = [_]f32{ 1, 2, 3, 4, 5, 6 };
    const x = [_]f32{ 7, 8, 9 };
    var out: [2]f32 = undefined;
    matvec(&out, &m, &x, 2, 3, null);
    try std.testing.expectEqual(@as(f32, 50), out[0]);
    try std.testing.expectEqual(@as(f32, 122), out[1]);
}

test "convStep: causal left-pad semantics" {
    // single channel with taps [a0,a1,a2,a3]; feed three tokens manually
    const alloc = std.testing.allocator;
    const conv = try alloc.alloc(f32, 4);
    defer alloc.free(conv);
    @memset(conv, 0);
    var st = Engine.GdnState{ .conv = conv, .conv_len = 0, .state = &.{} };
    var eng = Engine{
        .alloc = alloc,
        .cfg = .{},
        .g = undefined,
        .mm = undefined,
        .file = undefined,
        .io = undefined,
        .token_embd = .{ .f32 = &.{} },
        .output_weight = null,
        .output_norm = &.{},
        .exec = undefined,
        .layers = &.{},
        .kv = &.{},
        .gdn = &.{},
        .x = &.{},
        .x1 = &.{},
        .x2 = &.{},
        .qkv = &.{},
        .z = &.{},
        .ab = &.{},
        .gdn_out = &.{},
        .q = &.{},
        .k = &.{},
        .v = &.{},
        .attn = &.{},
        .mlp_h = &.{},
        .mlp_h1 = &.{},
        .logits = &.{},
        .delta = &.{},
        .kv_mem = &.{},
        .prefill_ms = 0,
    };
    // taps: w = [1, 0.5, 0.25, 2]
    const wc = [_]f32{ 1, 0.5, 0.25, 2 };
    var qkv = [_]f32{1};
    eng.convStep(&st, &qkv, &wc, 4);
    // token0: state all zero -> out = 1*2 = 2, silu(2)
    try std.testing.expectApproxEqAbs(@as(f32, 2.0 / (1.0 + @exp(@as(f32, -2)))), qkv[0], 1e-5);
    qkv[0] = 10; // token1
    eng.convStep(&st, &qkv, &wc, 4);
    // window: [0,0,1(raw t0),10] -> 0*1+0*.5+1*.25+10*2 = 20.25
    try std.testing.expectApproxEqAbs(silu(@as(f32, 20.25)), qkv[0], 1e-4);
    qkv[0] = 100; // token2
    eng.convStep(&st, &qkv, &wc, 4);
    // window: [0,1,10,100] -> 0+0.5+2.5+200 = 203
    try std.testing.expectApproxEqAbs(silu(@as(f32, 203)), qkv[0], 1e-3);
}

test "gdn delta rule one step, hand computed" {
    // 1 head, kd=2, vd=2. S starts 0.
    // q=[1,0], k=[0,1], v=[3,4], beta=0.5, gamma=0.9
    // (1) S *= 0.9 -> 0
    // (2) kv_mem = S^T k = [0,0]
    // (3) delta = 0.5*[3,4] = [1.5,2]; S += k outer delta = [[0,0],[1.5,2]]
    // (4) o = S^T q = row0 = [0,0]
    var state: [4]f32 = @splat(0);
    for (&state) |*p| p.* *= 0.9;
    const k = [_]f32{ 0, 1 };
    var delta: [2]f32 = undefined;
    var km: [2]f32 = undefined;
    @memset(&km, 0);
    for (0..2) |i| {
        for (0..2) |j| km[j] += state[i * 2 + j] * k[i];
    }
    const v = [_]f32{ 3, 4 };
    const beta: f32 = 0.5;
    for (0..2) |j| delta[j] = (v[j] - km[j]) * beta;
    for (0..2) |i| {
        for (0..2) |j| state[i * 2 + j] += k[i] * delta[j];
    }
    const q = [_]f32{ 1, 0 };
    var o: [2]f32 = undefined;
    for (0..2) |j| {
        var sum: f32 = 0;
        for (0..2) |i| sum += state[i * 2 + j] * q[i];
        o[j] = sum;
    }
    try std.testing.expectEqual(@as(f32, 0), o[0]);
    try std.testing.expectEqual(@as(f32, 0), o[1]);
    try std.testing.expectEqual(@as(f32, 0), state[0]);
    try std.testing.expectEqual(@as(f32, 0), state[1]);
    try std.testing.expectEqual(@as(f32, 1.5), state[2]);
    try std.testing.expectEqual(@as(f32, 2), state[3]);

    // second token with q=[0,1]: o = S^T q = row1 = [1.5,2]
    for (0..2) |j| {
        var sum: f32 = 0;
        for (0..2) |i| sum += state[i * 2 + j] * k[i]; // reuse k=[0,1] as q
        o[j] = sum;
    }
    try std.testing.expectEqual(@as(f32, 1.5), o[0]);
    try std.testing.expectEqual(@as(f32, 2), o[1]);
}

test "q8ScratchBytes: ceil(cols/32) blocks of 34 bytes" {
    try std.testing.expectEqual(@as(usize, 34), q8ScratchBytes(32));
    try std.testing.expectEqual(@as(usize, 34), q8ScratchBytes(1));
    try std.testing.expectEqual(@as(usize, 68), q8ScratchBytes(33));
    try std.testing.expectEqual(@as(usize, 68), q8ScratchBytes(64));
    // 27B ffn_down input: n_ff = 17408 = 544 blocks
    try std.testing.expectEqual(@as(usize, 544 * 34), q8ScratchBytes(17408));
}

test "maxQ8Cols: widest matvec INPUT is n_ff, not conv_dim / 2*q_dim" {
    // 27B (read from the GGUF metadata): n_ff=17408 > conv_dim=10240 >
    // q_dim=value_dim=6144 > hidden=5120
    const c27 = Config{
        .hidden = 5120,
        .n_heads = 24,
        .head_dim = 256,
        .n_ff = 17408,
        .gdn_n_heads = 48, // ssm.inner_size 6144 / state_size 128
        .gdn_head_dim = 128,
        .gdn_num_k_heads = 16, // ssm.group_count
    };
    // the old (buggy) sizing: conv_dim = 2*16*128 + 48*128 = 10240
    const conv_dim27 = 2 * @as(usize, c27.gdn_num_k_heads) * c27.gdn_head_dim +
        @as(usize, c27.gdn_n_heads) * c27.gdn_head_dim;
    try std.testing.expectEqual(@as(usize, 10240), conv_dim27);
    try std.testing.expect(maxQ8Cols(c27) > conv_dim27); // guard: n_ff must win
    try std.testing.expectEqual(@as(usize, 17408), maxQ8Cols(c27));

    // 0.8B defaults: n_ff=3584 is the widest INPUT (conv_dim=6144 is an
    // OUTPUT width of attn_qkv, whose input is hidden=1024 -- sizing the
    // scratch to conv_dim over-allocates here and under-allocates on 27B).
    const c08 = Config{};
    try std.testing.expectEqual(@as(usize, 3584), maxQ8Cols(c08));
    const conv_dim08 = 2 * @as(usize, c08.gdn_num_k_heads) * c08.gdn_head_dim +
        @as(usize, c08.gdn_n_heads) * c08.gdn_head_dim;
    try std.testing.expectEqual(@as(usize, 6144), conv_dim08);
    try std.testing.expect(conv_dim08 > maxQ8Cols(c08)); // old sizing: too big
}

test "checkQ8Scratch: rejects a scratch too small for the widest input" {
    const alloc = std.testing.allocator;
    const need = q8ScratchBytes(17408);
    const good = try alloc.alloc(u8, need);
    defer alloc.free(good);
    try checkQ8Scratch(good, 17408);
    // the old conv_dim-based sizing (10240 cols) is 2448 bytes short
    const bad = try alloc.alloc(u8, q8ScratchBytes(10240));
    defer alloc.free(bad);
    try std.testing.expectError(error.Q8ScratchTooSmall, checkQ8Scratch(bad, 17408));
    // exact fit is accepted, one byte short is not
    try std.testing.expectError(error.Q8ScratchTooSmall, checkQ8Scratch(good[0 .. need - 1], 17408));
}

test "matvecCols: reads input width from Q8_0 / K-IQ / f32 weights" {
    // Q8_0: blocks_per_row * 32 (native 32-element blocks)
    const q8 = Engine.LinearW{ .quant = .{ .data = &.{}, .ttype = .q8_0, .rows = 5120, .blocks_per_row = 544 } };
    try std.testing.expectEqual(@as(usize, 17408), Engine.matvecCols(q8, 5120));
    // K/IQ: blocks_per_row * 256 (native 256-element blocks)
    const q = Engine.LinearW{ .quant = .{ .data = &.{}, .ttype = .q4_k, .rows = 5120, .blocks_per_row = 68 } };
    try std.testing.expectEqual(@as(usize, 17408), Engine.matvecCols(q, 5120));
    // f32: flat [rows, cols] -> len / rows (small shape, 4 rows x 256 cols)
    var f32buf: [4 * 256]f32 = undefined;
    const f = Engine.LinearW{ .f32 = &f32buf };
    try std.testing.expectEqual(@as(usize, 256), Engine.matvecCols(f, 4));
}
