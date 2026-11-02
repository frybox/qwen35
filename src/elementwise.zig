//! Elementwise / state kernels: the numeric core the engine used to contain.
//!
//! Every function below is a BYTE-EXACT copy of the original code in engine.zig
//! (ranges recorded per function).  The loops, @Vector widths, tail-scalar
//! loops and accumulation order are load-bearing: they are what makes the CPU
//! path bit-identical to the historical engine.  Do not 'optimize', reorder,
//! or add @fast here.  `convStepKernel` is the body of Engine.convStep.
//!
//! backend_cpu.zig registers these as the elementwise slots of Exec, so the
//! engine reaches them through `self.exec.*` and never dereferences a Buf.

const std = @import("std");
const backend = @import("backend.zig");
const sampler = @import("sampler.zig");
const Buf = backend.Buf;

// Thin Buf adapters: the numeric kernels below take plain slices; these
// wrappers are what backend_cpu registers as the elementwise Exec slots.
pub fn bufRmsnorm(out: Buf, x: Buf, w: Buf, eps: f32) void { rmsNorm(out.slice(), x.slice(), w.slice(), eps); }
pub fn bufRope64(x: Buf, pos: u32, base: f32) void { applyRope64(x.slice(), pos, base); }
pub fn bufL2norm(x: Buf) void { l2norm(x.slice()); }
pub fn bufScaleVec(x: Buf, s: f32) void { scaleVec(x.slice(), s); }
pub fn bufSiluInPlace(x: Buf) void { siluVec(x.slice()); }
pub fn bufMulSilu(out: Buf, gate: Buf) void { mulSiluVec(out.slice(), gate.slice()); }
pub fn bufMulSigmoid(out: Buf, gate: Buf) void { mulSigmoidVec(out.slice(), gate.slice()); }
pub fn bufSoftmax(x: Buf) void { softmax(x.slice()); }
pub fn bufAddResid(dst: Buf, src: Buf) void { for (dst.slice(), src.slice()) |*d, v| d.* += v; }
pub fn bufSwiglu(gate: Buf, up: Buf) void { for (gate.slice(), up.slice()) |*g, u| g.* = silu(g.*) * u; }
pub fn bufGdnStep(state: Buf, k: Buf, v: Buf, q: Buf, km: Buf, delta: Buf, out: Buf, gamma: f32, beta: f32, vd: usize) void {
    gdnFused(state.slice(), k.slice(), v.slice(), q.slice(), km.slice(), delta.slice(), out.slice(), gamma, beta, vd);
}
pub fn bufConvStep(conv: Buf, y: Buf, wconv: Buf, K: usize) void {
    convStepKernel(conv.ptr, y.slice(), wconv.slice(), K);
}

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
pub fn sigmoidVec(x: []f32) void {
    const one: @Vector(GV, f32) = @splat(1.0);
    const p: [*]f32 = x.ptr;
    for (0..x.len / GV) |b| {
        const o = b * GV;
        vstoreF32(GV, p + o, one / (one + @exp(-vloadF32(GV, p + o))));
    }
}

/// out[i] *= silu(g[i]) for the whole slice.
pub fn mulSiluVec(out: []f32, gate: []const f32) void {
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
pub fn scaleVec(x: []f32, s: f32) void {
    const sv: @Vector(GV, f32) = @splat(s);
    const p: [*]f32 = x.ptr;
    for (0..x.len / GV) |b| {
        const o = b * GV;
        vstoreF32(GV, p + o, vloadF32(GV, p + o) * sv);
    }
    for (x.len - x.len % GV..x.len) |i| x[i] *= s;
}

/// (1) S *= gamma.
pub fn gdnDecay(sh: []f32, gamma: f32) void {
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
pub fn gdnKvMem(sh: []const f32, k: []const f32, km: []f32, vd: usize) void {
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
pub fn gdnOuter(sh: []f32, k: []const f32, v: []const f32, km: []const f32, delta: []f32, beta: f32, vd: usize) void {
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
pub fn gdnStateOut(sh: []const f32, q: []const f32, out: []f32, vd: usize) void {
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
pub inline fn gdnFused(sh: []f32, k: []const f32, v: []const f32, q: []const f32, km: []f32, delta: []f32, out: []f32, gamma: f32, beta: f32, vd: usize) void {
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
pub fn siluVec(x: []f32) void {
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
pub fn mulSigmoidVec(out: []f32, gate: []const f32) void {
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

/// Byte-exact copy of Engine.applyRope64 (engine.zig L506-518).
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

/// `conv` is the per-channel sliding history (conv_dim*3 floats, oldest..newest),
/// `y` is the raw projection row (conv_dim floats), updated in place.
pub fn convStepKernel(conv: [*]f32, y: []f32, wconv: []const f32, K: usize) void {
    std.debug.assert(K == 4);
    const h: [*]f32 = conv;
    const w: [*]const f32 = wconv.ptr;
    const out: [*]f32 = y.ptr;
    for (0..y.len) |i| {
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
    siluVec(y);
}


// Byte-exact copies of the attention/gamma bodies the engine used to run inline.
// `kcache`/`vcache` are this layer's KV rows (kv_dim wide per token, slot+1
// tokens already written).  Scores -> softmax -> PV accumulate -> gout in the
// original order, then the sigmoid gate, exactly as forwardFull did.
pub fn attnDecodeKernel(q: []const f32, gate: []const f32, kcache: []const f32, vcache: []const f32,
                        att: []f32, gout: []f32, slot: usize, heads: usize, gqa: usize,
                        hd: usize, kv_dim: usize, scale: f32) void {
    for (0..heads) |h| {
        const kvh = h / gqa;
        const qs = q[h * hd ..][0..hd];
        const scores = att[0 .. slot + 1];
        for (0..slot + 1) |t| {
            const ks = kcache[t * kv_dim + kvh * hd ..][0..hd];
            var acc: f32 = 0;
            for (qs, ks) |a, b| acc += a * b;
            scores[t] = acc * scale;
        }
        softmax(scores);
        var accv: [256]f32 = @splat(0);
        for (0..slot + 1) |t| {
            const w = scores[t];
            const vs = vcache[t * kv_dim + kvh * hd ..][0..hd];
            for (vs, 0..) |vv, j| accv[j] += w * vv;
        }
        @memcpy(gout[h * hd ..][0..hd], accv[0..hd]);
    }
    mulSigmoidVec(gout, gate);
}

/// Byte-exact copy of the beta/gamma loop of Engine.forwardGdn (engine.zig
/// L1392-1398): beta = sigmoid(b) written in place, gamma = exp(A*softplus(a+dt_bias)).
/// `a_raw`/`b_raw` are separate slices: the serial path passes the two halves of
/// one `ab` row, the batched path passes `abK[t*nv..]` and `abK[k*nv + t*nv..]`.
pub fn gdnGammaKernel(a_raw: []const f32, b_raw: []f32, ssm_a: []const f32, dt_bias: []const f32, gamma: []f32, nv: usize) void {
    for (0..nv) |h| {
        b_raw[h] = sigmoid(b_raw[h]);
        const g = ssm_a[h] * softplus(a_raw[h] + dt_bias[h]);
        gamma[h] = @exp(g); // g <= 0 -> gamma in (0,1]
    }
}

/// Copy `n` floats: a device backend maps this to cudaMemcpy; the host copy is
/// the CPU case.  Used for KV-cache writes and the MTP state snapshots.
pub fn copyKernel(dst: []f32, src: []const f32) void {
    @memcpy(dst, src);
}

/// Greedy argmax over the logits.  The backend runs it wherever the logits
/// live; on a device backend this is a 248320-element argmax kernel and the
/// only thing that crosses PCIe is the 4-byte token id.
pub fn argmaxKernel(logits: []const f32) u32 {
    var best: u32 = 0;
    var bv: f32 = -std.math.inf(f32);
    for (logits, 0..) |v, i| {
        if (v > bv) {
            bv = v;
            best = @intCast(i);
        }
    }
    return best;
}

pub fn bufAttnDecode(q: Buf, gate: Buf, kcache: Buf, vcache: Buf, att: Buf, gout: Buf,
                     slot: usize, heads: usize, gqa: usize, hd: usize, kv_dim: usize, scale: f32) void {
    attnDecodeKernel(q.slice(), gate.slice(), kcache.slice(), vcache.slice(), att.slice(),
        gout.slice(), slot, heads, gqa, hd, kv_dim, scale);
}

pub fn bufGdnGamma(a_raw: Buf, b_raw: Buf, ssm_a: Buf, dt_bias: Buf, gamma: Buf, nv: usize) void {
    gdnGammaKernel(a_raw.slice(), b_raw.slice(), ssm_a.slice(), dt_bias.slice(), gamma.slice(), nv);
}

pub fn bufCopyRange(dst: Buf, src: Buf) void {
    @memcpy(dst.slice(), src.slice());
}

/// Scalar reference for the q/gate split.  This is the definition both
/// backends must agree with: per head `h`, `q[h*hd..][hd]` is `src[h*2*hd..]`
/// and `gate[h*hd..][hd]` is the `hd` block right after it.
///
/// Deliberately written as the plain per-head loop -- it is the oracle the
/// SIMD/device versions are checked against, so it must stay obviously correct
/// rather than fast.
pub fn bufDeinterleaveQG(q: Buf, gate: Buf, src: Buf, heads: usize, hd: usize) void {
    for (0..heads) |h| {
        @memcpy(q.slice()[h * hd ..][0..hd], src.slice()[h * 2 * hd ..][0..hd]);
        @memcpy(gate.slice()[h * hd ..][0..hd], src.slice()[h * 2 * hd + hd ..][0..hd]);
    }
}

pub fn bufCopyFromHost(dst: Buf, src: []const f32) void {
    @memcpy(dst.slice(), src);
}

pub fn bufArgmax(logits: Buf) u32 {
    return argmaxKernel(logits.slice());
}

pub fn bufSample(logits: Buf, scratch: Buf, sp: *sampler.Sampler) u32 {
    return sp.sample(logits.slice(), scratch.slice());
}

test "bufDeinterleaveQG matches the per-head copyRange form it replaced" {
    // The rewrite that introduced `deinterleaveQG` replaced, per head,
    //   q[h*hd..]    <- qg[h*2*hd..]
    //   gate[h*hd..] <- qg[h*2*hd + hd..]
    // with one batched call.  This test pins the two to each other: the naive
    // loop below is literally the old sequence of `@memcpy`s, so a change to
    // the batched kernel (or to a backend slot above it) that reorders heads,
    // swaps the halves, or drops the gate block fails here.
    const alloc = std.testing.allocator;
    const heads = 6;
    const hd = 5; // deliberately not a power of two
    const n = heads * hd;

    var qg = try alloc.alloc(f32, 2 * n);
    defer alloc.free(qg);
    const q = try alloc.alloc(f32, n);
    defer alloc.free(q);
    const gate = try alloc.alloc(f32, n);
    defer alloc.free(gate);
    // Distinct value per element so any misplaced element is detectable.
    for (qg, 0..) |*v, i| v.* = @floatFromInt(i * 3 + 1);

    // `var`: the slices themselves are never rebound, but they are the copy
    // *destinations* below -- `@memcpy` needs a mutable target.
    const want_q = try alloc.alloc(f32, n);
    defer alloc.free(want_q);
    const want_gate = try alloc.alloc(f32, n);
    defer alloc.free(want_gate);

    for (0..heads) |h| {
        @memcpy(want_q[h * hd ..][0..hd], qg[h * 2 * hd ..][0..hd]);
        @memcpy(want_gate[h * hd ..][0..hd], qg[h * 2 * hd + hd ..][0..hd]);
    }

    bufDeinterleaveQG(
        Buf.wrap(q),
        Buf.wrap(gate),
        Buf.wrap(qg),
        heads,
        hd,
    );

    try std.testing.expectEqualSlices(f32, want_q, q);
    try std.testing.expectEqualSlices(f32, want_gate, gate);
}
