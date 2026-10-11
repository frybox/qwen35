//! CUDA host for the Qwen3.5 GGUF engine.
//!
//! Compiled with `zig cc -c` into an object file, then linked by `nvcc` with
//! `src/engine_cuda.cu` (the CUDA kernels).  The .cu's `int main()` forwards
//! to the exported `q35_main(argc, argv)` below, which drives the exact same
//! forward-pass the CPU engine does (src/engine.zig) — but every operator is
//! a `q35_*` CUDA launcher, so the GPU binary is bit-exact with the CPU one
//! (same greedy argmax, same prompt => same tokens).
//!
//! Weights are reused straight from `engine.Engine.load` (guaranteed
//! identical GGUF parsing / dims); each weight's bytes are copied to the
//! GPU once at load.  All scratch / state lives in persistent GPU buffers.

const std = @import("std");
const engine = @import("engine.zig");
const tokenizer = @import("tokenizer.zig");
const Io = std.Io;

// ---------------------------------------------------------------------------
// extern "C" launchers defined in src/engine_cuda.cu
// ---------------------------------------------------------------------------

extern "C" fn q35_cuda_init() c_int;
extern "C" fn q35_cuda_sync() c_int;
extern "C" fn q35_malloc(n: usize) ?*anyopaque;
extern "C" fn q35_free(p: ?*anyopaque) c_int;
extern "C" fn q35_h2d(d: ?*anyopaque, h: ?*anyopaque, n: usize) c_int;
extern "C" fn q35_d2h(h: ?*anyopaque, d: ?*anyopaque, n: usize) c_int;
extern "C" fn q35_d2d(d: ?*anyopaque, s: ?*anyopaque, n: usize) c_int;
extern "C" fn q35_zero(d: ?*anyopaque, n: usize) c_int;
extern "C" fn q35_quantize_q8(x: ?*anyopaque, out: ?*anyopaque, nblocks: c_int) void;
extern "C" fn q35_matvec_q8(w: ?*anyopaque, xq: ?*anyopaque, out: ?*anyopaque, rows: c_int, row_blocks: c_int) void;
extern "C" fn q35_matvec_f32(w: ?*anyopaque, x: ?*anyopaque, out: ?*anyopaque, rows: c_int, cols: c_int) void;
extern "C" fn q35_embed_row(w: ?*anyopaque, x: ?*anyopaque, n: c_int) void;
extern "C" fn q35_rmsnorm_rows(x: ?*anyopaque, w: ?*anyopaque, out: ?*anyopaque, rows: c_int, n: c_int, eps: f32) void;
extern "C" fn q35_rmsnorm_strided(x: ?*anyopaque, w: ?*anyopaque, out: ?*anyopaque, rows: c_int, n: c_int, stride: c_int, eps: f32) void;
extern "C" fn q35_l2norm(x: ?*anyopaque, rows: c_int, n: c_int, eps: f32, scale: f32) void;
extern "C" fn q35_rope(x: ?*anyopaque, rows: c_int, hd: c_int, rope_dim: c_int, stride: c_int, base: f32) void;
extern "C" fn q35_conv1d(wconv: ?*anyopaque, x: ?*anyopaque, hist: ?*anyopaque, conv_dim: c_int, k: c_int) void;
extern "C" fn q35_beta_gamma(a: ?*anyopaque, b: ?*anyopaque, abar: ?*anyopaque, dtb: ?*anyopaque, beta: ?*anyopaque, gamma: ?*anyopaque, nv: c_int) void;
extern "C" fn q35_gdn_state(qkv: ?*anyopaque, state: ?*anyopaque, beta: ?*anyopaque, gamma: ?*anyopaque, out: ?*anyopaque, nv: c_int, nk: c_int, kd: c_int, vd: c_int) void;
extern "C" fn q35_gated_rmsnorm(out: ?*anyopaque, w: ?*anyopaque, z: ?*anyopaque, rows: c_int, n: c_int, eps: f32) void;
extern "C" fn q35_attn(q: ?*anyopaque, kc: ?*anyopaque, vc: ?*anyopaque, gate: ?*anyopaque, out: ?*anyopaque, heads: c_int, kv_heads: c_int, hd: c_int, kv_dim: c_int, q_stride: c_int, scale: f32) void;
extern "C" fn q35_silu_mul(h: ?*anyopaque, h1: ?*anyopaque, n: c_int) void;
extern "C" fn q35_sigmoid_mul(o: ?*anyopaque, g: ?*anyopaque, n: c_int) void;
extern "C" fn q35_add(a: ?*anyopaque, b: ?*anyopaque, n: c_int) void;
extern "C" fn q35_argmax(x: ?*anyopaque, outidx: ?*anyopaque, n: c_int) void;
extern "C" fn q35_kvstore(kvcache: ?*anyopaque, src: ?*anyopaque, kv_dim: c_int) void;

// CUDA Graph API (kernel.cu).  token/pos are uploaded to a device param block;
// the whole-forward graph is captured once (Zig drives the forward under
// capture_begin/end) and replayed each step via q35_graph_launch.
extern "C" fn q35_graph_set_params(token: c_int, pos: c_int) c_int;
extern "C" fn q35_capture_begin() c_int;
extern "C" fn q35_capture_end() c_int;
extern "C" fn q35_graph_launch() c_int;
extern "C" fn q35_graph_destroy() c_int;

// ---------------------------------------------------------------------------
// pointer helpers
// ---------------------------------------------------------------------------

fn f32p(p: ?*anyopaque) *f32 {
    return @alignCast(@ptrCast(p.?));
}
/// const C source ptr -> non-const anyopaque for q35_h2d (const slice .ptr).
fn hsrcC(p: [*]const u8) ?*anyopaque {
    return @ptrCast(@constCast(p));
}
fn hsrcC32(p: [*]const f32) ?*anyopaque {
    return @ptrCast(@constCast(p));
}
fn u8p(p: ?*anyopaque) *u8 {
    return @ptrCast(p.?);
}
/// byte-offset a ?*anyopaque device pointer by `off` bytes.
fn offBytes(p: ?*anyopaque, off: usize) ?*anyopaque {
    const b: [*]u8 = @constCast(@ptrCast(p.?));
    const nb: *u8 = @ptrFromInt(@intFromPtr(b) + off);
    return @ptrCast(nb);
}
fn cf32(p: ?*anyopaque) ?*anyopaque {
    return p;
}
fn cu8(p: ?*anyopaque) ?*anyopaque {
    return p;
}
fn toC(n: usize) c_int {
    return @intCast(n);
}
fn mx(a: usize, b: usize) usize {
    return if (a > b) a else b;
}
fn toU32(n: usize) u32 {
    return @intCast(n);
}

// ---------------------------------------------------------------------------
// GPU-side state (one instance per model load)
// ---------------------------------------------------------------------------

const LinearGpu = struct {
    ptr: ?*anyopaque = null, // q8 raw blocks, or f32 matrix
    rows: usize = 0, // output dim
    cols: usize = 0, // input dim
    is_q8: bool = true,
};

const GpuLayer = struct {
    attn_norm: ?*anyopaque = null,
    post_attention_norm: ?*anyopaque = null,
    attn_qkv: LinearGpu = .{},
    attn_gate: LinearGpu = .{},
    ssm_alpha: LinearGpu = .{},
    ssm_beta: LinearGpu = .{},
    ssm_conv1d: ?*anyopaque = null,
    ssm_dt_bias: ?*anyopaque = null,
    ssm_a: ?*anyopaque = null,
    ssm_norm: ?*anyopaque = null,
    ssm_out: LinearGpu = .{},
    attn_q: LinearGpu = .{},
    attn_k: LinearGpu = .{},
    attn_v: LinearGpu = .{},
    attn_q_norm: ?*anyopaque = null,
    attn_k_norm: ?*anyopaque = null,
    attn_output: LinearGpu = .{},
    ffn_gate: LinearGpu = .{},
    ffn_up: LinearGpu = .{},
    ffn_down: LinearGpu = .{},
};

const Gpu = struct {
    cfg: engine.Config = undefined,
    n_layers: usize = 0,
    hidden: usize = 0,
    n_ff: usize = 0,
    heads: usize = 0,
    kv_heads: usize = 0,
    hd: usize = 0,
    q_dim: usize = 0,
    kv_dim: usize = 0,
    nv: usize = 0,
    nk: usize = 0,
    kd: usize = 0,
    vd: usize = 0,
    conv_dim: usize = 0,
    value_dim: usize = 0,
    K: usize = 0,
    max_seq: usize = 0,
    vocab: usize = 0,

    token_embd: ?*anyopaque = null,
    emb_is_q8: bool = true,
    output_norm: ?*anyopaque = null,

    layers: []GpuLayer = &.{},
    gdn_conv: []?*anyopaque = &.{},
    gdn_state: []?*anyopaque = &.{},
    kv_k: []?*anyopaque = &.{},
    kv_v: []?*anyopaque = &.{},
    kv_len: []u32 = &.{},

    x: ?*anyopaque = null,
    x1: ?*anyopaque = null,
    x2: ?*anyopaque = null,
    qkv: ?*anyopaque = null,
    z: ?*anyopaque = null,
    ab: ?*anyopaque = null,
    gdn_out: ?*anyopaque = null,
    k: ?*anyopaque = null,
    v: ?*anyopaque = null,
    attn: ?*anyopaque = null,
    mlp_h: ?*anyopaque = null,
    mlp_h1: ?*anyopaque = null,
    logits: ?*anyopaque = null,
    q8_scratch: ?*anyopaque = null,
    argmax_buf: ?*anyopaque = null,
    beta_buf: ?*anyopaque = null,
    gamma_buf: ?*anyopaque = null,
    emb_row: []f32 = &.{},

    gpu_ptrs: std.ArrayList(*anyopaque) = .empty,

    fn allocF32(self: *Gpu, arena: std.mem.Allocator, n: usize) !?*anyopaque {
        const p = q35_malloc(n * @sizeOf(f32)) orelse return error.GpuAlloc;
        self.gpu_ptrs.append(arena, p) catch return error.GpuAlloc;
        return p;
    }
    fn allocU8(self: *Gpu, arena: std.mem.Allocator, n: usize) !?*anyopaque {
        const p = q35_malloc(n) orelse return error.GpuAlloc;
        self.gpu_ptrs.append(arena, p) catch return error.GpuAlloc;
        return p;
    }
    fn deinit(self: *Gpu, arena: std.mem.Allocator) void {
        // Drop the CUDA Graph first (it references the device pointers below).
        _ = q35_graph_destroy();
        for (self.gpu_ptrs.items) |p| {
            _ = q35_free(p);
        }
        self.gpu_ptrs.deinit(arena);
    }
};

// ---------------------------------------------------------------------------
// upload helpers (CPU weight bytes -> GPU, one-time)
// ---------------------------------------------------------------------------

fn uploadF32(arena: std.mem.Allocator, gpu: *Gpu, arr: []const f32) !?*anyopaque {
    const n: usize = arr.len;
    const d = q35_malloc(n * 4) orelse return error.GpuAlloc;
    gpu.gpu_ptrs.append(arena, d) catch return error.GpuAlloc;
    _ = q35_h2d(d, hsrcC32(arr.ptr), n * 4);
    return d;
}

fn uploadLinear(arena: std.mem.Allocator, gpu: *Gpu, lw: engine.Engine.LinearW, rows: usize, cols: usize) !LinearGpu {
    switch (lw) {
        .q8 => |row| {
            const n: usize = row.data.len;
            const d = q35_malloc(n) orelse return error.GpuAlloc;
            gpu.gpu_ptrs.append(arena, d) catch return error.GpuAlloc;
            _ = q35_h2d(d, hsrcC(row.data.ptr), n);
            return .{ .ptr = d, .rows = rows, .cols = cols, .is_q8 = true };
        },
        .f32 => |arr| {
            const n: usize = arr.len;
            const d = q35_malloc(n * 4) orelse return error.GpuAlloc;
            gpu.gpu_ptrs.append(arena, d) catch return error.GpuAlloc;
            _ = q35_h2d(d, hsrcC32(arr.ptr), n * 4);
            return .{ .ptr = d, .rows = rows, .cols = cols, .is_q8 = false };
        },
    }
}

// ---------------------------------------------------------------------------
// matvec: out[rows] = W[rows, cols] * x[cols]; x is f32, quantized to Q8_0
// first (matches CPU linMatvec / matvecQ8_0 exactly).
// ---------------------------------------------------------------------------

fn linMatvec(gpu: *Gpu, w: *const LinearGpu, x: ?*anyopaque, out: ?*anyopaque) void {
    const rows = w.rows;
    const cols = w.cols;
    if (w.is_q8) {
        const row_blocks = cols / 32;
        q35_quantize_q8(cf32(x), gpu.q8_scratch, toC(row_blocks));
        q35_matvec_q8(cu8(w.ptr), cu8(gpu.q8_scratch), out, toC(rows), toC(row_blocks));
    } else {
        q35_matvec_f32(cf32(w.ptr), cf32(x), out, toC(rows), toC(cols));
    }
}

// ---------------------------------------------------------------------------
// per-layer forward passes (mirror src/engine.zig exactly, bit-exact)
// ---------------------------------------------------------------------------

const Ctx = struct {
    gpu: *Gpu,
    cfg: engine.Config,
    hidden: usize,
    n_ff: usize,
    heads: usize,
    kv_heads: usize,
    hd: usize,
    nv: usize,
    nk: usize,
    kd: usize,
    vd: usize,
    conv_dim: usize,
    K: usize,
};

// Build the CUDA Graph for one whole forward.  Runs runForward under stream
// capture so every kernel launch (all on the graph stream) is recorded as a
// graph node.  token/pos are irrelevant here (recorded as pointers/dims, not
// values); the param block is reset to (0,0) by q35_capture_end, and each real
// step calls q35_graph_set_params before q35_graph_launch.
fn captureGraph(ctx: *Ctx) void {
    _ = q35_capture_begin();
    runForward(ctx);
    _ = q35_capture_end();
}

fn makeCtx(g: *Gpu) Ctx {
    return .{
        .gpu = g,
        .cfg = g.cfg,
        .hidden = g.hidden,
        .n_ff = g.n_ff,
        .heads = g.heads,
        .kv_heads = g.kv_heads,
        .hd = g.hd,
        .nv = g.nv,
        .nk = g.nk,
        .kd = g.kd,
        .vd = g.vd,
        .conv_dim = g.conv_dim,
        .K = g.K,
    };
}

fn forwardGdn(ctx: *Ctx, lw: *GpuLayer, li: usize) void {
    const c = ctx.cfg;
    const hidden = ctx.hidden;
    const nv = ctx.nv;
    const nk = ctx.nk;
    const kd = ctx.kd;
    const vd = ctx.vd;
    const conv_dim = ctx.conv_dim;
    const kq_dim = nk * kd;
    const K = ctx.K;

    const g = ctx.gpu;

    // rmsNorm(x1, x, attn_norm, eps)
    q35_rmsnorm_rows(cf32(g.x), cf32(lw.attn_norm), g.x1, 1, toC(hidden), c.rms_eps);

    // qkv = attn_qkv @ x1   (conv_dim)
    linMatvec(g, &lw.attn_qkv, g.x1, g.qkv);
    // z = attn_gate @ x1     (value_dim)
    linMatvec(g, &lw.attn_gate, g.x1, g.z);
    // ab[0..nv] = ssm_alpha @ x1 ; ab[nv..] = ssm_beta @ x1
    linMatvec(g, &lw.ssm_alpha, g.x1, g.ab);
    const ab_b = offBytes(g.ab, nv * @sizeOf(f32));
    linMatvec(g, &lw.ssm_beta, g.x1, ab_b);

    // causal conv1d (in place qkv, per-layer history; silu in-kernel)
    const conv_hist = g.gdn_conv[li];
    q35_conv1d(cf32(lw.ssm_conv1d), g.qkv, conv_hist, toC(conv_dim), toC(K));

    // beta = sigmoid(b); gamma = exp(ssm_a * softplus(a + dt_bias))
    const ab_b2 = offBytes(g.ab, nv * @sizeOf(f32));
    q35_beta_gamma(cf32(g.ab), ab_b2, cf32(lw.ssm_a), cf32(lw.ssm_dt_bias), g.beta_buf, g.gamma_buf, toC(nv));

    // l2norm + qscale on the Q section (in place); K section l2norm only
    const qscale = 1.0 / std.math.sqrt(@as(f32, @floatFromInt(kd)));
    q35_l2norm(g.qkv, toC(nk), toC(kd), 1e-6, qscale);
    q35_l2norm(offBytes(g.qkv, kq_dim * @sizeOf(f32)), toC(nk), toC(kd), 1e-6, 1.0);

    // GDN state update (reads qkv, in-place state, writes gdn_out)
    const state = g.gdn_state[li];
    q35_gdn_state(g.qkv, state, cf32(g.beta_buf), cf32(g.gamma_buf), g.gdn_out, toC(nv), toC(nk), toC(kd), toC(vd));

    // gated rmsNorm: gdn_out = rmsnorm(gdn_out) * silu(z) * ssm_norm
    q35_gated_rmsnorm(g.gdn_out, cf32(lw.ssm_norm), cf32(g.z), toC(nv), toC(vd), c.rms_eps);

    // x += ssm_out @ gdn_out
    linMatvec(g, &lw.ssm_out, g.gdn_out, g.x2);
    // x += x2 (on device)
    q35_add(g.x, g.x2, toC(hidden));
}

fn forwardFull(ctx: *Ctx, lw: *GpuLayer, li: usize) void {
    const c = ctx.cfg;
    const hidden = ctx.hidden;
    const heads = ctx.heads;
    const kv_heads = ctx.kv_heads;
    const hd = ctx.hd;
    const kv_dim = kv_heads * hd;

    const g = ctx.gpu;

    // rmsNorm(x1, x, attn_norm, eps)
    q35_rmsnorm_rows(cf32(g.x), cf32(lw.attn_norm), g.x1, 1, toC(hidden), c.rms_eps);

    // qg = attn_q @ x1  (per-head [q|gate] interleaved in qkv)
    linMatvec(g, &lw.attn_q, g.x1, g.qkv);
    // k = attn_k @ x1 ; v = attn_v @ x1
    linMatvec(g, &lw.attn_k, g.x1, g.k);
    linMatvec(g, &lw.attn_v, g.x1, g.v);

    // gate section = qg + hd (in f32); rmsnorm_strided/attn take ?*anyopaque
    const gate = offBytes(g.qkv, hd * @sizeOf(f32));

    // q_norm / k_norm BEFORE RoPE (GGUF weights pre-folded 1+w), interleaved
    q35_rmsnorm_strided(g.qkv, cf32(lw.attn_q_norm), g.qkv, toC(heads), toC(hd), toC(2 * hd), c.rms_eps);
    q35_rmsnorm_strided(g.k, cf32(lw.attn_k_norm), g.k, toC(kv_heads), toC(hd), toC(hd), c.rms_eps);

    // RoPE on first rotary_dim dims of each q/k head (pos from device params)
    const rope_dim: usize = @intCast(c.rotary_dim);
    const base: f32 = c.rope_freq_base;
    q35_rope(g.qkv, toC(heads), toC(hd), toC(rope_dim), toC(2 * hd), base);
    q35_rope(g.k, toC(kv_heads), toC(hd), toC(rope_dim), toC(hd), base);

    // KV cache write.  The destination slot is `pos`, read by the kernel from
    // the device param block (a cudaMemcpy node can't express a pos-dependent
    // offset), so no host bookkeeping is needed here.
    q35_kvstore(g.kv_k[li], g.k, toC(kv_dim));
    q35_kvstore(g.kv_v[li], g.v, toC(kv_dim));

    // GQA attention (sigmoid gate applied in-kernel)
    const scale = 1.0 / std.math.sqrt(@as(f32, @floatFromInt(hd)));
    q35_attn(g.qkv, g.kv_k[li], g.kv_v[li], gate, g.gdn_out, toC(heads), toC(kv_heads), toC(hd), toC(kv_dim), toC(2 * hd), scale);

    // x += attn_output @ gdn_out
    linMatvec(g, &lw.attn_output, g.gdn_out, g.x2);
    // x += x2 (on device)
    q35_add(g.x, g.x2, toC(hidden));
}

fn forwardMlp(ctx: *Ctx, lw: *GpuLayer) void {
    const c = ctx.cfg;
    const hidden = ctx.hidden;
    const n_ff = ctx.n_ff;

    const g = ctx.gpu;

    // rmsNorm(x1, x, post_attention_norm, eps)
    q35_rmsnorm_rows(cf32(g.x), cf32(lw.post_attention_norm), g.x1, 1, toC(hidden), c.rms_eps);
    // mlp_h = ffn_gate @ x1 ; mlp_h1 = ffn_up @ x1
    linMatvec(g, &lw.ffn_gate, g.x1, g.mlp_h);
    linMatvec(g, &lw.ffn_up, g.x1, g.mlp_h1);
    // mlp_h = silu(mlp_h) * mlp_h1
    q35_silu_mul(g.mlp_h, cf32(g.mlp_h1), toC(n_ff));
    // x += ffn_down @ mlp_h
    linMatvec(g, &lw.ffn_down, g.mlp_h, g.x2);
    // x += x2 (on device)
    q35_add(g.x, g.x2, toC(hidden));
}

// The whole forward as a sequence of kernel launches (no host sync, no d2h/d2d
// — every q35_* below goes to the graph stream).  Run under stream capture to
// build the graph; conceptually "what the graph replays".  token/pos are NOT
// arguments: embed/rope/attn/kvstore read them from the device param block.
fn runForward(ctx: *Ctx) void {
    const c = ctx.cfg;
    const hidden = ctx.hidden;
    const g = ctx.gpu;

    // embedding lookup (row = token, read from device params)
    if (g.emb_is_q8) {
        q35_embed_row(cu8(g.token_embd), g.x, toC(hidden));
    } else {
        // f32 token_embd would need a host-bounce (not capturable).  Q8_0
        // models (the ones we ship) use the q8 path above; assert the f32
        // path is never taken while building a graph.
        std.debug.assert(false);
        q35_embed_row(cu8(g.token_embd), g.x, toC(hidden));
    }

    for (g.layers, 0..) |*lw, li| {
        if (c.isFullAttention(@intCast(li))) {
            forwardFull(ctx, lw, li);
        } else {
            forwardGdn(ctx, lw, li);
        }
        forwardMlp(ctx, lw);
    }

    // final rmsNorm + tied lm_head (logits = token_embd @ x1)
    q35_rmsnorm_rows(cf32(g.x), cf32(g.output_norm), g.x1, 1, toC(hidden), c.rms_eps);
    const vocab: usize = @intCast(c.vocab);
    if (g.emb_is_q8) {
        const row_blocks = hidden / 32;
        q35_quantize_q8(cf32(g.x1), g.q8_scratch, toC(row_blocks));
        q35_matvec_q8(cu8(g.token_embd), cu8(g.q8_scratch), g.logits, toC(vocab), toC(row_blocks));
    } else {
        q35_matvec_f32(cf32(g.token_embd), cf32(g.x1), g.logits, toC(vocab), toC(hidden));
    }

    // argmax (strict >, first max)
    q35_argmax(g.logits, g.argmax_buf, toC(vocab));
}

// One decode step via the CUDA Graph: upload (token,pos) to the device param
// block, launch the whole-forward graph (one launch replaces ~900 host
// launches), sync, and read back the argmax token.
fn step(ctx: *Ctx, token: u32, pos: u32) u32 {
    const g = ctx.gpu;
    _ = q35_graph_set_params(@as(c_int, @intCast(token)), @as(c_int, @intCast(pos)));
    _ = q35_graph_launch();
    // Read back the argmax token.  The blocking D2H (cudaMemcpy) implicitly
    // waits for the stream (and hence the whole graph) to finish, so no
    // separate device sync is needed.
    var idx: i32 = 0;
    _ = q35_d2h(@ptrCast(&idx), g.argmax_buf, 4);
    return @intCast(idx);
}

fn resetGdn(gpu: *Gpu) void {
    const conv_bytes = gpu.conv_dim * (gpu.K - 1) * @sizeOf(f32);
    const state_bytes = gpu.nv * gpu.kd * gpu.vd * @sizeOf(f32);
    for (gpu.kv_len) |*l| l.* = 0;
    for (gpu.gdn_conv) |hist| {
        if (hist != null) _ = q35_zero(hist, conv_bytes);
    }
    for (gpu.gdn_state) |st| {
        if (st != null) _ = q35_zero(st, state_bytes);
    }
}

// ---------------------------------------------------------------------------
// GPU init: upload all weights + allocate scratch (one-time)
// ---------------------------------------------------------------------------

fn gpuInit(arena: std.mem.Allocator, e: *engine.Engine) !*Gpu {
    const c = e.cfg;
    const g = try arena.create(Gpu);
    g.gpu_ptrs = .empty;

    const hidden: usize = @intCast(c.hidden);
    const n_ff: usize = @intCast(c.n_ff);
    const heads: usize = @intCast(c.n_heads);
    const kv_heads: usize = @intCast(c.n_kv_heads);
    const hd: usize = @intCast(c.head_dim);
    const q_dim: usize = heads * hd;
    const kv_dim: usize = kv_heads * hd;
    const n_layers: usize = @intCast(c.n_layers);
    const vocab: usize = @intCast(c.vocab);
    const max_seq: usize = @intCast(engine.Engine.max_seq);

    const nv: usize = @intCast(c.gdn_n_heads);
    const nk: usize = @intCast(c.gdn_num_k_heads);
    const kd: usize = @intCast(c.gdn_head_dim);
    const vd: usize = @intCast(c.gdn_head_dim);
    const conv_dim: usize = 2 * nk * kd + nv * kd;
    const value_dim: usize = nv * vd;
    const K: usize = @intCast(c.conv_kernel);

    g.cfg = c;
    g.n_layers = n_layers;
    g.hidden = hidden;
    g.n_ff = n_ff;
    g.heads = heads;
    g.kv_heads = kv_heads;
    g.hd = hd;
    g.q_dim = q_dim;
    g.kv_dim = kv_dim;
    g.nv = nv;
    g.nk = nk;
    g.kd = kd;
    g.vd = vd;
    g.conv_dim = conv_dim;
    g.value_dim = value_dim;
    g.K = K;
    g.max_seq = max_seq;
    g.vocab = vocab;

    g.layers = try arena.alloc(GpuLayer, n_layers);
    g.gdn_conv = try arena.alloc(?*anyopaque, n_layers);
    g.gdn_state = try arena.alloc(?*anyopaque, n_layers);
    g.kv_k = try arena.alloc(?*anyopaque, n_layers);
    g.kv_v = try arena.alloc(?*anyopaque, n_layers);
    g.kv_len = try arena.alloc(u32, n_layers);
    @memset(g.kv_len, 0);

    // scratch
    g.x = try g.allocF32(arena, hidden);
    g.x1 = try g.allocF32(arena, hidden);
    g.x2 = try g.allocF32(arena, hidden);
    const max_qkv = mx(conv_dim, 2 * q_dim);
    g.qkv = try g.allocF32(arena, max_qkv);
    const max_z = mx(value_dim, q_dim);
    g.z = try g.allocF32(arena, max_z);
    g.ab = try g.allocF32(arena, 2 * nv);
    const max_out = mx(value_dim, q_dim);
    g.gdn_out = try g.allocF32(arena, max_out);
    g.k = try g.allocF32(arena, kv_dim);
    g.v = try g.allocF32(arena, kv_dim);
    g.attn = try g.allocF32(arena, max_seq);
    g.mlp_h = try g.allocF32(arena, n_ff);
    g.mlp_h1 = try g.allocF32(arena, n_ff);
    g.logits = try g.allocF32(arena, vocab);
    // q8_scratch holds the Q8_0-quantized INPUT x for each Q8_0 matvec.  The
    // input length is the matvec's `cols`; the largest cols across all layers
    // is max(hidden, value_dim, q_dim, n_ff)  (ffn_down's input is n_ff,
    // attn_output's input is q_dim, ssm_out's is value_dim).  Sizing only to
    // mx(hidden,value_dim) under-allocated it (overflow, corrupted adjacent
    // GPU memory, fatal on 2B/4B where n_ff >> hidden).
    // q8_scratch holds the Q8_0-quantized INPUT x for each Q8_0 matvec.  The
    // input length is the matvec's `cols`; the widest Q8_0 matvec input is
    // max(hidden, value_dim, q_dim, n_ff)  (ffn_down's input is n_ff,
    // attn_output's is q_dim, ssm_out's is value_dim, token_embd's is hidden).
    // Sizing only to mx(hidden,value_dim) under-allocated it, so ffn_down's
    // Q8_0 quantize overflowed and corrupted adjacent GPU memory (garbage on
    // 2B/4B, where n_ff is largest).  Empirically confirmed: this is the
    // 2B/4B root cause (revert -> 2B "Tot!"/4B "!" ; fixed -> bit-exact).
    const max_cols = mx(mx(hidden, value_dim), mx(n_ff, q_dim));
    g.q8_scratch = try g.allocU8(arena, max_cols / 32 * 34);
    g.argmax_buf = try g.allocU8(arena, 4);
    g.beta_buf = try g.allocF32(arena, mx(nv, 64));
    g.gamma_buf = try g.allocF32(arena, mx(nv, 64));
    // host bounce buffer for the f32 (dequant) embedding path
    g.emb_row = try arena.alloc(f32, hidden);

    // token embedding + output norm
    switch (e.token_embd) {
        .q8 => |row| {
            const n: usize = row.data.len;
            const d = q35_malloc(n) orelse return error.GpuAlloc;
            g.gpu_ptrs.append(arena, d) catch return error.GpuAlloc;
            _ = q35_h2d(d, hsrcC(row.data.ptr), n);
            g.token_embd = d;
            g.emb_is_q8 = true;
        },
        .f32 => |arr| {
            const n: usize = arr.len;
            const d = q35_malloc(n * 4) orelse return error.GpuAlloc;
            g.gpu_ptrs.append(arena, d) catch return error.GpuAlloc;
            _ = q35_h2d(d, hsrcC32(arr.ptr), n * 4);
            g.token_embd = d;
            g.emb_is_q8 = false;
        },
    }
    g.output_norm = try uploadF32(arena, g, e.output_norm);

    // per-layer weights
    for (e.layers, 0..) |lw, li| {
        var gl: GpuLayer = .{};
        gl.attn_norm = try uploadF32(arena, g, lw.attn_norm);
        gl.post_attention_norm = try uploadF32(arena, g, lw.post_attention_norm);

        if (c.isFullAttention(@intCast(li))) {
            // full-attention layer (rows=output_dim, cols=input_dim)
            gl.attn_q = try uploadLinear(arena, g, lw.attn_q.?, 2 * q_dim, hidden);
            gl.attn_k = try uploadLinear(arena, g, lw.attn_k.?, kv_dim, hidden);
            gl.attn_v = try uploadLinear(arena, g, lw.attn_v.?, kv_dim, hidden);
            gl.attn_q_norm = try uploadF32(arena, g, lw.attn_q_norm.?);
            gl.attn_k_norm = try uploadF32(arena, g, lw.attn_k_norm.?);
            gl.attn_output = try uploadLinear(arena, g, lw.attn_output.?, hidden, q_dim);
            // GDN state slots stay null; KV cache allocated
            g.gdn_conv[li] = null;
            g.gdn_state[li] = null;
            const kp = q35_malloc(max_seq * kv_dim * 4) orelse return error.GpuAlloc;
            const vp = q35_malloc(max_seq * kv_dim * 4) orelse return error.GpuAlloc;
            g.gpu_ptrs.append(arena, kp) catch return error.GpuAlloc;
            g.gpu_ptrs.append(arena, vp) catch return error.GpuAlloc;
            g.kv_k[li] = kp;
            g.kv_v[li] = vp;
            g.kv_len[li] = 0;
        } else {
            // GDN layer
            gl.attn_qkv = try uploadLinear(arena, g, lw.attn_qkv.?, conv_dim, hidden);
            gl.attn_gate = try uploadLinear(arena, g, lw.attn_gate.?, value_dim, hidden);
            gl.ssm_alpha = try uploadLinear(arena, g, lw.ssm_alpha.?, nv, hidden);
            gl.ssm_beta = try uploadLinear(arena, g, lw.ssm_beta.?, nv, hidden);
            gl.ssm_conv1d = try uploadF32(arena, g, lw.ssm_conv1d.?);
            gl.ssm_dt_bias = try uploadF32(arena, g, lw.ssm_dt_bias.?);
            gl.ssm_a = try uploadF32(arena, g, lw.ssm_a.?);
            gl.ssm_norm = try uploadF32(arena, g, lw.ssm_norm.?);
            gl.ssm_out = try uploadLinear(arena, g, lw.ssm_out.?, hidden, value_dim);
            // KV slots stay null; GDN state allocated
            g.kv_k[li] = null;
            g.kv_v[li] = null;
            const chp = q35_malloc(conv_dim * (K - 1) * 4) orelse return error.GpuAlloc;
            const stp = q35_malloc(nv * kd * vd * 4) orelse return error.GpuAlloc;
            g.gpu_ptrs.append(arena, chp) catch return error.GpuAlloc;
            g.gpu_ptrs.append(arena, stp) catch return error.GpuAlloc;
            g.gdn_conv[li] = chp;
            g.gdn_state[li] = stp;
            g.kv_len[li] = 0;
        }

        gl.ffn_gate = try uploadLinear(arena, g, lw.ffn_gate, n_ff, hidden);
        gl.ffn_up = try uploadLinear(arena, g, lw.ffn_up, n_ff, hidden);
        gl.ffn_down = try uploadLinear(arena, g, lw.ffn_down, hidden, n_ff);

        g.layers[li] = gl;
    }

    return g;
}

// ---------------------------------------------------------------------------
// generation (mirror engine.zig generate; same token order / EOS handling)
// ---------------------------------------------------------------------------

const StreamCtx = struct {
    out: *Io.Writer,
    tok: *const tokenizer.Tokenizer,
    arena: std.mem.Allocator,

    fn onToken(ctx: ?*anyopaque, token: u32) void {
        const self: *StreamCtx = @ptrCast(@alignCast(ctx orelse return));
        const piece = self.tok.decode(self.arena, &.{token}) catch null;
        var s: []const u8 = "";
        if (piece != null) s = piece.?;
        if (s.len > 0) {
            self.out.print("{s}", .{s}) catch {};
        } else {
            self.out.print("[{d}]", .{token}) catch {};
        }
        self.out.flush() catch {};
    }
};

const GenRes = struct {
    tokens: []u32,
    prefill_ms: f64,
    decode_ms: f64,
    hit_eos: bool,
};

fn generate(
    arena: std.mem.Allocator,
    g: *Gpu,
    io: Io,
    prompt: []const u32,
    max_seq: usize,
    stream: *StreamCtx,
) !GenRes {
    var ctx = makeCtx(g);

    resetGdn(g);

    // Build the whole-forward CUDA Graph once (stream-capture runForward).
    captureGraph(&ctx);

    const t_start = Io.Clock.Timestamp.now(io, .awake);
    var next: u32 = 0;
    for (prompt, 0..) |t, i| next = step(&ctx, t, @intCast(i));
    const t_after = Io.Clock.Timestamp.now(io, .awake);
    const prefill_ns: i96 = t_start.durationTo(t_after).raw.nanoseconds;

    var out: std.ArrayList(u32) = .empty;
    defer out.deinit(arena);
    var pos: u32 = @intCast(prompt.len);
    var gen_ns: i96 = 0;
    var hit_eos = false;
    const eos: u32 = ctx.cfg.eos_token_id;
    while (pos < max_seq) : (pos += 1) {
        if (next == eos) {
            hit_eos = true;
            break;
        }
        try out.append(arena, next);
        StreamCtx.onToken(stream, next);
        const t0 = Io.Clock.Timestamp.now(io, .awake);
        next = step(&ctx, next, pos);
        gen_ns += t0.durationTo(Io.Clock.Timestamp.now(io, .awake)).raw.nanoseconds;
    }
    return .{
        .tokens = try out.toOwnedSlice(arena),
        .prefill_ms = @as(f64, @floatFromInt(prefill_ns)) / 1e6,
        .decode_ms = @as(f64, @floatFromInt(gen_ns)) / 1e6,
        .hit_eos = hit_eos,
    };
}

// ---------------------------------------------------------------------------
// C entry point (called by engine_cuda.cu's main)
// ---------------------------------------------------------------------------

pub fn main(init: std.process.Init) !void {
    const arena: std.mem.Allocator = init.arena.allocator();
    const io = init.io;
    var stdout_buffer: [32768]u8 = undefined;
    var stdout_file_writer: Io.File.Writer = .init(.stdout(), io, &stdout_buffer);
    const out = &stdout_file_writer.interface;

    var stderr_buffer: [4096]u8 = undefined;
    var stderr_file_writer: Io.File.Writer = .init(.stderr(), io, &stderr_buffer);
    const err_out = &stderr_file_writer.interface;

    const args = try init.minimal.args.toSlice(arena);
    if (args.len < 2) {
        try err_out.print("usage: {s} <file.gguf> [--full]\n      {s} run <file.gguf> <prompt> [-n N]\n\n", .{ args[0], args[0] });
        try err_out.flush();
        return error.MissingArgument;
    }
    if (!std.mem.eql(u8, args[1], "run")) {
        try err_out.print("error: unknown command '{s}'\n", .{args[1]});
        return error.UnknownCommand;
    }
    if (args.len < 4) {
        try err_out.print("usage: {s} run <file.gguf> <prompt>\n", .{args[0]});
        return error.MissingArgument;
    }
    const path = args[2];
    const prompt_text = args[3];
    const max_seq: usize = engine.Engine.max_seq;
    var e = engine.Engine.load(arena, path, io) catch |err| {
        err_out.print("error: failed to load '{s}': {s}\n", .{path, @errorName(err)}) catch {};
        return error.LoadFailed;
    };
    defer e.deinit();

    if (q35_cuda_init() != 0) {
        err_out.print("error: CUDA init failed\n", .{}) catch {};
        return error.CudaInitFailed;
    }
    var g = gpuInit(arena, &e) catch |err| {
        err_out.print("error: GPU init failed: {s}\n", .{@errorName(err)}) catch {};
        err_out.flush() catch {};
        return error.CudaInitFailed;
    };
    defer g.deinit(arena);

    var tok = tokenizer.Tokenizer.initFromGguf(arena, &e.g) catch {
        err_out.print("error: tokenizer init failed\n", .{}) catch {};
        return error.TokenizerInitFailed;
    };
    const rendered = tokenizer.Tokenizer.renderChatTurn(arena, prompt_text, false) catch {
        err_out.print("error: prompt render failed\n", .{}) catch {};
        return error.PromptRenderFailed;
    };
    const ptok = tok.encode(arena, rendered) catch {
        err_out.print("error: prompt encode failed\n", .{}) catch {};
        return error.PromptEncodeFailed;
    };
    out.print("[prompt: {d} tokens]\n", .{ptok.len}) catch {};
    out.print("Output: ", .{}) catch {};
    out.flush() catch {};

    var stream = StreamCtx{ .out = out, .tok = &tok, .arena = arena };
    const res = generate(arena, g, io, ptok, max_seq, &stream) catch |err| {
        err_out.print("error: generation failed: {s}\n", .{@errorName(err)}) catch {};
        err_out.flush() catch {};
        return error.GenerationFailed;
    };

    out.print("\n\n-------------------\n", .{}) catch {};
    out.print("Prefill: {d} tokens in {d:.3} s  ({d:.2} tok/s)\n", .{
        ptok.len,
        res.prefill_ms / 1000.0,
        if (res.prefill_ms > 0) @as(f64, @floatFromInt(ptok.len)) / (res.prefill_ms / 1000.0) else 0.0,
    }) catch {};
    out.print("Generate: {d} tokens in {d:.3} s  ({d:.2} tok/s)\n", .{
        res.tokens.len,
        res.decode_ms / 1000.0,
        if (res.decode_ms > 0) @as(f64, @floatFromInt(res.tokens.len)) / (res.decode_ms / 1000.0) else 0.0,
    }) catch {};
    out.print("{s}\n", .{if (res.hit_eos) "Stop reason: EOS" else "Stop reason: max tokens reached"}) catch {};
    out.flush() catch {};
}
