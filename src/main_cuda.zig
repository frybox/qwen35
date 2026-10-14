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
extern "C" fn q35_matmul_q8(w: ?*anyopaque, xf: ?*anyopaque, xq: ?*anyopaque, out: ?*anyopaque, b: c_int, rows: c_int, cols: c_int) void;
extern "C" fn q35_gdn_state_batch(qkv: ?*anyopaque, state: ?*anyopaque, beta: ?*anyopaque, gamma: ?*anyopaque, out: ?*anyopaque, nv: c_int, nk: c_int, kd: c_int, vd: c_int, B: c_int) void;
extern "C" fn q35_attn_batch(q: ?*anyopaque, kc: ?*anyopaque, vc: ?*anyopaque, out: ?*anyopaque, heads: c_int, kv_heads: c_int, hd: c_int, kv_dim: c_int, q_stride: c_int, B: c_int, pos0: c_int, scale: f32) void;
extern "C" fn q35_rope_batch(x: ?*anyopaque, rows: c_int, hd: c_int, rope_dim: c_int, stride: c_int, base: f32, pos0: c_int) void;
extern "C" fn q35_kvstore_batch(qkv: ?*anyopaque, kc: ?*anyopaque, vc: ?*anyopaque, kv_dim: c_int, B: c_int, pos0: c_int) void;
extern "C" fn q35_rope2(x: ?*anyopaque, B: c_int, heads: c_int, rope_dim: c_int, stride: c_int, base: f32, pos0: c_int) void;
extern "C" fn q35_kvstore2(src: ?*anyopaque, cache: ?*anyopaque, kv_dim: c_int, B: c_int, pos0: c_int) void;
extern "C" fn q35_embed_rows(w: ?*anyopaque, tok: ?*anyopaque, x: ?*anyopaque, B: c_int, n: c_int) void;
extern "C" fn q35_conv1d_batch(wconv: ?*anyopaque, x: ?*anyopaque, hist: ?*anyopaque, B: c_int, conv_dim: c_int, k: c_int) void;
extern "C" fn q35_beta_gamma_batch(a: ?*anyopaque, b: ?*anyopaque, abar: ?*anyopaque, dtb: ?*anyopaque, beta: ?*anyopaque, gamma: ?*anyopaque, nv: c_int, B: c_int) void;
extern "C" fn q35_l2norm_strided2d(x: ?*anyopaque, B: c_int, nheads: c_int, n: c_int, token_stride: c_int, eps: f32, scale: f32) void;
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
// Allocate the shared GQA flash-decoding workspace (see kernel.cu).  Called once
// before graph capture; q35_attn uses it when the model is hd=256/group=4.
extern "C" fn q35_attn_ws_alloc(n: c_long) c_int;
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

    // Batched-prefill working buffers (see prefillChunk / runForwardBatched).
    pb_x1: ?*anyopaque = null, // residual stream [B][hidden] (ORIGINAL, accumulated)
    pb_x1n: ?*anyopaque = null, // rmsnorm'd residual [B][hidden] (scratch)
    pb_qkv: ?*anyopaque = null, // gdn qkv [B][conv_dim] / full-attn q|gate [B][q_dim]
    pb_z: ?*anyopaque = null, // gdn attn_gate z [B][value_dim]
    pb_ab: ?*anyopaque = null, // ssm_alpha a [B][nv]
    pb_bb: ?*anyopaque = null, // ssm_beta b [B][nv]
    pb_beta: ?*anyopaque = null, // [B][nv]
    pb_gamma: ?*anyopaque = null, // [B][nv]
    pb_gdn_out: ?*anyopaque = null, // [B][value_dim]
    pb_k: ?*anyopaque = null, // full-attn k [B][kv_dim]
    pb_v: ?*anyopaque = null, // full-attn v [B][kv_dim]
    pb_attn_out: ?*anyopaque = null, // [B][q_dim]
    pb_mlp_h: ?*anyopaque = null, // [B][n_ff]
    pb_mlp_h1: ?*anyopaque = null, // [B][n_ff]
    pb_x2: ?*anyopaque = null, // [B][hidden] temp (linear output before add)
    pb_tok: ?*anyopaque = null, // [B] u32 token ids (device)
    pb_q8: ?*anyopaque = null, // Q8_0 scratch [B][max_cols/32*34]
    pb_logit_row: ?*anyopaque = null, // [hidden] f32 (last-token rmsnormed x)

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

// Decode fast path: the linear layers of one layer share the SAME input row,
// so quantize it once and let the matvecs below reuse g.q8_scratch read-only.
// Bit-exact vs calling linMatvec per matvec: the Q8_0 quantizer is a
// deterministic per-32-element function (same x => same q8_scratch) and
// k_matvec_q8 only READS xq (writes `out`).
fn quantInput(g: *Gpu, x: ?*anyopaque, cols: usize) void {
    q35_quantize_q8(cf32(x), g.q8_scratch, toC(cols / 32));
}

// out[rows] = W @ (already-quantized g.q8_scratch); see quantInput.
fn matvecQ(g: *Gpu, w: *const LinearGpu, out: ?*anyopaque) void {
    const rows = w.rows;
    const rb = w.cols / 32;
    q35_matvec_q8(cu8(w.ptr), cu8(g.q8_scratch), out, toC(rows), toC(rb));
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

    // Sampling (set in generate; defaults are harmless for capture).
    // sampler is shared by value-copied Ctx via pointer so decode and the
    // first-token draw consume one PRNG stream, exactly like the CPU engine.
    sampler: *engine.Sampler = undefined,
    sample_scratch: []f32 = &.{},
    host_logits: []f32 = &.{},
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
    const value_dim = nv * vd; // ssm_out's input width (g.gdn_out)

    const g = ctx.gpu;

    // rmsNorm(x1, x, attn_norm, eps)
    q35_rmsnorm_rows(cf32(g.x), cf32(lw.attn_norm), g.x1, 1, toC(hidden), c.rms_eps);

    // x1 is shared by all 4 Q8_0 projections below: quantize ONCE, then reuse
    // g.q8_scratch (bit-exact vs re-quantizing per matvec, see quantInput).
    quantInput(g, g.x1, hidden);
    // qkv = attn_qkv @ x1   (conv_dim)
    matvecQ(g, &lw.attn_qkv, g.qkv);
    // z = attn_gate @ x1     (value_dim)
    matvecQ(g, &lw.attn_gate, g.z);
    // ab[0..nv] = ssm_alpha @ x1 ; ab[nv..] = ssm_beta @ x1
    matvecQ(g, &lw.ssm_alpha, g.ab);
    const ab_b = offBytes(g.ab, nv * @sizeOf(f32));
    matvecQ(g, &lw.ssm_beta, ab_b);

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
    quantInput(g, g.gdn_out, value_dim);
    matvecQ(g, &lw.ssm_out, g.x2);
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
    const q_dim = heads * hd; // attn_output's input width (g.gdn_out)

    const g = ctx.gpu;

    // rmsNorm(x1, x, attn_norm, eps)
    q35_rmsnorm_rows(cf32(g.x), cf32(lw.attn_norm), g.x1, 1, toC(hidden), c.rms_eps);

    // x1 is shared by all 3 Q8_0 projections below: quantize ONCE, then reuse
    // g.q8_scratch (bit-exact vs re-quantizing per matvec, see quantInput).
    quantInput(g, g.x1, hidden);
    // qg = attn_q @ x1  (per-head [q|gate] interleaved in qkv)
    matvecQ(g, &lw.attn_q, g.qkv);
    // k = attn_k @ x1 ; v = attn_v @ x1
    matvecQ(g, &lw.attn_k, g.k);
    matvecQ(g, &lw.attn_v, g.v);

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
    quantInput(g, g.gdn_out, q_dim);
    matvecQ(g, &lw.attn_output, g.x2);
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
    // x1 is shared by both Q8_0 projections below: quantize ONCE, then reuse
    // g.q8_scratch (bit-exact vs re-quantizing per matvec, see quantInput).
    quantInput(g, g.x1, hidden);
    // mlp_h = ffn_gate @ x1 ; mlp_h1 = ffn_up @ x1
    matvecQ(g, &lw.ffn_gate, g.mlp_h);
    matvecQ(g, &lw.ffn_up, g.mlp_h1);
    // mlp_h = silu(mlp_h) * mlp_h1
    q35_silu_mul(g.mlp_h, cf32(g.mlp_h1), toC(n_ff));
    // x += ffn_down @ mlp_h
    quantInput(g, g.mlp_h, n_ff);
    matvecQ(g, &lw.ffn_down, g.x2);
    // x += x2 (on device)
    q35_add(g.x, g.x2, toC(hidden));
}

// ---------------------------------------------------------------------------
// batched prefill (chunks of PREFILL_CHUNK tokens)
// ---------------------------------------------------------------------------

const PREFILL_CHUNK: usize = 128;

/// Batched linear: out = w @ x over B token-rows.  x is [B][cols] f32
/// token-major, out is [B][rows] f32 token-major.  Q8_0 weights use the
/// batched GEMM; f32 weights fall back to per-token matvec (Q8_0 models —
/// the ones we ship — never hit the fallback).
fn linBatched(g: *Gpu, l: *const LinearGpu, x: ?*anyopaque, out: ?*anyopaque, B: usize, cols: usize) void {
    const w = l.ptr orelse return;
    if (l.is_q8) {
        q35_matmul_q8(cu8(w), cf32(x), g.pb_q8, cf32(out), @intCast(B), @intCast(l.rows), @intCast(cols));
    } else {
        for (0..B) |b| {
            q35_matvec_f32(cf32(w), offBytes(x, b * cols * 4), offBytes(out, b * l.rows * 4), @intCast(l.rows), @intCast(cols));
        }
    }
}

fn forwardGdnBatched(ctx: *Ctx, lw: *GpuLayer, li: usize, B: c_int, pos0: u32) void {
    const c = ctx.cfg;
    const g = ctx.gpu;
    const hidden: c_int = @intCast(ctx.hidden);
    const conv_dim: c_int = @intCast(ctx.conv_dim);
    const nv: c_int = @intCast(ctx.nv);
    const nk: c_int = @intCast(ctx.nk);
    const kd: c_int = @intCast(ctx.kd);
    const vd: c_int = @intCast(ctx.vd);
    const K: c_int = @intCast(ctx.K);

    _ = pos0; // conv history carries the chunk; no pos needed here
    // x1n = rmsNorm(x1, attn_norm); x1 (original residual) preserved for the add
    q35_rmsnorm_rows(g.pb_x1, cf32(lw.attn_norm), g.pb_x1n, B, hidden, c.rms_eps);
    // qkv = attn_qkv @ x1n ; z = attn_gate @ x1n ; a/b = ssm_alpha/ssm_beta @ x1n
    linBatched(g, &lw.attn_qkv, g.pb_x1n, g.pb_qkv, @intCast(B), ctx.hidden);
    linBatched(g, &lw.attn_gate, g.pb_x1n, g.pb_z, @intCast(B), ctx.hidden);
    linBatched(g, &lw.ssm_alpha, g.pb_x1n, g.pb_ab, @intCast(B), ctx.hidden);
    linBatched(g, &lw.ssm_beta, g.pb_x1n, g.pb_bb, @intCast(B), ctx.hidden);
    // causal conv1d in place (per-layer history carries the chunk)
    q35_conv1d_batch(cf32(lw.ssm_conv1d), g.pb_qkv, g.gdn_conv[li], B, conv_dim, K);
    // beta = sigmoid(b); gamma = exp(a_bar * softplus(a + dt_bias))
    q35_beta_gamma_batch(g.pb_ab, g.pb_bb, cf32(lw.ssm_a), cf32(lw.ssm_dt_bias), g.pb_beta, g.pb_gamma, nv, B);
    // l2norm + qscale on the Q section; l2norm on the K section
    const qscale = 1.0 / std.math.sqrt(@as(f32, @floatFromInt(ctx.kd)));
    q35_l2norm_strided2d(g.pb_qkv, B, nk, kd, conv_dim, 1e-6, qscale);
    q35_l2norm_strided2d(offBytes(g.pb_qkv, @intCast(nk * kd * 4)), B, nk, kd, conv_dim, 1e-6, 1.0);
    // GDN state update (in-place state, loops the chunk)
    q35_gdn_state_batch(g.pb_qkv, g.gdn_state[li], g.pb_beta, g.pb_gamma, g.pb_gdn_out, nv, nk, kd, vd, B);
    // gated rmsNorm: gdn_out = rmsnorm(gdn_out, ssm_norm) * silu(z)
    q35_gated_rmsnorm(g.pb_gdn_out, cf32(lw.ssm_norm), g.pb_z, B * nv, vd, c.rms_eps);
    // x += ssm_out @ gdn_out
    linBatched(g, &lw.ssm_out, g.pb_gdn_out, g.pb_x2, @intCast(B), g.value_dim);
    q35_add(g.pb_x1, g.pb_x2, B * hidden);
}

fn forwardFullBatched(ctx: *Ctx, lw: *GpuLayer, li: usize, B: c_int, pos0: u32) void {
    const c = ctx.cfg;
    const g = ctx.gpu;
    const hidden: c_int = @intCast(ctx.hidden);
    const heads: c_int = @intCast(ctx.heads);
    const kv_heads: c_int = @intCast(ctx.kv_heads);
    const hd: c_int = @intCast(ctx.hd);
    const kv_dim: c_int = @intCast(g.kv_dim);
    const q_stride: c_int = 2 * hd;
    const rope_dim: c_int = @intCast(c.rotary_dim);
    const base: f32 = c.rope_freq_base;
    const scale = 1.0 / std.math.sqrt(@as(f32, @floatFromInt(ctx.hd)));
    const pos0i: c_int = @intCast(pos0);

    q35_rmsnorm_rows(g.pb_x1, cf32(lw.attn_norm), g.pb_x1n, B, hidden, c.rms_eps);
    // q|gate interleaved [B][heads][2*hd]; k / v [B][kv_dim]
    linBatched(g, &lw.attn_q, g.pb_x1n, g.pb_qkv, @intCast(B), ctx.hidden);
    linBatched(g, &lw.attn_k, g.pb_x1n, g.pb_k, @intCast(B), ctx.hidden);
    linBatched(g, &lw.attn_v, g.pb_x1n, g.pb_v, @intCast(B), ctx.hidden);
    // q/k rmsNorm (in place, strided over token-major rows)
    q35_rmsnorm_strided(g.pb_qkv, cf32(lw.attn_q_norm), g.pb_qkv, B * heads, hd, q_stride, c.rms_eps);
    q35_rmsnorm_strided(g.pb_k, cf32(lw.attn_k_norm), g.pb_k, B * kv_heads, hd, hd, c.rms_eps);
    // RoPE per-token (token b at position pos0 + b)
    q35_rope2(g.pb_qkv, B, heads, rope_dim, q_stride, base, pos0i);
    q35_rope2(g.pb_k, B, kv_heads, rope_dim, hd, base, pos0i);
    // KV cache: k -> kv_k[li], v -> kv_v[li], slot pos0 + b
    q35_kvstore2(g.pb_k, g.kv_k[li], kv_dim, B, pos0i);
    q35_kvstore2(g.pb_v, g.kv_v[li], kv_dim, B, pos0i);
    // batched causal GQA attention (sigmoid gate applied in-kernel)
    q35_attn_batch(g.pb_qkv, g.kv_k[li], g.kv_v[li], g.pb_attn_out, heads, kv_heads, hd, kv_dim, q_stride, B, pos0i, scale);
    // x += attn_output @ attn_out
    linBatched(g, &lw.attn_output, g.pb_attn_out, g.pb_x2, @intCast(B), g.q_dim);
    q35_add(g.pb_x1, g.pb_x2, B * hidden);
}

fn forwardMlpBatched(ctx: *Ctx, lw: *GpuLayer, B: c_int) void {
    const c = ctx.cfg;
    const g = ctx.gpu;
    const hidden: c_int = @intCast(ctx.hidden);
    const n_ff: c_int = @intCast(ctx.n_ff);

    q35_rmsnorm_rows(g.pb_x1, cf32(lw.post_attention_norm), g.pb_x1n, B, hidden, c.rms_eps);
    linBatched(g, &lw.ffn_gate, g.pb_x1n, g.pb_mlp_h, @intCast(B), ctx.hidden);
    linBatched(g, &lw.ffn_up, g.pb_x1n, g.pb_mlp_h1, @intCast(B), ctx.hidden);
    q35_silu_mul(g.pb_mlp_h, cf32(g.pb_mlp_h1), B * n_ff);
    linBatched(g, &lw.ffn_down, g.pb_mlp_h, g.pb_x2, @intCast(B), ctx.n_ff);
    q35_add(g.pb_x1, g.pb_x2, B * hidden);
}

/// Batched forward over B prompt tokens starting at position pos0.  Fills the
/// KV-cache slots pos0..pos0+B-1 and the GDN conv/state history in place,
/// returns the argmax token of the LAST token (logits computed for that token
/// only, B=1, reusing the single-token quantize+matvec+argmax path).
fn forwardBatched(ctx: *Ctx, tokens: []const u32, pos0: u32) u32 {
    const c = ctx.cfg;
    const g = ctx.gpu;
    const hidden = ctx.hidden;
    const B = tokens.len;

    _ = q35_h2d(g.pb_tok, hsrcC(@ptrCast(tokens.ptr)), B * 4);
    q35_embed_rows(cu8(g.token_embd), cu8(g.pb_tok), g.pb_x1, @intCast(B), @intCast(hidden));

    for (g.layers, 0..) |*lw, li| {
        if (c.isFullAttention(@intCast(li))) {
            forwardFullBatched(ctx, lw, li, @intCast(B), pos0);
        } else {
            forwardGdnBatched(ctx, lw, li, @intCast(B), pos0);
        }
        forwardMlpBatched(ctx, lw, @intCast(B));
    }

    // final rmsNorm + tied lm_head + argmax for the last token only
    const last_row = offBytes(g.pb_x1, (B - 1) * hidden * 4);
    q35_rmsnorm_rows(last_row, cf32(g.output_norm), g.pb_logit_row, 1, @intCast(hidden), c.rms_eps);
    const vocab: usize = @intCast(c.vocab);
    if (g.emb_is_q8) {
        const row_blocks = hidden / 32;
        q35_quantize_q8(g.pb_logit_row, g.q8_scratch, @intCast(row_blocks));
        q35_matvec_q8(cu8(g.token_embd), cu8(g.q8_scratch), g.logits, @intCast(vocab), @intCast(row_blocks));
    } else {
        q35_matvec_f32(cf32(g.token_embd), g.pb_logit_row, g.logits, @intCast(vocab), @intCast(hidden));
    }
    q35_argmax(g.logits, g.argmax_buf, @intCast(vocab));
    var idx: i32 = 0;
    _ = q35_d2h(@ptrCast(&idx), g.argmax_buf, 4);
    return @intCast(idx);
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
    if (ctx.sampler.temperature > 0) {
        // Sampling: pull the whole logit row back and sample on the host
        // (same Sampler as the CPU engine, same PRNG stream).  The blocking
        // D2H implicitly waits for the whole graph, so no extra sync needed.
        _ = q35_d2h(hostPtr(ctx.host_logits), g.logits, g.vocab * @sizeOf(f32));
        return ctx.sampler.sample(ctx.host_logits, ctx.sample_scratch);
    }
    // Greedy: read back just the 4-byte argmax the graph already computed.
    var idx: i32 = 0;
    _ = q35_d2h(@ptrCast(&idx), g.argmax_buf, 4);
    return @intCast(idx);
}

/// Pointer to a host f32 slice, as the C d2h routine expects.
fn hostPtr(s: []f32) *anyopaque {
    return @ptrCast(s.ptr);
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

    // Batched-prefill working buffers.  Each prefill chunk is CHUNK tokens
    // (see prefillChunk); every [B] tensor below is sized for B = CHUNK.
    const CH: usize = 128;
    g.pb_x1 = try g.allocF32(arena, CH * hidden);
    g.pb_x1n = try g.allocF32(arena, CH * hidden);
    // gdn qkv is [conv_dim]; full-attn q|gate is [heads*2*hd = 2*q_dim].
    g.pb_qkv = try g.allocF32(arena, CH * mx(conv_dim, 2 * q_dim));
    g.pb_z = try g.allocF32(arena, CH * value_dim); // gdn attn_gate z
    g.pb_ab = try g.allocF32(arena, CH * nv);
    g.pb_bb = try g.allocF32(arena, CH * nv);
    g.pb_beta = try g.allocF32(arena, CH * nv);
    g.pb_gamma = try g.allocF32(arena, CH * nv);
    g.pb_gdn_out = try g.allocF32(arena, CH * value_dim);
    g.pb_k = try g.allocF32(arena, CH * kv_dim);
    g.pb_v = try g.allocF32(arena, CH * kv_dim);
    g.pb_attn_out = try g.allocF32(arena, CH * q_dim);
    g.pb_mlp_h = try g.allocF32(arena, CH * n_ff);
    g.pb_mlp_h1 = try g.allocF32(arena, CH * n_ff);
    g.pb_x2 = try g.allocF32(arena, CH * hidden);
    g.pb_tok = try g.allocU8(arena, CH * 4); // u32 token ids
    // Q8_0 scratch for batched GEMM: B * (max_cols/32) * 34 bytes.
    g.pb_q8 = try g.allocU8(arena, CH * (max_cols / 32) * 34);
    g.pb_logit_row = try g.allocF32(arena, hidden);

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

    // GQA flash-decoding attention workspace: [heads][max_seq/8][hd+2] floats,
    // shared by every full-attention layer (kernels run sequentially on one
    // stream).  Sized for the real model (hd=256, group=4); q35_attn falls
    // back to legacy k_attn if this is absent or the dims don't match.
    const attn_ws_n: usize = heads * 512 * (hd + 2) * 4;
    _ = q35_attn_ws_alloc(@intCast(attn_ws_n));

    return g;
}

// ---------------------------------------------------------------------------
// generation (mirror engine.zig generate; same token order / EOS handling)
// ---------------------------------------------------------------------------

// TEMP(debug): dump token ids when Q35_DUMP_IDS is set non-empty.
var dbg_dump_ids = false;
fn dbgIds(token: u32) void {
    if (dbg_dump_ids) std.debug.print("ID {d}\n", .{token});
}

const StreamCtx = struct {
    out: *Io.Writer,
    tok: *const tokenizer.Tokenizer,
    arena: std.mem.Allocator,

    fn onToken(ctx: ?*anyopaque, token: u32) void {
        dbgIds(token);
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
    temp: f32,
    top_k: usize,
    seed: u64,
) !GenRes {
    var ctx = makeCtx(g);

    // Host-side sampler shared by decode and the first generated token.
    var sampler = engine.Sampler.init(.{ .temperature = temp, .top_k = top_k, .seed = seed });
    ctx.sampler = &sampler;
    sampler.reseed();
    ctx.sample_scratch = try arena.alloc(f32, g.vocab);
    ctx.host_logits = try arena.alloc(f32, g.vocab);

    resetGdn(g);

    // Build the whole-forward CUDA Graph once (stream-capture runForward).
    captureGraph(&ctx);

    const t_start = Io.Clock.Timestamp.now(io, .awake);
    var next: u32 = 0;
    // Batched prefill: process the prompt in chunks of PREFILL_CHUNK tokens so
    // each weight is read once per chunk (B>1 GEMMs) instead of once per token.
    var ppos: usize = 0;
    while (ppos < prompt.len) {
        const end = @min(ppos + PREFILL_CHUNK, prompt.len);
        next = forwardBatched(&ctx, prompt[ppos..end], @intCast(ppos));
        ppos = end;
    }
    const t_after = Io.Clock.Timestamp.now(io, .awake);
    const prefill_ns: i96 = t_start.durationTo(t_after).raw.nanoseconds;

    // First generated token: when sampling, draw it from the last prefill
    // position's logits (already on device in g.logits).  Greedy path leaves
    // `next` as the argmax forwardBatched just returned.  This is exactly
    // where the CPU engine's PRNG stream begins too.
    if (sampler.temperature > 0) {
        _ = q35_d2h(hostPtr(ctx.host_logits), g.logits, g.vocab * @sizeOf(f32));
        next = sampler.sample(ctx.host_logits, ctx.sample_scratch);
    }

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
    if (init.environ_map.get("Q35_DUMP_IDS")) |ids| {
        if (ids.len != 0) dbg_dump_ids = true;
    }
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
    var max_seq: usize = engine.Engine.max_seq;
    var temp: f32 = 0;
    var top_k: usize = 0;
    var seed: u64 = 12345;
    var i: usize = 4;
    while (i + 1 < args.len + 1 and i < args.len) : (i += 1) {
        if (std.mem.eql(u8, args[i], "-n") and i + 1 < args.len) {
            max_seq = std.fmt.parseInt(usize, args[i + 1], 10) catch {
                err_out.print("error: -n expects a number\n", .{}) catch {};
                err_out.flush() catch {};
                return error.InvalidArgument;
            };
            i += 1;
        } else if (std.mem.eql(u8, args[i], "-t") and i + 1 < args.len) {
            temp = std.fmt.parseFloat(f32, args[i + 1]) catch {
                err_out.print("error: -t expects a number\n", .{}) catch {};
                err_out.flush() catch {};
                return error.InvalidArgument;
            };
            i += 1;
        } else if (std.mem.eql(u8, args[i], "-k") and i + 1 < args.len) {
            top_k = std.fmt.parseInt(usize, args[i + 1], 10) catch {
                err_out.print("error: -k expects a number\n", .{}) catch {};
                err_out.flush() catch {};
                return error.InvalidArgument;
            };
            i += 1;
        } else if (std.mem.eql(u8, args[i], "-s") and i + 1 < args.len) {
            seed = std.fmt.parseInt(u64, args[i + 1], 10) catch {
                err_out.print("error: -s expects a number\n", .{}) catch {};
                err_out.flush() catch {};
                return error.InvalidArgument;
            };
            i += 1;
        } else {
            err_out.print("error: unknown argument '{s}'\n", .{args[i]}) catch {};
            err_out.flush() catch {};
            return error.InvalidArgument;
        }
    }
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
    const res = generate(arena, g, io, ptok, max_seq, &stream, temp, top_k, seed) catch |err| {
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
