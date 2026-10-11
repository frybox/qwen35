//! Qwen3.5-0.8B CPU scalar inference engine (Q8_0 GGUF only).
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
const Io = std.Io;

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
pub fn rmsNorm(out: []f32, x: []const f32, w: []const f32, eps: f32) void {
    std.debug.assert(out.len == x.len and out.len == w.len);
    var ss: f32 = 0;
    for (x) |v| ss += v * v;
    const scale = 1.0 / @sqrt(ss / @as(f32, @floatFromInt(x.len)) + eps);
    for (out, x, w) |*o, v, wi| o.* = v * scale * wi;
}

pub fn sigmoid(x: f32) f32 {
    return 1.0 / (1.0 + @exp(-x));
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
pub fn l2norm(x: []f32) void {
    var ss: f32 = 0;
    for (x) |v| ss += v * v;
    const inv = 1.0 / @sqrt(ss + 1e-6);
    for (x) |*v| v.* *= inv;
}

/// Row-major [rows, cols] matvec: out[r] = sum_c mat[r*cols + c] * x[c].
pub fn matvec(out: []f32, mat: []const f32, x: []const f32, rows: usize, cols: usize) void {
    std.debug.assert(mat.len == rows * cols and out.len == rows and x.len == cols);
    for (0..rows) |r| {
        var acc: f32 = 0;
        const row = mat[r * cols ..][0..cols];
        for (row, x) |m, v| acc += m * v;
        out[r] = acc;
    }
}

/// Linear layer with Q8_0-native weights: dispatch to the integer dot-product
/// path when the weight is kept quantized, plain f32 otherwise. `q8` is the
/// engine's quantization scratch (must fit ceil(cols/32) blocks).
fn linMatvec(out: []f32, w: Engine.LinearW, x: []const f32, q8: []u8) void {
    switch (w) {
        .q8 => |row| matvecQ8_0(out, row, x, q8),
        .f32 => |m| matvec(out, m, x, out.len, x.len),
    }
}

// ---------------------------------------------------------------------------
// Q8_0 block format (ggml: 34 bytes per 32 elements: f16 scale + 32 x i8)
// ---------------------------------------------------------------------------

pub const q8_0_block_elems = 32;
pub const q8_0_block_bytes = 34;

/// Dequantize Q8_0 into f32 (used for small tensors / tests).
pub fn dequantizeQ8_0(data: []const u8, out: []f32) !void {
    if (out.len % q8_0_block_elems != 0) return error.NotMultipleOfBlock;
    if (data.len != out.len / q8_0_block_elems * q8_0_block_bytes) return error.SizeMismatch;
    var blk: usize = 0;
    var base: usize = 0;
    while (blk < out.len) : ({
        blk += q8_0_block_elems;
        base += q8_0_block_bytes;
    }) {
        const d: f16 = @bitCast(std.mem.readInt(u16, data[base..][0..2], .little));
        const df: f32 = @floatCast(d);
        const qs = data[base + 2 ..][0..q8_0_block_elems];
        for (qs, 0..) |q, j| out[blk + j] = df * @as(f32, @floatFromInt(@as(i8, @bitCast(q))));
    }
}

pub fn quantizeQ8_0(data: []const f32, out: []u8) !void {
    if (data.len % q8_0_block_elems != 0) return error.NotMultipleOfBlock;
    if (out.len != data.len / q8_0_block_elems * q8_0_block_bytes) return error.SizeMismatch;
    var idata: usize = 0;
    var iout: usize = 0;
    while (idata < data.len):
          ({ idata += q8_0_block_elems; iout += q8_0_block_bytes; }) {
        var max: f32 = 0;
        for (data[idata..][0..q8_0_block_elems]) |d| {
            const v = @abs(d);
            if (v > max) max = v;
        }
        const scale32: f32 = max / 127.0;
        const scale: f16 = @floatCast(scale32);
        std.mem.writeInt(u16, out[iout..][0..2], @bitCast(scale), .little);
        for (0..q8_0_block_elems) |i| {
            out[iout+2+i] = @bitCast(@as(i8, @intFromFloat(@round(data[idata+i]/scale32))));
        }
    }
}

// ---------------------------------------------------------------------------
// Q8_0-native matvec: weights stay in quantized blocks (zero-copy mmap
// view), dot products accumulate in i32 per block and rescale by the f16
// block scale. x is f32, quantized on the fly into a scratch buffer.
// ---------------------------------------------------------------------------

/// Per-row Q8_0 weight view into the mmapped GGUF data section.
pub const Q8Row = struct {
    /// raw block bytes, `rows * row_blocks * q8_0_block_bytes`
    data: []const u8,
    /// quantized blocks per row (cols / 32)
    row_blocks: usize,

    pub fn rowBytes(self: Q8Row) usize {
        return self.row_blocks * q8_0_block_bytes;
    }

    pub fn rows(self: Q8Row) usize {
        return self.data.len / self.rowBytes();
    }

    pub fn row(self: Q8Row, r: usize) []const u8 {
        return self.data[r * self.rowBytes() ..][0..self.rowBytes()];
    }
};

/// out[r] = sum_c W[r,c] * x[c], W row-major Q8_0 blocks. Scratch q8 must
/// hold ceil(cols / 32) blocks (x quantized per block with round-to-nearest,
/// matching quantizeQ8_0 semantics).
pub fn matvecQ8_0(out: []f32, w: Q8Row, x: []const f32, q8: []u8) void {
    std.debug.assert(w.rows() == out.len);
    const cols = w.row_blocks * q8_0_block_elems;
    std.debug.assert(x.len == cols);
    std.debug.assert(q8.len >= w.row_blocks * q8_0_block_bytes);

    // quantize x into scratch (same scale rule as quantizeQ8_0)
    var qblocks: usize = 0;
    var qi: usize = 0;
    while (qi < x.len) : ({
        qi += q8_0_block_elems;
        qblocks += 1;
    }) {
        const xblk = x[qi..][0..q8_0_block_elems];
        const qbase = q8[qblocks * q8_0_block_bytes ..][0..q8_0_block_bytes];
        var max: f32 = 0;
        for (xblk) |v| {
            const a = @abs(v);
            if (a > max) max = a;
        }
        const scale32: f32 = if (max > 0) max / 127.0 else 1.0;
        const scale: f16 = @floatCast(scale32);
        std.mem.writeInt(u16, qbase[0..2], @bitCast(scale), .little);
        for (0..q8_0_block_elems) |j| {
            const q: i32 = if (max > 0) @intFromFloat(@round(xblk[j] / scale32)) else 0;
            qbase[2 + j] = @bitCast(@as(i8, @intCast(std.math.clamp(q, -128, 127))));
        }
    }

    // dot products: per-block i32 accumulation, rescale, sum in f32
    for (out, 0..) |*o, r| {
        const wrow = w.row(r);
        var acc: f32 = 0;
        var wb: usize = 0;
        var xb: usize = 0;
        while (wb < w.row_blocks) : ({
            wb += 1;
            xb += 1;
        }) {
            const wb_slice = wrow[wb * q8_0_block_bytes ..][0..q8_0_block_bytes];
            const wscale: f16 = @bitCast(std.mem.readInt(u16, wb_slice[0..2], .little));
            const xscale: f16 = @bitCast(std.mem.readInt(u16, q8[xb * q8_0_block_bytes ..][0..2], .little));
            var dot: i32 = 0;
            inline for (0..q8_0_block_elems) |j| {
                const wq: i8 = @bitCast(wb_slice[2 + j]);
                const xq: i8 = @bitCast(q8[xb * q8_0_block_bytes + 2 + j]);
                dot += @as(i32, wq) * @as(i32, xq);
            }
            acc += @as(f32, @floatCast(wscale)) * @as(f32, @floatFromInt(dot)) * @as(f32, @floatCast(xscale));
        }
        o.* = acc;
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
        .q8_0 => try dequantizeQ8_0(data, out),
        else => return error.UnsupportedTensorType,
    }
    return out;
}

/// Load a 2D linear weight: Q8_0 tensors stay quantized (zero-copy view
/// into the mmap), anything else is dequantized to f32.
fn loadLinear(alloc: std.mem.Allocator, g: *gguf.File, name: []const u8) !Engine.LinearW {
    const t = g.findTensor(name) orelse return error.MissingTensor;
    if (t.ttype == .q8_0) return .{ .q8 = try Engine.q8RowOf(g, t) };
    return .{ .f32 = try loadTensorF32(alloc, g, name) };
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

pub const Engine = struct {
    alloc: std.mem.Allocator,
    cfg: Config,
    g: gguf.File,
    mm: std.Io.File.MemoryMap,
    file: std.Io.File,
    io: std.Io,
    /// set by `load` when it created the Threaded instance; freed in deinit
    owns_io: ?*std.Io.Threaded = null,

    token_embd: LinearW, // [vocab, hidden] row-major (tied output)
    output_norm: []f32, // [hidden]
    /// Q8 scratch for quantizing activations; sized for the widest matvec
    /// (GDN attn_qkv cols)
    q8_scratch: []u8,
    layers: []LayerWeights,
    kv: []KvCache,
    gdn: []GdnState,

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
    prefill_ms: f64 = 0,

    /// A linear layer weight: either kept in its native Q8_0 blocks
    /// (zero-copy view into the mmapped file) or dequantized to f32
    /// (small tensors already stored as F32 in the GGUF).
    pub const LinearW = union(enum) {
        q8: Q8Row,
        f32: []f32,
    };

    fn q8RowOf(g: *gguf.File, t: *const gguf.Tensor) !Q8Row {
        // GGUF dims are innermost-first: dims[0] = contiguous cols, dims[1] = rows
        std.debug.assert(t.dims.len == 2);
        std.debug.assert(t.dims[0] % q8_0_block_elems == 0);
        const row_blocks: usize = @intCast(t.dims[0] / q8_0_block_elems);
        std.debug.assert(t.nElems() / t.dims[0] * row_blocks * q8_0_block_bytes ==
            (try g.tensorData(t)).len);
        return .{
            .data = try g.tensorData(t),
            .row_blocks = row_blocks,
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

    pub fn load(alloc: std.mem.Allocator, path: []const u8, io: std.Io) !Engine {
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

        var cfg = Config{};
        if (g.getUint("qwen35.block_count")) |v| cfg.n_layers = @intCast(v);
        if (g.getUint("qwen35.embedding_length")) |v| cfg.hidden = @intCast(v);
        if (g.getUint("qwen35.feed_forward_length")) |v| cfg.n_ff = @intCast(v);
        if (g.getUint("qwen35.attention.head_count")) |v| cfg.n_heads = @intCast(v);
        if (g.getUint("qwen35.attention.head_count_kv")) |v| cfg.n_kv_heads = @intCast(v);
        if (g.getUint("qwen35.attention.key_length")) |v| cfg.head_dim = @intCast(v);
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
            .output_norm = &.{},
            .q8_scratch = &.{},
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
        };
        errdefer e.freeAll();

        e.token_embd = try loadLinear(alloc, &g, "token_embd.weight");
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
                .ffn_gate = try loadLinear(alloc, &g, tname(&b1, "blk.{d}.ffn_gate.weight", idx)),
                .ffn_up = try loadLinear(alloc, &g, tname(&b2, "blk.{d}.ffn_up.weight", idx)),
                .ffn_down = try loadLinear(alloc, &g, tname(&b1, "blk.{d}.ffn_down.weight", idx)),
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
                e.layers[li].attn_q = try loadLinear(alloc, &g, tname(&b1, "blk.{d}.attn_q.weight", idx));
                e.layers[li].attn_k = try loadLinear(alloc, &g, tname(&b2, "blk.{d}.attn_k.weight", idx));
                e.layers[li].attn_v = try loadLinear(alloc, &g, tname(&b1, "blk.{d}.attn_v.weight", idx));
                e.layers[li].attn_q_norm = try loadTensorF32(alloc, &g, tname(&b2, "blk.{d}.attn_q_norm.weight", idx));
                e.layers[li].attn_k_norm = try loadTensorF32(alloc, &g, tname(&b1, "blk.{d}.attn_k_norm.weight", idx));
                e.layers[li].attn_output = try loadLinear(alloc, &g, tname(&b2, "blk.{d}.attn_output.weight", idx));
            } else {
                e.layers[li].attn_qkv = try loadLinear(alloc, &g, tname(&b1, "blk.{d}.attn_qkv.weight", idx));
                e.layers[li].attn_gate = try loadLinear(alloc, &g, tname(&b2, "blk.{d}.attn_gate.weight", idx));
                e.layers[li].ssm_alpha = try loadLinear(alloc, &g, tname(&b1, "blk.{d}.ssm_alpha.weight", idx));
                e.layers[li].ssm_beta = try loadLinear(alloc, &g, tname(&b2, "blk.{d}.ssm_beta.weight", idx));
                e.layers[li].ssm_conv1d = try loadTensorF32(alloc, &g, tname(&b1, "blk.{d}.ssm_conv1d.weight", idx));
                e.layers[li].ssm_dt_bias = try loadTensorF32(alloc, &g, tname(&b2, "blk.{d}.ssm_dt.bias", idx));
                e.layers[li].ssm_a = try loadTensorF32(alloc, &g, tname(&b1, "blk.{d}.ssm_a", idx));
                e.layers[li].ssm_norm = try loadTensorF32(alloc, &g, tname(&b2, "blk.{d}.ssm_norm.weight", idx));
                e.layers[li].ssm_out = try loadLinear(alloc, &g, tname(&b1, "blk.{d}.ssm_out.weight", idx));
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
        // Largest Q8_0 activation row: attn_qkv cols on a GDN layer; on full
        // attention layers attn_q (2*q_dim) is smaller for every known config.
        const max_q8_row_bytes = @max(
            gdn_conv_dim,
            2 * @as(usize, cfg.n_heads) * @as(usize, cfg.head_dim),
        ) / q8_0_block_elems * q8_0_block_bytes;
        e.qkv = try alloc.alloc(f32, @max(gdn_conv_dim, 2 * @as(usize, cfg.n_heads) * cfg.head_dim));
        e.q8_scratch = try alloc.alloc(u8, max_q8_row_bytes);
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
        return e;
    }

    fn tname(buf: []u8, comptime fmt: []const u8, idx: u32) []const u8 {
        return std.fmt.bufPrint(buf, fmt, .{idx}) catch unreachable;
    }

    fn freeLinear(alloc: std.mem.Allocator, w: LinearW) void {
        // q8 views point into the mmap; only f32 copies are allocated
        if (w == .f32) alloc.free(w.f32);
    }

    fn freeLayerWeights(alloc: std.mem.Allocator, lw: LayerWeights) void {
        alloc.free(lw.attn_norm);
        alloc.free(lw.post_attention_norm);
        freeLinear(alloc, lw.ffn_gate);
        freeLinear(alloc, lw.ffn_up);
        freeLinear(alloc, lw.ffn_down);
    }

    fn freeLayerTyped(alloc: std.mem.Allocator, lw: *LayerWeights) void {
        // linear weights (q8 views point into the mmap, f32 copies are ours)
        inline for (.{
            &lw.attn_qkv, &lw.attn_gate, &lw.ssm_alpha, &lw.ssm_beta, &lw.ssm_out,
            &lw.attn_q,   &lw.attn_k,    &lw.attn_v,    &lw.attn_output,
        }) |p| {
            if (p.*) |w| freeLinear(alloc, w);
        }
        // plain f32 scratch tensors
        inline for (.{
            &lw.ssm_conv1d, &lw.ssm_dt_bias, &lw.ssm_a, &lw.ssm_norm,
            &lw.attn_q_norm, &lw.attn_k_norm,
        }) |p| {
            if (p.*) |w| alloc.free(w);
        }
    }

    fn freeCachesUpTo(self: *Engine, n: usize) void {
        for (self.kv[0..n]) |c| {
            if (c.k.len > 0) self.alloc.free(c.k);
            if (c.v.len > 0) self.alloc.free(c.v);
        }
        for (self.gdn[0..n]) |s| {
            if (s.conv.len > 0) self.alloc.free(s.conv);
            if (s.state.len > 0) self.alloc.free(s.state);
        }
    }

    fn freeAll(self: *Engine) void {
        const a = self.alloc;
        freeLinear(a, self.token_embd);
        if (self.output_norm.len > 0) a.free(self.output_norm);
        if (self.q8_scratch.len > 0) a.free(self.q8_scratch);
        if (self.layers.len > 0) {
            for (self.layers) |lw| freeLayerWeights(a, lw);
            for (self.layers) |*lw| freeLayerTyped(a, lw);
            a.free(self.layers);
        }
        self.freeCachesUpTo(self.cfg.n_layers);
        if (self.kv.len > 0) a.free(self.kv);
        if (self.gdn.len > 0) a.free(self.gdn);
        for ([_]*[]f32{ &self.x, &self.x1, &self.x2, &self.qkv, &self.z, &self.ab, &self.gdn_out, &self.q, &self.k, &self.v, &self.attn, &self.mlp_h, &self.mlp_h1, &self.logits, &self.delta, &self.kv_mem }) |p| {
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

    pub fn step(self: *Engine, token: u32, pos: u32) u32 {
        const c = self.cfg;
        const hidden: usize = c.hidden;
        // embedding lookup: dequantize the token row straight into x
        switch (self.token_embd) {
            .q8 => |row| dequantizeQ8_0(row.row(token), self.x[0..hidden]) catch unreachable,
            .f32 => |m| @memcpy(self.x[0..hidden], m[token * hidden ..][0..hidden]),
        }

        for (self.layers, 0..) |*lw, li| {
            if (c.isFullAttention(@intCast(li))) {
                self.forwardFull(lw, li, pos);
            } else {
                self.forwardGdn(lw, li);
            }
            self.forwardMlp(lw);
        }

        rmsNorm(self.x1, self.x, self.output_norm, c.rms_eps);
        // tied lm_head: logits = W_emb @ x1 with W_emb row-major [vocab, hidden]
        switch (self.token_embd) {
            .q8 => |row| matvecQ8_0(self.logits, row, self.x1, self.q8_scratch),
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
        return self.argmax();
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
        linMatvec(qkv, lw.attn_qkv.?, self.x1, self.q8_scratch);

        // z projection (attn_gate [hidden, value_dim])
        const z = self.z[0..value_dim];
        linMatvec(z, lw.attn_gate.?, self.x1, self.q8_scratch);

        // a/b projections: ssm_alpha = a (decay), ssm_beta = b (beta input),
        // one scalar per VALUE head (C ref: gdn_a/gdn_b/A/dt all sized
        // gdn_vheads; GGUF rows are [hidden, nv] on 0.8B/2B/4B alike)
        const ab = self.ab[0 .. 2 * nv];
        linMatvec(ab[0..nv], lw.ssm_alpha.?, self.x1, self.q8_scratch);
        linMatvec(ab[nv..], lw.ssm_beta.?, self.x1, self.q8_scratch);

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
            for (q_h) |*p| p.* *= qscale;

            // (1) decay
            for (sh) |*p| p.* *= gamma[h];
            // (2) kv_mem = S^T k (sum over k axis)
            const km = self.kv_mem[0..vd];
            @memset(km, 0);
            for (0..kd) |i| {
                const ki = k_h[i];
                const row = sh[i * vd ..][0..vd];
                for (row, 0..) |sv, j| km[j] += sv * ki;
            }
            // (3) delta = beta*(v - kv_mem); S += outer(k, delta)
            for (0..vd) |j| self.delta[j] = (v_h[j] - km[j]) * beta[h];
            for (0..kd) |i| {
                const ki = k_h[i];
                const row = sh[i * vd ..][0..vd];
                for (row, 0..) |*sv, j| sv.* += ki * self.delta[j];
            }
            // (4) o = S^T q with the UPDATED state (SPEC A4.6)
            const out_h = gout[h * vd ..][0..vd];
            for (0..vd) |j| {
                var sum: f32 = 0;
                for (0..kd) |i| sum += sh[i * vd + j] * q_h[i];
                out_h[j] = sum;
            }
        }

        // gated RMSNorm: norm -> weight -> silu(gate) (SPEC A6)
        for (0..nv) |h| {
            const out_h = gout[h * vd ..][0..vd];
            const zh = z[h * vd ..][0..vd];
            rmsNorm(out_h, out_h, lw.ssm_norm.?, c.rms_eps);
            for (out_h, zh) |*o, g| o.* *= silu(g);
        }

        // out projection [2048 -> 1024]
        linMatvec(self.x2[0..hidden], lw.ssm_out.?, gout, self.q8_scratch);
        for (0..hidden) |i| self.x[i] += self.x2[i];
    }

    /// Causal conv step: out[i] = sum_t w[i][t] * hist[i][t], where hist =
    /// [3 previous raw samples (oldest..newest), current raw sample].
    fn convStep(self: *Engine, st: *GdnState, qkv: []f32, wconv: []const f32, K: usize) void {
        _ = self;
        for (0..qkv.len) |i| {
            const hist = st.conv[i * (K - 1) ..][0 .. K - 1];
            const cur = qkv[i];
            var sum: f32 = hist[0] * wconv[i * K + 0] +
                hist[1] * wconv[i * K + 1] +
                hist[2] * wconv[i * K + 2] +
                cur * wconv[i * K + 3];
            sum = silu(sum);
            // shift: drop oldest, append current raw
            hist[0] = hist[1];
            hist[1] = hist[2];
            hist[2] = cur;
            qkv[i] = sum;
        }
    }

    // ------------- Full attention block (SPEC B) -------------

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
        linMatvec(qg, lw.attn_q.?, self.x1, self.q8_scratch);
        const q = self.q[0..q_dim];
        const gate = self.z[0..q_dim];
        for (0..heads) |h| {
            // interleave: q(256) = qg[h*512 .. h*512+256], gate = next 256
            @memcpy(q[h * hd ..][0..hd], qg[h * 2 * hd ..][0..hd]);
            @memcpy(gate[h * hd ..][0..hd], qg[h * 2 * hd + hd ..][0..hd]);
        }
        linMatvec(self.k[0..kv_dim], lw.attn_k.?, self.x1, self.q8_scratch);
        linMatvec(self.v[0..kv_dim], lw.attn_v.?, self.x1, self.q8_scratch);

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
        for (0..q_dim) |i| gout[i] *= sigmoid(gate[i]);
        linMatvec(self.x2[0..hidden], lw.attn_output.?, gout, self.q8_scratch);
        for (0..hidden) |i| self.x[i] += self.x2[i];
    }

    // ------------- MLP (SPEC C16) -------------

    fn forwardMlp(self: *Engine, lw: *LayerWeights) void {
        const c = self.cfg;
        const hidden: usize = c.hidden;
        rmsNorm(self.x1, self.x, lw.post_attention_norm, c.rms_eps);
        linMatvec(self.mlp_h, lw.ffn_gate, self.x1, self.q8_scratch);
        linMatvec(self.mlp_h1, lw.ffn_up, self.x1, self.q8_scratch);
        for (0..c.n_ff) |i| self.mlp_h[i] = silu(self.mlp_h[i]) * self.mlp_h1[i];
        linMatvec(self.x2[0..hidden], lw.ffn_down, self.mlp_h, self.q8_scratch);
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

    /// Prefill prompt, then greedy-generate up to max_new_tokens tokens,
    /// stopping at EOS. Caller owns result.tokens.
    pub fn generate(
        self: *Engine,
        alloc: std.mem.Allocator,
        prompt: []const u32,
        max_tokens: usize,
        greedy: bool,
        progress: ?*const fn (ctx: ?*anyopaque, tok: u32) void,
        progress_ctx: ?*anyopaque,
    ) !GenResult {
        _ = greedy; // greedy-only engine; argmax sampling
        self.reset();

        const io = self.io;
        const t_start = Io.Clock.Timestamp.now(io, .awake);
        var next: u32 = 0;
        for (prompt, 0..) |t, i| next = self.step(t, @intCast(i));
        const t_after = Io.Clock.Timestamp.now(io, .awake);
        self.prefill_ms = @as(f64, @floatFromInt(t_start.durationTo(t_after).raw.nanoseconds)) / 1e6;

        var out: std.ArrayList(u32) = .empty;
        errdefer out.deinit(alloc);
        var pos: u32 = @intCast(prompt.len);
        var gen_ns: i96 = 0;
        var hit_eos = false;
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

test "q8_0 dequantize: hand-computed block" {
    var data: [q8_0_block_bytes]u8 = @splat(0);
    std.mem.writeInt(u16, data[0..2], @bitCast(@as(f16, 1.5)), .little);
    const zeros: [27]i8 = @splat(0);
    const qs = [5]i8{ 0, 1, -1, 32, -128 } ++ zeros;
    for (qs, 0..) |q, i| data[2 + i] = @bitCast(q);
    var out: [32]f32 = undefined;
    try dequantizeQ8_0(&data, &out);
    try std.testing.expectEqual(@as(f32, 0), out[0]);
    try std.testing.expectEqual(@as(f32, 1.5), out[1]);
    try std.testing.expectEqual(@as(f32, -1.5), out[2]);
    try std.testing.expectEqual(@as(f32, 48.0), out[3]);
    try std.testing.expectEqual(@as(f32, -192.0), out[4]);
}

test "q8_0 dequantize rejects bad sizes" {
    var data: [2 * q8_0_block_bytes]u8 = @splat(0);
    var out: [33]f32 = undefined;
    try std.testing.expectError(error.NotMultipleOfBlock, dequantizeQ8_0(&data, &out));
    var out2: [64]f32 = undefined;
    try std.testing.expectError(error.SizeMismatch, dequantizeQ8_0(data[0 .. 2 * q8_0_block_bytes - 1], &out2));
}

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
    matvec(&out, &m, &x, 2, 3);
    try std.testing.expectEqual(@as(f32, 50), out[0]);
    try std.testing.expectEqual(@as(f32, 122), out[1]);
}

test "matvecQ8_0 matches f32 matvec on dequantized weights" {
    var prng = std.Random.DefaultPrng.init(42);
    const rand = prng.random();

    const rows = 5;
    const cols = 96; // 3 blocks
    var wq: [rows * cols / q8_0_block_elems * q8_0_block_bytes]u8 = undefined;
    var wf: [rows * cols]f32 = undefined;
    var x: [cols]f32 = undefined;

    // random f32 weights, quantize to Q8_0 blocks, keep f32 dequant copy
    for (0..rows * cols / q8_0_block_elems) |blk| {
        var f32blk: [q8_0_block_elems]f32 = undefined;
        for (&f32blk) |*v| v.* = rand.float(f32) * 2.0 - 1.0;
        try quantizeQ8_0(&f32blk, wq[blk * q8_0_block_bytes ..][0..q8_0_block_bytes]);
        try dequantizeQ8_0(wq[blk * q8_0_block_bytes ..][0..q8_0_block_bytes], wf[blk * q8_0_block_elems ..][0..q8_0_block_elems]);
    }
    for (&x) |*v| v.* = rand.float(f32) * 2.0 - 1.0;

    const w: Q8Row = .{ .data = &wq, .row_blocks = cols / q8_0_block_elems };
    var out_q8: [rows]f32 = undefined;
    var q8scratch: [cols / q8_0_block_elems * q8_0_block_bytes]u8 = undefined;
    matvecQ8_0(&out_q8, w, &x, &q8scratch);

    // reference: f32 matvec over dequantized weights AND dequantized x
    // (the q8 path necessarily quantizes x, so feed the same values in)
    var xq: [cols]f32 = undefined;
    for (0..cols / q8_0_block_elems) |blk| {
        try dequantizeQ8_0(q8scratch[blk * q8_0_block_bytes ..][0..q8_0_block_bytes], xq[blk * q8_0_block_elems ..][0..q8_0_block_elems]);
    }
    var out_f32: [rows]f32 = undefined;
    matvec(&out_f32, &wf, &xq, rows, cols);

    for (out_q8, out_f32) |a, b| {
        // same math (i32 exact per block vs f32 accumulate): tiny rounding only
        try std.testing.expectApproxEqAbs(b, a, @abs(b) * 1e-5 + 1e-4);
    }
}

test "matvecQ8_0 zero x gives zero out" {
    const cols = 32;
    var wq: [2 * q8_0_block_bytes]u8 = undefined;
    const wvals = [3]f32{ 1, 2, 3 } ++ @as([29]f32, @splat(0));
    try quantizeQ8_0(&wvals, wq[0..q8_0_block_bytes]);
    try quantizeQ8_0(&wvals, wq[q8_0_block_bytes..]);
    var x: [cols]f32 = @splat(0);
    var out: [2]f32 = undefined;
    var scratch: [q8_0_block_bytes]u8 = undefined;
    matvecQ8_0(&out, .{ .data = &wq, .row_blocks = 1 }, &x, &scratch);
    try std.testing.expectEqual(@as(f32, 0), out[0]);
    try std.testing.expectEqual(@as(f32, 0), out[1]);
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
        .output_norm = &.{},
        .q8_scratch = &.{},
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
