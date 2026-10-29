//! CPU backend: the only implementation wired up today.
//!
//! Everything the engine must not know about lives here: the ISA-tier
//! dispatcher, the C kernels (src/kernels/x86 via simd_dispatch), the
//! quantized-activation scratch, the per-row dequantizer used by the
//! embedding lookup, and the matvec family.
//!
//! The matvec code below is the engine's original implementation moved here
//! verbatim.  It keeps its structure on purpose: one decision only (the
//! weight's activation block width, 32 vs 256 elements), both shapes calling
//! the C kernel once per row.

const std = @import("std");
const gguf = @import("gguf.zig");
const backend = @import("backend.zig");
const quant = @import("quant_dequant.zig");
const pool_mod = @import("pool.zig");
const simd = @import("simd_dispatch.zig");

const LinearW = backend.LinearW;
const QuantRow = backend.QuantRow;
const Io = std.Io;

pub const blockElemsOf = gguf.blockElems;
pub const blockElemBytes = gguf.blockBytes;

pub const q8_0_block_elems = gguf.blockElems(.q8_0);
pub const q8_0_block_bytes = gguf.blockBytes(.q8_0);

// ---------------------------------------------------------------------------
// Matvec-level profiling
//
// Kept next to the matvec code it instruments.  `prof` is flipped by main()
// from QWEN35_PROF; the engine still owns the coarse per-phase timers.
// ---------------------------------------------------------------------------

pub var prof = false;
pub var prof_io: Io = undefined;

pub const ProfEntry = struct { key: [64]u8 = undefined, klen: usize = 0, ns: f64 = 0, calls: usize = 0 };
pub var prof_entries: [64]ProfEntry = std.mem.zeroes([64]ProfEntry);
pub var prof_n: usize = 0;

fn profStart(key: []const u8) usize {
    const n = prof_n;
    for (0..n) |i| {
        if (prof_entries[i].klen == key.len and std.mem.eql(u8, prof_entries[i].key[0..prof_entries[i].klen], key))
            return i;
    }
    var e = &prof_entries[n];
    e.klen = @min(key.len, 64);
    @memcpy(e.key[0..e.klen], key[0..e.klen]);
    e.ns = 0;
    e.calls = 0;
    prof_n = n + 1;
    return n;
}

fn profEnd(idx: usize, ns: f64) void {
    prof_entries[idx].ns += ns;
    prof_entries[idx].calls += 1;
}

fn profNs(io: Io, t0: Io.Clock.Timestamp) f64 {
    return @as(f64, @floatFromInt(t0.durationTo(Io.Clock.Timestamp.now(io, .awake)).raw.nanoseconds));
}

/// Dump the per-(ttype, rows) matvec timings, heaviest first.
/// The thread pool's own call/timing counters (PROF-P line).
pub fn pool_profReport() void {
    pool_mod.profReport();
}

pub fn profReportMatvec() void {
    var idx: [64]usize = undefined;
    for (0..prof_n) |i| idx[i] = i;
    for (0..prof_n) |i| {
        var best = i;
        for (i + 1..prof_n) |j| {
            if (prof_entries[idx[j]].ns > prof_entries[idx[best]].ns) best = j;
        }
        const t = idx[i];
        idx[i] = idx[best];
        idx[best] = t;
    }
    for (0..prof_n) |i| {
        const e = &prof_entries[idx[i]];
        std.debug.print("PROF-M {s} calls={d} ms={d:.2}\n", .{ e.key[0..e.klen], e.calls, e.ns / 1e6 });
    }
}

// ---------------------------------------------------------------------------
// Scratch sizing
// ---------------------------------------------------------------------------

/// Bytes the Q8_0 activation scratch must hold for an input of `cols`
/// elements (34 B per 32 elements), rounded up to whole blocks.
pub fn q8ScratchBytes(cols: usize) usize {
    return (cols + q8_0_block_elems - 1) / q8_0_block_elems * q8_0_block_bytes;
}

/// Bytes the Q8_K activation scratch must hold for an input of `cols`
/// elements (292 B per 256): f32 scale + 256 x i8 + 16 x i16 bsums.
pub fn q8kScratchBytes(cols: usize) usize {
    const be = gguf.blockElems(.q8_k);
    return (cols + be - 1) / be * gguf.blockBytes(.q8_k);
}

/// Guard the Q8_0 activation scratch against the widest matvec input.
pub fn checkQ8Scratch(scratch: []const u8, cols: usize) !void {
    if (scratch.len < q8ScratchBytes(cols)) return error.Q8ScratchTooSmall;
}

// ---------------------------------------------------------------------------
// Scalar fallback: Zig rowDot for types with no C dot kernel
// ---------------------------------------------------------------------------

fn rowDotFor(ttype: gguf.TensorType) ?*const fn (row: []const u8, x: []const f32) f32 {
    return switch (ttype) {
        .q4_k => &quant.rowDotQ4K,
        .q5_k => &quant.rowDotQ5K,
        .q6_k => &quant.rowDotQ6K,
        .q3_k => &quant.rowDotQ3K,
        .q2_k => &quant.rowDotQ2K,
        .iq4_xs => &quant.rowDotIQ4XS,
        .iq4_nl => &quant.rowDotIQ4NL,
        .iq3_s => &quant.rowDotIQ3S,
        .iq3_xxs => &quant.rowDotIQ3XXS,
        .iq2_s => &quant.rowDotIQ2S,
        .iq2_xxs => &quant.rowDotIQ2XXS,
        .iq2_xs => &quant.rowDotIQ2XS,
        .iq1_s => &quant.rowDotIQ1S,
        .iq1_m => &quant.rowDotIQ1M,
        .q4_0 => &quant.rowDotQ4_0,
        .q4_1 => &quant.rowDotQ4_1,
        .f16 => &quant.rowDotF16,
        .bf16 => &quant.rowDotBF16,
        else => null,
    };
}

fn matvecQuant(out: []f32, w: QuantRow, x: []const f32, q8k: []u8, q8: []u8, pool: ?*pool_mod.Pool) void {
    var pidx: usize = undefined;
    var pt0: Io.Clock.Timestamp = undefined;
    if (prof) {
        var kbuf: [64]u8 = undefined;
        const key = std.fmt.bufPrint(&kbuf, "{s}:{d}", .{ @tagName(w.ttype), w.rows }) catch "q?";
        pidx = profStart(key[0..@min(key.len, 64)]);
        pt0 = Io.Clock.Timestamp.now(prof_io, .awake);
        defer profEnd(pidx, profNs(prof_io, pt0));
    }
    const rows = w.rows;
    const bpr = w.blocks_per_row;
    // `blocks_per_row` counts the type's own row-blocks: 32 elements for
    // Q8_0, 256 for everything else (`QuantRow.rowBlockElems`).  The row
    // width is `bpr * rowBlockElems` (= bpr*256 for the 256-framed types)
    // -- NOT `bpr * blockElems()`, which for q4_0 (32-element blocks, 8 per
    // row-block) would give a 1/8-size row.
    const cols = w.cols();
    std.debug.assert(out.len == rows and x.len >= cols);
    // 32-element blocks <=> the type's own block IS 32 elements.  This is
    // the single, data-layout-derived branch; all kernel selection below
    // follows from it.
    const act_b32 = blockElemsOf(w.ttype) == 32;
    std.debug.assert(if (act_b32) bpr == cols / 32 else bpr == cols / 256);
    std.debug.assert(q8k.len >= q8kScratchBytes(cols));
    std.debug.assert(q8.len >= q8ScratchBytes(cols));

    if (act_b32) {
        // Q8_0 activation (34 B / 32 elems) + one 32-block row kernel per
        // row.  Q8_0 weights land here too: their activation quantizer and
        // dot call are exactly the family's, they are not special.
        const dot = switch (w.ttype) {
            .q8_0 => simd.d32row.q8_0_q8_0,
            .q4_0 => simd.d32row.q4_0_q8_0,
            .q4_1 => simd.d32row.q4_1_q8_0,
            .iq4_nl => simd.d32row.iq4_nl_q8_0,
            .q5_0 => simd.d32row.q5_0_q8_0,
            .q5_1 => simd.d32row.q5_1_q8_0,
            .q1_0 => simd.d32row.q1_0_q8_0,
            .q2_0 => simd.d32row.q2_0_q8_0,
            .mxfp4 => simd.d32row.mxfp4_q8_0,
            .nvfp4 => simd.d32row.nvfp4_q8_0,
            .f16 => simd.d32row.f16_q8_0,
            .bf16 => simd.d32row.bf16_q8_0,
            else => null,
        } orelse {
            // No C row dot for this 32-block type: Zig scalar rowDot
            // oracle (IQ types).  The .b32 shape is a bit-identity no-op
            // loop, so the assert would be wrong; return early below.
            matvecQuantScalar(out, w, x, cols, rows, bpr, pool);
            return;
        };
        simd.actQ8_0(cols, x.ptr, q8.ptr);
        pool_mod.parallel(pool, .{ .quantN = .{
            .act_block = .b32,
            .out = out.ptr,
            .data = w.data,
            .act = q8.ptr,
            .act_len = q8ScratchBytes(cols),
            .blk_bytes = blockElemBytes(w.ttype),
            .blk_elems = blockElemsOf(w.ttype),
            .bpr = bpr,
            .cols = cols,
            .rows = rows,
            .k = 1,
            .dot32 = dot,
            .dot256 = undefined,
        } });
        return;
    }

    // 256-element blocks: Q8_K activation (292 B / 256 elems) + one C dot
    // over the whole row.
    const dot = switch (w.ttype) {
        .q2_k => simd.d256.q2_k_q8_k,
        .q3_k => simd.d256.q3_k_q8_k,
        .q4_k => simd.d256.q4_k_q8_k,
        .q5_k => simd.d256.q5_k_q8_k,
        .q6_k => simd.d256.q6_k_q8_k,
        // Importance-sampled types: same Q8_K activation framing,
        // codebook gather instead of bit-unpacking.
        .iq2_xxs => simd.diq.iq2_xxs_q8_k,
        .iq2_xs => simd.diq.iq2_xs_q8_k,
        .iq2_s => simd.diq.iq2_s_q8_k,
        .iq3_xxs => simd.diq.iq3_xxs_q8_k,
        .iq3_s => simd.diq.iq3_s_q8_k,
        .iq1_s => simd.diq.iq1_s_q8_k,
        .iq1_m => simd.diq.iq1_m_q8_k,
        .iq4_xs => simd.diq.iq4_xs_q8_k,
        else => null,
    } orelse {
        matvecQuantScalar(out, w, x, cols, rows, bpr, pool);
        return;
    };
    simd.actQ8_K(cols, x.ptr, q8k.ptr);
    const rb256 = w.blockBytes(); // one 256-element block
    pool_mod.parallel(pool, .{ .quantN = .{
        .act_block = .b256,
        .out = out.ptr,
        .data = w.data,
        .act = q8k.ptr,
        .act_len = q8kScratchBytes(cols),
        .blk_bytes = rb256,
        .blk_elems = blockElemsOf(w.ttype),
        .bpr = bpr,
        .cols = cols,
        .rows = rows,
        .k = 1,
        .dot32 = undefined,
        .dot256 = dot,
    } });
}

/// Zig scalar reference path (`quant.rowDotXXX` over f32 activations),
/// shared by the serial matvec, the K-token batch and
/// QWEN35_FORCE_ROWDOT.  `act_b32` only decides which activation block
/// width the caller's scratch arrays were sized for; it is unused here
/// because this path never quantizes the activation.
fn matvecQuantScalar(out: []f32, w: QuantRow, x: []const f32, cols: usize, rows: usize, bpr: usize, pool: ?*pool_mod.Pool) void {
    const rowDot = rowDotFor(w.ttype) orelse unreachable; // only supported types load as .quant
    pool_mod.parallel(pool, .{ .quant_scalar = .{
        .out = out.ptr,
        .data = w.data,
        .x = x.ptr,
        .x_len = cols,
        .row_bytes = w.blockBytes(), // per row-block; pool indexes r*bpr*row_bytes
        .bpr = bpr,
        .rows = rows,
        .k = 1,
        .dot = rowDot,
    } });
}

/// Batched K-token matvec: `out` is [K, rows] row-major, `xK` is
/// [K, cols] row-major.  The activation is quantized once per token into
/// that token's slot of q8K/q8kK, then a single pool task covers K*rows
/// virtual rows (`tok = r/rows`, `rr = r%rows`).  It is the *same*
/// `quantN` task as the serial `matvecQuant` with `k > 1`; the per-row
/// kernel call is identical, so out[t*rows+rr] is bit-identical to
/// `linMatvec(out[t*rows .. t*rows+rows], w, xK[t*cols..])` per token.
/// There is no per-type branch here either: only the activation block
/// width of the weight decides between the two shapes.
fn matvecBatchK(out: []f32, w: LinearW, xK: []const f32, k: usize, q8K: []u8, q8kK: []u8, pool: ?*pool_mod.Pool) void {
    const rows = out.len / k;
    std.debug.assert(out.len % k == 0);
    switch (w) {
        .quant => |qrow| {
            const cols = qrow.cols();
            std.debug.assert(xK.len >= k * cols);
            const act_b32 = blockElemsOf(qrow.ttype) == 32;
            if (act_b32) {
                const dot = switch (qrow.ttype) {
                    .q8_0 => simd.d32row.q8_0_q8_0,
                    .q4_0 => simd.d32row.q4_0_q8_0,
                    .q4_1 => simd.d32row.q4_1_q8_0,
                    .iq4_nl => simd.d32row.iq4_nl_q8_0,
                    .q5_0 => simd.d32row.q5_0_q8_0,
                    .q5_1 => simd.d32row.q5_1_q8_0,
                    .q1_0 => simd.d32row.q1_0_q8_0,
                    .q2_0 => simd.d32row.q2_0_q8_0,
                    .mxfp4 => simd.d32row.mxfp4_q8_0,
                    .nvfp4 => simd.d32row.nvfp4_q8_0,
                    .f16 => simd.d32row.f16_q8_0,
                    .bf16 => simd.d32row.bf16_q8_0,
                    else => null,
                } orelse {
                    matvecBatchKScalar(out, qrow, xK, k, rows, cols, pool);
                    return;
                };
                const blen = q8ScratchBytes(cols);
                std.debug.assert(q8K.len >= k * blen);
                var t: usize = 0;
                while (t < k) : (t += 1)
                    simd.actQ8_0(cols, xK[t * cols ..].ptr, q8K[t * blen ..].ptr);
                pool_mod.parallel(pool, .{ .quantN = .{
                    .act_block = .b32,
                    .out = out.ptr,
                    .data = qrow.data,
                    .act = q8K.ptr,
                    .act_len = blen,
                    .blk_bytes = blockElemBytes(qrow.ttype),
                    .blk_elems = blockElemsOf(qrow.ttype),
                    .bpr = qrow.blocks_per_row,
                    .cols = cols,
                    .rows = rows,
                    .k = k,
                    .dot32 = dot,
                    .dot256 = undefined,
                } });
                return;
            }
            const dot = switch (qrow.ttype) {
                .q2_k => simd.d256.q2_k_q8_k,
                .q3_k => simd.d256.q3_k_q8_k,
                .q4_k => simd.d256.q4_k_q8_k,
                .q5_k => simd.d256.q5_k_q8_k,
                .q6_k => simd.d256.q6_k_q8_k,
                .iq2_xxs => simd.diq.iq2_xxs_q8_k,
                .iq2_xs => simd.diq.iq2_xs_q8_k,
                .iq2_s => simd.diq.iq2_s_q8_k,
                .iq3_xxs => simd.diq.iq3_xxs_q8_k,
                .iq3_s => simd.diq.iq3_s_q8_k,
                .iq1_s => simd.diq.iq1_s_q8_k,
                .iq1_m => simd.diq.iq1_m_q8_k,
                .iq4_xs => simd.diq.iq4_xs_q8_k,
                else => null,
            } orelse {
                matvecBatchKScalar(out, qrow, xK, k, rows, cols, pool);
                return;
            };
            const klen = q8kScratchBytes(cols);
            std.debug.assert(q8kK.len >= k * klen);
            var t: usize = 0;
            while (t < k) : (t += 1)
                simd.actQ8_K(cols, xK[t * cols ..].ptr, q8kK[t * klen ..].ptr);
            pool_mod.parallel(pool, .{ .quantN = .{
                .act_block = .b256,
                .out = out.ptr,
                .data = qrow.data,
                .act = q8kK.ptr,
                .act_len = klen,
                .blk_bytes = qrow.blockBytes(), // one 256-element row-block
                .blk_elems = blockElemsOf(qrow.ttype),
                .bpr = qrow.blocks_per_row,
                .cols = cols,
                .rows = rows,
                .k = k,
                .dot32 = undefined,
                .dot256 = dot,
            } });
        },
        .f32 => |m| {
            const cols = m.len / rows;
            std.debug.assert(xK.len >= k * cols);
            pool_mod.parallel(pool, .{ .f32 = .{
                .out = out.ptr,
                .mat = m,
                .x = xK.ptr,
                .x_len = cols,
                .cols = cols,
                .rows = rows,
                .k = k,
            } });
        },
    }
}

/// Batched twin of `matvecQuantScalar`: identical zig rowDot per block,
/// just with `k` tokens in one task.
fn matvecBatchKScalar(out: []f32, qrow: QuantRow, xK: []const f32, k: usize, rows: usize, cols: usize, pool: ?*pool_mod.Pool) void {
    const rowDot = rowDotFor(qrow.ttype) orelse unreachable;
    pool_mod.parallel(pool, .{ .quant_scalar = .{
        .out = out.ptr,
        .data = qrow.data,
        .x = xK.ptr,
        .x_len = cols,
        .row_bytes = qrow.blockBytes(), // per row-block; pool indexes r*bpr*row_bytes
        .bpr = qrow.blocks_per_row,
        .rows = rows,
        .k = k,
        .dot = rowDot,
    } });
}

/// [rows, cols] so cols = len / rows.

// ---------------------------------------------------------------------------
// Dequantization (row expansion)
// ---------------------------------------------------------------------------

pub const dequantRowFn = *const fn (row: []const u8, out: []f32) void;

/// Row dequantizer per ttype.  These stay pure Zig: the embedding lookup needs
/// *values*, not dots, it runs once per token (not in the hot loop), and the
/// Zig implementations are the reference the C kernels are checked against.
pub fn dequantRowFnFor(ttype: gguf.TensorType) ?dequantRowFn {
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

/// Row-major [rows, cols] f32 matvec: out[r] = sum_c mat[r*cols+c] * x[c].
/// Plain auto-vectorized Zig -- identical on every x86 tier -- so it is not
/// something the backend can usefully accelerate; it only needs the same
/// thread pool as the quantized kernels.
fn matvecF32(out: []f32, mat: []const f32, x: []const f32) void {
    const rows = out.len;
    const cols = x.len;
    std.debug.assert(mat.len == rows * cols);
    pool_mod.parallel(null, .{ .f32 = .{
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
// Weight loading
// ---------------------------------------------------------------------------

/// Load a 1D tensor as f32: f32/f16/bf16 are widened exactly, packed types
/// go through the reference dequant.  Small vectors (norms, biases) use this.
pub fn loadTensorF32(alloc: std.mem.Allocator, g: *const gguf.File, name: []const u8) ![]f32 {
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
        // Packed types: walk the tensor in the type's own blocks, so the
        // framing comes from the layout table instead of a per-type table.
        else => {
            const deq = dequantRowFnFor(t.ttype) orelse return error.UnsupportedTensorType;
            const be = gguf.blockElems(t.ttype);
            const bb = gguf.blockBytes(t.ttype);
            if (n % be != 0) return error.NotMultipleOfBlock;
            var b: usize = 0;
            while (b < n / be) : (b += 1) {
                deq(data[b * bb ..][0..bb], out[b * be ..][0..be]);
            }
        },
    }
    return out;
}


/// Zero-copy view of a quantized 2D weight; geometry comes from the GGUF
/// layout table (see gguf.QuantRow).
fn quantRowOf(g: *const gguf.File, t: *const gguf.Tensor) backend.LoadError!QuantRow {
    if (!t.ttype.isKnown()) return error.UnsupportedTensorType;
    const data = g.tensorData(t) catch return error.MissingTensor;
    return QuantRow.init(t.*, data);
}

/// Load a 2D linear weight: quantized types stay in their native packed
/// layout (zero-copy view into the mmap); f32/f16/bf16 are expanded to f32.
fn loadLinear(alloc: std.mem.Allocator, g: *const gguf.File, name: []const u8) backend.LoadError!LinearW {
    const t = g.findTensor(name) orelse return error.MissingTensor;
    switch (t.ttype) {
        .q8_0, .q4_0, .q4_1, .q4_k, .q5_k, .q6_k, .q3_k, .iq4_xs, .iq4_nl, .iq3_s, .iq3_xxs, .iq2_s, .iq2_xxs, .q2_k, .iq2_xs, .iq1_s, .iq1_m => {
            return .{ .quant = try quantRowOf(g, t) };
        },
        // f16/bf16 carry no block layout, so the packed path cannot hold them;
        // they are expanded to f32 here (exactly, no rounding).
        .f16, .bf16, .f32 => {
            const n: usize = @intCast(t.nElems());
            const data = g.tensorData(t) catch return error.MissingTensor;
            const out = alloc.alloc(f32, n) catch return error.OutOfMemory;
            errdefer alloc.free(out);
            switch (t.ttype) {
                .f32 => @memcpy(out, @as([*]const f32, @ptrCast(@alignCast(data)))[0..n]),
                .f16 => {
                    const src: [*]const u16 = @ptrCast(@alignCast(data));
                    for (0..n) |i| out[i] = @floatCast(@as(f16, @bitCast(src[i])));
                },
                .bf16 => {
                    // BF16 is the top 16 bits of an f32: widen by shifting,
                    // which is exact for every bf16 value.
                    const src: [*]const u16 = @ptrCast(@alignCast(data));
                    for (0..n) |i| out[i] = @as(f32, @bitCast(@as(u32, @intCast(src[i])) << 16));
                },
                else => unreachable,
            }
            return .{ .f32 = out };
        },
        else => return error.UnsupportedTensorType,
    }
}

// ---------------------------------------------------------------------------
// Backend state + Exec implementation
// ---------------------------------------------------------------------------

pub const CpuBackend = struct {
    alloc: std.mem.Allocator,
    g: *const gguf.File,
    pool: ?*pool_mod.Pool = null,
    /// Q8_0 activation scratch for one token.
    q8: []u8 = &.{},
    /// Q8_K activation scratch for one token.
    q8k: []u8 = &.{},
    /// Batched scratch for a K-token prefill chunk.
    q8K: []u8 = &.{},
    q8kK: []u8 = &.{},

    pub fn init(alloc: std.mem.Allocator, g: *const gguf.File) CpuBackend {
        // Bind the dispatch tables to the widest ISA tier this CPU supports
        // (CPUID + XGETBV).  Idempotent; before this the pointers are the
        // baseline scalar implementations.
        simd.init();
        return .{ .alloc = alloc, .g = g };
    }

    pub fn exec(self: *CpuBackend) backend.Exec {
        return .{ .ptr = self, .vtable = &vtable };
    }

    pub fn deinit(self: *CpuBackend) void {
        self.freeScratch();
        if (self.pool) |p| pool_mod.deinit(p, self.alloc);
        self.pool = null;
    }

    fn freeScratch(self: *CpuBackend) void {
        const a = self.alloc;
        for ([_]*[]u8{ &self.q8, &self.q8k, &self.q8K, &self.q8kK }) |p| {
            if (p.len > 0) a.free(p.*);
            p.* = &.{};
        }
    }

    // ---- vtable targets -----------------------------------------------------

    fn vLoadLinear(ctx: *anyopaque, name: []const u8) backend.LoadError!LinearW {
        const self: *CpuBackend = @ptrCast(@alignCast(ctx));
        return loadLinear(self.alloc, self.g, name);
    }

    fn vFreeLinear(ctx: *anyopaque, w: LinearW) void {
        const self: *CpuBackend = @ptrCast(@alignCast(ctx));
        w.deinit(self.alloc);
    }

    fn vAllocScratch(ctx: *anyopaque, max_cols: usize) anyerror!void {
        const self: *CpuBackend = @ptrCast(@alignCast(ctx));
        const a = self.alloc;
        if (self.q8k.len == 0) self.q8k = try a.alloc(u8, q8kScratchBytes(max_cols));
        if (self.q8.len == 0) self.q8 = try a.alloc(u8, q8ScratchBytes(max_cols));
        // Sizing guard: the scratch is sized from the config, so verify it
        // really covers the widest matvec input before the first matvec runs.
        // Without this a sizing miss is an out-of-bounds write inside a
        // kernel instead of a named load-time error.
        try checkQ8Scratch(self.q8, max_cols);
    }

    fn vAllocScratchK(ctx: *anyopaque, max_cols: usize, k: usize) anyerror!void {
        const self: *CpuBackend = @ptrCast(@alignCast(ctx));
        const a = self.alloc;
        if (self.q8kK.len < q8kScratchBytes(max_cols) * k) {
            if (self.q8kK.len > 0) a.free(self.q8kK);
            self.q8kK = try a.alloc(u8, q8kScratchBytes(max_cols) * k);
        }
        if (self.q8K.len < q8ScratchBytes(max_cols) * k) {
            if (self.q8K.len > 0) a.free(self.q8K);
            self.q8K = try a.alloc(u8, q8ScratchBytes(max_cols) * k);
        }
    }

    fn vFreeScratch(ctx: *anyopaque) void {
        const self: *CpuBackend = @ptrCast(@alignCast(ctx));
        self.freeScratch();
    }

    fn vDequantRow(ctx: *anyopaque, dst: []f32, w: QuantRow, row: usize) void {
        _ = ctx;
        // Q8_0 must NOT go through the generic per-row path: `dequantRowFnFor`
        // covers the 256-element K/IQ types, whose row framing is QK_K-based.
        // Q8_0 packs plain 32-element blocks with no 256 framing, so a row of
        // it is not a whole number of 256-element units and the generic path
        // silently drops the tail whenever cols % 256 != 0.
        if (w.ttype == .q8_0) {
            quant.dequantizeQ8_0(w.row(row), dst) catch unreachable;
            return;
        }
        const deq = dequantRowFnFor(w.ttype) orelse unreachable;
        deq(w.row(row), dst);
    }

    fn vMatvec(ctx: *anyopaque, out: []f32, w: LinearW, x: []const f32) void {
        const self: *CpuBackend = @ptrCast(@alignCast(ctx));
        switch (w) {
            .f32 => |m| matvecF32(out, m, x),
            .quant => |q| matvecQuant(out, q, x, self.q8k, self.q8, self.pool),
        }
    }

    fn vMatvecBatchK(ctx: *anyopaque, out: []f32, w: LinearW, xK: []const f32, k: usize) void {
        const self: *CpuBackend = @ptrCast(@alignCast(ctx));
        matvecBatchK(out, w, xK, k, self.q8K, self.q8kK, self.pool);
    }

    fn vSetThreads(ctx: *anyopaque, io: Io, nt: usize) void {
        const self: *CpuBackend = @ptrCast(@alignCast(ctx));
        self.pool = pool_mod.init(io, self.alloc, nt) catch null; // serial if spawn fails
        pool_mod.prof = prof;
        pool_mod.prof_io = io;
        prof_io = io;
    }

    fn vProfPhase(ctx: *anyopaque, on: bool) void {
        _ = ctx;
        pool_mod.prof_phase = on;
    }

    fn vProfReset(ctx: *anyopaque) void {
        _ = ctx;
        pool_mod.profReset();
    }

    fn vName(ctx: *anyopaque) []const u8 {
        _ = ctx;
        return "cpu";
    }

    fn vDescribe(ctx: *anyopaque) []const u8 {
        _ = ctx;
        return @tagName(simd.tier);
    }

    fn vSetTier(ctx: *anyopaque, tier_name: []const u8) void {
        _ = ctx;
        setSimdTier(tier_name);
    }

    fn vReportProfile(ctx: *anyopaque) void {
        _ = ctx;
        profReportMatvec();
    }

    const vtable = backend.Exec.VTable{
        .loadLinear = vLoadLinear,
        .freeLinear = vFreeLinear,
        .allocScratch = vAllocScratch,
        .allocScratchK = vAllocScratchK,
        .freeScratch = vFreeScratch,
        .dequantRow = vDequantRow,
        .matvec = vMatvec,
        .matvecBatchK = vMatvecBatchK,
        .setThreads = vSetThreads,
        .profPhase = vProfPhase,
        .profReset = vProfReset,
        .name = vName,
        .describe = vDescribe,
        .setTier = vSetTier,
        .reportProfile = vReportProfile,
    };
};

// ---------------------------------------------------------------------------
// SIMD tier selection
// ---------------------------------------------------------------------------

/// Force a SIMD tier (QWEN35_SIMD_TIER).  Unknown names are ignored so a typo
/// leaves the auto-detected tier in place instead of failing the run.
pub fn setSimdTier(tier_name: []const u8) void {
    const t: simd.Tier = if (std.mem.eql(u8, tier_name, "baseline"))
        .baseline
    else if (std.mem.eql(u8, tier_name, "avx2"))
        .avx2
    else if (std.mem.eql(u8, tier_name, "avx512"))
        .avx512
    else if (std.mem.eql(u8, tier_name, "avx512vnni"))
        .avx512vnni
    else if (std.mem.eql(u8, tier_name, "avx512vbmi"))
        .avx512vbmi
    else
        return;
    simd.force_tier = t;
}

// ---------------------------------------------------------------------------
// Tests: these exercise the kernels, not the model, so they live with the
// kernels (they were in engine.zig before the backend split).
// ---------------------------------------------------------------------------

test "matvecQuant Q8_0 matches f32 matvec on dequantized weights" {
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
        try quant.quantizeQ8_0(&f32blk, wq[blk * q8_0_block_bytes ..][0..q8_0_block_bytes]);
        try quant.dequantizeQ8_0(wq[blk * q8_0_block_bytes ..][0..q8_0_block_bytes], wf[blk * q8_0_block_elems ..][0..q8_0_block_elems]);
    }
    for (&x) |*v| v.* = rand.float(f32) * 2.0 - 1.0;

    const w: QuantRow = .{ .data = &wq, .ttype = .q8_0, .rows = rows, .blocks_per_row = cols / q8_0_block_elems };
    var out_q8: [rows]f32 = undefined;
    var q8scratch: [cols / q8_0_block_elems * q8_0_block_bytes]u8 = undefined;
    var q8kscratch: [q8kScratchBytes(cols)]u8 = undefined;
    matvecQuant(&out_q8, w, &x, &q8kscratch, &q8scratch, null);

    // reference: f32 matvec over dequantized weights AND dequantized x
    // (the q8 path necessarily quantizes x, so feed the same values in)
    var xq: [cols]f32 = undefined;
    for (0..cols / q8_0_block_elems) |blk| {
        try quant.dequantizeQ8_0(q8scratch[blk * q8_0_block_bytes ..][0..q8_0_block_bytes], xq[blk * q8_0_block_elems ..][0..q8_0_block_elems]);
    }
    var out_f32: [rows]f32 = undefined;
    matvecF32(&out_f32, &wf, &xq);

    for (out_q8, out_f32) |a, b| {
        // same math (i32 exact per block vs f32 accumulate): tiny rounding only
        try std.testing.expectApproxEqAbs(b, a, @abs(b) * 1e-5 + 1e-4);
    }
}

test "matvecQuant Q8_0 zero x gives zero out" {
    const cols = 32;
    var wq: [2 * q8_0_block_bytes]u8 = undefined;
    const wvals = [3]f32{ 1, 2, 3 } ++ @as([29]f32, @splat(0));
    try quant.quantizeQ8_0(&wvals, wq[0..q8_0_block_bytes]);
    try quant.quantizeQ8_0(&wvals, wq[q8_0_block_bytes..]);
    var x: [cols]f32 = @splat(0);
    var out: [2]f32 = undefined;
    var scratch: [q8_0_block_bytes]u8 = undefined;
    var kscratch: [q8kScratchBytes(cols)]u8 = undefined;
    matvecQuant(&out, .{ .data = &wq, .ttype = .q8_0, .rows = 2, .blocks_per_row = 1 }, &x, &kscratch, &scratch, null);
    try std.testing.expectEqual(@as(f32, 0), out[0]);
    try std.testing.expectEqual(@as(f32, 0), out[1]);
}

test "q8_0 dequantize: hand-computed block" {
    var data: [q8_0_block_bytes]u8 = @splat(0);
    std.mem.writeInt(u16, data[0..2], @bitCast(@as(f16, 1.5)), .little);
    const zeros: [27]i8 = @splat(0);
    const qs = [5]i8{ 0, 1, -1, 32, -128 } ++ zeros;
    for (qs, 0..) |q, i| data[2 + i] = @bitCast(q);
    var out: [32]f32 = undefined;
    try quant.dequantizeQ8_0(&data, &out);
    try std.testing.expectEqual(@as(f32, 0), out[0]);
    try std.testing.expectEqual(@as(f32, 1.5), out[1]);
    try std.testing.expectEqual(@as(f32, -1.5), out[2]);
    try std.testing.expectEqual(@as(f32, 48.0), out[3]);
    try std.testing.expectEqual(@as(f32, -192.0), out[4]);
}

test "q8_0 dequantize rejects bad sizes" {
    var data: [2 * q8_0_block_bytes]u8 = @splat(0);
    var out: [33]f32 = undefined;
    try std.testing.expectError(error.NotMultipleOfBlock, quant.dequantizeQ8_0(&data, &out));
    var out2: [64]f32 = undefined;
    try std.testing.expectError(error.SizeMismatch, quant.dequantizeQ8_0(data[0 .. 2 * q8_0_block_bytes - 1], &out2));
}

test "matvecF32 2x3 hand computed" {
    const m = [_]f32{ 1, 2, 3, 4, 5, 6 };
    const x = [_]f32{ 7, 8, 9 };
    var out: [2]f32 = undefined;
    matvecF32(&out, &m, &x);
    try std.testing.expectEqual(@as(f32, 50), out[0]);
    try std.testing.expectEqual(@as(f32, 122), out[1]);
}
