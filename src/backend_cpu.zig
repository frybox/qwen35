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
const simd_api = @import("simd_api.zig");

const LinearW = backend.LinearW;
const QuantRow = backend.QuantRow;
const Buf = backend.Buf;
const ew = @import("elementwise.zig");
const sampler = @import("sampler.zig");
const Io = std.Io;

pub const blockElemsOf = gguf.blockElems;
pub const blockElemBytes = gguf.blockBytes;

/// Activation family of a weight type: which activation block its matvec
/// consumes (Q8_0 = 34 B / 32 elems, Q8_K = 292 B / 256 elems).
///
/// This is NOT derivable from `gguf.blockElems` alone, and keying on it is a
/// live bug: BF16/F16 weights store ONE element per 2-byte block
/// (`blockLayout` = elems 1 / bytes 2) and are framed into 256-element GGUF
/// row-blocks, so `blockElems == 32` is false for them -- they fell into the
/// Q8_K branch, found no dot kernel there, and landed in `matvecQuantScalar`,
/// which reads ONE 2-byte "row" per 256-element activation window (garbage).
/// Their dot kernel is `bf16_q8_0` / `f16_q8_0` -- the Q8_0 family.  So the
/// family is stated explicitly here; the single source CPU and CUDA share.
pub const ActFamily = enum { q8_0, q8_k };
pub fn actFamily(ttype: gguf.TensorType) ActFamily {
    return switch (ttype) {
        // 2-byte unquantized weights: no block scale, decoded element-by-element
        // inside the dot, activated with Q8_0.
        .f16, .bf16 => .q8_0,
        else => if (blockElemsOf(ttype) == 32) .q8_0 else .q8_k,
    };
}

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
    // f16/bf16: no activation quantization.  The dequant happens inside the C
    // dot element-by-element, so the activation stays the engine's f32 vector
    // and the row's sum is bit-identical to the materialized-f32 path.
    if (w.ttype == .f16 or w.ttype == .bf16) {
        matvecPacked(out, w, x, pool);
        return;
    }
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
    // Activation family: Q8_0 (32-elem blocks) or Q8_K (256-elem blocks).
    // Explicit list, not `blockElems() == 32` -- see `actFamily`.
    const act_b32 = actFamily(w.ttype) == .q8_0;
    // `bpr` is always in row-blocks, and only q8_0 has 32-element row-blocks
    // (see `QuantRow.rowBlockElems`).  The framed 32-element-block types
    // (q4_0/q4_1/q5_*/iq4_nl/mxfp4...) still count 256-element row-blocks, so
    // they divide by 256 here exactly like the K/IQ types -- dividing by 32
    // would demand a bpr 8x too large and trip this assert on every Q4_0 row.
    std.debug.assert(bpr == cols / gguf.rowBlockElems(w.ttype));
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
/// shared by the serial matvec and the K-token batch (the entry point for
/// types with no C row dot, and for IQ types).  `act_b32` only decides which
/// activation block width the caller's scratch arrays were sized for; it is
/// unused here because this path never quantizes the activation.
///
/// WARNING: this path is only correct for 256-element-block types.  It hands
/// the pool `.row_bytes = blockBytes()` (ONE block) while the pool's
/// `quant_scalar` loop reads `xs[b*256 .. b*256+256]` -- i.e. it assumes one
/// 256-element row-block per `bpr` step.  For q4_0/iq4_nl (block=32,
/// row-block=256) the row stride and the activation window are therefore
/// wrong.  Not reachable for those types today (they have C row dots), but do
/// not use this as a Q4_0/IQ4_NL oracle.
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
            if (qrow.ttype == .f16 or qrow.ttype == .bf16) {
                // Same as the single-token packed path: f32 activations, dequant
                // inside the C dot.  `tok = r / rows` indexes the K tokens.
                pool_mod.parallel(pool, .{ .w16 = .{
                    .out = out.ptr,
                    .data = qrow.data,
                    .x = xK.ptr,
                    .x_len = cols,
                    .cols = cols,
                    .rows = rows,
                    .k = k,
                    .dot = if (qrow.ttype == .bf16) simd_api.dotBf16F32 else simd_api.dotF16F32,
                } });
                return;
            }
            const act_b32 = actFamily(qrow.ttype) == .q8_0;
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
        // Q8_0 is registered too: its row-block unit is its own 32-element
        // block, so `dequantRowInto`'s generic walk gives it exactly one block
        // per call.  No per-type special case needed anywhere downstream.
        .q8_0 => &quant.dequantQ8_0,
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
        // 2-byte unquantized types: no block metadata, one element per 2-byte
        // block.  Registered so `dequantRowInto` (and the embedding lookup that
        // runs through it) can serve BF16 token rows without any f32 copy.
        .f16 => &quant.dequantF16,
        .bf16 => &quant.dequantBF16,
        else => null,
    };
}

/// Row-major [rows, cols] f32 matvec: out[r] = sum_c mat[r*cols+c] * x[c].
///
/// `pool` is the caller's thread pool and MUST be forwarded: the f32 path used
/// to call `parallel(null, ...)`, so every bf16/f16/f32 tensor (the whole BF16
/// model!) was decoded single-threaded -- ~1.5 tok/s at `--threads 1` AND at
/// `--threads 32`, because the argument was dropped here, not because the work
/// was small.
fn matvecF32(out: []f32, mat: []const f32, x: []const f32, pool: ?*pool_mod.Pool) void {
    const rows = out.len;
    const cols = x.len;
    std.debug.assert(mat.len == rows * cols);
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

/// f16/bf16 matvec over the mmap weight bytes, with the activation left as the
/// engine's own f32 vector.  The dequant happens inside the C dot, per element
/// (`kernels_f32.c`), so:
///   - no f32 copy of the weights exists (the .f32 materialization is gone),
///   - the values summed are exactly the f32 values the materialized copy
///     would have held, so the row's result matches the f32-materialized path
///     bit-for-bit, and
///   - decode reads 2 B/elem instead of 4.
/// This is `ggml_vec_mad_bf16` in llama.cpp.
fn matvecPacked(out: []f32, w: QuantRow, x: []const f32, pool: ?*pool_mod.Pool) void {
    const rows = out.len;
    const cols = w.cols();
    std.debug.assert(x.len >= cols);
    std.debug.assert(w.data.len >= rows * cols * 2);
    pool_mod.parallel(pool, .{ .w16 = .{
        .out = out.ptr,
        .data = w.data,
        .x = x.ptr,
        .x_len = cols,
        .cols = cols,
        .rows = rows,
        .k = 1,
        .dot = if (w.ttype == .bf16) simd_api.dotBf16F32 else simd_api.dotF16F32,
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

/// Load a 2D linear weight: every type stays in its native packed layout as a
/// zero-copy view into the GGUF mapping, `.f32` included.
///
/// f16/bf16 used to be expanded to f32 here, which doubled the resident
/// footprint of a BF16 model (2 B/elem on disk -> 4 B/elem in RAM) and made
/// every BF16 matvec a pure f32 dot; the CUDA backend inherited the same
/// waste because it uploads what `loadLinear` returns.  They are now
/// `QuantRow`s with the GGUF row framing they actually use (256-element
/// row-blocks, 1 element per 2-byte block -- see `gguf.rowBlockElems`), and
/// the C kernels `f16_q8_0`/`bf16_q8_0` decode each element on the fly inside
/// the dot.  `.f32` tensors are the same deal: the tensor bytes are a
/// `[]const f32` view of the mapping (the GGUF format pads every tensor to a
/// 32-byte alignment, which covers the 4-byte f32 alignment), and the f32
/// dot kernel reads through it.  The 1-D `loadTensorF32` tensors are the
/// exception -- they are expanded in place (softplus / -exp(A) on `ssm_a`),
/// so they keep an owned copy.
fn loadLinear(g: *const gguf.File, name: []const u8) backend.LoadError!LinearW {
    const t = g.findTensor(name) orelse return error.MissingTensor;
    switch (t.ttype) {
        .q8_0, .q4_0, .q4_1, .q4_k, .q5_k, .q6_k, .q3_k, .iq4_xs, .iq4_nl, .iq3_s, .iq3_xxs, .iq2_s, .iq2_xxs, .q2_k, .iq2_xs, .iq1_s, .iq1_m, .f16, .bf16 => {
            return .{ .quant = try quantRowOf(g, t) };
        },
        .f32 => {
            const n: usize = @intCast(t.nElems());
            const data = g.tensorData(t) catch return error.MissingTensor;
            // Zero-copy: the mapping is read-only and outlives this view.
            return .{ .f32 = @as([*]const f32, @ptrCast(@alignCast(data)))[0..n] };
        },
        else => return error.UnsupportedTensorType,
    }
}

// ---------------------------------------------------------------------------
// Backend state + Exec implementation
// ---------------------------------------------------------------------------

/// Dequantize one row of `w` into `dst` (len == w.cols()).
///
/// Q8_0 takes the short path: its rows ARE a plain run of 32-element blocks
/// (`rowBlockElems(.q8_0) == 32`, the only type whose row-block unit is its
/// native block), so `dequantizeQ8_0` decodes the whole row in one call, with
/// size validation.  Going through the generic loop would be correct too --
/// `dequantRowFnFor(.q8_0)` is registered precisely so the row-block walk can
/// serve every packed type, see `engine.refErrF64` -- but one call over 34-byte
/// blocks is cheaper than one call per 32 elements and it keeps the
/// `row.len % 256 != 0` question off the table for the widest Q8_0 tensors.
///
/// `dequantRowFn` is the one-256-element-row-block reference: it loops `0..8`
/// over 8 x 32-element sub-blocks and writes exactly 256 floats.  A row wider
/// than 256 elements therefore has to be fed block by block -- handing the
/// whole row to one call fills only the first 256 entries and leaves the rest
/// of `dst` untouched.  That is a real case: the q4_0/q4_1/iq4_nl
/// 32-element-block types are still framed as 256-element row-blocks by GGUF,
/// so they reach this with cols > 256 (e.g. 1024).
pub fn dequantRowInto(dst: []f32, w: QuantRow, row: usize) void {
    if (w.ttype == .q8_0) {
        quant.dequantizeQ8_0(w.row(row), dst) catch unreachable;
        return;
    }
    const deq = dequantRowFnFor(w.ttype) orelse unreachable;
    // One call per 256-element row-block: `deq` consumes 8 x 32-element
    // sub-blocks and emits exactly `rbe` floats.  Walk `blocks_per_row`
    // (the row-block unit), NOT `rowBlocks()` (the type's own block unit,
    // which for q4_0 is 8x larger).
    const rbe = w.rowBlockElems();
    const blk = w.blockBytes();
    const per_rb = rbe / w.blockElems(); // 32-element blocks per row-block
    const src = w.row(row);
    var b: usize = 0;
    while (b < w.blocks_per_row) : (b += 1) {
        deq(src[b * per_rb * blk ..], dst[b * rbe ..]);
    }
}

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
    /// Every region handed out by `allocBuf`, so `freeScratch` releases them
    /// together with the matvec scratch.  The engine never frees a region
    /// individually; a device backend keeps the same contract (all regions
    /// released by `freeScratch`).
    bufs: std.ArrayList([]f32) = .empty,

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
        for (self.bufs.items) |s| a.free(s);
        self.bufs.clearAndFree(a);
    }

    // ---- vtable targets -----------------------------------------------------

    fn vLoadLinear(ctx: *anyopaque, name: []const u8) backend.LoadError!LinearW {
        const self: *CpuBackend = @ptrCast(@alignCast(ctx));
        return loadLinear(self.g, name);
    }

    fn vFreeLinear(ctx: *anyopaque, w: LinearW) void {
        _ = ctx;
        _ = w;
        // Zero-copy views into the mapping; nothing to release.
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

    fn vDequantRow(ctx: *anyopaque, dst: Buf, w: QuantRow, row: usize) void {
        _ = ctx;
        dequantRowInto(dst.slice(), w, row);
    }

    fn vMatvec(ctx: *anyopaque, out: Buf, w: LinearW, x: Buf) void {
        const self: *CpuBackend = @ptrCast(@alignCast(ctx));
        switch (w) {
            .f32 => |m| matvecF32(out.slice(), m, x.slice(), self.pool),
            .quant => |q| matvecQuant(out.slice(), q, x.slice(), self.q8k, self.q8, self.pool),
        }
    }

    fn vMatvecBatchK(ctx: *anyopaque, out: Buf, w: LinearW, xK: Buf, k: usize) void {
        const self: *CpuBackend = @ptrCast(@alignCast(ctx));
        matvecBatchK(out.slice(), w, xK.slice(), k, self.q8K, self.q8kK, self.pool);
    }

    /// The CPU `Buf` is a host slice allocated from the backend's allocator;
    /// every region is recorded so `freeScratch` releases all of them.
    fn vAllocBuf(ctx: *anyopaque, len: usize) anyerror!Buf {
        const self: *CpuBackend = @ptrCast(@alignCast(ctx));
        const s = try self.alloc.alloc(f32, len);
        self.bufs.append(self.alloc, s) catch return error.OutOfMemory;
        return Buf.wrap(s);
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

    /// End of the prefill loop.  The pool profiler already separates prefill
    /// from decode through `prof_phase` sampling, so nothing to do -- this hook
    /// exists for the CUDA backend's GPU-side window.
    fn vProfPrefillEnd(ctx: *anyopaque) void {
        _ = ctx;
    }

    /// Decode-step boundary: nothing to do on the CPU.  The bounds exist for a
    /// device backend that can replay a step as one graph; the CPU executes
    /// straight-line code, so there is nothing to capture or replay.
    fn vStepBegin(ctx: *anyopaque, pos: u32) void {
        _ = ctx;
        _ = pos;
    }

    fn vStepEnd(ctx: *anyopaque) void {
        _ = ctx;
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

    /// `Exec.profReport`: the CPU backend's complete one-shot dump is its
    /// per-(ttype, rows) matvec table, which `reportProfile` already prints
    /// (the pool's `PROF-P` line is emitted by the engine/pool itself).
    fn vProfReport(ctx: *anyopaque) void {
        _ = ctx;
        profReportMatvec();
    }

    // ---- elementwise / state slots ------------------------------------------
    //
    // The CPU `Buf` is a host slice, so each target is the engine's historical
    // Zig kernel, reached through src/elementwise.zig.  `ctx` is unused: these
    // kernels need no backend state, which is why a composing backend (the CUDA
    // trampoline) can hand out these same targets unchanged.

    fn vRmsnorm(ctx: *anyopaque, out: Buf, x: Buf, w: Buf, eps: f32) void {
        _ = ctx;
        ew.bufRmsnorm(out, x, w, eps);
    }

    /// Row-batched rmsnorm: `rows` independent rows of `w.len`.  Spelled as the
    /// loop over consecutive spans, i.e. literally the per-row calls the engine
    /// used to make, so the CPU path stays bit-identical to it and this slot
    /// cannot diverge from `vRmsnorm` on any single row.
    fn vRmsnormRows(ctx: *anyopaque, out: Buf, x: Buf, w: Buf, rows: usize, eps: f32) void {
        _ = ctx;
        const n = w.len;
        std.debug.assert(out.len == rows * n and x.len == rows * n);
        for (0..rows) |r| ew.bufRmsnorm(out.span(r * n, n), x.span(r * n, n), w, eps);
    }

    fn vRope64(ctx: *anyopaque, x: Buf, pos: u32, base: f32) void {
        _ = ctx;
        ew.bufRope64(x, pos, base);
    }

    /// Row-batched rope64: `rows` tokens `pos..pos+rows-1`, each `heads * hd`
    /// floats wide.  The CPU oracle is literally the loop of per-row calls, so
    /// row r is bit-identical to `vRope64` on that row with `pos + r` (same
    /// convention as `vRmsnormRows`).
    fn vRope64Rows(ctx: *anyopaque, x: Buf, pos: u32, base: f32, rows_in: usize, heads: usize, hd: usize) void {
        _ = ctx;
        const row_len = heads * hd;
        std.debug.assert(row_len != 0 and x.len % row_len == 0 and x.len / row_len == rows_in);
        for (0..rows_in) |r|
            ew.bufRope64(x.span(r * row_len, row_len), pos + @as(u32, @intCast(r)), base);
    }

    fn vL2norm(ctx: *anyopaque, x: Buf) void {
        _ = ctx;
        ew.bufL2norm(x);
    }

    /// Row-batched l2norm: `rows` independent rows of `n`.  Loop over the same
    /// per-row calls as `vL2norm` (see `vRmsnormRows` for why the loop shape of
    /// this slot is load-bearing on the CPU path).
    fn vL2normRows(ctx: *anyopaque, x: Buf, n: usize) void {
        _ = ctx;
        std.debug.assert(n != 0 and x.len % n == 0);
        var r: usize = 0;
        while (r < x.len) : (r += n) ew.bufL2norm(x.span(r, n));
    }

    /// Oracle for `matvecPair`: literally two `matvec` calls.  The fused device
    /// path must produce exactly these bytes, so writing the oracle as anything
    /// other than the pair would defeat the purpose.
    fn vMatvecPair(ctx: *anyopaque, out: Buf, a: backend.LinearW, b: backend.LinearW, x: Buf) void {
        const r = out.len / 2;
        vMatvec(ctx, out.span(0, r), a, x);
        vMatvec(ctx, out.span(r, r), b, x);
    }

    fn vScaleVec(ctx: *anyopaque, x: Buf, s: f32) void {
        _ = ctx;
        ew.bufScaleVec(x, s);
    }

    /// Oracle for `l2normQkScale`: literally the three calls it replaces, in
    /// the same order.  Keeping it as the composition (rather than a fused
    /// scalar loop) is the point -- the test asserts the device kernel agrees
    /// with the un-fused semantics, so writing the oracle any other way would
    /// hide a drift in either of them.
    fn vL2normQkScale(ctx: *anyopaque, x: Buf, nk: usize, kd: usize, k_offset: usize, qscale: f32) void {
        const nb = nk * kd;
        std.debug.assert(nk != 0 and kd != 0);
        std.debug.assert(k_offset >= nb and x.len >= k_offset + nb);
        vL2normRows(ctx, x.span(0, nb), kd);
        vL2normRows(ctx, x.span(k_offset, nb), kd);
        vScaleVec(ctx, x.span(0, nb), qscale);
    }

    fn vSiluInPlace(ctx: *anyopaque, x: Buf) void {
        _ = ctx;
        ew.bufSiluInPlace(x);
    }

    fn vMulSilu(ctx: *anyopaque, out: Buf, gate: Buf) void {
        _ = ctx;
        ew.bufMulSilu(out, gate);
    }

    fn vMulSigmoid(ctx: *anyopaque, out: Buf, gate: Buf) void {
        _ = ctx;
        ew.bufMulSigmoid(out, gate);
    }

    fn vSoftmax(ctx: *anyopaque, x: Buf) void {
        _ = ctx;
        ew.bufSoftmax(x);
    }

    fn vAddResid(ctx: *anyopaque, dst: Buf, src: Buf) void {
        _ = ctx;
        ew.bufAddResid(dst, src);
    }

    fn vSwiglu(ctx: *anyopaque, gate: Buf, up: Buf) void {
        _ = ctx;
        ew.bufSwiglu(gate, up);
    }

    fn vConvStep(ctx: *anyopaque, conv: Buf, y: Buf, wconv: Buf, k: usize) void {
        _ = ctx;
        ew.bufConvStep(conv, y, wconv, k);
    }

    fn vGdnStep(ctx: *anyopaque, state: Buf, k: Buf, v: Buf, q: Buf, km: Buf, delta: Buf, out: Buf, gamma: f32, beta: f32, vd: usize) void {
        _ = ctx;
        ew.bufGdnStep(state, k, v, q, km, delta, out, gamma, beta, vd);
    }

    /// Batched `gdnStep`: the `heads` single-row calls, in order, on the same
    /// per-head views.  This IS the reference for the device kernel -- the
    /// arithmetic is `bufGdnStep`'s unchanged, so a bit-identical result to the
    /// engine's old `for (0..nv) exec.gdnStep(...)` loop is the acceptance test,
    /// not a hope.
    ///
    /// The kv-head mapping `kh = h % nk` lives here because it is the loop's
    /// semantics, not a batching detail: on this model nv (48) > nk (16), so
    /// each k/q row is visited 3 times and `gdnStep` reads `q`/`k` but never
    /// writes them, making the repetition a plain re-read.  (Contrast
    /// `l2norm`, which the engine must apply once per repeat because it is not
    /// idempotent.)
    fn vGdnStepRows(ctx: *anyopaque, state: Buf, k: Buf, v: Buf, q: Buf, km: Buf, delta: Buf, out: Buf,
                    gamma: Buf, beta: Buf, heads: usize, nk: usize, kd: usize, vd: usize) void {
        _ = ctx;
        if (heads == 0 or vd == 0 or nk == 0) return;
        // Both are per-head coefficient regions (`gdnGamma`'s output and its
        // in-place sigmoid(b)); the vtable contract is "at least `heads` long".
        std.debug.assert(gamma.len >= heads and beta.len >= heads);
        const g = gamma.slice();
        const b = beta.slice();
        for (0..heads) |h| {
            const kh = h % nk;
            ew.bufGdnStep(
                state.span(h * kd * vd, kd * vd),
                k.span(kh * kd, kd),
                v.span(h * vd, vd),
                q.span(kh * kd, kd),
                km.span(h * vd, vd),
                delta.span(h * vd, vd),
                out.span(h * vd, vd),
                g[h],
                b[h],
                vd,
            );
        }
    }

    fn vAttnDecode(ctx: *anyopaque, q: Buf, gate: Buf, kcache: Buf, vcache: Buf, att: Buf,
                   gout: Buf, slot: usize, heads: usize, gqa: usize, hd: usize,
                   kv_dim: usize, scale: f32) void {
        _ = ctx;
        ew.bufAttnDecode(q, gate, kcache, vcache, att, gout, slot, heads, gqa, hd, kv_dim, scale);
    }

    /// CPU oracle for `VTable.gdnStepRowsChunk`: literally `rows` successive
    /// `gdnStepRows` calls, each on the next slice of every strided operand.
    ///
    /// The per-row slices are exactly what the engine's old
    /// `while (t < k) forwardGdnCore(...)` loop passed, so a device kernel that
    /// fuses the loop has something meaningful to be compared against instead
    /// of a copy checked against a copy.
    fn vGdnStepRowsChunk(ctx: *anyopaque, state: Buf, k: Buf, v: Buf, q: Buf, km: Buf, delta: Buf, out: Buf,
                         gamma: Buf, beta: Buf, heads: usize, nk: usize, kd: usize, vd: usize,
                         rows: usize, k_stride: usize, v_stride: usize, q_stride: usize) void {
        if (heads == 0 or vd == 0 or nk == 0) return;
        for (0..rows) |t| {
            vGdnStepRows(ctx,
                state,
                k.span(t * k_stride, nk * kd),
                v.span(t * v_stride, heads * vd),
                q.span(t * q_stride, nk * kd),
                km.span(t * heads * vd, heads * vd),
                delta.span(t * heads * vd, heads * vd),
                out.span(t * heads * vd, heads * vd),
                gamma.span(t * heads, heads),
                beta.span(t * heads, heads),
                heads, nk, kd, vd);
        }
    }

    /// CPU oracle for `VTable.attnChunk`: literally the per-row `attnDecode`
    /// calls the engine made before the chunk slot existed, in the same order.
    /// Keeping the naive loop here is what makes a device implementation's
    /// comparison meaningful instead of a copy checked against a copy.
    fn vAttnChunk(ctx: *anyopaque, q: Buf, gate: Buf, kcache: Buf, vcache: Buf, att: Buf,
                  gout: Buf, slot0: usize, rows: usize, heads: usize, gqa: usize,
                  hd: usize, kv_dim: usize, scale: f32) void {
            const row_len = heads * hd;
            var r: usize = 0;
            while (r < rows) : (r += 1) {
                vAttnDecode(ctx,
                    q.span(r * row_len, row_len),
                    gate.span(r * row_len, row_len),
                    kcache, vcache, att,
                    gout.span(r * row_len, row_len),
                    slot0 + r, heads, gqa, hd, kv_dim, scale);
            }
    }

    fn vGdnGamma(ctx: *anyopaque, a_raw: Buf, b_raw: Buf, ssm_a: Buf, dt_bias: Buf, gamma: Buf, nv: usize) void {
        _ = ctx;
        ew.bufGdnGamma(a_raw, b_raw, ssm_a, dt_bias, gamma, nv);
    }

    /// CPU oracle for `VTable.gdnPrefillCore`: literally the per-token calls the
    /// engine used to make, in the same order.  A device backend must reproduce
    /// these numbers; keeping this as the naive loop is what makes that a
    /// meaningful comparison rather than a copy against a copy.
    fn vGdnPrefillCore(ctx: *anyopaque, qkv: Buf, conv: Buf, wconv: Buf, ab: Buf,
        ssm_a: Buf, dt_bias: Buf, gamma: Buf, tokens: usize, conv_dim: usize,
        ab_a_offset: usize, ab_b_offset: usize, ab_stride: usize,
        gamma_stride: usize, kq_dim: usize, nv: usize, nk: usize, kd: usize,
        conv_k: usize, eps: f32) void {
        _ = ctx;
        const nk_reps = if (nk != 0) nv / nk else 0;
        const qscale = 1.0 / @sqrt(@as(f32, @floatFromInt(kd)));
        var t: usize = 0;
        while (t < tokens) : (t += 1) {
            const qkv_t = qkv.span(t * conv_dim, conv_dim);
            // 1. causal conv + silu over this token's whole conv_dim row.
            ew.bufConvStep(conv, qkv_t, wconv, conv_k);
            // 2. beta (in place) + gamma for this token's heads.
            ew.bufGdnGamma(
                ab.span(ab_a_offset + t * ab_stride, nv),
                ab.span(ab_b_offset + t * ab_stride, nv),
                ssm_a,
                dt_bias,
                gamma.span(t * gamma_stride, nv),
                nv,
            );
            // 3. l2norm(q) then l2norm(k) then q *= qscale, nk_reps times.
            if (nk != 0 and nv % nk == 0) {
                var rep: usize = 0;
                while (rep < nk_reps) : (rep += 1) {
                    ew.bufL2norm(qkv_t.span(0, kq_dim));
                    ew.bufL2norm(qkv_t.span(kq_dim, kq_dim));
                    ew.bufScaleVec(qkv_t.span(0, kq_dim), qscale);
                }
            }
        }
        _ = eps; // the CPU l2norm pins its own eps, exactly as `Exec.l2norm` does
    }

    fn vCopyRange(ctx: *anyopaque, dst: Buf, src: Buf) void {
        _ = ctx;
        ew.bufCopyRange(dst, src);
    }

    /// KV append.  Host memory has no captured schedule, so the slot is
    /// irrelevant here and this is `copyRange` (see `VTable.copyRangeSlot`).
    fn vCopyRangeSlot(ctx: *anyopaque, dst: Buf, src: Buf, slot: usize) void {
        _ = slot;
        CpuBackend.vCopyRange(ctx, dst, src);
    }

    /// Host regions on this backend, so the split is the plain per-head copy
    /// (`ew.bufDeinterleaveQG`); it exists as a slot only so a device backend
    /// can replace `2*heads` launches with one.
    fn vDeinterleaveQG(ctx: *anyopaque, dst_q: Buf, dst_gate: Buf, src: Buf, heads: usize, hd: usize) void {
        _ = ctx;
        ew.bufDeinterleaveQG(dst_q, dst_gate, src, heads, hd);
    }

    /// CPU oracle for `VTable.deinterleaveQGStrided`: the per-row calls, in row
    /// order.  Pure move, so this is exact, not approximate.
    fn vDeinterleaveQGStrided(ctx: *anyopaque, dst_q: Buf, dst_gate: Buf, src: Buf,
                              heads: usize, hd: usize, tokens: usize) void {
        const row_len = heads * hd;
        var t: usize = 0;
        while (t < tokens) : (t += 1) {
            vDeinterleaveQG(ctx,
                dst_q.span(t * row_len, row_len),
                dst_gate.span(t * row_len, row_len),
                src.span(t * 2 * row_len, 2 * row_len),
                heads, hd);
        }
    }

    fn vCopyFromHost(ctx: *anyopaque, dst: Buf, src: []const f32) void {
        _ = ctx;
        ew.bufCopyFromHost(dst, src);
    }

    /// Host slices on this backend, so the "copy" is a plain memcpy (and for
    /// the `beta`/`gamma` scalars, the very same region the caller reads).
    fn vCopyToHost(ctx: *anyopaque, dst: []f32, src: Buf) void {
        _ = ctx;
        std.debug.assert(dst.len <= src.len);
        @memcpy(dst, src.slice()[0..dst.len]);
    }

    fn vZeroBuf(ctx: *anyopaque, buf: Buf) void {
        _ = ctx;
        @memset(buf.slice(), 0);
    }

    fn vArgmax(ctx: *anyopaque, logits: Buf) u32 {
        _ = ctx;
        return ew.bufArgmax(logits);
    }

    fn vSample(ctx: *anyopaque, logits: Buf, scratch: Buf, sp: *sampler.Sampler) u32 {
        _ = ctx;
        return ew.bufSample(logits, scratch, sp);
    }

    /// The CPU sampler is synchronous, so `vSample` already returned the real
    /// token and this is the identity.  See `VTable.sampledToken`: the split
    /// exists only for backends that defer the download past `stepEnd`.
    fn vSampledToken(ctx: *anyopaque, hinted: u32) u32 {
        _ = ctx;
        return hinted;
    }

    /// Load a 1-D f32 weight vector by name and hand back a `Buf` over the
    /// widened copy.  The CPU backend keeps the copy in its own allocator and
    /// the `Buf` is a plain host slice; the CUDA backend overrides this to
    /// upload the same widened bytes to the device and hand back the device
    /// region (see `backend_cuda.zig` `vLoadVecF32`).  `loadTensorF32` is the
    /// single widening/dequant definition, so the CPU path is byte-identical
    /// to the historical engine that called it directly.
    fn vLoadVecF32(ctx: *anyopaque, name: []const u8) backend.LoadError!Buf {
        const self: *CpuBackend = @ptrCast(@alignCast(ctx));
        // `loadTensorF32` can surface a wide error set (every GGUF read error
        // plus `NotMultipleOfBlock` and the allocator's).  Fold it into the
        // backend's `LoadError`: missing tensor and OOM keep their names; every
        // other failure (bad magic, unexpected EOF, a row that is not a whole
        // block, ...) means "this tensor is unusable as a 1-D vector", which
        // is reported as unsupported rather than silently truncated.
        const s = loadTensorF32(self.alloc, self.g, name) catch |err| {
            return switch (err) {
                error.MissingTensor => error.MissingTensor,
                error.OutOfMemory => error.OutOfMemory,
                else => error.UnsupportedTensorType,
            };
        };
        return Buf.wrap(s);
    }

    fn vFreeVec(ctx: *anyopaque, v: Buf) void {
        const self: *CpuBackend = @ptrCast(@alignCast(ctx));
        if (v.len == 0) return; // `Buf.empty` sentinel: nothing was allocated.
        self.alloc.free(v.slice());
    }

    /// `pub` so a backend that composes (rather than replaces) the CPU
    /// implementation -- see `backend_cuda.zig`'s trampoline stage -- can hand
    /// out the CPU vtable unchanged.  Every target is a free function that
    /// recovers `*CpuBackend` from `ctx`, so the vtable works for any `ctx`
    /// that really points at a `CpuBackend`.
    pub const vtable = backend.Exec.VTable{
        .loadLinear = vLoadLinear,
        .freeLinear = vFreeLinear,
        .allocScratch = vAllocScratch,
        .allocScratchK = vAllocScratchK,
        .freeScratch = vFreeScratch,
        .dequantRow = vDequantRow,
        .matvec = vMatvec,
        .matvecPair = vMatvecPair,
        .matvecBatchK = vMatvecBatchK,
        .rmsnorm = vRmsnorm,
        .rmsnormRows = vRmsnormRows,
        .rope64 = vRope64,
        .rope64Rows = vRope64Rows,
        .l2norm = vL2norm,
        .l2normRows = vL2normRows,
        .scaleVec = vScaleVec,
        .l2normQkScale = vL2normQkScale,
        .siluInPlace = vSiluInPlace,
        .mulSilu = vMulSilu,
        .mulSigmoid = vMulSigmoid,
        .softmax = vSoftmax,
        .addResid = vAddResid,
        .swiglu = vSwiglu,
        .convStep = vConvStep,
        .gdnStep = vGdnStep,
        .gdnStepRows = vGdnStepRows,
        .gdnStepRowsChunk = vGdnStepRowsChunk,
        .loadVecF32 = vLoadVecF32,
        .freeVec = vFreeVec,
        .allocBuf = vAllocBuf,
        .attnDecode = vAttnDecode,
        .attnChunk = vAttnChunk,
        .gdnGamma = vGdnGamma,
        .gdnPrefillCore = vGdnPrefillCore,
        .copyRange = vCopyRange,
        .copyRangeSlot = vCopyRangeSlot,
        .deinterleaveQG = vDeinterleaveQG,
        .deinterleaveQGStrided = vDeinterleaveQGStrided,
        .copyFromHost = vCopyFromHost,
        .copyToHost = vCopyToHost,
        .zeroBuf = vZeroBuf,
        .argmax = vArgmax,
        .sample = vSample,
        .sampledToken = vSampledToken,
        .setThreads = vSetThreads,
        .profPhase = vProfPhase,
        .profReset = vProfReset,
        .profPrefillEnd = vProfPrefillEnd,
        .stepBegin = vStepBegin,
        .stepEnd = vStepEnd,
        .name = vName,
        .describe = vDescribe,
        .setTier = vSetTier,
        .reportProfile = vReportProfile,
        .profReport = vProfReport,
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
    matvecF32(&out_f32, &wf, &xq, null);

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
    matvecF32(&out, &m, &x, null);
    try std.testing.expectEqual(@as(f32, 50), out[0]);
    try std.testing.expectEqual(@as(f32, 122), out[1]);
}

test "matvecQuant Q4_0: 256-framed 32-block type matches rowDotQ4_0" {
    // Regression: q4_0 reports blockElems == 32 but GGUF frames its rows as
    // 256-element row-blocks, so `blocks_per_row` is cols/256 while the
    // activation is a Q8_0 (34 B / 32 elems) buffer over the whole row.  Two
    // bugs lived here: an assert demanding bpr == cols/32 (8x too large), and
    // dequantizing only the first 256 elements of a wider row.  Q8_0 alone
    // covers neither, because q8_0 really does have 32-element row-blocks.
    const cols = 256; // exactly one 256-element row-block
    const rows = 1;

    var wq: [rows * 8 * 18]u8 = undefined;
    var x: [cols]f32 = undefined;
    for (&x, 0..) |*v, i| v.* = @as(f32, @floatFromInt(i % 7)) * 0.25 - 0.75;
    for (0..8) |b| {
        // f16 d, then 16 bytes of packed nibbles; vary per block so a wrong
        // stride or activation window cannot silently match.
        const bits: u16 = @bitCast(@as(f16, 0.0625));
        wq[b * 18 + 0] = @truncate(bits);
        wq[b * 18 + 1] = @truncate(bits >> 8);
        for (0..16) |j| wq[b * 18 + 2 + j] = @truncate(j *% 37 +% b *% 11 +% 5);
    }

    const w: QuantRow = .{ .data = &wq, .ttype = .q4_0, .rows = rows, .blocks_per_row = cols / 256 };
    try std.testing.expectEqual(cols, w.cols());

    var out: [rows]f32 = undefined;
    var q8scratch: [q8ScratchBytes(cols)]u8 = undefined;
    var q8kscratch: [q8kScratchBytes(cols)]u8 = undefined;
    matvecQuant(&out, w, &x, &q8kscratch, &q8scratch, null);

    // The kernel quantizes x to Q8_0; the scalar reference does not, so the
    // two differ by the activation's quantization noise (measured ~0.2% here).
    const ref = quant.rowDotQ4_0(&wq, &x);
    try std.testing.expectApproxEqAbs(ref, out[0], @abs(ref) * 5e-3 + 5e-3);
}

test "dequantRowInto q4_0: every 256-element row-block of a wide row is written" {
    // Regression: `dequantRowFn` writes exactly 256 floats (it loops 0..8 over
    // 8 x 32-element sub-blocks).  Handing a whole 512-element row to one call
    // leaves the second half of the destination untouched -- silent garbage in
    // token embeddings, and on a real 1024-wide model row three quarters of
    // every embedding row stays zero.  Q8_0 has its own path, so a Q8_0-only
    // suite cannot see this.
    const cols = 512; // two 256-element row-blocks
    var wq: [16 * 18]u8 = undefined; // 16 x 32-element blocks
    for (0..16) |b| {
        const bits: u16 = @bitCast(@as(f16, 0.0625));
        wq[b * 18 + 0] = @truncate(bits);
        wq[b * 18 + 1] = @truncate(bits >> 8);
        for (0..16) |j| wq[b * 18 + 2 + j] = @truncate(j *% 13 +% b *% 7 +% 3);
    }

    const w: QuantRow = .{ .data = &wq, .ttype = .q4_0, .rows = 1, .blocks_per_row = cols / 256 };
    try std.testing.expectEqual(cols, w.cols());

    var out: [cols]f32 = @splat(0);
    dequantRowInto(&out, w, 0);

    // The second row-block must be written, not left at the 0 sentinel.
    try std.testing.expect(out[cols - 1] != 0);
    // ...and it must equal a direct 256-element dequant of those same bytes.
    var direct: [256]f32 = undefined;
    dequantRowFnFor(.q4_0).?(wq[8 * 18 ..], &direct);
    for (0..256) |i| try std.testing.expectEqual(direct[i], out[256 + i]);
}

test "gdnPrefillCore equals the per-token primitive sequence" {
    // The whole point of the prefill prelude is that it is CODE MOTION: the same
    // calls, made once for the chunk instead of once per token.  So the oracle
    // for it is literally the per-token loop, and this test pins that equality.
    //
    // Shapes are the 27B GDN geometry shrunk to something a test can hold:
    // nv = 6 value heads, nk = 2 k heads (so nk_reps = 3 -- the repeat count is
    // what makes l2norm non-idempotent, so it MUST be > 1 here), kd = 4.
    // conv_k is pinned at 4, the only width `ew.convStep` implements.
    const nv: usize = 6;
    const nk: usize = 2;
    const kd: usize = 4;
    const tokens: usize = 5;
    const conv_k: usize = 4;
    const kq_dim = nk * kd;
    const conv_dim = 2 * kq_dim + nv * kd;
    const ab_stride = nv;
    const ab_a_offset: usize = 0;
    const ab_b_offset = tokens * ab_stride;

    var prng = std.Random.DefaultPrng.init(7);
    const rand = prng.random();

    // qkv, the conv history (one token long: conv_k - 1 + 1), weights and the
    // a/b projections.  Two independent copies so the batched run cannot see
    // what the serial run wrote.
    var qkv_batch: [tokens * conv_dim]f32 = undefined;
    var qkv_serial: [tokens * conv_dim]f32 = undefined;
    for (&qkv_batch, 0..) |*v, i| {
        v.* = rand.float(f32) * 2.0 - 1.0;
        qkv_serial[i] = v.*;
    }
    // The conv history is `conv_k - 1` slots per channel, so 3 * conv_dim floats.
    var conv_batch: [3 * conv_dim]f32 = undefined;
    var conv_serial: [3 * conv_dim]f32 = undefined;
    for (&conv_batch, 0..) |*v, i| {
        v.* = rand.float(f32) * 2.0 - 1.0;
        conv_serial[i] = v.*;
    }
    var wconv: [conv_dim * conv_k]f32 = undefined;
    for (&wconv) |*v| v.* = rand.float(f32) * 2.0 - 1.0;
    var ab_batch: [2 * tokens * ab_stride]f32 = undefined;
    var ab_serial: [2 * tokens * ab_stride]f32 = undefined;
    for (&ab_batch, 0..) |*v, i| {
        v.* = rand.float(f32) * 2.0 - 1.0;
        ab_serial[i] = v.*;
    }
    var ssm_a: [nv]f32 = undefined;
    var dt_bias: [nv]f32 = undefined;
    for (&ssm_a) |*v| v.* = -(rand.float(f32) * 2.0 + 0.1);
    for (&dt_bias) |*v| v.* = rand.float(f32) * 3.0 - 1.0;
    var gamma_batch: [tokens * ab_stride]f32 = @splat(std.math.nan(f32));
    var gamma_serial: [tokens * ab_stride]f32 = @splat(std.math.nan(f32));

    const ctx: *anyopaque = undefined;
    CpuBackend.vGdnPrefillCore(ctx, Buf.wrap(&qkv_batch), Buf.wrap(&conv_batch), Buf.wrap(&wconv),
        Buf.wrap(&ab_batch), Buf.wrap(&ssm_a), Buf.wrap(&dt_bias),
        Buf.wrap(&gamma_batch), tokens, conv_dim, ab_a_offset, ab_b_offset,
        ab_stride, ab_stride, kq_dim, nv, nk, kd, conv_k, 1e-6);

    // ---- reference: exactly what `forwardGdnCore` used to do, per token ----
    const nk_reps = nv / nk;
    const qscale = 1.0 / @sqrt(@as(f32, @floatFromInt(kd)));
    for (0..tokens) |t| {
        const qkv_t = qkv_serial[t * conv_dim ..][0..conv_dim];
        ew.bufConvStep(Buf.wrap(&conv_serial), Buf.wrap(qkv_t), Buf.wrap(&wconv), conv_k);
        ew.bufGdnGamma(
            Buf.wrap(ab_serial[ab_a_offset + t * ab_stride ..][0..nv]),
            Buf.wrap(ab_serial[ab_b_offset + t * ab_stride ..][0..nv]),
            Buf.wrap(&ssm_a),
            Buf.wrap(&dt_bias),
            Buf.wrap(gamma_serial[t * ab_stride ..][0..nv]),
            nv,
        );
        for (0..nk_reps) |_| {
            ew.bufL2norm(Buf.wrap(qkv_t[0..kq_dim]));
            ew.bufL2norm(Buf.wrap(qkv_t[kq_dim..][0..kq_dim]));
            ew.bufScaleVec(Buf.wrap(qkv_t[0..kq_dim]), qscale);
        }
    }

    // The batched path must be bit-identical here: every underlying primitive is
    // per-row independent and the conv walks the tokens in the same ascending
    // order, so there is no reassociation to tolerate.
    try std.testing.expectEqualSlices(f32, &qkv_serial, &qkv_batch);
    try std.testing.expectEqualSlices(f32, &conv_serial, &conv_batch);
    try std.testing.expectEqualSlices(f32, &gamma_serial, &gamma_batch);
    try std.testing.expectEqualSlices(f32, &ab_serial, &ab_batch);
}
