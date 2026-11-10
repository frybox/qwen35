//! `engine.refErrF64` -- the f64 golden reference -- against an independent
//! decode of the same packed bytes.
//!
//! Why this file exists: `refErrF64` is the oracle every matvec result is
//! checked against (AGENTS.md §4, kernel-level parity), so a bug in *it* is
//! worse than a bug in a kernel -- it turns a correct kernel into a reported
//! failure and a wrong one into a pass.  Nothing ever noticed, because nothing
//! called it: Zig analyses unreferenced functions lazily, so its Q8_0 branch --
//! hand-inlined, with a `q.row_blocks()` typo for a method that is named
//! `rowBlocks` -- did not even compile until this file referenced it.  The
//! oracle for Q8_0 is now the same row-block walk as every other packed type.
//!
//! Layering: `engine.zig` must not import the quant module (AGENTS.md §1.1), so
//! the test lives here and imports both sides.

const std = @import("std");
const engine = @import("engine.zig");
const backend_cpu = @import("backend_cpu.zig");
const gguf = @import("gguf.zig");
const quant = @import("quant_dequant.zig");

const Q8_BLOCK_BYTES = 34; // f16 d + 32 x i8
const Q8_BLOCK_ELEMS = 32;

/// Independent Q8_0 row decode in f64 -- deliberately NOT `dequantizeQ8_0`, and
/// deliberately not the row-block walk inside `refErrF64`.  Written straight
/// from ggml's `block_q8_0` layout so the comparison is two derivations, not a
/// copy against itself.
fn refDotQ8_0F64(row: []const u8, x: []const f32) f64 {
    var acc: f64 = 0;
    var off: usize = 0;
    while (off < row.len) : (off += Q8_BLOCK_BYTES) {
        const d: f64 = @floatCast(@as(f16, @bitCast(std.mem.readInt(u16, row[off..][0..2], .little))));
        for (0..Q8_BLOCK_ELEMS) |j| {
            const w: f64 = d * @as(f64, @as(i8, @bitCast(row[off + 2 + j])));
            acc += w * @as(f64, x[off / Q8_BLOCK_BYTES * Q8_BLOCK_ELEMS + j]);
        }
    }
    return acc;
}

/// Populate a packed Q8_0 tensor covering `wq.len` bytes (whole 34-byte blocks).
fn fillQ8_0(wq: []u8, seed: u64) void {
    var prng = std.Random.DefaultPrng.init(seed);
    const rand = prng.random();
    var blk: usize = 0;
    while (blk < wq.len / Q8_BLOCK_BYTES) : (blk += 1) {
        var f32blk: [Q8_BLOCK_ELEMS]f32 = undefined;
        for (&f32blk) |*v| v.* = rand.float(f32) * 2.0 - 1.0;
        quant.quantizeQ8_0(&f32blk, wq[blk * Q8_BLOCK_BYTES ..][0..Q8_BLOCK_BYTES]) catch unreachable;
    }
}

test "refErrF64 q8_0: oracle matches an independent f64 row decode" {
    // 1024 is a real Q8_0 hidden width, and it is wider than the 256-element
    // row-block framing the OTHER types use -- the case where a type with its own
    // block unit can be mis-framed without any existing test noticing.
    const cols = 1024;
    const rows = 4;
    var wq: [rows * cols / Q8_BLOCK_ELEMS * Q8_BLOCK_BYTES]u8 = undefined;
    fillQ8_0(&wq, 7);
    var x: [cols]f32 = undefined;
    for (&x, 0..) |*v, i| v.* = @as(f32, @floatFromInt(@mod(i *% 37, 101))) * 0.02 - 1.0;

    // `blocks_per_row` is counted in ROW-blocks (32 for Q8_0, see
    // `gguf.rowBlockElems`), so it is cols/32 here -- the same number as this
    // type's own blocks per row, which is exactly why Q8_0 hides framing bugs.
    const w: gguf.QuantRow = .{ .data = &wq, .ttype = .q8_0, .rows = rows, .blocks_per_row = cols / 32 };
    try std.testing.expectEqual(cols, w.cols());

    // Feed the oracle a result computed from its own reference values, in f64:
    // the relative error must then be f32-rounding-level, not "1e-2 tolerance"
    // level, otherwise the test cannot tell a right reference from a shrug.
    var out: [rows]f32 = undefined;
    for (0..rows) |r| {
        const ref = refDotQ8_0F64(wq[r * w.rowBytes() ..][0..w.rowBytes()], &x);
        out[r] = @floatCast(ref);
    }

    const err = engine.refErrF64(.{ .quant = w }, &out, .{ .x = &x, .k = 1 }, cols);
    try std.testing.expect(err >= 0); // >= 0: reference computable for this ttype
    try std.testing.expectApproxEqAbs(@as(f64, 0), err, 1e-6);
}

test "refErrF64 q8_0: a row-block framing mistake is caught, not shrugged" {
    // What the merge buys: one row-block rule for every packed type.  So the
    // regression to guard against is a *sizing* mistake -- a reference built with
    // the wrong elements-per-block constant -- and it has to score badly rather
    // than plausibly.  1024 wide, i.e. 32 row-blocks of 32 elements: with
    // cols == 256 the Q8_0 block unit and a 256-element framing coincide, which
    // is exactly how the existing Q8_0-only tests stay blind to this class.
    const cols = 1024;
    const rows = 2;
    var wq: [rows * cols / Q8_BLOCK_ELEMS * Q8_BLOCK_BYTES]u8 = undefined;
    fillQ8_0(&wq, 11);
    var x: [cols]f32 = undefined;
    for (&x, 0..) |*v, i| v.* = @as(f32, @floatFromInt(i % 13)) * 0.1 - 0.6;

    const w: gguf.QuantRow = .{ .data = &wq, .ttype = .q8_0, .rows = rows, .blocks_per_row = cols / 32 };
    var out: [rows]f32 = undefined;
    for (0..rows) |r| {
        out[r] = @floatCast(refDotQ8_0F64(wq[r * w.rowBytes() ..][0..w.rowBytes()], &x));
    }
    const clean = engine.refErrF64(.{ .quant = w }, &out, .{ .x = &x, .k = 1 }, cols);
    try std.testing.expectApproxEqAbs(@as(f64, 0), clean, 1e-6);

    // A 256-element window on a row that really carries cols/32 32-element
    // blocks: only the first quarter of every row is summed.  Not zero error and
    // not small -- so an oracle that mis-sizes its row-block cannot pass as
    // "close enough".
    var quarter: [rows]f32 = undefined;
    for (0..rows) |r| {
        const row = wq[r * w.rowBytes() ..][0..w.rowBytes()];
        var acc: f64 = 0;
        for (0..256) |c| {
            const blk = c / Q8_BLOCK_ELEMS;
            const d: f64 = @floatCast(@as(f16, @bitCast(std.mem.readInt(u16, row[blk * Q8_BLOCK_BYTES ..][0..2], .little))));
            acc += d * @as(f64, @as(i8, @bitCast(row[blk * Q8_BLOCK_BYTES + 2 + (c % Q8_BLOCK_ELEMS)]))) * @as(f64, x[c]);
        }
        quarter[r] = @floatCast(acc);
    }
    const dirty = engine.refErrF64(.{ .quant = w }, &quarter, .{ .x = &x, .k = 1 }, cols);
    try std.testing.expect(dirty > 1e-2);
}

test "refErrF64 q8_0: agrees with dequantRowInto (same framing, two paths)" {
    // The oracle and the CPU's row expander must not disagree about a Q8_0 row,
    // because `refErrF64` documents itself as using the `dequantRowInto` framing.
    // A matvec is then judged against the same bytes both of them read.
    const cols = 512; // 16 x 32-element row-blocks
    var wq: [cols / Q8_BLOCK_ELEMS * Q8_BLOCK_BYTES]u8 = undefined;
    fillQ8_0(&wq, 3);
    const w: gguf.QuantRow = .{ .data = &wq, .ttype = .q8_0, .rows = 1, .blocks_per_row = cols / 32 };

    var wf: [cols]f32 = @splat(0);
    backend_cpu.dequantRowInto(&wf, w, 0);
    var acc: f64 = 0;
    var x: [cols]f32 = undefined;
    for (&x, 0..) |*v, i| v.* = @as(f32, @floatFromInt(i)) / @as(f32, @floatFromInt(cols));
    for (0..cols) |c| acc += @as(f64, wf[c]) * @as(f64, x[c]);

    const out = [_]f32{@floatCast(acc)};
    const err = engine.refErrF64(.{ .quant = w }, &out, .{ .x = &x, .k = 1 }, cols);
    try std.testing.expect(err >= 0);
    try std.testing.expectApproxEqAbs(@as(f64, 0), err, 1e-6);
}

test "refErrF64 q4_0: 256-framed row-block walk is unchanged by the merge" {
    // The generic branch was already row-block based; pin it so the Q8_0 merge
    // cannot be blamed for (or silently "fix") a framing change there.
    const cols = 512; // two 256-element row-blocks
    var wq: [16 * 18]u8 = undefined; // 16 x 32-element blocks
    for (0..16) |b| {
        const bits: u16 = @bitCast(@as(f16, 0.0625));
        wq[b * 18 + 0] = @truncate(bits);
        wq[b * 18 + 1] = @truncate(bits >> 8);
        for (0..16) |j| wq[b * 18 + 2 + j] = @truncate(j *% 29 +% b *% 17 +% 5);
    }
    const w: gguf.QuantRow = .{ .data = &wq, .ttype = .q4_0, .rows = 1, .blocks_per_row = cols / 256 };
    try std.testing.expectEqual(cols, w.cols());

    var wf: [cols]f32 = @splat(0);
    backend_cpu.dequantRowInto(&wf, w, 0);
    try std.testing.expect(wf[cols - 1] != 0);

    var x: [cols]f32 = undefined;
    for (&x, 0..) |*v, i| v.* = @as(f32, @floatFromInt(i % 9)) * 0.3 - 1.2;
    var acc: f64 = 0;
    for (0..cols) |c| acc += @as(f64, wf[c]) * @as(f64, x[c]);

    const out = [_]f32{@floatCast(acc)};
    const err = engine.refErrF64(.{ .quant = w }, &out, .{ .x = &x, .k = 1 }, cols);
    try std.testing.expect(err >= 0);
    try std.testing.expectApproxEqAbs(@as(f64, 0), err, 1e-6);
}

test "dequantRowFnFor q8_0: row-block adapter equals the whole-row dequant" {
    // The adapter the merge relies on: one call per 32-element row-block must
    // reproduce `dequantizeQ8_0` over the whole row, byte for byte.
    const cols = 256;
    var wq: [cols / Q8_BLOCK_ELEMS * Q8_BLOCK_BYTES]u8 = undefined;
    fillQ8_0(&wq, 5);

    var whole: [cols]f32 = undefined;
    quant.dequantizeQ8_0(&wq, &whole) catch unreachable;

    const deq = backend_cpu.dequantRowFnFor(.q8_0) orelse return error.MissingRowFn;
    var step: [cols]f32 = undefined;
    for (0..cols / Q8_BLOCK_ELEMS) |b| {
        deq(wq[b * Q8_BLOCK_BYTES ..][0..Q8_BLOCK_BYTES], step[b * Q8_BLOCK_ELEMS ..][0..Q8_BLOCK_ELEMS]);
    }
    try std.testing.expectEqualSlices(f32, &whole, &step);

    // And 256-wide in one call, which is what a 256-framed caller would ask.
    var wide: [cols]f32 = undefined;
    deq(&wq, &wide);
    try std.testing.expectEqualSlices(f32, &whole, &wide);
}
