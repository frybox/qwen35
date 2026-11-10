//! Parity tests for the chunk-wide vtable slots the CUDA prefill path added:
//! `attnChunk` and `deinterleaveQGStrided`.
//!
//! WHY THESE EXIST.  Both slots exist purely so a prefill chunk can make ONE
//! call where the engine used to make one per token (8400 -> 16 attention
//! launches, 940 -> 16 deinterleaves on a 525-token 27B run).  The risk of a
//! merge like that is not "the arithmetic is hard", it is "row t got pointed at
//! row t-1's buffer" -- an off-by-one that still produces plausible numbers.
//! So the contract checked here is exactly the merge:
//!
//!   attnChunk(slot0, k)            == attnDecode(slot0+r) for r = 0..k-1
//!   deinterleaveQGStrided(k)       == deinterleaveQG for t = 0..k-1
//!
//! The CPU backend implements both slots as literally those loops (see
//! `vAttnChunk` / `vDeinterleaveQGStrided` in backend_cpu.zig), which is what
//! makes this a meaningful oracle rather than a copy checked against a copy.
//! The CPU path is also the one that runs in `zig build test` (no -Dcuda), so
//! this is where the CACHE-ROW ARITHMETIC of the two forms is pinned down; the
//! CUDA kernel itself was gated separately against the same per-row form in
//! `/tmp/gate_attn_chunk.cu` (23 rows / 4 heads / slot0=37, bit-identical).

const std = @import("std");
const backend = @import("backend.zig");
const backend_cpu = @import("backend_cpu.zig");

/// Deterministic pseudo-random fill.  Magnitudes are kept O(1) so the softmax
/// does not saturate to one-hot and hide a wrong score.
fn fill(buf: []f32, seed: u64) void {
    var rng = std.Random.DefaultPrng.init(seed);
    const r = rng.random();
    for (buf) |*x| x.* = r.float(f32) * 2.0 - 1.0;
}

fn maxAbsDiff(a: []const f32, b: []const f32) f32 {
    var worst: f32 = 0;
    for (a, b) |x, y| worst = @max(worst, @abs(x - y));
    return worst;
}

test "attnChunk: one call for k rows == k per-row attnDecode calls" {
    const heads = 4; // 2 kv heads, gqa=2: exercises the h/gqa head map
    const gqa = 2;
    const hd = 32;
    const kv_dim = (heads / gqa) * hd;
    const slot0 = 37; // non-zero, non-multiple-of-anything on purpose
    const rows = 23; // ditto: a prefix length that is never rounded
    const row_len = heads * hd;
    const n_tot = slot0 + rows;

    var ctx_holder: u8 = 0;
    const alloc = std.testing.allocator;
    const q = try alloc.alloc(f32, rows * row_len);
    defer alloc.free(q);
    const gate = try alloc.alloc(f32, rows * row_len);
    defer alloc.free(gate);
    const kc = try alloc.alloc(f32, n_tot * kv_dim);
    defer alloc.free(kc);
    const vc = try alloc.alloc(f32, n_tot * kv_dim);
    defer alloc.free(vc);
    const att = try alloc.alloc(f32, n_tot);
    defer alloc.free(att);
    const got = try alloc.alloc(f32, rows * row_len);
    defer alloc.free(got);
    const want = try alloc.alloc(f32, rows * row_len);
    defer alloc.free(want);

    fill(q, 1);
    fill(gate, 2);
    fill(kc, 3);
    fill(vc, 4);

    // The CPU vtable directly: `CpuBackend.init` needs a `gguf.File`, and every
    // CPU slot ignores `ctx`, so this is the same code an `Exec` would reach.
    const vt = backend_cpu.CpuBackend.vtable;
    const ctx: *anyopaque = @ptrCast(&ctx_holder);
    const scale: f32 = 1.0 / @sqrt(@as(f32, @floatFromInt(hd)));

    // Chunk form: everything in ONE call.
    vt.attnChunk(ctx, .{ .ptr = q.ptr, .len = q.len }, .{ .ptr = gate.ptr, .len = gate.len },
        .{ .ptr = kc.ptr, .len = kc.len }, .{ .ptr = vc.ptr, .len = vc.len },
        .{ .ptr = att.ptr, .len = att.len }, .{ .ptr = got.ptr, .len = got.len },
        slot0, rows, heads, gqa, hd, kv_dim, scale);

    // Row form: the calls the engine made before the slot existed.
    for (0..rows) |r| {
        const o = r * row_len;
        vt.attnDecode(ctx, .{ .ptr = q.ptr + o, .len = row_len }, .{ .ptr = gate.ptr + o, .len = row_len },
            .{ .ptr = kc.ptr, .len = kc.len }, .{ .ptr = vc.ptr, .len = vc.len },
            .{ .ptr = att.ptr, .len = att.len }, .{ .ptr = want.ptr + o, .len = row_len },
            slot0 + r, heads, gqa, hd, kv_dim, scale);
    }

    // The CPU slots are the same code path, so this is expected to be exact.  A
    // non-zero result here means the ROW ADDRESSING of one form is wrong, not
    // that f32 drifted -- and an address error shows up as a large diff, which
    // the relative bound below is chosen to catch.
    try std.testing.expectEqual(@as(f32, 0), maxAbsDiff(got, want));

    // Guard against the vacuous pass: if both forms wrote nothing (a no-op
    // default implementation), they would also agree.  Check the rows actually
    // carry attention output.
    var nonzero: usize = 0;
    for (got) |x| {
        if (x != 0) nonzero += 1;
    }
    try std.testing.expect(nonzero * 4 > got.len);
}

test "attnChunk: row r sees a prefix that grows with r" {
    // The causal mask is the one thing the merge could plausibly get wrong while
    // still producing finite output: if every row used the last row's window,
    // row 0's result would equal the decode call at slot0+rows-1 instead of
    // slot0.  Perturbing a cache row BEYOND row r's window must not move row r.
    const heads = 2;
    const gqa = 1;
    const hd = 16;
    const kv_dim = (heads / gqa) * hd;
    const slot0 = 8;
    const rows = 5;
    const row_len = heads * hd;
    const n_tot = slot0 + rows;

    var ctx_holder: u8 = 0;
    const alloc = std.testing.allocator;
    const q = try alloc.alloc(f32, rows * row_len);
    defer alloc.free(q);
    const gate = try alloc.alloc(f32, rows * row_len);
    defer alloc.free(gate);
    const kc = try alloc.alloc(f32, n_tot * kv_dim);
    defer alloc.free(kc);
    const vc = try alloc.alloc(f32, n_tot * kv_dim);
    defer alloc.free(vc);
    const att = try alloc.alloc(f32, n_tot);
    defer alloc.free(att);
    const a = try alloc.alloc(f32, rows * row_len);
    defer alloc.free(a);
    const b = try alloc.alloc(f32, rows * row_len);
    defer alloc.free(b);

    fill(q, 11);
    fill(gate, 12);
    fill(kc, 13);
    fill(vc, 14);

    // The CPU vtable directly: `CpuBackend.init` needs a `gguf.File`, and every
    // CPU slot ignores `ctx`, so this is the same code an `Exec` would reach.
    const vt = backend_cpu.CpuBackend.vtable;
    const ctx: *anyopaque = @ptrCast(&ctx_holder);
    const scale: f32 = 1.0 / @sqrt(@as(f32, @floatFromInt(hd)));

    vt.attnChunk(ctx, .{ .ptr = q.ptr, .len = q.len }, .{ .ptr = gate.ptr, .len = gate.len },
        .{ .ptr = kc.ptr, .len = kc.len }, .{ .ptr = vc.ptr, .len = vc.len },
        .{ .ptr = att.ptr, .len = att.len }, .{ .ptr = a.ptr, .len = a.len },
        slot0, rows, heads, gqa, hd, kv_dim, scale);

    // Rewrite the LAST cache row (index n_tot-1), which only row `rows-1` may see.
    for (0..kv_dim) |i| {
        kc[(n_tot - 1) * kv_dim + i] += 3.0;
        vc[(n_tot - 1) * kv_dim + i] += 3.0;
    }

    vt.attnChunk(ctx, .{ .ptr = q.ptr, .len = q.len }, .{ .ptr = gate.ptr, .len = gate.len },
        .{ .ptr = kc.ptr, .len = kc.len }, .{ .ptr = vc.ptr, .len = vc.len },
        .{ .ptr = att.ptr, .len = att.len }, .{ .ptr = b.ptr, .len = b.len },
        slot0, rows, heads, gqa, hd, kv_dim, scale);

    // Rows 0..rows-2 must be untouched by a change past their window.
    try std.testing.expectEqual(@as(f32, 0), maxAbsDiff(a[0..(rows - 1) * row_len], b[0..(rows - 1) * row_len]));
    // The last row must have moved (otherwise the perturbation did nothing and
    // the check above is vacuous).
    const moved = maxAbsDiff(a[(rows - 1) * row_len ..], b[(rows - 1) * row_len ..]);
    try std.testing.expect(moved > 1e-3);
}

test "deinterleaveQGStrided: one call for k rows == k per-row calls" {
    const heads = 6;
    const hd = 5; // not a power of two: catches a stride taken as hd*hd
    const rows = 7;
    const row_len = heads * hd;

    var ctx_holder: u8 = 0;
    const alloc = std.testing.allocator;
    const src = try alloc.alloc(f32, rows * 2 * row_len);
    defer alloc.free(src);
    const q = try alloc.alloc(f32, rows * row_len);
    defer alloc.free(q);
    const gate = try alloc.alloc(f32, rows * row_len);
    defer alloc.free(gate);
    const q2 = try alloc.alloc(f32, rows * row_len);
    defer alloc.free(q2);
    const gate2 = try alloc.alloc(f32, rows * row_len);
    defer alloc.free(gate2);

    fill(src, 21);
    @memset(q, 0);
    @memset(gate, 0);
    @memset(q2, 0);
    @memset(gate2, 0);

    const vt = backend_cpu.CpuBackend.vtable;
    const ctx: *anyopaque = @ptrCast(&ctx_holder);
    vt.deinterleaveQGStrided(ctx, .{ .ptr = q.ptr, .len = q.len }, .{ .ptr = gate.ptr, .len = gate.len },
        .{ .ptr = src.ptr, .len = src.len }, heads, hd, rows);

    for (0..rows) |t| {
        const o = t * row_len;
        vt.deinterleaveQG(ctx, .{ .ptr = q2.ptr + o, .len = row_len },
            .{ .ptr = gate2.ptr + o, .len = row_len },
            .{ .ptr = src.ptr + t * 2 * row_len, .len = 2 * row_len }, heads, hd);
    }

    // A pure move: exact by construction, so demand exact.
    try std.testing.expectEqualSlices(f32, q2, q);
    try std.testing.expectEqualSlices(f32, gate2, gate);

    // And the split must actually be a split: q takes the first `hd` of each
    // interleaved pair, gate the second.
    for (0..rows) |t| {
        for (0..heads) |h| {
            const s = t * 2 * row_len + h * 2 * hd;
            const d = t * row_len + h * hd;
            try std.testing.expectEqualSlices(f32, src[s..][0..hd], q[d..][0..hd]);
            try std.testing.expectEqualSlices(f32, src[s + hd ..][0..hd], gate[d..][0..hd]);
        }
    }
}
