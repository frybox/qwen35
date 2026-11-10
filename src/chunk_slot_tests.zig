//! Parity tests for the chunk-wide vtable slots the CUDA prefill path added:
//! `attnChunk`, `deinterleaveQGStrided` and `gdnStepRowsChunk`.
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

// ---- gdnStepRowsChunk ----------------------------------------------------
//
// The GDN state update is a RECURRENCE (`state` is carried from row to row), so
// unlike attnChunk the merge here cannot be checked by comparing independent
// rows -- the whole point is that row t sees row t-1's state.  What has to hold
// is therefore:
//
//   gdnStepRowsChunk(rows)  ==  gdnStepRows(t = 0), gdnStepRows(t = 1), ...
//
// i.e. the fused call must produce exactly the state and outputs the sequential
// calls produce, INCLUDING the serialisation.  That is what the CPU oracle does
// (it literally calls `vGdnStepRows` in a loop), and it is why a device kernel
// that ran the rows in parallel would fail this test rather than pass it with
// small differences.
//
// Exact equality is demandable: the two forms execute the same statements in the
// same order, so any difference is a merge bug (wrong stride, off-by-one row, or
// a state block shared between heads).

test "gdnStepRowsChunk: one fused call == rows successive gdnStepRows calls" {
    const heads = 4;
    const nk = 2; // nk < heads: exercises the kh = h % nk kv-head map
    // kd/vd are 16, not 3/5: the CPU oracle (`elementwise.gdnKvMem`) vectorises
    // over 16 f32 lanes and asserts `vd % 16 == 0`.  A non-multiple width would
    // test the assert, not the merge.
    const kd = 16;
    const vd = 16;
    const rows = 7;
    const kq_dim = nk * kd;
    const value_dim = heads * vd;
    // Deliberately NOT kq_dim: the qkv rows interleave q/k/gate, so the
    // per-token stride is independent of the slice width.  Getting this wrong
    // (using kq_dim) was the first device-side bug.
    const conv_dim = 2 * kq_dim + value_dim;

    const alloc = std.testing.allocator;
    const state_a = try alloc.alloc(f32, heads * kd * vd);
    defer alloc.free(state_a);
    const state_b = try alloc.alloc(f32, heads * kd * vd);
    defer alloc.free(state_b);
    const qkv = try alloc.alloc(f32, rows * conv_dim);
    defer alloc.free(qkv);
    const km_a = try alloc.alloc(f32, rows * value_dim);
    defer alloc.free(km_a);
    const km_b = try alloc.alloc(f32, rows * value_dim);
    defer alloc.free(km_b);
    const dl_a = try alloc.alloc(f32, rows * value_dim);
    defer alloc.free(dl_a);
    const dl_b = try alloc.alloc(f32, rows * value_dim);
    defer alloc.free(dl_b);
    const out_a = try alloc.alloc(f32, rows * value_dim);
    defer alloc.free(out_a);
    const out_b = try alloc.alloc(f32, rows * value_dim);
    defer alloc.free(out_b);
    const gamma = try alloc.alloc(f32, rows * heads);
    defer alloc.free(gamma);
    const beta = try alloc.alloc(f32, rows * heads);
    defer alloc.free(beta);

    var rng = std.Random.DefaultPrng.init(4242);
    const r = rng.random();
    for (qkv) |*x| x.* = r.float(f32) * 2.0 - 1.0;
    // decay factors strictly inside (0, 1): outside that range the recurrence
    // explodes and a wrong order could hide inside inf/nan
    for (gamma) |*x| x.* = 0.5 + 0.4 * r.float(f32);
    for (beta) |*x| x.* = r.float(f32);

    for (state_a) |*x| x.* = r.float(f32) * 2.0 - 1.0;
    @memcpy(state_b, state_a);
    @memset(km_a, 0);
    @memset(km_b, 0);
    @memset(dl_a, 0);
    @memset(dl_b, 0);
    @memset(out_a, 0);
    @memset(out_b, 0);

    var ctx_holder: u8 = 0;
    const vt = backend_cpu.CpuBackend.vtable;
    const ctx: *anyopaque = @ptrCast(&ctx_holder);

    // (a) the fused chunk call
    vt.gdnStepRowsChunk(ctx,
        .{ .ptr = state_a.ptr, .len = state_a.len },
        .{ .ptr = qkv.ptr + kq_dim, .len = rows * conv_dim - kq_dim },
        .{ .ptr = qkv.ptr + 2 * kq_dim, .len = rows * conv_dim - 2 * kq_dim },
        .{ .ptr = qkv.ptr, .len = rows * conv_dim },
        .{ .ptr = km_a.ptr, .len = km_a.len },
        .{ .ptr = dl_a.ptr, .len = dl_a.len },
        .{ .ptr = out_a.ptr, .len = out_a.len },
        .{ .ptr = gamma.ptr, .len = gamma.len },
        .{ .ptr = beta.ptr, .len = beta.len },
        heads, nk, kd, vd, rows, conv_dim, conv_dim, conv_dim);

    // (b) the sequential form the engine used before the slot existed
    for (0..rows) |t| {
        const qkv_t = qkv[t * conv_dim ..][0..conv_dim];
        vt.gdnStepRows(ctx,
            .{ .ptr = state_b.ptr, .len = state_b.len },
            .{ .ptr = qkv_t.ptr + kq_dim, .len = kq_dim },
            .{ .ptr = qkv_t.ptr + 2 * kq_dim, .len = value_dim },
            .{ .ptr = qkv_t.ptr, .len = kq_dim },
            .{ .ptr = km_b.ptr + t * value_dim, .len = value_dim },
            .{ .ptr = dl_b.ptr + t * value_dim, .len = value_dim },
            .{ .ptr = out_b.ptr + t * value_dim, .len = value_dim },
            .{ .ptr = gamma.ptr + t * heads, .len = heads },
            .{ .ptr = beta.ptr + t * heads, .len = heads },
            heads, nk, kd, vd);
    }

    // Not "close": identical statements in identical order, so identical bits.
    try std.testing.expectEqualSlices(f32, state_b, state_a);
    try std.testing.expectEqualSlices(f32, out_b, out_a);
    try std.testing.expectEqualSlices(f32, km_b, km_a);
    try std.testing.expectEqualSlices(f32, dl_b, dl_a);
}

test "gdnStepRowsChunk: rows are serialised, not independent" {
    // A fused kernel that ran the rows in parallel would still match the
    // sequential form on `state_a` above if it only touched the FIRST row's
    // slice, so pin the recurrence down directly: row 1's output must depend on
    // row 0's state update.
    //
    // Two runs differing ONLY in row 0's k must produce different row 1
    // outputs, and the k of row 1 itself must not be what moved.
    const heads = 1;
    const nk = 1;
    const kd = 16; // same 16-lane constraint as the test above
    const vd = 16;
    const rows = 2;
    const conv_dim = 2 * nk * kd + heads * vd;

    const alloc = std.testing.allocator;
    const qkv = try alloc.alloc(f32, rows * conv_dim);
    defer alloc.free(qkv);
    const gamma = try alloc.alloc(f32, rows * heads);
    defer alloc.free(gamma);
    const beta = try alloc.alloc(f32, rows * heads);
    defer alloc.free(beta);
    const out = [2][]f32{ try alloc.alloc(f32, rows * heads * vd), try alloc.alloc(f32, rows * heads * vd) };
    defer alloc.free(out[0]);
    defer alloc.free(out[1]);

    for (gamma) |*x| x.* = 0.9;
    for (beta) |*x| x.* = 0.5;

    var ctx_holder: u8 = 0;
    const vt = backend_cpu.CpuBackend.vtable;
    const ctx: *anyopaque = @ptrCast(&ctx_holder);

    var run: usize = 0;
    while (run < 2) : (run += 1) {
        var rng = std.Random.DefaultPrng.init(99);
        const r = rng.random();
        for (qkv) |*x| x.* = r.float(f32) * 2.0 - 1.0;
        // Row 0's k[0]: k starts at offset nk*kd inside the qkv row, so
        // index 0 is q — perturbing it would change row 0's output directly
        // and prove nothing about the recurrence.
        if (run == 1) qkv[nk * kd] += 0.5;

        const state = try alloc.alloc(f32, heads * kd * vd);
        defer alloc.free(state);
        @memset(state, 0); // zero state: row 0's own effect is the only input
        const km = try alloc.alloc(f32, rows * heads * vd);
        defer alloc.free(km);
        const dl = try alloc.alloc(f32, rows * heads * vd);
        defer alloc.free(dl);
        @memset(out[run], 0);

        vt.gdnStepRowsChunk(ctx,
            .{ .ptr = state.ptr, .len = state.len },
            .{ .ptr = qkv.ptr + nk * kd, .len = rows * conv_dim - nk * kd },
            .{ .ptr = qkv.ptr + 2 * nk * kd, .len = rows * conv_dim - 2 * nk * kd },
            .{ .ptr = qkv.ptr, .len = rows * conv_dim },
            .{ .ptr = km.ptr, .len = km.len },
            .{ .ptr = dl.ptr, .len = dl.len },
            .{ .ptr = out[run].ptr, .len = out[run].len },
            .{ .ptr = gamma.ptr, .len = gamma.len },
            .{ .ptr = beta.ptr, .len = beta.len },
            heads, nk, kd, vd, rows, conv_dim, conv_dim, conv_dim);
    }

    // Row 1 read row 0's state, so row 0's k moved row 1's output.
    const row1 = heads * vd;
    try std.testing.expect(!std.mem.eql(f32, out[0][row1..], out[1][row1..]));
}
