//! Row-level parallel matvec pool.
//!
//! The engine's linear layers are `out[r] = dot(row r)`.  Rows are
//! independent, so splitting the row loop across worker threads does not
//! change the order of the per-row summation -> every `out[r]` is
//! bit-identical to the serial path, and generation output is identical.
//!
//! Protocol: the main thread quantizes the activation ONCE (shared, read-only
//! by all rows), publishes a task (gen + slot), bumps the `wake` futex word once
//! and issues ONE `Io.futexWake` for all workers, then claims row chunks itself.
//! Workers snapshot the task and claim chunks with `fetchAdd` until the row
//! range is exhausted, incrementing a completion count.  The main thread
//! spin-waits for the completion count (it is doing useful work while claiming,
//! so the spin is productive).
//! Workers park on the `wake` futex when idle -- never busy-wait.  A single
//! futex syscall per publish replaces the old n-1 `sem.post()` calls, which all
//! serialised on one mutex/cond while main was computing (measured ~23 ms/token
//! at threads 24).

const std = @import("std");
const Io = std.Io;
const simd_api = @import("simd_api.zig");
const simd = @import("simd_dispatch.zig");

const RowDotScalar = *const fn (row: []const u8, x: []const f32) f32;

// ---- TEMP(bytes): per-kernel-family weight bytes read, driven from pool tasks.
// `stat_on` gates the fetchAdd calls.  The counters live here (not inside the
// tier modules) because `runRows` is the single point that sees every row's
// slice length, and main can reach them via `qwen35.pool`.  Atomic because
// runRows runs on worker threads.
pub var stat_on: bool = false;
pub var stat256: std.atomic.Value(i64) = .init(0);
pub var stat32: std.atomic.Value(i64) = .init(0);
pub var statsc: std.atomic.Value(i64) = .init(0);
pub var statq8: std.atomic.Value(i64) = .init(0);
pub var statf32: std.atomic.Value(i64) = .init(0);

pub const StatSnap = struct { k256: i64, k32: i64, ksc: i64, kq8: i64, kf32: i64 };

pub fn snapshot() StatSnap {
    return .{
        .k256 = stat256.load(.acquire),
        .k32 = stat32.load(.acquire),
        .ksc = statsc.load(.acquire),
        .kq8 = statq8.load(.acquire),
        .kf32 = statf32.load(.acquire),
    };
}

pub fn reset() void {
    stat256.store(0, .release);
    stat32.store(0, .release);
    statsc.store(0, .release);
    statq8.store(0, .release);
    statf32.store(0, .release);
}

/// The AVX-512 VNNI row kernel (vpdpbusd, two interleaved block chains).
/// Always returns the C kernel for maximum performance.
inline fn q8RowDot() ?simd_api.Dot32 {
    return simd.d32row.q8_0_q8_0;
}

// ---------------------------------------------------------------------------
// Prof hooks: three-segment timing of one parallel() call.  Active only when
// `prof` and `prof_phase` are both set by the engine (QWEN35_PROF); when off,
// parallel() executes exactly the same code as before.  Only the main thread
// touches these counters, so no synchronisation is needed.
pub var prof: bool = false;
pub var prof_io: Io = undefined;
/// engine flips this true at decode start; only then do we accumulate.
pub var prof_phase: bool = false;
pub var prof_pub: f64 = 0; // chunk calc + slot publish + the single futexWake
pub var prof_slot: f64 = 0; // subset of pub: chunk calc + slot stores + gen store
pub var prof_post: f64 = 0; // subset of pub: wake store + one futexWake syscall
pub var prof_main: f64 = 0; // main thread's own runRows
pub var prof_tail: f64 = 0; // main thread spin-wait for worker finish
pub var prof_calls: usize = 0; // parallel() calls that went through the pool
pub var prof_serial: usize = 0;
pub var prof_serial_ns: f64 = 0; // serial fallback inside parallel() (small matvecs)
pub var prof_post_only: bool = false; // microprobe mode: parallel() does no compute
/// per-rows attribution: which matvec sizes pay the publish/tail cost.
const rb_n = 16;
var rb_rows: [rb_n]usize = std.mem.zeroes([rb_n]usize);
var rb_calls: [rb_n]usize = std.mem.zeroes([rb_n]usize);
var rb_pub: [rb_n]f64 = std.mem.zeroes([rb_n]f64);
var rb_main: [rb_n]f64 = std.mem.zeroes([rb_n]f64);
var rb_tail: [rb_n]f64 = std.mem.zeroes([rb_n]f64);

fn rbIdx(rows: usize) usize {
    var i: usize = 0;
    while (i < rb_n) : (i += 1) {
        if (rb_rows[i] == rows) return i;
    }
    i = 0;
    while (i < rb_n) : (i += 1) {
        if (rb_rows[i] == 0) {
            rb_rows[i] = rows;
            return i;
        }
    }
    return 0; // overflow: lump into entry 0
}

pub fn profReset() void {
    prof_pub = 0;
    prof_slot = 0;
    prof_post = 0;
    prof_main = 0;
    prof_tail = 0;
    prof_calls = 0;
    prof_serial = 0;
    prof_serial_ns = 0;
    for (0..rb_n) |i| {
        rb_rows[i] = 0;
        rb_calls[i] = 0;
        rb_pub[i] = 0;
        rb_main[i] = 0;
        rb_tail[i] = 0;
    }
}

/// Cost of one Io.Clock.Timestamp.now(prof_io, .awake) read, ns.
pub fn profClockNs() f64 {
    const t0 = Io.Clock.Timestamp.now(prof_io, .awake);
    var i: usize = 0;
    while (i < 1000) : (i += 1) _ = Io.Clock.Timestamp.now(prof_io, .awake);
    return @as(f64, @floatFromInt(t0.durationTo(Io.Clock.Timestamp.now(prof_io, .awake)).raw.nanoseconds)) / 1000.0;
}

pub fn profReport() void {
    const cal = profClockNs();
    std.debug.print("PROF-P calls={d} pub={d:.2}ms [slot={d:.2}ms post={d:.2}ms] main={d:.2}ms tail={d:.2}ms serialcalls={d} serialms={d:.2}ms clock_read={d:.3}us postonly={d}\n", .{
        prof_calls, prof_pub / 1e6, prof_slot / 1e6, prof_post / 1e6, prof_main / 1e6, prof_tail / 1e6,
        prof_serial, prof_serial_ns / 1e6, cal / 1e3, @intFromBool(prof_post_only),
    });
    // per matvec size (rows of the published task) attribution, pub+tail desc
    var ord: [rb_n]usize = undefined;
    var cnt: usize = 0;
    var qi: usize = 0;
    while (qi < rb_n) : (qi += 1) {
        if (rb_calls[qi] > 0) {
            ord[cnt] = qi;
            cnt += 1;
        }
    }
    var k: usize = 0;
    while (k < cnt) : (k += 1) {
        var best = k;
        var j = k + 1;
        while (j < cnt) : (j += 1) {
            const a = rb_pub[ord[j]] + rb_tail[ord[j]];
            const b = rb_pub[ord[best]] + rb_tail[ord[best]];
            if (a > b) best = j;
        }
        const tmp = ord[k];
        ord[k] = ord[best];
        ord[best] = tmp;
    }
    for (ord[0..cnt]) |i| std.debug.print("PROF-R rows={d} calls={d} pub={d:.2}ms main={d:.2}ms tail={d:.2}ms\n", .{
        rb_rows[i], rb_calls[i], rb_pub[i] / 1e6, rb_main[i] / 1e6, rb_tail[i] / 1e6,
    });
}

pub const Task = union(enum) {
    /// K/IQ types with a 256-block SIMD dot: out[r] = dot(bpr*256, row, q8k)
    quant256: struct {
        out: [*]f32,
        data: []const u8,
        q8k: [*]const u8,
        row_bytes: usize,
        bpr: usize,
        rows: usize,
        dot: simd_api.Dot256,
    },
    /// 32-block types framed as 256-element rows: per row, 8*bpr block dots.
    quant32: struct {
        out: [*]f32,
        data: []const u8,
        q8: [*]const u8,
        rb32: usize,
        bpr: usize,
        rows: usize,
        dot: simd_api.Dot32,
    },
    /// Remaining IQ types: scalar per-block rowDot against the f32 activation.
    quant_scalar: struct {
        out: [*]f32,
        data: []const u8,
        x: []const f32,
        row_bytes: usize,
        bpr: usize,
        rows: usize,
        dot: RowDotScalar,
    },
    /// Q8_0 rows: per row, row_blocks i32 block dots.  `dot` is the row-level
    /// kernel (AVX-512 VNNI vpdpbusd on tiers that have it); when it is null
    /// (baseline tier, or a caller that did not set it) the scalar per-block
    /// loop below is used instead.
    q8_0: struct {
        out: [*]f32,
        data: []const u8,
        q8: [*]const u8,
        row_blocks: usize,
        rows: usize,
        dot: ?simd_api.Dot32 = null,
    },
    /// f32 rows.
    f32: struct {
        out: [*]f32,
        mat: []const f32,
        x: []const f32,
        cols: usize,
        rows: usize,
    },

    // ---- batched (K-token) variants -------------------------------------
    //
    // `rows` is the row count of ONE token's matvec, `k` the number of tokens
    // batched into one task.  The virtual row index `r` covers `rows * k`
    // work units: token `t = r / rows`, row `rr = r % rows`, output slot
    // `out[t * rows + rr]`, activation slot `base[t * len ..]`.  The per-row
    // inner summation is the exact same `dot` call as the serial path, so
    // every out[t*rows+rr] is bit-identical to a serial per-token matvec.

    quant256b: struct {
        out: [*]f32,
        data: []const u8,
        q8k: [*]const u8,
        q8k_len: usize,
        row_bytes: usize,
        bpr: usize,
        rows: usize,
        k: usize,
        dot: simd_api.Dot256,
    },
    quant32b: struct {
        out: [*]f32,
        data: []const u8,
        q8: [*]const u8,
        q8_len: usize,
        rb32: usize,
        bpr: usize,
        rows: usize,
        k: usize,
        dot: simd_api.Dot32,
    },
    quant_scalarb: struct {
        out: [*]f32,
        data: []const u8,
        x: [*]const f32,
        x_len: usize,
        row_bytes: usize,
        bpr: usize,
        rows: usize,
        k: usize,
        dot: RowDotScalar,
    },
    q8_0b: struct {
        out: [*]f32,
        data: []const u8,
        q8: [*]const u8,
        q8_len: usize,
        row_blocks: usize,
        rows: usize,
        k: usize,
    },
    f32b: struct {
        out: [*]f32,
        mat: []const f32,
        x: [*]const f32,
        x_len: usize,
        cols: usize,
        rows: usize,
        k: usize,
    },
};

pub const Slot = struct {
    task: ?Task = null,
    next: std.atomic.Value(usize) = .init(0),
    done: std.atomic.Value(usize) = .init(0),
    chunks: usize = 0,
    chunk: usize = 1,
    rows: usize = 0,
};

pub const slots_n = 16;

pub const Pool = struct {
    io: Io,
    /// kept for API/compat; the hot wake-up path uses `wake` (one futex) instead
    /// of `n-1` `sem.post()` calls on a shared mutex/cond.
    sem: Io.Semaphore = .{},
    /// global task generation; the active slot is slots[gen % slots_n]
    gen: std.atomic.Value(u64) = .init(0),
    /// futex word for worker wake-ups.  Monotonic mod 2^32: `@truncate(gen)` is
    /// stored here, so every publish changes it -> a worker that read the old
    /// value cannot sleep (the kernel check `*ptr == expected` fails).
    wake: std.atomic.Value(u32) = .init(0),
    slots: [slots_n]Slot = std.mem.zeroes([slots_n]Slot),
    threads: []std.Thread = &.{},
    n: usize = 1,
    closed: std.atomic.Value(bool) = .init(false),
};

fn rowsOf(t: Task) usize {
    return switch (t) {
        .quant256 => |q| q.rows,
        .quant32 => |q| q.rows,
        .quant_scalar => |q| q.rows,
        .q8_0 => |q| q.rows,
        .f32 => |q| q.rows,
        .quant256b => |q| q.rows * q.k,
        .quant32b => |q| q.rows * q.k,
        .quant_scalarb => |q| q.rows * q.k,
        .q8_0b => |q| q.rows * q.k,
        .f32b => |q| q.rows * q.k,
    };
}

/// Minimum virtual-row count worth splitting across the pool (same threshold
/// as the serial path, applied to `rows * k`).
fn minParallelRows(t: Task) usize {
    return switch (t) {
        .quant256, .quant32, .quant_scalar, .q8_0, .f32 => 256,
        .quant256b, .quant32b, .quant_scalarb, .q8_0b, .f32b => 256,
    };
}

/// Compute rows [lo, hi) of a published task.  Row loop order within the chunk
/// is ascending, and each row's inner summation is unchanged -> bit-identical
/// to the serial path.
fn runRows(t: Task, lo: usize, hi: usize) void {
    switch (t) {
        .quant256 => |q| {
            var r = lo;
            while (r < hi) : (r += 1) {
                const wrow = q.data[r * q.bpr * q.row_bytes ..][0 .. q.bpr * q.row_bytes];
                if (stat_on) _ = stat256.fetchAdd(@intCast(q.bpr * q.row_bytes), .monotonic);
                q.out[r] = q.dot(q.bpr * 256, wrow.ptr, q.q8k);
            }
        },
        .quant32 => |q| {
            const q8sl: []const u8 = q.q8[0 .. q.bpr * 8 * 34];
            var r = lo;
            while (r < hi) : (r += 1) {
                const wrow = q.data[r * q.bpr * q.rb32 * 8 ..][0 .. q.bpr * q.rb32 * 8];
                if (stat_on) _ = stat32.fetchAdd(@intCast(q.bpr * q.rb32 * 8), .monotonic);
                var acc: f32 = 0;
                for (0..8 * q.bpr) |k| {
                    acc += q.dot(32, wrow[k * q.rb32 ..].ptr, q8sl[k * 34 ..].ptr);
                }
                q.out[r] = acc;
            }
        },
        .quant_scalar => |q| {
            var r = lo;
            while (r < hi) : (r += 1) {
                var acc: f32 = 0;
                for (0..q.bpr) |b| {
                    const blk = q.data[(r * q.bpr + b) * q.row_bytes ..];
                    if (stat_on) _ = statsc.fetchAdd(@intCast(q.row_bytes), .monotonic);
                    acc += q.dot(blk[0..q.row_bytes], q.x[b * 256 .. b * 256 + 256]);
                }
                q.out[r] = acc;
            }
        },
        .q8_0 => |q| {
            const blen = q.row_blocks * 34;
            const rowdot = q.dot orelse q8RowDot();
            if (rowdot) |f| {
                const n = q.row_blocks * 32;
                var r = lo;
                while (r < hi) : (r += 1) {
                    const wrow = q.data[r * blen ..][0..blen];
                    if (stat_on) _ = statq8.fetchAdd(@intCast(blen), .monotonic);
                    q.out[r] = f(n, wrow.ptr, q.q8);
                }
                return;
            }
            const q8sl: []const u8 = q.q8[0 .. q.row_blocks * 34];
            var r = lo;
            while (r < hi) : (r += 1) {
                const wrow = q.data[r * blen ..][0..blen];
                if (stat_on) _ = statq8.fetchAdd(@intCast(blen), .monotonic);
                var acc: f32 = 0;
                for (0..q.row_blocks) |b| {
                    const wb = wrow[b * 34 ..][0..34];
                    const wscale: f32 = @as(f32, @floatCast(@as(f16, @bitCast(std.mem.readInt(u16, wb[0..2], .little)))));
                    const xscale: f32 = @as(f32, @floatCast(@as(f16, @bitCast(std.mem.readInt(u16, q8sl[b * 34 ..][0..2], .little)))));
                    var dot: i32 = 0;
                    for (0..32) |j| {
                        dot += @as(i32, @as(i8, @bitCast(wb[2 + j]))) * @as(i32, @as(i8, @bitCast(q8sl[b * 34 + 2 + j])));
                    }
                    acc += wscale * @as(f32, @floatFromInt(dot)) * xscale;
                }
                q.out[r] = acc;
            }
        },
        .f32 => |q| {
            var r = lo;
            while (r < hi) : (r += 1) {
                const rowv = q.mat[r * q.cols ..][0..q.cols];
                if (stat_on) _ = statf32.fetchAdd(@intCast(q.cols * 4), .monotonic);
                var acc: f32 = 0;
                for (rowv, q.x) |mv, vv| acc += mv * vv;
                q.out[r] = acc;
            }
        },
        .quant256b => |q| {
            var r = lo;
            while (r < hi) : (r += 1) {
                const tok = r / q.rows;
                const rr = r % q.rows;
                const wrow = q.data[rr * q.bpr * q.row_bytes ..][0 .. q.bpr * q.row_bytes];
                if (stat_on) _ = stat256.fetchAdd(@intCast(q.bpr * q.row_bytes), .monotonic);
                q.out[tok * q.rows + rr] = q.dot(q.bpr * 256, wrow.ptr, q.q8k[tok * q.q8k_len ..]);
            }
        },
        .quant32b => |q| {
            var r = lo;
            while (r < hi) : (r += 1) {
                const tok = r / q.rows;
                const rr = r % q.rows;
                const q8sl: []const u8 = q.q8[tok * q.q8_len ..][0 .. q.bpr * 8 * 34];
                const wrow = q.data[rr * q.bpr * q.rb32 * 8 ..][0 .. q.bpr * q.rb32 * 8];
                if (stat_on) _ = stat32.fetchAdd(@intCast(q.bpr * q.rb32 * 8), .monotonic);
                var acc: f32 = 0;
                for (0..8 * q.bpr) |kk| {
                    acc += q.dot(32, wrow[kk * q.rb32 ..].ptr, q8sl[kk * 34 ..].ptr);
                }
                q.out[tok * q.rows + rr] = acc;
            }
        },
        .quant_scalarb => |q| {
            var r = lo;
            while (r < hi) : (r += 1) {
                const tok = r / q.rows;
                const rr = r % q.rows;
                const xs = q.x[tok * q.x_len ..];
                var acc: f32 = 0;
                for (0..q.bpr) |b| {
                    const blk = q.data[(rr * q.bpr + b) * q.row_bytes ..];
                    if (stat_on) _ = statsc.fetchAdd(@intCast(q.row_bytes), .monotonic);
                    acc += q.dot(blk[0..q.row_bytes], xs[b * 256 .. b * 256 + 256]);
                }
                q.out[tok * q.rows + rr] = acc;
            }
        },
        .q8_0b => |q| {
            const blen = q.row_blocks * 34;
            var r = lo;
            while (r < hi) : (r += 1) {
                const tok = r / q.rows;
                const rr = r % q.rows;
                const q8sl: []const u8 = q.q8[tok * q.q8_len ..][0..blen];
                const wrow = q.data[rr * blen ..][0..blen];
                if (stat_on) _ = statq8.fetchAdd(@intCast(blen), .monotonic);
                var acc: f32 = 0;
                for (0..q.row_blocks) |b| {
                    const wb = wrow[b * 34 ..][0..34];
                    const wscale: f32 = @as(f32, @floatCast(@as(f16, @bitCast(std.mem.readInt(u16, wb[0..2], .little)))));
                    const xscale: f32 = @as(f32, @floatCast(@as(f16, @bitCast(std.mem.readInt(u16, q8sl[b * 34 ..][0..2], .little)))));
                    var dot: i32 = 0;
                    for (0..32) |j| {
                        dot += @as(i32, @as(i8, @bitCast(wb[2 + j]))) * @as(i32, @as(i8, @bitCast(q8sl[b * 34 + 2 + j])));
                    }
                    acc += wscale * @as(f32, @floatFromInt(dot)) * xscale;
                }
                q.out[tok * q.rows + rr] = acc;
            }
        },
        .f32b => |q| {
            var r = lo;
            while (r < hi) : (r += 1) {
                const tok = r / q.rows;
                const rr = r % q.rows;
                const rowv = q.mat[rr * q.cols ..][0..q.cols];
                if (stat_on) _ = statf32.fetchAdd(@intCast(q.cols * 4), .monotonic);
                const xs = q.x[tok * q.x_len ..];
                var acc: f32 = 0;
                for (rowv, xs) |mv, vv| acc += mv * vv;
                q.out[tok * q.rows + rr] = acc;
            }
        },
    }
}

fn workerMain(pool: *Pool) void {
    var seen: u64 = 0;
    while (true) {
        const g = pool.gen.load(.acquire);
        if (g == seen) {
            // park on the futex.  Read the wake word FIRST, then sleep.  The
            // expected value is @truncate(seen) -- the generation this worker
            // has consumed -- not the freshly loaded word: if the main thread
            // publishes between the two loads, we see wake != @truncate(seen)
            // and skip the sleep, re-checking gen on the next iteration.  If it
            // publishes after our load and before the syscall, the kernel's
            // `*ptr == expected` check fails -> futexWait returns at once.
            const w = pool.wake.load(.acquire);
            if (w == @as(u32, @truncate(seen))) pool.io.futexWaitUncancelable(u32, &pool.wake.raw, w);
            continue; // spurious wake / this wake was for a gen already consumed
        }
        seen = g;
        if (pool.closed.load(.acquire)) return; // deinit() published the close sentinel
        const slot = &pool.slots[g % slots_n];
        const t = slot.task.?;
        const rows = rowsOf(t);
        const chunk = slot.chunk;
        while (true) {
            const lo = slot.next.fetchAdd(chunk, .acq_rel);
            if (lo >= rows) break;
            var hi = lo + chunk;
            if (hi > rows) hi = rows;
            runRows(t, lo, hi);
            _ = slot.done.fetchAdd(1, .acq_rel);
        }
    }
}

pub fn init(io: Io, alloc: std.mem.Allocator, n: usize) !*Pool {
    const pool = try alloc.create(Pool);
    pool.* = .{ .io = io, .n = n };
    if (n <= 1) return pool;
    pool.threads = try alloc.alloc(std.Thread, n - 1);
    var i: usize = 0;
    while (i < n - 1) : (i += 1) {
        // the worker parks on the `wake` futex; no work is published until
        // the first `parallel` call, so spawning here is safe.
        pool.threads[i] = try std.Thread.spawn(.{}, workerMain, .{pool});
    }
    return pool;
}

pub fn deinit(self: *Pool, alloc: std.mem.Allocator) void {
    if (self.n <= 1) {
        alloc.destroy(self);
        return;
    }
    // close: workers see the closed flag on the next generation bump and exit
    self.closed.store(true, .release);
    const g = self.gen.load(.monotonic) + 1;
    self.gen.store(g, .release);
    // bump the wake word and wake every parked worker exactly once
    self.wake.store(@truncate(g), .release);
    self.io.futexWake(u32, &self.wake.raw, @intCast(self.n - 1));
    for (self.threads) |t| t.join();
    alloc.free(self.threads);
    alloc.destroy(self);
}

/// Run a matvec task across the pool.  `pool.n` threads participate: the main
/// thread claims chunks too.  Falls back to the serial loop when there is no
/// pool / one thread / too few rows to be worth splitting.
pub fn parallel(pool: ?*Pool, t: Task) void {
    const rows = rowsOf(t);
    const p = pool orelse {
        runRows(t, 0, rows);
        return;
    };
    if (p.n <= 1 or p.closed.load(.acquire) or rows < minParallelRows(t)) {
        if (prof and prof_phase) {
            const ts = Io.Clock.Timestamp.now(p.io, .awake);
            runRows(t, 0, rows);
            prof_serial_ns += @as(f64, @floatFromInt(ts.durationTo(Io.Clock.Timestamp.now(p.io, .awake)).raw.nanoseconds));
            prof_serial += 1;
        } else {
            runRows(t, 0, rows);
        }
        return;
    }
    // segment clocks (main thread only; zero-cost when prof is off)
    const prof_on = prof and prof_phase;
    const t0 = if (prof_on) Io.Clock.Timestamp.now(p.io, .awake) else @as(Io.Clock.Timestamp, undefined);

    // chunk: aim for ~4 chunks per thread, bounded to keep atomics cheap
    const per: usize = 4 * p.n;
    var chunk = rows / per;
    if (chunk < 16) chunk = 16;
    if (chunk > 256) chunk = 256;
    const chunks = (rows + chunk - 1) / chunk;

    // publish into the slot for the next generation; old slots keep their
    // (exhausted) claim counters so a lagging worker from the previous task
    // never touches a reset slot.
    const g = p.gen.load(.monotonic) + 1;
    const slot = &p.slots[g % slots_n];
    slot.task = t;
    slot.next.store(0, .monotonic);
    slot.done.store(0, .monotonic);
    slot.chunks = chunks;
    slot.chunk = chunk;
    slot.rows = rows;
    p.gen.store(g, .release);
    const t0b = if (prof_on) Io.Clock.Timestamp.now(p.io, .awake) else @as(Io.Clock.Timestamp, undefined);
    // one futex wake-up for all n-1 workers (replaces n-1 sem.post()).  The wake
    // word must change on every publish, so that a worker which read it before
    // this store cannot fall asleep (`*ptr == expected` fails).
    p.wake.store(@truncate(g), .release);
    p.io.futexWake(u32, &p.wake.raw, @intCast(p.n - 1));

    // main participates
    const t1 = if (prof_on) Io.Clock.Timestamp.now(p.io, .awake) else @as(Io.Clock.Timestamp, undefined);
    if (prof_post_only) {
        // timing probe: main skips all compute and the tail wait, so pub is
        // measured with no main-thread memory pressure.  Output is garbage.
        if (prof_on) prof_pub += @as(f64, @floatFromInt(t0.durationTo(t1).raw.nanoseconds));
        if (prof_on) prof_slot += @as(f64, @floatFromInt(t0.durationTo(t0b).raw.nanoseconds));
        if (prof_on) prof_post += @as(f64, @floatFromInt(t0b.durationTo(t1).raw.nanoseconds));
        prof_calls += 1;
        return;
    }
    while (true) {
        const lo = slot.next.fetchAdd(chunk, .acq_rel);
        if (lo >= rows) break;
        var hi = lo + chunk;
        if (hi > rows) hi = rows;
        runRows(t, lo, hi);
        _ = slot.done.fetchAdd(1, .acq_rel);
    }
    const t2 = if (prof_on) Io.Clock.Timestamp.now(p.io, .awake) else @as(Io.Clock.Timestamp, undefined);
    while (slot.done.load(.acquire) < chunks) std.Thread.yield() catch {};
    if (prof_on) {
        const t3 = Io.Clock.Timestamp.now(p.io, .awake);
        const ri = rbIdx(rows);
        prof_pub += @as(f64, @floatFromInt(t0.durationTo(t1).raw.nanoseconds));
        prof_slot += @as(f64, @floatFromInt(t0.durationTo(t0b).raw.nanoseconds));
        prof_post += @as(f64, @floatFromInt(t0b.durationTo(t1).raw.nanoseconds));
        prof_main += @as(f64, @floatFromInt(t1.durationTo(t2).raw.nanoseconds));
        prof_tail += @as(f64, @floatFromInt(t2.durationTo(t3).raw.nanoseconds));
        prof_calls += 1;
        rb_calls[ri] += 1;
        rb_pub[ri] += @as(f64, @floatFromInt(t0.durationTo(t1).raw.nanoseconds));
        rb_main[ri] += @as(f64, @floatFromInt(t1.durationTo(t2).raw.nanoseconds));
        rb_tail[ri] += @as(f64, @floatFromInt(t2.durationTo(t3).raw.nanoseconds));
    }
}
