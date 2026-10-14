//! Deterministic sampler shared by the CPU and CUDA engines.
//!
//! Same logits + same seed => same token, on every platform.  The PRNG is a
//! xorshift64* (64-bit state, single stream) so a fixed `-s SEED` makes a
//! run bit-reproducible.
//!
//! Sampling policy (mirrors llama2's C reference, see /mnt/llm-dev/llama2/c/
//! llama2_cpu.c):
//!   * temperature <= 0          -> greedy argmax (ties: lowest index)
//!   * else                      -> p = softmax(logits / T), then top-k
//!                                   truncation, then draw one u32 from the
//!                                   PRNG and walk the cumulative mass.
//!
//! The whole thing is plain f32/f64 Zig with no platform-specific intrinsics,
//! so the CPU binary and the CUDA binary pick identical tokens whenever the
//! logits they feed in are identical (in practice they differ by ~1e-6 from
//! the two forward-pass implementations; sampling is usually stable across
//! that gap, a tie-flip is possible and expected).

const std = @import("std");

pub const Sampler = struct {
    state: u64,
    temperature: f32,
    top_k: usize,
    /// Original seed, kept so `reseed` can restore it (prefill draws consume
    /// the stream; decode must start from the seed on both backends).
    seed: u64,

    pub const Config = struct {
        /// 0 disables sampling (greedy argmax, the historical default).
        temperature: f32 = 0,
        /// 0 disables the top-k cutoff (pure temperature sampling).
        top_k: usize = 0,
        /// PRNG seed; 0 selects a fixed non-zero default.
        seed: u64 = 0,
    };

    pub fn init(cfg: Config) Sampler {
        var s = cfg.seed;
        if (s == 0) s = 0x9E3779B97F4A7C15; // non-zero golden-ratio constant
        return .{ .state = s, .temperature = cfg.temperature, .top_k = cfg.top_k, .seed = s };
    }

    /// Restore the PRNG to its original seed.
    pub fn reseed(self: *Sampler) void {
        self.state = self.seed;
    }

    /// Next 64-bit PRNG word (xorshift64*).  The final multiply wraps, so it
    /// is done in u128 and truncated (C's implicit low-64 wrap-around, which
    /// plain Zig `*` would trap on).
    pub fn nextU64(self: *Sampler) u64 {
        var s = self.state;
        s ^= s >> 12;
        s ^= s << 25;
        s ^= s >> 27;
        self.state = s;
        const full: u128 = @as(u128, s) * @as(u128, 0x2545F4914F6CDD1D);
        return @truncate(full);
    }

    /// Uniform float in [0, 1) from the top 24 bits.
    pub fn nextF32(self: *Sampler) f32 {
        return @as(f32, @floatFromInt(self.nextU64() >> 40)) / 16777216.0;
    }

    /// Draw one token.  `scratch` must hold `logits.len` floats when
    /// `top_k` is active (k-th order statistic via quickselect, order of
    /// scratch is destroyed); pass an empty slice otherwise.
    pub fn sample(self: *Sampler, logits: []const f32, scratch: []f32) u32 {
        const n = logits.len;
        if (self.temperature <= 0) {
            var best: u32 = 0;
            var bv = -std.math.floatMax(f32);
            for (logits, 0..) |v, i| {
                if (v > bv) {
                    bv = v;
                    best = @intCast(i);
                }
            }
            return best;
        }

        const t = self.temperature;

        // Running max for a numerically-stable softmax.
        var maxv: f32 = logits[0];
        for (logits[1..]) |v| {
            if (v > maxv) maxv = v;
        }

        // Top-k threshold: the k-th largest logit.  Ties at the threshold are
        // all kept (standard, and keeps CPU/CUDA behaviour identical).
        var thresh: f32 = -std.math.floatMax(f32);
        const k = self.top_k;
        if (k > 0 and k < n) {
            if (scratch.len < n) @panic("sampler: scratch too small for top_k");
            @memcpy(scratch, logits);
            thresh = kthLargest(scratch, k - 1);
        }

        // Mass over the surviving candidates (f64 accumulation: 248k terms).
        var sum: f64 = 0;
        for (logits) |v| {
            if (v >= thresh) sum += std.math.exp(@as(f32, (v - maxv) / t));
        }

        const coin: f64 = @as(f64, self.nextF32());
        var acc: f64 = 0;
        var i: u32 = 0;
        while (i < n) : (i += 1) {
            if (logits[i] < thresh) continue;
            acc += std.math.exp(@as(f32, (logits[i] - maxv) / t));
            if (coin < acc / sum) return i;
        }
        // Fallback: last surviving index (coin could sit just below the f64
        // rounding of the total mass).
        while (i > 0) : (i -= 1) {
            if (logits[i - 1] >= thresh) return i - 1;
        }
        return 0;
    }

    /// k-th largest element (0-based rank from the top), via a deterministic
    /// 3-way quickselect (median-of-three pivot) so the result is reproducible
    /// even with float ties.  Destroys the order of `a`; requires
    /// `rank < a.len` and `a.len >= 1`.
    pub fn kthLargest(a: []f32, rank: usize) f32 {
        var lo: usize = 0;
        var hi: usize = a.len; // exclusive
        var kk: usize = rank + 1; // 1-indexed k-th largest
        while (true) {
            if (hi - lo <= 1) return a[lo];
            // Deterministic pivot: median of three (a[lo], a[mid], a[hi-1]).
            const mid = lo + (hi - lo) / 2;
            var x = a[lo];
            var y = a[mid];
            var z = a[hi - 1];
            if (x > y) {
                const t = x;
                x = y;
                y = t;
            }
            if (x > z) {
                const t = x;
                x = z;
                z = t;
            }
            if (y > z) {
                const t = y;
                y = z;
                z = t;
            }
            const p = y; // median

            // 3-way Dutch-national-flag partition of a[lo..hi) around p:
            // invariant  a[lo..i) < p,  a[i..k] == p,  a[k+1..hi) > p.
            var i = lo;
            var k = hi - 1;
            var j = lo;
            while (j <= k) {
                if (a[j] < p) {
                    const t = a[j];
                    a[j] = a[i];
                    a[i] = t;
                    i += 1;
                    j += 1;
                } else if (a[j] > p) {
                    const t = a[j];
                    a[j] = a[k];
                    a[k] = t;
                    k -= 1;
                } else {
                    j += 1;
                }
            }
            // Descending order is [>p | ==p | <p].
            const g = hi - 1 - k; // count of > p
            const e = k - i + 1; // count of == p
            if (kk <= g) {
                lo = k + 1; // target in the >p block; kk unchanged
            } else if (kk <= g + e) {
                return p; // the pivot value is the k-th largest
            } else {
                kk -= g + e; // target in the <p block (a[lo..i))
                hi = i;
            }
        }
    }
};

test "sampler: greedy argmax picks the max, ties go to the low index" {
    var s = Sampler.init(.{ .temperature = 0, .seed = 1 });
    const l: [5]f32 = .{ 1.0, 3.0, 2.0, 3.0, -1.0 };
    try std.testing.expectEqual(@as(u32, 1), s.sample(&l, &[_]f32{}));
}

test "sampler: temperature sampling is deterministic for a fixed seed" {
    const l: [8]f32 = .{ 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8 };
    var a = Sampler.init(.{ .temperature = 0.8, .top_k = 0, .seed = 42 });
    var b = Sampler.init(.{ .temperature = 0.8, .top_k = 0, .seed = 42 });
    const ta = a.sample(&l, &[_]f32{});
    const tb = b.sample(&l, &[_]f32{});
    try std.testing.expectEqual(ta, tb);
    // Two draws from the same sampler must consume the PRNG stream
    // identically (determinism of the whole sequence).
    var c = Sampler.init(.{ .temperature = 0.8, .top_k = 0, .seed = 42 });
    _ = c.sample(&l, &[_]f32{});
    const second = c.sample(&l, &[_]f32{});
    var d = Sampler.init(.{ .temperature = 0.8, .top_k = 0, .seed = 42 });
    _ = d.sample(&l, &[_]f32{});
    const second2 = d.sample(&l, &[_]f32{});
    try std.testing.expectEqual(second, second2);
}

test "sampler: top-k excludes the low-prob tail" {
    // One far-away outlier must never be picked when k=1.
    const l: [6]f32 = .{ 0.0, 0.0, 0.0, 0.0, 10.0, 0.0 };
    var scratch: [6]f32 = undefined;
    @memset(&scratch, 0);
    var s = Sampler.init(.{ .temperature = 1.0, .top_k = 1, .seed = 7 });
    for (0..200) |_| {
        const t = s.sample(&l, &scratch);
        try std.testing.expectEqual(@as(u32, 4), t);
    }
}

test "kthLargest: ranks match a sorted reference" {
    const a = [_]f32{ 5, 1, 9, 3, 7, 2, 8, 4, 6 };
    var w = a;
    // k-th largest (0-based): 9,8,7,6,5,4,3,2,1
    try std.testing.expectEqual(@as(f32, 9), Sampler.kthLargest(&w, 0));
    @memcpy(&w, &a);
    try std.testing.expectEqual(@as(f32, 8), Sampler.kthLargest(&w, 1));
    @memcpy(&w, &a);
    try std.testing.expectEqual(@as(f32, 2), Sampler.kthLargest(&w, 7));
    @memcpy(&w, &a);
    try std.testing.expectEqual(@as(f32, 1), Sampler.kthLargest(&w, 8));
    // Determinism across calls on identical input.
    @memcpy(&w, &a);
    const r1 = Sampler.kthLargest(&w, 4);
    @memcpy(&w, &a);
    try std.testing.expectEqual(r1, Sampler.kthLargest(&w, 4));
}

test "kthLargest: float ties are kept (k-th largest is a value, not a slot)" {
    const a = [_]f32{ 3, 1, 3, 2, 3, 0, 3 };
    var w = a;
    // Sorted desc: 3,3,3,3,2,1,0 -> ranks 0..6
    try std.testing.expectEqual(@as(f32, 3), Sampler.kthLargest(&w, 0));
    @memcpy(&w, &a);
    try std.testing.expectEqual(@as(f32, 3), Sampler.kthLargest(&w, 3));
    @memcpy(&w, &a);
    try std.testing.expectEqual(@as(f32, 2), Sampler.kthLargest(&w, 4));
    @memcpy(&w, &a);
    try std.testing.expectEqual(@as(f32, 0), Sampler.kthLargest(&w, 6));
}

test "sampler: top-k with k >= n behaves like pure temperature" {
    const l: [4]f32 = .{ 0, 1, 2, 3 };
    var s = Sampler.init(.{ .temperature = 1.0, .top_k = 100, .seed = 3 });
    _ = s.sample(&l, &[_]f32{}); // must not touch scratch
}
