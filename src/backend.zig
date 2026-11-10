//! Hardware abstraction boundary.
//!
//! The engine models *one* architecture (Qwen3.5) and must not know how a
//! matvec is executed.  Everything that depends on the hardware -- which ISA
//! tier is available, whether weights are streamed from host memory or from
//! device memory, how a quantization block is unpacked -- lives behind `Exec`.
//!
//! Layout:
//!
//!   gguf.QuantRow   packed tensor + its layout geometry (no maths, no hardware)
//!   backend.zig     this file: the `Exec` interface + `LinearW` + dispatch
//!   backend_cpu.zig CPU implementation (C kernels under src/kernels/x86)
//!   backend_cuda.zig CUDA implementation (currently a CPU trampoline; see file)
//!
//! Adding a backend means implementing `Exec` and registering it in
//! `selectBackend`; the engine does not change.
//!
//! Activation and model state (residual stream, projections, KV cache, GDN
//! conv history and 128x128 state, logits) are **backend-owned regions**
//! (`Buf`), not host slices the engine dereferences.  Every per-op call takes
//! `Buf`s, so the arithmetic runs *inside* the backend: on a device backend the
//! activations stay on the device and nothing crosses PCIe between two ops; the
//! only host-visible values are the token ids the engine feeds in and the token
//! id it gets back.
//!
//! The CPU backend implements each elementwise slot with the same Zig loops the
//! engine used to contain (src/elementwise.zig), byte-for-byte, so the CPU path
//! is bit-identical with the historical engine.

const std = @import("std");
const gguf = @import("gguf.zig");
const sampler = @import("sampler.zig");

pub const QuantRow = gguf.QuantRow;

/// A 2-D linear weight as the engine sees it: either still packed in its
/// quantization format (the normal case, zero-copy -- this includes f16 and
/// bf16, which stay 2 B/elem and are widened inside the dot kernels) or, for
/// `.f32` tensors, the native f32 slice.
///
/// Both forms are zero-copy views into the GGUF mapping: `.f32` is the tensor
/// bytes reinterpreted as `[]const f32` (the mapping pads every tensor to a
/// 32-byte alignment, so the 4-byte f32 alignment holds).  Nothing in either
/// arm is allocated, so a `LinearW` carries no destructor of its own -- the
/// view is valid only while the mapping is, and a device backend uploads from
/// it without copying host-side first.
pub const LinearW = union(enum) {
    quant: QuantRow,
    f32: []const f32,
};

/// A region of activation / state memory owned by the backend.
///
/// This is what makes "skeleton on the host, arithmetic on the device" possible.
/// The engine no longer holds `[]f32` scratch it can only touch in host memory:
/// it holds opaque regions the backend may have made device-resident, and every
/// per-op call takes `Buf`s, so elementwise kernels (`rmsnorm`, `rope64`,
/// `convStep`, `gdnStep`, ...) read and write activations *inside* the backend
/// — nothing crosses PCIe between two ops.
///
/// `ptr` is a host pointer for the CPU backend and a device pointer for a
/// device backend; the engine must never dereference it.  `slice()` exists for
/// the CPU backend (and for the matvec/dequant signatures, which take plain
/// slices: on a device backend that slice is a device pointer, which the
/// backend's own kernel treats as such).
pub const Buf = struct {
    ptr: [*]f32,
    len: usize,

    pub fn slice(self: Buf) []f32 {
        return self.ptr[0..self.len];
    }
    /// Sub-window `[off, off+len)`.  Valid on both sides: pure pointer
    /// arithmetic on a `Buf`, the backend resolves its own base.
    pub fn span(self: Buf, off: usize, len: usize) Buf {
        std.debug.assert(off + len <= self.len);
        return .{ .ptr = self.ptr + off, .len = len };
    }
    /// Wrap an existing host slice (the CPU case for GGUF-side tensors: norm
    /// weights, conv taps, rope tables).  A device backend must instead upload
    /// the tensor at load time and hand back a device-backed `Buf` — see
    /// `loadVecF32`.
    pub fn wrap(s: []const f32) Buf {
        return .{ .ptr = @constCast(s.ptr), .len = s.len };
    }

    /// The "nothing allocated" sentinel: `len == 0`, pointer never
    /// dereferenced.  Use it for optional/unallocated scratch fields instead
    /// of `&.{}`, which is a host-slice value and says nothing about which
    /// address space the region lives in.
    pub const empty: Buf = .{ .ptr = undefined, .len = 0 };
};

/// Errors every backend can report while loading weights.
pub const LoadError = error{
    MissingTensor,
    UnsupportedTensorType,
    OutOfMemory,
    /// The selected backend (e.g. `--backend cuda`) cannot run on this host:
    /// no libcudart, no device, or a device too old for the compiled kernels.
    /// The reason is printed by the backend itself, because this error set only
    /// carries names.
    BackendUnavailable,
};

/// The hardware boundary.  One instance per loaded model.
///
/// Every method the engine calls during a forward pass is here; the engine
/// holds `*Exec` and never names a concrete backend.
pub const Exec = struct {
    ptr: *anyopaque,
    vtable: *const VTable,

    pub const VTable = struct {
        // ---- weight loading -------------------------------------------------
        /// Load one 2-D weight, keeping quantized tensors packed.
        loadLinear: *const fn (ctx: *anyopaque, name: []const u8) LoadError!LinearW,
        /// Release what `loadLinear`/`allocScratch` allocated.
        freeLinear: *const fn (ctx: *anyopaque, w: LinearW) void,
        /// Load one 1-D f32 weight vector (a norm's gamma, a conv tap row,
        /// `ssm_a`, `dt_bias`, ...) by tensor name.  The returned `Buf` is a
        /// backend region -- a host pointer on a CPU backend, a device pointer
        /// on a device backend -- that the elementwise slots read directly; the
        /// engine keeps only this handle and never dereferences it.  The tensor
        /// is widened (f16/bf16/quant -> f32) and, on a device backend, uploaded
        /// ONCE here at load time, not per token.
        loadVecF32: *const fn (ctx: *anyopaque, tensor_name: []const u8) LoadError!Buf,
        /// Release what `loadVecF32` allocated.  A `Buf.empty` argument is a
        /// no-op (the sentinel the engine uses for an unallocated optional
        /// vector), so callers can free unconditionally.
        freeVec: *const fn (ctx: *anyopaque, v: Buf) void,

        // ---- activation scratch --------------------------------------------
        /// Allocate the per-token / per-chunk scratch the kernels need.
        /// `max_cols` is the widest row any weight in the model can have.
        allocScratch: *const fn (ctx: *anyopaque, max_cols: usize) anyerror!void,
        /// Grow scratch for a K-token prefill chunk.
        allocScratchK: *const fn (ctx: *anyopaque, max_cols: usize, k: usize) anyerror!void,
        freeScratch: *const fn (ctx: *anyopaque) void,

        /// A region of activation/state memory the backend owns.  The engine
        /// asks for regions by length and never owns the memory itself: a CPU
        /// backend allocates from its allocator and wraps the slice, a device
        /// backend allocates on the device.  Regions live as long as the model
        /// run, released by `freeScratch` (all of them at once).
        allocBuf: *const fn (ctx: *anyopaque, len: usize) anyerror!Buf,

        // ---- dequantization -------------------------------------------------
        /// Expand one row of a packed weight into `dst` (f32, `cols` wide).
        /// Used by the embedding lookup, which needs values rather than dots.
        dequantRow: *const fn (ctx: *anyopaque, dst: Buf, w: QuantRow, row: usize) void,

        // ---- matvec ---------------------------------------------------------
        /// `out = W * x` with `W` packed, `x` `w.cols()` f32 values.  Both
        /// activation sides are `Buf`s, so on a device backend the input is the
        /// region the previous op wrote and the output is the region the next
        /// op reads: no PCIe between them.
        matvec: *const fn (ctx: *anyopaque, out: Buf, w: LinearW, x: Buf) void,
        /// Same, for `k` tokens at once: `out[j*rows + r] = W[r] . x_j`.
        matvecBatchK: *const fn (ctx: *anyopaque, out: Buf, w: LinearW, xK: Buf, k: usize) void,

        /// Create the intra-op thread pool (CPU); a device backend ignores it.
        /// `nt` is the worker count, 1 = serial.
        setThreads: *const fn (ctx: *anyopaque, io: std.Io, nt: usize) void,

        // ---- elementwise / state kernels -------------------------------------
        //
        // These used to be plain Zig loops inside the engine, which required the
        // activations to live in host memory.  They are backend calls now: the
        // engine hands `Buf`s (regions the backend owns), so a device backend
        // runs the arithmetic on the device and activation never leaves it.
        // The CPU backend implements each slot with the same Zig loops the
        // engine used to contain, byte-for-byte (src/elementwise.zig), so the
        // CPU path is bit-identical to the historical engine.

        /// out = x * rsqrt(mean(x^2) + eps) * w, all `n` wide.
        rmsnorm: *const fn (ctx: *anyopaque, out: Buf, x: Buf, w: Buf, eps: f32) void,
        /// `rows` consecutive `rmsnorm` calls on `out.span(r*h, h)` / `x.span(r*h, h)`.
        ///
        /// The ROW-BATCHED form: `out` and `x` are `rows * w.len` long and pack
        /// `rows` consecutive rows of `w.len` floats; `w` stays ONE row and is
        /// reused for every row.  Every row is normalised independently (its own
        /// sum of squares), so this is exactly `rows` consecutive `rmsnorm`
        /// calls on `out.span(r*h, h)` -- one host launch instead of `rows`.
        ///
        /// It is a SEPARATE slot rather than "infer rows from the lengths" (which
        /// `out.len == rows * w.len` would allow) because `copyRange` feeds host
        /// embedding rows through this vtable: a host-only `Buf` of length
        /// `rows * n` must NOT silently become a device launch.  A batch caller
        /// has to ask for the batch reading explicitly.
        rmsnormRows: *const fn (ctx: *anyopaque, out: Buf, x: Buf, w: Buf, rows: usize, eps: f32) void,
        /// `rows` consecutive `l2norm` calls on `x.span(r*n, n)`.  Same
        /// "explicit, never inferred" rule as `rmsnormRows`; each row's eps is
        /// inside its own sum, so the rows are fully independent.
        l2normRows: *const fn (ctx: *anyopaque, x: Buf, n: usize) void,
        /// Rotate the 64-dim rotary block of a row in place (cross-half pairing).
        rope64: *const fn (ctx: *anyopaque, x: Buf, pos: u32, base: f32) void,
        /// `rows` consecutive `rope64` calls on `x.span(r * heads * hd, heads * hd)`
        /// with `pos + r`, i.e. row r is prefill token `pos + r`.  `heads` is the
        /// head count PER ROW, so `x.len == rows * heads * hd`.  Same
        /// "explicit, never inferred" rule as `rmsnormRows`.
        ///
        /// PREFILL ONLY in practice: the decode path must keep issuing single-row
        /// `rope64` calls because the decode CUDA graph identifies and retargets
        /// those nodes by host-stub address (see `q35_func_rope64`).  Prefill
        /// never captures a graph, so the batch reading is free there.
        rope64Rows: *const fn (ctx: *anyopaque, x: Buf, pos: u32, base: f32, rows: usize, heads: usize, hd: usize) void,
        /// x *= rsqrt(sum(x^2) + 1e-6)
        l2norm: *const fn (ctx: *anyopaque, x: Buf) void,
        /// x *= s
        scaleVec: *const fn (ctx: *anyopaque, x: Buf, s: f32) void,
        /// x = silu(x) in place
        siluInPlace: *const fn (ctx: *anyopaque, x: Buf) void,
        /// out[i] *= silu(gate[i])
        mulSilu: *const fn (ctx: *anyopaque, out: Buf, gate: Buf) void,
        /// out[i] *= sigmoid(gate[i])
        mulSigmoid: *const fn (ctx: *anyopaque, out: Buf, gate: Buf) void,
        /// softmax over the whole slice
        softmax: *const fn (ctx: *anyopaque, x: Buf) void,
        /// dst[i] += src[i]
        addResid: *const fn (ctx: *anyopaque, dst: Buf, src: Buf) void,
        /// SwiGLU: gate[i] = silu(gate[i]) * up[i], in place
        swiglu: *const fn (ctx: *anyopaque, gate: Buf, up: Buf) void,
        /// Gated Delta Net: one token of the causal conv (history `conv` is
        /// `y.len * 3` floats, oldest..newest) + silu, in place on `y`.
        convStep: *const fn (ctx: *anyopaque, conv: Buf, y: Buf, wconv: Buf, k: usize) void,
        /// Gated Delta Net: one token of the delta-rule state update.
        /// state is [nv, kd, vd] (kd == vd == 128), `k`/`q` are kd wide,
        /// `v`/`km`/`delta`/`out` are vd wide, per head.
        gdnStep: *const fn (ctx: *anyopaque, state: Buf, k: Buf, v: Buf, q: Buf, km: Buf, delta: Buf, out: Buf, gamma: f32, beta: f32, vd: usize) void,
        /// Batched `gdnStep`: the SAME per-head call, `heads` times, in one
        /// launch instead of `heads` launches (48 of them per token per layer on
        /// this model).  Semantics are defined as "for h in 0..heads-1: the
        /// `gdnStep` call that `h` would have made", so the CPU implementation
        /// here is literally that loop.
        ///
        /// Rows are *strided bases*, not a packed array: the engine's regions
        /// interleave (a v-head run of `vd` floats inside `qkv`, a
        /// `kd*vd` state block per head), so each operand gets an explicit
        /// element stride.  `heads`, `nk` and `kd`/`vd` are passed for the same
        /// "explicit, never inferred" reason as `rmsnormRows`.
        ///
        ///   state[*h*st_state .. +kd*vd]   the [kd, vd] matrix for head h
        ///   k     [kh*kd .. +kd],  kh = h % nk      (kv-head mapping: a v head
        ///   q     [kh*kd .. +kd]    reuses k head h%nk, and `q` is implicitly
        ///                                            scaled by 1/sqrt(kd))
        ///   v     [h*vd  .. +vd]
        ///   out   [h*vd  .. +vd]
        ///   km    [h*vd  .. +vd]   PER-HEAD scratch, `heads*vd` long
        ///   delta [h*vd  .. +vd]   PER-HEAD scratch, `heads*vd` long
        ///   gamma [h], beta [h]    per-head scalars, in backend regions
        ///
        /// `km`/`delta` must be per-head (stride vd): with a single shared `vd`
        /// buffer the `heads` blocks would race, and the wrong answer would be
        /// silent.  The caller sizes them.
        ///
        /// `gamma`/`beta` are `Buf`s, i.e. backend regions (device memory on a
        /// device backend) of at least `heads` floats -- `gdnGamma`'s output
        /// and its in-place sigmoid(b) write-back respectively.  Passing them
        /// as regions is what keeps a token from making `2 * heads` floats of
        /// PCIe round trip per GDN layer (read back for an `f32` argument,
        /// then re-uploaded): a device backend just reads its own memory.
        ///
        /// Their CONTENTS change every token (per-layer, per-position), so they
        /// must NEVER be routed through a `devConst`-style cache: those key on
        /// pointer identity and never re-upload, which would silently serve the
        /// first layer's coefficients forever.
        gdnStepRows: *const fn (ctx: *anyopaque, state: Buf, k: Buf, v: Buf, q: Buf, km: Buf, delta: Buf, out: Buf,
            gamma: Buf, beta: Buf, heads: usize, nk: usize, kd: usize, vd: usize) void,

        /// Full-attention decode for one token: scores against the KV rows,
        /// softmax, PV accumulation into `gout`, then the sigmoid gate.  `kcache`
        /// / `vcache` are this layer's KV rows (`kv_dim` wide per token,
        /// `slot+1` tokens already written).
        attnDecode: *const fn (ctx: *anyopaque, q: Buf, gate: Buf, kcache: Buf, vcache: Buf,
            att: Buf, gout: Buf, slot: usize, heads: usize, gqa: usize, hd: usize,
            kv_dim: usize, scale: f32) void,
        /// GDN per-token beta/gamma from the raw a/b projections.
        gdnGamma: *const fn (ctx: *anyopaque, a_raw: Buf, b_raw: Buf, ssm_a: Buf, dt_bias: Buf, gamma: Buf, nv: usize) void,

        /// GDN chunk prelude: the three per-token GDN steps that do NOT depend on
        /// the running state, done for a whole prefill chunk in a handful of
        /// launches instead of per token.
        ///
        /// `forwardGdnCore` needs `tokens` x {convStep, gdnGamma, l2norm(x 2
        /// runs x nk_reps), scaleVec(x nk_reps)} before the first `gdnStepRows`.
        /// None of those read or write `state`, and `gdnStepRows` only *reads*
        /// qkv -- it writes state/km/delta/out -- so they can all run up front.
        /// The one genuine coupling is the conv history: token t's taps read what
        /// token t-1 shifted in, so a backend MUST process the tokens in ascending
        /// order (the CUDA kernel does, inside one thread per channel).
        ///
        /// Semantics are defined as exactly the CPU bodies below, so a backend
        /// is free to fuse/launch as it likes as long as the numbers match.
        /// Every region is `tokens` rows of `row_stride` floats:
        ///   qkv   [t*row_stride .. +conv_dim]   conv + silu applied in place
        ///   ab    [t*a_stride .. +nv]           a, and [t*b_stride .. +nv] b
        ///   gamma [t*gamma_stride .. +nv]       written by gamma, shared by tokens
        ///   l2q   [t*row_stride + row_offset .. ] the q run, normalised nk_reps
        ///   l2k   [t*row_stride + kq_offset ..]  the k run, same
        /// `nk_reps` is `nv / nk` and is applied to each run in exactly the order
        /// the per-head loop did (q then k, then the q scale, repeated).
        gdnPrefillCore: *const fn (ctx: *anyopaque, qkv: Buf, conv: Buf, wconv: Buf,
            ab: Buf, ssm_a: Buf, dt_bias: Buf, gamma: Buf,
            tokens: usize, conv_dim: usize, ab_a_offset: usize, ab_b_offset: usize,
            ab_stride: usize, gamma_stride: usize,
            kq_dim: usize, nv: usize, nk: usize, kd: usize, conv_k: usize, eps: f32) void,
        /// Region-to-region copy (KV writes, MTP state snapshots): a device
        /// backend maps this to cudaMemcpy, so a snapshot never touches host.
        copyRange: *const fn (ctx: *anyopaque, dst: Buf, src: Buf) void,
        /// `copyRange` whose destination advances once per token: the KV cache
        /// append, `dst = cache.k/v + slot * kv_dim`.
        ///
        /// Identical to `copyRange` for every backend except one that replays a
        /// captured schedule.  Such a backend snapshots a launch's arguments,
        /// and a destination that moves per token would then be re-recorded to
        /// the capture token's row forever -- the cache never grows and each
        /// token attends to the prefix it saw the first time.  Wrong numbers, no
        /// error, so the slot has to be declared rather than inferred: a bare
        /// device pointer carries no trace of which row of which cache it is.
        copyRangeSlot: *const fn (ctx: *anyopaque, dst: Buf, src: Buf, slot: usize) void,
        /// Split the interleaved `attn_q` projection into `q` and `gate`.
        ///
        /// The GGUF `attn_q` rows are ordered `[h][q_head | gate_head]` -- one
        /// `hd`-wide q block then one `hd`-wide gate block per head -- because
        /// upstream packs them that way; nothing in the engine wants that
        /// layout, both `q` and `gate` are consumed per head.  Doing it as
        /// `2*heads` `copyRange` calls costs one launch per 1 KB on a device
        /// backend (`PROF-COPY`: 768 calls/token, 99% of the `copy_range`
        /// slot), so it gets its own slot and moves the whole `2*q_dim` in one
        /// go.
        ///
        /// `dst_q.len == dst_gate.len == heads*hd`, `src.len == 2*heads*hd`.
        deinterleaveQG: *const fn (ctx: *anyopaque, dst_q: Buf, dst_gate: Buf, src: Buf, heads: usize, hd: usize) void,
        /// Copy a host tensor row into a backend region (the f32 embedding row
        /// when token_embd is unquantized).
        copyFromHost: *const fn (ctx: *anyopaque, dst: Buf, src: []const f32) void,
        /// Copy a small backend region back into a host slice.
        ///
        /// NOT on the per-token path any more: its one historical user was the
        /// per-layer `nv` scalars (`beta`/`gamma`) that `gdnGamma` produces on
        /// the backend, which `gdnStepRows` now takes as `Buf`s and reads on
        /// the device.  It survives for diagnostics and for callers that
        /// genuinely need a host copy of a small region (tests, `QWEN35_DUMP_*`
        /// style tooling, MTP debug paths).  Never use it to bring activations
        /// back -- that is what P2c removed, and a device->host copy also
        /// serialises the stream.
        copyToHost: *const fn (ctx: *anyopaque, dst: []f32, src: Buf) void,
        /// Zero a region (reset of the conv history / GDN state).
        zeroBuf: *const fn (ctx: *anyopaque, buf: Buf) void,
        /// Greedy argmax over logits.  On a device backend this runs on the
        /// device and the token id is the only thing that comes back.
        argmax: *const fn (ctx: *anyopaque, logits: Buf) u32,
        /// Sampling.  The policy/PRNG object stays in the engine (set from the
        /// CLI); the backend runs the algorithm wherever the logits live, so a
        /// device backend samples on the device and returns a 4-byte token id.
        /// Passing the sampler by pointer keeps its PRNG state advancing
        /// identically on both paths.
        sample: *const fn (ctx: *anyopaque, logits: Buf, scratch: Buf, sp: *sampler.Sampler) u32,
        /// Return the token that the LAST `sample` on this step produced.
        ///
        /// WHY THIS EXISTS: a device backend that captures its decode step into a
        /// CUDA graph needs the sampler to be INSIDE the graph (llama.cpp does
        /// the same: the sampler is a node of the same `ggml_cgraph`), otherwise
        /// the graph's tail is a serialised launch + full-vocabulary D2H at the
        /// one moment the device has nothing to do.  But the token id is the next
        /// token, so it must reach the host before the next step can exist -- and
        /// that download cannot happen inside a capture.
        ///
        /// So the split is: `sample` LAUNCHES the sampler (inside the span, so it
        /// is captured); `stepEnd` closes the graph and performs the download; and
        /// this method hands the value over afterwards.  Its argument is the value
        /// `sample` returned (`sample` must return a BEST GUESS that this call is
        /// free to override), which lets a CPU backend -- where the sampler
        /// already produced the answer synchronously -- be the identity.
        sampledToken: *const fn (ctx: *anyopaque, hinted: u32) u32,


        /// Mark a phase boundary in the pool's timing counters (no-op for a
        /// device backend, which reports its own timings).
        profPhase: *const fn (ctx: *anyopaque, on: bool) void,
        profReset: *const fn (ctx: *anyopaque) void,
        /// End of the prefill loop.  A CPU backend ignores it (its pool
        /// profiler already separates the phases); the CUDA backend uses it to
        /// close its GPU-side prefill window, because `profPhase(false)` only
        /// fires after the decode loop and would fold decode into prefill.
        profPrefillEnd: *const fn (ctx: *anyopaque) void,

        // ---- decode-step boundary -------------------------------------------
        //
        // WHAT THIS IS FOR: a CUDA graph can only wrap a whole sequence of
        // launches that is known in advance, and the only such sequence in this
        // engine is one decode step's forward pass (weights resident, shapes
        // fixed, `pos`/`slot` the sole variables).  The engine therefore tells
        // the backend where that sequence starts and ends; the backend decides
        // whether to capture it, replay it, or ignore the hint entirely.
        //
        // WHY THE HINT IS NEEDED AT ALL: the backend never sees the model's
        // control flow -- it is called one op at a time -- so without this it
        // could only guess where a step begins.  Guessing wrong is what a CUDA
        // graph punishes hardest (a graph spanning two steps replays stale
        // state, a graph spanning half a step omits ops: both give wrong
        // numbers with no error).
        //
        // `stepBegin` carries `pos`, the step's position, which is the value
        // that (a) identifies a capturable step and (b) changes every token.
        // `stepEnd` closes the span.  Both must be no-ops for a CPU backend,
        // and a device backend must be free to ignore them.
        stepBegin: *const fn (ctx: *anyopaque, pos: u32) void,
        stepEnd: *const fn (ctx: *anyopaque) void,

        // ---- introspection --------------------------------------------------
        name: *const fn (ctx: *anyopaque) []const u8,
        /// Human-readable one-liner for the tier actually in use.
        describe: *const fn (ctx: *anyopaque) []const u8,
        /// Restrict implementation, e.g. "baseline"/"avx2"/"avx512".
        setTier: *const fn (ctx: *anyopaque, tier: []const u8) void,

        // ---- profiling ------------------------------------------------------
        /// Matvec-domain counters, printed at shutdown when profiling is on.
        reportProfile: *const fn (ctx: *anyopaque) void,
        /// One-shot *complete* profile dump, called by `Engine.profBackendReport`
        /// at a caller-chosen point (the CLI calls it right after `generate`
        /// returns, gated on `$QWEN35_PROF`).
        ///
        /// Separate from `reportProfile` because that one is the host pool's
        /// table and is also what a device backend's `deinit` would print; the
        /// engine is arena-allocated, so `deinit` never runs on the CLI path and
        /// a device backend's per-type / `PROF-SLOT` / `PROF-COPY` tables would
        /// otherwise only appear at an internal trigger point (`% 500` matvecs,
        /// or the closing of the prefill window) -- which a long EOS-less decode
        /// never reaches, leaving the run with a single header line.
        profReport: *const fn (ctx: *anyopaque) void,
    };

    // Convenience forwarding.  Kept trivial on purpose: these are called on
    // the hot path, so each one is a single indirect call.

    pub fn loadLinear(self: Exec, tensor_name: []const u8) LoadError!LinearW {
        return self.vtable.loadLinear(self.ptr, tensor_name);
    }
    pub fn freeLinear(self: Exec, w: LinearW) void {
        self.vtable.freeLinear(self.ptr, w);
    }
    pub fn loadVecF32(self: Exec, tensor_name: []const u8) LoadError!Buf {
        return self.vtable.loadVecF32(self.ptr, tensor_name);
    }
    pub fn freeVec(self: Exec, v: Buf) void {
        self.vtable.freeVec(self.ptr, v);
    }
    pub fn allocScratch(self: Exec, max_cols: usize) anyerror!void {
        return self.vtable.allocScratch(self.ptr, max_cols);
    }
    pub fn allocScratchK(self: Exec, max_cols: usize, k: usize) anyerror!void {
        return self.vtable.allocScratchK(self.ptr, max_cols, k);
    }
    pub fn freeScratch(self: Exec) void {
        self.vtable.freeScratch(self.ptr);
    }
    pub fn allocBuf(self: Exec, len: usize) anyerror!Buf {
        return self.vtable.allocBuf(self.ptr, len);
    }
    pub fn dequantRow(self: Exec, dst: Buf, w: QuantRow, row: usize) void {
        self.vtable.dequantRow(self.ptr, dst, w, row);
    }
    pub fn matvec(self: Exec, out: Buf, w: LinearW, x: Buf) void {
        self.vtable.matvec(self.ptr, out, w, x);
    }
    pub fn matvecBatchK(self: Exec, out: Buf, w: LinearW, xK: Buf, k: usize) void {
        self.vtable.matvecBatchK(self.ptr, out, w, xK, k);
    }
    pub fn rmsnorm(self: Exec, out: Buf, x: Buf, w: Buf, eps: f32) void {
        self.vtable.rmsnorm(self.ptr, out, x, w, eps);
    }
    /// See `VTable.rmsnormRows`: `out` / `x` are `rows * w.len` long, `w` is one
    /// row.  Equivalent to `rows` calls of `rmsnorm` on consecutive sub-rows.
    pub fn rmsnormRows(self: Exec, out: Buf, x: Buf, w: Buf, rows: usize, eps: f32) void {
        self.vtable.rmsnormRows(self.ptr, out, x, w, rows, eps);
    }
    pub fn rope64(self: Exec, x: Buf, pos: u32, base: f32) void {
        self.vtable.rope64(self.ptr, x, pos, base);
    }
    /// See `VTable.rope64Rows`: `x` is `rows * heads * hd` long, row r (prefill
    /// token `pos + r`) is `heads * hd` floats.
    pub fn rope64Rows(self: Exec, x: Buf, pos: u32, base: f32, rows: usize, heads: usize, hd: usize) void {
        self.vtable.rope64Rows(self.ptr, x, pos, base, rows, heads, hd);
    }
    pub fn l2norm(self: Exec, x: Buf) void {
        self.vtable.l2norm(self.ptr, x);
    }
    /// See `VTable.l2normRows`: `x` is `rows * n` long.  Equivalent to `rows`
    /// calls of `l2norm` on `x.span(r*n, n)`.
    pub fn l2normRows(self: Exec, x: Buf, n: usize) void {
        self.vtable.l2normRows(self.ptr, x, n);
    }
    pub fn scaleVec(self: Exec, x: Buf, s: f32) void {
        self.vtable.scaleVec(self.ptr, x, s);
    }
    pub fn siluInPlace(self: Exec, x: Buf) void {
        self.vtable.siluInPlace(self.ptr, x);
    }
    pub fn mulSilu(self: Exec, out: Buf, gate: Buf) void {
        self.vtable.mulSilu(self.ptr, out, gate);
    }
    pub fn mulSigmoid(self: Exec, out: Buf, gate: Buf) void {
        self.vtable.mulSigmoid(self.ptr, out, gate);
    }
    pub fn softmax(self: Exec, x: Buf) void {
        self.vtable.softmax(self.ptr, x);
    }
    pub fn addResid(self: Exec, dst: Buf, src: Buf) void {
        self.vtable.addResid(self.ptr, dst, src);
    }
    pub fn swiglu(self: Exec, gate: Buf, up: Buf) void {
        self.vtable.swiglu(self.ptr, gate, up);
    }
    pub fn convStep(self: Exec, conv: Buf, y: Buf, wconv: Buf, k: usize) void {
        self.vtable.convStep(self.ptr, conv, y, wconv, k);
    }
    pub fn gdnStep(self: Exec, state: Buf, k: Buf, v: Buf, q: Buf, km: Buf, delta: Buf, out: Buf, gamma: f32, beta: f32, vd: usize) void {
        self.vtable.gdnStep(self.ptr, state, k, v, q, km, delta, out, gamma, beta, vd);
    }
    /// Batched `gdnStep` -- see `VTable.gdnStepRows` for the per-head strides.
    /// Byte-for-byte the `heads` single calls it replaces; the engine uses this
    /// one (the single-row slot is kept for fallback and for A/B).
    pub fn gdnStepRows(self: Exec, state: Buf, k: Buf, v: Buf, q: Buf, km: Buf, delta: Buf, out: Buf,
                       gamma: Buf, beta: Buf, heads: usize, nk: usize, kd: usize, vd: usize) void {
        self.vtable.gdnStepRows(self.ptr, state, k, v, q, km, delta, out, gamma, beta, heads, nk, kd, vd);
    }
    pub fn attnDecode(self: Exec, q: Buf, gate: Buf, kcache: Buf, vcache: Buf, att: Buf, gout: Buf,
                      slot: usize, heads: usize, gqa: usize, hd: usize, kv_dim: usize, scale: f32) void {
        self.vtable.attnDecode(self.ptr, q, gate, kcache, vcache, att, gout, slot, heads, gqa, hd, kv_dim, scale);
    }
    /// beta = sigmoid(b_raw) written IN PLACE into `b_raw`, and
    /// gamma[h] = exp(ssm_a[h] * softplus(a_raw[h] + dt_bias[h])).
    /// `a_raw` and `b_raw` are separate buffers because the batched prefill
    /// writes alpha as `abK[0..k*nv]` and beta as `abK[k*nv..]` -- two
    /// non-contiguous regions -- while the serial path passes the two halves
    /// of one row.
    pub fn gdnGamma(self: Exec, a_raw: Buf, b_raw: Buf, ssm_a: Buf, dt_bias: Buf, gamma: Buf, nv: usize) void {
        self.vtable.gdnGamma(self.ptr, a_raw, b_raw, ssm_a, dt_bias, gamma, nv);
    }

    /// GDN chunk prelude -- see `VTable.gdnPrefillCore` for the contract.
    pub fn gdnPrefillCore(self: Exec, qkv: Buf, conv: Buf, wconv: Buf, ab: Buf,
        ssm_a: Buf, dt_bias: Buf, gamma: Buf, tokens: usize, conv_dim: usize,
        ab_a_offset: usize, ab_b_offset: usize, ab_stride: usize,
        gamma_stride: usize, kq_dim: usize, nv: usize, nk: usize, kd: usize,
        conv_k: usize, eps: f32) void {
        self.vtable.gdnPrefillCore(self.ptr, qkv, conv, wconv, ab, ssm_a, dt_bias, gamma, tokens, conv_dim, ab_a_offset, ab_b_offset, ab_stride, gamma_stride, kq_dim, nv, nk, kd, conv_k, eps);
    }
    pub fn copyRange(self: Exec, dst: Buf, src: Buf) void {
        self.vtable.copyRange(self.ptr, dst, src);
    }

    /// KV-cache append: `copyRange` whose destination is `base + slot * stride`.
    /// See `VTable.copyRangeSlot` for why a replaying backend needs the slot
    /// spelled out.
    pub fn copyRangeSlot(self: Exec, dst: Buf, src: Buf, slot: usize) void {
        self.vtable.copyRangeSlot(self.ptr, dst, src, slot);
    }
    pub fn deinterleaveQG(self: Exec, dst_q: Buf, dst_gate: Buf, src: Buf, heads: usize, hd: usize) void {
        self.vtable.deinterleaveQG(self.ptr, dst_q, dst_gate, src, heads, hd);
    }
    pub fn copyFromHost(self: Exec, dst: Buf, src: []const f32) void {
        self.vtable.copyFromHost(self.ptr, dst, src);
    }
    pub fn copyToHost(self: Exec, dst: []f32, src: Buf) void {
        self.vtable.copyToHost(self.ptr, dst, src);
    }
    pub fn zeroBuf(self: Exec, buf: Buf) void {
        self.vtable.zeroBuf(self.ptr, buf);
    }
    pub fn argmax(self: Exec, logits: Buf) u32 {
        return self.vtable.argmax(self.ptr, logits);
    }
    pub fn sample(self: Exec, logits: Buf, scratch: Buf, sp: *sampler.Sampler) u32 {
        return self.vtable.sample(self.ptr, logits, scratch, sp);
    }
    /// The token the last `sample` of the current step produced; see the vtable
    /// slot for why sampling and its download are split around `stepEnd`.
    pub fn sampledToken(self: Exec, hinted: u32) u32 {
        return self.vtable.sampledToken(self.ptr, hinted);
    }
    pub fn profPhase(self: Exec, on: bool) void {
        self.vtable.profPhase(self.ptr, on);
    }
    pub fn profReset(self: Exec) void {
        self.vtable.profReset(self.ptr);
    }
    pub fn profPrefillEnd(self: Exec) void {
        self.vtable.profPrefillEnd(self.ptr);
    }
    /// Open / close one decode step.  See the vtable comments: this is a
    /// boundary hint the backend may ignore, and the CPU backend does.
    pub fn stepBegin(self: Exec, pos: u32) void {
        self.vtable.stepBegin(self.ptr, pos);
    }
    pub fn stepEnd(self: Exec) void {
        self.vtable.stepEnd(self.ptr);
    }
    pub fn setThreads(self: Exec, io: std.Io, nt: usize) void {
        self.vtable.setThreads(self.ptr, io, nt);
    }
    pub fn setName(self: Exec, tier: []const u8) void {
        self.vtable.setTier(self.ptr, tier);
    }
    pub fn name(self: Exec) []const u8 {
        return self.vtable.name(self.ptr);
    }
    pub fn describe(self: Exec) []const u8 {
        return self.vtable.describe(self.ptr);
    }
    pub fn reportProfile(self: Exec) void {
        self.vtable.reportProfile(self.ptr);
    }
    /// Print the complete profile report once.  Callers gate this on their own
    /// debug switch; the vtable slot itself is unconditional.
    pub fn profReport(self: Exec) void {
        self.vtable.profReport(self.ptr);
    }
};

/// Which backend a model run uses.
///
/// `.cuda` is wired up but currently a *trampoline*: `backend_cuda.CudaBackend`
/// embeds a CPU backend and hands out its `Exec`, so `--backend cuda` exercises
/// the selection path end-to-end with byte-identical numbers while the device
/// implementation does not exist yet.  See the header of `backend_cuda.zig`.
pub const Kind = enum { cpu, cuda };

pub fn selectBackend(kind: Kind) type {
    return switch (kind) {
        .cpu => @import("backend_cpu.zig").CpuBackend,
        .cuda => @import("backend_cuda.zig").CudaBackend,
    };
}
