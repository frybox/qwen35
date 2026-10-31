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
/// quantization format (the normal case, zero-copy) or already expanded to
/// f32 (f32/f16/bf16 tensors).
pub const LinearW = union(enum) {
    quant: QuantRow,
    f32: []f32,

    pub fn deinit(self: LinearW, alloc: std.mem.Allocator) void {
        // `quant` points into the GGUF mapping; only the f32 copy is owned.
        if (self == .f32) alloc.free(self.f32);
    }
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
    /// the tensor and return the device region — see `constBuf`.
    pub fn wrap(s: []const f32) Buf {
        return .{ .ptr = @constCast(s.ptr), .len = s.len };
    }
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
        /// Rotate the 64-dim rotary block of a row in place (cross-half pairing).
        rope64: *const fn (ctx: *anyopaque, x: Buf, pos: u32, base: f32) void,
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

        /// Full-attention decode for one token: scores against the KV rows,
        /// softmax, PV accumulation into `gout`, then the sigmoid gate.  `kcache`
        /// / `vcache` are this layer's KV rows (`kv_dim` wide per token,
        /// `slot+1` tokens already written).
        attnDecode: *const fn (ctx: *anyopaque, q: Buf, gate: Buf, kcache: Buf, vcache: Buf,
            att: Buf, gout: Buf, slot: usize, heads: usize, gqa: usize, hd: usize,
            kv_dim: usize, scale: f32) void,
        /// GDN per-token beta/gamma from the raw a/b projections.
        gdnGamma: *const fn (ctx: *anyopaque, ab: Buf, a: Buf, dt_bias: Buf, gamma: Buf, nv: usize) void,
        /// Region-to-region copy (KV writes, MTP state snapshots): a device
        /// backend maps this to cudaMemcpy, so a snapshot never touches host.
        copyRange: *const fn (ctx: *anyopaque, dst: Buf, src: Buf) void,
        /// Copy a host tensor row into a backend region (the f32 embedding row
        /// when token_embd is unquantized).
        copyFromHost: *const fn (ctx: *anyopaque, dst: Buf, src: []const f32) void,
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

        /// Make a constant tensor (norm weight, conv taps, rope table, ssm_a,
        /// dt_bias) available as a `Buf` the elementwise slots can read.  A CPU
        /// backend wraps the host slice; a device backend uploads it once and
        /// returns the device region (cached, so this is called per tensor at
        /// load time, not per token).  Copying is the backend's business; the
        /// engine keeps only the returned handle.
        constBuf: *const fn (ctx: *anyopaque, s: []const f32) Buf,

        /// Mark a phase boundary in the pool's timing counters (no-op for a
        /// device backend, which reports its own timings).
        profPhase: *const fn (ctx: *anyopaque, on: bool) void,
        profReset: *const fn (ctx: *anyopaque) void,

        // ---- introspection --------------------------------------------------
        name: *const fn (ctx: *anyopaque) []const u8,
        /// Human-readable one-liner for the tier actually in use.
        describe: *const fn (ctx: *anyopaque) []const u8,
        /// Restrict implementation, e.g. "baseline"/"avx2"/"avx512".
        setTier: *const fn (ctx: *anyopaque, tier: []const u8) void,

        // ---- profiling ------------------------------------------------------
        /// Matvec-domain counters, printed at shutdown when profiling is on.
        reportProfile: *const fn (ctx: *anyopaque) void,
    };

    // Convenience forwarding.  Kept trivial on purpose: these are called on
    // the hot path, so each one is a single indirect call.

    pub fn loadLinear(self: Exec, tensor_name: []const u8) LoadError!LinearW {
        return self.vtable.loadLinear(self.ptr, tensor_name);
    }
    pub fn freeLinear(self: Exec, w: LinearW) void {
        self.vtable.freeLinear(self.ptr, w);
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
    pub fn rope64(self: Exec, x: Buf, pos: u32, base: f32) void {
        self.vtable.rope64(self.ptr, x, pos, base);
    }
    pub fn l2norm(self: Exec, x: Buf) void {
        self.vtable.l2norm(self.ptr, x);
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
    pub fn attnDecode(self: Exec, q: Buf, gate: Buf, kcache: Buf, vcache: Buf, att: Buf, gout: Buf,
                      slot: usize, heads: usize, gqa: usize, hd: usize, kv_dim: usize, scale: f32) void {
        self.vtable.attnDecode(self.ptr, q, gate, kcache, vcache, att, gout, slot, heads, gqa, hd, kv_dim, scale);
    }
    pub fn gdnGamma(self: Exec, ab: Buf, a: Buf, dt_bias: Buf, gamma: Buf, nv: usize) void {
        self.vtable.gdnGamma(self.ptr, ab, a, dt_bias, gamma, nv);
    }
    pub fn copyRange(self: Exec, dst: Buf, src: Buf) void {
        self.vtable.copyRange(self.ptr, dst, src);
    }
    pub fn copyFromHost(self: Exec, dst: Buf, src: []const f32) void {
        self.vtable.copyFromHost(self.ptr, dst, src);
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
    pub fn constBuf(self: Exec, s: []const f32) Buf {
        return self.vtable.constBuf(self.ptr, s);
    }
    pub fn profPhase(self: Exec, on: bool) void {
        self.vtable.profPhase(self.ptr, on);
    }
    pub fn profReset(self: Exec) void {
        self.vtable.profReset(self.ptr);
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
