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
//! What is deliberately NOT abstracted: element-wise float maths (rmsNorm,
//! silu, rope, softmax).  Those are a dozen auto-vectorized loops that Zig
//! compiles to the same code on every x86 backend, so a vtable would only add
//! an indirect call per layer.  They stay in the engine as plain Zig.

const std = @import("std");
const gguf = @import("gguf.zig");

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

/// Errors every backend can report while loading weights.
pub const LoadError = error{
    MissingTensor,
    UnsupportedTensorType,
    OutOfMemory,
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

        // ---- dequantization -------------------------------------------------
        /// Expand one row of a packed weight into `dst` (f32, `cols` wide).
        /// Used by the embedding lookup, which needs values rather than dots.
        dequantRow: *const fn (ctx: *anyopaque, dst: []f32, w: QuantRow, row: usize) void,

        // ---- matvec ---------------------------------------------------------
        /// `out = W * x` with `W` packed, `x` f32 of length `w.cols()`.
        matvec: *const fn (ctx: *anyopaque, out: []f32, w: LinearW, x: []const f32) void,
        /// Same, for `k` tokens at once: `out[j*rows + r] = W[r] . x_j`.
        matvecBatchK: *const fn (ctx: *anyopaque, out: []f32, w: LinearW, xK: []const f32, k: usize) void,

        /// Create the intra-op thread pool (CPU); a device backend ignores it.
        /// `nt` is the worker count, 1 = serial.
        setThreads: *const fn (ctx: *anyopaque, io: std.Io, nt: usize) void,

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
    pub fn dequantRow(self: Exec, dst: []f32, w: QuantRow, row: usize) void {
        self.vtable.dequantRow(self.ptr, dst, w, row);
    }
    pub fn matvec(self: Exec, out: []f32, w: LinearW, x: []const f32) void {
        self.vtable.matvec(self.ptr, out, w, x);
    }
    pub fn matvecBatchK(self: Exec, out: []f32, w: LinearW, xK: []const f32, k: usize) void {
        self.vtable.matvecBatchK(self.ptr, out, w, xK, k);
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
