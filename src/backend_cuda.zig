//! GPU backend: every numeric kernel in the forward pass runs on the device.
//!
//! Why this file exists at all (rather than a CUDA build of `backend_cpu`): the
//! same `zig-out/bin/qwen35` binary has to run on hosts with no CUDA install, so
//! libcudart and the kernel `.so` are dlopen'd at run time, not linked (see
//! `cudart.zig`).  `--backend cpu` is then still a single binary, not a second
//! build product.
//!
//! Division of labour with `backend_cpu` (deliberate, see AGENTS.md §1.1):
//!
//!   * The embedded `CpuBackend` stays authoritative for *metadata* work:
//!     resolving the GGUF tensor, building the `QuantRow` view, expanding
//!     f16/bf16/f32 tensors to f32, and the geometry it validates at load time.
//!     That is startup work over the mmap, not a forward-pass computation.
//!   * Every *numeric* operation the engine asks for -- matvec, K-token batched
//!     matvec, row dequantization, and the activation scratch those need -- is
//!     done on the device by the kernels in `src/kernels/cuda`.  There is no CPU
//!     arithmetic fallback here: mixing backends per tensor makes a divergence
//!     impossible to attribute and every profile meaningless.
//!
//! Weight residency: `loadLinear` uploads *all* `LinearW.quant` types that the
//! CPU loader accepts, plus the f32 expansion, and records the device pointer in
//! `dev_rows`.  A failed upload is a hard load error, never a silent CPU retreat
//! -- `--backend cuda` must mean what it says.
//!
//! Memory budget: the backend does not try to outsmart the driver.  A failed
//! `cudaMalloc` is reported with the tensor name, its type, the byte count and
//! the device's remaining free memory (`cudaMemGetInfo`), then turned into
//! `error.OutOfMemory` from `loadLinear`, so the load aborts instead of
//! continuing on a mixed CPU/GPU path (slow *and* silently different in
//! attribution).  A model that does not fit needs `--backend cpu` or a smaller
//! quant; `$QWEN35_CUDA_MAX_DEV_MIB` caps uploads so the failure lands early and
//! is attributable to the budget rather than to a transient allocation.

const std = @import("std");
const backend = @import("backend.zig");
const cpu_mod = @import("backend_cpu.zig");
const gguf = @import("gguf.zig");
const cudart = @import("cudart.zig");

const mem = std.mem;
const Io = std.Io;

/// `zig-out/lib/libqwen35-cuda-kernels.so` unless `$QWEN35_CUDA_KERNELS` says
/// otherwise.
const default_kernel_so = "libqwen35-cuda-kernels.so";

// ==== activation geometry ====================================================

/// Activation block size (in *elements*) a matvec against `ttype` needs.
///
/// Two formats, not one, because a quantized matvec is table-driven per weight
/// type and the activation block must match the weight block:
///   * Q8_0 weights use 32-element blocks -> Q8_0 activations (34 B / 32).
///   * Every other packed type this backend loads uses 256-element blocks
///     (Q4_K/Q6_K/IQ*/... all consume Q8_K) -> Q8_K (292 B / 256).
///
/// Keyed on `gguf.blockElems` rather than a type list so a new 32-element type
/// cannot silently get the 256-element activation.  Note `rowBlockElems` is NOT
/// the right discriminator: it is 256 for q4_0/iq4_nl too (AGENTS.md §1.1).
fn actBlockElems(ttype: gguf.TensorType) usize {
    return if (gguf.blockElems(ttype) == gguf.blockElems(.q8_0))
        gguf.blockElems(.q8_0)
    else
        gguf.blockElems(.q8_k);
}

/// Packed bytes of one activation block for `ttype`'s matvec input.
fn actBlockBytes(ttype: gguf.TensorType) usize {
    return if (gguf.blockElems(ttype) == gguf.blockElems(.q8_0))
        gguf.blockBytes(.q8_0)
    else
        gguf.blockBytes(.q8_k);
}

/// Quantized-activation bytes for a `cols`-wide input: whole blocks, rounded up
/// exactly like `backend_cpu.q8ScratchBytes` / `q8kScratchBytes`, so a config the
/// CPU backend accepted cannot overflow the device buffer here.
fn actBytes(ttype: gguf.TensorType, cols: usize) usize {
    const be = actBlockElems(ttype);
    return (cols + be - 1) / be * actBlockBytes(ttype);
}

fn oomMsg(what: []const u8) void {
    std.debug.print("cuda backend: {s} device buffer allocation failed\n", .{what});
}

/// Nanoseconds between a timestamp and now.  `durationTo(...).raw.nanoseconds` is
/// `i96`; the counters are `u64` and only ever accumulate (monotonic clock), so
/// clamp rather than trap.
fn durNs(t0: Io.Clock.Timestamp, io: Io) u64 {
    const d = t0.durationTo(Io.Clock.Timestamp.now(io, .awake)).raw.nanoseconds;
    return if (d > 0) @intCast(@min(d, std.math.maxInt(u64))) else 0;
}

/// One device copy of a host weight.  `ttype` is null for the f32 expansion
/// (which has no quant type); `rows`/`cols` are stored because an f32 slice
/// carries no geometry of its own and both matvec paths need the launch shape.
const DevRow = struct {
    host: [*]const u8,
    dev: *anyopaque,
    bytes: usize,
    ttype: ?gguf.TensorType,
    rows: usize,
    cols: usize,
};

const Geometry = struct { rows: usize, cols: usize };

/// Resolved kernel entry points.  All of them come from the kernel `.so` (never
/// from cudart), so this struct is the single place to audit against
/// `src/kernels/cuda/api.h`.
///
/// ABI, as landed in `src/kernels/cuda/api.h`:
///   ttype   : `@intFromEnum(gguf.TensorType)` -- GGML `u32` tag.  api.h's
///             `Q35_TT_*` values are copied from that enum, so the numbers agree
///             without a translation table (they are even in the same order).
///   nblocks : count of *type-native* blocks (cols / blockElems), not rows, and
///             not elements -- 32 for the q8_0/q4_*/q5_*/iq4_nl family, 256 for
///             the K/IQ family.
///   stream  : `cudaStream_t` as an opaque pointer; null = default stream.
/// Everything returns `cudaError_t` as an int (0 == success).
pub const Kernels = struct {
    quantize_q8_0: *const fn (?[*]const f32, ?*anyopaque, c_int, ?*anyopaque) callconv(.c) c_int,
    quantize_q8_k: *const fn (?[*]const f32, ?*anyopaque, c_int, ?*anyopaque) callconv(.c) c_int,
    /// Whole-tensor dequantization; `nblocks` is in native blocks.
    dequantize: *const fn (c_int, ?*const anyopaque, ?[*]f32, c_int, ?*anyopaque) callconv(.c) c_int,
    /// One row of `cols` elements.  The row offset is device-side pointer
    /// arithmetic by the caller (see `vDequantRow`), so only `cols` is needed.
    dequantize_row: *const fn (c_int, ?*const anyopaque, ?[*]f32, c_int, ?*anyopaque) callconv(.c) c_int,
    /// Table-driven quantized matvec: `d_out[t*rows + r] = W[r] . act_t`.
    /// Note the argument order from api.h: rows, cols, tokens, **then** ttype.
    mul_mat_vec: *const fn (?*const anyopaque, ?*const anyopaque, ?[*]f32, c_int, c_int, c_int, c_int, ?*anyopaque) callconv(.c) c_int,
    /// Same shape for an f32 weight matrix (embedding / norms).
    mul_mat_vec_f32: *const fn (?[*]const f32, ?[*]const f32, ?[*]f32, c_int, c_int, c_int, ?*anyopaque) callconv(.c) c_int,
};

// Buffers for formatted error/description strings.  Static because
// `describe`/`name` return slices with no allocator in scope (same pattern as
// the CPU backend, which reports through `std.debug.print`).
var msg_buf: [512]u8 = undefined;
var kernel_path_buf: [std.fs.max_path_bytes]u8 = undefined;
var exe_dir_buf: [std.fs.max_path_bytes]u8 = undefined;
pub const CudaBackend = struct {
    alloc: std.mem.Allocator,
    /// Metadata/geometry authority (see the file comment).  Never used for
    /// arithmetic -- the one exception is the opt-in `$QWEN35_MVCHECK`
    /// cross-check in `mvCheck`, which is diagnostics and cannot affect output.
    cpu: cpu_mod.CpuBackend,
    /// The GGUF being served.  Needed only by `vLoadLinear` to recover the
    /// rows/cols of an f16/bf16/f32 weight: `LinearW.f32` is a flat slice with no
    /// geometry, and the device launches need `cols`.
    g: *const gguf.File,
    cu: ?cudart.Open = null,
    /// Why device init failed; re-reported from `allocScratch` so `Engine.load`
    /// turns it into a load error instead of silently running on the CPU.
    init_error: []const u8 = "cuda backend: not initialized",

    kernel_lib: ?std.DynLib = null,
    kernel_fns: ?Kernels = null,

    /// Host data pointer -> device copy.  Keyed by `q.data.ptr` (mmap view) or the
    /// arena slice pointer (f32), so a `LinearW` handed back to us resolves
    /// without threading a cookie through the `Exec` interface.
    dev_rows: std.ArrayList(DevRow) = .empty,
    /// Sum of successful uploads: the resident weight footprint.
    stat_uploaded_bytes: usize = 0,
    /// Device time per slot, so the `CUDA-M` line answers "is the GPU actually
    /// doing the work, and where does it go".
    stat_ns_matvec: u64 = 0,
    stat_ns_batchk: u64 = 0,
    stat_ns_dequant: u64 = 0,
    stat_n_matvec: usize = 0,
    stat_n_batchk: usize = 0,
    stat_n_dequant: usize = 0,
    /// Per-quant-type matvec counters, indexed by `@intFromEnum(TensorType)`.
    /// `TensorType` has holes, so index by the raw tag and size for the largest
    /// value we count (tq2_0 = 35).
    stat_by_type: [64]usize = @splat(0),
    /// Which types the `$QWEN35_MVCHECK` cross-check has already covered (see
    /// `mvCheck`).  Per type, not per process: a model has one type per tensor
    /// family, so a single sample would leave e.g. iq4_xs unchecked.
    mvcheck_seen: [64]bool = @splat(false),
    /// Clock for the slot timings.  Supplied by `setThreads`, which the engine
    /// calls before the first matvec (same contract as `backend_cpu.prof_io`);
    /// only read when `$QWEN35_PROF` output is wanted.
    io: Io = undefined,

    // ---- reusable device buffers -------------------------------------------
    //
    // One set, grown on demand, never shrunk: matvec is the decode hot path and a
    // `cudaMalloc` there costs far more than the kernel.  `ensureBuf` is the only
    // allocation site reachable from matvec/matvecBatchK/dequantRow.
    /// f32 activations (one token, or k*cols for a batch).
    d_act: ?*anyopaque = null,
    d_act_bytes: usize = 0,
    /// Quantized activations (Q8_0 or Q8_K, whichever the weight type needs).
    d_xq: ?*anyopaque = null,
    d_xq_bytes: usize = 0,
    /// f32 matvec output (rows, or k*rows).
    d_out: ?*anyopaque = null,
    d_out_bytes: usize = 0,
    /// f32 row-dequantization target (cols floats).
    d_deq: ?*anyopaque = null,
    d_deq_bytes: usize = 0,

    // ==== construction ======================================================

    /// The engine's factory contract (`engine.backendImplFor` calls
    /// `BackendImpl.init(alloc, g)`).  Returns a value rather than an error on
    /// purpose: `init` has no way to report one, so a host without a usable GPU
    /// gets a backend whose `allocScratch` fails with a named error -- that is
    /// where `Engine.load` can turn it into a load failure.
    pub fn init(alloc: std.mem.Allocator, g: *const gguf.File) CudaBackend {
        var self = CudaBackend{
            .alloc = alloc,
            .cpu = cpu_mod.CpuBackend.init(alloc, g),
            .g = g,
        };

        // Ask cudart for the device before touching weights, so a host with no GPU
        // fails here (and `allocScratch` reports it) rather than after a partial
        // upload.
        var cand: cudart.Open = undefined;
        cudart.open(&cand, 3 << 30) catch |e| {
            self.init_error = std.fmt.bufPrint(
                &msg_buf,
                "cuda backend: device unavailable ({s}); use --backend cpu{s}{s}",
                .{
                    @errorName(e),
                    if (cand.missing().len > 0) "; missing cudart symbols: " else "",
                    cand.missing(),
                },
            ) catch "cuda backend: device unavailable";
            return self;
        };
        self.cu = cand;

        if (!self.loadKernels()) {
            // Keep the device open so `deinit` and the profile line still work;
            // `init_error` carries the reason for `allocScratch` to repeat.
            self.init_error = "cuda backend: kernel library unusable (see the error above)";
            return self;
        }
        return self;
    }

    pub fn exec(self: *CudaBackend) backend.Exec {
        return .{ .ptr = self, .vtable = &vtable };
    }

    pub fn deinit(self: *CudaBackend) void {
        self.reportStats("deinit");
        if (self.cu) |*cu| {
            for (self.dev_rows.items) |e| _ = cu.lib.Free(e.dev);
            if (self.d_act) |p| _ = cu.lib.Free(p);
            if (self.d_xq) |p| _ = cu.lib.Free(p);
            if (self.d_out) |p| _ = cu.lib.Free(p);
            if (self.d_deq) |p| _ = cu.lib.Free(p);
            self.d_act = null;
            self.d_xq = null;
            self.d_out = null;
            self.d_deq = null;
            cu.lib.close();
            self.cu = null;
        }
        self.dev_rows.deinit(self.alloc);
        if (self.kernel_lib) |*l| l.close();
        self.kernel_lib = null;
        self.kernel_fns = null;
        self.cpu.deinit();
    }

    // ==== device plumbing ===================================================

    fn self_(ctx: *anyopaque) *CudaBackend {
        return @ptrCast(@alignCast(ctx));
    }

    /// True when the runtime *and* every kernel symbol resolved.  False (e.g.
    /// `--backend cuda` on a host without a GPU) is a hard error reported from
    /// `allocScratch`, which is why no slot needs a CPU fallback for it.
    fn gpuReady(self: *CudaBackend) bool {
        return self.cu != null and self.kernel_fns != null;
    }

    fn needHelp(self: *CudaBackend) []const u8 {
        if (!self.gpuReady()) return self.init_error;
        return "cuda backend: kernel launch failed";
    }

    fn kernels(self: *CudaBackend) *const Kernels {
        return &self.kernel_fns.?;
    }

    fn loadKernels(self: *CudaBackend) bool {
        const path = kernelPath();
        var lib = std.DynLib.open(path) catch |e| {
            std.debug.print(
                "error: cuda backend: cannot open kernels library '{s}': {s}\n" ++
                    "       build it with `zig build -Dcuda=true` (or set $QWEN35_CUDA_KERNELS); use --backend cpu\n",
                .{ path, @errorName(e) },
            );
            return false;
        };

        const Q = *const fn (?[*]const f32, ?*anyopaque, c_int, ?*anyopaque) callconv(.c) c_int;
        const D = *const fn (c_int, ?*const anyopaque, ?[*]f32, c_int, ?*anyopaque) callconv(.c) c_int;
        const M = *const fn (?*const anyopaque, ?*const anyopaque, ?[*]f32, c_int, c_int, c_int, c_int, ?*anyopaque) callconv(.c) c_int;
        const F = *const fn (?[*]const f32, ?[*]const f32, ?[*]f32, c_int, c_int, c_int, ?*anyopaque) callconv(.c) c_int;

        // Every symbol is required.  A kernel `.so` from an older revision (e.g.
        // the Q8_0-only one) is rejected here with a named symbol, instead of
        // failing mid-run at the first non-Q8_0 tensor.
        const quantize_q8_0 = lib.lookup(Q, "q35_launch_quantize_q8_0") orelse
            return missingKernel(&lib, path, "q35_launch_quantize_q8_0");
        const quantize_q8_k = lib.lookup(Q, "q35_launch_quantize_q8_K") orelse
            return missingKernel(&lib, path, "q35_launch_quantize_q8_K");
        const dequant = lib.lookup(D, "q35_launch_dequantize") orelse
            return missingKernel(&lib, path, "q35_launch_dequantize");
        const dequant_row = lib.lookup(D, "q35_launch_dequantize_row") orelse
            return missingKernel(&lib, path, "q35_launch_dequantize_row");
        const mmvq = lib.lookup(M, "q35_launch_mul_mat_vec") orelse
            return missingKernel(&lib, path, "q35_launch_mul_mat_vec");
        const mmvf = lib.lookup(F, "q35_launch_mul_mat_vec_f32") orelse
            return missingKernel(&lib, path, "q35_launch_mul_mat_vec_f32");

        self.kernel_lib = lib;
        self.kernel_fns = .{
            .quantize_q8_0 = quantize_q8_0,
            .quantize_q8_k = quantize_q8_k,
            .dequantize = dequant,
            .dequantize_row = dequant_row,
            .mul_mat_vec = mmvq,
            .mul_mat_vec_f32 = mmvf,
        };
        return true;
    }

    fn missingKernel(lib: *std.DynLib, path: []const u8, name: []const u8) bool {
        std.debug.print(
            "error: cuda backend: '{s}' is missing symbol {s}\n" ++
                "       rebuild the kernels (zig build -Dcuda=true); use --backend cpu\n",
            .{ path, name },
        );
        lib.close();
        return false;
    }

    /// `<exe dir>/../lib/libqwen35-cuda-kernels.so` (i.e. `zig-out/lib/` next to
    /// `zig-out/bin/qwen35`), unless `$QWEN35_CUDA_KERNELS` says otherwise.
    fn kernelPath() []const u8 {
        if (cudart.getenv("QWEN35_CUDA_KERNELS")) |p| return p;
        if (exeDir()) |d| {
            return std.fmt.bufPrint(&kernel_path_buf, "{s}/../lib/{s}", .{ d, default_kernel_so }) catch default_kernel_so;
        }
        return default_kernel_so;
    }

    /// Directory holding the running executable, via /proc/self/exe.  Null when the
    /// symlink cannot be read; callers then fall back to a plain relative name,
    /// which works from `zig-out/bin`.
    fn exeDir() ?[]const u8 {
        const c = struct {
            extern fn readlink(path: [*:0]const u8, buf: [*]u8, len: usize) isize;
        };
        const n = c.readlink("/proc/self/exe", &exe_dir_buf, exe_dir_buf.len);
        if (n <= 0) return null;
        const full = exe_dir_buf[0..@intCast(n)];
        const slash = std.mem.lastIndexOfScalar(u8, full, '/') orelse return null;
        return full[0..slash];
    }

    /// Grow `buf` to at least `need` bytes.
    ///
    /// Rounding up to 1 MiB keeps a varying prefill width (1..k tokens) from
    /// reallocating every call, and the buffer is never shrunk because matvec is on
    /// the decode hot path.  This is the **only** `cudaMalloc` reachable from
    /// `matvec` / `matvecBatchK` / `dequantRow`; everything else in those slots is a
    /// launch or a `cudaMemcpy`.
    fn ensureBuf(cu: *cudart.Lib, buf: *?*anyopaque, cur: *usize, need: usize) !void {
        if (buf.* != null and cur.* >= need) return;
        const want = mem.alignForward(usize, need, 1 << 20);
        if (buf.*) |p| _ = cu.Free(p);
        var np: ?*anyopaque = null;
        const rc = cu.Malloc(&np, want);
        if (rc != cudart.cudaSuccess) {
            buf.* = null;
            cur.* = 0;
            return error.OutOfMemory;
        }
        buf.* = np;
        cur.* = want;
    }

    fn findDev(self: *CudaBackend, host: [*]const u8) ?*DevRow {
        for (self.dev_rows.items) |*e| {
            if (e.host == host) return e;
        }
        return null;
    }

    /// Free device memory for the allocation-failure message.  0 when the query
    /// itself fails, in which case the caller leaves it out.
    fn freeDevMiB(cu: *cudart.Lib) usize {
        var freeb: usize = 0;
        var total: usize = 0;
        if (cu.MemGetInfo(&freeb, &total) != cudart.cudaSuccess) return 0;
        return freeb >> 20;
    }

    /// `$QWEN35_CUDA_MAX_DEV_MIB`: optional upload cap, so a model that will not fit
    /// fails during load with a budget message instead of a driver error deep into
    /// the layer stack.  Unset or unparsable = no cap.
    fn budgetExceeded(self: *CudaBackend) bool {
        const v = cudart.getenv("QWEN35_CUDA_MAX_DEV_MIB") orelse return false;
        const mib = std.fmt.parseInt(usize, mem.trim(u8, v, " "), 10) catch return false;
        return self.stat_uploaded_bytes >= (mib << 20);
    }

    /// Copy one activation vector to `d_act`, growing the buffer as needed.
    fn uploadAct(self: *CudaBackend, x: []const f32) void {
        const cu = &self.cu.?;
        const bytes = x.len * @sizeOf(f32);
        ensureBuf(&cu.lib, &self.d_act, &self.d_act_bytes, bytes) catch {
            std.debug.print("cuda backend: activation upload buffer allocation failed ({d} B)\n", .{bytes});
            return;
        };
        const rc = cu.lib.Memcpy(self.d_act, x.ptr, bytes, cudart.MemcpyHostToDevice);
        if (rc != cudart.cudaSuccess) {
            std.debug.print("cuda backend: activation upload failed: {s}\n", .{std.mem.span(cu.lib.errStr(rc))});
        }
    }

    fn downloadOut(self: *CudaBackend, out: []f32) void {
        const cu = &self.cu.?;
        const bytes = out.len * @sizeOf(f32);
        const rc = cu.lib.Memcpy(out.ptr, self.d_out, bytes, cudart.MemcpyDeviceToHost);
        if (rc != cudart.cudaSuccess) {
            std.debug.print("cuda backend: result download failed: {s}\n", .{std.mem.span(cu.lib.errStr(rc))});
        }
    }

    /// Quantize one token's activation into `self.d_xq`, in the format the weight
    /// type needs (Q8_0 for 32-element-block weights, Q8_K otherwise).  Returns the
    /// byte count written.
    fn quantizeAct(self: *CudaBackend, ttype: gguf.TensorType, cols: usize) !usize {
        const cu = &self.cu.?;
        const nb = cols / actBlockElems(ttype);
        std.debug.assert(nb * actBlockElems(ttype) == cols);
        const d_act: [*]const f32 = @ptrCast(@alignCast(self.d_act.?));
        const rc = if (actBlockElems(ttype) == gguf.blockElems(.q8_0))
            self.kernels().quantize_q8_0(d_act, self.d_xq.?, @intCast(nb), null)
        else
            self.kernels().quantize_q8_k(d_act, self.d_xq.?, @intCast(nb), null);
        if (rc != cudart.cudaSuccess) {
            std.debug.print("cuda backend: quantize({s}) launch failed: {s}\n", .{
                @tagName(ttype), std.mem.span(cu.lib.errStr(rc)),
            });
            return error.LaunchFailed;
        }
        return nb * actBlockBytes(ttype);
    }

    // ==== slot: loadLinear / freeLinear =====================================

    /// Upload every weight the CPU loader accepted, packed or f32.
    ///
    /// The CPU loader stays authoritative for geometry, mmap resolution and type
    /// support; it already reports missing/unknown tensors with the `LoadError` the
    /// engine expects.  This function adds exactly one decision: does the tensor
    /// fit on the device, and if not, say so loudly.
    fn vLoadLinear(ctx: *anyopaque, name: []const u8) backend.LoadError!backend.LinearW {
        const self = self_(ctx);
        const w = try self.cpu.exec().loadLinear(name);
        errdefer self.cpu.exec().freeLinear(w);

        if (!self.gpuReady()) {
            std.debug.print("cuda backend: refusing to load '{s}' on the CPU ({s})\n", .{
                name, self.needHelp(),
            });
            return error.BackendUnavailable;
        }
        const cu = &self.cu.?;

        // Host bytes to upload, plus what the launcher needs to know about the
        // tensor's shape and type.
        const holder: []const u8 = switch (w) {
            .quant => |q| q.data,
            .f32 => |m| mem.sliceAsBytes(m),
        };
        const ttype: ?gguf.TensorType = switch (w) {
            .quant => |q| q.ttype,
            .f32 => null,
        };
        // Geometry: the QuantRow view knows rows/cols for a packed weight.  The f32
        // expansion is a flat slice, so it is registered as one row of `len`
        // columns; the engine only sends f32 weights through `dequantRow` or a
        // `matvec` whose x is exactly that wide.
        const geom = switch (w) {
            .quant => |q| Geometry{ .rows = q.rows, .cols = q.cols() },
            // A f16/bf16/f32 weight is `LinearW.f32`, a flat expansion with no
            // geometry of its own; recover cols/rows from the tensor's GGUF dims
            // (GGML row-major: dims[0] is the row length).  Falls back to a
            // single row so a missing tensor still reports a clear shape error
            // from the matvec assert rather than silently using the wrong cols.
            .f32 => |m| if (self.g.findTensor(name)) |t|
                Geometry{
                    .rows = if (t.dims.len > 1) @intCast(t.nElems() / @max(t.dims[0], 1)) else 1,
                    .cols = @intCast(t.dims[0]),
                }
            else
                Geometry{ .rows = 1, .cols = m.len },
        };
        const bytes = holder.len;
        if (bytes == 0) return error.MissingTensor;

        if (self.budgetExceeded()) {
            std.debug.print(
                "cuda backend: $QWEN35_CUDA_MAX_DEV_MIB budget exceeded by '{s}' ({d} MiB; {d} MiB uploaded already)\n",
                .{ name, bytes >> 20, self.stat_uploaded_bytes >> 20 },
            );
            return error.OutOfMemory;
        }

        var dp: ?*anyopaque = null;
        const rc = cu.lib.Malloc(&dp, bytes);
        if (rc != cudart.cudaSuccess) {
            // No silent CPU retreat: a partially resident model would be both slow
            // and impossible to attribute.  Name the tensor, type, size and what is
            // left on the device, then stop the load.
            std.debug.print(
                "cuda backend: cudaMalloc({d} MiB) for '{s}' ({s}) failed: {s}; {d} MiB free on the device\n" ++
                    "       a cuda run must fit entirely in device memory; use --backend cpu if it does not\n",
                .{
                    bytes >> 20,
                    name,
                    if (ttype) |t| @tagName(t) else "f32",
                    std.mem.span(cu.lib.errStr(rc)),
                    freeDevMiB(&cu.lib),
                },
            );
            return error.OutOfMemory;
        }

        const rc2 = cu.lib.Memcpy(dp, holder.ptr, bytes, cudart.MemcpyHostToDevice);
        if (rc2 != cudart.cudaSuccess) {
            _ = cu.lib.Free(dp);
            std.debug.print("cuda backend: upload of '{s}' ({d} MiB) failed: {s}\n", .{
                name, bytes >> 20, std.mem.span(cu.lib.errStr(rc2)),
            });
            return error.OutOfMemory;
        }

        self.dev_rows.append(self.alloc, .{
            .host = holder.ptr,
            .dev = dp.?,
            .bytes = bytes,
            .ttype = ttype,
            .rows = geom.rows,
            .cols = geom.cols,
        }) catch {
            _ = cu.lib.Free(dp);
            return error.OutOfMemory;
        };
        self.stat_uploaded_bytes += bytes;
        return w;
    }

    fn vFreeLinear(ctx: *anyopaque, w: backend.LinearW) void {
        const self = self_(ctx);
        const host: [*]const u8 = switch (w) {
            .quant => |q| q.data.ptr,
            .f32 => |m| mem.sliceAsBytes(m).ptr,
        };
        if (self.cu) |cu| {
            var i: usize = 0;
            while (i < self.dev_rows.items.len) : (i += 1) {
                if (self.dev_rows.items[i].host == host) {
                    _ = cu.lib.Free(self.dev_rows.items[i].dev);
                    _ = self.dev_rows.swapRemove(i);
                    break;
                }
            }
        }
        // Host side: an mmap view (quant, not owned) or an arena copy (f32, owned
        // by the CPU backend).  Let it decide.
        self.cpu.exec().freeLinear(w);
    }

    // ==== slot: dequantRow ==================================================

    /// Expand one row of a packed weight on the device.
    ///
    /// The source row is read *on the device*: the kernel gets `dev + row *
    /// rowBytes` (device-side pointer arithmetic, no host round trip), so a 27B
    /// embedding table is never re-uploaded per token.  Only the `cols` f32 result
    /// is copied back.  Buffers are the shared pair, grown by `ensureBuf`, so a
    /// repeated lookup does no `cudaMalloc`.
    ///
    /// The engine calls this once per row (token embedding, lm_head), so it is a
    /// latency path rather than a throughput one.
    fn vDequantRow(ctx: *anyopaque, dst: []f32, w: gguf.QuantRow, row: usize) void {
        const self = self_(ctx);
        if (!self.gpuReady()) {
            std.debug.print("cuda backend: dequantRow: {s}\n", .{self.needHelp()});
            @memset(dst, 0);
            return;
        }
        const e = self.findDev(w.data.ptr) orelse {
            // Only reachable for a weight never loaded through this backend.  Say
            // so rather than compute a plausible-looking row on the CPU -- exactly
            // the class of silent divergence in AGENTS.md §5.2.
            std.debug.print("cuda backend: dequantRow: a '{s}' weight is not resident on the device\n", .{@tagName(w.ttype)});
            @memset(dst, 0);
            return;
        };
        const cu = &self.cu.?;
        const cols = w.cols();
        const row_bytes = w.rowBytes();
        std.debug.assert(dst.len >= cols);
        std.debug.assert(row < w.rows);

        const out_bytes = cols * @sizeOf(f32);
        ensureBuf(&cu.lib, &self.d_out, &self.d_out_bytes, out_bytes) catch {
            std.debug.print("cuda backend: dequantRow: cannot allocate {d} B of device output\n", .{out_bytes});
            @memset(dst, 0);
            return;
        };

        const t0 = Io.Clock.Timestamp.now(self.io, .awake);
        const src: [*]const u8 = @ptrCast(e.dev);
        const d_row = src + row * row_bytes;
        const d_out: [*]f32 = @ptrCast(@alignCast(self.d_out));
        const rc = self.kernels().dequantize_row(
            @intCast(@backingInt(w.ttype)),
            d_row,
            d_out,
            @intCast(cols),
            null,
        );
        if (rc != cudart.cudaSuccess) {
            std.debug.print("cuda backend: dequantize_row({s}) launch failed: {s}\n", .{
                @tagName(w.ttype), std.mem.span(cu.lib.errStr(rc)),
            });
            @memset(dst, 0);
            return;
        }
        const rc2 = cu.lib.Memcpy(dst.ptr, self.d_out, out_bytes, cudart.MemcpyDeviceToHost);
        if (rc2 != cudart.cudaSuccess) {
            std.debug.print("cuda backend: dequantRow download failed: {s}\n", .{std.mem.span(cu.lib.errStr(rc2))});
            @memset(dst, 0);
            return;
        }
        self.stat_ns_dequant += durNs(t0, self.io);
        self.stat_n_dequant += 1;
    }

    // ==== slot: matvec ======================================================

    /// `out = W * x` entirely on the device: upload x into `d_act`, quantize to the
    /// activation format matching the weight type, table-driven matvec, then
    /// download `rows` floats.  `ttype` only chooses the activation format and
    /// quantizer -- never a backend, and never a CPU branch.
    fn vMatvec(ctx: *anyopaque, out: []f32, w: backend.LinearW, x: []const f32) void {
        const self = self_(ctx);
        if (!self.gpuReady()) {
            std.debug.print("cuda backend: matvec: {s}\n", .{self.needHelp()});
            @memset(out, 0);
            return;
        }
        const cu = &self.cu.?;
        const t0 = Io.Clock.Timestamp.now(self.io, .awake);

        switch (w) {
            .f32 => |m| {
                const e = self.findDev(mem.sliceAsBytes(m).ptr) orelse {
                    std.debug.print("cuda backend: matvec: f32 weight is not resident on the device\n", .{});
                    @memset(out, 0);
                    return;
                };
                const rows = e.rows;
                const cols = e.cols;
                std.debug.assert(x.len >= cols and out.len >= rows);
                self.uploadAct(x);
                ensureBuf(&cu.lib, &self.d_out, &self.d_out_bytes, rows * @sizeOf(f32)) catch {
                    std.debug.print("cuda backend: matvec: device output allocation failed\n", .{});
                    @memset(out, 0);
                    return;
                };
                const rc = self.kernels().mul_mat_vec_f32(
                    @ptrCast(@alignCast(e.dev)),
                    @ptrCast(@alignCast(self.d_act)),
                    @ptrCast(@alignCast(self.d_out)),
                    @intCast(rows),
                    @intCast(cols),
                    1,
                    null,
                );
                if (rc != cudart.cudaSuccess) {
                    std.debug.print("cuda backend: mul_mat_vec_f32 launch failed: {s}\n", .{std.mem.span(cu.lib.errStr(rc))});
                    @memset(out, 0);
                    return;
                }
                self.downloadOut(out[0..rows]);
                self.stat_ns_matvec += durNs(t0, self.io);
                self.stat_by_type[@backingInt(gguf.TensorType.f32)] += 1;
                self.stat_n_matvec += 1;
                return;
            },
            .quant => |q| {
                const e = self.findDev(q.data.ptr) orelse {
                    std.debug.print("cuda backend: matvec: a '{s}' weight is not resident on the device\n", .{@tagName(q.ttype)});
                    @memset(out, 0);
                    return;
                };
                const cols = q.cols();
                const rows = q.rows;
                const xq_bytes = actBytes(q.ttype, cols);
                std.debug.assert(x.len >= cols and out.len >= rows);

                self.uploadAct(x);
                ensureBuf(&cu.lib, &self.d_xq, &self.d_xq_bytes, xq_bytes) catch {
                    std.debug.print("cuda backend: matvec: device activation allocation failed ({d} B)\n", .{xq_bytes});
                    @memset(out, 0);
                    return;
                };
                ensureBuf(&cu.lib, &self.d_out, &self.d_out_bytes, rows * @sizeOf(f32)) catch {
                    std.debug.print("cuda backend: matvec: device output allocation failed\n", .{});
                    @memset(out, 0);
                    return;
                };
                _ = self.quantizeAct(q.ttype, cols) catch {
                    @memset(out, 0);
                    return;
                };
                // api.h order: (d_w, d_xq, d_out, rows, cols, tokens, ttype, stream).
                const rc = self.kernels().mul_mat_vec(
                    e.dev,
                    self.d_xq,
                    @ptrCast(@alignCast(self.d_out)),
                    @intCast(rows),
                    @intCast(cols),
                    1,
                    @intCast(@backingInt(q.ttype)),
                    null,
                );
                if (rc != cudart.cudaSuccess) {
                    std.debug.print("cuda backend: mul_mat_vec({s}) launch failed: {s}\n", .{
                        @tagName(q.ttype), std.mem.span(cu.lib.errStr(rc)),
                    });
                    @memset(out, 0);
                    return;
                }
                self.downloadOut(out[0..rows]);
                self.stat_ns_matvec += durNs(t0, self.io);
                self.stat_by_type[@backingInt(q.ttype)] += 1;
                self.stat_n_matvec += 1;
                if (cudart.getenv("QWEN35_PROF") != null and self.stat_n_matvec == 1) {
                    // `reportProfile` has no call site on the CLI path and the engine
                    // is arena-allocated (so `deinit` never runs).  Announce the device
                    // once, on the first GPU matvec: the observable proof the kernel
                    // ran instead of a host fallback.
                    self.reportStats("first gpu matvec");
                }
                self.mvCheck(out, q, x);
            },
        }
    }

    /// `$QWEN35_MVCHECK`: **debug-only cross-check, not a computation path**.
    ///
    /// This is the only place in the file that calls the CPU matvec.  It is
    /// limited to the first matvec of a run, behind an env var, purely to compare
    /// device and host results (the class of bug in AGENTS.md §5.2: a kernel that
    /// is silently wrong for one quant type).  `out` already holds the device
    /// result and is never overwritten; the CPU value goes to a scratch slice that
    /// is freed here.
    fn mvCheck(self: *CudaBackend, out: []const f32, q: gguf.QuantRow, x: []const f32) void {
        if (cudart.getenv("QWEN35_MVCHECK") == null) return;
        if (self.mvcheck_seen[@backingInt(q.ttype)]) return;
        self.mvcheck_seen[@backingInt(q.ttype)] = true;

        // The CPU matvec needs its own activation scratch, and this backend never
        // calls `CpuBackend.allocScratch` (the device owns those buffers).  Hand it
        // a throwaway pair sized the way `backend_cpu.vAllocScratch` would, so the
        // reference run cannot assert on a zero-length scratch.
        const cols = q.cols();
        const cpu = &self.cpu.exec();
        cpu.allocScratch(cols) catch return;
        defer cpu.freeScratch();
        const tmp = self.alloc.alloc(f32, out.len) catch return;
        defer self.alloc.free(tmp);
        cpu.matvec(tmp, .{ .quant = q }, x);
        var bad: usize = 0;
        for (out, tmp) |a, b| {
            // Relative tolerance, not bit equality: the device kernel may reorder
            // the accumulation, so an exact match is not expected even when
            // correct.
            const denom = @max(@abs(b), 1e-3);
            if (@abs(a - b) / denom > 2e-2) {
                if (bad < 3) std.debug.print("  MVCHECK row: dev={d} cpu={d}\n", .{ a, b });
                bad += 1;
            }
        }
        std.debug.print("MVCHECK[{s}] rows={d} cols={d} mismatched={d}\n", .{
            @tagName(q.ttype), q.rows, q.cols(), bad,
        });
    }

    // ==== slot: matvecBatchK ================================================

    /// K prefetched tokens at once, all on the device.
    ///
    /// Activation layout matches the VTable contract (`out[j*rows + r]`) and the
    /// CPU reference: `xK` is `K` row-major token rows of `cols` floats, uploaded
    /// in one `cudaMemcpy`, quantized as `K` consecutive block groups in one buffer
    /// (`d_xq`, group `j` at `j * actBytes(ttype, cols)`), then a single launch
    /// computes `K*rows` outputs and one download returns them.
    fn vMatvecBatchK(ctx: *anyopaque, out: []f32, w: backend.LinearW, xK: []const f32, k: usize) void {
        const self = self_(ctx);
        if (!self.gpuReady()) {
            std.debug.print("cuda backend: matvecBatchK: {s}\n", .{self.needHelp()});
            @memset(out, 0);
            return;
        }
        const cu = &self.cu.?;
        const t0 = Io.Clock.Timestamp.now(self.io, .awake);

        switch (w) {
            .f32 => |m| {
                const e = self.findDev(mem.sliceAsBytes(m).ptr) orelse {
                    std.debug.print("cuda backend: matvecBatchK: f32 weight is not resident on the device\n", .{});
                    @memset(out, 0);
                    return;
                };
                const rows = e.rows;
                const cols = e.cols;
                std.debug.assert(xK.len >= k * cols and out.len >= k * rows);
                const act_bytes = k * cols * @sizeOf(f32);
                ensureBuf(&cu.lib, &self.d_act, &self.d_act_bytes, act_bytes) catch return oomMsg("batchK activation");
                const rc0 = cu.lib.Memcpy(self.d_act, xK.ptr, act_bytes, cudart.MemcpyHostToDevice);
                if (rc0 != cudart.cudaSuccess) {
                    std.debug.print("cuda backend: batchK f32 activation upload failed: {s}\n", .{std.mem.span(cu.lib.errStr(rc0))});
                    @memset(out, 0);
                    return;
                }
                ensureBuf(&cu.lib, &self.d_out, &self.d_out_bytes, k * rows * @sizeOf(f32)) catch return oomMsg("batchK output");
                const rc = self.kernels().mul_mat_vec_f32(
                    @ptrCast(@alignCast(e.dev)),
                    @ptrCast(@alignCast(self.d_act)),
                    @ptrCast(@alignCast(self.d_out)),
                    @intCast(rows),
                    @intCast(cols),
                    @intCast(k),
                    null,
                );
                if (rc != cudart.cudaSuccess) {
                    std.debug.print("cuda backend: mul_mat_vec_f32(K={d}) launch failed: {s}\n", .{
                        k, std.mem.span(cu.lib.errStr(rc)),
                    });
                    @memset(out, 0);
                    return;
                }
                self.downloadOut(out[0 .. k * rows]);
                self.stat_ns_batchk += durNs(t0, self.io);
                self.stat_n_batchk += 1;
                return;
            },
            .quant => |q| {
                const e = self.findDev(q.data.ptr) orelse {
                    std.debug.print("cuda backend: matvecBatchK: a '{s}' weight is not resident on the device\n", .{@tagName(q.ttype)});
                    @memset(out, 0);
                    return;
                };
                const cols = q.cols();
                const rows = q.rows;
                const per_tok = actBytes(q.ttype, cols);
                const nb_tok = cols / actBlockElems(q.ttype);
                std.debug.assert(xK.len >= k * cols and out.len >= k * rows);

                // One H2D for all K activations.
                const act_bytes = k * cols * @sizeOf(f32);
                ensureBuf(&cu.lib, &self.d_act, &self.d_act_bytes, act_bytes) catch return oomMsg("batchK activation");
                const rc0 = cu.lib.Memcpy(self.d_act, xK.ptr, act_bytes, cudart.MemcpyHostToDevice);
                if (rc0 != cudart.cudaSuccess) {
                    std.debug.print("cuda backend: batchK activation upload failed: {s}\n", .{std.mem.span(cu.lib.errStr(rc0))});
                    @memset(out, 0);
                    return;
                }
                ensureBuf(&cu.lib, &self.d_xq, &self.d_xq_bytes, per_tok * k) catch return oomMsg("batchK quantized activation");
                ensureBuf(&cu.lib, &self.d_out, &self.d_out_bytes, k * rows * @sizeOf(f32)) catch return oomMsg("batchK output");

                // Quantize token j into block group j.  The quantizers take a
                // contiguous f32 span plus a block count, so pointing them at
                // `d_act + j*cols` and `d_xq + j*per_tok` produces exactly that.
                const base_act: [*]const f32 = @ptrCast(@alignCast(self.d_act.?));
                const base_xq: [*]u8 = @ptrCast(self.d_xq.?);
                const q8_0_act = actBlockElems(q.ttype) == gguf.blockElems(.q8_0);
                for (0..k) |j| {
                    const xq_slot: ?*anyopaque = @ptrCast(base_xq + j * per_tok);
                    const rc = if (q8_0_act)
                        self.kernels().quantize_q8_0(base_act + j * cols, xq_slot, @intCast(nb_tok), null)
                    else
                        self.kernels().quantize_q8_k(base_act + j * cols, xq_slot, @intCast(nb_tok), null);
                    if (rc != cudart.cudaSuccess) {
                        std.debug.print("cuda backend: batchK quantize({s}) tok {d} failed: {s}\n", .{
                            @tagName(q.ttype), j, std.mem.span(cu.lib.errStr(rc)),
                        });
                        @memset(out, 0);
                        return;
                    }
                }

                const rc = self.kernels().mul_mat_vec(
                    e.dev,
                    self.d_xq,
                    @ptrCast(@alignCast(self.d_out)),
                    @intCast(rows),
                    @intCast(cols),
                    @intCast(k),
                    @intCast(@backingInt(q.ttype)),
                    null,
                );
                if (rc != cudart.cudaSuccess) {
                    std.debug.print("cuda backend: mul_mat_vec({s}, K={d}) launch failed: {s}\n", .{
                        @tagName(q.ttype), k, std.mem.span(cu.lib.errStr(rc)),
                    });
                    @memset(out, 0);
                    return;
                }
                self.downloadOut(out[0 .. k * rows]);
                self.stat_ns_batchk += durNs(t0, self.io);
                self.stat_by_type[@backingInt(q.ttype)] += 1;
                self.stat_n_batchk += 1;
                // Token 0 of the chunk is enough: all K rows run through the same
                // kernel, so a per-type mismatch shows up on the first one.
                self.mvCheck(out[0..rows], q, xK[0..cols]);
            },
        }
    }

    // ==== slots: device scratch =============================================

    /// Reserve device scratch for one token.
    ///
    /// Sizes follow `backend_cpu.q8ScratchBytes` / `q8kScratchBytes` exactly (34 B
    /// per 32 elements, 292 B per 256) so a config the CPU backend accepted cannot
    /// overflow here.  This is also where a failed device init becomes a load
    /// error: `init()` cannot report one (the factory contract is
    /// `init(alloc, g) -> Backend`), and quietly running the whole model on the CPU
    /// after `--backend cuda` would be a lie.
    fn dAllocScratch(ctx: *anyopaque, max_cols: usize) anyerror!void {
        const self = self_(ctx);
        if (!self.gpuReady()) return error.BackendUnavailable;
        const cu = &self.cu.?;
        try ensureBuf(&cu.lib, &self.d_act, &self.d_act_bytes, max_cols * @sizeOf(f32));
        // Both activation formats up front: which one a layer needs is only known at
        // matvec time, and a mid-model allocation is what this call exists to avoid.
        try ensureBuf(&cu.lib, &self.d_xq, &self.d_xq_bytes, actBytes(.q8_k, max_cols));
        try ensureBuf(&cu.lib, &self.d_out, &self.d_out_bytes, max_cols * @sizeOf(f32));
        try ensureBuf(&cu.lib, &self.d_deq, &self.d_deq_bytes, max_cols * @sizeOf(f32));
    }

    /// K-token variant: the same buffers scaled by `k`.  `d_out`/`d_xq` dominate, so
    /// a wide prefill chunk shows up in the `CUDA-M` buffer total.
    fn dAllocScratchK(ctx: *anyopaque, max_cols: usize, k: usize) anyerror!void {
        const self = self_(ctx);
        if (!self.gpuReady()) return error.BackendUnavailable;
        const cu = &self.cu.?;
        try ensureBuf(&cu.lib, &self.d_act, &self.d_act_bytes, k * max_cols * @sizeOf(f32));
        try ensureBuf(&cu.lib, &self.d_xq, &self.d_xq_bytes, k * actBytes(.q8_k, max_cols));
        try ensureBuf(&cu.lib, &self.d_out, &self.d_out_bytes, k * max_cols * @sizeOf(f32));
    }

    fn dFreeScratch(ctx: *anyopaque) void {
        const self = self_(ctx);
        const cu = if (self.cu) |*c| c else return;
        if (self.d_act) |p| _ = cu.lib.Free(p);
        if (self.d_xq) |p| _ = cu.lib.Free(p);
        if (self.d_out) |p| _ = cu.lib.Free(p);
        if (self.d_deq) |p| _ = cu.lib.Free(p);
        self.d_act = null;
        self.d_act_bytes = 0;
        self.d_xq = null;
        self.d_xq_bytes = 0;
        self.d_out = null;
        self.d_out_bytes = 0;
        self.d_deq = null;
        self.d_deq_bytes = 0;
    }

    // ==== profiling / introspection =========================================

    /// `$QWEN35_PROF`: one `CUDA-M [...]` line with what actually ran on the device,
    /// plus a per-type breakdown, so "did this model mix backends?" is answerable
    /// from the log.  Uploaded MiB is the resident weight footprint; bufs_MiB is the
    /// reusable activation/output scratch.
    fn reportStats(self: *CudaBackend, when: []const u8) void {
        if (self.cu == null) return;
        const buf_bytes = self.d_act_bytes + self.d_xq_bytes + self.d_out_bytes + self.d_deq_bytes;
        std.debug.print(
            "CUDA-M  [{s}] matvec={d} batchk={d} dequant_row={d} up_MiB={d} resident_rows={d} bufs_MiB={d} t_mv_ms={d} t_bk_ms={d} t_dq_ms={d}\n",
            .{
                when,
                self.stat_n_matvec,
                self.stat_n_batchk,
                self.stat_n_dequant,
                self.stat_uploaded_bytes >> 20,
                self.dev_rows.items.len,
                buf_bytes >> 20,
                self.stat_ns_matvec / 1_000_000,
                self.stat_ns_batchk / 1_000_000,
                self.stat_ns_dequant / 1_000_000,
            },
        );
        // Per-type breakdown, only for types that actually ran.  Assembled with
        // `bufPrint` into a growing slice: `std.io` writers are gone in Zig 0.17.
        var line: [448]u8 = undefined;
        var n: usize = 0;
        n += (std.fmt.bufPrint(line[n..], "CUDA-M  types:", .{}) catch return).len;
        var any = false;
        for (0..self.stat_by_type.len) |i| {
            const cnt = self.stat_by_type[i];
            if (cnt == 0) continue;
            const tt: gguf.TensorType = @fromBackingInt(@intCast(@as(u32, @intCast(i))));
            n += (std.fmt.bufPrint(line[n..], " {s}={d}", .{ @tagName(tt), cnt }) catch break).len;
            any = true;
        }
        if (!any) n += (std.fmt.bufPrint(line[n..], " (none)", .{}) catch line[n..n]).len;
        std.debug.print("{s}\n", .{line[0..n]});
    }

    fn vName(ctx: *anyopaque) []const u8 {
        const self = self_(ctx);
        if (self.cu) |cu| {
            return std.fmt.bufPrint(&msg_buf, "cuda ({s}, sm {d}.{d})", .{
                cu.devName(), cu.cc / 10, cu.cc % 10,
            }) catch "cuda";
        }
        return "cuda (unavailable)";
    }

    fn vDescribe(ctx: *anyopaque) []const u8 {
        const self = self_(ctx);
        if (self.cu) |cu| {
            return std.fmt.bufPrint(
                &msg_buf,
                "cuda sm {d}.{d} {s}, {d} MiB free; all quantized types on device",
                .{ cu.cc / 10, cu.cc % 10, cu.devName(), cu.free_bytes >> 20 },
            ) catch "cuda";
        }
        return self.init_error;
    }

    fn dSetTier(ctx: *anyopaque, tier_name: []const u8) void {
        // The device has one implementation per format; the CPU ISA tier only
        // affects the metadata loader now.  Accepted and ignored so `--tier` keeps
        // its diagnostic value for that part.
        _ = ctx;
        _ = tier_name;
    }

    fn dSetThreads(ctx: *anyopaque, io: std.Io, nt: usize) void {
        // Only the (non-computational) metadata loader runs on the host; the pool is
        // otherwise unused, but the CPU backend is still told so its profile lines
        // stay meaningful.
        const self = self_(ctx);
        self.io = io;
        self.cpu.exec().setThreads(io, nt);
    }

    fn dProfPhase(ctx: *anyopaque, on: bool) void {
        self_(ctx).cpu.exec().profPhase(on);
    }

    fn dProfReset(ctx: *anyopaque) void {
        const self = self_(ctx);
        self.stat_ns_matvec = 0;
        self.stat_ns_batchk = 0;
        self.stat_ns_dequant = 0;
        self.stat_n_matvec = 0;
        self.stat_n_batchk = 0;
        self.stat_n_dequant = 0;
        self.stat_by_type = @splat(0);
        self.cpu.exec().profReset();
    }

    fn dReportProfile(ctx: *anyopaque) void {
        const self = self_(ctx);
        self.cpu.exec().reportProfile();
        self.reportStats("report");
    }
};

// ==== the Exec vtable ========================================================

pub const vtable = backend.Exec.VTable{
    // Device: weights and every numeric kernel.
    .loadLinear = CudaBackend.vLoadLinear,
    .freeLinear = CudaBackend.vFreeLinear,
    .dequantRow = CudaBackend.vDequantRow,
    .matvec = CudaBackend.vMatvec,
    .matvecBatchK = CudaBackend.vMatvecBatchK,
    .allocScratch = CudaBackend.dAllocScratch,
    .allocScratchK = CudaBackend.dAllocScratchK,
    .freeScratch = CudaBackend.dFreeScratch,
    // Host, non-computational only: thread-pool bookkeeping for the metadata loader
    // and the timing counters it prints.
    .setThreads = CudaBackend.dSetThreads,
    .profPhase = CudaBackend.dProfPhase,
    .profReset = CudaBackend.dProfReset,
    .setTier = CudaBackend.dSetTier,
    .reportProfile = CudaBackend.dReportProfile,
    .name = CudaBackend.vName,
    .describe = CudaBackend.vDescribe,
};
