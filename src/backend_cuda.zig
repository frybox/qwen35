//! CUDA backend -- **stage 1: Q8_0 weights and dequant actually on the GPU**.
//!
//! Scope of this stage (docs/cuda-backend-brief.md §2/§3):
//!
//!   * `loadLinear`  -- Q8_0 tensors are copied into device memory once, at
//!                      model load; the host-side `QuantRow` (a view into the
//!                      mmap) is kept because the engine needs the geometry and
//!                      because the CPU fallbacks still use it.
//!   * `matvec`      -- Q8_0 weights run `q35_k_mul_mat_vec_q8_0` on the device.
//!   * `dequantRow`  -- Q8_0 rows run `q35_k_dequantize_q8_0` on the device
//!                      (token embedding / lm_head / norms go through here).
//!   * `freeLinear`  -- releases the device copy.
//!
//! Everything else delegates to the embedded `CpuBackend`'s vtable:
//! `allocScratch*`, `freeScratch`, `matvecBatchK`, `setThreads`, the profiling
//! hooks and introspection.  Mixed runs are the normal case (a Q4_0/Q6_K/BF16
//! model has no Q8_0 weights at all and therefore runs entirely on the CPU
//! through this same object), and the acceptance test requires those models to
//! keep producing byte-identical token ids.
//!
//! Decisions worth knowing:
//!
//!   * **Non-Q8_0 weights are NOT uploaded.**  They stay in the mmap and are
//!     served by the CPU path through the delegating `matvec`.  Uploading them
//!     would need per-format GPU kernels (out of scope) or a host-side f32
//!     dequant that triples memory and loses the zero-copy mmap property.
//!   * **cudart is dlopen'd, never linked** (`src/cudart.zig`), and the kernels
//!     live in a separately dlopen'd `.so`.  A host with no CUDA install must
//!     still run `--backend cpu`, so the executable itself may not carry a
//!     DT_NEEDED on libcudart.
//!   * **Activations are uploaded per matvec** into a reused device buffer
//!     (brief §2 decision (a)); no cudaMalloc in the hot path.
//!   * **`matvecBatchK` stays on the CPU.**  Porting the batched prefill path
//!     needs a K-token activation layout the device kernels do not have yet.
//!     `QWEN35_NO_BATCH=1` therefore routes prefill through the GPU too.
//!
//! Numeric contract: the device kernels reproduce the CPU float addition tree
//! exactly (one thread per row, blocks accumulated left to right with fma), so
//! `--backend cuda` produces bit-identical logits and ids.  See vecdotq.cuh.

const std = @import("std");
const gguf = @import("gguf.zig");
const backend = @import("backend.zig");
const cpu_mod = @import("backend_cpu.zig");
const cudart = @import("cudart.zig");

/// Kernel entry points, resolved with dlsym() from
/// `libqwen35-cuda-kernels.so` (built by `zig build -Dcuda=true`).
///
/// They are deliberately NOT `extern fn`.  The nvcc host stubs reference
/// libcudart symbols (`__cudaPushCallConfiguration`, `__cudaLaunchKernel`, ...),
/// so linking the objects into the CLI would give the executable a DT_NEEDED
/// on libcudart -- and `--backend cpu` would stop working on CUDA-less hosts.
const Kernels = struct {
    quantize_q8_0: *const fn (?[*]const f32, ?*anyopaque, c_int, ?*anyopaque) callconv(.c) c_int,
    dequantize_q8_0: *const fn (?*const anyopaque, ?[*]f32, c_int, ?*anyopaque) callconv(.c) c_int,
    mul_mat_vec_q8_0: *const fn (?*const anyopaque, ?*const anyopaque, ?[*]f32, c_int, c_int, c_int, ?*anyopaque) callconv(.c) c_int,
    mul_mat_vec_f32: *const fn (?[*]const f32, ?[*]const f32, ?[*]f32, c_int, c_int, c_int, ?*anyopaque) callconv(.c) c_int,
};

/// One uploaded tensor: the host mmap address doubles as the key, because
/// `LinearW.quant.data` points into the mapping and never moves.
const Entry = struct { host: [*]const u8, dev: *anyopaque, bytes: usize };

const default_kernel_so = "libqwen35-cuda-kernels.so";

pub const CudaBackend = struct {
    /// Embedded by value (not by pointer): a `CudaBackend` owns its CPU half,
    /// so `Engine.load`'s single `alloc.create` + `deinit` covers both and there
    /// is no second lifetime to get wrong.
    cpu: cpu_mod.CpuBackend,
    alloc: std.mem.Allocator,

    /// `null` when the runtime could not be opened.  The backend then refuses
    /// with a precise load error instead of silently falling back: a user who
    /// typed `--backend cuda` wants to know the GPU was not used.
    cu: ?cudart.Open = null,
    /// Human-readable reason for a failed `cudart.open`.
    init_error: []const u8 = &.{},
    /// Error `Engine.load` must return so main prints its line and exits != 0.
    init_err: backend.LoadError = error.BackendUnavailable,

    dev_rows: std.ArrayListUnmanaged(Entry) = .empty,

    /// Reused device buffers: f32 activation upload, Q8_0 quantized form, and
    /// the f32 result staging area.
    d_act: ?*anyopaque = null,
    d_act_bytes: usize = 0,
    d_xq: ?*anyopaque = null,
    d_xq_bytes: usize = 0,
    d_out: ?*anyopaque = null,
    d_out_bytes: usize = 0,

    kernel_lib: ?std.DynLib = null,
    kernel_fns: ?Kernels = null,

    /// Counters, printed by `reportProfile` (`QWEN35_PROF_POST`).
    stat_matvec_q8: usize = 0,
    stat_matvec_cpu: usize = 0,
    stat_mv_bytes: usize = 0,

    var msg_buf: [512]u8 = undefined;
    var exe_dir_buf: [512]u8 = undefined;
    var kernel_path_buf: [1024]u8 = undefined;

    // ---- construction -------------------------------------------------------

    pub fn init(alloc: std.mem.Allocator, g: *const gguf.File) CudaBackend {
        var self = CudaBackend{
            .cpu = cpu_mod.CpuBackend.init(alloc, g),
            .alloc = alloc,
        };
        self.tryDevice(g);
        if (self.cu != null) {
            if (!self.loadKernels()) {
                // Kernels unavailable: drop back to a clearly-labelled CPU-only
                // state instead of crashing on the first matvec.
                if (self.cu) |*cu| cu.lib.close();
                self.cu = null;
            }
        }
        return self;
    }

    /// Open libcudart and validate the device.  Never panics: a failure lands in
    /// `init_error` + `init_err`, and every GPU slot then degrades to the CPU.
    fn tryDevice(self: *CudaBackend, g: *const gguf.File) void {
        // Ask the runtime to confirm there is room for the Q8_0 weights plus
        // scratch.  Failing at load time is a named error, not an OOM halfway
        // through the first forward pass.
        var need: usize = 16 << 20;
        for (g.tensors) |t| {
            if (t.ttype != .q8_0) continue;
            if (t.dims.len < 2) continue;
            const elems: usize = @intCast(t.dims[0] * t.dims[1]);
            need += elems / gguf.blockElems(.q8_0) * gguf.blockBytes(.q8_0);
        }

        var probe: cudart.Open = undefined;
        cudart.open(&probe, need) catch |e| {
            self.init_err = error.BackendUnavailable;
            self.init_error = switch (e) {
                error.LibraryNotFound => std.fmt.bufPrint(
                    &msg_buf,
                    "libcudart.so not found (tried $QWEN35_CUDART, libcudart.so[.13/.12], " ++
                        "/usr/local/cuda*/lib64, <nvcc dir>/../lib64); use --backend cpu",
                    .{},
                ) catch "libcudart.so not found; use --backend cpu",
                error.MissingSymbol => std.fmt.bufPrint(
                    &msg_buf,
                    "libcudart.so is missing required symbols: {s}; use --backend cpu",
                    .{probe.missing()},
                ) catch "libcudart.so is missing required symbols",
                error.NoDevice => "no CUDA device available (cudaGetDeviceCount returned 0); use --backend cpu",
                error.ComputeCapabilityTooLow => std.fmt.bufPrint(
                    &msg_buf,
                    "device '{s}' has compute capability {d}.{d} but the kernels are built for " ++
                        "sm_89 (8.9); rebuild with a matching -arch or use --backend cpu",
                    .{ probe.devName(), probe.cc / 10, probe.cc % 10 },
                ) catch "device compute capability too low for sm_89 kernels",
                error.OutOfMemory => std.fmt.bufPrint(
                    &msg_buf,
                    "device '{s}' has {d} MiB free but {d} MiB is needed for the Q8_0 weights; " ++
                        "use --backend cpu",
                    .{ probe.devName(), probe.free_bytes >> 20, need >> 20 },
                ) catch "not enough device memory for the weights",
            };
            // `Engine.load` only knows the error *name*, so the readable reason
            // has to be printed here.
            std.debug.print("error: cuda backend: {s}\n", .{self.init_error});
            return;
        };
        self.cu = probe;
    }

    /// Open the kernels `.so`.  `libcudart.so` must already be resident with
    /// RTLD_GLOBAL (`cudart.open` did that) so the nvcc host stubs can bind
    /// `__cudaLaunchKernel` and friends at dlopen time.
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
        const D = *const fn (?*const anyopaque, ?[*]f32, c_int, ?*anyopaque) callconv(.c) c_int;
        const M = *const fn (?*const anyopaque, ?*const anyopaque, ?[*]f32, c_int, c_int, c_int, ?*anyopaque) callconv(.c) c_int;
        const F = *const fn (?[*]const f32, ?[*]const f32, ?[*]f32, c_int, c_int, c_int, ?*anyopaque) callconv(.c) c_int;

        const quantize = lib.lookup(Q, "q35_launch_quantize_q8_0") orelse
            return missingKernel(&lib, path, "q35_launch_quantize_q8_0");
        const dequant = lib.lookup(D, "q35_launch_dequantize_q8_0") orelse
            return missingKernel(&lib, path, "q35_launch_dequantize_q8_0");
        const mmvq = lib.lookup(M, "q35_launch_mul_mat_vec_q8_0") orelse
            return missingKernel(&lib, path, "q35_launch_mul_mat_vec_q8_0");
        const mmvf = lib.lookup(F, "q35_launch_mul_mat_vec_f32") orelse
            return missingKernel(&lib, path, "q35_launch_mul_mat_vec_f32");

        self.kernel_lib = lib;
        self.kernel_fns = .{
            .quantize_q8_0 = quantize,
            .dequantize_q8_0 = dequant,
            .mul_mat_vec_q8_0 = mmvq,
            .mul_mat_vec_f32 = mmvf,
        };
        return true;
    }

    fn missingKernel(lib: *std.DynLib, path: []const u8, name: []const u8) bool {
        std.debug.print("error: cuda backend: '{s}' is missing symbol {s}; use --backend cpu\n", .{ path, name });
        lib.close();
        return false;
    }

    fn kernels(self: *CudaBackend) *const Kernels {
        return &self.kernel_fns.?;
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

    /// Directory holding the running executable, via /proc/self/exe.  Null when
    /// the symlink cannot be read; callers then fall back to a plain relative
    /// name, which works from `zig-out/bin`.
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

    pub fn deinit(self: *CudaBackend) void {
        self.reportStats("deinit");
        if (self.cu) |*cu| {
            for (self.dev_rows.items) |e| _ = cu.lib.Free(e.dev);
            if (self.d_act) |p| _ = cu.lib.Free(p);
            if (self.d_xq) |p| _ = cu.lib.Free(p);
            if (self.d_out) |p| _ = cu.lib.Free(p);
            self.d_act = null;
            self.d_xq = null;
            self.d_out = null;
            cu.lib.close();
            self.cu = null;
        }
        self.dev_rows.deinit(self.alloc);
        if (self.kernel_lib) |*l| l.close();
        self.kernel_lib = null;
        self.kernel_fns = null;
        self.cpu.deinit();
    }

    pub fn exec(self: *CudaBackend) backend.Exec {
        return .{ .ptr = self, .vtable = &vtable };
    }

    fn self_(ctx: *anyopaque) *CudaBackend {
        return @ptrCast(@alignCast(ctx));
    }

    fn gpuReady(self: *CudaBackend) bool {
        return self.cu != null and self.kernel_fns != null;
    }

    // ---- device buffer management -------------------------------------------

    /// Grow `buf` to at least `need` bytes.  Rounding up to 1 MiB keeps a
    /// varying prefill width (1..k tokens) from reallocating every call, and the
    /// buffer is never shrunk -- matvec is on the decode hot path.
    fn ensureBuf(cu: *cudart.Lib, buf: *?*anyopaque, cur: *usize, need: usize) !void {
        if (buf.* != null and cur.* >= need) return;
        const want = std.mem.alignForward(usize, need, 1 << 20);
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

    fn findDev(self: *CudaBackend, host: [*]const u8) ?*anyopaque {
        for (self.dev_rows.items) |e| {
            if (e.host == host) return e.dev;
        }
        return null;
    }

    // ---- ported slot: loadLinear --------------------------------------------

    fn vLoadLinear(ctx: *anyopaque, name: []const u8) backend.LoadError!backend.LinearW {
        const self = self_(ctx);
        // The CPU loader stays authoritative for geometry, mmap resolution and
        // type support; it already reports missing/unknown tensors with the
        // LoadError the engine expects.
        const w = try self.cpu.exec().loadLinear(name);
        if (!self.gpuReady()) return w;

        const q = switch (w) {
            .quant => |q| q,
            .f32 => return w,
        };
        // Only Q8_0 has a device kernel in this stage; anything else stays in
        // the mmap and goes through the delegating CPU matvec.
        if (q.ttype != .q8_0) return w;

        const cu = &self.cu.?;
        const bytes = q.data.len;
        var dp: ?*anyopaque = null;
        const rc = cu.lib.Malloc(&dp, bytes);
        if (rc != cudart.cudaSuccess) {
            // Out of device memory for this tensor: keep the CPU copy so the
            // run stays correct (slower) and say why, once per tensor.
            std.debug.print("cuda backend: cudaMalloc({d} B) for '{s}' failed: {s}; using CPU for this tensor\n", .{
                bytes, name, std.mem.span(cu.lib.errStr(rc)),
            });
            return w;
        }
        const rc2 = cu.lib.Memcpy(dp, q.data.ptr, bytes, cudart.MemcpyHostToDevice);
        if (rc2 != cudart.cudaSuccess) {
            _ = cu.lib.Free(dp);
            std.debug.print("cuda backend: upload of '{s}' failed: {s}; using CPU for this tensor\n", .{
                name, std.mem.span(cu.lib.errStr(rc2)),
            });
            return w;
        }
        self.dev_rows.append(self.alloc, .{ .host = q.data.ptr, .dev = dp.?, .bytes = bytes }) catch {
            _ = cu.lib.Free(dp);
            return error.OutOfMemory;
        };
        return w;
    }

    fn vFreeLinear(ctx: *anyopaque, w: backend.LinearW) void {
        const self = self_(ctx);
        if (w == .quant) {
            const host = w.quant.data.ptr;
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
        }
        // The host side is either a mmap view (quant, not owned) or an arena
        // copy (f32, owned by the CPU backend).  Let it decide.
        self.cpu.exec().freeLinear(w);
    }

    // ---- ported slot: dequantRow --------------------------------------------

    /// Q8_0 rows are expanded on the device; everything else delegates.
    ///
    /// `dst` is always a host slice, so the result is copied back.  Call sites
    /// (token embedding, lm_head, output_norm) expand a single row per call;
    /// this is therefore one small upload + one `cols`-float download, not a
    /// throughput path.
    fn vDequantRow(ctx: *anyopaque, dst: []f32, w: gguf.QuantRow, row: usize) void {
        const self = self_(ctx);
        if (!self.gpuReady() or w.ttype != .q8_0) {
            return self.cpu.exec().dequantRow(dst, w, row);
        }
        const dev = self.findDev(w.data.ptr) orelse
            return self.cpu.exec().dequantRow(dst, w, row);

        const cu = &self.cu.?;
        const cols = w.cols();
        std.debug.assert(dst.len >= cols);
        const row_bytes = w.blocks_per_row * gguf.blockBytes(.q8_0);

        // One device buffer per call: `row_bytes` of Q8_0 source, `cols` floats of
        // output.  Both are freed at the end; this path runs at most a few times
        // per token (embedding + lm_head + norms), not per matvec.
        const out_bytes = cols * @sizeOf(f32);
        var d_row: ?*anyopaque = null;
        var d_f32: ?*anyopaque = null;
        if (cu.lib.Malloc(&d_row, row_bytes) != cudart.cudaSuccess) {
            return self.cpu.exec().dequantRow(dst, w, row);
        }
        defer _ = cu.lib.Free(d_row);
        if (cu.lib.Malloc(&d_f32, out_bytes) != cudart.cudaSuccess) {
            return self.cpu.exec().dequantRow(dst, w, row);
        }
        defer _ = cu.lib.Free(d_f32);

        const src: [*]const u8 = @ptrCast(dev);
        const rc = cu.lib.Memcpy(d_row, src + row * row_bytes, row_bytes, cudart.MemcpyHostToDevice);
        if (rc != cudart.cudaSuccess) {
            std.debug.print("cuda backend: dequantRow upload failed: {s}; using CPU\n", .{std.mem.span(cu.lib.errStr(rc))});
            return self.cpu.exec().dequantRow(dst, w, row);
        }
        const d_f32p: [*]f32 = @ptrCast(@alignCast(d_f32));
        const rc2 = self.kernels().dequantize_q8_0(d_row, d_f32p, @intCast(w.blocks_per_row), null);
        if (rc2 != cudart.cudaSuccess) {
            std.debug.print("cuda backend: dequantize launch failed: {s}; using CPU\n", .{std.mem.span(cu.lib.errStr(rc2))});
            return self.cpu.exec().dequantRow(dst, w, row);
        }
        const rc3 = cu.lib.Memcpy(dst.ptr, d_f32, out_bytes, cudart.MemcpyDeviceToHost);
        if (rc3 != cudart.cudaSuccess) {
            std.debug.print("cuda backend: dequantRow download failed: {s}; using CPU\n", .{std.mem.span(cu.lib.errStr(rc3))});
            return self.cpu.exec().dequantRow(dst, w, row);
        }
    }

    // ---- ported slot: matvec ------------------------------------------------

    fn vMatvec(ctx: *anyopaque, out: []f32, w: backend.LinearW, x: []const f32) void {
        const self = self_(ctx);
        const q = switch (w) {
            .quant => |q| q,
            .f32 => |m| return self.matvecF32(out, m, x),
        };
        if (!self.gpuReady() or q.ttype != .q8_0) {
            self.cpu.exec().matvec(out, w, x);
            self.stat_matvec_cpu += 1;
            return;
        }
        const dev = self.findDev(q.data.ptr) orelse {
            self.cpu.exec().matvec(out, w, x);
            self.stat_matvec_cpu += 1;
            return;
        };

        const cu = &self.cu.?;
        const cols = q.cols();
        const rows = q.rows;
        std.debug.assert(x.len >= cols and out.len >= rows);

        const act_bytes = cols * @sizeOf(f32);
        const xq_bytes = q.blocks_per_row * gguf.blockBytes(.q8_0);
        const out_bytes = rows * @sizeOf(f32);
        ensureBuf(&cu.lib, &self.d_act, &self.d_act_bytes, act_bytes) catch |e| return self.fallbackMatvec(out, w, x, e);
        ensureBuf(&cu.lib, &self.d_xq, &self.d_xq_bytes, xq_bytes) catch |e| return self.fallbackMatvec(out, w, x, e);
        ensureBuf(&cu.lib, &self.d_out, &self.d_out_bytes, out_bytes) catch |e| return self.fallbackMatvec(out, w, x, e);

        const rc0 = cu.lib.Memcpy(self.d_act, x.ptr, act_bytes, cudart.MemcpyHostToDevice);
        if (rc0 != cudart.cudaSuccess) return self.failMatvec(out, w, x, rc0, "activation upload");

        const d_act: [*]f32 = @ptrCast(@alignCast(self.d_act));
        const d_out: [*]f32 = @ptrCast(@alignCast(self.d_out));
        const k = self.kernels();

        const rc1 = k.quantize_q8_0(d_act, self.d_xq, @intCast(q.blocks_per_row), null);
        if (rc1 != cudart.cudaSuccess) return self.failMatvec(out, w, x, rc1, "quantize launch");

        const rc2 = k.mul_mat_vec_q8_0(dev, self.d_xq, d_out, @intCast(rows), @intCast(cols), 1, null);
        if (rc2 != cudart.cudaSuccess) return self.failMatvec(out, w, x, rc2, "matvec launch");

        const rc3 = cu.lib.Memcpy(out.ptr, self.d_out, out_bytes, cudart.MemcpyDeviceToHost);
        if (rc3 != cudart.cudaSuccess) return self.failMatvec(out, w, x, rc3, "result download");

        if (cudart.getenv("QWEN35_MVCHECK") != null and self.stat_matvec_q8 == 0) {
            // TEMP probe: whole-row GPU vs CPU comparison plus the quantized
            // activation bytes, to localise a divergence to quantize vs dot.
            const tmp = self.alloc.alloc(f32, rows) catch return;
            defer self.alloc.free(tmp);
            self.cpu.exec().matvec(tmp, w, x);
            var bad: usize = 0;
            for (0..rows) |i| {
                const a: u32 = @bitCast(out[i]);
                const b: u32 = @bitCast(tmp[i]);
                if (a != b) {
                    if (bad < 3) std.debug.print("  row {d}: gpu={d} cpu={d}\n", .{ i, out[i], tmp[i] });
                    bad += 1;
                }
            }
            std.debug.print("MVCHECK rows={d} cols={d} mismatched={d}\n", .{ rows, cols, bad });

            const hx = self.alloc.alloc(f32, 8) catch return;
            defer self.alloc.free(hx);
            _ = cu.lib.Memcpy(hx.ptr, self.d_act, 8 * 4, cudart.MemcpyDeviceToHost);
            std.debug.print("ACT x={any}\n    dev={any}\n", .{ x[0..8], hx });
            const nb = q.blocks_per_row;
            const hq = self.alloc.alloc(u8, nb * 34) catch return;
            defer self.alloc.free(hq);
            const cpuq = self.alloc.alloc(u8, nb * 34) catch return;
            defer self.alloc.free(cpuq);
            _ = cu.lib.Memcpy(hq.ptr, self.d_xq, nb * 34, cudart.MemcpyDeviceToHost);
            @import("c_kernels.zig").quantizeQ8_0(cols, x.ptr, cpuq.ptr);
            var diff: usize = 0;
            var first: usize = 0;
            for (0..nb * 34) |i| {
                if (hq[i] != cpuq[i]) {
                    if (diff == 0) first = i;
                    diff += 1;
                }
            }
            std.debug.print("XQCHECK nb={d} diffbytes={d} firstblk={d}\n", .{ nb, diff, first / 34 });
            const blk = first / 34;
            std.debug.print("  gpu={any}\n  cpu={any}\n", .{ hq[blk * 34 ..][0..10], cpuq[blk * 34 ..][0..10] });
        }
        self.stat_matvec_q8 += 1;
        self.stat_mv_bytes += q.data.len;
        if (cudart.getenv("QWEN35_PROF") != null and self.stat_matvec_q8 == 1) {
            // `reportProfile` has no call site on the CLI path and the engine is
            // arena-allocated (deinit never runs), so announce the device once,
            // on the first GPU matvec -- that is the observable proof the kernel
            // ran rather than the CPU fallback.
            self.reportStats("first gpu matvec");
        }
    }

    fn reportStats(self: *CudaBackend, when: []const u8) void {
        if (self.cu == null) return;
        std.debug.print("CUDA-M  [{s}] q8_on_gpu={d} cpu_fallback={d} uploaded_MiB={d} dev_rows={d}\n", .{
            when,
            self.stat_matvec_q8,
            self.stat_matvec_cpu,
            self.stat_mv_bytes >> 20,
            self.dev_rows.items.len,
        });
    }

    fn matvecF32(self: *CudaBackend, out: []f32, m: []f32, x: []const f32) void {
        // f32 weights here are the already-expanded tiny tensors; uploading them
        // per call would cost more than the CPU kernel, so delegate.
        self.cpu.exec().matvec(out, .{ .f32 = m }, x);
        self.stat_matvec_cpu += 1;
    }

    fn fallbackMatvec(self: *CudaBackend, out: []f32, w: backend.LinearW, x: []const f32, e: anyerror) void {
        std.debug.print("cuda backend: device buffer allocation failed ({s}); using CPU for this matvec\n", .{@errorName(e)});
        self.cpu.exec().matvec(out, w, x);
        self.stat_matvec_cpu += 1;
    }

    fn failMatvec(self: *CudaBackend, out: []f32, w: backend.LinearW, x: []const f32, code: c_int, what: []const u8) void {
        const cu = self.cu.?;
        std.debug.print("cuda backend: {s} failed: {s} ({s}); using CPU for this matvec\n", .{
            what,
            std.mem.span(cu.lib.errStr(code)),
            std.mem.span(cu.lib.errName(code)),
        });
        self.cpu.exec().matvec(out, w, x);
        self.stat_matvec_cpu += 1;
    }

    // ---- introspection ------------------------------------------------------

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
                "cuda sm {d}.{d} {s}, {d} MiB free; Q8_0 on device, other types on CPU",
                .{ cu.cc / 10, cu.cc % 10, cu.devName(), cu.free_bytes >> 20 },
            ) catch "cuda";
        }
        return self.init_error;
    }

    // ---- delegating slots ---------------------------------------------------
    //
    // The CPU vtable targets cast `ctx` to `*CpuBackend`, but the engine hands us
    // `&self` (a `*CudaBackend`), so each delegation re-aims `ctx` at the
    // embedded `cpu` field.  Passing `&self` through would rely on `cpu` being
    // the first field -- true today, but a silent-garbage trap the moment
    // anything is reordered.

    /// `allocScratch` is the first thing `Engine.load` calls on a backend, so it
    /// doubles as the place where a failed device init is turned into a load
    /// error: `init()` cannot return an error (the engine's `backendImplFor`
    /// contract is `init(alloc, g) -> Backend`), and silently running the whole
    /// model on the CPU after `--backend cuda` would be a lie.
    fn dAllocScratch(ctx: *anyopaque, max_cols: usize) anyerror!void {
        const self = self_(ctx);
        if (self.cu == null) return self.init_err;
        return cpu_mod.CpuBackend.vtable.allocScratch(@ptrCast(&self.cpu), max_cols);
    }
    fn dAllocScratchK(ctx: *anyopaque, max_cols: usize, k: usize) anyerror!void {
        const self = self_(ctx);
        if (self.cu == null) return self.init_err;
        return cpu_mod.CpuBackend.vtable.allocScratchK(@ptrCast(&self.cpu), max_cols, k);
    }
    fn dFreeScratch(ctx: *anyopaque) void {
        cpu_mod.CpuBackend.vtable.freeScratch(@ptrCast(&self_(ctx).cpu));
    }
    fn dMatvecBatchK(ctx: *anyopaque, out: []f32, w: backend.LinearW, xK: []const f32, k: usize) void {
        const self = self_(ctx);
        self.stat_matvec_cpu += 1;
        cpu_mod.CpuBackend.vtable.matvecBatchK(@ptrCast(&self.cpu), out, w, xK, k);
    }
    fn dSetThreads(ctx: *anyopaque, io: std.Io, nt: usize) void {
        cpu_mod.CpuBackend.vtable.setThreads(@ptrCast(&self_(ctx).cpu), io, nt);
    }
    fn dProfPhase(ctx: *anyopaque, on: bool) void {
        cpu_mod.CpuBackend.vtable.profPhase(@ptrCast(&self_(ctx).cpu), on);
    }
    fn dProfReset(ctx: *anyopaque) void {
        cpu_mod.CpuBackend.vtable.profReset(@ptrCast(&self_(ctx).cpu));
    }
    fn dSetTier(ctx: *anyopaque, tier_name: []const u8) void {
        cpu_mod.CpuBackend.vtable.setTier(@ptrCast(&self_(ctx).cpu), tier_name);
    }
    fn dReportProfile(ctx: *anyopaque) void {
        const self = self_(ctx);
        cpu_mod.CpuBackend.vtable.reportProfile(@ptrCast(&self.cpu));
        if (self.cu != null) {
            std.debug.print("CUDA-M  q8_on_gpu={d} cpu_fallback={d} uploaded_MiB={d} dev_rows={d}\n", .{
                self.stat_matvec_q8,
                self.stat_matvec_cpu,
                self.stat_mv_bytes >> 20,
                self.dev_rows.items.len,
            });
        }
    }

    pub const vtable = backend.Exec.VTable{
        // ported to the device by this stage
        .loadLinear = vLoadLinear,
        .freeLinear = vFreeLinear,
        .dequantRow = vDequantRow,
        .matvec = vMatvec,
        // still CPU (see the delegation note above)
        .allocScratch = dAllocScratch,
        .allocScratchK = dAllocScratchK,
        .freeScratch = dFreeScratch,
        .matvecBatchK = dMatvecBatchK,
        .setThreads = dSetThreads,
        .profPhase = dProfPhase,
        .profReset = dProfReset,
        .setTier = dSetTier,
        .reportProfile = dReportProfile,
        .name = vName,
        .describe = vDescribe,
    };
};
