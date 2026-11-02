//! GPU backend: every numeric kernel in the forward pass runs on the device.
//!
//! Why this file exists at all (rather than a CUDA build of `backend_cpu`): the
//! same `zig-out/bin/qwen35` binary has to run on hosts with no CUDA install.
//! Two different mechanisms keep that true, and they must not be confused:
//!
//!   * **The kernels are linked in.**  `src/kernels/cuda/*.cu` are compiled by
//!     `zig build -Dcuda=true` into objects that are linked into the executable;
//!     every entry point below is a top-level `extern fn q35_launch_*` that the
//!     linker resolves at build time.  There is no kernel `.so`, no `dlopen`, and
//!     no symbol lookup here.  A plain CPU build simply never references them
//!     (nothing calls this backend, see `main.zig`'s `--backend` parsing), and
//!     the objects are not on the link line either.
//!   * **libcudart is dlopen'd** (`cudart.zig`), so a CPU-only host never needs a
//!     DT_NEEDED on it.  That is the only run-time library dependency left, and
//!     `--backend cpu` still works on a host without any CUDA install.
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
const sampler = @import("sampler.zig");

/// Build-time facts injected by `build.zig` (`addOptions("build_options")`).
/// `cuda_linked` is the *same* `-Dcuda` value that gates whether
/// `src/kernels/cuda/*.cu` are put on the executable's link line.
const build_options = @import("build_options");

const mem = std.mem;
const Io = std.Io;

// ==== link-time kernel bindings =============================================
//
// One `extern` per entry point in src/kernels/cuda/api.h, with the header's name
// and signature.  `zig build -Dcuda=true` puts the `.cu` objects on the link
// line, so a missing or mismatched launcher is a **build** error instead of a
// run-time `MissingSymbol` -- which is the whole point of linking them in rather
// than dlopen'ing a `.so`.  Nothing can be rebound at run time, so these
// declarations must track api.h by hand; the comments below restate the header's
// contract so a change in one is visible in the other.
//
// ABI mapping: `int` -> `c_int`, `const void *` -> `?*const anyopaque`,
// `void *` -> `?*anyopaque`, `float *` -> `?[*]f32`, `const float *` ->
// `?[*]const f32`, `unsigned long long` -> `c_ulonglong`.
//
// Call sites go through the `Kernels` namespace below (`const kern = Launch;`)
// purely so the launch sites keep reading like `kern.mul_mat_vec(...)`; it holds
// no state and compiles to a direct call.
const Kernels = struct {
    /// `nblocks` counts 32-ELEMENT blocks (api.h Q35_API_QK8_0).
    extern fn q35_launch_quantize_q8_0(d_x: ?[*]const f32, d_out: ?*anyopaque, nblocks: c_int, stream: ?*anyopaque) callconv(.c) c_int;
    /// `nblocks` counts 256-ELEMENT blocks.
    extern fn q35_launch_quantize_q8_K(d_x: ?[*]const f32, d_out: ?*anyopaque, nblocks: c_int, stream: ?*anyopaque) callconv(.c) c_int;

    /// Whole-tensor dequantization; `nblocks` is in the type's NATIVE blocks.
    extern fn q35_launch_dequantize(ttype: c_int, d_src: ?*const anyopaque, d_dst: ?[*]f32, nblocks: c_int, stream: ?*anyopaque) callconv(.c) c_int;
    /// One row of `cols` elements.  The row offset is device-side pointer
    /// arithmetic by the caller (see `vDequantRow`), so only `cols` is needed.
    extern fn q35_launch_dequantize_row(ttype: c_int, d_src_row: ?*const anyopaque, d_dst: ?[*]f32, cols: c_int, stream: ?*anyopaque) callconv(.c) c_int;

    /// Table-driven quantized matvec: `d_out[t*rows + r] = W[r] . act_t`.
    /// Argument order is api.h's: rows, cols, tokens, **then** ttype.
    extern fn q35_launch_mul_mat_vec(d_w: ?*const anyopaque, d_xq: ?*const anyopaque, d_out: ?[*]f32, rows: c_int, cols: c_int, tokens: c_int, ttype: c_int, stream: ?*anyopaque) callconv(.c) c_int;
    /// Same shape for an f32 weight matrix (embedding / norms).
    extern fn q35_launch_mul_mat_vec_f32(d_w: ?[*]const f32, d_x: ?[*]const f32, d_out: ?[*]f32, rows: c_int, cols: c_int, tokens: c_int, stream: ?*anyopaque) callconv(.c) c_int;

    /// Upload the twelve IQ codebooks to the device (src/kernels/cuda/
    /// iq_tables.cu).  MUST succeed before any kernel that touches an IQ type:
    /// those read the tables through `__device__` pointers that only this call
    /// fills in, so a skipped or failed upload makes IQ matvecs index garbage
    /// (silently wrong output, not a crash).  Idempotent; returns a cudaError_t.
    extern fn q35_launch_init_tables(stream: ?*anyopaque) callconv(.c) c_int;
    /// Release the device copy of the codebooks (process-wide, so it lives as
    /// long as the backend does).
    extern fn q35_launch_deinit_tables() callconv(.c) void;
    /// Bytes of codebook resident on the device (0 before a successful upload).
    extern fn q35_iq_tables_bytes() callconv(.c) c_ulonglong;

    /// Pure elementwise activations (src/kernels/cuda/elementwise.cu).  `n` is an
    /// ELEMENT count of f32, the buffers are contiguous, and all four are
    /// bit-exact with the CPU implementations -- see the file's comment for why
    /// that must stay true.
    extern fn q35_launch_scale_vec(d_x: ?[*]f32, s: f32, n: c_int, stream: ?*anyopaque) callconv(.c) c_int;
    extern fn q35_launch_add_resid(d_dst: ?[*]f32, d_src: ?[*]const f32, n: c_int, stream: ?*anyopaque) callconv(.c) c_int;
    extern fn q35_launch_copy_range(d_dst: ?[*]f32, d_src: ?[*]const f32, n: c_int, stream: ?*anyopaque) callconv(.c) c_int;
    extern fn q35_launch_zero_buf(d_buf: ?[*]f32, n: c_int, stream: ?*anyopaque) callconv(.c) c_int;

    /// Norms (src/kernels/cuda/norm.cu), activations + conv
    /// (src/kernels/cuda/act.cu) and rotary (src/kernels/cuda/rope.cu).
    ///
    /// `rows` blocks of `n` f32; `d_w`/`d_wconv` are length-`n` per-layer
    /// weights reused for every row.  NONE of these is bit-exact with the CPU:
    /// the reductions use a block tree and the transcendentals are the device
    /// runtime's, not glibc's.  Per-kernel measured error is in api.h and in
    /// each file's header.
    extern fn q35_launch_rmsnorm(d_out: ?[*]f32, d_x: ?[*]const f32, d_w: ?[*]const f32, rows: c_int, n: c_int, eps: f32, stream: ?*anyopaque) callconv(.c) c_int;
    extern fn q35_launch_l2norm(d_x: ?[*]f32, rows: c_int, n: c_int, eps: f32, stream: ?*anyopaque) callconv(.c) c_int;
    extern fn q35_launch_silu(d_x: ?[*]f32, n: c_int, stream: ?*anyopaque) callconv(.c) c_int;
    extern fn q35_launch_mul_silu(d_out: ?[*]f32, d_gate: ?[*]const f32, n: c_int, stream: ?*anyopaque) callconv(.c) c_int;
    extern fn q35_launch_mul_sigmoid(d_out: ?[*]f32, d_gate: ?[*]const f32, n: c_int, stream: ?*anyopaque) callconv(.c) c_int;
    extern fn q35_launch_swiglu(d_gate: ?[*]f32, d_up: ?[*]const f32, n: c_int, stream: ?*anyopaque) callconv(.c) c_int;
    extern fn q35_launch_softmax(d_x: ?[*]f32, n: c_int, stream: ?*anyopaque) callconv(.c) c_int;
    extern fn q35_launch_conv_step(d_conv: ?[*]f32, d_y: ?[*]f32, d_wconv: ?[*]const f32, n: c_int, stream: ?*anyopaque) callconv(.c) c_int;
    /// Index of the largest logit; first index wins on ties.
    extern fn q35_launch_argmax(d_x: ?[*]const f32, n: c_int, d_out: ?*c_int, stream: ?*anyopaque) callconv(.c) c_int;

    // P2c: the two state machines.  One launch per value head for gdn_step,
    // one launch (block per head) for attn_decode; see the kernel headers.
    extern fn q35_launch_gdn_step(d_state: ?[*]f32, d_k: ?[*]const f32, d_v: ?[*]const f32, d_q: ?[*]const f32, d_km: ?[*]f32, d_delta: ?[*]f32, d_out: ?[*]f32, gamma: f32, beta: f32, vd: c_int, stream: ?*anyopaque) callconv(.c) c_int;
    extern fn q35_launch_gdn_gamma(d_a_raw: ?[*]const f32, d_b_raw: ?[*]f32, d_ssm_a: ?[*]const f32, d_dt_bias: ?[*]const f32, d_gamma: ?[*]f32, nv: c_int, stream: ?*anyopaque) callconv(.c) c_int;
    extern fn q35_launch_attn_decode(d_q: ?[*]const f32, d_gate: ?[*]const f32, d_kcache: ?[*]const f32, d_vcache: ?[*]const f32, d_att: ?[*]f32, d_gout: ?[*]f32, slot: c_int, heads: c_int, gqa: c_int, hd: c_int, kv_dim: c_int, scale: f32, stream: ?*anyopaque) callconv(.c) c_int;
    /// rope64 over `heads` heads of `hd` floats (only the first 64 rotate).
    extern fn q35_launch_rope64(d_x: ?[*]f32, heads: c_int, hd: c_int, pos: c_int, base: f32, stream: ?*anyopaque) callconv(.c) c_int;
};

// ==== link-time vs. CPU-only build =========================================
//
// The launchers above only exist in a `-Dcuda=true` build (it is that build that
// puts `src/kernels/cuda/*.cu` on the link line).  A plain `zig build` still
// compiles this file -- `engine.zig` instantiates `CudaBackend` for
// `--backend cuda` unconditionally, and the CLI advertises that flag either way
// -- so on a CPU-only build every launch site would be an undefined symbol.
//
// The discriminator is a BUILD-TIME CONSTANT, not a symbol probe: `build.zig`
// passes `-Dcuda` to the Zig code as `build_options.cuda_linked` (the same
// `cuda_enabled` variable that decides whether the nvcc objects are handed to
// `exe.root_module.addObjectFile`).  The `else` branch is a stub that reports
// the situation instead of calling into nothing; the choice is comptime, so a
// CPU-only binary has no dangling call to the launchers either.
//
// Probing for symbol existence does NOT work here, and must not be reintroduced.
// `@hasDecl(Kernels, "q35_launch_mul_mat_vec")` looks like it answers "was this
// linked in?" but it does not: the `fn`s above are *declarations inside the
// `Kernels` struct*, and `@hasDecl` only reports whether that struct has such a
// member.  Measured (2026-10): it is `false` even when the C definition is
// linked into the very same executable, so `Launch` collapsed to the stub branch
// in every build -- `-Dcuda=true` included -- and `--backend cuda` printed "this
// executable was built without CUDA kernels" while `nm` showed all nine
// `T q35_launch_*` present.  A true link-time check would have to ask the
// linker (e.g. a weak reference), which Zig does not expose at comptime; the
// build flag is exact, cheap, and single-sourced with the link itself.
const cuda_linked = build_options.cuda_linked;

/// Reason reported by every stub on a CPU-only build.  Deliberately points at
/// the build flag, because that is the only thing that can fix it.
const no_cuda_kernels_msg = "cuda backend: this executable was built without CUDA kernels (rebuild with -Dcuda=true, or use --backend cpu)";

fn stubNoCudaKernels() c_int {
    std.debug.print("error: {s}\n", .{no_cuda_kernels_msg});
    // Non-zero (cudaErrorInvalidDeviceFunction), so callers' `rc != cudaSuccess`
    // paths fire and surface the message instead of carrying on with garbage.
    return 98;
}

/// Compile-time-resolved launcher table.  `cuda_linked` is comptime-known, so
/// each field collapses to either the extern or the stub -- no indirection at
/// run time.
const Launch = if (cuda_linked) struct {
    const quantize_q8_0 = Kernels.q35_launch_quantize_q8_0;
    const quantize_q8_k = Kernels.q35_launch_quantize_q8_K;
    const dequantize = Kernels.q35_launch_dequantize;
    const dequantize_row = Kernels.q35_launch_dequantize_row;
    const mul_mat_vec = Kernels.q35_launch_mul_mat_vec;
    const mul_mat_vec_f32 = Kernels.q35_launch_mul_mat_vec_f32;
    const init_tables = Kernels.q35_launch_init_tables;
    const deinit_tables = Kernels.q35_launch_deinit_tables;
    const tables_bytes = Kernels.q35_iq_tables_bytes;
    const scale_vec = Kernels.q35_launch_scale_vec;
    const add_resid = Kernels.q35_launch_add_resid;
    const copy_range = Kernels.q35_launch_copy_range;
    const zero_buf = Kernels.q35_launch_zero_buf;
    const rmsnorm = Kernels.q35_launch_rmsnorm;
    const l2norm = Kernels.q35_launch_l2norm;
    const silu = Kernels.q35_launch_silu;
    const mul_silu = Kernels.q35_launch_mul_silu;
    const mul_sigmoid = Kernels.q35_launch_mul_sigmoid;
    const swiglu = Kernels.q35_launch_swiglu;
    const softmax = Kernels.q35_launch_softmax;
    const conv_step = Kernels.q35_launch_conv_step;
    const argmax = Kernels.q35_launch_argmax;
    const attn_decode = Kernels.q35_launch_attn_decode;
    const gdn_step = Kernels.q35_launch_gdn_step;
    const gdn_gamma = Kernels.q35_launch_gdn_gamma;
    const rope64 = Kernels.q35_launch_rope64;
} else struct {
    fn quantize_q8_0(d_x: ?[*]const f32, d_out: ?*anyopaque, nblocks: c_int, stream: ?*anyopaque) callconv(.c) c_int {
        _ = .{ d_x, d_out, nblocks, stream };
        return stubNoCudaKernels();
    }
    fn quantize_q8_k(d_x: ?[*]const f32, d_out: ?*anyopaque, nblocks: c_int, stream: ?*anyopaque) callconv(.c) c_int {
        _ = .{ d_x, d_out, nblocks, stream };
        return stubNoCudaKernels();
    }
    fn dequantize(ttype: c_int, d_src: ?*const anyopaque, d_dst: ?[*]f32, nblocks: c_int, stream: ?*anyopaque) callconv(.c) c_int {
        _ = .{ ttype, d_src, d_dst, nblocks, stream };
        return stubNoCudaKernels();
    }
    fn dequantize_row(ttype: c_int, d_src_row: ?*const anyopaque, d_dst: ?[*]f32, cols: c_int, stream: ?*anyopaque) callconv(.c) c_int {
        _ = .{ ttype, d_src_row, d_dst, cols, stream };
        return stubNoCudaKernels();
    }
    fn mul_mat_vec(d_w: ?*const anyopaque, d_xq: ?*const anyopaque, d_out: ?[*]f32, rows: c_int, cols: c_int, tokens: c_int, ttype: c_int, stream: ?*anyopaque) callconv(.c) c_int {
        _ = .{ d_w, d_xq, d_out, rows, cols, tokens, ttype, stream };
        return stubNoCudaKernels();
    }
    fn mul_mat_vec_f32(d_w: ?[*]const f32, d_x: ?[*]const f32, d_out: ?[*]f32, rows: c_int, cols: c_int, tokens: c_int, stream: ?*anyopaque) callconv(.c) c_int {
        _ = .{ d_w, d_x, d_out, rows, cols, tokens, stream };
        return stubNoCudaKernels();
    }
    fn init_tables(stream: ?*anyopaque) callconv(.c) c_int {
        _ = stream;
        return stubNoCudaKernels();
    }
    fn deinit_tables() callconv(.c) void {
        // Already reported by the failed `init_tables`; stay silent.
    }
    fn tables_bytes() callconv(.c) c_ulonglong {
        return 0;
    }
    fn scale_vec(d_x: ?[*]f32, s: f32, n: c_int, stream: ?*anyopaque) callconv(.c) c_int {
        _ = .{ d_x, s, n, stream };
        return stubNoCudaKernels();
    }
    fn add_resid(d_dst: ?[*]f32, d_src: ?[*]const f32, n: c_int, stream: ?*anyopaque) callconv(.c) c_int {
        _ = .{ d_dst, d_src, n, stream };
        return stubNoCudaKernels();
    }
    fn copy_range(d_dst: ?[*]f32, d_src: ?[*]const f32, n: c_int, stream: ?*anyopaque) callconv(.c) c_int {
        _ = .{ d_dst, d_src, n, stream };
        return stubNoCudaKernels();
    }
    fn zero_buf(d_buf: ?[*]f32, n: c_int, stream: ?*anyopaque) callconv(.c) c_int {
        _ = .{ d_buf, n, stream };
        return stubNoCudaKernels();
    }
    fn rmsnorm(d_out: ?[*]f32, d_x: ?[*]const f32, d_w: ?[*]const f32, rows: c_int, n: c_int, eps: f32, stream: ?*anyopaque) callconv(.c) c_int {
        _ = .{ d_out, d_x, d_w, rows, n, eps, stream };
        return stubNoCudaKernels();
    }
    fn l2norm(d_x: ?[*]f32, rows: c_int, n: c_int, eps: f32, stream: ?*anyopaque) callconv(.c) c_int {
        _ = .{ d_x, rows, n, eps, stream };
        return stubNoCudaKernels();
    }
    fn silu(d_x: ?[*]f32, n: c_int, stream: ?*anyopaque) callconv(.c) c_int {
        _ = .{ d_x, n, stream };
        return stubNoCudaKernels();
    }
    fn mul_silu(d_out: ?[*]f32, d_gate: ?[*]const f32, n: c_int, stream: ?*anyopaque) callconv(.c) c_int {
        _ = .{ d_out, d_gate, n, stream };
        return stubNoCudaKernels();
    }
    fn mul_sigmoid(d_out: ?[*]f32, d_gate: ?[*]const f32, n: c_int, stream: ?*anyopaque) callconv(.c) c_int {
        _ = .{ d_out, d_gate, n, stream };
        return stubNoCudaKernels();
    }
    fn swiglu(d_gate: ?[*]f32, d_up: ?[*]const f32, n: c_int, stream: ?*anyopaque) callconv(.c) c_int {
        _ = .{ d_gate, d_up, n, stream };
        return stubNoCudaKernels();
    }
    fn softmax(d_x: ?[*]f32, n: c_int, stream: ?*anyopaque) callconv(.c) c_int {
        _ = .{ d_x, n, stream };
        return stubNoCudaKernels();
    }
    fn conv_step(d_conv: ?[*]f32, d_y: ?[*]f32, d_wconv: ?[*]const f32, n: c_int, stream: ?*anyopaque) callconv(.c) c_int {
        _ = .{ d_conv, d_y, d_wconv, n, stream };
        return stubNoCudaKernels();
    }
    fn argmax(d_x: ?[*]const f32, n: c_int, d_out: ?*c_int, stream: ?*anyopaque) callconv(.c) c_int {
        _ = .{ d_x, n, d_out, stream };
        return stubNoCudaKernels();
    }
    fn attn_decode(d_q: ?[*]const f32, d_gate: ?[*]const f32, d_kcache: ?[*]const f32, d_vcache: ?[*]const f32, d_att: ?[*]f32, d_gout: ?[*]f32, slot: c_int, heads: c_int, gqa: c_int, hd: c_int, kv_dim: c_int, scale: f32, stream: ?*anyopaque) callconv(.c) c_int {
        _ = .{ d_q, d_gate, d_kcache, d_vcache, d_att, d_gout, slot, heads, gqa, hd, kv_dim, scale, stream };
        return stubNoCudaKernels();
    }
    fn gdn_step(d_state: ?[*]f32, d_k: ?[*]const f32, d_v: ?[*]const f32, d_q: ?[*]const f32, d_km: ?[*]f32, d_delta: ?[*]f32, d_out: ?[*]f32, gamma: f32, beta: f32, vd: c_int, stream: ?*anyopaque) callconv(.c) c_int {
        _ = .{ d_state, d_k, d_v, d_q, d_km, d_delta, d_out, gamma, beta, vd, stream };
        return stubNoCudaKernels();
    }
    fn gdn_gamma(d_a_raw: ?[*]const f32, d_b_raw: ?[*]f32, d_ssm_a: ?[*]const f32, d_dt_bias: ?[*]const f32, d_gamma: ?[*]f32, nv: c_int, stream: ?*anyopaque) callconv(.c) c_int {
        _ = .{ d_a_raw, d_b_raw, d_ssm_a, d_dt_bias, d_gamma, nv, stream };
        return stubNoCudaKernels();
    }
    fn rope64(d_x: ?[*]f32, heads: c_int, hd: c_int, pos: c_int, base: f32, stream: ?*anyopaque) callconv(.c) c_int {
        _ = .{ d_x, heads, hd, pos, base, stream };
        return stubNoCudaKernels();
    }
};

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

/// A read-only per-layer weight (gamma, conv1d kernel) kept resident on the
/// device.  `host`+`bytes` is the identity; see `CudaBackend.dev_const`.
const DevConst = struct {
    host: [*]const f32,
    bytes: usize,
    dev: *anyopaque,
};

/// Resolved kernel entry points.  They are link-time symbols from
/// `src/kernels/cuda/*.cu` (never from cudart), so `Kernels` above is the single
/// place to audit against `src/kernels/cuda/api.h`.
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
/// State of the IQ codebook upload (see `CudaBackend.uploadIqTables`).
/// `failed` is terminal on purpose: the upload is one cudaMalloc + twelve
/// cudaMemcpy + one kernel, so a failure is a device/driver-level problem, not
/// something a retry would fix, and the caller must not proceed to index the
/// (never-written) device pointers.
const IqTablesState = enum { pending, done, failed };

/// The two `dRope64` implementations, only for the startup log line.
const RopePath = enum { host, device };


// Buffers for formatted error/description strings.  Static because
// `describe`/`name` return slices with no allocator in scope (same pattern as
// the CPU backend, which reports through `std.debug.print`).
var msg_buf: [512]u8 = undefined;
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

    /// Whether the IQ codebooks are on the device.  `pending` -> `done` on the
    /// first successful `uploadIqTables`, `failed` if that never happened (and
    /// then stays put: see `uploadIqTables`).
    iq_tables_state: IqTablesState = .pending,

    /// Host data pointer -> device copy.  Keyed by `q.data.ptr` (mmap view) or the
    /// arena slice pointer (f32), so a `LinearW` handed back to us resolves
    /// without threading a cookie through the `Exec` interface.
    dev_rows: std.ArrayList(DevRow) = .empty,
    /// Sum of successful uploads: the resident weight footprint.
    stat_uploaded_bytes: usize = 0,
    /// Bytes moved host->device / device->host on the PER-TOKEN path, i.e. every
    /// `cudaMemcpy` this file issues after the weights are resident.  P2c's
    /// whole point is that this collapses: before it, every elementwise slot
    /// bounced its operands through staging, so each operator cost 2 transfers.
    /// Reported by `$QWEN35_STAT` so the claim is measurable rather than stated.
    stat_h2d_bytes: usize = 0,
    stat_d2h_bytes: usize = 0,
    /// How many times a numeric slot fell back to a host round trip because the
    /// caller passed a *host* `Buf` instead of a device region (`dBufPtr` null).
    /// These are the only two paths that cost a memcpy per call after P2d, and
    /// on the engine's per-token path both must stay at zero -- which is what
    /// makes "no per-token H2D/D2H" an assertion rather than a hope.
    /// `rope64`'s own round trip is counted separately (`stat_rope_trips`).
    stat_host_act_uploads: usize = 0,
    stat_host_out_downloads: usize = 0,
    /// `dRope64` round trips (one H2D + one D2H per rotated head, per token).
    /// The deliberate remainder described at that function: it buys bit-exact
    /// rotation, and the report quantifies it instead of hiding it.  Always 0
    /// under `$QWEN35_CUDA_ROPE=device`.
    stat_rope_trips: usize = 0,
    /// `dRope64` heads rotated by the device kernel.  Its complement to the
    /// total rotation count is `stat_rope_trips`, so the `CUDA-M` line says
    /// which of the two paths actually ran rather than which one was asked for.
    stat_rope_devices: usize = 0,
    /// `$QWEN35_STAT_TALKY` grouping state: last uploaded element count and how
    /// many consecutive uploads had that size.
    debug_h2d_len: usize = 0,
    debug_h2d_same: usize = 0,
    /// `$QWEN35_SYNC=1`: barrier after every matvec, to test whether a result
    /// depends on an unsynchronized producer.  Diagnostic only -- correctness
    /// does not need it (default stream ordering + synchronous memcpy).
    sync_debug: bool = false,
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
    /// Nanoseconds spent per backing type, summed over the same calls that
    /// `stat_by_type` counts.  Diagnostics only ($QWEN35_PROF): answers "which
    /// quant type owns the matvec time", which the count alone cannot.
    stat_ns_by_type: [64]u64 = @splat(0),
    /// Which types the `$QWEN35_MVCHECK` cross-check has already covered (see
    /// `mvCheck`).  Per type, not per process: a model has one type per tensor
    /// family, so a single sample would leave e.g. iq4_xs unchecked.
    mvcheck_seen: [64]bool = @splat(false),
    /// Clock for the slot timings.  Supplied by `setThreads`, which the engine
    /// calls before the first matvec (same contract as `backend_cpu.prof_io`);
    /// only read when `$QWEN35_PROF` output is wanted.
    io: Io = undefined,

    // ---- GPU-side timing ($QWEN35_PROF only) --------------------------------
    //
    // `durNs()` above is host wall clock wrapped around an *asynchronous*
    // launch, so `t_mv_ms` / `t_bk_ms` measure submission, not execution.  The
    // fields here come from CUDA events, which timestamp a point in the stream;
    // `ev_end - ev_begin` is what the device actually ran.
    //
    // Cost model, and why this is PROF-gated: a record is ~1 us, but the
    // *synchronize* needed to read an elapsed time drains the queue.  Measuring
    // every batchk call therefore serializes the whole prefill; the fields below
    // are only maintained when `ev_on` is set, and `ev_on` is only set when
    // `$QWEN35_PROF` was in the environment.  With PROF off, `gpuTimingOn()`
    // returns false before any event is created or touched.
    /// Same condition as every other `$QWEN35_PROF` check in this file.
    ev_on: bool = false,
    /// Created once (`ensureEvents`) and reused, never per token: an
    /// `cudaEventCreateWithFlags` in the hot path would cost more than the
    /// kernels being measured.
    ev_created: bool = false,
    /// Around one `matvecBatchK` call (one weight, one chunk of K tokens).
    ev_bk_a: ?*anyopaque = null,
    ev_bk_b: ?*anyopaque = null,
    /// Around the whole prefill window: `profReset` -> first `matvecBatchK` of
    /// the decode phase.  The wall clock version of the same window is
    /// `stat_ns_prefill_wall`, so the difference is the *host* gap that no
    /// kernel covers.
    ev_pre_a: ?*anyopaque = null,
    ev_pre_b: ?*anyopaque = null,
    /// True between `profReset` and the event read that closes the prefill
    /// window; decides whether a `matvecBatchK` call is prefill or decode.
    prefill_open: bool = false,
    /// `ev_pre_a` has been recorded (the window's first device position).
    ev_pre_started: bool = false,
    /// Host clock origin of the prefill window (`openPrefillWindow`).
    prefill_t0: Io.Clock.Timestamp = undefined,
    /// Host clock origin of the batchk span inside the prefill window.
    bk_wall_started: bool = false,
    bk_wall_t0: Io.Clock.Timestamp = undefined,
    /// GPU kernel time summed over every timed `matvecBatchK` call, in ns.  The
    /// span recorded is the call's first recorded point to its last, so it
    /// covers the activation quantize kernels *and* `mul_mat_vec` -- i.e. all
    /// the device work that call issued (not just the matmul).
    stat_ns_batchk_gpu: u64 = 0,
    /// Number of batchk calls that contributed to `stat_ns_batchk_gpu`.
    stat_n_batchk_gpu: usize = 0,
    /// Device event span of the whole prefill window (first prefill record to
    /// the event that closes it), in ns.  This is a *span*, not busy time: it
    /// includes device idle while the host is submitting, so it should match
    /// `stat_ns_prefill_wall` and is reported as a consistency check.
    stat_ns_prefill_span: u64 = 0,
    /// Host wall clock of that same prefill window, in ns.
    stat_ns_prefill_wall: u64 = 0,
    /// Host wall clock from the first to the last `matvecBatchK` of prefill:
    /// `wall - batchk_gpu` is then the host-side slack (launch gaps, the
    /// quantize launches' submission, the synchronous memcpys).
    stat_ns_prefill_bk_wall: u64 = 0,
    /// `cudaEventElapsedTime` resolution guard: number of reads that returned a
    /// cudart error (normally 0).
    stat_n_ev_err: usize = 0,
    /// Kernel launches issued by the batchk path during prefill (K quantizes +
    /// one mul_mat_vec per call).  A single `+= 1` per launch, so counting it
    /// costs nothing measurable and works with or without events -- which is
    /// what makes the launch *rate* comparable between a PROF run and a plain
    /// one.
    stat_n_launch_bk: usize = 0,
    /// Host time spent inside `cudaEventSynchronize` while measuring.  This IS
    /// the interference the events add, so it is reported rather than assumed
    /// small: an event read drains the queue, so every timed call waits for its
    /// own kernels before returning.
    stat_ns_ev_sync: u64 = 0,

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
    /// Single `int` cell the argmax kernel writes its index into.  It has to be
    /// device memory (the launcher stores through `d_out`), and it is the only
    /// per-token download the greedy path makes: 4 bytes.
    d_idx: ?*anyopaque = null,
    d_idx_bytes: usize = 0,
    /// Every region `dAllocBuf` handed to the engine (activations, KV cache,
    /// GDN state).  The `Buf` values themselves are the engine's; this list is
    /// only so `freeScratch` / `deinit` can release them, since a `Buf` carries
    /// no ownership flag.
    dev_bufs: std.ArrayList(*anyopaque) = .empty,
    /// Resident copies of the read-only per-layer weights the new elementwise
    /// slots consume (`rmsnorm`'s gamma, `convStep`'s conv1d kernel).  Their
    /// host buffers (`lw.attn_norm`, `lw.ssm_conv1d`, ...) are model metadata
    /// that lives for the whole run and is re-passed on EVERY token, so an H2D
    /// per call would be pure waste -- and for a 0.8B model it is not
    /// negligible: 18 layers x (1024 + 6144) f32 = 516 KB per token at
    /// 1000 tok/s = 0.5 GB/s of PCIe for data that never changes.
    ///
    /// Keyed on (host pointer, byte length), see `devConst`: the same page can
    /// legitimately be asked for at two different lengths by two layers
    /// sharing a weight buffer, and a longer request over an existing shorter
    /// region must not be satisfied with the short copy.
    ///
    /// LIFETIME: allocated on first use, freed in `dFreeScratch` along with the
    /// other scratch.  NOT freed per layer and not evicted under pressure --
    /// the set is tiny (one entry per distinct weight buffer, ~20 for a 27B
    /// model) and the alternative (an LRU) would cost more than it saves.
    /// P2c: entries kept; the key changes from "host ptr" to "device ptr of
    /// the already-resident tensor", i.e. `devConst` collapses to a lookup.
    dev_const: std.ArrayList(DevConst) = .empty,


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
        self.sync_debug = cudart.getenv("QWEN35_SYNC") != null;

        // The launchers themselves need no setup: they are link-time symbols, so
        // "the kernels are there" is enforced by the build.  The one run-time
        // precondition left is the IQ codebook upload, which `allocScratch`
        // performs (see `uploadIqTables`) because it is the first hook the engine
        // calls after this.
        return self;
    }

    pub fn exec(self: *CudaBackend) backend.Exec {
        return .{ .ptr = self, .vtable = &vtable };
    }

    pub fn deinit(self: *CudaBackend) void {
        self.reportStats("deinit");
        if (self.cu) |*cu| {
            // Events first: they belong to the context that is about to be
            // released, and each one was created exactly once in
            // `ensureEvents`, so this is the matching destroy.
            self.freeEvents();
            for (self.dev_rows.items) |e| _ = cu.lib.Free(e.dev);
            if (self.d_act) |p| _ = cu.lib.Free(p);
            if (self.d_xq) |p| _ = cu.lib.Free(p);
            if (self.d_out) |p| _ = cu.lib.Free(p);
            if (self.d_deq) |p| _ = cu.lib.Free(p);
            for (self.dev_bufs.items) |p| _ = cu.lib.Free(p);
            self.dev_bufs.clearRetainingCapacity();
            for (self.dev_const.items) |e| _ = cu.lib.Free(e.dev);
            self.dev_const.clearRetainingCapacity();
            self.d_act = null;
            self.d_xq = null;
            self.d_out = null;
            self.d_deq = null;
            // The codebook buffer is process-wide (it lives in the linked-in
            // iq_tables.cu, not in this struct), so release it here and reset the
            // flag: that re-arms the one-shot upload for any backend created
            // later in this process.
            if (self.iq_tables_state == .done) {
                Launch.deinit_tables();
                self.iq_tables_state = .pending;
            }
            cu.lib.close();
            self.cu = null;
        }
        self.dev_rows.deinit(self.alloc);
        self.dev_bufs.deinit(self.alloc);
        self.cpu.deinit();
    }

    // ==== device plumbing ===================================================

    fn self_(ctx: *anyopaque) *CudaBackend {
        return @ptrCast(@alignCast(ctx));
    }

    /// True when the cudart runtime is up.  The kernels need no analogous check:
    /// they are link-time symbols, so their presence was settled by the build.
    /// False (e.g. `--backend cuda` on a host without a GPU) is a hard error
    /// reported from `allocScratch`, which is why no slot needs a CPU fallback.
    fn gpuReady(self: *CudaBackend) bool {
        return self.cu != null;
    }

    fn needHelp(self: *CudaBackend) []const u8 {
        if (!self.gpuReady()) return self.init_error;
        return "cuda backend: kernel launch failed";
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

    /// `$QWEN35_CUDA_ROPE=host|device`: which implementation `dRope64` uses.
    /// Default `device` (zero-copy `q35_launch_rope64`); `host` is the escape
    /// hatch back to the CPU body, which is bit-exact against the reference but
    /// pays one D2H+H2D pair per head per layer per token.
    ///
    /// Measured on Qwen3.8-27B IQ3_XXS / 4090, 354-token prompt, 3 runs each
    /// (`scripts/bench.sh` style A/B, `$QWEN35_CUDA_ROPE` the only variable):
    /// prefill 10.20-10.54 -> 11.35-11.41 tok/s (+10.6%), generate 7.15-7.35 ->
    /// 7.48-7.50 tok/s (+4.4%).  The gain is smaller on generate because the
    /// per-head staging is amortised differently there, and it is NOT a scaling
    /// claim: as the matvec kernels improve, rope's share grows and so will this.
    ///
    /// This does NOT preserve host/device token identity, and is not meant to.
    /// The device libm's `powf` differs from glibc's by ~1 ULP, which is an
    /// absolute phase error of order `pos * 1.2e-7` rad; 65 layers of q/k
    /// rotation amplify it past the argmax margin well before pos 1000 (measured:
    /// first divergence at generated token 34 with a 17-token prompt, pos ~50;
    /// from token 1 with a 354-token prompt).  Each path is still deterministic
    /// on its own (3 runs, identical md5), so A/B debugging stays possible --
    /// what is lost is cross-path comparability, the same thing llama.cpp's own
    /// CPU-vs-CUDA backends do not offer either.
    ///
    /// Read once and cached: this is on the per-token path, and `getenv` per
    /// head per layer per token is exactly the kind of syscall this backend
    /// exists to avoid.  Any other value (including unset) keeps the default.
    fn ropeOnDevice() bool {
        const S = struct {
            var cached: ?bool = null;
        };
        if (S.cached) |v| return v;
        var on = true;
        if (cudart.getenv("QWEN35_CUDA_ROPE")) |raw| {
            const v = mem.trim(u8, raw, " \t\r\n");
            if (mem.eql(u8, v, "host")) {
                on = false;
            } else if (!mem.eql(u8, v, "device")) {
                std.debug.print(
                    "cuda backend: QWEN35_CUDA_ROPE='{s}' is neither 'host' nor 'device'; using device\n",
                    .{v},
                );
            }
            std.debug.print("cuda backend: rope64 = {s} ({s})\n", .{
                @tagName(if (on) RopePath.device else RopePath.host), "QWEN35_CUDA_ROPE",
            });
        }
        S.cached = on;
        return on;
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
        self.stat_h2d_bytes += bytes;
        self.stat_host_act_uploads += 1;
        if (rc != cudart.cudaSuccess) {
            std.debug.print("cuda backend: activation upload failed: {s}\n", .{std.mem.span(cu.lib.errStr(rc))});
        }
    }

    fn downloadOut(self: *CudaBackend, out: []f32) void {
        const cu = &self.cu.?;
        const bytes = out.len * @sizeOf(f32);
        const rc = cu.lib.Memcpy(out.ptr, self.d_out, bytes, cudart.MemcpyDeviceToHost);
        self.stat_d2h_bytes += bytes;
        self.stat_host_out_downloads += 1;
        if (rc != cudart.cudaSuccess) {
            std.debug.print("cuda backend: result download failed: {s}\n", .{std.mem.span(cu.lib.errStr(rc))});
        }
    }

    /// Quantize one token's activation into `self.d_xq`, in the format the weight
    /// type needs (Q8_0 for 32-element-block weights, Q8_K otherwise).  Returns the
    /// byte count written.
    ///
    /// The activation is read straight from a device pointer, which after P2d is
    /// normally the caller's own resident buffer (`xBuf`), not a `d_act` staging
    /// area -- staging is only written when a caller hands over a host slice.
    fn quantizeActFrom(self: *CudaBackend, d_src: [*]const f32, ttype: gguf.TensorType, cols: usize) !usize {
        const cu = &self.cu.?;
        const nb = cols / actBlockElems(ttype);
        std.debug.assert(nb * actBlockElems(ttype) == cols);
        const kern = Launch;
        const rc = if (actBlockElems(ttype) == gguf.blockElems(.q8_0))
            kern.quantize_q8_0(d_src, self.d_xq.?, @intCast(nb), null)
        else
            kern.quantize_q8_k(d_src, self.d_xq.?, @intCast(nb), null);
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
    /// embedding table is never re-uploaded per token.  The `cols` f32 result goes
    /// straight into `dstBuf`'s device region -- P2d removed the `downloadOut`
    /// that used to push every embedding row and lm_head row back through the
    /// host.  Buffers are the shared pair, grown by `ensureBuf`, so a repeated
    /// call does no `cudaMalloc`.
    ///
    /// The engine calls this once per row (token embedding, lm_head), so it is a
    /// latency path rather than a throughput one.
    fn vDequantRow(ctx: *anyopaque, dstBuf: backend.Buf, w: gguf.QuantRow, row: usize) void {
        const self = self_(ctx);
        if (!self.gpuReady()) {
            std.debug.print("cuda backend: dequantRow: {s}\n", .{self.needHelp()});
            self.clearDevBuf(dstBuf);
            return;
        }
        const e = self.findDev(w.data.ptr) orelse {
            // Only reachable for a weight never loaded through this backend.  Say
            // so rather than compute a plausible-looking row on the CPU -- exactly
            // the class of silent divergence in AGENTS.md §5.2.
            std.debug.print("cuda backend: dequantRow: a '{s}' weight is not resident on the device\n", .{@tagName(w.ttype)});
            self.clearDevBuf(dstBuf);
            return;
        };
        const cu = &self.cu.?;
        const cols = w.cols();
        const row_bytes = w.rowBytes();
        std.debug.assert(dstBuf.len >= cols);
        std.debug.assert(row < w.rows);

        const out_bytes = cols * @sizeOf(f32);
        // Destination: the caller's own device region when it has one (the normal
        // case), otherwise the shared `d_out` plus the one download that a host
        // caller still needs.
        const dst_dev: ?[*]f32 = dBufPtr(dstBuf);
        const d_out: [*]f32 = dst_dev orelse blk: {
            ensureBuf(&cu.lib, &self.d_out, &self.d_out_bytes, out_bytes) catch {
                std.debug.print("cuda backend: dequantRow: cannot allocate {d} B of device output\n", .{out_bytes});
                self.clearDevBuf(dstBuf);
                return;
            };
            break :blk @ptrCast(@alignCast(self.d_out));
        };

        const t0 = Io.Clock.Timestamp.now(self.io, .awake);
        // Kernel entry points are link-time externs, so the "table" is just the
        // namespace of those declarations -- see `Kernels` at the top of the file.
        const kern = Launch;
        const src: [*]const u8 = @ptrCast(e.dev);
        const d_row = src + row * row_bytes;
        const rc = kern.dequantize_row(
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
            self.clearDevBuf(dstBuf);
            return;
        }
        if (dst_dev == null) {
            const rc2 = cu.lib.Memcpy(dstBuf.slice().ptr, d_out, out_bytes, cudart.MemcpyDeviceToHost);
            if (rc2 != cudart.cudaSuccess) {
                std.debug.print("cuda backend: dequantRow download failed: {s}\n", .{std.mem.span(cu.lib.errStr(rc2))});
                @memset(dstBuf.slice(), 0);
                return;
            }
            // Only a caller that handed us host memory gets a D2H here; the
            // engine's embedding / lm_head buffers are device regions, so this
            // counter staying at 0 is the proof the rows never came back.
            self.stat_d2h_bytes += out_bytes;
            self.stat_host_out_downloads += 1;
        }
        self.stat_ns_dequant += durNs(t0, self.io);
        self.stat_n_dequant += 1;
    }

    // ==== slot: matvec ======================================================

    /// `out = W * x` entirely on the device: quantize x in place (or, for the
    /// one legitimate host caller, upload it once), table-driven matvec, result
    /// written straight into `outBuf`'s device region.
    ///
    /// P2d: no `slice()` and no `downloadOut` on either leg.  `xBuf` / `outBuf`
    /// are backend regions (`dAllocBuf`), so the kernel reads and writes them
    /// directly; the only host->device copy left is `uploadAct`, reached solely
    /// when a caller genuinely hands over a host slice (`dBufPtr` null).
    /// `ttype` only chooses the activation format and quantizer -- never a
    /// backend, and never a CPU branch.
    fn vMatvec(ctx: *anyopaque, outBuf: backend.Buf, w: backend.LinearW, xBuf: backend.Buf) void {
        const self = self_(ctx);
        if (!self.gpuReady()) {
            std.debug.print("cuda backend: matvec: {s}\n", .{self.needHelp()});
            self.clearDevBuf(outBuf);
            return;
        }
        const cu = &self.cu.?;
        const t0 = Io.Clock.Timestamp.now(self.io, .awake);

        // Device activation pointer: the resident buffer when the call site is
        // already device-side (the whole per-token path), otherwise a one-shot
        // upload into `d_act` (first use of a host-only tensor).
        const d_x: [*]const f32 = blk: {
            if (dBufPtr(xBuf)) |p| break :blk @ptrCast(p);
            if (xBuf.len != 0) self.uploadAct(xBuf.slice());
            break :blk @ptrCast(@alignCast(self.d_act.?));
        };

        switch (w) {
            .f32 => |m| {
                const e = self.findDev(mem.sliceAsBytes(m).ptr) orelse {
                    std.debug.print("cuda backend: matvec: f32 weight is not resident on the device\n", .{});
                    self.clearDevBuf(outBuf);
                    return;
                };
                const rows = e.rows;
                const cols = e.cols;
                std.debug.assert(xBuf.len >= cols and outBuf.len >= rows);
                if (dBufPtr(outBuf)) |pd| {
                    const rc = Launch.mul_mat_vec_f32(
                        @ptrCast(@alignCast(e.dev)),
                        d_x,
                        pd,
                        @intCast(rows),
                        @intCast(cols),
                        1,
                        null,
                    );
                    if (rc != cudart.cudaSuccess) {
                        std.debug.print("cuda backend: mul_mat_vec_f32 launch failed: {s}\n", .{std.mem.span(cu.lib.errStr(rc))});
                        self.clearDevBuf(outBuf);
                        return;
                    }
                } else {
                    ensureBuf(&cu.lib, &self.d_out, &self.d_out_bytes, rows * @sizeOf(f32)) catch {
                        std.debug.print("cuda backend: matvec: device output allocation failed\n", .{});
                        self.clearDevBuf(outBuf);
                        return;
                    };
                    const rc = Launch.mul_mat_vec_f32(
                        @ptrCast(@alignCast(e.dev)),
                        d_x,
                        @ptrCast(@alignCast(self.d_out)),
                        @intCast(rows),
                        @intCast(cols),
                        1,
                        null,
                    );
                    if (rc != cudart.cudaSuccess) {
                        std.debug.print("cuda backend: mul_mat_vec_f32 launch failed: {s}\n", .{std.mem.span(cu.lib.errStr(rc))});
                        self.clearDevBuf(outBuf);
                        return;
                    }
                    self.downloadOut(outBuf.slice()[0..rows]);
                }
                self.stat_ns_matvec += durNs(t0, self.io);
                self.stat_by_type[@backingInt(gguf.TensorType.f32)] += 1;
                self.stat_n_matvec += 1;
                return;
            },
            .quant => |q| {
                const e = self.findDev(q.data.ptr) orelse {
                    std.debug.print("cuda backend: matvec: a '{s}' weight is not resident on the device\n", .{@tagName(q.ttype)});
                    self.clearDevBuf(outBuf);
                    return;
                };
                const cols = q.cols();
                const rows = q.rows;
                const xq_bytes = actBytes(q.ttype, cols);
                std.debug.assert(xBuf.len >= cols and outBuf.len >= rows);

                ensureBuf(&cu.lib, &self.d_xq, &self.d_xq_bytes, xq_bytes) catch {
                    std.debug.print("cuda backend: matvec: device activation allocation failed ({d} B)\n", .{xq_bytes});
                    self.clearDevBuf(outBuf);
                    return;
                };
                _ = self.quantizeActFrom(d_x, q.ttype, cols) catch {
                    self.clearDevBuf(outBuf);
                    return;
                };
                if (dBufPtr(outBuf)) |pd| {
                    // api.h order: (d_w, d_xq, d_out, rows, cols, tokens, ttype, stream).
                    const rc = Launch.mul_mat_vec(
                        e.dev,
                        self.d_xq,
                        pd,
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
                        self.clearDevBuf(outBuf);
                        return;
                    }
                } else {
                    ensureBuf(&cu.lib, &self.d_out, &self.d_out_bytes, rows * @sizeOf(f32)) catch {
                        std.debug.print("cuda backend: matvec: device output allocation failed\n", .{});
                        self.clearDevBuf(outBuf);
                        return;
                    };
                    const rc = Launch.mul_mat_vec(
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
                        self.clearDevBuf(outBuf);
                        return;
                    }
                    self.downloadOut(outBuf.slice()[0..rows]);
                }
                self.stat_ns_matvec += durNs(t0, self.io);
                self.stat_by_type[@backingInt(q.ttype)] += 1;
                self.stat_ns_by_type[@backingInt(q.ttype)] += durNs(t0, self.io);
                self.stat_n_matvec += 1;
                if (cudart.getenv("QWEN35_PROF") != null and self.stat_n_matvec % 500 == 0) {
                    self.reportStats("matvec");
                }
                if (cudart.getenv("QWEN35_PROF") != null and self.stat_n_matvec == 1) {
                    // `reportProfile` has no call site on the CLI path and the engine
                    // is arena-allocated (so `deinit` never runs).  Announce the device
                    // once, on the first GPU matvec: the observable proof the kernel
                    // ran instead of a host fallback.
                    self.reportStats("first gpu matvec");
                }
                self.mvCheck(outBuf, q, xBuf);
            },
        }
        // `$QWEN35_SYNC`: strictly serialize after every matvec.  Purely a
        // diagnostic for the coherency question -- if a result ever depended on
        // an unsynchronized producer, inserting this barrier makes the bug
        // appear/disappear deterministically.  Not needed for correctness:
        // every launch and every copy in this file is on the default stream,
        // which orders them, and each `cudaMemcpy` is the synchronous form, so
        // it blocks until all prior work finished.
        if (self.sync_debug) _ = self.cu.?.lib.DeviceSynchronize();
    }

    /// `$QWEN35_MVCHECK`: **debug-only cross-check, not a computation path**.
    ///
    /// This is the only place in the file that calls the CPU matvec.  It is
    /// limited to the first matvec of a run, behind an env var, purely to compare
    /// device and host results (the class of bug in AGENTS.md §5.2: a kernel that
    /// is silently wrong for one quant type).  `outBuf` already holds the device
    /// result and is never overwritten; pulling it down for the comparison is
    /// the diagnostic's own cost (only with `QWEN35_MVCHECK` set).
    fn mvCheck(self: *CudaBackend, outBuf: backend.Buf, q: gguf.QuantRow, xBuf: backend.Buf) void {
        self.mvCheckVariant(outBuf, q, xBuf, false, 1);
    }

    /// `$QWEN35_MVCHECK` for the batch-K path: same comparison, CPU reference is
    /// `matvecBatchK` with the same K, so a per-token mismatch inside the chunk is
    /// visible instead of hidden behind token 0.
    fn mvCheckVariant(self: *CudaBackend, outBuf: backend.Buf, q: gguf.QuantRow, xBuf: backend.Buf, batchk: bool, k: usize) void {
        if (cudart.getenv("QWEN35_MVCHECK") == null) return;
        // `$QWEN35_MVCHECK_ONLY=iq1_s` restricts the check to one type: the
        // batch-K reference needs the CPU scratch shaped for `matvecBatchK`, and
        // not every type survives that in every launch order, so a targeted run
        // is the useful one when hunting a single kernel.
        if (cudart.getenv("QWEN35_MVCHECK_ONLY")) |only| {
            if (!std.mem.eql(u8, only, @tagName(q.ttype))) return;
        }
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
        // `allocScratch` sizes one token; a batch-K launch needs K activation
        // slots, so grow them here (same `q8ScratchBytes` / `q8kScratchBytes`
        // arithmetic as `backend_cpu.vAllocScratch`, through the public sizes).
        if (batchk and k > 1) {
            self.cpu.q8K = self.alloc.alloc(u8, k * cpu_mod.q8ScratchBytes(cols)) catch return;
            self.cpu.q8kK = self.alloc.alloc(u8, k * cpu_mod.q8kScratchBytes(cols)) catch return;
        }
        // `dev` holds the device result, `cput` the CPU reference.
        const dev = self.alloc.alloc(f32, outBuf.len) catch return;
        defer self.alloc.free(dev);
        // The device result is pulled down for the comparison only: MVCHECK is a
        // debug switch, never part of the compute path (and it is what remains of
        // this file's D2H traffic when it is set).
        if (dBufPtr(outBuf)) |pd| {
            const rc = self.cu.?.lib.Memcpy(
                @ptrCast(dev.ptr),
                @ptrCast(pd),
                dev.len * @sizeOf(f32),
                cudart.MemcpyDeviceToHost,
            );
            if (rc != cudart.cudaSuccess) return;
            self.stat_d2h_bytes += dev.len * @sizeOf(f32);
        } else @memcpy(dev, outBuf.slice());

        // Reference activations: device-resident in the normal case, so they have
        // to come down too -- the CPU matvec can only read host memory.
        const x_dev = dBufPtr(xBuf) != null;
        const x = self.alloc.alloc(f32, xBuf.len) catch return;
        defer self.alloc.free(x);
        if (x_dev) {
            const rc = self.cu.?.lib.Memcpy(
                @ptrCast(x.ptr),
                @ptrCast(xBuf.ptr),
                x.len * @sizeOf(f32),
                cudart.MemcpyDeviceToHost,
            );
            if (rc != cudart.cudaSuccess) return;
            self.stat_d2h_bytes += x.len * @sizeOf(f32);
        } else @memcpy(x, xBuf.slice());

        const cput = self.alloc.alloc(f32, outBuf.len) catch return;
        defer self.alloc.free(cput);
        if (batchk) cpu.matvecBatchK(backend.Buf.wrap(cput), .{ .quant = q }, backend.Buf.wrap(x), k) else cpu.matvec(backend.Buf.wrap(cput), .{ .quant = q }, backend.Buf.wrap(x));
        // NOT `cpu.exec()` above: that is the diagnostic's OWN backend, whose
        // scratch is sized here, one token wide -- `matvecBatchK` wants K slots.
        // Those CPUs are never fully initialised (`setThreads` etc.), so this runs
        // serially (`pool = null`) instead.
        var bad: usize = 0;
        for (dev, 0..) |a, i| {
            // Relative tolerance, not bit equality: the device kernel may reorder
            // the accumulation, so an exact match is not expected even when
            // correct.
            const denom = @max(@abs(cput[i]), 1e-3);
            if (@abs(a - cput[i]) / denom > 2e-2) {
                if (bad < 3) std.debug.print("  MVCHECK row: dev={d} cpu={d}\n", .{ a, cput[i] });
                bad += 1;
            }
        }
        std.debug.print("MVCHECK[{s}] rows={d} cols={d} mismatched={d} batchk={} k={d}\n", .{
            @tagName(q.ttype), q.rows, q.cols(), bad, batchk, k,
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
    fn vMatvecBatchK(ctx: *anyopaque, outBuf: backend.Buf, w: backend.LinearW, xKBuf: backend.Buf, k: usize) void {
        const self = self_(ctx);
        if (!self.gpuReady()) {
            std.debug.print("cuda backend: matvecBatchK: {s}\n", .{self.needHelp()});
            self.clearDevBuf(outBuf);
            return;
        }
        const cu = &self.cu.?;
        const t0 = Io.Clock.Timestamp.now(self.io, .awake);

        // Which device output to write: the caller's own region (normal) or the
        // shared `d_out` when the caller handed us host memory.
        const out_dev: ?[*]f32 = dBufPtr(outBuf);

        switch (w) {
            .f32 => |m| {
                const e = self.findDev(mem.sliceAsBytes(m).ptr) orelse {
                    std.debug.print("cuda backend: matvecBatchK: f32 weight is not resident on the device\n", .{});
                    self.clearDevBuf(outBuf);
                    return;
                };
                const rows = e.rows;
                const cols = e.cols;
                std.debug.assert(xKBuf.len >= k * cols and outBuf.len >= k * rows);
                // `$QWEN35_PROF`: same device-side clock as the quant path.  The
                // f32 expansion is only used for f16/bf16 weights, so on a
                // quantized model this branch never runs.
                if (self.gpuTimingOn()) self.batchKTimingBegin();
                const act_bytes = k * cols * @sizeOf(f32);
                const d_x: [*]const f32 = blk: {
                    if (dBufPtr(xKBuf)) |p| break :blk @ptrCast(p);
                    ensureBuf(&cu.lib, &self.d_act, &self.d_act_bytes, act_bytes) catch return oomMsg("batchK activation");
                    const rc0 = cu.lib.Memcpy(self.d_act, xKBuf.slice().ptr, act_bytes, cudart.MemcpyHostToDevice);
                    if (rc0 != cudart.cudaSuccess) {
                        std.debug.print("cuda backend: batchK f32 activation upload failed: {s}\n", .{std.mem.span(cu.lib.errStr(rc0))});
                        self.clearDevBuf(outBuf);
                        return;
                    }
                    self.stat_h2d_bytes += act_bytes;
                    break :blk @ptrCast(@alignCast(self.d_act.?));
                };
                var d_out: [*]f32 = undefined;
                if (out_dev) |p| {
                    d_out = p;
                } else {
                    ensureBuf(&cu.lib, &self.d_out, &self.d_out_bytes, k * rows * @sizeOf(f32)) catch return oomMsg("batchK output");
                    d_out = @ptrCast(@alignCast(self.d_out.?));
                }
                const rc = Launch.mul_mat_vec_f32(
                    @ptrCast(@alignCast(e.dev)),
                    d_x,
                    d_out,
                    @intCast(rows),
                    @intCast(cols),
                    @intCast(k),
                    null,
                );
                if (rc != cudart.cudaSuccess) {
                    std.debug.print("cuda backend: mul_mat_vec_f32(K={d}) launch failed: {s}\n", .{
                        k, std.mem.span(cu.lib.errStr(rc)),
                    });
                    self.clearDevBuf(outBuf);
                    return;
                }
                if (out_dev == null) self.downloadOut(outBuf.slice()[0 .. k * rows]);
                self.stat_ns_batchk += durNs(t0, self.io);
                self.stat_n_batchk += 1;
                if (self.gpuTimingOn()) self.batchKTimingEnd();
                return;
            },
            .quant => |q| {
                const e = self.findDev(q.data.ptr) orelse {
                    std.debug.print("cuda backend: matvecBatchK: a '{s}' weight is not resident on the device\n", .{@tagName(q.ttype)});
                    self.clearDevBuf(outBuf);
                    return;
                };
                const cols = q.cols();
                const rows = q.rows;
                const per_tok = actBytes(q.ttype, cols);
                const nb_tok = cols / actBlockElems(q.ttype);
                std.debug.assert(xKBuf.len >= k * cols and outBuf.len >= k * rows);
                // `$QWEN35_PROF`: start the device-side clock for this call.
                // Must run before the H2D below, so a synchronous upload is
                // counted as host slack rather than as GPU work; the recorded
                // point is ordered after any earlier queued kernel anyway.
                if (self.gpuTimingOn()) self.batchKTimingBegin();

                // Activation source: the resident K-row buffer when the engine
                // already keeps it on the device (the normal case after P2d),
                // otherwise one H2D for all K rows.
                const base_act: [*]const f32 = blk: {
                    if (dBufPtr(xKBuf)) |p| break :blk @ptrCast(p);
                    const act_bytes = k * cols * @sizeOf(f32);
                    ensureBuf(&cu.lib, &self.d_act, &self.d_act_bytes, act_bytes) catch return oomMsg("batchK activation");
                    const rc0 = cu.lib.Memcpy(self.d_act, xKBuf.slice().ptr, act_bytes, cudart.MemcpyHostToDevice);
                    if (rc0 != cudart.cudaSuccess) {
                        std.debug.print("cuda backend: batchK activation upload failed: {s}\n", .{std.mem.span(cu.lib.errStr(rc0))});
                        self.clearDevBuf(outBuf);
                        return;
                    }
                    self.stat_h2d_bytes += act_bytes;
                    break :blk @ptrCast(@alignCast(self.d_act.?));
                };
                ensureBuf(&cu.lib, &self.d_xq, &self.d_xq_bytes, per_tok * k) catch return oomMsg("batchK quantized activation");
                var d_out: [*]f32 = undefined;
                if (out_dev) |p| {
                    d_out = p;
                } else {
                    ensureBuf(&cu.lib, &self.d_out, &self.d_out_bytes, k * rows * @sizeOf(f32)) catch return oomMsg("batchK output");
                    d_out = @ptrCast(@alignCast(self.d_out.?));
                }

                // Quantize ALL k tokens in ONE launch.
                //
                // The k activation rows are contiguous in `base_act` (k*cols
                // f32) and the destination slots are contiguous in `d_xq`
                // (k*per_tok bytes, `per_tok == nb_tok * actBlockBytes`), so
                // the whole batch is one run of `k * nb_tok` blocks with the
                // SAME stride the quantizers already use -- no per-token offset
                // arithmetic on either side.  Both device quantizers
                // (q35_k_quantize_q8_K in mmvq.cu, q35_k_quantize_q8_0 in
                // convert.cu) are one-thread-per-block with `blockIdx`-derived
                // indices and read nothing outside `[0, nblocks)`, so this is
                // bit-exact vs. the old one-launch-per-token loop.
                const base_xq: [*]u8 = @ptrCast(self.d_xq.?);
                const q8_0_act = actBlockElems(q.ttype) == gguf.blockElems(.q8_0);
                const kern = Launch;
                // `k >= 2` on this path (engine.forwardChunk routes k == 1 to
                // the serial forward) and `cols >= 1` with cols a multiple of
                // actBlockElems, so `total_blocks >= 2`: the merged count can
                // never be 0 and can never be smaller than the per-token one.
                const total_blocks = nb_tok * k;
                std.debug.assert(k >= 2 and nb_tok >= 1);
                const rc = if (q8_0_act)
                    kern.quantize_q8_0(base_act, @ptrCast(base_xq), @intCast(total_blocks), null)
                else
                    kern.quantize_q8_k(base_act, @ptrCast(base_xq), @intCast(total_blocks), null);
                if (rc != cudart.cudaSuccess) {
                    // No single token index to blame any more: report the whole
                    // quantize that failed (all k rows, nb_tok blocks each).
                    std.debug.print("cuda backend: batchK quantize({s}) failed for k={d} tok x nblocks={d} (total {d} blocks): {s}\n", .{
                        @tagName(q.ttype), k, nb_tok, total_blocks, std.mem.span(cu.lib.errStr(rc)),
                    });
                    self.clearDevBuf(outBuf);
                    return;
                }
                // Diagnostics only.  This used to be +1 per token (K per call);
                // it is now +1 for the merged quantize, so one matvecBatchK call
                // costs 2 launches (quantize + mul_mat_vec) -- see the
                // `bk_launch` field on the `CUDA-M` line.
                self.stat_n_launch_bk += 1;

                const kern_mm = Launch;
                // New binding (shadowing the quantize's `rc` above): the old
                // per-token loop scoped that one to the loop body, so merging
                // it out would otherwise redeclare `rc` in this block.
                const rc_mm = kern_mm.mul_mat_vec(
                    e.dev,
                    self.d_xq,
                    d_out,
                    @intCast(rows),
                    @intCast(cols),
                    @intCast(k),
                    @intCast(@backingInt(q.ttype)),
                    null,
                );
                if (rc_mm != cudart.cudaSuccess) {
                    std.debug.print("cuda backend: mul_mat_vec({s}, K={d}) launch failed: {s}\n", .{
                        @tagName(q.ttype), k, std.mem.span(cu.lib.errStr(rc_mm)),
                    });
                    self.clearDevBuf(outBuf);
                    return;
                }
                if (out_dev == null) self.downloadOut(outBuf.slice()[0 .. k * rows]);
                self.stat_ns_batchk += durNs(t0, self.io);
                self.stat_by_type[@backingInt(q.ttype)] += 1;
                self.stat_ns_by_type[@backingInt(q.ttype)] += durNs(t0, self.io);
                self.stat_n_batchk += 1;
                // The matmul itself: with the merged quantize above this makes
                // the per-call launch count 2, independent of k.
                self.stat_n_launch_bk += 1;
                // `$QWEN35_PROF`: GPU-side time of what this call just issued
                // (quantize TK + mul_mat_vec).  The drain is inside `evElapsedNs`.
                if (self.gpuTimingOn()) self.batchKTimingEnd();
                // Interim dump: the CLI path never runs `deinit` (the engine is
                // arena-allocated), so without a periodic report the only profile
                // line is the one printed on the first matvec.  Every 200th chunk
                // keeps generation-phase numbers visible without spamming.
                if (cudart.getenv("QWEN35_PROF") != null and self.stat_n_batchk % 200 == 0) {
                    self.reportStats("batchk");
                }
                // Token 0 of the chunk is enough: all K rows run through the same
                // kernel, so a per-type mismatch shows up on the first one.
                self.mvCheckVariant(outBuf, q, xKBuf, true, k);
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
        try self.uploadIqTables();
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
        try self.uploadIqTables();
        const cu = &self.cu.?;
        try ensureBuf(&cu.lib, &self.d_act, &self.d_act_bytes, k * max_cols * @sizeOf(f32));
        try ensureBuf(&cu.lib, &self.d_xq, &self.d_xq_bytes, k * actBytes(.q8_k, max_cols));
        try ensureBuf(&cu.lib, &self.d_out, &self.d_out_bytes, k * max_cols * @sizeOf(f32));
    }

    /// Upload the twelve IQ codebooks to the device, at most once per process.
    ///
    /// WHY HERE: the tables are read by name through `__device__` pointers that
    /// only `q35_launch_init_tables` fills in (src/kernels/cuda/iq_tables.cu).
    /// A kernel that runs before the upload indexes address 0 -- wrong numbers,
    /// no crash, and the failure looks like a bad quant kernel -- so the upload
    /// has to be provably earlier than the first matvec/dequantRow.  Both
    /// scratch-allocation hooks are the place: `Engine.load` calls
    /// `exec.allocScratch(cols)` while opening the model, before it has any
    /// weights resident or computes anything (src/engine.zig, `load`), and
    /// `allocScratchK` only ever follows it.  Making this the first statement
    /// after the `gpuReady` check means no device buffer -- IQ ones included --
    /// exists before the codebooks do.
    ///
    /// Cached through `iq_tables_state` rather than re-called every time: the
    /// C side is idempotent too, but this way a failed upload is reported once
    /// instead of on every later matvec.
    fn uploadIqTables(self: *CudaBackend) anyerror!void {
        switch (self.iq_tables_state) {
            .done => return,
            // A previous attempt failed.  Do not retry: the reason was already
            // recorded in `init_error` and repeated failures of the same kind
            // would only spam the log.  Report the same error to the caller.
            .failed => return error.IqTablesNotUploaded,
            .pending => {},
        }
        // `gpuReady` is checked by both callers before they get here, so the
        // runtime handle exists; no separate "kernels loaded" guard is needed
        // because they are link-time symbols.
        const rc = Launch.init_tables(null);
        if (rc != cudart.cudaSuccess) {
            std.debug.print(
                "error: cuda backend: IQ codebook upload failed (cudaError {d}: {s})\n" ++
                    "       IQ quant types cannot run on the GPU without it; use --backend cpu\n",
                .{ rc, std.mem.span(self.cu.?.lib.GetErrorName(rc)) },
            );
            self.iq_tables_state = .failed;
            return error.IqTablesNotUploaded;
        }
        self.iq_tables_state = .done;
    }

    /// `allocBuf` for the device backend: a `cudaMalloc` region, so every
    /// `backend.Buf` the engine holds (activations, KV cache, GDN state) is
    /// DEVICE memory and the slots can hand `buf.ptr` straight to a kernel.
    ///
    /// This is the P2c change: P2b forwarded to the embedded CPU backend, which
    /// returned host memory and forced every slot to bounce its operands
    /// through staging.  The list of handed-out regions lives here (`dev_bufs`)
    /// because `freeScratch` must release them all at once and `backend.Buf`
    /// carries no ownership flag.
    ///
    /// Failure is a hard error, not a fallback: the engine's caller has no
    /// hybrid path any more (device memory is where the data must live), so a
    /// failed `cudaMalloc` has to surface as a load/allocation failure.
    fn dAllocBuf(ctx: *anyopaque, len: usize) anyerror!backend.Buf {
        const self = self_(ctx);
        if (len == 0) return backend.Buf.empty;
        const cu = if (self.cu) |*c| c else {
            std.debug.print("cuda backend: allocBuf({d}): {s}\n", .{ len, self.needHelp() });
            return error.DeviceUnavailable;
        };
        const bytes = len * @sizeOf(f32);
        var p: ?*anyopaque = null;
        const rc = cu.lib.Malloc(&p, bytes);
        if (rc != cudart.cudaSuccess or p == null) {
            std.debug.print("cuda backend: allocBuf({d} floats = {d} B) failed: {s}\n", .{
                len, bytes, std.mem.span(cu.lib.errStr(rc)),
            });
            return error.OutOfMemory;
        }
        // A `Buf` is a `[]f32`-shaped view; the pointer is reused as-is, which
        // is what every slot then casts back to a device pointer.
        const slice: []f32 = @as([*]f32, @ptrCast(@alignCast(p.?)))[0..len];
        self.dev_bufs.append(self.alloc, p.?) catch {
            _ = cu.lib.Free(p.?);
            return error.OutOfMemory;
        };
        return backend.Buf.wrap(slice);
    }

    /// Release every region `dAllocBuf` handed out, plus the reusable scratch
    /// (`d_act`/`d_xq`/`d_out`/`d_deq`) and the resident constant copies.
    ///
    /// Called once, by the engine at teardown (`exec.freeScratch()`), never per
    /// token.  `dev_bufs` is sorted-free because the order does not matter; the
    /// `Buf` values the engine still holds become dangling here, which is why
    /// the engine must not touch them afterwards (it does not: teardown follows).
    fn dFreeScratch(ctx: *anyopaque) void {
        const self = self_(ctx);
        const cu = if (self.cu) |*c| c else return;
        for (self.dev_bufs.items) |p| _ = cu.lib.Free(p);
        self.dev_bufs.clearRetainingCapacity();
        if (self.d_act) |p| _ = cu.lib.Free(p);
        if (self.d_xq) |p| _ = cu.lib.Free(p);
        if (self.d_out) |p| _ = cu.lib.Free(p);
        if (self.d_deq) |p| _ = cu.lib.Free(p);
        for (self.dev_const.items) |e| _ = cu.lib.Free(e.dev);
        self.dev_const.clearRetainingCapacity();
        self.d_act = null;
        self.d_act_bytes = 0;
        self.d_xq = null;
        self.d_xq_bytes = 0;
        self.d_out = null;
        self.d_out_bytes = 0;
        self.d_deq = null;
        self.d_deq_bytes = 0;
    }

    // ==== slots: pure elementwise (P2b) =====================================
    //
    // scaleVec / addResid / copyRange / zeroBuf are the first kernels to move
    // off the CPU.  None of them reduces, so each element is one IEEE single
    // operation and the GPU result must be BIT-EXACT with the CPU one (that is
    // what keeps the token stream identical); see
    // src/kernels/cuda/elementwise.cu.
    //
    // P2c: ACTIVATIONS ARE DEVICE MEMORY NOW.  `dAllocBuf` hands out
    // `cudaMalloc` regions, so a `backend.Buf` is already a device pointer and
    // every slot below passes `buf.ptr` straight to the launcher.  The P2b
    // staging triple and its `d_stage` slots are gone: there is exactly one
    // legal host->device path left in this file,
    // `devConst` (read-only weights, uploaded once), and no device->host path on
    // the per-token execution at all -- the only D2H left is `downloadOut` for
    // matvec results, which is the values the sampler needs.
    //
    // `dBufPtr` below is the whole replacement: it is `ensureGpu` + a pointer
    // cast, kept as one helper so the "is the device up, is this buffer empty"
    // check cannot drift between slots.

    /// Device pointer for a resident `backend.Buf`, or null for an empty buffer.
    ///
    /// No copy, no allocation, no ownership: after P2c the `Buf` IS the device
    /// region.  Null means "nothing to do" -- `Buf.empty()` is a legal value
    /// (`engine.zig` uses it for absent optional state) and the launchers all
    /// treat `n <= 0` as a no-op, so the null return simply skips the launch.
    ///
    /// The CPU fallback in the slots that call this is now a *correctness
    /// net*, not a hybrid path: if the runtime is down the buffers handed out by
    /// `dAllocBuf` would not exist either, so reaching it means something is
    /// already broken and the CPU path is the honest answer.
    fn dBufPtr(buf: backend.Buf) ?[*]f32 {
        if (buf.len == 0) return null;
        return @ptrCast(@alignCast(buf.ptr));
    }

    /// Zero a device `Buf` without a host round trip (the error path of every
    /// numeric slot: an empty result is the honest failure signal, and the CPU
    /// fallback of `elementwise` cannot be used because the buffer is device
    /// memory).  Falls back to the CPU memset only when the pointer is not a
    /// device region at all, i.e. the caller really handed us host memory.
    fn clearDevBuf(self: *CudaBackend, buf: backend.Buf) void {
        if (buf.len == 0) return;
        if (self.cu) |*c| {
            if (dBufPtr(buf)) |p| {
                if (c.lib.Memset(@ptrCast(p), 0, buf.len * @sizeOf(f32)) == cudart.cudaSuccess) return;
            }
        }
        @memset(buf.slice(), 0);
    }

    /// Failure text for a launch that returned a cudaError_t.
    fn ewLaunchFail(self: *CudaBackend, what: []const u8, rc: c_int) void {
        const cu = self.cu orelse return;
        std.debug.print("cuda backend: {s} launch failed: {s}\n", .{
            what, std.mem.span(cu.lib.errStr(rc)),
        });
    }

    /// Device copy of a read-only weight slice, uploaded at most once per
    /// (pointer, length).  Returns null when the device is unavailable or the
    /// upload fails -- the caller then takes the CPU path, exactly like a
    /// failed launch does.
    ///
    /// This is THE only host->device copy left on the per-token path, and it is
    /// a cache rather than a transfer: a hit returns the resident pointer with
    /// no `cudaMemcpy` at all, which is what makes re-passing a per-layer
    /// weight every token free.  `bytes` is the identity's second half for the
    /// reason in `dev_const`.
    fn devConst(self: *CudaBackend, host: []const f32) !?[*]f32 {
        if (host.len == 0) return null;
        const cu = if (self.cu) |*c| c else return error.DeviceUnavailable;
        const bytes = host.len * @sizeOf(f32);
        for (self.dev_const.items) |e| {
            if (e.host == host.ptr and e.bytes == bytes) return @ptrCast(@alignCast(e.dev));
        }
        var p: ?*anyopaque = null;
        var cap: usize = 0;
        try ensureBuf(&cu.lib, &p, &cap, bytes);
        const rc = cu.lib.Memcpy(p.?, host.ptr, bytes, cudart.MemcpyHostToDevice);
        if (rc != cudart.cudaSuccess) {
            _ = cu.lib.Free(p.?);
            std.debug.print("cuda backend: constant upload failed ({d} B): {s}\n", .{
                bytes, std.mem.span(cu.lib.errStr(rc)),
            });
            return error.MemcpyFailed;
        }
        try self.dev_const.append(self.alloc, .{
            .host = host.ptr,
            .bytes = bytes,
            .dev = p.?,
        });
        return @ptrCast(@alignCast(p.?));
    }

    // The four slots.  Each is a FORWARDER rather than an alias to
    // `cpu_mod.CpuBackend.vtable.*`, for the same reason `dAllocBuf` is: `ctx`
    // is a `*CudaBackend` and these need it (device scratch + stream).  Every
    // one of them degrades to the embedded CPU backend on any device-side
    // failure, so a broken/missing GPU still produces correct numbers instead of
    // silently leaving the buffer untouched -- none of these slots can report an
    // error through the `Exec` signature, and panicking the process would be a
    // worse outcome than a slow correct answer.

    fn dScaleVec(ctx: *anyopaque, x: backend.Buf, s: f32) void {
        const self = self_(ctx);
        if (self.cu != null) {
            if (dBufPtr(x)) |px| {
                const rc = Launch.scale_vec(px, s, @intCast(x.len), null);
                if (rc == cudart.cudaSuccess) return;
                self.ewLaunchFail("scale_vec", rc);
            } else return;
        }
        cpu_mod.CpuBackend.vtable.scaleVec(@ptrCast(&self.cpu), x, s);
    }

    fn dAddResid(ctx: *anyopaque, dst: backend.Buf, src: backend.Buf) void {
        const self = self_(ctx);
        // Equal lengths are the engine's contract (`elementwise.bufAddResid`
        // zips the two slices).  A mismatch is an engine bug, so refuse to guess
        // rather than silently truncating to the shorter.
        if (src.len != dst.len) {
            std.debug.print("cuda backend: addResid length mismatch (dst={d} src={d}); using the CPU path\n", .{ dst.len, src.len });
            cpu_mod.CpuBackend.vtable.addResid(@ptrCast(&self.cpu), dst, src);
            return;
        }
        if (self.cu != null) {
            if (dBufPtr(dst)) |pd| {
                if (dBufPtr(src)) |ps| {
                    const rc = Launch.add_resid(pd, ps, @intCast(dst.len), null);
                    if (rc == cudart.cudaSuccess) return;
                    self.ewLaunchFail("add_resid", rc);
                } else return;
            } else return;
        }
        cpu_mod.CpuBackend.vtable.addResid(@ptrCast(&self.cpu), dst, src);
    }

    /// True when this `Buf` is NOT one of ours, i.e. it wraps ordinary host
    /// memory (`Buf.wrap` of a `[]f32`, which the engine still does for f32
    /// token embeddings and for anything it owns itself).  The test is whether
    /// the pointer is in the `dev_bufs` list -- the only way a device region
    /// comes into existence, since `dAllocBuf` is the sole allocator.
    ///
    /// This exists because `Buf` cannot tell host from device by looking at
    /// it, and two slots legitimately receive host memory:
    ///   * `copyRange(device, Buf.wrap(host_embedding_row))` -- the f32
    ///     embedding lookup (engine.zig `embedToken`),
    ///   * `dCopyFromHost`'s callers.
    /// Treating such a `src` as a device pointer is an immediate illegal
    /// access -- caught in the P2c acceptance run on the Q8_K_XL model, whose
    /// embedding table is f32 (the Q6_K one is quantized and goes through
    /// `dequantRow`, which is why it passed first).
    fn isDevBuf(self: *CudaBackend, b: backend.Buf) bool {
        const p: *const anyopaque = @ptrCast(b.ptr);
        for (self.dev_bufs.items) |d| {
            if (d == p) return true;
        }
        return false;
    }

    fn dCopyRange(ctx: *anyopaque, dst: backend.Buf, src: backend.Buf) void {
        const self = self_(ctx);
        // Same contract as `addResid`: `elementwise.bufCopyRange` is a plain
        // `@memcpy`, which requires equal lengths.  Mismatch = engine bug ->
        // CPU path, not a silent partial copy.
        if (src.len != dst.len) {
            std.debug.print("cuda backend: copyRange length mismatch (dst={d} src={d}); using the CPU path\n", .{ dst.len, src.len });
            cpu_mod.CpuBackend.vtable.copyRange(@ptrCast(&self.cpu), dst, src);
            return;
        }
        if (self.cu != null) {
            if (dBufPtr(dst)) |pd| {
                if (self.isDevBuf(src)) {
                    if (dBufPtr(src)) |ps| {
                        // Both device-resident: a pure device-to-device move.
                        const rc = Launch.copy_range(pd, ps, @intCast(dst.len), null);
                        if (rc == cudart.cudaSuccess) return;
                        self.ewLaunchFail("copy_range", rc);
                    } else return;
                } else {
                    // `src` is host memory (an f32 embedding row).  Same kernel,
                    // opposite direction: this is the documented H2D path.
                    CudaBackend.dCopyFromHost(@ptrCast(self), dst, src.slice()[0..dst.len]);
                    return;
                }
            } else return;
        }
        cpu_mod.CpuBackend.vtable.copyRange(@ptrCast(&self.cpu), dst, src);
    }

    fn dZeroBuf(ctx: *anyopaque, buf: backend.Buf) void {
        const self = self_(ctx);
        if (self.cu != null) {
            if (dBufPtr(buf)) |p| {
                const rc = Launch.zero_buf(p, @intCast(buf.len), null);
                if (rc == cudart.cudaSuccess) return;
                self.ewLaunchFail("zero_buf", rc);
            } else return;
        }
        // Device path unavailable: still give dst the zero semantics it was
        // promised.  This touches host memory only when the buffer IS host
        // memory (i.e. the CPU fallback path), which is the honest reading.
        @memset(buf.slice(), 0);
    }

    // ==== slots: the two host<->device bridges ==============================
    //
    // These are the only two slots that intentionally cross PCIe, and both are
    // bounded by something other than "one copy per operator":
    //
    //   copyFromHost -- a token's f32 embedding row (unquantized token_embd
    //                   only), once per token, `hidden` floats.
    //   copyToHost   -- see backend.zig: `nv` (64) scalars per layer per token,
    //                   because `gdnStep` takes beta/gamma as `f32` arguments.
    //
    // Everything else in this file is device-to-device.

    fn dCopyFromHost(ctx: *anyopaque, dst: backend.Buf, src: []const f32) void {
        const self = self_(ctx);
        if (self.cu != null and src.len != 0 and src.len <= dst.len) {
            if (dBufPtr(dst)) |pd| {
                const cu = &self.cu.?;
                const rc = cu.lib.Memcpy(
                    @ptrCast(pd),
                    @ptrCast(src.ptr),
                    src.len * @sizeOf(f32),
                    cudart.MemcpyHostToDevice,
                );
                self.stat_h2d_bytes += src.len * @sizeOf(f32);
                // `$QWEN35_STAT_TALKY`: breakdown of who still hands this backend
                // host memory.  Grouped by element count, which identifies the
                // caller (hidden=1024 vs head_dim=256).
                if (cudart.getenv("QWEN35_STAT_TALKY") != null) {
                    if (self.debug_h2d_len == src.len) {
                        self.debug_h2d_same += 1;
                    } else {
                        std.debug.print("H2DCOPY len={d} count={d}\n", .{ self.debug_h2d_len, self.debug_h2d_same });
                        self.debug_h2d_len = src.len;
                        self.debug_h2d_same = 1;
                    }
                }
                if (rc == cudart.cudaSuccess) return;
                self.ewLaunchFail("copyFromHost", rc);
            } else return;
        }
        // A length mismatch is an engine bug; let the CPU reference report it.
        cpu_mod.CpuBackend.vtable.copyFromHost(@ptrCast(&self.cpu), dst, src);
    }

    fn dCopyToHost(ctx: *anyopaque, dst: []f32, src: backend.Buf) void {
        const self = self_(ctx);
        std.debug.assert(dst.len <= src.len);
        if (self.cu != null) {
            if (dBufPtr(src)) |ps| {
                // Blocking on purpose: it is `nv` floats per layer, and a
                // blocking copy into pageable host memory is both simpler and
                // (at this size) indistinguishable in cost from an async one.
                const cu = &self.cu.?;
                const rc = cu.lib.Memcpy(
                    @ptrCast(dst.ptr),
                    @ptrCast(ps),
                    dst.len * @sizeOf(f32),
                    cudart.MemcpyDeviceToHost,
                );
                self.stat_d2h_bytes += dst.len * @sizeOf(f32);
                if (rc == cudart.cudaSuccess) return;
                self.ewLaunchFail("copyToHost", rc);
            } else return;
        }
        cpu_mod.CpuBackend.vtable.copyToHost(@ptrCast(&self.cpu), dst, src);
    }

    // ==== slots: norms / activations (P2b, second batch) ====================
    //
    // `dBufPtr -> launch`, with the CPU slot as the fallback on ANY failure.
    //
    // Two properties worth stating:
    //
    //   * These are NOT bit-exact with the CPU.  rmsnorm/l2norm/softmax reduce
    //     in a different order (a block tree vs the CPU's serial loop), and
    //     every transcendental here (`expf`, `powf`, `cosf`, `sinf`) is a
    //     different implementation in the device runtime than in glibc.  The
    //     measured figures are in the .cu headers.
    //   * The read-only weight operand (`w` in rmsnorm, `wconv` in convStep) is
    //     a per-layer constant re-passed on every token, so it is NOT a device
    //     `Buf` -- `devConst` keeps it resident, and a cache hit costs nothing.

    fn dRmsnorm(ctx: *anyopaque, out: backend.Buf, x: backend.Buf, w: backend.Buf, eps: f32) void {
        const self = self_(ctx);
        // The CPU reference asserts out.len == x.len == w.len (one row).  The
        // kernel supports several rows (w reused per row) for the shape the
        // engine does not currently produce, so the only mismatch worth
        // refusing is a partial row.
        if (x.len != 0 and w.len != 0 and x.len % w.len == 0 and x.len == out.len and self.cu != null) {
            if (dBufPtr(x)) |px| {
                if (dBufPtr(out)) |po| {
                    if (self.devConst(w.slice())) |pw_unwrapped| {
                        if (pw_unwrapped) |pw| {
                            const rows: c_int = @intCast(x.len / w.len);
                            const rc = Launch.rmsnorm(po, px, pw, rows, @intCast(w.len), eps, null);
                            if (rc == cudart.cudaSuccess) return;
                            self.ewLaunchFail("rmsnorm", rc);
                        }
                    } else |_| {}
                } else return;
            } else return;
        }
        cpu_mod.CpuBackend.vtable.rmsnorm(@ptrCast(&self.cpu), out, x, w, eps);
    }

    fn dL2norm(ctx: *anyopaque, x: backend.Buf) void {
        const self = self_(ctx);
        if (self.cu != null) {
            if (dBufPtr(x)) |px| {
                // eps 1e-6 is elementwise.zig `l2norm`'s constant; it is passed
                // explicitly so the CPU reference stays the single definition.
                const rc = Launch.l2norm(px, 1, @intCast(x.len), 1e-6, null);
                if (rc == cudart.cudaSuccess) return;
                self.ewLaunchFail("l2norm", rc);
            } else return;
        }
        cpu_mod.CpuBackend.vtable.l2norm(@ptrCast(&self.cpu), x);
    }

    fn dSiluInPlace(ctx: *anyopaque, x: backend.Buf) void {
        const self = self_(ctx);
        if (self.cu != null) {
            if (dBufPtr(x)) |px| {
                const rc = Launch.silu(px, @intCast(x.len), null);
                if (rc == cudart.cudaSuccess) return;
                self.ewLaunchFail("silu", rc);
            } else return;
        }
        cpu_mod.CpuBackend.vtable.siluInPlace(@ptrCast(&self.cpu), x);
    }

    fn dMulSilu(ctx: *anyopaque, out: backend.Buf, gate: backend.Buf) void {
        const self = self_(ctx);
        if (gate.len != out.len) {
            std.debug.print("cuda backend: mulSilu length mismatch (out={d} gate={d}); using the CPU path\n", .{ out.len, gate.len });
            cpu_mod.CpuBackend.vtable.mulSilu(@ptrCast(&self.cpu), out, gate);
            return;
        }
        if (self.cu != null) {
            if (dBufPtr(out)) |po| {
                if (dBufPtr(gate)) |pg| {
                    const rc = Launch.mul_silu(po, pg, @intCast(out.len), null);
                    if (rc == cudart.cudaSuccess) return;
                    self.ewLaunchFail("mul_silu", rc);
                } else return;
            } else return;
        }
        cpu_mod.CpuBackend.vtable.mulSilu(@ptrCast(&self.cpu), out, gate);
    }

    fn dMulSigmoid(ctx: *anyopaque, out: backend.Buf, gate: backend.Buf) void {
        const self = self_(ctx);
        if (gate.len != out.len) {
            std.debug.print("cuda backend: mulSigmoid length mismatch (out={d} gate={d}); using the CPU path\n", .{ out.len, gate.len });
            cpu_mod.CpuBackend.vtable.mulSigmoid(@ptrCast(&self.cpu), out, gate);
            return;
        }
        if (self.cu != null) {
            if (dBufPtr(out)) |po| {
                if (dBufPtr(gate)) |pg| {
                    const rc = Launch.mul_sigmoid(po, pg, @intCast(out.len), null);
                    if (rc == cudart.cudaSuccess) return;
                    self.ewLaunchFail("mul_sigmoid", rc);
                } else return;
            } else return;
        }
        cpu_mod.CpuBackend.vtable.mulSigmoid(@ptrCast(&self.cpu), out, gate);
    }

    fn dSwiglu(ctx: *anyopaque, gate: backend.Buf, up: backend.Buf) void {
        const self = self_(ctx);
        if (up.len != gate.len) {
            std.debug.print("cuda backend: swiglu length mismatch (gate={d} up={d}); using the CPU path\n", .{ gate.len, up.len });
            cpu_mod.CpuBackend.vtable.swiglu(@ptrCast(&self.cpu), gate, up);
            return;
        }
        if (self.cu != null) {
            if (dBufPtr(gate)) |pg| {
                if (dBufPtr(up)) |pu| {
                    const rc = Launch.swiglu(pg, pu, @intCast(gate.len), null);
                    if (rc == cudart.cudaSuccess) return;
                    self.ewLaunchFail("swiglu", rc);
                } else return;
            } else return;
        }
        cpu_mod.CpuBackend.vtable.swiglu(@ptrCast(&self.cpu), gate, up);
    }

    fn dSoftmax(ctx: *anyopaque, x: backend.Buf) void {
        const self = self_(ctx);
        if (self.cu != null) {
            if (dBufPtr(x)) |px| {
                const rc = Launch.softmax(px, @intCast(x.len), null);
                if (rc == cudart.cudaSuccess) return;
                self.ewLaunchFail("softmax", rc);
            } else return;
        }
        cpu_mod.CpuBackend.vtable.softmax(@ptrCast(&self.cpu), x);
    }

    // ==== sampling (P2c: logits are device memory) ==========================
    //
    // The engine keeps `logits` in a backend region, so both of these used to
    // dereference a device pointer through the CPU implementation.  argmax is
    // a pure max and therefore goes to the device exactly (see the kernel);
    // `sample` stays on the CPU because the sampler is a sampling POLICY
    // (temperature, top-k/p, RNG) whose reference implementation must not be
    // duplicated, and it needs the whole distribution.

    fn dArgmax(ctx: *anyopaque, logits: backend.Buf) u32 {
        const self = self_(ctx);
        if (self.cu != null and logits.len != 0) {
            if (dBufPtr(logits)) |px| {
                // The launcher writes through `d_out`, so it MUST be device
                // memory: the previous build passed the address of a stack
                // `c_int`, which the kernel dereferenced as a GPU pointer.  A
                // dedicated 4-byte device cell is reused (grown once) and only
                // the final index -- 4 bytes -- comes back per token.
                const cu = &self.cu.?;
                ensureBuf(&cu.lib, &self.d_idx, &self.d_idx_bytes, @sizeOf(c_int)) catch return 0;
                const rc = Launch.argmax(px, @intCast(logits.len), @ptrCast(@alignCast(self.d_idx)), null);
                if (rc != cudart.cudaSuccess) {
                    self.ewLaunchFail("argmax", rc);
                    return 0;
                }
                var host_idx: c_int = 0;
                const rc2 = cu.lib.Memcpy(&host_idx, self.d_idx, @sizeOf(c_int), cudart.MemcpyDeviceToHost);
                if (rc2 != cudart.cudaSuccess) {
                    self.ewLaunchFail("argmax index download", rc2);
                    return 0;
                }
                self.stat_d2h_bytes += @sizeOf(c_int);
                return @intCast(host_idx);
            } else return 0;
        }
        return cpu_mod.CpuBackend.vtable.argmax(@ptrCast(&self.cpu), logits);
    }

    /// Host staging for `dSample`: pull the row down, run the reference
    /// sampler on it -- the one place the sampler's own `[]f32` contract has to
    /// be fed, and it is fed with a copy the caller cannot observe later.
    ///
    /// The scratch buffer handed in is a device region too, so its contents are
    /// not needed by the caller (`bufSample` treats it as write-only scratch);
    /// the sampler gets a temporary host array instead.
    fn sampleViaHost(self: *CudaBackend, logits: backend.Buf, scratch: backend.Buf, sp: *sampler.Sampler) u32 {
        const alloc = self.alloc;
        const host_logits = alloc.alloc(f32, logits.len) catch return 0;
        defer alloc.free(host_logits);
        const host_scratch = alloc.alloc(f32, @max(scratch.len, logits.len)) catch return 0;
        defer alloc.free(host_scratch);
        CudaBackend.dCopyToHost(@ptrCast(self), host_logits, logits);
        return sp.sample(host_logits, host_scratch);
    }

    fn dSample(ctx: *anyopaque, logits: backend.Buf, scratch: backend.Buf, sp: *sampler.Sampler) u32 {
        const self = self_(ctx);
        // Greedy fast path: `Sampler.sample` with temperature <= 0 is a pure
        // argmax (sampler.zig:73), so it needs no distribution -- run it on the
        // device and bring back 4 bytes instead of the whole vocabulary.  This
        // is the default decode path, so it is what keeps the per-token
        // cross-boundary traffic at a single int.
        if (sp.temperature <= 0) return CudaBackend.dArgmax(ctx, logits);
        // True sampling: one D2H per generated token, of the vocabulary-length
        // logits row, and it is unavoidable while the sampler is the CPU
        // reference.  It is also the only reason `scratch` has to be host
        // memory: the sampler writes its normalised distribution there.  Sized
        // by the engine.
        if (self.cu != null and logits.len != 0) {
            if (dBufPtr(logits) != null) {
                return self.sampleViaHost(logits, scratch, sp);
            }
        }
        return cpu_mod.CpuBackend.vtable.sample(@ptrCast(&self.cpu), logits, scratch, sp);
    }

    // ==== slots: the two state machines (P2c) ===============================
    //
    // These are the two slots that P2c exists for.  Until KV and the GDN state
    // became backend regions they COULD NOT be device slots at all: their
    // arguments are large slices of state (a 64 KB GDN state per head, up to
    // `slot+1` KV rows per layer), so a staging backend would have had to move
    // megabytes per launch -- orders of magnitude worse than leaving them on
    // the CPU.  With both resident, the launches are free of traffic.
    //
    // The numeric contract is in the kernel headers (src/kernels/cuda/gdn.cu,
    // attn.cu): accumulation order follows the CPU reference element by
    // element, and neither is bit-exact (FMA contraction on the device side).

    fn dGdnStep(ctx: *anyopaque, state: backend.Buf, k: backend.Buf, v: backend.Buf,
                q: backend.Buf, km: backend.Buf, delta: backend.Buf, out: backend.Buf,
                gamma: f32, beta: f32, vd: usize) void {
        const self = self_(ctx);
        // vd != 128 is the CPU `gdnFused`'s slow path (four separate kernels);
        // the device kernel is written for the fast path only, so route the odd
        // shape to the CPU rather than launching a kernel that would refuse.
        if (vd == 128 and self.cu != null and
            state.len >= vd * vd and k.len >= vd and q.len >= vd and
            v.len >= vd and km.len >= vd and delta.len >= vd and out.len >= vd)
        {
            if (dBufPtr(state)) |pstate| {
                if (dBufPtr(k)) |pk| {
                    if (dBufPtr(v)) |pv| {
                        if (dBufPtr(q)) |pq| {
                            if (dBufPtr(km)) |pkm| {
                                if (dBufPtr(delta)) |pdelta| {
                                    if (dBufPtr(out)) |pout| {
                                        const rc = Launch.gdn_step(pstate, pk, pv, pq, pkm, pdelta, pout, gamma, beta, @intCast(vd), null);
                                        if (rc == cudart.cudaSuccess) return;
                                        self.ewLaunchFail("gdn_step", rc);
                                    } else return;
                                } else return;
                            } else return;
                        } else return;
                    } else return;
                } else return;
            } else return;
        }
        cpu_mod.CpuBackend.vtable.gdnStep(@ptrCast(&self.cpu), state, k, v, q, km, delta, out, gamma, beta, vd);
    }

    fn dGdnGamma(ctx: *anyopaque, a_raw: backend.Buf, b_raw: backend.Buf, ssm_a: backend.Buf,
                 dt_bias: backend.Buf, gamma: backend.Buf, nv: usize) void {
        const self = self_(ctx);
        // Every operand here except `ssm_a`/`dt_bias` is a live backend region.
        // The two weight rows are per-layer constants, so they go through the
        // `devConst` cache exactly like the norm weights do.
        if (self.cu != null and nv != 0 and
            a_raw.len >= nv and b_raw.len >= nv and ssm_a.len >= nv and
            dt_bias.len >= nv and gamma.len >= nv)
        {
            if (dBufPtr(a_raw)) |pa| {
                if (dBufPtr(b_raw)) |pb| {
                    if (dBufPtr(gamma)) |pg| {
                        if (self.devConst(ssm_a.slice())) |sa_unwrapped| {
                            if (sa_unwrapped) |sa| {
                                if (self.devConst(dt_bias.slice())) |db_unwrapped| {
                                    if (db_unwrapped) |db| {
                                        const rc = Launch.gdn_gamma(pa, pb, sa, db, pg, @intCast(nv), null);
                                        if (rc == cudart.cudaSuccess) return;
                                        self.ewLaunchFail("gdn_gamma", rc);
                                    }
                                } else |_| {}
                            }
                        } else |_| {}
                    } else return;
                } else return;
            } else return;
        }
        cpu_mod.CpuBackend.vtable.gdnGamma(@ptrCast(&self.cpu), a_raw, b_raw, ssm_a, dt_bias, gamma, nv);
    }

    fn dAttnDecode(ctx: *anyopaque, q: backend.Buf, gate: backend.Buf, kcache: backend.Buf,
                   vcache: backend.Buf, att: backend.Buf, gout: backend.Buf, slot: usize,
                   heads: usize, gqa: usize, hd: usize, kv_dim: usize, scale: f32) void {
        const self = self_(ctx);
        // `kcache`/`vcache` are the first `slot+1` rows of the layer's cache, so
        // they are sized (slot+1)*kv_dim, not max_seq*kv_dim (the engine passes
        // exactly that span).  `att` needs slot+1 floats.
        //
        // `heads * hd <= gout.len` and `q.len >= heads*hd` are the engine's
        // contract; a violation is a bug, and the CPU reference's slicing would
        // trap on it, so fall through to it rather than launch on bad sizes.
        const nrows = slot + 1;
        if (self.cu != null and heads != 0 and hd != 0 and
            q.len >= heads * hd and gout.len >= heads * hd and
            att.len >= nrows and kcache.len >= nrows * kv_dim and vcache.len >= nrows * kv_dim)
        {
            if (dBufPtr(q)) |pq| {
                if (dBufPtr(gate)) |pgate| {
                    if (dBufPtr(kcache)) |pk| {
                        if (dBufPtr(vcache)) |pv| {
                            if (dBufPtr(att)) |patt| {
                                if (dBufPtr(gout)) |pgout| {
                                    const rc = Launch.attn_decode(pq, pgate, pk, pv, patt, pgout, @intCast(slot), @intCast(heads), @intCast(gqa), @intCast(hd), @intCast(kv_dim), scale, null);
                                    if (rc == cudart.cudaSuccess) return;
                                    self.ewLaunchFail("attn_decode", rc);
                                } else return;
                            } else return;
                        } else return;
                    } else return;
                } else return;
            } else return;
        }
        cpu_mod.CpuBackend.vtable.attnDecode(@ptrCast(&self.cpu), q, gate, kcache, vcache, att, gout, slot, heads, gqa, hd, kv_dim, scale);
    }

    // rope64: TWO paths, selected at runtime by `$QWEN35_CUDA_ROPE`.
    //
    //   host   (DEFAULT, unchanged from before) -- D2H, `elementwise.zig`
    //          `applyRope64` on the CPU, H2D.  The one remaining D2H+H2D pair on
    //          the per-token path: a head's 64 floats, twice, per q/k tensor per
    //          layer (~64 KB per token for the 0.8B: 48 rotations x 64 x 4 B x
    //          2 directions).  Bit-identical to the CPU backend *by
    //          construction*, because it IS the CPU backend.  These trips are
    //          counted in `stat_rope_trips` so `CUDA-M` shows the A/B really
    //          switched.
    //   device -- `q35_launch_rope64` (`src/kernels/cuda/rope.cu`), zero copy,
    //          `stat_rope_trips` stays 0.  NOT bit-exact; by how much is the
    //          number this section used to guess at.
    //
    // HISTORY (the reason `host` is the default): a device kernel was written,
    // measured and withdrawn.  What that measurement established:
    //
    //   The phase is `theta = pos * base^(-2j/64)`, an ABSOLUTE angle of order
    //   `pos`.  glibc's `powf` and the device runtime's differ by up to 1 ULP
    //   *relative to theta*, which at pos = 1e6 is ~1e-2 radians of phase --
    //   and `cos(theta)` then differs by up to 1e-2 outright, i.e. 5e-4
    //   relative on the rotated value (measured then: max relative error
    //   5.2e-4, max abs 2.5e-5 over 200 positions x 256-element heads).
    //   Recomputing theta in double precision on the device does NOT fix it
    //   (measured: still 5.2e-4) because the host's f32 `powf` is itself the
    //   lower-precision side and cannot be matched, only approximated.
    //
    // What that measurement did NOT establish is the sentence that followed it
    // ("far past what the token stream tolerates"): that was an inference, not
    // a test, and `pos` in that sweep went far past anything `Engine.max_seq`
    // (= 4096) can reach.  The switch exists so the claim can be measured on
    // the real token stream instead of argued about: run the same prompt with
    // `QWEN35_CUDA_ROPE=host` and `=device` and diff the token ids.
    //
    // If this ever goes zero-copy by default, the fix is to accept the drift
    // deliberately (with the measured number), NOT to rewrite the reference.
    fn dRope64(ctx: *anyopaque, x: backend.Buf, pos: u32, base: f32) void {
        const self = self_(ctx);
        const max_head: usize = 256;
        var host: [max_head]f32 = undefined;
        const n = @min(x.len, max_head);
        if (self.cu == null or n == 0 or n != x.len) {
            // Same length guard as the rest: rope is a per-head call (hd = 128
            // or 64 in every Qwen3.5 variant), so anything larger means the
            // engine's shape changed and the CPU reference must own it.
            cpu_mod.CpuBackend.vtable.rope64(@ptrCast(&self.cpu), x, pos, base);
            return;
        }
        if (ropeOnDevice()) {
            // The launcher works on `heads * hd`; `x` is exactly one head, so
            // heads = 1 and hd = n.  The engine calls this once per head.
            if (dBufPtr(x)) |px| {
                const rc = Launch.rope64(px, 1, @intCast(n), @bitCast(pos), base, null);
                if (rc == cudart.cudaSuccess) {
                    self.stat_rope_devices += 1;
                    return;
                }
                self.ewLaunchFail("rope64", rc);
            }
            // No device pointer (or the launch failed): the CPU path below is
            // still correct, so fall through to it rather than leave `x`
            // unrotated.  `stat_rope_trips` then counts this head as the host
            // trip it actually was.
        }
        CudaBackend.dCopyToHost(@ptrCast(self), host[0..n], x);
        cpu_mod.CpuBackend.vtable.rope64(@ptrCast(&self.cpu), backend.Buf.wrap(host[0..n]), pos, base);
        CudaBackend.dCopyFromHost(@ptrCast(self), x, host[0..n]);
        // One H2D + one D2H per rotated head, per position.  Counted so the
        // report can state this remainder's exact share instead of guessing.
        self.stat_rope_trips += 1;
    }

    fn dConvStep(ctx: *anyopaque, conv: backend.Buf, y: backend.Buf, wconv: backend.Buf, k: usize) void {
        const self = self_(ctx);
        // `conv` is the GDN sliding window and is BOTH read and written (the
        // kernel shifts it in place); `y` likewise (read as the 4th tap, then
        // overwritten).  Both are device-resident, so both are passed directly.
        //
        // `k` is the conv width and the kernel hardcodes 4 taps (as does
        // elementwise.zig `convStepKernel`, which asserts it).  A model with a
        // different window must reach the CPU rather than get a wrong answer.
        if (k != 4) {
            std.debug.print("cuda backend: convStep kernel width {d} != 4; using the CPU path\n", .{k});
            cpu_mod.CpuBackend.vtable.convStep(@ptrCast(&self.cpu), conv, y, wconv, k);
            return;
        }
        if (conv.len == 3 * y.len and wconv.len == 4 * y.len and self.cu != null) {
            if (dBufPtr(conv)) |pc| {
                if (dBufPtr(y)) |py| {
                    if (self.devConst(wconv.slice())) |pw_unwrapped| {
                        if (pw_unwrapped) |pw| {
                            const rc = Launch.conv_step(pc, py, pw, @intCast(y.len), null);
                            if (rc == cudart.cudaSuccess) return;
                            self.ewLaunchFail("conv_step", rc);
                        }
                    } else |_| {}
                } else return;
            } else return;
        }
        cpu_mod.CpuBackend.vtable.convStep(@ptrCast(&self.cpu), conv, y, wconv, k);
    }

    // ==== profiling / introspection =========================================

    /// Bytes of IQ codebook currently on the device (0 when the upload has not
    /// happened).  Asked of the kernel object rather than cached in Zig so the
    /// number can never disagree with what the kernels actually index.
    fn iqTablesBytes(self: *CudaBackend) u64 {
        if (self.iq_tables_state != .done) return 0;
        return @intCast(Launch.tables_bytes());
    }

    // ---- GPU-side timing helpers ($QWEN35_PROF only) -----------------------
    //
    // Every entry point is a no-op unless `ev_on` is set, and `ev_on` is only
    // set from `dProfReset`, which the engine calls once before prefill.  So the
    // default path checks a bool and returns: no event is created, recorded or
    // waited on, and no stream is drained.

    /// Create the four events once.  Idempotent; a failure disables the whole
    /// facility rather than silently reporting zeros, because a zero GPU time
    /// would look like "the GPU is idle" -- the opposite of the truth.
    fn ensureEvents(self: *CudaBackend) bool {
        if (self.ev_created) return true;
        const lib = if (self.cu) |*cu| &cu.lib else return false;
        var made: [4]?*anyopaque = .{ null, null, null, null };
        const names = [_][:0]const u8{ "ev_bk_a", "ev_bk_b", "ev_pre_a", "ev_pre_b" };
        for (&made, names) |*slot, nm| {
            var h: ?*anyopaque = null;
            // `cudaEventDefault` (flags 0): the default deliberately keeps host
            // sync enabled, which `EventSynchronize`/`ElapsedTime` need.
            if (lib.EventCreate(&h, 0) != cudart.cudaSuccess or h == null) {
                // Bounded cleanup: destroy whatever was already made so a
                // partial setup cannot leak into the next attempt.
                for (made) |m| {
                    if (m) |p| _ = lib.EventDestroy(p);
                }
                std.debug.print("cuda backend: $QWEN35_PROF event timing disabled ({s} create failed)\n", .{nm});
                return false;
            }
            slot.* = h;
        }
        self.ev_bk_a = made[0];
        self.ev_bk_b = made[1];
        self.ev_pre_a = made[2];
        self.ev_pre_b = made[3];
        self.ev_created = true;
        return true;
    }

    fn freeEvents(self: *CudaBackend) void {
        if (!self.ev_created) return;
        const lib = if (self.cu) |*cu| &cu.lib else null;
        if (lib) |l| {
            for ([_]?*anyopaque{ self.ev_bk_a, self.ev_bk_b, self.ev_pre_a, self.ev_pre_b }) |p| {
                if (p) |q| _ = l.EventDestroy(q);
            }
        }
        self.ev_bk_a = null;
        self.ev_bk_b = null;
        self.ev_pre_a = null;
        self.ev_pre_b = null;
        self.ev_created = false;
    }

    /// True when the caller should record events at all.  Deliberately reads
    /// the env var as well: `profReset` is the only intended setter, but a
    /// second line of defence keeps a stray `ev_on = true` from turning into a
    /// default-path sync on a build where PROF is off.
    fn gpuTimingOn(self: *const CudaBackend) bool {
        return self.ev_on and self.cu != null;
    }

    /// Milliseconds between two recorded events, waiting for the second.  The
    /// wait is what makes the queue drain, so this is the expensive call; it is
    /// made once per timed `matvecBatchK`, which is once per weight per chunk.
    /// Returns null on error (and counts it) so the caller leaves the totals
    /// alone instead of adding a bogus 0.
    fn evElapsedNs(self: *CudaBackend, a: ?*anyopaque, b: ?*anyopaque) ?u64 {
        const lib = if (self.cu) |*cu| &cu.lib else return null;
        const t_sync = Io.Clock.Timestamp.now(self.io, .awake);
        defer self.stat_ns_ev_sync += durNs(t_sync, self.io);
        if (lib.EventSynchronize(b) != cudart.cudaSuccess) {
            self.stat_n_ev_err += 1;
            return null;
        }
        var ms: f32 = 0;
        if (lib.EventElapsedTime(&ms, a, b) != cudart.cudaSuccess) {
            self.stat_n_ev_err += 1;
            return null;
        }
        if (!(ms >= 0)) {
            self.stat_n_ev_err += 1;
            return null;
        }
        return @intFromFloat(@as(f64, ms) * 1_000_000.0);
    }

    /// Begin the device-side clock for one `matvecBatchK` call.  Records into the
    /// default stream, so the timestamp is taken *after* every kernel already
    /// queued by earlier calls -- the gap between two calls is therefore
    /// attributed to whoever issues work next, never smeared into the kernel
    /// timings.
    ///
    /// The prefill window's start event (`ev_pre_a`) is recorded here, on the
    /// first call issued while the window is open: that is the first device
    /// position that belongs to prefill.
    fn batchKTimingBegin(self: *CudaBackend) void {
        if (!self.ensureEvents()) {
            self.ev_on = false;
            return;
        }
        const lib = if (self.cu) |*cu| &cu.lib else return;
        if (self.prefill_open and !self.ev_pre_started) {
            if (lib.EventRecord(self.ev_pre_a, null) != cudart.cudaSuccess) {
                self.stat_n_ev_err += 1;
            }
            self.ev_pre_started = true;
        }
        if (lib.EventRecord(self.ev_bk_a, null) != cudart.cudaSuccess) self.stat_n_ev_err += 1;
    }

    /// Close the device-side clock for one `matvecBatchK` call.
    ///
    /// `ev_bk_b - ev_bk_a` is everything that call issued to the device.
    fn batchKTimingEnd(self: *CudaBackend) void {
        const lib = if (self.cu) |*cu| &cu.lib else return;
        if (lib.EventRecord(self.ev_bk_b, null) != cudart.cudaSuccess) self.stat_n_ev_err += 1;
        if (self.evElapsedNs(self.ev_bk_a, self.ev_bk_b)) |ns| {
            self.stat_ns_batchk_gpu += ns;
            self.stat_n_batchk_gpu += 1;
        }
        // Host wall clock spanned by the batchk calls of the prefill window.  A
        // monotonic host clock, incremented under the same `prefill_open` gate,
        // so this measures "first timed batchk entry -> last timed batchk exit"
        // and its complement (`pre_wall - pre_bkwall`) is the part of prefill
        // that issued no matvec.
        if (self.prefill_open) {
            const now = Io.Clock.Timestamp.now(self.io, .awake);
            if (!self.bk_wall_started) {
                self.bk_wall_t0 = now;
                self.bk_wall_started = true;
            }
            const d = self.bk_wall_t0.durationTo(now).raw.nanoseconds;
            self.stat_ns_prefill_bk_wall = if (d > 0) @intCast(@min(d, std.math.maxInt(u64))) else 0;
        }
    }

    /// Open the prefill window.  The engine calls `profReset` immediately before
    /// prefill, so the host wall clock starts here too.
    fn openPrefillWindow(self: *CudaBackend) void {
        if (!self.ensureEvents()) {
            self.ev_on = false;
            return;
        }
        self.prefill_open = true;
        self.ev_pre_started = false;
        self.bk_wall_started = false;
        self.prefill_t0 = Io.Clock.Timestamp.now(self.io, .awake);
    }

    /// Close the prefill window: record the end event, wait for it, and read out
    /// both the device span and the host wall clock of the same window.
    ///
    /// Called from the first single-token `matvec` after prefill (the decode
    /// path), and again, harmlessly, from `dProfPhase(false)` -- the second call
    /// finds the window already shut.  Waiting for the end event here is what
    /// the measurement costs: it drains the queue, so the following decode
    /// launch cannot overlap the read.  That cost lands *after* the number it
    /// measures, and the report quantifies the interference by comparing a
    /// PROF run against a plain one.
    fn closePrefillWindow(self: *CudaBackend) void {
        if (!self.prefill_open) return;
        self.prefill_open = false;
        const now = Io.Clock.Timestamp.now(self.io, .awake);
        const wall = self.prefill_t0.durationTo(now).raw.nanoseconds;
        self.stat_ns_prefill_wall = if (wall > 0) @intCast(@min(wall, std.math.maxInt(u64))) else 0;
        if (!self.ev_created or !self.ev_pre_started) {
            self.reportStats("prefill done");
            return;
        }
        if (self.cu) |*cu| {
            if (cu.lib.EventRecord(self.ev_pre_b, null) != cudart.cudaSuccess) {
                self.stat_n_ev_err += 1;
            } else if (self.evElapsedNs(self.ev_pre_a, self.ev_pre_b)) |ns| {
                self.stat_ns_prefill_span = ns;
            }
        }
        // `reportProfile` has no call site on the CLI path (the engine is
        // arena-allocated, so `deinit` never runs), so this is the only place the
        // fully-formed decomposition is ever printed.
        self.reportStats("prefill done");
    }

    /// `$QWEN35_PROF`: one `CUDA-M [...]` line with what actually ran on the device,
    /// plus a per-type breakdown, so "did this model mix backends?" is answerable
    /// from the log.  Uploaded MiB is the resident weight footprint; bufs_MiB is the
    /// reusable activation/output scratch.
    fn reportStats(self: *CudaBackend, when: []const u8) void {
        if (self.cu == null) return;
        const buf_bytes = self.d_act_bytes + self.d_xq_bytes + self.d_out_bytes + self.d_deq_bytes;
        std.debug.print(
            "CUDA-M  [{s}] matvec={d} batchk={d} dequant_row={d} up_MiB={d} h2d_kB={d} d2h_kB={d} resident_rows={d} bufs_MiB={d} iq_tbl_kB={d} t_mv_ms={d} t_bk_ms={d} t_dq_ms={d} hostpaths act={d} out={d} rope={d} rope_dev={d} bk_gpu_ms={d} bk_n={d} pre_wall_ms={d} pre_bkwall_ms={d} pre_span_ms={d} bk_launch={d} ev_sync_ms={d} gpu_ev_err={d}\n",
            .{
                when,
                self.stat_n_matvec,
                self.stat_n_batchk,
                self.stat_n_dequant,
                self.stat_uploaded_bytes >> 20,
                self.stat_h2d_bytes >> 10,
                self.stat_d2h_bytes >> 10,
                self.dev_rows.items.len,
                buf_bytes >> 20,
                self.iqTablesBytes() >> 10,
                self.stat_ns_matvec / 1_000_000,
                self.stat_ns_batchk / 1_000_000,
                self.stat_ns_dequant / 1_000_000,
                // Non-zero here means a numeric slot was handed host memory.  On
                // the per-token path all three must be 0 except `rope`, which is
                // the documented remainder.  `rope` / `rope_dev` partition the
                // same rotations between the two `$QWEN35_CUDA_ROPE` paths, so
                // the line itself proves which one ran (device: rope=0,
                // rope_dev>0; host: the reverse).
                self.stat_host_act_uploads,
                self.stat_host_out_downloads,
                self.stat_rope_trips,
                self.stat_rope_devices,
                // GPU-side numbers from CUDA events ($QWEN35_PROF only; 0
                // without it, because no event is ever recorded then).
                //   bk_gpu_ms     device time summed over timed batchk calls --
                //                 the actual GPU execution of the matmuls +
                //                 their activation quantize kernels
                //   bk_n          how many calls that sum covers
                //   pre_wall_ms   host wall clock of the prefill window
                //   pre_bkwall_ms host wall spanned by the batchk calls
                //   pre_gpu_span_ms
                //                 device event span of the whole window; equals
                //                 pre_wall_ms up to the drain, so it is a
                //                 sanity check (they must agree), not the busy
                //                 number
                // Decomposition of pre_wall:
                //   bk_gpu        GPU busy in matvecBatchK
                //   bk_host       pre_bkwall - bk_gpu  (host slack between
                //                 batchk calls: quantize/upload/download/launch)
                //   non_bk        pre_wall - pre_bkwall (attention, GDN, rmsnorm,
                //                 sampling -- issued outside matvecBatchK)
                self.stat_ns_batchk_gpu / 1_000_000,
                self.stat_n_batchk_gpu,
                self.stat_ns_prefill_wall / 1_000_000,
                self.stat_ns_prefill_bk_wall / 1_000_000,
                self.stat_ns_prefill_span / 1_000_000,
                self.stat_n_launch_bk,
                self.stat_ns_ev_sync / 1_000_000,
                self.stat_n_ev_err,
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

        // Per-type wall time, largest first.  `stat_ns_matvec` includes the
        // blocking activation upload and the output download, so these sums add
        // up to more than the kernel time -- that is deliberate: the question
        // being answered is "where does the wall clock go per type".
        var tline: [448]u8 = undefined;
        var tn: usize = 0;
        tn += (std.fmt.bufPrint(tline[tn..], "CUDA-M  time:", .{}) catch return).len;
        const total: u64 = blk: {
            var s: u64 = 0;
            for (self.stat_ns_by_type) |v| s += v;
            break :blk s;
        };
        for (0..self.stat_by_type.len) |i| {
            const ns = self.stat_ns_by_type[i];
            if (ns == 0) continue;
            const tt: gguf.TensorType = @fromBackingInt(@intCast(@as(u32, @intCast(i))));
            const pct: u64 = if (total == 0) 0 else ns * 100 / total;
            tn += (std.fmt.bufPrint(tline[tn..], " {s}={d}ms/{d}%", .{
                @tagName(tt), ns / 1_000_000, pct,
            }) catch break).len;
        }
        std.debug.print("{s}\n", .{tline[0..tn]});
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

    /// End of the prefill loop: the one point where "prefill" is known to have
    /// finished on the host side.  Events recorded here are still ordered after
    /// every prefill kernel in the default stream, so waiting for the end event
    /// cannot truncate the span.
    fn dProfPrefillEnd(ctx: *anyopaque) void {
        self_(ctx).closePrefillWindow();
    }

    fn dProfPhase(ctx: *anyopaque, on: bool) void {
        const self = self_(ctx);
        self.cpu.exec().profPhase(on);
        // Safety net only: the prefill window is normally already shut by the
        // first decode `matvec` (`closePrefillWindow` is idempotent).  This
        // catches the degenerate run where prefill issued no decode matvec at
        // all (e.g. `-n 0`), so the `CUDA-M [prefill done]` line still appears.
        if (!on) self.closePrefillWindow();
    }

    fn dProfReset(ctx: *anyopaque) void {
        const self = self_(ctx);
        self.stat_ns_matvec = 0;
        self.stat_ns_batchk = 0;
        self.stat_ns_dequant = 0;
        self.stat_n_matvec = 0;
        self.stat_n_batchk = 0;
        self.stat_n_dequant = 0;
        self.stat_h2d_bytes = 0;
        self.stat_d2h_bytes = 0;
        self.stat_host_act_uploads = 0;
        self.stat_host_out_downloads = 0;
        self.stat_rope_trips = 0;
        self.stat_by_type = @splat(0);
        self.stat_ns_by_type = @splat(0);
        // GPU-side counters and the prefill window.  `ev_on` is set here and
        // nowhere else, and `profReset` is only reached from
        // `Engine.generate`, which runs it unconditionally -- so the gate is the
        // environment variable, checked once, and the default path never
        // creates an event.
        self.stat_ns_batchk_gpu = 0;
        self.stat_n_batchk_gpu = 0;
        self.stat_ns_prefill_span = 0;
        self.stat_ns_prefill_wall = 0;
        self.stat_ns_prefill_bk_wall = 0;
        self.stat_n_ev_err = 0;
        self.stat_n_launch_bk = 0;
        self.stat_ns_ev_sync = 0;
        if (cudart.getenv("QWEN35_PROF") != null) {
            self.ev_on = self.cu != null;
            if (self.ev_on) self.openPrefillWindow();
        } else {
            self.ev_on = false;
        }
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
    .profPrefillEnd = CudaBackend.dProfPrefillEnd,
    .setTier = CudaBackend.dSetTier,
    .reportProfile = CudaBackend.dReportProfile,
    .name = CudaBackend.vName,
    .describe = CudaBackend.vDescribe,
    // Elementwise / state slots: ALL on the device.
    //
    // P2b left seven of these on the CPU because every call was an H2D + D2H
    // round trip of a few KB, and for the small hot buffers that round trip cost
    // more than the CPU work it replaced (measured: binding all nine sank
    // Generate from 38.7 to 22.1 tok/s).  P2c removed the round trip -- both
    // sides are device memory now -- so the bindings below cost a launch and
    // nothing else, and the measurement inverts (see the report).
    //
    // rope64 stays on the CPU on purpose: it is the one slot whose device
    // implementation cannot match the CPU bit-for-bit (sin/cos of a different
    // libm), and a per-head 64-element call is not worth any divergence in the
    // token stream.  See `dRope64`'s comment.
    .rmsnorm = CudaBackend.dRmsnorm,
    .rope64 = CudaBackend.dRope64,
    .l2norm = CudaBackend.dL2norm,
    .siluInPlace = CudaBackend.dSiluInPlace,
    .mulSilu = CudaBackend.dMulSilu,
    .mulSigmoid = CudaBackend.dMulSigmoid,
    .softmax = CudaBackend.dSoftmax,
    .swiglu = CudaBackend.dSwiglu,
    .convStep = CudaBackend.dConvStep,
    .gdnStep = CudaBackend.dGdnStep,
    .constBuf = cpu_mod.CpuBackend.vtable.constBuf,
    .attnDecode = CudaBackend.dAttnDecode,
    .gdnGamma = CudaBackend.dGdnGamma,
    .copyFromHost = CudaBackend.dCopyFromHost,
    .copyToHost = CudaBackend.dCopyToHost,
    .argmax = CudaBackend.dArgmax,
    .sample = CudaBackend.dSample,
    // The four pure elementwise slots.  Forwarders, NOT aliases: they need
    // `ctx` to reach the device, and the ctx here is a `*CudaBackend`
    // (aliasing a CPU body would work only because those bodies ignore ctx).
    // Each still falls back to the CPU path on a device-side failure.
    .scaleVec = CudaBackend.dScaleVec,
    .addResid = CudaBackend.dAddResid,
    .copyRange = CudaBackend.dCopyRange,
    .zeroBuf = CudaBackend.dZeroBuf,
    // Activations are PROPER device regions after P2c, so `allocBuf` is a real
    // `cudaMalloc` and `freeScratch` releases them (`dAllocBuf`/`dFreeScratch`).
    .allocBuf = CudaBackend.dAllocBuf,

};
