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
//!     resolving the GGUF tensor, building the `QuantRow` view, and the geometry
//!     it validates at load time.  That is startup work over the mmap, not a
//!     forward-pass computation.  Only `.f32` tensors are widened; f16/bf16
//!     weights stay in their native 2-byte form (see `loadLinear` in
//!     `backend_cpu.zig`) and are uploaded 2 B/elem, widened inside the kernel
//!     by `mmvq.cu`'s w16 family.
//!   * Every *numeric* operation the engine asks for -- matvec, K-token batched
//!     matvec, row dequantization, and the activation scratch those need -- is
//!     done on the device by the kernels in `src/kernels/cuda`.  There is no CPU
//!     arithmetic fallback here: mixing backends per tensor makes a divergence
//!     impossible to attribute and every profile meaningless.
//!
//! Weight residency: `loadLinear` uploads *all* `LinearW.quant` types that the
//! CPU loader accepts (f16/bf16 included, at 2 B/elem) plus the `.f32` tensors,
//! and records the device pointer in
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
    /// Same, and additionally writes the `block_q8_1_mmq` reshape (288 B per
    /// 256-element block) that the dp4a prefill kernel reads.  See
    /// `q35_k_quantize_q8_K` in mmvq.cu.
    extern fn q35_launch_quantize_q8_K_mmq(d_x: ?[*]const f32, d_out: ?*anyopaque, d_mmq: ?*anyopaque, nblocks: c_int, stream: ?*anyopaque) callconv(.c) c_int;
    /// 1 when the prefill matvec for this type runs on the dp4a kernel, which
    /// needs the mmq activation layout rather than q8_K.
    extern fn q35_mmq_dp4a_wants(ttype: c_int) c_int;
    /// Bytes per 256-element block of the mmq activation layout (288).
    extern fn q35_mmq_act_block_bytes() c_int;

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
    /// f16/bf16 weights in their native 2-byte form x f32 activation.
    /// `d_w` is rows*cols u16 (2 B/elem) -- NO f32 copy on the device; the
    /// dequant happens inside the kernel (mmvq.cu's w16 header).
    extern fn q35_launch_mul_mat_vec_w16(d_w: ?*const anyopaque, d_x: ?[*]const f32, d_out: ?[*]f32, rows: c_int, cols: c_int, tokens: c_int, ttype: c_int, stream: ?*anyopaque) callconv(.c) c_int;

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
    extern fn q35_launch_deinterleave_qg(d_q: ?[*]f32, d_gate: ?[*]f32, d_src: ?[*]const f32, heads: c_int, hd: c_int, stream: ?*anyopaque) callconv(.c) c_int;
    extern fn q35_launch_deinterleave_qg_strided(d_q: ?[*]f32, d_gate: ?[*]f32, d_src: ?[*]const f32, heads: c_int, hd: c_int, tokens: c_int, stream: ?*anyopaque) callconv(.c) c_int;
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

    // Device sampler (sample.cu).  The per-token parameters -- temperature,
    // top_k, top_p, PRNG state -- travel through `d_ctl`, a RESIDENT device
    // control block rewritten by a 64-byte H2D before each replay, so a captured
    // graph node never needs its parameters retargeted.  `n` (vocab) is a launch
    // argument because it decides the grid shape and never changes.
    extern fn q35_launch_sample(d_logits: ?[*]const f32, n: c_int, d_ctl: ?*c_int, d_scratch: ?*anyopaque, stream: ?*anyopaque) callconv(.c) c_int;
    extern fn q35_launch_sample_init(d_ctl: ?*?*anyopaque, d_scratch: ?*?*anyopaque) callconv(.c) c_int;
    extern fn q35_launch_sample_deinit(d_ctl: ?*?*anyopaque, d_scratch: ?*?*anyopaque) callconv(.c) void;
    extern fn q35_sample_scratch_ints() callconv(.c) c_int;
    extern fn q35_sample_ctl_ints() callconv(.c) c_int;

    // P2c: the two state machines.  One launch per value head for gdn_step,
    // one launch (block per head) for attn_decode; see the kernel headers.
    extern fn q35_launch_gdn_step(d_state: ?[*]f32, d_k: ?[*]const f32, d_v: ?[*]const f32, d_q: ?[*]const f32, d_km: ?[*]f32, d_delta: ?[*]f32, d_out: ?[*]f32, gamma: f32, beta: f32, vd: c_int, stream: ?*anyopaque) callconv(.c) c_int;
    extern fn q35_launch_gdn_step_rows(d_state: ?[*]f32, state_stride: c_int, d_k: ?[*]const f32, k_stride: c_int, d_v: ?[*]const f32, v_stride: c_int, d_q: ?[*]const f32, q_stride: c_int, d_km: ?[*]f32, d_delta: ?[*]f32, d_out: ?[*]f32, d_gamma: ?[*]const f32, d_beta: ?[*]const f32, heads: c_int, nk: c_int, kd: c_int, vd: c_int, stream: ?*anyopaque) callconv(.c) c_int;
    extern fn q35_launch_gdn_step_rows_chunk(d_state: ?[*]f32, state_stride: c_int, d_k: ?[*]const f32, k_stride: c_int, d_v: ?[*]const f32, v_stride: c_int, d_q: ?[*]const f32, q_stride: c_int, d_km: ?[*]f32, km_stride: c_int, d_delta: ?[*]f32, delta_stride: c_int, d_out: ?[*]f32, out_stride: c_int, d_gamma: ?[*]const f32, gamma_stride: c_int, d_beta: ?[*]const f32, beta_stride: c_int, heads: c_int, nk: c_int, kd: c_int, vd: c_int, rows: c_int, stream: ?*anyopaque) callconv(.c) c_int;
    extern fn q35_launch_gdn_gamma(d_a_raw: ?[*]const f32, d_b_raw: ?[*]f32, d_ssm_a: ?[*]const f32, d_dt_bias: ?[*]const f32, d_gamma: ?[*]f32, nv: c_int, stream: ?*anyopaque) callconv(.c) c_int;

    // Prefill (batched) twins of conv_step / gdn_gamma / l2norm / scale_vec.
    // These exist only to collapse the per-token GDN prelude into a handful of
    // launches per chunk -- they are NOT part of the decode graph (prefill never
    // captures), so they are called directly and deliberately kept out of
    // `Launch`.  Each one is bit-identical to the per-token kernel it replaces.
    extern fn q35_launch_conv_step_rows(d_conv: ?[*]f32, d_y: ?[*]f32, d_wconv: ?[*]const f32, n: c_int, tokens: c_int, y_stride: c_int, stream: ?*anyopaque) callconv(.c) c_int;
    extern fn q35_launch_gdn_gamma_rows(d_a_raw: ?[*]const f32, d_b_raw: ?[*]f32, d_ssm_a: ?[*]const f32, d_dt_bias: ?[*]const f32, d_gamma: ?[*]f32, nv: c_int, tokens: c_int, a_stride: c_int, b_stride: c_int, gamma_stride: c_int, stream: ?*anyopaque) callconv(.c) c_int;
    extern fn q35_launch_l2norm_strided(d_x: ?[*]f32, row_offset: c_int, rows: c_int, n: c_int, eps: f32, tokens: c_int, row_stride: c_int, stream: ?*anyopaque) callconv(.c) c_int;
    extern fn q35_launch_scale_vec_strided(d_x: ?[*]f32, offset: c_int, s: f32, n: c_int, tokens: c_int, stride: c_int, stream: ?*anyopaque) callconv(.c) c_int;
    extern fn q35_launch_attn_decode(d_q: ?[*]const f32, d_gate: ?[*]const f32, d_kcache: ?[*]const f32, d_vcache: ?[*]const f32, d_att: ?[*]f32, d_gout: ?[*]f32, slot: c_int, heads: c_int, gqa: c_int, hd: c_int, kv_dim: c_int, scale: f32, stream: ?*anyopaque) callconv(.c) c_int;
    extern fn q35_launch_attn_chunk(d_q: ?[*]const f32, d_gate: ?[*]const f32, d_kcache: ?[*]const f32, d_vcache: ?[*]const f32, d_gout: ?[*]f32, slot0: c_int, rows: c_int, heads: c_int, gqa: c_int, hd: c_int, kv_dim: c_int, scale: f32, stream: ?*anyopaque) callconv(.c) c_int;
    extern fn q35_attn_chunk_max_rows(slot0: c_int) callconv(.c) c_int;
    /// Warp-tiled / split decode attention (see api.h for the measurements).
    extern fn q35_launch_attn_decode_coal(d_q: ?[*]const f32, d_gate: ?[*]const f32, d_kcache: ?[*]const f32, d_vcache: ?[*]const f32, d_att: ?[*]f32, d_gout: ?[*]f32, slot: c_int, heads: c_int, gqa: c_int, hd: c_int, kv_dim: c_int, scale: f32, stream: ?*anyopaque) callconv(.c) c_int;
    extern fn q35_launch_attn_decode_split_dyn(d_q: ?[*]const f32, d_gate: ?[*]const f32, d_kcache: ?[*]const f32, d_vcache: ?[*]const f32, d_att: ?[*]f32, d_gout: ?[*]f32, d_part: ?[*]f32, d_slot: ?[*]const c_int, heads: c_int, gqa: c_int, hd: c_int, kv_dim: c_int, scale: f32, split: c_int, stream: ?*anyopaque) callconv(.c) c_int;
    extern fn q35_func_attn_score_split_dyn() callconv(.c) ?*anyopaque;
    extern fn q35_func_attn_pv_split_dyn() callconv(.c) ?*anyopaque;
    extern fn q35_func_attn_softmax_dyn() callconv(.c) ?*anyopaque;
    extern fn q35_func_attn_reduce() callconv(.c) ?*anyopaque;
    extern fn q35_launch_attn_decode_split2(d_q: ?[*]const f32, d_gate: ?[*]const f32, d_kcache: ?[*]const f32, d_vcache: ?[*]const f32, d_att: ?[*]f32, d_gout: ?[*]f32, d_part: ?[*]f32, slot: c_int, heads: c_int, gqa: c_int, hd: c_int, kv_dim: c_int, scale: f32, split: c_int, stream: ?*anyopaque) callconv(.c) c_int;
    /// Device-`slot` form of the same kernel; see `dAttnDecode_body`.
    extern fn q35_launch_attn_decode_dyn(d_q: ?[*]const f32, d_gate: ?[*]const f32, d_kcache: ?[*]const f32, d_vcache: ?[*]const f32, d_att: ?[*]f32, d_gout: ?[*]f32, d_slot: ?[*]const c_int, heads: c_int, gqa: c_int, hd: c_int, kv_dim: c_int, scale: f32, stream: ?*anyopaque) callconv(.c) c_int;
    /// Write one int32 from a 1-thread kernel (graph-safe per-token scalar).
    extern fn q35_launch_set_int1(d: ?[*]c_int, v: c_int, stream: ?*anyopaque) callconv(.c) c_int;
    /// rope64 over `heads` heads of `hd` floats (only the first 64 rotate).
    extern fn q35_launch_rope64(d_x: ?[*]f32, heads: c_int, hd: c_int, pos: c_int, base: f32, stream: ?*anyopaque) callconv(.c) c_int;
    extern fn q35_launch_rope64_dpos(d_x: ?[*]f32, heads: c_int, hd: c_int, d_dyn: ?*const anyopaque, stream: ?*anyopaque) callconv(.c) c_int;
    extern fn q35_launch_rope64_setenv(d_dyn: ?*anyopaque, pos: c_int, base: f32, stream: ?*anyopaque) callconv(.c) c_int;
    /// Row-batched rope64: `rows` tokens `pos0..pos0+rows-1`, each `heads * hd`
    /// floats wide.  PREFILL ONLY (see the note on `VTable.rope64Rows`).
    extern fn q35_launch_rope64_rows(d_x: ?[*]f32, rows: c_int, heads: c_int, hd: c_int, pos0: c_int, base: f32, stream: ?*anyopaque) callconv(.c) c_int;
    /// Host-stub address of the device kernel `q35_launch_rope64` launches.  A
    /// captured graph node reports exactly this value as `KernelNodeParams.func`,
    /// so comparing against it is how the decode-graph path finds the node whose
    /// `pos` it has to update every token.  Reading the argument values instead
    /// is not equivalent: a node's `kernelParams` array has no length, so
    /// probing it reads past the end of kernels with fewer arguments and
    /// segfaults (observed).
    extern fn q35_func_rope64() callconv(.c) ?*anyopaque;
    extern fn q35_func_rope64_dpos() callconv(.c) ?*anyopaque;
    extern fn q35_func_attn_decode() callconv(.c) ?*anyopaque;
    extern fn q35_func_attn_decode_dyn() callconv(.c) ?*anyopaque;
    extern fn q35_func_copy_range() callconv(.c) ?*anyopaque;
};

/// Row-range split factor for decode attention.  `0` disables it (one block per
/// head, the pre-2026-11 behaviour), `2..16` divides BOTH row walks of every
/// head across that many blocks.
///
/// Default is `16` because it is a strict win and it does not change any result:
/// the score sum keeps the serial `j` order, the softmax keeps the CPU's
/// max/serial-sum/div order, and the PV t-sum is the only 2-level tree -- and
/// even that came out bit-identical on the probe (`mismatched=0 worst=0` against
/// the one-block reference for every head and slot 0/5/63/300/516-519/530/700/
/// 1023).  A single block per head is 24 blocks on this GPU's 114 SMs (21%
/// occupancy) and every lane then walks all `n` rows of a KV cache that does not
/// fit in L2, which measured **33 GB/s**; splitting reaches **325 GB/s**.
///
/// `$QWEN35_CUDA_ATTN_SPLIT` overrides it (0 to turn it off for A/B).
fn attnSplitFromEnv() usize {
    const v = cudart.getenv("QWEN35_CUDA_ATTN_SPLIT") orelse return 16;
    const n = std.fmt.parseInt(usize, std.mem.trim(u8, v, " \t"), 10) catch return 16;
    return @min(n, 16);
}


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
    const quantize_q8_k_mmq = Kernels.q35_launch_quantize_q8_K_mmq;
    const mmq_dp4a_wants = Kernels.q35_mmq_dp4a_wants;
    const mmq_act_block_bytes = Kernels.q35_mmq_act_block_bytes;
    const dequantize = Kernels.q35_launch_dequantize;
    const dequantize_row = Kernels.q35_launch_dequantize_row;
    const mul_mat_vec = Kernels.q35_launch_mul_mat_vec;
    const mul_mat_vec_f32 = Kernels.q35_launch_mul_mat_vec_f32;
    const mul_mat_vec_w16 = Kernels.q35_launch_mul_mat_vec_w16;
    const init_tables = Kernels.q35_launch_init_tables;
    const deinit_tables = Kernels.q35_launch_deinit_tables;
    const tables_bytes = Kernels.q35_iq_tables_bytes;
    const scale_vec = Kernels.q35_launch_scale_vec;
    const add_resid = Kernels.q35_launch_add_resid;
    const copy_range = Kernels.q35_launch_copy_range;
    const deinterleave_qg = Kernels.q35_launch_deinterleave_qg;
    const deinterleave_qg_strided = Kernels.q35_launch_deinterleave_qg_strided;
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
    const sample = Kernels.q35_launch_sample;
    const sample_init = Kernels.q35_launch_sample_init;
    const sample_deinit = Kernels.q35_launch_sample_deinit;
    const sample_scratch_ints = Kernels.q35_sample_scratch_ints;
    const sample_ctl_ints = Kernels.q35_sample_ctl_ints;
    const attn_decode = Kernels.q35_launch_attn_decode;
    const attn_chunk = Kernels.q35_launch_attn_chunk;
    const attn_decode_dyn = Kernels.q35_launch_attn_decode_dyn;
    const attn_decode_coal = Kernels.q35_launch_attn_decode_coal;
    const attn_decode_split2 = Kernels.q35_launch_attn_decode_split2;
    const attn_decode_split_dyn = Kernels.q35_launch_attn_decode_split_dyn;
    const set_int1 = Kernels.q35_launch_set_int1;
    const gdn_step = Kernels.q35_launch_gdn_step;
    const gdn_step_rows = Kernels.q35_launch_gdn_step_rows;
    const gdn_step_rows_chunk = Kernels.q35_launch_gdn_step_rows_chunk;
    const gdn_gamma = Kernels.q35_launch_gdn_gamma;
    // Prefill-only row/strided twins.  Listed here (and in the stub branch, in
    // the same order) so `Launch` is the only seam between a launcher and its
    // no-CUDA fallback: the `dGdnPrefillCore` call sites used to name
    // `Kernels.q35_launch_*` directly, which made those 4 stubs dead code and
    // broke `zig build` (without -Dcuda) at LINK time on an undefined symbol -- a
    // class of failure the type errors above cannot catch.
    const conv_step_rows = Kernels.q35_launch_conv_step_rows;
    const gdn_gamma_rows = Kernels.q35_launch_gdn_gamma_rows;
    const l2norm_strided = Kernels.q35_launch_l2norm_strided;
    const scale_vec_strided = Kernels.q35_launch_scale_vec_strided;
    const rope64 = Kernels.q35_launch_rope64;
    const rope64_dpos = Kernels.q35_launch_rope64_dpos;
    const rope64_setenv = Kernels.q35_launch_rope64_setenv;
    const rope64_rows = Kernels.q35_launch_rope64_rows;
} else struct {
    fn quantize_q8_0(d_x: ?[*]const f32, d_out: ?*anyopaque, nblocks: c_int, stream: ?*anyopaque) callconv(.c) c_int {
        _ = .{ d_x, d_out, nblocks, stream };
        return stubNoCudaKernels();
    }
    fn quantize_q8_k(d_x: ?[*]const f32, d_out: ?*anyopaque, nblocks: c_int, stream: ?*anyopaque) callconv(.c) c_int {
        _ = .{ d_x, d_out, nblocks, stream };
        return stubNoCudaKernels();
    }
    fn quantize_q8_k_mmq(d_x: ?[*]const f32, d_out: ?*anyopaque, d_mmq: ?*anyopaque, nblocks: c_int, stream: ?*anyopaque) callconv(.c) c_int {
        _ = .{ d_x, d_out, d_mmq, nblocks, stream };
        return stubNoCudaKernels();
    }
    /// No CUDA kernels means no dp4a kernel either, so the caller keeps the
    /// plain q8_K path.  0 is the answer that requires no second buffer.
    fn mmq_dp4a_wants(ttype: c_int) c_int {
        _ = ttype;
        return 0;
    }
    /// Never used (mmq_dp4a_wants is always 0 here) but kept so `Launch` has the
    /// same shape in both branches.
    fn mmq_act_block_bytes() c_int {
        return 288;
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
    fn mul_mat_vec_w16(d_w: ?*const anyopaque, d_x: ?[*]const f32, d_out: ?[*]f32, rows: c_int, cols: c_int, tokens: c_int, ttype: c_int, stream: ?*anyopaque) callconv(.c) c_int {
        _ = .{ d_w, d_x, d_out, rows, cols, tokens, ttype, stream };
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
    fn deinterleave_qg(d_q: ?[*]f32, d_gate: ?[*]f32, d_src: ?[*]const f32, heads: c_int, hd: c_int, stream: ?*anyopaque) callconv(.c) c_int {
        _ = .{ d_q, d_gate, d_src, heads, hd, stream };
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
    fn sample(d_logits: ?[*]const f32, n: c_int, d_ctl: ?*c_int, d_scratch: ?*anyopaque, stream: ?*anyopaque) callconv(.c) c_int {
        _ = .{ d_logits, n, d_ctl, d_scratch, stream };
        return stubNoCudaKernels();
    }
    fn sample_init(d_ctl: ?*?*anyopaque, d_scratch: ?*?*anyopaque) callconv(.c) c_int {
        _ = .{ d_ctl, d_scratch };
        return stubNoCudaKernels();
    }
    fn sample_deinit(d_ctl: ?*?*anyopaque, d_scratch: ?*?*anyopaque) callconv(.c) void {
        _ = .{ d_ctl, d_scratch };
    }
    fn sample_scratch_ints() callconv(.c) c_int {
        return 0;
    }
    fn sample_ctl_ints() callconv(.c) c_int {
        return 0;
    }
    fn attn_decode(d_q: ?[*]const f32, d_gate: ?[*]const f32, d_kcache: ?[*]const f32, d_vcache: ?[*]const f32, d_att: ?[*]f32, d_gout: ?[*]f32, slot: c_int, heads: c_int, gqa: c_int, hd: c_int, kv_dim: c_int, scale: f32, stream: ?*anyopaque) callconv(.c) c_int {
        _ = .{ d_q, d_gate, d_kcache, d_vcache, d_att, d_gout, slot, heads, gqa, hd, kv_dim, scale, stream };
        return stubNoCudaKernels();
    }
    fn attn_chunk(d_q: ?[*]const f32, d_gate: ?[*]const f32, d_kcache: ?[*]const f32, d_vcache: ?[*]const f32, d_gout: ?[*]f32, slot0: c_int, rows: c_int, heads: c_int, gqa: c_int, hd: c_int, kv_dim: c_int, scale: f32, stream: ?*anyopaque) callconv(.c) c_int {
        _ = .{ d_q, d_gate, d_kcache, d_vcache, d_gout, slot0, rows, heads, gqa, hd, kv_dim, scale, stream };
        return stubNoCudaKernels();
    }
    fn deinterleave_qg_strided(d_q: ?[*]f32, d_gate: ?[*]f32, d_src: ?[*]const f32, heads: c_int, hd: c_int, tokens: c_int, stream: ?*anyopaque) callconv(.c) c_int {
        _ = .{ d_q, d_gate, d_src, heads, hd, tokens, stream };
        return stubNoCudaKernels();
    }
    // Same order and same signatures as the `Kernels` aliases above: every
    // launcher the CUDA branch exposes must have a member here, or any call site
    // added later breaks `zig build` (no -Dcuda) with `no member named ...` --
    // which is exactly how `set_int1` and the three `attn_decode_*` variants got
    // away: they were aliased and called long after this struct was last read.
    fn attn_decode_dyn(d_q: ?[*]const f32, d_gate: ?[*]const f32, d_kcache: ?[*]const f32, d_vcache: ?[*]const f32, d_att: ?[*]f32, d_gout: ?[*]f32, d_slot: ?[*]const c_int, heads: c_int, gqa: c_int, hd: c_int, kv_dim: c_int, scale: f32, stream: ?*anyopaque) callconv(.c) c_int {
        _ = .{ d_q, d_gate, d_kcache, d_vcache, d_att, d_gout, d_slot, heads, gqa, hd, kv_dim, scale, stream };
        return stubNoCudaKernels();
    }
    fn attn_decode_coal(d_q: ?[*]const f32, d_gate: ?[*]const f32, d_kcache: ?[*]const f32, d_vcache: ?[*]const f32, d_att: ?[*]f32, d_gout: ?[*]f32, slot: c_int, heads: c_int, gqa: c_int, hd: c_int, kv_dim: c_int, scale: f32, stream: ?*anyopaque) callconv(.c) c_int {
        _ = .{ d_q, d_gate, d_kcache, d_vcache, d_att, d_gout, slot, heads, gqa, hd, kv_dim, scale, stream };
        return stubNoCudaKernels();
    }
    fn attn_decode_split2(d_q: ?[*]const f32, d_gate: ?[*]const f32, d_kcache: ?[*]const f32, d_vcache: ?[*]const f32, d_att: ?[*]f32, d_gout: ?[*]f32, d_part: ?[*]f32, slot: c_int, heads: c_int, gqa: c_int, hd: c_int, kv_dim: c_int, scale: f32, split: c_int, stream: ?*anyopaque) callconv(.c) c_int {
        _ = .{ d_q, d_gate, d_kcache, d_vcache, d_att, d_gout, d_part, slot, heads, gqa, hd, kv_dim, scale, split, stream };
        return stubNoCudaKernels();
    }
    fn attn_decode_split_dyn(d_q: ?[*]const f32, d_gate: ?[*]const f32, d_kcache: ?[*]const f32, d_vcache: ?[*]const f32, d_att: ?[*]f32, d_gout: ?[*]f32, d_part: ?[*]f32, d_slot: ?[*]const c_int, heads: c_int, gqa: c_int, hd: c_int, kv_dim: c_int, scale: f32, split: c_int, stream: ?*anyopaque) callconv(.c) c_int {
        _ = .{ d_q, d_gate, d_kcache, d_vcache, d_att, d_gout, d_part, d_slot, heads, gqa, hd, kv_dim, scale, split, stream };
        return stubNoCudaKernels();
    }
    fn set_int1(d: ?[*]c_int, v: c_int, stream: ?*anyopaque) callconv(.c) c_int {
        _ = .{ d, v, stream };
        return stubNoCudaKernels();
    }
    fn gdn_step(d_state: ?[*]f32, d_k: ?[*]const f32, d_v: ?[*]const f32, d_q: ?[*]const f32, d_km: ?[*]f32, d_delta: ?[*]f32, d_out: ?[*]f32, gamma: f32, beta: f32, vd: c_int, stream: ?*anyopaque) callconv(.c) c_int {
        _ = .{ d_state, d_k, d_v, d_q, d_km, d_delta, d_out, gamma, beta, vd, stream };
        return stubNoCudaKernels();
    }
    fn gdn_step_rows(d_state: ?[*]f32, state_stride: c_int, d_k: ?[*]const f32, k_stride: c_int, d_v: ?[*]const f32, v_stride: c_int, d_q: ?[*]const f32, q_stride: c_int, d_km: ?[*]f32, d_delta: ?[*]f32, d_out: ?[*]f32, d_gamma: ?[*]const f32, d_beta: ?[*]const f32, heads: c_int, nk: c_int, kd: c_int, vd: c_int, stream: ?*anyopaque) callconv(.c) c_int {
        _ = .{ d_state, state_stride, d_k, k_stride, d_v, v_stride, d_q, q_stride, d_km, d_delta, d_out, d_gamma, d_beta, heads, nk, kd, vd, stream };
        return stubNoCudaKernels();
    }
    fn gdn_step_rows_chunk(d_state: ?[*]f32, state_stride: c_int, d_k: ?[*]const f32, k_stride: c_int, d_v: ?[*]const f32, v_stride: c_int, d_q: ?[*]const f32, q_stride: c_int, d_km: ?[*]f32, km_stride: c_int, d_delta: ?[*]f32, delta_stride: c_int, d_out: ?[*]f32, out_stride: c_int, d_gamma: ?[*]const f32, gamma_stride: c_int, d_beta: ?[*]const f32, beta_stride: c_int, heads: c_int, nk: c_int, kd: c_int, vd: c_int, rows: c_int, stream: ?*anyopaque) callconv(.c) c_int {
        _ = .{ d_state, state_stride, d_k, k_stride, d_v, v_stride, d_q, q_stride, d_km, km_stride, d_delta, delta_stride, d_out, out_stride, d_gamma, gamma_stride, d_beta, beta_stride, heads, nk, kd, vd, rows, stream };
        return stubNoCudaKernels();
    }
    fn gdn_gamma(d_a_raw: ?[*]const f32, d_b_raw: ?[*]f32, d_ssm_a: ?[*]const f32, d_dt_bias: ?[*]const f32, d_gamma: ?[*]f32, nv: c_int, stream: ?*anyopaque) callconv(.c) c_int {
        _ = .{ d_a_raw, d_b_raw, d_ssm_a, d_dt_bias, d_gamma, nv, stream };
        return stubNoCudaKernels();
    }
    // Prefill-only twins, now reached through `Launch` like every other launcher
    // (see the note in the CUDA branch: they used to bypass the seam).
    fn conv_step_rows(d_conv: ?[*]f32, d_y: ?[*]f32, d_wconv: ?[*]const f32, n: c_int, tokens: c_int, y_stride: c_int, stream: ?*anyopaque) callconv(.c) c_int {
        _ = .{ d_conv, d_y, d_wconv, n, tokens, y_stride, stream };
        return stubNoCudaKernels();
    }
    fn gdn_gamma_rows(d_a_raw: ?[*]const f32, d_b_raw: ?[*]f32, d_ssm_a: ?[*]const f32, d_dt_bias: ?[*]const f32, d_gamma: ?[*]f32, nv: c_int, tokens: c_int, a_stride: c_int, b_stride: c_int, gamma_stride: c_int, stream: ?*anyopaque) callconv(.c) c_int {
        _ = .{ d_a_raw, d_b_raw, d_ssm_a, d_dt_bias, d_gamma, nv, tokens, a_stride, b_stride, gamma_stride, stream };
        return stubNoCudaKernels();
    }
    fn l2norm_strided(d_x: ?[*]f32, row_offset: c_int, rows: c_int, n: c_int, eps: f32, tokens: c_int, row_stride: c_int, stream: ?*anyopaque) callconv(.c) c_int {
        _ = .{ d_x, row_offset, rows, n, eps, tokens, row_stride, stream };
        return stubNoCudaKernels();
    }
    fn scale_vec_strided(d_x: ?[*]f32, offset: c_int, s: f32, n: c_int, tokens: c_int, stride: c_int, stream: ?*anyopaque) callconv(.c) c_int {
        _ = .{ d_x, offset, s, n, tokens, stride, stream };
        return stubNoCudaKernels();
    }
    fn rope64(d_x: ?[*]f32, heads: c_int, hd: c_int, pos: c_int, base: f32, stream: ?*anyopaque) callconv(.c) c_int {
        _ = .{ d_x, heads, hd, pos, base, stream };
        return stubNoCudaKernels();
    }
    fn rope64_dpos(d_x: ?[*]f32, heads: c_int, hd: c_int, d_dyn: ?*const anyopaque, stream: ?*anyopaque) callconv(.c) c_int {
        _ = .{ d_x, heads, hd, d_dyn, stream };
        return stubNoCudaKernels();
    }
    fn rope64_setenv(d_dyn: ?*anyopaque, pos: c_int, base: f32, stream: ?*anyopaque) callconv(.c) c_int {
        _ = .{ d_dyn, pos, base, stream };
        return stubNoCudaKernels();
    }
    fn rope64_rows(d_x: ?[*]f32, rows: c_int, heads: c_int, hd: c_int, pos0: c_int, base: f32, stream: ?*anyopaque) callconv(.c) c_int {
        _ = .{ d_x, rows, heads, hd, pos0, base, stream };
        return stubNoCudaKernels();
    }
};

// ==== activation geometry ====================================================

/// Activation block size (in *elements*) a matvec against `ttype` needs.
///
/// Two formats, not one, because a quantized matvec is table-driven per weight
/// type and the activation block must match the weight block:
///   * Q8_0 weights use 32-element blocks -> Q8_0 activations (34 B / 32).
///   * BF16/F16 weights (one element per 2-byte block, framed into 256-element
///     GGUF row-blocks) are dotted by the Q8_0 family too -- see
///     `backend_cpu.actFamily`, which is the single source for this.
///   * Every other packed type this backend loads uses 256-element blocks
///     (Q4_K/Q6_K/IQ*/... all consume Q8_K) -> Q8_K (292 B / 256).
///
/// Keying on `gguf.blockElems` alone was a live bug: BF16/F16 have
/// `blockElems == 1`, so they fell into the Q8_K family and the launcher
/// sized their activation as 292 B / 256 elements while the weight dot reads
/// Q8_0 blocks.
fn actBlockElems(ttype: gguf.TensorType) usize {
    return if (cpu_mod.actFamily(ttype) == .q8_0)
        gguf.blockElems(.q8_0)
    else
        gguf.blockElems(.q8_k);
}

/// Packed bytes of one activation block for `ttype`'s matvec input.
fn actBlockBytes(ttype: gguf.TensorType) usize {
    return if (cpu_mod.actFamily(ttype) == .q8_0)
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

// ---- per-slot profiling ($QWEN35_PROF only) --------------------------------

/// The `backend.Exec` slots, one entry each, plus the two weight/scratch slots
/// that are not part of the per-token budget but are worth seeing (they are
/// the load-time and allocation costs).
///
/// Order is the report order, so keep the hot entries first when re-reading a
/// `PROF-SLOT` dump; `reportSlotStats` sorts by total anyway.
pub const Slot = enum(u8) {
    matvec,
    matvec_batchk,
    dequant_row,
    load_linear,
    rmsnorm,
    rope64,
    rope64_rows,
    l2norm,
    attn_decode,
    /// Chunk-wide prefill attention (`VTable.attnChunk`).  Separate from
    /// `attn_decode` so the profile shows the two shapes apart: prefill should
    /// drop to a handful of calls per layer while decode keeps one per token.
    attn_chunk,
    gdn_gamma,
    gdn_prelude,
    gdn_step,
    conv_step,
    silu,
    mul_silu,
    mul_sigmoid,
    softmax,
    swiglu,
    scale_vec,
    add_resid,
    copy_range,
    deinterleave_qg,
    zero_buf,
    copy_from_host,
    copy_to_host,
    argmax,
    sample,
    alloc_buf,
    alloc_scratch,
    alloc_scratch_k,
    const_buf,
};

/// Element-count buckets for the `PROF-COPY` breakdown.  Fixed, not dynamic:
/// the widths this model actually issues are a handful of distinct values, and
/// a fixed array keeps the instrumentation allocation-free on the hot path.
///
/// Order is significant only in that `copyBinOf` scans it in order; the bounds
/// are ascending so the first match is the tightest bucket.
const copy_bin_bounds = [_]usize{ 128, 256, 512, 1024, 2048, 4096 };
const copy_bin_count = copy_bin_bounds.len + 1;


/// One `copyRange` family: how many calls and how long they took on the host.
const CopyBin = struct {
    n: u32 = 0,
    ns: u64 = 0,
    /// Elements moved, summed -- divided by `n` it gives the per-call payload,
    /// which is how a bucket reveals whether it is the 512-B per-head family or
    /// a whole embedding row.
    elems: u64 = 0,
};

/// Tightest bucket for an element count (`copy_bin_count-1` = bigger than the
/// last bound, e.g. a `kv_dim` cache write on the 27B).
fn copyBinOf(elems: usize) usize {
    for (copy_bin_bounds, 0..) |b, i| {
        if (elems <= b) return i;
    }
    return copy_bin_bounds.len;
}

/// Number of `Slot` entries; sizes `stat_ns_slot` / `stat_n_slot`.
///
/// Spelled out rather than derived from `@typeInfo`: Zig 0.17 reworked
/// `@typeInfo`, and `slotName`'s exhaustive `switch` already makes a forgotten
/// enum member a compile error, so this only has to stay in sync by being one
/// past the last enumerator (`const_buf`).
const slot_count = @intFromEnum(Slot.const_buf) + 1;

/// Event pairs available for `$QWEN35_PROF_GPU` per-slot device timing.  A decode
/// token issues ~600 timed slot calls (measured: 599 on the 0.8B), so 1024 pairs
/// cover one token with headroom; a token that overflows is counted in
/// `stat_gpu_overflow` rather than dropped silently.
///
/// The 27B needs far more: a decode token issues ~2273 timed slot calls (measured
/// from PROF-SLOT: 111k calls / 49 tokens).  With 1024 pairs the pool overflowed
/// by 93566, so the PROF-GPU breakdown was unusable on 27B and the device-side
/// attribution had to be inferred from QWEN35_ABLATE instead of measured.  4096
/// pairs cover one 27B token with headroom.  This is a debug-only path
/// (`$QWEN35_PROF_GPU` in the environment); the events are created lazily by
/// `gpuEvEnsure` and destroyed on teardown, so raising it costs nothing at
/// runtime outside a profiling run.
///
/// RAISED AGAIN: the 27B decode token issues ~2273 *timed* calls, but each
/// call records a pair and the pool is also consumed by the draining slot, so
/// 4096 still overflowed (measured: 7519 in one 60-token run, i.e. the
/// attribution below 0.5 ms was being silently dropped).  16384 is the next
/// power of two with real headroom; it costs 2 * 16384 * sizeof(cudaEvent_t)
/// in the debug path only.
const gpu_ev_capacity = 16384;

/// Human-readable slot names for the `PROF-SLOT` line.  Indexed by
/// `@intFromEnum`, so a `comptime`-checked length ties them to the enum.
fn slotName(s: Slot) []const u8 {
    return switch (s) {
        .matvec => "matvec",
        .matvec_batchk => "mv_batchk",
        .dequant_row => "dequant_row",
        .load_linear => "load_linear",
        .rmsnorm => "rmsnorm",
        .rope64 => "rope64",
        .rope64_rows => "rope64_rows",
        .l2norm => "l2norm",
        .attn_decode => "attn_decode",
        .attn_chunk => "attn_chunk",
        .gdn_gamma => "gdn_gamma",
        .gdn_prelude => "gdn_prelude",
        .gdn_step => "gdn_step",
        .conv_step => "conv_step",
        .silu => "silu",
        .mul_silu => "mul_silu",
        .mul_sigmoid => "mul_sigmoid",
        .softmax => "softmax",
        .swiglu => "swiglu",
        .scale_vec => "scale_vec",
        .add_resid => "add_resid",
        .copy_range => "copy_range",
        .deinterleave_qg => "deinterleave_qg",
        .zero_buf => "zero_buf",
        .copy_from_host => "copy_from_host",
        .copy_to_host => "copy_to_host",
        .argmax => "argmax",
        .sample => "sample",
        .alloc_buf => "alloc_buf",
        .alloc_scratch => "alloc_scratch",
        .alloc_scratch_k => "alloc_scratch_k",
        .const_buf => "const_buf",
    };
}

/// True for the slots whose forwarder drains the stream, so the recorded host
/// interval really contains device execution.  Everything else records
/// submission cost only (see the note on `stat_ns_slot`).
fn slotSyncs(s: Slot) bool {
    return switch (s) {
        // MAY drain -- but in the steady state it does not.  `downloadOut` (a
        // blocking `cudaMemcpy` D2H) only runs when the caller handed us a HOST
        // output buffer; when `dBufPtr(out)` resolves, the kernel writes that
        // device region directly and no copy back happens.  The engine passes
        // device buffers on the whole decode path, so this flag marks a
        // capability, not an observed wait: measured with per-call probes, not
        // one of the ~24k matvecs in a 48-token decode of the 27B model blocked
        // on the host, and the slot's cost is the launch (~0.03 ms/call).  Only
        // the first-touch constant uploads (one per layer weight, ~353 of them)
        // wait, and those land in the prefill chunk, not in decode.
        .matvec, .matvec_batchk, .dequant_row => true,
        // Reads an index / floats back to the host.
        .argmax, .copy_to_host, .sample => true,
        // Host-side bookkeeping, no device work to attribute.
        .load_linear, .const_buf, .alloc_buf, .alloc_scratch, .alloc_scratch_k => false,
        else => false,
    };
}

/// One device copy of a host weight.  `ttype` is null for the `.f32` tensors
/// (which have no quant type); `rows`/`cols` are stored because an f32 slice
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
/// One `dAllocBuf` region: the cudaMalloc base and its length in f32 elements.
/// `isDevBuf` needs both to recognise a `span()`ed view of our own memory.
const DevRegion = struct { ptr: *anyopaque, len: usize };

/// Upper bound on the number of transformer layers whose `rope64` /
/// `attn_decode` nodes the graph tracks.  The engine's default is 24 and
/// Capacity for the per-token node arrays, **not** a layer count.
///
/// The name is a leftover from the first cut, which assumed one `rope64` node
/// per layer; the engine actually calls `rope64` once per head-bearing tensor
/// (`q` and `k`, and again per head for some variants), so the node count for a
/// decode step is a small multiple of `n_layers`, not `n_layers`.  0.8B needs 48
/// and 27B needs several hundred, which is why the old 64 tripped the capacity
/// check on the first real run and the feature switched itself off -- correct,
/// but useless.
///
/// Each entry costs a few hundred bytes (args + params + ptr arrays), so 1024 is
/// ~800 KiB -- acceptable for a backend that already holds the weights.
///
/// Measured on the 27B (65 layers, 24 q-heads, 4 kv-heads, interval 4): a decode
/// token captures **448 `rope64` + 16 `attn_decode` + 32 kv-append** nodes.  The
/// earlier claim here ("128 rope64, room to spare") was off by 3.5x and made the
/// 256 capacity silently reject the graph on every 27B run.  Exceeding the
/// capacity is not a correctness problem: the graph path switches itself off and
/// falls back to direct launches, so a too-small value is invisible in the exit
/// code and only shows up as decode throughput.  0.8B needs 48.
const graph_max_layers: usize = 1024;

/// One flag per op class the captured decode step can contain.  See
/// `CudaBackend.g_op_seen` for why the gating exists at all.
const GhOpFlags = struct {
    matvec: bool = false,
    rmsnorm: bool = false,
    rmsnorm_rows: bool = false,
    add_resid: bool = false,
    scale_vec: bool = false,
    silu_in_place: bool = false,
    mul_silu: bool = false,
    mul_sigmoid: bool = false,
    swiglu: bool = false,
    softmax: bool = false,
    zero_buf: bool = false,
    gdn_step: bool = false,
    gdn_step_rows: bool = false,
    gdn_gamma: bool = false,
    conv_step: bool = false,
    attn_decode: bool = false,
    rope64: bool = false,
    rope64_rows: bool = false,
    kv_append: bool = false,
};

/// Node budget for one captured decode step.  Used for the per-function arrays
/// (a 0.8B decode step is ~800 nodes) and for the ordering buffers
/// (`g_order_nodes`), which are the reason it is this large.
const capture_nodes_max = 4096;

/// See the `g_state` field comment in `CudaBackend` for the transitions.
const GraphState = enum { off, warm, rec, live };

/// How many argument slots `argHoldsU32`/`argHoldsInt` will inspect.  The
/// widest launcher in this backend (`gdn_step_rows`) takes 19 arguments, so 24
/// is comfortably above every real signature while keeping a linear scan of an
/// argument array trivially cheap (it runs once per kernel node, once per
/// process, at capture time).
/// Largest kernel signature this backend launches.  `attn_decode`'s device
/// kernel takes 12 arguments and `gdn_step_rows`' takes 17; 24 leaves headroom
/// and costs `24 * 16` bytes per tracked node, i.e. nothing.
const kernel_max_args: usize = 24;

/// One argument slot of a tracked kernel: `width` bytes of value storage, so a
/// f32 (4) and a 64-bit pointer (8) both fit without a variant type.  Alignment
/// is 8 because the widest thing stored is a pointer.
const ArgSlot = struct { bytes: [8]u8 align(8) = @splat(0) };

pub const CudaBackend = struct {
    alloc: std.mem.Allocator,
    /// Metadata/geometry authority (see the file comment).  Never used for
    /// arithmetic -- the one exception is the opt-in `$QWEN35_MVCHECK`
    /// cross-check in `mvCheck`, which is diagnostics and cannot affect output.
    cpu: cpu_mod.CpuBackend,
    /// The GGUF being served.  Needed only by `vLoadLinear` to recover the
    /// rows/cols of a `.f32` weight: `LinearW.f32` is a flat slice with no
    /// geometry, and the device launches need `cols`.  Every quant row -- f16
    /// and bf16 included -- carries its own rows/cols, so none of them uses this.
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
    /// ttype of the batchK call currently bracketed by `ev_bk_a`/`ev_bk_b`, so
    /// `batchKTimingEnd` can attribute the device interval to a type.  Only
    /// meaningful while `gpuTimingOn()`; written by `batchKTimingBegin`.
    bk_ttype: i32 = -1,
    /// `$QWEN35_PROF`: per-type device ns / call count from the batchK events.
    /// `stat_ns_batchk_gpu` is the same intervals summed over all types; these
    /// say WHERE they went, which is what picks the next optimisation target.
    stat_ns_bk_by_type: [64]u64 = @splat(0),
    stat_n_bk_by_type: [64]u32 = @splat(0),
    ev_bk_b: ?*anyopaque = null,
    /// Around the whole prefill window: `profReset` -> first `matvecBatchK` of
    /// the decode phase.  The wall clock version of the same window is
    /// `stat_ns_prefill_wall`, so the difference is the *host* gap that no
    /// kernel covers.
    ev_pre_a: ?*anyopaque = null,
    ev_pre_b: ?*anyopaque = null,
    /// Around one `argmax` launch + its sync D2H copy.  `ev_arg_a` is recorded
    /// on the device *before* the argmax kernel, `ev_arg_b` *after* it, so
    /// `ev_arg_b - ev_arg_a` is the tail of the queue as the device saw it.
    /// Measured only under `$QWEN35_PROF`; see `dArgmax_body`.
    ev_arg_a: ?*anyopaque = null,
    ev_arg_b: ?*anyopaque = null,
    /// Marks the device position at the END of the previous token (recorded right
    /// after that token's argmax copy).  `tok_end - prev_tok_end` is one whole
    /// decode token as the device saw it: every kernel of that token.  The gap
    /// between that and the host's wall clock for the same token is the host's
    /// own share (submission + the blocking copy).
    ev_tok_end: ?*anyopaque = null,
    ev_prev_tok_end: ?*anyopaque = null,
    tok_end_valid: bool = false,
    stat_ns_token_dev: u64 = 0,
    stat_n_token_dev: u64 = 0,
    /// Device time of the argmax kernel's own launch window, summed (`$QWEN35_PROF`).
    stat_ns_argmax_gpu: u64 = 0,
    /// Host wall clock spent inside `dArgmax_body`, summed (`$QWEN35_PROF`).  The
    /// difference against `stat_ns_argmax_gpu` is the host's share of the drain:
    /// driver submission + the blocking copy, rather than device execution.
    stat_ns_argmax_host: u64 = 0,
    stat_n_argmax_timed: u64 = 0,
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

    // ---- per-slot host wall clock ($QWEN35_PROF only) ----------------------
    //
    // `t_mv_ms` / `t_bk_ms` / `t_dq_ms` only cover the three slots that had a
    // counter.  Everything else the engine calls through `backend.Exec`
    // (rmsnorm, rope64, attention, the GDN state updates, the elementwise
    // kernels, lm_head, argmax) was invisible, so the prefill budget could not
    // be closed: wall minus (matvec + batchk + dequant) left ~2/3 of the
    // window unaccounted for.
    //
    // Every forwarder below now wraps its body in `slotEnter`/`slotExit`, which
    // are `QWEN35_PROF`-gated (a bool test when off, so the plain path is
    // unchanged).  These are HOST wall-clock intervals around the forwarder
    // call, which is exactly what the prefill window is measured in -- so the
    // slots and `stat_ns_prefill_wall` are directly comparable.
    //
    // Caveat, important when reading the table: the CUDA forwarders launch on
    // the default stream and do NOT drain it.  For those slots the interval
    // measures *submission* (python-side dispatch cost plus the memcpys and any
    // implicit sync), not device execution; the GPU time they queued shows up
    // instead inside whichever later slot drains the stream.  Slots whose
    // forwarder does synchronize -- `argmax`, `copyToHost`, and the batchK /
    // matvec paths that call `downloadOut` -- DO include their GPU time.  The
    // per-slot line marks the draining slots; see `slotSyncs`.
    stat_ns_slot: [slot_count]u64 = @splat(0),
    stat_n_slot: [slot_count]u32 = @splat(0),
    /// Set while a timed slot is open, so `slotExit` can be paired.  Not
    /// re-entrant by design: the forwarders do not call each other except via
    /// explicit `CpuBackend.vtable.*` calls, which bypass the timers.
    slot_depth: u32 = 0,
    /// Wall clock spent inside timed slots, all slots summed (`QWEN35_PROF`
    /// only).  `pre_wall_ms - slot_total_ms` is what the engine itself spent
    /// outside any `Exec` call.
    stat_ns_slot_total: u64 = 0,

    // ---- per-slot DEVICE time ($QWEN35_PROF_GPU) ---------------------------
    //
    // WHY THIS EXISTS: the `QWEN35_PROF` slot table is HOST wall clock, and for
    // the 28 non-draining slots that is submission time only -- their queued GPU
    // work surfaces in whichever later slot drains the stream.  So the table can
    // say "matvec = 9 ms/token" without saying whether that is 187 kernel
    // submissions (the launch-bound regime, fixable with a graph) or 187 kernels
    // that really ran 9 ms on the device (the bandwidth-bound regime, fixable
    // only with a faster kernel).  A/B on `QWEN35_CUDA_GRAPH` settled the
    // question numerically (graph=1 survives CUDA_LAUNCH_BLOCKING at 122 tok/s
    // while graph=0 drops to 78), so the device number is what is left to
    // measure.
    //
    // HOW: one pair of `cudaEventCreateWithFlags(..., EventDisableTiming)` per
    // timed slot, recorded on the slot's launch stream at enter/exit.  Reading
    // an elapsed time REQUIRES the pair to be complete, and each pair is
    // recorded on the stream the launch goes to, so `b - a` is the device time
    // between the two records: the slot's own kernels plus any device work
    // already queued ahead of them.  That is exactly why the read happens once
    // per token, at the argmax drain, when the stream is empty -- a pair read in
    // the middle of a token would measure the queue, not the slot.
    //
    // WHY A POOL AND NOT PER-TOKEN CREATION: `cudaEventCreateWithFlags` in the
    // hot path costs as much as the launch being measured.  The pool is created
    // once, and a pair is reused as soon as its interval has been read.
    // HOW: one pair of `cudaEventCreateWithFlags(..., EventDisableTiming)` per
    // timed slot, recorded on the slot's launch stream at enter/exit.  Reading
    // an elapsed time REQUIRES the pair to be complete, and each pair is
    // recorded on the stream the launch goes to, so `b - a` is the device time
    // between the two records: the slot's own kernels plus any device work
    // already queued ahead of them.  That is exactly why the read happens once
    // per token, at the argmax drain, when the stream is empty -- a pair read in
    // the middle of a token would measure the queue, not the slot.
    //
    // WHY A POOL AND NOT PER-TOKEN CREATION: `cudaEventCreateWithFlags` in the
    // hot path costs as much as the launch being measured.  The pool is created
    // once, and a pair is reused as soon as its interval has been read.
    ev_pair_a: [gpu_ev_capacity]?*anyopaque = @splat(null),
    ev_pair_b: [gpu_ev_capacity]?*anyopaque = @splat(null),
    gpu_ev_ready: bool = false,
    /// Pool usage for the token currently in flight.
    gpu_ev_used: usize = 0,
    /// Slot id per pair, so a read can be attributed.
    gpu_ev_slot: [gpu_ev_capacity]u8 = @splat(0),
    /// Device ns per slot, summed over all tokens read.
    stat_gpu_ns_slot: [slot_count]u64 = @splat(0),
    stat_gpu_n_slot: [slot_count]u32 = @splat(0),
    /// Pairs that did not fit in the pool for this token (counted, not dropped).
    stat_gpu_overflow: usize = 0,
    /// TEMP(ablation): bitmap of `Slot`s whose launch is SKIPPED.  Driven by
    /// `QWEN35_ABLATE=<comma-separated slot names>`.  Model output is garbage
    /// when this is on; the ONLY thing read is the tok/s drop, which is how a
    /// slot's real device contribution is attributed when nsys is unavailable.
    ablate_bits: u32 = 0,
    /// `EventElapsedTime` failures while reading the pool.
    stat_gpu_ev_err: usize = 0,
    /// Whole-token device ns from the pool (sum of all pairs), for the
    /// reconciliation against `stat_ns_token_dev` (the token-end event pair).
    stat_gpu_ns_token_sum: u64 = 0,
    stat_gpu_n_token: usize = 0,
    /// Set when `$QWEN35_PROF_GPU` is in the environment.
    gpu_ev_on: bool = false,
    /// Pair index opened by the current `slotEnter`, consumed by its `slotExit`.
    gpu_ev_open: ?usize = null,

    /// `copyRange` broken down by *length* and *route* (`QWEN35_PROF` only).
    ///
    /// The slot total says `copy_range` is expensive but not *which* call site
    /// pays: the engine issues it from nine places at four very different
    /// widths (a 512-B per-head de-interleave, a `kv_dim` KV-cache write, a
    /// one-shot embedding H2D, and the per-row GDN copies).  Bucketing by
    /// element count makes each family its own row, because in this model the
    /// widths do not overlap: 128 = one head, 512/1024 = q/gate or the GDN
    /// `kv_mem` slices, 2048+ = kv_dim (27B: 8*256) and embedding rows.
    ///
    /// The route split matters separately from the width: a d2d move is a
    /// kernel launch, a host->device move is a `cudaMemcpy`, and they have
    /// different per-call costs for the same byte count.  Counting them
    /// together is what made the P2b-era table unreadable.
    stat_copy_bins: [copy_bin_count]CopyBin = @splat(.{}),

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
    /// Second activation buffer holding the `block_q8_1_mmq` reshape, written by
    /// the same quantization pass.  Only allocated when a dp4a type is launched
    /// (see `q35_mmq_dp4a_wants`); the mma path never touches it.  Both buffers
    /// are needed because one prefill chunk mixes types -- e.g. iq3_xxs on dp4a
    /// next to iq3_s on mma -- and they share `d_xq` otherwise.
    d_xqm: ?*anyopaque = null,
    d_xqm_bytes: usize = 0,
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

    // ---- device sampler (src/kernels/cuda/sample.cu) ------------------------
    //
    // `d_sample_ctl` is a RESIDENT 64-byte control block holding every
    // per-token sampler input (temperature, top_k, top_p, PRNG state) plus the
    // result.  It exists so the sampler can live INSIDE the captured decode
    // graph: a captured node keeps the parameters it was recorded with, so all
    // of them have to arrive through memory instead of through `SetParams`.
    // `d_sample_scratch` is the kernel's working set (histogram, per-block
    // extrema, the f64 mass), sized once by `q35_sample_scratch_ints()`.
    //
    // Both are allocated lazily by `ensureSampleBufs` BEFORE any capture opens,
    // because `cudaMalloc` is illegal inside a capture interval.
    d_sample_ctl: ?*anyopaque = null,
    d_sample_scratch: ?*anyopaque = null,
    /// Host row for the `$QWEN35_CUDA_SAMPLE_CPU` A/B, grown on demand.
    h_sample_ab: ?[]f32 = null,
    /// Host scratch for the same A/B: the CPU sampler sorts through it.
    h_sample_ab_scratch: ?[]f32 = null,
    /// `$QWEN35_CUDA_SAMPLE_CPU`: route `sample` to the CPU reference on the
    /// device's own logits, so the two samplers can be compared without the
    /// matvec divergence in the way.  Debug only; default off.
    sample_cpu_ab: bool = false,
    /// Pinned HOST mirror of the first `SampleCtl.host_ints` ints of
    /// `d_sample_ctl`.  Pinned so the per-token H2D is an async copy of a small
    /// constant address rather than a pageable one (pageable H2D inside a
    /// capture is rejected outright).
    h_sample_ctl: ?*anyopaque = null,
    /// The token `sample` produced on the current step, collected by
    /// `sampledToken` after `stepEnd` has done the download.  The split exists
    /// because the download cannot be captured but the sampler must be.
    sample_pending: bool = false,
    sample_token: u32 = 0,
    /// The sampler whose PRNG state the pending download must advance.  A bare
    /// pointer on purpose: the engine owns the object and outlives every step
    /// (it is allocated with the engine's arena), so the backend must not copy
    /// or free it -- only write `state` through it, exactly as `dSample` would.
    sample_sampler: ?*sampler.Sampler = null,

    // ---- decode CUDA graph ($QWEN35_CUDA_GRAPH) -----------------------------
    //
    // WHY THIS EXISTS: a decode token is ~700 kernel launches of which almost
    // none is dominated by its own execution time -- the 0.8B runs at ~8.2 ms
    // per token on this box, and a measured 240-launch probe moved 6.57 -> 6.12
    // ms/token purely by replacing the host's per-launch submission with one
    // graph replay.  The saving is a fixed per-kernel submit cost, so a token
    // with more launches gains proportionally more.
    //
    // WHY A GRAPH AND NOT A STREAM: every launch in this file passes `null`,
    // i.e. the LEGACY default stream, and CUDA forbids capturing it.  So the
    // graph is captured on a private non-default stream (`g_stream`) and every
    // `Launch.*` call inside the captured forward takes that stream instead of
    // `null` -- see `gSlice`-style stream substitution at the call sites.
    // Atomicity of a token is preserved: legacy-default-stream work is
    // implicitly ordered against all blocking streams, but during capture there
    // IS no other stream doing anything, and the replay is the only work
    // queued, so ordering between the graph's own nodes is what the captured
    // dependencies already encode.
    //
    // WHAT IS NOT IN THE GRAPH: `dArgmax_body`'s blocking D2H.  A synchronous
    // copy cannot be captured, and more fundamentally its result has to reach
    // the host before the next token's `pos`/`slot` are even known -- that is a
    // real structural dependency, not an implementation detail.  So the graph
    // spans exactly one `forwardLogits`, and argmax runs after it, directly.
    //
    // STATE MACHINE (`g_state`):
    //   off     -- feature disabled or already gave up; direct launches only.
    //   warm    -- feature on but the capture has not started.
    //   rec     -- inside `cudaStreamBeginCapture`..`EndCapture`; every launch
    //              must be redirectable to `g_stream`.
    //   live    -- `g_exec` holds a replayable graph.
    g_state: GraphState = .off,
    /// Non-default stream the graph was captured on and is replayed on.
    /// Created lazily (a CUDA build that never enables the feature pays
    /// nothing), never the legacy default stream -- capture of that one is
    /// rejected by the runtime with `cudaErrorStreamCaptureUnsupported`.
    g_stream: ?*anyopaque = null,
    /// Event used purely for ORDERING the graph against the legacy default
    /// stream (where the embedding lookup runs).  The graph's stream is
    /// non-blocking, so the two have no implicit ordering: without this the
    /// replay could start before the embedding wrote its input.
    g_ev: ?*anyopaque = null,
    /// Second ordering event, kept separate from `g_ev` on purpose: `g_ev`
    /// records the DEFAULT stream (to make the graph stream wait for the
    /// embedding) and `g_ev_out` records the GRAPH stream (to make the host wait
    /// for the replay).  Sharing one event between the two directions makes each
    /// `Record` clobber the other direction's pending wait inside one token;
    /// two events remove that coupling for the cost of one extra handle.
    g_ev_out: ?*anyopaque = null,
    /// Two device int32 {pos, bit pattern of base}: the per-token inputs of
    /// `k_rope64_dpos`.  Allocated on first use and then just written, so the
    /// 448 rope nodes of a 27B decode graph carry a constant pointer and never
    /// need `GraphExecKernelNodeSetParams`.  See `dRope64_body`.
    g_rope_dyn: ?*anyopaque = null,
    /// Host alias of `g_rope_dyn` (the mapped pinned page).  Writing here is how
    /// a token's `pos`/`base` reaches `k_rope64_dpos`.
    /// What is currently published in `g_rope_dyn`, so the writer kernel runs
    /// once per token instead of once per head.
    /// What `d_xq` currently holds (see `actQValid`).  `n == 0` means empty.
    q_hit: usize = 0,
    q_miss: usize = 0,
    /// QWEN35_GRTIME: pure graph-replay device time (no host wait inside).
    gr_ev: [2]?*anyopaque = .{ null, null },
    gr_ms: f64 = 0,
    gr_n: usize = 0,
    /// QWEN35_MVTIME: per-matvec device time accumulator (events, not wall).
    mv_ev: [2]?*anyopaque = .{ null, null },
    mv_ns: u64 = 0,
    mv_on: bool = false,
    act_q: struct {
        src: ?[*]const f32 = null,
        n: usize = 0,
        cols: usize = 0,
        fam: cpu_mod.ActFamily = .q8_0,
    } = .{},
    g_rope_dyn_pos: u32 = 0xFFFF_FFFF,
    g_rope_dyn_base: f32 = 0.0,
    /// Device int holding the decode `slot` (the KV prefix length minus one).
    /// `attn_decode` reads it instead of taking `slot` as a launch argument, so
    /// the captured node's argument block AND its `shared_mem_bytes` are
    /// constants and the per-token/per-layer `GraphExecKernelNodeSetParams` --
    /// which cost 14.3 ms of the 27B's 33.8 ms decode -- disappears entirely.
    /// Allocated once, written once per token by `set_int1`.
    g_attn_dyn: ?*anyopaque = null,
    /// PV partials for the split attention path: `heads * split * hd` floats.
    d_att_part: ?*anyopaque = null,
    d_att_part_bytes: usize = 0,
    /// Per-head probability scratch for the split attention path.  The engine
    /// hands every head the SAME `slot + 1` floats (`self.attn`), because the
    /// reference kernel runs one block per head and each block re-derives the
    /// same scores -- but the split path runs the softmax as its OWN kernel, one
    /// block per head, so writing all heads into one shared row would race.
    /// Sized `heads * Q35_ATTN_MAX_ROWS` so the stride matches the kernel.
    d_att_prob: ?*anyopaque = null,
    d_att_prob_bytes: usize = 0,
    /// Rows-per-head reduction factor for the split attention path
    /// (`$QWEN35_CUDA_ATTN_SPLIT`, 0 = off -> coalesced kernel).
    attn_split: usize = 0,
    /// The captured graph, kept only between EndCapture and Instantiate.
    g_graph: ?*anyopaque = null,
    /// Replayable instantiation.  This is the only thing used per token.
    g_exec: ?*anyopaque = null,
    /// Kernel nodes to retarget every token, in the order they were launched.
    /// Both are variables of the *decode step*, not of the model:
    ///   - `rope64`'s `pos` advances once per token (one launch per layer);
    ///   - `attn_decode`'s `slot` advances once per token (one launch per full
    ///     attention layer).
    /// Nothing else in a decode token changes its arguments -- `conv_step`
    /// shifts its window IN the kernel (fixed pointers), the GDN state kernels
    /// read fixed pointers, and matvec's operands are all fixed regions of the
    /// KV cache / weights.  That was checked against every `Launch.*` call site
    /// in this file, not assumed.
    g_rope_nodes: [graph_max_layers]?*anyopaque = @splat(null),
    g_rope_n: usize = 0,
    g_attn_nodes: [graph_max_layers]?*anyopaque = @splat(null),
    g_attn_n: usize = 0,
    /// Argument blocks handed to `cudaGraphExecKernelNodeSetParams`.  Each is
    /// the *value* of one changed argument (`pos`, `slot`), so these are four
    /// bytes each and outlive the call only in the sense that the runtime
    /// copies them -- but they must stay addressable between the `SetParams`
    /// and it returning, which a plain field guarantees.
    g_rope_args: [graph_max_layers]c_uint = @splat(0),
    /// `rope64`'s `base` (theta), the OTHER per-token argument of that kernel.
    /// It is not a constant of the model: the engine re-derives it each step
    /// (context-length dependent RoPE scaling), so a snapshot would rotate
    /// every replayed token with the capture token's theta.
    g_rope_base: [graph_max_layers]f32 = @splat(0),
    g_attn_args: [graph_max_layers]c_int = @splat(0),
    /// Per-node parameter definitions captured at capture time (grid/block/func
    /// plus a pointer to the argument array).  Rebuilt rather than mutated
    /// because `cudaGraphKernelNodeGetParams` may return either a
    /// `kernelParams` array or an `extra` blob and only the former is safe to
    /// hand back to `SetParams`.
    g_rope_params: [graph_max_layers]cudart.KernelNodeParams = @splat(undefined),
    g_attn_params: [graph_max_layers]cudart.KernelNodeParams = @splat(undefined),
    /// Owned copies of every tracked node's argument VALUES, one
    /// `[kernel_max_args]ArgSlot` per node.  `cudaGraphKernelNodeGetParams`
    /// hands back a `kernelParams` array that belongs to the runtime and that we
    /// must not write through; instead the values are copied here at capture
    /// time and the `kernelParams` pointer handed to `SetParams` points into
    /// this array.  `pos`/`slot` occupy one slot each and that slot's pointer is
    /// retargeted at `g_rope_args[i]`/`g_attn_args[i]`, so a later token only has
    /// to store a new value in that field.
    g_rope_argv: [graph_max_layers][kernel_max_args]ArgSlot = @splat(@splat(.{})),
    g_attn_argv: [graph_max_layers][kernel_max_args]ArgSlot = @splat(@splat(.{})),
    /// The pointers handed to `SetParams` as `kernelParams`: one per tracked
    /// node, each pointing at that node's `g_*_argv[i]` row above.
    g_rope_kptr: [graph_max_layers][kernel_max_args]?*anyopaque = @splat(@splat(null)),
    g_attn_kptr: [graph_max_layers][kernel_max_args]?*anyopaque = @splat(@splat(null)),
    /// How many `rope64` / `attn_decode` launches the engine issued while the
    /// capture was open.  Reconciled against the number of matched nodes before
    /// the graph is trusted: a shortfall means a launch produced no node we can
    /// retarget, so replay would feed it a stale `pos`/`slot` -- wrong numbers,
    /// no error.  See `graphBuild`.
    g_expect_rope: usize = 0,
    g_expect_attn: usize = 0,
    /// `pos`/`slot` the capture recorded, used to prove the node identification
    /// actually found the right nodes before trusting the graph (see
    /// `graphSelfCheck`).
    g_rec_pos: u32 = 0,
    g_rec_slot: usize = 0,
    /// The rope64 `base`/theta the graph was CAPTURED with.  It is a model
    /// constant (`rope_freq_base`), so the captured value stays right for every
    /// replay -- recorded rather than assumed so a future RoPE regime change is
    /// a visible mismatch instead of silent garbage.
    g_base_captured: f32 = 0,
    /// KV-cache writes captured from `copyRange`.  Their DESTINATION pointer is
    /// `cache.{k,v} + slot * kv_dim`, i.e. it advances by exactly one row per
    /// token, so a plain snapshot would append every later token's k/v to the
    /// capture token's slot -- each token then attends to a cache that never
    /// grew, which is wrong numbers with no error at all.  `g_copy_base[i]` is
    /// the address the capture used and `g_copy_stride[i]` the per-token byte
    /// step, so replay can recompute it.
    /// Destination of each KV append the capture issued, in order, plus its
    /// byte stride per token (`kv_dim * sizeof(f32)`).  `g_kv_expect` is the
    /// count the engine declared; it has to equal the number of nodes found, or
    /// the graph is dropped (see `graphBuild`).
    g_kv_dst_captured: [graph_max_layers]?*anyopaque = @splat(null),
    g_kv_stride: [graph_max_layers]usize = @splat(0),
    g_expect_kv: usize = 0,
    /// `g_order_nodes[0..g_order_n]` is the captured decode step's nodes in
    /// EXECUTION order, recovered by `graphCaptureOrder` -- see the note there
    /// for why the list `cudaGraphGetNodes` returns cannot be used as-is.
    g_order_nodes: [capture_nodes_max]?*anyopaque = undefined,
    g_order_n: usize = 0,

    /// Append counter of the layer cluster currently being walked by
    /// `graphBuild`: 0 before the cluster's first append, 1 after it.  See
    /// `kvSlotFromCluster`.
    g_cluster_n: usize = 0,
    g_cluster_appends: usize = 0,

    /// For every KV append captured, the slot that the OTHER append of the same
    /// layer group belongs to: during a capture the engine issues a group's two
    /// appends consecutively, so the second one of a cluster is the partner of
    /// the first (see `kvSlotFromCluster`).
    g_kv_partner: [graph_max_layers]usize = @splat(0),
    g_kv_nodes: [graph_max_layers]?*anyopaque = @splat(null),
    g_kv_n: usize = 0,
    g_kv_params: [graph_max_layers]cudart.KernelNodeParams = @splat(undefined),
    g_kv_argv: [graph_max_layers][kernel_max_args]ArgSlot = @splat(@splat(.{})),
    g_kv_kptr: [graph_max_layers][kernel_max_args]?*anyopaque = @splat(@splat(null)),
    /// Per-node cell holding the destination POINTER for `SetParams`.  A token
    /// advances it by `g_kv_stride[i]` bytes, so this is the one argument that
    /// `copyArgs` must leave pointing at our storage rather than copying.
    g_kv_dst_cells: [graph_max_layers]?*anyopaque = @splat(null),
    /// True while a capture is being assembled for the token whose argument
    /// values are (`g_rec_pos`, `g_rec_slot`); argmax's blocking copy and any
    /// host fallback clear it so a doomed capture is abandoned instead of
    /// instantiated.
    g_capturing: bool = false,
    /// True between `stepBegin` opening a capture and `stepEnd` closing it.
    /// Distinct from `g_capturing`, which a mid-step abandonment clears: this
    /// one records *ownership* of the open capture so `stepEnd` knows it has to
    /// close something (or clean up after an abandonment) rather than being
    /// called on an uncaptured step.
    g_capturing_step: bool = false,
    /// WHICH ops the captured graph actually contains.  Set during capture by
    /// `gOpGate`, consumed during replay to suppress the host-side launch.
    ///
    /// This is not an optimisation -- it is what makes replay correct.  A
    /// capture does NOT stop the engine: `forwardLogits` keeps calling every
    /// op's body, and a body that launches unconditionally emits a second copy
    /// of the same work onto the default stream while the graph replays the
    /// first copy on `g_stream`.  The two then interleave on the same buffers
    /// and the token's result is neither copy: measured 2026-11, 187 host
    /// matvec launches per replayed token, and `gdn.k` already holding the
    /// graph's output when the host path read it.  Recursive state (GDN) is
    /// advanced TWICE per token that way, which is why the first replayed token
    /// was already numerically wrong.
    ///
    /// Recording the flags from an actual capture (rather than guessing a list)
    /// is deliberate: an op that never entered the graph must keep running on
    /// the host, and one that did must not.
    g_op_seen: GhOpFlags = .{},
    /// Counters, printed once when the graph first goes live and again at
    /// deinit, so a report can state whether replay actually happened.
    stat_g_replay: usize = 0,
    stat_g_abandon: usize = 0,
    /// Set when `$QWEN35_CUDA_GRAPH` said "off" or the first capture failed;
    /// makes the decision one-way so a per-token retry cannot thrash.
    g_failed: bool = false,
    // No gamma/beta staging buffer any more: `gdnGamma` writes both INTO
    // backend regions (`gamma_scratch`, and the b half of `ab` in place) and
    // `gdnStepRows` consumes them straight from those device pointers.  The
    // scratch that used to live here existed only to re-upload host stack
    // arrays, which cost 2 * heads floats of H2D per layer per token plus a
    // matching D2H to produce them.  Those two regions must never be treated
    // as read-only weights either: their contents change every layer every
    // token, so an upload-once pointer-keyed copy would serve stale data.
    /// Every region `dAllocBuf` handed to the engine (activations, KV cache,
    /// GDN state).  The `Buf` values themselves are the engine's; this list is
    /// only so `freeScratch` / `deinit` can release them, since a `Buf` carries
    /// no ownership flag.
    /// Allocations from `dAllocBuf`, as (base, element count).  The count is not
    /// decoration: `isDevBuf` has to answer for Bufs the engine built with
    /// `span()`, whose pointer is base + offset and therefore never equals the
    /// base.  Comparing pointers alone made every strided device view look like
    /// host memory, which silently sent device-to-device copies through the CPU
    /// host bounce (correct answers, ~2.5 s of needless PCIe on a 355-token
    /// prefill).
    dev_bufs: std.ArrayList(DevRegion) = .empty,
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
        self.ablateInit();

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

    /// One-shot *complete* profile report: `CUDA-M [...]` + `CUDA-M  types:`
    /// + `CUDA-M  time:` + the `PROF-SLOT` and `PROF-COPY` tables.
    ///
    /// The reason this entry exists instead of relying on the existing
    /// `reportStats` call sites: those fire at internal trigger points
    /// (`% 500` matvecs, the first matvec, the closing of the prefill window)
    /// and in `deinit`, but the engine is arena-allocated so `deinit` never
    /// runs on the CLI path.  A long decode that stops before the next `% 500`
    /// boundary -- or any BF16/f32 run, whose `matvec` branch never calls
    /// `reportStats` at all -- therefore ended with a single header line and no
    /// tables.  This is called once, by `Engine.profBackendReport`, right after
    /// `generate` returns (the CLI gates it on `$QWEN35_PROF`).
    ///
    /// Unconditional by design: the *caller* owns the debug switch, so the
    /// semantics here are "print the whole thing now" and nothing else.  Purely
    /// diagnostic -- it reads counters and prints, touching no device state.
    pub fn profReport(self: *CudaBackend) void {
        self.reportStats("final");
    }

    pub fn deinit(self: *CudaBackend) void {
        self.reportStats("deinit");
        // The graph's stream is a non-default stream and every object it owns
        // (exec, graph, stream) must die before the context is released below.
        self.graphDestroy();
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
            for (self.dev_bufs.items) |r| _ = cu.lib.Free(r.ptr);
            self.dev_bufs.clearRetainingCapacity();
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

    // ==== decode CUDA graph ==================================================
    //
    // Read `g_state`'s comment for the design; this section is the mechanics.
    //
    // THREE INVARIANTS the code below exists to uphold, each one a silent
    // wrong-numbers bug if broken:
    //
    //  1. NOTHING inside the graph may allocate.  `cudaMalloc` during capture is
    //     legal in relax mode but produces a graph whose replay reuses the same
    //     address -- correct only if the allocation never moves.  `ensureBuf`
    //     *does* move (it frees and reallocates when the size grows), so a
    //     matvec that grew `d_out` after capture would leave every subsequent
    //     node writing through a dangling pointer.  `graphReserve` therefore
    //     forces every scratch buffer this backend can grow up to its
    //     prefill-sized maximum BEFORE the capture opens, and
    //     `graphed-ensureBuf` asserts rather than growing during `rec`.
    //
    //  2. NOTHING inside the graph may be synchronous.  `cudaMemcpy` (the
    //     blocking flavour, used by `dCopyToHost`/`dCopyFromHost` and by the
    //     `rope64` host path) invalidates a capture outright.  That is why
    //     capture forces the device rope path and why `dArgmax_body` is outside
    //     the graph's span by construction.
    //
    //  3. The graph must contain ALL of one `forwardLogits` and nothing else.
    //     Node identification below is by argument value, not by position,
    //     because `cudaGraphGetNodes`'s order is unspecified.

    /// Whether `$QWEN35_CUDA_GRAPH` allows the feature.  Unset means ON: the
    /// measurement this was written from shows a monotone win and the fallback
    /// is automatic, so requiring an opt-in would just hide it.  `=0` (or any
    /// of false/no/off) turns it fully off -- that is the A/B switch and the
    /// escape hatch, and with it set nothing below ever runs.
    fn graphWanted() bool {
        const S = struct {
            var cached: ?bool = null;
        };
        if (S.cached) |v| return v;
        var on = true;
        if (cudart.getenv("QWEN35_CUDA_GRAPH")) |raw| {
            const v = mem.trim(u8, raw, " \t\r\n");
            if (v.len != 0) {
                on = !(mem.eql(u8, v, "0") or mem.eql(u8, v, "false") or
                    mem.eql(u8, v, "no") or mem.eql(u8, v, "off"));
                std.debug.print("cuda backend: decode CUDA graph = {s} ($QWEN35_CUDA_GRAPH)\n", .{
                    if (on) "on" else "off",
                });
            }
        }
        S.cached = on;
        return on;
    }

    /// The stream every kernel launch in the captured forward must use.
    ///
    /// `null` (the legacy default stream) everywhere EXCEPT while a capture is
    /// open, when it must be the private capture stream.  Centralising it here
    /// rather than at ~28 call sites means there is exactly one place that can
    /// forget, and the `g_capturing`/`g_state` pair makes forgetting a loud
    /// capture error (`cudaErrorStreamCaptureUnsupported`) instead of a graph
    /// that is missing nodes.
    fn launchStream(self: *CudaBackend) ?*anyopaque {
        return switch (self.g_state) {
            .rec => self.g_stream,
            else => null,
        };
    }

    /// True when a launch happening right now is being captured, i.e. when its
    /// arguments must be recorded for the per-token retarget.
    fn capturing(self: *const CudaBackend) bool {
        return self.g_state == .rec;
    }

    /// Suppress a host-side launch when the captured graph already contains it.
    ///
    /// Returns false while a capture is open (after recording that this op made
    /// it into the graph) and false while no graph is live; returns true only
    /// for a REPLAYED step, whose work the graph has already done on `g_stream`.
    /// Without this the host path re-runs the whole step on the default stream
    /// next to the graph -- see `g_op_seen`.
    fn gOpGate(self: *CudaBackend, seen: *bool) bool {
        switch (self.g_state) {
            .rec => {
                seen.* = true;
                return false;
            },
            .live => return seen.*,
            else => return false,
        }
    }

    /// Called from every place that would otherwise make a capture invalid:
    /// a blocking copy, a host fallback, a launch error.  Abandons the capture
    /// for this token so the caller falls back to direct launches.  `why` is
    /// printed once (not per token) -- a per-token flood would bury the real
    /// message and cost measurable time in the very path being measured.
    fn graphAbandon(self: *CudaBackend, why: []const u8) void {
        if (self.g_state != .rec) return;
        self.stat_g_abandon += 1;
        if (self.stat_g_abandon == 1) {
            std.debug.print(
                "cuda backend: decode graph abandoned this token ({s}); falling back to direct launches\n",
                .{why},
            );
        }
        // ORDER MATTERS: the capture has to be CLOSED first, because until
        // `cudaStreamEndCapture` returns the stream is still "capturing" and any
        // subsequent launch on it (argmax's, the direct-path kernels, the next
        // token's) fails with `cudaErrorStreamCaptureUnsupported` -- "the
        // operation cannot be performed in the present state".  Only then is the
        // half-recorded graph destroyed (it is never instantiated or replayed).
        if (self.cu) |*cu| {
            var g: ?*anyopaque = null;
            _ = cu.lib.StreamEndCapture(self.g_stream, &g);
            // THE ABANDONED CAPTURE STILL HAS TO RUN.  Capture mode records
            // launches instead of executing them, so this token has had no
            // kernel at all: no KV row appended, no GDN state advanced, and its
            // logits are the previous step's.  Destroying the graph (the old
            // behaviour) silently dropped a whole step -- measured on a model
            // where identification then failed, the token stream came back as
            // `0 220` followed by the correct sequence shifted by two, i.e. the
            // capture step's tokens were emitted from a stale logits buffer.
            // Instantiate-and-launch-once replays exactly what was recorded,
            // which is the direct path's own kernel list, so the step is
            // performed and the numbers stay right even though no graph is kept.
            if (g) |gg| {
                var ab_exec: ?*anyopaque = null;
                if (cu.lib.GraphInstantiate(&ab_exec, gg, 0) == cudart.cudaSuccess and ab_exec != null) {
                    const lrc = cu.lib.GraphLaunch(ab_exec.?, self.g_stream);
                    if (lrc != cudart.cudaSuccess) {
                        self.ewLaunchFail("decode graph: abandon replay", lrc);
                    }
                    if (self.g_stream) |st| {
                        _ = cu.lib.StreamSynchronize(st);
                    }
                    _ = cu.lib.GraphExecDestroy(ab_exec.?);
                }
                _ = cu.lib.GraphDestroy(gg);
            }
            self.g_graph = null;
        }
        self.g_state = .warm;
        self.g_rope_n = 0;
        self.g_attn_n = 0;
        self.g_failed = true;
    }

    /// Bring every growable scratch buffer up to the size a *prefill* call would
    /// need, so no `ensureBuf` can fire inside the capture (invariant 1).
    ///
    /// WHY THE PREFILL SIZE: the decode step's needs are a strict subset of a
    /// prefill chunk's, and prefill has already run `allocScratchK` by the time
    /// generation starts.  Re-reserving with the same numbers here is not
    /// redundant -- it makes the requirement local and checkable instead of
    /// "the engine happens to have called allocScratchK first", and it also
    /// covers `d_idx`, which nothing pre-grows today because 4 bytes never
    /// needed it.
    fn graphReserve(self: *CudaBackend, max_cols: usize) !void {
        if (!self.gpuReady()) return error.BackendUnavailable;
        const cu = &self.cu.?;
        // q8_k is the wider of the two activation formats; reserving it for
        // both means `ensureBuf` at matvec time can never grow `d_xq` either.
        try ensureBuf(&cu.lib, &self.d_act, &self.d_act_bytes, max_cols * @sizeOf(f32));
        try ensureBuf(&cu.lib, &self.d_xq, &self.d_xq_bytes, actBytes(.q8_k, max_cols));
        try ensureBuf(&cu.lib, &self.d_out, &self.d_out_bytes, max_cols * @sizeOf(f32));
        try ensureBuf(&cu.lib, &self.d_deq, &self.d_deq_bytes, max_cols * @sizeOf(f32));
        try ensureBuf(&cu.lib, &self.d_idx, &self.d_idx_bytes, @sizeOf(c_int));
        // The reservation is what makes capture safe, so it is also what arms the
        // feature: before this point `g_state` stays `.off` and `stepBegin` does
        // nothing, which is right because a capture that early would have to
        // allocate scratch inside the graph.
        if (self.g_state == .off and graphWanted()) self.g_state = .warm;
    }

    /// Open the capture.  `pos` and `slot` are this token's values, recorded so
    /// node identification can verify itself afterwards.
    ///
    /// Returns false when the capture could not be opened (no stream, runtime
    /// refused); the caller then runs the token with direct launches.  A false
    /// return is NOT permanent -- a later token may try again -- except when
    /// `g_failed` was set after a real failure.
    fn graphBegin(self: *CudaBackend, pos: u32, slot: usize) bool {
        const cu = &(self.cu orelse return false);
        if (self.g_failed) return false;
        if (self.g_state == .live) {
            // A live graph is being replayed; a second capture would have to
            // destroy it, which is never what we want mid-generation.
            return false;
        }
        // Invariant 2: the host rope path is a pair of blocking copies and
        // cannot be captured.  Rather than silently switching the user's
        // chosen rope path, decline to capture and keep the direct path.
        if (!ropeOnDevice()) return false;
        if (self.g_stream == null) {
            var s: ?*anyopaque = null;
            // Non-blocking so the stream does not implicitly synchronise with
            // the legacy default stream: an implicit sync would make every
            // replay wait for the argmax copy of the *previous* token, which is
            // exactly the dependency we want but at the wrong granularity (and
            // it would also make the capture refuse).
            const rc = cu.lib.StreamCreateWithFlags(&s, cudart.StreamNonBlocking);
            if (rc != cudart.cudaSuccess or s == null) {
                self.ewLaunchFail("decode graph: stream create", rc);
                self.g_failed = true;
                return false;
            }
            self.g_stream = s;
        }
        if (self.g_ev == null) {
            // Disable timing: this event is a pure ordering token, and the
            // timing flag makes every record cost a timestamp write.
            var ev: ?*anyopaque = null;
            const erc = cu.lib.EventCreate(&ev, cudart.EventDisableTiming);
            if (erc != cudart.cudaSuccess or ev == null) {
                self.ewLaunchFail("decode graph: event create", erc);
                self.g_failed = true;
                return false;
            }
            self.g_ev = ev;
        }
        if (self.g_ev_out == null) {
            var ev: ?*anyopaque = null;
            const erc = cu.lib.EventCreate(&ev, cudart.EventDisableTiming);
            if (erc != cudart.cudaSuccess or ev == null) {
                self.ewLaunchFail("decode graph: ordering event create", erc);
                self.g_failed = true;
                return false;
            }
            self.g_ev_out = ev;
        }
        // The capture only records what runs on `g_stream`, and the embedding
        // that precedes it is on the default stream; ordering first means the
        // capture starts from a state where that work is at least queued in a
        // stream the capture stream waits on.
        self.graphSyncFromDefault();
        const rc = cu.lib.StreamBeginCapture(self.g_stream, cudart.StreamCaptureModeThreadLocal);
        if (rc != cudart.cudaSuccess) {
            self.ewLaunchFail("decode graph: begin capture", rc);
            self.g_failed = true;
            return false;
        }
        self.g_state = .rec;
        self.g_capturing = true;
        self.g_rec_pos = pos;
        self.g_rec_slot = slot;
        self.g_rope_n = 0;
        self.g_attn_n = 0;
        self.g_expect_rope = 0;
        self.g_expect_attn = 0;
        self.g_expect_kv = 0;
        return true;
    }

    /// Record that a `rope64` launch just issued into the capture, storing the
    /// argument block `SetParams` will rewrite later.
    ///
    /// `pos` is written into `g_rope_args[i]` and the params' `kernelParams`
    /// array points at it, so the *same* `KernelNodeParams` instance serves
    /// every subsequent token -- only the value behind the pointer changes.
    fn graphNoteRope(self: *CudaBackend, pos: u32, base: f32) void {
        if (self.g_rope_n >= graph_max_layers) {
            self.graphAbandon("too many rope64 nodes");
            return;
        }
        const i = self.g_rope_n;
        self.g_rope_args[i] = @bitCast(pos);
        self.g_rope_base[i] = base;
        if (self.g_rope_n == 0) self.g_base_captured = base;
        self.g_rope_n += 1;
        self.g_expect_rope += 1;
    }

    /// A KV append: `copyRange` to `cache + slot * kv_dim`.  Records the exact
    /// destination address the capture used, so `graphBuild` can single out the
    /// matching node among the many `copy_range` launches of a token (the
    /// tensor copies use the same kernel and the same stub).
    fn graphNoteKvDst(self: *CudaBackend, dst: *anyopaque, stride_bytes: usize) void {
        if (self.g_expect_kv >= graph_max_layers) {
            self.graphAbandon("too many kv append nodes");
            return;
        }
        const i = self.g_expect_kv;
        self.g_kv_dst_captured[i] = dst;
        self.g_kv_stride[i] = stride_bytes;
        // A layer group's two appends are issued back to back, so they arrive
        // here as consecutive slots.  `graphBuild` walks the graph in execution
        // order and needs to know, for the FIRST append of a cluster, which slot
        // the second one owns -- the count is odd for the A/B split's lone
        // append and even for the layer groups, which is what the pairing below
        // keys on (see `kvSlotFromCluster`).
        if (i % 2 == 1) {
            self.g_kv_partner[i - 1] = i;
            self.g_kv_partner[i] = i - 1;
        }
        self.g_expect_kv += 1;
    }

    /// Same for `attn_decode`, whose only per-token variable is `slot`.
    fn graphNoteAttn(self: *CudaBackend, slot: usize) void {
        if (self.g_attn_n >= graph_max_layers) {
            self.graphAbandon("too many attn_decode nodes");
            return;
        }
        const i = self.g_attn_n;
        self.g_attn_args[i] = @intCast(slot);
        self.g_attn_n += 1;
        self.g_expect_attn += 1;
    }

    /// `graphNoteAttn` for a launcher that emits `n` attn-class nodes per call.
    fn graphNoteAttnN(self: *CudaBackend, slot: usize, n: usize) void {
        var i: usize = 0;
        while (i < n) : (i += 1) self.graphNoteAttn(slot);
    }

    fn graphEnd(self: *CudaBackend) bool {
        const cu = &(self.cu orelse return false);
        self.g_capturing = false;
        if (self.g_state != .rec) return false;
        var graph: ?*anyopaque = null;
        const rc = cu.lib.StreamEndCapture(self.g_stream, &graph);
        if (rc != cudart.cudaSuccess or graph == null) {
            self.ewLaunchFail("decode graph: end capture", rc);
            if (graph) |g| _ = cu.lib.GraphDestroy(g);
            self.g_state = .warm;
            self.g_failed = true;
            return false;
        }
        return self.graphBuild(graph.?);
    }

    /// Reconstruct the capture's EXECUTION ORDER for a captured graph.
    ///
    /// `cudaGraphGetNodes` returns the nodes in an order the documentation leaves
    /// unspecified, and the runtime hands a freshly captured graph back in
    /// essentially REVERSE launch order.  `graphBuild` matched the KV-append nodes
    /// against `g_kv_dst_captured[]`, which `graphNoteKvDst` filled in LAUNCH
    /// order, so walking a reversed list compared node 0 with the LAST captured
    /// address and every KV append in the graph ended up on another layer.
    ///
    /// The count reconciliation could not catch it: it only checks HOW MANY nodes
    /// matched (12 of 12), never WHICH.  Diagnosed 2026-11 by stamping each node
    /// with its list index -- the rope nodes come back bottom layer first (their
    /// `pos` arguments read 16, 16, ..., 0), i.e. the list is the reverse of the
    /// order the engine issued.
    ///
    /// `cudaGraphGetEdges` cannot read the chain back: it fails with
    /// `cudaErrorNotPermitted` on a graph produced by stream capture (measured).
    /// So the order is rebuilt from a second capture, with API this project
    /// already binds:
    ///
    ///  * clone the recorded graph (same node objects, all dependencies) and,
    ///    inside a RELAXED capture, make it a child of the stream with
    ///    `cudaStreamUpdateCaptureDependencies(clone, ADD_ONLY)`;
    ///  * then one probe at a time: add a zero-length `copy_range` probe and
    ///    `cudaStreamUpdateCaptureDependencies(probe, ADD_ONLY)`, which makes the
    ///    probe a child of the last node on the stream -- after the first probe
    ///    that is the previous probe, so the probes chain toe to toe and the clone
    ///    stays their common parent.  The rebuilt graph's dependency set is
    ///    unchanged (it stays instantiable) and the probes' order is what the node
    ///    list reports.
    ///
    /// `orderFromProbes` reads each probe's single dependency and lays the live
    /// nodes out in the order they were probed; the recorded list and the probe
    /// list are two sequences over the SAME node objects.  Returns the rebuilt
    /// graph, or null when the order cannot be established.
    fn graphCaptureOrder(self: *CudaBackend, graph: *anyopaque) ?*anyopaque {
        const cu = &(self.cu orelse return null);
        var n: usize = 0;
        if (cu.lib.GraphGetNodes(graph, null, &n) != cudart.cudaSuccess or n < 2 or n > capture_nodes_max) return null;
        var recorded: [capture_nodes_max]?*anyopaque = undefined;
        if (cu.lib.GraphGetNodes(graph, &recorded, &n) != cudart.cudaSuccess) return null;

        var clone: ?*anyopaque = null;
        if (cu.lib.GraphClone(&clone, graph) != cudart.cudaSuccess or clone == null) return null;
        const cl = clone.?;
        defer _ = cu.lib.GraphDestroy(cl);

        var probe_unsigned: u32 = 0;
        const probe_args = [_]?*anyopaque{@ptrCast(&probe_unsigned)};
        const probe_func = (if (cuda_linked) Kernels.q35_func_copy_range() else null) orelse return null;
        var params = std.mem.zeroes(cudart.KernelNodeParams);
        params.func = probe_func;
        params.grid_dim_x = 0; // n == 0: the launchers' documented no-op
        params.grid_dim_y = 1;
        params.grid_dim_z = 1;
        params.block_dim_x = 1;
        params.block_dim_y = 1;
        params.block_dim_z = 1;
        params.shared_mem_bytes = 0;
        params.kernel_params = @constCast(@ptrCast(&probe_args));

        var rebuilt_dummy: ?*anyopaque = null;
        const rc_begin = cu.lib.StreamBeginCapture(self.g_stream, cudart.StreamCaptureModeThreadLocal);
        if (rc_begin != cudart.cudaSuccess) {
            if (rc_begin != cudart.cudaSuccess) _ = cu.lib.StreamEndCapture(self.g_stream, &rebuilt_dummy);
            std.debug.print("GCO begin rc={d}\n", .{rc_begin});
            return null;
        }
        var step_node = [_]?*anyopaque{cl};
        const rc_step = cu.lib.StreamUpdateCaptureDependencies(self.g_stream, @constCast(&step_node), null, 0, cudart.stream_add_only);
        // The clone is ONE node as far as the stream is concerned, so the probes
        // chain after it: with an empty list they would all depend on the clone
        // and their own sequence would be lost.

        var cap_status: c_int = -1;
        var cap_id: usize = 0;
        var cap_graph: ?*anyopaque = null;
        var cap_nodes: ?*anyopaque = null;
        var cap_n: usize = 0;
        const rc_q = cu.lib.StreamGetCaptureInfo(self.g_stream, &cap_status, &cap_id, &cap_graph, &cap_nodes, &cap_n);
        var ntype: c_int = -1;
        const rc_t = if (cap_nodes) |nd| cu.lib.GraphNodeGetType(nd, &ntype) else -1;
        std.debug.print("GCO q rc={d} status={d} nodes={d} firsttype={d} rct={d} stepdeps={d}\n", .{ rc_q, cap_status, cap_n, ntype, rc_t, rc_step });
        var ok = rc_step == cudart.cudaSuccess;
        var emitted: usize = 0;
        while (ok and emitted < n) : (emitted += 1) {
            var probe: ?*anyopaque = null;
            if (cu.lib.GraphAddKernelNode(&probe, cl, null, 0, &params) != cudart.cudaSuccess) {
                ok = false;
                break;
            }
            var dep = [_]?*anyopaque{probe};
            const rc_dep = cu.lib.StreamUpdateCaptureDependencies(self.g_stream, @constCast(&dep), null, if (emitted == 0) 0 else 1, cudart.stream_add_only);
            if (rc_dep != cudart.cudaSuccess) {
                std.debug.print("GCO dep rc={d} at {d}\n", .{ rc_dep, emitted });
                ok = false;
                break;
            }
        }
        var rebuilt: ?*anyopaque = null;
        const rc_end = cu.lib.StreamEndCapture(self.g_stream, &rebuilt);
        if (!ok or rc_end != cudart.cudaSuccess or rebuilt == null) {
            std.debug.print("GCO end ok={} rc={d} g={} emitted={d} n={d}\n", .{ ok, rc_end, rebuilt != null, emitted, n });
            if (rebuilt) |g| _ = cu.lib.GraphDestroy(g);
            return null;
        }
        const rg = rebuilt.?;
        if (!self.orderFromProbes(cu, rg, recorded[0..n], probe_func)) {
            _ = cu.lib.GraphDestroy(rg);
            return null;
        }
        _ = cu.lib.GraphDestroy(rg);
        return self.buildOrderedGraph();
    }

    /// `graphCaptureOrder`'s second half: pair every probe of the rebuilt graph
    /// with the live node it depends on and lay those nodes out in
    /// `g_order_nodes`, the execution order of the recorded graph.
    fn orderFromProbes(
        self: *CudaBackend,
        cu: *cudart.Open,
        rebuilt: *anyopaque,
        recorded: []const ?*anyopaque,
        probe_func: *anyopaque,
    ) bool {
        const lib = &cu.lib;
        const n = recorded.len;
        var total: usize = 0;
        if (lib.GraphGetNodes(rebuilt, null, &total) != cudart.cudaSuccess) return false;
        if (total > capture_nodes_max) return false;
        var all: [capture_nodes_max]?*anyopaque = undefined;
        if (lib.GraphGetNodes(rebuilt, &all, &total) != cudart.cudaSuccess) return false;

        // Probes are the stub's nodes launched with grid x == 0; every other
        // kernel node is one of the recorded ones.
        var live: [capture_nodes_max]?*anyopaque = undefined;
        var live_n: usize = 0;
        var probes: [capture_nodes_max]?*anyopaque = undefined;
        var probe_n: usize = 0;
        for (all[0..total]) |maybe| {
            const node = maybe orelse return false;
            var ntype: c_int = -1;
            if (lib.GraphNodeGetType(node, &ntype) != cudart.cudaSuccess) return false;
            if (ntype != cudart.GraphNodeTypeKernel) return false;
            var kp: cudart.KernelNodeParams = std.mem.zeroes(cudart.KernelNodeParams);
            if (lib.GraphKernelNodeGetParams(node, &kp) != cudart.cudaSuccess) return false;
            const is_probe = kp.func != null and @intFromPtr(kp.func.?) == @intFromPtr(probe_func) and kp.grid_dim_x == 0;
            if (is_probe) {
                if (probe_n >= probes.len) return false;
                probes[probe_n] = node;
                probe_n += 1;
            } else {
                if (live_n >= live.len) return false;
                live[live_n] = node;
                live_n += 1;
            }
        }
        if (live_n != n or probe_n != n) {
            std.debug.print("cuda backend: decode graph skipped: order probe shape live={d} probe={d} expect {d}\n", .{ live_n, probe_n, n });
            return false;
        }

        var seen: [capture_nodes_max]bool = undefined;
        @memset(seen[0..n], false);
        for (probes[0..probe_n], 0..) |probe, i| {
            var deps: [4]?*anyopaque = undefined;
            var nd: usize = 0;
            if (lib.GraphNodeGetDependencies(probe.?, &deps, &nd) != cudart.cudaSuccess) return false;
            if (nd != 1) {
                std.debug.print("cuda backend: decode graph skipped: order probe {d} has {d} dependencies\n", .{ i, nd });
                return false;
            }
            const dep = deps[0] orelse return false;
            var at: ?usize = null;
            for (live[0..live_n], 0..) |lv, k| {
                if (lv != null and @intFromPtr(lv.?) == @intFromPtr(dep)) {
                    at = k;
                    break;
                }
            }
            const k = at orelse {
                std.debug.print("cuda backend: decode graph skipped: order probe {d} depends on a probe\n", .{i});
                return false;
            };
            if (seen[k]) return false; // a live node claimed twice: not a permutation
            seen[k] = true;
            self.g_order_nodes[i] = live[k];
        }
        // The probe order must be a permutation of the recorded list.
        for (self.g_order_nodes[0..n]) |maybe| {
            const node = maybe orelse return false;
            var found = false;
            for (recorded) |maybe2| {
                if (maybe2 != null and @intFromPtr(maybe2.?) == @intFromPtr(node)) {
                    found = true;
                    break;
                }
            }
            if (!found) return false;
        }
        self.g_order_n = n;
        return true;
    }

    /// Put `g_order_nodes` into a fresh graph as ONE dependency chain: the chain
    /// is what keeps the list the caller enumerates identical to the order the
    /// probes proved (and holds the same node objects, which is what `SetParams`
    /// needs).
    fn buildOrderedGraph(self: *CudaBackend) ?*anyopaque {
        const cu = &(self.cu orelse return null);
        var graph: ?*anyopaque = null;
        if (cu.lib.GraphCreate(&graph, 0) != cudart.cudaSuccess or graph == null) return null;
        const g = graph.?;
        var prev: ?*anyopaque = null;
        var prev_buf: [1]?*anyopaque = undefined;
        for (self.g_order_nodes[0..self.g_order_n]) |maybe| {
            const node = maybe orelse {
                _ = cu.lib.GraphDestroy(g);
                return null;
            };
            var deps_len: usize = 0;
            if (prev) |pv| {
                prev_buf[0] = pv;
                deps_len = 1;
            }
            var added: ?*anyopaque = null;
            if (cu.lib.GraphAddNode(&added, g, @ptrCast(node), &prev_buf, deps_len) != cudart.cudaSuccess or added == null) {
                _ = cu.lib.GraphDestroy(g);
                return null;
            }
            prev = added;
        }
        return g;
    }

    /// Which retarget slot a KV-append node owns, decided from the EXECUTION
    /// order `graphCaptureOrder` recovered rather than from its captured
    /// destination address.
    ///
    /// A layer's kernels all sit between the two appends that surround them, so
    /// in the ordered list the first append after a cluster start belongs to
    /// layer `g_cluster_n` and the second to `g_kv_partner[g_cluster_n]` (the
    /// other full-attention layer of the same group -- at capture time the
    /// engine issues them in layer order, so the partner of layer 0 is layer 1).
    /// The two appends that open the step, before any cluster, are the halves of
    /// the A/B split the engine issues ahead of the layers.
    ///
    /// Matching on the address instead (what this did until 2026-11) cannot work:
    /// `g_kv_dst_captured[i]` is the address cache entry `i` had at capture time,
    /// so it names a row, not an append.
    fn kvSlotFromCluster(self: *CudaBackend) ?usize {
        if (self.g_cluster_appends == 0) {
            self.g_cluster_appends = 1;
            return self.g_cluster_n;
        }
        const layer = self.g_cluster_n;
        self.g_cluster_appends = 0;
        self.g_cluster_n += 1;
        if (self.g_cluster_n >= graph_max_layers) return null;
        return self.g_kv_partner[layer];
    }

    /// Identify the retargetable nodes inside a freshly captured graph, then
    /// instantiate.
    ///
    /// IDENTIFICATION: `cudaGraphGetNodes` returns the nodes in an unspecified
    /// order, so position cannot be trusted.  Instead every kernel node's
    /// arguments are read back with `cudaGraphKernelNodeGetParams` and matched
    /// against the values the capture recorded (`g_rec_pos` for rope64,
    /// `g_rec_slot` for attn_decode).  That is a *value* match on a parameter
    /// only those two launches carry at those magnitudes, and it is verified:
    /// the count has to equal what the engine issued, and if it does not the
    /// graph is thrown away rather than replayed with stale arguments.
    ///
    /// The argument index is found by scanning `kernelParams` for the recorded
    /// value rather than hard-coding an offset: the launcher passes `pos` /
    /// `slot` by value at a position that would be a maintenance trap to
    /// duplicate here, and a wrong offset would silently rewrite (say) `hd`.
    fn graphBuild(self: *CudaBackend, recorded: *anyopaque) bool {
        const cu = &(self.cu orelse return false);
        KernelFns.resolve();
        // ORDERING FIRST.  `cudaGraphGetNodes` lists the nodes of a captured
        // graph in an order the documentation leaves unspecified and the runtime
        // really does hand back reversed, while `graphNoteKvDst` recorded the KV
        // destinations in LAUNCH order.  Matching KV nodes by position therefore
        // paired every node with another layer's address; the count check could
        // not see it (12 matched, 12 expected) and the replay moved every append
        // to the wrong cache row.  Diagnosed 2026-11 -- see `graphCaptureOrder`.
        // The recorded graph is used AS-IS: every node is identified by its
        // host-stub `func` and its arguments are matched by VALUE, so nothing
        // here depends on the order `cudaGraphGetNodes` happens to return (the
        // documentation leaves it unspecified, and it is not launch order).
        const graph = recorded;
        // `graph` is ours to free unless it becomes the live graph.
        var graph_kept = false;
        defer {
            if (!graph_kept) _ = cu.lib.GraphDestroy(graph);
        }
        var n: usize = 0;
        if (cu.lib.GraphGetNodes(graph, null, &n) != cudart.cudaSuccess or n == 0) {
            self.graphAbandon("graph has no nodes");
            return false;
        }
        // Stack-allocated, capped: a decode token is ~700 nodes and the arrays
        // only live for the duration of this function.
        const max_nodes = capture_nodes_max;
        if (n > max_nodes) {
            self.graphAbandon("graph node count over the identification budget");
            return false;
        }
        var nodes: [max_nodes]?*anyopaque = undefined;
        if (cu.lib.GraphGetNodes(graph, &nodes, &n) != cudart.cudaSuccess) {
            self.graphAbandon("graph node enumeration failed");
            return false;
        }

        var rope_found: usize = 0;
        var rope_dpos_found: usize = 0;
        var attn_found: usize = 0;
        var attn_dyn_found: usize = 0;
        var kv_found: usize = 0;
        var kernel_nodes: usize = 0;
        self.g_cluster_n = 0;
        self.g_cluster_appends = 0;
        for (nodes[0..n]) |maybe_node| {
            const node = maybe_node orelse continue;
            var ntype: c_int = -1;
            if (cu.lib.GraphNodeGetType(node, &ntype) != cudart.cudaSuccess) continue;
            if (ntype != cudart.GraphNodeTypeKernel) continue;
            kernel_nodes += 1;
            var p: cudart.KernelNodeParams = std.mem.zeroes(cudart.KernelNodeParams);
            if (cu.lib.GraphKernelNodeGetParams(node, &p) != cudart.cudaSuccess) continue;
            if (classifyNode(&p)) |cls| {
                switch (cls) {
                    .rope => {
                        if (rope_found >= graph_max_layers) continue;
                        // ONLY `pos` is a per-token variable.  `base` used to be
                        // registered here too, on the theory that it needs a cell
                        // to live in -- but a variable argument is COPIED FROM
                        // nowhere (its cell is filled by the backend, never by
                        // `copyArgs`), so `base` ended up as whatever the cell
                        // held: zero on the capture step, and `powf(0, exp)` is
                        // inf, i.e. every rope output became NaN and the replayed
                        // text was garbage from the very first token.  `base` is a
                        // constant for the whole run, so it belongs in the
                        // ordinary argument snapshot.
                        if (!copyArgs(.rope, &p, &self.g_rope_argv[rope_found], &self.g_rope_kptr[rope_found], &.{
                            .{ .at = rope_pos_arg, .cell = &self.g_rope_args[rope_found] },
                        })) continue;
                        self.g_rope_params[rope_found] = p;
                        self.g_rope_nodes[rope_found] = node;
                        rope_found += 1;
                    },
                    .rope_dpos => {
                        // Exact and free: `d_dyn` is a constant device pointer,
                        // so nothing about this node ever changes.  Counted (so
                        // the banner reports how much of the graph is frozen)
                        // but never recorded for retargeting.
                        rope_dpos_found += 1;
                    },
                    .attn_score => {
                        attn_dyn_found += 1;
                    },
                    .attn_reduce => {
                        attn_dyn_found += 1;
                    },
                    .attn_dyn => {
                        // Constant arguments AND constant shared size: nothing to
                        // retarget, ever.  Counted so the node census matches
                        // `g_expect_attn` (which the dyn path still increments).
                        attn_dyn_found += 1;
                    },
                    .attn => {
                        if (attn_found >= graph_max_layers) continue;
                        if (!copyArgs(.attn, &p, &self.g_attn_argv[attn_found], &self.g_attn_kptr[attn_found], &.{
                            .{ .at = attn_slot_arg, .cell = &self.g_attn_args[attn_found] },
                        })) continue;
                        self.g_attn_params[attn_found] = p;
                        self.g_attn_nodes[attn_found] = node;
                        attn_found += 1;
                    },
                    .kv => {
                        // Every `copy_range` node matches the stub, so the stub
                        // cannot say WHICH append this is.  The captured
                        // DESTINATION does: `graphNoteKvDst` recorded the exact
                        // address (`cache + slot * kv_dim`) of every append in
                        // launch order, so this node's argument #0 is an exact
                        // key.  Matching by address instead of by enumeration
                        // order is what makes this correct: the runtime does not
                        // promise `cudaGraphGetNodes` order, and on this CUDA
                        // build it hands back the reverse, which paired every
                        // append with another layer's cache row.  Measured: the
                        // inference from order disagreed with the recorded
                        // address on the very first append, and the resulting
                        // cross-layer writes corrupted the cache from the 4th
                        // replayed token on.
                        const at = self.kvSlotOf(&p) orelse continue;
                        if (self.g_kv_nodes[at] != null) continue;
                        if (!copyKvArgs(&p, &self.g_kv_argv[at], &self.g_kv_kptr[at], &self.g_kv_dst_cells[at])) continue;
                        self.g_kv_params[at] = p;
                        self.g_kv_nodes[at] = node;
                        kv_found += 1;
                    },
                }
            }
        }

        // The self-check that decides whether identification may be trusted.
        // `rope64` issues once per layer; `attn_decode` once per full-attention
        // layer, which is <= the layer count.  Zero attn nodes is legal (an
        // all-GDN model); a count over the layer budget is not.
        self.g_rope_n = rope_found;
        self.g_attn_n = attn_found;
        self.g_kv_n = kv_found;
        if (rope_found == 0 and attn_found == 0 and attn_dyn_found == 0) {
            self.graphAbandon("no rope64/attn_decode argument matched; identification untrustworthy");
            _ = cu.lib.GraphDestroy(graph);
            return false;
        }
        if (rope_found > graph_max_layers or attn_found > graph_max_layers) {
            self.graphAbandon("more retargetable nodes than layers");
            _ = cu.lib.GraphDestroy(graph);
            return false;
        }

        // RECONCILIATION, and the only real proof that identification worked:
        // the number of nodes whose stub matched must equal the number of times
        // the engine actually called those two launchers during the capture.
        //
        // Why not "count every kernel node and compare to every launch": a
        // launcher can issue more than one device kernel (`attn_decode` issues
        // the scores/softmax/PV kernel AND the gate kernel), so a plain node
        // count is not comparable to a launch count -- it was 808 vs 784 on a
        // 12-layer model, i.e. exactly 2 per full-attention layer.
        //
        // Why this comparison is the right one: a decode token's only
        // per-token-varying arguments are `pos` (one `rope64` per layer) and
        // `slot` (one `attn_decode` per full-attention layer).  If every one of
        // those launches produced exactly one matched node, the graph's
        // retargeting covers every argument that moves.  Any shortfall means a
        // node went unmatched -- a stub address that changed, a node the runtime
        // encoded differently -- and replaying it would use a stale `pos`/`slot`,
        // which is wrong numbers with no error.  So: abstain.
        if (kv_found != self.g_expect_kv) {
            std.debug.print(
                "cuda backend: decode graph skipped: matched {d}/{d} KV append nodes\n",
                .{ kv_found, self.g_expect_kv },
            );
            self.graphAbandon("KV append node count mismatch");
            return false;
        }
        if (rope_found != self.g_rope_n or attn_found != self.g_attn_n or
            rope_found + rope_dpos_found != self.g_expect_rope or
            attn_found + attn_dyn_found != self.g_expect_attn)
        {
            std.debug.print(
                "cuda backend: decode graph skipped: matched {d}/{d} rope64 and {d}/{d} attn_decode nodes\n",
                .{ rope_found, self.g_expect_rope, attn_found, self.g_expect_attn },
            );
            self.graphAbandon("retargetable node count mismatch");
            return false;
        }
        if (kernel_nodes == 0) {
            self.graphAbandon("graph contains no kernel nodes");
            return false;
        }

        var new_exec: ?*anyopaque = null;
        const rc = cu.lib.GraphInstantiate(&new_exec, graph, 0);
        if (rc != cudart.cudaSuccess or new_exec == null) {
            self.ewLaunchFail("decode graph: instantiate", rc);
            self.g_state = .warm;
            self.g_failed = true;
            return false;
        }
        graph_kept = true;
        // THE GRAPH IS KEPT ALIVE ON PURPOSE.  `cudaGraphExecKernelNodeSetParams`
        // takes the node handle from the *graph*, not from the exec, so
        // destroying the graph here would invalidate every `g_rope_nodes` /
        // `g_attn_nodes` entry and the next token's retarget would fail with
        // `invalid argument` (measured: exactly that).  Both objects are
        // released together in `graphDestroy`.
        self.g_graph = graph;
        self.g_exec = new_exec;
        self.g_state = .live;
        // THE CAPTURE STEP NEVER RAN.  `cudaStreamBeginCapture` puts the stream
        // in capture mode, where launches are *recorded* instead of executed --
        // so the token whose arguments are in the graph has had no kernel run
        // for it at all: no KV row appended, no GDN state advanced, and its
        // logits are the prefill's leftovers.  Measured: the first token looked
        // fine (argmax happened to agree) and every replayed token after it was
        // garbage, because the graph kept rewriting the capture slot and the
        // cache permanently missed one row.  Launching the graph once, with the
        // capture step's own arguments, is what makes that step happen.
        if (!self.graphReplayInner(self.g_rec_pos, self.g_rec_slot, false)) return false;
        // Upload once so the first replay does not pay the loading cost inside
        // the token being measured.
        _ = cu.lib.GraphUpload(self.g_exec, self.g_stream);
        std.debug.print(
            "cuda backend: decode graph live ({d} kernels, {d} rope64 (frozen: pos from device) + {d} rope64-retargeted + {d} attn_decode + {d} kv-append nodes)\n",
            .{ kernel_nodes, rope_dpos_found, rope_found, attn_found + attn_dyn_found, kv_found },
        );
        return true;
    }

    /// Argument index of the per-token variable inside each tracked kernel's
    /// *device* parameter list.  These are facts about the kernels, and they are
    /// stated here once rather than rediscovered by scanning:
    ///
    ///   `k_rope64(d_x, heads, hd, pos, base)`                     -> pos is #3
    ///   `k_rope64(d_x, heads, hd, pos, base)`          -> pos is #3, base is #4
    ///   `q35_attn_decode_kernel(d_q, d_gate, d_kcache, d_vcache,
    ///                           d_att, d_gout, slot, heads, ...)`  -> slot is #6
    ///   `k_copy_range(dst, src, n)`                       -> dst is #0
    ///
    /// All three were read off the launchers in src/kernels/cuda/{rope,attn,
    /// elementwise}.cu, and all three are checked at capture time: `copyArgs`
    /// refuses a node whose grid does not match the shape those launchers
    /// produce, so a kernel gaining an argument shifts the index into a loud
    /// abstention rather than silently rewriting the wrong value.
    ///
    /// NOTE the KV variable is a POINTER, not a scalar, and it is the one
    /// argument that moves per token because the destination is
    /// `cache.{k,v} + slot * kv_dim`.  Its cell is therefore `?*anyopaque`
    /// (`g_kv_dst_cells`), not `c_int` like the other two.
    ///
    /// NOTE ALSO rope64 has TWO per-token arguments, not one.  `base` is the
    /// RoPE theta and the engine re-derives it per step (context-dependent
    /// scaling), so a node retargeted only on `pos` rotates every replayed
    /// token with the capture token's theta.  That was the decode-graph bug:
    /// measured 2026-11, replay produced garbage from the FIRST replayed token
    /// while the captured step -- which executes rather than replays -- was
    /// correct.  Both are listed here and `graphReplay` writes both.
    const rope_pos_arg: usize = 3;
    const rope_base_arg: usize = 4;
    const attn_slot_arg: usize = 6;
    const kv_dst_arg: usize = 0;
    /// Device-kernel arity of each tracked kernel, i.e. how many entries of
    /// `kernelParams` are meaningful.  `cudaKernelNodeParams` carries no count,
    /// so this is what keeps `copyArgs` from reading past the end of the
    /// runtime's array -- the alternative (scan until a null) reads unowned
    /// memory and segfaults, which is how this code first failed.
    const rope_arity: usize = 5;
    const attn_arity: usize = 12;
    const kv_arity: usize = 3;

    /// `cudaHostAllocMapped`.  Spelled out rather than bound from the header:
    /// this file deliberately keeps to a small hand-written cudart surface (see
    /// src/cudart.zig), and the flag value is fixed by the ABI.
    const cudaHostAllocMapped: c_uint = 0x02;

    /// `attn_dyn` is the device-`slot` attention kernel: its argument block AND
    /// its `shared_mem_bytes` are constant for the whole run, so -- like
    /// `rope_dpos` -- it is recognised (so the node census matches) but never
    /// recorded for retargeting.  That is the entire point of the variant:
    /// retargeting the scalar-`slot` node cost one `SetParams` per LAYER per
    /// token, 14.3 ms of a 33.8 ms decode on the 27B.
    const NodeClass = enum { rope, rope_dpos, attn, attn_dyn, attn_score, attn_reduce, kv };

/// `(Q35_ATTN_MAX_ROWS + 2) * sizeof(float)` -- the fixed dynamic shared block of
/// `q35_attn_decode_kernel_dyn` (src/kernels/cuda/attn.cu).  Both sides must
/// agree: the kernel is launched with it, and capture verifies the node carries
/// it (a mismatch means the node is not the one we think it is).
const q35_attn_smem_dynslot: usize = (8192 + 2) * @sizeOf(f32);

/// Widest split-path attention kernel: `q35_attn_score_split_kernel_dyn(d_q,
/// d_kcache, d_att, d_slot, heads, gqa, hd, kv_dim, scale, split)`.
const attn_score_arity: usize = 10;

    /// Decide whether a captured kernel node is one of the two retargetable
    /// launches, by matching its `func` against the host-stub address each
    /// launcher exposes (`q35_func_rope64` / `q35_func_attn_decode`).
    ///
    /// WHY NOT MATCH ON ARGUMENT VALUES: `cudaKernelNodeParams` carries no
    /// argument COUNT -- `kernelParams` is a bare pointer array -- so probing it
    /// for `pos` past a kernel's real arity reads unowned memory and segfaults
    /// (this code did exactly that before the stub comparison replaced it).  The
    /// stub address is exact, needs no probing, and costs one comparison per
    /// node.
    fn classifyNode(p: *const cudart.KernelNodeParams) ?NodeClass {
        const f = p.func orelse return null;
        if (f == KernelFns.rope) return .rope;
        // `k_rope64_dpos` reads {pos, base} from device memory, so its node
        // parameters are constant for the whole run.  Reporting it as `.rope`
        // would put it in the retarget loop for no reason; reporting it as
        // nothing at all would be indistinguishable from an unknown kernel, and
        // `graphCaptureEnd` treats "nodes present but not all recognised" as a
        // reason to abandon the graph.  `.rope_dpos` says exactly what it is:
        // a rope node that needs no per-token work.
        if (f == KernelFns.rope_dpos) return .rope_dpos;
        if (f == KernelFns.attn) return .attn;
        if (f == KernelFns.attn_dyn) return .attn_dyn;
        // Split-path kernels: constant arguments like `.attn_dyn`, but 256 threads
        // and no dynamic shared memory, so they get their own class.
        if (f == KernelFns.attn_score_dyn) return .attn_score;
        if (f == KernelFns.attn_pv_dyn) return .attn_score;
        if (f == KernelFns.attn_smax_dyn) return .attn_score;
        // `q35_attn_reduce_kernel(d_part, d_gate, d_gout, heads, hd, split)` --
        // six arguments, three pointers then three ints.  Its own class because
        // `.attn_dyn`'s argument widths describe a twelve-argument kernel.
        if (f == KernelFns.attn_reduce) return .attn_reduce;
        if (f == KernelFns.kv) return .kv;
        return null;
    }

    /// Cached host-stub addresses, resolved once.  The two calls are pure symbol
    /// lookups with no device work, and `cudaGraphKernelNodeGetParams` reports
    /// `func` as exactly this address (verified against `&k_rope64` under
    /// `-rdc=true`).
    const KernelFns = struct {
        var resolved: bool = false;
        var rope: ?*anyopaque = null;
        var rope_dpos: ?*anyopaque = null;
        var attn: ?*anyopaque = null;
        var attn_dyn: ?*anyopaque = null;
        var attn_score_dyn: ?*anyopaque = null;
        var attn_pv_dyn: ?*anyopaque = null;
        var attn_smax_dyn: ?*anyopaque = null;
        var attn_reduce: ?*anyopaque = null;
        var kv: ?*anyopaque = null;

        fn resolve() void {
            if (resolved) return;
            // Same comptime discriminator as `Launch` (see the long note above
            // it): a CPU-only build has no `q35_func_*` to call, so both stay
            // null and `classifyNode` matches nothing.  That is correct -- a
            // build without CUDA kernels cannot be capturing a CUDA graph.
            if (cuda_linked) {
                rope = Kernels.q35_func_rope64();
                rope_dpos = Kernels.q35_func_rope64_dpos();
                attn = Kernels.q35_func_attn_decode();
                attn_dyn = Kernels.q35_func_attn_decode_dyn();
                attn_score_dyn = Kernels.q35_func_attn_score_split_dyn();
                attn_pv_dyn = Kernels.q35_func_attn_pv_split_dyn();
                attn_smax_dyn = Kernels.q35_func_attn_softmax_dyn();
                attn_reduce = Kernels.q35_func_attn_reduce();
                kv = Kernels.q35_func_copy_range();
            }
            resolved = true;
        }
    };

    /// Snapshot a tracked node's arguments into our own storage and build the
    /// `kernelParams` array to hand to `SetParams` later.
    ///
    /// `variable_at` is the index of the per-token variable; that slot's pointer
    /// is pointed at `variable_cell` (the caller's `g_rope_args[i]` /
    /// `g_attn_args[i]`, or `g_kv_dst_cells[i]` for the KV append -- see
    /// `copyKvArgs`) instead of the copied value, so updating the token only
    /// writes that one field.  `cls` selects the kernel's signature: arity from
    /// `arityOf`, per-argument widths from `argWidth`, and the launch-shape check
    /// above.  IT IS A REQUIRED PARAMETER ON PURPOSE: the kind can never be
    /// inferred from `variable_at` again, because the KV variable is argument #0
    /// and #0 is also what a caller with no variable would pass.
    /// Read a node's first argument as a pointer, without trusting anything
    /// about the kernel beyond the `copy_range(dst, src, n)` signature -- used
    /// only for nodes already matched to `k_copy_range` by stub address.
    fn firstArgPtr(p: *const cudart.KernelNodeParams) ?*anyopaque {
        const kpl: [*]?*anyopaque = @ptrCast(p.kernel_params orelse return null);
        const slot0 = kpl[0] orelse return null;
        const as_ptr: *const ?*anyopaque = @alignCast(@ptrCast(slot0));
        return as_ptr.*;
    }


    /// One per-token argument of a tracked node: which `kernelParams` slot it
    /// lives in, and the cell the launcher/replay writes it through.
    const VarArg = struct { at: usize, cell: *anyopaque };

    fn copyArgs(
        cls: NodeClass,
        p: *const cudart.KernelNodeParams,
        values: *[kernel_max_args]ArgSlot,
        ptrs: *[kernel_max_args]?*anyopaque,
        var_args: []const VarArg,
    ) bool {
        const kpl: [*]?*anyopaque = @ptrCast(p.kernel_params orelse return false);
        const arity: usize = arityOf(cls);
        // Shape check: rope64 launches exactly one block of `kHalf *
        // kHeadsPerBlock` threads for the single head the engine passes;
        // attn_decode launches one block per query head with 128 threads and a
        // dynamic shared block of `(slot + 1 + 2) * 4` bytes; the KV append is a
        // linear `copy_range` (gridFor(n) blocks of kThreads = 256, no dynamic
        // shared memory), so its grid is whatever division the length produces
        // and only "non-degenerate" can be required.  Requiring these means an
        // argument-index drift cannot turn into a wrong launch geometry -- it
        // just declines the node.
        switch (cls) {
            .rope, .rope_dpos => {
                if (p.grid_dim_x != 1 or p.grid_dim_y != 1) return false;
                if (p.shared_mem_bytes != 0) return false;
            },
            .attn_reduce, .attn_score => {
                if (p.grid_dim_x == 0 or p.grid_dim_y != 1) return false;
                if (p.block_dim_x != 256 or p.block_dim_y != 1) return false;
                if (p.shared_mem_bytes != 0) return false;
            },
            .attn, .attn_dyn => {
                if (p.grid_dim_x == 0 or p.grid_dim_y != 1) return false;
                // `.attn_dyn` here is `q35_attn_decode_kernel_dyn` (128 threads,
                // fixed shared block); the split-path kernels are also classified
                // `.attn_dyn` but run 256 threads with NO shared memory, so this
                // branch cannot be the one that validates them -- they are checked
                // by `attnScoreShaped` below.
                if (p.block_dim_x != 128 or p.block_dim_y != 1) return false;
                // `.attn_dyn` additionally has a FIXED shared size; that is what
                // makes it un-retargetable, and a different size means this is
                // not the node we think it is.
                if (cls == .attn_dyn and p.shared_mem_bytes != q35_attn_smem_dynslot) return false;
            },
            .kv => {
                // DELIBERATELY NOT the attn rule: `k_copy_range` is launched with
                // 256 threads, not 128, and with one block per 256 elements --
                // `dst` is a whole cache row (kv_dim >= 128), so grid_x is
                // neither 1 nor `heads`.  Applying either neighbour's rule would
                // reject every genuine KV node and the graph would abstain on
                // every model.  What actually binds for this kind is the
                // argument WIDTHS below (8/8/4, source of truth
                // src/kernels/cuda/elementwise.cu:60).  Which append a node is
                // comes from the execution order, not from here -- see
                // `kvSlotFromCluster`.
                if (p.grid_dim_x == 0 or p.block_dim_x == 0) return false;
                if (p.grid_dim_y != 1 or p.block_dim_y != 1) return false;
                if (p.shared_mem_bytes != 0) return false;
            },
        }
        for (0..kernel_max_args) |i| ptrs[i] = null;
        for (0..arity) |i| {
            const src = kpl[i] orelse return false;
            var is_variable = false;
            for (var_args) |v| {
                if (v.at != i) continue;
                // Not copied: this slot must track the per-token field.  The
                // cell's width is the argument's width (`argWidth`), so the
                // caller's cell must be at least that wide -- a `c_uint` for
                // rope64's `pos`, an `f32` for its `base`, a pointer cell for
                // the KV destination.
                ptrs[i] = v.cell;
                is_variable = true;
                break;
            }
            if (is_variable) continue;
            // Copy at the argument's real width (pointer slots are 8 bytes, the
            // int/float ones 4), taken from the kernel signatures spellt out in
            // `argWidth`.  Copying the wrong width would not corrupt the value --
            // it would only read a few adjacent bytes and write them back
            // unchanged -- but it would also mean the widths here are wrong, so
            // the signature is the single source for them.
            const width: usize = argWidth(cls, i);
            @memcpy(values[i].bytes[0..width], @as([*]const u8, @ptrCast(src))[0..width]);
            ptrs[i] = @ptrCast(&values[i].bytes);
        }
        return true;
    }

    /// Device-kernel arity per tracked kind.  `kernelParams` carries no count,
    /// so this is the single place the three signatures are stated as counts.
    fn arityOf(cls: NodeClass) usize {
        return switch (cls) {
            .rope => rope_arity,
            // k_rope64_dpos(float *d_x, int heads, int hd, const void *d_dyn)
            .rope_dpos => 4,
            .attn => attn_arity,
            // q35_attn_decode_kernel_dyn(... const int *d_slot, int heads,
            //   int gqa, int hd, int kv_dim, float scale) -- same arity.
            .attn_dyn => attn_arity,
            .attn_score => attn_score_arity,
            .attn_reduce => 6,
            .kv => kv_arity,
        };
    }

    /// KV-append entry point.  The per-token variable here is the destination
    /// POINTER, so its storage is a `?*anyopaque` cell (`g_kv_dst_cells[i]`):
    /// `graphReplay` writes the new address into that exact cell and relies on
    /// the node's `kernelParams[0]` still pointing at it.
    ///
    /// WHY A TYPED PARAMETER IN BETWEEN INSTEAD OF A CAST AT THE CALL SITE:
    /// `&cell` for a `?*anyopaque` is a `*?*anyopaque`, whose address is only
    /// the payload's address by luck (`?*anyopaque` is 8 bytes here -- an
    /// optional pointer's tag is squeezed into the alignment, verified), and it
    /// is NOT provably aligned to the pointer-size the kernel will read.  So the
    /// conversion is done ONCE, here, with the alignment the target type
    /// requires; doing it at each call site would let a future call pass the
    /// address of something else that merely happens to be eight bytes wide and
    /// get a silently wrong destination (garbage KV rows, no error).
    fn copyKvArgs(
        p: *const cudart.KernelNodeParams,
        values: *[kernel_max_args]ArgSlot,
        ptrs: *[kernel_max_args]?*anyopaque,
        dst_cell: *?*anyopaque,
    ) bool {
        return copyArgs(.kv, p, values, ptrs, &.{
            .{ .at = kv_dst_arg, .cell = @ptrCast(@alignCast(dst_cell)) },
        });
    }

    /// Slot index of a KV append node, from its captured destination address.
    ///
    /// `graphNoteKvDst` stored the address every append wrote during the capture,
    /// in launch order, so the node's own argument #0 identifies it exactly --
    /// no dependence on the order `cudaGraphGetNodes` returns (which is not
    /// launch order and is not even stable across versions).  Returns null when
    /// no captured append used this address, which means the node is a plain
    /// tensor `copy_range` rather than a cache append.
    fn kvSlotOf(self: *CudaBackend, p: *const cudart.KernelNodeParams) ?usize {
        const want = kvDstOf(p) orelse return null;
        for (self.g_kv_dst_captured[0..self.g_expect_kv], 0..) |maybe, i| {
            const have = maybe orelse continue;
            if (have == want) return i;
        }
        return null;
    }

    /// The destination pointer argument of a `k_copy_range` node.
    fn kvDstOf(p: *const cudart.KernelNodeParams) ?*anyopaque {
        const kpl: [*]?*anyopaque = @ptrCast(p.kernel_params orelse return null);
        const cell = kpl[kv_dst_arg] orelse return null;
        var got: ?*anyopaque = null;
        @memcpy(std.mem.asBytes(&got), @as([*]const u8, @ptrCast(cell))[0..@sizeOf(?*anyopaque)]);
        return got;
    }

    /// Byte width of argument `i` of a tracked kernel: 8 for the pointer
    /// arguments, 4 for the integer/float ones.
    fn argWidth(cls: NodeClass, i: usize) usize {
        return switch (cls) {
            // k_rope64(float *d_x, int heads, int hd, int pos, float base)
            .rope => switch (i) {
                0 => 8,
                else => 4,
            },
            // k_rope64_dpos(float *d_x, int heads, int hd, const void *d_dyn)
            .rope_dpos => switch (i) {
                0, 3 => 8,
                else => 4,
            },
            // q35_attn_decode_kernel(const float *d_q, const float *d_gate,
            //   const float *d_kcache, const float *d_vcache, float *d_att,
            //   const float *d_gout, int slot, int heads, int gqa, int hd,
            //   int kv_dim, float scale)
            .attn => switch (i) {
                0, 1, 2, 3, 4, 5 => 8,
                else => 4,
            },
            // Same signature as `.attn`: the only change is that `slot` is now a
            // device pointer (still an 8-byte argument in slot 6).
            .attn_dyn => switch (i) {
                0, 1, 2, 3, 4, 5 => 8,
                else => 4,
            },
            // q35_attn_reduce_kernel(const float *d_part, const float *d_gate,
            //   float *d_gout, int heads, int hd, int split)
            .attn_reduce => switch (i) {
                0, 1, 2 => 8,
                else => 4,
            },
            // The three split-path kernels: score_split(d_q, d_kcache, d_att,
            // d_slot, heads, gqa, hd, kv_dim, scale, split) = 10 args;
            // softmax_dyn(d_att, d_slot, heads) = 3;
            // pv_split_dyn(d_vcache, d_part, d_prob, d_slot, heads, gqa, hd,
            //   kv_dim, split) = 9.  Widths below cover the widest (10).
            .attn_score => switch (i) {
                0, 1, 2, 3 => 8,           // pointers (score kernel)
                else => 4,
            },
            // k_copy_range(float *dst, const float *src, int n) -- THREE
            // arguments (src/kernels/cuda/elementwise.cu:60).  While the kind was
            // inferred from the variable index, a KV node (whose variable is #0,
            // same as no variable at all) was measured against the 12-argument
            // attention signature and `copyArgs` would read nine slots past the
            // end of the runtime's array.
            .kv => switch (i) {
                0, 1 => 8,
                else => 4,
            },
        };
    }

    /// Order the graph's stream after everything already queued on the legacy
    /// default stream (the embedding lookup that fills the graph's input).
    ///
    /// WHY THIS IS NEEDED AT ALL: `g_stream` is created with
    /// `cudaStreamNonBlocking`, which is mandatory here (capture on the legacy
    /// default stream is illegal), and a non-blocking stream does NOT take part
    /// in the default stream's implicit synchronization.  So without this the
    /// replay may start before the embedding kernel has written `x`, and the
    /// whole token is computed from the PREVIOUS token's activations: no error,
    /// no garbage, just a wrong answer several tokens later.
    ///
    /// Cost is one event record + one wait per token, both host-side, ~1 us.
    fn grMark(self: *CudaBackend, begin: bool) void {
        const cu = &(self.cu orelse return);
        const st = self.g_stream orelse return;
        var i: usize = 0;
        if (!begin) i = 1;
        if (self.gr_ev[i] == null) {
            var e: ?*anyopaque = null;
            if (cu.lib.EventCreate(&e, 0) != cudart.cudaSuccess) return;
            self.gr_ev[i] = e;
        }
        if (begin) {
            _ = cu.lib.EventRecord(self.gr_ev[0], st);
            return;
        }
        _ = cu.lib.EventRecord(self.gr_ev[1], st);
        _ = cu.lib.EventSynchronize(self.gr_ev[1]);
        var ms: f32 = 0;
        if (cu.lib.EventElapsedTime(&ms, self.gr_ev[0], self.gr_ev[1]) == cudart.cudaSuccess) {
            self.gr_ms += ms;
            self.gr_n += 1;
        }
    }

    fn graphSyncFromDefault(self: *CudaBackend) void {
        const cu = &(self.cu orelse return);
        const st = self.g_stream orelse return;
        const ev = self.g_ev_out orelse return;
        if (cu.lib.EventRecord(ev, null) != cudart.cudaSuccess) return;
        _ = cu.lib.StreamWaitEvent(st, ev, 0);
    }

    /// The mirror image, for the END of the step: whatever the graph produced
    /// (the KV row it just appended, the GDN state it just advanced) has to be
    /// visible to the default stream before the next step's embedding and before
    /// `sample`.  The default stream is the LEGACY one, and — the trap this
    /// used to fall into — `cudaStreamWaitEvent(0, ...)` is not merely "the
    /// cheapest correct barrier", it is *rejected*: the legacy stream cannot be
    /// made to wait on an event, the call returns `cudaErrorInvalidResourceHandle`
    /// and the `_ =` below threw that away, so the barrier silently did not
    /// exist.  Everything after this point (the argmax D2H that reads the logits,
    /// the next token's embedding write) then raced the graph.  Measured: with
    /// every node's arguments frozen the replayed text still changed from run to
    /// run, and `CUDA_LAUNCH_BLOCKING=1` made the same run correct.  Blocking the
    /// host on the event is what actually orders the two streams; the overlap it
    /// gives up is the host-side sampling, which cannot start before this point
    /// anyway.
    fn graphSyncToDefault(self: *CudaBackend) void {
        const cu = &(self.cu orelse return);
        const st = self.g_stream orelse return;
        const ev = self.g_ev_out orelse return;
        if (cu.lib.EventRecord(ev, st) != cudart.cudaSuccess) return;
        if (cu.lib.EventSynchronize(ev) != cudart.cudaSuccess) return;
    }

    /// Prove, per launch, the two claims a replay silently depends on.
    ///
    /// A CUDA graph is only a faithful stand-in for the captured decode step if
    /// the per-token arguments are the ONLY thing that changes.  Until this
    /// guard existed the code simply asserted that, and the assertion was wrong
    /// in one direction that matters: `rope64` takes the angle from `pos` *and*
    /// `base`, and only `pos` was retargeted, so the capture token's theta was
    /// replayed forever (see the `rope_pos_arg` note for the measurement).  A
    /// wrong constant here is invisible -- the graph launches, no error, the
    /// text is garbage -- so the values are now checked instead of assumed:
    ///
    ///   - `pos` advances by exactly one per step: the `k`-th replay must see
    ///     `rec_pos + k`.  Anything else means the engine inserted host-side
    ///     work between the graph's launches, so the graph is no longer the
    ///     captured step;
    ///   - `slot` follows the same law (this engine's `slot == kv.len`,
    ///     advanced once per step -- see `dStepBegin`);
    ///   - `base` must still be the capture's value.  It is a model constant
    ///     (`rope_freq_base`), so a difference means the model changed RoPE
    ///     regime mid-run and the captured nodes must be rebuilt.
    ///
    /// On the direct/warm path this only records.  On the live path a violation
    /// tears the graph down, so the numbers stay right and only the speedup is
    /// lost -- never silently wrong text.
    fn graphVarsChanged(self: *CudaBackend, cls: NodeClass, pos_or_slot: u64) bool {
        if (self.g_state != .live) return false;
        const expected: u64 = @as(u64, self.g_rec_pos) + @as(u64, self.stat_g_replay) + 1;
        if (pos_or_slot == expected) return false;
        std.debug.print(
            "cuda backend: decode graph invalidated at replay {d} -- {s} value {d} (expected {d}); falling back to direct launches\n",
            .{ self.stat_g_replay, @tagName(cls), pos_or_slot, expected },
        );
        return true;
    }


    /// `base` is deliberately NOT retargeted here: it is a launch-time
    /// argument of `rope64` that the engine derives from the model
    /// (`rope_freq_base`, `src/engine.zig:48`), i.e. a CONSTANT for the whole
    /// run, so the captured value is the right one for every replay too.
    /// `graphNoteRope` still records it so `graphVarsChanged` can prove that
    /// claim per token instead of assuming it.
    fn graphReplay(self: *CudaBackend, pos: u32, slot: usize) bool {
        return self.graphReplayInner(pos, slot, true);
    }

    /// `advance` false means "this launch re-executes the CAPTURE step itself"
    /// (`graphBuild`'s obligatory re-execution, whose arguments are the capture
    /// step's own): the step counter must not move, because the pos/slot law in
    /// `graphVarsChanged` counts TOKENS replayed, and the capture step is the
    /// token at `g_rec_pos`, not one after it.
    fn graphReplayInner(self: *CudaBackend, pos: u32, slot: usize, advance: bool) bool {
        const cu = &(self.cu orelse return false);
        if (self.g_state != .live) return false;
        const gexec = self.g_exec orelse return false;
        // The pos/slot law, checked HERE rather than inside the ops: since
        // `gOpGate` suppresses every gated launch on a replayed step, the ops
        // that used to prove it (`rope64`, `attn_decode`) no longer run on this
        // path at all, so their check would never fire.  A replayed step must be
        // exactly one token after the capture, and the k-th replay exactly k
        // after it; anything else means host work was interleaved and the
        // captured node list no longer describes this token.
        if (advance and self.graphVarsChanged(.attn, slot)) return false;

        for (self.g_rope_nodes[0..self.g_rope_n], 0..) |maybe_node, i| {
            const node = maybe_node orelse continue;
            // The VALUE first: the node's `kernelParams[rope_pos_arg]` points at
            // this very cell (see `copyArgs`), so writing it is all a new `pos`
            // needs.  `SetParams` is still called because the runtime snapshots
            // the argument block at set time rather than reading it at launch.
            self.g_rope_args[i] = @bitCast(pos);
            var p = self.g_rope_params[i];
                        p.kernel_params = @ptrCast(&self.g_rope_kptr[i]);
            p.extra = null;
            const rc = cu.lib.GraphExecKernelNodeSetParams(gexec, node, &p);
            if (rc != cudart.cudaSuccess) {
                self.ewLaunchFail("decode graph: rope64 setParams", rc);
                return false;
            }
        }
        for (self.g_attn_nodes[0..self.g_attn_n], 0..) |maybe_node, i| {
            const node = maybe_node orelse continue;
            // `slot` grows every token, and with it the number of KV rows this
            // kernel must read (`slot + 1`).  The cache POINTERS do not change --
            // the engine hands over a longer prefix of the same device region --
            // so retargeting the integer is the whole update.
            self.g_attn_args[i] = @intCast(slot);
            var p = self.g_attn_params[i];
            p.kernel_params = @ptrCast(&self.g_attn_kptr[i]);
            p.extra = null;
            // `shared_mem_bytes` is a SECOND per-token input, not just an
            // integer argument: `q35_launch_attn_decode` sizes the dynamic
            // block as `(slot + 1 + 2) * sizeof(float)`, so it grows with the
            // KV prefix the kernel reads.  A snapshot keeps the capture's value
            // forever, which under-sizes the block on every later token -- the
            // softmax phases then read past their share and the logits go
            // quietly wrong (no launch error, no bad node count).  Recompute it
            // here from the same formula the launcher uses.
            p.shared_mem_bytes = @intCast((@as(usize, slot) + 1 + 2) * @sizeOf(f32));
            const rc = cu.lib.GraphExecKernelNodeSetParams(gexec, node, &p);
            if (rc != cudart.cudaSuccess) {
                self.ewLaunchFail("decode graph: attn_decode setParams", rc);
                return false;
            }
        }
        // KV appends: the destination moves one cache row per token, so replay
        // recomputes it from the captured address.  Without this the graph
        // overwrites the capture token's slot forever and every later token
        // attends to a cache that never grew -- correct-looking control flow,
        // garbage text, no error anywhere.
        const slot_delta = slot - self.g_rec_slot;
        for (self.g_kv_nodes[0..self.g_kv_n], 0..) |maybe_node, i| {
            const node = maybe_node orelse continue;
            const dst_at_capture = self.g_kv_dst_captured[i] orelse continue;
            const moved: *anyopaque = @ptrFromInt(@intFromPtr(dst_at_capture) + slot_delta * self.g_kv_stride[i]);
            self.g_kv_dst_cells[i] = moved;
            var p = self.g_kv_params[i];
            p.kernel_params = @ptrCast(&self.g_kv_kptr[i]);
            p.extra = null;
            const rc2 = cu.lib.GraphExecKernelNodeSetParams(gexec, node, &p);
            if (rc2 != cudart.cudaSuccess) {
                self.ewLaunchFail("decode graph: kv append setParams", rc2);
                return false;
            }
        }
        // PUBLISH THE PER-TOKEN SCALARS THE GRAPH READS FROM DEVICE MEMORY.
        //
        // `gOpGate` suppresses every gated op on a replayed step, so the writer
        // kernels that carry these values are NOT re-run by the ops -- they
        // appear only as captured nodes holding the CAPTURE token's value, and a
        // replay would then recompute the capture token forever.  Measured: the
        // first decode token after capture came back as id 0.
        //
        // So the writes go here, ahead of the launch, on the default stream with
        // an event to order them before `g_stream`.  They are 1-thread kernels:
        // two launches per token instead of ~2 node `SetParams` per LAYER.
        // Published on `g_stream` ITSELF, immediately before the graph launch:
        // stream order alone guarantees the write is visible to every node of the
        // graph, so there is no event dance to get wrong.
        if (self.g_attn_dyn) |pdyn| {
            const sslot: [*]c_int = @ptrCast(@alignCast(pdyn));
            const wrc = Launch.set_int1(sslot, @intCast(slot), self.g_stream);
            if (wrc != cudart.cudaSuccess) {
                self.ewLaunchFail("decode graph: attn slot publish", wrc);
                return false;
            }
        }
        const rc = cu.lib.GraphLaunch(gexec, self.g_stream);
        if (rc != cudart.cudaSuccess) {
            self.ewLaunchFail("decode graph: launch", rc);
            return false;
        }
        // ORDER THE REST OF THE STEP AFTER THE GRAPH.  `GraphLaunch` is
        // asynchronous and lands on `g_stream`, but `launchStream()` hands out
        // the DEFAULT stream while the graph is live -- so every kernel the
        // rest of this token issues (the lm_head matvec, the argmax that reads
        // its result) runs concurrently with the graph unless something orders
        // them.  A stream-capture mode that silently drops the barrier is a
        // wrong-number bug with no error: measured, replaying a graph with all
        // of its arguments frozen still gave a different token on every run,
        // and one barrier here made it bit-identical to the direct path.
        //
        // The event wait goes on BOTH sides on purpose: the event orders the
        // legacy default stream, and the `StreamSynchronize` covers the case
        // where the runtime refuses `StreamWaitEvent(NULL, ...)` (it is not a
        // supported wait target).  The direct path pays the same wait inside
        // the argmax `cudaMemcpy`, so this costs the replay nothing.
        const ev = self.g_ev_out orelse return true;
        if (cu.lib.EventRecord(ev, self.g_stream) == cudart.cudaSuccess) {
            _ = cu.lib.StreamWaitEvent(null, ev, 0);
            _ = cu.lib.EventSynchronize(ev);
        }
        _ = cu.lib.StreamSynchronize(self.g_stream);
        if (advance) self.stat_g_replay += 1;
        return true;
    }

    /// Tear the graph down after a *replay-time* failure, keeping the direct
    /// path (and the stream, which later steps still use for their own launches
    /// only if a new capture opens).  One-way: a live graph that fails to replay
    /// must never be retried, because the failure means the node list or the
    /// stream no longer matches what was instantiated, and replaying again would
    /// hand back the same failure at best.
    fn graphDestroyForFallback(self: *CudaBackend) void {
        // The device channel may already hold a value for a pos this run never
        // published through the direct path; drop the cache so the next `rope64`
        // rewrites it unconditionally instead of assuming it is current.
        self.g_rope_dyn_pos = 0xFFFF_FFFF;
        if (self.cu) |*cu| {
            if (self.g_exec) |e| {
                _ = cu.lib.GraphExecDestroy(e);
                self.g_exec = null;
            }
        }
        self.g_state = .warm;
        self.g_failed = true;
        self.g_rope_n = 0;
        self.g_attn_n = 0;
    }

    /// Free the graph state.  Idempotent; called from `deinit`.
    fn graphDestroy(self: *CudaBackend) void {
        if (self.cu) |*cu| {
            if (self.g_exec) |e| {
                _ = cu.lib.GraphExecDestroy(e);
                self.g_exec = null;
            }
            if (self.g_graph) |g| {
                _ = cu.lib.GraphDestroy(g);
                self.g_graph = null;
            }
            if (self.g_ev_out) |ev| {
                _ = cu.lib.EventDestroy(ev);
                self.g_ev_out = null;
            }
            if (self.g_ev) |ev| {
                _ = cu.lib.EventDestroy(ev);
                self.g_ev = null;
            }
            if (self.g_stream) |s| {
                _ = cu.lib.StreamDestroy(s);
                self.g_stream = null;
            }
        }
        self.g_state = .off;
        if (self.stat_g_replay != 0 or self.stat_g_abandon != 0) {
            std.debug.print("cuda: graph replays={d} abandoned-tokens={d}\n", .{
                self.stat_g_replay, self.stat_g_abandon,
            });
        }
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

    /// Launch the f16/bf16 (w16) matvec.  `d_w` is the resident 2-byte-per-element
    /// weights, `d_x` the f32 activation.  Writes straight into `outBuf` when that
    /// is already device memory (the CUDA-graph decode path), otherwise into the
    /// shared staging buffer and downloads.  Returns the launch result code so the
    /// caller reports it; 0 (cudaSuccess) means the work is queued/done.
    fn launchW16(self: *CudaBackend, d_w: ?*const anyopaque, d_x: ?[*]const f32,
                 ttype: gguf.TensorType, rows: usize, cols: usize, tokens: usize,
                 outBuf: backend.Buf) c_int {
        const n = tokens * rows;
        if (dBufPtr(outBuf)) |pd| {
            return Launch.mul_mat_vec_w16(d_w, d_x, pd, @intCast(rows), @intCast(cols),
                                          @intCast(tokens), @intCast(@backingInt(ttype)),
                                          self.launchStream());
        }
        ensureBuf(&self.cu.?.lib, &self.d_out, &self.d_out_bytes, n * @sizeOf(f32)) catch {
            std.debug.print("cuda backend: w16 matvec: device output allocation failed ({d} B)\n", .{n * @sizeOf(f32)});
            // Negative, not a cudaSuccess value, so the caller's error path fires
            // (the same shape `stubNoCudaKernels` uses).
            return -1;
        };
        const rc = Launch.mul_mat_vec_w16(d_w, d_x, @ptrCast(@alignCast(self.d_out)),
                                          @intCast(rows), @intCast(cols), @intCast(tokens),
                                          @intCast(@backingInt(ttype)), self.launchStream());
        // The staging branch must download (same contract as the f32 branch: a
        // download abandons any open graph capture through `dCopyToHost`).
        if (rc == cudart.cudaSuccess) self.downloadOut(outBuf.slice()[0..n]);
        return rc;
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
            kern.quantize_q8_0(d_src, self.d_xq.?, @intCast(nb), self.launchStream())
        else
            kern.quantize_q8_k(d_src, self.d_xq.?, @intCast(nb), self.launchStream());
        if (rc != cudart.cudaSuccess) {
            if (self.capturing()) self.graphAbandon("quantize activation launch failed");
            std.debug.print("cuda backend: quantize({s}) launch failed: {s}\n", .{
                @tagName(ttype), std.mem.span(cu.lib.errStr(rc)),
            });
            return error.LaunchFailed;
        }
        return nb * actBlockBytes(ttype);
    }

    /// Memoisation for `quantizeActFrom`: does `d_xq` already hold the
    /// activation of `src` quantised to `fam` at width `cols`?  Cleared by
    /// `stepBegin` and by every host write into an activation buffer
    /// (`copyFromHost`), the only two ways those bytes can change under us.
    fn actQValid(self: *CudaBackend, src: [*]const f32, fam: cpu_mod.ActFamily, cols: usize) bool {
        const k = self.act_q;
        // NB: no ttype in the key.  The packed layout is a function of the
        // FAMILY alone (`actFamily` collapses every 256-element block type onto
        // Q8_K), so q4_k -> iq4_xs -> iq3_xxs in one phase must hit.  Keying on
        // the ttype too made every one of those transitions a MISS and put the
        // quantize back on the critical path: measured 103 hits / 399 misses per
        // token, which is the 9.6 ms/token the replay bench reproduces exactly
        // (matvecs alone 22.7 ms, +quantize-per-matvec 32.2 ms).
        return k.n != 0 and k.src == src and k.cols == cols and k.fam == fam;
    }

    fn actQMark(self: *CudaBackend, src: [*]const f32, fam: cpu_mod.ActFamily, cols: usize) void {
        self.act_q = .{ .src = src, .n = 1, .cols = cols, .fam = fam };
    }

    fn actQClear(self: *CudaBackend) void {
        self.act_q = .{ .src = null, .n = 0, .cols = 0, .fam = .q8_0 };
    }

    // ==== slot: loadLinear / freeLinear =====================================

    /// Upload every weight the CPU loader accepted, packed or f32.
    ///
    /// The CPU loader stays authoritative for geometry, mmap resolution and type
    /// support; it already reports missing/unknown tensors with the `LoadError` the
    /// engine expects.  This function adds exactly one decision: does the tensor
    /// fit on the device, and if not, say so loudly.
    fn vLoadLinear_body(ctx: *anyopaque, name: []const u8) backend.LoadError!backend.LinearW {
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
        // Geometry: the QuantRow view knows rows/cols for a packed weight.  A
        // `.f32` tensor is a flat slice, so it is registered as one row of `len`
        // columns; the engine only sends f32 weights through `dequantRow` or a
        // `matvec` whose x is exactly that wide.
        const geom = switch (w) {
            .quant => |q| Geometry{ .rows = q.rows, .cols = q.cols() },
            // A `.f32` weight is `LinearW.f32`, a flat f32 slice with no
            // geometry of its own; recover cols/rows from the tensor's GGUF dims
            // (GGML row-major: dims[0] is the row length).  Falls back to a
            // single row so a missing tensor still reports a clear shape error
            // from the matvec assert rather than silently using the wrong cols.
            // f16/bf16 never land here: they are `QuantRow`s with the GGUF
            // 256-element row framing, so `.quant` above already gives the shape.
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
        // Host side: both arms are now zero-copy views into the GGUF mapping
        // (`.quant` and `.f32` alike), so there is nothing host-side to release
        // here -- only the device copy above.
        self.cpu.exec().freeLinear(w);
    }

    // ==== slot: loadVecF32 ==================================================

    /// Load a 1-D f32 weight vector by name and hand back a device-resident
    /// `Buf`.  llama.cpp-style residency: the tensor is uploaded exactly once,
    /// at load time, and the returned `Buf` IS the residency -- its `ptr` is
    /// the device pointer the elementwise kernels (`rmsnorm`, `convStep`,
    /// `gdnGamma`, ...) read.  There is no host copy and no first-use upload:
    /// the old `devConst` path is gone.
    ///
    /// The widening/dequant is the CPU loader's single definition (f16/bf16/
    /// q8_0 -> f32); we upload those bytes and then release the host slice, so
    /// the weight lives only on the device from here on.
    fn vLoadVecF32(ctx: *anyopaque, name: []const u8) backend.LoadError!backend.Buf {
        const self = self_(ctx);
        const host = self.cpu.exec().loadVecF32(name) catch |err| return err;
        errdefer self.cpu.exec().freeVec(host);
        if (host.len == 0) return backend.Buf.empty;
        if (!self.gpuReady()) {
            return error.BackendUnavailable;
        }
        const cu = &self.cu.?;
        const bytes = host.len * @sizeOf(f32);
        var dp: ?*anyopaque = null;
        const rc = cu.lib.Malloc(&dp, bytes);
        if (rc != cudart.cudaSuccess) {
            std.debug.print("cuda backend: 1-D weight '{s}' ({d} B) cudaMalloc failed: {s}\n", .{
                name, bytes, std.mem.span(cu.lib.errStr(rc)),
            });
            return error.OutOfMemory;
        }
        const host_bytes = mem.sliceAsBytes(host.slice());
        const rc2 = cu.lib.Memcpy(dp, host_bytes.ptr, bytes, cudart.MemcpyHostToDevice);
        if (rc2 != cudart.cudaSuccess) {
            _ = cu.lib.Free(dp);
            std.debug.print("cuda backend: 1-D weight '{s}' ({d} B) upload failed: {s}\n", .{
                name, bytes, std.mem.span(cu.lib.errStr(rc2)),
            });
            return error.OutOfMemory;
        }
        // Uploaded: the host slice is dead; the `Buf` we hand back is the device
        // region.  (Success return, so the `errdefer` above does NOT run.)
        self.cpu.exec().freeVec(host);
        const dev: [*]f32 = @ptrCast(@alignCast(dp.?));
        return backend.Buf{ .ptr = dev, .len = host.len };
    }

    fn vFreeVec(ctx: *anyopaque, v: backend.Buf) void {
        const self = self_(ctx);
        if (v.len == 0) return; // `Buf.empty` sentinel: nothing was uploaded.
        if (self.cu) |cu| {
            _ = cu.lib.Free(@ptrCast(v.ptr));
        }
        // Context already torn down: the device memory died with it.
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
    fn vDequantRow_body(ctx: *anyopaque, dstBuf: backend.Buf, w: gguf.QuantRow, row: usize) void {
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
            self.launchStream(),
        );
        if (rc != cudart.cudaSuccess) {
            if (self.capturing()) self.graphAbandon("dequantize_row launch failed");
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
    fn vMatvec_body(ctx: *anyopaque, outBuf: backend.Buf, w: backend.LinearW, xBuf: backend.Buf) void {
        const self = self_(ctx);
        if (self.gOpGate(&self.g_op_seen.matvec)) return;
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
                        self.launchStream(),
                    );
                    if (rc != cudart.cudaSuccess) {
                        if (self.capturing()) self.graphAbandon("mul_mat_vec_f32 launch failed");
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
                        self.launchStream(),
                    );
                    if (rc != cudart.cudaSuccess) {
                        if (self.capturing()) self.graphAbandon("mul_mat_vec_f32 launch failed");
                        std.debug.print("cuda backend: mul_mat_vec_f32 launch failed: {s}\n", .{std.mem.span(cu.lib.errStr(rc))});
                        self.clearDevBuf(outBuf);
                        return;
                    }
                    // This branch downloads the result, which abandons any open
                    // capture via `dCopyToHost`; no separate check needed.
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
                std.debug.assert(xBuf.len >= cols and outBuf.len >= rows);

                // f16/bf16: no activation quantization, no f32 weight copy.  The
                // weights are resident in their native 2-byte form and the kernel
                // widens them element by element against the engine's f32
                // activation (`q35_k_mul_mat_vec_w16` in mmvq.cu).
                if (q.ttype == .f16 or q.ttype == .bf16) {
                    const rcw = self.launchW16(e.dev, d_x, q.ttype, rows, cols, 1, outBuf);
                    if (rcw != cudart.cudaSuccess) {
                        if (self.capturing()) self.graphAbandon("mul_mat_vec_w16 launch failed");
                        std.debug.print("cuda backend: mul_mat_vec({s}) launch failed: {s}\n", .{
                            @tagName(q.ttype), std.mem.span(cu.lib.errStr(rcw)),
                        });
                        self.clearDevBuf(outBuf);
                        return;
                    }
                    self.stat_ns_matvec += durNs(t0, self.io);
                    self.stat_by_type[@backingInt(q.ttype)] += 1;
                    self.stat_n_matvec += 1;
                    return;
                }

                if (cudart.getenv("QWEN35_MVLIST") != null) {
                    std.debug.print("[mvl] {s} rows={d} cols={d}\n", .{ @tagName(q.ttype), rows, cols });
                }
                const xq_bytes = actBytes(q.ttype, cols);
                ensureBuf(&cu.lib, &self.d_xq, &self.d_xq_bytes, xq_bytes) catch {
                    std.debug.print("cuda backend: matvec: device activation allocation failed ({d} B)\n", .{xq_bytes});
                    self.clearDevBuf(outBuf);
                    return;
                };
                // QUANTISE ONCE PER (source, width, activation FAMILY), not once
                // per matvec.  The engine hands every matvec of one phase the
                // SAME activation buffer (measured on the 27B: 497 matvecs per
                // token touch exactly 3 distinct buffers), and the packed form
                // depends only on the family -- `actFamily` collapses all 256-
                // element block types onto one Q8_K layout -- so the same bytes
                // were being re-quantised up to a hundred times per token.
                //
                // Why it matters: the quantize is a launch on the CRITICAL PATH
                // ahead of the matvec, so the GPU cannot start the 60+ us of
                // weight traffic until it drains.  Measured on 17408x5120
                // iq3_xxs: matvec alone 54.9 us (666 GB/s), matvec behind its
                // own quantize 75.2 us (486 GB/s) -- a third of the bandwidth
                // lost to duplicated work.
                //
                // Sound because a matvec only READS `x` (engine.zig writes the
                // projections into other buffers) and `stepBegin` clears this
                // cache every token.  Keying on the family rather than on the
                // ttype is what makes it legal: q4_k / iq4_xs / iq3_xxs all take
                // the same Q8_K activation, so switching between them must NOT
                // invalidate it.
                const fam = cpu_mod.actFamily(q.ttype);
                const _hit = self.actQValid(d_x, fam, cols);
                if (cudart.getenv("QWEN35_QSTAT") != null) {
                    self.q_hit += @intFromBool(_hit);
                    self.q_miss += @intFromBool(!_hit);
                }
                if (!_hit) {
                    _ = self.quantizeActFrom(d_x, q.ttype, cols) catch {
                        self.clearDevBuf(outBuf);
                        return;
                    };
                    self.actQMark(d_x, fam, cols);
                }
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
                        self.launchStream(),
                    );
                    if (rc != cudart.cudaSuccess) {
                        if (self.capturing()) self.graphAbandon("mul_mat_vec launch failed");
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
                        self.launchStream(),
                    );
                    if (rc != cudart.cudaSuccess) {
                        if (self.capturing()) self.graphAbandon("mul_mat_vec launch failed");
                        std.debug.print("cuda backend: mul_mat_vec({s}) launch failed: {s}\n", .{
                            @tagName(q.ttype), std.mem.span(cu.lib.errStr(rc)),
                        });
                        self.clearDevBuf(outBuf);
                        return;
                    }
                    // As above: `downloadOut` runs the blocking D2H that
                    // abandons the capture, so nothing extra is needed here.
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
    fn vMatvecBatchK_body(ctx: *anyopaque, outBuf: backend.Buf, w: backend.LinearW, xKBuf: backend.Buf, k: usize) void {
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
                // `$QWEN35_PROF`: same device-side clock as the quant path.  This
                // branch serves `.f32` weights only; f16/bf16 take the w16 branch
                // below, so a model whose linear weights are 2-byte (BF16/F16) or
                // packed never reaches it.
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
                if (self.gpuTimingOn()) {
                    // Attribute this call's device interval to its type; see
                    // `stat_ns_bk_by_type`.  Set before the event so a failure
                    // in between still names the type that was issued.
                    self.bk_ttype = @intCast(@backingInt(q.ttype));
                    self.batchKTimingBegin();
                }

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
                // f16/bf16: no activation quantization at all.  The K f32 rows
                // are already resident (`base_act`), the kernel widens the
                // 2-byte weights against them, and `d_xq` is never touched --
                // so this branch skips the quantize launch and its buffer
                // entirely.
                if (q.ttype == .f16 or q.ttype == .bf16) {
                    var d_out_w: [*]f32 = undefined;
                    if (out_dev) |p| {
                        d_out_w = p;
                    } else {
                        ensureBuf(&cu.lib, &self.d_out, &self.d_out_bytes, k * rows * @sizeOf(f32)) catch return oomMsg("batchK output");
                        d_out_w = @ptrCast(@alignCast(self.d_out.?));
                    }
                    const rcw = Launch.mul_mat_vec_w16(
                        e.dev,
                        base_act,
                        d_out_w,
                        @intCast(rows),
                        @intCast(cols),
                        @intCast(k),
                        @intCast(@backingInt(q.ttype)),
                        self.launchStream(),
                    );
                    if (rcw != cudart.cudaSuccess) {
                        std.debug.print("cuda backend: mul_mat_vec({s}, K={d}) launch failed: {s}\n", .{
                            @tagName(q.ttype), k, std.mem.span(cu.lib.errStr(rcw)),
                        });
                        self.clearDevBuf(outBuf);
                        return;
                    }
                    if (out_dev == null) self.downloadOut(outBuf.slice()[0 .. k * rows]);
                    self.stat_ns_batchk += durNs(t0, self.io);
                    self.stat_n_batchk += 1;
                    if (self.gpuTimingOn()) self.batchKTimingEnd();
                    return;
                }

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
                // The dp4a kernel reads `block_q8_1_mmq`, not q8_K, and used to
                // re-transpose on every launch.  When it is enabled for this
                // type the quantizer emits that reshape in the SAME pass (one
                // extra store per value, no extra f32 load), and the kernel does
                // a contiguous tile copy -- which is what llama.cpp does by
                // having quantize_mmq_q8_1 write the layout directly.  q8_K is
                // still produced because other types in the same chunk and the
                // decode path consume it.
                const wants_mmq = !q8_0_act and Launch.mmq_dp4a_wants(@intCast(@backingInt(q.ttype))) != 0;
                if (wants_mmq) {
                    ensureBuf(&cu.lib, &self.d_xqm, &self.d_xqm_bytes, @as(usize, @intCast(total_blocks)) * @as(usize, @intCast(Launch.mmq_act_block_bytes()))) catch return oomMsg("batchK mmq activations");
                }
                const rc = if (q8_0_act)
                    kern.quantize_q8_0(base_act, @ptrCast(base_xq), @intCast(total_blocks), null)
                else if (wants_mmq)
                    kern.quantize_q8_k_mmq(base_act, @ptrCast(base_xq), self.d_xqm, @intCast(total_blocks), null)
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
                // The dp4a kernel takes the mmq layout, everything else q8_K.
                // `d_xqm` is non-null exactly when `wants_mmq` was set above (it
                // is `ensureBuf`'d right there), and `q35_mmq_dp4a_wants` is the
                // same predicate the kernel uses to pick its branch, so the two
                // cannot disagree.
                const rc_mm = kern_mm.mul_mat_vec(
                    e.dev,
                    if (wants_mmq) self.d_xqm else self.d_xq,
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
    fn dAllocScratch_body(ctx: *anyopaque, max_cols: usize) anyerror!void {
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
        // `d_idx` (the argmax result cell) is 4 bytes and was allocated lazily on
        // the first argmax.  It must not be allocated *during* a captured step,
        // so it is reserved here with everything else.
        try ensureBuf(&cu.lib, &self.d_idx, &self.d_idx_bytes, @sizeOf(c_int));
        // PV partials for the split attention path.  Sized for the widest head
        // count the config can produce at the largest split, allocated here so
        // the split path never allocates inside a capture.
        self.attn_split = attnSplitFromEnv();
        if (self.attn_split > 0) {
            try ensureBuf(&cu.lib, &self.d_att_part, &self.d_att_part_bytes, 64 * self.attn_split * 512 * @sizeOf(f32));
            // `heads * Q35_ATTN_MAX_ROWS` floats; 64 heads x 8192 = 2 MB.
            try ensureBuf(&cu.lib, &self.d_att_prob, &self.d_att_prob_bytes, 64 * 8192 * @sizeOf(f32));
        }
        // Same reasoning for the device `slot` cell that `attn_decode` reads:
        // allocating it on first use would land inside a capture, where
        // `cudaMalloc` is illegal ("operation not permitted when stream is
        // capturing").  Reserved here so the dyn path is always ready.
        _ = self.ensureAttnDyn();
    }

    /// K-token variant: the same buffers scaled by `k`.  `d_out`/`d_xq` dominate, so
    /// a wide prefill chunk shows up in the `CUDA-M` buffer total.
    fn dAllocScratchK_body(ctx: *anyopaque, max_cols: usize, k: usize) anyerror!void {
        const self = self_(ctx);
        if (!self.gpuReady()) return error.BackendUnavailable;
        try self.uploadIqTables();
        const cu = &self.cu.?;
        try ensureBuf(&cu.lib, &self.d_act, &self.d_act_bytes, k * max_cols * @sizeOf(f32));
        try ensureBuf(&cu.lib, &self.d_xq, &self.d_xq_bytes, k * actBytes(.q8_k, max_cols));
        try ensureBuf(&cu.lib, &self.d_out, &self.d_out_bytes, k * max_cols * @sizeOf(f32));
        // The decode-graph path runs after prefill, so THIS call is where the
        // scratch reaches its final size.  Re-assert the reservation for the
        // non-K sizes too: a decode step needs `max_cols`-sized `d_deq`/`d_idx`,
        // which `allocScratch` already provided, but making the requirement
        // explicit here means a future caller that only ever calls
        // `allocScratchK` still cannot allocate inside a capture.
        try self.graphReserve(max_cols);
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
    fn dAllocBuf_body(ctx: *anyopaque, len: usize) anyerror!backend.Buf {
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
        self.dev_bufs.append(self.alloc, .{ .ptr = p.?, .len = len }) catch {
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
        for (self.dev_bufs.items) |r| _ = cu.lib.Free(r.ptr);
        self.dev_bufs.clearRetainingCapacity();
        if (self.d_act) |p| _ = cu.lib.Free(p);
        if (self.d_xq) |p| _ = cu.lib.Free(p);
        if (self.d_xqm) |p| _ = cu.lib.Free(p);
        if (self.d_out) |p| _ = cu.lib.Free(p);
        if (self.d_deq) |p| _ = cu.lib.Free(p);
        self.d_act = null;
        self.d_act_bytes = 0;
        self.d_xq = null;
        self.d_xq_bytes = 0;
        self.d_xqm = null;
        self.d_xqm_bytes = 0;
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
    // `vLoadVecF32` (read-only 1-D weight vectors, uploaded once at load time),
    // and no device->host path on the per-token execution at all -- the only D2H
    // left is `downloadOut` for
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

    // The four slots.  Each is a FORWARDER rather than an alias to
    // `cpu_mod.CpuBackend.vtable.*`, for the same reason `dAllocBuf` is: `ctx`
    // is a `*CudaBackend` and these need it (device scratch + stream).  Every
    // one of them degrades to the embedded CPU backend on any device-side
    // failure, so a broken/missing GPU still produces correct numbers instead of
    // silently leaving the buffer untouched -- none of these slots can report an
    // error through the `Exec` signature, and panicking the process would be a
    // worse outcome than a slow correct answer.

    fn dScaleVec_body(ctx: *anyopaque, x: backend.Buf, s: f32) void {
        const self = self_(ctx);
        if (self.gOpGate(&self.g_op_seen.scale_vec)) return;
        if (self.cu != null) {
            if (dBufPtr(x)) |px| {
                const rc = Launch.scale_vec(px, s, @intCast(x.len), self.launchStream());
                if (rc == cudart.cudaSuccess) return;
                self.ewLaunchFail("scale_vec", rc);
            } else return;
        }
        cpu_mod.CpuBackend.vtable.scaleVec(@ptrCast(&self.cpu), x, s);
    }

    fn dAddResid_body(ctx: *anyopaque, dst: backend.Buf, src: backend.Buf) void {
        const self = self_(ctx);
        if (self.gOpGate(&self.g_op_seen.add_resid)) return;
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
                    const rc = Launch.add_resid(pd, ps, @intCast(dst.len), self.launchStream());
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
        const lo: usize = @intFromPtr(b.ptr);
        const hi: usize = lo + b.len * @sizeOf(f32);
        for (self.dev_bufs.items) |r| {
            const rlo: usize = @intFromPtr(r.ptr);
            const rhi: usize = rlo + r.len * @sizeOf(f32);
            // Containment, not equality: `Buf.span` advances the pointer, so a
            // strided view of our own allocation is still ours.
            if (lo >= rlo and hi <= rhi) return true;
        }
        return false;
    }

    /// KV append.  Byte-identical to `dCopyRange_body` except that, while a
    /// capture is open, the destination address is recorded so the graph can
    /// recompute it per token (`base + slot * stride`).  A plain `copyRange`
    /// cannot be used here: nothing about the arguments says which of a token's
    /// dozens of `copy_range` launches is the cache append.
    fn dCopyRangeSlot_body(ctx: *anyopaque, dst: backend.Buf, src: backend.Buf, slot: usize) void {
        const self = self_(ctx);
        // The KV append is INSIDE the graph: the engine hands over `cache.row(slot)`
        // as the destination, a pointer that advances one row per token, and the
        // captured node is retargeted to it.  Running the host copy as well would
        // write the same row twice (idempotent, so invisible in the numbers) while
        // making every replay pay one extra D2D move per layer.
        if (self.gOpGate(&self.g_op_seen.kv_append)) return;
        _ = slot;
        if (self.capturing() and src.len == dst.len) {
            if (dBufPtr(dst)) |pd| {
                if (self.isDevBuf(src)) {
                    // Stride is one row of the cache, i.e. however far this
                    // destination moves per token.  The engine hands the same
                    // `kv_dim`-wide row every step, so the length IS the stride.
                    self.graphNoteKvDst(pd, dst.len * @sizeOf(f32));
                }
            }
        }
        CudaBackend.dCopyRange_body(ctx, dst, src);
    }

    fn dCopyRange_body(ctx: *anyopaque, dst: backend.Buf, src: backend.Buf) void {
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
                        const rc = Launch.copy_range(pd, ps, @intCast(dst.len), self.launchStream());
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

    fn dZeroBuf_body(ctx: *anyopaque, buf: backend.Buf) void {
        const self = self_(ctx);
        if (self.gOpGate(&self.g_op_seen.zero_buf)) return;
        if (self.cu != null) {
            if (dBufPtr(buf)) |p| {
                const rc = Launch.zero_buf(p, @intCast(buf.len), self.launchStream());
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
    // These are the only two slots that can cross PCIe:
    //
    //   copyFromHost -- a token's f32 embedding row (unquantized token_embd
    //                   only), once per token, `hidden` floats.
    //   copyToHost   -- diagnostic only: nothing on the per-token path calls
    //                   it any more.  The last user was the per-layer
    //                   gamma/beta read-back, which `gdnStepRows` now takes as
    //                   device regions (`Buf`) instead of host `f32` arrays.
    //
    // Everything else in this file is device-to-device.

    fn dCopyFromHost_body(ctx: *anyopaque, dst: backend.Buf, src: []const f32) void {
        const self = self_(ctx);
        // The bytes behind `dst` just changed, so anything quantised from it is
        // stale.  Conservative and free: the next matvec re-quantises.
        self.actQClear();
        // A blocking copy INVALIDATES an open capture (the runtime returns
        // `cudaErrorStreamCaptureUnsupported` and the graph is left empty), so
        // the capture is abandoned here instead: this token then runs on the
        // direct path and the next one may capture again.  Detection has to be
        // here rather than at each engine call site because the engine has no
        // idea a capture is open -- by design (AGENTS.md 1.1: it must not know
        // about the hardware's execution strategy).
        if (self.capturing()) self.graphAbandon("host-to-device copy inside forward");
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

    fn dCopyToHost_body(ctx: *anyopaque, dst: []f32, src: backend.Buf) void {
        const self = self_(ctx);
        std.debug.assert(dst.len <= src.len);
        if (self.capturing()) self.graphAbandon("device-to-host copy inside forward");
        if (self.cu != null) {
            if (dBufPtr(src)) |ps| {
                // Blocking on purpose: this slot is for small diagnostic
                // regions now (no per-token caller), and a blocking copy into
                // pageable host memory is both simpler and, at that size,
                // indistinguishable in cost from an async one.
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
    //     a per-layer constant re-passed on every token, so it IS a device `Buf`
    //     uploaded exactly once at load time (`loadVecF32`); each call just reads
    //     the resident device pointer -- no per-token upload.

    fn dRmsnorm_body(ctx: *anyopaque, out: backend.Buf, x: backend.Buf, w: backend.Buf, eps: f32) void {
        const self = self_(ctx);
        if (self.gOpGate(&self.g_op_seen.rmsnorm)) return;
        // The CPU reference asserts out.len == x.len == w.len (one row).  The
        // kernel supports several rows (w reused per row) for the shape the
        // engine does not currently produce, so the only mismatch worth
        // refusing is a partial row.
        if (x.len != 0 and w.len != 0 and x.len % w.len == 0 and x.len == out.len and self.cu != null) {
            if (dBufPtr(x)) |px| {
                if (dBufPtr(out)) |po| {
                    if (dBufPtr(w)) |pw| {
                        const rows: c_int = @intCast(x.len / w.len);
                        const rc = Launch.rmsnorm(po, px, pw, rows, @intCast(w.len), eps, self.launchStream());
                        if (rc == cudart.cudaSuccess) return;
                        self.ewLaunchFail("rmsnorm", rc);
                    } else return;
                } else return;
            } else return;
        }
        cpu_mod.CpuBackend.vtable.rmsnorm(@ptrCast(&self.cpu), out, x, w, eps);
    }

    /// Row-batched `rmsnorm`: one launch, `rows` blocks, `w` reused per row.
    /// `norm.cu` `k_rmsnorm` already indexes `x`/`out` by `blockIdx.x * n`, so
    /// nothing in the kernel changes -- each block reduces its own row exactly
    /// as it did when it was the only block.  Rows are therefore bit-identical
    /// to the per-row launches (the reduction order within a row is untouched).
    ///
    /// Falls back to the per-row loop (`rmsnormRows` semantics) whenever the
    /// geometry does not hold, which is what the CPU side does anyway.
    fn dRmsnormRows_body(ctx: *anyopaque, out: backend.Buf, x: backend.Buf, w: backend.Buf, rows_in: usize, eps: f32) void {
        const self = self_(ctx);
        if (self.gOpGate(&self.g_op_seen.rmsnorm_rows)) return;
        const n = w.len;
        if (rows_in != 0 and n != 0 and x.len == rows_in * n and out.len == rows_in * n and self.cu != null) {
            if (dBufPtr(x)) |px| {
                if (dBufPtr(out)) |po| {
                    if (dBufPtr(w)) |pw| {
                        const rc = Launch.rmsnorm(po, px, pw, @intCast(rows_in), @intCast(n), eps, self.launchStream());
                        if (rc == cudart.cudaSuccess) {
                            return;
                        }
                        self.ewLaunchFail("rmsnorm_rows", rc);
                    } else return;
                } else return;
            } else return;
        }
        cpu_mod.CpuBackend.vtable.rmsnormRows(@ptrCast(&self.cpu), out, x, w, rows_in, eps);
    }

    fn dL2norm_body(ctx: *anyopaque, x: backend.Buf) void {
        const self = self_(ctx);
        if (self.cu != null) {
            if (dBufPtr(x)) |px| {
                // eps 1e-6 is elementwise.zig `l2norm`'s constant; it is passed
                // explicitly so the CPU reference stays the single definition.
                const rc = Launch.l2norm(px, 1, @intCast(x.len), 1e-6, self.launchStream());
                if (rc == cudart.cudaSuccess) return;
                self.ewLaunchFail("l2norm", rc);
            } else return;
        }
        cpu_mod.CpuBackend.vtable.l2norm(@ptrCast(&self.cpu), x);
    }

    /// Row-batched `l2norm`: one launch, `rows = x.len / n` blocks, each
    /// normalising its own `n`-element row with its own eps-inside-the-sum
    /// reduction.  `k_l2norm` already strides by `blockIdx.x * n`, so this is
    /// the same arithmetic as `x.len / n` single-row launches.
    fn dL2normRows_body(ctx: *anyopaque, x: backend.Buf, n_in: usize) void {
        const self = self_(ctx);
        const n = n_in;
        if (n != 0 and x.len % n == 0 and self.cu != null) {
            if (dBufPtr(x)) |px| {
                const rows: c_int = @intCast(x.len / n);
                const rc = Launch.l2norm(px, rows, @intCast(n), 1e-6, self.launchStream());
                if (rc == cudart.cudaSuccess) return;
                self.ewLaunchFail("l2norm_rows", rc);
            } else return;
        }
        // No batching possible (unknown row width): the whole-slice reading is
        // the single-row one, which is what `dL2norm_body` means.
        cpu_mod.CpuBackend.vtable.l2normRows(@ptrCast(&self.cpu), x, n);
    }

    fn dSiluInPlace_body(ctx: *anyopaque, x: backend.Buf) void {
        const self = self_(ctx);
        if (self.gOpGate(&self.g_op_seen.silu_in_place)) return;
        if (self.cu != null) {
            if (dBufPtr(x)) |px| {
                const rc = Launch.silu(px, @intCast(x.len), self.launchStream());
                if (rc == cudart.cudaSuccess) return;
                self.ewLaunchFail("silu", rc);
            } else return;
        }
        cpu_mod.CpuBackend.vtable.siluInPlace(@ptrCast(&self.cpu), x);
    }

    fn dMulSilu_body(ctx: *anyopaque, out: backend.Buf, gate: backend.Buf) void {
        const self = self_(ctx);
        if (self.gOpGate(&self.g_op_seen.mul_silu)) return;
        if (gate.len != out.len) {
            std.debug.print("cuda backend: mulSilu length mismatch (out={d} gate={d}); using the CPU path\n", .{ out.len, gate.len });
            cpu_mod.CpuBackend.vtable.mulSilu(@ptrCast(&self.cpu), out, gate);
            return;
        }
        if (self.cu != null) {
            if (dBufPtr(out)) |po| {
                if (dBufPtr(gate)) |pg| {
                    const rc = Launch.mul_silu(po, pg, @intCast(out.len), self.launchStream());
                    if (rc == cudart.cudaSuccess) return;
                    self.ewLaunchFail("mul_silu", rc);
                } else return;
            } else return;
        }
        cpu_mod.CpuBackend.vtable.mulSilu(@ptrCast(&self.cpu), out, gate);
    }

    fn dMulSigmoid_body(ctx: *anyopaque, out: backend.Buf, gate: backend.Buf) void {
        const self = self_(ctx);
        if (self.gOpGate(&self.g_op_seen.mul_sigmoid)) return;
        if (gate.len != out.len) {
            std.debug.print("cuda backend: mulSigmoid length mismatch (out={d} gate={d}); using the CPU path\n", .{ out.len, gate.len });
            cpu_mod.CpuBackend.vtable.mulSigmoid(@ptrCast(&self.cpu), out, gate);
            return;
        }
        if (self.cu != null) {
            if (dBufPtr(out)) |po| {
                if (dBufPtr(gate)) |pg| {
                    const rc = Launch.mul_sigmoid(po, pg, @intCast(out.len), self.launchStream());
                    if (rc == cudart.cudaSuccess) return;
                    self.ewLaunchFail("mul_sigmoid", rc);
                } else return;
            } else return;
        }
        cpu_mod.CpuBackend.vtable.mulSigmoid(@ptrCast(&self.cpu), out, gate);
    }

    fn dSwiglu_body(ctx: *anyopaque, gate: backend.Buf, up: backend.Buf) void {
        const self = self_(ctx);
        if (self.gOpGate(&self.g_op_seen.swiglu)) return;
        if (up.len != gate.len) {
            std.debug.print("cuda backend: swiglu length mismatch (gate={d} up={d}); using the CPU path\n", .{ gate.len, up.len });
            cpu_mod.CpuBackend.vtable.swiglu(@ptrCast(&self.cpu), gate, up);
            return;
        }
        if (self.cu != null) {
            if (dBufPtr(gate)) |pg| {
                if (dBufPtr(up)) |pu| {
                    const rc = Launch.swiglu(pg, pu, @intCast(gate.len), self.launchStream());
                    if (rc == cudart.cudaSuccess) return;
                    self.ewLaunchFail("swiglu", rc);
                } else return;
            } else return;
        }
        cpu_mod.CpuBackend.vtable.swiglu(@ptrCast(&self.cpu), gate, up);
    }

    fn dSoftmax_body(ctx: *anyopaque, x: backend.Buf) void {
        const self = self_(ctx);
        if (self.gOpGate(&self.g_op_seen.softmax)) return;
        if (self.cu != null) {
            if (dBufPtr(x)) |px| {
                const rc = Launch.softmax(px, @intCast(x.len), self.launchStream());
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

    fn dArgmax_body(ctx: *anyopaque, logits: backend.Buf) u32 {
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
                // $QWEN35_PROF: split the per-token drain into "device was busy"
                // and "host was waiting".  `ev_arg_a` is recorded before the
                // argmax launch and `ev_arg_b` right after it; both are queue
                // positions, so `b - a` is what the device spent finishing that
                // tail.  The host wall clock around the blocking copy is what
                // the CPU actually paid.  Both are only touched when
                // gpuTimingOn(), so the default path is one predictable branch.
                const timed = self.gpuTimingOn();
                var host_t0: ?Io.Clock.Timestamp = null;
                if (timed) {
                    if (!self.ensureEvents()) {
                        self.ev_on = false;
                    } else {
                        host_t0 = Io.Clock.Timestamp.now(self.io, .awake);
                        if (cu.lib.EventRecord(self.ev_arg_a, null) != cudart.cudaSuccess) {
                            self.stat_n_ev_err += 1;
                        }
                    }
                }
                const rc = Launch.argmax(px, @intCast(logits.len), @ptrCast(@alignCast(self.d_idx)), null);
                if (rc != cudart.cudaSuccess) {
                    self.ewLaunchFail("argmax", rc);
                    return 0;
                }
                if (timed and host_t0 != null) {
                    if (cu.lib.EventRecord(self.ev_arg_b, null) != cudart.cudaSuccess) {
                        self.stat_n_ev_err += 1;
                    }
                }
                var host_idx: c_int = 0;
                // Measured 2026-11: a pinned `MemcpyAsync` + stream sync is
                // 5191us -> 5100us per token, i.e. noise.  The wait here is the
                // device finishing the 240 kernels this token already queued
                // (8.25 ms of real device work against an 8.20 ms host token),
                // not copy overhead, so the plain blocking copy stays.
                const rc2 = cu.lib.Memcpy(&host_idx, self.d_idx, @sizeOf(c_int), cudart.MemcpyDeviceToHost);
                if (rc2 != cudart.cudaSuccess) {
                    self.ewLaunchFail("argmax index download", rc2);
                    return 0;
                }
                if (timed) {
                    if (host_t0) |t0| {
                        self.stat_ns_argmax_host += durNs(t0, self.io);
                        self.stat_n_argmax_timed += 1;
                    }
                    // The events are now complete (the copy above waited on the
                    // whole queue), so reading the interval synchronizes nothing
                    // extra.  Guarded: `evElapsedNs` returns null on error.
                    if (self.evElapsedNs(self.ev_arg_a, self.ev_arg_b)) |ns| {
                        self.stat_ns_argmax_gpu += ns;
                    }
                    // Record the device position that closes THIS token, then
                    // compare it against the previous token's close.  That spans
                    // every kernel of the token, which is the number that has to
                    // be compared with the host wall clock.
                    if (cu.lib.EventRecord(self.ev_tok_end, null) == cudart.cudaSuccess) {
                        if (self.tok_end_valid) {
                            if (self.evElapsedNs(self.ev_prev_tok_end, self.ev_tok_end)) |ns| {
                                self.stat_ns_token_dev += ns;
                                self.stat_n_token_dev += 1;
                            }
                        }
                        if (cu.lib.EventRecord(self.ev_prev_tok_end, null) != cudart.cudaSuccess) {
                            self.stat_n_ev_err += 1;
                        }
                        self.tok_end_valid = true;
                    } else self.stat_n_ev_err += 1;
                }
                self.stat_d2h_bytes += @sizeOf(c_int);
                if (self.gpu_ev_on) self.gpuEvReadToken();
                return @intCast(host_idx);
            } else return 0;
        }
        return cpu_mod.CpuBackend.vtable.argmax(@ptrCast(&self.cpu), logits);
    }

    /// Grow the A/B host row to at least `n` floats.  Only ever called from the
    /// debug branch, i.e. outside any capture, so a plain allocation is fine.
    fn ensureSampleAbHost(self: *CudaBackend, n: usize) bool {
        if (self.h_sample_ab) |b| {
            if (b.len >= n) return true;
        }
        const alloc = std.heap.page_allocator;
        if (self.h_sample_ab) |b| alloc.free(b);
        self.h_sample_ab = alloc.alloc(f32, n) catch {
            self.h_sample_ab = null;
            return false;
        };
        return true;
    }

    /// Allocate (once) the device control block, its pinned host mirror, and
    /// the kernel scratch.  Must run BEFORE any capture opens: `cudaMalloc`
    /// inside a capture interval is illegal, and the sampler's very first launch
    /// would land there.
    fn ensureSampleBufs(self: *CudaBackend) bool {
        if (self.d_sample_ctl == null or self.d_sample_scratch == null) {
            const rc = Launch.sample_init(&self.d_sample_ctl, &self.d_sample_scratch);
            if (rc != cudart.cudaSuccess) {
                self.ewLaunchFail("sample init", rc);
                return false;
            }
        }
        if (self.h_sample_ctl == null) {
            var p: ?*anyopaque = null;
            const n: usize = @intCast(@max(Launch.sample_ctl_ints(), 16));
            // flags 0 = plain pinned memory: page-locked but not mapped, which
            // is all `cudaMemcpyAsync` H2D needs to stay a true async copy.
            const hrc = (&(self.cu orelse return false)).lib.HostAlloc(&p, n * @sizeOf(c_int), 0);
            if (hrc != cudart.cudaSuccess or p == null) {
                self.ewLaunchFail("sample ctl host alloc", hrc);
                return false;
            }
            const hc: [*]c_int = @ptrCast(@alignCast(p.?));
            @memset(hc[0..n], 0);
            self.h_sample_ctl = p;
        }
        return true;
    }

    /// Mirror of `q35_sample_ctl` in `src/kernels/cuda/sample.cu`.  Field order
    /// is the WIRE FORMAT of the 64-byte H2D -- change both files or neither.
    const SampleCtl = extern struct {
        vocab: c_int = 0,
        temperature_bits: c_int = 0,
        top_k: c_int = 0,
        top_p_bits: c_int = 0,
        rng_lo: c_int = 0,
        rng_hi: c_int = 0,
        token_out: c_int = 0,
        rng_out_lo: c_int = 0,
        rng_out_hi: c_int = 0,
        thresh_bits: c_int = 0,
        flags: c_int = 0,
        sel_prefix: c_int = 0,
        sel_mask: c_int = 0,
        sel_rank: c_int = 0,
        reserved0: c_int = 0,
        reserved1: c_int = 0,
    };

    /// Publish this token's sampler parameters into the device control block.
    /// One 64-byte H2D onto the caller's stream; no kernel, no allocation.
    ///
    /// `sp.seed` is not re-read every token: `Sampler.state` is the live PRNG
    /// position, and it advances exactly once per drawn token (the kernel writes
    /// the advanced value back into `rng_out_*`, which `dSample_body` copies into
    /// `sp.state` when it collects the token).  Publishing `state` here is what
    /// makes a replayed stream continue instead of restarting.
    fn publishSampleCtl(self: *CudaBackend, logits_len: usize, sp: *const sampler.Sampler, stream: ?*anyopaque) bool {
        const h = self.h_sample_ctl orelse return false;
        const hc: *SampleCtl = @ptrCast(@alignCast(h));
        hc.* = .{};
        hc.vocab = @intCast(logits_len);
        hc.temperature_bits = @bitCast(sp.temperature);
        hc.top_k = @intCast(sp.top_k);
        hc.top_p_bits = @bitCast(sp.top_p);
        const seed: u64 = sp.state;
        hc.rng_lo = @bitCast(@as(u32, @truncate(seed)));
        hc.rng_hi = @bitCast(@as(u32, @truncate(seed >> 32)));
        const cu = &(self.cu orelse return false);
        const rc = cu.lib.MemcpyAsync(
            self.d_sample_ctl, @ptrCast(h), @sizeOf(SampleCtl), cudart.MemcpyHostToDevice, stream,
        );
        if (rc != cudart.cudaSuccess) {
            self.ewLaunchFail("sample ctl publish", rc);
            return false;
        }
        self.stat_h2d_bytes += @sizeOf(SampleCtl);
        return true;
    }

    /// Read back the 4-byte token and the advanced PRNG state, and hand the
    /// token to the engine.  Called from `stepEnd` (outside any capture) and, on
    /// the un-captured path, directly from `dSample_body`.
    fn collectSampledToken(self: *CudaBackend, sp: *sampler.Sampler) u32 {
        const cu = &(self.cu orelse return self.sample_token);
        const h = self.h_sample_ctl orelse return self.sample_token;
        const hc: *SampleCtl = @ptrCast(@alignCast(h));
        // Blocking 64-byte pull (the stream already carries every dependency:
        // this runs on the DEFAULT stream, which `graphSyncToDefault` ordered
        // after the graph).  Reading the last 64 bytes of the block -- rather
        // than only `token_out` -- also brings back the advanced PRNG state, so
        // the next token's `publishSampleCtl` continues the same stream without
        // a second copy.  Cost is the same: one D2H of one cache line.
        const rc = cu.lib.Memcpy(
            @ptrCast(h), self.d_sample_ctl, @sizeOf(SampleCtl), cudart.MemcpyDeviceToHost,
        );
        if (rc != cudart.cudaSuccess) {
            self.ewLaunchFail("sample token download", rc);
            return self.sample_token;
        }
        self.stat_d2h_bytes += @sizeOf(SampleCtl);
        // `Sampler.sample` consumes exactly one `nextU64` per non-greedy draw
        // (sampler.zig:120 `nextF32`), and the kernel does the same, so the state
        // it wrote back IS the position the CPU reference would be at.
        if (sp.temperature > 0.0) {
            sp.state = (@as(u64, @as(u32, @bitCast(hc.rng_out_hi))) << 32) | @as(u32, @bitCast(hc.rng_out_lo));
        }
        const tok: u32 = @intCast(@as(u32, @bitCast(hc.token_out)));
        self.sample_token = tok;
        if (self.gpu_ev_on) self.gpuEvReadToken();
        return tok;
    }

    /// Device sampling.
    ///
    /// The whole policy -- greedy argmax, temperature, top-k, top-p, PRNG advance
    /// -- runs in `sample.cu`.  Two things matter about HOW it is invoked:
    ///
    ///   * The launch goes to `launchStream()`, i.e. to the capture stream while a
    ///     decode graph is being recorded, so the sampler becomes a node of that
    ///     graph exactly like the matvecs.  On a replay the gate below suppresses
    ///     the host launch because the graph already contains it.
    ///   * The result is NOT downloaded here on the captured path.  A D2H cannot
    ///     be captured; `stepEnd` performs it after the graph has been closed, and
    ///     `sampledToken` hands the value to the engine.  On the direct path the
    ///     download happens immediately and `sampledToken` is the identity.
    fn dSample_body(ctx: *anyopaque, logits: backend.Buf, scratch: backend.Buf, sp: *sampler.Sampler) u32 {
        const self = self_(ctx);
        // A/B switch: run the CPU reference sampler on the GPU's own logits.
        //
        // This is the ONLY way to compare the two samplers end to end, because
        // every other comparison runs into the CPU/CUDA matvec divergence first
        // (f32 accumulation order, documented in AGENTS.md §4) and the samplers
        // are then fed different logits.  With this set, both runs consume the
        // identical device logits row, so any token difference is the sampler's
        // fault alone.  Read once per process, on the host, before any capture.
        if (self.sample_cpu_ab and self.capturing()) {
            // A blocking D2H inside a capture is illegal and would silently
            // corrupt the graph.  Refuse loudly: this switch is for A/B on the
            // direct path only, and `graphAbandon` is the documented way to say
            // "this capture cannot continue".
            self.graphAbandon("QWEN35_CUDA_SAMPLE_CPU inside a capture");
        }
        if (self.sample_cpu_ab) {
            // The CPU sampler indexes the row directly, so it needs a HOST copy:
            // `logits` is a device pointer here.  `h_sample_ab` is grown on
            // demand and reused, and the copy is blocking -- this whole branch is
            // a debug A/B, never the compute path.
            if (!self.ensureSampleAbHost(logits.len)) {
                return cpu_mod.CpuBackend.vtable.sample(@ptrCast(&self.cpu), logits, scratch, sp);
            }
            const dst = self.h_sample_ab.?[0..logits.len];
            // Inlined blocking D2H (`dCopyToHost_body` is a sibling struct's
            // method and is not reachable by name from here).
            const cu = &(self.cu orelse return cpu_mod.CpuBackend.vtable.sample(
                @ptrCast(&self.cpu), logits, scratch, sp,
            ));
            const ps = dBufPtr(logits) orelse return cpu_mod.CpuBackend.vtable.sample(
                @ptrCast(&self.cpu), logits, scratch, sp,
            );
            const rc = cu.lib.Memcpy(
                @ptrCast(dst.ptr), @ptrCast(ps), dst.len * @sizeOf(f32), cudart.MemcpyDeviceToHost,
            );
            if (rc != cudart.cudaSuccess) {
                self.ewLaunchFail("sample ab row download", rc);
                return cpu_mod.CpuBackend.vtable.sample(@ptrCast(&self.cpu), logits, scratch, sp);
            }
            const host_row = backend.Buf.wrap(dst);
            // `scratch` reaches the CPU sampler too, and the engine's copy is a
            // DEVICE buffer when this backend is in use; hand the reference a
            // host scratch of the same length or it will sort a device pointer.
            if (self.h_sample_ab_scratch == null or self.h_sample_ab_scratch.?.len < logits.len) {
                const alloc = std.heap.page_allocator;
                if (self.h_sample_ab_scratch) |b| alloc.free(b);
                self.h_sample_ab_scratch = alloc.alloc(f32, logits.len) catch null;
            }
            const hs = self.h_sample_ab_scratch orelse
                return cpu_mod.CpuBackend.vtable.sample(@ptrCast(&self.cpu), host_row, scratch, sp);
            return cpu_mod.CpuBackend.vtable.sample(
                @ptrCast(&self.cpu), host_row, backend.Buf.wrap(hs[0..logits.len]), sp,
            );
        }
        // The device sampler owns its own scratch (`d_sample_scratch`); the
        // engine-provided one is only needed by the CPU fallbacks below.
        if (self.cu == null or logits.len == 0) {
            return cpu_mod.CpuBackend.vtable.sample(@ptrCast(&self.cpu), logits, scratch, sp);
        }
        // `gOpGate` cannot be used verbatim: it returns true for a REPLAYED step,
        // where this call still has to produce a token (from the graph's own
        // output), so the sampler branch is spelled out.
        const replayed = self.g_state == .live;
        if (replayed) {
            // The graph already ran the sampler on `g_stream`.  `stepEnd` ordered
            // that stream before the default one, so the control block holds this
            // token's result; `collectSampledToken` is deferred to `stepEnd`.
            self.sample_pending = true;
            self.sample_sampler = sp;
            return 0; // placeholder; `sampledToken` overrides it
        }
        if (!self.ensureSampleBufs()) {
            return cpu_mod.CpuBackend.vtable.sample(@ptrCast(&self.cpu), logits, scratch, sp);
        }
        const px = dBufPtr(logits) orelse
            return cpu_mod.CpuBackend.vtable.sample(@ptrCast(&self.cpu), logits, scratch, sp);
        const stream = self.launchStream(); // capture stream while recording
        if (!self.publishSampleCtl(logits.len, sp, stream)) {
            return cpu_mod.CpuBackend.vtable.sample(@ptrCast(&self.cpu), logits, scratch, sp);
        }
        const rc = Launch.sample(
            px, @intCast(logits.len), @ptrCast(@alignCast(self.d_sample_ctl)), self.d_sample_scratch, stream,
        );
        if (rc != cudart.cudaSuccess) {
            self.ewLaunchFail("sample", rc);
            return cpu_mod.CpuBackend.vtable.sample(@ptrCast(&self.cpu), logits, scratch, sp);
        }
        if (self.capturing()) {
            // Inside the capture: nothing can be downloaded, and nothing needs to
            // be -- this step's token comes back through `stepEnd`.
            self.sample_pending = true;
            self.sample_sampler = sp;
            return 0;
        }
        // Direct path: the launch is on the default stream, so a blocking copy
        // right here is already ordered after it.
        self.sample_pending = false;
        self.sample_sampler = null;
        return self.collectSampledToken(sp);
    }

    /// The token from the current step's `sample`, after `stepEnd` downloaded it.
    fn dSampledToken(ctx: *anyopaque, hinted: u32) u32 {
        const self = self_(ctx);
        if (!self.sample_pending) return hinted;
        self.sample_pending = false;
        return self.sample_token;
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

    fn dGdnStep_body(ctx: *anyopaque, state: backend.Buf, k: backend.Buf, v: backend.Buf,
                q: backend.Buf, km: backend.Buf, delta: backend.Buf, out: backend.Buf,
                gamma: f32, beta: f32, vd: usize) void {
        const self = self_(ctx);
        if (self.gOpGate(&self.g_op_seen.gdn_step)) return;
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
                                        const rc = Launch.gdn_step(pstate, pk, pv, pq, pkm, pdelta, pout, gamma, beta, @intCast(vd), self.launchStream());
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

    /// Batched `dGdnStep_body`: one launch for all `heads`, otherwise the same
    /// per-head call.  Shaped like `dRmsnormRows_body` / `dL2normRows_body`:
    /// check every precondition, launch, and on launch failure log once and
    /// fall through to the CPU reference -- which for this slot is the per-head
    /// loop, so a fallback costs `heads` host calls, never a wrong answer.
    ///
    /// The precondition set is the batch's geometry, not a heuristic: the
    /// per-head views the kernel computes must lie inside the buffers the
    /// caller handed us (a head whose slice overruns would read a neighbouring
    /// head's state, which is silent garbage rather than a fault).
    fn dGdnStepRows_body(ctx: *anyopaque, state: backend.Buf, k: backend.Buf, v: backend.Buf,
                         q: backend.Buf, km: backend.Buf, delta: backend.Buf, out: backend.Buf,
                         gamma: backend.Buf, beta: backend.Buf,
                         heads: usize, nk: usize, kd: usize, vd: usize) void {
        const self = self_(ctx);
        if (self.gOpGate(&self.g_op_seen.gdn_step_rows)) return;
        if (vd == 128 and kd == 128 and heads != 0 and nk != 0 and nk <= heads and
            gamma.len >= heads and beta.len >= heads and
            state.len >= heads * kd * vd and k.len >= nk * kd and q.len >= nk * kd and
            v.len >= heads * vd and out.len >= heads * vd and
            km.len >= heads * vd and delta.len >= heads * vd and
            self.cu != null)
        {
            if (dBufPtr(state)) |pstate| {
                if (dBufPtr(k)) |pk| {
                    if (dBufPtr(v)) |pv| {
                        if (dBufPtr(q)) |pq| {
                            if (dBufPtr(km)) |pkm| {
                                if (dBufPtr(delta)) |pdelta| {
                                    if (dBufPtr(out)) |pout| {
                                        // gamma/beta are backend regions
                                        // (`gdnGamma`'s output and its in-place
                                        // sigmoid(b) write-back), so the kernel
                                        // reads them where they already live:
                                        // no H2D, no D2H, nothing per token.
                                        // They must NOT be cached as read-only
                                        // weights (pointer-keyed, upload-once):
                                        // their contents change per layer per
                                        // token, which would silently freeze
                                        // layer 0's coefficients for the run.
                                        if (dBufPtr(gamma)) |pgamma| {
                                            if (dBufPtr(beta)) |pbeta| {
                                                const rc = Launch.gdn_step_rows(
                                                    pstate, @intCast(kd * vd),
                                                    pk, @intCast(kd),
                                                    pv, @intCast(vd),
                                                    pq, @intCast(kd),
                                                    pkm, pdelta, pout,
                                                    pgamma, pbeta,
                                                    @intCast(heads), @intCast(nk),
                                                    @intCast(kd), @intCast(vd), self.launchStream(),
                                                );
                                                if (rc == cudart.cudaSuccess) return;
                                                if (self.capturing()) self.graphAbandon("gdn_step_rows launch failed");
                                                self.ewLaunchFail("gdn_step_rows", rc);
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
        cpu_mod.CpuBackend.vtable.gdnStepRows(@ptrCast(&self.cpu), state, k, v, q, km, delta, out, gamma, beta, heads, nk, kd, vd);
    }

    fn dGdnGamma_body(ctx: *anyopaque, a_raw: backend.Buf, b_raw: backend.Buf, ssm_a: backend.Buf,
                 dt_bias: backend.Buf, gamma: backend.Buf, nv: usize) void {
        const self = self_(ctx);
        if (self.gOpGate(&self.g_op_seen.gdn_gamma)) return;
        // Every operand here is a backend region.  The two weight rows
        // (`ssm_a`, `dt_bias`) are per-layer constants and, like the norm
        // weights, are device `Buf`s uploaded exactly once at load time
        // (`loadVecF32`); the rest are live per-token regions.
        if (self.cu != null and nv != 0 and
            a_raw.len >= nv and b_raw.len >= nv and ssm_a.len >= nv and
            dt_bias.len >= nv and gamma.len >= nv)
        {
            if (dBufPtr(a_raw)) |pa| {
                if (dBufPtr(b_raw)) |pb| {
                    if (dBufPtr(gamma)) |pg| {
                        if (dBufPtr(ssm_a)) |sa| {
                            if (dBufPtr(dt_bias)) |db| {
                                const rc = Launch.gdn_gamma(pa, pb, sa, db, pg, @intCast(nv), self.launchStream());
                                if (rc == cudart.cudaSuccess) return;
                                self.ewLaunchFail("gdn_gamma", rc);
                            } else return;
                        } else return;
                    } else return;
                } else return;
            } else return;
        }
        cpu_mod.CpuBackend.vtable.gdnGamma(@ptrCast(&self.cpu), a_raw, b_raw, ssm_a, dt_bias, gamma, nv);
    }

    /// CUDA form of `VTable.gdnPrefillCore`: the whole chunk's conv+silu, gamma /
    /// beta and l2norm/scale passes in FOUR launches instead of `tokens` x ~14.
    ///
    /// Order matters and is preserved: conv first (token t's taps need what
    /// token t-1 shifted in -- the kernel walks the tokens ascending inside one
    /// thread per channel), then gamma (reads the RAW a/b projections, which conv
    /// does not touch), then the norm/scale passes (read the conv output).
    /// `gdnStepRows` afterwards only READS qkv, so hoisting all of this ahead of
    /// the state loop is order-preserving, not just commute-safe.
    ///
    /// Falls back to the CPU body when any region is not device-resident or a
    /// launch fails -- same policy as every other forwarder here (numbers stay
    /// right, only the speedup is lost).
    fn dGdnPrefillCore_body(ctx: *anyopaque, qkv: backend.Buf, conv: backend.Buf,
                            wconv: backend.Buf, ab: backend.Buf, ssm_a: backend.Buf,
                            dt_bias: backend.Buf, gamma: backend.Buf, tokens: usize,
                            conv_dim: usize, ab_a_offset: usize, ab_b_offset: usize,
                            ab_stride: usize, gamma_stride: usize, kq_dim: usize,
                            nv: usize, nk: usize, kd: usize, conv_k: usize, eps: f32) void {
        const self = self_(ctx);
        const stream = self.launchStream();
        const nk_reps = if (nk != 0) nv / nk else 0;
        const qscale: f32 = 1.0 / @sqrt(@as(f32, @floatFromInt(kd)));

        // Shape gates: conv is fixed at 4 taps (elementwise.zig `convStepKernel`
        // asserts it), the q/k runs must be `nk` whole `kd` rows, and the tiled
        // layout is what makes the batched norm legal at all.
        const shaped = tokens > 0 and nv > 0 and nk > 0 and kd > 0 and
            conv_k == 4 and nv % nk == 0 and nk_reps >= 1 and
            kq_dim == nk * kd and
            conv_dim >= 2 * kq_dim and
            qkv.len >= tokens * conv_dim and
            ab.len >= @max(ab_a_offset, ab_b_offset) + tokens * ab_stride and
            gamma.len >= tokens * gamma_stride;
        if (self.cu != null and shaped) {
            if (dBufPtr(qkv)) |pq| {
                if (dBufPtr(conv)) |pc| {
                    if (dBufPtr(wconv)) |pw| {
                        if (dBufPtr(ab)) |pab| {
                            if (dBufPtr(ssm_a)) |sa| {
                                if (dBufPtr(dt_bias)) |db| {
                                    if (dBufPtr(gamma)) |pg| {
                                        // 1. conv + silu for every token, one shared history.
                                        var rc = Launch.conv_step_rows(
                                            pc, pq, pw, @intCast(conv_dim), @intCast(tokens),
                                            @intCast(conv_dim), stream);
                                        if (rc != cudart.cudaSuccess) {
                                            self.ewLaunchFail("conv_step_rows", rc);
                                            return;
                                        }
                                        // 2. beta (in place) + gamma, one row per token.
                                        rc = Launch.gdn_gamma_rows(
                                            pab + ab_a_offset, pab + ab_b_offset, sa, db, pg,
                                            @intCast(nv), @intCast(tokens),
                                            @intCast(ab_stride), @intCast(ab_stride),
                                            @intCast(gamma_stride), stream);
                                        if (rc != cudart.cudaSuccess) {
                                            self.ewLaunchFail("gdn_gamma_rows", rc);
                                            return;
                                        }
                                        // 3. the q/k norm + q scale passes, `nk_reps`
                                        //    times in the order the per-head loop made
                                        //    them: l2norm(q), l2norm(k), q *= qscale.
                                        var rep: usize = 0;
                                        while (rep < nk_reps) : (rep += 1) {
                                            rc = Launch.l2norm_strided(
                                                pq, 0, @intCast(nk), @intCast(kd), eps,
                                                @intCast(tokens), @intCast(conv_dim), stream);
                                            if (rc != cudart.cudaSuccess) {
                                                self.ewLaunchFail("l2norm_strided_q", rc);
                                                return;
                                            }
                                            rc = Launch.l2norm_strided(
                                                pq, @intCast(kq_dim), @intCast(nk), @intCast(kd),
                                                eps, @intCast(tokens), @intCast(conv_dim), stream);
                                            if (rc != cudart.cudaSuccess) {
                                                self.ewLaunchFail("l2norm_strided_k", rc);
                                                return;
                                            }
                                            rc = Launch.scale_vec_strided(
                                                pq, 0, qscale, @intCast(kq_dim),
                                                @intCast(tokens), @intCast(conv_dim), stream);
                                            if (rc != cudart.cudaSuccess) {
                                                self.ewLaunchFail("scale_vec_strided", rc);
                                                return;
                                            }
                                        }
                                        return;
                                    } else return;
                                } else return;
                            } else return;
                        } else return;
                    } else return;
                } else return;
            } else return;
        }
        cpu_mod.CpuBackend.vtable.gdnPrefillCore(
            @ptrCast(&self.cpu), qkv, conv, wconv, ab, ssm_a, dt_bias, gamma, tokens,
            conv_dim, ab_a_offset, ab_b_offset, ab_stride, gamma_stride, kq_dim,
            nv, nk, kd, conv_k, eps);
    }

    fn dAttnDecode_body(ctx: *anyopaque, q: backend.Buf, gate: backend.Buf, kcache: backend.Buf,
                   vcache: backend.Buf, att: backend.Buf, gout: backend.Buf, slot: usize,
                   heads: usize, gqa: usize, hd: usize, kv_dim: usize, scale: f32) void {
        const self = self_(ctx);
        if (self.gOpGate(&self.g_op_seen.attn_decode)) return;
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
                                    // Take the DEVICE-slot path whenever this launch
                                    // may be captured (or is being replayed).  On that
                                    // path the node's arguments and its
                                    // `shared_mem_bytes` are both constants, so the
                                    // replay never calls `GraphExecKernelNodeSetParams`
                                    // for it.  Measured on the 27B: retargeting this
                                    // one node per layer per token cost 14.3 ms of a
                                    // 33.8 ms token (43%), against 1.4 ms of actual
                                    // kernel time.  The scalar-`slot` form stays for
                                    // the direct path, where there is no node to
                                    // retarget and the smaller shared block is free.
                                    const want_dyn = graphWanted() and
                                        (self.g_state == .live or self.g_state == .rec or self.capturing());
                                    if (want_dyn) {
                                        if (self.ensureAttnDyn()) |pdyn| {
                                            // DO NOT publish `slot` here.  During
                                            // capture `launchStream()` is the capture
                                            // stream, so a publish issued now would
                                            // become a graph NODE with this token's
                                            // value baked into its arguments -- and on
                                            // every replay that node would overwrite
                                            // the fresh value `graphReplayInner`
                                            // writes, pinning the whole graph to the
                                            // capture token's KV length.  (Measured:
                                            // the first 6 tokens matched, then the text
                                            // diverged exactly when replay began, and
                                            // the device cell read back the capture
                                            // value.)  The publish lives in
                                            // `graphReplayInner`, on `g_stream`,
                                            // immediately before `GraphLaunch`.
                                            const sslot = @as([*]c_int, @ptrCast(@alignCast(pdyn)));
                                            // Prefer the SPLIT device-slot path when
                                            // enabled: four constant-argument nodes,
                                            // still zero retargeting, and the row
                                            // walk is spread over `split` blocks per
                                            // head instead of one.
                                            if (self.attn_split > 1) {
                                                if (self.d_att_part) |ppart| {
                                                    const s2 = @as([*]f32, @ptrCast(@alignCast(ppart)));
                                                    const pprob = if (self.d_att_prob) |pp| @as([*]f32, @ptrCast(@alignCast(pp))) else patt;
                                                    const rcs = Launch.attn_decode_split_dyn(pq, pgate, pk, pv, pprob, pgout, s2, sslot, @intCast(heads), @intCast(gqa), @intCast(hd), @intCast(kv_dim), scale, @intCast(self.attn_split), self.launchStream());
                                                    if (rcs == cudart.cudaSuccess) {
                                                        // FOUR nodes (score, softmax,
                                                        // PV, reduce+gate), all
                                                        // `.attn_dyn`-class: none is
                                                        // ever retargeted.  The census
                                                        // must see all four.
                                                        if (self.capturing()) self.graphNoteAttnN(slot, 4);
                                                        return;
                                                    }
                                                    if (self.capturing()) self.graphAbandon("attn_decode split_dyn launch failed");
                                                    self.ewLaunchFail("attn_decode_split_dyn", rcs);
                                                }
                                            }
                                            const rc = Launch.attn_decode_dyn(pq, pgate, pk, pv, patt, pgout, sslot, @intCast(heads), @intCast(gqa), @intCast(hd), @intCast(kv_dim), scale, self.launchStream());
                                            if (rc == cudart.cudaSuccess) {
                                                // Counted, not recorded: the node is
                                                // classified `.attn_dyn` and never
                                                // retargeted, but `graphCaptureEnd`
                                                // demands the census agree.
                                                if (self.capturing()) self.graphNoteAttn(slot);
                                                return;
                                            }
                                            if (self.capturing()) self.graphAbandon("attn_decode dyn launch failed");
                                            self.ewLaunchFail("attn_decode_dyn", rc);
                                        }
                                    }
                                    // `slot` (and only `slot`) changes per token: the
                                    // cache POINTERS are fixed regions and the engine
                                    // merely hands over a longer prefix of them, so a
                                    // captured node stays valid as long as the kernel is
                                    // told the new slot.  That is the one retarget.
                                    //
                                    // With `$QWEN35_CUDA_ATTN_SPLIT` set, take the
                                    // row-split variant instead: it divides BOTH row
                                    // walks across that many blocks per head, because
                                    // one block per head is 24 blocks on 114 SMs (21%)
                                    // and every lane then walks all `n` rows serially
                                    // (measured 33 GB/s on non-L2-resident KV caches).
                                    if (self.attn_split > 1) {
                                        if (self.d_att_part) |ppart| {
                                            const s2 = @as([*]f32, @ptrCast(@alignCast(ppart)));
                                            // `d_att_prob`, NOT `patt`: see the field note.
                                            const pprob = if (self.d_att_prob) |pp| @as([*]f32, @ptrCast(@alignCast(pp))) else patt;
                                            const rc2 = Launch.attn_decode_split2(pq, pgate, pk, pv, pprob, pgout, s2, @intCast(slot), @intCast(heads), @intCast(gqa), @intCast(hd), @intCast(kv_dim), scale, @intCast(self.attn_split), self.launchStream());
                                            if (rc2 == cudart.cudaSuccess) return;
                                            self.ewLaunchFail("attn_decode split", rc2);
                                        }
                                    }
                                    const rc = Launch.attn_decode(pq, pgate, pk, pv, patt, pgout, @intCast(slot), @intCast(heads), @intCast(gqa), @intCast(hd), @intCast(kv_dim), scale, self.launchStream());
                                    if (rc == cudart.cudaSuccess) {
                                        if (self.capturing()) self.graphNoteAttn(slot);
                                        return;
                                    }
                                    // A failed launch cannot be part of a graph: the
                                    // capture would then be missing a whole layer's
                                    // attention.  Abandon rather than instantiate.
                                    if (self.capturing()) self.graphAbandon("attn_decode launch failed");
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
    /// Lazily allocate the two-int {pos, base} channel for `k_rope64_dpos`.
    /// Returns the DEVICE address the kernel reads, or null on failure (caller
    /// then falls back to the constant-argument kernel, so a missing 8 bytes
    /// degrades to "retarget per token" rather than to wrong math).
    ///
    /// The two rejected designs are worth recording, because both look right:
    ///   1. `cudaMalloc` + `cudaMemcpyAsync(H2D)` on the launch stream -- a
    ///      host-to-device copy is not capturable, so the first captured
    ///      `rope64` failed and the whole decode graph was abandoned
    ///      (`rope64 launch failed: operation failed due to a previous error
    ///      during capture`).
    ///   2. `cudaHostAlloc(Mapped)` + CPU stores -- no copy and capture-safe,
    ///      but every rope kernel then reads the value over PCIe: measured
    ///      26.9 -> 12.2 tok/s on the 27B.
    /// What works is a plain DEVICE buffer written by a 1-thread kernel on the
    /// launch stream (`k_rope64_setenv`), which captures like any other node and
    /// costs one launch per token instead of 448 retargets.
    fn ensureRopeDyn(self: *CudaBackend) ?*anyopaque {
        if (self.g_rope_dyn) |p| return p;
        const cu = &(self.cu orelse return null);
        var p: ?*anyopaque = null;
        const rc = cu.lib.Malloc(&p, @sizeOf([2]i32));
        if (rc != cudart.cudaSuccess or p == null) {
            self.ewLaunchFail("rope64_dpos alloc", rc);
            return null;
        }
        self.g_rope_dyn = p;
        return p;
    }

    /// Lazily allocate the device int that carries the decode `slot` to
    /// `q35_attn_decode_kernel_dyn`.  Returns null only if the allocation fails.
    fn ensureAttnDyn(self: *CudaBackend) ?*anyopaque {
        if (self.g_attn_dyn) |p| return p;
        const cu = &(self.cu orelse return null);
        var p: ?*anyopaque = null;
        const rc = cu.lib.Malloc(&p, @sizeOf(c_int));
        if (rc != cudart.cudaSuccess or p == null) {
            self.ewLaunchFail("attn_dyn alloc", rc);
            return null;
        }
        self.g_attn_dyn = p;
        return p;
    }

    /// Publish this token's `{pos, base}` into the device channel, but only when
    /// it actually changed (the engine calls `rope64` once per head, so the same
    /// token would otherwise publish 448 times).  Returns true if it launched the
    /// writer, i.e. if the caller must not assume the values are already there.
    fn ropeDynPublished(self: *CudaBackend, pos: u32, base: f32) bool {
        if (self.g_rope_dyn == null) return false;
        if (self.g_rope_dyn_pos == pos and self.g_rope_dyn_base == base) return false;
        const rc = Launch.rope64_setenv(self.g_rope_dyn, @bitCast(pos), base, self.launchStream());
        if (rc != cudart.cudaSuccess) {
            self.ewLaunchFail("rope64_setenv", rc);
            return false;
        }
        self.g_rope_dyn_pos = pos;
        self.g_rope_dyn_base = base;
        return true;
    }

    fn dRope64_body(ctx: *anyopaque, x: backend.Buf, pos: u32, base: f32) void {
        const self = self_(ctx);
        if (self.gOpGate(&self.g_op_seen.rope64)) return;
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
                // BOTH `pos` and `base` are variables of the decode step, not
                // constants: `pos` advances one per token, and `base` is the
                // context-dependent theta the engine derives per step (it
                // switches once the position passes the model's
                // `rope_freqs`/scaling threshold).  A captured node carries a
                // snapshot of both, so both are recorded here and retargeted
                // by `graphReplay` -- retargeting only `pos` rotates q/k with
                // the capture token's base, which is a wrong angle on every
                // replay (and if the swap happens to occur *during* capture it
                // is wrong from the first replayed token).
                // DEVICE-SIDE `pos` (see `k_rope64_dpos` in rope.cu).  The
                // kernel reads {pos, base} out of a two-int device buffer, so
                // the node's parameters never change and the decode graph can
                // replay 448 rope nodes on a 27B without a single
                // `cudaGraphExecKernelNodeSetParams`.  Measured on the 27B
                // IQ3_XXS: retargeting those 448 nodes per token was the
                // largest host cost in decode -- 17.5 `cuEventSynchronize` and
                // ~1.9k libcuda calls per token, with the GPU idle 64% of the
                // wall time waiting for the host to hand it the next token.
                //
                // The buffer is written with an ASYNC copy on the same stream,
                // never a blocking one: a blocking `cuMemcpy` inside an open
                // capture invalidates the graph (`cudaErrorStreamCaptureUnsupported`),
                // which is what `copyFromHost` has to guard against.
                if (self.ensureRopeDyn()) |dyn| {
                    // Publish {pos, base} once per token (the engine calls this
                    // once per head).  The writer is an ordinary kernel on the
                    // launch stream, so it is captured and replayed with the
                    // rest of the step and never needs re-parameterising.
                    _ = self.ropeDynPublished(pos, base);
                    const rc = Launch.rope64_dpos(px, 1, @intCast(n), dyn, self.launchStream());
                    if (rc == cudart.cudaSuccess) {
                        self.stat_rope_devices += 1;
                        if (self.capturing()) self.graphNoteRope(pos, base);
                        return;
                    }
                    if (self.capturing()) self.graphAbandon("rope64_dpos launch failed");
                    self.ewLaunchFail("rope64_dpos", rc);
                }
                const rc = Launch.rope64(px, 1, @intCast(n), @bitCast(pos), base, self.launchStream());
                if (rc == cudart.cudaSuccess) {
                    self.stat_rope_devices += 1;
                    if (self.capturing()) self.graphNoteRope(pos, base);
                    return;
                }
                if (self.capturing()) self.graphAbandon("rope64 launch failed");
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

    /// Row-batched `rope64` for prefill: ONE launch for `rows` tokens x `heads`
    /// heads instead of `rows * heads` launches.  The engine's prefill loop used
    /// to issue 28 per-token calls per full-attention layer (24 q heads + 4 kv
    /// heads); `k_rope64_rows` indexes rows and heads off a single flat block
    /// id and rotates row r at `pos + r`, so row r is the same arithmetic as the
    /// single-row call it replaces.
    ///
    /// NEVER used during a decode capture: this slot would add a node the graph
    /// cannot retarget (it matches `q35_func_rope64` node-by-node), so it bails
    /// out to the per-row loop when a capture is live.  Prefill does not capture.
    fn dRope64Rows_body(ctx: *anyopaque, x: backend.Buf, pos: u32, base: f32, rows_in: usize, heads_in: usize, hd_in: usize) void {
        const self = self_(ctx);
        if (self.gOpGate(&self.g_op_seen.rope64_rows)) return;
        const row_len = heads_in * hd_in;
        if (!ropeOnDevice() or self.capturing() or self.g_state == .live or
            heads_in == 0 or hd_in < 64 or rows_in == 0 or
            row_len == 0 or x.len != rows_in * row_len or self.cu == null)
        {
            cpu_mod.CpuBackend.vtable.rope64Rows(@ptrCast(&self.cpu), x, pos, base, rows_in, heads_in, hd_in);
            return;
        }
        if (dBufPtr(x)) |px| {
            const rc = Launch.rope64_rows(px, @intCast(rows_in), @intCast(heads_in), @intCast(hd_in), @bitCast(pos), base, self.launchStream());
            if (rc == cudart.cudaSuccess) return;
            self.ewLaunchFail("rope64_rows", rc);
        }
        cpu_mod.CpuBackend.vtable.rope64Rows(@ptrCast(&self.cpu), x, pos, base, rows_in, heads_in, hd_in);
    }

    fn dConvStep_body(ctx: *anyopaque, conv: backend.Buf, y: backend.Buf, wconv: backend.Buf, k: usize) void {
        const self = self_(ctx);
        if (self.gOpGate(&self.g_op_seen.conv_step)) return;
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
                    if (dBufPtr(wconv)) |pw| {
                        const rc = Launch.conv_step(pc, py, pw, @intCast(y.len), self.launchStream());
                        if (rc == cudart.cudaSuccess) return;
                        self.ewLaunchFail("conv_step", rc);
                    } else return;
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
        var made: [8]?*anyopaque = .{ null, null, null, null, null, null, null, null };
        const names = [_][:0]const u8{ "ev_bk_a", "ev_bk_b", "ev_pre_a", "ev_pre_b", "ev_arg_a", "ev_arg_b", "ev_tok_end", "ev_prev_tok_end" };
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
        self.ev_arg_a = made[4];
        self.ev_arg_b = made[5];
        self.ev_tok_end = made[6];
        self.ev_prev_tok_end = made[7];
        self.ev_created = true;
        return true;
    }

    fn freeEvents(self: *CudaBackend) void {
        if (!self.ev_created) return;
        const lib = if (self.cu) |*cu| &cu.lib else null;
        if (lib) |l| {
            for ([_]?*anyopaque{ self.ev_bk_a, self.ev_bk_b, self.ev_pre_a, self.ev_pre_b, self.ev_arg_a, self.ev_arg_b, self.ev_tok_end, self.ev_prev_tok_end }) |p| {
                if (p) |q| _ = l.EventDestroy(q);
            }
        }
        self.ev_bk_a = null;
        self.ev_bk_b = null;
        self.ev_pre_a = null;
        self.ev_pre_b = null;
        self.ev_arg_a = null;
        self.ev_arg_b = null;
        self.ev_tok_end = null;
        self.ev_prev_tok_end = null;
        self.ev_created = false;
    }

    /// True when the caller should record events at all.  Deliberately reads
    /// the env var as well: `profReset` is the only intended setter, but a
    /// second line of defence keeps a stray `ev_on = true` from turning into a
    /// default-path sync on a build where PROF is off.
    fn gpuTimingOn(self: *const CudaBackend) bool {
        return self.ev_on and self.cu != null;
    }

    /// Start timing one `backend.Exec` slot; returns the clock origin to hand
    /// to `slotExit`, or null when profiling is off (the returned value is then
    /// never used, so the hot path is one predictable branch and no clock read).
    ///
    /// Nesting is tracked by `slot_depth` only so `slot_total` does not
    /// double-count if a forwarder ever calls another forwarder; today the
    /// forwarders call `CpuBackend.vtable.*` directly, which bypasses these.
    inline fn slotEnter(self: *CudaBackend) ?Io.Clock.Timestamp {
        // `$QWEN35_PROF_GPU` opens one device-timing pair for the outermost slot.
        // The slot id is NOT known here -- every forwarder calls `slotEnter()`
        // with no argument and hands the slot to `slotExit` -- so `gpuEvEnter`
        // takes only the pair index and `slotExit` attributes it.  A field, not a
        // returned value: the 32 forwarders pass only the host `t0` back.
        self.gpu_ev_open = self.gpuEvEnter();
        if (!self.ev_on) {
            self.slot_depth += 1;
            return null;
        }
        self.slot_depth += 1;
        return Io.Clock.Timestamp.now(self.io, .awake);
    }

    /// Close a slot opened by `slotEnter` and add the interval to its counter.
    /// `t0 == null` (profiling off) is a no-op.
    inline fn slotExit(self: *CudaBackend, s: Slot, t0: ?Io.Clock.Timestamp) void {
        // Close the device-timing pair first: the closing record has to land on
        // the same stream the launch used, which `launchStream()` still reports
        // correctly here (the capture state has not changed during the slot).
        self.gpuEvExit(self.gpu_ev_open, s);
        self.gpu_ev_open = null;
        // INVALIDATE THE QUANTISED-ACTIVATION CACHE for the slots that can
        // WRITE an activation buffer.  Matvecs only READ their `x`; the engine
        // reuses a single `x1` across phases (`rmsnorm(x1) -> 4 matvecs ->
        // rmsnorm(x1, post_attention_norm) -> 2 matvecs ...`), so a cached hit
        // surviving a norm would hand the second phase the FIRST phase's
        // quantised activation.
        //
        // The list is an ALLOW-list of writers, not "everything but matvec".
        // Clearing on every other slot threw the cache away several times per
        // phase, which put the quantize back on the critical path: measured
        // 176 hits / 326 misses per token and four distinct activations being
        // re-quantised 180/80/80/60 times.  The slots NOT listed here
        // (`rope64`, `attn_decode`, `gdn_step`, `conv_step`, `gdn_gamma`,
        // `scale_vec`, `zero_buf`, `deinterleave_qg`, `silu`, `softmax`,
        // `argmax`, `sample`, the allocators) operate on the projection
        // buffers (`qkv`, `z`, `ab`, `mlp_h`, the KV cache) -- never on the
        // hidden-state buffer a matvec is about to consume -- so keeping the
        // cache across them is sound.  Anything added to the engine that writes
        // the ACTIVATION of a pending matvec must be added here too.
        //
        // This must run before the `t0 orelse return` below because the
        // profiling toggle is off by default.
        switch (s) {
            .rmsnorm, .l2norm, .add_resid, .copy_range, .copy_from_host, .swiglu, .mul_silu, .dequant_row, .load_linear, .const_buf => self.act_q.n = 0,
            else => {},
        }
        const start = t0 orelse return;
        const ns = durNs(start, self.io);
        const i = @intFromEnum(s);
        self.stat_ns_slot[i] += ns;
        self.stat_n_slot[i] += 1;
        // Only the outermost interval counts toward the window total, so the
        // "accounted" number stays comparable with `stat_ns_prefill_wall`.
        self.slot_depth -= 1;
        if (self.slot_depth == 0) self.stat_ns_slot_total += ns;
    }

    /// Create the event pool for `$QWEN35_PROF_GPU` once.  Events are created
    /// with timing ENABLED (flags 0): `EventElapsedTime` needs it, and
    /// `EventDisableTiming` pairs return `cudaErrorInvalidResourceHandle`.
    fn gpuEvEnsure(self: *CudaBackend) bool {
        if (self.gpu_ev_ready) return true;
        const lib = if (self.cu) |*cu| &cu.lib else return false;
        var made: usize = 0;
        for (0..gpu_ev_capacity) |i| {
            var a: ?*anyopaque = null;
            var b: ?*anyopaque = null;
            if (lib.EventCreate(&a, 0) != cudart.cudaSuccess or a == null) break;
            self.ev_pair_a[i] = a;
            made += 1;
            if (lib.EventCreate(&b, 0) != cudart.cudaSuccess or b == null) break;
            self.ev_pair_b[i] = b;
            made += 1;
            self.gpu_ev_ready = true;
        }
        if (!self.gpu_ev_ready) {
            // Bounded cleanup: destroy what was created so a partial pool cannot
            // leak, and say what we got rather than silently measuring nothing.
            for (0..gpu_ev_capacity) |i| {
                if (self.ev_pair_a[i]) |p| { _ = lib.EventDestroy(p); self.ev_pair_a[i] = null; }
                if (self.ev_pair_b[i]) |p| { _ = lib.EventDestroy(p); self.ev_pair_b[i] = null; }
            }
            std.debug.print("cuda backend: $QWEN35_PROF_GPU event pool disabled (created {d} of {d} events)\n",
                .{ made, gpu_ev_capacity * 2 });
            return false;
        }
        return true;
    }

    /// Read every complete pair of the token in flight and add each interval to
    /// its slot.  Called at the argmax drain, where the blocking D2H has already
    /// completed the stream, so `EventElapsedTime` on any pair of this token
    /// returns without waiting: the read costs one host call per kernel, not one
    /// sync per kernel.
    fn gpuEvReadToken(self: *CudaBackend) void {
        const lib = if (self.cu) |*cu| &cu.lib else return;
        var sum: u64 = 0;
        for (0..self.gpu_ev_used) |i| {
            const a = self.ev_pair_a[i];
            const b = self.ev_pair_b[i];
            if (a == null or b == null) continue;
            var ms: f32 = 0;
            if (lib.EventElapsedTime(&ms, a, b) == cudart.cudaSuccess and ms >= 0) {
                const ns: u64 = @intFromFloat(@as(f64, ms) * 1_000_000.0);
                const s = self.gpu_ev_slot[i];
                self.stat_gpu_ns_slot[s] += ns;
                self.stat_gpu_n_slot[s] += 1;
                sum += ns;
            } else self.stat_gpu_ev_err += 1;
            // Reset for reuse: events stay allocated, only their recorded
            // position is discarded by the next token's record.
            self.gpu_ev_slot[i] = 0;
        }
        self.stat_gpu_ns_token_sum += sum;
        self.stat_gpu_n_token += 1;
        self.gpu_ev_used = 0;
    }

    /// Open one GPU timing pair for a slot.  Only the outermost slot gets a pair
    /// (`slot_depth == 0`), so nested slots cannot double-count device time.
    /// The slot id is filled in by `slotExit` (which is the only place that knows
    /// it), so this takes none.
    /// Returns the pair index, or null when timing is off, capture is open (an
    /// event record inside a graph is not what we want to measure), the pool is
    /// exhausted (counted as overflow), or event creation failed.
    /// TEMP(ablation): true when slot `s`'s launch is to be SKIPPED.  Model
    /// output is garbage while any compute slot is ablated; the only consumer of
    /// this is the tok/s delta, which attributes a slot's device cost.
    inline fn ablated(self: *CudaBackend, s: Slot) bool {
        return (self.ablate_bits >> @intFromEnum(s)) & 1 != 0;
    }

    /// Parse `QWEN35_ABLATE="slot,slot,..."` into `ablate_bits` (names are the
    /// `PROF-SLOT` names).  Unknown names are counted and reported once.
    fn ablateInit(self: *CudaBackend) void {
        // Value, not mere presence: `FOO=0` must mean OFF, like every other
        // switch in the table (`QWEN35_CUDA_GRAPH=0` is documented as "disable").
        self.sample_cpu_ab = if (cudart.getenv("QWEN35_CUDA_SAMPLE_CPU")) |v|
            v.len > 0 and !std.mem.eql(u8, v, "0")
        else
            false;
        const raw = cudart.getenv("QWEN35_ABLATE") orelse return;
        var it = std.mem.splitScalar(u8, std.mem.trim(u8, raw, " \t\r\n"), ',');
        var n_ablated: usize = 0;
        while (it.next()) |tok| {
            const t = std.mem.trim(u8, tok, " \t");
            if (t.len == 0) continue;
            const s = std.meta.stringToEnum(Slot, t) orelse continue;
            self.ablate_bits |= @as(u32, 1) << @intCast(@intFromEnum(s));
            n_ablated += 1;
        }
        if (n_ablated != 0) std.debug.print("cuda backend: $QWEN35_ABLATE skips {d} slot(s) (model output is garbage; tok/s delta is the measurement)\n", .{n_ablated});
    }

    inline fn gpuEvEnter(self: *CudaBackend) ?usize {
        if (!self.gpu_ev_on) return null;
        if (self.capturing()) return null;
        // A live decode graph replays one graph per token and suppresses the
        // host-side launches (`gOpGate`), so there is nothing per slot to record
        // -- the pairs would measure empty intervals.  The graph path reports
        // its own totals via `stat_ns_token_dev`; disable the pool instead of
        // printing a table of zeros.
        if (self.g_state == .live) return null;
        if (self.slot_depth != 0) return null;
        if (!self.gpuEvEnsure()) {
            self.gpu_ev_on = false;
            return null;
        }
        if (self.gpu_ev_used >= gpu_ev_capacity) {
            self.stat_gpu_overflow += 1;
            return null;
        }
        const i = self.gpu_ev_used;
        self.gpu_ev_used += 1;
        self.gpu_ev_slot[i] = 255; // set by slotExit once the slot is known
        const lib = &self.cu.?.lib;
        if (lib.EventRecord(self.ev_pair_a[i], self.launchStream()) != cudart.cudaSuccess) {
            self.stat_gpu_ev_err += 1;
            self.gpu_ev_used -= 1;
            return null;
        }
        return i;
    }

    /// Close a pair opened by `gpuEvEnter`, attributing it to slot `s`; the pair
    /// is read at the token drain.
    inline fn gpuEvExit(self: *CudaBackend, i: ?usize, s: Slot) void {
        const idx = i orelse return;
        const lib = &self.cu.?.lib;
        self.gpu_ev_slot[idx] = @intFromEnum(s);
        if (lib.EventRecord(self.ev_pair_b[idx], self.launchStream()) != cudart.cudaSuccess) {
            self.stat_gpu_ev_err += 1;
            // A pair whose closing record never landed cannot be read; free the
            // slot by moving the last live pair into it.
            const last = self.gpu_ev_used - 1;
            self.ev_pair_a[idx] = self.ev_pair_a[last];
            self.ev_pair_b[idx] = self.ev_pair_b[last];
            self.gpu_ev_slot[idx] = self.gpu_ev_slot[last];
            self.gpu_ev_used = last;
        }
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
            const ti: usize = @intCast(self.bk_ttype);
            if (ti < self.stat_ns_bk_by_type.len) {
                self.stat_ns_bk_by_type[ti] +|= ns;
                self.stat_n_bk_by_type[ti] +|= 1;
            }
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

    /// Map a `Q35_TT_*` code (api.h) back to a printable name for `CUDA-BKT`.
    /// The CUDA type codes are NOT contiguous and NOT in GGUF order (see the
    /// table in api.h), so a positional table here would mislabel every type --
    /// which is exactly the mistake AGENTS.md warns about for microbenchmarks.
    fn bkTypeName(code: u8) []const u8 {
        return switch (code) {
            0 => "f32", 1 => "f16", 2 => "q4_0", 3 => "q4_1",
            6 => "q5_0", 7 => "q5_1", 8 => "q8_0", 9 => "q8_1",
            10 => "q2_k", 11 => "q3_k", 12 => "q4_k", 13 => "q5_k", 14 => "q6_k",
            15 => "q8_k", 16 => "iq2_xxs", 17 => "iq2_xs", 18 => "iq3_xxs",
            19 => "iq1_s", 20 => "iq4_nl", 21 => "iq3_s", 22 => "iq2_s",
            23 => "iq4_xs", 29 => "iq1_m", 30 => "bf16",
            else => "?",
        };
    }

    /// `$QWEN35_PROF`: one `CUDA-M [...]` line with what actually ran on the device,
    /// plus a per-type breakdown, so "did this model mix backends?" is answerable
    /// from the log.  Uploaded MiB is the resident weight footprint; bufs_MiB is the
    /// reusable activation/output scratch.
    fn reportStats(self: *CudaBackend, when: []const u8) void {
        if (self.cu == null) return;
        if (self.stat_n_batchk_gpu > 0) {
            // Per-type device cost of the batchK calls.  Without this the only
            // number available is the average over all 500-odd launches in a
            // 525-token prefill, which is dominated by whichever type is
            // biggest and hides whether ANY of them is running at the
            // standalone-kernel rate.  Emitted only when the events were on.
            std.debug.print("CUDA-BKT [{s}]", .{when});
            var seen: usize = 0;
            var order: [64]usize = undefined;
            var ti: usize = 0;
            while (ti < 64) : (ti += 1) {
                if (self.stat_n_bk_by_type[ti] != 0) {
                    order[seen] = ti;
                    seen += 1;
                }
            }
            std.mem.sort(usize, order[0..seen], self, struct {
                fn lt(ctx: *CudaBackend, a: usize, b: usize) bool {
                    return ctx.stat_ns_bk_by_type[a] > ctx.stat_ns_bk_by_type[b];
                }
            }.lt);
            for (order[0..seen]) |ix| {
                const ns = self.stat_ns_bk_by_type[ix];
                const n = self.stat_n_bk_by_type[ix];
                std.debug.print(" {s}({d})={d}us/{d}={d}us", .{
                    bkTypeName(@intCast(ix)), ix,
                    ns / 1000, n, ns / 1000 / n,
                });
            }
            std.debug.print("\n", .{});
        }
        // `d_xqm` is only non-zero on the opt-in dp4a path, so it usually adds
        // nothing here; counting it keeps the number honest when it does.
        const buf_bytes = self.d_act_bytes + self.d_xq_bytes + self.d_xqm_bytes + self.d_out_bytes + self.d_deq_bytes;
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

        self.reportSlotStats();
    }

    /// `PROF-SLOT`: one line per `backend.Exec` slot, largest total first, plus
    /// the reconciliation against the prefill window.
    ///
    /// `slot_sum_ms` adds the host wall clock of every timed slot call,
    /// `unaccounted_ms = pre_wall_ms - slot_sum_ms` is the prefill loop itself
    /// (engine arithmetic, the sampler, the batchK chunk scheduler, the token
    /// loop bookkeeping), i.e. exactly the part that never crosses the `Exec`
    /// vtable.  A large unaccounted number means a *path*, not a slot, is still
    /// missing from the table.
    ///
    /// Fields per line: `name calls total_ms pct sync`.  `sync=1` marks the
    /// slots whose forwarder can drain the stream, so their ms MAY include
    /// device execution; `sync=0` slots measure submission only and their
    /// queued GPU work surfaces inside the later draining slot (or in
    /// `bk_gpu_ms` for the batchK path).  Note that `sync` is a static property
    /// of the slot, NOT evidence that this run blocked: `matvec` is flagged
    /// because its forwarder contains `downloadOut`, but that path only runs
    /// for a host output buffer, which the decode path never passes -- see the
    /// note on `slotSyncs`.  So `sync=1` on `matvec` does not by itself mean
    /// host waiting, and `total_ms` there is launch cost, not a stall.  That is the one caveat needed to read the table as
    /// "where did the wall clock go": submission time is real host time and is
    /// never double-counted, but it is not the same question as GPU busy time.
    fn reportSlotStats(self: *CudaBackend) void {
        if (self.cu == null) return;
        // Sort indices by total, descending.  N=28, so an insertion sort into a
        // fixed array beats anything fancier and allocates nothing.
        var order: [slot_count]u8 = undefined;
        var n: usize = 0;
        for (0..slot_count) |i| {
            if (self.stat_n_slot[i] == 0) continue;
            order[n] = @intCast(i);
            n += 1;
        }
        var i: usize = 1;
        while (i < n) : (i += 1) {
            const cur = order[i];
            const cur_ns = self.stat_ns_slot[cur];
            var j: usize = i;
            while (j > 0 and self.stat_ns_slot[order[j - 1]] < cur_ns) : (j -= 1) {
                order[j] = order[j - 1];
            }
            order[j] = cur;
        }

        const wall_ns = self.stat_ns_prefill_wall;
        const sum_ns = self.stat_ns_slot_total;
        std.debug.print(
            "PROF-SLOT window_ms={d} slot_sum_ms={d} unaccounted_ms={d} accounted_pct={d}.{d}% slots={d} note=\"sync=0 slots are launch-only; their GPU time is counted by the draining slot\"\n",
            .{
                wall_ns / 1_000_000,
                sum_ns / 1_000_000,
                (wall_ns -| sum_ns) / 1_000_000,
                // Fraction of the prefill window covered by timed slots.  Shown
                // as a percentage so a reader does not have to divide two ms
                // numbers, and so a *regression* in coverage (a new engine path
                // that skips the vtable) shows up as a falling number here.
                if (wall_ns == 0) 0 else (sum_ns * 1000 / wall_ns) / 10,
                if (wall_ns == 0) 0 else (sum_ns * 1000 / wall_ns) % 10,
                n,
            },
        );
        for (order[0..n]) |idx| {
            const s: Slot = @enumFromInt(idx);
            const ns = self.stat_ns_slot[idx];
            const pct: u64 = if (sum_ns == 0) 0 else ns * 1000 / sum_ns;
            std.debug.print(
                "PROF-SLOT {s:<15} calls={d:<7} total_ms={d:<6} pct={d}.{d}% sync={d}\n",
                .{
                    slotName(s),
                    self.stat_n_slot[idx],
                    ns / 1_000_000,
                    pct / 10,
                    pct % 10,
                    @intFromBool(slotSyncs(s)),
                },
            );
        }
        self.reportCopyStats();
        self.reportGpuSlotStats();
        self.reportArgmaxDrain();
    }

    /// `PROF-GPU`: the same slot table as `PROF-SLOT`, but measured by the device.
    ///
    /// `dev_us` is the interval the GPU recorded between the slot's opening and
    /// closing event records.  Because every record sits on the stream the launch
    /// goes to, the interval includes device work queued ahead of the slot -- so
    /// the per-slot numbers of a token SUM to that token's device time, and the
    /// sum is printed against `stat_ns_token_dev` (the independent token-end pair)
    /// as the reconciliation.  The difference between the two is the device idle
    /// the engine left between slots: the launch-bound signature.
    fn reportGpuSlotStats(self: *CudaBackend) void {
        const divCeil = struct {
            fn f(a: u64, b: u64) u64 {
                return if (b == 0) 0 else (a + b - 1) / b;
            }
        }.f;
        if (!self.gpu_ev_on) return;
        const tokens: usize = if (self.stat_gpu_n_token == 0) 1 else self.stat_gpu_n_token;
        var order: [slot_count]u8 = undefined;
        var n: usize = 0;
        for (0..slot_count) |i| {
            if (self.stat_gpu_n_slot[i] == 0) continue;
            order[n] = @intCast(i);
            n += 1;
        }
        var i: usize = 1;
        while (i < n) : (i += 1) {
            const cur = order[i];
            const cur_ns = self.stat_gpu_ns_slot[cur];
            var j: usize = i;
            while (j > 0 and self.stat_gpu_ns_slot[order[j - 1]] < cur_ns) : (j -= 1) {
                order[j] = order[j - 1];
            }
            order[j] = cur;
        }
        const sum_ns = self.stat_gpu_ns_token_sum;
        const tok_ns = self.stat_ns_token_dev;
        std.debug.print(
            "PROF-GPU tokens={d} gpu_sum_ms={d} tok_dev_ms={d} idle_ms={d} overflow={d} ev_err={d} note=\"dev_us = device interval per slot, per token\"\n",
            .{
                self.stat_gpu_n_token,
                sum_ns / 1_000_000,
                tok_ns / 1_000_000,
                (tok_ns -| sum_ns) / 1_000_000,
                self.stat_gpu_overflow,
                self.stat_gpu_ev_err,
            },
        );
        for (order[0..n]) |idx| {
            const s: Slot = @enumFromInt(idx);
            const ns = self.stat_gpu_ns_slot[idx];
            const calls = self.stat_gpu_n_slot[idx];
            std.debug.print(
                "PROF-GPU {s:<15} calls/tok={d:<5} dev_us/tok={d:<8} avg_us/call={d:<6} pct={d}.{d}%\n",
                .{
                    slotName(s),
                    divCeil(calls, tokens),
                    divCeil(ns, tokens * 1_000),
                    divCeil(ns, calls * 1_000),
                    if (sum_ns == 0) 0 else (ns * 1000 / sum_ns) / 10,
                    if (sum_ns == 0) 0 else (ns * 1000 / sum_ns) % 10,
                },
            );
        }
    }

    /// `PROF-DRAIN`: the per-token argmax drain split into the device's share and
    /// the host's share.
    ///
    /// `PROF-SLOT argmax` gives one number for the whole slot, but that number
    /// mixes two very different costs: the device finishing the queue the token
    /// already issued (unavoidable compute that would happen anyway) and the
    /// host sitting in a blocking D2H copy (pure lost overlap).  This line
    /// separates them, because a plan that shrinks one does nothing for the
    /// other -- e.g. a CUDA graph attacks submission overhead, which shows up
    /// here as device-side tail, not as the copy.
    fn reportArgmaxDrain(self: *CudaBackend) void {
        if (self.stat_n_argmax_timed == 0) return;
        const n = self.stat_n_argmax_timed;
        const host_ns = self.stat_ns_argmax_host;
        const dev_ns = self.stat_ns_argmax_gpu;
        // Per call, in microseconds, so the numbers are comparable across runs
        // with different token counts.
        const host_us = host_ns / n / 1000;
        const dev_us = dev_ns / n / 1000;
        // Device share of what the host waited for.  <100% means the host paid
        // for more than the device was actually executing (submission, driver,
        // copy latency); that gap is what a graph replay can reclaim.
        const dev_pct: u64 = if (host_ns == 0) 0 else (dev_ns * 100) / host_ns;
        // Whole-token device time: `ev_prev_tok_end .. ev_tok_end` spans every
        // kernel of one decode token.  Compared against the per-token host wall
        // clock this says whether the host or the device is the ceiling.
        const tok_dev_us = if (self.stat_n_token_dev == 0) 0 else self.stat_ns_token_dev / self.stat_n_token_dev / 1000;
        std.debug.print(
            "PROF-DRAIN argmax calls={d} host_us={d} dev_us={d} host_ms={d} dev_ms={d} dev_pct={d}% tok_dev_us={d} note=\"host_us is the blocking copy wait; dev_us is the argmax kernel alone; tok_dev_us is ALL device work for one token\"\n",
            .{ n, host_us, dev_us, host_ns / 1_000_000, dev_ns / 1_000_000, dev_pct, tok_dev_us },
        );
    }

    /// `PROF-COPY`: `copyRange` narrowed down by element count.
    ///
    /// `PROF-SLOT` gives one line for the whole slot; this says which width is
    /// paying for it.  Printed after the slot table because it is a drill-down,
    /// and only when at least one call happened (a run that never copies -- the
    /// all-quantized-embedding models -- prints nothing).
    ///
    /// Fields: `elems<= bound: calls= total_ms= bytes_each=`.  `bytes_each` is
    /// the per-call payload, which is what makes a 512-B-per-call family
    /// visible as such instead of being averaged with the embedding row.
    fn reportCopyStats(self: *CudaBackend) void {
        if (self.cu == null) return;
        var total: u32 = 0;
        for (self.stat_copy_bins) |b| total += b.n;
        if (total == 0) return;
        for (self.stat_copy_bins, 0..) |b, i| {
            if (b.n == 0) continue;
            const bound = if (i < copy_bin_bounds.len) copy_bin_bounds[i] else 0;
            std.debug.print(
                "PROF-COPY elems<={d:<5} calls={d:<7} total_ms={d:<6} bytes_each={d}\n",
                .{
                    // 0 marks the overflow bucket ("bigger than the last bound").
                    bound,
                    b.n,
                    b.ns / 1_000_000,
                    b.elems / b.n * @sizeOf(f32),
                },
            );
        }
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
        const self = self_(ctx);
        self.closePrefillWindow();
    }

    /// Open a decode step: the one place a CUDA graph can be recorded.
    ///
    /// The first step after a successful warm-up records; every later step
    /// replays.  A step that cannot be captured or replayed simply runs as
    /// direct launches -- `stepEnd` then does nothing, because no capture is
    /// open and every `Launch.*` in the step already used the default stream.
    fn dStepBegin(ctx: *anyopaque, pos: u32) void {
        const self = self_(ctx);
        if (!self.gpuReady()) return;
        // A new token writes new activations, possibly at the same addresses.
        self.actQClear();
        if (!graphWanted()) {
            // `$QWEN35_CUDA_GRAPH=0`: the direct path, untouched.
            self.g_state = .off;
            return;
        }
        if (cudart.getenv("QWEN35_GRTIME") != null) self.grMark(true);
        if (self.g_state == .live) {
            // `slot` starts at `pos` in this engine (`slot == kv.len`, advanced
            // once per step), but it is only a *seed*: `attn_decode`'s own slot is
            // recorded from the launch itself, so the replay retargets whichever
            // layers actually ran.  `pos` is the only value this hook needs.
            self.graphSyncFromDefault();
            if (!self.graphReplay(pos, pos)) {
                // Replay failed (e.g. a SetParams rejected).  The graph is torn
                // down so this and every later step run directly -- the numbers
                // stay right, only the speedup is lost.
                self.graphDestroyForFallback();
            }
                        return;
        }
        if (self.g_state == .warm and !self.g_failed) {
            // Allocate the rope {pos, base} channel BEFORE the capture opens.
            // A `cudaMalloc` inside an open stream capture is not legal, and the
            // first `rope64` of the capture step is where it would have landed:
            // measured as "rope64 launch failed: operation failed due to a
            // previous error during capture" with the graph abandoned from the
            // very first token.
            _ = self.ensureRopeDyn();
            if (self.graphBegin(pos, pos)) self.g_capturing_step = true;
        }
    }

    /// Close a decode step: if this step was captured, instantiate the graph so
    /// the NEXT step can replay it.
    fn dStepEnd(ctx: *anyopaque) void {
        if (cudart.getenv("QWEN35_QSTAT") != null) {
            const self0 = self_(ctx);
            std.debug.print("[qstat] hit={d} miss={d}\n", .{ self0.q_hit, self0.q_miss });
        }
        const self = self_(ctx);
        if (!self.g_capturing_step) {
            // A REPLAYED step lands here.  Its launches went to `g_stream`, and
            // nothing else orders that private stream against the legacy default
            // one: `sample`/`argmax` reads logits on the default stream and the
            // next token's embedding is written there, so without this barrier
            // the host reads whatever logits happen to be in the buffer --
            // measured: the very first replayed token came out as garbage while
            // the capture step (identical arguments, same stream) was correct.
            // This is the mirror of `graphSyncFromDefault` in `dStepBegin`, and
            // a replay step is precisely when it is needed, so it cannot live
            // behind the `g_capturing_step` early return above.
            if (cudart.getenv("QWEN35_GRTIME") != null) {
                self.grMark(false);
                std.debug.print("[grt] n={d} gpu_ms={d:.3} avg={d:.3} tok/s={d:.2}\n",
                    .{ self.gr_n, self.gr_ms, if (self.gr_n > 0) self.gr_ms / @as(f64, @floatFromInt(self.gr_n)) else 0,
                        if (self.gr_ms > 0) @as(f64, @floatFromInt(self.gr_n)) * 1000.0 / self.gr_ms else 0 });
            }
            if (self.g_state == .live) self.graphSyncToDefault();
            // The graph's tail IS the sampler, so its result is only readable now
            // -- after the ordering barrier above, and outside any capture, which
            // is the only place a blocking D2H is legal.  This is the single
            // 64-byte-per-token crossing the device sampler exists to produce:
            // 4 bytes of token plus the advanced PRNG state.
            if (self.sample_pending) {
                if (self.sample_sampler) |sp| _ = self.collectSampledToken(sp);
            }
            return;
        }
        self.g_capturing_step = false;
        _ = self.graphEnd();
        // The captured step ran on `g_stream` (capture still executes).  Its
        // results -- the token's logits, and the KV row it appended -- have to be
        // ordered before the default stream samples them and before the next
        // token's embedding lands; that is exactly what `graphSyncToDefault`
        // does for a replayed step.  It must ALSO happen here, and the check is
        // `g_state` rather than "did we just replay": `graphEnd` flips the state
        // to `.live` when instantiation succeeds, so the one step that needs the
        // barrier most -- the capture step, the only one which executed on the
        // graph stream before any replay -- is the one where `g_state` is
        // already `.live` on entry to this function.
        if (self.g_state == .live) self.graphSyncToDefault();
        // The capture step executed too (capture still runs the kernels), so its
        // token has to be downloaded the same way a replayed step's is.
        if (self.sample_pending) {
            if (self.sample_sampler) |sp| _ = self.collectSampledToken(sp);
        }
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
        // Per-slot timers (QWEN35_PROF only; the counters stay zero when off).
        self.stat_ns_slot = @splat(0);
        self.stat_n_slot = @splat(0);
        self.stat_ns_slot_total = 0;
        self.slot_depth = 0;
        // Per-slot DEVICE time ($QWEN35_PROF_GPU).  The pool is created lazily
        // on the first timed slot, not here, so a run without the variable never
        // touches an event.
        self.stat_gpu_ns_slot = @splat(0);
        self.stat_gpu_n_slot = @splat(0);
        self.stat_gpu_overflow = 0;
        self.stat_gpu_ev_err = 0;
        self.stat_gpu_ns_token_sum = 0;
        self.stat_gpu_n_token = 0;
        self.gpu_ev_used = 0;
        self.gpu_ev_open = null;
        self.gpu_ev_on = cudart.getenv("QWEN35_PROF_GPU") != null and self.cu != null;
        if (self.gpu_ev_on) {
            std.debug.print("cuda backend: $QWEN35_PROF_GPU per-slot device timing on ({d} event pairs)\n", .{gpu_ev_capacity});
        }
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

    /// `Exec.profReport`: the CUDA backend's complete one-shot dump.  Reaches
    /// the same `reportStats("final")` as the public `CudaBackend.profReport`;
    /// the engine only ever sees the `Exec`, so the vtable path is the one the
    /// CLI uses.
    fn dProfReport(ctx: *anyopaque) void {
        const self = self_(ctx);
        self.profReport();
    }
// ==== slot timing wrappers ($QWEN35_PROF only) ==============================
//
// One thin forwarder per `backend.Exec` slot.  Each one opens a clock, calls
// the unmodified `*_body` implementation, and closes the clock on the way out
// (`defer` covers the early-return and error paths inside the bodies, which is
// why the bodies keep their own `return` statements untouched).
//
// Why wrap instead of instrumenting in place: the bodies have several
// `return`/`catch return` exits each; adding a `slotExit` at every one of them
// is where a profiling patch starts lying.  A wrapper cannot miss an exit.
//
// With `$QWEN35_PROF` unset, `slotEnter` returns null after one predictable
// branch and `slotExit` returns immediately, so the plain path costs a call
// and a not-taken branch -- no clock read, no counter write.

    /// The `loadLinear` slot, timed (`$QWEN35_PROF`).
    fn vLoadLinear(ctx: *anyopaque, name: []const u8) backend.LoadError!backend.LinearW {
        const self = self_(ctx);
        const t0 = self.slotEnter();
        defer self.slotExit(.load_linear, t0);
        return vLoadLinear_body(ctx, name);
    }

    /// The `dequantRow` slot, timed (`$QWEN35_PROF`).
    fn vDequantRow(ctx: *anyopaque, dstBuf: backend.Buf, w: gguf.QuantRow, row: usize) void {
        const self = self_(ctx);
        const t0 = self.slotEnter();
        defer self.slotExit(.dequant_row, t0);
        vDequantRow_body(ctx, dstBuf, w, row);
    }

    /// The `matvec` slot, timed (`$QWEN35_PROF`).
    fn vMatvec(ctx: *anyopaque, outBuf: backend.Buf, w: backend.LinearW, xBuf: backend.Buf) void {
        const self = self_(ctx);
        const t0 = self.slotEnter();
        defer self.slotExit(.matvec, t0);
        if (self.ablated(.matvec)) return ;
        vMatvec_body(ctx, outBuf, w, xBuf);
    }

    /// The `matvecBatchK` slot, timed (`$QWEN35_PROF`).
    fn vMatvecBatchK(ctx: *anyopaque, outBuf: backend.Buf, w: backend.LinearW, xKBuf: backend.Buf, k: usize) void {
        const self = self_(ctx);
        const t0 = self.slotEnter();
        defer self.slotExit(.matvec_batchk, t0);
        if (self.ablated(.matvec_batchk)) return ;
        vMatvecBatchK_body(ctx, outBuf, w, xKBuf, k);
    }

    /// The `allocScratch` slot, timed (`$QWEN35_PROF`).
    fn dAllocScratch(ctx: *anyopaque, max_cols: usize) anyerror!void {
        const self = self_(ctx);
        const t0 = self.slotEnter();
        defer self.slotExit(.alloc_scratch, t0);
        return dAllocScratch_body(ctx, max_cols);
    }

    /// The `allocScratchK` slot, timed (`$QWEN35_PROF`).
    fn dAllocScratchK(ctx: *anyopaque, max_cols: usize, k: usize) anyerror!void {
        const self = self_(ctx);
        const t0 = self.slotEnter();
        defer self.slotExit(.alloc_scratch_k, t0);
        return dAllocScratchK_body(ctx, max_cols, k);
    }

    /// The `allocBuf` slot, timed (`$QWEN35_PROF`).
    fn dAllocBuf(ctx: *anyopaque, len: usize) anyerror!backend.Buf {
        const self = self_(ctx);
        const t0 = self.slotEnter();
        defer self.slotExit(.alloc_buf, t0);
        return dAllocBuf_body(ctx, len);
    }

    /// The `scaleVec` slot, timed (`$QWEN35_PROF`).
    fn dScaleVec(ctx: *anyopaque, x: backend.Buf, s: f32) void {
        const self = self_(ctx);
        const t0 = self.slotEnter();
        defer self.slotExit(.scale_vec, t0);
        if (self.ablated(.scale_vec)) return ;
        dScaleVec_body(ctx, x, s);
    }

    /// The `addResid` slot, timed (`$QWEN35_PROF`).
    fn dAddResid(ctx: *anyopaque, dst: backend.Buf, src: backend.Buf) void {
        const self = self_(ctx);
        const t0 = self.slotEnter();
        defer self.slotExit(.add_resid, t0);
        if (self.ablated(.add_resid)) return ;
        dAddResid_body(ctx, dst, src);
    }

    /// The `copyRange` slot, timed (`$QWEN35_PROF`).
    fn dCopyRange(ctx: *anyopaque, dst: backend.Buf, src: backend.Buf) void {
        const self = self_(ctx);
        const t0 = self.slotEnter();
        // Per-width accounting, alongside the slot total.  The bucket comes from
        // `src.len` alone, so it is chosen on the way in and does not need the
        // route decision the body makes; two defers keep the body call in the
        // tail position so the fast path is unchanged when profiling is off.
        if (t0) |t| {
            const b = &self.stat_copy_bins[copyBinOf(src.len)];
            b.n += 1;
            b.elems += src.len;
            dCopyRange_body(ctx, dst, src);
            const ns = durNs(t, self.io);
            b.ns += ns;
            self.slotExit(.copy_range, t);
            return;
        }
        dCopyRange_body(ctx, dst, src);
    }

    /// KV append, counted in the same slot as `copyRange` (it IS a copy; the
    /// only difference is that the capture records its destination).  Kept
    /// apart from `dCopyRange` so the recording cannot be reached from the
    /// generic path, where a moving destination does not exist.
    fn dCopyRangeSlot(ctx: *anyopaque, dst: backend.Buf, src: backend.Buf, slot: usize) void {
        const self = self_(ctx);
        const t0 = self.slotEnter();
        if (t0) |t| {
            const b = &self.stat_copy_bins[copyBinOf(src.len)];
            b.n += 1;
            b.elems += src.len;
            dCopyRangeSlot_body(ctx, dst, src, slot);
            const ns = durNs(t, self.io);
            b.ns += ns;
            self.slotExit(.copy_range, t);
            return;
        }
        dCopyRangeSlot_body(ctx, dst, src, slot);
    }

    /// The `deinterleaveQG` slot, timed (`$QWEN35_PROF`).
    ///
    /// Every argument is a device region in the P2c engine (`qK`/`z`/`qkv` are
    /// all backend allocations), so this is the one launch that replaces
    /// `2*heads` `copyRange` launches.  Not a `cudaMemcpy`: the two halves of a
    /// head are interleaved with the *other* half, so no single linear copy
    /// can produce the dense layout -- hence the kernel.
    fn dDeinterleaveQG(ctx: *anyopaque, dst_q: backend.Buf, dst_gate: backend.Buf, src: backend.Buf, heads: usize, hd: usize) void {
        const self = self_(ctx);
        const t0 = self.slotEnter();
        defer self.slotExit(.deinterleave_qg, t0);
        if (self.ablated(.deinterleave_qg)) return ;
        if (self.cu != null) {
            if (dBufPtr(dst_q)) |pq| {
                if (dBufPtr(dst_gate)) |pg| {
                    if (dBufPtr(src)) |ps| {
                        const rc = Launch.deinterleave_qg(pq, pg, ps, @intCast(heads), @intCast(hd), self.launchStream());
                        if (rc == cudart.cudaSuccess) return;
                        self.ewLaunchFail("deinterleave_qg", rc);
                    }
                }
            }
        }
        // CPU fallback: either there is no device, or a pointer was not a
        // device region (host `Buf`s reach this vtable only in tests).
        cpu_mod.CpuBackend.vtable.deinterleaveQG(@ptrCast(&self.cpu), dst_q, dst_gate, src, heads, hd);
    }

    /// Chunk-wide `deinterleaveQG`: `tokens` rows in one launch.  Same move as
    /// `dDeinterleaveQG`, so this is exact; it exists because the per-row form
    /// cost 940 launches per 525-token prefill (1.2% of device time at 6 us).
    fn dDeinterleaveQGStrided(ctx: *anyopaque, dst_q: backend.Buf, dst_gate: backend.Buf,
                              src: backend.Buf, heads: usize, hd: usize, tokens: usize) void {
        const self = self_(ctx);
        const t0 = self.slotEnter();
        defer self.slotExit(.deinterleave_qg, t0);
        if (self.ablated(.deinterleave_qg)) return;
        const row_len = heads * hd;
        const shapes_ok = tokens != 0 and heads != 0 and hd != 0 and
            dst_q.len >= tokens * row_len and dst_gate.len >= tokens * row_len and
            src.len >= tokens * 2 * row_len;
        if (self.cu != null and shapes_ok) {
            if (dBufPtr(dst_q)) |pq| {
                if (dBufPtr(dst_gate)) |pg| {
                    if (dBufPtr(src)) |ps| {
                        const rc = Launch.deinterleave_qg_strided(pq, pg, ps,
                            @intCast(heads), @intCast(hd), @intCast(tokens), self.launchStream());
                        if (rc == cudart.cudaSuccess) return;
                        self.ewLaunchFail("deinterleave_qg_strided", rc);
                    }
                }
            }
        }
        cpu_mod.CpuBackend.vtable.deinterleaveQGStrided(@ptrCast(&self.cpu), dst_q, dst_gate, src, heads, hd, tokens);
    }

    /// The `zeroBuf` slot, timed (`$QWEN35_PROF`).
    fn dZeroBuf(ctx: *anyopaque, buf: backend.Buf) void {
        const self = self_(ctx);
        const t0 = self.slotEnter();
        defer self.slotExit(.zero_buf, t0);
        if (self.ablated(.zero_buf)) return ;
        dZeroBuf_body(ctx, buf);
    }

    /// The `copyFromHost` slot, timed (`$QWEN35_PROF`).
    fn dCopyFromHost(ctx: *anyopaque, dst: backend.Buf, src: []const f32) void {
        const self = self_(ctx);
        const t0 = self.slotEnter();
        defer self.slotExit(.copy_from_host, t0);
        if (self.ablated(.copy_from_host)) return ;
        dCopyFromHost_body(ctx, dst, src);
    }

    /// The `copyToHost` slot, timed (`$QWEN35_PROF`).
    fn dCopyToHost(ctx: *anyopaque, dst: []f32, src: backend.Buf) void {
        const self = self_(ctx);
        const t0 = self.slotEnter();
        defer self.slotExit(.copy_to_host, t0);
        dCopyToHost_body(ctx, dst, src);
    }

    /// The `rmsnorm` slot, timed (`$QWEN35_PROF`).
    fn dRmsnorm(ctx: *anyopaque, out: backend.Buf, x: backend.Buf, w: backend.Buf, eps: f32) void {
        const self = self_(ctx);
        const t0 = self.slotEnter();
        defer self.slotExit(.rmsnorm, t0);
        if (self.ablated(.rmsnorm)) return ;
        dRmsnorm_body(ctx, out, x, w, eps);
    }

    /// Row-batched twin of `dRmsnorm`.  Charged to the `rmsnorm` slot so the
    /// `PROF-SLOT` total for rmsnorm stays comparable across the change (only
    /// its `calls` count moves, which is the point of the merge).
    fn dRmsnormRows(ctx: *anyopaque, out: backend.Buf, x: backend.Buf, w: backend.Buf, rows_in: usize, eps: f32) void {
        const self = self_(ctx);
        const t0 = self.slotEnter();
        defer self.slotExit(.rmsnorm, t0);
        if (self.ablated(.rmsnorm)) return ;
        dRmsnormRows_body(ctx, out, x, w, rows_in, eps);
    }

    /// `l2norm(slot)`, timed (`$QWEN35_PROF`).
    fn dL2norm(ctx: *anyopaque, x: backend.Buf) void {
        const self = self_(ctx);
        const t0 = self.slotEnter();
        defer self.slotExit(.l2norm, t0);
        if (self.ablated(.l2norm)) return;
        dL2norm_body(ctx, x);
    }

    /// Row-batched twin of `dL2norm`; charged to the `l2norm` slot.
    fn dL2normRows(ctx: *anyopaque, x: backend.Buf, n: usize) void {
        const self = self_(ctx);
        const t0 = self.slotEnter();
        defer self.slotExit(.l2norm, t0);
        dL2normRows_body(ctx, x, n);
    }

    /// The `siluInPlace` slot, timed (`$QWEN35_PROF`).
    fn dSiluInPlace(ctx: *anyopaque, x: backend.Buf) void {
        const self = self_(ctx);
        const t0 = self.slotEnter();
        defer self.slotExit(.silu, t0);
        if (self.ablated(.silu)) return ;
        dSiluInPlace_body(ctx, x);
    }

    /// The `mulSilu` slot, timed (`$QWEN35_PROF`).
    fn dMulSilu(ctx: *anyopaque, out: backend.Buf, gate: backend.Buf) void {
        const self = self_(ctx);
        const t0 = self.slotEnter();
        defer self.slotExit(.mul_silu, t0);
        if (self.ablated(.mul_silu)) return ;
        dMulSilu_body(ctx, out, gate);
    }

    /// The `mulSigmoid` slot, timed (`$QWEN35_PROF`).
    fn dMulSigmoid(ctx: *anyopaque, out: backend.Buf, gate: backend.Buf) void {
        const self = self_(ctx);
        const t0 = self.slotEnter();
        defer self.slotExit(.mul_sigmoid, t0);
        if (self.ablated(.mul_sigmoid)) return ;
        dMulSigmoid_body(ctx, out, gate);
    }

    /// The `swiglu` slot, timed (`$QWEN35_PROF`).
    fn dSwiglu(ctx: *anyopaque, gate: backend.Buf, up: backend.Buf) void {
        const self = self_(ctx);
        const t0 = self.slotEnter();
        defer self.slotExit(.swiglu, t0);
        if (self.ablated(.swiglu)) return ;
        dSwiglu_body(ctx, gate, up);
    }

    /// The `softmax` slot, timed (`$QWEN35_PROF`).
    fn dSoftmax(ctx: *anyopaque, x: backend.Buf) void {
        const self = self_(ctx);
        const t0 = self.slotEnter();
        defer self.slotExit(.softmax, t0);
        if (self.ablated(.softmax)) return ;
        dSoftmax_body(ctx, x);
    }

    /// The `argmax` slot, timed (`$QWEN35_PROF`).
    fn dArgmax(ctx: *anyopaque, logits: backend.Buf) u32 {
        const self = self_(ctx);
        const t0 = self.slotEnter();
        defer self.slotExit(.argmax, t0);
        if (self.ablated(.argmax)) return 0;
        return dArgmax_body(ctx, logits);
    }

    /// The `sample` slot, timed (`$QWEN35_PROF`).
    fn dSample(ctx: *anyopaque, logits: backend.Buf, scratch: backend.Buf, sp: *sampler.Sampler) u32 {
        const self = self_(ctx);
        const t0 = self.slotEnter();
        defer self.slotExit(.sample, t0);
        if (self.ablated(.sample)) return 0;
        return dSample_body(ctx, logits, scratch, sp);
    }

    /// The `gdnStep` slot, timed (`$QWEN35_PROF`).
    fn dGdnStep(ctx: *anyopaque, state: backend.Buf, k: backend.Buf, v: backend.Buf,
                q: backend.Buf, km: backend.Buf, delta: backend.Buf, out: backend.Buf,
                gamma: f32, beta: f32, vd: usize) void {
        const self = self_(ctx);
        const t0 = self.slotEnter();
        defer self.slotExit(.gdn_step, t0);
        if (self.ablated(.gdn_step)) return ;
        dGdnStep_body(ctx, state, k, v, q, km, delta, out, gamma, beta, vd);
    }

    /// The batched `gdnStep` slot, timed under the SAME `.gdn_step` slot as the
    /// single-row one: it does the same work for all heads, so the `PROF-SLOT`
    /// line stays comparable across the change (calls drops by `heads`x, which
    /// is the check that the batching actually took effect).
    fn dGdnStepRows(ctx: *anyopaque, state: backend.Buf, k: backend.Buf, v: backend.Buf,
                    q: backend.Buf, km: backend.Buf, delta: backend.Buf, out: backend.Buf,
                    gamma: backend.Buf, beta: backend.Buf,
                    heads: usize, nk: usize, kd: usize, vd: usize) void {
        const self = self_(ctx);
        const t0 = self.slotEnter();
        defer self.slotExit(.gdn_step, t0);
        if (self.ablated(.gdn_step)) return ;
        dGdnStepRows_body(ctx, state, k, v, q, km, delta, out, gamma, beta, heads, nk, kd, vd);
    }

    /// Chunk-wide `gdnStepRows` (`VTable.gdnStepRowsChunk`): the same state
    /// recurrence for `rows` tokens in ONE launch.
    ///
    /// The steps are sequential -- `state` is carried -- so this does not add
    /// parallelism.  What it removes is the launch: the per-token loop issued
    /// 10,500 launches for a 525-token prefill (48 GDN layers), whose device
    /// time was 42 ms but whose wall clock was 641 ms.  Deleting the loop
    /// outright (measurement only) took prefill 299 -> 453 tok/s.
    fn dGdnStepRowsChunk_body(ctx: *anyopaque, state: backend.Buf, k: backend.Buf, v: backend.Buf,
                              q: backend.Buf, km: backend.Buf, delta: backend.Buf, out: backend.Buf,
                              gamma: backend.Buf, beta: backend.Buf,
                              heads: usize, nk: usize, kd: usize, vd: usize, rows: usize,
                              k_stride: usize, v_stride: usize, q_stride: usize) void {
        const self = self_(ctx);
        // Same shape gate as the per-row body, plus the strides the chunk form
        // needs.  The last row must be fully inside each buffer, so the length
        // check uses (rows-1)*stride + the row width -- not rows*stride, which
        // would reject a view whose tail exactly ends the buffer.
        const need = struct {
            fn n(r: usize, s: usize, w: usize) usize {
                return if (r == 0) 0 else (r - 1) * s + w;
            }
        }.n;
        if (vd == 128 and kd == 128 and heads != 0 and nk != 0 and nk <= heads and rows != 0 and
            gamma.len >= rows * heads and beta.len >= rows * heads and
            state.len >= heads * kd * vd and
            k.len >= need(rows, k_stride, nk * kd) and q.len >= need(rows, q_stride, nk * kd) and
            v.len >= need(rows, v_stride, heads * vd) and
            out.len >= rows * heads * vd and
            km.len >= rows * heads * vd and delta.len >= rows * heads * vd and
            self.cu != null)
        {
            if (dBufPtr(state)) |pstate| {
                if (dBufPtr(k)) |pk| {
                    if (dBufPtr(v)) |pv| {
                        if (dBufPtr(q)) |pq| {
                            if (dBufPtr(km)) |pkm| {
                                if (dBufPtr(delta)) |pdelta| {
                                    if (dBufPtr(out)) |pout| {
                                        if (dBufPtr(gamma)) |pgamma| {
                                            if (dBufPtr(beta)) |pbeta| {
                                                const rc = Launch.gdn_step_rows_chunk(
                                                    pstate, @intCast(kd * vd),
                                                    pk, @intCast(k_stride),
                                                    pv, @intCast(v_stride),
                                                    pq, @intCast(q_stride),
                                                    pkm, @intCast(heads * vd),
                                                    pdelta, @intCast(heads * vd),
                                                    pout, @intCast(heads * vd),
                                                    pgamma, @intCast(heads),
                                                    pbeta, @intCast(heads),
                                                    @intCast(heads), @intCast(nk),
                                                    @intCast(kd), @intCast(vd), @intCast(rows),
                                                    self.launchStream(),
                                                );
                                                if (rc == cudart.cudaSuccess) return;
                                                self.ewLaunchFail("gdn_step_rows_chunk", rc);
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
        // Fallback: the CPU oracle walks the rows one at a time, so a shape or
        // residency miss costs speed, never correctness.
        cpu_mod.CpuBackend.vtable.gdnStepRowsChunk(@ptrCast(&self.cpu), state, k, v, q, km, delta, out, gamma, beta, heads, nk, kd, vd, rows, k_stride, v_stride, q_stride);
    }

    fn dGdnStepRowsChunk(ctx: *anyopaque, state: backend.Buf, k: backend.Buf, v: backend.Buf,
                         q: backend.Buf, km: backend.Buf, delta: backend.Buf, out: backend.Buf,
                         gamma: backend.Buf, beta: backend.Buf,
                         heads: usize, nk: usize, kd: usize, vd: usize, rows: usize,
                         k_stride: usize, v_stride: usize, q_stride: usize) void {
        const self = self_(ctx);
        const t0 = self.slotEnter();
        defer self.slotExit(.gdn_step, t0);
        if (self.ablated(.gdn_step)) return ;
        dGdnStepRowsChunk_body(ctx, state, k, v, q, km, delta, out, gamma, beta, heads, nk, kd, vd, rows, k_stride, v_stride, q_stride);
    }

    /// The `gdnGamma` slot, timed (`$QWEN35_PROF`).
    fn dGdnGamma(ctx: *anyopaque, a_raw: backend.Buf, b_raw: backend.Buf, ssm_a: backend.Buf,
                 dt_bias: backend.Buf, gamma: backend.Buf, nv: usize) void {
        const self = self_(ctx);
        const t0 = self.slotEnter();
        defer self.slotExit(.gdn_gamma, t0);
        if (self.ablated(.gdn_gamma)) return ;
        dGdnGamma_body(ctx, a_raw, b_raw, ssm_a, dt_bias, gamma, nv);
    }

    /// The prefill-only GDN prelude, timed separately so a profile shows
    /// directly how many launches the per-token form used to cost.
    fn dGdnPrefillCore(ctx: *anyopaque, qkv: backend.Buf, conv: backend.Buf,
                       wconv: backend.Buf, ab: backend.Buf, ssm_a: backend.Buf,
                       dt_bias: backend.Buf, gamma: backend.Buf, tokens: usize,
                       conv_dim: usize, ab_a_offset: usize, ab_b_offset: usize,
                       ab_stride: usize, gamma_stride: usize, kq_dim: usize,
                       nv: usize, nk: usize, kd: usize, conv_k: usize, eps: f32) void {
        const self = self_(ctx);
        const t0 = self.slotEnter();
        defer self.slotExit(.gdn_prelude, t0);
        dGdnPrefillCore_body(ctx, qkv, conv, wconv, ab, ssm_a, dt_bias, gamma, tokens,
            conv_dim, ab_a_offset, ab_b_offset, ab_stride, gamma_stride, kq_dim,
            nv, nk, kd, conv_k, eps);
    }

    /// The `attnDecode` slot, timed (`$QWEN35_PROF`).
    fn dAttnDecode(ctx: *anyopaque, q: backend.Buf, gate: backend.Buf, kcache: backend.Buf,
                   vcache: backend.Buf, att: backend.Buf, gout: backend.Buf, slot: usize,
                   heads: usize, gqa: usize, hd: usize, kv_dim: usize, scale: f32) void {
        const self = self_(ctx);
        const t0 = self.slotEnter();
        defer self.slotExit(.attn_decode, t0);
        if (self.ablated(.attn_decode)) return ;
        dAttnDecode_body(ctx, q, gate, kcache, vcache, att, gout, slot, heads, gqa, hd, kv_dim, scale);
    }

    /// `attnChunk` -- the chunk-wide prefill attention, timed separately
    /// (`$QWEN35_PROF` shows `attn_chunk` beside `attn_decode`).
    ///
    /// Falls back to the per-row `dAttnDecode_body` whenever the device path is
    /// unavailable or the caller's buffers do not match the chunk contract, so
    /// a size violation degrades to known-good behaviour instead of a launch on
    /// bad pointers.  The CUDA kernel was gated bit-identical against exactly
    /// that per-row form (`/tmp/gate_attn_chunk.cu`).
    fn dAttnChunk(ctx: *anyopaque, q: backend.Buf, gate: backend.Buf, kcache: backend.Buf,
                  vcache: backend.Buf, att: backend.Buf, gout: backend.Buf, slot0: usize,
                  rows: usize, heads: usize, gqa: usize, hd: usize, kv_dim: usize,
                  scale: f32) void {
        const self = self_(ctx);
        const t0 = self.slotEnter();
        defer self.slotExit(.attn_chunk, t0);
        if (self.ablated(.attn_chunk)) return;

        const row_len = heads * hd;
        const n_tot = slot0 + rows;
        // The gate kernel that the per-row path runs separately is folded into
        // the chunk kernel, so this path is only taken when EVERY input lines up.
        const shapes_ok = rows != 0 and heads != 0 and hd != 0 and gqa != 0 and
            q.len >= rows * row_len and gate.len >= rows * row_len and
            gout.len >= rows * row_len and
            kcache.len >= n_tot * kv_dim and vcache.len >= n_tot * kv_dim;

        // A chunk appended to a long KV cache can need more `s_scores` than the
        // 48 KB dynamic budget allows; `q35_attn_chunk_max_rows` reports the
        // slice size, and slicing is safe because row r depends only on earlier
        // rows and the cache prefix it needs.
        var maxrows: usize = 0;
        if (self.cu != null and shapes_ok) {
            const m = Kernels.q35_attn_chunk_max_rows(@intCast(slot0));
            if (m > 0) maxrows = @intCast(m);
        }

        if (maxrows != 0) {
            if (dBufPtr(q)) |pq| {
                if (dBufPtr(gate)) |pgate| {
                    if (dBufPtr(kcache)) |pk| {
                        if (dBufPtr(vcache)) |pv| {
                            if (dBufPtr(gout)) |pgout| {
                                var done: usize = 0;
                                while (done < rows) {
                                    // `maxrows` shrinks as `slot0` grows, so
                                    // recompute it per slice rather than
                                    // trusting the first value.
                                    const m2 = Kernels.q35_attn_chunk_max_rows(@intCast(slot0 + done));
                                    if (m2 <= 0) break;
                                    const take = @min(rows - done, @as(usize, @intCast(m2)));
                                    const rc = Launch.attn_chunk(
                                        pq + done * row_len, pgate + done * row_len,
                                        pk, pv,
                                        pgout + done * row_len,
                                        @intCast(slot0 + done), @intCast(take),
                                        @intCast(heads), @intCast(gqa), @intCast(hd),
                                        @intCast(kv_dim), scale, self.launchStream());
                                    if (rc != 0) {
                                        std.debug.print("cuda backend: attn_chunk launch failed ({d})\n", .{rc});
                                        break;
                                    }
                                    done += take;
                                }
                                if (done == rows) return;
                            }
                        }
                    }
                }
            }
        }
        // Fall back row by row (bad shapes, no device, or a rejected slice).
        var r: usize = 0;
        while (r < rows) : (r += 1) {
            dAttnDecode_body(ctx,
                q.span(r * row_len, row_len),
                gate.span(r * row_len, row_len),
                kcache, vcache, att,
                gout.span(r * row_len, row_len),
                slot0 + r, heads, gqa, hd, kv_dim, scale);
        }
    }

    /// `rope64(slot)`, timed (`$QWEN35_PROF`).
    fn dRope64(ctx: *anyopaque, x: backend.Buf, pos: u32, base: f32) void {
        const self = self_(ctx);
        const t0 = self.slotEnter();
        defer self.slotExit(.rope64, t0);
        if (self.ablated(.rope64)) return;
        dRope64_body(ctx, x, pos, base);
    }

    /// The prefill row-batched `rope64` slot, timed (`$QWEN35_PROF`).
    fn dRope64Rows(ctx: *anyopaque, x: backend.Buf, pos: u32, base: f32, rows_in: usize, heads_in: usize, hd_in: usize) void {
        const self = self_(ctx);
        const t0 = self.slotEnter();
        defer self.slotExit(.rope64_rows, t0);
        if (self.ablated(.rope64_rows)) return;
        dRope64Rows_body(ctx, x, pos, base, rows_in, heads_in, hd_in);
    }

    /// The `convStep` slot, timed (`$QWEN35_PROF`).
    fn dConvStep(ctx: *anyopaque, conv: backend.Buf, y: backend.Buf, wconv: backend.Buf, k: usize) void {
        const self = self_(ctx);
        const t0 = self.slotEnter();
        defer self.slotExit(.conv_step, t0);
        if (self.ablated(.conv_step)) return ;
        dConvStep_body(ctx, conv, y, wconv, k);
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
    .stepBegin = CudaBackend.dStepBegin,
    .stepEnd = CudaBackend.dStepEnd,
    .setTier = CudaBackend.dSetTier,
    .reportProfile = CudaBackend.dReportProfile,
    .profReport = CudaBackend.dProfReport,
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
    .rmsnormRows = CudaBackend.dRmsnormRows,
    .rope64 = CudaBackend.dRope64,
    .rope64Rows = CudaBackend.dRope64Rows,
    .l2norm = CudaBackend.dL2norm,
    .l2normRows = CudaBackend.dL2normRows,
    .siluInPlace = CudaBackend.dSiluInPlace,
    .mulSilu = CudaBackend.dMulSilu,
    .mulSigmoid = CudaBackend.dMulSigmoid,
    .softmax = CudaBackend.dSoftmax,
    .swiglu = CudaBackend.dSwiglu,
    .convStep = CudaBackend.dConvStep,
    .gdnStep = CudaBackend.dGdnStep,
    .gdnStepRows = CudaBackend.dGdnStepRows,
    .gdnStepRowsChunk = CudaBackend.dGdnStepRowsChunk,
    .loadVecF32 = CudaBackend.vLoadVecF32,
    .freeVec = CudaBackend.vFreeVec,
    .attnDecode = CudaBackend.dAttnDecode,
    .attnChunk = CudaBackend.dAttnChunk,
    .gdnGamma = CudaBackend.dGdnGamma,
    .gdnPrefillCore = CudaBackend.dGdnPrefillCore,
    .copyFromHost = CudaBackend.dCopyFromHost,
    .copyToHost = CudaBackend.dCopyToHost,
    .argmax = CudaBackend.dArgmax,
    .sample = CudaBackend.dSample,
    .sampledToken = CudaBackend.dSampledToken,
    // The four pure elementwise slots.  Forwarders, NOT aliases: they need
    // `ctx` to reach the device, and the ctx here is a `*CudaBackend`
    // (aliasing a CPU body would work only because those bodies ignore ctx).
    // Each still falls back to the CPU path on a device-side failure.
    .scaleVec = CudaBackend.dScaleVec,
    .addResid = CudaBackend.dAddResid,
    .copyRange = CudaBackend.dCopyRange,
    .copyRangeSlot = CudaBackend.dCopyRangeSlot,
    .deinterleaveQG = CudaBackend.dDeinterleaveQG,
    .deinterleaveQGStrided = CudaBackend.dDeinterleaveQGStrided,
    .zeroBuf = CudaBackend.dZeroBuf,
    // Activations are PROPER device regions after P2c, so `allocBuf` is a real
    // `cudaMalloc` and `freeScratch` releases them (`dAllocBuf`/`dFreeScratch`).
    .allocBuf = CudaBackend.dAllocBuf,

};
