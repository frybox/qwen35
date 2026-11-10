// api.h -- the only surface the Zig backend sees of the CUDA kernels.
//
// Why a separate header: `src/backend_cuda.zig` declares these with `extern`
// and never includes a CUDA header, so the Zig side cannot accidentally pick
// up a link-time dependency on cudart.  Everything that touches the CUDA
// runtime lives in the .cu files on the other side of this boundary, and those
// .cu files are compiled into a plain .o that is linked into the executable.
//
// The launchers return the numeric value of `cudaError_t` (0 == cudaSuccess)
// cast to int, so the caller can turn a failure into a named Zig error plus
// cudaGetErrorString() text instead of panicking.  Device pointers are opaque
// `void *` and the stream is `void *` too, which keeps this header free of
// cuda_runtime.h (and therefore includable from plain C/C++ if ever needed).
#ifndef Q35_CUDA_API_H
#define Q35_CUDA_API_H

#ifdef __cplusplus
extern "C" {
#endif

// Layout the host side must satisfy (mirrors ggml block_q8_0):
//   32 elements per block, 34 bytes per block (f16 d @0, int8 qs @2).
#define Q35_API_QK8_0 32
#define Q35_API_Q8_0_BLOCK_BYTES 34

// Layout of a Q8_K activation block (mirrors ggml block_q8_K):
//   float d @0, int8 qs[256] @4, int16 bsums[16] @260 -> 292 bytes / 256 elems.
#define Q35_API_QK_K 256
#define Q35_API_Q8_K_BLOCK_BYTES 292

// --- device capability probe (src/kernels/cuda/caps.cu) -------------------
// The CUDA-side mirror of src/kernels/x86/cpu_caps.c: probed once on first
// use, every later call is a cached read.  cc is major*100 + minor (809 ==
// sm_89); -1 / 0 when no usable device was found (caps stay at zero).
int q35_gpu_cc(void);
int q35_gpu_sm_count(void);
const char *q35_gpu_name(void);

// ---------------------------------------------------------------------------
// TensorType ids.
//
// MUST MATCH src/gguf.zig's `TensorType` enum, both in VALUE and in NAME --
// the numeric values below are copied from that enum, so Zig can pass
// `@intFromEnum(ttype)` straight through without a translation table.
// Renumbering anything here without renumbering gguf.zig silently makes every
// CUDA kernel read the wrong block geometry (the kernels then read
// out of bounds rather than producing obviously-wrong numbers).
//
// Types with no CUDA dot/dequant implementation are deliberately absent; the
// host launchers reject unknown ids with cudaErrorInvalidValue.
// ---------------------------------------------------------------------------
#define Q35_TT_F32      0
#define Q35_TT_F16      1
#define Q35_TT_Q4_0     2
#define Q35_TT_Q4_1     3
#define Q35_TT_Q5_0     6
#define Q35_TT_Q5_1     7
#define Q35_TT_Q8_0     8
#define Q35_TT_Q8_1     9
#define Q35_TT_Q2_K     10
#define Q35_TT_Q3_K     11
#define Q35_TT_Q4_K     12
#define Q35_TT_Q5_K     13
#define Q35_TT_Q6_K     14
#define Q35_TT_Q8_K     15
#define Q35_TT_IQ2_XXS  16
#define Q35_TT_IQ2_XS   17
#define Q35_TT_IQ3_XXS  18
#define Q35_TT_IQ1_S    19
#define Q35_TT_IQ4_NL   20
#define Q35_TT_IQ3_S    21
#define Q35_TT_IQ2_S    22
#define Q35_TT_IQ4_XS   23
#define Q35_TT_BF16     30
#define Q35_TT_IQ1_M    29

// --- activation quantization (src/kernels/cuda/convert.cu, mmvq.cu) --------
// `nblocks` is a count of 32-ELEMENT blocks, not a count of elements.
int q35_launch_quantize_q8_0(const float *d_x, void *d_out, int nblocks,
                             void *stream);
// `nblocks` is a count of 256-ELEMENT blocks.  Bit-exact with the CPU
// quantize_q8_K (src/kernels/x86/kernels256.c:392).
int q35_launch_quantize_q8_K(const float *d_x, void *d_out, int nblocks,
                             void *stream);
// MMQ activation layout (block_q8_1_mmq, 160 B per 128 values, D4 scales).
// `nblocks_k` counts 128-VALUE blocks per row and `nrows` token rows, so the
// destination holds nrows*nblocks_k blocks.  This is the layout the prefill
// tile kernel reads; see src/kernels/cuda/mmq-smem.cuh.
int q35_launch_quantize_q8_1_mmq(const float *d_x, void *d_y, int nblocks_k,
                                 int nrows, void *stream);

// --- dequantization (src/kernels/cuda/convert.cu) --------------------------
// `nblocks` is a count of the type's NATIVE blocks (32 elements for
// q8_0/q4_0/q4_1/q5_0/q5_1/iq4_nl, 256 for the K/IQ family), NOT elements.
int q35_launch_dequantize(int ttype, const void *d_src, float *d_dst, int nblocks,
                          void *stream);
// Dequantize one weight row of `cols` floats.  `cols` must be a multiple of the
// type's native block size.  Same semantics as src/backend_cpu.zig's
// dequantRowInto: handles the row block by block, so it never needs a
// temporary allocation.
int q35_launch_dequantize_row(int ttype, const void *d_src_row, float *d_dst,
                              int cols, void *stream);

// --- matvec (src/kernels/cuda/mmvq.cu) -------------------------------------
// out[t*rows + r] = W[r] . xq[t] for `tokens` activation rows.
// `xq` must be quantized in the layout the type asks for: Q8_0 (34 B / 32
// elems) for the 32-element family, Q8_K (292 B / 256 elems) for the 256-elem
// family -- see q35_quant_act_block_elems().  `cols` must be a multiple of the
// type's native block size.  Returns cudaErrorInvalidValue for an unknown type.
int q35_launch_mul_mat_vec(const void *d_w, const void *d_xq, float *d_out,
                           int rows, int cols, int tokens, int ttype,
                           void *stream);

// Backwards-compatible alias for
// q35_launch_mul_mat_vec(..., Q35_TT_Q8_0, ...).
int q35_launch_mul_mat_vec_q8_0(const void *d_w, const void *d_xq, float *d_out,
                                int rows, int cols, int tokens, void *stream);

int q35_launch_mul_mat_vec_f32(const float *d_w, const float *d_x, float *d_out,
                               int rows, int cols, int tokens, void *stream);

// --- pure elementwise activations (src/kernels/cuda/elementwise.cu) --------
//
// `n` is always an ELEMENT count of f32 (never bytes), and every buffer is a
// contiguous span.  `n <= 0` returns cudaSuccess without launching.
//
// All four are BIT-EXACT with their CPU counterparts in src/elementwise.zig
// (one IEEE single multiply / add / copy / zero per element, no fusion, no
// reassociation), which is why the engine's token stream must not change when
// these are served by the GPU.  See the file's own comment.
int q35_launch_scale_vec(float *d_x, float s, int n, void *stream);
// Prefill twin: `tokens` runs of `n` elements, consecutive runs `stride` floats
// apart, `offset` floats into `d_x`.  One launch instead of one per token.
int q35_launch_scale_vec_strided(float *d_x, int offset, float s, int n,
                                 int tokens, int stride, void *stream);
int q35_launch_add_resid(float *d_dst, const float *d_src, int n, void *stream);
int q35_launch_copy_range(float *d_dst, const float *d_src, int n, void *stream);
int q35_launch_deinterleave_qg(float *d_q, float *d_gate, const float *d_src, int heads, int hd, void *stream);
// `tokens` rows of [q|gate] split in ONE launch (chunk-wide prefill).  Row t
// starts at `t * 2 * heads * hd`; bit-identical to `tokens` calls above.
int q35_launch_deinterleave_qg_strided(float *d_q, float *d_gate, const float *d_src,
                                       int heads, int hd, int tokens, void *stream);
int q35_launch_zero_buf(float *d_buf, int n, void *stream);

// Write one int32 from a 1-thread kernel.  The graph-safe way to pass a
// per-token scalar: the reader takes a device POINTER, so the captured node's
// argument block never has to be retargeted with GraphExecKernelNodeSetParams.
int q35_launch_set_int1(int *d, int v, void *stream);

// --- normalisation slots (src/kernels/cuda/norm.cu) ------------------------
//
// `rows` blocks of `n` f32 each, contiguous.  `d_w` is a length-`n` weight
// (the same one for every row); l2norm has none.  `eps` is passed by the Zig
// side so the CPU reference's constant lives in one place (elementwise.zig
// `l2norm` hardcodes 1e-6, `rmsNorm` takes it from the model config).
//
// NOT bit-exact with the CPU: the sum of squares is a block tree reduction
// whereas the CPU sums serially, so the last bits of the scale differ.  The
// elementwise multiply and the `1/sqrtf` are exact.  Measured max relative
// error 8.8e-7 (13 ULP) for rmsnorm, 8.3e-7 (12 ULP) for l2norm; see norm.cu's
// header for why the ULP count is higher than the scale's own 1-2.
int q35_launch_rmsnorm(float *d_out, const float *d_x, const float *d_w,
                       int rows, int n, float eps, void *stream);
int q35_launch_l2norm(float *d_x, int rows, int n, float eps, void *stream);
// Prefill twin: `tokens` x `rows` normalisations in one launch; token t's row
// block starts `row_stride` floats past token t-1's, and `row_offset` selects
// the q run (0) or the k run (kq_dim) inside that block.
int q35_launch_l2norm_strided(float *d_x, int row_offset, int rows, int n,
                              float eps, int tokens, int row_stride, void *stream);

// --- activation / conv slots (src/kernels/cuda/act.cu) ---------------------
//
// Not bit-exact either, for one reason and one only: the device `expf` is a
// different implementation from glibc's (max 2 ULP measured), while everything
// around it is spelled as the same sequence of rounded operations the CPU
// performs.  softmax additionally sums over a block instead of serially.
// Measured: 4-5 ULP / <=3.7e-7 relative for the four activations, 34 ULP but
// only 2.1e-6 relative for softmax (its outputs are all < 1).  See act.cu.
int q35_launch_silu(float *d_x, int n, void *stream);
int q35_launch_mul_silu(float *d_out, const float *d_gate, int n, void *stream);
int q35_launch_mul_sigmoid(float *d_out, const float *d_gate, int n, void *stream);
int q35_launch_swiglu(float *d_gate, const float *d_up, int n, void *stream);
int q35_launch_softmax(float *d_x, int n, void *stream);
// 4-tap causal conv over the per-channel history `d_conv` (3 f32 per channel,
// shifted in place), then silu over the whole row.  Two launches internally.
int q35_launch_conv_step(float *d_conv, float *d_y, const float *d_wconv,
                         int n, void *stream);
// Prefill twin: `tokens` rows of `d_y` (`y_stride` floats apart) sharing ONE
// per-channel history `d_conv`, in two launches instead of `2 * tokens`.
// Equivalent to calling `q35_launch_conv_step` on each row in ascending order.
int q35_launch_conv_step_rows(float *d_conv, float *d_y, const float *d_wconv,
                              int n, int tokens, int y_stride, void *stream);

// index of the largest logit (first index wins on ties, matching
// elementwise.argmaxKernel).  Input is device logits, output a single int.
int q35_launch_argmax(const float *d_x, int n, int *d_out, void *stream);

// --- rotary embedding (src/kernels/cuda/rope.cu) ---------------------------
//
// `d_x` is `heads * hd` contiguous f32; the FIRST 64 floats of every `hd`-wide
// head are rotated with cross-half pairing (j, j+32), dims 64..hd-1 pass
// through.  `pos` is the absolute position, `base` the model's rope freq base.
//
// NOT bit-exact with the CPU, and cannot be: the device `powf` differs from
// glibc's by ~1 ULP relative, which is an absolute phase error growing with
// `pos` (see rope.cu's header).  On by default on the Zig side anyway: the
// drift costs cross-path token identity, which was judged worth +10.6% prefill
// / +4.4% generate, and is documented with the measured divergence in
// rope.cu's header and backend_cuda.zig `ropeOnDevice`.  $QWEN35_CUDA_ROPE=host
// reverts to the bit-exact CPU body.
int q35_launch_rope64(float *d_x, int heads, int hd, int pos, float base,
                      void *stream);

// `rope64` with the per-token inputs kept in DEVICE memory: `d_dyn` points at
// two device int32, {pos, bit pattern of the f32 base}.  Same arithmetic as
// `q35_launch_rope64` (see rope.cu), but because no kernel parameter changes
// from token to token, a captured decode graph can replay these nodes without
// `cudaGraphExecKernelNodeSetParams` -- which on a 27B meant 448 calls per
// token and was the single largest host cost in decode.
int q35_launch_rope64_dpos(float *d_x, int heads, int hd, const void *d_dyn,
                           void *stream);
void *q35_func_rope64_dpos(void);
// Publish {pos, float-bits(base)} into the channel `q35_launch_rope64_dpos`
// reads.  A kernel, not a memcpy, because only a kernel is capturable.
int q35_launch_rope64_setenv(int *d_dyn, int pos, float base, void *stream);

// --- Gated Delta Net state machine -----------------------------------------
// All `heads` value heads of one delta-rule state update in ONE launch
// (`heads` blocks; block h is the call above for head h).
//
// NOTE: q35_launch_gdn_step_rows() below is the batched form.  Both exist: the
// engine uses the batched one, the single-row one is kept for fallback/A-B.
//
// Strided bases, all in elements.  Head h touches
//   d_state + h*state_stride   (kd*vd floats, updated in place)
//   d_k / d_q  + (h % nk)*k_stride / q_stride   (kd floats)
//   d_v / d_out / d_km / d_delta + h*v_stride   (vd floats)
// and scalars d_gamma[h], d_beta[h].  d_km / d_delta are PER-HEAD scratch and
// be `heads*vd` long: the blocks run concurrently, so a shared vd-float buffer
// would be a silent race (not a crash).  The engine sizes them.
// Returns cudaErrorInvalidValue unless vd == kd == 128 and heads > 0.
int q35_launch_gdn_step_rows(float *d_state, int state_stride,
                             const float *d_k, int k_stride,
                             const float *d_v, int v_stride,
                             const float *d_q, int q_stride,
                             float *d_km, float *d_delta,
                             float *d_out,
                             const float *d_gamma,
                             const float *d_beta,
                             int heads, int nk, int kd, int vd,
                             void *stream);

// One token of the delta-rule state update, one launch per value head.
// d_state is the [kd, vd] slice for THIS head (row-major S[k][v], kd*vd floats),
// updated in place; d_k/d_q are kd, d_v/d_km/d_delta/d_out are vd.  See
// gdn.cu for the accumulation-order contract with engine.zig's gdnFused.
int q35_launch_gdn_step(float *d_state, const float *d_k, const float *d_v,
                        const float *d_q, float *d_km, float *d_delta,
                        float *d_out, float gamma, float beta, int vd,
                        void *stream);

// beta = sigmoid(b) written IN PLACE into d_b_raw; gamma = exp(ssm_a *
// softplus(a + dt_bias)).  nv = per-value-head scalars.
int q35_launch_gdn_gamma(const float *d_a_raw, float *d_b_raw,
                         const float *d_ssm_a, const float *d_dt_bias,
                         float *d_gamma, int nv, void *stream);
// Prefill twin: `tokens` tokens' gamma/beta in ONE launch.  Token t reads its
// `nv`-wide a/b slices `a_stride`/`b_stride` floats apart, writes gamma
// `gamma_stride` floats apart; `d_ssm_a`/`d_dt_bias` are shared per-head model
// weights.  beta is still overwritten in place with sigmoid(b).
// Per-token arithmetic is identical to `q35_launch_gdn_gamma`.
int q35_launch_gdn_gamma_rows(const float *d_a_raw, float *d_b_raw,
                              const float *d_ssm_a, const float *d_dt_bias,
                              float *d_gamma, int nv, int tokens, int a_stride,
                              int b_stride, int gamma_stride, void *stream);

// --- Full-attention decode -------------------------------------------------
// scores(q,K) -> softmax -> P.V -> sigmoid gate, for every head, in ONE launch.
// d_q / d_gate / d_gout are heads*hd; d_kcache / d_vcache are (slot+1) rows of
// kv_dim; d_att is scratch, (slot+1) floats (reused per head).
int q35_launch_attn_decode(const float *d_q, const float *d_gate,
                           const float *d_kcache, const float *d_vcache,
                           float *d_att, float *d_gout, int slot, int heads,
                           int gqa, int hd, int kv_dim, float scale,
                           void *stream);

// Chunk-wide causal attention for PREFILL: one launch for `rows` tokens instead
// of `rows` launches.  `d_q`/`d_gate`/`d_gout` are `rows * heads * hd` (row r at
// `r*heads*hd + h*hd`); the KV cache must already hold slots [0, slot0+rows).
// Row r attends to [0, slot0+r] inclusive, so the r-loop inside the kernel is
// serial -- what the merge removes is the per-token KERNEL BOUNDARY (8400
// launches -> 16 on a 525-token 27B prefill), not the dependency.
//
// `s_scores` is `slot0+rows` floats, so a chunk appended to a long KV cache can
// exceed the 48 KB dynamic budget; the caller slices with
// `q35_attn_chunk_max_rows` in that case.  Slicing is safe: row r depends only
// on earlier rows and the cache prefix it needs.
int q35_launch_attn_chunk(const float *d_q, const float *d_gate,
                          const float *d_kcache, const float *d_vcache,
                          float *d_gout, int slot0, int rows, int heads,
                          int gqa, int hd, int kv_dim, float scale,
                          void *stream);
int q35_attn_chunk_max_rows(int slot0);

// Same kernel, but `slot` is read from DEVICE memory (`d_slot`, one int) and the
// dynamic shared block is the fixed maximum.  Because neither the argument block
// nor `shared_mem_bytes` changes between tokens, a captured decode-graph node
// for this launch never needs `cudaGraphExecKernelNodeSetParams` -- which cost
// 14.3 ms/token (43% of decode) on the 27B when it ran once per layer per token.
// Preferred on the graph path; the scalar-`slot` form above stays for prefill,
// the direct path and the fallback.
void *q35_func_attn_decode_dyn(void);
void *q35_func_attn_decode_coal(void);
void *q35_func_attn_score_split(void);
int q35_launch_attn_decode_dyn(const float *d_q, const float *d_gate,
                               const float *d_kcache, const float *d_vcache,
                               float *d_att, float *d_gout, const int *d_slot,
                               int heads, int gqa, int hd, int kv_dim,
                               float scale, void *stream);

// Warp-tiled decode attention: the row loop is re-tiled so a warp reads ONE
// contiguous run of a KV row instead of 32 rows 4 KB apart (which over-fetches
// by kv_dim/hd and defeats coalescing).  Still bit-exact -- the score j-sum is
// the serial order and the PV t-sum walks rows in increasing order.
// Measured (24 layers, distinct 100-180 MB KV caches, i.e. NOT L2-resident):
// 3.1 -> 1.5 ms/token at slot=518, 4.9 -> 2.5 ms at slot=917 (33 -> 72 GB/s).
int q35_launch_attn_decode_coal(const float *d_q, const float *d_gate,
                                const float *d_kcache, const float *d_vcache,
                                float *d_att, float *d_gout, int slot, int heads,
                                int gqa, int hd, int kv_dim, float scale,
                                void *stream);

// Split decode attention: BOTH walks (score and PV) divide the row range across
// `split` blocks per head, so the kernel stops being 24-blocks-on-114-SMs.
// `d_part` must hold `heads * split * hd` floats.
// Measured: 325 GB/s / 0.555 ms at slot=917 (vs 4.93 ms naive) with output
// BIT-IDENTICAL to the reference (the score sum keeps the serial j order and the
// softmax keeps the CPU's max/exp/serial-sum/div order; only the PV t-sum is a
// 2-level tree, which re-associates f32 addition -- and even that came out
// identical on the probe).  `split <= 1` falls back to the coalesced kernel.
void *q35_func_attn_score_split_dyn(void);
void *q35_func_attn_pv_split_dyn(void);
void *q35_func_attn_softmax_dyn(void);
void *q35_func_attn_reduce(void);
// Device-`slot` split path: four launches, every argument constant, so a
// captured decode-graph node never needs retargeting.  `split` is a constant.
int q35_launch_attn_decode_split_dyn(const float *d_q, const float *d_gate,
                                     const float *d_kcache, const float *d_vcache,
                                     float *d_att, float *d_gout, float *d_part,
                                     const int *d_slot, int heads, int gqa, int hd,
                                     int kv_dim, float scale, int split,
                                     void *stream);
int q35_launch_attn_decode_split2(const float *d_q, const float *d_gate,
                                  const float *d_kcache, const float *d_vcache,
                                  float *d_att, float *d_gout, float *d_part,
                                  int slot, int heads, int gqa, int hd,
                                  int kv_dim, float scale, int split,
                                  void *stream);

// --- IQ codebook upload (call ONCE, before the first IQ kernel) -------------
//
// The twelve IQ lookup tables are DEFINED ONCE, in src/iq_tables.zig, and are
// compiled into this executable (the CUDA kernels are linked into the CLI now,
// not into a separately dlopen'd .so, so their host addresses are ordinary
// link-time symbols).  Device code cannot read host memory, so this call copies
// all twelve to a single ~40 KB device buffer at start-up and points the
// __device__ pointers of iq_tables.cuh at their slices; every kernel that
// touches an IQ type reads that copy, and none of them can work before this has
// succeeded.
//
// Returns 0 (cudaSuccess) or the numeric cudaError_t of the first failure -- a
// failure is never swallowed, the backend turns it into a load error rather than
// launching IQ kernels over a null table pointer.  Idempotent: later calls return
// the first result without touching the device again.  See iq_tables.cu for the
// upload plan and for why it uses cudaMemcpyToSymbolAsync rather than an init
// kernel (measured, on this toolchain).
int q35_launch_init_tables(void *stream);

// Bytes resident on the device for the codebooks (0 before a successful init),
// and 1 once the upload succeeded.  Both are pure host reads: no CUDA call.
unsigned long long q35_iq_tables_bytes(void);
int q35_iq_tables_ready(void);

// Free the device buffer and reset the idempotence flag (so a later backend in
// the same process re-uploads).  Safe to call when init never ran.
void q35_launch_deinit_tables(void);

// --- geometry queries (pure host code, no launch) --------------------------
// 256 for types that want Q8_K activations, 32 for Q8_0, 0 for unknown types.
int q35_quant_act_block_elems(int ttype);
// Native block geometry of a type.  Returns 1 and fills the out-params on
// success, 0 for an unknown type.
int q35_quant_geom(int ttype, int *block_elems, int *block_bytes);

#ifdef __cplusplus
}
#endif

#endif // Q35_CUDA_API_H
