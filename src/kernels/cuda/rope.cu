// rotary embedding (rope64): cross-half pairing over the first 64 elements of
// a head, `(i, i+32)`, one rotation per head, per position.
//
// Reference: src/elementwise.zig `applyRope64` / src/engine.zig `applyRope64`
// (both are the same body -- `elementwise.zig` says so in its own comment):
//
//     half = 32
//     for j in 0..32:
//         theta = (f32)pos * powf(base, -2j/64)      // f32 powf, f32 product
//         c = cosf(theta); s = sinf(theta)
//         a = x[j]; b = x[j+32]
//         x[j]    = a*c - b*s
//         x[j+32] = a*s + b*c
//
// This kernel is the SAME expression sequence with the same operand types, and
// nothing else.  Two deliberate choices:
//
//  * `powf` / `cosf` / `sinf`, NOT `__powf` / `__cosf` / `__sinf`.  The fast
//    intrinsics are a different (lower-accuracy) polynomial; the whole point of
//    this kernel is to answer "how far does the device libm drift from glibc's
//    on this phase", so it must be the standard device libm.
//  * every arithmetic op is the rounded intrinsic (`__fmul_rn`, `__fadd_rn`,
//    `__fsub_rn`) even though the compiler defaults already forbid contraction
//    here: nvcc fuses with `-fmad=true` by default and `a*c - b*s` is exactly
//    the shape that becomes two FMAs.  Same rationale as elementwise.cu.
//
// ACCURACY CONTRACT: NOT bit-exact with the CPU, and cannot be.  On the CPU the
// angle is `(f32)pos * glibc_powf(base, ...)`; glibc's `powf` and the device's
// `powf` differ by ~1 ULP relative, which is an ABSOLUTE phase error of order
// `pos * 1.2e-7` radians -- ~1e-4 rad at pos = 1000, ~1e-2 rad at pos = 1e6.
// `cos` of that differs by up to the same amount, so the rotated value carries
// up to that relative error.  Recomputing theta in double on the device does
// NOT help (measured: still 5.2e-4): the host's f32 `powf` is the lower-
// precision side, so matching it is not the same as being more accurate.
//
// THE DRIFT IS NOT HARMLESS AT THIS MODEL'S DEPTH, and that is accepted, not
// overlooked.  Measured on Qwen3.8-27B IQ3_XXS: the first greedy (argmax)
// token that differs from the host path lands at generated token 34 for a
// 17-token prompt (pos ~50) and at generated token 1 for a 354-token prompt --
// 65 layers of q/k rotation amplify a ~1e-4 rad phase error past the top-1 /
// top-2 logit margin long before pos 1000.  Each path stays deterministic on
// its own (3 runs, identical md5), so the A/B is still usable; what is given up
// is cross-path comparability, which llama.cpp's CPU-vs-CUDA backends do not
// offer either.  The trade was taken for throughput: +10.6% prefill / +4.4%
// generate on the numbers in `src/backend_cuda.zig` `ropeOnDevice`.
//
// `$QWEN35_CUDA_ROPE=host` reverts to the CPU body (bit-exact) for exactly this
// A/B; see the note there.
#include "common.cuh"

namespace {

// One head = 64 floats = 32 rotations.  A warp per head: lane j < 32 does
// rotation j, so the whole head's trigonometry is 32 lanes wide and the 64
// loads/stores are one coalesced 256 B transaction.  Blocks are 8 heads
// (256 threads), matching the launch shape of the rest of this kernel set.
constexpr int kHalf = 32;
constexpr int kHeadsPerBlock = 8;

__global__ void k_rope64(float *__restrict__ x, int heads, int hd, int pos, float base) {
    const int lane = threadIdx.x & (kHalf - 1);
    const int local_head = threadIdx.x / kHalf;
    const int h = blockIdx.x * kHeadsPerBlock + local_head;
    if (h >= heads) return;
    // `hd` is always a multiple of 64 (the engine passes hd = 64 or 128 with
    // rotary_dim = 64), so staying inside dims 0..63 leaves dims 64..hd-1
    // untouched, exactly like the reference, which only ever sees a 64-float
    // slice of the head.
    float *hp = x + (size_t)h * (size_t)hd;
    const int j = lane;
    // -2j/64 in f32, then the f32 product with (f32)pos -- same two roundings
    // as `(f32)pos * pow(f32, base, -(f32)(2j)/64.0f)`.
    const float exponent = -2.0f * (float)j / 64.0f;
    const float theta = __fmul_rn((float)pos, powf(base, exponent));
    const float c = cosf(theta);
    const float s = sinf(theta);
    const float a = hp[j];
    const float b = hp[j + kHalf];
    hp[j] = __fsub_rn(__fmul_rn(a, c), __fmul_rn(b, s));
    hp[j + kHalf] = __fadd_rn(__fmul_rn(a, s), __fmul_rn(b, c));
}

__global__ void k_rope64_rows(float *__restrict__ x, int rows, int heads, int hd,
                             int pos0, float base) {
    // Decode the flat (row, head) pair this warp handles.  `rows` is the token
    // count and `heads` the head count PER ROW, so the full tensor is
    // `rows * heads * hd` floats and row r starts at `r * heads * hd`.
    const int lane = threadIdx.x & (kHalf - 1);
    const int local_head = threadIdx.x / kHalf;
    const int flat = blockIdx.x * kHeadsPerBlock + local_head;
    const int rh = flat;
    if (rh >= rows * heads) return;
    const int r = rh / heads;
    // `h` is implied by `rh` (the row-major (row, head) order) and is not used
    // directly: the pointer below already accounts for it.
    float *hp = x + ((size_t)rh) * (size_t)hd;
    const int j = lane;
    // Each row rotates at its own position: row r is token `pos0 + r`.  The
    // arithmetic below is the single-row `k_rope64` body unchanged, so row r of
    // this launch is bit-identical to a `k_rope64` launch with `pos = pos0 + r`.
    const int pos = pos0 + r;
    const float exponent = -2.0f * (float)j / 64.0f;
    const float theta = __fmul_rn((float)pos, powf(base, exponent));
    const float c = cosf(theta);
    const float s = sinf(theta);
    const float a = hp[j];
    const float b = hp[j + kHalf];
    hp[j] = __fsub_rn(__fmul_rn(a, c), __fmul_rn(b, s));
    hp[j + kHalf] = __fadd_rn(__fmul_rn(a, s), __fmul_rn(b, c));
}

// Device-position variant of `k_rope64`.
//
// WHY THIS EXISTS (this is the whole point of the kernel):
//
// In decode the same `rope64` op is issued hundreds of times per token -- once
// per tensor per layer, 448 of them on a 27B.  Inside the captured decode CUDA
// graph every one of those nodes used to carry `pos` as a KERNEL PARAMETER, so
// the only way to advance a token was `cudaGraphExecKernelNodeSetParams` on each
// of the 448 nodes, once per token.  Device-side profiling of the 27B IQ3_XXS
// decode showed the cost of that clearly:
//
//     tok_dev_ms = 34.4     (wall time per token, on device)
//     gpu_sum_ms = 12.4     (sum of all kernel execution)
//     idle_ms    = 22.0     (64% of the token: the GPU has nothing to run)
//
// Half the token was the CPU re-parameterising the graph.  Reading `pos` from a
// device scalar instead makes every one of those nodes' parameters CONSTANT, so
// a token costs one tiny H2D of the position and no per-node retargeting at all.
// The arithmetic is copied expression for expression from `k_rope64` above, so a
// node running this kernel produces bit-identical output for the same `pos`.
__global__ void k_rope64_dpos(float *__restrict__ x, int heads, int hd, const void *dyn_p) {
    // `dyn_p` points at TWO device int32: {pos, float-bits(base)}.  `base` moves
    // too -- the engine derives it from the position (it switches once a token
    // passes the model's rope-scaling threshold) -- so freezing only `pos` would
    // rotate replays with the capture token's theta.
    const int lane = threadIdx.x & (kHalf - 1);
    const int local_head = threadIdx.x / kHalf;
    const int h = blockIdx.x * kHeadsPerBlock + local_head;
    if (h >= heads) return;
    const int *dyn = (const int *)dyn_p;
    const int pos = dyn[0];
    const float base = __int_as_float(dyn[1]);
    float *hp = x + (size_t)h * (size_t)hd;
    const int j = lane;
    const float exponent = -2.0f * (float)j / 64.0f;
    const float theta = __fmul_rn((float)pos, powf(base, exponent));
    const float c = cosf(theta);
    const float s = sinf(theta);
    const float a = hp[j];
    const float b = hp[j + kHalf];
    hp[j] = __fsub_rn(__fmul_rn(a, c), __fmul_rn(b, s));
    hp[j + kHalf] = __fadd_rn(__fmul_rn(a, s), __fmul_rn(b, c));
}

// Device-position twin of `k_rope64_rows`: identical arithmetic, but `{pos,
// base}` come from the same device channel `k_rope64_dpos` reads instead of
// being kernel arguments.
//
// WHY THIS EXISTS: the decode path has to keep its rope nodes recognisable to
// the captured graph, and a node is retargetable only if its host-stub address
// is known (`q35_func_rope64_dpos`).  That forced the decode engine to issue
// ONE `k_rope64_dpos` PER HEAD -- `heads`(40) + `kv_heads`(8) = 48 launches per
// full-attn layer, i.e. 768 per token over the 16 attn layers.  ncu on 27B
// IQ3_XXS decode measured `k_rope64_dpos` at 12.84 ms / 6188 calls over 20
// tokens (2.07 us each for 64 cos+sin), 27.9% of all non-matvec kernel time and
// the single largest elementwise kernel.  Every one of those launches rotates
// the SAME `{pos, base}`, so they can be batched into one node per tensor
// without changing a single value: row r of this launch is bit-identical to
// `k_rope64_rows` with `pos0 = pos + r` and the same base.
__global__ void k_rope64_rows_dpos(float *__restrict__ x, int rows, int heads, int hd,
                                   const void *dyn_p) {
    const int lane = threadIdx.x & (kHalf - 1);
    const int local_head = threadIdx.x / kHalf;
    const int rh = blockIdx.x * kHeadsPerBlock + local_head;
    if (rh >= rows * heads) return;
    const int *dyn = (const int *)dyn_p;
    // Row r is token `pos + r`; decode passes rows == 1, so this reduces to the
    // per-head kernel's own indexing.
    const int r = rh / heads;
    const int pos = dyn[0] + r;
    const float base = __int_as_float(dyn[1]);
    float *hp = x + ((size_t)rh) * (size_t)hd;
    const int j = lane;
    const float exponent = -2.0f * (float)j / 64.0f;
    const float theta = __fmul_rn((float)pos, powf(base, exponent));
    const float c = cosf(theta);
    const float s = sinf(theta);
    const float a = hp[j];
    const float b = hp[j + kHalf];
    hp[j] = __fsub_rn(__fmul_rn(a, c), __fmul_rn(b, s));
    hp[j + kHalf] = __fadd_rn(__fmul_rn(a, s), __fmul_rn(b, c));
}

} // namespace

// `d_pos` is a device int32 holding the position this token rotates at.  The
// launcher takes the address rather than the value so that the captured decode
// graph never has to rewrite this node: the caller stores the new position once
// per token and every rope node in the graph reads it.
extern "C" int q35_launch_rope64_dpos(float *d_x, int heads, int hd, const void *d_dyn,
                                      void *stream_v) {
    if (heads <= 0 || hd < 64) return (int)cudaSuccess;
    cudaStream_t stream = (cudaStream_t)stream_v;
    const int threads = kHalf * kHeadsPerBlock;
    const int blocks = (heads + kHeadsPerBlock - 1) / kHeadsPerBlock;
    k_rope64_dpos<<<blocks, threads, 0, stream>>>(d_x, heads, hd, d_dyn);
    return (int)cudaGetLastError();
}

// Host-stub address of `k_rope64_dpos`, for the decode graph to recognise these
// nodes -- the same role `q35_func_rope64` plays for the constant-`pos` kernel.
extern "C" void *q35_func_rope64_dpos(void) { return (void *)&k_rope64_dpos; }

// Publish {pos, base} into the channel `k_rope64_dpos` reads.  A one-thread
// kernel rather than a host-to-device copy on purpose: a copy is not capturable,
// so it would kill the decode graph the moment it is first recorded, while a
// kernel is just another node.  Launched once per token by the backend (the
// engine calls `rope64` once per head, and re-publishing 448 times would add a
// launch per head).
__global__ void k_rope64_setenv(int *dyn, int pos, float base) {
    dyn[0] = pos;
    dyn[1] = __float_as_int(base);
}

extern "C" int q35_launch_rope64_setenv(int *d_dyn, int pos, float base, void *stream_v) {
    if (d_dyn == nullptr) return (int)cudaSuccess;
    cudaStream_t stream = (cudaStream_t)stream_v;
    k_rope64_setenv<<<1, 1, 0, stream>>>(d_dyn, pos, base);
    return (int)cudaGetLastError();
}

// Row-batched `rope64`: `d_x` is `rows * heads * hd` contiguous f32, row r
// (token `pos0 + r`) is `heads * hd` floats of which each `hd`-wide head gets
// its first 64 rotated.  Row r is the same arithmetic as a single
// `q35_launch_rope64(d_x + r*heads*hd, heads, hd, pos0 + r, base)`.
//
// PREFILL ONLY.  The decode path must keep issuing one `k_rope64` per tensor per
// token because the decode CUDA graph identifies and retargets those nodes by
// host-stub address (`q35_func_rope64`); a batched node has no such identity.
// Prefill never captures a graph (see `forwardChunk`'s `stepEnd` contract), so
// batching there is free.
extern "C" int q35_launch_rope64_rows(float *d_x, int rows, int heads, int hd, int pos0,
                                     float base, void *stream_v) {
    if (rows <= 0 || heads <= 0 || hd < 64) return (int)cudaSuccess;
    cudaStream_t stream = (cudaStream_t)stream_v;
    const int threads = kHalf * kHeadsPerBlock;
    const int total = rows * heads;
    const int blocks = (total + kHeadsPerBlock - 1) / kHeadsPerBlock;
    k_rope64_rows<<<blocks, threads, 0, stream>>>(d_x, rows, heads, hd, pos0, base);
    return (int)cudaGetLastError();
}
extern "C" int q35_launch_rope64_rows_dpos(float *d_x, int rows, int heads, int hd,
                                           const void *d_dyn, void *stream_v) {
    if (rows <= 0 || heads <= 0 || hd < 64) return (int)cudaSuccess;
    cudaStream_t stream = (cudaStream_t)stream_v;
    const int threads = kHalf * kHeadsPerBlock;
    const int total = rows * heads;
    const int blocks = (total + kHeadsPerBlock - 1) / kHeadsPerBlock;
    k_rope64_rows_dpos<<<blocks, threads, 0, stream>>>(d_x, rows, heads, hd, d_dyn);
    return (int)cudaGetLastError();
}

extern "C" void *q35_func_rope64_rows_dpos(void) { return (void *)&k_rope64_rows_dpos; }

// `d_x` is `heads * hd` contiguous f32; only the first 64 of each `hd` are
// rotated.  `heads * hd` must be the tensor's element count.  hd <= 0 or
// heads <= 0 returns success without launching (same convention as the rest of
// api.h).
// Host-stub address of the kernel the launcher above launches, so the decode
// CUDA graph can recognise this op's node inside a captured graph
// (`cudaGraphKernelNodeGetParams` reports `func` as exactly this address).  A
// graph replays a node's arguments from a snapshot, so the ones that move per
// token have to be found and rewritten; matching on the stub is what makes that
// identification exact instead of a guess from argument values, which cannot
// distinguish kernels that merely happen to have a small int in a similar slot.
extern "C" void *q35_func_rope64(void) { return (void *)&k_rope64; }

extern "C" int q35_launch_rope64(float *d_x, int heads, int hd, int pos, float base, void *stream_v) {
    if (heads <= 0 || hd < 64) return (int)cudaSuccess;
    cudaStream_t stream = (cudaStream_t)stream_v;
    const int threads = kHalf * kHeadsPerBlock;
    const int blocks = (heads + kHeadsPerBlock - 1) / kHeadsPerBlock;
    k_rope64<<<blocks, threads, 0, stream>>>(d_x, heads, hd, pos, base);
    return (int)cudaGetLastError();
}
