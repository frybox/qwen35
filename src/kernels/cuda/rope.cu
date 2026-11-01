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

} // namespace

// `d_x` is `heads * hd` contiguous f32; only the first 64 of each `hd` are
// rotated.  `heads * hd` must be the tensor's element count.  hd <= 0 or
// heads <= 0 returns success without launching (same convention as the rest of
// api.h).
extern "C" int q35_launch_rope64(float *d_x, int heads, int hd, int pos, float base, void *stream_v) {
    if (heads <= 0 || hd < 64) return (int)cudaSuccess;
    cudaStream_t stream = (cudaStream_t)stream_v;
    const int threads = kHalf * kHeadsPerBlock;
    const int blocks = (heads + kHeadsPerBlock - 1) / kHeadsPerBlock;
    k_rope64<<<blocks, threads, 0, stream>>>(d_x, heads, hd, pos, base);
    return (int)cudaGetLastError();
}
