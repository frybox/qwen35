# CUDA elementwise: measured ulp distance vs the CPU path (2026-10-31)

Probe: `/tmp/zulp.zig` (Zig `-Ofast`, `@exp/@log/@sqrt/@cos` as the engine uses
them) vs `expf/log1pf/sqrtf/cosf` on `sm_89` (RTX 4090 D), 65536 samples per
value range, byte-for-byte comparison. Range chosen to match what the model
actually produces.

| op | range | differing samples | worst ulp |
|----|-------|-------------------|-----------|
| `sqrt`  | sum-of-squares argument [0, 1e6]  | 0 / 65536    | **0** |
| `exp`   | softmax/sigmoid argument [-30, 30] | 20652 / 65536 | 2 |
| `cos`   | rope angle [0, 4096]              | 12836 / 65536 | 1 |
| `softplus` = `log1p(exp(x))` | [0, 20] | 5047 / 65536 | 2 (inherits exp) |

Consequence for sinking elementwise kernels to the device:

* **Bit-exact-safe per element** (pure f32 arithmetic, IEEE-754 reproducible):
  `scaleVec`, `addResid`, `copyRange`, `zeroBuf`, `gdnStep` (gamma/beta are
  inputs, the body is mul/add), and the *elementwise* part of `rmsnorm`/`l2norm`.
* **Reduction order, not the op, is what breaks exactness** in `rmsnorm`/`l2norm`:
  the CPU accumulates the sum of squares **sequentially** (`for (x) |v| ss += v*v`),
  so a tree/parallel device reduction gives a different last-bit result even
  though every multiply-add is IEEE-exact. This is *accepted* (see the policy
  below) — do **not** force a single-thread sequential reduction on the device
  to match it; that cancels the entire reason for sinking the kernel.
* **NOT bit-exact-safe** (transcendentals): `softmax` (exp), `sigmoid`/`silu`
  (exp), `rope64` (sin/cos), `gdnGamma` (exp + log1p), `convStep` (silu),
  `attnDecode` (exp in softmax), `sample`.

A 1-2 ulp difference in a softmax score can flip a token when two logits are
close, so a fully device-side elementwise path cannot be *guaranteed* to
reproduce CPU token ids. **Therefore bit-exact reproduction of CPU token ids is
NOT the acceptance criterion** (see `cuda-backend-brief.md` §2bis):

- kernel level: still strict — `QWEN35_MVCHECK` row-by-row, plus a Zig scalar
  oracle per kernel (AGENTS.md §4).
- end-to-end: token id *distribution* agrees with CPU plus top-5 logits within
  tolerance and no perplexity regression. An occasional flipped token where two
  logits are close is expected and acceptable.

Do not chase md5 equality of `QWEN35_DUMP_IDS` here. Also note: the divergences
this probe was written to explain (Q4_0 / IQ4_XS / UD-Q4_K_XL forking at token
25) cannot come from elementwise ulp at all while every elementwise slot still
points at the CPU function (`backend_cuda.zig:1318-1340`); they come from
`mmvq` accumulation order / activation quantization. Localise those with
`QWEN35_MVCHECK`, not with this table.

`sqrt` being 0-ulp is what makes rmsnorm/l2norm sinkable at all — their
transcendental cost is one `sqrt`, which matches exactly; only the reduction
order differs, and that is inside the accepted tolerance.

Provenance note: the probe harness lived at `/tmp/zulp.zig` and is not in the
repo; this table is the record of that measurement. Re-measuring later on a
different arch/compiler may shift the counts, but the qualitative split
(`sqrt` exact / `exp`+`cos` 1-2 ulp) is what the sequencing decisions rest on.
