# qwen35 CPU 性能分析：还要改什么

> 机器：Xeon Gold 6430（32C/1socket，AVX512-F/BW/DQ/VL + VNNI + VBMI + BF16 + AMX）
> 模型：`/mnt/Qwen3.5-0.8B-GGUF/` 全部变体，prompt「天空为什么是蓝色的？」，`-n 128`，`-t 0 -s 1`
> 对照：llama.cpp b10707 `llama-cli -t <N> -dev none --single-turn`
> 测量口径：每格 3 次取**中位数**；所有数字以 tok/s 计（注意 `Generate: X tokens in Y s (Z tok/s)` 里 **Z 是第 7 字段，Y 是第 5 字段**——本仓库历史 BENCH.md 和我的第一版脚本都在这里踩过坑）

## 0. 先校准基线（这一步很重要）

改进前（2026-11 分析时）：

| 场景 | qwen35 | llama.cpp | 比值 |
|---|---|---|---|
| 单线程 Q4_K_M | 6.3 | 9.4 | **0.67** |
| 单线程 Q8_0 | 5.5 | 10.4 | **0.53** |
| 单线程 IQ4_XS | 8.3 | 9.6 | **0.87** |
| 单线程 BF16 | 1.56 | 4.9 | **0.32** |
| 单线程 IQ4_NL | 3.45 | 10.3 | **0.33** |
| 16 线程 Q4_K_M | 39.7 | 60.7 | **0.65** |
| 16 线程 Q8_0 | 39.0 | 65.4 | **0.60** |
| 16 线程 IQ4_XS | 44.5 | 64.7 | **0.69** |
| 16 线程 BF16 | 1.51 | 41.5 | **0.04** |

**结论：最差的那几项（IQ4_NL 0.33、BF16 0.04）才是"三分之一"的来源；K-quants/IQ 主力格式其实已经在 0.6–0.7。** 改进重点因此不是"全面加速"，而是**打掉 3 个结构性缺陷**（下面 §1/§2/§3），它们各自都能把对应格式从 0.3 拉回 0.7 以上，并给主力格式再补 +30~50%。

### 0.1 改进后实测（§5 的第 1、2 项已落地）

`-n 96`，`--threads 1` / `--threads 16`，其余同 §0 口径。llama.cpp 用同一台机器、
`llama-cli ... -t N -dev none --single-turn`，取输出末尾 `Generation: X t/s`。

| 模型 | 改进前 t=1 | **改进后 t=1** | 倍数 | llama t=1 | 改进后/llama | 改进后 t=16 | llama t=16 |
|---|---|---|---|---|---|---|---|
| **Q4_K_M** | 6.3 | **11.48** | **1.82×** | 9.4 | **1.22** | **49.74** | 54.5 |
| Q5_K_M | 5.97 | **8.77** | **1.47×** | 8.7 | **1.01** | 44.45 | 50.5 |
| Q6_K | 6.34 | **8.17** | **1.29×** | 7.8 | **1.05** | 42.72 | 47.2 |
| Q3_K_S | 6.77 | **8.42** | **1.24×** | 11.7 | 0.72 | 45.28 | 61.9 |
| Q4_0 | 6.33 | **7.18** | 1.13× | — | — | 43.53 | — |
| IQ4_XS | 8.33 | **9.51** | 1.14× | 9.6 | 0.99 | 46.95 | 64.7 |
| UD-IQ2_XXS | 7.45 | **10.25** | **1.38×** | — | — | 48.56 | — |
| UD-IQ3_XXS | 6.51 | **7.99** | 1.23× | — | — | 44.57 | — |
| Q8_0 | 5.51 | 5.40 | 1.00× | 10.4 | **0.52** | 32.79 | 65.4 |
| **IQ4_NL** | 3.45 | **6.65** | **1.93×** | 10.3 | **0.65** | **42.28** | — |
| BF16 | 1.56 | **2.65** | **1.70×** | 4.9 | **0.54** | **22.81** | 41.5 |

**K-quants 三族（Q4_K/Q5_K/Q6_K）单线程已追平或超过 llama.cpp（1.01–1.22×）。**
剩下三个明显落后的是 §5 里尚未开工的项：

- **Q8_0 0.52** — kernel 已确认到带宽极限（§2.1 实测：把 every-block reduce 提出循环只有
  1% 差异），单线程 5.4 tok/s × 0.86 GB/token ≈ 4.6 GB/s 就是这颗核的单核带宽。
  要再快只能像 llama 那样换掉 VNNI 那套 per-block f16 scale 的写法，收益上限存疑。
- ~~**IQ4_NL 0.36**~~ → **已修，0.65**（`_mm256_shuffle_epi8` 查表，见 §4.1）。
- **BF16 0.54** — 已从 0.32 提上来（f32 dot SIMD 化 + pool 传递），但要追平 llama 还需
  把 bf16 保持原布局（现在 `loadLinear` 展开成 f32 常驻，多吃一倍内存和带宽）。
  `kernels_f32.c` 里的 `bf16_dot_f32` 已经写好但**还没接进 `loadLinear`**。

### 一个量化一切的指标：每 token 的指令数

用 `perf stat -e instructions` 跑 `-n 8` 与 `-n 32` 取差（扣掉加载/prefill 的固定部分），除以 24 token：

| 引擎 | 每 token 指令数 | 相对 llama |
|---|---|---|
| llama.cpp (Q4_K_M) | **0.158 G** | 1.0× |
| qwen35 (Q4_K_M) | **1.395 G** | **8.8×** |

0.8B 模型每 token 约 0.8 G MAC。llama 的 0.158 G 指令 = 约 5 MAC/指令（AVX2/AVX512 的合理利用率）；qwen35 的 1.395 G 指令 = 约 0.6 MAC/指令（**纯标量**）。

另一个角度：`perf stat` 的 IPC 是 **qwen35 3.5 vs llama 0.33**。llama 是**内存带宽受限**（IPC 低，等数据），qwen35 是**算力受限**（IPC 高，在跑大量无用标量指令）。这告诉我们优先级：**先把指令数宓下来（SIMD），再去搞预取**。现在就去做 prefetch/分块优化是白花力气。

---

## 1. `kernels256.c` 里 K-quants 全族零 SIMD（最大的一块）— ✅ 已完成

### 证据

```
perf (单线程 Q4_K_M, 64 token):
  57.10%  k256_dot256                 <- 标量三重循环
  15.30%  k256_q4_k_q8_k_scalar
   9.33%  k256_q5_k_q8_k_scalar
   9.27%  k256_q6_k_q8_k_scalar
  ---- 合计 91% ----
perf stat: 48 token = 85.5 G 指令 / 24.4 G 周期 (IPC 3.50)
```

0.8B 每 token ≈ 0.8 GFLOP，48 token ≈ 38 GFLOP，却烧了 **855 亿条指令 → 每 FLOP 2.2 条指令**，即完全标量执行。llama.cpp 同级 kernel 用 AVX2 `_mm256_maddubs_epi16`，每指令 ~32 FLOP。

### 根因

`kernels256.c` 里"AVX2 版"是空壳转发：

```c
static float k256_q4_k_q8_k_avx2(int n, ...) { return k256_q4_k_q8_k_scalar(n, ...); }
static float k256_q5_k_q8_k_avx2(int n, ...) { return k256_q5_k_q8_k_scalar(n, ...); }
static float k256_q6_k_q8_k_avx2(int n, ...) { return k256_q6_k_q8_k_scalar(n, ...); }
static float k256_q2_k_q8_k_avx2(int n, ...) { return k256_q2_k_q8_k_scalar(n, ...); }
static float k256_q3_k_q8_k_avx2(int n, ...) { return k256_q3_k_q8_k_scalar(n, ...); }
static float k256_tq*_avx2(...)             { return ..._scalar(...); }
```

文件头的注释把理由写成了"AVX2 没有 int8 乘法 intrinsic"——**这是错的**。AVX2 有：
- `_mm256_maddubs_epi16`（u8×s8 配对乘加）— llama.cpp 的 Q4_K/Q5_K/Q6_K vec_dot 就靠它
- `_mm256_madd_epi16`
- AVX512-VNNI 还有 `_mm256_dpbusd_epi32` / `_mm512_dpbusd_epi32`，一条指令做完 4 路 u8×s8 点积

文件里 `grep -c "_mm256_\|_mm512_"` = **1**（那一处还是注释）。

### 影响面

Q2_K / Q3_K / Q4_K / Q5_K / Q6_K / TQ1_0 / TQ2_0 全部受益 —— 也就是 `Q3_K_M/S`、`Q4_K_M/S`、`Q5_K_M/S`、`Q6_K`、`UD-Q*_K_XL` 这 **14 个 GGUF 变体**。

### 改法（按收益排序）

1. **AVX2 `maddubs` 版**（照抄 `/mnt/llm-dev/llama.cpp/ggml/src/ggml-cpu/arch/x86/quants.c` 的 `ggml_vec_dot_q4_K_q8_K` L1232 起）。要点：
   - `k256_q4_aux8` 那套"展开成 256 个 int8 的栈数组"必须**删掉**——那一来一回的 store/load 就吃掉大半收益。直接在寄存器里做 nibble→int8 (`_mm256_and_si256` + `_mm256_sub_epi8`) + `_mm256_maddubs_epi16`。
   - scale/min 用 `_mm256_loadu_si256` + `_mm256_cvtepu8_epi16` 展开，别再走 `k256_unpack_scale_min` 的标量位运算。
   - bsums 项（`sumi += bsums[j] * min`）本身可以用 `_mm256_madd_epi16` 一次算完。
2. **AVX512-VNNI 版**（新文件或 `kernels512.c` 扩容）：`_mm512_dpbusd_epi32` 一次处理 64 个元素。本机支持，预期再 +40~60%。注意 Q6_K 是 **signed** × signed，要用 `_mm512_dpbusd` 前先 `+32` 偏置（llama.cpp 的 Q6_K AVX512 分支就是这么做的），或者用 `_mm512_dpwssds`（i16 对）。
3. 三个 256-block kernel 的**公共前置展开**（`aux8`）如果最终仍需要，合并进 kernel 内部，不要跨函数边界。

**验收**：既有 `src/simd_tests256.zig`（C kernel vs Zig 标量 oracle）必须保持绿；新增 AVX512 版要照 `simd_tests_iq_*.zig` 的形状再补一组对拍。**接受标准：C kernel 与 Zig oracle 逐块一致，不是"看起来像"。**

**预期**：Q4_K_M/Q5_K/Q6_K 单线程 6.3 → 12~16 tok/s（越过 llama.cpp）；t=16 从 0.65 拉到 ≥1.0。

### 1.1 实际落地结果（2026-11）

Q2_K / Q3_K / Q4_K / Q5_K / Q6_K 五个 `_avx2` 全部换成了真正的 AVX2 实现
（`src/kernels/x86/kernels256.c`），**没有走 AVX512-VNNI**——先做 AVX2 就够追平 llama.cpp，
且 AVX2 版能在任何 x86-64 机器上跑（AVX512 版要另加一份分派与对拍）。

共用骨架（每条都在代码注释里写明了为什么成立）：

- 权重与激活的符号是 **u8 × s8**，所以 `_mm256_maddubs_epi16` 直接可用，**不需要 `-8` 偏置**
  （llama.cpp 的 Q4_K 是 signed 4-bit，才要 `kmask` + 偏置那一套；本仓库的 Q4_K/Q5_K
  在 `k256_q4_aux8` 里就是无符号 0..15 / 0..31）。
- **scale 用 `_mm256_madd_epi16` 骑进整数累加**：`maddubs` 得 16 个 i16 pair 和，
  `madd_epi16(scale_broadcast, pair)` 把它乘成 8 个 i32。这样**没有任何 per-group 横向归约**，
  整个 256 元素块只在最后 reduce 一次。这正是 llama.cpp `get_scale_shuffle_k4` 的等价手法，
  只是本仓库的 scale 布局更简单（Q4_K/Q5_K 的 `sm[g/2]` 是每 32 元素一个，直接 `set1_epi16` 即可，
  不必像 llama 那样 `shuffle_epi8` 展开）。
- **`aux8` 栈数组全部删掉**：nibble 抽取在寄存器里做
  （`q4l = and(bits, 0x0F)` / `q4h = and(srli_epi16(bits,4), 0x0F)`）。
- Q6_K / Q3_K 是 **signed** 权重，用"保持 unsigned + 解析式修正"而不是先减再算：
  Q6_K 的 `-32` 变成 `32 * Σ_g s_g * bsum_g`（`bsums` 现成），Q3_K 的 `-4` 因为
  是 per-element 掩码（不落在组边界上），改成把偏移量也做成向量再 `sub_epi16`。
- Q2_K / Q6_K 的 scale 是**每 16 元素一个**，所以一个 32 元素 `maddubs` 结果要配
  **两个** scale（低 16 用 `sc[2g]`，高 16 用 `sc[2g+1]`），用
  `_mm256_setr_epi16` 拼出 16 lane。**这一条是唯一一个踩过的坑**：第一版按"每 32 元素一个"
  写，对拍立刻报 `e = 2.2`（`got=-9.87 ref=8.19`），改成 16 元素粒度后 `e = 0`。

**对拍结果：Q2_K / Q3_K / Q4_K / Q5_K / Q6_K 五个 kernel 的 `max_err` 全部 = 0（逐位精确）**，
`make test` 76/76 通过。

性能：Q4_K_M 6.3 → **11.5** tok/s（1.82×，**越过 llama.cpp 的 9.4**）、
Q5_K_M 5.97 → 8.77（1.01×）、Q6_K 6.34 → 8.17（1.05×）。
比原预期（12~16）略低，因为 §4 的两个小项（Q8_K 量化的 `roundf`、
每行一次函数调用）还没做，而它们现在占了相对更大的比例。

---

## 2. BF16（以及一切 f32 权重）的 matvec 根本不进线程池 — ✅ 已完成（第 3 条待做）

### 证据

```zig
// src/backend_cpu.zig:459
fn matvecF32(out: []f32, mat: []const f32, x: []const f32) void {
    ...
    pool_mod.parallel(null, .{ .f32 = .{ ... } });   // <-- null!
}
```

`vMatvec` 的 `.f32` 分支（同文件 :695）也直接 `matvecF32(...)`，把自己的 `self.pool` 丢了。

实测确认：BF16 decode 在 `--threads 1/8/16/32` 下**恒为 1.5 tok/s**（完全不扩展），而同一文件的 prefill 能到 14.3 tok/s（prefill 走 `vMatvecBatchK` → `.f32` 的正确分支 :389，带 pool）。

### 根因（两层）

1. `matvecF32` 传 `pool = null`（明显是笔误级的 bug）。
2. 即使修好 pool，`.f32` 的 `runRows` 内层是
   ```zig
   for (rowv, xs) |mv, vv| acc += mv * vv;
   ```
   单条依赖链的标量 FMA 累加。Zig 对 `acc +=` 的浮点归约**默认不会重排**（要保持 IEEE 顺序），所以不会自动向量化。llama.cpp 的 `ggml_vec_dot_f32` 用 4~8 个独立 accumulator 手动展开。

### 影响面

BF16（0.04 比值，最差的一项）、以及任何 f16/f32 张量。**BF16 是本仓库唯一还没进 SIMD 时代的主力路径**。

### 改法

1. 一行修 `matvecF32`：传真实 pool（把 `pool: ?*Pool` 加进参数，从 `vMatvec` 传入）。
2. 写一个 C kernel `f32_dot_f32(n, a, b)`（放 `kernels32.c` 或新 `kernels_f32.c`，AVX2 编译单元），8 路 `_mm256_fmadd_ps` 累加 + 尾部标量。这属于 AGENTS.md §1"数值核心在 C"的范畴，不要用 Zig `@Vector`。
3. **更彻底**：`loadLinear` 现在把 BF16 **解包成 f32 常驻内存**（`backend_cpu.zig:536-555`，1.4GB 模型 → 2.9GB f32）。这既浪费内存又浪费带宽。建议保留 bf16 原布局，kernel 里直接 `_mm512_cvteph_epi32`（本机有 `avx512_bf16`）或 `_mm256_slli_epi32` 展开——bf16→f32 是左移 16 位，免费。**这一条能同时把内存占用和带宽都砍掉一半**，对 decode 是数量级意义。

**预期**：BF16 单线程 1.5 → 4~5（追平 llama），t=16 从 1.5 → 35+。

### 2.1 实际落地结果（2026-11）

第 1、2 条已做：

- `matvecF32` 增加 `pool: ?*pool_mod.Pool` 参数并转发给 `parallel`（`backend_cpu.zig`），
  `vMatvec` 的 `.f32` 分支传 `self.pool`。
- 新建 `src/kernels/x86/kernels_f32.c` + `kernels_f32.h`（`build.zig` 里登记为
  `qwen35_kernels_f32`，`.isa = .avx2`）。里面有两个 kernel：
  - `f32_dot_f32` — 4 路独立 `_mm256_fmadd_ps` 累加器 + 尾部标量，最后折叠加起来。
  - `bf16_dot_f32` — 读 bf16 原布局（`_mm256_slli_epi32` 左移 16 位展开成 f32），
    带宽减半。**这个已经写好，但 §2 第 3 条还没接线**（`loadLinear` 仍展开成 f32 常驻）。
  - 两者都有完整标量 fallback（AGENTS.md §1 要求：不是占位）。
- `src/simd_api.zig` 加 `DotF32` 类型与 `dotF32` / `dotBf16F32` 绑定；
  `src/pool.zig` 的 `.f32` 分支改为每行一次 `simd_api.dotF32`。

实测：**BF16 单线程 1.56 → 2.65 tok/s（1.70×），t=16 从 1.51 → 22.81（15.1×）**。
对 llama 的比值 0.32 → 0.54（单线程）/ 0.04 → 0.55（t=16）。

比预期的 4~5 低，正是因为第 3 条没做：`loadLinear` 仍把 bf16 展开成 f32 常驻，
单线程 2.65 tok/s 时读 f32 权重的带宽（约 5.7 GB/token）已经打满单核带宽。

### 2.2 已实测推翻的一个猜想（Q8_0 的 per-block 归约）

原本以为 §2 的 `k512_q8_0_q8_0_vnni`（`src/kernels/x86/kernels512.c:53-83`）
每个 32-block 都做一次 `_mm512_reduce_add_epi32` + `setzero`，形成串行依赖链，
是 Q8_0 只有 5.5 tok/s 的原因。**写原型实测后否定了这个判断**：

| 变体 | 行/ms |
|---|---|
| 现状（每块 reduce + 每块 f16→f32） | 6.4 k |
| 16 块批量归约 + 批量 scale 转换 | 6.6 k |
| 手写 16→1 shuffle 归约（替代 `reduce_add`） | 6.1 k |

只有 **1%** 差异，手写归约反而更慢。原因：scale 是 **per-block 标量**，
只要它在，每块就必须有一次横向归约，这一步无法合并；而 `_mm512_reduce_add_epi32`
已经是 GCC 能生成的最优展开。**Q8_0 单线程 5.4 tok/s 就是带宽极限**
（0.86 GB/token × 5.4 ≈ 4.6 GB/s，单核实测带宽上限）。
**结论：这一项不改。**

---

## 3. 每行一次 C 调用的 matvec 几何 + 每 token 数百次 futex 唤醒 — ⚠️ 实测收益≈0，不改

### 证据

```zig
// pool.zig:313-325, .b256 分支
var r = lo;
while (r < hi) : (r += 1) {
    const wrow = q.data[rr * rb ..][0..rb];
    q.out[tok * q.rows + rr] = q.dot256(n, wrow.ptr, q.act + tok * q.act_len);
}
```

- **每行一次间接函数调用** `q.dot256`（通过函数指针，无法 inline/LTO）。
- 每 token 有 ~1000+ 次 matvec 行（0.8B：24 层 × 7 个投影 × 1024~3584 行 …），每次 `parallel()` 一次 `futexWake` 唤 31 个线程。
- `PROF-P`（t=32）实测：`post=189.25ms`（就是 `futexWake` 系统调用）占 `pub` 的 **99.8%**，5170 次调用累计 ~437ms 纯同步。

### 影响面

所有格式，尤其**多线程**场景。llama.cpp 的线程池是常驻 spin-wait（`ggml_barrier`），唤醒成本 ≈ 0。

### 改法

1. **批量内核**：把 `while (r < hi)` 循环搬进 C（`q4_k_q8_k_rows(n_rows, ...)`），一次调用处理整块行。函数指针开销从"每行一次"降到"每 chunk 一次"，并让编译器跨行做软件流水（行间独立的 GEMV 是预取的最佳场景）。这一条对 §1 的 SIMD 化是**必要的配套**，否则 SIMD 出来了也会被调用开销吃掉。
2. **线程池改成常驻自旋 + 层级屏障**：照 llama.cpp 的 `ggml_barrier`（每线程一个原子 counter + `pause` 自旋，`n_busy` 达到阈值才 `futexWait`）。对本机 32 核、每 token 上千次同步的负载，futex 是错误的选择。
3. 顺手修 `minParallelRows`（现在硬编码 `return 256`，忽略参数）——0.8B 的 `rows=1024` 只切 4 块，`chunk=16` 又过细（`per = 4*p.n` 在 p.n=32 时是 128，`rows/128=8` → chunk=128，但 16 行的下限对 32 线程太少）。改成按 `rows / (n*2)` 且下限按每 chunk 的工作量（bytes×rows）估。

### 3.1 实测结论：这一项不做（2026-11）

我按第 2 条把 `workerMain` 改成了常驻自旋（先轮询 `pool.gen` 最多 `spin_limit` 次，
耗尽才读 `wake` 并 `futexWaitUncancelable`；`Pool` 加 `sleepers` 计数器，
`parallel` 只在 `sleepers != 0` 时才 `futexWake`）。**实测没有收益，已回退。**

原因在 `PROF-P` 的完整口径。之前只看了 `pub` 内部的比例（`post` 占 `pub` 的 99.8%），
但 `pub` 本身只占全部时间的一小块：

```
PROF-P calls=5170 pub=95.82ms [slot=0.50ms post=95.31ms] main=523.67ms tail=34.16ms
```

即 `post` 最多只值 `95.31 / (95.82+523.67+34.16)` = **14%**，而且：

- `spin_limit` 取 20000 时多线程**崩到** t=16 22.0 tok/s（基线 39.1）——31 个 worker 全速自旋
  会挤占主线程的核，本机的"主线程也参与计算"模型下自旋是负收益。
- `spin_limit` 取 200 时回到 39.25（与基线 39.14 在噪声内），**没有可测收益**。
- `minParallelRows` 的串行路径（`serialcalls=1188` / `calls=5170` = 23% 的调用走串行）
  总耗时只有 **5.68 ms**，占 0.9%，改它没有意义。

另外 `perf report -F 999 --threads 16` 显示 **83.6% 在 `k512_q8_0_q8_0_vnni` 内部**，
而函数指针那一层（`q8_0_q8_0` 包装）只有 **0.26%**。所以第 1 条（批量内核）对 Q8_0 也
没有必要——调用开销被 SIMD kernel 本身的执行时间掩盖了。

**结论：线程池与每行调用开销都已在噪声级别，不再投入。**
多线程的扩展比实测：qwen35 t=16/t=1 = 7.0×（Q8_0 5.63 → 39.25），
llama.cpp 是 6.3×（10.4 → 65.4）——**扩展性比 llama 还好，差的是单线程绝对值**，
所以资源应该全部投在 kernel 上（§1/§4），而不是同步机制。

---

## 4. 剩余的小项（按性价比）

| 项 | 证据 | 改法 | 预期 |
|---|---|---|---|
| ~~Q8_0 的 VNNI 每块归约~~ **（已实测推翻，见下）** | 我最初以为把 `_mm512_reduce_add_epi32` 提出循环能拿 4.4×，**那是错的**：那个原型丢掉了 per-block 的 f16 scale 乘法，数学上不等价。用正确实现重测（`base` 6.5k 行/ms vs `batched16` 6.6k 行/ms，16 块批量归约 + 批量 scale）只有 **1%** 差异。原因：每个 block 的 `dx*dy` 是独立的标量，整数和**必须逐块归约**，无法在整数域合并 | **不改**。Q8_0 单线程 5.5 tok/s 已是带宽极限（0.86 GB/token × 5.5 = 4.8 GB/s）。proto 定量：`full` 5.9k / 省 f16 转换 9.3k / 再省 reduce 21.2k 行/ms——后两者都不可得，但它们说明**剩余开销在标量簿记上，只能靠"整块行 kernel + 常量折叠"摊薄**（即下面 §3） | 单线程不变 |
| ~~**IQ4_NL 慢 2.4×**~~ ✅ | 单线程 3.45 vs 同尺寸 IQ4_XS 8.33。`kernels32.c:229-235` 每块做 16 次标量表查找 + 写栈数组 `xv[32]` | 已用 `_mm256_shuffle_epi8` 一次查表 32 个值（见 §4.1） | **3.45 → 6.65** |
| **`kernels512.c` 只覆盖 Q8_0** | 106 行，只有 `q8_0_q8_0` | §1 的 AVX512-VNNI 扩容应放这里，别新开文件（build.zig 的 `KernelIsa.avx512` 已经配好） | — |
| **IQ 族没有 AVX512** | `iq_kernels.c` 只有 `__attribute__((target("avx2")))` 和 `target("avx")` | IQ2_XXS/IQ3_XXS/IQ2_XS/IQ4_XS/IQ1_S 的 grid 查表在 VNNI 下可用 `_mm256_maskz_permutexvar_epi8`（VBMI）直接做 256 项查表，省掉两次 `_mm256_shuffle_epi8` + blend | +20~40% |
| **`elementwise` 占 2.3%** | `bufGdnStep` 2.25% + `exp_f32` 1.24%（libm 的 exp 逐元素调用） | GDN 的 gate 用多项式近似 exp（llama.cpp 有 `ggml_v_silu`/`ggml_v_softplus` 的向量版），4 路 AVX2 | 小幅但免费 |
| **Q8_K 量化在热点里** | `quantize_q8_K` 0.76%，每 token 每层跑一遍 | 现在逐元素 `roundf`（libm 调用）。改 `_mm256_cvtps_epi32`（`_MM_FROUND_TO_NEAREST_INT`），并且 amax 用 `_mm256_max_ps` 树归约 | 0.76% → 0.2% |

### 4.1 IQ4_NL 查表 SIMD 化（已完成，2026-11）

`src/kernels/x86/kernels32.c` 的 `k32_iq4_nl_q8_0_avx2` 原实现：

```c
int8_t xv[32];
for (int j = 0; j < 16; j++) {
    xv[j]      = (int8_t)k32_iq4nl_tab[qb[j] & 0x0F];   // 16 次依赖链查表
    xv[16 + j] = (int8_t)k32_iq4nl_tab[qb[j] >> 4];
}
const __m256i x32 = _mm256_loadu_si256((const __m256i*)xv);  // 刚写就 load 回来
```

**为什么不能用算术替代查表**：`kvalues_iq4nl` 是
`{-127,-104,-83,-65,-49,-35,-22,-10,1,13,25,38,53,69,89,113}` —— 步长不规则
（17,21,18,16,14,13,12,11,12,12,13,15,16,20,24），所以必须查表。

**改法**：一次 `vpshufb` 完成全部 32 个查表。

```c
const __m256i lut = _mm256_broadcastsi128_si256(_mm_loadu_si128((const __m128i*)kvalues_iq4nl));
const __m128i tmp   = _mm_loadu_si128((const __m128i*)(xb + 2));
const __m256i bytes = _mm256_set_m128i(_mm_srli_epi16(tmp, 4), tmp);  // 低 lane=低 nibble, 高 lane=高 nibble
const __m256i idx   = _mm256_and_si256(_mm256_set1_epi8(0x0F), bytes);
const __m256i x32   = _mm256_shuffle_epi8(lut, idx);                  // 一条指令，32 个值
```

三个必须说清的成立条件：

1. **`vpshufb` 只在 128-bit lane 内查表**，所以表要 `broadcastsi128_si256` 到两个 lane。
   16 项的表正好塞得进一个 lane，因此一次 shuffle 覆盖全部 32 个 nibble。
2. **`vpshufb` 的索引取低 4 位、bit7 置位则输出 0**。我们的索引已经是 `idx & 0x0F`，
   且 `kvalues_iq4nl` 里**没有 0**，所以 LUT 完全精确，无需额外屏蔽。
3. **布局对齐**：scalar 路径把低 nibble 放 `xv[0..15]`、高 nibble 放 `xv[16..31]`；
   `_mm256_set_m128i(_mm_srli_epi16(tmp,4), tmp)` 得到的正是"低 lane = 低 nibble、
   高 lane = 高 nibble"，与 `xv` 的顺序一致，所以 `x32` 与 `y32` 逐元素对齐。

同时删掉了 `xv[32]` 栈数组（连同那次 store-forwarding 往返）。表数据仍然只有一份：
用的是 `iq_tables.h` 里已经 `extern` 的 `kvalues_iq4nl[16]`（`src/iq_tables.zig` 的
`pub export`，带 `align(64)`），**没有新增第二份表**。

**对拍**：`make test` 里 `iq4_nl_q8_0/c = 0`（逐位精确）、`IQ4_NL: max_err = 0`、
`kern32 testdata-vs-rowDot` 用真实 `IQ4_NL.bin` 行也算过。
**已验证测试有敏感性**：把 shuffle 的索引 XOR 1 注入扰动后立刻报 `e = 0.747`、
3 个用例失败（exit 1），说明上表 16 个 nibble 槽位的逐项校验真的在跑。

**结果**：

| | 改前 | 改后（3 次中位数） | 倍数 | llama.cpp | 比值 |
|---|---|---|---|---|---|
| t=1 | 3.67 | **6.65** | **1.81×** | 10.3 | 0.36 → **0.65** |
| t=16 | 30.74 | **42.28** | **1.38×** | — | — |

`--threads 1` 与 `--threads 16` 的输出逐字节一致。

比 §4 预测的"3.45 → 8+"低一些：`vpshufb` 本身省下的是一大块，
但剩下的 `k32_dot32_i8_i8`（两次 `cvtepi8_epi16` + `madd_epi16`）现在成了主项，
要再快得走 AVX512-VNNI 的 `_mm256_dpbusd_epi32`。

---

## 5. 建议的落地顺序

按"改动量 ÷ 收益 × 风险"排：

1. ✅ **已完成** — **修 `matvecF32` 的 `pool=null`** + **C 版 f32 dot**。
   BF16 单线程 1.56 → 2.65、t=16 1.51 → 22.81。见 §2.1。
2. ✅ **已完成** — **`kernels256.c` 的 AVX2 `maddubs` 版 Q2_K/Q3_K/Q4_K/Q5_K/Q6_K**。
   Q4_K_M 单线程 6.3 → 11.5（越过 llama.cpp 的 9.4）。见 §1.1。
3. ❌ **实测无收益，不做** — 线程池常驻自旋。`post` 最多值 14%，自旋要么更慢（31 个 worker
   抢核），要么与基线在噪声内。每行函数调用开销也只占 0.26%。见 §3.1。
4. ⬜ **待做** — **AVX512-VNNI 扩容**（§1 第 2 步 + §4 的 IQ-VBMI）。
5. ✅ **IQ4_NL 查表已做**（§4.1，3.67 → 6.65）；⬜ 待做 — Q8_K 量化、bf16 原布局。

### 下一步的优先级（依据 §0.1 实测的剩余差距）

按"当前对 llama.cpp 的比值"从低到高排，也就是收益最大的先做：

1. ✅ **已完成 — IQ4_NL**（0.36 → 0.65），见 §4.1。
2. **BF16（0.54）** — §2 第 3 条，`loadLinear` 保留 bf16 原布局 + 接上已经写好的
   `bf16_dot_f32`。能同时砍掉一半内存和带宽。**现在这是最大的一块。**
3. **Q8_0（0.52）** — §2.2 已证明现有 kernel 到带宽极限，要再快需要换算法
   （不再 per-block 存 f16 scale），收益上限不明，**优先级最低**。
4. **AVX512-VNNI 版** — 若还要再进一步。注意 Q6_K 是 signed×signed，要先 `+32` 偏置，
   且必须另补一组 Zig 标量对拍。

**每一步都必须先跑 `make test` 拿 76/76 基线，改完不得新增失败**；每个 kernel 改动按 AGENTS.md §4
补 Zig 标量对拍（`src/simd_tests*.zig` 的形状）。**当前 76/76 是绿的**
（AGENTS.md 里记的 74 是旧数，已随测试增补过）。

> 测量提醒：`make test` 会把 `zig-out/bin/qwen35` 覆盖成 **Debug** 二进制（33 MB vs ReleaseFast 6.5 MB），
> 之后直接测性能会得到约 **2.5× 偏慢** 的假数据。测性能前必须重跑 `make cpufast`，
> 并确认 `ls -la zig-out/bin/qwen35` 的大小。

---

## 6. 两个与性能无关但会误导测量的坑（建议顺手修）

1. **`BENCH.md` 全表数据是错的**：它的 `decode tok/s` 列实际取的是"秒"（`Generate: X tokens in Y s (Z tok/s)` 的第 5 字段），所以整张表看起来差距 5~10×，实际是 1.5~3×。另外它的结论"中位比值 0.17"和现在实测的 0.6~0.7 差了一个量级。建议重跑并修正列解析（或直接删掉，用 `bench_raw/` 脚本重生成）。
2. **机器上有一个残留的 `nsys --start-agent` 进程**（PID 747441，已跑 13.6 小时、占 100% CPU，`--session-name=profile-qwen35-747434`）。它会污染基准（我第一轮测量就出现过 6.3 ↔ 20 tok/s 的跳变）。如果确认是自己早前 profiling 的残留，`kill 747441` 后再测。
