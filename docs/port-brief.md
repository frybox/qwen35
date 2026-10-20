# llama.cpp CPU 量化内核移植简报（worker 共用锚点）

目标：把 llama.cpp 的 CPU 量化点积内核移植成**纯 Zig `@Vector`**（禁 C/C++、禁 inline asm），
接入 qwen35 的三 ISA tier 分层编译，加速各量化类型的 matvec。

## 0. 不要探索，直接开工

本文件已给出全部锚点。允许你 `read` 列出的具体行号区间，**不要**做全仓库 grep 巡览。

## 1. 分层编译机制（已搭好，勿改）

- `build.zig` 把 `src/simd/` 逐文件复制进 build cache 的 `baseline/`、`avx2/`、`avx512/` 三个子目录，
  各自用不同 `-mcpu` 编成三个 module，注入主 module 的名字是
  `@import("simd_baseline")` / `@import("simd_avx2")` / `@import("simd_avx512")`。
- 每个 tier 目录里 `tier.zig` 由 build.zig 生成（`pub const tier = "avx512"` 等 + `isAvx512` 等 comptime 常量）。
  **`src/simd/` 下故意没有 tier.zig**（复制文件与生成文件同路径会冲突，复制版会赢，导致三 tier 全变 baseline）。
- 三档 `-mcpu`：
  - baseline: `+ssse3,+f16c`
  - avx2: `+avx2,+fma,+f16c`
  - avx512: `+avx512f,+avx512bw,+avx512vl,+avx512dq,+fma,+f16c`（**故意不含 avx512vnni**：本机 Xeon Gold 6430 有 AVX-512 但无 VNNI，开了 LLVM 会发 `vdpbusd` → SIGILL。内核里不要用 vdpbusd。）
- 同一份 `@Vector` IR，LLVM 自动在 avx2 档展成 ymm、avx512 档展成 zmm。**内核里不需要写 tier 分支**，除非要写 AVX-512 专属的 64 宽循环（用 `if (comptime isAvx512) { ... }`）。

## 2. 内核签名约定

```zig
pub fn q4_k_q8_k(n: usize, vx: [*]const u8, vy: [*]const u8) callconv(.c) f32
```
- `n` = 元素数；`vx` = 打包权重；`vy` = 量化后的激活。
- 32 元素超块族：`vy` 是 **Q8_0** 块数组（2 字节 f16 scale + 32×i8 = 34 字节 / 32 元素）。
- 256 元素超块族：`vy` 是 **Q8_K** 块数组（4 字节 f32 scale + 256×i8 = 260 字节 / 256 元素）。
- 只实现 `nrc == 1`（单行）情形，与 llama.cpp 一致。
- 每个 SIMD 内核配一个标量参考版 `pub fn <name>_ref(...)`，照 C 的 `*_generic` 逐行翻译。

## 3. 可用助手（`src/simd/util.zig`，只读复用，缺什么在内核文件内自建 local helper）

| 助手 | 用途 |
|---|---|
| `load(T, ptr)` / `loadF32(N, ptr)` / `storeF32(ptr, v)` | 非对齐载入/存出 @Vector |
| `u16At(p)` / `u32At(p)` / `f16(p, off)` / `f32ToF16Bits(x)` | 小端标量取数、f16↔f32 |
| `f16x8(@Vector(8,u16))` / `f16x4` / `f16x2` | 批量 f16→f32（AVX 上是 vcmpps） |
| `dot32i8` / `dot16i8` / `dot64i8` | i8·i8 点积（vpmaddwd + 水平加） |
| `dot32u8i8` / `dot16u8i8` | u8·i8 点积（vpmaddubsw 语义） |
| `sumI16` / `sumI16x32` / `sumI16x64` | i16 向量水平加到 i32 |
| `niblo/nibhi`（→f32）、`nibloI8/nibhiI8`（→i8）、`niblo8/nibhi8`、`nibloI8_8/nibhiI8_8` | 半字节展开 |
| `bit(v, comptime pos)` | 取位 |
| 类型别名 | `V8u8 V16u8 V16i8 V32i8 V32u8 V64i8 V64u8 V8f32 V16f32 V16i16 V16i32 V32i16 V32i32` |

**Zig 0.17 的坑**（都实测踩过）：
- 没有 `@slice` / `@sliceElements` / `@extend` / `@setFunctionMCU` / `@optimizeFor`。
- 拆向量：`const arr: [N]T = @bitCast(v); const lo: @Vector(M,T) = @bitCast(arr[0..M].*);`
- 加宽：`@as(@Vector(N, i32), v16i16)`（i16→i32 直接 `@as`）。
- `@ptrCast` 到 `@Vector` 要给结果类型：`@as(@Vector(N,T), @ptrCast(p))`；更稳的是 `@bitCast(@as(*align(1) const [N]u8, @ptrCast(p)).*)`。
- `std.io` 不存在（是 `std.Io`）；`std.testing.io` 是测试里的 io 句柄。

## 4. llama.cpp 源锚点

`/mnt/strata/llama.cpp/ggml/src/ggml-cpu/quants.c` —— **`*_generic` 标量参考（语义基准）**：

| 内核 | 行 |
|---|---|
| q1_0_q8_0_generic | 127 |
| q2_0_q8_0_generic | 177 |
| q4_0_q8_0_generic | 225 |
| q4_1_q8_1_generic | 262 |
| mxfp4_q8_0_generic | 298 |
| nvfp4_q8_0_generic | 330 |
| q5_0_q8_0_generic | 365 |
| q5_1_q8_1_generic | 408 |
| q8_0_q8_0_generic | 451 |
| tq1_0_q8_K_generic | 481 |
| tq2_0_q8_K_generic | 533 |
| q2_K_q8_K_generic | 565 |
| q3_K_q8_K_generic | 617 |
| q4_K_q8_K_generic | 696 |
| q5_K_q8_K_generic | 771 |
| q6_K_q8_K_generic | 851 |
| iq2_xxs_q8_K_generic | 906 |
| iq2_xs_q8_K_generic | 948 |
| iq2_s_q8_K_generic | 998 |
| iq3_xxs_q8_K_generic | 1050 |
| iq3_s_q8_K_generic | 1094 |
| iq1_s_q8_K_generic | 1150 |
| iq1_m_q8_K_generic | 1193 |
| iq4_nl_q8_0_generic | 1254 |
| iq4_xs_q8_K_generic | 1283 |

`/mnt/strata/llama.cpp/ggml/src/ggml-cpu/arch/x86/quants.c` —— **AVX2/AVX 手写 SIMD 版（抄它的向量化思路）**：

| 内核 | 行 |
|---|---|
| q1_0 | 555 |
| q2_0 | 734 |
| q4_0 | 765 |
| q4_1 | 923 |
| mxfp4 | 982 |
| nvfp4 | 1068 |
| q5_0 | 1206 |
| q5_1 | 1286 |
| q8_0 | 1372 |
| tq1_0 | 1440 |
| tq2_0 | 1572 |
| q2_K | 1638 |
| q3_K | 1830 |
| q4_K | 2102 |
| q5_K | 2280 |
| q6_K | 2490 |
| iq2_xxs | 2724 |
| iq2_xs | 2842 |
| iq2_s | 3139 |
| iq3_xxs | 3324 |
| iq3_s | 3448 |
| iq1_s | 3658 |
| iq1_m | 3777 |
| iq4_nl | 3984 |
| iq4_xs | 4068 |

块结构体定义：`/mnt/strata/llama.cpp/ggml/src/ggml-common.h`（`block_q4_0` / `block_q8_K` / `block_iq1_s` …）。
码本表原始数据：`/mnt/strata/llama.cpp/ggml/src/ggml-impl.h`、`ggml/src/ggml-cpu/vec.h`（`mul_sum_i8_pairs_float` 等 intrinsic 组合）。

## 5. qwen35 侧锚点

`src/quant_dequant.zig`（标量语义基准 + 对拍数据）：

| 函数 | 行 |
|---|---|
| dequantQ4K 48 / dequantQ4_0 72 / dequantQ4_1 86 | |
| rowDotQ4_0 115 / rowDotQ4_1 133 / rowDotQ4_0_scalar 152 / rowDotQ4_1_scalar 166 | |
| dequantQ5K 182 / dequantQ6K 216 / dequantQ3K 246 | |
| dequantIQ4XS 318 / dequantIQ4NL 341 / dequantIQ3S 356 / dequantIQ3XXS 417 | |
| rowDotQ4K 453 / rowDotQ5K 478 / rowDotQ6K 514 / rowDotQ3K 546 | |
| rowDotIQ4XS 616 / rowDotIQ4NL 641 / rowDotIQ3S 658 / rowDotIQ3XXS 712 | |
| dequantIQ2XXS 745 / rowDotIQ2XXS 764 / dequantIQ2XS 787 / rowDotIQ2XS 809 | |
| dequantIQ2S 836 / rowDotIQ2S 859 / dequantQ2K 886 / rowDotQ2K 915 | |
| dequantIQ1S 948 / rowDotIQ1S 968 / dequantIQ1M 992 / rowDotIQ1M 1032 | |

- 码本表：`src/ggml_tables.zig`（`iq3s_grid[512]u32`、`kmask_iq2xs[8]u8`、`ksigns_iq2xs[128]u8`、`iq3xxs_grid[256]u32`）；
  `src/ggml_tables_extra.zig`（`iq2xxs_grid[256]u64` 7、`iq2xs_grid[512]u64` 43、`iq2s_grid[1024]u64` 111、`iq1s_grid[2048]u64` 243）。
  同 module 内 import 写法：`const ggml_tables = @import("ggml_tables.zig");`
- **缺表**：`iq1m_grid[2048]u64`、`ksigns64[128]u8`（IQ1_M 用）。补进 `ggml_tables_extra.zig`，用脚本从 llama.cpp 源抽，别手抄。
- 对拍数据 `testdata/udtest/`：`A.f32`、`A_x.f32`（256 个 f32），以及
  F16/IQ3_S/IQ3_XXS/IQ4_NL/IQ4_XS/Q3_K/Q4_0/Q4_1/Q4_K/Q5_K/Q6_K/Q8_0 的 `.bin` + `_expected.f32`（96 行 × 256 元素）。
- 现有测试：`src/quant_dequant.zig:1120`（rowDot vs dequant 点积）、`:1169`（vs C ground-truth）。
  跑：`cd /mnt/llm-dev/qwen35 && zig build test`。基线输出末行 `worst overall = 0`。
  （注意：`zig build test` 末尾会打印一行 `failed command: ... --listen=-`，这是测试进程与 `--listen=-` 协议交互的既有现象，改动前就有，**不是**失败标志；以测试打印的 `worst overall` 为准。）
- 激活量化器已写好：`src/simd/act.zig` 的 `quantizeQ8_0`（34 字节/32 元素）与 `quantizeQ8_K`（260 字节/256 元素）。
  规则：`scale = amax/127`（Q8_0 存 f16，Q8_K 存 f32），`q = round(x/scale)` 夹到 [-128,127]，`amax==0 → scale=1, q=0`。
  与 `src/engine.zig:242 matvecQ8_0` 的内联量化完全一致。
- 引擎热路径（集成阶段用，移植阶段**不要动**）：
  - `src/engine.zig:124 linMatvec` → `.q8 => matvecQ8_0`（Q8_0 权重）、`.quant => matvecQuant`（K/IQ 权重，当前直接吃 f32 激活）、`.f32 => matvec`。
  - `src/engine.zig:535 matvecQuant`：逐 256 块调 `rowDot(quantBlock(w,r,b), x[b*256..])`。
  - `src/engine.zig:500 quantRowBytes`：q4_k 144 / q5_k 176 / q6_k 210 / q3_k 110 / iq4_xs 136 / iq4_nl 144 /
    iq3_s 110 / iq3_xxs 98 / iq2_s 82 / iq2_xxs 66 / q2_k 84 / iq2_xs 74 / iq1_s 50 / iq1_m 56。
  - `src/engine.zig:143 q8ScratchBytes` / `:160 maxQ8Cols`：Q8_0 激活 scratch 按最宽输入算。
  - 已支持加载的类型：q8_0, q4_0, q4_1, q4_k, q5_k, q6_k, q3_k, iq4_xs, iq4_nl, iq3_s, iq3_xxs, iq2_s, iq2_xxs, q2_k, iq2_xs, iq1_s, iq1_m。
    **未支持**：q5_0, q5_1, q1_0, q2_0, tq1_0, tq2_0, mxfp4, nvfp4（集成阶段要补加载 + 行字节表）。

## 6. 验证命令

```bash
cd /mnt/llm-dev/qwen35
Z=/home/wisedu/software/zig-x86_64-linux-0.17.0/zig
$Z build -Doptimize=ReleaseFast          # 全量（三 tier 都编）
$Z build test                            # 现有 + 新测试
$Z build-obj -Ofast -target x86_64-linux -mcpu baseline+avx2+fma+f16c src/simd/kernXX.zig   # 单文件快验
/usr/bin/objdump -d zig-out/bin/qwen35 | grep -c zmm    # 确认 AVX-512 代码进二进制
```
`zig objdump` 不可用（只打印 TODO），用系统 `/usr/bin/objdump -d`。

---

## 7. 已跑通的模板（照它写，不要自创风格）

`src/simd/kern32.zig` 已含 8 个内核（q8_0/q4_0/q4_1/q5_0/q5_1/q1_0/q2_0/iq4_nl × Q8_0），
**三 tier 与标量 `_ref` 误差 0，真实数据行 vs 引擎 rowDot 误差 3.2e-6**。它是唯一范例，照它的结构写：

1. 文件头注释（来源 + 签名说明）
2. `const util = @import("util.zig"); const tier = @import("tier.zig");`
3. 文件内 local helper（`qi8` 取有符号字节、`ylo/yhi` 拆 Q8_0、`signs32/q2vals/qhLo/qhHi` 位展开）
4. 每个内核：`pub fn name(...)` SIMD 版 + `pub fn name_ref(...)` 标量版，注释写 C 函数名与行号

**位展开技巧**（Q1_0/Q2_0/Q5_0 的 bit 抽取，已验证可编译且 LLVM 出向量码）：
用 `@shuffle(T, bytes, undefined, comptime_index_array)` 把字节铺到多 lane，
再 `>> @as(@Vector(N, u3), @bitCast(shift_array))` 取位。索引数组写成
`const spread4: [32]i32 = blk: { ... break :blk v; };`（**顶层 const 里不要再写 `comptime`**，会报 redundant comptime）。

## 8. Zig 0.17 新增实测坑（都踩过）

- `usingnamespace` **已移除**（util.zig 里已删）。
- `f16` 是原生类型名，**函数不能叫 `f16`**（util 里的取数函数已改名 `f16At`）。
- **不能用运行时下标索引 `@Vector`**（`v[k]` 且 k 是运行时值 → "vector index not comptime known"）。
  要逐元素处理就先 `const a: [N]T = @bitCast(v);` 再索引数组。
- `@eql` 不存在；比较 comptime 字符串用 `std.mem.eql(u8, ...)`。
- `@shuffleVectors` 不存在，是 `@shuffle(T, a, b, indices)`。
- `std.simd.deinterlace(comptime count, vec)` 要两个参数。
- 顶层 `const x = blk: {...}` 里不要写 `comptime blk`。
- 测试里 `TESTDATA ++ c.file` 要求 `c.file` comptime-known；运行时循环里拼路径用
  `std.fmt.allocPrint(alloc, "{s}{s}", .{ TESTDATA, c.file })`。
- 文档注释 `///` **不能贴在 `test` 上**（"documentation comments cannot be attached to tests"），用 `//`。
- 测试里遍历 tier 用 `inline for (.{ .{.name=..,.m=simd_base}, ... })`，
  把 module 当**类型**传进 `comptime m: type` 参数，再 `@field(m.dot32, name)` 取内核。
- 累加器**必须在循环里保持向量**，循环外只做一次 `@reduce`；
  在循环里对每个块 `@reduce` 会让 LLVM 生成大量标量搬运。

## 9. 单文件编译检查（不要跑全量 build，太慢）

```bash
TIER=avx2 /tmp/chk.sh kern256.zig     # 复制 src/simd/*.zig 到 /tmp/simdchk + 生成 stub tier.zig
TIER=avx512 /tmp/chk.sh kerniq.zig
```
`/tmp/chk.sh` 已存在（若丢失：复制 `src/simd/*.zig` 与 `src/ggml_tables*.zig` 到临时目录，
写一个 stub `tier.zig`（内容见 build.zig 的 `tierSource`），
再 `zig build-obj -Ofast -target x86_64-linux -mcpu <feats> 文件名.zig`）。
`-mcpu` 三档：baseline `x86_64+ssse3+f16c` / avx2 `x86_64+avx2+fma+f16c` / avx512 `x86_64+avx512f+avx512bw+avx512vl+avx512dq+fma+f16c`。

## 10. 测试注册（避免并行冲突，已预先做好）

`src/root.zig` 已含 `test { _ = @import("simd_tests32.zig"); }`。
**组B/组C 不要改 `src/root.zig`**：主流程会在你们完成后统一加两行。
测试文件写法照 `src/simd_tests32.zig`（tier module 由 build.zig 注入，裸 `zig test` 跑不了，
用 `/tmp/chk.sh` 验编译，语义正确性由主流程跑 `zig build test` 验收）。

## 11. 对拍基准的正确姿势

`testdata/udtest/*.bin` 的行字节是 **256 元素**（Q4_0 144 / Q4_1 160 / Q8_0 320 / Q4_K 144 /
Q5_K 176 / Q6_K 210 / Q3_K 110 / IQ4_XS 136 / IQ4_NL 144 / IQ3_S 110 / IQ3_XXS 98 / IQ2_* 见 engine.zig）。
与引擎 `rowDot*`（吃 f32 激活）对比时，**必须把 Q8_K/Q8_0 激活反量化成精确 f32 再喂给 rowDot**，
否则激活量化误差会污染内核误差（实测从 3e-6 恶化到 3.6e-2）。反量化：
`xq[b*32+j] = d_b * (i8)qs[b*34+2+j]`（Q8_0，d 是 f16）；Q8_K 同理，d 是 f32、块 260 字节。
