# qwen35 开发总体要求（长期有效，无需重复说明）

## 1. Kernel 开发语言与目录约定

所有 kernel（包括 CPU SIMD kernel 和 GPU kernel）一律使用**硬件最成熟的语言**开发，
**不使用 Zig 编写 kernel**——Zig 的 `Vector` 实现对硬件适配效果不好，生成的代码质量不如 C。

- **x86 SIMD kernel**（SSE / AVX / AVX2 / AVX512 等）：使用 **C** 开发，放在 `src/kernels/x86/`。
  现状：`iq_kernels.c`（IQ 量化点积）、`kernels256.c`、`kernels32.c`、`kernels512.c`、
  `cpu_caps.c`（特性探测，单一来源）、`generated_tables.h`、
  `src/ggml_tables.zig` / `src/ggml_tables_extra.zig`（Zig 侧表）。
  其中 `generated_tables.h` 文件头写明是从 `llama.cpp/ggml/src/ggml-common.h` **手工提取**的，
  并无生成脚本——改它必须对照 ggml 原文，不要凭记忆。
  （注意：`scripts/gen_simd_tables.py` 生成的是 `src/simd/tables.zig`，而 `src/simd/` 已随
  "去 Zig SIMD" 改造被删除，该脚本目前**过期失效**，不要执行它。）
- **CUDA kernel**：使用 **CUDA C/C++**，放在 `src/kernels/cuda/`（尚未建立，新建时遵循同一约定）。
- Zig 侧只负责模型加载、内存布局、调度、`extern` 声明与分发（`src/c_kernels.zig`、
  `src/simd_dispatch.zig`、`src/simd_api.zig`）。quant / dequant / matvec 等数值核心都应在 C kernel 中完成。
- C kernel 按指令集分层提供实现（AVX2 / AVX / scalar fallback）并运行时分发；
  **scalar fallback 必须是语义完整的正确实现，禁止占位或简化版本**
  （已知违规：`iq_kernels.c` 中 `iq3_xxs_q8_k` 的 `#else` 分支只累加 scale，未算点积）。
- 新 kernel 一律**同时**提供 Zig 侧对拍测试（见 §4），不接受"只有 SIMD 路径、没有标量参照"的改动。

## 2. 执行方式

- **不要使用子任务（subagent）**，所有工作在前台直接执行，用户要看到完整过程。
- 探索、读代码、改代码、编译、测试都由主 agent 亲自完成。

## 3. 构建

- Zig：`0.17.0`，路径 `/home/wisedu/software/zig-x86_64-linux-0.17.0/zig`（`make` 已自动 fallback）。
- **构建缓存必须走本地盘**：项目位于 NFS，直接 `zig build` 会偶发
  `failed to write cache manifest: Unexpected`。用 `--cache-dir /tmp/qwen35-cache`
  （Makefile 里的 `$(ZIGCACHE)` 已包含），不要省略。
- 常用命令：
  - `make cpu` / `make`（CPU + CUDA）
  - `make cpufast`（`-Doptimize=ReleaseFast`，性能测试用这个，Debug 数值路径慢且不可比）
  - `make cuda`（两步链：`zig cc` 出 host.o，再由 `nvcc` 编译 `src/kernel.cu` 并完成最终链接）
  - `make test`
- 新增 C 源文件必须同时登记到 `build.zig` 的 `c_kernels.root_module.addCSourceFile`
  并带上对应 `-m*-` 指令集 flag，否则不会进静态库。

## 4. 正确性与测试

- **当前 `make test` 基线是红的，不是绿色**（2026-10 实测：60 个用例中 54 pass / 4 fail / 2 crash）。
  规则改为：**改动前先记录基线，改动后不得新增失败**；已有失败要当 bug 修，
  不要用 `skip` / 放宽阈值 / 删除用例的方式"变绿"。
  已知失败清单（`zig build test --cache-dir /tmp/qwen35-cache`）：
  - `simd_tests_iq_a/b/c.test.simd kerniq*: SIMD (3 tiers) + _ref vs engine rowDot (synthetic)` — 3 个失败。
    输出里可见运行时对拍偏差：`iq2_xxs_q8_k = 0.0218`、`iq3_xxs_q8_k = 0.00173`、
    `iq4_xs_q8_k = 0.00397`、`iq2_xs_q8_k = 0.00066`；
    **这些正是 C kernel(用 `keven_signs_q2xs`) 与 Zig rowDot(用 `ksigns_iq2xs`) 符号表不一致的表现**，
    与 §1 的表格问题、以及模型乱码同源——修符号表后这批用例应转绿，是修复的验收信号。
  - `simd_tests32.test.simd kern32: SIMD (3 tiers) vs scalar ref` 与
    `... row kernels == sum of block kernels (3 tiers)` — 2 个 **crash**：
    `panic: load of misaligned address ... for type 'const uint32_t'`（疑为测试/kernel 构造或读取数据未对齐）。
  - `simd_tests32.test.simd kern32: real testdata rows vs engine rowDot` — 1 个失败。
- 测试入口：`src/root.zig` 里的 `test { _ = @import("simd_tests_*.zig"); }` 列表
  + `refAllDecls`；`make test` 实际跑两个 test artifact（`mod_tests` 与 `exe_tests`，见 `build.zig`），
  且 `mod_tests` 会 `linkLibrary(c_kernels)`，因此**C kernel 也在对拍范围内**。
- 对拍测试文件：`src/simd_tests32.zig`、`src/simd_tests256.zig`、
  `src/simd_tests_iq_a.zig` / `_iq_b.zig` / `_iq_c.zig`；C kernel 的 Zig 侧调用封装在 `src/simd_c.zig`。
  **新增 kernel 必须在这里补一个"Zig 标量参照 vs C SIMD 结果"的用例**，并在 `src/root.zig` 挂上 import。
- 端到端验收：`tests/run_acceptance.sh [file.gguf]`（注意其默认 ZIG 仍指向 0.16.0，
  与本项目 0.17.0 不一致，跑之前显式 `ZIG=<0.17.0 路径>`）。
- 量化格式的**权威源码**在本机可直接查：
  `/mnt/llm-dev/llama.cpp/ggml/src/ggml-quants.c`（`dequantize_row_*`、`*_dot_*`、`*_vec_*`
  与 `keven_signs_q2xs` / `ksigns_iq2xs` / `iq*_grid` 等表的原始定义），
  CUDA 参照在 `/mnt/llm-dev/llama.cpp/ggml/src/ggml-cuda/`。
  本仓库的 `src/kernels/x86/iq_kernels.c` 是从这里移植的，
  `src/quant_dequant.zig` 注释里的 `ggml-quants.c:2575` 等行号锚点也指向该文件。
  **ggml 的 `dequantize_row_*` 与 `*_dot_*`/`*_vec_*` 是权威定义**，移植时逐表达式对齐。
  本地文档：`ref/llama.cpp-quant-formats.md`。
- 符号表有两族、不可混用：`ksigns_iq2xs[128]`（位掩码，scalar 用）与
  `keven_signs_q2xs[1024]`（int8 ±1 展开，SIMD 用）。二者格式不同，
  Zig/C 任一侧选错表会产生看似合理但全错的数值。

## 5. 调试开关（环境变量，均已实现，优先用它们做 A/B 而不是改代码）

| 变量 | 作用 |
|------|------|
| `QWEN35_SIMD_TIER` | 强制 ISA tier，取值 `baseline`/`avx2`/`avx512`/`avx512vnni`/`avx512vbmi`（其它值静默忽略），定位某档 kernel 出错 |
| `QWEN35_FORCE_SCALAR` | **值为逗号分隔的 probe 名**（`avx2 avx512f avx512bw avx512vl avx512dq vnni vbmi fma f16c ssse3`），关掉这些特性探测，让对应 dispatcher 走标量 oracle 路径。用于 A/B "垃圾来自 SIMD kernel 还是引擎" |
| `QWEN35_FORCE_F32` | 全部量化权重走 Zig dequant + f32 matvec，绕开所有 C kernel（A/B 两半流水线） |
| `QWEN35_F32_TYPES` | 只把列出的类型转 f32（空列表 = FORCE_F32 原语义，q8_0 保持原生） |
| `QWEN35_FORCE_ROWDOT` | 强制 Zig 量化域 rowDot |
| `QWEN35_KERNEL_CHECK` | kernel 结果与 Zig 参照逐块比对 |
| `QWEN35_DUMP_OUT` / `QWEN35_DUMP_IDS` | 导出中间输出 / token id |
| `QWEN35_NO_BATCH` | 关闭 K-token 批量 prefill chunk |
| `QWEN35_PROF` / `QWEN35_PROF_POST` / `QWEN35_STAT` | 性能剖析 |

调试方法论：
1. **先定位是"哪一层"错**，不要猜。用 tier / scalar / force_f32 三个开关做二分，
   把"C kernel 侧"与"Zig 侧"彻底隔离开。
2. **输出乱码 / 空输出时优先怀疑最早的环节**：token embedding（`engine.zig` 的 `embedRow`）
   和 lm_head，而不是中间层。embedding 错则全链路输入皆错。
3. **注意 Zig 数值路径的残留调用点**：`dequantToRows` / `dequantRowFnFor`（`engine.zig`）
   除 `QWEN35_FORCE_F32` 调试路径外，仍在 `embedRow`、`dumpOut` 等主链路被调用。
   按 §1，这些是应迁移到 C kernel 的历史遗留；改它们时以 C kernel 行为为准。
4. 改动前后固定复跑同一句探针：
   `zig-out/bin/qwen35 run <gguf> "天空为什么是蓝色的？" -n 48 --threads 1`
   （单线程是为了结果可比，勿随手加线程数）。

## 6. 运行探针（模型与 target）

- 量化格式覆盖很广，`make run0.8b` 会跑全部 22 个 GGUF 变体；回归时按需单独跑
  （如 `make run0.8biq3xxs`、`make run27biq3xxs`、`make run0.8biq2xxs-llamacpp`）。
- 模型位于 `/mnt/Qwen3.5-0.8B-GGUF/` 等目录；`-ud` 变体用 `make run0.8bud`。

## 7. 代码风格与提交

- 注释写**为什么**与**来源锚点**（如 `ggml-quants.c:2575`），移植代码标注"逐表达式对应 C 的哪一段"。
- Zig 侧保持现有命名与分区习惯（`// ==== Phase N ====` 分节）；不要重排无关代码。
- commit message 用 conventional commits + scope，例：
  `fix(iq3_xxs): use keven_signs_q2xs in Zig dequant to match C kernel`。
- 一次提交只做一件事：kernel 改动与调用方改动分开，便于二分定位回归。

## 8. 文档一致性提醒

- `docs/port-brief.md` 是**过期文档**：它要求"把内核移植成纯 Zig `@Vector`、禁 C/C++、禁 inline asm"，
  与 §1 直接冲突（git 历史 `f0e36ea Remove Zig SIMD kernels, use C kernels only` 已推翻该方案，
  `src/simd/` 目录也已删除）。以本文件为准；需要时可归档或删除该简报，不要据它开工。
