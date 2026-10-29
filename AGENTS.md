# qwen35 开发总体要求（长期有效，无需重复说明）

## 1. Kernel 开发语言与目录约定

所有 kernel（包括 CPU SIMD kernel 和 GPU kernel）一律使用**硬件最成熟的语言**开发，
**不使用 Zig 编写 kernel**——Zig 的 `Vector` 实现对硬件适配效果不好，生成的代码质量不如 C。

- **x86 SIMD kernel**（SSE / AVX / AVX2 / AVX512 等）：使用 **C** 开发，放在 `src/kernels/x86/`。
  现状：`iq_kernels.c`（IQ 量化点积）、`kernels256.c`、`kernels32.c`、`kernels512.c`、
  `cpu_caps.c`（特性探测，单一来源）、`iq_tables.h`、
  `src/ggml_tables.zig` / `src/ggml_tables_extra.zig`（Zig 侧表）。
  其中 `iq_tables.h` 文件头写明是从 `llama.cpp/ggml/src/ggml-common.h` **手工提取**的，
  并无生成脚本——改它必须对照 ggml 原文，不要凭记忆。
  （注意：`scripts/gen_simd_tables.py` 生成的是 `src/simd/tables.zig`，而 `src/simd/` 已随
  "去 Zig SIMD" 改造被删除，该脚本目前**过期失效**，不要执行它。）
- **CUDA kernel**：使用 **CUDA C/C++**，放在 `src/kernels/cuda/`（尚未建立，新建时遵循同一约定）。
- Zig 侧只负责模型加载、内存布局、调度、`extern` 声明与分发（`src/c_kernels.zig`、
  `src/simd_dispatch.zig`、`src/simd_api.zig`）。quant / dequant / matvec 等数值核心都应在 C kernel 中完成。

### 1.1 模块分层（2026-10 重构后，必须遵守）

```
gguf.zig          叶子：Tensor/TensorType + QuantRow（纯字节布局）+ blockElems/blockBytes
   ↑
backend.zig       LinearW + Exec 接口/VTable + selectBackend()
   ↑
backend_cpu.zig   CPU 实现：matvec 全族 / loadLinear / dequant / scratch / pool / tier / prof
   ↑
engine.zig        Qwen3.5 架构本身（不再接触任何硬件细节）
```

- `engine.zig` 的 import 只有 `std`、`gguf`、`backend`、`backend_cpu`、`sampler`。
  **不要再往 engine 里塞 kernel/quant/线程池的 import**；一切硬件能力经 `backend.Exec` 调用。
- 换后端（CUDA 等）= 新增一个实现 `backend.Exec` 的模块，`engine.zig` 一行不改。
- `QuantRow.blocks_per_row` 的**单位是 row-block**：`q8_0` 是 32 元素，**其余类型一律 256 元素**
  （含 `q4_0`/`q4_1`/`q5_0`/`q5_1`/`iq4_nl`/`mxfp4`——GGUF 把它们按 8×32 帧成 256 元素的 row-block）。
  所以 `cols() = blocks_per_row * rowBlockElems()`，`rowBlockElems(q4_0) == 256`，不是 32。
  **踩过的坑**：把 `rowBlockElems` 改成"所有 32 元素块类型都返回 32"会让 `cols()` 小 8 倍，
  且 `matvecQuant` 的自洽断言仍然通过，只在 Q4_0/IQ4_NL 上静默输出乱码。
  `rowBlockElems` 只按 `t == .q8_0` 分支，不要按 `blockElems(t) == 32` 分支。
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

- **当前 `zig build test` 是绿的：17/17 steps succeeded, `test success`**（2026-10 重构后实测）。
  在此之前旧的 simd 对拍用例（60 用例 / 54 pass / 4 fail / 2 crash）已随 `33e2bb8`
  连同 `simd_tests*.zig`、`bench.zig` 一起**删除**。
- **必须知道的代价**：删掉那 5 个 `simd_tests*.zig` 后，**C kernel 不再有 Zig 侧对拍覆盖**。
  §1 提到的 IQ 符号表风险（`ksigns_iq2xs` vs `keven_signs_q2xs` 混用）现在**没有任何自动化测试能发现**，
  只能靠端到端探针（见 §6）肉眼看输出是否通顺。
  规则：**改动前先跑一遍 `zig build test` 与 §6 的探针记录基线，改动后不得新增失败或让探针退化**。
  不要用 `skip` / 放宽阈值 / 删用例的方式"变绿"。
- 现有测试入口：`src/root.zig` 的 import 列表 + `refAllDecls`；`make test` 跑两个 test artifact
  （`mod_tests` 与 `exe_tests`，见 `build.zig`），`mod_tests` 会 `linkLibrary(c_kernels)`。
- `src/backend_cpu.zig` 内有少量内联测试（如 Q8_0 matvec 用例），新增后端能力时在此补对拍。
  **新增 kernel 仍应补一个"Zig 标量参照 vs C SIMD 结果"的用例**——对被删掉的这批，
  想要恢复保护就得重建（可用 `git show 33e2bb8^:src/simd_tests32.zig` 取回原文）。
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
| `QWEN35_FORCE_F32` | ~~全部量化权重走 Zig dequant + f32 matvec~~ **已于 `0cdfb23` 删除**，设了也无效 |
| `QWEN35_F32_TYPES` | ~~只把列出的类型转 f32~~ **已于 `0cdfb23` 删除** |
| `QWEN35_FORCE_ROWDOT` | ~~强制 Zig 量化域 rowDot~~ **已于 `0cdfb23` 删除** |
| `QWEN35_KERNEL_CHECK` | ~~kernel 结果与 Zig 参照逐块比对~~ **已于 `0cdfb23` 删除** |
| `QWEN35_DUMP_OUT` / `QWEN35_DUMP_IDS` | 导出中间输出 / token id |
| `QWEN35_NO_BATCH` | 关闭 K-token 批量 prefill chunk |
| `QWEN35_PROF` / `QWEN35_PROF_POST` / `QWEN35_STAT` | 性能剖析（`PROF-M` 行由 `backend_cpu` 输出） |

调试方法论：
1. **先定位是"哪一层"错**，不要猜。重构后只剩 `QWEN35_SIMD_TIER` 与 `QWEN35_FORCE_SCALAR`
   两条 A/B 路径（4 个更强的开关已删）。如果 tier/scalar 全档一致地错，
   说明**不是 ISA 问题**，而是几何/布局/索引问题——去查 `QuantRow` 的
   `blocks_per_row` / `rowBlockElems` / `cols()` 三者是否自洽（见 §1.1 的坑）。
2. **输出乱码 / 空输出时优先怀疑最早的环节**：token embedding 和 lm_head，而非中间层。
   embedding 错则全链路输入皆错。`vDequantRow`（现为 `dequantRowInto`）对 `q8_0` 有**专用旁路**，
   不能删：`dequantRowFnFor` 只覆盖 256 元素块的 K/IQ 类型，Q8_0 塞进去当 `cols % 256 != 0` 时会丢尾巴。
   注意 `dequantRowFn` **每次只写 256 个 float**，宽行必须按 `blocks_per_row` 分块调用——
   用 `rowBlocks()`（本类型自己的块数，q4_0 是 8 倍）当循环上界会越界。
3. **`matvecQuantScalar` 只对 256 元素块类型正确**：它传 `.row_bytes = blockBytes()`（单块字节）
   而 pool 用 `xs[b*256 .. b*256+256]` 一次吃一个 256 元素 row-block，
   对 `q4_0`/`iq4_nl` 这类"块 32、row-block 256"的类型会算错行跨与激活窗口。
   已知未修；A/B 时不要拿它当 Q4_0 的 oracle。
4. 改动前后固定复跑同一句探针：
   `zig-out/bin/qwen35 run <gguf> "天空为什么是蓝色的？" -n 48 --threads 1`
   （单线程是为了结果可比，勿随手加线程数）。
5. **跨量化格式必须逐个验**，不能只验 Q8_0 就收工：Q8_0 通过**不代表**其它类型通过。
   至少覆盖 Q8_0 / BF16 / Q4_0 / IQ4_NL / Q4_K_M / Q6_K。
   这不是理论风险：2026-10 的 `5f654f3` 修的就是两个只在 Q8_0 之外才发作的 bug——
   `matvecQuant` 断言 `bpr == cols/32`（Q4_0 等实际是 cols/256），以及
   `vDequantRow` 把整行交给只写 256 个 float 的 `dequantRowFn`（1024 宽的行有 3/4 未写）。
   **Q8_0 恰好是 32 块与 row-block 单位重合的唯一类型**，所以它单测通过而其它全坏。
   定位手段是给 `dequantRowInto` 写 Q4_0 对拍测试，而不是读代码猜；
   新增类型时照此办：先在测试里造块字节，与 `quant.rowDotXxx` 参照对齐。

## 6. 运行探针（模型与 target）

- 量化格式覆盖很广，`make run0.8b` 会跑全部 22 个 GGUF 变体；回归时按需单独跑
  （如 `make run0.8biq3xxs`、`make run27biq3xxs`、`make run0.8biq2xxs-llamacpp`）。
- 模型位于 `/mnt/Qwen3.5-0.8B-GGUF/` 等目录；`-ud` 变体用 `make run0.8bud`。
- **llama-cli 必须加 `--single-turn`**（脚本/非交互调用时）。不带它 llama-cli 进入交互 REPL
  模式，非交互上下文里读不到输入就静默退出：stdout 0 行、退出码 0、无任何日志——
  看起来像"没跑"。性能对比命令模板：
  `llama-cli -m <gguf> -p "天空为什么是蓝色的？" -n 48 -t 1 -dev none --single-turn`
  （单线程保可比，`-dev none` 禁 GPU）。

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
