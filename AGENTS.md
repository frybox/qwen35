# qwen35 开发总体要求（长期有效，无需重复说明）

## 1. Kernel 开发语言与目录约定

所有 kernel（包括 CPU SIMD kernel 和 GPU kernel）一律使用**硬件最成熟的语言**开发，
**不使用 Zig 编写 kernel**——Zig 的 `Vector` 实现对硬件适配效果不好，生成的代码质量不如 C。

- **x86 SIMD kernel**（SSE / AVX / AVX2 / AVX512 等）：使用 **C** 开发，放在 `src/kernels/x86/`。
  现状：`iq_kernels.c`（IQ 量化点积）、`kernels256.c`、`kernels32.c`、`kernels512.c`、
  `cpu_caps.c`（特性探测，单一来源）、`iq_tables.h`（**纯 extern 声明**，不含数据）。
- **CUDA kernel**：使用 **CUDA C/C++**，放在 `src/kernels/cuda/`（`vecdotq.cuh`、`mmvq.cu`、
  `convert.cu`、`iq_tables.{cu,cuh}`、`api.h`）。

### 1.0 IQ 查找表只有一份数据源（2026-10 改造，必须遵守）

`src/iq_tables.zig`（**由 `scripts/gen_iq_tables.py` 从
`llama.cpp/ggml/src/ggml-common.h` 生成**，文件头有源 sha256 与逐表行号锚点）
是 12 张 IQ 表的**唯一定义**，形如 `pub export const iq1s_grid: [2048]u64 = .{...}`。
**仓库根不再有 `src/iq_tables.h`**（2026-10 删除：它是 translate-c 的中间产物，
数据挪进 Zig 源后成了多余的第二份）。三种消费者都只是引用同一批符号：

| 消费者 | 形态 | 位置 |
|--------|------|------|
| Zig（宿主侧唯一定义） | `pub export const T[n]` | `src/iq_tables.zig` |
| x86 C kernel | `extern const T[n];`（只声明） | `src/kernels/x86/iq_tables.h` |
| CUDA device | `extern __device__ const T *q35_d_<name>;`（指针）+ `#define <name> q35_d_<name>` 别名 | `src/kernels/cuda/iq_tables.cuh` |

- **Zig 侧持有宿主侧的定义**，x86 kernel 只 `extern` 引用，
  **不要**再写 `src/kernels/x86/iq_tables_data.c` 之类的定义单元——曾试过，链接器把它和
  生成物的符号分开，出现同一张表两个地址，被 `src/iq_table_tests.zig` 抓到。
- **CUDA 侧的表是 host 符号 + 启动时 H2D**（2026-10 改造，取代旧的
  `Q35_TABLES_CUDA` 宏实例化 `__device__` 数组）。数据仍只有一份：
  12 张表是**可执行文件里** `src/iq_tables.zig` 的那批 host 符号，
  `.cu` 目标文件被链接进同一个 `zig-out/bin/qwen35`，因此设备侧既没有第二份数据、
  也没有 `.so` 或任何运行期符号查找（`src/backend_cuda.zig` 头部注释是这条约定的权威表述）。
  - `src/kernels/cuda/iq_tables.cuh` 用 X-macro 声明 12 个 `extern __device__ const T *q35_d_<表名>;`
    —— 是**指针**，每个 device 指针指向一份 `cudaMalloc` 常驻缓冲里的一段，
    再由 `q35_launch_init_tables()` 把表数据 H2D 拷进去。
    同文件底部另有 12 条 `#define <表名> q35_d_<表名>` 别名，把 `iq1s_grid[idx]` 这类
    索引表达式（`convert.cu` + `vecdotq.cuh` 约 46 处）原样还原成表名，调用点一字不改。
  - `src/kernels/cuda/iq_tables.cu` 定义这 12 个指针 + 一份常驻设备缓冲（一次 `cudaMalloc` 装下
    12 张表），`q35_launch_init_tables()` 在**首次 GPU 计算前**（`CudaBackend` 的 scratch 分配里）
    先逐表 `cudaMemcpyAsync` 拷数据，再用**每表一次 `cudaMemcpyToSymbolAsync`** 把该表的设备地址
    写进对应的 `q35_d_*` 指针符号。显式**不用** init kernel（`<<<1,1>>>` 里写 `__device__` 指针，
    `iq_tables.cu` 头部注释记录了当年实测翻车的经过）。幂等，失败即报错不静默继续。
  - 主机地址来自 `src/kernels/cuda/iq_tables_host.c`（纯 C，无设备命名空间）：
    它直接 `extern const T <表名>[n];` —— 与 Zig 侧导出名完全相同、**没有前缀** ——
    并实现 `q35_host_tables()` 返回 (地址, 字节数) 的静态记录数组，顺序即
    `iq_tables_list.inc` 的顺序。`build.zig` 用 `zig cc` 单独把它编成 .o 再
    `addObjectFile` 给可执行文件（不登记进 CPU kernel 静态库），所以只有 `-Dcuda=true` 的
    exe 带它。
  - **设备指针符号为什么叫 `q35_d_<表名>`（踩过的坑，不要改回去）**：
    `__device__ const T *iq1s_grid;` 的定义会**同时生成一个同名 host stub**
    （`readelf -sW` 显示 `OBJECT GLOBAL HIDDEN ... 8 iq1s_grid`）。CUDA kernel 以前在独立的
    `libqwen35-cuda-kernels.so` 里，那里 `extern __device__` 只产生未定义引用，没有东西可冲突；
    现在 `.cu` 目标文件与 `src/iq_tables.zig`（`pub export const iq1s_grid[2048]`）
    在**同一个可执行文件**里，于是链接器直接报
    `ld: multiple definition of 'iq1s_grid'`（实测）。强行加
    `--allow-multiple-definition` 能链过但**运行期段错误**，所以改名不是打包细节。
  - 需全局内存而非 `__constant__`：网格是逐元素索引，同一 warp 的 lane 读不同偏移，
    常量缓存会串行化；表约 40 KB，放 L1/L2 更好。
- **`src/kernels/cuda/iq_tables_list.inc` 是"表名 ↔ 元素类型 ↔ 长度 ↔ 顺序"的唯一来源**，
  `iq_tables.cuh`（设备指针声明）、`iq_tables_host.c`（host 记录）都只展开它，
  `iq_tables.cu` 的偏移与 symbol 数组也由它推。**新增/删除表必须同时改它和
  `src/iq_tables.zig`**（漏改的表现是 `q35_host_tables()` 的 count 与 `.cu` 侧不一致，
  或某个消费方编译不过）。
- **`-rdc=true` + 设备链接**仍由 `build.zig` 的 stage1/stage2 带着
  （全链：`nvcc -rdc=true -c` 编每个 `.cu` → `zig cc` 编 `iq_tables_host.c` →
  `nvcc -rdc=true -dlink` 合并设备镜像 → 全部 `addObjectFile` 进 exe，
  并 `linkSystemLibrary` 上 cudart/c++ 的导入库）：
  `.cu` 之间的 `__device__` 引用（`iq_tables.cuh` 的 12 个指针在 `mmvq.cu` /
  `convert.cu` 里被读）需要 nvlink 才能跨编译单元解析，不要摘掉。
- **派生表**（`keven_signs_q2xs`、`ksigns64_derived`、`k32_iq4nl_tab`、`k32_nvfp4_values`）
  不写进生成物，由 `src/iq_tables.zig` 在 comptime 从规范表算出并 `pub export`；
  C 侧只 `extern` 声明。这样对拍测试比较的是"两种独立推导"，而不是"抄件对抄件"。
- 改表流程：改 `ggml-common.h` 对应内容 → `python3 scripts/gen_iq_tables.py` →
  `python3 scripts/gen_iq_tables.py --check` 应通过 → `zig build test`。
  **不要手改 `src/iq_tables.zig` 的数据行**，它会被下次生成覆盖。
- 注意：`scripts/gen_simd_tables.py` 生成的是 `src/simd/tables.zig`，而 `src/simd/` 已随
  "去 Zig SIMD" 改造被删除，该脚本**过期失效**，不要执行它。
- Zig 侧只负责模型加载、内存布局、调度、`extern` 声明与分发（`src/iq_tables.zig`、`src/c_kernels.zig`、
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
  - `make cuda`（= `zig build -Dcuda=true`：nvcc 编 `src/kernels/cuda/*.cu` → `-dlink`
    设备链接 → 静态链进 `zig-out/bin/qwen35`，产物仍是**一个** CLI 可执行文件）
  - `make test`
- 新增 C 源文件必须同时登记到 `build.zig` 的 `c_kernels.root_module.addCSourceFile`
  并带上对应 `-m*-` 指令集 flag，否则不会进静态库。

## 4. 正确性与测试

- **当前 `zig build test` 是绿的：18/18 steps succeeded, 74/74 tests passed**（2026-10-31 实测）。
  必须用 `make test`（它已带 `--summary all`）：Zig 0.17.0 用 `--listen=-` IPC 协议跑 test artifact，
  握手噪音会打印 `failed command: ...test --listen=-` **且退出码 0、全部测试其实通过**
  （裸的 2 测试空项目也复现，不是本仓库的 bug）。**只有 `--summary all` 的最后一行才是可信信号**：
  绿 = `Build Summary: 18/18 steps succeeded; 74/74 tests passed`（exit 0）；
  真失败 = `16/18 steps succeeded (1 failed); 71/74 tests passed (3 failed)` + `test transitive failure`（exit 1）。
- **C kernel 的 Zig 侧对拍覆盖已恢复**（`33e2bb8` 删掉的 5 个 `src/simd_tests*.zig` 已取回并重新接进
  `src/root.zig`：`simd_tests32`、`simd_tests256`、`simd_tests_iq_a/b/c`，共 15 个用例）。
  辅助层从未被删，所以取回即可用：`src/simd_c.zig`（C kernel 的 extern 声明，测试专用 tier 模块）
  + `src/simd_ref.zig`（Zig 标量 oracle）。每个 kernel 三路校验：C kernel vs Zig 标量参照、
  vs 引擎自己的 `rowDot*`（用 `testdata/udtest` 的真实打包行）、row kernel == 各 block kernel 之和。
  **已实证敏感性**：把 `quant_dequant.zig` IQ3_XXS rowDot 的 `ksigns_iq2xs` 位反转后，
  3 个用例立刻抓到（`quant_dequant` rowDot、`kerniq synthetic`、`kerniq testdata`，worst=2.0），exit 1。
  §1 的 IQ 符号表风险（`ksigns_iq2xs` vs `keven_signs_q2xs` 混用）**重新被自动化覆盖**。
  - 已知覆盖缺口（要补时先看清）：`simd_ref.zig` 的 `deq*` 系列（`deqIQ2XXS`/`deqIQ3XXS`/`deqIQ4XS` …）
    **没有被任何用例调用**——对拍只走 `simd_ref.dot32` 与 `simd_ref.dotiq`。往这些 `deq*` 里注入扰动
    测试不会报警，别把它们当作已有保护。
  - 数量口径（实测）：恢复前 `zig build test` = 54；恢复后 74 = 54 + 15 个对拍用例
    + 5 个 `test { _ = @import("simd_testsX.zig"); }` 包装块（每个包装块自身算一个 test）。
    15 个用例名逐个核对确认真在跑（`grep simd_tests /tmp/…` = 15 行），不是空跑。
- 规则：**改动前先跑一遍 `make test`，改动后不得新增失败**。
  不要用 `skip` / 放宽阈值 / 删用例的方式"变绿"。
- **token 流不是验收标准**（2026-11 决定：项目后续支持多模型 / 多硬件，固定 prompt
  的 token 比对不可扩展；llama.cpp 也没有这种探针——它的跨后端验收是逐 op 的 nmse
  容差 `tests/test-backend-ops.cpp` + 手工 perplexity）。

  **也不得作为单次改动的验收条件。** 2026-11 起明确：不等 token 流逐位一致再提交。
  理由不是"麻烦"，而是**逐位一致在算法级改动面前是错误的目标**：
  - 任何改变浮点归约顺序的改动（分块算法取代递归、flash-attention 取代逐 token、
    换 kernel 的 tiling）在数学上等价，但末位必然不同。用逐位一致当门槛等于
    **禁止所有算法级优化**，只能做"代码搬到别处"型的改动——而 prefill 剩余的差距
    （对照 llama.cpp 的 `build_delta_net_chunking` / `flash_attn_ext`）恰恰只能靠
    算法级改动消除。
  - 它还会给出**假阴性**：改动正确但 token 流分叉时会被误判成回归；
    反之局部 bug 可能因为采样落在同一 token 而不被发现。
  - CPU/CUDA 本来就不逐位相同（f32 收尾 / 归约顺序 / 超越函数 1–2 ulp，见
    `docs/cuda-elementwise-ulp.md`；MVCHECK 全过时 token 流仍会分叉，实测见
    `src/kernels/cuda/mmq.cu:1105`），llama.cpp 默认采样是 temp 0.8 + 随机 seed，
    其参照输出本身不可复现。

  **正确的做法**：
  - kernel 级对拍（`make test` 三路对拍 + `QWEN35_MVCHECK`）保证算子正确；
  - 算法级改动另写**等价性论证**（为什么数学上等价）并记录在提交说明里；
  - 端到端只用 `QWEN35_DUMP_IDS` 看输出是否通顺 / 长度是否合理，作为烟雾信号；
  - `QWEN35_DUMP_IDS` 仍是**调试**手段（定位分叉起点），不是判据。
  零容忍的场景只剩一种：**改动前后都不应出现乱码 / 空输出 / 提前 EOS**。
  完整口径见 `docs/cuda-backend-brief.md` §2bis。
- 现有测试入口：`src/root.zig` 的 import 列表 + `refAllDecls`；`make test` 跑两个 test artifact
  （`mod_tests` 与 `exe_tests`，见 `build.zig`），`mod_tests` 会 `linkLibrary(c_kernels)`。
  注意 `-Doptimize=ReleaseFast` 下 `exe_tests` 那条链不参与，`make test` 用默认 optimize。
- `src/backend_cpu.zig` 内有少量内联测试（如 Q8_0 matvec 用例），新增后端能力时在此补对拍。
  **新增 kernel 仍应同时补一个"Zig 标量参照 vs C 结果"的用例**（照 `src/simd_tests*.zig` 的形状办），
  不接受"只有 SIMD 路径、没有标量参照"的改动。
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
| `QWEN35_DUMP_OUT` / `QWEN35_DUMP_IDS` | 导出中间输出 / token id。**调试用**（定位分叉起点），不是验收判据，见 §4 |
| `QWEN35_NO_BATCH` | 关闭 K-token 批量 prefill chunk |
| `QWEN35_GDN_WY` | **仅 CUDA 后端**。`=0` 退回逐 token 的 `q35_gdn_step_rows_chunk_kernel`（即 WY 版的对照/回归参照），其它值或不设走 WY 分块版。见 §5 末尾的 GDN 说明 |
| `QWEN35_PROF` / `QWEN35_PROF_POST` / `QWEN35_STAT` | 性能剖析（`PROF-M` 行由 `backend_cpu` 输出） |
| `QWEN35_MVCHECK` | **仅 CUDA 后端、仅调试用，不是计算路径**。每个量化 matvec 之后，用内嵌的 CPU 后端对同一行再算一遍并逐行比对，打印 `MVCHECK[<type>] rows= cols= mismatched= batchk= k=`。用于确认"GPU kernel 结果与 CPU 参照一致"（CUDA 逐格式验收的自动化手段）。开销极大，只在 A/B 时开 |
| `QWEN35_MVCHECK_ONLY` | 限定只对某一种量化类型做 MVCHECK（值 = `q.ttype` 的 tagName，如 `iq1_s`）。batch-K 参照需要 CPU scratch 按 K 塑形，不是每种类型在每个 launch 顺序下都能活，所以追单一 kernel 时用这个定向跑 |
| `QWEN35_CUDA_SAMPLE_CPU` | **仅 CUDA 后端、仅调试**。把 `sample` 交给 CPU 参照，喂的是**同一份 GPU logits**（D2H 后）。这是唯一能隔离采样器差异的 A/B。capture 内会 `graphAbandon` 降级，故配合 `QWEN35_CUDA_GRAPH=0` 使用 |
| `QWEN35_CUDA_COOP_WARPS` | **仅 CUDA 后端**。decode（tokens==1）warp-per-row matvec 的 block 形状（1..8 个 warp/block），A/B 用。不设时由硬件探测决定：sm_89 → 4（本机实测最佳），其他 cc → 占用率公式（cudaOccupancyMaxActiveBlocksPerMultiprocessor 选驻留 warp 最多的形状），见 `mmvq.cu` 的 `q35_coop_warps()`。探测来源是 `caps.cu`（`q35_gpu_cc()`/`q35_gpu_sm_count()`/`q35_gpu_name()`，一次探测、缓存读，x86 侧 `cpu_caps.c` 的 CUDA 镜像） |
| `QWEN35_CUDA_SPLIT` | **仅 CUDA 后端**。decode K-split 的每 block lane 数，默认 8。合法值 0/1/2/4/8/16/32；launcher 会把它 clamp 成 `q35_split_units(ttype)` 的最大因子，并对每种类型再按 `q35_split_min_per` 收窄。`split<2` 退回 coop kernel |
| `QWEN35_SPLITDBG` | **仅 CUDA 后端**。每次走 split 路径的 launcher 往 stderr 打一行 `[split] ttype= rows= cols= nblk= split= warps= shmem=`。确认某类型**真的**走了 split、以及 per 是多少，只需看这一行——不要凭 tok/s 反推 |
| `QWEN35_MVLIST` | **仅 CUDA 后端**。打印每个 matvec 调用的 `[mvl] <type> rows= cols=`。用来算「一 token 里每种类型多少流量」，是决定优化优先级的唯一可靠依据（按 launch 次数或按代码直觉都会排错序） |
| `QWEN35_CUDA_MEMONLY` | **仅 CUDA 后端**。只搬权重不做点积，测「纯访存下限」。27B IQ3_XXS 实测 27.0 tok/s → 说明瓶颈是访问模式而非设备带宽（4090D 理论 ~1000 GB/s，实测各类型 130–660） |

调试方法论：
0. **CUDA 侧的坑（2026-11 全部踩过，逐条都有代价）**：
   - **绝对不要在 `__device__`/`__device__ __host__` 函数里 `getenv`**。
     `q35_split_units` 是 `__device__ __host__` 且被 split kernel 每 block 调用；
     在那里加 `getenv` 做 A/B 编译能过、host 侧正常，**设备侧结果被静默污染**：
     输出 3579 个 token 的乱码，而 MVCHECK 全是 0。A/B 开关只放 launcher（host-only）。
     该函数必须保持成无副作用常量表。
   - **`_part` body 不能对任何 range `return 0.0f` 来「拒绝」**。
     曾把 IQ3_S 的 `min_per` 放宽到 1 却没改 body，旧的 `if (lo & 1) return 0.0f`
     让一半 slice 算 0 → "27.03 tok/s"（比真值快 4%）而输出是「茶，写茶，，」。
   - **采样器在 decode graph 内，下载在 graph 外**（2026-11）。`sample` 必须在
     `stepBegin`..`stepEnd` 之间调用（`engine.zig` 的 `step()` 已这样排），才能被
     capture 进 decode graph；4 字节 token 的 D2H **不能** capture，所以它在 `stepEnd`
     里做，再由 vtable 的 `sampledToken` 交给 engine（CPU 后端是恒等）。
     - 逐 token 的采样参数（temperature / top_k / top_p / PRNG 状态）走**设备常驻控制块**
       `d_sample_ctl`（64 B，每 token 一次 H2D），**不要**用 `GraphExecKernelNodeSetParams`
       ——那会把采样变回每个 replay 都要 retarget 的节点。
     - `sample.cu` 的节点表是**扁平无条件**的：host 读不到设备内存里的参数，所以
       greedy/k/p 的分支全在 kernel 内（`k_plan` 写 flags，其余 kernel 早退）。
       由此 graph 里固定 18 个采样节点，代价是禁用阶段的微秒级空跑。
     - `QWEN35_CUDA_SAMPLE_CPU=1` 是唯一能端到端比两个采样器的开关：它把**同一份 GPU
       logits** D2H 后交给 CPU 参照（其它比法都会先撞上 CPU/CUDA matvec 的 f32 分叉，
       拿到的是两组不同 logits）。它**不能在 capture 内**跑（阻塞 D2H 非法），
       开了会 `graphAbandon` 并降级——所以只用图形关闭时（`QWEN35_CUDA_GRAPH=0`）做 A/B。
       实测：graph=0 下 device sampler 与 CPU 参照在 greedy / temperature / top_k /
       top_p 各组合上 token 流完全相同；greedy 下与改动前的 argmax 路径也相同。
       注意这是**两个采样器喂同一份 logits**的特例，不能用同样的办法去要求
       两个不同 kernel 的 logits 一致（见 §4：token 流不是验收标准）。
     - graph on/off 的 token 流**本来就会分叉**（既存，已用改动前的构建复现）：
       不要把它当成采样器的回归信号，用 `QWEN35_DUMP_IDS` 看分叉起点。
     **被拒绝的 slice 返回 0 = 假加速**。切分必须能算它拿到的每一种 range；
     `q35_split_min_per` 的 clamp 只是第一道防线。
   - **切分数必须从 `_acc` 的循环步长推导，不能猜**。IQ4_XS 是 4 不是 8
     （`_acc` 每步 `h >>= 4` 一次）。三个错误的 rebase 都只影响靠后的 slice，
     `lo=0` 仍然正确，模型能正常跑一个 token——所以必须在**奇数 `lo` 与非零 `lo`
     都覆盖**的整数 gate 下验（见 `/tmp/mb/gate_3s.cu`）。
   - **切分边界要让每个 lane 的权重读取连续**。Q4_K 按 16 元素子块切是 298→316 GB/s
     （每个 lane 读 16 个跨步字节），改成按 64 元素的 q4 字节**行**切是 298→**553 GB/s**。
     同理 IQ2_XXS/IQ2_XS/IQ1_S 都用 32 元素 `ib32` 组边界。
   - **共享数组会让 split 反向变慢**。Q5_K 与 Q4_K 行结构相同，但 `qh[32]` 是 4 个
     row slice 都要**整块重读**的共享数组（每个 slice 用不同 bit 掩码取同一批字节），
     4 路 split 把 qh 流量变成 4 倍：实测 25.29 → 24.09 tok/s。
     所以 `q35_split_units(Q35_TT_Q5_K)` 返回 **0**，`_part` body 留着等共享内存 staging。
   - **CUDA graph capture 期间 kernel 不执行**。开着 graph 去 dump 中间缓冲只会读到
     陈旧零值，看着像「输出全是 0.0」。调试一律先 `QWEN35_CUDA_GRAPH=0`。
   - **发散 warp 上的全掩码 `__syncwarp()` 是硬挂死**，不是崩溃：
     表现为进程不退出（exit 124）。`nblk % kBlocksPerStep != 0` 时要把轮数向上取整，
     并让所有 lane 每轮都对哑值做 `__shfl_down_sync` 以保持收敛。
   - **多形状 microbench 会污染 context**：`cudaFree` + event 复用后，第一次出错之后的
     所有行都打印垃圾 GB/s（1e12+）配 "invalid resource handle"。要么一进程一形状，
     要么先验算再计时（`/tmp/mb/mb_ok.cu` 会先检查每行输出非零且有限，再进计时循环）。
   - **microbench 的 ttype 编号是 `Q35_TT_*`（`api.h`），不是 GGUF 的枚举**。
     两套编号不同：GGUF 里 IQ3_S 是 19，而 `Q35_TT_IQ1_S` 才是 19。混用会静默
     bench 到另一种格式（曾把每个 "iq3_s" 行测成 IQ1_S 的耗时）。
   - **`/tmp/mb/*.o` 是手编的，不会跟着 `make cuda` 更新**。改了 `.cuh` 之后
     一定要重编 `mmvq.o` 再链接，否则 bench 跑的仍是旧 kernel（症状：新加的 split
     完全不生效，`[split]` 一行都不打印）。
   - **固定 `-n` 才能比 tok/s**。不同 `-n` 会走到不同长度的分叉，token 数不同时
     耗时不可直接比（曾把 536-token 的 25.29 和 578-token 的 24.09 拿来比）。
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
4. 改动前后固定复跑同一条命令（仅 A/B 调试用，不是验收标准）：
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
6. **GDN 的 state 更新走 WY/UT 分块（`e33de40`），不是逐 token 递推**。这条值得单独记，
   因为它把 27B prefill 的整条差距一次性消掉了，而根因不是带宽而是**深度**：
   - 旧 kernel `q35_gdn_step_rows_chunk_kernel` 串行走 t，并行轴只有 (head, column)
     = 48×128 = 6144 线程（占 4090D 的 2.6%），且每线程的 `acc` 是串行 FADD 链，
     **4 cycle 延迟完全暴露**。ablation 实测 93 ms / 419 ms = prefill 的 22%，
     而 llama.cpp 侧同一 ablation 是 ~0。
   - 门控递推在 t 上非线性，**不能**直接并行；但可以压**深度**：C 个 token 的块内
     更新对 C 个 delta 是仿射的 → 一次严格下三角求解（前向代入），只有块边界串行。
   - 并行轴改成**按值列切**（`state` 的列 j 只被 v/km/delta/out 的列 j 碰），
     grid = heads×(vd/V) = 96 blocks —— 与 llama.cpp `gated-delta-net-mma.cu` 的
     `slice32` 同一个思路。**切 t 不行**，t 的块边界就是串行部分。
   - `(C, V, threads) = (8, 64, 1024)` 来自 device 扫参：rows=525 时 1.44 ms/call
     vs 逐 token 3.53；rows=1024 时 2.78 vs 10.84。C 小赢在 solve 串行序言短，V=64 赢过 32/128。
   - **`km` 不能写成 `g_r·(k_r^T S_prev)`**：它是 `gamma_r·k_r^T S_{r-1}`，`S_{r-1}` 已含
     块内前 r 行。用 `km_r = v_r - delta_r/beta_r`（由 delta 定义恒等）。第一版按前者写，
     **每块第 0 行精确、第 1 行起全错**，是 f64 参照抓到的——不是 MVCHECK。
   - 验证办法（新算子都应照此办）：① f64 参照的块形式 vs 字面逐 token 循环，
     C=8/16/32 × rows=525 全部 1e-7；② kernel vs 逐 token kernel 在 device 上：
     out 3.3e-7 / state 1.6e-7 / delta 6.9e-8 / km 1.8e-6。
   - **不与逐 token 版逐位一致**（重结合），所以 CPU oracle（`engine.zig` 的 `gdnFused`）
     保持逐 token 形式；旧 kernel 留在文件里作参照与 fallback，`QWEN35_GDN_WY=0` 可切回。
   - 收益：prefill 1245 → 1430 tok/s（+14.8%），decode 不变（rows=1 退化回同样工作量）。
     完成后**非 GEMM 算子只剩 21 ms / 364 ms（5.6%）**，剩下的全是 mma gemm。

7. **prefill 的差距全在 `matvec_batchk`（mma gemm），而 mma 的差距在主因是 barrier 数不是指令数**
   （`679b0c1`，2026-11）：
   - 定位工具是**同形状 A/B microbench**（`/tmp/mbm/ggml_ref.cu` 跑 llama.cpp 的 ggml
     MUL_MAT，`/tmp/mbm/mb_ab.cu` 跑我们的 `q35_launch_mul_mat_vec_tile`，**同 rows/cols/tokens、
     同 IQ3_XXS 权重**）。端到端只有 1.15×，单形状上是 1.83× —— 端到端被非 gemm 算子稀释了。
     **不要用手工估算 FLOPs 代替这个 A/B**。
   - 归因办法是**在 kernel 里按 ablation 拆时间**（unpack / staging / mma 三段），
     而不是猜。实测 ffn gate/up 0.956 ms 里 unpack 只有 0.12 ms。
   - **真相：sB 一次只装 `Q35_MMQ_TILE_TOKENS`(16) 行，所以每 16 个 token 就要重新 staging +
     两次 `__syncthreads()`**。128-token 的 J slab 里 = 每个 super-block 16 个 barrier，
     而 llama.cpp 的 `mul_mat_q_process_tile` 一次装 128 行、两次 `vec_dot` 夹一对 barrier = **4 个**。
     我们多 4×；而每 SM 只驻留 1 个 block 时，一次 barrier 就是整个 SM 排空。
   - 修法：`Q35_MMQ_STAGE_TILES`（NSTG）让 NSTG 个 acc-tile 共用一次 staging。
     **mma tile 仍是 TT 个 token，所以 `acc[MAX_TILES][NACC]` 和各 fragment 数组都不用动**，
     只有 sB 变深 —— 这是寄存器没有爆炸的原因。token 循环因此拆成
     「外层 staging 循环 + 内层 sub-tile 循环」。
   - **NSTG=4 是尖峰不是平台**：1402(1) / 1462(2) / 1532(4) / 1492(8)。过了 4，
     sB（17 KB）的代价超过省下的 barrier。
   - **动态 shared 必须把 `cudaFuncSetAttribute` 设成真实字节数，不能设 100 KB**：
     本机 per-SM 上限约 99 KB，请求 100 KB 会静默吃掉三分之一的吞吐
     （1430 → 1000 tok/s），而且看起来像完全无关的回归。设成实际 `shmem` 后反而略快于
     静态版（1456 vs 1430）。
   - 结果：prefill 1430 → 1520（+6.3%，对 llama.cpp 1.06×），
     小格式涨得更多（IQ2_XXS 1957→2200 **反超 llama.cpp 1.29×**、IQ1_S 1787→2050），
     ffn gate/up 98.0 → 108.9 TFLOPS。**decode 不受影响**（此 kernel 只服务 prefill）。
   - 仍然欠 llama.cpp ~1.65×（108.9 vs 179.3 TFLOPS）。**这个缺口已经查过一轮，
     下列方向都实测无效，不要重复**（2026-11）：
     - **硬件 mma 上限已实测**：`/tmp/mmc/mp.cu` 纯 mma 探针在 4090D 上是
       **586 TFLOPS**（int8 `m16n8k32`）；2 warps/SM 就 289、4 warps/SM 就 577。
       所以我们是 **18.6% of mma peak**，llama.cpp 是 30.6% —— 两边都远没打满，
       不是"我们已经接近硬件极限"。
     - **不是 barrier**（已修，见上）；**不是 B 分片载入形式**
       （`load_generic` 8 次标量 LDS → `load_ldmatrix` 1 次 LDSM：实测无差别，
       ptxas 本来就已选 LDSM）；**不是 `isum` 的整型/浮点形式**
       （改浮点累加 1485 vs 1520，更慢）；**不是 `ls_w` 重复 LDS**
       （hoist 进寄存器 10240 次读 → 32 次：实测无差别，编译器早已缓存）；
       **不是 mma 依赖链**（把 NT 个 mma 发到独立 D 寄存器再消费：无差别）。
     - **不是 occupancy**：168 regs × 256 thr = 43008 regs → 1 block/SM（8 warps）。
       想凑 2 blocks 必须 `-maxrregcount=128`（65536/512 = 128），
       nw=8/nstg=1 时 shared 47.2 KB 也确实塞得下两个 —— **实测 1240 tok/s，更差**
       （寄存器溢出代价 > 多一倍 warp 的收益）。**先验不要拿 occupancy 当默认答案。**
     - **不是 tail/stream-K**：`tiles=680`、`114 SM` → 5.96 waves，
       ffn gate/up 的 tail 浪费只有 **0.6%**，却在四个形状里最慢（1.83×）。
       所以 stream-K 对我们最大形状基本零收益（`ffn down` 12.3%、`q_proj` 29.8%
       有收益，但那是小头）。**别再拿"llama.cpp 有 stream_k"当解释。**
     - **权重重复读的成本已量化**：grid.y 按 128 token 切 slab，525 token = 5 slab，
       ffn gate/up 的 34 MB 权重被读 5 次 → 按 1008 GB/s 约 0.135 ms，
       是 0.860 ms 的 **16%**。但把 `MAX_TILES` 提到 16 让一个 block 吃下 525 token
       实测 **1431 vs 1497，更差**（acc[] 溢到 local memory，旧注释早有记载）。
       L2（72 MB）确实吸收得比预想好。
     - **【已解决】工具链阻塞**：CUDA 13.2 的 `cuobjdump`/`nvdisasm` 能反汇编 sm_89
       （12.0 版报 `Support for 'sm_4' has been removed`）；`ncu` 的阻塞是驱动参数
       `RmProfilingAdminOnly=1`，设为 0 后可用（`/etc/modprobe.d/nvidia-profiling.conf`，
       重启生效；之前用 `sudo ncu` 也行）。**有了 SASS + ncu 之后下面这些都是实测，不是推断。**
     - **SASS 实测（修掉了一个错误结论）**：热循环是 8 IMMA 配 35 LDS + 25 IMAD。
       之前据此断定"IMAD 是 mma 的 4.7 倍所以是瓶颈"——**错了**。
       探针 `/tmp/mmc/mix.cu` 每条 mma 加 4 或 8 条 IMAD，TFLOPS 563 → 570 不变：
       **IMMA 走 tensor pipe、IMAD 走 FMA pipe，真正并行，4:1 比例是免费的**。
       同理 `load_generic` 的 8 次标量 LDS 换成 1 次 `ldmatrix`（LDS 36→16 已证实进 SASS），
       端到端只从 1519 → 1533 —— LDS 也不是主力。
     - **ncu 给出的真数字**：ffn kernel **138 TFLOPS = 586 上限的 23.6%**，
       且 mma 条数只比理想多 **1.18×**（算术几乎无浪费）。
       所以时间不在算术里，**在 grid 形状里**：
       tail-wave 效率 `ffn gate/up 87.7%` / **`ffn down 70.2%`** / `attn qkv 99.4%`
       —— ffn down 那个 GEMM 白扔 30% 的 block 槽位。
     - **但"按波效率自适应选 njsplit"实测更差**（1516 vs 固定值的 1536）。
       几何模型预测不了真实最优，最优 njsplit 是经验值。
       **不要再用"波数对齐 SM 数"来推 njsplit**，只能扫。
     - **stream-K（按 K 切分）已实测，此路不通，不要重走**（2026-11）。
       动机曾看起来很硬：27B ffn down 是 `rows=5120 cols=17408`，8 warps 下一 block
       吃 128 行 → 只有 **40 blocks / 114 SM（35% 占用）**，且 K 长达 68 个 super-block。
       实现路径是给 `q35_k_mmq_mma_iq` 加 `ksplit`/`grid.z`，切片写独立部分和平面，
       再用 `q35_k_mmq_kreduce` 按 z 升序折叠（**不用 atomicAdd**：那会乱序 float 求和，
       2026-10 就是因此被否的，而按 z 折叠是确定性的）。
       数值上完全成立——先按 §4 条款 6 建了 f64 参照（`/tmp/sc/streamk_ref.cu`）+ f32 分叉检查
       （`/tmp/sc/f32check.cu`）：切分本身 rel=0（f64 无损），f32 侧 rel≈1e-7，
       比 MVCHECK 的 2e-2 容差小五个数量级。**所以失败不是精度问题，是流量问题。**
       实测端到端 tok/s（525-token IQ3_XXS）：
       `ksplit=0`（基线）1531 / `auto` 1500 / `2` 1365 / `4` 1328 / `8` 1226 —— **单调变差**。
       根因是折叠代价：我的部分和平面是 `ksplit × tokens × rows` 个 float，
       ffn down 在 ksplit=8 时是 **86 MB** 额外流量（权重本身才 38 MB），
       ncu 单独计时 `q35_k_mmq_kreduce` 就 **95–347 µs**，而整个 GEMM 才 566 µs。
       另一个隐性代价：切完 K 每 block 只剩 23 个 super-block，
       prologue/epilogue 的固定开销占比反而上升。
       **llama.cpp 能用 stream-K 是因为它的 fixup buffer 只存每个 block 的边界 tile**
       （`mul_mat_q_stream_k_fixup` 的 `tmp_last_tile` 是 `J*I` 每 block，且靠连续的
       kbc 空间切分只处理跨界 tile），总量约等于**一个**输出矩阵（10.8 MB）而不是 ksplit 份。
       要追这条路必须整套移植那个 kbc 映射，代价远大于本轮收益预期，暂不投入。
     - **prefill 的组成（PROF，525-token）**：`mv_batchk` 73%、`gdn_step` 8.7%、
       `gdn_prelude` 7.3%、`add_resid` 4.3%、`rope64_rows` 2.6%。GEMM 之外已无大头。
     - **`QWEN35_ABLATE` 的坑（本轮踩到）**：关掉 `rmsnorm` 测得 1658 tok/s（+7%），
       看着像大发现，但**输出变成 `!!!!!!!!`**——污染后的 NaN/异常值让后续计算走了捷径。
       **ablation 前必须先看输出是否仍然通顺**，否则量到的是"算得更少"而不是"算得更快"。
       同样关掉 `add_resid,swiglu` 的 +6.8% 也不可信。只关 `rope64_rows`/`copy_range`
       这类输出仍正常的项，速度没有提升 —— 说明非 GEMM 部分确实只有那 27%。

## 6. 手动运行与性能对比（模型与 target）

> 固定 prompt 的"token 流探针"已完全退出验收（见 §4 规则）：既不是判据，也不是
> 提交前必跑的步骤。以下 target 只用于手动调试与性能对比。

- 量化格式覆盖很广，`make run0.8b` 会跑全部 22 个 GGUF 变体；单个变体按需跑
  （如 `make run0.8biq3xxs`、`make run27biq3xxs`、`make run0.8biq2xxs-llamacpp`）。
- 模型位于 `/mnt/Qwen3.5-0.8B-GGUF/` 等目录；`-ud` 变体用 `make run0.8bud`。
  **文件名是大小写混合的**：`/mnt/Qwen3.5-0.8B-GGUF/Qwen3.5-0.8B-Q8_0.gguf`（大写 `Qwen3.5-0.8B-` 前缀
  + 大写格式名）。写成 `qwen35-0.8b-q8_0.gguf` 会得到 `OpenFailed`——这个错看起来像"模型坏了"，
  其实是路径不存在。
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
