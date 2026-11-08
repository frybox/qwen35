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
  容差 `tests/test-backend-ops.cpp` + 手工 perplexity）。§6 的固定 prompt 运行仅作手动调试 /
  性能对比用，不进回归判据：
  - CPU/CUDA 的 token 流不可能稳定逐位相同（f32 收尾 / 归约顺序 / 超越函数 1–2 ulp，
    见 `docs/cuda-elementwise-ulp.md`；MVCHECK 全过时 token 流仍会分叉，实测见
    `src/kernels/cuda/mmq.cu:1105`）；
  - llama.cpp 默认采样是 temp 0.8 + 随机 seed，其参照输出本身不可复现。
  正确性以 kernel 级对拍为准（`make test` 三路对拍 + `QWEN35_MVCHECK`），
  端到端只作输出是否通顺的烟雾信号（完整口径见 `docs/cuda-backend-brief.md` §2bis）。
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
| `QWEN35_DUMP_OUT` / `QWEN35_DUMP_IDS` | 导出中间输出 / token id |
| `QWEN35_NO_BATCH` | 关闭 K-token 批量 prefill chunk |
| `QWEN35_PROF` / `QWEN35_PROF_POST` / `QWEN35_STAT` | 性能剖析（`PROF-M` 行由 `backend_cpu` 输出） |
| `QWEN35_MVCHECK` | **仅 CUDA 后端、仅调试用，不是计算路径**。每个量化 matvec 之后，用内嵌的 CPU 后端对同一行再算一遍并逐行比对，打印 `MVCHECK[<type>] rows= cols= mismatched= batchk= k=`。用于确认"GPU kernel 结果与 CPU 参照一致"（CUDA 逐格式验收的自动化手段）。开销极大，只在 A/B 时开 |
| `QWEN35_MVCHECK_ONLY` | 限定只对某一种量化类型做 MVCHECK（值 = `q.ttype` 的 tagName，如 `iq1_s`）。batch-K 参照需要 CPU scratch 按 K 塑形，不是每种类型在每个 launch 顺序下都能活，所以追单一 kernel 时用这个定向跑 |
| `QWEN35_CUDA_COOP_WARPS` | **仅 CUDA 后端**。decode（tokens==1）warp-per-row matvec 的 block 形状（1..8 个 warp/block），A/B 用。不设时由硬件探测决定：sm_89 → 4（本机实测最佳），其他 cc → 占用率公式（cudaOccupancyMaxActiveBlocksPerMultiprocessor 选驻留 warp 最多的形状），见 `mmvq.cu` 的 `q35_coop_warps()`。探测来源是 `caps.cu`（`q35_gpu_cc()`/`q35_gpu_sm_count()`/`q35_gpu_name()`，一次探测、缓存读，x86 侧 `cpu_caps.c` 的 CUDA 镜像） |

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

## 6. 手动运行与性能对比（模型与 target）

> 固定 prompt 的"token 流探针"已移出验收标准（见 §4 规则）。以下 target 只用于手动调试与
> 性能对比，不属于回归判据。

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
