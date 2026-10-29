# CUDA 后端实施简报（backend.Exec 版）

> 目标：在已重构的分层（`gguf → backend → backend_cpu → engine`）上，新增一个实现
> `backend.Exec` 的 CUDA 后端。**不改 `engine.zig`**；量化字节直接驻留显存，GPU 上做
> 量化点积（不再学旧实现把 K/IQ 在宿主 dequant 成 f32 —— 27B 需要 ~108 GB，装不下）。
>
> 本文件记录 2026-10 调研结论；所有锚点均为 `path:line`。

---

## 1. Exec 接口适配表

| vtable 方法 | CPU 实现 | CUDA 侧做法 |
|---|---|---|
| `loadLinear` | `.quant` 直接指 mmap，不拷贝（`backend_cpu.zig:525-529`）；`.f32` 分配 | `.quant`：`cudaMalloc(data.len)` + h2d 原样字节；返回的 `QuantRow.data` 换成设备指针。`.f32` 同 |
| `freeLinear` | `vFreeLinear`（`backend_cpu.zig:735`） | 必须实现：`cudaFree` 设备指针。注意 `LinearW.deinit`（`backend.zig:35-39`）只释放 `.f32` 的宿主内存，**设备内存只能靠 `freeLinear` 释放** |
| `allocScratch(max_cols)` | 宿主 `q8` 缓冲，34 B / 32 elems（`backend_cpu.zig:98`） | `cudaMalloc` 同尺寸设备缓冲 + 复用宿主 pinned staging。**Q8_0 只需 `d`（34 B），但 mmvq 的 `vecdotq` 走 `block_q8_1`（36 B，`ggml-common.h:257-268`）——两套缓冲都备** |
| `allocScratchK(max_cols, k)` | `q8k` / `q8kK`（292 B / 256 elems） | `cudaMalloc`；若走 mmvq，K-类型激活用 `block_q8_K`（`ggml-common.h:370-375`） |
| `dequantRow(dst, w, row)` | `dequantRowInto`，按 `blocks_per_row` 逐 row-block（`backend_cpu.zig:577`） | **频次仅每 token 1 次**（`engine.zig:1189`，prefill 每 token 1 次），output 4 KB / 行（0.8B hidden=1024）。可先 d2h 到宿主由 CPU 算（0 成本），P2 再换 GPU kernel |
| `matvec(out, w, x)` | 每行一次 C kernel（`backend_cpu.zig:142-271`） | `out`/`x` 都是**宿主 slice**（`backend.zig:107-115`）→ h2d x、launch、d2h out |
| `matvecBatchK` | prefill 每 chunk 每线性 1 次（`engine.zig:1232-1265`） | 同上，K 行一次上传 |
| `profPhase/profReset/reportProfile` | `backend_cpu` 的 prof（`backend_cpu.zig:49-80`） | 先空实现；P3 用 `cudaEvent` 计时 |

调用频次（一次 decode，24 层）：`matvec` ≈25 次（`engine.zig:1326,1330,1336,1337,1390,1562,1570,1571,1623,1633,1634,1636` + lm_head `1145/1150`），`dequantRow` 1 次。

---

## 2. 数据流：激活走哪条路

三条候选：

- **(a) 每次 matvec 传激活**（引擎零改动）：0.8B ≈ **2.2 MB/token**（lm_head 输出 0.99 MB 独占），PCIe4 ≈0.11 ms，相对 CPU 单线程 70 ms/token 可忽略；27B ≈ **14 MB/token ≈0.7 ms**（30 tok/s 时占 2%）。真实成本是 ~50 次小 memcpy 的**启动延迟** ≈0.3–0.5 ms/token，用 pinned staging + 批量合并可压。
- **(b) 激活常驻显存**：需要把 `engine.zig:814-822` 起的激活 slice 语义整体改成 device slice，`engine.zig` 大改 → 违背分层目标，暂不做。
- **旧实现走的是第三条路**（`main_cuda.zig` + `kernel.cu`）：整套前向平行复刻在 GPU 侧，激活全常驻，CUDA Graph 把 ~900 次 launch 压成 1 次（`main_cuda.zig:66-73, 339-344, 692-702`）。代价是 `uploadLinear:252-273` 只认 `is_q8` 或 f32 —— K/IQ 在宿主 dequant 后上传。

**决策：P0–P2 走 (a)**（先把正确性打通，engine 不动）。

**注意力单独处理（用户 2026-10 决定：27B 要求 GPU attention）**：
25B 长上下文时，K/V cache 每步全量上传不可行（27B、2k 上下文约 0.3 GB/步），所以 K/V cache 必须
**归属后端**而不是引擎。这不等于方案 (b)：只需在 `Exec` 上新增两个方法

- `kvWrite(ctx, layer, pos, k: []const f32, v: []const f32)` — 替代 `engine.zig:1594-1595` 的 `@memcpy`；
- `attention(ctx, out, layer, q: []const f32, n_kv: usize, gating: []const f32)` — 替代
  `engine.zig:1596-1624` 的整段 dot/softmax/加权求和。

CPU 后端把这两者实现为对宿主 `cache.k`/`cache.v`（`engine.zig:872`）的原地操作，行为不变；
CUDA 后端让 cache 常驻显存、用 `k_gqa_partial:704` + `k_gqa_reduce:790` 计算，只上传 q 与取回 out
（每 token 每层 `q_dim + q_dim` 个 f32，27B 约 8 KB × 65 层 ≈ 0.5 MB）。激活仍按 (a) 每次上传。
**P4（可选，暂不做）**：把激活整体搬上显存。

---

## 3. Kernel 清单与阶段

目录：`src/kernels/cuda/`（AGENTS.md §1 规定 CUDA C/C++ 放这里）。

| 文件 | 内容 | 来源锚点 |
|---|---|---|
| `common.cuh` | `block_q4_0/q8_0/q8_1/q4_K/q6_K/iq*` 的设备侧结构体、`__dp4a`/`__hfma2` 帮助宏；网格帮助 `q35_grid`（现 `kernel.cu:1035`）、f16 存取 `q35_st_scale`（`kernel.cu:146`） | `ggml-common.h:192-270, 370-375` |
| `vecdotq.cuh` | `vec_dot_q8_0_q8_0`、`vec_dot_q4_0_q8_0`、`vec_dot_q4_K_q8_1`、`vec_dot_q6_K_q8_1`、IQ 系列 | `vecdotq.cuh:246, 774, 118, 142, 175, 206, 367-676` |
| `convert.cu` | `dequantize_block_*`（`dequantRow` 用）；**需新增** `quantize_row_q8_1`（旧 `k_quantize_q8` 只写 34 B 的 `d`，缺 `s`，`kernel.cu:170-197`） | `convert.cu` |
| `mmvq.cu` | 单 token / 小 batch matvec（每行一 warp 或一 block，量化权重 × q8 激活） | `mmvq.cu`, `mmvq.cuh` |
| `mmq.cu` | 多 token GEMM（prefill，`matvecBatchK`） | `mmq.cu`, `mmq-load-tiles.cuh` |
| `elementwise.cu` | 暂不需要（rmsNorm/rope/silu 留在引擎 Zig，见 `backend.zig:18-21`） | — |
| `api.cpp` | 与 Zig 的扁平 `extern "C"` 边界 | — |

阶段划分（每阶段都有端到端可跑 + 对拍）：

- **P0：只做 Q8_0**，`matvec` + `dequantRow` + `matvecBatchK`，端到端跑通 `Qwen3.5-0.8B-Q8_0.gguf`。
  判据：`-n 24 --threads 1` 输出与 CPU 版**逐字相同**，`QWEN35_DUMP_IDS` md5 一致。
- **P1：0.8B 全集**（Q8_0 / BF16 / F16 / Q4_0 / Q4_1 / Q5_0 / Q5_1 / IQ4_NL / Q3_K_S,M / Q4_K_S,M / Q5_K_S,M / Q6_K / Q2_K_XL / IQ2_XXS / IQ2_M / IQ3_XXS / IQ4_XS / Q8_K_XL）。
  **砍掉**（用户 2026-10 决定）：MXFP4 / NVFP4 / Q1_0 / Q2_0 —— CPU 侧都很少用；且 MXFP4/NVFP4 的硬件
  块缩放指令在 sm_120，4090（sm_89）需软件模拟，不值得。
  判据：`make run0.8b` 剩余变体输出通顺且首个 16 token id 与 CPU 一致。
- **P2：`Exec.attention` / `kvWrite` + 显存常驻 K/V cache**（见 §2 末）。
  判据：CPU 后端经同一接口跑出的结果与原实现逐字相同（这一步是纯重构）；
  CUDA 后端 attention 与 CPU 版 logits top-5 一致。
- **P3：27B 主要类型**（IQ3_XXS / Q4_K_M / Q6_K / Q8_K_XL）。判据同 P1，另加全部权重驻留显存成功。
- **P4：prefill GEMM（mmq）+ CUDA Graph + prof**。判据：prefill tok/s 超过 CPU 版 ≥5 倍，decode 给出基线。

---

## 4. 构建集成

**推荐方案 B**：`build.zig` 用 `addSystemCommand(nvcc -arch=sm_89 -O3 -c src/kernels/cuda/*.cu)` 产出 `.o`，再 `addObjectFile` 链进 exe。风险点与对策：

- **禁 `-rdc`、禁 CUB**（device-link 会破坏 Zig 的链接流程；本项目不需要 CUB）。
- 架构写 `-arch=sm_89`（本机 4090D，compute_cap 8.9），不写 `-arch=native` —— 后者让构建不可复现也无法跨机。
- 缓存：`.o` 由 nvcc 生成，Zig 的 cache 不跟踪 `.cu` 依赖，需要 `b.addSystemCommand` 里显式列出输入的 `addFileArg`/`addDirectoryArg`，否则改 `.cuh` 不重编。

备选方案 A（沿用 Makefile 两步链：`zig cc -c` 出 host.o → nvcc 编译+链接）继续可用作 fallback，把 `src/main_cuda.zig` 换成新的 host 入口即可。

**单 exe**：`qwen35 run <gguf> ... --backend cuda`，在 `main.zig:130-160` 的 CLI 解析里加一支，映射到 `backend.Kind`（`backend.zig:155-161` 加 `.cuda`）。CUDA 后端**惰性加载** `libcudart`（`std.DynLib`），没有 GPU 的机器上 `--backend cpu` 照常工作，二进制不硬依赖 cudart。

---

## 5. 正确性对拍

1. **逐块对拍**（无 GPU 也可跑）：`src/kernels/cuda/vecdotq.cuh` 用 `-DCPU_ORACLE` 把 `__dp4a` 等 intrinsic 换成标量实现，与 `src/kernels/x86/iq_kernels.c` + Zig `quant.rowDotXxx` 三方对同一块字节求点积。
2. **端到端 token id**：`QWEN35_DUMP_IDS`（`main.zig:38-50, 204-212`）**改动量 0**，CPU/CUDA 各跑一次比 md5。
3. **logits 对比**：新增 `QWEN35_DUMP_LOGITS`，在 `forwardLogits` 返回处（`engine.zig:1178` 附近 `self.logits`）打印 top-5 `(id, value)`，约 30 行。判据：top-5 id 一致，`|Δlogit| < 1e-2`（量化激活差异允许的量级，需实测标定）。
4. **自检子命令**：`qwen35-cuda --selftest <gguf>`，对每种类型随机抽 32 行做 GPU dot vs CPU dot 全量比对。

---

## 6. 旧代码处置

`src/main_cuda.zig`（1158 行）+ `src/kernel.cu`（1739 行）是**平行前向**，不是后端。

**可迁移的设备 kernel**（`kernel.cu`）：`k_rmsnorm_rows:327`、`k_rmsnorm_strided:355`、`k_l2norm_rows:382`、
`k_l2norm_strided2d:1108`、`k_rope:405`、`k_rope_batch:1240`、`k_rope2:1547`、`k_silu_mul:424`、
`k_sigmoid_mul:430`、`k_add:436`、`k_argmax:441`、`k_conv1d:485`、`k_conv1d_batch:1163`、
`k_beta_gamma:500`、`k_beta_gamma_batch:1190`、`k_gdn_state:526`、`k_gdn_state_batch:1271`、
`k_gated_rmsnorm:594`、`k_attn:626`、`k_attn_batch:1461`、`k_gqa_partial:704`、`k_gqa_reduce:790`、
`k_kvstore:1419`、`k_kvstore_batch:1436`、`k_kvstore2:1578`。
（上面行号是各函数的签名行，共 25 个；此外 `q35_grid:1035`、`q35_st_scale:146` 这类小工具直接进 `common.cuh`。）

**必须废弃**：`k_matvec_f32:297`、`k_embed_row:314` / `k_embed_rows:1597`（f32 权重路径）、
`k_gemm_q8:819`（被 mmq 取代）、`k_quantize_q8:170` / `k_quantize_q8_batch:201`（只写 34 B 的 `d`，缺 `block_q8_1` 的 `s`，mmvq 需要）。

**可作骨架**：`k_matvec_q8:239` —— 结构是"每行一 block、Q8_0 权重 × Q8_0 激活"，正是 mmvq 的简化版，P0 可直接改出来。

`main_cuda.zig` 整套（含 `Gpu` 结构、`pb_*` 乒乓缓冲、CUDA Graph 封装）**不复用**，因为新架构的激活留在宿主，显存里只有权重与 scratch。

---

## 7. 落地步骤清单

| # | 步骤 | 判据 | 依赖 |
|---|---|---|---|
| S1 | `backend.zig` 加 `Kind.cuda`；`src/backend_cuda.zig` 骨架（`loadLinear/freeLinear/scratch/matvec` 全走 CPU 转发） | `--backend cuda` 能跑通 0.8B Q8_0，与 CPU 逐字相同 | — |
| S2 | `build.zig` 接 nvcc（方案 B），`api.cpp` 空实现 + `q35_cuda_init` | `make` 一条命令出带 CUDA 的 exe | S1 |
| S3 | 权重上传：`.quant` 原样字节 h2d，`freeLinear` cudaFree；显存统计日志 | 0.8B 全权重驻留，打印 MB 数与预期一致 | S2 |
| S4 | `k_quantize_q8_1` + `mmvq` Q8_0 路径（改 `k_matvec_q8:239`） | Q8_0 端到端 tok id 一致，top-5 logits 一致 | S3 |
| S5 | `--selftest`：Q8_0 GPU dot vs CPU dot 抽 32 行全等 | 全等（bit-exact i32 累加） | S4 |
| S6 | `convert.cu` 的 dequantize_block_* + `dequantRow` GPU 路径 | 0.8B 各类型 embedding 行与 CPU 逐 float 相等 | S5 |
| S7 | `vecdotq.cuh` 覆盖 32 元素块族（Q4_0/Q4_1/Q5_0/Q5_1/IQ4_NL/MXFP4） | `make run0.8b` 中这些类型 tok id 一致 | S6 |
| S8 | `vecdotq.cuh` 覆盖 256 元素块族（K / IQ） | 同上 | S7 |
| S9 | `matvecBatchK` → mmq（多 token GEMM） | prefill 后 logits top-5 与 CPU 一致（容差内） | S8 |
| S10 | 27B IQ3_XXS / Q4_K_M / Q6_K / Q8_K_XL 跑通 | 显存装得下，输出通顺 | S9 |
| S11 | CUDA Graph + prof + 性能基线写入 `BENCH.md` | 给出 tok/s 数字，与 llama.cpp 同机对比 | S10 |

---

## 8. 风险与坑

- **R1 Zig↔CUDA 指针/对齐**：`QuantRow.data` 是 `[]const u8`（无对齐保证），设备 pointer 塞回去要 `@ptrFromInt`；`cudaMalloc` 给 256 B 对齐，无问题。反向（设备指针伪装成 `[]u8`）需要在 `QuantRow` 加注释，避免有人拿它做宿主 memcpy。
- **R2 CUDA Graph 与动态形状**：旧实现靠定长 buffer + Graph 压掉 ~900 次 launch；新架构 `matvec` 的 `rows/cols` 随层变化（`n_ff` vs `hidden` vs `vocab`），Graph 需要按 (rows, cols) 分桶或多 Graph，P3 再谈。
- **R3 sm_89 指令集**：`__dp4a` 自 sm_61 可用，`__hfma2`/`__hadd2` 自 sm_53；sm_89 两者都有，无坑。**block-scaled 格式（MXFP4/NVFP4）的硬件指令在 sm_120，本卡不可用**，需软件模拟。
- **R4 Q8_1 scale 布局差异**：`block_q8_1` 是 `{d, s}` 18→36 B，其中 `s = d * sum(qs)`（`ggml-common.h:257-268`）；本项目 Zig/C 侧 Q8_0 只有 `d`（34 B）。**混用会静默算错**——`vecdotq.cuh` 的部分 `vec_dot_*_q8_1` 直接用 `s` 而不是重新求和，必须保证量化 kernel 真的写了 `s`。
- **R5 MTP/nextn 层**：`engine.zig` 已在 `load` 时把 `block_count` 减掉 `nextn_predict_layers`（`engine.zig:757-762`），但 GGUF 里若仍有 `blk.<N>.` 张量，`loadLinear` 需跳过而不是报错。**待验证 27B GGUF 是否真带此类张量**。
- **R6 K/V cache 归属**：新架构里 K/V cache 是引擎宿主内存（`engine.zig` 的 `self.k`/`self.v`），GPU attention 需要每步 h2d —— 与旧实现常驻显存不同。若 27B 长上下文需要 GPU attention，就得提前做 §2 的方案 (b)。
- **R7 设备内存泄漏**：`LinearW.deinit`（`backend.zig:35-39`）只释放 `.f32` 的宿主内存，设备内存在 `freeLinear` 之外没有任何释放点。CUDA 后端必须在 `loadLinear` 里把设备指针登记到一个 owned 列表，`freeLinear` / `deinit` 两处都清。
- **R8 CUDA 版的可移植性**：`-arch=sm_89` 硬编码使二进制在其它显卡上不跑；若要发版，改成 `-arch=sm_89 -code=sm_89` + 运行期用 `cudaGetDeviceProperties` 检查，不支持就回退 CPU 并打印原因。

---

## 9. 需要拍板

1. **exe 是否硬依赖 cudart**：推荐"惰性 `std.DynLib` 加载 + 运行时 `--backend cuda`"，代价是多一层函数指针；若接受 `make` 只在装了 CUDA 的机器上可用，可以静态链。
2. **P1 的类型名单**：上表 22 个变体是否有需要砍掉的（例如 MXFP4/NVFP4/Q1_0/Q2_0 这类本项目 CPU 侧也很少用的）。
3. **27B 是否必须 GPU attention**：若要求长上下文（>8k）在 4090 上跑，R6 会迫使提前做方案 (b)，工作量翻倍；若只要 decode 吞吐，P3 的 GGUF 常驻显存 + 宿主 attention 已够。
