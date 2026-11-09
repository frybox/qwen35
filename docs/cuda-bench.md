# Qwen3.5-0.8B CUDA 速度对比：qwen35 (`--backend cuda`) vs llama.cpp (llama-cli, GPU)

日期：2026-10-31。原始数据：`bench_raw/cuda_qwen35.tsv`（69 行）、`bench_raw/cuda_llamacpp.tsv`（66 行）。

## 测试口径

- **机器**：Intel Xeon Gold 6430（32 核）+ **NVIDIA GeForce RTX 4090 D**（24564 MiB，driver 580.178.04）。跑前 `nvidia-smi` 确认无其它 GPU 进程，全程串行、一次一个引擎。
- **模型**：`/mnt/Qwen3.5-0.8B-GGUF/` 全部 22 个 GGUF（排除 mmproj）。
- **prompt**：`天空为什么是蓝色的？`；**`-n 48`**；贪心（temperature 0）；两侧各 **3 次**，表中取**中位数**。
- **qwen35**：`make run0.8b-cuda`（= 每个量化一个 target，命令
  `zig-out/bin/qwen35 run <gguf> 天空为什么是蓝色的？ -n 48 --threads 1 --backend cuda`；
  另含 1 个不带 `-n 48` 的 Q8_0 长生成 run）。构建 = `make cuda`
  （zig 0.17.0，ReleaseFast，`-Dcuda=true`，HEAD `7dd2d19` + 未提交的 `-n` 语义 WIP，见下）。
- **llama.cpp**：`make run0.8b-llamacpp-cuda`（= 每个量化一个 target，命令
  `/mnt/Qwen3.8-27B-GGUF/llama-server/llama-b10707/llama-cli -m <gguf> -p "天空为什么是蓝色的？" -n 48 -t 1 --single-turn`），
  build `b10707-62acc89c2`，默认设备（CUDA GPU），无 `-dev none`。
- **token 口径**：qwen35 当前（WIP）`-n` = **预测 N 个新 token**（与 llama-cli 的 `-n` 一致），
  两侧 decode 均为 **48 tokens**（qwen35 输出 `Generate: 48 tokens`，stop = max tokens）。
  旧 `-n`（总序列预算）语义下只会生成 31 个 token。
- **prefill 口径差异（不可直接比绝对耗时）**：qwen35 走 chat 模板 → **17** prompt tokens；
  llama-cli 裸 prompt → **5** tokens。prefill tok/s 只能作参考。
- 输出内容不同但均正常：qwen35（模板内）直接中文作答；llama-cli（裸 prompt）输出 thinking
  文本（英文）。两侧 48 个 decode token 数相同，速率可直接比。

## 结果（每格 = 3 次中位数）

| 文件 | q35 prefill tok/s | q35 decode tok/s | q35 总耗时 s | llama prefill tok/s | llama decode tok/s | llama 总耗时 s | decode 比值(q35/llama) | 总耗时比(llama 快) |
|---|---|---|---|---|---|---|---|---|
| Q8_0 | 104.9 | 51.6 | 1.092 | 402.6 | 351.8 | 0.149 | 0.147 | 7.3× |
| BF16 | 84.5 | 23.1 | 2.284 | 364.4 | 285.0 | 0.182 | 0.081 | 12.5× |
| IQ4_NL | 107.9 | 58.9 | 0.973 | 367.8 | 350.3 | 0.151 | 0.168 | 6.5× |
| IQ4_XS | 97.3 | 56.0 | 1.033 | 362.9 | 357.5 | 0.148 | 0.157 | 7.0× |
| Q3_K_M | 101.4 | 58.0 | 0.996 | 340.3 | 335.8 | 0.158 | 0.173 | 6.3× |
| Q3_K_S | 100.9 | 57.3 | 1.006 | 364.8 | 342.8 | 0.154 | 0.167 | 6.5× |
| Q4_0 | 109.2 | 59.9 | 0.958 | 341.6 | 361.2 | 0.148 | 0.166 | 6.5× |
| Q4_1 | 102.9 | 56.0 | 1.022 | 364.9 | 356.2 | 0.148 | 0.157 | 6.9× |
| Q4_K_M | 96.7 | 58.4 | 0.998 | 340.3 | 357.3 | 0.149 | 0.163 | 6.7× |
| Q4_K_S | 95.5 | 55.1 | 1.049 | 363.3 | 356.7 | 0.148 | 0.155 | 7.1× |
| Q5_K_M | 92.8 | 54.0 | 1.072 | 368.0 | 369.0 | 0.144 | 0.146 | 7.5× |
| Q5_K_S | 95.9 | 56.8 | 1.023 | 376.3 | 364.5 | 0.145 | 0.156 | 7.1× |
| Q6_K | 94.6 | 48.5 | 1.170 | 387.8 | 349.7 | 0.150 | 0.139 | 7.8× |
| UD-IQ2_M | 101.8 | 56.4 | 1.018 | 326.9 | 310.1 | 0.170 | 0.182 | 6.0× |
| UD-IQ2_XXS | 100.9 | 55.2 | 1.037 | 310.6 | 351.6 | 0.153 | 0.157 | 6.8× |
| UD-IQ3_XXS | 101.3 | 55.5 | 1.033 | 299.5 | 345.2 | 0.156 | 0.161 | 6.6× |
| UD-Q2_K_XL | 101.5 | 58.5 | 0.989 | 283.4 | 335.1 | 0.161 | 0.174 | 6.1× |
| UD-Q3_K_XL | 99.2 | 53.6 | 1.066 | 297.2 | 347.0 | 0.155 | 0.155 | 6.9× |
| UD-Q4_K_XL | 92.1 | 46.8 | 1.211 | 315.7 | 356.9 | 0.150 | 0.131 | 8.1× |
| UD-Q5_K_XL | 96.5 | 47.4 | 1.188 | 332.3 | 348.0 | 0.153 | 0.136 | 7.8× |
| UD-Q6_K_XL | 101.7 | 46.0 | 1.211 | 362.1 | 345.3 | 0.153 | 0.133 | 7.9× |
| UD-Q8_K_XL | 93.7 | 32.1 | 1.675 | 389.0 | 312.8 | 0.166 | 0.103 | 10.1× |

总耗时 = prefill(17/5 tok) + decode(48 tok)，不含模型加载。

## 结论

1. **decode：llama.cpp 全面领先约 6.4×（比值中位 0.156，平均 0.150 → 6.4–6.7×）**。
   qwen35 全部 22 个量化 decode 中位 23.1–59.9 tok/s；llama.cpp 285.0–369.0 tok/s。
2. **prefill：llama.cpp 快约 3.5×**（比值中位 ≈0.28）。注意两侧 prompt token 数不同
   （17 vs 5），tok/s 只作参考；若比绝对耗时，qwen35 prefill ≈0.16–0.20 s，llama ≈0.012–0.018 s。
3. **总耗时（prefill+decode）**：llama.cpp 中位快 **6.9×**（0.144–0.182 s vs 0.958–2.284 s）。
4. **qwen35 的格式分布比 llama 平坦得多**：llama decode 几乎不随量化变化（285–369，极差 1.3×），
   qwen35 极差 2.6×。qwen35 的短板：
   - **BF16 23.1 tok/s**（全表最低，比值 0.081）——GPU 后端上 BF16 路径仍是最慢一档；
   - **UD-Q8_K_XL 32.1 tok/s**（比值 0.103）；
   - **Q6_K 48.5 / UD-Q4_K_XL 46.8 / UD-Q5_K_XL 47.4 / UD-Q6_K_XL 46.0**（比值 0.131–0.139）。
   最好的一项是 Q4_0（59.9）、Q4_K_M（58.4）、IQ4_NL（58.9）、UD-Q2_K_XL（58.5）。
5. **持续 decode（无上限 run）**：`run0.8b-cuda` 末尾的 Q8_0 不带 `-n` 的 run 生成到 EOS
   （3 次均 381 tokens），qwen35 **42.2–44.3 tok/s**——比 48-token 的 51.6 低约 15%
   （KV 增长，符合预期）；llama 侧本次未跑无上限口径。
6. **对比 CPU 时代（BENCH.md v3/v4）**：CPU 单线程上 llama 快约 2.7×（比值中位 0.378）；
   上 GPU 后 llama 侧 decode 10–13 → 285–369 tok/s（约 25–30× 提升），qwen35 侧
   1.4–8.6 → 23–60 tok/s（约 4–10× 提升），**相对差距从 ~2.7× 拉大到 ~6.4×**。
   GPU 上 llama.cpp 的 i-quants/UD 系与 qwen35 CPU 时代的短板（IQ2/IQ3 等）已无对应——
   llama 侧所有量化都在 300+ tok/s 一条线上。
7. **正确性**：qwen35 CUDA 侧 69 次 run（22 量化 × 3 + Q8_0 长 run × 3）输出全部为正常中文
   （含 天空/蓝/散射 关键词、无西里尔/泰文/阿拉伯/替换字符等乱码特征）；
   llama.cpp 22×3 次 run 全部正常产出。

## 注意事项 / 数据说明

- **pass 污染（已剔除）**：首轮的 pass 3 qwen35 在第 8 个 run（Q4_1）因**另一终端并发
  `make`（CPU-only 构建）替换了 `zig-out/bin/qwen35`** 而失败
  （`IQ codebook upload failed (cudaError 98)`）。qwen35 数据全部取自重建 CUDA 二进制后的
  **pass 4–6**（`-n` WIP 语义，decode 48 tokens）；llama 数据取自 **pass 1–3**（全 rc=0）。
- **pass 4 首样本偏低**：约 10 个文件的 pass 4 首样本 decode 只有 25–39 tok/s
  （其余各 pass 为 46–61），疑似与当时并发 `make test` 的 CPU 负载重叠；
  每个文件只有 1 个离群样本，中位数不受影响（BF16/UD-Q8_K_XL 的三次都低，是真实水平）。
- **qwen35 `-n` 语义为 WIP**（`src/main.zig`/`src/engine.zig` 未提交改动：
  总序列预算 → 预测 N 个新 token，与 llama-cli 对齐）。本表数值属于该 WIP 构建；
  旧语义下 qwen35 只 decode 31 tokens，速率口径不变但绝对耗时会变。
- 单线程（`--threads 1` / `-t 1`）指 CPU 侧线程数；两侧量化 matvec 都在 GPU 上。
- llama-cli 输出为 thinking 流（裸 prompt 触发），qwen35 为直接作答（chat 模板），
  内容不可逐字比，速率可比。
- 原始逐次数据：`bench_raw/cuda_qwen35.tsv`（列：pass/file/kind/prompt_tok/prefill_tok_s/
  prefill_s/gen_tok/gen_tok_s/gen_s；kind=capped 为 `-n 48` 的 22 量化 run，
  uncapped 为 Q8_0 长 run）、`bench_raw/cuda_llamacpp.tsv`（列：pass/file/prompt_tok_s/gen_tok_s）。

---

# 27B IQ3_XXS decode 追平 llama.cpp：逐类型带宽归因（2026-11）

上表是 0.8B 全量化矩阵。真正的长跑基准是 **`make run27biq3xxs-cuda-long`**
（`/mnt/Qwen3.8-27B-GGUF/Qwen3.8-27B-UD-IQ3_XXS.gguf`，prompt 取自 `Makefile:396`），
对手是 `make run27biq3xxs-llamacpp-cuda-long`（**54.5 tok/s**）。

## 方法：先归因，再优化

`QWEN35_MVLIST=1` 打出一个完整 decode token 的 489 次 matvec launch，按
`rows * cols/256 * block_bytes` 汇总权重流量（**每 token 10.10 GB**，加 lm_head 的
0.72 GB = 10.82 GB）；再用单进程 microbench（`/tmp/mb/mb_ok.cu`，先验算再计时）测每种
类型的实际带宽。`MB / GB_s` 就是该类型在一 token 里的耗时占比：

| 类型 | MB/token | 流量% | GB/s | ms/token | 耗时% |
|---|---|---|---|---|---|
| iq3_s | 2599.7 | 25.8 | 439 → **633** | 5.27 → 4.11 | 24.2 → 20.2 |
| iq3_xxs | 2988.5 | 29.6 | 654 | 4.57 | 21.0 |
| q4_k（含 lm_head 715 MB） | 1657.4 | 16.4 | 522 / 556 | 3.18 | 14.6 |
| iq4_xs | 1412.1 | 14.0 | 498 | 2.84 | 13.0 |
| iq2_s | 834.6 | 8.3 | 619 | 1.35 | 6.2 |
| iq2_xxs | 477.1 | 4.7 | 392 | 1.22 | 5.6 |
| iq1_s | 139.3 | 1.4 | 131 → **306** | 1.06 → 0.45 | 4.9 → 2.0 |
| q2_k | 166.9 | 1.7 | 235 | 0.71 | 3.3 |
| iq2_xs | 292.5 | 2.9 | 496 | 0.59 | 2.7 |
| q3_k | 155.4 | 1.5 | 268 | 0.58 | 2.7 |
| iq1_m | 58.5 | 0.6 | 182 | 0.32 | 1.5 |
| q5_k | 25.2 | 0.2 | 317 | 0.08 | 0.4 |

纯权重读取的理论下限是 **45.9 tok/s**（1090 cuMemcpy 的带宽上限远高于此，
所以瓶颈是**访问模式**而非设备带宽：`QWEN35_CUDA_MEMONLY=1` 只到 27.0 tok/s）。

## 已完成的四个 split（P3）

decode matvec 占 decode 时间 **87%**；把 256 元素权重块按 K 维切给同一 warp 的多个 lane、
最后 butterfly 归约，是唯一有效的杠杆（`q35_k_mul_mat_vec_q_split_t<WARPS,SPLIT>`，
`q35_split_units()` 声明每种类型的合法切分数）。**切分边界必须让每个 lane 的权重读取连续**：

| 提交 | 类型 | units | 依据 | 收益 |
|---|---|---|---|---|
| `3997196` | iq3_xxs | 8 | 8 个 `ib32` 组 | 17.60 → 20.08 tok/s |
| `7deaa48` | iq2_s | 8 | 同上（dp4a body） | → 21.08 |
| `9031a4b` | q4_k | 4 | 4 条 64 元素 q4 字节行 | 298 → 553 GB/s |
| `c45534e` | iq4_xs | 4 | `_acc` 每步 `h >>= 4` 一次，一步=一个 64 元素 `ib` | 22.7 → 23.91 |
| `97dbeef` | iq2_xxs / iq2_xs | 8 | 8 个 `ib32` 组 | 23.91 → 25.29 |
| `14358dc` | iq1_s | 8 | 8 个彼此独立的 `ib`（各带 qh u16 + 2 个 bsum） | IQ1_S 131 → 306 GB/s；25.48 → 25.94 |
| `2f9b048` | iq3_s（余下最大项） | 8 | 从绝对 `ib32` 推 nibble，允许奇数 `lo` | IQ3_S 439 → 633 GB/s；25.94 → **26.94** |

## 三条血泪教训（都已写进代码注释）

1. **`q35_split_units` 必须是无副作用常量表**。它标了 `__device__ __host__`，
   split kernel 每 block 调一次；曾在此处加 `getenv()` 做 A/B 开关，host 侧正常执行，
   设备侧结果被静默污染 —— 3579 个 token 的乱码，而 MVCHECK 全 0。
   A/B 开关只能放在 launcher（host-only）。
2. **切分数由 `_acc` 的循环步长推导，不能猜**。IQ4_XS 是 4 不是 8（`h >>= 4` 每步一次）；
   三个错误的 rebase 都只影响靠后的 slice，`lo=0` 仍正确，模型看起来能跑一个 token。
3. **让 `_part` 拒绝一个 range 就等于造假速度**。把 IQ3_S 的 `min_per` 放宽到 1 而没改 body，
   旧的 `if (lo & 1) return 0.0f` 让一半 slice 算 0 → "27.03 tok/s"（比真的快 4%），
   输出变成"茶，写茶，，"。切分必须能算它拿到的每一种 range。

## 复现命令

```bash
make cuda                          # -Dcuda=true -Doptimize=ReleaseFast
P=$(sed -n 396p Makefile | sed 's/.*qwen35 run [^ ]* "//; s/" --backend cuda.*//')
# 固定 -n 400 才有可比性（不同 token 数会走不同分叉，耗时不可比）
./zig-out/bin/qwen35 run /mnt/Qwen3.8-27B-GGUF/Qwen3.8-27B-UD-IQ3_XXS.gguf "$P" \
  --backend cuda -n 400
# 诊断开关
QWEN35_SPLITDBG=1      # 每次 split launch 打印 ttype/rows/cols/nblk/split/warps/shmem
QWEN35_MVLIST=1        # 打印每个 matvec 形状
QWEN35_CUDA_GRAPH=0    # 关 CUDA graph，否则 dump 读到的是 capture 期的陈旧零值
QWEN35_MVCHECK=1       # 逐类型与 CPU 后端比对（开销极大）
```

## decode 35.4 ms 的成分拆解（2026-11，27B IQ3_XXS）

这一节记录"到底慢在哪"的实测结论，**所有数字都可用给出的 microbench 复现**，
不要凭直觉再猜一次。

### 口径修正：microbench 曾经测的是 L2，不是显存

`/tmp/mb/mb_one` 用 `iq3_xxs 17408x5120` 报 353–660 GB/s，看起来"kernel 已经接近带宽上限"。
但这个形状的权重只有 **34 MB**，而 4090 D 的 L2 是 **72 MB** —— 整个缓冲区都在 L2 里，
测到的是 **L2 带宽**。把同样的 kernel 换成 **1 GB** 缓冲区（`/tmp/mb/mb_big`）后：

| 缓冲 | ms | GB/s |
|------|-----|------|
| 32 MB | 0.052 | 651 |
| 256 MB | 0.443 | 606 |
| **1024 MB** | **1.756** | **612** |

结论：**单形状 matvec 在真实显存上能做到 612 GB/s**，不是 353。
"kernel 效率不足"这个判断当时被 L2 污染了。

### 真实差距：quantize 在关键路径上

用 `QWEN35_MVLIST` dump 出真实 token 的 **500 个 matvec 序列**，把 51 个权重缓冲区
散在 **10.8 GB arena**（= 真实模型足迹，避开 L2）里重放（`/tmp/mb/mb_spread`）：

| 配置 | ms/token | GB/s |
|------|---------|------|
| 只跑 500 个 matvec | 22.67 | 483 |
| + 每 matvec 之间插入 l2norm 等小 kernel | 23.09 | 475 |
| **+ 每个 matvec 前做一次 quantize（真实 app 的形状）** | **32.24** | **340** |
| + 两者都有（≈ 真实 app） | 33.69 | 325 |

**真实 app 实测 35.4 ms / 310 GB/s，与最后一行吻合。**
两个结论因此是硬的：

1. **显存带宽不是瓶颈**：同样的 500 次 matvec 单独跑只要 22.7 ms（就是 483 GB/s，
   已经是 DRAM 的合理水平）。app 的 35.4 ms 里 **约 10 ms 纯粹是 quantize 的串行代价**。
   因为 quantize 与 matvec 在同一条 stream 上，matvec 必须等自己的 quantize 排空，
   这 10 ms 期间显存是**闲着的**。
2. **CUDA graph 无辜**：同一个 launch 序列 bare-loop 21.85 ms vs graph 21.47 ms。
   之前"graph 比 standalone 慢 13 ms"是 bench 假象（`mb_replay` 复用同一个 1 MB 输出缓冲区，
   写上被 L2 吸收，且没做 quantize）。

### decode attention 的访存（已修，`f36a45d`）

`attn_decode` 曾是 decode 里除 matvec 之外最大的一项：**`QWEN35_ABLATE=attn_decode`
能省 14.3 ms/token**（33.8 → 19.4 ms），且**与上下文长度无关**（n=60/300/900 都是 14.3）。
当时误判成 CUDA graph 的 `GraphExecKernelNodeSetParams`，花了一轮把 `slot` 改成设备标量
（`q35_attn_decode_kernel_dyn` + `q35_launch_set_int1`）——**24 次/token 的 SetParams 全部消失，
tok/s 一点没动**。所以那个归因是错的。

真实原因是**访存**，而且必须用 24 份各自独立的 per-layer KV cache 才量得出来：

- `mb_attn`（共享一份 cache，n≤300）给出 0.0174/0.0578 ms —— **这是 L2 命中，不是 DRAM**，
  被它误导了整整一轮。换成 24 份 102/141/180 MB 的 cache 后真相是
  **3.05 / 3.86 / 4.97 ms**（slot 518/717/917），只有 **33–37 GB/s**。
- 原因：一个 head 一个 block → **24 个 block 打在 114 个 SM 上（21% 占用率）**，
  而每个 lane 串行走完整个 n 行；naive kernel 还把第 `t` 行分给第 `t` 号线程，
  一个 warp 的 32 个线程读相距 4 KB 的 32 行，每行只用 `hd/kv_dim = 256/1024`。
- 两步修复，都在 `attn.cu`：
  1. `q35_attn_decode_kernel_coal`：warp 读一行的连续段、lane 铺 `j`、
     `__shfl_down_sync` 归约 → **33 → 68 GB/s**（3.09 → 1.50 ms），逐位不变。
  2. `q35_attn_score_split_kernel` + `q35_attn_softmax_kernel` +
     `q35_attn_decode_kernel_split` + `q35_attn_reduce_kernel`（`split2`）：
     **score 和 PV 两段行走都按行切成 `split` 份**（切一段没用，实测只切 PV
     反而 1.52 → 1.66 ms）→ **325 GB/s / 0.555 ms @ slot 917**，8.9×。
- **精确性**：score 的 j 求和仍是串行顺序，softmax 仍是 CPU 的 max/串行求和/除法顺序，
  唯一被二级树重结合的是 PV 的 t 求和 —— 即便这样也是 `mismatched=0 worst=0`
  （对每个 head、slot 0/5/63/300/516–519/530/700/1023 全对）。
- **必须让 `split` 留在 CUDA graph 里**才有收益（graph 关掉时 28.99 → 30.90，
  开 graph 才到 31.5）。做法是设备标量版 `q35_launch_attn_decode_split_dyn`：
  每层 **4 个常量参数节点**（score/softmax/PV/reduce+gate），
  在节点普查里各有自己的 class（`attn_score`/`attn_reduce`），永不需要 SetParams。
- 三个 bug 全是 microbench 抓到的，值得记下：
  1. softmax 拆成独立 kernel 后**每个 head 一个 block**，但引擎给所有 head 的是
     **同一份 `slot+1` 行**（原 kernel 一个 head 一个 block 所以无所谓）。
     不改就是 head>0 读到别的 head 的分数 → **输出非确定**（同一条命令跑两次 md5 不同）。
     现在 split 路径用自己的 `heads * Q35_ATTN_MAX_ROWS` scratch（`d_att_prob`）。
  2. PV kernel 的 `d_prob` 少了同样的 per-head 步长，`split2` 与 `split_dyn` 差 0.35。
  3. 文件里还留着旧 launcher 的副本，仍在跑"不切分的 pass 1"。

**结果：`QWEN35_ABLATE=attn_decode` 现在只省 1.2 ms**（31.36 → 32.55 tok/s），
即那 14.3 ms 已经基本消灭，decode 重新是 **95%+ matvec**（ablate matvec → 196.9 tok/s）。
decode **29.3 → 31.5 tok/s**（`-n 400`，`make test` 20/20，`QWEN35_MVCHECK` mismatched=0）。

### 激活量化缓存（已修）

`backend_cuda.zig` 有 `actQValid`/`actQMark` 记忆化：`d_xq` 里现成装着
`(src, family, cols)` 的 Q8_K 激活就不重复量化。两处曾经让它形同虚设：

- **key 里带了 `ttype`**。打包布局只由 **family** 决定（`actFamily` 把所有 256 元素块类型
  折叠成 Q8_K），所以 q4_k→iq4_xs→iq3_xxs 的切换每次都 miss。去掉 `ttype` 后
  hit 从 103 升到 176（**28.26 → 29.23 tok/s**）。
- **`slotExit` 用"非 matvec 即失效"**。实际只有 `rmsnorm`/`add_resid`/`l2norm`/`swiglu`/
  `copy_range`/`copy_from_host` 等会写激活缓冲区；`rope64`/`attn_decode`/`gdn_*`/`conv_step`/
  `scale_vec`/`zero_buf` 只碰 qkv/z/ab/KV cache。改成**写者白名单**后 hit 176→192
  （**29.23 → 29.39 tok/s**）。

两次改动都验证 **`QWEN35_DUMP_IDS` 80 token 逐位相同**。

**剩余的 310 次 miss 是真实工作量，不是缓存缺陷**：用 `QWEN35_QDBG` 逐条打印
`[qmiss] … last=<slot>` 可以确认，每次 miss 前面都紧跟一个**真的写了那个缓冲区**的
slot（169 次 `add_resid`、84 次 `swiglu`、63 次 `l2norm`、21 次 `rmsnorm`），
即每层一次。**缓存已经到设计上限，继续抠 key 没有收益。**

### 已否决：合并 GDN `alpha`/`beta` 的两个 matvec（`matvec2`）

`x1 -> ssm_alpha` 和 `x1 -> ssm_beta` 是两个相邻的 `matvec`，输入相同，
看起来是"共享一次激活量化"的现成机会，而且 `PROF` 里 `q8_0` 占 **20%**（6.4 ms/token，
24 层 × 2 次 = 48 次/token）。实测否掉了：

- 单独量 `24x5120` 的 mmvq：**0.0054 ms/call**，48 次/token = **0.26 ms**,不是 6.4 ms。
  那 20% 是 `stat_by_type` 记的**host wall**（含已被 graph 隐藏的 launch 开销），不是 kernel。
- 分离的两次调用本来就**共享量化缓存**（key 里没有 ttype，第二次直接 hit），
  所以"合并"省不到东西。
- 真做出来（`vtable.matvec2` + CUDA 里共享 `actQValid` 再背靠背发两个 launch）
  **稳定掉 0.5 tok/s**（31.45 → 31.0，两轮）：多出来的 `findDev` 查找、
  env 检查和重复的 stat 记账比省下的那点 probe 更贵。已完整回滚。

### 权重 32-bit 读取：4×u8 → 2×u16（活实现 +4%，纯加载 +13%）

`q35_ld_u32()`（`vecdotq.cuh`）原本写成 `p[0] | p[1]<<8 | p[2]<<16 | p[3]<<24`。
看起来编译器会合并成一条 32-bit load，**实际不会**：`-arch=sm_89` 的 PTX 里是
**四条 `ld.global.u8`**。`__builtin_memcpy`、packed struct 解引用、对 `const unsigned*`
用 `__ldg` 三种写法全都还是四条——nvcc 无法从 `unsigned char *` 推出对齐。
每个 IQ 类型的内层循环都调它，所以整个家族每个 weight word 都在发 4 倍的访存指令。

**为什么不能用一条 u32**：q3_K 的 `scales` 字段在 110 字节块的偏移 96 处，只有 2 字节对齐，
`ld.u32` 在那里直接 `misaligned address` 崩掉（实测）。
但每个 weight 块**基址都是偶数**（块大小全为偶数：q3_K 110、iq3_xxs 98、q8_0 34……），
所以 2 字节对齐的 `ld.u16` 永远合法——这是这里最宽的可用宽度。

**为什么用 generic `ld` 而不是 `ld.global`**：同一函数也被调用在非 global 地址空间的缓冲上。
强制 `.global` 会崩，而且报错**延迟一轮**才冒出来：
`constant upload failed (20480 B): operation not supported on global/shared address space`
出现在**下一个** `cudaMemcpy` 上，指向的是无关的 norm 权重上传。`.nc` 同样不行。
generic `ld` 运行期解析地址空间，正是原来那条字节链编译出来的东西。

测量（`/tmp/mb/mb_ld.cu`，2 GB >L2 缓冲，98 字节步长、每块 24 word，即真实 IQ3_XXS 访问模式）：

| 写法 | GB/s |
|------|------|
| 4× u8 链 | 742.9 |
| **2× u16** | **841.8** |

**位精确性**：`mb_exact.cu` 对 4,194,304 个 word 全比对，`0 mismatches`——u16×2 与 u8×4
得到的是同一个整数（小端 `ld` 返回的就是字节链拼出来的值）。MVCHECK 在
Q8_0 / Q4_0 / IQ4_NL / Q4_K_M / Q6_K / IQ2_S / IQ3_S / IQ4_XS / Q3_K_M / IQ2_XXS
十种格式上全部 `mismatched=0`。

27B IQ3_XXS 端到端：**32.21 → 33.66 tok/s**（`-n 400`，两次 33.67/33.65）。

### `q35_ld_u16`：同一个缺陷的第二处（+0.2%）

`q35_ld_u32` 修好之后 mmvq 的 PTX 里还剩 **10905 条 `ld.global.u8`**——因为
`q35_ld_u16()` 是同一段字节链的 16 位版本：`p[0] | p[1]<<8`，一样编译成**两条**
`ld.global.u8`（nvcc 推不出 `unsigned char *` 的对齐，原因与 u32 那处完全相同）。
改成复用 `q35_ld_u16_low()` 后 mmvq 的 u8 load 降到 **9555**，端到端
33.55 → **33.74 tok/s**（`-n 400` 两次 33.77/33.71）。

影响比 u32 小得多，因为 24 个调用点里多数是每块只读一次的 scale 字段，不在内层循环里。

### 已否决：IQ4_NL 的 16 字节码本放进 `__constant__`

`kvalues_iq4nl` 只有 16 字节、永不变，看起来该放 `__constant__`。做了，隔离单类型测试
IQ4_XS 从 **506 → 584 GB/s（+15%）**，但**端到端 33.6 → 28.3 tok/s（−18%）**。
原因是常量内存是广播式的：warp 内各 lane 读到**不同** nibble 下标时访问会串行化。
单类型 bench 上看不出来（一次只跑一种类型），整机混跑时这个代价远超省下的查表。
**`__ldg` 版本则完全无变化（33.55 vs 33.60）**——16 字节本来就是 L1 常驻，
真正的权重流才是瓶颈。两者都已回滚。

### 已否决：把 IQ2_XXS 点积改写成整数域（dp4a）

理由看起来充分：`q35_vec_dot_iq2_xxs_q8_k_acc` 是**纯 float 标量**——每个 32 元素组跑 8 次
`fadd(db*g*q8)`，完全没有 dp4a；而结构完全相同的 IQ3_XXS 走整数域、实测 644 GB/s，
IQ2_XXS 只有 349。改写后（`q35_iq_signs8` + `__vsub4` + `dp4a`，与 IQ3_XXS 同构）
**MVCHECK 全部 `mismatched=0`，数值正确——但带宽 349.3 → 349.3，一点没动。**
所以 IQ2_XXS 不是计算瓶颈，改计算路径无效。已回滚。

真正的原因在 split 阶梯上：`units=8` 与 IQ3_XXS 相同，但 iq2_xxs 块只有 **66 字节**
（IQ3_XXS 98 字节）。split=2/4/8 给 224.5 / 299.3 / 348.6 GB/s，**单调上升且被 units=8 卡住**，
即并行度不足而非算力不足。而 `_part` 的最小粒度是 8 元素（一个 dp4a），无法再切细分。

### 已否决：按 microbench 挑 warps

单类型 microbench 里 `warps=1` 明显最好（iq3_xxs 672 vs warps=2 的 643、iq2_xxs 382 vs 347），
但端到端 `-n 400` 上 warps=1 (32.18) 反而**不如** warps=2 (32.42)。
又一次证明**微基准的排名不能迁移到整机**，只能以端到端为准。

### 27B decode 的 matvec：4-warp → 2-warp（+0.7 tok/s）

`q35_coop_warps()` 在 sm_89 上原本写死 4 warps（128 线程），依据是 header 里那条
124 vs 77 tok/s 的旧测量。**那条测量早于 K-split decode 路径**：行被拆到 lane 上之后，
限制占用率的是每 block 的 shared memory（`warps * rpb * nblk` 个 float），
64 线程让驻留 block 数翻倍。27B IQ3_XXS 重测（`-n 300` 与 `-n 400` 各两轮）：

| warps | -n 300 | 长 prompt -n 400 |
|-------|--------|------------------|
| 1 | 32.18 / 32.03 | 31.89 |
| **2** | **32.42 / 32.28** | **32.30** |
| 3 | 28.90 / 28.92 | — |
| 4 | 31.76 / 31.55 | 31.59 |

`warps=3` 一律比 2 和 4 更差：launch 的 `switch` 只有 `{1,2,4,8}`，3 会落到 `default:`
即 8-warp kernel。占用率探测 `q35_coop_occ_best_warps()` 因此也改成把 3 夹到 2，
不再让一条"探测出来却执行成别的形状"的路径存在。默认值现在是 2：**31.6 → 32.2**。

### matvec 带宽随形状塌陷（微基准，`/tmp/mb/mb_ladder.cu`）

同一类型（iq3_xxs, cols=5120）只改 rows，看孤立调用的 achieved bandwidth：

| rows | MB | us/call | GB/s |
|------|----|---------|------|
| 48 | 0.09 | 3.7 | 25 |
| 512 | 0.96 | 4.6 | 217 |
| 1024 | 1.91 | 6.6 | 303 |
| 2048 | 3.83 | 9.3 | 431 |
| 4096 | 7.66 | 15.1 | 532 |
| 8192 | 15.31 | 26.4 | 608 |
| 16384 | 30.62 | 49.1 | 654 |
| 17408 | 32.54 | 51.9 | 657 |

**3.7 us 是 launch 延迟地板**，与传输量无关；657 GB/s 是这块卡的上限。
27B 每 token 的 498 次 matvec 里，96 次是 `q8_0 48x5120`（0.25 MB，纯地板），
239 次落在 8–32 MB（41% 时间），129 次 >32 MB（55% 时间）。

**注意：这是"背靠背独立 launch"的数字。** 真实 decode 是依赖链
（matvec → norm → matvec），`/tmp/mb/mb_dep.cu` 测出链式惩罚：
1.9 MB +55%，3.8 MB +39%，7.7 MB +24%，15.3 MB +14%，30.6 MB +7%。
把这两个模型套到真实的 498 次调用分布上：独立 17.4 ms，链式 19.4 ms，
而实测 `mv_batchk` 是 **24.8 ms**。剩下的 5.4 ms 尚未归因。

### 已排除的假设（都实测过，不要再试）

| 假设 | 测法 | 结论 |
|------|------|------|
| 跨 buffer 轮转导致 TLB 抖动 | `mb_rot.cu`，1/8/64/400 个独立 32.5 MB buffer | 577 vs 672 GB/s，只有 10% |
| 形状切换打断流水 | `mb_seq.cu`，按真实形状序列轮转 17 个 | 524 GB/s，正常 |
| 小 kernel 插在中间 | 每次 matvec 后加一个 rmsnorm | 524 → 509 GB/s，3% |
| elementwise 槽位耗时 | `QWEN35_ABLATE` 逐个关 | 每个都 ≤0.6 tok/s，全部被 graph 隐藏 |
| 时钟/功耗受限 | 运行中 `nvidia-smi` | 2670 MHz / 333 W / util 98% |
| CUDA graph 开销 | graph=0 vs 1 | 31.11 vs 31.57 |
| GDN alpha/beta 合并 | `matvec2` 全实现 | −0.5 tok/s，已回滚 |

### 下一步的唯一有效方向

要拿回那 ~10 ms，只有把 quantize 挪开关键路径，而不是让它更快或更少：

- **跨层流水**：为第 `i+1` 层的激活做 quantize 的同时跑第 `i` 层的 matvec。
  需要 `d_xq` 双缓冲 + 提前拿到下一个激活。引擎按 phase 产激活，天然可预取。

**已否决："llama.cpp 的 matvec 内部融合了 quantize，照它做就能消灭这个 launch"。**
这条曾经写在本节里，是错的，2026-11 逐行核对了 `/mnt/llm-dev/llama.cpp/ggml/src/ggml-cuda/`：

- **decode（mmvq）**：`mmvq.cu:1534` 在发起 matvec 之前调
  `quantize_row_q8_1_cuda(src1_d, nullptr, src1_q8_1.get(), ...)`，
  它落在 `quantize.cu:570` 的 `ggml_cuda_kernel_launch(quantize_q8_1, ...)` ——
  **独立的 kernel launch**，和本项目的 `q35_launch_quantize_q8k` 是一回事。
- **prefill（mmq）**：`mmq.cu:253` 同样先调 `quantize_mmq_q8_1_cuda(...)`，
  它在 `quantize.cu:574` 也是独立 `__global__`（`CUDA_QUANTIZE_BLOCK_SIZE_MMQ` 的 block）。
- `vecdotq.cuh` 里没有"从 f32 现算 Q8_K"的路径：所有 dot kernel 的第二个参数都是
  已量化好的 `block_q8_1` / `block_q8_K`。llama.cpp **没有**融合这一步。

所以 llama.cpp 也有这个 launch；它比我们快不是因为它省了它。
（它确实省了一点：`src1_q8_1` 分配到**一个** pool buffer，而我们的
`actQValid/actQMark` 记忆化已经做到 192 hit / 310 miss，比逐次量化更好。）
这条不要再当依据。

