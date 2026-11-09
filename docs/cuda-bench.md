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

### 下一步的唯一有效方向

要拿回那 ~10 ms，只有把 quantize 挪开关键路径，而不是让它更快或更少：

- **跨层流水**：为第 `i+1` 层的激活做 quantize 的同时跑第 `i` 层的 matvec。
  需要 `d_xq` 双缓冲 + 提前拿到下一个激活。引擎按 phase 产激活，天然可预取。
- 或者 **matvec 内部融合 quantize**（`vecdotq.cuh` 直接从 f32 现算 Q8_K 量化值），
  彻底消灭这个独立 launch —— llama.cpp 的 mmq 在 prefill 走的就是这条路。

