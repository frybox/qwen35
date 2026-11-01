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
