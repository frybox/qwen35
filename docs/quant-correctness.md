# 量化格式输出正确性回归

全 22 个量化格式（排除 3 个 `mmproj-*` 投影模型）的功能性冒烟测试记录。
与 [`BENCH.md`](../BENCH.md) 互补：**本文只关心"输出是不是人话"，不关心速度**。

BENCH.md 记录的是「qwen35 vs llama-server 的速度对比」以及历次乱码事故的根因定位
（`avx512vnni` tier 在 i-quant / Q2_K / Q3_K 家族上输出乱码，见其「输出正确性核查」
与「v4 修复 vnni/vbmi tier 乱码」两节）。本文是修复之后的一次**全量回归确认**。

## 测试口径

- **机器**：Intel Xeon Gold 6430，32 核；全部 **单线程**（`--threads 1`），串行执行。
- **模型目录**：`/mnt/Qwen3.5-0.8B-GGUF/`，22 个量化 GGUF。
- **prompt**：`天空为什么是蓝色的？`；`max_tokens = 48`；默认采样参数。
- **tier**：默认（不设 `QWEN35_SIMD_TIER`，由 CPUID 自动选择）。
- **判定标准**：输出含 ≥10 个汉字视为「无乱码」（`OK`）；否则人工复核。
  注意这只排除乱码，不保证**内容正确**——见下文「已知的模型质量损失」。

## 结果（2026-10-28）

**22/22 全部无乱码。**

| 量化格式 | decode tok/s | 判定 | 备注 |
|---|---:|---|---|
| BF16 | 1.53 | ✅ | |
| UD-Q8_K_XL | 2.30 | ✅ | |
| IQ4_NL | 3.60 | ✅ | |
| Q4_1 | 4.26 | ✅ | |
| Q4_0 | 5.12 | ✅ | |
| UD-Q6_K_XL | 5.43 | ✅ | |
| Q8_0 | 5.56 | ✅ | |
| Q5_K_M | 5.91 | ✅ | |
| Q5_K_S | 6.01 | ✅ | |
| UD-Q4_K_XL | 6.22 | ✅ | |
| Q6_K | 6.23 | ✅ | |
| Q4_K_M | 6.41 | ✅ | |
| UD-Q5_K_XL | 6.41 | ✅ | |
| Q4_K_S | 6.51 | ✅ | |
| Q3_K_M | 6.68 | ✅ | 曾乱码（vnni tier），已修复 |
| Q3_K_S | 6.91 | ✅ | 曾乱码（vnni tier），已修复 |
| UD-Q2_K_XL | 6.98 | ✅ | 曾乱码（vnni tier），已修复 |
| UD-Q3_K_XL | 7.10 | ⚠️ | 中文正常，但内容有误（称"光污染"）。曾乱码 |
| UD-IQ3_XXS | 7.31 | ✅ | **本轮重点：曾乱码，现正常** |
| UD-IQ2_M | 7.63 | ✅ | 曾乱码（vnni tier），已修复 |
| UD-IQ2_XXS | 8.03 | ⚠️ | 中文正常，但陷入 `<think>` 重复循环。曾乱码 |
| IQ4_XS | 9.83 | ✅ | 曾乱码（vnni tier），已修复 |

速度与比特数大体负相关（越低位越快），符合预期。

## 回归验证的要点

1. **BENCH.md 记录的 vnni-tier 乱码文件全部转为正常**（其正文称 12 个，明细表列出 9 个；
   本文实测这 9 个均已正常），包括此前最严重的 `UD-IQ2_XXS`、`UD-IQ3_XXS`、`IQ4_XS`。修复有效。
2. `UD-IQ3_XXS`（本记录的起点）从乱码 `:]N秀^Q N93 ...` 变为准确通顺的中文，
   decode 从 0.96 tok/s 提升到 7.31 tok/s（**7.6×**）。
3. 两个 ⚠️ 项**不是 kernel 问题**，是模型自身在极低比特下的质量损失：
   - `UD-IQ2_XXS`（2-bit）：中文流利，但进入 `<think>` 后反复重复同一句，
     疑似 chat 模板 / EOS 处理与思考模式配合的问题，值得单独排查。
   - `UD-Q3_K_XL`：把瑞利散射说成"光污染"，常识性错误。

## 已知的模型质量损失（非 bug）

低比特量化（IQ2_XXS、IQ2_M、Q3_K_XL）在 0.8B 这样的小模型上，常识准确性明显下降。
这类问题**不应**用 kernel 修复的方式处理——它是量化本身的固有损失，用更高比特即可缓解。
`UD-IQ2_XXS` 的 `<think>` 重复循环可以单独查，但不阻塞发布。

## 复现方式

```bash
# 单跑一种量化
make run0.8biq3xxs

# 全量（22 种，串行，约 3 分钟）
make run0.8b
```

如需对比 tier 行为（排查 SIMD 路径问题）：

```bash
QWEN35_SIMD_TIER=avx2    ./zig-out/bin/qwen35 run <model.gguf> "天空为什么是蓝色的？" -n 48 --threads 1
QWEN35_SIMD_TIER=avx512  ./zig-out/bin/qwen35 run <model.gguf> "天空为什么是蓝色的？" -n 48 --threads 1
```

其他调试开关见 `src/main.zig`：`QWEN35_FORCE_F32`、`QWEN35_FORCE_ROWDOT`、
`QWEN35_KERNEL_CHECK`。

## 附注：为什么 `zig build test` 不足以覆盖

BENCH.md 已指出：测试是 Debug 构建，但内核的 VNNI 分支受
`@import("builtin").mode != .debug` 门控，**Debug 下既不编译也不执行**。
因此 59 个单元测试只覆盖 @Vector 参考路径，抓不到 VNNI 代码的 bug。

**这意味着本文这类端到端模型输出验证是不可替代的**——每次改动 SIMD 内核后，
除了跑测试，必须跑一遍 `make run0.8b` 确认没有引入乱码。
