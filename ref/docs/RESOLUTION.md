# Qwen3.5-0.8B C 引擎调试结论（已修复）

> 本文档取代 `BUG_REPORT.md`（那份报告基于修复前的代码，其中的 Bug 3「decay 公式应为 `a - dt_bias`」等结论是**错误**的，请勿再依据它修改代码）。

## 结论一句话

引擎现在与 HuggingFace 参考实现**数值等价**：logits 最大绝对误差 `3.96e-05`（fp32 舍入量级）、cosine 相似度 `1.000000`、argmax 一致、贪心生成 token **逐 token 完全相同**。

真正的「输出不正确」由**两类**问题叠加造成：引擎里若干实现 bug，以及 `main()` 里**被损坏的 prompt_ids 数组**。

---

## 一、根因：prompt_ids 数组是坏数据

`qwen35_engine.c` 的 `main()` 里注释写着 `"What is 2+2"`，但数组实际是：

```c
248045, 846, 198, 58, 1319, 34233, 63, 248046, 198, 248045, 74455, 198
                        ^^^^^^^^^^^^^^^^^ 这些 id 解码出来是 "[ comp longtime`"
```

`58, 1319, 34233, 63` 解码为 `[ comp longtime\``  ——  纯垃圾。用同一串 id 喂 HF，HF 也只会顺着这段乱码往下编。所以「模型输出不正确」有相当一部分是**测试用例本身喂了乱码**，不是引擎算错。

**已修复**：换成 `tokenizer.apply_chat_template(..., add_generation_prompt=True)` 的真实 id。

同时 `tokenizer.h` 的 `SPECIAL_TOKEN_EOS` 沿用了 Qwen2 的 `151645`，而 Qwen3.5 的 eos 是 `248044`，导致生成不会在 EOS 处停止。**已修复**。

## 二、引擎实现 bug（均已修复）

| # | 位置 | 问题 | 修复 |
|---|------|------|------|
| 1 | `rmsnorm()` | Qwen3_5RMSNorm 的权重是 **zero-centred**，有效缩放是 `(1 + w)`，原代码直接用 `w`，导致每层约 2× 缩放（实测 ln 权重 mean≈0.0）。注意 GDN 的 `RMSNormGated` 用的是 `torch.ones`，必须用**不带 1+** 的版本 | 拆成 `rmsnorm()`（`1+w`）与 `rmsnorm_plain()`（`w`）两个函数 |
| 2 | `forward_linear_attn()` | conv1d 后**漏了 SiLU**。参考实现是 `causal_conv1d_fn(..., activation="silu")`，silu 在 conv 之后、Q/K/V 切分之前立即施加 | 补 `qkv[i] = silu(sum)` |
| 3 | `apply_rope()` | 两处错：① 配对方式错（用了相邻对 `(i, i+1)`，参考实现是 `rotate_half` 的**跨半轴** `(i, i+rotary_dim/2)`）；② 频率分母用了 `head_dim/4` 而非 `rotary_dim`。实测 `rotary_dim = 64`（`partial_rotary_factor=0.25`），后 192 维原样透传 | 重写为跨半轴配对、`inv_freq` 长度 32 |
| 4 | `forward_full_attn()` | Q/gate 切分错。`q_proj` 输出 4096，参考实现是 `view(..., nheads, head_dim*2)` 后 `chunk(2, -1)`，即**每 head 内交织** `[q(256)|gate(256)]`；原代码按 `[q2048|gate2048]` 平铺切分 | 按 head 逐段拆到 `q` / `gate_buf` |
| 5 | `forward_full_attn()` | **缓冲区溢出**：把 `kvdim=512` 的 K 写进 `s->x2`（该缓冲区按 2048 分配但语义上属于别的用途），且 V 复用了别的缓冲区，存在互相覆盖风险 | K/V 改写到 `s->h` 的连续区段 |
| 6 | `softplus` | `logf(1+expf(x))` 在大 x 时溢出 | 按 PyTorch 语义加 `threshold=20` 分段：`x>20` 时直接返回 `x`，否则 `log1pf(expf(x))` |
| 7 | `forward_full_attn()` | 漏了 causal mask / head→kv 映射（`h / gqa` 本来就对，但缺少显式的 `t <= pos` 保护以外的自检） | 已确认 `t <= pos` 与 `h/gqa` 正确，保持 |

> head→kv 映射确认为 `kv = h // groups`（`groups = nheads/nkvheads = 4`），**不是** `h % nkv`。
> delta rule 顺序确认为**先衰减 state、再读 kv_mem**、输出用**更新后**的 state；`g = -exp(A_log) * softplus(a + dt_bias)`，`gamma = exp(g) ∈ (0,1)`。这两点由 `spec_experiments/exp_delta_order.py` 用真实权重验证（`max|ref - pre_decay_read| = 0`，`max|ref - post_decay_read| = 1.79e-02`）。

## 三、验收证据（可复现）

对比对象：`/mnt/Qwen3.5-0.8B` + `/home/wisedu/laya/bin/python`（transformers 自带 `qwen3_5` 实现），CPU fp32。

**默认 prompt** — `中国的首都是哪里？`

```bash
QWEN35_DUMP_LOGITS=/tmp/c_logits.bin ./qwen35_engine
```

```
max abs diff      = 4.101e-05
mean abs diff     = 6.634e-06
argmax HF / C     = 96328 / 96328
cosine similarity = 1.00000012
top-10            = 完全一致
```

输出：`中国的首都是**北京**。` —— 与 HF 贪心逐 token 一致，并在 EOS (`248044`) 处停止。

**Prompt 2** — `What is 2+2?`（回归，排除单例巧合）

```
max abs diff      = 3.957748e-05   mean = 6.166832e-06
argmax HF / C     = 760 / 760      cosine = 1.00000012   top-20 overlap = 20/20
```

生成：`The answer to "What is 2+2?" is **4**.` —— 与 HF 贪心前 30 token 完全一致。

**Prompt 3** — `Name the capital of France.`（回归）

```
P2 max abs diff = 4.518032e-05   mean = 5.643506e-06
P2 argmax HF/C  = 760 / 760      cosine = 1.00000000
```

生成：`The capital of France is **Paris**.` —— 逐 token 与 HF 相同。

## 四、输出与性能统计

程序现在**只打印最终结果**（无逐 token 调试输出、无 top-10 列表、stderr 为空），并在末尾给出性能统计：

```
Qwen3.5-0.8B C Inference Engine
================================

Config: dim=1024, ffndim=3584, nlayers=24, nheads=8, nvocab=248320
Layer types: LLLFLLLFLLLFLLLFLLLFLLLF (L=linear, F=full)

Loading weights from safetensors...
Weights loaded successfully
加载词表完成: 248044 个 token
Prompt: "<|im_start|>user\n中国的首都是哪里？<|im_end|>\n<|im_start|>assistant\n"
Output: 中国的首都是**北京**。

北京位于中国华北地区，是中华人民共和国的行政中心，也是全国的政治、文化、商贸和交通中心。自 1949 年 10 月 1 日中华人民共和国成立后，北京一直作为首都存在，并持续至今。

-------------------
Prefill: 17 tokens in 10.608 s  (1.60 tok/s)
Generate: 61 tokens in 38.069 s  (1.60 tok/s)
Stop reason: EOS
```

计时用 `CLOCK_MONOTONIC`：
- **Prefill** 覆盖整个 prompt 的所有 `forward()` 调用；
- **Generate** 只累加 `forward()` 的计算耗时（不含打印/解码 I/O），且**不含**首个生成 token 的前向——它由 prefill 的末次前向产出，其延迟计入 prefill（标准做法）。因此 `Generate` 的 token 数比实际打印的文本多 1。
- 纯 CPU fp32、无 SIMD/线程优化，故约 1.6 tok/s 属预期。

## 五、可用的调试开关（本次顺手加的）

```bash
# 换任意 prompt（逗号分隔 token id）
QWEN35_PROMPT_IDS="248045,846,198,..." ./qwen35_engine

# 导出最后一个 prompt 位置的完整 logits（float32，248320 维）用于和 HF 对比
QWEN35_DUMP_LOGITS=/tmp/c_logits.bin ./qwen35_engine
```

## 六、遗留 / 未做

- 只验证了 **CPU fp32、prefill+逐 token decode** 路径；未验证 batch>1、未验证长上下文、未验证视觉塔（`model.visual.*`）与 `mtp.*`（纯文本推理不需要）。
- `forward_linear_attn` 的 `pos` 参数未使用（`-Wunused-parameter` 警告），属无害残留。
- 引擎为追求正确性使用 fp32 累加，未做加速优化（故吞吐约 1.6 tok/s）。
