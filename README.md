# qwen35 — GGUF 解析里程碑

从零用 Zig 实现 Qwen3.5 推理引擎，不依赖任何框架（不调 llama.cpp / gguf-py / PyTorch）。
本里程碑：**解析 GGUF 文件，打印全部 metadata 与 tensor 清单**。

## 状态

| 项 | 状态 |
|---|---|
| GGUF v2/v3 header + KV metadata + tensor info | 完成 |
| mmap 读取（不整体载入内存） | 完成 |
| 未知 `ggml_type` 硬报错 | 完成 |
| 布局自检 `data_offset + Σalign_up(bytes) == file_size` | 完成 |
| 与 gguf-py 逐项对拍（0.8B-Q8_0 / 4B-IQ4_NL） | 通过，零差异 |
| 单元测试 | 8/8 |

## 构建与运行

```bash
zig build
./zig-out/bin/qwen35 /mnt/Qwen3.5-0.8B-GGUF/Qwen3.5-0.8B-Q8_0.gguf          # 长字符串截断
./zig-out/bin/qwen35 /mnt/Qwen3.5-0.8B-GGUF/Qwen3.5-0.8B-Q8_0.gguf --full   # 全部展开
```

输出结构：

```
file <path>
file_size / version / tensor_count / metadata_kv_count / alignment
tensor_info_end / data_offset / sum_aligned_bytes
check data_offset+sum == file_size : OK

=== metadata (N) ===
  0 general.architecture = "qwen35"
  ...
 12 tokenizer.ggml.tokens = ARRAY<STRING> len=248320 "..." "..." ... (248304 more)

=== tensors (N) ===
name                                         type     dims                 offset   abs_offset        bytes
blk.0.attn_q.weight                          Q8_0     [2048,1024]         ...
```

`offset` 是文件里存的**原始字段**（相对数据区起点）；`abs_offset = data_offset + offset`
是文件内绝对位置。gguf-py 的 `tensor.data_offset` 打印的是后者，两者都对，只是口径不同。

## 一键验收

```bash
tests/run_acceptance.sh /mnt/Qwen3.5-0.8B-GGUF/Qwen3.5-0.8B-Q8_0.gguf
```

它会：① `zig build test`；② 用 gguf-py 生成参考 dump；③ 跑 zig 打印；
④ 逐项比对 header / metadata / 每个 tensor 的 名字·类型·维度·offset·字节数；
⑤ 对 tokens / merges / token_type 三个大数组做逐元素比对。

## 实测结果（Qwen3.5-0.8B-Q8_0.gguf）

```
version=3  tensor_count=320  metadata_kv_count=46  alignment=32  data_offset=10961920
check data_offset+sum == file_size : OK        (sum=800881920, file=811843840)
类型分布: F32 133, Q8_0 187                    ← 混合精度，逐张量按类型算字节数
vocab_size=248320  n_merges=247587
token_type 直方图: {1: 248044, 3: 27, 4: 6, 5: 243}
```

架构关键 KV（`general.architecture = "qwen35"`）：

```
qwen35.block_count                   24      qwen35.embedding_length        1024
qwen35.feed_forward_length           3584    qwen35.context_length          262144
qwen35.attention.head_count          8       qwen35.attention.head_count_kv 2
qwen35.attention.key_length          256     qwen35.attention.value_length  256
qwen35.rope.freq_base                1e7     qwen35.attention.layer_norm_rms_epsilon 1e-6
qwen35.ssm.conv_kernel               4       qwen35.ssm.state_size          128
qwen35.ssm.group_count               16      qwen35.ssm.time_step_rank      16
qwen35.ssm.inner_size                2048
tokenizer.ggml.model "gpt2"   tokenizer.ggml.pre "qwen35"
tokenizer.ggml.eos_token_id 248046   tokenizer.ggml.padding_token_id 248055
```

> 注意：词表是 248320，无 `bos_token_id` 键（不要沿用 Qwen3-0.6B 的 151643/151645）。

张量结构印证了 Qwen3.5 的**混合层型** `LLLF` 循环 ×6 = 24 层：

- **18 层 GDN 线性注意力**：`ssm_a`(F32[16]) `ssm_alpha/ssm_beta`(Q8_0[1024,16])
  `ssm_dt.bias`(F32[16]) `ssm_conv1d.weight`(F32[4,6144]) `ssm_norm.weight`(F32[128])
  `ssm_out.weight`(Q8_0[2048,1024]) `attn_qkv.weight`(Q8_0[1024,6144]) `attn_gate.weight`(Q8_0[1024,2048])
- **6 层全注意力**：`attn_q/k/v`、`attn_q_norm`/`attn_k_norm`、`attn_output`
- 所有层都有：`attn_norm`、`post_attention_norm`、`ffn_gate`/`ffn_up`/`ffn_down`
- 顶层：`token_embd.weight` [1024,248320]、`output_norm.weight`


## 跨文件鲁棒性

`/mnt/Qwen3.5-4B-GGUF/` 与 `/mnt/Qwen3.5-0.8B-GGUF/` 下 21 个文件（含 mmproj 视觉侧文件）
全部解析成功、恒等式全部成立，覆盖的量化类型包括：

```
F32 F16 BF16 IQ2_XXS IQ2_M IQ3_XXS IQ4_NL IQ4_XS
Q3_K_S Q3_K_M Q4_0 Q4_1 Q4_K_S Q4_K_M Q5_K_S Q5_K_M Q6_K Q8_0
以及 UD 动态量化（UD-IQ2_M / UD-Q2_K_XL / UD-Q3_K_XL …）
```

其中 4B-IQ4_NL 与 gguf-py 逐项对拍亦为零差异（426 张量，5 种类型混用）：
`F32 177, IQ4_NL 168, Q5_K 32, Q8_0 48, Q6_K 1`。

## 错误处理

CLI 对以下情况给出可读信息并以 `rc=1` 退出，不 panic、不打印 Zig 回溯：

| 输入 | 输出 |
|---|---|
| 相对路径 | 正常工作（`openFileAbsolute` 遇相对路径会 assert，故用 `cwd().openFile`） |
| 空文件 | `error: '<path>' is empty` |
| 非 GGUF 文件 | 报 `BadMagic` 并打印前 4 字节与期望值 |
| 截断的 GGUF | 报 `UnexpectedEof` |
| 头部合法但数据区对不上 | 报 `SizeMismatch`（提示"tensor 尺寸铺不满数据区"） |
| 未知 ggml_type | 硬报错，拒绝猜测宽度 |
| 文件不存在 | `error: cannot open '<path>': FileNotFound` |

## 目录

```
src/gguf.zig     解析器（无外部依赖）+ `File.print` / `printFull` 转储
src/main.zig     CLI：mmap + 参数解析，转储交给 `File.print`
src/root.zig     模块根
tests/gen_ref.py           gguf-py 参考 dump（仅作对拍，引擎本体不依赖）
tests/compare.py           逐项比对器
tests/verify_big_arrays.py 三大数组逐元素核对
tests/run_acceptance.sh    一键验收
ref/                       C 参考实现与文档
```

## 踩过的坑

1. **tensor 的 `dims` 是 u64**，而 `n_dims` 是 u32。按 u32 读会让 tensor info 区整体错位，
   报出莫名其妙的 `UnknownValueType`。此坑已由单元测试钉死。
2. **未知 `ggml_type` 必须报错**，不能按默认宽度跳过——否则后续解析全部错位，且是静默错误。
3. **`dims` 最内维在前**，与 PyTorch 相反。
4. **混合精度**：同一个文件里 norm 是 F32、权重是 Q8_0，必须逐张量按 `tensor_type` 算字节数。
5. **`offset` 语义**：规范里是相对数据区，gguf-py 暴露的是绝对值。
