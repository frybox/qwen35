# 源码学习导读（CODE_INDEX）

按源码顺序列出各文件的顶层定义与关键实现，供重抄/精读时对照。行号对应当前版本
（`qwen35_engine.c` 931 行 / `safetensors.c` 425 行 / `tokenizer.c` 1136 行）。

## 一、qwen35_engine.c

| 行号 | 名称 | 说明 |
| --- | --- | --- |
| 18–22 | `now_sec` | `clock_gettime` 墙钟计时，用于吞吐统计 |
| 31–44 | `Config` | 超参结构；`layer_types[24]` 在 L43 生成 |
| 50–81 | `Weights` | 全部权重指针（每个 `[24]`） |
| 87–101 | `State` | 激活 / KV cache / GDN 状态缓冲 |
| 103–108 | `Transformer` | Config + Weights + State + safetensors 句柄 |
| 114–127 | `die` / `check_ptr` / `xmalloc` | 错误处理与分配包装 |
| 133–151 | `init_config` | 写死 0.8B 超参；L149 `(i%4==3)?1:0` 得到 `LLLF` 层型 |
| 158–183 | `load_tensor_f32` | 按名取张量并转 FP32 |
| 185–287 | `load_weights` | 全部张量名映射：L195 embed、L205 final norm、L214 起 24 层循环（L224 GDN 子张量、L253 全注意力子张量、L276 MLP） |
| 293–321 | `malloc_state` | 运行时缓冲分配 |
| 331–336 | `rmsnorm` | `(1+w) * rsqrt(mean(x²)+eps)`（Qwen3.5 的 zero-init 式） |
| 339–342 | `softplus_f` | 阈值 20，对齐 PyTorch |
| 345–350 | `rmsnorm_plain` | `w * rsqrt(...)`（GDN gated norm 用），注意与上面参数化不同 |
| 352–358 | `softmax` | 数值稳定版（当前 main 未用） |
| 360–367 | `matmul` | `o[d] = W[d,n]·x[n]`，朴素行主序，无 BLAS |
| 369–371 | `silu` | `x·σ(x)` |
| 377–411 | `apply_rope` | 部分 RoPE：`rotary_dim=64`，配对为 `i ↔ i+32`（rotate_half/NeoX 排布，非相邻），`[64,256)` 直通 |
| 417–506 | `forward_full_attn` | 全注意力层，见下 |
| 512–659 | `forward_linear_attn` | GDN 层，见下（`pos` 未使用，已标 `(void)pos`） |
| 665–685 | `forward_mlp` | SwiGLU：`down(silu(gate·x) * up·x)` |
| 691–718 | `forward` | 主干：embedding 查表 → 24 层按 type 分派 → MLP → final norm → shared-embedding logits（L715） |
| 724–734 | `sample_argmax` | 贪心 |
| 737–754 | `sample_temp` | 温度采样（当前未用） |
| 760–932 | `main` | prefill / decode / 计时 |

**forward_full_attn（417–506）**
LN → `q_proj`(4096) 按 head **交错切出 [Q, gate]**（L435–445，不是 2048 处平切）→ per-head
q_norm → k_proj → k_norm（RoPE 之前）→ `apply_rope` → 写 KV cache
（布局 `[nlayers][nctx][kvdim]`，偏移 L466）→ GQA 4:1 打分（L474）→ sigmoid gate（L496–498）
→ `o_proj` → 残差。

**forward_linear_attn（512–659）**
LN → `in_proj_qkv`(6144)，Q/K/V 为**三段顺序**排布（L581–589）→ depthwise causal conv（k=4）
+ silu，conv 状态左移（L537）→ `z/a/b` 投影，b → sigmoid、a 原值（L570–572）→
per-head **L2 归一化 q/k**（L594–598）→ `g = -exp(A_log)·softplus(a+dt)`、`gamma=exp(g)`（L609–611）
→ 状态衰减 → `kv_mem = state·k`（L619–624）→ `delta = (v - kv_mem)·beta`（L628）→
外积更新 state（L634–636）→ `out = state·q`（L641–645）→ gated RMSNorm（共享 [128] 权重）× `silu(z)`（L648–651）
→ `out_proj` → 残差。状态**跨 token 保持**，所以不需要位置参数。

**main（760–932）**：默认 prompt 硬编码为 token id 数组（L~835），可用 `QWEN35_PROMPT_IDS` 覆盖；
L852–859 prefill 计时；L866–877 可用 `QWEN35_DUMP_LOGITS` 导出 248320 个 fp32 logits；
L886–915 贪心 decode（EOS 不打印，计时只包 `forward`）；L919–925 输出 tok/s。

## 二、safetensors.c / .h

| 行号 | 名称 | 说明 |
| --- | --- | --- |
| 8–18 | `parse_dtype` | dtype 字符串→枚举 |
| 20–32 | `dtype_size` | 每元素字节数 |
| 34–73 | `json_find_key` / `json_parse_string` / `json_parse_int` | 极简手写 JSON 取键 |
| 75–135 | `parse_tensor_meta` | 解析单个张量 name/dtype/shape/data_offsets（L130 累加 data_offset） |
| 137–219 | `safetensors_open` | 读 8 字节头长度 + JSON 元数据（L166 `data_start = 8 + meta_len`），建张量表 |
| 221–234 | `safetensors_close` | 释放 |
| 236–246 | `safetensors_get_tensor` | 线性查找（无哈希） |
| 248–258 | `safetensors_load_tensor` | `fseek` + `fread` 拷原始字节 |
| 260–267 | `bf16_to_f32` | 左移 16 位（截位，非四舍五入） |
| 269–303 | `f16_to_f32` | 手写 IEEE half→float |
| 305–337 | `safetensors_load_tensor_as_f32` | 按 dtype 分派 F32/BF16/F16 |
| 339–371 | `safetensors_print_info` | 打印张量清单 |
| 373–419 | `safetensors_test` + `main` | 自测入口（`-DSAFETENSORS_MAIN` 时启用） |

解决的问题：解析 `header_len + JSON` 容器，按名定位偏移并把 BF16/F16 读成 FP32，
上层不必关心 dtype。**实现是 stdio 随机读，没有 mmap，也没有多分片索引**——分片路径由 `main` 直接传单文件。
文件读取量小（一次一个张量），所以速度不是瓶颈。

## 三、tokenizer.c / .h

| 行号 | 名称 | 说明 |
| --- | --- | --- |
| 26–77 | `hash_string` + `hashmap_*` | 自研链地址哈希表 |
| 93 | `gpt2_byte_to_unicode[256]` | GPT-2 byte↔unicode 表 |
| 126–192 | `unicode_to_utf8` / `utf8_char_len` / `utf8_to_unicode` / `init_byte_encoder` | UTF-8 与字节编码互转 |
| 194–236 | `parse_json_string` | 带转义与 `\uXXXX` 的 JSON 串解析 |
| 238–291 | `tokenizer_init` / `tokenizer_free` | 生命周期 |
| 293–364 | `tokenizer_load_vocab` | 读 `vocab.json`，建 id↔string 双向表 |
| 366–428 | `tokenizer_load_merges` | 读 `merges.txt` 建 rank 表（L416–420） |
| 430–497 | `tokenizer_load_special_tokens` / `is_special_token` | 特殊 token 表与最长匹配 |
| 499–585 | `bpe_encode_bytes` | **BPE 核心**：循环取最小 rank 合并（L517–528） |
| 587–693 | `nfc_lookup_*` / `is_combining_mark` / `nfc_normalize` | 基于 `nfc_table.h` 的 NFC 归一化 |
| 695–910 | `classify_char` / `scan_space_run` / `next_pretoken` | GPT-2 预分词切分（Qwen 的正则） |
| 912–943 | `tokenizer_encode` | 文本 → ids |
| 945–995 | `tokenizer_decode` | ids → 原始字节 → UTF-8（动态缓冲，无 256 上限） |
| 998–1011 | `tokenizer_get_token` / `tokenizer_get_id` | 查表 |
| 1013–1136 | `test_tokenizer` + `main` | 自测入口（`-DTOKENIZER_MAIN` 时启用） |

解决的问题：纯 C 复刻 Qwen3.5 的 byte-level BPE（GPT-2 byte 编码 + NFC 归一化 + Qwen 预分词正则），
并实现 decode 回 UTF-8。引擎推理路径只用 `init` / `load_vocab` / `decode`。

## 四、抄写时最容易错的点

- **MoE 之外的层分派**：`layer_types[i] = (i % 4 == 3)`，24 层即 `LLLF` 重复 6 次（:149）。
- **张量名前缀是 `model.language_model.`**，不是 `model.`；`model.visual.*` 与 `mtp.*` 要忽略（:195 起）。
- **两种 RMSNorm 参数化**：`(1+w)` vs `w`，混用就全错（:331–335 vs :345–349）。
- **RoPE**：只转前 64 维，且是 `i ↔ i+32` 的 rotate_half 排布，不是相邻配对（:383–384, :398–400）。
- **q_proj 输出是 per-head 交错的 [Q, gate]**（:435–445），最易抄错的一处。
- **KV cache 偏移**：`layer*nctx*kvdim + pos*kvdim`，每层独占一段（:466）。
- **GDN 的 conv 状态**：按 `gdn_idx*6144*4` 索引，只对线性层递增（:537）。
- **GDN 的 decay**：`g = -exp(A_log)·softplus(a + dt_bias)`，`gamma = exp(g)`（:609–611）。
- **final norm 用 `model.language_model.norm.weight`**，缺失时退化为全 1 占位（:203–211）。
- **prefill 与 decode 共用同一个 `pos` 递增的 `forward`**，没有 batched prefill（:852 起）。

## 五、已知缺口

- `tokenizer_load_merges` 已实现但 `main` **没有调用**，所以 `tokenizer_encode` 目前不能在推理路径上用；
  prompt 是硬编码 id（可用 `QWEN35_PROMPT_IDS` 覆盖）。要支持任意文本输入，需在 main 里加载 `merges.txt`。
