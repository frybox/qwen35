# Qwen3.5-0.8B C 推理引擎修复 SPEC

**唯一依据**：`/home/wisedu/laya/lib/python3.12/site-packages/transformers/models/qwen3_5/modeling_qwen3_5.py`
（下文所有 `MN:行号` 均指该文件；实测数据来自 `spec_experiments/*.py` 的输出 `*.out`）。

**模型**：`/mnt/Qwen3.5-0.8B/`，单文件权重 `model.safetensors-00001-of-00001.safetensors`（320 个张量）。
张量名前缀 **`model.language_model.`**（不是 `model.`），视觉部分为 `model.visual.*`，需忽略 `mtp.*`（MN:1670, 925）。

## 0. 关键 config 实测值（不是默认值，是 /mnt/Qwen3.5-0.8B/config.json 里真实写的）

| 字段 | 值 |
|---|---|
| `hidden_size` | 1024 |
| `num_hidden_layers` | 24 |
| `layer_types` | `['linear_attention','linear_attention','linear_attention','full_attention', ...]` 见 Q15 |
| `full_attention_interval` | 4 |
| `num_attention_heads` / `num_key_value_heads` / `head_dim` | 8 / 2 / 256 |
| `linear_num_key_heads` / `linear_num_value_heads` | 16 / 16 |
| `linear_key_head_dim` / `linear_value_head_dim` | 128 / 128 |
| `linear_conv_kernel_dim` | 4 |
| `rms_norm_eps` | 1e-6 |
| `intermediate_size` | 3584 |
| `hidden_act` | `silu` |
| `attention_bias` | false（所有权重无 bias） |
| `attention_dropout` | 0.0 |
| `max_position_embeddings` | 262144 |
| `vocab_size` | 248320 |
| `tie_word_embeddings` | **true** |
| `rope_parameters` | 见 Q9 |

派生量：`key_dim = 128*16 = 2048`，`value_dim = 128*16 = 2048`，`conv_dim = 2*2048+2048 = 6144`（MN:512-521）。

---

# A. GDN / linear_attention 层

## A1. `in_proj_qkv` 输出 6144 维的分段顺序 = **[Q | K | V] 连续三段，不是交织**

MN:603-611 原文：

```python
mixed_qkv = mixed_qkv.transpose(1, 2)
query, key, value = torch.split(
    mixed_qkv,
    [
        self.key_dim,        # 2048
        self.key_dim,        # 2048
        self.value_dim,      # 2048
    ],
    dim=-1,
)
```

- 顺序就是 **`[0:2048]=Q`, `[2048:4096]=K`, `[4096:6144]=V`**，连续块切分，无交织。
- `torch.split` 的 sizes 列表顺序 == `[key_dim, key_dim, value_dim]` == `[head_k_dim*num_k_heads, head_k_dim*num_k_heads, head_v_dim*num_v_heads]`（`key_dim = head_k_dim * num_k_heads`，MN:512-513）。
- 随后 `query.reshape(b, s, -1, head_k_dim)` → `[b,s,16,128]`（MN:613-615）。
- 注意 `torch.split` 的边界只由 `key_dim` 决定，**`key_dim/2` 与切分无关**；8192 之类的拆分是错的。

## A2. `causal_conv1d`：权重语义 / 无 bias / 左补 K-1 / activation="silu" / silu 在 conv 之后立即施加

MN:522-529（构造）：

```python
self.conv1d = nn.Conv1d(
    in_channels=self.conv_dim,
    out_channels=self.conv_dim,
    bias=False,
    kernel_size=self.conv_kernel_size,
    groups=self.conv_dim,
    padding=self.conv_kernel_size - 1,
)
```

MN:270-289（真实前向实现 `causal_conv1d_fn`）：

```python
_, hidden_size, seq_len = hidden_states.shape
padding = weight.shape[-1] - 1
out = F.conv1d(
    hidden_states.to(weight.dtype),
    weight=weight.unsqueeze(1),
    bias=bias,
    padding=padding,
    groups=hidden_size,
)[:, :, :seq_len]
if activation is not None:
    out = ACT2FN[activation](out)
```

| 项 | 结论 |
|---|---|
| 权重形状 | 检查点里 `conv1d.weight` = **(6144, 1, 4)**；即 per-channel `[C, 1, K]`。算子里调用时 `self.conv1d.weight.squeeze(1)` 得到 `[6144, 4]`，再 `unsqueeze(1)` 回 `[6144,1,4]`（MN:280-286） |
| bias | **无**（`bias=False`；`self.conv1d.bias` 为 `None`，MN:581/593 传入的就是 None） |
| 分组 | depthwise：`groups = conv_dim = 6144`，每个通道独立 K=4 卷积 |
| padding | `padding = K-1 = 3`，`F.conv1d` 的 padding 是**左右都补 3**，然后取 `[:seq_len]` ⇒ **有效等价于左侧补 3 个零（causal，左补 K-1）**。实测：`max|F.pad(x,(3,0)) 再 padding=0 - padding=3 截断| = 1.19e-7`（`exp_decay.out`） |
| activation | `self.activation = config.hidden_act` = **"silu"**（MN:517），作为 `activation=` 传入（MN:594） |
| silu 时机 | **conv 之后立即施加**（MN:287-288），**在 transpose 和 Q/K/V 切分之前** |
| 输入布局 | conv 作用在 `mixed_qkv.transpose(1,2)` 上 → `[b, 6144, seq]`；卷积核为**按 token 的一维 causal 卷积** |

⚠️ 单 token decode 用 `causal_conv1d_update`（MN:250-266），需要维护 `conv_state`（长度 `state_len = K-1 = 3` 的历史 `hidden_states`），把 `cat([conv_state, x])` 后跑 `padding=0` 的 conv 再取最后 `seq_len` 列，并 `conv_state.copy_(...[:, :, -state_len:])`。silu 同样在 conv 之后。

## A3. q/k 的 L2 normalize

MN:292-297 原文：

```python
# NOTE: the FLA package computes `x / torch.sqrt((x * x).sum(dim=dim, keepdim=True) + eps)` instead, so if we align
# with the GatedRMSNorm, maybe we can make that change as well.
def l2norm(x: torch.FloatTensor, dim: int = -1, eps: float = 1e-6):
    """This function is intended to align with the l2norm implementation in the FLA library."""
    inv_norm = torch.rsqrt((x * x).sum(dim=dim, keepdim=True) + eps)
    return x * inv_norm
```

- eps = **1e-6**，写死在调用处：`l2norm(query, dim=-1, eps=1e-6)`（MN:344-345 chunk 路径；MN:464-465 recurrent 路径）。
- 写法是 **`x * rsqrt(sum(x^2) + eps)`**，即 `x * rsqrt(clamp/sum + 1e-6)`；**不是** `x / sqrt(sum+eps)` 之外的任何形式。注意 eps 是**加在平方和内部**的（不是加在 sqrt 之后）。
- 归一化在 fp32 里做：`query, key, value, beta, decay = [x.transpose(1,2).to(torch.float32, ...)]`（MN:458-461 / 338-341），l2norm 在 upcast 之后。
- **之后**再乘 `1/sqrt(head_k_dim)`：

  MN:467-468（recurrent） / MN:346-348（chunk）原文：
  ```python
  # And always normalize queries by the head dimension
  scaling = query.shape[-1] ** -0.5
  query = query * scaling
  ```
  head_k_dim=128 → `query *= 1/sqrt(128) ≈ 0.08839`。**只有 q 被缩放，k 不缩放。**
- 缩放施加在 q 上（不是 attention score 上），且是在 l2norm 之后。

## A4. delta rule 的精确运算顺序 ★

### A4.1 权威原文（MN:479-491，真实执行路径）

```python
    # Loop over each token and update the recurrent state
    for i in range(sequence_length):
        q_t, k_t, v_t = query[:, :, i], key[:, :, i], value[:, :, i]
        # Decay the recurrent state
        decay_t = decay[:, :, i].exp()[..., None, None]
        last_recurrent_state = last_recurrent_state * decay_t
        # Update the recurrent state with the current token
        beta_t = beta[:, :, i].unsqueeze(-1)
        kv_mem = (last_recurrent_state * k_t.unsqueeze(-1)).sum(dim=-2)
        delta = (v_t - kv_mem) * beta_t
        last_recurrent_state = last_recurrent_state + k_t.unsqueeze(-1) * delta.unsqueeze(-2)
        # And use it to compute the attention output for the current token
        core_attn_out[:, :, i] = (last_recurrent_state * q_t.unsqueeze(-1)).sum(dim=-2)
```

### A4.2 结论：顺序是 **(ii)「先衰减 state，再用衰减后的 state 算 kv_mem」**

精确逐 token 伪码（C 引擎照抄）：

```
S  *= gamma                           # (1) 先衰减
kv_mem  = k^T · S                     # (2) 用【已衰减】的 S 读 (k 是 [k_dim], S 是 [k_dim, v_dim])
delta   = beta * (v - kv_mem)          #     delta/v/kv_mem 都是 [v_dim]
S      += outer(k, delta)              # (3) = k[k_dim,1] * delta[1,v_dim]
o       = q^T · S                      # (4) 用【本 token 更新后】的 S 输出
```
即 **(ii)**，不是 (i)。用户给的两种候选里，正确答案是「`S = gamma*S` 后 `delta = beta*(v - (S_decayed·k))`」。
等价地：`delta = beta*(v - (gamma*S_old)·k)`，`S_new = gamma*S_old + k⊗delta`。

**数值实验证明**（`spec_experiments/exp_delta_order.py`，输出见 `exp_delta_order.out`）：

```
[3] cross-check against the REAL torch_recurrent_gated_delta_rule (modeling_qwen3_5.py:438)
    max |ref - pre_decay_read | = 0.000000e+00     <-- 逐位相同
    max |ref - post_decay_read| = 1.789570e-02     <-- 不等价
    max |ref_S - pre_S  |       = 0.000000e+00
    max |ref_S - post_S |       = 4.273649e-02
    ==> REFERENCE MATCHES: pre_decay_read (decay FIRST, then read kv_mem)
```
两种顺序的差异量级：**初始 state 为 0 时 max|Δo| = 2.63e-2（相对 3.74%）**；**有初始 state（decode 路径）时 max|Δo| = 7.61e-2**；两者绝不相等。
用真实 checkpoint 的 layer 0 端到端复核（`exp_decay.py`）：

```
max|manual(correct order) - real module| = 5.960e-08   (fp32 精度内完全一致)
max|manual(WRONG  order) - real module| = 5.695e-02
max|manual(no l2norm)    - real module| = 3.029e-01
```

### A4.3 `g` 的形状与含义

- `g` 形状 = **`[batch, seq, num_v_heads]`**（`= [1, T, 16]`），**每个 (token, v-head) 一个标量**，不是每 k-channel 一个。
- 文档串 MN:319-320 原文：
  ```python
  g: Decay (in log space) tensor of shape [batch_size, sequence_length, num_v_heads]: the recurrent state is
      multiplied by exp(g) at each step, so entries must be <= 0.
  ```
- 标量 `gamma = exp(g)` 会**广播**到整个 `[k_dim, v_dim]` state 上（`.exp()[..., None, None]`，MN:483）→ 整个状态矩阵乘同一个数。

### A4.4 `g = exp(g)` 还是别的？

- **`gamma = exp(g)`**，且 `g = -A_log.exp() * softplus(a + dt_bias)`（MN:619）：
  ```python
  beta = b.sigmoid()
  # If the model is loaded in fp16, without the .float() here, A might be -inf
  g = -self.A_log.float().exp() * F.softplus(a.float() + self.dt_bias)
  ```
- `beta = sigmoid(b)`（MN:617），也是 per-v-head 标量，形状 `[b,T,16]`。
- softplus 细节：用的是 **`F.softplus(x)`，torch 默认 `beta=1.0, threshold=20.0`**，即 `x>20` 时直接返回 `x`，否则 `log1p(exp(x))`。实测（`exp_decay.out`）：真实 `a + dt_bias` 的范围 `[-13.2146, 8.4654]`，**最大值 8.47 < 20，threshold 分支永不触发**；`max|F.softplus - log1p(exp(x))| = 3.73e-09`（fp32 舍入内相等，非逐位相同）。所以用 `log1p(exp(x))` 安全，但为完全对齐建议照抄 `threshold=20` 的分段写法。
- 计算在 **fp32** 里做（`.float()`，MN:619）。

### A4.5 state 初始值 / 维数布局

- 初始值：`None` 时 `torch.zeros((batch_size, num_v_heads, k_head_dim, v_head_dim))`（MN:472-474）→ 布局是 **`[num_v_heads, k_dim, v_dim]`**，即 `S[k, v]`，**不是转置**。
- `kv_mem = (S * k[..., None]).sum(dim=-2)`：`k` 在 **-2（k_dim）** 轴上广播、沿 k_dim 求和 → `kv_mem[v_dim]`。这正是 `S^T k`（S 为 `[k,v]`）。
- 写回 `S += k[...,None] * delta[...,None,:]` 是 `[k_dim,1] ⊗ [1,v_dim]`。**两个轴写反会完全错。**
- 输出 `o = (S * q[..., None]).sum(dim=-2)` → `o[v_dim]`，即 `S^T q`。

### A4.6 输出用的是更新后还是更新前的 state

**更新后**。MN:491 在 MN:489 的写回**之后**。实验（`exp_delta_order.out`）：

```
[5] output read with PRE-update vs POST-update state: max|diff| = 2.571420e-01
    => output MUST use the state AFTER the current token's update.
```

### A4.7 chunk 路径（prefill 用）与 recurrent 路径等价

MN:363-434 的 `torch_chunk_gated_delta_rule` 是同一语义的并行化实现（UT transform + chunk 内 pairwise decay），**默认 `chunk_size=64`**（MN:307）。其中关键一致性点：
- MN:414 `key = key * (cum_decay[..., -1:] - cum_decay).exp()`：chunk 内 k 用「到 chunk 末」的衰减；
- MN:421-423 `v_new = new_values - k_cumdecay @ S`、`core_attn_out = query @ S + intra_chunk_attn @ v_new`；
- MN:425 `S = S * chunk_decay + key^T @ v_new`（chunk 末整体衰减）。
实测 chunk_size=4 与 recurrent 的差 `2.98e-08`（纯 fp32 误差）。**C 引擎做 prefill 时可以只用 recurrent 循环**（等价且更简单）。

## A5. `num_v_heads // num_k_heads > 1` 与 repeat_interleave

MN:508-509, 620-622 原文：

```python
self.num_v_heads = config.linear_num_value_heads
self.num_k_heads = config.linear_num_key_heads
...
if self.num_v_heads // self.num_k_heads > 1:
    query = query.repeat_interleave(self.num_v_heads // self.num_k_heads, dim=2)
    key = key.repeat_interleave(self.num_v_heads // self.num_k_heads, dim=2)
```

- **本例：`linear_num_key_heads = 16`, `linear_num_value_heads = 16` → 商 = 1，条件为 False，`repeat_interleave` 完全不执行**（`exp_decay.out`）。
- 作用（若触发）：把 Q/K 的 head 轴从 `num_k_heads` 展开到 `num_v_heads`，让每个 k-head 服务 `v/k` 个 v-head；`repeat_interleave` 的映射是 **v_head = k_head（整除）**，即第 `h` 个 v-head 用第 `h // ratio` 个 k-head（与 B13 的 `h // groups` 同构，**不是 `h % nkh`**）。
- 对本模型的 C 实现：**不需要实现这段**，但要知道 head 数 16 对 16 直接一一对应。

## A6. gated RMSNorm：先 norm 再乘 `silu(gate)`；eps=1e-6；weight 是 per-head 共享 `[128]`

MN:217-234 原文：

```python
@use_kernel_forward_from_hub("RMSNormGated")
class Qwen3_5RMSNormGated(nn.Module):
    def __init__(self, hidden_size: int, eps: float = 1e-6, **kwargs) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps
        self.activation = "silu"

    def forward(self, hidden_states: torch.Tensor, gate: torch.Tensor) -> torch.Tensor:
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        # Norm before gate
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        hidden_states = self.weight * hidden_states.to(input_dtype)
        hidden_states = hidden_states * ACT2FN[self.activation](gate.to(torch.float32))

        return hidden_states.to(input_dtype)
```

- 顺序：**先 RMSNorm（无 mean 减除、per 最后一维即 head_v_dim=128），再 `* weight`，最后 `* silu(gate)`**。源码注释直接写了 `# Norm before gate`。
- `eps = layer_norm_epsilon = config.rms_norm_eps = 1e-6`（MN:518, 539）。
- `weight` 形状 **`(128,)` = head_v_dim**，是**单个向量、被所有 head 共享**（`Qwen3_5RMSNormGated(self.head_v_dim)`，MN:539）。不是 `[16,128]`。
- ⚠️ 与 `Qwen3_5RMSNorm` 不同：这里的 `weight` 初始化是 **`torch.ones(hidden_size)`**（MN:221），前向直接 `weight * x`，**不是 `(1+w)`**。检查点里真实值 `mean=0.952, min=0.523, max=1.060`（`exp_decay.out`）——因为初始化就是 1，训练后仍在 1 附近，**若错用 `1+w` 会得到约 2 倍的缩放**。
- 调用方式（MN:656-660）：先把 `core_attn_out` 和 `z` 都 reshape 成 `(-1, head_v_dim)`，`self.norm(core_attn_out, z)`，再 reshape 回 `[b,s,value_dim]`。
- `gate`（即 `z`）来自 **`in_proj_z`**（MN:568-569），形状 `[b,s,16,128]`，`silu` 由 `ACT2FN["silu"]` 施加。

## A7. `A_log` 是「已经取过 log 的 A」

MN:535-537（构造）：

```python
# Lower bound kept away from 0 so log(A) never becomes -inf
A = torch.empty(self.num_v_heads).uniform_(0.01, 16)
self.A_log = nn.Parameter(torch.log(A))
```

MN:619 原文用：

```python
g = -self.A_log.float().exp() * F.softplus(a.float() + self.dt_bias)
```

- **`A_log` 里存的是 `log(A)`，A ∈ [0.01, 16]；代码里用 `A_log.exp()` 还原 A。**
- 实测 layer 0：`A_log = [0.258, -2.031, -2.125, ...]`，`A_log.exp() = [1.294, 0.131, 0.119, ..., 0.00137]`（范围 0.0014..1.294）——**确实都在 [0.01,16] 内**，证明是 `log(A)`。
- ⚠️ 所以正确写法是 `A = exp(A_log)`，**不是** `A = A_log`，也不是 `exp(exp(A_log))`。

## A8. decay 的符号与 gamma 的实际范围

- `g = -A * softplus(a + dt_bias)`，`A = exp(A_log) > 0`，`softplus(·) ≥ 0` ⇒ **`g ≤ 0`**。
- `gamma = exp(g) = exp(-A*softplus(...))` ∈ **`(0, 1]`**：softplus 在输入极负时 fp32 下溢为 0，此时 `g = -0.0`、`gamma = 1.0`。
- 实测（真实权重 layer 0，真实 hidden，`exp_decay.out`）：
  ```
  g        range: [-6.735915, -0.000000]     (严格 <= 0)
  gamma=exp(g) range: [0.00118749, 1.00000000]
  per-token gamma mean = 0.764758
  gamma shape = (5, 16) = [batch, seq, num_v_heads]
  ```
  即 gamma 实际落在 **[0.0012, 1.0]**，**不是 (0,1) 开区间**（有恰好 =1.0 的下溢项）。C 引擎别写 `assert gamma < 1`。
- ⚠️ **符号极易写反**：若写成 `+A*softplus`，则 gamma > 1，状态会指数爆炸。实验里 `flip sign` 的误差 = 9.18e-1（`exp_delta_order.out` 步骤 [6]），`forget exp` 的误差 = 3.06e-1。

---

# B. 全注意力（full_attention）层

## B9. RoPE 参数与实测 rotary_dim

`text_config.rope_parameters` **完整内容**（原样）：

```json
{"rope_theta": 10000000, "rope_type": "default", "partial_rotary_factor": 0.25,
 "mrope_section": [11, 11, 10], "mrope_interleaved": false}
```

源码（MN:176-184）：

```python
base = config.rope_parameters["rope_theta"]
partial_rotary_factor = config.rope_parameters.get("partial_rotary_factor", 1.0)
head_dim = getattr(config, "head_dim", None) or config.hidden_size // config.num_attention_heads
dim = int(head_dim * partial_rotary_factor)

attention_factor = 1.0  # Unused in this type of RoPE
# Compute the inverse frequencies
inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2, dtype=torch.float) / dim))
return inv_freq.to(device), attention_factor
```

MN:152-156：`rope_type = "default"` ⇒ 走 `compute_default_rope_parameters`（注意条件写的是 `if self.rope_type != "default"` 才查表）。

**实测（不是推断）**（`exp_rotary.out` / `exp_attention.out`）：

```
rope.inv_freq.shape = (32,)        attention_scaling = 1.0     rope_type = 'default'
rope.mrope_section = [11, 11, 10]
rotary_dim = 2 * len(inv_freq) = 64
check against head_dim*partial_rotary_factor = int(256*0.25) = 64   ✓
head_dim = 256  => UN-rotated tail length = 192
inv_freq[:6] = [1.0, 0.60429639, 0.36517414, 0.2206734, 0.13335215, 0.08058421]
inv_freq[-3:]= [4.532e-07, 2.738e-07, 1.655e-07]
```

（`inv_freq[j] = 1e7 ** (-2j/64)`，j=0..31：`inv_freq[0]=1.0`，`inv_freq[31]=1.655e-07`。可以手工核对：`1e7**(-62/64)=1.66e-7` ✓）

**结论：`rotary_dim = 64`**（`head_dim=256` 的**前 1/4**）。`inv_freq` 只有 **32** 个值，因为 `arange(0, 64, 2)/64` 产生 32 个指数；`recomposition_frequencies` 末尾的 `cat((f, f), -1)`（MN:213）负责把它变成 64 宽的 cos/sin。`attention_scaling = 1.0`（乘上无影响）。

## B10. `apply_rotary_pos_emb` / `rotate_half` / cos-sin 排布 / 纯文本 position_ids ★

`rotate_half`（MN:666-670）原文：

```python
def rotate_half(x):
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)
```

`apply_rotary_pos_emb`（MN:674-709）原文：

```python
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)

    # Keep half or full tensor for later concatenation
    rotary_dim = cos.shape[-1]
    q_rot, q_pass = q[..., :rotary_dim], q[..., rotary_dim:]
    k_rot, k_pass = k[..., :rotary_dim], k[..., rotary_dim:]

    # Apply rotary embeddings on the first half or full tensor
    q_embed = (q_rot * cos) + (rotate_half(q_rot) * sin)
    k_embed = (k_rot * cos) + (rotate_half(k_rot) * sin)

    # Concatenate back to full shape
    q_embed = torch.cat([q_embed, q_pass], dim=-1)
    k_embed = torch.cat([k_embed, k_pass], dim=-1)
    return q_embed, k_embed
```

- **配对方式：跨半轴的配对 `(i, i + d/2)`，不是相邻配对 `(2i, 2i+1)`。** 实测：`rotate_half([0..7]) = [-4,-5,-6,-7, 0,1,2,3]`。
- 旋转块内部（`d = rotary_dim = 64`，half = 32）：
  ```
  y[i]     = x[i]*cos[i] - x[i+32]*sin[i]      i = 0..31
  y[i+32]  = x[i]*sin[i] + x[i+32]*cos[i]      i = 0..31
  ```
  实测 `max|手写 - 真实现| = 0`（`exp_rotary.out`）。
- **cos/sin 的排布**：`cos = [c0, c1, ..., c31, c0, c1, ..., c31]`（前半 == 后半，重复两次）。实测 `torch.equal(cos[..., :32], cos[..., 32:64]) = True`（`exp_rotary.out`）。
- `recomposition_frequencies`（MN:204-213）原文：
  ```python
  def recomposition_frequencies(self, freq):
      """
      Recompose the frequencies into the final spatial layout used per each grid.
      """
      freqs_thw = freq[0]  # just overwrite the first dimension T
      for dim, offset in enumerate((1, 2), start=1):  # H, W
          length = self.mrope_section[dim] * 3
          idx = slice(offset, length, 3)
          freqs_thw[..., idx] = freq[dim, ..., idx]
      return torch.cat((freqs_thw, freqs_thw), dim=-1)
  ```
  **`cat(freq_hw, freq_hw)` 的作用**：把 32 个频率复制成 64 宽的 cos/sin，以匹配 `rotate_half` 的「跨半轴配对」约定——这样 `(x_rot*cos)` 与 `(rotate_half(x_rot)*sin)` 才能逐元素对应（配对的两半用同一个频率）。
- **纯文本（无 image token）时的 position_ids 与 cos/sin（实测）**：
  构造见 MN:1259-1271 原文：
  ```python
  # the hard coded `4` is for text, temporal, height and width.
  if position_ids is None:
      past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
      position_ids = torch.arange(inputs_embeds.shape[1], device=inputs_embeds.device) + past_seen_tokens
      position_ids = position_ids.view(1, 1, -1).expand(4, inputs_embeds.shape[0], -1)
  elif position_ids.ndim == 2:
      position_ids = position_ids[None, ...].expand(4, position_ids.shape[0], -1)

  if position_ids.ndim == 3 and position_ids.shape[0] == 4:
      text_position_ids = position_ids[0]
      position_ids = position_ids[1:]
  ```
  实测（N=3 tokens，bs=1）：
  ```
  full position_ids shape = (4, 1, 3)     (4, bs, N)
  text_position_ids shape = (1, 3)  values = [[0, 1, 2]]
  given to rotary (=[1:]) shape = (3, 1, 3)  values = [[0,1,2],[0,1,2],[0,1,2]]   <- 三个 grid 完全相同
  ```
  ⇒ `recomposition_frequencies` 里 H/W 的覆盖（MN:209-212）在纯文本下**数值上什么都不改变**（`grid0==grid1==grid2` 为 True，实测）。**纯文本等价于普通 1D RoPE**：
  ```
  cos = cat(cos(pos*inv_freq), cos(pos*inv_freq)) ; sin 同理
  ```
  实测 `max|真实现 - 直接 cat(cos,cos)| = 0.0`（`exp_rotary.out`）。
  **cos/sin 前 8 个值（pos = 0, 1, 2，rotary_dim=64）**：
  ```
   pos | cos[0:8]
    0  | [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
    1  | [0.54030234, 0.82290208, 0.93406159, 0.97575027, 0.99112177, 0.99675483, 0.99881452, 0.99956703]
    2  | [-0.41614684, 0.35433567, 0.74494207, 0.90417719, 0.96464473, 0.98704046, 0.99526101, 0.99826854]
   pos | sin[0:8]
    0  | [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    1  | [0.84147096, 0.56818324, 0.35711196, 0.21888675, 0.13295726, 0.08049703, 0.04867751, 0.02942303]
    2  | [0.90929741, 0.93511832, 0.66712916, 0.42715758, 0.26355368, 0.16047160, 0.09723961, 0.05882058]
  ```
  （pos=0 时 cos≡1、sin≡0；inv_freq[0]=1.0 对应最高频。）
- 位置 `p` 的第 `j` 个频率（j=0..31）：`theta_j = p * inv_freq[j]`，`inv_freq[j] = 10000000^(-2j/64)`。**注意分母是 `dim = 64`（rotated 部分），不是 head_dim=256。**

## B11. 未被旋转的 192 维如何处理

MN:699-708：`q_pass = q[..., rotary_dim:]` / `k_pass = k[..., rotary_dim:]`，最后 `torch.cat([q_embed, q_pass], dim=-1)` —— **原样拼接、不缩放、不旋转**。

实测 `exp_rotary.out`：

```
q_embed[..., 64:] == q[..., 64:] ?  True
q_embed[..., :64] == q[..., :64] ?  False    (必须为 False)
```

即 head_dim=256 里 **dims 0..63 被旋转，dims 64..255（192 维）原样通过**。

## B12. q_norm / k_norm（RMSNorm，[256]）在 RoPE **之前**

MN:787-797 原文（`Qwen3_5Attention.forward`）：

```python
        query_states, gate = torch.chunk(
            self.q_proj(hidden_states).view(*input_shape, -1, self.head_dim * 2), 2, dim=-1
        )
        gate = gate.reshape(*input_shape, -1)

        query_states = self.q_norm(query_states.view(hidden_shape)).transpose(1, 2)
        key_states = self.k_norm(self.k_proj(hidden_states).view(hidden_shape)).transpose(1, 2)
        value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)

        cos, sin = position_embeddings
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)
```

- **顺序：`q_proj` → reshape 到 head → `q_norm`（RMSNorm over head_dim=256）→ `apply_rotary_pos_emb`。**
  即在 **RoPE 之前**（MN:792-793 是 norm，MN:797 是 RoPE）。
- `q_norm = Qwen3_5RMSNorm(self.head_dim, eps=config.rms_norm_eps)`（MN:773-774），weight 形状 **`(256,)`**，实测 `q_norm.weight` mean=0.427 / min=-0.211 / max=1.227；`k_norm.weight` mean=0.417 / min=-0.898 / max=1.633。
- ⚠️ `Qwen3_5RMSNorm` 是 **zero-centred**：`weight` 初始化 `torch.zeros(dim)`（MN:845），前向 `output = output * (1.0 + self.weight.float())`（MN:854）。**必须用 `1+w`，不是 `w`。** 反例实测：把 norm 放到 RoPE 之后（错误）→ `max|diff| = 2.694e-02`（`exp_attention.out`）。

## B13. attention scale / causal mask / head→kv 映射 ★

MN:756-760 原文：

```python
self.head_dim = getattr(config, "head_dim", config.hidden_size // config.num_attention_heads)
self.num_key_value_groups = config.num_attention_heads // config.num_key_value_heads
self.scaling = self.head_dim**-0.5
self.attention_dropout = config.attention_dropout
self.is_causal = True
```

- **scale = `head_dim ** -0.5` = `256**-0.5 = 0.0625`**（MN:758；传入 `eager_attention_forward(..., scaling=self.scaling)`，MN:813）。施加点：`attn_weights = torch.matmul(query, key_states.transpose(2,3)) * scaling`（MN:737）。
- **causal mask：是。** `self.is_causal = True`（MN:760）；掩码由 `Qwen3_5TextModel.forward` 用 `create_causal_mask(**mask_kwargs)` 造好并按 layer type 分发（MN:1283-1295）。`eager_attention_forward` 里 `attn_weights = attn_weights + attention_mask`（MN:738-739），softmax 在 fp32（MN:741）。实测去掉 mask 的误差 = 5.29e-1。
- **`repeat_kv` 顺序（MN:712-721）原文**——这决定 head→kv 映射：
  ```python
  def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
      batch, num_key_value_heads, slen, head_dim = hidden_states.shape
      if n_rep == 1:
          return hidden_states
      hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, slen, head_dim)
      return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)
  ```
  expand 出来的轴顺序是 `(nkv, n_rep)`，flatten 后 head 索引为 `h = kv * n_rep + r` ⇒ **`kv = h // n_rep`，即 `kv_head = h // num_key_value_groups`**。
- **实测数值证明**（`exp_rotary.out`，nkv=2, groups=4，kv head0=1、kv head1=2）：
  ```
  repeat_kv(...)[per head] = [1.0, 1.0, 1.0, 1.0, 2.0, 2.0, 2.0, 2.0]
  => head h 用 kv head floor(h/groups) = [0,0,0,0,1,1,1,1]
     (NOT h % nkv -> [0,1,0,1,0,1,0,1])
  ```
  本例 `nkv = 2, num_attention_heads = 8, groups = 4` ⇒ **`kv = h / 4`（整除）**：heads 0-3 用 kv0，heads 4-7 用 kv1。
  用 `h % nkv` 的误差实测 = **8.561e-01**（`exp_attention.out`，是最大的单项误差）。
- 手工按 `h//groups` + causal mask + scale=1/16 复算，与真模块**逐位一致（max diff = 0.000e+00）**。

## B14. q_proj 输出 4096 维的分段方式与 gate

MN:787-790, 817-818 原文：

```python
        query_states, gate = torch.chunk(
            self.q_proj(hidden_states).view(*input_shape, -1, self.head_dim * 2), 2, dim=-1
        )
        gate = gate.reshape(*input_shape, -1)
...
        attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        attn_output = attn_output * torch.sigmoid(gate)
```

- `q_proj = nn.Linear(hidden_size, num_attention_heads * head_dim * 2, bias=False)`（MN:761-763）→ 4096 = 8 × 512。
- 先 `.view(*input_shape, -1, head_dim*2)` 即 `[b, s, 8, 512]`，再 `torch.chunk(..., 2, dim=-1)` 切成 `[b,s,8,256]` 两块。
- ⇒ **是「每 head 内部 `[q(256) | gate(256)]`」的交织式分段**，**不是** `[q(2048) | gate(2048)]` 的分块式。
  错误的分块切法实测误差 = **6.330e-01**（`exp_attention.out`）。
- gate 用 **`torch.sigmoid`**（MN:818，**不是 silu**），乘在 `o_proj` **之前**、attention 输出 reshape 成 `[b,s,2048]` **之后**（MN:817-820）。

---

# C. 整体

## C15. `layer_types` 精确内容（24 项）

```
['linear_attention', 'linear_attention', 'linear_attention', 'full_attention',
 'linear_attention', 'linear_attention', 'linear_attention', 'full_attention',
 'linear_attention', 'linear_attention', 'linear_attention', 'full_attention',
 'linear_attention', 'linear_attention', 'linear_attention', 'full_attention',
 'linear_attention', 'linear_attention', 'linear_attention', 'full_attention',
 'linear_attention', 'linear_attention', 'linear_attention', 'full_attention']
```

- `full_attention` 的下标：**[3, 7, 11, 15, 19, 23]**（共 6 层）；`linear_attention` 18 层。`full_attention_interval = 4`。
- 用法：`self.block_type = config.layer_types[layer_idx]`（MN:865），`linear_attention` → `Qwen3_5GatedDeltaNet`，`full_attention` → `Qwen3_5Attention`（MN:866-869）。
- 层结构（MN:883-914）：`h = h + TokenMixer(input_layernorm(h))`；`h = h + mlp(post_attention_layernorm(h))`。两个 RMSNorm 都是 zero-centred（`1+w`）。

## C16. final norm + lm_head / tie / 权重名 / logits 方向

- **tie embeddings：是。** `tie_word_embeddings = true`；MN:1665 原文：
  ```python
  _tied_weights_keys = {"lm_head.weight": "model.embed_tokens.weight"}
  ```
  检查点里**没有 `lm_head.weight`**（实测 `any("lm_head" in k) == False`）。运行时 `mdl.lm_head.weight.data_ptr() == mdl.model.embed_tokens.weight.data_ptr()` 为 **True**。
- **`lm_head` 权重在 safetensors 里的名字**：**不存在**。要用的张量是
  **`model.language_model.embed_tokens.weight`**，形状 **(248320, 1024)**（fp16/bf16 存盘，`dtype` 见 dump）。
  也可忽略 `_keys_to_ignore_on_load_unexpected = [r"^mtp.*", r"^model.visual.*"]`（MN:1670）。
- final norm 权重名：**`model.language_model.norm.weight`**，形状 `(1024,)`，zero-centred（`1+w`），在最后一层之后、lm_head 之前（MN:1302）。
- `self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)`（MN:1676）；
  MN:1730：`logits = self.lm_head(hidden_states[:, slice_indices, :])`。
- **logits = `x @ W^T`**（`nn.Linear` 的 `weight` 布局是 `[out_features, in_features] = [248320, 1024]`）。
  **不是 `W @ x`**（那需要 W 为 `[hidden, vocab]`，形状不匹配）。
- 实测端到端：prompt `"The capital of France is"`，最后一位 logits argmax = **11751 → `' Paris'`**，手动 `final_norm(x) @ W^T` 的 argmax 同为 11751（`exp_attention.out`）。
- MLP（MN:824-837）：`down_proj(act_fn(gate_proj(x)) * up_proj(x))`，`act_fn = silu`；权重 `(3584,1024)/(3584,1024)/(1024,3584)`，无 bias。

## C17. 对 C 引擎的「常见踩坑清单」

按「易写反 × 后果严重」排序。每条给正确写法。

| # | 坑（错误写法） | 正确写法（附出处） |
|---|---|---|
| 1 | **head→kv 映射用 `h % nkv`** | **`kv = h / groups`（整除），groups = 8/2 = 4**。出处 MN:720-721 expand 轴顺序 `(nkv, n_rep)`。实测错法误差 8.56e-1（最大） |
| 2 | **delta rule 用未衰减的 state 读 kv_mem** | **先 `S *= exp(g)`，再 `kv_mem = S^T k`**，然后 `delta = beta*(v-kv_mem)`、`S += k⊗delta`、`o = S^T q`。出处 MN:483-491；实测差异 7.6e-2 |
| 3 | **`A_log` 直接当 A 用** | **`A = exp(A_log)`**，`g = -A*softplus(a+dt_bias)`。出处 MN:535-537 + MN:619 |
| 4 | **decay 符号写反 / 忘 exp** | `gamma = exp(g)`，`g ≤ 0`，`gamma ∈ (0,1]`；**乘**（不是除）到 state 上。实测错法误差 9.2e-1 |
| 5 | **state 用 `[v,k]` 布局** | **`S` 是 `[k_dim, v_dim]`**（MN:473）。`kv_mem` 沿 -2（k_dim）求和，写回是 `k[:,None] * delta[None,:]` |
| 6 | **conv 右补 / 补错方向 / silu 放错位置 / 加 bias** | **左补 K-1 = 3 个零**（`F.conv1d(..., padding=3)[:, :, :seq_len]`，MN:278-286）；**无 bias**；**silu 紧跟 conv、在 QKV 切分之前**（MN:287-288, 594） |
| 7 | **QKV 切分按下标交织** | `[0:2048]=Q, [2048:4096]=K, [4096:6144]=V` 连续三段（MN:603-611） |
| 8 | **l2norm 用 `x/sqrt(sum+eps)` 之外的形式 / eps 放错位置 / 忘了 q 的 `1/sqrt(128)`** | `x * rsqrt(sum(x^2, -1, keepdim) + 1e-6)`，eps **加在平方和里面**；之后 `q *= 128**-0.5`（MN:294-297, 344-345, 467-468）。只对 q 缩放 |
| 9 | **q_proj 4096 维按 `[q2048\|gate2048]` 切** | 每 head `[q(256)\|gate(256)]`，即 `view(...,8,512)` 后 `chunk(2,-1)`；gate 用 **sigmoid**（MN:787-789, 818） |
| 10 | **RoPE 旋转整个 head_dim / 用相邻配对** | 只旋转 **前 64 维**（`partial_rotary_factor=0.25`）；配对是**跨半轴 `(i, i+32)`**；cos/sin 是 32 个频率**重复两次**；频率分母是 `dim=64`（MN:179-183, 666-670, 698-708） |
| 11 | **norm 用 `w` 而不是 `1+w`（分层） / gated-norm 用 `1+w`** | `Qwen3_5RMSNorm`（input_layernorm / post_attention_layernorm / q_norm / k_norm / final norm）用 **`(1+w)`**（MN:845, 854）；`Qwen3_5RMSNormGated`（GDN 的 norm）用 **`w`（初始化即 1）**（MN:221, 231） |
| 12 | **gated norm 顺序写成 `silu(gate)*norm(x)` 或先 gate 后 norm** | **先 RMSNorm(最后一维=128) → 乘 weight → 再乘 `silu(gate)`**（MN:228-232，源码注释 `# Norm before gate`） |
| 13 | **k_norm/q_norm 放到 RoPE 之后** | 先 norm 后 RoPE（MN:792-797）。实测错法误差 2.69e-2 |
| 14 | **q/k 的 RoPE 用了错误的 inv_freq 分母或漏 `attention_scaling`** | `inv_freq = 1/(1e7 ** (arange(0,64,2)/64))`，32 个值；`attention_scaling = 1.0` |
| 15 | **attention 忘 scale 或 scale 用 `1/sqrt(128)`** | `scale = 256**-0.5 = 0.0625`（MN:758）。错用 1.0 的误差 2.91e-1 |
| 16 | **注意力不 causal / 忘加 mask** | causal（MN:760）；错法误差 5.29e-1 |
| 17 | **输出用更新前的 state** | `o` 用**本 token 更新后**的 S（MN:489 → 491）。错法误差 2.57e-1 |
| 18 | **decode 时忘记维护 conv_state / recurrent state** | `conv_state` 长 3（MN:258-261）；recurrent state 形状 `[n_vh,128,128]`（MN:473）。单 token 走 `torch_recurrent_gated_delta_rule`（MN:625-637），多 token 走 chunk（MN:639-650） |
| 19 | **实现 `num_v_heads//num_k_heads > 1` 的 repeat_interleave** | 本例 **16 vs 16，不触发**（MN:620-622）；多余实现反而引入错误 |
| 20 | **lm_head 去找 `lm_head.weight`** | 权重名 `model.language_model.embed_tokens.weight`，**tied**（MN:1665）；logits = `x @ W^T`，`W` 形状 `[248320, 1024]` |
| 21 | **用 `model.` 前缀加载** | 实际前缀 **`model.language_model.`**；忽略 `model.visual.*` 与 `mtp.*`（MN:1670） |
| 22 | **final norm 漏掉** | MN:1302：最后一层输出后必须过 `self.norm`（zero-centred, `1+w`）再进 lm_head |
| 23 | **g 当成 per-k-channel 向量** | `g` 形状 `[b, T, num_v_heads]`（MN:319-320），一个标量广播到整个 `[k,v]` 状态 |
| 24 | **`rope_type` 判断写反** | MN:154 `if self.rope_type != "default"` 才走查表函数；`"default"` 用 `compute_default_rope_parameters` |

---

## 附：实验脚本与可复现输出

| 脚本 | 覆盖问题 | 关键输出文件 |
|---|---|---|
| `spec_experiments/exp_delta_order.py` | Q4（两种顺序的数值差异 + 与真实现逐位对比） | `exp_delta_order.out` |
| `spec_experiments/exp_decay.py` | Q1,Q2,Q3,Q4c,Q5,Q6,Q7,Q8 + layer-0 GDN 端到端复核 | `exp_decay.out` |
| `spec_experiments/exp_rotary.py` | Q9,Q10,Q11,Q13 | `exp_rotary.out` |
| `spec_experiments/exp_attention.py` | Q9,Q10,Q12,Q13,Q14,Q16 + 整模型 logits | `exp_attention.out` |
| `spec_experiments/dump_cfg.txt`, `dump_ckpt.txt` | Q0,Q15,Q16（config 与张量清单实测） | — |

全部脚本用 `/home/wisedu/laya/bin/python` 在 **CPU** 上运行；三个 `exp_*.py` 均 exit 0。

**未在本 SPEC 覆盖**：视觉塔（`model.visual.*`，MN:961+）与 MTP 头（`mtp.*`）——纯文本推理不需要，且用户未要求。
