# Qwen3.5 GDN 线性注意力实现细节

> 源码位置：`transformers/models/qwen3_next/modeling_qwen3_next.py`

## 1. 层类型判断

### 判断条件
通过 `config.layer_types[layer_idx]` 区分层类型：
- `"linear_attention"` → GDN 层（Qwen3NextGatedDeltaNet）
- `"full_attention"` → 标准全注意力层（Qwen3NextAttention）

### 代码片段
```python
# modeling_qwen3_next.py:874-879
self.block_type = config.layer_types[layer_idx]
if self.block_type == "linear_attention":
    self.linear_attn = Qwen3NextGatedDeltaNet(config, layer_idx)
elif self.block_type == "full_attention":
    self.self_attn = Qwen3NextAttention(config, layer_idx)
```

### 配置默认值
```python
# configuration_qwen3_next.py:131-137
if self.layer_types is None:
    # 默认交替模式：每3层一个 full_attention
    self.layer_types = [
        "linear_attention" if (i % 3 != 2) else "full_attention"
        for i in range(self.num_hidden_layers)
    ]
```

---

## 2. GDN 核心公式

### 2.1 输入投影

**权重名称和维度**：
| 权重 | 形状 | 说明 |
|------|------|------|
| `in_proj_qkvz` | [hidden_size, key_dim*2 + value_dim*2] | 投影到 Q, K, V, Z |
| `in_proj_ba` | [hidden_size, num_v_heads * 2] | 投影到 beta, alpha |
| `conv1d.weight` | [conv_dim, 1, kernel_size] | 因果卷积（conv_dim = key_dim*2 + value_dim） |
| `out_proj` | [value_dim, hidden_size] | 输出投影 |

**代码片段**：
```python
# modeling_qwen3_next.py:607-611
projection_size_qkvz = self.key_dim * 2 + self.value_dim * 2  # K*2 + V*2
projection_size_ba = self.num_v_heads * 2
self.in_proj_qkvz = nn.Linear(self.hidden_size, projection_size_qkvz, bias=False)
self.in_proj_ba = nn.Linear(self.hidden_size, projection_size_ba, bias=False)
```

### 2.2 门控机制

**gate_beta 计算**：
```python
# modeling_qwen3_next.py:719
beta = b.sigmoid()  # b 来自 in_proj_ba 的第一部分
```

**gate_gamma（decay）计算**：
```python
# modeling_qwen3_next.py:721
g = -self.A_log.float().exp() * F.softplus(a.float() + self.dt_bias)
```
- `a` 来自 `in_proj_ba` 的第二部分
- `A_log` 是可学习参数，初始化为 `log(A)`，其中 `A ~ Uniform(0.01, 16)`
- `dt_bias` 是可学习的时间步偏置

**代码片段**：
```python
# modeling_qwen3_next.py:615-619
self.dt_bias = nn.Parameter(torch.ones(self.num_v_heads))

# A 初始化
A = torch.empty(self.num_v_heads).uniform_(0.01, 16)
self.A_log = nn.Parameter(torch.log(A))
```

### 2.3 线性注意力状态

**KV 状态的初始化**：
```python
# modeling_qwen3_next.py:549-551 (recurrent 模式)
recurrent_state_shape = (batch_size, num_v_heads, k_head_dim, v_head_dim)
last_recurrent_state = torch.zeros(recurrent_state_shape, dtype=value.dtype, device=value.device)
```

**状态更新公式**（recurrent 模式）：
```python
# modeling_qwen3_next.py:557-568
for i in range(sequence_length):
    q_t, k_t, v_t = query[:, :, i], key[:, :, i], value[:, :, i]
    
    # 1. 衰减旧状态
    decay_t = decay[:, :, i].exp()[..., None, None]  # exp(g)
    last_recurrent_state = last_recurrent_state * decay_t
    
    # 2. 计算当前 token 的增量
    beta_t = beta[:, :, i].unsqueeze(-1)
    kv_mem = (last_recurrent_state * k_t.unsqueeze(-1)).sum(dim=-2)
    delta = (v_t - kv_mem) * beta_t  # delta rule
    
    # 3. 更新状态
    last_recurrent_state = last_recurrent_state + k_t.unsqueeze(-1) * delta.unsqueeze(-2)
    
    # 4. 计算输出
    core_attn_out[:, :, i] = (last_recurrent_state * q_t.unsqueeze(-1)).sum(dim=-2)
```

**数学公式**：
```
S_t = exp(g_t) * S_{t-1}                    # 衰减
kv_mem = S_t @ k_t                          # 预测
delta = (v_t - kv_mem) * beta_t             # 增量（delta rule）
S_t = S_t + k_t @ delta^T                   # 更新
output_t = S_t @ q_t                        # 输出
```

### 2.4 输出计算

**最终输出公式**：
```python
# modeling_qwen3_next.py:758-766
# 1. 对 attention 输出做 gated RMSNorm
core_attn_out = self.norm(core_attn_out, z)  # z 来自 in_proj_qkvz 的最后一部分

# 2. 输出投影
output = self.out_proj(core_attn_out)
```

**Gated RMSNorm 实现**：
```python
# modeling_qwen3_next.py:66-75
def forward(self, hidden_states: torch.Tensor, gate: torch.Tensor) -> torch.Tensor:
    input_dtype = hidden_states.dtype
    hidden_states = hidden_states.to(torch.float32)
    variance = hidden_states.pow(2).mean(-1, keepdim=True)
    hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
    hidden_states = self.weight * hidden_states.to(input_dtype)
    hidden_states = hidden_states * ACT2FN[self.activation](gate.to(torch.float32))  # silu(gate)
    return hidden_states.to(input_dtype)
```

---

## 3. 标准全注意力

### 与 GDN 的区别

| 特性 | GDN | 标准全注意力 |
|------|-----|-------------|
| 复杂度 | O(1) 状态大小 | O(L) KV cache |
| 位置编码 | 无 RoPE | 使用 RoPE |
| 门控 | beta + gamma decay | sigmoid(gate) |
| 注意力计算 | 线性递归 | Softmax(QK^T)V |

### 关键代码片段
```python
# modeling_qwen3_next.py:277-310
# Q 投影包含 gate
query_states, gate = torch.chunk(
    self.q_proj(hidden_states).view(*input_shape, -1, self.head_dim * 2), 2, dim=-1
)

# QK normalization
query_states = self.q_norm(query_states.view(hidden_shape)).transpose(1, 2)
key_states = self.k_norm(self.k_proj(hidden_states).view(hidden_shape)).transpose(1, 2)

# RoPE
query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

# 标准 attention
attn_output, attn_weights = attention_interface(...)

# gate 输出
attn_output = attn_output * torch.sigmoid(gate)
```

---

## 4. RoPE 实现

### 应用方式
RoPE 仅用于标准全注意力层，GDN 层不使用 RoPE。

```python
# modeling_qwen3_next.py:162-197
def apply_rotary_pos_emb(q, k, cos, sin, unsqueeze_dim=1):
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)
    
    rotary_dim = cos.shape[-1]
    q_rot, q_pass = q[..., :rotary_dim], q[..., rotary_dim:]
    k_rot, k_pass = k[..., :rotary_dim], k[..., rotary_dim:]
    
    # 应用旋转
    q_embed = (q_rot * cos) + (rotate_half(q_rot) * sin)
    k_embed = (k_rot * cos) + (rotate_half(k_rot) * sin)
    
    q_embed = torch.cat([q_embed, q_pass], dim=-1)
    k_embed = torch.cat([k_embed, k_pass], dim=-1)
    return q_embed, k_embed

def rotate_half(x):
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)
```

---

## 5. 数据类型

### 类型转换处理

**A_log 和 norm.weight 为 F32，其余 BF16**：

```python
# modeling_qwen3_next.py:721
g = -self.A_log.float().exp() * F.softplus(a.float() + self.dt_bias)

# RMSNorm 内部 (modeling_qwen3_next.py:67-73)
hidden_states = hidden_states.to(torch.float32)
variance = hidden_states.pow(2).mean(-1, keepdim=True)
hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
hidden_states = self.weight * hidden_states.to(input_dtype)
```

**原因**：数值稳定性。A_log 的指数运算和 RMSNorm 需要更高精度。

---

## 6. 张量 Shape 汇总

### GDN 层张量

| 张量名称 | Shape | dtype | 说明 |
|----------|-------|-------|------|
| `hidden_states` | [batch, seq_len, hidden_size] | BF16 | 输入 |
| `in_proj_qkvz.weight` | [key_dim*2 + value_dim*2, hidden_size] | BF16 | QKVZ 投影 |
| `in_proj_ba.weight` | [num_v_heads*2, hidden_size] | BF16 | beta/alpha 投影 |
| `query` | [batch, seq_len, num_k_heads, head_k_dim] | BF16 | 查询 |
| `key` | [batch, seq_len, num_k_heads, head_k_dim] | BF16 | 键 |
| `value` | [batch, seq_len, num_v_heads, head_v_dim] | BF16 | 值 |
| `z` | [batch, seq_len, num_v_heads, head_v_dim] | BF16 | 输出门控 |
| `b` | [batch, seq_len, num_v_heads] | BF16 | beta 原始值 |
| `a` | [batch, seq_len, num_v_heads] | BF16 | alpha 原始值 |
| `beta` | [batch, seq_len, num_v_heads] | BF16 | sigmoid(b) |
| `g` | [num_v_heads] | F32 | decay 参数 |
| `A_log` | [num_v_heads] | F32 | 可学习衰减对数 |
| `dt_bias` | [num_v_heads] | BF16 | 时间步偏置 |
| `conv1d.weight` | [key_dim*2 + value_dim, 1, kernel_size] | BF16 | 因果卷积 |
| `recurrent_state` | [batch, num_v_heads, head_k_dim, head_v_dim] | F32 | KV 状态 |
| `norm.weight` | [head_v_dim] | F32 | RMSNorm 权重 |
| `out_proj.weight` | [hidden_size, value_dim] | BF16 | 输出投影 |

### 配置参数默认值
```python
# configuration_qwen3_next.py
linear_key_head_dim = 128        # head_k_dim
linear_value_head_dim = 128      # head_v_dim
linear_num_key_heads = 16        # num_k_heads
linear_num_value_heads = 32      # num_v_heads
linear_conv_kernel_dim = 4       # 卷积核大小

# 计算得出
key_dim = linear_num_key_heads * linear_key_head_dim      # 16 * 128 = 2048
value_dim = linear_num_value_heads * linear_value_head_dim # 32 * 128 = 4096
```

---

## 7. 核心流程图

```
输入 hidden_states [B, L, H]
    │
    ├──► in_proj_qkvz ──► [Q, K, V, Z]
    │
    └──► in_proj_ba ──► [b, a]
              │
              ├──► b.sigmoid() ──► beta
              │
              └──► -A_log.exp() * softplus(a + dt_bias) ──► g (decay)
              
因果卷积 Conv1D 应用于 Q, K, V
    │
    ▼
┌─────────────────────────────────┐
│  Gated Delta Rule 线性注意力    │
│  (recurrent 或 chunk 模式)      │
│                                 │
│  S_t = exp(g) * S_{t-1}         │
│  delta = (V - S_t @ K) * beta   │
│  S_t = S_t + K @ delta^T        │
│  output = S_t @ Q               │
└─────────────────────────────────┘
    │
    ▼
Gated RMSNorm(output, Z)
    │
    ▼
out_proj ──► 最终输出
```