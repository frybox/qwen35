# HANDOFF — Qwen3.5 推理引擎 CUDA 加速

> 本文件是给新会话的任务交接书。旧会话上下文已污染（大量 read 输出被折叠），
> 一切以仓库里的代码和本文件为准；动手前请重新 read 关键文件确认细节。

## 1. 任务要求（使命）

把 `/mnt/llm-dev/qwen35` 里已可用的 **CPU 版 Qwen3.5 GGUF 推理引擎移植/加速到 CUDA**：

- 从零实现，不依赖框架（不调 llama.cpp / ggml / PyTorch）。
- 硬件：RTX 4090 D ×8（`nvidia-smi` 可见）；`nvcc` CUDA 13.2 已装（`/usr/local/cuda/bin/nvcc`）。
- 模型文件：
  - `/mnt/Qwen3.5-0.8B-GGUF/Qwen3.5-0.8B-Q8_0.gguf`
  - `/mnt/Qwen3.5-2B-GGUF/Qwen3.5-2B-Q8_0.gguf`
  - `/mnt/Qwen3.5-4B-GGUF/Qwen3.5-4B-Q8_0.gguf`

### 验收标准

1. **正确性**：CUDA 版与 CPU 版在 0.8B（必）和 4B（必）上**逐 token 对拍一致**
   （greedy argmax，同一 prompt）。
2. **速度**：报告 prefill / decode 的 ms/token（或 tok/s），确认加速成立。
   现有 CLI 已打印 prefill/decode 统计，直接可比。
3. **可复现构建**：`make`（或 `zig build`）一键产出 CPU 与 CUDA 两个二进制。
4. 原有 CPU 单元测试保持通过（`zig build test`）。

## 2. 仓库现状（git 状态已核对）

```
git log:  5d00af1 engine: matvec directly on Q8_0 weights...
          5790f08 Initial commit
git status:  M src/engine.zig        (CPU 引擎，未提交的 4B 支持等改动)
             ?? Makefile             (CPU 目标：z / zfast / run0.8b / run2b / run4b)
             ?? src/engine_cuda.cu   (★ 已写好全部 kernel + launcher，未接线)
```

- `src/engine.zig`：CPU 引擎（GGUF 解析 + 加载 + step/generate），**已可用**。
- `src/engine_cuda.cu`（648 行）：全部 `__global__` kernel + 扁平 `extern "C"` launcher
  已写好，**但没有构建接线**（build.zig / Makefile 均无 nvcc）。
- `build.zig`：目前只构建 CPU 二进制 `zig-out/bin/qwen35` 和 test。
- CPU 版运行方式：`zig-out/bin/qwen35 run <file.gguf> <prompt> [-n N]`
  （Makefile 里有 `run0.8b` / `run2b` / `run4b`）。
- 参考项目（可工作的 CUDA+Zig 构建模板）：`/mnt/llm-dev/llama2/zig/`
  （看它的 `build.zig` 里 nvcc 编译与链接部分；`src/llama2q_cuda.zig` 是 host 侧范式）。
- 数值规范文档：`ref/docs/SPEC.md`、`ref/docs/GDN_IMPL.md`；C 参考实现 `ref/c/`。
- GGUF 解析验收：`tests/run_acceptance.sh`（本任务不需要动，别弄坏）。

## 3. engine_cuda.cu 已提供的 C 接口（host 直接 extern 调用即可）

运行时（内部检查 CUDA 错误，失败打 stderr 并 exit）：

```c
int    q35_cuda_init(void);
int    q35_cuda_sync(void);
void  *q35_malloc(size_t n);
int    q35_free(void *p);
int    q35_h2d(void *d, const void *h, size_t n);
int    q35_d2h(void *h, const void *d, size_t n);
int    q35_d2d(void *d, const void *s, size_t n);   // 设备→设备
int    q35_zero(void *d, size_t n);
```

算子 launcher（全部已存在，签名逐字如下，**不要改 .cu 除非必须**）：

```c
void q35_quantize_q8(const float *x, unsigned char *out, int nblocks);
// out = Q8_0 量化 x（nblocks*32 个元素）；Q8_0 块 = 34 字节 = f16 scale(小端) + 32×i8

void q35_matvec_q8(const unsigned char *w, const unsigned char *xq,
                  float *out, int rows, int row_blocks);
// W: rows 行 Q8_0（行主序，row_blocks=cols/32），xq: 已量化的 x；out[r]=W[r]·x

void q35_matvec_f32(const float *w, const float *x, float *out, int rows, int cols);

void q35_embed_row(const unsigned char *w, float *x, int row, int n);
// 从 Q8_0 权重表取第 row 行反量化到 x（embedding 查表）

void q35_rmsnorm_rows(const float *x, const float *w, float *out,
                      int rows, int n, float eps);

void q35_rmsnorm_strided(const float *x, const float *w, float *out,
                         int rows, int n, int stride, float eps);
// q/k 按 head 切分做 RMSNorm 用

void q35_l2norm(float *x, int rows, int n, float eps, float scale);
// x[r] = l2norm(x[r]) * scale；eps 在 sum 内：1/sqrt(Σx²+eps)

void q35_rope(float *x, int rows, int hd, int rope_dim, int stride,
              int pos, float base);
// 只旋转每行前 rope_dim(=64) 维，交叉半区配对 (i, i+32)

void q35_conv1d(const float *wconv, float *x, float *hist, int conv_dim, int k);
// 因果 conv，in-place x；hist 是 [conv_dim]*(k-1) 的历史（原始未 silu 样本）；
// wconv 是 [conv_dim]*k channel-major taps；conv 后 silu

void q35_beta_gamma(const float *a, const float *b, const float *abar,
                    const float *dtb, float *beta, float *gamma, int nv);
// beta=sigmoid(b)；gamma=exp(abar * softplus(a+dtb))

void q35_gdn_state(float *qkv, float *state, const float *beta,
                  const float *gamma, float *out, int nv, int nk, int kd, int vd);
// GDN 状态更新（decay/kv_mem/outer/o=Sᵀq），q、k 从 qkv 里按 CPU 版相同布局取

void q35_gated_rmsnorm(float *out, const float *w, const float *z,
                       int rows, int n, float eps);
// out = rmsnorm(out)*w*silu(z)，按 head 行

void q35_attn(const float *q, const float *kc, const float *vc, const float *gate,
              float *out, int heads, int kv_heads, int hd, int n_t,
              int kv_dim, int q_stride, float scale);
// GQA 全注意力 + sigmoid gate 一步完成（kv head = h // (heads/kv_heads)）

void q35_silu_mul(float *h, const float *h1, int n);   // h = silu(h)*h1
void q35_sigmoid_mul(float *o, const float *g, int n);  // o *= sigmoid(g)
void q35_add(float *a, const float *b, int n);         // a += b
void q35_argmax(const float *x, int *outidx, int n);   // 首个最大值（平局取低索引）
```

入口约定（**重要**）：`engine_cuda.cu` 末尾自带

```c
extern "C" int q35_main(int argc, char **argv);
int main(int argc, char **argv) { return q35_main(argc, argv); }
```

即：进程入口是 .cu 的 `main()`，**Zig host 必须导出 `q35_main`**。
新 host 文件建议命名 `src/main_cuda.zig`，其中

```zig
export fn q35_main(argc: c_int, argv: [*] [*]u8) c_int;
```

构建时把 `engine_cuda.cu` 用 nvcc 编成目标文件/共享库，与 Zig 编译单元一起链接
（`-lcudart`）。两个 main 冲突的问题参照 `/mnt/llm-dev/llama2/zig/build.zig`
的解法（那边有能跑通的完整示例）。

## 4. engine.zig CPU 版结构（对拍基准 + 复刻对象）

- `Config`：hidden, n_layers, n_heads, n_kv_heads, head_dim, n_ff, vocab,
  `gdn_n_heads`（nv，0.8B=16 / 4B=32）、`gdn_num_k_heads`（nk，两者都=16）、
  `gdn_head_dim`（kd=128）、`conv_kernel`=4、`rope_freq_base`、`rotary_dim`=64、
  `rms_eps`、`eos_token_id`。0.8B: hidden=1024；4B: hidden=2560, conv_dim=8192。
  0.8B 全注意力层：n_heads=16, n_kv_heads=4, head_dim=256（gqa=4）。
- `Engine` 字段（scratch，设备版照抄尺寸）：`x, x1, x2`（hidden）、`qkv`
  （max(gdn_conv_dim, 2*n_heads*head_dim)）、`z, gdn_out`（max(value_dim, q_dim)）、
  `ab`（2*nv）、`q`（q_dim）、`k, v`（kv_dim）、`attn`（max_seq）、
  `mlp_h, mlp_h1`（n_ff）、`logits`（vocab）、`delta, kv_mem`（kd）、
  `q8_scratch`（`max(gdn_conv_dim, 2*n_heads*head_dim)/32*34` 字节）、
  每层 `kv`（k/v 各 max_seq*kv_dim f32）、每 GDN 层 `gdn`（conv 历史
  `conv_dim*(K-1)`、state `nv*kd*vd`）。
- 线性层 `LinearW`：`.q8`（`Q8Row`：mmap 视图，`row_blocks=cols/32`）或 `.f32`。
  Q8_0 块格式与 §3 相同。
- `step(token, pos)` 流程（设备版逐行对应）：
  1. embedding 查表 → `x`（q8: `k_embed_row`）。
  2. 每层按 `cfg.isFullAttention(li)` 走 GDN 或全注意力，再 MLP。
  3. 最后 `rmsNorm(x1, x, output_norm)` → `logits = token_embd @ x1`
     （tied，vocab 行 Q8 matvec）→ `argmax`。
- **GDN 层**（`forwardGdn`）：rmsnorm → `attn_qkv`[hidden, conv_dim] →
  `attn_gate`[hidden, value_dim]=z → `ssm_alpha`=a、`ssm_beta`=b（各 [hidden, nv]）
  → conv1d（in-place qkv，含历史+silu）→ beta/gamma → 每 v head h：
  kh=h%nk，q_h=qkv[kh*kd], k_h=qkv[nk*kd+kh*kd], v_h=qkv[2*nk*kd+h*kd]；
  l2norm(q_h)、l2norm(k_h)，q_h *= 1/sqrt(kd)；decay S*=gamma →
  kv_mem=Sᵀk → delta=beta*(v-kv_mem) → S+=outer(k,delta) → o=Sᵀq（更新后）
  → gated rmsnorm（ssm_norm 权重, silu(z)）→ `ssm_out` matvec → x+=。
  - **V slot 是 tiled 顺序**：slot h ↔ HF v head (h%nk)*nvp + h/nk，其 k head 是
    h%nk。4B 时两个 v head 共用同一 k head（重复 l2norm 无副作用）。
  - `ssm_a` 在 GGUF 里已是 `-exp(A_log)`（负值），直接当 A 用。
- **全注意力层**（`forwardFull`）：rmsnorm → `attn_q`[hidden, 2*q_dim]→qg，
  交织拆 q/gate（每 head 前 256 是 q，后 256 是 gate）→ k, v matvec →
  每 head q_norm / 每 kv head k_norm（RMSNorm，GGUF 权重已折叠 1+w）→
  RoPE64（仅前 64 维）→ KV cache 写 slot=cache.len → GQA 注意力
  （kv head = h//gqa，scale=1/sqrt(hd)）→ sigmoid(gate) → `attn_output`
  matvec → x+=。
- **MLP**：rmsnorm → ffn_gate / ffn_up matvec → silu(gate)*up → ffn_down
  matvec → x+=。
- 数学细节（与 .cu 已按此实现）：l2norm eps=1e-6 在 sum 内；softplus 有
  x>20 阈值分支；softmax 减 max；argmax 平局取低索引。

## 5. 下一步（执行计划）

1. **写 `src/main_cuda.zig`**（Zig host，导出 `q35_main`）：
   - 复刻 `engine.zig::load/step/generate` 的控制流；
   - 权重：GGUF mmap 直接当设备指针用（mmap 是 host 内存，首次用前
     `q35_h2d` 到设备，或评估 pinned 拷贝策略；Q8 权重可留在 host 映射区
     但 kernel 需要设备地址——**先 h2d 一次全部权重**，0.8B ~0.8GB 没问题）；
   - 所有 CPU 算子换成 §3 的 `q35_*` launcher；
   - scratch（x/x1/qkv/state/…）全部 `q35_malloc`；
   - 需要 CPU 侧搬数据的少数步骤（如 qg 交织、KV cache 写 slot、embedding
     查表之外的拷贝）：v1 用 `q35_d2h`+`q35_h2d` 或 `q35_d2d` 解决，正确优先。
2. **构建接线**：`build.zig` 增加 CUDA 目标（nvcc 编 `engine_cuda.cu`
   → 目标文件，与 Zig 链接，`-lcudart`）；Makefile 加 `cuda` /
   `run0.8b-cuda` 等目标。模板：`/mnt/llm-dev/llama2/zig/build.zig`。
3. **调试对拍**：0.8B 先跑通，与 CPU 版同 prompt 逐 token diff；
   对不上时逐层加打印定位（embedding、conv、gdn_state、attn 各阶段）。
4. **4B 对拍**（注意 4B 的 nv≠nk 分支）。
5. **测速 + 汇报**：prefill/decode tok/s，CPU vs CUDA 对比。

## 6. 已知坑 / 注意事项

- **旧会话教训**：本目录 `read` 大文件输出会被折叠，且旧会话多次重复读同一
  文件把上下文撑爆。新会话：一次 read 定位到行段，别整文件反复读；
  大文件用 grep + 小段 read。
- .cu 里 kernel 语义**尚未数值验证**（头注释声称与 CPU 逐位一致，但没跑过）。
  对拍不过时，先怀疑 .cu 的 kernel 细节（尤其 `k_gdn_state`、`k_attn` 的
  GQA/strided 索引、`k_matvec_q8` 的块累加）。
- 8 张卡：固定 `CUDA_VISIBLE_DEVICES=0` 跑测试，避免多卡干扰。
- 别动 `tests/`、`ref/`；`zig build test` 必须保持绿。
- 提交习惯：每完成一个可验证里程碑就 git commit。
- **已修复 (2026-07-14, 32acdc3)**：长 prompt 乱码根因 = `k_gemm_q8` 的
  `B<=16` 分支（batched matvec）X 的 Q8_0 块指针以 `x0`（含 `lane*34`）
  为基准做 `i+32` 步进，多带了 lane 偏移：lane 1..31 在 `row_blocks>32`
  （cols>1024，即 MLP）时读到错误块/标度，只有 lane 0 对。该分支只被
  prefill 里 `B∈[2,16]` 的尾 chunk 命中（decode 走 B==1、整 chunk 走 B>16
  tiled），所以仅当 prompt 最后一个 chunk 恰好 ≤16 token 时乱码（517
  token=4×128+5 → "хар" EOS）。改为与 B==1 一致的绝对指针 `xb+(i+32)*34`。
  判别特征：乱码与否只取决于 `prompt_tokens % 128`（尾 chunk 大小），与
  文本内容无关。
- **注意**：`-n` 是 `max_seq`（总上下文上限，默认 4096），不是"生成 N 个
  token"。`-n < prompt_len` 时 decode 循环不跑，输出 0 token（是正常行为，
  不是 bug）。
- 工作树另有（已随本次提交一并入库）：`src/main.zig`、`src/main_cuda.zig`
  各加了一段 `TEMP(debug)` token-ID dump（`Q35_DUMP_IDS` 非空时打印
  `ID <n>`），供对拍用；`src/kernel_gdn.cu`、`src/kernel_matvec.cu` 是
  WIP 优化内核（自测用、尚未接进 engine，Makefile 不编它们）；
  `ref/c/qwen35_engine_trace.c` + `trace_first_token.txt` 是逐层数值 trace
  工具与其首 token 输出。
