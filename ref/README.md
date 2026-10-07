# Qwen3.5-0.8B CPU 推理引擎

从零用 C 实现的 Qwen3.5-0.8B 推理引擎（单文件核心 + safetensors 读取 + BPE tokenizer），
不依赖任何框架，直接 mmap 官方 safetensors 权重。

## 构建与运行

```bash
cd c
make                       # 生成 qwen35_engine
./qwen35_engine [权重路径] [vocab.json 路径]
# 或
make run                   # 使用 MODEL_DIR（默认 /mnt/Qwen3.5-0.8B）
```

默认参数：
- 权重：`/mnt/Qwen3.5-0.8B/model.safetensors-00001-of-00001.safetensors`
- 词表：`/mnt/Qwen3.5-0.8B/vocab.json`

可选环境变量：
- `QWEN35_PROMPT_IDS=248045,846,...` 覆盖默认 prompt（用于回归测试）
- `QWEN35_DUMP_LOGITS=/path/out.bin` 导出最后一个 prompt 位置的 248320 个 fp32 logits（与 HF 参考对比用）

实测输出（本机，纯 CPU，单线程）：

```
Output: 中国的首都是**北京**。
Prefill: 17 tokens in 10.640 s  (1.60 tok/s)
Generate: 61 tokens in 38.282 s  (1.59 tok/s)
Stop reason: EOS
```

## 模型要点

| 项 | 值 |
| --- | --- |
| 层数 | 24（层型 `LLLF` 循环：18 层 GDN 线性注意力 + 6 层全注意力） |
| dim / ffn | 1024 / 3584 |
| heads | 8 Q heads，2 KV heads，head_dim 256（GQA） |
| 词表 | 248320 |
| 张量前缀 | `model.language_model.`（视觉 / mtp 部分忽略） |

## 目录结构

```
qwen35/
├── c/                        # 引擎源码（只保留推理所需）
│   ├── qwen35_engine.c       # 主实现：权重结构、GDN、全注意力、forward、采样、main
│   ├── safetensors.c/.h      # safetensors 解析 + mmap + BF16→FP32
│   ├── tokenizer.c/.h        # BPE tokenizer（含 nfc_table.h 归一化表）
│   ├── nfc_table.h           # Unicode NFC 归一化查找表（生成物）
│   └── Makefile
├── docs/                     # 文档
│   ├── README.md             # 文档索引
│   ├── CODE_INDEX.md         # 源码学习导读（函数/行号索引，按源码顺序）
│   ├── SPEC.md               # 与 HF 参考实现逐行对齐的修复规格
│   ├── RESOLUTION.md         # 调试最终结论：与 HF 数值等价（logits 误差 3.96e-05）
│   ├── GDN_IMPL.md           # GDN（gated delta net）线性注意力实现细节
│   ├── model_tensors.txt     # 全部 320 个张量的形状/类型清单
│   └── spec_experiments/     # 定位 GDN/RoPE 公式时跑的 PyTorch 实验脚本与输出
└── archive/                  # 历史调试产物打包（不含源码删除风险，可解压回查）
```

## 正确性

与 HuggingFace 参考实现数值等价（详见 `docs/RESOLUTION.md`）：

- logits 最大绝对误差 `3.96e-05`（fp32 舍入量级）
- cosine 相似度 `1.000000`，argmax 一致
- 贪心生成逐 token 与参考完全相同

## 待办 / 后续方向

- [ ] 性能：1.6 tok/s（纯 CPU、单线程、无 SIMD）→ 加多线程 / AVX2 / 量化
- [ ] prompt 目前硬编码 + 环境变量覆盖，可接 tokenizer 从文本编码
- [ ] 采样仅 argmax，可扩展 top-k / top-p / 温度
