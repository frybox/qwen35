# docs 索引

引擎本身在 `../c/`，这里只放文档与定位公式时用过的一次性实验脚本。

| 文件 | 内容 | 什么时候看 |
| --- | --- | --- |
| [CODE_INDEX.md](CODE_INDEX.md) | `qwen35_engine.c` / `safetensors.c` / `tokenizer.c` 的函数与行号索引，按源码顺序，附抄写注意事项 | 重抄代码前先扫一遍，知道每段在干什么 |
| [SPEC.md](SPEC.md) | 与 HF `modeling_qwen3_5.py` 逐行对齐的规格（张量名、公式、行号引用） | 想确认某个公式为什么这么写时 |
| [RESOLUTION.md](RESOLUTION.md) | 调试最终结论：与 HF 数值等价（logits max err 3.96e-05、argmax 一致、逐 token 相同） | 关心正确性证据时 |
| [GDN_IMPL.md](GDN_IMPL.md) | GDN（gated delta net）线性注意力的实现细节梳理 | 抄 `forward_linear_attn` 前 |
| [model_tensors.txt](model_tensors.txt) | 全部 320 个张量的名字/形状/dtype 清单 | 调 `load_weights` 的张量名映射时 |
| [spec_experiments/](spec_experiments/) | 定位 decay / delta 顺序 / RoPE / 全注意力公式时跑的 PyTorch 实验脚本与输出（`*.py` + `*.out`） | 想复核某个公式的实测依据时 |

历史调试代码（大量 test_*/debug_*/check_* 与中间报告）已删除，打包在 `../archive/`。
