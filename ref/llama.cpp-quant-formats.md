# llama.cpp 量化格式学习报告

> 读者：用 Zig 从零实现 GGUF 推理引擎的开发者（/mnt/llm-dev/qwen35）。
> 目标：理解 llama.cpp 支持的所有量化格式（ggml 数据类型）的结构、字节数、bpw、
> 以及「能产出 / 只能反量化 / 两者都有」。
> 源码根：`/mnt/strata/llama.cpp`（另有一份 `/mnt/strata/Strata/third_party/llama.cpp`）。
> 本报告每个断言均标注源码文件+行号/枚举值。所有字节数/bpw 均按 `sizeof(block_X)`
> 的 `static_assert` 实测推导，bpw = 块字节数 × 8 / 块元素数。

---

## 0. 三个正交的维度（先建立心智模型）

llama.cpp 的量化涉及**三个不同层面**，学习时务必区分，否则会混淆：

1. **ggml 数据类型**（`enum ggml_type`）—— 最底层，决定张量在内存里怎么存。
   定义：`ggml/include/ggml.h:390-433`（`GGML_TYPE_COUNT = 43`，含 8 个已移除的占位空洞）。
2. **llama 量化档位 / ftype**（`enum llama_ftype`）—— llama 模型导出时"主力量化"
   的语义档位，1 个档位映射到 1 个 ggml 类型。定义：`include/llama.h:119-162`。
3. **GGUF 文件里的 `ggml_type` 字段** —— 每个 tensor 在 GGUF 里存的就是 (1) 的枚举值。

对应关系：**llama_ftype（语义） → ggml_type（存储）**，映射在
`src/llama-quant.cpp:909` `llama_ftype_get_default_type()`。
GGUF 解析时你直接读的是 ggml_type（第 1 层），不需要关心 ftype。

---

## 1. ggml 数据类型全集（权威来源：`ggml.h` 枚举 + `ggml.c` type_traits）

### 1.1 枚举值（`ggml/include/ggml.h:390-433`）

| 值 | 名称 | 类别 |
|----|------|------|
| 0  | F32  | fp |
| 1  | F16  | fp |
| 2  | Q4_0 | 老式 4bit |
| 3  | Q4_1 | 老式 4bit |
| 4  | ~~Q4_2~~ 已移除（占位） | — |
| 5  | ~~Q4_3~~ 已移除（占位） | — |
| 6  | Q5_0 | 老式 5bit |
| 7  | Q5_1 | 老式 5bit |
| 8  | Q8_0 | 老式 8bit |
| 9  | Q8_1 | 老式 8bit |
| 10 | Q2_K | K-quant |
| 11 | Q3_K | K-quant |
| 12 | Q4_K | K-quant |
| 13 | Q5_K | K-quant |
| 14 | Q6_K | K-quant |
| 15 | Q8_K | K-quant |
| 16 | IQ2_XXS | i-quant |
| 17 | IQ2_XS  | i-quant |
| 18 | IQ3_XXS | i-quant |
| 19 | IQ1_S   | i-quant |
| 20 | IQ4_NL  | i-quant |
| 21 | IQ3_S   | i-quant |
| 22 | IQ2_S   | i-quant |
| 23 | IQ4_XS  | i-quant |
| 24 | I8   | 整数（非量化） |
| 25 | I16  | 整数（非量化） |
| 26 | I32  | 整数（非量化） |
| 27 | I64  | 整数（非量化） |
| 28 | F64  | fp |
| 29 | IQ1_M | i-quant |
| 30 | BF16 | fp |
| 31 | ~~Q4_0_4_4~~ 已移除（占位） | — |
| 32 | ~~Q4_0_4_8~~ 已移除（占位） | — |
| 33 | ~~Q4_0_8_8~~ 已移除（占位） | — |
| 34 | TQ1_0 | ternary（三值） |
| 35 | TQ2_0 | ternary（三值） |
| 36 | ~~IQ4_NL_4_4~~ 已移除（占位） | — |
| 37 | ~~IQ4_NL_4_8~~ 已移除（占位） | — |
| 38 | ~~IQ4_NL_8_8~~ 已移除（占位） | — |
| 39 | MXFP4  | 新式 fp4（MX 标准） |
| 40 | NVFP4  | 新式 fp4（NVIDIA） |
| 41 | Q1_0   | 老式 1bit |
| 42 | Q2_0   | 老式 2bit |
| 43 | GGML_TYPE_COUNT | — |

> 说明：枚举是稀疏的（含 8 个"已移除"占位）。**真实可用类型 = 35 个**（0..42 去掉 4,5,31,32,33,36,37,38）。
> 其中量化权重类型 29 个；fp/整数类型 6 个（F32/F16/BF16/F64/I8/I16/I32/I64 里 I 类为整数非量化）。

### 1.2 type_traits 表（权威：`ggml/src/ggml.c` 的 `type_traits[]`）

每个类型有一个 `ggml_type_traits` 条目，字段含义：
- `type_name`：字符串名（GGUF 里 tensor 的 `ggml_type` 对应的名字）。
- `blck_size`：一个 block 含多少个元素（fp/整数为 1）。
- `type_size`：`sizeof(block_X)`，一个 block 的字节数。
- `is_quantized`：是否量化类型。
- `to_float`：反量化函数指针（`dequantize_row_*`）。**为 NULL 表示通用反量化不支持**。
- `from_float_ref`：量化函数指针（`quantize_row_*_ref`）。**为 NULL 表示通用量化不支持**。

这是判断「能产出 / 能反量化」的**权威依据**（见 §4 的矩阵）。

---

## 2. block 结构与字节数（权威：`ggml/src/ggml-common.h` 的 struct + static_assert）

K-quant 家族统一块大小 `#define QK_K 256`（`ggml-common.h:89`）。
老式家族块大小多为 32（`QK4_0/QK5_0/QK8_0=32`）。

### 2.1 老式家族（block = 32 或 64 或 128 元素）

| 类型 | block 字段 | 块字节 | 块元素 | bpw |
|------|-----------|-------|-------|-----|
| Q1_0 | d(f16) + qs[16] (每 1bit) | 18 | 128 | 1.125 |
| Q2_0 | d(f16) + qs[16] (2bit)  | 18 | 64  | 2.25 |
| Q4_0 | d(f16) + qs[16] (nibble) | 18 | 32  | 4.5 |
| Q4_1 | d(f16) dmin(f16) + qs[16] | 20 | 32 | 5.0 |
| Q5_0 | d(f16) qh(u32) + qs[16]  | 22 | 32  | 5.5 |
| Q5_1 | d dmin qh(u32) + qs[16]   | 24 | 32  | 6.0 |
| Q8_0 | d(f16) + qs[32] (int8)    | 34 | 32  | 8.5 |
| Q8_1 | d(f16) s(f16) + qs[32]     | 36 | 32  | 9.0 |

结构体定义：`ggml-common.h:181-269`（含各 `static_assert(sizeof(block_X) == ...)`）。

### 2.2 K-quant 家族（block = 256 元素，两级 scale：super-block f16 + 每 32 元素一组 6bit scale）

| 类型 | block 字段 | 块字节 | bpw |
|------|-----------|-------|-----|
| Q2_K | scales[16] qs[64] {d,dmin}(f16) | 84  | 2.625 |
| Q3_K | hmask[32] qs[64] scales[12] {d} | 110 | 3.4375 |
| Q4_K | scales[12] qs[128] {d,dmin}(f16) | 144 | 4.5 |
| Q5_K | scales[12] qh[32] qs[128] {d,dmin} | 176 | 5.5 |
| Q6_K | ql[128] qh[64] scales[16] {d}    | 210 | 6.5625 |
| Q8_K | d(f32) qs[256](int8) bsums[16](i16) | 292 | 9.125 |

结构体定义：`ggml-common.h:276-376`。

**两级 scale 机制**（K-quant 核心）：
- 一个 256 元素的 super-block 用 1 个 f16 `d`（和 `dmin`）做全局缩放/偏移；
- 内部再切 8 组，每组 32 元素，每组再有一个 6bit 量化 scale/min，
  打包在 `scales[12]` 里（`K_SCALE_SIZE = 12`，`ggml-common.h:90`）。
- 反量化时 `y = d * scale_grp * q - dmin * min_grp`（两级放大）。

### 2.3 i-quant 家族（IQ = integer quant，codebook/网格量化，block 多为 256）

i-quant 用**预计算的 128 点码本网格**（grid）来存 4bit 索引，配合低比特 scale，
比 K-quant 更省 bit。结构定义：`ggml-common.h:378-460`。

| 类型 | block 字段 | 块字节 | bpw |
|------|-----------|-------|-----|
| IQ2_XXS | d(f16) + qs[32](u16) | 66  | 2.0625 |
| IQ2_XS  | d(f16) + qs[32](u16) + scales[8] | 74  | 2.3125 |
| IQ2_S   | d(f16) + qs[64] + qh[8] + scales[8] | 82 | 2.5625 |
| IQ3_XXS | d(f16) + qs[96] | 98  | 3.0625 |
| IQ3_S   | d(f16) + qs[64] + qh[8] + signs[32] + scales[4] | 110 | 3.4375 |
| IQ1_S   | d(f16) + qs[32] + qh[8](u16) | 50 | 1.5625 |
| IQ1_M   | qs[32] + qh[8] + scales[8]（无 d） | 48 | 1.5 |
| IQ4_NL  | d(f16) + qs[16]（块=32，非线性） | 18 | 4.5 |
| IQ4_XS  | d(f16) + scales_h(u16) + scales_l[4] + qs[128] | 136 | 4.25 |

### 2.4 ternary 家族（TQ，三值，block=256）

| 类型 | block 字段 | 块字节 | bpw |
|------|-----------|-------|-----|
| TQ1_0 | d(f16) + qs[48]（每 5 元素 1bit，243<256） | 50 | 1.5625 |
| TQ2_0 | d(f16) + qs[64]（2bit） | 66 | 2.0625 |

结构：`ggml-common.h:276-288`。

### 2.5 新式 fp4 家族（MX/NV 标准，2024+）

| 类型 | block 字段 | 块字节 | 块元素 | bpw |
|------|-----------|-------|-------|-----|
| MXFP4  | d(u8 E8M0 共享指数) + qs[16](E4M3 4bit) | 17 | 32 | 4.25 |
| NVFP4  | d[4](UE4M3 子块 scale) + qs[32](E2M1 4bit) | 36 | 64 | 4.5 |

结构：`ggml-common.h:214-227`。

### 2.6 fp / 整数（block_size=1，type_size=元素字节）

F32(4B) / F16(2B) / BF16(2B) / F64(8B) / I8(1B) / I16(2B) / I32(4B) / I64(8B)。

---

## 3. 反量化（dequantize）数学

### 3.1 典型 K-quant：q4_K（`ggml-quants.c:1458` 附近，`dequantize_row_q4_K`）

```c
const float d   = fp16(x.d);      // super-block scale
const float min = fp16(x.dmin);   // super-block offset
for (每个 32 元素组) {
    get_scale_min_k4(is, x.scales, &sc, &m);  // 从 scales[12] 解 6bit scale sc 与 min m
    const float d1 = d * sc;  const float m1 = min * m;
    for (l=0;l<32;++l) *y++ = d1 * (q[l] & 0xF)  - m1;  // 低 4bit
    for (l=0;l<32;++l) *y++ = d2 * (q[l] >> 4)   - m2;  // 高 4bit
}
```
要点：**两级 scale**（super f16 `d` × 组 6bit `sc`），`qs` 是 4bit 有符号偏移。

### 3.2 典型 i-quant：iq4_xs（codebook 网格量化）

- `qs[128]` 存 4bit 的**码本索引**，指向预计算的 128 点网格（grid）。
- `scales_h(u16)` + `scales_l[4]` 存 4 组 4bit scale，配合 `d(f16)` 两级放大。
- 反量化：`y = d * scale_grp * grid[qs 索引]`。codebook 在编译期固定（`ggml-quants.c` 的 `iq4_nl`/`iq4_xs` grid 常量表）。

> i-quant 的本质：不用线性 `scale*q+min`，而是用**预计算网格**逼近权重分布，
> 在同样 bit 数下精度更高（尤其 2~3bit）。这就是 IQ 系列比老式 qN 更"聪明"的地方。

### 3.3 反量化函数全集（`ggml-quants.c`，`void dequantize_row_*`）

q1_0, q2_0, q4_0, q4_1, q5_0, q5_1, q8_0, q8_1, q2_K, q3_K, q4_K, q5_K, q6_K,
**q8_K**, iq2_xxs, iq2_xs, iq3_xxs, iq3_s, iq2_s, iq1_s, iq1_m, iq4_nl, iq4_xs,
mxfp4, nvfp4, tq1_0, tq2_0。

> 注意：**Q8_K 有 `dequantize_row_q8_K`（`ggml-quants.c:2808`）但 `type_traits` 里
> `to_float = NULL`**（`ggml.c:252-257`）。即 Q8_K 不接通用反量化路径，
> 而是由 CPU 后端专用 kernel 直接反量化（`ggml-cpu/quants.c`）。
> 对推理引擎实现者：**Q8_K 能推理，但不能走通用 dequant 接口**。

---

## 4. 「能产出 / 能反量化 / 两者」权威矩阵

产出权威：`ggml.c:8080` `ggml_quantize_chunk()` 的 switch（llama-quant 实际调用它），
以及 `ggml.c` type_traits 的 `from_float_ref` 非空。
反量化权威：type_traits 的 `to_float` 非空（或 CPU 后端专用 kernel）。

| 类型 | 产出(quant) | 反量化(dequant) | 说明 |
|------|:---:|:---:|------|
| Q1_0 | ✅ | ✅ | |
| Q2_0 | ✅ | ✅ | |
| Q4_0 | ✅ | ✅ | |
| Q4_1 | ✅ | ✅ | |
| Q5_0 | ✅ | ✅ | |
| Q5_1 | ✅ | ✅ | |
| Q8_0 | ✅ | ✅ | |
| Q8_1 | ❌ | ✅ | 只有反量化，无产出路径（历史遗留） |
| Q2_K | ✅ | ✅ | |
| Q3_K | ✅ | ✅ | |
| Q4_K | ✅ | ✅ | 最常用档 |
| Q5_K | ✅ | ✅ | |
| Q6_K | ✅ | ✅ | |
| Q8_K | ✅ | ⚠️ CPU-only | 产出有；dequant 仅 CPU 专用 kernel（type_traits.to_float=NULL） |
| IQ2_XXS | ✅ | ✅ | |
| IQ2_XS  | ✅ | ✅ | |
| IQ3_XXS | ✅ | ✅ | |
| IQ3_S   | ✅ | ✅ | |
| IQ2_S   | ✅ | ✅ | |
| IQ1_S   | ✅ | ✅ | |
| IQ1_M   | ✅ | ✅ | |
| IQ4_NL  | ✅ | ✅ | |
| IQ4_XS  | ✅ | ✅ | |
| TQ1_0   | ✅ | ✅ | |
| TQ2_0   | ✅ | ✅ | |
| MXFP4   | ✅ | ✅ | |
| NVFP4   | ✅ | ✅ | |
| F16    | ✅ | ✅ | fp（非量化权重） |
| BF16   | ✅ | ✅ | fp |
| F32    | ✅ | ✅ | fp |
| I8/I16/I32/I64/F64 | — | — | 整数/浮点，非量化权重路径 |

**关键结论**：
- 现代 llama.cpp 里**几乎所有量化类型既能产出又能反量化**（Q8_1 是仅反量化特例）。
- 唯一"能推理但不能走通用 dequant"的是 **Q8_K**（CPU 专用 kernel）。
- 已移除类型（Q4_2/Q4_3/Q4_0_4_4 等 8 个占位）：**既不能产出也不能反量化**，
  遇到应报错/回退到 Q4_0。

---

## 5. llama 量化档位（ftype）→ ggml_type 映射

权威：`include/llama.h:119-162`（枚举）+ `src/llama-quant.cpp:909`
`llama_ftype_get_default_type()`（映射）+ `src/llama-model-loader.cpp:38-73`（显示名）。

| ftype | 显示名 | → ggml_type | bpw |
|-------|--------|------------|-----|
| ALL_F32 | all F32 | F32 | 32 |
| MOSTLY_F16 | F16 | F16 | 16 |
| MOSTLY_BF16 | BF16 | BF16 | 16 |
| MOSTLY_Q4_0 | Q4_0 | Q4_0 | 4.5 |
| MOSTLY_Q4_1 | Q4_1 | Q4_1 | 5.0 |
| MOSTLY_Q5_0 | Q5_0 | Q5_0 | 5.5 |
| MOSTLY_Q5_1 | Q5_1 | Q5_1 | 6.0 |
| MOSTLY_Q8_0 | Q8_0 | Q8_0 | 8.5 |
| MOSTLY_Q1_0 | Q1_0 | Q1_0 | 1.125 |
| MOSTLY_Q2_0 | Q2_0 | Q2_0 | 2.25 |
| MOSTLY_Q2_K | Q2_K - Medium | Q2_K | 2.625 |
| MOSTLY_Q2_K_S | Q2_K - Small | Q2_K | 2.625 |
| MOSTLY_Q3_K_S/M/L | Q3_K S/M/L | Q3_K | 3.4375 |
| MOSTLY_Q4_K_S/M | Q4_K S/M | Q4_K | 4.5 |
| MOSTLY_Q5_K_S/M | Q5_K S/M | Q5_K | 5.5 |
| MOSTLY_Q6_K | Q6_K | Q6_K | 6.5625 |
| MOSTLY_IQ2_XXS | IQ2_XXS - 2.0625 | IQ2_XXS | 2.0625 |
| MOSTLY_IQ2_XS  | IQ2_XS - 2.3125 | IQ2_XS | 2.3125 |
| MOSTLY_IQ2_S   | IQ2_S - 2.5 | → **IQ2_XS** | 2.3125 |
| MOSTLY_IQ2_M   | IQ2_M - 2.7 | → **IQ2_S** | 2.5625 |
| MOSTLY_IQ3_XS  | IQ3_XS - 3.3 | → **IQ3_S** | 3.4375 |
| MOSTLY_IQ3_XXS | IQ3_XXS - 3.0625 | IQ3_XXS | 3.0625 |
| MOSTLY_IQ3_S   | IQ3_S - 3.4375 | IQ3_S | 3.4375 |
| MOSTLY_IQ3_M   | IQ3_S mix - 3.66 | → **IQ3_S** | 3.4375 |
| MOSTLY_IQ1_S   | IQ1_S - 1.5625 | IQ1_S | 1.5625 |
| MOSTLY_IQ1_M   | IQ1_M - 1.75 | IQ1_M | 1.5 |
| MOSTLY_IQ4_NL  | IQ4_NL - 4.5 | IQ4_NL | 4.5 |
| MOSTLY_IQ4_XS  | IQ4_XS - 4.25 | IQ4_XS | 4.25 |
| MOSTLY_TQ1_0   | TQ1_0 - 1.69 ternary | TQ1_0 | 1.5625 |
| MOSTLY_TQ2_0   | TQ2_0 - 2.06 ternary | TQ2_0 | 2.0625 |
| MOSTLY_MXFP4_MOE | MXFP4 MoE | MXFP4 | 4.25 |
| MOSTLY_NVFP4   | NVFP4 | NVFP4 | 4.5 |

> **S/M/L 命名陷阱**（实现者必看）：ftype 的 "S/M/L" 是**档位粗细**，不等于 ggml 类型名。
> 例：`MOSTLY_IQ2_S`→IQ2_XS、`MOSTLY_IQ2_M`→IQ2_S、`MOSTLY_IQ3_XS`→IQ3_S、
> `MOSTLY_IQ3_M`→IQ3_S。即"档位 S 实际产出 IQ2_XS"。解析 GGUF 时你直接读
> ggml_type 不受影响，但若从 ftype 推导需注意此映射。
> ftype=1024 为 `LLAMA_FTYPE_GUESSED`（文件未指定，需从 GGUF tensor 推断）。

---

## 6. 文档与代码一致性核对

- `src/llama-model-loader.cpp:38-73` 的显示名与 `llama.h` ftype 注释**一致**（含 bpw）。
- 显示名里的 bpw（如 IQ2_XXS 2.0625、IQ4_XS 4.25、TQ1_0 1.69、TQ2_0 2.06）与
  §2 按 `sizeof` 推导的 bpw **完全吻合**（TQ1_0 显示 1.69 是 1.5625 的"营销取整"，
  实际 1.5625）。
- `llama-quant.cpp:917-918` 新增 Q1_0/Q2_0（老式最低档），与 ggml.h 枚举 41/42 对应，一致。
- 未发现文档与代码不一致之处（bpw 均为近似显示，非错误）。

---

## 7. 给 GGUF 引擎实现者的要点清单

1. **GGUF 里 tensor 的 `ggml_type` 字段 = 第 1 层 ggml 枚举值**（0..42），
   按本报告 §1.1 对照即可识别。
2. **读 tensor 数据时**：按 `type_traits[type].blck_size` 算元素数，
   按 `type_size` 算字节数；量化 tensor 的元素数必为 `blck_size` 整数倍。
3. **反量化**：优先走 `type_traits.to_float`（`dequantize_row_*`）；
   **Q8_K 例外**——`to_float=NULL`，需自行实现或走 CPU 后端。
4. **已移除类型**（4,5,31,32,33,36,37,38）：遇到应报错/回退 Q4_0，不要尝试反量化。
5. **产出/量化**（若实现 llama-quant）：调 `ggml_quantize_chunk(type, ...)`，
   它对 25 种类型有完整 switch（§4 矩阵 ✅ 行）。
6. **K-quant 两级 scale、i-quant codebook 网格**是两大核心机制，
   分别见 §3.1 / §3.2。
7. **MXFP4/NVFP4** 是新标准 fp4（E4M3/E2M1 + 共享指数），
   结构见 §2.5，bpw 4.25/4.5，比 Q4_K(4.5) 更省且是硬件友好格式。

---

## 附：核对过的源码位置

- 枚举全集：`ggml/include/ggml.h:390-433`
- type_traits 表：`ggml/src/ggml.c:680-321`（`type_traits[]`）
- block 结构 + 字节 static_assert：`ggml/src/ggml-common.h:89-460`
- 反量化函数：`ggml/src/ggml-quants.c`（`dequantize_row_*`，含 q8_K:2808）
- 产出入口：`ggml/src/ggml.c:8080`（`ggml_quantize_chunk`）
- ftype 枚举：`include/llama.h:119-162`
- ftype→ggml 映射：`src/llama-quant.cpp:909-946`
- ftype 显示名：`src/llama-model-loader.cpp:38-73`
- q4_K 反量化数学：`ggml-quants.c:1458` 附近
- worker 中间产物（完整 type_traits 表）：`/tmp/traits.txt`（321 行）
