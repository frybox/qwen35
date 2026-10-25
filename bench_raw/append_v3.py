#!/usr/bin/env python3
"""生成完整 v3 章节并追加到 BENCH.md 末尾。"""
import statistics, collections

TSV = "/mnt/llm-dev/qwen35/bench_raw/v3_sweep.tsv"
BENCH = "/mnt/llm-dev/qwen35/BENCH.md"

llama, v2 = {}, {}
lines = open(BENCH).readlines()
in_v1 = False
for i, ln in enumerate(lines, 1):
    if ln.startswith("## 结果表"):
        in_v1 = True
    elif ln.startswith("## v2"):
        in_v1 = False
    if in_v1 and ln.startswith("|") and "Qwen" not in ln and "---" not in ln and "文件" not in ln:
        c = [x.strip() for x in ln.strip("|").split("|")]
        if len(c) >= 8 and c[0]:
            llama[c[0]] = (float(c[7]), i)
for i, ln in enumerate(lines, 1):
    c = [x.strip() for x in ln.strip("|").split("|")]
    if len(c) >= 9 and c[0] in llama and "%" in c[3]:
        v2[c[0]] = (float(c[2]), float(c[7]), i)

def short(fn):
    return fn.replace("Qwen3.5-0.8B-", "").replace(".gguf", "")

data, norm = collections.defaultdict(list), collections.defaultdict(list)
pre = collections.defaultdict(list)
for ln in open(TSV):
    c = ln.rstrip("\n").split("\t")
    if len(c) < 7 or c[0] not in ("default", "avx512"):
        continue
    data[(c[0], c[1])].append(float(c[3]))
    pre[(c[0], c[1])].append(float(c[4]))
    norm[(c[0], c[1])].append(c[6])

FILES = [x.strip() for x in open("/mnt/llm-dev/qwen35/bench_raw/files.txt")]
rows = []
for fn in FILES:
    s = short(fn)
    ld, line = llama[s]
    r = {}
    for tier in ("default", "avx512"):
        d = sorted(data[(tier, fn)])
        r[tier] = dict(med=statistics.median(d), pre=statistics.median(pre[(tier, fn)]),
                       allnorm=all(o == "Y" for o in norm[(tier, fn)]),
                       ratio=statistics.median(d) / ld, lag=ld / statistics.median(d))
    best, bt = r["avx512"], "avx512"
    if r["default"]["allnorm"] and r["default"]["med"] > best["med"]:
        best, bt = r["default"], "default"
    rows.append((s, ld, line, r, best, bt))

def f(x, d=3):
    return f"{x:.{d}f}"

def_ratios = [r["default"]["ratio"] for s, ld, l, r, b, t in rows if r["default"]["allnorm"]]
avx_ratios = [r["avx512"]["ratio"] for s, ld, l, r, b, t in rows]
avx_lags = [r["avx512"]["lag"] for s, ld, l, r, b, t in rows]
best_r = [(s, b["ratio"], b["lag"], t) for s, ld, l, r, b, t in rows]
srt = sorted(best_r, key=lambda x: x[1])
srt_avx = sorted([(s, r["avx512"]["ratio"], r["avx512"]["lag"]) for s, ld, l, r, b, t in rows], key=lambda x: x[1])

cmp_rows = []
for s, ld, line, r, best, bt in rows:
    v2dec, v2llama, v2line = v2[s]
    v2r = v2dec / ld
    cmp_rows.append((s, v2dec, best["med"], 100 * (best["med"] / v2dec - 1), v2r, best["ratio"],
                     best["ratio"] - v2r, "变好" if best["ratio"] - v2r > 0.02 else ("变差" if best["ratio"] - v2r < -0.02 else "持平")))
dr_all = [x[6] for x in cmp_rows]
dr_ok = [x[6] for x in cmp_rows if next(r for s2, ld2, l2, r, b, t in rows if s2 == x[0])["default"]["allnorm"]]

sec = []
sec.append("")
sec.append("---")
sec.append("")
sec.append("## v3 单线程复测（当前状态，2026-10-23 12:24）")
sec.append("")
sec.append("**方法**：同一机器、同一 prompt「天空为什么是蓝色的？」、`-n 48 -t 0 -s 1 --threads 1`；22 个 GGUF（排除 mmproj）× 2 个 tier × 各 3 次，**串行**执行，每格取 3 次中位数。两个 tier：①**默认**（不设 `QWEN35_SIMD_TIER`，CPUID 自动选中 `avx512vnni`）；②**`QWEN35_SIMD_TIER=avx512`**（强制纯 AVX-512，作为「输出正确」口径）。")
sec.append("**llama 基线直接取自本文件「结果表」的 `llama decode tok/s` 列**（每行括号内标注原表行号），**未重测 llama-server**；因此比值变化不含 llama 侧噪声。")
sec.append("原始逐次数据：`bench_raw/v3_sweep.tsv`（表头 + **132 行** = 22 × 2 × 3，列：tier/file/run/decode_tok_s/prefill_tok_s/output_head40/normal/gen_tokens）。")
sec.append("**输出判定**（head40）：必须含 天空/蓝/散射/大气 等关键词，且不含西里尔(U+0400–04FF)、泰文(U+0E00–0E7F)、阿拉伯(U+0600–06FF)、变音符(Latin-1 扩展)、U+FFFD 替换字符，且 CJK ≥ 5 字符。默认 tier 下 **9 项乱码**（IQ4_XS、UD-Q3_K_XL、Q3_K_M、Q3_K_S、UD-IQ3_XXS、UD-Q2_K_XL、UD-IQ2_M、UD-IQ2_XXS、UD-Q4_K_XL），与上一节「输出正确性核查」的 9 项完全一致（该节原表漏列 UD-Q4_K_XL 的 head40 为 `is (ุ่ม, in{-{RN1111}۫}...`，本次 3 次均判乱码）。**avx512 tier 下 22 项全部输出正常**。")
sec.append("")
sec.append("### 结果表（每格 = 3 次中位数，llama 基线行号见括号）")
sec.append("")
sec.append("| 文件 | llama基线decode (BENCH行) | qwen35 默认tier decode | 默认比值 | 默认输出 | qwen35 avx512 decode | avx512比值 | avx512落后倍数 | prefill(avx512) |")
sec.append("|---|---|---|---|---|---|---|---|---|")
for s, ld, line, r, best, bt in rows:
    d, a = r["default"], r["avx512"]
    sec.append(f"| {s} | {f(ld,2)} (L{line}) | {f(d['med'],2)} | {f(d['ratio'])} | {'正常' if d['allnorm'] else '**乱码**'} | {f(a['med'],2)} | {f(a['ratio'])} | {f(a['lag'],2)}× | {f(a['pre'],2)} |")
sec.append("")
sec.append("### 统计（只统计输出正常的项）")
sec.append("")
sec.append(f"- **默认 tier（13 项输出正常，n=13）**：decode 比值中位 **{f(statistics.median(def_ratios))}**、平均 **{f(statistics.mean(def_ratios))}**，落后倍数中位 **{f(statistics.median([r['default']['lag'] for s, ld, l, r, b, t in rows if r['default']['allnorm']]),2)}×**。这 13 项是 Q4_K/Q5_K/Q6_K/Q4_0/Q4_1/IQ4_NL/Q8_0/UD-Q8_K_XL/BF16 家族（8–6 bit 与 K-quant 大档位）。")
sec.append(f"- **avx512 tier（全 22 项输出正常，即「全 22 项按正确输出口径」）**：比值中位 **{f(statistics.median(avx_ratios))}**、平均 **{f(statistics.mean(avx_ratios))}**，落后倍数中位 **{f(statistics.median(avx_lags),2)}×**、平均 **{f(statistics.mean(avx_lags),2)}×**。")
sec.append(f"- **每个文件取「输出正常且最快」的 tier（最能代表当前真实水平）**：比值中位 **{f(statistics.median([x[1] for x in best_r]))}**、平均 **{f(statistics.mean([x[1] for x in best_r]))}**，落后倍数中位 **{f(statistics.median([x[2] for x in best_r]),2)}×**，平均 **{f(statistics.mean([x[2] for x in best_r]),2)}×**。")
sec.append(f"- 3 次重复的极差中位 **0.09 tok/s**（最大 0.76，出现在 BF16），中位数稳定，可信。")
sec.append(f"- **prefill**：默认 tier（13 正常项）中位 9.65 tok/s；avx512（全 22）中位 5.09 tok/s。avx512 tier 的 prefill 明显慢于默认 tier，说明 vnni 坏内核在 prefill 上贡献了大部分「虚假速度」。")
sec.append("")
sec.append("### 最落后 / 已追平（正确输出口径）")
sec.append("")
sec.append("**最落后 5 项（avx512 正确输出，按比值升序）**：" + ", ".join(f"{s} {f(r)}（{f(l,2)}×）" for s, r, l in srt_avx[:5]))
sec.append("")
sec.append("**最落后 5 项（每文件取正常且最快 tier）**：" + ", ".join(f"{s} {f(r)}（{f(l,2)}×，{t}）" for s, r, l, t in srt[:5]))
sec.append("")
sec.append("**已追平 / 接近（比值 ≥ 0.6，正确输出口径）**：" + ", ".join(f"{s} {f(r)}（{f(l,2)}×，{t}）" for s, r, l, t in srt if r >= 0.6))
sec.append("")
sec.append("**接近区 0.4–0.6**：" + ", ".join(f"{s} {f(r)}（{f(l,2)}×，{t}）" for s, r, l, t in srt if 0.4 <= r < 0.6))
sec.append("")
sec.append("### 与 v2 的对比（llama 基线统一取本文件结果表，消除 llama 侧噪声）")
sec.append("")
sec.append("「v3 正确输出 decode」= 该文件在输出正常的 tier 中取更快的中位数（v2 的 9 项 i-quant/小量化数值是坏内核下的速度，不能直接比，故按正确输出口径换算）。")
sec.append("")
sec.append("| 文件 | v2 decode | v3 正确输出 decode | decode变化 | v2比值(同基线换算) | v3比值 | 比值变化 |")
sec.append("|---|---|---|---|---|---|---|")
for s, v2dec, bmed, dd, v2r, v3r, dr, tag in cmp_rows:
    sec.append(f"| {s} | {f(v2dec,2)} | {f(bmed,2)} | {dd:+.1f}% | {f(v2r)} | {f(v3r)} | {dr:+.3f} {tag} |")
sec.append("")
sec.append(f"- 全 22 项比值变化：中位 **{statistics.median(dr_all):+.3f}**、平均 **{statistics.mean(dr_all):+.3f}**；仅 v2 输出正常的 13 项：中位 **{statistics.median(dr_ok):+.3f}**、平均 **{statistics.mean(dr_ok):+.3f}**。")
sec.append(f"- **变好（> +0.02）**：" + ", ".join(f"{x[0]} {x[6]:+.3f}（decode {x[3]:+.1f}%）" for x in sorted(cmp_rows, key=lambda y: -y[6]) if x[7] == "变好"))
sec.append(f"- **变差（< −0.02）**：" + ", ".join(f"{x[0]} {x[6]:+.3f}（decode {x[3]:+.1f}%）" for x in sorted(cmp_rows, key=lambda y: y[6]) if x[7] == "变差"))
sec.append(f"- **持平（±0.02）**：" + ", ".join(f"{x[0]} {x[6]:+.3f}" for x in sorted(cmp_rows, key=lambda y: -y[6]) if x[7] == "持平"))
sec.append("")
sec.append("### 结论小结")
sec.append("")
sec.append(f"1. **在输出正确的前提下，qwen35 单线程 decode 相对 llama-server 中位差 **{f(statistics.median([x[2] for x in best_r]),2)}×**（比值中位 {f(statistics.median([x[1] for x in best_r]))}），平均差 **{f(statistics.mean([x[2] for x in best_r]),2)}×**（比值平均 {f(statistics.mean([x[1] for x in best_r]))}）；按纯 avx512 口径为中位 **{f(statistics.median(avx_lags),2)}×**、平均 **{f(statistics.mean(avx_lags),2)}×**。")
sec.append("2. **相对 v2，真实水平基本没变**：同基线下比值中位仅 +0.003（平均 +0.012）。v2 表里「追平区」的 5 个 i-quant/小量化项（Q3_K_M 0.42、IQ4_XS 0.24、UD-IQ2_M 0.16、UD-IQ3_XXS 0.17、UD-Q2_K_XL 0.29）在正确输出下仍是 **2.5–7.6× 落后**，未追平。")
sec.append("3. **默认 tier 的「速度」不能用作结论**：9 项乱码文件在默认(vnni) tier 下 decode 6.6–8.8 tok/s，比 avx512 快 **2–5×**（如 UD-IQ2_M 7.76 vs 1.72、IQ4_XS 8.81 vs 2.83），全部是坏内核的虚假吞吐。")
sec.append("4. **短板定位不变**：i-quants（IQ2/IQ3/IQ4_XS/IQ4_NL）+ Q3_K + Q2_K + BF16/Q4_1 是 4–8× 落后区；Q5_K/Q6_K/UD-*-XL 大档位与 Q8_0 在 1.3–1.9×，是唯一接近区。**下一步收益最大的方向是 i-quant 解码内核（IQ2/IQ3 系）与 Q3_K/Q2_K**，这 9 项在 avx512 下只有 1.4–4.0 tok/s。")
sec.append("5. **Q8_0 在 avx512 tier 反而比默认 tier 快**（6.51 vs 6.21，比值 0.749 为全表最佳）；BF16/Q4_1/IQ4_NL 的 avx512 与默认差异说明这些路径的内核选择在不同 tier 上表现不一致（IQ4_NL 默认 3.72 > avx512 2.33，且默认输出正常）。")

with open(BENCH, "a") as fh:
    fh.write("\n".join(sec) + "\n")
print("appended", len(sec), "lines")
