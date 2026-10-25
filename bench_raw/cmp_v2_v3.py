#!/usr/bin/env python3
"""v3 与 v2 对比：统一用 BENCH.md 的 llama 基线 decode（v2 的比值是相对 llama 本轮实测，含噪声）。"""
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
for ln in open(TSV):
    c = ln.rstrip("\n").split("\t")
    if len(c) < 7 or c[0] not in ("default", "avx512"):
        continue
    data[(c[0], c[1])].append(float(c[3]))
    norm[(c[0], c[1])].append(c[6])

FILES = [x.strip() for x in open("/mnt/llm-dev/qwen35/bench_raw/files.txt")]
rows = []
for fn in FILES:
    s = short(fn)
    ld = llama[s][0]
    r = {}
    for tier in ("default", "avx512"):
        d = sorted(data[(tier, fn)])
        r[tier] = dict(med=statistics.median(d),
                       allnorm=all(o == "Y" for o in norm[(tier, fn)]),
                       ratio=statistics.median(d) / ld)
    best = r["avx512"]
    if r["default"]["allnorm"] and r["default"]["med"] > best["med"]:
        best = r["default"]
    rows.append((s, ld, r, best))

print("| 文件 | v2 qwen35 decode | v3 正确输出decode(取最快正常tier) | decode变化 | v2比值(同基线换算) | v3比值 | 比值变化 |")
print("|---|---|---|---|---|---|---|")
deltas = []
comparable = []
for s, ld, r, best in rows:
    v2dec, v2llama, v2line = v2[s]
    v2r_norm = v2dec / ld            # v2 decode 换算到同一 llama 基线
    v3r = best["ratio"]
    dr = v3r - v2r_norm
    dd = 100 * (best["med"] / v2dec - 1)
    tag = "变好" if dr > 0.02 else ("变差" if dr < -0.02 else "持平")
    deltas.append((s, dr, dd, tag))
    if r["default"]["allnorm"]:      # v2 时输出正常 -> 可比
        comparable.append(dr)
    print(f"| {s} | {v2dec:.2f} | {best['med']:.2f} | {dd:+.1f}% | {v2r_norm:.3f} | {v3r:.3f} | {dr:+.3f} {tag} |")

print()
print("仅 v2 输出正常项(13 项) 比值变化：中位", f"{statistics.median(comparable):+.3f}",
      " 平均", f"{statistics.mean(comparable):+.3f}")
print("全部 22 项比值变化：中位", f"{statistics.median([d[1] for d in deltas]):+.3f}",
      " 平均", f"{statistics.mean([d[1] for d in deltas]):+.3f}")
print("变好:", ", ".join(f"{s}({d:+.3f})" for s, d, dd, t in sorted(deltas, key=lambda x: -x[1]) if t == "变好"))
print("变差:", ", ".join(f"{s}({d:+.3f})" for s, d, dd, t in sorted(deltas, key=lambda x: x[1]) if t == "变差"))
print("持平:", ", ".join(f"{s}({d:+.3f})" for s, d, dd, t in sorted(deltas, key=lambda x: -x[1]) if t == "持平"))
