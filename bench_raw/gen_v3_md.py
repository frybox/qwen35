#!/usr/bin/env python3
"""生成 BENCH.md 的 v3 章节 markdown。"""
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
    data[(c[0], c[1])].append((int(c[2]), float(c[3]), float(c[4])))
    norm[(c[0], c[1])].append(c[6])

FILES = [x.strip() for x in open("/mnt/llm-dev/qwen35/bench_raw/files.txt")]
rows = []
for fn in FILES:
    s = short(fn)
    ld, line = llama[s]
    r = {}
    for tier in ("default", "avx512"):
        pts = data[(tier, fn)]
        decs = sorted(x[1] for x in pts)
        r[tier] = dict(med=statistics.median(decs),
                       medpre=statistics.median(sorted(x[2] for x in pts)),
                       n=len(pts), allnorm=all(o == "Y" for o in norm[(tier, fn)]),
                       ratio=statistics.median(decs) / ld, lag=ld / statistics.median(decs))
    rows.append((s, fn, ld, line, v2.get(s), r))

def f(x, d=3):
    return f"{x:.{d}f}"

out = []
out.append("| 文件 | llama基线decode (BENCH行) | qwen35 默认tier decode | 默认比值 | 默认输出 | qwen35 avx512 decode | avx512比值 | avx512落后倍数 |")
out.append("|---|---|---|---|---|---|---|---|")
for s, fn, ld, line, v, r in rows:
    d, a = r["default"], r["avx512"]
    out.append(f"| {s} | {f(ld,2)} (L{line}) | {f(d['med'],2)} | {f(d['ratio'])} | {'正常' if d['allnorm'] else '乱码'} | {f(a['med'],2)} | {f(a['ratio'])} | {f(a['lag'],2)}× |")

# 统计
def stat(rs):
    return statistics.median(rs), statistics.mean(rs)

def_ratios = [r["default"]["ratio"] for s, fn, ld, l, v, r in rows if r["default"]["allnorm"]]
avx_ratios = [r["avx512"]["ratio"] for s, fn, ld, l, v, r in rows if r["avx512"]["allnorm"]]
dm, da = stat(def_ratios)
am, aa = stat(avx_ratios)
def_lag_med = statistics.median([r["default"]["lag"] for s, fn, ld, l, v, r in rows if r["default"]["allnorm"]])
avx_lag_med = statistics.median([r["avx512"]["lag"] for s, fn, ld, l, v, r in rows if r["avx512"]["allnorm"]])

best = []
for s, fn, ld, line, v, r in rows:
    cand = [(r["avx512"]["med"], "avx512", r["avx512"]["ratio"], r["avx512"]["lag"])]
    if r["default"]["allnorm"]:
        cand.append((r["default"]["med"], "default", r["default"]["ratio"], r["default"]["lag"]))
    b = max(cand)
    best.append((s, b[0], b[1], b[2], b[3]))

bm, ba = stat([x[3] for x in best])
blag = statistics.median([x[4] for x in best])

lines_out = []
lines_out.append(f"- **默认 tier（14 项输出正常）**：decode 比值中位 **{f(dm)}**，平均 **{f(da)}**，落后倍数中位 **{f(def_lag_med,2)}×**（n={len(def_ratios)}）。")
lines_out.append(f"- **avx512 tier（22 项全部输出正常）**：decode 比值中位 **{f(am)}**，平均 **{f(aa)}**，落后倍数中位 **{f(avx_lag_med,2)}×**（n=22）。这即「全 22 项按正确输出口径」的统计。")
lines_out.append(f"- **每个文件取「输出正常且最快」的 tier**：比值中位 **{f(bm)}**，平均 **{f(ba)}**，落后倍数中位 **{f(blag,2)}×**。")
lines_out.append("")
srt = sorted(best, key=lambda x: x[3])
lines_out.append("最落后 5 项（正确输出口径）：" + ", ".join(f"{s} {f(r)}（{f(l,2)}×，{t}）" for s, med, t, r, l in srt[:5]))
lines_out.append("")
lines_out.append("追平区（比值 ≥ 0.6）：" + ", ".join(f"{s} {f(r)}（{f(l,2)}×，{t}）" for s, med, t, r, l in srt if r >= 0.6))
lines_out.append("")
lines_out.append("接近区（0.4–0.6）：" + ", ".join(f"{s} {f(r)}" for s, med, t, r, l in srt if 0.4 <= r < 0.6))

print("\n".join(out))
print()
print("\n".join(lines_out))
