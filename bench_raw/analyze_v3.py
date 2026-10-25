#!/usr/bin/env python3
"""v3 复测分析：中位数、比值、落后倍数、统计、与 v2 对比。"""
import statistics, re, os, collections

TSV = "/mnt/llm-dev/qwen35/bench_raw/v3_sweep.tsv"
BENCH = "/mnt/llm-dev/qwen35/BENCH.md"

# 1) 从 BENCH.md 原表解析 llama decode 基线 + 行号 + v2 比值
llama = {}   # short name -> (decode, line_no)
v2 = {}      # short name -> (qwen35_v2_decode, v2_ratio, line_no)
with open(BENCH) as f:
    lines = f.readlines()
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
with open(BENCH) as f:
    txt = f.readlines()
for i, ln in enumerate(txt, 1):
    if ln.startswith("| Q") or ln.startswith("| UD") or ln.startswith("| BF") or ln.startswith("| IQ"):
        c = [x.strip() for x in ln.strip("|").split("|")]
        if len(c) >= 9 and c[0] in llama and "%" in c[3]:
            v2[c[0]] = (float(c[2]), float(c[7]), i)

def short(fn):
    return fn.replace("Qwen3.5-0.8B-", "").replace(".gguf", "")

# 2) 读 TSV，按 (tier, file) 收集 3 次
data = collections.defaultdict(list)
norm = collections.defaultdict(list)
heads = {}
with open(TSV) as f:
    hdr = f.readline()
    for ln in f:
        c = ln.rstrip("\n").split("\t")
        if len(c) < 7:
            continue
        tier, fn, run, dec, pre, head, ok = c[0], c[1], int(c[2]), c[3], c[4], c[5], c[6]
        data[(tier, fn)].append((run, float(dec), float(pre)))
        norm[(tier, fn)].append(ok)
        heads[(tier, fn)] = head

rows = []
for fn in [x.strip() for x in open("/mnt/llm-dev/qwen35/bench_raw/files.txt")]:
    s = short(fn)
    ld, line = llama[s]
    r = {}
    for tier in ("default", "avx512"):
        pts = data[(tier, fn)]
        if len(pts) == 0:
            r[tier] = dict(med=float("nan"), medpre=float("nan"), decs=[], ratio=float("nan"),
                           lag=float("nan"), n=0, allnorm=False, normcnt=0, head="")
            continue
        decs = sorted(x[1] for x in pts)
        pres = sorted(x[2] for x in pts)
        med = statistics.median(decs)
        r[tier] = dict(med=med, medpre=statistics.median(pres), decs=decs,
                       ratio=med / ld, lag=ld / med, n=len(pts),
                       allnorm=all(o == "Y" for o in norm[(tier, fn)]),
                       normcnt=sum(1 for o in norm[(tier, fn)] if o == "Y"),
                       head=heads[(tier, fn)])
    rows.append((s, fn, ld, line, v2.get(s), r))

def fmt(x, d=2):
    return f"{x:.{d}f}"

# 3) 打印结果表
out_lines = []
out_lines.append("| 文件 | BENCH行 | llama基线decode | qwen35默认decode | 默认比值 | 默认正常? | qwen35 avx512 decode | avx512比值 | avx512落后倍数 |")
out_lines.append("|---|---|---|---|---|---|---|---|---|")
for s, fn, ld, line, v, r in rows:
    d, a = r["default"], r["avx512"]
    out_lines.append(f"| {s} | L{line} | {fmt(ld)} | {fmt(d['med'])} | {fmt(d['ratio'],3)} | {d['normcnt']}/3 | {fmt(a['med'])} | {fmt(a['ratio'],3)} | {fmt(a['lag'],2)}× |")

# 4) 统计
def stats(ratio_list):
    if not ratio_list:
        return None
    return statistics.median(ratio_list), statistics.mean(ratio_list), len(ratio_list)

default_ok = [(s, r["default"]["ratio"]) for s, fn, ld, line, v, r in rows if r["default"]["allnorm"]]
avx_ok = [(s, r["avx512"]["ratio"]) for s, fn, ld, line, v, r in rows if r["avx512"]["allnorm"]]

print("=== 22 行汇总 ===")
for l in out_lines:
    print(l)

print()
dm = stats([x[1] for x in default_ok])
am = stats([x[1] for x in avx_ok])
print(f"默认tier 输出正常项: n={dm[2]} 中位比值={fmt(dm[0],3)} 平均={fmt(dm[1],3)} 落后倍数中位={fmt(1/dm[0],2)}×")
print(f"avx512   输出正常项: n={am[2]} 中位比值={fmt(am[0],3)} 平均={fmt(am[1],3)} 落后倍数中位={fmt(1/am[0],2)}×")
print("默认tier 输出正常文件:", ", ".join(x[0] for x in default_ok))
print("avx512   输出非正常文件:", ", ".join(s for s, fn, ld, line, v, r in rows if not r["avx512"]["allnorm"]))

# 5) 最落后 5 项 / 追平项（以 avx512 口径）
sorted_avx = sorted([(s, r["avx512"]["ratio"], r["avx512"]["lag"]) for s, fn, ld, line, v, r in rows], key=lambda x: x[1])
sorted_def = sorted([(s, r["default"]["ratio"], r["default"]["lag"]) for s, fn, ld, line, v, r in rows
                     if r["default"]["allnorm"]], key=lambda x: x[1])
print()
print("最落后5项(avx512口径):", ", ".join(f"{s} {fmt(r,3)}({fmt(l,2)}×)" for s, r, l in sorted_avx[:5]))
print("追平(>=0.6, avx512):", ", ".join(f"{s} {fmt(r,3)}({fmt(l,2)}×)" for s, r, l in sorted_avx if r >= 0.6))
print("接近(0.4~0.6, avx512):", ", ".join(f"{s} {fmt(r,3)}({fmt(l,2)}×)" for s, r, l in sorted_avx if 0.4 <= r < 0.6))
print("落后倍数中位数(各文件 lag 的中位): avx512 =", fmt(statistics.median([x[2] for x in sorted_avx]), 2),
      " default(正常项) =", fmt(statistics.median([x[2] for x in sorted_def]), 2))
# 每个文件取「输出正常且最快」的 tier（可信最佳口径）
best = []
for s, fn, ld, line, v, r in rows:
    cand = [(r["avx512"]["med"], "avx512", r["avx512"]["ratio"], r["avx512"]["lag"])]
    if r["default"]["allnorm"]:
        cand.append((r["default"]["med"], "default", r["default"]["ratio"], r["default"]["lag"]))
    b = max(cand)
    best.append((s, b[0], b[1], b[2], b[3]))
print()
print("最佳正确输出口径(每项取正常且最快的tier):")
for s, med, tier, ra, lg in sorted(best, key=lambda x: x[3]):
    print(f"  {s}: {fmt(med)} tok/s ({tier}) ratio={fmt(ra,3)} lag={fmt(lg,2)}×")
print("中位比值 =", fmt(statistics.median([x[3] for x in best]), 3),
      " 平均 =", fmt(statistics.mean([x[3] for x in best]), 3),
      " 落后倍数中位 =", fmt(statistics.median([x[4] for x in best]), 2))

# 6) 与 v2 对比
print()
print("=== 与 v2 对比（v2比值 -> v3 avx512比值）===")
better, worse = [], []
for s, fn, ld, line, v, r in rows:
    if v is None:
        print(f"{s}: 无v2记录"); continue
    v2ratio = v[1]
    cur = r["avx512"]["ratio"]
    delta = cur - v2ratio
    tag = "变好" if delta > 0.02 else ("变差" if delta < -0.02 else "持平")
    (better if delta > 0.02 else worse if delta < -0.02 else []).append((s, v2ratio, cur, delta, v[2]))
    print(f"{s}: v2(L{v[2]}) {fmt(v2ratio,3)} -> v3avx {fmt(cur,3)} ({delta:+.3f}) {tag}; 默认tier比值 {fmt(r['default']['ratio'],3)}")
print()
print("变好:", ", ".join(f"{s}({d:+.2f})" for s, a, b, d, l in sorted(better, key=lambda x: -x[3])))
print("变差:", ", ".join(f"{s}({d:+.2f})" for s, a, b, d, l in sorted(worse, key=lambda x: x[3])))

# 7) qwen35 自身 decode 变化（v2本轮decode -> v3各tier中位），剔除 llama 侧噪声
print()
print("=== qwen35 decode 自身变化 (v2本轮 -> v3 avx512 / v3 默认) ===")
ch = []
for s, fn, ld, line, v, r in rows:
    if v is None:
        continue
    d2, d3a, d3d = v[0], r["avx512"]["med"], r["default"]["med"]
    ch.append((s, d2, d3a, d3d, 100*(d3a/d2-1), 100*(d3d/d2-1)))
for s, d2, d3a, d3d, pa, pd in sorted(ch, key=lambda x: -x[4]):
    print(f"{s}: v2={fmt(d2)} -> v3avx={fmt(d3a)} ({pa:+.1f}%) | v3默认={fmt(d3d)} ({pd:+.1f}%)")
med_a = statistics.median([x[4] for x in ch]); med_d = statistics.median([x[5] for x in ch])
mean_a = statistics.mean([x[4] for x in ch]); mean_d = statistics.mean([x[5] for x in ch])
print(f"v3avx vs v2: 中位 {med_a:+.1f}% 平均 {mean_a:+.1f}%")
print(f"v3默认 vs v2: 中位 {med_d:+.1f}% 平均 {mean_d:+.1f}%")
# 仅输出正常项的变化
ok_avx = [x[4] for s, d2, d3a, d3d, pa, pd in ch if next(r for s2, fn, ld, l, v, r in rows if s2 == s)["avx512"]["allnorm"]]
print(f"v3avx(仅avx512输出正常项 n={len(ok_avx)}): 中位 {statistics.median(ok_avx):+.1f}% 平均 {statistics.mean(ok_avx):+.1f}%")
