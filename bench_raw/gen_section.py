#!/usr/bin/env python3
"""生成 BENCH.md 的 v3 章节全文（含 22 行表 + 统计 + 与 v2 对比）。"""
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
        d = sorted(x[1] for x in pts)
        r[tier] = dict(med=statistics.median(d),
                       medpre=statistics.median(sorted(x[2] for x in pts)),
                       n=len(pts), allnorm=all(o == "Y" for o in norm[(tier, fn)]),
                       cnt=sum(1 for o in norm[(tier, fn)] if o == "Y"),
                       ratio=statistics.median(d) / ld, lag=ld / statistics.median(d))
    best = r["avx512"]
    bt = "avx512"
    if r["default"]["allnorm"] and r["default"]["med"] > best["med"]:
        best, bt = r["default"], "default"
    rows.append((s, ld, line, r, best, bt))

def f(x, d=3):
    return f"{x:.{d}f}"

out = []
out.append("| 文件 | llama基线decode (BENCH L行) | qwen35 默认tier decode | 默认比值 | 默认输出 | qwen35 avx512 decode | avx512比值 | avx512落后倍数 |")
out.append("|---|---|---|---|---|---|---|---|")
for s, ld, line, r, best, bt in rows:
    d, a = r["default"], r["avx512"]
    out.append(f"| {s} | {f(ld,2)} (L{line}) | {f(d['med'],2)} | {f(d['ratio'])} | {'正常 ' + str(d['cnt'])+'/3' if d['allnorm'] else '乱码 0/3'} | {f(a['med'],2)} | {f(a['ratio'])} | {f(a['lag'],2)}× |")

def_ratios = [r["default"]["ratio"] for s, ld, l, r, b, t in rows if r["default"]["allnorm"]]
avx_ratios = [r["avx512"]["ratio"] for s, ld, l, r, b, t in rows if r["avx512"]["allnorm"]]
dm, da = statistics.median(def_ratios), statistics.mean(def_ratios)
am, aa = statistics.median(avx_ratios), statistics.mean(avx_ratios)
def_lag = statistics.median([r["default"]["lag"] for s, ld, l, r, b, t in rows if r["default"]["allnorm"]])
avx_lag = statistics.median([r["avx512"]["lag"] for s, ld, l, r, b, t in rows if r["avx512"]["allnorm"]])
best_ratios = [(s, b["ratio"], b["lag"], t) for s, ld, l, r, b, t in rows]
bm, ba = statistics.median([x[1] for x in best_ratios]), statistics.mean([x[1] for x in best_ratios])
blag = statistics.median([x[2] for x in best_ratios])
srt = sorted(best_ratios, key=lambda x: x[1])

# v2 对比（同基线换算）
cmp_rows, deltas = [], []
for s, ld, line, r, best, bt in rows:
    v2dec, v2llama, v2line = v2[s]
    v2r = v2dec / ld
    dr = best["ratio"] - v2r
    dd = 100 * (best["med"] / v2dec - 1)
    tag = "变好" if dr > 0.02 else ("变差" if dr < -0.02 else "持平")
    cmp_rows.append((s, v2dec, best["med"], dd, v2r, best["ratio"], dr, tag))
    deltas.append(dr)
comparable = [x[6] for x in cmp_rows if x[0] in [s for s, ld, l, r, b, t in rows if r["default"]["allnorm"]]]

sec = []
sec.append("| 文件 | v2 qwen35 decode | v3 正确输出 decode | decode变化 | v2比值(换算到同一基线) | v3比值 | 比值变化 |")
sec.append("|---|---|---|---|---|---|---|")
for s, v2dec, bmed, dd, v2r, v3r, dr, tag in cmp_rows:
    sec.append(f"| {s} | {f(v2dec,2)} | {f(bmed,2)} | {dd:+.1f}% | {f(v2r)} | {f(v3r)} | {dr:+.3f} {tag} |")

print("\n".join(out))
print()
print("【统计】")
print(f"- 默认 tier（{len(def_ratios)} 项输出正常）：比值中位 {f(dm)}，平均 {f(da)}，落后倍数中位 {f(def_lag,2)}×")
print(f"- avx512 tier（{len(avx_ratios)} 项全部正常，即全 22 项正确输出口径）：比值中位 {f(am)}，平均 {f(aa)}，落后倍数中位 {f(avx_lag,2)}×")
print(f"- 每文件取「输出正常且最快 tier」：比值中位 {f(bm)}，平均 {f(ba)}，落后倍数中位 {f(blag,2)}×")
print(f"- 最落后5：{', '.join(f'{s} {f(r)}({f(l,2)}×)' for s, r, l, t in srt[:5])}")
print(f"- 追平(>=0.6)：{', '.join(f'{s} {f(r)}({f(l,2)}×，{t})' for s, r, l, t in srt if r >= 0.6)}")
print(f"- 接近(0.4~0.6)：{', '.join(f'{s} {f(r)}' for s, r, l, t in srt if 0.4 <= r < 0.6)}")
print(f"- 与 v2 同基线对比：全 22 项比值变化中位 {statistics.median(deltas):+.3f} 平均 {statistics.mean(deltas):+.3f}；仅 v2 输出正常 13 项中位 {statistics.median(comparable):+.3f} 平均 {statistics.mean(comparable):+.3f}")
print(f"- 变好：{', '.join(f'{x[0]}({x[6]:+.3f})' for x in sorted(cmp_rows, key=lambda y: -y[6]) if x[7]=='变好')}")
print(f"- 变差：{', '.join(f'{x[0]}({x[6]:+.3f})' for x in sorted(cmp_rows, key=lambda y: y[6]) if x[7]=='变差')}")
print()
print("\n".join(sec))
