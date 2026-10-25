#!/usr/bin/env python3
"""按严格规则重判 normal 列（基于 head40），保留原 heuristic 列。"""
import collections

TSV = "/mnt/llm-dev/qwen35/bench_raw/v3_sweep.tsv"

def strict(h):
    kw = any(k in h for k in ("天空", "蓝", "散射", "大气", "光的", "颜色", "sky", "blue"))
    susp = sum(1 for c in h if 0x0400 <= ord(c) <= 0x04FF
               or 0x0E00 <= ord(c) <= 0x0E7F
               or 0x0600 <= ord(c) <= 0x06FF
               or ord(c) == 0xFFFD
               or 0x0080 <= ord(c) <= 0x00FF)
    cjk = sum(1 for c in h if 0x4E00 <= ord(c) <= 0x9FFF)
    return kw and susp == 0 and cjk >= 5

rows = []
with open(TSV) as f:
    hdr = f.readline().rstrip("\n").split("\t")
    for ln in f:
        c = ln.rstrip("\n").split("\t")
        if len(c) < 9:
            continue
        s = strict(c[5])
        c[6] = "Y" if s else "N"
        rows.append(c)

hdr[6] = "normal"
with open(TSV, "w") as f:
    f.write("\t".join(hdr) + "\n")
    for c in rows:
        f.write("\t".join(c) + "\n")
print("rewritten", len(rows))
cnt = collections.Counter((c[0], c[6]) for c in rows)
print(cnt)
