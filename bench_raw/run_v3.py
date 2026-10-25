#!/usr/bin/env python3
"""v3 单线程复测：22 个 GGUF × 2 tier × 3 次，串行执行。
输出 bench_raw/v3_sweep.tsv (tab 分隔)：tier 文件 序号 decode tok/s prefill tok/s 输出摘要 正常?
支持断点续跑（跳过已完成的 tier/file/run 组合）。
"""
import os, re, subprocess, sys, time, unicodedata

BIN = "/mnt/llm-dev/qwen35/zig-out/bin/qwen35"
MODEL_DIR = "/mnt/Qwen3.5-0.8B-GGUF"
OUT = "/mnt/llm-dev/qwen35/bench_raw/v3_sweep.tsv"
PROMPT = "天空为什么是蓝色的？"
FILES = [
    "Qwen3.5-0.8B-BF16.gguf",
    "Qwen3.5-0.8B-Q8_0.gguf",
    "Qwen3.5-0.8B-UD-Q8_K_XL.gguf",
    "Qwen3.5-0.8B-Q6_K.gguf",
    "Qwen3.5-0.8B-UD-Q6_K_XL.gguf",
    "Qwen3.5-0.8B-Q5_K_M.gguf",
    "Qwen3.5-0.8B-UD-Q5_K_XL.gguf",
    "Qwen3.5-0.8B-Q5_K_S.gguf",
    "Qwen3.5-0.8B-Q4_K_M.gguf",
    "Qwen3.5-0.8B-UD-Q4_K_XL.gguf",
    "Qwen3.5-0.8B-Q4_K_S.gguf",
    "Qwen3.5-0.8B-Q4_1.gguf",
    "Qwen3.5-0.8B-Q4_0.gguf",
    "Qwen3.5-0.8B-IQ4_NL.gguf",
    "Qwen3.5-0.8B-IQ4_XS.gguf",
    "Qwen3.5-0.8B-UD-Q3_K_XL.gguf",
    "Qwen3.5-0.8B-Q3_K_M.gguf",
    "Qwen3.5-0.8B-Q3_K_S.gguf",
    "Qwen3.5-0.8B-UD-IQ3_XXS.gguf",
    "Qwen3.5-0.8B-UD-Q2_K_XL.gguf",
    "Qwen3.5-0.8B-UD-IQ2_M.gguf",
    "Qwen3.5-0.8B-UD-IQ2_XXS.gguf",
]
KEYWORDS = ("天空", "蓝", "散射", "大气", "波", "光", "颜色", "sky", "blue")

def is_normal(text: str) -> bool:
    """判定「正常中文」：含关键词，且不是拉丁+西里尔+CJK 无标点乱排。"""
    if not text:
        return False
    if any(k in text for k in KEYWORDS):
        return True
    # 无关键词：统计 CJK 比例与可读标点
    chars = [c for c in text if not c.isspace()]
    if not chars:
        return False
    cjk = sum(1 for c in chars if 0x4E00 <= ord(c) <= 0x9FFF)
    cyr = sum(1 for c in chars if 0x0400 <= ord(c) <= 0x04FF)
    lat = sum(1 for c in chars if c.isascii() and c.isalpha())
    punct = sum(1 for c in chars if c in "，。、；：？！【】（）《》“” '*#-")
    if cjk == 0 and cyr == 0:
        return True  # 纯拉丁（少见），视为可读
    # 乱码特征：CJK 与西里尔/拉丁混排，且几乎没有中文标点
    if cyr > 0 and cjk > 0:
        return False
    if cjk > 0 and punct == 0:
        return False
    if cjk / len(chars) < 0.3 and lat / len(chars) > 0.5:
        return False
    return cjk > 0

def parse(out: str):
    m = re.search(r"Output:\s*(.*?)\n-+", out, re.S)
    text = m.group(1).strip() if m else ""
    g = re.search(r"Generate:\s*(\d+)\s+tokens\s+in\s+([\d.]+)\s+s\s+\(([\d.]+)\s+tok/s\)", out)
    p = re.search(r"Prefill:\s*(\d+)\s+tokens\s+in\s+([\d.]+)\s+s\s+\(([\d.]+)\s+tok/s\)", out)
    decode = float(g.group(3)) if g else float("nan")
    ntok = int(g.group(1)) if g else -1
    prefill = float(p.group(3)) if p else float("nan")
    return text, decode, prefill, ntok

def done_keys():
    keys = set()
    if os.path.exists(OUT):
        with open(OUT) as f:
            for line in f:
                c = line.rstrip("\n").split("\t")
                if len(c) >= 3 and c[0] in ("default", "avx512"):
                    keys.add((c[0], c[1], int(c[2])))
    return keys

def main():
    tiers = [("default", {}), ("avx512", {"QWEN35_SIMD_TIER": "avx512"})]
    keys = done_keys()
    new = not os.path.exists(OUT)
    f = open(OUT, "a")
    if new:
        f.write("tier\tfile\trun\tdecode_tok_s\tprefill_tok_s\toutput_head40\tnormal\tgen_tokens\tstdout_path\n")
        f.flush()
    total = len(FILES) * len(tiers) * 3
    i = 0
    for tier, env in tiers:
        for fn in FILES:
            for run in (1, 2, 3):
                i += 1
                if (tier, fn, run) in keys:
                    continue
                e = dict(os.environ)
                e.update(env)
                t0 = time.time()
                p = subprocess.run([BIN, "run", os.path.join(MODEL_DIR, fn), PROMPT,
                                    "-n", "48", "-t", "0", "-s", "1", "--threads", "1"],
                                   capture_output=True, env=e, timeout=300)
                out = p.stdout.decode("utf-8", "replace")
                text, dec, pre, ntok = parse(out)
                ok = "Y" if is_normal(text) else "N"
                head = text[:40].replace("\t", " ").replace("\n", " ")
                rec = [tier, fn, run, f"{dec:.2f}", f"{pre:.2f}", head, ok, ntok, ""]
                f.write("\t".join(map(str, rec)) + "\n")
                f.flush()
                print(f"[{i}/{total}] {tier} {fn} run{run} decode={dec:.2f} prefill={pre:.2f} ok={ok} ntok={ntok} ({time.time()-t0:.1f}s)", flush=True)
    f.close()
    print("DONE", flush=True)

if __name__ == "__main__":
    main()
