#!/usr/bin/env python3
"""Compare qwen35_v2 / llama_v2 against baseline raw.tsv (medians).

Fixed version:
  - baseline raw.tsv uses engine label "llama" (v2 uses "llama-server"):
    match any engine starting with "llama".
  - FAIL rows: the Q5_K_S llama failure has note "rerun-1 FAIL: HTTP 503",
    so filter on "FAIL" substring, not equality; also guard int() parsing.
  - rerun: if a file has >=3 rows tagged "rerun", those form the official
    complete 3-run set -> use ONLY the rerun rows as the baseline sample.
"""
import csv, statistics, sys, os

BASE = "bench_raw/raw.tsv"
V2Q = "bench_raw/qwen35_v2.tsv"
V2L = "bench_raw/llama_v2.tsv"

def load(path, engine):
    rows = {}
    with open(path) as f:
        r = csv.DictReader(f, delimiter="\t")
        for d in r:
            eng = d["engine"].strip()
            if engine.startswith("llama"):
                if not eng.startswith("llama"): continue
            elif eng != engine: continue
            note = (d.get("note") or "").strip()
            if "FAIL" in note: continue          # skip failed runs
            pt, dt = d["prefill_tok"].strip(), d["decode_tok"].strip()
            if not pt.isdigit() or not dt.isdigit():
                continue                          # malformed row
            tag = "rerun" if "rerun" in note else "plain"
            rows.setdefault(d["file"], []).append(
                (tag, int(pt), float(d["prefill_toks_per_s"]),
                 int(dt), float(d["decode_toks_per_s"])))
    out = {}
    for k, v in rows.items():
        rer = [x for x in v if x[0] == "rerun"]
        if len(rer) >= 3:
            v = rer                              # rerun set replaces baseline set
        out[k] = {
            "ptok": statistics.median([x[1] for x in v]),
            "pre":  statistics.median([x[2] for x in v]),
            "dtok": statistics.median([x[3] for x in v]),
            "dec":  statistics.median([x[4] for x in v]),
            "n":    len(v),
        }
    return out

def main():
    for p in (BASE, V2Q, V2L):
        if not os.path.exists(p):
            print(f"MISSING {p}"); sys.exit(1)
    base_q = load(BASE, "qwen35"); base_l = load(BASE, "llama")
    new_q = load(V2Q, "qwen35");   new_l = load(V2L, "llama-server")
    files = sorted(base_q.keys(), key=lambda k: -base_q[k]["dec"])
    hdr = ["file", "q_old_dec", "q_new_dec", "q_gain%", "ll_old_dec", "ll_new_dec",
           "ll_gain%", "new_ratio(q/llama)", "old_ratio", "q_new_pre", "q_old_pre", "pre_gain%"]
    print("\t".join(hdr))
    ratios_old, ratios_new = [], []
    for f in files:
        qn, qo = new_q.get(f), base_q.get(f)
        lo = base_l.get(f); ln = new_l.get(f)
        ro = rnew = None
        if qo and lo: ro = qo["dec"]/lo["dec"]; ratios_old.append(ro)
        if qn and ln: rnew = qn["dec"]/ln["dec"]; ratios_new.append(rnew)
        row = [f,
            f"{qo['dec']:.2f}" if qo else "-",
            f"{qn['dec']:.2f}" if qn else "MISSING",
            f"{(qn['dec']/qo['dec']-1)*100:+.1f}" if qo and qn else "-",
            f"{lo['dec']:.2f}" if lo else "-",
            f"{ln['dec']:.2f}" if ln else "MISSING",
            f"{(ln['dec']/lo['dec']-1)*100:+.1f}" if lo and ln else "-",
            f"{rnew:.2f}" if rnew else "-", f"{ro:.2f}" if ro else "-",
            f"{qn['pre']:.2f}" if qn else "-",
            f"{qo['pre']:.2f}" if qo else "-",
            f"{(qn['pre']/qo['pre']-1)*100:+.1f}" if qo and qn else "-"]
        print("\t".join(row))
    print(f"\nfiles q_old={len(base_q)} q_new={len(new_q)} ll_old={len(base_l)} ll_new={len(new_l)}")
    if ratios_old:
        print(f"median old ratio {statistics.median(ratios_old):.3f}  mean old {statistics.mean(ratios_old):.3f}")
    if ratios_new:
        print(f"median new ratio {statistics.median(ratios_new):.3f}  mean new {statistics.mean(ratios_new):.3f}")
    qg = [(f, (new_q[f]["dec"]/base_q[f]["dec"]-1)*100) for f in files if f in new_q]
    print("q decode gain: median %.1f%% mean %.1f%% over %d files" %
          (statistics.median([g for _, g in qg]), statistics.mean([g for _, g in qg]), len(qg)))
    pre = [(f, (new_q[f]["pre"]/base_q[f]["pre"]-1)*100) for f in files if f in new_q]
    print("q prefill gain: median %.1f%% mean %.1f%%" %
          (statistics.median([g for _, g in pre]), statistics.mean([g for _, g in pre])))
    srt = sorted(qg, key=lambda x: -x[1])
    print("top3 gain:", [(f, round(g, 1)) for f, g in srt[:3]])
    print("bottom3 gain:", [(f, round(g, 1)) for f, g in srt[-3:]])

main()
