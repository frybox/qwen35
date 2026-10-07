"""Independent element-for-element check of the three huge tokenizer arrays.

The zig dump (--full) renders strings with \\xNN byte escapes; gguf-py is the
oracle. Every element must match exactly.
"""
import sys, re
sys.path.insert(0, '/home/wisedu/llama.cpp/gguf-py')
from gguf import GGUFReader
from collections import Counter

zpath, gguf_path = sys.argv[1], sys.argv[2]
r = GGUFReader(gguf_path, 'r')

KEYS = ('tokenizer.ggml.tokens', 'tokenizer.ggml.merges', 'tokenizer.ggml.token_type')

def unescape(lit: str) -> bytes:
    """Turn a zig-rendered string literal (without surrounding quotes) into bytes."""
    assert lit.startswith('"') and lit.endswith('"'), lit[:40]
    body = lit[1:-1]
    out = bytearray()
    i = 0
    while i < len(body):
        ch = body[i]
        if ch == '\\':
            nxt = body[i+1]
            if nxt == 'x':
                out.append(int(body[i+2:i+4], 16)); i += 4
            elif nxt == 'n': out.append(0x0a); i += 2
            elif nxt == 'r': out.append(0x0d); i += 2
            elif nxt == 't': out.append(0x09); i += 2
            elif nxt in '"\\': out.append(ord(nxt)); i += 2
            else: raise ValueError("bad escape %r" % body[i:i+4])
        else:
            out.extend(ch.encode('utf-8')); i += 1
    return bytes(out)

# --- locate the array start lines -------------------------------------------
lines = open(zpath, encoding='utf-8', errors='replace').read().split('\n')
vars_ = {}
for idx, line in enumerate(lines):
    m = re.match(r'^\s*\d+ (\S+) = ARRAY<(\w+)> len=(\d+) ?(.*)$', line)
    if m and m.group(1) in KEYS:
        vars_[m.group(1)] = (idx, m.group(2), int(m.group(3)), m.group(4))

fails = []
for key in KEYS:
    if key not in vars_:
        fails.append("%s: not found in zig output" % key); continue
    idx, et, n, first = vars_[key]
    # gather until we hit a non-array line (blank / section header / next KV)
    parts = [first]
    j = idx + 1
    while j < len(lines):
        ln = lines[j]
        if ln.startswith('===') or ln.strip() == '' or re.match(r'^\s*\d+ \S+ = ', ln):
            break
        parts.append(ln); j += 1
    body = ' '.join(parts).strip()
    if et == 'STRING':
        elems = re.findall(r'"(?:\\.|[^"\\])*"', body)
        zig = [unescape(e) for e in elems]
        ref = [t.encode('utf-8') for t in r.fields[key].contents()]
    else:
        elems = [e.strip() for e in body.strip('[]').split(',') if e.strip()]
        zig = [int(e) for e in elems]
        ref = [int(x) for x in r.fields[key].contents()]

    if len(zig) != len(ref):
        fails.append("%s: len zig=%d ref=%d" % (key, len(zig), len(ref)))
        continue
    bad = None
    for i, (a, b) in enumerate(zip(zig, ref)):
        if a != b:
            bad = i; break
    if bad is not None:
        fails.append("%s: elem %d differs: zig=%r ref=%r" % (key, bad, zig[bad], ref[bad]))
    print("%-30s %-6s n=%-7d %s" % (key, et, n, "OK" if bad is None else "MISMATCH"))

# --- token_type insight, verified numerically --------------------------------
tt = [int(x) for x in r.fields['tokenizer.ggml.token_type'].contents()]
toks = r.fields['tokenizer.ggml.tokens'].contents()
print("\ntoken_type histogram:", dict(sorted(Counter(tt).items())))
nonnormal = [i for i, t in enumerate(tt) if t != 1]
print("ids with token_type != 1: count=%d" % len(nonnormal))
for i in nonnormal[:40]:
    print("   id %-7d type=%d token=%r" % (i, tt[i], toks[i]))
if len(nonnormal) > 40:
    print("   ... and %d more" % (len(nonnormal) - 40))
print("vocab_size=%d  n_merges=%d" % (len(toks), len(r.fields['tokenizer.ggml.merges'].contents())))

print()
print("FAIL: %s" % fails if fails else "PASS: all three arrays match gguf-py element-for-element")
sys.exit(1 if fails else 0)
