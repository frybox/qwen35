"""Compare zig `qwen35 <file>` output against gguf-py reference dump.

Both sides are normalized to (key -> typed value) and
(tensor name -> (type, dims, offset, bytes)). Floats compare numerically.
Exits non-zero on any mismatch.
"""
import sys, re, ast

def parse_zig(path):
    meta, tens = {}, {}
    hdr = {}
    mode = None
    for line in open(path, encoding='utf-8', errors='replace'):
        line = line.rstrip('\n')
        if line.startswith('=== metadata'):
            mode = 'meta'; continue
        if line.startswith('=== tensors'):
            mode = 'tens'; continue
        if not line.strip():
            continue
        if mode is None:
            k, _, v = line.partition(' ')
            hdr[k] = v
        elif mode == 'meta':
            m = re.match(r'^\s*\d+ (.*?) = (.*)$', line)
            if m:
                meta[m.group(1)] = m.group(2)
        else:
            if line.startswith('name '):
                continue
            cols = line.split()
            if cols[0] == 'name' or cols[0].startswith('('):
                continue
            # name has no spaces; columns: name type dims offset abs_offset bytes
            name, typ, dims, off, absoff, nbytes = cols[0], cols[1], cols[2], cols[3], cols[4], cols[5]
            tens[name] = (typ, dims, int(absoff), int(nbytes))
    return hdr, meta, tens

def parse_ref(path):
    meta, tens = {}, {}
    hdr = {}
    mode = None
    for line in open(path, encoding='utf-8'):
        line = line.rstrip('\n')
        if line.startswith('=== metadata'):
            mode = 'meta'; continue
        if line.startswith('=== tensors'):
            mode = 'tens'; continue
        if not line.strip():
            continue
        if mode is None:
            k, _, v = line.partition(' ')
            hdr[k] = v
        elif mode == 'meta':
            key, _, rest = line.partition(' | ')
            meta[key] = rest
        else:
            if line.count('|') != 4:
                k, _, v = line.partition(' ')
                hdr[k] = v
                continue
            name, typ, dims, off, nbytes = [c.strip() for c in line.split('|')]
            tens[name] = (typ, '[' + dims + ']', int(off), int(nbytes.split('=')[1]))
    return hdr, meta, tens

def norm_value(s):
    """Normalize a rendered KV value for comparison."""
    s = s.strip()
    m = re.match(r'^(UINT8|INT8|UINT16|INT16|UINT32|INT32|UINT64|INT64|FLOAT32|FLOAT64|BOOL)\s+(.*)$', s)
    if m:
        t, v = m.group(1), m.group(2)
        if t.startswith('FLOAT'):
            return (t, round(float(v), 12))
        if t == 'BOOL':
            return (t, v.lower() == 'true')
        return (t, int(v))
    # zig prints bare "..." ; ref prints `STRING "..."` — accept both
    m = re.match(r'^STRING (".*")$', s, re.S)
    if m:
        s = m.group(1)
    # string literal (possibly with a trailing "(len=N)" annotation in zig)
    m = re.match(r'^"(.*?)"( \(len=\d+\))?$', s, re.S)
    if m:
        return ('STRING', ast.literal_eval('"' + m.group(1) + '"'))
    # ARRAY
    m = re.match(r'^ARRAY<(\w+)> len=(\d+)( .*)?$', s, re.S)
    if m:
        et, n, rest = m.group(1), int(m.group(2)), (m.group(3) or '').strip()
        return ('ARRAY', et, n, rest)
    return ('RAW', s)

def main():
    zig_path, ref_path = sys.argv[1], sys.argv[2]
    zh, zm, zt = parse_zig(zig_path)
    rh, rm, rt = parse_ref(ref_path)
    fails = []

    for k in ('file_size', 'version', 'tensor_count', 'metadata_kv_count', 'alignment', 'data_offset'):
        if zh.get(k) != rh.get(k):
            fails.append("header %s: zig=%r ref=%r" % (k, zh.get(k), rh.get(k)))
    print("header keys checked: %s" % ", ".join('%s=%s' % (k, zh.get(k)) for k in ('version','tensor_count','metadata_kv_count','alignment','data_offset')))

    if set(zm) != set(rm):
        fails.append("metadata key set differs: only-zig=%s only-ref=%s" % (
            sorted(set(zm) - set(rm)), sorted(set(rm) - set(zm))))
    n_meta_diff = 0
    for k in sorted(set(zm) & set(rm)):
        zv, rv = norm_value(zm[k]), norm_value(rm[k])
        ok = False
        if zv == rv:
            ok = True
        elif zv[0] == 'ARRAY' and rv[0] == 'ARRAY':
            # always require identical element type and length
            if zv[1] == rv[1] and zv[2] == rv[2]:
                zrest, rrest = zv[3], rv[3]
                if zrest.startswith('...') or rrest.startswith('...') or not zrest or not rrest:
                    ok = True  # long array: only shape is asserted
                else:
                    def elems(t):
                        t = t.strip()
                        if t.startswith('[') and t.endswith(']'):
                            body = t[1:-1]
                            return [e.strip() for e in body.split(',') if e.strip()]
                        # space-separated quoted strings
                        return re.findall(r'"(?:\\.|[^"\\])*"', t)
                    ze, re_ = elems(zrest), elems(rrest)
                    ok = len(ze) == len(re_) and all(
                        a.strip('"') == b.strip('"') for a, b in zip(ze, re_))
        if not ok:
            n_meta_diff += 1
            if n_meta_diff <= 10:
                fails.append("metadata %s:\n   zig=%s\n   ref=%s" % (k, zm[k][:160], rm[k][:160]))
    print("metadata entries compared: %d (mismatches shown: %d)" % (len(set(zm) & set(rm)), n_meta_diff))

    if set(zt) != set(rt):
        fails.append("tensor name set differs: only-zig=%s only-ref=%s" % (
            sorted(set(zt) - set(rt))[:5], sorted(set(rt) - set(zt))[:5]))
    n_ten_diff = 0
    for k in sorted(set(zt) & set(rt)):
        if zt[k] != rt[k]:
            n_ten_diff += 1
            if n_ten_diff <= 10:
                fails.append("tensor %s:\n   zig=%s\n   ref=%s" % (k, zt[k], rt[k]))
    print("tensors compared: %d (mismatches: %d)" % (len(set(zt) & set(rt)), n_ten_diff))

    # type histogram
    from collections import Counter
    zc, rc = Counter(v[0] for v in zt.values()), Counter(v[0] for v in rt.values())
    print("zig type histogram: %s" % dict(sorted(zc.items())))
    if zc != rc:
        fails.append("type histogram differs: zig=%s ref=%s" % (dict(zc), dict(rc)))
    sc = sum((v[3] + 31)//32*32 for v in zt.values())
    print("zig sum_aligned=%d  ref sum_aligned=%s" % (sc, rh.get('sum_aligned_bytes')))

    print()
    if fails:
        print("FAIL: %d problem(s)" % len(fails))
        for f in fails[:20]:
            print(" -", f)
        sys.exit(1)
    print("PASS: zig output matches gguf-py reference")

main()
