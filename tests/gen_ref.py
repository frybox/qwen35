"""Generate reference dump of a GGUF file using gguf-py (independent implementation).
Usage: python3 tests/gen_ref.py <file.gguf> <out.txt>
"""
import sys, os, json
sys.path.insert(0, '/home/wisedu/llama.cpp/gguf-py')
import numpy as np
from gguf import GGUFReader
from gguf.constants import GGML_QUANT_SIZES

TNAME = {0:'UINT8',1:'INT8',2:'UINT16',3:'INT16',4:'UINT32',5:'INT32',6:'FLOAT32',
         7:'BOOL',8:'STRING',9:'ARRAY',10:'UINT64',11:'INT64',12:'FLOAT64'}

def fmt_scalar(t, v):
    if t == 7:
        return 'true' if v else 'false'
    if t in (6, 12):
        return repr(float(v))
    return str(int(v))

def fmt_str(s):
    out = ['"']
    for ch in s:
        if ch == '"': out.append('\\"')
        elif ch == '\\': out.append('\\\\')
        elif ch == '\n': out.append('\\n')
        elif ch == '\r': out.append('\\r')
        elif ch == '\t': out.append('\\t')
        elif ord(ch) < 0x20: out.append('\\x%02x' % ord(ch))
        else: out.append(ch)
    out.append('"')
    return ''.join(out)

def main():
    path, outp = sys.argv[1], sys.argv[2]
    r = GGUFReader(path, 'r')
    fsize = os.path.getsize(path)
    L = []
    L.append("file_size %d" % fsize)
    L.append("version %d" % int(r.fields['GGUF.version'].parts[-1][0]))
    L.append("tensor_count %d" % len(r.tensors))
    L.append("metadata_kv_count %d" % int(r.fields['GGUF.kv_count'].parts[-1][0]))
    L.append("alignment %d" % r.alignment)
    L.append("data_offset %d" % r.data_offset)
    L.append("")
    L.append("=== metadata ===")
    keys = [k for k in r.fields.keys() if not k.startswith('GGUF.')]
    for k in keys:
        f = r.fields[k]
        mt = f.types[0]
        try:
            mtn = mt.name
        except AttributeError:
            mtn = TNAME.get(int(mt), 'UNKNOWN')
        if mtn == 'ARRAY':
            sub = f.types[-1]
            stn = sub.name if hasattr(sub, 'name') else TNAME.get(int(sub), 'UNKNOWN')
            n = int(f.data.shape[0]) if hasattr(f.data, 'shape') and f.data.ndim else 0
            vals = f.contents()
            if stn == 'STRING':
                elems = vals
            else:
                elems = [fmt_scalar(int(sub), x) for x in np.atleast_1d(vals)]
            if len(elems) <= 16:
                body = ", ".join(fmt_str(e) if stn == 'STRING' else e for e in elems)
                L.append("%s | ARRAY<%s> len=%d [%s]" % (k, stn, len(elems), body))
            else:
                L.append("%s | ARRAY<%s> len=%d" % (k, stn, len(elems)))
        else:
            v = np.atleast_1d(f.contents())[0]
            if mtn == 'STRING':
                L.append("%s | STRING %s" % (k, fmt_str(str(v))))
            else:
                L.append("%s | %s %s" % (k, mtn, fmt_scalar(int(mt), v)))
    L.append("")
    L.append("=== tensors ===")
    total = 0
    for t in r.tensors:
        dims = [int(x) for x in t.shape]
        n = 1
        for d in dims: n *= d
        bs, ts = GGML_QUANT_SIZES[t.tensor_type]
        nbytes = n * ts // bs
        total += (nbytes + r.alignment - 1) // r.alignment * r.alignment
        L.append("%s | %s | %s | %d | bytes=%d" % (
            t.name, t.tensor_type.name, ",".join(str(d) for d in dims), t.data_offset, nbytes))
    L.append("")
    L.append("sum_aligned_bytes %d" % total)
    L.append("check data_offset+sum == file_size : %s" % (
        "OK" if r.data_offset + total == fsize else "MISMATCH diff=%d" % (fsize - r.data_offset - total)))
    open(outp, 'w').write("\n".join(L) + "\n")
    print("wrote %s lines=%d" % (outp, len(L)))

main()
