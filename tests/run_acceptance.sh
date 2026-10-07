#!/usr/bin/env bash
# End-to-end acceptance for the GGUF parser.
#
#   tests/run_acceptance.sh [file.gguf]
#
# Steps:
#   1. zig build test
#   2. zig build
#   3. generate a reference dump with gguf-py (independent implementation)
#   4. run the zig CLI with --full
#   5. compare header / metadata / per-tensor name-type-dims-offset-bytes
#   6. compare the three huge tokenizer arrays element-for-element
#
# Exits non-zero if any step fails.
set -uo pipefail

cd "$(dirname "$0")/.."
GGUF="${1:-/mnt/Qwen3.5-0.8B-GGUF/Qwen3.5-0.8B-Q8_0.gguf}"
ZIG="${ZIG:-/home/wisedu/software/zig-x86_64-linux-0.16.0/zig}"
PY="${PY:-python3}"
WORK="$(mktemp -d)"
trap 'rm -rf "$WORK"' EXIT

ok=0; fail=0
step() { printf '\n\033[1m== %s\033[0m\n' "$1"; }
pass() { printf '  \033[32mPASS\033[0m %s\n' "$1"; ok=$((ok+1)); }
bad()  { printf '  \033[31mFAIL\033[0m %s\n' "$1"; fail=$((fail+1)); }

if [ ! -f "$GGUF" ]; then echo "no such file: $GGUF" >&2; exit 2; fi
echo "target: $GGUF ($(stat -c%s "$GGUF") bytes)"

step "1/6 zig build test"
if "$ZIG" build test; then pass "unit tests"; else bad "unit tests"; fi

step "2/6 zig build"
if "$ZIG" build; then pass "build"; else bad "build"; exit 1; fi

step "3/6 reference dump via gguf-py"
if "$PY" tests/gen_ref.py "$GGUF" "$WORK/ref.txt" >/dev/null; then
  pass "gen_ref.py"
else
  bad "gen_ref.py"; exit 1
fi

step "4/6 zig dump (--full)"
if ./zig-out/bin/qwen35 "$GGUF" --full > "$WORK/zig.txt" 2>"$WORK/zig.err"; then
  pass "qwen35"
else
  bad "qwen35"; sed -n '1,5p' "$WORK/zig.err"; exit 1
fi
grep -E '^(version|tensor_count|metadata_kv_count|alignment|data_offset) ' "$WORK/zig.txt" | sed 's/^/    /'
grep -E '^(check|sum_aligned_bytes) ' "$WORK/zig.txt" | sed 's/^/    /'

step "5/6 structural comparison"
if "$PY" tests/compare.py "$WORK/zig.txt" "$WORK/ref.txt"; then
  pass "header + metadata + all tensors identical"
else
  bad "structural comparison"
fi

step "6/6 huge tokenizer arrays, element-for-element"
if "$PY" tests/verify_big_arrays.py "$WORK/zig.txt" "$GGUF" 2>&1 | grep -v '^   id '; then
  if "$PY" tests/verify_big_arrays.py "$WORK/zig.txt" "$GGUF" >/dev/null 2>&1; then
    pass "tokens / merges / token_type"
  else
    bad "tokens / merges / token_type"
  fi
else
  bad "tokens / merges / token_type"
fi

printf '\n\033[1m%s\033[0m  (%d passed, %d failed)\n' \
  "$([ "$fail" -eq 0 ] && echo 'ALL GREEN' || echo 'FAILURES')" "$ok" "$fail"
exit "$([ "$fail" -eq 0 ] && echo 0 || echo 1)"
