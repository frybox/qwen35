#!/usr/bin/env python3
"""Regenerate src/simd/tables.zig from the ggml table files.

src/simd/ is compiled once per ISA tier: build.zig feeds the compiler a
generated directory that contains ONLY src/simd/*.zig plus a generated
tier.zig, so a kernel there cannot `@import("../ggml_tables.zig")`.  The IQ
codebooks therefore live in src/simd/tables.zig, a mechanical mirror of
src/ggml_tables.zig + src/ggml_tables_extra.zig.

Run this after touching either ggml_tables file:

    python3 scripts/gen_simd_tables.py
    python3 scripts/gen_simd_tables.py --check   # fail if tables.zig is stale
"""

import pathlib
import re
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
SRC = ROOT / "src"
SOURCES = ("ggml_tables.zig", "ggml_tables_extra.zig")
DEST = SRC / "simd" / "tables.zig"

HEADER = """// AUTO-GENERATED from src/ggml_tables.zig + src/ggml_tables_extra.zig.
// DO NOT EDIT -- regenerate with scripts/gen_simd_tables.py.
//
// src/simd/ is compiled once per ISA tier by build.zig (addCopyDirectory of
// that directory only), so kernels there cannot @import("../ggml_tables.zig").
// The IQ codebooks are mirrored here so every tier sees the same tables.
// Regenerate after touching the ggml_tables* files.
"""

# `pub const NAME: [N]T =.{ ... };` -- the array bodies are multi-line.
DECL = re.compile(r"^pub const (\w+): (\[[^\]]+\]\w+) = \.\{\n(.*?)^\};\n", re.S | re.M)


def render() -> str:
    out = [HEADER]
    seen = []
    for f in SOURCES:
        text = (SRC / f).read_text()
        for name, ty, body in DECL.findall(text):
            seen.append(name)
            out.append("\n/// %s (mirror of %s)\npub const %s: %s = .{\n%s};\n" % (name, f, name, ty, body))
    if not seen:
        raise SystemExit("no table declarations found in %s" % (SOURCES,))
    return "".join(out)


def main() -> int:
    want = render()
    check = "--check" in sys.argv
    have = DEST.read_text() if DEST.exists() else ""
    if check:
        if have != want:
            print("stale: %s -- run scripts/gen_simd_tables.py" % DEST, file=sys.stderr)
            return 1
        print("%s is up to date" % DEST)
        return 0
    DEST.write_text(want)
    print("wrote %s (%d bytes)" % (DEST, len(want)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
