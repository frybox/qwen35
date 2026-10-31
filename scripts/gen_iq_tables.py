#!/usr/bin/env python3
"""Regenerate src/iq_tables.zig from the upstream ggml table definitions.

`src/iq_tables.zig` is the single authoritative *definition* of the IQ lookup
tables in this repository, and it is a generated file.  There is no C header
carrying the data any more: the twelve canonical tables are written here as Zig
array literals declared `pub export const`, so the same symbol is what

  * the Zig host code reads (it is the ordinary Zig declaration), and
  * the x86 C kernels link against: src/kernels/x86/iq_tables.h declares each
    table `extern const T[n]` and the linker resolves it to this `pub export`.

Why `pub export` and not plain `pub`: the C kernels need a global symbol with
the C spelling of the name.  Why an explicit `[N]u64` and not `[_]u64`: the
latter is an anonymous array literal whose type is a *slice*, and exporting a
slice emits a 16-byte fat pointer instead of the table.  The length is therefore
always written out.

Derived tables (keven_signs_q2xs, ksigns64_derived, k32_iq4nl_tab,
k32_nvfp4_values) are emitted below the data as comptime expressions of the
canonical tables, not as copied bytes.  That is what makes the cross-checks in
src/iq_table_tests.zig meaningful: they compare two independently-derived forms
instead of a copy against itself.

CUDA is NOT served by this file (device code cannot reference host objects); a
later task moves the device-side codebooks onto their own definition.

Usage, after touching the upstream file:

    python3 scripts/gen_iq_tables.py            # rewrite src/iq_tables.zig
    python3 scripts/gen_iq_tables.py --check    # fail if it is stale

The upstream path defaults to the local llama.cpp checkout; override with
--source when that checkout moves.
"""

import argparse
import hashlib
import pathlib
import re
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
DEST = ROOT / "src" / "iq_tables.zig"
DEFAULT_SOURCE = pathlib.Path("/mnt/llm-dev/llama.cpp/ggml/src/ggml-common.h")

# name -> (C type, note).  Order is the order emitted in the generated file, and
# is the order the kernels' extern declarations use.
TABLES = [
    ("kmask_iq2xs", "uint8_t", "bit j of the byte -> element j set to -k"),
    ("ksigns_iq2xs", "uint8_t", "sign bytes; mask form consumed by the scalar path"),
    ("ksigns64", "uint64_t", "same masks packed as one u64, 0xFF == -1"),
    ("iq2xxs_grid", "uint64_t", "IQ2_XXS codebook"),
    ("iq2xs_grid", "uint64_t", "IQ2_XS codebook"),
    ("iq2s_grid", "uint64_t", "IQ2_S codebook"),
    ("iq3xxs_grid", "uint32_t", "IQ3_XXS codebook"),
    ("iq3s_grid", "uint32_t", "IQ3_S codebook"),
    ("kvalues_iq4nl", "int8_t", "IQ4_NL nibble values"),
    ("kvalues_fp4", "int8_t", "NVFP4/MXFP4 nibble values (kvalues_mxfp4 is an alias)"),
    ("iq1s_grid", "uint64_t", "IQ1_S codebook"),
    # Not a GGML_TABLE: upstream declares it as a local array inside
    # ggml-cpu/quants.c:493.  Kept here so the Q3_K kernels have one source too.
    ("iq_pow3", "uint8_t", "source ggml-cpu/quants.c:493 (uint8_t pow3[6])"),
]

# Tables upstream does not expose via GGML_TABLE_BEGIN, so they cannot be parsed
# out of ggml-common.h.  Values verified against the upstream source; the anchor
# is recorded so a future upstream change is a diff, not a guess.
LOCAL_TABLES = {
    "iq_pow3": (6, [1, 3, 9, 27, 81, 243]),
}

CONSTANTS = {
    "NGRID_IQ1S": 2048,
}

# Upstream C type -> the Zig type of the exported array.  `u8`/`i8`/`u32`/`u64`
# are the exact-width integers, so the ABI matches the C kernels' extern
# declarations byte for byte.
ZIG_TYPES = {
    "uint8_t": "u8",
    "int8_t": "i8",
    "uint32_t": "u32",
    "uint64_t": "u64",
}

PREAMBLE = """//! IQ-series lookup tables: THE single definition of the data in this repository.
//!
//! GENERATED FILE -- do not edit by hand.  Regenerate with
//!
//!     python3 scripts/gen_iq_tables.py
//!
//! (and `--check` to verify it is up to date).  The generator reads the
//! upstream ggml definitions listed in the anchors at the bottom of this file.
//!
//! There is exactly one copy of the bytes: the `pub export const` arrays below.
//! `pub export` is deliberate -- it gives each table a global symbol under its C
//! name, which is how the x86 kernels in src/kernels/x86/ reach it: they declare
//! `extern const T[n]` (src/kernels/x86/iq_tables.h, declarations only) and the
//! linker resolves that to the object defined here.  Do not turn these into
//! plain `pub` (the C side would stop linking) or into `[_]T` (that is a slice:
//! export emits a 16-byte fat pointer instead of the table).
//!
//! The layout is part of the contract with the C side too: `extern const
//! uint64_t iq1s_grid[2048]` must see the same element type and length, so the
//! array type here is always written with an explicit length in the mapped
//! integer type.
//!
//! The derived tables at the bottom (keven_signs_q2xs, ksigns64_derived,
//! k32_iq4nl_tab, k32_nvfp4_values) are NOT stored data: they are comptime
//! functions of the canonical tables above, exported under the names the C
//! kernels use.  Keeping them derived is what lets src/iq_table_tests.zig
//! cross-check one form against another instead of comparing a copy with
//! itself.

"""

DATA_HEADER = """// ===========================================================================
// Canonical tables (generated from the anchors at the bottom of this file)
// ===========================================================================

"""

DERIVED = """
// ===========================================================================
// Derived tables
// ===========================================================================
// Pure functions of the canonical tables above, computed at comptime and
// exported under the names the C kernels declare `extern` in
// src/kernels/x86/iq_tables.h.  Nothing here is a second copy of the data:
// if a canonical table changes, these change with it, and src/iq_table_tests.zig
// is what catches an "independent" derivation that disagrees.

/// Sign expansion of `ksigns_iq2xs`: bit j of byte i set -> -1, else +1.
/// This is the form the SIMD sign-application step wants (int8 +-1 lanes);
/// `ksigns_iq2xs` is the bitmask form the scalar path wants.
pub export const keven_signs_q2xs: [1024]i8 = blk: {
    @setEvalBranchQuota(100_000);
    var out: [1024]i8 = undefined;
    for (ksigns_iq2xs, 0..) |mask, i| {
        for (0..8) |b| {
            out[i * 8 + b] = if (mask & (@as(u8, 1) << @as(u3, @intCast(b))) != 0) -1 else 1;
        }
    }
    break :blk out;
};

/// `ksigns_iq2xs` widened to one u64 per mask, 0xFF bytes for -1.
/// Must equal `ksigns64`; the test suite asserts exactly that, so the two
/// independent derivations police each other.
pub export const ksigns64_derived: [128]u64 = blk: {
    @setEvalBranchQuota(100_000);
    var out: [128]u64 = undefined;
    for (ksigns_iq2xs, 0..) |mask, i| {
        var w: u64 = 0;
        for (0..8) |b| {
            if (mask & (@as(u8, 1) << @as(u3, @intCast(b))) != 0) w |= @as(u64, 0xFF) << @as(u6, @intCast(b * 8));
        }
        out[i] = w;
    }
    break :blk out;
};

/// int32 mirror of the IQ4_NL nibble values (SIMD kernels index these with
/// 0..15 and want a 32-bit lane, not a sign-extended byte load).
pub export const k32_iq4nl_tab: [16]i32 = blk: {
    var out: [16]i32 = undefined;
    for (kvalues_iq4nl, 0..) |v, i| out[i] = v;
    break :blk out;
};

/// f32 view of the FP4 nibble values (mxfp4/nvfp4 activation scaling).
pub export const k32_nvfp4_values: [16]f32 = blk: {
    var out: [16]f32 = undefined;
    for (kvalues_fp4, 0..) |v, i| out[i] = @floatFromInt(v);
    break :blk out;
};

// ===========================================================================
// Comptime self-check
// ===========================================================================
// Cheap invariants that catch a mangled extraction at compile time instead of
// at the first wrong dot product.  The per-table cross-checks against the
// kernels live in src/iq_table_tests.zig; these are the ones that fit in a
// comptime block.

comptime {
    // The loops below cross-check ~1.2k entries; each iteration costs several
    // backward branches, so raise the comptime quota explicitly.  (Without this
    // the "+-1 / mask" cross-check trips "evaluation exceeded 1000 backwards
    // branches".)
    @setEvalBranchQuota(200_000);
    // Element counts the kernels index into (iq3s_grid[qs | 256] needs 512).
    if (kmask_iq2xs.len != 8) @compileError("kmask_iq2xs: wrong length");
    if (ksigns_iq2xs.len != 128) @compileError("ksigns_iq2xs: wrong length");
    if (ksigns64.len != 128) @compileError("ksigns64: wrong length");
    if (iq2xxs_grid.len != 256) @compileError("iq2xxs_grid: wrong length");
    if (iq2xs_grid.len != 512) @compileError("iq2xs_grid: wrong length");
    if (iq2s_grid.len != 1024) @compileError("iq2s_grid: wrong length");
    if (iq3xxs_grid.len != 256) @compileError("iq3xxs_grid: wrong length");
    if (iq3s_grid.len != 512) @compileError("iq3s_grid: wrong length");
    if (kvalues_iq4nl.len != 16) @compileError("kvalues_iq4nl: wrong length");
    if (kvalues_fp4.len != 16) @compileError("kvalues_fp4: wrong length");
    if (iq1s_grid.len != 2048) @compileError("iq1s_grid: wrong length");
    if (iq_pow3.len != 6) @compileError("iq_pow3: wrong length");

    // kmask_iq2xs is 1 << i; anything else silently corrupts the IQ2_XXS sign
    // application.
    for (kmask_iq2xs, 0..) |m, i| {
        if (m != @as(u8, 1) << @as(u3, @intCast(i))) @compileError("kmask_iq2xs: not a bit mask");
    }
    // The derived iq4_nl mirror must be the byte table widened, not a second
    // hand-written list.
    for (kvalues_iq4nl, 0..) |v, i| {
        if (k32_iq4nl_tab[i] != v) @compileError("k32_iq4nl_tab disagrees with kvalues_iq4nl");
    }
    for (kvalues_fp4, 0..) |v, i| {
        if (k32_nvfp4_values[i] != @as(f32, @floatFromInt(v))) @compileError("k32_nvfp4_values disagrees with kvalues_fp4");
    }
    // sign forms: keven_signs_q2xs / ksigns64_derived must agree with the mask
    // table and with each other.
    for (ksigns_iq2xs, 0..) |mask, i| {
        var w: u64 = 0;
        for (0..8) |b| {
            const neg = mask & (@as(u8, 1) << @as(u3, @intCast(b))) != 0;
            if (keven_signs_q2xs[i * 8 + b] != (if (neg) @as(i8, -1) else 1)) @compileError("keven_signs_q2xs disagrees with ksigns_iq2xs");
            if (neg) w |= @as(u64, 0xFF) << @as(u6, @intCast(b * 8));
        }
        if (ksigns64_derived[i] != w) @compileError("ksigns64_derived disagrees with ksigns_iq2xs");
    }
}

// NOTE: `ksigns64_derived` vs the upstream `ksigns64` is checked at *runtime* in
// src/iq_table_tests.zig, not here -- that comparison is the point of a test
// (it proves the upstream table and this derivation agree), and an equality
// between two 128-word comptime tables is not something the compiler should
// have to prove on every build.
"""

FOOTER_TEMPLATE = """
// ===========================================================================
// Source anchors (recorded from the upstream files this was generated from)
// ===========================================================================
// Source: {source}
// Source sha256: {sha}
//
// Upstream anchors (ggml-common.h line numbers):
{anchors}
"""


def extract(source: pathlib.Path):
    text = source.read_text()
    out = {}
    for m in re.finditer(
        r"GGML_TABLE_BEGIN\((\w+), (\w+), ([A-Za-z0-9_]+)\)\n(.*?)\nGGML_TABLE_END",
        text,
        re.S,
    ):
        ctype, name, size = m.group(1), m.group(2), m.group(3)
        body = m.group(4)
        vals = [int(v, 16) if v.startswith("0x") else int(v) for v in
                re.findall(r"0x[0-9a-fA-F]+|-?\d+", body)]
        declared = CONSTANTS.get(size, None)
        if declared is not None and len(vals) != declared:
            sys.exit(f"{name}: expected {declared} values, parsed {len(vals)}")
        out[name] = (ctype, len(vals), vals)
    return out


def line_numbers(source: pathlib.Path):
    out = []
    for i, line in enumerate(source.read_text().splitlines(), 1):
        m = re.match(r"GGML_TABLE_BEGIN\(\w+, (\w+), \w+\)", line)
        if m:
            out.append((m.group(1), i))
    return out


def fmt(v: int, ctype: str) -> str:
    """Format one value as a Zig literal.

    64-bit grids are hex (0x0011223344556677 is a codebook word, not a
    number), 32-bit grids are hex, and the small signed byte tables are decimal
    because that is how a human checks kvalues_iq4nl (and Zig's i8 takes
    `-127` as written; no suffix, the destination type is explicit).
    """
    if ctype == "uint64_t":
        return f"0x{v:016x}"
    if ctype == "uint32_t":
        return f"0x{v:08x}"
    return str(v)


def zig_type(ctype: str) -> str:
    try:
        return ZIG_TYPES[ctype]
    except KeyError:
        sys.exit(f"no Zig type mapping for C type {ctype}")


def render(tables, source: pathlib.Path) -> str:
    sha = hashlib.sha256(source.read_bytes()).hexdigest()
    lines = dict(line_numbers(source))
    anchors = []
    for name, _ctype, note in TABLES:
        if name in lines:
            anchors.append(f"//   {name}: ggml-common.h line {lines[name]}")
        else:
            anchors.append(f"//   {name}: {note}")

    out = [PREAMBLE, DATA_HEADER]

    for name, ctype, note in TABLES:
        if name in LOCAL_TABLES:
            n, vals = LOCAL_TABLES[name]
        else:
            if name not in tables:
                sys.exit(f"table {name} not found in {source}")
            _upstream_ctype, n, vals = tables[name]
        anchor = f"ggml-common.h:{lines[name]}" if name in lines else note
        out.append(
            f"/// {note}. {n} entries ({ctype}).\n"
            f"/// Source: {anchor}\n"
            f"pub export const {name}: [{n}]{zig_type(ctype)} align(64) = .{{\n"
        )
        # Leading four spaces so the numbers line up the way upstream does; 8 per
        # line keeps the diff readable and the file under a reasonable width.
        for i in range(0, n, 8):
            out.append("    " + ", ".join(fmt(v, ctype) for v in vals[i:i + 8]) + ",\n")
        out.append("};\n\n")

    out.append(DERIVED)
    out.append(FOOTER_TEMPLATE.replace("{source}", str(source))
               .replace("{sha}", sha)
               .replace("{anchors}", "\n".join(anchors)))
    return "".join(out)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", type=pathlib.Path, default=DEFAULT_SOURCE)
    ap.add_argument("--check", action="store_true", help="fail if the generated file is stale")
    args = ap.parse_args()

    if not args.source.exists():
        sys.exit(f"upstream source not found: {args.source}")

    want = render(extract(args.source), args.source)
    if args.check:
        have = DEST.read_text() if DEST.exists() else ""
        if have != want:
            sys.exit(f"{DEST} is stale: regenerate with scripts/gen_iq_tables.py")
        print(f"{DEST} is up to date")
        return 0
    DEST.write_text(want)
    print(f"wrote {DEST} ({len(want)} bytes)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
