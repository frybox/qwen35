//! Qwen3.5 byte-level BPE tokenizer, ported from ref/c/tokenizer.c.
//!
//! Pipeline (matches HF tokenizer.json for `tokenizer.ggml.pre = "qwen35"`):
//!   text -> NFC normalize -> split on special tokens -> GPT-2 style
//!   pre-tokenization (7-branch state machine) -> byte-to-GPT2-unicode
//!   mapping -> greedy lowest-rank BPE merges -> vocab ids.
//!
//! Decoding is the inverse: id -> vocab string -> GPT-2 unicode back to
//! bytes -> concatenate. All vocab/merges data is borrowed from the parsed
//! GGUF (zero-copy); this struct owns only its hash maps and scratch.
//!
//! GPT-2 byte<->unicode mapping: bytes 0x21..0x7E and 0xA1..0xFF map to
//! themselves; every other byte b maps to codepoint 256+n where n is the
//! index of b in the (ascending) list of non-printable bytes.

const std = @import("std");
const nfc = @import("nfc_table.zig");
const gguf = @import("gguf.zig");

pub const TokenizerError = error{
    MissingTokenizerData,
    InvalidVocab,
    InvalidMerge,
    UnknownByteToken,
    TextTooLong,
};

/// Special tokens, pinned to the ids from Qwen3.5 tokenizer_config.json
/// added_tokens_decoder (verified against the GGUF vocab at 248044..248071).
const special_tokens = [_]struct { tok: []const u8, id: u32 }{
    .{ .tok = "<|endoftext|>", .id = 248044 },
    .{ .tok = "<|im_start|>", .id = 248045 },
    .{ .tok = "<|im_end|>", .id = 248046 },
    .{ .tok = "<|object_ref_start|>", .id = 248047 },
    .{ .tok = "<|object_ref_end|>", .id = 248048 },
    .{ .tok = "<|box_start|>", .id = 248049 },
    .{ .tok = "<|box_end|>", .id = 248050 },
    .{ .tok = "<|quad_start|>", .id = 248051 },
    .{ .tok = "<|quad_end|>", .id = 248052 },
    .{ .tok = "<|vision_start|>", .id = 248053 },
    .{ .tok = "<|vision_end|>", .id = 248054 },
    .{ .tok = "<|vision_pad|>", .id = 248055 },
    .{ .tok = "<|image_pad|>", .id = 248056 },
    .{ .tok = "<|video_pad|>", .id = 248057 },
    .{ .tok = "<tool_call>", .id = 248058 },
    .{ .tok = "</tool_call>", .id = 248059 },
    .{ .tok = "<|fim_prefix|>", .id = 248060 },
    .{ .tok = "<|fim_middle|>", .id = 248061 },
    .{ .tok = "<|fim_suffix|>", .id = 248062 },
    .{ .tok = "<|fim_pad|>", .id = 248063 },
    .{ .tok = "<|repo_name|>", .id = 248064 },
    .{ .tok = "<|file_sep|>", .id = 248065 },
    .{ .tok = "<tool_response>", .id = 248066 },
    .{ .tok = "</tool_response>", .id = 248067 },
    .{ .tok = "<think>", .id = 248068 },
    .{ .tok = "</think>", .id = 248069 },
    .{ .tok = "<|audio_start|>", .id = 248070 },
    .{ .tok = "<|audio_end|>", .id = 248071 },
};

/// GPT-2 byte -> unicode codepoint table (built lazily, see byteToUnicode).
var gpt2_table: [256]u21 = undefined;
var gpt2_table_built = false;

fn isGpt2Printable(b: usize) bool {
    // GPT-2 bytes_to_unicode: 0x21..0x7E, 0xA1..0xAC, 0xAE..0xFF (0xAD excluded).
    return (b >= 0x21 and b <= 0x7E) or (b >= 0xA1 and b <= 0xAC) or (b >= 0xAE and b <= 0xFF);
}

/// The non-printable bytes (everything outside the GPT-2 printable set),
/// in ascending order; byte b maps to 256 + index_in_this_list.
const non_printable = blk: {
    var list: [68]u8 = undefined; // 33 (0x00-0x20) + 34 (0x7F-0xA0) + 1 (0xAD)
    var n: usize = 0;
    var b: usize = 0;
    while (b < 256) : (b += 1) {
        const printable = isGpt2Printable(b);
        if (!printable) {
            list[n] = @intCast(b);
            n += 1;
        }
    }
    break :blk list;
};

fn buildGpt2Table() void {
    for (0..256) |b| {
        const printable = isGpt2Printable(b);
        gpt2_table[b] = if (printable) @intCast(b) else @intCast(256 + blk: {
            for (non_printable, 0..) |np, i| {
                if (np == b) break :blk i;
            }
            unreachable;
        });
    }
    gpt2_table_built = true;
}

pub fn byteToUnicode(b: u8) u21 {
    if (!gpt2_table_built) buildGpt2Table();
    return gpt2_table[b];
}

/// Inverse of byteToUnicode; null when `cp` is not a GPT-2 mapped codepoint.
pub fn unicodeToByte(cp: u21) ?u8 {
    if (!gpt2_table_built) buildGpt2Table();
    for (gpt2_table, 0..) |mapped, b| {
        if (mapped == cp) return @intCast(b);
    }
    return null;
}

// ---------------------------------------------------------------------------
// NFC normalization (ported from tokenizer.c:640-671; tables in nfc_table.zig)
// ---------------------------------------------------------------------------

/// Codepoints that appear as the *second* element of some canonical composition
/// pair, i.e. the marks that can actually compose onto a preceding character.
/// Derived from `nfc.pairs` at comptime, then sorted for binary search.
///
/// This is deliberately NOT the same as the `\p{M}` predicate in
/// `isCombiningMark`: many canonical combiners are neither in the hand-written
/// `\p{M}` ranges nor meant to join letter runs during pre-tokenization —
/// Hangul V/T jamo (U+1161..U+1175, U+11A8..U+11C2) and Indic vowel signs such
/// as U+09BE/U+0BBE/U+0BC6 are the important cases. Using `isCombiningMark`
/// here left 11 277 of the 12 113 composition pairs unreachable (Hangul 11 172,
/// Indic 105), so decomposed Hangul/Jamo and Tamil/Bengali/Malayalam/Telugu/
/// Oriya text tokenized differently from HF.
const canonical_combiners = blk: {
    @setEvalBranchQuota(1 << 21);
    var all: [nfc.pairs.len]u21 = undefined;
    for (nfc.pairs, 0..) |row, i| all[i] = @intCast(row[1]);
    std.mem.sort(u21, &all, {}, std.sort.asc(u21));

    // Deduplicate in place.
    var uniq: [nfc.pairs.len]u21 = undefined;
    var n: usize = 0;
    for (all) |c| {
        if (n == 0 or uniq[n - 1] != c) {
            uniq[n] = c;
            n += 1;
        }
    }

    var out: [n]u21 = undefined;
    @memcpy(&out, uniq[0..n]);
    break :blk out;
};

/// True if `cp` can canonically compose onto a preceding character.
fn isCanonicalCombiner(cp: u21) bool {
    const t = &canonical_combiners;
    var lo: usize = 0;
    var hi: usize = t.len;
    while (lo < hi) {
        const mid = (lo + hi) / 2;
        if (t[mid] < cp) {
            lo = mid + 1;
        } else if (t[mid] > cp) {
            hi = mid;
        } else return true;
    }
    return false;
}

fn nfcLookupSingle(cp: u21) ?u21 {
    const t = &nfc.single;
    var lo: usize = 0;
    var hi: usize = t.len;
    while (lo < hi) {
        const mid = (lo + hi) / 2;
        const row = t[mid];
        if (row[0] < cp) lo = mid + 1 else if (row[0] > cp) hi = mid else return @intCast(row[1]);
    }
    return null;
}

fn nfcLookupPair(starter: u21, combining: u21) ?u21 {
    const t = &nfc.pairs;
    var lo: usize = 0;
    var hi: usize = t.len;
    while (lo < hi) {
        const mid = (lo + hi) / 2;
        const row = t[mid];
        if (row[0] < starter or (row[0] == starter and row[1] < combining)) {
            lo = mid + 1;
        } else if (row[0] > starter or (row[0] == starter and row[1] > combining)) {
            hi = mid;
        } else return @intCast(row[2]);
    }
    return null;
}

fn isCombiningMark(cp: u21) bool {
    return (cp >= 0x0300 and cp <= 0x036F) or
        (cp >= 0x0483 and cp <= 0x0489) or
        (cp >= 0x0591 and cp <= 0x05BD) or
        (cp >= 0x0610 and cp <= 0x061A) or
        (cp >= 0x064B and cp <= 0x065F) or
        cp == 0x0670 or
        (cp >= 0x06D6 and cp <= 0x06DC) or
        (cp >= 0x0900 and cp <= 0x0903) or
        (cp >= 0x093A and cp <= 0x094F) or
        (cp >= 0x0951 and cp <= 0x0957) or
        (cp >= 0x0E31 and cp <= 0x0E3A) or
        (cp >= 0x0E47 and cp <= 0x0E4E) or
        (cp >= 0x1AB0 and cp <= 0x1AFF) or
        (cp >= 0x1DC0 and cp <= 0x1DFF) or
        (cp >= 0x20D0 and cp <= 0x20F0) or
        (cp >= 0xFE20 and cp <= 0xFE2F);
}

/// NFC normalize: single-codepoint replacement + canonical composition of
/// adjacent starter+combiner pairs.
///
/// Known deviation from full NFC: there is NO canonical-ordering (CCC) pass, so
/// inputs carrying several combining marks on one base — e.g. `A` + U+0300 +
/// U+0323, which NFC reorders to `Ạ` + U+0300 — are composed in input order
/// instead. Single-mark text (the overwhelming majority) is exact. Fixing this
/// properly needs a canonical combining class table, which is not in
/// `nfc_table.zig`.
pub fn nfcNormalize(allocator: std.mem.Allocator, text: []const u8) ![]u8 {
    // Decode + single replacement. Invalid UTF-8 is tolerated here: like the
    // reference `utf8_to_unicode`, each bad byte degrades to its own value as a
    // codepoint, so dirty input is normalized (and then tokenized) instead of
    // trapping. (`Utf8View.initUnchecked` + `catch unreachable` used to panic on
    // such input; `decodeAt` is the same degradation already used by the
    // pre-tokenizer.)
    var cps = try std.ArrayList(u21).initCapacity(allocator, text.len);
    defer cps.deinit(allocator);
    var pos: usize = 0;
    while (pos < text.len) {
        const d = decodeAt(text, pos);
        pos += d.len;
        try cps.append(allocator, nfcLookupSingle(d.cp) orelse d.cp);
    }

    // Composition: each canonical combiner merges into the nearest preceding
    // character (which may itself already be a composed char, e.g. Hangul LV).
    var m: usize = 0;
    for (cps.items) |cp_in| {
        var cp = cp_in;
        if (m > 0 and isCanonicalCombiner(cp)) {
            var j: usize = m - 1;
            while (true) {
                if (nfcLookupPair(cps.items[j], cp)) |composed| {
                    cps.items[j] = composed;
                    cp = 0;
                    break;
                }
                // Keep walking back over other combiners (Hangul LV+T chains,
                // mark-on-mark) until a starter that cannot combine is reached.
                if (!isCanonicalCombiner(cps.items[j])) break;
                if (j == 0) break;
                j -= 1;
            }
            if (cp == 0) continue;
        }
        cps.items[m] = cp;
        m += 1;
    }

    // Re-encode.
    var out = try std.ArrayList(u8).initCapacity(allocator, 4 * m);
    errdefer out.deinit(allocator);
    for (cps.items[0..m]) |cp| {
        var buf: [4]u8 = undefined;
        const n = std.unicode.utf8Encode(cp, &buf) catch continue;
        try out.appendSlice(allocator, buf[0..n]);
    }
    return out.toOwnedSlice(allocator);
}

// ---------------------------------------------------------------------------
// Pre-tokenization (ported from tokenizer.c:686-913)
// ---------------------------------------------------------------------------

const CharClass = enum { letter, number, space, newline, other };

fn classifyChar(cp: u21) CharClass {
    if (cp == '\r' or cp == '\n') return .newline;
    if (cp == ' ' or cp == '\t' or cp == 0x0B or cp == 0x0C) return .space;
    if (cp < 0x80) {
        if ((cp >= 'a' and cp <= 'z') or (cp >= 'A' and cp <= 'Z')) return .letter;
        if (cp >= '0' and cp <= '9') return .number;
        return .other;
    }
    // \p{L} ranges (same coverage as ref).
    if ((cp >= 0x00C0 and cp <= 0x024F) or
        (cp >= 0x0370 and cp <= 0x04FF) or
        (cp >= 0x0590 and cp <= 0x06FF) or
        (cp >= 0x0900 and cp <= 0x0DFF) or
        (cp >= 0x0E00 and cp <= 0x0EFF) or
        (cp >= 0x1000 and cp <= 0x109F) or
        (cp >= 0x1100 and cp <= 0x11FF) or
        (cp >= 0x1E00 and cp <= 0x1FFF) or
        (cp >= 0x2C60 and cp <= 0x2CDF) or
        (cp >= 0x3040 and cp <= 0x30FF) or
        (cp >= 0x3130 and cp <= 0x318F) or
        (cp >= 0x3400 and cp <= 0x4DBF) or
        (cp >= 0x4E00 and cp <= 0x9FFF) or
        (cp >= 0xA000 and cp <= 0xA4CF) or
        (cp >= 0xAC00 and cp <= 0xD7AF) or
        (cp >= 0xF900 and cp <= 0xFAFF) or
        (cp >= 0x1E000 and cp <= 0x1EFFF) or
        (cp >= 0x20000 and cp <= 0x2FA1F)) return .letter;
    // \p{M} combining marks also join letter runs.
    if (isCombiningMark(cp)) return .letter;
    // \p{N} digits (incl. fullwidth, roman numerals, CJK counts).
    if ((cp >= 0x0660 and cp <= 0x0669) or
        (cp >= 0x2160 and cp <= 0x2182) or // roman numerals I..XIII
        (cp >= 0x3007 and cp <= 0x3007) or // 〇
        (cp >= 0x3021 and cp <= 0x3029) or // hangzhou numerals
        (cp >= 0x3038 and cp <= 0x303A) or
        (cp >= 0x06F0 and cp <= 0x06F9) or
        (cp >= 0x0966 and cp <= 0x096F) or
        (cp >= 0x09E6 and cp <= 0x09EF) or
        (cp >= 0x0A66 and cp <= 0x0A6F) or
        (cp >= 0x0BE6 and cp <= 0x0BEF) or
        (cp >= 0x0C66 and cp <= 0x0C6F) or
        (cp >= 0x0E50 and cp <= 0x0E59) or
        (cp >= 0xFF10 and cp <= 0xFF19)) return .number;
    // Remaining Unicode whitespace.
    if (cp == 0x00A0 or (cp >= 0x2000 and cp <= 0x200A) or
        cp == 0x2028 or cp == 0x2029 or cp == 0x202F or cp == 0x205F or
        cp == 0x3000 or cp == 0x1680) return .space;
    return .other;
}

/// Decode the UTF-8 codepoint at `s[i]`; on invalid encoding, treat the byte
/// as a latin-1 codepoint (matches ref behavior of classifying raw bytes).
fn decodeAt(s: []const u8, i: usize) struct { cp: u21, len: usize } {
    const n = std.unicode.utf8ByteSequenceLength(s[i]) catch return .{ .cp = s[i], .len = 1 };
    if (i + n > s.len) return .{ .cp = s[i], .len = 1 };
    const cp = std.unicode.utf8Decode(s[i .. i + n]) catch return .{ .cp = s[i], .len = 1 };
    return .{ .cp = cp, .len = n };
}

// ---------------------------------------------------------------------------
// Tokenizer
// ---------------------------------------------------------------------------

pub const Tokenizer = struct {
    allocator: std.mem.Allocator,
    /// id -> vocab string (GPT-2 unicode form); slices point into the GGUF.
    vocab: [][]const u8,
    /// "left right" -> rank. Keys point into the GGUF merges array.
    merge_rank: std.StringHashMapUnmanaged(u32) = .empty,
    /// Exact token string -> id (GPT-2 unicode form for normal tokens).
    token_to_id: std.StringHashMapUnmanaged(u32) = .empty,
    /// Special token id, or null when not found at this position.
    special_map: std.StringHashMapUnmanaged(u32) = .empty,
    vocab_size: u32,

    /// Load vocab + merges + special tokens from a parsed GGUF.
    /// `file` must stay alive (and its data stable) for this Tokenizer's life.
    pub fn initFromGguf(allocator: std.mem.Allocator, file: *const gguf.File) !Tokenizer {
        const kv_tokens = file.findKey("tokenizer.ggml.tokens") orelse return error.MissingTokenizerData;
        const kv_merges = file.findKey("tokenizer.ggml.merges") orelse return error.MissingTokenizerData;
        const tokens_arr = switch (kv_tokens.*) {
            .array => |a| a,
            else => return error.InvalidVocab,
        };
        const merges_arr = switch (kv_merges.*) {
            .array => |a| a,
            else => return error.InvalidMerge,
        };
        if (tokens_arr.element_type != .string or merges_arr.element_type != .string) {
            return error.InvalidVocab;
        }

        var vocab = try allocator.alloc([]const u8, tokens_arr.count);
        errdefer allocator.free(vocab);

        // Fill vocab from the GGUF string array.
        var tr = gguf.arrayReader(file, tokens_arr);
        for (0..tokens_arr.count) |i| {
            vocab[i] = try tr.readString();
        }

        var self = Tokenizer{
            .allocator = allocator,
            .vocab = vocab,
            .vocab_size = @intCast(tokens_arr.count),
        };
        errdefer {
            self.merge_rank.deinit(allocator);
            self.token_to_id.deinit(allocator);
            self.special_map.deinit(allocator);
        }

        // token_to_id for every non-NULL vocab entry.
        for (vocab, 0..) |tok, i| {
            if (tok.len == 0) continue;
            try self.token_to_id.put(allocator, tok, @intCast(i));
        }

        // merge_rank: "left right" -> index (lower = earlier = higher priority).
        var mr = gguf.arrayReader(file, merges_arr);
        for (0..merges_arr.count) |i| {
            const pair = try mr.readString();
            try self.merge_rank.put(allocator, pair, @intCast(i));
        }

        // Special tokens: verify id against vocab when the slot is non-empty,
        // but trust our pinned table (matches ref behavior).
        // Special tokens: encode side uses the pinned id table (matches ref).
        // Decode side needs the string in the vocab at the right slot, so a
        // GGUF whose vocab is too small for our ids is rejected up front
        // rather than silently producing holes.
        for (special_tokens) |st| {
            if (st.id >= self.vocab_size) return error.InvalidVocab;
            self.special_map.put(allocator, st.tok, st.id) catch return error.OutOfMemory;
            vocab[st.id] = st.tok;
        }

        return self;
    }

    pub fn deinit(self: *Tokenizer) void {
        self.allocator.free(self.vocab);
        self.merge_rank.deinit(self.allocator);
        self.token_to_id.deinit(self.allocator);
        self.special_map.deinit(self.allocator);
    }

    pub fn getId(self: *const Tokenizer, token: []const u8) ?u32 {
        return self.token_to_id.get(token);
    }

    pub fn getToken(self: *const Tokenizer, id: u32) ?[]const u8 {
        if (id >= self.vocab_size) return null;
        return self.vocab[id];
    }

    /// Length of the longest special token that starts at s[0], if any.
    fn specialAt(self: *const Tokenizer, s: []const u8) ?usize {
        var best: ?usize = null;
        var it = self.special_map.keyIterator();
        while (it.next()) |key| {
            const k = key.*;
            if (s.len >= k.len and std.mem.startsWith(u8, s, k)) {
                if (best == null or k.len > best.?) best = k.len;
            }
        }
        return best;
    }

    /// Cut one pre-token starting at s[0]; returns byte length.
    /// Caller loops until s is exhausted.
    fn nextPretoken(self: *const Tokenizer, s: []const u8) usize {
        if (s.len == 0) return 0;
        if (self.specialAt(s)) |sp| return sp;

        const c0 = decodeAt(s, 0);
        const cls0 = classifyChar(c0.cp);

        // Branch 1: 's|'t|'re|'ve|'m|'ll|'d (case-insensitive).
        if (c0.cp == '\'' and s.len > c0.len) {
            const sufs = [_][]const u8{ "s", "t", "re", "ve", "m", "ll", "d" };
            for (sufs) |suf| {
                if (s.len >= c0.len + suf.len) {
                    var match = true;
                    for (suf, 0..) |sc, j| {
                        const c = s[c0.len + j];
                        const lc = if (c >= 'A' and c <= 'Z') c + 32 else c;
                        if (lc != sc) {
                            match = false;
                            break;
                        }
                    }
                    if (match) return c0.len + suf.len;
                }
            }
        }

        // Branch 2: [^\r\n\p{L}\p{N}]?[\p{L}\p{M}]+
        if (cls0 != .letter and cls0 != .number and cls0 != .newline) {
            if (s.len > c0.len) {
                const c1 = decodeAt(s, c0.len);
                if (classifyChar(c1.cp) == .letter) {
                    var q = c0.len + c1.len;
                    while (q < s.len) {
                        const d = decodeAt(s, q);
                        if (classifyChar(d.cp) != .letter) break;
                        if (self.specialAt(s[q..])) |_| break;
                        q += d.len;
                    }
                    return q;
                }
            }
        }

        // Branch 2b: pure letter word.
        if (cls0 == .letter) {
            var q = c0.len;
            while (q < s.len) {
                if (self.specialAt(s[q..])) |_| break;
                const d = decodeAt(s, q);
                if (classifyChar(d.cp) != .letter) break;
                q += d.len;
            }
            return q;
        }

        // Branch 3: single number.
        if (cls0 == .number) return c0.len;

        // Branch 4: " ?[^\s\p{L}\p{M}\p{N}]+[\r\n]*
        if (cls0 == .other or (cls0 == .space and c0.cp == ' ')) {
            var q: usize = if (cls0 == .space) c0.len else 0;
            if (q < s.len) {
                const c1 = decodeAt(s, q);
                if (classifyChar(c1.cp) == .other) {
                    while (q < s.len) {
                        if (self.specialAt(s[q..])) |_| break;
                        const d = decodeAt(s, q);
                        if (classifyChar(d.cp) != .other) break;
                        q += d.len;
                    }
                    while (q < s.len) {
                        if (self.specialAt(s[q..])) |_| break;
                        const d = decodeAt(s, q);
                        if (d.cp != '\r' and d.cp != '\n') break;
                        q += d.len;
                    }
                    return q;
                }
            }
            if (cls0 == .space) {
                // Space followed by non-punct: fall through to whitespace.
            }
        }

        // Branches 5/6/7: whitespace runs.
        if (cls0 == .space or cls0 == .newline) {
            var q = c0.len;
            var last_nl_end: ?usize = null;
            while (q < s.len) {
                const d = decodeAt(s, q);
                const cc = classifyChar(d.cp);
                if (cc == .newline) {
                    last_nl_end = q + d.len;
                    q += d.len;
                } else if (cc == .space) {
                    q += d.len;
                } else break;
            }
            // Whitespace run ends right before a special token: cut here.
            if (self.specialAt(s[q..]) != null) return q;
            // Branch 5: \s*[\r\n]+ up to the end of the last newline run.
            if (last_nl_end) |e| return e;
            // Branch 6: \s+(?!\S) — yield the last space if followed by \S.
            if (q < s.len) {
                if (q >= 2) return q - 1;
                return q;
            }
            return q;
        }

        // Fallback: single character.
        return c0.len;
    }

    /// Greedy lowest-rank BPE over one pre-token's bytes; appends ids.
    fn bpeEncodeBytes(self: *const Tokenizer, allocator: std.mem.Allocator, bytes: []const u8, out: *std.ArrayList(u32)) !void {
        if (bytes.len == 0) return;
        // Arena for all strings created during this pre-token's BPE. The
        // parts list references slices into `str_arena`, so everything must
        // live until the end of the function — a resettable arena gives that
        // for free and invalidates nothing while parts are being merged.
        var arena_state = std.heap.ArenaAllocator.init(allocator);
        defer arena_state.deinit();
        const arena = arena_state.allocator();

        var parts = try std.ArrayList([]const u8).initCapacity(arena, bytes.len);
        for (bytes) |b| {
            var buf: [4]u8 = undefined;
            const n = std.unicode.utf8Encode(byteToUnicode(b), &buf) catch unreachable;
            const s = try arena.dupe(u8, buf[0..n]);
            parts.appendAssumeCapacity(s);
        }

        // Merge loop: repeatedly merge the adjacent pair with the lowest rank.
        while (parts.items.len > 1) {
            var best_rank: ?u32 = null;
            var best_idx: usize = 0;
            for (0..parts.items.len - 1) |i| {
                var key_buf: [128]u8 = undefined;
                if (parts.items[i].len + parts.items[i + 1].len + 1 > key_buf.len) continue;
                @memcpy(key_buf[0..parts.items[i].len], parts.items[i]);
                key_buf[parts.items[i].len] = ' ';
                @memcpy(key_buf[parts.items[i].len + 1 ..][0..parts.items[i + 1].len], parts.items[i + 1]);
                const key = key_buf[0 .. parts.items[i].len + 1 + parts.items[i + 1].len];
                if (self.merge_rank.get(key)) |rank| {
                    if (best_rank == null or rank < best_rank.?) {
                        best_rank = rank;
                        best_idx = i;
                    }
                }
            }
            if (best_rank == null) break;

            // Merge: concatenation of the two parts' strings.
            const merged = try std.mem.concat(arena, u8, &.{ parts.items[best_idx], parts.items[best_idx + 1] });
            parts.items[best_idx] = merged;

            // Shift left.
            var i = best_idx + 1;
            while (i < parts.items.len - 1) : (i += 1) {
                parts.items[i] = parts.items[i + 1];
            }
            parts.items.len -= 1;
        }

        // Map final parts to ids; fall back to per-codepoint lookup.
        for (parts.items) |p| {
            if (self.token_to_id.get(p)) |id| {
                try out.append(allocator, id);
            } else {
                var j: usize = 0;
                while (j < p.len) {
                    const n = std.unicode.utf8ByteSequenceLength(p[j]) catch 1;
                    const end = @min(j + n, p.len);
                    if (self.token_to_id.get(p[j..end])) |id| {
                        try out.append(allocator, id);
                    }
                    j = end;
                }
            }
        }
    }

    /// Encode text (chat template already applied by caller, or raw text) to
    /// token ids. Returned slice is owned by `allocator`.
    pub fn encode(self: *const Tokenizer, allocator: std.mem.Allocator, text: []const u8) ![]u32 {
        var ids: std.ArrayList(u32) = .empty;
        errdefer ids.deinit(allocator);

        const normed = try nfcNormalize(allocator, text);
        defer allocator.free(normed);

        var pos: usize = 0;
        while (pos < normed.len) {
            const rest = normed[pos..];
            // Special token?
            if (self.specialAt(rest)) |sp| {
                const id = self.special_map.get(rest[0..sp]).?;
                try ids.append(allocator, id);
                pos += sp;
                continue;
            }
            const seg_len = self.nextPretoken(rest);
            if (seg_len == 0) break;
            try self.bpeEncodeBytes(allocator, rest[0..seg_len], &ids);
            pos += seg_len;
        }
        return ids.toOwnedSlice(allocator);
    }

    /// Decode ids to text; invalid ids are skipped (matches ref).
    /// Returned slice is owned by `allocator`.
    pub fn decode(self: *const Tokenizer, allocator: std.mem.Allocator, ids: []const u32) ![]u8 {
        var out: std.ArrayList(u8) = .empty;
        errdefer out.deinit(allocator);
        for (ids) |id| {
            const tok = self.getToken(id) orelse continue;
            var it = std.unicode.Utf8View.initUnchecked(tok).iterator();
            while (it.nextCodepoint()) |cp| {
                if (unicodeToByte(cp)) |b| {
                    try out.append(allocator, b);
                } else {
                    var buf: [4]u8 = undefined;
                    const n = std.unicode.utf8Encode(cp, &buf) catch unreachable;
                    try out.appendSlice(allocator, buf[0..n]);
                }
            }
        }
        return out.toOwnedSlice(allocator);
    }

    // -----------------------------------------------------------------------
    // Chat template (subset for plain text chats; mirrors ref engine L880-883
    // and the non-tools path of the GGUF chat_template).
    // -----------------------------------------------------------------------

    /// Render one user turn + generation prompt, non-thinking mode:
    ///   <|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n
    pub fn renderChatTurn(allocator: std.mem.Allocator, user_text: []const u8, thinking: bool) ![]u8 {
        var out: std.ArrayList(u8) = .empty;
        errdefer out.deinit(allocator);
        try out.appendSlice(allocator, "<|im_start|>user\n");
        try out.appendSlice(allocator, user_text);
        try out.appendSlice(allocator, "<|im_end|>\n<|im_start|>assistant\n");
        try out.appendSlice(allocator, if (thinking) "<think>\n" else "<think>\n\n</think>\n\n");
        return out.toOwnedSlice(allocator);
    }
};

// ---------------------------------------------------------------------------
// Tests: build a mini GGUF with a hand-written vocab/merges and exercise the
// full encode/decode pipeline against std.testing.allocator (which fails the
// test on any leak or double free).
// ---------------------------------------------------------------------------

const testing = std.testing;

const TestBuilder = struct {
    buf: std.ArrayList(u8) = .empty,

    fn u32le(self: *TestBuilder, v: u32) !void {
        try self.buf.appendSlice(testing.allocator, std.mem.asBytes(&v));
    }
    fn u64le(self: *TestBuilder, v: u64) !void {
        try self.buf.appendSlice(testing.allocator, std.mem.asBytes(&v));
    }
    fn str(self: *TestBuilder, v: []const u8) !void {
        try self.u64le(v.len);
        try self.buf.appendSlice(testing.allocator, v);
    }
    fn kvStringArray(self: *TestBuilder, k: []const u8, vs: []const []const u8) !void {
        try self.str(k);
        try self.u32le(9); // ARRAY
        try self.u32le(8); // STRING elements
        try self.u64le(vs.len);
        for (vs) |v| try self.str(v);
    }
};

// Vocab: 0..255 are the single GPT-2 chars (byte-level fallback), then a few
// merges. We build the vocab exactly the way real converters do: the first
// 256 entries are the byte tokens in byte order, then merged tokens.
const test_words = [_][]const u8{ "hello", "world", "the", " quick" };
const test_merges = [_][2][]const u8{
    .{ "h", "e" },       .{ "he", "l" },      .{ "hel", "l" },  .{ "hell", "o" },
    .{ "w", "o" },       .{ "wo", "r" },      .{ "wor", "l" },  .{ "worl", "d" },
    .{ "t", "h" },       .{ "th", "e" },      .{ "Ġ", "q" },    .{ "Ġq", "u" },
    .{ "Ġqu", "i" },     .{ "Ġqui", "c" },    .{ "Ġquic", "k" },
};

fn buildTestVocab(b: *TestBuilder, vocab: *std.ArrayList([]const u8)) !void {
    _ = b;
    // byte tokens 0..255 as GPT-2 unicode strings
    for (0..256) |byte| {
        const cp = byteToUnicode(@intCast(byte));
        var tmp: [4]u8 = undefined;
        const n = std.unicode.utf8Encode(cp, &tmp) catch unreachable;
        try vocab.append(testing.allocator, try testing.allocator.dupe(u8, tmp[0..n]));
    }
    for (test_words) |w| try vocab.append(testing.allocator, try testing.allocator.dupe(u8, w));
    for (test_merges) |m| {
        const merged = try std.mem.concat(testing.allocator, u8, &.{ m[0], m[1] });
        try vocab.append(testing.allocator, merged);
    }
    // Fill up to the highest pinned special id (248071) with empty strings so
    // the pinned special tokens fit the mini vocab exactly like a real one.
    while (vocab.items.len <= 248071) try vocab.append(testing.allocator, try testing.allocator.dupe(u8, ""));
}

test "encode/decode round trip through a mini gguf" {
    const a = testing.allocator;

    var vocab_list: std.ArrayList([]const u8) = .empty;
    defer {
        for (vocab_list.items) |v| a.free(v);
        vocab_list.deinit(a);
    }
    try buildTestVocab(undefined, &vocab_list);

    var b = TestBuilder{};
    defer b.buf.deinit(a);
    try b.buf.appendSlice(a, "GGUF");
    try b.u32le(3);
    try b.u64le(0); // tensors
    try b.u64le(2); // kvs
    try b.kvStringArray("tokenizer.ggml.tokens", vocab_list.items);

    // merges as "left right" strings
    var merge_strs: std.ArrayList([]const u8) = .empty;
    defer {
        for (merge_strs.items) |m| a.free(m);
        merge_strs.deinit(a);
    }
    for (test_merges) |m| {
        try merge_strs.append(a, try std.mem.concat(a, u8, &.{ m[0], " ", m[1] }));
    }
    try b.kvStringArray("tokenizer.ggml.merges", merge_strs.items);

    // no tensors: data section is just padding to alignment
    const alignment: u64 = 32;
    while (b.buf.items.len % alignment != 0) try b.buf.append(a, 0);

    var g = try gguf.parse(a, b.buf.items);
    defer g.deinit();

    var tok = try Tokenizer.initFromGguf(a, &g);
    defer tok.deinit();

    try testing.expectEqual(@as(u32, @intCast(vocab_list.items.len)), tok.vocab_size);

    // "hello" should merge all the way to the single "hello" token (id 256).
    {
        const ids = try tok.encode(a, "hello");
        defer a.free(ids);
        try testing.expectEqual(@as(usize, 1), ids.len);
        // merged "hello" token: 256 bytes + words(4) + 4 merges before it = 263? 
        // words: hello=256,world=257,the=258," quick"=259; merges: he=260, hel=261, hell=262, hello=263
        try testing.expectEqual(@as(u32, 263), ids[0]);
    }
    // " world" (leading space): no Ġ+world merge exists -> ["Ġ", "world"].
    {
        const ids = try tok.encode(a, " world");
        defer a.free(ids);
        try testing.expectEqual(@as(usize, 2), ids.len);
        try testing.expectEqualStrings("Ġ", tok.getToken(ids[0]).?);
        try testing.expectEqualStrings("world", tok.getToken(ids[1]).?);
        // merged "world" token = 256 bytes + words(4) + merges up to it
        try testing.expectEqual(@as(u32, 267), ids[1]);
    }
    // Decode round trip.
    {
        const text = "hello world, the quick\tbrown!";
        const ids = try tok.encode(a, text);
        defer a.free(ids);
        const decoded = try tok.decode(a, ids);
        defer a.free(decoded);
        try testing.expectEqualStrings(text, decoded);
    }
    // Special tokens are recognized inside text and survive decode.
    {
        const text = "a<|im_end|>b";
        const ids = try tok.encode(a, text);
        defer a.free(ids);
        // 'a' + special + 'b' = 3 tokens, middle is 248046
        try testing.expectEqual(@as(usize, 3), ids.len);
        try testing.expectEqual(@as(u32, 248046), ids[1]);
        const decoded = try tok.decode(a, ids);
        defer a.free(decoded);
        try testing.expectEqualStrings(text, decoded);
    }
    // UTF-8 multibyte round trip (CJK: 3 bytes each, byte-level fallback).
    {
        const text = "你好，世界！";
        const ids = try tok.encode(a, text);
        defer a.free(ids);
        const decoded = try tok.decode(a, ids);
        defer a.free(decoded);
        try testing.expectEqualStrings(text, decoded);
    }
}

test "nfc compose: decomposed e-acute becomes single codepoint" {
    const a = testing.allocator;
    // e + U+0301 -> é (U+00E9)
    const decomposed = "e\u{0301}";
    const normed = try nfcNormalize(a, decomposed);
    defer a.free(normed);
    try testing.expectEqualStrings("é", normed);
    // Already-composed text passes through unchanged.
    const composed = "é";
    const normed2 = try nfcNormalize(a, composed);
    defer a.free(normed2);
    try testing.expectEqualStrings(composed, normed2);
}

test "nfc compose: Hangul jamo L+V and L+V+T compose to syllables" {
    const a = testing.allocator;
    // U+1100 U+1161 -> U+AC00 (가). This regressed while the composition pass
    // used the hand-written \p{M} ranges: U+1161 was not listed there, so the
    // 11 172 Hangul pairs in the table were unreachable.
    const jamo = "\u{1100}\u{1161}";
    const normed = try nfcNormalize(a, jamo);
    defer a.free(normed);
    try testing.expectEqualStrings("가", normed);

    // U+1100 U+1161 U+11A8 -> U+AC01 (각)
    const lvt = "\u{1100}\u{1161}\u{11A8}";
    const normed_lvt = try nfcNormalize(a, lvt);
    defer a.free(normed_lvt);
    try testing.expectEqualStrings("각", normed_lvt);

    // U+AC00 U+11A8 -> U+AC01 (LV syllable + trailing jamo)
    const lv_t = "\u{AC00}\u{11A8}";
    const normed_lv_t = try nfcNormalize(a, lv_t);
    defer a.free(normed_lv_t);
    try testing.expectEqualStrings("각", normed_lv_t);

    // The jamo are canonical combiners but NOT \p{M}-classified by
    // isCombiningMark; the two predicates must stay separate.
    try testing.expect(isCanonicalCombiner(0x1161));
    try testing.expect(isCanonicalCombiner(0x11A8));
    try testing.expect(!isCombiningMark(0x1161));
}

test "nfc compose: Indic vowel signs compose" {
    const a = testing.allocator;
    // Bengali U+09C7 + U+09BE -> U+09CB
    const bengali = "\u{09C7}\u{09BE}";
    const normed = try nfcNormalize(a, bengali);
    defer a.free(normed);
    try testing.expectEqualStrings("\u{09CB}", normed);

    // Tamil U+0BC6 + U+0BBE -> U+0BCA
    const tamil = "\u{0BC6}\u{0BBE}";
    const normed_tamil = try nfcNormalize(a, tamil);
    defer a.free(normed_tamil);
    try testing.expectEqualStrings("\u{0BCA}", normed_tamil);

    // Hebrew U+05D0 + U+05B7 -> U+FB2E is *not* a composition pair, and
    // U+05D0+U+05BC must be left alone (no canonical composition exists).
    const hebrew = "\u{05D0}\u{05BC}";
    const normed_hebrew = try nfcNormalize(a, hebrew);
    defer a.free(normed_hebrew);
    try testing.expectEqualStrings(hebrew, normed_hebrew);
}

test "nfc: text already in NFC is byte-identical" {
    const a = testing.allocator;
    // A sample of already-composed strings across scripts must survive
    // toNFC() unchanged; this is the property that keeps real-world text
    // byte-stable through the tokenizer.
    const samples = [_][]const u8{
        "héllo",     "café",       "Ångström",  "Việt Nam",
        "日本語",       "한국어",       "가각갂",      "தமிழ்",
        "বাংলা",      "Ελληνικά",   "Русский",   " עברית",
        "caf\u{00E9}", "\u{AC01}",   "\u{1E00}",
    };
    for (samples) |s| {
        const n = try nfcNormalize(a, s);
        defer a.free(n);
        try testing.expectEqualStrings(s, n);
    }
}

test "nfc: invalid UTF-8 degrades byte-by-byte instead of panicking" {
    const a = testing.allocator;
    // Regression: nfcNormalize used to run `Utf8View.initUnchecked` +
    // `nextCodepoint() catch unreachable`, so a lone continuation byte trapped.
    // Now it mirrors the reference `utf8_to_unicode`: each undecodable byte
    // becomes its own latin-1 codepoint, which re-encodes to two UTF-8 bytes
    // (0x80 -> 0xC2 0x80).
    const cases = [_]struct { in: []const u8, out: []const u8 }{
        .{ .in = "\x80", .out = "\xC2\x80" },
        .{ .in = "\xFF\xFE\xFD", .out = "\xC3\xBF\xC3\xBE\xC3\xBD" },
        .{ .in = "\xE4\xB8", .out = "\xC3\xA4\xC2\xB8" }, // truncated 3-byte seq
        .{ .in = "\xC3\x28", .out = "\xC3\x83\x28" }, // bad continuation
        .{ .in = "ok\x80ok", .out = "ok\xC2\x80ok" }, // valid text around a bad byte
    };
    for (cases) |c| {
        const n = try nfcNormalize(a, c.in);
        defer a.free(n);
        try testing.expectEqualStrings(c.out, n);
    }
}

test "gpt2 byte mapping is a bijection" {
    var seen = std.AutoHashMap(u21, void).init(testing.allocator);
    defer seen.deinit();
    for (0..256) |byte| {
        const cp = byteToUnicode(@intCast(byte));
        try testing.expect(!seen.contains(cp));
        try seen.put(cp, {});
        try testing.expectEqual(@as(u8, @intCast(byte)), unicodeToByte(cp).?);
    }
    try testing.expectEqual(@as(u21, 0x0120), byteToUnicode(0x20)); // space -> Ġ
    try testing.expectEqual(@as(u21, 0x0142), byteToUnicode(0xA0)); // NBSP -> ł
    try testing.expect(unicodeToByte('中') == null); // CJK is not a GPT-2 mapped codepoint
}

test "chat template renders non-thinking turn" {
    const a = testing.allocator;
    const rendered = try Tokenizer.renderChatTurn(a, "hi", false);
    defer a.free(rendered);
    try testing.expectEqualStrings(
        "<|im_start|>user\nhi<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n",
        rendered,
    );
}
