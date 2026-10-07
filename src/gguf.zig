//! Minimal, dependency-free GGUF v2/v3 reader.
//!
//! Layout: [Header 24B][KV metadata][Tensor info][pad to alignment][Tensor data]
//! See https://github.com/ggml-org/ggml/blob/master/docs/gguf.md
//!
//! Design rules baked in (learned the hard way, see tests/README.md):
//!   * magic doubles as endianness marker: 'GGUF' little, 'FUGG' big.
//!   * unknown `ggml_type` is a HARD ERROR. Some real files use ids beyond
//!     GGML_TYPE_COUNT; silently assuming a width desyncs the tensor-info
//!     region and produces garbage that is very hard to trace back.
//!   * metadata may be mixed precision (F32 norms + Q8_0 weights), so byte
//!     size is always computed per tensor from its own type.
//!   * `dims` are innermost-first (opposite of PyTorch).
//!   * tensor `offset` is relative to the start of the data section.
//!   * the identity `data_offset + sum(align_up(bytes)) == file_size` is
//!     asserted after parsing; it catches almost every header bug at once.

const std = @import("std");
const Io = std.Io;

pub const default_alignment: u64 = 32;

pub const GgufError = error{
    BadMagic,
    UnsupportedVersion,
    StringTooLong,
    InvalidAlignment,
    UnknownValueType,
    UnknownTensorType,
    UnexpectedEof,
    DuplicateKey,
    SizeMismatch,
};

pub const Version = enum(u32) {
    v2 = 2,
    v3 = 3,
};

/// Value types used in the KV section. Values match the on-disk u32 tags.
pub const ValueType = enum(u32) {
    uint8 = 0,
    int8 = 1,
    uint16 = 2,
    int16 = 3,
    uint32 = 4,
    int32 = 5,
    float32 = 6,
    boolean = 7,
    string = 8,
    array = 9,
    uint64 = 10,
    int64 = 11,
    float64 = 12,

    pub fn name(self: ValueType) []const u8 {
        return switch (self) {
            .uint8 => "UINT8",
            .int8 => "INT8",
            .uint16 => "UINT16",
            .int16 => "INT16",
            .uint32 => "UINT32",
            .int32 => "INT32",
            .float32 => "FLOAT32",
            .boolean => "BOOL",
            .string => "STRING",
            .array => "ARRAY",
            .uint64 => "UINT64",
            .int64 => "INT64",
            .float64 => "FLOAT64",
        };
    }

    /// Element width in bytes, or null for variable-length types.
    pub fn byteWidth(self: ValueType) ?u64 {
        return switch (self) {
            .uint8, .int8, .boolean => 1,
            .uint16, .int16 => 2,
            .uint32, .int32, .float32 => 4,
            .uint64, .int64, .float64 => 8,
            .string, .array => null,
        };
    }
};

/// ggml tensor storage type. Values are the on-disk u32 tags.
pub const TensorType = enum(u32) {
    f32 = 0,
    f16 = 1,
    q4_0 = 2,
    q4_1 = 3,
    q5_0 = 6,
    q5_1 = 7,
    q8_0 = 8,
    q8_1 = 9,
    q2_k = 10,
    q3_k = 11,
    q4_k = 12,
    q5_k = 13,
    q6_k = 14,
    q8_k = 15,
    iq2_xxs = 16,
    iq2_xs = 17,
    iq3_xxs = 18,
    iq1_s = 19,
    iq4_nl = 20,
    iq3_s = 21,
    iq2_s = 22,
    iq4_xs = 23,
    i8 = 24,
    i16 = 25,
    i32 = 26,
    i64 = 27,
    f64 = 28,
    iq1_m = 29,
    bf16 = 30,
    tq1_0 = 34,
    tq2_0 = 35,
    mxfp4 = 39,
    nvfp4 = 40,
    q1_0 = 41,
    q2_0 = 42,
    _,

    pub fn name(self: TensorType) []const u8 {
        return switch (self) {
            .f32 => "F32",
            .f16 => "F16",
            .q4_0 => "Q4_0",
            .q4_1 => "Q4_1",
            .q5_0 => "Q5_0",
            .q5_1 => "Q5_1",
            .q8_0 => "Q8_0",
            .q8_1 => "Q8_1",
            .q2_k => "Q2_K",
            .q3_k => "Q3_K",
            .q4_k => "Q4_K",
            .q5_k => "Q5_K",
            .q6_k => "Q6_K",
            .q8_k => "Q8_K",
            .iq2_xxs => "IQ2_XXS",
            .iq2_xs => "IQ2_XS",
            .iq3_xxs => "IQ3_XXS",
            .iq1_s => "IQ1_S",
            .iq4_nl => "IQ4_NL",
            .iq3_s => "IQ3_S",
            .iq2_s => "IQ2_S",
            .iq4_xs => "IQ4_XS",
            .i8 => "I8",
            .i16 => "I16",
            .i32 => "I32",
            .i64 => "I64",
            .f64 => "F64",
            .iq1_m => "IQ1_M",
            .bf16 => "BF16",
            .tq1_0 => "TQ1_0",
            .tq2_0 => "TQ2_0",
            .mxfp4 => "MXFP4",
            .nvfp4 => "NVFP4",
            .q1_0 => "Q1_0",
            .q2_0 => "Q2_0",
            _ => "UNKNOWN",
        };
    }

    /// (block_element_count, block_byte_size) — from ggml's
    /// `ggml_type_size` / `ggml_blck_size`. QK_K = 256.
    pub fn blockLayout(self: TensorType) ?struct { elems: u64, bytes: u64 } {
        return switch (self) {
            .f32 => .{ .elems = 1, .bytes = 4 },
            .f16 => .{ .elems = 1, .bytes = 2 },
            .bf16 => .{ .elems = 1, .bytes = 2 },
            .f64 => .{ .elems = 1, .bytes = 8 },
            .i8 => .{ .elems = 1, .bytes = 1 },
            .i16 => .{ .elems = 1, .bytes = 2 },
            .i32 => .{ .elems = 1, .bytes = 4 },
            .i64 => .{ .elems = 1, .bytes = 8 },
            .q4_0 => .{ .elems = 32, .bytes = 18 },
            .q4_1 => .{ .elems = 32, .bytes = 20 },
            .q5_0 => .{ .elems = 32, .bytes = 22 },
            .q5_1 => .{ .elems = 32, .bytes = 24 },
            .q8_0 => .{ .elems = 32, .bytes = 34 },
            .q8_1 => .{ .elems = 32, .bytes = 36 },
            .iq4_nl => .{ .elems = 32, .bytes = 18 },
            .mxfp4 => .{ .elems = 32, .bytes = 17 },
            .nvfp4 => .{ .elems = 64, .bytes = 36 },
            .q1_0 => .{ .elems = 128, .bytes = 18 },
            .q2_0 => .{ .elems = 64, .bytes = 18 },
            .q2_k => .{ .elems = 256, .bytes = 84 },
            .q3_k => .{ .elems = 256, .bytes = 110 },
            .q4_k => .{ .elems = 256, .bytes = 144 },
            .q5_k => .{ .elems = 256, .bytes = 176 },
            .q6_k => .{ .elems = 256, .bytes = 210 },
            .q8_k => .{ .elems = 256, .bytes = 292 },
            .iq2_xxs => .{ .elems = 256, .bytes = 66 },
            .iq2_xs => .{ .elems = 256, .bytes = 74 },
            .iq3_xxs => .{ .elems = 256, .bytes = 98 },
            .iq1_s => .{ .elems = 256, .bytes = 50 },
            .iq3_s => .{ .elems = 256, .bytes = 110 },
            .iq2_s => .{ .elems = 256, .bytes = 82 },
            .iq4_xs => .{ .elems = 256, .bytes = 136 },
            .iq1_m => .{ .elems = 256, .bytes = 56 },
            .tq1_0 => .{ .elems = 256, .bytes = 54 },
            .tq2_0 => .{ .elems = 256, .bytes = 66 },
            _ => null,
        };
    }

    pub fn isKnown(self: TensorType) bool {
        return self.blockLayout() != null;
    }
};

/// A parsed KV value. Strings and arrays borrow from the mapped file.
pub const Value = union(ValueType) {
    uint8: u8,
    int8: i8,
    uint16: u16,
    int16: i16,
    uint32: u32,
    int32: i32,
    float32: f32,
    boolean: bool,
    string: []const u8,
    array: Array,
    uint64: u64,
    int64: i64,
    float64: f64,

    pub const Array = struct {
        element_type: ValueType,
        count: u64,
        /// Elements are read lazily; we only keep the raw region descriptor.
        raw_offset: usize,
    };
};

pub const KeyValue = struct {
    key: []const u8,
    value: Value,
};

pub const Tensor = struct {
    name: []const u8,
    ttype: TensorType,
    /// Innermost dimension first. On disk these are u64.
    dims: []const u64,
    /// Byte offset relative to the start of the data section.
    offset: u64,
    /// Byte size of the payload, computed from `ttype` and `dims`.
    nbytes: u64,

    pub fn nElems(self: Tensor) u64 {
        var n: u64 = 1;
        for (self.dims) |d| n *= d;
        return n;
    }
};

/// Cursor over a byte slice with bounds-checked little-endian reads.
pub const ByteReader = struct {
    buf: []const u8,
    pos: usize = 0,

    pub fn init(buf: []const u8) ByteReader {
        return .{ .buf = buf };
    }

    pub fn remaining(self: *const ByteReader) usize {
        return self.buf.len - self.pos;
    }

    pub fn readBytes(self: *ByteReader, n: usize) GgufError![]const u8 {
        if (self.remaining() < n) return error.UnexpectedEof;
        const out = self.buf[self.pos..][0..n];
        self.pos += n;
        return out;
    }

    pub fn readU32(self: *ByteReader) GgufError!u32 {
        const b = try self.readBytes(4);
        return std.mem.readInt(u32, b[0..4], .little);
    }

    pub fn readI32(self: *ByteReader) GgufError!i32 {
        const b = try self.readBytes(4);
        return std.mem.readInt(i32, b[0..4], .little);
    }

    pub fn readU64(self: *ByteReader) GgufError!u64 {
        const b = try self.readBytes(8);
        return std.mem.readInt(u64, b[0..8], .little);
    }

    pub fn readI64(self: *ByteReader) GgufError!i64 {
        const b = try self.readBytes(8);
        return std.mem.readInt(i64, b[0..8], .little);
    }

    pub fn readF32(self: *ByteReader) GgufError!f32 {
        return @bitCast(try self.readU32());
    }

    pub fn readF64(self: *ByteReader) GgufError!f64 {
        return @bitCast(try self.readU64());
    }

    /// GGUF strings are u64 length prefixed, no NUL terminator.
    pub fn readString(self: *ByteReader) GgufError![]const u8 {
        const len = try self.readU64();
        if (len > std.math.maxInt(u32)) return error.StringTooLong;
        const n: usize = @intCast(len);
        return self.readBytes(n);
    }
};

pub const File = struct {
    allocator: std.mem.Allocator,
    /// The whole mapped file, borrowed; not owned.
    data: []const u8,
    version: Version,
    alignment: u64,
    declared_tensor_count: u64,
    declared_kv_count: u64,
    data_offset: u64,
    /// Byte just past the tensor-info region (before alignment padding).
    tensor_info_end: u64,
    kvs: []KeyValue,
    tensors: []Tensor,

    pub fn deinit(self: *File) void {
        for (self.tensors) |t| self.allocator.free(t.dims);
        self.allocator.free(self.tensors);
        self.allocator.free(self.kvs);
    }

    pub fn findKey(self: *const File, key: []const u8) ?*const Value {
        for (self.kvs) |*kv| {
            if (std.mem.eql(u8, kv.key, key)) return &kv.value;
        }
        return null;
    }

    pub fn getString(self: *const File, key: []const u8) ?[]const u8 {
        const v = self.findKey(key) orelse return null;
        return switch (v.*) {
            .string => |s| s,
            else => null,
        };
    }

    pub fn getUint(self: *const File, key: []const u8) ?u64 {
        const v = self.findKey(key) orelse return null;
        return switch (v.*) {
            .uint8 => |x| x,
            .uint16 => |x| x,
            .uint32 => |x| x,
            .uint64 => |x| x,
            .int8 => |x| @intCast(x),
            .int16 => |x| @intCast(x),
            .int32 => |x| @intCast(x),
            .int64 => |x| @intCast(x),
            else => null,
        };
    }

    pub fn getFloat(self: *const File, key: []const u8) ?f64 {
        const v = self.findKey(key) orelse return null;
        return switch (v.*) {
            .float32 => |x| x,
            .float64 => |x| x,
            else => null,
        };
    }

    pub fn findTensor(self: *const File, name: []const u8) ?*const Tensor {
        for (self.tensors) |*t| {
            if (std.mem.eql(u8, t.name, name)) return t;
        }
        return null;
    }

    /// Byte slice of a tensor's payload inside the mapped file.
    pub fn tensorData(self: *const File, t: *const Tensor) GgufError![]const u8 {
        const start: usize = @intCast(self.data_offset + t.offset);
        const len: usize = @intCast(t.nbytes);
        if (start + len > self.data.len) return error.UnexpectedEof;
        return self.data[start..][0..len];
    }

    /// Human-readable dump of everything this reader knows: the summary block,
    /// the whole metadata table and the tensor table. Long arrays (tokenizer
    /// tokens/merges) are sampled so a terminal stays usable; call `printFull`
    /// to dump every element.
    ///
    /// Note that the "which file was opened" line is NOT printed here: a `File`
    /// only knows its bytes, not the path it was mapped from. The CLI prints
    /// that line itself before calling this.
    pub fn print(self: *const File, out: *Io.Writer) !void {
        return self.printImpl(out, false);
    }

    /// Same as `print`, but no array is truncated (the `--full` CLI mode).
    pub fn printFull(self: *const File, out: *Io.Writer) !void {
        return self.printImpl(out, true);
    }

    fn printImpl(self: *const File, out: *Io.Writer, full: bool) !void {
        const file_size: u64 = @intCast(self.data.len);

        // --- summary ----------------------------------------------------------
        try out.print("file_size {d}\n", .{file_size});
        try out.print("version {d}\n", .{@intFromEnum(self.version)});
        try out.print("tensor_count {d}\n", .{self.tensors.len});
        try out.print("metadata_kv_count {d}\n", .{self.kvs.len});
        try out.print("alignment {d}\n", .{self.alignment});
        try out.print("tensor_info_end {d}\n", .{self.tensor_info_end});
        try out.print("data_offset {d}\n", .{self.data_offset});
        {
            var sum: u64 = 0;
            for (self.tensors) |t| sum += std.mem.alignForward(u64, t.nbytes, self.alignment);
            try out.print("sum_aligned_bytes {d}\n", .{sum});
            try out.print("check data_offset+sum == file_size : {s}\n", .{
                if (self.data_offset + sum == file_size) "OK" else "MISMATCH",
            });
        }

        // --- metadata ---------------------------------------------------------
        try out.print("\n=== metadata ({d}) ===\n", .{self.kvs.len});
        for (self.kvs, 0..) |kv, i| {
            try out.print("{d:>3} {s} = ", .{ i, kv.key });
            try self.printValue(out, kv.value, full);
            try out.writeByte('\n');
        }

        // --- tensors ----------------------------------------------------------
        try out.print("\n=== tensors ({d}) ===\n", .{self.tensors.len});
        try out.print("{s: <44} {s: <8} {s: <22} {s: >12} {s: >12} {s: >12}\n", .{
            "name", "type", "dims", "offset", "abs_offset", "bytes",
        });
        try out.print("(offset = raw field, relative to data section; abs = data_offset + offset)\n", .{});
        for (self.tensors) |t| {
            try out.print("{s: <44} {s: <8} ", .{ t.name, t.ttype.name() });
            try printDims(out, t.dims);
            try out.print(" {d: >12} {d: >12} {d: >12}\n", .{
                t.offset, self.data_offset + t.offset, t.nbytes,
            });
        }
    }

    fn printValue(self: *const File, out: *Io.Writer, v: Value, full: bool) !void {
        switch (v) {
            .uint8 => |x| try out.print("{s} {d}", .{ "UINT8", x }),
            .int8 => |x| try out.print("{s} {d}", .{ "INT8", x }),
            .uint16 => |x| try out.print("{s} {d}", .{ "UINT16", x }),
            .int16 => |x| try out.print("{s} {d}", .{ "INT16", x }),
            .uint32 => |x| try out.print("{s} {d}", .{ "UINT32", x }),
            .int32 => |x| try out.print("{s} {d}", .{ "INT32", x }),
            .uint64 => |x| try out.print("{s} {d}", .{ "UINT64", x }),
            .int64 => |x| try out.print("{s} {d}", .{ "INT64", x }),
            .float32 => |x| try out.print("{s} {d}", .{ "FLOAT32", x }),
            .float64 => |x| try out.print("{s} {d}", .{ "FLOAT64", x }),
            .boolean => |x| try out.print("{s} {s}", .{ "BOOL", if (x) "true" else "false" }),
            .string => |s| try printQuoted(out, s, true),
            .array => |arr| {
                try out.print("ARRAY<{s}> len={d}", .{ arr.element_type.name(), arr.count });
                // Small arrays are shown in full; huge ones (tokens/merges, 200k+
                // entries) are sampled so the terminal stays usable. `--full`
                // dumps everything.
                const limit: u64 = if (full or arr.count <= max_inline_array)
                    arr.count
                else
                    max_inline_array;
                if (arr.element_type == .string) {
                    var r = arrayReader(self, arr);
                    var i: u64 = 0;
                    while (i < limit) : (i += 1) {
                        const s = try r.readString();
                        try out.writeByte(' ');
                        try printQuoted(out, s, full);
                    }
                } else {
                    var r = arrayReader(self, arr);
                    var i: u64 = 0;
                    try out.writeByte(' ');
                    try out.writeByte('[');
                    while (i < limit) : (i += 1) {
                        if (i != 0) try out.print(", ", .{});
                        const raw = try arrayElemU64(&r, arr.element_type);
                        try printElem(out, arr.element_type, raw);
                    }
                    try out.writeByte(']');
                }
                if (limit < arr.count) try out.print(" ... ({d} more)", .{arr.count - limit});
            },
        }
    }
};

fn printDims(out: *Io.Writer, dims: []const u64) !void {
    try out.writeByte('[');
    for (dims, 0..) |d, i| {
        if (i != 0) try out.writeByte(',');
        try out.print("{d}", .{d});
    }
    try out.writeByte(']');
}

fn printElem(out: *Io.Writer, t: ValueType, raw: u64) !void {
    switch (t) {
        .float32 => try out.print("{d}", .{@as(f32, @bitCast(@as(u32, @truncate(raw))))}),
        .float64 => try out.print("{d}", .{@as(f64, @bitCast(raw))}),
        .int8 => try out.print("{d}", .{@as(i8, @truncate(@as(i64, @bitCast(raw))))}),
        .int16 => try out.print("{d}", .{@as(i16, @truncate(@as(i64, @bitCast(raw))))}),
        .int32 => try out.print("{d}", .{@as(i32, @truncate(@as(i64, @bitCast(raw))))}),
        .int64 => try out.print("{d}", .{@as(i64, @bitCast(raw))}),
        else => try out.print("{d}", .{raw}),
    }
}

/// Arrays longer than this are sampled unless `--full` is given. Individual
/// strings are always printed whole: the useful metadata strings here are a few
/// KB (chat_template is ~4.1 kB), and truncating them loses real information.
const max_inline_array: u64 = 16;

fn printQuoted(out: *Io.Writer, s: []const u8, full: bool) !void {
    _ = full;
    const show = s.len;
    try out.writeByte('"');
    for (s[0..show]) |ch| {
        switch (ch) {
            '"', '\\' => {
                try out.writeByte('\\');
                try out.writeByte(ch);
            },
            '\n' => try out.writeAll("\\n"),
            '\r' => try out.writeAll("\\r"),
            '\t' => try out.writeAll("\\t"),
            0x20...0x21, 0x23...0x5b, 0x5d...0x7e => try out.writeByte(ch),
            0x7f...0xff => try out.print("\\x{x:0>2}", .{ch}),
            else => try out.print("\\x{x:0>2}", .{ch}),
        }
    }
    try out.writeByte('"');
    if (show < s.len) try out.print(" (len={d})", .{s.len});
}

pub fn parse(allocator: std.mem.Allocator, data: []const u8) (GgufError || std.mem.Allocator.Error)!File {
    var r = ByteReader.init(data);

    // --- header -----------------------------------------------------------
    const magic_bytes = try r.readBytes(4);
    const little = std.mem.eql(u8, magic_bytes, "GGUF");
    const big = std.mem.eql(u8, magic_bytes, "FUGG");
    if (!little and !big) return error.BadMagic;
    // Only little-endian is needed here; reject big-endian explicitly rather
    // than silently swapping every field.
    if (big) return error.BadMagic;

    const version_raw = try r.readU32();
    const version: Version = switch (version_raw) {
        2 => .v2,
        3 => .v3,
        else => return error.UnsupportedVersion,
    };
    const declared_tensor_count = try r.readU64();
    const declared_kv_count = try r.readU64();

    // --- KV metadata ------------------------------------------------------
    const kvs = try allocator.alloc(KeyValue, @intCast(declared_kv_count));
    errdefer allocator.free(kvs);
    var alignment: u64 = default_alignment;
    for (kvs) |*kv| {
        kv.* = try readKeyValue(&r);
        if (std.mem.eql(u8, kv.key, "general.alignment")) {
            alignment = switch (kv.value) {
                .uint32 => |v| v,
                .uint64 => |v| v,
                else => return error.InvalidAlignment,
            };
        }
    }

    // --- tensor info ------------------------------------------------------
    const tensors = try allocator.alloc(Tensor, @intCast(declared_tensor_count));
    errdefer allocator.free(tensors);
    var allocated: usize = 0;
    errdefer for (tensors[0..allocated]) |t| allocator.free(t.dims);
    for (tensors) |*t| {
        const name = try r.readString();
        const n_dims = try r.readU32();
        if (n_dims > 4) return error.UnknownTensorType; // GGML_MAX_DIMS
        const dims = try allocator.alloc(u64, n_dims);
        errdefer allocator.free(dims);
        // NB: dims are u64 on disk (unlike n_dims, which is u32).
        for (dims) |*d| d.* = try r.readU64();
        const ttype_raw = try r.readU32();
        const ttype: TensorType = @enumFromInt(ttype_raw);
        if (!ttype.isKnown()) return error.UnknownTensorType;
        const offset = try r.readU64();
        var n_elems: u64 = 1;
        for (dims) |d| n_elems *= d;
        const layout = ttype.blockLayout().?;
        // Rows must be a whole number of blocks.
        if (n_elems % layout.elems != 0) return error.SizeMismatch;
        t.* = .{
            .name = name,
            .ttype = ttype,
            .dims = dims,
            .offset = offset,
            .nbytes = (n_elems / layout.elems) * layout.bytes,
        };
        allocated += 1;
    }

    // --- alignment + self-check ------------------------------------------
    if (alignment == 0 or (alignment & (alignment - 1)) != 0) return error.InvalidAlignment;
    const tensor_info_end: u64 = @intCast(r.pos);
    const data_offset = std.mem.alignForward(u64, tensor_info_end, alignment);

    var sum_aligned: u64 = 0;
    for (tensors) |t| sum_aligned += std.mem.alignForward(u64, t.nbytes, alignment);
    if (data_offset + sum_aligned != data.len) return error.SizeMismatch;

    return .{
        .allocator = allocator,
        .data = data,
        .version = version,
        .alignment = alignment,
        .declared_tensor_count = declared_tensor_count,
        .declared_kv_count = declared_kv_count,
        .data_offset = data_offset,
        .tensor_info_end = tensor_info_end,
        .kvs = kvs,
        .tensors = tensors,
    };
}

fn readKeyValue(r: *ByteReader) GgufError!KeyValue {
    const key = try r.readString();
    const raw_type = try r.readU32();
    const vt: ValueType = std.enums.fromInt(ValueType, raw_type) orelse
        return error.UnknownValueType;
    return .{ .key = key, .value = try readValue(r, vt) };
}

fn readValue(r: *ByteReader, vt: ValueType) GgufError!Value {
    return switch (vt) {
        .uint8 => .{ .uint8 = (try r.readBytes(1))[0] },
        .int8 => .{ .int8 = @bitCast((try r.readBytes(1))[0]) },
        .uint16 => .{ .uint16 = std.mem.readInt(u16, (try r.readBytes(2))[0..2], .little) },
        .int16 => .{ .int16 = std.mem.readInt(i16, (try r.readBytes(2))[0..2], .little) },
        .uint32 => .{ .uint32 = try r.readU32() },
        .int32 => .{ .int32 = try r.readI32() },
        .float32 => .{ .float32 = try r.readF32() },
        .boolean => .{ .boolean = (try r.readBytes(1))[0] != 0 },
        .string => .{ .string = try r.readString() },
        .uint64 => .{ .uint64 = try r.readU64() },
        .int64 => .{ .int64 = try r.readI64() },
        .float64 => .{ .float64 = try r.readF64() },
        .array => blk: {
            // Arrays are kept as a descriptor; callers iterate via
            // `arrayElemAt`, which re-reads from the raw region.
            const elem_raw = try r.readU32();
            const elem_type = std.enums.fromInt(ValueType, elem_raw) orelse
                return error.UnknownValueType;
            if (elem_type == .array) return error.UnknownValueType; // not supported
            const count = try r.readU64();
            const raw_offset = r.pos;
            // Skip over the payload to advance the cursor.
            switch (elem_type) {
                .string => {
                    var i: u64 = 0;
                    while (i < count) : (i += 1) _ = try r.readString();
                },
                else => {
                    const w = elem_type.byteWidth().?;
                    _ = try r.readBytes(@intCast(count * w));
                },
            }
            break :blk .{ .array = .{
                .element_type = elem_type,
                .count = count,
                .raw_offset = raw_offset,
            } };
        },
    };
}

/// Reader positioned at an array's first element.
pub fn arrayReader(file: *const File, arr: Value.Array) ByteReader {
    return .{ .buf = file.data, .pos = arr.raw_offset };
}

/// Read one array element and return its raw bit pattern widened to u64.
/// Signed values are sign-extended; floats keep their IEEE bits.
pub fn arrayElemU64(r: *ByteReader, elem_type: ValueType) GgufError!u64 {
    return switch (elem_type) {
        .uint8, .boolean => (try r.readBytes(1))[0],
        .int8 => @bitCast(@as(i64, @as(i8, @bitCast((try r.readBytes(1))[0])))),
        .uint16 => std.mem.readInt(u16, (try r.readBytes(2))[0..2], .little),
        .int16 => @bitCast(@as(i64, std.mem.readInt(i16, (try r.readBytes(2))[0..2], .little))),
        .uint32 => try r.readU32(),
        .int32 => @bitCast(@as(i64, try r.readI32())),
        .uint64 => try r.readU64(),
        .int64 => @bitCast(try r.readI64()),
        .float32 => try r.readU32(),
        .float64 => try r.readU64(),
        else => error.UnknownValueType,
    };
}

test "header rejects bad magic" {
    const zeros: [16]u8 = @splat(0);
    const buf = "XXXX\x03\x00\x00\x00" ++ zeros[0..16];
    try std.testing.expectError(error.BadMagic, parse(std.testing.allocator, buf));
}

/// Build a tiny synthetic GGUF in memory so the parser's structural rules
/// (u64 dims, alignment, size identity) are pinned down by tests.
const Builder = struct {
    buf: std.ArrayList(u8) = .empty,

    fn u32le(self: *Builder, v: u32) !void {
        try self.buf.appendSlice(std.testing.allocator, std.mem.asBytes(&v));
    }
    fn u64le(self: *Builder, v: u64) !void {
        try self.buf.appendSlice(std.testing.allocator, std.mem.asBytes(&v));
    }
    fn str(self: *Builder, v: []const u8) !void {
        try self.u64le(v.len);
        try self.buf.appendSlice(std.testing.allocator, v);
    }
    fn kvString(self: *Builder, k: []const u8, v: []const u8) !void {
        try self.str(k);
        try self.u32le(8);
        try self.str(v);
    }
    fn kvU32(self: *Builder, k: []const u8, v: u32) !void {
        try self.str(k);
        try self.u32le(4);
        try self.u32le(v);
    }
    fn kvI32Array(self: *Builder, k: []const u8, vs: []const i32) !void {
        try self.str(k);
        try self.u32le(9); // ARRAY
        try self.u32le(5); // INT32 elements
        try self.u64le(vs.len);
        for (vs) |v| try self.buf.appendSlice(std.testing.allocator, std.mem.asBytes(&v));
    }
    fn tensor(self: *Builder, name: []const u8, dims: []const u64, ttype: u32, off: u64) !void {
        try self.str(name);
        try self.u32le(@intCast(dims.len));
        for (dims) |d| try self.u64le(d);
        try self.u32le(ttype);
        try self.u64le(off);
    }
};

test "parse a minimal hand-built gguf" {
    const a = std.testing.allocator;
    var b = Builder{};
    defer b.buf.deinit(a);

    // header
    try b.buf.appendSlice(a, "GGUF");
    try b.u32le(3);
    try b.u64le(1); // tensors
    try b.u64le(1); // kv
    try b.kvString("general.architecture", "qwen3");
    // one F32 tensor [2,3] = 24 bytes at offset 0
    try b.tensor("token_embd.weight", &.{ 2, 3 }, 0, 0);

    const info_end = b.buf.items.len;
    const alignment: u64 = 32;
    while (b.buf.items.len % alignment != 0) try b.buf.append(a, 0);
    const data_off = b.buf.items.len;
    try std.testing.expectEqual(std.mem.alignForward(u64, info_end, alignment), data_off);
    // One tensor of 24 bytes, padded to 32 like a real file.
    for (0..std.mem.alignForward(u64, 24, alignment)) |_| try b.buf.append(a, 0);

    var g = try parse(a, b.buf.items);
    defer g.deinit();

    try std.testing.expectEqual(@as(u64, 1), g.declared_tensor_count);
    try std.testing.expectEqual(@as(u64, data_off), g.data_offset);
    try std.testing.expectEqualStrings("qwen3", g.getString("general.architecture").?);
    try std.testing.expectEqual(@as(usize, 1), g.tensors.len);
    const t = &g.tensors[0];
    try std.testing.expectEqualStrings("token_embd.weight", t.name);
    try std.testing.expectEqual(TensorType.f32, t.ttype);
    // dims must survive as u64 and keep on-disk (innermost-first) order
    try std.testing.expectEqual(@as(u64, 2), t.dims[0]);
    try std.testing.expectEqual(@as(u64, 3), t.dims[1]);
    try std.testing.expectEqual(@as(u64, 24), t.nbytes);
    const payload = try g.tensorData(t);
    try std.testing.expectEqual(@as(usize, 24), payload.len);
}

test "reject unknown ggml type instead of guessing a width" {
    const a = std.testing.allocator;
    var b = Builder{};
    defer b.buf.deinit(a);
    try b.buf.appendSlice(a, "GGUF");
    try b.u32le(3);
    try b.u64le(1);
    try b.u64le(0);
    try b.tensor("weird", &.{32}, 143, 0); // 143 is not a valid ggml_type
    try std.testing.expectError(error.UnknownTensorType, parse(a, b.buf.items));
}

test "reject a file whose tensors do not tile the data section" {
    const a = std.testing.allocator;
    var b = Builder{};
    defer b.buf.deinit(a);
    try b.buf.appendSlice(a, "GGUF");
    try b.u32le(3);
    try b.u64le(1);
    try b.u64le(0);
    try b.tensor("t", &.{2}, 0, 0); // declares 8 bytes...
    while (b.buf.items.len % 32 != 0) try b.buf.append(a, 0);
    for (0..4) |_| try b.buf.append(a, 0); // ...but the file only holds 4
    try std.testing.expectError(error.SizeMismatch, parse(a, b.buf.items));
}

test "reject invalid alignment" {
    const a = std.testing.allocator;
    var b = Builder{};
    defer b.buf.deinit(a);
    try b.buf.appendSlice(a, "GGUF");
    try b.u32le(3);
    try b.u64le(0);
    try b.u64le(1);
    try b.str("general.alignment");
    try b.u32le(4);
    try b.u32le(24); // not a power of two
    try std.testing.expectError(error.InvalidAlignment, parse(a, b.buf.items));
}

test "block layout covers every type we claim to know" {
    // Spot-check the arithmetic that turns element counts into byte counts.
    try std.testing.expectEqual(@as(u64, 34), TensorType.q8_0.blockLayout().?.bytes);
    try std.testing.expectEqual(@as(u64, 32), TensorType.q8_0.blockLayout().?.elems);
    try std.testing.expectEqual(@as(u64, 256), TensorType.q6_k.blockLayout().?.elems);
    try std.testing.expectEqual(@as(u64, 4), TensorType.f32.blockLayout().?.bytes);
    try std.testing.expect(!TensorType.iq3_s.isKnown() == false);
}

test "print dumps the documented layout and does not truncate small arrays" {
    const a = std.testing.allocator;
    var b = Builder{};
    defer b.buf.deinit(a);

    try b.buf.appendSlice(a, "GGUF");
    try b.u32le(3);
    try b.u64le(1); // 1 tensor
    try b.u64le(3); // 3 kvs
    try b.kvString("general.architecture", "qwen3");
    try b.kvU32("qwen35.block_count", 24);
    try b.kvI32Array("rope.dimension_sections", &.{ 11, 11, 10, 0 });
    try b.tensor("token_embd.weight", &.{ 2, 3 }, 0, 0);

    const info_end = b.buf.items.len;
    const alignment: u64 = 32;
    while (b.buf.items.len % alignment != 0) try b.buf.append(a, 0);
    const data_off = b.buf.items.len;
    for (0..std.mem.alignForward(u64, 24, alignment)) |_| try b.buf.append(a, 0);

    var g = try parse(a, b.buf.items);
    defer g.deinit();

    var aw: Io.Writer.Allocating = .init(a);
    defer aw.deinit();
    try g.print(&aw.writer);
    const got = aw.written();

    // Golden text; the two byte offsets are the only values that depend on the
    // builder, so they are interpolated rather than hard-coded.
    const want = try std.fmt.allocPrint(a,
        \\file_size {d}
        \\version 3
        \\tensor_count 1
        \\metadata_kv_count 3
        \\alignment 32
        \\tensor_info_end {d}
        \\data_offset {d}
        \\sum_aligned_bytes 32
        \\check data_offset+sum == file_size : OK
        \\
        \\=== metadata (3) ===
        \\  0 general.architecture = "qwen3"
        \\  1 qwen35.block_count = UINT32 24
        \\  2 rope.dimension_sections = ARRAY<INT32> len=4 [11, 11, 10, 0]
        \\
        \\=== tensors (1) ===
        \\name                                         type     dims                         offset   abs_offset        bytes
        \\(offset = raw field, relative to data section; abs = data_offset + offset)
        \\token_embd.weight                            F32      [2,3] {d: >12} {d: >12} {d: >12}
        \\
    , .{ b.buf.items.len, info_end, data_off, 0, data_off, 24 });
    defer a.free(want);

    try std.testing.expectEqualStrings(want, got);

    // `printFull` must agree for small inputs (nothing to truncate).
    var fw: Io.Writer.Allocating = .init(a);
    defer fw.deinit();
    try g.printFull(&fw.writer);
    try std.testing.expectEqualStrings(want, fw.written());
}
