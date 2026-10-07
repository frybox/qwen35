//! Dev-only CLI: encode text with our tokenizer from a GGUF and print ids —
//! used to diff against HF transformers tokenization.
//! Usage: qwen35-tok <model.gguf> <text>...
//!
//! Output per text argument, always exactly three lines:
//!   <id> <id> ...        -- encoded token ids
//!   DECODED: <b> <b> ... -- the ids decoded back to raw bytes (decimal)
//!   TEXT: <utf8>         -- the same bytes rendered as UTF-8, for eyeballing

const std = @import("std");
const Io = std.Io;
const qwen35 = @import("qwen35");

pub fn main(init: std.process.Init) !void {
    const arena: std.mem.Allocator = init.arena.allocator();
    const io = init.io;

    var stdout_buffer: [1 << 16]u8 = undefined;
    var stdout_writer: Io.File.Writer = .init(.stdout(), io, &stdout_buffer);
    const out = &stdout_writer.interface;

    const args = try init.minimal.args.toSlice(arena);
    if (args.len < 3) {
        try out.print("usage: {s} <model.gguf> <text>...\n", .{args[0]});
        try out.flush();
        return error.MissingArgument;
    }
    const gguf_path = args[1];

    var file = try Io.Dir.cwd().openFile(io, gguf_path, .{});
    defer file.close(io);
    const stat = try file.stat(io);
    var mm = try file.createMemoryMap(io, .{
        .len = @intCast(stat.size),
        .protection = .{ .read = true },
    });
    defer mm.destroy(io);

    var g = try qwen35.gguf.parse(arena, mm.memory);
    defer g.deinit();

    var tok = try qwen35.tokenizer.Tokenizer.initFromGguf(arena, &g);

    for (args[2..]) |text| {
        const ids = try tok.encode(arena, text);
        for (ids, 0..) |id, i| {
            if (i != 0) try out.print(" ", .{});
            try out.print("{d}", .{id});
        }
        try out.print("\n", .{});
        const decoded = try tok.decode(arena, ids);
        try out.print("DECODED:", .{});
        for (decoded) |b| try out.print(" {d}", .{b});
        try out.print("\n", .{});
        try out.print("TEXT: ", .{});
        if (std.unicode.utf8ValidateSlice(decoded)) {
            // Valid UTF-8: print the decoded text itself.
            try out.writeAll(decoded);
        } else {
            // BPE ids need not decode to valid UTF-8 (a token may hold a
            // partial sequence). Fall back to an explicit byte dump so the
            // line is never lossy or garbled.
            try out.print("<invalid utf8>", .{});
            for (decoded) |b| try out.print(" {d}", .{b});
        }
        try out.print("\n", .{});
    }
    try out.flush();
}
