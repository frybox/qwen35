const std = @import("std");
const Io = std.Io;

const qwen35 = @import("qwen35");
const gguf = qwen35.gguf;

/// Context threaded through the engine's per-token progress callback: prints
/// each generated token as soon as it is produced, C-style.
const StreamCtx = struct {
    out: *Io.Writer,
    tok: *const qwen35.tokenizer.Tokenizer,
    arena: std.mem.Allocator,

    fn onToken(ctx: ?*anyopaque, token: u32) void {
        const self: *StreamCtx = @ptrCast(@alignCast(ctx orelse return));
        if (qwen35.pool.stat_on) {
            const s = qwen35.pool.snapshot();
            qwen35.pool.reset();
            std.debug.print("STATS tok={d} k256={d} k32={d} ksc={d} kq8={d} kf32={d} total={d}\n", .{
                token, s.k256, s.k32, s.ksc, s.kq8, s.kf32, s.k256 + s.k32 + s.ksc + s.kq8 + s.kf32,
            });
        }
        const piece = self.tok.decode(self.arena, &.{token}) catch null;
        if (piece) |s| {
            if (s.len > 0) {
                self.out.print("{s}", .{s}) catch {};
            } else {
                self.out.print("[{d}]", .{token}) catch {};
            }
        } else {
            // Decode failed: fall back to the token id, like the C engine.
            self.out.print("[{d}]", .{token}) catch {};
        }
        self.out.flush() catch {};
    }
};

/// `QWEN35_DUMP_IDS`: print the raw generated token ids (A/B md5 probes).
var dump_ids = false;

pub fn main(init: std.process.Init) !void {
    // Hand the environment to the CUDA backend: Zig 0.17 only exposes it
    // through `Init`, and the backend needs $QWEN35_CUDART / $PATH to locate
    // libcudart and nvcc's sibling lib64.
    qwen35.cudart.environ_map = init.environ_map;
    if (init.environ_map.get("QWEN35_SIMD_TIER")) |t| qwen35.engine.Engine.setSimdTier(t);
    if (init.environ_map.get("QWEN35_NO_BATCH") != null) qwen35.engine.no_batch = true;
    if (init.environ_map.get("QWEN35_PROF") != null) {
        qwen35.engine.prof = true;
        qwen35.backend_cpu.prof = true;
    }
    if (init.environ_map.get("QWEN35_PROF_POST") != null) qwen35.pool.prof_post_only = true;
    if (init.environ_map.get("QWEN35_STAT") != null) qwen35.pool.stat_on = true;
    dump_ids = init.environ_map.get("QWEN35_DUMP_IDS") != null;
    const arena: std.mem.Allocator = init.arena.allocator();
    const io = init.io;

    var stdout_buffer: [1 << 16]u8 = undefined;
    var stdout_file_writer: Io.File.Writer = .init(.stdout(), io, &stdout_buffer);
    const out = &stdout_file_writer.interface;

    var stderr_buffer: [4096]u8 = undefined;
    var stderr_file_writer: Io.File.Writer = .init(.stderr(), io, &stderr_buffer);
    const err_out = &stderr_file_writer.interface;

    // `run` reports failures through `err_out` and returns an error; catching it
    // here keeps the exit status meaningful without dumping a Zig stack trace
    // over the diagnostics the user actually needs.
    const args = try init.minimal.args.toSlice(arena);
    run(arena, io, args, out, err_out) catch |e| {
        // `run` already printed a human-readable message; only add the error
        // name when it did not (e.g. an out-of-memory during printing).
        if (e == error.WriteFailed) err_out.print("error: {s}\n", .{@errorName(e)}) catch {};
        err_out.flush() catch {};
        std.process.exit(1);
    };
    out.flush() catch {};
}

fn run(
    arena: std.mem.Allocator,
    io: Io,
    args: []const [:0]const u8,
    out: *Io.Writer,
    err_out: *Io.Writer,
) !void {
    if (args.len < 2) {
        try err_out.print(
            \\usage: {s} <file.gguf> [--full]
            \\       {s} run <file.gguf> <prompt> [-n N]
            \\
        , .{ args[0], args[0] });
        try err_out.flush();
        return error.MissingArgument;
    }

    if (std.mem.eql(u8, args[1], "run")) {
        return runInference(arena, io, args, out, err_out);
    }

    return dumpGguf(arena, io, args, out, err_out);
}

/// `run <gguf> <prompt> [-n N]`: load the engine, greedy-generate, print the
/// text and per-token timing.
fn runInference(
    arena: std.mem.Allocator,
    io: Io,
    args: []const [:0]const u8,
    out: *Io.Writer,
    err_out: *Io.Writer,
) !void {
    if (args.len < 4) {
        try err_out.print(
            "usage: {s} run <file.gguf> <prompt> [-n N] [-t TEMPERATURE] [-k K] [-s SEED]\n" ++
                "  -t  temperature; 0 (default) = greedy argmax\n" ++
                "  -k  top-k cutoff (0 = off, default)\n" ++
                "  -s  PRNG seed for sampling (default 12345)\n" ++
                "  --threads N  intra-op worker count (0 = default)\n" ++
                "  --backend cpu|cuda  execution backend (default cpu; cuda is a\n" ++
                "      CPU-backed trampoline today, numerically identical to cpu)\n",
            .{args[0]},
        );
        try err_out.flush();
        return error.MissingArgument;
    }
    const path = args[2];
    const prompt_text = args[3];
    const engine = qwen35.engine;
    var max_seq: usize = engine.Engine.max_seq;
    var temp: f32 = 0;
    var top_k: usize = 0;
    var seed: u64 = 12345;
    var n_threads: usize = 0; // 0 = default (min(16, cpus))
    var backend_kind: qwen35.backend.Kind = .cpu;
    var i: usize = 4;
    while (i + 1 < args.len + 1 and i < args.len) : (i += 1) {
        if (std.mem.eql(u8, args[i], "--backend") and i + 1 < args.len) {
            // Keep the accepted set in sync with `backend.Kind`; naming the
            // valid values in the error is what makes a typo recoverable.
            backend_kind = std.meta.stringToEnum(qwen35.backend.Kind, args[i + 1]) orelse {
                try err_out.print("error: --backend expects 'cpu' or 'cuda', got '{s}'\n", .{args[i + 1]});
                try err_out.flush();
                return error.InvalidArgument;
            };
            i += 1;
        } else if (std.mem.eql(u8, args[i], "--threads") and i + 1 < args.len) {
            n_threads = std.fmt.parseInt(usize, args[i + 1], 10) catch {
                try err_out.print("error: --threads expects a number, got '{s}'\n", .{args[i + 1]});
                try err_out.flush();
                return error.InvalidArgument;
            };
            i += 1;
        } else if (std.mem.eql(u8, args[i], "-n") and i + 1 < args.len) {
            max_seq = std.fmt.parseInt(usize, args[i + 1], 10) catch {
                try err_out.print("error: -n expects a number, got '{s}'\n", .{args[i + 1]});
                try err_out.flush();
                return error.InvalidArgument;
            };
            i += 1;
        } else if (std.mem.eql(u8, args[i], "-t") and i + 1 < args.len) {
            temp = std.fmt.parseFloat(f32, args[i + 1]) catch {
                try err_out.print("error: -t expects a number, got '{s}'\n", .{args[i + 1]});
                try err_out.flush();
                return error.InvalidArgument;
            };
            i += 1;
        } else if (std.mem.eql(u8, args[i], "-k") and i + 1 < args.len) {
            top_k = std.fmt.parseInt(usize, args[i + 1], 10) catch {
                try err_out.print("error: -k expects a number, got '{s}'\n", .{args[i + 1]});
                try err_out.flush();
                return error.InvalidArgument;
            };
            i += 1;
        } else if (std.mem.eql(u8, args[i], "-s") and i + 1 < args.len) {
            seed = std.fmt.parseInt(u64, args[i + 1], 10) catch {
                try err_out.print("error: -s expects a number, got '{s}'\n", .{args[i + 1]});
                try err_out.flush();
                return error.InvalidArgument;
            };
            i += 1;
        } else {
            try err_out.print("error: unknown argument '{s}'\n", .{args[i]});
            try err_out.flush();
            return error.InvalidArgument;
        }
    }

    var e = engine.Engine.load(arena, path, io, n_threads, backend_kind) catch |e2| {
        try err_out.print("error: failed to load '{s}': {s}\n", .{ path, @errorName(e2) });
        try err_out.flush();
        return e2;
    };
    defer e.deinit();

    e.sampler = engine.Sampler.init(.{ .temperature = temp, .top_k = top_k, .seed = seed });

    var tok = try qwen35.tokenizer.Tokenizer.initFromGguf(arena, &e.g);

    const rendered = try qwen35.tokenizer.Tokenizer.renderChatTurn(arena, prompt_text, false);
    const prompt = try tok.encode(arena, rendered);
    try out.print("[prompt: {d} tokens]\n", .{prompt.len});

    var stream_ctx = StreamCtx{ .out = out, .tok = &tok, .arena = arena };
    try out.print("Output: ", .{});
    try out.flush();
    const res = try e.generate(arena, prompt, max_seq, StreamCtx.onToken, &stream_ctx);

    try out.print("\n\n-------------------\n", .{});
    try out.print("Prefill: {d} tokens in {d:.3} s  ({d:.2} tok/s)\n", .{
        prompt.len,
        res.prefill_ms / 1000.0,
        if (res.prefill_ms > 0) @as(f64, @floatFromInt(prompt.len)) / (res.prefill_ms / 1000.0) else 0.0,
    });
    try out.print("Generate: {d} tokens in {d:.3} s  ({d:.2} tok/s)\n", .{
        res.tokens.len,
        res.decode_ms / 1000.0,
        if (res.decode_ms > 0) @as(f64, @floatFromInt(res.tokens.len)) / (res.decode_ms / 1000.0) else 0.0,
    });
    try out.print("{s}\n", .{if (res.hit_eos) "Stop reason: EOS" else "Stop reason: max tokens reached"});
    // `QWEN35_DUMP_IDS`: print the raw generated token ids.  Used for A/B md5
    // probes -- text output can hide a divergence that shows up in ids, and ids
    // are what a refactor must keep byte-identical.
    if (dump_ids) {
        try out.print("IDS:", .{});
        for (res.tokens) |id| try out.print(" {d}", .{id});
        try out.print("\n", .{});
    }
    if (qwen35.engine.prof) { qwen35.engine.profReport(); qwen35.backend_cpu.profReportMatvec(); }
    try out.flush();
}

fn dumpGguf(
    arena: std.mem.Allocator,
    io: Io,
    args: []const [:0]const u8,
    out: *Io.Writer,
    err_out: *Io.Writer,
) !void {
    const path = args[1];
    var full = false;
    for (args[2..]) |a| {
        if (std.mem.eql(u8, a, "--full")) full = true;
    }

    // --- open + mmap (never read the whole file into memory) --------------
    // NB: `openFileAbsolute` asserts on relative paths (and panics), so use
    // `cwd().openFile`, which accepts both relative and absolute paths.
    const file = Io.Dir.cwd().openFile(io, path, .{}) catch |e| {
        try err_out.print("error: cannot open '{s}': {s}\n", .{ path, @errorName(e) });
        try err_out.flush();
        return e;
    };
    defer file.close(io);
    const stat = try file.stat(io);
    const file_size: usize = @intCast(stat.size);
    if (file_size == 0) {
        try err_out.print("error: '{s}' is empty\n", .{path});
        try err_out.flush();
        return error.EmptyFile;
    }
    var mm = try Io.File.MemoryMap.create(io, file, .{
        .len = file_size,
        .protection = .{ .read = true },
    });
    defer mm.destroy(io);

    var g = gguf.parse(arena, mm.memory) catch |e| {
        try err_out.print("error: {s} is not a valid GGUF file: {s}\n", .{ path, @errorName(e) });
        if (e == error.BadMagic) {
            if (mm.memory.len >= 4) {
                const head = mm.memory[0..4];
                try err_out.print("       first bytes: {x} {x} {x} {x} (expected 47 47 55 46 = \"GGUF\")\n", .{
                    head[0], head[1], head[2], head[3],
                });
            } else {
                try err_out.print("       file is only {d} byte(s) long\n", .{mm.memory.len});
            }
        } else if (e == error.SizeMismatch) {
            try err_out.print("       tensor byte sizes do not tile the data section " ++
                "(header parsed, but the file is truncated or corrupt)\n", .{});
        } else if (e == error.UnknownTensorType) {
            try err_out.print("       a tensor uses a ggml_type this build does not know; " ++
                "refusing to guess its size\n", .{});
        }
        try err_out.flush();
        return e;
    };
    defer g.deinit();

    // --- dump -------------------------------------------------------------
    // The dump itself lives in `gguf.File.print`, so any tool that already
    // parsed a GGUF can reuse it. The reader only knows its bytes, not the
    // path it was mapped from, so the "file ..." line stays here.
    try out.print("file {s}\n", .{path});
    if (full) try g.printFull(out) else try g.print(out);

    try out.flush();
}
