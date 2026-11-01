const std = @import("std");

// Although this function looks imperative, it does not perform the build
// directly and instead it mutates the build graph (`b`) that will be then
// executed by an external runner. The functions in `std.Build` implement a DSL
// for defining build steps and express dependencies between them, allowing the
// build runner to parallelize the build automatically (and the cache system to
// know when a step doesn't need to be re-run).
/// The x86-64 feature set each SIMD kernel translation unit is compiled with.
///
/// These are deliberately attached to *separate build targets* (one per C file)
/// rather than to per-file `-mavx2` style flags. Per-file flags are overridden
/// by the `-mcpu <model>` flag that Zig appends for the compilation as a whole,
/// so `zig build -Dcpu=x86_64` used to silently strip the AVX-512/AVX2 flags
/// from the kernels, turning `#ifdef __AVX512VNNI__` style blocks into empty
/// stubs while `cpu_caps.c` still reported the feature as present at runtime.
/// Giving every kernel unit its own resolved target makes the ISA a property of
/// the unit, immune to whatever `-Dcpu` the user passes for the Zig code.
const KernelIsa = enum {
    /// AVX-512 F/BW/DQ/VL + VNNI (kernels512.c).
    avx512,
    /// AVX2 + FMA + F16C (kernels32.c, kernels256.c, iq_kernels.c).
    avx2,
    /// Plain x86-64 baseline (cpu_caps.c: pure CPUID, must run on anything).
    baseline,

    fn features(self: KernelIsa) std.Target.Cpu.Feature.Set {
        const x86 = std.Target.x86;
        return switch (self) {
            .avx512 => x86.featureSet(&.{
                .avx512f,
                .avx512bw,
                .avx512dq,
                .avx512vl,
                .avx512vnni,
            }),
            .avx2 => x86.featureSet(&.{ .avx2, .fma, .f16c }),
            .baseline => .empty,
        };
    }
};

/// Builds the target query for one kernel translation unit: x86-64 baseline CPU
/// model plus that unit's ISA features, for this host's OS/ABI.
fn kernelTarget(b: *std.Build, isa: KernelIsa) std.Build.ResolvedTarget {
    const host = b.graph.host.result;
    return b.resolveTargetQuery(.{
        .cpu_arch = .x86_64,
        .os_tag = host.os.tag,
        .abi = host.abi,
        .cpu_model = .baseline,
        .cpu_features_add = isa.features(),
    });
}

/// Locates `nvcc` the same way the Makefile does -- through PATH, never a
/// hardcoded /usr/local/cuda/bin.  Returns null when the toolkit is absent so
/// `-Dcuda=true` on a CUDA-less host fails with a readable message instead of
/// an inscrutable "file not found".
fn findNvcc(b: *std.Build) ?[]const u8 {
    const path_env = b.graph.environ_map.get("PATH") orelse return null;
    var it = std.mem.splitScalar(u8, path_env, ':');
    while (it.next()) |dir| {
        if (dir.len == 0) continue;
        const candidate = std.fs.path.join(b.allocator, &.{ dir, "nvcc" }) catch continue;
        std.Io.Dir.cwd().access(b.graph.io, candidate, .{}) catch continue;
        return candidate;
    }
    return null;
}

/// The device code target.  Fixed rather than `-arch=native`: the build has to
/// be reproducible, and docs/cuda-backend-brief.md §4 pins sm_89 (the 4090D on
/// this box).  Both the per-.cu compile (stage 1) and the device link (stage 2,
/// `nvcc -dlink`) use it, so the two stages can never disagree about what
/// SASS/PTX the executable ends up carrying.
const cuda_arch = "-arch=sm_89";

/// The .cu files that make up the CUDA backend.
///
/// Why an explicit list instead of globbing the directory: `b.path()` needs a
/// static path, so a directory scan cannot feed `addFileArg` without the build
/// script reaching into the filesystem behind Zig's back.  The consequence is
/// that **a NEW .cu file must be registered here** -- otherwise it silently
/// does not get compiled, the object never reaches the executable, and the Zig
/// side's lookup of its launcher fails at run time with "missing kernel"
/// instead of at build time.
/// `listCudaInputs` below enumerates every file in the directory, and the
/// registration check inside `if (cuda_enabled)` compares the two so the
/// omission fails loudly at build time instead of silently at run time.
const cuda_sources = [_][]const u8{
    // block quantize / dequantize launchers (q35_launch_quantize_q8_0,
    // q35_launch_dequantize_q8_0)
    "convert.cu",
    // Placeholder for the device-side IQ codebooks.  The data used to be defined
    // here (via the macro-instantiated host header); it now lives only in
    // src/iq_tables.zig, and the device side has not been ported yet.  Kept in
    // the list so the build graph is unchanged; see the file's header comment.
    "iq_tables.cu",
    // mat-vec launchers (q35_launch_mul_mat_vec_q8_0, q35_launch_mul_mat_vec_f32)
    "mmvq.cu",
    // pure elementwise activation launchers (q35_launch_scale_vec,
    // _add_resid, _copy_range, _zero_buf) -- P2b, first batch
    "elementwise.cu",
    // normalisation launchers (q35_launch_rmsnorm, _l2norm) -- P2b, second
    // batch; first kernels here whose reduction order differs from the CPU's
    "norm.cu",
    // activation launchers (q35_launch_silu, _mul_silu, _mul_sigmoid,
    // _swiglu, _softmax, _conv_step) -- P2b, second batch.  ULP-level (the
    // device expf is not glibc's), see the file header.
    "act.cu",
    // P2c: the state-machine kernels that only became device-able once KV and
    // the GDN state are device regions -- q35_launch_attn_decode
    // (scores/softmax/PV/gate) and q35_launch_gdn_step + _gdn_gamma (delta
    // rule).  These are the only kernels here with a WRITTEN accumulation
    // contract, because GDN state error carries across tokens; see both files.
    "attn.cu",
    "gdn.cu",
    // q35_launch_rope64 -- rotary embedding.  NOT bit-exact (the device powf
    // is not glibc's, and the phase error grows with `pos`), so dRope64 keeps
    // it behind $QWEN35_CUDA_ROPE; see rope.cu's header for why.
    "rope.cu",
};

/// Every regular file under src/kernels/cuda/.  These become explicit inputs of
/// the nvcc steps so that a `.cuh` edit invalidates the cached invocation --
/// Zig's cache tracks only the declared inputs, and `addDirectoryArg` (needed
/// for `-I`) does not create a dependency.
fn listCudaInputs(b: *std.Build) ![]const []const u8 {
    const alloc = b.allocator;
    var dir = try std.Io.Dir.cwd().openDir(b.graph.io, "src/kernels/cuda", .{ .iterate = true });
    defer dir.close(b.graph.io);
    var list: std.ArrayList([]const u8) = .empty;
    var it = dir.iterate();
    while (try it.next(b.graph.io)) |e| {
        if (e.kind != .file) continue;
        const path = try std.fmt.allocPrint(alloc, "src/kernels/cuda/{s}", .{e.name});
        try list.append(alloc, path);
    }
    std.mem.sort([]const u8, list.items, {}, struct {
        fn lt(_: void, a: []const u8, c: []const u8) bool {
            return std.mem.lessThan(u8, a, c);
        }
    }.lt);
    return list.toOwnedSlice(alloc);
}

pub fn build(b: *std.Build) void {
    // Default to the x86-64 *baseline* CPU model instead of `native`.
    //
    // native makes the compiler emit every instruction the build host happens
    // to support, so a binary built on an AVX-512 machine crashes with
    // "Illegal instruction" on an AVX2-only machine. SIMD is instead chosen at
    // *run time* by src/kernels/x86/cpu_caps.c, so the Zig half of the program
    // must stay baseline. `-Dcpu=native` (or `-Dtarget=...`) still opts into a
    // host-tuned build for maximum single-machine speed.
    const target = b.standardTargetOptions(.{
        .default_target = .{
            .cpu_arch = .x86_64,
            .cpu_model = .baseline,
        },
    });
    // Standard optimization options allow the person running `zig build` to select
    // between Debug, ReleaseSafe, ReleaseFast, and ReleaseSmall. Here we do not
    // set a preferred release mode, allowing the user to decide how to optimize.
    const optimize = b.standardOptimizeOption(.{});
    // It's also possible to define more custom flags to toggle optional features
    // of this build script using `b.option()`. All defined flags (including
    // target and optimize options) will be listed when running `zig build --help`
    // in this directory.

    // ---------------------------------------------------------------------------
    // `build_options`: the one build-time bit the Zig code is allowed to read
    // ---------------------------------------------------------------------------
    // The CUDA backend needs exactly one compile-time fact -- "were the .cu
    // objects put on the link line?" -- and it must come from the build, not
    // from probing the symbol table.  The previous mechanism
    // (`@hasDecl(Kernels, "q35_launch_mul_mat_vec")` over a struct of
    // `extern fn`) was wrong: `@hasDecl` asks "does this type have such a
    // member?", and it answers `false` for these even when the C definition is
    // linked into the very same executable -- so every CUDA build silently took
    // the CPU-only stub branch, printing "this executable was built without CUDA
    // kernels" while `nm` showed all nine `T q35_launch_*` linked in.
    //
    // ONE variable, ONE source of truth: `cuda_enabled` is read here and used
    // both for this option and for the `if (cuda_enabled)` block below that
    // actually compiles and links the nvcc objects, so "the bit is true" and
    // "the objects are on the link line" cannot drift apart.
    //
    // Attached to `mod` (below) rather than to each executable's root module:
    // `backend_cuda.zig` lives inside `mod`, and the import table of the module
    // a file belongs to is what `@import("build_options")` resolves against.
    // `mod` is shared by the CLI, `qwen35-tok` and BOTH test artifacts, so one
    // `addOptions` covers every compilation of that file -- a target that pulls
    // in `mod` without the option would otherwise fail with "no module named
    // 'build_options'" before compiling a single line.
    const cuda_enabled = b.option(bool, "cuda", "Build the CUDA backend kernels (requires nvcc on PATH)") orelse false;
    const build_options = b.addOptions();
    build_options.addOption(bool, "cuda_linked", cuda_enabled);

    // This creates a module, which represents a collection of source files alongside
    // some compilation options, such as optimization mode and linked system libraries.
    // Zig modules are the preferred way of making Zig code available to consumers.
    // addModule defines a module that we intend to make available for importing
    // to our consumers. We must give it a name because a Zig package can expose
    // multiple modules and consumers will need to be able to specify which
    // module they want to access.
    const mod = b.addModule("qwen35", .{
        // The root source file is the "entry point" of this module. Users of
        // this module will only be able to access public declarations contained
        // in this file, which means that if you have declarations that you
        // intend to expose to consumers that were defined in other files part
        // of this module, you will have to make sure to re-export them from
        // the root file.
        .root_source_file = b.path("src/root.zig"),
        // Later on we'll use this module as the root module of a test executable
        // which requires us to specify a target.
        .target = target,
        .link_libc = true,
    });

    // `backend_cuda.zig` lives inside `mod`, so its `@import("build_options")`
    // resolves against `mod`'s import table.  Attaching it here (and not on
    // each consumer) is what covers every compilation of that file at once: the
    // CLI, `qwen35-tok` and both test artifacts all pull `mod` in, and none of
    // them can fail with "no module named 'build_options'".  Doing it twice --
    // here and again on `exe.root_module` -- would be pointless: `exe` already
    // imports `mod`, so it sees the same object; a second `addOptions` on the
    // exe is a re-export of an identical value, not a distinct configuration.
    mod.addOptions("build_options", build_options);

    // ---------------------------------------------------------------------------
    // The IQ lookup tables: ONE Zig source file, already the definition
    // ---------------------------------------------------------------------------
    // src/iq_tables.zig (generated by scripts/gen_iq_tables.py) is the only
    // place the twelve IQ codebooks are written down, and it declares them
    // `pub export const`, so the arrays ARE the global symbols the x86 kernels
    // link against (src/kernels/x86/iq_tables.h declares them `extern const` --
    // declarations only).  There is no translate-c step any more: the data used
    // to live in a C header with three macro-selected instantiations, and Zig's
    // module was the translate-c output of it.
    //
    // CUDA device code cannot reference host objects, so the device-side
    // codebooks are not served by this file; see src/kernels/cuda/.

    // ---------------------------------------------------------------------------
    // C kernels for SIMD-optimized IQ quantization (ported from llama.cpp)
    // ---------------------------------------------------------------------------
    // One static library PER kernel translation unit, each with its OWN
    // resolved target carrying that unit's ISA features (see KernelIsa above).
    // Zig appends a single `-mcpu <model>` for the whole compilation of a
    // module, and that flag wins over per-file `-mavx2` / `-mavx512*` flags --
    // which is exactly how the AVX-512 kernels used to degrade into empty
    // stubs under `-Dcpu=x86_64`. Splitting the units makes the ISA a property
    // of the target instead of a flag that can be overridden.
    const kernel_units = [_]struct {
        name: []const u8,
        file: []const u8,
        isa: KernelIsa,
    }{
        // CPUID probes: baseline, so the dispatcher itself can never fault on
        // an old CPU before it has decided what is available.
        .{ .name = "qwen35_cpu_caps", .file = "src/kernels/x86/cpu_caps.c", .isa = .baseline },
        .{ .name = "qwen35_kernels_avx2", .file = "src/kernels/x86/kernels32.c", .isa = .avx2 },
        .{ .name = "qwen35_kernels256_avx2", .file = "src/kernels/x86/kernels256.c", .isa = .avx2 },
        .{ .name = "qwen35_iq_kernels_avx2", .file = "src/kernels/x86/iq_kernels.c", .isa = .avx2 },
        // AVX-512 VNNI kernels are a SEPARATE compile unit so the AVX2 units
        // above keep their own codegen (adding -mavx512vnni to them would
        // change the AVX2 kernels' output). Dispatch is runtime
        // (cpu_has_avx512_vnni()).
        .{ .name = "qwen35_kernels_avx512", .file = "src/kernels/x86/kernels512.c", .isa = .avx512 },
        // Table cross-check: reads the tables through the extern declarations
        // and counts mismatches against upstream anchors. Baseline ISA on
        // purpose -- it must link even when no SIMD tier is available.
        .{ .name = "qwen35_iq_table_check", .file = "src/kernels/x86/iq_table_check.c", .isa = .baseline },
    };

    var kernel_libs: [kernel_units.len]*std.Build.Step.Compile = undefined;
    for (kernel_units, 0..) |unit, i| {
        const lib = b.addLibrary(.{
            .name = unit.name,
            .linkage = .static,
            .root_module = b.createModule(.{
                // The ISA comes from this unit's own target; the user's
                // `-Dcpu` only affects the Zig code and the final link.
                .target = kernelTarget(b, unit.isa),
                .optimize = optimize,
                .link_libc = true,
            }),
        });
        lib.root_module.addCSourceFile(.{
            .file = b.path(unit.file),
            .flags = &.{ "-std=c11", "-O3" },
        });
        b.installArtifact(lib);
        kernel_libs[i] = lib;
    }

    // Links every kernel unit into `step`, in one place, so the executable and
    // both test binaries can never disagree about which kernels they include.
    const linkKernels = struct {
        fn f(step: *std.Build.Step.Compile, libs: []const *std.Build.Step.Compile) void {
            for (libs) |lib| step.root_module.linkLibrary(lib);
        }
    }.f;

    // Here we define an executable. An executable needs to have a root module
    // which needs to expose a `main` function. While we could add a main function
    // to the module defined above, it's sometimes preferable to split business
    // logic and the CLI into two separate modules.
    //
    // If your goal is to create a Zig library for others to use, consider if
    // it might benefit from also exposing a CLI tool. A parser library for a
    // data serialization format could also bundle a CLI syntax checker, for example.
    //
    // If instead your goal is to create an executable, consider if users might
    // be interested in also being able to embed the core functionality of your
    // program in their own executable in order to avoid the overhead involved in
    // subprocessing your CLI tool.
    //
    // If neither case applies to you, feel free to delete the declaration you
    // don't need and to put everything under a single module.
    const exe = b.addExecutable(.{
        .name = "qwen35",
        .root_module = b.createModule(.{
            // b.createModule defines a new module just like b.addModule but,
            // unlike b.addModule, it does not expose the module to consumers of
            // this package, which is why in this case we don't have to give it a name.
            .root_source_file = b.path("src/main.zig"),
            // Target and optimization levels must be explicitly wired in when
            // defining an executable or library (in the root module), and you
            // can also hardcode a specific target for an executable or library
            // definition if desireable (e.g. firmware for embedded devices).
            .target = target,
            .optimize = optimize,
            // List of modules available for import in source files part of the
            // root module.
            .imports = &.{
                // Here "qwen35" is the name you will use in your source code to
                // import this module (e.g. `@import("qwen35")`). The name is
                // repeated because you are allowed to rename your imports, which
                // can be extremely useful in case of collisions (which can happen
                // importing modules from different packages).
                .{ .name = "qwen35", .module = mod },
            },
        }),
    });

    // `build_options` is attached to `mod` only (see where `mod` is built);
    // `exe` imports `mod`, so re-attaching it here would duplicate the same
    // value on the exe's own import table.

    // Link all C kernel units (each built for its own ISA).
    linkKernels(exe, &kernel_libs);

    // -------------------------------------------------------------------------
    // CUDA kernels for the `--backend cuda` path (src/kernels/cuda/)
    // -------------------------------------------------------------------------
    // Option-driven, because the CUDA half must not become a dependency of the
    // plain `qwen35`/`mod_tests` builds: those must keep working on hosts with
    // no toolkit installed.  `-Dcuda=true` compiles the .cu files with nvcc and
    // links the resulting objects into the CLI.
    //
    // Why a system command: Zig cannot compile .cu, and nvcc's own host
    // compilation must stay outside the Zig cache (it is a *reproducible*
    // external step, but Zig has no device-code support).  The cost is that the
    // cache does NOT track includes for us -- so every file in
    // src/kernels/cuda/ is declared as an input explicitly
    // (`addDirectoryArg`).  Without that, editing a .cuh would be a no-op.
    //
    // `cuda_enabled` is read once, near the `build_options` step above: the same
    // value both injects `cuda_linked` into the Zig code and gates the nvcc
    // objects on the link line, so the two can never disagree.
    if (cuda_enabled) {
        const nvcc = findNvcc(b) orelse {
            std.debug.print("error: -Dcuda=true but nvcc was not found on PATH\n", .{});
            @panic("nvcc not found; install the CUDA toolkit or build without -Dcuda");
        };
        const cuda_dir = "src/kernels/cuda";
        // Every file in the directory, used as an *input dependency* below so a
        // .cuh edit invalidates the nvcc step.
        const cuda_files = listCudaInputs(b) catch |e| {
            std.debug.print("error: -Dcuda=true but src/kernels/cuda/ could not be read: {s}\n", .{@errorName(e)});
            @panic("cannot enumerate src/kernels/cuda");
        };

        // Registration check: every .cu on disk must be in `cuda_sources` at the
        // top of this file.  Without this, adding a kernel file is a silent
        // no-op at build time and only shows up at run time as
        // `missingKernel(lib, path, "q35_launch_...")` from backend_cuda.zig.
        std.debug.print("cuda: {} file(s) in {s}, {} registered for nvcc -c\n", .{ cuda_files.len, cuda_dir, cuda_sources.len });
        var unregistered: usize = 0;
        for (cuda_files) |p| {
            if (!std.mem.endsWith(u8, p, ".cu")) continue; // .cuh/.h need no -c
            const base = p["src/kernels/cuda/".len..];
            var listed = false;
            for (cuda_sources) |src| {
                if (std.mem.eql(u8, src, base)) listed = true;
            }
            if (!listed) {
                unregistered += 1;
                std.debug.print("error: {s} is not registered in `cuda_sources` (build.zig); " ++
                    "it would not be compiled into the CUDA objects linked into the executable\n", .{p});
            }
        }
        if (unregistered != 0) @panic("unregistered .cu file(s); add them to `cuda_sources` in build.zig");

        // Standalone compile check: `zig build cuda-check -Dcuda=true` nvcc's
        // every registered .cu with `-c` into a throwaway object (nvcc has no
        // `-fsyntax-only`) and starts no link, so a failure here is always a
        // source problem, never the device/executable link.  Not part of the
        // default step.
        const cuda_check_step = b.step("cuda-check", "compile each src/kernels/cuda/*.cu with nvcc -c only (needs -Dcuda=true)");

        // Two stages, then the objects go straight into the executable.
        //
        // The kernels are LINKED IN rather than packed into a .so: a shared
        // object would be a second artifact (and a run-time dlopen target whose
        // resolution depends on the loader's search order).  Static objects keep
        // one binary, one artifact list.
        //
        // Stage 1: one object per .cu, with -rdc=true (relocatable device code).
        //
        // -rdc is what lets nvlink (stage 2) fold every .cu's device code into
        // ONE device image.  Without it nvcc closes each .cu into its own device
        // module: any definition shared across translation units (the IQ
        // codebook pointers of iq_tables.cu) gets a private copy per unit, and a
        // cross-unit device reference does not link at all.
        //
        // -Xcompiler -fPIC: the objects are handed to Zig's linker through
        // `addObjectFile`, i.e. they end up in an ordinary (PIE) executable, so
        // the host half must be position independent.
        var objs: []std.Build.LazyPath = &.{};
        for (cuda_sources) |src| {
            // -I points at src/kernels/cuda so the sibling `common.cuh`,
            // `vecdotq.cuh`, `iq_tables.cuh` and `api.h` includes resolve.
            const step = b.addSystemCommand(&.{ nvcc, cuda_arch, "-O3", "--std=c++17", "-rdc=true", "-c", "-Xcompiler", "-fPIC" });
            step.has_side_effects = false;
            step.addArg("-I");
            step.addDirectoryArg(b.path(cuda_dir));
            step.addFileArg(b.path(b.fmt("{s}/{s}", .{ cuda_dir, src })));
            step.addArg("-o");
            const obj = step.addOutputFileArg(b.fmt("{s}.o", .{src[0 .. src.len - 3]}));
            // Inputs that are NOT on the command line: headers and siblings.
            // `addDirectoryArg` only sets the include search path, so without
            // these the nvcc step would not be invalidated by a .cuh edit (the
            // trap called out in docs/cuda-backend-brief.md §4).
            // `addFileInput` registers the dependency without appending an
            // argument, which `addFileArg` would do (and break the single-input
            // `nvcc -c ... -o` contract).
            for (cuda_files) |p| step.addFileInput(b.path(p));
            objs = b.allocator.realloc(objs, objs.len + 1) catch @panic("oom");
            objs[objs.len - 1] = obj;

            // cuda-check: same flags, but `-c` into a scratch output and no
            // link, so it isolates "source does not compile" from "does not
            // link".  Reuses cuda-check only to keep its inputs in the cache.
            const check = b.addSystemCommand(&.{ nvcc, cuda_arch, "-O3", "--std=c++17", "-c", "-I" });
            check.has_side_effects = false;
            check.addDirectoryArg(b.path(cuda_dir));
            check.addFileArg(b.path(b.fmt("{s}/{s}", .{ cuda_dir, src })));
            check.addArg("-o");
            _ = check.addOutputFileArg(b.fmt("check-{s}.o", .{src[0 .. src.len - 3]}));
            for (cuda_files) |p| check.addFileInput(b.path(p));
            cuda_check_step.dependOn(&check.step);
        }

        // Stage 2: the device link.  `nvcc -rdc=true -dlink` runs nvlink over the
        // stage-1 objects and emits a host-side object whose only job is to carry
        // the merged device image (plus the registration stub that makes cudart
        // load it).  It has no kernels of its own; it is what turns N per-unit
        // device modules into one, and it must name the same `-arch` as stage 1
        // or nvlink has nothing to merge.
        //
        // -Xcompiler -fPIC for the same reason as stage 1: this object is also
        // fed to Zig's linker.
        const dlink = b.addSystemCommand(&.{ nvcc, cuda_arch, "-rdc=true", "-dlink", "-Xcompiler", "-fPIC", "-o" });
        const dlink_obj = dlink.addOutputFileArg("cuda-kernels.dlink.o");
        // Every stage-1 object, in `cuda_sources` order, and only those: nvcc
        // takes the object list, not the header dependencies (the .cu inputs
        // already chain through stage 1).
        for (objs) |o| dlink.addFileArg(o);

        // The host half of the IQ codebook upload: `q35_host_tables()` in
        // src/kernels/cuda/iq_tables_host.c.  It is PLAIN C, not CUDA -- it only
        // declares the twelve arrays that src/iq_tables.zig defines and records
        // their addresses and `sizeof`s -- so it is compiled with Zig's own C
        // compiler rather than nvcc, and it lands on the same link line as the
        // .cu objects because that is where its caller (iq_tables.cu) lives.
        //
        // Why it must be in the CUDA block and not compiled unconditionally: the
        // records are relocations against the IQ tables (`U iq1s_grid`, ...).
        // Linking it into a CPU-only build would change the CPU artifact list for
        // a file only the CUDA path needs.
        //
        // Why it is not an `extern` in a .cu instead: nvcc folds a host `extern
        // const T name[n]` declared in the same translation unit as the
        // `__device__ ... *name` pointer into one HIDDEN host object (bind to the
        // pointer's bytes, not the table) -- see the file's own comment.  A
        // separate C translation unit is what keeps the two names apart.
        //
        // Registered here, NOT with `addCSourceFile`: that would compile it into
        // the CPU kernel archive, which every target (including the test
        // binaries and the tok executable) links.  This way exactly one object
        // exists and only the -Dcuda=true executable gets it.
        {
            const step = b.addSystemCommand(&.{ b.graph.zig_exe, "cc", "-std=c11", "-O2", "-c", "-I" });
            step.has_side_effects = false;
            step.addDirectoryArg(b.path(cuda_dir));
            step.addFileArg(b.path(b.fmt("{s}/iq_tables_host.c", .{cuda_dir})));
            step.addArg("-o");
            const host_obj = step.addOutputFileArg("iq_tables_host.o");
            // Same dependency discipline as stage 1: the .inc it expands and the
            // .cuh it may include are not on the command line.
            for (cuda_files) |p| step.addFileInput(b.path(p));
            // `q35_host_tables` need not be device-linked, so it goes to the
            // executable directly (nvlink would ignore a host-only object).
            exe.root_module.addObjectFile(host_obj);
        }

        // Hand the objects to the executable.  `addObjectFile` appends to the
        // module's link objects, so each object is linked exactly once, in the
        // order the graph lists them.
        for (objs) |o| exe.root_module.addObjectFile(o);
        exe.root_module.addObjectFile(dlink_obj);

        // The CUDA runtime, for the host stubs (`__cudaRegisterFatBinary`,
        // `__cudaPushCallConfiguration`, ...) that the stage-2 object calls.
        // The library directory is derived from nvcc's own location
        // (`<toolkit>/bin/nvcc` -> `<toolkit>/lib64`, with `<toolkit>/lib` as the
        // fallback) instead of hardcoding /usr/local/cuda, so a toolkit installed
        // anywhere on PATH works.
        //
        // libc++ as well: nvcc compiles the host half of a .cu as C++, so the
        // objects reference the C++ runtime (`__cxa_guard_acquire`,
        // `__gxx_personality_v0`, `_Unwind_Resume`) even though the kernels are
        // all `extern "C"`.  Nothing else in the graph pulls it in -- the Zig
        // half is C-ABI only -- so the link fails with "undefined symbol:
        // __cxa_guard_abort" without it.
        exe.root_module.linkSystemLibrary("cudart", .{});
        exe.root_module.linkSystemLibrary("c++", .{});
        const nvcc_bin_dir = std.fs.path.dirname(nvcc) orelse ".";
        const toolkit_dir = std.fs.path.dirname(nvcc_bin_dir) orelse nvcc_bin_dir;
        const lib64_dir = b.pathJoin(&.{ toolkit_dir, "lib64" });
        if (std.Io.Dir.cwd().access(b.graph.io, lib64_dir, .{})) |_| {
            exe.root_module.addLibraryPath(.{ .cwd_relative = lib64_dir });
        } else |_| {
            exe.root_module.addLibraryPath(.{ .cwd_relative = b.pathJoin(&.{ toolkit_dir, "lib" }) });
        }
        // A DT_NEEDED on libcudart.so is deliberate: the executable is the only
        // artifact now, so "contains CUDA kernels" and "needs cudart" are the
        // same build (-Dcuda=true).  The CPU-only build never enters this block.
    }

    // This declares intent for the executable to be installed into the
    // install prefix when running `zig build` (i.e. when executing the default
    // step). By default the install prefix is `zig-out/` but can be overridden
    // by passing `--prefix` or `-p`.
    b.installArtifact(exe);

    // Dev-only CLI: tokenization diff tool against HF transformers.
    const tok_cli = b.addExecutable(.{
        .name = "qwen35-tok",
        .root_module = b.createModule(.{
            .root_source_file = b.path("src/tok_cli.zig"),
            .target = target,
            .optimize = optimize,
            .imports = &.{
                .{ .name = "qwen35", .module = mod },
            },
        }),
    });
    b.installArtifact(tok_cli);

    // This creates a top level step. Top level steps have a name and can be
    // invoked by name when running `zig build` (e.g. `zig build run`).
    // This will evaluate the `run` step rather than the default step.
    // For a top level step to actually do something, it must depend on other
    // steps (e.g. a Run step, as we will see in a moment).
    const run_step = b.step("run", "Run the app");

    // This creates a RunArtifact step in the build graph. A RunArtifact step
    // invokes an executable compiled by Zig. Steps will only be executed by the
    // runner if invoked directly by the user (in the case of top level steps)
    // or if another step depends on it, so it's up to you to define when and
    // how this Run step will be executed. In our case we want to run it when
    // the user runs `zig build run`, so we create a dependency link.
    const run_cmd = b.addRunArtifact(exe);
    run_step.dependOn(&run_cmd.step);

    // By making the run step depend on the default step, it will be run from the
    // installation directory rather than directly from within the cache directory.
    run_cmd.step.dependOn(b.getInstallStep());

    // This allows the user to pass arguments to the application in the build
    // command itself, like this: `zig build run -- arg1 arg2 etc`.
    // Zig 0.17 removed the `b.args` field; the run step instead declares a
    // passthrough slot which the maker fills with the `-- ...` tail at run
    // time.
    run_cmd.addPassthruArgs();

    // Creates an executable that will run `test` blocks from the provided module.
    // Here `mod` needs to define a target, which is why earlier we made sure to
    // set the releative field.
    const mod_tests = b.addTest(.{
        .root_module = mod,
    });
    // The test-only helpers src/simd_c.zig declare extern C symbols, so the
    // test binary must link the C kernels library (undefined symbol:
    // cpu_has_avx2 / q4_0_q8_0 otherwise). The helper files are imported by
    // path from files already in `mod`, so no separate module is needed.
    linkKernels(mod_tests, &kernel_libs);

    // A run step that will run the test executable.
    const run_mod_tests = b.addRunArtifact(mod_tests);

    // Creates an executable that will run `test` blocks from the executable's
    // root module. Note that test executables only test one module at a time,
    // hence why we have to create two separate ones.
    //
    // A fresh module rather than `exe.root_module`: the executable's module also
    // carries the ISA-specific C kernel units (`linkKernels`), whose symbols are
    // exactly what `mod` -- imported here as `qwen35` -- already provides, so
    // reusing that module would link every kernel twice.  A test binary gets its
    // kernels from `mod` (through the import) exactly as `mod_tests` does.
    //
    // (Historically this comment also named `src/kernels/cuda/iq_tables_host.c`:
    // that file used to be compiled into the executable.  It is not any more --
    // see the CUDA block above -- so the `h_*` duplicate-symbol reason no longer
    // applies.)
    const exe_tests = b.addTest(.{
        .root_module = b.createModule(.{
            .root_source_file = b.path("src/main.zig"),
            .target = target,
            .optimize = optimize,
            .imports = &.{
                .{ .name = "qwen35", .module = mod },
            },
        }),
    });
    linkKernels(exe_tests, &kernel_libs);

    // A run step that will run the second test executable.
    const run_exe_tests = b.addRunArtifact(exe_tests);

    // A top level step for running all tests. dependOn can be called multiple
    // times and since the two run steps do not depend on one another, this will
    // make the two of them run in parallel.
    const test_step = b.step("test", "Run tests");
    test_step.dependOn(&run_mod_tests.step);
    test_step.dependOn(&run_exe_tests.step);

    // Just like flags, top level steps are also listed in the `--help` menu.
    //
    // The Zig build system is entirely implemented in userland, which means
    // that it cannot hook into private compiler APIs. All compilation work
    // orchestrated by the build system will result in other Zig compiler
    // subcommands being invoked with the right flags defined. You can observe
    // these invocations when one fails (or you pass a flag to increase
    // verbosity) to validate assumptions and diagnose problems.
    //
    // Lastly, the Zig build system is relatively simple and self-contained,
    // and reading its source code will allow you to master it.
}
