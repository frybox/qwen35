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
/// this box).  Both the per-.cu compile and the -shared link use it, so the two
/// stages can never disagree about what SASS/PTX the .so contains.
const cuda_arch = "-arch=sm_89";

/// The .cu files that make up libqwen35-cuda-kernels.so.
///
/// Why an explicit list instead of globbing the directory: `b.path()` needs a
/// static path, so a directory scan cannot feed `addFileArg` without the build
/// script reaching into the filesystem behind Zig's back.  The consequence is
/// that **a NEW .cu file must be registered here** -- otherwise it silently
/// does not get compiled into the .so and the Zig side's dlopen/lookup of its
/// launcher fails at run time with "missing kernel", not at build time.
/// `listCudaInputs` below enumerates every file in the directory, and the
/// `cuda-check` step (`zig build cuda-check -Dcuda=true`) compares the two so
/// the omission fails loudly at build time instead of silently at run time.
const cuda_sources = [_][]const u8{
    // block quantize / dequantize launchers (q35_launch_quantize_q8_0,
    // q35_launch_dequantize_q8_0)
    "convert.cu",
    // IQ codebook definitions (__device__ __constant__), instantiated once.
    // Must be here: the other .cu files only declare them extern.
    "iq_tables.cu",
    // mat-vec launchers (q35_launch_mul_mat_vec_q8_0, q35_launch_mul_mat_vec_f32)
    "mmvq.cu",
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

    // ---------------------------------------------------------------------------
    // The IQ lookup tables: ONE C source file, translated for Zig
    // ---------------------------------------------------------------------------
    // src/iq_tables.h is the only place the twelve IQ codebooks are written
    // down.  Three consumers instantiate it three ways:
    //
    //   * CUDA kernels  -- src/kernels/cuda/iq_tables.cuh defines
    //                      Q35_TABLES_CUDA   -> `__device__ __constant__`
    //   * Zig           -- translated here      -> `pub export const`; this IS
    //                      the host-side definition, and src/kernels/x86/
    //                      iq_tables.h declares the same objects extern for the
    //                      C kernels (so the x86 side compiles no copy)
    //
    // Zig 0.17 has no `@cImport` builtin; C headers go through a translate-c
    // step that produces a normal Zig module (see src/iq_tables.zig).
    const tables_c = b.addTranslateC(.{
        .root_source_file = b.path("src/iq_tables.h"),
        .target = target,
        .optimize = optimize,
        .link_libc = true,
    });
    tables_c.addIncludePath(b.path("src"));
    mod.addImport("iq_tables_c", tables_c.createModule());

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
    const cuda_enabled = b.option(bool, "cuda", "Build the CUDA backend kernels (requires nvcc on PATH)") orelse false;
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
                    "it would not be compiled into libqwen35-cuda-kernels.so\n", .{p});
            }
        }
        if (unregistered != 0) @panic("unregistered .cu file(s); add them to `cuda_sources` in build.zig");

        // Standalone compile check: `zig build cuda-check -Dcuda=true` nvcc's
        // every registered .cu with `-c` into a throwaway object (nvcc has no
        // `-fsyntax-only`) and starts no link, so a failure here is always a
        // source problem, never the .so link.  Not part of the default step.
        const cuda_check_step = b.step("cuda-check", "compile each src/kernels/cuda/*.cu with nvcc -c only (needs -Dcuda=true)");

        // Two-stage, because the kernels must NOT be linked into the
        // executable: the nvcc runtime support the host stubs pull in
        // (__cudaLaunchKernel, __cudaPushCallConfiguration, ...) lives in
        // libcudart, and a link-time reference would give the CLI a DT_NEEDED
        // on cudart on *every* machine -- breaking `--backend cpu` on hosts
        // without CUDA.  Instead the objects are packed into a shared library
        // that src/cudart.zig dlopen()s at run time, after it has opened
        // libcudart itself with RTLD_GLOBAL so those symbols resolve.
        //
        // Stage 1: one object per .cu, with -rdc=true (relocatable device code).
        //
        // -rdc is what makes the IQ codebooks exist ONCE.  Without it nvcc
        // closes each .cu into its own device module, so every .cu that
        // includes iq_tables.cuh got its own private copy of the twelve
        // __constant__ tables -- ~52 KB of duplicated constant bank per
        // translation unit, and no way for the linker to merge them.  With
        // -rdc the definition in iq_tables.cu stays a single strong symbol and
        // the other units carry only relocations against it; nvlink (stage 2,
        // below) resolves them.  Verified by inspecting the objects: only
        // iq_tables.o defines the grids, convert.o and mmvq.o no longer do.
        var objs: []std.Build.LazyPath = &.{};
        for (cuda_sources) |src| {
            // -I points at src/kernels/cuda so the sibling `common.cuh`,
            // `vecdotq.cuh`, `iq_tables.cuh` and `api.h` includes resolve.
            // -Xcompiler -fPIC: the objects end up in a .so, so they must be PIC.
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

        // Stage 2: link the objects into libqwen35-cuda-kernels.so, with
        // -rdc=true so nvcc runs nvlink over the relocatable device code.
        // Stage 1 objects hold -rdc device objects; without -rdc here the device
        // link never happens and the extern __constant__ declarations in
        // convert.cu / mmvq.cu would stay unresolved at run time.
        const link = b.addSystemCommand(&.{ nvcc, cuda_arch, "-rdc=true", "-shared", "-o" });
        const so = link.addOutputFileArg("libqwen35-cuda-kernels.so");
        // Every stage-1 object, in `cuda_sources` order.
        for (objs) |o| link.addFileArg(o);
        // The .so gets its DT_NEEDED from nvcc's own -lcudart default.  That is
        // fine: it is the *kernels* library that depends on cudart, and it is
        // only ever opened when `--backend cuda` is selected and a usable
        // device was found.  The CLI itself stays independent.
        // Install next to the CLI's other libraries: <prefix>/lib/.  (A plain
        // addInstallFile with "../lib/..." resolves *above* the prefix.)
        const install_so = b.addInstallFileWithDir(so, .lib, "libqwen35-cuda-kernels.so");
        b.getInstallStep().dependOn(&install_so.step);

        // The device side resolves its runtime through dlopen (src/cudart.zig),
        // so nothing here links -lcudart.
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
    const exe_tests = b.addTest(.{
        .root_module = exe.root_module,
    });

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
