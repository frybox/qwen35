//! CUDA backend -- **stage 0: a transparent CPU trampoline**.
//!
//! Nothing in this file talks to a GPU.  `CudaBackend` embeds a `CpuBackend`
//! and `exec()` hands back the *CPU* vtable paired with `&self.cpu`, so every
//! call the engine makes against `--backend cuda` lands in the exact code path
//! a pure-CPU run would take.  There is no cudart dependency, no `.cu` file,
//! and no build.zig change behind this module.
//!
//! Why the trampoline exists at all: it lets the *selection* path be verified
//! end-to-end before any device code is written -- `--backend cuda` in the CLI,
//! `backend.Kind.cuda`, `selectBackend`, `Engine.load`, the `*Exec` the engine
//! holds, and `deinit` ordering.  Those are the wiring points that are painful
//! to debug once real GPU work starts failing, and they cost nothing to check
//! now: the acceptance criterion is that the token ids are byte-identical to
//! `--backend cpu` on the same weights (md5 of the `IDS:` line).
//!
//! How the follow-up stages replace this, one vtable slot at a time:
//!
//!   1. Add a real `pub const vtable = backend.Exec.VTable{...}` here whose
//!      targets recover `*CudaBackend` from `ctx` (the `ctx` the engine holds
//!      already points at the `CudaBackend`, not at `self.cpu`, because
//!      `exec()` below sets `ptr = self`).
//!   2. Port the device work load by load (`loadLinear` -> upload, `matvec` ->
//!      kernel launch) and point the corresponding slot at the new target
//!      while every not-yet-ported slot delegates to
//!      `backend_cpu.CpuBackend.vtable` with `ctx = &self.cpu`.  Mixed runs are
//!      therefore possible at all times and each port can be A/B'd against the
//!      md5 baseline the trampoline establishes today.
//!   3. When every slot is a real target, delete the delegation and the
//!      embedded `cpu` field.
//!
//! The engine (`src/engine.zig`) must not need a single edit for any of the
//! above: it only ever sees `*backend.Exec`.

const std = @import("std");
const gguf = @import("gguf.zig");
const backend = @import("backend.zig");
const cpu_mod = @import("backend_cpu.zig");

pub const CudaBackend = struct {
    /// Embedded by value, not by pointer: a `CudaBackend` owns its CPU half,
    /// so `Engine.load`'s single `alloc.create(BackendImpl)` + `deinit()` keeps
    /// covering both, and there is no second lifetime to get wrong.
    cpu: cpu_mod.CpuBackend,

    pub fn init(alloc: std.mem.Allocator, g: *const gguf.File) CudaBackend {
        return .{ .cpu = cpu_mod.CpuBackend.init(alloc, g) };
    }

    pub fn deinit(self: *CudaBackend) void {
        self.cpu.deinit();
    }

    /// Forward the `Exec` by plain delegation to the embedded CPU backend.
    ///
    /// The returned struct is a copy (`Exec` is two words) whose `ptr` is
    /// `&self.cpu` and whose `vtable` is the CPU one, which is exactly what the
    /// trampoline needs -- and it is what `CpuBackend.exec()` already builds,
    /// so delegating is both the smallest change and the one that cannot drift
    /// from the CPU backend's own contract.  Hand-building
    /// `.{ .ptr = &self.cpu, .vtable = &cpu_mod.CpuBackend.vtable }` would
    /// silently drop any vtable slot added later.
    ///
    /// This is safe only because `Engine.load` heap-allocates the backend once
    /// and never copies it: the `*CpuBackend` the engine ends up holding points
    /// inside this very struct, so `ptr`'s alignment and lifetime are that of
    /// the heap allocation.  Stage 1 (file header) replaces this with a real
    /// vtable whose targets `@ptrCast(ctx)` back to `*CudaBackend`.
    pub fn exec(self: *CudaBackend) backend.Exec {
        return self.cpu.exec();
    }
};
