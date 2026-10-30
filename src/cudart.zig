//! Runtime (dlopen) binding to the CUDA runtime library.
//!
//! Why dlopen instead of `-lcudart`: the same `zig-out/bin/qwen35` binary must
//! run on hosts with no CUDA install at all, where `--backend cpu` is the only
//! usable mode.  A link-time DT_NEEDED on libcudart.so would turn "no GPU" into
//! a dynamic-loader failure before `main` even parses `--backend`.  (This is
//! also why the executable does not link the .o from src/kernels/cuda/*.cu
//! unless the CUDA build is requested -- see build.zig.)
//!
//! Every cudart entry point used by the backend is a function pointer resolved
//! with `std.DynLib.lookup`.  A missing symbol is collected in `missing` and
//! reported by the caller as a load error naming the symbol, never as a panic.
//!
//! Library search order (first hit wins), mirroring what the CUDA toolkit
//! itself does so a plain `nvcc` install works without LD_LIBRARY_PATH:
//!   1. $QWEN35_CUDART (explicit path; also the hook used to test the
//!      "no libcudart" error path by pointing it at a nonexistent file)
//!   2. "libcudart.so" / "libcudart.so.13" / ... through the dynamic loader
//!   3. $CUDA_HOME/lib64/libcudart.so*, /usr/local/cuda*/lib64/libcudart.so*,
//!      $HOME/cuda*/lib64/libcudart.so*
//!   4. the directory containing nvcc, if it is on PATH and its ../lib64 has
//!      a libcudart (this machine's layout: ~/cuda13/bin/nvcc for both tools)

const std = @import("std");
const builtin = @import("builtin");

/// Numeric mirrors of the cudart enums the backend reads.  Kept numeric so no
/// cuda_runtime_api.h is needed on the Zig side (and so the ABI stays whatever
/// the library we dlopen actually implements).
pub const cudaSuccess: c_int = 0;

/// Mirrors `cudaDeviceProp`.  Only the fields this backend reads are named; the
/// padding is sized so that every one of them lands on its real offset, because
/// the struct is filled by the **installed** runtime, not by a header we
/// control.
///
/// Verified against CUDA 13's `driver_types.h` (`sizeof == 1008`; the same
/// numbers hold for CUDA 11/12).
///
/// The previous definition here was 784 bytes with `totalGlobalMem` at 256 and
/// `major`/`minor` at 272/276 -- 32 bytes short of the first field and 88 bytes
/// short of the compute capability.  Reading those made `cc()` cast garbage
/// into a u32 (`@intCast` panics in Debug; in ReleaseFast it yields a value
/// that passes or fails the `min_cc` check at random), so the layout is
/// corrected to the truth instead of papered over with a bigger `_pad`:
///
///     +0    name[256]
///     +256  uuid[16]                 <- the 32-byte gap the old layout missed
///     +272  luid[8]
///     +280  luidDeviceNodeMask (4) + 4 bytes padding to 8-byte alignment
///     +288  totalGlobalMem (size_t)
///     +296  sharedMemPerBlock (size_t)
///     +304  regsPerBlock, warpSize (int, int)
///     +312  memPitch (size_t)
///     +320  maxThreadsPerBlock (int)
///     +324  maxThreadsDim[3] (12)
///     +336  maxGridSize[3] (12)
///     +348  clockRate (int)
///     +352  totalConstMem (size_t)
///     +360  major (int)
///     +364  minor (int)
///     +368  textureAlignment, texturePitchAlignment (size_t, size_t)
///     +384  deviceOverlap (int)
///     +388  multiProcessorCount (int)
///
/// The tail is padded to the true 1008 bytes so that a caller which ever copies
/// the struct by value cannot read past what the runtime wrote.
pub const DeviceProp = extern struct {
    name: [256]u8,
    uuid: [16]u8,
    luid: [8]u8,
    luid_device_node_mask: u32,
    _pad0: u32,
    total_global_mem: usize,
    shared_mem_per_block: usize,
    _pad1: [8]u8, // regsPerBlock, warpSize
    mem_pitch: usize,
    _pad2: [28]u8, // maxThreadsPerBlock, maxThreadsDim[3], maxGridSize[3], clockRate
    total_const_mem: usize,
    major: c_int,
    minor: c_int,
    _pad3: [16]u8, // textureAlignment, texturePitchAlignment
    device_overlap: c_int,
    multi_processor_count: c_int,
    _pad_tail: [1008 - 392]u8,

    comptime {
        // A wrong offset here is silent garbage, not a compile error, so pin the
        // three fields the backend actually reads.
        std.debug.assert(@offsetOf(DeviceProp, "name") == 0);
        std.debug.assert(@offsetOf(DeviceProp, "total_global_mem") == 288);
        std.debug.assert(@offsetOf(DeviceProp, "major") == 360);
        std.debug.assert(@offsetOf(DeviceProp, "minor") == 364);
        std.debug.assert(@offsetOf(DeviceProp, "multi_processor_count") == 388);
        std.debug.assert(@sizeOf(DeviceProp) == 1008);
    }

    pub fn cc(self: DeviceProp) u32 {
        return @as(u32, @intCast(self.major)) * 10 + @as(u32, @intCast(self.minor));
    }
    pub fn nameSlice(self: DeviceProp) []const u8 {
        return std.mem.sliceTo(&self.name, 0);
    }
};

/// `cudaMemcpyKind`
pub const MemcpyHostToDevice: c_int = 1;
pub const MemcpyDeviceToHost: c_int = 2;

/// Environment, injected by `main` before the backend is created.  Zig 0.17
/// removed `std.posix.getenv`; environment now arrives through
/// `std.process.Init.environ_map`, which only `main` has.  The backend needs it
/// for `$QWEN35_CUDART` (explicit library path) and `$PATH` (find nvcc's
/// sibling lib64), so `main.zig` hands the map over here.
pub var environ_map: ?*const std.process.Environ.Map = null;

pub fn getenv(name: []const u8) ?[]const u8 {
    const m = environ_map orelse return null;
    return m.get(name);
}

/// Minimum compute capability the kernels in src/kernels/cuda are built for.
/// `-arch=sm_89` bakes sm_89 SASS into the object; a pre-8.9 device would fall
/// back to PTX JIT (or fail), so the backend checks explicitly instead of
/// silently running a mismatched binary.
pub const min_cc: u32 = 89;

const dl = struct {
    extern fn dlopen(path: ?[*:0]const u8, mode: std.c.RTLD) ?*anyopaque;
    extern fn dlsym(handle: ?*anyopaque, sym: [*:0]const u8) ?*anyopaque;
    extern fn dlclose(handle: ?*anyopaque) c_int;
    extern fn dlerror() ?[*:0]u8;
};

/// RTLD_GLOBAL is required: the kernel `.so` is opened afterwards and its nvcc
/// host stubs (`__cudaLaunchKernel`, `__cudaPushCallConfiguration`, ...) must be
/// able to bind to cudart's symbols from the global scope.  RTLD_NOW resolves
/// everything up front so a broken library fails at open time, not mid-run.
const rtld_global_now = std.c.RTLD{ .GLOBAL = true, .NOW = true };

pub const Lib = struct {
    handle: ?*anyopaque,

    // --- driver-ish / device management -------------------------------------
    GetDeviceCount: *const fn (*c_int) callconv(.c) c_int,
    GetDeviceProperties: *const fn (*DeviceProp, c_int) callconv(.c) c_int,
    SetDevice: *const fn (c_int) callconv(.c) c_int,
    GetLastError: *const fn () callconv(.c) c_int,
    GetErrorString: *const fn (c_int) callconv(.c) [*:0]const u8,
    GetErrorName: *const fn (c_int) callconv(.c) [*:0]const u8,

    // --- memory --------------------------------------------------------------
    Malloc: *const fn (*?*anyopaque, usize) callconv(.c) c_int,
    Free: *const fn (?*anyopaque) callconv(.c) c_int,
    Memcpy: *const fn (?*anyopaque, ?*const anyopaque, usize, c_int) callconv(.c) c_int,
    MemGetInfo: *const fn (*usize, *usize) callconv(.c) c_int,
    /// Blocking on the default stream; device synchronize is the cheapest way
    /// to make an async launch error observable before we read results back.
    DeviceSynchronize: *const fn () callconv(.c) c_int,

    // --- streams -------------------------------------------------------------
    /// Kept for future overlap; today the backend uses the default stream and
    /// only needs the type to pass `null` through the launcher ABI.
    StreamCreate: *const fn (*?*anyopaque) callconv(.c) c_int,
    StreamDestroy: *const fn (?*anyopaque) callconv(.c) c_int,

    /// Close the library handle (process exit would do it too; explicit is
    /// better for leak checkers).
    pub fn close(self: *Lib) void {
        _ = dl.dlclose(self.handle);
        self.handle = null;
    }

    pub fn errName(self: Lib, code: c_int) [*:0]const u8 {
        return self.GetErrorName(code);
    }
    pub fn errStr(self: Lib, code: c_int) [*:0]const u8 {
        return self.GetErrorString(code);
    }
};

pub const OpenError = error{
    LibraryNotFound,
    MissingSymbol,
    NoDevice,
    ComputeCapabilityTooLow,
    OutOfMemory,
};

/// A validated, usable CUDA device plus the resolved runtime entry points.
pub const Open = struct {
    lib: Lib,
    /// Device name, NUL-trimmed.  Stored inline so it does not alias the
    /// missing-symbol accumulator buffer.
    name_buf: [256]u8 = undefined,
    name_len: usize = 0,
    cc: u32,
    free_bytes: usize,
    total_bytes: usize,
    /// Comma-separated unresolved symbol names; empty when every bind worked.
    missing_buf: [512]u8 = undefined,
    missing_len: usize = 0,

    pub fn devName(self: *const Open) []const u8 {
        return self.name_buf[0..self.name_len];
    }
    pub fn missing(self: *const Open) []const u8 {
        return self.missing_buf[0..self.missing_len];
    }
};

/// Try each candidate in order, then validate a usable device.
///
/// On failure the returned error is precise (`LibraryNotFound` vs
/// `MissingSymbol` vs `ComputeCapabilityTooLow` ...) and, for the missing-symbol
/// case, `out.missing` holds the names.  `out` is written even when an error is
/// returned so the caller can quote the partially-filled fields.
pub fn open(out: *Open, want_bytes: usize) OpenError!void {
    out.* = .{
        .lib = undefined,
        .cc = 0,
        .free_bytes = 0,
        .total_bytes = 0,
    };

    out.lib.handle = try loadLibrary();

    var miss_len: usize = 0;
    const mbuf: *[512]u8 = &out.missing_buf;
    const bind = struct {
        fn f(l: *Lib, comptime T: type, comptime sym: [:0]const u8, miss: *[512]u8, len: *usize) ?T {
            if (dl.dlsym(l.handle, sym.ptr)) |p| return @ptrCast(@alignCast(p));
            const rest = miss[len.*..];
            if (rest.len > sym.len + 1) {
                @memcpy(rest[0..sym.len], sym);
                rest[sym.len] = ',';
                len.* += sym.len + 1;
            }
            return null;
        }
    }.f;

    const L = &out.lib;
    L.GetDeviceCount = bind(&out.lib, *const fn (*c_int) callconv(.c) c_int, "cudaGetDeviceCount", mbuf, &miss_len) orelse return error.MissingSymbol;
    L.GetDeviceProperties = bind(&out.lib, *const fn (*DeviceProp, c_int) callconv(.c) c_int, "cudaGetDeviceProperties", mbuf, &miss_len) orelse return error.MissingSymbol;
    L.SetDevice = bind(&out.lib, *const fn (c_int) callconv(.c) c_int, "cudaSetDevice", mbuf, &miss_len) orelse return error.MissingSymbol;
    L.GetLastError = bind(&out.lib, *const fn () callconv(.c) c_int, "cudaGetLastError", mbuf, &miss_len) orelse return error.MissingSymbol;
    L.GetErrorString = bind(&out.lib, *const fn (c_int) callconv(.c) [*:0]const u8, "cudaGetErrorString", mbuf, &miss_len) orelse return error.MissingSymbol;
    L.GetErrorName = bind(&out.lib, *const fn (c_int) callconv(.c) [*:0]const u8, "cudaGetErrorName", mbuf, &miss_len) orelse return error.MissingSymbol;
    L.Malloc = bind(&out.lib, *const fn (*?*anyopaque, usize) callconv(.c) c_int, "cudaMalloc", mbuf, &miss_len) orelse return error.MissingSymbol;
    L.Free = bind(&out.lib, *const fn (?*anyopaque) callconv(.c) c_int, "cudaFree", mbuf, &miss_len) orelse return error.MissingSymbol;
    L.Memcpy = bind(&out.lib, *const fn (?*anyopaque, ?*const anyopaque, usize, c_int) callconv(.c) c_int, "cudaMemcpy", mbuf, &miss_len) orelse return error.MissingSymbol;
    L.MemGetInfo = bind(&out.lib, *const fn (*usize, *usize) callconv(.c) c_int, "cudaMemGetInfo", mbuf, &miss_len) orelse return error.MissingSymbol;
    L.DeviceSynchronize = bind(&out.lib, *const fn () callconv(.c) c_int, "cudaDeviceSynchronize", mbuf, &miss_len) orelse return error.MissingSymbol;
    L.StreamCreate = bind(&out.lib, *const fn (*?*anyopaque) callconv(.c) c_int, "cudaStreamCreate", mbuf, &miss_len) orelse return error.MissingSymbol;
    L.StreamDestroy = bind(&out.lib, *const fn (?*anyopaque) callconv(.c) c_int, "cudaStreamDestroy", mbuf, &miss_len) orelse return error.MissingSymbol;
    out.missing_len = miss_len;

    // --- device validation ---------------------------------------------------
    var count: c_int = 0;
    if (L.GetDeviceCount(&count) != cudaSuccess or count <= 0) return error.NoDevice;
    if (L.SetDevice(0) != cudaSuccess) return error.NoDevice;

    var prop: DeviceProp = std.mem.zeroes(DeviceProp);
    if (L.GetDeviceProperties(&prop, 0) != cudaSuccess) return error.NoDevice;
    const nm = prop.nameSlice();
    out.name_len = @min(nm.len, out.name_buf.len);
    @memcpy(out.name_buf[0..out.name_len], nm[0..out.name_len]);
    out.cc = prop.cc();

    if (out.cc < min_cc) return error.ComputeCapabilityTooLow;

    var free_b: usize = 0;
    var total_b: usize = 0;
    if (L.MemGetInfo(&free_b, &total_b) != cudaSuccess) return error.NoDevice;
    out.free_bytes = free_b;
    out.total_bytes = total_b;

    // Reserve room for the caller's device buffers; failing here (at load time)
    // is a named error instead of an OOM in the middle of a forward pass.
    if (want_bytes > free_b) return error.OutOfMemory;
}

/// dlopen in the order documented at the top of this file.  Returns the raw
/// handle (RTLD_GLOBAL) so dlopen'd kernels can bind cudart symbols later.
fn loadLibrary() OpenError!*anyopaque {
    // 1. Explicit override.  Authoritative: when set, no other candidate is
    //    tried, so pointing it at a nonexistent file is a deterministic way to
    //    exercise the "no libcudart" error path (documented in the message).
    if (getenv("QWEN35_CUDART")) |p| {
        return tryOpen(p) orelse error.LibraryNotFound;
    }

    // 2. Plain sonames -- the loader applies its own search path
    //    (LD_LIBRARY_PATH, ld.so.cache).
    const sonames = [_][]const u8{
        "libcudart.so",     "libcudart.so.13",    "libcudart.so.12",
        "libcudart.so.13.0", "libcudart.so.12.0",
    };
    for (sonames) |s| {
        if (tryOpen(s)) |h| return h;
    }

    // 3. Usual toolkit prefixes (dlopen needs a real path for these).
    const fixed = [_][]const u8{
        "/usr/local/cuda/lib64/libcudart.so",
        "/usr/local/cuda/lib64/libcudart.so.13",
        "/usr/local/cuda/lib64/libcudart.so.12",
        "/usr/lib/x86_64-linux-gnu/libcudart.so.13",
        "/usr/lib/x86_64-linux-gnu/libcudart.so.12",
    };
    for (fixed) |s| {
        if (tryOpen(s)) |h| return h;
    }

    // 4. <dir of nvcc>/../lib64/libcudart.so -- this machine has ~/cuda13 with
    //    no /usr/local/cuda symlink, so nvcc's own prefix is the reliable hint.
    if (findNearNvcc()) |path| {
        if (tryOpen(path)) |h| return h;
    }

    return error.LibraryNotFound;
}

var path_buf: [1024]u8 = undefined;

/// NUL-terminate and dlopen.  `path` need not be sentinel-terminated.
fn tryOpen(path: []const u8) ?*anyopaque {
    if (path.len == 0 or path.len >= path_buf.len) return null;
    @memcpy(path_buf[0..path.len], path);
    path_buf[path.len] = 0;
    return openLib(path_buf[0..path.len :0]);
}

fn openLib(path: [:0]const u8) ?*anyopaque {
    return dl.dlopen(path.ptr, rtld_global_now);
}

/// `access(path, F_OK)` without dragging an `std.Io` through the backend.
fn access_ok(path: [:0]const u8) bool {
    const c = struct {
        extern fn access(path: [*:0]const u8, amode: c_int) c_int;
    };
    return c.access(path.ptr, 0) == 0;
}

var nvcc_lib_buf: [1024]u8 = undefined;

/// `<dir of PATH/nvcc>/../lib64/libcudart.so`.  Uses its own buffer so it does
/// not clobber `path_buf` (which tryOpen reuses).
fn findNearNvcc() ?[]const u8 {
    const path_env = getenv("PATH") orelse return null;
    var it = std.mem.splitScalar(u8, path_env, ':');
    while (it.next()) |dir| {
        if (dir.len == 0) continue;
        var nb: [513]u8 = undefined;
        const nvcc_s = std.fmt.bufPrint(nb[0..512], "{s}/nvcc", .{dir}) catch continue;
        nb[nvcc_s.len] = 0; // sentinel for the libc call
        const nvcc: [:0]const u8 = nb[0..nvcc_s.len :0];
        // Plain libc access(): no Io handle is available inside the backend
        // (Engine hands every backend `init(alloc, g)` and nothing else), and
        // bringing std.Io through here would mean threading a parameter through
        // code the engine owns.  F_OK on an absolute path is all we need.
        if (access_ok(nvcc) == false) continue;
        // <dir>/nvcc -> <dir>/../lib64/libcudart.so
        // dlopen resolves ".." itself, so no path normalization is needed.
        return std.fmt.bufPrint(&nvcc_lib_buf, "{s}/../lib64/libcudart.so", .{dir}) catch continue;
    }
    return null;
}
