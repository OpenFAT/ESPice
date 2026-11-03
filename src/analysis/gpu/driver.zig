//! Every gompute call that can start the GPU driver: cuInit, a primary
//! context, a module load. In: a backend, a device ordinal, a kernel name.
//! Out: gompute's handle and, in a test binary only, one count per call
//! (`test_hooks.touches`): the oracle that a deck `auto` prices on the CPU
//! never starts the driver (stage 1 of the price plan). Every other file
//! reaches the driver through here.
//! Cites: none (gompute's runtime API).
//!
//! Contents:
//! - `pub const test_hooks` (`touches`)
//! - `pub fn rawKernelByName(comptime be: gompute.Backend, name: []const u8, ordinal: c_int) !gompute.RawByName(be)`
//! - `pub fn compute(preferred: ?Backend) Error!Compute`
//! - `pub fn computeOn(preferred: ?Backend, ordinal: c_int) Error!Compute`
//! - `pub fn deviceCount(be: Backend) Error!u32`
//! - `pub fn cudaContext(ordinal: c_int) Error!gompute.runtime.cuda.Context`

const std = @import("std");
const gompute = @import("gompute");

const dynamic = gompute.runtime.dynamic;
const Backend = dynamic.Backend;
const Compute = dynamic.Compute;
const Error = dynamic.Error;
const is_test = @import("builtin").is_test;

/// Test binaries only: the driver entries called so far, from any thread.
/// Empty outside tests, so a release build carries no counter and no
/// increment (`touch` is a comptime-false branch there).
pub const test_hooks = if (is_test) struct {
    pub var touches: std.atomic.Value(u32) = .init(0);
} else struct {};

/// `gompute.rawKernelByName`: loads the image holding `name` on device
/// `ordinal`, starting the driver if nothing has.
pub fn rawKernelByName(comptime be: gompute.Backend, name: []const u8, ordinal: c_int) !gompute.RawByName(be) {
    touch();
    return gompute.rawKernelByName(be, name, ordinal);
}

/// `Compute.init`: device 0 of `preferred`, or the first backend that opens.
pub fn compute(preferred: ?Backend) Error!Compute {
    touch();
    return Compute.init(preferred);
}

/// `Compute.initDevice`: device `ordinal` of `preferred` (CUDA when null).
pub fn computeOn(preferred: ?Backend, ordinal: c_int) Error!Compute {
    touch();
    return Compute.initDevice(preferred, ordinal);
}

/// `deviceCount`: the devices `be`'s driver lists; starts it.
pub fn deviceCount(be: Backend) Error!u32 {
    touch();
    return dynamic.deviceCount(be);
}

/// `cuda.Context.init` on device `ordinal` (a runtime `.hdl` image's JIT).
pub fn cudaContext(ordinal: c_int) Error!gompute.runtime.cuda.Context {
    touch();
    return gompute.runtime.cuda.Context.init(ordinal);
}

// ---- private ----

inline fn touch() void {
    if (is_test) _ = test_hooks.touches.fetchAdd(1, .monotonic);
}
