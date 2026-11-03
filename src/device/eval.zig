//! Device evaluation shared by the host, the GPU kernels and runtime-loaded
//! devices, and the build root of every per-model object, GPU image and HDL
//! `.so`: the generated `models` in, each model's `arp_device_<name>` vtable
//! getter (host) or `arp_*_<name>` kernels (device) out. The work lives in
//! the sibling files; this root holds the root-only declarations and the
//! export loop, and re-exports what the test suites and the `.so` shim call.
//! Cites VerA's device contract (`contract`) and dev/devices/abi.md and
//! dev/devices/gpu-evaluation.md for the ABI and the GPU path.
//!
//! Contents:
//! - `pub const std_options: std.Options`, `pub const panic`
//! - `pub const vera_validate_contract: bool`
//! - `pub const device_object_root`
//! - `pub const NoiseSource`, `NoiseGen`, `PsdTerm`, `Planes`, `Batch`, `Proto` (abi)
//! - `pub const Dual`, `DeviceBatch`, `ProtoStore`, `permuteInPlace`,
//!   `snapLimited`, `checkHost`, `exportDevice`, `deviceVtable`
//! - `pub const test_access`

const std = @import("std");
const builtin = @import("builtin");
const gompute = @import("gompute");
const abi = @import("device_abi");
const batch = @import("batch.zig");
const dual = @import("dual.zig");
const kernels = @import("kernels.zig");
const lane_kernels = @import("lane_kernels.zig");
const layout = @import("layout.zig");
const proto_store = @import("proto_store.zig");
const stamp = @import("stamp.zig");
const traits = @import("traits.zig");
const breakpoints = @import("breakpoints.zig");
const vtable = @import("vtable.zig");

/// No alternate signal stack, in any mode. std reserves it as a 256 KB
/// threadlocal per compilation unit, and every per-model object is its own
/// unit, so Debug's static TLS reached 16.25 MB with 64 models. That exceeds
/// `Io.Threaded`'s 16 MB default thread stack, and glibc's pthread_create
/// returns EINVAL for a stack that cannot hold the TLS (#24). This root
/// spawns no thread, so it never attached the stack; the exe's root keeps
/// Debug's stack-overflow trace for the threads it does spawn.
pub const std_options: std.Options = .{ .signal_stack_size = null };

/// Safety panics of this root's objects. On a GPU a failed check traps: the
/// default handler formats its message through an `Io.Writer` (a 4 KB stack
/// buffer and indirect calls) that no device prints, and that AMDGPU's LLVM
/// backend cannot lay out in ReleaseSafe ("Cannot scavenge register in FI
/// elimination"). The host keeps the default.
pub const panic = switch (builtin.target.cpu.arch) {
    .nvptx64, .amdgcn => std.debug.no_panic,
    else => std.debug.FullPanic(std.debug.defaultPanic),
};

/// VerA's device/host contract checks (`contract.validating`), on in Debug
/// only: off, a device build spends 0.4-1.8% fewer instructions. Off too in a
/// build with GPU kernels (`models.gpu_kernels`, build.zig): validating adds a
/// Debug-only `su_ok` field to Model, and the kernels compile at .fast without
/// it, so the uploaded rows would read at the wrong stride. Only a host device
/// object (eval.zig as the build root) has the per-model `models` module; a
/// runtime `.hdl` build (VerA's shim re-exports this decl from `dyn`) has
/// neither it nor GPU kernels, so it keeps the Debug checks.
pub const vera_validate_contract = builtin.mode == .debug and
    !(@import("root") == @This() and gpuKernelsBuilt());

/// The marker `layout.object_root` looks for on the build root: present
/// exactly when eval.zig is the root, a linked per-model object or GPU image.
pub const device_object_root = {};

/// `abi.NoiseSource`, one priced generator of `Hooks.collect_noise`.
pub const NoiseSource = abi.NoiseSource;
/// `abi.NoiseGen`, the entries of a device's `noise_gens`.
pub const NoiseGen = abi.NoiseGen;
/// `abi.PsdTerm`, what a device's `noisePsd` returns per generator.
pub const PsdTerm = abi.PsdTerm;
/// `abi.Planes`, the value planes `Batch.eval` stamps into.
pub const Planes = abi.Planes;
/// `abi.Batch`, what `ProtoStore.finalize` returns.
pub const Batch = abi.Batch;
/// `abi.Proto`, the staging handle `deviceVtable(...).proto_create` returns.
pub const Proto = abi.Proto;

/// Returns a `Dual` family over lanes `0..N`, unknown u on lane u.
pub const Dual = dual.Dual;
/// Returns the frozen per-type batch of device type `D`.
pub const DeviceBatch = batch.DeviceBatch;
/// Returns the construction-time store of device type `D`.
pub const ProtoStore = proto_store.ProtoStore;
/// Moves `items[k]` to `items[dst[k]]` in place.
pub const permuteInPlace = proto_store.permuteInPlace;
/// Writes back limited unknowns within 16 ulps of their probe (#49).
pub const snapLimited = stamp.snapLimited;
/// Compile error unless this host provides everything `D` requires.
pub const checkHost = vtable.checkHost;
/// Exports `D` under the runtime device ABI (`arp_*`), for the `.so` shim.
pub const exportDevice = vtable.exportDevice;
/// Returns `D`'s vtable without a shared library.
pub const deviceVtable = vtable.deviceVtable;

/// Private decls exposed to the test suites. Tests live in tests/eval.zig,
/// not here: this file is a dependency module of every test root, and Zig
/// runs only the root module's tests.
pub const test_access = if (builtin.is_test) .{
    .narrowEligible = layout.narrowEligible,
    .NoiseFamily = stamp.NoiseFamily,
    .gpuRefusal = traits.gpuRefusal,
    .GpuRefusal = traits.GpuRefusal,
    .Real = dual.Real,
    .Sparse = dual.DualFor(&.{ 0, 1 }, stamp.host_pad, false),
    // Three lanes pad to four: the shuffle path with a pad lane.
    .Sparse3 = dual.DualFor(&.{ 0, 1, 2 }, stamp.host_pad, false),
    .anyNonzero = stamp.anyNonzero,
    .segmentSum = kernels.segmentSum,
    .repMask = layout.repMask,
    .spicePulseBreak = breakpoints.spicePulseBreak,
    .spicePwlBreak = breakpoints.spicePwlBreak,
    .ParamRef = abi.ParamRef,
    .PatternView = abi.PatternView,
} else {};

// As the build root, export every model in `models`: its GPU kernels (or a
// no-op placeholder when it is not GPU-eligible) in a device compilation, and
// its `arp_device_<name>` vtable getter on the host. One `exportRaw` per
// kernel, so the kernel set is a table of `traits` predicates.
comptime {
    if (@import("root") == @This() and !builtin.is_test) {
        const models = @import("models");
        for (@typeInfo(models).@"struct".decl_names) |decl_name| {
            // `gpu_kernels` is a build flag and `ltra_ac` a term, not devices.
            if (@TypeOf(@field(models, decl_name)) != type) continue;
            const D = @field(models, decl_name);
            if (!@hasDecl(D, "Model")) continue;
            if (gompute.is_device) exportKernels(D, decl_name) else exportVtable(D, decl_name);
        }
        gompute.exportKernels(.{});
    }
}

// ---- private ----

fn gpuKernelsBuilt() bool {
    const models = @import("models");
    return @hasDecl(models, "gpu_kernels") and models.gpu_kernels;
}

/// Exports `D`'s per-instance kernels and, with `-Dgpu-lanes`, its lane
/// kernels at the lane block width; a placeholder when D stays on the host.
fn exportKernels(comptime D: type, comptime decl_name: []const u8) void {
    if (!traits.gpuEligible(D)) return gompute.exportRaw("arp_nop_" ++ decl_name, &placeholder);
    const sym = traits.kernelSym;
    const b = abi.gpu_block_size;
    gompute.exportRaw(sym(D, "eval"), &kernels.DeviceKernel(D, b, false).run);
    if (traits.constJacobian(D)) gompute.exportRaw(sym(D, "evalc"), &kernels.DeviceKernel(D, b, true).run);
    if (traits.hasStateKernel(D)) gompute.exportRaw(sym(D, "lim"), &kernels.StateKernel(D, b).run);
    if (traits.hasCtlKernel(D)) gompute.exportRaw(sym(D, "ctl"), &kernels.CtlKernel(D, b).run);
    if (@hasDecl(D, "q")) gompute.exportRaw(sym(D, "qtp"), &kernels.QTapeKernel(D, b).run);
    if (@hasDecl(D, "q")) gompute.exportRaw(sym(D, "qonly"), &kernels.ChargeKernel(D, b).run);
    gompute.exportRaw(sym(D, "reduce"), &kernels.ReduceKernel(D, b).run);
    if (!@import("lane_options").enabled) return;
    const lb = abi.gpu_lane_block_size;
    gompute.exportRaw(sym(D, "leval"), &lane_kernels.LaneEvalKernel(D, lb).run);
    if (traits.hasStateKernel(D)) gompute.exportRaw(sym(D, "llim"), &lane_kernels.LaneStateKernel(D, lb).run);
    if (traits.hasCtlKernel(D)) gompute.exportRaw(sym(D, "lctl"), &lane_kernels.LaneCtlKernel(D, lb).run);
    if (@hasDecl(D, "q")) gompute.exportRaw(sym(D, "lchg"), &lane_kernels.LaneChargeKernel(D, lb).run);
    if (@hasDecl(D, "nextBreakpoint") or traits.walksPending(D)) gompute.exportRaw(sym(D, "lbrk"), &lane_kernels.LaneBreakKernel(D, lb).run);
}

/// Checks `D` against this host and exports its `arp_device_<name>` getter.
fn exportVtable(comptime D: type, comptime decl_name: []const u8) void {
    vtable.checkHost(D);
    // A built-in has no `.hdl` card for the host to refuse by name.
    if (@hasDecl(D, "systf_calls") and D.systf_calls.len != 0)
        @compileError("built-in model " ++ decl_name ++ " calls " ++ D.systf_calls[0].name ++
            ": no VPI application is registered (LRM 12.32), so its value is unknown; see #197");
    if (!@hasDecl(D, "eval")) return;
    const Export = struct {
        fn get() callconv(.c) *const abi.DeviceVtable {
            return vtable.deviceVtable(D, @typeName(D));
        }
    };
    @export(&Export.get, .{ .name = "arp_device_" ++ decl_name });
}

fn placeholder(_: u64) callconv(gompute.kernel_callconv) void {}
