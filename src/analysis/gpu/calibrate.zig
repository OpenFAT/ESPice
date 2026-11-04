//! The measurements of `espice calibrate` that need the GPU driver. In: the
//! driver, one prepared circuit per generated deck, generated LU matrices.
//! Out: the card's attributes with its launch, bus and copy rates
//! (`device`), the driver's in-process start (`probeStart`), one model's
//! device and host eval times at one size (`sample`), and one matrix's LU
//! row (`lu`). Each reuses the path a query takes: `GpuContext` evals, the
//! host batch eval the cost probe times, ParEval, `GpuLu` on a `direct`
//! solver as `lu_bench.zig` drives it; no kernel of its own.
//! The spine that generates the decks and writes the file is
//! `espice/calibrate.zig`.
//! Cites: the price plan sec. 4 (each constant, how it is measured, and its
//! accuracy).
//!
//! Contents:
//! - `pub const Error`
//! - `pub const Device = struct`
//! - `pub const Cpus = struct` (`Range`, `parse`)
//! - `pub const Sample = struct`
//! - `pub const Grid = struct`
//! - `pub fn probeStart() !f64`
//! - `pub fn device() !Device`
//! - `pub fn sample(gpa: Allocator, io: std.Io, template: *const Prepared, knobs: Knobs, cpus: Cpus, lanes: u32, compile_all: bool) !Sample`
//! - `pub fn lu(gpa: Allocator, grid: Grid, knobs: Knobs) !machine.Lu`

const std = @import("std");
const assert = std.debug.assert;
const Allocator = std.mem.Allocator;
const linux = std.os.linux;
const gompute = @import("gompute");
const solver = @import("solver");
const Prepared = @import("device").Circuit;
const device_ir = @import("device").abi;
const Circuit = @import("../Circuit.zig").Circuit;
const Knobs = @import("../knobs.zig").Knobs;
const machine = @import("../machine.zig");
const ParEval = @import("../par_eval.zig").ParEval;
const drv = @import("driver.zig");
const devices = @import("devices.zig");
const context = @import("context.zig");
const ctx_cost = @import("context_cost.zig");
const gpu_lu = @import("lu.zig");

const backend = devices.backend;
const nowNs = context.nowNs;

pub const Error = error{
    /// This build carries no kernel images: there is nothing to calibrate.
    NoGpuArtifacts,
    /// A heavy image the driver has not compiled yet, without `--compile-all`.
    ImageNotCompiled,
    /// The generated circuit has no batch with a kernel image.
    CircuitNotEligible,
    /// A device eval or solve the driver refused partway.
    DeviceEvalFailed,
};

/// The card's attributes and its fixed costs.
pub const Device = struct {
    fp64_ratio: f64,
    sms: f64,
    /// Compute capability, major.minor.
    cc: f64,
    mem_gb: f64,
    /// One empty launch and its sync, us.
    launch_us: f64,
    /// Pinned transfers: latency, us, and the rates up and down, GB/s.
    bus_lat_us: f64,
    bus_up_gbs: f64,
    bus_down_gbs: f64,
    /// A device-to-device copy, bytes per us: the staging clear and reduce
    /// stream at about this rate.
    copy_b_per_us: f64,
};

/// The CPUs host timings pin to: the performance cores, and the efficiency
/// cores when the CPU has them.
pub const Cpus = struct {
    p: Range,
    e: ?Range,

    pub const Range = struct { first: u16, last: u16 };

    /// From `Identity`'s CPU lists (`0-15`); a uniform CPU (empty lists)
    /// is all `threads` performance cores. Only a list's first range counts.
    pub fn parse(id: *const machine.Identity) Cpus {
        assert(id.threads > 0);
        const all: Range = .{ .first = 0, .last = @intCast(@min(id.threads, cpus_max) - 1) };
        return .{ .p = range(id.p_cpus.slice()) orelse all, .e = range(id.e_cpus.slice()) };
    }
};

/// One model at one size: what a `machine.Model` row holds per size.
pub const Sample = struct {
    /// The kernel image's Wyhash, the row's key.
    image: u64,
    dev_us: f64,
    host_p_us: f64,
    host_e_us: f64,
    /// This process's first load of the image: its JIT, or the driver
    /// cache's copy; ~0 when an earlier sample loaded it.
    jit_ms: f64,
    /// ParEval on `lanes` lanes over one; 1 when `lanes` is 1.
    lane_gain: f64,
};

/// A `side`^`dims` grid Laplacian (5- or 7-point): its flops per column
/// grow with `side`, steeply in 3-D, which spans the price's F/n axis.
pub const Grid = struct { side: u32, dims: u2 };

/// A fitted line t = at0 + slope * x.
const Line = struct { at0: f64, slope: f64 };

/// CPUs a `cpu_set_t` names.
const cpus_max = @bitSizeOf(linux.cpu_set_t);

/// One sample batches calls until it lasts this long, so a sub-microsecond
/// eval is not timer noise.
const sample_ns: u64 = 20 * std.time.ns_per_us;
/// Samples are taken until this much wall has gone, at least 3...
const budget_ns: u64 = 15 * std.time.ns_per_ms;
/// ...and at most this many.
const samples_max = 40;

/// The driver's in-process start: init, the primary context, and the LU
/// image's load, ms. `espice calibrate --probe-start` runs it in a child
/// whose wall, against a child that exits at once, is `start_ms`.
pub fn probeStart() !f64 {
    if (comptime backend == null) return Error.NoGpuArtifacts;
    const t0 = nowNs();
    var k = try drv.rawKernelByName(backend.?, "arp_gc_combine", 0);
    defer k.deinit();
    return ms(nowNs() - t0);
}

/// The card's attributes, the launch latency (an empty `arp_gc_combine`),
/// pinned transfers fitted over 4 KiB..64 MiB by least squares, and a
/// device-to-device copy of 64 MiB. Starts the driver.
pub fn device() !Device {
    if (comptime backend == null) return Error.NoGpuArtifacts;
    var k = try drv.rawKernelByName(backend.?, "arp_gc_combine", devices.ordinal);
    defer k.deinit();
    const info = try k.context.info();
    const big = 64 << 20;
    var d = try k.alloc(big);
    defer d.free();
    var d2 = try k.alloc(big);
    defer d2.free();
    const pin = try k.allocPinned(big);
    defer k.freePinned(pin);
    @memset(pin, 1);

    // Launch: n = 0, so every thread returns at once.
    var n: u64 = 0;
    var alpha: f64 = 0;
    const args = [_]gompute.interface.Arg{ gompute.interface.arg(&n), gompute.interface.arg(&alpha), d.argPtr(), d.argPtr(), d.argPtr() };
    var launch = struct {
        k: *@TypeOf(k),
        args: []const gompute.interface.Arg,
        failed: bool = false,
        fn run(s: *@This()) void {
            s.k.launch(.{ .x = 1 }, .{ .x = 32 }, 0, s.args) catch {
                s.failed = true;
            };
            s.k.synchronize() catch {
                s.failed = true;
            };
        }
    }{ .k = &k, .args = &args };
    const launch_us = medianUs(&launch);
    if (launch.failed) return Error.DeviceEvalFailed;

    // Bus: t = lat + bytes / rate per direction.
    const sizes = [_]usize{ 4 << 10, 64 << 10, 1 << 20, 8 << 20, 64 << 20 };
    var fit: [2]Line = undefined;
    for (&fit, [_]bool{ true, false }) |*f, up| {
        var ts: [sizes.len]f64 = undefined;
        for (sizes, &ts) |size, *t| {
            var copy = struct {
                d: *@TypeOf(d),
                pin: []u8,
                size: usize,
                up: bool,
                failed: bool = false,
                fn run(s: *@This()) void {
                    const r = if (s.up) s.d.upload(s.pin.ptr, s.size) else s.d.download(s.pin.ptr, s.size);
                    r catch {
                        s.failed = true;
                    };
                }
            }{ .d = &d, .pin = pin, .size = size, .up = up };
            t.* = medianUs(&copy);
            if (copy.failed) return Error.DeviceEvalFailed;
        }
        f.* = leastSquares(&sizes, &ts);
    }
    var dd = struct {
        k: *@TypeOf(k),
        dst: *@TypeOf(d),
        src: *@TypeOf(d),
        failed: bool = false,
        fn run(s: *@This()) void {
            s.dst.copyFrom(s.src, 0, 0, big) catch {
                s.failed = true;
            };
            s.k.synchronize() catch {
                s.failed = true;
            };
        }
    }{ .k = &k, .dst = &d2, .src = &d };
    const dd_us = medianUs(&dd);
    if (dd.failed) return Error.DeviceEvalFailed;

    const cc = info.compute_capability;
    const out: Device = .{
        .fp64_ratio = @floatFromInt(info.fp64_ratio),
        .sms = @floatFromInt(info.sms),
        .cc = @as(f64, @floatFromInt(cc / 10)) + @as(f64, @floatFromInt(cc % 10)) / 10,
        .mem_gb = @as(f64, @floatFromInt(info.mem_total)) / (1 << 30),
        .launch_us = launch_us,
        .bus_lat_us = (fit[0].at0 + fit[1].at0) / 2,
        // us per byte -> GB/s.
        .bus_up_gbs = 1e-3 / fit[0].slope,
        .bus_down_gbs = 1e-3 / fit[1].slope,
        .copy_b_per_us = big / dd_us,
    };
    assert(out.launch_us > 0);
    assert(std.math.isFinite(out.bus_up_gbs) and std.math.isFinite(out.bus_down_gbs));
    return out;
}

/// One model at `template`'s size: its eligible batch evaluated on the
/// device (`GpuContext.eval`, after 10 warm evals), on the host pinned to a
/// performance core and to an efficiency core (the cost probe's eval, at
/// its spread x), and, when `lanes` > 1, through ParEval on `lanes` lanes
/// of the performance cores against one. Fails with `ImageNotCompiled` for
/// a heavy image the driver has not compiled, unless `compile_all`.
/// Asserts `template` was never evaluated.
pub fn sample(gpa: Allocator, io: std.Io, template: *const Prepared, knobs: Knobs, cpus: Cpus, lanes: u32, compile_all: bool) !Sample {
    if (comptime backend == null) return Error.NoGpuArtifacts;
    assert(lanes >= 1);
    var c = try Circuit.instantiate(template, gpa);
    defer c.deinit();
    const b = for (c.batches) |b| {
        if (b.hooks.gpu_payload != null) break b;
    } else return Error.CircuitNotEligible;
    const p = b.hooks.gpu_payload.?(b.ctx);
    const image = ctx_cost.imageOf(p.kernel) orelse return Error.CircuitNotEligible;
    if (!compile_all and image.len > ctx_cost.heavy_image_bytes and !ctx_cost.imageWarm(image, knobs)) return Error.ImageNotCompiled;
    const t_jit = nowNs();
    var held = try drv.rawKernelByName(backend.?, p.kernel, devices.ordinal);
    defer held.deinit();
    const jit_ms = ms(nowNs() - t_jit);
    ctx_cost.markWarm(image, knobs);

    // The cost probe's spread x (`context_cost.quote`).
    const x = try gpa.alloc(f64, c.n + 1);
    defer gpa.free(x);
    for (x, 0..) |*xi, i| xi.* = @as(f64, @floatFromInt(i % 7)) * 0.3;
    const pl = c.ownPlanes();
    var host = struct {
        b: device_ir.Batch,
        pl: *const device_ir.Planes,
        x: []const f64,
        fn run(s: *@This()) void {
            s.b.eval(s.b.ctx, s.pl, 0, s.b.count, s.x, 0);
        }
    }{ .b = b, .pl = &pl, .x = x };
    const old = affinity() orelse return error.Unexpected;
    defer setAffinity(old);
    setAffinity(cpuSet(cpus.p.first, cpus.p.first));
    const host_p_us = medianUs(&host);
    const host_e_us = if (cpus.e) |e| blk: {
        setAffinity(cpuSet(e.first, e.first));
        break :blk medianUs(&host);
    } else host_p_us;

    var lane_gain: f64 = 1;
    if (lanes > 1) {
        setAffinity(cpuSet(cpus.p.first, cpus.p.last));
        var us: [2]f64 = undefined;
        for ([_]u32{ 1, lanes }, &us) |l, *u| {
            var par = try ParEval.init(gpa, io, c.batches, c.nnz, c.n, c.has_charge, l);
            defer par.deinit();
            var walk = struct {
                par: *ParEval,
                c: *Circuit,
                x: []const f64,
                fn run(s: *@This()) void {
                    s.par.run(s.c.batches, s.c.ownPlanes(), .{}, s.x, 0, .full);
                }
            }{ .par = &par, .c = &c, .x = x };
            u.* = medianUs(&walk);
        }
        lane_gain = us[0] / us[1];
    }
    setAffinity(cpuSet(cpus.p.first, cpus.p.last));

    const gctx = try context.GpuContext.init(gpa, .{ .batches = c.batches, .planes = c.ownPlanes(), .n = c.n, .trash_slot = c.trash_slot, .has_charge = c.has_charge, .knobs = knobs }, null, true, 1, 1e3, null);
    defer gctx.deinit();
    var dev = struct {
        g: *context.GpuContext,
        x: []const f64,
        failed: bool = false,
        fn run(s: *@This()) void {
            if (!s.g.eval(s.x, 0, .{}, false)) s.failed = true;
        }
    }{ .g = gctx, .x = x };
    for (0..10) |_| dev.run();
    const dev_us = medianUs(&dev);
    if (dev.failed) return Error.DeviceEvalFailed;
    return .{ .image = std.hash.Wyhash.hash(0, image), .dev_us = dev_us, .host_p_us = host_p_us, .host_e_us = host_e_us, .jit_ms = jit_ms, .lane_gain = lane_gain };
}

/// `grid`'s LU row: the host full factor, then device refactors and solves
/// (`GpuLu.solve`, forced past its admission bars) on values that change
/// every call, against the host's refactor and solve on the same values.
/// The first device call builds the epoch's tables (`table_us`). Load the
/// LU image first (`device`), or the first row's tables carry its load.
pub fn lu(gpa: Allocator, grid: Grid, knobs: Knobs) !machine.Lu {
    if (comptime backend == null) return Error.NoGpuArtifacts;
    assert(grid.dims >= 2 and grid.dims <= 3);
    assert(grid.side >= 2);
    const n: u32 = std.math.pow(u32, grid.side, grid.dims);
    const col_ptr = try gpa.alloc(u32, n + 1);
    defer gpa.free(col_ptr);
    const row_idx = try gpa.alloc(u32, n * (2 * @as(u32, grid.dims) + 1));
    defer gpa.free(row_idx);
    const nnz = laplacian(grid, col_ptr, row_idx);
    const vals = try gpa.alloc(f64, nnz);
    defer gpa.free(vals);
    for (0..n) |j| for (col_ptr[j]..col_ptr[j + 1]) |q| {
        vals[q] = if (row_idx[q] == j) 2 * @as(f64, @floatFromInt(grid.dims)) + 1 else -1;
    };
    var slv = try solver.direct.Solver.init(gpa, n, col_ptr, row_idx[0..nnz], null, .{});
    defer slv.deinit();
    try slv.factor(vals, .{});
    const rhs = try gpa.alloc(f64, 2 * n);
    defer gpa.free(rhs);
    @memset(rhs[0..n], 1);
    const dx = rhs[n..];

    const g = try gpu_lu.GpuLu.init(gpa, n, nnz, knobs);
    defer g.deinit();
    g.forced = true;
    g.place = true;
    var call = struct {
        g: *gpu_lu.GpuLu,
        slv: *solver.direct.Solver,
        vals: []f64,
        rhs: []const f64,
        dx: []f64,
        bump: f64 = 1,
        failed: bool = false,
        fn run(s: *@This()) void {
            // A new value every call, so every call refactors.
            s.bump += 1e-9;
            s.vals[0] = 7 * s.bump;
            if (!(s.g.solve(s.slv, s.vals, s.rhs, s.dx) orelse false)) s.failed = true;
        }
    }{ .g = g, .slv = &slv, .vals = vals, .rhs = rhs[0..n], .dx = dx };
    call.run();
    if (call.failed) return Error.DeviceEvalFailed;
    const table_us = @as(f64, @floatFromInt(g.st.load_ns)) * 1e-3;
    const dev_us = medianUs(&call);
    if (call.failed) return Error.DeviceEvalFailed;

    var host = struct {
        slv: *solver.direct.Solver,
        vals: []const f64,
        dx: []f64,
        failed: bool = false,
        fn run(s: *@This()) void {
            const sp = s.slv.sparse().?;
            sp.refactor(s.slv.col_ptr, s.vals, s.slv.params.refactor_growth_limit) catch {
                s.failed = true;
            };
            sp.solve(s.dx, s.dx);
        }
    }{ .slv = &slv, .vals = vals, .dx = dx };
    const host_us = medianUs(&host);
    if (host.failed) return Error.DeviceEvalFailed;
    assert(g.flops > 0);
    return .{ .n = n, .flops = g.flops, .dev_us = dev_us, .host_us = host_us, .table_us = table_us };
}

// ---- private ----

/// Median us of one `f.run()`. The batch-growing pass doubles as a warm-up.
fn medianUs(f: anytype) f64 {
    var inner: u32 = 1;
    for (0..24) |_| {
        const t0 = nowNs();
        for (0..inner) |_| f.run();
        if (nowNs() - t0 >= sample_ns) break;
        inner *= 2;
    }
    var xs: [samples_max]f64 = undefined;
    var count: usize = 0;
    const start = nowNs();
    while (count < samples_max and (count < 3 or nowNs() - start < budget_ns)) : (count += 1) {
        const t0 = nowNs();
        for (0..inner) |_| f.run();
        xs[count] = @as(f64, @floatFromInt(nowNs() - t0)) * 1e-3 / @as(f64, @floatFromInt(inner));
    }
    assert(count >= 3);
    std.mem.sort(f64, xs[0..count], {}, std.sort.asc(f64));
    return xs[count / 2];
}

fn ms(ns: u64) f64 {
    return @as(f64, @floatFromInt(ns)) * 1e-6;
}

/// Least squares t = at0 + slope * bytes.
fn leastSquares(bytes: []const usize, ts: []const f64) Line {
    assert(bytes.len == ts.len);
    assert(bytes.len >= 2);
    var sx: f64 = 0;
    var sy: f64 = 0;
    var sxx: f64 = 0;
    var sxy: f64 = 0;
    for (bytes, ts) |b, t| {
        const x: f64 = @floatFromInt(b);
        sx += x;
        sy += t;
        sxx += x * x;
        sxy += x * t;
    }
    const k: f64 = @floatFromInt(bytes.len);
    const slope = (k * sxy - sx * sy) / (k * sxx - sx * sx);
    return .{ .at0 = (sy - slope * sx) / k, .slope = slope };
}

/// The first `a-b` (or lone `a`) of a kernel CPU list; null when empty.
fn range(list: []const u8) ?Cpus.Range {
    const head = list[0 .. std.mem.indexOfScalar(u8, list, ',') orelse list.len];
    if (head.len == 0) return null;
    const dash = std.mem.indexOfScalar(u8, head, '-');
    const first = std.fmt.parseInt(u16, head[0 .. dash orelse head.len], 10) catch return null;
    const last = if (dash) |d| std.fmt.parseInt(u16, head[d + 1 ..], 10) catch return null else first;
    if (last < first or last >= cpus_max) return null;
    return .{ .first = first, .last = last };
}

fn cpuSet(first: u16, last: u16) linux.cpu_set_t {
    assert(first <= last);
    var set: linux.cpu_set_t = @splat(0);
    const bits = @bitSizeOf(usize);
    for (first..@as(usize, last) + 1) |cpu| set[cpu / bits] |= @as(usize, 1) << @intCast(cpu % bits);
    return set;
}

/// The calling thread's CPU set, or null where the kernel will not say.
fn affinity() ?linux.cpu_set_t {
    var set: linux.cpu_set_t = undefined;
    if (linux.sched_getaffinity(0, @sizeOf(linux.cpu_set_t), &set) != 0) return null;
    return set;
}

/// Best effort: a refused pin times on whatever core the thread has.
fn setAffinity(set: linux.cpu_set_t) void {
    linux.sched_setaffinity(0, &set) catch {};
}

/// Fills `grid`'s Laplacian pattern in CSC (rows ascending per column) and
/// returns its entry count.
fn laplacian(grid: Grid, col_ptr: []u32, row_idx: []u32) u32 {
    const n: u32 = @intCast(col_ptr.len - 1);
    var strides: [3]u32 = .{ 1, grid.side, grid.side * grid.side };
    var q: u32 = 0;
    for (0..n) |jj| {
        const j: u32 = @intCast(jj);
        col_ptr[j] = q;
        // Below the diagonal, farthest first, then above, nearest first.
        var d: usize = grid.dims;
        while (d > 0) {
            d -= 1;
            const s = strides[d];
            if ((j / s) % grid.side > 0) {
                row_idx[q] = j - s;
                q += 1;
            }
        }
        row_idx[q] = j;
        q += 1;
        for (strides[0..grid.dims]) |s| if ((j / s) % grid.side + 1 < grid.side) {
            row_idx[q] = j + s;
            q += 1;
        };
    }
    col_ptr[n] = q;
    assert(q <= row_idx.len);
    return q;
}

test "calibrate: Laplacian pattern rows ascend and pair across the diagonal" {
    // Oracle: a 3x3 2-D grid by hand: corners have 3 entries, edges 4, the
    // centre 5: 4*3 + 4*4 + 5 = 33.
    var col_ptr: [10]u32 = undefined;
    var row_idx: [45]u32 = undefined;
    const nnz = laplacian(.{ .side = 3, .dims = 2 }, &col_ptr, &row_idx);
    try std.testing.expectEqual(@as(u32, 33), nnz);
    try std.testing.expectEqualSlices(u32, &.{ 1, 3, 4, 5, 7 }, row_idx[col_ptr[4]..col_ptr[5]]);
    for (0..9) |j| {
        const rows = row_idx[col_ptr[j]..col_ptr[j + 1]];
        for (rows[1..], rows[0 .. rows.len - 1]) |b, a| try std.testing.expect(a < b);
        // Symmetric: (i, j) present iff (j, i) is.
        for (rows) |i| try std.testing.expect(std.mem.indexOfScalar(u32, row_idx[col_ptr[i]..col_ptr[i + 1]], @intCast(j)) != null);
    }
}

test "calibrate: CPU lists, the least-squares line" {
    try std.testing.expectEqual(Cpus.Range{ .first = 0, .last = 15 }, range("0-15").?);
    try std.testing.expectEqual(Cpus.Range{ .first = 16, .last = 16 }, range("16,18-19").?);
    try std.testing.expect(range("") == null);
    try std.testing.expect(range("9-3") == null);
    // Oracle: points on t = 5 + 2 b exactly.
    const l = leastSquares(&.{ 1, 10, 100 }, &.{ 7, 25, 205 });
    try std.testing.expectApproxEqAbs(@as(f64, 5), l.at0, 1e-9);
    try std.testing.expectApproxEqAbs(@as(f64, 2), l.slope, 1e-12);
}
