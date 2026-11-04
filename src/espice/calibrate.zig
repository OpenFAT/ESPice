//! `espice calibrate`: measures this machine and writes the calibration file
//! `auto`'s price reads (`analysis.machine`). In: the live identity from
//! sysfs/procfs, the GPU driver, and decks this file generates (one per
//! compact model and size, through the frontend). Out: the file, written
//! atomically through `machine.write` and read back bit for bit, and one
//! progress line per stage. The measurements are `analysis.calibrate`'s;
//! this is their spine and the constants derived from them.
//! Cites: the price plan sec. 4 (the rows, how each is measured, the ~25 s
//! budget, `--compile-all`) and sec. 9 stage 3.
//!
//! Contents:
//! - `pub const Options = struct`
//! - `pub const Error`
//! - `pub fn run(gpa: Allocator, io: std.Io, opts: Options, log: *std.Io.Writer) !void`
//! - `pub fn probe(which: Probe, out: *std.Io.Writer) !void`, `pub const Probe`

const std = @import("std");
const assert = std.debug.assert;
const Allocator = std.mem.Allocator;
const analysis = @import("analysis");
const frontend = @import("frontend");

const machine = analysis.machine;
const cal = analysis.calibrate;

/// What `run` measures and where it writes.
pub const Options = struct {
    /// Also load heavy images the driver has not compiled (minutes each,
    /// once per driver); without it they get no row and stay on the host.
    compile_all: bool = false,
    /// Root of sysfs and procfs: "" for the live system, a fake tree in tests.
    sys_root: []const u8 = "",
    /// The cache root (`xdg_cache_home`, `home`) the file goes under.
    knobs: analysis.Knobs = .{},
    /// This executable, which the start probes run as children.
    exe: []const u8 = "/proc/self/exe",
    /// The espice version the file records.
    version: []const u8,
};

pub const Error = error{
    /// This build carries no kernel images.
    NoGpuArtifacts,
    /// sysfs shows no GPU driver, or procfs no card: nothing to measure.
    NoGpuDriver,
    /// A start probe child failed or printed no time.
    ProbeFailed,
    /// No model or no LU matrix could be measured.
    NothingMeasured,
    /// The written file did not read back to the bytes written.
    ReadBackMismatch,
};

/// The two children `run` times: one that starts the driver and one that
/// exits at once (`espice calibrate --probe-start` / `--probe-none`).
pub const Probe = enum { start, none };

/// Measures every row of the calibration file and writes it. Fails, with
/// nothing written, on `Error`, a driver error while measuring the card, a
/// write error (`machine.WriteError`) or out of memory. A model that cannot
/// be built or measured is skipped with a line on `log`: it has no row, so
/// `auto` keeps it on the host.
pub fn run(gpa: Allocator, io: std.Io, opts: Options, log: *std.Io.Writer) !void {
    const t_run = analysis.nowNs();
    const vendor = machine.vendor orelse return Error.NoGpuArtifacts;
    const id = machine.identity(opts.sys_root, vendor);
    if (id.driver.len == 0 or id.cards == 0) return Error.NoGpuDriver;
    const cpus = cal.Cpus.parse(&id);

    // Start: children, before this process touches the driver.
    var start = [2][probe_runs]f64{ undefined, undefined };
    var start_in: [probe_runs]f64 = undefined;
    for (0..probe_runs) |r| {
        start[0][r] = try child(gpa, io, opts.exe, .none, null);
        start[1][r] = try child(gpa, io, opts.exe, .start, &start_in[r]);
    }
    const start_ms = @max(0, median(&start[1]) - median(&start[0]));
    const start_in_ms = median(&start_in);
    try log.print("calibrate: driver {s}: start {d:.0} ms ({d:.0} in process)\n", .{ id.driver.slice(), start_ms, start_in_ms });
    try log.flush();

    const dev = try cal.device();
    try log.print("calibrate: launch {d:.1} us, bus {d:.1} us + {d:.1}/{d:.1} GB/s up/down, copy {d:.0} GB/s, FP64 1:{d}, {d} SMs\n", .{ dev.launch_us, dev.bus_lat_us, dev.bus_up_gbs, dev.bus_down_gbs, dev.copy_b_per_us * 1e-3, dev.fp64_ratio, dev.sms });
    try log.flush();

    var arena_state: std.heap.ArenaAllocator = .init(gpa);
    defer arena_state.deinit();
    const arena = arena_state.allocator();
    var lus: std.ArrayList(machine.Lu) = .empty;
    for (grids) |g| {
        const row = cal.lu(gpa, g, opts.knobs) catch |e| {
            try log.print("calibrate: lu {d}^{d}: skipped ({s})\n", .{ g.side, g.dims, @errorName(e) });
            continue;
        };
        try lus.append(arena, row);
        try log.print("calibrate: lu n={d} F/n={d:.0}: device {d:.0} us, host {d:.0} us, tables {d:.1} ms\n", .{ row.n, @as(f64, @floatFromInt(row.flops)) / @as(f64, @floatFromInt(row.n)), row.dev_us, row.host_us, row.table_us * 1e-3 });
    }
    try log.flush();

    var models: std.ArrayList(machine.Model) = .empty;
    for (cards) |card| {
        const row = measureModel(gpa, io, card, cpus, opts) catch |e| {
            try log.print("calibrate: {s}: skipped ({s})\n", .{ card.name, @errorName(e) });
            try log.flush();
            continue;
        };
        // Two cards on one kernel image measure it once.
        if (for (models.items) |m| {
            if (m.image == row.image) break true;
        } else false) continue;
        try models.append(arena, row);
        try log.print("calibrate: {s}: device {d:.0}/{d:.0} us, host P {d:.1}/{d:.0} us, E/P {d:.2}, lanes {d:.2}x, jit {d:.0} ms (1/16384 instances)\n", .{ card.name, row.dev_us[0], row.dev_us[3], row.host_p_us[0], row.host_p_us[3], row.host_e_us[2] / row.host_p_us[2], row.lane_gain, row.jit_ms });
        try log.flush();
    }
    if (models.items.len == 0 or lus.items.len == 0) return Error.NothingMeasured;

    const m: machine.Machine = .{
        .id = id,
        .espice = machine.Text.init(opts.version) orelse .{},
        .c = derive(dev, start_ms, start_in_ms, models.items, lus.items),
        .models = models.items,
        .lus = lus.items,
    };
    try machine.write(gpa, &m, opts.knobs);
    // The pair to `write`: the file must read back to the bytes formatted.
    const back = machine.load(arena, opts.knobs, &id) orelse return Error.ReadBackMismatch;
    var a: std.Io.Writer.Allocating = .init(arena);
    var b: std.Io.Writer.Allocating = .init(arena);
    try machine.format(&m, &a.writer);
    try machine.format(&back, &b.writer);
    if (!std.mem.eql(u8, a.written(), b.written())) return Error.ReadBackMismatch;
    var buf: [512]u8 = undefined;
    try log.print("calibrate: wrote {s} ({d} models, {d} LU rows) in {d:.1} s\n", .{ machine.filePath(&buf, opts.knobs) orelse "?", models.items.len, lus.items.len, @as(f64, @floatFromInt(analysis.nowNs() - t_run)) * 1e-9 });
    try log.flush();
}

/// A start probe child's body: `.start` brings the driver up and prints
/// the in-process ms on `out`; `.none` does nothing.
pub fn probe(which: Probe, out: *std.Io.Writer) !void {
    switch (which) {
        .none => {},
        .start => try out.print("{d}\n", .{try cal.probeStart()}),
    }
}

// ---- private ----

/// Children per start probe (the plan's median of 7).
const probe_runs = 7;

/// The LU matrices: 2-D grids for low flops per column, 3-D for high
/// (the plan's F/n ~60 to ~4000), each at a size the host factors fast.
const grids = [_]cal.Grid{
    .{ .side = 48, .dims = 2 },
    .{ .side = 160, .dims = 2 },
    .{ .side = 16, .dims = 3 },
    .{ .side = 28, .dims = 3 },
};

/// One generated deck's card: the letter, the `.model` type and LEVEL
/// (`frontend` spice.zig's tables), and the terminals it wires.
const Card = struct { name: []const u8, letter: u8, kind: []const u8, level: u16, nodes: u8 };

/// Every LEVEL-table compact model, one LEVEL each. A model this build
/// lacks fails to build and is skipped.
const cards = [_]Card{
    .{ .name = "mos1", .letter = 'm', .kind = "nmos", .level = 1, .nodes = 4 },
    .{ .name = "mos2", .letter = 'm', .kind = "nmos", .level = 2, .nodes = 4 },
    .{ .name = "mos3", .letter = 'm', .kind = "nmos", .level = 3, .nodes = 4 },
    .{ .name = "bsim1", .letter = 'm', .kind = "nmos", .level = 4, .nodes = 4 },
    .{ .name = "bsim2", .letter = 'm', .kind = "nmos", .level = 5, .nodes = 4 },
    .{ .name = "mos6", .letter = 'm', .kind = "nmos", .level = 6, .nodes = 4 },
    .{ .name = "bsim3", .letter = 'm', .kind = "nmos", .level = 8, .nodes = 4 },
    .{ .name = "mos9", .letter = 'm', .kind = "nmos", .level = 9, .nodes = 4 },
    .{ .name = "bsimsoi", .letter = 'm', .kind = "nmos", .level = 10, .nodes = 4 },
    .{ .name = "bsim4va", .letter = 'm', .kind = "nmos", .level = 54, .nodes = 4 },
    .{ .name = "b3soifd", .letter = 'm', .kind = "nmos", .level = 55, .nodes = 4 },
    .{ .name = "b3soidd", .letter = 'm', .kind = "nmos", .level = 56, .nodes = 4 },
    .{ .name = "b3soipd", .letter = 'm', .kind = "nmos", .level = 57, .nodes = 4 },
    .{ .name = "soi3", .letter = 'm', .kind = "nmos", .level = 60, .nodes = 4 },
    .{ .name = "hisim2", .letter = 'm', .kind = "nmos", .level = 68, .nodes = 4 },
    .{ .name = "hisimhv", .letter = 'm', .kind = "nmos", .level = 73, .nodes = 4 },
    .{ .name = "psp103", .letter = 'm', .kind = "nmos", .level = 1040, .nodes = 4 },
    .{ .name = "bsimcmg", .letter = 'm', .kind = "nmos", .level = 72, .nodes = 4 },
    .{ .name = "l_utsoi", .letter = 'm', .kind = "nmos", .level = 76, .nodes = 4 },
    .{ .name = "bsimbulk", .letter = 'm', .kind = "nmos", .level = 77, .nodes = 4 },
    .{ .name = "bsimimg", .letter = 'm', .kind = "nmos", .level = 78, .nodes = 4 },
    .{ .name = "gummel_poon", .letter = 'q', .kind = "npn", .level = 1, .nodes = 3 },
    .{ .name = "vbic", .letter = 'q', .kind = "npn", .level = 4, .nodes = 3 },
    .{ .name = "mextram", .letter = 'q', .kind = "npn", .level = 6, .nodes = 3 },
    .{ .name = "hicumL2", .letter = 'q', .kind = "npn", .level = 8, .nodes = 3 },
    .{ .name = "diode", .letter = 'd', .kind = "d", .level = 1, .nodes = 2 },
    .{ .name = "juncap200", .letter = 'd', .kind = "d", .level = 6, .nodes = 2 },
    .{ .name = "jfet", .letter = 'j', .kind = "njf", .level = 1, .nodes = 3 },
    .{ .name = "jfet2", .letter = 'j', .kind = "njf", .level = 2, .nodes = 3 },
    .{ .name = "mes", .letter = 'z', .kind = "nmf", .level = 1, .nodes = 3 },
    .{ .name = "mesa", .letter = 'z', .kind = "nmf", .level = 2, .nodes = 3 },
    .{ .name = "hfet1", .letter = 'z', .kind = "nmf", .level = 5, .nodes = 3 },
    .{ .name = "hfet2", .letter = 'z', .kind = "nmf", .level = 6, .nodes = 3 },
};

/// Gate nets the instances share, so the cost probe's spread x puts them
/// in a mix of regions.
const shared_nets = 13;

/// Policy, not measurement: the stage-2 price's bars and its history-walk
/// terms, carried until stage 4 derives the margin from the placement
/// oracle and prices history from the model rows (price plan sec. 5, 7).
const policy = .{ .margin = 1.3, .min_ratio = 2.0, .hist_ns_per_byte = 0.3, .hist_floor_bytes = 4096.0, .host_ratio = 4.0, .host_loss_us = 1000.0 };

/// Wall of one start-probe child, ms; `.start` also returns the child's
/// own in-process ms in `in_ms`.
fn child(gpa: Allocator, io: std.Io, exe: []const u8, which: Probe, in_ms: ?*f64) !f64 {
    const t0 = analysis.nowNs();
    const r = std.process.run(gpa, io, .{ .argv = &.{ exe, "calibrate", if (which == .start) "--probe-start" else "--probe-none" }, .stdout_limit = .limited(256), .stderr_limit = .limited(4096) }) catch return Error.ProbeFailed;
    const wall = @as(f64, @floatFromInt(analysis.nowNs() - t0)) * 1e-6;
    defer gpa.free(r.stdout);
    defer gpa.free(r.stderr);
    if (r.term != .exited or r.term.exited != 0) return Error.ProbeFailed;
    if (in_ms) |p| p.* = std.fmt.parseFloat(f64, std.mem.trim(u8, r.stdout, " \n")) catch return Error.ProbeFailed;
    return wall;
}

/// `card`'s row: one generated deck per `machine.model_sizes` count, the
/// lane gain on the widest.
fn measureModel(gpa: Allocator, io: std.Io, card: Card, cpus: cal.Cpus, opts: Options) !machine.Model {
    var row: machine.Model = .{ .image = 0, .dev_us = undefined, .host_p_us = undefined, .host_e_us = undefined, .jit_ms = 0, .lane_gain = 1 };
    for (machine.model_sizes, 0..) |count, k| {
        var arena_state: std.heap.ArenaAllocator = .init(gpa);
        defer arena_state.deinit();
        const arena = arena_state.allocator();
        const text = try deck(arena, card, count);
        var library = try frontend.Library.init(gpa);
        defer library.deinit();
        const ast = try frontend.prepare(io, arena, .{ .bytes = .{ .data = text, .origin = "calibrate.sp" } }, .ngspice, &.{}, .{});
        try frontend.loadModels(io, &library, arena, ast, "calibrate.sp", &.{});
        var prepared = try frontend.build(&library, arena, arena, ast);
        defer prepared.deinit();
        const widest = k + 1 == machine.model_sizes.len;
        const s = try cal.sample(gpa, io, &prepared.circuit, opts.knobs, cpus, if (widest) analysis.ThreadPool.per_query else 1, opts.compile_all);
        if (k == 0) {
            row.image = s.image;
            row.jit_ms = s.jit_ms;
        }
        assert(s.image == row.image);
        row.dev_us[k] = s.dev_us;
        row.host_p_us[k] = s.host_p_us;
        row.host_e_us[k] = s.host_e_us;
        if (widest) row.lane_gain = s.lane_gain;
    }
    return row;
}

/// `count` instances of `card`'s model: each its own first net, the second
/// on one of `shared_nets`, the rest on ground.
fn deck(arena: Allocator, card: Card, count: u32) ![]const u8 {
    var w: std.Io.Writer.Allocating = .init(arena);
    try w.writer.print("* espice calibrate: {s}, {d} instances\n.model cm {s} level={d}\n", .{ card.name, count, card.kind, card.level });
    for (0..count) |i| {
        try w.writer.print("{c}{d} a{d}", .{ card.letter, i, i });
        if (card.nodes > 1) try w.writer.print(" b{d}", .{i % shared_nets});
        for (2..card.nodes) |_| try w.writer.writeAll(" 0");
        try w.writer.writeAll(" cm\n");
    }
    try w.writer.writeAll(".end\n");
    return w.written();
}

/// The stage-2 price's constants from the measurements (each named for
/// its row in `machine.Constants`).
fn derive(dev: cal.Device, start_ms: f64, start_in_ms: f64, models: []const machine.Model, lus: []const machine.Lu) machine.Constants {
    assert(models.len > 0);
    assert(lus.len > 0);
    var scratch: [128]f64 = undefined;
    const ms_ = models[0..@min(models.len, scratch.len)];
    // One instance of the cheapest model on the device is the launches,
    // the copies and the wait: the fixed cost of an eval.
    var fixed: f64 = std.math.inf(f64);
    for (ms_) |m| fixed = @min(fixed, m.dev_us[0]);
    // 64 instances run in one wave, so their kernel time is one thread's
    // eval: `slowdown` host instances. Only models whose kernel shows past
    // the fixed cost say anything.
    var n_s: usize = 0;
    for (ms_) |m| if (m.dev_us[1] > 2 * fixed) {
        scratch[n_s] = (m.dev_us[1] - fixed) / (m.host_p_us[1] / 64);
        n_s += 1;
    };
    const slowdown = if (n_s == 0) 1 else @max(1, median(scratch[0..n_s]));
    // 16384 instances take `waves` of that one-thread time.
    var n_c: usize = 0;
    for (ms_) |m| if (m.dev_us[1] > 2 * fixed) {
        const s = (m.dev_us[1] - fixed) / (m.host_p_us[1] / 64);
        const waves = @max(1, (m.dev_us[3] - fixed) / (s * m.host_p_us[3] / 16384));
        scratch[n_c] = 16384 / waves;
        n_c += 1;
    };
    const concurrency = if (n_c == 0) 16384 else std.math.clamp(median(scratch[0..n_c]), 1, 16384);
    const lanes: f64 = @floatFromInt(analysis.ThreadPool.per_query);
    for (ms_, 0..) |m, i| scratch[i] = @max(0, (m.lane_gain - 1) / (lanes - 1));
    const lane_gain = median(scratch[0..ms_.len]);
    for (ms_, 0..) |m, i| scratch[i] = m.host_e_us[2] / m.host_p_us[2];
    const pe_ratio = median(scratch[0..ms_.len]);
    const ls = lus[0..@min(lus.len, scratch.len)];
    for (ls, 0..) |l, i| scratch[i] = l.table_us * 1e3 / @as(f64, @floatFromInt(l.flops));
    const table = median(scratch[0..ls.len]);
    return .{
        .start_ms = start_ms,
        .eval_fixed_us = fixed,
        .bus_b_per_us = 1e3 * 2 / (1 / dev.bus_up_gbs + 1 / dev.bus_down_gbs),
        .stage_b_per_us = dev.copy_b_per_us,
        .host_lane_gain = lane_gain,
        .margin = policy.margin,
        .min_ratio = policy.min_ratio,
        .slowdown = slowdown,
        .concurrency = concurrency,
        .hist_ns_per_byte = policy.hist_ns_per_byte,
        .hist_floor_bytes = policy.hist_floor_bytes,
        .host_ratio = policy.host_ratio,
        .host_loss_us = policy.host_loss_us,
        .fp64_ratio = dev.fp64_ratio,
        .sms = dev.sms,
        .cc = dev.cc,
        .mem_gb = dev.mem_gb,
        .start_in_ms = start_in_ms,
        .launch_us = dev.launch_us,
        .bus_lat_us = dev.bus_lat_us,
        .bus_up_gbs = dev.bus_up_gbs,
        .bus_down_gbs = dev.bus_down_gbs,
        .lu_table_ns_per_flop = table,
        .pe_ratio = pe_ratio,
    };
}

/// The median of `xs`, sorting it in place.
fn median(xs: []f64) f64 {
    assert(xs.len > 0);
    std.mem.sort(f64, xs, {}, std.sort.asc(f64));
    return xs[xs.len / 2];
}
