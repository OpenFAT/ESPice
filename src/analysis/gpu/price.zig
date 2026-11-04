//! The placement price of a GPU query. In: one `Row` per GPU-eligible
//! batch (its width, one host thread's eval of it, the bytes each instance
//! holds) and the machine's calibrated constants (`machine.Constants`).
//! Out: host and device microseconds per eval, per batch and for the query,
//! and the history walk's share alone, which needs no probe. Pure
//! arithmetic; `admit` (context_cost.zig) measures `host_us`.
//!
//! The device runs one thread per instance, `slowdown` times slower than a
//! host core on the same instance, `concurrency` threads at once. So a batch
//! wins on the device only past ~`slowdown` instances: one waveform-history
//! line loses there, a batch of a thousand wins.
//! Cites #126, #139 and the price plan sec. 4 (the constants' file).
//!
//! Contents:
//! - `pub const Row = struct`
//! - `pub const Cost = struct`
//! - `pub fn ofBatch(c: *const Constants, r: Row) Cost`
//! - `pub fn ofQuery(c: *const Constants, rows: []const Row, device_fixed_us: f64, lanes: f64) Cost`
//! - `pub fn ofHistory(c: *const Constants, rows: []const Row) Cost`
//! - `pub fn historyLoses(c: *const Constants, cost: Cost) bool`

const std = @import("std");
const assert = std.debug.assert;
const Constants = @import("../machine.zig").Constants;

/// One GPU-eligible batch, as the price sees it.
pub const Row = struct {
    /// Instances; the device runs one thread each.
    count: u32,
    /// One host thread's eval of the whole batch at the probe's x,
    /// microseconds. The probe runs before the first step, so it sees an
    /// empty history: `held_bytes` prices that part.
    host_us: f64,
    /// Bytes each instance holds (`GpuPayload.instances` plus `states`).
    held_bytes: f64,
};

/// Microseconds per eval on each side.
pub const Cost = struct { cpu_us: f64, gpu_us: f64 };

/// The batch's price on each side; NaN in, NaN out.
pub fn ofBatch(c: *const Constants, r: Row) Cost {
    assert(r.count > 0);
    assert(r.host_us >= 0 or std.math.isNan(r.host_us));
    assert(c.concurrency >= 1);
    const count: f64 = @floatFromInt(r.count);
    const cpu_us = r.host_us + count * @max(0, r.held_bytes - c.hist_floor_bytes) * c.hist_ns_per_byte * 1e-3;
    const waves = @ceil(count / c.concurrency);
    return .{ .cpu_us = cpu_us, .gpu_us = cpu_us * c.slowdown * waves / count };
}

/// The query's price: `device_fixed_us` (launches, the wait, the bus) plus
/// every batch on the device, against the host's eval spread over `lanes`.
pub fn ofQuery(c: *const Constants, rows: []const Row, device_fixed_us: f64, lanes: f64) Cost {
    assert(lanes >= 1);
    var total: Cost = .{ .cpu_us = 0, .gpu_us = device_fixed_us };
    for (rows) |r| {
        const b = ofBatch(c, r);
        total.cpu_us += b.cpu_us / lanes;
        total.gpu_us += b.gpu_us;
    }
    return total;
}

/// The history walk's price alone: `ofQuery` with no probe and no fixed
/// cost, from counts and held bytes only, so the same on every run.
pub fn ofHistory(c: *const Constants, rows: []const Row) Cost {
    var total: Cost = .{ .cpu_us = 0, .gpu_us = 0 };
    for (rows) |r| {
        const b = ofBatch(c, .{ .count = r.count, .host_us = 0, .held_bytes = r.held_bytes });
        total.cpu_us += b.cpu_us;
        total.gpu_us += b.gpu_us;
    }
    return total;
}

/// Whether `ofHistory`'s price loses on the device by far: `host_ratio`
/// times and `host_loss_us` per eval, so a compact model's few hundred held
/// bytes never trip it.
pub fn historyLoses(c: *const Constants, cost: Cost) bool {
    return cost.gpu_us > c.host_ratio * cost.cpu_us and cost.gpu_us - cost.cpu_us > c.host_loss_us;
}

// ---- private ----

/// The constants the tests price with: the placement fixture's.
fn fixture() !Constants {
    var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena.deinit();
    const machine = @import("../machine.zig");
    return (try machine.parse(arena.allocator(), machine.test_hooks.fixture)).c;
}

test "price: one thread is slowdown times a core, a wide batch shares waves" {
    const c = try fixture();
    const one = ofBatch(&c, .{ .count = 1, .host_us = 10, .held_bytes = 0 });
    try std.testing.expectEqual(@as(f64, 10), one.cpu_us);
    try std.testing.expectEqual(10 * c.slowdown, one.gpu_us);
    // 2000 instances: two waves, 0.12 of the host thread.
    const wide = ofBatch(&c, .{ .count = 2000, .host_us = 7300, .held_bytes = 0 });
    try std.testing.expectApproxEqRel(@as(f64, 876), wide.gpu_us, 1e-12);
    try std.testing.expect(std.math.isNan(ofBatch(&c, .{ .count = 1, .host_us = std.math.nan(f64), .held_bytes = 0 }).gpu_us));
}

test "price: one line loses its history walk on the device, a thousand win it" {
    const c = try fixture();
    // Per-instance held bytes of the shipped lines (models/*.va caps).
    const ltra: f64 = 5 * 8192 * 8;
    const cpl: f64 = (1 + 4 * 2) * 2048 * 8;
    const txl: f64 = 5 * 2048 * 8;
    for ([_]f64{ ltra, cpl, txl }) |bytes| {
        const one = ofBatch(&c, .{ .count = 1, .host_us = 0, .held_bytes = bytes });
        try std.testing.expect(one.gpu_us > one.cpu_us);
        try std.testing.expect(historyLoses(&c, ofHistory(&c, &.{.{ .count = 1, .host_us = 0, .held_bytes = bytes }})));
        const wide = ofBatch(&c, .{ .count = 2000, .host_us = 0, .held_bytes = bytes });
        try std.testing.expect(wide.gpu_us * 4 < wide.cpu_us);
        try std.testing.expect(!historyLoses(&c, ofHistory(&c, &.{.{ .count = 2000, .host_us = 0, .held_bytes = bytes }})));
    }
    // A compact model's instance holds no history the probe missed.
    try std.testing.expect(!historyLoses(&c, ofHistory(&c, &.{.{ .count = 1, .host_us = 0, .held_bytes = 2048 }})));
}

test "price: ofQuery prices every batch on the device" {
    const c = try fixture();
    const rows = [_]Row{
        .{ .count = 2, .host_us = 0.1, .held_bytes = 64 },
        .{ .count = 1, .host_us = 0, .held_bytes = 5 * 8192 * 8 },
    };
    const q = ofQuery(&c, &rows, 40, 1);
    const a = ofBatch(&c, rows[0]);
    const b = ofBatch(&c, rows[1]);
    try std.testing.expectApproxEqRel(a.cpu_us + b.cpu_us, q.cpu_us, 1e-12);
    try std.testing.expectApproxEqRel(40 + a.gpu_us + b.gpu_us, q.gpu_us, 1e-12);
    try std.testing.expect(q.gpu_us > q.cpu_us);
}
