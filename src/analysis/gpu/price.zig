//! The placement price of a GPU query. In: one `Row` per GPU-eligible
//! batch (its width, one host thread's eval of it, the bytes each instance
//! holds). Out: host and device microseconds per eval, per batch and for the
//! query, and the history walk's share alone, which needs no probe. Pure
//! arithmetic; `admit` (context_cost.zig) measures `host_us`.
//!
//! The device runs one thread per instance, `slowdown` times slower than a
//! host core on the same instance, `concurrency` threads at once. So a batch
//! wins on the device only past ~`slowdown` instances: one waveform-history
//! line loses there, a batch of a thousand wins.
//!
//! Contents:
//! - `pub const Row = struct`
//! - `pub const Cost = struct`
//! - `pub const slowdown: f64`
//! - `pub const concurrency: f64`
//! - `pub const hist_ns_per_byte: f64`
//! - `pub const hist_floor_bytes: f64`
//! - `pub fn ofBatch(r: Row) Cost`
//! - `pub fn ofQuery(rows: []const Row, device_fixed_us: f64, lanes: f64) Cost`
//! - `pub fn ofHistory(rows: []const Row) Cost`
//! - `pub fn historyLoses(c: Cost) bool`

const std = @import("std");
const assert = std.debug.assert;

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

/// One device thread against one host core on the same instance's eval,
/// measured on an RTX 4060 Laptop (sm_89) against an i9-14900HX, ReleaseSafe,
/// one instance per batch (2026-11-02): dc/device_hisim2 1328 us against
/// 10 us per eval (130x, `ESPICE_GPU_STATS` wait against the `auto` probe),
/// tran/ltra_tran_long_run_past_8192 39.2 ms against 444 us (88x) and
/// tran/cpl_tran_history_past_2048 7.83 ms against 42 us (186x), the last
/// two from `--timing-in-depth`'s eval split over the whole run.
pub const slowdown: f64 = 120;

/// Device threads that run at once, so a batch past it takes
/// `ceil(count / concurrency)` waves. Two waves at 2000 instances give the
/// device 0.12 of one host thread's time, against 0.093 measured on
/// mos1 (42 us against ~450 us) and 0.096 on bsim4va (~700 us against
/// ~7.3 ms).
pub const concurrency: f64 = 1000;

/// Host nanoseconds per held byte past `hist_floor_bytes` per eval: the
/// history walk the probe, at an empty history, cannot see.
/// cpl_tran_history_past_2048 42 us over 147 KB (0.29),
/// ltra_tran_long_run_past_8192 444 us over 320 KB (1.36, averaged over
/// its fill). The lower keeps the device's side of `historyLoses` short.
pub const hist_ns_per_byte: f64 = 0.3;

/// Held bytes the probe already pays for: an instance's own fields, a few
/// hundred bytes for a compact model. Past them, the bytes are a history.
pub const hist_floor_bytes: f64 = 4096;

/// `historyLoses`: the device's estimate must exceed the host's by this
/// factor...
const host_ratio: f64 = 4;

/// ...and by this much per eval.
const host_loss_us: f64 = 1000;

/// The batch's price on each side; NaN in, NaN out.
pub fn ofBatch(r: Row) Cost {
    assert(r.count > 0);
    assert(r.host_us >= 0 or std.math.isNan(r.host_us));
    const count: f64 = @floatFromInt(r.count);
    const cpu_us = r.host_us + count * @max(0, r.held_bytes - hist_floor_bytes) * hist_ns_per_byte * 1e-3;
    const waves = @ceil(count / concurrency);
    return .{ .cpu_us = cpu_us, .gpu_us = cpu_us * slowdown * waves / count };
}

/// The query's price: `device_fixed_us` (launches, the wait, the bus) plus
/// every batch on the device, against the host's eval spread over `lanes`.
pub fn ofQuery(rows: []const Row, device_fixed_us: f64, lanes: f64) Cost {
    assert(lanes >= 1);
    var total: Cost = .{ .cpu_us = 0, .gpu_us = device_fixed_us };
    for (rows) |r| {
        const c = ofBatch(r);
        total.cpu_us += c.cpu_us / lanes;
        total.gpu_us += c.gpu_us;
    }
    return total;
}

/// The history walk's price alone: `ofQuery` with no probe and no fixed
/// cost, from counts and held bytes only, so the same on every run.
pub fn ofHistory(rows: []const Row) Cost {
    var total: Cost = .{ .cpu_us = 0, .gpu_us = 0 };
    for (rows) |r| {
        const c = ofBatch(.{ .count = r.count, .host_us = 0, .held_bytes = r.held_bytes });
        total.cpu_us += c.cpu_us;
        total.gpu_us += c.gpu_us;
    }
    return total;
}

/// Whether `ofHistory`'s price loses on the device by far: `host_ratio`
/// times and `host_loss_us` per eval, so a compact model's few hundred held
/// bytes never trip it.
pub fn historyLoses(c: Cost) bool {
    return c.gpu_us > host_ratio * c.cpu_us and c.gpu_us - c.cpu_us > host_loss_us;
}

test "price: one thread is slowdown times a core, a wide batch shares waves" {
    const one = ofBatch(.{ .count = 1, .host_us = 10, .held_bytes = 0 });
    try std.testing.expectEqual(@as(f64, 10), one.cpu_us);
    try std.testing.expectEqual(10 * slowdown, one.gpu_us);
    // 2000 instances: two waves, 0.12 of the host thread.
    const wide = ofBatch(.{ .count = 2000, .host_us = 7300, .held_bytes = 0 });
    try std.testing.expectApproxEqRel(@as(f64, 876), wide.gpu_us, 1e-12);
    try std.testing.expect(std.math.isNan(ofBatch(.{ .count = 1, .host_us = std.math.nan(f64), .held_bytes = 0 }).gpu_us));
}

test "price: one line loses its history walk on the device, a thousand win it" {
    // Per-instance held bytes of the shipped lines (models/*.va caps).
    const ltra: f64 = 5 * 8192 * 8;
    const cpl: f64 = (1 + 4 * 2) * 2048 * 8;
    const txl: f64 = 5 * 2048 * 8;
    for ([_]f64{ ltra, cpl, txl }) |bytes| {
        const one = ofBatch(.{ .count = 1, .host_us = 0, .held_bytes = bytes });
        try std.testing.expect(one.gpu_us > one.cpu_us);
        try std.testing.expect(historyLoses(ofHistory(&.{.{ .count = 1, .host_us = 0, .held_bytes = bytes }})));
        const wide = ofBatch(.{ .count = 2000, .host_us = 0, .held_bytes = bytes });
        try std.testing.expect(wide.gpu_us * 4 < wide.cpu_us);
        try std.testing.expect(!historyLoses(ofHistory(&.{.{ .count = 2000, .host_us = 0, .held_bytes = bytes }})));
    }
    // A compact model's instance holds no history the probe missed.
    try std.testing.expect(!historyLoses(ofHistory(&.{.{ .count = 1, .host_us = 0, .held_bytes = 2048 }})));
}

test "price: ofQuery prices every batch on the device" {
    const rows = [_]Row{
        .{ .count = 2, .host_us = 0.1, .held_bytes = 64 },
        .{ .count = 1, .host_us = 0, .held_bytes = 5 * 8192 * 8 },
    };
    const q = ofQuery(&rows, 40, 1);
    const a = ofBatch(rows[0]);
    const b = ofBatch(rows[1]);
    try std.testing.expectApproxEqRel(a.cpu_us + b.cpu_us, q.cpu_us, 1e-12);
    try std.testing.expectApproxEqRel(40 + a.gpu_us + b.gpu_us, q.gpu_us, 1e-12);
    try std.testing.expect(q.gpu_us > q.cpu_us);
}
