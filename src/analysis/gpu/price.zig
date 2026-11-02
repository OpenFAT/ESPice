//! The placement price of a GPU query. In: one `Row` per GPU-eligible
//! batch (its width, one host thread's eval of it, the bytes each instance
//! holds). Out: host and device microseconds per eval, per batch and for the
//! query, and which batches stay on the host under any backend. Pure
//! arithmetic; `admit` (context_cost.zig) measures `host_us`.
//!
//! The device runs one thread per instance, `slowdown` times slower than a
//! host core on the same instance, `concurrency` threads at once. So a batch
//! wins on the device only past ~`slowdown` instances, and a held waveform
//! history, which one thread walks serially, never wins on its own.
//!
//! Contents:
//! - `pub const Row = struct`
//! - `pub const Cost = struct`
//! - `pub const slowdown: f64`
//! - `pub const concurrency: f64`
//! - `pub const hist_ns_per_byte: f64`
//! - `pub const hist_floor_bytes: f64`
//! - `pub const host_ratio: f64`
//! - `pub fn ofBatch(r: Row) Cost`
//! - `pub fn staysOnHost(count: u32, held_bytes: f64) bool`
//! - `pub fn ofQuery(rows: []const Row, device_fixed_us: f64, lanes: f64) Cost`

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
/// its fill). The lower keeps the device's side of `staysOnHost` short.
pub const hist_ns_per_byte: f64 = 0.3;

/// Held bytes the probe already pays for: an instance's own fields, a few
/// hundred bytes for a compact model. Past them, the bytes are a history.
pub const hist_floor_bytes: f64 = 4096;

/// `staysOnHost`: the device's estimate must exceed the host's by this
/// factor...
pub const host_ratio: f64 = 4;

/// ...and by this much per eval, so a compact model's few hundred held
/// bytes never leave an explicit device run.
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

/// Whether a batch of `count` instances holding `held_bytes` each stays on
/// the host under every backend. Priced on the history alone, no probe, so
/// the verdict is the same on every run: ltra (320 KB), cpl (147 KB) and
/// txl (80 KB) stay; compact models (well under 2 KB) do not.
/// ponytail: a history-heavy batch is all-or-nothing on one side; a kernel
/// that walks one instance's history across a warp (#226) would lift the
/// device's side by up to 32x.
pub fn staysOnHost(count: u32, held_bytes: f64) bool {
    assert(held_bytes >= 0);
    if (count == 0) return false;
    const c = ofBatch(.{ .count = count, .host_us = 0, .held_bytes = held_bytes });
    return c.gpu_us > host_ratio * c.cpu_us and c.gpu_us - c.cpu_us > host_loss_us;
}

/// The query's price: `device_fixed_us` (launches, the wait, the bus) plus
/// the device's batches, against the host's eval spread over `lanes`. A
/// `staysOnHost` batch costs its host time on both sides.
pub fn ofQuery(rows: []const Row, device_fixed_us: f64, lanes: f64) Cost {
    assert(lanes >= 1);
    var total: Cost = .{ .cpu_us = 0, .gpu_us = device_fixed_us };
    for (rows) |r| {
        const c = ofBatch(r);
        total.cpu_us += c.cpu_us / lanes;
        total.gpu_us += if (staysOnHost(r.count, r.held_bytes)) c.cpu_us / lanes else c.gpu_us;
    }
    return total;
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

test "price: staysOnHost: the history lines stay, compact models and wide batches go" {
    // Per-instance held bytes of the shipped lines (models/*.va caps).
    const ltra: f64 = 5 * 8192 * 8;
    const cpl: f64 = (1 + 4 * 2) * 2048 * 8;
    const txl: f64 = 5 * 2048 * 8;
    for ([_]f64{ ltra, cpl, txl }) |bytes| try std.testing.expect(staysOnHost(1, bytes));
    // Ten ltra lines still lose: 10x the host's walk against 120x a core.
    try std.testing.expect(staysOnHost(10, ltra));
    // A compact model's instance, alone or 2000 wide.
    try std.testing.expect(!staysOnHost(1, 2048));
    try std.testing.expect(!staysOnHost(2000, 2048));
    // Enough lines fill the device, and it wins them back.
    try std.testing.expect(!staysOnHost(1000, txl));
    try std.testing.expect(!staysOnHost(0, ltra));
}

test "price: ofQuery: a history batch prices its host time on both sides" {
    const rows = [_]Row{
        .{ .count = 2, .host_us = 0.1, .held_bytes = 64 },
        .{ .count = 1, .host_us = 0, .held_bytes = 5 * 8192 * 8 },
    };
    const q = ofQuery(&rows, 40, 1);
    const r = ofBatch(rows[0]);
    const line = ofBatch(rows[1]).cpu_us;
    try std.testing.expectApproxEqRel(r.cpu_us + line, q.cpu_us, 1e-12);
    try std.testing.expectApproxEqRel(40 + r.gpu_us + line, q.gpu_us, 1e-12);
    // The CPU wins this query: a 40 us floor over a ~0.1 us host eval.
    try std.testing.expect(q.gpu_us > q.cpu_us);
}
