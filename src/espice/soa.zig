//! Completed operating points, DC sweeps and transients in, `.option soa`
//! excursions out (#238): each result held to the deck's model-card limits
//! (`core.Soa`), as `soa:` lines (`print_soa`), as JSON objects
//! (`write_measures_json`), or as one result's count (`soa_count()`).
//! Cites: none.
//!
//! Contents:
//! - `pub const Tally = struct`
//! - `Tally.session(tally: *Tally, s: *const Session) !void`
//! - `Tally.result(tally: *Tally, s: *const Session, job: Query, kind: requests.Kind, res: output.Result) !void`
//! - `pub fn printSoa(self: *const Problem, out: *std.Io.Writer) !void`
//! - `pub fn checked(kind: requests.Kind) bool`

const std = @import("std");
const assert = std.debug.assert;
const analysis = @import("analysis");
const core = @import("core");
const output = @import("output");
const points = @import("points.zig");
const Problem = @import("Problem.zig");
const requests = core.query;
const Query = requests.Query;
const Session = analysis.session.Session;

/// `Problem.print_soa`'s running count: the excursions, the instances they
/// happened on, and the (limit, result) pairs a missing column skipped.
/// Each excursion prints to `out` while `max_warns` allows, and with `json`
/// is also a JSON object there, led by `,\n` (`write_measures_json`).
pub const Tally = struct {
    gpa: std.mem.Allocator,
    out: ?*std.Io.Writer = null,
    json: ?*std.Io.Writer = null,
    max_warns: u32 = 0,
    excursions: u64 = 0,
    instances: std.StringHashMapUnmanaged(void) = .empty,
    skipped: u64 = 0,

    /// Every SOA limit of `s`'s deck over each of its completed operating
    /// points, DC sweeps and transients, in request order.
    pub fn session(tally: *Tally, s: *const Session) !void {
        for (s.outputs.items) |id| {
            const info = try s.info(id);
            if (!checked(info.kind) or info.status != .complete or s.released(id)) continue;
            try tally.result(s, try s.query(id), info.kind, try s.result(id));
        }
    }

    /// Every SOA limit of `s`'s deck over `res`, the result of `job`.
    pub fn result(tally: *Tally, s: *const Session, job: Query, kind: requests.Kind, res: output.Result) !void {
        for (s.deck.soa.limits) |lim| try tally.scan(lim, res, s, job, kind);
    }

    // ---- private ----

    /// `lim` over every row of `res`; `kind` says what column 0 is: a
    /// transient's time, a DC sweep's value, nothing for an operating point.
    fn scan(tally: *Tally, lim: core.SoaLimit, res: output.Result, s: *const Session, job: Query, kind: requests.Kind) !void {
        const w = res.varnames.len;
        if (res.data.len != res.npoints * w) {
            tally.skipped += 1;
            return;
        }
        var cols: [2]?usize = .{ null, null };
        for (&cols, [2][]const u8{ lim.pos, lim.neg }) |*col, name| if (name.len != 0) {
            col.* = for (res.varnames, 0..) |v, k| {
                if (std.mem.eql(u8, v, name)) break k;
            } else {
                tally.skipped += 1;
                return;
            };
        };
        var from: ?f64 = null;
        var peak: f64 = 0;
        for (0..res.npoints) |i| {
            const row = res.data[i * w ..][0..w];
            const d = (if (cols[0]) |c| row[c] else 0) - (if (cols[1]) |c| row[c] else 0);
            const v = if (lim.abs) @abs(d) else d;
            const at = if (kind == .op) 0 else row[0];
            if (v > lim.limit) {
                peak = if (from == null) v else @max(peak, v);
                if (from == null) from = at;
            } else if (from) |f| {
                try tally.excursion(lim, peak, f, at, s, job, kind);
                from = null;
            }
        }
        if (from) |f| try tally.excursion(lim, peak, f, null, s, job, kind);
    }

    /// Counts one excursion, `from` to `to` (null: past the last row) at
    /// `peak`, prints it while `max_warns` allows and writes it to `json`.
    fn excursion(tally: *Tally, lim: core.SoaLimit, peak: f64, from: f64, to: ?f64, s: *const Session, job: Query, kind: requests.Kind) !void {
        tally.excursions += 1;
        try tally.instances.put(tally.gpa, lim.name, {});
        const row = points.variantOf(job);
        if (tally.json) |j| {
            const nan = std.math.nan(f64);
            const windowed = kind != .op;
            try j.writeAll(",\n    {\"instance\": ");
            try std.json.Stringify.encodeJsonString(lim.name, .{}, j);
            try j.print(", \"quantity\": \"{s}\", \"key\": \"{s}\", \"limit\": ", .{ lim.quantity, lim.key });
            try output.writeJsonNumber(j, lim.limit);
            try j.writeAll(", \"peak\": ");
            try output.writeJsonNumber(j, peak);
            try j.writeAll(", \"from\": ");
            try output.writeJsonNumber(j, if (windowed) from else nan);
            try j.writeAll(", \"to\": ");
            try output.writeJsonNumber(j, if (windowed) (to orelse nan) else nan);
            try j.print(", \"analysis\": \"{t}\", \"variant\": ", .{kind});
            try output.writeVariantJson(j, s.deck.variants, row, points.tempOf(job));
            try j.writeAll("}");
        }
        const out = tally.out orelse return;
        if (tally.excursions > tally.max_warns) return;
        const label = if (row) |v| s.deck.variants.labelAt(v) else "";
        try out.print("soa: {s} {s} = {d:.4} V > {s} {d} V", .{ lim.name, lim.quantity, peak, lim.key, lim.limit });
        if (kind != .op) {
            try out.print(" from {e:.3}", .{from});
            if (to) |t| try out.print(" to {e:.3}", .{t}) else try out.writeAll(" to the end");
            if (kind == .tran) try out.writeAll(" s");
        }
        try out.print(" ({t}{s}{s})\n", .{ kind, if (label.len != 0) " " else "", label });
    }
};

/// Holds every accepted point of the Problem's completed results to the
/// deck's SOA limits and prints the excursions and their counts
/// (`Problem.print_soa` documents the contract).
pub fn printSoa(self: *const Problem, out: *std.Io.Writer) !void {
    const soa = self.prepared.deck.soa;
    if (soa.limits.len == 0) return;
    var tally: Tally = .{ .gpa = self.gpa, .out = out, .max_warns = soa.max_warns };
    defer tally.instances.deinit(self.gpa);
    try tally.session(&self.session);
    for (self.runs) |*run| try tally.session(&run.session);
    assert(tally.instances.count() <= tally.excursions);
    if (tally.excursions != 0) try out.print("soa: {d} excursions on {d} instances\n", .{ tally.excursions, tally.instances.count() });
    if (tally.skipped != 0) try out.print("soa: {d} limit checks skipped: a terminal is not saved\n", .{tally.skipped});
}

/// Whether `.option soa` reads a `kind` result: only these publish the
/// accepted points.
pub fn checked(kind: requests.Kind) bool {
    return kind == .op or kind == .dc or kind == .tran;
}
