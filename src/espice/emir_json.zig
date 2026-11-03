//! Completed results' terminal currents in, `--emir`'s `espice-emir/1`
//! documents out (#237): one per variant of the main session and of each
//! variant run, in output order, each instance named by its card and
//! `.model`, each terminal by its port and net.
//! Cites: none.
//!
//! Contents:
//! - `pub fn writeEmirJson(self: *const Problem, out: *std.Io.Writer, deck: []const u8, k: usize) !bool`

const std = @import("std");
const assert = std.debug.assert;
const analysis = @import("analysis");
const core = @import("core");
const frontend = @import("frontend");
const output = @import("output");
const points = @import("points.zig");
const Problem = @import("Problem.zig");
const requests = core.query;
const QueryId = requests.QueryId;
const Session = analysis.session.Session;

/// Writes document `k` and returns true; false, writing nothing, past the
/// last one (`Problem.write_emir_json` documents the contract).
pub fn writeEmirJson(self: *const Problem, out: *std.Io.Writer, deck: []const u8, k: usize) !bool {
    var n: usize = 0;
    for (0..1 + self.runs.len) |s| {
        const session = if (s == 0) &self.session else &self.runs[s - 1].session;
        const outputs = session.outputs.items;
        for (outputs, 0..) |id, j| {
            if (!try emirFirst(session, outputs[0..j], id)) continue;
            if (n == k) {
                try writeEmir(out, deck, &self.library, if (s == 0) &self.prepared else &self.prepared.runs[s - 1], session, id);
                return true;
            }
            n += 1;
        }
    }
    assert(n <= k);
    return false;
}

// ---- private ----

/// Whether `id` is a completed query with terminal currents and the first
/// such of its variant after those in `before`.
// ponytail: quadratic in the outputs; a key set once sweeps reach thousands of points.
fn emirFirst(session: *const Session, before: []const QueryId, id: QueryId) !bool {
    if (!try hasEmir(session, id)) return false;
    const job = try session.query(id);
    for (before) |b| if (try hasEmir(session, b)) {
        const other = try session.query(b);
        if (std.meta.eql(points.variantOf(other), points.variantOf(job)) and std.meta.eql(points.tempOf(other), points.tempOf(job))) return false;
    };
    return true;
}

/// Whether `id` is a completed, kept transient, `.op` or `.dc` that read
/// terminal currents.
fn hasEmir(session: *const Session, id: QueryId) !bool {
    const info = try session.info(id);
    const kinds = info.kind == .tran or info.kind == .op or info.kind == .dc;
    if (!kinds or info.status != .complete or session.released(id)) return false;
    return (try session.result(id)).emir.terminals.len != 0;
}

/// `writeEmirJson`'s document for query `id` of `session`, which runs
/// `prepared`.
fn writeEmir(out: *std.Io.Writer, deck: []const u8, lib: *const frontend.Library, prepared: *const frontend.Prepared, session: *const Session, id: QueryId) !void {
    const e = (try session.result(id)).emir;
    const job = try session.query(id);
    const d = &prepared.deck;
    const ckt = &prepared.circuit;
    try out.writeAll("{\n  \"format\": \"espice-emir/1\",\n  \"deck\": ");
    try std.json.Stringify.encodeJsonString(deck, .{}, out);
    try out.print(",\n  \"analysis\": \"{t}\",\n  \"variant\": ", .{@as(requests.Kind, job)});
    try output.writeVariantJson(out, d.variants, points.variantOf(job), points.tempOf(job));
    try out.writeAll(",\n  \"temp\": ");
    if (points.tempOf(job) orelse d.deck_temp) |temp| try output.writeJsonNumber(out, temp) else try out.writeAll("null");
    // A static point has no window.
    if (job == .tran) {
        try out.writeAll(",\n  \"window\": [");
        try output.writeJsonNumber(out, e.from);
        try out.writeAll(", ");
        try output.writeJsonNumber(out, e.to);
        try out.writeAll("]");
    } else try out.writeAll(",\n  \"window\": null");
    try out.writeAll(",\n  \"em_recovery\": ");
    try output.writeJsonNumber(out, d.em_recovery);
    // `Wave.emAvg`'s value from the same halves and span; one sample's |i|.
    const span = e.to - e.from;
    inline for (.{ false, true }) |sources| {
        try out.writeAll(if (sources) "\n  ],\n  \"sources\": [" else ",\n  \"terminals\": [");
        var first = true;
        for (e.terminals) |term| {
            const b = ckt.batches[term.batch];
            const source = std.mem.eql(u8, b.type_name, "vsource") or std.mem.eql(u8, b.type_name, "isource");
            if (source != sources) continue;
            try out.writeAll(if (first) "\n    {\"instance\": " else ",\n    {\"instance\": ");
            first = false;
            const t = ckt.batch_types[term.batch];
            try std.json.Stringify.encodeJsonString(d.card_names.get(t, term.instance) orelse b.type_name, .{}, out);
            if (!sources) {
                try out.writeAll(", \"model\": ");
                if (d.card_models.get(t, term.instance)) |m| try std.json.Stringify.encodeJsonString(m, .{}, out) else try out.writeAll("null");
            }
            try out.writeAll(", \"terminal\": ");
            const ports = lib.vtable(t).unknown_names;
            if (term.port < ports.len) try std.json.Stringify.encodeJsonString(ports[term.port], .{}, out) else try out.print("\"{d}\"", .{term.port});
            try out.writeAll(", \"net\": ");
            try std.json.Stringify.encodeJsonString(if (term.row == core.GROUND or term.row >= ckt.n) "0" else ckt.nodeName(term.row), .{}, out);
            const em_avg = if (span == 0) @abs(term.avg) else (@max(term.pos, term.neg) - d.em_recovery * @min(term.pos, term.neg)) / span;
            for ([_][]const u8{ "avg", "em_avg", "rms", "peak" }, [_]f64{ term.avg, em_avg, term.rms, term.peak }) |key, v| {
                try out.print(", \"{s}\": ", .{key});
                try output.writeJsonNumber(out, v);
            }
            try out.writeAll("}");
        }
    }
    try out.writeAll("\n  ]\n}\n");
}
