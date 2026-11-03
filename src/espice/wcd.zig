//! A deck's `.wcd` cards in, worst-case distances out (#240): per base
//! point and spec, FORM (`core.form`) over the deck's statistical
//! parameters, each batch of points run as one session of the spec's
//! nominal queries, then the report `print_measures` and
//! `write_measures_json` write after the Monte Carlo statistics.
//! Cites: none.
//!
//! Contents:
//! - `pub const Run = struct { form: core.form.Form, sims: u32 }`
//! - `pub fn worstCase(self: *Problem) !void`
//! - `pub fn writeWorstCase(gpa: std.mem.Allocator, out: *std.Io.Writer, json: bool, w: *const frontend.WorstCase, runs: []const Run, measures: []const core.Measure) !void`

const std = @import("std");
const assert = std.debug.assert;
const core = @import("core");
const frontend = @import("frontend");
const output = @import("output");
const optimizer = @import("optimizer.zig");
const Problem = @import("Problem.zig");
const requests = core.query;
const Query = requests.Query;

/// One `.wcd` search and the real simulations it ran (a tangent row is
/// none).
pub const Run = struct { form: core.form.Form, sims: u32 };

/// `.wcd` (#240), once before any query advances: per base point and spec,
/// FORM over the deck's statistical parameters. Each batch of points is one
/// session of the nominal queries of the spec's analysis, a variant row per
/// point (`WorstCase.rows`). A gradient round of DC or transient queries
/// runs only its point and takes the neighbours from tangent rows;
/// otherwise each point is a simulation, `max_parallel` at a time. Every
/// operating point starts cold, so the result is the same at any
/// `max_parallel`. `UnsupportedCard` for a spec whose analysis runs no
/// nominal query (a `SWEEP DATA` card).
// ponytail: each round re-simulates the line point it starts from, and
// every spec simulates its own z = 0 round; share them when sims count.
pub fn worstCase(self: *Problem) !void {
    const w = if (self.prepared.wcd) |*wcd| wcd else return;
    if (self.worst_case.len != 0) return;
    const arena = self.arena.allocator();
    const gpa = self.gpa;
    const deck = &self.prepared.deck;
    const specs = w.spec.specs;
    const d = w.keys.len;
    const runs = try arena.alloc(Run, w.bases.len * specs.len);
    var template: std.ArrayList(Query) = .empty;
    defer template.deinit(gpa);
    var values: std.ArrayList(f64) = .empty;
    defer values.deinit(gpa);
    var margins: std.ArrayList(f64) = .empty;
    defer margins.deinit(gpa);
    for (runs, 0..) |*run, k| {
        const base = k / specs.len;
        const spec = specs[k % specs.len];
        const kind = deck.measures[spec.measure].analysis;
        const variant: ?u32 = if (w.stepped) @intCast(base) else null;
        template.clearRetainingCapacity();
        for (deck.queries) |job| switch (job) {
            inline else => |o| {
                const nominal = if (variant) |v| o.tol.variant == v else o.tol.variant == null;
                if (nominal and @as(requests.Kind, job) == kind) try template.append(gpa, job);
            },
        };
        if (template.items.len == 0) {
            if (!@import("builtin").is_test) std.log.err(".wcd: {s}: no nominal {t} analysis runs the card", .{ deck.measures[spec.measure].name, kind });
            return error.UnsupportedCard;
        }
        const direct = !self.session.config.knobs.no_tangents and for (template.items) |job| {
            if (!optimizer.tangentable(self, job)) break false;
        } else true;
        run.* = .{ .form = try core.form.Form.init(arena, @intCast(d), .{ .maxiter = w.spec.maxiter, .tol = w.spec.tol }), .sims = 0 };
        const form = &run.form;
        while (form.points().len != 0) {
            const points = form.points();
            const count = points.len / d;
            const batched = direct and form.batched();
            var scratch = std.heap.ArenaAllocator.init(gpa);
            defer scratch.deinit();
            try values.resize(gpa, count * deck.measures.len);
            try optimizer.evaluateRows(self, try w.rows(scratch.allocator(), &self.prepared.circuit, points, base), template.items, batched, &.{}, values.items, null);
            const sims: u32 = if (batched) 1 else @intCast(count);
            run.sims += sims;
            try margins.resize(gpa, count);
            for (margins.items, 0..) |*g, p| g.* = spec.margin(values.items[p * deck.measures.len + spec.measure]);
            form.feed(margins.items);
        }
    }
    self.worst_case = runs;
    assert(self.worst_case.len == w.bases.len * specs.len);
}

/// The `.wcd` report (#240), once `worstCase` has run: per base point, a
/// line per spec with its bound, β_w, the partial yield Φ(β_w), the gradient
/// rounds, the real simulations and the `TOP` coordinates of the worst-case
/// point by size, in sigmas. A search that stopped unconverged says why;
/// its β is the last simulated point's, never a prediction, and a margin no
/// coordinate moves has none. As JSON, one object per spec and base point
/// led by `,\n`: `"kind": "wcd"`, `spec`, `bound`, `corner` (the `.step`
/// label, "" without), `beta` and `yield` (null without β), `converged`,
/// `status`, `iterations`, `sims`, and `point`, every coordinate as
/// `{"name", "z"}` in coordinate order (names can repeat).
pub fn writeWorstCase(gpa: std.mem.Allocator, out: *std.Io.Writer, json: bool, w: *const frontend.WorstCase, runs: []const Run, measures: []const core.Measure) !void {
    const specs = w.spec.specs;
    const order = try gpa.alloc(u32, w.keys.len);
    defer gpa.free(order);
    for (runs, 0..) |*run, k| {
        const base = w.bases[k / specs.len];
        const spec = specs[k % specs.len];
        const f = &run.form;
        const name = measures[spec.measure].name;
        const beta: ?f64 = if (f.status == .no_gradient) null else f.beta;
        const partial = if (beta) |b| core.numerics.normalCdf(b) else std.math.nan(f64);
        if (json) {
            try out.writeAll(",\n    {\"kind\": \"wcd\", \"spec\": ");
            try std.json.Stringify.encodeJsonString(name, .{}, out);
            try out.writeAll(", \"bound\": ");
            try std.json.Stringify.encodeJsonString(spec.bound, .{}, out);
            try out.writeAll(", \"corner\": ");
            try std.json.Stringify.encodeJsonString(base.label, .{}, out);
            try out.writeAll(", \"beta\": ");
            try output.writeJsonNumber(out, beta orelse std.math.nan(f64));
            try out.writeAll(", \"yield\": ");
            try output.writeJsonNumber(out, partial);
            try out.print(", \"converged\": {s}, \"status\": \"{t}\", \"iterations\": {d}, \"sims\": {d}, \"point\": [", .{ if (f.converged()) "true" else "false", f.status, f.iterations, run.sims });
            for (w.labels, f.z, 0..) |label, z, i| {
                try out.writeAll(if (i == 0) "{\"name\": " else ", {\"name\": ");
                try std.json.Stringify.encodeJsonString(label, .{}, out);
                try out.writeAll(", \"z\": ");
                try output.writeJsonNumber(out, z);
                try out.writeAll("}");
            }
            try out.writeAll("]}");
            continue;
        }
        // By |z| down, ties in coordinate order.
        for (order, 0..) |*o, i| o.* = @intCast(i);
        std.mem.sort(u32, order, @as([]const f64, f.z), struct {
            fn greater(z: []const f64, i: u32, j: u32) bool {
                return @abs(z[i]) > @abs(z[j]) or (@abs(z[i]) == @abs(z[j]) and i < j);
            }
        }.greater);
        const top = order[0..@min(w.spec.top, order.len)];
        if (k % specs.len == 0) try out.print("\n  Worst-case distances (.wcd){s}{s}\n  spec            bound        beta_w      yield(Phi)    iters  sims   worst-case point (top {d})\n", .{ if (base.label.len != 0) " at " else "", base.label, top.len });
        try out.print("  {s:<15} {s:<12} ", .{ name, spec.bound });
        if (beta) |b| try out.print("{d:<11.4} {d:<13.10} ", .{ b, partial }) else try out.writeAll("-           -             ");
        try out.print("{d:<6} {d:<6}", .{ f.iterations, run.sims });
        for (top, 0..) |i, n| try out.print("{s}{s} {s}{d:.2}s", .{ if (n == 0) " " else ", ", w.labels[i], if (f.z[i] >= 0) "+" else "", f.z[i] });
        try out.writeAll(switch (f.status) {
            .converged => "\n",
            .maxiter => "  (not converged: MAXITER reached)\n",
            .line_search => "  (not converged: no step along the linearized boundary lowered the merit)\n",
            .no_gradient => "  (no statistical parameter moves this measure)\n",
            .failed => "  (not converged: a point could not be simulated or measured)\n",
            .running => unreachable, // `worstCase` runs every search to its end
        });
    }
}
