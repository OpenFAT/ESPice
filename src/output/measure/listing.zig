//! `.meas` listings: a deck's cards and one result (or many Monte Carlo
//! trials, or a scaled-sigma run's counts) in, ngspice's `Measurements for
//! ... Analysis` block, or the trial statistics, out, and each also as JSON
//! objects (`json.zig`, #235). A card that names a missing vector or whose
//! event never happens writes ngspice's error line to the error writer
//! instead.
//! Cites: ngspice 45 com_measure2.c and measure.c (headings, error lines,
//! PARAM cards evaluated after the others).
//!
//! Contents:
//! - `pub const Point = struct { row: u32, label: []const u8 }`
//! - `pub const Sss = struct`
//! - `pub fn print(out: *Writer, err: *Writer, measures: []const core.Measure, analysis: Kind, result: core.Result, point: ?Point, json: ?Json) Writer.Error!void`
//! - `pub fn printFailed(out: *Writer, measures: []const core.Measure, analysis: Kind, point: Point, json: ?Json) Writer.Error!void`
//! - `pub fn printStatistics(out: *Writer, measures: []const core.Measure, analysis: Kind, trials: []const core.Result) Writer.Error!void`
//! - `pub fn writeStatistics(out: *Writer, measures: []const core.Measure, analysis: Kind, trials: []const core.Result) Writer.Error!void`
//! - `pub fn printSss(out: *Writer, s: Sss) Writer.Error!void`
//! - `pub fn writeSss(out: *Writer, analysis: Kind, s: Sss) Writer.Error!void`

const std = @import("std");
const assert = std.debug.assert;
const Writer = std.Io.Writer;
const core = @import("core");
const card = @import("card.zig");
const Wave = @import("Wave.zig");
const sci = @import("sci.zig").sci;
const quiet = @import("sci.zig").quiet;
const results_json = @import("json.zig");
const Json = results_json.Json;
const writeResult = results_json.writeResult;
const writeNumber = results_json.writeNumber;
const Clause = core.MeasureClause;
const Kind = core.query.Kind;

/// A sweep point's row in the variant table and its label (`rv=1k, ...`),
/// which names its `.meas` block.
pub const Point = struct { row: u32, label: []const u8 };

/// A SIGMA_SCALE run's counts and fit (#239): `failures[j]` of `trials`
/// failed at `scales[j]`.
pub const Sss = struct {
    /// FAIL as written.
    fail: []const u8,
    scales: []const f64,
    trials: u64,
    failures: []const u64,
    estimate: core.sss.Error!core.sss.Estimate,
};

/// Cards `print` keeps results for, and `printStatistics` reports; a PARAM
/// card reads NaN for any card past it.
const cards_max = 256;
/// Bytes of listing text `print`'s first pass keeps; a card past them is
/// evaluated again in the second.
const kept_bytes_max = 16 * 1024;

/// What `print`'s first pass kept of a card: nothing (`fresh`, evaluate it
/// again), or its text and how it ended.
const Kept = enum(u8) { fresh, value, no_such_vector, out_of_interval };

/// A card's Welford accumulator over the trials; every field moves together.
const Acc = struct {
    n: f64 = 0,
    mean: f64 = 0,
    m2: f64 = 0,
    lo: f64 = std.math.inf(f64),
    hi: f64 = -std.math.inf(f64),

    /// With Bessel's correction; 0 for one sample.
    fn sigma(a: Acc) f64 {
        return if (a.n > 1) @sqrt(a.m2 / (a.n - 1)) else 0;
    }
};

const nan = std.math.nan(f64);

/// v(a) = t and v(b) = 4 - t on t = 0..4, sampled every 1 (test fixture).
const ramp: core.Result = .{ .plotname = "t", .varnames = &.{ "time", "v(a)", "v(b)" }, .is_complex = false, .npoints = 5, .data = &.{ 0, 0, 4, 1, 1, 3, 2, 2, 2, 3, 3, 1, 4, 4, 0 } };

comptime {
    assert(kept_bytes_max <= std.math.maxInt(u16)); // `kept_len` holds any kept text.
}

/// Writes every card in `measures` that targets `analysis`, evaluated over
/// `result`, under ngspice's heading. Writes nothing when none targets it.
/// A card that names a missing vector or whose event never happens writes
/// ngspice's error line to `err` and no value; at a sweep `point`, which
/// the heading names, it also writes `name = NaN`. A PARAM card reads NaN
/// for any card past the 256th. With `json`, each line is also a JSON
/// result there, in the same order, and a failed card a `null` one with
/// its error.
pub fn print(out: *Writer, err: *Writer, measures: []const core.Measure, analysis: Kind, result: core.Result, point: ?Point, json: ?Json) Writer.Error!void {
    // As ngspice's measure.c, PARAM cards read every other card's result,
    // so those are evaluated first; each card still prints in deck order.
    var values: [cards_max]f64 = @splat(nan);
    var done: Wave.Done = undefined;
    // The first pass keeps each card's printed text (and error), so the
    // second prints it instead of evaluating the card again: a sweep prints
    // a block per point, and each number goes through `%e`'s exact decimal
    // conversion. The kept texts lie back to back in deck order, and so do
    // their JSON results; a card that overflows either is evaluated again.
    var text: [kept_bytes_max]u8 = undefined;
    var used: usize = 0;
    var json_text: [kept_bytes_max]u8 = undefined;
    var json_used: usize = 0;
    var kept: [cards_max]Kept = @splat(.fresh);
    var kept_len: [cards_max]u16 = @splat(0);
    var kept_json_len: [cards_max]u16 = @splat(0);
    for (measures, 0..) |m, i| {
        if (!card.targets(m, analysis, result) or m.func == .param or i >= values.len) continue;
        var w: Writer = .fixed(text[used..]);
        var jw: Writer = .fixed(json_text[json_used..]);
        var wave = card.cardWave(result, analysis, &values, i, &done);
        if (json) |j| wave.json = .{ .out = &jw, .head = j.head };
        const r = card.evaluate(&w, m, wave);
        if (r == error.WriteFailed) {
            // Out of buffer: the value as before, through a discarding writer.
            var discard_buf: [64]u8 = undefined;
            var discard = quiet(&discard_buf);
            values[i] = card.evaluate(&discard.writer, m, card.cardWave(result, analysis, &values, i, &done)) catch nan;
            continue;
        }
        values[i] = r catch nan;
        kept[i] = if (r) |_| .value else |e| switch (e) {
            error.NoSuchVector => .no_such_vector,
            error.OutOfInterval => .out_of_interval,
            error.WriteFailed => unreachable, // handled above
        };
        kept_len[i] = @intCast(w.end);
        kept_json_len[i] = @intCast(jw.end);
        used += w.end;
        json_used += jw.end;
    }
    assert(used <= text.len);
    assert(json_used <= json_text.len);
    var headed = false;
    var replayed: usize = 0;
    var json_replayed: usize = 0;
    for (measures, 0..) |m, i| {
        if (!card.targets(m, analysis, result)) continue;
        if (!headed) try writeHeading(out, analysis, point);
        headed = true;
        const status: Kept = if (i < kept.len) kept[i] else .fresh;
        const outcome: card.Error!f64 = switch (status) {
            .fresh => blk: {
                var wave = card.cardWave(result, analysis, &values, i, &done);
                wave.json = json;
                break :blk card.evaluate(out, m, wave);
            },
            .value, .no_such_vector, .out_of_interval => blk: {
                try out.writeAll(text[replayed..][0..kept_len[i]]);
                replayed += kept_len[i];
                if (json) |j| try j.out.writeAll(json_text[json_replayed..][0..kept_json_len[i]]);
                json_replayed += kept_json_len[i];
                break :blk switch (status) {
                    .value => values[i],
                    .no_such_vector => error.NoSuchVector,
                    .out_of_interval => error.OutOfInterval,
                    .fresh => unreachable,
                };
            },
        };
        const value = outcome catch |e| switch (e) {
            error.WriteFailed => return error.WriteFailed,
            error.NoSuchVector => blk: {
                try err.print("\nError: measure  {s} : no such vector\n", .{m.name});
                if (json) |j| try writeResult(j, m.name, nan, .{}, "no such vector");
                break :blk nan;
            },
            error.OutOfInterval => blk: {
                try err.print("\nError: measure  {s} : out of interval\n", .{m.name});
                if (json) |j| try writeResult(j, m.name, nan, .{}, "out of interval");
                break :blk nan;
            },
        };
        if (point != null and std.math.isNan(value)) try out.print("{s:<20}=  NaN\n", .{m.name});
        if (m.func == .param and i < values.len) values[i] = value;
    }
    assert(replayed == used);
    assert(json_replayed == json_used);
}

/// `print` for a sweep point whose `analysis` failed: its heading and
/// `name = NaN` for every card of that analysis, and with `json` a `null`
/// result for each.
pub fn printFailed(out: *Writer, measures: []const core.Measure, analysis: Kind, point: Point, json: ?Json) Writer.Error!void {
    var headed = false;
    for (measures) |m| {
        if (m.analysis != analysis) continue;
        if (!headed) try writeHeading(out, analysis, point);
        headed = true;
        try out.print("{s:<20}=  NaN\n", .{m.name});
        if (json) |j| try writeResult(j, m.name, nan, .{}, "analysis failed");
    }
}

/// Writes mean, sigma, min and max of each card over Monte Carlo trials,
/// as `mean(name) = x` lines (sigma with Bessel's correction). `trials`
/// holds the results the cards are evaluated over; a card that fails in a
/// trial leaves that trial out. Cards past the 256th are not reported.
/// O(cards · trials · samples).
pub fn printStatistics(out: *Writer, measures: []const core.Measure, analysis: Kind, trials: []const core.Result) Writer.Error!void {
    var acc: [cards_max]Acc = @splat(.{});
    const cards = statistics(measures, analysis, trials, &acc);
    var heading = false;
    for (cards, acc[0..cards.len]) |m, a| {
        if (a.n == 0) continue;
        if (!heading) {
            heading = true;
            try out.print("\n  Monte Carlo statistics over {d} trials\n\n", .{trials.len});
        }
        try out.print("mean({s}) = {f}\nsigma({s}) = {f}\nmin({s}) = {f}\nmax({s}) = {f}\n", .{ m.name, sci(a.mean, 6), m.name, sci(a.sigma(), 6), m.name, sci(a.lo, 6), m.name, sci(a.hi, 6) });
    }
}

/// `printStatistics` as JSON objects, each led by `,\n` (#235): the
/// card, `analysis`, `"kind": "monte"`, the trial count `n`, the trials it
/// `failed` in, and mean, sigma, min and max (null when every trial
/// failed). Every card of `analysis` is written.
pub fn writeStatistics(out: *Writer, measures: []const core.Measure, analysis: Kind, trials: []const core.Result) Writer.Error!void {
    var acc: [cards_max]Acc = @splat(.{});
    const cards = statistics(measures, analysis, trials, &acc);
    for (cards, acc[0..cards.len]) |m, a| {
        if (m.analysis != analysis) continue;
        try out.writeAll(",\n    {\"name\": ");
        try std.json.Stringify.encodeJsonString(m.name, .{}, out);
        try out.print(", \"analysis\": \"{t}\", \"kind\": \"monte\", \"n\": {d}, \"failed\": {d}", .{ analysis, trials.len, trials.len - @as(usize, @intFromFloat(a.n)) });
        const names = [_][]const u8{ "mean", "sigma", "min", "max" };
        for (names, [_]f64{ a.mean, a.sigma(), a.lo, a.hi }) |field, x| {
            try out.print(", \"{s}\": ", .{field});
            try writeNumber(out, if (a.n != 0) x else nan);
        }
        try out.writeAll("}");
    }
}

/// The scaled-sigma table, then P_f(1) with its 95 % interval and its
/// equivalent sigma, or the refusal naming the scales the fit could not
/// use (none or all of their trials failed).
pub fn printSss(out: *Writer, s: Sss) Writer.Error!void {
    try out.print("\n  Scaled-sigma sampling: FAIL = {s}\n  scale   trials  failures\n", .{s.fail});
    for (s.scales, s.failures) |scale, k| try out.print("  {d:<6}  {d:<6}  {d}\n", .{ scale, s.trials, k });
    const e = s.estimate catch |err| {
        if (err == error.Singular) return out.writeAll("  refused: the scales cannot tell the three fit parameters apart\n");
        try out.writeAll("  refused: fewer than three scales with some, but not all, trials failing; left out:");
        for (s.scales, s.failures) |scale, k| if (k == 0 or k == s.trials) try out.print(" {d} ({d} failures)", .{ scale, k });
        return out.writeAll("\n");
    };
    const iv = e.interval();
    try out.print("  P_f(1) = {f}  [{f}, {f}]  ({d:.2} sigma)\n", .{ sci(e.p(), 1), sci(iv[0], 1), sci(iv[1], 1), core.sss.sigmaOf(e.p()) });
}

/// `printSss` as one JSON object led by `,\n` (#235): `"kind": "sss"`,
/// `analysis`, `fail`, `trials`, `scales`, `failures`, then `p`, `lo`,
/// `hi` and `sigma`, null with an `error` when the fit is refused.
pub fn writeSss(out: *Writer, analysis: Kind, s: Sss) Writer.Error!void {
    try out.print(",\n    {{\"kind\": \"sss\", \"analysis\": \"{t}\", \"fail\": ", .{analysis});
    try std.json.Stringify.encodeJsonString(s.fail, .{}, out);
    try out.print(", \"trials\": {d}, \"scales\": [", .{s.trials});
    for (s.scales, 0..) |x, j| {
        if (j != 0) try out.writeAll(", ");
        try writeNumber(out, x);
    }
    try out.writeAll("], \"failures\": [");
    for (s.failures, 0..) |k, j| try out.print("{s}{d}", .{ if (j == 0) "" else ", ", k });
    try out.writeAll("]");
    const e = s.estimate catch |err| return out.print(", \"p\": null, \"lo\": null, \"hi\": null, \"sigma\": null, \"error\": \"{t}\"}}", .{err});
    const iv = e.interval();
    for ([_][]const u8{ "p", "lo", "hi", "sigma" }, [_]f64{ e.p(), iv[0], iv[1], core.sss.sigmaOf(e.p()) }) |field, x| {
        try out.print(", \"{s}\": ", .{field});
        try writeNumber(out, x);
    }
    try out.writeAll("}");
}

// ---- private ----

/// ngspice's block heading, naming the sweep point when there is one.
fn writeHeading(out: *Writer, analysis: Kind, point: ?Point) Writer.Error!void {
    try out.print("\n  Measurements for {s} Analysis", .{switch (analysis) {
        .tran => "Transient",
        .tran_noise => "Transient Noise",
        .ac => "AC",
        .sp => "SP",
        .fft => "FFT",
        .dcmatch => "DC Mismatch",
        .acmatch => "AC Mismatch",
        .lstb => "Loop Stability",
        .phasenoise => "Phase Noise",
        .pnoise => "Periodic Time-Dependent Noise",
        .op => "Operating Point",
        else => "DC", // else: any other analysis takes the DC heading; a kind with its own title adds an arm
    }});
    if (point) |p| try out.print(", point {d} ({s})", .{ p.row, p.label });
    try out.writeAll("\n\n");
}

/// Accumulates each card of `analysis` over `trials` into `acc`; returns
/// the cards `acc` covers, the first 256. O(cards · trials · samples).
fn statistics(measures: []const core.Measure, analysis: Kind, trials: []const core.Result, acc: *[cards_max]Acc) []const core.Measure {
    var discard_buf: [64]u8 = undefined;
    var discard = quiet(&discard_buf);
    const cards = measures[0..@min(measures.len, acc.len)];
    for (trials) |result| {
        // One pass in deck order: a card reads only the cards before it (a
        // later one is still NaN), so each card sees what it would alone.
        var values: [cards_max]f64 = @splat(nan);
        var done: Wave.Done = undefined;
        for (cards, 0..) |m, k| {
            if (m.analysis != analysis) continue;
            values[k] = card.evaluate(&discard.writer, m, card.cardWave(result, analysis, &values, k, &done)) catch nan;
            const x = values[k];
            if (!std.math.isFinite(x)) continue;
            const a = &acc[k];
            a.n += 1;
            const delta = x - a.mean;
            a.mean += delta / a.n;
            a.m2 += delta * (x - a.mean);
            a.lo = @min(a.lo, x);
            a.hi = @max(a.hi, x);
        }
    }
    return cards;
}

/// A parsed JSON number as an f64; NaN for anything else (test helper).
fn jsonFloat(v: std.json.Value) f64 {
    return switch (v) {
        .integer => |i| @floatFromInt(i),
        .float => |f| f,
        else => nan, // else: null and the non-numbers have no value
    };
}

test "measurements match ngspice on a sampled ramp" {
    // v(a) = t on t = 0..4, sampled every 1; v(b) falls 4..0.
    const data = [_]f64{ 0, 0, 4, 1, 1, 3, 2, 2, 2, 3, 3, 1, 4, 4, 0 };
    const result: core.Result = .{ .plotname = "t", .varnames = &.{ "time", "v(a)", "v(b)" }, .is_complex = false, .npoints = 5, .data = &data };
    const cards = [_]core.Measure{
        .{ .analysis = .tran, .name = "d", .func = .trig_targ, .first = .{ .vec = "v(a)", .val = 0.5, .rise = 1 }, .second = .{ .vec = "b", .val = 1.5, .fall = 1 } },
        .{ .analysis = .tran, .name = "f", .func = .find, .first = .{ .vec = "v(b)", .at = 2.5 } },
        .{ .analysis = .tran, .name = "i", .func = .integ, .first = .{ .vec = "v(a)" } },
        .{ .analysis = .tran, .name = "mx", .func = .max, .first = .{ .vec = "v(a)", .to = 3 } },
        .{ .analysis = .tran, .name = "gone", .func = .when, .first = .{ .vec = "v(a)", .val = 9 } },
        .{ .analysis = .ac, .name = "skipped", .func = .max, .first = .{ .vec = "v(a)" } },
        // HSPICE forms: DERIVATIVE, ERR2 (t = 0 falls under YMIN), PARAM.
        .{ .analysis = .tran, .name = "s", .func = .deriv, .first = .{ .vec = "v(b)", .at = 2.5 } },
        .{ .analysis = .tran, .name = "e", .func = .err2, .first = .{ .vec = "v(a)", .vec2 = "v(b)" } },
        .{ .analysis = .tran, .name = "p", .func = .param, .first = .{}, .expr = &.{ .{ .measure = 0 }, .{ .num = 2 }, .mul } },
    };
    var out_buf: [1024]u8 = undefined;
    var err_buf: [256]u8 = undefined;
    var out: Writer = .fixed(&out_buf);
    var err: Writer = .fixed(&err_buf);
    try print(&out, &err, &cards, .tran, result, null, null);
    // Rise 1 of v(a) happens in the first interval, where ngspice only
    // initialises its counters: it is measured on the next one, by
    // extrapolating that segment (exact here, as v(a) is a line).
    try std.testing.expectEqualStrings(
        \\
        \\  Measurements for Transient Analysis
        \\
        \\d                   =  2.000000e+00 targ=  2.500000e+00 trig=  5.000000e-01
        \\f                   =  1.500000e+00
        \\i                   =   8.00000e+00 from=  0.00000e+00 to=  4.00000e+00
        \\mx                  =  3.000000e+00 at=  3.000000e+00
        \\s                   =  -1.000000e+00
        \\e                   =  9.166667e-01
        \\p                   =  4.00000e+00
        \\
    , out.buffered());
    try std.testing.expect(std.mem.indexOf(u8, err.buffered(), "gone") != null);
}

test "print's JSON results: each printed line in order, the ramp's exact values, a failed card null with its error (#235)" {
    const a = std.testing.allocator;
    const cards = [_]core.Measure{
        .{ .analysis = .tran, .name = "d", .func = .trig_targ, .first = .{ .vec = "v(a)", .val = 0.5, .rise = 1 }, .second = .{ .vec = "b", .val = 1.5, .fall = 1 } },
        .{ .analysis = .tran, .name = "i", .func = .integ, .first = .{ .vec = "v(a)" } },
        .{ .analysis = .tran, .name = "mx", .func = .max, .first = .{ .vec = "v(a)", .to = 3 } },
        .{ .analysis = .tran, .name = "c", .func = .find, .cont = true, .first = .{ .vec = "v(a)", .at = 1.5 } },
        .{ .analysis = .tran, .name = "p", .func = .param, .first = .{}, .expr = &.{ .{ .measure = 0 }, .{ .num = 2 }, .mul } },
        .{ .analysis = .tran, .name = "gone", .func = .when, .first = .{ .vec = "v(a)", .val = 9 } },
    };
    var text: Writer.Allocating = .init(a);
    defer text.deinit();
    var json: Writer.Allocating = .init(a);
    defer json.deinit();
    try print(&text.writer, &text.writer, &cards, .tran, ramp, null, .{ .out = &json.writer, .head = "\"analysis\": \"tran\"" });
    const doc = try std.fmt.allocPrint(a, "[{s}]", .{json.written()[1..]});
    defer a.free(doc);
    const parsed = try std.json.parseFromSlice(std.json.Value, a, doc, .{});
    defer parsed.deinit();
    const results = parsed.value.array.items;
    // Oracle: the text, whose `%e` digits `sci` matches C's, line for line.
    var lines = std.mem.splitScalar(u8, text.written(), '\n');
    var k: usize = 0;
    while (lines.next()) |text_line| {
        const eq = std.mem.indexOfScalar(u8, text_line, '=') orelse continue;
        if (std.mem.startsWith(u8, text_line, "Error")) continue;
        const r = results[k].object;
        k += 1;
        try std.testing.expectEqualStrings(std.mem.trim(u8, text_line[0..eq], " "), r.get("name").?.string);
        var fields = std.mem.tokenizeScalar(u8, text_line[eq + 1 ..], ' ');
        try std.testing.expectApproxEqRel(try std.fmt.parseFloat(f64, fields.next().?), jsonFloat(r.get("value").?), 1e-5);
    }
    try std.testing.expectEqual(@as(usize, 5), k);
    // Analytic oracle: v(a) = t, so the integral over 0..4 is 8 and the
    // maximum to t = 3 is 3 at 3, in full f64.
    try std.testing.expectEqual(@as(f64, 8), jsonFloat(results[1].object.get("value").?));
    try std.testing.expectEqual(@as(f64, 4), jsonFloat(results[1].object.get("to").?));
    try std.testing.expectEqual(@as(f64, 3), jsonFloat(results[2].object.get("at").?));
    try std.testing.expectEqualStrings("c[1]", results[3].object.get("name").?.string);
    const gone = results[5].object;
    try std.testing.expectEqualStrings("gone", gone.get("name").?.string);
    try std.testing.expect(gone.get("value").? == .null);
    try std.testing.expectEqualStrings("out of interval", gone.get("error").?.string);
    try std.testing.expectEqual(@as(usize, 6), results.len);
}

test "a _CONT FIND or DERIV at a fixed AT= has one event and prints one line" {
    const cards = [_]core.Measure{
        .{ .analysis = .tran, .name = "x", .func = .find, .cont = true, .first = .{ .vec = "v(a)", .at = 1.5 } },
        .{ .analysis = .tran, .name = "d", .func = .deriv, .cont = true, .first = .{ .vec = "v(b)", .at = 1.5 } },
    };
    var values: [cards.len]f64 = undefined;
    card.evaluateAll(&cards, .tran, ramp, &values);
    try std.testing.expectEqualSlices(f64, &.{ 1.5, -1 }, &values);
    var buf: [256]u8 = undefined;
    var out: Writer = .fixed(&buf);
    try print(&out, &out, &cards, .tran, ramp, null, null);
    try std.testing.expectEqualStrings(
        \\
        \\  Measurements for Transient Analysis
        \\
        \\x[1]                =  1.500000e+00
        \\d[1]                =  -1.000000e+00
        \\
    , out.buffered());
}

test "every function over a result with no samples reads NaN or zero, never panics" {
    const funcs = comptime std.enums.values(core.MeasureFunc);
    var cards: [funcs.len]core.Measure = undefined;
    for (funcs, &cards) |f, *c| c.* = .{ .analysis = .tran, .name = "m", .func = f, .first = .{ .vec = "v(a)", .vec2 = "v(b)" }, .second = .{ .vec = "v(b)" } };
    var empty = ramp;
    empty.npoints = 0;
    empty.data = &.{};
    // A streamed result keeps its point count but no samples.
    var streamed = ramp;
    streamed.data = &.{};
    var buf: [64]u8 = undefined;
    var sink: Writer.Discarding = .init(&buf);
    for ([_]core.Result{ empty, streamed }) |result| {
        var values: [funcs.len]f64 = undefined;
        card.evaluateAll(&cards, .tran, result, &values);
        for (cards, values) |c, v| switch (c.func) {
            // ngspice measure_minMaxAvg reports 0 over an empty window, and
            // a check over no samples counts no violation.
            .avg, .min, .max, .min_at, .max_at, .pp, .check_slew, .check_forbid, .check_require, .check_level, .dout => try std.testing.expectEqual(0, v),
            else => try std.testing.expect(std.math.isNan(v)), // else: every other function has no value over no samples
        };
        try print(&sink.writer, &sink.writer, &cards, .tran, result, null, null);
        var out: [3]f64 = undefined;
        try card.sample(.{ .vec = "v(a)" }, result, &.{ -1, 0, 9 }, &out);
        for (out) |v| try std.testing.expect(std.math.isNan(v));
    }
}

test printStatistics {
    var doubled = ramp;
    doubled.data = &.{ 0, 0, 0, 1, 2, 0, 2, 4, 0, 3, 6, 0, 4, 8, 0 };
    const cards = [_]core.Measure{
        .{ .analysis = .tran, .name = "mx", .func = .max, .first = .{ .vec = "a" } },
        .{ .analysis = .tran, .name = "p", .func = .param, .first = .{}, .expr = &.{ .{ .measure = 0 }, .{ .num = 2 }, .mul } },
        // A PARAM reads only the cards before it: `late` is NaN here.
        .{ .analysis = .tran, .name = "q", .func = .param, .first = .{}, .expr = &.{.{ .measure = 3 }} },
        .{ .analysis = .tran, .name = "late", .func = .max, .first = .{ .vec = "v(a)" } },
        .{ .analysis = .ac, .name = "other", .func = .max, .first = .{ .vec = "v(a)" } },
    };
    var buf: [1024]u8 = undefined;
    var out: Writer = .fixed(&buf);
    try printStatistics(&out, &cards, .tran, &.{ ramp, doubled });
    try std.testing.expectEqualStrings(
        \\
        \\  Monte Carlo statistics over 2 trials
        \\
        \\mean(mx) = 6.000000e+00
        \\sigma(mx) = 2.828427e+00
        \\min(mx) = 4.000000e+00
        \\max(mx) = 8.000000e+00
        \\mean(p) = 1.200000e+01
        \\sigma(p) = 5.656854e+00
        \\min(p) = 8.000000e+00
        \\max(p) = 1.600000e+01
        \\mean(late) = 6.000000e+00
        \\sigma(late) = 2.828427e+00
        \\min(late) = 4.000000e+00
        \\max(late) = 8.000000e+00
        \\
    , out.buffered());
}

test "crossings, windows, NaN samples and a single point match ngspice 45" {
    // Each expected listing is what ngspice 45 prints for the same cards over
    // the same samples, loaded from an ASCII raw file; a failed card prints
    // nothing to `out`.
    const tri: core.Result = .{ .plotname = "t", .varnames = &.{ "time", "v(a)" }, .is_complex = false, .npoints = 9, .data = &.{ 0, 0, 1, 2, 2, 0, 3, 2, 4, 0, 5, 2, 6, 0, 7, 2, 8, 0 } };
    const with_nan: core.Result = .{ .plotname = "t", .varnames = &.{ "time", "v(a)" }, .is_complex = false, .npoints = 5, .data = &.{ 0, 0, 1, 1, 2, nan, 3, 3, 4, 2 } };
    const one: core.Result = .{ .plotname = "t", .varnames = &.{ "time", "v(a)" }, .is_complex = false, .npoints = 1, .data = &.{ 0, 5 } };
    const last = core.measure_last;
    const a: Clause = .{ .vec = "v(a)" };
    const tri_cards = [_]core.Measure{
        // RISE=1 falls in the first interval, which only primes the counters:
        // ngspice reports the next crossing, the fall at 1.5.
        .{ .analysis = .tran, .name = "r1", .func = .when, .first = .{ .vec = "v(a)", .val = 1, .rise = 1 } },
        .{ .analysis = .tran, .name = "r2", .func = .when, .first = .{ .vec = "v(a)", .val = 1, .rise = 2 } },
        .{ .analysis = .tran, .name = "f1", .func = .when, .first = .{ .vec = "v(a)", .val = 1, .fall = 1 } },
        .{ .analysis = .tran, .name = "c1", .func = .when, .first = .{ .vec = "v(a)", .val = 1, .cross = 1 } },
        .{ .analysis = .tran, .name = "c3", .func = .when, .first = .{ .vec = "v(a)", .val = 1, .cross = 3 } },
        .{ .analysis = .tran, .name = "fl", .func = .when, .first = .{ .vec = "v(a)", .val = 1, .fall = last } },
        .{ .analysis = .tran, .name = "rl", .func = .when, .first = .{ .vec = "v(a)", .val = 1, .rise = last } },
        .{ .analysis = .tran, .name = "cl", .func = .when, .first = .{ .vec = "v(a)", .val = 1, .cross = last } },
        .{ .analysis = .tran, .name = "w0", .func = .when, .first = .{ .vec = "v(a)", .val = 1 } },
        .{ .analysis = .tran, .name = "tt", .func = .trig_targ, .first = .{ .vec = "v(a)", .val = 1, .rise = 2 }, .second = .{ .vec = "v(a)", .val = 1, .fall = 2 } },
        .{ .analysis = .tran, .name = "x", .func = .find, .first = .{ .vec = "v(a)", .at = 2.25 } },
        .{ .analysis = .tran, .name = "mx", .func = .max, .first = .{ .vec = "v(a)", .from = 0.5, .to = 3.5 } },
        .{ .analysis = .tran, .name = "av", .func = .avg, .first = .{ .vec = "v(a)", .to = 4 } },
        .{ .analysis = .tran, .name = "ig", .func = .integ, .first = .{ .vec = "v(a)", .from = 0.5, .to = 3.5 } },
        .{ .analysis = .tran, .name = "rm", .func = .rms, .first = a },
        .{ .analysis = .tran, .name = "pp", .func = .pp, .first = a },
        .{ .analysis = .tran, .name = "mn", .func = .min, .first = .{ .vec = "v(a)", .from = 1, .to = 3 } },
        .{ .analysis = .tran, .name = "wd", .func = .when, .first = .{ .vec = "v(a)", .val = 1, .td = 2.6 } },
    };
    const sparse_cards = [_]core.Measure{
        .{ .analysis = .tran, .name = "mx", .func = .max, .first = a },
        .{ .analysis = .tran, .name = "mn", .func = .min, .first = a },
        .{ .analysis = .tran, .name = "w", .func = .when, .first = .{ .vec = "v(a)", .val = 1.5 } },
        .{ .analysis = .tran, .name = "w2", .func = .when, .first = .{ .vec = "v(a)", .val = 2.5, .fall = 1 } },
        .{ .analysis = .tran, .name = "x", .func = .find, .first = .{ .vec = "v(a)", .at = 0.5 } },
        .{ .analysis = .tran, .name = "x0", .func = .find, .first = .{ .vec = "v(a)", .at = 0 } },
        .{ .analysis = .tran, .name = "av", .func = .avg, .first = a },
        .{ .analysis = .tran, .name = "ig", .func = .integ, .first = a },
        .{ .analysis = .tran, .name = "rm", .func = .rms, .first = a },
        .{ .analysis = .tran, .name = "pp", .func = .pp, .first = a },
    };
    const cases = [_]struct { core.Result, []const core.Measure, []const u8 }{
        .{
            tri, &tri_cards,
            \\r1                  =   1.50000e+00
            \\r2                  =   2.50000e+00
            \\f1                  =   1.50000e+00
            \\c3                  =   2.50000e+00
            \\fl                  =   7.50000e+00
            \\rl                  =   6.50000e+00
            \\cl                  =   7.50000e+00
            \\w0                  =   1.50000e+00
            \\tt                  =  1.000000e+00 targ=  3.500000e+00 trig=  2.500000e+00
            \\x                   =  5.000000e-01
            \\mx                  =  2.000000e+00 at=  3.000000e+00
            \\av                  =  1.000000e+00 from=  0.000000e+00 to=  5.000000e+00
            \\ig                  =   2.50000e+00 from=  5.00000e-01 to=  3.50000e+00
            \\rm                  =   1.47196e+00 from=  0.00000e+00 to=  8.00000e+00
            \\pp                  =  2.000000e+00 from=  0.000000e+00 to=  0.000000e+00
            \\mn                  =  0.000000e+00 at=  2.000000e+00
            \\wd                  =   4.50000e+00
            \\
        },
        .{
            with_nan, &sparse_cards,
            \\mx                  =  3.000000e+00 at=  3.000000e+00
            \\mn                  =  0.000000e+00 at=  0.000000e+00
            \\w2                  =   3.50000e+00
            \\x                   =  5.000000e-01
            \\x0                  =  0.000000e+00
            \\pp                  =  3.000000e+00 from=  0.000000e+00 to=  0.000000e+00
            \\
        },
        .{
            one, &sparse_cards,
            \\mx                  =  5.000000e+00 at=  0.000000e+00
            \\mn                  =  5.000000e+00 at=  0.000000e+00
            \\ig                  =   0.00000e+00 from=  0.00000e+00 to=  0.00000e+00
            \\pp                  =  0.000000e+00 from=  0.000000e+00 to=  0.000000e+00
            \\
        },
    };
    for (cases) |case| {
        var buf: [2048]u8 = undefined;
        var out: Writer = .fixed(&buf);
        var sink: Writer.Discarding = .init(&.{});
        try print(&out, &sink.writer, case[1], .tran, case[0], null, null);
        const heading = "\n  Measurements for Transient Analysis\n\n";
        try std.testing.expectEqualStrings(heading, out.buffered()[0..heading.len]);
        try std.testing.expectEqualStrings(case[2], out.buffered()[heading.len..]);
    }
}
