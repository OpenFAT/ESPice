//! `.meas` results as JSON (#235): a card's value and the window its line
//! prints in, a variant row, or one f64 in; a JSON object or number on a
//! `std.Io.Writer` out. `print` and the `--measures` writers lead each
//! object with `,\n` so a caller frames them as one array.
//! Cites: RFC 8259 (numbers; no NaN or infinity, so those write `null`).
//!
//! Contents:
//! - `pub const Json = struct { out: *Writer, head: []const u8 }`
//! - `pub const Extra = struct { from: ?f64, to: ?f64, at: ?f64 }`
//! - `pub fn writeResult(j: Json, name: []const u8, value: f64, extra: Extra, failure: ?[]const u8) Writer.Error!void`
//! - `pub fn writeNumber(out: *Writer, x: f64) Writer.Error!void`
//! - `pub fn writeVariant(out: *Writer, variants: core.Variants, row: ?u32, temp_c: ?f64) Writer.Error!void`

const std = @import("std");
const Writer = std.Io.Writer;
const core = @import("core");

/// Where `print` also writes each result line as a JSON object (#235),
/// each led by `,\n`: `head` holds the fields between its name and its
/// value, `"analysis": ..., "variant": ...`.
pub const Json = struct { out: *Writer, head: []const u8 };

/// A result line's window and position, as its text prints them.
pub const Extra = struct { from: ?f64 = null, to: ?f64 = null, at: ?f64 = null };

const nan = std.math.nan(f64);

/// One JSON result led by `,\n`: `value` (null when not finite), the
/// `extra` fields its line prints, and `failure` as `error`.
pub fn writeResult(j: Json, name: []const u8, value: f64, extra: Extra, failure: ?[]const u8) Writer.Error!void {
    try j.out.writeAll(",\n    {\"name\": ");
    try std.json.Stringify.encodeJsonString(name, .{}, j.out);
    try j.out.print(", {s}, \"value\": ", .{j.head});
    try writeNumber(j.out, value);
    inline for (.{ "from", "to", "at" }) |field| if (@field(extra, field)) |x| {
        try j.out.writeAll(", \"" ++ field ++ "\": ");
        try writeNumber(j.out, x);
    };
    if (failure) |f| {
        try j.out.writeAll(", \"error\": ");
        try std.json.Stringify.encodeJsonString(f, .{}, j.out);
    }
    try j.out.writeAll("}");
}

/// `x` as the shortest JSON number that reads back as the same f64 (the
/// shorter of Ryu's decimal and scientific forms); `null` when not finite.
pub fn writeNumber(out: *Writer, x: f64) Writer.Error!void {
    if (!std.math.isFinite(x)) return out.writeAll("null");
    var dec: [std.fmt.float.bufferSize(.decimal, f64)]u8 = undefined;
    var exp: [std.fmt.float.min_buffer_size]u8 = undefined;
    // Both buffers hold every finite f64.
    const d = std.fmt.float.render(&dec, x, .{ .mode = .decimal }) catch unreachable;
    const e = std.fmt.float.render(&exp, x, .{ .mode = .scientific }) catch unreachable;
    try out.writeAll(if (e.len < d.len) e else d);
}

/// The JSON `variant` object of a result from row `row` of `variants`
/// (null: the nominal circuit) at `temp_c`, an HSPICE `.temp` list entry
/// (#235): `{"kind": "nominal"}`, or `"temp"` for a nominal run at a
/// `.temp` entry; a step or data row's `params`, an `.alter` row's
/// `index`, a Monte Carlo row's `trial`. A row of an `.alter` run of
/// another kind adds `alter`, a `.temp` entry adds `temp`. The values read
/// back from the row's label, which prints each shortest (`{d}`), so they
/// are the row's own f64s.
pub fn writeVariant(out: *Writer, variants: core.Variants, row: ?u32, temp_c: ?f64) Writer.Error!void {
    if (row) |v| {
        const kind = variants.kindAt(v);
        try out.print("{{\"kind\": \"{t}\"", .{kind});
        var params = false;
        var pairs = std.mem.splitSequence(u8, variants.labelAt(v), ", ");
        while (pairs.next()) |pair| {
            const eq = std.mem.indexOfScalar(u8, pair, '=') orelse continue;
            const name = pair[0..eq];
            const text = pair[eq + 1 ..];
            // `alter=k` and `monte=t` print integers; `optimize=` names
            // the optimization, no coordinate of the point.
            if (std.mem.eql(u8, name, "alter")) {
                try out.print(", \"{s}\": {s}", .{ if (kind == .alter) "index" else "alter", text });
            } else if (std.mem.eql(u8, name, "monte")) {
                try out.print(", \"trial\": {s}", .{text});
            } else if (!std.mem.eql(u8, name, "optimize")) {
                try out.writeAll(if (params) ", " else ", \"params\": {");
                params = true;
                try std.json.Stringify.encodeJsonString(name, .{}, out);
                try out.writeAll(": ");
                try writeNumber(out, std.fmt.parseFloat(f64, text) catch nan);
            }
        }
        if (params) try out.writeAll("}");
    } else try out.writeAll(if (temp_c == null) "{\"kind\": \"nominal\"" else "{\"kind\": \"temp\"");
    if (temp_c) |t| {
        try out.writeAll(", \"temp\": ");
        try writeNumber(out, t);
    }
    try out.writeAll("}");
}

test writeNumber {
    var buf: [64]u8 = undefined;
    // The shorter form wins: decimal for 144.242, scientific for 1000.
    for ([_]struct { f64, []const u8 }{
        .{ 144.242, "144.242" },
        .{ 9e-6, "9e-6" },
        .{ 30304900, "30304900" },
        .{ 1000, "1e3" },
        .{ 1e300, "1e300" },
        .{ std.math.nan(f64), "null" },
        .{ -std.math.inf(f64), "null" },
    }) |case| {
        var w: Writer = .fixed(&buf);
        try writeNumber(&w, case[0]);
        try std.testing.expectEqualStrings(case[1], w.buffered());
    }
    // Oracle: the text reads back as the same f64, bit for bit.
    for ([_]f64{ 9.999999999999999e-06, 0.1 + 0.2, 5e-324, std.math.floatMax(f64), -2.5e-17, 0.000627185 }) |x| {
        var w: Writer = .fixed(&buf);
        try writeNumber(&w, x);
        try std.testing.expectEqual(@as(u64, @bitCast(x)), @as(u64, @bitCast(try std.fmt.parseFloat(f64, w.buffered()))));
    }
}

test writeVariant {
    const v: core.Variants = .{
        .labels = &.{ "ib=0.000009999999999999999, w=2", "rv=1000", "monte=17", "alter=2", "alter=1, monte=3" },
        .kinds = &.{ .step, .data, .monte, .alter, .monte },
    };
    var buf: [256]u8 = undefined;
    for ([_]struct { ?u32, ?f64, []const u8 }{
        .{ null, null, "{\"kind\": \"nominal\"}" },
        .{ null, 85, "{\"kind\": \"temp\", \"temp\": 85}" },
        .{ 0, null, "{\"kind\": \"step\", \"params\": {\"ib\": 9.999999999999999e-6, \"w\": 2}}" },
        .{ 1, null, "{\"kind\": \"data\", \"params\": {\"rv\": 1e3}}" },
        .{ 2, null, "{\"kind\": \"monte\", \"trial\": 17}" },
        .{ 3, null, "{\"kind\": \"alter\", \"index\": 2}" },
        .{ 4, -40, "{\"kind\": \"monte\", \"alter\": 1, \"trial\": 3, \"temp\": -40}" },
    }) |case| {
        var w: Writer = .fixed(&buf);
        try writeVariant(&w, v, case[0], case[1]);
        try std.testing.expectEqualStrings(case[2], w.buffered());
    }
}
