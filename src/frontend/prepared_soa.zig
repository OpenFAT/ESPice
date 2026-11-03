//! Safe-operating-area limits (#238). In: the deck's `.options soalimits=`
//! file and its M, Q, D, R and C cards' model cards. Out: the file's
//! ratings (`Netlist.soa_ratings`), checked at the trust boundary, and one
//! `core.SoaLimit` per limit a device's model card or the file states.
//! Cites: #238; ngspice's `*soachk.c` (the limit names and terminals).
//!
//! Contents:
//! - `pub fn ratings(io: std.Io, session: Allocator, file: []const u8) ![]const netlist.SoaRating`
//! - `pub fn limits(arena: Allocator, nl: *const netlist.Netlist, nb: *const builder.NetBuilder, circuit: *const device.Circuit) ![]const core.SoaLimit`

const std = @import("std");
const Allocator = std.mem.Allocator;
const core = @import("core");
const netlist = @import("netlist");
const device = @import("device");
const builder = @import("builder");
const NO_NODE = netlist.none;
const GROUND = core.GROUND;

/// A model-card SOA limit (ngspice's names) of card letter `card`: held
/// across terminals `pos` and `neg`, by magnitude unless `abs` is false.
const SoaKey = struct { card: u8, key: []const u8, quantity: []const u8, pos: u8, neg: u8, abs: bool = true };

/// MOS pins d g s b, BJT c b e, diode a k; a diode's `bv_max` holds the
/// reverse voltage v(k) - v(a), as ngspice's diosoachk.c does.
const soa_keys = [_]SoaKey{
    .{ .card = 'm', .key = "vgs_max", .quantity = "vgs", .pos = 1, .neg = 2 },
    .{ .card = 'm', .key = "vgd_max", .quantity = "vgd", .pos = 1, .neg = 0 },
    .{ .card = 'm', .key = "vgb_max", .quantity = "vgb", .pos = 1, .neg = 3 },
    .{ .card = 'm', .key = "vds_max", .quantity = "vds", .pos = 0, .neg = 2 },
    .{ .card = 'm', .key = "vbs_max", .quantity = "vbs", .pos = 3, .neg = 2 },
    .{ .card = 'm', .key = "vbd_max", .quantity = "vbd", .pos = 3, .neg = 0 },
    .{ .card = 'q', .key = "vbe_max", .quantity = "vbe", .pos = 1, .neg = 2 },
    .{ .card = 'q', .key = "vbc_max", .quantity = "vbc", .pos = 1, .neg = 0 },
    .{ .card = 'q', .key = "vce_max", .quantity = "vce", .pos = 0, .neg = 2 },
    .{ .card = 'd', .key = "bv_max", .quantity = "vr", .pos = 1, .neg = 0, .abs = false },
    .{ .card = 'r', .key = "bv_max", .quantity = "v", .pos = 0, .neg = 1 },
    .{ .card = 'c', .key = "bv_max", .quantity = "v", .pos = 0, .neg = 1 },
};

/// The `.options soalimits=` file at `file` (already resolved against the
/// deck), checked here, at the trust boundary: one JSON object of model
/// name or name prefix to an object of `soa_keys` limits, each a finite
/// number (#238). `InvalidSoaLimits`, logged with the file, for anything
/// else: bad JSON, a repeated name, an unknown key, a value that is no
/// finite number; the read error of an unreadable file.
pub fn ratings(io: std.Io, session: Allocator, file: []const u8) ![]const netlist.SoaRating {
    const bytes = std.Io.Dir.cwd().readFileAlloc(io, file, session, .limited(1 << 24)) catch |err| {
        if (!@import("builtin").is_test) std.log.err("netlist: cannot read soalimits file {s}: {s}", .{ file, @errorName(err) });
        return err;
    };
    const Limits = std.json.ArrayHashMap(f64);
    const models = std.json.parseFromSliceLeaky(std.json.ArrayHashMap(Limits), session, bytes, .{}) catch |err| {
        if (!@import("builtin").is_test) std.log.err("netlist: soalimits file {s}: {s}", .{ file, @errorName(err) });
        return error.InvalidSoaLimits;
    };
    var out: std.ArrayList(netlist.SoaRating) = .empty;
    for (models.map.keys(), models.map.values()) |model, limit_map| for (limit_map.map.keys(), limit_map.map.values()) |key, limit| {
        const known = for (soa_keys) |k| {
            if (std.mem.eql(u8, k.key, key)) break true;
        } else false;
        if (!known or !std.math.isFinite(limit)) {
            if (!@import("builtin").is_test) std.log.err("netlist: soalimits file {s}: {s}: {s} is {s}", .{ file, model, key, if (known) "no finite number" else "no SOA limit" });
            return error.InvalidSoaLimits;
        }
        try out.append(session, .{ .model = model, .key = key, .limit = limit });
    };
    return out.items;
}

/// One `core.SoaLimit` per `soa_keys` limit the model card of an M, Q, D,
/// R or C card states, each terminal named by its result column as
/// `NetBuilder.publish` labels it. A model card that states none takes
/// the `soalimits` file's limits for the longest name that is its own or
/// a prefix of it (`Netlist.soa_ratings`), so a card value is never
/// shadowed. A limit that is no number, or names a terminal the card does
/// not connect, is skipped.
pub fn limits(arena: Allocator, nl: *const netlist.Netlist, nb: *const builder.NetBuilder, circuit: *const device.Circuit) ![]const core.SoaLimit {
    var out: std.ArrayList(core.SoaLimit) = .empty;
    for ("mqdrc") |card| for (nl.bucket(card)) |e| {
        const d = nl.device(e);
        const m = d.model orelse continue;
        const stated = for (soa_keys) |k| {
            if (k.card == card and cardValue(m.kv, k.key) != null) break true;
        } else false;
        // ponytail: a scan of the file's rows per device; a map by model
        // name when a kit's file holds thousands.
        var rated: ?[]const u8 = null;
        if (!stated) for (nl.soa_ratings) |r| {
            if (std.ascii.startsWithIgnoreCase(m.name, r.model) and (rated == null or r.model.len > rated.?.len)) rated = r.model;
        };
        for (soa_keys) |k| {
            if (k.card != card or @max(k.pos, k.neg) >= d.pins.len) continue;
            const value: ?f64 = if (rated) |model| rating(nl.soa_ratings, model, k.key) else if (cardValue(m.kv, k.key)) |v| v.number() else null;
            const limit = value orelse continue;
            var columns: [2][]const u8 = undefined;
            for (&columns, [2]u8{ k.pos, k.neg }) |*column, pin| {
                const row = nb.frozenRow(d.pins[pin].index());
                column.* = if (row == NO_NODE or row == GROUND) "" else try std.mem.concat(arena, u8, &.{ "v(", circuit.nodeName(row), ")" });
            }
            try out.append(arena, .{ .name = try arena.dupe(u8, d.name), .quantity = k.quantity, .key = k.key, .limit = limit, .pos = columns[0], .neg = columns[1], .abs = k.abs });
        }
    };
    return out.items;
}

// ---- private ----

/// The value model-card pairs `kv` give `key`; null when they give none.
fn cardValue(kv: []const netlist.Kv, key: []const u8) ?netlist.Value {
    for (kv) |item| if (std.mem.eql(u8, item.key, key)) return item.value;
    return null;
}

/// The `soalimits` file's `key` limit for `model`; null when it gives none.
fn rating(rows: []const netlist.SoaRating, model: []const u8, key: []const u8) ?f64 {
    for (rows) |r| if (std.mem.eql(u8, r.model, model) and std.mem.eql(u8, r.key, key)) return r.limit;
    return null;
}
