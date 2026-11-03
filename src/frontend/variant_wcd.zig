//! `.wcd` worst-case distance points as variant rows (#240). In: the
//! planner of the deck's main run and points in standard-normal space, one
//! coordinate per statistical parameter. Out: one fixed-z trial row per
//! point, made through `Planner.add` as `Tuner.rows` makes its rows.
//! Cites: #240; dev/analysis/variants.md.
//!
//! Contents:
//! - `pub const WorstCase`
//! - `pub fn init(planner: variant_planner.Planner) !WorstCase`
//! - `pub fn rows(w: *WorstCase, arena: std.mem.Allocator, nominal: *const device.Circuit, points: []const f64, base: usize) !core.Variants`
//! - `pub fn sample(self: Shapes, site: u64, f: expr.Fn, args: []const f64) f64`

const std = @import("std");
const core = @import("core");
const device = @import("device");
const netlist = @import("netlist");
const expr = netlist.expr;
const variant_draws = @import("variant_draws.zig");
const variant_planner = @import("variant_planner.zig");
const variant_mismatch = @import("variant_mismatch.zig");
const variants = @import("variants.zig");

/// Turns `.wcd` points in standard-normal space into variant rows (#240):
/// coordinate i is draw key `keys[i]`, which a `.variation`/`DEV`/`LOT` row
/// draws under (one per device, or one per model and parameter) or a
/// distribution site that moves a parameter; z = 0 is every draw at its
/// nominal, also under `.option seed`. Each point is a fixed-z trial
/// through `Planner.add`, as `Tuner.rows`' points are. Lives in the parse
/// arena, with the netlist it re-evaluates.
pub const WorstCase = struct {
    planner: variant_planner.Planner,
    spec: netlist.Wcd,
    keys: []const u64,
    /// `<card>@<param>` or `<model>@<param>`, as the mismatch groups.
    labels: []const []const u8,
    /// One search per `.step` point, or the nominal one
    /// (`Planner.basePoints`); `stepped` when they are `.step` points,
    /// whose queries run variant row b.
    bases: []const variants.Point,
    stepped: bool,
    /// The trial's sampler, which only a `limit` that moves nothing reads.
    sampler: *variant_draws.Sampler,
    /// The point `rows` sets, by key.
    z: std.AutoHashMapUnmanaged(u64, f64) = .empty,

    /// A worst-case search over the planner of the deck's main run.
    /// Asserts that the deck has a `.wcd`; `UnsupportedCard` with `.alter`
    /// (each run is its own circuit), without any statistical parameter,
    /// or when one is a `limit` or a multiplier draw, which no single
    /// normal coordinate moves smoothly.
    pub fn init(planner: variant_planner.Planner) !WorstCase {
        var p = planner;
        const nl = p.nl;
        const spec = nl.deck.wcd.?;
        if (nl.deck.alters.len != 0) return p.refuse(spec.line, ".wcd together with .alter");
        var keys: std.ArrayList(u64) = .empty;
        var labels: std.ArrayList([]const u8) = .empty;
        var seen: std.AutoHashMapUnmanaged(u64, void) = .empty;
        const targets = try variant_mismatch.variationTargets(&p);
        const names = try core.query.CardNames.init(p.scratch, p.cards);
        for (nl.deck.variations, 0..) |v, i| {
            const lo = targets.starts[i];
            const hi = targets.starts[i + 1];
            for (targets.ords[lo..hi], targets.keys[lo..hi]) |ord, key| {
                if (v.dist == .limit) return p.refuse(spec.line, ".wcd over a limit variation");
                if ((try seen.getOrPut(p.scratch, key)).found_existing) continue;
                const r = p.refs.at(ord);
                try keys.append(p.scratch, key);
                try labels.append(p.scratch, try std.fmt.allocPrint(p.scratch, "{s}@{s}", .{ if (v.per_device) names.get(r.type, r.index) orelse "?" else v.model, v.param }));
            }
        }
        if (nl.live.site_keys.len != 0) {
            const groups = try p.siteVariations();
            var refused: std.AutoHashMapUnmanaged(u64, void) = .empty;
            try refused.ensureTotalCapacity(p.scratch, @intCast(nl.live.site_keys.len));
            _ = try nl.setLive(p.scratch, &p.stack, nl.live.nominal, Shapes{ .refused = &refused });
            _ = try nl.setLive(p.scratch, &p.stack, nl.live.nominal, null);
            for (groups.labels, p.site_groups.items) |label, site| {
                if (refused.contains(site)) return p.refuse(spec.line, ".wcd over a limit() or a multiplier draw");
                // The key `TrialDraw` draws a site's first call under.
                const key = netlist.siteKey(site, 0);
                if ((try seen.getOrPut(p.scratch, key)).found_existing) continue;
                try keys.append(p.scratch, key);
                try labels.append(p.scratch, label);
            }
        }
        if (keys.items.len == 0) return p.refuse(spec.line, ".wcd with no statistical parameter: no .variation, DEV/LOT or distribution call moves a device");
        const sampler = try p.scratch.create(variant_draws.Sampler);
        sampler.* = try .init(p.scratch, nl.deck.config, 1);
        const bases = try p.basePoints();
        return .{ .planner = p, .spec = spec, .keys = keys.items, .labels = labels.items, .bases = bases, .stepped = nl.deck.steps.len != 0, .sampler = sampler };
    }

    /// One row per point of `points` (`keys.len` coordinates each) on top
    /// of `bases[base]`, allocated in `arena`, as `Tuner.rows` makes them;
    /// `UnsupportedCard` for a point that changes the topology. `nominal`
    /// is the prepared circuit the rows write into.
    pub fn rows(w: *WorstCase, arena: std.mem.Allocator, nominal: *const device.Circuit, points: []const f64, base: usize) !core.Variants {
        const p = &w.planner;
        p.nominal = nominal;
        p.sim = arena;
        inline for (.{ &p.labels, &p.temps, &p.axes, &p.kinds, &p.write_refs, &p.write_values }) |list| list.clearRetainingCapacity();
        p.starts.shrinkRetainingCapacity(1);
        const b = w.bases[base];
        const d = w.keys.len;
        std.debug.assert(points.len % d == 0);
        var at: usize = 0;
        while (at < points.len) : (at += d) {
            for (w.keys, points[at..][0..d]) |key, x| try w.z.put(p.scratch, key, x);
            try p.add(.{ .label = "", .axis = 0, .live = b.live, .temp = b.temp, .cards = b.cards, .trial = .{ .sampler = w.sampler, .trial = 1, .index = 0, .z = &w.z } });
            if (p.rebuilds.items.len != 0) return p.refuse(w.spec.line, "a .wcd point that changes the topology");
        }
        return p.table();
    }
};

/// Answers every distribution call with its nominal, and marks each site
/// whose call is a `limit` or carries a multiplier above 1, as
/// `TrialDraw` reads them (`WorstCase.init`). `refused` has room for every site.
const Shapes = struct {
    refused: *std.AutoHashMapUnmanaged(u64, void),

    /// Asserts that `args` holds at least the nominal value.
    pub fn sample(self: Shapes, site: u64, f: expr.Fn, args: []const f64) f64 {
        std.debug.assert(args.len >= 1);
        const mult_at: usize = if (f == .agauss or f == .gauss) 3 else 2;
        if (f == .limit or (args.len > mult_at and args[mult_at] >= 2)) self.refused.putAssumeCapacity(site, {});
        return args[0];
    }
};
