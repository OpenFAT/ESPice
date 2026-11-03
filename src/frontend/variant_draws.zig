//! Monte Carlo draws. In: a deck's `.option seed`/`sampling` and a trial.
//! Out: counter-based draws per distribution site (Latin hypercube
//! optional), and the live values of a point or trial set on the netlist.
//! Cites: HSPICE [SA] Ch.25 (Monte Carlo), #239 (SIGMA_SCALE).
//!
//! Contents:
//! - `pub const Sampler`
//! - `pub fn init(arena: std.mem.Allocator, config: []const netlist.Config, n: u32) !Sampler`
//! - `pub fn uniform(s: *Sampler, site: u64, trial: u32, index: u32) f64`
//! - `pub const OneSigma`
//! - `pub fn sample(self: OneSigma, site: u64, f: expr.Fn, args: []const f64) f64`
//! - `pub const Trial`
//! - `pub fn uniform(t: Trial, site: u64) f64`
//! - `pub fn normal(t: Trial, site: u64) f64`
//! - `pub fn unit(t: Trial, site: u64) f64`
//! - `pub const Sss`
//! - `pub fn scaleOf(s: Sss, trial: u64) ?usize`
//! - `pub fn fails(s: Sss, gpa: std.mem.Allocator, values: []const f64) !bool`
//! - `pub fn sample(self: TrialDraw, site: u64, f: expr.Fn, args: []const f64) f64`
//! - `pub fn setPoint(scratch: std.mem.Allocator, stack: *std.ArrayList(expr.Val), nl: *Netlist, pt: variants.Point) !void`
//! - `pub fn drawOnce(scratch: std.mem.Allocator, nl: *Netlist) !void`
//! - `pub fn sssCards(arena: std.mem.Allocator, deck: netlist.Deck) ![]const Sss`

const std = @import("std");
const assert = std.debug.assert;
const testing = std.testing;
const Allocator = std.mem.Allocator;
const core = @import("core");
const device = @import("device");
const netlist = @import("netlist");
const builder = @import("builder");
const analyses = @import("analyses.zig");
const expr = netlist.expr;
const Netlist = netlist.Netlist;
const ParamRef = device.abi.ParamRef;
const variants = @import("variants.zig");

/// Draws for one sweep's trials: counter-based, so trial k's value at a
/// site depends only on (seed, k, site), never on order or other sites.
/// Latin hypercube stratifies each site over the sweep's trials.
pub const Sampler = struct {
    seed: u64,
    n: u32,
    lhs: bool,
    arena: std.mem.Allocator,
    /// Latin hypercube: per site, trial index to stratum.
    strata: std.AutoHashMapUnmanaged(u64, []const u32) = .empty,

    /// Reads `.option seed=` (HSPICE default 1) and `sampling_method=lhs`;
    /// any method other than `lhs` or `srs` is `UnsupportedCard`.
    pub fn init(arena: std.mem.Allocator, config: []const netlist.Config, n: u32) !Sampler {
        var s: Sampler = .{ .seed = 1, .n = n, .lhs = false, .arena = arena };
        for (config) |c| if (!c.temp) for (c.args, 0..) |a, i| {
            if (a != .name or i + 1 >= c.args.len) continue;
            const next = c.args[i + 1];
            // lossyCast: a NaN or out-of-range seed saturates, never traps.
            if (std.ascii.eqlIgnoreCase(a.name, "seed") and next == .num) s.seed = std.math.lossyCast(u64, @abs(next.num));
            if (std.ascii.eqlIgnoreCase(a.name, "sampling_method") and next == .name) {
                if (std.ascii.eqlIgnoreCase(next.name, "lhs")) s.lhs = true else if (!std.ascii.eqlIgnoreCase(next.name, "srs")) return error.UnsupportedCard;
            }
        };
        return s;
    }

    /// Uniform in (0, 1) for HSPICE trial number `trial` at `site`.
    /// `index` (0-based within the sweep) picks the Latin hypercube
    /// stratum: it must be below `n`, which only the safety-checked
    /// `strata[index]` enforces. The first Latin hypercube draw at a site
    /// allocates its strata in `arena` (on allocation failure the draw
    /// falls back to plain sampling).
    pub fn uniform(s: *Sampler, site: u64, trial: u32, index: u32) f64 {
        const bits = netlist.siteKey(netlist.siteKey(s.seed, trial), site);
        const u = (@as(f64, @floatFromInt(bits >> 11)) + 0.5) * 0x1.0p-53;
        if (!s.lhs) return u;
        const strata = s.strataOf(site) catch return u;
        return (@as(f64, @floatFromInt(strata[index])) + u) / @as(f64, @floatFromInt(s.n));
    }

    /// A random permutation of the `n` strata for `site`.
    fn strataOf(s: *Sampler, site: u64) ![]const u32 {
        const gop = try s.strata.getOrPut(s.arena, site);
        if (gop.found_existing) return gop.value_ptr.*;
        const perm = try s.arena.alloc(u32, s.n);
        for (perm, 0..) |*x, i| x.* = @intCast(i);
        var prng = std.Random.DefaultPrng.init(netlist.siteKey(s.seed, site));
        // A u64 index draws the same bits on every target; `shuffle`'s
        // usize index permuted differently on wasm32.
        prng.random().shuffleWithIndex(u32, perm, u64);
        gop.value_ptr.* = perm;
        return perm;
    }
};

/// Holds every distribution call at its nominal except those at `site`,
/// which sit one sigma above it: a uniform's half range over sqrt(3), a
/// `limit`'s whole range (`Planner.siteVariations`).
pub const OneSigma = struct {
    site: u64,

    /// Asserts that `args` holds at least the nominal value.
    pub fn sample(self: OneSigma, site: u64, f: expr.Fn, args: []const f64) f64 {
        std.debug.assert(args.len >= 1);
        const nom = args[0];
        if (site != self.site) return nom;
        const width = if (args.len > 1) args[1] else 0;
        const gaussian = f == .agauss or f == .gauss;
        const sigmas = if (gaussian and args.len > 2 and args[2] != 0) args[2] else 1;
        return nom + switch (f) {
            .agauss => width / sigmas,
            .gauss => nom * width / sigmas,
            .aunif => width / @sqrt(3.0),
            .unif => nom * width / @sqrt(3.0),
            else => width, // else: `limit`; only distribution calls reach a sampler
        };
    }
};

/// One Monte Carlo trial of a sweep.
pub const Trial = struct {
    sampler: *Sampler,
    /// HSPICE trial number; keys the draws.
    trial: u32,
    /// Position within the sweep; picks the Latin hypercube stratum.
    index: u32,
    /// `SIGMA_SCALE` (#239): every draw's sigma times this.
    scale: f64 = 1,
    /// `.wcd` (#240): each site's standard normal coordinate, fixed; a site
    /// it lacks sits at 0, its nominal. The sampler then answers only a
    /// `limit`, which `WorstCase` refuses wherever it moves a parameter.
    z: ?*const std.AutoHashMapUnmanaged(u64, f64) = null,

    pub fn uniform(t: Trial, site: u64) f64 {
        return t.sampler.uniform(site, t.trial, t.index);
    }

    /// The standard normal coordinate of `site` times the sigma scale; at
    /// scale 1 the bits of `normalQuantile(uniform)`. Under `z`, its entry.
    pub fn normal(t: Trial, site: u64) f64 {
        if (t.z) |z| return z.get(site) orelse 0;
        return t.scale * core.numerics.normalQuantile(t.uniform(site));
    }

    /// The uniform of `site` through the scaled normal, Φ(s·Φ⁻¹(u)): the
    /// scale acts in the standard-normal space, so a uniform's mass moves
    /// to its edges. At scale 1 it is `uniform`, bit for bit. Under `z`,
    /// Φ of its entry (Rosenblatt's map).
    pub fn unit(t: Trial, site: u64) f64 {
        return if (t.scale == 1 and t.z == null) t.uniform(site) else core.numerics.normalCdf(t.normal(site));
    }
};

/// A `SWEEP MONTE=n SIGMA_SCALE=... FAIL=...` card (#239): trial
/// `first + j*n + k` of its run drew at `scales[j]`.
pub const Sss = struct {
    /// The analysis whose `.meas` cards FAIL reads.
    kind: core.query.Kind,
    /// Its `n`, `first`, `scales` and compiled `fail`.
    monte: netlist.Monte,

    /// The scale row of HSPICE trial `trial`; null for a trial the card
    /// did not plan.
    pub fn scaleOf(s: Sss, trial: u64) ?usize {
        if (trial < s.monte.first) return null;
        const j = (trial - s.monte.first) / s.monte.n;
        if (j >= s.monte.scales.len) return null;
        const row: usize = @intCast(j);
        return row;
    }

    /// Whether the trial whose `.meas` values are `values` (parallel to
    /// the deck's cards) fails: a card FAIL reads has no value, or FAIL
    /// is NaN or nonzero. Fails only with `OutOfMemory`, in `gpa`.
    pub fn fails(s: Sss, gpa: std.mem.Allocator, values: []const f64) !bool {
        const f = s.monte.fail;
        for (f.ops) |op| if (op.code == .live and !std.math.isFinite(values[op.a])) return true;
        var stack: std.ArrayList(expr.Val) = .empty;
        defer stack.deinit(gpa);
        // NaN != 0 too.
        return (try expr.eval(gpa, &stack, f.ops, f.consts, values, null)) != 0;
    }
};

/// Answers the distribution calls of one trial [SA Ch.20 "Monte Carlo
/// Parameter Distribution"]: `agauss(nom, abs, sigmas[, mult])`,
/// `gauss(nom, rel, sigmas[, mult])`, `aunif(nom, abs[, mult])`,
/// `unif(nom, rel[, mult])`, `limit(nom, abs)`. With a multiplier m the
/// draw of largest deviation among m is kept.
const TrialDraw = struct {
    trial: Trial,

    /// The trial's draw for distribution call `f` at `site`. Asserts that
    /// `args` holds at least the nominal value.
    pub fn sample(self: TrialDraw, site: u64, f: expr.Fn, args: []const f64) f64 {
        std.debug.assert(args.len >= 1);
        const nom = args[0];
        const width = if (args.len > 1) args[1] else 0;
        const gaussian = f == .agauss or f == .gauss;
        const sigmas = if (gaussian and args.len > 2 and args[2] != 0) args[2] else 1;
        const mult_at: usize = if (gaussian) 3 else 2;
        const mult: u32 = if (f != .limit and args.len > mult_at and args[mult_at] >= 1) @intFromFloat(@min(args[mult_at], 1e6)) else 1;
        const scale = switch (f) {
            .agauss => width / sigmas,
            .gauss => nom * width / sigmas,
            .unif => nom * width,
            else => width,
        };
        var best: f64 = 0;
        for (0..mult) |m| {
            const key = netlist.siteKey(site, m);
            // A `limit` is two points either side: no sigma scale moves it.
            const d = switch (f) {
                .agauss, .gauss => scale * self.trial.normal(key),
                .limit => if (self.trial.uniform(key) < 0.5) -scale else scale,
                else => scale * (2 * self.trial.unit(key) - 1),
            };
            if (m == 0 or @abs(d) > @abs(best)) best = d;
        }
        return nom + best;
    }
};

/// Sets `nl`'s live values to `pt`'s, drawing its Monte Carlo trial if it
/// has one. `stack` is evaluation scratch, grown with `scratch`.
pub fn setPoint(scratch: std.mem.Allocator, stack: *std.ArrayList(expr.Val), nl: *Netlist, pt: variants.Point) !void {
    if (pt.trial) |t| {
        _ = try nl.setLive(scratch, stack, pt.live, TrialDraw{ .trial = t });
    } else _ = try nl.setLive(scratch, stack, pt.live, null);
}

/// Draws every distribution site of `nl` once, as trial 1 of the deck's
/// `.option seed`, into its live values (`Deck.draw_seed`). `scratch` holds
/// the sampler and the evaluation stack.
pub fn drawOnce(scratch: std.mem.Allocator, nl: *Netlist) !void {
    var sampler = try Sampler.init(scratch, nl.deck.config, 1);
    var stack: std.ArrayList(expr.Val) = .empty;
    _ = try nl.setLive(scratch, &stack, nl.live.nominal, TrialDraw{ .trial = .{ .sampler = &sampler, .trial = 1, .index = 0 } });
}

/// The deck's SIGMA_SCALE cards, their slices copied into `arena`.
pub fn sssCards(arena: std.mem.Allocator, deck: netlist.Deck) ![]const Sss {
    var out: std.ArrayList(Sss) = .empty;
    for (deck.analyses) |a| if (a.sweep == .monte and a.sweep.monte.scales.len != 0) {
        var m = a.sweep.monte;
        m.scales = try arena.dupe(f64, m.scales);
        m.fail = .{ .text = try arena.dupe(u8, m.fail.text), .ops = try arena.dupe(expr.Op, m.fail.ops), .consts = try arena.dupe(f64, m.fail.consts) };
        try out.append(arena, .{ .kind = a.kind, .monte = m });
    };
    return out.items;
}

// ---- private ----

test "Sampler.init reads seed and sampling method, and a wild seed saturates" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const cfg = struct {
        fn of(args: []const netlist.Value) [1]netlist.Config {
            return .{.{ .temp = false, .args = args }};
        }
    }.of;
    var c = cfg(&.{ .{ .name = "seed" }, .{ .num = -7 }, .{ .name = "sampling_method" }, .{ .name = "LHS" } });
    const s = try Sampler.init(a, &c, 4);
    try std.testing.expectEqual(@as(u64, 7), s.seed);
    try std.testing.expect(s.lhs);
    c = cfg(&.{ .{ .name = "seed" }, .{ .num = comptime std.math.nan(f64) } });
    try std.testing.expectEqual(@as(u64, 0), (try Sampler.init(a, &c, 4)).seed);
    c = cfg(&.{ .{ .name = "seed" }, .{ .num = 1e300 } });
    try std.testing.expectEqual(@as(u64, std.math.maxInt(u64)), (try Sampler.init(a, &c, 4)).seed);
    c = cfg(&.{ .{ .name = "sampling_method" }, .{ .name = "sobol" } });
    try std.testing.expectError(error.UnsupportedCard, Sampler.init(a, &c, 4));
    // A trailing key with no value is skipped.
    c = cfg(&.{.{ .name = "seed" }});
    try std.testing.expectEqual(@as(u64, 1), (try Sampler.init(a, &c, 4)).seed);
}

test "TrialDraw keeps the largest deviation of a multiplier's draws" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    var sampler = try Sampler.init(arena.allocator(), &.{}, 1);
    const draw: TrialDraw = .{ .trial = .{ .sampler = &sampler, .trial = 1, .index = 0 } };
    const one = draw.sample(5, .agauss, &.{ 1, 0.1, 1 });
    const many = draw.sample(5, .agauss, &.{ 1, 0.1, 1, 8 });
    // m = 0 is the single draw, so the max over eight is at least as far.
    try std.testing.expect(@abs(many - 1) >= @abs(one - 1));
    // `limit` lands on nom +- abs exactly; `aunif` stays inside the band.
    try std.testing.expectEqual(@as(f64, 0.5), @abs(draw.sample(9, .limit, &.{ 2, 0.5 }) - 2));
    try std.testing.expect(@abs(draw.sample(9, .aunif, &.{ 2, 0.5 }) - 2) < 0.5);
    // Nominal only: no spread.
    try std.testing.expectEqual(@as(f64, 3), draw.sample(9, .agauss, &.{3}));
}

test "TrialDraw at sigma scale 1 is the unscaled draw's bits; at s, s times its normal coordinate (#239)" {
    // Oracle: the draw written out from the sampler's uniform, as it was
    // before the scale existed, and Φ(s·Φ⁻¹(u)) for a uniform.
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    var sampler = try Sampler.init(arena.allocator(), &.{}, 1);
    const q = core.numerics.normalQuantile;
    const u = sampler.uniform(netlist.siteKey(5, 0), 1, 0);
    for ([_]f64{ 1, 3 }) |s| {
        const draw: TrialDraw = .{ .trial = .{ .sampler = &sampler, .trial = 1, .index = 0, .scale = s } };
        const gauss = draw.sample(5, .agauss, &.{ 0, 0.5, 2 });
        const unif = draw.sample(5, .aunif, &.{ 0, 0.5 });
        if (s == 1) {
            try std.testing.expectEqual(@as(u64, @bitCast(0.25 * q(u))), @as(u64, @bitCast(gauss)));
            try std.testing.expectEqual(@as(u64, @bitCast(0.5 * (2 * u - 1))), @as(u64, @bitCast(unif)));
        } else {
            try std.testing.expectApproxEqRel(s * 0.25 * q(u), gauss, 1e-14);
            try std.testing.expectApproxEqRel(0.5 * (2 * core.numerics.normalCdf(s * q(u)) - 1), unif, 1e-14);
        }
        // A two-point `limit` does not move with the scale.
        try std.testing.expectEqual(@as(f64, if (u < 0.5) 1.5 else 2.5), draw.sample(5, .limit, &.{ 2, 0.5 }));
    }
}
