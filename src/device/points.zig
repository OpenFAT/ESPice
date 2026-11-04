//! A scalar `DualFor` family and a point count `W` in, the batch family VerA's
//! `batch_ok` devices evaluate W instances with (`V = @Vector(W, f64)`) out:
//! each value is W scalar values side by side, so every primitive is the
//! scalar one per point, bit for bit, except that `exp`, `log`, `expm1` and
//! `pow` take all W values in one gompute vector call (each lane the scalar
//! call's bits). The win is the call count; the arithmetic stays the scalar
//! family's. Implements the lead protocol (`batch_lead`) and `instLane`
//! (`batch_inst`).
//! Cites VerA's batch contract (`contract/lead.zig`, docs/host/batching.md)
//! and `contract.family_primitives` for the numerics.
//!
//! Contents:
//! - `pub fn PointFamily(comptime S1: type, comptime W: usize, comptime Inst: type) type`,
//!   whose decls are `V`, `collapse_applied`, `Lane`, `lane_count`, `Inner`,
//!   `Of`, `con`, `lift`, `probe`, `sel`, `laneOf`, `fromLanes`, `instLane`,
//!   `bind`, `leadBegin`, `leadNext`, `leadMerge`, `runs`; on `Inner` also
//!   `decide`, `decideI`, `strip`; each value `to`, `val`, `ddxAt`,
//!   `leadKeep`, `add`, `sub`, `mul`, `div`, `neg`, `scale`, `addC`, `exp`,
//!   `log`, `expm1`, `pow`, `log1p`, `sqrt`, `sin`, `cos`, `tanh`, `atan`,
//!   `sinh`, `cosh`, `lt`, `le`, `eq`

const std = @import("std");
const assert = std.debug.assert;
const contract = @import("contract");
const gompute = @import("gompute");
const dmath = gompute.math;

/// The batch family over `W` points of scalar family `S1` (a `DualFor`),
/// each point reading its own `Inst` row. `bind` names the W instances
/// before each batched call. The lead state and the bound instances are
/// per thread, so concurrent ParEval slices each own theirs.
pub fn PointFamily(comptime S1: type, comptime W: usize, comptime Inst: type) type {
    comptime assert(W >= 2);
    comptime assert(W <= 32); // `contract.LeadState`'s bit sets
    const Pt = @Vector(W, f64);
    return struct {
        /// One value per point (contract `family_fns`).
        pub const V = Pt;
        pub const collapse_applied = S1.collapse_applied;
        /// The scalar family each point runs on (`zPowL` and friends).
        pub const Lane = S1;
        pub const lane_count = W;
        /// The family the lead protocol's runs use: this one, with the
        /// per-point decisions.
        pub const Inner = Core;
        pub const Of = Core.Of;
        pub const con = Core.con;
        pub const lift = Core.lift;
        pub const probe = Core.probe;
        pub const sel = Core.sel;
        pub const laneOf = Core.laneOf;
        pub const fromLanes = Core.fromLanes;
        pub const instLane = Core.instLane;

        /// Names point w's instance; read by `instLane` until the next call.
        pub fn bind(insts: [W]*const Inst) void {
            bound = insts;
        }
        pub fn leadBegin() void {
            lead.begin();
            runs += 1;
        }
        pub fn leadNext() bool {
            const more = lead.next();
            if (more) runs += 1;
            return more;
        }
        pub fn leadMerge(out: anytype, new: @TypeOf(out.*)) void {
            contract.leadMergeInto(out, new, lead.keep());
        }

        /// Runs this thread made since its count was last cleared: a
        /// divergent batch costs one more run per extra path.
        pub threadlocal var runs: u64 = 0;
        threadlocal var lead: contract.LeadState(W, false) = .{};
        threadlocal var bound: [W]*const Inst = undefined;

        const Core = struct {
            pub const V = Pt;
            pub const collapse_applied = S1.collapse_applied;
            pub const Lane = S1;
            pub const lane_count = W;

            pub fn Of(comptime m: anytype) type {
                return P(S1.Of(m));
            }
            pub fn con(c: f64) P(@TypeOf(S1.con(0.0))) {
                return .{ .l = @splat(S1.con(c)) };
            }
            pub fn lift(v: Pt) P(@TypeOf(S1.con(0.0))) {
                var r: P(@TypeOf(S1.con(0.0))) = undefined;
                inline for (0..W) |w| r.l[w] = S1.con(v[w]);
                return r;
            }
            pub fn probe(comptime u: usize, v: Pt) P(@TypeOf(S1.probe(u, 0.0))) {
                var r: P(@TypeOf(S1.probe(u, 0.0))) = undefined;
                inline for (0..W) |w| r.l[w] = S1.probe(u, v[w]);
                return r;
            }
            pub fn sel(c: anytype, a: anytype, b: anytype) P(@TypeOf(S1.sel(c.l[0], a.l[0], b.l[0]))) {
                var r: P(@TypeOf(S1.sel(c.l[0], a.l[0], b.l[0]))) = undefined;
                inline for (0..W) |w| r.l[w] = S1.sel(c.l[w], a.l[w], b.l[w]);
                return r;
            }
            pub fn laneOf(a: anytype, comptime w: usize) @TypeOf(a.l[w]) {
                return a.l[w];
            }
            pub fn fromLanes(comptime m: anytype, ls: [W]S1.Of(m)) P(S1.Of(m)) {
                return .{ .l = ls };
            }
            /// The leader's outcome of `a op b`; points that disagree diverge.
            pub fn decide(comptime op: std.math.CompareOperator, a: anytype, b: anytype) bool {
                var c: @Vector(W, bool) = undefined;
                inline for (0..W) |w| c[w] = std.math.compare(a.l[w].val(), op, b.l[w].val());
                return lead.mark(c);
            }
            pub fn decideI(a: anytype) i64 {
                var r: @Vector(W, i64) = undefined;
                inline for (0..W) |w| r[w] = std.math.lossyCast(i64, @round(a.l[w].val()));
                return lead.markI(r);
            }
            /// Each point's value without lanes, as the scalar `zStrip`.
            pub fn strip(a: anytype) P(@TypeOf(S1.con(0.0))) {
                var r: P(@TypeOf(S1.con(0.0))) = undefined;
                inline for (0..W) |w| r.l[w] = S1.con(a.l[w].val());
                return r;
            }
            pub fn instLane(comptime T: type, comptime w: usize) *const T {
                comptime assert(T == Inst);
                return bound[w];
            }
        };

        /// W values of scalar type `T`, point w at `l[w]`.
        fn P(comptime T: type) type {
            return struct {
                l: [W]T,
                const Self = @This();

                pub fn to(a: Self, comptime m: anytype) Of(m) {
                    var r: Of(m) = undefined;
                    inline for (0..W) |w| r.l[w] = a.l[w].to(m);
                    return r;
                }
                /// The leader's value: what a decision outside `decide` sees.
                pub fn val(a: Self) f64 {
                    return a.l[lead.leader].val();
                }
                pub fn ddxAt(a: Self, comptime u: usize) f64 {
                    return a.l[lead.leader].ddxAt(u);
                }
                /// `contract.leadMergeInto`'s per-point copy.
                pub fn leadKeep(a: *Self, n: Self, keep: @Vector(W, bool)) void {
                    inline for (0..W) |w| {
                        if (keep[w]) a.l[w] = n.l[w];
                    }
                }
                pub fn add(a: Self, b: anytype) P(@TypeOf(a.l[0].add(b.l[0]))) {
                    var r: P(@TypeOf(a.l[0].add(b.l[0]))) = undefined;
                    inline for (0..W) |w| r.l[w] = a.l[w].add(b.l[w]);
                    return r;
                }
                pub fn sub(a: Self, b: anytype) P(@TypeOf(a.l[0].sub(b.l[0]))) {
                    var r: P(@TypeOf(a.l[0].sub(b.l[0]))) = undefined;
                    inline for (0..W) |w| r.l[w] = a.l[w].sub(b.l[w]);
                    return r;
                }
                pub fn mul(a: Self, b: anytype) P(@TypeOf(a.l[0].mul(b.l[0]))) {
                    var r: P(@TypeOf(a.l[0].mul(b.l[0]))) = undefined;
                    inline for (0..W) |w| r.l[w] = a.l[w].mul(b.l[w]);
                    return r;
                }
                pub fn div(a: Self, b: anytype) P(@TypeOf(a.l[0].div(b.l[0]))) {
                    var r: P(@TypeOf(a.l[0].div(b.l[0]))) = undefined;
                    inline for (0..W) |w| r.l[w] = a.l[w].div(b.l[w]);
                    return r;
                }
                pub fn neg(a: Self) Self {
                    return a.each("neg");
                }
                pub fn scale(a: Self, c: f64) Self {
                    var r: Self = undefined;
                    inline for (0..W) |w| r.l[w] = a.l[w].scale(c);
                    return r;
                }
                pub fn addC(a: Self, c: f64) Self {
                    var r: Self = undefined;
                    inline for (0..W) |w| r.l[w] = a.l[w].addC(c);
                    return r;
                }
                // The four with a vector body in gompute: one call for all W
                // values, then each point's own lanes (`DualFor`'s `map`).
                pub fn exp(a: Self) Self {
                    const e = dmath.exp(a.vals());
                    var r: Self = undefined;
                    inline for (0..W) |w| r.l[w] = a.l[w].map(e[w], e[w]);
                    return r;
                }
                pub fn log(a: Self) Self {
                    const g = dmath.log(a.vals());
                    var r: Self = undefined;
                    inline for (0..W) |w| r.l[w] = a.l[w].map(g[w], 1.0 / a.l[w].v);
                    return r;
                }
                pub fn expm1(a: Self) Self {
                    const v = a.vals();
                    const m = dmath.expm1(v);
                    const e = dmath.exp(v);
                    var r: Self = undefined;
                    inline for (0..W) |w| r.l[w] = a.l[w].map(m[w], e[w]);
                    return r;
                }
                pub fn pow(a: Self, c: f64) Self {
                    const p = dmath.pow(a.vals(), @as(Pt, @splat(c)));
                    var r: Self = undefined;
                    inline for (0..W) |w| r.l[w] = a.l[w].powFrom(c, p[w]);
                    return r;
                }
                pub fn log1p(a: Self) Self {
                    return a.each("log1p");
                }
                pub fn sqrt(a: Self) Self {
                    return a.each("sqrt");
                }
                pub fn sin(a: Self) Self {
                    return a.each("sin");
                }
                pub fn cos(a: Self) Self {
                    return a.each("cos");
                }
                pub fn tanh(a: Self) Self {
                    return a.each("tanh");
                }
                pub fn atan(a: Self) Self {
                    return a.each("atan");
                }
                pub fn sinh(a: Self) Self {
                    return a.each("sinh");
                }
                pub fn cosh(a: Self) Self {
                    return a.each("cosh");
                }
                pub fn lt(a: Self, b: anytype) P(@TypeOf(S1.con(0.0))) {
                    var r: P(@TypeOf(S1.con(0.0))) = undefined;
                    inline for (0..W) |w| r.l[w] = a.l[w].lt(b.l[w]);
                    return r;
                }
                pub fn le(a: Self, b: anytype) P(@TypeOf(S1.con(0.0))) {
                    var r: P(@TypeOf(S1.con(0.0))) = undefined;
                    inline for (0..W) |w| r.l[w] = a.l[w].le(b.l[w]);
                    return r;
                }
                pub fn eq(a: Self, b: anytype) P(@TypeOf(S1.con(0.0))) {
                    var r: P(@TypeOf(S1.con(0.0))) = undefined;
                    inline for (0..W) |w| r.l[w] = a.l[w].eq(b.l[w]);
                    return r;
                }

                inline fn vals(a: Self) Pt {
                    var v: Pt = undefined;
                    inline for (0..W) |w| v[w] = a.l[w].v;
                    return v;
                }
                inline fn each(a: Self, comptime f: []const u8) Self {
                    var r: Self = undefined;
                    inline for (0..W) |w| r.l[w] = @field(T, f)(a.l[w]);
                    return r;
                }
            };
        }
    };
}
