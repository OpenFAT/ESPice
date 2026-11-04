//! The one physics body both backends run: a device's gathered unknowns,
//! Model and Instance in, its residual, charge, both Jacobians and LTE
//! charges stamped out through a sink. The host sink adds into the planes;
//! the GPU sinks store per-contribution staging cells or lane cells.
//! Also the limiting pass (`limitRange`) and the charge-only pass.
//! Cites the LRM sections named at each use (§6.3.6 `$mfactor`) and #49 for
//! the limiter snap; dev/devices/gpu-evaluation.md for the ground masking.
//!
//! Contents:
//! - `pub const host_pad: bool`
//! - `pub inline fn mfactorOf(comptime D: type, inst: anytype) f64`
//! - `pub fn NoiseFamily(comptime D: type) type`
//! - `pub fn evalRange(comptime D: type, comptime narrow: bool, comptime pad: bool, comptime pw: usize, sink: anytype, first: u32, end: u32, sim: SimState, limiting: bool) void`
//! - `pub fn evalQRange(comptime D: type, comptime S: type, sink: anytype, first: u32, end: u32, sim: SimState, batch: anytype) void`
//! - `pub inline fn snapLimited(comptime n_u: usize, comptime reads: Mask(n_u), comptime writes: Mask(n_u), lx: *[n_u]f64, cur: [n_u]f64) void`
//! - `pub fn limitRange(comptime D: type, sink: anytype, first: u32, end: u32, lim_active: bool, sim: SimState) f64`
//! - `pub inline fn anyNonzero(comptime w: usize, v: @Vector(w, f64)) bool`

const std = @import("std");
const assert = std.debug.assert;
const contract = @import("contract");
const abi = @import("device_abi");
const dual = @import("dual.zig");
const layout = @import("layout.zig");
const points = @import("points.zig");
const GROUND = abi.GROUND;
const SimState = abi.SimState;
const Mask = layout.Mask;
const DualFor = dual.DualFor;
const Real = dual.Real;

/// The host pads each value's lanes to a power of two. Callgrind Ir per
/// instance, dense exact / sparse pow2, 100-instance DC sweeps:
/// hisimhv_va 23.7M/6.8M, hisim2_va 37.1M/13.7M, bsimsoi_va 7.36M/3.95M,
/// vbic13_4t 7.74M/3.83M, b3soidd 4.95M/3.02M, bsim4va 4.14M/2.32M,
/// hicumL2_va 4.53M/3.82M, gummel_poon 3.46M/1.96M, mesa 2.03M/1.58M, mos1
/// with RD/RS 1.33M/1.14M, bsim2 1.35M/0.99M, jfet 0.48M/0.37M, mes
/// 0.46M/0.34M; no wide model lost. On the narrow basis (at most 4 lanes)
/// sparse lost up to 2.8% to dense (bsim1, mos2/3/9); the dense layout went
/// with VerA 0.11's exact masks.
pub const host_pad = true;

/// LRM §6.3.6 `$mfactor` of one instance: every row it stamps (residual,
/// charge, both Jacobians, LTE charges, AC terms, noise PSDs) scales by it,
/// which is exactly `mfactor` parallel copies, since an instance's internal
/// rows are homogeneous in their own unknowns. VerA scales hierarchical
/// children itself and leaves the top-level instance to the host. A model
/// with its own `m` parameter keeps `mfactor` at 1.
pub inline fn mfactorOf(comptime D: type, inst: anytype) f64 {
    return if (comptime @hasField(D.Instance, "mfactor")) inst.mfactor else 1.0;
}

/// `noisePsd`'s family: `eval`'s wide basis in f64, so a `ddx` in
/// a PSD argument (a channel noise's gm) reads its partial. `Real` would read
/// it as 0 (#119).
pub fn NoiseFamily(comptime D: type) type {
    const lane = comptime layout.basisOf(D, false).lane[0..contract.nU(D)].*;
    return DualFor(&lane, false, @hasDecl(D, "collapse"));
}

/// Evaluates instances `[first, end)` of D and stamps residual, charge and
/// both Jacobians through `sink`. The one physics body: the host sink adds
/// into the planes, the GPU sink writes per-contribution staging cells.
///
/// `narrow` selects the reduced derivative basis (see `Basis`) and is sound
/// only on maximally collapsed instances, which `ProtoStore.finalize` sorts to
/// `[0, narrow_count)`. `pad` is `host_pad` on the host and false in
/// `DeviceKernel`, where pad lanes cost registers.
///
/// `pw` is how many instances one device call evaluates (`points.zig`; 1 on
/// the GPU). Up to `pw` consecutive instances sharing a Model row (VerA's
/// batch key) run as one call, and each is then stamped in id order, so
/// every plane cell sums the same values in the same order as at `pw` 1.
/// Asserts that `pw` > 1 only for a host sink and a `batch_ok` device.
pub fn evalRange(comptime D: type, comptime narrow: bool, comptime pad: bool, comptime pw: usize, sink: anytype, first: u32, end: u32, sim: SimState, limiting: bool) void {
    @setEvalBranchQuota(1_000_000);
    const Ev = Eval(D, narrow, pad);
    const use_lim = if (comptime Ev.has_limit) limiting else false;
    assert(first <= end);
    var id: u32 = first;
    if (comptime pw > 1) {
        comptime assert(!@typeInfo(@TypeOf(sink)).pointer.child.on_device);
        comptime assert(D.batch_ok);
        const F = points.PointFamily(Ev.S, pw, D.Instance);
        while (end - id >= 2) {
            const model = sink.model(id);
            var k: u32 = 1;
            while (k < pw and id + k < end and sink.model(id + k) == model) k += 1;
            if (k == 1) {
                Ev.one(sink, id, sim, use_lim);
                id += 1;
                continue;
            }
            // A short run repeats its last instance in the spare points:
            // same inputs, same path, so they add no lead-protocol run.
            var g: [pw]Ev.Gathered = undefined;
            var insts: [pw]*const D.Instance = undefined;
            for (&g, &insts, 0..) |*gw, *iw, w| {
                const src = id + @min(@as(u32, @intCast(w)), k - 1);
                gw.* = Ev.gather(sink, src, use_lim);
                iw.* = sink.inst(src);
            }
            var xv: [Ev.n_u]F.V = undefined;
            inline for (0..Ev.n_u) |u| {
                var pts: [pw]f64 = undefined;
                for (&pts, &g) |*p, gw| p.* = gw.lx[u];
                xv[u] = pts;
            }
            F.bind(insts);
            const r = Ev.compute(F, &xv, model, sink.inst(id), sim);
            for (0..k) |w| Ev.stamp(sink, id + @as(u32, @intCast(w)), Ev.pointOf(F, &r, w), &g[w], model, use_lim);
            id += k;
        }
        assert(id <= end);
    }
    while (id < end) : (id += 1) Ev.one(sink, id, sim, use_lim);
}

/// Restamps `q_vec` and `q_tape` for instances `[first, end)` and leaves the
/// other planes alone. Host only. With `S = RealFor(...)` the charges are
/// bit-identical to a full `evalRange` pass, at no gradient cost.
///
/// The transient calls this once per accepted step: Newton returns without
/// reassembling at the accepted iterate, so the planes still hold the
/// previous iterate's charge. The limiting correction is skipped because
/// limits are cleared before Newton returns.
///
/// With `batch` non-null each instance runs `acceptQ` instead of `q`: the
/// same charges, plus the staging `updateState` does, off one core run.
pub fn evalQRange(comptime D: type, comptime S: type, sink: anytype, first: u32, end: u32, sim: SimState, batch: anytype) void {
    @setEvalBranchQuota(1_000_000);
    const n_u = comptime contract.nU(D);
    const q_row = comptime layout.writtenRows(D, "q_rows");
    var id: u32 = first;
    while (id < end) : (id += 1) {
        var lx: [n_u]f64 = undefined;
        inline for (0..n_u) |u| lx[u] = sink.x(sink.gath(id, u));
        const qs = if (comptime @TypeOf(batch) == @TypeOf(null))
            D.q(S, &lx, sink.model(id), sink.inst(id), sim)
        else
            D.acceptQ(S, &lx, sink.model(id), &batch.instances[id], &batch.states[id], sim);
        const qo = contract.qRows(D, S, qs);
        const mf = mfactorOf(D, sink.inst(id));
        // Same `q_row` gate as `evalRange`, or the two passes would differ.
        inline for (0..n_u) |ru| if (comptime q_row[ru]) sink.scatterQ(sink.rhsRow(id, ru), qo[ru].v * mf);
        inline for (comptime layout.lteSites(D), 0..) |k, j| sink.tapeQ(id, j, qs[k].v * mf);
    }
}

/// Writes back `cur[u]` for every written unknown whose limited value lies
/// within 16 ulps of the instance's largest read. At a fixed point every
/// limiter returns its probe, but a derived leg rebuilt from it (x_d = x_s +
/// vds) rounds, and that few-ulp gap would run the limiting correction at
/// every converged iterate (#49). A real clamp is orders larger.
pub inline fn snapLimited(comptime n_u: usize, comptime reads: Mask(n_u), comptime writes: Mask(n_u), lx: *[n_u]f64, cur: [n_u]f64) void {
    var scale: f64 = 0;
    inline for (0..n_u) |u| if (comptime layout.has(reads, u)) {
        scale = @max(scale, @abs(cur[u]));
    };
    const tol = 16 * std.math.floatEps(f64) * scale;
    inline for (0..n_u) |u| if (comptime layout.has(writes & reads, u)) {
        if (@abs(lx[u] - cur[u]) <= tol) lx[u] = cur[u];
    };
}

/// SPICE-style limiting over instances `[first, end)`: `lim = D.limit(cur,
/// old)`, where `cur` is the local x and `old` is the previous limited value
/// (once limiting is active) or the local `x_old`. Returns 1.0 when any
/// instance reports `converged = false`, which forces another Newton
/// iteration; the device decides whether its clamp was large enough.
///
/// Unknowns `limit` reads but never writes take `old` from `x_old`. That
/// equals the previous limited copy whenever `postStep` runs once per `x_old`
/// update, which holds on every CPU path (only GPU backtracking calls it
/// twice).
pub fn limitRange(comptime D: type, sink: anytype, first: u32, end: u32, lim_active: bool, sim: SimState) f64 {
    const n_u = comptime contract.nU(D);
    // The device's read and write sets; a MOS ladder reads 4 of its 8
    // unknowns and writes 2. All ones when the device does not declare them.
    const reads = comptime layout.limitReadsOf(D);
    const writes = comptime layout.limitWritesOf(D);
    // f64 rather than bool: measured +0.1% instructions with a bool here.
    var flag: f64 = 0;
    var id: u32 = first;
    while (id < end) : (id += 1) {
        // Lanes outside `reads` stay undefined: `limit` never reads them, and
        // their outputs fall outside `writes`, so nothing stores them.
        var cur: [n_u]f64 = undefined;
        var old: [n_u]f64 = undefined;
        inline for (0..n_u) |u| if (comptime layout.has(reads, u)) {
            const gi = sink.gath(id, u);
            cur[u] = sink.x(gi);
            old[u] = if (comptime !layout.has(writes, u))
                sink.xOld(gi)
            else if (lim_active)
                sink.lim(id, u)
            else
                sink.xOld(gi);
        };
        // A plain call on purpose. LLVM inlines it anyway; forcing
        // `.always_inline` measured 0.2-0.3% slower.
        var lm = D.limit(Real, sink.model(id), sink.inst(id), cur, old, sim);
        snapLimited(n_u, reads, writes, &lm.x, cur);
        if (!lm.converged) flag = 1;
        inline for (0..n_u) |u| if (comptime layout.has(writes, u)) {
            sink.setLim(id, u, lm.x[u]);
        };
    }
    return flag;
}

/// `@reduce(.Or, v != 0)` computed on the bits: true when any lane is nonzero
/// or NaN, false for ±0. Lowers to vpaddq + vptest instead of a mask shuffle.
/// Differential test against the float compare: tests/eval.zig.
pub inline fn anyNonzero(comptime w: usize, v: @Vector(w, f64)) bool {
    return @reduce(.Or, @as(@Vector(w, u64), @bitCast(v)) << @splat(1)) != 0;
}

// ---- private ----

/// `evalRange`'s steps for device `D` on one basis: the comptime facts, then
/// gather one instance, run the device, take one point of a batched run,
/// and stamp one instance.
fn Eval(comptime D: type, comptime narrow: bool, comptime pad: bool) type {
    @setEvalBranchQuota(1_000_000);
    return struct {
        const n_u = contract.nU(D);
        const has_limit = @hasDecl(D, "limit");
        const reads = q: { @setEvalBranchQuota(1_000_000); break :q layout.widen(n_u, contract.derivReads(D)); };
        const jac_pat = q: { @setEvalBranchQuota(1_000_000); break :q layout.rowPattern(D, "jac_pattern"); };
        const q_pat = q: { @setEvalBranchQuota(1_000_000); break :q layout.rowPattern(D, "q_pattern"); };
        const jac_row = q: { @setEvalBranchQuota(1_000_000); break :q layout.writtenRows(D, "jac_rows"); };
        const q_row = q: { @setEvalBranchQuota(1_000_000); break :q layout.writtenRows(D, "q_rows"); };
        const B = q: { @setEvalBranchQuota(1_000_000); break :q layout.basisOf(D, narrow); };
        const W = B.w;
        const lane = B.lane[0..n_u].*;
        const alias = blk: {
            var a: [n_u]bool = @splat(false);
            if (narrow) for (&a, D.collapse_full) |*o, c| {
                o.* = c != null;
            };
            break :blk a;
        };
        const jac_rep = q: { @setEvalBranchQuota(1_000_000); break :q layout.repMask(n_u, lane, alias, jac_pat); };
        const q_rep = q: { @setEvalBranchQuota(1_000_000); break :q layout.repMask(n_u, lane, alias, q_pat); };
        const S = DualFor(&lane, pad, @hasDecl(D, "collapse"));
        // Every lane of the basis: the layout the limiting correction sums in.
        const full = q: { @setEvalBranchQuota(1_000_000); break :q S.Of(reads).lanes; };
        const lim_writes = q: { @setEvalBranchQuota(1_000_000); break :q if (has_limit) layout.limitWritesOf(D) else 0; };
        const has_q = @hasDecl(D, "q");
        // `evalQ` computes both halves from one call into the model core;
        // calling `eval` and `q` separately runs the core twice.
        const fuse = has_q and @hasDecl(D, "evalQ");

        /// One instance's gathered unknowns.
        const Gathered = struct {
            lx: [n_u]f64,
            active: [n_u]bool,
            /// x minus its limited image, per lane. `canNarrow` guarantees
            /// no two corrected unknowns share a lane.
            corr: @Vector(W, f64),
        };

        fn Result(comptime F: type) type {
            return struct {
                out: contract.Rows(D, F),
                qs: if (has_q) contract.Sites(D, F) else void,
            };
        }

        inline fn one(sink: anytype, id: u32, sim: SimState, use_lim: bool) void {
            const g = gather(sink, id, use_lim);
            const model = sink.model(id);
            stamp(sink, id, compute(S, &g.lx, model, sink.inst(id), sim), &g, model, use_lim);
        }

        inline fn gather(sink: anytype, id: u32, use_lim: bool) Gathered {
            var g: Gathered = .{ .lx = undefined, .active = undefined, .corr = @splat(0) };
            inline for (0..n_u) |u| {
                const gi = sink.gath(id, u);
                g.active[u] = gi != GROUND;
                const xg = sink.x(gi);
                g.lx[u] = xg;
                // Only unknowns `limit` writes have a limited image; the
                // other lanes of `corr` stay zero.
                if (comptime layout.has(lim_writes, u)) if (use_lim) {
                    const l = sink.lim(id, u);
                    g.lx[u] = l;
                    g.corr[lane[u]] = xg - l;
                };
            }
            return g;
        }

        inline fn compute(comptime F: type, x: *const [n_u]F.V, model: anytype, inst: anytype, sim: SimState) Result(F) {
            // `q` returns one charge per ddt site. The planes take the rows
            // (`qRows`); the host `q_tape` takes the LTE sites.
            if (comptime fuse) {
                const both = @call(.always_inline, D.evalQ, .{ F, x, model, inst, sim });
                return .{ .out = both.res, .qs = both.q };
            }
            return .{ .out = D.eval(F, x, model, inst, sim), .qs = if (comptime has_q) D.q(F, x, model, inst, sim) else {} };
        }

        /// Point `w` of a batched result, as the scalar family's values.
        inline fn pointOf(comptime F: type, r: *const Result(F), w: usize) Result(S) {
            var o: Result(S) = undefined;
            inline for (0..n_u) |ru| o.out[ru] = r.out[ru].l[w];
            if (comptime has_q) inline for (0..contract.nQ(D)) |k| {
                o.qs[k] = r.qs[k].l[w];
            };
            return o;
        }

        inline fn stamp(sink: anytype, id: u32, r: Result(S), g: *const Gathered, model: anytype, use_lim: bool) void {
            const SinkT = @typeInfo(@TypeOf(sink)).pointer.child;
            // Ground predicates only on the GPU. On the host a ground stamp
            // lands in the shared trash slot/row, so the add is cheaper than
            // the test. On the GPU that add is a contended write on one
            // address, measured 2x on 40,000 instances
            // (dev/devices/gpu-evaluation.md, "One eval").
            const mask_ground = comptime SinkT.on_device;
            const active = g.active;
            // Near convergence the limiter leaves almost every instance
            // alone, so one test skips the correction dot products.
            const corr_live = use_lim and anyNonzero(W, g.corr);
            const mf = mfactorOf(D, sink.inst(id));
            const out = r.out;
            const qs = r.qs;
            const qo = if (comptime has_q) contract.qRows(D, S, qs) else {};

            // Rows the device never writes (`jac_row`) and entries it can
            // never fill (`jac_pat`) are dropped at comptime; the planes start
            // at +0.0, so a skipped `+= 0.0` cannot even change a sign bit.
            // Resistive and reactive halves stay in separate passes: fusing
            // them measured slower on the mos1 kernel from register pressure.
            // A column without a lane stamps its `jac_const` partial in the
            // same (row, column) order, so a shared slot sums as the tape
            // orders it.
            inline for (0..n_u) |ru| if (comptime jac_row[ru]) if (!mask_ground or active[ru]) {
                const row = sink.rhsRow(id, ru);
                var val = out[ru].v;
                if (comptime has_limit and @TypeOf(out[ru]).lanes != 0) {
                    // The correction lands on the f64 residual. A row without
                    // lanes has zero gradient, so it has no correction either.
                    if (corr_live) val += @reduce(.Add, head(W, out[ru].toLanes(full).grad()) * g.corr);
                }
                sink.scatterRes(row, val * mf);
                if (comptime !SinkT.skip_g and jac_pat[ru] != 0) {
                    inline for (0..n_u) |cu| if (!mask_ground or active[cu]) {
                        if (comptime layout.has(jac_rep[ru], cu)) {
                            sink.scatterJac(id, ru, cu, out[ru].ddxAt(cu) * mf);
                        } else if (comptime layout.constEntry(D, ru, cu)) |e| {
                            if (comptime e.g != 0) if (contract.jacConstApplies(D, e, model, S.collapse_applied)) sink.scatterJac(id, ru, cu, e.g * mf);
                        }
                    };
                }
            };

            if (comptime has_q) {
                // Gated on `q_row`, never on `q_pat`: a `ddt()` of something
                // that varies in t but not in x has a clear pattern row and a
                // live charge.
                inline for (0..n_u) |ru| if (comptime q_row[ru]) {
                    const row = sink.rhsRow(id, ru);
                    var qv = qo[ru].v;
                    if (comptime has_limit and @TypeOf(qo[ru]).lanes != 0) {
                        if (corr_live) qv += @reduce(.Add, head(W, qo[ru].toLanes(full).grad()) * g.corr);
                    }
                    sink.scatterQ(row, qv * mf);
                    if (comptime !SinkT.skip_c and q_pat[ru] != 0) {
                        if (!mask_ground or active[ru]) {
                            inline for (0..n_u) |cu| if (!mask_ground or active[cu]) {
                                if (comptime layout.has(q_rep[ru], cu)) {
                                    sink.scatterQJac(id, ru, cu, qo[ru].ddxAt(cu) * mf);
                                } else if (comptime layout.constEntry(D, ru, cu)) |e| {
                                    if (comptime e.c != 0) if (contract.jacConstApplies(D, e, model, S.collapse_applied)) sink.scatterQJac(id, ru, cu, e.c * mf);
                                }
                            };
                        }
                    }
                };
                if (comptime !SinkT.on_device and !(@hasDecl(SinkT, "skip_tape") and SinkT.skip_tape)) inline for (comptime layout.lteSites(D), 0..) |k, j| {
                    var qv = qs[k].v;
                    if (comptime has_limit and @TypeOf(qs[k]).lanes != 0) {
                        if (corr_live) qv += @reduce(.Add, head(W, qs[k].toLanes(full).grad()) * g.corr);
                    }
                    sink.tapeQ(id, j, qv * mf);
                };
            }
        }
    };
}

/// The first `w` lanes of `v`: the unpadded basis, so the correction dot
/// product sums exactly the lanes it sums without padding.
inline fn head(comptime w: usize, v: anytype) @Vector(w, f64) {
    return @shuffle(f64, v, undefined, std.simd.iota(i32, w));
}
