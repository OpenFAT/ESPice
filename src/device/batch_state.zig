//! The analysis-state hooks of one device batch: the host's iterates,
//! accepted points and settings in, the batch's lim plane, Instance
//! history, State latches and stored `SimState` out. Newton history,
//! limiting, accepted-step updates, temperature and homotopy knobs,
//! breakpoints and the status and display channels.
//! Cites LRM §5.2.1 (`analog initial`), §5.10 (held variables and timers),
//! §9.4 (display tasks), §9.7.3 (status), §9.10 (`$temperature`), §9.15
//! (`$simparam`), §9.17.2 (`$bound_step`), §4.5.4 (`idt`), §3.6.3.2
//! (nodesets), ngspice's VSRCaccept, and #15 for the split passes.
//!
//! Contents:
//! - `pub fn State(comptime D: type) type`, with `advanceIteration`,
//!   `checkConvergence`, `markInternalRows`, `predictFirstIterate`, `seedFn`,
//!   `seedNodeset`, `seedIc`, `markCurrentRows`, `applyLimits`,
//!   `limitRangeHook`, `armLimits`, `updateRange`, `updateDone`,
//!   `clearLimits`, `updateState`, `sayFn`, `statusFn`, `boundStep`,
//!   `stateCtl`, `setTemp`, `setHomotopy`, `setSimState`,
//!   `recomputePrecomputed`, `minDelay`, `nextBreakpointFn`

const std = @import("std");
const assert = std.debug.assert;
const Allocator = std.mem.Allocator;
const contract = @import("contract");
const abi = @import("device_abi");
const layout = @import("layout.zig");
const stamp = @import("stamp.zig");
const traits = @import("traits.zig");
const DeviceBatch = @import("batch.zig").DeviceBatch;
const breakpoints = @import("breakpoints.zig");
const dual = @import("dual.zig");
const Bypass = @import("batch_bypass.zig").Bypass;
const Sink = @import("sink.zig").Sink;
const GROUND = abi.GROUND;
const SimState = abi.SimState;
const StateCtlOp = abi.StateCtlOp;
const Planes = abi.Planes;
const LimMap = layout.LimMap;
const RealFor = dual.RealFor;
const Real = dual.Real;

/// The analysis-state hooks of `DeviceBatch(D)`, each over a type-erased batch.
pub fn State(comptime D: type) type {
    return struct {
        const Batch = DeviceBatch(D);
        const n_u: usize = contract.nU(D);
        const has_state = @hasDecl(D, "State");
        const has_q = @hasDecl(D, "q");
        const has_limit = @hasDecl(D, "limit");
        // Only timer-only models: their breakpoints are pure in (model, t).
        const has_bp = @hasDecl(D, "nextBreakpoint") and traits.skipsTimerState(D);
        const has_src_brk = traits.hasSpiceBreaks(D);
        const Ac = traits.AcTerms(D);
        /// The `tp<k>_t` time keys of D's `vera_timepoint` caches that hold
        /// a slot (`tp<k>_s<j>`) as long as a dirty-tracked held array (one
        /// with a `<name>__dirty` range). Others save no copy by a rerun
        /// (ltra caches only scalars), so they keep their cache.
        const tp_keys: []const []const u8 = blk: {
            var keys: []const []const u8 = &.{};
            if (!@hasDecl(D, "Instance")) break :blk keys;
            @setEvalBranchQuota(10_000_000);
            const I = @typeInfo(D.Instance).@"struct";
            for (I.field_names) |key| {
                if (!(key.len > 4 and std.mem.startsWith(u8, key, "tp") and std.mem.endsWith(u8, key, "_t"))) continue;
                const slot = key[0 .. key.len - 1] ++ "s";
                const holds_tracked = for (I.field_names, I.field_types) |f, T| {
                    if (!std.mem.startsWith(u8, f, slot) or @typeInfo(T) != .array) continue;
                    const tracked = for (I.field_names, I.field_types) |h, H| {
                        if (@hasField(D.Instance, h ++ "__dirty") and H == T) break true;
                    } else false;
                    if (tracked) break true;
                } else false;
                if (holds_tracked) keys = keys ++ .{key};
            }
            break :blk keys;
        };

        pub fn advanceIteration(ctx: *anyopaque, previous_x: []const f64) void {
            const self = Batch.of(ctx);
            for (0..self.count) |id|
                D.advanceIteration(Real, self.model(id), &self.instances[id], self.localX(previous_x, id), self.sim);
        }

        pub fn checkConvergence(ctx: *anyopaque, x: []const f64) bool {
            const self = Batch.of(ctx);
            for (0..self.count) |id| {
                if (!D.checkConvergence(Real, self.model(id), &self.instances[id], self.localX(x, id), self.sim)) return false;
            }
            return true;
        }

        pub fn markInternalRows(ctx: *anyopaque, mask: []bool) void {
            const self = Batch.of(ctx);
            for (0..self.count) |id| {
                inline for (D.num_ports..n_u) |u| if (comptime D.u_kinds[u] == .voltage) {
                    const row = self.gath[id * n_u + u];
                    if (row != GROUND) mask[row] = true;
                };
            }
        }

        pub fn predictFirstIterate(ctx: *anyopaque, trial: []f64, cur: []const f64, prev: []const f64, xfact: f64) void {
            const self = Batch.of(ctx);
            for (0..self.count) |id| {
                inline for (D.num_ports..n_u) |u| if (comptime D.u_kinds[u] == .voltage) {
                    const row = self.gath[id * n_u + u];
                    if (row != GROUND) trial[row] = cur[row] + xfact * (cur[row] - prev[row]);
                };
            }
        }

        pub fn seedFn(ctx: *anyopaque, x: []f64) void {
            const self = Batch.of(ctx);
            if (comptime has_limit) {
                // Only the slots `limitRange` maintains are ever read back.
                const writes = comptime layout.limitWritesOf(D);
                for (0..self.count) |id| {
                    const sv = D.seed(Real, self.model(id), &self.instances[id], self.sim);
                    inline for (0..n_u) |u| if (comptime layout.has(writes, u)) {
                        self.lim_x[id * LimMap(D).k + comptime LimMap(D).at(u)] = sv[u] orelse x[self.gath[id * n_u + u]];
                    };
                }
                self.lim_active = true;
            }
        }

        pub fn seedNodeset(ctx: *anyopaque, x: []f64) void {
            const self = Batch.of(ctx);
            for (0..self.count) |id| inline for (D.u_nodeset, 0..) |ns, u| if (ns) |value| {
                const row = self.gath[id * n_u + u];
                if (row != GROUND) x[row] = value;
            };
        }

        pub fn seedIc(ctx: *anyopaque, x: []f64) void {
            const self = Batch.of(ctx);
            const R = RealFor(@hasDecl(D, "collapse"));
            for (0..self.count) |id| inline for (comptime traits.idtUnknowns(D)) |u| {
                const gi = self.gath[id * n_u + u];
                if (gi != GROUND) {
                    // The static row is s - ic when the idt has an ic (slope
                    // 1) and does not read s otherwise (slope 0). One exact
                    // Newton step on the row lands s on ic.
                    var lx = self.localX(x, id);
                    const r0 = D.eval(R, &lx, self.model(id), &self.instances[id], self.sim)[u].v;
                    lx[u] += 1.0;
                    const slope = D.eval(R, &lx, self.model(id), &self.instances[id], self.sim)[u].v - r0;
                    if (slope != 0) x[gi] -= r0 / slope;
                }
            };
        }

        /// Marks the rows of non-voltage unknowns (branch currents) unless the
        /// same node is also one of the instance's voltage unknowns.
        pub fn markCurrentRows(ctx: *anyopaque, mask: []bool) void {
            const self = Batch.of(ctx);
            // Runtime loops: n_u^2 unrolled overflows the quota past 64 unknowns.
            for (0..self.count) |id| {
                const nd = self.gath[id * n_u ..][0..n_u];
                for (nd, D.u_kinds) |node, kind| {
                    if (kind == .voltage or node == GROUND) continue;
                    const voltage_alias = for (nd, D.u_kinds) |other, k| {
                        if (k == .voltage and other == node) break true;
                    } else false;
                    if (!voltage_alias) mask[node] = true;
                }
            }
        }

        pub fn applyLimits(ctx: *anyopaque, x: []f64, x_old: []const f64) bool {
            const self = Batch.of(ctx);
            const any = limitRangeHook(ctx, x, x_old, 0, self.count);
            self.lim_active = true;
            return any;
        }

        pub fn limitRangeHook(ctx: *anyopaque, x: []f64, x_old: []const f64, first: u32, last: u32) bool {
            const self = Batch.of(ctx);
            // `limitRange` never touches the planes, but `host` reads their
            // pointers, and `undefined` would trap in Debug.
            const no_planes: Planes = .{ .g_vals = &.{}, .c_vals = &.{}, .rhs = &.{}, .q_vec = &.{} };
            var sink = Sink(D, false, false, false).host(self, &no_planes, x, x_old.ptr);
            return stamp.limitRange(D, &sink, first, last, self.lim_active, self.sim) != 0;
        }

        pub fn armLimits(ctx: *anyopaque) void {
            const self = Batch.of(ctx);
            self.lim_active = true;
        }

        pub fn updateRange(ctx: *anyopaque, x: []const f64, first: u32, last: u32) ?f64 {
            const self = Batch.of(ctx);
            assert(first <= last);
            assert(last <= self.count);
            var min_reject: ?f64 = null;
            for (first..last) |id| {
                // VerA's `vera_timepoint` cache on a hit makes `updateState`
                // copy every cached held array whole into the Instance and
                // dirty it end to end, so `stateCtl` copies it whole again:
                // 2 x 96 KB per TXL line per step. Dropped (NaN, which no
                // time equals), the statement reruns in place from the held
                // state and SimState the cached eval read, so the same
                // values, and dirties only the elements it stores. A rerun
                // that shifts most of a full history costs more than the two
                // copies, so `updateDone` turns it off then.
                if (self.tp_rerun) inline for (tp_keys) |f| {
                    @field(self.instances[id], f) = std.math.nan(f64);
                };
                const lx = self.localX(x, id);
                switch (D.updateState(Real, self.model(id), &self.instances[id], lx, &self.states[id], self.sim)) {
                    .ok => {},
                    .request_reject_at => |tr| {
                        min_reject = if (min_reject) |cur| @min(cur, tr) else tr;
                    },
                }
            }
            return min_reject;
        }

        pub fn updateDone(ctx: *anyopaque) void {
            const self = Batch.of(ctx);
            self.held_initial = true;
            self.sim.analog_initial = false;
            if (comptime tp_keys.len != 0) decideRerun(self);
        }

        pub fn clearLimits(ctx: *anyopaque) void {
            const self = Batch.of(ctx);
            self.lim_active = false;
        }

        /// Runs `D.updateState` on every instance at `x` and returns the
        /// earliest time any of them asks the step to be rejected at, if any.
        /// Called once per converged solve (or per accepted step for
        /// `commit_state`), not per Newton iteration.
        pub fn updateState(ctx: *anyopaque, x: []const f64) ?f64 {
            const self = Batch.of(ctx);
            const min_reject = updateRange(ctx, x, 0, self.count);
            updateDone(ctx);
            return min_reject;
        }

        /// Values one `say` call can record: every site once, or 256, so a
        /// task in a loop has room too. More counts as lost (`formatSay`).
        const say_len = if (@hasDecl(D, "say")) blk: {
            var n: usize = 0;
            for (D.say_sites) |site| n += 1 + site.nargs;
            break :blk @max(n, 256);
        } else 0;

        /// `Hooks.say`: instance `id`'s display tasks at x, rendered to `w`.
        /// Reads the host Instance copy; a device with `say` is never
        /// resident (the built-ins are generated without it).
        pub fn sayFn(ctx: *anyopaque, x: []const f64, id: u32, name: []const u8, w: *std.Io.Writer) void {
            const self = Batch.of(ctx);
            assert(id < self.count);
            var buf: [say_len]f64 = undefined;
            var out: contract.Say = .{ .buf = &buf };
            const lx = self.localX(x, id);
            D.say(Real, &lx, self.model(id), &self.instances[id], self.sim, &out);
            assert(out.len <= buf.len);
            // A failed writer drops the text; the host prints nothing for it.
            contract.formatSay(D, &out, name, w) catch |err| switch (err) {
                error.WriteFailed => {},
            };
        }

        /// `Hooks.status`. Reads the host Instance copy; for a resident batch
        /// the GPU launcher downloads it first (`GpuContext.syncStatus`).
        pub fn statusFn(ctx: *anyopaque, msg: []u8) ?abi.StatusHit {
            const self = Batch.of(ctx);
            for (self.instances[0..self.count], 0..) |*inst, i| {
                if (inst.vera_status__ == 0) continue;
                var w: std.Io.Writer = .fixed(msg);
                // A full buffer truncates the message, which is all `msg` can hold.
                contract.formatStatus(D, inst, &w) catch |err| switch (err) {
                    error.WriteFailed => {},
                };
                return .{ .index = @intCast(i), .len = @intCast(w.end) };
            }
            return null;
        }

        /// Tightest `$bound_step` across this batch's instances. Read after an
        /// accepted step, so `updateState` has already refreshed every one.
        pub fn boundStep(ctx: *anyopaque) f64 {
            const self = Batch.of(ctx);
            var b = std.math.inf(f64);
            for (self.instances[0..self.count]) |*inst| b = @min(b, inst.bound_step);
            return b;
        }

        pub fn stateCtl(ctx: *anyopaque, op: StateCtlOp) bool {
            const self = Batch.of(ctx);
            var dirty = false;
            for (0..self.count) |id| {
                if (D.stateCtl(self.model(id), &self.instances[id], &self.states[id], @fromBackingInt(@intCast(@backingInt(op))))) dirty = true;
            }
            switch (op) {
                .commit => self.committed_initial = self.held_initial,
                .revert => {
                    self.held_initial = self.committed_initial;
                    self.sim.analog_initial = !self.held_initial;
                },
                .query => {},
            }
            return dirty;
        }

        /// Sets every Model row's temperature from Celsius (`.temp`) to the
        /// Kelvin `$temperature` reads (LRM §9.10), then reruns `setup` and
        /// `precompute`. Rows stay rows: every one gets the same value.
        pub fn setTemp(ctx: *anyopaque, temp_c: f32) void {
            const self = Batch.of(ctx);
            for (self.models) |*m| m.temperature__ = @as(f64, temp_c) + 273.15;
            self.reprep();
        }

        /// Writes the §9.15 homotopy knobs into every Model row, then reruns
        /// `setup` when it caches either one (VerA `setup_simparams`).
        /// Parameter defaults derived from them keep their build-time value.
        pub fn setHomotopy(ctx: *anyopaque, gmin: f64, source_scale: f64) void {
            const self = Batch.of(ctx);
            for (self.models) |*m| {
                if (comptime @hasField(D.Model, "gmin__")) m.gmin__ = gmin;
                if (comptime @hasField(D.Model, "source_scale__")) m.source_scale__ = source_scale;
            }
            const cached = comptime if (@hasDecl(D, "setup_simparams")) blk: {
                for (D.setup_simparams) |n| {
                    if (std.mem.eql(u8, n, "gmin") or std.mem.eql(u8, n, "sourceScaleFactor")) break :blk true;
                }
                break :blk false;
            } else false;
            if (cached) {
                self.setupRows();
                if (comptime @hasDecl(D, "precompute")) {
                    for (self.instances, 0..) |*inst, id| D.precompute(inst, self.model(id));
                }
            }
        }

        pub fn setSimState(ctx: *anyopaque, st: SimState) void {
            const self = Batch.of(ctx);
            // Serial and between solves, so the counts are final and the
            // decision does not depend on how ParEval split the batch. Only
            // a transient decides: a cold operating point moves every
            // instance, which says nothing about the latency to come.
            if (comptime Bypass(D).bypassable) if (self.bypass) |bp| {
                if (st.kind != self.sim.kind) {
                    bp.hits.store(0, .monotonic);
                    bp.misses.store(0, .monotonic);
                } else if (st.kind == .tran) bp.review(self.count);
            };
            self.sim = st;
            if (self.held_initial) self.sim.analog_initial = false;
        }

        /// Reruns `setup` and `precompute` and returns false when `collapse` now asks for
        /// a node aliasing the frozen tapes do not have.
        pub fn recomputePrecomputed(ctx: *anyopaque) bool {
            const self = Batch.of(ctx);
            self.reprep();
            if (comptime @hasDecl(D, "collapse")) {
                for (self.instances, 0..) |*inst, id| {
                    const col = D.collapse(Real, self.model(id), inst);
                    const nd = self.gath[id * n_u ..][0..n_u];
                    inline for (D.num_ports..n_u) |u| {
                        if (col[u]) |target| {
                            if (nd[u] != nd[target]) return false;
                        } else if (std.mem.indexOfScalar(u32, nd[0..u], nd[u]) != null) {
                            // The builder gave every unaliased internal its own node.
                            return false;
                        }
                    }
                }
            }
            return true;
        }

        pub fn minDelay(ctx: *anyopaque) f64 {
            const self = Batch.of(ctx);
            var min_td = std.math.inf(f64);
            for (self.models) |*m| {
                for (D.delays(m)) |d| min_td = @min(min_td, d);
            }
            return min_td;
        }

        /// The earliest model or instance breakpoint strictly after `t`. The
        /// transient asks once per step; for a timer-only model (`has_bp`)
        /// the answer only moves when `t` reaches it, so the walk (68 timers
        /// per vsource model, ~425 Ir) runs once per breakpoint instead. A
        /// PULSE or V-source PWL corner rounds from the accepted time `sim.t`
        /// (the transient asks right after accepting it), as ngspice's
        /// VSRCaccept does.
        pub fn nextBreakpointFn(ctx: *anyopaque, t: f64) ?f64 {
            const self = Batch.of(ctx);
            if (comptime has_bp) {
                if (self.bp.lo <= t and t < self.bp.hi) return if (self.bp.hi == std.math.inf(f64)) null else self.bp.hi;
            }
            const t_acc = if (self.sim.kind == .tran and self.sim.t <= t) self.sim.t else t;
            var best: f64 = std.math.inf(f64);
            if (comptime @hasDecl(D, "nextBreakpoint")) for (self.models, 0..) |*m, i| {
                if (comptime has_src_brk) {
                    best = @min(best, breakpoints.srcBreak(D, &self.src_brk[i], m, t_acc, t));
                    continue;
                }
                if (D.nextBreakpoint(m, t)) |bp| best = @min(best, bp);
            };
            // §5.10.3.3 re-armed fire times exist only in the committed
            // Instance, so every commit can move them: no cache.
            if (comptime traits.walksPending(D)) for (self.instances) |*inst| {
                if (D.pendingBreakpoint(inst, t)) |bp| best = @min(best, bp);
            };
            if (comptime has_bp) self.bp = .{ .lo = t, .hi = best };
            return if (best == std.math.inf(f64)) null else best;
        }

        // ---- private ----

        /// Keeps `tp_rerun` while the reruns of one accepted solve stored at
        /// most 1/8 of the tracked held elements (a TXL line appending a
        /// point stores 1 of 2048; one at capacity shifts about half), and
        /// reruns again 64 solves after turning it off. Serial (`updateDone`).
        // ponytail: one flag per batch from its summed spans; per instance
        // if one batch ever holds lines on both sides.
        fn decideRerun(self: *Batch) void {
            assert(tp_keys.len != 0);
            if (!self.tp_rerun) {
                self.tp_idle += 1;
                if (self.tp_idle == 64) self.tp_rerun = true;
                return;
            }
            var stored: u64 = 0;
            var tracked: u64 = 0;
            for (self.instances[0..self.count]) |*inst| inline for (comptime @typeInfo(D.Instance).@"struct".field_names) |f| {
                if (comptime std.mem.endsWith(u8, f, "__dirty")) {
                    const d = @field(inst.*, f);
                    const len = @field(inst.*, f[0 .. f.len - "__dirty".len]).len;
                    // Whole: a fresh or restarted history (`initState`), not
                    // a rerun's stores; it says nothing about the next step.
                    if (d[0] == 0 and d[1] == len - 1) return;
                    if (d[0] <= d[1]) stored += @intCast(d[1] - d[0] + 1);
                    tracked += len;
                }
            };
            assert(stored <= tracked);
            self.tp_rerun = stored * 8 <= tracked;
            self.tp_idle = 0;
        }
    };
}
