//! The latency bypass of one device batch: a Newton iterate's limited
//! unknowns in, either the instance's recorded stamps replayed to first
//! order or a fresh, recorded evaluation out. Opt-in (`Hooks.set_bypass`):
//! it moves values within the Newton tolerance. Cites SPICE's bypass test
//! (the unknowns and the predicted current both within tolerance) and the
//! hand-kept model list `traits.bypass_models`.
//!
//! Contents:
//! - `pub fn Bypass(comptime D: type) type`: the per-batch record (`tol`,
//!   `ref`, `meta`, `inst`, `cells`, `vals`, `rows`, `cols`, `hits`,
//!   `misses`, `off`) with `bypassable`, `rec_cap`, `review`, `setBypass`,
//!   `freeBypass`, `evalBypass`

const std = @import("std");
const assert = std.debug.assert;
const contract = @import("contract");
const abi = @import("device_abi");
const dual = @import("dual.zig");
const layout = @import("layout.zig");
const stamp = @import("stamp.zig");
const traits = @import("traits.zig");
const sink_mod = @import("sink.zig");
const staging_gpa = @import("proto_store.zig").staging_gpa;
const DeviceBatch = @import("batch.zig").DeviceBatch;
const Planes = abi.Planes;
const SimState = abi.SimState;
const RecordCap = sink_mod.RecordCap;
const RecordSink = sink_mod.RecordSink;

/// Per instance of `DeviceBatch(D)`: the limited unknowns and the writes of
/// its last full Newton evaluation, and what that evaluation depended on.
/// Lives on `staging_gpa`, like every buffer a worker thread may create.
pub fn Bypass(comptime D: type) type {
    const n_u: usize = contract.nU(D);
    const has_q = @hasDecl(D, "q");
    const has_limit = @hasDecl(D, "limit");
    return struct {
        tol: abi.BypassTol,
        /// `count * n_u`: each instance's limited unknowns at its recording.
        ref: []f64,
        /// `count`: what each recording depends on; `class` 0 is none.
        meta: []Meta,
        /// `count * @sizeOf(D.Instance)`: each instance's bytes at its
        /// recording, which hold latched and held values. `State` is
        /// `updateState`'s staging, which `eval` never reads.
        inst: []u8,
        /// `RecordSink`'s columns, `count *` `cells_len`, `cells_len`,
        /// `rec_cap.res + rec_cap.jac` and `rec_cap.jac`.
        cells: []*f64,
        vals: []f64,
        rows: []u8,
        cols: []u8,
        /// Instance evaluations that replayed and that evaluated, summed
        /// over ParEval's lanes; read between solves (`setSimState`).
        /// usize: wasm32 has no 64-bit atomics.
        // ponytail: wraps after 2^32 evaluations on 32-bit targets, which
        // only skews `review`'s ratio there; u64 again if that matters.
        hits: std.atomic.Value(usize) = .init(0),
        misses: std.atomic.Value(usize) = .init(0),
        /// Set once the hit rate proved too low to pay for recording: the
        /// batch evaluates plainly for the rest of the run.
        off: bool = false,

        const Bp = @This();
        const Batch = DeviceBatch(D);
        /// A recording's iteration class (0: none), its segment lengths,
        /// and the key of the planes its cells address.
        const Meta = struct { class: u16 = 0, n_res: u16 = 0, n_jac: u16 = 0, n_qjac: u16 = 0, plane: u64 = 0 };

        /// Whether a Newton evaluation of D is a pure function of the
        /// instance's limited unknowns, its bytes and the iteration class:
        /// no time-dependent stimulus, no Newton history, no evaluation that
        /// writes the instance beyond a status latch, and not a constant
        /// Jacobian (whose evaluation costs less than a replay).
        pub const bypassable = !(@hasDecl(D, "nextBreakpoint") and traits.skipsTimerState(D)) and
            !traits.hasSpiceBreaks(D) and !traits.skipsTimerState(D) and
            !@hasDecl(D, "advanceIteration") and !@hasDecl(D, "checkConvergence") and
            !(@hasDecl(D, "mutable_eval") and D.mutable_eval and !traits.statusOnlyMutable(D)) and
            !(@hasDecl(D, "constant") and D.constant.g) and @hasDecl(D, "u_kinds") and
            @sizeOf(D.Instance) <= 4096 and traits.bypass_models.has(traits.baseName(D));

        /// Upper bounds on one instance's recorded writes in a Newton pass:
        /// one residual per written row, its row's pattern entries, and the
        /// charge rows' pattern entries.
        pub const rec_cap: RecordCap = blk: {
            @setEvalBranchQuota(100_000);
            const jp = layout.rowPattern(D, "jac_pattern");
            const jr = layout.writtenRows(D, "jac_rows");
            var c: RecordCap = .{ .res = 0, .jac = 0, .qjac = 0 };
            for (0..n_u) |ru| if (jr[ru]) {
                c.res += 1;
                c.jac += @popCount(jp[ru]);
            };
            if (has_q) {
                const qp = layout.rowPattern(D, "q_pattern");
                const qr = layout.writtenRows(D, "q_rows");
                for (0..n_u) |ru| if (qr[ru]) {
                    c.qjac += @popCount(qp[ru]);
                };
            }
            break :blk c;
        };
        const cells_len = rec_cap.res + rec_cap.jac + rec_cap.qjac;
        const rows_len = rec_cap.res + rec_cap.jac;

        comptime {
            // Local rows and columns ride in u8, a segment length in u16.
            if (bypassable) assert(n_u <= 256);
            if (bypassable) assert(cells_len <= std.math.maxInt(u16));
        }

        /// Turns the bypass off once, after four evaluations per instance, a
        /// batch replays under one in five: it pays the recording for
        /// nothing (a BSIM4 whose instance bytes move every step: 3%; a CMOS
        /// chain: 85%).
        pub fn review(bp: *Bp, count: u32) void {
            if (bp.off) return;
            const h = bp.hits.load(.monotonic);
            const total = h + bp.misses.load(.monotonic);
            if (total < 4 * @as(u64, count)) return;
            if (h * 5 < total) bp.off = true;
        }

        /// `Hooks.set_bypass`: sets the tolerances, clearing every
        /// recording, or frees the bypass when `tol` is null. Best effort:
        /// out of memory leaves the batch evaluating plainly.
        pub fn setBypass(ctx: *anyopaque, tol: ?abi.BypassTol) void {
            const self = Batch.of(ctx);
            if (comptime !bypassable) return;
            if (self.bypass) |bp| {
                bp.tol = tol orelse return freeBypass(self);
                if (!bp.off) @memset(bp.meta, .{});
                return;
            }
            self.bypass = create(self.count, tol orelse return) catch null;
        }

        /// Frees the batch's bypass records, if any.
        pub fn freeBypass(self: *Batch) void {
            const bp = self.bypass orelse return;
            staging_gpa.free(bp.cols);
            staging_gpa.free(bp.rows);
            staging_gpa.free(bp.vals);
            staging_gpa.free(bp.cells);
            staging_gpa.free(bp.inst);
            staging_gpa.free(bp.meta);
            staging_gpa.free(bp.ref);
            staging_gpa.destroy(bp);
            self.bypass = null;
        }

        /// Newton evaluation of `[first, last)` with the latency bypass: each
        /// instance replays its recorded writes when its inputs are within
        /// tolerance of the recording, else evaluates (recording afresh).
        /// Each plane cell's writes replay in their recorded order, so it
        /// sums as a fresh evaluation with those values would.
        pub fn evalBypass(self: *Batch, bp: *Bp, sink: anytype, first: u32, last: u32, sim: SimState, limiting: bool, pl: *const Planes, comptime staged: bool) void {
            assert(first <= last);
            assert(last <= self.count);
            const SinkT = @typeInfo(@TypeOf(sink)).pointer.child;
            // The first iterate neither replays nor records (Meyer models
            // branch on it), so a later iterate's recording reaches the next
            // timepoint.
            if (sim.iteration <= 1) return self.evalSplit(sink, first, last, sim, limiting);
            // Which planes the recorded pointers address.
            var key = std.hash.Wyhash.init(@intFromBool(staged));
            key.update(std.mem.asBytes(&[_]usize{ @intFromPtr(pl.g_vals.ptr), @intFromPtr(pl.c_vals.ptr), @intFromPtr(pl.rhs.ptr), @intFromPtr(pl.q_vec.ptr) }));
            const plane_key = key.final();
            const class = classOf(sim);
            const lim_writes = comptime if (has_limit) layout.limitWritesOf(D) else 0;
            var hits: usize = 0;
            var misses: usize = 0;
            defer {
                _ = bp.hits.fetchAdd(hits, .monotonic);
                _ = bp.misses.fetchAdd(misses, .monotonic);
            }
            var id = first;
            while (id < last) : (id += 1) {
                var lx: [n_u]f64 = undefined;
                var corr = false;
                inline for (0..n_u) |u| {
                    const xg = sink.x(sink.gath(id, u));
                    lx[u] = xg;
                    if (comptime layout.has(lim_writes, u)) if (limiting) {
                        const l = sink.lim(id, u);
                        lx[u] = l;
                        if (@as(u64, @bitCast(xg)) != @as(u64, @bitCast(l))) corr = true;
                    };
                }
                const m = &bp.meta[id];
                const ref = bp.ref[@as(usize, id) * n_u ..][0..n_u];
                const inst_now = std.mem.asBytes(&self.instances[id]);
                const inst_rec = bp.inst[@as(usize, id) * @sizeOf(D.Instance) ..][0..@sizeOf(D.Instance)];
                if (!corr and m.class == class and m.plane == plane_key and std.mem.eql(u8, inst_now, inst_rec) and
                    replay(bp, id, m.*, &lx, ref, SinkT.stores))
                {
                    // Charges and the tape are re-read exactly (values only):
                    // a Meyer model's C is not dq/dx, and the transient
                    // multiplies a charge error by 2/dt.
                    if (comptime has_q) stamp.evalQRange(D, dual.RealFor(@hasDecl(D, "collapse")), sink, id, id + 1, sim, null);
                    hits += 1;
                    continue;
                }
                misses += 1;
                // Full evaluation, recorded, keyed on the bytes it read.
                @memcpy(inst_rec, inst_now);
                var rec: RecordSink(SinkT, rec_cap) = .{
                    .inner = sink,
                    .id = id,
                    .cells = bp.cells[@as(usize, id) * cells_len ..][0..cells_len],
                    .vals = bp.vals[@as(usize, id) * cells_len ..][0..cells_len],
                    .rows = bp.rows[@as(usize, id) * rows_len ..][0..rows_len],
                    .cols = bp.cols[@as(usize, id) * rec_cap.jac ..][0..rec_cap.jac],
                };
                self.evalSplit(&rec, id, id + 1, sim, limiting);
                @memcpy(ref, &lx);
                m.* = if (corr or rec.overflow) .{} else .{
                    .class = class,
                    .n_res = @intCast(rec.n_res),
                    .n_jac = @intCast(rec.n_jac),
                    .n_qjac = @intCast(rec.n_qjac),
                    .plane = plane_key,
                };
            }
            assert(id == last);
        }

        // ---- private ----

        fn create(count: u32, tol: abi.BypassTol) !*Bp {
            const bp = try staging_gpa.create(Bp);
            errdefer staging_gpa.destroy(bp);
            const ref = try staging_gpa.alloc(f64, count * n_u);
            errdefer staging_gpa.free(ref);
            const meta = try staging_gpa.alloc(Meta, count);
            errdefer staging_gpa.free(meta);
            const inst = try staging_gpa.alloc(u8, count * @sizeOf(D.Instance));
            errdefer staging_gpa.free(inst);
            const cells = try staging_gpa.alloc(*f64, count * cells_len);
            errdefer staging_gpa.free(cells);
            const vals = try staging_gpa.alloc(f64, count * cells_len);
            errdefer staging_gpa.free(vals);
            const rows = try staging_gpa.alloc(u8, count * rows_len);
            errdefer staging_gpa.free(rows);
            const cols = try staging_gpa.alloc(u8, count * rec_cap.jac);
            @memset(meta, .{});
            bp.* = .{ .tol = tol, .ref = ref, .meta = meta, .inst = inst, .cells = cells, .vals = vals, .rows = rows, .cols = cols };
            assert(bp.meta.len == count);
            return bp;
        }

        /// The iteration class a recording is keyed on: the analysis kind and
        /// the flags a model branches on. Never 0, the empty `Meta`'s class.
        fn classOf(sim: SimState) u16 {
            const class: u16 = 1 + 2 * @as(u16, @intCast(@backingInt(sim.kind))) +
                64 * @as(u16, @intFromBool(sim.analog_initial)) +
                128 * @as(u16, @intFromBool(sim.initial_step)) +
                256 * @as(u16, @intFromBool(sim.final_step)) +
                512 * @as(u16, @intFromBool(sim.t > 0));
            assert(class & 1 == 1);
            return class;
        }

        /// Replays instance `id`'s recording `m` at `lx` in residual form,
        /// f(x) ~ f(ref) + J (x - ref), when every unknown and every
        /// predicted residual change is within tolerance; false, writing
        /// nothing, otherwise. `stores`: the cells are staging positions.
        fn replay(bp: *const Bp, id: u32, m: Meta, lx: *const [n_u]f64, ref: *const [n_u]f64, comptime stores: bool) bool {
            assert(m.class != 0);
            assert(m.n_res <= rec_cap.res);
            assert(m.n_jac <= rec_cap.jac);
            assert(m.n_qjac <= rec_cap.qjac);
            var d: [n_u]f64 = undefined;
            inline for (0..n_u) |u| {
                const tol = bp.tol.reltol * @max(@abs(lx[u]), @abs(ref[u])) +
                    (if (comptime D.u_kinds[u] == .voltage) bp.tol.vntol else bp.tol.abstol);
                d[u] = lx[u] - ref[u];
                // An infinite `d` passes an infinite `tol`; the dense
                // product this replaced turned it into a NaN and refused.
                if (!(@abs(d[u]) <= tol) or std.math.isInf(d[u])) return false;
            }
            const cells = bp.cells[@as(usize, id) * cells_len ..][0..cells_len];
            const vals = bp.vals[@as(usize, id) * cells_len ..][0..cells_len];
            const rows = bp.rows[@as(usize, id) * rows_len ..][0..rows_len];
            const cols = bp.cols[@as(usize, id) * rec_cap.jac ..][0..rec_cap.jac];
            // J (x - ref) per local row, each row's columns in ascending
            // order from +0.0. A pattern entry no write fills would add a ±0
            // product, which leaves a sum that is never -0 unchanged.
            var jd: [n_u]f64 = @splat(0);
            const js = rec_cap.res;
            for (cols[0..m.n_jac], rows[js..][0..m.n_jac], vals[js..][0..m.n_jac]) |c, r, v| jd[r] += v * d[c];
            // SPICE's second test: the predicted change itself within
            // tolerance (an exponential junction moves far more current than
            // a vntol-sized step suggests).
            for (rows[0..m.n_res], vals[0..m.n_res]) |r, v| {
                if (!(@abs(jd[r]) <= bp.tol.reltol * @abs(v) + bp.tol.abstol)) return false;
            }
            for (cells[0..m.n_res], rows[0..m.n_res], vals[0..m.n_res]) |p, r, v| put(p, v + jd[r], stores);
            for (cells[js..][0..m.n_jac], vals[js..][0..m.n_jac]) |p, v| put(p, v, stores);
            const qs = rows_len;
            for (cells[qs..][0..m.n_qjac], vals[qs..][0..m.n_qjac]) |p, v| put(p, v, stores);
            return true;
        }

        inline fn put(p: *f64, v: f64, comptime stores: bool) void {
            if (stores) p.* = v else p.* += v;
        }
    };
}
