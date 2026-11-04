//! Golden test placeholder (pre-change recording).

const std = @import("std");
const testing = std.testing;

// ---- private ----

const Hash = std.hash.Wyhash;

/// Hashes `vals` bit by bit, every NaN as one canonical quiet NaN: the
/// payload a NaN carries is the backend's, not the algorithm's.
fn hashF64(h: *Hash, vals: []const f64) void {
    for (vals) |v| {
        const bits: u64 = if (v != v) 0x7ff8_0000_0000_0000 else @bitCast(v);
        h.update(std.mem.asBytes(&bits));
    }
}

fn hashV(comptime V: type, h: *Hash, vals: []const V) void {
    hashF64(h, @as([*]const f64, @ptrCast(vals.ptr))[0 .. vals.len * @sizeOf(V) / 8]);
}

/// Lane l of `vals`: scaled 1 +- 5%, one entry redrawn, one column zeroed,
/// or one entry NaN or -inf, by `r`.
fn perturb(comptime W: usize, r: std.Random, col_ptr: []const u32, base: []const f64, out: [][W]f64) void {
    const n = col_ptr.len - 1;
    for (out, base) |*v, a| v.* = @splat(a);
    for (0..W) |l| switch (r.uintLessThan(u8, 8)) {
        0 => out[r.uintLessThan(usize, out.len)][l] = 4 * r.float(f64) - 2,
        1 => {
            const c = r.uintLessThan(usize, n);
            for (col_ptr[c]..col_ptr[c + 1]) |p| out[p][l] = 0;
        },
        2 => out[r.uintLessThan(usize, out.len)][l] = std.math.nan(f64),
        3 => out[r.uintLessThan(usize, out.len)][l] = -std.math.inf(f64),
        else => for (out, base) |*v, a| {
            v[l] = a * (0.95 + 0.1 * r.float(f64));
        },
    };
}

/// Every output bit of `LaneLu(W)`, `ComplexLaneLu(W)` and the lane-major
/// GPU bodies run on the host (`Bodies(HostP)` of lane.zig and
/// frequency.zig) over random systems, some with fabricated pivots.
fn golden(comptime W: usize, hs: *[4]Hash) !usize {
    const gpa = testing.allocator;
    const direct = @import("../direct.zig");
    const SparseLu = @import("../sparse_lu.zig").SparseLu;
    const LaneLu = @import("../lane/lu.zig").LaneLu(W);
    const ComplexLaneLu = @import("../lane/clu.zig").ComplexLaneLu(W);
    const lk = @import("lane.zig");
    const B = lk.Bodies(lk.HostP);
    const FB = @import("frequency.zig").Bodies(lk.HostP);
    const V = LaneLu.V;
    var prng = std.Random.DefaultPrng.init(0x601d + W);
    const r = prng.random();
    var voids: usize = 0;
    for (0..48) |trial| {
        const n: u32 = @intCast(2 + trial % 23);
        const sys = try SparseLu.test_access.TestSystem.init(gpa, r, n, 0.3);
        defer sys.deinit(gpa);
        const nnz = sys.vals.len;
        // A zeroed column in the lead's values leaves a fabricated pivot.
        if (trial % 5 == 0) {
            const c = r.uintLessThan(usize, n);
            for (sys.col_ptr[c]..sys.col_ptr[c + 1]) |p| sys.vals[p] = 0;
        }
        var lead = try direct.Solver.init(gpa, n, sys.col_ptr, sys.row_idx, null, .{});
        defer lead.deinit();
        lead.factor(sys.vals, .{}) catch continue;
        const lu = lead.sparse() orelse continue;
        if (std.mem.indexOfScalar(bool, lu.void_col, true) != null) voids += 1;

        const vals = try gpa.alloc(V, 2 * nnz);
        defer gpa.free(vals);
        const arr = @as([*][W]f64, @ptrCast(vals.ptr))[0 .. 2 * nnz];
        perturb(W, r, sys.col_ptr, sys.vals, arr[0..nnz]);
        perturb(W, r, sys.col_ptr, sys.vals, arr[nnz..]);
        const re: []const V = vals[0..nnz];
        const im: []const V = vals[nnz..];
        const rhs = try gpa.alloc(V, 2 * n);
        defer gpa.free(rhs);
        for (rhs) |*v| {
            var a: [W]f64 = undefined;
            for (&a, 0..) |*e, l| e.* = if (l % 3 == 2) 0 else 2 * r.float(f64) - 1;
            v.* = a;
        }
        const out = try gpa.alloc(V, 4 * n);
        defer gpa.free(out);

        for ([_]f64{ lead.params.refactor_growth_limit, 0 }) |growth| {
            var h = &hs[0];
            var ll = try LaneLu.init(gpa, lu);
            defer ll.deinit(gpa);
            for ([_][]V{ ll.lx, ll.ux, ll.udiag }) |p| @memset(p, @splat(0));
            const mask = ll.refactor(LaneLu.Plane{ .col_ptr = sys.col_ptr, .vals = re }, growth);
            h.update(std.mem.asBytes(&mask));
            for ([_][]const V{ ll.lx, ll.ux, ll.udiag }) |p| hashV(V, h, p);
            ll.solve(rhs[0..n], out[0..n]);
            ll.solveT(rhs[0..n], out[n .. 2 * n]);
            hashV(V, h, out[0 .. 2 * n]);

            h = &hs[1];
            var cl = try ComplexLaneLu.init(gpa, lu);
            defer cl.deinit(gpa);
            for ([_][]V{ cl.lr, cl.li, cl.ur, cl.ui, cl.dr, cl.di }) |p| @memset(p, @splat(0));
            const cmask = cl.refactor(ComplexLaneLu.Plane{ .col_ptr = sys.col_ptr, .re = re, .im = im }, growth);
            h.update(std.mem.asBytes(&cmask));
            for ([_][]const V{ cl.lr, cl.li, cl.ur, cl.ui, cl.dr, cl.di }) |p| hashV(V, h, p);
            cl.solve(rhs[0..n], rhs[n..], out[0..n], out[n .. 2 * n]);
            cl.solveH(rhs[0..n], rhs[n..], out[2 * n .. 3 * n], out[3 * n ..]);
            hashV(V, h, out);
        }

        // The GPU bodies, W lanes lane-major (element i of lane l at i * W + l).
        const N = W;
        const built = try lk.buildTab(gpa, lu, sys.col_ptr, lead.params.refactor_growth_limit, direct.pivot_tol);
        defer gpa.free(built.idx);
        const t = built.tab;
        const x = built.idx.ptr;
        const a: [*]f64 = @ptrCast(vals.ptr);
        const nl: usize = @max(t.nnz_l, 1);
        const nu: usize = @max(t.nnz_u, 1);
        const buf = try gpa.alloc(f64, (2 * nl + 2 * nu + 2 * 2 * n + 3 * 2 * n) * N);
        defer gpa.free(buf);
        @memset(buf, 0);
        const lx = buf[0 .. 2 * nl * N];
        const ux = buf[2 * nl * N ..][0 .. 2 * nu * N];
        const d = ux.ptr + 2 * nu * N;
        const w = d + 2 * n * N;
        const y = w + 2 * n * N;
        const o = y + 2 * n * N;
        const o2 = o + 2 * n * N;
        const b: [*]f64 = @ptrCast(rhs.ptr);
        var h = &hs[2];
        for (0..N) |l| {
            const fail = B.refactor(t, x, a, lx.ptr, ux.ptr, d, w, N, @intCast(l));
            h.update(std.mem.asBytes(&fail));
            if (!fail) {
                B.solveNeg(t, x, lx.ptr, ux.ptr, d, b, y, o, N, @intCast(l));
                FB.solve(t, x, lx.ptr, ux.ptr, d, b, y, o2, N, l, true);
            }
        }
        hashF64(h, buf);
        @memset(buf, 0);
        // The complex bodies read `a` as 2 nnz rows: re, then im.
        h = &hs[3];
        for (0..N) |l| {
            const fail = FB.complexRefactor(t, x, a, lx.ptr, ux.ptr, d, w, N, @intCast(l));
            h.update(std.mem.asBytes(&fail));
            if (!fail) {
                FB.complexSolve(t, x, lx.ptr, ux.ptr, d, b, y, o, N, l, false);
                FB.complexSolve(t, x, lx.ptr, ux.ptr, d, b, y, o2, N, l, true);
            }
        }
        hashF64(h, buf);
    }
    return voids;
}

test "lane LU bodies keep their recorded output bits on LaneLu, ComplexLaneLu and the GPU lane bodies, W = 1, 4, 8, NaN lanes and fabricated pivots" {
    // Per W: LaneLu, ComplexLaneLu, the real GPU bodies, the complex ones.
    const want = [3][4]u64{ .{ 0, 0, 0, 0 }, .{ 0, 0, 0, 0 }, .{ 0, 0, 0, 0 } };
    var got: [3][4]u64 = undefined;
    inline for (.{ 1, 4, 8 }, 0..) |W, i| {
        // Twice, with the freed heap scribbled between: a hash that reads
        // memory nobody wrote differs.
        var again: [4]u64 = undefined;
        for (0..2) |pass| {
            var hs: [4]Hash = @splat(Hash.init(0));
            const voids = try golden(W, &hs);
            try testing.expect(voids > 0);
            for (&got[i], &hs) |*g, *h| g.* = h.final();
            if (pass == 0) {
                again = got[i];
                const junk = try testing.allocator.alloc(u8, 1 << 20);
                @memset(junk, 0x5c);
                testing.allocator.free(junk);
            }
        }
        std.debug.print("golden W={d}: 0x{x} 0x{x} 0x{x} 0x{x} (first pass 0x{x} 0x{x} 0x{x} 0x{x})\n", .{ W, got[i][0], got[i][1], got[i][2], got[i][3], again[0], again[1], again[2], again[3] });
        try testing.expectEqualSlices(u64, &again, &got[i]);
    }
    for (want, got) |w, g| try testing.expectEqualSlices(u64, &w, &g);
}
