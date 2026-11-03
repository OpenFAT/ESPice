//! Device operating-point variables (`@m1[gm]`, #236). In: the deck's
//! `.save`, `.probe` and `.meas` names and the frozen circuit. Out: each
//! variable resolved to its instance (`core.OpVar`), and the probe rows
//! that record it.
//! Cites: #236; HSPICE Reference Manual: Elements and Device Models,
//! "MOSFET Output Templates"; VerA `op_vars`.
//!
//! Contents:
//! - `pub const OpVars`
//! - `pub fn opVars(sim_arena: Allocator, scratch: Allocator, lib: *const device.Library, circuit: *const device.Circuit, cards: []const requests.CardRef, probes: []const u32, labels: []const []const u8, nl: *const netlist.Netlist, queries: []const requests.Query) !OpVars`

const std = @import("std");
const Allocator = std.mem.Allocator;
const core = @import("core");
const requests = core.query;
const netlist = @import("netlist");
const device = @import("device");

/// The variables, and the probe rows that record them: the deck's own
/// `probes` and `labels`, then each named instance's rows, unlabeled.
pub const OpVars = struct { vars: []const core.OpVar, probes: []const u32, labels: []const []const u8 };

/// `core.OpVar.Quantity`'s names, for the error that lists them.
const opvar_names = blk: {
    var s: []const u8 = "";
    for (@typeInfo(core.OpVar.Quantity).@"enum".field_names) |n| s = s ++ " " ++ n;
    break :blk s[1..];
};

/// HSPICE's MOSFET element templates [HSPICE Reference Manual: Elements
/// and Device Models, "MOSFET Output Templates"]: `lx<N>`/`lv<N>` to the
/// variable it reads. Another number refuses.
// ponytail: lx19/lx20 (HSPICE's ∂Qg/∂Vd and ∂Qg/∂Vs, -cgd and -cgs) need a
// sign this table cannot carry; add one if a deck reads them.
const hspice_templates = std.StaticStringMap([]const u8).initComptime(.{
    .{ "lx1", "vbs" }, .{ "lx2", "vgs" },    .{ "lx3", "vds" },  .{ "lx4", "id" },
    .{ "lx7", "gm" },  .{ "lx8", "gds" },    .{ "lx9", "gmbs" }, .{ "lx18", "cgg" },
    .{ "lv9", "vth" }, .{ "lv10", "vdsat" },
});

/// A device operating-point variable's name, split: ngspice
/// `@<card>[<variable>]`, Spectre `<device>:<variable>`, or HSPICE
/// `lx<N>(<device>)`/`lv<N>(<device>)` through `hspice_templates`. A
/// `<device>` (`path`) is the deck's own name (`x1.mn1`), which the card's
/// flattened name leads with its letter (`m.x1.mn1`).
const OpVarName = struct {
    device: []const u8,
    variable: []const u8,
    path: bool,

    /// Null for a name of none of these forms.
    fn of(name: []const u8) ?OpVarName {
        if (name.len != 0 and name[0] == '@') {
            const open = std.mem.indexOfScalar(u8, name, '[') orelse return null;
            if (open < 2 or name[name.len - 1] != ']') return null;
            return .{ .device = name[1..open], .variable = name[open + 1 .. name.len - 1], .path = false };
        }
        if (std.mem.indexOfScalar(u8, name, ':')) |colon| {
            if (colon == 0 or colon + 1 == name.len) return null;
            return .{ .device = name[0..colon], .variable = name[colon + 1 ..], .path = true };
        }
        const open = std.mem.indexOfScalar(u8, name, '(') orelse return null;
        if (open + 2 >= name.len or name[name.len - 1] != ')') return null;
        const variable = hspice_templates.get(name[0..open]) orelse return null;
        return .{ .device = name[open + 1 .. name.len - 1], .variable = variable, .path = true };
    }

    /// Whether `card`, a flattened card name, is the device `n` names.
    fn matches(n: OpVarName, card: []const u8) bool {
        if (std.ascii.eqlIgnoreCase(card, n.device)) return true;
        return n.path and card.len == n.device.len + 2 and card[1] == '.' and std.ascii.eqlIgnoreCase(card[2..], n.device);
    }
};

/// The device operating-point variables (`@m1[gm]`, #236) the deck's
/// `.save`, `.probe` and `.meas` cards name, and what records them:
/// `probes` and `labels`, then each named instance's rows, unlabeled. A
/// variable is the model's own (VerA `op_vars`) or, for a MOSFET, a
/// terminal quantity (`core.OpVar.Quantity`). A name that is neither, or no
/// `OpVarName` of a device in the deck, is `UnsupportedCard`, logged with
/// the variables there are. So is one under a `.dc` that sweeps the
/// temperature or a device of its type: the instance would be evaluated at
/// the sweep's last state. HSPICE's `.probe op` adds every variable of
/// every device (`probeAll`).
// ponytail: `.probe op` is a column per variable per device at every point
// of every .op, .dc and .tran; narrow it to the .op if a large circuit's
// transient runs out of memory.
pub fn opVars(
    sim_arena: Allocator,
    scratch: Allocator,
    lib: *const device.Library,
    circuit: *const device.Circuit,
    cards: []const requests.CardRef,
    probes: []const u32,
    labels: []const []const u8,
    nl: *const netlist.Netlist,
    queries: []const requests.Query,
) !OpVars {
    const dialect = nl.deck.dialect;
    var names: std.StringArrayHashMapUnmanaged(void) = .empty;
    for (nl.deck.opvar_saves) |s| try names.put(scratch, s, {});
    for (nl.deck.measures) |m| for ([_]core.MeasureClause{ m.first, m.second }) |c| {
        for ([_][]const u8{ c.vec, c.vec2 }) |v| if (netlist.isOpVar(dialect, v)) try names.put(scratch, v, {});
        for ([_][]const core.MeasureOp{ c.ops, c.ops2 }) |ops| for (ops) |op| if (op == .vector and netlist.isOpVar(dialect, op.vector)) {
            try names.put(scratch, op.vector, {});
        };
    };
    if (names.count() == 0 and !nl.deck.probe_op) return .{ .vars = &.{}, .probes = &.{}, .labels = &.{} };
    // The terminal quantities are a MOSFET's (d, g, s, b), whose letter a
    // Spectre card's name need not carry; each maps to whether its card
    // is p-type, as `.mosra` reads it.
    // ponytail: the card kind only; a `type=-1` on an nmos or Verilog-A
    // module card reads as n-type. Read the bound model's polarity if a
    // deck spells it that way.
    var mosfets: std.StringHashMapUnmanaged(bool) = .empty;
    for (nl.bucket('m')) |e| {
        const d = nl.device(e);
        try mosfets.put(scratch, d.name, if (d.model) |m| std.ascii.eqlIgnoreCase(m.kind, "pmos") else false);
    }
    var list: std.ArrayList(core.OpVar) = .empty;
    for (names.keys()) |name| try list.append(sim_arena, try opVar(sim_arena, lib, circuit, cards, &mosfets, name));
    if (nl.deck.probe_op) for (cards) |ref| try probeAll(sim_arena, lib, circuit, ref, mosfets.get(ref.name), &names, &list);
    const vars = list.items;
    var rows: std.ArrayList(u32) = .empty;
    try rows.appendSlice(sim_arena, probes);
    var row_labels: std.ArrayList([]const u8) = .empty;
    try row_labels.appendSlice(sim_arena, labels);
    for (vars) |v| {
        const b = circuit.batches[v.batch];
        for (b.hooks.scatter_tapes(b.ctx).rows[@as(usize, v.index) * b.n_u ..][0..b.n_u]) |row| {
            // A ground unknown's row is `n`.
            if (row >= circuit.n or std.mem.indexOfScalar(u32, rows.items[probes.len..], row) != null) continue;
            try rows.append(sim_arena, row);
            try row_labels.append(sim_arena, "");
        }
    }
    for (queries) |q| if (q == .dc) for ([_]?requests.Dc.SweepTarget{ q.dc.target, q.dc.target2 }) |target| {
        const swept = target orelse continue;
        for (vars) |v| if (swept == .temp or swept.device.type == circuit.batch_types[v.batch]) {
            if (!@import("builtin").is_test) std.log.err("netlist: {s} along a .dc that sweeps the temperature or a device of its type is not supported (#236)", .{v.label});
            return error.UnsupportedCard;
        };
    };
    return .{ .vars = vars, .probes = rows.items, .labels = row_labels.items };
}

// ---- private ----

/// `name`, an `OpVarName`, resolved to its instance; `mosfets` maps the
/// MOSFETs' card names to whether each is p-type. See `opVars` for the
/// refusals.
fn opVar(arena: Allocator, lib: *const device.Library, circuit: *const device.Circuit, cards: []const requests.CardRef, mosfets: *const std.StringHashMapUnmanaged(bool), name: []const u8) !core.OpVar {
    const split = OpVarName.of(name) orelse {
        if (!@import("builtin").is_test) std.log.err("netlist: {s}: a device operating-point variable is @<device>[<variable>], <device>:<variable>, or lx1-lx4, lx7-lx9, lx18, lv9 or lv10 (<device>)", .{name});
        return error.UnsupportedCard;
    };
    const ref = for (cards) |c| {
        if (split.matches(c.name)) break c;
    } else {
        if (!@import("builtin").is_test) std.log.err("netlist: {s}: no device {s}", .{ name, split.device });
        return error.UnsupportedCard;
    };
    return opVarOf(arena, lib, circuit, ref, mosfets.get(ref.name), name, split.variable);
}

/// Appends to `list` every operating-point variable of instance `ref`
/// that `names` does not hold, as `@<card>[<variable>]` (HSPICE `.probe
/// op`): the model's own, then for a MOSFET (`pmos_card`, as `opVarOf`
/// reads it) the terminal quantities it does not shadow, without the
/// alias `ids` and, with three terminals, the bulk ones.
fn probeAll(arena: Allocator, lib: *const device.Library, circuit: *const device.Circuit, ref: requests.CardRef, pmos_card: ?bool, names: *const std.StringArrayHashMapUnmanaged(void), list: *std.ArrayList(core.OpVar)) !void {
    const hooks = circuit.batches[std.mem.indexOfScalar(core.DeviceType, circuit.batch_types, ref.type).?].hooks;
    if (hooks.local_eval == null) return;
    const ports = lib.vtable(ref.type).num_ports;
    const bulk = [_]core.OpVar.Quantity{ .ib, .gmbs, .cgb, .vbs };
    var all: std.ArrayList([]const u8) = .empty;
    try all.appendSlice(arena, hooks.op_vars);
    if (pmos_card != null and ports >= 3) for (std.enums.values(core.OpVar.Quantity)) |q| {
        if (q == .ids or (ports == 3 and std.mem.indexOfScalar(core.OpVar.Quantity, &bulk, q) != null)) continue;
        for (hooks.op_vars) |own| {
            if (std.ascii.eqlIgnoreCase(own, @tagName(q))) break;
        } else try all.append(arena, @tagName(q));
    };
    for (all.items) |variable| {
        const label = try arena.print("@{s}[{s}]", .{ ref.name, variable });
        if (names.contains(std.ascii.lowerString(label, label))) continue;
        try list.append(arena, try opVarOf(arena, lib, circuit, ref, pmos_card, label, variable));
    }
}

/// `variable` of instance `ref`, published as `label`; `pmos_card` is null
/// unless its card is a MOSFET's, and then whether it is p-type.
fn opVarOf(arena: Allocator, lib: *const device.Library, circuit: *const device.Circuit, ref: requests.CardRef, pmos_card: ?bool, label: []const u8, variable: []const u8) !core.OpVar {
    const batch = std.mem.indexOfScalar(core.DeviceType, circuit.batch_types, ref.type).?;
    const hooks = circuit.batches[batch].hooks;
    const ports = lib.vtable(ref.type).num_ports;
    const mosfet = pmos_card != null and ports >= 3;
    if (hooks.local_eval == null) return refuseOpVar(arena, label, ref.name, variable, &.{}, false);
    var v: core.OpVar = .{ .label = try arena.dupe(u8, label), .batch = @intCast(batch), .index = ref.index, .ports = ports, .pmos = pmos_card orelse false };
    // The model's own variable first: BSIM-CMG's `ids` is its own.
    for (hooks.op_vars, 0..) |own, k| {
        if (std.ascii.eqlIgnoreCase(own, variable)) {
            v.own = @intCast(k);
            break;
        }
    } else {
        const terminal = if (mosfet) std.meta.stringToEnum(core.OpVar.Quantity, variable) else null;
        v.quantity = terminal orelse return refuseOpVar(arena, label, ref.name, variable, hooks.op_vars, mosfet);
    }
    return v;
}

/// Logs that `card` has no variable `variable`, listing the model's own
/// variables `own` and, for a MOSFET, the terminal ones; `UnsupportedCard`.
fn refuseOpVar(arena: Allocator, name: []const u8, card: []const u8, variable: []const u8, own: []const []const u8, mosfet: bool) error{ OutOfMemory, UnsupportedCard } {
    if (!@import("builtin").is_test) {
        const listed = std.mem.join(arena, " ", own) catch return error.OutOfMemory;
        const sep = if (listed.len != 0 and mosfet) " " else "";
        std.log.err("netlist: {s}: {s} has no variable '{s}'; its variables are: {s}{s}{s}", .{ name, card, variable, listed, sep, if (mosfet) opvar_names else "" });
    }
    return error.UnsupportedCard;
}
