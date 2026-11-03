//! Device cards read once, then replayed: a subcircuit body card's fields as
//! its first instance read them (#229), and the top-level cards scanned on
//! several threads before they are read in line order (#230). In: a card's
//! line. Out: its nodes, model word and value fields (`Replay`), which
//! `readDevice` reads under each frame as a whole read would.
//! Cites: #229, #230.
//!
//! Contents:
//! - `pub fn Methods(comptime S: type) type`
//! - `pub fn replay(r: *R, rp: *const Replay, line: []const u8, head: []const u8, letter: u8, frame: *const Frame) Error!void`
//! - `pub fn cardNodes(r: *const R, a: Allocator, f: *F, letter: u8, port_card: bool, nodes: *std.ArrayList([]const u8)) Error!?[]const u8`
//! - `pub fn readFields(comptime pure: bool, r: if (pure) *const R else *R, a: Allocator, line: []const u8, letter: u8, f: *F, frame: if (pure) void else *const Frame, fields: ?*std.ArrayList(ReplayField), kv_text: if (pure) void else *std.ArrayList([]const u8)) (if (pure) error{ OutOfMemory, Fallback } else Error)!void`
//! - `pub fn scan(r: *const R, a: Allocator, row: usize, nodes: *std.ArrayList([]const u8), fields: *std.ArrayList(ReplayField)) ?*const Replay`
//! - `pub fn scanCards(r: *R, rows: []const u32, opts: ParseOptions, arenas: *[scan_threads_max]std.heap.ArenaAllocator) Error!usize`
//! - `pub fn scanRun(r: *const R, s: *std.heap.ArenaAllocator, part: []const u32, out: []?*const Replay) void`

const std = @import("std");
const Allocator = std.mem.Allocator;
const core = @import("core");
const reader = @import("reader.zig");
const card_rows = @import("card_rows.zig");
const card_table = @import("card_table.zig");
const reader_frame = @import("reader_frame.zig");
const Error = card_rows.Error;
const Value = card_rows.Value;
const Frame = reader_frame.Frame;
const Replay = reader_frame.Replay;
const ReplayField = reader_frame.ReplayField;
const ParseOptions = reader_frame.ParseOptions;
const scan_threads_max = reader_frame.scan_threads_max;
const bTransformationWanted = card_table.bTransformationWanted;

/// The stage's `Reader` methods over dialect syntax `S`; `Reader(S)`
/// aliases each one as its own decl.
pub fn Methods(comptime S: type) type {
    const F = S.Split;
    return struct {
        const R = reader.Reader(S);
        const fixedPins = R.fixedPins;
        const mixedPort = R.mixedPort;

        /// Reads card `line` from the `Replay` its first read or `scan` kept: the
        /// nodes and literals reused, every other field re-read under `frame`
        /// in card order, then split and committed as `readDevice` does.
        pub fn replay(r: *R, rp: *const Replay, line: []const u8, head: []const u8, letter: u8, frame: *const Frame) Error!void {
            const arena = r.arena;
            r.nodes.clearRetainingCapacity();
            r.positional.clearRetainingCapacity();
            r.card_kv.clearRetainingCapacity();
            try r.nodes.appendSlice(arena, rp.nodes);
            if (rp.model) |m| try r.positional.append(arena, try r.nameValue(m, frame, false));
            var f = F.init(line);
            for (rp.fields) |field| {
                f.pos = field.pos;
                const value: Value = if (field.num) |n| .{ .num = n } else if (field.keyed) try r.kvValue(&f, frame, false) else try r.valueAt(field.word, &f, frame, true, false);
                if (field.keyed) try r.card_kv.append(arena, .{ .key = field.word, .value = value }) else try r.positional.append(arena, value);
            }
            if (letter == 'q' or letter == 'm' or letter == 'd') try r.splitTerminals();
            return r.commit(head, letter, frame);
        }

        /// A card's nodes into `nodes`, `f` just past its head, and the
        /// model or subcircuit word after them: the `(`-listed nodes and
        /// the word after the `)`, the fixed pins and null, or the words
        /// before the first value. Reads only the line (#230).
        pub fn cardNodes(r: *const R, a: Allocator, f: *F, letter: u8, port_card: bool, nodes: *std.ArrayList([]const u8)) Error!?[]const u8 {
            if ((f.nextByte() orelse return error.ParseError) == '(') {
                _ = f.next();
                while (true) {
                    const t = f.next() orelse return error.ParseError;
                    if (t[0] == ')') break;
                    if (t[0] == ',') continue;
                    if (!F.isWord(t)) return error.ParseError;
                    try nodes.append(a, t);
                }
                const model = f.next() orelse return error.ParseError;
                if (!F.isWord(model)) return error.ParseError;
                return model;
            }
            if (fixedPins(letter)) |fixed| {
                const count = if (port_card and mixedPort(f.*)) 3 else fixed;
                try nodes.ensureUnusedCapacity(a, count);
                for (0..count) |_| {
                    const t = f.next() orelse return error.ParseError;
                    if (!F.isWord(t)) return error.ParseError;
                    nodes.appendAssumeCapacity(t);
                }
                return null;
            }
            // Variable-terminal cards end their word list with the model/subcircuit name.
            while (true) {
                const save = f.pos;
                const t = f.next() orelse break;
                if (!F.isWord(t) or f.takeEq()) {
                    f.pos = save;
                    break;
                }
                try nodes.append(a, t);
            }
            // ngspice reads `X1 a b sub params: p=1` (numparam's subcircuit call).
            if (letter == 'x' and r.dialect == .ngspice and nodes.items.len > 0 and std.mem.eql(u8, nodes.items[nodes.items.len - 1], "params:")) _ = nodes.pop();
            return nodes.pop() orelse return error.ParseError;
        }

        /// The value fields after a card's nodes and model, in card order,
        /// each noted in `fields` when it is not null. Not `pure`, each value
        /// is read into `positional` or `card_kv` as it comes, and an X
        /// card's value texts into `kv_text`. `pure` reads only the line,
        /// so threads may share `r` (#230): a literal is kept in `fields`,
        /// and a field whose read would fold, allocate a group or refuse is
        /// `error.Fallback`, for `readDevice` to read the card whole.
        pub fn readFields(
            comptime pure: bool,
            r: if (pure) *const R else *R,
            a: Allocator,
            line: []const u8,
            letter: u8,
            f: *F,
            frame: if (pure) void else *const Frame,
            fields: ?*std.ArrayList(ReplayField),
            kv_text: if (pure) void else *std.ArrayList([]const u8),
        ) (if (pure) error{ OutOfMemory, Fallback } else Error)!void {
            while (f.next()) |t| {
                if (t[0] == ',') continue;
                if (F.isWord(t) and f.takeEq()) {
                    const start = f.pos;
                    if (pure) {
                        const n = (r.literalValue(f) catch return error.Fallback) orelse return error.Fallback;
                        try fields.?.append(a, .{ .word = t, .keyed = true, .num = n, .pos = start });
                    } else {
                        try r.card_kv.append(a, .{ .key = t, .value = try r.kvValue(f, frame, false) });
                        if (letter == 'x') try kv_text.append(a, std.mem.trim(u8, f.line[start..f.pos], " \t"));
                        if (fields) |list| {
                            // `kvValue` read it as a literal exactly when this does.
                            var probe = f.*;
                            probe.pos = start;
                            try list.append(a, .{ .word = t, .keyed = true, .num = try r.literalValue(&probe), .pos = start });
                        }
                    }
                } else {
                    const at = f.pos;
                    if (pure) {
                        // `valueAt` reads a word as a number or a name, and
                        // a brace or quote as an expression `replay` folds;
                        // a `name(` group allocates.
                        const bare = F.isWord(t);
                        if (r.operatorLed(t) or (bare and f.nextByte() == '(') or (!bare and t[0] != '{' and !F.isQuote(t[0]))) return error.Fallback;
                        try fields.?.append(a, .{ .word = t, .keyed = false, .num = if (bare) S.parseNum(t) else null, .pos = at });
                    } else {
                        if (letter != 'x') try r.refuseOperatorField(line, t);
                        try r.positional.append(a, try r.valueAt(t, f, frame, true, false));
                        if (fields) |list| try list.append(a, .{ .word = t, .keyed = false, .num = null, .pos = at });
                    }
                }
            }
        }

        /// Top-level card `row` read as far as its text alone decides: the
        /// `Replay` that `readDevice` reads it from in line order, to the
        /// result a whole read gives. Null for a card whose read takes
        /// another path, would refuse it, or holds a field `readFields`
        /// cannot scan; `readDevice` reads that one whole.
        /// Reads only the dialect, the lines and their letters, fixed before
        /// any device is read, so threads may share `r` (#230). `nodes`
        /// and `fields` are scratch.
        pub fn scan(r: *const R, a: Allocator, row: usize, nodes: *std.ArrayList([]const u8), fields: *std.ArrayList(ReplayField)) ?*const Replay {
            const line = r.lines.items[row];
            // `readDevice` reads an HSPICE P card from a rewritten line.
            if (r.dialect == .hspice and std.ascii.toLower(line[0]) == 'p') return null;
            var f = F.init(line);
            const head = f.next() orelse return null;
            if (!F.isWord(head) or !std.ascii.isAlphabetic(head[0])) return null;
            const kind = r.kindOf(row);
            const letter = if (kind != 0) kind else std.ascii.toLower(head[0]);
            // The letters `readDevice` reads apart from the plain path, and
            // an ngspice R card that may be a B source (`behaviouralResistor`
            // tests a part of the line).
            if (std.mem.indexOfScalar(u8, "bxefgh", letter) != null) return null;
            if (r.dialect == .hspice and std.mem.indexOfScalar(u8, "usw", letter) != null) return null;
            if (letter == 'r' and r.dialect == .ngspice and bTransformationWanted(line)) return null;
            nodes.clearRetainingCapacity();
            fields.clearRetainingCapacity();
            const model = r.cardNodes(a, &f, letter, false, nodes) catch return null;
            if (model) |m| if (r.operatorLed(m)) return null;
            readFields(true, r, a, line, letter, &f, {}, fields, {}) catch return null;
            const rp = a.create(Replay) catch return null;
            rp.* = .{
                .nodes = a.dupe([]const u8, nodes.items) catch return null,
                .model = model,
                .fields = a.dupe(ReplayField, fields.items) catch return null,
            };
            return rp;
        }

        /// Scans top-level cards `rows` into `replays` before any is read
        /// (#230), on `opts.threads` threads, each over a contiguous run of
        /// `rows` with its own arena in `arenas`. A record depends only on
        /// its line, and `readDevice` still reads every card in line order,
        /// so every id and row comes out as without the scan. Returns the
        /// arenas begun, which the caller ends once the cards are read; 0
        /// when it scans nothing.
        pub fn scanCards(r: *R, rows: []const u32, opts: ParseOptions, arenas: *[scan_threads_max]std.heap.ArenaAllocator) Error!usize {
            if (rows.len == 0 or rows.len < opts.scan_min) return 0;
            const want: usize = opts.threads orelse (std.Thread.getCpuCount() catch 1);
            if (want == 0) return 0;
            if (r.replays.len == 0) {
                r.replays = try r.arena.alloc(?*const Replay, r.lines.items.len);
                @memset(r.replays, null);
            }
            const k: usize = if (@import("builtin").single_threaded) 1 else @min(want, scan_threads_max, rows.len);
            std.debug.assert(k >= 1);
            std.debug.assert(k <= rows.len);
            for (arenas[0..k]) |*s| s.* = .init(core.gpa);
            const per = (rows.len + k - 1) / k;
            var threads: [scan_threads_max - 1]std.Thread = undefined;
            var spawned: usize = 0;
            if (!@import("builtin").single_threaded) while (spawned + 1 < k) : (spawned += 1) {
                const lo = @min((spawned + 1) * per, rows.len);
                const hi = @min(lo + per, rows.len);
                threads[spawned] = std.Thread.spawn(.{}, scanRun, .{ @as(*const R, r), &arenas[spawned + 1], rows[lo..hi], r.replays }) catch break;
            };
            // The calling thread scans the first run, then any run whose
            // thread failed to start.
            for (0..k) |j| if (j == 0 or j > spawned) {
                const lo = @min(j * per, rows.len);
                scanRun(r, &arenas[0], rows[lo..@min(lo + per, rows.len)], r.replays);
            };
            for (threads[0..spawned]) |t| t.join();
            return k;
        }

        /// `scan` of each row in `part` into `out`, allocating from `s`.
        pub fn scanRun(r: *const R, s: *std.heap.ArenaAllocator, part: []const u32, out: []?*const Replay) void {
            var nodes: std.ArrayList([]const u8) = .empty;
            var fields: std.ArrayList(ReplayField) = .empty;
            for (part) |i| out[i] = r.scan(s.allocator(), i, &nodes, &fields);
        }
    };
}
