//! The machine calibration file: what `auto`'s price knows about this machine.
//! In: the file's bytes, and the live identity read from sysfs and procfs
//! alone (never the GPU driver). Out: a `Machine`, or null when the file is
//! missing, malformed or stale, which keeps `auto` on the CPU; and the file's
//! bytes from a `Machine`, written atomically (`.tmp` + rename).
//! One file per machine: `<cache>/espice/machine/<machine-id[0..16]>.tsv`,
//! `<cache>` being `$XDG_CACHE_HOME`, else `$HOME/.cache`. Text, tab-separated,
//! one value per row, so a person can read and diff it.
//! Cites: the price plan, sec. 2 (one immutable `Machine` per Problem, in its
//! arena) and sec. 4 (the rows, identity, staleness, the note).
//!
//! Contents:
//! - `pub const Vendor = enum { nvidia, amd }`, `pub const vendor: ?Vendor`
//! - `pub const cards_max: usize`, `pub const model_sizes: [4]u32`
//! - `pub const Text = struct` (`init`, `slice`)
//! - `pub const Identity = struct` (`cardsKey`)
//! - `pub const Constants = struct`
//! - `pub const Model = struct`
//! - `pub const Lu = struct`
//! - `pub const Machine = struct`
//! - `pub const ParseError`, `pub const WriteError`
//! - `pub const test_hooks` (`notes`, `fixture`)
//! - `pub fn identity(root: []const u8, v: Vendor) Identity`
//! - `pub fn stale(file: *const Identity, live: *const Identity) bool`
//! - `pub fn load(arena: Allocator, knobs: Knobs, live: *const Identity) ?Machine`
//! - `pub fn parse(arena: Allocator, text: []const u8) ParseError!Machine`
//! - `pub fn format(m: *const Machine, w: *std.Io.Writer) std.Io.Writer.Error!void`
//! - `pub fn write(gpa: Allocator, m: *const Machine, knobs: Knobs) WriteError!void`
//! - `pub fn filePath(buf: []u8, knobs: Knobs) ?[:0]const u8`
//! - `pub fn noteUncalibrated(live: *const Identity) void`
//! - `pub fn cachePath(buf: []u8, comptime dir: []const u8, key: u64, comptime ext: []const u8, knobs: Knobs) ?[:0]const u8`
//! - `pub fn makeParents(buf: []u8, path: []const u8) bool`

const std = @import("std");
const assert = std.debug.assert;
const Allocator = std.mem.Allocator;
const artifacts = @import("gompute_kernels");
const Knobs = @import("knobs.zig").Knobs;

/// Whose driver and cards `identity` reads.
pub const Vendor = enum { nvidia, amd };

/// This build's kernel vendor; null without kernel images (no GPU at all).
pub const vendor: ?Vendor = if (artifacts.has_cuda) .nvidia else if (artifacts.has_hip) .amd else null;

/// Cards one machine lists, at most (`gpu/devices.zig` `devices_max`).
pub const cards_max: usize = 16;

/// The instance counts a `Model` row is measured at.
pub const model_sizes = [4]u32{ 1, 64, 1024, 16384 };

/// A short text field (a version, a CPU list) held by value.
pub const Text = struct {
    buf: [text_max]u8 = undefined,
    len: u8 = 0,

    /// `s` by value; null when longer than `text_max`.
    pub fn init(s: []const u8) ?Text {
        if (s.len > text_max) return null;
        var t: Text = .{ .len = @intCast(s.len) };
        @memcpy(t.buf[0..s.len], s);
        return t;
    }

    pub fn slice(t: *const Text) []const u8 {
        return t.buf[0..t.len];
    }
};

/// What a calibration was measured on. A file whose identity differs from
/// the live one (`stale`) prices nothing.
pub const Identity = struct {
    /// The kernel module's version (`/sys/module/nvidia/version`); empty
    /// when no driver is loaded.
    driver: Text = .{},
    /// Every card's uuid; the file keeps CUDA ordinal order, `identity`
    /// procfs order. Compared as a set.
    uuids: [cards_max][16]u8 = undefined,
    cards: u8 = 0,
    /// Host threads (`std.Thread.getCpuCount`): the host lanes' gain.
    threads: u32 = 0,
    /// The kernel's CPU lists of the performance and efficiency cores
    /// (`/sys/devices/cpu_{core,atom}/cpus`); empty on a uniform CPU.
    p_cpus: Text = .{},
    e_cpus: Text = .{},

    /// The wrapping sum of `Wyhash(0, uuid)` over the cards: order-free,
    /// and for one CUDA card the key `cuDeviceGetUuid`'s bytes gave
    /// (`CostCache.key`). 0 without cards.
    pub fn cardsKey(id: *const Identity) u64 {
        assert(id.cards <= cards_max);
        var key: u64 = 0;
        for (id.uuids[0..id.cards]) |*u| key +%= std.hash.Wyhash.hash(0, u);
        return key;
    }
};

/// The measured constants the price reads, one file row each, named as
/// the rows are. Every row must be present; a file without one is stale.
/// The first fixture (`gpu/placement.machine.tsv`) holds the constants the
/// price carried in code before (RTX 4060 Laptop sm_89, i9-14900HX).
pub const Constants = struct {
    /// One-time driver setup a query pays when the driver is not up yet
    /// (cuInit, the context, module loads), ms.
    start_ms: f64,
    /// Per device eval: the launches and the one host wait, us.
    eval_fixed_us: f64,
    /// Pinned host-device transfer rate, bytes per us.
    bus_b_per_us: f64,
    /// Staging clear and the two reduce levels, staging bytes per us.
    stage_b_per_us: f64,
    /// Host lanes ParEval is worth per thread past the first.
    host_lane_gain: f64,
    /// The GPU must beat the CPU by this factor over the whole query...
    margin: f64,
    /// ...and by this factor per eval.
    min_ratio: f64,
    /// One device thread against one host core on the same instance's eval.
    slowdown: f64,
    /// Device threads that run at once (a batch past it takes waves).
    concurrency: f64,
    /// Host ns per held byte past `hist_floor_bytes`, per eval: the history
    /// walk an empty-history probe cannot see.
    hist_ns_per_byte: f64,
    /// Held bytes per instance the probe already pays for.
    hist_floor_bytes: f64,
    /// A history price loses on the device past this factor...
    host_ratio: f64,
    /// ...and past this many us per eval.
    host_loss_us: f64,
    /// The card's FP32:FP64 throughput ratio (2 on datacenter parts, 64 on
    /// a laptop RTX 4060); the GPU lane policy reads it.
    fp64_ratio: f64,
    /// Streaming multiprocessors (compute units).
    sms: f64,
    /// Compute capability, major.minor.
    cc: f64,
    /// Device memory, GB.
    mem_gb: f64,
    /// The driver start overlap can hide (in-process start), ms. Priced
    /// from stage 4 of the plan on, as are the rows below.
    start_in_ms: f64,
    /// One empty kernel's launch and sync, us.
    launch_us: f64,
    /// Pinned transfer latency, us, and the up/down rates, GB/s.
    bus_lat_us: f64,
    bus_up_gbs: f64,
    bus_down_gbs: f64,
    /// Device-LU host table build per pivot epoch, ns per LU flop.
    lu_table_ns_per_flop: f64,
    /// One eval's host time on an efficiency core over a performance core
    /// (the price plan's P/E ratio; 1 on a uniform CPU).
    pe_ratio: f64,
};

/// One kernel image's measured row, keyed by the image's Wyhash, so a
/// rebuild that leaves a model's kernel unchanged keeps its row. Each
/// array holds one value per `model_sizes` count.
pub const Model = struct {
    image: u64,
    /// Device us per eval of the whole batch.
    dev_us: [4]f64,
    /// Host us per eval on one performance core, and on one efficiency core.
    host_p_us: [4]f64,
    host_e_us: [4]f64,
    /// The image's first module load (its JIT), ms; 0 when SASS fits.
    jit_ms: f64,
    /// ParEval's gain on the lanes a query borrows (`ThreadPool.per_query`)
    /// over one lane, at the widest size.
    lane_gain: f64,
};

/// One generated matrix's measured LU, raw: the price fits its rates over
/// these rows (flops per column `flops / n` is the axis).
pub const Lu = struct {
    n: u32,
    /// Flops of one refactor (`GpuLu.flops`).
    flops: u64,
    /// One device refactor and both solves, upload to download, us.
    dev_us: f64,
    /// One host refactor and solve (`SparseLu`, one thread), us.
    host_us: f64,
    /// The device tables' build and upload for the pivot epoch, us.
    table_us: f64,
};

/// The calibration file, parsed. Immutable after `load`; every query of a
/// Problem reads it. Text lives in the struct, `models` in the loader's arena.
pub const Machine = struct {
    id: Identity,
    /// The espice version that wrote the file.
    espice: Text = .{},
    c: Constants,
    models: []const Model = &.{},
    lus: []const Lu = &.{},
};

pub const ParseError = error{
    /// A row whose value does not parse, or a missing constant or header row.
    Malformed,
    OutOfMemory,
};

pub const WriteError = error{
    /// Neither `$XDG_CACHE_HOME` nor `$HOME`, or a path past the buffer.
    NoCacheDir,
    /// `/etc/machine-id` is unreadable: there is no file name.
    NoMachineId,
    /// An open, write or rename failed; the old file, if any, is untouched.
    WriteFailed,
    OutOfMemory,
};

/// Test binaries only: notes `noteUncalibrated` printed, from any thread,
/// and the placement oracle's fixture file (`gpu/placement.machine.tsv`).
pub const test_hooks = if (@import("builtin").is_test) struct {
    pub var notes: std.atomic.Value(u32) = .init(0);
    pub const fixture = @embedFile("gpu/placement.machine.tsv");
} else struct {};

/// Longest `Text`: a driver version or CPU list is a few bytes.
const text_max = 64;

/// A calibration file past this is not one (`load` refuses it): ~90 model
/// rows are ~25 KB.
const file_max = 64 << 10;

/// The machine's identity, read under `root` ("" for the live system; a
/// fake sysfs tree in tests) from sysfs and procfs only: no driver call.
/// Fields it cannot read stay empty; `threads` is always the live count.
pub fn identity(root: []const u8, v: Vendor) Identity {
    var id: Identity = .{ .threads = @intCast(@max(1, std.Thread.getCpuCount() catch 1)) };
    var buf: [4096]u8 = undefined;
    const driver = switch (v) {
        .nvidia => "/sys/module/nvidia/version",
        .amd => "/sys/module/amdgpu/version",
    };
    if (readAt(root, driver, &buf)) |t| id.driver = Text.init(trim(t)) orelse .{};
    if (readAt(root, "/sys/devices/cpu_core/cpus", &buf)) |t| id.p_cpus = Text.init(trim(t)) orelse .{};
    if (readAt(root, "/sys/devices/cpu_atom/cpus", &buf)) |t| id.e_cpus = Text.init(trim(t)) orelse .{};
    readCards(root, v, &id);
    assert(id.cards <= cards_max);
    assert(id.threads > 0);
    return id;
}

/// Whether a file measured on `file` must not price on `live`: another
/// driver version, another set of cards, or another host thread count.
pub fn stale(file: *const Identity, live: *const Identity) bool {
    assert(file.cards <= cards_max);
    assert(live.cards <= cards_max);
    if (!std.mem.eql(u8, file.driver.slice(), live.driver.slice())) return true;
    if (file.threads != live.threads or file.cards != live.cards) return true;
    for (file.uuids[0..file.cards]) |*u| {
        const found = for (live.uuids[0..live.cards]) |*w| {
            if (std.mem.eql(u8, u, w)) break true;
        } else false;
        if (!found) return true;
    }
    return false;
}

/// The calibration file under `knobs`' cache root when it parses and was
/// measured on `live`; null when missing, unreadable, malformed or stale.
/// `models` and the file's bytes live in `arena`.
pub fn load(arena: Allocator, knobs: Knobs, live: *const Identity) ?Machine {
    var buf: [512]u8 = undefined;
    const p = filePath(&buf, knobs) orelse return null;
    const fd = std.c.open(p, .{ .ACCMODE = .RDONLY }, @as(c_uint, 0));
    if (fd < 0) return null;
    defer _ = std.c.close(fd);
    const text = arena.alloc(u8, file_max) catch return null;
    var len: usize = 0;
    for (0..file_max) |_| {
        const got = std.c.read(fd, text[len..].ptr, text.len - len);
        if (got < 0) return null;
        if (got == 0) break;
        len += @intCast(got);
        if (len == text.len) return null;
    }
    const m = parse(arena, text[0..len]) catch return null;
    if (stale(&m.id, live)) return null;
    return m;
}

/// Parses a calibration file. Unknown rows (a newer espice's) are skipped;
/// every `Constants` row and the `driver` and `threads` rows are required.
pub fn parse(arena: Allocator, text: []const u8) ParseError!Machine {
    const names = @typeInfo(Constants).@"struct".field_names;
    var m: Machine = .{ .id = .{}, .c = undefined };
    var seen: std.StaticBitSet(names.len) = .empty;
    var have_driver = false;
    var models: std.ArrayList(Model) = .empty;
    var lus: std.ArrayList(Lu) = .empty;
    var lines = std.mem.splitScalar(u8, text, '\n');
    while (lines.next()) |raw| {
        const line = std.mem.trimEnd(u8, raw, "\r");
        if (line.len == 0 or line[0] == '#') continue;
        const tab = std.mem.indexOfScalar(u8, line, '\t') orelse return error.Malformed;
        const key = line[0..tab];
        const value = line[tab + 1 ..];
        if (std.mem.eql(u8, key, "driver")) {
            m.id.driver = Text.init(value) orelse return error.Malformed;
            have_driver = true;
        } else if (std.mem.eql(u8, key, "uuid")) {
            if (m.id.cards == cards_max) return error.Malformed;
            m.id.uuids[m.id.cards] = hexUuid(value) orelse return error.Malformed;
            m.id.cards += 1;
        } else if (std.mem.eql(u8, key, "threads")) {
            m.id.threads = std.fmt.parseInt(u32, value, 10) catch return error.Malformed;
        } else if (std.mem.eql(u8, key, "p_cpus")) {
            m.id.p_cpus = Text.init(value) orelse return error.Malformed;
        } else if (std.mem.eql(u8, key, "e_cpus")) {
            m.id.e_cpus = Text.init(value) orelse return error.Malformed;
        } else if (std.mem.eql(u8, key, "espice")) {
            m.espice = Text.init(value) orelse return error.Malformed;
        } else if (std.mem.eql(u8, key, "model")) {
            try models.append(arena, try parseModel(value));
        } else if (std.mem.eql(u8, key, "lu")) {
            try lus.append(arena, try parseLu(value));
        } else inline for (names, 0..) |name, i| {
            if (std.mem.eql(u8, key, name)) {
                @field(m.c, name) = try parseNumber(value);
                seen.set(i);
            }
        }
    }
    if (seen.count() != names.len or !have_driver or m.id.threads == 0) return error.Malformed;
    m.models = try models.toOwnedSlice(arena);
    m.lus = try lus.toOwnedSlice(arena);
    assert(m.id.cards <= cards_max);
    return m;
}

/// Writes `m` as the file `parse` reads back bit for bit.
pub fn format(m: *const Machine, w: *std.Io.Writer) std.Io.Writer.Error!void {
    assert(m.id.cards <= cards_max);
    assert(m.id.threads > 0);
    try w.writeAll(
        \\# espice machine calibration (price plan sec. 4): one value per row, tab-separated.
        \\# model: image Wyhash, then dev_us, host_p_us and host_e_us at 1, 64, 1024 and 16384
        \\# instances, jit_ms, lane_gain. lu: n, flops, dev_us, host_us, table_us.
        \\
    );
    try w.print("driver\t{s}\n", .{m.id.driver.slice()});
    for (m.id.uuids[0..m.id.cards]) |*u| {
        try w.writeAll("uuid\tGPU-");
        for (u, 0..) |byte, i| {
            if (i == 4 or i == 6 or i == 8 or i == 10) try w.writeByte('-');
            try w.print("{x:0>2}", .{byte});
        }
        try w.writeByte('\n');
    }
    try w.print("threads\t{d}\np_cpus\t{s}\ne_cpus\t{s}\nespice\t{s}\n", .{ m.id.threads, m.id.p_cpus.slice(), m.id.e_cpus.slice(), m.espice.slice() });
    inline for (@typeInfo(Constants).@"struct".field_names) |name| {
        const x = @field(m.c, name);
        assert(std.math.isFinite(x));
        try w.print(name ++ "\t{d}\n", .{x});
    }
    for (m.models) |*row| {
        try w.print("model\t{x:0>16}", .{row.image});
        inline for (@typeInfo(Model).@"struct".field_names[1..]) |name| {
            const v = @field(row, name);
            const xs: []const f64 = if (@FieldType(Model, name) == f64) &.{v} else &v;
            for (xs) |x| try w.print("\t{d}", .{x});
        }
        try w.writeByte('\n');
    }
    for (m.lus) |*row| {
        assert(row.n > 0);
        try w.print("lu\t{d}\t{d}\t{d}\t{d}\t{d}\n", .{ row.n, row.flops, row.dev_us, row.host_us, row.table_us });
    }
}

/// Writes `m` to its path under `knobs`' cache root: a `.tmp` beside it,
/// then a rename, so a reader sees the old file or the new one, never half.
pub fn write(gpa: Allocator, m: *const Machine, knobs: Knobs) WriteError!void {
    var buf: [512]u8 = undefined;
    const p = filePath(&buf, knobs) orelse return if (machineId() == null) error.NoMachineId else error.NoCacheDir;
    if (!makeParents(&buf, p)) return error.NoCacheDir;
    var tmp_buf: [520]u8 = undefined;
    const tmp = std.mem.printSentinel(&tmp_buf, "{s}.tmp", .{p}, 0) catch return error.NoCacheDir;
    var out: std.Io.Writer.Allocating = .init(gpa);
    defer out.deinit();
    format(m, &out.writer) catch return error.OutOfMemory;
    const bytes = out.written();
    const fd = std.c.open(tmp, .{ .ACCMODE = .WRONLY, .CREAT = true, .TRUNC = true }, @as(c_uint, 0o644));
    if (fd < 0) return error.WriteFailed;
    var done: usize = 0;
    const ok = for (0..bytes.len + 1) |_| {
        if (done == bytes.len) break true;
        const put = std.c.write(fd, bytes[done..].ptr, bytes.len - done);
        if (put <= 0) break false;
        done += @intCast(put);
    } else false;
    _ = std.c.close(fd);
    if (!ok or std.c.rename(tmp, p) != 0) {
        _ = std.c.unlink(tmp);
        return error.WriteFailed;
    }
    assert(done == bytes.len);
}

/// This machine's file in `buf`: `cachePath` keyed by `/etc/machine-id`'s
/// first 16 hex digits, so an NFS home shared by several machines keeps one
/// each. Null without a machine id or a cache root.
pub fn filePath(buf: []u8, knobs: Knobs) ?[:0]const u8 {
    return cachePath(buf, "machine", machineId() orelse return null, ".tsv", knobs);
}

/// Prints, once per process, why `auto` runs on the CPU: no calibration
/// for the driver `live` shows. Silent without a driver (nothing to
/// calibrate); the caller checks the build has kernels.
pub fn noteUncalibrated(live: *const Identity) void {
    const Once = struct {
        var printed: std.atomic.Value(bool) = .init(false);
    };
    if (live.driver.len == 0) return;
    if (Once.printed.swap(true, .monotonic)) return;
    if (@import("builtin").is_test) _ = test_hooks.notes.fetchAdd(1, .monotonic);
    std.debug.print("note: auto runs on the CPU: no GPU calibration for driver {s}; run 'espice calibrate' (~25 s)\n", .{live.driver.slice()});
}

/// `<cache>/espice/<dir>/<key as 16 hex digits><ext>` in `buf`, `<cache>`
/// being `$XDG_CACHE_HOME`, else `$HOME/.cache`. Null without either, or
/// past `buf` (no path, never a truncated one).
pub fn cachePath(buf: []u8, comptime dir: []const u8, key: u64, comptime ext: []const u8, knobs: Knobs) ?[:0]const u8 {
    if (knobs.xdg_cache_home) |x|
        return std.mem.printSentinel(buf, "{s}/espice/" ++ dir ++ "/{x:0>16}" ++ ext, .{ x, key }, 0) catch null;
    const home = knobs.home orelse return null;
    return std.mem.printSentinel(buf, "{s}/.cache/espice/" ++ dir ++ "/{x:0>16}" ++ ext, .{ home, key }, 0) catch null;
}

/// mkdir -p for `path`'s `espice/` and `<dir>/` (`cachePath`), whose bytes
/// live at the start of `buf`; EEXIST is the common case. False when
/// `path` has fewer than two slashes. Best effort: a failed mkdir surfaces
/// as the caller's failed open.
pub fn makeParents(buf: []u8, path_: []const u8) bool {
    const leaf = std.mem.lastIndexOfScalar(u8, path_, '/') orelse return false;
    const app = std.mem.lastIndexOfScalar(u8, path_[0..leaf], '/') orelse return false;
    for ([_]usize{ app, leaf }) |end| {
        buf[end] = 0;
        _ = std.c.mkdir(@ptrCast(buf.ptr), 0o755);
        buf[end] = '/';
    }
    return true;
}

// ---- private ----

fn trim(t: []const u8) []const u8 {
    return std.mem.trim(u8, t, " \t\r\n");
}

/// `/etc/machine-id`'s first 16 hex digits, or null.
fn machineId() ?u64 {
    // ponytail: no `/var/lib/dbus/machine-id` fallback; a container without
    // either keeps `auto` on the CPU.
    var buf: [64]u8 = undefined;
    const t = readAt("", "/etc/machine-id", &buf) orelse return null;
    if (t.len < 16) return null;
    return std.fmt.parseInt(u64, t[0..16], 16) catch null;
}

/// Appends every card procfs (NVIDIA) or the kfd topology (AMD) lists under
/// `root` to `id`, at most `cards_max`.
fn readCards(root: []const u8, v: Vendor, id: *Identity) void {
    if (comptime @import("builtin").os.tag != .linux) return;
    const dir_name = switch (v) {
        .nvidia => "/proc/driver/nvidia/gpus",
        .amd => "/sys/class/kfd/kfd/topology/nodes",
    };
    var dir_buf: [512]u8 = undefined;
    const dir_path = std.mem.printSentinel(&dir_buf, "{s}{s}", .{ root, dir_name }, 0) catch return;
    const dir = std.c.opendir(dir_path) orelse return;
    defer _ = std.c.closedir(dir);
    // Bounded: one entry per card plus `.` and `..` (and kfd's CPU nodes).
    for (0..4 * cards_max) |_| {
        const ent = std.c.readdir(dir) orelse break;
        const name = std.mem.sliceTo(&ent.name, 0);
        if (name.len == 0 or name[0] == '.') continue;
        var path_buf: [600]u8 = undefined;
        const leaf = switch (v) {
            .nvidia => "information",
            .amd => "properties",
        };
        const p = std.fmt.bufPrint(&path_buf, "{s}/{s}/{s}", .{ dir_name, name, leaf }) catch continue;
        var text_buf: [4096]u8 = undefined;
        const text = readAt(root, p, &text_buf) orelse continue;
        const uuid = switch (v) {
            .nvidia => nvidiaUuid(text),
            .amd => kfdUuid(text),
        } orelse continue;
        if (id.cards == cards_max) break;
        id.uuids[id.cards] = uuid;
        id.cards += 1;
    }
}

/// The 16 bytes of the `GPU UUID: GPU-xxxxxxxx-xxxx-...` line of an
/// NVIDIA procfs `information` file, or null when it has none.
fn nvidiaUuid(text: []const u8) ?[16]u8 {
    const key = "GPU UUID:";
    const at = std.mem.indexOf(u8, text, key) orelse return null;
    const rest = std.mem.trimStart(u8, text[at + key.len ..], " \t");
    const end = std.mem.indexOfAny(u8, rest, " \t\r\n") orelse rest.len;
    return hexUuid(rest[0..end]);
}

/// `GPU-xxxxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx` (the `GPU-` and the dashes
/// optional) as 16 bytes, or null: not 32 hex digits.
fn hexUuid(text: []const u8) ?[16]u8 {
    const hex = if (std.mem.startsWith(u8, text, "GPU-")) text["GPU-".len..] else text;
    var out: [16]u8 = undefined;
    var nibbles: usize = 0;
    for (hex) |c| {
        if (c == '-') continue;
        if (nibbles == 32) return null;
        const v = std.fmt.charToDigit(c, 16) catch return null;
        if (nibbles % 2 == 0) out[nibbles / 2] = v << 4 else out[nibbles / 2] |= v;
        nibbles += 1;
    }
    if (nibbles != 32) return null;
    return out;
}

/// A kfd node's nonzero `unique_id` as 16 bytes (little-endian, zero
/// padded), or null: a CPU node, or no such line.
fn kfdUuid(text: []const u8) ?[16]u8 {
    var lines = std.mem.splitScalar(u8, text, '\n');
    while (lines.next()) |line| {
        if (!std.mem.startsWith(u8, line, "unique_id ")) continue;
        const v = std.fmt.parseInt(u64, trim(line["unique_id ".len..]), 10) catch return null;
        if (v == 0) return null;
        var out: [16]u8 = @splat(0);
        std.mem.writeInt(u64, out[0..8], v, .little);
        return out;
    }
    return null;
}

/// `root ++ rel`'s first `buf.len` bytes, or null when it cannot be read.
fn readAt(root: []const u8, rel: []const u8, buf: []u8) ?[]const u8 {
    var path_buf: [1024]u8 = undefined;
    const p = std.mem.printSentinel(&path_buf, "{s}{s}", .{ root, rel }, 0) catch return null;
    const fd = std.c.open(p, .{ .ACCMODE = .RDONLY }, @as(c_uint, 0));
    if (fd < 0) return null;
    defer _ = std.c.close(fd);
    const got = std.c.read(fd, buf.ptr, buf.len);
    if (got <= 0) return null;
    return buf[0..@intCast(got)];
}

fn parseNumber(value: []const u8) ParseError!f64 {
    const x = std.fmt.parseFloat(f64, value) catch return error.Malformed;
    if (!std.math.isFinite(x)) return error.Malformed;
    return x;
}

/// A `model` row's value: the image hash, then `Model`'s numbers in order.
fn parseModel(value: []const u8) ParseError!Model {
    var cols = std.mem.splitScalar(u8, value, '\t');
    var row: Model = undefined;
    row.image = std.fmt.parseInt(u64, cols.next() orelse return error.Malformed, 16) catch return error.Malformed;
    inline for (@typeInfo(Model).@"struct".field_names[1..]) |name| {
        if (@FieldType(Model, name) == f64) {
            @field(row, name) = try parseNumber(cols.next() orelse return error.Malformed);
        } else for (&@field(row, name)) |*x| {
            x.* = try parseNumber(cols.next() orelse return error.Malformed);
        }
    }
    if (cols.next() != null) return error.Malformed;
    return row;
}

/// An `lu` row's value: n, flops, then `Lu`'s three times.
fn parseLu(value: []const u8) ParseError!Lu {
    var cols = std.mem.splitScalar(u8, value, '\t');
    var row: Lu = undefined;
    row.n = std.fmt.parseInt(u32, cols.next() orelse return error.Malformed, 10) catch return error.Malformed;
    row.flops = std.fmt.parseInt(u64, cols.next() orelse return error.Malformed, 10) catch return error.Malformed;
    inline for (.{ "dev_us", "host_us", "table_us" }) |name| @field(row, name) = try parseNumber(cols.next() orelse return error.Malformed);
    if (cols.next() != null or row.n == 0) return error.Malformed;
    return row;
}

test "machine: format then parse is bit-exact, and parse refuses a short file" {
    // Oracle: the round trip itself (format -> parse -> format), on values
    // chosen to need all 17 digits, and the fixture's own bytes.
    var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    var m = try parse(a, test_hooks.fixture);
    try std.testing.expectEqual(@as(f64, 350), m.c.start_ms);
    try std.testing.expectEqual(@as(f64, 64), m.c.fp64_ratio);
    m.c.slowdown = 0.1 + 0.2;
    m.c.margin = std.math.floatEps(f64) * 3;
    m.c.cc = 1e300;
    const models = [_]Model{.{ .image = 0xdeadbeef00c0ffee, .dev_us = .{ 1.0 / 3.0, 2, 3e-9, 4 }, .host_p_us = @splat(0.7), .host_e_us = @splat(1.9), .jit_ms = 15000, .lane_gain = 3.3 }};
    m.models = &models;
    const lus = [_]Lu{.{ .n = 4096, .flops = 734_000_001, .dev_us = 0.1 + 0.7, .host_us = 1e-7, .table_us = 123456.789 }};
    m.lus = &lus;
    var w1: std.Io.Writer.Allocating = .init(a);
    try format(&m, &w1.writer);
    const back = try parse(a, w1.written());
    try std.testing.expectEqual(std.mem.toBytes(m.c), std.mem.toBytes(back.c));
    try std.testing.expectEqual(std.mem.toBytes(models[0]), std.mem.toBytes(back.models[0]));
    try std.testing.expectEqual(std.mem.toBytes(lus[0]), std.mem.toBytes(back.lus[0]));
    try std.testing.expectEqual(m.id.cardsKey(), back.id.cardsKey());
    try std.testing.expect(!stale(&m.id, &back.id));
    var w2: std.Io.Writer.Allocating = .init(a);
    try format(&back, &w2.writer);
    try std.testing.expectEqualStrings(w1.written(), w2.written());
    // A row missing, or one that does not parse: no machine.
    const text = w1.written();
    const row = std.mem.indexOf(u8, text, "slowdown\t").?;
    const row_end = row + std.mem.indexOfScalar(u8, text[row..], '\n').? + 1;
    const cut = try std.mem.concat(a, u8, &.{ text[0..row], text[row_end..] });
    try std.testing.expectError(error.Malformed, parse(a, cut));
    const bad = try std.mem.concat(a, u8, &.{ text[0..row], "slowdown\tfast\n", text[row_end..] });
    try std.testing.expectError(error.Malformed, parse(a, bad));
    const nan = try std.mem.concat(a, u8, &.{ text[0..row], "slowdown\tnan\n", text[row_end..] });
    try std.testing.expectError(error.Malformed, parse(a, nan));
    // A newer espice's row is skipped.
    const newer = try std.mem.concat(a, u8, &.{ text, "pe_ratio\t2.4\n" });
    try std.testing.expectEqual(std.mem.toBytes(m.c), std.mem.toBytes((try parse(a, newer)).c));
}

test "machine: identity reads a fake sysfs root, and staleness follows driver, cards and threads" {
    // Oracle: a hand-made sysfs/procfs tree with this RTX 4060 Laptop's
    // files, its uuid bytes transcribed by hand from the hex.
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    try tmp.dir.createDirPath(std.testing.io, "sys/module/nvidia");
    try tmp.dir.createDirPath(std.testing.io, "sys/devices/cpu_core");
    try tmp.dir.createDirPath(std.testing.io, "sys/devices/cpu_atom");
    try tmp.dir.createDirPath(std.testing.io, "proc/driver/nvidia/gpus/0000:01:00.0");
    try tmp.dir.createDirPath(std.testing.io, "proc/driver/nvidia/gpus/0000:02:00.0");
    const files = [_][2][]const u8{
        .{ "sys/module/nvidia/version", "595.71.05\n" },
        .{ "sys/devices/cpu_core/cpus", "0-15\n" },
        .{ "sys/devices/cpu_atom/cpus", "16-31\n" },
        .{ "proc/driver/nvidia/gpus/0000:01:00.0/information", "Model: \t\t NVIDIA GeForce RTX 4060 Laptop GPU\nIRQ:   \t\t 187\nGPU UUID: \t GPU-a5b262fa-fc3e-0653-9187-0d58e7372cbd\nVideo BIOS: \t 95.07.41.00.56\n" },
        // A second card whose file has no uuid line: not counted.
        .{ "proc/driver/nvidia/gpus/0000:02:00.0/information", "Model: \t\t x\n" },
    };
    for (files) |f| try tmp.dir.writeFile(std.testing.io, .{ .sub_path = f[0], .data = f[1] });
    var root_buf: [128]u8 = undefined;
    const root = try std.fmt.bufPrint(&root_buf, ".zig-cache/tmp/{s}", .{&tmp.sub_path});
    const id = identity(root, .nvidia);
    try std.testing.expectEqualStrings("595.71.05", id.driver.slice());
    try std.testing.expectEqualStrings("0-15", id.p_cpus.slice());
    try std.testing.expectEqualStrings("16-31", id.e_cpus.slice());
    try std.testing.expectEqual(@as(u8, 1), id.cards);
    const want = [16]u8{ 0xa5, 0xb2, 0x62, 0xfa, 0xfc, 0x3e, 0x06, 0x53, 0x91, 0x87, 0x0d, 0x58, 0xe7, 0x37, 0x2c, 0xbd };
    try std.testing.expectEqualSlices(u8, &want, &id.uuids[0]);
    try std.testing.expectEqual(std.hash.Wyhash.hash(0, &want), id.cardsKey());
    // A root with nothing: no driver, no cards, so `noteUncalibrated` is silent.
    const none = identity("/nonexistent-espice-root", .nvidia);
    try std.testing.expectEqual(@as(usize, 0), none.driver.len);
    try std.testing.expectEqual(@as(u8, 0), none.cards);

    try std.testing.expect(!stale(&id, &id));
    var other = id;
    other.driver = Text.init("595.71.06").?;
    try std.testing.expect(stale(&id, &other));
    other = id;
    other.threads += 1;
    try std.testing.expect(stale(&id, &other));
    other = id;
    other.uuids[0][15] ^= 1;
    try std.testing.expect(stale(&id, &other));
    other = id;
    other.cards = 0;
    try std.testing.expect(stale(&id, &other));
    // Same cards in another order: the same machine.
    var two = id;
    two.uuids[1] = @splat(7);
    two.cards = 2;
    var swapped = two;
    swapped.uuids[0] = two.uuids[1];
    swapped.uuids[1] = two.uuids[0];
    try std.testing.expect(!stale(&two, &swapped));
    try std.testing.expectEqual(two.cardsKey(), swapped.cardsKey());
    // Other CPU lists alone do not make a file stale.
    other = id;
    other.p_cpus = .{};
    try std.testing.expect(!stale(&id, &other));
}

test "machine: write then load, a stale identity loads nothing, an empty cache root has no file" {
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    var root_buf: [128]u8 = undefined;
    const knobs: Knobs = .{ .xdg_cache_home = try std.fmt.bufPrint(&root_buf, ".zig-cache/tmp/{s}", .{&tmp.sub_path}) };
    var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const live = identity("", .nvidia);
    try std.testing.expect(load(a, knobs, &live) == null);
    var m = try parse(a, test_hooks.fixture);
    m.id = live;
    try write(std.testing.allocator, &m, knobs);
    const got = load(a, knobs, &live) orelse return error.TestUnexpectedResult;
    try std.testing.expectEqual(std.mem.toBytes(m.c), std.mem.toBytes(got.c));
    // A rewrite replaces it whole; no `.tmp` stays behind.
    m.c.slowdown = 7;
    try write(std.testing.allocator, &m, knobs);
    try std.testing.expectEqual(@as(f64, 7), (load(a, knobs, &live) orelse return error.TestUnexpectedResult).c.slowdown);
    var buf: [512]u8 = undefined;
    const p = filePath(&buf, knobs).?;
    var tmp_buf: [520]u8 = undefined;
    try std.testing.expect(std.c.access(try std.mem.printSentinel(&tmp_buf, "{s}.tmp", .{p}, 0), 0) != 0);
    var moved = live;
    moved.threads += 1;
    try std.testing.expect(load(a, knobs, &moved) == null);
}
