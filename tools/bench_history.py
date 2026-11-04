"""ESPice performance and binary size over commits: one history.tsv row per bench run.

    python3 tools/bench_history.py append [zig-out/benchmark-results.md] [history.tsv] [--exe PATH]
    python3 tools/bench_history.py plot [history.tsv] [zig-out/bench-history]
    python3 tools/bench_history.py --backfill COMMIT... [--history history.tsv]

append: from a `zig build bench` report and the espice it timed (the report's
`- espice:` line names it; --exe overrides), one row per commit (a rerun
replaces it): the commit and its date, the machine, the corpus total of the first
ESPice column, its median per analysis, the geometric-mean speedup over ngspice
on the decks both ran (a same-machine ratio: the one number comparable across
runners), and `size -A`'s .text, .rodata and .debug* bytes.
plot: history-performance.png (corpus total, speedup) and history-size.png.
history.tsv lives on the rolling `bench-history` release, never on main:
.github/workflows/bench.yml downloads it, appends, plots and re-uploads it.

--backfill (local, slow, never in CI): per commit, a temporary worktree, its own
`nix develop .#ci -c zig build -Doptimize=ReleaseFast` (BUILD; ~16 min cold), then
BACKFILL_DECKS timed in that espice and ngspice (median of 3 after a warm-up, `-b -r`),
written as a bench report and appended. Needs ngspice on PATH (nix develop .#benchmarking).
Rows from a backfill and from CI come from different machines; compare the speedup.
"""
import csv, re, statistics, subprocess, sys, tempfile, time
from collections import defaultdict
from pathlib import Path

import bench_plot

COLUMNS = ("commit", "date", "machine", "decks", "total_s", "speedup_vs_ngspice",
           "text_bytes", "rodata_bytes", "debug_bytes", "median_ms_by_analysis")
BUILD = ("nix", "develop", ".#ci", "-c", "zig", "build", "-Doptimize=ReleaseFast")
# A fixed, quick spread of sizes and analyses for --backfill.
BACKFILL_DECKS = ("op/balanced_bridge.sp", "ac/bench_bjt_common_emitter.sp", "tran/bench_digital_buffer_rc.sp",
                  "stress/scaling_rc_ladder_1k.sp", "stress/scaling_resistor_grid_100x100.sp",
                  "stress/scaling_inverter_chain_256.sp", "stress/scaling_parallel_inverters_2000.sp")
FIXTURES = Path("tests/conformance/tests/netlist")


def append(report: Path, history: Path, exe: Path = None, commit: str = None, fixtures: Path = FIXTURES):
    """Add (or replace) `commit`'s row in `history` from `report` and `exe`."""
    header, rows, _, _ = bench_plot.read_report(report)
    if exe is None:
        exe = Path(re.search(r"^- espice: `([^`]+)`", report.read_text(), re.M).group(1))
    machine, head, _ = bench_plot.provenance()
    commit = commit or head
    date = git("show", "-s", "--format=%cI", commit) or time.strftime("%Y-%m-%dT%H:%M:%S")
    ran = [(rel, t[0]) for rel, t in rows if t[0] is not None]
    assert ran, f"{report}: the first ESPice column ran no deck"
    by = defaultdict(list)
    for rel, ms in ran:
        by[bench_plot.analysis(rel, fixtures)].append(ms)
    ng = next((i for i, h in enumerate(header) if h.lower() == "ngspice"), None)
    ratios = [t[ng] / t[0] for _, t in rows if ng is not None and t[0] and t[ng]]
    sizes = sections(exe)
    row = {"commit": commit[:8], "date": date, "machine": machine, "decks": len(ran),
           "total_s": f"{sum(ms for _, ms in ran) / 1e3:.3f}",
           "speedup_vs_ngspice": f"{statistics.geometric_mean(ratios):.3f}" if ratios else "",
           "text_bytes": sizes[".text"], "rodata_bytes": sizes[".rodata"], "debug_bytes": sizes["debug"],
           "median_ms_by_analysis": ";".join(f"{a}={statistics.median(v):.3f}" for a, v in sorted(by.items()))}
    old = [r for r in read(history) if r["commit"] != row["commit"]]
    with history.open("w", newline="") as f:
        w = csv.DictWriter(f, COLUMNS, delimiter="\t", lineterminator="\n")
        w.writeheader()
        w.writerows(old + [row])
    return row


def read(history: Path):
    """history.tsv's rows, oldest commit first; none when it does not exist yet."""
    if not history.exists():
        return []
    return sorted(csv.DictReader(history.read_text().splitlines(), delimiter="\t"), key=lambda r: r["date"])


def plot(history: Path, out: Path):
    """history-performance.png and history-size.png, one point per commit."""
    import matplotlib.pyplot as plt
    rows = read(history)
    assert rows, f"{history}: no rows"
    plt.rcParams.update(bench_plot.STYLE)
    out.mkdir(parents=True, exist_ok=True)
    x = range(len(rows))
    ticks = [f"{r['commit']}\n{r['date'][:10]}" for r in rows]
    foot = (f"One bench run per commit (median of 3 per deck). Machines: {'; '.join(sorted({r['machine'] for r in rows}))}. "
            "Wall times compare only within one machine; the speedup is a same-machine ratio.")

    fig, (top, bottom) = plt.subplots(2, 1, sharex=True, figsize=(9, 6.5))
    top.plot(x, [float(r["total_s"]) for r in rows], marker="o", ms=5, lw=1.5, color=bench_plot.SLOTS[0])
    top.set(ylabel="corpus total, s", title="ESPice corpus wall time (lower is faster)")
    speed = [(i, float(r["speedup_vs_ngspice"])) for i, r in enumerate(rows) if r["speedup_vs_ngspice"]]
    if speed:
        bottom.plot(*zip(*speed), marker="o", ms=5, lw=1.5, color=bench_plot.SLOTS[0])
    bottom.axhline(1, color="#52514e", lw=0.8, ls="--")
    bottom.set(ylabel="speedup over ngspice (geomean)", title="ESPice speedup over ngspice (higher is faster)")
    finish(fig, [top, bottom], ticks, foot, out / "history-performance.png")

    fig, ax = plt.subplots(figsize=(9, 5))
    for k, (col, label) in enumerate((("text_bytes", ".text"), ("rodata_bytes", ".rodata"), ("debug_bytes", "debug"))):
        ax.plot(x, [int(r[col]) / 1e6 for r in rows], marker="os^"[k], ms=5, lw=1.5, color=bench_plot.SLOTS[k], label=label)
    ax.set(ylabel="MB", title="espice binary size by section")
    ax.legend()
    finish(fig, [ax], ticks, "Bytes per section from `size -A` on the benchmarked espice.", out / "history-size.png")


def backfill(commits, history: Path):
    """Build each commit in a temporary worktree and append its timed row (see the header)."""
    import shutil
    root = Path(git("rev-parse", "--show-toplevel"))
    for commit in commits:
        with tempfile.TemporaryDirectory() as d:
            wt, report = Path(d) / "wt", Path(d) / "report.md"
            subprocess.run(["git", "worktree", "add", "--detach", str(wt), commit], check=True)
            try:
                if (root / ".zig-cache").is_dir():  # AGENTS.md: worktrees share the build cache
                    (wt / ".zig-cache").symlink_to(root / ".zig-cache")
                subprocess.run(BUILD, cwd=wt, check=True)
                exe = wt / "zig-out/bin/espice"
                lines = ["| Fixture | espice ms | ngspice ms |", "|---|---:|---:|"]
                for rel in BACKFILL_DECKS:
                    t = [timed([str(exe), "-b", "-r", f"{d}/e.raw", str(root / FIXTURES / rel)]),
                         timed(["ngspice", "-b", "-r", f"{d}/n.raw", str(root / FIXTURES / rel)])]
                    lines.append(f"| {rel} | " + " | ".join(f"{v:.3f}" if v else "—" for v in t) + " |")
                report.write_text("\n".join(lines) + "\n")
                print(append(report, history, exe, git("rev-parse", commit), root / FIXTURES))
            finally:
                subprocess.run(["git", "worktree", "remove", "--force", str(wt)], check=True)
                shutil.rmtree(wt, ignore_errors=True)


def git(*args):
    return subprocess.run(["git", *args], capture_output=True, text=True).stdout.strip()


def sections(exe: Path):
    """`size -A` bytes of .text, .rodata and every .debug* section ("debug")."""
    sizes = {".text": 0, ".rodata": 0, "debug": 0}
    for line in subprocess.run(["size", "-A", str(exe)], capture_output=True, text=True, check=True).stdout.splitlines():
        f = line.split()
        if len(f) >= 2 and f[1].isdigit():
            key = "debug" if f[0].startswith(".debug") else f[0]
            if key in sizes:
                sizes[key] += int(f[1])
    return sizes


def timed(argv):
    """Median wall ms of 3 runs after a warm-up; None if any run fails."""
    walls = []
    for _ in range(4):
        t = time.perf_counter()
        if subprocess.run(argv, capture_output=True, timeout=600).returncode != 0:
            return None
        walls.append((time.perf_counter() - t) * 1e3)
    return statistics.median(walls[1:])


def finish(fig, axes, ticks, foot, path):
    import matplotlib.pyplot as plt
    for ax in axes:
        ax.grid(True, which="major")
        ax.set_ylim(bottom=0)
    axes[-1].set_xticks(range(len(ticks)), ticks, rotation=60, ha="right", fontsize=7)
    fig.text(0.01, 0.01, foot, fontsize=7.5, color="#52514e", wrap=True)
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    fig.savefig(path, dpi=130)
    plt.close(fig)


if __name__ == "__main__":
    args = sys.argv[1:]

    def option(name, default=None):
        if name not in args:
            return default
        i = args.index(name)
        value = args[i + 1]
        del args[i : i + 2]
        return Path(value)

    if not args:
        sys.exit(__doc__)
    if args[0] == "--backfill":
        history = option("--history", Path("history.tsv"))
        backfill(args[1:], history)
    elif args[0] == "append":
        exe = option("--exe")
        print(append(Path(args[1] if len(args) > 1 else "zig-out/benchmark-results.md"),
                     Path(args[2] if len(args) > 2 else "history.tsv"), exe))
    elif args[0] == "plot":
        plot(Path(args[1] if len(args) > 1 else "history.tsv"), Path(args[2] if len(args) > 2 else "zig-out/bench-history"))
    else:
        sys.exit(__doc__)
