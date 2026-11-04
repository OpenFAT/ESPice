"""Measure every GPU-eligible deck on this machine's NVIDIA GPU, as the input of
tools/gpu_extrapolate.py (the dashed lines of tools/bench_plot.py --gpu-phases).

    nix develop -c python3 tools/gpu_phases.py zig-out/bin/espice --ref rtx4060_laptop \
        [--fixtures tests/conformance/tests/netlist] [--filter stress/]... \
        [--trace path/to/libtrace.so] [--out docs/performance/gpu-phases.tsv] [--timeout 900]

Needs the GPU (CI has none: bench.yml renders from the committed file). `--ref` names
this GPU's entry in tools/gpu_devices.json. Per deck, one `--backend cuda` run under
ESPICE_GPU_STATS=phases; a deck whose run prints no `gpu-` line put nothing on the GPU
and is left out. With `--trace`, the same run loads the CUPTI activity tracer of the
A100 model study (CUDA_INJECTION64_PATH; it writes kernel and copy totals to
$TRACE_OUT). Then the cpu and cuda walls, median of 3 without stats or tracer.
Writes gpu_extrapolate.py's workload format: `# gpu/driver/commit/date/ref` lines, a
`_deck` row per deck (its `gpu-` lines in `phases`; host_s = cuda wall less traced
kernels and copies) and, with --trace, a time-only row per kernel. Counted kernels
(warp_inst, fp64_warp_inst, gsect, lsect) come from that study's SASS counting
injector, not from here: without them bench_plot draws no extrapolation for the deck.
"""
import argparse, csv, datetime, os, statistics, subprocess, tempfile, time
from pathlib import Path

import bench_plot

COLUMNS = ("deck", "path", "kernel", "class", "launches", "threads", "block", "regs", "warp_inst", "fp64_warp_inst",
           "gsect", "lsect", "t_ref_us", "host_s", "bus_bytes", "copy_s", "cpu_wall", "cuda_wall", "source",
           "devices", "analysis", "phases")


def measure(exe: Path, fixtures: Path, rel: str, trace, timeout, scratch):
    """The deck's workload rows, or [] when nothing of it ran on the GPU or a run failed."""
    deck = fixtures / rel
    env = dict(os.environ, ESPICE_GPU_STATS="phases")
    if trace:
        env.update(CUDA_INJECTION64_PATH=str(trace), TRACE_OUT=f"{scratch}/trace.tsv")
        Path(f"{scratch}/trace.tsv").unlink(missing_ok=True)  # never the previous deck's
    _, err = run(exe, deck, "cuda", env, timeout, scratch)
    stats = [l for l in (err or "").splitlines() if l.startswith("gpu-")]
    if not stats:
        return []
    walls = {b: [run(exe, deck, b, dict(os.environ), timeout, scratch)[0] for _ in range(3)] for b in ("cpu", "cuda")}
    if None in walls["cpu"] + walls["cuda"]:
        return []
    walls = {b: statistics.median(w) for b, w in walls.items()}
    path = "lanes" if any(l.startswith("gpu-lanes:") for l in stats) else \
        "device eval + device LU" if any(l.startswith("gpu-lu:") for l in stats) else "device eval"
    kernels, copy_s, bus = traced(Path(f"{scratch}/trace.tsv")) if trace else ([], 0.0, 0)
    busy = sum(k["launches"] * k["t_ref_us"] * 1e-6 for k in kernels) + copy_s
    deck_row = {"deck": rel, "path": path, "kernel": "_deck", "host_s": f"{max(0.0, walls['cuda'] - busy):.4f}",
                "bus_bytes": bus, "copy_s": f"{copy_s:.4f}", "cpu_wall": f"{walls['cpu']:.3f}",
                "cuda_wall": f"{walls['cuda']:.3f}", "source": "measured", "devices": bench_plot.devices(deck),
                "analysis": bench_plot.analysis(rel, fixtures), "phases": " | ".join(stats)}
    return [deck_row] + [dict(k, deck=rel, path=path, **{"class": path}, source="time-only") for k in kernels]


def run(exe, deck, backend, env, timeout, scratch):
    """(wall seconds, stderr) of one espice run; (None, stderr) when it fails."""
    argv = [str(exe), "--backend", backend, "--format=binary", "-b", "-r", f"{scratch}/out.raw", deck.name]
    t = time.perf_counter()
    try:
        p = subprocess.run(argv, cwd=deck.parent, env=env, capture_output=True, text=True, timeout=timeout)
    except subprocess.TimeoutExpired:
        print(f"{deck}: {backend} run timed out after {timeout} s", flush=True)
        return None, None
    if p.returncode != 0:
        print(f"{deck}: {backend} run failed ({p.returncode}): {p.stderr[-500:]}", flush=True)
        return None, p.stderr
    return time.perf_counter() - t, p.stderr


def traced(path: Path):
    """The tracer's per-kernel rows (per-launch means) and the copies' seconds and bytes."""
    kernels, copy_s, bus = [], 0.0, 0
    for line in path.read_text().splitlines():
        f = line.split("\t")
        if f[0] == "#copy":
            bus += int(f[3])
            copy_s += float(f[4]) / 1e3
        elif not f[0].startswith("#") and f[0] != "kernel":
            n = int(f[1])
            kernels.append({"kernel": f[0], "launches": n, "t_ref_us": float(f[3]), "block": int(f[7]),
                            "regs": int(f[8]), "threads": int(f[10]) / n})
    return kernels, copy_s, bus


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("exe", type=Path)
    ap.add_argument("--ref", required=True, help="this GPU's key in tools/gpu_devices.json")
    ap.add_argument("--fixtures", type=Path, default=Path("tests/conformance/tests/netlist"))
    ap.add_argument("--filter", action="append", default=[], help="deck path prefix (repeatable)")
    ap.add_argument("--trace", type=Path)
    ap.add_argument("--out", type=Path, default=Path("docs/performance/gpu-phases.tsv"))
    ap.add_argument("--timeout", type=int, default=900)
    a = ap.parse_args()
    gpu, driver = subprocess.run(["nvidia-smi", "--query-gpu=name,driver_version", "--format=csv,noheader"],
                                 capture_output=True, text=True, check=True).stdout.splitlines()[0].split(", ")
    decks = sorted(p for p in a.fixtures.rglob("*.sp")
                   if not a.filter or any(str(p.relative_to(a.fixtures)).startswith(f) for f in a.filter))
    rows = []
    with tempfile.TemporaryDirectory() as scratch:
        for deck in decks:
            got = measure(a.exe.resolve(), a.fixtures.resolve(), str(deck.relative_to(a.fixtures)), a.trace, a.timeout, scratch)
            print(f"{deck}: {'cuda %s s' % got[0]['cuda_wall'] if got else 'nothing on the GPU'}", flush=True)
            rows += got
    a.out.parent.mkdir(parents=True, exist_ok=True)
    with a.out.open("w", newline="") as f:
        commit = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
        f.write(f"# gpu: {gpu}\n# driver: {driver}\n# commit: {commit}\n# date: {datetime.date.today()}\n# ref: {a.ref}\n"
                f"# host: {bench_plot.provenance()[0]}\n")
        w = csv.DictWriter(f, COLUMNS, delimiter="\t", lineterminator="\n")
        w.writeheader()
        w.writerows(rows)
    print(f"{sum(r['kernel'] == '_deck' for r in rows)} GPU decks of {len(decks)} -> {a.out}")


if __name__ == "__main__":
    main()
