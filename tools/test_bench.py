"""Bench graph tools on a tiny fixture: plots and history get written, and the GPU
extrapolation reproduces the measuring GPU's own walls.

    nix develop .#benchmarking -c python3 -m unittest discover -s tools -p 'test_bench*.py'
"""
import subprocess, sys, tempfile, unittest
from pathlib import Path

import bench_history, bench_plot, gpu_extrapolate as gx

TOOLS = Path(__file__).parent
REPORT = """# Simulator benchmark

| Fixture | espice-cpu ms | espice-cuda ms | ngspice ms | espice-cpu/ngspice | espice-cuda/ngspice |
|---|---:|---:|---:|---|---|
| op/divider.sp | 1.000 | 9.000 | 2.000 | agree | agree |
| tran/chain.sp | 3.000 | 2.000 | 24.000 | agree | agree |
| tran/missing.sp | — | — | — | unavailable | unavailable |
"""
DECKS = {"op/divider.sp": "v1 1 0 1\nr1 1 2 1k\nr2 2 0 1k\n.op\n.end\n",
         "tran/chain.sp": ".subckt inv a y\nr1 a y 1k\nc1 y 0 1p\n.ends\nv1 1 0 1\nx1 1 2 inv\nx2 2 3 inv\n.tran 1n 10n\n.end\n"}
# One counted deck (host 0.5 s + copies 1 ms + 10 x 50 us + 2 x 100 us = 0.5017 s) and one
# time-only deck, which the plot must leave out.
PHASES = """# gpu: Fixture GPU
# driver: 1.0
# commit: 0123456789abcdef
# date: 2026-11-04
# ref: rtx4060_laptop
deck\tpath\tkernel\tclass\tlaunches\tthreads\tblock\tregs\twarp_inst\tfp64_warp_inst\tgsect\tlsect\tt_ref_us\thost_s\tbus_bytes\tcopy_s\tcpu_wall\tcuda_wall\tsource\tdevices\tanalysis
counted\tdevice eval\t_deck\t\t\t\t\t\t\t\t\t\t\t0.5\t1000000\t0.001\t0.9\t0.5017\tmeasured\t4000\ttran
counted\tdevice eval\tarp_eval\tdevice eval\t10\t6400\t64\t72\t2000\t500\t1000\t0\t50\t\t\t\t\t\tmeasured\t\t
counted\tdevice eval\tarp_reduce\tdevice eval\t2\t64\t64\t32\t\t\t\t\t100\t\t\t\t\t\ttime-only\t\t
timeonly\tlanes\t_deck\t\t\t\t\t\t\t\t\t\t\t0.2\t0\t0\t0.1\t0.3\tmeasured\t26\top
timeonly\tlanes\tarp_leval\tlanes\t1000\t8192\t64\t255\t\t\t\t\t100\t\t\t\t\t\ttime-only\t\t
"""


class BenchGraphs(unittest.TestCase):
    def setUp(self):
        self.dir = Path(tempfile.mkdtemp())
        for rel, text in DECKS.items():
            (self.dir / "fx" / rel).parent.mkdir(parents=True, exist_ok=True)
            (self.dir / "fx" / rel).write_text(text)
        (self.dir / "report.md").write_text(REPORT)
        (self.dir / "phases.tsv").write_text(PHASES)

    def test_plot_writes_one_graph_per_analysis_with_the_extrapolation(self):
        out = self.dir / "plots"
        subprocess.run([sys.executable, TOOLS / "bench_plot.py", self.dir / "report.md", self.dir / "fx", out,
                        "--gpu-phases", self.dir / "phases.tsv"], check=True, capture_output=True)
        for name in ("all.png", "op.png", "tran.png", "speedup.png", "headline.svg", "summary.md"):
            self.assertGreater((out / name).stat().st_size, 1000 if name.endswith("png") else 0, name)
        # espice-cuda is an ESPice column, not a reference: only ngspice is compared.
        summary = (out / "summary.md").read_text()
        self.assertIn("| ngspice | 2 | 2 | 5.0x | 2 / 0 |", summary)
        self.assertNotIn("| espice-cuda |", summary)
        pts, note = bench_plot.extrapolate(self.dir / "phases.tsv")
        self.assertEqual([p[2] for p in pts], list(bench_plot.EXTRAPOLATE))  # the time-only deck is left out
        self.assertIn("not measurements", note)

    def test_extrapolation_reproduces_the_measuring_gpu(self):
        devs = gx.devices()
        for path in (self.dir / "phases.tsv", TOOLS.parent / "docs/performance/gpu-phases.tsv"):
            ref = devs[gx.meta(path)["ref"]]
            for name, d in gx.load(path).items():
                opt, cons = gx.predict(d, ref, ref)
                self.assertAlmostEqual(opt / d["cuda_wall"], 1, delta=1e-3, msg=f"{path.name}: {name}")
                self.assertGreaterEqual(cons, opt)

    def test_history_appends_a_row_per_commit_and_plots(self):
        history = self.dir / "history.tsv"
        for commit in ("aaaaaaaa", "bbbbbbbb", "aaaaaaaa"):  # a rerun replaces its row
            row = bench_history.append(self.dir / "report.md", history, Path(sys.executable), commit, self.dir / "fx")
        self.assertEqual([r["commit"] for r in bench_history.read(history)], ["bbbbbbbb", "aaaaaaaa"])
        self.assertEqual(row["total_s"], "0.004")  # 1 ms + 3 ms; the deck that ran nowhere is out
        self.assertEqual(row["speedup_vs_ngspice"], "4.000")  # geomean of 2/1 and 24/3
        self.assertEqual(row["median_ms_by_analysis"], "op=1.000;tran=3.000")
        self.assertGreater(int(row["text_bytes"]), 0)
        bench_history.plot(history, self.dir / "hist")
        for name in ("history-performance.png", "history-size.png"):
            self.assertGreater((self.dir / "hist" / name).stat().st_size, 1000, name)


if __name__ == "__main__":
    unittest.main()
