#!/usr/bin/env python3
"""Extrapolate ESPice cuda-backend walls measured on one GPU (the workload file's `# ref:`
device, a key of gpu_devices.json) to another GPU. Copied from the A100 model study
(RTX 4060 Laptop, 2026-11); tools/bench_plot.py --gpu-phases draws its predictions.

usage: gpu_extrapolate.py --device NAME [--workload docs/performance/gpu-phases.tsv] [--devices tools/gpu_devices.json] [--ncu DIR]
       gpu_extrapolate.py --check          the ref GPU reproduces its own walls (round trip + held-out e)
       from gpu_extrapolate import load, predict; opt, cons = predict(decks[deck], devices()["a100_80_sxm"], devices()[ref])
The workload: `# key: value` lines (gpu, driver, commit, date, ref), then one `_deck` row
per deck (host_s, bus_bytes, copy_s, cpu_wall, cuda_wall, devices, analysis) and one row
per kernel (launches, threads, block, t_ref_us: mean per launch on the ref GPU; counted
kernels add warp_inst, fp64_warp_inst, gsect, lsect). tools/gpu_phases.py writes it.

Per kernel launch the roofline terms (seconds) on device g are
  fp64  = FP64-pipe warp inst * 64 / fp64 FLOP/s * u   (u = tail imbalance of the grid over g's SMs)
  issue = warp inst / warp-issue rate * u
  mem   = max(32 * (global+local) sectors / L2 B/s, DRAM bytes / DRAM B/s)   (DRAM bytes only from ncu)
  lat   = warp inst per warp * CPI / f                (one warp's dependent chain)
  floor = 1 us
e = model_4060 / measured_4060 (a calibration, not clamped), fitted per (deck, kernel). Throughput-bound on the 4060:
opt = model_g / e, cons = max(model_g / (e/2), model_g + unexplained 4060 time). Latency/floor-bound
on the 4060: opt = measured, cons = measured * f_4060 / f_g. Uncounted kernels keep their time the same
way. Deck wall = host seconds (held at this machine's i9-14900HX) + kernels + copies, i.e. the GPU is
assumed on the critical path; copies keep their per-copy latency and move bytes at g's bus rate.
"""
import argparse, csv, json, math, os, statistics
HERE = os.path.dirname(os.path.abspath(__file__))
REF, CPI, FLOOR = "rtx4060_laptop", 4.0, 1e-6
NUM = ("launches", "threads", "block", "warp_inst", "fp64_warp_inst", "gsect", "lsect", "t_ref_us",
       "host_s", "bus_bytes", "copy_s", "cpu_wall", "cuda_wall")
_DEV = {}

def meta(workload):
    """The workload's `# key: value` lines as a dict (gpu, driver, commit, date, ref)."""
    return dict(map(str.strip, l[1:].split(":", 1)) for l in lines(workload) if l.startswith("#") and ":" in l)

def lines(path):
    with open(path) as f:
        return f.read().splitlines()

def load(workload, ncu=None):
    """Deck name -> deck dict (`_deck` row fields plus `kernels`, a list of kernel row dicts)."""
    decks = {}
    for r in csv.DictReader([l for l in lines(workload) if l.strip() and not l.startswith("#")], delimiter="\t"):
        for c in NUM:
            r[c] = float(r[c]) if r.get(c) else None
        d = decks.setdefault(r["deck"], {"kernels": []})
        if r["kernel"] == "_deck": d.update(r)
        else: d["kernels"].append(r)
    for name, d in decks.items():
        p = "%s/%s.csv" % (ncu, name)
        if ncu and os.path.exists(p): apply_ncu(d, p)
    return decks

def apply_ncu(deck, path):
    """Replace counted work with ncu's per-launch means (real DRAM bytes and FP64-pipe instructions)."""
    acc = {}
    for r in csv.DictReader(open(path)):
        if not r.get("ID", "").isdigit(): continue
        a = acc.setdefault(r["Kernel Name"].split("(")[0], [])
        a.append({m: float(r[m].replace(",", "")) for m in ("smsp__inst_executed.sum", "smsp__inst_executed_pipe_fp64.sum",
                                                             "dram__bytes_read.sum", "dram__bytes_write.sum", "lts__t_bytes.sum")})
    for k in deck["kernels"]:
        a = acc.get(k["kernel"])
        if not a: continue
        mean = lambda m: statistics.fmean(x[m] for x in a)
        k.update(warp_inst=mean("smsp__inst_executed.sum"), fp64_warp_inst=mean("smsp__inst_executed_pipe_fp64.sum"),
                 dram_bytes=mean("dram__bytes_read.sum") + mean("dram__bytes_write.sum"), l2_bytes=mean("lts__t_bytes.sum"), source="ncu")

def terms(k, g):
    blocks = max(1.0, k["threads"] / k["block"]); warps = max(1.0, k["threads"] / 32)
    u = math.ceil(blocks / g["sms"]) * g["sms"] / blocks
    l2b = k.get("l2_bytes") or 32 * (k["gsect"] + k["lsect"])
    return {"fp64": k["fp64_warp_inst"] * 64 / g["fp64"] * u, "issue": k["warp_inst"] / g["issue"] * u,
            "mem": max(l2b / g["l2"], k.get("dram_bytes", 0) / g["dram"]),
            "lat": k["warp_inst"] / warps * CPI / g["f"], "floor": FLOOR}

def bind(t):
    b = max(t, key=t.get)
    return b, t[b]

def fit(k, ref):
    """(4060 binding, model seconds, e) of one kernel row, or None when it has no counts."""
    if k["warp_inst"] is None: return None
    b, v = bind(terms(k, ref))
    return b, v, v / (k["t_ref_us"] * 1e-6)

def kernel(k, g, ref, e=None):
    """(opt, cons) seconds per launch of kernel row `k` on device `g`; `e` overrides the fitted efficiency."""
    m4 = k["t_ref_us"] * 1e-6
    f = fit(k, ref)
    if f is None or f[0] in ("lat", "floor"):
        return m4, m4 * max(1.0, ref["f"] / g["f"])
    b4, v4, e4 = f
    e = e or e4
    vg = bind(terms(k, g))[1]
    opt = max(vg / e, FLOOR)
    return opt, max(opt, vg / (e / 2), vg + max(0.0, m4 - v4))

def copies(d, g, ref):
    lat = max(0.0, d["copy_s"] - d["bus_bytes"] / ref["bus"])   # per-copy latency part, kept
    return lat + d["bus_bytes"] / g["bus"]

def predict(deck, device, ref=None):
    """(opt, cons) predicted wall seconds of one deck dict from load() on one device dict."""
    ref = ref or devices()[REF]
    lo = hi = deck["host_s"] + copies(deck, device, ref)
    for k in deck["kernels"]:
        a, b = kernel(k, device, ref)
        lo += a * k["launches"]; hi += b * k["launches"]
    return lo, hi

def by_class(deck, device, ref):
    out = {}
    for k in deck["kernels"]:
        a, b = kernel(k, device, ref); m = k["t_ref_us"] * 1e-6
        s = out.setdefault(k["class"], [0.0, 0.0, 0.0]); n = k["launches"]
        s[0] += m * n; s[1] += a * n; s[2] += b * n
    return out

def devices(path=HERE + "/gpu_devices.json"):
    if path not in _DEV:
        _DEV[path] = {k: v for k, v in json.loads("\n".join(lines(path))).items() if not k.startswith("_")}
    return _DEV[path]

def check(decks, ref):
    """Round trip (exact) and held-out e: each counted kernel's e taken from the other decks that run it."""
    es = {}
    for name, d in decks.items():
        for k in d["kernels"]:
            f = fit(k, ref)
            if f: es.setdefault(k["kernel"], []).append((name, f[2], k["t_ref_us"] * k["launches"]))
    worst = {}
    print("deck                  ref wall  model/meas (unfitted)  held-out wall  err")
    for name, d in sorted(decks.items()):
        lo, _ = predict(d, ref, ref)
        assert abs(lo - d["cuda_wall"]) <= 1e-3 * d["cuda_wall"], (name, lo, d["cuda_wall"])
        held, mod, meas = d["host_s"] + d["copy_s"], 0.0, 0.0
        for k in d["kernels"]:
            m = k["t_ref_us"] * 1e-6 * k["launches"]; f = fit(k, ref)
            other = [(e, w) for n, e, w in es.get(k["kernel"], []) if n != name and n.replace("_nolu", "") != name.replace("_nolu", "")]
            if f is None or not other:
                held += m; continue
            mod += f[1] * k["launches"]; meas += m
            held += f[1] / (sum(e * w for e, w in other) / sum(w for _, w in other)) * k["launches"]
        err = held / d["cuda_wall"] - 1; lu = "LU" in d["path"]; worst[lu] = max(worst.get(lu, 0.0), abs(err))
        print("%-20s %8.2f s  %s  %10.2f s  %+5.1f%%" % (name, d["cuda_wall"], "%6.2f" % (mod / meas) if meas else "     -", held, 100 * err))
    print("round trip exact on %d decks; held-out e worst wall error %.1f%% (no device LU), %.1f%% (device LU)" % (
        len(decks), 100 * worst[False], 100 * worst.get(True, 0.0)))
    # Stated error (the A100 model study): every deck wall within 30% with e held out.
    assert max(worst.values()) <= 0.30, worst

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device"); ap.add_argument("--check", action="store_true")
    ap.add_argument("--workload", default="docs/performance/gpu-phases.tsv"); ap.add_argument("--devices", default=HERE + "/gpu_devices.json")
    ap.add_argument("--ncu")
    a = ap.parse_args()
    decks, devs = load(a.workload, a.ncu), devices(a.devices)
    ref = devs[meta(a.workload).get("ref", REF)]
    if a.check: return check(decks, ref)
    g = devs[a.device]
    print("deck                  path                      cpu s    ref s   %s s (opt..cons)   vs cpu" % a.device)
    for name, d in sorted(decks.items(), key=lambda x: x[1]["path"]):
        lo, hi = predict(d, g, ref)
        print("%-21s %-24s %7.2f %8.2f %9.2f..%-8.2f %5.2f..%.2fx" % (name, d["path"], d["cpu_wall"], d["cuda_wall"], lo, hi, d["cpu_wall"] / hi, d["cpu_wall"] / lo))
        for c, (m, x, y) in sorted(by_class(d, g, ref).items()):
            print("    %-14s kernels ref %7.3f s -> %7.3f..%.3f s" % (c, m, x, y))

if __name__ == "__main__":
    main()
