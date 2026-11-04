# Changelog

Newest first.

## Unreleased

- HSPICE: a `+` line with a blank after the `+` (`+ x = '1u \\`) continues at its trailing `\\` or ` \`, so a quoted expression broken across lines keeps its plus signs; such a card failed with `ParseError` (#263).
- HSPICE: `.dc v1 0 1 0.1 MONTE=5` (a bare `MONTE=` or `DATA=` after the swept source, outside HSPICE's `.DC` forms) is refused as `UnsupportedCard` with the reason and the `SWEEP MONTE=` form to write; it failed later as `AnalysisSourceNotFound`, reading `monte` as a second source (#263).

## 0.20.0 (2026-11-02)

0.3.0 was never released; its entries are folded in below.

- `auto`'s GPU estimate prices each device type by its instance count (one GPU thread runs ~120x slower than a CPU core on the tested RTX 4060 Laptop) and by the waveform history it holds (`LTRA`, `CPL`, `TXL`), so a single line stays on the CPU and a batch of a thousand goes to the GPU; under `--backend cuda`, where every device still runs on the GPU, `tran/ltra_tran_long_run_past_8192` took 1973 s and `tran/cpl_tran_history_past_2048` 317 s against 11.0 s and 0.58 s on the CPU, and a `note:` now says so. The release CUDA corpus runs `--backend auto`.
- The repository is github.com/OpenFAT/ESPice (it was OpenFAT/EGSpice; GitHub redirects the old URLs, and `nix run github:OpenFAT/EGSpice` still resolves).
- The source is reorganised by owner (frontend, core, device, solver, analysis, output, the facade and the C API), every file under 500 code lines; all 951 conformance decks give byte-identical output before and after.
- Runtime `.hdl` builds print each model's VerA warnings and its `loader: compiling` line in the deck's file order once the builds finish, so stderr is the same on every run (it followed build completion order before).
- A loaded VPI application's design, registries and paths are freed at exit; the corpus runner fails a run whose allocator reports a leak.
- A built device library that will not open logs the system's reason (`loader: cannot open <path>: <reason>`) beside the error.
- The merge gate runs the allocation-failure suites with `-Dcertify=true`, so a certificate that panics on out-of-memory (as before #244) fails a pull request instead of the nightly.
- The test that a Problem run batch after batch costs the same per batch counts allocation calls instead of comparing wall times, which failed once on a loaded machine (712 ms early, 2355 ms late); the C API batch test drops its wall-time and RSS checks.
- A cold operating point whose gmin stepping crawls (its step factor under 1.122 for 20 solves in a row) moves on to source stepping: the `.op` of a 100k-transistor BSIM4 chain converges in 14:07 with a 2.8 GiB peak RSS, where it was still crawling the gmin ladder when killed at 32 min (#218).
- `.wcd m1 '>= L1' m2 '<= L2' ...` finds each spec's worst-case distance β_w over the statistical parameters `SWEEP MONTE` draws, by FORM (iHL-RF, tangent rows for `.dc` and `.tran`), and reports it with Φ(β_w), the iterations, the simulations and the worst-case point; an unconverged search says so and never prints a guess; `--measures` writes it as `"kind": "wcd"` (#240).
- `.four` and `.fft` read the deck's `.tran` result instead of integrating the same transient again, where that run keeps its waveform with the same options and operating point (CPU backend); the values are bit for bit the same (#231).
- `--emir=FILE` writes each device terminal's and independent source's average, EM_AVG, RMS and peak current over the deck's first `.tran` (or `--emir-window=T1:T2`), else its `.op`/`.dc` point, as JSON with each instance's `.model` name, one file per `.step`/`.alter`/sweep variant; a Spectre analysis's `emirformat`/`emirfile` sets it (#237).
- A mixed batch (some instances fully collapsed) no longer names its instances, `@m[...]` variables or swept parameters by add order: cards are renumbered to the frozen batch order (#237).
- `SWEEP MONTE=n SIGMA_SCALE=s1,... FAIL='expr'` (scaled-sigma sampling) runs n trials per sigma scale on `.tran`, `.ac` or a swept `.dc`, counts the trials whose `.meas` results fail, and extrapolates the failure probability to the real sigma with its 95 % interval and equivalent sigma; `--measures` writes it as `"kind": "sss"` (#239).
- `--measures=FILE` also writes every `.meas` result as JSON, labelled with its analysis and variant (`.step`, `SWEEP DATA`, `.alter`, Monte Carlo trial, `.temp` entry), a failed one as `null` with its error; `--measures=-` puts it on stdout (#235).
- `.option soalimits="file.json"` gives SOA limits, by model name or prefix, to the models whose cards state none; `.meas ... param='soa_count()'` counts a result's excursions, and `--measures` writes each as an object under `soa` (#238).
- `.option soa=1` (ngspice `warn=1`) checks every accepted point of an operating point, DC sweep and transient against the model-card SOA limits (`vds_max`, `vbe_max`, `bv_max`, ...), one line per excursion, `maxwarns` of them, then the count (#238).
- `.save @m1[gm]` and `.meas ... find @m1[id]`: MOSFET operating-point variables (terminal currents, gm, gds, gmbs, capacitances, ft, terminal voltages) at the operating point, along `.dc` and in `.tran`; an unknown name refuses the deck instead of being dropped (#236).
- `.save @m1[vth]` and `.meas ... find @m1[gm]`: device operating-point variables at the operating point, along `.dc` and in `.tran`: a model's own `desc` or `units` variables (`vth`, `vdsat`, `region`, BSIM-CMG's `ids`) and a MOSFET's terminal quantities (currents, gm, gds, gmbs, capacitances, ft, terminal voltages); an unknown name refuses the deck instead of being dropped (#236).
- A pfet's `@m[id]`, `is`, `ig`, `ib`, `vgs`, `vds` and `vbs` are in ngspice's frame (type × the current into the terminal and × the voltage), so a saturated pfet's `id` is positive as ngspice's; BSIM4's `@m[cgg]` is its own intrinsic `cggb`, without the overlap, as ngspice's, while `ft` keeps the total gate capacitance (#236).
- HSPICE `.probe op` (every device's operating-point variables) and `.probe lx7(m1)`/`lv9(m1)` (the MOSFET element templates lx1-lx4, lx7-lx9, lx18, lv9, lv10), Spectre `save m1:gm x1.mn1:vdsat`, and the same names in `.meas`; BSIM4 (`bsim4va`) gains `vth` and `vdsat` (#236).
- `espice --version` prints the ESPice and VerA versions and the GPU backends built in.
- `-I DIR` / `--include=DIR` adds a search directory for `.hdl` includes.
- `espice_release_points` also frees each point's parameter row and label (#225).
- `.pz` on circuits above 1000 unknowns returns the 20 poles and 20 zeros nearest s = 0 (the plot is named "partial").
- A `.tran ... uic` run no longer solves an operating point it does not use: a 100k-transistor chain started in seconds instead of hours.
- Operating points: gmin stepping and source stepping follow ngspice more closely; a few decks that needed them converge with slightly different iteration counts.
- Faster and leaner on large decks: quicker sparse LU, faster parsing of large hierarchies (a 100k-transistor chain parses in half the memory), CSV and `.print` sweeps stream to disk (a 200-point sweep: ~1 GB → ~50 MB), and Monte Carlo shares unchanged model rows (1.04 → 0.19 GB on a 1k chain).
- The compact models ship in the source tree; building no longer downloads them.
- Cached `.hdl` builds rebuild once (new device ABI).
- VerA v0.23.0 and Gompute 1.5.0.
- A `.hdl` model's `$strobe`/`$display` prints `%e` with a precision of 59 or more as the number instead of `?` (VerA v0.21.0, OpenFAT/VerA#98).
- `.hdl` models VerA now refuses: an analog assignment to a `reg`, or to a bit or part of one (E0316), `ac_stim` or `flicker_noise` with a fourth argument (E0506), and a protected model that needs a license (E0146). Inputs that crashed VerA (a 5000-arm `case`, a 5000-deep `begin` nest, a NaN noise-table knot) are a coded refusal or compile (VerA v0.21.0-v0.22.0, OpenFAT/VerA#114, #116).
- `.hdl` models VerA now accepts: an instance whose two ports share a net (`res r1(a, a)`, was E0315) and a read of the flow into an instance's own port (`I(<a>)`, was E0997) (VerA v0.21.0-v0.22.0, OpenFAT/VerA#116).
- GPU lane batches also size their waves to each device's free memory, so a batch too large for the card runs in smaller waves instead of on the CPU (#217).
- Releases attach `.deb`, `.rpm` and Arch packages (`espice`; `espice-cuda` for NVIDIA GPUs; `espice-hip` with a build per AMD GPU target, picked at start), an AUR `PKGBUILD` (`espice-bin`) and a Homebrew formula; `tools/install.sh` installs a release from `sources.json` with a checked, cached download; `nix run .#latest` and `.#"<version>"` install release binaries.
- `nint(-0.5)` is -0, as in ngspice: `{1/nint(-0.5)}` folded to +inf instead of -inf.
- Removed: the `ESPICE_LINEAR_CACHE` variable, the `-Dbsim4-narrow` build option and the `test-gpu-*` build steps (`test-gpu` runs the corpus on CUDA).

- Sweeps (`.step`, `.mc`, `.dc DATA=`/`MONTE=`, `.temp` ranges) run their points in parallel across `--jobs`, about 2x faster with flat memory; output is the same at every `--jobs`. Monte Carlo draws differ from 1.1 (#213).
- New CLI defaults: `--jobs` is one per CPU, and `--measures-only`, `--early-stop` and `--warm-start` turn on by themselves when nothing reads the transient waveforms; `--full-waveforms` turns them off. `--tokenizer` detects the dialect.
- `--measures-only`: `.meas` without storing waveforms (a 1000-row sweep: 6.5 GB → 168 MB). `--early-stop` ends a `.tran` once every `.meas` resolves; `--abort-after`/`--abort-measure` fail a point that has not.
- `--tangents=P,...` prints d(.meas)/d(P); `OPTIMIZE=` on `.tran` uses them for a cleaner Jacobian (#97).
- C API points: run many parameter points on one template and read each point's measures, tangents and status (`espice_append_points` and friends).
- `--lanes` batches sweep points (bitwise equal on the CPU); on the GPU each sweep point of a transient runs as one thread (#207–#211).
- `.meas op ... find i(vdd)` reads from the operating point.
- VerA v0.11.4: a user `.hdl` model writing `- -x` needs `-(-x)`. `.hdl` reads Verilog-AMS 2023, and 2.3-era models compile inside `begin_keywords "VAMS-2.3"`.
- A title-only deck succeeds, as in ngspice; espice.wasm works again (#8).

## 0.2.0 (2026-10-19)

- **Dialects:** Spectre netlists (`--tokenizer=spectre`); HSPICE quoting, braces, `$` comments and operators follow the HSPICE manual (#203); many ngspice parsing details match ngspice (#121).
- **New:** VPI applications (`--vpi app.so`) for Verilog-A `$systf` calls and analog callbacks (#197, #198); `.dc` over a `.param`; `.vec` files; `.qpss` on multi-tone HB; `OPTIMIZE=` grid steps; `live_params` / `set_params` to change `.param` values without re-parsing (#101).
- **Output:** `.hdl` display tasks print once per accepted point; Touchstone 2.0 with per-port z0; CSV, PSF and CITIfile fixes for downstream tools.
- **GPU:** BSIM4, VBIC, MOS 1–9, sources, lines and filters evaluate on the GPU; the device LU takes large post-layout circuits and is 4x faster there (#139–#141).
- **Accuracy:** dozens of fixes to match ngspice 45 on devices (MOS, BSIM2, BSIM4, JFET, diode, resistor TC), noise, `.tf`, `.sens`, `.four`, `.ac` sweep points and sources (#44–#168).
- **Robustness:** bad decks fail with an error naming the card instead of running wrong; out-of-memory and Newton runaway are reported instead of publishing zeros.
- **Models:** third-party compact models (PSP, BSIM-CMG/BULK/IMG/SOI, HiSIM, HICUM, Mextram, ASM-HEMT, MVSG and more) from their upstream releases, unmodified.
- `ESPICE_THREADS` gives the same output at any thread count (#137).
- Nix flake (`nix run github:OpenFAT/ESPice`, CUDA package), the playground (`zig build wasm`) and the user documentation site.

## 0.1.0 (2026-10-05)

- First release: Zig 0.17, Apache-2.0 (models keep their own licences).
- sky130 PDK decks run as written; BSIM4 has ngspice-style limiting.
- Verilog-A models see gmin and source stepping through `$simparam`, unlike ngspice.
- `--backend cuda|hip` fails if no usable GPU is found; `auto` falls back to the CPU.
- About 90 hardening fixes; 797 of 799 corpus decks pass.
- `zig build bench` and `bench-suites` compare against ngspice and VACASK.
