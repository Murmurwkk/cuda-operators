# Selective scan: checked CUDA implementations and measured controls

The package generates its own inputs. It contains an independent FP64 recurrence,
a failure-enforcing correctness harness, serial benchmark runner, allocation-only
adapters and CSV-driven plots. No navigation logs or private datasets are needed.

The same files work in two layouts:

- Work repository: `bench_mur/selective_scan/`, kernels in `op_mur/`.
- Public repository: `benchmarks/selective_scan/`, kernels in `selective_scan/`.

In the commands below, run from this package's directory. The scripts locate the
repository root from their own location. Python 3.10+ is sufficient for building,
measuring and checking results; only plotting needs Matplotlib (`pip install matplotlib`).
Use a CUDA toolkit supporting your GPU and a host compiler compatible with it.

## Build and reproduce

```powershell
python run.py build --arch sm_120 --results-dir out
python run.py correctness --results-dir out
python run.py sanitize --results-dir out
python run.py measure --results-dir out
python run.py sass --results-dir out
python verify_results.py --results out
python plot_timings.py --results out --out-dir figures
```

The build locates `nvcc` and, on Windows, an x64 Visual Studio developer environment.
If discovery fails, start in a developer shell. `--arch` must match the GPU.
The measured Windows setup used CUDA 13.1.115, MSVC toolset 14.51, RTX 5070 Laptop
(8151 MiB, SM 120), driver 610.88, and the flags stored in `results/manifest.json`.
CUDA 13.1 does not officially support that MSVC version; the build records the
required `-allow-unsupported-compiler` and `_ALLOW_COMPILER_AND_STL_VERSION_MISMATCH`
compatibility switches instead of hiding them. Linux does not use these switches.

`results/` is the checked-in reference run. Use `--results-dir out` for a fresh run
so it does not mix with that reference. Binaries, compiler logs and full SASS dumps
are local build products under `build/`; they are not committed. A missing binary,
CUDA failure, invalid result or incomplete sanitizer summary is a hard failure.

Sanitizer injection may need permission to launch/instrument a child process on
Windows. Run `sanitize` with those permissions if the target cannot be launched;
an unsuccessful tool launch is never recorded as a pass. No global driver setting
is changed by these commands.

## Correctness before timing

`bench.cpp` evaluates the recurrence independently in double precision. Each
invocation poisons the output and validates two calls before warming or timing.
It rejects NaN/Inf, missing output writes and any element with
`abs(actual-reference) > 2e-5 + 2e-4 * abs(reference)`. The output is a JSON row;
failed validation exits nonzero and emits no performance row. `--self-test`
checks that NaN, infinity and an above-tolerance result are rejected.

The separate correctness sweep uses 13 boundary shapes, three seeds (17/29/101)
and six implementations: 234 cases. It includes odd N, N=64/65, partial chunks,
L=1023/1024/1025 and more than one batch. `sanitize` runs memcheck, racecheck and
synccheck for N=32/64/65/128 at B=1,L=33,D=7. Its summary parser requires zero
reported errors and, for racecheck, zero warnings/hazards as well.

Random inputs use mt19937: u/B/C/skip uniform in [-0.5,0.5], delta in [0.01,0.06],
A in [-0.6,-0.1]. Main measurements use seed 17. The finite input range and the
absolute/relative tolerance are part of the experiment, not a guarantee for every
possible floating-point input.

## Timing and controls

Every cell uses six warm-up calls, 30 timed solve calls and three independent
process rounds. Variant order is deterministically shuffled between rounds.
The device column is the CUDA-event interval per solve; it includes gaps between
launches, allocation/free effects and all kernels inside the solve interface.
The wall column covers the same batch through GPU completion. Neither column
should be described as a sum of kernel durations.

`measure` runs these suites in sequence, never concurrently:

| Suite | Conditions | Purpose |
|---|---|---|
| matrix | Original 12 shapes, six implementations | Re-evaluate the whole-call ranking after the synchronization fix |
| workspace | B=1,D=N=64,L=2048/4096/8192; seq/improved with per-call and caller-owned workspace | Isolate allocation policy without editing the kernels |
| chains | B=1,N=64,L=1024/8192,D=1/2/4/8/16/32/64; six implementations | Test how chain count changes the serial/parallel comparison |

Each suite retains every process result in `*_raw.csv`, plus median/min/max
summaries. Figure bands are the three-run ranges, not confidence intervals.
`verify_results.py` checks the exact shape/variant/round coverage, finite data,
error limits, recomputed summaries and winners. It prints a JSON report that can
also be saved with `--out <file>`.
The three default suites do not require historical results. If either historical
CSV exists, both the raw and summary CSV must be present and valid. Pass
`--require-history` to require the historical comparison explicitly as well.

The allocation adapters include the original translation units. `seq_workspace.cu`
reuses the same four scan kernels but allocates their four buffers once before
validation/timing. `improved_workspace.cu` calls the existing
`solve_with_workspace_stream` API with one caller-owned buffer. Their preparation
and destruction are outside the timed region; the original interfaces include
allocation/free as before. All four variants run the same independent oracle.

## Historical synchronization audit (explicit opt-in)

`historical/ssm_before_sync.cu` is an **unsafe historical source snapshot**. It lacks
the cross-lane initialization ordering fixed by `__syncwarp()` in both warp paths.
It is supplied only so the before/after comparison can be reproduced; it is not
compiled or benchmarked by the default commands.

```powershell
python run.py build --arch sm_120 --include-unsafe-history --results-dir out
python run.py measure --suite sync_comparison --include-unsafe-history --results-dir out
python verify_results.py --results out --require-history
```

The saved comparison uses the same harness/inputs and interleaves old and new
executables. Corrected/original median ratios span 0.845–1.134 over the 12 shapes.
The L=8192 ratio is 1.134; the run does not establish zero overhead. Passing a
numerical sample also does not make the historical source race-free.

## Short diagnostic profiles

Install Nsight Systems (`nsys`), Nsight Compute (`ncu`) and CUDA Compute Sanitizer
in addition to the build toolchain. After the normal build, run:

```powershell
python profile.py --output-dir build/profiles --results-dir out
```

This traces the four allocation variants for B=1,L=2048,D=N=64, exports SQLite
reports and extracts allocation/launch call counts. It also tries one NCU capture
using existing counter permissions; a counter permission failure is recorded,
not treated as measured stall evidence. Profiles are diagnostic runs and never
replace the unprofiled timing tables. Full reports stay in the chosen local
output directory; the compact `profiling.json` is written to `--results-dir`.

The default profile command does not run the unsafe historical binary. To repeat
the explicit historical Racecheck audit after building it with the option above:

```powershell
python profile.py --include-unsafe-history --output-dir build/profiles --results-dir out
```

Every required binary is checked before profiling starts. The saved historical
audit reported 1,736 warnings; the corrected sanitizer cases reported zero hazards.

## Provenance and publication

`manifest.json` records CUDA/host compiler versions, GPU/driver, exact flags and
SHA-256 identities of sources, adapters, harness, runner and measured binaries.
Paths are normalized to `$ROOT`, with no usernames or machine-specific source
locations. The binary hashes identify the measured executables; fresh builds may
differ because toolchains and embedded paths differ. `sass.csv` counts addressed
instructions only, excluding encoding-continuation lines.

The blog's plots can be regenerated from the committed summaries with
`python plot_timings.py`; in the work repository they default to the article's
asset directory, and in the public repository to a local figures directory.
On this measurement machine, an existing Python 3.12 write restriction required
writing figures to a writable output directory and then copying them to the blog;
this does not change the CSV data or plotting code.
