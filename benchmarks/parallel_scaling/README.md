# ParScale: fits, checkpoint probes and direct cached decode

Run the commands from this directory. The same files are exported to
`cuda-operators/benchmarks/parallel_scaling/`. Model weights are downloaded
separately and are never included in this package.

## Environment and pinned inputs

Install a CUDA-enabled PyTorch build for the GPU first, then the remaining
dependencies in `requirements.txt`. Measurements record exact package versions,
GPU/driver, CUDA runtime, source hashes, model code and tokenizer hashes in their
JSON provenance. Both checkpoint weight hashes are checked against
`checkpoints.json`, which pins immutable Hugging Face revisions.

```sh
python -m pip install -r requirements.txt
python download_checkpoints.py --dest models
```

The downloaded model repositories contain executable custom model code. Their
fixed revisions and hashes are part of the experiment identity. `probe_parscale.py`
loads those local copies in bf16 with SDPA attention.

## Direct cached decode

```sh
python decode_latency.py --model models/ParScale-0.7B-P1-Pile --out results/decode_p1.json
python decode_latency.py --model models/ParScale-0.7B-P8-Pile --out results/decode_p8.json
python make_figs.py --only serving --out-dir figures
```

The default matrix is batch 1/2/4/8, a 128-token prompt and **32 cached one-token
forwards after prefill**. Prefill produces the first output token, so one request
produces 33 tokens in total; decode throughput uses only the 32 cached steps.
EOS does not shorten the run. Each point first checks finite logits and compares
the direct loop with greedy `generate`, then runs three warm-up requests and seven
measured requests. Repeated token outputs must agree.

Raw per-step CUDA event durations and full-request wall time are retained. CUDA
events span the device timeline, including idle time while Python submits work;
they are not sums of kernel execution time. Wall time also includes waiting for
GPU completion. The peak is PyTorch allocated memory, not total GPU/driver usage.
Model loading, tokenization and correctness checks are outside measurement.

`results/lat_p1.json` and `lat_p8.json` are retired historical outputs from a
difference-of-minima estimate, which yielded negative prefill estimates. They
must not be used for the current serving figure or exported as current results.
`probe_parscale.py --mode latency` now delegates to the direct protocol.

## Ensemble correction and CPU/GPU check

```sh
python ensemble_curve.py --model models/ParScale-0.7B-P8-Pile --device cuda --out results/ensemble_p8_gpu.json
python verify_results.py --gpu results/ensemble_p8_gpu.json --out results/ensemble_parity.json
python make_figs2.py --only ensemble --ensemble-json results/ensemble_p8.json --out-dir figures
```

`results/ensemble_p8.json` preserves the corrected CPU rerun: bf16, PyTorch 2.6.0+cpu
and transformers 4.51.3. Figure 1 uses this CPU result. Every subset is scored over
all 203 predicted positions of the same 204-token text. There are C(8,k) subsets;
for k=8 the range has zero width. The corrected best/worst pairs are 3.51/5.55.
The original version mixed text chunks into the subset range and weighted its
11-token last chunk like a 32-token chunk; its 2.63/6.33 range is retired.

The new GPU output is kept separately. `verify_results.py` validates all 255
combinations and their saved summary statistics, then compares the same subsets
across devices with a predeclared absolute tolerance of 0.02 nats. This tolerance
is a numerical parity check on this text, not an accuracy benchmark.

## Refit, stream probe and historical full-forward prefill

```sh
python fit_law.py --out results/fit_recomputed.json
python make_figs.py --only fit --out-dir figures
python make_figs2.py --only exchange --out-dir figures
python probe_parscale.py --model models/ParScale-0.7B-P8-Pile --mode probe --out results/probe_rerun_p8.json
python prefill_latency.py --model models/ParScale-0.7B-P1-Pile --out results/prefill_rerun_p1.json
python make_figs2.py --only prefill --out-dir figures
```

The paper loss tables are embedded in `fit_law.py`. No training is needed.
`make_figs.py` and `make_figs2.py` read `results/` by default and accept `--data-dir`
and `--out-dir`. To plot a new fit, put it in a separate data directory as
`fit.json` and pass that directory. The per-size gain and predicted-table indexing
have been corrected without changing the existing fitted parameters.

The historical `prefill_p*.json` measurements use a **full forward without KV
caching and with logits for every prompt position**, batch 1, lengths
128/512/1024/2048, two warm-ups and seven runs. This differs from the direct
cached-decode benchmark's prefill, which computes only the last position's logits.
The historical probe/prefill JSONs are provenance from earlier runs, not new
measurements of the revised direct loop. The article identifies each protocol.

Only the fixed text in `probe_parscale.py` is used for the stream probe. Model
training, large-scale task accuracy and the paper's hardware-model estimates are
not inputs to the local timing experiment. See the article's source table for the
paper conditions.

## Short Nsight Systems trace

```sh
nsys profile --trace=cuda,nvtx --sample=none --capture-range=cudaProfilerApi --capture-range-end=stop --force-overwrite=true --output=direct_decode_trace python profile_decode.py --model models/ParScale-0.7B-P1-Pile --batch 1 --decode-steps 8
nsys export --type sqlite --force-overwrite=true --output=direct_decode_trace.sqlite direct_decode_trace.nsys-rep
python analyze_trace.py --sqlite direct_decode_trace.sqlite --report direct_decode_trace.nsys-rep --out results/decode_trace_summary_rerun.json
```

The captured request is warmed three times before `cudaProfilerStart`. The saved
`results/decode_trace_summary.json` records 14,482 kernels, whose interval union
covers 67.665 ms of the 403.739 ms first-to-last-kernel span (16.8%). This fraction
is neither SM utilization nor proof that the remainder is Python overhead or GPU
idle time: copies, synchronization and scheduling also need timeline inspection.
The trace uses profiler instrumentation and does not replace standalone timing.
The original report and SQLite remain local because they contain host details;
the public summary keeps their report hash and source identities.
