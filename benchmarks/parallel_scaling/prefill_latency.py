"""Prefill (time-to-first-token) cost of P parallel streams on one prompt.

Measures a full forward pass with no KV cache, at batch 1, for prompt
lengths 128..2048. This is the compute-bound half of inference that the
paper's decode-latency argument does not cover.
"""

import argparse
import json
import statistics
from pathlib import Path

import torch

from probe_parscale import TEXT, load


def timed_forward(model, ids, reps=7):
    times = []
    with torch.no_grad():
        for _ in range(2):
            model(ids, use_cache=False)
        torch.cuda.synchronize()
        for _ in range(reps):
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            model(ids, use_cache=False)
            end.record()
            torch.cuda.synchronize()
            times.append(start.elapsed_time(end))
    return {"median_ms": statistics.median(times), "min_ms": min(times), "all_ms": times}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    tok, model = load(args.model)
    base = tok(TEXT, return_tensors="pt").input_ids[0]
    rows = []
    for length in (128, 512, 1024, 2048):
        reps = (length + base.numel() - 1) // base.numel()
        ids = base.repeat(reps)[:length].unsqueeze(0).cuda()
        r = timed_forward(model, ids)
        r["tokens"] = length
        r["ms_per_1k"] = r["median_ms"] / length * 1000
        rows.append(r)
        print(f"P={model.config.parscale_n} L={length}: {r['median_ms']:.1f} ms "
              f"({r['ms_per_1k']:.1f} ms/1k tok)", flush=True)

    out = {"model": Path(args.model).name, "parscale_n": model.config.parscale_n,
           "batch": 1, "rows": rows}
    text = json.dumps(out, indent=2)
    if args.out:
        Path(args.out).write_text(text, encoding="utf-8")
    else:
        print(text)


if __name__ == "__main__":
    main()
