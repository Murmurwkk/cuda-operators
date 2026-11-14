"""Capture one warmed request with Nsight Systems' cudaProfilerApi range.

Run under `nsys profile --trace=cuda,nvtx --sample=none
--capture-range=cudaProfilerApi --capture-range-end=stop ...`.
Use the resulting trace to inspect submission gaps, not as benchmark latency.
"""
import argparse
import torch
from decode_latency import one_request
from probe_parscale import load

ap = argparse.ArgumentParser(description=__doc__)
ap.add_argument('--model', required=True)
ap.add_argument('--batch', type=int, default=1)
ap.add_argument('--decode-steps', type=int, default=8)
args = ap.parse_args()
tok, model = load(args.model)
ids = tok('Hello, world. ' * 64, return_tensors='pt').input_ids[:, :128].cuda().repeat(args.batch, 1)
for _ in range(3):
    one_request(model, ids, args.decode_steps)
torch.cuda.synchronize()
torch.cuda.profiler.start()
torch.cuda.nvtx.range_push('direct-prefill-and-cached-decode')
one_request(model, ids, args.decode_steps)
torch.cuda.nvtx.range_pop()
torch.cuda.synchronize()
torch.cuda.profiler.stop()
