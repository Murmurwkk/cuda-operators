"""Direct cached decode measurements, without subtracting separate generations.

One prefill produces the first output token. Then exactly --decode-steps
cached one-token forwards produce that many additional tokens (EOS is ignored).
CUDA events cover the device timeline INCLUDING any GPU idle time while Python
submits work. Wall time includes waiting for completion. Neither is kernel-only.
"""
import argparse
import gc
import hashlib
import json
import math
import statistics
import time
from pathlib import Path

import torch

from evidence import provenance, sha256
from probe_parscale import load


def one_request(model, ids, steps, validate=False):
    torch.cuda.synchronize()
    events = [torch.cuda.Event(enable_timing=True) for _ in range(steps + 2)]
    tokens = []
    outputs = []
    wall_start = time.perf_counter()
    events[0].record()
    with torch.inference_mode():
        result = model(ids, use_cache=True, num_logits_to_keep=1)
        next_id = result.logits[:, -1].argmax(-1, keepdim=True)
        cache = result.past_key_values
        tokens.append(next_id)
        if validate:
            outputs.append(result.logits)
        events[1].record()
        for step in range(steps):
            result = model(next_id, past_key_values=cache, use_cache=True,
                           num_logits_to_keep=1)
            cache = result.past_key_values
            next_id = result.logits[:, -1].argmax(-1, keepdim=True)
            tokens.append(next_id)
            if validate:
                outputs.append(result.logits)
            events[step + 2].record()
    events[-1].synchronize()
    wall_ms = (time.perf_counter() - wall_start) * 1000
    generated = torch.cat(tokens, dim=1).cpu()
    if validate and not all(bool(torch.isfinite(logits).all()) for logits in outputs):
        raise RuntimeError('Non-finite logits in correctness pass')
    values = {
        'prefill_cuda_ms': events[0].elapsed_time(events[1]),
        'decode_cuda_ms': events[1].elapsed_time(events[-1]),
        'request_cuda_ms': events[0].elapsed_time(events[-1]),
        'request_wall_ms': wall_ms,
        'decode_step_cuda_ms': [events[i + 1].elapsed_time(events[i + 2]) for i in range(steps)],
        'generated_tokens_per_sequence': generated.shape[1],
        'output_token_sha256': hashlib.sha256(generated.numpy().tobytes()).hexdigest(),
    }
    return values, generated


def benchmark(model, tok, batch_sizes, prompt_tokens=128, steps=32, repeats=7, warmup=3):
    text_ids = tok('Hello, world. ' * (prompt_tokens + 1), return_tensors='pt').input_ids
    prompt = text_ids[:, :prompt_tokens].cuda()
    if prompt.shape[1] != prompt_tokens:
        raise ValueError('Prompt tokenizer returned too few tokens')
    rows = {}
    for batch in batch_sizes:
        ids = prompt.repeat(batch, 1)
        checked, reference_tokens = one_request(model, ids, steps, validate=True)
        with torch.inference_mode():
            generated = model.generate(
                ids, min_new_tokens=steps + 1, max_new_tokens=steps + 1,
                do_sample=False, use_cache=True, pad_token_id=tok.eos_token_id,
                eos_token_id=None)
        if not torch.equal(generated[:, prompt_tokens:].cpu(), reference_tokens):
            raise RuntimeError('Direct cached loop differs from greedy generate')
        del generated
        for _ in range(warmup):
            one_request(model, ids, steps)
        torch.cuda.synchronize()
        gc.collect()
        torch.cuda.reset_peak_memory_stats()
        samples = []
        for _ in range(repeats):
            row, output = one_request(model, ids, steps)
            if not torch.equal(output, reference_tokens):
                raise RuntimeError('Greedy output changed across repeated requests')
            samples.append(row)
        summary = {key: statistics.median(row[key] for row in samples)
                   for key in ['prefill_cuda_ms', 'decode_cuda_ms',
                               'request_cuda_ms', 'request_wall_ms']}
        summary['ms_per_batch_step'] = summary['decode_cuda_ms'] / steps
        summary['cached_decode_tokens_per_second'] = batch * steps / (summary['decode_cuda_ms'] / 1000)
        summary['full_request_tokens_per_second_wall'] = batch * (steps + 1) / (summary['request_wall_ms'] / 1000)
        summary['peak_allocated_gb'] = torch.cuda.max_memory_allocated() / 1e9
        summary['decode_step_range_ms'] = [min(row['decode_cuda_ms'] / steps for row in samples),
                                           max(row['decode_cuda_ms'] / steps for row in samples)]
        if not all(math.isfinite(v) and v > 0 for v in [summary['prefill_cuda_ms'], summary['decode_cuda_ms'], summary['request_wall_ms']]):
            raise RuntimeError('Invalid timing')
        rows[str(batch)] = {'summary': summary, 'samples': samples,
                            'correctness': {'finite_logits': True, 'greedy_generate_match': True,
                                            'repeated_tokens_match': True,
                                            'output_token_sha256': checked['output_token_sha256']}}
        print(f'P={model.config.parscale_n} batch={batch}: '
              f'{summary["ms_per_batch_step"]:.3f} ms/cached step, '
              f'{summary["cached_decode_tokens_per_second"]:.2f} tok/s', flush=True)
    return rows


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--model', required=True)
    ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--prompt-tokens', type=int, default=128)
    ap.add_argument('--decode-steps', type=int, default=32)
    ap.add_argument('--repeats', type=int, default=7)
    ap.add_argument('--warmup', type=int, default=3)
    ap.add_argument('--batch-sizes', type=int, nargs='+', default=[1, 2, 4, 8])
    args = ap.parse_args()
    if min(args.prompt_tokens, args.decode_steps, args.repeats, *args.batch_sizes) <= 0 or args.warmup < 0:
        ap.error('Counts must be positive (warmup may be zero)')
    tok, model = load(args.model)
    identity = provenance(args.model, 'cuda')
    identity['script_sha256'] = sha256(__file__)
    output = {'schema': 'direct-cached-decode-v1', 'provenance': identity,
              'parscale_n': model.config.parscale_n,
              'protocol': {'prompt_tokens': args.prompt_tokens, 'cached_decode_steps': args.decode_steps,
                           'total_generated_tokens': args.decode_steps + 1, 'warmup_requests': args.warmup,
                           'repetitions': args.repeats, 'dtype': 'bfloat16', 'attention': 'sdpa',
                           'eos': 'ignored; fixed request length',
                           'prompt_text': 'Hello, world. repeated then tokenized and sliced',
                           'timing': 'CUDA timeline including submission gaps; wall time includes completion'},
              'latency': benchmark(model, tok, args.batch_sizes, args.prompt_tokens,
                                   args.decode_steps, args.repeats, args.warmup)}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(output, indent=2), encoding='utf-8')


if __name__ == '__main__':
    main()
