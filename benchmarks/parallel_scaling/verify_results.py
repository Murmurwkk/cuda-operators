"""Check the saved ensemble statistics and optional CPU/GPU numerical agreement."""
import argparse
import itertools
import json
import math
from pathlib import Path


def validate(data):
    if data['predicted_positions'] != data['tokens'] - 1:
        raise ValueError('Wrong next-token denominator')
    for k in range(1, 9):
        records = data['subset_nll'][str(k)]
        expected = set(itertools.combinations(range(8), k))
        actual = [tuple(row['streams']) for row in records]
        if len(actual) != len(expected) or set(actual) != expected:
            raise ValueError(f'Missing or duplicate subsets for k={k}')
        values = [row['nll'] for row in records]
        if not all(math.isfinite(value) for value in values):
            raise ValueError('Non-finite NLL')
        curve = data['subset_curve'][str(k)]
        if curve['n_subsets'] != len(expected):
            raise ValueError('Wrong subset count')
        for key, value in [('min', min(values)), ('max', max(values)), ('mean', sum(values) / len(values))]:
            if not math.isclose(curve[key], value, rel_tol=1e-10, abs_tol=1e-10):
                raise ValueError(f'Inconsistent {key} for k={k}')
    return True


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--cpu', type=Path, default=Path(__file__).with_name('results') / 'ensemble_p8.json')
    ap.add_argument('--gpu', type=Path)
    ap.add_argument('--max-nll-delta', type=float, default=0.02)
    ap.add_argument('--out', type=Path)
    args = ap.parse_args()
    cpu = json.loads(args.cpu.read_text(encoding='utf-8'))
    result = {'cpu_structure_pass': validate(cpu), 'subsets': 255}
    if args.gpu:
        gpu = json.loads(args.gpu.read_text(encoding='utf-8'))
        result['gpu_structure_pass'] = validate(gpu)
        if (cpu['tokens'], cpu['predicted_positions']) != (gpu['tokens'], gpu['predicted_positions']):
            raise ValueError('CPU/GPU text lengths differ')
        deltas = [abs(a['nll'] - b['nll']) for k in map(str, range(1, 9))
                  for a, b in zip(cpu['subset_nll'][k], gpu['subset_nll'][k])]
        result['max_subset_nll_delta'] = max(deltas)
        result['gate_nll_delta'] = abs(cpu['gate_nll'] - gpu['gate_nll'])
        result['oracle_nll_delta'] = abs(cpu['oracle_nll'] - gpu['oracle_nll'])
        result['absolute_tolerance_nats'] = args.max_nll_delta
        result['cpu_gpu_pass'] = max(deltas + [result['gate_nll_delta'], result['oracle_nll_delta']]) <= args.max_nll_delta
        if not result['cpu_gpu_pass']:
            raise ValueError(json.dumps(result))
    print(json.dumps(result, indent=2))
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(result, indent=2), encoding='utf-8')


if __name__ == '__main__':
    main()
