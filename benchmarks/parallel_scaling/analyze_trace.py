"""Export a path-free summary of one marked Nsight Systems request."""
import argparse
import json
import sqlite3
from pathlib import Path
from evidence import sha256

ap = argparse.ArgumentParser(description=__doc__)
ap.add_argument('--sqlite', type=Path, required=True)
ap.add_argument('--report', type=Path, required=True)
ap.add_argument('--out', type=Path, required=True)
args = ap.parse_args()
db = sqlite3.connect(args.sqlite)
start, end = db.execute("SELECT start,end FROM NVTX_EVENTS WHERE text='direct-prefill-and-cached-decode'").fetchone()
intervals = db.execute('SELECT start,end FROM CUPTI_ACTIVITY_KIND_KERNEL WHERE start>=? AND end<=? ORDER BY start', (start,end)).fetchall()
merged = []
for left, right in intervals:
    if merged and left <= merged[-1][1]:
        merged[-1][1] = max(merged[-1][1], right)
    else:
        merged.append([left, right])
busy = sum(right-left for left,right in merged)
span = intervals[-1][1] - intervals[0][0]
summary = {
    'model': 'ParScale-0.7B-P1-Pile', 'batch': 1, 'prompt_tokens': 128,
    'cached_decode_steps': 8, 'warmup_requests': 3,
    'nsys': '2026.3.1', 'trace': 'cuda,nvtx', 'sample': 'none',
    'capture_range': 'cudaProfilerApi',
    'report_sha256': sha256(args.report),
    'profile_script_sha256': sha256(Path(__file__).with_name('profile_decode.py')),
    'decode_script_sha256': sha256(Path(__file__).with_name('decode_latency.py')),
    'marked_request_wall_ms': (end-start)/1e6,
    'kernel_count': len(intervals),
    'first_to_last_kernel_ms': span/1e6,
    'kernel_interval_union_ms': busy/1e6,
    'kernel_interval_union_fraction': busy/span,
    'interpretation': 'Trace-only: fraction of first-to-last-kernel span covered by kernel intervals; not SM utilization or DRAM utilization. Profiling timings do not replace independent benchmark results.'
}
args.out.parent.mkdir(parents=True, exist_ok=True)
args.out.write_text(json.dumps(summary, indent=2), encoding='utf-8')
print(json.dumps(summary, indent=2))
