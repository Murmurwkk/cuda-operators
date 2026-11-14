"""Download pinned public models; weights remain outside the reproduction package."""
import argparse
import json
from pathlib import Path
from huggingface_hub import snapshot_download

ap = argparse.ArgumentParser(description=__doc__)
ap.add_argument('--dest', type=Path, default=Path('models'))
ap.add_argument('--streams', type=int, nargs='+', choices=[1, 8], default=[1, 8])
args = ap.parse_args()
pins = json.loads(Path(__file__).with_name('checkpoints.json').read_text(encoding='utf-8'))
for streams in args.streams:
    name = f'ParScale-0.7B-P{streams}-Pile'
    pin = pins[name]
    snapshot_download(pin['repo_id'], revision=pin['revision'], local_dir=args.dest / name,
                      allow_patterns=['*.json', '*.py', '*.safetensors'])
