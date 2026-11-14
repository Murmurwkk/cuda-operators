"""Content identities for local checkpoint experiments; no machine paths in output."""
import hashlib
import importlib.metadata
import platform
import subprocess
import json
from pathlib import Path


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as source:
        for block in iter(lambda: source.read(8 * 1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def provenance(model_dir, device):
    import torch
    directory = Path(model_dir)
    names = ['config.json', 'generation_config.json', 'tokenizer.json',
             'tokenizer_config.json', 'modeling_qwen2_parscale.py',
             'configuration_qwen2_parscale.py', 'model.safetensors']
    files = {name: {'sha256': sha256(directory / name),
                    'bytes': (directory / name).stat().st_size}
             for name in names if (directory / name).is_file()}
    result = {
        'model': directory.name,
        'checkpoint_files': files,
        'python': platform.python_version(),
        'platform': platform.system(),
        'packages': {name: importlib.metadata.version(name)
                     for name in ['torch', 'transformers', 'einops']},
        'device': device,
        'torch_cuda': torch.version.cuda,
    }
    manifest_path = Path(__file__).with_name('checkpoints.json')
    if manifest_path.is_file():
        pin = json.loads(manifest_path.read_text(encoding='utf-8')).get(directory.name)
        if pin:
            if files.get('model.safetensors', {}).get('sha256') != pin['weight_sha256']:
                raise RuntimeError('Checkpoint weights differ from pinned public revision')
            result['huggingface'] = pin
    if device.startswith('cuda'):
        result['gpu'] = torch.cuda.get_device_name()
        result['compute_capability'] = list(torch.cuda.get_device_capability())
        result['driver'] = subprocess.check_output(
            ['nvidia-smi', '--query-gpu=driver_version', '--format=csv,noheader'],
            text=True).strip()
    return result
