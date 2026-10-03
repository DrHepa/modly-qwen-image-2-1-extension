#!/usr/bin/env python3
"""Idempotent Modly environment installer. Never installs checkpoint assets."""
from __future__ import annotations

import datetime
import json
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys
import tempfile
import traceback
import uuid

ROOT = Path(__file__).resolve().parent
DIFFUSERS_REVISION = 'e0abab83b5df05de9e7abd788643c1a7c1e42e28'
TORCH_ARM = 'https://download-r2.pytorch.org/whl/cu130/torch-2.11.0%2Bcu130-cp311-cp311-manylinux_2_28_aarch64.whl#sha256=6304535e9e4cd1beeab449e407712602aa473a97e7b310dc5650ef50940bd94f'
VISION_ARM = 'https://download-r2.pytorch.org/whl/cu130/torchvision-0.26.0%2Bcu130-cp311-cp311-manylinux_2_28_aarch64.whl#sha256=31f87cd00c09e071980d6a4ce218289a73302ad6a7ce0b3b62a74a4081fc339d'


def log(message): print(f'[setup] {message}', flush=True)


def parse_args(argv):
    if len(argv) == 2:
        try: payload = json.loads(argv[1])
        except ValueError as exc: raise ValueError('Expected one Modly JSON object.') from exc
    elif 4 <= len(argv) <= 5:
        payload = {'python_exe': argv[1], 'ext_dir': argv[2], 'gpu_sm': argv[3],
                   'cuda_version': argv[4] if len(argv) == 5 else 0}
    else: raise ValueError('Usage: setup.py <Modly JSON> OR setup.py <python_exe> <ext_dir> <gpu_sm> [cuda_version]')
    if not isinstance(payload, dict): raise ValueError('Setup payload must be a JSON object.')
    for key in ('python_exe', 'ext_dir'):
        if not isinstance(payload.get(key), str) or not payload[key].strip(): raise ValueError(f'Missing setup {key}.')
    if Path(payload['ext_dir']).expanduser().resolve() != ROOT:
        raise ValueError('ext_dir must identify the extension directory containing this setup.py. Refusing to change another environment.')
    for key in ('gpu_sm', 'cuda_version'):
        raw = payload.get(key, 0)
        try:
            parsed = int(raw)
            if isinstance(raw, bool) or float(raw) != parsed or parsed < 0: raise ValueError()
        except (ValueError, TypeError, OverflowError): raise ValueError(f'{key} must be a non-negative integer.') from None
        payload[key] = parsed
    payload['ext_dir'] = str(ROOT)
    return payload


def select_stack(system, machine, gpu_sm, cuda_version, version):
    if tuple(version[:2]) not in ((3, 11), (3, 12)):
        raise RuntimeError(f'CPython 3.11 or 3.12 is required for this pinned environment; selected interpreter is {version}.')
    machine = machine.lower()
    if system not in ('Linux', 'Windows') or machine not in ('aarch64', 'arm64', 'x86_64', 'amd64'):
        raise RuntimeError(f'Unsupported platform {system}/{machine}; CUDA Linux ARM64/x64 and Windows x64 are the only installation candidates. Other hardware is unqualified.')
    if system == 'Windows' and machine not in ('x86_64', 'amd64'):
        raise RuntimeError('Windows ARM64 is not supported by the selected PyTorch wheel stack.')
    if gpu_sm < 80 or cuda_version < 130:
        raise RuntimeError(f'This BF16 CUDA13.0 stack requires SM80+ and a CUDA13-capable driver (reported sm={gpu_sm}, cuda={cuda_version}). Update NVIDIA driver or select a separately audited stack; no silent CPU fallback.')
    if system == 'Linux' and machine in ('aarch64', 'arm64'):
        if tuple(version[:2]) == (3, 12):
            return [
                'https://download-r2.pytorch.org/whl/cu130/torch-2.11.0%2Bcu130-cp312-cp312-manylinux_2_28_aarch64.whl#sha256=252f237d417fac3ba59b1635815c1f035a8241f2af038f2c076ed430932d89f1',
                'https://download-r2.pytorch.org/whl/cu130/torchvision-0.26.0%2Bcu130-cp312-cp312-manylinux_2_28_aarch64.whl#sha256=e2b39db78be674ee4ce7e921f54b70e5c281594c9267d981c061684ed38df936',
            ]
        return [TORCH_ARM, VISION_ARM]
    return ['torch==2.11.0+cu130', 'torchvision==0.26.0+cu130', '--index-url', 'https://download.pytorch.org/whl/cu130']


def capture(args):
    result = subprocess.run([str(x) for x in args], check=True, text=True, stdout=subprocess.PIPE,
                            stderr=subprocess.PIPE, timeout=60)
    return result.stdout.strip()


def probe_gpu(payload):
    sm, cuda = payload['gpu_sm'], payload['cuda_version']
    try:
        raw = capture(['nvidia-smi', '--query-gpu=compute_cap,driver_version', '--format=csv,noheader'])
        first = raw.splitlines()[0].split(',')
        actual_sm = round(float(first[0].strip()) * 10)
        driver = first[1].strip()
        major = int(driver.split('.')[0])
        log(f'Verified NVIDIA compute capability={actual_sm}, driver={driver}; host hints sm={sm}, cuda={cuda}.')
        sm = actual_sm
        if major >= 580: cuda = max(cuda, 130)
        elif cuda >= 130:
            raise RuntimeError(f'Host CUDA13 hint conflicts with NVIDIA driver {driver}; refusing incompatible wheels.')
    except (FileNotFoundError, subprocess.SubprocessError, ValueError, IndexError) as exc:
        log(f'NVIDIA probe unavailable ({type(exc).__name__}); validating supplied host hints and later Torch CUDA execution.')
    return sm, cuda


def run(args, env):
    log('Running: ' + ' '.join(str(x) for x in args))
    # Inherit BOTH streams: pip progress/errors remain live and cannot deadlock
    # from an undrained stderr pipe while stdout is being consumed.
    subprocess.run([str(x) for x in args], check=True, env=env)


def inspect_python(python):
    return json.loads(capture([python, '-c', 'import json,sys,platform; print(json.dumps({"version":list(sys.version_info[:2]),"system":platform.system(),"machine":platform.machine(),"implementation":platform.python_implementation(),"prefix":sys.prefix,"base_prefix":sys.base_prefix}))']))


def validate_venv_config(venv):
    """Check ownership and isolation without executing a possibly broken interpreter."""
    venv = Path(venv)
    if venv.is_symlink(): raise RuntimeError(f'Refusing symlinked venv root: {venv}. Repair will not modify an external environment.')
    if not venv.is_dir(): raise RuntimeError(f'Extension venv was not created as a directory: {venv}')
    config = venv / 'pyvenv.cfg'
    if config.is_symlink() or not config.is_file():
        raise RuntimeError(f'Owned venv requires a regular, non-symlink pyvenv.cfg: {config}')
    values = {}
    for line in config.read_text(encoding='utf-8').splitlines():
        key, separator, value = line.partition('=')
        if separator: values[key.strip().lower()] = value.strip()
    if not values.get('home') or values.get('include-system-site-packages', '').lower() != 'false':
        raise RuntimeError(f'Invalid or non-isolated pyvenv.cfg: {config}. Repair will not install into it.')


def validate_venv(venv, python):
    """Prove the intended environment boundary before any pip invocation.

    Standard bin/python symlinks to a base interpreter remain valid. The root,
    configuration and interpreter-reported prefix establish ownership instead.
    """
    venv, python = Path(venv), Path(python)
    validate_venv_config(venv)
    try:
        info = inspect_python(python)
    except (OSError, subprocess.SubprocessError, ValueError) as exc:
        raise RuntimeError(f'Cannot validate the extension venv interpreter: {python}') from exc
    prefix, base = info.get('prefix'), info.get('base_prefix')
    if not prefix or not base or Path(prefix).resolve() != venv.resolve() or Path(prefix).resolve() == Path(base).resolve():
        raise RuntimeError(f'Interpreter prefix does not identify the isolated extension venv: {python}')
    return info


def interpreter_abi(info):
    return {key: info[key] for key in ('version', 'system', 'machine', 'implementation')}


def preserve_incompatible_venv(venv):
    backup = ROOT / ('venv.incompatible-' + datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%S%f') + '-' + uuid.uuid4().hex)
    log(f'Preserving incompatible venv as {backup.name}; model directories are not touched.')
    venv.rename(backup)
    return backup


VERIFY = r'''
import importlib.metadata as metadata
import json
import torch
import torchvision
from torchvision.ops import nms
from diffusers import QwenImage21Pipeline
from transformers import AutoProcessor, AutoModelForImageTextToText
import accelerate, PIL, tokenizers, safetensors, huggingface_hub
assert torch.__version__.split('+')[0] == '2.11.0', torch.__version__
assert torchvision.__version__.split('+')[0] == '0.26.0', torchvision.__version__
assert torch.version.cuda == '13.0', torch.version.cuda
assert torch.cuda.is_available(), 'Torch CUDA is unavailable'
assert torch.cuda.is_bf16_supported(), 'CUDA device does not support BF16'
x = torch.ones((16, 16), device='cuda', dtype=torch.bfloat16)
assert (x @ x).float().sum().item() == 4096
boxes = torch.tensor([[0., 0., 2., 2.], [0., 0., 2., 2.]], device='cuda')
assert nms(boxes, torch.tensor([.9, .8], device='cuda'), .5).numel() == 1
for package, version in {'transformers':'5.17.0','tokenizers':'0.23.1','huggingface-hub':'1.32.0','safetensors':'0.8.0','accelerate':'1.11.0','Pillow':'11.1.0'}.items():
    assert metadata.version(package) == version, (package, metadata.version(package))
direct = json.loads(metadata.distribution('diffusers').read_text('direct_url.json'))
assert direct.get('vcs_info', {}).get('commit_id') == 'e0abab83b5df05de9e7abd788643c1a7c1e42e28', direct
print(json.dumps({'torch':torch.__version__, 'torchvision':torchvision.__version__, 'cuda':torch.version.cuda,
'device':torch.cuda.get_device_name(), 'capability':list(torch.cuda.get_device_capability()),
'diffusers':metadata.version('diffusers'), 'transformers':metadata.version('transformers'),
'cuda_bf16_matmul':'passed', 'torchvision_cuda_nms':'passed'}))
'''


def main(argv=None):
    payload = parse_args(sys.argv if argv is None else argv)
    selected = inspect_python(payload['python_exe'])
    if selected['implementation'] != 'CPython': raise RuntimeError('CPython is required.')
    sm, cuda = probe_gpu(payload)
    stack = select_stack(selected['system'], selected['machine'], sm, cuda, selected['version'])
    if not shutil.which('git'): raise RuntimeError('Git is required to install the immutable Diffusers commit. Install Git, then run Modly Repair.')
    venv = ROOT / 'venv'
    python = venv / ('Scripts/python.exe' if selected['system'] == 'Windows' else 'bin/python')
    env = os.environ.copy()
    env.setdefault('PIP_DEFAULT_TIMEOUT', '180')
    env.setdefault('PIP_RETRIES', '8')
    env['PYTHONUNBUFFERED'] = '1'
    if venv.exists() or venv.is_symlink():
        validate_venv_config(venv)
        if python.is_symlink() and not python.exists():
            log('The owned venv interpreter symlink is dangling; its previous base Python was removed.')
            preserve_incompatible_venv(venv)
        else:
            previous = validate_venv(venv, python)
            if interpreter_abi(previous) != interpreter_abi(selected):
                preserve_incompatible_venv(venv)
    if not python.is_file():
        log('Creating extension-owned venv.')
        run([payload['python_exe'], '-m', 'venv', str(venv)], env)
    else: log('Reusing compatible extension-owned venv (Repair is idempotent).')
    owned = validate_venv(venv, python)
    if interpreter_abi(owned) != interpreter_abi(selected):
        raise RuntimeError('Created/reused venv does not match the selected interpreter ABI; no pip command was run.')
    log('Installing pinned accelerator stack; no model weights are installed by setup.')
    run([python, '-m', 'pip', 'install', *stack], env)
    log('Installing pinned image and prompt-enhancer runtime dependencies.')
    run([python, '-m', 'pip', 'install', '-r', ROOT / 'requirements.txt'], env)
    log('Checking dependency graph.')
    run([python, '-m', 'pip', 'check'], env)
    log('Verifying pinned imports, BF16 CUDA matmul and torchvision native CUDA ops (not model inference).')
    run([python, '-c', VERIFY], env)
    # A setup marker means dependency verification only, never model readiness.
    state = {'verified_at': datetime.datetime.now(datetime.timezone.utc).isoformat(),
             'interpreter': selected, 'gpu_sm': sm, 'cuda_hint': cuda,
             'requirements_sha256': __import__('hashlib').sha256((ROOT/'requirements.txt').read_bytes()).hexdigest(),
             'tier': 'dependency-and-native-ops-verified; checkpoint-inference-not-tested'}
    with tempfile.NamedTemporaryFile('w', dir=ROOT, prefix='.setup-state-', delete=False, encoding='utf-8') as stream:
        temp = Path(stream.name)
        json.dump(state, stream, indent=2); stream.write('\n'); stream.flush(); os.fsync(stream.fileno())
    os.replace(temp, ROOT / '.setup-state.json')
    log('Setup completed. Next, download the selected node in Modly Models. Real image inference is a separate validation step.')
    return 0


if __name__ == '__main__':
    try: raise SystemExit(main())
    except Exception as exc:
        traceback.print_exc()
        print(f'[setup] ERROR: {exc}\n[setup] Fix the reported dependency/driver error and run Repair. Checkpoint data has not been downloaded or deleted.', file=sys.stderr, flush=True)
        raise SystemExit(1)
