"""Model loading and reproducible experiment configuration."""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
METHODS = ('baseline', 'additive', 'l2s', 'spherical', 's2s_l', 's2s_r')
MODELS = {
    'qwen05': ('Qwen/Qwen2.5-0.5B-Instruct', '7ae557604adf67be50417f59c2c2f167def9a775', 16),
    'qwen15': ('Qwen/Qwen2.5-1.5B-Instruct', '989aa7980e4cf806f80c7fef2b1adb7bc71aa306', 20),
    'qwen3': ('Qwen/Qwen2.5-3B-Instruct', 'aa8e72537993ba99e69dfaafa59ed015b17504d1', 22),
    'qwen14': ('Qwen/Qwen2.5-14B-Instruct', 'cf98f3b3bbb457ad9e2bb7baf9a0125b6b88caa8', 32),
    'qwen35': ('Qwen/Qwen3.5-9B', 'c202236235762e1c871ad0ccb60c8ee5ba337b9a', 21),
    'qwen32': ('Qwen/Qwen2.5-32B-Instruct', '5ede1c97bbab6ce5cda5812749b4c0bdf79b18dd', 43),
    'nemotron_nano8b': ('nvidia/Llama-3.1-Nemotron-Nano-8B-v1', '54641c1611fcff44fa4865626462445e0a153fc7', 21),
}


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n')
    tmp.replace(path)


def fingerprint(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def model_identity(location, repo, revision):
    path = Path(location)
    if not path.is_dir():
        if location != repo:
            raise ValueError('Use the declared official Hub repo or a local downloaded directory')
        return dict(repo_id=repo, revision=revision)
    receipt = path / 's2s_download.json'
    if receipt.exists() and json.loads(receipt.read_text()) != dict(repo_id=repo, revision=revision):
        raise ValueError('Downloaded model receipt does not match declared model/revision')
    identity = {}
    for file in sorted(path.glob('*')):
        if file.name != 's2s_download.json' and file.suffix in ('.json', '.safetensors', '.model', '.txt', '.jinja', '.tiktoken'):
            stat = file.stat()
            identity[file.name] = dict(size=stat.st_size, mtime_ns=stat.st_mtime_ns)
    return dict(directory=str(path.resolve()), files=identity,
                receipt=json.loads(receipt.read_text()) if receipt.exists() else None)


def source_identity():
    files = [*Path(__file__).parent.glob('*.py')]
    for directory in ('methods/S2S_R', 'methods/l2s_linear', 'methods/spherical_steering'):
        files.extend((ROOT / directory).glob('*.py'))
    files.extend([ROOT / 'methods/inference_residual_noise.py', ROOT / 'truthfulqa/models.py',
                  ROOT / 'truthfulqa/utilities.py', ROOT / 'data/truthfulqa/TruthfulQA.csv'])
    return fingerprint({str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
                        for p in sorted(files)})


def parameter_grid(method, tune):
    if method == 'baseline':
        return [{}]
    if method in ('additive', 'l2s'):
        values = [0.5, 1.0, 1.4, 2.0] if tune else [1.4 if method == 'l2s' else 1.0]
        return [{'alpha': x} for x in values]
    if method == 'spherical':
        return [dict(kappa=k, alpha=a, beta=b)
                for k in ([1.0, 3.0, 5.0, 10.0, 20.0, 50.0] if tune else [20.0])
                for a in ([0.15, 0.3, 0.6] if tune else [0.6])
                for b in ([0.1, 0.4] if tune else [0.4])]
    if method == 's2s_l':
        return [dict(strength=s, slerp_t=t)
                for s in ([0.5, 1.0, 2.0, 3.0] if tune else [1.0])
                for t in ([0.0, 0.2, 0.5, 0.8, 1.0] if tune else [0.5])]
    if method in ('s2s', 's2s_r'):
        return [dict(weight_kappa=k, margin=m)
                for k in ([1.0, 3.0, 5.0] if tune else [3.0])
                for m in ([-0.7, -0.3, 0.0] if tune else [-0.7])]
    raise ValueError(method)


def decoder_layers(model):
    for path in ('model.layers', 'model.language_model.layers', 'language_model.layers'):
        try:
            layers = model.get_submodule(path)
            if len(layers):
                return path, layers
        except (AttributeError, KeyError):
            pass
    raise ValueError('Unsupported decoder structure; refusing to hook an unrelated module')


def execution_device(module):
    """Resolve Accelerate execution devices, including CPU/disk offload hooks."""
    import torch
    def from_hook(hook):
        if hook is None:
            return None
        value = getattr(hook, 'execution_device', None)
        if value is not None and str(value) != 'meta':
            return torch.device('cuda', value) if isinstance(value, int) else torch.device(value)
        for child in getattr(hook, 'hooks', ()):
            value = from_hook(child)
            if value is not None:
                return value
        return None
    value = from_hook(getattr(module, '_hf_hook', None))
    if value is not None:
        return value
    # Offload can be attached to individual projections inside a decoder block.
    for child in module.modules():
        value = from_hook(getattr(child, '_hf_hook', None))
        if value is not None:
            return value
    for parameter in module.parameters():
        if parameter.device.type != 'meta':
            return parameter.device
    raise RuntimeError('Cannot determine execution device for a fully offloaded module')


def load_model(config):
    import torch
    from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
    if config['device'] == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA requested but unavailable; run outside the read-only sandbox/on a GPU host')
    location = config['model_dir']
    # Hub downloads are pinned; downloaded directories carry an explicit receipt.
    kwargs = {} if Path(location).is_dir() else {'revision': config['revision']}
    hf_config = AutoConfig.from_pretrained(location, trust_remote_code=False, **kwargs)
    tokenizer = AutoTokenizer.from_pretrained(location, trust_remote_code=False, **kwargs)
    loader = AutoModelForCausalLM
    if hf_config.model_type == 'qwen3_5':
        from transformers import Qwen3_5ForConditionalGeneration
        loader = Qwen3_5ForConditionalGeneration
    options = dict(dtype=getattr(torch, config['dtype']), low_cpu_mem_usage=True,
                   trust_remote_code=False, **kwargs)
    if config['device'] == 'cuda':
        # Reserve space for logits/temporary activations. Shared GPUs are not cleared.
        # The default guard leaves room for temporary tensors on shared GPUs.
        # A single-card run may opt out when the full BF16 model otherwise fits.
        reserve_gib = float(__import__('os').environ.get('S2S_GPU_RESERVE_GIB', '3'))
        if reserve_gib < 0:
            raise ValueError('S2S_GPU_RESERVE_GIB must be non-negative')
        free = {i: max(0, int(torch.cuda.mem_get_info(i)[0]) - int(reserve_gib * 1024**3))
                for i in range(torch.cuda.device_count())}
        options.update(device_map='auto', max_memory={**free, 'cpu': config['cpu_memory']},
                       offload_folder=str(Path(config['output_dir']) / 'offload'), offload_state_dict=True)
    if config['load_in_4bit']:
        from transformers import BitsAndBytesConfig
        options['quantization_config'] = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_quant_type='nf4', bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=getattr(torch, config['dtype']))
    model = loader.from_pretrained(location, **options).eval()
    if not config['load_in_4bit']:
        expected = getattr(torch, config['dtype'])
        wrong = {str(p.dtype) for p in model.parameters() if p.is_floating_point() and p.dtype != expected}
        if wrong:
            raise RuntimeError(f'Base model precision mismatch: expected {expected}, found {wrong}')
    if config['device'] == 'cpu':
        model.to('cpu')
    path, layers = decoder_layers(model)
    if not 0 <= config['layer'] < len(layers):
        raise ValueError(f"Invalid layer {config['layer']} for {len(layers)} blocks")
    return model, tokenizer, path, layers


def aggregate(reports):
    """Question-weighted metrics; reject overlapping/incomplete method coverage."""
    rows = []
    coverage = None
    for method in METHODS:
        items = [r for r in reports if r['method'] == method]
        if {r['fold'] for r in items} != {0, 1} or len(items) != 2:
            raise ValueError(f'Need exactly two folds for {method}')
        ids = [q for r in items for q in r['question_ids']]
        if not ids or len(set(ids)) != len(ids):
            raise ValueError('Held-out questions must be nonempty and disjoint')
        if coverage is None:
            coverage = set(ids)
        if set(ids) != coverage:
            raise ValueError('All methods must evaluate exactly the same questions')
        n = len(ids)
        metrics = {key: sum(r['metrics'][key] * len(r['question_ids']) for r in items) / n
                   for key in ('MC1', 'MC2', 'MC3')}
        if not all(math.isfinite(v) and 0 <= v <= 1 for v in metrics.values()):
            raise ValueError('Invalid MC metrics')
        rows.append(dict(method=method, n_questions=n, **metrics,
                         Avg=sum(metrics.values()) / 3))
    return rows
