#!/usr/bin/env python3
"""Download pinned official weights, with resumable Hub transfers and receipts."""
import argparse
import json
from pathlib import Path
import sys
import time
from concurrent.futures import ThreadPoolExecutor

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from experiments.model_scaling_ablation.common import MODELS, write_json


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--models', nargs='+', choices=MODELS, default=list(MODELS))
    p.add_argument('--model-root', type=Path, default=ROOT / 'models')
    p.add_argument('--workers', type=int, default=4)
    args = p.parse_args()
    from huggingface_hub import snapshot_download
    def download_one(key):
        repo, revision, _ = MODELS[key]
        dest = args.model_root.resolve() / repo.split('/')[-1]
        receipt = dest / 's2s_download.json'
        for attempt in range(1, 5):
            try:
                print(f'Downloading {repo}@{revision} -> {dest} (attempt {attempt})', flush=True)
                snapshot_download(repo, revision=revision, local_dir=str(dest), max_workers=args.workers,
                                  allow_patterns=['*.json', '*.safetensors', '*.model', '*.txt', '*.jinja', '*.tiktoken'])
                index_path = dest / 'model.safetensors.index.json'
                if index_path.exists():
                    weights = set(json.loads(index_path.read_text())['weight_map'].values())
                else:
                    weights = {'model.safetensors'}
                missing = [name for name in weights if not (dest / name).is_file()]
                if missing:
                    raise ValueError(f'Missing weight shards: {missing}')
                from safetensors import safe_open
                for name in weights:
                    with safe_open(str(dest / name), framework='pt', device='cpu') as weight:
                        if not list(weight.keys()):
                            raise ValueError(f'Empty weight file: {name}')
                write_json(receipt, dict(repo_id=repo, revision=revision))
                print(f'Complete: {dest}', flush=True)
                break
            except Exception:
                if attempt == 4:
                    raise
                time.sleep(min(30, 5 * attempt))
    # Independent downloads do not delay the larger model behind the first.
    with ThreadPoolExecutor(max_workers=len(args.models)) as pool:
        list(pool.map(download_one, args.models))


if __name__ == '__main__':
    main()
