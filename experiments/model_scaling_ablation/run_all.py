#!/usr/bin/env python3
"""Durable sequential job: download -> real-weight smoke -> full comparison."""
import argparse
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
from experiments.model_scaling_ablation.common import MODELS, write_json


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model-root', type=Path, required=True)
    p.add_argument('--job-dir', type=Path, required=True)
    p.add_argument('--load-in-4bit', action='store_true')
    p.add_argument('--dtype', choices=['bfloat16', 'float16', 'float32'], default='bfloat16')
    p.add_argument('--cpu-memory', default='40GiB')
    p.add_argument('--no-tune', action='store_true')
    p.add_argument('--models', nargs='+', choices=MODELS, default=list(MODELS))
    args = p.parse_args()
    job = args.job_dir.resolve()
    job.mkdir(parents=True, exist_ok=True)
    write_json(job / 'job.json', dict(pid=os.getpid(), models=args.models,
                                     model_root=str(args.model_root.resolve()),
                                     load_in_4bit=args.load_in_4bit, dtype=args.dtype, cpu_memory=args.cpu_memory, tune=not args.no_tune))
    for key in args.models:
        stages = [('download', [sys.executable, str(HERE / 'download.py'), '--models', key,
                                '--model-root', str(args.model_root)])]
        common = [sys.executable, str(HERE / 'run.py'), '--model', key, '--model-dir',
                  str(args.model_root / MODELS[key][0].split('/')[-1]),
                  '--dtype', args.dtype, '--cpu-memory', args.cpu_memory]
        if args.load_in_4bit:
            common.append('--load-in-4bit')
        if args.no_tune:
            common.append('--no-tune')
        stages += [('smoke', common + ['--smoke']), ('full', common)]
        for stage, command in stages:
            log_path = job / f'{key}_{stage}.log'
            write_json(job / 'status.json', dict(state='running', model=key, stage=stage,
                                                 log=str(log_path), pid=os.getpid()))
            with log_path.open('a') as log:
                result = subprocess.run(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
            if result.returncode:
                write_json(job / 'status.json', dict(state='failed', model=key, stage=stage,
                                                     log=str(log_path), returncode=result.returncode))
                raise SystemExit(result.returncode)
    write_json(job / 'status.json', dict(state='complete', models=args.models))


if __name__ == '__main__':
    main()
