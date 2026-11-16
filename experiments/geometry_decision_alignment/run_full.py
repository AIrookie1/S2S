"""Supervise two GPU workers and write a complete-set report after both finish."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent


def main():
    p = argparse.ArgumentParser(__doc__)
    p.add_argument('--gpus', nargs='+', default=['1', '2'])
    p.add_argument('--output', type=Path, default=ROOT / 'results/experiment/geometry_decision_alignment_full')
    args = p.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    source = ROOT / 'results/experiment/geometry_decision_alignment_multitask_smoke/copa'
    if not (args.output / 'copa').exists():
        shutil.copytree(source, args.output / 'copa')
    workers, logs = [], []
    status = {'started_utc': datetime.now(timezone.utc).isoformat(), 'status': 'running',
              'datasets': ['truthfulqa', 'storycloze', 'mmlu_global', 'winogrande', 'boolq'],
              'reused': 'COPA full 100-question run', 'workers': []}
    def save_status():
        temp = args.output / 'run_status.tmp'
        temp.write_text(json.dumps(status, indent=2))
        temp.replace(args.output / 'run_status.json')
    for shard, gpu in enumerate(args.gpus):
        log = (args.output / f'worker_gpu{gpu}.log').open('a', buffering=1)
        logs.append(log)
        env = {**os.environ, 'CUDA_VISIBLE_DEVICES': gpu, 'HF_HUB_OFFLINE': '1',
               'HF_DATASETS_OFFLINE': '1', 'OMP_NUM_THREADS': '4', 'MKL_NUM_THREADS': '4',
               'PYTORCH_CUDA_ALLOC_CONF': 'expandable_segments:True'}
        command = [sys.executable, str(HERE / 'run.py'), '--datasets', *status['datasets'],
                   '--max-questions', '0', '--batch-size', '1', '--output', str(args.output),
                   '--shard-count', str(len(args.gpus)), '--shard-index', str(shard)]
        worker = subprocess.Popen(command, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT)
        workers.append(worker)
        status['workers'].append({'gpu': gpu, 'pid': worker.pid, 'returncode': None})
    save_status()
    while any(worker.poll() is None for worker in workers):
        for info, worker in zip(status['workers'], workers):
            info['returncode'] = worker.poll()
        status['progress'] = {dataset: len(list((args.output / dataset).glob('q*.json')))
                              for dataset in status['datasets']}
        status['updated_utc'] = datetime.now(timezone.utc).isoformat()
        save_status()
        time.sleep(15)
    for info, worker in zip(status['workers'], workers):
        info['returncode'] = worker.returncode
    for log in logs:
        log.close()
    with (args.output / 'report.log').open('w') as log:
        subprocess.run([sys.executable, str(HERE / 'summarize.py'), str(args.output)],
                       stdout=log, stderr=subprocess.STDOUT)
        if all(worker.returncode == 0 for worker in workers):
            report = subprocess.run([sys.executable, str(HERE / 'report.py'), str(args.output)],
                                    stdout=log, stderr=subprocess.STDOUT)
            status['status'] = 'complete' if report.returncode == 0 else 'report_failed'
        else:
            status['status'] = 'worker_failed'
    status['finished_utc'] = datetime.now(timezone.utc).isoformat()
    save_status()


if __name__ == '__main__':
    main()
