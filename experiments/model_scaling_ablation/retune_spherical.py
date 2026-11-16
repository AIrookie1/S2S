#!/usr/bin/env python3
"""Resume a wider, independently selected Spherical Steering sweep on six models."""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, wait
import csv
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from experiments.model_scaling_ablation.common import fingerprint, parameter_grid, write_json

BASE = ROOT / 'results/experiment/model_scaling_ablation'
OUT = BASE / 'spherical_retuned_v1'
RUNS = {
    'Qwen2.5-0.5B-Instruct': 'bfloat16',
    'Qwen2.5-1.5B-Instruct': 'bfloat16',
    'Qwen2.5-3B-Instruct': 'bfloat16',
    'Qwen2.5-14B-Instruct': 'nf4',
    'Qwen3.5-9B': 'bfloat16_fullgpu_e250',
    'Llama-3.1-Nemotron-Nano-8B-v1': 'bfloat16_fullgpu_e250',
}
METRICS = ('MC1', 'MC2', 'MC3')


def read(path):
    return json.loads(Path(path).read_text())


def expanded_grid():
    # Keep old candidates first so ties retain their original deterministic order.
    grid = parameter_grid('spherical', True)
    grid += [dict(kappa=k, alpha=a, beta=.1)
             for k in (100., 200., 500.) for a in (.3, .6, .9)]
    grid += [dict(kappa=k, alpha=a, beta=0.)
             for k in (20., 50., 100., 200.) for a in (.3, .6)]
    grid += [dict(kappa=k, alpha=a, beta=-.1)
             for k in (20., 50.) for a in (.3, .6)]
    grid += [dict(kappa=k, alpha=.9, beta=.1) for k in (20., 50.)]
    grid += [dict(kappa=k, alpha=.6, beta=.4) for k in (100., 200., 500.)]
    grid += [dict(kappa=20., alpha=0., beta=.1)]  # Explicit no-intervention candidate.
    return grid


def validate_split(split):
    groups = [set(split[k]) for k in ('geometry', 'predictor', 'dev', 'test')]
    if any(not g for g in groups) or any(groups[i] & groups[j] for i in range(4) for j in range(i)):
        raise ValueError('Geometry/predictor/dev/test must be nonempty and disjoint')
    if any(len(split[k]) != len(set(split[k])) for k in ('geometry', 'predictor', 'dev', 'test')):
        raise ValueError('Duplicate question IDs in split')


def validate_report(report, ids, parameters):
    details = report['details']
    actual = [d['question_id'] for d in details]
    if len(actual) != len(ids) or set(actual) != set(ids) or report['parameters'] != parameters:
        raise ValueError('Cached parameters or question coverage mismatch')
    for metric in METRICS:
        values = [d[metric] for d in details]
        if any(not math.isfinite(v) or not -1e-12 <= v <= 1 + 1e-12 for v in values):
            raise ValueError('Invalid per-question metric')
        if not math.isclose(sum(values) / len(values), report['metrics'][metric], abs_tol=1e-10):
            raise ValueError('Cached metrics disagree with per-question scores')


def select(candidates):
    return max(candidates, key=lambda c: (c['metrics']['MC1'], c['metrics']['MC2']))


def weighted(reports):
    ids = [q for r in reports for q in r['question_ids']]
    if len(ids) != 817 or set(ids) != set(range(817)):
        raise ValueError('Expected exactly 817 unique held-out questions')
    return {key: sum(r['metrics'][key] * len(r['question_ids']) for r in reports) / 817 for key in METRICS}


def summarize():
    rows = []
    for name, run in RUNS.items():
        source, output = BASE / name / run, OUT / name
        old = weighted([read(source / f'fold{f}/spherical_test.json') for f in (0, 1)])
        paths = [output / f'fold{f}/spherical_test.json' for f in (0, 1)]
        complete = all(p.exists() for p in paths)
        reports = [read(p) for p in paths] if complete else []
        new = weighted(reports) if complete else {key: None for key in METRICS}
        status = read(output / 'status.json') if (output / 'status.json').exists() else {'state': 'pending'}
        row = dict(model=name, precision='NF4' if run == 'nf4' else 'BF16', state='complete' if complete else status['state'])
        for key in METRICS:
            row['original_' + key] = old[key]
            row['retuned_' + key] = new[key]
        row['MC1_gain_pp'] = 100 * (new['MC1'] - old['MC1']) if complete else None
        row['fold_parameters'] = [r['parameters'] for r in reports]
        rows.append(row)
    OUT.mkdir(parents=True, exist_ok=True)
    write_json(OUT / 'comparison.json', dict(protocol='Independent per-fold dev MC1 then MC2; all old candidates plus expanded grid; test evaluated only after selection; no pooled-dev selection.', rows=rows))
    with (OUT / 'comparison.csv').open('w') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    lines = ['| Model | Precision | State | Old MC1 | New MC1 | Gain (pp) | New MC2 | New MC3 |',
             '|---|---|---|---:|---:|---:|---:|---:|']
    def pct(x):
        return 'pending' if x is None else f'{100*x:.2f}'
    for row in rows:
        gain = 'pending' if row['MC1_gain_pp'] is None else f"{row['MC1_gain_pp']:+.2f}"
        lines.append(f"| {row['model']} | {row['precision']} | {row['state']} | {pct(row['original_MC1'])} | {pct(row['retuned_MC1'])} | {gain} | {pct(row['retuned_MC2'])} | {pct(row['retuned_MC3'])} |")
    lines += ['', 'Selection uses only the corresponding fold dev set (82 questions). Test metrics are question-weighted over 817 questions.',
              'A larger dev grid does not guarantee improved held-out scores. Both original and retuned results are retained.',
              '14B keeps the original NF4 precision; other models keep BF16. Layer, prototypes, prompts and model weights are unchanged.', '']
    for row in rows:
        if row['fold_parameters']:
            lines.append(f"- {row['model']}: fold0 / fold1 = `{json.dumps(row['fold_parameters'], sort_keys=True)}`")
    (OUT / 'comparison.md').write_text('\n'.join(lines) + '\n')
    return rows


def worker(name):
    import torch
    from experiments.model_scaling_ablation.common import load_model
    from experiments.model_scaling_ablation.run import evaluate, make_hook
    from method.dyn3d_sga.eval_utils import load_truthfulqa_mc_dataframe, set_seed
    torch.set_num_threads(4)
    source, output = BASE / name / RUNS[name], OUT / name
    config = read(source / 'run_config.json')
    splits = [read(source / f'fold{f}/split.json') for f in (0, 1)]
    for split in splits:
        validate_split(split)
    # The other fold dev overlaps this fold test: never pool their selection scores.
    manifest = dict(source_dir=str(source), source_config=config, grid=expanded_grid(),
                    selection='independent_fold_dev_MC1_then_MC2_then_grid_order',
                    hashes={str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
                            for p in [Path(__file__), ROOT / 'method/spherical_steering/steering.py',
                                      ROOT / 'experiments/model_scaling_ablation/run.py',
                                      ROOT / 'experiments/model_scaling_ablation/common.py',
                                      ROOT / 'method/dyn3d_sga/eval_utils.py',
                                      ROOT / 'truthfulqa/models.py',
                                      ROOT / 'TruthfulQA/data/v1/TruthfulQA.csv',
                                      *[source / f'fold{f}' / filename for f in (0, 1)
                                        for filename in ('geometry.npz', 'split.json')]]})
    output.mkdir(parents=True, exist_ok=True)
    target = output / 'run_config.json'
    if target.exists() and read(target) != manifest:
        raise ValueError('Retune inputs/source changed: choose a new output directory')
    write_json(target, manifest)
    run_id = fingerprint(manifest)
    config = dict(config, output_dir=str(output))
    set_seed(config['seed'])
    model = tokenizer = block = frame = None

    def status(state, **kw):
        write_json(output / 'status.json', dict(state=state, pid=os.getpid(), updated_at=time.time(), **kw))

    def evaluate_params(params, fold, ids):
        nonlocal model, tokenizer, block, frame
        if model is None:
            status('loading', fold=fold)
            model, tokenizer, layer_path, layers = load_model(config)
            block = layers[config['layer']]
            write_json(output / 'runtime.json', dict(model_class=type(model).__name__, layer_path=layer_path,
                       devices={str(k): str(v) for k, v in getattr(model, 'hf_device_map', {}).items()},
                       torch=torch.__version__, cuda_visible_devices=os.environ.get('CUDA_VISIBLE_DEVICES')))
            frame, _ = load_truthfulqa_mc_dataframe(str(ROOT / 'TruthfulQA/data/v1/TruthfulQA.csv'))
        hook = make_hook('spherical', params, source / f'fold{fold}', block)
        metrics, details = evaluate(model, tokenizer, block, frame, ids, hook, config)
        return dict(parameters=params, metrics=metrics, details=details)

    try:
        for fold, split in enumerate(splits):
            candidates = []
            cache = {}
            for path in sorted((source / f'fold{fold}/dev').glob('spherical_*.json')):
                cached = read(path)
                validate_report(cached, split['dev'], cached['parameters'])
                cache[fingerprint(cached['parameters'])] = (path, cached)
            for index, params in enumerate(expanded_grid()):
                path = output / f'fold{fold}/dev' / f'spherical_{index}.json'
                status('dev_search', fold=fold, candidate=index, total=len(expanded_grid()), parameters=params)
                start = time.monotonic()
                if path.exists():
                    candidate = read(path)
                    if candidate.get('run') != run_id:
                        raise ValueError('Retune cache provenance mismatch')
                elif fingerprint(params) in cache:
                    original_path, candidate = cache[fingerprint(params)]
                    candidate = dict(candidate, reused_from=str(original_path), run=run_id)
                    write_json(path, candidate)
                else:
                    candidate = dict(evaluate_params(params, fold, split['dev']), run=run_id)
                    write_json(path, candidate)
                validate_report(candidate, split['dev'], params)
                candidates.append(candidate)
                print(name, 'dev', fold, index, params, candidate['metrics'],
                      f'{time.monotonic()-start:.1f}s', flush=True)
            best = select(candidates)
            selection = dict(parameters=best['parameters'], metrics=best['metrics'],
                             selection=manifest['selection'], dev_question_ids=split['dev'], run=run_id,
                             candidates=[{k: v for k, v in c.items() if k != 'details'} for c in candidates])
            write_json(output / f'fold{fold}/spherical_selection.json', selection)
            path = output / f'fold{fold}/spherical_test.json'
            params = best['parameters']
            status('test_evaluation', fold=fold, parameters=params)
            original_path = source / f'fold{fold}/spherical_test.json'
            original = read(original_path)
            if path.exists():
                report = read(path)
                if report.get('run') != run_id:
                    raise ValueError('Test report provenance mismatch')
            elif params == original['parameters']:
                report = dict(original, run=run_id, reused_from=str(original_path))
            elif params['alpha'] == 0.:
                baseline_path = source / f'fold{fold}/baseline_test.json'
                report = dict(read(baseline_path), method='spherical', parameters=params,
                              run=run_id, reused_from=str(baseline_path))
            else:
                report = dict(evaluate_params(params, fold, split['test']), run=run_id,
                              fold=fold, method='spherical', question_ids=split['test'])
            validate_report(report, split['test'], params)
            if report['question_ids'] != split['test']:
                raise ValueError('Test question IDs mismatch')
            write_json(path, report)
            print(name, 'test', fold, params, report['metrics'], flush=True)
        metrics = weighted([read(output / f'fold{f}/spherical_test.json') for f in (0, 1)])
        write_json(output / 'summary.json', dict(model=name, method='spherical', metrics=metrics, n_questions=817, run=run_id))
        status('complete', metrics=metrics)
    except Exception as exc:
        status('failed', error=repr(exc))
        raise


def launch():
    OUT.mkdir(parents=True, exist_ok=True)
    summarize()
    def queue(gpu, names):
        outcomes = []
        for name in names:
            with (OUT / f'{name}.log').open('a') as stream:
                result = subprocess.run([sys.executable, '-u', str(Path(__file__).resolve()), '--model', name],
                    cwd=ROOT, env={**os.environ, 'CUDA_VISIBLE_DEVICES': str(gpu),
                                  'OMP_NUM_THREADS': '4', 'MKL_NUM_THREADS': '4',
                                  'S2S_GPU_RESERVE_GIB': '3',
                                  'HF_HUB_OFFLINE': '1', 'HF_DATASETS_OFFLINE': '1'},
                    stdout=stream, stderr=subprocess.STDOUT)
            outcomes.append(dict(model=name, returncode=result.returncode))
        return outcomes
    write_json(OUT / 'status.json', dict(state='running', pid=os.getpid(), started_at=time.time()))
    with ThreadPoolExecutor(max_workers=3) as pool:
        pending = {pool.submit(queue, 0, ['Qwen2.5-0.5B-Instruct', 'Qwen2.5-1.5B-Instruct']),
                   pool.submit(queue, 1, ['Qwen2.5-3B-Instruct', 'Qwen2.5-14B-Instruct']),
                   pool.submit(queue, 2, ['Llama-3.1-Nemotron-Nano-8B-v1', 'Qwen3.5-9B'])}
        futures = list(pending)
        while pending:
            _, pending = wait(pending, timeout=30)
            summarize()
        outcomes = [item for future in futures for item in future.result()]
    summarize()
    complete = all(item['returncode'] == 0 for item in outcomes)
    write_json(OUT / 'status.json', dict(state='complete' if complete else 'failed', outcomes=outcomes))
    if not complete:
        raise SystemExit(1)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', choices=RUNS)
    parser.add_argument('--summarize', action='store_true')
    parser.add_argument('--plan', action='store_true')
    args = parser.parse_args()
    if args.plan:
        print(json.dumps(dict(runs=RUNS, output_dir=str(OUT), grid=expanded_grid(),
                              candidates_per_fold=len(expanded_grid()),
                              new_candidates_per_fold=len(expanded_grid())-36), indent=2))
    elif args.summarize:
        summarize()
    else:
        lock_dir = OUT / args.model if args.model else OUT
        lock_dir.mkdir(parents=True, exist_ok=True)
        with (lock_dir / '.worker.lock').open('a') as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            worker(args.model) if args.model else launch()
