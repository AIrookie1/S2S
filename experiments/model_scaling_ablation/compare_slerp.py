#!/usr/bin/env python3
"""Retune strength on dev for fixed t=.2/.4, then evaluate the original test folds."""
from __future__ import annotations
import argparse
from concurrent.futures import ThreadPoolExecutor, wait
import csv
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from experiments.model_scaling_ablation.common import write_json

BASE = ROOT / 'results/experiment/model_scaling_ablation'
OUT = BASE / 'slerp_02_04_retuned_strength'
RUNS = {
    'Qwen2.5-0.5B-Instruct': 'bfloat16',
    'Qwen2.5-1.5B-Instruct': 'bfloat16',
    'Qwen2.5-3B-Instruct': 'bfloat16',
    'Qwen2.5-14B-Instruct': 'nf4',
    'Qwen3.5-9B': 'bfloat16_fullgpu_e250',
    'Llama-3.1-Nemotron-Nano-8B-v1': 'bfloat16_fullgpu_e250',
}
ORDER = ['Qwen2.5-0.5B-Instruct', 'Qwen2.5-1.5B-Instruct', 'Qwen2.5-3B-Instruct',
         'Qwen2.5-7B-Instruct', 'llama3.1-8B-Instruct',
         'Llama-3.1-Nemotron-Nano-8B-v1', 'Qwen3.5-9B', 'Qwen2.5-14B-Instruct']

def read(path):
    return json.loads(path.read_text())

def weighted(reports):
    ids = [q for r in reports for q in r['question_ids']]
    if len(ids) != 817 or len(set(ids)) != 817:
        raise ValueError('Expected exactly 817 unique held-out questions')
    return sum(r['metrics']['MC1'] * len(r['question_ids']) for r in reports) / 817

def pair(values):
    return str(values[0]) if values[0] == values[1] else ' / '.join(map(str, values))

def summarize():
    rows = []
    for name in ORDER:
        if name in RUNS:
            source = BASE / name / RUNS[name]
            original = [read(source / f'fold{f}/s2s_l_test.json') for f in (0, 1)]
            for fold in (0, 1):
                added_test = OUT / name / f'fold{fold}_t0.4.json'
                added_selection = OUT / name / f'fold{fold}_t0.4_selection.json'
                if added_test.exists() and added_selection.exists():
                    old_candidates = read(source / f'fold{fold}/s2s_l_selection.json')['candidates']
                    new_candidates = read(added_selection)['candidates']
                    chosen = max(old_candidates + new_candidates, key=lambda c: (c['metrics']['MC1'], c['metrics']['MC2']))
                    if chosen['parameters'] == read(added_test)['parameters']:
                        original[fold] = read(added_test)
            row = dict(model=name, strength=pair([r['parameters']['strength'] for r in original]),
                       best_t=pair([r['parameters']['slerp_t'] for r in original]), best_MC1=weighted(original))
            for t in (.2, .4):
                paths = [OUT / name / f'fold{f}_t{t}.json' for f in (0, 1)]
                row[f'MC1_t{t}'] = weighted([read(p) for p in paths]) if all(p.exists() for p in paths) else None
                row[f'strength_t{t}'] = pair([read(p)['parameters']['strength'] for p in paths]) if all(p.exists() for p in paths) else None
        else:
            path = ROOT / 'results/experiment/dyn2d_ablation/summaries' / f'{name}_fold_mean.json'
            cells = read(path)['cells']
            best = max(cells, key=lambda c: (c['MC1'], c['MC2']))
            row = dict(model=name, strength=str(best['strength']), best_t=str(best['slerp_t']))
            for t in (.2, .4):
                cell = max((c for c in cells if c['slerp_t'] == t), key=lambda c: (c['MC1'], c['MC2']))
                row[f'MC1_t{t}'] = (cell['fold0_MC1'] * 408 + cell['fold1_MC1'] * 409) / 817
                row[f'strength_t{t}'] = str(cell['strength'])
            row['best_MC1'] = (best['fold0_MC1'] * 408 + best['fold1_MC1'] * 409) / 817
        rows.append(row)
    OUT.mkdir(parents=True, exist_ok=True)
    write_json(OUT / 'comparison.json', dict(protocol='Scaling models: separately retune strength on original dev for each fixed t, tie-break MC1 then MC2 then grid order. 7B/8B reuse existing test-grid optima. Best t is selected on the union of original dev grid and newly evaluated t=.4 candidates. MC1 question-weighted over 817 questions.', rows=rows))
    with (OUT / 'comparison.csv').open('w') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    def score(x):
        return '待完成' if x is None else f'{100*x:.2f}'
    lines = ['| 模型 | 最优权重 t | MC1 (%) | t=0.2 | MC1 (%) | t=0.4 | MC1 (%) |',
             '|---|---:|---:|---:|---:|---:|---:|']
    for r in rows:
        lines.append(f"| {r['model']} | {r['best_t']} | {score(r['best_MC1'])} | 0.2 (s={r['strength_t0.2'] or '待完成'}) | {score(r['MC1_t0.2'])} | 0.4 (s={r['strength_t0.4'] or '待完成'}) | {score(r['MC1_t0.4'])} |")
    lines += ['', '每个固定 t 重新选择 strength；s 为 strength，a / b 表示 fold 0 / fold 1。',
              '所有 MC1 均按两折题数加权；14B 保持 NF4，其余保持原精度。',
              '规模实验：strength 候选 [0.5, 1.0, 2.0, 3.0]，每折在原 dev 集选择；最优权重在原 dev 网格与新增 t=0.4 候选的并集上选择，不按 test 分数重选。7B/8B：复用已有测试集网格（strength=1..6），属于测试集网格最优，未与 dev 选择混称。', '',
              'Strength: ' + '; '.join(f"{r['model']}={r['strength']}" for r in rows)]
    (OUT / 'comparison.md').write_text('\n'.join(lines) + '\n')
    return rows

def worker(name):
    import torch
    from experiments.model_scaling_ablation.common import load_model
    from experiments.model_scaling_ablation.run import evaluate, make_hook
    from method.dyn3d_sga.eval_utils import load_truthfulqa_mc_dataframe, set_seed
    torch.set_num_threads(4)
    source = BASE / name / RUNS[name]
    output = OUT / name
    config = read(source / 'run_config.json')
    config['output_dir'] = str(output)
    output.mkdir(parents=True, exist_ok=True)
    manifest = dict(source_dir=str(source), original_config=read(source / 'run_config.json'),
                    protocol='retune_strength_on_dev_for_each_fixed_t', t_values=[.2, .4], strength_grid=[.5, 1., 2., 3.],
                    checkpoint_sha256={str(f): hashlib.sha256((source / f'fold{f}/s2s_l.npz').read_bytes()).hexdigest() for f in (0, 1)})
    target = output / 'comparison_config.json'
    if target.exists() and read(target) != manifest:
        raise ValueError('Comparison provenance changed')
    write_json(target, manifest)
    set_seed(config['seed'])
    model = None
    try:
        for fold in (0, 1):
            original = read(source / f'fold{fold}/s2s_l_test.json')
            split = read(source / f'fold{fold}/split.json')
            if original['question_ids'] != split['test']:
                raise ValueError('Source test split mismatch')
            for t in (.2, .4):
                path = output / f'fold{fold}_t{t}.json'
                selection_path = output / f'fold{fold}_t{t}_selection.json'
                if selection_path.exists():
                    selection = read(selection_path)
                else:
                    candidates = []
                    for strength in (.5, 1., 2., 3.):
                        params = dict(strength=strength, slerp_t=t)
                        candidate_path = output / 'dev' / f'fold{fold}_s{strength}_t{t}.json'
                        cached_source = next((p for p in (source / f'fold{fold}/dev').glob('s2s_l_*.json')
                                              if read(p)['parameters'] == params), None)
                        if candidate_path.exists():
                            candidate = read(candidate_path)
                        elif cached_source is not None:
                            candidate = dict(read(cached_source), reused_from=str(cached_source))
                            write_json(candidate_path, candidate)
                        else:
                            write_json(output / 'status.json', dict(state='dev_search', fold=fold, parameters=params, pid=os.getpid()))
                            if model is None:
                                model, tokenizer, _, layers = load_model(config)
                                block = layers[config['layer']]
                                frame, _ = load_truthfulqa_mc_dataframe(str(ROOT / 'TruthfulQA/data/v1/TruthfulQA.csv'))
                            hook = make_hook('s2s_l', params, source / f'fold{fold}', block)
                            metrics, details = evaluate(model, tokenizer, block, frame, split['dev'], hook, config)
                            candidate = dict(parameters=params, metrics=metrics, details=details)
                            write_json(candidate_path, candidate)
                        if {d['question_id'] for d in candidate['details']} != set(split['dev']):
                            raise ValueError('Dev candidate question IDs mismatch')
                        candidates.append(candidate)
                    best = max(candidates, key=lambda c: (c['metrics']['MC1'], c['metrics']['MC2']))
                    selection = dict(parameters=best['parameters'], selection='dev_MC1_then_MC2_then_grid_order',
                                     dev_question_ids=split['dev'], candidates=candidates)
                    write_json(selection_path, selection)
                params = selection['parameters']
                if path.exists():
                    cached = read(path)
                    if cached['parameters'] != params or cached['question_ids'] != split['test']:
                        raise ValueError('Cached report mismatch')
                    continue
                write_json(output / 'status.json', dict(state='evaluating', fold=fold, parameters=params, pid=os.getpid()))
                if original['parameters'] == params:
                    report = dict(original, reused_from=str(source / f'fold{fold}/s2s_l_test.json'))
                else:
                    if model is None:
                        model, tokenizer, _, layers = load_model(config)
                        block = layers[config['layer']]
                        frame, _ = load_truthfulqa_mc_dataframe(str(ROOT / 'TruthfulQA/data/v1/TruthfulQA.csv'))
                    hook = make_hook('s2s_l', params, source / f'fold{fold}', block)
                    metrics, details = evaluate(model, tokenizer, block, frame, split['test'], hook, config)
                    report = dict(fold=fold, method='s2s_l', parameters=params, question_ids=split['test'], metrics=metrics, details=details)
                write_json(path, report)
                print(name, fold, params, report['metrics'], flush=True)
        write_json(output / 'status.json', dict(state='complete', pid=os.getpid()))
    except Exception as exc:
        write_json(output / 'status.json', dict(state='failed', error=str(exc), pid=os.getpid()))
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
                                                      'HF_HUB_OFFLINE': '1', 'HF_DATASETS_OFFLINE': '1'},
                                        stdout=stream, stderr=subprocess.STDOUT)
            outcomes.append(dict(model=name, returncode=result.returncode))
        return outcomes
    write_json(OUT / 'status.json', dict(state='running', pid=os.getpid()))
    with ThreadPoolExecutor(max_workers=3) as pool:
        futures = [pool.submit(queue, 0, ['Qwen3.5-9B', 'Llama-3.1-Nemotron-Nano-8B-v1']),
                   pool.submit(queue, 1, ['Qwen2.5-0.5B-Instruct', 'Qwen2.5-1.5B-Instruct', 'Qwen2.5-3B-Instruct']),
                   pool.submit(queue, 2, ['Qwen2.5-14B-Instruct'])]
        pending = set(futures)
        while pending:
            _, pending = wait(pending, timeout=30)
            summarize()
        outcomes = [item for future in futures for item in future.result()]
    rows = summarize()
    complete = all(r['MC1_t0.2'] is not None and r['MC1_t0.4'] is not None for r in rows)
    write_json(OUT / 'status.json', dict(state='complete' if complete else 'failed', outcomes=outcomes))
    if not complete:
        raise SystemExit(1)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', choices=RUNS)
    parser.add_argument('--summarize', action='store_true')
    args = parser.parse_args()
    if args.model:
        import fcntl
        lock_dir = OUT / args.model
        lock_dir.mkdir(parents=True, exist_ok=True)
        with (lock_dir / '.worker.lock').open('a') as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            worker(args.model)
    elif args.summarize: summarize()
    else: launch()
