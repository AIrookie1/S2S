#!/usr/bin/env python3
"""Complete Spherical Steering kappa/alpha/beta sweep on existing artifacts."""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from experiments.rebuttal_R1W2_model_scaling.common import load_model, write_json
from experiments.rebuttal_R1W2_model_scaling.rebuttal_R1W2_run import evaluate, make_hook

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--source-dir', type=Path, required=True)
    p.add_argument('--output-dir', type=Path, required=True)
    args = p.parse_args()
    source, output = args.source_dir.resolve(), args.output_dir.resolve()
    config = json.loads((source / 'run_config.json').read_text())
    config['output_dir'] = str(output)
    output.mkdir(parents=True, exist_ok=True)
    import torch
    from methods.S2S_R.eval_utils import load_truthfulqa_mc_dataframe, set_seed
    torch.set_num_threads(4); set_seed(config['seed'])
    model, tokenizer, _, layers = load_model(config)
    block = layers[config['layer']]
    frame, _ = load_truthfulqa_mc_dataframe(str(ROOT / 'data/truthfulqa/TruthfulQA.csv'))
    grid = [dict(kappa=k, alpha=a, beta=b)
            for k in (1.0, 3.0, 5.0, 10.0, 20.0, 50.0)
            for a in (0.15, 0.3, 0.6) for b in (0.1, 0.4)]
    candidates = []
    for index, params in enumerate(grid):
        fold_results = []
        for fold in (0, 1):
            split = json.loads((source / f'fold{fold}' / 'split.json').read_text())
            dev_dir = source / f'fold{fold}' / 'dev'
            cached = next((path for path in dev_dir.glob('spherical_*.json')
                           if json.loads(path.read_text()).get('parameters') == params), None)
            if cached is not None:
                candidate = json.loads(cached.read_text())
            else:
                hook = make_hook('spherical', params, source / f'fold{fold}', block)
                metrics, details = evaluate(model, tokenizer, block, frame, split['dev'], hook, config)
                candidate = dict(parameters=params, metrics=metrics, details=details)
            fold_results.append(candidate)
        n = sum(len(x['details']) for x in fold_results)
        metrics = {key: sum(x['metrics'][key] * len(x['details']) for x in fold_results) / n
                   for key in ('MC1', 'MC2', 'MC3')}
        candidates.append(dict(index=index, parameters=params, metrics=metrics))
        print(index, params, metrics, flush=True)
    best = max(candidates, key=lambda x: (x['metrics']['MC1'], x['metrics']['MC2']))
    selections = dict(selection='pooled_dev_MC1_then_MC2_then_grid_order', parameters=best['parameters'],
                      pooled_dev_metrics=best['metrics'], candidates=candidates)
    write_json(output / 'selection.json', selections)
    reports=[]
    for fold in (0, 1):
        split=json.loads((source / f'fold{fold}' / 'split.json').read_text())
        hook=make_hook('spherical', best['parameters'], source / f'fold{fold}', block)
        metrics, details=evaluate(model, tokenizer, block, frame, split['test'], hook, config)
        report=dict(fold=fold, method='spherical', parameters=best['parameters'], question_ids=split['test'], metrics=metrics, details=details)
        write_json(output / f'fold{fold}_test.json', report); reports.append(report)
        print('test', fold, metrics, flush=True)
    write_json(output / 'test_reports.json', reports)

if __name__ == '__main__': main()
