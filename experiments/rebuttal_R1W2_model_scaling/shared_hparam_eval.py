#!/usr/bin/env python3
"""Select one shared dev configuration per method, then evaluate both test folds."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from experiments.rebuttal_R1W2_model_scaling.common import METHODS, aggregate, load_model, write_json
from experiments.rebuttal_R1W2_model_scaling.rebuttal_R1W2_run import evaluate, make_hook


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-dir', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args()
    source = args.source_dir.resolve()
    output = args.output_dir.resolve()
    config = json.loads((source / 'run_config.json').read_text())
    output.mkdir(parents=True, exist_ok=True)
    config['output_dir'] = str(output)

    selections = {}
    for method in METHODS:
        if method == 'baseline':
            selections[method] = dict(parameters={}, selection='no_parameters')
            continue
        cells = []
        for index in range(10_000):
            paths = [source / f'fold{fold}' / 'dev' / f'{method}_{index}.json' for fold in (0, 1)]
            if not all(path.exists() for path in paths):
                if index == 0:
                    raise FileNotFoundError(f'Missing dev candidate for {method}')
                break
            candidates = [json.loads(path.read_text()) for path in paths]
            if candidates[0]['parameters'] != candidates[1]['parameters']:
                raise ValueError(f'Dev grid mismatch for {method} candidate {index}')
            metrics = {
                key: sum(candidate['metrics'][key] * len(candidate['details']) for candidate in candidates)
                / sum(len(candidate['details']) for candidate in candidates)
                for key in ('MC1', 'MC2', 'MC3')
            }
            cells.append(dict(index=index, parameters=candidates[0]['parameters'], metrics=metrics))
        best = max(cells, key=lambda cell: (cell['metrics']['MC1'], cell['metrics']['MC2']))
        selections[method] = dict(parameters=best['parameters'], selected_candidate=best['index'],
                                  pooled_dev_metrics=best['metrics'], candidates=cells,
                                  selection='pooled_dev_MC1_then_MC2_then_grid_order')
    write_json(output / 'shared_selections.json', selections)

    import torch
    from method.S2S_R.eval_utils import load_truthfulqa_mc_dataframe, set_seed
    from method.S2S_R.eval_utils import load_truthfulqa_mc_dataframe
    torch.set_num_threads(4)
    set_seed(config['seed'])
    model = None
    reports = []
    for fold in (0, 1):
        split = json.loads((source / f'fold{fold}' / 'split.json').read_text())
        for method in METHODS:
            parameters = selections[method]['parameters']
            source_report = source / f'fold{fold}' / f'{method}_test.json'
            if source_report.exists():
                cached = json.loads(source_report.read_text())
                if cached['parameters'] == parameters and cached['question_ids'] == split['test']:
                    write_json(output / f'fold{fold}' / f'{method}_test.json', cached)
                    reports.append(cached)
                    print(f'{method} fold={fold}: reused matching test parameters', flush=True)
                    continue
            if model is None:
                model, tokenizer, layer_path, layers = load_model(config)
                block = layers[config['layer']]
                frame, _dataset = load_truthfulqa_mc_dataframe(str(ROOT / 'TruthfulQA/data/v1/TruthfulQA.csv'))
            hook = make_hook(method, parameters, source / f'fold{fold}', block)
            metrics, details = evaluate(model, tokenizer, block, frame, split['test'], hook, config)
            report = dict(fold=fold, method=method, parameters=parameters,
                          question_ids=split['test'], metrics=metrics, details=details)
            write_json(output / f'fold{fold}' / f'{method}_test.json', report)
            reports.append(report)
            print(f"{method} fold={fold}: {metrics}", flush=True)
    rows = aggregate(reports)
    if any(row['n_questions'] != 817 for row in rows):
        raise ValueError('Expected complete 817-question coverage')
    write_json(output / 'summary.json', dict(source_dir=str(source), config=config,
                                             selections=selections, rows=rows, folds=reports))
    import csv
    with (output / 'summary.csv').open('w') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    with (output / 'summary.md').open('w') as stream:
        stream.write('| Method | MC1 | MC2 | MC3 | Avg. | N | Shared parameters |\n')
        stream.write('|---|---:|---:|---:|---:|---:|---|\n')
        for row in rows:
            params = json.dumps(selections[row['method']]['parameters'], sort_keys=True)
            stream.write(f"| {row['method']} | " + ' | '.join(f"{100*row[key]:.2f}" for key in ('MC1','MC2','MC3','Avg'))
                         + f" | {row['n_questions']} | `{params}` |\n")


if __name__ == '__main__':
    main()
