"""Question-paired outcome comparisons; no gradient-based causal claims."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
from scipy.stats import binomtest, spearmanr

CONDITIONS = ('baseline', 'spherical', 'pred_adaptive', 'oracle_adaptive',
              'pred_selected', 'oracle_selected')


def bootstrap_difference(values, seed=42, samples=10000):
    values = np.asarray(values, dtype=float)
    rng = np.random.default_rng(seed)
    boot = np.concatenate([values[rng.integers(0, len(values), (min(256, samples-i), len(values)))].mean(1)
                           for i in range(0, samples, 256)])
    return tuple(float(x * 100) for x in np.quantile(boot, [.025, .975]))


def dataset_report(folder):
    spec = json.loads((folder / 'manifest.json').read_text())
    expected = set(spec['selected_q_indices'])
    records = [json.loads(p.read_text()) for p in sorted(folder.glob('q*.json'))]
    found = {r['q_idx'] for r in records}
    if found != expected or len(records) != len(expected):
        raise ValueError(f'{folder.name}: incomplete or duplicate records ({len(found)}/{len(expected)})')
    if any(r['signature'] != spec['signature'] for r in records):
        raise ValueError(f'{folder.name}: mismatched configuration signatures')
    correct = {k: np.array([r['margins'][k] > 0 for r in records]) for k in CONDITIONS}
    result = {'dataset': folder.name, 'questions': len(records)}
    for k in CONDITIONS:
        result[k + '_accuracy_pct'] = float(correct[k].mean() * 100)
    change = correct['pred_adaptive'].astype(int) - correct['spherical'].astype(int)
    corrected, broken = int((change > 0).sum()), int((change < 0).sum())
    ci_low, ci_high = bootstrap_difference(change)
    result.update(corrected=corrected, broken=broken,
        delta_pp=float(change.mean() * 100), ci_low_pp=ci_low,
        ci_high_pp=ci_high,
        mcnemar_exact_p=float(binomtest(corrected, corrected+broken, .5).pvalue) if corrected+broken else 1.)
    result['oracle_residual_gain_pp'] = result['oracle_adaptive_accuracy_pct'] - result['pred_adaptive_accuracy_pct']
    result['weight_selection_gain_pp'] = result['pred_selected_accuracy_pct'] - result['pred_adaptive_accuracy_pct']
    result['joint_oracle_gain_pp'] = result['oracle_selected_accuracy_pct'] - result['pred_adaptive_accuracy_pct']
    finite_differences, linear_predictions = [], []
    cosine, benefits, radial, probe_margin = [], [], [], {k: [] for k in ('full', 'radial', 'tangent', 'mean', 'residual', 'oracle_tangent')}
    for r in records:
        negative = [c for c in r['candidates'] if not c['positive']]
        cosine.append(float(np.mean([c['geometry']['pred_target_cosine'] for c in negative])))
        radial.append(float(np.mean([c['geometry']['radial_energy_fraction'] for c in negative])))
        benefits.append(r['margins']['pred_adaptive'] - r['margins']['spherical'])
        for c in r['candidates']:
            finite_differences.append(c['geometry']['fd_derivative'])
            linear_predictions.append(c['geometry']['fd_linear'])
        for k in probe_margin:
            probe_margin[k].append(r['fixed_competitor_margins'][f'probe_{k}'] - r['fixed_competitor_margins']['baseline'])
    if np.std(cosine) > 0 and np.std(benefits) > 0:
        result['vector_cosine_vs_margin_benefit_spearman'] = float(spearmanr(cosine, benefits).statistic)
    else:
        result['vector_cosine_vs_margin_benefit_spearman'] = None
    result['radial_energy_fraction_mean'] = float(np.mean(radial))
    for k, values in probe_margin.items():
        result[f'probe_{k}_positive_margin_fraction'] = float((np.asarray(values) > 0).mean())
    result['fd_sign_agreement'] = float(np.mean(np.asarray(finite_differences) * np.asarray(linear_predictions) > 0))
    result['gradient_interpretation'] = 'not validated at full-model FP16 precision'
    return result


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('input', type=Path)
    parser.add_argument('--datasets', nargs='+', default=['truthfulqa', 'copa', 'storycloze', 'mmlu_global', 'winogrande', 'boolq'])
    args = parser.parse_args()
    rows = [dataset_report(args.input / dataset) for dataset in args.datasets]
    (args.input / 'final_results.json').write_text(json.dumps(rows, indent=2, allow_nan=False))
    with (args.input / 'final_results.csv').open('w') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    lines = ['# Full-set geometry/decision experiment', '',
        'Llama-3.1-8B-Instruct; current repository checkpoints and task hyperparameters. All expected question IDs and record signatures were verified. Generic uses official labeled validation/eval splits, with the repository balanced 2000-question MMLU test subset. TruthfulQA uses held-out questions routed separately to MLP and spherical-prototype banks. Scores sum answer-token log probabilities on TruthfulQA and average them on generic tasks.', '',
        'Oracle residuals replace negative candidates only. Selected fusion weights use answer labels over a fixed grid with source-adaptive gates held fixed, and are diagnostic upper bounds, not deployable accuracy. FP16 gradient finite differences are not validated; conclusions here use actual rescored outcomes. Current COPA S2S-R previously reproduced 90%, versus the paper table 92%; these results characterize this implementation and are not an exact paper-table reproduction.', '',
        '| Dataset | N | Baseline | Spherical | S2S-R | Oracle residual | Selected weights | Joint oracle |',
        '|---|---:|---:|---:|---:|---:|---:|---:|']
    for r in rows:
        vals = ' | '.join(f"{r[k + '_accuracy_pct']:.2f}" for k in CONDITIONS)
        lines.append(f"| {r['dataset']} | {r['questions']} | {vals} |")
    lines += ['', '| Dataset | S2S − spherical (pp) | Paired bootstrap 95% CI | Corrected | Broken | Exact McNemar p |',
              '|---|---:|---|---:|---:|---:|']
    for r in rows:
        lines.append(f"| {r['dataset']} | {r['delta_pp']:+.2f} | [{r['ci_low_pp']:+.2f}, {r['ci_high_pp']:+.2f}] | {r['corrected']} | {r['broken']} | {r['mcnemar_exact_p']:.4g} |")
    lines += ['', 'P values are per-dataset exploratory tests without multiplicity adjustment. Bootstrap samples questions within each task; checkpoints are treated as fixed. Positive margin means the designated correct answer outranks every incorrect answer; exact ties count as failures.', '',
        '## Outcome interpretation', '']
    for r in rows:
        residual_gap = r['oracle_adaptive_accuracy_pct'] - r['spherical_accuracy_pct']
        lines.append(f"- {r['dataset']}: S2S changes {r['corrected']} spherical errors into correct decisions and {r['broken']} correct decisions into errors. Oracle residual gain = {r['oracle_residual_gain_pp']:+.2f} pp; label-selected fusion gain = {r['weight_selection_gain_pp']:+.2f} pp; oracle-residual accuracy remains {residual_gap:+.2f} pp relative to spherical. These distinguish outcome effects under the specified interventions and do not identify a universal cause.")
    (args.input / 'final_report.md').write_text('\n'.join(lines) + '\n')
    print('\n'.join(lines))


if __name__ == '__main__':
    main()
