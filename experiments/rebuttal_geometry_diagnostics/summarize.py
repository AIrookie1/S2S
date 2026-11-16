"""Summarize completed mechanism records, including numerical audit limits."""
import argparse
import csv
import json
from pathlib import Path

import numpy as np


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('input', type=Path)
    args = parser.parse_args()
    rows, audits = [], []
    for folder in sorted(args.input.iterdir()):
        if not folder.is_dir() or not (folder / 'manifest.json').exists():
            continue
        spec = json.loads((folder / 'manifest.json').read_text())
        records = [json.loads(p.read_text()) for p in sorted(folder.glob('q*.json'))]
        if not records:
            continue
        conditions = ['baseline', 'spherical', 'pred_adaptive', 'oracle_adaptive',
                      'pred_selected', 'oracle_selected']
        row = {'dataset': folder.name, 'completed': len(records),
               'expected': len(spec['selected_q_indices'])}
        for condition in conditions:
            row[condition + '_accuracy_pct'] = 100 * np.mean([r['margins'][condition] > 0 for r in records])
            row[condition + '_mean_margin'] = float(np.mean([r['margins'][condition] for r in records]))
        row['corrected_questions'] = sum(r['margins']['spherical'] <= 0 < r['margins']['pred_adaptive'] for r in records)
        row['broken_questions'] = sum(r['margins']['pred_adaptive'] <= 0 < r['margins']['spherical'] for r in records)
        candidates = [c for r in records for c in r['candidates']]
        parity = [a['absolute_error'] for r in records for a in r['parity_audits']]
        row['parity_max_absolute_error'] = max(parity) if parity else None
        comparisons = [(c['geometry']['fd_derivative'], c['geometry']['fd_linear']) for c in candidates]
        row['fd_sign_agreement'] = float(np.mean([a * b > 0 for a, b in comparisons]))
        row['fd_median_absolute_error'] = float(np.median([abs(a-b) for a,b in comparisons]))
        rows.append(row)
        audits.append({'dataset': folder.name, 'parity_max': row['parity_max_absolute_error'],
            'fd_sign_agreement': row['fd_sign_agreement'],
            'gradient_validation_status': 'FP16 finite differences are diagnostic only; not a passed validation gate'})
    if not rows:
        raise ValueError('No completed question records')
    with (args.input / 'summary.csv').open('w') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    (args.input / 'numerical_audit.json').write_text(json.dumps(audits, indent=2))
    lines = ['# Mechanism run verification', '',
        'Completed records only. Tiny smoke samples must not be used for benchmark or significance claims. Selected-weight conditions use ground-truth labels and are analysis-only upper bounds. FP16 small finite differences can disagree with analytic gradients; gradient-based mechanism conclusions require higher-precision verification.', '',
        '| Dataset | Completed / expected | Full/suffix max score error | FD sign agreement |',
        '|---|---:|---:|---:|']
    for row in rows:
        lines.append(f"| {row['dataset']} | {row['completed']} / {row['expected']} | {row['parity_max_absolute_error']:.6f} | {row['fd_sign_agreement']:.3f} |")
    (args.input / 'verification.md').write_text('\n'.join(lines) + '\n')
    print('\n'.join(lines))


if __name__ == '__main__':
    main()
