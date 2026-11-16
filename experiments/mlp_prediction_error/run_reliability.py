"""Recompute direct-MLP diagnostics using the exact L2S reliability protocol."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys

import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from experiments.l2s_reliability.metrics import rowwise_vector_metrics, aggregate_by_question
from experiments.mlp_prediction_error.run import write_csv
from method.dyn3d_sga.dyn_mlp import DirectVectorMLP
from method.l2s_linear.model import load_l2s_pairs


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--input', type=Path, default=ROOT / 'results/experiment/mlp_prediction_error/official')
    parser.add_argument('--device', default='cpu')
    args = parser.parse_args()
    torch.set_num_threads(4)
    output = args.input / 'reliability'
    output.mkdir(exist_ok=True)
    with (args.input / 'fold_metrics.csv').open() as f:
        manifest = list(csv.DictReader(f))
    question_rows, fold_rows = [], []
    for item in manifest:
        print(item['dataset'], item['layer'], item['fold'], item['split'], flush=True)
        feature, checkpoint = Path(item['feature_path']), Path(item['checkpoint'])
        # Same loader used by experiments/l2s_reliability/evaluate.py: stored
        # raw pair vectors have priority over answer-level positive means.
        negative, target, q, _ = load_l2s_pairs(feature)
        with np.load(checkpoint, allow_pickle=False) as data:
            net = DirectVectorMLP(int(data['hidden_dim']), int(data['mlp_hidden'])).to(args.device)
            net.load_state_dict({k: torch.from_numpy(data['mlp_' + k]) for k in net.state_dict()})
            if item['split'] == 'mlp_train':
                mask = np.isin(q, data['mlp_train_q_indices'])
            elif item['dataset'] == 'truthfulqa':
                mask = np.isin(q, data['test_q_indices'])
            else:
                mask = np.ones(len(q), dtype=bool)
        negative, target, q = negative[mask], target[mask], q[mask]
        net.eval()
        cosine_chunks, magnitude_chunks = [], []
        with torch.inference_mode():
            for start in range(0, len(q), 256):
                x = torch.tensor(negative[start:start+256], device=args.device)
                # Retain this direct-MLP checkpoint's actual inference rule.
                predicted = net(F.normalize(x, p=2, dim=-1, eps=1e-8)).cpu().numpy()
                cosine, magnitude = rowwise_vector_metrics(predicted, target[start:start+256])
                cosine_chunks.append(cosine)
                magnitude_chunks.append(magnitude)
        cosine, magnitude = np.concatenate(cosine_chunks), np.concatenate(magnitude_chunks)
        aggregated = aggregate_by_question(q, cosine, magnitude)
        tags = {'dataset': item['dataset'], 'layer': int(item['layer']),
                'fold': int(item['fold']), 'split': item['split']}
        question_rows.extend({**tags, **r} for r in aggregated)
        fold_rows.append({**tags, 'questions': len(aggregated), 'pairs': len(q),
            'cosine_similarity': float(np.mean([r['cosine_similarity'] for r in aggregated])),
            'magnitude_relative_error': float(np.mean([r['magnitude_relative_error'] for r in aggregated])),
            'checkpoint': str(checkpoint), 'feature_path': str(feature)})
    write_csv(output / 'diagnostics_question.csv', question_rows)
    write_csv(output / 'fold_metrics.csv', fold_rows)
    summary = []
    for dataset, layer, split in dict.fromkeys((r['dataset'], r['layer'], r['split']) for r in fold_rows):
        # Reliability plots pool question rows across folds before taking the
        # centroid. Do the same, including slightly unequal TruthfulQA folds.
        selected = [r for r in question_rows if (r['dataset'], r['layer'], r['split']) == (dataset, layer, split)]
        summary.append({'dataset': dataset, 'layer': layer, 'split': split,
            'question_evaluations': len(selected), 'pair_evaluations': sum(r['num_pairs'] for r in selected),
            'cosine_similarity': float(np.mean([r['cosine_similarity'] for r in selected])),
            'magnitude_relative_error': float(np.mean([r['magnitude_relative_error'] for r in selected]))})
    write_csv(output / 'summary.csv', summary)
    (output / 'summary.json').write_text(json.dumps(summary, indent=2, allow_nan=False))
    lines = ['# Direct-MLP diagnostics using the L2S reliability protocol', '',
        'Reuses method/l2s_linear/model.py:load_l2s_pairs and experiments/l2s_reliability/metrics.py:rowwise_vector_metrics, aggregate_by_question. Target: raw v_T=z_pos_raw-z_neg_raw; stored pair fields take precedence. Direction: mean cosine similarity (higher is better), not angle in degrees. Magnitude: abs(||v_pred||-||v_T||)/||v_T|| (lower is better, dimensionless). Pair metrics are averaged within each question and question rows pooled across folds, exactly as the reliability plot centroid. Zero predictions have cosine 0 and relative norm error 1; near-zero targets are rejected. Existing direct-MLP weights and normalize(h_neg) input rule are retained. MLPs were trained against mean-positive targets; raw-pair diagnostics do not change their training targets. Generic evaluation uses the previously extracted official labeled splits (MMLU: repository balanced 2000-question test subset); TruthfulQA uses existing held-out folds. This evaluates the direct MLP, not a replacement L2S checkpoint.', '',
        '| Dataset | Layer | Split | Questions | Pairs | Cosine similarity ↑ | Relative norm error ↓ |',
        '|---|---:|---|---:|---:|---:|---:|']
    for r in summary:
        lines.append(f"| {r['dataset']} | {r['layer']} | {r['split']} | {r['question_evaluations']} | {r['pair_evaluations']} | {r['cosine_similarity']:.6f} | {r['magnitude_relative_error']:.6f} |")
    (output / 'report.md').write_text('\n'.join(lines) + '\n')
    print('\n'.join(lines), flush=True)


if __name__ == '__main__':
    main()
