"""Causal geometry/decision alignment study of released S2S-R checkpoints."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
import time

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from experiments.rebuttal_geometry_diagnostics.core import (
    GRID, SuffixEngine, cosine, dynamic_segment, spherical_segment,
    split_radial, equal_budget, norm, margin,
)
from experiments.rebuttal_geometry_diagnostics.data import (
    MODEL, LAYERS, items, tokens, components, config,
)
from methods.S2S_R.evaluate_S2S_R import get_s2s_r_intervention
from methods.spherical_steering.steering import get_spherical_intervention


def process(engine, tokenizer, item, dataset, hp, parts, verify=False, batch_size=4):
    model = engine.model
    average, predictor, prototypes, fold, checkpoint, proto = parts
    layer = LAYERS[dataset]
    candidate_ids, start = tokens(item, tokenizer, dataset)
    captures = [engine.capture(ids, layer) for ids in candidate_ids]
    positives = torch.stack([captures[j][0, -1].float() for j in item['positives']])
    negatives = [j for j in range(len(captures)) if j not in item['positives']]
    candidate_results, audits = [], []
    for index, (ids, captured) in enumerate(zip(candidate_ids, captures)):
        original = captured[0, start:].float()
        with torch.no_grad():
            predicted = predictor(norm(original[-1:]))[0]
            residual = norm(predicted - average)
            if index in negatives:
                nearest = torch.argmax(norm(positives) @ norm(original[-1]))
                target = positives[nearest] - original[-1]
                oracle = norm(target - average)
                if (target - average).norm() <= 1e-8:
                    oracle = norm(average)
            else:
                # Never invent a self-target for a correct candidate. Oracle
                # replacement is negative-only; positive predictions stay fixed.
                target, oracle = predicted, residual
            pred_edit, pred_gate, weights = dynamic_segment(original, average, residual,
                hp['weight_kappa'], hp['margin'])
            oracle_edit, oracle_gate, _ = dynamic_segment(original, average, oracle,
                hp['weight_kappa'], hp['margin'])
            edits = {'spherical': spherical_segment(original, prototypes, hp['spherical']),
                     'pred_adaptive': pred_edit, 'oracle_adaptive': oracle_edit}
            for source, r, active in [('pred', residual, pred_gate), ('oracle', oracle, oracle_gate)]:
                for k, triple in enumerate(GRID):
                    edits[f'{source}_grid{k}'] = dynamic_segment(original, average, r,
                        hp['weight_kappa'], hp['margin'], weights=triple, gate=active)[0]
            full = predicted.expand_as(original)
            radial, tangent = split_radial(full, original)
            _, mean_tangent = split_radial(average.expand_as(original), original)
            _, residual_tangent = split_radial(residual.expand_as(original), original)
            _, oracle_tangent = split_radial(target.expand_as(original), original)
            directions = {'full': full, 'radial': radial, 'tangent': tangent,
                          'mean': mean_tangent, 'residual': residual_tangent,
                          'oracle_tangent': oracle_tangent}
            for name, direction in directions.items():
                edits[f'probe_{name}'] = original + equal_budget(direction, original, .01)
            fd = equal_budget(full, original, .002)
            edits['fd_plus'] = original + fd
            edits['fd_minus'] = original - fd
            edits['probe_full_minus'] = original - equal_budget(full, original, .01)
        leaf = captured.detach().float().requires_grad_(True)
        score, gradient = engine.scores(leaf, ids, start, layer, dataset != 'truthfulqa', grad=True)
        gradient = gradient[0, start:].float()
        baseline = float(score[0])
        scores = engine.score_edits(captured, ids, start, layer, dataset != 'truthfulqa', edits, batch_size)
        scores['baseline'] = baseline
        linear = {name: float((gradient * (edited - original)).sum()) for name, edited in edits.items()}
        geometry = {'pred_target_cosine': cosine(predicted, target) if index in negatives else None,
            'residual_target_cosine': cosine(residual, target - average) if index in negatives else None,
            'radial_energy_fraction': float(radial.square().sum() / full.square().sum().clamp_min(1e-12)),
            'gate_fraction': float(pred_gate.float().mean()),
            'alpha': float(weights[:, 0].mean()), 'beta': float(weights[:, 1].mean()),
            'gamma': float(weights[:, 2].mean()),
            'mean_weight_prefers_residual': float((weights[:, 2] > weights[:, 1]).float().mean()),
            'gradient_prefers_residual': float(((gradient * equal_budget(residual_tangent, original)).sum(-1)
                > (gradient * equal_budget(mean_tangent, original)).sum(-1)).float().mean()),
            'rotation_deg': float(torch.rad2deg(torch.acos((norm(original) * norm(pred_edit)).sum(-1).clamp(-1, 1))).mean()),
            'gradient_edit_cosine': cosine(gradient, pred_edit - original),
            'geometric_edit_cosine': cosine(pred_edit - original, target.expand_as(original)) if index in negatives else None,
            'final_token_gradient_norm': float(gradient[-1].norm()),
            'segment_gradient_norm': float(gradient.norm()),
            'fd_derivative': (scores['fd_plus'] - scores['fd_minus']) / 2,
            'fd_linear': float((gradient * fd).sum()),
            'fd_large_derivative': (scores['probe_full'] - scores['probe_full_minus']) / 2,
            'fd_large_linear': linear['probe_full'],
            'weight_preference': (weights[:, 2] - weights[:, 1]).detach().cpu().tolist(),
            'decision_preference': ((gradient * equal_budget(residual_tangent, original)).sum(-1)
                - (gradient * equal_budget(mean_tangent, original)).sum(-1)).detach().cpu().tolist()}
        # Position-level diagnostics allow checking early/late answer tokens.
        n = len(original) - 1  # final answer token predicts no scored token
        geometry['position_bins'] = []
        for group in np.array_split(np.arange(n), min(4, n)):
            g = torch.tensor(group, device=original.device)
            geometry['position_bins'].append({'positions': len(group),
                'gamma': float(weights[g, 2].mean()),
                'gate': float(pred_gate[g].float().mean()),
                'linear_score_change': float((gradient[g] * (pred_edit[g] - original[g])).sum())})
        candidate_results.append({'index': index, 'positive': index in item['positives'],
            'answer_tokens': ids.shape[1] - start - 1, 'scores': scores, 'linear': linear,
            'geometry': geometry})
        if verify and index == 0:
            # Test exact suffix replay and hook parity at the real working strength.
            from baukit import TraceDict
            from functools import partial
            positions = torch.arange(start, ids.shape[1] - 1, device=engine.device)
            hook = get_s2s_r_intervention(average, predictor,
                weight_kappa=hp['weight_kappa'], margin=hp['margin'])
            sph_hook = get_spherical_intervention(*prototypes, **hp['spherical'])
            for name, edit_hook in [('baseline', None), ('pred_adaptive', hook), ('spherical', sph_hook)]:
                with torch.no_grad(), TraceDict(model, [f'model.layers.{layer}'],
                        edit_output=partial(edit_hook, start_idx=start) if edit_hook else None):
                    logits = model(ids.to(engine.device), use_cache=False, logits_to_keep=positions).logits.float()
                labels = ids.to(engine.device)[:, start+1:]
                lp = logits.log_softmax(-1).gather(-1, labels[..., None]).squeeze(-1)
                direct = float(lp.mean() if dataset != 'truthfulqa' else lp.sum())
                error = abs(direct - scores[name])
                audits.append({'condition': name, 'full_score': direct,
                    'suffix_score': scores[name], 'absolute_error': error})
                if error > 2e-2:
                    raise ValueError(f'Hook/suffix parity failed: {audits[-1]}')
    correct = item['correct']
    names = list(candidate_results[0]['scores'])
    all_scores = {name: [r['scores'][name] for r in candidate_results] for name in names}
    all_margins = {name: margin(value, correct, negatives) for name, value in all_scores.items()}
    competitor = max(negatives, key=lambda i: all_scores['baseline'][i])
    fixed = {name: value[correct] - value[competitor] for name, value in all_scores.items()}
    linear_margin = {name: candidate_results[correct]['linear'][name] - candidate_results[competitor]['linear'][name]
                     for name in candidate_results[0]['linear']}
    selections = {}
    for source in ('pred', 'oracle'):
        family = [f'{source}_adaptive'] + [f'{source}_grid{k}' for k in range(len(GRID))]
        winner = max(family, key=lambda key: all_margins[key])
        selections[source] = winner
        all_margins[f'{source}_selected'] = all_margins[winner]
    return {'dataset': dataset, 'q_idx': item['q_idx'], 'fold': fold,
        'correct': correct, 'negatives': negatives, 'baseline_competitor': competitor,
        'layer': layer, 'checkpoint': checkpoint, 'prototype': proto,
        'scoring': 'sum' if dataset == 'truthfulqa' else 'mean',
        'margins': all_margins, 'fixed_competitor_margins': fixed,
        'linear_margin_changes': linear_margin, 'selections': selections,
        'candidates': candidate_results, 'parity_audits': audits}


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--datasets', nargs='+', default=list(LAYERS), choices=list(LAYERS))
    parser.add_argument('--max-questions', type=int, default=128, help='0 for full evaluation sets')
    parser.add_argument('--seed', type=int, default=20261111)
    parser.add_argument('--batch-size', type=int, default=4)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--shard-count', type=int, default=1)
    parser.add_argument('--shard-index', type=int, default=0)
    parser.add_argument('--model-dir', default='models/Meta-Llama-3.1-8B-Instruct')
    parser.add_argument('--output', type=Path, default=ROOT / 'results/experiment/rebuttal_geometry_diagnostics')
    args = parser.parse_args()
    if args.shard_count < 1 or not 0 <= args.shard_index < args.shard_count:
        parser.error('shard-index must be in [0, shard-count)')
    args.output.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(4)
    torch.manual_seed(42)
    tokenizer = AutoTokenizer.from_pretrained(args.model_dir)
    tokenizer.pad_token = tokenizer.eos_token
    device = torch.device(args.device if not args.device.startswith('cuda') or torch.cuda.is_available() else 'cpu')
    dtype = torch.float16 if device.type == 'cuda' else torch.float32
    model = AutoModelForCausalLM.from_pretrained(args.model_dir, dtype=dtype,
        device_map={'': str(device)}, attn_implementation='sdpa').eval()
    engine = SuffixEngine(model)
    for dataset in args.datasets:
        hp = config(dataset)
        data = items(dataset, tokenizer)
        rng = np.random.default_rng(args.seed)
        count = len(data) if args.max_questions == 0 or dataset == 'copa' else min(args.max_questions, len(data))
        selected = np.sort(rng.choice(len(data), count, replace=False))
        folder = args.output / dataset
        folder.mkdir(exist_ok=True)
        spec = {'version': 1, 'dataset': dataset, 'seed': args.seed,
            'model_dir': args.model_dir, 'dtype': 'float16', 'hp': hp,
            'selected_q_indices': [data[int(i)]['q_idx'] for i in selected],
            'grid': GRID, 'probe_norm_fraction': .01, 'fd_norm_fraction': .002,
            'oracle': 'negative-only nearest-positive; positive candidate predictions unchanged',
            'grid_gate': 'held fixed to each residual source adaptive gate',
            'selection': 'one setting per question across all candidates; label-dependent analysis only'}
        signature = hashlib.sha256(json.dumps(spec, sort_keys=True).encode()).hexdigest()
        manifest = folder / 'manifest.json'
        if manifest.exists() and json.loads(manifest.read_text())['signature'] != signature:
            raise ValueError(f'Output configuration mismatch: {folder}')
        manifest_temp = folder / f'manifest.shard{args.shard_index}.tmp'
        manifest_temp.write_text(json.dumps({**spec, 'signature': signature}, indent=2))
        manifest_temp.replace(manifest)
        parts_cache = {}
        for position, idx in enumerate(selected):
            if position % args.shard_count != args.shard_index:
                continue
            item = data[int(idx)]
            path = folder / f"q{item['q_idx']:05d}.json"
            if path.exists():
                continue
            begin = time.time()
            key = item['q_idx'] if dataset == 'truthfulqa' else 0
            if dataset == 'truthfulqa':
                # Only two tiny predictor/prototype banks; route by held-out fold.
                parts = components(dataset, key, engine.device, folder)
            else:
                if key not in parts_cache:
                    parts_cache[key] = components(dataset, key, engine.device, folder)
                parts = parts_cache[key]
            result = process(engine, tokenizer, item, dataset, hp, parts,
                             verify=position == args.shard_index, batch_size=args.batch_size)
            result['signature'] = signature
            temporary = path.with_suffix('.tmp')
            temporary.write_text(json.dumps(result, allow_nan=False))
            temporary.replace(path)
            print(f"{dataset} {position+1}/{len(selected)} q={item['q_idx']} "
                  f"elapsed={time.time()-begin:.1f}s baseline={result['margins']['baseline']:.3f} "
                  f"spherical={result['margins']['spherical']:.3f} s2s={result['margins']['pred_adaptive']:.3f}", flush=True)


if __name__ == '__main__':
    main()
