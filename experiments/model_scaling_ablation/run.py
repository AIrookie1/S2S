#!/usr/bin/env python3
"""Five-method comparison on Qwen3.5-9B / Qwen2.5-32B-Instruct."""
from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path
import subprocess
import sys

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT))
from experiments.model_scaling_ablation.common import (
    METHODS, MODELS, aggregate, execution_device, fingerprint, load_model, model_identity, parameter_grid,
    source_identity, write_json,
)


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model', choices=MODELS, required=True)
    p.add_argument('--model-dir', help='Downloaded official model directory (or pinned Hub repository)')
    p.add_argument('--output-dir', type=Path)
    p.add_argument('--layer', type=int)
    p.add_argument('--device', choices=['cuda', 'cpu'], default='cuda')
    p.add_argument('--dtype', choices=['bfloat16', 'float16', 'float32'], default='bfloat16')
    p.add_argument('--cpu-memory', default='40GiB')
    p.add_argument('--load-in-4bit', action='store_true', help='NF4 for ALL five methods; recorded separately')
    p.add_argument('--epochs', type=int, default=2000)
    p.add_argument('--mlp-hidden', type=int, default=256)
    p.add_argument('--batch-size', type=int, default=256)
    p.add_argument('--learning-rate', type=float, default=0.001)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--dev-fraction', type=float, default=0.2)
    p.add_argument('--no-tune', action='store_true', help='Use declared fixed hyperparameters; no dev search')
    p.add_argument('--smoke', action='store_true', help='Only 12 questions, 2 epochs, one parameter cell; never a paper result')
    p.add_argument('--dry-run', action='store_true')
    args = p.parse_args()
    if args.epochs <= 0 or args.mlp_hidden <= 0 or args.batch_size <= 0 or args.learning_rate <= 0:
        p.error('Training settings must be positive')
    if not 0 < args.dev_fraction < 0.5:
        p.error('dev-fraction must be in (0,0.5)')
    if args.load_in_4bit and args.device != 'cuda':
        p.error('4-bit mode requires CUDA')
    repo, revision, layer = MODELS[args.model]
    output = args.output_dir or ROOT / 'results/experiment/model_scaling_ablation' / repo.split('/')[-1] / (
        ('nf4' if args.load_in_4bit else args.dtype) + ('_smoke' if args.smoke else ''))
    config = dict(model_key=args.model, repo_id=repo, revision=revision,
                  model_dir=str(Path(args.model_dir).resolve()) if args.model_dir and Path(args.model_dir).is_dir() else (args.model_dir or repo),
                  layer=args.layer if args.layer is not None else layer,
                  output_dir=str(output.resolve()), device=args.device, dtype=args.dtype,
                  cpu_memory=args.cpu_memory, load_in_4bit=args.load_in_4bit,
                  epochs=2 if args.smoke else args.epochs, mlp_hidden=args.mlp_hidden,
                  batch_size=args.batch_size, learning_rate=args.learning_rate, seed=args.seed,
                  dev_fraction=args.dev_fraction, tune=not (args.no_tune or args.smoke),
                  smoke=args.smoke, source=source_identity(),
                  prompt='repository_zero_shot_instruction_no_chat_template',
                  feature_prompt='repository_standard_Q_question_A_choice',
                  s2s_l_variant='S2S-L (current get_2d_dynamic_intervention)',
                  s2s_r_variant='S2S-R (current get_3d_dynamic_intervention)',
                  additive='raw difference of answer-class centroids; geometry training questions only',
                  l2s='repository L2S-LINEAR, raw Cartesian pair targets',
                  selection='dev_MC1_then_MC2_then_grid_order' if not (args.no_tune or args.smoke) else 'fixed_parameters')
    config['model_identity'] = model_identity(config['model_dir'], repo, revision)
    if config['layer'] < 0:
        p.error('layer must be nonnegative')
    return args, config


def prepare_features(model, tokenizer, block, config, output):
    import numpy as np
    import torch
    from tqdm import tqdm
    from method.dyn3d_sga.data import get_truthfulqa_data
    data = get_truthfulqa_data(style='standard')
    keep = data['q_indices'] < 12 if config['smoke'] else np.ones(len(data['labels']), dtype=bool)
    values = []
    captured = []
    def capture(_module, _inputs, result):
        h = result[0] if isinstance(result, (tuple, list)) else result
        captured.append(h[0, -1].detach().float().cpu().numpy().copy())
    handle = block.register_forward_hook(capture)
    try:
        device = execution_device(model.get_input_embeddings())
        with torch.inference_mode():
            for prompt in tqdm(data['prompts'][keep], desc='Extract raw block outputs'):
                captured.clear()
                inputs = tokenizer(str(prompt), return_tensors='pt').to(device)
                model(**inputs, use_cache=False, logits_to_keep=1)
                if len(captured) != 1:
                    raise RuntimeError('Expected exactly one activation per forward pass')
                values.append(captured[0])
    finally:
        handle.remove()
    values = np.stack(values)
    if not np.isfinite(values).all():
        raise ValueError('Non-finite block activations')
    output.parent.mkdir(parents=True, exist_ok=True)
    tmp = output.with_suffix('.tmp.npz')
    np.savez_compressed(tmp, activations=values,
                        **{key: data[key][keep] for key in ('labels', 'q_indices', 'choice_indices', 'choices')},
                        layer=np.asarray(config['layer']), model_name=np.asarray(config['repo_id']))
    tmp.replace(output)


def prepare_fold(feature, config, fold, folder):
    import numpy as np
    from method.l2s_linear.model import split_like_dyn3d
    folder.mkdir(parents=True, exist_ok=True)
    with np.load(feature, allow_pickle=False) as file:
        fields = {key: file[key] for key in file.files}
    questions = np.unique(fields['q_indices'])
    halves = np.array_split(questions, 2)
    outer, test = halves[fold], halves[1 - fold]
    shuffled = np.random.default_rng(config['seed'] + fold).permutation(outer)
    n_dev = max(1, round(len(outer) * config['dev_fraction']))
    dev, training = np.sort(shuffled[:n_dev]), np.sort(shuffled[n_dev:])
    if len(training) < 2:
        raise ValueError('Insufficient questions for mean/MLP training')
    # The existing trainers interpret test_q_indices as the selected training
    # fold and train_q_indices as its complement. Preserve that convention.
    metadata = dict(test_q_indices=training, train_q_indices=np.sort(np.r_[test, dev]))
    split = split_like_dyn3d(fields['q_indices'], metadata, num_folds=2, fold_idx=fold,
                            val_fraction=0.5, seed=config['seed'])
    split_record = dict(outer_train=outer.tolist(), geometry=split['geometry_train_q_indices'].tolist(),
                        predictor=split['predictor_q_indices'].tolist(), dev=dev.tolist(), test=test.tolist())
    groups = [set(split_record[key]) for key in ('geometry', 'predictor', 'dev', 'test')]
    if set.union(*groups) != set(questions) or sum(map(len, groups)) != len(questions):
        raise ValueError('Question leakage in train/dev/test split')
    write_json(folder / 'split.json', split_record)
    train_feature = folder / 'training_features.npz'
    np.savez_compressed(train_feature, **fields, **metadata)
    for method, script in [('l2s', 'method/l2s_linear/train_l2s.py'),
                           ('s2s_l', 'method/dyn3d_sga/train_dyn_mlp.py'),
                           ('s2s_r', 'method/dyn3d_sga/train_dyn_mlp.py')]:
        checkpoint = folder / ('s2s.npz' if method == 's2s_r' else f'{method}.npz')
        if checkpoint.exists():
            try:
                with np.load(checkpoint, allow_pickle=False) as saved:
                    # Read every compressed array to reject interrupted writes.
                    for name in saved.files:
                        saved[name]
                    if set(saved['train_q_indices']) != set(training):
                        raise ValueError('Checkpoint split mismatch')
                continue
            except (OSError, ValueError, EOFError):
                checkpoint.unlink()
        # Train the tiny predictor on CPU while the base model remains resident.
        command = [sys.executable, str(ROOT / script), '--feature_path', str(train_feature),
                   '--output_path', str(checkpoint), '--fold_idx', str(fold), '--num_folds', '2',
                   '--val_fraction', '0.5', '--epochs', str(config['epochs']),
                   '--mlp_hidden', str(config['mlp_hidden']), '--batch_size', str(config['batch_size']),
                   '--learning_rate', str(config['learning_rate']), '--seed', str(config['seed']), '--device', 'cpu']
        with (folder / f'train_{method}.log').open('w') as log:
            subprocess.run(command, cwd=ROOT, check=True, stdout=log, stderr=subprocess.STDOUT,
                           env={**os.environ, 'OMP_NUM_THREADS': '4', 'MKL_NUM_THREADS': '4'})
    # Unweighted answer-class centroids: the conventional static additive control.
    mask = np.isin(fields['q_indices'], split['geometry_train_q_indices'])
    h, y = fields['activations'][mask], fields['labels'][mask]
    diff = h[y == 1].mean(0) - h[y == 0].mean(0)
    norm = np.linalg.norm(diff)
    if not np.isfinite(diff).all() or norm < 1e-12:
        raise ValueError('Invalid static direction')
    np.savez_compressed(folder / 'geometry.npz', diff=diff, mu_T=diff / norm, mu_H=-diff / norm)
    return split_record


def make_hook(method, parameters, folder, block):
    import numpy as np
    import torch
    from method.dyn3d_sga.eval_utils import get_additive_intervention
    from method.spherical_steering.steering import get_spherical_intervention
    from method.l2s_linear.model import load_l2s_checkpoint
    from method.l2s_linear.steering import get_l2s_linear_intervention
    from method.dyn3d_sga.evaluate_dyn_mlp import load_components
    from method.dyn3d_sga.evaluate_3d_dyn import get_3d_dynamic_intervention
    # Accelerate's execution device can differ from a CPU-offloaded parameter.
    device = execution_device(block)
    if method == 'baseline':
        return None
    if method in ('additive', 'spherical'):
        with np.load(folder / 'geometry.npz') as geometry:
            tensors = {key: torch.tensor(geometry[key], device=device) for key in geometry.files}
        if method == 'additive':
            return get_additive_intervention(tensors['diff'], **parameters)
        return get_spherical_intervention(tensors['mu_T'], tensors['mu_H'], **parameters)
    if method == 'l2s':
        checkpoint, predictor = load_l2s_checkpoint(folder / 'l2s.npz', device)
        checkpoint.close()
        return get_l2s_linear_intervention(predictor, **parameters)
    checkpoint_name = 's2s_l.npz' if method == 's2s_l' else 's2s.npz'
    checkpoint, average, predictor = load_components(str(folder / checkpoint_name), device)
    checkpoint.close()
    if method == 's2s_l':
        from method.dyn2d.steering import get_2d_dynamic_intervention
        return get_2d_dynamic_intervention(average, predictor, method='S2S_L_Static', **parameters)
    return get_3d_dynamic_intervention(average, predictor, **parameters)


def score_answer(model, tokenizer, block, question, answer, hook, model_name):
    import torch
    from method.dyn3d_sga.eval_utils import get_instruction
    base = get_instruction(model_name) + '\n\n' + f'Q: {question} A:'
    device = execution_device(model.get_input_embeddings())
    prompt = tokenizer(base, return_tensors='pt').input_ids.to(device)
    choice = tokenizer(' ' + answer, add_special_tokens=False, return_tensors='pt').input_ids.to(device)
    if choice.shape[1] == 0:
        raise ValueError('Empty answer tokenization')
    start = prompt.shape[1] - 1
    ids = torch.cat([prompt, choice], dim=1)
    def edit(_module, _inputs, result):
        return hook(result, layer_name=None, start_idx=start)
    handle = block.register_forward_hook(edit) if hook else None
    try:
        with torch.inference_mode():
            logits = model(input_ids=ids, use_cache=False, logits_to_keep=choice.shape[1] + 1).logits[0, :-1].float()
            labels = choice[0].to(logits.device)
            score = -torch.nn.functional.cross_entropy(logits, labels, reduction='sum').item()
    finally:
        if handle:
            handle.remove()
    if not __import__('math').isfinite(score):
        raise ValueError('Non-finite answer log probability')
    return score


def evaluate(model, tokenizer, block, frame, ids, hook, config):
    import numpy as np
    from tqdm import tqdm
    from method.dyn3d_sga.eval_utils import format_best, split_multi_answer
    from truthfulqa.models import MC_calcs, set_columns
    frame = frame[frame['hf_idx'].isin(ids)].copy().reset_index(drop=True)
    if set(map(int, frame['hf_idx'])) != set(ids):
        raise ValueError('Incomplete CSV/HF question mapping')
    tag = config['repo_id']
    set_columns(tag, frame)
    details = []
    for i in tqdm(range(len(frame)), desc='TruthfulQA MC', leave=False):
        row = frame.iloc[i]
        true, false = split_multi_answer(row['Correct Answers']), split_multi_answer(row['Incorrect Answers'])
        if not true or not false:
            raise ValueError('Missing answer class')
        scores = [[score_answer(model, tokenizer, block, row['Question'], answer, hook, tag)
                   for answer in answers] for answers in (true, false)]
        MC_calcs(tag, frame, i, *scores, true, format_best(row['Best Answer']))
        details.append(dict(question_id=int(row['hf_idx']), question=row['Question'],
                            true_answers=true, false_answers=false, true_scores=scores[0], false_scores=scores[1],
                            **{metric: float(frame.loc[i, f'{tag} {metric}']) for metric in ('MC1', 'MC2', 'MC3')}))
    metrics = {key: float(np.mean([d[key] for d in details])) for key in ('MC1', 'MC2', 'MC3')}
    return metrics, details


def main():
    args, config = parse_args()
    if args.dry_run:
        print(json.dumps({**config, 'parameter_grid': {m: parameter_grid(m, config['tune']) for m in METHODS}}, indent=2))
        return
    import torch
    import transformers
    import numpy as np
    from method.dyn3d_sga.eval_utils import load_truthfulqa_mc_dataframe, set_seed
    torch.set_num_threads(4)
    set_seed(config['seed'])
    output = Path(config['output_dir'])
    # A configuration mismatch must never silently reuse an artifact.
    manifest = output / 'run_config.json'
    if manifest.exists() and json.loads(manifest.read_text()) != config:
        raise ValueError(f'Configuration/source changed: choose a new --output-dir ({output})')
    write_json(manifest, config)
    write_json(output / 'status.json', dict(state='loading', run=fingerprint(config), pid=os.getpid()))
    model, tokenizer, layer_path, layers = load_model(config)
    block = layers[config['layer']]
    write_json(output / 'runtime.json', dict(torch=torch.__version__, transformers=transformers.__version__,
                                           layer_path=f"{layer_path}.{config['layer']}",
                                           devices={str(k): str(v) for k, v in getattr(model, 'hf_device_map', {}).items()},
                                           model_class=type(model).__name__,
                                           base_parameter_dtypes=sorted({str(p.dtype) for p in model.parameters() if p.is_floating_point()})))
    feature = output / 'features.npz'
    if not feature.exists():
        write_json(output / 'status.json', dict(state='extracting', pid=os.getpid()))
        prepare_features(model, tokenizer, block, config, feature)
    frame, dataset = load_truthfulqa_mc_dataframe(str(ROOT / 'TruthfulQA/data/v1/TruthfulQA.csv'))
    reports = []
    for fold in (0, 1):
        folder = output / f'fold{fold}'
        write_json(output / 'status.json', dict(state='training', fold=fold, pid=os.getpid()))
        split = prepare_fold(feature, config, fold, folder)
        for method in METHODS:
            path = folder / f'{method}_test.json'
            if path.exists():
                report = json.loads(path.read_text())
                if report['question_ids'] != split['test'] or report['run'] != fingerprint(config):
                    raise ValueError('Mismatched cached report')
                reports.append(report)
                continue
            grid = parameter_grid(method, config['tune'])
            candidates = []
            if len(grid) > 1:
                for index, parameters in enumerate(grid):
                    cell = folder / 'dev' / f'{method}_{index}.json'
                    if cell.exists():
                        candidate = json.loads(cell.read_text())
                    else:
                        hook = make_hook(method, parameters, folder, block)
                        metrics, details = evaluate(model, tokenizer, block, frame, split['dev'], hook, config)
                        candidate = dict(parameters=parameters, metrics=metrics, details=details)
                        write_json(cell, candidate)
                    candidates.append(candidate)
                chosen = max(candidates, key=lambda c: (c['metrics']['MC1'], c['metrics']['MC2']))['parameters']
            else:
                chosen = grid[0]
            write_json(folder / f'{method}_selection.json', dict(selection=config['selection'], parameters=chosen,
                                                               dev_question_ids=split['dev'], candidates=candidates))
            write_json(output / 'status.json', dict(state='evaluating', fold=fold, method=method, pid=os.getpid()))
            hook = make_hook(method, chosen, folder, block)
            metrics, details = evaluate(model, tokenizer, block, frame, split['test'], hook, config)
            report = dict(run=fingerprint(config), fold=fold, method=method, parameters=chosen,
                          question_ids=split['test'], metrics=metrics, details=details)
            write_json(path, report)
            reports.append(report)
            print(f'{method} fold={fold}: {metrics}', flush=True)
    rows = aggregate(reports)
    expected = 12 if config['smoke'] else len(dataset)
    if any(row['n_questions'] != expected for row in rows):
        raise ValueError('Incomplete final evaluation coverage')
    write_json(output / 'summary.json', dict(config=config, rows=rows, folds=reports))
    with (output / 'summary.csv').open('w') as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    with (output / 'summary.md').open('w') as file:
        file.write(('SMOKE TEST — NOT A PAPER RESULT\n\n' if config['smoke'] else '') +
                   '| Method | MC1 | MC2 | MC3 | Avg. | N |\n|---|---:|---:|---:|---:|---:|\n')
        for row in rows:
            file.write(f"| {row['method']} | " + ' | '.join(f"{100 * row[k]:.2f}" for k in ('MC1', 'MC2', 'MC3', 'Avg')) + f" | {row['n_questions']} |\n")
    write_json(output / 'status.json', dict(state='complete', smoke=config['smoke'], pid=os.getpid()))
    print(json.dumps(rows, indent=2))


if __name__ == '__main__':
    try:
        main()
    except Exception:
        import traceback
        traceback.print_exc()
        raise SystemExit(1)
