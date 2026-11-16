"""Official task inputs and checkpoint routing; no test-based selection."""
from __future__ import annotations

import subprocess
from pathlib import Path
import numpy as np
import torch
from datasets import load_dataset, Dataset, concatenate_datasets

from method.S2S_R.generic.prompts_generic import (
    format_boolq_prompt, format_copa_prompt, format_storycloze_prompt,
    format_winogrande_prompt, format_mmlu_prompt,
)
from method.S2S_R.generic.utils_generic import MMLU_CATEGORIES
from method.S2S_R.eval_utils import (
    get_instruction, load_truthfulqa_mc_dataframe, format_best, split_multi_answer,
)
from method.S2S_R.evaluate_predictor import load_components

ROOT = Path(__file__).resolve().parents[2]
MODEL = 'llama3.1-8B-Instruct'
LAYERS = {'truthfulqa': 14, 'boolq': 12, 'copa': 16, 'storycloze': 27,
          'mmlu_global': 28, 'winogrande': 16}


def config(dataset):
    suffix = 'truthfulqa.sh' if dataset == 'truthfulqa' else f'generic/{dataset}.sh'
    def source(method, names):
        script = 'source "$1"; ' + 'printf "%s\\n" ' + ' '.join('"$' + n + '"' for n in names)
        result = subprocess.check_output(['bash', '-c', script, 'config',
            str(ROOT / f'hyperparams/{method}/LLAMA/{suffix}')], text=True).splitlines()
        return dict(zip(names, map(float, result)))
    dyn = source('S2S_R', ['WEIGHT_KAPPA', 'MARGIN'])
    sph = source('spherical_steering', ['KAPPA', 'ALPHA', 'BETA'])
    return {'weight_kappa': dyn['WEIGHT_KAPPA'], 'margin': dyn['MARGIN'],
            'spherical': {k.lower(): v for k, v in sph.items()}}


def items(dataset, tokenizer):
    output = []
    def add(base, choices, correct, positives=None, qid=None):
        output.append({'q_idx': len(output) if qid is None else int(qid), 'base': base,
            'choices': list(map(str, choices)), 'correct': int(correct),
            'positives': [int(correct)] if positives is None else positives})
    if dataset in ('boolq', 'copa'):
        for row in load_dataset('super_glue', dataset, split='validation'):
            if dataset == 'boolq':
                add(format_boolq_prompt(row['passage'], row['question']), ['no', 'yes'], row['label'])
            else:
                choices = [row['choice1'], row['choice2']]
                add(format_copa_prompt(row['premise'], row['question'], *choices), choices, row['label'])
    elif dataset == 'storycloze':
        for row in load_dataset('juletxara/xstory_cloze', 'en', split='eval'):
            choices = [row['sentence_quiz1'], row['sentence_quiz2']]
            story = [row[f'input_sentence_{i}'] for i in range(1, 5)]
            add(format_storycloze_prompt(story, *choices), choices, row['answer_right_ending'] - 1)
    elif dataset == 'winogrande':
        for row in load_dataset('winogrande', 'winogrande_xl', split='validation'):
            choices = [row['option1'], row['option2']]
            add(format_winogrande_prompt(row['sentence'], *choices), choices, int(row['answer']) - 1)
    elif dataset == 'mmlu_global':
        for category, subjects in MMLU_CATEGORIES.items():
            pool = concatenate_datasets([Dataset.from_parquet(str(Path('data/mmlu') /
                subject / 'test-00000-of-00001.parquet')) for subject in subjects]).shuffle(seed=42)
            for row in pool.select(range(500, min(1000, len(pool)))):
                add(format_mmlu_prompt(tokenizer, row['question'], row['choices']), row['choices'], row['answer'])
                output[-1]['category'] = category
    else:
        frame, _ = load_truthfulqa_mc_dataframe(str(ROOT / 'TruthfulQA/data/v1/TruthfulQA.csv'))
        for _, row in frame.iterrows():
            best = format_best(row['Best Answer'])
            true = list(dict.fromkeys([best] + split_multi_answer(row['Correct Answers'])))
            false = split_multi_answer(row['Incorrect Answers'])
            add(get_instruction(MODEL) + '\n\n' + f"Q: {row['Question']} A:",
                true + false, 0, list(range(len(true))), row['hf_idx'])
            output[-1]['question'] = str(row['Question'])
    return output


def tokens(item, tokenizer, dataset):
    prompt = tokenizer(item['base'], add_special_tokens=dataset == 'truthfulqa', return_tensors='pt').input_ids
    prefix = ' ' if item['base'].strip().endswith(':') else ''
    candidates = []
    for answer in item['choices']:
        option = tokenizer(prefix + answer, add_special_tokens=False, return_tensors='pt').input_ids
        candidates.append(torch.cat([prompt, option], 1))
    return candidates, prompt.shape[1] - 1


def components(dataset, qid, device, output):
    layer = LAYERS[dataset]
    fold = 0
    if dataset == 'truthfulqa':
        for f in (0, 1):
            cp = ROOT / f'results/reproduction/checkpoints/S2S_R/truthfulqa/{MODEL}/{MODEL}_layer14_predictor_fold{f}_mean.npz'
            with np.load(cp) as d:
                if qid in d['test_q_indices']:
                    fold = f
                    break
        else:
            raise ValueError('No held-out MLP for question')
        checkpoint = ROOT / f'results/reproduction/checkpoints/S2S_R/truthfulqa/{MODEL}/{MODEL}_layer14_predictor_fold{fold}_mean.npz'
        # Prototype and direct-MLP fold labels use different split conventions.
        # Route each bank independently by its recorded held-out questions.
        for prototype_fold in (0, 1):
            proto = ROOT / f'results/reproduction/prototypes/{MODEL}/{MODEL}_layer14_fold{prototype_fold}.npz'
            with np.load(proto) as d:
                if qid in d['test_q_indices']:
                    break
        else:
            raise ValueError('No held-out spherical prototype for question')
    else:
        checkpoint = ROOT / f'results/reproduction/generic/checkpoints/S2S_R/{MODEL}/{MODEL}_{dataset}_l{layer}_S2S_R.npz'
        if not checkpoint.exists():
            checkpoint = ROOT / f'results/experiment/rebuttal_prediction_error/official/checkpoints/{dataset}_production_e2000.npz'
        proto = ROOT / f'results/reproduction/generic/prototypes/spherical_steering/{MODEL}/{MODEL}_{dataset}_train_l{layer}_proto.npz'
        if not proto.exists():
            # Rebuild exactly the released generic prototype recipe, train only.
            feature = ROOT / f'results/reproduction/generic/features/S2S_R/{MODEL}/{MODEL}_{dataset}_train_l{layer}.npz'
            with np.load(feature) as d:
                x, y = d['activations'], d['labels']
                delta = x[y == 1].mean(0) - x[y == 0].mean(0)
            mu = delta / np.linalg.norm(delta)
            proto = output / f'{dataset}_train_prototype.npz'
            np.savez(proto, mu_T=mu, mu_H=-mu)
    data, average, predictor = load_components(str(checkpoint), torch.device(device))
    data.close()
    with np.load(proto) as d:
        prototypes = [torch.tensor(d[k], device=device) for k in ('mu_T', 'mu_H')]
        if dataset == 'truthfulqa' and qid not in d['test_q_indices']:
            raise ValueError('Spherical prototype is not held-out')
    return average, predictor, prototypes, fold, str(checkpoint), str(proto)
