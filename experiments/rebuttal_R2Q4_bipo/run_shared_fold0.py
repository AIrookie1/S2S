"""Evaluate both folds with corrected fold-0-selected S2S hyperparameters."""
from __future__ import annotations
import hashlib
import json
import shutil
import sys
from pathlib import Path

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
import numpy as np
import torch
from transformers import AutoModelForCausalLM,AutoTokenizer
from experiments.rebuttal_R2Q4_bipo.rebuttal_R2Q4_run import evaluate,make_hook,load_truthfulqa_mc_dataframe,write_json
from methods.S2S_R.predictor import DirectVectorMLP

BASE=ROOT/'results/experiment/rebuttal_R2Q4_bipo'
RETUNED=BASE/'mlp_s2s_retuned'
OUTPUT=BASE/'mlp_s2s_shared_fold0'
METRICS=('MC1','MC2','MC3')


def verified_rows(path,ids):
    rows=json.loads(path.read_text())
    if len(rows)!=len(ids) or {r['q_idx'] for r in rows}!=set(ids):
        raise ValueError(f'Coverage mismatch: {path}')
    if any(not np.isfinite(r[m]) for r in rows for m in METRICS):
        raise ValueError(f'Nonfinite metrics: {path}')
    return rows


def load_predictor(path,train_ids,device):
    with np.load(path,allow_pickle=False) as data:
        if set(data['mlp_train_q_indices'])!=set(train_ids):
            raise ValueError('MLP trained on unexpected questions')
        predictor=DirectVectorMLP(int(data['hidden_dim']),int(data['mlp_hidden']))
        state={key:torch.from_numpy(data['mlp_'+key].copy()) for key in predictor.state_dict()}
    predictor.load_state_dict(state)
    return predictor.to(device).eval()


def main():
    torch.set_num_threads(4)
    OUTPUT.mkdir(exist_ok=True)
    selected={method:json.loads((RETUNED/f'fold0_{method}_selection.json').read_text())['chosen']['parameters']
              for method in ('bipo_mlp_s2s_l','bipo_mlp_s2s_r')}
    assert selected=={'bipo_mlp_s2s_l':dict(strength=1.,slerp_t=.2),
                      'bipo_mlp_s2s_r':dict(weight_kappa=1.,margin=-.3)}
    parameters={'bipo':dict(strength=1.),**selected}
    split=[json.loads((BASE/f'fold{fold}/split.json').read_text()) for fold in (0,1)]
    config=dict(parameters_shared_across_folds=parameters,
                model='Llama-3.1-8B-Instruct',layer=14,dtype='float16',
                selection='Human requested corrected fold0 dev-optimal settings for both folds; no further search',
                scope='Inference hyperparameters shared; independently fitted fold checkpoints retained',
                padding='MLP hook receives each real unpadded answer',
                limitation='Fold0 development questions overlap fold1 test questions. This is a user-requested shared-setting re-evaluation; pooled scores are not a new independent test of hyperparameter selection.')
    manifest=OUTPUT/'config.json'
    if manifest.exists() and json.loads(manifest.read_text())!=config:raise ValueError('Configuration mismatch')
    write_json(manifest,config)
    model=tokenizer=cache=None
    results=[]
    for fold in (0,1):
        folder=OUTPUT/f'fold{fold}';folder.mkdir(exist_ok=True)
        for method,params in parameters.items():
            path=folder/f'test_{method}.json'
            source=None
            if method=='bipo':
                chosen=json.loads((BASE/f'fold{fold}/selection_bipo.json').read_text())['chosen']['parameters']
                if chosen!=params:raise ValueError('Baseline strength mismatch')
                source=BASE/f'fold{fold}/test_bipo.json'
            elif fold==0:source=RETUNED/f'fold0/test_retuned_{method}.json'
            if not path.exists() and source is not None:
                verified_rows(source,split[fold]['test'])
                shutil.copyfile(source,path)
            if source is not None:
                rows=verified_rows(path,split[fold]['test'])
                metrics={m:float(np.mean([r[m] for r in rows])) for m in METRICS}
            else:
                if model is None:
                    location=json.loads((BASE/'config.json').read_text())['model_dir']
                    model=AutoModelForCausalLM.from_pretrained(location,torch_dtype=torch.float16,
                        attn_implementation='sdpa',local_files_only=True,low_cpu_mem_usage=True,
                        device_map={'model.embed_tokens':0,
                            **{f'model.layers.{i}':('cpu' if i<=14 else 0) for i in range(32)},
                            'model.norm':0,'lm_head':0}).eval()
                    for p in model.parameters():p.requires_grad_(False)
                    tokenizer=AutoTokenizer.from_pretrained(location,local_files_only=True)
                    records=torch.load(BASE/'prefix_cache.pt',weights_only=False,map_location='cpu')
                    cache={tuple(r['ids'].tolist()):r['hidden'] for r in records}
                    frame,_=load_truthfulqa_mc_dataframe(str(ROOT/'data/truthfulqa/TruthfulQA.csv'))
                device=model.device
                predictor=load_predictor(BASE/f'fold{fold}/mlp_predictor.npz',split[fold]['train'],device)
                bipo=torch.load(BASE/f'fold{fold}/bipo.pt',weights_only=False,map_location='cpu')['vector'].to(device)
                hook=make_hook('s2s_l' if method.endswith('_l') else 's2s_r',bipo,bipo,params,predictor=predictor)
                metrics=evaluate(model,tokenizer,model.model.layers[14],frame,split[fold]['test'],hook,path,8,cache)
                rows=verified_rows(path,split[fold]['test'])
            results.append(dict(fold=fold,method=method,parameters=params,n_questions=len(rows),
                                reused_from=str(source) if source else None,**metrics))
            write_json(OUTPUT/'fold_metrics.json',results)
            print('RESULT',results[-1],flush=True)
    pooled=[]
    for method in parameters:
        rows=[r for fold in (0,1) for r in verified_rows(OUTPUT/f'fold{fold}/test_{method}.json',split[fold]['test'])]
        if len(rows)!=817 or len({r['q_idx'] for r in rows})!=817:raise ValueError('Pooled coverage mismatch')
        pooled.append(dict(method=method,parameters=parameters[method],n_questions=len(rows),
                           **{m:float(np.mean([r[m] for r in rows])) for m in METRICS}))
    write_json(OUTPUT/'performance.json',pooled)
    lines=['# Same inference hyperparameters in both folds','',
           'User-requested corrected Fold 0 development-optimal settings, applied identically to both folds.',
           'BIPO and MLP checkpoints remain independently fitted per fold; MLPs use only their fold training questions.',
           'MLP input uses the actual last nonpadding answer token. No test-based parameter search was run.',
           '', '| Method | Shared parameters | MC1 (%) | MC2 (%) | MC3 (%) | N |',
           '|---|---|---:|---:|---:|---:|']
    for r in pooled:
        lines.append(f"| {r['method']} | `{json.dumps(r['parameters'],sort_keys=True)}` | "+
                     ' | '.join(f'{100*r[m]:.2f}' for m in METRICS)+f" | {r['n_questions']} |")
    lines+=['','## Per fold','','| Fold | Method | MC1 (%) | MC2 (%) | MC3 (%) | N |','|---|---|---:|---:|---:|---:|']
    for r in results:
        lines.append(f"| {r['fold']} | {r['method']} | "+' | '.join(f'{100*r[m]:.2f}' for m in METRICS)+f" | {r['n_questions']} |")
    lines+=['','## Interpretation','',config['limitation'],
            'The shared hyperparameters were chosen on Fold 0 development questions; some of these questions are part of Fold 1 test. Thus these pooled scores describe the requested common-setting comparison, not leakage-free nested cross-validation. Original per-fold tuning reports remain separate.']
    (OUTPUT/'REPORT.md').write_text('\n'.join(lines)+'\n')
    write_json(OUTPUT/'audit.json',dict(coverage='817 unique held-out predictions per method',
        shared_parameters=parameters,source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()))
    print('\n'.join(lines),flush=True)

if __name__=='__main__':main()
