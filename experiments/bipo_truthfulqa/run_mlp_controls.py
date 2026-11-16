"""MLP controls without BiPO: mean-anchor S2S-L and raw additive steering."""
from __future__ import annotations
import json
import sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM,AutoTokenizer
from experiments.bipo_truthfulqa.run import evaluate,make_hook,load_truthfulqa_mc_dataframe,write_json
from experiments.bipo_truthfulqa.run_shared_fold0 import load_predictor,verified_rows,METRICS
BASE=ROOT/'results/experiment/bipo_truthfulqa'
OUTPUT=BASE/'mlp_controls_shared'


def additive_mlp_hook(predictor,alpha=1.):
    """Match S2S sequence probe/normalization/broadcast, with raw additive geometry."""
    def hook(output,start_idx=0,**kwargs):
        h=output[0] if isinstance(output,tuple) else output
        start=max(0,min(int(start_idx),h.shape[1]-1))
        segment=h[:,start:].float()
        prediction=predictor(F.normalize(segment[:,-1],dim=-1))
        edited=h.clone()
        edited[:,start:]=(segment+alpha*prediction[:,None]).to(h.dtype)
        return (edited,)+output[1:] if isinstance(output,tuple) else edited
    return hook


def main():
    torch.set_num_threads(4)
    OUTPUT.mkdir(exist_ok=True)
    config=dict(methods={'mlp_mean_s2s_l':dict(strength=1.,slerp_t=.2),
                         'mlp_additive':dict(alpha=1.)},
                mean='training-only Cartesian raw answer difference, existing mean.pt',
                probe='normalized true last nonpadding token, one prediction broadcast per answer',
                model='Llama-3.1-8B-Instruct',layer=14,all_token_intervention=True,
                shared_parameters=True,selection='fixed settings matching preceding common-setting comparison; no tuning')
    manifest=OUTPUT/'config.json'
    if manifest.exists() and json.loads(manifest.read_text())!=config:raise ValueError('Config mismatch')
    write_json(manifest,config)
    location=json.loads((BASE/'config.json').read_text())['model_dir']
    model=AutoModelForCausalLM.from_pretrained(location,torch_dtype=torch.float16,
        attn_implementation='sdpa',local_files_only=True,low_cpu_mem_usage=True,
        device_map={'model.embed_tokens':0,**{f'model.layers.{i}':('cpu' if i<=14 else 0) for i in range(32)},
                    'model.norm':0,'lm_head':0}).eval()
    for p in model.parameters():p.requires_grad_(False)
    tokenizer=AutoTokenizer.from_pretrained(location,local_files_only=True)
    records=torch.load(BASE/'prefix_cache.pt',weights_only=False,map_location='cpu')
    cache={tuple(r['ids'].tolist()):r['hidden'] for r in records}
    frame,_=load_truthfulqa_mc_dataframe(str(ROOT/'TruthfulQA/data/v1/TruthfulQA.csv'))
    fold_rows=[]
    pooled={m:[] for m in config['methods']}
    for fold in (0,1):
        source=BASE/f'fold{fold}'
        split=json.loads((source/'split.json').read_text())
        folder=OUTPUT/f'fold{fold}';folder.mkdir(exist_ok=True)
        predictor=load_predictor(source/'mlp_predictor.npz',split['train'],model.device)
        average=torch.load(source/'mean.pt',weights_only=True,map_location='cpu').to(model.device)
        for method,params in config['methods'].items():
            hook=make_hook('s2s_l',None,average,params,predictor=predictor) if method=='mlp_mean_s2s_l' else additive_mlp_hook(predictor,**params)
            path=folder/f'test_{method}.json'
            metrics=evaluate(model,tokenizer,model.model.layers[14],frame,split['test'],hook,path,8,cache)
            rows=verified_rows(path,split['test'])
            pooled[method].extend(rows)
            fold_rows.append(dict(fold=fold,method=method,n_questions=len(rows),parameters=params,**metrics))
            write_json(OUTPUT/'fold_metrics.json',fold_rows)
            print('RESULT',fold_rows[-1],flush=True)
    summary=[]
    for method,rows in pooled.items():
        assert len(rows)==len({r['q_idx'] for r in rows})==817
        summary.append(dict(method=method,parameters=config['methods'][method],n_questions=len(rows),
                            **{m:float(np.mean([r[m] for r in rows])) for m in METRICS}))
    write_json(OUTPUT/'performance.json',summary)
    previous=json.loads((BASE/'mlp_s2s_shared_fold0/performance.json').read_text())
    lines=['# MLP-only controls and shared-parameter BiPO comparison','',
           'All five methods use the same fold-specific checkpoints, same held-out questions, true nonpadding MLP probe and all-token intervention.',
           'MLP mean-anchor S2S-L uses the training-only mean.pt instead of the BiPO vector.',
           'MLP additive uses h_new = h + alpha*v_hat, alpha=1; this is additive steering, not convex hidden-state interpolation.',
           '', '| Method | Parameters (same in both folds) | MC1 (%) | MC2 (%) | MC3 (%) | N |',
           '|---|---|---:|---:|---:|---:|']
    for r in previous+summary:
        lines.append(f"| {r['method']} | `{json.dumps(r['parameters'],sort_keys=True)}` | "+' | '.join(f'{100*r[m]:.2f}' for m in METRICS)+f" | {r['n_questions']} |")
    lines+=['','Mean and MLP training use only each fold training questions. Parameters are fixed, no new test-set search. The BiPO/S2S settings were selected on Fold 0 development data, which overlaps Fold 1 test; the combined table is a descriptive common-setting comparison, not a new nested validation.']
    (OUTPUT/'REPORT.md').write_text('\n'.join(lines)+'\n')
    print('\n'.join(lines),flush=True)

if __name__=='__main__':main()
