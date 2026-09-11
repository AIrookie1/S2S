"""Layer-wise comparison of raw-vs-mean and linear-vs-spherical steering.

Four conditions are evaluated on the same training-fold MC1 questions:
  1. linear + exact answer-pair vector
  2. linear + shared mean vector
  3. spherical + exact answer-pair vector
  4. spherical + shared mean vector

Unlike the older Fig1 code, every scored choice is intervened on.  For an exact
answer-pair vector v_ij, the truthful and false candidate in that pair are both
scored with the same intervention.  MC1 is one only when the truthful answer
wins every pairwise comparison.  For the shared mean, one common intervention
is used for the truthful answer and all false candidates.
"""
from __future__ import annotations
import argparse, glob, json, os, re, sys
from pathlib import Path
import numpy as np
import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

ROOT=Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path: sys.path.insert(0,str(ROOT))
from experiments.steering_vector_ablation.vector_utils import find_answer_rows, load_pair_vector_data
from method.dyn3d_sga.eval_utils import HF_NAMES, calculate_scores_zeroshot, get_additive_intervention, get_layer_name, load_truthfulqa_mc_dataframe, resolve_device, set_seed
from method.spherical_steering.steering import get_spherical_intervention

METHODS=("linear_pair_specific","linear_shared_mean","spherical_pair_specific","spherical_shared_mean")

def parse_args():
    p=argparse.ArgumentParser(description="Steering-vector ablation across layers")
    p.add_argument('model_name',nargs='?',default='llama3.1-8B-Instruct')
    p.add_argument('--feature_template',required=True,help='Path template with {layer} and optionally {fold}.')
    p.add_argument('--layers',nargs='+',type=int,default=None); p.add_argument('--folds',nargs='+',type=int,default=[0,1])
    p.add_argument('--model_dir',default=None); p.add_argument('--csv_path',default='./TruthfulQA/data/v1/TruthfulQA.csv')
    p.add_argument('--linear_alpha',type=float,default=1.0)
    p.add_argument('--spherical_kappa',type=float,default=20.0); p.add_argument('--spherical_alpha',type=float,required=True); p.add_argument('--spherical_beta',type=float,required=True)
    p.add_argument('--no_instruction',action='store_true'); p.add_argument('--device',choices=['cuda','cpu'],default='cuda'); p.add_argument('--dtype',choices=['float16','bfloat16','float32'],default='float16')
    p.add_argument('--seed',type=int,default=42); p.add_argument('--max_questions',type=int,default=None); p.add_argument('--output_dir',default='./results/experiment/steering_vector_ablation'); p.add_argument('--allow_missing',action='store_true')
    a=p.parse_args()
    if a.linear_alpha==0: p.error('--linear_alpha must be nonzero')
    if not a.folds or len(set(a.folds))!=len(a.folds): p.error('--folds must contain unique values')
    return a

def discover_layers(template):
    probe=template.replace('{layer}','*').replace('{fold}','*'); found=set()
    for path in glob.glob(probe):
        m=re.search(r'layer(\d+)',Path(path).name)
        if m: found.add(int(m.group(1)))
    if not found: raise FileNotFoundError(f'No feature files matched {probe}')
    return sorted(found)

def write_json(path,payload):
    path.parent.mkdir(parents=True,exist_ok=True); tmp=path.with_suffix(path.suffix+'.tmp'); tmp.write_text(json.dumps(payload,indent=2,ensure_ascii=False)+'\n',encoding='utf-8'); os.replace(tmp,path)

def normalize_vector(v,device):
    t=torch.as_tensor(v,dtype=torch.float32,device=device).flatten(); n=torch.linalg.norm(t)
    if n.item()==0: raise ValueError('Steering vector has zero norm')
    return t/n

def make_hook(kind,vector,args,device):
    if kind=='linear': return get_additive_intervention(torch.as_tensor(vector,dtype=torch.float32,device=device),args.linear_alpha)
    mu_t=normalize_vector(vector,device); mu_h=-mu_t
    return get_spherical_intervention(mu_t,mu_h,args.spherical_kappa,args.spherical_alpha,args.spherical_beta)

def score(model,tokenizer,question,answer,hook,layer_name,device,args):
    return calculate_scores_zeroshot(model,tokenizer,question,[answer],hook,layer_name,device,use_instruction=not args.no_instruction,model_name=args.model_name)[0]

def load_model(args,device):
    path=args.model_dir or HF_NAMES.get(args.model_name,args.model_name); tok=AutoTokenizer.from_pretrained(path,trust_remote_code=True)
    if tok.pad_token is None: tok.pad_token=tok.eos_token
    dtype=torch.float32 if device.type=='cpu' else {'float16':torch.float16,'bfloat16':torch.bfloat16,'float32':torch.float32}[args.dtype]
    model=AutoModelForCausalLM.from_pretrained(path,low_cpu_mem_usage=True,torch_dtype=dtype,device_map='auto' if device.type=='cuda' else None,trust_remote_code=True)
    if device.type=='cpu': model=model.to(device)
    model.eval(); model.generation_config.pad_token_id=tok.pad_token_id
    return model,tok,path

def evaluate_layer(layer,args,fold_data,hf_dataset,model,tok,device):
    totals={m:0 for m in METHODS}; count=0; lname=get_layer_name(args.model_name,layer)
    for data in fold_data:
        ids=data.train_questions.tolist(); ids=ids[:args.max_questions] if args.max_questions else ids
        mappings={}
        for qid in ids:
            targets=hf_dataset[int(qid)]['mc1_targets']; choices=list(targets['choices']); labels=list(targets['labels'])
            correct=[c for c,l in zip(choices,labels) if l==1]; false=[c for c,l in zip(choices,labels) if l==0]
            if len(correct)!=1: raise ValueError(f'Question {qid}: expected exactly one MC1 truthful answer')
            mappings[qid]=find_answer_rows(data,qid,correct[0],false)
        for qid in tqdm(ids,desc=f'layer {layer} fold {data.fold_idx}',leave=False):
            pos,false_rows=mappings[qid]; question=str(hf_dataset[qid]['question']); truth=data.choices[pos]
            mean=data.shared_mean
            # Shared-mean conditions: same intervention for every choice.
            for kind,name in [('linear','linear_shared_mean'),('spherical','spherical_shared_mean')]:
                hook=make_hook(kind,mean,args,device); truth_score=score(model,tok,question,truth,hook,lname,device,args)
                false_scores=[score(model,tok,question,data.choices[r],hook,lname,device,args) for r in false_rows]
                totals[name]+=int(truth_score>max(false_scores))
            # Exact answer-pair conditions: for each false candidate, both members of
            # that pair are scored with that pair's raw v_ij intervention.
            for kind,name in [('linear','linear_pair_specific'),('spherical','spherical_pair_specific')]:
                wins=[]
                for r in false_rows:
                    v=data.vectors[(pos,r)]; hook=make_hook(kind,v,args,device)
                    ts=score(model,tok,question,truth,hook,lname,device,args); fs=score(model,tok,question,data.choices[r],hook,lname,device,args)
                    wins.append(ts>fs)
                totals[name]+=int(all(wins))
            count+=1
    if count==0: raise ValueError(f'Layer {layer}: no questions')
    return {'layer':layer,'model_name':args.model_name,'folds':[int(x) for x in args.folds],'num_questions':count,'parameters':{'linear_alpha':args.linear_alpha,'spherical_kappa':args.spherical_kappa,'spherical_alpha':args.spherical_alpha,'spherical_beta':args.spherical_beta},'MC1':{m:float(totals[m]/count) for m in METHODS}}

def main():
    args=parse_args(); set_seed(args.seed); device=resolve_device(args.device); layers=sorted(set(args.layers)) if args.layers is not None else discover_layers(args.feature_template)
    _,hf=load_truthfulqa_mc_dataframe(args.csv_path); model,tok,model_path=load_model(args,device); out=Path(args.output_dir); out.mkdir(parents=True,exist_ok=True)
    print(f'Steering vector ablation | model={model_path} | layers={layers} | folds={args.folds}')
    for layer in layers:
        paths=[Path(args.feature_template.format(layer=layer,fold=f)) for f in args.folds]; missing=[str(p) for p in paths if not p.is_file()]
        if missing:
            if args.allow_missing: print(f'WARNING: layer {layer} missing {missing}; skipped'); continue
            raise FileNotFoundError(f'Layer {layer} missing feature files: {missing}')
        payload=evaluate_layer(layer,args,[load_pair_vector_data(str(p)) for p in paths],hf,model,tok,device)
        payload['model_path']=model_path; write_json(out/f'{args.model_name}_layer{layer}_mc1.json',payload); print(f'layer {layer}: {payload["MC1"]}')
if __name__=='__main__': main()
