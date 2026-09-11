#!/usr/bin/env python3
from __future__ import annotations
import argparse, sys
from pathlib import Path
import numpy as np
HERE = Path(__file__).resolve().parent
EXPERIMENTS_DIR = HERE.parent
if str(EXPERIMENTS_DIR) not in sys.path: sys.path.insert(0, str(EXPERIMENTS_DIR))
from collect_common import checkpoint_scalar, load_mc1, update_summary

def main():
    p=argparse.ArgumentParser(); p.add_argument('--router-top-k',type=int,required=True); p.add_argument('--model',required=True); p.add_argument('--model-label',required=True); p.add_argument('--layer',type=int,required=True); p.add_argument('--fold',type=int,required=True); p.add_argument('--num-folds',type=int,default=2); p.add_argument('--eval-json',type=Path,required=True); p.add_argument('--checkpoint',type=Path,required=True); p.add_argument('--output',type=Path,required=True); a=p.parse_args()
    mc1,ev=load_mc1(a.eval_json)
    with np.load(a.checkpoint,allow_pickle=False) as ck:
        fold=int(checkpoint_scalar(ck,'fold_idx')); rank=int(checkpoint_scalar(ck,'rank')); effective=int(checkpoint_scalar(ck,'router_top_k')); requested=int(checkpoint_scalar(ck,'router_top_k_requested',effective))
    if fold != a.fold: raise ValueError(f'checkpoint fold={fold}, requested={a.fold}')
    if requested != a.router_top_k: raise ValueError(f'checkpoint requested top_k={requested}, CLI requested={a.router_top_k}')
    params=ev.get('parameters',{}); eval_kappa=float(params.get('weight_kappa',float('nan'))); eval_margin=float(params.get('margin',float('nan')))
    meta={'experiment':'router_topk_ablation_mc1','method':'Similarity-Graph Adaptive 3D Spherical Steering','model':a.model,'model_label':a.model_label,'layer':a.layer,'fold_idx':a.fold,'num_folds':a.num_folds,'x_axis':'router_top_k_requested','y_axis':'MC1'}
    rec={'router_top_k_requested':a.router_top_k,'effective_router_top_k':effective,'rank':rank,'was_clamped':effective != a.router_top_k,'eval_weight_kappa':eval_kappa,'eval_margin':eval_margin,'MC1':mc1,'checkpoint':str(a.checkpoint),'evaluation':str(a.eval_json)}
    update_summary(a.output,meta,rec,('router_top_k_requested',),('router_top_k_requested',)); print(f'top_k={a.router_top_k} effective={effective} rank={rank} MC1={mc1:.6f}')
if __name__=='__main__': main()
