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
    p=argparse.ArgumentParser(); p.add_argument('--val-fraction',type=float,required=True); p.add_argument('--seed',type=int,required=True); p.add_argument('--model',required=True); p.add_argument('--model-label',required=True); p.add_argument('--layer',type=int,required=True); p.add_argument('--fold',type=int,default=0); p.add_argument('--num-folds',type=int,default=2); p.add_argument('--eval-json',type=Path,required=True); p.add_argument('--checkpoint',type=Path,required=True); p.add_argument('--output',type=Path,required=True); a=p.parse_args()
    if not 0<a.val_fraction<1: raise ValueError('val_fraction must be in (0,1)')
    mc1,ev=load_mc1(a.eval_json)
    with np.load(a.checkpoint,allow_pickle=False) as ck:
        fold=int(checkpoint_scalar(ck,'fold_idx')); hidden_dim=int(checkpoint_scalar(ck,'mlp_hidden', checkpoint_scalar(ck,'hidden_dim',0))); actual=float(checkpoint_scalar(ck,'val_fraction',a.val_fraction))
    if fold != a.fold or not np.isclose(actual,a.val_fraction): raise ValueError('checkpoint metadata does not match requested fold/val_fraction')
    params=ev.get('parameters',{}); eval_kappa=float(params.get('weight_kappa',float('nan'))); eval_margin=float(params.get('margin',float('nan')))
    meta={'experiment':'val_fraction_ablation_mc1','method':'Similarity-Graph Adaptive 3D Spherical Steering','model':a.model,'model_label':a.model_label,'layer':a.layer,'fold_idx':a.fold,'num_folds':a.num_folds,'x_axis':'val_fraction','y_axis':'MC1','aggregation':'mean +/- sample SD across seeds'}
    rec={'val_fraction':a.val_fraction,'seed':a.seed,'hidden_dim':hidden_dim,'eval_weight_kappa':eval_kappa,'eval_margin':eval_margin,'MC1':mc1,'checkpoint':str(a.checkpoint),'evaluation':str(a.eval_json)}
    update_summary(a.output,meta,rec,('val_fraction','seed'),('val_fraction','seed')); print(f'val_fraction={a.val_fraction:g} seed={a.seed} MC1={mc1:.6f}')
if __name__=='__main__': main()
