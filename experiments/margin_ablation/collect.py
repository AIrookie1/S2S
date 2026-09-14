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
    p=argparse.ArgumentParser(); p.add_argument('--margin',type=float,required=True); p.add_argument('--model',required=True); p.add_argument('--model-label',required=True); p.add_argument('--layer',type=int,required=True); p.add_argument('--fold',type=int,required=True); p.add_argument('--num-folds',type=int,default=2); p.add_argument('--eval-json',type=Path,required=True); p.add_argument('--checkpoint',type=Path,required=True); p.add_argument('--output',type=Path,required=True); a=p.parse_args()
    mc1,ev=load_mc1(a.eval_json); actual=float(ev.get('parameters',{}).get('margin',float('nan')))
    if not np.isclose(actual,a.margin): raise ValueError(f'eval margin={actual}, requested={a.margin}')
    with np.load(a.checkpoint,allow_pickle=False) as ck: fold=int(checkpoint_scalar(ck,'fold_idx')); rank=int(checkpoint_scalar(ck,'rank'))
    if fold != a.fold: raise ValueError('checkpoint fold mismatch')
    meta={'experiment':'margin_ablation_mc1','method':'Similarity-Graph Adaptive 3D Spherical Steering','model':a.model,'model_label':a.model_label,'layer':a.layer,'fold_idx':a.fold,'num_folds':a.num_folds,'x_axis':'margin','y_axis':'MC1'}
    rec={'margin':a.margin,'rank':rank,'MC1':mc1,'checkpoint':str(a.checkpoint),'evaluation':str(a.eval_json)}
    update_summary(a.output,meta,rec,('margin',),('margin',)); print(f'margin={a.margin:g} MC1={mc1:.6f}')
if __name__=='__main__': main()
