#!/usr/bin/env python3
from __future__ import annotations
import argparse, sys
from pathlib import Path
import matplotlib.pyplot as plt
HERE=Path(__file__).resolve().parent; sys.path.insert(0,str(HERE.parent))
from plot_common import finish, load_fold_series, plot_mean_sd, save, setup_plot, write_csv
p=argparse.ArgumentParser(); p.add_argument('--input-dir',type=Path,default=HERE.parents[1]/'results'/'experiment'/'load_balance_ablation'/'summaries'); p.add_argument('--output-prefix',type=Path,default=HERE.parents[1]/'results'/'experiment'/'load_balance_ablation'/'load_balance_ablation_mc1'); p.add_argument('--num-folds',type=int,default=2); a=p.parse_args()
setup_plot(); series=load_fold_series(a.input_dir,'load_balance_loss_weight',a.num_folds); fig,ax=plt.subplots(figsize=(5.0,4.0)); plot_mean_sd(ax,series)

finish(ax,'Load-balancing loss weight'); fig.tight_layout(); save(fig,a.output_prefix); csv=write_csv(a.output_prefix,'load_balance_loss_weight',series); plt.close(fig); print(f'Saved {a.output_prefix} and {csv}')
