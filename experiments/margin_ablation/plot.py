#!/usr/bin/env python3
from __future__ import annotations
import argparse, sys
from pathlib import Path
import matplotlib.pyplot as plt
HERE=Path(__file__).resolve().parent; sys.path.insert(0,str(HERE.parent))
from plot_large import finish, load_fold_series, plot_mean_sd, save, setup_plot, write_csv
p=argparse.ArgumentParser(); p.add_argument('--input-dir',type=Path,default=HERE.parents[1]/'results'/'experiment'/'margin_ablation'/'summaries'); p.add_argument('--output-prefix',type=Path,default=HERE.parents[1]/'results'/'experiment'/'margin_ablation'/'margin_ablation_mc1'); p.add_argument('--num-folds',type=int,default=2); a=p.parse_args()
setup_plot(); series=load_fold_series(a.input_dir,'margin',a.num_folds); fig,ax=plt.subplots(figsize=(10.0,4.0)); plot_mean_sd(ax,series)

finish(ax,'Margin threshold'); fig.tight_layout(); save(fig,a.output_prefix); csv=write_csv(a.output_prefix,'margin',series); plt.close(fig); print(f'Saved {a.output_prefix} and {csv}')
