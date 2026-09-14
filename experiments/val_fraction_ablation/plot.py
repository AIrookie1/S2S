#!/usr/bin/env python3
from __future__ import annotations
import argparse,csv,json,sys
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
HERE=Path(__file__).resolve().parent; sys.path.insert(0,str(HERE.parent))
from plot_common import finish,save,setup_plot
p=argparse.ArgumentParser(); p.add_argument('--input-dir',type=Path,default=HERE.parents[1]/'results'/'experiment'/'val_fraction_ablation'/'summaries'); p.add_argument('--output-prefix',type=Path,default=HERE.parents[1]/'results'/'experiment'/'val_fraction_ablation'/'val_fraction_ablation_mc1'); p.add_argument('--seeds',default='42 123 0'); a=p.parse_args(); seeds=[int(x) for x in a.seeds.split()]
setup_plot(); series=[]
for path in sorted(a.input_dir.glob('*.json')):
    d=json.loads(path.read_text()); grouped={}
    for r in d['results']:
        if int(r['seed']) in seeds: grouped[(float(r['val_fraction']),int(r['seed']))]=float(r['MC1'])
    xs=np.asarray(sorted({x for x,_ in grouped})); vals=np.asarray([[grouped[(float(x),s)] for x in xs] for s in seeds]); series.append((d['model'],d.get('model_label',d['model']),xs,vals.mean(0),vals.std(0,ddof=1)))
if not series: raise ValueError('No summaries found')
fig,ax=plt.subplots(figsize=(5.0,4.0))
for model,label,x,mean,sd in series: ax.errorbar(x,mean,yerr=sd,marker='o',capsize=2,label=label)
finish(ax,'Validation fraction'); fig.tight_layout(); save(fig,a.output_prefix)
csv_path=a.output_prefix.with_name(a.output_prefix.name+'_data.csv')
with csv_path.open('w',newline='',encoding='utf-8') as f:
    w=csv.writer(f); w.writerow(['model','val_fraction','mean_MC1','sd_MC1']);
    for model,label,x,mean,sd in series:
        for i,v in enumerate(x): w.writerow([model,float(v),float(mean[i]),float(sd[i])])
plt.close(fig); print(f'Saved {a.output_prefix} and {csv_path}')
