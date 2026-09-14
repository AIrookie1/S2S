from __future__ import annotations
import argparse, json, re
from pathlib import Path
import matplotlib as mpl
import matplotlib.pyplot as plt

def main():
    p=argparse.ArgumentParser(); p.add_argument('--input_dir',type=Path,default=Path('./results/experiment/steering_vector_ablation')); p.add_argument('--model_name',default='llama3.1-8B-Instruct'); p.add_argument('--output_prefix',type=Path,default=Path('./results/experiment/steering_vector_ablation/steering_vector_ablation_mc1')); a=p.parse_args()
    rows=[]
    for f in sorted(a.input_dir.glob(f'{a.model_name}_layer*_mc1.json')):
        d=json.loads(f.read_text(encoding='utf-8')); rows.append((int(d['layer']),d['MC1']))
    if not rows: raise FileNotFoundError(f'No per-layer JSON files for {a.model_name} in {a.input_dir}')
    rows.sort(); layers=[x[0] for x in rows]
    labels={'linear_pair_specific':'Linear + raw pair vector','linear_shared_mean':'Linear + shared mean','spherical_pair_specific':'Spherical + raw pair vector','spherical_shared_mean':'Spherical + shared mean'}
    mpl.rcParams.update({'font.family':'sans-serif','pdf.fonttype':42,'svg.fonttype':'none'})
    fig,ax=plt.subplots(figsize=(6.4,4.2))
    for key,label in labels.items(): ax.plot(layers,[x[1][key] for x in rows],marker='o',markersize=3,label=label)
    ax.set_xlabel('Layer'); ax.set_ylabel('MC1'); ax.set_ylim(0.3,0.7); ax.legend(frameon=False,ncol=2); fig.tight_layout(); a.output_prefix.parent.mkdir(parents=True,exist_ok=True)
    for ext in ['png','pdf','svg']: fig.savefig(a.output_prefix.with_suffix('.'+ext),dpi=300 if ext=='png' else None)
    plt.close(fig); print(f'Saved {a.output_prefix}.[png|pdf|svg]')
if __name__=='__main__': main()
