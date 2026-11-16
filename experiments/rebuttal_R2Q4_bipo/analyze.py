"""Aggregate paired question diagnostics and export standalone reliability figures."""
import argparse,json
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--input',type=Path,default=Path('results/experiment/rebuttal_R2Q4_bipo'))
    args=p.parse_args()
    tables=[]
    for fold in [0,1]:
        tables.append(pd.read_csv(args.input/f'fold{fold}'/'diagnostics_question.csv'))
    frame=pd.concat(tables,ignore_index=True)
    frame.to_csv(args.input/'diagnostics_question.csv',index=False)
    columns=['cosine_similarity','angle_degrees','magnitude_relative_error','predicted_norm','target_norm']
    summary=frame.groupby(['source','split'])[columns].mean()
    summary.to_csv(args.input/'diagnostics_summary.csv')
    # Training vs testing aggregate discrepancy, not a claim of overfitting.
    bipo=frame[frame.source=='BIPO']
    gaps=[]
    for fold in [0,1]:
        per=bipo[bipo.fold==fold].groupby('split')[columns].mean()
        gaps.append(dict(fold=fold,**{key:float(per.loc['test',key]-per.loc['train',key]) for key in columns}))
    (args.input/'train_test_gaps.json').write_text(json.dumps(gaps,indent=2)+'\n')
    paired=[]
    method_rows={}
    for method in ['bipo','s2s_l','s2s_r']:
        rows=[r for fold in [0,1] for r in json.loads((args.input/f'fold{fold}'/f'test_{method}.json').read_text())]
        method_rows[method]={r['q_idx']:r for r in rows}
    ids=sorted(method_rows['bipo'])
    if any(set(rows)!=set(ids) for rows in method_rows.values()):
        raise ValueError('Methods must cover the identical held-out questions')
    rng=np.random.default_rng(42)
    # Question-paired bootstrap; conditional on the trained fold checkpoints.
    samples=rng.integers(0,len(ids),size=(10000,len(ids)))
    for method in ['s2s_l','s2s_r']:
        for metric in ['MC1','MC2','MC3']:
            delta=np.asarray([method_rows[method][q][metric]-method_rows['bipo'][q][metric] for q in ids])
            ci=np.quantile(delta[samples].mean(1),[.025,.975])
            paired.append(dict(method=method,metric=metric,delta_percentage_points=float(100*delta.mean()),ci95_percentage_points=(100*ci).tolist()))
    (args.input/'paired_differences.json').write_text(json.dumps(paired,indent=2)+'\n')
    report=args.input/'REPORT.md'
    original=report.read_text().split('\n## Protocol and selection')[0]
    extra=['','## Protocol and selection','','Llama-3.1-8B-Instruct, block 14 (zero indexed), FP16; two-fold held-out evaluation. ',
           'BIPO trained up to 20 epochs. Checkpoint selected by bidirectional development loss; ',
           'method parameters selected by development MC1, then MC2. All methods intervene at all tokens.',
           'S2S-L uses the repository static interpolation variant t=0.2; S2S-R uses adaptive weights.',
           'The mean anchor uses only BIPO optimization questions. Raw diagnostics use the existing plain Q/A feature artifact.',
           '', '| Fold | Best epoch | Train | Dev | Test | BIPO strength | S2S-L strength | S2S-R margin |',
           '|---|---:|---:|---:|---:|---:|---:|---:|']
    for fold in [0,1]:
        folder=args.input/f'fold{fold}'
        split=json.loads((folder/'split.json').read_text())
        history=json.loads((folder/'history.json').read_text())
        best=min(history,key=lambda r:r['dev_loss'])
        selections={m:json.loads((folder/f'selection_{m}.json').read_text())['chosen']['parameters'] for m in ['bipo','s2s_l','s2s_r']}
        extra.append(f"| {fold} | {best['epoch']} | {len(split['train'])} | {len(split['dev'])} | {len(split['test'])} | {selections['bipo']['strength']} | {selections['s2s_l']['strength']} | {selections['s2s_r']['margin']} |")
    extra+=['','## Paired differences against BIPO','','| Method | Metric | Difference (pp) | 95% paired bootstrap CI (pp) |','|---|---|---:|---|']
    for r in paired:
        lo,hi=r['ci95_percentage_points']
        extra.append(f"| {r['method']} | {r['metric']} | {r['delta_percentage_points']:+.2f} | [{lo:+.2f}, {hi:+.2f}] |")
    extra+=['','Bootstrap intervals condition on the learned checkpoints and do not measure training-seed uncertainty.',
            'Geometric discrepancy from raw answer differences does not by itself prove a causal link to MC performance.',
            'A shared BIPO vector has the same norm on train and test; differences below concern its alignment/error against each split’s targets.',
            '','## Per-fold test minus train reliability','','| Fold | Cosine gap | Angle gap (degrees) | Relative norm error gap | Target norm gap |','|---|---:|---:|---:|---:|']
    for r in gaps:
        extra.append(f"| {r['fold']} | {r['cosine_similarity']:+.4f} | {r['angle_degrees']:+.4f} | {r['magnitude_relative_error']:+.4f} | {r['target_norm']:+.4f} |")
    report.write_text(original+'\n'.join(extra)+'\n')
    plt.rcParams.update({'font.size':9,'axes.spines.top':False,'axes.spines.right':False,'pdf.fonttype':42,'svg.fonttype':'none'})
    fig,axes=plt.subplots(1,2,figsize=(7.2,2.9),layout='constrained')
    colors={'BIPO':'#db6b32','Mean':'#286fa3'}
    markers={'train':'o','test':'^'}
    for source in ['BIPO','Mean']:
        for split in ['train','test']:
            rows=frame[(frame.source==source)&(frame.split==split)]
            axes[0].scatter(rows.cosine_similarity,rows.magnitude_relative_error,c=colors[source],marker=markers[split],alpha=.2,s=10,label=f'{source} {split}')
            axes[0].scatter(rows.cosine_similarity.mean(),rows.magnitude_relative_error.mean(),c=colors[source],marker=markers[split],s=65,edgecolor='black',linewidth=.7)
    axes[0].set(xlabel='Cosine similarity to raw answer difference',ylabel='Relative norm error')
    axes[0].legend(fontsize=7,frameon=False)
    values=json.loads((args.input/'performance.json').read_text())
    x=np.arange(3)
    for i,metric in enumerate(['MC1','MC2','MC3']):
        axes[1].bar(x+(i-1)*.24,[100*r[metric] for r in values],width=.24,label=metric)
    axes[1].set(xticks=x,xticklabels=['BIPO','+S2S-L','+S2S-R'],ylabel='TruthfulQA score (%)')
    axes[1].legend(frameon=False,fontsize=7)
    for suffix in ['png','pdf','svg']:fig.savefig(args.input/f'bipo_reliability.{suffix}',dpi=300)
    plt.close(fig)
    print(summary)
    print('Test minus train:',gaps)

if __name__=='__main__':main()
