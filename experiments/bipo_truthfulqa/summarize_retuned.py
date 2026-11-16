"""Export validation ablation grids and paired held-out comparisons."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

BASE=Path(__file__).resolve().parents[2]/'results/experiment/bipo_truthfulqa'
OUT=BASE/'mlp_s2s_retuned'

def main():
    rows=[]
    for fold in (0,1):
        for method in ('bipo_mlp_s2s_l','bipo_mlp_s2s_r'):
            selection=json.loads((OUT/f'fold{fold}_{method}_selection.json').read_text())
            assert len(selection['candidates'])==(20 if method.endswith('_l') else 9)
            for cell in selection['candidates']:
                rows.append(dict(fold=fold,method=method,selected=cell==selection['chosen'],**cell['parameters'],**{m:cell[m] for m in ('MC1','MC2','MC3')}))
    frame=pd.DataFrame(rows)
    frame.to_csv(OUT/'validation_grid.csv',index=False)
    fig,axes=plt.subplots(2,2,figsize=(9,6),layout='constrained')
    for fold in (0,1):
        for col,method in enumerate(('bipo_mlp_s2s_l','bipo_mlp_s2s_r')):
            sub=frame[(frame.fold==fold)&(frame.method==method)]
            x,y=('slerp_t','strength') if col==0 else ('margin','weight_kappa')
            table=sub.pivot(index=y,columns=x,values='MC1').sort_index()
            ax=axes[fold,col]
            mesh=ax.imshow(100*table.to_numpy(),vmin=0,vmax=80,cmap='viridis',aspect='auto')
            ax.set(xticks=range(len(table.columns)),xticklabels=table.columns,yticks=range(len(table.index)),yticklabels=table.index,xlabel=x,ylabel=y,title=f'Fold {fold}: '+('S2S-L' if col==0 else 'S2S-R'))
            for i in range(len(table)):
                for j in range(len(table.columns)):
                    ax.text(j,i,f'{100*table.iloc[i,j]:.1f}',ha='center',va='center',color='white',fontsize=8)
    fig.colorbar(mesh,ax=axes,label='Validation MC1 (%)',shrink=.8)
    fig.savefig(OUT/'validation_grid.png',dpi=220)
    fig.savefig(OUT/'validation_grid.pdf')
    plt.close(fig)
    baseline=[]
    for fold in (0,1):
        baseline.extend(json.loads((BASE/f'fold{fold}'/'test_bipo.json').read_text()))
    original={r['q_idx']:r for r in baseline}
    assert len(original)==817
    rng=np.random.default_rng(42)
    samples=rng.integers(0,817,size=(10000,817))
    paired=[]
    for method in ('bipo_mlp_s2s_l','bipo_mlp_s2s_r'):
        details=[r for fold in (0,1) for r in json.loads((OUT/f'fold{fold}'/f'test_retuned_{method}.json').read_text())]
        assert len(details)==817 and {r['q_idx'] for r in details}==set(original)
        for metric in ('MC1','MC2','MC3'):
            delta=np.array([r[metric]-original[r['q_idx']][metric] for r in details])
            interval=100*np.quantile(delta[samples].mean(1),[.025,.975])
            paired.append(dict(method=method,metric=metric,difference_pp=float(100*delta.mean()),ci95_pp=interval.tolist()))
    (OUT/'paired_differences.json').write_text(json.dumps(paired,indent=2)+'\n')
    report=OUT/'REPORT.md'
    text=report.read_text()
    text+='\n## Numerical correction\n\nThe evaluator now trims every answer to its real nonpadding length before asking the MLP for a prediction. Earlier MLP-S2S fixed and padded-sweep scores are superseded and must not be compared as valid baselines. Original BIPO uses a constant vector and is unaffected. Six regression tests pass, including matching single-sequence and padded-batch hook outputs.\n'
    text+='\n## Difference against original BIPO\n\n| Method | Metric | Difference (pp) | Paired 95% bootstrap CI (pp) |\n|---|---|---:|---|\n'
    for r in paired:
        lo,hi=r['ci95_pp'];text+=f"| {r['method']} | {r['metric']} | {r['difference_pp']:+.2f} | [{lo:+.2f}, {hi:+.2f}] |\n"
    text+='\nIntervals condition on these two fitted checkpoints; training-seed variation is not measured. The grid is finite and may not contain the globally best parameters. S2S-L is the existing static interpolation variant. All methods intervene on all tokens.\n'
    report.write_text(text)
    print(text)

if __name__=='__main__':main()
