"""Prepare matched Figure 2(b,c) inputs, evaluate, and plot; default prints a plan."""
import argparse,json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from reproducibility.run import Plan, execute


def main():
    p=argparse.ArgumentParser(__doc__)
    p.add_argument('--model',choices=['llama','qwen'],default='llama')
    p.add_argument('--model-dir',required=True)
    p.add_argument('--work-dir',type=Path,default=Path('results/reproduction'))
    p.add_argument('--gpu',default='0');p.add_argument('--execute',action='store_true');p.add_argument('--resume',action='store_true')
    a=p.parse_args();a.suite='figure2';a.folds=[0,1];a.device='cuda';a.dtype='float16';a.seed=42;a.epochs=2000
    plan=Plan(a);plan.prepare(l2s=True)
    out=a.work_dir/'paper/fig2bc'/plan.name
    template=str(plan.l2s(0)).replace('fold0','fold{fold}')
    plan.add('fig2bc-'+plan.name,plan.py('experiments/fig2bc_l2s_reliability/evaluate.py',plan.name,
        '--model_dir',a.model_dir,'--feature_path',plan.feature,'--checkpoint_template',template,
        '--layer',plan.layer,'--folds',0,1,'--output_dir',out,'--dtype',a.dtype),[out/'summary.json'])
    plan.add('fig2bc-plot-'+plan.name,plan.py('experiments/fig2bc_l2s_reliability/fig2bc_plot.py',
        '--input_dir',out,'--output_prefix',out/'fig2'),[out/'fig2_performance.png',out/'fig2_vector_reliability.png'])
    if a.execute:execute(plan.jobs,a)
    else:print(json.dumps(plan.jobs,indent=2))
if __name__=='__main__':main()
