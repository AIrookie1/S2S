"""Run the retained label-informed train-fold oracle reference, not a deployable baseline."""
import argparse,json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from reproducibility.run import Plan, execute


def main():
    p=argparse.ArgumentParser(__doc__);p.add_argument('--model',choices=['llama','qwen'],default='llama')
    p.add_argument('--model-dir',required=True);p.add_argument('--work-dir',type=Path,default=Path('results/reproduction'))
    p.add_argument('--gpu',default='0');p.add_argument('--execute',action='store_true');p.add_argument('--resume',action='store_true')
    a=p.parse_args();a.suite='figure2a';a.folds=[0,1];a.device='cuda';a.dtype='float16';a.seed=42;a.epochs=2000
    plan=Plan(a);plan.prepare(train=False)
    for fold in a.folds:
        out=a.work_dir/'paper/fig2a'/plan.name/f'fold{fold}.json'
        plan.add(f'fig2a-{plan.name}-{fold}',plan.py('experiments/fig2a_oracle_vectors/evaluate_reference.py',plan.name,
            '--model_dir',a.model_dir,'--feature_path',plan.feature,'--layer',plan.layer,
            '--fold_idx',fold,'--output_path',out,'--save_details'),[out])
    if a.execute:execute(plan.jobs,a)
    else:print(json.dumps(plan.jobs,indent=2))
if __name__=='__main__':main()
