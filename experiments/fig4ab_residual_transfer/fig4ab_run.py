"""Run both residual-bearing geometries with the prepared main checkpoints."""
import argparse,json,os,subprocess,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from reproducibility.run import Plan


def main():
    p=argparse.ArgumentParser(__doc__);p.add_argument('--model',choices=['llama','qwen'],default='llama')
    p.add_argument('--model-dir',required=True);p.add_argument('--work-dir',type=Path,default=Path('results/reproduction'))
    p.add_argument('--folds',nargs='+',type=int,choices=[0,1],default=[0,1]);p.add_argument('--gpu',default='0')
    p.add_argument('--execute',action='store_true');a=p.parse_args();plan=Plan(a)
    jobs=[]
    for fold in a.folds:
        for method in ('S2S_R','S2S_L_Static'):
            env=dict(FAMILY=a.model.upper(),METHOD=method,FOLD_IDX=str(fold),GPU=a.gpu,PYTHON_BIN=sys.executable,
                LLAMA_MODEL_DIR=a.model_dir,QWEN_MODEL_DIR=a.model_dir,CHECKPOINT_PATH=str(plan.s2s(fold)),
                SPHERICAL_PROTO_DIR=str(plan.proto_dir),SPHERICAL_LAYER=str(plan.layer),
                WEIGHT_KAPPA='3',MARGIN=str(-.3 if a.model=='qwen' and fold==0 else -.7),
                OUTPUT_DIR=str(a.work_dir/'paper/fig4ab'/plan.name/method/f'fold{fold}'))
            cmd=['bash','experiments/fig4ab_residual_transfer/fig4ab_run.sh'];jobs.append(dict(command=cmd,env=env))
            if a.execute:
                if not plan.s2s(fold).is_file():raise FileNotFoundError('Run Table 1 MC preparation first')
                subprocess.run(cmd,cwd=ROOT,env={**os.environ,**env,'PYTHONPATH':str(ROOT)},check=True)
    if not a.execute:print(json.dumps(jobs,indent=2))
if __name__=='__main__':main()
