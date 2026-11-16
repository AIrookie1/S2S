"""Inference residual-direction noise with clean frozen predictors; not target noise."""
import argparse,json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from reproducibility.run import Plan,execute


def main():
    p=argparse.ArgumentParser(__doc__)
    p.add_argument('--model',choices=['llama','qwen'],default='llama');p.add_argument('--model-dir',required=True)
    p.add_argument('--folds',type=int,nargs='+',choices=[0,1],default=[0])
    p.add_argument('--sigmas',type=float,nargs='+',default=[0,.2,.4,.8,1,2,3,4,5])
    p.add_argument('--gpu',default='0');p.add_argument('--seed',type=int,default=42)
    p.add_argument('--work-dir',type=Path,default=Path('results/reproduction'))
    p.add_argument('--execute',action='store_true');p.add_argument('--resume',action='store_true')
    a=p.parse_args()
    if len(set(a.folds))!=len(a.folds) or len(set(a.sigmas))!=len(a.sigmas) or any(x<0 or not __import__('math').isfinite(x) for x in a.sigmas):
        p.error('folds/sigmas must be unique and sigmas finite/nonnegative')
    a.suite='inference_noise';a.device='cuda';a.dtype='float16';a.epochs=2000
    plan=Plan(a);plan.prepare()
    for fold in a.folds:
        for sigma in a.sigmas:
            for method in ('s2s_l','s2s_r'):
                out=a.work_dir/'rebuttal/inference_noise'/plan.name/f'fold{fold}'/f'sigma{sigma:g}'/(method+'.json')
                options=['--checkpoint_path',plan.s2s(fold),'--eval_split','test','--residual_noise_sigma',sigma,
                    '--residual_noise_seed',a.seed,'--dtype',a.dtype,'--output_path',out,'--save_details']
                if method=='s2s_l':
                    cmd=plan.entry('S2S_L','evaluate_S2S_L.py',*plan.common(fold),*options,
                        '--method','S2S_L_Static','--strength',3,'--slerp_t',.2)
                else:
                    cmd=plan.entry('S2S_R','evaluate_S2S_R.py',*plan.common(fold),*options,
                        '--weight_kappa',3,'--margin',(-.3 if a.model=='qwen' and fold==0 else -.7))
                plan.add(f'inference-noise-{plan.name}-{fold}-{sigma:g}-{method}',cmd,[out])
    if a.execute:execute(plan.jobs,a)
    else:print(json.dumps(dict(noise_location='inference_residual_direction',training_target_noise=0,jobs=plan.jobs),indent=2))
if __name__=='__main__':main()
