"""Table 1 generation commands using fold-matched native evaluators and judges."""
import argparse,json,os,subprocess,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from reproducibility.run import Plan


def build_jobs(a):
    plan=Plan(a);jobs=[]
    for fold in a.folds:
        for method in a.methods:
            out=a.work_dir/'table1_generation'/plan.name/f'fold{fold}'/method
            common=[plan.name,'--model_dir',a.model_dir,'--layer',str(plan.layer),
                '--truth_judge_path',a.truth_judge_dir,'--info_judge_path',a.info_judge_dir,'--max_new_tokens','50',
                '--preset','null']
            if method in ('baseline','s2s_l','s2s_r'):
                file='method/S2S_L/evaluate_llm_judge_S2S_L.py' if method=='s2s_l' else 'method/S2S_R/evaluate_llm_judge_S2S_R.py'
                args=[*common,'--checkpoint_path',str(plan.s2s(fold)),'--output_dir',str(out)]
                if method=='s2s_l':args+=['--strength','3','--slerp_t','.2']
                else:args+=['--weight_kappa','3','--margin',str(-.3 if a.model=='qwen' and fold==0 else -.7)]
                if method=='baseline':args+=['--disable_steering']
            elif method=='l2s':
                file='method/l2s_linear/evaluate_llm_judge.py'
                args=[*common,'--checkpoint_path',str(plan.l2s(fold)),'--alpha',str(1.4 if a.model=='qwen' else 1.),
                      '--output_dir',str(out)]
            else:
                file='method/spherical_steering/evaluate_llm_judge.py'
                alpha,beta=(.7,-.15) if a.model=='llama' else (.6,.4)
                args=[*common,'--prototype_path',str(plan.prototype(fold)),'--kappa','20',
                      '--alpha',str(alpha),'--beta',str(beta),'--output_dir',str(out)]
            jobs.append(dict(method=method,fold=fold,command=[sys.executable,file,*args]))
    return jobs


def main():
    p=argparse.ArgumentParser(__doc__);p.add_argument('--model',choices=['llama','qwen'],default='llama')
    p.add_argument('--model-dir',required=True);p.add_argument('--truth-judge-dir',required=True);p.add_argument('--info-judge-dir',required=True)
    p.add_argument('--work-dir',type=Path,default=Path('results/reproduction'));p.add_argument('--folds',type=int,nargs='+',choices=[0,1],default=[0,1])
    p.add_argument('--methods',nargs='+',choices=['baseline','l2s','spherical','s2s_l','s2s_r'],default=['baseline','l2s','spherical','s2s_l','s2s_r'])
    p.add_argument('--gpu',default='0');p.add_argument('--execute',action='store_true');a=p.parse_args()
    jobs=build_jobs(a)
    if not a.execute:print(json.dumps(jobs,indent=2));return
    if any(not Path(x).is_dir() for x in (a.truth_judge_dir,a.info_judge_dir)):raise FileNotFoundError('Both local TruthfulQA judge models are required')
    for job in jobs:
        subprocess.run(job['command'],cwd=ROOT,env={**os.environ,'CUDA_VISIBLE_DEVICES':a.gpu,'PYTHONPATH':str(ROOT)},check=True)
if __name__=='__main__':main()
