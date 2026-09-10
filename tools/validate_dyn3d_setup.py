#!/usr/bin/env python3
from __future__ import annotations
import os, py_compile, re, subprocess
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
EXPECTED_EXPERIMENTS={'load_balance_ablation','router_topk_ablation','val_fraction_ablation','weight_kappa_ablation','margin_ablation'}
EXPECTED_GENERIC={'boolq','copa','storycloze','mmlu_global','winogrande'}
def run(cmd, **kwargs): return subprocess.run(cmd,cwd=ROOT,check=True,stdout=subprocess.DEVNULL,stderr=subprocess.PIPE,text=True,**kwargs)
def main():
    errors=[]
    for p in ROOT.rglob('*.sh'):
        try: run(['bash','-n',str(p)])
        except subprocess.CalledProcessError as e: errors.append(f'bash -n {p.relative_to(ROOT)}: {e.stderr.strip()}')
    for p in ROOT.rglob('*.py'):
        if '__pycache__' in p.parts: continue
        try: py_compile.compile(str(p),doraise=True)
        except Exception as e: errors.append(f'py_compile {p.relative_to(ROOT)}: {e}')
    for old in ('fig3a','fig4a','fig5','fig6'):
        if (ROOT/'experiments'/old).exists(): errors.append(f'legacy experiment directory still exists: {old}')
    for name in EXPECTED_EXPERIMENTS:
        for f in ('run.sh','collect.py','plot.py','plot.sh'):
            if not (ROOT/'experiments'/name/f).is_file(): errors.append(f'missing experiments/{name}/{f}')
        if not (ROOT/'hyperparams/experiments'/name/'config.sh').is_file(): errors.append(f'missing hyperparams config for {name}')
    for ds in EXPECTED_GENERIC:
        if not (ROOT/'hyperparams/generic'/ds/'config.sh').is_file(): errors.append(f'missing generic config for {ds}')
        if not (ROOT/'generic/script'/f'{ds}_generic.sh').is_file(): errors.append(f'missing generic runner for {ds}')
    scan='\n'.join(p.read_text(errors='ignore') for base in (ROOT/'experiments',ROOT/'generic'/'script') for p in base.rglob('*') if p.is_file() and p.suffix in {'.sh','.py'})
    if re.search(r'(^|\s)--rank(\s|$)|\bSVD_RANK\b',scan): errors.append('manual SVD rank remains in generic/experiments')
    base=(ROOT/'hyperparams'/'base.sh').read_text()
    for needle in ('HP_SVD_ENERGY=0.99','HP_ROUTER_TOP_K=40','HP_LOAD_BALANCE_LOSS_WEIGHT=0.02','HP_WEIGHT_KAPPA=8','HP_MARGIN=-0.3'):
        if needle not in base: errors.append(f'baseline mismatch: {needle}')
    dry_env=os.environ.copy(); dry_env.update({'DRY_RUN':'1','AUTO_PLOT':'0','MODELS':'llama','FOLDS':'0','SEEDS':'42 123'})
    for name in EXPECTED_EXPERIMENTS:
        try: run(['bash',f'experiments/{name}/run.sh'],env=dry_env)
        except subprocess.CalledProcessError as e: errors.append(f'dry-run {name}: {e.stderr.strip()}')
    generic_env=dry_env.copy(); generic_env['RUN_EXTRACT']='1'
    for ds in EXPECTED_GENERIC:
        try: run(['bash',f'generic/script/{ds}_generic.sh'],env=generic_env)
        except subprocess.CalledProcessError as e: errors.append(f'dry-run generic {ds}: {e.stderr.strip()}')
    if errors:
        print('Validation failed:'); print('\n'.join(f'- {e}' for e in errors)); return 1
    print('Validation OK: shell syntax, Python syntax, config layout, energy-only policy, and dry-run command construction all pass.'); return 0
if __name__=='__main__': raise SystemExit(main())
