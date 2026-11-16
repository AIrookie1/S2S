"""Checks for paper/rebuttal entry-point wiring, not final model accuracy."""
import argparse
from contextlib import redirect_stdout
import importlib.util
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

from reproducibility.run import build_plan
from tests.reproducibility.test_workflow import args
ROOT=Path(__file__).resolve().parents[2]


class PaperEntryTests(unittest.TestCase):
    def test_qwen_main_parameters_match_archived_fold_settings(self):
        jobs=build_plan(args('main',model='qwen'))
        for job in jobs:
            cmd=job['command']
            if job['name'].startswith('main-') and job['name'].endswith('-l2s'):
                self.assertEqual(cmd[cmd.index('--alpha')+1],'1.4')
            if job['name'].startswith('main-') and job['name'].endswith('-s2s_r'):
                expected='-0.3' if '-0-s2s_r' in job['name'] else '-0.7'
                self.assertEqual(cmd[cmd.index('--margin')+1],expected)

    def test_qwen_copa_uses_the_native_400_epoch_setting(self):
        jobs=build_plan(args('generic',model='qwen',datasets=['copa']))
        learned=[j for j in jobs if j['command'][0]=='bash' and 'spherical_steering' not in j['name']]
        self.assertEqual(len(learned),3)
        self.assertTrue(all(j['env']['EPOCHS']=='400' for j in learned))

    def test_generation_includes_fold_matched_spherical_prototypes(self):
        jobs=[j for j in build_plan(args('generation')) if j['name'].startswith('generation-')]
        self.assertEqual(len(jobs),2)
        for job in jobs:
            cmd=job['command'];self.assertIn('--spherical-prototype',cmd)
            fold=0 if job['name'].endswith('-0') else 1
            self.assertTrue(cmd[cmd.index('--spherical-prototype')+1].endswith(f'_fold{1-fold}.npz'))

    def test_inference_noise_changes_only_evaluation_noise(self):
        from experiments.rebuttal_R1W1_inference_noise.rebuttal_R1W1_run import main
        stream=io.StringIO()
        with patch.object(sys,'argv',['run.py','--model-dir','models/Llama','--sigmas','0','0.4']),redirect_stdout(stream):main()
        plan=json.loads(stream.getvalue())
        self.assertEqual(plan['training_target_noise'],0)
        evaluations=[j for j in plan['jobs'] if j['name'].startswith('inference-noise-')]
        self.assertEqual(len(evaluations),4)
        for job in evaluations:
            self.assertIn('--residual_noise_sigma',job['command'])
            self.assertNotIn('--target_noise_sigma',job['command'])

    def test_generation_parameters_match_mc_parameters(self):
        from experiments.table1_truthfulqa.table1_run_generation import build_jobs
        a=args(model='qwen');a.truth_judge_dir='models/truth';a.info_judge_dir='models/info'
        a.methods=['baseline','l2s','spherical','s2s_l','s2s_r']
        for job in build_jobs(a):
            cmd=job['command']
            if job['method']=='l2s':self.assertEqual(cmd[cmd.index('--alpha')+1],'1.4')
            if job['method']=='s2s_r':self.assertEqual(cmd[cmd.index('--margin')+1],str(-.3 if job['fold']==0 else -.7))

    def test_indexed_experiments_and_cleanup_are_consistent(self):
        import csv
        with (ROOT/'docs/EXPERIMENT_INDEX.csv').open() as f:rows=list(csv.DictReader(f))
        for row in rows:
            if row['entrypoint']=='not_bundled':continue
            self.assertTrue((ROOT/'experiments'/row['directory']).is_dir(),row['directory'])
        for old in ('cosine_similarity','word_number','truthfulqa'):
            self.assertFalse((ROOT/'experiments'/old).exists())

    def test_readme_script_paths_exist(self):
        import re
        for name in re.findall(r'experiments/[A-Za-z0-9_./-]+\.(?:py|sh)',(ROOT/'README.md').read_text()):
            self.assertTrue((ROOT/name).is_file(),name)

    def test_local_native_script_paths_exist(self):
        import re
        pattern=re.compile(r'\$(?:HERE|SCRIPT_DIR)/([A-Za-z0-9_.-]+\.(?:py|sh))')
        for script in (ROOT/'experiments').rglob('*.sh'):
            for name in pattern.findall(script.read_text()):
                self.assertTrue((script.parent/name).is_file(),str(script.relative_to(ROOT))+': '+name)


@unittest.skipUnless(importlib.util.find_spec('torch') is not None,'Needs the main ML environment')
class SphericalAdapterTests(unittest.TestCase):
    def test_native_spherical_formula_is_preserved(self):
        import torch
        from experiments.fig2bc_l2s_reliability.spherical_adapter import get_l2s_spherical_intervention
        from methods.spherical_steering.steering import get_spherical_intervention
        class Constant(torch.nn.Module):
            def __init__(self):
                super().__init__();self.vector=torch.nn.Parameter(torch.tensor([1.,2.,3.]),requires_grad=False)
            def forward(self,x):return self.vector.expand_as(x)
        torch.manual_seed(1);hidden=torch.randn(2,5,3);net=Constant()
        expected=get_spherical_intervention(net.vector/net.vector.norm(),-net.vector/net.vector.norm(),20.,.7,-.15)(hidden.clone(),None,start_idx=2)
        actual=get_l2s_spherical_intervention(net,kappa=20.,alpha=.7,beta=-.15)(hidden.clone(),start_idx=2)
        torch.testing.assert_close(actual,expected)
        torch.testing.assert_close(actual[:,:2],hidden[:,:2])

    def test_spherical_generation_hook_uses_native_signature(self):
        import torch
        from types import SimpleNamespace
        from reproducibility.generation import generate
        from methods.spherical_steering.steering import get_spherical_intervention
        class Tokenizer:
            def __call__(self,text,**kwargs):return SimpleNamespace(input_ids=torch.tensor([[1,2]]))
            def decode(self,ids,**kwargs):return 'generated'
        class Model(torch.nn.Module):
            def __init__(self):
                super().__init__();self.embedding=torch.nn.Embedding(6,3);self.block=torch.nn.Identity();self.head=torch.nn.Linear(3,6)
            def get_input_embeddings(self):return self.embedding
            def forward(self,input_ids,**kwargs):return SimpleNamespace(logits=self.head(self.block(self.embedding(input_ids))))
        model=Model();stats={'total':0,'steered':0};mu=torch.tensor([1.,0.,0.])
        result=generate(model,Tokenizer(),'prompt',2,hook=get_spherical_intervention(mu,-mu,stats=stats),block=model.block)
        self.assertEqual(result['generated_tokens'],2);self.assertEqual(stats['total'],2)

    def test_current_checkpoint_training_ids_are_used(self):
        import numpy as np
        from experiments.fig2bc_l2s_reliability.evaluate import require_split
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'checkpoint.npz'
            np.savez(p,predictor_q_indices=[1,3],train_q_indices=[0,1,2,3],test_q_indices=[4,5],fold_idx=0,num_folds=2)
            with np.load(p) as ck:train,test=require_split(ck,p,fold=0,num_folds=2)
            self.assertEqual(train.tolist(),[1,3]);self.assertEqual(test.tolist(),[4,5])


if __name__=='__main__':unittest.main()
