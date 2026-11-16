import numpy as np
import torch
from experiments.bipo_truthfulqa.run import bipo_loss, make_hook, sum_logps


def test_negative_steering_reverses_preferences():
    chosen=torch.tensor([2.0],requires_grad=True)
    rejected=torch.tensor([0.0],requires_grad=True)
    ref=torch.tensor([0.0])
    positive=bipo_loss(chosen,rejected,ref,ref,1,.1)
    negative=bipo_loss(-chosen,-rejected,ref,ref,-1,.1)
    torch.testing.assert_close(positive,negative)
    positive.backward()
    assert chosen.grad.item()<0 and rejected.grad.item()>0


def test_bipo_is_added_to_prompt_as_in_official_wrapper():
    hidden=torch.zeros(2,5,3)
    vector=torch.tensor([1.,2.,3.])
    result=make_hook('bipo',vector,vector,{'strength':2})((hidden,'rest'),start_idx=3)
    torch.testing.assert_close(result[0],(vector*2).expand_as(hidden))
    assert result[1]=='rest'


def test_s2s_r_preserves_hidden_norm_and_prompt_prefix():
    torch.manual_seed(2)
    hidden=torch.randn(2,5,8)
    vector=torch.randn(8)
    mean=torch.randn(8)
    hook=make_hook('s2s_r',vector,mean,dict(weight_kappa=3.,margin=-.7))
    result=hook(hidden,start_idx=2)
    torch.testing.assert_close(result[:,:2],hidden[:,:2])
    torch.testing.assert_close(result.norm(dim=-1),hidden.norm(dim=-1),atol=1e-5,rtol=1e-5)


def test_completion_logprobs_ignore_prompt_and_padding():
    logits=torch.zeros(1,5,7)
    labels=torch.tensor([[-100,-100,2,3,-100]])
    torch.testing.assert_close(sum_logps(logits,labels),torch.tensor([-2*np.log(7)],dtype=torch.float32))


def test_sparse_head_matches_full_completion_projection_and_gradient():
    from experiments.bipo_truthfulqa.run import hidden_logps
    torch.manual_seed(5)
    model=torch.nn.Module()
    model.lm_head=torch.nn.Linear(4,9,bias=False)
    h=torch.randn(2,6,4,requires_grad=True)
    labels=torch.tensor([[-100,-100,1,2,3,-100],[-100,-100,-100,4,5,6]])
    sparse=hidden_logps(model,h,labels)
    dense=sum_logps(model.lm_head(h),labels)
    torch.testing.assert_close(sparse,dense)
    sparse_grad=torch.autograd.grad(sparse.sum(),h,retain_graph=True)[0]
    dense_grad=torch.autograd.grad(dense.sum(),h)[0]
    torch.testing.assert_close(sparse_grad,dense_grad)


def test_mlp_probe_uses_real_answer_end_in_padded_batch():
    from experiments.bipo_truthfulqa.run import edit_unpadded
    from method.dyn3d_sga.dyn_mlp import DirectVectorMLP
    torch.manual_seed(42)
    avg=torch.randn(8)
    predictor=DirectVectorMLP(8,4,8).eval()
    hidden=torch.randn(2,6,8)
    mask=torch.tensor([[1,1,1,0,0,0],[1,1,1,1,1,1]])
    hook=make_hook('s2s_l',None,avg,dict(strength=1.,slerp_t=.2),predictor=predictor)
    batched=edit_unpadded(hidden,mask,hook)
    torch.testing.assert_close(batched[:1,:3],hook(hidden[:1,:3],start_idx=0))
    torch.testing.assert_close(batched[:1,3:],hidden[:1,3:])
    torch.testing.assert_close(batched[1:],hook(hidden[1:],start_idx=0))


def test_mlp_additive_matches_s2s_probe_and_raw_magnitude():
    from experiments.bipo_truthfulqa.run_mlp_controls import additive_mlp_hook
    class Predictor(torch.nn.Module):
        def forward(self, x):
            return 2*x
    h=torch.tensor([[[3.,0.],[0.,4.]]])
    result=additive_mlp_hook(Predictor(),alpha=.5)(h)
    torch.testing.assert_close(result,h+torch.tensor([0.,1.]))
    torch.testing.assert_close(additive_mlp_hook(Predictor(),alpha=0)(h),h)
