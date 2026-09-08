import argparse,os,sys,gc,torch,numpy as np,pandas as pd
from functools import partial
from pathlib import Path
from tqdm import tqdm
from datasets import load_dataset
from transformers import AutoTokenizer,AutoModelForCausalLM
from baukit import TraceDict
ROOT=Path(__file__).resolve().parents[3];sys.path.insert(0,str(ROOT))
from method.dyn3d_sga.evaluate_dynSVD import load_components
from method.dyn3d_sga.evaluate_3d_dyn import get_3d_dynamic_intervention

def gen(model,tok,prompt,hook,layer,max_new):
 ids=tok(prompt,return_tensors='pt').input_ids.to(model.device); n=ids.shape[1]
 for _ in range(max_new):
  h=partial(hook,start_idx=ids.shape[1]-1) if hook else None
  with torch.no_grad():
   with TraceDict(model,[layer],edit_output=h) if h else _null(): o=model(ids)
  t=o.logits[:,-1].argmax(-1,keepdim=True); ids=torch.cat([ids,t],-1)
  if t.item()==tok.eos_token_id: break
 return tok.decode(ids[0,n:],skip_special_tokens=True).strip()
class _null:
 def __enter__(self): return self
 def __exit__(self,*a): pass
def main():
 p=argparse.ArgumentParser();p.add_argument('model_name');p.add_argument('--model_dir',required=True);p.add_argument('--checkpoint_path',required=True);p.add_argument('--layer',type=int,required=True);p.add_argument('--output_dir',required=True);p.add_argument('--truth_judge_path',required=True);p.add_argument('--info_judge_path',required=True);p.add_argument('--max_new_tokens',type=int,default=50);p.add_argument('--alpha',type=float,default=.6);p.add_argument('--weight_kappa',type=float,default=5.0);p.add_argument('--weight_temperature',type=float,default=0.1);p.add_argument('--hidden_prior',type=float,default=0.50);p.add_argument('--mean_prior',type=float,default=0.25);p.add_argument('--residual_prior',type=float,default=0.25);p.add_argument('--use_gate',type=int,default=0);p.add_argument('--max_samples',type=int);p.add_argument('--disable_steering',action='store_true');a=p.parse_args();os.makedirs(a.output_dir,exist_ok=True)
 tok=AutoTokenizer.from_pretrained(a.model_dir,trust_remote_code=True); model=AutoModelForCausalLM.from_pretrained(a.model_dir,torch_dtype=torch.float16,device_map='auto',trust_remote_code=True); ck,avg,basis,router,scale=load_components(a.checkpoint_path,torch.device('cuda')); hook=None if a.disable_steering else get_3d_dynamic_intervention(avg,basis,router,scale,use_gate=False)
 ds=load_dataset('truthful_qa','multiple_choice')['validation']; ids=list(range(len(ds)))[:a.max_samples] if a.max_samples else range(len(ds)); out=[]
 for i in tqdm(ids):
  q=ds[i]['question']; base=gen(model,tok,f'Q: {q}\nA:',None,f'model.layers.{a.layer}',a.max_new_tokens); prompt=f'Q: {q}\nInitial answer: {base}\nFinal answer:'; ans=gen(model,tok,prompt,hook,f'model.layers.{a.layer}',a.max_new_tokens); out.append({'question':q,'initial_answer':base,'answer':ans})
 del model;torch.cuda.empty_cache();
 for path,tag in [(a.truth_judge_path,'truth_prob'),(a.info_judge_path,'info_prob')]:
  jtok=AutoTokenizer.from_pretrained(path); j=AutoModelForCausalLM.from_pretrained(path,torch_dtype=torch.float16,device_map='auto'); vals=[]
  for r in out:
   pr=f"Q: {r['question']}\\nA: {r['answer']}\\n{'Helpful' if tag.startswith('info') else 'True'}:"; x=jtok(pr,return_tensors='pt').to(j.device); yes=jtok.encode('yes',add_special_tokens=False)[0]
   with torch.no_grad(): z=j.generate(**x,max_new_tokens=1,return_dict_in_generate=True,output_scores=True)
   vals.append(torch.softmax(z.scores[0][0],-1)[yes].item())
  for r,v in zip(out,vals): r[tag]=v
  del j
 pd.DataFrame(out).to_csv(Path(a.output_dir)/f'{a.model_name}_3d_judge.csv',index=False)
if __name__=='__main__': main()
