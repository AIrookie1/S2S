"""TruthfulQA BiPO adaptation, held-out S2S calibration and vector diagnostics."""
from __future__ import annotations
import argparse, contextlib, csv, hashlib, json, math, random, sys, time
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
from experiments.rebuttal_R1W2_model_scaling.common import write_json
from methods.S2S_R.eval_utils import get_instruction, load_truthfulqa_mc_dataframe, split_multi_answer, format_best
from methods.S2S_R.predictor import load_feature_pairs
from methods.S2S_L.steering import get_s2s_l_intervention
from methods.S2S_R.evaluate_S2S_R import get_s2s_r_intervention
from experiments.fig2bc_l2s_reliability.metrics import rowwise_vector_metrics
from truthfulqa.models import MC_calcs, set_columns

UPSTREAM = 'https://github.com/CaoYuanpu/BiPO'
COMMIT = 'cef1d00ab108d0e265578589e6b344b200ea3db8'
MODEL = 'llama3.1-8B-Instruct'

class ConstantPredictor(nn.Module):
    """Expose the shared BiPO vector through existing S2S predictor hooks."""
    def __init__(self, vector):
        super().__init__()
        self.register_buffer('vector', vector.float())
    def forward(self, h):
        return self.vector.expand(*h.shape[:-1], -1)


def bipo_loss(chosen, rejected, ref_chosen, ref_rejected, sign, beta):
    # Official bipo_trainer.py dpo_loss: reverse the preference logits for -v.
    return -F.logsigmoid(beta * sign * ((chosen-rejected)-(ref_chosen-ref_rejected))).mean()


def make_hook(method, vector, average, params, predictor=None):
    if method == 'bipo':
        # Official BiPO adds its vector to ALL positions, including the prompt.
        def additive(output, **kwargs):
            h = output[0] if isinstance(output, tuple) else output
            edited = (h.float() + params['strength'] * vector).to(h.dtype)
            return (edited,) + output[1:] if isinstance(output, tuple) else edited
        return additive
    predictor = predictor or ConstantPredictor(vector)
    if method == 's2s_l':
        return get_s2s_l_intervention(average, predictor, method='S2S_L_Static', **params)
    if method == 's2s_r':
        return get_s2s_r_intervention(average, predictor, **params)
    raise ValueError(method)


def tokenize_answer(tokenizer, question, answer):
    base = get_instruction(MODEL) + '\n\n' + f'Q: {question} A:'
    prompt = tokenizer(base, return_tensors='pt').input_ids[0]
    completion = tokenizer(' ' + answer, add_special_tokens=False, return_tensors='pt').input_ids[0]
    if not len(completion):
        raise ValueError('Empty answer')
    ids = torch.cat([prompt, completion])
    labels = ids.clone()
    labels[:len(prompt)] = -100
    return ids, labels, len(prompt)-1


def collate(records, device, hidden=False):
    length = max(len(r['ids']) for r in records)
    ids = torch.zeros(len(records), length, dtype=torch.long)
    labels = torch.full_like(ids, -100)
    mask = torch.zeros_like(ids)
    states = torch.zeros(len(records), length, records[0]['hidden'].shape[-1], dtype=torch.float16) if hidden else None
    for i,r in enumerate(records):
        n = len(r['ids'])
        ids[i,:n], labels[i,:n], mask[i,:n] = r['ids'], r['labels'], 1
        if hidden:
            states[i,:n] = r['hidden']
    return ids.to(device), labels.to(device), mask.to(device), states.to(device) if hidden else None


def sum_logps(logits, labels):
    labels = labels[:,1:]
    # CE gives exactly the sum of completion-token log probabilities.
    losses = F.cross_entropy(logits[:,:-1].float().transpose(1,2), labels, ignore_index=-100, reduction='none')
    return -losses.sum(-1)


def hidden_logps(model, hidden, labels):
    """Project only scored completion positions through the large vocabulary head."""
    targets=labels[:,1:]
    batch_idx,token_idx=torch.where(targets!=-100)
    selected=hidden[:,:-1][batch_idx,token_idx]
    logits=model.lm_head(selected).float()
    losses=F.cross_entropy(logits,targets[batch_idx,token_idx],reduction='none')
    return torch.zeros(len(labels),device=hidden.device,dtype=torch.float32).scatter_add(0,batch_idx,-losses)


@contextlib.contextmanager
def tail_model(model, layer):
    original = model.model.layers
    model.model.layers = nn.ModuleList(list(original)[layer+1:])
    try:
        yield
    finally:
        model.model.layers = original


def tail_logps(model, records, vector=None, sign=1):
    _, labels, mask, hidden = collate(records, model.device, hidden=True)
    if vector is not None:
        hidden = (hidden.float() + sign*vector).to(hidden.dtype)
    output = model.model(inputs_embeds=hidden, attention_mask=mask, use_cache=False)
    return hidden_logps(model, output.last_hidden_state, labels)


def build_cache(model, tokenizer, block, dataset, path, batch_size):
    if path.exists():
        return torch.load(path, weights_only=False, map_location='cpu')
    records = []
    for q,item in enumerate(dataset):
        target = item['mc2_targets']
        for answer,label in zip(target['choices'],target['labels']):
            ids, labels, start = tokenize_answer(tokenizer,item['question'],answer)
            records.append(dict(q=q, positive=int(label), ids=ids, labels=labels, start=start))
    # Stop directly after the intervention block; do not compute unused upper blocks.
    class Captured(Exception):
        pass
    captured = []
    def capture(_module,_inputs,output):
        h = output[0] if isinstance(output,tuple) else output
        captured.append(h.detach().cpu())
        raise Captured()
    handle = block.register_forward_hook(capture)
    try:
        with torch.no_grad():
            for offset in tqdm(range(0,len(records),batch_size),desc='Cache frozen prefix'):
                batch = records[offset:offset+batch_size]
                ids,_,mask,_ = collate(batch,model.device)
                captured.clear()
                try:
                    model(input_ids=ids,attention_mask=mask,use_cache=False)
                except Captured:
                    pass
                if len(captured)!=1:
                    raise RuntimeError('Prefix capture failed')
                for r,h in zip(batch,captured[0]):
                    r['hidden'] = h[:len(r['ids'])].clone()
    finally:
        handle.remove()
    path.parent.mkdir(parents=True,exist_ok=True)
    tmp = path.with_suffix('.tmp')
    torch.save(records,tmp)
    tmp.replace(path)
    return records


def pairs_for_questions(records, questions):
    groups = {}
    for i,r in enumerate(records):
        groups.setdefault(r['q'],{0:[],1:[]})[r['positive']].append(i)
    return {int(q):groups[int(q)] for q in questions}


def train(model, records, split, folder, args):
    path = folder/'bipo.pt'
    if path.exists():
        saved = torch.load(path,weights_only=False,map_location='cpu')
        return saved['vector'].to(model.device)
    rng = random.Random(args.seed + split['fold'])
    torch.manual_seed(args.seed+split['fold'])
    groups = pairs_for_questions(records,split['train'])
    dev = pairs_for_questions(records,split['dev'])
    dev_pairs = [(g[1][0],g[0][0]) for g in dev.values()]
    # Reference probabilities are computed once, with zero steering, on identical tokens.
    needed = sorted({i for g in [*groups.values(),*dev.values()] for c in g.values() for i in c})
    reference = {}
    ref_path = folder/'reference.pt'
    if ref_path.exists():
        reference = torch.load(ref_path,weights_only=False)
    else:
        with torch.no_grad():
            for start in tqdm(range(0,len(needed),args.batch_size*2),desc='Reference logps'):
                indices = needed[start:start+args.batch_size*2]
                values = tail_logps(model,[records[i] for i in indices])
                reference.update({i:float(v) for i,v in zip(indices,values.cpu())})
        torch.save(reference,ref_path)
    vector = nn.Parameter(torch.zeros(model.config.hidden_size,device=model.device,dtype=torch.float32))
    optimizer = torch.optim.AdamW([vector],lr=args.lr,weight_decay=0.0)
    total_steps = args.epochs*math.ceil(len(groups)/args.batch_size)
    history, best, best_vector, step = [], float('inf'), None, 0
    for epoch in range(1,args.epochs+1):
        # Question-balanced stochastic Cartesian pairing: all truthful and false answers
        # remain eligible; draw one pos/neg pair per training question each epoch.
        pairs = [(rng.choice(g[1]),rng.choice(g[0])) for g in groups.values()]
        rng.shuffle(pairs)
        losses=[]
        start_time=time.monotonic()
        for offset in range(0,len(pairs),args.batch_size):
            batch=pairs[offset:offset+args.batch_size]
            indices=[p for p,n in batch]+[n for p,n in batch]
            sign=rng.choice([-1.,1.])
            values=tail_logps(model,[records[i] for i in indices],vector,sign)
            n=len(batch)
            ref=torch.tensor([reference[i] for i in indices],device=model.device)
            loss=bipo_loss(values[:n],values[n:],ref[:n],ref[n:],sign,args.beta)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_([vector],1.0)
            # Same official cosine schedule / 100 warmup steps.
            step+=1
            scale=step/100 if step<=100 else .5*(1+math.cos(math.pi*(step-100)/max(1,total_steps-100)))
            optimizer.param_groups[0]['lr']=args.lr*scale
            optimizer.step()
            losses.append(float(loss.detach()))
        dev_losses=[]
        with torch.no_grad():
            for offset in range(0,len(dev_pairs),args.batch_size):
                batch=dev_pairs[offset:offset+args.batch_size]
                indices=[p for p,n in batch]+[n for p,n in batch]
                ref=torch.tensor([reference[i] for i in indices],device=model.device)
                n=len(batch)
                for sign in [-1.,1.]:
                    values=tail_logps(model,[records[i] for i in indices],vector,sign)
                    dev_losses.append(float(bipo_loss(values[:n],values[n:],ref[:n],ref[n:],sign,args.beta))*n)
        dev_loss=sum(dev_losses)/(2*len(dev_pairs))
        record=dict(epoch=epoch,train_loss=float(np.mean(losses)),dev_loss=dev_loss,norm=float(vector.detach().norm()),seconds=time.monotonic()-start_time)
        history.append(record)
        print('TRAIN',split['fold'],record,flush=True)
        if dev_loss<best:
            best,best_vector=dev_loss,vector.detach().cpu().clone()
            torch.save(dict(vector=best_vector,best_epoch=epoch,dev_loss=best,split=split),path)
        torch.save(dict(vector=vector.detach().cpu(),optimizer=optimizer.state_dict(),epoch=epoch),folder/'last.pt')
        write_json(folder/'history.json',history)
    return best_vector.to(model.device)


def diagnostics(vector, average, split, folder, feature):
    neg,pos,q,_=load_feature_pairs(feature,positive_mode='cartesian')
    targets=pos-neg
    records=[]
    for source,pred in [('BIPO',vector),('Mean',average)]:
        pred=pred.detach().float().cpu().numpy()
        cos,error=rowwise_vector_metrics(np.broadcast_to(pred,targets.shape),targets)
        angle=np.degrees(np.arccos(cos))
        target_norm=np.linalg.norm(targets,axis=1)
        for part in ['train','dev','test']:
            for question in split[part]:
                keep=q==question
                records.append(dict(fold=split['fold'],source=source,split=part,q_idx=int(question),num_pairs=int(keep.sum()),cosine_similarity=float(cos[keep].mean()),angle_degrees=float(angle[keep].mean()),magnitude_relative_error=float(error[keep].mean()),predicted_norm=float(np.linalg.norm(pred)),target_norm=float(target_norm[keep].mean())))
    with (folder/'diagnostics_question.csv').open('w') as f:
        writer=csv.DictWriter(f,fieldnames=list(records[0]));writer.writeheader();writer.writerows(records)
    summary=[]
    for source in ['BIPO','Mean']:
        for part in ['train','dev','test']:
            rows=[r for r in records if r['source']==source and r['split']==part]
            summary.append(dict(source=source,split=part,n_questions=len(rows),**{k:float(np.mean([r[k] for r in rows])) for k in ['cosine_similarity','angle_degrees','magnitude_relative_error','predicted_norm','target_norm']}))
    write_json(folder/'diagnostics_summary.json',summary)


def edit_unpadded(hidden, mask, hook):
    """Apply each sequence's MLP probe at its actual last nonpadding token."""
    edited = hidden.clone()
    for i, length in enumerate(mask.sum(-1).tolist()):
        edited[i:i+1, :length] = hook(hidden[i:i+1, :length], start_idx=0)
    return edited


def evaluate(model,tokenizer,block,frame,questions,hook,path,batch_size,cache):
    details=json.loads(path.read_text()) if path.exists() else []
    done={r['q_idx'] for r in details}
    tag='BiPO'
    for q in tqdm(questions,desc=path.stem):
        if q in done:continue
        sub=frame[frame.hf_idx==q].copy().reset_index(drop=True)
        if len(sub)!=1:raise ValueError('CSV/HF mismatch')
        row=sub.iloc[0]
        true,false=split_multi_answer(row['Correct Answers']),split_multi_answer(row['Incorrect Answers'])
        candidates=[]
        for answer in true+false:
            ids,labels,start=tokenize_answer(tokenizer,row['Question'],answer)
            candidates.append(dict(ids=ids,labels=labels,start=start))
        # Reuse the frozen prefix for exact token sequences already cached for
        # HF answers; compute missing CSV variants once without steering.
        missing=[r for r in candidates if tuple(r['ids'].tolist()) not in cache]
        if missing:
            captured=[]
            class Captured(Exception):pass
            def capture(_m,_i,output):
                captured.append((output[0] if isinstance(output,tuple) else output).detach().cpu())
                raise Captured()
            handle=block.register_forward_hook(capture)
            try:
                with torch.no_grad():
                    for offset in range(0,len(missing),batch_size*2):
                        batch=missing[offset:offset+batch_size*2]
                        ids,_,mask,_=collate(batch,model.device)
                        captured.clear()
                        try:model(input_ids=ids,attention_mask=mask,use_cache=False)
                        except Captured:pass
                        for r,h in zip(batch,captured[0]):
                            cache[tuple(r['ids'].tolist())]=h[:len(r['ids'])].clone()
            finally:handle.remove()
        for r in candidates:r['hidden']=cache[tuple(r['ids'].tolist())]
        scores=[]
        with torch.no_grad(),tail_model(model,14):
            for offset in range(0,len(candidates),batch_size*2):
                _,labels,mask,hidden=collate(candidates[offset:offset+batch_size*2],model.device,hidden=True)
                if hook:hidden=edit_unpadded(hidden,mask,hook)
                result=model.model(inputs_embeds=hidden,attention_mask=mask,use_cache=False)
                scores.extend(hidden_logps(model,result.last_hidden_state,labels).cpu().tolist())
        set_columns(tag,sub)
        MC_calcs(tag,sub,0,scores[:len(true)],scores[len(true):],true,format_best(row['Best Answer']))
        details.append(dict(q_idx=int(q),true_answers=true,false_answers=false,true_scores=scores[:len(true)],false_scores=scores[len(true):],**{m:float(sub.loc[0,f'{tag} {m}']) for m in ['MC1','MC2','MC3']}))
        if len(details)%10==0:write_json(path,details)
    write_json(path,details)
    if {r['q_idx'] for r in details}!=set(questions):raise ValueError('Evaluation coverage mismatch')
    return {m:float(np.mean([r[m] for r in details])) for m in ['MC1','MC2','MC3']}


def report(output):
    all_results=[]
    for fold in [0,1]:
        path=output/f'fold{fold}'/'test_summary.json'
        if not path.exists():return
        all_results.extend(json.loads(path.read_text()))
    rows=[]
    for method in ['bipo','s2s_l','s2s_r']:
        values=[r for r in all_results if r['method']==method]
        rows.append(dict(method=method,n_questions=sum(r['n_questions'] for r in values),**{m:sum(r[m]*r['n_questions'] for r in values)/sum(r['n_questions'] for r in values) for m in ['MC1','MC2','MC3']}))
    write_json(output/'performance.json',rows)
    lines=['# BIPO TruthfulQA results','', '| Method | MC1 (%) | MC2 (%) | MC3 (%) | N |','|---|---:|---:|---:|---:|']
    for r in rows:lines.append('| '+r['method']+' | '+' | '.join(f'{100*r[m]:.2f}' for m in ['MC1','MC2','MC3'])+f" | {r['n_questions']} |")
    lines+=['','## Reliability (question-weighted; Cartesian answer differences)','', '| Split | Cosine | Angle (degrees) | Relative norm error | BIPO norm | Target norm |','|---|---:|---:|---:|---:|---:|']
    summaries=[json.loads((output/f'fold{fold}'/'diagnostics_summary.json').read_text()) for fold in [0,1]]
    for part in ['train','dev','test']:
        values=[r for ss in summaries for r in ss if r['source']=='BIPO' and r['split']==part]
        means={k:sum(r[k]*r['n_questions'] for r in values)/sum(r['n_questions'] for r in values) for k in ['cosine_similarity','angle_degrees','magnitude_relative_error','predicted_norm','target_norm']}
        lines.append('| '+part+' | '+' | '.join(f'{means[k]:.4f}' for k in means)+' |')
    (output/'REPORT.md').write_text('\n'.join(lines)+'\n')
    print('\n'.join(lines),flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model-dir',default='models/Meta-Llama-3.1-8B-Instruct')
    p.add_argument('--feature',default=str(ROOT/'features/llama3.1-8B-Instruct_layer14.npz'))
    p.add_argument('--output',type=Path,default=ROOT/'results/experiment/rebuttal_R2Q4_bipo')
    p.add_argument('--epochs',type=int,default=20)
    p.add_argument('--lr',type=float,default=5e-4)
    p.add_argument('--beta',type=float,default=.1)
    p.add_argument('--batch-size',type=int,default=2)
    p.add_argument('--seed',type=int,default=42)
    p.add_argument('--folds',type=int,nargs='+',default=[0,1])
    p.add_argument('--smoke',action='store_true')
    args=p.parse_args()
    if args.epochs<=0 or args.batch_size<=0: p.error('epochs/batch size must be positive')
    if args.smoke:args.output=args.output.with_name(args.output.name+'_smoke');args.epochs=1
    args.output.mkdir(parents=True,exist_ok=True)
    torch.set_num_threads(4)
    config={k:str(v) if isinstance(v,Path) else v for k,v in vars(args).items() if k!='folds'}
    config.update(source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),upstream=UPSTREAM,commit=COMMIT,model=MODEL,layer=14,dtype='float16',pair_sampling='one random Cartesian pair per question per epoch',train_intervention='all tokens',s2s_intervention='all tokens, matched to original BIPO',selection='bidirectional dev loss for checkpoint; dev MC1 then MC2 for method parameters',feature_sha256=hashlib.sha256(Path(args.feature).read_bytes()).hexdigest())
    manifest=args.output/'config.json'
    if manifest.exists() and json.loads(manifest.read_text())!=config:raise ValueError('Output config mismatch')
    write_json(manifest,config)
    tokenizer=AutoTokenizer.from_pretrained(args.model_dir,local_files_only=True)
    model=AutoModelForCausalLM.from_pretrained(args.model_dir,torch_dtype=torch.float16,attn_implementation='sdpa',local_files_only=True).to('cuda').eval()
    for parameter in model.parameters():parameter.requires_grad_(False)
    from datasets import load_dataset
    dataset=load_dataset('truthful_qa','multiple_choice')['validation']
    if args.smoke:dataset=dataset.select(range(12))
    block=model.model.layers[14]
    records=build_cache(model,tokenizer,block,dataset,args.output/'prefix_cache.pt',args.batch_size*2)
    cache={tuple(r['ids'].tolist()):r['hidden'] for r in records}
    # Audit the split execution using a freshly captured prefix at identical
    # batch/sequence shape. Low precision GEMMs can vary when batch shapes change.
    audit_records=[dict(r) for r in records[:args.batch_size*2]]
    audit_hidden=[]
    def audit_capture(_module,_inputs,result):
        audit_hidden.append((result[0] if isinstance(result,tuple) else result).detach())
    handle=block.register_forward_hook(audit_capture)
    with torch.no_grad():
        ids,labels,mask,_=collate(audit_records,model.device)
        full=sum_logps(model(input_ids=ids,attention_mask=mask,use_cache=False).logits,labels)
        handle.remove()
        for r,h in zip(audit_records,audit_hidden[0]):
            r['hidden']=h[:len(r['ids'])].cpu().clone()
        with tail_model(model,14):cached=tail_logps(model,audit_records)
        error=float((full-cached).abs().max())
        if error>.02:raise ValueError(f'Cached tail equivalence failed: {error}')
    write_json(args.output/'cache_audit.json',dict(max_logprob_error=error))
    frame,_=load_truthfulqa_mc_dataframe('./data/truthfulqa/TruthfulQA.csv')
    neg,pos,vector_q,_=load_feature_pairs(args.feature,positive_mode='cartesian')
    raw_vectors=pos-neg
    for fold in args.folds:
        halves=np.array_split(np.arange(len(dataset)),2)
        outer,test=halves[fold],halves[1-fold]
        shuffled=np.random.default_rng(args.seed+fold).permutation(outer)
        n_dev=max(1,round(len(outer)*.2))
        dev,train_ids=np.sort(shuffled[:n_dev]),np.sort(shuffled[n_dev:])
        split=dict(fold=fold,train=train_ids.tolist(),dev=dev.tolist(),test=test.tolist())
        assert not(set(train_ids)&set(dev) or set(outer)&set(test))
        folder=args.output/f'fold{fold}';folder.mkdir(exist_ok=True)
        write_json(folder/'split.json',split)
        # The same training questions supply BIPO optimization and the mean anchor.
        average=torch.tensor(raw_vectors[np.isin(vector_q,train_ids)].mean(0),device=model.device)
        torch.save(average.cpu(),folder/'mean.pt')
        with tail_model(model,14):vector=train(model,records,split,folder,args)
        diagnostics(vector,average,split,folder,args.feature)
        grids=dict(bipo=[dict(strength=s) for s in [1.,2.,3.]],s2s_l=[dict(strength=s,slerp_t=.2) for s in [1.,2.,3.]],s2s_r=[dict(weight_kappa=3.,margin=m) for m in [-.7,-.3,0.]])
        results=[]
        for method,grid in grids.items():
            cells=[]
            for i,params in enumerate(grid):
                hook=make_hook(method,vector,average,params)
                metrics=evaluate(model,tokenizer,block,frame,split['dev'],hook,folder/f'dev_{method}_{i}.json',args.batch_size,cache)
                cells.append(dict(parameters=params,**metrics))
            chosen=max(cells,key=lambda c:(c['MC1'],c['MC2']))
            write_json(folder/f'selection_{method}.json',dict(candidates=cells,chosen=chosen))
            hook=make_hook(method,vector,average,chosen['parameters'])
            metrics=evaluate(model,tokenizer,block,frame,split['test'],hook,folder/f'test_{method}.json',args.batch_size,cache)
            results.append(dict(method=method,fold=fold,n_questions=len(test),parameters=chosen['parameters'],**metrics))
            print('TEST',results[-1],flush=True)
            write_json(folder/'test_summary.json',results)
    report(args.output)

if __name__=='__main__':main()
