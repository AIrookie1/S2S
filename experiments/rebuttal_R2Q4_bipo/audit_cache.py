"""Check full versus cached-prefix evaluation for all three selected interventions."""
import argparse,json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from experiments.rebuttal_R2Q4_bipo.rebuttal_R2Q4_run import (
    AutoModelForCausalLM,collate,hidden_logps,make_hook,tail_model,torch,write_json,
)


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--input',type=Path,default=ROOT/'results/experiment/rebuttal_R2Q4_bipo')
    args=p.parse_args()
    config=json.loads((args.input/'config.json').read_text())
    torch.set_num_threads(4)
    model=AutoModelForCausalLM.from_pretrained(config['model_dir'],torch_dtype=torch.float16,attn_implementation='sdpa',local_files_only=True).cuda().eval()
    records=torch.load(args.input/'prefix_cache.pt',weights_only=False,map_location='cpu')[:4]
    block=model.model.layers[14]
    ids,labels,mask,stored=collate(records,model.device,hidden=True)
    prefix=[]
    handle=block.register_forward_hook(lambda _m,_i,out:prefix.append(out.detach().clone()))
    with torch.no_grad():model.model(input_ids=ids,attention_mask=mask,use_cache=False)
    handle.remove()
    results=[]
    for fold in [0,1]:
        folder=args.input/f'fold{fold}'
        vector=torch.load(folder/'bipo.pt',weights_only=False)['vector'].to(model.device)
        average=torch.load(folder/'mean.pt',weights_only=True).to(model.device)
        for method in ['bipo','s2s_l','s2s_r']:
            params=json.loads((folder/f'selection_{method}.json').read_text())['chosen']['parameters']
            hook=make_hook(method,vector,average,params)
            def edit(_m,_i,out):return hook(out,start_idx=0)
            handle=block.register_forward_hook(edit)
            with torch.no_grad():
                h=model.model(input_ids=ids,attention_mask=mask,use_cache=False).last_hidden_state
                full=hidden_logps(model,h,labels)
            handle.remove()
            with torch.no_grad(),tail_model(model,14):
                edited=hook(prefix[0],start_idx=0)
                h=model.model(inputs_embeds=edited,attention_mask=mask,use_cache=False).last_hidden_state
                cached=hidden_logps(model,h,labels)
            error=float((full-cached).abs().max())
            if error>.03:raise ValueError(f'Steered cache audit failed: fold={fold} method={method} error={error}')
            with torch.no_grad(),tail_model(model,14):
                edited=hook(stored,start_idx=0)
                h=model.model(inputs_embeds=edited,attention_mask=mask,use_cache=False).last_hidden_state
                actual=hidden_logps(model,h,labels)
            stored_error=float((full-actual).abs().max())
            if stored_error>.1:raise ValueError(f'Stored-prefix FP16 audit failed: {stored_error}')
            results.append(dict(fold=fold,method=method,max_answer_logprob_error=error,stored_prefix_max_logprob_error=stored_error))
    write_json(args.input/'steered_cache_audit.json',results)
    print(json.dumps(results,indent=2))

if __name__=='__main__':main()
