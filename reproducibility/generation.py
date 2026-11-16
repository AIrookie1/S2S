"""Free-generation controls with actual EOS/length accounting; no answer labels used."""
from __future__ import annotations

import argparse
from contextlib import nullcontext
import hashlib
import importlib.metadata
import json
from pathlib import Path
import random
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def fingerprint(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read_prompts(path):
    rows = [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]
    if not rows:
        raise ValueError("The prompt dataset is empty")
    ids = []
    for row in rows:
        if any(k in row for k in ("image", "images", "image_path", "video", "audio")):
            raise ValueError("This is a text-only adapter; multimodal MMsafe needs its original multimodal loader")
        if "id" not in row or not isinstance(row.get("prompt"), str) or not row["prompt"].strip():
            raise ValueError("Each JSONL row needs an id and a nonempty prompt string")
        ids.append(str(row["id"]))
    if len(set(ids)) != len(ids):
        raise ValueError("Prompt IDs must be unique")
    return rows


def generate(model, tokenizer, prompt, max_new_tokens, hook=None, block=None,
             predictor=None, controls=None, eos_ids=()):
    import torch
    from methods.S2S_R.evaluate_llm_judge_S2S_R import fixed_dynamic_controls
    ids = tokenizer(prompt, return_tensors="pt").input_ids.to(model.get_input_embeddings().weight.device)
    emitted = []
    reason = "length"
    for _ in range(max_new_tokens):
        handle = None
        if hook is not None:
            def edit(_module, _inputs, output):
                h = output[0] if isinstance(output, (tuple, list)) else output
                return hook(output, layer_name=None, start_idx=h.shape[1] - 1)
            handle = block.register_forward_hook(edit)
        try:
            context = fixed_dynamic_controls(predictor, controls) if controls is not None else nullcontext()
            with context, torch.inference_mode():
                logits = model(input_ids=ids, use_cache=False).logits[:, -1]
            token = int(logits.argmax(dim=-1).item())
            emitted.append(token)
            ids = torch.cat([ids, torch.tensor([[token]], device=ids.device, dtype=ids.dtype)], dim=-1)
        finally:
            if handle is not None:
                handle.remove()
        if token in eos_ids:
            reason = "eos"
            break
    return dict(text=tokenizer.decode(emitted, skip_special_tokens=True).strip(),
                generated_tokens=len(emitted), token_ids=emitted, finish_reason=reason,
                hit_token_limit=reason == "length")


def render_prompt(tokenizer, row, mode):
    prompt = row["prompt"]
    if mode == "qa":
        return f"Q: {prompt}\nA:"
    if mode == "chat":
        return tokenizer.apply_chat_template([dict(role="user", content=prompt)], tokenize=False,
                                             add_generation_prompt=True)
    return prompt


def validate_checkpoints(s2s_path, l2s_path, layer):
    import numpy as np
    info = {}
    for name, path in (("s2s", s2s_path), ("l2s", l2s_path)):
        with np.load(path, allow_pickle=False) as ck:
            if "layer" in ck.files and int(ck["layer"]) != layer:
                raise ValueError(f"{name} checkpoint layer differs from requested layer")
            info[name] = {k: sorted(int(x) for x in ck[k].reshape(-1))
                          for k in ("train_q_indices", "test_q_indices") if k in ck.files}
    if info["s2s"].get("test_q_indices") != info["l2s"].get("test_q_indices"):
        raise ValueError("L2S and S2S checkpoint test folds differ")
    return info


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", required=True, help="Official Hub ID or downloaded model directory")
    p.add_argument("--model-name", required=True, help="Stable anonymous model label")
    p.add_argument("--layer", type=int, required=True)
    p.add_argument("--s2s-checkpoint", type=Path, required=True)
    p.add_argument("--l2s-checkpoint", type=Path, required=True)
    p.add_argument("--spherical-prototype", type=Path)
    p.add_argument("--prompts", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--max-new-tokens", type=int, default=200)
    p.add_argument("--prompt-format", choices=["as-is", "qa", "chat"], default="as-is")
    p.add_argument("--predictor-mode", choices=["fixed-probe", "online"], default="fixed-probe")
    p.add_argument("--strength", type=float, default=3.0)
    p.add_argument("--slerp-t", type=float, default=0.2)
    p.add_argument("--l2s-alpha", type=float, default=1.0)
    p.add_argument("--weight-kappa", type=float, default=3.0)
    p.add_argument("--margin", type=float, default=-0.7)
    p.add_argument("--dtype", choices=["float16", "bfloat16", "float32"], default="float16")
    p.add_argument("--device", choices=["cuda", "cpu"], default="cuda")
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()
    if args.max_new_tokens <= 0:
        p.error("max-new-tokens must be positive")
    rows = read_prompts(args.prompts)
    splits = validate_checkpoints(args.s2s_checkpoint, args.l2s_checkpoint, args.layer)
    import numpy as np
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from methods.S2S_R.evaluate_predictor import load_components
    from methods.S2S_R.evaluate_S2S_R import get_s2s_r_intervention
    from methods.S2S_L.steering import get_s2s_l_intervention
    from methods.l2s_linear.model import load_l2s_checkpoint
    from methods.l2s_linear.steering import get_l2s_linear_intervention
    from methods.S2S_R.evaluate_llm_judge_S2S_R import capture_probe_last_hidden, controls_from_cached_hidden
    from experiments.rebuttal_R1W2_model_scaling.common import decoder_layers
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    config = {k: v for k, v in vars(args).items()
              if k not in ("model", "prompts", "s2s_checkpoint", "l2s_checkpoint", "spherical_prototype", "output_dir")}
    config.update(dataset_sha256=fingerprint(args.prompts), s2s_sha256=fingerprint(args.s2s_checkpoint),
                  l2s_sha256=fingerprint(args.l2s_checkpoint), checkpoint_splits=splits,
                  model_label=args.model_name, decoding="greedy; EOS allowed; no minimum length; no forced padding",
                  methods=["baseline", "l2s", "s2s_l", "s2s_r"] + (["spherical"] if args.spherical_prototype else []),
                  spherical_sha256=fingerprint(args.spherical_prototype) if args.spherical_prototype else None,
                  token_count="emitted tokens including terminal EOS; limit hit only when EOS did not stop generation",
                  source_sha256={str(x.relative_to(ROOT)):fingerprint(x) for folder in ("methods", "reproducibility")
                                 for x in sorted((ROOT / folder).rglob("*.py"))},
                  versions={name: importlib.metadata.version(name) for name in ("torch", "transformers", "numpy")})
    args.output_dir.mkdir(parents=True, exist_ok=True)
    manifest = args.output_dir / "config.json"
    config_bytes = json.dumps(config, indent=2, sort_keys=True) + "\n"
    if manifest.exists() and manifest.read_text() != config_bytes:
        raise ValueError("Generation configuration changed; choose a new output directory")
    manifest.write_text(config_bytes)
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=False)
    model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=getattr(torch, args.dtype),
                                                 trust_remote_code=False).to(args.device).eval()
    layer_path, blocks = decoder_layers(model)
    if not 0 <= args.layer < len(blocks):
        raise ValueError("Layer index out of range")
    block = blocks[args.layer]
    checkpoint, average, predictor = load_components(str(args.s2s_checkpoint), torch.device(args.device))
    checkpoint.close()
    checkpoint, l2s = load_l2s_checkpoint(args.l2s_checkpoint, torch.device(args.device))
    checkpoint.close()
    predictor.eval()
    l2s.eval()
    hooks = dict(baseline=None, l2s=get_l2s_linear_intervention(l2s, alpha=args.l2s_alpha),
                 s2s_l=get_s2s_l_intervention(average, predictor, method="S2S_L_Static",
                                                   strength=args.strength, slerp_t=args.slerp_t),
                 s2s_r=get_s2s_r_intervention(average, predictor, weight_kappa=args.weight_kappa,
                                                  margin=args.margin))
    if args.spherical_prototype:
        from methods.spherical_steering.steering import get_spherical_intervention
        with np.load(args.spherical_prototype, allow_pickle=False) as proto:
            if set(int(q) for q in proto['test_q_indices']) != set(splits['s2s']['test_q_indices']):
                raise ValueError('Spherical and S2S checkpoints have different test question IDs')
            mu_t=torch.tensor(proto['mu_T'],device=args.device,dtype=torch.float32)
            mu_h=torch.tensor(proto['mu_H'],device=args.device,dtype=torch.float32)
        alpha,beta=(.6,.4) if 'qwen' in args.model_name.lower() else (.7,-.15)
        hooks['spherical']=get_spherical_intervention(mu_t,mu_h,kappa=20.,alpha=alpha,beta=beta)
    eos = model.generation_config.eos_token_id
    eos_ids = set(eos if isinstance(eos, list) else ([] if eos is None else [eos]))
    if tokenizer.eos_token_id is not None:
        eos_ids.add(tokenizer.eos_token_id)
    (args.output_dir / "model_identity.json").write_text(json.dumps(dict(
        config_sha256=hashlib.sha256(model.config.to_json_string().encode()).hexdigest(),
        model_type=model.config.model_type, resolved_commit=getattr(model.config, "_commit_hash", None),
        eos_ids=sorted(eos_ids)), indent=2) + "\n")
    output_rows = []
    for row in rows:
        key = hashlib.sha256(str(row["id"]).encode()).hexdigest()[:24]
        prompt = render_prompt(tokenizer, row, args.prompt_format)
        saved = {}
        for method in hooks:
            target = args.output_dir / "records" / f"{key}_{method}.json"
            if target.exists():
                saved[method] = json.loads(target.read_text())
                if saved[method]["id"] != str(row["id"]) or saved[method]["method"] != method:
                    raise ValueError("Generation cache identity mismatch")
                output_rows.append(saved[method])
                continue
            controls = None
            if method in ("s2s_l", "s2s_r") and args.predictor_mode == "fixed-probe":
                probe = capture_probe_last_hidden(model, tokenizer, "", saved["baseline"]["text"],
                    f"{layer_path}.{args.layer}", prompt_text=prompt)
                controls = controls_from_cached_hidden(hooks[method], predictor, probe["probe_last_hidden"],
                                                       f"{layer_path}.{args.layer}")
            result = generate(model, tokenizer, prompt, args.max_new_tokens, hook=hooks[method], block=block,
                              predictor=predictor, controls=controls, eos_ids=eos_ids)
            result.update(id=str(row["id"]), method=method, prompt=row["prompt"],
                          rendered_prompt_sha256=hashlib.sha256(prompt.encode()).hexdigest(),
                          protocol=args.predictor_mode if method.startswith("s2s_") else "online")
            target.parent.mkdir(parents=True, exist_ok=True)
            tmp = target.with_suffix(".tmp")
            tmp.write_text(json.dumps(result, ensure_ascii=False) + "\n")
            tmp.replace(target)
            saved[method] = result
            output_rows.append(result)
    (args.output_dir / "generations.jsonl").write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in output_rows))
    summary = {m: dict(n=sum(r["method"] == m for r in output_rows),
                      token_limit_rate=sum(r["hit_token_limit"] for r in output_rows if r["method"] == m) / len(rows),
                      mean_generated_tokens=sum(r["generated_tokens"] for r in output_rows if r["method"] == m) / len(rows))
               for m in hooks}
    (args.output_dir / "length_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
