#!/usr/bin/env python3
"""Table 4 generation-efficiency benchmark on one GPU.

The benchmark uses the repository's current steering implementations without
editing them.  Every final generation emits exactly ``max_new_tokens`` tokens
(no early EOS stop) so all methods have an identical numerator.

Two timing views are reported:

* generation_only: one final generation.  For S2S-L/R, the Pass-1/2 probe
  cache is prebuilt outside timing, while cached-hidden -> predictor controls
  is included in Pass 3.
* end_to_end: for S2S-L/R, Pass 1 + Pass 2 + controls + Pass 3 are all timed;
  only the final Pass-3 tokens are counted in the throughput numerator.

No Steering and Spherical Steering have no probe passes, so the two views are
identical for those methods.
"""
from __future__ import annotations

import argparse
import csv
import gc
import json
import math
import sys
import time
from functools import partial
from pathlib import Path
from typing import Any, Callable

import numpy as np
import torch
from baukit import TraceDict
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from methods.S2S_L.steering import get_s2s_l_intervention
from methods.S2S_R.evaluate_S2S_R import get_s2s_r_intervention
from methods.S2S_R.evaluate_predictor import load_components
from methods.S2S_R.evaluate_llm_judge import format_generation_prompt
from methods.S2S_R.evaluate_llm_judge_S2S_R import (
    _NullContext,
    _model_input_device,
    capture_probe_last_hidden,
    controls_from_cached_hidden,
    fixed_dynamic_controls,
)
from methods.spherical_steering.steering import get_spherical_intervention


METHOD_ORDER = (
    "No Steering",
    "Spherical Steering",
    "S2S-L (Ours)",
    "S2S-R (Ours)",
)


def _sync() -> None:
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def _time_call(fn: Callable[[], Any]) -> tuple[float, Any]:
    _sync()
    start = time.perf_counter()
    value = fn()
    _sync()
    return time.perf_counter() - start, value


def generate_fixed_tokens(
    model,
    tokenizer,
    prompt_text: str,
    *,
    hook,
    layer_name: str,
    max_new_tokens: int,
    predictor=None,
    fixed_controls=None,
) -> str:
    """Current token-by-token greedy protocol, but never stop early at EOS."""
    inputs = tokenizer(prompt_text, return_tensors="pt")
    input_ids = inputs.input_ids.to(_model_input_device(model))
    prompt_len = input_ids.shape[1]

    with torch.no_grad():
        for _ in range(max_new_tokens):
            edit = (
                partial(hook, start_idx=input_ids.shape[1] - 1)
                if hook is not None
                else None
            )
            trace_ctx = (
                TraceDict(model, [layer_name], edit_output=edit)
                if edit is not None
                else _NullContext()
            )
            controls_ctx = (
                fixed_dynamic_controls(predictor, fixed_controls)
                if edit is not None and predictor is not None
                else _NullContext()
            )
            with controls_ctx:
                with trace_ctx:
                    outputs = model(input_ids)
            next_token = outputs.logits[:, -1, :].argmax(dim=-1, keepdim=True)
            input_ids = torch.cat([input_ids, next_token], dim=-1)

    answer = tokenizer.decode(
        input_ids[0, prompt_len:], skip_special_tokens=True
    ).strip()
    if "Q:" in answer:
        answer = answer.split("Q:", 1)[0].strip()
    return answer


def _make_prompts(dataset, qids: list[int], args) -> dict[int, str]:
    return {
        qid: format_generation_prompt(
            dataset[qid]["question"],
            preset=args.preset,
            use_instruction=not args.no_instruction,
            model_name=args.model_name,
        )
        for qid in qids
    }


def _single_generation_phase(
    *,
    model,
    tokenizer,
    qids: list[int],
    prompts: dict[int, str],
    layer_name: str,
    max_new_tokens: int,
    hook,
) -> int:
    for qid in qids:
        generate_fixed_tokens(
            model,
            tokenizer,
            prompts[qid],
            hook=hook,
            layer_name=layer_name,
            max_new_tokens=max_new_tokens,
        )
    return len(qids) * max_new_tokens


def _build_probe_cache(
    *,
    model,
    tokenizer,
    dataset,
    qids: list[int],
    prompts: dict[int, str],
    layer_name: str,
    max_new_tokens: int,
) -> dict[int, dict[str, Any]]:
    """Build Pass-1/2 cache outside the generation-only timed region."""
    print(f"Building untimed Pass-1/2 cache for {len(qids)} questions...")
    answers: dict[int, str] = {}
    for i, qid in enumerate(qids, 1):
        answers[qid] = generate_fixed_tokens(
            model,
            tokenizer,
            prompts[qid],
            hook=None,
            layer_name=layer_name,
            max_new_tokens=max_new_tokens,
        )
        if i % 25 == 0 or i == len(qids):
            print(f"  Pass 1 cache: {i}/{len(qids)}")

    cache: dict[int, dict[str, Any]] = {}
    for i, qid in enumerate(qids, 1):
        question = dataset[qid]["question"]
        probe = capture_probe_last_hidden(
            model,
            tokenizer,
            question,
            answers[qid],
            layer_name,
            prompt_text=prompts[qid],
        )
        cache[qid] = {"answer": answers[qid], **probe}
        if i % 25 == 0 or i == len(qids):
            print(f"  Pass 2 cache: {i}/{len(qids)}")
    return cache


def _s2s_generation_only_repeat(
    *,
    model,
    tokenizer,
    qids,
    prompts,
    cache,
    prediction_hook,
    dynamic_hook,
    predictor,
    layer_name,
    max_new_tokens,
) -> dict[str, float | int]:
    def pass3():
        for qid in qids:
            fixed_controls = controls_from_cached_hidden(
                prediction_hook,
                predictor,
                cache[qid]["probe_last_hidden"],
                layer_name,
            )
            generate_fixed_tokens(
                model,
                tokenizer,
                prompts[qid],
                hook=dynamic_hook,
                layer_name=layer_name,
                max_new_tokens=max_new_tokens,
                predictor=predictor,
                fixed_controls=fixed_controls,
            )

    pass3_seconds, _ = _time_call(pass3)
    generated_tokens = len(qids) * max_new_tokens
    return {
        "pass1_seconds": 0.0,
        "pass2_seconds": 0.0,
        "pass3_seconds": pass3_seconds,
        "total_seconds": pass3_seconds,
        "generated_tokens": generated_tokens,
        "tokens_per_sec": generated_tokens / pass3_seconds,
    }


def _s2s_end_to_end_repeat(
    *,
    model,
    tokenizer,
    dataset,
    qids,
    prompts,
    prediction_hook,
    dynamic_hook,
    predictor,
    layer_name,
    max_new_tokens,
) -> dict[str, float | int]:
    answers: dict[int, str] = {}

    def pass1():
        for qid in qids:
            answers[qid] = generate_fixed_tokens(
                model,
                tokenizer,
                prompts[qid],
                hook=None,
                layer_name=layer_name,
                max_new_tokens=max_new_tokens,
            )

    pass1_seconds, _ = _time_call(pass1)

    probes: dict[int, Any] = {}

    def pass2():
        for qid in qids:
            probe = capture_probe_last_hidden(
                model,
                tokenizer,
                dataset[qid]["question"],
                answers[qid],
                layer_name,
                prompt_text=prompts[qid],
            )
            probes[qid] = probe["probe_last_hidden"]

    pass2_seconds, _ = _time_call(pass2)

    def pass3():
        for qid in qids:
            fixed_controls = controls_from_cached_hidden(
                prediction_hook,
                predictor,
                probes[qid],
                layer_name,
            )
            generate_fixed_tokens(
                model,
                tokenizer,
                prompts[qid],
                hook=dynamic_hook,
                layer_name=layer_name,
                max_new_tokens=max_new_tokens,
                predictor=predictor,
                fixed_controls=fixed_controls,
            )

    pass3_seconds, _ = _time_call(pass3)
    total = pass1_seconds + pass2_seconds + pass3_seconds
    generated_tokens = len(qids) * max_new_tokens
    return {
        "pass1_seconds": pass1_seconds,
        "pass2_seconds": pass2_seconds,
        "pass3_seconds": pass3_seconds,
        "total_seconds": total,
        "generated_tokens": generated_tokens,
        "tokens_per_sec": generated_tokens / total,
    }


def _simple_repeat(
    *,
    model,
    tokenizer,
    qids,
    prompts,
    hook,
    layer_name,
    max_new_tokens,
) -> dict[str, float | int]:
    def run():
        return _single_generation_phase(
            model=model,
            tokenizer=tokenizer,
            qids=qids,
            prompts=prompts,
            layer_name=layer_name,
            max_new_tokens=max_new_tokens,
            hook=hook,
        )

    seconds, generated_tokens = _time_call(run)
    return {
        "pass1_seconds": 0.0,
        "pass2_seconds": 0.0,
        "pass3_seconds": seconds,
        "total_seconds": seconds,
        "generated_tokens": int(generated_tokens),
        "tokens_per_sec": generated_tokens / seconds,
    }


def _summarize(repeats: list[dict[str, float | int]]) -> dict[str, Any]:
    tps = np.asarray([float(x["tokens_per_sec"]) for x in repeats], dtype=np.float64)
    total = np.asarray([float(x["total_seconds"]) for x in repeats], dtype=np.float64)
    summary = {
        "tokens_per_sec_mean": float(tps.mean()),
        "tokens_per_sec_std": float(tps.std(ddof=1)) if len(tps) > 1 else 0.0,
        "total_seconds_mean": float(total.mean()),
        "total_seconds_std": float(total.std(ddof=1)) if len(total) > 1 else 0.0,
        "generated_tokens_per_repeat": int(repeats[0]["generated_tokens"]),
        "repeats": repeats,
    }
    for phase in ("pass1_seconds", "pass2_seconds", "pass3_seconds"):
        values = np.asarray([float(x[phase]) for x in repeats], dtype=np.float64)
        summary[f"{phase}_mean"] = float(values.mean())
        summary[f"{phase}_std"] = float(values.std(ddof=1)) if len(values) > 1 else 0.0
    return summary


def _add_overheads(methods: dict[str, dict[str, Any]]) -> None:
    baseline_seconds = methods["No Steering"]["total_seconds_mean"]
    for method in METHOD_ORDER:
        seconds = methods[method]["total_seconds_mean"]
        methods[method]["overhead_percent"] = 100.0 * (seconds / baseline_seconds - 1.0)


def _write_mode_table(output_dir: Path, mode: str, methods: dict[str, dict[str, Any]]) -> None:
    csv_path = output_dir / f"table4_{mode}.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["Method", "Tokens/s", "Std", "Overhead (%)"],
        )
        writer.writeheader()
        for method in METHOD_ORDER:
            row = methods[method]
            writer.writerow(
                {
                    "Method": method,
                    "Tokens/s": row["tokens_per_sec_mean"],
                    "Std": row["tokens_per_sec_std"],
                    "Overhead (%)": row["overhead_percent"],
                }
            )

    tex = [
        r"\begin{tabular}{lrr}",
        r"\toprule",
        r"Method & Tokens/s & Overhead \\",
        r"\midrule",
    ]
    for method in METHOD_ORDER:
        row = methods[method]
        overhead = "--" if method == "No Steering" else f"{row['overhead_percent']:+.2f}\\%"
        tex.append(
            f"{method} & {row['tokens_per_sec_mean']:.1f}\\(\\pm{row['tokens_per_sec_std']:.1f}\\) & {overhead} \\\\"
        )
    tex.extend([r"\bottomrule", r"\end{tabular}"])
    (output_dir / f"table4_{mode}.tex").write_text("\n".join(tex) + "\n", encoding="utf-8")


def _warmup_simple(model, tokenizer, prompts, qids, hook, layer_name, tokens, samples):
    for qid in qids[:samples]:
        generate_fixed_tokens(
            model,
            tokenizer,
            prompts[qid],
            hook=hook,
            layer_name=layer_name,
            max_new_tokens=tokens,
        )
    _sync()


def _warmup_s2s(
    model,
    tokenizer,
    prompts,
    qids,
    cache,
    prediction_hook,
    dynamic_hook,
    predictor,
    layer_name,
    tokens,
    samples,
):
    for qid in qids[:samples]:
        controls = controls_from_cached_hidden(
            prediction_hook, predictor, cache[qid]["probe_last_hidden"], layer_name
        )
        generate_fixed_tokens(
            model,
            tokenizer,
            prompts[qid],
            hook=dynamic_hook,
            layer_name=layer_name,
            max_new_tokens=tokens,
            predictor=predictor,
            fixed_controls=controls,
        )
    _sync()


def main() -> None:
    parser = argparse.ArgumentParser(description="Table 4 generation efficiency benchmark")
    parser.add_argument("model_name", nargs="?", default="llama3.1-8B-Instruct")
    parser.add_argument("--model_dir", required=True)
    parser.add_argument("--s2sl_checkpoint", required=True)
    parser.add_argument("--s2sr_checkpoint", required=True)
    parser.add_argument("--spherical_prototype", required=True)
    parser.add_argument("--layer", type=int, default=14)
    parser.add_argument("--expected_fold", type=int, default=0)
    parser.add_argument("--max_new_tokens", type=int, default=50)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--warmup_samples", type=int, default=2)
    parser.add_argument("--warmup_tokens", type=int, default=8)
    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument("--preset", choices=["qa", "null"], default="null")
    parser.add_argument("--no_instruction", action="store_true")
    parser.add_argument("--spherical_kappa", type=float, default=20.0)
    parser.add_argument("--spherical_alpha", type=float, default=0.7)
    parser.add_argument("--spherical_beta", type=float, default=-0.15)
    parser.add_argument("--s2sl_strength", type=float, default=3.0)
    parser.add_argument("--s2sl_slerp_t", type=float, default=0.2)
    parser.add_argument("--s2sr_weight_kappa", type=float, default=3.0)
    parser.add_argument("--s2sr_margin", type=float, default=-0.7)
    parser.add_argument("--output_dir", type=Path, default=ROOT / "results" / "experiment" / "table4_generation_efficiency")
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("Table 4 is a GPU benchmark; CUDA is required")
    if args.max_new_tokens <= 0 or args.repeats <= 0:
        parser.error("--max_new_tokens and --repeats must be positive")
    if args.warmup_samples < 0 or args.warmup_tokens <= 0:
        parser.error("invalid warmup settings")

    device = torch.device("cuda")
    gpu_name = torch.cuda.get_device_name(0)
    print(f"GPU: {gpu_name}")
    if "3090" not in gpu_name:
        print("WARNING: paper Table 4 is specified for one RTX 3090; recording this GPU name in metadata.")

    ck_l, avg_l, predictor_l = load_components(args.s2sl_checkpoint, device)
    ck_r, avg_r, predictor_r = load_components(args.s2sr_checkpoint, device)
    predictor_l.eval()
    predictor_r.eval()

    for name, ck in (("S2S-L", ck_l), ("S2S-R", ck_r)):
        if "fold_idx" not in ck.files:
            raise KeyError(f"{name} checkpoint missing fold_idx")
        fold = int(np.asarray(ck["fold_idx"]).item())
        if fold != args.expected_fold:
            raise ValueError(f"{name} checkpoint fold={fold}, expected {args.expected_fold}")

    test_l = sorted(int(x) for x in np.asarray(ck_l["test_q_indices"]).reshape(-1))
    test_r = sorted(int(x) for x in np.asarray(ck_r["test_q_indices"]).reshape(-1))
    if test_l != test_r:
        raise ValueError("S2S-L and S2S-R test_q_indices differ; efficiency must use identical prompts")
    qids = test_r
    if args.max_samples is not None:
        if args.max_samples <= 0:
            parser.error("--max_samples must be positive")
        qids = qids[: args.max_samples]
    if not qids:
        raise ValueError("no benchmark questions selected")

    with np.load(args.spherical_prototype, allow_pickle=False) as proto:
        mu_t = torch.as_tensor(proto["mu_T"], dtype=torch.float32, device=device)
        mu_h = torch.as_tensor(proto["mu_H"], dtype=torch.float32, device=device)
        if "test_q_indices" not in proto.files:
            raise KeyError("spherical prototype missing test_q_indices")
        spherical_test = sorted(int(x) for x in np.asarray(proto["test_q_indices"]).reshape(-1))
        spherical_fold = int(np.asarray(proto["fold_idx"]).item()) if "fold_idx" in proto.files else None
    # This catches the repository's inverse fold convention: Dyn fold 0 has the
    # same held-out questions as Spherical fold 1 when NUM_FOLDS=2.
    if spherical_test != test_r:
        raise ValueError(
            "spherical prototype held-out questions do not match the S2S fold. "
            "For the current 2-fold code, S2S fold 0 normally pairs with spherical fold 1."
        )

    dataset = load_dataset("truthful_qa", "multiple_choice")["validation"]
    prompts = _make_prompts(dataset, qids, args)
    layer_name = f"model.layers.{args.layer}"

    tokenizer = AutoTokenizer.from_pretrained(args.model_dir, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        args.model_dir,
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True,
    )
    model.eval()

    spherical_hook = get_spherical_intervention(
        mu_t,
        mu_h,
        args.spherical_kappa,
        args.spherical_alpha,
        args.spherical_beta,
    )
    s2sl_prediction_hook = get_s2s_l_intervention(
        avg_l,
        predictor_l,
        strength=args.s2sl_strength,
        slerp_t=args.s2sl_slerp_t,
    )
    s2sl_dynamic_hook = get_s2s_l_intervention(
        avg_l,
        predictor_l,
        strength=args.s2sl_strength,
        slerp_t=args.s2sl_slerp_t,
    )
    s2sr_prediction_hook = get_s2s_r_intervention(
        avg_r,
        predictor_r,
        weight_kappa=args.s2sr_weight_kappa,
        margin=args.s2sr_margin,
    )
    s2sr_dynamic_hook = get_s2s_r_intervention(
        avg_r,
        predictor_r,
        weight_kappa=args.s2sr_weight_kappa,
        margin=args.s2sr_margin,
    )

    # The cache is intentionally outside generation-only timing.
    probe_cache = _build_probe_cache(
        model=model,
        tokenizer=tokenizer,
        dataset=dataset,
        qids=qids,
        prompts=prompts,
        layer_name=layer_name,
        max_new_tokens=args.max_new_tokens,
    )

    warm_n = min(args.warmup_samples, len(qids))
    if warm_n:
        print("Warm-up...")
        _warmup_simple(model, tokenizer, prompts, qids, None, layer_name, args.warmup_tokens, warm_n)
        _warmup_simple(model, tokenizer, prompts, qids, spherical_hook, layer_name, args.warmup_tokens, warm_n)
        _warmup_s2s(model, tokenizer, prompts, qids, probe_cache, s2sl_prediction_hook, s2sl_dynamic_hook, predictor_l, layer_name, args.warmup_tokens, warm_n)
        _warmup_s2s(model, tokenizer, prompts, qids, probe_cache, s2sr_prediction_hook, s2sr_dynamic_hook, predictor_r, layer_name, args.warmup_tokens, warm_n)

    generation_only_raw: dict[str, list[dict[str, float | int]]] = {m: [] for m in METHOD_ORDER}
    print(f"\nGeneration-only benchmark: {args.repeats} repeats")
    for repeat in range(args.repeats):
        print(f"  repeat {repeat + 1}/{args.repeats}: No Steering")
        generation_only_raw["No Steering"].append(
            _simple_repeat(
                model=model, tokenizer=tokenizer, qids=qids, prompts=prompts,
                hook=None, layer_name=layer_name, max_new_tokens=args.max_new_tokens,
            )
        )
        print(f"  repeat {repeat + 1}/{args.repeats}: Spherical Steering")
        generation_only_raw["Spherical Steering"].append(
            _simple_repeat(
                model=model, tokenizer=tokenizer, qids=qids, prompts=prompts,
                hook=spherical_hook, layer_name=layer_name, max_new_tokens=args.max_new_tokens,
            )
        )
        print(f"  repeat {repeat + 1}/{args.repeats}: S2S-L")
        generation_only_raw["S2S-L (Ours)"].append(
            _s2s_generation_only_repeat(
                model=model, tokenizer=tokenizer, qids=qids, prompts=prompts,
                cache=probe_cache, prediction_hook=s2sl_prediction_hook,
                dynamic_hook=s2sl_dynamic_hook, predictor=predictor_l,
                layer_name=layer_name, max_new_tokens=args.max_new_tokens,
            )
        )
        print(f"  repeat {repeat + 1}/{args.repeats}: S2S-R")
        generation_only_raw["S2S-R (Ours)"].append(
            _s2s_generation_only_repeat(
                model=model, tokenizer=tokenizer, qids=qids, prompts=prompts,
                cache=probe_cache, prediction_hook=s2sr_prediction_hook,
                dynamic_hook=s2sr_dynamic_hook, predictor=predictor_r,
                layer_name=layer_name, max_new_tokens=args.max_new_tokens,
            )
        )

    end_to_end_raw: dict[str, list[dict[str, float | int]]] = {
        "No Steering": list(generation_only_raw["No Steering"]),
        "Spherical Steering": list(generation_only_raw["Spherical Steering"]),
        "S2S-L (Ours)": [],
        "S2S-R (Ours)": [],
    }
    print(f"\nEnd-to-end S2S benchmark: {args.repeats} repeats")
    for repeat in range(args.repeats):
        print(f"  repeat {repeat + 1}/{args.repeats}: S2S-L Pass1+Pass2+Pass3")
        end_to_end_raw["S2S-L (Ours)"].append(
            _s2s_end_to_end_repeat(
                model=model, tokenizer=tokenizer, dataset=dataset, qids=qids,
                prompts=prompts, prediction_hook=s2sl_prediction_hook,
                dynamic_hook=s2sl_dynamic_hook, predictor=predictor_l,
                layer_name=layer_name, max_new_tokens=args.max_new_tokens,
            )
        )
        print(f"  repeat {repeat + 1}/{args.repeats}: S2S-R Pass1+Pass2+Pass3")
        end_to_end_raw["S2S-R (Ours)"].append(
            _s2s_end_to_end_repeat(
                model=model, tokenizer=tokenizer, dataset=dataset, qids=qids,
                prompts=prompts, prediction_hook=s2sr_prediction_hook,
                dynamic_hook=s2sr_dynamic_hook, predictor=predictor_r,
                layer_name=layer_name, max_new_tokens=args.max_new_tokens,
            )
        )

    generation_only = {method: _summarize(generation_only_raw[method]) for method in METHOD_ORDER}
    end_to_end = {method: _summarize(end_to_end_raw[method]) for method in METHOD_ORDER}
    _add_overheads(generation_only)
    _add_overheads(end_to_end)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    report = {
        "experiment": "Table 4 - Generation efficiency on one GPU",
        "hardware": {"gpu": gpu_name, "cuda_visible_device_count": torch.cuda.device_count()},
        "model_name": args.model_name,
        "model_dir": args.model_dir,
        "layer": int(args.layer),
        "s2s_fold_idx": int(args.expected_fold),
        "spherical_fold_idx": spherical_fold,
        "num_questions": len(qids),
        "question_indices": qids,
        "protocol": {
            "greedy": True,
            "fixed_generated_tokens": int(args.max_new_tokens),
            "early_eos_stop": False,
            "repeats": int(args.repeats),
            "warmup_samples": int(warm_n),
            "warmup_tokens": int(args.warmup_tokens),
            "throughput_numerator": "final generated tokens only",
            "generation_only_s2s": "Pass-1/2 cache outside timing; cached hidden -> predictor controls included in Pass 3",
            "end_to_end_s2s": "Pass 1 + Pass 2 + cached-hidden controls + Pass 3 all timed; only Pass-3 tokens counted",
            "overhead": "100 * (mean runtime / mean No-Steering runtime - 1)",
        },
        "parameters": {
            "spherical": {"kappa": args.spherical_kappa, "alpha": args.spherical_alpha, "beta": args.spherical_beta},
            "s2s_l": {"strength": args.s2sl_strength, "slerp_t": args.s2sl_slerp_t},
            "s2s_r": {"weight_kappa": args.s2sr_weight_kappa, "margin": args.s2sr_margin},
            "preset": args.preset,
            "instruction": not args.no_instruction,
        },
        "paths": {
            "s2sl_checkpoint": str(args.s2sl_checkpoint),
            "s2sr_checkpoint": str(args.s2sr_checkpoint),
            "spherical_prototype": str(args.spherical_prototype),
        },
        "generation_only": generation_only,
        "end_to_end": end_to_end,
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    _write_mode_table(args.output_dir, "generation_only", generation_only)
    _write_mode_table(args.output_dir, "end_to_end", end_to_end)

    for title, methods in (("Generation-only", generation_only), ("End-to-end", end_to_end)):
        print(f"\n{title}")
        print("Method                    Tokens/s        Overhead")
        for method in METHOD_ORDER:
            row = methods[method]
            overhead = "--" if method == "No Steering" else f"{row['overhead_percent']:+.2f}%"
            print(
                f"{method:<25} {row['tokens_per_sec_mean']:.2f}+-{row['tokens_per_sec_std']:.2f}   {overhead}"
            )
    print(f"Saved Table 4 outputs to {args.output_dir.resolve()}")

    del model
    torch.cuda.empty_cache()
    gc.collect()


if __name__ == "__main__":
    main()
