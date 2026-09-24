#!/usr/bin/env python3
"""Table 3: component ablation for S2S-R on TruthfulQA MC.

This experiment intentionally lives outside ``method/`` so the released method
implementation remains unchanged.  The full row delegates to the current
``get_3d_dynamic_intervention`` hook verbatim; the four ablated rows reuse the
same current primitives and checkpoint.

Rows (all evaluated on the checkpoint's held-out split):

1. MLP x, S2S x, AW x, Margin x:
   SLERP(normalize(h), normalize(v_avg), t=0.4), restore ||h||.
2. MLP y, S2S x, AW x, Margin x:
   predict v_hat from the segment's last position, then
   SLERP(normalize(h), normalize(v_hat), t=0.4), restore ||h||.
3. MLP y, S2S y, AW x, Margin x:
   current symmetric 2-SLERP composition over X/V/R with fixed
   (alpha,beta,gamma)=(0.6,0.3,0.1).
4. MLP y, S2S y, AW y, Margin x:
   current adaptive weights and symmetric 2-SLERP, applied to every token.
5. MLP y, S2S y, AW y, Margin y:
   the unmodified current S2S-R hook, including its margin gate.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path
from typing import Callable

import numpy as np
import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from method.dyn3d_sga.dyn_mlp import predict_residual
from method.dyn3d_sga.evaluate_dyn_mlp import load_components
from method.dyn3d_sga.evaluate_3d_dyn import (
    adaptive_3way_weights,
    get_3d_dynamic_intervention,
    norm,
    slerp,
    two_slerp_sym,
)
from method.dyn3d_sga.eval_utils import (
    HF_NAMES,
    calculate_scores_fewshot,
    calculate_scores_zeroshot,
    format_best,
    get_layer_name,
    load_truthfulqa_mc_dataframe,
    resolve_device,
    set_seed,
    split_multi_answer,
)
from truthfulqa.models import MC_calcs, set_columns


VARIANTS = (
    "vavg",
    "mlp",
    "s2s_fixed",
    "s2s_adaptive",
    "s2s_full",
)

VARIANT_META = {
    "vavg": {"MLP": False, "S2S": False, "AW": False, "Margin Gate": False},
    "mlp": {"MLP": True, "S2S": False, "AW": False, "Margin Gate": False},
    "s2s_fixed": {"MLP": True, "S2S": True, "AW": False, "Margin Gate": False},
    "s2s_adaptive": {"MLP": True, "S2S": True, "AW": True, "Margin Gate": False},
    "s2s_full": {"MLP": True, "S2S": True, "AW": True, "Margin Gate": True},
}


def _validate_fixed_weights(weights: tuple[float, float, float]) -> None:
    if any((not math.isfinite(x) or x < 0.0) for x in weights):
        raise ValueError("fixed weights must be finite and non-negative")
    if not math.isclose(sum(weights), 1.0, rel_tol=0.0, abs_tol=1e-8):
        raise ValueError(f"fixed weights must sum to 1, got {sum(weights)}")


def get_ablation_intervention(
    avg: torch.Tensor,
    predictor,
    variant: str,
    *,
    single_slerp_t: float = 0.4,
    fixed_weights: tuple[float, float, float] = (0.6, 0.3, 0.1),
    weight_kappa: float = 3.0,
    margin: float = -0.7,
    eps: float = 1e-8,
    small_angle_threshold: float = 1e-4,
    stats: dict | None = None,
) -> Callable:
    """Return one Table-3 ablation hook.

    ``s2s_full`` is delegated to the current method implementation so this
    experiment cannot silently diverge from the released S2S-R code.
    """
    if variant not in VARIANTS:
        raise ValueError(f"unknown variant {variant!r}; choose from {VARIANTS}")
    if not math.isfinite(single_slerp_t) or not 0.0 <= single_slerp_t <= 1.0:
        raise ValueError("single_slerp_t must be finite and in [0,1]")
    _validate_fixed_weights(fixed_weights)

    if variant == "s2s_full":
        return get_3d_dynamic_intervention(
            avg,
            predictor,
            weight_kappa=weight_kappa,
            margin=margin,
            eps=eps,
            small_angle_threshold=small_angle_threshold,
            stats=stats,
        )

    mean_direction = norm(avg, eps)

    def hook(output, layer_name=None, start_idx=None):
        del layer_name
        hidden, rest = (
            (output[0], output[1:]) if isinstance(output, tuple) else (output, None)
        )
        if not torch.is_tensor(hidden) or hidden.ndim != 3:
            return output

        start = max(
            0,
            min(
                int(start_idx if start_idx is not None else hidden.shape[1] - 1),
                hidden.shape[1] - 1,
            ),
        )
        original = hidden[:, start:, :].float()
        original_norm = torch.linalg.vector_norm(
            original, dim=-1, keepdim=True
        ).clamp_min(eps)
        x = norm(original, eps)
        token_count = int(original.shape[0] * original.shape[1])

        if variant == "vavg":
            target = mean_direction.view(1, 1, -1).expand_as(x)
            rotated_direction = slerp(
                x,
                target,
                single_slerp_t,
                eps,
                small_angle_threshold,
            )
            alpha = beta = gamma = None

        elif variant == "mlp":
            # Match current S2S-R sequence semantics: predict once from the
            # final position of the intervention segment and broadcast it.
            v_hat, _ = predict_residual(predictor, x[:, -1, :], avg, eps=eps)
            target = norm(v_hat, eps).view(v_hat.shape[0], 1, -1).expand_as(x)
            rotated_direction = slerp(
                x,
                target,
                single_slerp_t,
                eps,
                small_angle_threshold,
            )
            alpha = beta = gamma = None

        else:
            _, residual = predict_residual(predictor, x[:, -1, :], avg, eps=eps)
            v = mean_direction.view(1, 1, -1).expand_as(x)
            r = residual.float().view(residual.shape[0], 1, -1).expand_as(x)

            if variant == "s2s_fixed":
                a0, b0, g0 = fixed_weights
                alpha = torch.full_like(x[..., :1], a0)
                beta = torch.full_like(x[..., :1], b0)
                gamma = torch.full_like(x[..., :1], g0)
            elif variant == "s2s_adaptive":
                alpha, beta, gamma, _ = adaptive_3way_weights(
                    x,
                    v,
                    r,
                    kappa=weight_kappa,
                    temperature=1.0,
                )
            else:  # pragma: no cover - guarded above
                raise AssertionError(variant)

            rotated_direction = two_slerp_sym(
                x,
                v,
                r,
                alpha,
                beta,
                gamma,
                eps,
                small_angle_threshold,
            )

        candidate = rotated_direction * original_norm
        edited = hidden.clone()
        # All four ablated rows have no margin gate by experimental design.
        edited[:, start:, :] = candidate.to(hidden.dtype)

        if stats is not None:
            stats["total"] = stats.get("total", 0) + token_count
            stats["steered"] = stats.get("steered", 0) + token_count
            if alpha is not None:
                stats["weight_count"] = stats.get("weight_count", 0) + token_count
                stats["alpha_sum"] = stats.get("alpha_sum", 0.0) + float(alpha.sum().item())
                stats["beta_sum"] = stats.get("beta_sum", 0.0) + float(beta.sum().item())
                stats["gamma_sum"] = stats.get("gamma_sum", 0.0) + float(gamma.sum().item())

        return (edited,) + rest if rest is not None else edited

    return hook


def _evaluate_variant(
    *,
    variant: str,
    dataframe,
    ids: set[int],
    model,
    tokenizer,
    avg,
    predictor,
    args,
    layer_name: str,
):
    results = dataframe[dataframe["hf_idx"].isin(ids)].copy().reset_index(drop=True)
    if args.max_samples is not None:
        results = results.iloc[: args.max_samples].copy().reset_index(drop=True)
    if len(results) == 0:
        raise ValueError("evaluation split is empty")

    tag = variant
    set_columns(tag, results)
    stats = {"total": 0, "steered": 0, "weight_count": 0}
    hook = get_ablation_intervention(
        avg,
        predictor,
        variant,
        single_slerp_t=args.single_slerp_t,
        fixed_weights=tuple(args.fixed_weights),
        weight_kappa=args.weight_kappa,
        margin=args.margin,
        eps=args.slerp_eps,
        small_angle_threshold=args.small_angle_threshold,
        stats=stats,
    )

    progress = tqdm(range(len(results)), desc=variant, unit="q")
    for i in progress:
        row = results.iloc[i]
        true_answers = split_multi_answer(row["Correct Answers"])
        false_answers = split_multi_answer(row["Incorrect Answers"])
        common = dict(
            model=model,
            tokenizer=tokenizer,
            question=row["Question"],
            base_hook_fn=hook,
            layer_name=layer_name,
            device=args.resolved_device,
            use_instruction=not args.no_instruction,
            model_name=args.model_name,
        )
        if args.few_shot:
            scores_true = calculate_scores_fewshot(
                choices=true_answers, preset=args.preset, **common
            )
            scores_false = calculate_scores_fewshot(
                choices=false_answers, preset=args.preset, **common
            )
        else:
            scores_true = [
                calculate_scores_zeroshot(choices=[answer], **common)[0]
                for answer in true_answers
            ]
            scores_false = [
                calculate_scores_zeroshot(choices=[answer], **common)[0]
                for answer in false_answers
            ]

        MC_calcs(
            tag,
            results,
            i,
            scores_true,
            scores_false,
            true_answers,
            format_best(row["Best Answer"]),
        )
        running = {
            metric: float(results[f"{tag} {metric}"].iloc[: i + 1].mean())
            for metric in ("MC1", "MC2", "MC3")
        }
        progress.set_description(
            f"{variant} MC1:{running['MC1']:.3f} "
            f"MC2:{running['MC2']:.3f} MC3:{running['MC3']:.3f}"
        )

        if args.save_details:
            results.loc[i, f"{tag} scores_true"] = str(scores_true)
            results.loc[i, f"{tag} scores_false"] = str(scores_false)

    metrics = {
        metric: float(results[f"{tag} {metric}"].mean())
        for metric in ("MC1", "MC2", "MC3")
    }
    metrics["Avg"] = float(np.mean([metrics["MC1"], metrics["MC2"], metrics["MC3"]]))

    row = {"variant": variant, **VARIANT_META[variant], **metrics}
    row["steer_token_ratio"] = stats["steered"] / max(stats["total"], 1)
    if stats.get("weight_count", 0):
        count = stats["weight_count"]
        row["mean_weights"] = {
            "alpha": stats["alpha_sum"] / count,
            "beta": stats["beta_sum"] / count,
            "gamma": stats["gamma_sum"] / count,
        }

    detail = {
        "experiment": "Table 3 - S2S-R component ablation",
        "variant": variant,
        "components": VARIANT_META[variant],
        "model_name": args.model_name,
        "layer": int(args.layer),
        "fold_idx": int(args.expected_fold),
        "eval_split": args.eval_split,
        "checkpoint_path": str(args.checkpoint_path),
        "parameters": {
            "single_slerp_t": float(args.single_slerp_t),
            "fixed_weights": [float(x) for x in args.fixed_weights],
            "weight_kappa": float(args.weight_kappa),
            "margin": float(args.margin),
            "few_shot": bool(args.few_shot),
            "preset": args.preset,
        },
        "metrics": metrics,
        "steering_stats": {
            "steered_tokens": int(stats["steered"]),
            "total_tokens": int(stats["total"]),
            "steer_token_ratio": row["steer_token_ratio"],
        },
        "results": json.loads(results.to_json(orient="records", force_ascii=False)),
    }
    if "mean_weights" in row:
        detail["steering_stats"]["mean_weights"] = row["mean_weights"]
    return row, detail


def _write_outputs(rows: list[dict], output_dir: Path, metadata: dict) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    details_dir = output_dir / "details"
    details_dir.mkdir(parents=True, exist_ok=True)

    summary = {"metadata": metadata, "rows": rows}
    (output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )

    csv_path = output_dir / "table3_component_ablation.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["MLP", "S2S", "AW", "Margin Gate", "MC1", "MC2", "MC3", "Avg"],
        )
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row[key] for key in writer.fieldnames})

    tex_lines = [
        r"\begin{tabular}{cccccccc}",
        r"\toprule",
        r"MLP & S2S & AW & Margin Gate & MC1 & MC2 & MC3 & Avg. \\",
        r"\midrule",
    ]
    for row in rows:
        marks = [r"\checkmark" if row[key] else r"\times" for key in ("MLP", "S2S", "AW", "Margin Gate")]
        tex_lines.append(
            " & ".join(marks)
            + f" & {row['MC1']:.3f} & {row['MC2']:.3f} & {row['MC3']:.3f} & {row['Avg']:.3f} \\\\" 
        )
    tex_lines.extend([r"\bottomrule", r"\end{tabular}"])
    (output_dir / "table3_component_ablation.tex").write_text(
        "\n".join(tex_lines) + "\n", encoding="utf-8"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Table 3 S2S-R component ablation")
    parser.add_argument("model_name", nargs="?", default="llama3.1-8B-Instruct")
    parser.add_argument("--model_dir", default=None)
    parser.add_argument("--checkpoint_path", required=True)
    parser.add_argument("--csv_path", default="./TruthfulQA/data/v1/TruthfulQA.csv")
    parser.add_argument("--layer", type=int, default=14)
    parser.add_argument("--expected_fold", type=int, default=0)
    parser.add_argument("--eval_split", choices=["test", "train", "all"], default="test")
    parser.add_argument("--single_slerp_t", type=float, default=0.4)
    parser.add_argument("--fixed_weights", type=float, nargs=3, default=(0.6, 0.3, 0.1), metavar=("ALPHA", "BETA", "GAMMA"))
    parser.add_argument("--weight_kappa", type=float, default=3.0)
    parser.add_argument("--margin", type=float, default=-0.7)
    parser.add_argument("--few_shot", action="store_true")
    parser.add_argument("--preset", default="qa")
    parser.add_argument("--no_instruction", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", choices=["cuda", "cpu"], default="cuda")
    parser.add_argument("--dtype", choices=["float16", "bfloat16", "float32"], default="float16")
    parser.add_argument("--slerp_eps", type=float, default=1e-8)
    parser.add_argument("--small_angle_threshold", type=float, default=1e-4)
    parser.add_argument("--variants", nargs="+", choices=VARIANTS, default=list(VARIANTS))
    parser.add_argument("--max_samples", type=int, default=None, help="Debug only; official run leaves this unset.")
    parser.add_argument("--save_details", action="store_true")
    parser.add_argument("--output_dir", type=Path, default=ROOT / "results" / "experiment" / "component_ablation")
    args = parser.parse_args()

    _validate_fixed_weights(tuple(args.fixed_weights))
    set_seed(args.seed)
    device = resolve_device(args.device)
    args.resolved_device = device

    checkpoint, avg, predictor = load_components(args.checkpoint_path, device)
    if "fold_idx" not in checkpoint.files:
        raise KeyError("checkpoint must contain fold_idx for the official ablation")
    actual_fold = int(np.asarray(checkpoint["fold_idx"]).item())
    if actual_fold != args.expected_fold:
        raise ValueError(
            f"checkpoint fold mismatch: expected fold {args.expected_fold}, got {actual_fold}"
        )

    model_path = args.model_dir or HF_NAMES.get(args.model_name, args.model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    if device.type == "cpu":
        dtype, device_map = torch.float32, None
    else:
        dtype = {
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
            "float32": torch.float32,
        }[args.dtype]
        device_map = "auto"
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        low_cpu_mem_usage=True,
        torch_dtype=dtype,
        device_map=device_map,
        trust_remote_code=True,
    )
    if device.type == "cpu":
        model = model.to(device)
    model.eval()
    tokenizer.pad_token = tokenizer.pad_token or tokenizer.eos_token
    model.generation_config.pad_token_id = tokenizer.pad_token_id

    dataframe, hf_dataset = load_truthfulqa_mc_dataframe(args.csv_path)
    if args.eval_split == "all":
        ids = set(range(len(hf_dataset)))
    else:
        key = "train_q_indices" if args.eval_split == "train" else "test_q_indices"
        if key not in checkpoint.files:
            raise KeyError(f"checkpoint missing {key}")
        ids = set(int(x) for x in np.asarray(checkpoint[key]).reshape(-1))

    layer_name = get_layer_name(args.model_name, args.layer)
    rows = []
    args.output_dir.mkdir(parents=True, exist_ok=True)
    details_dir = args.output_dir / "details"
    details_dir.mkdir(parents=True, exist_ok=True)

    for variant in args.variants:
        print(f"\n=== Table 3 variant: {variant} ===")
        row, detail = _evaluate_variant(
            variant=variant,
            dataframe=dataframe,
            ids=ids,
            model=model,
            tokenizer=tokenizer,
            avg=avg,
            predictor=predictor,
            args=args,
            layer_name=layer_name,
        )
        rows.append(row)
        (details_dir / f"{variant}.json").write_text(
            json.dumps(detail, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )

    # Preserve canonical paper row order even if --variants was supplied out of order.
    order = {name: i for i, name in enumerate(VARIANTS)}
    rows.sort(key=lambda row: order[row["variant"]])
    metadata = {
        "experiment": "Table 3 - Component ablation on TruthfulQA",
        "model_name": args.model_name,
        "model_path": str(model_path),
        "layer": int(args.layer),
        "fold_idx": actual_fold,
        "eval_split": args.eval_split,
        "checkpoint_path": str(args.checkpoint_path),
        "definitions": {
            "single_slerp_t": float(args.single_slerp_t),
            "fixed_weights_alpha_beta_gamma": [float(x) for x in args.fixed_weights],
            "weight_kappa": float(args.weight_kappa),
            "margin": float(args.margin),
            "avg": "(MC1 + MC2 + MC3) / 3",
            "full_row": "delegates to current method/dyn3d_sga/evaluate_3d_dyn.py",
        },
    }
    _write_outputs(rows, args.output_dir, metadata)

    print("\nMLP S2S AW Gate | MC1    MC2    MC3    Avg")
    for row in rows:
        marks = ["Y" if row[k] else "N" for k in ("MLP", "S2S", "AW", "Margin Gate")]
        print(
            f" {'   '.join(marks)} | {row['MC1']:.4f} {row['MC2']:.4f} "
            f"{row['MC3']:.4f} {row['Avg']:.4f}"
        )
    print(f"Saved Table 3 outputs to {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
