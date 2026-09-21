"""GT-only layer-wise MC1 evaluation for the steering-vector ablation.

For each MC1 question:
  * score the truthful answer with no intervention;
  * for each false answer, build the exact pair direction
        v_ij = z_pos_raw - z_neg_raw
    and apply the additive intervention
        h <- h + v_ij
    from the final prompt token through the answer span at the selected layer;
  * count the question correct iff the truthful-answer sequence log-probability
    is larger than every steered false-answer sequence log-probability.

``calculate_scores_zeroshot`` supplies ``start_idx = len(prompt_ids) - 1`` to the
hook.  Therefore the final prompt hidden state (which predicts the first answer
token) and subsequent answer hidden states are edited.  This is the causal span
that contributes to the answer sequence likelihood used by MC1.

Only ``gt_neg_to_pos`` is recomputed here.  If the layer JSON already exists,
the new GT result is merged into it so previously computed
``linear_shared_mean`` and ``spherical_shared_mean`` values are preserved.
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import re
import sys
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.steering_vector_ablation.vector_utils import (  # noqa: E402
    find_answer_rows,
    load_pair_vector_data,
)
from method.dyn3d_sga.eval_utils import (  # noqa: E402
    HF_NAMES,
    calculate_scores_zeroshot,
    get_additive_intervention,
    get_layer_name,
    load_truthfulqa_mc_dataframe,
    resolve_device,
    set_seed,
)


METHOD = "gt_neg_to_pos"
GT_ALPHA = 1.0


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="GT-only neg->pos pair-vector MC1 evaluation across layers"
    )
    p.add_argument("model_name", nargs="?", default="llama3.1-8B-Instruct")
    p.add_argument(
        "--feature_template",
        required=True,
        help=(
            "Single base pair-feature path template containing {layer}, e.g. "
            "features/model_layer{layer}_dynSVD_pairs.npz. Do not include {fold}."
        ),
    )
    p.add_argument("--layers", nargs="+", type=int, default=None)
    p.add_argument("--folds", nargs="+", type=int, default=[0, 1])
    p.add_argument("--num_folds", type=int, default=2)
    p.add_argument("--model_dir", default=None)
    p.add_argument("--csv_path", default="./TruthfulQA/data/v1/TruthfulQA.csv")
    p.add_argument("--no_instruction", action="store_true")
    p.add_argument("--device", choices=["cuda", "cpu"], default="cuda")
    p.add_argument(
        "--dtype",
        choices=["float16", "bfloat16", "float32"],
        default="float16",
    )
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--max_questions", type=int, default=None)
    p.add_argument(
        "--output_dir",
        default="./results/experiment/steering_vector_ablation",
    )
    p.add_argument("--allow_missing", action="store_true")
    args = p.parse_args()

    if args.num_folds < 2:
        p.error("--num_folds must be >= 2")
    if not args.folds or len(set(args.folds)) != len(args.folds):
        p.error("--folds must contain unique values")
    if any(fold < 0 or fold >= args.num_folds for fold in args.folds):
        p.error("each --folds value must satisfy 0 <= fold < num_folds")
    if "{fold}" in args.feature_template:
        p.error(
            "--feature_template must point to the single *_dynSVD_pairs.npz file; "
            "remove the {fold} placeholder"
        )
    return args


def discover_layers(template: str) -> list[int]:
    probe = template.replace("{layer}", "*")
    found: set[int] = set()
    for path in glob.glob(probe):
        match = re.search(r"layer(\d+)", Path(path).name)
        if match:
            found.add(int(match.group(1)))
    if not found:
        raise FileNotFoundError(f"No feature files matched {probe}")
    return sorted(found)


def render_feature_path(template: str, layer: int) -> Path:
    try:
        rendered = template.format(layer=layer)
    except (KeyError, IndexError, ValueError) as exc:
        raise ValueError(
            f"Invalid --feature_template {template!r}; expected only {{layer}}"
        ) from exc
    return Path(rendered)


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    os.replace(tmp, path)


def make_gt_neg_to_pos_hook(
    z_neg_raw: np.ndarray,
    z_pos_raw: np.ndarray,
    device: torch.device,
):
    """Return h <- h + (z_pos_raw - z_neg_raw), alpha=1.

    The returned hook is the same additive intervention used by the regular
    linear steering code.  ``calculate_scores_zeroshot`` later binds
    ``start_idx`` so the edit begins at the final prompt token and therefore
    affects the logits used to score every answer token.
    """
    z_neg = torch.as_tensor(z_neg_raw, dtype=torch.float32, device=device).flatten()
    z_pos = torch.as_tensor(z_pos_raw, dtype=torch.float32, device=device).flatten()
    if z_neg.shape != z_pos.shape:
        raise ValueError(
            "GT pair shape mismatch: "
            f"z_neg_raw={tuple(z_neg.shape)}, z_pos_raw={tuple(z_pos.shape)}"
        )

    pair_vector = z_pos - z_neg
    if pair_vector.numel() == 0:
        raise ValueError("GT pair vector is empty")

    return get_additive_intervention(pair_vector, GT_ALPHA)


def score(
    model,
    tokenizer,
    question: str,
    answer: str,
    hook,
    layer_name: str,
    device: torch.device,
    args: argparse.Namespace,
) -> float:
    return calculate_scores_zeroshot(
        model,
        tokenizer,
        question,
        [answer],
        hook,
        layer_name,
        device,
        use_instruction=not args.no_instruction,
        model_name=args.model_name,
    )[0]


def load_model(args: argparse.Namespace, device: torch.device):
    path = args.model_dir or HF_NAMES.get(args.model_name, args.model_name)
    tokenizer = AutoTokenizer.from_pretrained(path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    dtype = (
        torch.float32
        if device.type == "cpu"
        else {
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
            "float32": torch.float32,
        }[args.dtype]
    )
    model = AutoModelForCausalLM.from_pretrained(
        path,
        low_cpu_mem_usage=True,
        torch_dtype=dtype,
        device_map="auto" if device.type == "cuda" else None,
        trust_remote_code=True,
    )
    if device.type == "cpu":
        model = model.to(device)
    model.eval()
    model.generation_config.pad_token_id = tokenizer.pad_token_id
    return model, tokenizer, path


def evaluate_layer(
    layer: int,
    args: argparse.Namespace,
    fold_data: list,
    hf_dataset,
    model,
    tokenizer,
    device: torch.device,
) -> dict:
    total_correct = 0
    count = 0
    layer_name = get_layer_name(args.model_name, layer)

    for data in fold_data:
        ids = data.train_questions.tolist()
        if args.max_questions is not None:
            ids = ids[: args.max_questions]

        mappings = {}
        for question_id in ids:
            targets = hf_dataset[int(question_id)]["mc1_targets"]
            choices = list(targets["choices"])
            labels = list(targets["labels"])
            correct = [
                choice for choice, label in zip(choices, labels) if label == 1
            ]
            false = [
                choice for choice, label in zip(choices, labels) if label == 0
            ]
            if len(correct) != 1:
                raise ValueError(
                    f"Question {question_id}: expected exactly one MC1 truthful answer"
                )
            mappings[question_id] = find_answer_rows(
                data,
                question_id,
                correct[0],
                false,
            )

        for question_id in tqdm(
            ids,
            desc=f"GT layer {layer} fold {data.fold_idx}",
            leave=False,
        ):
            positive_row, false_rows = mappings[question_id]
            question = str(hf_dataset[question_id]["question"])
            truth = data.choices[positive_row]

            # The truthful MC1 candidate is always scored exactly as-is.
            truth_score = score(
                model,
                tokenizer,
                question,
                truth,
                None,
                layer_name,
                device,
                args,
            )

            # Each false candidate gets its own oracle pair direction:
            #     v_ij = z_pos_raw - z_neg_raw
            # and the additive hook is active over the causal answer-scoring span.
            false_scores = []
            for row in false_rows:
                key = (positive_row, row)
                if (
                    key not in data.raw_negative_states
                    or key not in data.raw_positive_states
                ):
                    raise KeyError(
                        "GT neg->pos evaluation requires z_neg_raw and z_pos_raw "
                        f"for answer pair {key}. Re-extract the base pairs NPZ."
                    )

                z_neg = data.raw_negative_states[key]
                z_pos = data.raw_positive_states[key]
                hook = make_gt_neg_to_pos_hook(z_neg, z_pos, device)

                # Optional consistency guard: vector_utils should store the same
                # raw pair direction as z_pos_raw - z_neg_raw.
                if key in data.vectors:
                    raw_delta = np.asarray(z_pos, np.float32) - np.asarray(z_neg, np.float32)
                    stored_delta = np.asarray(data.vectors[key], np.float32)
                    if not np.allclose(raw_delta, stored_delta, rtol=1e-4, atol=1e-5):
                        raise ValueError(
                            f"Pair {key}: stored v_T does not match "
                            "z_pos_raw - z_neg_raw"
                        )

                false_scores.append(
                    score(
                        model,
                        tokenizer,
                        question,
                        data.choices[row],
                        hook,
                        layer_name,
                        device,
                        args,
                    )
                )

            total_correct += int(truth_score > max(false_scores))
            count += 1

    if count == 0:
        raise ValueError(f"Layer {layer}: no questions")

    return {
        "layer": layer,
        "model_name": args.model_name,
        "folds": [int(x) for x in args.folds],
        "num_folds": int(args.num_folds),
        "num_questions": count,
        "fold_storage": "single_base_pairs_npz_in_memory_split",
        "parameters": {
            "gt_alpha": GT_ALPHA,
        },
        "definitions": {
            METHOD: (
                "For each false MC1 answer, apply the exact raw pair direction "
                "v_ij = z_pos_raw - z_neg_raw as h <- h + v_ij (alpha=1) "
                "from the final prompt token through the answer-scoring span at "
                "the target decoder layer. The truthful answer is scored without "
                "intervention."
            ),
        },
        "MC1": {METHOD: float(total_correct / count)},
    }


def _check_merge_compatibility(existing: dict, fresh: dict, path: Path) -> None:
    """Refuse to merge GT into an obviously incompatible old result file."""
    for field in ("layer", "model_name", "folds", "num_folds", "num_questions"):
        if field in existing and field in fresh and existing[field] != fresh[field]:
            raise ValueError(
                f"Refusing to merge GT result into {path}: field {field!r} "
                f"differs (existing={existing[field]!r}, fresh={fresh[field]!r})."
            )


def merge_with_existing(path: Path, fresh: dict) -> dict:
    """Preserve already-computed shared-mean results and replace only GT."""
    if not path.is_file():
        return fresh

    existing = json.loads(path.read_text(encoding="utf-8"))
    _check_merge_compatibility(existing, fresh, path)

    merged = dict(existing)
    # Refresh common metadata from this run while retaining unrelated old fields.
    for field in (
        "layer",
        "model_name",
        "folds",
        "num_folds",
        "num_questions",
        "fold_storage",
        "model_path",
        "feature_path",
    ):
        if field in fresh:
            merged[field] = fresh[field]

    parameters = dict(existing.get("parameters", {}))
    parameters.update(fresh.get("parameters", {}))
    merged["parameters"] = parameters

    definitions = dict(existing.get("definitions", {}))
    definitions.update(fresh.get("definitions", {}))
    merged["definitions"] = definitions

    mc1 = dict(existing.get("MC1", {}))
    mc1[METHOD] = fresh["MC1"][METHOD]
    merged["MC1"] = mc1
    return merged


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    device = resolve_device(args.device)
    layers = (
        sorted(set(args.layers))
        if args.layers is not None
        else discover_layers(args.feature_template)
    )

    _, hf_dataset = load_truthfulqa_mc_dataframe(args.csv_path)
    model, tokenizer, model_path = load_model(args, device)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(
        "GT neg->pos MC1 only | "
        f"model={model_path} | layers={layers} | folds={args.folds} | "
        "truth=no intervention | false=h+(z_pos_raw-z_neg_raw)"
    )

    for layer in layers:
        feature_path = render_feature_path(args.feature_template, layer)
        if not feature_path.is_file():
            if args.allow_missing:
                print(f"WARNING: layer {layer} missing {feature_path}; skipped")
                continue
            raise FileNotFoundError(
                f"Layer {layer} missing base pair feature: {feature_path}"
            )

        fold_data = [
            load_pair_vector_data(
                str(feature_path),
                fold_idx=fold,
                num_folds=args.num_folds,
            )
            for fold in args.folds
        ]

        fresh = evaluate_layer(
            layer,
            args,
            fold_data,
            hf_dataset,
            model,
            tokenizer,
            device,
        )
        fresh["model_path"] = model_path
        fresh["feature_path"] = str(feature_path)

        output_path = output_dir / f"{args.model_name}_layer{layer}_mc1.json"
        payload = merge_with_existing(output_path, fresh)
        write_json(output_path, payload)
        print(f"layer {layer}: {METHOD}={payload['MC1'][METHOD]:.12f}")


if __name__ == "__main__":
    main()
