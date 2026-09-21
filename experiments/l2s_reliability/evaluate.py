#!/usr/bin/env python3
"""Mean-vs-L2S × Linear-vs-Spherical reliability experiment on TruthfulQA.

The experiment is deliberately a clean 2×2 design:

    vector source:       Mean vector | L2S predicted vector
    steering geometry:  Linear      | Spherical

Fairness constraints:
* The mean vector is computed from exactly ``l2s_train_q_indices`` stored in
  each L2S checkpoint -- the same questions used to optimize that fold's MLP.
* MC1 is evaluated only on the checkpoint's held-out ``test_q_indices``.
* Linear conditions share one ``linear_alpha``.
* Spherical conditions share the exact Spherical Steering geometry and one
  ``kappa/alpha/beta`` setting.
* Direction/magnitude diagnostics use the same raw pair targets
  ``v_T = z_pos_raw - z_neg_raw``. Pair metrics are averaged within question
  before being written to the main diagnostics CSV.

Outputs:
* mc1.csv                per-fold MC1/MC2/MC3 for the 2×2 conditions (+ baseline)
* mc1_question.csv       question-level MC1 values for auditable pooling
* diagnostics_question.csv
                         question-level cosine and relative-norm error
* diagnostics_pair.csv  optional pair-level diagnostics
* summary.json           experiment metadata
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.l2s_reliability.metrics import (
    aggregate_by_question,
    mean_vector_from_questions,
    rowwise_vector_metrics,
)
from method.dyn3d_sga.eval_utils import (
    HF_NAMES,
    calculate_scores_zeroshot,
    format_best,
    get_additive_intervention,
    get_layer_name,
    load_truthfulqa_mc_dataframe,
    resolve_device,
    set_seed,
    split_multi_answer,
)
from method.l2s.model import load_l2s_checkpoint, load_l2s_pairs
from method.l2s.spherical_steering import get_l2s_spherical_intervention
from method.l2s.steering import get_l2s_linear_intervention
from method.spherical_steering.steering import get_spherical_intervention
from truthfulqa.models import MC_calcs, set_columns


MODEL_DEFAULTS = {
    "llama3.1-8B-Instruct": {
        "layer": 14,
        "spherical_kappa": 20.0,
        "spherical_alpha": 0.7,
        "spherical_beta": -0.15,
    },
    "Qwen2.5-7B-Instruct": {
        "layer": 19,
        "spherical_kappa": 20.0,
        "spherical_alpha": 0.6,
        "spherical_beta": 0.4,
    },
}

CONDITIONS = (
    ("mean_linear", "Mean", "Linear"),
    ("mean_spherical", "Mean", "Spherical"),
    ("l2s_linear", "L2S", "Linear"),
    ("l2s_spherical", "L2S", "Spherical"),
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="TruthfulQA L2S reliability: Mean/L2S × Linear/Spherical"
    )
    p.add_argument("model_name", nargs="?", default="llama3.1-8B-Instruct")
    p.add_argument(
        "--checkpoint_template",
        required=True,
        help="L2S checkpoint template containing {fold}",
    )
    p.add_argument(
        "--feature_path",
        default=None,
        help="Optional feature NPZ override. Otherwise use checkpoint feature_path.",
    )
    p.add_argument("--folds", type=int, nargs="+", default=[0, 1])
    p.add_argument("--num_folds", type=int, default=2)
    p.add_argument("--model_dir", default=None)
    p.add_argument("--csv_path", default="./TruthfulQA/data/v1/TruthfulQA.csv")
    p.add_argument(
        "--layer",
        type=int,
        default=None,
        help="Intervention layer. Default: feature metadata, then model quickstart default.",
    )

    p.add_argument(
        "--linear_alpha",
        type=float,
        default=1.4,
        help="Shared additive strength for Mean+Linear and L2S+Linear.",
    )
    p.add_argument("--spherical_kappa", type=float, default=None)
    p.add_argument("--spherical_alpha", type=float, default=None)
    p.add_argument("--spherical_beta", type=float, default=None)

    p.add_argument("--diagnostic_batch_size", type=int, default=4096)
    p.add_argument("--no_instruction", action="store_true")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", choices=["cuda", "cpu"], default="cuda")
    p.add_argument(
        "--dtype",
        choices=["float16", "bfloat16", "float32"],
        default="float16",
    )
    p.add_argument(
        "--output_dir",
        default="./results/experiment/l2s_reliability",
    )
    p.add_argument(
        "--save_pair_diagnostics",
        action="store_true",
        help="Also write pair-level cosine/norm-error rows.",
    )
    p.add_argument(
        "--no_baseline",
        action="store_true",
        help="Skip the no-steering MC1 reference line.",
    )
    p.add_argument(
        "--skip_mc",
        action="store_true",
        help="Run vector diagnostics only (useful before loading the base LLM).",
    )
    p.add_argument(
        "--skip_diagnostics",
        action="store_true",
        help="Run MC evaluation only.",
    )
    p.add_argument(
        "--max_mc_questions",
        type=int,
        default=None,
        help="Debug/smoke-test only: truncate each fold's MC test questions.",
    )
    args = p.parse_args()

    if "{fold}" not in args.checkpoint_template:
        p.error("--checkpoint_template must contain {fold}")
    if args.num_folds < 2:
        p.error("--num_folds must be >= 2")
    if not args.folds or len(set(args.folds)) != len(args.folds):
        p.error("--folds must contain unique fold indices")
    if any(f < 0 or f >= args.num_folds for f in args.folds):
        p.error("every fold must satisfy 0 <= fold < num_folds")
    if not math.isfinite(args.linear_alpha) or args.linear_alpha == 0:
        p.error("--linear_alpha must be finite and nonzero")
    if args.diagnostic_batch_size <= 0:
        p.error("--diagnostic_batch_size must be positive")
    if args.max_mc_questions is not None and args.max_mc_questions <= 0:
        p.error("--max_mc_questions must be positive")
    if args.skip_mc and args.skip_diagnostics:
        p.error("cannot use --skip_mc and --skip_diagnostics together")
    return args


def render_checkpoint_path(template: str, fold: int) -> Path:
    try:
        rendered = template.format(fold=fold)
    except (KeyError, IndexError, ValueError) as exc:
        raise ValueError(
            f"invalid checkpoint template {template!r}; expected only {{fold}}"
        ) from exc
    path = Path(rendered).expanduser()
    if not path.is_file():
        raise FileNotFoundError(f"checkpoint not found: {path}")
    return path


def _model_defaults(model_name: str) -> dict[str, float | int]:
    if model_name in MODEL_DEFAULTS:
        return dict(MODEL_DEFAULTS[model_name])
    lower = model_name.lower()
    if "llama-3.1" in lower or "llama3.1" in lower:
        return dict(MODEL_DEFAULTS["llama3.1-8B-Instruct"])
    if "qwen2.5" in lower:
        return dict(MODEL_DEFAULTS["Qwen2.5-7B-Instruct"])
    return {
        "layer": 14,
        "spherical_kappa": 20.0,
        "spherical_alpha": 0.15,
        "spherical_beta": 0.1,
    }


def resolve_feature_path(
    checkpoint: Any,
    checkpoint_path: Path,
    override: str | None,
) -> Path:
    if override:
        path = Path(override).expanduser()
        if not path.is_file():
            raise FileNotFoundError(f"feature file not found: {path}")
        return path.resolve()

    if "feature_path" not in checkpoint.files:
        raise KeyError(
            f"{checkpoint_path} has no feature_path; pass --feature_path explicitly"
        )
    raw = str(np.asarray(checkpoint["feature_path"]).item())
    candidates = [
        Path(raw).expanduser(),
        ROOT / raw,
        checkpoint_path.resolve().parent / raw,
    ]
    for path in candidates:
        if path.is_file():
            return path.resolve()
    raise FileNotFoundError(
        f"checkpoint-recorded feature path not found: {raw}; "
        "pass --feature_path explicitly"
    )


def require_split(
    checkpoint: Any,
    checkpoint_path: Path,
    *,
    fold: int,
    num_folds: int,
) -> tuple[np.ndarray, np.ndarray]:
    required = {"l2s_train_q_indices", "test_q_indices"}
    missing = required.difference(checkpoint.files)
    if missing:
        raise KeyError(
            f"{checkpoint_path} missing required split fields: {sorted(missing)}"
        )

    if "fold_idx" in checkpoint.files:
        stored = int(np.asarray(checkpoint["fold_idx"]).item())
        if stored != fold:
            raise ValueError(
                f"{checkpoint_path} stores fold_idx={stored}, expected {fold}"
            )
    if "num_folds" in checkpoint.files:
        stored = int(np.asarray(checkpoint["num_folds"]).item())
        if stored != num_folds:
            raise ValueError(
                f"{checkpoint_path} stores num_folds={stored}, expected {num_folds}"
            )

    train_q = np.sort(
        np.unique(
            np.asarray(checkpoint["l2s_train_q_indices"], dtype=np.int64).reshape(-1)
        )
    )
    test_q = np.sort(
        np.unique(np.asarray(checkpoint["test_q_indices"], dtype=np.int64).reshape(-1))
    )
    if len(train_q) == 0 or len(test_q) == 0:
        raise ValueError("L2S train/test question split must be non-empty")
    overlap = np.intersect1d(train_q, test_q)
    if len(overlap):
        raise ValueError(
            f"L2S train/test questions overlap: {overlap[:10].tolist()}"
        )
    return train_q, test_q


def resolve_experiment_hparams(
    args: argparse.Namespace,
    feature_meta: dict[str, Any],
) -> dict[str, float | int]:
    defaults = _model_defaults(args.model_name)

    if args.layer is not None:
        layer = int(args.layer)
    elif "layer" in feature_meta:
        layer = int(np.asarray(feature_meta["layer"]).item())
    else:
        layer = int(defaults["layer"])

    kappa = (
        float(args.spherical_kappa)
        if args.spherical_kappa is not None
        else float(defaults["spherical_kappa"])
    )
    alpha = (
        float(args.spherical_alpha)
        if args.spherical_alpha is not None
        else float(defaults["spherical_alpha"])
    )
    beta = (
        float(args.spherical_beta)
        if args.spherical_beta is not None
        else float(defaults["spherical_beta"])
    )
    if not math.isfinite(kappa) or kappa <= 0:
        raise ValueError("spherical kappa must be finite and positive")
    if not math.isfinite(alpha) or not 0.0 <= alpha <= 1.0:
        raise ValueError("spherical alpha must be finite and in [0, 1]")
    if not math.isfinite(beta) or beta >= 1.0:
        raise ValueError("spherical beta must be finite and < 1")

    return {
        "layer": layer,
        "linear_alpha": float(args.linear_alpha),
        "spherical_kappa": kappa,
        "spherical_alpha": alpha,
        "spherical_beta": beta,
    }


def predict_l2s_vectors(
    mlp,
    inputs: np.ndarray,
    *,
    batch_size: int,
) -> np.ndarray:
    inputs = np.asarray(inputs, dtype=np.float32)
    if inputs.ndim != 2:
        raise ValueError("L2S diagnostic inputs must have shape [N, D]")
    device = next(mlp.parameters()).device
    chunks: list[np.ndarray] = []
    mlp.eval()
    with torch.no_grad():
        for start in range(0, len(inputs), batch_size):
            x = torch.as_tensor(
                inputs[start : start + batch_size],
                dtype=torch.float32,
                device=device,
            )
            chunks.append(mlp(x).detach().cpu().numpy().astype(np.float32))
    return np.concatenate(chunks, axis=0)


def diagnostic_rows_for_fold(
    *,
    fold: int,
    mlp,
    z_neg: np.ndarray,
    v_t: np.ndarray,
    q_indices: np.ndarray,
    train_q: np.ndarray,
    test_q: np.ndarray,
    mean_vector: np.ndarray,
    batch_size: int,
    save_pairs: bool,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    question_rows: list[dict[str, Any]] = []
    pair_rows: list[dict[str, Any]] = []

    for split, split_q in (("Train", train_q), ("Test", test_q)):
        mask = np.isin(q_indices, split_q)
        if not mask.any():
            raise ValueError(f"fold {fold} {split} has no feature pairs")
        split_z = z_neg[mask]
        split_v = v_t[mask]
        split_ids = q_indices[mask]

        l2s_pred = predict_l2s_vectors(
            mlp,
            split_z,
            batch_size=batch_size,
        )
        mean_pred = np.broadcast_to(mean_vector, split_v.shape).copy()

        for source, prediction in (("Mean", mean_pred), ("L2S", l2s_pred)):
            cosine, mag_error = rowwise_vector_metrics(prediction, split_v)
            aggregated = aggregate_by_question(split_ids, cosine, mag_error)
            for row in aggregated:
                question_rows.append(
                    {
                        "fold": fold,
                        "split": split,
                        "vector_source": source,
                        **row,
                    }
                )

            if save_pairs:
                local_indices = np.nonzero(mask)[0]
                for j, global_idx in enumerate(local_indices):
                    pair_rows.append(
                        {
                            "fold": fold,
                            "split": split,
                            "vector_source": source,
                            "pair_index": int(global_idx),
                            "q_idx": int(split_ids[j]),
                            "cosine_similarity": float(cosine[j]),
                            "magnitude_relative_error": float(mag_error[j]),
                            "predicted_norm": float(np.linalg.norm(prediction[j])),
                            "target_norm": float(np.linalg.norm(split_v[j])),
                        }
                    )
    return question_rows, pair_rows


def normalize_vector(vector: np.ndarray, device: torch.device) -> torch.Tensor:
    tensor = torch.as_tensor(vector, dtype=torch.float32, device=device).flatten()
    norm = torch.linalg.vector_norm(tensor)
    if norm.item() <= 1e-8:
        raise ValueError("cannot use a zero mean vector for spherical steering")
    return tensor / norm


def make_condition_hook(
    condition: str,
    *,
    mean_vector: np.ndarray,
    mlp,
    hparams: dict[str, float | int],
    device: torch.device,
    stats: dict[str, int | float],
):
    if condition == "mean_linear":
        return get_additive_intervention(
            torch.as_tensor(mean_vector, dtype=torch.float32, device=device),
            float(hparams["linear_alpha"]),
            stats=stats,
        )
    if condition == "mean_spherical":
        mu_t = normalize_vector(mean_vector, device)
        return get_spherical_intervention(
            mu_t,
            -mu_t,
            float(hparams["spherical_kappa"]),
            float(hparams["spherical_alpha"]),
            float(hparams["spherical_beta"]),
            stats=stats,
        )
    if condition == "l2s_linear":
        return get_l2s_linear_intervention(
            mlp,
            alpha=float(hparams["linear_alpha"]),
            stats=stats,
        )
    if condition == "l2s_spherical":
        return get_l2s_spherical_intervention(
            mlp,
            kappa=float(hparams["spherical_kappa"]),
            alpha=float(hparams["spherical_alpha"]),
            beta=float(hparams["spherical_beta"]),
            stats=stats,
        )
    raise KeyError(f"unknown condition: {condition}")


def load_base_model(
    args: argparse.Namespace,
    device: torch.device,
):
    model_path = args.model_dir or HF_NAMES.get(args.model_name, args.model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
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
        model_path,
        low_cpu_mem_usage=True,
        torch_dtype=dtype,
        device_map="auto" if device.type == "cuda" else None,
        trust_remote_code=True,
    )
    if device.type == "cpu":
        model = model.to(device)
    model.eval()
    model.generation_config.pad_token_id = tokenizer.pad_token_id
    return model, tokenizer, model_path


def evaluate_mc_condition(
    *,
    condition: str,
    vector_source: str,
    geometry: str,
    fold: int,
    base_dataframe,
    model,
    tokenizer,
    hook,
    layer_name: str,
    device: torch.device,
    model_name: str,
    no_instruction: bool,
    stats: dict[str, Any],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    dataframe = base_dataframe.copy().reset_index(drop=True)
    tag = f"rel_{condition}"
    set_columns(tag, dataframe)

    progress = tqdm(
        range(len(dataframe)),
        desc=f"fold{fold} {condition}",
        leave=False,
    )
    for i in progress:
        row = dataframe.iloc[i]
        true = split_multi_answer(row["Correct Answers"])
        false = split_multi_answer(row["Incorrect Answers"])
        common = dict(
            model=model,
            tokenizer=tokenizer,
            question=row["Question"],
            base_hook_fn=hook,
            layer_name=layer_name,
            device=device,
            use_instruction=not no_instruction,
            model_name=model_name,
        )
        scores_true = [
            calculate_scores_zeroshot(choices=[answer], **common)[0]
            for answer in true
        ]
        scores_false = [
            calculate_scores_zeroshot(choices=[answer], **common)[0]
            for answer in false
        ]
        MC_calcs(
            tag,
            dataframe,
            i,
            scores_true,
            scores_false,
            true,
            format_best(row["Best Answer"]),
        )
        running_mc1 = float(dataframe[f"{tag} MC1"].iloc[: i + 1].mean())
        progress.set_postfix(MC1=f"{running_mc1:.3f}")

    metrics = {
        name: float(dataframe[f"{tag} {name}"].mean())
        for name in ("MC1", "MC2", "MC3")
    }
    metrics.update(
        {
            "fold": int(fold),
            "condition": condition,
            "vector_source": vector_source,
            "geometry": geometry,
            "questions": int(len(dataframe)),
            "steered_tokens": int(stats.get("steered", 0)),
            "total_tokens": int(stats.get("total", 0)),
            "steering_rate": float(
                stats.get("steered", 0) / max(stats.get("total", 0), 1)
            ),
        }
    )

    question_rows = [
        {
            "fold": int(fold),
            "condition": condition,
            "vector_source": vector_source,
            "geometry": geometry,
            "hf_idx": int(dataframe.iloc[i]["hf_idx"]),
            "MC1": float(dataframe.iloc[i][f"{tag} MC1"]),
        }
        for i in range(len(dataframe))
    ]
    return metrics, question_rows


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    os.replace(tmp, path)


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    os.replace(tmp, path)


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    device = resolve_device(args.device)

    checkpoint_paths = {
        fold: render_checkpoint_path(args.checkpoint_template, fold)
        for fold in args.folds
    }

    # Resolve the common feature artifact from the first checkpoint.
    first_fold = args.folds[0]
    first_checkpoint, first_mlp = load_l2s_checkpoint(
        checkpoint_paths[first_fold], device="cpu"
    )
    try:
        feature_path = resolve_feature_path(
            first_checkpoint,
            checkpoint_paths[first_fold],
            args.feature_path,
        )
    finally:
        first_checkpoint.close()
        del first_mlp

    z_neg, v_t, q_indices, feature_meta = load_l2s_pairs(feature_path)
    hparams = resolve_experiment_hparams(args, feature_meta)
    layer_name = get_layer_name(args.model_name, int(hparams["layer"]))

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    model = tokenizer = model_path = None
    base_dataframe = None
    if not args.skip_mc:
        model, tokenizer, model_path = load_base_model(args, device)
        base_dataframe, _ = load_truthfulqa_mc_dataframe(args.csv_path)

    mc_rows: list[dict[str, Any]] = []
    mc_question_rows: list[dict[str, Any]] = []
    diagnostic_question_rows: list[dict[str, Any]] = []
    diagnostic_pair_rows: list[dict[str, Any]] = []
    split_audit: list[dict[str, Any]] = []

    print(
        "L2S reliability experiment | "
        f"model={args.model_name} layer={hparams['layer']} folds={args.folds} | "
        f"linear_alpha={hparams['linear_alpha']} | "
        f"spherical=(kappa={hparams['spherical_kappa']}, "
        f"alpha={hparams['spherical_alpha']}, beta={hparams['spherical_beta']})"
    )
    print(f"Feature pairs: {feature_path} | N={len(v_t)} dim={v_t.shape[1]}")

    for fold in args.folds:
        checkpoint_path = checkpoint_paths[fold]
        checkpoint, mlp = load_l2s_checkpoint(checkpoint_path, device=device)
        try:
            train_q, test_q = require_split(
                checkpoint,
                checkpoint_path,
                fold=fold,
                num_folds=args.num_folds,
            )
            if int(getattr(mlp, "input_dim", v_t.shape[1])) != int(v_t.shape[1]):
                raise ValueError(
                    f"fold {fold} MLP dim={mlp.input_dim} but feature dim={v_t.shape[1]}"
                )

            # If the checkpoint carries its own feature path, make sure all folds
            # resolve to the same raw pair artifact unless the user overrode it.
            if args.feature_path is None:
                fold_feature = resolve_feature_path(
                    checkpoint,
                    checkpoint_path,
                    None,
                )
                if fold_feature != feature_path:
                    raise ValueError(
                        f"fold {fold} resolves to a different feature file: "
                        f"{fold_feature} != {feature_path}"
                    )

            mean_vector = mean_vector_from_questions(v_t, q_indices, train_q)
            split_audit.append(
                {
                    "fold": int(fold),
                    "l2s_train_questions": int(len(train_q)),
                    "test_questions": int(len(test_q)),
                    "mean_vector_norm": float(np.linalg.norm(mean_vector)),
                }
            )

            if not args.skip_diagnostics:
                q_rows, pair_rows = diagnostic_rows_for_fold(
                    fold=fold,
                    mlp=mlp,
                    z_neg=z_neg,
                    v_t=v_t,
                    q_indices=q_indices,
                    train_q=train_q,
                    test_q=test_q,
                    mean_vector=mean_vector,
                    batch_size=args.diagnostic_batch_size,
                    save_pairs=args.save_pair_diagnostics,
                )
                diagnostic_question_rows.extend(q_rows)
                diagnostic_pair_rows.extend(pair_rows)

            if not args.skip_mc:
                assert model is not None
                assert tokenizer is not None
                assert base_dataframe is not None
                fold_df = base_dataframe[
                    base_dataframe["hf_idx"].isin(set(int(x) for x in test_q))
                ].copy()
                fold_df = fold_df.sort_values("hf_idx").reset_index(drop=True)
                if args.max_mc_questions is not None:
                    fold_df = fold_df.iloc[: args.max_mc_questions].copy()
                if len(fold_df) == 0:
                    raise ValueError(f"fold {fold} has no TruthfulQA MC test rows")

                eval_conditions = list(CONDITIONS)
                if not args.no_baseline:
                    eval_conditions = [
                        ("baseline", "None", "No steering"),
                        *eval_conditions,
                    ]

                for condition, vector_source, geometry in eval_conditions:
                    stats: dict[str, Any] = {"total": 0, "steered": 0}
                    hook = (
                        None
                        if condition == "baseline"
                        else make_condition_hook(
                            condition,
                            mean_vector=mean_vector,
                            mlp=mlp,
                            hparams=hparams,
                            device=device,
                            stats=stats,
                        )
                    )
                    metrics, question_rows = evaluate_mc_condition(
                        condition=condition,
                        vector_source=vector_source,
                        geometry=geometry,
                        fold=fold,
                        base_dataframe=fold_df,
                        model=model,
                        tokenizer=tokenizer,
                        hook=hook,
                        layer_name=layer_name,
                        device=device,
                        model_name=args.model_name,
                        no_instruction=args.no_instruction,
                        stats=stats,
                    )
                    mc_rows.append(metrics)
                    mc_question_rows.extend(question_rows)
                    print(
                        f"fold {fold} {condition}: "
                        f"MC1={metrics['MC1']:.4f} "
                        f"steer={metrics['steering_rate']:.3f}"
                    )
        finally:
            checkpoint.close()
            del mlp
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    if mc_rows:
        write_csv(
            output_dir / "mc1.csv",
            mc_rows,
            [
                "fold",
                "condition",
                "vector_source",
                "geometry",
                "questions",
                "MC1",
                "MC2",
                "MC3",
                "steering_rate",
                "steered_tokens",
                "total_tokens",
            ],
        )
        write_csv(
            output_dir / "mc1_question.csv",
            mc_question_rows,
            ["fold", "condition", "vector_source", "geometry", "hf_idx", "MC1"],
        )

    if diagnostic_question_rows:
        write_csv(
            output_dir / "diagnostics_question.csv",
            diagnostic_question_rows,
            [
                "fold",
                "split",
                "vector_source",
                "q_idx",
                "num_pairs",
                "cosine_similarity",
                "magnitude_relative_error",
            ],
        )
    if diagnostic_pair_rows:
        write_csv(
            output_dir / "diagnostics_pair.csv",
            diagnostic_pair_rows,
            [
                "fold",
                "split",
                "vector_source",
                "pair_index",
                "q_idx",
                "cosine_similarity",
                "magnitude_relative_error",
                "predicted_norm",
                "target_norm",
            ],
        )

    summary = {
        "experiment": "L2S vector reliability: Mean/L2S × Linear/Spherical",
        "model_name": args.model_name,
        "model_path": str(model_path) if model_path is not None else None,
        "feature_path": str(feature_path),
        "checkpoint_template": args.checkpoint_template,
        "folds": [int(x) for x in args.folds],
        "num_folds": int(args.num_folds),
        "parameters": {
            **hparams,
            "vector_target": "v_T = z_pos_raw - z_neg_raw",
            "mean_vector_training_questions": "l2s_train_q_indices",
            "mc_split": "test_q_indices",
            "magnitude_metric": "abs(||v_pred|| - ||v_T||) / ||v_T||",
            "diagnostic_unit": "question mean over Cartesian answer pairs",
            "zero_shot": True,
            "instruction": not args.no_instruction,
        },
        "split_audit": split_audit,
        "files": {
            "mc1": "mc1.csv" if mc_rows else None,
            "mc1_question": "mc1_question.csv" if mc_question_rows else None,
            "diagnostics_question": (
                "diagnostics_question.csv" if diagnostic_question_rows else None
            ),
            "diagnostics_pair": (
                "diagnostics_pair.csv" if diagnostic_pair_rows else None
            ),
        },
    }
    write_json(output_dir / "summary.json", summary)
    print(f"Saved experiment outputs to {output_dir}")


if __name__ == "__main__":
    main()
