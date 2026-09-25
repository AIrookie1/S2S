#!/usr/bin/env python3
"""Residual-direction analysis for S2S-R and S2S-L-Static.

This experiment implements the three residual analyses described in the paper
notes while reusing the released S2S-R / S2S-L-Static steering definitions:

A. Residual/oracle alignment with matched, within-category shuffled, globally shuffled, and cross-category
   cosine-nearest predicted residuals.
B. Residual-swap MC intervention with Mean only, Shuffled residual, Cross-category cosine-nearest residual,
   Predicted residual, and Oracle residual.  The Mean-only baseline follows the selected
   steering family: S2S-R uses the released cross-fitted Spherical Steering
   baseline, while S2S-L-Static uses standard linear additive mean steering
   ``h_new = h + strength * v_avg``.  For S2S-R, shuffled/oracle residuals
   recompute adaptive weights and the gate.  For S2S-L-Static, they use the
   method's fixed strength/slerp_t while changing only the residual direction;
   the recipient sample keeps its own ||v_hat||.
C. Residual quality versus geometric steering gain, using the selected method's
   released geometry (S2S-R rotation or S2S-L-Static additive steering).

The oracle pre-pass uses exactly the same evaluation prompt/instruction/layer
as MC scoring.  For each incorrect candidate a^- of question q, the truthful
target is the single truthful-answer hidden state with the highest cosine
similarity to h(q,a^-):

    z_pos_star(q,a^-) = argmax_{z in Z_pos(q)} cos(h(q,a^-), z)
    v_oracle(q,a^-) = z_pos_star(q,a^-) - h(q,a^-)
    r_oracle(q,a^-) = normalize(v_oracle(q,a^-) - v_avg)

This same per-negative nearest-positive target is used by Panels (a,c).  For
truthful candidates needed only by Oracle MC scoring, the previous truthful
centroid target is retained to avoid the degenerate self-match target.  Oracle
quantities are analysis-only.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from baukit import TraceDict
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.residual_analysis.metrics import (
    aggregate_mean_by_question,
    bootstrap_mean_ci,
    constrained_shuffle_indices,
    equal_count_bins,
    row_normalize,
)
from method.dyn3d_sga.dyn_mlp import predict_residual
from method.dyn3d_sga.evaluate_3d_dyn import (
    adaptive_3way_weights,
    norm,
    two_slerp_sym,
)
from method.dyn3d_sga.evaluate_dyn_mlp import load_components
from method.dyn2d.steering import slerp as dyn2d_slerp
from method.dyn3d_sga.eval_utils import (
    HF_NAMES,
    format_best,
    get_additive_intervention,
    get_instruction,
    get_layer_name,
    load_truthfulqa_mc_dataframe,
    resolve_device,
    set_seed,
    split_multi_answer,
)
from method.spherical_steering.steering import (
    get_spherical_intervention,
    spherical_geometric_logic,
)
from truthfulqa.models import MC_calcs, set_columns
from truthfulqa.presets import preset_map
from truthfulqa.utilities import format_prompt_with_answer_strings


CONDITION_ORDER = (
    "mean_only",
    "shuffled_residual",
    "cross_category_cosine_nearest_residual",
    "predicted_residual",
    "oracle_residual",
)
CONDITION_LABELS = {
    "mean_only": "Mean only",
    "shuffled_residual": "Shuffled residual",
    "cross_category_cosine_nearest_residual": "Cross-category cosine-nearest residual",
    "predicted_residual": "Predicted residual",
    "oracle_residual": "Oracle residual",
}


@dataclass(frozen=True)
class SphericalPrototype:
    fold_idx: int
    mu_T: np.ndarray
    mu_H: np.ndarray
    source_path: str


@dataclass
class SphericalPrototypeBank:
    by_question: dict[int, SphericalPrototype]
    source_paths: list[str]

    def for_question(self, q_idx: int) -> SphericalPrototype:
        try:
            return self.by_question[int(q_idx)]
        except KeyError as exc:
            raise KeyError(
                f"No held-out spherical-steering prototype covers question {q_idx}. "
                "Pass all cross-validation prototype folds via "
                "--spherical_prototype_paths."
            ) from exc


@dataclass
class CandidateRecord:
    candidate_id: str
    q_idx: int
    row_idx: int
    category: str
    question: str
    label: int
    answer_index: int
    answer: str
    input_ids: torch.Tensor
    start_idx: int
    hidden: np.ndarray
    positive_target: np.ndarray
    positive_target_kind: str
    positive_target_answer_index: int
    positive_target_cosine: float
    predicted_residual: np.ndarray
    predicted_vector_norm: float
    predicted_residual_raw_norm: float
    predicted_valid: bool
    oracle_residual: np.ndarray
    oracle_residual_raw_norm: float
    oracle_valid: bool


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Residual direction analysis on held-out TruthfulQA"
    )
    p.add_argument("model_name")
    p.add_argument("--checkpoint_path", required=True)
    p.add_argument("--model_dir", default=None)
    p.add_argument("--csv_path", default="./TruthfulQA/data/v1/TruthfulQA.csv")
    p.add_argument("--layer", type=int, default=14)
    p.add_argument("--eval_split", choices=["test", "train", "all"], default="test")
    p.add_argument("--expected_fold", type=int, default=None)
    p.add_argument(
        "--steering_method",
        choices=["S2S_R", "S2S_L_Static"],
        default="S2S_R",
        help="Residual-bearing steering method used for Predicted/Shuffled/Oracle.",
    )
    p.add_argument("--weight_kappa", type=float, default=3.0)
    p.add_argument("--margin", type=float, default=-0.7)
    p.add_argument(
        "--strength",
        type=float,
        default=3.0,
        help="S2S_L_Static additive steering strength.",
    )
    p.add_argument(
        "--slerp_t",
        type=float,
        default=0.2,
        help="S2S_L_Static SLERP coefficient: 0=mean, 1=residual.",
    )
    p.add_argument(
        "--spherical_prototype_paths",
        nargs="*",
        default=None,
        help=(
            "Cross-validation prototype files for the S2S-R spherical Mean-only "
            "baseline. Required only when --steering_method=S2S_R; pass every fold "
            "so each evaluated question can use the fold where it is held out."
        ),
    )
    p.add_argument(
        "--spherical_layer",
        type=int,
        default=None,
        help="Spherical-steering layer. Defaults to --layer.",
    )
    p.add_argument("--spherical_kappa", type=float, default=20.0)
    p.add_argument("--spherical_alpha", type=float, default=0.7)
    p.add_argument("--spherical_beta", type=float, default=-0.15)
    p.add_argument("--few_shot", action="store_true")
    p.add_argument("--preset", default="qa")
    p.add_argument("--no_instruction", action="store_true")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", choices=["cuda", "cpu"], default="cuda")
    p.add_argument(
        "--dtype",
        choices=["float16", "bfloat16", "float32"],
        default="float16",
    )
    p.add_argument("--slerp_eps", type=float, default=1e-8)
    p.add_argument("--small_angle_threshold", type=float, default=1e-4)
    p.add_argument(
        "--num_shuffle_seeds",
        type=int,
        default=100,
        help="Offline matched-vs-shuffled residual seeds for Panel (a).",
    )
    p.add_argument(
        "--mc_shuffle_seeds",
        type=int,
        default=5,
        help=(
            "Expensive full-MC shuffled-residual seeds for Panel (b). "
            "Increase to 100 for the strictest final run if compute permits."
        ),
    )
    p.add_argument("--bootstrap_samples", type=int, default=1000)
    p.add_argument("--quintiles", type=int, default=5)
    p.add_argument("--output_dir", default="./results/experiment/residual_analysis")
    p.add_argument(
        "--skip_mc",
        action="store_true",
        help="Run only the inexpensive geometry/alignment analyses.",
    )
    p.add_argument(
        "--max_questions",
        type=int,
        default=None,
        help="Debug/smoke-test only: truncate the selected question split.",
    )
    p.add_argument(
        "--save_mc_scores",
        action="store_true",
        help="Save per-answer MC log-probabilities in mc_scores.jsonl.",
    )
    args = p.parse_args()

    if not math.isfinite(args.weight_kappa) or args.weight_kappa <= 0:
        p.error("--weight_kappa must be finite and positive")
    if not math.isfinite(args.margin):
        p.error("--margin must be finite")
    if not math.isfinite(args.strength):
        p.error("--strength must be finite")
    if not math.isfinite(args.slerp_t) or not 0.0 <= args.slerp_t <= 1.0:
        p.error("--slerp_t must be finite and in [0, 1]")
    if args.spherical_layer is None:
        args.spherical_layer = args.layer
    if args.steering_method == "S2S_R":
        if not args.spherical_prototype_paths:
            p.error(
                "--spherical_prototype_paths is required for S2S_R because its "
                "Mean-only baseline is cross-fitted Spherical Steering"
            )
        if args.spherical_layer != args.layer:
            p.error(
                "--spherical_layer must equal --layer for Panel (c), because angular gain "
                "must compare directions in the same hidden-state space"
            )
        if not math.isfinite(args.spherical_kappa) or args.spherical_kappa <= 0:
            p.error("--spherical_kappa must be finite and positive")
        if not math.isfinite(args.spherical_alpha) or not 0.0 <= args.spherical_alpha <= 1.0:
            p.error("--spherical_alpha must be finite and in [0, 1]")
        if not math.isfinite(args.spherical_beta) or not -1.0 <= args.spherical_beta < 1.0:
            p.error("--spherical_beta must be finite and in [-1, 1)")
    if args.num_shuffle_seeds <= 0:
        p.error("--num_shuffle_seeds must be positive")
    if args.mc_shuffle_seeds <= 0:
        p.error("--mc_shuffle_seeds must be positive")
    if args.bootstrap_samples <= 0:
        p.error("--bootstrap_samples must be positive")
    if args.quintiles <= 1:
        p.error("--quintiles must be at least 2")
    if args.max_questions is not None and args.max_questions <= 0:
        p.error("--max_questions must be positive")
    return args


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        raise ValueError(f"Refusing to write empty CSV: {path}")
    fieldnames: list[str] = []
    seen: set[str] = set()
    for row in rows:
        for key in row:
            if key not in seen:
                fieldnames.append(key)
                seen.add(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _candidate_id(q_idx: int, label: int, answer_index: int) -> str:
    kind = "T" if int(label) == 1 else "F"
    return f"q{int(q_idx)}:{kind}{int(answer_index)}"


def _build_choice_input(
    tokenizer,
    *,
    question: str,
    choice: str,
    few_shot: bool,
    preset: str,
    use_instruction: bool,
    model_name: str,
) -> tuple[torch.Tensor, int]:
    """Tokenize exactly like ``eval_utils.calculate_scores_*``."""
    if not few_shot:
        base = f"Q: {question} A:"
        if use_instruction:
            base = get_instruction(model_name) + "\n\n" + base
        prompt_ids = tokenizer(base, return_tensors="pt").input_ids
        start_idx = int(prompt_ids.shape[1] - 1)
        choice_ids = tokenizer(
            f" {choice}",
            add_special_tokens=False,
            return_tensors="pt",
        ).input_ids
        return torch.cat([prompt_ids, choice_ids], dim=1).cpu(), start_idx

    if preset == "qa":
        base_prompt = preset_map[preset] + "\n\nQ: " + question
    else:
        base_prompt = "Q: " + question
    if use_instruction:
        base_prompt = get_instruction(model_name) + "\n\n" + base_prompt
    base_ids = tokenizer(base_prompt, return_tensors="pt").input_ids
    base_len = int(base_ids.shape[-1])

    prompt = format_prompt_with_answer_strings(
        question,
        choice,
        preset,
        format="general",
    )
    if use_instruction:
        prompt = get_instruction(model_name) + "\n\n" + prompt
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids
    return input_ids.cpu(), base_len


class _LayerLastTokenCapture:
    def __init__(self, model, layer_name: str):
        self.last: torch.Tensor | None = None
        module = model.get_submodule(layer_name)
        self.handle = module.register_forward_hook(self._hook)

    def _hook(self, _module, _inputs, output):
        hidden = output[0] if isinstance(output, tuple) else output
        if not torch.is_tensor(hidden) or hidden.ndim != 3:
            raise RuntimeError("Unexpected transformer-layer output while capturing hidden state")
        self.last = hidden[:, -1, :].detach().float().cpu()

    def close(self) -> None:
        self.handle.remove()


def _extract_hidden_for_input(
    model,
    capture: _LayerLastTokenCapture,
    input_ids_cpu: torch.Tensor,
    device: torch.device,
) -> np.ndarray:
    capture.last = None
    with torch.no_grad():
        _ = model(input_ids_cpu.to(device))
    if capture.last is None:
        raise RuntimeError("Layer hook did not capture a hidden state")
    if capture.last.shape[0] != 1:
        raise RuntimeError(f"Expected batch size 1, got {tuple(capture.last.shape)}")
    return capture.last[0].numpy().astype(np.float32, copy=True)


def _normalize_vector_np(x: np.ndarray, eps: float) -> tuple[np.ndarray, float, bool]:
    x = np.asarray(x, dtype=np.float32).reshape(-1)
    n = float(np.linalg.norm(x))
    if not np.isfinite(n):
        raise ValueError("vector norm is non-finite")
    if n <= eps:
        return np.zeros_like(x), n, False
    return (x / n).astype(np.float32), n, True


def _nearest_positive_target(
    hidden: np.ndarray,
    positive_hidden: np.ndarray,
    eps: float,
) -> tuple[np.ndarray, int, float]:
    """Return the truthful hidden with maximum cosine similarity to ``hidden``."""
    h = np.asarray(hidden, dtype=np.float32).reshape(-1)
    positives = np.asarray(positive_hidden, dtype=np.float32)
    if positives.ndim != 2 or positives.shape[0] == 0:
        raise ValueError("positive_hidden must contain at least one positive vector")
    h_norm = float(np.linalg.norm(h))
    p_norm = np.linalg.norm(positives, axis=1)
    if h_norm <= eps or not np.isfinite(h_norm):
        raise ValueError("candidate hidden is near-zero or non-finite")
    valid = np.isfinite(p_norm) & (p_norm > eps)
    if not np.any(valid):
        raise ValueError("all truthful hidden vectors are near-zero or non-finite")
    cosine = np.full(positives.shape[0], -np.inf, dtype=np.float64)
    cosine[valid] = (positives[valid] @ h.astype(np.float64)) / (
        p_norm[valid].astype(np.float64) * h_norm
    )
    best = int(np.argmax(cosine))
    return positives[best].copy(), best, float(np.clip(cosine[best], -1.0, 1.0))


def load_spherical_prototype_bank(
    paths: list[str],
    *,
    expected_dim: int,
) -> SphericalPrototypeBank:
    """Load all spherical-steering CV folds and map held-out questions to folds.

    This is deliberately cross-fitted: question ``q`` is evaluated with the
    prototype file whose ``test_q_indices`` contains ``q``.  That preserves the
    original spherical-steering train/test discipline while letting the
    residual experiment retain the full S2S-R held-out question set.
    """
    if not paths:
        raise ValueError("At least one spherical prototype path is required")

    by_question: dict[int, SphericalPrototype] = {}
    source_paths: list[str] = []
    for raw_path in paths:
        path = Path(raw_path)
        if not path.is_file():
            raise FileNotFoundError(f"Missing spherical prototype: {path}")
        with np.load(path, allow_pickle=False) as data:
            required = {"mu_T", "mu_H", "test_q_indices"}
            missing = required.difference(data.files)
            if missing:
                raise KeyError(
                    f"Spherical prototype {path} missing fields: {sorted(missing)}"
                )
            mu_t = np.asarray(data["mu_T"], dtype=np.float32).reshape(-1)
            mu_h = np.asarray(data["mu_H"], dtype=np.float32).reshape(-1)
            if mu_t.shape != (expected_dim,) or mu_h.shape != (expected_dim,):
                raise ValueError(
                    f"Spherical prototype dim mismatch in {path}: "
                    f"mu_T={mu_t.shape}, mu_H={mu_h.shape}, expected {(expected_dim,)}"
                )
            mu_t, _, valid_t = _normalize_vector_np(mu_t, 1e-12)
            mu_h, _, valid_h = _normalize_vector_np(mu_h, 1e-12)
            if not valid_t or not valid_h:
                raise ValueError(f"Near-zero spherical prototype in {path}")
            fold_idx = (
                int(np.asarray(data["fold_idx"]).item())
                if "fold_idx" in data.files
                else len(source_paths)
            )
            test_q = np.asarray(data["test_q_indices"], dtype=np.int64).reshape(-1)

        proto = SphericalPrototype(
            fold_idx=fold_idx,
            mu_T=mu_t,
            mu_H=mu_h,
            source_path=str(path),
        )
        for q_idx in test_q:
            q_int = int(q_idx)
            if q_int in by_question:
                previous = by_question[q_int]
                raise ValueError(
                    f"Question {q_int} appears in multiple spherical test folds: "
                    f"{previous.source_path} and {path}"
                )
            by_question[q_int] = proto
        source_paths.append(str(path))

    return SphericalPrototypeBank(
        by_question=by_question,
        source_paths=source_paths,
    )


def validate_spherical_coverage(
    bank: SphericalPrototypeBank,
    q_indices: set[int],
) -> None:
    missing = sorted(int(q) for q in q_indices if int(q) not in bank.by_question)
    if missing:
        preview = ", ".join(str(q) for q in missing[:12])
        suffix = "..." if len(missing) > 12 else ""
        raise ValueError(
            f"Spherical prototype folds do not cover {len(missing)} evaluated questions "
            f"({preview}{suffix}). Pass every spherical-steering CV fold so each "
            "question is scored only with a prototype that held it out."
        )


def collect_candidate_records(
    *,
    results: pd.DataFrame,
    model,
    tokenizer,
    layer_name: str,
    device: torch.device,
    avg: torch.Tensor,
    predictor,
    args: argparse.Namespace,
) -> list[CandidateRecord]:
    """Unsteered pre-pass used by all three residual panels."""
    records: list[CandidateRecord] = []
    capture = _LayerLastTokenCapture(model, layer_name)
    avg_np = avg.detach().float().cpu().numpy().reshape(-1)

    try:
        progress = tqdm(range(len(results)), desc="oracle pre-pass", unit="q")
        for row_idx in progress:
            row = results.iloc[row_idx]
            q_idx = int(row["hf_idx"])
            question = str(row["Question"])
            category = str(row.get("Category", "Unknown"))
            true_answers = split_multi_answer(row["Correct Answers"])
            false_answers = split_multi_answer(row["Incorrect Answers"])
            answer_specs = [
                (1, answer_idx, answer)
                for answer_idx, answer in enumerate(true_answers)
            ] + [
                (0, answer_idx, answer)
                for answer_idx, answer in enumerate(false_answers)
            ]

            temporary: list[dict[str, Any]] = []
            for label, answer_idx, answer in answer_specs:
                input_ids, start_idx = _build_choice_input(
                    tokenizer,
                    question=question,
                    choice=answer,
                    few_shot=args.few_shot,
                    preset=args.preset,
                    use_instruction=not args.no_instruction,
                    model_name=args.model_name,
                )
                hidden = _extract_hidden_for_input(
                    model,
                    capture,
                    input_ids,
                    device,
                )
                temporary.append(
                    {
                        "label": label,
                        "answer_idx": answer_idx,
                        "answer": answer,
                        "input_ids": input_ids,
                        "start_idx": start_idx,
                        "hidden": hidden,
                    }
                )

            positive_items = [item for item in temporary if item["label"] == 1]
            positive_hidden = np.stack(
                [item["hidden"] for item in positive_items],
                axis=0,
            )
            if len(positive_hidden) == 0:
                raise ValueError(f"Question {q_idx} has no truthful answer")
            positive_centroid = positive_hidden.mean(axis=0, dtype=np.float64).astype(np.float32)

            all_hidden = np.stack([item["hidden"] for item in temporary], axis=0)
            with torch.no_grad():
                h_tensor = torch.as_tensor(all_hidden, dtype=torch.float32, device=avg.device)
                v_hat, predicted_residual = predict_residual(
                    predictor,
                    h_tensor,
                    avg,
                    eps=args.slerp_eps,
                )
            predicted_np = predicted_residual.detach().float().cpu().numpy().astype(np.float32)
            predicted_vector_norm = (
                torch.linalg.vector_norm(v_hat, dim=-1)
                .detach()
                .float()
                .cpu()
                .numpy()
            )
            predicted_raw_norm = (
                torch.linalg.vector_norm(v_hat - avg.view(1, -1), dim=-1)
                .detach()
                .float()
                .cpu()
                .numpy()
            )

            for item, r_pred, v_pred_norm, r_pred_raw_norm in zip(
                temporary,
                predicted_np,
                predicted_vector_norm,
                predicted_raw_norm,
            ):
                if int(item["label"]) == 0:
                    positive_target, matched_pos_idx, matched_pos_cosine = _nearest_positive_target(
                        item["hidden"],
                        positive_hidden,
                        args.slerp_eps,
                    )
                    positive_target_kind = "nearest_truthful_answer"
                    positive_target_answer_index = int(
                        positive_items[matched_pos_idx]["answer_idx"]
                    )
                else:
                    # Oracle MC also scores truthful candidates.  Using the candidate
                    # itself as its nearest truthful target would make v_oracle=0 and
                    # create a degenerate residual definition, so preserve the prior
                    # truthful-centroid target for these rows.  Panels (a,c) only use
                    # incorrect candidates and therefore always use nearest-positive.
                    positive_target = positive_centroid
                    positive_target_kind = "truthful_centroid_for_oracle_mc"
                    positive_target_answer_index = -1
                    matched_pos_cosine = float("nan")

                v_oracle = positive_target - item["hidden"]
                r_oracle_raw = v_oracle - avg_np
                r_oracle, raw_norm, oracle_valid = _normalize_vector_np(
                    r_oracle_raw,
                    args.slerp_eps,
                )
                records.append(
                    CandidateRecord(
                        candidate_id=_candidate_id(q_idx, item["label"], item["answer_idx"]),
                        q_idx=q_idx,
                        row_idx=int(row_idx),
                        category=category,
                        question=question,
                        label=int(item["label"]),
                        answer_index=int(item["answer_idx"]),
                        answer=str(item["answer"]),
                        input_ids=item["input_ids"],
                        start_idx=int(item["start_idx"]),
                        hidden=np.asarray(item["hidden"], dtype=np.float32),
                        positive_target=np.asarray(positive_target, dtype=np.float32),
                        positive_target_kind=positive_target_kind,
                        positive_target_answer_index=positive_target_answer_index,
                        positive_target_cosine=float(matched_pos_cosine),
                        predicted_residual=np.asarray(r_pred, dtype=np.float32),
                        predicted_vector_norm=float(v_pred_norm),
                        predicted_residual_raw_norm=float(r_pred_raw_norm),
                        predicted_valid=bool(float(r_pred_raw_norm) > args.slerp_eps),
                        oracle_residual=np.asarray(r_oracle, dtype=np.float32),
                        oracle_residual_raw_norm=float(raw_norm),
                        oracle_valid=bool(oracle_valid),
                    )
                )
    finally:
        capture.close()

    return records


def _angle_degrees_torch(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    dot = (x * y).sum(dim=-1).clamp(-1.0, 1.0)
    return torch.rad2deg(torch.acos(dot))


def compute_negative_geometry(
    records: list[CandidateRecord],
    *,
    avg: torch.Tensor,
    steering_method: str,
    strength: float,
    slerp_t: float,
    spherical_bank: SphericalPrototypeBank | None,
    spherical_kappa: float,
    spherical_alpha: float,
    spherical_beta: float,
    weight_kappa: float,
    margin: float,
    eps: float,
    small_angle_threshold: float,
) -> list[dict[str, Any]]:
    """Compute Panels (a,c) on held-out incorrect answers only."""
    negative = [record for record in records if record.label == 0]
    if not negative:
        raise ValueError("No incorrect-answer candidates were collected")

    hidden = np.stack([record.hidden for record in negative], axis=0)
    r_pred = np.stack([record.predicted_residual for record in negative], axis=0)
    r_oracle = np.stack([record.oracle_residual for record in negative], axis=0)
    pos = np.stack([record.positive_target for record in negative], axis=0)

    a_np, a_valid = row_normalize(hidden, eps)
    pos_np, pos_valid = row_normalize(pos, eps)
    if not a_valid.all() or not pos_valid.all():
        raise ValueError("Encountered a near-zero hidden or selected positive-target vector")

    b0 = norm(avg.detach().float().cpu(), eps)
    if steering_method == "S2S_R":
        a = torch.as_tensor(a_np, dtype=torch.float32).unsqueeze(1)
        c = torch.as_tensor(r_pred, dtype=torch.float32).unsqueeze(1)
        b = b0.view(1, 1, -1).expand_as(a)
        alpha, beta, gamma, _ = adaptive_3way_weights(
            a,
            b,
            c,
            kappa=weight_kappa,
            temperature=1.0,
        )
        matched_rotated = two_slerp_sym(
            a,
            b,
            c,
            alpha,
            beta,
            gamma,
            eps,
            small_angle_threshold,
        )
        should_steer = torch.maximum(beta, gamma) > (alpha + margin)
        matched_final = torch.where(should_steer, matched_rotated, a)
        alpha_values = alpha[:, 0, 0]
        beta_values = beta[:, 0, 0]
        gamma_values = gamma[:, 0, 0]
        gate_values = should_steer[:, 0, 0]
    elif steering_method == "S2S_L_Static":
        hidden_tensor = torch.as_tensor(hidden, dtype=torch.float32)
        c = torch.as_tensor(r_pred, dtype=torch.float32)
        b = b0.view(1, -1).expand_as(c)
        direction = dyn2d_slerp(
            b,
            c,
            slerp_t,
            eps=eps,
            small_angle_threshold=small_angle_threshold,
        )
        predicted_norm = torch.as_tensor(
            [record.predicted_vector_norm for record in negative],
            dtype=torch.float32,
        ).view(-1, 1)
        edited_hidden = hidden_tensor + float(strength) * direction * predicted_norm
        matched_final = norm(edited_hidden, eps).unsqueeze(1)
        nan_values = torch.full((len(negative),), float("nan"), dtype=torch.float32)
        alpha_values = nan_values
        beta_values = nan_values.clone()
        gamma_values = nan_values.clone()
        # S2S-L-Static has no margin gate.  Keep the CSV gate field explicitly
        # non-applicable rather than pretending that a gate fired.
        gate_values = None
    else:
        raise ValueError(f"Unsupported steering_method: {steering_method}")

    # Mean-only follows the selected steering family.  S2S-R uses the released
    # cross-fitted Spherical Steering baseline.  S2S-L-Static uses standard
    # linear additive mean steering with the same Dyn2D strength:
    #     h_mean = h + strength * v_avg
    spherical_triggered: list[bool] = []
    spherical_folds: list[int | None] = []
    if steering_method == "S2S_R":
        if spherical_bank is None:
            raise ValueError("S2S_R geometry requires a spherical prototype bank")
        spherical_final_rows: list[torch.Tensor] = []
        for record in negative:
            proto = spherical_bank.for_question(record.q_idx)
            spherical_vec, triggered = spherical_geometric_logic(
                torch.as_tensor(record.hidden, dtype=torch.float32),
                torch.as_tensor(proto.mu_T, dtype=torch.float32),
                torch.as_tensor(proto.mu_H, dtype=torch.float32),
                spherical_kappa,
                spherical_alpha,
                spherical_beta,
            )
            spherical_final_rows.append(norm(spherical_vec, eps))
            spherical_triggered.append(bool(triggered))
            spherical_folds.append(int(proto.fold_idx))
        mean_final = torch.stack(spherical_final_rows, dim=0).unsqueeze(1)
        mean_baseline_kind = "spherical_steering"
    else:
        hidden_tensor = torch.as_tensor(hidden, dtype=torch.float32)
        avg_row = avg.detach().float().cpu().view(1, -1)
        linear_mean_hidden = hidden_tensor + float(strength) * avg_row
        mean_final = norm(linear_mean_hidden, eps).unsqueeze(1)
        spherical_triggered = [False] * len(negative)
        spherical_folds = [None] * len(negative)
        mean_baseline_kind = "linear_additive_v_avg"
    target = torch.as_tensor(pos_np, dtype=torch.float32).unsqueeze(1)

    theta_pred = _angle_degrees_torch(matched_final[:, 0, :], target[:, 0, :])
    theta_mean = _angle_degrees_torch(mean_final[:, 0, :], target[:, 0, :])
    gain = theta_mean - theta_pred

    rows: list[dict[str, Any]] = []
    for idx, record in enumerate(negative):
        alignment_valid = bool(record.predicted_valid and record.oracle_valid)
        matched_alignment = (
            float(np.clip(np.dot(r_pred[idx], r_oracle[idx]), -1.0, 1.0))
            if alignment_valid
            else float("nan")
        )
        rows.append(
            {
                "candidate_id": record.candidate_id,
                "q_idx": record.q_idx,
                "row_idx": record.row_idx,
                "category": record.category,
                "question": record.question,
                "answer_index": record.answer_index,
                "positive_target_kind": record.positive_target_kind,
                "positive_target_answer_index": record.positive_target_answer_index,
                "positive_target_cosine": record.positive_target_cosine,
                "matched_alignment": matched_alignment,
                "predicted_residual_raw_norm": record.predicted_residual_raw_norm,
                "predicted_vector_norm": record.predicted_vector_norm,
                "predicted_valid": int(record.predicted_valid),
                "oracle_residual_raw_norm": record.oracle_residual_raw_norm,
                "oracle_valid": int(record.oracle_valid),
                "alignment_valid": int(alignment_valid),
                "steering_method": steering_method,
                "alpha": float(alpha_values[idx]),
                "beta": float(beta_values[idx]),
                "gamma": float(gamma_values[idx]),
                "gate": (
                    int(bool(gate_values[idx]))
                    if gate_values is not None
                    else ""
                ),
                "strength": float(strength) if steering_method == "S2S_L_Static" else float("nan"),
                "slerp_t": float(slerp_t) if steering_method == "S2S_L_Static" else float("nan"),
                "mean_baseline_kind": mean_baseline_kind,
                "spherical_mean_gate": (
                    int(spherical_triggered[idx]) if steering_method == "S2S_R" else ""
                ),
                "spherical_mean_fold": (
                    int(spherical_folds[idx])
                    if steering_method == "S2S_R" and spherical_folds[idx] is not None
                    else ""
                ),
                "linear_mean_strength": (
                    float(strength) if steering_method == "S2S_L_Static" else float("nan")
                ),
                "theta_mean_deg": float(theta_mean[idx]),
                "theta_predicted_deg": float(theta_pred[idx]),
                "angular_gain_deg": float(gain[idx]),
            }
        )
    return rows


def cross_category_cosine_nearest_indices(
    residuals: np.ndarray,
    categories: np.ndarray,
    question_ids: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Return the most cosine-similar donor from a different category.

    Residual rows are expected to be unit-normalized.  A donor must have a
    different category and, when ``question_ids`` is supplied, a different
    question.  Ties are resolved by the smallest row index, making this control
    deterministic and independent of shuffle seeds.
    """
    residuals = np.asarray(residuals, dtype=np.float64)
    categories = np.asarray(categories, dtype=object)
    if residuals.ndim != 2 or len(residuals) != len(categories):
        raise ValueError("residuals/categories shape mismatch")
    q = None if question_ids is None else np.asarray(question_ids)
    if q is not None and len(q) != len(residuals):
        raise ValueError("question_ids length mismatch")

    norms = np.linalg.norm(residuals, axis=1, keepdims=True)
    unit = residuals / np.maximum(norms, 1e-12)
    cosine = np.clip(unit @ unit.T, -1.0, 1.0)
    donors = np.full(len(residuals), -1, dtype=np.int64)
    best_cos = np.full(len(residuals), np.nan, dtype=np.float64)
    for i in range(len(residuals)):
        eligible = categories != categories[i]
        if q is not None:
            eligible &= q != q[i]
        eligible[i] = False
        candidates = np.flatnonzero(eligible)
        if len(candidates):
            local = int(np.argmax(cosine[i, candidates]))
            donors[i] = int(candidates[local])
            best_cos[i] = float(cosine[i, donors[i]])
    return donors, best_cos


def compute_alignment_controls(
    geometry_rows: list[dict[str, Any]],
    records: list[CandidateRecord],
    *,
    num_seeds: int,
    seed: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    """Create Panel (a) question rows plus shuffle-null diagnostics."""
    record_by_id = {record.candidate_id: record for record in records}
    valid_geometry_rows = [row for row in geometry_rows if int(row["alignment_valid"]) == 1]
    if not valid_geometry_rows:
        raise ValueError("No negative candidates have valid predicted and oracle residual directions")
    negative_records = [record_by_id[row["candidate_id"]] for row in valid_geometry_rows]
    q = np.asarray([record.q_idx for record in negative_records], dtype=np.int64)
    categories = np.asarray([record.category for record in negative_records], dtype=object)
    pred = np.stack([record.predicted_residual for record in negative_records], axis=0)
    oracle = np.stack([record.oracle_residual for record in negative_records], axis=0)
    valid = np.ones(len(negative_records), dtype=bool)
    matched = np.asarray([row["matched_alignment"] for row in valid_geometry_rows], dtype=float)

    matched_q, matched_q_values = aggregate_mean_by_question(q, matched)
    matched_map = dict(zip(matched_q.tolist(), matched_q_values.tolist()))
    observed_mean = float(np.mean(matched_q_values))

    within_per_q: dict[int, list[float]] = {int(qi): [] for qi in matched_q}
    global_per_q: dict[int, list[float]] = {int(qi): [] for qi in matched_q}

    cross_donor, cross_donor_cos = cross_category_cosine_nearest_indices(
        pred, categories, question_ids=q
    )
    cross_eligible = cross_donor >= 0
    cross_alignment = np.full(len(q), np.nan, dtype=np.float64)
    cross_alignment[cross_eligible] = np.sum(
        pred[cross_donor[cross_eligible]] * oracle[cross_eligible], axis=1
    )
    cross_alignment[cross_eligible] = np.clip(
        cross_alignment[cross_eligible], -1.0, 1.0
    )
    cross_q, cross_q_values = aggregate_mean_by_question(
        q[cross_eligible], cross_alignment[cross_eligible]
    )
    cross_map = dict(zip(cross_q.tolist(), cross_q_values.tolist()))

    null_rows: list[dict[str, Any]] = []
    exact_counts = {"within_category": 0, "global": 0}
    eligible_counts = {"within_category": 0, "global": 0}

    for shuffle_idx in range(num_seeds):
        rng_within = np.random.default_rng(seed + 100003 * (shuffle_idx + 1))
        rng_global = np.random.default_rng(seed + 200003 * (shuffle_idx + 1))
        within_donor, within_exact = constrained_shuffle_indices(
            q,
            groups=categories,
            rng=rng_within,
        )
        global_donor, global_exact = constrained_shuffle_indices(q, rng=rng_global)

        for name, donor, exact in (
            ("within_category", within_donor, within_exact),
            ("global", global_donor, global_exact),
        ):
            eligible = valid & (donor >= 0)
            shuffled = np.full(len(q), np.nan, dtype=np.float64)
            shuffled[eligible] = np.sum(pred[donor[eligible]] * oracle[eligible], axis=1)
            shuffled[eligible] = np.clip(shuffled[eligible], -1.0, 1.0)
            q_ids, q_values = aggregate_mean_by_question(q[eligible], shuffled[eligible])
            q_map = dict(zip(q_ids.tolist(), q_values.tolist()))
            destination = within_per_q if name == "within_category" else global_per_q
            for qi in matched_q:
                if int(qi) in q_map:
                    destination[int(qi)].append(float(q_map[int(qi)]))

            mean_value = float(np.mean(q_values)) if len(q_values) else float("nan")
            matched_same_questions = (
                float(np.mean([matched_map[int(qi)] for qi in q_ids]))
                if len(q_ids)
                else float("nan")
            )
            null_rows.append(
                {
                    "shuffle_seed_index": shuffle_idx,
                    "condition": name,
                    "matched_mean_same_questions": matched_same_questions,
                    "mean_question_alignment": mean_value,
                    "delta_from_matched": matched_same_questions - mean_value,
                    "num_questions": int(len(q_values)),
                    "exact_permutation_fraction": float(np.mean(exact[eligible]))
                    if np.any(eligible)
                    else float("nan"),
                }
            )
            exact_counts[name] += int(np.sum(exact[eligible]))
            eligible_counts[name] += int(np.sum(eligible))

    question_rows: list[dict[str, Any]] = []
    for qi in matched_q:
        within_values = np.asarray(within_per_q[int(qi)], dtype=float)
        global_values = np.asarray(global_per_q[int(qi)], dtype=float)
        question_rows.append(
            {
                "q_idx": int(qi),
                "matched": float(matched_map[int(qi)]),
                "within_category_shuffled": float(np.mean(within_values))
                if len(within_values)
                else float("nan"),
                "within_category_shuffle_sd": float(np.std(within_values, ddof=1))
                if len(within_values) > 1
                else 0.0,
                "global_shuffled": float(np.mean(global_values))
                if len(global_values)
                else float("nan"),
                "global_shuffle_sd": float(np.std(global_values, ddof=1))
                if len(global_values) > 1
                else 0.0,
                "cross_category_cosine_nearest": float(cross_map.get(int(qi), float("nan"))),
            }
        )

    within_delta = np.asarray(
        [row["delta_from_matched"] for row in null_rows if row["condition"] == "within_category"],
        dtype=float,
    )
    global_delta = np.asarray(
        [row["delta_from_matched"] for row in null_rows if row["condition"] == "global"],
        dtype=float,
    )
    within_delta = within_delta[np.isfinite(within_delta)]
    global_delta = global_delta[np.isfinite(global_delta)]
    stats = {
        "observed_matched_question_mean": observed_mean,
        "within_category_shuffle_p_one_sided": (
            float((1 + np.sum(within_delta <= 0.0)) / (len(within_delta) + 1))
            if len(within_delta)
            else float("nan")
        ),
        "global_shuffle_p_one_sided": (
            float((1 + np.sum(global_delta <= 0.0)) / (len(global_delta) + 1))
            if len(global_delta)
            else float("nan")
        ),
        "within_category_exact_permutation_fraction": exact_counts["within_category"]
        / max(eligible_counts["within_category"], 1),
        "global_exact_permutation_fraction": exact_counts["global"]
        / max(eligible_counts["global"], 1),
        "cross_category_cosine_nearest_eligible_fraction": float(np.mean(cross_eligible)),
        "cross_category_cosine_nearest_donor_cosine_mean": (
            float(np.nanmean(cross_donor_cos)) if np.any(cross_eligible) else float("nan")
        ),
    }
    return question_rows, null_rows, stats


def summarize_alignment(
    alignment_question_rows: list[dict[str, Any]],
    *,
    n_bootstrap: int,
    seed: int,
) -> list[dict[str, Any]]:
    output = []
    for offset, (column, label) in enumerate(
        (
            ("matched", "Matched"),
            ("within_category_shuffled", "Category-shuffled"),
            ("global_shuffled", "Global-shuffled"),
            ("cross_category_cosine_nearest", "Cross-category cosine-nearest"),
        )
    ):
        values = np.asarray([float(row[column]) for row in alignment_question_rows], dtype=float)
        finite = values[np.isfinite(values)]
        if len(finite):
            mean, lo, hi = bootstrap_mean_ci(
                finite,
                n_bootstrap=n_bootstrap,
                seed=seed + offset,
            )
        else:
            mean = lo = hi = float("nan")
        output.append(
            {
                "condition": column,
                "label": label,
                "mean": mean,
                "ci_low": lo,
                "ci_high": hi,
                "num_questions": int(len(finite)),
            }
        )
    return output


def summarize_quintiles(
    geometry_rows: list[dict[str, Any]],
    *,
    n_quintiles: int,
    n_bootstrap: int,
    seed: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    valid_rows = [row for row in geometry_rows if int(row["alignment_valid"]) == 1]
    if len(valid_rows) < n_quintiles:
        raise ValueError("Too few valid oracle residuals for the requested quintiles")
    align = np.asarray([float(row["matched_alignment"]) for row in valid_rows])
    bins = equal_count_bins(align, n_quintiles)

    quintile_by_candidate = {
        str(row["candidate_id"]): int(bin_idx)
        for row, bin_idx in zip(valid_rows, bins)
    }
    pair_rows: list[dict[str, Any]] = []
    for row in geometry_rows:
        copied = dict(row)
        copied["alignment_quintile"] = quintile_by_candidate.get(str(row["candidate_id"]), "")
        pair_rows.append(copied)

    valid_pair_rows = [row for row in pair_rows if row["alignment_quintile"] != ""]
    summary_rows: list[dict[str, Any]] = []
    for bin_idx in range(1, int(np.max(bins)) + 1):
        selected = [row for row in valid_pair_rows if int(row["alignment_quintile"]) == bin_idx]
        x_mean = float(np.mean([float(row["matched_alignment"]) for row in selected]))

        # Statistical unit = question.  A question with several incorrect
        # answers in one quintile contributes one averaged gain to the bootstrap.
        by_q: dict[int, list[float]] = {}
        for row in selected:
            by_q.setdefault(int(row["q_idx"]), []).append(float(row["angular_gain_deg"]))
        q_gain = [float(np.mean(values)) for values in by_q.values()]
        mean, lo, hi = bootstrap_mean_ci(
            q_gain,
            n_bootstrap=n_bootstrap,
            seed=seed + 100 + bin_idx,
        )
        summary_rows.append(
            {
                "quintile": int(bin_idx),
                "mean_alignment": x_mean,
                "mean_angular_gain_deg": mean,
                "ci_low": lo,
                "ci_high": hi,
                "num_pairs": len(selected),
                "num_questions": len(by_q),
            }
        )
    return pair_rows, summary_rows


def get_dynamic_residual_intervention(
    *,
    avg: torch.Tensor,
    predictor,
    condition: str,
    residual_override: np.ndarray | None,
    oracle_override_valid: bool = True,
    weight_kappa: float,
    margin: float,
    eps: float,
    small_angle_threshold: float,
    stats: dict[str, float] | None = None,
):
    """Build an S2S-R hook whose adaptive weights follow the active residual.

    Predicted uses the normal S2S-R residual.  Shuffled and Oracle replace that
    residual *before* ``adaptive_3way_weights`` and therefore recompute
    alpha/beta/gamma and the margin gate from their own direction.  Mean-only
    is intentionally not handled here; it uses ``method/spherical_steering``.
    """
    if condition not in CONDITION_ORDER:
        raise ValueError(f"Unknown residual condition: {condition}")
    if condition == "mean_only":
        raise ValueError("mean_only uses spherical steering, not the S2S-R hook")
    mean_direction = norm(avg, eps)
    override_tensor = None
    if residual_override is not None:
        override_tensor = torch.as_tensor(
            residual_override,
            dtype=torch.float32,
            device=avg.device,
        ).reshape(-1)

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
        a = norm(original, eps)
        b = mean_direction.to(a.device).view(1, 1, -1).expand_as(a)
        if condition == "predicted_residual":
            _, predicted_residual = predict_residual(
                predictor,
                a[:, -1, :],
                avg,
                eps=eps,
            )
            used_c = predicted_residual.float().view(
                predicted_residual.shape[0], 1, -1
            ).expand_as(a)
        elif condition == "oracle_residual" and not oracle_override_valid:
            # Oracle residual direction is undefined when the raw residual is
            # near zero.  In that exact case the oracle correction is already
            # explained by v_avg, so use the mean direction as the residual
            # fallback and still recompute S2S-R weights/gate dynamically.
            used_c = b
        else:
            if override_tensor is None:
                raise ValueError(f"{condition} requires residual_override")
            if override_tensor.numel() != a.shape[-1]:
                raise ValueError(
                    f"Residual override dim {override_tensor.numel()} != hidden dim {a.shape[-1]}"
                )
            c_override = override_tensor.to(a.device).view(1, 1, -1)
            used_c = c_override.expand_as(a)

        used_alpha, used_beta, used_gamma, _ = adaptive_3way_weights(
            a,
            b,
            used_c,
            kappa=weight_kappa,
            temperature=1.0,
        )
        should_steer = torch.maximum(used_beta, used_gamma) > (
            used_alpha + margin
        )

        if stats is not None:
            token_count = int(used_alpha.numel())
            stats["total"] = stats.get("total", 0) + token_count
            stats["steered"] = stats.get("steered", 0) + int(
                should_steer.sum().item()
            )
            stats["weight_count"] = stats.get("weight_count", 0) + token_count
            stats["alpha_sum"] = stats.get("alpha_sum", 0.0) + float(
                used_alpha.sum().item()
            )
            stats["beta_sum"] = stats.get("beta_sum", 0.0) + float(
                used_beta.sum().item()
            )
            stats["gamma_sum"] = stats.get("gamma_sum", 0.0) + float(
                used_gamma.sum().item()
            )

        rotated = two_slerp_sym(
            a,
            b,
            used_c,
            used_alpha,
            used_beta,
            used_gamma,
            eps,
            small_angle_threshold,
        )
        candidate = rotated * original_norm
        edited_segment = torch.where(should_steer, candidate, original)
        edited = hidden.clone()
        edited[:, start:, :] = edited_segment.to(hidden.dtype)
        return (edited,) + rest if rest is not None else edited

    return hook


def get_static_2d_residual_intervention(
    *,
    avg: torch.Tensor,
    predictor,
    condition: str,
    residual_override: np.ndarray | None,
    oracle_override_valid: bool = True,
    strength: float,
    slerp_t: float,
    eps: float,
    small_angle_threshold: float,
    stats: dict[str, float] | None = None,
):
    """Build the released S2S-L-Static additive-steering hook for residual swaps.

    Only the residual *direction* is swapped.  The current recipient sample
    always supplies ``||v_hat||`` exactly as in ``method/dyn2d/steering.py``:

        D = SLERP(normalize(v_avg), active_residual, slerp_t)
        h_new = h + strength * ||v_hat_recipient|| * D

    S2S-L-Static has no adaptive weights and no margin gate.
    """
    if condition not in CONDITION_ORDER:
        raise ValueError(f"Unknown residual condition: {condition}")
    if condition == "mean_only":
        raise ValueError("mean_only uses spherical steering, not S2S-L-Static")
    if not math.isfinite(strength):
        raise ValueError("strength must be finite")
    if not math.isfinite(slerp_t) or not 0.0 <= slerp_t <= 1.0:
        raise ValueError("slerp_t must be finite and in [0, 1]")

    mean_direction = norm(avg, eps)
    override_tensor = None
    if residual_override is not None:
        override_tensor = torch.as_tensor(
            residual_override,
            dtype=torch.float32,
            device=avg.device,
        ).reshape(-1)

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
        v_hat, predicted_residual = predict_residual(
            predictor,
            original[:, -1, :],
            avg,
            eps=eps,
        )
        mean_batch = mean_direction.to(original.device).view(1, -1).expand_as(
            predicted_residual
        )

        if condition == "predicted_residual":
            used_c = predicted_residual.float()
        elif condition == "oracle_residual" and not oracle_override_valid:
            used_c = mean_batch
        else:
            if override_tensor is None:
                raise ValueError(f"{condition} requires residual_override")
            if override_tensor.numel() != original.shape[-1]:
                raise ValueError(
                    f"Residual override dim {override_tensor.numel()} != hidden dim {original.shape[-1]}"
                )
            used_c = override_tensor.to(original.device).view(1, -1).expand_as(
                predicted_residual
            )

        direction = dyn2d_slerp(
            mean_batch,
            used_c,
            slerp_t,
            eps=eps,
            small_angle_threshold=small_angle_threshold,
        )
        predicted_norm = torch.linalg.vector_norm(v_hat.float(), dim=-1, keepdim=True)
        scaled_direction = direction * predicted_norm
        steering = float(strength) * scaled_direction[:, None, :]

        edited = hidden.clone()
        edited[:, start:, :] = (original + steering).to(hidden.dtype)

        if stats is not None:
            token_count = int(original.shape[0] * original.shape[1])
            prediction_count = int(v_hat.shape[0])
            stats["total"] = stats.get("total", 0) + token_count
            stats["steered"] = stats.get("steered", 0) + (
                token_count if float(strength) != 0.0 else 0
            )
            stats["prediction_count"] = stats.get("prediction_count", 0) + prediction_count
            stats["predicted_norm_sum"] = stats.get("predicted_norm_sum", 0.0) + float(
                predicted_norm.sum().item()
            )
            stats["applied_norm_sum"] = stats.get("applied_norm_sum", 0.0) + float(
                (predicted_norm * abs(float(strength))).sum().item()
            )
            stats["slerp_t_sum"] = stats.get("slerp_t_sum", 0.0) + float(slerp_t) * prediction_count
            stats["slerp_t_count"] = stats.get("slerp_t_count", 0) + prediction_count

        return (edited,) + rest if rest is not None else edited

    return hook


def _score_candidate(
    *,
    model,
    record: CandidateRecord,
    hook,
    layer_name: str,
    device: torch.device,
    few_shot: bool,
) -> float:
    input_ids = record.input_ids.to(device)
    edit_hook = partial(hook, start_idx=record.start_idx)
    with torch.no_grad():
        with TraceDict(model, [layer_name], edit_output=edit_hook):
            outputs = model(input_ids)

    if not few_shot:
        logits = outputs.logits
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = input_ids[..., 1:].contiguous()
        log_probs = torch.nn.functional.log_softmax(shift_logits, dim=-1)
        token_lp = torch.gather(
            log_probs,
            dim=2,
            index=shift_labels.unsqueeze(2),
        ).squeeze(2)
        return float(token_lp[0, record.start_idx :].sum().item())

    base_len = record.start_idx
    log_probs = outputs.logits[0].log_softmax(-1)
    answer_log_probs = log_probs[base_len - 1 : -1, :]
    answer_ids = input_ids[0, base_len:]
    lp = answer_log_probs[
        torch.arange(answer_log_probs.shape[0], device=answer_log_probs.device),
        answer_ids,
    ]
    return float(lp[3:].sum().item())


def _make_mc_shuffle_map(
    records: list[CandidateRecord],
    *,
    seed: int,
) -> tuple[dict[str, np.ndarray], float]:
    q = np.asarray([record.q_idx for record in records], dtype=np.int64)
    donors, exact = constrained_shuffle_indices(q, rng=np.random.default_rng(seed))
    if np.any(donors < 0):
        raise RuntimeError("Global MC shuffle unexpectedly found a candidate with no donor")
    mapping = {
        record.candidate_id: records[int(donors[idx])].predicted_residual
        for idx, record in enumerate(records)
    }
    return mapping, float(np.mean(exact))


def _make_cross_category_cosine_nearest_map(
    records: list[CandidateRecord],
) -> tuple[dict[str, np.ndarray], dict[str, float]]:
    """Map each candidate to its closest predicted residual from another category."""
    residuals = np.stack([record.predicted_residual for record in records], axis=0)
    categories = np.asarray([record.category for record in records], dtype=object)
    q = np.asarray([record.q_idx for record in records], dtype=np.int64)
    donors, donor_cos = cross_category_cosine_nearest_indices(
        residuals, categories, question_ids=q
    )
    if np.any(donors < 0):
        missing = int(np.sum(donors < 0))
        raise RuntimeError(
            f"Cross-category cosine-nearest control has no eligible donor for {missing} candidates"
        )
    mapping = {
        record.candidate_id: records[int(donors[idx])].predicted_residual
        for idx, record in enumerate(records)
    }
    meta = {
        "eligible_fraction": float(np.mean(donors >= 0)),
        "mean_donor_cosine": float(np.mean(donor_cos)),
        "min_donor_cosine": float(np.min(donor_cos)),
        "max_donor_cosine": float(np.max(donor_cos)),
    }
    return mapping, meta


def evaluate_mc_condition(
    *,
    condition: str,
    shuffle_seed_index: int | None,
    shuffle_map: dict[str, np.ndarray] | None,
    results_template: pd.DataFrame,
    records_by_id: dict[str, CandidateRecord],
    model,
    avg: torch.Tensor,
    predictor,
    layer_name: str,
    spherical_layer_name: str,
    spherical_bank: SphericalPrototypeBank | None,
    device: torch.device,
    args: argparse.Namespace,
    score_log_handle,
) -> list[dict[str, Any]]:
    tag = condition if shuffle_seed_index is None else f"{condition}_{shuffle_seed_index}"
    results = results_template.copy().reset_index(drop=True)
    set_columns(tag, results)
    output_rows: list[dict[str, Any]] = []

    progress_label = CONDITION_LABELS[condition]
    if shuffle_seed_index is not None:
        progress_label += f" s{shuffle_seed_index}"
    progress = tqdm(range(len(results)), desc=progress_label, unit="q")

    for row_idx in progress:
        row = results.iloc[row_idx]
        q_idx = int(row["hf_idx"])
        true_answers = split_multi_answer(row["Correct Answers"])
        false_answers = split_multi_answer(row["Incorrect Answers"])
        scores_true: list[float] = []
        scores_false: list[float] = []
        question_stats: dict[str, float] = {
            "total": 0,
            "steered": 0,
            "weight_count": 0,
            "alpha_sum": 0.0,
            "beta_sum": 0.0,
            "gamma_sum": 0.0,
            "prediction_count": 0,
            "predicted_norm_sum": 0.0,
            "applied_norm_sum": 0.0,
            "slerp_t_sum": 0.0,
            "slerp_t_count": 0,
        }

        for label, answers, destination in (
            (1, true_answers, scores_true),
            (0, false_answers, scores_false),
        ):
            for answer_idx, answer in enumerate(answers):
                cid = _candidate_id(q_idx, label, answer_idx)
                record = records_by_id[cid]
                residual_override = None
                oracle_valid = True
                active_layer_name = layer_name
                if condition == "mean_only":
                    if args.steering_method == "S2S_R":
                        if spherical_bank is None:
                            raise ValueError(
                                "S2S_R mean_only requires a spherical prototype bank"
                            )
                        proto = spherical_bank.for_question(q_idx)
                        hook = get_spherical_intervention(
                            torch.as_tensor(proto.mu_T, dtype=torch.float32),
                            torch.as_tensor(proto.mu_H, dtype=torch.float32),
                            kappa=args.spherical_kappa,
                            alpha=args.spherical_alpha,
                            beta=args.spherical_beta,
                            stats=question_stats,
                        )
                        active_layer_name = spherical_layer_name
                    elif args.steering_method == "S2S_L_Static":
                        # Standard linear mean steering: preserve the checkpoint's
                        # complete v_avg vector (including its magnitude).
                        hook = get_additive_intervention(
                            avg,
                            args.strength,
                            stats=question_stats,
                        )
                        active_layer_name = layer_name
                    else:
                        raise ValueError(
                            f"Unsupported steering_method: {args.steering_method}"
                        )
                elif condition in ("shuffled_residual", "cross_category_cosine_nearest_residual"):
                    if shuffle_map is None:
                        raise ValueError(f"{condition} requires a residual override map")
                    residual_override = shuffle_map[cid]
                elif condition == "oracle_residual":
                    residual_override = record.oracle_residual
                    oracle_valid = record.oracle_valid
                if condition != "mean_only":
                    if args.steering_method == "S2S_R":
                        hook = get_dynamic_residual_intervention(
                            avg=avg,
                            predictor=predictor,
                            condition=condition,
                            residual_override=residual_override,
                            oracle_override_valid=oracle_valid,
                            weight_kappa=args.weight_kappa,
                            margin=args.margin,
                            eps=args.slerp_eps,
                            small_angle_threshold=args.small_angle_threshold,
                            stats=question_stats,
                        )
                    elif args.steering_method == "S2S_L_Static":
                        hook = get_static_2d_residual_intervention(
                            avg=avg,
                            predictor=predictor,
                            condition=condition,
                            residual_override=residual_override,
                            oracle_override_valid=oracle_valid,
                            strength=args.strength,
                            slerp_t=args.slerp_t,
                            eps=args.slerp_eps,
                            small_angle_threshold=args.small_angle_threshold,
                            stats=question_stats,
                        )
                    else:
                        raise ValueError(
                            f"Unsupported steering_method: {args.steering_method}"
                        )
                score = _score_candidate(
                    model=model,
                    record=record,
                    hook=hook,
                    layer_name=active_layer_name,
                    device=device,
                    few_shot=args.few_shot,
                )
                destination.append(score)
                if score_log_handle is not None:
                    score_log_handle.write(
                        json.dumps(
                            {
                                "condition": condition,
                                "shuffle_seed_index": shuffle_seed_index,
                                "candidate_id": cid,
                                "q_idx": q_idx,
                                "label": label,
                                "answer_index": answer_idx,
                                "answer": answer,
                                "score": score,
                            },
                            ensure_ascii=False,
                        )
                        + "\n"
                    )

        MC_calcs(
            tag,
            results,
            row_idx,
            scores_true,
            scores_false,
            true_answers,
            format_best(row["Best Answer"]),
        )
        mc_values = {
            metric: float(results.loc[row_idx, f"{tag} {metric}"])
            for metric in ("MC1", "MC2", "MC3")
        }
        output_rows.append(
            {
                "condition": condition,
                "label": CONDITION_LABELS[condition],
                "steering_method": args.steering_method,
                "shuffle_seed_index": "" if shuffle_seed_index is None else shuffle_seed_index,
                "q_idx": q_idx,
                "category": str(row.get("Category", "Unknown")),
                "question": str(row["Question"]),
                "steer_token_ratio": float(question_stats["steered"])
                / max(float(question_stats["total"]), 1.0),
                "alpha_mean": (
                    float(question_stats["alpha_sum"])
                    / float(question_stats["weight_count"])
                    if question_stats["weight_count"]
                    else float("nan")
                ),
                "beta_mean": (
                    float(question_stats["beta_sum"])
                    / float(question_stats["weight_count"])
                    if question_stats["weight_count"]
                    else float("nan")
                ),
                "gamma_mean": (
                    float(question_stats["gamma_sum"])
                    / float(question_stats["weight_count"])
                    if question_stats["weight_count"]
                    else float("nan")
                ),
                "slerp_t_mean": (
                    float(question_stats["slerp_t_sum"])
                    / float(question_stats["slerp_t_count"])
                    if question_stats["slerp_t_count"]
                    else float("nan")
                ),
                "predicted_vector_norm_mean": (
                    float(question_stats["predicted_norm_sum"])
                    / float(question_stats["prediction_count"])
                    if question_stats["prediction_count"]
                    else float("nan")
                ),
                "applied_steering_norm_mean": (
                    float(question_stats["applied_norm_sum"])
                    / float(question_stats["prediction_count"])
                    if question_stats["prediction_count"]
                    else float("nan")
                ),
                **mc_values,
            }
        )
        running = float(np.mean([entry["MC1"] for entry in output_rows]))
        progress.set_description(f"{progress_label} MC1:{running:.3f}")

    return output_rows


def summarize_mc(
    mc_rows: list[dict[str, Any]],
    *,
    n_bootstrap: int,
    seed: int,
) -> list[dict[str, Any]]:
    frame = pd.DataFrame(mc_rows)
    if frame.empty:
        return []

    # Average the expensive shuffled-residual seeds *within question first*.
    # This preserves question as the independent statistical unit.
    question_condition = (
        frame.groupby(["condition", "q_idx"], as_index=False)[["MC1", "MC2", "MC3"]]
        .mean()
    )
    mean_only = question_condition[question_condition["condition"] == "mean_only"].set_index("q_idx")
    if mean_only.empty:
        raise ValueError("MC summary requires mean_only rows")

    summary: list[dict[str, Any]] = []
    for offset, condition in enumerate(CONDITION_ORDER):
        selected = question_condition[question_condition["condition"] == condition].set_index("q_idx")
        common_q = mean_only.index.intersection(selected.index)
        if len(common_q) == 0:
            continue
        delta_mc1_pp = (
            selected.loc[common_q, "MC1"].to_numpy(dtype=float)
            - mean_only.loc[common_q, "MC1"].to_numpy(dtype=float)
        ) * 100.0
        delta_mean, delta_lo, delta_hi = bootstrap_mean_ci(
            delta_mc1_pp,
            n_bootstrap=n_bootstrap,
            seed=seed + 500 + offset,
        )
        row = {
            "condition": condition,
            "label": CONDITION_LABELS[condition],
            "num_questions": int(len(common_q)),
            "MC1": float(selected.loc[common_q, "MC1"].mean()),
            "MC2": float(selected.loc[common_q, "MC2"].mean()),
            "MC3": float(selected.loc[common_q, "MC3"].mean()),
            "delta_MC1_pp_over_mean_only": delta_mean,
            "delta_MC1_pp_ci_low": delta_lo,
            "delta_MC1_pp_ci_high": delta_hi,
        }
        summary.append(row)
    return summary


def _load_model(args: argparse.Namespace, device: torch.device):
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
    return model, tokenizer


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    device = resolve_device(args.device)
    checkpoint_data, avg, predictor = load_components(args.checkpoint_path, device)
    fold_idx = None
    if "fold_idx" in checkpoint_data.files:
        fold_idx = int(np.asarray(checkpoint_data["fold_idx"]).item())
    if args.expected_fold is not None and fold_idx != args.expected_fold:
        raise ValueError(
            f"checkpoint fold mismatch: expected {args.expected_fold}, found {fold_idx}"
        )

    if args.steering_method == "S2S_R":
        method_parameters = f"kappa={args.weight_kappa} margin={args.margin}"
    else:
        method_parameters = f"strength={args.strength} slerp_t={args.slerp_t}"
    print(
        f"{args.steering_method} residual analysis | model={args.model_name} "
        f"layer={args.layer} fold={fold_idx} {method_parameters}"
    )
    if args.steering_method == "S2S_R":
        print(
            "Mean-only baseline | method=spherical_steering "
            f"layer={args.spherical_layer} kappa={args.spherical_kappa} "
            f"alpha={args.spherical_alpha} beta={args.spherical_beta}"
        )
    else:
        print(
            "Mean-only baseline | method=linear_additive_mean "
            f"layer={args.layer} strength={args.strength} formula=h+strength*v_avg"
        )
    model, tokenizer = _load_model(args, device)
    dataframe, hf_dataset = load_truthfulqa_mc_dataframe(args.csv_path)
    if args.eval_split == "all":
        selected_ids = set(range(len(hf_dataset)))
    else:
        split_key = "train_q_indices" if args.eval_split == "train" else "test_q_indices"
        if split_key not in checkpoint_data.files:
            raise KeyError(f"checkpoint missing {split_key}")
        selected_ids = set(int(x) for x in np.asarray(checkpoint_data[split_key]).reshape(-1))
    results = dataframe[dataframe["hf_idx"].isin(selected_ids)].copy().reset_index(drop=True)
    if args.max_questions is not None:
        results = results.iloc[: args.max_questions].copy().reset_index(drop=True)
    if len(results) == 0:
        raise ValueError("Selected evaluation split is empty")

    evaluated_q = set(int(x) for x in results["hf_idx"].tolist())
    spherical_bank: SphericalPrototypeBank | None = None
    if args.steering_method == "S2S_R":
        spherical_bank = load_spherical_prototype_bank(
            list(args.spherical_prototype_paths or []),
            expected_dim=int(avg.numel()),
        )
        validate_spherical_coverage(spherical_bank, evaluated_q)

    layer_name = get_layer_name(args.model_name, args.layer)
    spherical_layer_name = (
        get_layer_name(args.model_name, args.spherical_layer)
        if args.steering_method == "S2S_R"
        else layer_name
    )
    records = collect_candidate_records(
        results=results,
        model=model,
        tokenizer=tokenizer,
        layer_name=layer_name,
        device=device,
        avg=avg,
        predictor=predictor,
        args=args,
    )
    records_by_id = {record.candidate_id: record for record in records}
    if len(records_by_id) != len(records):
        raise RuntimeError("Candidate IDs are not unique")

    # ---------------- Panel (a) + Panel (c): cheap offline geometry ----------------
    geometry_rows = compute_negative_geometry(
        records,
        avg=avg,
        steering_method=args.steering_method,
        strength=args.strength,
        slerp_t=args.slerp_t,
        spherical_bank=spherical_bank,
        spherical_kappa=args.spherical_kappa,
        spherical_alpha=args.spherical_alpha,
        spherical_beta=args.spherical_beta,
        weight_kappa=args.weight_kappa,
        margin=args.margin,
        eps=args.slerp_eps,
        small_angle_threshold=args.small_angle_threshold,
    )
    alignment_question_rows, shuffle_null_rows, alignment_stats = compute_alignment_controls(
        geometry_rows,
        records,
        num_seeds=args.num_shuffle_seeds,
        seed=args.seed,
    )
    alignment_summary_rows = summarize_alignment(
        alignment_question_rows,
        n_bootstrap=args.bootstrap_samples,
        seed=args.seed,
    )
    geometry_with_quintiles, quintile_summary_rows = summarize_quintiles(
        geometry_rows,
        n_quintiles=args.quintiles,
        n_bootstrap=args.bootstrap_samples,
        seed=args.seed,
    )

    _write_csv(output_dir / "geometry_pairs.csv", geometry_with_quintiles)
    _write_csv(output_dir / "alignment_question.csv", alignment_question_rows)
    _write_csv(output_dir / "alignment_shuffle_null.csv", shuffle_null_rows)
    _write_csv(output_dir / "alignment_summary.csv", alignment_summary_rows)
    _write_csv(output_dir / "quintile_summary.csv", quintile_summary_rows)

    # ---------------- Panel (b): full MC residual-swap intervention ----------------
    mc_rows: list[dict[str, Any]] = []
    mc_summary_rows: list[dict[str, Any]] = []
    mc_shuffle_meta: list[dict[str, Any]] = []
    score_log_handle = None
    try:
        if args.save_mc_scores and not args.skip_mc:
            score_log_handle = (output_dir / "mc_scores.jsonl").open(
                "w", encoding="utf-8", buffering=1
            )
        if not args.skip_mc:
            for condition in ("mean_only", "predicted_residual", "oracle_residual"):
                mc_rows.extend(
                    evaluate_mc_condition(
                        condition=condition,
                        shuffle_seed_index=None,
                        shuffle_map=None,
                        results_template=results,
                        records_by_id=records_by_id,
                        model=model,
                        avg=avg,
                        predictor=predictor,
                        layer_name=layer_name,
                        spherical_layer_name=spherical_layer_name,
                        spherical_bank=spherical_bank,
                        device=device,
                        args=args,
                        score_log_handle=score_log_handle,
                    )
                )

            cross_category_map, cross_category_meta = _make_cross_category_cosine_nearest_map(records)
            mc_shuffle_meta.append(
                {"condition": "cross_category_cosine_nearest_residual", **cross_category_meta}
            )
            mc_rows.extend(
                evaluate_mc_condition(
                    condition="cross_category_cosine_nearest_residual",
                    shuffle_seed_index=None,
                    shuffle_map=cross_category_map,
                    results_template=results,
                    records_by_id=records_by_id,
                    model=model,
                    avg=avg,
                    predictor=predictor,
                    layer_name=layer_name,
                    spherical_layer_name=spherical_layer_name,
                    spherical_bank=spherical_bank,
                    device=device,
                    args=args,
                    score_log_handle=score_log_handle,
                )
            )

            for shuffle_idx in range(args.mc_shuffle_seeds):
                shuffle_map, exact_fraction = _make_mc_shuffle_map(
                    records,
                    seed=args.seed + 900001 * (shuffle_idx + 1),
                )
                mc_shuffle_meta.append(
                    {
                        "condition": "shuffled_residual",
                        "shuffle_seed_index": shuffle_idx,
                        "exact_permutation_fraction": exact_fraction,
                    }
                )
                mc_rows.extend(
                    evaluate_mc_condition(
                        condition="shuffled_residual",
                        shuffle_seed_index=shuffle_idx,
                        shuffle_map=shuffle_map,
                        results_template=results,
                        records_by_id=records_by_id,
                        model=model,
                        avg=avg,
                        predictor=predictor,
                        layer_name=layer_name,
                        spherical_layer_name=spherical_layer_name,
                        spherical_bank=spherical_bank,
                        device=device,
                        args=args,
                        score_log_handle=score_log_handle,
                    )
                )

            _write_csv(output_dir / "mc_question.csv", mc_rows)
            mc_summary_rows = summarize_mc(
                mc_rows,
                n_bootstrap=args.bootstrap_samples,
                seed=args.seed,
            )
            _write_csv(output_dir / "mc_summary.csv", mc_summary_rows)
    finally:
        if score_log_handle is not None:
            score_log_handle.close()

    predicted_invalid_all = sum(not record.predicted_valid for record in records)
    predicted_invalid_negative = sum(
        (record.label == 0 and not record.predicted_valid) for record in records
    )
    oracle_invalid_all = sum(not record.oracle_valid for record in records)
    oracle_invalid_negative = sum(
        (record.label == 0 and not record.oracle_valid) for record in records
    )
    summary = {
        "experiment": "residual_direction_analysis",
        "steering_method": args.steering_method,
        "mean_only_baseline": (
            "spherical_steering"
            if args.steering_method == "S2S_R"
            else "linear_additive_v_avg"
        ),
        "model_name": args.model_name,
        "layer": int(args.layer),
        "fold_idx": fold_idx,
        "checkpoint_path": str(args.checkpoint_path),
        "eval_split": args.eval_split,
        "num_questions": int(len(results)),
        "num_candidates": int(len(records)),
        "num_negative_candidates": int(sum(record.label == 0 for record in records)),
        "predicted_invalid_candidates": int(predicted_invalid_all),
        "predicted_invalid_negative_candidates": int(predicted_invalid_negative),
        "oracle_invalid_candidates": int(oracle_invalid_all),
        "oracle_invalid_negative_candidates": int(oracle_invalid_negative),
        "parameters": {
            "weight_kappa": float(args.weight_kappa),
            "margin": float(args.margin),
            "strength": float(args.strength),
            "slerp_t": float(args.slerp_t),
            "spherical_layer": int(args.spherical_layer),
            "spherical_kappa": float(args.spherical_kappa),
            "spherical_alpha": float(args.spherical_alpha),
            "spherical_beta": float(args.spherical_beta),
            "spherical_prototype_paths": (
                list(spherical_bank.source_paths) if spherical_bank is not None else []
            ),
            "few_shot": bool(args.few_shot),
            "preset": args.preset,
            "use_instruction": not args.no_instruction,
            "num_shuffle_seeds": int(args.num_shuffle_seeds),
            "mc_shuffle_seeds": int(args.mc_shuffle_seeds),
            "bootstrap_samples": int(args.bootstrap_samples),
            "quintiles": int(args.quintiles),
            "seed": int(args.seed),
        },
        "definitions": {
            "predicted_residual": "normalize(MLP(normalize(h_last)) - v_avg)",
            "oracle_correction": (
                "for incorrect candidates: z_pos_star(question,candidate) - h_last(candidate), "
                "where z_pos_star is the same-question truthful hidden with maximum cosine; "
                "truthful Oracle-MC rows retain the truthful centroid target to avoid self-match degeneracy"
            ),
            "oracle_residual": "normalize(oracle_correction - v_avg)",
            "panel_a_population": "held-out incorrect-answer candidates; averaged within question for inference",
            "mean_only": (
                "S2S_R: released method/spherical_steering baseline using "
                "hyperparams/spherical_steering, cross-fitted by question. "
                "S2S_L_Static: standard linear additive mean steering "
                "h_new = h + strength * v_avg using the Dyn2D strength."
            ),
            "panel_b_residual_control": (
                "For S2S_R, shuffled/cross-category-cosine-nearest/oracle replace c before "
                "adaptive_3way_weights so alpha/beta/gamma and the margin gate are recomputed "
                "from the active residual. For S2S_L_Static, shuffled/cross-category-cosine-nearest/"
                "oracle replace only the residual direction; "
                "the recipient sample keeps its own predicted-vector magnitude and the method's "
                "fixed strength/slerp_t are used exactly as in method/dyn2d/steering.py."
            ),
            "panel_c_gain": (
                "angle(method-family mean-only edited last-token direction, nearest-positive truthful direction) - "
                "angle(selected-method predicted-residual edited last-token direction, nearest-positive truthful direction). "
                "The mean-only output is Spherical Steering for S2S_R and h+strength*v_avg for S2S_L_Static."
            ),
            "near_zero_policy": (
                "exclude undefined predicted/oracle residual directions from Panels (a,c); "
                "in Oracle MC, an undefined oracle residual uses the mean direction as c; "
                "S2S_R then recomputes dynamic weights/gate, while S2S_L_Static applies its "
                "fixed SLERP/additive rule."
            ),
        },
        "alignment_stats": alignment_stats,
        "alignment_summary": alignment_summary_rows,
        "quintile_summary": quintile_summary_rows,
        "mc_summary": mc_summary_rows,
        "mc_shuffle_metadata": mc_shuffle_meta,
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"Saved residual analysis to {output_dir}")
    if args.skip_mc:
        print("MC residual-swap evaluation skipped (--skip_mc).")


if __name__ == "__main__":
    main()
