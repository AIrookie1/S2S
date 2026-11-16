"""Measure S2S-R gate decisions on labeled TruthfulQA candidates.

The production intervention and zero-shot scoring stay unchanged. This wrapper
observes the exact production weight functions and checks counts against the
production hook on every candidate in both apply and observe modes.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
from datetime import datetime, timezone

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import numpy as np
import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from method.dyn3d_sga.dyn_mlp import predict_residual
from method.dyn3d_sga.evaluate_3d_dyn import (
    adaptive_3way_weights, get_3d_dynamic_intervention, norm,
)
from method.dyn3d_sga.evaluate_dyn_mlp import load_components
from method.dyn3d_sga.eval_utils import (
    calculate_scores_zeroshot, format_best, get_layer_name,
    load_truthfulqa_mc_dataframe, set_seed, split_multi_answer,
)
from experiments.gate_correctness.analyze import analyze


def digest(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def write_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def summarize_mask(mask):
    count = int(mask.numel())
    if count == 0:
        raise ValueError("Empty answer interval")
    triggered = int(mask.sum().item())
    return {"tokens": count, "triggered": triggered, "rate": triggered / count}


def window_counts(mask):
    """Segment = final prompt token followed by all N candidate tokens.

    Answer hidden states are [1:N+1]; positions predicting the N scored answer
    tokens are [0:N]. The legacy hook includes the whole [0:N+1] segment.
    """
    if mask.ndim != 1 or mask.numel() < 2:
        raise ValueError("Expected final prompt position plus >=1 answer token")
    return {
        "answer_tokens": summarize_mask(mask[1:]),
        "scored_answer_predictors": summarize_mask(mask[:-1]),
        "legacy_intervention_segment": summarize_mask(mask),
    }


class GateRecorder:
    def __init__(self, avg, predictor, kappa, margin, mode):
        if mode not in {"apply", "observe"}:
            raise ValueError(mode)
        self.avg, self.predictor = avg, predictor
        self.kappa, self.margin, self.mode = kappa, margin, mode
        self.stats = {}
        self.native = get_3d_dynamic_intervention(
            avg, predictor, weight_kappa=kappa, margin=margin, stats=self.stats,
        )
        self.records = []

    def as_hook(self):
        # baukit inspects full argument lists; callable instances/bound methods
        # expose self and confuse its optional-argument dispatch. Use a closure.
        def hook(output, layer_name=None, start_idx=None):
            return self(output, layer_name=layer_name, start_idx=start_idx)
        return hook

    def __call__(self, output, layer_name=None, start_idx=None):
        hidden = output[0] if isinstance(output, tuple) else output
        if hidden.ndim != 3 or hidden.shape[0] != 1:
            raise ValueError("This experiment requires one unpadded candidate per forward")
        start = int(start_idx)
        if not 0 <= start < hidden.shape[1] - 1:
            raise ValueError("Invalid prompt/answer boundary")
        a = norm(hidden[:, start:, :].float())
        _, residual = predict_residual(self.predictor, a[:, -1, :], self.avg)
        b = norm(self.avg).view(1, 1, -1).expand_as(a)
        c = residual.float().view(1, 1, -1).expand_as(a)
        alpha, beta, gamma, _ = adaptive_3way_weights(a, b, c, kappa=self.kappa, temperature=1.0)
        gap = torch.maximum(beta, gamma) - alpha
        mask = (torch.maximum(beta, gamma) > alpha + self.margin).reshape(-1)
        self.stats.clear()
        edited = self.native(output, layer_name=layer_name, start_idx=start)
        if self.stats["total"] != mask.numel() or self.stats["steered"] != int(mask.sum().item()):
            raise AssertionError("Observed gate counts differ from production hook")
        for key, weights in (("alpha_sum", alpha), ("beta_sum", beta), ("gamma_sum", gamma)):
            if not np.isclose(self.stats[key], weights.sum().item(), atol=1e-6, rtol=1e-6):
                raise AssertionError("Observed weights differ from production hook")
        self.records.append({
            "windows": window_counts(mask),
            "gate_mask_segment": mask.cpu().tolist(),
            "weight_means_answer": [float(w[:, 1:, :].mean().item()) for w in (alpha, beta, gamma)],
            "gate_gap_answer": gap.reshape(-1)[1:].cpu().tolist(),
            "native_count_check": True,
        })
        return edited if self.mode == "apply" else output


def validate_splits(checkpoints, n_questions):
    test_sets, metadata = [], []
    for requested_fold, path in checkpoints:
        with np.load(path, allow_pickle=False) as d:
            if int(d["fold_idx"]) != requested_fold or int(d["num_folds"]) != 2:
                raise ValueError("Unexpected checkpoint fold metadata")
            if str(d["method"].item()) != "dyn3d_direct_mlp":
                raise ValueError("Expected current direct-MLP checkpoint")
            if str(d["positive_mode"].item()) != "mean":
                raise ValueError("Expected mean-positive checkpoint")
            test = set(map(int, d["test_q_indices"]))
            if not test or not test.issubset(set(range(n_questions))):
                raise ValueError("Invalid test indices")
            train = set()
            counts = {}
            for key in ("outer_train_q_indices", "train_q_indices", "mean_train_q_indices", "mlp_train_q_indices"):
                values = set(map(int, d[key]))
                if test & values:
                    raise ValueError(f"Test/train overlap in {key}")
                train |= values
                counts[key] = len(values)
            if train | test != set(range(n_questions)):
                raise ValueError("Checkpoint partition does not cover TruthfulQA")
            if any(test & previous for previous in test_sets):
                raise ValueError("Test questions repeat across folds")
            test_sets.append(test)
            metadata.append({"fold": requested_fold, "checkpoint": str(path.resolve()),
                             "sha256": digest(path), "test_questions": len(test), **counts})
    if len(checkpoints) == 2 and set.union(*test_sets) != set(range(n_questions)):
        raise ValueError("Two held-out folds must cover all questions exactly once")
    return test_sets, metadata


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model_name", default="llama3.1-8B-Instruct")
    parser.add_argument("--model_dir", default="/mnt/nas-public-bing/pinxuanli11/Meta-Llama-3.1-8B-Instruct")
    parser.add_argument("--checkpoint_dir", default=str(ROOT / "checkpoints/dyn3d_sga/truthfulqa/llama3.1-8B-Instruct"))
    parser.add_argument("--csv_path", default=str(ROOT / "TruthfulQA/data/v1/TruthfulQA.csv"))
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--folds", type=int, nargs="+", default=[0, 1])
    parser.add_argument("--modes", nargs="+", choices=["apply", "observe"], default=["apply", "observe"])
    parser.add_argument("--layer", type=int, default=14)
    parser.add_argument("--weight_kappa", type=float, default=3.0)
    parser.add_argument("--margin", type=float, default=-0.7)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--bootstrap_repeats", type=int, default=20000)
    parser.add_argument("--max_questions_per_fold", type=int, default=0)
    parser.add_argument("--dtype", choices=["float16", "bfloat16", "float32"], default="float16")
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required; refusing an accidental CPU full run")
    if len(set(args.folds)) != len(args.folds) or not set(args.folds) <= {0, 1}:
        raise ValueError("Folds must be a unique subset of 0,1")
    if len(set(args.modes)) != len(args.modes) or args.max_questions_per_fold < 0:
        raise ValueError("Invalid modes or question limit")
    if args.bootstrap_repeats < 1000:
        raise ValueError("Use at least 1000 bootstrap draws")
    if not np.isfinite(args.margin) or not np.isfinite(args.weight_kappa) or args.weight_kappa <= 0:
        raise ValueError("Invalid gate parameters")
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    if any((out / f).exists() for f in ("config.json", "candidates.jsonl")):
        raise FileExistsError("Output already contains a run; choose a fresh directory")
    set_seed(args.seed)
    frame, hf = load_truthfulqa_mc_dataframe(args.csv_path)
    checkpoints = [(f, Path(args.checkpoint_dir) / f"{args.model_name}_layer{args.layer}_dynMLP_fold{f}_mean.npz") for f in args.folds]
    test_sets, metadata = validate_splits(checkpoints, len(hf))
    source_paths = [Path(__file__), Path(__file__).with_name("analyze.py"),
                    ROOT / "method/dyn3d_sga/evaluate_3d_dyn.py",
                    ROOT / "method/dyn3d_sga/eval_utils.py", ROOT / "method/dyn3d_sga/dyn_mlp.py"]
    config = {
        **vars(args), "started_utc": datetime.now(timezone.utc).isoformat(),
        "gpu_visibility": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "gpu_name": torch.cuda.get_device_name(0), "torch": torch.__version__,
        "checkpoints": metadata, "csv_sha256": digest(args.csv_path),
        "sources_sha256": {str(p.relative_to(ROOT)): digest(p) for p in source_paths},
        "git_head": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        "prompt": "repository zero-shot with default Llama instruction; all CSV correct/incorrect answers",
        "primary_window": "answer_tokens: exclude final prompt position, include all candidate hidden states",
        "predictor_policy": "production sequence-level residual from last candidate hidden state, broadcast to segment",
        "bootstrap_policy": "paired question resampling within each fold; fixed trained checkpoints",
        "run_kind": "smoke" if args.max_questions_per_fold else "full",
    }
    write_json(out / "config.json", config)
    status = {"state": "loading_model", "completed_questions": 0, "candidate_mode_records": 0}
    write_json(out / "status.json", status)
    try:
        tokenizer = AutoTokenizer.from_pretrained(args.model_dir, local_files_only=True)
        model = AutoModelForCausalLM.from_pretrained(
            args.model_dir, torch_dtype=getattr(torch, args.dtype),
            device_map={"": 0}, local_files_only=True,
        ).eval()
        model.config.use_cache = False
        records = []
        with (out / "candidates.jsonl").open("w") as stream, torch.inference_mode():
            for (fold, checkpoint), test in zip(checkpoints, test_sets):
                data, avg, predictor = load_components(str(checkpoint), torch.device("cuda:0"))
                selected = frame[frame.hf_idx.isin(test)].sort_values("hf_idx")
                if args.max_questions_per_fold:
                    selected = selected.head(args.max_questions_per_fold)
                progress = tqdm(selected.to_dict("records"), desc=f"fold{fold}")
                for row in progress:
                    true = split_multi_answer(row["Correct Answers"])
                    false = split_multi_answer(row["Incorrect Answers"])
                    if not true or not false:
                        raise ValueError("Every question must have both candidate classes")
                    for label, choices in (("correct", true), ("incorrect", false)):
                        for index, answer in enumerate(choices):
                            paired = []
                            answer_ids = tokenizer(" " + answer, add_special_tokens=False).input_ids
                            for mode in args.modes:
                                recorder = GateRecorder(avg, predictor, args.weight_kappa, args.margin, mode)
                                score = calculate_scores_zeroshot(
                                    model=model, tokenizer=tokenizer, question=row["Question"],
                                    choices=[answer], base_hook_fn=recorder.as_hook(),
                                    layer_name=get_layer_name(args.model_name, args.layer),
                                    device=torch.device("cuda:0"), use_instruction=True,
                                    model_name=args.model_name,
                                )[0]
                                if len(recorder.records) != 1 or not np.isfinite(score):
                                    raise AssertionError("Expected one finite-score forward per candidate")
                                observation = recorder.records[0]
                                if observation["windows"]["answer_tokens"]["tokens"] != len(answer_ids):
                                    raise AssertionError("Answer token boundary mismatch")
                                record = {
                                    "fold": fold, "hf_idx": int(row["hf_idx"]), "question": row["Question"],
                                    "category": row["Category"], "label": label, "candidate_index": index,
                                    "answer": answer, "best_answer": format_best(row["Best Answer"]),
                                    "mode": mode, "log_probability": score,
                                    "answer_token_ids": answer_ids, **observation,
                                }
                                stream.write(json.dumps(record, ensure_ascii=False, allow_nan=False) + "\n")
                                records.append(record)
                                paired.append(record)
                            if len(paired) == 2 and paired[0]["gate_mask_segment"] != paired[1]["gate_mask_segment"]:
                                raise AssertionError("Single-layer apply/observe gates unexpectedly differ")
                    stream.flush()
                    status.update(state="running", fold=fold,
                                  completed_questions=status["completed_questions"] + 1,
                                  candidate_mode_records=len(records))
                    write_json(out / "status.json", status)
                    progress.set_postfix(records=len(records))
                data.close()
        analyze(records, out, args.bootstrap_repeats, args.seed)
        status.update(state="complete", completed_utc=datetime.now(timezone.utc).isoformat())
        write_json(out / "status.json", status)
    except BaseException as exc:
        status.update(state="failed", error=f"{type(exc).__name__}: {exc}")
        write_json(out / "status.json", status)
        raise


if __name__ == "__main__":
    main()
