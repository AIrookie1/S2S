"""Report margin-wise gate rates for TruthfulQA correct/incorrect candidates."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
import sys

import numpy as np
import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from experiments.rebuttal_R3Q3_candidate_gates.evaluate import validate_splits
from method.S2S_R.predictor import predict_residual
from method.S2S_R.evaluate_S2S_R import adaptive_3way_weights, get_s2s_r_intervention, norm
from method.S2S_R.evaluate_predictor import load_components
from method.S2S_R.eval_utils import (
    calculate_scores_zeroshot,
    format_best,
    get_layer_name,
    load_truthfulqa_mc_dataframe,
    set_seed,
    split_multi_answer,
)


DEFAULT_MARGINS = [-0.8, -0.6, -0.4, -0.2, 0.0, 0.2, 0.4, 0.6, 0.8, 1.0]


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def write_json(path: Path, value) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def summarize_mask(mask: torch.Tensor) -> dict:
    tokens = int(mask.numel())
    if tokens == 0:
        raise ValueError("Empty answer interval")
    triggered = int(mask.sum().item())
    return {"tokens": tokens, "triggered": triggered, "rate": triggered / tokens}


def window_counts(mask: torch.Tensor) -> dict:
    if mask.ndim != 1 or mask.numel() < 2:
        raise ValueError("Expected final prompt position plus at least one answer token")
    return {
        "answer_tokens": summarize_mask(mask[1:]),
        "scored_answer_predictors": summarize_mask(mask[:-1]),
        "legacy_intervention_segment": summarize_mask(mask),
    }


class MultiMarginGateRecorder:
    """Compute the production gate gap once and threshold it at all margins."""

    def __init__(self, avg, predictor, kappa: float, margins: list[float]):
        self.avg = avg
        self.predictor = predictor
        self.kappa = kappa
        self.margins = margins
        self.records = []
        self.reference_stats = {}
        self.reference_margin = margins[0]
        self.native = get_s2s_r_intervention(
            avg,
            predictor,
            weight_kappa=kappa,
            margin=self.reference_margin,
            stats=self.reference_stats,
        )

    def as_hook(self):
        def hook(output, layer_name=None, start_idx=None):
            return self(output, layer_name=layer_name, start_idx=start_idx)

        return hook

    def __call__(self, output, layer_name=None, start_idx=None):
        del layer_name
        hidden = output[0] if isinstance(output, tuple) else output
        if hidden.ndim != 3 or hidden.shape[0] != 1:
            raise ValueError("This experiment requires one unpadded candidate per forward")
        start = int(start_idx)
        if not 0 <= start < hidden.shape[1] - 1:
            raise ValueError("Invalid prompt/answer boundary")

        segment = norm(hidden[:, start:, :].float())
        _, residual = predict_residual(self.predictor, segment[:, -1, :], self.avg)
        mean_direction = norm(self.avg).view(1, 1, -1).expand_as(segment)
        residual_direction = residual.float().view(1, 1, -1).expand_as(segment)
        alpha, beta, gamma, _ = adaptive_3way_weights(
            segment,
            mean_direction,
            residual_direction,
            kappa=self.kappa,
            temperature=1.0,
        )
        gap = (torch.maximum(beta, gamma) - alpha).reshape(-1)

        self.reference_stats.clear()
        self.native(output, start_idx=start)
        reference_mask = gap > self.reference_margin
        if (
            self.reference_stats.get("total") != int(reference_mask.numel())
            or self.reference_stats.get("steered") != int(reference_mask.sum().item())
        ):
            raise AssertionError("Multi-margin observer disagrees with production gate")

        self.records.append(
            {
                "windows": {
                    str(margin): window_counts(gap > margin) for margin in self.margins
                },
                "native_count_check": True,
            }
        )
        return output


def load_mc1_by_margin(eval_root: Path, model_name: str, fold: int) -> dict[float, float]:
    directory = eval_root / model_name / f"fold{fold}"
    values = {}
    for path in sorted(directory.glob("*.json")):
        data = json.loads(path.read_text())
        parameters = data.get("parameters", {})
        if "margin" not in parameters:
            continue
        values[float(parameters["margin"])] = float(data["metrics"]["MC1"])
    return values


def write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def aggregate(records: list[dict], margins: list[float], mc1_by_fold: dict[int, dict[float, float]]):
    question_groups = defaultdict(list)
    for record in records:
        question_groups[(record["fold"], record["hf_idx"], record["label"])].append(record)

    rows = []
    summaries = {}
    folds = sorted({record["fold"] for record in records})
    for margin in margins:
        margin_key = str(margin)
        summaries[margin_key] = {}
        for scope in ["pooled"] + [f"fold{fold}" for fold in folds]:
            selected = [
                record
                for record in records
                if scope == "pooled" or record["fold"] == int(scope.removeprefix("fold"))
            ]
            question_rates = {"correct": [], "incorrect": []}
            class_tokens = {"correct": 0, "incorrect": 0}
            class_triggered = {"correct": 0, "incorrect": 0}
            class_candidates = {"correct": 0, "incorrect": 0}
            for (fold, hf_idx, label), candidates in question_groups.items():
                if scope != "pooled" and fold != int(scope.removeprefix("fold")):
                    continue
                margin_stats = [candidate["windows"][margin_key]["answer_tokens"] for candidate in candidates]
                question_rates[label].append(float(np.mean([stat["rate"] for stat in margin_stats])))
                class_candidates[label] += len(candidates)
                class_tokens[label] += sum(stat["tokens"] for stat in margin_stats)
                class_triggered[label] += sum(stat["triggered"] for stat in margin_stats)

            question_count = len({(record["fold"], record["hf_idx"]) for record in selected})
            correct_rate = float(np.mean(question_rates["correct"]))
            incorrect_rate = float(np.mean(question_rates["incorrect"]))
            fold_ids = folds if scope == "pooled" else [int(scope.removeprefix("fold"))]
            mc1_weights = [
                (len({record["hf_idx"] for record in records if record["fold"] == fold}), mc1_by_fold[fold][margin])
                for fold in fold_ids
            ]
            mc1 = sum(count * value for count, value in mc1_weights) / sum(count for count, _ in mc1_weights)
            summary = {
                "margin": margin,
                "scope": scope,
                "questions": question_count,
                "MC1": mc1,
                "correct_rate": correct_rate,
                "incorrect_rate": incorrect_rate,
                "incorrect_minus_correct": incorrect_rate - correct_rate,
                "correct_micro_rate": class_triggered["correct"] / class_tokens["correct"],
                "incorrect_micro_rate": class_triggered["incorrect"] / class_tokens["incorrect"],
                "correct_candidates": class_candidates["correct"],
                "incorrect_candidates": class_candidates["incorrect"],
                "correct_tokens": class_tokens["correct"],
                "incorrect_tokens": class_tokens["incorrect"],
                "correct_triggered": class_triggered["correct"],
                "incorrect_triggered": class_triggered["incorrect"],
            }
            summaries[margin_key][scope] = summary
            rows.append(summary)
    return summaries, rows


def plot(summaries: dict, margins: list[float], output: Path, model_name: str) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    pooled = [summaries[str(m)]["pooled"] for m in margins]
    x = np.arange(len(margins))
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    axes[0].plot(x, [100 * row["correct_rate"] for row in pooled], marker="o", label="Correct")
    axes[0].plot(x, [100 * row["incorrect_rate"] for row in pooled], marker="o", label="Incorrect")
    axes[0].set_ylabel("Gate trigger rate (%)")
    axes[0].set_xlabel("Margin")
    axes[0].set_ylim(0, 105)
    axes[0].legend()
    axes[1].plot(x, [100 * row["MC1"] for row in pooled], marker="o", color="#444444")
    axes[1].set_ylabel("MC1 (%)")
    axes[1].set_xlabel("Margin")
    axes[1].set_ylim(0, 100)
    for axis in axes:
        axis.set_xticks(x, [f"{margin:g}" for margin in margins])
        axis.spines[["top", "right"]].set_visible(False)
        axis.grid(axis="y", alpha=0.25)
    fig.suptitle(model_name)
    fig.tight_layout()
    for suffix in ("png", "pdf", "svg"):
        fig.savefig(output.with_suffix(f".{suffix}"), dpi=180)
    plt.close(fig)


def report(all_summaries: dict, margins: list[float], output: Path) -> None:
    lines = [
        "# Margin × TruthfulQA 候选类别门控触发率与 MC1",
        "",
        "主触发率：每题内对同一类别候选等权平均，再对题目等权平均；同时保留 token-weighted 触发率。",
        "MC1：复用同一 margin、同一 fold 的已有 TruthfulQA 评测结果，并按 fold 测试题数量加权。",
        "",
    ]
    for model_name, summaries in all_summaries.items():
        lines.extend([f"## {model_name}", "", "| margin | 正确 token 触发率 | 错误 token 触发率 | 差值（错误-正确） | 正确按题等权 | 错误按题等权 | MC1 |", "|---:|---:|---:|---:|---:|---:|---:|"])
        for margin in margins:
            row = summaries[str(margin)]["pooled"]
            lines.append(
                f"| {margin:g} | {row['correct_micro_rate']:.2%} | {row['incorrect_micro_rate']:.2%} | "
                f"{(row['incorrect_micro_rate'] - row['correct_micro_rate']) * 100:+.2f} pp | "
                f"{row['correct_rate']:.2%} | {row['incorrect_rate']:.2%} | {row['MC1']:.2%} |"
            )
        lines.append("")
    lines.extend([
        "## 统计口径",
        "",
        "门控触发条件为生产实现中的 `max(beta, gamma) > alpha + margin`。",
        "当前分析是 TruthfulQA teacher-forcing 候选评分，不等价于自由生成时的在线门控行为。",
        "",
    ])
    output.write_text("\n".join(lines))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model_name", default="llama3.1-8B-Instruct")
    parser.add_argument("--model_dir", default=__import__("os").environ.get("LLAMA_MODEL_DIR", "models/Meta-Llama-3.1-8B-Instruct"))
    parser.add_argument("--checkpoint_dir", default=str(ROOT / "checkpoints/S2S_R/truthfulqa/llama3.1-8B-Instruct"))
    parser.add_argument("--eval_root", default=str(ROOT / "results/experiment/fig3c_gate_margin/evaluations"))
    parser.add_argument("--csv_path", default=str(ROOT / "TruthfulQA/data/v1/TruthfulQA.csv"))
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--folds", type=int, nargs="+", default=[0, 1])
    parser.add_argument("--margins", type=float, nargs="+", default=DEFAULT_MARGINS)
    parser.add_argument("--layer", type=int, default=14)
    parser.add_argument("--weight_kappa", type=float, default=3.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max_questions_per_fold", type=int, default=0)
    parser.add_argument("--dtype", choices=["float16", "bfloat16", "float32"], default="float16")
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    if len(set(args.folds)) != len(args.folds) or not set(args.folds) <= {0, 1}:
        raise ValueError("Folds must be a unique subset of 0,1")
    if len(set(args.margins)) != len(args.margins):
        raise ValueError("Margins must be unique")

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=False)
    set_seed(args.seed)
    frame, hf_dataset = load_truthfulqa_mc_dataframe(args.csv_path)
    checkpoints = [
        (fold, Path(args.checkpoint_dir) / f"{args.model_name}_layer{args.layer}_predictor_fold{fold}_mean.npz")
        for fold in args.folds
    ]
    test_sets, checkpoint_metadata = validate_splits(checkpoints, len(hf_dataset))
    mc1_by_fold = {fold: load_mc1_by_margin(Path(args.eval_root), args.model_name, fold) for fold in args.folds}
    for fold in args.folds:
        missing = set(args.margins) - set(mc1_by_fold[fold])
        if missing:
            raise FileNotFoundError(f"Missing existing MC1 results for fold {fold}: {sorted(missing)}")

    write_json(out / "config.json", {
        "model_name": args.model_name,
        "model_dir": args.model_dir,
        "checkpoint_dir": args.checkpoint_dir,
        "eval_root": args.eval_root,
        "csv_path": args.csv_path,
        "output_dir": str(out),
        "folds": args.folds,
        "margins": args.margins,
        "layer": args.layer,
        "weight_kappa": args.weight_kappa,
        "seed": args.seed,
        "max_questions_per_fold": args.max_questions_per_fold,
        "dtype": args.dtype,
        "started_utc": datetime.now(timezone.utc).isoformat(),
        "checkpoint_metadata": checkpoint_metadata,
        "csv_sha256": digest(Path(args.csv_path)),
    })

    tokenizer = AutoTokenizer.from_pretrained(args.model_dir, local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_dir,
        torch_dtype=getattr(torch, args.dtype),
        device_map={"": 0},
        local_files_only=True,
    ).eval()
    model.config.use_cache = False
    records = []
    with torch.inference_mode():
        for (fold, checkpoint), test in zip(checkpoints, test_sets):
            data, avg, predictor = load_components(str(checkpoint), torch.device("cuda:0"))
            selected = frame[frame.hf_idx.isin(test)].sort_values("hf_idx")
            if args.max_questions_per_fold:
                selected = selected.head(args.max_questions_per_fold)
            for row in tqdm(selected.to_dict("records"), desc=f"{args.model_name} fold{fold}"):
                for label, choices in (("correct", split_multi_answer(row["Correct Answers"])), ("incorrect", split_multi_answer(row["Incorrect Answers"]))):
                    for index, answer in enumerate(choices):
                        recorder = MultiMarginGateRecorder(avg, predictor, args.weight_kappa, args.margins)
                        answer_ids = tokenizer(" " + answer, add_special_tokens=False).input_ids
                        calculate_scores_zeroshot(
                            model=model,
                            tokenizer=tokenizer,
                            question=row["Question"],
                            choices=[answer],
                            base_hook_fn=recorder.as_hook(),
                            layer_name=get_layer_name(args.model_name, args.layer),
                            device=torch.device("cuda:0"),
                            use_instruction=True,
                            model_name=args.model_name,
                        )
                        if len(recorder.records) != 1:
                            raise AssertionError("Expected one forward per candidate")
                        records.append({
                            "fold": fold,
                            "hf_idx": int(row["hf_idx"]),
                            "question": row["Question"],
                            "category": row["Category"],
                            "label": label,
                            "candidate_index": index,
                            "answer": answer,
                            "best_answer": format_best(row["Best Answer"]),
                            "answer_tokens": len(answer_ids),
                            **recorder.records[0],
                        })
            data.close()

    summaries, summary_rows = aggregate(records, args.margins, mc1_by_fold)
    write_json(out / "summary.json", {
        "model_name": args.model_name,
        "margins": args.margins,
        "primary_metric": "token-weighted answer-token trigger rate by label; question-balanced candidate mean also reported",
        "summaries": summaries,
        "native_hook_checks": all(record["native_count_check"] for record in records),
        "records": len(records),
    })
    write_csv(out / "summary.csv", summary_rows)
    (out / "candidates.jsonl").write_text("\n".join(json.dumps(record, ensure_ascii=False) for record in records) + "\n")
    plot(summaries, args.margins, out / "rebuttal_R3Q3_margin_gates", args.model_name)
    report({args.model_name: summaries}, args.margins, out / "REPORT.md")
    write_json(out / "status.json", {"state": "complete", "questions": len({(r['fold'], r['hf_idx']) for r in records}), "records": len(records)})


if __name__ == "__main__":
    main()
