"""Summarize the two TruthfulQA source folds on each generic dataset."""
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
import statistics

import numpy as np


def summarize(evaluations: list[Path]) -> dict:
    rows = []
    identity = None
    for path in evaluations:
        result = json.loads(path.read_text(encoding="utf-8"))
        score = float(result["MC1"])
        if not math.isfinite(score) or not 0 <= score <= 1 or not result["steering_enabled"]:
            raise ValueError(f"Invalid steered MC1 result: {path}")
        checkpoint = Path(result["checkpoint_path"])
        with np.load(checkpoint, allow_pickle=False) as data:
            if str(data["experiment"].item()) != "truthfulqa_generic_transfer":
                raise ValueError("Expected an experiment transfer checkpoint")
            fold = int(data["fold_idx"].item())
            current = (result["model_name"], result["dataset"], result["layer"], result["parameters"])
            if identity is not None and identity != current:
                raise ValueError("Fold evaluations use different model/dataset/layer/parameters")
            identity = current
            if (result["model_name"] != str(data["model_name"].item())
                    or result["dataset"] != str(data["target_dataset"].item())
                    or result["layer"] != int(data["layer"].item())):
                raise ValueError("Evaluation and checkpoint metadata disagree")
            if (str(data["v_avg_source_dataset"].item()) != "truthful_qa"
                    or str(data["mlp_source_dataset"].item()) != "truthful_qa"):
                raise ValueError("Both the MLP and v_avg must come from TruthfulQA")
            source_path = Path(str(data["mlp_source_checkpoint"].item()))
            with np.load(source_path, allow_pickle=False) as source:
                if int(source["fold_idx"].item()) != fold or int(source["num_folds"].item()) != 2:
                    raise ValueError("Transfer checkpoint has the wrong TruthfulQA source fold")
                for key in ("v_avg", "mean_train_q_indices", "mlp_train_q_indices",
                            "mlp_fc1.weight", "mlp_fc1.bias", "mlp_fc2.weight", "mlp_fc2.bias"):
                    if not np.array_equal(data[key], source[key]):
                        raise ValueError(f"Transfer checkpoint changed source TruthfulQA {key}")
            rows.append({"source_fold": fold, "MC1": score,
                         "evaluation": str(path.resolve()), "checkpoint": str(checkpoint.resolve()),
                         "mlp_source_checkpoint": str(data["mlp_source_checkpoint"].item()),
                         "v_avg_feature_path": str(data["v_avg_feature_path"].item()),
                         "truthfulqa_mean_questions": len(data["mean_train_q_indices"])})
    if sorted(row["source_fold"] for row in rows) != [0, 1]:
        raise ValueError("Exactly one result for each TruthfulQA source fold 0 and 1 is required")
    rows.sort(key=lambda row: row["source_fold"])
    scores = [row["MC1"] for row in rows]
    return {"experiment": "truthfulqa_generic_transfer", "method": "S2S_R",
            "model": identity[0], "dataset": identity[1], "layer": identity[2],
            "parameters": identity[3], "mlp_source_dataset": "truthful_qa",
            "v_avg_source_dataset": "truthful_qa",
            "aggregation": "arithmetic mean and sample SD over TruthfulQA source folds 0 and 1",
            "MC1_mean": statistics.mean(scores), "MC1_std": statistics.stdev(scores), "folds": rows}


def write_outputs(summaries: list[dict], output_dir: Path):
    output_dir.mkdir(parents=True, exist_ok=True)
    payload = summaries[0] if len(summaries) == 1 else {
        "experiment": "truthfulqa_generic_transfer", "results": summaries}
    (output_dir / "summary.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    with (output_dir / "summary.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["model", "dataset", "layer", "weight_kappa", "margin", "fold0_MC1", "fold1_MC1", "MC1_mean", "MC1_std"])
        for item in summaries:
            scores = {row["source_fold"]: row["MC1"] for row in item["folds"]}
            writer.writerow([item["model"], item["dataset"], item["layer"],
                             item["parameters"]["weight_kappa"], item["parameters"]["margin"],
                             scores[0], scores[1], item["MC1_mean"], item["MC1_std"]])
            print(f"{item['dataset']}: MC1={item['MC1_mean']:.6f} +/- {item['MC1_std']:.6f}")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    group = p.add_mutually_exclusive_group(required=True)
    group.add_argument("--evaluations", type=Path, nargs="+")
    group.add_argument("--summaries", type=Path, nargs="+")
    p.add_argument("--output-dir", type=Path, required=True)
    args = p.parse_args()
    if args.evaluations:
        summaries = [summarize(args.evaluations)]
    else:
        summaries = [json.loads(path.read_text(encoding="utf-8")) for path in args.summaries]
    write_outputs(summaries, args.output_dir)


if __name__ == "__main__":
    main()
