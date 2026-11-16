"""Compare actual generic MLP fitting splits with official evaluation splits."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from experiments.rebuttal_prediction_error.rebuttal_prediction_error_run import MODEL, LAYERS, evaluate, write_csv
from experiments.rebuttal_prediction_error.extract_official import SPLITS


def main():
    p = argparse.ArgumentParser(__doc__)
    p.add_argument("--output", type=Path, default=ROOT / "results/experiment/rebuttal_prediction_error/official")
    p.add_argument("--device", default="cuda")
    p.add_argument("--skip-extract", action="store_true")
    p.add_argument("--prepare-only", action="store_true", help="Train missing MLPs without feature extraction/evaluation")
    args = p.parse_args()
    torch.set_num_threads(4)
    args.output.mkdir(parents=True, exist_ok=True)
    if not args.skip_extract and not args.prepare_only:
        subprocess.run([sys.executable, str(Path(__file__).with_name("extract_official.py")),
                        "--output", str(args.output / "features")], check=True)
    if not args.prepare_only:
        audit_path = args.output / "training_feature_audit.json"
        if not audit_path.exists():
            subprocess.run([sys.executable, str(Path(__file__).with_name("extract_official.py")),
                            "--output", str(args.output / "features"), "--audit-only"], check=True)
        audit = json.loads(audit_path.read_text())
        if {row["dataset"] for row in audit if row["passed"]} != set(LAYERS):
            raise ValueError("Training feature audit did not pass for every dataset")
    jobs, seen = [], set()
    for dataset, layer in LAYERS.items():
        feature = ROOT / f"results/reproduction/generic/features/S2S_R/{MODEL}/{MODEL}_{dataset}_train_l{layer}.npz"
        ckpt = ROOT / f"results/reproduction/generic/checkpoints/S2S_R/{MODEL}/{MODEL}_{dataset}_l{layer}_S2S_R.npz"
        if not ckpt.exists():
            ckpt = args.output / f"checkpoints/{dataset}_production_e2000.npz"
            ckpt.parent.mkdir(exist_ok=True)
            if not ckpt.exists():
                print(f"Training missing production MLP: {dataset}", flush=True)
                with ckpt.with_suffix(".log").open("w") as log:
                    subprocess.run([sys.executable, str(ROOT / "method/S2S_R/train_predictor.py"),
                        "--feature_path", str(feature), "--output_path", str(ckpt),
                        "--num_folds", "1", "--fold_idx", "0", "--val_fraction", "0.5",
                        "--mlp_hidden", "256", "--epochs", "2000", "--batch_size", "256",
                        "--learning_rate", "0.001", "--seed", "42", "--device", args.device],
                        cwd=ROOT, check=True, stdout=log, stderr=subprocess.STDOUT)
        test_feature = args.output / f"features/{dataset}_official_l{layer}.npz"
        if not args.prepare_only:
            with np.load(test_feature, allow_pickle=False) as official:
                if (str(official["source_split"].item()) != SPLITS[dataset]
                        or str(official["model_name"].item()) != MODEL
                        or int(official["layer"]) != layer):
                    raise ValueError(f"Wrong official feature provenance: {test_feature}")
                expected = {"boolq": 3270, "copa": 100, "storycloze": 1511,
                            "mmlu_global": 2000, "winogrande": 1267}[dataset]
                if len(np.unique(official["q_indices"])) != expected:
                    raise ValueError(f"Incomplete official split: {test_feature}")
        jobs.append((dataset, layer, 0, ckpt, feature, test_feature))
        if layer not in seen:
            for fold in (0, 1):
                source = ROOT / f"results/experiment/rebuttal_R2Q2_cross_task_transfer/{MODEL}/{dataset}/checkpoints/truthfulqa_fold{fold}.npz"
                jobs.append(("truthfulqa", layer, fold, source,
                             ROOT / f"results/reproduction/features/{MODEL}_layer{layer}.npz", None))
            seen.add(layer)
    for fold in (0, 1):
        source = ROOT / f"results/reproduction/checkpoints/S2S_R/truthfulqa/{MODEL}/{MODEL}_layer14_predictor_fold{fold}_mean.npz"
        jobs.append(("truthfulqa", 14, fold, source, ROOT / f"results/reproduction/features/{MODEL}_layer14.npz", None))
    if args.prepare_only:
        return
    rows, questions = [], []
    for dataset, layer, fold, ckpt, feature, test_feature in jobs:
        print(f"Evaluating {dataset} L{layer} fold{fold}", flush=True)
        rr, qq = evaluate(ckpt, feature, args.device, test_feature=test_feature)
        tags = {"dataset": dataset, "layer": layer, "fold": fold}
        rows.extend({**tags, **r} for r in rr)
        questions.extend({**tags, **r} for r in qq)
    write_csv(args.output / "fold_metrics.csv", rows)
    write_csv(args.output / "question_metrics.csv", questions)
    metrics = ["angle_deg", "norm_abs_error", "norm_rel_error_pct", "norm_signed_error",
               "target_norm", "prediction_norm", "vector_l2_error", "vector_mse", "residual_angle_deg"]
    summary = []
    for dataset, layer in dict.fromkeys((r["dataset"], r["layer"]) for r in rows):
        for split in ("mlp_train", "test"):
            selected = [r for r in rows if (r["dataset"], r["layer"], r["split"]) == (dataset, layer, split)]
            row = {"dataset": dataset, "layer": layer, "split": split,
                   "folds": len(selected), "question_evaluations": sum(r["questions"] for r in selected),
                   "pair_evaluations": sum(r["pairs"] for r in selected)}
            for metric in metrics:
                values = [r[metric] for r in selected if r[metric] is not None]
                row[metric] = float(np.mean(values)) if values else None
            summary.append(row)
    write_csv(args.output / "summary.csv", summary)
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False))
    lines = ["# MLP training / official evaluation geometry", "",
        "Model: Llama-3.1-8B-Instruct. All MLPs: width 256, 2000 epochs, seed 42, mean-positive targets. Generic: production one-fold setup, 50% cached training questions fit MLP and 50% estimate v_avg; official evaluation features never fit MLP or mean. BoolQ/COPA/Winogrande: full official validation; StoryCloze: full official eval. MMLU: repository balanced test subset, 500 questions per category from shuffled official test rows 500:1000, disjoint from training rows 0:500. TruthfulQA: existing two-fold held-out testing within official validation (no separate labeled test split); 25% of all questions fit each MLP, 25% estimate its mean and 50% test. Training sizes therefore differ between generic and TruthfulQA. Within-question means, then equal question weighting; TruthfulQA folds averaged equally. Absolute norm errors depend on activation scale; compare relative errors and angles at the same layer.", "",
        "| Dataset | Layer | Split | Questions evaluated | Angle (deg) | Norm MAE | Norm MAPE (%) | Residual angle (deg) |",
        "|---|---:|---|---:|---:|---:|---:|---:|"]
    for r in summary:
        lines.append(f"| {r['dataset']} | {r['layer']} | {r['split']} | {r['question_evaluations']} | {r['angle_deg']:.3f} | {r['norm_abs_error']:.3f} | {r['norm_rel_error_pct']:.3f} | {r['residual_angle_deg']:.3f} |")
    (args.output / "report.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines), flush=True)


if __name__ == "__main__":
    main()
