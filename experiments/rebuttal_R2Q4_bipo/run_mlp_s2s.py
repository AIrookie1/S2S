"""Use each trained BiPO vector as v_avg for MLP-based S2S-L/R."""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import numpy as np
import torch

from experiments.rebuttal_R2Q4_bipo.rebuttal_R2Q4_run import (
    MODEL,
    evaluate,
    load_truthfulqa_mc_dataframe,
    make_hook,
    tail_model,
    write_json,
)
from methods.S2S_R.predictor import DirectVectorMLP, fit_dynamic_mlp, load_feature_pairs


def train_fold(feature: Path, folder: Path, split: dict, args) -> DirectVectorMLP:
    checkpoint = folder / "mlp_predictor.npz"
    h_neg, h_pos, q_indices, metadata = load_feature_pairs(feature, positive_mode="cartesian")
    train_q = np.asarray(split["train"], dtype=np.int64)
    train_mask = np.isin(q_indices, train_q)
    if not train_mask.any() or not set(np.unique(q_indices[train_mask])).issuperset(train_q):
        raise ValueError("MLP training features do not cover the BiPO optimization split")

    if checkpoint.exists():
        with np.load(checkpoint, allow_pickle=False) as saved:
            if not np.array_equal(saved["train_q_indices"], train_q):
                raise ValueError(f"MLP checkpoint split mismatch: {checkpoint}")
            hidden_dim = int(saved["hidden_dim"])
            mlp_hidden = int(saved["mlp_hidden"])
            state = {key.removeprefix("mlp_"): torch.from_numpy(saved[key].copy())
                     for key in saved.files if key in {
                         "mlp_fc1.weight", "mlp_fc1.bias", "mlp_fc2.weight", "mlp_fc2.bias"
                     }}
    else:
        # v_avg from this trainer is discarded: S2S receives the learned BiPO vector.
        state, history, metrics = fit_dynamic_mlp(
            h_neg,
            h_pos,
            q_indices,
            train_mask,
            train_mask,
            mlp_hidden=args.mlp_hidden,
            epochs=args.epochs,
            batch_size=args.train_batch_size,
            learning_rate=args.learning_rate,
            device="cuda",
            seed=args.seed + int(split["fold"]),
        )
        hidden_dim = int(state["hidden_dim"])
        mlp_hidden = int(state["mlp_hidden"])
        state_dict = state["predictor"].state_dict()
        np.savez_compressed(
            checkpoint,
            hidden_dim=np.asarray(hidden_dim),
            mlp_hidden=np.asarray(mlp_hidden),
            train_q_indices=train_q,
            mlp_train_q_indices=np.unique(q_indices[train_mask]),
            positive_mode=np.asarray("cartesian"),
            target_definition=np.asarray("h_pos_raw - h_neg_raw"),
            input_definition=np.asarray("normalize(h_neg_raw)"),
            history_json=np.asarray(json.dumps(history)),
            metrics_json=np.asarray(json.dumps(metrics)),
            **{f"mlp_{key}": value.detach().cpu().numpy()
               for key, value in state_dict.items()},
        )
        state = {key: value.detach().cpu() for key, value in state_dict.items()}

    predictor = DirectVectorMLP(hidden_dim, mlp_hidden, hidden_dim)
    predictor.load_state_dict(state)
    predictor.eval()
    return predictor


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=ROOT / "results/experiment/rebuttal_R2Q4_bipo")
    parser.add_argument("--feature", type=Path, default=ROOT / "features/llama3.1-8B-Instruct_layer14.npz")
    parser.add_argument("--model-dir", default=__import__("os").environ.get("LLAMA_MODEL_DIR", "models/Meta-Llama-3.1-8B-Instruct"))
    parser.add_argument("--epochs", type=int, default=2000)
    parser.add_argument("--mlp-hidden", type=int, default=256)
    parser.add_argument("--train-batch-size", type=int, default=256)
    parser.add_argument("--eval-batch-size", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if args.epochs <= 0 or args.mlp_hidden <= 0 or args.eval_batch_size <= 0:
        parser.error("epochs, hidden width and batch sizes must be positive")
    if not (args.input / "prefix_cache.pt").is_file():
        parser.error("BiPO run must finish first and provide prefix_cache.pt")

    output = args.input / "mlp_s2s_retuned"
    output.mkdir(parents=True, exist_ok=True)
    config = {
        "model": MODEL,
        "layer": 14,
        "bipo_run": str(args.input.resolve()),
        "feature": str(args.feature.resolve()),
        "feature_sha256": __import__("hashlib").sha256(args.feature.read_bytes()).hexdigest(),
        "epochs": args.epochs,
        "mlp_hidden": args.mlp_hidden,
        "train_batch_size": args.train_batch_size,
        "learning_rate": args.learning_rate,
        "seed": args.seed,
        "v_avg_source": "same-fold BiPO optimized vector",
        "predictor_target": "Cartesian raw h_pos - h_neg, training questions only",
        "selection": "same-fold development MC1; ties broken by MC2, separately per method",
        "all_token_intervention": True,
    }
    manifest = output / "config.json"
    if manifest.exists() and json.loads(manifest.read_text()) != config:
        raise ValueError("MLP-S2S output configuration mismatch")
    write_json(manifest, config)

    fold_splits = [json.loads((args.input / f"fold{fold}" / "split.json").read_text()) for fold in (0, 1)]
    predictors = [train_fold(args.feature, args.input / f"fold{fold}", split, args)
                  for fold, split in enumerate(fold_splits)]

    torch.set_num_threads(4)
    model = __import__("transformers").AutoModelForCausalLM.from_pretrained(
        args.model_dir,
        torch_dtype=torch.float16,
        attn_implementation="sdpa",
        device_map={
            "model.embed_tokens": 0,
            **{f"model.layers.{i}": ("cpu" if i <= 14 else 0) for i in range(32)},
            "model.norm": 0,
            "lm_head": 0,
        },
        low_cpu_mem_usage=True,
        local_files_only=True,
    ).eval()
    tokenizer = __import__("transformers").AutoTokenizer.from_pretrained(args.model_dir, local_files_only=True)
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    records = torch.load(args.input / "prefix_cache.pt", weights_only=False, map_location="cpu")
    cache = {tuple(record["ids"].tolist()): record["hidden"] for record in records}
    frame, _ = load_truthfulqa_mc_dataframe("./data/truthfulqa/TruthfulQA.csv")
    block = model.model.layers[14]
    block_device = model.device
    predictors = [predictor.to(block_device).eval() for predictor in predictors]
    summaries = []
    for fold, (split, predictor) in enumerate(zip(fold_splits, predictors)):
        source_folder = args.input / f"fold{fold}"
        folder = output / f"fold{fold}"
        folder.mkdir(exist_ok=True)
        bipo = torch.load(source_folder / "bipo.pt", weights_only=False, map_location="cpu")["vector"].to(block_device)
        if not torch.isfinite(bipo).all() or not torch.isfinite(predictor.fc2.weight).all():
            raise ValueError("Non-finite BIPO vector or MLP weights")
        cases = {
            "bipo_mlp_s2s_l": [
                dict(strength=strength, slerp_t=slerp_t)
                for strength in (0.5, 1.0, 2.0, 3.0)
                for slerp_t in (0.0, 0.2, 0.5, 0.8, 1.0)
            ],
            "bipo_mlp_s2s_r": [
                dict(weight_kappa=kappa, margin=margin)
                for kappa in (1.0, 3.0, 5.0)
                for margin in (-0.7, -0.3, 0.0)
            ],
        }
        fold_rows = []
        for method, grid in cases.items():
            candidates = []
            for index, params in enumerate(grid):
                hook = make_hook("s2s_l" if method.endswith("s2s_l") else "s2s_r",
                                 bipo, bipo, params, predictor=predictor)
                metrics = evaluate(
                    model, tokenizer, block, frame, split["dev"], hook,
                    folder / f"dev_retuned_{method}_{index}.json", args.eval_batch_size, cache,
                )
                candidates.append(dict(parameters=params, **metrics))
            chosen = max(candidates, key=lambda item: (item["MC1"], item["MC2"]))
            params = chosen["parameters"]
            dev_metrics = {metric: chosen[metric] for metric in ("MC1", "MC2", "MC3")}
            write_json(output / f"fold{fold}_{method}_selection.json", {
                "candidates": candidates,
                "chosen": chosen,
                "selection_metric": "development MC1, then MC2",
            })
            hook = make_hook("s2s_l" if method.endswith("s2s_l") else "s2s_r",
                             bipo, bipo, params, predictor=predictor)
            metrics = evaluate(
                model, tokenizer, block, frame, split["test"], hook,
                folder / f"test_retuned_{method}.json", args.eval_batch_size, cache,
            )
            row = dict(method=method, fold=fold, n_questions=len(split["test"]),
                       parameters=params, development=dev_metrics, **metrics)
            fold_rows.append(row)
            print("TEST", row, flush=True)
            write_json(folder / "mlp_s2s_test_summary.json", fold_rows)
        summaries.extend(fold_rows)

    pooled = []
    for method in ("bipo_mlp_s2s_l", "bipo_mlp_s2s_r"):
        rows = [row for row in summaries if row["method"] == method]
        pooled.append(dict(method=method, n_questions=sum(row["n_questions"] for row in rows),
                           **{metric: sum(row[metric] * row["n_questions"] for row in rows) /
                              sum(row["n_questions"] for row in rows)
                              for metric in ("MC1", "MC2", "MC3")}))
    write_json(output / "performance.json", pooled)
    lines = ["# Retuned BiPO mean vector + trained MLP S2S results", "",
             "The fold's optimized BiPO vector is used as `v_avg`; the MLP predicts raw Cartesian answer-difference vectors.",
             "The MLP uses training questions only. Each fold selects S2S parameters on its development split; test data remains held out.", "",
             "| Method | MC1 (%) | MC2 (%) | MC3 (%) | N |", "|---|---:|---:|---:|---:|"]
    for row in pooled:
        lines.append(f"| {row['method']} | " + " | ".join(f"{100*row[m]:.2f}" for m in ("MC1", "MC2", "MC3")) + f" | {row['n_questions']} |")
    lines += ["", "## Fold selections", "", "| Fold | Method | Parameters | Dev MC1 | Dev MC2 | Test MC1 | Test MC2 | Test MC3 |", "|---|---|---|---:|---:|---:|---:|---:|"]
    for row in summaries:
        params = json.dumps(row["parameters"], sort_keys=True)
        lines.append(f"| {row['fold']} | {row['method']} | `{params}` | {100*row['development']['MC1']:.2f} | {100*row['development']['MC2']:.2f} | {100*row['MC1']:.2f} | {100*row['MC2']:.2f} | {100*row['MC3']:.2f} |")
    (output / "REPORT.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines), flush=True)


if __name__ == "__main__":
    main()
