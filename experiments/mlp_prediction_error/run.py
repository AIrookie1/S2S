"""Question-disjoint direct-MLP train/test geometry on cached features."""
from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from method.dyn3d_sga.dyn_mlp import DirectVectorMLP, load_feature_pairs

MODEL = "llama3.1-8B-Instruct"
LAYERS = {"boolq": 12, "copa": 16, "storycloze": 27,
          "mmlu_global": 28, "winogrande": 16}


def geometry(pred, target, avg):
    """Per-pair metrics; undefined angles/relative errors are NaN, never zero."""
    if not (torch.isfinite(pred).all() and torch.isfinite(target).all() and torch.isfinite(avg).all()):
        raise ValueError("Non-finite vector in geometry evaluation")
    pred, target = pred.double(), target.double()
    pn, tn = pred.norm(dim=1), target.norm(dim=1)

    def angles(x, y):
        xn, yn = x.norm(dim=1), y.norm(dim=1)
        valid = (xn > 1e-8) & (yn > 1e-8)
        out = torch.full_like(xn, float("nan"))
        out[valid] = torch.rad2deg(torch.acos(
            ((x[valid] * y[valid]).sum(1) / (xn[valid] * yn[valid])).clamp(-1, 1)))
        return out

    ae = (pn - tn).abs()
    rel = torch.full_like(tn, float("nan"))
    rel[tn > 1e-8] = 100 * ae[tn > 1e-8] / tn[tn > 1e-8]
    return {k: v.cpu().numpy() for k, v in {
        "angle_deg": angles(pred, target),
        "norm_abs_error": ae,
        "norm_rel_error_pct": rel,
        "norm_signed_error": pn - tn,
        "target_norm": tn,
        "prediction_norm": pn,
        "vector_l2_error": (pred - target).norm(dim=1),
        "vector_mse": ((pred - target) ** 2).mean(1),
        "residual_angle_deg": angles(pred - avg.double(), target - avg.double()),
    }.items()}


def evaluate(path, feature, device, test_feature=None):
    with np.load(path, allow_pickle=False) as d:
        neg, pos, q, _ = load_feature_pairs(feature, str(d["positive_mode"].item()))
        groups = {k: np.asarray(d[k + "_q_indices"], dtype=np.int64)
                  for k in ("mlp_train", "mean_train", "test")}
        required_groups = ("mlp_train", "mean_train") if test_feature else tuple(groups)
        if any(len(groups[k]) == 0 for k in required_groups):
            raise ValueError(f"Empty split in {path}")
        for a, b in (("mlp_train", "test"), ("mean_train", "test"),
                     ("mlp_train", "mean_train")):
            if np.intersect1d(groups[a], groups[b]).size:
                raise ValueError(f"Overlapping {a}/{b} questions in {path}")
        if not all(np.isin(v, q).all() for v in groups.values()):
            raise ValueError(f"Checkpoint questions absent from features: {path}")
        net = DirectVectorMLP(int(d["hidden_dim"]), int(d["mlp_hidden"])).to(device)
        net.load_state_dict({k: torch.from_numpy(d["mlp_" + k])
                             for k in net.state_dict()})
        avg = torch.tensor(d["v_avg"], device=device)
        stored_mse = json.loads(str(d["metrics_json"].item()))["train_vector_mse"]
    net.eval()
    records, question_records = [], []
    for split in ("mlp_train", "test"):
        source = feature
        if split == "test" and test_feature is not None:
            neg, pos, q, _ = load_feature_pairs(test_feature, "mean")
            source = test_feature
            mask = np.ones(len(q), dtype=bool)
        else:
            mask = np.isin(q, groups[split])
        ids = np.flatnonzero(mask)
        chunks = []
        with torch.inference_mode():
            for start in range(0, len(ids), 256):
                idx = ids[start:start + 256]
                x = torch.tensor(neg[idx], device=device)
                y = torch.tensor(pos[idx] - neg[idx], device=device)
                chunks.append(geometry(net(F.normalize(x, dim=1)), y, avg))
        values = {k: np.concatenate([c[k] for c in chunks]) for k in chunks[0]}
        pair_mse = float(np.mean(values["vector_mse"]))
        if split == "mlp_train" and not np.isclose(pair_mse, stored_mse, rtol=1e-3, atol=1e-7):
            raise ValueError(f"Recomputed training MSE {pair_mse} != checkpoint {stored_mse}")
        question_values = []
        for qi in np.unique(q[mask]):
            qm = q[mask] == qi
            row = {"split": split, "question": int(qi), "pairs": int(qm.sum())}
            for k, v in values.items():
                valid = np.isfinite(v[qm])
                row[k] = float(v[qm][valid].mean()) if valid.any() else None
            question_records.append(row)
            question_values.append(row)
        row = {"split": split, "questions": len(question_values), "pairs": len(ids),
               "checkpoint": str(path), "feature_path": str(source),
               "pair_weighted_vector_mse": pair_mse,
               "checkpoint_train_mse": stored_mse}
        for k, v in values.items():
            vals = [r[k] for r in question_values if r[k] is not None]
            row[k] = float(np.mean(vals)) if vals else None
            row[k + "_valid_pairs"] = int(np.isfinite(v).sum())
        records.append(row)
    return records, question_records


def write_csv(path, rows):
    with path.open("w") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main():
    p = argparse.ArgumentParser(__doc__)
    p.add_argument("--datasets", nargs="+", default=list(LAYERS), choices=list(LAYERS))
    p.add_argument("--output", type=Path, default=ROOT / "results/experiment/mlp_prediction_error")
    p.add_argument("--device", default="cuda")
    p.add_argument("--epochs", type=int, default=2000)
    args = p.parse_args()
    torch.set_num_threads(4)
    args.output.mkdir(parents=True, exist_ok=True)
    rows, questions, done_truthful = [], [], set()
    jobs = []
    for dataset in args.datasets:
        layer = LAYERS[dataset]
        feature = ROOT / f"features/generic/dyn3d_sga/{MODEL}/{MODEL}_{dataset}_train_l{layer}.npz"
        for fold in (0, 1):
            ckpt = args.output / f"checkpoints/{dataset}_fold{fold}_e{args.epochs}.npz"
            ckpt.parent.mkdir(exist_ok=True)
            if not ckpt.exists():
                print(f"Training {dataset}, fold {fold}; {args.epochs} epochs", flush=True)
                with ckpt.with_suffix(".log").open("w") as log:
                    subprocess.run([sys.executable, str(ROOT / "method/dyn3d_sga/train_dyn_mlp.py"),
                        "--feature_path", str(feature), "--output_path", str(ckpt),
                        "--num_folds", "2", "--fold_idx", str(fold), "--val_fraction", "0.5",
                        "--mlp_hidden", "256", "--epochs", str(args.epochs), "--batch_size", "256",
                        "--learning_rate", "0.001", "--seed", "42", "--device", args.device],
                        check=True, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT,
                        env={**os.environ, "OMP_NUM_THREADS": "4", "MKL_NUM_THREADS": "4"})
            jobs.append((dataset, layer, fold, ckpt, feature))
        if layer not in done_truthful:
            for fold in (0, 1):
                ckpt = ROOT / f"results/experiment/truthfulqa_generic_transfer/{MODEL}/{dataset}/checkpoints/truthfulqa_fold{fold}.npz"
                jobs.append(("truthfulqa", layer, fold, ckpt,
                             ROOT / f"features/{MODEL}_layer{layer}.npz"))
            done_truthful.add(layer)
    # Include the released TruthfulQA layer as an additional reference.
    for fold in (0, 1):
        ckpt = ROOT / f"checkpoints/dyn3d_sga/truthfulqa/{MODEL}/{MODEL}_layer14_dynMLP_fold{fold}_mean.npz"
        jobs.append(("truthfulqa", 14, fold, ckpt, ROOT / f"features/{MODEL}_layer14.npz"))
    for dataset, layer, fold, ckpt, feature in jobs:
        print(f"Evaluating {dataset} L{layer} fold{fold}", flush=True)
        rr, qq = evaluate(ckpt, feature, args.device)
        tags = {"dataset": dataset, "layer": layer, "fold": fold}
        rows.extend({**tags, **r} for r in rr)
        questions.extend({**tags, **r} for r in qq)
    write_csv(args.output / "fold_metrics.csv", rows)
    write_csv(args.output / "question_metrics.csv", questions)
    summary = []
    metrics = ["angle_deg", "norm_abs_error", "norm_rel_error_pct", "norm_signed_error",
               "target_norm", "prediction_norm", "vector_l2_error", "vector_mse", "residual_angle_deg"]
    for dataset, layer in dict.fromkeys((r["dataset"], r["layer"]) for r in rows):
        for split in ("mlp_train", "test"):
            selected = [r for r in rows if (r["dataset"], r["layer"], r["split"]) == (dataset, layer, split)]
            item = {"dataset": dataset, "layer": layer, "split": split,
                    "folds": len(selected), "question_evaluations": sum(r["questions"] for r in selected),
                    "pair_evaluations": sum(r["pairs"] for r in selected)}
            for k in metrics:
                vals = [r[k] for r in selected if r[k] is not None]
                item[k] = float(np.mean(vals)) if vals else None
            summary.append(item)
    write_csv(args.output / "summary.csv", summary)
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False))
    report = ["# Direct-MLP train / held-out geometry", "",
        "Llama-3.1-8B-Instruct; two outer question folds; 50% of each outer training fold fits the MLP, 50% estimates v_avg. Test questions are disjoint from both. Generic held-out tests come from the cached train split, not official validation/test. Targets are mean-positive h_pos minus h_neg. Metrics average negatives within question, then questions within fold, then the two folds equally. Absolute norms retain raw activation units. No early stopping or test-set model selection.", "",
        "| Dataset | Layer | Split | Angle (deg) | Norm MAE | Norm MAPE (%) | Residual angle (deg) |", "|---|---:|---|---:|---:|---:|---:|"]
    for r in summary:
        report.append(f"| {r['dataset']} | {r['layer']} | {r['split']} | {r['angle_deg']:.3f} | {r['norm_abs_error']:.3f} | {r['norm_rel_error_pct']:.3f} | {r['residual_angle_deg']:.3f} |")
    (args.output / "report.md").write_text("\n".join(report) + "\n")
    print("\n".join(report), flush=True)


if __name__ == "__main__":
    main()
