#!/usr/bin/env python3
"""Focused cosine evaluation for L2S and adaptive L2S-SLERP.

For every outer fold this script reuses the trained L2S MLP and computes
``v_avg`` only from ground-truth raw ``v_T`` pairs whose question IDs belong to
``geometry_train_q_indices``.

Methods:

* L2S: ``cosine(L2S(z_neg_raw), v_T)``
* Adaptive L2S-SLERP:

  ``v_hat = L2S(z_neg_raw)``
  ``r = normalize(v_hat - v_avg)``
  ``(w_h,w_m,w_r) = adaptive_3way_weights(normalize(z_neg_raw), normalize(v_avg), r)``
  ``t = w_r / (w_m + w_r)``
  ``v_pred = ||v_hat|| * SLERP(normalize(v_avg), r, t)``

There is no additive steering alpha in this direction-only experiment and no
manual ``t`` sweep.  ``weight_kappa`` is the same adaptive-weight sharpness
used by the repository's three-sphere interpolation.
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
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from method.l2s.model import load_l2s_checkpoint
from method.l2s_slerp.steering import l2s_prediction_to_slerp


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Evaluate L2S vs adaptive L2S-SLERP raw-v_T cosine similarity")
    p.add_argument("--feature_path", required=True)
    p.add_argument("--l2s_checkpoint_template", required=True, help="Path template containing {fold}")
    p.add_argument("--folds", type=int, nargs="+", default=[0, 1])
    p.add_argument("--num_folds", type=int, default=2)
    p.add_argument(
        "--weight_kappa",
        type=float,
        default=3.0,
        help="Same adaptive_3way_weights kappa used by the three-sphere method",
    )
    p.add_argument("--batch_size", type=int, default=4096)
    p.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    p.add_argument("--output_dir", default="./results/experiment/cosine_similarity")
    p.add_argument("--output_stem", default="l2s_slerp_adaptive")
    args = p.parse_args()

    if args.num_folds <= 1:
        p.error("--num_folds must be > 1")
    if not args.folds or len(set(args.folds)) != len(args.folds):
        p.error("--folds must contain unique fold indices")
    if any(f < 0 or f >= args.num_folds for f in args.folds):
        p.error("every fold must satisfy 0 <= fold < num_folds")
    if "{fold}" not in args.l2s_checkpoint_template:
        p.error("--l2s_checkpoint_template must contain {fold}")
    if not math.isfinite(args.weight_kappa) or args.weight_kappa <= 0:
        p.error("--weight_kappa must be finite and positive")
    if args.batch_size <= 0:
        p.error("--batch_size must be positive")
    if args.device == "cuda" and not torch.cuda.is_available():
        p.error("--device cuda requested but CUDA is not available")
    return args


def load_features(path: str | Path):
    path = Path(path)
    if not path.is_file() or path.stat().st_size == 0:
        raise FileNotFoundError(f"Missing feature file: {path}")
    with np.load(path, allow_pickle=False) as data:
        required = {"z_neg_raw", "v_T", "vector_q_indices"}
        missing = required.difference(data.files)
        if missing:
            raise KeyError(f"Feature file missing fields: {sorted(missing)}")
        z_neg = np.asarray(data["z_neg_raw"], dtype=np.float32)
        v_t = np.asarray(data["v_T"], dtype=np.float32)
        q = np.asarray(data["vector_q_indices"], dtype=np.int64).reshape(-1)
        if "negative_sample_indices" in data.files:
            negative_ids = np.asarray(data["negative_sample_indices"], dtype=np.int64).reshape(-1)
        else:
            negative_ids = np.arange(len(z_neg), dtype=np.int64)
        meta = {
            k: np.asarray(data[k])
            for k in ("layer", "model_name", "model_path", "vector_definition")
            if k in data.files
        }
    if z_neg.ndim != 2 or v_t.shape != z_neg.shape:
        raise ValueError("z_neg_raw and v_T must have matching [num_pairs, hidden_dim] shapes")
    if len(q) != len(z_neg) or len(negative_ids) != len(z_neg):
        raise ValueError("pair IDs must align with z_neg_raw/v_T")
    if not (np.isfinite(z_neg).all() and np.isfinite(v_t).all()):
        raise ValueError("feature pairs contain non-finite values")
    return z_neg, v_t, q, negative_ids, meta


def render(template: str, fold: int) -> Path:
    return Path(template.format(fold=fold))


def unique_ids(data, key: str, path: Path) -> np.ndarray:
    if key not in data.files:
        raise KeyError(f"Checkpoint {path} missing {key}")
    out = np.sort(np.unique(np.asarray(data[key], dtype=np.int64).reshape(-1)))
    if len(out) == 0:
        raise ValueError(f"Checkpoint {path} has empty {key}")
    return out


def predict(model, x: np.ndarray, device: torch.device, batch_size: int) -> np.ndarray:
    chunks: list[np.ndarray] = []
    model.eval()
    with torch.inference_mode():
        for start in range(0, len(x), batch_size):
            batch = torch.as_tensor(x[start : start + batch_size], dtype=torch.float32, device=device)
            chunks.append(model(batch).detach().float().cpu().numpy())
    return np.concatenate(chunks, axis=0).astype(np.float32, copy=False)


def predict_adaptive_slerp(
    z_neg: np.ndarray,
    direct: np.ndarray,
    v_avg: torch.Tensor,
    *,
    weight_kappa: float,
    device: torch.device,
    batch_size: int,
):
    pred_chunks: list[np.ndarray] = []
    t_chunks: list[np.ndarray] = []
    w_chunks: list[np.ndarray] = []
    with torch.inference_mode():
        for start in range(0, len(direct), batch_size):
            z = torch.as_tensor(z_neg[start : start + batch_size], dtype=torch.float32, device=device)
            d = torch.as_tensor(direct[start : start + batch_size], dtype=torch.float32, device=device)
            pred, info = l2s_prediction_to_slerp(
                z,
                d,
                v_avg,
                weight_kappa=weight_kappa,
                return_info=True,
            )
            pred_chunks.append(pred.detach().float().cpu().numpy())
            t_chunks.append(info["t_adaptive"].detach().float().cpu().numpy().reshape(-1))
            w_chunks.append(info["weights"].detach().float().cpu().numpy().reshape(-1, 3))
    return (
        np.concatenate(pred_chunks, axis=0).astype(np.float32, copy=False),
        np.concatenate(t_chunks, axis=0).astype(np.float64, copy=False),
        np.concatenate(w_chunks, axis=0).astype(np.float64, copy=False),
    )


def cosine_values(pred: np.ndarray, target: np.ndarray) -> np.ndarray:
    p = torch.as_tensor(pred, dtype=torch.float32)
    t = torch.as_tensor(target, dtype=torch.float32)
    return F.cosine_similarity(p, t, dim=-1, eps=1e-8).cpu().numpy().astype(np.float64)


def summary(values: np.ndarray, pred: np.ndarray, target: np.ndarray, negative_ids: np.ndarray) -> dict[str, Any]:
    return {
        "cosine": float(values.mean()),
        "std": float(values.std(ddof=0)),
        "min": float(values.min()),
        "max": float(values.max()),
        "num_pairs": int(len(values)),
        "num_negatives": int(len(np.unique(negative_ids))),
        "zero_prediction_count": int(np.count_nonzero(np.linalg.norm(pred, axis=1) <= 1e-12)),
        "zero_target_count": int(np.count_nonzero(np.linalg.norm(target, axis=1) <= 1e-12)),
    }


def adaptive_summary(t_values: np.ndarray, weights: np.ndarray) -> dict[str, float]:
    return {
        "adaptive_t_mean": float(t_values.mean()),
        "adaptive_t_std": float(t_values.std(ddof=0)),
        "adaptive_t_min": float(t_values.min()),
        "adaptive_t_max": float(t_values.max()),
        "w_hidden_mean": float(weights[:, 0].mean()),
        "w_mean_mean": float(weights[:, 1].mean()),
        "w_residual_mean": float(weights[:, 2].mean()),
    }


def scalar_meta(meta: dict[str, np.ndarray]) -> dict[str, Any]:
    out = {}
    for k, v in meta.items():
        a = np.asarray(v)
        value = a.item() if a.ndim == 0 else a.tolist()
        if isinstance(value, bytes):
            value = value.decode("utf-8", errors="replace")
        out[k] = value
    return out


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    fields = [
        "scope", "fold", "split", "method", "weight_kappa", "cosine", "std", "min", "max",
        "num_pairs", "num_negatives", "zero_prediction_count", "zero_target_count",
        "adaptive_t_mean", "adaptive_t_std", "adaptive_t_min", "adaptive_t_max",
        "w_hidden_mean", "w_mean_mean", "w_residual_mean",
    ]
    with tmp.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    os.replace(tmp, path)


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)
    z_neg, v_t, q, negative_ids, feature_meta = load_features(args.feature_path)

    fold_payload: dict[str, Any] = {}
    csv_rows: list[dict[str, Any]] = []
    aggregate: dict[str, dict[str, list[np.ndarray]]] = {
        "l2s": {"values": [], "pred": [], "target": [], "neg": []},
        "l2s_slerp": {"values": [], "pred": [], "target": [], "neg": [], "t": [], "weights": []},
    }

    for fold in args.folds:
        checkpoint_path = render(args.l2s_checkpoint_template, fold)
        if not checkpoint_path.is_file():
            raise FileNotFoundError(f"Missing L2S checkpoint: {checkpoint_path}")
        data, model = load_l2s_checkpoint(checkpoint_path, device=device)
        try:
            if "fold_idx" in data.files and int(np.asarray(data["fold_idx"]).item()) != fold:
                raise ValueError(f"Checkpoint {checkpoint_path} fold_idx mismatch")
            if "num_folds" in data.files and int(np.asarray(data["num_folds"]).item()) != args.num_folds:
                raise ValueError(f"Checkpoint {checkpoint_path} num_folds mismatch")
            train_q = unique_ids(data, "outer_train_q_indices", checkpoint_path)
            test_q = unique_ids(data, "test_q_indices", checkpoint_path)
            geometry_q = unique_ids(data, "geometry_train_q_indices", checkpoint_path)
            if np.intersect1d(train_q, test_q).size:
                raise ValueError(f"Fold {fold} outer train/test questions overlap")

            geometry_mask = np.isin(q, geometry_q)
            missing_geom = np.setdiff1d(geometry_q, np.unique(q[geometry_mask]))
            if not geometry_mask.any() or len(missing_geom):
                raise ValueError(f"Fold {fold} feature file does not cover all geometry-train questions")
            v_avg_np = v_t[geometry_mask].mean(axis=0, dtype=np.float64).astype(np.float32)
            v_avg = torch.as_tensor(v_avg_np, dtype=torch.float32, device=device)

            fold_results: dict[str, Any] = {}
            for split, question_ids in (("train", train_q), ("test", test_q)):
                mask = np.isin(q, question_ids)
                if not mask.any():
                    raise ValueError(f"Fold {fold} has no feature pairs for {split}")
                x = z_neg[mask]
                target = v_t[mask]
                neg = negative_ids[mask]
                direct = predict(model, x, device, args.batch_size)

                direct_values = cosine_values(direct, target)
                direct_stats = summary(direct_values, direct, target, neg)
                fold_results.setdefault(split, {})["l2s"] = direct_stats
                csv_rows.append({
                    "scope": "fold", "fold": fold, "split": split, "method": "l2s",
                    "weight_kappa": "", **direct_stats,
                    "adaptive_t_mean": "", "adaptive_t_std": "", "adaptive_t_min": "", "adaptive_t_max": "",
                    "w_hidden_mean": "", "w_mean_mean": "", "w_residual_mean": "",
                })

                adaptive_pred, adaptive_t, adaptive_weights = predict_adaptive_slerp(
                    x,
                    direct,
                    v_avg,
                    weight_kappa=args.weight_kappa,
                    device=device,
                    batch_size=args.batch_size,
                )
                adaptive_values = cosine_values(adaptive_pred, target)
                adaptive_stats = summary(adaptive_values, adaptive_pred, target, neg)
                adaptive_stats.update(adaptive_summary(adaptive_t, adaptive_weights))
                fold_results[split]["l2s_slerp"] = adaptive_stats
                csv_rows.append({
                    "scope": "fold", "fold": fold, "split": split, "method": "l2s_slerp",
                    "weight_kappa": float(args.weight_kappa), **adaptive_stats,
                })

                if split == "test":
                    neg_offset = neg + fold * (int(neg.max()) + 1 if len(neg) else 1)
                    aggregate["l2s"]["values"].append(direct_values)
                    aggregate["l2s"]["pred"].append(direct)
                    aggregate["l2s"]["target"].append(target)
                    aggregate["l2s"]["neg"].append(neg_offset)
                    aggregate["l2s_slerp"]["values"].append(adaptive_values)
                    aggregate["l2s_slerp"]["pred"].append(adaptive_pred)
                    aggregate["l2s_slerp"]["target"].append(target)
                    aggregate["l2s_slerp"]["neg"].append(neg_offset)
                    aggregate["l2s_slerp"]["t"].append(adaptive_t)
                    aggregate["l2s_slerp"]["weights"].append(adaptive_weights)

            fold_payload[f"fold{fold}"] = {
                "checkpoint": str(checkpoint_path),
                "v_avg_norm": float(np.linalg.norm(v_avg_np)),
                "geometry_train_q_indices": geometry_q.tolist(),
                "outer_train_q_indices": train_q.tolist(),
                "test_q_indices": test_q.tolist(),
                "results": fold_results,
            }
            print(f"\nFold {fold}")
            print(f"  L2S test cosine: {fold_results['test']['l2s']['cosine']:.6f}")
            a = fold_results["test"]["l2s_slerp"]
            print(
                f"  Adaptive L2S-SLERP: {a['cosine']:.6f} | "
                f"t mean={a['adaptive_t_mean']:.4f} "
                f"range=[{a['adaptive_t_min']:.4f},{a['adaptive_t_max']:.4f}]"
            )
        finally:
            data.close()

    aggregate_results: dict[str, Any] = {}
    for method in ("l2s", "l2s_slerp"):
        item = aggregate[method]
        values = np.concatenate(item["values"])
        pred = np.concatenate(item["pred"], axis=0)
        target = np.concatenate(item["target"], axis=0)
        neg = np.concatenate(item["neg"])
        stats = summary(values, pred, target, neg)
        if method == "l2s_slerp":
            t_values = np.concatenate(item["t"])
            weights = np.concatenate(item["weights"], axis=0)
            stats.update(adaptive_summary(t_values, weights))
        aggregate_results[method] = stats
        csv_rows.append({
            "scope": "aggregate", "fold": "all", "split": "test", "method": method,
            "weight_kappa": float(args.weight_kappa) if method == "l2s_slerp" else "",
            **stats,
            **({} if method == "l2s_slerp" else {
                "adaptive_t_mean": "", "adaptive_t_std": "", "adaptive_t_min": "", "adaptive_t_max": "",
                "w_hidden_mean": "", "w_mean_mean": "", "w_residual_mean": "",
            }),
        })

    payload = {
        "experiment": "l2s_vs_adaptive_l2s_slerp_cosine",
        "definitions": {
            "ground_truth": "raw v_T = z_pos_raw - z_neg_raw",
            "l2s": "direct raw L2S prediction; cosine is scale-invariant",
            "v_avg": "mean raw ground-truth v_T over geometry_train_q_indices",
            "residual": "normalize(L2S(z_neg_raw) - v_avg)",
            "adaptive_weights": (
                "same adaptive_3way_weights as three-sphere Dyn3D, using "
                "x=normalize(z_neg_raw), v=normalize(v_avg), r=residual"
            ),
            "adaptive_t": "w_residual / (w_mean + w_residual)",
            "l2s_slerp": (
                "||L2S(z_neg_raw)|| * SLERP(normalize(v_avg), residual, adaptive_t); "
                "SLERP changes direction only and restores the raw L2S magnitude"
            ),
            "alpha": "not used in cosine experiment",
        },
        "feature_path": str(Path(args.feature_path)),
        "feature_metadata": scalar_meta(feature_meta),
        "num_folds": args.num_folds,
        "evaluated_folds": args.folds,
        "weight_kappa": float(args.weight_kappa),
        "device": str(device),
        "folds": fold_payload,
        "aggregate_test": aggregate_results,
    }
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / f"{args.output_stem}.json"
    csv_path = output_dir / f"{args.output_stem}.csv"
    json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    write_csv(csv_path, csv_rows)

    print("\nAggregate held-out test cosine")
    print(f"  L2S: {aggregate_results['l2s']['cosine']:.6f}")
    a = aggregate_results["l2s_slerp"]
    print(
        f"  Adaptive L2S-SLERP: {a['cosine']:.6f} | "
        f"t mean={a['adaptive_t_mean']:.4f} "
        f"range=[{a['adaptive_t_min']:.4f},{a['adaptive_t_max']:.4f}]"
    )
    print(f"Saved JSON: {json_path}")
    print(f"Saved CSV : {csv_path}")


if __name__ == "__main__":
    main()
