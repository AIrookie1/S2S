"""Direct vector predictor used by Dyn3D.

The model receives an L2-normalized negative/current hidden state and directly
predicts the full truthful steering vector

    v_i = h_i^+ - h_i^-

in the original hidden-state scale.  Dyn3D keeps a separately estimated mean
vector ``v_avg`` and constructs the dynamic residual at inference time as

    residual = normalize(v_hat_i - v_avg)

There is no SVD basis, coefficient routing, Top-K selection, or load-balancing
objective in this implementation.
"""
from __future__ import annotations

import random
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn


class DirectVectorMLP(nn.Module):
    """Two-layer MLP that maps a normalized hidden state to a full ``v_i``."""

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 256,
        output_dim: int | None = None,
    ) -> None:
        super().__init__()
        if input_dim <= 0 or hidden_dim <= 0:
            raise ValueError("input_dim and hidden_dim must be positive")
        output_dim = input_dim if output_dim is None else int(output_dim)
        if output_dim <= 0:
            raise ValueError("output_dim must be positive")
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)
        self.input_dim = int(input_dim)
        self.hidden_dim = int(hidden_dim)
        self.output_dim = int(output_dim)

    def forward(self, h_normalized: torch.Tensor) -> torch.Tensor:
        return self.fc2(F.gelu(self.fc1(h_normalized)))


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_feature_pairs(
    path: str | Path,
    positive_mode: str = "mean",
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
    """Load raw negative/positive hidden-state pairs.

    Pair construction happens at training time.  ``positive_mode='mean'``
    uses the mean of all positive answers for a question, exactly as the
    original Dyn3D pipeline did.  ``mc1`` keeps the single MC1-positive answer.
    """
    if positive_mode not in {"mean", "mc1", "cartesian"}:
        raise ValueError("positive_mode must be one of: mean, mc1, cartesian")

    with np.load(path, allow_pickle=False) as data:
        keys = set(data.files)
        if positive_mode in {"mean", "mc1"} and {
            "activations",
            "labels",
            "q_indices",
        }.issubset(keys):
            activations = np.asarray(data["activations"], np.float32)
            labels = np.asarray(data["labels"]).reshape(-1)
            q_indices = np.asarray(data["q_indices"]).reshape(-1)
            choice_indices = np.asarray(
                data["choice_indices"]
                if "choice_indices" in keys
                else np.arange(len(labels))
            ).reshape(-1)

            if positive_mode == "mc1":
                if "choices" not in keys:
                    raise ValueError("positive_mode='mc1' requires a choices field")
                from datasets import load_dataset

                dataset = load_dataset("truthful_qa", "multiple_choice")["validation"]
                mc1_by_question = {}
                for q_idx, item in enumerate(dataset):
                    mc1 = item["mc1_targets"]
                    mc1_by_question[q_idx] = {
                        str(choice): int(label)
                        for choice, label in zip(mc1["choices"], mc1["labels"])
                    }
                choices = np.asarray(data["choices"]).reshape(-1)
                mc1_labels = np.asarray(
                    [
                        mc1_by_question[int(q)].get(str(choice), 0)
                        for q, choice in zip(q_indices, choices)
                    ],
                    dtype=np.int64,
                )
            else:
                mc1_labels = None

            h_neg, h_pos, vector_q = _build_training_pairs(
                activations,
                labels,
                q_indices,
                choice_indices,
                positive_mode,
                mc1_labels,
            )
            metadata = {
                key: np.asarray(data[key])
                for key in (
                    "train_q_indices",
                    "test_q_indices",
                    "fold_idx",
                    "layer",
                    "model_name",
                )
                if key in keys
            }
            metadata["positive_mode"] = np.asarray(positive_mode)
            return h_neg, h_pos, vector_q, metadata

        if {"z_neg_raw", "z_pos_raw"}.issubset(keys):
            h_neg = np.asarray(data["z_neg_raw"], np.float32)
            h_pos = np.asarray(data["z_pos_raw"], np.float32)
        elif {"z_neg", "z_pos"}.issubset(keys):
            raise KeyError(
                "feature file contains only normalized z_neg/z_pos pair fields. "
                "Dyn3D direct-vector training requires raw z_neg_raw/z_pos_raw "
                "or answer-level raw activations. Re-run extract_features.py."
            )
        else:
            raise KeyError(
                "feature file must contain answer-level raw activations or "
                "z_neg_raw/z_pos_raw"
            )
        if "vector_q_indices" not in keys:
            raise KeyError("feature file must contain vector_q_indices")
        q = np.asarray(data["vector_q_indices"], dtype=np.int64).reshape(-1)
        if len(q) != len(h_neg):
            raise ValueError("vector_q_indices must align with pairs")
        metadata = {
            key: np.asarray(data[key])
            for key in (
                "train_q_indices",
                "test_q_indices",
                "fold_idx",
                "layer",
                "model_name",
                "positive_mode",
            )
            if key in keys
        }
        return h_neg, h_pos, q, metadata


def _build_training_pairs(
    activations: np.ndarray,
    labels: np.ndarray,
    q_indices: np.ndarray,
    choice_indices: np.ndarray,
    positive_mode: str,
    mc1_labels: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    del choice_indices  # Pairing only needs question grouping and labels.
    h_neg: list[np.ndarray] = []
    h_pos: list[np.ndarray] = []
    vector_q: list[int] = []

    for q in np.unique(q_indices):
        rows = np.flatnonzero(q_indices == q)
        neg_rows = rows[labels[rows] == 0]
        pos_rows = rows[labels[rows] == 1]
        if positive_mode == "mc1":
            if mc1_labels is None:
                raise ValueError("mc1 labels are required for positive_mode='mc1'")
            pos_rows = rows[mc1_labels[rows] == 1]
            if len(pos_rows) != 1:
                raise ValueError(
                    f"Question {q} must have exactly one MC1-positive answer, "
                    f"found {len(pos_rows)}"
                )
        if len(pos_rows) == 0 or len(neg_rows) == 0:
            raise ValueError(f"Question {q} has no valid positive/negative answers")

        if positive_mode == "mean":
            positives = [activations[pos_rows].mean(axis=0)]
        else:
            positives = [activations[int(row)] for row in pos_rows]

        for positive in positives:
            for neg_row in neg_rows:
                h_pos.append(positive)
                h_neg.append(activations[int(neg_row)])
                vector_q.append(int(q))

    if not h_pos:
        raise ValueError("No positive-negative pairs generated")
    return (
        np.stack(h_neg).astype(np.float32),
        np.stack(h_pos).astype(np.float32),
        np.asarray(vector_q, np.int64),
    )


def predict_residual(
    predictor: DirectVectorMLP,
    hidden: torch.Tensor,
    v_avg: torch.Tensor,
    eps: float = 1e-8,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Predict ``v_i`` from normalized ``hidden`` and return its residual.

    Returns ``(v_hat, normalize(v_hat - v_avg))``.  The caller may pass raw
    hidden states; this helper always applies the training-time normalization.
    """
    h_normalized = F.normalize(hidden.float(), p=2, dim=-1, eps=eps)
    v_hat = predictor(h_normalized)
    avg = v_avg.to(device=v_hat.device, dtype=v_hat.dtype)
    residual = F.normalize(v_hat - avg, p=2, dim=-1, eps=eps)
    return v_hat, residual


def fit_dynamic_mlp(
    h_neg: np.ndarray,
    h_pos: np.ndarray,
    q_indices: np.ndarray,
    mean_mask: np.ndarray,
    mlp_mask: np.ndarray,
    *,
    mlp_hidden: int = 256,
    epochs: int = 2000,
    batch_size: int = 256,
    learning_rate: float = 1e-3,
    device: str = "cpu",
    seed: int = 42,
) -> tuple[dict[str, Any], list[dict[str, float]], dict[str, Any]]:
    """Fit the direct vector MLP while preserving Dyn3D's split semantics.

    ``v_avg`` is estimated from ``mean_mask``.  The MLP is optimized only on
    negative hidden states selected by ``mlp_mask``:

        x = normalize(h_neg)
        y = v_i = h_pos - h_neg
        loss = MSE(MLP(x), y)

    No positive-state samples or interpolated auxiliary samples are added.
    """
    set_seed(seed)
    if mlp_hidden <= 0 or epochs <= 0 or batch_size <= 0 or learning_rate <= 0:
        raise ValueError(
            "mlp_hidden, epochs, batch_size, and learning_rate must be positive"
        )

    h_neg_arr = np.asarray(h_neg, dtype=np.float32)
    h_pos_arr = np.asarray(h_pos, dtype=np.float32)
    q_arr = np.asarray(q_indices, dtype=np.int64).reshape(-1)
    if h_neg_arr.ndim != 2 or h_pos_arr.shape != h_neg_arr.shape:
        raise ValueError("h_neg and h_pos must have matching [N, hidden_dim] shapes")
    if len(q_arr) != len(h_neg_arr):
        raise ValueError("q_indices must align with h_neg/h_pos")
    finite_rows = np.isfinite(h_neg_arr).all(axis=1) & np.isfinite(h_pos_arr).all(axis=1)
    if not finite_rows.all():
        bad = int((~finite_rows).sum())
        raise ValueError(f"Feature file contains non-finite values in {bad} pair rows")

    mean_mask_arr = np.asarray(mean_mask, dtype=bool).reshape(-1)
    mlp_mask_arr = np.asarray(mlp_mask, dtype=bool).reshape(-1)
    if len(mean_mask_arr) != len(h_neg_arr) or len(mlp_mask_arr) != len(h_neg_arr):
        raise ValueError("mean_mask and mlp_mask must align with pair rows")
    if not mean_mask_arr.any() or not mlp_mask_arr.any():
        raise ValueError("mean and MLP training masks must both be non-empty")

    dev = torch.device(device if device != "cuda" or torch.cuda.is_available() else "cpu")
    neg = torch.as_tensor(h_neg_arr, dtype=torch.float32, device=dev)
    pos = torch.as_tensor(h_pos_arr, dtype=torch.float32, device=dev)
    vectors = pos - neg
    mean_tr = torch.as_tensor(mean_mask_arr, dtype=torch.bool, device=dev)
    mlp_tr = torch.as_tensor(mlp_mask_arr, dtype=torch.bool, device=dev)

    v_avg = vectors[mean_tr].mean(dim=0)
    raw_inputs = neg[mlp_tr]
    inputs = F.normalize(raw_inputs, p=2, dim=-1, eps=1e-8)
    targets = vectors[mlp_tr]

    predictor = DirectVectorMLP(
        input_dim=neg.shape[-1],
        hidden_dim=mlp_hidden,
        output_dim=vectors.shape[-1],
    ).to(dev)
    optimizer = torch.optim.Adam(predictor.parameters(), lr=learning_rate)
    history: list[dict[str, float]] = []

    predictor.train()
    for epoch in range(1, epochs + 1):
        order = torch.randperm(len(inputs), device=dev)
        losses: list[float] = []
        for start in range(0, len(inputs), batch_size):
            idx = order[start : start + batch_size]
            prediction = predictor(inputs[idx])
            loss = F.mse_loss(prediction, targets[idx])
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach().cpu()))

        with torch.no_grad():
            prediction = predictor(inputs)
            train_mse = float(F.mse_loss(prediction, targets).cpu())
            predicted_residual = F.normalize(
                prediction - v_avg,
                p=2,
                dim=-1,
                eps=1e-8,
            )
            target_residual = F.normalize(
                targets - v_avg,
                p=2,
                dim=-1,
                eps=1e-8,
            )
            residual_cosine = float(
                F.cosine_similarity(
                    predicted_residual,
                    target_residual,
                    dim=-1,
                ).mean().cpu()
            )
        record = {
            "epoch": float(epoch),
            "loss": float(np.mean(losses)),
            "train_vector_mse": train_mse,
            "train_residual_cosine": residual_cosine,
        }
        history.append(record)
        print(
            f"epoch {epoch:03d} | loss={record['loss']:.6e} "
            f"vector_mse={train_mse:.6e} residual_cos={residual_cosine:.6f}",
            flush=True,
        )

    predictor.eval()
    with torch.no_grad():
        prediction = predictor(inputs)
        prediction_error = prediction - targets
        predicted_residual_raw = prediction - v_avg
        target_residual_raw = targets - v_avg
        predicted_residual = F.normalize(predicted_residual_raw, p=2, dim=-1, eps=1e-8)
        target_residual = F.normalize(target_residual_raw, p=2, dim=-1, eps=1e-8)
        metrics: dict[str, Any] = {
            "train_vector_mse": float(F.mse_loss(prediction, targets).cpu()),
            "train_residual_raw_mse": float(
                F.mse_loss(predicted_residual_raw, target_residual_raw).cpu()
            ),
            "train_residual_direction_mse": float(
                F.mse_loss(predicted_residual, target_residual).cpu()
            ),
            "train_residual_cosine": float(
                F.cosine_similarity(predicted_residual, target_residual, dim=-1)
                .mean()
                .cpu()
            ),
            "prediction_error_l2_mean": float(
                torch.linalg.vector_norm(prediction_error, dim=-1).mean().cpu()
            ),
            "target_vector_norm_mean": float(
                torch.linalg.vector_norm(targets, dim=-1).mean().cpu()
            ),
            "prediction_vector_norm_mean": float(
                torch.linalg.vector_norm(prediction, dim=-1).mean().cpu()
            ),
            "v_avg_norm": float(torch.linalg.vector_norm(v_avg).cpu()),
            "raw_input_norm_mean": float(
                torch.linalg.vector_norm(raw_inputs, dim=-1).mean().cpu()
            ),
            "normalized_input_norm_mean": float(
                torch.linalg.vector_norm(inputs, dim=-1).mean().cpu()
            ),
            "mean_train_pairs": int(mean_tr.sum().item()),
            "mlp_train_pairs": int(mlp_tr.sum().item()),
            "normalization_policy": (
                "MLP input h_neg is L2-normalized; target v_i=h_pos-h_neg and "
                "v_avg remain on the original hidden-state scale; inference "
                "uses normalize(MLP(normalize(h))-v_avg) as the residual direction"
            ),
        }

    state: dict[str, Any] = {
        "v_avg": v_avg.detach().cpu(),
        "predictor": predictor.cpu(),
        "hidden_dim": int(neg.shape[-1]),
        "mlp_hidden": int(mlp_hidden),
        "mean_q_indices": np.unique(q_arr[mean_mask_arr]),
        "mlp_q_indices": np.unique(q_arr[mlp_mask_arr]),
    }
    return state, history, metrics
