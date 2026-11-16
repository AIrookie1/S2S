"""Core data loading, splitting, and MLP training for L2S-LINEAR.

L2S learns a direct mapping from a raw negative hidden state to its raw
truthful-minus-hallucinated pair vector:

    f_theta(z_neg) ~= v_T = z_pos - z_neg

No L2 normalization is performed in training or in the standard additive
steering rule. Inference applies ``h_new = h + alpha * f_theta(h)`` directly.
"""
from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn


class L2SMLP(nn.Module):
    """A direct two-layer MLP mapping hidden states to steering vectors."""

    def __init__(self, input_dim: int, hidden_dim: int = 256, output_dim: int | None = None) -> None:
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

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.fc2(F.gelu(self.fc1(z)))


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _as_float_pairs(data: Any) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Read raw pair inputs/targets from an already-open NPZ object."""
    keys = set(data.files)
    if {"z_neg_raw", "v_T", "vector_q_indices"}.issubset(keys):
        z_neg = np.asarray(data["z_neg_raw"], dtype=np.float32)
        vectors = np.asarray(data["v_T"], dtype=np.float32)
        q_indices = np.asarray(data["vector_q_indices"], dtype=np.int64).reshape(-1)
    elif {"z_neg_raw", "z_pos_raw", "vector_q_indices"}.issubset(keys):
        z_neg = np.asarray(data["z_neg_raw"], dtype=np.float32)
        vectors = (
            np.asarray(data["z_pos_raw"], dtype=np.float32)
            - z_neg
        )
        q_indices = np.asarray(data["vector_q_indices"], dtype=np.int64).reshape(-1)
    elif {"activations", "labels", "q_indices"}.issubset(keys):
        # Compatibility with older feature files.  Pair construction still
        # uses the raw answer-level activations and therefore remains
        # unnormalized.  The current extractor writes the pair-level fields.
        from methods.S2S_R.data import build_question_vectors

        activations = np.asarray(data["activations"], dtype=np.float32)
        labels = np.asarray(data["labels"], dtype=np.int64).reshape(-1)
        question_indices = np.asarray(data["q_indices"], dtype=np.int64).reshape(-1)
        choice_indices = np.asarray(
            data["choice_indices"]
            if "choice_indices" in keys
            else np.arange(len(labels), dtype=np.int64)
        ).reshape(-1)
        built = build_question_vectors(
            activations,
            labels,
            question_indices,
            choice_indices,
        )
        q_indices = built[0]
        z_neg = built[7]
        vectors = built[8]
    else:
        raise KeyError(
            "L2S requires raw z_neg_raw and v_T (or z_pos_raw), or raw "
            "answer-level activations/labels/q_indices. Normalized-only "
            "z_neg/z_pos feature files are not valid for L2S training."
        )

    if z_neg.ndim != 2 or vectors.ndim != 2:
        raise ValueError("L2S inputs and targets must have shape [N, hidden_dim]")
    if z_neg.shape != vectors.shape or len(q_indices) != len(z_neg):
        raise ValueError("L2S pair arrays must have matching lengths and dimensions")
    finite = np.isfinite(z_neg).all(axis=1) & np.isfinite(vectors).all(axis=1)
    if not finite.all():
        raise ValueError(
            f"L2S feature pairs contain {int((~finite).sum())} non-finite rows"
        )
    if len(z_neg) == 0:
        raise ValueError("L2S feature file contains no training pairs")
    return z_neg, vectors, q_indices


def load_l2s_pairs(path: str | Path) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
    """Load raw ``z_neg`` inputs and raw ``v_T`` targets for L2S."""
    with np.load(path, allow_pickle=False) as data:
        z_neg, vectors, q_indices = _as_float_pairs(data)
        metadata = {
            key: np.asarray(data[key])
            for key in ("train_q_indices", "test_q_indices", "fold_idx", "layer", "model_name")
            if key in data.files
        }
    return z_neg, vectors, q_indices, metadata


def split_like_s2s_r(
    q_indices: np.ndarray,
    metadata: dict[str, Any],
    *,
    num_folds: int,
    fold_idx: int,
    val_fraction: float,
    seed: int,
) -> dict[str, np.ndarray]:
    """Reproduce S2S_R question splitting exactly.

    The current S2S_R feature extractor has no precomputed fold fields, so the
    normal path uses sorted questions plus ``np.array_split``.  If a feature
    artifact already contains ``train_q_indices``/``test_q_indices``, the
    S2S_R convention is preserved: ``test_q_indices`` is the selected outer
    fold and ``train_q_indices`` is the held-out complement.
    """
    if num_folds < 1 or not 0 <= fold_idx < num_folds:
        raise ValueError("num_folds must be positive and fold_idx must be in range")
    if not 0.0 < val_fraction < 1.0:
        raise ValueError("val_fraction must be in (0, 1)")

    q_indices = np.asarray(q_indices, dtype=np.int64).reshape(-1)
    if {"train_q_indices", "test_q_indices"}.issubset(metadata):
        selected_q = np.asarray(metadata["test_q_indices"], dtype=np.int64).reshape(-1)
        heldout_q = np.asarray(metadata["train_q_indices"], dtype=np.int64).reshape(-1)
    else:
        questions = np.unique(q_indices)
        folds = np.array_split(questions, num_folds)
        selected_q = np.asarray(folds[fold_idx], dtype=np.int64)
        heldout_parts = [folds[i] for i in range(num_folds) if i != fold_idx]
        heldout_q = (
            np.concatenate(heldout_parts).astype(np.int64)
            if heldout_parts
            else np.empty(0, dtype=np.int64)
        )

    selected_q = np.sort(np.unique(selected_q))
    heldout_q = np.sort(np.unique(heldout_q))
    if len(selected_q) < 2:
        raise ValueError(
            "selected fold must contain at least two questions for the "
            "train/validation split"
        )

    rng = np.random.default_rng(seed)
    shuffled_q = rng.permutation(selected_q)
    n_val = int(round(len(shuffled_q) * val_fraction))
    n_val = max(1, min(len(shuffled_q) - 1, n_val))
    geometry_train_q = np.sort(shuffled_q[:-n_val])
    val_q = np.sort(shuffled_q[-n_val:])

    return {
        "outer_train_q_indices": selected_q,
        "geometry_train_q_indices": geometry_train_q,
        "val_q_indices": val_q,
        "predictor_q_indices": val_q.copy(),
        "test_q_indices": heldout_q,
    }


def fit_l2s(
    z_neg: np.ndarray,
    vectors: np.ndarray,
    q_indices: np.ndarray,
    train_q_indices: np.ndarray,
    *,
    hidden_dim: int = 256,
    epochs: int = 2000,
    batch_size: int = 256,
    learning_rate: float = 1e-3,
    device: str = "cpu",
    seed: int = 42,
    target_noise_sigma: float = 0.0,
    noise_seed: int | None = None,
) -> tuple[L2SMLP, list[dict[str, float]], dict[str, float]]:
    """Fit the direct raw-scale MLP using the requested question subset.

    When ``target_noise_sigma`` is nonzero, one relative-noise vector is
    sampled per selected target before optimization and the resulting
    corrupted targets are reused unchanged for every epoch/batch.
    """
    set_seed(seed)
    z_neg = np.asarray(z_neg, dtype=np.float32)
    vectors = np.asarray(vectors, dtype=np.float32)
    q_indices = np.asarray(q_indices, dtype=np.int64).reshape(-1)
    train_q_indices = np.asarray(train_q_indices, dtype=np.int64).reshape(-1)
    if z_neg.ndim != 2 or vectors.shape != z_neg.shape or len(q_indices) != len(z_neg):
        raise ValueError("z_neg, vectors, and q_indices are misaligned")
    if len(train_q_indices) == 0:
        raise ValueError("L2S optimization split is empty")
    train_mask = np.isin(q_indices, train_q_indices)
    if not train_mask.any():
        raise ValueError("No L2S pairs belong to the optimization split")
    if epochs <= 0 or batch_size <= 0 or learning_rate <= 0:
        raise ValueError("epochs, batch_size, and learning_rate must be positive")
    if not np.isfinite(target_noise_sigma) or target_noise_sigma < 0:
        raise ValueError("target_noise_sigma must be finite and non-negative")

    dev = torch.device(device if device != "cuda" or torch.cuda.is_available() else "cpu")
    x = torch.as_tensor(z_neg[train_mask], dtype=torch.float32, device=dev)
    y_clean = torch.as_tensor(vectors[train_mask], dtype=torch.float32, device=dev)
    y = y_clean
    if target_noise_sigma > 0:
        generator = torch.Generator(device=dev)
        generator.manual_seed(int(seed if noise_seed is None else noise_seed))
        eps = torch.randn(y_clean.shape, generator=generator, device=dev, dtype=y_clean.dtype)
        eps = F.normalize(eps, p=2, dim=-1, eps=1e-12)
        target_norm = torch.linalg.vector_norm(y_clean, dim=-1, keepdim=True)
        y = y_clean + float(target_noise_sigma) * target_norm * eps
    model = L2SMLP(x.shape[-1], hidden_dim, y.shape[-1]).to(dev)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    history: list[dict[str, float]] = []

    model.train()
    for epoch in range(1, epochs + 1):
        order = torch.randperm(len(x), device=dev)
        batch_losses: list[float] = []
        for start in range(0, len(x), batch_size):
            idx = order[start : start + batch_size]
            prediction = model(x[idx])
            loss = F.mse_loss(prediction, y[idx])
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            batch_losses.append(float(loss.detach().cpu()))
        with torch.no_grad():
            train_mse = float(F.mse_loss(model(x), y).cpu())
            clean_train_mse = float(F.mse_loss(model(x), y_clean).cpu())
        record = {
            "epoch": float(epoch),
            "loss": float(np.mean(batch_losses)),
            "train_mse": train_mse,
            "clean_train_mse": clean_train_mse,
        }
        history.append(record)
        print(
            f"epoch {epoch:03d} | loss={record['loss']:.6e} "
            f"train_mse={train_mse:.6e} clean_mse={clean_train_mse:.6e}",
            flush=True,
        )

    model.eval()
    with torch.no_grad():
        all_x = torch.as_tensor(z_neg, dtype=torch.float32, device=dev)
        all_y = torch.as_tensor(vectors, dtype=torch.float32, device=dev)
        prediction = model(all_x)
        all_mse = float(F.mse_loss(prediction, all_y).cpu())
        train_mse = float(F.mse_loss(prediction[torch.as_tensor(train_mask, device=dev)], all_y[torch.as_tensor(train_mask, device=dev)]).cpu())
        metrics = {
            "train_mse": train_mse,
            "all_mse": all_mse,
            "train_pairs": float(train_mask.sum()),
            "all_pairs": float(len(z_neg)),
            "target_norm_mean": float(torch.linalg.vector_norm(all_y, dim=-1).mean().cpu()),
            "prediction_norm_mean": float(torch.linalg.vector_norm(prediction, dim=-1).mean().cpu()),
            "raw_input_norm_mean": float(torch.linalg.vector_norm(all_x, dim=-1).mean().cpu()),
            "target_noise_sigma": float(target_noise_sigma),
            "noise_seed": int(seed if noise_seed is None else noise_seed),
            "noisy_train_target_mse": float(F.mse_loss(y, y_clean).cpu()),
        }
    return model.cpu(), history, metrics


def load_l2s_checkpoint(path: str | Path, device: torch.device | str = "cpu") -> tuple[np.lib.npyio.NpzFile, L2SMLP]:
    """Load an L2S checkpoint and reconstruct its MLP."""
    data = np.load(path, allow_pickle=False)
    required = {"hidden_dim", "mlp_hidden", "mlp_fc1.weight", "mlp_fc1.bias", "mlp_fc2.weight", "mlp_fc2.bias"}
    missing = required.difference(data.files)
    if missing:
        data.close()
        raise KeyError(f"L2S checkpoint missing fields: {sorted(missing)}")
    dim = int(np.asarray(data["hidden_dim"]).item())
    hidden = int(np.asarray(data["mlp_hidden"]).item())
    model = L2SMLP(dim, hidden, dim).to(device)
    state = {
        "fc1.weight": torch.as_tensor(data["mlp_fc1.weight"], dtype=torch.float32),
        "fc1.bias": torch.as_tensor(data["mlp_fc1.bias"], dtype=torch.float32),
        "fc2.weight": torch.as_tensor(data["mlp_fc2.weight"], dtype=torch.float32),
        "fc2.bias": torch.as_tensor(data["mlp_fc2.bias"], dtype=torch.float32),
    }
    model.load_state_dict(state)
    model.eval()
    return data, model


def json_metrics(history: list[dict[str, float]], metrics: dict[str, float]) -> str:
    """Serialize training information without requiring NumPy scalars."""
    return json.dumps({"history": history, "metrics": metrics}, ensure_ascii=False)
