#!/usr/bin/env python3
"""Compare L2S, L2S-SLERP, standard mean steering, Dyn3D, DynLearnedSLERP, dynMLP, and SADI against raw pairwise v_T.

Experiment definition (per outer fold):

* Ground truth for every pair is the extractor's raw Cartesian vector
  ``v_T = z_pos_raw - z_neg_raw``.
* L2S predicts ``v_T`` directly from raw ``z_neg_raw``.
* L2S-SLERP reuses L2S's predicted ``v_T``, subtracts the pair-level training
  ``v_avg`` to obtain a predicted residual, computes the same adaptive 3-way
  weights as Dyn3D from ``normalize(z_neg)``, ``normalize(v_avg)`` and the
  residual, uses ``t = w_residual / (w_mean + w_residual)``, then restores the
  per-sample L2S prediction norm after SLERP.
* Standard Steer uses the mean of *all* raw ``v_T`` pairs whose question is in
  the outer-train fold, then compares that same mean against every train/test
  pair vector.
* Dyn3D reuses the trained checkpoint's own ``v_avg``, router and SVD basis.
  The raw negative state is used only to obtain the routed residual and the
  existing adaptive weights.  The prediction is a single direct SLERP from
  normalized ``v_avg`` to the normalized routed residual with
  ``t = gamma / (beta + gamma)``.  That SLERP output itself is the steering
  vector; no symmetric 2-SLERP, norm rescaling, or ``candidate - z_neg`` is
  used in this cosine experiment.
* DynLearnedSLERP learns both an SVD-routed residual direction and the first
  SLERP coordinate. Its training includes a joint final-SLERP cosine loss.
* dynMLP uses the same pair-level v_avg/residual target and the same direct
  SLERP as Dyn3D, but predicts the normalized residual directly with a
  two-layer MLP instead of routing through an SVD basis.
* SADI uses the uploaded method's hidden branch at the target LLaMA layer:
  hidden units are ranked on geometry-train MLP true-minus-false differences,
  count/strength are chosen on validation only, and the dynamic decoder-layer
  prediction is ``(strength - 1) * mlp_z_neg`` on selected coordinates.
* Ground truth follows all-pairs mean: every positive/negative Cartesian
  pair contributes one cosine value, and the final cosine is the arithmetic
  mean over all pair rows. No nearest/best GT is selected.

The script never changes training checkpoints. Outer-fold membership is taken
from the checkpoints and cross-validated across L2S, Dyn3D, DynLearnedSLERP, dynMLP, and SADI
so all methods are evaluated on exactly the same questions.
"""
from __future__ import annotations

import argparse
import ast
import csv
import json
import math
import os
import sys
from pathlib import Path
from typing import Any, Callable

import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from method.dyn3d_sga.dynSVD import ResidualRouter, normalized_basis_combination
from method.dynlearnedslerp.model import (
    load_checkpoint as load_dynlearnedslerp_checkpoint,
    predict_direction as predict_dynlearnedslerp_direction,
)
from method.dynmlp.model import load_dynmlp_checkpoint
from method.l2s.model import load_l2s_checkpoint
from method.l2s_slerp.steering import l2s_prediction_to_slerp
from method.sadi.hidden import load_sadi_checkpoint, predict_hidden_vectors

METHOD_ORDER = ("l2s", "l2s_slerp", "steer", "dyn3d", "DynLearnedSLERP", "dynMLP", "sadi")
SPLIT_ORDER = ("train", "test")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Raw-v_T cosine similarity for L2S, L2S-SLERP, Standard Steer, Dyn3D, DynLearnedSLERP, dynMLP, and SADI"
    )
    p.add_argument("--feature_path", required=True)
    p.add_argument(
        "--l2s_checkpoint_template",
        required=True,
        help="L2S checkpoint path template containing {fold}",
    )
    p.add_argument(
        "--dyn3d_checkpoint_template",
        required=True,
        help="Dyn3D checkpoint path template containing {fold}",
    )
    p.add_argument(
        "--dynlearnedslerp_checkpoint_template",
        required=True,
        help="DynLearnedSLERP checkpoint path template containing {fold}",
    )
    p.add_argument(
        "--dynmlp_checkpoint_template",
        required=True,
        help="dynMLP checkpoint path template containing {fold}",
    )
    p.add_argument(
        "--sadi_checkpoint_template",
        required=True,
        help="SADI-hidden checkpoint path template containing {fold}",
    )
    p.add_argument("--folds", type=int, nargs="+", default=[0, 1])
    p.add_argument("--num_folds", type=int, default=2)
    p.add_argument(
        "--weight_kappa",
        type=float,
        default=3.0,
        help="Dyn3D/dynMLP adaptive_3way_weights kappa; should match evaluation hyperparameters",
    )
    p.add_argument("--batch_size", type=int, default=4096)
    p.add_argument("--device", choices=["cuda", "cpu"], default="cpu")
    p.add_argument(
        "--output_dir",
        default="./results/experiment/cosine_similarity",
    )
    p.add_argument(
        "--output_stem",
        default="cosine_similarity",
        help="Writes <stem>.json and <stem>.csv inside --output_dir",
    )
    args = p.parse_args()

    if args.num_folds < 2:
        p.error("--num_folds must be >= 2")
    if not args.folds or len(set(args.folds)) != len(args.folds):
        p.error("--folds must contain unique fold indices")
    if any(f < 0 or f >= args.num_folds for f in args.folds):
        p.error("every --folds value must satisfy 0 <= fold < num_folds")
    if "{fold}" not in args.l2s_checkpoint_template:
        p.error("--l2s_checkpoint_template must contain {fold}")
    if "{fold}" not in args.dyn3d_checkpoint_template:
        p.error("--dyn3d_checkpoint_template must contain {fold}")
    if "{fold}" not in args.dynlearnedslerp_checkpoint_template:
        p.error("--dynlearnedslerp_checkpoint_template must contain {fold}")
    if "{fold}" not in args.dynmlp_checkpoint_template:
        p.error("--dynmlp_checkpoint_template must contain {fold}")
    if "{fold}" not in args.sadi_checkpoint_template:
        p.error("--sadi_checkpoint_template must contain {fold}")
    if not math.isfinite(args.weight_kappa) or args.weight_kappa <= 0:
        p.error("--weight_kappa must be finite and positive")
    if args.batch_size <= 0:
        p.error("--batch_size must be positive")
    if args.device == "cuda" and not torch.cuda.is_available():
        p.error("--device cuda requested but CUDA is not available")
    return args


def render_fold_path(template: str, fold: int) -> Path:
    try:
        rendered = template.format(fold=fold)
    except (KeyError, IndexError, ValueError) as exc:
        raise ValueError(
            f"Invalid checkpoint template {template!r}; expected only {{fold}}"
        ) from exc
    return Path(rendered)


def _require_file(path: Path, label: str) -> None:
    if not path.is_file() or path.stat().st_size == 0:
        raise FileNotFoundError(f"Missing {label}: {path}")


def load_raw_pair_features(
    path: str | Path,
) -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    dict[str, Any],
]:
    """Load raw pair data plus IDs used for nearest-correct-GT grouping."""
    path = Path(path)
    _require_file(path, "feature file")
    with np.load(path, allow_pickle=False) as data:
        required = {
            "z_neg_raw",
            "v_T",
            "vector_q_indices",
            "negative_sample_indices",
            "sadi_mlp_z_neg_raw",
        }
        missing = required.difference(data.files)
        if missing:
            raise KeyError(
                f"Feature file {path} is missing raw pair fields {sorted(missing)}. "
                "Use the integrated method/dyn3d_sga/extract_features.py output."
            )
        z_neg = np.asarray(data["z_neg_raw"], dtype=np.float32)
        v_t = np.asarray(data["v_T"], dtype=np.float32)
        sadi_mlp_z_neg = np.asarray(data["sadi_mlp_z_neg_raw"], dtype=np.float32)
        q = np.asarray(data["vector_q_indices"], dtype=np.int64).reshape(-1)
        negative_ids = np.asarray(
            data["negative_sample_indices"], dtype=np.int64
        ).reshape(-1)
        meta = {
            key: np.asarray(data[key])
            for key in ("layer", "model_name", "model_path", "vector_definition")
            if key in data.files
        }
        if "z_pos_raw" in data.files:
            z_pos = np.asarray(data["z_pos_raw"], dtype=np.float32)
            if z_pos.shape != z_neg.shape:
                raise ValueError("z_pos_raw and z_neg_raw shape mismatch")
            reconstructed = z_pos - z_neg
            if not np.allclose(reconstructed, v_t, rtol=2e-5, atol=2e-5):
                max_err = float(np.max(np.abs(reconstructed - v_t)))
                raise ValueError(
                    "Feature v_T is not consistent with z_pos_raw - z_neg_raw; "
                    f"max absolute difference={max_err:.6e}"
                )

    if (
        z_neg.ndim != 2
        or v_t.ndim != 2
        or z_neg.shape != v_t.shape
        or sadi_mlp_z_neg.shape != z_neg.shape
    ):
        raise ValueError(
            "z_neg_raw, v_T, and sadi_mlp_z_neg_raw must have matching "
            "[num_pairs, hidden_dim] shapes"
        )
    if len(q) != len(z_neg) or len(negative_ids) != len(z_neg):
        raise ValueError(
            "vector_q_indices and negative_sample_indices must align with raw pair rows"
        )
    if len(z_neg) == 0:
        raise ValueError("Feature file contains no raw pair rows")
    finite = (
        np.isfinite(z_neg).all(axis=1)
        & np.isfinite(v_t).all(axis=1)
        & np.isfinite(sadi_mlp_z_neg).all(axis=1)
    )
    if not finite.all():
        raise ValueError(f"Feature file contains {int((~finite).sum())} non-finite pair rows")
    return z_neg, v_t, q, negative_ids, sadi_mlp_z_neg, meta


def _checkpoint_split(data: Any, *, path: Path, fold: int, num_folds: int) -> tuple[np.ndarray, np.ndarray]:
    required = {"outer_train_q_indices", "test_q_indices"}
    missing = required.difference(data.files)
    if missing:
        raise KeyError(f"Checkpoint {path} missing split fields: {sorted(missing)}")

    if "fold_idx" in data.files:
        stored_fold = int(np.asarray(data["fold_idx"]).item())
        if stored_fold != fold:
            raise ValueError(f"Checkpoint {path} stores fold_idx={stored_fold}, expected {fold}")
    if "num_folds" in data.files:
        stored_num = int(np.asarray(data["num_folds"]).item())
        if stored_num != num_folds:
            raise ValueError(
                f"Checkpoint {path} stores num_folds={stored_num}, expected {num_folds}"
            )

    train_q = np.sort(np.unique(np.asarray(data["outer_train_q_indices"], dtype=np.int64).reshape(-1)))
    test_q = np.sort(np.unique(np.asarray(data["test_q_indices"], dtype=np.int64).reshape(-1)))
    if len(train_q) == 0 or len(test_q) == 0:
        raise ValueError(f"Checkpoint {path} has an empty outer train/test split")
    overlap = np.intersect1d(train_q, test_q)
    if len(overlap):
        raise ValueError(f"Checkpoint {path} train/test question sets overlap: {overlap[:10].tolist()}")
    return train_q, test_q


def _same_questions(a: np.ndarray, b: np.ndarray) -> bool:
    return np.array_equal(np.sort(np.unique(a)), np.sort(np.unique(b)))


def _mask_for_questions(q: np.ndarray, question_ids: np.ndarray, *, label: str) -> np.ndarray:
    mask = np.isin(q, question_ids)
    if not mask.any():
        raise ValueError(f"No raw v_T pairs found for {label} questions")
    missing = np.setdiff1d(question_ids, np.unique(q[mask]))
    if len(missing):
        raise ValueError(f"Feature file is missing {label} questions: {missing[:10].tolist()}")
    return mask


def _cosine_values(pred: np.ndarray, target: np.ndarray) -> np.ndarray:
    if pred.shape != target.shape or pred.ndim != 2:
        raise ValueError("cosine inputs must have the same [N, D] shape")
    p = torch.as_tensor(pred, dtype=torch.float32)
    t = torch.as_tensor(target, dtype=torch.float32)
    return F.cosine_similarity(p, t, dim=-1, eps=1e-8).cpu().numpy().astype(np.float64)


def _summarize_cosine(
    pred: np.ndarray,
    target: np.ndarray,
    negative_ids: np.ndarray,
) -> dict[str, float | int]:
    """Summarize cosine over every positive/negative Cartesian pair.

    A repeated negative may have multiple correct GT vectors. Each pair gets its
    own cosine value and all pair values contribute equally to the final mean.
    ``negative_ids`` is retained only for reporting the number of unique
    negatives; it is not used to choose or aggregate GTs.
    """
    values = _cosine_values(pred, target)
    negative_ids = np.asarray(negative_ids, dtype=np.int64).reshape(-1)
    if len(negative_ids) != len(values):
        raise ValueError("negative_ids must align with cosine pair rows")
    if len(values) == 0:
        raise ValueError("pairwise cosine requires at least one pair row")

    pred_norm = np.linalg.norm(pred, axis=1)
    target_norm = np.linalg.norm(target, axis=1)
    return {
        "cosine": float(values.mean()),
        "std": float(values.std(ddof=0)),
        "min": float(values.min()),
        "max": float(values.max()),
        "num_pairs": int(len(values)),
        "num_negatives": int(len(np.unique(negative_ids))),
        "zero_prediction_count": int(np.count_nonzero(pred_norm <= 1e-12)),
        "zero_target_count": int(np.count_nonzero(target_norm <= 1e-12)),
    }


def predict_l2s(
    model: torch.nn.Module,
    z_neg: np.ndarray,
    *,
    device: torch.device,
    batch_size: int,
) -> np.ndarray:
    outputs: list[np.ndarray] = []
    model = model.to(device)
    model.eval()
    with torch.inference_mode():
        for start in range(0, len(z_neg), batch_size):
            batch = torch.as_tensor(
                z_neg[start : start + batch_size], dtype=torch.float32, device=device
            )
            outputs.append(model(batch).detach().float().cpu().numpy())
    return np.concatenate(outputs, axis=0).astype(np.float32, copy=False)



def predict_l2s_slerp_vectors(
    z_neg: np.ndarray,
    l2s_pred: np.ndarray,
    *,
    avg: torch.Tensor,
    weight_kappa: float,
    device: torch.device,
    batch_size: int,
) -> np.ndarray:
    """Convert direct L2S predictions into adaptive mean/residual SLERP.

    For each pair::

        v_hat = L2S(z_neg_raw)
        r_hat = normalize(v_hat - v_avg)
        (w_h, w_mean, w_res) = adaptive_3way_weights(
            normalize(z_neg_raw), normalize(v_avg), r_hat
        )
        t = w_res / (w_mean + w_res)
        v_pred = ||v_hat|| * SLERP(normalize(v_avg), r_hat, t)

    The L2S checkpoint is reused as-is; no additional predictor is trained.
    """
    if z_neg.shape != l2s_pred.shape or l2s_pred.ndim != 2:
        raise ValueError("z_neg and l2s_pred must match [num_pairs, hidden_dim]")

    outputs: list[np.ndarray] = []
    with torch.inference_mode():
        for start in range(0, len(l2s_pred), batch_size):
            batch_z = torch.as_tensor(
                z_neg[start : start + batch_size],
                dtype=torch.float32,
                device=device,
            )
            direct_v = torch.as_tensor(
                l2s_pred[start : start + batch_size],
                dtype=torch.float32,
                device=device,
            )
            predicted_vector = l2s_prediction_to_slerp(
                batch_z,
                direct_v,
                avg,
                weight_kappa=weight_kappa,
            )
            outputs.append(predicted_vector.detach().float().cpu().numpy())

    return np.concatenate(outputs, axis=0).astype(np.float32, copy=False)


def _load_existing_functions(
    source_path: Path,
    names: tuple[str, ...],
    namespace: dict[str, Any],
) -> dict[str, Callable[..., Any]]:
    """Load selected pure functions from an existing repository module.

    The Dyn3D evaluation modules also import Transformers, baukit, datasets and
    the external TruthfulQA package at module import time.  This cosine
    experiment only needs their pure checkpoint/geometry functions, and should
    remain runnable from the much lighter training environment.  Parsing and
    compiling the selected function definitions reuses the *actual repository
    implementation* without copying its formulas or importing unrelated LM
    evaluation dependencies.
    """
    tree = ast.parse(source_path.read_text(encoding="utf-8"), filename=str(source_path))
    wanted = set(names)
    selected = [
        node
        for node in tree.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name in wanted
    ]
    found = {node.name for node in selected}
    missing = wanted.difference(found)
    if missing:
        raise RuntimeError(
            f"Could not find required functions {sorted(missing)} in {source_path}"
        )
    module = ast.Module(body=selected, type_ignores=[])
    ast.fix_missing_locations(module)
    env: dict[str, Any] = dict(namespace)
    exec(compile(module, str(source_path), "exec"), env, env)
    return {name: env[name] for name in names}


def load_dyn3d_runtime(path: Path, device: torch.device):
    """Load Dyn3D checkpoint plus the existing direct-SLERP geometry functions.

    Only the required function definitions are loaded from the repository
    source, avoiding unrelated language-model evaluation dependencies.
    """
    _require_file(path, "Dyn3D checkpoint")

    loader = _load_existing_functions(
        ROOT / "method" / "dyn3d_sga" / "evaluate_dynSVD.py",
        ("load_components",),
        {"np": np, "torch": torch, "ResidualRouter": ResidualRouter},
    )["load_components"]
    geometry = _load_existing_functions(
        ROOT / "method" / "dyn3d_sga" / "evaluate_3d_dyn.py",
        ("norm", "adaptive_3way_weights", "slerp"),
        {"np": np, "torch": torch, "F": F},
    )

    data, avg, basis, router = loader(str(path), device)
    return (
        data,
        avg,
        basis,
        router,
        geometry["norm"],
        geometry["adaptive_3way_weights"],
        geometry["slerp"],
    )


def predict_dyn3d_vectors(
    z_neg: np.ndarray,
    *,
    avg: torch.Tensor,
    basis: torch.Tensor,
    router: torch.nn.Module,
    norm_fn: Callable[..., torch.Tensor],
    adaptive_weights_fn: Callable[
        ..., tuple[torch.Tensor, torch.Tensor, torch.Tensor, dict]
    ],
    slerp_fn: Callable[..., torch.Tensor],
    weight_kappa: float,
    device: torch.device,
    batch_size: int,
) -> np.ndarray:
    """Return the direct mean/residual SLERP steering vector.

    For each negative state:

    1. normalize ``z_neg`` and feed it to the trained router;
    2. form the L2-normalized routed residual from the SVD basis;
    3. compute the existing adaptive ``alpha, beta, gamma`` weights using
       ``z_neg``, normalized ``v_avg``, and the residual;
    4. discard the original-state branch from the final interpolation and set
       ``t = gamma / (beta + gamma)``;
    5. return ``SLERP(norm(v_avg), residual, t)`` directly as ``v_pred``.

    There is no symmetric 2-SLERP, no multiplication by ``||z_neg||``, and no
    subtraction of ``z_neg`` in this cosine-similarity experiment.
    """
    eps = 1e-8
    mean_direction = norm_fn(avg, eps)
    outputs: list[np.ndarray] = []
    router.eval()

    with torch.inference_mode():
        for start in range(0, len(z_neg), batch_size):
            original = torch.as_tensor(
                z_neg[start : start + batch_size],
                dtype=torch.float32,
                device=device,
            )

            # z_neg still conditions the router and adaptive weights.
            a = norm_fn(original, eps)
            coeff = router(a)
            residual = normalized_basis_combination(coeff, basis).float()

            # Both endpoints are unit directions.
            b = mean_direction.view(1, -1).expand_as(a)
            c = residual

            _alpha, beta, gamma, _ = adaptive_weights_fn(
                a,
                b,
                c,
                kappa=weight_kappa,
                temperature=1.0,
            )

            # Remove the original-state branch from the final interpolation.
            # beta/gamma are positive softmax weights, but clamp for numerical
            # robustness and clamp t to its geometric interpolation interval.
            directional_mass = (beta + gamma).clamp_min(eps)
            t = (gamma / directional_mass).clamp(0.0, 1.0)

            predicted_vector = slerp_fn(
                b,
                c,
                t,
                eps,
                1e-4,
            )
            outputs.append(
                predicted_vector.detach().float().cpu().numpy()
            )

    return np.concatenate(outputs, axis=0).astype(np.float32, copy=False)



def predict_dynmlp_vectors(
    z_neg: np.ndarray,
    *,
    avg: torch.Tensor,
    model: torch.nn.Module,
    norm_fn: Callable[..., torch.Tensor],
    adaptive_weights_fn: Callable[
        ..., tuple[torch.Tensor, torch.Tensor, torch.Tensor, dict]
    ],
    slerp_fn: Callable[..., torch.Tensor],
    weight_kappa: float,
    device: torch.device,
    batch_size: int,
) -> np.ndarray:
    """Return dynMLP's direct mean/residual SLERP steering vector.

    This intentionally mirrors ``predict_dyn3d_vectors``. The only difference
    is how the residual endpoint is predicted:

        Dyn3D : router(normalize(z_neg)) -> SVD basis combination
        dynMLP: MLP(normalize(z_neg))    -> direct residual vector

    The MLP output is L2-normalized before adaptive weighting and SLERP.
    """
    eps = 1e-8
    mean_direction = norm_fn(avg, eps)
    outputs: list[np.ndarray] = []
    model.eval()

    with torch.inference_mode():
        for start in range(0, len(z_neg), batch_size):
            original = torch.as_tensor(
                z_neg[start : start + batch_size],
                dtype=torch.float32,
                device=device,
            )
            a = norm_fn(original, eps)
            residual = F.normalize(
                model(a).float(),
                p=2,
                dim=-1,
                eps=eps,
            )
            b = mean_direction.view(1, -1).expand_as(a)
            c = residual

            _alpha, beta, gamma, _ = adaptive_weights_fn(
                a,
                b,
                c,
                kappa=weight_kappa,
                temperature=1.0,
            )
            directional_mass = (beta + gamma).clamp_min(eps)
            t = (gamma / directional_mass).clamp(0.0, 1.0)
            predicted_vector = slerp_fn(
                b,
                c,
                t,
                eps,
                1e-4,
            )
            outputs.append(
                predicted_vector.detach().float().cpu().numpy()
            )

    return np.concatenate(outputs, axis=0).astype(np.float32, copy=False)


def predict_dynlearnedslerp_vectors(
    z_neg: np.ndarray,
    *,
    avg: torch.Tensor,
    basis: torch.Tensor,
    router: torch.nn.Module,
    residual_params_net: torch.nn.Module,
    device: torch.device,
    batch_size: int,
) -> np.ndarray:
    """Return the learned residual-direction + learned-SLERP prediction."""
    outputs: list[np.ndarray] = []
    router.eval()
    residual_params_net.eval()

    with torch.inference_mode():
        for start in range(0, len(z_neg), batch_size):
            batch = torch.as_tensor(
                z_neg[start : start + batch_size],
                dtype=torch.float32,
                device=device,
            )
            predicted, _ = predict_dynlearnedslerp_direction(
                batch,
                avg,
                basis,
                router,
                residual_params_net,
            )
            outputs.append(
                predicted.detach().float().cpu().numpy()
            )

    return np.concatenate(outputs, axis=0).astype(np.float32, copy=False)

def _scalar_metadata(meta: dict[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for key, value in meta.items():
        arr = np.asarray(value)
        if arr.ndim == 0:
            out[key] = arr.item()
        elif arr.size == 1:
            out[key] = arr.reshape(-1)[0].item()
        else:
            out[key] = arr.tolist()
        if isinstance(out[key], bytes):
            out[key] = out[key].decode("utf-8", errors="replace")
    return out


def evaluate_fold(
    *,
    fold: int,
    num_folds: int,
    z_neg: np.ndarray,
    v_t: np.ndarray,
    q: np.ndarray,
    negative_ids: np.ndarray,
    sadi_mlp_z_neg: np.ndarray,
    l2s_path: Path,
    dyn3d_path: Path,
    dynlearnedslerp_path: Path,
    dynmlp_path: Path,
    sadi_path: Path,
    device: torch.device,
    weight_kappa: float,
    batch_size: int,
) -> tuple[dict[str, dict[str, dict[str, float | int]]], dict[str, Any]]:
    _require_file(l2s_path, "L2S checkpoint")
    _require_file(dyn3d_path, "Dyn3D checkpoint")
    _require_file(dynlearnedslerp_path, "DynLearnedSLERP checkpoint")
    _require_file(dynmlp_path, "dynMLP checkpoint")
    _require_file(sadi_path, "SADI checkpoint")

    l2s_data, l2s_model = load_l2s_checkpoint(l2s_path, device=device)
    sadi_data = None
    dyn_data = None
    dynlearned_data = None
    dynmlp_data = None
    try:
        l2s_train_q, l2s_test_q = _checkpoint_split(
            l2s_data, path=l2s_path, fold=fold, num_folds=num_folds
        )

        sadi_data = load_sadi_checkpoint(sadi_path)
        sadi_gt_mode = str(np.asarray(sadi_data["cosine_gt_mode"]).item())
        if sadi_gt_mode != "mean_all_cartesian_pairs":
            raise ValueError(
                f"Fold {fold}: SADI checkpoint uses cosine_gt_mode={sadi_gt_mode!r}; "
                "retrain SADI with the all-pairs mean cosine metric"
            )
        sadi_train_q, sadi_test_q = _checkpoint_split(
            sadi_data, path=sadi_path, fold=fold, num_folds=num_folds
        )
        if not _same_questions(l2s_train_q, sadi_train_q):
            raise ValueError(
                f"Fold {fold}: L2S and SADI outer-train question IDs differ"
            )
        if not _same_questions(l2s_test_q, sadi_test_q):
            raise ValueError(
                f"Fold {fold}: L2S and SADI held-out question IDs differ"
            )
        sadi_ranked = np.asarray(
            sadi_data["ranked_hidden_indices"], dtype=np.int64
        ).reshape(-1)
        sadi_count = int(np.asarray(sadi_data["best_count"]).item())
        sadi_strength = float(np.asarray(sadi_data["best_strength"]).item())
        if "hidden_dim" in sadi_data.files:
            sadi_dim = int(np.asarray(sadi_data["hidden_dim"]).item())
            if sadi_dim != sadi_mlp_z_neg.shape[1]:
                raise ValueError(
                    f"Fold {fold}: SADI checkpoint hidden_dim={sadi_dim}, "
                    f"feature hidden_dim={sadi_mlp_z_neg.shape[1]}"
                )

        (
            dyn_data,
            dyn_avg,
            dyn_basis,
            dyn_router,
            norm_fn,
            adaptive_weights_fn,
            slerp_fn,
        ) = load_dyn3d_runtime(dyn3d_path, device)
        dyn_train_q, dyn_test_q = _checkpoint_split(
            dyn_data, path=dyn3d_path, fold=fold, num_folds=num_folds
        )
        if not _same_questions(l2s_train_q, dyn_train_q):
            raise ValueError(
                f"Fold {fold}: L2S and Dyn3D outer-train question IDs differ"
            )
        if not _same_questions(l2s_test_q, dyn_test_q):
            raise ValueError(
                f"Fold {fold}: L2S and Dyn3D held-out question IDs differ"
            )

        (
            dynlearned_data,
            dynlearned_avg,
            dynlearned_basis,
            dynlearned_router,
            dynlearned_params,
        ) = load_dynlearnedslerp_checkpoint(
            dynlearnedslerp_path,
            device=device,
        )
        dynlearned_train_q, dynlearned_test_q = _checkpoint_split(
            dynlearned_data,
            path=dynlearnedslerp_path,
            fold=fold,
            num_folds=num_folds,
        )
        if not _same_questions(l2s_train_q, dynlearned_train_q):
            raise ValueError(
                f"Fold {fold}: L2S and DynLearnedSLERP outer-train question IDs differ"
            )
        if not _same_questions(l2s_test_q, dynlearned_test_q):
            raise ValueError(
                f"Fold {fold}: L2S and DynLearnedSLERP held-out question IDs differ"
            )

        dynmlp_data, dynmlp_avg, dynmlp_model = load_dynmlp_checkpoint(
            dynmlp_path,
            device=device,
        )
        dynmlp_train_q, dynmlp_test_q = _checkpoint_split(
            dynmlp_data, path=dynmlp_path, fold=fold, num_folds=num_folds
        )
        if not _same_questions(l2s_train_q, dynmlp_train_q):
            raise ValueError(
                f"Fold {fold}: L2S and dynMLP outer-train question IDs differ"
            )
        if not _same_questions(l2s_test_q, dynmlp_test_q):
            raise ValueError(
                f"Fold {fold}: L2S and dynMLP held-out question IDs differ"
            )

        train_mask = _mask_for_questions(
            q, l2s_train_q, label=f"fold {fold} train"
        )
        test_mask = _mask_for_questions(
            q, l2s_test_q, label=f"fold {fold} test"
        )

        if "geometry_train_q_indices" not in l2s_data.files:
            raise KeyError(
                f"Fold {fold}: L2S checkpoint lacks geometry_train_q_indices "
                "required for L2S-SLERP v_avg"
            )
        l2s_geometry_q = np.asarray(
            l2s_data["geometry_train_q_indices"],
            dtype=np.int64,
        ).reshape(-1)
        l2s_geometry_mask = _mask_for_questions(
            q,
            l2s_geometry_q,
            label=f"fold {fold} L2S-SLERP geometry",
        )
        l2s_slerp_avg_np = v_t[l2s_geometry_mask].mean(
            axis=0,
            dtype=np.float64,
        ).astype(np.float32)
        l2s_slerp_avg = torch.as_tensor(
            l2s_slerp_avg_np,
            dtype=torch.float32,
            device=device,
        )
        if np.any(train_mask & test_mask):
            raise ValueError(f"Fold {fold}: raw-pair train/test masks overlap")

        # Under the current matched protocol, Dyn3D, dynMLP and L2S-SLERP
        # should all use the same pair-level v_avg fitted on geometry-train.
        dyn_avg_np = dyn_avg.detach().float().cpu().numpy()
        dynlearned_avg_np = dynlearned_avg.detach().float().cpu().numpy()
        dynmlp_avg_np = dynmlp_avg.detach().float().cpu().numpy()
        if not np.allclose(l2s_slerp_avg_np, dyn_avg_np, rtol=2e-5, atol=2e-5):
            raise ValueError(
                f"Fold {fold}: L2S-SLERP and Dyn3D v_avg differ under the "
                "matched split protocol"
            )
        if not np.allclose(l2s_slerp_avg_np, dynlearned_avg_np, rtol=2e-5, atol=2e-5):
            raise ValueError(
                f"Fold {fold}: L2S-SLERP and DynLearnedSLERP v_avg differ under the "
                "matched split protocol"
            )
        if not np.allclose(l2s_slerp_avg_np, dynmlp_avg_np, rtol=2e-5, atol=2e-5):
            raise ValueError(
                f"Fold {fold}: L2S-SLERP and dynMLP v_avg differ under the "
                "matched split protocol"
            )

        # Standard Steer uses all raw Cartesian v_T pairs in the complete
        # outer-train fold. Dyn3D/dynMLP retain their own checkpoint v_avg,
        # fitted on the shared geometry-training subset.
        standard_v_avg = v_t[train_mask].mean(
            axis=0,
            dtype=np.float64,
        ).astype(np.float32)

        fold_results: dict[str, dict[str, dict[str, float | int]]] = {
            "train": {},
            "test": {},
        }
        for split, mask in (("train", train_mask), ("test", test_mask)):
            split_z = z_neg[mask]
            split_v = v_t[mask]
            split_negative_ids = negative_ids[mask]
            split_sadi_mlp_z = sadi_mlp_z_neg[mask]

            l2s_pred = predict_l2s(
                l2s_model,
                split_z,
                device=device,
                batch_size=batch_size,
            )
            l2s_slerp_pred = predict_l2s_slerp_vectors(
                split_z,
                l2s_pred,
                avg=l2s_slerp_avg,
                weight_kappa=weight_kappa,
                device=device,
                batch_size=batch_size,
            )
            steer_pred = np.broadcast_to(
                standard_v_avg.reshape(1, -1),
                split_v.shape,
            ).copy()
            dyn_pred = predict_dyn3d_vectors(
                split_z,
                avg=dyn_avg,
                basis=dyn_basis,
                router=dyn_router,
                norm_fn=norm_fn,
                adaptive_weights_fn=adaptive_weights_fn,
                slerp_fn=slerp_fn,
                weight_kappa=weight_kappa,
                device=device,
                batch_size=batch_size,
            )
            dynlearned_pred = predict_dynlearnedslerp_vectors(
                split_z,
                avg=dynlearned_avg,
                basis=dynlearned_basis,
                router=dynlearned_router,
                residual_params_net=dynlearned_params,
                device=device,
                batch_size=batch_size,
            )
            dynmlp_pred = predict_dynmlp_vectors(
                split_z,
                avg=dynmlp_avg,
                model=dynmlp_model,
                norm_fn=norm_fn,
                adaptive_weights_fn=adaptive_weights_fn,
                slerp_fn=slerp_fn,
                weight_kappa=weight_kappa,
                device=device,
                batch_size=batch_size,
            )
            sadi_pred = predict_hidden_vectors(
                split_sadi_mlp_z,
                sadi_ranked,
                count=sadi_count,
                strength=sadi_strength,
            )

            fold_results[split]["l2s"] = _summarize_cosine(
                l2s_pred, split_v, split_negative_ids
            )
            fold_results[split]["l2s_slerp"] = _summarize_cosine(
                l2s_slerp_pred, split_v, split_negative_ids
            )
            fold_results[split]["steer"] = _summarize_cosine(
                steer_pred, split_v, split_negative_ids
            )
            fold_results[split]["dyn3d"] = _summarize_cosine(
                dyn_pred, split_v, split_negative_ids
            )
            fold_results[split]["DynLearnedSLERP"] = _summarize_cosine(
                dynlearned_pred, split_v, split_negative_ids
            )
            fold_results[split]["dynMLP"] = _summarize_cosine(
                dynmlp_pred, split_v, split_negative_ids
            )
            fold_results[split]["sadi"] = _summarize_cosine(
                sadi_pred, split_v, split_negative_ids
            )

        fold_meta = {
            "fold": int(fold),
            "outer_train_questions": int(len(l2s_train_q)),
            "test_questions": int(len(l2s_test_q)),
            "outer_train_q_indices": l2s_train_q.tolist(),
            "test_q_indices": l2s_test_q.tolist(),
            "standard_steer_v_avg_norm": float(
                np.linalg.norm(standard_v_avg)
            ),
            "l2s_slerp_v_avg_norm": float(
                np.linalg.norm(l2s_slerp_avg_np)
            ),
            "l2s_slerp_geometry_q_indices": l2s_geometry_q.tolist(),
            "dyn3d_checkpoint_v_avg_norm": float(
                torch.linalg.vector_norm(dyn_avg).detach().cpu().item()
            ),
            "dynlearnedslerp_checkpoint_v_avg_norm": float(
                torch.linalg.vector_norm(dynlearned_avg).detach().cpu().item()
            ),
            "dynmlp_checkpoint_v_avg_norm": float(
                torch.linalg.vector_norm(dynmlp_avg).detach().cpu().item()
            ),
            "l2s_checkpoint": str(l2s_path),
            "dyn3d_checkpoint": str(dyn3d_path),
            "dynlearnedslerp_checkpoint": str(dynlearnedslerp_path),
            "dynmlp_checkpoint": str(dynmlp_path),
            "sadi_checkpoint": str(sadi_path),
            "sadi_best_count": int(sadi_count),
            "sadi_best_strength": float(sadi_strength),
            "sadi_best_val_cosine": (
                float(np.asarray(sadi_data["best_val_cosine"]).item())
                if "best_val_cosine" in sadi_data.files
                else None
            ),
        }
        return fold_results, fold_meta
    finally:
        if dynmlp_data is not None:
            dynmlp_data.close()
        if dynlearned_data is not None:
            dynlearned_data.close()
        if dyn_data is not None:
            dyn_data.close()
        if sadi_data is not None:
            sadi_data.close()
        l2s_data.close()


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def write_csv(path: Path, results: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    fields = [
        "fold",
        "split",
        "method",
        "cosine",
        "std",
        "min",
        "max",
        "num_pairs",
        "num_negatives",
        "zero_prediction_count",
        "zero_target_count",
    ]
    with tmp.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for fold_key in sorted(results, key=lambda x: int(x.replace("fold", ""))):
            fold = int(fold_key.replace("fold", ""))
            for split in SPLIT_ORDER:
                for method in METHOD_ORDER:
                    row = {"fold": fold, "split": split, "method": method}
                    row.update(results[fold_key][split][method])
                    writer.writerow(row)
    os.replace(tmp, path)


def print_fold(fold: int, fold_results: dict[str, Any]) -> None:
    print(f"\nFold {fold}")
    print("-" * 72)
    for split in SPLIT_ORDER:
        print(f"{split.upper()}:")
        for method in METHOD_ORDER:
            stats = fold_results[split][method]
            print(
                f"  {method:>6s}  cosine={stats['cosine']:.6f}  "
                f"std={stats['std']:.6f}  negatives={stats['num_negatives']} "
                f"(pairs={stats['num_pairs']})"
            )


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)
    (
        z_neg,
        v_t,
        q,
        negative_ids,
        sadi_mlp_z_neg,
        feature_meta,
    ) = load_raw_pair_features(args.feature_path)

    results: dict[str, Any] = {}
    folds_meta: dict[str, Any] = {}
    for fold in args.folds:
        l2s_path = render_fold_path(args.l2s_checkpoint_template, fold)
        dyn3d_path = render_fold_path(args.dyn3d_checkpoint_template, fold)
        dynlearnedslerp_path = render_fold_path(
            args.dynlearnedslerp_checkpoint_template, fold
        )
        dynmlp_path = render_fold_path(args.dynmlp_checkpoint_template, fold)
        sadi_path = render_fold_path(args.sadi_checkpoint_template, fold)
        fold_results, fold_meta = evaluate_fold(
            fold=fold,
            num_folds=args.num_folds,
            z_neg=z_neg,
            v_t=v_t,
            q=q,
            negative_ids=negative_ids,
            sadi_mlp_z_neg=sadi_mlp_z_neg,
            l2s_path=l2s_path,
            dyn3d_path=dyn3d_path,
            dynlearnedslerp_path=dynlearnedslerp_path,
            dynmlp_path=dynmlp_path,
            sadi_path=sadi_path,
            device=device,
            weight_kappa=args.weight_kappa,
            batch_size=args.batch_size,
        )
        key = f"fold{fold}"
        results[key] = fold_results
        folds_meta[key] = fold_meta
        print_fold(fold, fold_results)

    payload = {
        "experiment": "raw_vT_cosine_similarity",
        "definitions": {
            "ground_truth": "extractor raw v_T = z_pos_raw - z_neg_raw",
            "l2s": "direct MLP prediction from raw z_neg_raw",
            "l2s_slerp": (
                "reuse direct L2S v_hat; residual = normalize(v_hat - v_avg) "
                "with v_avg from the shared geometry-train Cartesian v_T pairs; "
                "then use the same adaptive_3way_weights as Dyn3D on normalize(z_neg), "
                "normalize(v_avg), and residual; set t=w_residual/(w_mean+w_residual); "
                "finally restore the per-sample raw L2S norm ||v_hat||"
            ),
            "standard_steer": "mean raw v_T over the complete outer-train fold",
            "dyn3d": (
                "existing checkpoint router/basis/v_avg + existing adaptive_3way_weights; "
                "z_neg conditions router/weights only; predicted steering vector = "
                "slerp(norm(v_avg), normalized routed residual, gamma/(beta+gamma)); "
                "no two_slerp_sym, norm rescaling, candidate-z_neg, or margin gate"
            ),
            "DynLearnedSLERP": (
                "renamed uploaded learned-SLERP Dyn3D variant: sparse Top-K softmax Router "
                "predicts an SVD residual direction, an independent ResidualParamsNet predicts "
                "t in [0,1], and the cosine prediction is "
                "slerp(norm(v_avg), residual_direction, t); training uses residual-direction "
                "cosine + t MSE + joint final-SLERP cosine + load-balance losses"
            ),
            "dynMLP": (
                "direct MLP residual ablation: train on the same Cartesian pair residual "
                "target normalize(v_T - v_avg), predict residual from normalize(z_neg), "
                "then use the same adaptive weights and direct "
                "slerp(norm(v_avg), normalized MLP residual, gamma/(beta+gamma)) as Dyn3D"
            ),
            "sadi": (
                "SADI hidden branch at the target layer: rank MLP-output hidden units on "
                "geometry-train true-minus-false differences; choose count/strength on validation; "
                "predicted decoder-layer delta on selected units = (strength - 1) * mlp_z_neg"
            ),
            "gt_pairing": (
                "evaluate every positive-negative Cartesian pair independently and take the "
                "arithmetic mean of all pairwise cosine values; no nearest/best GT selection"
            ),
        },
        "feature_path": str(Path(args.feature_path)),
        "feature_metadata": _scalar_metadata(feature_meta),
        "num_folds": int(args.num_folds),
        "evaluated_folds": [int(x) for x in args.folds],
        "weight_kappa": float(args.weight_kappa),
        "l2s_slerp_adaptive_t": "w_residual/(w_mean+w_residual)",
        "device": str(device),
        "fold_metadata": folds_meta,
        "results": results,
    }

    output_dir = Path(args.output_dir)
    json_path = output_dir / f"{args.output_stem}.json"
    csv_path = output_dir / f"{args.output_stem}.csv"
    write_json(json_path, payload)
    write_csv(csv_path, results)
    print(f"\nSaved JSON: {json_path}")
    print(f"Saved CSV : {csv_path}")


if __name__ == "__main__":
    main()
