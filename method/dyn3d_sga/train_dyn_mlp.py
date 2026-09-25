"""Train Dyn3D's direct MLP vector predictor."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from method.dyn3d_sga.dyn_mlp import fit_dynamic_mlp, load_feature_pairs


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train Dyn3D direct MLP: normalize(h_neg) -> v_i"
    )
    parser.add_argument(
        "--feature_path",
        "--prototype_path",
        dest="feature_path",
        required=True,
    )
    parser.add_argument("--output_path", required=True)
    parser.add_argument(
        "--positive_mode",
        choices=["mean", "mc1"],
        default="mean",
        help=(
            "For mean mode, each negative answer is paired with the mean of "
            "that question's positive answer activations."
        ),
    )
    parser.add_argument("--num_folds", type=int, default=2)
    parser.add_argument("--fold_idx", type=int, default=0)
    parser.add_argument(
        "--val_fraction",
        type=float,
        default=0.5,
        help=(
            "Fraction of questions in the selected outer fold allocated to "
            "direct-MLP training. The complementary subset estimates v_avg."
        ),
    )
    parser.add_argument(
        "--mlp_hidden",
        "--hidden_dim",
        dest="mlp_hidden",
        type=int,
        default=256,
    )
    parser.add_argument("--epochs", type=int, default=2000)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument(
        "--learning_rate",
        "--lr",
        dest="learning_rate",
        type=float,
        default=1e-3,
    )
    parser.add_argument("--device", choices=["cuda", "cpu"], default="cuda")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--target_noise_sigma", type=float, default=0.0,
                        help="Relative Gaussian noise sampled once per MLP target before training and then held fixed across epochs/batches.")
    parser.add_argument("--noise_seed", type=int, default=None,
                        help="Noise RNG seed; defaults to --seed.")
    args = parser.parse_args()

    h_neg, h_pos, q_indices, metadata = load_feature_pairs(
        args.feature_path,
        positive_mode=args.positive_mode,
    )
    if args.num_folds < 1 or not 0 <= args.fold_idx < args.num_folds:
        raise ValueError("num_folds must be positive and fold_idx must be in range")
    if not 0.0 < args.val_fraction < 1.0:
        raise ValueError("val_fraction must be in (0, 1)")

    if {"train_q_indices", "test_q_indices"}.issubset(metadata):
        selected_q = np.asarray(
            metadata["test_q_indices"], dtype=np.int64
        ).reshape(-1)
        heldout_q = np.asarray(
            metadata["train_q_indices"], dtype=np.int64
        ).reshape(-1)
    else:
        questions = np.unique(q_indices)
        folds = np.array_split(questions, args.num_folds)
        selected_q = np.asarray(folds[args.fold_idx], dtype=np.int64)
        heldout_parts = [
            folds[i] for i in range(args.num_folds) if i != args.fold_idx
        ]
        heldout_q = (
            np.concatenate(heldout_parts).astype(np.int64)
            if heldout_parts
            else np.empty(0, dtype=np.int64)
        )

    selected_q = np.sort(np.unique(selected_q))
    heldout_q = np.sort(np.unique(heldout_q))
    if len(selected_q) < 2:
        raise ValueError(
            "selected fold must contain at least two questions for the mean/MLP split"
        )

    rng = np.random.default_rng(args.seed)
    shuffled_q = rng.permutation(selected_q)
    n_mlp = int(round(len(shuffled_q) * args.val_fraction))
    n_mlp = max(1, min(len(shuffled_q) - 1, n_mlp))
    mean_train_q = np.sort(shuffled_q[:-n_mlp])
    mlp_train_q = np.sort(shuffled_q[-n_mlp:])

    mean_mask = np.isin(q_indices, mean_train_q)
    mlp_mask = np.isin(q_indices, mlp_train_q)

    print(
        f"Selected fold {args.fold_idx}: {len(selected_q)} questions; "
        f"held-out={len(heldout_q)}"
    )
    print(
        f"Within-fold split: mean-train={len(mean_train_q)} questions "
        f"({1.0 - args.val_fraction:.0%}) / direct-MLP-train={len(mlp_train_q)} "
        f"questions ({args.val_fraction:.0%})"
    )

    state, history, metrics = fit_dynamic_mlp(
        h_neg,
        h_pos,
        q_indices,
        mean_mask,
        mlp_mask,
        mlp_hidden=args.mlp_hidden,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        device=args.device,
        seed=args.seed,
        target_noise_sigma=args.target_noise_sigma,
        noise_seed=args.noise_seed,
    )

    output = Path(args.output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    predictor_state = state["predictor"].state_dict()
    positive_mode = metadata.get("positive_mode", np.asarray(args.positive_mode))
    positive_mode_value = str(np.asarray(positive_mode).item())

    payload = {
        "method": np.asarray("dyn3d_direct_mlp"),
        "hidden_dim": np.asarray(state["hidden_dim"], np.int64),
        "mlp_hidden": np.asarray(state["mlp_hidden"], np.int64),
        "val_fraction": np.asarray(args.val_fraction, np.float32),
        "epochs": np.asarray(args.epochs, np.int64),
        "batch_size": np.asarray(args.batch_size, np.int64),
        "learning_rate": np.asarray(args.learning_rate, np.float32),
        "seed": np.asarray(args.seed, np.int64),
        "target_noise_sigma": np.asarray(args.target_noise_sigma, np.float32),
        "noise_seed": np.asarray(args.seed if args.noise_seed is None else args.noise_seed, np.int64),
        "target_noise_definition": np.asarray("v_tilde = v + sigma * ||v||_2 * normalize(epsilon), epsilon ~ N(0,I); MLP supervision only"),
        "target_noise_sampling": np.asarray("one draw per MLP target before optimization; fixed across all epochs and batches"),
        "v_avg": state["v_avg"].numpy().astype(np.float32),
        "history_json": np.asarray(json.dumps(history, ensure_ascii=False)),
        "metrics_json": np.asarray(json.dumps(metrics, ensure_ascii=False)),
        "outer_train_q_indices": selected_q,
        "train_q_indices": selected_q,
        "mean_train_q_indices": state["mean_q_indices"],
        "mlp_train_q_indices": state["mlp_q_indices"],
        "test_q_indices": heldout_q,
        "fold_idx": np.asarray(args.fold_idx, np.int64),
        "num_folds": np.asarray(args.num_folds, np.int64),
        "feature_path": np.asarray(str(args.feature_path)),
        "positive_mode": np.asarray(positive_mode_value),
        "input_definition": np.asarray("normalize(h_neg)"),
        "target_definition": np.asarray("v_i = h_pos - h_neg"),
        "residual_definition": np.asarray("normalize(v_hat_i - v_avg)"),
    }
    for key, value in predictor_state.items():
        payload[f"mlp_{key}"] = value.detach().cpu().numpy().astype(np.float32)

    np.savez_compressed(output, **payload)
    print(f"Saved Dyn3D direct-MLP checkpoint to {output}")
    print(
        f"vector_mse={metrics['train_vector_mse']:.6e} "
        f"residual_cosine={metrics['train_residual_cosine']:.6f} "
        f"pairs={metrics['mlp_train_pairs']}"
    )


if __name__ == "__main__":
    main()
