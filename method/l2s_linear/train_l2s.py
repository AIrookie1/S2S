"""Train the L2S-LINEAR direct MLP vector predictor."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from method.l2s_linear.model import fit_l2s, load_l2s_pairs, split_like_dyn3d


def main() -> None:
    parser = argparse.ArgumentParser(description="Train L2S-LINEAR direct MLP vector predictor")
    parser.add_argument("--feature_path", required=True)
    parser.add_argument("--output_path", required=True)
    parser.add_argument("--num_folds", type=int, default=2)
    parser.add_argument("--fold_idx", type=int, default=0)
    parser.add_argument("--val_fraction", type=float, default=0.5)
    parser.add_argument("--mlp_hidden", "--hidden_dim", dest="mlp_hidden", type=int, default=256)
    parser.add_argument("--epochs", type=int, default=2000)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--learning_rate", "--lr", dest="learning_rate", type=float, default=1e-3)
    parser.add_argument("--device", choices=["cuda", "cpu"], default="cuda")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    z_neg, vectors, q_indices, metadata = load_l2s_pairs(args.feature_path)
    split = split_like_dyn3d(
        q_indices,
        metadata,
        num_folds=args.num_folds,
        fold_idx=args.fold_idx,
        val_fraction=args.val_fraction,
        seed=args.seed,
    )
    # Use the same within-fold optimization subset as Dyn3D's direct MLP so
    # the comparison retains identical question-level splitting.
    optimization_q = split["predictor_q_indices"]
    print(
        f"Selected fold {args.fold_idx}: outer-train={len(split['outer_train_q_indices'])} "
        f"held-out={len(split['test_q_indices'])}"
    )
    print(
        f"Within-fold split: mean-train={len(split['geometry_train_q_indices'])} "
        f"predictor/L2S-LINEAR-train={len(optimization_q)} "
        f"val_fraction={args.val_fraction:.3f}"
    )

    model, history, metrics = fit_l2s(
        z_neg,
        vectors,
        q_indices,
        optimization_q,
        hidden_dim=args.mlp_hidden,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        device=args.device,
        seed=args.seed,
    )

    output = Path(args.output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    state = model.state_dict()
    payload = {
        "method": np.asarray("L2S-LINEAR"),
        "hidden_dim": np.asarray(z_neg.shape[1], dtype=np.int64),
        "mlp_hidden": np.asarray(args.mlp_hidden, dtype=np.int64),
        "epochs": np.asarray(args.epochs, dtype=np.int64),
        "batch_size": np.asarray(args.batch_size, dtype=np.int64),
        "learning_rate": np.asarray(args.learning_rate, dtype=np.float32),
        "seed": np.asarray(args.seed, dtype=np.int64),
        "val_fraction": np.asarray(args.val_fraction, dtype=np.float32),
        "fold_idx": np.asarray(args.fold_idx, dtype=np.int64),
        "num_folds": np.asarray(args.num_folds, dtype=np.int64),
        "feature_path": np.asarray(str(args.feature_path)),
        "target_definition": np.asarray("v_T = z_pos_raw - z_neg_raw"),
        "normalization_policy": np.asarray(
            "raw z_neg inputs and raw v_T targets during training and direct additive inference; no L2 normalization is applied by L2S-LINEAR"
        ),
        "history_json": np.asarray(json.dumps(history, ensure_ascii=False)),
        "metrics_json": np.asarray(json.dumps(metrics, ensure_ascii=False)),
        # Keep the exact Dyn3D split names, plus an explicit L2S-LINEAR training name.
        "outer_train_q_indices": split["outer_train_q_indices"],
        "geometry_train_q_indices": split["geometry_train_q_indices"],
        # Keep Dyn3D's checkpoint convention: train_q_indices denotes the
        # complete selected outer fold.  The actual L2S-LINEAR optimizer subset is
        # recorded separately below, just as Dyn3D records its MLP subset.
        "train_q_indices": split["outer_train_q_indices"],
        "val_q_indices": split["val_q_indices"],
        "predictor_q_indices": split["predictor_q_indices"],
        "l2s_linear_train_q_indices": optimization_q,
        "test_q_indices": split["test_q_indices"],
    }
    for key, value in state.items():
        payload[f"mlp_{key}"] = value.detach().cpu().numpy().astype(np.float32)
    np.savez_compressed(output, **payload)
    print(f"Saved L2S-LINEAR checkpoint to {output}")
    print(
        f"train_mse={metrics['train_mse']:.6e} all_mse={metrics['all_mse']:.6e} "
        f"pairs={int(metrics['train_pairs'])}/{int(metrics['all_pairs'])}"
    )


if __name__ == "__main__":
    main()
