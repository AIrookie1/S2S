from __future__ import annotations
import argparse, json, sys
from pathlib import Path
import numpy as np
ROOT = Path.cwd()
if str(ROOT) not in sys.path: sys.path.insert(0, str(ROOT))
from method.dyn3d_sga.dynSVD import fit_dynamic_svd, load_feature_pairs

def main() -> None:
    p = argparse.ArgumentParser(description="Train dynSVD router and scale network")
    p.add_argument("--feature_path", "--prototype_path", dest="feature_path", required=True); p.add_argument("--output_path", required=True)
    p.add_argument("--rank", type=int, default=None, help="Number of SVD basis vectors to use (manual rank); when omitted, --energy selects the rank automatically.")
    p.add_argument("--positive_mode", choices=["mean", "mc1"], default="mean", help="Construct training pairs from answer-level activations using mean MC2 positives or one MC1 positive.")
    p.add_argument("--energy", type=float, default=.95, help="Cumulative squared-singular-value energy target in (0, 1]; used when --rank is omitted."); p.add_argument("--num_folds", type=int, default=2); p.add_argument("--fold_idx", type=int, default=0); p.add_argument("--val_fraction", type=float, default=0.5, help="Question-level validation fraction within the selected fold (default 0.2)."); p.add_argument("--router_hidden", type=int, default=256); p.add_argument("--scale_loss_weight", type=float, default=1.0, help="Weight of scale-vs-vector-norm MSE loss."); p.add_argument("--cosine_loss_weight", type=float, default=1.0, help="Weight of cosine-direction loss."); p.add_argument("--use_zpos_router", type=int, choices=[0, 1], default=1, help="Legacy z+ switch; overridden by use_data_mode."); p.add_argument("--use_data_mode", type=int, choices=[0, 1, 2, 3], default=0, help="0=neg, 1=neg+pos, 2=neg+auxiliary interpolation, 3=neg+pos+auxiliary interpolation."); p.add_argument("--aux_interp_count", type=int, default=1, help="Number of random interpolation samples per pair when mode 2/3 is selected."); p.add_argument("--epochs", type=int, default=20); p.add_argument("--batch_size", type=int, default=128); p.add_argument("--learning_rate", "--lr", type=float, default=1e-3); p.add_argument("--device", choices=["cuda", "cpu"], default="cuda"); p.add_argument("--seed", type=int, default=42)
    args = p.parse_args(); hneg, hpos, q, meta = load_feature_pairs(args.feature_path, positive_mode=args.positive_mode)
    if args.num_folds < 1 or not 0 <= args.fold_idx < args.num_folds:
        raise ValueError("num_folds must be positive and fold_idx must be in range")
    if not 0.0 < args.val_fraction < 1.0:
        raise ValueError("val_fraction must be in (0, 1)")
    if "train_q_indices" in meta and "test_q_indices" in meta:
        # Fold artifacts store the selected fold in test_q_indices and the
        # opposite, held-out fold in train_q_indices.
        selected_q = np.asarray(meta["test_q_indices"], dtype=np.int64).reshape(-1)
        heldout_q = np.asarray(meta["train_q_indices"], dtype=np.int64).reshape(-1)
    else:
        questions = np.unique(q); folds = np.array_split(questions, args.num_folds)
        selected_q = np.asarray(folds[args.fold_idx], dtype=np.int64)
        heldout_folds = [folds[i] for i in range(args.num_folds) if i != args.fold_idx]
        heldout_q = (
            np.concatenate(heldout_folds).astype(np.int64)
            if heldout_folds else np.empty(0, dtype=np.int64)
        )
    if len(selected_q) < 2:
        raise ValueError("selected fold must contain at least two questions for an 8:2 split")
    # Only the selected fold is used. Split its questions deterministically
    # into the router-validation subset (20%) and its complementary subset;
    # The selected fold is used for all geometry. Its 80% subset supplies
    # v_avg, while the complete fold supplies the SVD residual basis.
    rng = np.random.default_rng(args.seed)
    shuffled_q = rng.permutation(np.unique(selected_q))
    n_val = int(round(len(shuffled_q) * args.val_fraction))
    n_val = max(1, min(len(shuffled_q) - 1, n_val))
    geometry_train_q = np.sort(shuffled_q[:-n_val])
    val_q = np.sort(shuffled_q[-n_val:])
    test_q = np.sort(np.unique(heldout_q))
    # SVD/mean use every pair from the selected fold, including the questions
    # reserved for Router/ScaleNet validation. The held-out fold is excluded.
    geometry_mask = np.isin(q, selected_q)
    mean_mask = np.isin(q, geometry_train_q)
    router_mask = np.isin(q, val_q)
    print(f"Selected fold {args.fold_idx}: {len(selected_q)} questions; held-out fold unused for training: {len(test_q)} questions")
    print(f"Within-fold split: SVD-all={len(selected_q)} questions / mean-train={len(geometry_train_q)} questions (80%) / router-validation={len(val_q)} questions ({args.val_fraction:.0%})")
    state, history, metrics = fit_dynamic_svd(hneg, hpos, q, geometry_mask, args.rank, args.energy, args.router_hidden, args.epochs, args.batch_size, args.learning_rate, args.device, args.seed, args.scale_loss_weight, args.cosine_loss_weight, router_mask=router_mask, use_zpos_router=bool(args.use_zpos_router), use_data_mode=args.use_data_mode, aux_interp_count=args.aux_interp_count, mean_mask=mean_mask)
    out = Path(args.output_path); out.parent.mkdir(parents=True, exist_ok=True); router = state["router"].state_dict(); scale_net = state["scale_net"].state_dict()
    payload = {"method": np.asarray("dynSVD"), "hidden_dim": np.asarray(state["hidden_dim"], np.int64), "rank": np.asarray(state["rank"], np.int64), "router_hidden": np.asarray(state["router_hidden"], np.int64), "v_avg": state["v_avg"].numpy().astype(np.float32), "basis": state["basis"].numpy().astype(np.float32), "history_json": np.asarray(json.dumps(history)), "metrics_json": np.asarray(json.dumps({k: v for k, v in metrics.items() if k != "singular_values"})), "singular_values": metrics["singular_values"].astype(np.float32), "outer_train_q_indices": selected_q, "train_q_indices": state["train_q_indices"], "val_q_indices": val_q, "router_q_indices": state["router_q_indices"], "test_q_indices": test_q, "fold_idx": np.asarray(args.fold_idx, np.int64), "num_folds": np.asarray(args.num_folds, np.int64), "feature_path": np.asarray(str(args.feature_path)), "positive_mode": np.asarray(meta.get("positive_mode", "unknown")).item(), "use_data_mode": np.asarray(args.use_data_mode, np.int64), "aux_interp_count": np.asarray(args.aux_interp_count, np.int64)}
    for key, value in router.items(): payload[f"router_{key}"] = value.numpy()
    for key, value in scale_net.items(): payload[f"scale_{key}"] = value.numpy()
    np.savez_compressed(out, **payload); print(f"Saved dynSVD checkpoint to {out}"); print(f"rank={state['rank']} explained_energy={metrics['explained_energy']:.4f} svd_mse={metrics['svd_reconstruction_mse']:.6e}")
if __name__ == "__main__": main()
