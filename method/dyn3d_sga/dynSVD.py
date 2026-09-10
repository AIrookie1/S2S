"""Dynamic SVD Twin-Steering: shared mean vector, routed basis coefficients, and learned scale."""
from __future__ import annotations

import math
import random
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn


class ResidualRouter(nn.Module):
    def __init__(
        self,
        input_dim: int,
        rank: int,
        hidden_dim: int = 256,
        shared_fc1: nn.Linear | None = None,
        top_k: int | None = None,
    ) -> None:
        super().__init__()
        if rank <= 0 or hidden_dim <= 0:
            raise ValueError("rank and hidden_dim must be positive")
        if top_k is not None and top_k <= 0:
            raise ValueError("top_k must be positive")
        self.fc1 = shared_fc1 if shared_fc1 is not None else nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, rank)
        self.input_dim, self.rank, self.hidden_dim = input_dim, rank, hidden_dim
        # ``None`` preserves the legacy dense router. New training passes an
        # explicit Top-K value and stores it in the checkpoint.
        self.top_k = rank if top_k is None else min(int(top_k), rank)

    def logits(self, z: torch.Tensor) -> torch.Tensor:
        return self.fc2(F.gelu(self.fc1(z)))

    def route_logits(self, logits: torch.Tensor) -> torch.Tensor:
        """Apply sparse Top-K routing and renormalize selected coefficients."""
        if logits.shape[-1] != self.rank:
            raise ValueError(
                f"router logits last dim must be rank={self.rank}, got {logits.shape[-1]}"
            )
        if self.top_k >= self.rank:
            return F.softmax(logits, dim=-1)
        topk_idx = torch.topk(logits, k=self.top_k, dim=-1).indices
        selected = torch.zeros_like(logits, dtype=torch.bool).scatter_(-1, topk_idx, True)
        sparse_logits = logits.masked_fill(~selected, float("-inf"))
        return F.softmax(sparse_logits, dim=-1)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        # Training and evaluation intentionally share exactly the same sparse
        # routing path so a Top-K checkpoint never falls back to dense softmax.
        return self.route_logits(self.logits(z))


def router_load_balancing_loss(logits: torch.Tensor, top_k: int) -> torch.Tensor:
    """Switch-style auxiliary loss for balancing Top-K basis utilization.

    ``importance`` is the mean dense router probability and ``load`` is the
    normalized fraction of Top-K assignments received by each basis. The hard
    load is treated as a routing statistic while gradients flow through the
    dense probabilities, including currently unselected bases. A perfectly
    balanced assignment has loss near 1; collapse increases the loss.
    """
    if logits.ndim != 2:
        raise ValueError("load-balancing loss expects [batch, num_bases] logits")
    num_bases = logits.shape[-1]
    if num_bases <= 0 or top_k <= 0:
        raise ValueError("num_bases and top_k must be positive")
    k = min(int(top_k), num_bases)
    if k >= num_bases:
        # Every basis is selected, so dispatch load is already perfectly even.
        return logits.new_zeros(())
    dense_prob = F.softmax(logits, dim=-1)
    topk_idx = torch.topk(logits, k=k, dim=-1).indices
    assignments = torch.zeros_like(dense_prob).scatter_(-1, topk_idx, 1.0)
    load = (assignments.mean(dim=0) / float(k)).detach()
    importance = dense_prob.mean(dim=0)
    return num_bases * torch.sum(load * importance)


def router_expert_load_fraction(logits: torch.Tensor, top_k: int) -> torch.Tensor:
    """Return normalized Top-K assignment frequency for each routed basis."""
    if logits.ndim != 2:
        raise ValueError("expert load expects [batch, num_bases] logits")
    num_bases = logits.shape[-1]
    k = min(int(top_k), num_bases)
    topk_idx = torch.topk(logits, k=k, dim=-1).indices
    assignments = torch.zeros_like(logits).scatter_(-1, topk_idx, 1.0)
    return assignments.mean(dim=0) / float(k)

class ScaleNet(nn.Module):
    """MLP that predicts the scalar applied to the routed residual combination.

    The scalar is supervised against the L2 magnitude of the *raw* residual
    target. Because training deliberately keeps ``coeff @ basis`` unnormalized,
    the final routed residual magnitude is ``scale * ||coeff @ basis||_2``.
    """
    def __init__(self, input_dim: int, hidden_dim: int = 256, shared_fc1: nn.Linear | None = None) -> None:
        super().__init__()
        if input_dim <= 0 or hidden_dim <= 0:
            raise ValueError("input_dim and hidden_dim must be positive")
        self.fc1 = shared_fc1 if shared_fc1 is not None else nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, 1)
        self.input_dim, self.hidden_dim = input_dim, hidden_dim

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return F.softplus(self.fc2(F.gelu(self.fc1(z)))).squeeze(-1)


def raw_basis_combination(coeff: torch.Tensor, basis: torch.Tensor) -> torch.Tensor:
    """Return the routed SVD-basis combination before direction normalization."""
    return coeff @ basis


def normalize_basis_combination(coeff: torch.Tensor, basis: torch.Tensor) -> torch.Tensor:
    """Normalize the routed SVD combination to a unit residual direction.

    This normalization is intentionally retained in Router training:
    ``routed_direction`` is a direction, while ``scale`` supplies its
    magnitude.  Hidden-state geometry used to build v_avg/residual/SVD/targets
    remains on the original, unnormalized scale.
    """
    return F.normalize(raw_basis_combination(coeff, basis), p=2, dim=-1, eps=1e-8)



def set_seed(seed: int) -> None:
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)


def choose_rank(singular_values: torch.Tensor, rank: int | None, energy: float) -> int:
    if rank is not None:
        if rank <= 0: raise ValueError("rank must be positive")
        return min(rank, singular_values.numel())
    if not 0 < energy <= 1: raise ValueError("energy must be in (0, 1]")
    power = singular_values.square(); total = power.sum()
    if float(total) == 0: return 1
    cumulative = torch.cumsum(power, 0) / total
    return int(torch.searchsorted(cumulative, torch.tensor(energy, device=power.device)).item() + 1)


def load_feature_pairs(path: str | Path, positive_mode: str = "mean") -> tuple[np.ndarray, np.ndarray, np.ndarray, dict]:
    """Load answer-level features and construct training pairs on demand.

    Pair construction intentionally happens during training so the feature
    extractor remains unchanged. ``mean`` averages all MC2-positive answers
    per question; ``mc1`` selects the single answer marked positive by
    ``mc1_targets``. The legacy Cartesian pairs are also accepted when the
    feature file lacks answer-level fields.
    """
    if positive_mode not in {"mean", "mc1", "cartesian"}:
        raise ValueError("positive_mode must be one of: mean, mc1, cartesian")
    data = np.load(path, allow_pickle=False)
    keys = set(data.files)
    if positive_mode in {"mean", "mc1"} and {"activations", "labels", "q_indices"}.issubset(keys):
        activations = np.asarray(data["activations"], np.float32)
        labels = np.asarray(data["labels"]).reshape(-1)
        q_indices = np.asarray(data["q_indices"]).reshape(-1)
        # Generic MC features do not need dataset-specific choice indices;
        # pair construction only requires question grouping and binary labels.
        choice_indices = np.asarray(
            data["choice_indices"] if "choice_indices" in keys else np.arange(len(labels))
        ).reshape(-1)
        if positive_mode == "mc1":
            if "choices" not in keys:
                raise ValueError("positive_mode='mc1' requires a choices field")
            from datasets import load_dataset
            dataset = load_dataset("truthful_qa", "multiple_choice")["validation"]
            mc1_by_question = {}
            for q_idx, item in enumerate(dataset):
                mc1 = item["mc1_targets"]
                mc1_by_question[q_idx] = {str(c): int(y) for c, y in zip(mc1["choices"], mc1["labels"])}
            mc1_labels = np.asarray([mc1_by_question[int(q)].get(str(c), 0) for q, c in zip(q_indices, np.asarray(data["choices"]).reshape(-1))], dtype=np.int64)
        else:
            mc1_labels = None
        pairs = _build_training_pairs(activations, labels, q_indices, choice_indices, positive_mode, mc1_labels)
        meta = {k: data[k] for k in ("train_q_indices", "test_q_indices", "fold_idx", "layer", "model_name") if k in keys}
        meta["positive_mode"] = np.asarray(positive_mode)
        return pairs[0], pairs[1], pairs[2], meta
    # Prefer raw answer-pair representations whenever they are available.
    # The normalized z_neg/z_pos fields are retained for compatibility with
    # older artifacts, but must not be used for the new raw-vector objective.
    if {"z_neg_raw", "z_pos_raw"}.issubset(keys):
        h_neg = np.asarray(data["z_neg_raw"], np.float32)
        h_pos = np.asarray(data["z_pos_raw"], np.float32)
    elif {"z_neg", "z_pos"}.issubset(keys):
        raise KeyError(
            "feature file contains only legacy z_neg/z_pos pair fields, which "
            "may already be L2-normalized. Raw-scale dynSVD training requires "
            "z_neg_raw/z_pos_raw (or answer-level activations/labels/q_indices). "
            "Re-run extract_features.py to regenerate raw pair states."
        )
    else:
        raise KeyError(
            "feature file must contain answer-level raw activations or "
            "z_neg_raw/z_pos_raw"
        )
    if "vector_q_indices" not in keys:
        raise KeyError("feature file must contain vector_q_indices")
    q = np.asarray(data["vector_q_indices"]).reshape(-1)
    if len(q) != len(h_neg):
        raise ValueError("vector_q_indices must align with pairs")
    meta = {k: data[k] for k in ("train_q_indices", "test_q_indices", "fold_idx", "layer", "model_name", "positive_mode") if k in keys}
    return h_neg, h_pos, q, meta


def _build_training_pairs(activations, labels, q_indices, choice_indices, positive_mode, mc1_labels=None):
    """Return (z_neg, z_pos, vector_q_indices) for one selected mode."""
    z_neg, z_pos, vector_q = [], [], []
    for q in np.unique(q_indices):
        rows = np.flatnonzero(q_indices == q)
        neg_rows = rows[labels[rows] == 0]
        pos_rows = rows[labels[rows] == 1]
        if positive_mode == "mc1":
            pos_rows = rows[mc1_labels[rows] == 1]
            if len(pos_rows) != 1:
                raise ValueError(f"Question {q} must have exactly one MC1-positive answer, found {len(pos_rows)}")
        if len(pos_rows) == 0 or len(neg_rows) == 0:
            raise ValueError(f"Question {q} has no valid positive/negative answers")
        if positive_mode == "mean":
            pos_items = [activations[pos_rows].mean(axis=0)]
        else:
            pos_items = [activations[int(r)] for r in pos_rows]
        for pos in pos_items:
            for neg_row in neg_rows:
                z_pos.append(pos)
                z_neg.append(activations[int(neg_row)])
                vector_q.append(int(q))
    if not z_pos:
        raise ValueError("No positive-negative pairs generated")
    return np.stack(z_neg).astype(np.float32), np.stack(z_pos).astype(np.float32), np.asarray(vector_q, np.int64)



def fit_dynamic_svd(h_neg: np.ndarray, h_pos: np.ndarray, q_indices: np.ndarray, train_mask: np.ndarray, rank: int | None = None, energy: float = .99, router_hidden: int = 256, epochs: int = 2000, batch_size: int = 256, learning_rate: float = 1e-3, device: str = "cpu", seed: int = 42, scale_loss_weight: float = 0.03, cosine_loss_weight: float = 0.0, router_mask: np.ndarray | None = None, use_zpos_router: bool = True, use_data_mode: int | None = None, aux_interp_count: int = 8, mean_mask: np.ndarray | None = None, vector_loss_weight: float = 1.0, router_top_k: int = 40, load_balance_loss_weight: float = 0.02) -> tuple[dict, list[dict], dict]:
    set_seed(seed)
    if use_data_mode is None:
        use_data_mode = 1 if use_zpos_router else 0
    if use_data_mode not in (0, 1, 2, 3):
        raise ValueError("use_data_mode must be one of 0, 1, 2, 3")
    if aux_interp_count < 0:
        raise ValueError("aux_interp_count must be nonnegative")
    if router_top_k <= 0:
        raise ValueError("router_top_k must be positive")
    if epochs <= 0 or batch_size <= 0 or learning_rate <= 0 or scale_loss_weight < 0 or cosine_loss_weight < 0 or vector_loss_weight < 0 or load_balance_loss_weight < 0:
        raise ValueError("epochs, batch_size, learning_rate, and loss weights must be valid")
    if vector_loss_weight == 0 and scale_loss_weight == 0 and cosine_loss_weight == 0 and load_balance_loss_weight == 0:
        raise ValueError("at least one loss weight must be positive")
    h_neg_arr = np.asarray(h_neg)
    h_pos_arr = np.asarray(h_pos)
    finite_rows = np.isfinite(h_neg_arr).all(axis=1) & np.isfinite(h_pos_arr).all(axis=1)
    if not finite_rows.all():
        bad = int((~finite_rows).sum())
        raise ValueError(
            f"Feature file contains non-finite values in {bad}/{len(finite_rows)} "
            "pair rows. Re-run feature extraction with numerically stable dtype "
            "(bfloat16/float32); refusing to start SVD."
        )
    dev = torch.device(device if device != "cuda" or torch.cuda.is_available() else "cpu")
    # Preserve the original hidden-state scale for all geometry. In
    # particular, vectors/residual/SVD and their training targets are defined
    # from the raw positive and negative representations.
    neg = torch.as_tensor(h_neg, dtype=torch.float32, device=dev)
    pos = torch.as_tensor(h_pos, dtype=torch.float32, device=dev)
    vectors = pos - neg
    # ``train_mask`` is used only for the geometric quantities (mean and SVD
    # residual basis). The optional router mask is the held-out validation
    # question split used to fit the router and ScaleNet.
    tr = torch.as_tensor(train_mask, dtype=torch.bool, device=dev)
    mean_tr = tr if mean_mask is None else torch.as_tensor(mean_mask, dtype=torch.bool, device=dev)
    router_tr = tr if router_mask is None else torch.as_tensor(router_mask, dtype=torch.bool, device=dev)
    if int(tr.sum()) == 0 or int(mean_tr.sum()) == 0 or int(router_tr.sum()) == 0:
        raise ValueError("geometry, mean, and router masks must all be non-empty")
    v_avg = vectors[mean_tr].mean(0)
    residual = vectors[tr] - v_avg
    # SVD uses the complete selected fold, while v_avg uses only mean_tr
    # (the complementary mean-training subset). Auxiliary samples are excluded from both.
    print(
        f"Computing SVD: residual_shape={tuple(residual.shape)} device={dev}",
        flush=True,
    )
    _, singular, vh = torch.linalg.svd(residual, full_matrices=False)
    print("SVD complete", flush=True)
    r = choose_rank(singular, rank, energy); basis = vh[:r].contiguous()
    print(f"r的大小{r}")
    effective_router_top_k = min(int(router_top_k), r)
    if effective_router_top_k != router_top_k:
        print(
            f"router_top_k={router_top_k} exceeds rank={r}; "
            f"using top_k={effective_router_top_k}",
            flush=True,
        )
    shared_fc1 = nn.Linear(neg.shape[-1], router_hidden).to(dev)
    router = ResidualRouter(
        neg.shape[-1], r, router_hidden, shared_fc1, top_k=effective_router_top_k
    ).to(dev)
    scale_net = ScaleNet(neg.shape[-1], router_hidden, shared_fc1).to(dev)
    unique_params = {id(p): p for p in list(router.parameters()) + list(scale_net.parameters())}
    opt = torch.optim.Adam(list(unique_params.values()), lr=learning_rate); history = []
    # Router fitting uses both sides of every validation pair.  The original
    # z- examples retain their pair-difference target; additional z+ examples
    # are trained with zero residual/scale targets so they do not produce a
    # routed intervention when the current state is already z+.
    fit_neg = neg[router_tr]
    fit_pos = pos[router_tr]
    fit_vec = vectors[router_tr]
    n_router = len(fit_neg)
    sample_inputs = [fit_neg]
    sample_dyn_targets = [fit_vec]
    sample_residual_targets = [fit_vec - v_avg]
    sample_types = [torch.zeros(n_router, dtype=torch.long, device=dev)]  # 0=neg
    if use_data_mode in (1, 3):
        sample_inputs.append(fit_pos)
        sample_dyn_targets.append(torch.zeros_like(fit_pos))
        sample_residual_targets.append(torch.zeros_like(fit_pos))
        sample_types.append(torch.ones(n_router, dtype=torch.long, device=dev))  # 1=pos
    if use_data_mode in (2, 3) and aux_interp_count > 0:
        beta = torch.rand((n_router, aux_interp_count, 1), device=dev)
        aux_inputs = (fit_neg[:, None, :] + beta * (fit_pos - fit_neg)[:, None, :]).reshape(-1, fit_neg.shape[-1])
        aux_dyn_targets = ((1.0 - beta) * fit_vec[:, None, :]).reshape(-1, fit_vec.shape[-1])
        aux_residual_targets = aux_dyn_targets - v_avg
        sample_inputs.append(aux_inputs)
        sample_dyn_targets.append(aux_dyn_targets)
        sample_residual_targets.append(aux_residual_targets)
        sample_types.append(torch.full((aux_inputs.shape[0],), 2, dtype=torch.long, device=dev))  # 2=aux
    # Normalize only the hidden-state copies consumed by Router/ScaleNet.
    # Targets, residual magnitudes, and all geometric quantities above remain
    # on the raw hidden-state scale. This also covers synthetic interpolation
    # samples after they are constructed from raw states.
    raw_router_inputs = torch.cat(sample_inputs, dim=0)
    router_inputs = F.normalize(raw_router_inputs, p=2, dim=-1, eps=1e-8)
    router_dyn_targets = torch.cat(sample_dyn_targets, dim=0)
    router_residual_targets = torch.cat(sample_residual_targets, dim=0)
    router_sample_type = torch.cat(sample_types, dim=0)
    aux_router_samples = int((router_sample_type == 2).sum().item())
    router_is_positive = router_sample_type == 1
    print(
        f"Router/ScaleNet samples: total={len(router_inputs)} "
        f"neg={n_router} pos={int(router_is_positive.sum())} "
        f"aux={aux_router_samples} (per_pair={aux_interp_count})"
    )
    for epoch in range(1, epochs + 1):
        order = torch.randperm(len(router_inputs), device=dev)
        losses = []
        for start in range(0, len(router_inputs), batch_size):
            idx = order[start:start + batch_size]
            router_logits = router.logits(router_inputs[idx])
            coeff = router.route_logits(router_logits)
            scale = scale_net(router_inputs[idx])
            routed_direction = normalize_basis_combination(coeff, basis)
            dyn = v_avg + scale[:, None] * routed_direction
            vector_mse_batch = F.mse_loss(dyn, router_dyn_targets[idx])
            # print(torch.linalg.vector_norm(router_residual_targets[idx], dim=-1))
            scale_mse_batch = F.mse_loss(scale, torch.linalg.vector_norm(router_residual_targets[idx], dim=-1))
            # Preserve cosine loss for z- inputs. z+ has a zero direction
            # target, so it contributes no cosine term.
            is_pos = router_is_positive[idx]
            cosine_terms = []
            if (~is_pos).any():
                cosine_terms.append(1.0 - F.cosine_similarity(
                    dyn[~is_pos], router_dyn_targets[idx][~is_pos], dim=-1
                ))
            cosine_loss_batch = (torch.cat(cosine_terms).mean()
                                 if cosine_terms else torch.zeros((), device=dev))
            load_balance_loss_batch = router_load_balancing_loss(
                router_logits, router.top_k
            )
            loss = (
                vector_loss_weight * vector_mse_batch
                + scale_loss_weight * scale_mse_batch
                + cosine_loss_weight * cosine_loss_batch
                + load_balance_loss_weight * load_balance_loss_batch
            )
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            losses.append(float(loss.detach().cpu()))

        with torch.no_grad():
            router_logits = router.logits(router_inputs)
            coeff = router.route_logits(router_logits)
            scale = scale_net(router_inputs)
            routed_direction = normalize_basis_combination(coeff, basis)
            dyn = v_avg + scale[:, None] * routed_direction
            vector_mse = float(F.mse_loss(dyn, router_dyn_targets).cpu())
            cosine_mask = ~router_is_positive
            cosine_loss = float((1.0 - F.cosine_similarity(dyn[cosine_mask], router_dyn_targets[cosine_mask], dim=-1)).mean().cpu())
            residual_mag = float(
                torch.linalg.vector_norm(
                    scale[:, None] * routed_direction, dim=-1
                ).mean().cpu()
            )
            scale_mse = float(
                F.mse_loss(scale, torch.linalg.vector_norm(router_residual_targets, dim=-1)).cpu()
            )
            scale_mean = float(scale.mean().cpu())
            routed_combination_norm = float(
                torch.linalg.vector_norm(routed_direction, dim=-1).mean().cpu()
            )
            load_balance_loss = float(
                router_load_balancing_loss(router_logits, router.top_k).cpu()
            )
            expert_load_fraction = router_expert_load_fraction(
                router_logits, router.top_k
            ).cpu().tolist()

        rec = {
            "epoch": epoch,
            "loss": float(np.mean(losses)),
            "vector_mse": vector_mse,
            "cosine_loss": cosine_loss,
            "residual_magnitude": residual_mag,
            "scale_mse": scale_mse,
            "scale_mean": scale_mean,
            "routed_combination_norm": routed_combination_norm,
            "load_balance_loss": load_balance_loss,
        }
        history.append(rec)
        print(
            "epoch %03d | loss=%.6e vector_mse=%.6e scale_mse=%.6e cosine_loss=%.6e load_balance=%.6e"
            % (epoch, rec["loss"], vector_mse, scale_mse, cosine_loss, load_balance_loss)
        )

    sv_power = singular.square()
    explained = float(
        sv_power[:r].sum().cpu() / (sv_power.sum().cpu() + 1e-12)
    )
    with torch.no_grad():
        raw_router_input_norms = torch.linalg.vector_norm(raw_router_inputs, dim=-1)
        normalized_router_input_norms = torch.linalg.vector_norm(router_inputs, dim=-1)
        basis_row_norms = torch.linalg.vector_norm(basis, dim=-1)
        target_residual_norms = torch.linalg.vector_norm(router_residual_targets, dim=-1)
        norm_audit = {
            "raw_router_input_norm_mean": float(raw_router_input_norms.mean().cpu()),
            "router_input_after_normalize_norm_mean": float(normalized_router_input_norms.mean().cpu()),
            "v_avg_norm": float(torch.linalg.vector_norm(v_avg).cpu()),
            "svd_residual_norm_mean": float(torch.linalg.vector_norm(residual, dim=-1).mean().cpu()),
            "target_residual_norm_mean": float(target_residual_norms.mean().cpu()),
            "basis_row_norm_mean": float(basis_row_norms.mean().cpu()),
            "basis_row_norm_min": float(basis_row_norms.min().cpu()),
            "basis_row_norm_max": float(basis_row_norms.max().cpu()),
            "routed_combination_norm_mean": routed_combination_norm,
            "scaled_routed_residual_norm_mean": residual_mag,
        }

    metrics = {
        "rank": r,
        "singular_values": singular.detach().cpu().numpy(),
        "explained_energy": explained,
        "svd_reconstruction_mse": float(
            ((residual - (residual @ basis.T) @ basis).square().mean()).cpu()
        ),
        "geometry_train_pairs": int(tr.sum()),
        "router_train_pairs": int(router_tr.sum()),
        "router_optimization_samples": int(len(router_inputs)),
        # Auxiliary samples are used only by the router/scale optimizer.
        "aux_router_samples": aux_router_samples,
        "aux_interp_count": int(aux_interp_count),
        "use_data_mode": int(use_data_mode),
        "train_pairs": int(tr.sum()),
        "scale_loss_weight": scale_loss_weight,
        "cosine_loss_weight": cosine_loss_weight,
        "vector_loss_weight": vector_loss_weight,
        "load_balance_loss_weight": load_balance_loss_weight,
        "router_top_k": int(router.top_k),
        "load_balance_loss": load_balance_loss,
        "expert_load_fraction": expert_load_fraction,
        "normalization_policy": "raw hidden-state geometry is preserved; Router/ScaleNet hidden-state inputs and routed residual direction are L2-normalized during training",
        "norm_audit": norm_audit,
        "use_zpos_router": bool(use_zpos_router),
        "vector_mse": vector_mse,
        "cosine_loss": cosine_loss,
        "scale_mse": scale_mse,
        "scale_mean": scale_mean,
    }
    state = {
        "v_avg": v_avg.detach().cpu(),
        "basis": basis.detach().cpu(),
        "router": router.cpu(),
        "scale_net": scale_net.cpu(),
        "hidden_dim": int(neg.shape[-1]),
        "rank": r,
        "router_hidden": router_hidden,
        "scale_loss_weight": scale_loss_weight,
        "cosine_loss_weight": cosine_loss_weight,
        "vector_loss_weight": vector_loss_weight,
        "load_balance_loss_weight": load_balance_loss_weight,
        "router_top_k": int(router.top_k),
        "train_q_indices": np.unique(q_indices[train_mask]),
        "router_q_indices": np.unique(q_indices[np.asarray(router_mask, dtype=bool)]) if router_mask is not None else np.unique(q_indices[train_mask]),
    }
    return state, history, {
        **metrics,
        "geometry": {
            "train_vector_norm": float(
                torch.linalg.vector_norm(vectors[tr], dim=-1).mean().cpu()
            )
        },
    }
