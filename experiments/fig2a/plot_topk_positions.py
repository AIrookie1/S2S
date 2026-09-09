"""Fig. 2a: Top-K router positions on the held-out fold.

For the project's standard two-fold setup:
    fold0 router -> use only fold1 questions
    fold1 router -> use only fold0 questions

The current dynSVD trainer stores the held-out question IDs directly in the
router checkpoint as ``test_q_indices``.  This script uses that field first.
For compatibility with older checkpoints, it can reconstruct the same
deterministic two-fold split from the base feature file's q_indices.

Question selection:
    Select ``num_questions`` held-out questions while requiring at least
    ``min_unique_peaks`` distinct maximum-router positions.  By default,
    min_unique_peaks=3.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from method.dyn3d_sga.dynSVD import ResidualRouter  # noqa: E402
from method.dyn3d_sga.eval_utils import (  # noqa: E402
    HF_NAMES,
    get_instruction,
    load_truthfulqa_mc_dataframe,
    resolve_device,
    set_seed,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Select held-out questions with at least a specified number of "
            "distinct maximum router indices, then plot their full router "
            "weights and Top-K positions."
        )
    )
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument(
        "--feature_path",
        default=None,
        help=(
            "Base feature NPZ used to train dynSVD. It is only needed as a "
            "fallback when an old checkpoint has no test_q_indices."
        ),
    )
    parser.add_argument("--model_name", default="llama3.1-8B-Instruct")
    parser.add_argument("--model_dir", default=None)
    parser.add_argument(
        "--csv_path", default="./TruthfulQA/data/v1/TruthfulQA.csv"
    )
    parser.add_argument("--layer", type=int, default=None)
    parser.add_argument("--router_fold", type=int, default=None)
    parser.add_argument("--eval_fold", type=int, default=None)
    parser.add_argument(
        "--question_ids",
        nargs="+",
        type=int,
        default=None,
        help=(
            "Optional candidate IDs. Every supplied ID must belong to the "
            "held-out fold."
        ),
    )
    parser.add_argument("--num_questions", type=int, default=5)
    parser.add_argument(
        "--min_unique_peaks",
        type=int,
        default=3,
        help=(
            "Minimum number of distinct maximum-router positions among the "
            "selected questions. Default: 3."
        ),
    )
    parser.add_argument("--top_k", type=int, default=32)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no_instruction", action="store_true")
    parser.add_argument("--device", choices=["cuda", "cpu"], default="cuda")
    parser.add_argument(
        "--dtype",
        choices=["float16", "bfloat16", "float32"],
        default="float16",
    )
    parser.add_argument(
        "--output_prefix",
        default="./experiments/fig2a/figures/fig2a_topk_positions",
    )
    args = parser.parse_args()

    if args.num_questions <= 0:
        parser.error("--num_questions must be positive")
    if args.min_unique_peaks <= 0:
        parser.error("--min_unique_peaks must be positive")
    if args.min_unique_peaks > args.num_questions:
        parser.error(
            "--min_unique_peaks cannot exceed --num_questions"
        )
    if args.top_k <= 0:
        parser.error("--top_k must be positive")
    return args


def setup_style() -> None:
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
            "font.size": 18,
            "axes.labelsize": 18,
            "axes.titlesize": 18,
            "xtick.labelsize": 15,
            "ytick.labelsize": 18,
            "axes.linewidth": 1.5,
            "xtick.direction": "out",
            "ytick.direction": "out",
            "xtick.major.width": 1.5,
            "ytick.major.width": 1.5,
            "pdf.fonttype": 42,
            "svg.fonttype": "none",
            "savefig.bbox": "tight",
        }
    )


def _npz_scalar(data, key: str, default=None):
    if key not in data.files:
        return default
    return np.asarray(data[key]).item()


def infer_fold_from_filename(path: str) -> int | None:
    match = re.search(
        r"(?:^|[_-])fold(\d+)(?:[_\-.]|$)",
        Path(path).name,
    )
    return int(match.group(1)) if match else None


def infer_layer_from_filename(path: str) -> int | None:
    match = re.search(
        r"(?:^|[_-])layer(\d+)(?:[_\-.]|$)",
        Path(path).name,
    )
    return int(match.group(1)) if match else None


def load_router_checkpoint(
    path: str,
    device: torch.device,
) -> dict:
    """Load the dynSVD router without depending on experiments/fig2."""
    with np.load(path, allow_pickle=False) as data:
        required = (
            "hidden_dim",
            "rank",
            "router_hidden",
            "router_fc1.weight",
            "router_fc1.bias",
            "router_fc2.weight",
            "router_fc2.bias",
        )
        missing = [key for key in required if key not in data.files]
        if missing:
            raise KeyError(
                f"Checkpoint {path} is missing router fields: {missing}"
            )

        hidden_dim = int(np.asarray(data["hidden_dim"]).item())
        rank = int(np.asarray(data["rank"]).item())
        router_hidden = int(np.asarray(data["router_hidden"]).item())

        router = ResidualRouter(
            hidden_dim,
            rank,
            router_hidden,
        ).to(device)

        router_state = {
            "fc1.weight": torch.as_tensor(
                data["router_fc1.weight"], dtype=torch.float32
            ),
            "fc1.bias": torch.as_tensor(
                data["router_fc1.bias"], dtype=torch.float32
            ),
            "fc2.weight": torch.as_tensor(
                data["router_fc2.weight"], dtype=torch.float32
            ),
            "fc2.bias": torch.as_tensor(
                data["router_fc2.bias"], dtype=torch.float32
            ),
        }
        router.load_state_dict(router_state)
        router.eval()

        checkpoint_fold = (
            int(np.asarray(data["fold_idx"]).item())
            if "fold_idx" in data.files
            else None
        )
        num_folds = (
            int(np.asarray(data["num_folds"]).item())
            if "num_folds" in data.files
            else 2
        )

    return {
        "router": router,
        "hidden_dim": hidden_dim,
        "rank": rank,
        "router_hidden": router_hidden,
        "fold_idx": checkpoint_fold,
        "num_folds": num_folds,
    }


def resolve_fold_pair(
    checkpoint_path: str,
    checkpoint: dict,
    router_fold_arg: int | None,
    eval_fold_arg: int | None,
) -> tuple[int, int]:
    """Resolve and validate fold0-router -> fold1-eval semantics."""
    router_fold = router_fold_arg
    if router_fold is None:
        router_fold = checkpoint.get("fold_idx")
    if router_fold is None:
        router_fold = infer_fold_from_filename(checkpoint_path)
    if router_fold is None:
        raise ValueError(
            "Could not determine router fold. Pass --router_fold explicitly."
        )

    checkpoint_fold = checkpoint.get("fold_idx")
    if checkpoint_fold is not None and checkpoint_fold != router_fold:
        raise ValueError(
            f"--router_fold={router_fold}, but checkpoint fold_idx="
            f"{checkpoint_fold}."
        )

    num_folds = int(checkpoint.get("num_folds", 2))
    if num_folds != 2:
        raise ValueError(
            f"Fig.2a currently expects the project's two-fold setup, "
            f"but checkpoint num_folds={num_folds}."
        )

    eval_fold = (
        1 - router_fold if eval_fold_arg is None else int(eval_fold_arg)
    )
    if eval_fold != 1 - router_fold:
        raise ValueError(
            f"For two folds, router fold {router_fold} must be evaluated on "
            f"fold {1 - router_fold}, not fold {eval_fold}."
        )

    return int(router_fold), int(eval_fold)


def _validate_question_ids(
    values,
    dataset_size: int,
    source: str,
) -> list[int]:
    ids = [
        int(x)
        for x in np.asarray(values).reshape(-1).tolist()
    ]
    ids = list(dict.fromkeys(ids))

    invalid = [
        qid for qid in ids
        if qid < 0 or qid >= dataset_size
    ]
    if invalid:
        raise ValueError(
            f"{source} contains out-of-range question IDs "
            f"(dataset size={dataset_size}): {invalid[:20]}"
        )
    if not ids:
        raise ValueError(f"{source} contains no question IDs.")
    return ids


def _resolve_feature_path(
    checkpoint_path: str,
    explicit_feature_path: str | None,
) -> Path | None:
    """Find the base feature file used by training, if available."""
    candidates: list[Path] = []

    if explicit_feature_path:
        candidates.append(Path(explicit_feature_path))

    try:
        with np.load(checkpoint_path, allow_pickle=False) as data:
            if "feature_path" in data.files:
                stored = str(np.asarray(data["feature_path"]).item())
                candidates.append(Path(stored))
    except Exception:
        pass

    for candidate in candidates:
        if candidate.is_file():
            return candidate

        rooted = ROOT / candidate
        if rooted.is_file():
            return rooted

    return None


def load_heldout_question_ids(
    checkpoint_path: str,
    feature_path: str | None,
    dataset_size: int,
    router_fold: int,
    eval_fold: int,
) -> tuple[list[int], str]:
    """
    Load the exact held-out fold.

    Current checkpoints produced by train_dynSVD.py contain:
        test_q_indices = held-out fold
    so that field is authoritative.

    For older checkpoints, reconstruct the fold from the base feature file with
    exactly the same deterministic np.array_split logic used by train_dynSVD.py.
    """
    with np.load(checkpoint_path, allow_pickle=False) as data:
        if "fold_idx" in data.files:
            stored_fold = int(np.asarray(data["fold_idx"]).item())
            if stored_fold != router_fold:
                raise ValueError(
                    f"Checkpoint fold_idx={stored_fold}, "
                    f"but router_fold={router_fold}."
                )

        # This is the exact field written by the project's train_dynSVD.py.
        if "test_q_indices" in data.files:
            ids = _validate_question_ids(
                data["test_q_indices"],
                dataset_size,
                f"{checkpoint_path}:test_q_indices",
            )
            return (
                ids,
                f"{checkpoint_path}:test_q_indices "
                f"(held-out fold {eval_fold})",
            )

        # Another exact field in current checkpoints: selected/training fold.
        # In a two-fold setup, its complement is the held-out fold.
        for key in ("outer_train_q_indices", "train_q_indices"):
            if key in data.files:
                train_ids = set(
                    _validate_question_ids(
                        data[key],
                        dataset_size,
                        f"{checkpoint_path}:{key}",
                    )
                )
                ids = [
                    qid for qid in range(dataset_size)
                    if qid not in train_ids
                ]
                if ids:
                    return (
                        ids,
                        f"complement of {checkpoint_path}:{key} "
                        f"(held-out fold {eval_fold})",
                    )

    # Compatibility fallback for an older checkpoint.
    resolved_feature = _resolve_feature_path(
        checkpoint_path,
        feature_path,
    )
    if resolved_feature is None:
        raise RuntimeError(
            "Checkpoint has no test_q_indices and the base feature file "
            "could not be found. Expected the current checkpoint to contain "
            "'test_q_indices', or pass --feature_path "
            "features/<model>_layer<layer>.npz."
        )

    with np.load(resolved_feature, allow_pickle=False) as features:
        if "q_indices" in features.files:
            q_indices = np.asarray(
                features["q_indices"], dtype=np.int64
            ).reshape(-1)
        elif "vector_q_indices" in features.files:
            q_indices = np.asarray(
                features["vector_q_indices"], dtype=np.int64
            ).reshape(-1)
        else:
            raise KeyError(
                f"{resolved_feature} has neither q_indices nor "
                "vector_q_indices, so the training fold cannot be rebuilt."
            )

    # This matches train_dynSVD.py:
    # questions = np.unique(q)
    # folds = np.array_split(questions, num_folds)
    questions = np.unique(q_indices)
    folds = np.array_split(questions, 2)
    ids = _validate_question_ids(
        folds[eval_fold],
        dataset_size,
        f"{resolved_feature}:reconstructed_fold{eval_fold}",
    )
    return (
        ids,
        f"{resolved_feature}: reconstructed fold {eval_fold} "
        "with np.array_split (same as train_dynSVD.py)",
    )


def load_model(
    model_name: str,
    model_dir: str | None,
    device: torch.device,
    dtype_name: str,
):
    """Load the base LM in the same style as the evaluation code."""
    model_path = model_dir or HF_NAMES.get(model_name, model_name)
    print(f"Loading model: {model_path}")

    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        trust_remote_code=True,
    )

    if tokenizer.pad_token_id is None:
        if tokenizer.eos_token_id is None:
            raise ValueError(
                "Tokenizer has neither pad_token_id nor eos_token_id."
            )
        tokenizer.pad_token = tokenizer.eos_token

    if device.type == "cpu":
        dtype = torch.float32
        device_map = None
    else:
        dtype = {
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
            "float32": torch.float32,
        }[dtype_name]
        device_map = "auto"

    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        low_cpu_mem_usage=True,
        torch_dtype=dtype,
        device_map=device_map,
        trust_remote_code=True,
    )
    if device.type == "cpu":
        model = model.to(device)

    model.eval()
    return model, tokenizer


@torch.inference_mode()
def question_hidden_state(
    model,
    tokenizer,
    question: str,
    layer: int,
    router_device: torch.device,
    use_instruction: bool,
    model_name: str,
) -> torch.Tensor:
    """
    Return the normalized final prompt-token state used as Router input.

    This mirrors zero-shot evaluation:
        [instruction]

        Q: {question} A:

    Router training and evaluate_3d_dyn.py both L2-normalize the hidden state
    before feeding it to the router, so Fig.2 uses the same normalization.
    """
    prompt = f"Q: {question} A:"
    if use_instruction:
        prompt = get_instruction(model_name) + "\n\n" + prompt

    input_device = next(
        model.get_input_embeddings().parameters()
    ).device
    inputs = tokenizer(prompt, return_tensors="pt")
    inputs = {
        key: value.to(input_device)
        for key, value in inputs.items()
    }

    outputs = model(
        **inputs,
        output_hidden_states=True,
        use_cache=False,
    )

    hidden_states = outputs.hidden_states
    hidden_index = layer + 1
    if hidden_index < 0 or hidden_index >= len(hidden_states):
        raise ValueError(
            f"Requested layer={layer}, but model returned "
            f"{len(hidden_states) - 1} transformer-layer hidden states."
        )

    hidden = hidden_states[hidden_index][:, -1, :].float()
    hidden = F.normalize(hidden, p=2, dim=-1, eps=1e-8)
    return hidden.to(router_device)


@torch.inference_mode()
def router_weights_for_question(
    checkpoint: dict,
    question: str,
    model,
    tokenizer,
    model_name: str,
    layer: int,
    device: torch.device,
    use_instruction: bool,
) -> np.ndarray:
    hidden = question_hidden_state(
        model=model,
        tokenizer=tokenizer,
        question=question,
        layer=layer,
        router_device=device,
        use_instruction=use_instruction,
        model_name=model_name,
    )

    if hidden.shape[-1] != checkpoint["hidden_dim"]:
        raise ValueError(
            f"Model hidden dim {hidden.shape[-1]} does not match checkpoint "
            f"hidden dim {checkpoint['hidden_dim']}."
        )

    weights = torch.softmax(
        checkpoint["router"].logits(hidden),
        dim=-1,
    )[0]
    return weights.detach().float().cpu().numpy()


@torch.inference_mode()
def find_questions_with_unique_peaks(
    checkpoint: dict,
    candidate_ids: list[int],
    questions: dict[int, str],
    num_questions: int,
    min_unique_peaks: int,
    model,
    tokenizer,
    model_name: str,
    layer: int,
    device: torch.device,
    use_instruction: bool,
) -> tuple[list[int], np.ndarray, list[int], int]:
    """
    Select held-out questions with at least ``min_unique_peaks`` distinct
    maximum-router positions.

    Example:
        num_questions=4
        min_unique_peaks=3

    Then peak positions such as [10, 25, 71, 25] are valid because there are
    three distinct maximum-router positions.
    """
    if min_unique_peaks <= 0:
        raise ValueError("min_unique_peaks must be positive.")
    if min_unique_peaks > num_questions:
        raise ValueError(
            f"min_unique_peaks={min_unique_peaks} cannot exceed "
            f"num_questions={num_questions}."
        )

    # Cache every evaluated candidate so that no question needs a second
    # forward pass through the base language model.
    evaluated: list[tuple[int, np.ndarray, int]] = []

    chosen_ids: list[int] = []
    chosen_weights: list[np.ndarray] = []
    chosen_peaks: list[int] = []
    chosen_id_set: set[int] = set()
    seen_peaks: set[int] = set()

    scanned = 0

    # Stage 1:
    # Search until we have enough distinct peak positions.
    for question_id in candidate_ids:
        scanned += 1

        weights = router_weights_for_question(
            checkpoint=checkpoint,
            question=questions[question_id],
            model=model,
            tokenizer=tokenizer,
            model_name=model_name,
            layer=layer,
            device=device,
            use_instruction=use_instruction,
        )
        peak = int(np.argmax(weights))
        evaluated.append((question_id, weights, peak))

        if peak in seen_peaks:
            continue

        seen_peaks.add(peak)
        chosen_ids.append(question_id)
        chosen_weights.append(weights)
        chosen_peaks.append(peak)
        chosen_id_set.add(question_id)

        print(
            f"[unique {len(seen_peaks)}/{min_unique_peaks}] "
            f"question_id={question_id}, peak_vector={peak + 1}"
        )

        if len(seen_peaks) >= min_unique_peaks:
            break

    if len(seen_peaks) < min_unique_peaks:
        raise RuntimeError(
            f"Only found {len(seen_peaks)} distinct peak indices after "
            f"scanning {scanned}/{len(candidate_ids)} held-out questions. "
            f"Need at least {min_unique_peaks}. Peak vectors found: "
            f"{sorted(x + 1 for x in seen_peaks)}"
        )

    # Stage 2:
    # Fill the remaining slots from already evaluated candidates first.
    # Repeated peak positions are allowed now.
    for question_id, weights, peak in evaluated:
        if len(chosen_ids) >= num_questions:
            break

        if question_id in chosen_id_set:
            continue

        chosen_ids.append(question_id)
        chosen_weights.append(weights)
        chosen_peaks.append(peak)
        chosen_id_set.add(question_id)

        print(
            f"[fill {len(chosen_ids)}/{num_questions}] "
            f"question_id={question_id}, peak_vector={peak + 1}"
        )

    # If needed, continue scanning the remaining held-out candidates.
    if len(chosen_ids) < num_questions:
        for question_id in candidate_ids[scanned:]:
            scanned += 1

            weights = router_weights_for_question(
                checkpoint=checkpoint,
                question=questions[question_id],
                model=model,
                tokenizer=tokenizer,
                model_name=model_name,
                layer=layer,
                device=device,
                use_instruction=use_instruction,
            )
            peak = int(np.argmax(weights))

            chosen_ids.append(question_id)
            chosen_weights.append(weights)
            chosen_peaks.append(peak)
            chosen_id_set.add(question_id)

            print(
                f"[fill {len(chosen_ids)}/{num_questions}] "
                f"question_id={question_id}, peak_vector={peak + 1}"
            )

            if len(chosen_ids) >= num_questions:
                break

    if len(chosen_ids) < num_questions:
        raise RuntimeError(
            f"Only found {len(chosen_ids)} usable held-out questions, "
            f"but num_questions={num_questions}."
        )

    return (
        chosen_ids,
        np.stack(chosen_weights),
        chosen_peaks,
        scanned,
    )


def build_candidate_ids(
    heldout_ids: list[int],
    requested_ids: list[int] | None,
    seed: int,
) -> list[int]:
    """Candidates are always restricted to the held-out fold."""
    heldout_set = set(heldout_ids)

    if requested_ids is not None:
        requested = list(dict.fromkeys(int(x) for x in requested_ids))
        outside = [
            qid for qid in requested
            if qid not in heldout_set
        ]
        if outside:
            raise ValueError(
                "--question_ids contains questions outside the held-out "
                f"fold: {outside}"
            )
        return requested

    candidates = np.asarray(
        heldout_ids,
        dtype=np.int64,
    ).copy()
    rng = np.random.default_rng(seed)
    rng.shuffle(candidates)
    return candidates.tolist()


def topk_indices(
    weights: np.ndarray,
    top_k: int,
) -> np.ndarray:
    """Return 0-based Top-K positions sorted by vector index."""
    k = min(top_k, weights.shape[1])
    selected = np.argpartition(
        weights,
        -k,
        axis=1,
    )[:, -k:]
    return np.sort(selected, axis=1)


def plot(
    weights: np.ndarray,
    selected: np.ndarray,
    eval_fold: int,
    output_prefix: Path,
) -> None:
    """Plot all router weights and overlay Top-K and maximum positions."""
    num_questions, rank = weights.shape
    top_k = selected.shape[1]

    figure, ax = plt.subplots(figsize=(10.45, 5.15))
    image = ax.imshow(
        weights,
        cmap="viridis",
        vmin=0.0,
        vmax=max(float(weights.max()), 1e-8),
        aspect="auto",
        interpolation="nearest",
    )

    for row in range(num_questions):
        ax.scatter(
            selected[row],
            np.full(top_k, row),
            marker="s",
            s=52,
            facecolors="none",
            edgecolors="#FF4D4D",
            linewidths=0.75,
            label=f"Top-{top_k} position" if row == 0 else None,
        )

        peak = int(np.argmax(weights[row]))
        ax.scatter(
            peak,
            row,
            marker="*",
            s=180,
            color="white",
            edgecolors="black",
            linewidths=0.55,
            zorder=4,
            label="Maximum" if row == 0 else None,
        )

    ax.set_xlim(-0.5, rank - 0.5)
    ax.set_ylim(num_questions - 0.5, -0.5)
    ax.set_yticks(np.arange(num_questions))
    ax.set_yticklabels(
        [f"Q{i + 1}" for i in range(num_questions)]
    )

    tick_step = (
        32 if rank >= 128
        else max(1, rank // 8)
    )
    ticks = [
        0,
        *range(tick_step - 1, rank - 1, tick_step),
    ]
    ax.set_xticks(ticks)
    ax.set_xticklabels(
        [str(index + 1) for index in ticks]
    )

    ax.set_xlabel("SVD vector index (full router basis)")
    ax.set_ylabel("Question")
    ax.set_title(
        f"Top-{top_k} router positions on held-out fold {eval_fold}",
        pad=8,
    )

    ax.set_yticks(
        np.arange(-0.5, num_questions, 1),
        minor=True,
    )
    ax.grid(
        which="minor",
        axis="y",
        color="white",
        linewidth=1.25,
    )
    ax.tick_params(which="minor", left=False)
    ax.tick_params(top=False, right=False)

    colorbar = figure.colorbar(
        image,
        ax=ax,
        pad=0.015,
        fraction=0.035,
    )
    colorbar.set_label("Router weight")
    colorbar.ax.tick_params(labelsize=13)

    ax.legend(
        frameon=False,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.34),
        ncol=2,
        fontsize=14,
        handletextpad=0.4,
    )

    figure.subplots_adjust(
        left=0.16,
        right=0.91,
        bottom=0.31,
        top=0.83,
    )

    output_prefix.parent.mkdir(
        parents=True,
        exist_ok=True,
    )
    for suffix in (".png", ".pdf", ".svg"):
        figure.savefig(
            output_prefix.with_suffix(suffix),
            dpi=600,
        )
    plt.close(figure)


def write_data(
    output_prefix: Path,
    checkpoint_path: str,
    feature_path: str | None,
    router_fold: int,
    eval_fold: int,
    fold_source: str,
    heldout_size: int,
    question_ids: list[int],
    weights: np.ndarray,
    selected: np.ndarray,
    scanned: int,
    min_unique_peaks: int,
) -> None:
    selected_sets = [
        set(row.tolist())
        for row in selected
    ]

    csv_path = output_prefix.with_suffix(".csv")
    csv_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with csv_path.open(
        "w",
        encoding="utf-8",
        newline="",
    ) as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "question_id",
                "vector",
                "router_weight",
                "is_top_k",
                "is_peak",
            ]
        )
        for row, question_id in enumerate(question_ids):
            peak = int(np.argmax(weights[row]))
            for vector, weight in enumerate(weights[row]):
                writer.writerow(
                    [
                        question_id,
                        vector + 1,
                        f"{float(weight):.9g}",
                        vector in selected_sets[row],
                        vector == peak,
                    ]
                )

    peak_vectors = [
        int(np.argmax(weights[row]) + 1)
        for row in range(len(question_ids))
    ]

    payload = {
        "experiment": (
            "fig2a_topk_positions_min_distinct_peaks_heldout_fold"
        ),
        "checkpoint": checkpoint_path,
        "feature_path": feature_path,
        "router_fold": router_fold,
        "eval_fold": eval_fold,
        "fold_id_source": fold_source,
        "heldout_fold_size": heldout_size,
        "selected_question_ids": question_ids,
        "rank": int(weights.shape[1]),
        "top_k": int(selected.shape[1]),
        "min_unique_peaks": int(min_unique_peaks),
        "actual_unique_peaks": int(len(set(peak_vectors))),
        "scanned_heldout_candidates": scanned,
        "questions": [
            {
                "plot_row": row + 1,
                "question_id": question_id,
                "peak_vector": int(
                    np.argmax(weights[row]) + 1
                ),
                "top_k_vectors": [
                    int(index + 1)
                    for index in selected[row]
                ],
            }
            for row, question_id in enumerate(question_ids)
        ],
    }

    output_prefix.with_suffix(".json").write_text(
        json.dumps(
            payload,
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )


def main() -> None:
    args = parse_args()
    setup_style()
    set_seed(args.seed)

    device = resolve_device(args.device)
    checkpoint = load_router_checkpoint(
        args.checkpoint,
        device,
    )

    router_fold, eval_fold = resolve_fold_pair(
        checkpoint_path=args.checkpoint,
        checkpoint=checkpoint,
        router_fold_arg=args.router_fold,
        eval_fold_arg=args.eval_fold,
    )

    layer = args.layer
    if layer is None:
        layer = infer_layer_from_filename(args.checkpoint)
    if layer is None:
        raise ValueError(
            "Could not infer layer from checkpoint filename. "
            "Pass --layer explicitly."
        )

    if args.top_k > checkpoint["rank"]:
        raise ValueError(
            f"top_k={args.top_k} exceeds "
            f"checkpoint rank={checkpoint['rank']}."
        )

    if args.min_unique_peaks > checkpoint["rank"]:
        raise ValueError(
            f"Cannot request {args.min_unique_peaks} distinct peak indices "
            f"when router rank={checkpoint['rank']}."
        )

    _, dataset = load_truthfulqa_mc_dataframe(
        args.csv_path
    )
    dataset_size = len(dataset)

    heldout_ids, fold_source = (
        load_heldout_question_ids(
            checkpoint_path=args.checkpoint,
            feature_path=args.feature_path,
            dataset_size=dataset_size,
            router_fold=router_fold,
            eval_fold=eval_fold,
        )
    )

    candidate_ids = build_candidate_ids(
        heldout_ids=heldout_ids,
        requested_ids=args.question_ids,
        seed=args.seed,
    )

    if len(candidate_ids) < args.num_questions:
        raise ValueError(
            f"Held-out fold has only {len(candidate_ids)} candidates, "
            f"but --num_questions={args.num_questions}."
        )

    print(f"Router fold: {router_fold}")
    print(f"Evaluation fold: {eval_fold}")
    print(f"Held-out fold size: {len(heldout_ids)}")
    print(f"Fold membership source: {fold_source}")
    print(
        f"Selection rule: {args.num_questions} questions with at least "
        f"{args.min_unique_peaks} distinct peak vectors"
    )

    questions = {
        question_id: str(
            dataset[question_id]["question"]
        )
        for question_id in candidate_ids
    }

    model, tokenizer = load_model(
        args.model_name,
        args.model_dir,
        device,
        args.dtype,
    )

    (
        question_ids,
        weights,
        peak_indices,
        scanned,
    ) = find_questions_with_unique_peaks(
        checkpoint=checkpoint,
        candidate_ids=candidate_ids,
        questions=questions,
        num_questions=args.num_questions,
        min_unique_peaks=args.min_unique_peaks,
        model=model,
        tokenizer=tokenizer,
        model_name=args.model_name,
        layer=layer,
        device=device,
        use_instruction=not args.no_instruction,
    )

    selected = topk_indices(
        weights,
        args.top_k,
    )
    output_prefix = Path(args.output_prefix)

    write_data(
        output_prefix=output_prefix,
        checkpoint_path=args.checkpoint,
        feature_path=args.feature_path,
        router_fold=router_fold,
        eval_fold=eval_fold,
        fold_source=fold_source,
        heldout_size=len(heldout_ids),
        question_ids=question_ids,
        weights=weights,
        selected=selected,
        scanned=scanned,
        min_unique_peaks=args.min_unique_peaks,
    )

    plot(
        weights=weights,
        selected=selected,
        eval_fold=eval_fold,
        output_prefix=output_prefix,
    )

    unique_peak_count = len(set(peak_indices))

    print(f"Questions: {question_ids}")
    print(
        "Peak vectors (1-based): "
        f"{[index + 1 for index in peak_indices]}"
    )
    print(
        f"Distinct peak vectors: {unique_peak_count} "
        f"(required >= {args.min_unique_peaks})"
    )
    print(
        f"Scanned held-out candidates: "
        f"{scanned}/{len(candidate_ids)}"
    )
    print(
        f"Full router rank: {weights.shape[1]} | "
        f"highlighted Top-K: {args.top_k}"
    )
    print(
        f"Figure: "
        f"{output_prefix.with_suffix('.png')}"
    )


if __name__ == "__main__":
    main()
