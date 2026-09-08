"""Visualize where each question's Top-K router vectors lie in the full basis."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.fig2.plot_router_weights import (  # noqa: E402
    choose_question_ids,
    load_checkpoint,
    load_model,
    question_hidden_state,
)
from method.dyn3d_sga.eval_utils import (  # noqa: E402
    load_truthfulqa_mc_dataframe,
    resolve_device,
    set_seed,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot all router vectors and highlight the Top-K positions."
    )
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--model_name", default="llama3.1-8B-Instruct")
    parser.add_argument("--model_dir", default=None)
    parser.add_argument(
        "--csv_path", default="./TruthfulQA/data/v1/TruthfulQA.csv"
    )
    parser.add_argument("--layer", type=int, default=None)
    parser.add_argument("--question_ids", nargs="+", type=int, default=None)
    parser.add_argument("--num_questions", type=int, default=5)
    parser.add_argument("--top_k", type=int, default=32)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no_instruction", action="store_true")
    parser.add_argument("--device", choices=["cuda", "cpu"], default="cuda")
    parser.add_argument(
        "--dtype", choices=["float16", "bfloat16", "float32"], default="float16"
    )
    parser.add_argument(
        "--output_prefix",
        default="./experiments/fig2a/figures/fig2a_topk_positions",
    )
    args = parser.parse_args()
    if args.num_questions <= 0:
        parser.error("--num_questions must be positive")
    if args.top_k <= 0:
        parser.error("--top_k must be positive")
    return args


@torch.inference_mode()
def compute_router_weights(
    checkpoint,
    question_ids,
    questions,
    model,
    tokenizer,
    model_name,
    layer,
    device,
    use_instruction,
):
    """Return full-rank router weights and sorted Top-K vector indices."""
    all_weights = []
    for question_id in question_ids:
        hidden = question_hidden_state(
            model,
            tokenizer,
            questions[question_id],
            layer,
            device,
            use_instruction,
            model_name,
        )
        if hidden.shape[-1] != checkpoint["hidden_dim"]:
            raise ValueError(
                f"Model hidden dim {hidden.shape[-1]} does not match checkpoint "
                f"hidden dim {checkpoint['hidden_dim']}"
            )
        weights = torch.softmax(checkpoint["router"].logits(hidden), dim=-1)[0]
        all_weights.append(weights.detach().float().cpu().numpy())
    return np.stack(all_weights)


def topk_indices(weights: np.ndarray, top_k: int) -> np.ndarray:
    """Return 0-based Top-K positions, sorted by vector index for plotting."""
    k = min(top_k, weights.shape[1])
    selected = np.argpartition(weights, -k, axis=1)[:, -k:]
    return np.sort(selected, axis=1)


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


def plot(
    weights: np.ndarray,
    selected: np.ndarray,
    question_ids: list[int],
    output_prefix: Path,
) -> None:
    """Show all router weights and overlay every Top-K location."""
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

    # Hollow markers preserve the underlying heatmap while making all Top-K
    # positions visible, including low-valued members of the selected set.
    for row in range(num_questions):
        ax.scatter(
            selected[row],
            np.full(selected.shape[1], row),
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
    ax.set_yticklabels([f"Q{i + 1}" for i in range(num_questions)])
    tick_step = 32 if rank >= 128 else max(1, rank // 8)
    # Avoid a cramped label at the far-right edge next to the colorbar.
    ticks = [0, *range(tick_step - 1, rank - 1, tick_step)]
    ax.set_xticks(ticks)
    ax.set_xticklabels([str(index + 1) for index in ticks])
    ax.set_xlabel("SVD vector index (full router basis)")
    ax.set_ylabel("Question")
    ax.set_title(f"Top-{top_k} router positions across all SVD vectors", pad=8)
    ax.set_yticks(np.arange(-0.5, num_questions, 1), minor=True)
    ax.grid(which="minor", axis="y", color="white", linewidth=1.25)
    ax.tick_params(which="minor", left=False)
    ax.tick_params(top=False, right=False)

    colorbar = figure.colorbar(image, ax=ax, pad=0.015, fraction=0.035)
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
    figure.subplots_adjust(left=0.16, right=0.91, bottom=0.31, top=0.83)

    output_prefix.parent.mkdir(parents=True, exist_ok=True)
    for suffix in (".png", ".pdf", ".svg"):
        figure.savefig(output_prefix.with_suffix(suffix), dpi=600)
    plt.close(figure)


def write_data(
    output_prefix: Path,
    checkpoint_path: str,
    question_ids: list[int],
    weights: np.ndarray,
    selected: np.ndarray,
) -> None:
    """Save every router value and an explicit Top-K membership flag."""
    selected_sets = [set(row.tolist()) for row in selected]
    csv_path = output_prefix.with_suffix(".csv")
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["question_id", "vector", "router_weight", "is_top_k"])
        for row, question_id in enumerate(question_ids):
            for vector, weight in enumerate(weights[row]):
                writer.writerow(
                    [question_id, vector + 1, f"{float(weight):.9g}", vector in selected_sets[row]]
                )

    payload = {
        "experiment": "fig2a_topk_positions_in_full_router",
        "checkpoint": checkpoint_path,
        "question_ids": question_ids,
        "rank": int(weights.shape[1]),
        "top_k": int(selected.shape[1]),
        "questions": [
            {
                "question_id": question_id,
                "peak_vector": int(np.argmax(weights[row]) + 1),
                "top_k_vectors": [int(index + 1) for index in selected[row]],
            }
            for row, question_id in enumerate(question_ids)
        ],
    }
    output_prefix.with_suffix(".json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def main() -> None:
    args = parse_args()
    setup_style()
    set_seed(args.seed)
    device = resolve_device(args.device)
    checkpoint = load_checkpoint(args.checkpoint, device)
    layer = checkpoint["layer"] if args.layer is None else args.layer
    if args.top_k > checkpoint["rank"]:
        raise ValueError(
            f"top_k={args.top_k} exceeds checkpoint rank={checkpoint['rank']}"
        )

    question_ids = choose_question_ids(
        checkpoint, args.question_ids, args.num_questions, args.seed
    )
    _, dataset = load_truthfulqa_mc_dataframe(args.csv_path)
    questions = {
        question_id: str(dataset[question_id]["question"])
        for question_id in question_ids
    }
    model, tokenizer = load_model(
        args.model_name, args.model_dir, device, args.dtype
    )
    weights = compute_router_weights(
        checkpoint,
        question_ids,
        questions,
        model,
        tokenizer,
        args.model_name,
        layer,
        device,
        not args.no_instruction,
    )
    selected = topk_indices(weights, args.top_k)
    output_prefix = Path(args.output_prefix)
    write_data(output_prefix, args.checkpoint, question_ids, weights, selected)
    plot(weights, selected, question_ids, output_prefix)
    print(f"Questions: {question_ids}")
    print(f"Full router rank: {weights.shape[1]} | highlighted Top-K: {args.top_k}")
    print(f"Figure: {output_prefix.with_suffix('.png')}")


if __name__ == "__main__":
    main()
