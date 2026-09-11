"""Plot dynSVD router weights for evaluation questions.

Each dynSVD checkpoint contains an MLP router whose output is a softmax
distribution over the SVD residual vectors (basis[0] ... basis[r-1]). This
script evaluates the router at the same question-prompt boundary used by
evaluate_3d_dyn.py. For each checkpoint it draws a 5-question by 32-vector
heatmap; color intensity encodes the router softmax value. The script does not
run evaluation or training and also writes the underlying CSV/JSON tables.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any

import matplotlib as mpl

mpl.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
        "font.size": 18,
        "axes.titlesize": 18,
        "axes.labelsize": 18,
        "xtick.labelsize": 14,
        "ytick.labelsize": 18,
        "legend.fontsize": 18,
        "figure.titlesize": 18,
        "axes.linewidth": 1.5,
        "xtick.direction": "out",
        "ytick.direction": "out",
        "xtick.major.width": 1.5,
        "ytick.major.width": 1.5,
        "pdf.fonttype": 42,
        "svg.fonttype": "none",
        "savefig.bbox": "tight",
        "savefig.dpi": 600,
    }
)

import matplotlib.pyplot as plt
import numpy as np
import torch
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
        description="Plot dynSVD router softmax weights for evaluation questions."
    )
    parser.add_argument(
        "--checkpoints",
        "--checkpoint",
        dest="checkpoints",
        nargs="+",
        required=True,
        help="One or more dynSVD .npz checkpoints; each is treated as a dataset.",
    )
    parser.add_argument(
        "--dataset_labels",
        nargs="+",
        default=None,
        help="Optional labels, one per checkpoint. Defaults to checkpoint stems.",
    )
    parser.add_argument("--model_name", default="llama3.1-8B-Instruct")
    parser.add_argument(
        "--model_dir",
        default=None,
        help="Local model directory. If omitted, HF_NAMES/model_name is used.",
    )
    parser.add_argument(
        "--csv_path",
        default="./TruthfulQA/data/v1/TruthfulQA.csv",
        help="TruthfulQA CSV used to align question IDs with the HF dataset.",
    )
    parser.add_argument("--layer", type=int, default=None)
    parser.add_argument(
        "--question_ids",
        nargs="+",
        type=int,
        default=None,
        help="Question IDs to plot. If omitted, randomly sample test IDs using --seed.",
    )
    parser.add_argument(
        "--num_questions",
        type=int,
        default=5,
        help="Number of questions when --question_ids is omitted (default: 5).",
    )
    parser.add_argument(
        "--num_vectors",
        type=int,
        default=32,
        help="Number of leading SVD vectors to display (default: 32).",
    )
    parser.add_argument(
        "--no_instruction",
        action="store_true",
        help="Use Q: ... A: without the evaluator instruction prefix.",
    )
    parser.add_argument("--device", choices=["cuda", "cpu"], default="cuda")
    parser.add_argument(
        "--dtype", choices=["float16", "bfloat16", "float32"], default="float16"
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--output_prefix",
        default="./results/experiment/fig2/fig2_router_weights",
        help="Output path prefix; extensions are added automatically.",
    )
    return parser.parse_args()


def _scalar(data: np.lib.npyio.NpzFile, key: str, default: Any = None) -> Any:
    if key not in data.files:
        return default
    value = data[key]
    return value.item() if np.asarray(value).ndim == 0 else value


def load_checkpoint(path: str | Path, device: torch.device) -> dict[str, Any]:
    """Load one NPZ checkpoint and reconstruct its ResidualRouter."""
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Checkpoint does not exist: {path}")
    with np.load(path, allow_pickle=False) as data:
        required = {
            "basis",
            "hidden_dim",
            "rank",
            "router_hidden",
            "router_fc1.weight",
            "router_fc1.bias",
            "router_fc2.weight",
            "router_fc2.bias",
        }
        missing = sorted(required.difference(data.files))
        if missing:
            raise KeyError(f"{path} is missing checkpoint fields: {missing}")
        basis = np.asarray(data["basis"], dtype=np.float32)
        rank = int(_scalar(data, "rank"))
        hidden_dim = int(_scalar(data, "hidden_dim"))
        router_hidden = int(_scalar(data, "router_hidden"))
        if basis.ndim != 2 or basis.shape != (rank, hidden_dim):
            raise ValueError(
                f"{path}: basis shape {basis.shape} does not match "
                f"(rank, hidden_dim)=({rank}, {hidden_dim})"
            )
        router = ResidualRouter(hidden_dim, rank, router_hidden).to(device)
        state = {
            "fc1.weight": torch.as_tensor(data["router_fc1.weight"], dtype=torch.float32),
            "fc1.bias": torch.as_tensor(data["router_fc1.bias"], dtype=torch.float32),
            "fc2.weight": torch.as_tensor(data["router_fc2.weight"], dtype=torch.float32),
            "fc2.bias": torch.as_tensor(data["router_fc2.bias"], dtype=torch.float32),
        }
        router.load_state_dict(state)
        router.eval()
        return {
            "path": path,
            "router": router,
            "basis": basis,
            "rank": rank,
            "hidden_dim": hidden_dim,
            "layer": int(_scalar(data, "layer", 14)),
            "test_q_indices": (
                np.asarray(data["test_q_indices"], dtype=np.int64).reshape(-1)
                if "test_q_indices" in data.files
                else np.asarray([], dtype=np.int64)
            ),
        }


def load_model(
    model_name: str,
    model_dir: str | None,
    device: torch.device,
    dtype_name: str,
) -> tuple[Any, Any]:
    model_path = model_dir or HF_NAMES.get(model_name, model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        if tokenizer.eos_token_id is None:
            raise ValueError("Tokenizer has neither pad_token_id nor eos_token_id.")
        tokenizer.pad_token = tokenizer.eos_token
    dtype = torch.float32 if device.type == "cpu" else {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }[dtype_name]
    kwargs: dict[str, Any] = {
        "low_cpu_mem_usage": True,
        "torch_dtype": dtype,
        "trust_remote_code": True,
    }
    if device.type == "cuda":
        kwargs["device_map"] = "auto"
    model = AutoModelForCausalLM.from_pretrained(model_path, **kwargs)
    if device.type == "cpu":
        model = model.to(device)
    model.eval()
    return model, tokenizer


@torch.inference_mode()
def question_hidden_state(
    model: Any,
    tokenizer: Any,
    question: str,
    layer: int,
    device: torch.device,
    use_instruction: bool,
    model_name: str,
) -> torch.Tensor:
    """Get the raw hidden state at the final Q/A prompt token."""
    base = f"Q: {question} A:"
    if use_instruction:
        base = get_instruction(model_name) + "\n\n" + base
    encoded = tokenizer(base, return_tensors="pt")
    input_device = next(model.get_input_embeddings().parameters()).device
    encoded = {key: value.to(input_device) for key, value in encoded.items()}
    outputs = model(**encoded, output_hidden_states=True, use_cache=False)
    index = layer + 1  # hidden_states[0] is the embedding output.
    if index >= len(outputs.hidden_states):
        raise IndexError(
            f"Requested layer {layer}, but model returned "
            f"{len(outputs.hidden_states) - 1} layers."
        )
    hidden = outputs.hidden_states[index][:, -1, :].float()
    if hidden.shape[-1] == 0 or not torch.isfinite(hidden).all():
        raise ValueError(f"Non-finite/empty hidden state for question: {question!r}")
    return hidden.to(device)


def choose_question_ids(
    first_checkpoint: dict[str, Any],
    explicit_ids: list[int] | None,
    num_questions: int,
    seed: int,
) -> list[int]:
    if explicit_ids is not None:
        if not explicit_ids:
            raise ValueError("--question_ids cannot be empty.")
        if len(set(explicit_ids)) != len(explicit_ids):
            raise ValueError("--question_ids contains duplicates.")
        return explicit_ids
    if num_questions <= 0:
        raise ValueError("--num_questions must be positive.")
    candidates = sorted(set(int(x) for x in first_checkpoint["test_q_indices"]))
    if not candidates:
        raise ValueError(
            "Checkpoint has no test_q_indices. Pass --question_ids explicitly."
        )
    if len(candidates) < num_questions:
        raise ValueError(
            f"Checkpoint contains only {len(candidates)} test questions, "
            f"but --num_questions={num_questions}."
        )
    rng = np.random.default_rng(seed)
    selected = rng.choice(candidates, size=num_questions, replace=False)
    return [int(question_id) for question_id in selected]


def _centered_window(peak_index: int, rank: int, width: int) -> tuple[int, int]:
    """Return a fixed-width [start, end) window centered near the peak."""
    width = min(width, rank)
    left = (width - 1) // 2
    start = max(0, min(peak_index - left, rank - width))
    return start, start + width


def compute_weights(
    checkpoints: list[dict[str, Any]],
    question_ids: list[int],
    questions: dict[int, str],
    model: Any,
    tokenizer: Any,
    model_name: str,
    layer: int,
    device: torch.device,
    use_instruction: bool,
    num_vectors: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Select a contiguous window around each question's global router peak."""
    width = min(num_vectors, min(item["rank"] for item in checkpoints))
    weights = np.empty(
        (len(checkpoints), len(question_ids), width), dtype=np.float32
    )
    vector_indices = np.empty_like(weights, dtype=np.int64)
    peak_indices = np.empty((len(checkpoints), len(question_ids)), dtype=np.int64)

    for q_idx, question_id in enumerate(question_ids):
        hidden = question_hidden_state(
            model,
            tokenizer,
            questions[question_id],
            layer,
            device,
            use_instruction,
            model_name,
        )
        for d_idx, checkpoint in enumerate(checkpoints):
            if hidden.shape[-1] != checkpoint["hidden_dim"]:
                raise ValueError(
                    f"{checkpoint['path']}: model hidden dim {hidden.shape[-1]} "
                    f"!= checkpoint hidden dim {checkpoint['hidden_dim']}"
                )
            with torch.inference_mode():
                full = torch.softmax(checkpoint["router"].logits(hidden), dim=-1)[0]
            peak = int(torch.argmax(full).item())
            window_start, window_end = _centered_window(
                peak, checkpoint["rank"], width
            )
            weights[d_idx, q_idx] = (
                full[window_start:window_end].detach().float().cpu().numpy()
            )
            vector_indices[d_idx, q_idx] = np.arange(
                window_start + 1, window_end + 1, dtype=np.int64
            )
            peak_indices[d_idx, q_idx] = peak + 1
    return weights, vector_indices, peak_indices


def write_tables(
    prefix: Path,
    checkpoints: list[dict[str, Any]],
    labels: list[str],
    question_ids: list[int],
    weights: np.ndarray,
    vector_indices: np.ndarray,
    peak_indices: np.ndarray,
) -> None:
    """Write long-form CSV and a grouped JSON table."""
    csv_path = prefix.with_suffix(".csv")
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["dataset", "checkpoint", "question_id", "vector", "weight"])
        for d_idx, checkpoint in enumerate(checkpoints):
            for q_idx, question_id in enumerate(question_ids):
                for v_idx, value in enumerate(weights[d_idx, q_idx]):
                    if np.isfinite(value):
                        writer.writerow(
                            [
                                labels[d_idx],
                                str(checkpoint["path"]),
                                question_id,
                                int(vector_indices[d_idx, q_idx, v_idx]),
                                f"{float(value):.9g}",
                            ]
                        )
    payload = {
        "experiment": "fig2_router_weights",
        "definition": (
            "softmax(router.logits(raw hidden state at the final Q/A prompt "
            "token)); values correspond to SVD residual basis rows"
        ),
        "question_ids": question_ids,
        "datasets": [
            {
                "label": labels[d_idx],
                "checkpoint": str(checkpoint["path"]),
                "rank": checkpoint["rank"],
                "layer": checkpoint["layer"],
                "questions": {
                    str(question_id): {
                        "window_start": int(vector_indices[d_idx, q_idx, 0]),
                        "window_end": int(vector_indices[d_idx, q_idx, -1]),
                        "peak_vector": int(peak_indices[d_idx, q_idx]),
                        "vectors": [
                            {
                                "index": int(vector_index),
                                "weight": float(value),
                            }
                            for vector_index, value in zip(
                                vector_indices[d_idx, q_idx], weights[d_idx, q_idx]
                            )
                        ],
                    }
                    for q_idx, question_id in enumerate(question_ids)
                },
            }
            for d_idx, checkpoint in enumerate(checkpoints)
        ],
    }
    prefix.with_suffix(".json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8"
    )


def _safe_label(label: str) -> str:
    return "".join(ch if ch.isalnum() or ch in "-_" else "_" for ch in label).strip("_")


def build_figure(
    weights: np.ndarray,
    labels: list[str],
    question_ids: list[int],
    vector_indices: np.ndarray,
    peak_indices: np.ndarray,
    output_prefix: Path,
) -> list[Path]:
    """Draw one 5 x N router-weight heatmap for each checkpoint."""
    n_datasets, n_questions, n_vectors = weights.shape
    finite = weights[np.isfinite(weights)]
    color_max = float(finite.max()) if finite.size else 1.0
    color_max = max(color_max, 1e-8)
    output_prefix.parent.mkdir(parents=True, exist_ok=True)
    outputs = []

    for dataset_idx, label in enumerate(labels):
        values = weights[dataset_idx]
        figure, ax = plt.subplots(figsize=(10.45, 3.15))
        image = ax.imshow(
            values,
            cmap="viridis",
            vmin=0.0,
            vmax=color_max,
            aspect="auto",
            interpolation="nearest",
        )

        # Five question rows, with thin separators matching the reference style.
        ax.set_yticks(np.arange(n_questions))
        row_labels = [
            f"Q{i + 1}  V{vector_indices[dataset_idx, i, 0]}–"
            f"V{vector_indices[dataset_idx, i, -1]}  "
            f"(peak V{peak_indices[dataset_idx, i]})"
            for i in range(n_questions)
        ]
        ax.set_yticklabels(row_labels)
        tick_positions = sorted(set([0, *range(3, n_vectors, 4), n_vectors - 1]))
        ax.set_xticks(tick_positions)
        ax.set_xticklabels([str(i + 1) for i in tick_positions])
        ax.set_xlabel(f"Position within each {n_vectors}-vector window")
        ax.set_ylabel("Question")
        ax.set_title("Router weights for five evaluation questions", pad=8)
        ax.set_yticks(np.arange(-0.5, n_questions, 1), minor=True)
        ax.grid(which="minor", axis="y", color="white", linewidth=1.5)
        ax.tick_params(which="minor", left=False)
        ax.tick_params(top=False, right=False)

        colorbar = figure.colorbar(image, ax=ax, pad=0.015, fraction=0.035)
        colorbar.set_label("Router weight")
        colorbar.ax.tick_params(labelsize=13)
        figure.subplots_adjust(left=0.25, right=0.91, bottom=0.24, top=0.84)

        prefix = (
            output_prefix
            if n_datasets == 1
            else output_prefix.with_name(f"{output_prefix.name}_{_safe_label(label)}")
        )
        for suffix in (".png", ".pdf", ".svg"):
            figure.savefig(prefix.with_suffix(suffix), dpi=600, bbox_inches="tight")
        outputs.append(prefix.with_suffix(".png"))
        plt.close(figure)
    return outputs


def main() -> None:
    args = parse_args()
    if args.num_vectors <= 0:
        raise ValueError("--num_vectors must be positive.")
    if args.layer is not None and args.layer < 0:
        raise ValueError("--layer must be non-negative.")
    if args.dataset_labels is not None and len(args.dataset_labels) != len(args.checkpoints):
        raise ValueError("--dataset_labels must have one value per checkpoint.")
    labels = args.dataset_labels or [Path(path).stem for path in args.checkpoints]
    set_seed(args.seed)
    device = resolve_device(args.device)
    checkpoints = [load_checkpoint(path, device) for path in args.checkpoints]
    layer_values = {item["layer"] for item in checkpoints}
    layer = args.layer if args.layer is not None else min(layer_values)
    if args.layer is None and len(layer_values) > 1:
        print(
            f"WARNING: checkpoints use layers {sorted(layer_values)}; "
            f"using layer={layer} for model inputs. Pass --layer to override."
        )
    question_ids = choose_question_ids(
        checkpoints[0], args.question_ids, args.num_questions, args.seed
    )
    _, hf_dataset = load_truthfulqa_mc_dataframe(args.csv_path)
    questions = {}
    for question_id in question_ids:
        if question_id < 0 or question_id >= len(hf_dataset):
            raise IndexError(
                f"Question ID {question_id} is outside dataset [0, {len(hf_dataset)-1}]."
            )
        questions[question_id] = str(hf_dataset[question_id]["question"])
    model, tokenizer = load_model(args.model_name, args.model_dir, device, args.dtype)
    weights, vector_indices, peak_indices = compute_weights(
        checkpoints,
        question_ids,
        questions,
        model,
        tokenizer,
        args.model_name,
        layer,
        device,
        not args.no_instruction,
        args.num_vectors,
    )
    prefix = Path(args.output_prefix)
    write_tables(
        prefix, checkpoints, labels, question_ids,
        weights, vector_indices, peak_indices,
    )
    figure_paths = build_figure(
        weights, labels, question_ids, vector_indices, peak_indices, prefix
    )
    print(f"Questions: {question_ids}")
    print(f"Datasets: {labels}")
    print(f"Router weights: {prefix.with_suffix('.csv')}")
    print("Figures: " + ", ".join(str(path) for path in figure_paths))


if __name__ == "__main__":
    main()
