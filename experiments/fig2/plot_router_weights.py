"""Plot dynSVD router weights for evaluation questions.

Each dynSVD checkpoint contains an MLP router whose output is a softmax
distribution over the SVD residual vectors (basis[0] ... basis[r-1]). This
script evaluates the router on the question prompt boundary used by
evaluate_dynSVD.py and draws one panel per question/checkpoint.

With five checkpoints and five questions the default layout is a 5 x 5 grid:
columns are datasets/checkpoints and rows are questions. The script does not
run evaluation or training; it only computes router weights and writes
PNG/PDF/SVG plus CSV/JSON tables.
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
        "font.sans-serif": ["Arial", "Helvetica", "Liberation Sans"],
        "font.size": 11,
        "axes.titlesize": 11,
        "axes.labelsize": 11,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "legend.fontsize": 10,
        "figure.titlesize": 13,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.linewidth": 1.2,
        "xtick.direction": "out",
        "ytick.direction": "out",
        "xtick.major.width": 1.2,
        "ytick.major.width": 1.2,
        "pdf.fonttype": 42,
        "svg.fonttype": "none",
        "savefig.bbox": "tight",
        "savefig.dpi": 300,
    }
)

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from matplotlib.ticker import MaxNLocator
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

PALETTE = [
    "#2166AC",
    "#B2182B",
    "#1B7837",
    "#F1A340",
    "#762A83",
    "#4393C3",
    "#D6604D",
    "#5AAE61",
]


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
        help="Question IDs to plot. Defaults to first test IDs from checkpoint 1.",
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
        default=5,
        help="Number of SVD vectors to display (default: 5).",
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
        default="./experiments/fig2/figures/fig2_router_weights",
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
    """Get normalized hidden state at the final Q/A prompt token."""
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
    return F.normalize(hidden, dim=-1).to(device)


def choose_question_ids(
    first_checkpoint: dict[str, Any],
    explicit_ids: list[int] | None,
    num_questions: int,
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
    return candidates[:num_questions]


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
) -> np.ndarray:
    rank_limit = min(num_vectors, max(item["rank"] for item in checkpoints))
    weights = np.full(
        (len(checkpoints), len(question_ids), rank_limit), np.nan, dtype=np.float32
    )
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
                output = checkpoint["router"](hidden)
            n = min(num_vectors, checkpoint["rank"])
            weights[d_idx, q_idx, :n] = (
                output[0, :n].detach().float().cpu().numpy()
            )
    return weights


def write_tables(
    prefix: Path,
    checkpoints: list[dict[str, Any]],
    labels: list[str],
    question_ids: list[int],
    weights: np.ndarray,
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
                                v_idx + 1,
                                f"{float(value):.9g}",
                            ]
                        )
    payload = {
        "experiment": "fig2_router_weights",
        "definition": (
            "softmax(router(normalized hidden state at the final Q/A prompt "
            "token)); values correspond to SVD residual basis rows"
        ),
        "question_ids": question_ids,
        "datasets": [
            {
                "label": labels[d_idx],
                "checkpoint": str(checkpoint["path"]),
                "rank": checkpoint["rank"],
                "layer": checkpoint["layer"],
                "weights": {
                    str(question_id): [
                        None if not np.isfinite(value) else float(value)
                        for value in weights[d_idx, q_idx]
                    ]
                    for q_idx, question_id in enumerate(question_ids)
                },
            }
            for d_idx, checkpoint in enumerate(checkpoints)
        ],
    }
    prefix.with_suffix(".json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8"
    )


def build_figure(
    weights: np.ndarray,
    labels: list[str],
    question_ids: list[int],
    output_prefix: Path,
) -> None:
    """Draw one bar chart per (question, dataset) in a shared-y grid."""
    n_datasets, n_questions, n_vectors_actual = weights.shape
    figure, axes = plt.subplots(
        n_questions,
        n_datasets,
        squeeze=False,
        figsize=(max(3.2 * n_datasets, 4.0), max(2.45 * n_questions, 2.8)),
        sharey=True,
    )
    x = np.arange(1, n_vectors_actual + 1)
    ymax = float(np.nanmax(weights)) if np.isfinite(weights).any() else 1.0
    ymax = min(1.0, max(0.20, ymax * 1.22 + 0.02))
    for q_idx, question_id in enumerate(question_ids):
        for d_idx, label in enumerate(labels):
            ax = axes[q_idx, d_idx]
            values = weights[d_idx, q_idx]
            valid = np.isfinite(values)
            ax.bar(
                x[valid],
                values[valid],
                color=PALETTE[d_idx % len(PALETTE)],
                width=0.72,
                edgecolor="none",
            )
            ax.set_ylim(0, ymax)
            ax.set_xlim(0.4, n_vectors_actual + 0.6)
            ax.set_xticks(x)
            ax.yaxis.set_major_locator(MaxNLocator(nbins=4))
            ax.grid(False)
            ax.tick_params(axis="x", length=0)
            if q_idx == 0:
                ax.set_title(label, fontweight="bold", pad=8)
            if d_idx == 0:
                ax.set_ylabel(f"Q{q_idx + 1}\nrouter weight")
            if q_idx == n_questions - 1:
                ax.set_xlabel("SVD vector")
            ax.text(
                0.02,
                0.95,
                f"id={question_id}",
                transform=ax.transAxes,
                ha="left",
                va="top",
                fontsize=8,
                color="#444444",
            )
    figure.suptitle("dynSVD router weights on evaluation questions", y=1.01)
    figure.subplots_adjust(
        left=0.10,
        right=0.995,
        bottom=0.10,
        top=0.92,
        wspace=0.20,
        hspace=0.34,
    )
    output_prefix.parent.mkdir(parents=True, exist_ok=True)
    for suffix in (".png", ".pdf", ".svg"):
        figure.savefig(output_prefix.with_suffix(suffix), bbox_inches="tight", dpi=300)
    plt.close(figure)


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
        checkpoints[0], args.question_ids, args.num_questions
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
    weights = compute_weights(
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
    write_tables(prefix, checkpoints, labels, question_ids, weights)
    build_figure(weights, labels, question_ids, prefix)
    print(f"Questions: {question_ids}")
    print(f"Datasets: {labels}")
    print(f"Router weights: {prefix.with_suffix('.csv')}")
    print(f"Figure: {prefix.with_suffix('.png')}")


if __name__ == "__main__":
    main()
