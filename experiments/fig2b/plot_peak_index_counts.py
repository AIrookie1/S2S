"""Fig. 2b: Most frequent Top-K router indices on the held-out fold.

For the project's standard two-fold setup:
    fold0 router -> evaluate only fold1 questions
    fold1 router -> evaluate only fold0 questions

The procedure has two Top-K stages:

1. Per-question Top-K:
   For every held-out question, select the K router indices with the
   largest router weights.

2. Global Top-K:
   Count how many held-out questions contain each router index in their
   per-question Top-K selection. Then retain only the globally most
   frequent router indices for visualization.

The plotted y-value is NOT the raw count. Instead, for router index i:

    frequency(i)
        = number of held-out questions where i appears in Top-K
          -----------------------------------------------------
                    total number of held-out questions

Therefore the y-axis represents the fraction / percentage of held-out
questions for which each router index is selected.

The current dynSVD checkpoint stores held-out question IDs as
``test_q_indices``. This field is used directly. For compatibility with an
older checkpoint, the same deterministic fold can be reconstructed from the
base feature file's q_indices using the train_dynSVD.py np.array_split rule.
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
from matplotlib.ticker import PercentFormatter
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
            "On the held-out fold, select the Top-K router indices "
            "for every question, count their frequencies, and display "
            "only the globally most frequent router indices."
        )
    )

    parser.add_argument(
        "--checkpoint",
        required=True,
    )

    parser.add_argument(
        "--feature_path",
        default=None,
        help=(
            "Base feature NPZ used to train dynSVD. Only needed as a "
            "fallback when an old checkpoint has no test_q_indices."
        ),
    )

    parser.add_argument(
        "--model_name",
        default="llama3.1-8B-Instruct",
    )

    parser.add_argument(
        "--model_dir",
        default=None,
    )

    parser.add_argument(
        "--csv_path",
        default="./TruthfulQA/data/v1/TruthfulQA.csv",
    )

    parser.add_argument(
        "--layer",
        type=int,
        default=None,
    )

    parser.add_argument(
        "--router_fold",
        type=int,
        default=None,
    )

    parser.add_argument(
        "--eval_fold",
        type=int,
        default=None,
    )

    # ---------------------------------------------------------
    # Stage 1:
    # For every individual question, select Top-K router indices.
    # ---------------------------------------------------------
    parser.add_argument(
        "--topk",
        type=int,
        default=30,
        help=(
            "Number of highest-weight router indices selected "
            "for each held-out question."
        ),
    )

    # ---------------------------------------------------------
    # Stage 2:
    # After accumulating counts over the entire held-out fold,
    # display only the globally most frequent indices.
    # ---------------------------------------------------------
    parser.add_argument(
        "--display_topk",
        type=int,
        default=30,
        help=(
            "Number of globally most frequent router indices "
            "to display after counting all held-out questions."
        ),
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
    )

    parser.add_argument(
        "--no_instruction",
        action="store_true",
    )

    parser.add_argument(
        "--device",
        choices=["cuda", "cpu"],
        default="cuda",
    )

    parser.add_argument(
        "--dtype",
        choices=[
            "float16",
            "bfloat16",
            "float32",
        ],
        default="float16",
    )

    parser.add_argument(
        "--output_prefix",
        default="./experiments/fig2b/figures/fig2b_topk",
    )

    return parser.parse_args()


def setup_style() -> None:
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": [
                "Arial",
                "Helvetica",
                "DejaVu Sans",
            ],
            "font.size": 18,
            "axes.labelsize": 18,
            "axes.titlesize": 18,
            "xtick.labelsize": 14,
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


def infer_fold_from_filename(
    path: str,
) -> int | None:
    match = re.search(
        r"(?:^|[_-])fold(\d+)(?:[_\-.]|$)",
        Path(path).name,
    )
    return (
        int(match.group(1))
        if match
        else None
    )


def infer_layer_from_filename(
    path: str,
) -> int | None:
    match = re.search(
        r"(?:^|[_-])layer(\d+)(?:[_\-.]|$)",
        Path(path).name,
    )
    return (
        int(match.group(1))
        if match
        else None
    )


def load_router_checkpoint(
    path: str,
    device: torch.device,
) -> dict:
    """Load the dynSVD router."""

    with np.load(
        path,
        allow_pickle=False,
    ) as data:

        required = (
            "hidden_dim",
            "rank",
            "router_hidden",
            "router_fc1.weight",
            "router_fc1.bias",
            "router_fc2.weight",
            "router_fc2.bias",
        )

        missing = [
            key
            for key in required
            if key not in data.files
        ]

        if missing:
            raise KeyError(
                f"Checkpoint {path} is missing "
                f"router fields: {missing}"
            )

        hidden_dim = int(
            np.asarray(
                data["hidden_dim"]
            ).item()
        )

        rank = int(
            np.asarray(
                data["rank"]
            ).item()
        )

        router_hidden = int(
            np.asarray(
                data["router_hidden"]
            ).item()
        )

        router = ResidualRouter(
            hidden_dim,
            rank,
            router_hidden,
        ).to(device)

        router.load_state_dict(
            {
                "fc1.weight": torch.as_tensor(
                    data["router_fc1.weight"],
                    dtype=torch.float32,
                ),
                "fc1.bias": torch.as_tensor(
                    data["router_fc1.bias"],
                    dtype=torch.float32,
                ),
                "fc2.weight": torch.as_tensor(
                    data["router_fc2.weight"],
                    dtype=torch.float32,
                ),
                "fc2.bias": torch.as_tensor(
                    data["router_fc2.bias"],
                    dtype=torch.float32,
                ),
            }
        )

        router.eval()

        checkpoint_fold = (
            int(
                np.asarray(
                    data["fold_idx"]
                ).item()
            )
            if "fold_idx" in data.files
            else None
        )

        num_folds = (
            int(
                np.asarray(
                    data["num_folds"]
                ).item()
            )
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

    router_fold = router_fold_arg

    if router_fold is None:
        router_fold = checkpoint.get(
            "fold_idx"
        )

    if router_fold is None:
        router_fold = infer_fold_from_filename(
            checkpoint_path
        )

    if router_fold is None:
        raise ValueError(
            "Could not determine router fold. "
            "Pass --router_fold explicitly."
        )

    checkpoint_fold = checkpoint.get(
        "fold_idx"
    )

    if (
        checkpoint_fold is not None
        and checkpoint_fold != router_fold
    ):
        raise ValueError(
            f"--router_fold={router_fold}, "
            f"but checkpoint fold_idx="
            f"{checkpoint_fold}."
        )

    num_folds = int(
        checkpoint.get(
            "num_folds",
            2,
        )
    )

    if num_folds != 2:
        raise ValueError(
            "Fig.2b expects the project's "
            "two-fold setup, but checkpoint "
            f"num_folds={num_folds}."
        )

    eval_fold = (
        1 - router_fold
        if eval_fold_arg is None
        else int(eval_fold_arg)
    )

    if eval_fold != 1 - router_fold:
        raise ValueError(
            f"For two folds, router fold "
            f"{router_fold} must be evaluated "
            f"on fold {1 - router_fold}, "
            f"not fold {eval_fold}."
        )

    return (
        int(router_fold),
        int(eval_fold),
    )


def _validate_question_ids(
    values,
    dataset_size: int,
    source: str,
) -> list[int]:

    ids = [
        int(x)
        for x in np.asarray(
            values
        ).reshape(-1).tolist()
    ]

    # Remove duplicates while preserving order.
    ids = list(
        dict.fromkeys(ids)
    )

    invalid = [
        qid
        for qid in ids
        if qid < 0
        or qid >= dataset_size
    ]

    if invalid:
        raise ValueError(
            f"{source} contains out-of-range "
            f"question IDs "
            f"(dataset size={dataset_size}): "
            f"{invalid[:20]}"
        )

    if not ids:
        raise ValueError(
            f"{source} contains no question IDs."
        )

    return ids


def _resolve_feature_path(
    checkpoint_path: str,
    explicit_feature_path: str | None,
) -> Path | None:

    candidates: list[Path] = []

    if explicit_feature_path:
        candidates.append(
            Path(explicit_feature_path)
        )

    try:
        with np.load(
            checkpoint_path,
            allow_pickle=False,
        ) as data:

            if "feature_path" in data.files:
                stored = str(
                    np.asarray(
                        data["feature_path"]
                    ).item()
                )
                candidates.append(
                    Path(stored)
                )

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
    Return only the held-out fold.

    train_dynSVD.py writes:

        test_q_indices = test_q = heldout_q

    Therefore, for a fold0 router, test_q_indices are exactly
    the fold1 questions used for evaluation.
    """

    with np.load(
        checkpoint_path,
        allow_pickle=False,
    ) as data:

        if "fold_idx" in data.files:
            stored_fold = int(
                np.asarray(
                    data["fold_idx"]
                ).item()
            )

            if stored_fold != router_fold:
                raise ValueError(
                    f"Checkpoint fold_idx="
                    f"{stored_fold}, but "
                    f"router_fold={router_fold}."
                )

        # -----------------------------------------------------
        # Preferred/current checkpoint format.
        # -----------------------------------------------------
        if "test_q_indices" in data.files:

            ids = _validate_question_ids(
                data["test_q_indices"],
                dataset_size,
                (
                    f"{checkpoint_path}:"
                    "test_q_indices"
                ),
            )

            return (
                ids,
                (
                    f"{checkpoint_path}:"
                    f"test_q_indices "
                    f"(held-out fold {eval_fold})"
                ),
            )

        # -----------------------------------------------------
        # Fallback if only training IDs are available.
        # -----------------------------------------------------
        for key in (
            "outer_train_q_indices",
            "train_q_indices",
        ):

            if key in data.files:

                train_ids = set(
                    _validate_question_ids(
                        data[key],
                        dataset_size,
                        (
                            f"{checkpoint_path}:"
                            f"{key}"
                        ),
                    )
                )

                ids = [
                    qid
                    for qid in range(
                        dataset_size
                    )
                    if qid not in train_ids
                ]

                if ids:
                    return (
                        ids,
                        (
                            f"complement of "
                            f"{checkpoint_path}:"
                            f"{key} "
                            f"(held-out fold "
                            f"{eval_fold})"
                        ),
                    )

    # ---------------------------------------------------------
    # Old-checkpoint fallback:
    # reconstruct folds from the feature NPZ.
    # ---------------------------------------------------------
    resolved_feature = _resolve_feature_path(
        checkpoint_path,
        feature_path,
    )

    if resolved_feature is None:
        raise RuntimeError(
            "Checkpoint has no test_q_indices "
            "and the base feature file could "
            "not be found. Pass --feature_path "
            "features/<model>_layer<layer>.npz."
        )

    with np.load(
        resolved_feature,
        allow_pickle=False,
    ) as features:

        if "q_indices" in features.files:

            q_indices = np.asarray(
                features["q_indices"],
                dtype=np.int64,
            ).reshape(-1)

        elif (
            "vector_q_indices"
            in features.files
        ):

            q_indices = np.asarray(
                features[
                    "vector_q_indices"
                ],
                dtype=np.int64,
            ).reshape(-1)

        else:
            raise KeyError(
                f"{resolved_feature} has neither "
                "q_indices nor vector_q_indices."
            )

    questions = np.unique(
        q_indices
    )

    folds = np.array_split(
        questions,
        2,
    )

    ids = _validate_question_ids(
        folds[eval_fold],
        dataset_size,
        (
            f"{resolved_feature}:"
            f"reconstructed_fold"
            f"{eval_fold}"
        ),
    )

    return (
        ids,
        (
            f"{resolved_feature}: "
            f"reconstructed fold "
            f"{eval_fold} with "
            "np.array_split "
            "(same as train_dynSVD.py)"
        ),
    )


def load_model(
    model_name: str,
    model_dir: str | None,
    device: torch.device,
    dtype_name: str,
):

    model_path = (
        model_dir
        or HF_NAMES.get(
            model_name,
            model_name,
        )
    )

    print(
        f"Loading model: {model_path}"
    )

    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        trust_remote_code=True,
    )

    if tokenizer.pad_token_id is None:

        if tokenizer.eos_token_id is None:
            raise ValueError(
                "Tokenizer has neither "
                "pad_token_id nor eos_token_id."
            )

        tokenizer.pad_token = (
            tokenizer.eos_token
        )

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
        model = model.to(
            device
        )

    model.eval()

    return (
        model,
        tokenizer,
    )


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
    Normalized final prompt-token hidden state,
    matching router evaluation.
    """

    prompt = (
        f"Q: {question} A:"
    )

    if use_instruction:
        prompt = (
            get_instruction(
                model_name
            )
            + "\n\n"
            + prompt
        )

    input_device = next(
        model.get_input_embeddings()
        .parameters()
    ).device

    inputs = tokenizer(
        prompt,
        return_tensors="pt",
    )

    inputs = {
        key: value.to(
            input_device
        )
        for key, value
        in inputs.items()
    }

    outputs = model(
        **inputs,
        output_hidden_states=True,
        use_cache=False,
    )

    hidden_index = (
        layer + 1
    )

    if (
        hidden_index < 0
        or hidden_index
        >= len(
            outputs.hidden_states
        )
    ):
        raise ValueError(
            f"Requested layer={layer}, "
            f"but model returned "
            f"{len(outputs.hidden_states) - 1} "
            "transformer layers."
        )

    hidden = outputs.hidden_states[
        hidden_index
    ][:, -1, :].float()

    hidden = F.normalize(
        hidden,
        p=2,
        dim=-1,
        eps=1e-8,
    )

    return hidden.to(
        router_device
    )


@torch.inference_mode()
def compute_topk_statistics(
    checkpoint: dict,
    question_ids: list[int],
    questions: dict[int, str],
    model,
    tokenizer,
    model_name: str,
    layer: int,
    device: torch.device,
    use_instruction: bool,
    topk: int,
) -> tuple[
    np.ndarray,
    list[dict[str, object]],
]:
    """
    Stage 1:

    For every held-out question, select its Top-K router
    indices and count how often every router index appears.
    """

    rank = int(
        checkpoint["rank"]
    )

    if topk <= 0:
        raise ValueError(
            f"--topk must be positive, "
            f"got {topk}."
        )

    if topk > rank:
        raise ValueError(
            f"--topk={topk} exceeds "
            f"router rank={rank}."
        )

    counts = np.zeros(
        rank,
        dtype=np.int64,
    )

    per_question: list[
        dict[str, object]
    ] = []

    total = len(
        question_ids
    )

    for position, question_id in enumerate(
        question_ids,
        start=1,
    ):

        hidden = question_hidden_state(
            model=model,
            tokenizer=tokenizer,
            question=questions[
                question_id
            ],
            layer=layer,
            router_device=device,
            use_instruction=use_instruction,
            model_name=model_name,
        )

        if (
            hidden.shape[-1]
            != checkpoint["hidden_dim"]
        ):
            raise ValueError(
                f"Model hidden dim "
                f"{hidden.shape[-1]} "
                "does not match checkpoint "
                f"hidden dim "
                f"{checkpoint['hidden_dim']}."
            )

        weights = torch.softmax(
            checkpoint[
                "router"
            ].logits(hidden),
            dim=-1,
        )[0]

        topk_weights, topk_indices = (
            torch.topk(
                weights,
                k=topk,
                largest=True,
                sorted=True,
            )
        )

        indices_np = (
            topk_indices
            .detach()
            .cpu()
            .numpy()
            .astype(np.int64)
        )

        weights_np = (
            topk_weights
            .detach()
            .float()
            .cpu()
            .numpy()
        )

        # Each index can occur at most once
        # for a single question.
        counts[
            indices_np
        ] += 1

        per_question.append(
            {
                "question_id": int(
                    question_id
                ),
                "topk_vectors": [
                    int(index + 1)
                    for index
                    in indices_np.tolist()
                ],
                "topk_weights": [
                    float(weight)
                    for weight
                    in weights_np.tolist()
                ],
            }
        )

        if (
            position == 1
            or position % 50 == 0
            or position == total
        ):
            print(
                f"Processed "
                f"{position}/{total} "
                "held-out questions"
            )

    # ---------------------------------------------------------
    # Sanity check:
    #
    # Every question contributes exactly K selections.
    # ---------------------------------------------------------
    expected = (
        total * topk
    )

    actual = int(
        counts.sum()
    )

    if actual != expected:
        raise RuntimeError(
            "Top-K selection count mismatch: "
            f"expected {expected}, "
            f"got {actual}."
        )

    return (
        counts,
        per_question,
    )


def get_global_top_indices(
    counts: np.ndarray,
    display_topk: int,
) -> np.ndarray:
    """
    Stage 2:

    Sort router indices by total appearance frequency
    and retain only the globally most frequent indices.

    Ties are broken by smaller router index.
    """

    rank = int(
        counts.shape[0]
    )

    if display_topk <= 0:
        raise ValueError(
            "--display_topk must be positive, "
            f"got {display_topk}."
        )

    if display_topk > rank:
        raise ValueError(
            f"--display_topk="
            f"{display_topk} exceeds "
            f"router rank={rank}."
        )

    # Primary key: descending count.
    # Secondary key: ascending original router index.
    indices = np.arange(
        rank,
        dtype=np.int64,
    )

    sorted_indices = np.lexsort(
        (
            indices,
            -counts,
        )
    )

    return sorted_indices[
        :display_topk
    ]


def plot_topk_fractions(
    counts: np.ndarray,
    num_questions: int,
    eval_fold: int,
    question_topk: int,
    display_topk: int,
    output_prefix: Path,
) -> None:
    """
    Plot only the globally most frequent router indices.

    The y-value is:

        count(index) / number_of_heldout_questions

    rather than the raw count.
    """

    if num_questions <= 0:
        raise ValueError(
            "num_questions must be positive."
        )

    global_top_indices = (
        get_global_top_indices(
            counts,
            display_topk,
        )
    )

    global_counts = counts[
        global_top_indices
    ]

    global_fractions = (
        global_counts.astype(
            np.float64
        )
        / float(num_questions)
    )

    # Use compact categorical positions.
    positions = np.arange(
        display_topk
    )

    # Display router indices using 1-based numbering.
    router_labels = (
        global_top_indices + 1
    )

    figure, ax = plt.subplots(
        figsize=(11.5, 5.6)
    )

    ax.bar(
        positions,
        global_fractions,
        width=0.82,
    )

    ax.set_xticks(
        positions
    )

    ax.set_xticklabels(
        [
            str(index)
            for index
            in router_labels
        ],
        rotation=45,
        ha="right",
    )

    ax.set_xlim(
        -0.6,
        display_topk - 0.4,
    )

    max_fraction = float(
        global_fractions.max()
    )

    ax.set_ylim(
        0.0,
        min(
            1.0,
            max(
                0.05,
                max_fraction * 1.08,
            ),
        ),
    )

    # Show fractions as percentages.
    ax.yaxis.set_major_formatter(
        PercentFormatter(
            xmax=1.0,
            decimals=0,
        )
    )

    ax.set_xlabel(
        "Router index"
    )

    ax.set_ylabel(
        "Fraction of held-out questions"
    )

    ax.set_title(
        (
            f"Top {display_topk} most frequent "
            f"router indices "
            f"(per-question Top-{question_topk}, "
            f"held-out fold {eval_fold})"
        ),
        pad=8,
    )

    ax.tick_params(
        top=False,
        right=False,
    )

    ax.grid(
        axis="y",
        linewidth=0.8,
        alpha=0.25,
    )

    ax.set_axisbelow(
        True
    )

    figure.subplots_adjust(
        left=0.13,
        right=0.98,
        bottom=0.25,
        top=0.84,
    )

    output_prefix.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    for suffix in (
        ".png",
        ".pdf",
        ".svg",
    ):
        figure.savefig(
            output_prefix.with_suffix(
                suffix
            ),
            dpi=600,
        )

    plt.close(
        figure
    )


def write_data(
    output_prefix: Path,
    checkpoint_path: str,
    feature_path: str | None,
    router_fold: int,
    eval_fold: int,
    fold_source: str,
    question_ids: list[int],
    counts: np.ndarray,
    per_question: list[
        dict[str, object]
    ],
    question_topk: int,
    display_topk: int,
) -> None:
    """
    Write outputs corresponding to the final global Top-K figure.

    The main CSV and JSON store fractions instead of raw counts
    as the primary reported statistic.
    """

    total = len(
        question_ids
    )

    if total <= 0:
        raise ValueError(
            "No held-out questions."
        )

    global_top_indices = (
        get_global_top_indices(
            counts,
            display_topk,
        )
    )

    global_fractions = (
        counts[
            global_top_indices
        ].astype(
            np.float64
        )
        / float(total)
    )

    # ---------------------------------------------------------
    # 1. Final global Top-K frequency CSV
    #
    # Exactly `display_topk` rows.
    # ---------------------------------------------------------
    frequency_csv = (
        output_prefix.with_name(
            output_prefix.name
            + "_frequencies"
        ).with_suffix(
            ".csv"
        )
    )

    frequency_csv.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with frequency_csv.open(
        "w",
        encoding="utf-8",
        newline="",
    ) as handle:

        writer = csv.writer(
            handle
        )

        writer.writerow(
            [
                "frequency_rank",
                "index",
                "dataset_fraction",
                "dataset_percentage",
            ]
        )

        for (
            frequency_rank,
            (
                index,
                fraction,
            ),
        ) in enumerate(
            zip(
                global_top_indices,
                global_fractions,
            ),
            start=1,
        ):

            writer.writerow(
                [
                    int(
                        frequency_rank
                    ),
                    int(
                        index + 1
                    ),
                    f"{float(fraction):.9g}",
                    f"{float(fraction) * 100:.6g}",
                ]
            )

    # ---------------------------------------------------------
    # 2. Per-question selections.
    # Keep this because it allows the figure to be audited.
    # ---------------------------------------------------------
    question_csv = (
        output_prefix.with_name(
            output_prefix.name
            + "_questions"
        ).with_suffix(
            ".csv"
        )
    )

    with question_csv.open(
        "w",
        encoding="utf-8",
        newline="",
    ) as handle:

        writer = csv.writer(
            handle
        )

        writer.writerow(
            [
                "question_id",
                "topk_vectors",
                "topk_weights",
            ]
        )

        for item in per_question:

            vectors = ";".join(
                str(int(value))
                for value
                in item[
                    "topk_vectors"
                ]
            )

            weights = ";".join(
                f"{float(value):.9g}"
                for value
                in item[
                    "topk_weights"
                ]
            )

            writer.writerow(
                [
                    item[
                        "question_id"
                    ],
                    vectors,
                    weights,
                ]
            )

    # ---------------------------------------------------------
    # 3. JSON summary.
    # ---------------------------------------------------------
    payload = {
        "experiment": (
            "fig2b_global_topk_router_frequency_heldout_fold"
        ),
        "checkpoint": checkpoint_path,
        "feature_path": feature_path,
        "router_fold": int(
            router_fold
        ),
        "eval_fold": int(
            eval_fold
        ),
        "fold_id_source": fold_source,
        "num_heldout_questions": int(
            total
        ),
        "router_rank": int(
            counts.shape[0]
        ),

        # Per-question K.
        "question_topk": int(
            question_topk
        ),

        # Final number of displayed indices.
        "display_topk": int(
            display_topk
        ),

        "fraction_denominator": (
            "num_heldout_questions"
        ),

        "total_topk_selections": int(
            counts.sum()
        ),

        "expected_topk_selections": int(
            total
            * question_topk
        ),

        "num_distinct_indices_selected": int(
            np.count_nonzero(
                counts
            )
        ),

        "global_top_indices": [
            {
                "frequency_rank": int(
                    frequency_rank
                ),
                "index": int(
                    index + 1
                ),
                "dataset_fraction": float(
                    fraction
                ),
                "dataset_percentage": float(
                    fraction * 100.0
                ),
            }
            for frequency_rank, (
                index,
                fraction,
            )
            in enumerate(
                zip(
                    global_top_indices,
                    global_fractions,
                ),
                start=1,
            )
        ],
    }

    output_prefix.with_suffix(
        ".json"
    ).write_text(
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
    set_seed(
        args.seed
    )

    device = resolve_device(
        args.device
    )

    checkpoint = (
        load_router_checkpoint(
            args.checkpoint,
            device,
        )
    )

    router_fold, eval_fold = (
        resolve_fold_pair(
            checkpoint_path=(
                args.checkpoint
            ),
            checkpoint=checkpoint,
            router_fold_arg=(
                args.router_fold
            ),
            eval_fold_arg=(
                args.eval_fold
            ),
        )
    )

    rank = int(
        checkpoint["rank"]
    )

    if args.topk <= 0:
        raise ValueError(
            f"--topk must be positive, "
            f"got {args.topk}."
        )

    if args.topk > rank:
        raise ValueError(
            f"--topk={args.topk} "
            f"exceeds router rank="
            f"{rank}."
        )

    if args.display_topk <= 0:
        raise ValueError(
            "--display_topk must be "
            f"positive, got "
            f"{args.display_topk}."
        )

    if args.display_topk > rank:
        raise ValueError(
            f"--display_topk="
            f"{args.display_topk} "
            f"exceeds router rank="
            f"{rank}."
        )

    layer = args.layer

    if layer is None:
        layer = infer_layer_from_filename(
            args.checkpoint
        )

    if layer is None:
        raise ValueError(
            "Could not infer layer from "
            "checkpoint filename. "
            "Pass --layer explicitly."
        )

    _, dataset = (
        load_truthfulqa_mc_dataframe(
            args.csv_path
        )
    )

    dataset_size = len(
        dataset
    )

    question_ids, fold_source = (
        load_heldout_question_ids(
            checkpoint_path=(
                args.checkpoint
            ),
            feature_path=(
                args.feature_path
            ),
            dataset_size=(
                dataset_size
            ),
            router_fold=(
                router_fold
            ),
            eval_fold=(
                eval_fold
            ),
        )
    )

    print(
        f"Router fold: "
        f"{router_fold}"
    )

    print(
        f"Evaluation fold: "
        f"{eval_fold}"
    )

    print(
        f"Per-question Top-K: "
        f"{args.topk}"
    )

    print(
        f"Global display Top-K: "
        f"{args.display_topk}"
    )

    print(
        f"Held-out questions: "
        f"{len(question_ids)}"
    )

    print(
        f"Fold membership source: "
        f"{fold_source}"
    )

    questions = {
        question_id: str(
            dataset[
                question_id
            ]["question"]
        )
        for question_id
        in question_ids
    }

    model, tokenizer = (
        load_model(
            args.model_name,
            args.model_dir,
            device,
            args.dtype,
        )
    )

    counts, per_question = (
        compute_topk_statistics(
            checkpoint=checkpoint,
            question_ids=question_ids,
            questions=questions,
            model=model,
            tokenizer=tokenizer,
            model_name=args.model_name,
            layer=layer,
            device=device,
            use_instruction=(
                not args.no_instruction
            ),
            topk=args.topk,
        )
    )

    output_prefix = Path(
        args.output_prefix
    )

    write_data(
        output_prefix=output_prefix,
        checkpoint_path=(
            args.checkpoint
        ),
        feature_path=(
            args.feature_path
        ),
        router_fold=router_fold,
        eval_fold=eval_fold,
        fold_source=fold_source,
        question_ids=question_ids,
        counts=counts,
        per_question=per_question,
        question_topk=args.topk,
        display_topk=(
            args.display_topk
        ),
    )

    plot_topk_fractions(
        counts=counts,
        num_questions=(
            len(question_ids)
        ),
        eval_fold=eval_fold,
        question_topk=args.topk,
        display_topk=(
            args.display_topk
        ),
        output_prefix=(
            output_prefix
        ),
    )

    # ---------------------------------------------------------
    # Terminal summary:
    # show percentage rather than raw count.
    # ---------------------------------------------------------
    global_top_indices = (
        get_global_top_indices(
            counts,
            args.display_topk,
        )
    )

    fractions = (
        counts[
            global_top_indices
        ].astype(
            np.float64
        )
        / float(
            len(question_ids)
        )
    )

    print()
    print(
        f"Questions processed: "
        f"{len(question_ids)}"
    )

    print(
        f"Full router rank: "
        f"{rank}"
    )

    print(
        f"Per-question Top-K: "
        f"{args.topk}"
    )

    print(
        f"Global Top-K displayed: "
        f"{args.display_topk}"
    )

    print(
        f"Total Top-K selections: "
        f"{int(counts.sum())}"
    )

    print(
        "Expected Top-K selections: "
        f"{len(question_ids) * args.topk}"
    )

    print()
    print(
        "Globally most frequent "
        "router indices:"
    )

    for (
        frequency_rank,
        (
            index,
            fraction,
        ),
    ) in enumerate(
        zip(
            global_top_indices,
            fractions,
        ),
        start=1,
    ):

        print(
            f"  {frequency_rank:2d}. "
            f"index {index + 1:4d}: "
            f"{fraction * 100:6.2f}%"
        )

    print()
    print(
        f"Figure: "
        f"{output_prefix.with_suffix('.png')}"
    )

    print(
        "Frequency CSV: "
        f"{output_prefix.with_name(output_prefix.name + '_frequencies').with_suffix('.csv')}"
    )


if __name__ == "__main__":
    main()
