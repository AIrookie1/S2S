"""
Stage 1: extract TruthfulQA hidden states and build pairwise contrastive vectors.

TruthfulQA MC2 normally contains several truthful choices and several
hallucinated choices for each question.

This script stores the hidden state of every question-answer sample. It does
not average choices within a question. Instead, every truthful choice is paired
with every hallucinated choice from the same question.

For question i, truthful choice p, and hallucinated choice n:

    z_i_pos_p = hidden state of truthful choice p
    z_i_neg_n = hidden state of hallucinated choice n

    v_i_p_n_T = z_i_pos_p - z_i_neg_n
    v_i_p_n_H = -v_i_p_n_T

If a question contains P truthful choices and N hallucinated choices, the
question produces P * N contrastive vectors.

The hidden state is taken from the last non-padding token at the requested
transformer layer.

Example:
    python stage1.py llama3.1-8B --layer 14 --save_dir ./features

Output NPZ fields:

Answer-level fields:
    activations            [num_answer_choices, hidden_dim]
    labels                 [num_answer_choices]
    q_indices              [num_answer_choices]
    choice_indices         [num_answer_choices]
    prompts                [num_answer_choices]
    choices                [num_answer_choices]
    questions              [num_questions]
    question_offsets       [num_questions + 1]

Pair-level fields:
    z_pos                  [num_positive_negative_pairs, hidden_dim]
    z_neg                  [num_positive_negative_pairs, hidden_dim]
    v_T                    [num_positive_negative_pairs, hidden_dim]
    v_H                    [num_positive_negative_pairs, hidden_dim]

    vector_q_indices       [num_positive_negative_pairs]
    positive_sample_indices
                            [num_positive_negative_pairs]
    negative_sample_indices
                            [num_positive_negative_pairs]
    positive_choice_indices
                            [num_positive_negative_pairs]
    negative_choice_indices
                            [num_positive_negative_pairs]

Question-level pair metadata:
    vector_question_offsets
                            [num_questions + 1]
    positive_counts        [num_questions]
    negative_counts        [num_questions]
    pair_counts            [num_questions]

Other metadata:
    layer
    model_name
    model_path
    dataset_name
    dataset_config
    target_name
    vector_definition

Contrast-protected fields:
    hallucination_covariance     [hidden_dim, hidden_dim]
    truth_support_matrix         [hidden_dim, hidden_dim]
    bad_basis                    [hidden_dim, k]
    eigenvalues                  [hidden_dim]
    good_steer, g_steer, v       [hidden_dim]
"""

import argparse
import math
import os
from typing import Dict, List, Tuple

import numpy as np
import torch
from datasets import load_dataset
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer


HF_NAMES = {
    "llama3.1-8B": "meta-llama/Llama-3.1-8B",
    "llama3.1-8B-Instruct": "meta-llama/Llama-3.1-8B-Instruct",
    "Qwen2.5-7B-Instruct": "Qwen/Qwen2.5-7B-Instruct",
}


def format_truthfulqa_prompt(
    question: str,
    choice: str,
    style: str = "standard",
) -> str:
    """
    Construct the input prompt for one TruthfulQA answer choice.
    """
    if style != "standard":
        raise ValueError(f"Unknown style: {style}")

    return f"Q: {question} A: {choice}"


def get_truthfulqa_data(style: str = "standard") -> Dict[str, np.ndarray]:
    """
    Load TruthfulQA MC2 and flatten all answer choices.

    Each answer choice becomes one model input. The returned q_indices array
    records which question each answer choice belongs to.
    """
    print(f"Loading TruthfulQA dataset (style={style}, MC2 targets)...")

    dataset = load_dataset(
        "truthful_qa",
        "multiple_choice",
    )["validation"]

    questions: List[str] = []
    prompts: List[str] = []
    choices: List[str] = []
    labels: List[int] = []
    q_indices: List[int] = []
    choice_indices: List[int] = []

    # question_offsets[i] is the first answer-level row for question i.
    question_offsets = [0]

    for q_idx, item in enumerate(dataset):
        question = item["question"]
        questions.append(question)

        mc2 = item["mc2_targets"]
        mc2_choices = mc2["choices"]
        mc2_labels = mc2["labels"]

        if len(mc2_choices) != len(mc2_labels):
            raise ValueError(
                f"Question {q_idx} has mismatched MC2 choices and labels: "
                f"{len(mc2_choices)} choices versus "
                f"{len(mc2_labels)} labels."
            )

        for choice_idx, (choice, label) in enumerate(
            zip(mc2_choices, mc2_labels)
        ):
            prompts.append(
                format_truthfulqa_prompt(
                    question=question,
                    choice=choice,
                    style=style,
                )
            )
            choices.append(choice)
            labels.append(int(label))
            q_indices.append(q_idx)
            choice_indices.append(choice_idx)

        question_offsets.append(len(prompts))

    print(
        f"Processed {len(questions)} questions into "
        f"{len(prompts)} question-answer samples."
    )

    return {
        "questions": np.asarray(questions, dtype=str),
        "prompts": np.asarray(prompts, dtype=str),
        "choices": np.asarray(choices, dtype=str),
        "labels": np.asarray(labels, dtype=np.int64),
        "q_indices": np.asarray(q_indices, dtype=np.int64),
        "choice_indices": np.asarray(choice_indices, dtype=np.int64),
        "question_offsets": np.asarray(
            question_offsets,
            dtype=np.int64,
        ),
    }


def resolve_device(requested: str) -> torch.device:
    """
    Resolve the requested inference device.
    """
    if requested == "cuda" and not torch.cuda.is_available():
        print(
            "CUDA was requested but is unavailable; "
            "falling back to CPU."
        )
        return torch.device("cpu")

    return torch.device(requested)


def resolve_dtype(
    dtype_name: str,
    device: torch.device,
) -> torch.dtype:
    """
    Resolve the requested model dtype.

    CPU inference uses float32 because float16 operations may not be fully
    supported or efficient on CPU.
    """
    if device.type == "cpu":
        if dtype_name != "float32":
            print(
                f"Requested dtype {dtype_name} on CPU; "
                "using float32 instead."
            )
        return torch.float32

    dtype_map = {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }

    return dtype_map[dtype_name]


def get_model_input_device(model) -> torch.device:
    """
    Find the device on which token IDs should be placed.

    This is safer than assuming cuda:0 when device_map='auto' is used.
    """
    try:
        embedding_device = model.get_input_embeddings().weight.device

        if embedding_device.type != "meta":
            return embedding_device
    except (AttributeError, RuntimeError):
        pass

    for parameter in model.parameters():
        if parameter.device.type != "meta":
            return parameter.device

    raise RuntimeError(
        "Could not determine a valid model input device."
    )


def extract_layer_activations(
    model,
    tokenizer,
    prompts: np.ndarray,
    layer_idx: int,
    input_device: torch.device,
    batch_size: int,
    max_length: int,
) -> np.ndarray:
    """
    Extract one hidden state for every prompt.

    For each prompt, the hidden state is selected from the last token whose
    attention-mask value is 1. This works with both left and right padding.

    layer_idx is zero-based:
        layer_idx=0 means the first transformer block.
    """
    activations: List[np.ndarray] = []

    model.eval()

    tokenizer_kwargs = {
        "padding": True,
        "return_tensors": "pt",
    }

    if max_length > 0:
        tokenizer_kwargs.update(
            {
                "truncation": True,
                "max_length": max_length,
            }
        )
    else:
        tokenizer_kwargs.update(
            {
                "truncation": False,
            }
        )

    with torch.inference_mode():
        for start in tqdm(
            range(0, len(prompts), batch_size),
            desc=f"Extracting layer {layer_idx}",
        ):
            end = min(start + batch_size, len(prompts))
            batch_prompts = prompts[start:end].tolist()

            encoded = tokenizer(
                batch_prompts,
                **tokenizer_kwargs,
            )

            encoded = {
                key: value.to(input_device)
                for key, value in encoded.items()
            }

            outputs = model(
                **encoded,
                output_hidden_states=True,
                use_cache=False,
                return_dict=True,
            )

            hidden_states = outputs.hidden_states

            if hidden_states is None:
                raise RuntimeError(
                    "The model did not return hidden states."
                )

            # hidden_states[0] is the embedding output.
            # hidden_states[layer_idx + 1] is transformer block layer_idx.
            hidden_position = layer_idx + 1
            num_transformer_layers = len(hidden_states) - 1

            if (
                hidden_position <= 0
                or hidden_position >= len(hidden_states)
            ):
                raise IndexError(
                    f"Requested transformer layer {layer_idx}, but the "
                    f"model returned {num_transformer_layers} transformer "
                    "layers. Valid layer indices are "
                    f"0 to {num_transformer_layers - 1}."
                )

            layer_hidden = hidden_states[hidden_position]

            # With model sharding, layer_hidden may be on a different device
            # from the original attention mask.
            hidden_device = layer_hidden.device
            attention_mask = encoded["attention_mask"].to(hidden_device)

            sequence_length = attention_mask.shape[1]

            token_positions = torch.arange(
                sequence_length,
                device=hidden_device,
            ).unsqueeze(0)

            # Multiplying positions by the attention mask gives zero for
            # padding tokens. The maximum remaining position is therefore
            # the last non-padding token.
            last_positions = (
                token_positions * attention_mask
            ).max(dim=1).values

            batch_rows = torch.arange(
                layer_hidden.shape[0],
                device=hidden_device,
            )

            batch_activations = layer_hidden[
                batch_rows,
                last_positions,
                :,
            ]

            batch_activations = (
                batch_activations
                .detach()
                .float()
                .cpu()
                .numpy()
            )

            finite_rows = np.isfinite(batch_activations).all(axis=1)

            if not finite_rows.all():
                bad_local_rows = np.where(~finite_rows)[0]
                bad_global_rows = start + bad_local_rows

                raise FloatingPointError(
                    "Non-finite activation detected for global sample "
                    f"indices {bad_global_rows.tolist()}."
                )

            activations.append(batch_activations)

            # Avoid retaining hidden states longer than necessary.
            del outputs
            del hidden_states
            del layer_hidden
            del batch_activations

    if not activations:
        raise ValueError("No prompts were provided for activation extraction.")

    return np.concatenate(
        activations,
        axis=0,
    ).astype(
        np.float32,
        copy=False,
    )


def build_question_vectors(
    activations: np.ndarray,
    labels: np.ndarray,
    q_indices: np.ndarray,
    choice_indices: np.ndarray,
) -> Tuple[np.ndarray, ...]:
    """
    Build every truthful-hallucinated pair within each question.

    No mean or averaging operation is performed.

    For every question q, every positive answer p, and every negative answer n:

        z_pos = activation[q, p]
        z_neg = activation[q, n]

        v_T = z_pos - z_neg
        v_H = -v_T

    If a question has P positive choices and N negative choices, this function
    creates P * N vectors for that question.
    """
    num_samples = activations.shape[0]

    if labels.shape[0] != num_samples:
        raise ValueError(
            "labels and activations must have the same first dimension."
        )

    if q_indices.shape[0] != num_samples:
        raise ValueError(
            "q_indices and activations must have the same first dimension."
        )

    if choice_indices.shape[0] != num_samples:
        raise ValueError(
            "choice_indices and activations must have the same first "
            "dimension."
        )

    if activations.ndim != 2:
        raise ValueError(
            "activations must have shape "
            "[num_answer_choices, hidden_dim]."
        )

    unique_q_indices = np.unique(q_indices).astype(np.int64)

    vector_q_indices: List[int] = []

    positive_sample_indices: List[int] = []
    negative_sample_indices: List[int] = []

    positive_choice_indices: List[int] = []
    negative_choice_indices: List[int] = []

    z_pos: List[np.ndarray] = []
    z_neg: List[np.ndarray] = []
    v_t: List[np.ndarray] = []

    positive_counts: List[int] = []
    negative_counts: List[int] = []
    pair_counts: List[int] = []

    # vector_question_offsets[q] gives the beginning of question q's
    # pair-level vectors, assuming questions are stored in ascending order.
    vector_question_offsets = [0]

    for q_idx in unique_q_indices:
        q_rows = np.flatnonzero(q_indices == q_idx)

        pos_rows = q_rows[labels[q_rows] == 1]
        neg_rows = q_rows[labels[q_rows] == 0]

        num_positive = len(pos_rows)
        num_negative = len(neg_rows)

        if num_positive == 0 or num_negative == 0:
            raise ValueError(
                f"Question {q_idx} does not contain both truthful and "
                "hallucinated MC2 choices. "
                f"positive={num_positive}, negative={num_negative}."
            )

        positive_counts.append(num_positive)
        negative_counts.append(num_negative)

        question_pair_count = num_positive * num_negative
        pair_counts.append(question_pair_count)

        # Cartesian product:
        # every truthful choice is compared with every hallucinated choice.
        for pos_row in pos_rows:
            for neg_row in neg_rows:
                pos_activation = activations[pos_row]
                neg_activation = activations[neg_row]

                difference = pos_activation - neg_activation

                z_pos.append(pos_activation)
                z_neg.append(neg_activation)
                v_t.append(difference)

                vector_q_indices.append(int(q_idx))

                positive_sample_indices.append(int(pos_row))
                negative_sample_indices.append(int(neg_row))

                positive_choice_indices.append(
                    int(choice_indices[pos_row])
                )
                negative_choice_indices.append(
                    int(choice_indices[neg_row])
                )

        vector_question_offsets.append(len(v_t))

    if not v_t:
        raise ValueError(
            "No positive-negative contrastive pairs were generated."
        )

    z_pos_array = np.stack(
        z_pos,
        axis=0,
    ).astype(
        np.float32,
        copy=False,
    )

    z_neg_array = np.stack(
        z_neg,
        axis=0,
    ).astype(
        np.float32,
        copy=False,
    )

    v_t_array = np.stack(
        v_t,
        axis=0,
    ).astype(
        np.float32,
        copy=False,
    )

    v_h_array = (-v_t_array).astype(
        np.float32,
        copy=False,
    )

    return (
        np.asarray(vector_q_indices, dtype=np.int64),
        np.asarray(positive_sample_indices, dtype=np.int64),
        np.asarray(negative_sample_indices, dtype=np.int64),
        np.asarray(positive_choice_indices, dtype=np.int64),
        np.asarray(negative_choice_indices, dtype=np.int64),
        np.asarray(vector_question_offsets, dtype=np.int64),
        z_pos_array,
        z_neg_array,
        v_t_array,
        v_h_array,
        np.asarray(positive_counts, dtype=np.int64),
        np.asarray(negative_counts, dtype=np.int64),
        np.asarray(pair_counts, dtype=np.int64),
    )


@torch.no_grad()
def build_bad_orthogonalized_steer(
    activations: np.ndarray,
    labels: np.ndarray,
    q_indices: np.ndarray,
    energy_threshold: float = 0.95,
    k_max: int | None = 64,
    remove_strength: float = 1.0,
    specificity_threshold: float = 1.0,
    eps: float = 1e-8,
) -> Tuple[dict, dict]:
    """Construct contrast-protected hallucination-subspace steering.

    For each question, all truthful choices are averaged (the usual dataset
    has one) and all hallucinated choices form a question-local center.  The
    hallucination covariance is averaged per question, while the truth
    support matrix is built from normalized per-question contrasts.
    """
    activations = np.asarray(activations, dtype=np.float32)
    labels = np.asarray(labels).reshape(-1)
    q_indices = np.asarray(q_indices).reshape(-1)
    if activations.ndim != 2 or len(labels) != len(activations) or len(q_indices) != len(activations):
        raise ValueError("activations must be [num_samples, hidden_dim] and align with labels/q_indices")
    if k_max is not None and k_max <= 0:
        raise ValueError("k_max must be positive")
    if not math.isfinite(remove_strength) or not 0.0 <= remove_strength <= 1.0:
        raise ValueError("remove_strength must be in [0, 1]")
    if not math.isfinite(specificity_threshold) or specificity_threshold <= 0:
        raise ValueError("specificity_threshold must be positive")

    g_questions, selected_q = [], []
    covariance_sum = None
    support_sum = None
    question_count = 0
    for q in np.unique(q_indices):
        rows = np.flatnonzero(q_indices == q)
        neg = rows[labels[rows] == 0]
        pos = rows[labels[rows] == 1]
        if len(neg) == 0 or len(pos) == 0:
            continue
        h_pos = activations[pos]
        h_neg = activations[neg]
        mu_t = h_pos.mean(axis=0)
        mu_h = h_neg.mean(axis=0)
        g_q = mu_t - mu_h
        residual_h = h_neg - mu_h
        covariance_q = (residual_h.T @ residual_h) / float(len(neg))
        g_questions.append(g_q)
        g_hat = g_q / (np.linalg.norm(g_q) + eps)
        support_q = np.outer(g_hat, g_hat)
        covariance_sum = covariance_q if covariance_sum is None else covariance_sum + covariance_q
        support_sum = support_q if support_sum is None else support_sum + support_q
        question_count += 1
        selected_q.append(int(q))

    if not g_questions:
        raise ValueError("Training data must contain questions with truthful and hallucinated choices")
    g = torch.as_tensor(np.stack(g_questions), dtype=torch.float32).mean(dim=0)
    C_H = torch.as_tensor(covariance_sum / float(question_count), dtype=torch.float32)
    C_G = torch.as_tensor(support_sum / float(question_count), dtype=torch.float32)
    eigvals, eigvecs = torch.linalg.eigh(C_H)
    order = torch.argsort(eigvals, descending=True)
    eigvals = eigvals[order]
    eigvecs = eigvecs[:, order]
    k = min(int(k_max), eigvecs.shape[1]) if k_max is not None else eigvecs.shape[1]
    Q = eigvecs[:, :k].contiguous()
    if k:
        trace_h = torch.trace(C_H)
        C_H_bar = C_H / (trace_h + eps)
        hallucination_strength = torch.diag(Q.T @ C_H_bar @ Q)
        truth_support = torch.diag(Q.T @ C_G @ Q)
        ratios = hallucination_strength / (truth_support + eps)
        soft = torch.where(
            ratios > float(specificity_threshold),
            torch.clamp(1.0 - 1.0 / ratios, min=0.0),
            torch.zeros_like(ratios),
        )
        g_norm = torch.linalg.norm(g)
        cosine = torch.abs(Q.T @ g) / (g_norm + eps)
        protection = torch.clamp(1.0 - cosine.square(), min=0.0, max=1.0)
        weights = soft * protection
        g_steer = g - float(remove_strength) * (Q * (weights * (Q.T @ g))).sum(dim=1)
    else:
        trace_h = torch.trace(C_H)
        C_H_bar = C_H
        hallucination_strength = truth_support = ratios = soft = protection = weights = torch.empty(0)
        g_steer = g.clone()
    v = g_steer / (g_steer.norm() + eps)
    g_norm = g.norm()
    stats = {
        "bad_overlap": float((Q.T @ g).norm() / (g_norm + eps)) if k else 0.0,
        "retained_ratio": float(v.norm()),
        "k": k,
        "d": int(Q.shape[0]),
        "subspace_ratio": float(k / Q.shape[0]),
        "specificity_threshold": float(specificity_threshold),
        "covariance_trace": float(trace_h),
    }
    arrays = {
        "hallucination_covariance": C_H.numpy(), "truth_support_matrix": C_G.numpy(),
        "bad_matrix": C_H.numpy(), "bad_basis": Q.numpy(), "eigenvalues": eigvals.numpy(),
        "specificity_ratios": ratios.numpy(), "hallucination_strength": hallucination_strength.numpy(),
        "truth_support": truth_support.numpy(), "suppression_weights": weights.numpy(),
        "protection_weights": protection.numpy(),
        "g_raw": g.numpy(),
        "good_steer": g.numpy(), "g_steer": v.numpy(),
        "bad_q_indices": np.asarray(selected_q, dtype=np.int64),
    }
    return arrays, stats


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Extract every TruthfulQA answer-choice hidden state and build "
            "one contrastive vector for every truthful-hallucinated choice "
            "pair within each question. No question-level averaging is used."
        )
    )

    parser.add_argument(
        "model_name",
        type=str,
        help=(
            "Model alias or Hugging Face model name. "
            f"Known aliases: {list(HF_NAMES.keys())}"
        ),
    )

    parser.add_argument(
        "--layer",
        type=int,
        default=14,
        help=(
            "Zero-based transformer layer index. "
            "For example, --layer 14 selects transformer block 14."
        ),
    )

    parser.add_argument(
        "--save_dir",
        type=str,
        default="./features",
    )

    parser.add_argument(
        "--output_name",
        type=str,
        default=None,
        help=(
            "Optional output filename. "
            "Defaults to <model>_layer<L>.npz."
        ),
    )

    parser.add_argument(
        "--model_dir",
        type=str,
        default=None,
        help=(
            "Optional local model directory. "
            "When provided, it overrides model_name."
        ),
    )

    parser.add_argument(
        "--batch_size",
        type=int,
        default=1,
    )

    parser.add_argument(
        "--max_length",
        type=int,
        default=0,
        help=(
            "Maximum token length. "
            "Use 0 to disable truncation."
        ),
    )

    parser.add_argument(
        "--device",
        choices=["cuda", "cpu"],
        default="cuda",
    )

    parser.add_argument(
        "--dtype",
        choices=["float16", "bfloat16", "float32"],
        default="float16",
    )
    parser.add_argument("--energy_threshold", type=float, default=0.95,
                        help="Deprecated compatibility option; candidate count is controlled by --k_max.")
    parser.add_argument("--k_max", type=int, default=None)
    parser.add_argument("--remove_strength", type=float, default=1.0)
    parser.add_argument("--specificity_threshold", type=float, default=1.0)
    parser.add_argument("--train_fraction", type=float, default=0.8,
                        help="Fraction of questions used to construct steering statistics.")
    parser.add_argument("--split_seed", type=int, default=42)

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.layer < 0:
        raise ValueError("--layer must be greater than or equal to 0.")

    if args.batch_size <= 0:
        raise ValueError("--batch_size must be positive.")

    if args.max_length < 0:
        raise ValueError("--max_length must be greater than or equal to 0.")

    device = resolve_device(args.device)
    dtype = resolve_dtype(args.dtype, device)

    model_path = (
        args.model_dir
        or HF_NAMES.get(args.model_name, args.model_name)
    )

    print(f"Loading model: {model_path}")
    print(f"Requested device: {device}")
    print(f"Model dtype: {dtype}")

    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        trust_remote_code=True,
    )

    if tokenizer.pad_token_id is None:
        if tokenizer.eos_token_id is None:
            raise ValueError(
                "The tokenizer has neither a pad token nor an EOS token. "
                "A valid padding token is required for batching."
            )

        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        low_cpu_mem_usage=True,
        dtype=torch.float16,
        trust_remote_code=True,
    ).to("cuda")

    model.eval()

    input_device = get_model_input_device(model)

    print(f"Model input device: {input_device}")

    data = get_truthfulqa_data(style="standard")

    print("\n=== Data statistics ===")
    print(f"Questions: {len(data['questions'])}")
    print(f"QA samples: {len(data['prompts'])}")
    print(f"Truthful samples: {int(data['labels'].sum())}")
    print(
        "Hallucinated samples: "
        f"{int((data['labels'] == 0).sum())}"
    )

    activations = extract_layer_activations(
        model=model,
        tokenizer=tokenizer,
        prompts=data["prompts"],
        layer_idx=args.layer,
        input_device=input_device,
        batch_size=args.batch_size,
        max_length=args.max_length,
    )

    if not 0.0 < args.train_fraction <= 1.0:
        raise ValueError("--train_fraction must be in (0, 1]")
    all_questions = np.unique(data["q_indices"])
    rng = np.random.default_rng(args.split_seed)
    shuffled = all_questions.copy()
    rng.shuffle(shuffled)
    n_train = max(1, int(round(len(shuffled) * args.train_fraction)))
    train_questions = np.sort(shuffled[:n_train])
    test_questions = np.sort(shuffled[n_train:])
    train_mask = np.isin(data["q_indices"], train_questions)

    (
        vector_q_indices,
        positive_sample_indices,
        negative_sample_indices,
        positive_choice_indices,
        negative_choice_indices,
        vector_question_offsets,
        z_pos,
        z_neg,
        v_t,
        v_h,
        positive_counts,
        negative_counts,
        pair_counts,
    ) = build_question_vectors(
        activations=activations,
        labels=data["labels"],
        q_indices=data["q_indices"],
        choice_indices=data["choice_indices"],
    )

    bad_data, bad_stats = build_bad_orthogonalized_steer(
        activations=activations[train_mask],
        labels=data["labels"][train_mask],
        q_indices=data["q_indices"][train_mask],
        energy_threshold=args.energy_threshold,
        k_max=args.k_max,
        remove_strength=args.remove_strength,
        specificity_threshold=args.specificity_threshold,
    )
    print(
        "Bad subspace: "
        f"Q={bad_data['bad_basis'].shape}, k={bad_stats['k']}, "
        f"bad_overlap={bad_stats['bad_overlap']:.4f}, "
        f"retained_ratio={bad_stats['retained_ratio']:.4f}"
    )

    print("\n=== Pair statistics ===")
    print(f"Total contrastive pairs: {len(v_t)}")
    print(
        "Minimum pairs per question: "
        f"{int(pair_counts.min())}"
    )
    print(
        "Maximum pairs per question: "
        f"{int(pair_counts.max())}"
    )
    print(
        "Mean pairs per question: "
        f"{float(pair_counts.mean()):.2f}"
    )

    os.makedirs(
        args.save_dir,
        exist_ok=True,
    )

    safe_model_name = (
        args.model_name
        .replace("/", "_")
        .replace("\\", "_")
    )

    save_name = (
        args.output_name
        or f"{safe_model_name}_layer{args.layer}.npz"
    )

    save_path = os.path.join(
        args.save_dir,
        save_name,
    )

    np.savez_compressed(
        save_path,

        # Answer-level data
        activations=activations,
        labels=data["labels"],
        q_indices=data["q_indices"],
        choice_indices=data["choice_indices"],
        prompts=data["prompts"],
        choices=data["choices"],
        questions=data["questions"],
        question_offsets=data["question_offsets"],
        train_q_indices=train_questions,
        test_q_indices=test_questions,
        split_seed=np.asarray(args.split_seed, dtype=np.int64),

        # Pair-level mapping
        vector_q_indices=vector_q_indices,
        positive_sample_indices=positive_sample_indices,
        negative_sample_indices=negative_sample_indices,
        positive_choice_indices=positive_choice_indices,
        negative_choice_indices=negative_choice_indices,
        vector_question_offsets=vector_question_offsets,

        # Pair-level hidden states and vectors
        z_pos=z_pos,
        z_neg=z_neg,
        v_T=v_t,
        v_H=v_h,

        # Question-level counts
        positive_counts=positive_counts,
        negative_counts=negative_counts,
        pair_counts=pair_counts,

        # Metadata
        layer=np.asarray(args.layer, dtype=np.int64),
        model_name=np.asarray(args.model_name),
        model_path=np.asarray(model_path),
        dataset_name=np.asarray("truthful_qa"),
        dataset_config=np.asarray("multiple_choice"),
        target_name=np.asarray("mc2_targets"),
        pairing_method=np.asarray(
            "cartesian_product_within_question"
        ),
        vector_definition=np.asarray(
            "Contrast-protected hallucination subspace steering: per-question "
            "g_q=mean(truthful)-mean(hallucinated), g=mean_q(g_q); C_H is "
            "the equal-question-weighted hallucination residual covariance; "
            "C_G is normalized truth-contrast support; candidate directions "
            "are softly suppressed using specificity and global-truth protection."
        ),
        # Contrast-protected hallucination-subspace steering outputs.
        **bad_data,
        bad_overlap=np.asarray(bad_stats["bad_overlap"], dtype=np.float32),
        retained_ratio=np.asarray(bad_stats["retained_ratio"], dtype=np.float32),
        subspace_ratio=np.asarray(bad_stats["subspace_ratio"], dtype=np.float32),
        bad_subspace_k=np.asarray(bad_stats["k"], dtype=np.int64),
        bad_subspace_energy_threshold=np.asarray(args.energy_threshold, dtype=np.float32),
        bad_subspace_remove_strength=np.asarray(args.remove_strength, dtype=np.float32),
        bad_subspace_specificity_threshold=np.asarray(args.specificity_threshold, dtype=np.float32),
        train_question_count=np.asarray(len(train_questions), dtype=np.int64),
        # ``v`` is the existing static-loader key and now contains g_steer.
        v=bad_data["g_steer"],
    )

    print(f"\nSaved: {save_path}")
    print(f"Answer activation shape: {activations.shape}")
    print(f"Pair z_pos shape: {z_pos.shape}")
    print(f"Pair z_neg shape: {z_neg.shape}")
    print(f"Pair v_T shape: {v_t.shape}")
    print(f"Pair v_H shape: {v_h.shape}")
    print("Stage 1 complete.")


if __name__ == "__main__":
    main()
