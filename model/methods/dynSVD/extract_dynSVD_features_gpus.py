"""Extract paired representations in the format consumed by dynSVD.

Unlike the legacy feature builder, this program does not construct a bad
subspace or an additive steering vector. It only extracts answer-level hidden
states and writes truthful/hallucinated pairs for the dynSVD objective:

    R * normalize(z_neg) ~= normalize(z_pos)

This version manually shards Transformer layers across two CUDA devices.
It intentionally does NOT use device_map="auto" / accelerate dispatch, so the
model's original forward path and output_hidden_states semantics are preserved.
"""
from __future__ import annotations

import argparse
import gc
import os
import sys
from pathlib import Path
from typing import Any, Optional, Sequence, Tuple

import numpy as np
import torch
from torch import nn
from transformers import AutoModelForCausalLM, AutoTokenizer

ROOT = Path.cwd()
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from model.methods.base.stage1 import (
    HF_NAMES,
    build_question_vectors,
    get_truthfulqa_data,
)


# ---------------------------------------------------------------------------
# Manual two-GPU model parallelism
# ---------------------------------------------------------------------------

# Common Hugging Face / trust_remote_code layer-container paths.
# The script only needs to know where the repeated Transformer blocks live.
LAYER_PATH_CANDIDATES: Tuple[str, ...] = (
    "model.layers",                  # Llama / Qwen2 / Mistral / Gemma / Phi3 ...
    "transformer.h",                 # GPT2-like / some trust_remote_code models
    "transformer.layers",
    "gpt_neox.layers",               # GPT-NeoX / Pythia
    "model.decoder.layers",          # OPT-like
    "transformer.encoder.layers",
    "model.transformer.layers",
    "language_model.model.layers",   # some multimodal/custom wrappers
)

# Modules normally executed after all Transformer blocks.
# Keeping these on the second GPU avoids an unnecessary GPU1 -> GPU0 transfer
# before the final normalization/projection when the architecture is known.
POST_LAYER_PATH_CANDIDATES: Tuple[str, ...] = (
    "model.norm",
    "model.final_layernorm",
    "model.final_layer_norm",
    "model.norm_f",
    "transformer.ln_f",
    "transformer.norm",
    "gpt_neox.final_layer_norm",
    "model.decoder.final_layer_norm",
    "model.decoder.project_out",
    "language_model.model.norm",
)


def _get_module_by_path(root: nn.Module, path: str) -> Optional[nn.Module]:
    """Return a nested module by dotted path, or None if it does not exist."""
    obj: Any = root
    for part in path.split("."):
        if not hasattr(obj, part):
            return None
        obj = getattr(obj, part)
    return obj if isinstance(obj, nn.Module) else None


def _find_module_name(root: nn.Module, target: Optional[nn.Module]) -> Optional[str]:
    """Find the dotted module name for an already-resolved module object."""
    if target is None:
        return None
    for name, module in root.named_modules():
        if module is target:
            return name
    return None


def _resolve_transformer_layers(model: nn.Module) -> Tuple[str, Sequence[nn.Module]]:
    """Locate the repeated Transformer block container."""
    for path in LAYER_PATH_CANDIDATES:
        module = _get_module_by_path(model, path)
        if module is None:
            continue
        try:
            n_layers = len(module)  # ModuleList / Sequential / list-like module
        except TypeError:
            continue
        if n_layers > 0:
            return path, module  # type: ignore[return-value]

    # Helpful diagnostics for custom trust_remote_code architectures.
    modulelist_names = [
        name
        for name, module in model.named_modules()
        if isinstance(module, (nn.ModuleList, nn.Sequential)) and len(module) > 0
    ]
    raise RuntimeError(
        "Could not locate the Transformer layer container. "
        f"Tried: {LAYER_PATH_CANDIDATES}. "
        f"Detected ModuleList/Sequential modules: {modulelist_names[:30]}. "
        "Add your model's layer path to LAYER_PATH_CANDIDATES."
    )


def _is_same_or_child(name: str, prefix: str) -> bool:
    return name == prefix or name.startswith(prefix + ".")


def _move_tree_to_device(obj: Any, device: torch.device) -> Any:
    """Recursively move tensors in normal layer args/kwargs to one device.

    use_cache=False is used by this extractor, so we intentionally avoid
    mutating custom KV-cache objects from trust_remote_code models.
    """
    if torch.is_tensor(obj):
        if obj.device == device:
            return obj
        return obj.to(device, non_blocking=True)

    if isinstance(obj, tuple):
        return tuple(_move_tree_to_device(x, device) for x in obj)

    if isinstance(obj, list):
        return [_move_tree_to_device(x, device) for x in obj]

    if isinstance(obj, dict):
        return {k: _move_tree_to_device(v, device) for k, v in obj.items()}

    return obj


def _make_device_pre_hook(device: torch.device):
    """Create a forward-pre-hook that only relocates tensor inputs."""
    def hook(module: nn.Module, args: tuple, kwargs: dict):
        del module
        return (
            _move_tree_to_device(args, device),
            _move_tree_to_device(kwargs, device),
        )

    return hook


def _module_has_children(module: nn.Module) -> bool:
    return next(module.children(), None) is not None


def _module_nbytes(module: nn.Module) -> int:
    """Parameter+buffer bytes, counting each tensor storage once per module."""
    seen = set()
    total = 0
    for tensor in list(module.parameters(recurse=True)) + list(module.buffers(recurse=True)):
        key = id(tensor)
        if key in seen:
            continue
        seen.add(key)
        total += tensor.numel() * tensor.element_size()
    return total


def _format_gib(num_bytes: int) -> str:
    return f"{num_bytes / (1024 ** 3):.2f} GiB"


def shard_model_across_two_gpus(
    model: nn.Module,
    gpu0: torch.device,
    gpu1: torch.device,
    gpu0_layers: Optional[int] = None,
) -> Tuple[str, int]:
    """Manually shard Transformer blocks over two CUDA devices.

    Design goals:
      1. Do not use device_map="auto" or Accelerate dispatch hooks.
      2. Do not rewrite the model's forward / attention implementation.
      3. Preserve output_hidden_states=True behavior.
      4. Move tensors only at module boundaries where a CUDA-device change is
         actually required.

    Returns:
        (layer_container_path, number_of_layers_on_gpu0)
    """
    layer_path, layers = _resolve_transformer_layers(model)
    num_layers = len(layers)

    if num_layers < 2:
        raise RuntimeError(f"Need at least 2 Transformer layers, got {num_layers}")

    if gpu0_layers is None:
        split_idx = num_layers // 2
    else:
        split_idx = int(gpu0_layers)

    if not (1 <= split_idx < num_layers):
        raise ValueError(
            f"--gpu0_layers must be in [1, {num_layers - 1}], got {split_idx}"
        )

    input_embeddings = model.get_input_embeddings()
    output_embeddings = model.get_output_embeddings()

    if input_embeddings is None:
        raise RuntimeError("model.get_input_embeddings() returned None")

    # Detect tied input/output weights BEFORE moving anything.
    tied_input_output = False
    if output_embeddings is not None:
        in_weight = getattr(input_embeddings, "weight", None)
        out_weight = getattr(output_embeddings, "weight", None)
        tied_input_output = (
            in_weight is not None
            and out_weight is not None
            and in_weight is out_weight
        )

    input_embedding_name = _find_module_name(model, input_embeddings)
    output_embedding_name = _find_module_name(model, output_embeddings)

    print("\n=== Manual two-GPU sharding ===")
    print(f"Transformer layer path : {layer_path}")
    print(f"Total Transformer layers: {num_layers}")
    print(f"{gpu0}: layers 0 .. {split_idx - 1}")
    print(f"{gpu1}: layers {split_idx} .. {num_layers - 1}")
    print(f"Input embeddings       : {input_embedding_name or '<unknown>'} -> {gpu0}")
    print(
        "Input/output weights   : "
        + ("tied" if tied_input_output else "not tied / unavailable")
    )

    # 1) Move each complete Transformer block.
    layer0_bytes = 0
    layer1_bytes = 0
    for i, layer in enumerate(layers):
        target = gpu0 if i < split_idx else gpu1
        nbytes = _module_nbytes(layer)
        if i < split_idx:
            layer0_bytes += nbytes
        else:
            layer1_bytes += nbytes

        layer.to(target)

        # Only the top-level block gets a cross-device hook. Its internals stay
        # entirely on the same GPU, so attention/MLP computations are untouched.
        layer.register_forward_pre_hook(
            _make_device_pre_hook(target),
            with_kwargs=True,
        )

    # 2) Move non-layer leaf modules.
    #
    # Most pre-layer modules (token embeddings, position modules, etc.) live on
    # GPU0. Known post-layer modules (final norm/project_out) live on GPU1.
    # The output head also lives on GPU1 unless it shares the embedding weight,
    # in which case both must remain together on GPU0.
    #
    # We move only leaf modules here so that calling .to() never recursively
    # re-moves the Transformer blocks we just placed.
    post_layer_paths = {
        path
        for path in POST_LAYER_PATH_CANDIDATES
        if _get_module_by_path(model, path) is not None
    }

    output_device = gpu0 if tied_input_output else gpu1

    outside_gpu0_bytes = 0
    outside_gpu1_bytes = 0

    for name, module in model.named_modules():
        if not name:
            continue

        # Transformer blocks and all their descendants were already handled.
        if _is_same_or_child(name, layer_path):
            continue

        # Only move leaves to avoid recursively touching layer containers.
        if _module_has_children(module):
            continue

        target = gpu0

        # Known post-layer modules should follow the second half of the model.
        for post_path in post_layer_paths:
            if _is_same_or_child(name, post_path):
                target = gpu1
                break

        # Output head placement takes precedence.
        if output_embedding_name is not None and _is_same_or_child(name, output_embedding_name):
            target = output_device

        # Input embedding must stay on GPU0.
        if input_embedding_name is not None and _is_same_or_child(name, input_embedding_name):
            target = gpu0

        module_bytes = _module_nbytes(module)
        module.to(target)
        module.register_forward_pre_hook(
            _make_device_pre_hook(target),
            with_kwargs=True,
        )

        if target == gpu0:
            outside_gpu0_bytes += module_bytes
        else:
            outside_gpu1_bytes += module_bytes

    # Re-establish tying if the architecture exposes tie_weights().
    # This is only relevant when input/output embeddings were tied originally.
    if tied_input_output and hasattr(model, "tie_weights"):
        model.tie_weights()

    gc.collect()
    torch.cuda.empty_cache()

    print(f"Approx. Transformer bytes on {gpu0}: {_format_gib(layer0_bytes)}")
    print(f"Approx. Transformer bytes on {gpu1}: {_format_gib(layer1_bytes)}")
    print(f"Approx. non-layer leaf bytes on {gpu0}: {_format_gib(outside_gpu0_bytes)}")
    print(f"Approx. non-layer leaf bytes on {gpu1}: {_format_gib(outside_gpu1_bytes)}")
    print(
        "Output embeddings      : "
        f"{output_embedding_name or '<none>'} -> {output_device}"
    )

    # Sanity check: report any CPU parameters that remain.
    cpu_param_names = [
        name for name, param in model.named_parameters() if param.device.type == "cpu"
    ]
    if cpu_param_names:
        print(
            "WARNING: some parameters are still on CPU. "
            "First examples: "
            + ", ".join(cpu_param_names[:20])
        )
        print(
            "If forward fails because of one of these modules, add its module "
            "path to POST_LAYER_PATH_CANDIDATES or the model-specific placement."
        )
    else:
        print("All model parameters are placed on CUDA devices.")

    print("================================\n")
    return layer_path, split_idx


# ---------------------------------------------------------------------------
# Feature extraction
# ---------------------------------------------------------------------------

def extract_last_token_activations(
    model: nn.Module,
    tokenizer,
    prompts,
    layer_idx: int,
    max_length: int = 0,
) -> np.ndarray:
    """Extract one last-token activation per prompt.

    This intentionally keeps the reference definition:

        outputs.hidden_states[layer_idx + 1][0, -1, :]

    We do NOT replace this with a custom representation, steering vector, or
    Accelerate-hook output.
    """
    activations = []
    model.eval()

    input_embeddings = model.get_input_embeddings()
    if input_embeddings is None:
        raise RuntimeError("model.get_input_embeddings() returned None")

    try:
        input_device = next(input_embeddings.parameters()).device
    except StopIteration:
        input_device = torch.device("cuda:0")

    print(f"Tokenizer inputs will be placed on: {input_device}")
    print(f"Extracting outputs.hidden_states[{layer_idx + 1}]")

    from tqdm import tqdm

    with torch.inference_mode():
        for i, prompt in enumerate(tqdm(prompts, desc="Extracting features")):
            tokenizer_kwargs = {"return_tensors": "pt"}
            if max_length > 0:
                tokenizer_kwargs.update(
                    truncation=True,
                    max_length=max_length,
                )

            inputs = tokenizer(prompt, **tokenizer_kwargs)
            inputs = {
                k: v.to(input_device, non_blocking=True)
                for k, v in inputs.items()
            }

            # Important:
            # - output_hidden_states=True preserves the exact reference feature.
            # - use_cache=False avoids cross-device KV-cache complications.
            outputs = model(
                **inputs,
                output_hidden_states=True,
                use_cache=False,
                return_dict=True,
            )

            if outputs.hidden_states is None:
                raise RuntimeError(
                    "Model returned hidden_states=None even though "
                    "output_hidden_states=True."
                )

            hs_index = layer_idx + 1
            if hs_index >= len(outputs.hidden_states):
                raise IndexError(
                    f"Requested hidden_states[{hs_index}], but model returned "
                    f"only {len(outputs.hidden_states)} hidden-state tensors."
                )

            hs = outputs.hidden_states[hs_index]
            last_token_act = hs[0, -1, :].detach().float().cpu().numpy()

            # Optional debugging for non-finite values.
            # if np.isnan(last_token_act).any():
            #     print(f"NaN detected at prompt index: {i}")
            #     print("Prompt:", prompt[:300])
            #     print("Hidden-state device:", hs.device)
            #     print("NaN count:", int(np.isnan(last_token_act).sum()))
            # if not np.isfinite(last_token_act).all():
            #     print(f"Non-finite activation detected at prompt index: {i}")
            #     print("Inf count:", int(np.isinf(last_token_act).sum()))

            activations.append(last_token_act)

            # Drop references promptly; do not empty CUDA cache every sample
            # because that would significantly slow extraction.
            del outputs, hs, inputs

    activations = np.stack(activations, axis=0)

    if np.isnan(activations).any():
        print("Final activations contain NaN")
        print("Total NaN count:", int(np.isnan(activations).sum()))
        print("NaN positions:", np.argwhere(np.isnan(activations))[:20])
    else:
        print("No NaN found in activations")

    if not np.isfinite(activations).all():
        print("WARNING: final activations contain non-finite values.")
        print("Total Inf count:", int(np.isinf(activations).sum()))

    return activations


def main() -> None:
    p = argparse.ArgumentParser(
        description="Extract dynSVD paired hidden representations with manual two-GPU sharding"
    )
    p.add_argument("model_name")
    p.add_argument("--layer", type=int, default=14)
    p.add_argument("--save_dir", default="./features")
    p.add_argument("--output_name", default=None)
    p.add_argument("--model_dir", default=None)
    p.add_argument("--batch_size", type=int, default=1)
    p.add_argument("--max_length", type=int, default=0)
    p.add_argument("--device", choices=["cuda", "cpu"], default="cuda")
    p.add_argument(
        "--dtype",
        choices=["float16", "bfloat16", "float32"],
        default="float16",
    )
    p.add_argument("--train_fraction", type=float, default=0.8)
    p.add_argument(
        "--split_seed",
        type=int,
        default=42,
        help="Kept for compatibility; question K-fold ordering is deterministic and unsuffled.",
    )
    p.add_argument("--num_folds", type=int, default=2)

    # New two-GPU options.
    p.add_argument(
        "--gpu0",
        type=int,
        default=0,
        help="CUDA index for the first half / embeddings (default: 0).",
    )
    p.add_argument(
        "--gpu1",
        type=int,
        default=1,
        help="CUDA index for the second half (default: 1).",
    )
    p.add_argument(
        "--gpu0_layers",
        type=int,
        default=None,
        help=(
            "Number of Transformer layers placed on the first GPU. "
            "Default: half of all layers. Reduce this if GPU0 has less free "
            "memory because embeddings/tied lm_head also live there."
        ),
    )

    args = p.parse_args()

    if args.layer < 0 or args.batch_size <= 0 or args.max_length < 0:
        raise ValueError(
            "layer must be nonnegative, batch_size positive, max_length nonnegative"
        )
    if args.num_folds < 2:
        raise ValueError("num_folds must be at least 2")

    if args.device != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("This reference-compatible extractor requires CUDA")

    if args.gpu0 == args.gpu1:
        raise ValueError("--gpu0 and --gpu1 must refer to different CUDA devices")

    available_gpus = torch.cuda.device_count()
    if args.gpu0 >= available_gpus or args.gpu1 >= available_gpus:
        raise RuntimeError(
            f"Requested cuda:{args.gpu0} and cuda:{args.gpu1}, "
            f"but torch sees only {available_gpus} CUDA device(s)."
        )

    gpu0 = torch.device(f"cuda:{args.gpu0}")
    gpu1 = torch.device(f"cuda:{args.gpu1}")

    dtype_map = {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }
    torch_dtype = dtype_map[args.dtype]

    model_path = args.model_dir or HF_NAMES.get(args.model_name, args.model_name)

    print(
        f"Loading model on CPU first: {model_path} | "
        f"dtype={torch_dtype} | then manually shard to {gpu0}, {gpu1}"
    )

    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        trust_remote_code=True,
    )

    if tokenizer.pad_token_id is None:
        if tokenizer.eos_token_id is None:
            raise ValueError("tokenizer has neither pad_token nor eos_token")
        tokenizer.pad_token = tokenizer.eos_token

    # IMPORTANT:
    # No .to("cuda") here.
    # No device_map="auto" here.
    # Load on CPU first, then move complete Transformer blocks manually.
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        low_cpu_mem_usage=True,
        torch_dtype=torch_dtype,
        trust_remote_code=True,
    )

    layer_path, split_idx = shard_model_across_two_gpus(
        model=model,
        gpu0=gpu0,
        gpu1=gpu1,
        gpu0_layers=args.gpu0_layers,
    )

    _, layers = _resolve_transformer_layers(model)
    num_layers = len(layers)

    if args.layer >= num_layers:
        raise ValueError(
            f"--layer={args.layer} is out of range for this model; "
            f"detected {num_layers} Transformer layers (valid: 0..{num_layers - 1})."
        )

    target_layer_device = gpu0 if args.layer < split_idx else gpu1
    print(
        f"Target feature layer: {args.layer} "
        f"(outputs.hidden_states[{args.layer + 1}]) "
        f"should reside on {target_layer_device}"
    )

    if args.batch_size != 1:
        print(
            "NOTE: --batch_size is kept for command-line compatibility, but "
            "this extractor intentionally processes one prompt at a time to "
            "preserve the reference last-token semantics exactly."
        )

    model.eval()

    data = get_truthfulqa_data(style="standard")

    activations = extract_last_token_activations(
        model=model,
        tokenizer=tokenizer,
        prompts=data["prompts"],
        layer_idx=args.layer,
        max_length=args.max_length,
    )

    all_questions = np.unique(data["q_indices"])

    # Match Spherical-Steering exactly: deterministic K-fold over complete
    # questions (equivalent to KFold(shuffle=False)); answer rows never split.
    folds = np.array_split(all_questions, args.num_folds)

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
        activations,
        data["labels"],
        data["q_indices"],
        data["choice_indices"],
    )

    # dynSVD is trained on equal-length unit representations. Keep raw pair
    # states for traceability, while exposing normalized z_neg/z_pos as the
    # canonical fields consumed by the trainer.
    z_neg_raw, z_pos_raw = z_neg, z_pos
    z_neg = z_neg_raw / np.maximum(
        np.linalg.norm(z_neg_raw, axis=1, keepdims=True),
        1e-12,
    )
    z_pos = z_pos_raw / np.maximum(
        np.linalg.norm(z_pos_raw, axis=1, keepdims=True),
        1e-12,
    )

    os.makedirs(args.save_dir, exist_ok=True)

    safe = args.model_name.replace("/", "_").replace("\\", "_")
    name = (
        args.output_name
        or f"{safe}_layer{args.layer}_dynSVD_pairs.npz"
    )
    path = os.path.join(args.save_dir, name)

    np.savez_compressed(
        path,
        activations=activations,
        labels=data["labels"],
        q_indices=data["q_indices"],
        choice_indices=data["choice_indices"],
        prompts=data["prompts"],
        choices=data["choices"],
        questions=data["questions"],
        question_offsets=data["question_offsets"],
        split_seed=np.asarray(args.split_seed, dtype=np.int64),
        vector_q_indices=vector_q_indices,
        positive_sample_indices=positive_sample_indices,
        negative_sample_indices=negative_sample_indices,
        positive_choice_indices=positive_choice_indices,
        negative_choice_indices=negative_choice_indices,
        vector_question_offsets=vector_question_offsets,
        z_pos=z_pos,
        z_neg=z_neg,
        z_minus=z_neg,
        z_plus=z_pos,
        z_pos_raw=z_pos_raw,
        z_neg_raw=z_neg_raw,
        v_T=v_t,
        v_H=v_h,
        positive_counts=positive_counts,
        negative_counts=negative_counts,
        pair_counts=pair_counts,
        layer=np.asarray(args.layer, dtype=np.int64),
        model_name=np.asarray(args.model_name),
        model_path=np.asarray(model_path),
        dataset_name=np.asarray("truthful_qa"),
        dataset_config=np.asarray("multiple_choice"),
        target_name=np.asarray("mc2_targets"),
        method=np.asarray("dynSVD paired feature extraction"),
        vector_definition=np.asarray(
            "z_neg -> z_pos pairs; no additive vector or bad-subspace removal"
        ),
        manual_two_gpu=np.asarray(True),
        layer_container_path=np.asarray(layer_path),
        gpu0=np.asarray(str(gpu0)),
        gpu1=np.asarray(str(gpu1)),
        gpu0_layers=np.asarray(split_idx, dtype=np.int64),
    )

    print(f"Saved {path}")

    # Also materialize one fold artifact per question-level split. These are
    # the files consumed by train_router.py, mirroring the reference project.
    for fold_idx, test_questions in enumerate(folds):
        train_questions = np.concatenate(
            [folds[i] for i in range(args.num_folds) if i != fold_idx]
        ).astype(np.int64)

        fold_path = os.path.join(
            args.save_dir,
            f"{safe}_layer{args.layer}_dynSVD_pairs_fold{fold_idx}.npz",
        )

        np.savez_compressed(
            fold_path,
            activations=activations,
            labels=data["labels"],
            q_indices=data["q_indices"],
            choice_indices=data["choice_indices"],
            prompts=data["prompts"],
            choices=data["choices"],
            questions=data["questions"],
            question_offsets=data["question_offsets"],
            train_q_indices=train_questions,
            test_q_indices=np.asarray(test_questions, dtype=np.int64),
            split_seed=np.asarray(args.split_seed, dtype=np.int64),
            fold_idx=np.asarray(fold_idx, dtype=np.int64),
            vector_q_indices=vector_q_indices,
            positive_sample_indices=positive_sample_indices,
            negative_sample_indices=negative_sample_indices,
            positive_choice_indices=positive_choice_indices,
            negative_choice_indices=negative_choice_indices,
            vector_question_offsets=vector_question_offsets,
            z_pos=z_pos,
            z_neg=z_neg,
            z_minus=z_neg,
            z_plus=z_pos,
            z_pos_raw=z_pos_raw,
            z_neg_raw=z_neg_raw,
            v_T=v_t,
            v_H=v_h,
            positive_counts=positive_counts,
            negative_counts=negative_counts,
            pair_counts=pair_counts,
            layer=np.asarray(args.layer, dtype=np.int64),
            model_name=np.asarray(args.model_name),
            model_path=np.asarray(model_path),
            dataset_name=np.asarray("truthful_qa"),
            dataset_config=np.asarray("multiple_choice"),
            target_name=np.asarray("mc2_targets"),
            method=np.asarray("dynSVD question-level fold feature"),
            manual_two_gpu=np.asarray(True),
            layer_container_path=np.asarray(layer_path),
            gpu0=np.asarray(str(gpu0)),
            gpu1=np.asarray(str(gpu1)),
            gpu0_layers=np.asarray(split_idx, dtype=np.int64),
        )

        print(
            f"Fold {fold_idx + 1}/{args.num_folds}: "
            f"train={len(train_questions)} questions, "
            f"test/val={len(test_questions)} questions -> {fold_path}"
        )

    print(
        f"answer activations={activations.shape}; "
        f"unit pairs={z_neg.shape}"
    )


if __name__ == "__main__":
    main()
