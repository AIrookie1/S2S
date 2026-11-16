"""
Utility functions for generic MC task evaluation with Spherical Steering.

Supports: COPA, StoryCloze, MMLU (global), Winogrande, BoolQ

Training prompts are built from the same dataset-specific templates used by
``evaluate_generic.py``.
"""

import os
import torch
import numpy as np
import random
from baukit import TraceDict
from datasets import load_dataset, concatenate_datasets, Dataset
from tqdm import tqdm


# ==================== Random Seeds ====================

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ==================== Model Paths ====================

HF_NAMES = {
    'llama3.1-8B': 'meta-llama/Llama-3.1-8B',
    'llama3.1-8B-Instruct': 'meta-llama/Llama-3.1-8B-Instruct',
    'Qwen2.5-7B-Instruct': 'Qwen/Qwen2.5-7B-Instruct',
}


# ==================== MMLU Categories ====================

MMLU_CATEGORIES = {
    "STEM": [
        "abstract_algebra", "astronomy", "college_biology", "college_chemistry",
        "college_computer_science", "college_mathematics", "college_physics",
        "computer_security", "conceptual_physics", "electrical_engineering",
        "elementary_mathematics", "high_school_biology", "high_school_chemistry",
        "high_school_computer_science", "high_school_mathematics", "high_school_physics",
        "high_school_statistics", "machine_learning"
    ],
    "Humanities": [
        "formal_logic", "high_school_european_history", "high_school_us_history",
        "high_school_world_history", "international_law", "jurisprudence",
        "logical_fallacies", "moral_disputes", "moral_scenarios", "philosophy",
        "prehistory", "professional_law", "world_religions"
    ],
    "Social_Sciences": [
        "econometrics", "high_school_geography", "high_school_government_and_politics",
        "high_school_macroeconomics", "high_school_microeconomics", "high_school_psychology",
        "human_sexuality", "professional_psychology", "public_relations",
        "security_studies", "sociology", "us_foreign_policy"
    ],
    "Other": [
        "anatomy", "business_ethics", "clinical_knowledge", "college_medicine",
        "global_facts", "human_aging", "management", "marketing", "medical_genetics",
        "miscellaneous", "nutrition", "professional_accounting", "professional_medicine",
        "virology"
    ]
}


# ==================== Prompt Formatting ====================

from method.S2S_R.generic.prompts_generic import (
    append_option,
    format_boolq_prompt,
    format_copa_prompt,
    format_mmlu_prompt,
    format_storycloze_prompt,
    format_winogrande_prompt,
)

# ============================================================
# Data Loaders for Prototype Extraction (Training)
# Returns: prompts, labels, q_indices
# ============================================================

def get_copa_data(seed=42, tokenizer=None):
    """
    Load COPA (super_glue) train split (400 samples).
    For each item: correct choice → label=1, incorrect → label=0.
    """
    print(f"Loading COPA for prototype extraction...")
    dataset = load_dataset("super_glue", "copa", split="train")
    dataset = dataset.shuffle(seed=seed)

    prompts, labels, q_indices = [], [], []

    for i, item in enumerate(dataset):
        premise = item["premise"]
        q_type = item["question"]
        choices = [item["choice1"], item["choice2"]]
        correct_label = item["label"]  # 0 or 1
        base = format_copa_prompt(premise, q_type, *choices)

        # Correct
        prompts.append(append_option(base, choices[correct_label]))
        labels.append(1)
        q_indices.append(i)
        # Incorrect
        prompts.append(append_option(base, choices[1 - correct_label]))
        labels.append(0)
        q_indices.append(i)

    print(f"  Total: {len(prompts)} samples from {len(dataset)} questions")
    return prompts, np.array(labels), np.array(q_indices)


def get_storycloze_data(split='train', num_samples=None, tokenizer=None):
    """
    Load XStoryCloze (English).
    For each story: correct ending → label=1, incorrect → label=0.
    """
    print(f"Loading XStoryCloze (en, split={split})...")
    hf_split = 'train' if split == 'train' else 'eval'
    dataset = load_dataset("juletxara/xstory_cloze", "en")[hf_split]

    if num_samples is not None:
        dataset = dataset.select(range(min(num_samples, len(dataset))))

    prompts, labels, q_indices = [], [], []

    for i, item in enumerate(dataset):
        story = [
            item['input_sentence_1'], item['input_sentence_2'],
            item['input_sentence_3'], item['input_sentence_4']
        ]
        opt1 = item['sentence_quiz1']
        opt2 = item['sentence_quiz2']
        correct_idx = item['answer_right_ending']  # 1 or 2

        if correct_idx == 1:
            correct_text, incorrect_text = opt1, opt2
        else:
            correct_text, incorrect_text = opt2, opt1

        base = format_storycloze_prompt(story, opt1, opt2)
        prompts.append(append_option(base, correct_text))
        labels.append(1)
        q_indices.append(i)
        prompts.append(append_option(base, incorrect_text))
        labels.append(0)
        q_indices.append(i)

    print(f"  Total: {len(prompts)} samples from {len(dataset)} questions")
    return prompts, np.array(labels), np.array(q_indices)


def get_mmlu_global_data(split='train', seed=42, tokenizer=None):
    """
    Load MMLU Global Balanced Dataset (all 4 categories combined).

    Split strategy (per category):
      - train: indices 0-500   (500/cat, 2000 total)
      - test:  indices 500-1000 (500/cat, 2000 total)

    Creates a single balanced dataset for computing ONE global prototype
    and ONE micro-average accuracy across all of MMLU.
    """
    if tokenizer is None:
        raise ValueError("MMLU prompt construction requires a tokenizer")
    print(f"Loading MMLU Global Balanced (split={split})...")

    all_prompts, all_labels, all_q_indices = [], [], []
    global_q_idx = 0

    for cat_name, subsets in MMLU_CATEGORIES.items():
        loaded = []
        for sub in subsets:
            try:
                local_root = os.environ.get(
                    "MMLU_CACHE_DIR", "data/mmlu"
                )
                # ``dev`` has only five rows per subject.  Use the full test
                # corpus as the deterministic source pool for the documented
                # 500/500 per-category train/test split.
                local_file = os.path.join(
                    local_root, sub, "test-00000-of-00001.parquet"
                )
                loaded.append(Dataset.from_parquet(local_file))
            except Exception as e:
                print(f"  Warning: Failed to load {sub}: {e}")
        if not loaded:
            continue

        cat_dataset = concatenate_datasets(loaded).shuffle(seed=seed)
        total = len(cat_dataset)

        if split == 'train':
            start_idx, end_idx = 0, min(500, total)
        elif split == 'test':
            start_idx, end_idx = 500, min(1000, total)
        else:
            raise ValueError(f"Unknown split: {split}. Use 'train' or 'test'.")

        if start_idx >= total:
            continue

        dataset = cat_dataset.select(range(start_idx, end_idx))
        print(f"  {cat_name}: {len(dataset)} questions")

        for item in dataset:
            question = item['question']
            choices = item['choices']
            correct_idx = item['answer']

            base = format_mmlu_prompt(tokenizer, question, choices)
            all_prompts.append(append_option(base, choices[correct_idx]))
            all_labels.append(1)
            all_q_indices.append(global_q_idx)
            for j, c in enumerate(choices):
                if j != correct_idx:
                    all_prompts.append(append_option(base, c))
                    all_labels.append(0)
                    all_q_indices.append(global_q_idx)
            global_q_idx += 1

    print(f"  Total: {len(all_prompts)} samples from {global_q_idx} questions")
    return all_prompts, np.array(all_labels), np.array(all_q_indices)


def get_winogrande_data(split='train', num_samples=None, seed=42, tokenizer=None):
    """
    Load Winogrande (winogrande_xl).
    For each item: correct option → label=1, incorrect → label=0.
    """
    print(f"Loading Winogrande (split={split})...")
    hf_split = 'train' if split == 'train' else 'validation'
    dataset = load_dataset("winogrande", "winogrande_xl")[hf_split]

    if split == 'train':
        dataset = dataset.shuffle(seed=seed)

    # Default to 2000 train samples if not specified
    if num_samples is None and split == 'train':
        num_samples = 2000
        print("  Using default num_samples=2000 for Winogrande train split")

    if num_samples is not None:
        dataset = dataset.select(range(min(num_samples, len(dataset))))

    prompts, labels, q_indices = [], [], []
    for i, item in enumerate(dataset):
        sentence = item['sentence']
        opt1 = item['option1']
        opt2 = item['option2']
        correct_idx = item['answer']  # '1' or '2'

        if correct_idx == '1':
            correct_text, incorrect_text = opt1, opt2
        else:
            correct_text, incorrect_text = opt2, opt1

        base = format_winogrande_prompt(sentence, opt1, opt2)
        prompts.append(append_option(base, correct_text))
        labels.append(1)
        q_indices.append(i)
        prompts.append(append_option(base, incorrect_text))
        labels.append(0)
        q_indices.append(i)

    print(f"  Total: {len(prompts)} samples from {len(dataset)} questions")
    return prompts, np.array(labels), np.array(q_indices)


def get_boolq_data(num_questions=2000, seed=42, tokenizer=None):
    """
    Load BoolQ (super_glue) train split.
    Options: ["no", "yes"], label: 0→no, 1→yes.
    """
    print(f"Loading BoolQ for prototype extraction...")
    dataset = load_dataset("super_glue", "boolq", split="train")
    dataset = dataset.shuffle(seed=seed)

    if num_questions is not None:
        dataset = dataset.select(range(min(num_questions, len(dataset))))

    options = ["no", "yes"]
    prompts, labels, q_indices = [], [], []

    for i, item in enumerate(dataset):
        passage = item["passage"]
        question = item["question"]
        label = int(item["label"])  # 0 or 1

        base = format_boolq_prompt(passage, question)

        prompts.append(base + " " + options[label])
        labels.append(1)
        q_indices.append(i)
        prompts.append(base + " " + options[1 - label])
        labels.append(0)
        q_indices.append(i)

    print(f"  Total: {len(prompts)} samples from {len(dataset)} questions")
    return prompts, np.array(labels), np.array(q_indices)


# ==================== Unified Data Loader ====================

def get_dataset_data(
    dataset_name, split='train', num_samples=None, seed=42, tokenizer=None
):
    """Unified data loader interface."""
    if dataset_name == 'copa':
        return get_copa_data(seed=seed, tokenizer=tokenizer)
    elif dataset_name == 'storycloze':
        return get_storycloze_data(split, num_samples, tokenizer=tokenizer)
    elif dataset_name == 'mmlu_global':
        return get_mmlu_global_data(split, seed, tokenizer=tokenizer)
    elif dataset_name == 'winogrande':
        return get_winogrande_data(split, num_samples, seed, tokenizer=tokenizer)
    elif dataset_name == 'boolq':
        return get_boolq_data(
            num_questions=num_samples or 2000, seed=seed, tokenizer=tokenizer
        )
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}. "
                         f"Choose from: copa, storycloze, mmlu_global, winogrande, boolq")


# ==================== Feature Extraction ====================

def get_layer_activations(model, tokenizer, prompts, layer_idx, device):
    """
    Extract last-token activations from a specific layer using baukit.
    Prompts are processed in small padded batches.  For each row we select
    the last non-padding token, which is equivalent to the original
    single-prompt implementation but avoids thousands of separate forwards.
    """
    layer_name = f"model.layers.{layer_idx}"
    activations = []
    batch_size = max(1, int(os.environ.get("EXTRACT_BATCH_SIZE", "8")))

    print(f"Extracting activations from {layer_name}...")
    model.eval()

    with torch.no_grad():
        for start in tqdm(range(0, len(prompts), batch_size), desc="Extracting"):
            batch_prompts = prompts[start : start + batch_size]
            # Evaluation tokenizes the base prompt and option without adding
            # another BOS token. Do the same here, especially for MMLU where
            # the chat template already contains its own special tokens.
            inputs = tokenizer(
                batch_prompts,
                return_tensors='pt',
                padding=True,
                add_special_tokens=False,
            ).to(device)

            with TraceDict(model, [layer_name]) as ret:
                model(**inputs)

            layer_output = ret[layer_name].output
            if isinstance(layer_output, tuple):
                layer_output = layer_output[0]

            if isinstance(layer_output, tuple):
                layer_output = layer_output[0]
            mask = inputs["attention_mask"]
            last_indices = mask.sum(dim=1).to(dtype=torch.long) - 1
            row_indices = torch.arange(layer_output.shape[0], device=layer_output.device)
            last_token_act = layer_output[row_indices, last_indices, :].cpu().numpy()
            activations.extend(last_token_act)

    return np.array(activations)
