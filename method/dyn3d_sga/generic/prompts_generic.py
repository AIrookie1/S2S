"""Shared train/evaluation prompts for the generic MC benchmarks."""

from __future__ import annotations


def format_copa_prompt(premise, question_type, choice1, choice2):
    return (
        f"Question:\n{premise} Based on the previous passage, "
        f"choose the most reasonable {question_type}.\n"
        f"A: {choice1}\nB: {choice2}\n\nAnswer:"
    )


def format_storycloze_prompt(sentences, opt1, opt2):
    story = " ".join(sentences) if isinstance(sentences, list) else sentences
    return (
        f"{story}\n\nQuestion: Which ending makes more sense?\n"
        f"A. {opt1}\nB. {opt2}\nAnswer:"
    )


def format_mmlu_prompt(tokenizer, question, choices):
    user_content = (
        "You are solving a multiple-choice question.\n\n"
        "Choose the correct option AND output the option text EXACTLY as written.\n"
        "Rules:\n"
        "- Output must be a verbatim copy of ONE option's text.\n"
        "- Do NOT output the option letter (A/B/C/D).\n"
        "- Do NOT add explanations, punctuation, or extra words.\n"
        "- Do NOT paraphrase. Copy the option text exactly.\n\n"
        f"Question: {question}\n"
        "Options:\n"
        f"A) {choices[0]}\nB) {choices[1]}\n"
        f"C) {choices[2]}\nD) {choices[3]}\n\n"
        "Final answer (verbatim option text only):"
    )
    messages = [{"role": "user", "content": user_content}]
    return tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )


def format_winogrande_prompt(sentence, option1, option2):
    return (
        f"Q: {sentence}\n"
        f"Which option correctly fills the blank?\n"
        f"1) {option1}\n2) {option2}\nA:"
    )


def format_boolq_prompt(passage, question):
    return f"Passage: {passage}\nQuestion: {question}\nA:"


def append_option(base_prompt, option_text):
    """Use the exact option boundary shared by feature extraction and scoring."""
    prefix = " " if base_prompt.strip().endswith(":") else ""
    return base_prompt + prefix + str(option_text)
