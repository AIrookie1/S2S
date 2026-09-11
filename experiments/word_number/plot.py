from __future__ import annotations

import argparse
import difflib
import json
import math
import re
from collections import Counter, defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import FuncFormatter

# If plot_common.py is in the same directory, reuse its common setup first.
# The local settings below are then applied on top, so this file remains
# fully controllable from one place.
try:
    from plot_common import setup_plot as setup_common_plot
except ImportError:
    setup_common_plot = None


# ============================================================
# Plot style configuration
# Change figure appearance here only.
# ============================================================

# ---- Font ----
FONT_FAMILY = "DejaVu Sans"
FONT_SIZE = 18.5
AXES_TITLE_SIZE = 20
AXES_LABEL_SIZE = 18
XTICK_LABEL_SIZE = 17.4
YTICK_LABEL_SIZE = 18
LEGEND_FONT_SIZE = 18
SUPTITLE_FONT_SIZE = 15.5
META_TEXT_FONT_SIZE = 18
SUMMARY_TEXT_FONT_SIZE = 17.8
VALUE_LABEL_FONT_SIZE = 17

# ---- Figure size ----
TWO_PANEL_FIGSIZE = (18, 6.9)
SINGLE_PANEL_FIGSIZE = (9, 5.6)

# ---- Colors ----
DELETE_COLOR = "#C55A4A"
ADD_COLOR = "#4778A8"
CENTER_LINE_COLOR = "#333333"
VALUE_TEXT_COLOR = "#444444"
SECONDARY_TEXT_COLOR = "#555555"

# ---- Line / spine / grid ----
AXES_LINEWIDTH = 1.75
CENTER_LINEWIDTH = 1.8
STEM_LINEWIDTH = 2.0
GRID_LINEWIDTH = 1.45
GRID_ALPHA = 0.13
TICK_WIDTH = 0.7
TICK_LENGTH = 3

# ---- Bar / marker ----
BAR_HEIGHT = 0.46
BAR_ALPHA = 0.82
STEM_ALPHA = 0.95
END_MARKER_SIZE = 18

# ---- Layout ----
XLIM_SCALE = 1.25
VALUE_LABEL_PAD_RATIO = 0.025
PANEL_TITLE_PAD = 8
YTICK_PAD = 5
TWO_PANEL_LEFT = 0.10
TWO_PANEL_RIGHT = 0.985
TWO_PANEL_BOTTOM = 0.10
TWO_PANEL_TOP = 0.88
TWO_PANEL_WSPACE = 0.26

# ---- Figure-level text / legend positions ----
SUPTITLE_X = 0.06
SUPTITLE_Y = 0.985
META_TEXT_X = 0.06
META_TEXT_Y = 0.952
SUMMARY_TEXT_Y = 0.018
FIG_LEGEND_X = 0.985
FIG_LEGEND_Y = 0.988
SINGLE_LEGEND_Y = 1.02

# ---- Saving ----
SAVE_DPI = 600
PDF_FONT_TYPE = 42
PS_FONT_TYPE = 42


def setup_plot_style():
    """
    Apply all plotting style settings.

    If plot_common.setup_plot() is available, call it first.
    Local settings in this file override plot_common where needed.
    """
    if setup_common_plot is not None:
        setup_common_plot()

    plt.rcParams.update({
        "font.family": FONT_FAMILY,
        "font.size": FONT_SIZE,
        "axes.titlesize": AXES_TITLE_SIZE,
        "axes.labelsize": AXES_LABEL_SIZE,
        "xtick.labelsize": XTICK_LABEL_SIZE,
        "ytick.labelsize": YTICK_LABEL_SIZE,
        "legend.fontsize": LEGEND_FONT_SIZE,
        "axes.linewidth": AXES_LINEWIDTH,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "pdf.fonttype": PDF_FONT_TYPE,
        "ps.fonttype": PS_FONT_TYPE,
    })


# ============================================================
# Default configuration
# ============================================================

DEFAULT_JSON_PATH = Path(
    "/root/pinxuanli/ICLR2026/Spherical-Steering-DynSVD/"
    "results/truthfulqa/dyn3d_sga/llama3.1-8B-Instruct/llm_judge/"
    "llama3.1-8B-Instruct_l14_fold0_llm_judge_3d_fixed_router.json"
)

DEFAULT_OUT_DIR = Path(
    "/root/pinxuanli/ICLR2026/Spherical-Steering-DynSVD/"
    "results/experiment/word_number"
)

TOPK = 5

# Top-K figure filtering only
REMOVE_STOPWORDS_FROM_TOPK = True
REMOVE_SINGLE_CHAR_FROM_TOPK = True
REMOVE_NUMERIC_FROM_TOPK = True


# ============================================================
# Topic-specificity classification
# ============================================================

# Prefer a dataset-provided category/domain/topic label when available.
TOPIC_FIELD_CANDIDATES = (
    "category",
    "domain",
    "topic",
    "question_category",
)

# Used only when no explicit topic/category field is available.
DEFAULT_N_TOPIC_CLUSTERS = 12
DEFAULT_EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

# Bayesian shrinkage:
# Low-frequency words are pulled toward the corpus-level topic distribution,
# preventing a word seen once from looking maximally domain-specific.
TOPIC_PRIOR_STRENGTH = 8.0

# Reliability shrinkage:
# specificity *= df / (df + tau)
TOPIC_RELIABILITY_TAU = 3.0

# A data-relative split:
# words at/above this quantile of topic-specificity among edited content words
# are labeled domain-specific.
DOMAIN_SPECIFICITY_QUANTILE = 0.70


GENERAL_FORCE = {
    # "however",
}

DOMAIN_FORCE = {
    # "contrails",
}


# ============================================================
# Tokenization
# ============================================================

WORD_RE = re.compile(
    r"[A-Za-z]+(?:['’][A-Za-z]+)?|\d+(?:\.\d+)?"
)


def tokenize(text):
    """Lowercase English word-level tokenization."""
    if not text:
        return []

    return [
        m.group(0).lower().replace("’", "'")
        for m in WORD_RE.finditer(text)
    ]


# ============================================================
# Stopwords
# ============================================================

STOPWORDS = set("""
a an the and or but if while
of to in on at by for from with as
is am are was were be been being
i you he she it we they
me him her us them
my your his its our their
mine yours hers ours theirs
this that these those
there here
have has had
do does did
not no yes
can could may might must
shall should will would
than then so such very
more most some any all
each every both either neither
what which who whom whose
when where why how
""".split())


# ============================================================
# Pairwise sequence diff
# ============================================================

def diff_answers(unsteered_answer, steered_answer):
    """
    Compare one unsteered answer with one steered answer.

    Returns
    -------
    added : list[str]
        Tokens introduced in the steered answer.

    deleted : list[str]
        Tokens removed from the unsteered answer.

    Replacement is treated as:
        old tokens -> deleted
        new tokens -> added
    """
    old_tokens = tokenize(unsteered_answer)
    new_tokens = tokenize(steered_answer)

    matcher = difflib.SequenceMatcher(
        a=old_tokens,
        b=new_tokens,
        autojunk=False,
    )

    added = []
    deleted = []

    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag == "equal":
            continue

        if tag == "delete":
            deleted.extend(old_tokens[i1:i2])

        elif tag == "insert":
            added.extend(new_tokens[j1:j2])

        elif tag == "replace":
            deleted.extend(old_tokens[i1:i2])
            added.extend(new_tokens[j1:j2])

    return added, deleted


# ============================================================
# Analyse JSON
# ============================================================

def load_json_results(json_path):
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    if "results" not in data:
        raise KeyError(
            f'Input JSON does not contain a top-level "results" field: '
            f"{json_path}"
        )

    return data, data["results"]


def analyse_json(json_path):
    _, results = load_json_results(json_path)

    added_counter = Counter()
    deleted_counter = Counter()

    unsteered_freq = Counter()
    steered_freq = Counter()

    document_frequency = Counter()

    changed_samples = 0
    identical_samples = 0

    sample_rows = []
    detailed_diffs = []

    for item in results:
        question_index = item.get("question_index")
        question = item.get("question", "")

        unsteered_answer = (
            item.get("unsteered", {}).get("answer", "") or ""
        )
        steered_answer = (
            item.get("steered", {}).get("answer", "") or ""
        )

        old_tokens = tokenize(unsteered_answer)
        new_tokens = tokenize(steered_answer)

        unsteered_freq.update(old_tokens)
        steered_freq.update(new_tokens)

        words_in_question = set(old_tokens) | set(new_tokens)
        for word in words_in_question:
            document_frequency[word] += 1

        added, deleted = diff_answers(
            unsteered_answer,
            steered_answer,
        )

        changed = bool(added or deleted)

        if changed:
            changed_samples += 1
        else:
            identical_samples += 1

        added_counter.update(added)
        deleted_counter.update(deleted)

        sample_rows.append({
            "question_index": question_index,
            "changed": changed,
            "n_added": len(added),
            "n_deleted": len(deleted),
            "net_token_change": len(added) - len(deleted),
            "unsteered_length": len(old_tokens),
            "steered_length": len(new_tokens),
        })

        detailed_diffs.append({
            "question_index": question_index,
            "question": question,
            "added_tokens": added,
            "deleted_tokens": deleted,
            "unsteered_answer": unsteered_answer,
            "steered_answer": steered_answer,
        })

    all_words = (
        set(unsteered_freq)
        | set(steered_freq)
        | set(added_counter)
        | set(deleted_counter)
    )

    total_samples = len(results)
    word_rows = []

    for word in all_words:
        u = unsteered_freq[word]
        s = steered_freq[word]
        add_n = added_counter[word]
        del_n = deleted_counter[word]
        df = document_frequency[word]

        word_rows.append({
            "word": word,
            "unsteered_count": u,
            "steered_count": s,
            "added": add_n,
            "deleted": del_n,
            "net_frequency_change": s - u,
            "edit_activity": add_n + del_n,
            "document_frequency": df,
            "document_frequency_ratio": (
                df / total_samples if total_samples else 0.0
            ),
        })

    word_df = pd.DataFrame(word_rows)

    if not word_df.empty:
        word_df = word_df.sort_values(
            ["edit_activity", "word"],
            ascending=[False, True],
        ).reset_index(drop=True)

    sample_df = pd.DataFrame(sample_rows)

    summary = {
        "total_samples": total_samples,
        "changed_samples": changed_samples,
        "identical_samples": identical_samples,
        "changed_ratio": (
            changed_samples / total_samples
            if total_samples else 0.0
        ),
        "total_added": sum(added_counter.values()),
        "total_deleted": sum(deleted_counter.values()),
        "net_length_change": (
            sum(added_counter.values())
            - sum(deleted_counter.values())
        ),
        "unique_added_words": len(added_counter),
        "unique_deleted_words": len(deleted_counter),
    }

    return (
        word_df,
        sample_df,
        detailed_diffs,
        summary,
    )


# ============================================================
# Topic inference
# ============================================================

def find_explicit_topic_field(results):
    """
    Return the first candidate field that is present and non-empty
    for every sample. Otherwise return None.
    """
    if not results:
        return None

    for key in TOPIC_FIELD_CANDIDATES:
        values = [item.get(key) for item in results]
        if all(
            value is not None and str(value).strip() != ""
            for value in values
        ):
            return key

    return None


def infer_topics_from_dataset(results):
    """
    Use an explicit category/domain/topic field already stored in the JSON.
    """
    key = find_explicit_topic_field(results)

    if key is None:
        raise ValueError(
            "No complete dataset topic/category field was found. "
            f"Tried: {', '.join(TOPIC_FIELD_CANDIDATES)}"
        )

    labels = [
        str(item[key]).strip()
        for item in results
    ]

    return labels, f"dataset field: {key}"


def infer_topics_with_embeddings(
    results,
    n_topics=DEFAULT_N_TOPIC_CLUSTERS,
    embedding_model=DEFAULT_EMBEDDING_MODEL,
):
    """
    Cluster questions using sentence embeddings + KMeans.

    This path may require a locally cached sentence-transformers model or
    internet access on first use.
    """
    try:
        from sentence_transformers import SentenceTransformer
        from sklearn.cluster import KMeans
    except ImportError as exc:
        raise ImportError(
            "Embedding topic inference requires:\n"
            "  pip install sentence-transformers scikit-learn"
        ) from exc

    questions = [
        item.get("question", "") or ""
        for item in results
    ]

    n_samples = len(questions)

    if n_samples == 0:
        return [], "semantic embeddings: no samples"

    if n_samples == 1:
        return ["topic_00"], "semantic embeddings: single topic"

    n_topics = max(
        2,
        min(int(n_topics), n_samples),
    )

    model = SentenceTransformer(
        embedding_model
    )

    embeddings = model.encode(
        questions,
        normalize_embeddings=True,
        show_progress_bar=True,
    )

    clusterer = KMeans(
        n_clusters=n_topics,
        random_state=42,
        n_init=20,
    )

    cluster_ids = clusterer.fit_predict(
        embeddings
    )

    labels = [
        f"topic_{cluster_id:02d}"
        for cluster_id in cluster_ids
    ]

    return (
        labels,
        f"semantic embeddings + KMeans, K={n_topics}",
    )


def infer_topics_with_tfidf(
    results,
    n_topics=DEFAULT_N_TOPIC_CLUSTERS,
):
    """
    Offline fallback:
    cluster questions using TF-IDF + KMeans.
    """
    try:
        from sklearn.cluster import KMeans
        from sklearn.feature_extraction.text import TfidfVectorizer
    except ImportError as exc:
        raise ImportError(
            "TF-IDF topic inference requires:\n"
            "  pip install scikit-learn"
        ) from exc

    questions = [
        item.get("question", "") or ""
        for item in results
    ]

    n_samples = len(questions)

    if n_samples == 0:
        return [], "TF-IDF: no samples"

    if n_samples == 1:
        return ["topic_00"], "TF-IDF: single topic"

    n_topics = max(
        2,
        min(int(n_topics), n_samples),
    )

    vectorizer = TfidfVectorizer(
        lowercase=True,
        stop_words="english",
        ngram_range=(1, 2),
        min_df=1,
        max_df=0.95,
        sublinear_tf=True,
    )

    x = vectorizer.fit_transform(
        questions
    )

    clusterer = KMeans(
        n_clusters=n_topics,
        random_state=42,
        n_init=20,
    )

    cluster_ids = clusterer.fit_predict(
        x
    )

    labels = [
        f"topic_{cluster_id:02d}"
        for cluster_id in cluster_ids
    ]

    return (
        labels,
        f"TF-IDF + KMeans, K={n_topics}",
    )


def infer_topic_labels(
    results,
    source="auto",
    n_topics=DEFAULT_N_TOPIC_CLUSTERS,
    embedding_model=DEFAULT_EMBEDDING_MODEL,
):
    """
    Determine a semantic topic/domain label for each sample.

    source:
        auto
            1. explicit dataset category/domain/topic
            2. sentence embedding + KMeans
            3. TF-IDF + KMeans fallback

        dataset
            require an explicit dataset category/domain/topic

        embedding
            force sentence embedding + KMeans

        tfidf
            force TF-IDF + KMeans
    """
    source = source.lower()

    if source == "dataset":
        return infer_topics_from_dataset(
            results
        )

    if source == "embedding":
        return infer_topics_with_embeddings(
            results,
            n_topics=n_topics,
            embedding_model=embedding_model,
        )

    if source == "tfidf":
        return infer_topics_with_tfidf(
            results,
            n_topics=n_topics,
        )

    if source != "auto":
        raise ValueError(
            f"Unknown topic source: {source}"
        )

    explicit_key = find_explicit_topic_field(
        results
    )

    if explicit_key is not None:
        return infer_topics_from_dataset(
            results
        )

    # Prefer semantic embeddings when available and usable.
    try:
        return infer_topics_with_embeddings(
            results,
            n_topics=n_topics,
            embedding_model=embedding_model,
        )
    except Exception as exc:
        print(
            "\nEmbedding topic inference was unavailable."
        )
        print(
            f"Reason: {type(exc).__name__}: {exc}"
        )
        print(
            "Falling back to TF-IDF + KMeans."
        )

    return infer_topics_with_tfidf(
        results,
        n_topics=n_topics,
    )


# ============================================================
# Word-topic statistics and topic specificity
# ============================================================

def build_word_topic_statistics(
    results,
    topic_labels,
):
    """
    Build topic-conditioned document frequencies.

    Important:
    - Topic labels are inferred from QUESTION semantics.
    - Word-topic counts are based on answer-word PRESENCE per sample,
      not raw token repetition.
    - The union of unsteered and steered answer words is used, so lexical
      classification is not determined by whether a word happened to be
      added or deleted.
    """
    if len(results) != len(topic_labels):
        raise ValueError(
            "results and topic_labels must have the same length."
        )

    topic_document_count = Counter()
    word_topic_count = defaultdict(Counter)

    for item, topic in zip(
        results,
        topic_labels,
    ):
        topic_document_count[topic] += 1

        unsteered_answer = (
            item.get("unsteered", {})
            .get("answer", "")
            or ""
        )

        steered_answer = (
            item.get("steered", {})
            .get("answer", "")
            or ""
        )

        words = (
            set(tokenize(unsteered_answer))
            | set(tokenize(steered_answer))
        )

        for word in words:
            word_topic_count[word][topic] += 1

    return (
        word_topic_count,
        topic_document_count,
    )


def js_divergence(p, q):
    """
    Jensen-Shannon divergence in [0, 1] using log base 2 normalization.
    """
    p = np.asarray(p, dtype=float)
    q = np.asarray(q, dtype=float)

    if p.ndim != 1 or q.ndim != 1:
        raise ValueError(
            "p and q must be one-dimensional."
        )

    if len(p) != len(q):
        raise ValueError(
            "p and q must have the same length."
        )

    p_sum = p.sum()
    q_sum = q.sum()

    if p_sum <= 0 or q_sum <= 0:
        return 0.0

    p = p / p_sum
    q = q / q_sum

    m = 0.5 * (p + q)

    def kl_div(a, b):
        mask = a > 0
        return float(
            np.sum(
                a[mask]
                * np.log(
                    a[mask] / b[mask]
                )
            )
        )

    jsd_nats = (
        0.5 * kl_div(p, m)
        + 0.5 * kl_div(q, m)
    )

    return float(
        jsd_nats / np.log(2.0)
    )


def compute_topic_specificity(
    word,
    word_topic_count,
    topic_document_count,
    prior_strength=TOPIC_PRIOR_STRENGTH,
    reliability_tau=TOPIC_RELIABILITY_TAU,
):
    """
    Estimate how concentrated a word is in particular semantic topics.

    raw_jsd:
        JSD(P(topic | word), P(topic))

    Bayesian smoothing:
        observed word-topic counts are shrunk toward the corpus-level
        topic prior, which prevents singleton words from becoming
        artificially maximally domain-specific.

    reliability:
        df / (df + tau)

    topic_specificity:
        raw_jsd * reliability

    Returns
    -------
    specificity : float
    raw_jsd : float
    reliability : float
    dominant_topic : str | None
    dominant_topic_share : float
    """
    topics = sorted(
        topic_document_count.keys()
    )

    if not topics:
        return (
            0.0,
            0.0,
            0.0,
            None,
            0.0,
        )

    background_counts = np.array(
        [
            topic_document_count[t]
            for t in topics
        ],
        dtype=float,
    )

    background = (
        background_counts
        / background_counts.sum()
    )

    counts = np.array(
        [
            word_topic_count[word][t]
            for t in topics
        ],
        dtype=float,
    )

    df = float(
        counts.sum()
    )

    if df <= 0:
        return (
            0.0,
            0.0,
            0.0,
            None,
            0.0,
        )

    prior_strength = max(
        0.0,
        float(prior_strength),
    )

    smoothed = (
        counts
        + prior_strength * background
    )

    word_distribution = (
        smoothed / smoothed.sum()
    )

    raw_jsd = js_divergence(
        word_distribution,
        background,
    )

    reliability_tau = max(
        0.0,
        float(reliability_tau),
    )

    if reliability_tau == 0:
        reliability = 1.0
    else:
        reliability = (
            df / (df + reliability_tau)
        )

    specificity = (
        raw_jsd * reliability
    )

    dominant_idx = int(
        np.argmax(word_distribution)
    )

    dominant_topic = topics[
        dominant_idx
    ]

    dominant_topic_share = float(
        word_distribution[dominant_idx]
    )

    return (
        float(specificity),
        float(raw_jsd),
        float(reliability),
        dominant_topic,
        dominant_topic_share,
    )


def classify_all_words_by_topic(
    word_df,
    word_topic_count,
    topic_document_count,
    domain_quantile=DOMAIN_SPECIFICITY_QUANTILE,
    prior_strength=TOPIC_PRIOR_STRENGTH,
    reliability_tau=TOPIC_RELIABILITY_TAU,
):
    """
    Classify lexical items using topic-conditioned specificity.

    Domain-specific words are the high-specificity tail among words that:
    - are actually edited at least once,
    - are not stopwords,
    - are not single-character tokens,
    - are not purely numeric.

    Manual GENERAL_FORCE / DOMAIN_FORCE overrides have highest priority.
    Stopwords are always general unless manually forced domain-specific.
    """
    if word_df.empty:
        out = word_df.copy()
        for col in [
            "topic_specificity",
            "topic_jsd",
            "topic_reliability",
            "dominant_topic",
            "dominant_topic_share",
            "specificity_percentile",
            "lexical_class",
            "classification_reason",
        ]:
            out[col] = pd.Series(dtype="object")
        return out, 0.0

    rows = []

    for _, row in word_df.iterrows():
        word = row["word"]

        (
            specificity,
            raw_jsd,
            reliability,
            dominant_topic,
            dominant_topic_share,
        ) = compute_topic_specificity(
            word=word,
            word_topic_count=word_topic_count,
            topic_document_count=topic_document_count,
            prior_strength=prior_strength,
            reliability_tau=reliability_tau,
        )

        row_dict = row.to_dict()
        row_dict[
            "topic_specificity"
        ] = specificity
        row_dict[
            "topic_jsd"
        ] = raw_jsd
        row_dict[
            "topic_reliability"
        ] = reliability
        row_dict[
            "dominant_topic"
        ] = dominant_topic
        row_dict[
            "dominant_topic_share"
        ] = dominant_topic_share

        rows.append(
            row_dict
        )

    out = pd.DataFrame(
        rows
    )

    numeric_mask = out[
        "word"
    ].str.fullmatch(
        r"\d+(?:\.\d+)?"
    )

    candidate_mask = (
        (out["edit_activity"] > 0)
        & (~out["word"].isin(STOPWORDS))
        & (out["word"].str.len() > 1)
        & (~numeric_mask)
    )

    candidates = out[
        candidate_mask
    ].copy()

    domain_quantile = float(
        np.clip(
            domain_quantile,
            0.0,
            1.0,
        )
    )

    out[
        "specificity_percentile"
    ] = np.nan

    if candidates.empty:
        threshold = math.inf
    else:
        threshold = float(
            candidates[
                "topic_specificity"
            ].quantile(
                domain_quantile
            )
        )

        percentiles = candidates[
            "topic_specificity"
        ].rank(
            method="average",
            pct=True,
        )

        out.loc[
            candidates.index,
            "specificity_percentile",
        ] = percentiles

    lexical_classes = []
    reasons = []

    for idx, row in out.iterrows():
        word = row["word"]

        if word in GENERAL_FORCE:
            lexical_classes.append(
                "general"
            )
            reasons.append(
                "manual_general"
            )
            continue

        if word in DOMAIN_FORCE:
            lexical_classes.append(
                "domain-specific"
            )
            reasons.append(
                "manual_domain"
            )
            continue

        if word in STOPWORDS:
            lexical_classes.append(
                "general"
            )
            reasons.append(
                "stopword"
            )
            continue

        if (
            len(word) <= 1
            or re.fullmatch(
                r"\d+(?:\.\d+)?",
                word,
            )
        ):
            lexical_classes.append(
                "general"
            )
            reasons.append(
                "non_content_token"
            )
            continue

        specificity = float(
            row["topic_specificity"]
        )

        percentile = row[
            "specificity_percentile"
        ]

        if (
            np.isfinite(threshold)
            and specificity > 0.0
            and specificity >= threshold
            and pd.notna(percentile)
            and percentile >= domain_quantile
        ):
            lexical_classes.append(
                "domain-specific"
            )
            reasons.append(
                "high_topic_specificity"
            )
        else:
            lexical_classes.append(
                "general"
            )
            reasons.append(
                "broad_cross_topic_distribution"
            )

    out[
        "lexical_class"
    ] = lexical_classes

    out[
        "classification_reason"
    ] = reasons

    return (
        out,
        threshold,
    )


# ============================================================
# Topic audit table
# ============================================================

def build_topic_assignment_df(
    results,
    topic_labels,
):
    rows = []

    for item, topic in zip(
        results,
        topic_labels,
    ):
        rows.append({
            "question_index": item.get(
                "question_index"
            ),
            "topic": topic,
            "question": item.get(
                "question",
                "",
            ),
        })

    return pd.DataFrame(
        rows
    )


# ============================================================
# Filtering for Top-K visualization only
# ============================================================

def filter_for_topk(
    df,
    remove_stopwords=True,
    remove_single_char=True,
    remove_numeric=True,
):
    out = df.copy()

    if remove_stopwords:
        out = out[
            ~out["word"].isin(STOPWORDS)
        ].copy()

    if remove_single_char:
        out = out[
            out["word"].str.len() > 1
        ].copy()

    if remove_numeric:
        out = out[
            ~out["word"].str.fullmatch(
                r"\d+(?:\.\d+)?"
            )
        ].copy()

    return out


# ============================================================
# Build one panel's word set:
# union(Top-K added, Top-K deleted)
# ============================================================

def build_combined_panel_df(
    classified_df,
    lexical_class,
    k=12,
    remove_stopwords=True,
    remove_single_char=True,
    remove_numeric=True,
):
    """
    For one lexical class (general or domain-specific):
    1. get Top-K by added
    2. get Top-K by deleted
    3. take union
    4. sort for plotting
    """
    df = filter_for_topk(
        classified_df,
        remove_stopwords=remove_stopwords,
        remove_single_char=remove_single_char,
        remove_numeric=remove_numeric,
    )

    df = df[
        df["lexical_class"] == lexical_class
    ].copy()

    top_added = (
        df[df["added"] > 0]
        .sort_values(
            ["added", "word"],
            ascending=[False, True],
        )
        .head(k)
    )

    top_deleted = (
        df[df["deleted"] > 0]
        .sort_values(
            ["deleted", "word"],
            ascending=[False, True],
        )
        .head(k)
    )

    combined = pd.concat(
        [top_added, top_deleted],
        ignore_index=True,
    ).drop_duplicates(subset="word").copy()

    if combined.empty:
        return combined

    combined["max_change"] = combined[
        ["added", "deleted"]
    ].max(axis=1)

    combined["balance"] = combined["added"] - combined["deleted"]

    # Sort by strongest change, then keep more positively shifted words first
    combined = combined.sort_values(
        ["max_change", "edit_activity", "balance", "word"],
        ascending=[False, False, False, True],
    ).reset_index(drop=True)

    return combined


# ============================================================
# Nature-style diverging panel
# ============================================================

def draw_diverging_panel(
    ax,
    panel_df,
    title,
    panel_letter,
):
    """
    One horizontal diverging plot:
    left  = deleted
    right = added
    """
    if panel_df.empty:
        ax.text(
            0.5,
            0.5,
            "No words available",
            ha="center",
            va="center",
            transform=ax.transAxes,
        )
        ax.set_axis_off()
        return

    # Highest-ranked word should appear at top
    plot_df = panel_df.iloc[::-1].reset_index(drop=True)

    words = plot_df["word"].tolist()
    y = list(range(len(plot_df)))

    deleted = -plot_df["deleted"].astype(int).to_numpy()
    added = plot_df["added"].astype(int).to_numpy()

    max_value = max(
        int(plot_df["added"].max()),
        int(plot_df["deleted"].max()),
        1,
    )

    # central line
    ax.axvline(
        0,
        color=CENTER_LINE_COLOR,
        linewidth=CENTER_LINEWIDTH,
        zorder=3,
    )

    # thin stems
    for yi, dv, av in zip(y, deleted, added):
        if dv != 0:
            ax.hlines(
                yi,
                dv,
                0,
                linewidth=STEM_LINEWIDTH,
                color=DELETE_COLOR,
                alpha=STEM_ALPHA,
                zorder=2,
            )
        if av != 0:
            ax.hlines(
                yi,
                0,
                av,
                linewidth=STEM_LINEWIDTH,
                color=ADD_COLOR,
                alpha=STEM_ALPHA,
                zorder=2,
            )

    # bars
    ax.barh(
        y,
        deleted,
        height=BAR_HEIGHT,
        color=DELETE_COLOR,
        alpha=BAR_ALPHA,
        label="Deleted from unsteered",
        zorder=2,
    )

    ax.barh(
        y,
        added,
        height=BAR_HEIGHT,
        color=ADD_COLOR,
        alpha=BAR_ALPHA,
        label="Added in steered",
        zorder=2,
    )

    # end markers
    ax.scatter(
        deleted,
        y,
        s=END_MARKER_SIZE,
        color=DELETE_COLOR,
        zorder=4,
    )

    ax.scatter(
        added,
        y,
        s=END_MARKER_SIZE,
        color=ADD_COLOR,
        zorder=4,
    )

    # numeric labels
    pad = max_value * VALUE_LABEL_PAD_RATIO

    for yi, dv, av in zip(
        y,
        plot_df["deleted"].astype(int).tolist(),
        plot_df["added"].astype(int).tolist(),
    ):
        if dv > 0:
            ax.text(
                -dv - pad,
                yi,
                f"-{dv}",
                ha="right",
                va="center",
                fontsize=VALUE_LABEL_FONT_SIZE,
                color=VALUE_TEXT_COLOR,
            )

        if av > 0:
            ax.text(
                av + pad,
                yi,
                f"+{av}",
                ha="left",
                va="center",
                fontsize=VALUE_LABEL_FONT_SIZE,
                color=VALUE_TEXT_COLOR,
            )

    ax.set_yticks(y)
    ax.set_yticklabels(words)

    ax.set_xlim(
        -int(plot_df["deleted"].max()) * XLIM_SCALE,
        int(plot_df["added"].max()) * XLIM_SCALE,
    )

    ax.xaxis.set_major_formatter(
        FuncFormatter(lambda x, pos: f"{abs(int(x))}")
    )

    ax.set_xlabel("Edit count   ← deleted     |     added →")

    # panel_title = (
    #     f"{panel_letter}   {title}"
    #     if panel_letter
    #     else title
    # )

    # ax.set_title(
    #     panel_title,
    #     loc="left",
    #     fontweight="bold",
    #     pad=PANEL_TITLE_PAD,
    # )

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_visible(False)

    ax.tick_params(
        axis="y",
        length=0,
        pad=YTICK_PAD,
    )

    ax.tick_params(
        axis="x",
        direction="out",
        length=TICK_LENGTH,
        width=TICK_WIDTH,
    )

    ax.grid(
        axis="x",
        linestyle="-",
        linewidth=GRID_LINEWIDTH,
        alpha=GRID_ALPHA,
        zorder=0,
    )


# ============================================================
# Full two-panel figure
# ============================================================

def plot_general_domain_two_panel(
    general_df,
    domain_df,
    summary,
    output_path,
    classifier_label,
    remove_stopwords=True,
):
    """
    One figure with two panels:

    Left:
        general words, with added and deleted shown together

    Right:
        domain-specific words, with added and deleted shown together
    """
    setup_plot_style()

    fig, axes = plt.subplots(
        1,
        2,
        figsize=TWO_PANEL_FIGSIZE,
    )

    draw_diverging_panel(
        axes[0],
        general_df,
        title="General words",
        panel_letter="a",
    )

    draw_diverging_panel(
        axes[1],
        domain_df,
        title="Domain-specific words",
        panel_letter="b",
    )

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        frameon=False,
        ncol=2,
        loc="upper right",
        bbox_to_anchor=(FIG_LEGEND_X, FIG_LEGEND_Y),
    )

    filter_label = (
        "stopwords excluded from Top-K"
        if remove_stopwords
        else "stopwords retained"
    )

    fig.suptitle(
        "Word-level lexical changes induced by steering",
        x=SUPTITLE_X,
        y=SUPTITLE_Y,
        ha="left",
        fontsize=SUPTITLE_FONT_SIZE,
        fontweight="bold",
    )

    fig.text(
        META_TEXT_X,
        META_TEXT_Y,
        f"Classification: {classifier_label}; {filter_label}",
        ha="left",
        va="top",
        fontsize=META_TEXT_FONT_SIZE,
        color=SECONDARY_TEXT_COLOR,
    )

    summary_text = (
        f"N = {summary['total_samples']}   |   "
        f"changed = {summary['changed_samples']} "
        f"({summary['changed_ratio']:.1%})   |   "
        f"added tokens = {summary['total_added']:,}   |   "
        f"deleted tokens = {summary['total_deleted']:,}   |   "
        f"net = {summary['net_length_change']:+,}"
    )

    fig.text(
        0.5,
        SUMMARY_TEXT_Y,
        summary_text,
        ha="center",
        va="bottom",
        fontsize=SUMMARY_TEXT_FONT_SIZE,
        color=SECONDARY_TEXT_COLOR,
    )

    fig.subplots_adjust(
        left=TWO_PANEL_LEFT,
        right=TWO_PANEL_RIGHT,
        bottom=TWO_PANEL_BOTTOM,
        top=TWO_PANEL_TOP,
        wspace=TWO_PANEL_WSPACE,
    )

    fig.savefig(
        output_path,
        dpi=SAVE_DPI,
        bbox_inches="tight",
    )

    fig.savefig(
        output_path.with_suffix(".pdf"),
        bbox_inches="tight",
    )

    plt.show()


# ============================================================
# Save one standalone panel
# ============================================================

def save_single_panel(
    panel_df,
    title,
    output_path,
):
    setup_plot_style()

    fig, ax = plt.subplots(
        figsize=SINGLE_PANEL_FIGSIZE,
    )

    draw_diverging_panel(
        ax,
        panel_df,
        title=title,
        panel_letter="",
    )

    handles, labels = ax.get_legend_handles_labels()
    ax.legend(
        handles,
        labels,
        frameon=False,
        ncol=2,
        loc="lower center",
        bbox_to_anchor=(0.5, SINGLE_LEGEND_Y),
    )

    plt.tight_layout()

    fig.savefig(
        output_path,
        dpi=SAVE_DPI,
        bbox_inches="tight",
    )

    fig.savefig(
        output_path.with_suffix(".pdf"),
        bbox_inches="tight",
    )

    plt.close(fig)


# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description=(
            "Compare unsteered and steered answers, classify changed "
            "words by topic-conditioned specificity, and draw two "
            "Nature-style diverging panels: general vs domain-specific."
        )
    )

    parser.add_argument(
        "json_path",
        nargs="?",
        default=str(DEFAULT_JSON_PATH),
        help="Input JSON path.",
    )

    parser.add_argument(
        "--topk",
        type=int,
        default=TOPK,
        help=(
            f"Top-K added and Top-K deleted within each category "
            f"(default: {TOPK})."
        ),
    )

    parser.add_argument(
        "--outdir",
        default=str(DEFAULT_OUT_DIR),
        help="Output directory.",
    )

    parser.add_argument(
        "--keep-stopwords",
        action="store_true",
        help="Keep stopwords in the Top-K figure.",
    )

    parser.add_argument(
        "--keep-single-char",
        action="store_true",
        help="Keep one-character tokens such as q in the Top-K figure.",
    )

    parser.add_argument(
        "--keep-numeric",
        action="store_true",
        help="Keep numeric tokens in the Top-K figure.",
    )

    parser.add_argument(
        "--topic-source",
        choices=[
            "auto",
            "dataset",
            "embedding",
            "tfidf",
        ],
        default="auto",
        help=(
            "How to obtain question topics. "
            "auto: dataset category if available, otherwise embedding "
            "clustering, then TF-IDF fallback. "
            "dataset: require category/domain/topic in JSON. "
            "embedding: sentence-transformers + KMeans. "
            "tfidf: TF-IDF + KMeans."
        ),
    )

    parser.add_argument(
        "--topic-clusters",
        type=int,
        default=DEFAULT_N_TOPIC_CLUSTERS,
        help=(
            "Number of semantic clusters when topic labels are inferred "
            f"(default: {DEFAULT_N_TOPIC_CLUSTERS})."
        ),
    )

    parser.add_argument(
        "--embedding-model",
        default=DEFAULT_EMBEDDING_MODEL,
        help=(
            "Sentence-transformers model used for embedding topic "
            "clustering."
        ),
    )

    parser.add_argument(
        "--domain-quantile",
        type=float,
        default=DOMAIN_SPECIFICITY_QUANTILE,
        help=(
            "High-specificity quantile used to label edited content "
            "words as domain-specific "
            f"(default: {DOMAIN_SPECIFICITY_QUANTILE})."
        ),
    )

    parser.add_argument(
        "--topic-prior-strength",
        type=float,
        default=TOPIC_PRIOR_STRENGTH,
        help=(
            "Bayesian prior strength for shrinking low-frequency "
            "word-topic distributions toward the corpus topic prior "
            f"(default: {TOPIC_PRIOR_STRENGTH})."
        ),
    )

    parser.add_argument(
        "--topic-reliability-tau",
        type=float,
        default=TOPIC_RELIABILITY_TAU,
        help=(
            "Low-frequency reliability shrinkage tau in df/(df+tau) "
            f"(default: {TOPIC_RELIABILITY_TAU})."
        ),
    )

    args = parser.parse_args()

    json_path = Path(args.json_path)
    out_dir = Path(args.outdir)
    out_dir.mkdir(parents=True, exist_ok=True)

    remove_stopwords = not args.keep_stopwords
    remove_single_char = not args.keep_single_char
    remove_numeric = not args.keep_numeric

    print("=" * 80)
    print("ACTUALLY READING JSON:")
    print(json_path.resolve())
    print("=" * 80)

    if setup_common_plot is not None:
        print("Plot style: plot_common.setup_plot() + local style overrides")
    else:
        print("Plot style: local style configuration only")

    (
        word_df,
        sample_df,
        detailed_diffs,
        summary,
    ) = analyse_json(
        json_path
    )

    _, results = load_json_results(
        json_path
    )

    topic_labels, topic_method = infer_topic_labels(
        results=results,
        source=args.topic_source,
        n_topics=max(
            2,
            args.topic_clusters,
        ),
        embedding_model=args.embedding_model,
    )

    (
        word_topic_count,
        topic_document_count,
    ) = build_word_topic_statistics(
        results=results,
        topic_labels=topic_labels,
    )

    (
        classified_df,
        specificity_threshold,
    ) = classify_all_words_by_topic(
        word_df=word_df,
        word_topic_count=word_topic_count,
        topic_document_count=topic_document_count,
        domain_quantile=args.domain_quantile,
        prior_strength=args.topic_prior_strength,
        reliability_tau=args.topic_reliability_tau,
    )

    topic_assignment_df = build_topic_assignment_df(
        results=results,
        topic_labels=topic_labels,
    )

    classifier_label = (
        "topic specificity via JSD"
    )

    general_panel_df = build_combined_panel_df(
        classified_df,
        lexical_class="general",
        k=max(1, args.topk),
        remove_stopwords=remove_stopwords,
        remove_single_char=remove_single_char,
        remove_numeric=remove_numeric,
    )

    domain_panel_df = build_combined_panel_df(
        classified_df,
        lexical_class="domain-specific",
        k=max(1, args.topk),
        remove_stopwords=remove_stopwords,
        remove_single_char=remove_single_char,
        remove_numeric=remove_numeric,
    )

    # --------------------------------------------------------
    # Save statistics
    # --------------------------------------------------------

    word_df.to_csv(
        out_dir / "token_diff_statistics.csv",
        index=False,
        encoding="utf-8-sig",
    )

    classified_df.to_csv(
        out_dir / "word_classification_all.csv",
        index=False,
        encoding="utf-8-sig",
    )

    sample_df.to_csv(
        out_dir / "sample_diff_statistics.csv",
        index=False,
        encoding="utf-8-sig",
    )

    topic_assignment_df.to_csv(
        out_dir / "question_topic_assignments.csv",
        index=False,
        encoding="utf-8-sig",
    )

    with open(
        out_dir / "sample_diff_details.jsonl",
        "w",
        encoding="utf-8",
    ) as f:
        for row in detailed_diffs:
            f.write(
                json.dumps(row, ensure_ascii=False) + "\n"
            )

    general_panel_df.to_csv(
        out_dir / "topk_general_combined.csv",
        index=False,
        encoding="utf-8-sig",
    )

    domain_panel_df.to_csv(
        out_dir / "topk_domain_combined.csv",
        index=False,
        encoding="utf-8-sig",
    )

    metadata = {
        "classification_method": "topic_specificity_jsd",
        "topic_method": topic_method,
        "topic_source_requested": args.topic_source,
        "n_topics": len(topic_document_count),
        "topic_document_count": dict(
            sorted(
                topic_document_count.items(),
                key=lambda x: str(x[0]),
            )
        ),
        "domain_quantile": args.domain_quantile,
        "topic_prior_strength": args.topic_prior_strength,
        "topic_reliability_tau": args.topic_reliability_tau,
        "specificity_threshold": (
            None
            if not np.isfinite(
                specificity_threshold
            )
            else float(
                specificity_threshold
            )
        ),
    }

    with open(
        out_dir / "topic_specificity_metadata.json",
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            metadata,
            f,
            ensure_ascii=False,
            indent=2,
        )

    # --------------------------------------------------------
    # Console summary
    # --------------------------------------------------------

    print("\n" + "=" * 72)
    print("UNSTEERED vs STEERED WORD-DIFF SUMMARY")
    print("=" * 72)
    print(f"Total samples        : {summary['total_samples']}")
    print(f"Changed samples      : {summary['changed_samples']}")
    print(f"Identical samples    : {summary['identical_samples']}")
    print(f"Changed ratio        : {summary['changed_ratio']:.2%}")
    print(f"Added tokens         : {summary['total_added']:,}")
    print(f"Deleted tokens       : {summary['total_deleted']:,}")
    print(f"Net token change     : {summary['net_length_change']:+,}")

    print("\n" + "=" * 72)
    print("TOPIC-SPECIFICITY CLASSIFIER")
    print("=" * 72)
    print(f"Topic method         : {topic_method}")
    print(f"Number of topics     : {len(topic_document_count)}")
    print(f"Domain quantile      : {args.domain_quantile:.3f}")
    print(
        "Specificity threshold: "
        + (
            f"{specificity_threshold:.6f}"
            if np.isfinite(specificity_threshold)
            else "N/A"
        )
    )
    print(
        f"Prior strength       : {args.topic_prior_strength:.3f}"
    )
    print(
        f"Reliability tau      : {args.topic_reliability_tau:.3f}"
    )

    print("\nTopic sample counts:")
    for topic, count in sorted(
        topic_document_count.items(),
        key=lambda x: str(x[0]),
    ):
        print(
            f"  {topic}: {count}"
        )

    display_columns = [
        "word",
        "added",
        "deleted",
        "document_frequency",
        "topic_specificity",
        "topic_jsd",
        "topic_reliability",
        "dominant_topic",
        "dominant_topic_share",
        "classification_reason",
    ]

    print("\n" + "=" * 72)
    print("GENERAL PANEL WORDS")
    print("=" * 72)

    if general_panel_df.empty:
        print("(empty)")
    else:
        print(
            general_panel_df[
                display_columns
            ].to_string(
                index=False
            )
        )

    print("\n" + "=" * 72)
    print("DOMAIN-SPECIFIC PANEL WORDS")
    print("=" * 72)

    if domain_panel_df.empty:
        print("(empty)")
    else:
        print(
            domain_panel_df[
                display_columns
            ].to_string(
                index=False
            )
        )

    # --------------------------------------------------------
    # Plot
    # --------------------------------------------------------

    combined_path = (
        out_dir / "general_domain_diverging_two_panel.png"
    )

    plot_general_domain_two_panel(
        general_df=general_panel_df,
        domain_df=domain_panel_df,
        summary=summary,
        output_path=combined_path,
        classifier_label=classifier_label,
        remove_stopwords=remove_stopwords,
    )

    save_single_panel(
        panel_df=general_panel_df,
        title="General words",
        output_path=out_dir / "general_diverging_single_panel.png",
    )

    save_single_panel(
        panel_df=domain_panel_df,
        title="Domain-specific words",
        output_path=out_dir / "domain_diverging_single_panel.png",
    )

    print("\n" + "=" * 80)
    print("OUTPUT")
    print("=" * 80)

    print("\nCombined figure:")
    print(combined_path.resolve())
    print(combined_path.with_suffix(".pdf").resolve())

    print("\nStandalone figures:")
    print(
        (out_dir / "general_diverging_single_panel.png").resolve()
    )
    print(
        (out_dir / "general_diverging_single_panel.pdf").resolve()
    )
    print(
        (out_dir / "domain_diverging_single_panel.png").resolve()
    )
    print(
        (out_dir / "domain_diverging_single_panel.pdf").resolve()
    )

    print("\nCSV / JSON:")
    print((out_dir / "topk_general_combined.csv").resolve())
    print((out_dir / "topk_domain_combined.csv").resolve())
    print((out_dir / "word_classification_all.csv").resolve())
    print((out_dir / "question_topic_assignments.csv").resolve())
    print((out_dir / "topic_specificity_metadata.json").resolve())


if __name__ == "__main__":
    main()
