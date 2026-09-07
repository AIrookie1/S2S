# Academic Figure Skill Asset Confirmation (verified against assets/figures/)
# (a) BarComparison → assets/figures/BarComparison/plot_comparison_GeneRegulatory.py → param inherit
# RULE: "native run" = load pre-rendered PNG via Image.open().ax.imshow().
#       "param inherit" = drawing function below that copies Class A/B/C values.
#       If a panel says "native run" and you write a drawing function, you broke the contract.

# Academic Figure Skill Typography Baseline — COPY VERBATIM, place at TOP of script
import matplotlib as mpl
mpl.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "Liberation Sans"],
    "font.size": 14,
    "axes.titlesize": 14,
    "axes.labelsize": 14,
    "xtick.labelsize": 14,
    "ytick.labelsize": 14,
    "legend.fontsize": 14,
    "figure.titlesize": 14,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.linewidth": 1.6,
    "xtick.direction": "out",
    "ytick.direction": "out",
    "xtick.major.width": 1.6,
    "ytick.major.width": 1.6,
    "legend.frameon": False,
})

# Academic Figure Skill Nature/Cell/Science Color Palette -- COPY VERBATIM
CATEGORICAL = ["#2166AC", "#B2182B", "#1B7837", "#F1A340", "#762A83", "#666666"]
CATEGORICAL_EXTENDED = [
    "#2166AC", "#B2182B", "#1B7837", "#F1A340", "#762A83", "#666666",
    "#4393C3", "#D6604D", "#5AAE61", "#B35806", "#9970AB", "#999999",
]
DIVERGING   = ["#2166AC", "#F7F7F7", "#B2182B"]
SEQUENTIAL  = ["#F7FBFF", "#6BAED6", "#08306B"]
ACCENT_RED  = "#B2182B"
GREY        = "#999999"
BLACK       = "#222222"

# Academic Figure Skill Export Baseline — COPY VERBATIM
mpl.rcParams.update({
    "pdf.fonttype": 42,         # TrueType font embedding
    "svg.fonttype": "none",     # editable text in SVG
    "savefig.bbox": "tight",    # trim whitespace
    "savefig.dpi": 300,
})

def save_cns_figure(fig, filename):
    """Standard Academic Figure Skill export: vector PDF + 300dpi PNG preview."""
    fig.savefig(f"{filename}.pdf", bbox_inches="tight", dpi=300)
    fig.savefig(f"{filename}.png", bbox_inches="tight", dpi=300)


import argparse
import json
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import MaxNLocator


METHODS = ("pair_specific", "shared_mean")
METHOD_COLORS = (CATEGORICAL[0], ACCENT_RED)
MM_TO_INCH = 1 / 25.4
DEFAULT_FIGSIZE_MM = (89.0, 79.0)


def positive_float(value: str) -> float:
    value_f = float(value)
    if value_f <= 0:
        raise argparse.ArgumentTypeError("Figure dimensions must be > 0.")
    return value_f


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Publication-grade Fig. 1 MC1 answer-pair ablation"
    )
    parser.add_argument("results", nargs="+", help="Completed fold JSON files")
    parser.add_argument(
        "--output_prefix",
        default="./experiments/fig1_train_mean/figures/fig1_pair_vs_mean_mc1",
        help="Output path prefix without extension.",
    )
    parser.add_argument(
        "--figsize_mm",
        nargs=2,
        type=positive_float,
        metavar=("WIDTH", "HEIGHT"),
        default=DEFAULT_FIGSIZE_MM,
        help=(
            "Final figure size in millimetres: WIDTH HEIGHT. "
            "Default: 89 70 (Nature-family single-column width)."
        ),
    )
    return parser.parse_args()


def load_results(paths: list[str]) -> tuple[list[dict], np.ndarray, list[str]]:
    payloads = [json.loads(Path(path).read_text(encoding="utf-8")) for path in paths]
    values: list[list[float]] = []
    fold_ids: list[object] = []
    labels: list[str] | None = None

    for path, payload in zip(paths, payloads):
        missing = set(METHODS).difference(payload.get("methods", {}))
        if missing:
            raise ValueError(f"{path} is missing methods: {sorted(missing)}")

        fold_values = np.asarray(
            [payload["methods"][method]["metrics"]["MC1"] for method in METHODS],
            dtype=float,
        )
        if not np.isfinite(fold_values).all() or np.any(
            (fold_values < 0) | (fold_values > 1)
        ):
            raise ValueError(
                f"{path} contains invalid MC1 values: {fold_values.tolist()}"
            )

        values.append(fold_values.tolist())
        fold_ids.append(payload.get("fold_idx", Path(path).stem))

        current_labels = [payload["methods"][method]["label"] for method in METHODS]
        if labels is None:
            labels = current_labels
        elif current_labels != labels:
            raise ValueError(f"{path} uses inconsistent method labels")

    if len(set(map(str, fold_ids))) != len(fold_ids):
        raise ValueError(f"Duplicate fold identifiers: {fold_ids}")

    return payloads, np.asarray(values, dtype=float), labels or list(METHODS)


def summarize(values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    means = values.mean(axis=0)
    sd = values.std(axis=0, ddof=1) if len(values) > 1 else np.zeros(len(METHODS))

    # Academic Figure Skill bar validation:
    # abs(means.max() - means.min()) / means.max() > 0.05
    relative_range = float(
        np.ptp(means) / max(float(means.max()), np.finfo(float).eps)
    )
    if relative_range <= 0.05:
        warnings.warn(
            "The two mean MC1 values differ by no more than 5%; "
            "the expected bar-chart signal is visually weak.",
            stacklevel=2,
        )

    return means, sd


def build_figure(
    values: np.ndarray,
    means: np.ndarray,
    sd: np.ndarray,
    labels: list[str],
    figsize_mm: tuple[float, float],
) -> plt.Figure:
    """
    Param-inherit BarComparison drawing.

    Class A (hard parameters inherited/adapted):
      - categorical fills
      - no bar outlines
      - black error bars with caps
      - no grid

    Class B (scaling parameters):
      - physical figure size is configurable in mm
      - Academic Figure Skill typography baseline remains fixed

    Class C (logic parameters):
      - top/right spines hidden
      - no fold-point overlay; bars show fold means with SD
      - mean labels above bars
      - title summarizes effect direction
    """
    width_mm, height_mm = figsize_mm
    fig, ax = plt.subplots(
        figsize=(width_mm * MM_TO_INCH, height_mm * MM_TO_INCH)
    )

    x = np.arange(len(METHODS), dtype=float)

    bars = ax.bar(
        x,
        means,
        width=0.56,
        color=METHOD_COLORS,
        linewidth=0,
        zorder=2,
    )

    if len(values) > 1:
        ax.errorbar(
            x,
            means,
            yerr=sd,
            fmt="none",
            ecolor=BLACK,
            elinewidth=0.8,
            capsize=3,
            capthick=0.8,
            zorder=4,
        )

    upper_data = float(max(values.max(), np.max(means + sd)))
    y_upper = min(
        1.0,
        max(0.20, upper_data + max(0.06, upper_data * 0.18)),
    )

    ax.set_ylim(0, y_upper)
    ax.set_xlim(-0.65, len(METHODS) - 0.35)
    ax.yaxis.set_major_locator(MaxNLocator(nbins=5, prune="upper"))

    ax.set_ylabel("Training-set MC1")
    ax.set_xticks(x, labels)
    ax.tick_params(axis="x", length=0, pad=5)
    ax.grid(False)

    for bar, mean, error in zip(bars, means, sd):
        label_y = min(
            y_upper * 0.94,
            float(mean + error + y_upper * 0.035),
        )
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            label_y,
            f"{mean:.3f}",
            ha="center",
            va="bottom",
            color=BLACK,
            fontweight="bold",
            fontsize=8,
        )

    delta_pp = 100.0 * float(means[1] - means[0])
    if delta_pp < 0:
        title = f"Shared mean reduces MC1 by {abs(delta_pp):.1f} pp"
    elif delta_pp > 0:
        title = f"Shared mean increases MC1 by {delta_pp:.1f} pp"
    else:
        title = "Shared mean leaves MC1 unchanged"

    ax.set_title(title, pad=16, fontweight="bold")
    ax.text(
        0.5,
        1.02,
        "Mean ± SD across outer folds",
        transform=ax.transAxes,
        ha="center",
        va="bottom",
        fontsize=7,
        color=BLACK,
    )

    # Fractions are dimensionless, so the composition remains stable when
    # WIDTH/HEIGHT is changed from the CLI.
    fig.subplots_adjust(
        left=0.19,
        right=0.98,
        bottom=0.20,
        top=0.78,
    )
    return fig


def write_statistics(
    path: Path,
    result_paths: list[str],
    payloads: list[dict],
    values: np.ndarray,
    means: np.ndarray,
    sd: np.ndarray,
    labels: list[str],
    figsize_mm: tuple[float, float],
) -> None:
    report = {
        "metric": "TruthfulQA MC1",
        "evaluation_split": "outer-fold training questions",
        "replicate_definition": "Each point is one outer fold.",
        "n_folds": int(len(values)),
        "center": "arithmetic mean across folds",
        "error_bars": "sample standard deviation across folds (ddof=1)",
        "statistical_test": None,
        "multiple_comparison_correction": None,
        "baseline": "answer-pair raw vector v_ij = h_i+ - h_j-",
        "figure_size_mm": {
            "width": float(figsize_mm[0]),
            "height": float(figsize_mm[1]),
        },
        "source_result_files": [str(Path(item)) for item in result_paths],
        "fold_ids": [payload.get("fold_idx") for payload in payloads],
        "methods": {
            method: {
                "label": label,
                "fold_values": values[:, idx].tolist(),
                "mean": float(means[idx]),
                "sd": float(sd[idx]),
            }
            for idx, (method, label) in enumerate(zip(METHODS, labels))
        },
        "shared_mean_minus_pair_specific_pp": 100.0 * float(
            means[1] - means[0]
        ),
    }

    path.write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def main() -> None:
    args = parse_args()

    payloads, values, labels = load_results(args.results)
    means, sd = summarize(values)

    figsize_mm = (float(args.figsize_mm[0]), float(args.figsize_mm[1]))

    prefix = Path(args.output_prefix)
    prefix.parent.mkdir(parents=True, exist_ok=True)

    fig = build_figure(
        values=values,
        means=means,
        sd=sd,
        labels=labels,
        figsize_mm=figsize_mm,
    )

    save_cns_figure(fig, str(prefix))
    fig.savefig(f"{prefix}.svg", bbox_inches="tight", dpi=300)
    plt.close(fig)

    statistics_path = prefix.with_name(f"{prefix.name}_statistics.json")
    write_statistics(
        statistics_path,
        args.results,
        payloads,
        values,
        means,
        sd,
        labels,
        figsize_mm,
    )

    print(
        f"Figure size: {figsize_mm[0]:g} × {figsize_mm[1]:g} mm "
        f"({figsize_mm[0] * MM_TO_INCH:.3f} × "
        f"{figsize_mm[1] * MM_TO_INCH:.3f} in)"
    )
    print(f"Saved {prefix}.pdf")
    print(f"Saved {prefix}.svg")
    print(f"Saved {prefix}.png")
    print(f"Saved {statistics_path}")


if __name__ == "__main__":
    main()
