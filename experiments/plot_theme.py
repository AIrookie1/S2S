#!/usr/bin/env python3
"""Global typography, line, marker, color and saving settings for experiments."""
from __future__ import annotations

from copy import deepcopy
from typing import Any, Final

import matplotlib as mpl

# ---------------------------------------------------------------------------
# Global palette
# ---------------------------------------------------------------------------

COLORS: Final = {
    "llama": "#3C5488",
    "qwen": "#E64B35",
    "blue": "#3C5488",
    "red": "#E64B35",
    "green": "#009E73",
    "orange": "#D55E00",
    "black": "#222222",
    "dark_gray": "#4D4D4D",
    "mid_gray": "#777777",
    "light_gray": "#D9D9D9",
    "white": "#FFFFFF",
    # Semantic colors used by word-number analysis.
    "delete": "#C55A4A",
    "add": "#4778A8",
    "secondary_text": "#555555",
    "value_text": "#444444",
}

MODEL_STYLES: Final = {
    "llama3.1-8B-Instruct": {
        "label": "Llama 3.1 8B",
        "color": COLORS["llama"],
        "marker": "o",
        "linestyle": "-",
    },
    "Qwen2.5-7B-Instruct": {
        "label": "Qwen 2.5 7B",
        "color": COLORS["qwen"],
        "marker": "s",
        "linestyle": "--",
    },
}

METHOD_STYLES: Final = {
    "linear_shared_mean": {"color": COLORS["blue"], "marker": "o"},
    "spherical_shared_mean": {"color": COLORS["orange"], "marker": "s"},
    "gt_neg_to_pos": {"color": COLORS["green"], "marker": "^"},
}


RELIABILITY_GROUP_COLORS: Final = {
    ("Mean", "Train"): COLORS["blue"],
    ("Mean", "Test"): COLORS["orange"],
    ("L2S", "Train"): COLORS["green"],
    ("L2S", "Test"): COLORS["red"],
}

DENSITY_COLORS: Final = (
    "#F7FBFF",
    "#D7E6F5",
    "#86A8CE",
    COLORS["blue"],
    COLORS["red"],
)

# ---------------------------------------------------------------------------
# Typography / line profiles
# ---------------------------------------------------------------------------
# The standard profiles deliberately preserve the two visual scales that were
# already used by the project's 10-inch and 5-inch wide ablation figures.
# Edit these dictionaries to resize text/markers/lines across every figure that
# uses the corresponding profile.

STYLE_PROFILES: Final[dict[str, dict[str, float]]] = {
    # Four independently editable standard-size profiles.  The two half-width
    # presets and two quarter-width presets intentionally start with matching
    # values, but each dictionary can be tuned separately later.
    "half_half": {
        "font_size": 24.0,
        "axes_label_size": 24.0,
        "axes_title_size": 24.0,
        "tick_label_size": 20.5,
        "legend_font_size": 20.5,
        "annotation_font_size": 20.5,
        "panel_label_size": 24.0,
        "axes_line_width": 0.8,
        "line_width": 2.5,
        "marker_size": 10.0,
        "marker_edge_width": 1.3,
        "errorbar_line_width": 1.3,
        "errorbar_cap_size": 2.6,
        "errorbar_cap_thickness": 1.3,
        "tick_width": 2.8,
        "tick_length": 8.0,
    },
    "half_quarter": {
        "font_size": 24.0,
        "axes_label_size": 24.0,
        "axes_title_size": 24.0,
        "tick_label_size": 20.5,
        "legend_font_size": 20.5,
        "annotation_font_size": 20.5,
        "panel_label_size": 24.0,
        "axes_line_width": 0.8,
        "line_width": 2.5,
        "marker_size": 10.0,
        "marker_edge_width": 1.3,
        "errorbar_line_width": 1.3,
        "errorbar_cap_size": 2.6,
        "errorbar_cap_thickness": 1.3,
        "tick_width": 2.8,
        "tick_length": 8.0,
    },
    "quarter_half": {
        "font_size": 18.0,
        "axes_label_size": 19.0,
        "axes_title_size": 19.0,
        "tick_label_size": 17.5,
        "legend_font_size": 17.5,
        "annotation_font_size": 17.0,
        "panel_label_size": 19.0,
        "axes_line_width": 0.8,
        "line_width": 1.5,
        "marker_size": 8.0,
        "marker_edge_width": 0.9,
        "errorbar_line_width": 0.9,
        "errorbar_cap_size": 2.2,
        "errorbar_cap_thickness": 0.9,
        "tick_width": 1.8,
        "tick_length": 5.0,
    },
    "quarter_quarter": {
        "font_size": 18.0,
        "axes_label_size": 19.0,
        "axes_title_size": 19.0,
        "tick_label_size": 17.5,
        "legend_font_size": 17.5,
        "annotation_font_size": 17.0,
        "panel_label_size": 19.0,
        "axes_line_width": 0.8,
        "line_width": 1.5,
        "marker_size": 8.0,
        "marker_edge_width": 0.9,
        "errorbar_line_width": 0.9,
        "errorbar_cap_size": 2.2,
        "errorbar_cap_thickness": 0.9,
        "tick_width": 1.8,
        "tick_length": 5.0,
    },
    "steering_vector": {
        "font_size": 24.0,
        "axes_label_size": 24.0,
        "axes_title_size": 24.0,
        "tick_label_size": 24.0,
        "legend_font_size": 24.0,
        "annotation_font_size": 24.0,
        "panel_label_size": 24.0,
        "axes_line_width": 1.5,
        "line_width": 1.7,
        "marker_size": 7.5,
        "marker_edge_width": 0.8,
        "errorbar_line_width": 1.0,
        "errorbar_cap_size": 2.0,
        "errorbar_cap_thickness": 1.0,
        "tick_width": 1.3,
        "tick_length": 4.0,
    },
    "heatmap": {
        "font_size": 28.0,
        "axes_label_size": 26.0,
        "axes_title_size": 26.0,
        "tick_label_size": 20.0,
        "legend_font_size": 20.0,
        "annotation_font_size": 20.0,
        "panel_label_size": 28.0,
        "axes_line_width": 0.8,
        "line_width": 1.5,
        "marker_size": 8.0,
        "marker_edge_width": 0.9,
        "errorbar_line_width": 0.9,
        "errorbar_cap_size": 2.0,
        "errorbar_cap_thickness": 0.9,
        "tick_width": 1.0,
        "tick_length": 4.0,
    },
    # Compact custom figures such as L2S reliability retain their existing
    # physical scale while sharing the same central configuration mechanism.
    "compact": {
        "font_size": 18.0,
        "axes_label_size": 19.0,
        "axes_title_size": 19.0,
        "tick_label_size": 17.5,
        "legend_font_size": 17.5,
        "annotation_font_size": 17.0,
        "panel_label_size": 19.0,
        "axes_line_width": 0.8,
        "line_width": 1.5,
        "marker_size": 8.0,
        "marker_edge_width": 0.9,
        "errorbar_line_width": 0.9,
        "errorbar_cap_size": 2.2,
        "errorbar_cap_thickness": 0.9,
        "tick_width": 1.8,
        "tick_length": 5.0,
    },
    # Used by the word-number figure, which is intentionally large and dense.
    "word_number": {
        "font_size": 18.5,
        "axes_label_size": 18.0,
        "axes_title_size": 20.0,
        "tick_label_size": 17.4,
        "legend_font_size": 18.0,
        "annotation_font_size": 17.0,
        "panel_label_size": 20.0,
        "axes_line_width": 1.75,
        "line_width": 2.0,
        "marker_size": 18.0,
        "marker_edge_width": 0.9,
        "errorbar_line_width": 1.0,
        "errorbar_cap_size": 2.0,
        "errorbar_cap_thickness": 1.0,
        "tick_width": 0.7,
        "tick_length": 3.0,
    },
}

SAVE_SETTINGS: Final = {
    "png_dpi": 600,
    "facecolor": "white",
    "bbox_inches": "tight",
    "pad_inches": 0.04,
    "pdf_fonttype": 42,
    "ps_fonttype": 42,
    "svg_fonttype": "none",
}


def get_style(profile_name: str) -> dict[str, float]:
    if profile_name not in STYLE_PROFILES:
        raise KeyError(
            f"Unknown style profile {profile_name!r}; "
            f"choose one of {sorted(STYLE_PROFILES)}"
        )
    return deepcopy(STYLE_PROFILES[profile_name])


def apply_style(profile_name: str = "quarter_quarter") -> dict[str, float]:
    """Apply a central Matplotlib style profile and return its values."""
    style = get_style(profile_name)
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
            "font.size": style["font_size"],
            "axes.labelsize": style["axes_label_size"],
            "axes.titlesize": style["axes_title_size"],
            "xtick.labelsize": style["tick_label_size"],
            "ytick.labelsize": style["tick_label_size"],
            "legend.fontsize": style["legend_font_size"],
            "axes.linewidth": style["axes_line_width"],
            "lines.linewidth": style["line_width"],
            "lines.markersize": style["marker_size"],
            "xtick.direction": "out",
            "ytick.direction": "out",
            "xtick.major.width": style["tick_width"],
            "ytick.major.width": style["tick_width"],
            "xtick.major.size": style["tick_length"],
            "ytick.major.size": style["tick_length"],
            "axes.unicode_minus": False,
            "pdf.fonttype": SAVE_SETTINGS["pdf_fonttype"],
            "ps.fonttype": SAVE_SETTINGS["ps_fonttype"],
            "svg.fonttype": SAVE_SETTINGS["svg_fonttype"],
            "savefig.bbox": SAVE_SETTINGS["bbox_inches"],
            "savefig.pad_inches": SAVE_SETTINGS["pad_inches"],
            "savefig.facecolor": SAVE_SETTINGS["facecolor"],
        }
    )
    return style


def merged_style(profile_name: str, overrides: dict[str, Any] | None = None) -> dict[str, Any]:
    style: dict[str, Any] = get_style(profile_name)
    if overrides:
        style.update(overrides)
    return style
