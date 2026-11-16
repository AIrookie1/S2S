#!/usr/bin/env python3
"""Per-figure layout, text, visibility and annotation configuration.

This is the file to edit for figure-specific labels, whether text is shown,
legend/annotation placement and special layout values.  Global typography,
colors and line/marker sizes live in plot_theme.py; physical sizes live in
plot_sizes.py.
"""
from __future__ import annotations

from copy import deepcopy
from typing import Any, Final

from plot_sizes import get_size


FIGURE_SPECS: Final[dict[str, dict[str, Any]]] = {
    "rebuttal_R3Q3_candidate_gates": {
        "size": "half_quarter",
        "style": "quarter_quarter",
        "text": {"ylabel": "Gate trigger rate (%)", "correct": "Correct candidates",
                 "incorrect": "Incorrect candidates", "apply": "Steering applied",
                 "observe": "Observe only"},
        "legend": {"loc": "upper center", "ncol": 2, "bbox_to_anchor": (0.5, 1.2)},
    },
    "fig3c_gate_margin": {
        "size": "half_quarter",
        "style": "half_quarter",
        "text": {"xlabel": "Margin threshold", "ylabel": "MC1", "title": None},
        "show": {"xlabel": True, "ylabel": True, "title": False, "legend": True},
        "legend": {"loc": "best"},
    },
    "fig3c_weight_sharpness": {
        "size": "half_quarter",
        "style": "half_quarter",
        "text": {"xlabel": "Steering strength kappa", "ylabel": "MC1", "title": None},
        "show": {"xlabel": True, "ylabel": True, "title": False, "legend": True},
        "legend": {"loc": "best"},
    },
    "fig3b_hidden_size": {
        "size": "quarter_quarter",
        "style": "quarter_quarter",
        "text": {"xlabel": "MLP hidden size", "ylabel": "MC1", "title": None},
        "show": {"xlabel": True, "ylabel": True, "title": False, "legend": True},
        "legend": {"loc": "best"},
        "xticks": [16, 64, 256, 1024, 2048],
    },
    "fig3b_predictor_fraction": {
        "size": "quarter_quarter",
        "style": "quarter_quarter",
        "text": {"xlabel": "Validation fraction", "ylabel": "MC1", "title": None},
        "show": {"xlabel": True, "ylabel": True, "title": False, "legend": True},
        "legend": {"loc": "best"},
    },
    "fig2a_oracle_vectors": {
        "size": "fig2a_oracle_vectors",
        "style": "steering_vector",
        "text": {
            "xlabel": "layer",
            "ylabel": "MC1",
            "title": None,
            "linear_shared_mean": "mean steer vector",
            "spherical_shared_mean": "Spherical + shared mean",
            "gt_neg_to_pos": "sample-specific vector",
        },
        "show": {
            "xlabel": True,
            "ylabel": True,
            "title": False,
            "legend": True,
            # Preserve the current two-curve output.  Set True whenever the
            # spherical_shared_mean series should be displayed as a third line.
            "spherical_shared_mean": False,
        },
        "legend": {
            "loc": "upper left",
            "ncol": 1,
            "handlelength": 2.0,
            "handletextpad": 0.7,
            "labelspacing": 0.5,
            "borderaxespad": 0.7,
        },
        "layout": {"left": 0.115, "right": 0.988, "bottom": 0.225, "top": 0.97},
    },
    "S2S_L_heatmap": {
        "size": "S2S_L_heatmap",
        "style": "heatmap",
        "text": {
            "xlabel": "Steering strength",
            "ylabel": "SLERP coefficient",
            "title": "S2S_L Ablation: TruthfulQA MC1 (fold mean)",
            "colorbar": "MC1",
        },
        "show": {
            "xlabel": True,
            "ylabel": True,
            "title": False,
            "cell_values": True,
            "colorbar": True,
        },
        "heatmap": {"cmap": "viridis", "value_format": ".3f"},
    },
    "fig3d_adaptive_weights": {
        "size": "fig3d_adaptive_weights",
        "style": "quarter_quarter",
        "text": {"title": None},
        "show": {
            "title": False,
            "vertex_labels": False,
            "axes": False,
            "ticks": False,
            "legend": False,
            "annotation": False,
            "colorbar": False,
        },
        "ternary": {
            "grid_levels": (0.2, 0.4, 0.6, 0.8),
            "grid_line_width": 0.45,
            "grid_alpha": 0.55,
            "boundary_line_width": 1.5,
            "mean_marker": "*",
            "mean_marker_size": 600,
            "mean_marker_line_width": 1.6,
            "xlim": (-0.06, 1.06),
            "ylim_pad": 0.055,
            "tight_pad": 0.15,
        },
    },
    "l2s_performance": {
        "size": "l2s_performance",
        "style": "compact",
        "text": {
            "xlabel": None,
            "ylabel": "MC1",
            "title": None,
            "mean_tick": "Mean vector",
            "l2s_tick": "L2S vector",
            "baseline_legend": "No steering",
        },
        "show": {"xlabel": False, "ylabel": True, "title": False, "legend": True, "annotation": True},
        "annotations": {
            "delta": {
                "template": r"$\Delta_{{S-L}}$={delta:+.3f}",
                "xy": (0.98, 0.035),
                "ha": "right",
                "va": "bottom",
            }
        },
        "layout": {"left": 0.17, "right": 0.985, "bottom": 0.19, "top": 0.94},
    },
    "l2s_vector_reliability": {
        "size": "l2s_vector_reliability",
        "style": "compact",
        "text": {"xlabel": r"Cosine similarity", "ylabel": "Relative norm error", "title": None},
        "show": {"xlabel": True, "ylabel": True, "title": False, "legend": True, "annotation": False},
        "annotations": {
            "better": {"text": "better ↘", "xy": (0.985, 0.035), "ha": "right", "va": "bottom"}
        },
        "legend": {"loc": "upper right", "ncol": 2},
        "layout": {"left": 0.18, "right": 0.985, "bottom": 0.19, "top": 0.94},
    },
    "fig4ab_residual_transfer": {
        "size": "fig4ab_residual_transfer",
        "style": "compact",
        "text": {
            "panel_a_ylabel": "Cosine similarity to oracle residual",
            "panel_b_ylabel": r"$\Delta$ MC1 over mean-only (pp)",
            "panel_c_xlabel": "Residual-oracle alignment quintile",
            "panel_c_ylabel": "Angular-error reduction vs. mean-only (deg)",
        },
        "show": {"title": False, "legend": False},
        "layout": {"left": 0.19, "right": 0.985, "bottom": 0.22, "top": 0.88},
    },
    "word_number": {
        "size": "word_number_two_panel",
        "style": "word_number",
        "text": {
            "xlabel": "Edit count   ← deleted     |     added →",
            "suptitle": "Word-level lexical changes induced by steering",
            "general_panel_title": "General words",
            "domain_panel_title": "Domain-specific words",
            "deleted_legend": "Deleted from unsteered",
            "added_legend": "Added in steered",
            "meta_template": "Classification: {classifier_label}; {filter_label}",
            "summary_template": (
                "N = {total_samples}   |   changed = {changed_samples} "
                "({changed_ratio:.1%})   |   added tokens = {total_added:,}   |   "
                "deleted tokens = {total_deleted:,}   |   net = {net_length_change:+,}"
            ),
            "show_xlabel": True,
            "show_panel_title": False,
            "show_value_labels": True,
            "show_suptitle": True,
            "show_meta_text": True,
            "show_summary_text": True,
            "show_legend": True,
        },
        "word_number": {
            "two_panel_size": "word_number_two_panel",
            "single_panel_size": "word_number_single_panel",
            "suptitle_font_size": 15.5,
            "meta_text_font_size": 18.0,
            "summary_text_font_size": 17.8,
            "value_label_font_size": 17.0,
            "center_line_width": 1.8,
            "stem_line_width": 2.0,
            "grid_line_width": 1.45,
            "grid_alpha": 0.13,
            "bar_height": 0.46,
            "bar_alpha": 0.82,
            "stem_alpha": 0.95,
            "end_marker_size": 18.0,
            "xlim_scale": 1.25,
            "value_label_pad_ratio": 0.025,
            "panel_title_pad": 8,
            "ytick_pad": 5,
            "two_panel_layout": {"left": 0.10, "right": 0.985, "bottom": 0.10, "top": 0.88, "wspace": 0.26},
            "suptitle_xy": (0.06, 0.985),
            "meta_text_xy": (0.06, 0.952),
            "summary_text_y": 0.018,
            "figure_legend_xy": (0.985, 0.988),
            "single_legend_y": 1.02,
        },
    },
}


def get_figure_spec(name: str) -> dict[str, Any]:
    if name not in FIGURE_SPECS:
        raise KeyError(f"Unknown figure spec {name!r}; choose one of {sorted(FIGURE_SPECS)}")
    return deepcopy(FIGURE_SPECS[name])


def get_figure_size(name: str) -> tuple[float, float]:
    return get_size(str(get_figure_spec(name)["size"]))


def text_value(name: str, key: str, default: Any = None) -> Any:
    return get_figure_spec(name).get("text", {}).get(key, default)


def is_visible(name: str, key: str, default: bool = True) -> bool:
    return bool(get_figure_spec(name).get("show", {}).get(key, default))
