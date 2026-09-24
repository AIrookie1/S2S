#!/usr/bin/env python3
"""Central figure-size presets for all experiment plots.

The four standard presets are the Cartesian product of two width fractions
and two height fractions.  Their numeric values intentionally preserve the
sizes already used by the main ablation figures:

* 1/2 width  = 10 in
* 1/4 width  = 5 in
* 1/2 height = 8 in
* 1/4 height = 4 in

Special figures can keep a custom aspect ratio, but those sizes live here as
well so there is still one place to change every experiment figure size.
"""
from __future__ import annotations

from typing import Final

WIDTH_INCHES: Final = {
    "1/2": 10.0,
    "1/4": 5.0,
}

HEIGHT_INCHES: Final = {
    "1/2": 8.0,
    "1/4": 4.0,
}

STANDARD_FIGURE_SIZES: Final = {
    "half_half": (WIDTH_INCHES["1/2"], HEIGHT_INCHES["1/2"]),
    "half_quarter": (WIDTH_INCHES["1/2"], HEIGHT_INCHES["1/4"]),
    "quarter_half": (WIDTH_INCHES["1/4"], HEIGHT_INCHES["1/2"]),
    "quarter_quarter": (WIDTH_INCHES["1/4"], HEIGHT_INCHES["1/4"]),
}

# Non-standard aspect ratios that are useful for a particular figure type.
# They remain centralized and can be replaced by a standard preset at any time.
CUSTOM_FIGURE_SIZES: Final = {
    "steering_vector_ablation": (WIDTH_INCHES["1/2"], HEIGHT_INCHES["1/4"]),
    "dyn2d_heatmap": (9.0, 7.0),
    "kappa_view": (5.25, 4.95),
    "l2s_performance": (WIDTH_INCHES["1/4"]-1, HEIGHT_INCHES["1/4"]-1),
    "l2s_vector_reliability": (WIDTH_INCHES["1/4"], HEIGHT_INCHES["1/4"]-0.2),
    "residual_analysis": (15.0, 4.6),
    "word_number_two_panel": (18.0, 6.9),
    "word_number_single_panel": (9.0, 5.6),
}


def get_size(size_name: str) -> tuple[float, float]:
    """Return a standard or custom figure size by name."""
    if size_name in STANDARD_FIGURE_SIZES:
        return STANDARD_FIGURE_SIZES[size_name]
    if size_name in CUSTOM_FIGURE_SIZES:
        return CUSTOM_FIGURE_SIZES[size_name]
    choices = sorted((*STANDARD_FIGURE_SIZES, *CUSTOM_FIGURE_SIZES))
    raise KeyError(f"Unknown figure size {size_name!r}; choose one of {choices}")
