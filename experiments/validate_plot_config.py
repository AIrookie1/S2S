#!/usr/bin/env python3
"""Validate centralized experiment plotting configuration."""
from __future__ import annotations

from plot_sizes import get_size
from plot_specs import FIGURE_SPECS
from plot_theme import STYLE_PROFILES

REQUIRED_STYLE_KEYS = {
    "font_size",
    "axes_label_size",
    "axes_title_size",
    "tick_label_size",
    "legend_font_size",
    "annotation_font_size",
    "axes_line_width",
    "line_width",
    "marker_size",
    "marker_edge_width",
    "errorbar_line_width",
    "errorbar_cap_size",
    "errorbar_cap_thickness",
    "tick_width",
    "tick_length",
}


def main() -> None:
    errors: list[str] = []
    for name, spec in sorted(FIGURE_SPECS.items()):
        try:
            width, height = get_size(str(spec["size"]))
            if width <= 0 or height <= 0:
                errors.append(f"{name}: non-positive figure size {(width, height)}")
        except (KeyError, TypeError) as exc:
            errors.append(f"{name}: invalid size: {exc}")

        profile = str(spec.get("style", ""))
        if profile not in STYLE_PROFILES:
            errors.append(f"{name}: unknown style profile {profile!r}")
            continue
        missing = REQUIRED_STYLE_KEYS.difference(STYLE_PROFILES[profile])
        if missing:
            errors.append(f"{name}: style {profile!r} missing {sorted(missing)}")

    if errors:
        raise SystemExit("Plot configuration errors:\n- " + "\n- ".join(errors))

    print(f"Plot configuration OK: {len(FIGURE_SPECS)} figure specs validated")


if __name__ == "__main__":
    main()
