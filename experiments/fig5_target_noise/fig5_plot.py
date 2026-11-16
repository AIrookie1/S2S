#!/usr/bin/env python3
"""Nature-style robustness plot: MC1 versus target-vector noise."""
from __future__ import annotations
import argparse, csv, sys
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from plot_common import save
from plot_theme import apply_style, get_style, COLORS

ORDER = ("L2S-Linear", "L2S-SLERP", "S2S_L_Static", "S2S_R")
MARKERS = {"L2S-Linear":"o", "L2S-SLERP":"s", "S2S_L_Static":"^", "S2S_R":"D"}
LINESTYLES = {"L2S-Linear":"--", "L2S-SLERP":"--", "S2S_L_Static":"-", "S2S_R":"-"}
COLORS_BY_METHOD = {
    "L2S-Linear": COLORS["dark_gray"],
    "L2S-SLERP": COLORS["blue"],
    "S2S_L_Static": COLORS["orange"],
    "S2S_R": COLORS["red"],
}

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--input", type=Path, required=True)
    p.add_argument("--output_prefix", type=Path, default=None)
    a = p.parse_args()
    with a.input.open(encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    apply_style("compact")
    style = get_style("compact")
    fig, ax = plt.subplots(figsize=(6.2, 4.6))
    for method in ORDER:
        selected = sorted((r for r in rows if r["method"] == method), key=lambda r: float(r["noise_sigma"]))
        x = np.asarray([float(r["noise_sigma"]) for r in selected])
        y = 100*np.asarray([float(r["MC1"]) for r in selected])
        ax.plot(x, y, label=method, marker=MARKERS[method], linestyle=LINESTYLES[method],
                color=COLORS_BY_METHOD[method], linewidth=style["line_width"],
                markersize=style["marker_size"], markeredgewidth=style["marker_edge_width"])
    ax.set_xlabel(r"Target noise level $\sigma$")
    ax.set_ylabel("TruthfulQA MC1 (%)")
    observed_x = sorted({float(r["noise_sigma"]) for r in rows})
    ax.set_xticks(observed_x, [f"{value:g}" for value in observed_x])
    ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
    ax.grid(axis="y", linewidth=.45, alpha=.20); ax.set_axisbelow(True)
    ax.legend(frameon=False, ncol=2, loc="best", handlelength=2.2, columnspacing=1.0)
    fig.subplots_adjust(left=.17, right=.98, bottom=.18, top=.97)
    save(fig, a.output_prefix or (a.input.parent / "fig5_target_noise"))
    plt.close(fig)

if __name__ == "__main__":
    main()
