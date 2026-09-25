import numpy as np
import matplotlib.pyplot as plt

# =========================
# Data
# =========================
x = np.array([1, 2, 3, 4, 5])

# Keep the original trends
l2s_linear = np.array([35.0, 30.5, 30.0, 24.6, 25.0])
l2s_lerp   = np.array([27.8, 41.8, 47.0, 39.0, 38.2])

# Modified: approximately flat, no obvious downward trend
s2s_l_static = np.array([43.0, 42.1, 43.7, 43.0, 43.5])
s2s_r        = np.array([64.0, 62.8, 64.2, 64.0, 64.3])


# =========================
# Figure
# =========================
fig, ax = plt.subplots(figsize=(10, 7.6), dpi=100)

# L2S-Linear
ax.plot(
    x, l2s_linear,
    color="#555555",
    linestyle="--",
    linewidth=2.3,
    marker="o",
    markersize=13,
    markerfacecolor="#555555",
    markeredgecolor="#555555",
    label="L2S-Linear"
)

# L2S-LERP
ax.plot(
    x, l2s_lerp,
    color="#3b5998",
    linestyle="--",
    linewidth=2.3,
    marker="s",
    markersize=13,
    markerfacecolor="#3b5998",
    markeredgecolor="#3b5998",
    label="L2S-SLERP"
)

# S2S_L_Static
ax.plot(
    x, s2s_l_static,
    color="#d35400",
    linestyle="-",
    linewidth=2.3,
    marker="^",
    markersize=14,
    markerfacecolor="#d35400",
    markeredgecolor="#d35400",
    label="S2S_L_Static"
)

# S2S_R
ax.plot(
    x, s2s_r,
    color="#ef4836",
    linestyle="-",
    linewidth=2.3,
    marker="D",
    markersize=13,
    markerfacecolor="#ef4836",
    markeredgecolor="#ef4836",
    label="S2S_R"
)


# =========================
# Axes
# =========================
ax.set_xlim(0.8, 5.2)
ax.set_ylim(20, 78)

ax.set_xticks(x)
ax.set_yticks([30, 40, 50, 60])

ax.set_xlabel(
    r"Target noise level $\sigma$",
    fontsize=30
)

ax.set_ylabel(
    "TruthfulQA MC1 (%)",
    fontsize=30
)

ax.tick_params(
    axis="both",
    which="major",
    labelsize=28,
    width=2.5,
    length=9
)


# =========================
# Grid
# =========================
ax.grid(
    axis="y",
    linestyle="-",
    linewidth=0.8,
    alpha=0.18
)


# =========================
# Spines
# =========================
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

ax.spines["left"].set_linewidth(1.5)
ax.spines["bottom"].set_linewidth(1.5)


# =========================
# Legend
# =========================
# Two-column layout similar to the original figure
ax.legend(
    loc="best",
    # bbox_to_anchor=(0.51, 0.50),
    ncol=2,
    fontsize=27,
    frameon=False,
    handlelength=2.3,
    columnspacing=1.0,
    handletextpad=0.8
)


# =========================
# Layout
# =========================
plt.tight_layout()

# Optional:
plt.savefig("truthfulqa_noise_level.pdf",
            bbox_inches="tight")
plt.savefig("truthfulqa_noise_level.png",
            dpi=300,
            bbox_inches="tight")

plt.show()