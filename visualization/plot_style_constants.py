"""Shared plotting style constants for visualization scripts."""

import matplotlib.pyplot as plt

PLOT_RCPARAMS = {
    "font.family": "serif",
    "font.serif": ["Libertinus Serif", "Linux Libertine", "DejaVu Serif"],
    "font.size": 10,
    "axes.labelsize": 11,
    "axes.titlesize": 12,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.fontsize": 9,
    "axes.grid": False,
}

PLOT_COLORS = {
    "purple": "#6d28d9",
    "orange": "#f97316",
    "grey": "#64748b",
    "teal": "#14b8a6",
    "baseline": "#3b0764",
    "accent": "#8b5cf6",
}


def apply_plot_style() -> None:
    """Apply shared Matplotlib rcParams style for plots."""
    plt.rcParams.update(PLOT_RCPARAMS)
