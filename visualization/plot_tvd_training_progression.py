from __future__ import annotations

import argparse
import re
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import pandas as pd

from config.paths import FIG_DIR
from visualization.constants import apply_plot_style, PLOT_COLORS

apply_plot_style()

# ── data paths ────────────────────────────────────────────────────────────────
_BASE = Path("/scratch3/ip004/data/results/eval")

# Non-parameterized (Grover) set
_NON_PARAM_DIRS = {
    "Baseline\n(Qwen3-8B)": _BASE / "grover_sets/baseline/baseline_qwen3_8b",
    "SFT + GRPO\nStage 1":  _BASE / "grover_sets/sft_grpo",
    "SFT + GRPO\nStage 2":  _BASE / "grover_sets/sft_grpo/sft_grpo_grover_grpo_641",
}

# Parameterized (Random) set
_PARAM_DIRS = {
    "Baseline\n(Qwen3-8B)": _BASE / "random_sets/baseline/qwen3_8b",
    "SFT + GRPO\nStage 1":  _BASE / "random_sets/sft_grpo",
    "SFT + GRPO\nStage 2":  _BASE / "random_sets/sft_grpo/sft_grpo_random_grpo_1152",
}

_TEST_SPLITS = ["test_1", "test_2", "test_3"]
_SPLIT_LABELS = ["Set 1", "Set 2", "Set 3"]

# ── colours ───────────────────────────────────────────────────────────────────
# Stage colours (for grouped bar chart – one colour per stage)
_STAGE_COLORS = [
    PLOT_COLORS["grey"],     # Baseline
    PLOT_COLORS["accent"],   # Stage 1
    PLOT_COLORS["purple"],   # Stage 2
]

# Test-set colours (for line plot – one colour per test split)
_SPLIT_COLORS = [
    PLOT_COLORS["teal"],
    PLOT_COLORS["orange"],
    PLOT_COLORS["baseline"],
]

# ── parsing ───────────────────────────────────────────────────────────────────
_TVD_RE = re.compile(
    r"Average TVD Top-k \(k=\d+\):\s*([+-]?\d*\.?\d+(?:[eE][+-]?\d+)?)"
)


def _parse_tvd(path: Path) -> float | None:
    """Return the first 'Average TVD' value from a summary file."""
    try:
        text = path.read_text(encoding="utf-8", errors="ignore")
    except FileNotFoundError:
        return None
    m = _TVD_RE.search(text)
    return float(m.group(1)) if m else None


def _load_data(
    stage_dirs: dict[str, Path],
    splits: list[str],
) -> dict[str, list[float | None]]:
    """
    Returns {stage_label: [tvd_split1, tvd_split2, tvd_split3]}.
    """
    return {
        stage: [_parse_tvd(d / f"{split}_summary.txt") for split in splits]
        for stage, d in stage_dirs.items()
    }


# ── bar chart ─────────────────────────────────────────────────────────────────

def _bar_panel(
    ax: plt.Axes,
    data: dict[str, list[float | None]],
    split_labels: list[str],
    title: str,
    show_legend: bool = False,
    show_ylabel: bool = True,
) -> None:
    """Draw a grouped bar chart on *ax*.

    Groups = test splits (x-axis),  bars within each group = training stages.
    """
    stages = list(data.keys())
    n_splits = len(split_labels)
    n_stages = len(stages)
    total_width = 0.72
    bar_w = total_width / n_stages
    x = np.arange(n_splits)

    for i, (stage, color) in enumerate(zip(stages, _STAGE_COLORS)):
        values = [v if v is not None else 0.0 for v in data[stage]]
        offset = (i - (n_stages - 1) / 2) * bar_w
        ax.bar(
            x + offset,
            values,
            bar_w * 0.92,
            label=stage.replace("\n", " "),
            color=color,
            alpha=0.88,
            edgecolor="white",
            linewidth=0.6,
        )
        # Value labels above bars
        for j, v in enumerate(values):
            if v is not None:
                ax.text(
                    x[j] + offset,
                    v + 0.012,
                    f"{v:.3f}",
                    ha="center",
                    va="bottom",
                    fontsize=7,
                    color=color,
                    fontweight="bold",
                )

    ax.set_title(title, fontsize=12, pad=10)
    ax.set_xticks(x)
    ax.set_xticklabels(split_labels, fontsize=9)
    ax.set_xlabel("Test Split", fontsize=10)
    if show_ylabel:
        ax.set_ylabel("Average TVD", fontsize=10)
    ax.set_ylim(0, 1.1)
    ax.yaxis.set_major_locator(ticker.MultipleLocator(0.2))
    ax.yaxis.set_minor_locator(ticker.MultipleLocator(0.1))
    ax.grid(True, axis="y", which="major", linestyle="--", alpha=0.15, zorder=0)
    if show_legend:
        ax.legend(
            loc="upper right",
            fontsize=8.5,
            framealpha=0.9,
            title_fontsize=8.5,
            borderpad=0.8,
            labelspacing=0.6,
        )


def plot_bar_chart(
    non_param_data: dict[str, list[float | None]],
    param_data: dict[str, list[float | None]],
    save_path: Path,
) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5), sharey=True)

    _bar_panel(
        axes[0],
        non_param_data,
        _SPLIT_LABELS,
        title="Non-parameterized Set",
        show_legend=True,
        show_ylabel=True,
    )
    _bar_panel(
        axes[1],
        param_data,
        _SPLIT_LABELS,
        title="Parameterized Set",
        show_legend=False,
        show_ylabel=False,
    )

    fig.tight_layout()
    fig.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Bar chart saved → {save_path}")


# ── line plot ─────────────────────────────────────────────────────────────────

def _line_panel(
    ax: plt.Axes,
    data: dict[str, list[float | None]],
    split_labels: list[str],
    title: str,
    show_ylabel: bool = True,
) -> None:
    """Draw a line plot on *ax*.

    x-axis = training stages,  one line per test split.
    """
    stages = list(data.keys())
    stage_ticks = [s.replace("\n", "\n") for s in stages]
    x = np.arange(len(stages))

    # Collect (xi, v, color, text) for all series; place after to avoid overlaps
    _label_info: list[tuple[int, float, str, str]] = []

    # Reorganise data: per split across stages
    for split_idx, (split_label, color) in enumerate(
        zip(split_labels, _SPLIT_COLORS)
    ):
        values = [
            (data[stage][split_idx] if data[stage][split_idx] is not None else np.nan)
            for stage in stages
        ]
        ax.plot(
            x,
            values,
            marker="o",
            markersize=7,
            linewidth=2.0,
            color=color,
            label=split_label,
            alpha=0.9,
            zorder=3,
        )
        # Store raw label info for collision-aware placement below
        for xi, v in zip(x, values):
            if not np.isnan(v):
                _label_info.append((xi, v, color, f"{v:.3f}"))

    # ── collision-aware label placement ──────────────────────────────────────
    # Group labels by x position, sorted ascending by value.
    # When two neighbours are within min_gap of each other, flip the lower one
    # to sit below its dot rather than pushing both upward.
    from itertools import groupby
    min_gap = 0.05  # in data units – if two labels are closer than this, flip
    _label_info.sort(key=lambda t: (t[0], t[1]))
    for _xi, group in groupby(_label_info, key=lambda t: t[0]):
        grp = list(group)  # sorted ascending by v
        # Decide position: default is above the dot
        positions = [(v + 0.022, "bottom") for _, v, _, _ in grp]
        # Check each adjacent pair; flip the lower one below if too close
        for k in range(len(grp) - 1):
            v_lo = grp[k][1]
            v_hi = grp[k + 1][1]
            if (v_hi + 0.022) - (v_lo + 0.022) < min_gap:
                positions[k] = (v_lo - 0.022, "top")  # flip lower label below dot
        for (_, v, color, text), (yp, va) in zip(grp, positions):
            ax.text(
                _xi, yp, text,
                ha="center", va=va,
                fontsize=7.5, color=color, fontweight="bold",
            )

    ax.set_title(title, fontsize=12, pad=10)
    ax.set_xticks(x)
    ax.set_xticklabels(stage_ticks, fontsize=9)
    ax.set_xlabel("Training Stage", fontsize=10)
    if show_ylabel:
        ax.set_ylabel("Average TVD", fontsize=10)
    ax.set_ylim(0, 1.15)
    ax.yaxis.set_major_locator(ticker.MultipleLocator(0.2))
    ax.yaxis.set_minor_locator(ticker.MultipleLocator(0.1))
    ax.grid(True, axis="y", which="major", linestyle="--", alpha=0.15)
    ax.grid(True, axis="x", which="major", linestyle="--", alpha=0.10)

    # Shade background per stage region for readability
    for xi in x[1:]:
        ax.axvline(xi, color=PLOT_COLORS["grey"], linewidth=0.5, alpha=0.3, linestyle=":")


def plot_line_chart(
    non_param_data: dict[str, list[float | None]],
    param_data: dict[str, list[float | None]],
    save_path: Path,
) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5), sharey=True)

    _line_panel(axes[0], non_param_data, _SPLIT_LABELS,
                title="Non-parameterized Set", show_ylabel=True)
    _line_panel(axes[1], param_data,     _SPLIT_LABELS,
                title="Parameterized Set",    show_ylabel=False)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles, labels,
        loc="center left",
        bbox_to_anchor=(1.0, 0.5),
        fontsize=9,
        framealpha=0.9,
        title_fontsize=9,
        borderpad=1.0,
        labelspacing=0.8,
    )

    fig.tight_layout()
    fig.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Line chart saved → {save_path}")


# ── combined figure (bar + line stacked) ─────────────────────────────────────




# ── token-limit exceedance plot ───────────────────────────────────────────────

_TOKEN_LIMIT = 32_768

# CSVs for the parameterized (random) set – same ordering as _PARAM_DIRS
_PARAM_CSV_DIRS = {
    "Baseline\n(Qwen3-8B)": _BASE / "random_sets/baseline/qwen3_8b",
    "SFT + GRPO\nStage 1":  _BASE / "random_sets/sft_grpo",
    "SFT + GRPO\nStage 2":  _BASE / "random_sets/sft_grpo/sft_grpo_random_grpo_1152",
}

# CSVs for the non-parameterized (Grover) set – same ordering as _NON_PARAM_DIRS
_NON_PARAM_CSV_DIRS = {
    "Baseline\n(Qwen3-8B)": _BASE / "grover_sets/baseline/baseline_qwen3_8b",
    "SFT + GRPO\nStage 1":  _BASE / "grover_sets/sft_grpo",
    "SFT + GRPO\nStage 2":  _BASE / "grover_sets/sft_grpo/sft_grpo_grover_grpo_641",
}


def _load_token_exceedance(
    stage_dirs: dict[str, Path],
    splits: list[str],
    limit: int = _TOKEN_LIMIT,
) -> dict[str, dict[str, tuple[int, float]]]:
    """
    Returns {stage: {split: (count_exceeded, pct_exceeded)}}.
    """
    result: dict[str, dict[str, tuple[int, float]]] = {}
    for stage, d in stage_dirs.items():
        result[stage] = {}
        for split in splits:
            try:
                df = pd.read_csv(d / f"{split}.csv", usecols=["tokens"])
                n_exc = int((df["tokens"] >= limit).sum())
                pct   = 100.0 * n_exc / len(df)
            except FileNotFoundError:
                n_exc, pct = 0, 0.0
            result[stage][split] = (n_exc, pct)
    return result


def plot_token_limit_chart(
    np_token_data: dict[str, dict[str, tuple[int, float]]],
    param_token_data: dict[str, dict[str, tuple[int, float]]],
    split_labels: list[str],
    splits: list[str],
    save_path: Path,
    limit: int = _TOKEN_LIMIT,
) -> None:
    """Two-panel grouped bar chart (non-param left, param right).

    Shows percentage of samples exceeding the token limit per split × stage.
    """
    stages   = list(np_token_data.keys())
    n_splits = len(splits)
    n_stages = len(stages)
    total_width = 0.72
    bar_w = total_width / n_stages
    x = np.arange(n_splits)

    # Compute a shared y-limit with headroom for value labels
    all_pcts = [
        td[stage][split][1]
        for td in (np_token_data, param_token_data)
        for stage in stages
        for split in splits
    ]
    y_top = max(all_pcts) * 1.25 + 2.0  # headroom for text labels

    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))


    panel_specs = [
        (axes[0], np_token_data,    "Non-parameterized Set", True),
        (axes[1], param_token_data, "Parameterized Set",     False),
    ]

    for ax, token_data, panel_title, show_ylabel in panel_specs:
        for i, (stage, color) in enumerate(zip(stages, _STAGE_COLORS)):
            values = [token_data[stage][split][1] for split in splits]  # pct only
            offset = (i - (n_stages - 1) / 2) * bar_w
            ax.bar(
                x + offset,
                values,
                bar_w * 0.92,
                label=stage.replace("\n", " "),
                color=color,
                alpha=0.88,
                edgecolor="white",
                linewidth=0.6,
            )
            for j, v in enumerate(values):
                ax.text(
                    x[j] + offset,
                    v + 0.4,
                    f"{v:.1f}%",
                    ha="center",
                    va="bottom",
                    fontsize=7,
                    color=color,
                    fontweight="bold",
                )

        ax.set_title(panel_title, fontsize=12, pad=10)
        ax.set_xticks(x)
        ax.set_xticklabels(split_labels, fontsize=9)
        ax.set_ylabel("Samples Exceeding Token Limit (%)" if show_ylabel else "", fontsize=10)
        ax.grid(True, axis="y", which="major", linestyle="--", alpha=0.15, zorder=0)
        ax.set_ylim(0, y_top)
        ax.yaxis.set_major_formatter(ticker.PercentFormatter(decimals=0))

    # Shared legend to the right, outside both panels
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles, labels,
        loc="center left",
        bbox_to_anchor=(0.755, 0.5),
        fontsize=9,
        framealpha=0.9,
        title="Training Stage",
        title_fontsize=9,
        borderpad=1.0,
        labelspacing=0.8,
    )

    fig.subplots_adjust(left=0.08, right=0.74, top=0.90, bottom=0.08, wspace=0.28)
    fig.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Token-limit chart saved → {save_path}")


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot TVD changes across training stages (Baseline → SFT Stage 1 → SFT+GRPO Stage 2)."
    )
    parser.add_argument("--output-dir",  type=Path, default=Path(FIG_DIR))
    parser.add_argument("--line-file",   type=str,  default="tvd_training_line.png")
    parser.add_argument("--token-file",    type=str,  default="tvd_token_limit.png")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    non_param_data = _load_data(_NON_PARAM_DIRS, _TEST_SPLITS)
    param_data     = _load_data(_PARAM_DIRS,     _TEST_SPLITS)

    # Print loaded data for inspection
    print("\n── Non-parameterized (Grover) TVD values ──")
    for stage, vals in non_param_data.items():
        label = stage.replace("\n", " ")
        print(f"  {label}: {[f'{v:.4f}' if v is not None else 'N/A' for v in vals]}")

    print("\n── Parameterized (Random) TVD values ──")
    for stage, vals in param_data.items():
        label = stage.replace("\n", " ")
        print(f"  {label}: {[f'{v:.4f}' if v is not None else 'N/A' for v in vals]}")
    print()

    plot_line_chart(non_param_data, param_data, args.output_dir / args.line_file)

    # Token-limit exceedance (both sets combined)
    token_data_param = _load_token_exceedance(_PARAM_CSV_DIRS, _TEST_SPLITS)
    token_data_np    = _load_token_exceedance(_NON_PARAM_CSV_DIRS, _TEST_SPLITS)

    print("\n── Token-limit exceedances (parameterized set, limit=32768) ──")
    for stage, split_vals in token_data_param.items():
        label = stage.replace("\n", " ")
        print(f"  {label}: " + ", ".join(
            f"{s}: {c} ({p:.1f}%)" for s, (c, p) in split_vals.items()
        ))
    print("\n── Token-limit exceedances (non-parameterized set, limit=32768) ──")
    for stage, split_vals in token_data_np.items():
        label = stage.replace("\n", " ")
        print(f"  {label}: " + ", ".join(
            f"{s}: {c} ({p:.1f}%)" for s, (c, p) in split_vals.items()
        ))
    print()
    plot_token_limit_chart(
        token_data_np, token_data_param,
        _SPLIT_LABELS, _TEST_SPLITS,
        args.output_dir / args.token_file,
    )
