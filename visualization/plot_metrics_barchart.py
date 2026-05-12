"""Plot grouped bar charts of Avg Fidelity broken down by:
  - Number of qubits
  - Circuit depth
  - Number of gates

for four models across two evaluation sets (non-parameterized / parameterized).
Produces a single figure with 2 rows × 3 columns (2 sets × 3 dimensions).

Usage
-----
python -m visualization.plot_metrics_barchart \
    [--non-param-paths  p1 p2 p3 p4] \
    [--param-paths      p1 p2 p3 p4] \
    [--model-labels     l1 l2 l3 l4] \
    [--metric           "Avg Fidelity"] \
    [--output-dir  visualization/figures] \
    [--output-file metrics_barchart.png]

Paths are ordered: Qwen3-8B, GPT-OSS, SFT, SFT+GRPO.
"""
from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np

from config.paths import FIG_DIR
from visualization.constants import apply_plot_style, PLOT_COLORS

apply_plot_style()

# ── defaults ──────────────────────────────────────────────────────────────────
_BASE = Path("/scratch3/ip004/data/results/eval")

DEFAULT_NON_PARAM_PATHS = [
    _BASE / "grover_sets/baseline/baseline_qwen3_8b/test_1_summary.txt",
    _BASE / "grover_sets/baseline/baseline_gptoss/test_1_summary.txt",
    _BASE / "grover_sets/sft/test_1_summary.txt",
    _BASE / "grover_sets/sft_grpo/sft_grpo_grover_grpo_641/test_1_summary.txt",
]
DEFAULT_PARAM_PATHS = [
    _BASE / "random_sets/baseline/qwen3_8b/test_1_summary.txt",
    _BASE / "random_sets/baseline/baseline_gptoss/test_1_summary.txt",
    _BASE / "random_sets/sft/test_1_summary.txt",
    _BASE / "random_sets/sft_grpo/sft_grpo_random_grpo_1152/test_1_summary.txt",
]
DEFAULT_MODEL_LABELS = ["Qwen3-8B", "GPT-OSS", "SFT", "SFT + GRPO"]
MODEL_COLORS = [
    PLOT_COLORS["grey"],
    PLOT_COLORS["teal"],
    PLOT_COLORS["accent"],
    PLOT_COLORS["baseline"],
]

# ── regex ─────────────────────────────────────────────────────────────────────
_DEPTH_SECTION_RE = re.compile(
    r"METRICS BY CIRCUIT DEPTH:(.*?)(?:\n\s{2}METRICS BY|\Z)", re.DOTALL
)
_QUBIT_SECTION_RE = re.compile(
    r"METRICS BY NUMBER OF QUBITS:(.*?)(?:\n\s{2}METRICS BY|\Z)", re.DOTALL
)
_GATES_SECTION_RE = re.compile(
    r"METRICS BY NUMBER OF GATES:(.*?)(?:\n {2}[A-Z]|\Z)", re.DOTALL
)

# Group header lines, e.g. "Depth 1-10 (n=3393):" or "1_qubits (n=1851):" or "Gates 1-10 (n=2000):"
_GROUP_HEADER_RE = re.compile(
    r"^\s{4}(Depth \d+-\d+|Gates \d+-\d+|\d+_qubits)\s*\(n=(\d+)\)\s*:", re.MULTILINE
)
_AVG_TVD_RE = re.compile(r"Avg TVD Top-k:\s*([+-]?\d*\.?\d+(?:[eE][+-]?\d+)?)")


def _parse_section(
    section_text: str,
) -> dict[str, dict[str, float | int]]:
    """Return {group_label: {"fidelity": float, "n": int}} for one section."""
    result: dict[str, dict[str, float | int]] = {}
    headers = list(_GROUP_HEADER_RE.finditer(section_text))
    for idx, hdr in enumerate(headers):
        label = hdr.group(1)
        n = int(hdr.group(2))
        block_start = hdr.end()
        block_end = headers[idx + 1].start() if idx + 1 < len(headers) else len(section_text)
        block = section_text[block_start:block_end]
        m = _AVG_TVD_RE.search(block)
        fidelity = float(m.group(1)) if m else 0.0
        result[label] = {"fidelity": fidelity, "n": n}
    return result


def parse_summary(path: Path) -> dict[str, Any]:
    """Parse one summary .txt into depth / qubit / gate fidelity breakdowns."""
    text = path.read_text(encoding="utf-8", errors="ignore")
    out: dict[str, Any] = {}
    for key, pattern in [
        ("depth", _DEPTH_SECTION_RE),
        ("qubits", _QUBIT_SECTION_RE),
        ("gates", _GATES_SECTION_RE),
    ]:
        m = pattern.search(text)
        out[key] = _parse_section(m.group(1)) if m else {}
    return out


# ── pretty label helpers ──────────────────────────────────────────────────────
def _pretty_depth(label: str) -> str:
    # "Depth 1-10" → "1-10"
    return label.replace("Depth ", "")


def _pretty_gates(label: str) -> str:
    return label.replace("Gates ", "")


def _pretty_qubits(label: str) -> str:
    # "1_qubits" → "1"
    return label.replace("_qubits", "")


_PRETTY = {"depth": _pretty_depth, "gates": _pretty_gates, "qubits": _pretty_qubits}

_XLABEL = {
    "depth": "Circuit Depth",
    "gates": "Number of Gates",
    "qubits": "Number of Qubits",
}

# ── plotting ──────────────────────────────────────────────────────────────────

def _plot_group(
    ax: plt.Axes,
    dimension: str,
    all_model_data: list[dict[str, Any]],
    model_labels: list[str],
    set_title: str,
) -> None:
    """Draw a single grouped bar chart on *ax*."""
    # Collect ordered x-axis categories from the first non-empty model
    categories: list[str] = []
    for mdata in all_model_data:
        cats = list(mdata[dimension].keys())
        if cats:
            categories = cats
            break
    if not categories:
        ax.set_visible(False)
        return

    pretty = [_PRETTY[dimension](c) for c in categories]
    n_cats = len(categories)
    n_models = len(model_labels)
    total_width = 0.72
    bar_w = total_width / n_models
    x = np.arange(n_cats)

    for i, (mdata, label, color) in enumerate(zip(all_model_data, model_labels, MODEL_COLORS)):
        values = [mdata[dimension].get(cat, {}).get("fidelity", 0.0) for cat in categories]
        offset = (i - (n_models - 1) / 2) * bar_w
        bars = ax.bar(
            x + offset,
            values,
            bar_w * 0.92,
            label=label,
            color=color,
            alpha=0.88,
            edgecolor="white",
            linewidth=0.6,
        )


    if set_title:
        ax.set_title(set_title, fontsize=11, pad=6)
    ax.set_xlabel(_XLABEL[dimension], fontsize=10)
    ax.set_ylabel("Average TVD", fontsize=10)
    ax.set_xticks(x)
    ax.set_xticklabels(pretty, fontsize=9)
    ax.set_ylim(0, 1.0)
    ax.yaxis.set_major_locator(ticker.MultipleLocator(0.2))
    ax.yaxis.set_minor_locator(ticker.MultipleLocator(0.1))
    ax.grid(True, axis="y", which="major", linestyle="-", alpha=0.15)


def plot_barchart(
    non_param_data: list[dict[str, Any]],
    param_data: list[dict[str, Any]],
    model_labels: list[str],
    save_path: Path,
) -> None:
    # rows = metric dimension, cols = set, rightmost col = legend
    dimensions  = ["qubits", "depth", "gates"]
    col_labels  = ["Non-parameterized Set", "Parameterized Set"]
    col_data    = [non_param_data,         param_data]

    fig = plt.figure(figsize=(13, 13))
    gs = fig.add_gridspec(
        3, 3,
        width_ratios=[1, 1, 0.18],
        hspace=0.48,
        wspace=0.32,
    )
    axes = [[fig.add_subplot(gs[r, c]) for c in range(2)] for r in range(3)]

    for col_idx, (cdata, clabel) in enumerate(zip(col_data, col_labels)):
        for row_idx, dim in enumerate(dimensions):
            ax = axes[row_idx][col_idx]
            if row_idx == 0:
                ax.set_title(clabel, fontsize=11, pad=18)
            _plot_group(ax, dim, cdata, model_labels, "")

    # Shared legend in the right-hand panel spanning all rows
    legend_ax = fig.add_subplot(gs[:, 2])
    legend_ax.set_axis_off()
    handles, labels = axes[0][0].get_legend_handles_labels()
    legend_ax.legend(
        handles,
        labels,
        loc="center",
        fontsize=10,
        framealpha=0.9,
        title="Model",
        title_fontsize=10,
        borderpad=1.0,
        labelspacing=0.8,
    )

    fig.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot grouped bar charts of Avg Fidelity by qubits/depth/gates for four models × two sets."
    )
    parser.add_argument(
        "--non-param-paths", nargs=4, type=Path,
        default=DEFAULT_NON_PARAM_PATHS,
        metavar=("QWEN", "GPTOSS", "SFT", "SFT_GRPO"),
        help="Four summary .txt paths for the non-parameterized set (Qwen3-8B, GPT-OSS, SFT, SFT+GRPO).",
    )
    parser.add_argument(
        "--param-paths", nargs=4, type=Path,
        default=DEFAULT_PARAM_PATHS,
        metavar=("QWEN", "GPTOSS", "SFT", "SFT_GRPO"),
        help="Four summary .txt paths for the parameterized set.",
    )
    parser.add_argument(
        "--model-labels", nargs=4, type=str,
        default=DEFAULT_MODEL_LABELS,
        help="Display labels for the four models.",
    )
    parser.add_argument("--output-dir", type=Path, default=Path(FIG_DIR))
    parser.add_argument("--output-file", type=str, default="metrics_tvd_barchart.png")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    non_param_data = [parse_summary(Path(p)) for p in args.non_param_paths]
    param_data = [parse_summary(Path(p)) for p in args.param_paths]

    args.output_dir.mkdir(parents=True, exist_ok=True)
    save_path = args.output_dir / args.output_file

    plot_barchart(non_param_data, param_data, args.model_labels, save_path)
    print(f"Metrics bar chart saved to: {save_path}")
