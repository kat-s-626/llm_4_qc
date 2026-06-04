"""Plot step-wise fidelity for two sets (non-parametric and parametric) side by
side in a single figure with a shared legend placed to the right of the panels.

Usage
-----
python -m visualization.plot_step_by_step_combined \
    --non-param-csv  /path/to/non_parametric_stepwise.csv \
    --param-csv      /path/to/parametric_stepwise.csv \
    [--output-dir    visualization/figures] \
    [--output-file   combined_stepwise_fidelity.png]
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

from config.paths import FIG_DIR
from visualization.constants import (
    apply_plot_style,
    REFERENCE_LINE_STYLE,
    STEP_FIDELITY_STYLES,
    STEPWISE_PLOT_CONFIG,
)
from visualization.plot_step_by_step import load_stepwise_fidelity

apply_plot_style()

DRAW_ORDER = ("1_qubits", "2_qubits", "3_qubits", "4_qubits", "5_qubits", "overall")

PANEL_TITLES = {
    "non_param": "Non-Parametric Set",
    "param": "Parametric Set",
}


def _plot_panel(
    ax: plt.Axes,
    results: dict[str, dict[int, dict]],
    title: str,
    show_ylabel: bool,
) -> list[tuple]:
    """Draw one panel onto *ax*.  Returns (handle, label) pairs for the legend."""
    cfg = STEPWISE_PLOT_CONFIG
    draw_order = [*DRAW_ORDER, *sorted(k for k in results if k not in DRAW_ORDER)]

    handles_labels: list[tuple] = []
    for key in draw_order:
        if key not in results:
            continue
        step_data = results[key]
        if not step_data:
            continue
        steps = sorted(step_data.keys())
        fidelities = [step_data[s]["fidelity"] for s in steps]
        sty = STEP_FIDELITY_STYLES.get(key)
        if sty:
            (line,) = ax.plot(
                steps, fidelities,
                label=sty["label"],
                color=sty["color"],
                linewidth=sty["linewidth"],
            )
        else:
            (line,) = ax.plot(
                steps, fidelities,
                label=key,
                linewidth=cfg["linewidth_default"],
            )
        handles_labels.append((line, line.get_label()))

    x_right = ax.get_xlim()[1] if ax.get_xlim()[1] > 1 else 51
    for threshold, ref_label in cfg["reference_lines"]:
        ax.axhline(threshold, **REFERENCE_LINE_STYLE)
        ax.text(
            x_right, threshold, f" {ref_label}",
            va="center",
            fontsize=cfg["reference_label_fontsize"],
            color=REFERENCE_LINE_STYLE["color"],
        )

    ax.set_title(title, fontsize=12, pad=6)
    ax.set_xlabel(cfg["xlabel"], fontsize=cfg["xlabel_fontsize"])
    if show_ylabel:
        ax.set_ylabel(cfg["ylabel"], fontsize=cfg["ylabel_fontsize"])
    else:
        ax.set_ylabel("")
        ax.tick_params(labelleft=False)
    ax.set_ylim(*cfg["ylim"])
    ax.set_xlim(left=0)
    ax.yaxis.set_major_locator(ticker.MultipleLocator(cfg["y_major_tick"]))
    ax.yaxis.set_minor_locator(ticker.MultipleLocator(cfg["y_minor_tick"]))
    ax.grid(True, which="major", linestyle="-", alpha=cfg["grid_alpha"])

    return handles_labels


def plot_combined(
    non_param_results: dict,
    param_results: dict,
    save_path: Path,
) -> bool:
    if not non_param_results and not param_results:
        return False

    fig, (ax_left, ax_right) = plt.subplots(
        1, 2,
        figsize=(13, 5),
        sharey=True,
    )

    hl_left  = _plot_panel(ax_left,  non_param_results,
                           PANEL_TITLES["non_param"], show_ylabel=True)
    hl_right = _plot_panel(ax_right, param_results,
                           PANEL_TITLES["param"],    show_ylabel=False)

    # Deduplicate legend entries (prefer left panel order, fill in extras from right)
    seen: dict[str, plt.Line2D] = {}
    for handle, label in [*hl_left, *hl_right]:
        if label not in seen:
            seen[label] = handle

    fig.legend(
        list(seen.values()),
        list(seen.keys()),
        loc="center left",
        bbox_to_anchor=(1.01, 0.5),
        fontsize=STEPWISE_PLOT_CONFIG["legend_fontsize"],
        framealpha=STEPWISE_PLOT_CONFIG["legend_framealpha"],
        borderaxespad=0,
    )

    plt.tight_layout()
    fig.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return True


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot non-parametric and parametric step-wise fidelity side by side."
    )
    parser.add_argument("--non-param-csv", type=Path, required=True,
                        help="Parsed step-wise fidelity CSV for the non-parametric (Grover) set.")
    parser.add_argument("--param-csv", type=Path, required=True,
                        help="Parsed step-wise fidelity CSV for the parametric (random) set.")
    parser.add_argument("--output-dir", type=Path, default=Path(FIG_DIR),
                        help="Directory for saved plot.")
    parser.add_argument("--output-file", type=str,
                        default="combined_stepwise_fidelity.png",
                        help="Output filename.")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    for p in (args.non_param_csv, args.param_csv):
        if not p.exists():
            raise FileNotFoundError(f"CSV not found: {p}")

    non_param_results = load_stepwise_fidelity(args.non_param_csv.resolve())
    param_results     = load_stepwise_fidelity(args.param_csv.resolve())

    args.output_dir.mkdir(parents=True, exist_ok=True)
    save_path = args.output_dir / args.output_file

    saved = plot_combined(non_param_results, param_results, save_path)
    if saved:
        print(f"Combined step-wise fidelity plot saved to: {save_path}")
    else:
        print("Plot skipped: no plottable fidelity data.")
