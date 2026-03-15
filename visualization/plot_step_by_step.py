from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import pandas as pd

from config.paths import FIG_DIR
from visualization.constants import apply_plot_style, REFERENCE_LINE_STYLE, STEP_FIDELITY_STYLES

apply_plot_style()


DRAW_ORDER = ("1_qubits", "2_qubits", "3_qubits", "4_qubits", "5_qubits", "overall")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot step-wise quantum state fidelity from parsed SFT CSV.")
    parser.add_argument(
        "--csv-path",
        type=Path,
        required=True,
        help="Path to parsed step-wise fidelity CSV from visualization.utils.sft_log_parser.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(FIG_DIR),
        help="Directory for saved plot. Defaults to config.paths.FIG_DIR.",
    )
    parser.add_argument(
        "--output-file",
        type=str,
        default="stepwise_fidelity_plot.png",
        help="Filename for step-wise fidelity plot.",
    )
    return parser.parse_args()

def load_stepwise_fidelity(csv_path: Path) -> dict[str, dict[int, dict[str, float | int | None]]]:
    df = pd.read_csv(csv_path)
    required_columns = {"step", "group", "fidelity"}
    missing_columns = required_columns.difference(df.columns)
    if missing_columns:
        raise ValueError(f"Missing required columns in CSV: {sorted(missing_columns)}")

    df["step"] = pd.to_numeric(df["step"], errors="coerce")
    df["fidelity"] = pd.to_numeric(df["fidelity"], errors="coerce")
    if "n" in df.columns:
        df["n"] = pd.to_numeric(df["n"], errors="coerce")
    else:
        df["n"] = pd.NA

    df = df.dropna(subset=["step", "group", "fidelity"]).copy()
    df["step"] = df["step"].astype(int)

    results: dict[str, dict[int, dict[str, float | int | None]]] = {}
    for group_name, group_df in df.groupby("group", sort=False):
        group_df = group_df.sort_values("step").drop_duplicates(subset=["step"], keep="last")
        step_map: dict[int, dict[str, float | int | None]] = {}
        for _, row in group_df.iterrows():
            n_value = row["n"]
            step_map[int(row["step"])] = {
                "fidelity": float(row["fidelity"]),
                "n": int(n_value) if pd.notna(n_value) else None,
            }
        results[str(group_name)] = step_map

    return results


def plot_stepwise_fidelity(results: dict[str, dict[int, dict[str, float | int | None]]], save_path: Path) -> bool:
    draw_order = [*DRAW_ORDER, *sorted(key for key in results if key not in DRAW_ORDER)]

    fig, ax = plt.subplots(figsize=(7, 6))
    has_any_data = False

    for key in draw_order:
        if key not in results:
            continue
        step_data = results[key]
        if not step_data:
            continue
        steps     = sorted(step_data.keys())
        fidelities = [step_data[s]["fidelity"] for s in steps]
        sty = STEP_FIDELITY_STYLES.get(key)
        if sty:
            ax.plot(
                steps,
                fidelities,
                label=sty["label"],
                color=sty["color"],
                linewidth=sty["linewidth"],
            )
        else:
            ax.plot(steps, fidelities, label=key, linewidth=1.8)
        has_any_data = True

    if not has_any_data:
        plt.close(fig)
        return False

    for threshold, label in [(0.99, "99%"), (0.95, "95%"), (0.90, "90%")]:
        ax.axhline(threshold, **REFERENCE_LINE_STYLE)
        ax.text(ax.get_xlim()[1] if ax.get_xlim()[1] > 1 else 51,
                threshold, f" {label}", va="center",
            fontsize=9, color=REFERENCE_LINE_STYLE["color"])

    ax.set_xlabel("Number of Gates", fontsize=13)
    ax.set_ylabel("Quantum State Fidelity", fontsize=13)
    ax.set_ylim(0, 1.05)
    ax.set_xlim(left=0)
    ax.yaxis.set_major_locator(ticker.MultipleLocator(0.2))
    ax.yaxis.set_minor_locator(ticker.MultipleLocator(0.1))
    ax.grid(True, which="major", linestyle="-", alpha=0.15)
    ax.legend(loc="lower left", fontsize=10, framealpha=0.9)

    plt.tight_layout()
    fig.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return True


if __name__ == "__main__":
    args = parse_args()
    csv_path = args.csv_path.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    if not csv_path.exists() or not csv_path.is_file():
        raise FileNotFoundError(f"CSV not found: {csv_path}")

    save_path = output_dir / args.output_file
    results = load_stepwise_fidelity(csv_path)
    saved = plot_stepwise_fidelity(results, save_path)

    print(f"CSV source: {csv_path}")
    if saved:
        print(f"Step-wise fidelity plot saved to: {save_path}")
    else:
        print("Step-wise fidelity plot skipped: no plottable fidelity data.")