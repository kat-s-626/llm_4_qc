from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from config.paths import FIG_DIR
from visualization.constants import LOSS_CURVE_STYLES, PLOT_COLORS, apply_plot_style

apply_plot_style()

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot training and validation loss from aggregated CSV.")
    parser.add_argument(
        "--csv-path",
        type=Path,
        required=True,
        help="Path to aggregated train/validation loss CSV from visualization.utils.train_valid_loss_log_parser.",
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
        default="train_valid_loss_plot.png",
        help="Filename for train-vs-validation loss plot.",
    )
    return parser.parse_args()


def load_metrics(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    if "step" not in df.columns:
        raise ValueError("CSV must include a 'step' column.")

    df["step"] = pd.to_numeric(df["step"], errors="coerce")
    if "train/loss" in df.columns:
        df["train/loss"] = pd.to_numeric(df["train/loss"], errors="coerce")
    if "val/loss" in df.columns:
        df["val/loss"] = pd.to_numeric(df["val/loss"], errors="coerce")

    df = df.dropna(subset=["step"]).copy()
    df["step"] = df["step"].astype(int)
    df = df.sort_values("step").drop_duplicates(subset=["step"], keep="last")
    return df


def plot_loss(df: pd.DataFrame, output_file: Path) -> bool:
    fig, ax = plt.subplots(figsize=(7, 6))

    train_style = LOSS_CURVE_STYLES["train"]
    validation_style = LOSS_CURVE_STYLES["validation"]
    has_any_data = False

    if "train/loss" in df.columns:
        train_df = df[["step", "train/loss"]].dropna(subset=["train/loss"])
        if not train_df.empty:
            ax.plot(
                train_df["step"],
                train_df["train/loss"],
                color=train_style["color"],
                linewidth=train_style["linewidth"],
                linestyle=train_style["linestyle"],
                label=train_style["label"],
                alpha=train_style["alpha"],
            )
            has_any_data = True

    if "val/loss" in df.columns:
        val_df = df[["step", "val/loss"]].dropna(subset=["val/loss"])
        if not val_df.empty:
            ax.plot(
                val_df["step"],
                val_df["val/loss"],
                color=validation_style["color"],
                linewidth=validation_style["linewidth"],
                linestyle=validation_style["linestyle"],
                label=validation_style["label"],
                alpha=validation_style["alpha"],
            )
            has_any_data = True

    if not has_any_data:
        plt.close(fig)
        return False

    ax.set_xlabel("Training Step", fontsize=13, labelpad=8)
    ax.set_ylabel("Loss", fontsize=13, labelpad=8)
    ax.tick_params(axis="both", labelsize=11)

    ax.set_ylim(0.0, 1.0)
    ax.set_xlim(left=0)

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_visible(False)
    ax.spines["bottom"].set_color(PLOT_COLORS["axis"])

    ax.legend(frameon=False, fontsize=11, loc="upper right")

    ax.grid(True, linestyle=":", linewidth=0.6, alpha=0.6)
    plt.tight_layout()

    fig.savefig(output_file, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return True


def main() -> None:
    args = parse_args()
    csv_path = args.csv_path.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    if not csv_path.exists() or not csv_path.is_file():
        raise FileNotFoundError(f"CSV not found: {csv_path}")

    save_path = output_dir / args.output_file
    df = load_metrics(csv_path)
    saved = plot_loss(df, save_path)

    print(f"Loaded rows: {len(df)}")
    print(f"CSV source: {csv_path}")
    if saved:
        print(f"Train/validation loss plot saved to: {save_path}")
    else:
        print("Train/validation loss plot skipped: no plottable loss data.")


if __name__ == "__main__":
    main()
