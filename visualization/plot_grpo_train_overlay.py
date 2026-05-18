"""Plot GRPO training metrics for two runs overlaid on the same axes.

Produces a single figure with 3 subplots (one per metric), each showing both
the non-parameterized and parameterized sets together for easy comparison.

Usage
-----
python -m visualization.plot_grpo_train_overlay \
    [--left-csv  path/to/non_param.csv] \
    [--right-csv path/to/param.csv] \
    [--left-label  "Non-Parameterized Set"] \
    [--right-label "Parameterized Set"] \
    [--max-step 1000] \
    [--output-dir visualization/figures] \
    [--plot-file grpo_overlay.png]
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from config.paths import FIG_DIR
from visualization.constants import PLOT_COLORS, apply_plot_style

apply_plot_style()

METRICS_TO_PLOT = (
	("critic/rewards/mean", "Mean Reward"),
	("actor/entropy", "Actor Entropy"),
	("response_length/mean", "Mean Response Length"),
)

# Consistent colours for the two runs across all subplots
LEFT_COLOR  = PLOT_COLORS["purple"]
RIGHT_COLOR = PLOT_COLORS["teal"]


def clean_spines(axis: plt.Axes) -> None:
	axis.spines["top"].set_visible(False)
	axis.spines["right"].set_visible(False)


def ensure_step_column(df: pd.DataFrame) -> pd.DataFrame:
	if "step" not in df.columns and "training/global_step" in df.columns:
		df["step"] = pd.to_numeric(df["training/global_step"], errors="coerce")
	df["step"] = pd.to_numeric(df.get("step"), errors="coerce")
	df = df.dropna(subset=["step"]).copy()
	df["step"] = df["step"].astype(int)
	df = df.sort_values("step")
	return df


def load_metrics(csv_path: Path) -> pd.DataFrame:
	if not csv_path.exists() or not csv_path.is_file():
		raise FileNotFoundError(f"CSV not found: {csv_path}")
	df = pd.read_csv(csv_path)
	return ensure_step_column(df)


def prepare_metric(df: pd.DataFrame, metric_key: str) -> pd.DataFrame:
	if metric_key not in df.columns:
		return pd.DataFrame(columns=["step", metric_key])
	plot_df = df[["step", metric_key]].copy()
	plot_df[metric_key] = pd.to_numeric(plot_df[metric_key], errors="coerce")
	plot_df = plot_df.dropna(subset=[metric_key])
	return plot_df


def plot_overlay(
	left_df: pd.DataFrame,
	right_df: pd.DataFrame,
	left_label: str,
	right_label: str,
	output_file: Path,
	max_step: int | None,
) -> bool:
	fig, axes = plt.subplots(1, len(METRICS_TO_PLOT), figsize=(16, 4))
	has_any_data = False

	for ax, (metric_key, metric_label) in zip(axes, METRICS_TO_PLOT):
		left_m  = prepare_metric(left_df,  metric_key)
		right_m = prepare_metric(right_df, metric_key)

		if max_step is not None:
			left_m  = left_m[left_m["step"]   <= max_step]
			right_m = right_m[right_m["step"] <= max_step]

		left_has  = not left_m.empty
		right_has = not right_m.empty
		has_any_data = has_any_data or left_has or right_has

		if left_has:
			ax.plot(
				left_m["step"], left_m[metric_key],
				color=LEFT_COLOR, linewidth=1.8,
				label=left_label,
			)
		if right_has:
			ax.plot(
				right_m["step"], right_m[metric_key],
				color=RIGHT_COLOR, linewidth=1.8,
				linestyle="--",
				label=right_label,
			)

		ax.set_ylabel(metric_label, fontsize=11)
		ax.set_xlabel("Training Step", fontsize=11)
		ax.grid(True, which="major", linestyle="-", alpha=0.15)
		clean_spines(ax)

	if has_any_data:
		# Single shared legend centred below all subplots
		handles, labels = axes[0].get_legend_handles_labels()
		fig.legend(
			handles, labels,
			loc="lower center",
			ncol=2,
			fontsize=11,
			framealpha=0.9,
			bbox_to_anchor=(0.5, -0.08),
		)
		fig.tight_layout()
		fig.savefig(output_file, dpi=300, bbox_inches="tight")

	plt.close(fig)
	return has_any_data


def parse_args() -> argparse.Namespace:
	default_parsed_logs = Path(__file__).resolve().parent / "parsed_logs"

	parser = argparse.ArgumentParser(
		description=(
			"Plot GRPO training metrics for two runs overlaid on the same axes "
			"(3 subplots, one per metric)."
		)
	)
	parser.add_argument(
		"--left-csv", type=Path,
		default=default_parsed_logs / "groverset_sftgrpo" / "grpo_metrics_aggregated.csv",
		help="CSV for the non-parameterized set.",
	)
	parser.add_argument(
		"--right-csv", type=Path,
		default=default_parsed_logs / "randomset_sftgrpo_05" / "grpo_metrics_aggregated.csv",
		help="CSV for the parameterized set.",
	)
	parser.add_argument("--left-label",  type=str, default="Non-Parameterized Set")
	parser.add_argument("--right-label", type=str, default="Parameterized Set")
	parser.add_argument("--max-step", type=int, default=None,
		help="Optional maximum training step to include.")
	parser.add_argument("--output-dir",  type=Path, default=Path(FIG_DIR))
	parser.add_argument("--plot-file",   type=str,
		default="grpo_overlay_non_parameterized_vs_parameterized_horizontal.png")
	return parser.parse_args()


def main() -> None:
	args = parse_args()
	output_dir = args.output_dir.resolve()
	output_dir.mkdir(parents=True, exist_ok=True)

	left_df  = load_metrics(args.left_csv.resolve())
	right_df = load_metrics(args.right_csv.resolve())

	output_file = output_dir / args.plot_file
	saved = plot_overlay(
		left_df=left_df,
		right_df=right_df,
		left_label=args.left_label,
		right_label=args.right_label,
		output_file=output_file,
		max_step=args.max_step,
	)

	print(f"Left rows:  {len(left_df)}")
	print(f"Right rows: {len(right_df)}")
	if args.max_step is not None:
		print(f"Max step filter: {args.max_step}")
	if saved:
		print(f"Overlay plot saved to: {output_file}")
	else:
		print("Overlay plot skipped: no plottable data for selected metrics.")


if __name__ == "__main__":
	main()
