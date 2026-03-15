from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import pandas as pd

from config.paths import FIG_DIR
from visualization.constants import PLOT_COLORS, apply_plot_style

apply_plot_style()

PLOT_METRICS = (
	("critic/rewards/mean", "Mean Reward", PLOT_COLORS["purple"]),
	("actor/entropy", "Actor Entropy", PLOT_COLORS["teal"]),
	("response_length/mean", "Mean Response Length", PLOT_COLORS["orange"]),
)


def parse_args() -> argparse.Namespace:
	parser = argparse.ArgumentParser(description="Plot GRPO metrics from aggregated CSV.")
	parser.add_argument(
		"--csv-path",
		type=Path,
		required=True,
		help="Path to aggregated metrics CSV from grpo_log_parser.",
	)
	parser.add_argument(
		"--output-dir",
		type=Path,
		default=Path(FIG_DIR),
		help="Directory for saved plots. Defaults to config.paths.FIG_DIR.",
	)
	parser.add_argument(
		"--reward-plot-file",
		type=str,
		default="grpo_reward_curve.png",
		help="Filename for reward-vs-step plot.",
	)
	parser.add_argument(
		"--metrics-plot-file",
		type=str,
		default="grpo_selected_metrics.png",
		help="Filename for selected metrics subplot figure.",
	)
	return parser.parse_args()


def clean_spines(axis: plt.Axes) -> None:
	axis.spines["top"].set_visible(False)
	axis.spines["right"].set_visible(False)


def load_metrics(csv_path: Path) -> pd.DataFrame:
	df = pd.read_csv(csv_path)
	if "step" not in df.columns and "training/global_step" in df.columns:
		df["step"] = pd.to_numeric(df["training/global_step"], errors="coerce")

	df["step"] = pd.to_numeric(df.get("step"), errors="coerce")
	df = df.dropna(subset=["step"]).copy()
	df["step"] = df["step"].astype(int)
	df = df.sort_values("step")
	return df


def plot_reward_curve(df: pd.DataFrame, output_file: Path) -> bool:
	if "critic/rewards/mean" not in df.columns:
		return False

	plot_df = df[["step", "critic/rewards/mean"]].copy()
	plot_df["critic/rewards/mean"] = pd.to_numeric(plot_df["critic/rewards/mean"], errors="coerce")
	plot_df = plot_df.dropna(subset=["critic/rewards/mean"])
	if plot_df.empty:
		return False

	fig, ax = plt.subplots(figsize=(7, 6))
	ax.plot(
		plot_df["step"],
		plot_df["critic/rewards/mean"],
		color=PLOT_COLORS["purple"],
		linewidth=1.8,
		label="critic/rewards/mean",
	)
	ax.set_xlabel("Training Step", fontsize=13)
	ax.set_ylabel("Mean Reward", fontsize=13)
	ax.set_xlim(left=0)
	ax.yaxis.set_major_locator(ticker.MaxNLocator(nbins=6))
	ax.grid(True, which="major", linestyle="-", alpha=0.15)
	clean_spines(ax)
	ax.legend(loc="lower right", fontsize=10, framealpha=0.9)

	fig.tight_layout()
	fig.savefig(output_file, dpi=300, bbox_inches="tight")
	plt.close(fig)
	return True


def plot_selected_metrics(df: pd.DataFrame, output_file: Path) -> bool:
	fig, axes = plt.subplots(3, 1, figsize=(7, 9), sharex=True)
	has_any_data = False

	for axis, (metric_key, y_label, color) in zip(axes, PLOT_METRICS, strict=True):
		if metric_key not in df.columns:
			axis.set_ylabel(y_label, fontsize=11)
			axis.grid(True, which="major", linestyle="-", alpha=0.15)
			clean_spines(axis)
			continue

		plot_df = df[["step", metric_key]].copy()
		plot_df[metric_key] = pd.to_numeric(plot_df[metric_key], errors="coerce")
		plot_df = plot_df.dropna(subset=[metric_key])

		if not plot_df.empty:
			has_any_data = True
			axis.plot(plot_df["step"], plot_df[metric_key], color=color, linewidth=1.8)

		axis.set_ylabel(y_label, fontsize=11)
		axis.grid(True, which="major", linestyle="-", alpha=0.15)
		clean_spines(axis)

	axes[-1].set_xlabel("Training Step", fontsize=13)
	axes[-1].set_xlim(left=0)

	if has_any_data:
		fig.tight_layout()
		fig.savefig(output_file, dpi=300, bbox_inches="tight")
	plt.close(fig)
	return has_any_data


def main() -> None:
	args = parse_args()
	csv_path = args.csv_path.resolve()
	output_dir = args.output_dir.resolve()
	output_dir.mkdir(parents=True, exist_ok=True)

	if not csv_path.exists() or not csv_path.is_file():
		raise FileNotFoundError(f"CSV not found: {csv_path}")

	df = load_metrics(csv_path)
	reward_plot_path = output_dir / args.reward_plot_file
	metrics_plot_path = output_dir / args.metrics_plot_file

	reward_saved = plot_reward_curve(df, reward_plot_path)
	metrics_saved = plot_selected_metrics(df, metrics_plot_path)

	print(f"Loaded rows: {len(df)}")
	print(f"CSV source: {csv_path}")
	if reward_saved:
		print(f"Reward curve saved to: {reward_plot_path}")
	else:
		print("Reward curve skipped: no valid critic/rewards/mean data.")
	if metrics_saved:
		print(f"Selected metrics plot saved to: {metrics_plot_path}")
	else:
		print("Selected metrics plot skipped: no plottable metric data.")


if __name__ == "__main__":
	main()