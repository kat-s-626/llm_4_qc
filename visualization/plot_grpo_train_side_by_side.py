from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from config.paths import FIG_DIR
from visualization.constants import PLOT_COLORS, apply_plot_style

apply_plot_style()

METRICS_TO_COMPARE = (
	("critic/rewards/mean", "Mean Reward", PLOT_COLORS["purple"]),
	("actor/entropy", "Actor Entropy", PLOT_COLORS["teal"]),
	("response_length/mean", "Mean Response Length", PLOT_COLORS["orange"]),
)


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


def get_shared_y_limits(a_values: pd.Series, b_values: pd.Series) -> tuple[float, float] | None:
	combined = pd.concat([a_values, b_values], ignore_index=True).dropna()
	if combined.empty:
		return None

	y_min = float(combined.min())
	y_max = float(combined.max())
	if y_min == y_max:
		padding = 1.0 if y_min == 0 else abs(y_min) * 0.05
		return y_min - padding, y_max + padding

	padding = (y_max - y_min) * 0.05
	return y_min - padding, y_max + padding


def plot_side_by_side(
	run_a_df: pd.DataFrame,
	run_b_df: pd.DataFrame,
	run_a_label: str,
	run_b_label: str,
	output_file: Path,
	max_step: int | None,
) -> bool:
	fig, axes = plt.subplots(len(METRICS_TO_COMPARE), 2, figsize=(12, 9), sharex=False)
	has_any_data = False

	for row_idx, (metric_key, metric_label, metric_color) in enumerate(METRICS_TO_COMPARE):
		left_axis = axes[row_idx, 0]
		right_axis = axes[row_idx, 1]

		run_a_metric_df = prepare_metric(run_a_df, metric_key)
		run_b_metric_df = prepare_metric(run_b_df, metric_key)

		if max_step is not None:
			run_a_metric_df = run_a_metric_df[run_a_metric_df["step"] <= max_step]
			run_b_metric_df = run_b_metric_df[run_b_metric_df["step"] <= max_step]

		run_a_has = not run_a_metric_df.empty
		run_b_has = not run_b_metric_df.empty
		has_any_data = has_any_data or run_a_has or run_b_has

		if run_a_has:
			left_axis.plot(
				run_a_metric_df["step"],
				run_a_metric_df[metric_key],
				color=metric_color,
				linewidth=1.8,
				label=run_a_label,
			)
		if run_b_has:
			right_axis.plot(
				run_b_metric_df["step"],
				run_b_metric_df[metric_key],
				color=metric_color,
				linewidth=1.8,
				label=run_b_label,
			)

		x_max_candidates: list[int] = []
		if run_a_has:
			x_max_candidates.append(int(run_a_metric_df["step"].max()))
		if run_b_has:
			x_max_candidates.append(int(run_b_metric_df["step"].max()))
		if max_step is not None:
			x_max_candidates.append(max_step)

		shared_x_max = max(x_max_candidates) if x_max_candidates else 1
		y_limits = get_shared_y_limits(
			run_a_metric_df[metric_key] if run_a_has else pd.Series(dtype=float),
			run_b_metric_df[metric_key] if run_b_has else pd.Series(dtype=float),
		)

		for axis in (left_axis, right_axis):
			axis.set_xlim(left=0, right=shared_x_max)
			if y_limits is not None:
				axis.set_ylim(*y_limits)
			axis.grid(True, which="major", linestyle="-", alpha=0.15)
			clean_spines(axis)

		left_axis.set_ylabel(metric_label, fontsize=11)
		left_axis.set_title(run_a_label, fontsize=11)
		right_axis.set_title(run_b_label, fontsize=11)

		if run_a_has:
			left_axis.legend(loc="best", framealpha=0.9)
		if run_b_has:
			right_axis.legend(loc="best", framealpha=0.9)

	axes[-1, 0].set_xlabel("Training Step", fontsize=12)
	axes[-1, 1].set_xlabel("Training Step", fontsize=12)

	if has_any_data:
		fig.tight_layout()
		fig.savefig(output_file, dpi=300, bbox_inches="tight")

	plt.close(fig)
	return has_any_data


def parse_args() -> argparse.Namespace:
	default_parsed_logs = Path(__file__).resolve().parent / "parsed_logs"

	parser = argparse.ArgumentParser(
		description=(
			"Create side-by-side GRPO metric plots for two runs with synchronized x/y scales "
			"for easier visual comparison."
		)
	)
	parser.add_argument(
		"--left-csv",
		type=Path,
		default=default_parsed_logs / "non_parametric_set_sftgrpo" / "grpo_metrics_aggregated.csv",
		help="CSV for left plot column (default: non_parametric_set_sftgrpo).",
	)
	parser.add_argument(
		"--right-csv",
		type=Path,
		default=default_parsed_logs / "parametric_set_sftgrpo_05" / "grpo_metrics_aggregated.csv",
		help="CSV for right plot column (default: parametric_set_sftgrpo_05).",
	)
	parser.add_argument("--left-label", type=str, default="non_parametric_set_sftgrpo", help="Legend/title label for left column.")
	parser.add_argument("--right-label", type=str, default="parametric_set_sftgrpo_05", help="Legend/title label for right column.")
	parser.add_argument(
		"--max-step",
		type=int,
		default=None,
		help="Optional maximum step to include. If omitted, plot all available steps.",
	)
	parser.add_argument(
		"--output-dir",
		type=Path,
		default=Path(FIG_DIR),
		help="Directory for saved plot.",
	)
	parser.add_argument(
		"--plot-file",
		type=str,
		default="grpo_side_by_side_non_parametric_set_vs_parametric_set_05.png",
		help="Filename for side-by-side comparison figure.",
	)
	return parser.parse_args()


def main() -> None:
	args = parse_args()
	output_dir = args.output_dir.resolve()
	output_dir.mkdir(parents=True, exist_ok=True)

	left_df = load_metrics(args.left_csv.resolve())
	right_df = load_metrics(args.right_csv.resolve())

	output_file = output_dir / args.plot_file
	plot_saved = plot_side_by_side(
		run_a_df=left_df,
		run_b_df=right_df,
		run_a_label=args.left_label,
		run_b_label=args.right_label,
		output_file=output_file,
		max_step=args.max_step,
	)

	print(f"Left rows: {len(left_df)}")
	print(f"Right rows: {len(right_df)}")
	if args.max_step is not None:
		print(f"Max step filter: {args.max_step}")
	if plot_saved:
		print(f"Side-by-side plot saved to: {output_file}")
	else:
		print("Side-by-side plot skipped: no plottable data for selected metrics.")


if __name__ == "__main__":
	main()