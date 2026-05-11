from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from config.paths import FIG_DIR
from visualization.constants import PLOT_COLORS, apply_plot_style
from visualization.utils.grpo_log_parser import aggregate_logs, discover_log_files

apply_plot_style()

METRICS_TO_COMPARE = (
	("critic/rewards/mean", "Mean Reward"),
	("actor/entropy", "Actor Entropy"),
	("response_length/mean", "Mean Response Length"),
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


def load_metrics_from_csv(csv_path: Path) -> pd.DataFrame:
	df = pd.read_csv(csv_path)
	return ensure_step_column(df)


def load_metrics_from_logs(log_dir: Path) -> pd.DataFrame:
	log_files = discover_log_files(log_dir)
	if not log_files:
		raise FileNotFoundError(f"No .out/.log/.txt files found under: {log_dir}")

	_, metric_rows = aggregate_logs(log_files, log_dir)
	if not metric_rows:
		raise ValueError(f"No metric rows found in logs under: {log_dir}")

	df = pd.DataFrame(metric_rows)
	return ensure_step_column(df)


def load_run_dataframe(csv_path: Path | None, log_dir: Path | None) -> pd.DataFrame:
	if csv_path is not None:
		if not csv_path.exists() or not csv_path.is_file():
			raise FileNotFoundError(f"CSV not found: {csv_path}")
		return load_metrics_from_csv(csv_path)

	if log_dir is not None:
		if not log_dir.exists() or not log_dir.is_dir():
			raise FileNotFoundError(f"Log directory not found: {log_dir}")
		return load_metrics_from_logs(log_dir)

	raise ValueError("Either csv_path or log_dir must be provided for each run.")


def trim_to_max_step(df: pd.DataFrame, max_step: int) -> pd.DataFrame:
	trimmed = df[df["step"] <= max_step].copy()
	return trimmed.sort_values("step")


def write_summary(
	run_a_df: pd.DataFrame,
	run_b_df: pd.DataFrame,
	run_a_label: str,
	run_b_label: str,
	max_step: int,
	output_file: Path,
) -> bool:
	lines = [f"Comparison summary up to step {max_step}", ""]
	has_values = False

	for metric_key, metric_label in METRICS_TO_COMPARE:
		lines.append(f"{metric_label} ({metric_key})")
		for run_label, run_df in ((run_a_label, run_a_df), (run_b_label, run_b_df)):
			if metric_key not in run_df.columns:
				lines.append(f"  {run_label}: missing metric")
				continue

			series = pd.to_numeric(run_df[metric_key], errors="coerce").dropna()
			if series.empty:
				lines.append(f"  {run_label}: no valid values")
				continue

			has_values = True
			lines.append(f"  {run_label}:")
			lines.append(f"    min: {float(series.min()):.6g}")
			lines.append(f"    max: {float(series.max()):.6g}")
			lines.append(f"    final: {float(series.iloc[-1]):.6g}")
			lines.append(f"    mean: {float(series.mean()):.6g}")
		lines.append("")

	if not has_values:
		return False

	output_file.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")
	return True


def plot_comparison(
	run_a_df: pd.DataFrame,
	run_b_df: pd.DataFrame,
	run_a_label: str,
	run_b_label: str,
	max_step: int,
	output_file: Path,
) -> bool:
	fig, axes = plt.subplots(3, 1, figsize=(8, 9), sharex=True)
	has_any_data = False

	for axis, (metric_key, y_label) in zip(axes, METRICS_TO_COMPARE, strict=True):
		run_a_has = False
		run_b_has = False

		if metric_key in run_a_df.columns:
			a_df = run_a_df[["step", metric_key]].copy()
			a_df[metric_key] = pd.to_numeric(a_df[metric_key], errors="coerce")
			a_df = a_df.dropna(subset=[metric_key])
			if not a_df.empty:
				run_a_has = True
				has_any_data = True
				axis.plot(a_df["step"], a_df[metric_key], color=PLOT_COLORS["red"], linewidth=1.8, label=run_a_label)

		if metric_key in run_b_df.columns:
			b_df = run_b_df[["step", metric_key]].copy()
			b_df[metric_key] = pd.to_numeric(b_df[metric_key], errors="coerce")
			b_df = b_df.dropna(subset=[metric_key])
			if not b_df.empty:
				run_b_has = True
				has_any_data = True
				axis.plot(b_df["step"], b_df[metric_key], color=PLOT_COLORS["blue"], linewidth=1.8, label=run_b_label)

		axis.set_ylabel(y_label, fontsize=11)
		axis.grid(True, which="major", linestyle="-", alpha=0.15)
		clean_spines(axis)
		if run_a_has or run_b_has:
			axis.legend(loc="best", framealpha=0.9)

	axes[-1].set_xlabel("Training Step", fontsize=12)
	axes[-1].set_xlim(left=0, right=max_step)

	if has_any_data:
		fig.tight_layout()
		fig.savefig(output_file, dpi=300, bbox_inches="tight")
	plt.close(fig)
	return has_any_data


def parse_args() -> argparse.Namespace:
	default_parsed_logs = Path(__file__).resolve().parent / "parsed_logs"
	default_run_a_csv = default_parsed_logs / "parametric_set_sftgrpo" / "grpo_metrics_aggregated.csv"
	default_run_b_csv = default_parsed_logs / "parametric_set_sftgrpo_05" / "grpo_metrics_aggregated.csv"

	parser = argparse.ArgumentParser(
		description=(
			"Compare two GRPO training runs on selected metrics and limit the plot to the first N steps. "
			"Use CSV inputs or log directories."
		)
	)
	parser.add_argument("--run-a-label", type=str, default="TVD <= 0.01", help="Legend label for run A.")
	parser.add_argument("--run-b-label", type=str, default="TVD <= 0.05", help="Legend label for run B.")
	parser.add_argument(
		"--run-a-csv",
		type=Path,
		default=default_run_a_csv,
		help="Path to aggregated metrics CSV for run A.",
	)
	parser.add_argument(
		"--run-b-csv",
		type=Path,
		default=default_run_b_csv,
		help="Path to aggregated metrics CSV for run B.",
	)
	parser.add_argument("--run-a-log-dir", type=Path, default=None, help="Raw log directory for run A if CSV is not available.")
	parser.add_argument("--run-b-log-dir", type=Path, default=None, help="Raw log directory for run B if CSV is not available.")
	parser.add_argument("--max-step", type=int, default=693, help="Maximum training step to include in plots.")
	parser.add_argument(
		"--output-dir",
		type=Path,
		default=Path(FIG_DIR),
		help="Directory for saved comparison artifacts. Defaults to config.paths.FIG_DIR.",
	)
	parser.add_argument(
		"--plot-file",
		type=str,
		default="grpo_compare_first_693_steps.png",
		help="Filename for the comparison plot figure.",
	)
	parser.add_argument(
		"--summary-file",
		type=str,
		default="grpo_compare_first_693_steps_summary.txt",
		help="Filename for min/max/mean/final summary text.",
	)
	return parser.parse_args()


def main() -> None:
	args = parse_args()
	output_dir = args.output_dir.resolve()
	output_dir.mkdir(parents=True, exist_ok=True)

	run_a_csv = args.run_a_csv.resolve() if args.run_a_csv else None
	run_b_csv = args.run_b_csv.resolve() if args.run_b_csv else None
	run_a_log_dir = args.run_a_log_dir.resolve() if args.run_a_log_dir else None
	run_b_log_dir = args.run_b_log_dir.resolve() if args.run_b_log_dir else None

	run_a_df = load_run_dataframe(run_a_csv, run_a_log_dir)
	run_b_df = load_run_dataframe(run_b_csv, run_b_log_dir)

	run_a_df = trim_to_max_step(run_a_df, args.max_step)
	run_b_df = trim_to_max_step(run_b_df, args.max_step)

	plot_path = output_dir / args.plot_file
	summary_path = output_dir / args.summary_file

	plot_saved = plot_comparison(
		run_a_df=run_a_df,
		run_b_df=run_b_df,
		run_a_label=args.run_a_label,
		run_b_label=args.run_b_label,
		max_step=args.max_step,
		output_file=plot_path,
	)
	summary_saved = write_summary(
		run_a_df=run_a_df,
		run_b_df=run_b_df,
		run_a_label=args.run_a_label,
		run_b_label=args.run_b_label,
		max_step=args.max_step,
		output_file=summary_path,
	)

	print(f"Run A rows up to step {args.max_step}: {len(run_a_df)}")
	print(f"Run B rows up to step {args.max_step}: {len(run_b_df)}")
	if plot_saved:
		print(f"Comparison plot saved to: {plot_path}")
	else:
		print("Comparison plot skipped: no plottable data for selected metrics.")
	if summary_saved:
		print(f"Summary saved to: {summary_path}")
	else:
		print("Summary skipped: no valid selected metric values found.")


if __name__ == "__main__":
	main()