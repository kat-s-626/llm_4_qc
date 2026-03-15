from __future__ import annotations

import argparse
import csv
import re
from pathlib import Path

ANSI_ESCAPE_RE = re.compile(r"\x1B\[[0-?]*[ -/]*[@-~]")
PID_PREFIX_RE = re.compile(
	r"^(?:\[[0-9;]*m)?\((?:TaskRunner|WorkerDict)\s+pid=\d+(?:,\s*ip=[^)]+)?\)(?:\[[0-9;]*m)?\s*"
)
STEP_RE = re.compile(r"\bstep[:=\s]+(\d+)\b", re.IGNORECASE)
TRAIN_LOSS_RE = re.compile(r"\btrain/loss:\s*([+-]?\d*\.?\d+(?:[eE][+-]?\d+)?)")
VAL_LOSS_RE = re.compile(r"\bval/loss:\s*([+-]?\d*\.?\d+(?:[eE][+-]?\d+)?)")


def parse_args() -> argparse.Namespace:
	parser = argparse.ArgumentParser(description="Parse training logs and generate aggregated train/validation loss CSV.")
	parser.add_argument(
		"--log-dir",
		type=Path,
		required=True,
		help="Directory containing one or more training log files.",
	)
	parser.add_argument(
		"--output-dir",
		type=Path,
		default=None,
		help="Output directory for parsed CSV. Defaults to log dir.",
	)
	parser.add_argument(
		"--csv-file",
		type=str,
		default="train_valid_loss_aggregated.csv",
		help="Filename for aggregated train/validation loss CSV.",
	)
	return parser.parse_args()


def clean_log_line(raw_line: str) -> str:
	cleaned = ANSI_ESCAPE_RE.sub("", raw_line.rstrip("\n"))
	cleaned = PID_PREFIX_RE.sub("", cleaned)
	return cleaned


def parse_log_file(file_path: Path, source_file: str) -> list[dict[str, float | int | str]]:
	rows: list[dict[str, float | int | str]] = []
	current_step: int | None = None

	with file_path.open("r", encoding="utf-8", errors="ignore") as handle:
		for raw_line in handle:
			line = clean_log_line(raw_line)

			step_match = STEP_RE.search(line)
			if step_match:
				current_step = int(step_match.group(1))

			if current_step is None:
				continue

			row: dict[str, float | int | str] = {"step": current_step, "source_file": source_file}
			has_metric = False

			train_match = TRAIN_LOSS_RE.search(line)
			if train_match:
				row["train/loss"] = float(train_match.group(1))
				has_metric = True

			val_match = VAL_LOSS_RE.search(line)
			if val_match:
				row["val/loss"] = float(val_match.group(1))
				has_metric = True

			if has_metric:
				rows.append(row)

	return rows


def discover_log_files(log_dir: Path) -> list[Path]:
	candidates = [
		path
		for path in log_dir.rglob("*")
		if path.is_file() and path.suffix.lower() in {".out", ".log", ".txt"}
	]

	def sort_key(path: Path) -> tuple[int, int, str]:
		try:
			log_number = int(path.stem)
			return (0, log_number, str(path))
		except ValueError:
			match = re.search(r"\d+", path.stem)
			if match:
				return (0, int(match.group(0)), str(path))
			return (1, 0, str(path))

	return sorted(candidates, key=sort_key)


def aggregate_logs(log_files: list[Path], log_dir: Path) -> list[dict[str, float | int | str | None]]:
	metrics_by_step: dict[int, dict[str, float | int | str | None]] = {}

	for file_path in log_files:
		relative_source = str(file_path.relative_to(log_dir))
		for row in parse_log_file(file_path, relative_source):
			step_value = int(row["step"])
			aggregated_row = metrics_by_step.setdefault(
				step_value,
				{"step": step_value, "train/loss": None, "val/loss": None, "source_file": relative_source},
			)

			if "train/loss" in row:
				aggregated_row["train/loss"] = row["train/loss"]
				aggregated_row["source_file"] = relative_source

			if "val/loss" in row:
				aggregated_row["val/loss"] = row["val/loss"]
				aggregated_row["source_file"] = relative_source

	return [metrics_by_step[step] for step in sorted(metrics_by_step)]


def write_metrics_csv(rows: list[dict[str, float | int | str | None]], output_file: Path) -> None:
	fieldnames = ["step", "train/loss", "val/loss", "source_file"]
	with output_file.open("w", encoding="utf-8", newline="") as handle:
		writer = csv.DictWriter(handle, fieldnames=fieldnames)
		writer.writeheader()
		for row in rows:
			writer.writerow({column: row.get(column) for column in fieldnames})


def main() -> None:
	args = parse_args()
	log_dir = args.log_dir.resolve()
	output_dir = (args.output_dir or args.log_dir).resolve()
	output_dir.mkdir(parents=True, exist_ok=True)

	if not log_dir.exists() or not log_dir.is_dir():
		raise FileNotFoundError(f"Log directory not found: {log_dir}")

	log_files = discover_log_files(log_dir)
	if not log_files:
		raise FileNotFoundError(f"No .out/.log/.txt files found under: {log_dir}")

	rows = aggregate_logs(log_files, log_dir)

	csv_file = output_dir / args.csv_file
	write_metrics_csv(rows, csv_file)

	print(f"Parsed log files: {len(log_files)}")
	print(f"Aggregated training steps: {len(rows)}")
	print(f"Metrics CSV written to: {csv_file}")


if __name__ == "__main__":
	main()