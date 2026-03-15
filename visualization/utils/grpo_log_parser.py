from __future__ import annotations

import argparse
import csv
import re
from pathlib import Path
from typing import Any

from visualization.constants import LOG_METRIC_FIELDS

PROMPT_MARKER = "[prompt] user"
GROUND_TRUTH_MARKER = "[ground_truth]"
METRIC_LINE_ANCHOR = "training/global_step:"
STEP_LINE_RE = re.compile(r"^step:\s*(\d+)\b.*")

SELECTED_METRIC_FIELDS = (
	"training/global_step",
	"critic/rewards/mean",
	"actor/entropy",
	"response_length/mean",
)

ANSI_ESCAPE_RE = re.compile(r"\x1B\[[0-?]*[ -/]*[@-~]")
PID_PREFIX_RE = re.compile(
	r"^(?:\[[0-9;]*m)?\((?:TaskRunner|WorkerDict)\s+pid=\d+\)(?:\[[0-9;]*m)?\s*"
)


def parse_args() -> argparse.Namespace:
	parser = argparse.ArgumentParser(description="Parse GRPO training logs and generate extracted responses plus aggregated CSV.")
	parser.add_argument(
		"--log-dir",
		type=Path,
		required=True,
		help="Directory containing one or more GRPO training log files.",
	)
	parser.add_argument(
		"--output-dir",
		type=Path,
		default=None,
		help="Output directory for parsed responses and CSV. Defaults to log dir.",
	)
	parser.add_argument(
		"--responses-file",
		type=str,
		default="parsed_responses.txt",
		help="Filename for extracted response blocks.",
	)
	parser.add_argument(
		"--csv-file",
		type=str,
		default="grpo_metrics_aggregated.csv",
		help="Filename for aggregated metrics CSV.",
	)
	return parser.parse_args()


def clean_log_line(raw_line: str) -> str:
	cleaned = ANSI_ESCAPE_RE.sub("", raw_line.rstrip("\n"))
	cleaned = PID_PREFIX_RE.sub("", cleaned)
	return cleaned


def parse_number(value: str) -> Any:
	stripped = value.strip().strip('"')
	if not stripped:
		return None
	try:
		if re.fullmatch(r"[+-]?\d+", stripped):
			return int(stripped)
		return float(stripped)
	except ValueError:
		return stripped


def parse_metric_line(line: str) -> dict[str, Any]:
	if METRIC_LINE_ANCHOR not in line:
		return {}

	parsed: dict[str, Any] = {}
	chunks = [chunk.strip() for chunk in line.split(" - ") if chunk.strip()]

	for chunk in chunks:
		normalized = chunk.lstrip().lstrip(":")
		if ":" not in normalized:
			continue
		key, value = normalized.split(":", 1)
		key = key.strip()
		if not key:
			continue
		if key in LOG_METRIC_FIELDS:
			parsed[key] = parse_number(value)

	if "training/global_step" in parsed:
		parsed["step"] = int(parsed["training/global_step"])

	return parsed


def parse_log_file(file_path: Path) -> tuple[list[tuple[str, str | None]], list[dict[str, Any]]]:
	responses: list[tuple[str, str | None]] = []
	metric_rows: list[dict[str, Any]] = []

	in_response_block = False
	current_response_lines: list[str] = []
	pending_response_index: int | None = None

	with file_path.open("r", encoding="utf-8", errors="ignore") as handle:
		for raw_line in handle:
			line = clean_log_line(raw_line)
			stripped_line = line.strip()

			if pending_response_index is not None and STEP_LINE_RE.match(stripped_line):
				response_text, _ = responses[pending_response_index]
				responses[pending_response_index] = (response_text, stripped_line)
				pending_response_index = None

			if PROMPT_MARKER in line:
				if in_response_block and current_response_lines:
					responses.append(("\n".join(current_response_lines).strip(), None))
					pending_response_index = len(responses) - 1
				in_response_block = True
				current_response_lines = []
				continue

			if in_response_block and line.startswith(GROUND_TRUTH_MARKER):
				if current_response_lines:
					responses.append(("\n".join(current_response_lines).strip(), None))
					pending_response_index = len(responses) - 1
				in_response_block = False
				current_response_lines = []
				continue

			if in_response_block:
				current_response_lines.append(line)

			parsed_metrics = parse_metric_line(line)
			if parsed_metrics:
				metric_rows.append(parsed_metrics)

	if in_response_block and current_response_lines:
		responses.append(("\n".join(current_response_lines).strip(), None))

	return responses, metric_rows


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


def write_responses(responses: list[tuple[str, str, str | None]], output_file: Path) -> None:
	with output_file.open("w", encoding="utf-8") as handle:
		for index, (source_file, response_text, response_step_line) in enumerate(responses, start=1):
			step_value = "unknown"
			if response_step_line:
				step_match = STEP_LINE_RE.match(response_step_line.strip())
				if step_match:
					step_value = step_match.group(1)

			handle.write(f"===== step: {step_value} | response {index} | file: {source_file} =====\n")
			handle.write(response_text)
			if response_step_line:
				handle.write("\n")
				handle.write(response_step_line)
			handle.write("\n\n")


def write_metrics_csv(rows: list[dict[str, Any]], output_file: Path) -> None:
	if not rows:
		with output_file.open("w", encoding="utf-8", newline="") as handle:
			writer = csv.writer(handle)
			writer.writerow(["step", "training/global_step", *SELECTED_METRIC_FIELDS[1:], "source_file"])
		return

	all_columns = ["step", *LOG_METRIC_FIELDS, "source_file"]
	unique_columns: list[str] = []
	seen_columns: set[str] = set()
	for column in all_columns:
		if column not in seen_columns:
			seen_columns.add(column)
			unique_columns.append(column)

	with output_file.open("w", encoding="utf-8", newline="") as handle:
		writer = csv.DictWriter(handle, fieldnames=unique_columns)
		writer.writeheader()
		for row in rows:
			writer.writerow({column: row.get(column) for column in unique_columns})


def aggregate_logs(log_files: list[Path], log_dir: Path) -> tuple[list[tuple[str, str, str | None]], list[dict[str, Any]]]:
	responses_with_source: list[tuple[str, str, str | None]] = []
	metrics_by_step: dict[int, dict[str, Any]] = {}

	for file_path in log_files:
		relative_source = str(file_path.relative_to(log_dir))
		responses, metric_rows = parse_log_file(file_path)

		for response_text, response_step_line in responses:
			if response_text:
				responses_with_source.append((relative_source, response_text, response_step_line))

		for metric_row in metric_rows:
			step_value = metric_row.get("training/global_step")
			if not isinstance(step_value, int):
				continue

			enriched_row = dict(metric_row)
			enriched_row["step"] = step_value
			enriched_row["source_file"] = relative_source
			metrics_by_step[step_value] = enriched_row

	ordered_rows = [metrics_by_step[step] for step in sorted(metrics_by_step)]
	return responses_with_source, ordered_rows


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

	responses, metrics_rows = aggregate_logs(log_files, log_dir)

	responses_file = output_dir / args.responses_file
	csv_file = output_dir / args.csv_file

	write_responses(responses, responses_file)
	write_metrics_csv(metrics_rows, csv_file)

	print(f"Parsed log files: {len(log_files)}")
	print(f"Extracted response blocks: {len(responses)}")
	print(f"Aggregated training steps: {len(metrics_rows)}")
	print(f"Responses written to: {responses_file}")
	print(f"Metrics CSV written to: {csv_file}")


if __name__ == "__main__":
	main()