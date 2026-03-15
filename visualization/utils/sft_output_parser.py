from __future__ import annotations

import argparse
import csv
import re
from pathlib import Path
from typing import Any


OVERALL_SECTION_RE = re.compile(
	r"PER-STEP FIDELITY:\s*\n(.*?)\n\s*PER-STEP FIDELITY BY NUMBER OF QUBITS",
	re.DOTALL,
)
QUBIT_SECTION_RE = re.compile(
	r"PER-STEP FIDELITY BY NUMBER OF QUBITS:(.*?)(?:\n\s*QUANTUM STATE PARSE STATISTICS|\Z)",
	re.DOTALL,
)
QUBIT_BLOCK_RE = re.compile(r"(\d+_qubits):\s*\n(.*?)(?=\n\s+\d+_qubits:|\Z)", re.DOTALL)
STEP_ENTRY_RE = re.compile(r"step_(\d+):\s+([+-]?\d*\.?\d+(?:[eE][+-]?\d+)?)\s+\(n=(\d+)\)")


def parse_args() -> argparse.Namespace:
	parser = argparse.ArgumentParser(description="Parse SFT evaluation logs and generate step-wise fidelity CSV.")
	parser.add_argument(
		"--log-dir",
		type=Path,
		required=True,
		help="Directory containing one or more SFT evaluation log files.",
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
		default="sft_stepwise_fidelity.csv",
		help="Filename for parsed step-wise fidelity CSV.",
	)
	return parser.parse_args()


def _parse_step_block(block_text: str, group_name: str, source_file: str) -> list[dict[str, Any]]:
	rows: list[dict[str, Any]] = []
	for match in STEP_ENTRY_RE.finditer(block_text):
		rows.append(
			{
				"step": int(match.group(1)),
				"group": group_name,
				"fidelity": float(match.group(2)),
				"n": int(match.group(3)),
				"source_file": source_file,
			}
		)
	return rows


def parse_log_file(file_path: Path, source_file: str) -> list[dict[str, Any]]:
	with file_path.open("r", encoding="utf-8", errors="ignore") as handle:
		log_text = handle.read()

	rows: list[dict[str, Any]] = []

	overall_section = OVERALL_SECTION_RE.search(log_text)
	if overall_section:
		rows.extend(_parse_step_block(overall_section.group(1), "overall", source_file))

	qubit_section = QUBIT_SECTION_RE.search(log_text)
	if qubit_section:
		for qubit_label, block in QUBIT_BLOCK_RE.findall(qubit_section.group(1)):
			rows.extend(_parse_step_block(block, qubit_label, source_file))

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


def aggregate_logs(log_files: list[Path], log_dir: Path) -> list[dict[str, Any]]:
	rows_by_group_step: dict[tuple[str, int], dict[str, Any]] = {}

	for file_path in log_files:
		relative_source = str(file_path.relative_to(log_dir))
		rows = parse_log_file(file_path, relative_source)
		for row in rows:
			rows_by_group_step[(row["group"], row["step"])] = row

	ordered_rows = [
		rows_by_group_step[key]
		for key in sorted(rows_by_group_step, key=lambda item: (item[0], item[1]))
	]
	return ordered_rows


def write_fidelity_csv(rows: list[dict[str, Any]], output_file: Path) -> None:
	fieldnames = ["step", "group", "fidelity", "n", "source_file"]
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
	write_fidelity_csv(rows, csv_file)

	print(f"Parsed log files: {len(log_files)}")
	print(f"Parsed step-group rows: {len(rows)}")
	print(f"Metrics CSV written to: {csv_file}")


if __name__ == "__main__":
	main()